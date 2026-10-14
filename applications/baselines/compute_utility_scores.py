"""Compute EL2N / GradAlign / checkpoint-averaged AUM for the naive 20k pool.

- EL2N: base Qwen 1.5B t0 forward -> ||p - e_y||_2 on the label-token logits.
- GradAlign: per-candidate gradient (w.r.t. loss on gold label token) dotted
  with a stored val-gradient vector, computed via forward + backward.
- AUM: mean gold-vs-best-other margin over the base model plus 2--4 compatible
  fine-tuning checkpoints. Hugging Face model directories/ids and the
  ``model_state_dict`` .pt files written by curriculum_ft.py are supported.

Outputs a JSONL with one row per candidate:
  {"idx": i, "el2n": ..., "gradalign": ..., "aum_proxy": ...}

The 20k row order is preserved; select_rejection.py joins by idx.
Rows rejected by the trainer's exact generated-row converter are retained as
explicit invalid records; all valid rows must have finite scores. Missing
validation data, GradAlign parameters, or distinct AUM checkpoints fail fast.
"""

from __future__ import annotations

# Resolve this checkout independently of the caller's working directory.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = next(p for p in _RepoPath(__file__).resolve().parents
                  if (p / "EAP-IG").is_dir() and (p / "applications").is_dir())
_SUPPLEMENTAL_ROOT = _REPO_ROOT / "results" / "supplemental"
for _import_root in (_REPO_ROOT, _REPO_ROOT / "EAP-IG",
                     _REPO_ROOT / "EAP-IG" / "src", _REPO_ROOT / "data_gens"):
    if str(_import_root) not in _repo_sys.path:
        _repo_sys.path.insert(0, str(_import_root))


import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from applications.experiment_utils.io import NAIVE_RAW_PATH, read_jsonl, write_json, write_jsonl

# The base 1.5B (t0) that produced the pool; also serves as scoring model.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _row_to_mcqa(row: Mapping, *, row_name: str) -> Tuple[str, List[str], List[str], str]:
    """Normalize with the trainer's exact generated-row converter.

    Validation rows are already trainer examples. Raw generated candidates are
    passed through curriculum_ft.generated_row_to_example so the validity mask,
    choice cleaning, full-text answer resolution, and prompt text exactly match
    what downstream fine-tuning consumes.
    """
    if row.get("question_text") and row.get("gold_label"):
        normalized = row
    else:
        for import_root in (str(_REPO_ROOT), str(_REPO_ROOT / 'applications')):
            if import_root not in sys.path:
                sys.path.insert(0, import_root)
        from applications.curriculum_ft import generated_row_to_example

        normalized = generated_row_to_example(row, dataset_name="sciq")
        if normalized is None:
            raise ValueError(f"{row_name}: rejected by curriculum_ft.generated_row_to_example")

    question = normalized.get("question_text") or ""
    choices = normalized.get("choices")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{row_name}: missing question/question_text")
    if not isinstance(choices, (list, tuple)) or len(choices) < 2:
        raise ValueError(f"{row_name}: expected at least two choices")
    choices = [str(choice) for choice in choices]

    label_space = normalized.get("label_space")
    if label_space is None:
        label_space = [chr(ord("A") + i) for i in range(len(choices))]
    label_space = [str(label).upper() for label in label_space]
    if len(label_space) != len(choices) or len(set(label_space)) != len(label_space):
        raise ValueError(
            f"{row_name}: label_space must have one unique label per choice "
            f"(labels={label_space}, n_choices={len(choices)})"
        )

    gold = normalized.get("gold_label")
    gold = str(gold).upper() if gold is not None else ""
    if gold not in label_space:
        raise ValueError(f"{row_name}: gold label {gold!r} is not in {label_space}")
    return question, choices, label_space, gold


def build_prompt(
    tokenizer, question: str, choices: Sequence[str], label_space: Sequence[str]
) -> Tuple[str, Dict[str, int]]:
    """Reuse the data_gens.mcqa formatter for consistency with downstream eval.

    Returns (prompt_str, label_token_ids_dict) where label_token_ids_dict maps
    each label letter -> the single token id that follows the answer prefix
    (this is what get_option_logits / the trainer's eval use).
    """
    sys.path.insert(0, str(_REPO_ROOT))
    from data_gens.mcqa import format_mcqa_prompt  # type: ignore
    prompt, _answer_prefix, label_token_ids = format_mcqa_prompt(
        question_text=question, choices=choices, label_space=label_space, tokenizer=tokenizer,
    )
    missing = [label for label in label_space if label not in label_token_ids]
    if missing:
        raise ValueError(f"formatter did not return token ids for labels={missing}")
    return str(prompt), {str(k): int(v) for k, v in label_token_ids.items()}


def logits_at_prompt_end(model, tokenizer, prompt: str, device) -> torch.Tensor:
    """Return the model's next-token logits (1D over vocab) at end of prompt."""
    inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]  # (V,)
    return logits


def loss_at_gold(model, tokenizer, prompt: str, gold_id: int, device) -> torch.Tensor:
    inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**inp)
    logits = out.logits[0, -1, :]
    logp = F.log_softmax(logits, dim=-1)
    return -logp[gold_id]


def select_grad_params(model, scope: str) -> Set[str]:
    """Return the set of parameter names whose gradients we track for GradAlign.

    Collecting full-model gradients per candidate for 20k candidates is
    intractable.  The repository's original GradAlign implementation
    (``data_gens/da_score_collect.py:get_last_layer_params``) defines its
    "last layer" as the output linear weight.

    scope='head'     → original last-linear definition: lm_head/embed_out weight;
                       tied-head models use the shared embed_tokens weight (default)
    scope='last'     → last transformer block only (explicit faster proxy)
    scope='full'     → every parameter (exact definition; slow, ~intractable for 20k)
    """
    if scope == "full":
        return {n for n, p in model.named_parameters() if p.requires_grad}
    if scope == "head":
        # Match da_score_collect.get_last_layer_params exactly: select one
        # output-layer weight, never the input embedding merely because it is
        # present.  With tied weights, named_parameters() keeps only the first
        # canonical alias (usually model.embed_tokens.weight), so identity with
        # lm_head.weight intentionally resolves to that shared parameter name.
        target = None
        for attr in ("embed_out", "lm_head"):
            module = getattr(model, attr, None)
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.nn.Parameter):
                target = weight
                break
        if target is None:
            parameters = list(model.parameters())
            if parameters:
                target = parameters[-1]
        return {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter is target
        }
    # scope == "last": highest transformer-block index. Match common HF block
    # container names so unrelated numeric path components cannot win.
    import re

    block_names: List[Tuple[str, int]] = []
    pattern = re.compile(r"(?:^|\.)(layers|h|blocks|block)\.(\d+)(?:\.|$)")
    for name, _ in model.named_parameters():
        match = pattern.search(name)
        if match:
            block_names.append((match.group(1), int(match.group(2))))
    names: Set[str] = set()
    if block_names:
        family, max_idx = max(block_names, key=lambda item: item[1])
        marker = f".{family}.{max_idx}."
        names = {
            n for n, p in model.named_parameters()
            if p.requires_grad and marker in f".{n}."
        }
    if not names:  # fallback: tied head
        names = {
            n for n, p in model.named_parameters()
            if p.requires_grad
            and (
                "lm_head" in n.lower()
                or "embed_tokens" in n.lower()
                or "embed_out" in n.lower()
            )
        }
    return names


def compute_val_grad(model, tokenizer, val_rows, device, max_val: int = 128, grad_names=None):
    """Mean gradient on a small held-out val subset — the GradAlign reference.

    Reference grads are kept ON THE MODEL DEVICE (GPU) so the per-candidate dot
    product needs no CPU transfer.
    """
    model.zero_grad()
    total_loss = 0.0
    n = 0
    if not val_rows:
        raise ValueError("validation source is empty")
    for idx, row in enumerate(val_rows[:max_val]):
        q, choices, label_space, gold_letter = _row_to_mcqa(row, row_name=f"val[{idx}]")
        prompt, lab_ids = build_prompt(tokenizer, q, choices, label_space)
        gold_id = int(lab_ids[gold_letter])
        loss = loss_at_gold(model, tokenizer, prompt, gold_id, device)
        loss.backward()
        total_loss += float(loss.detach().item())
        n += 1
    grads = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None and (grad_names is None or p_name in grad_names):
            grads[p_name] = (p.grad.detach().float() / max(1, n)).clone()  # stays on GPU
    model.zero_grad()
    if n == 0:
        raise RuntimeError("validation source produced zero GradAlign examples")
    if not grads:
        raise RuntimeError(
            "GradAlign reference produced zero parameter gradients; "
            "check --grad-scope and tied-embedding parameter names"
        )
    return grads, {"n_val": n, "mean_loss": total_loss / max(1, n)}


def dot_grads(g1, g2) -> float:
    s = 0.0
    for name, v1 in g1.items():
        if name in g2:
            s += float(torch.sum(v1.to(g2[name].dtype) * g2[name]).item())
    return s


def compute_scores_for_row(
    model, tokenizer, row, device, *, row_name: str, ref_grads=None,
):
    """Return (el2n, gradalign_dot, margin) for one candidate.

    gradalign_dot: if ref_grads is given, we compute the per-sample gradient
    (restricted to grad_names) and immediately dot it against ref_grads, so we
    never keep 20k full-model gradients around. Returns None if ref_grads None.
    """
    q, choices, label_space, gold_letter = _row_to_mcqa(row, row_name=row_name)
    prompt, lab_ids = build_prompt(tokenizer, q, choices, label_space)
    gold_idx = label_space.index(gold_letter)
    option_ids = [int(lab_ids[l]) for l in label_space]

    # ---- EL2N + margin (forward only) ----
    logits = logits_at_prompt_end(model, tokenizer, prompt, device)
    option_logits = torch.stack([logits[i] for i in option_ids])
    probs = F.softmax(option_logits, dim=-1)
    onehot = torch.zeros_like(probs)
    onehot[gold_idx] = 1.0
    el2n = float(torch.norm(probs - onehot, p=2).item())
    others = torch.cat([option_logits[:gold_idx], option_logits[gold_idx + 1:]])
    margin = float(option_logits[gold_idx].item() - torch.max(others).item())

    # ---- per-sample gradient dotted against ref (GradAlign), all on GPU ----
    ga = None
    if ref_grads is not None:
        model.zero_grad()
        loss = loss_at_gold(model, tokenizer, prompt, int(lab_ids[gold_letter]), device)
        loss.backward()
        s = 0.0
        for p_name, p in model.named_parameters():
            if p.grad is None or p_name not in ref_grads:
                continue
            s += float(torch.sum(p.grad.detach().float() * ref_grads[p_name]).item())
        ga = s
        model.zero_grad()
    return el2n, ga, margin


def _new_base_model(model_id: str, torch_dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True
    )
    return model.to(device).eval()


def load_aum_checkpoint(base_model_id: str, checkpoint: str, torch_dtype, device: str):
    """Load a HF checkpoint or curriculum_ft.py ``model_state_dict`` .pt file."""
    path = Path(checkpoint).expanduser()
    if not path.exists() and (path.is_absolute() or path.suffix.lower() in {".pt", ".pth", ".bin"}):
        raise FileNotFoundError(f"AUM checkpoint path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in {".pt", ".pth", ".bin"}:
            raise ValueError(
                f"unsupported checkpoint file {path}; expected .pt/.pth/.bin "
                "or a Hugging Face directory/model id"
            )
        model = _new_base_model(base_model_id, torch_dtype, "cpu")
        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: checkpoint payload is not a mapping")
        state_dict = payload.get("model_state_dict", payload)
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"{path}: missing model_state_dict")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{path} is not architecture-compatible with base model "
                f"{base_model_id}: {exc}"
            ) from exc
        return model.to(device).eval()
    return _new_base_model(checkpoint, torch_dtype, device)


def resolve_aum_checkpoints(
    explicit: str,
    checkpoint_root: str,
    checkpoint_manifest: str,
    *,
    expected_model: str,
    min_total: int,
    max_total: int,
) -> List[str]:
    """Resolve extra checkpoints; the base/t0 model counts as checkpoint one."""
    if min_total < 2:
        raise ValueError("--min-aum-checkpoints must be at least 2")
    if max_total < min_total:
        raise ValueError("--max-aum-checkpoints must be >= --min-aum-checkpoints")

    candidates = [item.strip() for item in explicit.split(",") if item.strip()]
    if checkpoint_manifest:
        manifest_path = Path(checkpoint_manifest).expanduser()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("checkpoints") if isinstance(payload, Mapping) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"invalid/empty AUM checkpoint manifest: {manifest_path}")
        if payload.get("model_name_or_path") != expected_model:
            raise RuntimeError(
                f"AUM manifest model={payload.get('model_name_or_path')!r} does not match "
                f"scoring model={expected_model!r}: {manifest_path}"
            )
        fingerprints = [entry.get("parameter_sample_sha256") for entry in entries]
        if any(not fingerprint for fingerprint in fingerprints) or len(set(fingerprints)) != len(fingerprints):
            raise RuntimeError(f"AUM manifest contains missing/duplicate model-state fingerprints: {manifest_path}")
        manifest_paths = [Path(str(entry.get("path") or "")).expanduser() for entry in entries]
        missing_paths = [str(path) for path in manifest_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"AUM manifest references missing checkpoint(s): {missing_paths[:5]}"
            )
        candidates.extend(str(path) for path in manifest_paths)
    if checkpoint_root:
        root = Path(checkpoint_root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"AUM checkpoint root does not exist: {root}")
        discovered = sorted(
            root.rglob("*.pt"),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )
        candidates.extend(str(path) for path in discovered)

    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        key = str(candidate_path.resolve()) if candidate_path.exists() else candidate
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    unique = unique[: max_total - 1]
    if 1 + len(unique) < min_total:
        raise RuntimeError(
            "AUM requires the base model plus compatible fine-tuning checkpoints: "
            f"found total={1 + len(unique)}, required>={min_total}. Pass "
            "--extra-checkpoints or --checkpoint-root (curriculum_ft.py .pt is supported)."
        )
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(NAIVE_RAW_PATH))
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--val-source", default="", help="jsonl of held-out rows for GradAlign reference")
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="cap #candidates for smoke tests")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--extra-checkpoints",
        default="",
        help="comma-separated compatible .pt files, HF directories, or HF ids for AUM",
    )
    ap.add_argument(
        "--checkpoint-root",
        default="",
        help="recursively discover compatible .pt AUM checkpoints below this directory",
    )
    ap.add_argument(
        "--checkpoint-manifest",
        default="",
        help="ordered distinct epoch-state manifest from aum_curriculum_entry.py",
    )
    ap.add_argument("--min-aum-checkpoints", type=int, default=3, help="minimum total including base/t0")
    ap.add_argument("--max-aum-checkpoints", type=int, default=5, help="maximum total including base/t0")
    ap.add_argument(
        "--grad-scope", default="head", choices=["last", "head", "full"],
        help="Which params contribute to the GradAlign per-sample gradient. "
        "'head' (default) = the original da_score_collect.py last-linear "
        "lm_head/embed_out weight definition (shared embed_tokens.weight for a "
        "tied head); 'last' = explicit faster final-transformer-block proxy; "
        "'full' = exact definition but ~intractable for 20k candidates.",
    )
    args = ap.parse_args()

    if not args.val_source:
        raise RuntimeError(
            "GradAlign requires --val-source. Run the AUM reference/seed training "
            "first or pass a trainer data/val.jsonl explicitly."
        )
    extra_ckpts = resolve_aum_checkpoints(
        args.extra_checkpoints,
        args.checkpoint_root,
        args.checkpoint_manifest,
        expected_model=args.model,
        min_total=args.min_aum_checkpoints,
        max_total=args.max_aum_checkpoints,
    )

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"[load] tokenizer/model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    grad_names = select_grad_params(model, args.grad_scope)
    if not grad_names:
        raise RuntimeError(
            f"--grad-scope={args.grad_scope} selected zero parameter tensors; "
            "for tied-output models use 'last' or 'head' (head includes embed_tokens)"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in grad_names)
    print(f"[grad-scope] {args.grad_scope}: tracking {len(grad_names)} param tensors for GradAlign")

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"input contains zero candidates: {args.input}")
    valid_indices: List[int] = []
    invalid_manifest: List[dict] = []
    for idx, row in enumerate(rows):
        try:
            _row_to_mcqa(row, row_name=f"candidate[{idx}]")
            valid_indices.append(idx)
        except (TypeError, ValueError) as exc:
            invalid_manifest.append(
                {
                    "idx": idx,
                    "candidate_id": row.get("candidate_id") or row.get("record_id"),
                    "reason": str(exc),
                }
            )
    if not valid_indices:
        raise RuntimeError(f"input contains zero trainer-convertible candidates: {args.input}")
    print(
        f"[input] raw={len(rows)} valid={len(valid_indices)} "
        f"invalid={len(invalid_manifest)}"
    )

    # Reference gradient for GradAlign
    val_rows = read_jsonl(args.val_source)
    ref_grads, val_meta = compute_val_grad(
        model, tokenizer, val_rows, device=args.device,
        max_val=args.n_val, grad_names=grad_names,
    )
    print(f"[grad-ref] {val_meta}  (tracked {len(ref_grads)} tensors)")

    print(f"[aum] base/t0 + {len(extra_ckpts)} checkpoint(s) = {1 + len(extra_ckpts)} total")
    margin_stacks: List[List[float]] = [[] for _ in range(len(rows))]

    def _pass_over_rows(current_model):
        for position, i in enumerate(valid_indices):
            row = rows[i]
            _el2n, _ga, mgn = compute_scores_for_row(
                current_model, tokenizer, row, args.device,
                row_name=f"candidate[{i}]", ref_grads=None,
            )
            if not math.isfinite(mgn):
                raise RuntimeError(f"candidate[{i}] produced non-finite AUM margin: {mgn}")
            margin_stacks[i].append(mgn)
            if position % 500 == 0:
                print(f"  pass valid row {position}/{len(valid_indices)} (raw idx={i}) margin={mgn}")

    # Base pass: computes EL2N, GradAlign (dotted inline), first margin sample
    out_rows: List[dict] = [None] * len(rows)  # type: ignore[list-item]
    invalid_by_idx = {item["idx"]: item for item in invalid_manifest}
    for idx, invalid in invalid_by_idx.items():
        out_rows[idx] = {
            "idx": idx,
            "candidate_id": invalid["candidate_id"],
            "valid": False,
            "invalid_reason": invalid["reason"],
            "el2n": None,
            "gradalign": None,
            "aum_proxy": None,
            "aum_proxy_ckpts": 0,
        }
    for position, i in enumerate(valid_indices):
        row = rows[i]
        el2n, ga, mgn = compute_scores_for_row(
            model, tokenizer, row, args.device,
            row_name=f"candidate[{i}]", ref_grads=ref_grads,
        )
        ga_val = float(ga)
        for key, value in (("el2n", el2n), ("gradalign", ga_val), ("margin", mgn)):
            if not math.isfinite(value):
                raise RuntimeError(f"candidate[{i}] produced non-finite {key}: {value}")
        margin_stacks[i].append(mgn)
        out_rows[i] = {
            "idx": i,
            "candidate_id": row.get("candidate_id") or row.get("record_id"),
            "valid": True,
            "el2n": el2n,
            "gradalign": ga_val,
        }
        if position % 500 == 0:
            print(
                f"[score] {position}/{len(valid_indices)} (raw idx={i}) "
                f"el2n={el2n:.3f} ga={ga_val:.3f} margin={mgn:.3f}"
            )

    # Extra checkpoints for AUM. GradAlign is complete, so release its GPU copy.
    del ref_grads
    from applications.baselines.aum_curriculum_entry import parameter_sample_sha256

    state_fingerprints = [parameter_sample_sha256(model)]
    for ckpt in extra_ckpts:
        print(f"[aum] loading extra checkpoint {ckpt}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_aum_checkpoint(args.model, ckpt, torch_dtype, args.device)
        fingerprint = parameter_sample_sha256(model)
        if fingerprint in state_fingerprints:
            raise RuntimeError(f"AUM checkpoint is a duplicate model state: {ckpt}")
        state_fingerprints.append(fingerprint)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _pass_over_rows(model)

    # Aggregate AUM = mean margin across checkpoints.
    for i in valid_indices:
        out = out_rows[i]
        stack = margin_stacks[i]
        if len(stack) != 1 + len(extra_ckpts) or not all(math.isfinite(x) for x in stack):
            raise RuntimeError(f"candidate[{i}] has incomplete AUM stack: {stack}")
        out["aum_proxy"] = float(math.fsum(stack) / len(stack))
        out["aum_proxy_ckpts"] = len(stack)
    write_jsonl(args.output, out_rows)
    meta_path = Path(args.output).with_suffix(Path(args.output).suffix + ".meta.json")
    invalid_path = Path(args.output).with_suffix(Path(args.output).suffix + ".invalid.jsonl")
    write_jsonl(invalid_path, invalid_manifest)
    write_json(
        meta_path,
        {
            "input": str(Path(args.input).expanduser().resolve()),
            "output": str(Path(args.output).expanduser().resolve()),
            "model": args.model,
            "n_raw": len(out_rows),
            "n_valid": len(valid_indices),
            "n_invalid": len(invalid_manifest),
            "invalid_manifest": str(invalid_path.resolve()),
            "val_source": str(Path(args.val_source).expanduser().resolve()),
            "validation": val_meta,
            "grad_scope": args.grad_scope,
            "grad_definition": (
                "original last-linear output weight from "
                "data_gens/da_score_collect.py:get_last_layer_params; tied heads "
                "use the shared input-embedding weight"
                if args.grad_scope == "head"
                else "explicit proxy: final transformer block gradients"
                if args.grad_scope == "last"
                else "full-model parameter gradients"
            ),
            "grad_parameter_names": sorted(grad_names),
            "aum_definition": "mean gold-vs-best-other margin over listed model states",
            "aum_checkpoints": [f"base:{args.model}", *extra_ckpts],
            "aum_parameter_sample_sha256": state_fingerprints,
            "aum_checkpoint_count": 1 + len(extra_ckpts),
        },
    )
    print(
        f"[write] {args.output} (raw={len(out_rows)}, valid={len(valid_indices)}, "
        f"invalid={len(invalid_manifest)}, aum checkpoints={1 + len(extra_ckpts)}); "
        f"metadata={meta_path} invalid_manifest={invalid_path}"
    )


if __name__ == "__main__":
    main()
