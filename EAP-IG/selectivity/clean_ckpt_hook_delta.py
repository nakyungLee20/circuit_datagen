"""Acute circuit effect on the clean checkpoint: hook off vs hook on, weights fixed.

Every other Exp4 contrast compares two *training trajectories*, so a row's number
mixes "this circuit matters" with "training on this ablation collapsed". Here the
weights never change: one clean full-trajectory checkpoint is scored twice, with
and without the persistent zero-ablation hook, so the difference is purely the
acute effect of removing those 250 edges.

AUM and EL2N are trajectory-accumulated during training; their single-checkpoint
instantaneous forms are used here, matching ``zero_ablation_finetune`` exactly:

    aum  = gold_logit - max(other option logits)          (option logits only)
    el2n = || softmax(option_logits) - onehot(gold) ||_2

GradAlign uses one shared reference gradient computed on the clean checkpoint with
no hook, so the hook-on/hook-off difference reflects the per-sample gradient change
rather than a change of reference frame.

One circuit per invocation so the four rows can run concurrently on one GPU.
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
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = _REPO_ROOT
if str(ROOT / "EAP-IG") not in sys.path:
    sys.path.append(str(ROOT / "EAP-IG"))

from selectivity.cross_metric_eval import (  # type: ignore  # noqa: E402
    ApplicationMCQADataset,
    PromptScoreDataset,
    _canonical_path,
    _limit_rows,
    _read_json,
    _sha256,
    build_ablation_graph,
    build_dataloader,
    build_random_ablation_graph_from_model,
    collate_prompt_score,
    compute_reference_vector,
    forward_with_zero_ablation,
    get_target_params,
    load_all_score_rows,
    load_checkpoint_into_model,
    load_model,
    prepare_dataset,
    score_samples_by_dot,
    str_to_dtype,
    validation_metrics,
)

CIRCUIT_KEYS = ("aum", "el2n", "grad_align", "random_model")


@torch.no_grad()
def score_margin_el2n(
    model: Any,
    loader: DataLoader,
    forward_fn: Callable[..., torch.Tensor],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    device: str,
) -> Dict[str, Dict[str, float]]:
    """Instantaneous AUM (option margin) and EL2N at the current weights."""
    model.eval()
    out: Dict[str, Dict[str, float]] = {}
    for batch in loader:
        tokens = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        hf_ids = [str(x) for x in batch["hf_ids"]]
        model.reset_hooks(including_permanent=True)
        logits = forward_fn(tokens, attention)
        index = torch.arange(logits.size(0), device=logits.device)
        last = logits[index, pred_pos, :].float()
        for position, hf_id in enumerate(hf_ids):
            row = rows_by_id.get(hf_id)
            if row is None:
                continue
            label_space = list(row.get("label_space") or [])
            label_token_ids = dict(row.get("label_token_ids") or {})
            gold_label = str(row.get("gold_label", ""))
            option_ids = [int(label_token_ids[l]) for l in label_space if l in label_token_ids]
            if len(option_ids) < 2 or gold_label not in label_space:
                continue
            gold_index = label_space.index(gold_label)
            options = last[position][torch.tensor(option_ids, device=last.device)]
            gold_logit = options[gold_index]
            other = torch.cat([options[:gold_index], options[gold_index + 1 :]], dim=0)
            margin = float((gold_logit - torch.max(other)).item())
            probs = torch.softmax(options, dim=-1)
            onehot = torch.zeros_like(probs)
            onehot[gold_index] = 1.0
            out[hf_id] = {
                "aum": margin,
                "el2n": float(torch.linalg.norm(probs - onehot, ord=2).item()),
            }
    return out


def _read_json_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, dict):
        obj = obj.get("hf_ids", [])
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected a JSON list of hf_ids")
    return obj


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--circuit-score-key", required=True, choices=CIRCUIT_KEYS)
    parser.add_argument("--circuit-pt", default="")
    parser.add_argument("--subset-json", default="", help="JSON list of hf_ids to score")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument("--random-edges-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-reference-items", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    input_path = Path(args.input_json).expanduser()
    checkpoint_path = Path(args.full_checkpoint).expanduser()
    output_path = Path(args.output_json).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    circuit_path: Optional[Path] = None
    if args.circuit_score_key != "random_model":
        circuit_path = Path(args.circuit_pt).expanduser()
        if not circuit_path.is_file():
            raise FileNotFoundError(circuit_path)

    subset_ids: Optional[List[str]] = None
    if args.subset_json:
        subset_ids = [str(x) for x in _read_json_list(Path(args.subset_json))]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = str_to_dtype(args.dtype)
    if not args.device.startswith("cuda") and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    model = load_model(args.model_name, args.device, dtype)
    load_checkpoint_into_model(model, checkpoint_path, args.device, "clean full trajectory")
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = int(model.tokenizer.pad_token_id)

    source_meta = _read_json(input_path).get("meta", {})
    all_rows, load_stats = load_all_score_rows(input_path, "grad_align", tokenizer=model.tokenizer)
    if subset_ids is not None:
        eval_rows = [all_rows[hf_id] for hf_id in subset_ids if hf_id in all_rows]
        if len(eval_rows) != len(subset_ids):
            raise RuntimeError(
                f"subset mismatch: requested={len(subset_ids)}, resolved={len(eval_rows)}"
            )
    else:
        eval_rows = list(all_rows.values())
    eval_dataset = PromptScoreDataset(eval_rows, model.tokenizer, args.max_length)
    if len(eval_dataset) == 0:
        raise RuntimeError("no train samples to score")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )

    metadata_seed = int(source_meta.get("seed", args.seed))
    fewshot_k = int(source_meta.get("fewshot_k", 3))
    train_fraction = float(source_meta.get("train_frac", 0.8))
    _, validation_examples, fewshot_examples = prepare_dataset(
        "sciq", metadata_seed, train_fraction, fewshot_k
    )
    expected_fewshot = [str(value) for value in source_meta.get("fewshot_hf_ids", [])]
    actual_fewshot = [str(row["hf_id"]) for row in fewshot_examples]
    if expected_fewshot and actual_fewshot != expected_fewshot:
        raise ValueError("few-shot reconstruction mismatch")
    validation_examples = _limit_rows(validation_examples, args.max_val_items, args.seed + 37)
    validation_dataset = ApplicationMCQADataset(
        validation_examples,
        model.tokenizer,
        fewshot_examples=fewshot_examples,
        system_prompt=str(source_meta.get("system_prompt", "")),
        answer_instruction=str(source_meta.get("answer_instruction", "")),
        max_length=args.max_length,
        use_chat_template=bool(source_meta.get("use_chat_template", True)),
    )
    validation_loader = build_dataloader(
        validation_dataset, model.tokenizer, args.batch_size, shuffle=False
    )
    reference_rows = _limit_rows(
        validation_dataset.examples, args.max_reference_items, args.seed + 41
    )
    reference_dataset = PromptScoreDataset(reference_rows, model.tokenizer, args.max_length)
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )

    if args.circuit_score_key == "random_model":
        ablation_graph, edge_meta = build_random_ablation_graph_from_model(
            model=model, topk_edges=args.topk_edges, random_seed=args.random_edges_seed
        )
        edge_meta["circuit_score_key"] = "random_model"
    else:
        assert circuit_path is not None
        ablation_graph, edge_meta = build_ablation_graph(
            model=model, circuit_pt=circuit_path, topk_edges=args.topk_edges, rank_abs=True
        )
        edge_meta["circuit_score_key"] = args.circuit_score_key
    if int(edge_meta.get("mapped_edges_in_runtime", -1)) != args.topk_edges:
        raise RuntimeError(
            f"edge-budget mismatch: mapped={edge_meta.get('mapped_edges_in_runtime')}"
        )

    def forward_full(tokens: torch.Tensor, attention: torch.Tensor, **_: Any) -> torch.Tensor:
        return model(tokens, attention_mask=attention)

    def forward_ablated(tokens: torch.Tensor, attention: torch.Tensor, **_: Any) -> torch.Tensor:
        return forward_with_zero_ablation(model, ablation_graph, tokens, attention)

    target_params = get_target_params(model)
    reference = compute_reference_vector(
        model, reference_loader, forward_full, target_params, args.device, args.quiet
    )
    ga_off = score_samples_by_dot(
        model, eval_loader, forward_full, target_params, reference, args.device, 1.0, args.quiet
    )
    ga_on = score_samples_by_dot(
        model, eval_loader, forward_ablated, target_params, reference, args.device, 1.0, args.quiet
    )
    margins_off = score_margin_el2n(model, eval_loader, forward_full, all_rows, args.device)
    margins_on = score_margin_el2n(model, eval_loader, forward_ablated, all_rows, args.device)
    eval_off = validation_metrics(model, validation_loader, forward_full, args.device)
    eval_on = validation_metrics(model, validation_loader, forward_ablated, args.device)

    matched = sorted(set(ga_off) & set(ga_on) & set(margins_off) & set(margins_on))
    items = [
        {
            "hf_id": hf_id,
            "hook_off": {
                "aum": margins_off[hf_id]["aum"],
                "el2n": margins_off[hf_id]["el2n"],
                "grad_align": float(ga_off[hf_id]),
            },
            "hook_on": {
                "aum": margins_on[hf_id]["aum"],
                "el2n": margins_on[hf_id]["el2n"],
                "grad_align": float(ga_on[hf_id]),
            },
        }
        for hf_id in matched
    ]
    metadata = {
        "input_json": _canonical_path(input_path),
        "checkpoint": _canonical_path(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "circuit_score_key": args.circuit_score_key,
        "circuit_pt": _canonical_path(circuit_path) if circuit_path else "",
        "circuit_pt_sha256": _sha256(circuit_path) if circuit_path else "",
        "topk_edges": args.topk_edges,
        "design": "one clean checkpoint scored twice; weights identical, hook off vs hook on",
        "aum_definition": "gold option logit minus max other option logit, single checkpoint",
        "el2n_definition": "L2 norm of (softmax(option logits) - onehot(gold)), single checkpoint",
        "grad_align_definition": (
            "per-sample last-layer loss gradient dot the shared clean no-hook "
            "SciQ validation reference gradient"
        ),
        "reference_l2_norm": float(torch.linalg.vector_norm(reference.float()).item()),
        "seed": args.seed,
        "model_name": args.model_name,
        "dtype": str(dtype).removeprefix("torch."),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_scored": len(items),
        "n_reference": len(reference_dataset),
        "subset_json": _canonical_path(Path(args.subset_json)) if args.subset_json else "",
        "source_load_stats": load_stats,
        "edge_meta": edge_meta,
        "evaluation": {"hook_off": eval_off, "hook_on": eval_on},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"meta": metadata, "items": items}, handle, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)
    print(f"[ok] clean-checkpoint hook delta ({args.circuit_score_key}): {output_path}")


if __name__ == "__main__":
    main()
