"""Exact GradAlign and validation evaluation for one fixed ablated circuit.

The training script couples ``score_key`` (input utility) and circuit lookup.
This Exp4-only evaluator separates them: ``--circuit-score-key`` selects the
fixed circuit, while the measured utility is always true GradAlign. It loads
the final persistent-ablation checkpoint, recomputes the validation-gradient
dot product for every training sample, and records paired no-hook/ablated-hook
scores at that *same checkpoint*. Primary validation Accuracy/loss instead
compare the matched no-ablation trajectory final checkpoint with the persistent-
ablation trajectory final checkpoint; the same-ablated-checkpoint no-hook result
is retained only as an acute diagnostic.

This is GPU-heavy and is not run by unit tests. ``--dry-run`` resolves every
input and circuit without loading the model.
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
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = _REPO_ROOT
for directory in (
    ROOT,
    ROOT / "EAP-IG",
    ROOT / "EAP-IG" / "src",
    ROOT / "data_gens",
):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

from scen1_eapig import load_model, str_to_dtype  # type: ignore  # noqa: E402
from app_score_collect import (  # type: ignore  # noqa: E402
    ApplicationMCQADataset,
    build_dataloader,
    prepare_dataset,
)
from zero_ablation import (  # type: ignore  # noqa: E402
    PromptScoreDataset,
    build_ablation_graph,
    build_random_ablation_graph_from_model,
    collate_prompt_score,
    compute_reference_vector,
    find_latest_circuit_pt,
    forward_with_zero_ablation,
    get_target_params,
    infer_score_group,
    load_all_score_rows,
    score_samples_by_dot,
)


CIRCUIT_KEYS = ("aum", "el2n", "grad_align", "random_model")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def _limit_rows(rows: Sequence[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    result = list(rows)
    if count > 0 and len(result) > count:
        rng = random.Random(seed)
        result = rng.sample(result, count)
        result.sort(key=lambda row: str(row.get("hf_id", "")))
    return result


def _canonical_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def reusable_overlay(
    existing: Dict[str, Any], expected_meta: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate an overlay without loading a model.

    This deliberately treats missing legacy metadata as stale: reuse is safe
    only when every output-affecting evaluator option is certified.
    """
    reasons: List[str] = []
    meta = existing.get("meta") if isinstance(existing.get("meta"), dict) else {}
    for key, expected in expected_meta.items():
        actual = meta.get(key)
        if actual != expected:
            reasons.append(f"meta.{key}: existing={actual!r}, expected={expected!r}")

    evaluation = meta.get("evaluation")
    if not isinstance(evaluation, dict):
        reasons.append("meta.evaluation is missing")
    else:
        for variant in ("baseline", "ablated", "acute_no_hook_same_checkpoint"):
            values = evaluation.get(variant)
            if not isinstance(values, dict):
                reasons.append(f"meta.evaluation.{variant} is missing")
                continue
            for key in ("accuracy", "loss"):
                try:
                    finite = math.isfinite(float(values.get(key)))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    reasons.append(f"meta.evaluation.{variant}.{key} is not finite")
            try:
                valid_n = int(values.get("n", 0)) > 0
            except (TypeError, ValueError):
                valid_n = False
            if not valid_n:
                reasons.append(f"meta.evaluation.{variant}.n is not positive")

    items = existing.get("items") if isinstance(existing.get("items"), list) else []
    if not items:
        reasons.append("items are missing")
    ids: List[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            reasons.append(f"items[{index}] is not an object")
            continue
        hf_id = str(item.get("hf_id", "")).strip()
        if not hf_id:
            reasons.append(f"items[{index}].hf_id is missing")
        ids.append(hf_id)
        for section in ("scores", "full_recomputed_scores"):
            values = item.get(section)
            value = values.get("grad_align") if isinstance(values, dict) else None
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                reasons.append(f"items[{index}].{section}.grad_align is not finite")
    if len(ids) != len(set(ids)):
        reasons.append("items contain duplicate hf_id values")
    if meta.get("n_scored") != len(items):
        reasons.append(
            f"meta.n_scored={meta.get('n_scored')!r}, actual items={len(items)}"
        )
    return not reasons, reasons


def resolve_circuit(
    *,
    circuit_score_key: str,
    circuit_pt_override: str,
    circuits_root: Path,
) -> Optional[Path]:
    if circuit_score_key == "random_model":
        if circuit_pt_override:
            raise ValueError("--circuit-pt cannot be used with random_model")
        return None
    if circuit_pt_override:
        path = Path(circuit_pt_override)
        if not path.is_file():
            raise FileNotFoundError(f"circuit pt not found: {path}")
        return path
    group = infer_score_group(circuit_score_key)
    path = find_latest_circuit_pt(
        circuits_root=circuits_root,
        dataset="sciq",
        score_key=circuit_score_key,
        bucket="high",
        corrupt_type="gaussian",
        group=group,
    )
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"high circuit not found for score_key={circuit_score_key} under {circuits_root}"
        )
    return path


def validation_metrics(
    model: Any,
    dataloader: DataLoader,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> Dict[str, float]:
    """MCQA option accuracy plus full-vocabulary gold-token CE loss."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in dataloader:
            tokens = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            positions = batch["pred_pos"].to(device)
            model.reset_hooks(including_permanent=True)
            logits = forward_fn(tokens, attention)
            for index in range(logits.shape[0]):
                position = int(positions[index].item())
                vector = logits[index, position, :].float()
                gold_token = int(batch["gold_token_id"][index])
                loss_sum += float(
                    F.cross_entropy(
                        vector.unsqueeze(0),
                        torch.tensor([gold_token], device=vector.device),
                    ).item()
                )
                labels = [str(value) for value in batch["label_space"][index]]
                token_map = batch["label_token_ids"][index]
                gold_label = str(batch["gold_label"][index])
                option_ids = [int(token_map[label]) for label in labels]
                predicted = labels[int(torch.argmax(vector[option_ids]).item())]
                correct += int(predicted == gold_label)
                total += 1
    if total == 0:
        raise RuntimeError("validation dataloader produced no samples")
    return {
        "accuracy": correct / total,
        "loss": loss_sum / total,
        "n": int(total),
    }


def load_checkpoint_into_model(
    model: Any, checkpoint_path: Path, device: str, label: str
) -> None:
    state_obj = torch.load(str(checkpoint_path), map_location=device)
    state_dict = (
        state_obj.get("model_state_dict", state_obj)
        if isinstance(state_obj, dict)
        else state_obj
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{label} checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="all_grad_align_sciq.json")
    parser.add_argument("--checkpoint", required=True, help="final.pt from the trajectory")
    parser.add_argument(
        "--full-checkpoint",
        required=True,
        help="Matched no-ablation trajectory final.pt for primary Acc/Loss",
    )
    parser.add_argument("--circuit-score-key", required=True, choices=CIRCUIT_KEYS)
    parser.add_argument("--circuit-pt", default="", help="Exact circuit used in training")
    parser.add_argument(
        "--circuits-root",
        default=str(_REPO_ROOT / 'circuit_results/Qwen__Qwen2.5-1.5B-Instruct'),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument("--random-edges-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all train samples")
    parser.add_argument("--max-val-items", type=int, default=0, help="0 = all validation samples")
    parser.add_argument(
        "--max-reference-items", type=int, default=0,
        help="0 = all validation samples for the validation gradient",
    )
    parser.add_argument(
        "--reference-vector-source", choices=("full", "ablated"), default="full"
    )
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-if-valid", action="store_true",
        help="Skip only when the existing artifact matches every input/config field.",
    )
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    input_path = Path(args.input_json)
    checkpoint_path = Path(args.checkpoint)
    full_checkpoint_path = Path(args.full_checkpoint)
    output_path = Path(args.output_json)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not full_checkpoint_path.is_file():
        raise FileNotFoundError(full_checkpoint_path)
    if checkpoint_path.resolve() == full_checkpoint_path.resolve():
        raise ValueError("--checkpoint and --full-checkpoint must be distinct trajectories")
    if args.topk_edges <= 0:
        raise ValueError("--topk-edges must be positive")

    source_obj = _read_json(input_path)
    source_meta = source_obj.get("meta") if isinstance(source_obj.get("meta"), dict) else {}
    if str(source_meta.get("score_key")) != "grad_align":
        raise ValueError(f"{input_path}: score_key is not grad_align")
    circuit_path = resolve_circuit(
        circuit_score_key=args.circuit_score_key,
        circuit_pt_override=args.circuit_pt,
        circuits_root=Path(args.circuits_root),
    )
    dry_meta = {
        "input_json": _canonical_path(input_path),
        "input_json_sha256": _sha256(input_path),
        "checkpoint": _canonical_path(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "full_checkpoint": _canonical_path(full_checkpoint_path),
        "full_checkpoint_sha256": _sha256(full_checkpoint_path),
        "circuit_score_key": args.circuit_score_key,
        "circuit_pt": _canonical_path(circuit_path) if circuit_path else "",
        "circuit_pt_sha256": _sha256(circuit_path) if circuit_path else "",
        "fingerprint_algorithm": "sha256",
        "topk_edges": int(args.topk_edges),
        "output_json": _canonical_path(output_path),
        "seed": int(args.seed),
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_length": int(args.max_length),
        "max_items": int(args.max_items),
        "max_val_items": int(args.max_val_items),
        "max_reference_items": int(args.max_reference_items),
        "reference_vector_source": args.reference_vector_source,
        "random_edges_seed": int(args.random_edges_seed),
    }
    if args.dry_run:
        print(json.dumps(dry_meta, indent=2))
        return
    if args.reuse_if_valid and output_path.is_file():
        existing = _read_json(output_path)
        expected_pairs = {
            **{key: value for key, value in dry_meta.items() if key != "output_json"},
            "metric_score_key": "grad_align",
            "full_score_source": "same-checkpoint no-hook q_full_recomputed",
            "ablated_score_source": "same-checkpoint persistent zero-ablation q_ablated",
            "reference_split": "sciq/validation excluding the original few-shot examples",
            "evaluation_primary": (
                "matched full-trajectory final no-hook vs persistent-ablated "
                "trajectory final with hook"
            ),
            "evaluation_acute_diagnostic": (
                "same ablated-trained checkpoint: no-hook vs persistent hook"
            ),
        }
        matches, mismatch_reasons = reusable_overlay(existing, expected_pairs)
        if matches:
            print(f"[skip] matching exact cross-metric overlay: {output_path}")
            return
        print(f"[stale] existing overlay does not match current inputs; recomputing: {output_path}")
        for reason in mismatch_reasons[:8]:
            print(f"  - {reason}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = str_to_dtype(args.dtype)
    if not args.device.startswith("cuda") and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    model = load_model(args.model_name, args.device, dtype)
    load_checkpoint_into_model(model, checkpoint_path, args.device, "ablated trajectory")
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = int(model.tokenizer.pad_token_id)

    all_rows, load_stats = load_all_score_rows(
        input_path, "grad_align", tokenizer=model.tokenizer
    )
    eval_rows = _limit_rows(list(all_rows.values()), args.max_items, args.seed + 31)
    eval_dataset = PromptScoreDataset(eval_rows, model.tokenizer, args.max_length)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )
    if len(eval_dataset) == 0:
        raise RuntimeError("no train samples available for GradAlign scoring")

    metadata_seed = int(source_meta.get("seed", args.seed))
    fewshot_k = int(source_meta.get("fewshot_k", 3))
    train_fraction = float(source_meta.get("train_frac", 0.8))
    _, validation_examples, fewshot_examples = prepare_dataset(
        "sciq", metadata_seed, train_fraction, fewshot_k
    )
    expected_fewshot = [str(value) for value in source_meta.get("fewshot_hf_ids", [])]
    actual_fewshot = [str(row["hf_id"]) for row in fewshot_examples]
    if expected_fewshot and actual_fewshot != expected_fewshot:
        raise ValueError(
            f"few-shot reconstruction mismatch: expected={expected_fewshot}, actual={actual_fewshot}"
        )
    validation_examples = _limit_rows(
        validation_examples, args.max_val_items, args.seed + 37
    )
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
    if len(reference_dataset) == 0:
        raise RuntimeError("no SciQ validation samples available for reference gradient")

    edge_meta: Dict[str, Any]
    if args.circuit_score_key == "random_model":
        ablation_graph, edge_meta = build_random_ablation_graph_from_model(
            model=model,
            topk_edges=args.topk_edges,
            random_seed=args.random_edges_seed,
        )
        edge_meta["circuit_score_key"] = "random_model"
    else:
        assert circuit_path is not None
        ablation_graph, edge_meta = build_ablation_graph(
            model=model,
            circuit_pt=circuit_path,
            topk_edges=args.topk_edges,
            rank_abs=True,
        )
        edge_meta["circuit_score_key"] = args.circuit_score_key
    if int(edge_meta.get("mapped_edges_in_runtime", -1)) != args.topk_edges:
        raise RuntimeError(
            "edge-budget mismatch: "
            f"mapped={edge_meta.get('mapped_edges_in_runtime')}, expected={args.topk_edges}"
        )

    def forward_full(tokens: torch.Tensor, attention: torch.Tensor, **_: Any) -> torch.Tensor:
        return model(tokens, attention_mask=attention)

    def forward_ablated(tokens: torch.Tensor, attention: torch.Tensor, **_: Any) -> torch.Tensor:
        return forward_with_zero_ablation(model, ablation_graph, tokens, attention)

    target_params = get_target_params(model)
    full_reference = compute_reference_vector(
        model, reference_loader, forward_full, target_params,
        args.device, args.quiet,
    )
    if args.reference_vector_source == "ablated":
        ablated_reference = compute_reference_vector(
            model, reference_loader, forward_ablated, target_params,
            args.device, args.quiet,
        )
    else:
        ablated_reference = full_reference

    full_scores = score_samples_by_dot(
        model, eval_loader, forward_full, target_params, full_reference,
        args.device, 1.0, args.quiet,
    )
    ablated_scores = score_samples_by_dot(
        model, eval_loader, forward_ablated, target_params, ablated_reference,
        args.device, 1.0, args.quiet,
    )
    matched_ids = sorted(set(full_scores) & set(ablated_scores))
    if len(matched_ids) != len(eval_dataset):
        raise RuntimeError(
            f"GradAlign output lost samples: matched={len(matched_ids)}, expected={len(eval_dataset)}"
        )

    acute_baseline_eval = validation_metrics(
        model, validation_loader, forward_full, args.device
    )
    ablated_eval = validation_metrics(model, validation_loader, forward_ablated, args.device)
    model.reset_hooks(including_permanent=True)
    load_checkpoint_into_model(
        model, full_checkpoint_path, args.device, "matched full trajectory"
    )
    matched_full_eval = validation_metrics(
        model, validation_loader, forward_full, args.device
    )
    items = [
        {
            "hf_id": hf_id,
            "scores": {"grad_align": float(ablated_scores[hf_id])},
            "full_recomputed_scores": {"grad_align": float(full_scores[hf_id])},
        }
        for hf_id in matched_ids
    ]
    metadata = {
        **dry_meta,
        "metric_score_key": "grad_align",
        "definition": "per-sample last-layer loss gradient dot SciQ validation gradient",
        "full_score_source": "same-checkpoint no-hook q_full_recomputed",
        "ablated_score_source": "same-checkpoint persistent zero-ablation q_ablated",
        "reference_vector_source": args.reference_vector_source,
        "reference_split": "sciq/validation excluding the original few-shot examples",
        "evaluation_primary": (
            "matched full-trajectory final no-hook vs persistent-ablated "
            "trajectory final with hook"
        ),
        "evaluation_acute_diagnostic": (
            "same ablated-trained checkpoint: no-hook vs persistent hook"
        ),
        "edge_meta": edge_meta,
        "effective_dtype": str(dtype).removeprefix("torch."),
        "n_scored": len(items),
        "n_reference": len(reference_dataset),
        "source_load_stats": load_stats,
        "evaluation": {
            "baseline": matched_full_eval,
            "ablated": ablated_eval,
            "acute_no_hook_same_checkpoint": acute_baseline_eval,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"meta": metadata, "items": items}, handle, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)
    print(f"[ok] exact cross-metric overlay: {output_path}")


if __name__ == "__main__":
    main()
