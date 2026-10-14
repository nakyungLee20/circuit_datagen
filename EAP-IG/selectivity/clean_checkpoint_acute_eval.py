#!/usr/bin/env python3
"""Acute cross-metric circuit evaluation at one frozen clean checkpoint.

The persistent-ablation Exp4 matrix compares separately trained checkpoints.
That makes raw GradAlign incomparable because each damaged checkpoint creates
its own validation-gradient scale.  This evaluator instead:

1. loads the clean trajectory's final checkpoint exactly once;
2. computes one clean, no-hook validation reference gradient;
3. applies each circuit only during the forward pass (no optimizer/update);
4. scores AUM, EL2N, and GradAlign for the same matched sample IDs; and
5. evaluates validation loss with the same frozen weights, hook off versus on.

The headline statistic is mean(clean_no_hook - clean_hooked) / SD(clean_no_hook)
for AUM/EL2N/GradAlign.  Loss retains the degradation-positive convention
hooked - clean_no_hook.
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
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    flatten_grads,
    forward_with_zero_ablation,
    get_target_params,
    infer_score_group,
    load_all_score_rows,
)
from selectivity.cross_metric_eval import (  # noqa: E402
    _limit_rows,
    load_checkpoint_into_model,
    validation_metrics,
)


CIRCUITS = ("aum", "el2n", "grad_align", "random_model")
LABELS = {
    "aum": "AUM-high",
    "el2n": "EL2N-high",
    "grad_align": "GA-high",
    "random_model": "Random (250-edge matched)",
}
OUTPUT_NAMES = {
    "aum": "aum_high",
    "el2n": "el2n_high",
    "grad_align": "gradalign_high",
    "random_model": "random_edge_matched",
}
METRICS = ("aum", "el2n", "grad_align")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--full-gradalign-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--buckets-root",
        default=(
            str(_REPO_ROOT / 'data_gens/buckets/application/Qwen__Qwen2.5-1.5B-Instruct')
        ),
    )
    parser.add_argument(
        "--circuits-root",
        default=str(_REPO_ROOT / 'circuit_results/Qwen__Qwen2.5-1.5B-Instruct'),
    )
    parser.add_argument(
        "--circuits",
        default=",".join(CIRCUITS),
        help="Comma-separated subset of aum,el2n,grad_align,random_model",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument("--random-edges-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ga-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Deterministic development subset after building the bucket union; 0=all.",
    )
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-reference-items", type=int, default=0)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for hf_id in ids:
        digest.update(hf_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bucket_union(buckets_root: Path) -> Set[str]:
    ids: Set[str] = set()
    for metric in METRICS:
        for bucket in ("high", "low"):
            path = buckets_root / metric / "sciq" / f"{metric}_sciq_{bucket}.json"
            payload = _read_json(path)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError(f"{path}: missing items list")
            ids.update(str(item["hf_id"]) for item in items if item.get("hf_id"))
    return ids


def _resolve_circuit(key: str, circuits_root: Path) -> Optional[Path]:
    if key == "random_model":
        return None
    path = find_latest_circuit_pt(
        circuits_root=circuits_root,
        dataset="sciq",
        score_key=key,
        bucket="high",
        corrupt_type="gaussian",
        group=infer_score_group(key),
    )
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing high circuit for {key} under {circuits_root}")
    return path


def _option_maps(
    input_obj: Mapping[str, Any], selected_ids: Set[str]
) -> tuple[Dict[str, List[int]], Dict[str, int]]:
    option_ids: Dict[str, List[int]] = {}
    gold_indices: Dict[str, int] = {}
    items = input_obj.get("items")
    if not isinstance(items, list):
        raise ValueError("input JSON has no items list")
    for item in items:
        hf_id = str(item.get("hf_id", ""))
        if hf_id not in selected_ids:
            continue
        labels = item.get("label_space")
        token_map = item.get("label_token_ids")
        gold_label = str(item.get("gold_label", ""))
        if not isinstance(labels, list) or not isinstance(token_map, dict):
            raise ValueError(f"{hf_id}: missing label_space/label_token_ids")
        normalized_labels = [str(label) for label in labels]
        if gold_label not in normalized_labels:
            raise ValueError(f"{hf_id}: gold label is not in label_space")
        option_ids[hf_id] = [int(token_map[label]) for label in normalized_labels]
        gold_indices[hf_id] = normalized_labels.index(gold_label)
    missing = selected_ids - set(option_ids)
    if missing:
        raise ValueError(f"missing option-token metadata for {len(missing)} selected IDs")
    return option_ids, gold_indices


def _logit_metrics(
    vector: torch.Tensor, option_ids: Sequence[int], gold_index: int
) -> tuple[float, float]:
    option = vector[torch.tensor(option_ids, device=vector.device)]
    gold = option[gold_index]
    other = torch.cat((option[:gold_index], option[gold_index + 1 :]), dim=0)
    margin = float((gold - torch.max(other)).detach().item())
    probs = torch.softmax(option.detach().float(), dim=-1)
    target = torch.zeros_like(probs)
    target[gold_index] = 1.0
    el2n = float(torch.linalg.vector_norm(probs - target).item())
    return margin, el2n


def _score_snapshot_batched(
    model: Any,
    dataloader: DataLoader,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    option_ids: Mapping[str, Sequence[int]],
    gold_indices: Mapping[str, int],
    device: str,
    description: str,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    output: Dict[str, Dict[str, float]] = {}
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=description, leave=False):
            tokens = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            positions = batch["pred_pos"].to(device)
            model.reset_hooks(including_permanent=True)
            logits = forward_fn(tokens, attention)
            for index, hf_id in enumerate(batch["hf_ids"]):
                vector = logits[index, int(positions[index].item()), :].float()
                aum, el2n = _logit_metrics(
                    vector, option_ids[hf_id], gold_indices[hf_id]
                )
                output[str(hf_id)] = {"aum": aum, "el2n": el2n}
    return output


def _score_hooked(
    model: Any,
    dataloader: DataLoader,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_params: Sequence[torch.Tensor],
    reference: torch.Tensor,
    option_ids: Mapping[str, Sequence[int]],
    gold_indices: Mapping[str, int],
    device: str,
    description: str,
) -> Dict[str, Dict[str, float]]:
    """One per-sample forward/backward supplies all three requested metrics."""
    model.eval()
    output: Dict[str, Dict[str, float]] = {}
    for batch in tqdm(dataloader, desc=description):
        tokens = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        positions = batch["pred_pos"].to(device)
        gold_ids = batch["gold_ids"].to(device)
        for index, hf_id in enumerate(batch["hf_ids"]):
            model.reset_hooks(including_permanent=True)
            model.zero_grad(set_to_none=True)
            logits = forward_fn(tokens[index : index + 1], attention[index : index + 1])
            vector = logits[0, int(positions[index].item()), :].float()
            aum, el2n = _logit_metrics(
                vector, option_ids[hf_id], gold_indices[hf_id]
            )
            loss = F.cross_entropy(vector.unsqueeze(0), gold_ids[index : index + 1])
            grads = torch.autograd.grad(
                loss, target_params, create_graph=False, retain_graph=False
            )
            grad_align = float(torch.dot(flatten_grads(grads), reference).item())
            output[str(hf_id)] = {
                "aum": aum,
                "el2n": el2n,
                "grad_align": grad_align,
            }
    return output


def _load_full_ga(path: Path, selected_ids: Set[str]) -> tuple[Dict[str, float], Dict[str, Any]]:
    payload = _read_json(path)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    values: Dict[str, float] = {}
    for item in payload.get("items", []):
        hf_id = str(item.get("hf_id", ""))
        if hf_id not in selected_ids:
            continue
        score = _finite((item.get("scores") or {}).get("grad_align"))
        if score is not None:
            values[hf_id] = score
    if set(values) != selected_ids:
        raise ValueError(
            f"{path}: full GA coverage mismatch, got={len(values)}, expected={len(selected_ids)}"
        )
    return values, dict(meta)


def _output_is_reusable(path: Path, expected: Mapping[str, Any], n_items: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if any(meta.get(key) != value for key, value in expected.items()):
        return False
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != n_items:
        return False
    for item in items:
        scores = item.get("scores") if isinstance(item, dict) else None
        if not isinstance(scores, dict):
            return False
        if any(_finite(scores.get(metric)) is None for metric in METRICS):
            return False
    return True


def _summarize(
    full: Mapping[str, Mapping[str, float]],
    hooked: Mapping[str, Mapping[str, float]],
    metric: str,
    ids: Sequence[str],
) -> Dict[str, float]:
    left = np.asarray([full[hf_id][metric] for hf_id in ids], dtype=np.float64)
    right = np.asarray([hooked[hf_id][metric] for hf_id in ids], dtype=np.float64)
    delta = float(np.mean(left - right))
    full_std = float(np.std(left, ddof=1))
    return {
        "full_mean": float(np.mean(left)),
        "hooked_mean": float(np.mean(right)),
        "raw_delta_full_minus_hooked": delta,
        "full_std": full_std,
        "delta_over_full_std": delta / full_std,
        "n": int(len(ids)),
    }


def _render_outputs(
    output_dir: Path,
    baseline: Dict[str, Any],
    row_payloads: Mapping[str, Dict[str, Any]],
    selected_ids: Sequence[str],
) -> None:
    full = {
        str(item["hf_id"]): {
            metric: float(item["scores"][metric]) for metric in METRICS
        }
        for item in baseline["items"]
    }
    full_eval = baseline["meta"]["evaluation"]
    records: List[Dict[str, Any]] = []
    for key in CIRCUITS:
        payload = row_payloads.get(key)
        if payload is None:
            continue
        hooked = {
            str(item["hf_id"]): {
                metric: float(item["scores"][metric]) for metric in METRICS
            }
            for item in payload["items"]
        }
        record: Dict[str, Any] = {"row": OUTPUT_NAMES[key], "label": LABELS[key]}
        for metric in METRICS:
            stats = _summarize(full, hooked, metric, selected_ids)
            for name, value in stats.items():
                record[f"{metric}_{name}"] = value
        hooked_eval = payload["meta"]["evaluation"]
        record["full_loss"] = float(full_eval["loss"])
        record["hooked_loss"] = float(hooked_eval["loss"])
        record["delta_loss_hooked_minus_full"] = (
            float(hooked_eval["loss"]) - float(full_eval["loss"])
        )
        record["full_accuracy"] = float(full_eval["accuracy"])
        record["hooked_accuracy"] = float(hooked_eval["accuracy"])
        records.append(record)

    meta = {
        "design": (
            "one frozen clean checkpoint; no-hook baseline versus acute hook-on; "
            "one clean no-hook validation reference gradient shared by every circuit"
        ),
        "delta": "mean(clean_no_hook - clean_hooked), paired by hf_id",
        "standardized": "delta / sample SD(clean_no_hook) on the same IDs",
        "loss_delta": "hooked validation loss - clean no-hook validation loss",
        "sample_ids_sha256": _ids_sha256(selected_ids),
        "n_samples": len(selected_ids),
        "records": records,
    }
    _write_json_atomic(output_dir / "clean_acute_matrix_meta.json", meta)

    columns = [
        "row",
        "label",
        "aum_raw_delta_full_minus_hooked",
        "el2n_raw_delta_full_minus_hooked",
        "grad_align_raw_delta_full_minus_hooked",
        "aum_delta_over_full_std",
        "el2n_delta_over_full_std",
        "grad_align_delta_over_full_std",
        "delta_loss_hooked_minus_full",
        "full_accuracy",
        "hooked_accuracy",
        "full_loss",
        "hooked_loss",
        "n",
    ]
    csv_path = output_dir / "clean_acute_matrix.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for record in records:
            values = {
                "row": record["row"],
                "label": record["label"],
                "aum_raw_delta_full_minus_hooked": record[
                    "aum_raw_delta_full_minus_hooked"
                ],
                "el2n_raw_delta_full_minus_hooked": record[
                    "el2n_raw_delta_full_minus_hooked"
                ],
                "grad_align_raw_delta_full_minus_hooked": record[
                    "grad_align_raw_delta_full_minus_hooked"
                ],
                "aum_delta_over_full_std": record["aum_delta_over_full_std"],
                "el2n_delta_over_full_std": record["el2n_delta_over_full_std"],
                "grad_align_delta_over_full_std": record[
                    "grad_align_delta_over_full_std"
                ],
                "delta_loss_hooked_minus_full": record[
                    "delta_loss_hooked_minus_full"
                ],
                "full_accuracy": record["full_accuracy"],
                "hooked_accuracy": record["hooked_accuracy"],
                "full_loss": record["full_loss"],
                "hooked_loss": record["hooked_loss"],
                "n": record["aum_n"],
            }
            handle.write(",".join(str(values[column]) for column in columns) + "\n")

    lines = [
        "| Ablated circuit | ΔAUM / SD(full) | ΔEL2N / SD(full) | "
        "ΔGA / SD(full) | raw ΔLoss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {label} | {aum:.4f} | {el2n:.4f} | {ga:.4f} | {loss:.4f} |".format(
                label=record["label"],
                aum=record["aum_delta_over_full_std"],
                el2n=record["el2n_delta_over_full_std"],
                ga=record["grad_align_delta_over_full_std"],
                loss=record["delta_loss_hooked_minus_full"],
            )
        )
    (output_dir / "clean_acute_matrix.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    input_path = Path(args.input_json)
    checkpoint_path = Path(args.checkpoint)
    full_ga_path = Path(args.full_gradalign_json)
    output_dir = Path(args.output_dir)
    buckets_root = Path(args.buckets_root)
    circuits_root = Path(args.circuits_root)
    for path in (input_path, checkpoint_path, full_ga_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.topk_edges <= 0:
        raise ValueError("--topk-edges must be positive")

    requested = [key.strip() for key in args.circuits.split(",") if key.strip()]
    if not requested or any(key not in CIRCUITS for key in requested):
        raise ValueError(f"--circuits must be a subset of {CIRCUITS}")
    if len(requested) != len(set(requested)):
        raise ValueError("--circuits contains duplicates")

    input_obj = _read_json(input_path)
    input_meta = input_obj.get("meta") if isinstance(input_obj.get("meta"), dict) else {}
    if input_meta.get("score_key") != "grad_align":
        raise ValueError("--input-json must be the exact all_grad_align_sciq.json")

    available_ids = {
        str(item.get("hf_id", ""))
        for item in input_obj.get("items", [])
        if item.get("hf_id")
    }
    selected_ids = sorted(_bucket_union(buckets_root) & available_ids)
    if args.max_items > 0 and len(selected_ids) > args.max_items:
        rng = random.Random(args.seed + 53)
        selected_ids = sorted(rng.sample(selected_ids, args.max_items))
    if not selected_ids:
        raise RuntimeError("bucket-union selection produced no samples")
    selected_set = set(selected_ids)
    option_ids, gold_indices = _option_maps(input_obj, selected_set)
    resolved_circuits = {
        key: _resolve_circuit(key, circuits_root) for key in requested
    }

    contract = {
        "input_json": str(input_path.resolve()),
        "input_json_sha256": _sha256(input_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "full_gradalign_json": str(full_ga_path.resolve()),
        "full_gradalign_json_sha256": _sha256(full_ga_path),
        "sample_policy": "union of AUM/EL2N/GradAlign high+low buckets",
        "sample_ids_sha256": _ids_sha256(selected_ids),
        "n_samples": len(selected_ids),
        "topk_edges": int(args.topk_edges),
        "random_edges_seed": int(args.random_edges_seed),
        "seed": int(args.seed),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "max_length": int(args.max_length),
        "max_val_items": int(args.max_val_items),
        "max_reference_items": int(args.max_reference_items),
        "circuits": {
            key: {
                "path": str(path.resolve()) if path else "",
                "sha256": _sha256(path) if path else "",
            }
            for key, path in resolved_circuits.items()
        },
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = str_to_dtype(args.dtype)
    if not args.device.startswith("cuda") and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_name, args.device, dtype)
    load_checkpoint_into_model(model, checkpoint_path, args.device, "clean trajectory")
    model.eval()
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = int(model.tokenizer.pad_token_id)

    all_rows, load_stats = load_all_score_rows(
        input_path, "grad_align", tokenizer=model.tokenizer
    )
    eval_rows = [all_rows[hf_id] for hf_id in selected_ids]
    eval_dataset = PromptScoreDataset(eval_rows, model.tokenizer, args.max_length)
    if len(eval_dataset) != len(selected_ids):
        raise RuntimeError(
            f"evaluation dataset lost samples: {len(eval_dataset)} != {len(selected_ids)}"
        )
    snapshot_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )
    grad_loader = DataLoader(
        eval_dataset,
        batch_size=args.ga_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )

    metadata_seed = int(input_meta.get("seed", args.seed))
    fewshot_k = int(input_meta.get("fewshot_k", 3))
    train_fraction = float(input_meta.get("train_frac", 0.8))
    _, validation_examples, fewshot_examples = prepare_dataset(
        "sciq", metadata_seed, train_fraction, fewshot_k
    )
    expected_fewshot = [str(value) for value in input_meta.get("fewshot_hf_ids", [])]
    actual_fewshot = [str(row["hf_id"]) for row in fewshot_examples]
    if expected_fewshot and actual_fewshot != expected_fewshot:
        raise ValueError("few-shot reconstruction mismatch")
    validation_examples = _limit_rows(
        validation_examples, args.max_val_items, args.seed + 37
    )
    validation_dataset = ApplicationMCQADataset(
        validation_examples,
        model.tokenizer,
        fewshot_examples=fewshot_examples,
        system_prompt=str(input_meta.get("system_prompt", "")),
        answer_instruction=str(input_meta.get("answer_instruction", "")),
        max_length=args.max_length,
        use_chat_template=bool(input_meta.get("use_chat_template", True)),
    )
    validation_loader = build_dataloader(
        validation_dataset, model.tokenizer, args.batch_size, shuffle=False
    )
    reference_rows = _limit_rows(
        validation_dataset.examples, args.max_reference_items, args.seed + 41
    )
    reference_dataset = PromptScoreDataset(
        reference_rows, model.tokenizer, args.max_length
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.ga_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )

    def forward_full(tokens: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        return model(tokens, attention_mask=attention)

    full_ga, full_ga_meta = _load_full_ga(full_ga_path, selected_set)
    if full_ga_meta.get("full_checkpoint_sha256") != contract["checkpoint_sha256"]:
        raise ValueError("full GradAlign artifact does not match the requested clean checkpoint")

    baseline_path = output_dir / "clean_no_hook.json"
    baseline_expected = {
        **{key: value for key, value in contract.items() if key != "circuits"},
        "condition": "clean_no_hook",
    }
    baseline: Dict[str, Any]
    if (
        not args.force
        and baseline_path.is_file()
        and _output_is_reusable(baseline_path, baseline_expected, len(selected_ids))
    ):
        print(f"[reuse] {baseline_path}")
        baseline = _read_json(baseline_path)
    else:
        snapshot = _score_snapshot_batched(
            model,
            snapshot_loader,
            forward_full,
            option_ids,
            gold_indices,
            args.device,
            "clean no-hook AUM/EL2N",
        )
        baseline_eval = validation_metrics(
            model, validation_loader, forward_full, args.device
        )
        baseline = {
            "meta": {
                **baseline_expected,
                "definition": {
                    "aum": "gold option logit - max other option logit",
                    "el2n": "L2(softmax(option logits), onehot gold)",
                    "grad_align": (
                        "per-sample last-layer loss gradient dot one shared clean "
                        "validation gradient"
                    ),
                },
                "evaluation": baseline_eval,
                "source_load_stats": load_stats,
            },
            "items": [
                {
                    "hf_id": hf_id,
                    "scores": {
                        "aum": snapshot[hf_id]["aum"],
                        "el2n": snapshot[hf_id]["el2n"],
                        "grad_align": full_ga[hf_id],
                    },
                }
                for hf_id in selected_ids
            ],
        }
        _write_json_atomic(baseline_path, baseline)
        print(f"[saved] {baseline_path}")

    target_params = get_target_params(model)
    reference = compute_reference_vector(
        model,
        reference_loader,
        forward_full,
        target_params,
        args.device,
        quiet=False,
    )
    reference_norm = float(torch.linalg.vector_norm(reference.float()).item())
    expected_reference_norm = _finite(full_ga_meta.get("reference_l2_norm"))
    if (
        expected_reference_norm is not None
        and not math.isclose(reference_norm, expected_reference_norm, rel_tol=5e-3)
    ):
        raise RuntimeError(
            "clean reference-gradient norm mismatch: "
            f"recomputed={reference_norm}, stored={expected_reference_norm}"
        )
    print(f"[reference] clean no-hook L2={reference_norm:.8f}")

    row_payloads: Dict[str, Dict[str, Any]] = {}
    for key in CIRCUITS:
        row_path = output_dir / f"{OUTPUT_NAMES[key]}.json"
        if key not in requested:
            if row_path.is_file():
                row_payloads[key] = _read_json(row_path)
            continue
        circuit_path = resolved_circuits[key]
        row_expected = {
            **{name: value for name, value in contract.items() if name != "circuits"},
            "condition": "clean_checkpoint_acute_hook",
            "circuit_key": key,
            "circuit_path": str(circuit_path.resolve()) if circuit_path else "",
            "circuit_sha256": _sha256(circuit_path) if circuit_path else "",
        }
        if (
            not args.force
            and _output_is_reusable(row_path, row_expected, len(selected_ids))
        ):
            print(f"[reuse] {row_path}")
            row_payloads[key] = _read_json(row_path)
            continue

        if key == "random_model":
            graph, edge_meta = build_random_ablation_graph_from_model(
                model=model,
                topk_edges=args.topk_edges,
                random_seed=args.random_edges_seed,
            )
        else:
            assert circuit_path is not None
            graph, edge_meta = build_ablation_graph(
                model=model,
                circuit_pt=circuit_path,
                topk_edges=args.topk_edges,
                rank_abs=True,
            )
        if int(edge_meta.get("mapped_edges_in_runtime", -1)) != args.topk_edges:
            raise RuntimeError(
                f"{key}: mapped edge mismatch: {edge_meta.get('mapped_edges_in_runtime')}"
            )

        def forward_hooked(
            tokens: torch.Tensor, attention: torch.Tensor
        ) -> torch.Tensor:
            return forward_with_zero_ablation(model, graph, tokens, attention)

        scores = _score_hooked(
            model,
            grad_loader,
            forward_hooked,
            target_params,
            reference,
            option_ids,
            gold_indices,
            args.device,
            f"clean acute {LABELS[key]}",
        )
        if set(scores) != selected_set:
            raise RuntimeError(
                f"{key}: score coverage mismatch {len(scores)} != {len(selected_ids)}"
            )
        hooked_eval = validation_metrics(
            model, validation_loader, forward_hooked, args.device
        )
        payload = {
            "meta": {
                **row_expected,
                "reference_vector_source": "one clean checkpoint, no hook",
                "reference_l2_norm": reference_norm,
                "weights_updated": False,
                "edge_meta": edge_meta,
                "evaluation": hooked_eval,
            },
            "items": [
                {"hf_id": hf_id, "scores": scores[hf_id]} for hf_id in selected_ids
            ],
        }
        _write_json_atomic(row_path, payload)
        print(f"[saved] {row_path}")
        row_payloads[key] = payload
        del graph, scores
        model.reset_hooks(including_permanent=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _render_outputs(output_dir, baseline, row_payloads, selected_ids)

    _render_outputs(output_dir, baseline, row_payloads, selected_ids)
    _write_json_atomic(output_dir / "clean_acute_contract.json", contract)
    print(f"[done] {output_dir / 'clean_acute_matrix.md'}")


if __name__ == "__main__":
    main()
