from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gens.mcqa import (  # type: ignore
    MCQATokenDataset,
    canonical_dataset_name,
    collate_batch,
    get_option_logits,
    load_checkpoint_state_dict,
    prepare_task_splits,
    set_seed,
)

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "curriculum_ft"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "curriculum_if"
CHECKPOINT_CHOICE = ("best", "final", "both")
EVAL_SPLIT_CHOICES = ("train", "val", "test")

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)

def save_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip()).strip("_") or "run"

def maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))

def entropy_from_probs(probs: torch.Tensor) -> float:
    return float((-(probs * torch.log(probs.clamp_min(1e-8))).sum()).item())

def entropy_from_counts(counts: Mapping[str, int]) -> float:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        prob = float(count) / total
        if prob > 0:
            ent -= prob * math.log(prob)
    return float(ent)

def resolve_torch_dtype(dtype_name: str, device: str) -> Optional[torch.dtype]:
    if not device.startswith("cuda"):
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

def build_dataloader(dataset: MCQATokenDataset, *, batch_size: int, num_workers: int, pad_id: int, device: str) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.startswith("cuda"),
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        drop_last=False,
    )

def maybe_limit_rows(rows: Sequence[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    arr = list(rows)
    if limit <= 0 or len(arr) <= limit:
        return arr
    rng = torch.Generator()
    rng.manual_seed(int(seed))
    perm = torch.randperm(len(arr), generator=rng).tolist()
    return [arr[idx] for idx in perm[:limit]]

def list_summary_paths(search_root: Path) -> List[Path]:
    if not search_root.exists():
        return []
    return sorted(search_root.rglob("run_summary.json"))

def summary_matches_filters(
    summary: Mapping[str, Any],
    *,
    baseline_mode: str,
    curriculum_mode: str,
    variant_name: str,
    variant_name_contains: str,
    raw_score_key: str,
    run_name: str,
) -> bool:
    if baseline_mode and str(summary.get("baseline_mode", "")) != baseline_mode:
        return False
    if curriculum_mode and str(summary.get("curriculum_mode", "")) != curriculum_mode:
        return False
    if variant_name and str(summary.get("variant_name", "")) != variant_name:
        return False
    if variant_name_contains and variant_name_contains not in str(summary.get("variant_name", "")):
        return False
    if raw_score_key and str(summary.get("raw_score_key", "") or "") != raw_score_key:
        return False
    if run_name:
        run_dir = Path(str(summary.get("run_dir", "")))
        if run_dir.name != run_name:
            return False
    return True

def discover_run_summaries(args: argparse.Namespace) -> List[Tuple[Path, Dict[str, Any]]]:
    if args.run_dir:
        out: List[Tuple[Path, Dict[str, Any]]] = []
        for raw in args.run_dir:
            run_dir = Path(raw).expanduser()
            summary_path = run_dir / "run_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"run_summary.json not found under run_dir={run_dir}")
            out.append((run_dir, load_json(summary_path)))
        return out

    search_root = Path(args.input_root).expanduser()
    if args.dataset:
        search_root = search_root / args.dataset
    if args.model_key:
        search_root = search_root / args.model_key

    matches: List[Tuple[Path, Dict[str, Any]]] = []
    for summary_path in list_summary_paths(search_root):
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        if not summary_matches_filters(
            summary,
            baseline_mode=str(args.baseline_mode or ""),
            curriculum_mode=str(args.curriculum_mode or ""),
            variant_name=str(args.variant_name or ""),
            variant_name_contains=str(args.variant_name_contains or ""),
            raw_score_key=str(args.raw_score_key or ""),
            run_name=str(args.run_name or ""),
        ):
            continue
        run_dir = Path(str(summary.get("run_dir", summary_path.parent)))
        matches.append((run_dir, summary))

    if args.latest_per_variant:
        latest: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
        for run_dir, summary in matches:
            key = str(summary.get("variant_name", run_dir.parent.name))
            prev = latest.get(key)
            if prev is None or run_dir.stat().st_mtime > prev[0].stat().st_mtime:
                latest[key] = (run_dir, summary)
        matches = sorted(latest.values(), key=lambda item: str(item[0]))

    return matches

def iter_checkpoint_targets(run_dir: Path, summary: Mapping[str, Any], checkpoint_choice: str) -> List[Tuple[str, Path]]:
    checkpoints = summary.get("checkpoints", {}) if isinstance(summary.get("checkpoints"), dict) else {}
    candidates: List[Tuple[str, Path]] = []
    kinds = ["best", "final"] if checkpoint_choice == "both" else [checkpoint_choice]
    for kind in kinds:
        path = Path(str(checkpoints.get(kind, run_dir / "checkpoints" / f"{kind}.pt")))
        if not path.is_file():
            raise FileNotFoundError(f"{kind} checkpoint not found: {path}")
        candidates.append((kind, path))
    return candidates

def infer_model_name(summary: Mapping[str, Any], override: str) -> str:
    if override:
        return override
    model_name = str(summary.get("model_name_or_path") or summary.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("Could not infer model_name_or_path from run summary. Pass --model-name-or-path explicitly.")
    return model_name

def find_epoch_row(summary: Mapping[str, Any], epoch: int) -> Optional[Mapping[str, Any]]:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    rows = metrics.get("epochs", []) if isinstance(metrics.get("epochs"), list) else []
    for row in rows:
        if int(row.get("epoch", -1)) == int(epoch):
            return row
    return None

def compute_checkpoint_budget(summary: Mapping[str, Any], ckpt_meta: Mapping[str, Any], checkpoint_kind: str) -> Dict[str, Any]:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    epoch_rows = metrics.get("epochs", []) if isinstance(metrics.get("epochs"), list) else []
    stage_schedule = summary.get("stage_schedule", {}) if isinstance(summary.get("stage_schedule"), dict) else {}
    stage_token_summaries = stage_schedule.get("stage_token_summaries", {}) if isinstance(stage_schedule.get("stage_token_summaries"), dict) else {}

    checkpoint_epoch = int(ckpt_meta.get("epoch") or (epoch_rows[-1]["epoch"] if epoch_rows else 0))
    selected_epoch_rows = [row for row in epoch_rows if int(row.get("epoch", 0)) <= checkpoint_epoch]
    selected_epoch_row = find_epoch_row(summary, checkpoint_epoch)
    if selected_epoch_row is None and selected_epoch_rows:
        selected_epoch_row = selected_epoch_rows[-1]

    cumulative_seconds = sum(float(row.get("epoch_seconds", 0.0)) for row in selected_epoch_rows)
    cumulative_examples_seen = 0
    cumulative_tokens_seen = 0
    for row in selected_epoch_rows:
        stage_id = int(row.get("stage_id", 0))
        stage_key = f"stage_{stage_id}"
        stage_info = stage_token_summaries.get(stage_key, {}) if isinstance(stage_token_summaries.get(stage_key), dict) else {}
        cumulative_examples_seen += int(stage_info.get("n_examples", 0))
        cumulative_tokens_seen += int(stage_info.get("total_tokens_once", 0))

    selected_val_acc = maybe_float(selected_epoch_row.get("val_acc")) if isinstance(selected_epoch_row, Mapping) else None
    selected_stage_id = int(selected_epoch_row.get("stage_id", 0)) if isinstance(selected_epoch_row, Mapping) else None
    selected_stage_name = str(selected_epoch_row.get("stage_name", "")) if isinstance(selected_epoch_row, Mapping) else ""

    return {
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_stage_id": selected_stage_id,
        "checkpoint_stage_name": selected_stage_name,
        "checkpoint_val_acc": selected_val_acc,
        "checkpoint_train_seconds": float(cumulative_seconds),
        "checkpoint_examples_seen": int(cumulative_examples_seen),
        "checkpoint_tokens_seen": int(cumulative_tokens_seen),
    }

@torch.no_grad()
def evaluate_checkpoint(
    *,
    model: Any,
    dataloader: DataLoader,
    device: str,
    save_predictions: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    correct = 0
    total = 0
    pred_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    correct_by_label: Counter[str] = Counter()
    entropies: List[float] = []
    margins: List[float] = []
    gold_probs: List[float] = []
    pred_probs: List[float] = []
    nlls: List[float] = []
    predictions: List[Dict[str, Any]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits
        bsz = int(logits.shape[0])

        for idx in range(bsz):
            pos = int(pred_pos[idx].item())
            label_space = list(batch["label_space"][idx])
            label_token_ids = batch["label_token_ids"][idx]
            gold_label = str(batch["gold_label"][idx])
            option_logits = get_option_logits(logits[idx, pos, :], label_space, label_token_ids)
            probs = torch.softmax(option_logits, dim=-1)

            pred_idx = int(torch.argmax(probs).item())
            pred_label = label_space[pred_idx]
            gold_idx = label_space.index(gold_label)
            gold_prob = float(probs[gold_idx].item())
            pred_prob = float(probs[pred_idx].item())
            topk = torch.topk(probs, k=min(2, int(probs.shape[0]))).values
            margin = float(topk[0].item()) if int(topk.shape[0]) == 1 else float(topk[0].item() - topk[1].item())
            entropy = entropy_from_probs(probs)
            is_correct = int(pred_idx == gold_idx)

            correct += is_correct
            total += 1
            pred_counts[pred_label] += 1
            gold_counts[gold_label] += 1
            correct_by_label[gold_label] += is_correct
            entropies.append(entropy)
            margins.append(margin)
            gold_probs.append(gold_prob)
            pred_probs.append(pred_prob)
            nlls.append(float(-math.log(max(gold_prob, 1e-8))))

            if save_predictions:
                predictions.append(
                    {
                        "hf_id": str(batch["hf_id"][idx]),
                        "gold_label": gold_label,
                        "pred_label": pred_label,
                        "correct": bool(is_correct),
                        "gold_prob": float(gold_prob),
                        "pred_prob": float(pred_prob),
                        "margin": float(margin),
                        "option_entropy": float(entropy),
                        "option_probs": {lab: float(prob) for lab, prob in zip(label_space, probs.tolist())},
                    }
                )

    metrics = {
        "accuracy": float(correct / max(1, total)),
        "mean_option_entropy": avg(entropies),
        "mean_margin": avg(margins),
        "mean_gold_prob": avg(gold_probs),
        "mean_pred_prob": avg(pred_probs),
        "mean_nll": avg(nlls),
        "pred_label_entropy": float(entropy_from_counts(pred_counts)),
        "n_eval": int(total),
        "gold_label_counts": dict(gold_counts),
        "pred_label_counts": dict(pred_counts),
        "per_label_accuracy": {
            label: float(correct_by_label[label] / max(1, gold_counts[label]))
            for label in sorted(gold_counts.keys())
        },
    }
    model.train()
    return metrics, predictions

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

def flatten_for_csv(prefix: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_for_csv(name, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out

def build_eval_record(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    summary: Mapping[str, Any],
    ckpt_kind: str,
    ckpt_path: Path,
    ckpt_meta: Mapping[str, Any],
    budget: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    train_eff = summary.get("efficiency", {}) if isinstance(summary.get("efficiency"), dict) else {}
    train_cfg = summary.get("train_config", {}) if isinstance(summary.get("train_config"), dict) else {}
    sizes = summary.get("sizes", {}) if isinstance(summary.get("sizes"), dict) else {}
    best_val_acc = maybe_float(summary.get("metrics", {}).get("best_val_acc")) if isinstance(summary.get("metrics"), dict) else None
    final_val_acc = maybe_float(summary.get("metrics", {}).get("final_val_acc")) if isinstance(summary.get("metrics"), dict) else None
    checkpoint_val_acc = maybe_float(budget.get("checkpoint_val_acc"))
    test_acc = maybe_float(test_metrics.get("accuracy"))
    checkpoint_train_seconds = maybe_float(budget.get("checkpoint_train_seconds"))
    checkpoint_tokens_seen = maybe_float(budget.get("checkpoint_tokens_seen"))
    train_examples_per_stage = maybe_float(sizes.get("n_train_requested_per_stage"))
    record = {
        "dataset": str(summary.get("dataset", args.dataset)),
        "eval_split": str(args.eval_split),
        "run_dir": str(run_dir),
        "variant_name": str(summary.get("variant_name", "")),
        "baseline_mode": str(summary.get("baseline_mode", "")),
        "curriculum_mode": str(summary.get("curriculum_mode", "")),
        "raw_score_key": str(summary.get("raw_score_key", "") or ""),
        "model_name_or_path": str(summary.get("model_name_or_path", "")),
        "checkpoint_kind": ckpt_kind,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": int(budget.get("checkpoint_epoch") or 0),
        "checkpoint_stage_id": budget.get("checkpoint_stage_id"),
        "checkpoint_stage_name": str(budget.get("checkpoint_stage_name", "")),
        "checkpoint_val_acc": checkpoint_val_acc,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "test_accuracy": test_acc,
        "test_mean_option_entropy": maybe_float(test_metrics.get("mean_option_entropy")),
        "test_mean_margin": maybe_float(test_metrics.get("mean_margin")),
        "test_mean_gold_prob": maybe_float(test_metrics.get("mean_gold_prob")),
        "test_mean_pred_prob": maybe_float(test_metrics.get("mean_pred_prob")),
        "test_mean_nll": maybe_float(test_metrics.get("mean_nll")),
        "test_pred_label_entropy": maybe_float(test_metrics.get("pred_label_entropy")),
        "n_eval": int(test_metrics.get("n_eval", 0)),
        "generalization_gap_vs_checkpoint_val": (checkpoint_val_acc - test_acc) if checkpoint_val_acc is not None and test_acc is not None else None,
        "generalization_gap_vs_best_val": (best_val_acc - test_acc) if best_val_acc is not None and test_acc is not None else None,
        "generalization_gap_vs_final_val": (final_val_acc - test_acc) if final_val_acc is not None and test_acc is not None else None,
        "checkpoint_train_seconds": checkpoint_train_seconds,
        "checkpoint_tokens_seen": checkpoint_tokens_seen,
        "checkpoint_examples_seen": int(budget.get("checkpoint_examples_seen") or 0),
        "total_train_seconds_full": maybe_float(train_eff.get("total_train_seconds")),
        "tokens_seen_total_full": maybe_float(train_eff.get("tokens_seen_total")),
        "global_steps_full": maybe_float(train_eff.get("global_steps")),
        "train_examples_per_stage": train_examples_per_stage,
        "epochs": maybe_float(train_cfg.get("epochs")),
        "batch_size": maybe_float(train_cfg.get("batch_size")),
        "eval_batch_size": maybe_float(train_cfg.get("eval_batch_size")),
        "lr": maybe_float(train_cfg.get("lr")),
    }
    if checkpoint_train_seconds and checkpoint_train_seconds > 0 and test_acc is not None:
        record["test_acc_per_train_hour"] = float(test_acc / (checkpoint_train_seconds / 3600.0))
    else:
        record["test_acc_per_train_hour"] = None
    if checkpoint_tokens_seen and checkpoint_tokens_seen > 0 and test_acc is not None:
        record["test_acc_per_million_tokens"] = float(test_acc / (checkpoint_tokens_seen / 1_000_000.0))
    else:
        record["test_acc_per_million_tokens"] = None
    if train_examples_per_stage and train_examples_per_stage > 0 and test_acc is not None:
        record["test_acc_per_1k_train_examples"] = float(test_acc / (train_examples_per_stage / 1000.0))
    else:
        record["test_acc_per_1k_train_examples"] = None
    # Per-label accuracy and gold counts — needed for per-class analysis in the results table
    for label, label_acc in sorted((test_metrics.get("per_label_accuracy") or {}).items()):
        record[f"test_acc_label_{label}"] = float(label_acc) if label_acc is not None else None
    for label, count in sorted((test_metrics.get("gold_label_counts") or {}).items()):
        record[f"gold_n_label_{label}"] = int(count)
    return record

def evaluate_run_checkpoint(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    summary: Mapping[str, Any],
    ckpt_kind: str,
    ckpt_path: Path,
    eval_examples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    model_name_or_path = infer_model_name(summary, args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    eval_ds = MCQATokenDataset(eval_examples, tokenizer=tokenizer, max_length=int(args.max_length))
    if len(eval_ds) == 0:
        raise RuntimeError("Evaluation dataset is empty after tokenization")
    eval_dl = build_dataloader(
        eval_ds,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pad_id=pad_id,
        device=args.device,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.dtype, args.device),
    )
    state_dict, ckpt_obj = load_checkpoint_state_dict(str(ckpt_path), map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        raise RuntimeError(
            f"Checkpoint load mismatch for {ckpt_path}: "
            f"missing={getattr(load_result, 'missing_keys', None)} "
            f"unexpected={getattr(load_result, 'unexpected_keys', None)}"
        )
    model.to(args.device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    test_metrics, predictions = evaluate_checkpoint(
        model=model,
        dataloader=eval_dl,
        device=args.device,
        save_predictions=bool(args.save_predictions),
    )
    budget = compute_checkpoint_budget(summary, ckpt_obj.get("meta", {}) if isinstance(ckpt_obj, dict) else {}, ckpt_kind)
    record = build_eval_record(
        args=args,
        run_dir=run_dir,
        summary=summary,
        ckpt_kind=ckpt_kind,
        ckpt_path=ckpt_path,
        ckpt_meta=ckpt_obj.get("meta", {}) if isinstance(ckpt_obj, dict) else {},
        budget=budget,
        test_metrics=test_metrics,
    )

    eval_dir = ensure_dir(run_dir / "eval")
    per_ckpt_payload = {
        "record": record,
        "test_metrics": dict(test_metrics),
        "budget": dict(budget),
        "checkpoint_meta": ckpt_obj.get("meta", {}) if isinstance(ckpt_obj, dict) else {},
    }
    save_json(eval_dir / f"{args.eval_split}_{ckpt_kind}_metrics.json", per_ckpt_payload)
    if args.save_predictions:
        save_jsonl(eval_dir / f"{args.eval_split}_{ckpt_kind}_predictions.jsonl", predictions)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate curriculum-ft checkpoints on held-out MCQA splits")
    p.add_argument("--dataset", type=str, default="sciq", choices=["piqa", "winogrande", "sciq", "arc_easy"])
    p.add_argument("--input-root", "--input_root", dest="input_root", type=str, default=str(DEFAULT_INPUT_ROOT))
    p.add_argument("--output-root", "--output_root", dest="output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--eval-name", "--eval_name", dest="eval_name", type=str, default="")
    p.add_argument("--run-dir", "--run_dir", dest="run_dir", action="append", default=[], help="Explicit run directory to evaluate. Can be passed multiple times.")
    p.add_argument("--model-key", "--model_key", dest="model_key", type=str, default="")
    p.add_argument("--variant-name", "--variant_name", dest="variant_name", type=str, default="")
    p.add_argument(
        "--variant-name-contains", "--variant_name_contains",
        dest="variant_name_contains", type=str, default="",
        help="Substring filter applied to summary['variant_name'] during auto-discovery.",
    )
    p.add_argument("--baseline-mode", "--baseline_mode", dest="baseline_mode", type=str, default="")
    p.add_argument("--curriculum-mode", "--curriculum_mode", dest="curriculum_mode", type=str, default="")
    p.add_argument("--raw-score-key", "--raw_score_key", dest="raw_score_key", type=str, default="")
    p.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="")
    p.add_argument("--latest-per-variant", "--latest_per_variant", dest="latest_per_variant", action="store_true")
    p.add_argument("--checkpoint-choice", "--checkpoint_choice", dest="checkpoint_choice", type=str, default="both", choices=CHECKPOINT_CHOICE)
    p.add_argument("--eval-split", "--eval_split", dest="eval_split", type=str, default="test", choices=EVAL_SPLIT_CHOICES)
    p.add_argument("--max-examples", "--max_examples", dest="max_examples", type=int, default=0)
    p.add_argument("--model-name-or-path", "--model_name_or_path", dest="model_name_or_path", type=str, default="")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=8)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    p.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=512)
    p.add_argument("--save-predictions", "--save_predictions", dest="save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.dataset = canonical_dataset_name(args.dataset)
    set_seed(int(args.seed))

    splits = prepare_task_splits(dataset_name=args.dataset, seed=int(args.seed))
    eval_examples = maybe_limit_rows(splits[args.eval_split], int(args.max_examples), seed=int(args.seed))

    discovered = discover_run_summaries(args)
    if not discovered:
        raise FileNotFoundError(
            f"No run summaries matched under {args.input_root} "
            f"for dataset={args.dataset}, model_key={args.model_key or '<any>'}, "
            f"baseline_mode={args.baseline_mode or '<any>'}, curriculum_mode={args.curriculum_mode or '<any>'}, "
            f"variant_name_contains={args.variant_name_contains or '<any>'}"
        )

    aggregate_records: List[Dict[str, Any]] = []
    for run_dir, summary in discovered:
        for ckpt_kind, ckpt_path in iter_checkpoint_targets(run_dir, summary, args.checkpoint_choice):
            record = evaluate_run_checkpoint(
                args=args,
                run_dir=run_dir,
                summary=summary,
                ckpt_kind=ckpt_kind,
                ckpt_path=ckpt_path,
                eval_examples=eval_examples,
            )
            aggregate_records.append(record)
            print(
                f"[DONE] variant={record['variant_name']} checkpoint={ckpt_kind} "
                f"test_acc={record['test_accuracy']:.4f} run_dir={run_dir}"
            )

    eval_name = args.eval_name or now_str()
    model_key = args.model_key or "mixed"
    out_dir = ensure_dir(Path(args.output_root).expanduser() / args.dataset / sanitize_name(model_key) / eval_name)
    save_jsonl(out_dir / "aggregate_results.jsonl", aggregate_records)
    write_csv(out_dir / "aggregate_results.csv", [flatten_for_csv("", row) for row in aggregate_records])

    best_by_acc = max(aggregate_records, key=lambda row: float(row.get("test_accuracy") or -1.0))
    summary_payload = {
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "n_runs_evaluated": len(discovered),
        "n_checkpoint_evals": len(aggregate_records),
        "input_root": str(args.input_root),
        "filters": {
            "model_key": args.model_key,
            "variant_name": args.variant_name,
            "variant_name_contains": args.variant_name_contains,
            "baseline_mode": args.baseline_mode,
            "curriculum_mode": args.curriculum_mode,
            "raw_score_key": args.raw_score_key,
            "run_name": args.run_name,
            "latest_per_variant": bool(args.latest_per_variant),
            "checkpoint_choice": args.checkpoint_choice,
        },
        "best_test_accuracy_record": best_by_acc,
        "mean_test_accuracy": avg(row.get("test_accuracy") for row in aggregate_records),
        "mean_test_acc_per_train_hour": avg(row.get("test_acc_per_train_hour") for row in aggregate_records),
        "mean_test_acc_per_million_tokens": avg(row.get("test_acc_per_million_tokens") for row in aggregate_records),
        "results_path": str(out_dir / "aggregate_results.jsonl"),
        "csv_path": str(out_dir / "aggregate_results.csv"),
    }
    save_json(out_dir / "aggregate_summary.json", summary_payload)


if __name__ == "__main__":
    main()
