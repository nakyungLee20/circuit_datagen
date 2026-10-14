import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from applications.filter_data import build_training_variants, load_score_csv
from data_gens.mcqa import (MCQATokenDataset, canonical_dataset_name, collate_batch, eval_accuracy, prepare_task_splits, set_seed,)

SCORE_REQUIRED_MODES = {
    "oris_filter",
    "highc_filter",
    "lowc_filter",
    "bothc_filter",
    "contc_filter",
    "randc_filter",
}
SCORE_COVERAGE_WARN_THRESHOLD = 0.95
EXPERIMENT_MODES = ["full", "rand_filter", "oris_filter", "highc_filter", "lowc_filter", "bothc_filter", "contc_filter", "randc_filter"]
DISPLAY_NAME_MAP = {
    "full": "Full",
    "rand_filter": "Rand_Filter",
    "oris_filter": "OriS_Filter",
    "highc_filter": "HighC_Filter",
    "lowc_filter": "LowC_Filter",
    "bothc_filter": "BothC_Filter",
    "contc_filter": "ContC_Filter",
    "randc_filter": "RandC_Filter",
}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def now_str() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

def parse_modes(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in (raw or "").split(","):
        m = x.strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out

def maybe_subsample(rows: Sequence[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    arr = list(rows)
    if limit <= 0 or len(arr) <= limit:
        return arr
    rng = random.Random(seed)
    idx = list(range(len(arr)))
    rng.shuffle(idx)
    pick = set(idx[:limit])
    return [arr[i] for i in range(len(arr)) if i in pick]

def display_name_for_variant(variant: str) -> str:
    return DISPLAY_NAME_MAP.get(variant, variant)

def maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def maybe_load_completed_summary(*, run_dir: Path, dataset: str, variant_name: str, model_name: str, seed: int,
    expected_n_train: int, expected_n_train_full: int, epochs: int, batch_size: int, eval_batch_size: int, lr: float,
    weight_decay: float, grad_accum: int, warmup_ratio: float, max_grad_norm: float, max_length: int, num_workers: int,
    dtype: str, save_epoch_checkpoints: bool,) -> Optional[Dict[str, Any]]:
    summary_path = run_dir / "run_summary.json"
    ckpt_dir = run_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"
    final_ckpt = ckpt_dir / "final.pt"
    if not summary_path.is_file() or not best_ckpt.is_file() or not final_ckpt.is_file():
        return None

    try:
        summary = load_json(summary_path)
    except Exception as e:
        print(f"[WARN] mode={variant_name}: failed to read existing summary {summary_path}: {e}")
        return None

    mismatches: List[str] = []
    if str(summary.get("dataset", "")) != dataset:
        mismatches.append(f"dataset mismatch ({summary.get('dataset')} != {dataset})")
    if str(summary.get("variant", "")) != variant_name:
        mismatches.append(f"variant mismatch ({summary.get('variant')} != {variant_name})")
    if str(summary.get("model_name", "")) != model_name:
        mismatches.append("model_name mismatch")
    summary_seed = maybe_int(summary.get("seed"))
    if summary_seed != int(seed):
        mismatches.append(f"seed mismatch ({summary_seed} != {int(seed)})")

    sizes = summary.get("sizes", {})
    if not isinstance(sizes, dict):
        mismatches.append("sizes metadata missing")
        sizes = {}
    saved_n_train = maybe_int(sizes.get("n_train"))
    saved_n_train_full = maybe_int(sizes.get("n_train_full"))
    if saved_n_train != int(expected_n_train):
        mismatches.append(f"n_train mismatch ({saved_n_train} != {int(expected_n_train)})")
    if saved_n_train_full != int(expected_n_train_full):
        mismatches.append(f"n_train_full mismatch ({saved_n_train_full} != {int(expected_n_train_full)})")

    expected_cfg = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "eval_batch_size": int(eval_batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "grad_accum": int(grad_accum),
        "warmup_ratio": float(warmup_ratio),
        "max_grad_norm": float(max_grad_norm),
        "max_length": int(max_length),
        "num_workers": int(num_workers),
        "dtype": str(dtype),
        "save_epoch_checkpoints": bool(save_epoch_checkpoints),
    }
    train_config = summary.get("train_config", {})
    if isinstance(train_config, dict) and train_config:
        for key, expected_value in expected_cfg.items():
            saved_value = train_config.get(key)
            if saved_value != expected_value:
                mismatches.append(f"train_config.{key} mismatch ({saved_value} != {expected_value})")
    else:
        epoch_rows = summary.get("metrics", {}).get("epochs", [])
        saved_epochs = len(epoch_rows) if isinstance(epoch_rows, list) else None
        if saved_epochs != int(epochs):
            mismatches.append(f"epochs mismatch ({saved_epochs} != {int(epochs)})")

    recorded_ckpts = summary.get("checkpoints", {})
    if isinstance(recorded_ckpts, dict):
        recorded_best = str(recorded_ckpts.get("best", "")).strip()
        recorded_final = str(recorded_ckpts.get("final", "")).strip()
        if recorded_best and Path(recorded_best) != best_ckpt:
            mismatches.append("recorded best checkpoint path mismatch")
        if recorded_final and Path(recorded_final) != final_ckpt:
            mismatches.append("recorded final checkpoint path mismatch")

    if mismatches:
        print(
            f"[WARN] mode={variant_name}: existing artifacts found in {run_dir}, "
            f"but will retrain because {'; '.join(mismatches)}"
        )
        return None

    return summary

def save_checkpoint(path: Path, model: Any, epoch: int, global_step: int, meta: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "meta": meta,
        },
        str(path),
    )

def train_one_variant(*, variant_name: str, train_examples: Sequence[Dict[str, Any]], val_examples: Sequence[Dict[str, Any]], dataset: str,
    model_name: str, full_train_size: int, run_dir: Path, seed: int, device: str, epochs: int, batch_size: int, eval_batch_size: int, lr: float, weight_decay: float,
    grad_accum: int, warmup_ratio: float, max_grad_norm: float, max_length: int, num_workers: int, dtype: str, wandb_use: bool, wandb_project: str, wandb_entity: str, wandb_group: str, wandb_name: str, wandb_log_every: int, save_epoch_checkpoints: bool,) -> Dict[str, Any]:
    set_seed(int(seed))
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()
    num_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_ds = MCQATokenDataset(train_examples, tokenizer=tokenizer, max_length=int(max_length))
    val_ds = MCQATokenDataset(val_examples, tokenizer=tokenizer, max_length=int(max_length))
    if len(train_ds) == 0:
        raise RuntimeError(f"variant={variant_name}: train dataset is empty after tokenization")
    if len(val_ds) == 0:
        raise RuntimeError("validation dataset is empty after tokenization")

    train_dl = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers), pin_memory=device.startswith("cuda"), collate_fn=lambda b: collate_batch(b, pad_id), drop_last=False,)
    val_dl = DataLoader(val_ds, batch_size=int(eval_batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=device.startswith("cuda"), collate_fn=lambda b: collate_batch(b, pad_id), drop_last=False,)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    updates_per_epoch = max(1, math.ceil(len(train_dl) / max(1, int(grad_accum))))
    total_updates = max(1, int(epochs) * updates_per_epoch)
    warmup_steps = int(total_updates * float(warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates,)

    use_fp16 = device.startswith("cuda") and dtype == "float16"
    use_bf16 = device.startswith("cuda") and dtype == "bfloat16"
    amp_enabled = use_fp16 or use_bf16
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    wb_run = None
    if wandb_use:
        if wandb is None:
            raise ImportError("wandb is not installed, but --use_wandb was set")
        wb_run = wandb.init(
            project=wandb_project,
            entity=(wandb_entity or None),
            group=(wandb_group or None),
            name=wandb_name,
            config={
                "dataset": dataset,
                "variant": variant_name,
                "model_name": model_name,
                "seed": int(seed),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "eval_batch_size": int(eval_batch_size),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_accum": int(grad_accum),
                "warmup_ratio": float(warmup_ratio),
                "max_grad_norm": float(max_grad_norm),
                "max_length": int(max_length),
                "dtype": dtype,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "sample_ratio": float(len(train_ds) / max(1, full_train_size)),
                "run_dir": str(run_dir),
            },
        )

    epoch_metrics: List[Dict[str, Any]] = []
    global_step = 0
    seen_batches = 0
    best_val_acc = float("-inf")
    best_ckpt = ckpt_dir / "best.pt"
    final_ckpt = ckpt_dir / "final.pt"

    total_tokens_once = int(sum(train_ds.lengths))
    t0 = time.time()
    cuda_device = torch.device(device) if device.startswith("cuda") and torch.cuda.is_available() else None
    if cuda_device is not None:
        try:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        except Exception:
            pass

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(epochs)):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        running_items = 0

        for step, batch in enumerate(train_dl, start=1):
            seen_batches += 1
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if amp_enabled else nullcontext()
            with autocast_ctx:
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                raw_loss = out.loss
                loss = raw_loss / max(1, int(grad_accum))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(raw_loss.detach().item()) * int(input_ids.shape[0])
            running_items += int(input_ids.shape[0])
            do_step = (step % max(1, int(grad_accum)) == 0) or (step == len(train_dl))
            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if float(max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if wb_run is not None and (global_step % max(1, int(wandb_log_every)) == 0):
                    wb_run.log(
                        {
                            "train/loss_step": float(raw_loss.detach().item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/global_step": float(global_step),
                            "train/epoch": float(epoch + 1),
                        },
                        step=global_step,
                    )

        val_acc = eval_accuracy(model, val_dl, device=device)
        epoch_sec = float(time.time() - epoch_t0)
        train_loss_epoch = float(running_loss / max(1, running_items))
        ckpt_meta = {
            "dataset": dataset,
            "variant": variant_name,
            "epoch": int(epoch + 1),
            "global_step": int(global_step),
            "val_acc": float(val_acc),
            "train_loss": float(train_loss_epoch),
        }
        if save_epoch_checkpoints:
            epoch_ckpt = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(epoch_ckpt, model, epoch=epoch + 1, global_step=global_step, meta=ckpt_meta)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            save_checkpoint(best_ckpt, model, epoch=epoch + 1, global_step=global_step, meta=ckpt_meta)

        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss_epoch),
            "val_acc": float(val_acc),
            "global_step": int(global_step),
            "epoch_seconds": float(epoch_sec),
        }
        epoch_metrics.append(row)

        if wb_run is not None:
            wb_run.log(
                {
                    "epoch/train_loss": float(train_loss_epoch),
                    "epoch/val_acc": float(val_acc),
                    "epoch/seconds": float(epoch_sec),
                    "epoch/num": float(epoch + 1),
                },
                step=global_step,
            )

    peak_gpu_mem_alloc_bytes = 0
    peak_gpu_mem_reserved_bytes = 0
    if cuda_device is not None:
        try:
            torch.cuda.synchronize(cuda_device)
        except Exception:
            pass
        peak_gpu_mem_alloc_bytes = int(torch.cuda.max_memory_allocated(cuda_device))
        peak_gpu_mem_reserved_bytes = int(torch.cuda.max_memory_reserved(cuda_device))

    total_sec = float(time.time() - t0)
    save_checkpoint(
        final_ckpt,
        model,
        epoch=int(epochs),
        global_step=global_step,
        meta={"dataset": dataset, "variant": variant_name, "final": True},
    )

    final_val_acc = float(epoch_metrics[-1]["val_acc"]) if epoch_metrics else float("nan")
    sample_ratio = float(len(train_ds) / max(1, full_train_size))
    summary = {
        "dataset": dataset,
        "variant": variant_name,
        "model_name": model_name,
        "seed": int(seed),
        "train_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_accum": int(grad_accum),
            "warmup_ratio": float(warmup_ratio),
            "max_grad_norm": float(max_grad_norm),
            "max_length": int(max_length),
            "num_workers": int(num_workers),
            "dtype": dtype,
            "save_epoch_checkpoints": bool(save_epoch_checkpoints),
        },
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "checkpoints": {
            "best": str(best_ckpt),
            "final": str(final_ckpt),
        },
        "sizes": {
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "n_train_full": int(full_train_size),
            "sample_ratio": float(sample_ratio),
        },
        "model": {
            "num_params": int(num_params),
            "trainable_params": int(trainable_params),
            "trainable_ratio": float(trainable_params / max(1, num_params)),
        },
        "efficiency": {
            "train_tokens_once": int(total_tokens_once),
            "train_tokens_seen_total": int(total_tokens_once * max(1, int(epochs))),
            "avg_tokens_per_sample": float(total_tokens_once / max(1, len(train_ds))),
            "global_steps": int(global_step),
            "seen_batches": int(seen_batches),
            "total_train_seconds": float(total_sec),
            "steps_per_second": float(global_step / max(1e-6, total_sec)),
            "tokens_per_second": float((total_tokens_once * max(1, int(epochs))) / max(1e-6, total_sec)),
            "peak_gpu_mem_alloc_bytes": int(peak_gpu_mem_alloc_bytes),
            "peak_gpu_mem_reserved_bytes": int(peak_gpu_mem_reserved_bytes),
            "peak_gpu_mem_alloc_gib": float(peak_gpu_mem_alloc_bytes / float(1024**3)),
            "peak_gpu_mem_reserved_gib": float(peak_gpu_mem_reserved_bytes / float(1024**3)),
        },
        "metrics": {
            "best_val_acc": float(best_val_acc if best_val_acc > -1e9 else float("nan")),
            "final_val_acc": float(final_val_acc),
            "epochs": epoch_metrics,
        },
    }
    save_json(summary, run_dir / "run_summary.json")

    if wb_run is not None:
        wb_run.log(
            {
                "run/best_val_acc": float(summary["metrics"]["best_val_acc"]),
                "run/final_val_acc": float(summary["metrics"]["final_val_acc"]),
                "run/train_seconds": float(total_sec),
                "run/sample_ratio": float(sample_ratio),
                "run/num_params": int(num_params),
                "run/trainable_params": int(trainable_params),
                "run/peak_gpu_mem_alloc_gib": float(summary["efficiency"]["peak_gpu_mem_alloc_gib"]),
                "run/peak_gpu_mem_reserved_gib": float(summary["efficiency"]["peak_gpu_mem_reserved_gib"]),
            },
            step=global_step,
        )
        wb_run.finish()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Circuit-aware data selection finetuning runner")
    ap.add_argument("--dataset", type=str, required=True, choices=["piqa", "winogrande", "sciq", "arc_easy"])
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output_root", type=str, default=str(_ROOT / "checkpoints" / "app_runs"))
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--score_csv", type=str, default="", help="sample_scores.csv path")
    ap.add_argument("--score_key", type=str, default="aum", help="Original score key for OriS filter (e.g., aum, grad_align)")
    ap.add_argument("--auto_build_score_csv", action="store_true", help="If score-based modes are requested and --score_csv is empty, build train-split circuit score CSV automatically.")
    ap.add_argument("--auto_build_score_csv_force", action="store_true", help="Force re-computation when auto-building score CSV.")
    ap.add_argument("--circuits_root", type=str, default=str(_ROOT / "circuit_results"))
    ap.add_argument("--buckets_root", type=str, default=str(_ROOT / "data_gens" / "buckets"))
    ap.add_argument("--corrupt_type", type=str, default="gaussian")
    ap.add_argument("--score_model_name", type=str, default="", help="Optional override model name for score CSV generation.")
    ap.add_argument("--score_device", type=str, default="", help="Optional device override for score CSV generation. Empty means --device.")
    ap.add_argument("--score_dtype", type=str, default="", choices=["", "float16", "bfloat16", "float32"], help="Optional dtype override for score CSV generation. Empty means --dtype.")
    ap.add_argument("--score_ig_steps", type=int, default=0, help="0 means auto from circuit run metadata.")
    ap.add_argument("--score_incorrect_agg", type=str, default="auto", choices=["auto", "max", "mean", "min"])
    ap.add_argument("--score_noise_sigma", type=float, default=0.1)
    ap.add_argument("--score_topk_edges", type=int, default=500)
    ap.add_argument("--score_topk_rank_signed", dest="score_topk_rank_abs", action="store_false")
    ap.add_argument("--score_sum_abs", action="store_true")
    ap.add_argument(
        "--score_direction_aware",
        action="store_true",
        help="Use direction-aware circuit score: sum(attribution * sign(circuit edge score)) for high/low circuits.",
    )
    ap.add_argument(
        "--score_weighted_direction_aware",
        action="store_true",
        help="Use weighted direction-aware score: sum(attribution * circuit edge score) for high/low circuits.",
    )
    ap.add_argument(
        "--score_random_baseline_trials",
        type=int,
        default=100,
        help="0 disables randc score. Any value >0 averages raw signed sums over that many fixed random-edge subsets (independent from high/low score_mode).",
    )
    ap.add_argument("--score_resume", dest="score_resume", action="store_true")
    ap.add_argument("--no_score_resume", dest="score_resume", action="store_false")
    ap.add_argument("--score_save_every", type=int, default=100)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--top_q", type=float, default=0.25)
    ap.add_argument(
        "--contc_abs_diff",
        type=int,
        default=1,
        choices=[0, 1],
        help="ContC ranking score: 1=abs(high-low), 0=(high-low) raw signed difference.",
    )
    ap.add_argument("--rand_filter_policy", type=str, default="match_scored_q", choices=["match_scored_q", "full_q"], help="match_scored_q: rand_filter size matches score-based selected count; full_q: rand_filter uses full train top_q count.",)
    ap.add_argument("--modes", type=str, default=",".join(EXPERIMENT_MODES), help="Comma-separated variants to train.",)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=2.0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_train_examples", type=int, default=0, help="0 means all")
    ap.add_argument("--save_epoch_checkpoints", dest="save_epoch_checkpoints", action="store_true", help="Save per-epoch checkpoints (epoch_XXX.pt)")
    ap.add_argument("--no_save_epoch_checkpoints", dest="save_epoch_checkpoints", action="store_false", help="Only save best.pt and final.pt")
    ap.add_argument("--skip_completed_modes", dest="skip_completed_modes", action="store_true", help="Skip a mode when run_summary.json and best/final checkpoints already exist and match the requested config.")
    ap.add_argument("--no_skip_completed_modes", dest="skip_completed_modes", action="store_false", help="Retrain requested modes even if matching completed artifacts already exist.")
    ap.add_argument("--use_wandb", action="store_false")
    ap.add_argument("--wandb_project", type=str, default="data_mi")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_log_every", type=int, default=20)
    ap.set_defaults(score_topk_rank_abs=True)
    ap.set_defaults(score_resume=True)
    ap.set_defaults(save_epoch_checkpoints=True)
    ap.set_defaults(skip_completed_modes=True)
    return ap

def main() -> None:
    args = build_argparser().parse_args()
    set_seed(int(args.seed))

    dataset = canonical_dataset_name(args.dataset)
    requested_modes = parse_modes(args.modes)
    if not requested_modes:
        raise ValueError("No training modes specified")
    unknown_modes = [m for m in requested_modes if m not in EXPERIMENT_MODES]
    if unknown_modes:
        raise ValueError(f"Unsupported modes: {unknown_modes}. allowed={EXPERIMENT_MODES}")

    requested_score_modes = [m for m in requested_modes if m in SCORE_REQUIRED_MODES]
    splits = prepare_task_splits(dataset_name=dataset, seed=int(args.seed), train_frac=float(args.train_frac))

    run_name = args.run_name or f"{dataset}_{now_str()}"
    exp_dir = Path(args.output_root) / dataset / run_name
    ensure_dir(exp_dir)

    score_csv_used = str(args.score_csv or "").strip()
    auto_score_meta: Optional[Dict[str, Any]] = None
    if requested_score_modes and (not score_csv_used):
        if not bool(args.auto_build_score_csv):
            raise ValueError(
                "--score_csv is required for score-based modes (oris/highc/lowc/bothc/contc/randc). "
                "Or enable --auto_build_score_csv."
            )
        from applications.circuit_scores import build_full_split_score_csv

        auto_csv = exp_dir / f"train_sample_scores_{dataset}_{args.score_key}.csv"
        score_device = str(args.score_device).strip() or str(args.device)
        score_dtype = str(args.score_dtype).strip() or str(args.dtype)
        auto_score_meta = build_full_split_score_csv(
            dataset=dataset,
            score_key=args.score_key,
            split_examples=splits["train"],
            out_csv_path=str(auto_csv),
            circuits_root=str(args.circuits_root),
            buckets_root=str(args.buckets_root),
            corrupt_type=str(args.corrupt_type),
            topk_edges=int(args.score_topk_edges),
            topk_rank_abs=bool(args.score_topk_rank_abs),
            sum_abs=bool(args.score_sum_abs),
            direction_aware=bool(args.score_direction_aware),
            weighted_direction_aware=bool(args.score_weighted_direction_aware),
            random_baseline_trials=int(args.score_random_baseline_trials),
            seed=int(args.seed),
            model_name_override=str(args.score_model_name),
            device=score_device,
            dtype=score_dtype,
            ig_steps_override=int(args.score_ig_steps),
            incorrect_agg_override=str(args.score_incorrect_agg),
            noise_sigma=float(args.score_noise_sigma),
            split_name="train",
            resume=bool(args.score_resume),
            force_recompute=bool(args.auto_build_score_csv_force),
            save_every=int(args.score_save_every),
            quiet=False,
        )
        score_csv_used = str(auto_csv)
        print(f"[INFO] auto-built score_csv: {score_csv_used}")

    score_map = load_score_csv(score_csv_used, original_score_key=args.score_key) if score_csv_used else None

    variants, filter_meta = build_training_variants(
        train_examples=splits["train"],
        score_map=score_map,
        top_q=float(args.top_q),
        seed=int(args.seed),
        original_score_key=args.score_key,
        rand_filter_policy=args.rand_filter_policy,
        contc_abs_diff=bool(args.contc_abs_diff),
    )
    if requested_score_modes and "rand_filter" in requested_modes:
        n_train_full = int(filter_meta.get("full_train_size", len(splits["train"])))
        n_select_full = int(filter_meta.get("n_select_full", 0))
        n_select_scored = int(filter_meta.get("n_select_scored", 0))
        n_select_rand = int(filter_meta.get("n_select_rand", len(variants.get("rand_filter", []))))
        rand_policy = str(filter_meta.get("rand_filter_policy", args.rand_filter_policy))
        print(
            "[INFO] rand_filter sizing "
            f"(policy={rand_policy}): n_select_rand={n_select_rand}, "
            f"n_select_scored={n_select_scored}, n_select_full={n_select_full}, "
            f"n_train_full={n_train_full}"
        )
    if requested_score_modes:
        cov = filter_meta.get("score_coverage", {})
        coverage_ratio = float(cov.get("coverage_ratio", 0.0))
        n_train_full = int(filter_meta.get("full_train_size", len(splits["train"])))
        n_with_scores = int(cov.get("n_with_scores", 0))
        n_missing_scores = int(cov.get("n_missing_scores", max(0, n_train_full - n_with_scores)))
        n_select_scored = int(filter_meta.get("n_select_scored", 0))
        top_q_used = float(filter_meta.get("top_q", float(args.top_q)))

        if coverage_ratio < SCORE_COVERAGE_WARN_THRESHOLD:
            usable_full_pct = 100.0 * float(n_with_scores / max(1, n_train_full))
            selected_full_pct = 100.0 * float(n_select_scored / max(1, n_train_full))
            selected_scored_pct = 100.0 * float(n_select_scored / max(1, n_with_scores))
            warn_msg = (
                f"Low score coverage detected: coverage_ratio={coverage_ratio:.4f} "
                f"(< {SCORE_COVERAGE_WARN_THRESHOLD:.2f})."
            )
            print(f"[WARN] {warn_msg}")
            print(
                f"[WARN] score-matched train samples: {n_with_scores}/{n_train_full} "
                f"({usable_full_pct:.2f}%), missing={n_missing_scores}"
            )
            print(
                f"[WARN] score-based top-q selection uses {n_select_scored} samples "
                f"(top_q={top_q_used:.4f}), which is {selected_full_pct:.2f}% of full train "
                f"and {selected_scored_pct:.2f}% of scored subset."
            )
            for mode in requested_score_modes:
                n_mode = int(len(variants.get(mode, [])))
                mode_full_pct = 100.0 * float(n_mode / max(1, n_train_full))
                print(
                    f"[WARN] planned train size for {mode}: {n_mode}/{n_train_full} "
                    f"({mode_full_pct:.2f}% of full train)"
                )
            filter_meta.setdefault("notes", []).append(warn_msg)

    modes: List[str] = []
    skipped_modes: Dict[str, str] = {}
    for m in requested_modes:
        if m not in variants:
            skipped_modes[m] = "mode is not available from current filtering setup"
            continue
        if len(variants[m]) == 0:
            skipped_modes[m] = "variant has zero samples"
            continue
        modes.append(m)
    if not modes:
        raise RuntimeError(f"No trainable modes left. skipped={skipped_modes}")

    save_jsonl_rows(splits["val"], exp_dir / "val.jsonl")
    save_jsonl_rows(splits["test"], exp_dir / "test.jsonl")
    for m in modes:
        save_jsonl_rows(variants[m], exp_dir / f"train_{m}.jsonl")

    run_summaries: List[Dict[str, Any]] = []
    reused_run_summaries: List[Dict[str, Any]] = []
    for mode in modes:
        train_rows = maybe_subsample(variants[mode], limit=int(args.max_train_examples), seed=int(args.seed),)
        mode_dir = exp_dir / mode
        if bool(args.skip_completed_modes):
            existing_summary = maybe_load_completed_summary(
                run_dir=mode_dir,
                dataset=dataset,
                variant_name=mode,
                model_name=args.model_name,
                seed=int(args.seed),
                expected_n_train=len(train_rows),
                expected_n_train_full=len(splits["train"]),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                eval_batch_size=int(args.eval_batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                grad_accum=int(args.grad_accum),
                warmup_ratio=float(args.warmup_ratio),
                max_grad_norm=float(args.max_grad_norm),
                max_length=int(args.max_length),
                num_workers=int(args.num_workers),
                dtype=args.dtype,
                save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
            )
            if existing_summary is not None:
                existing_summary["trained_with_examples"] = int(len(train_rows))
                reused_run_summaries.append(existing_summary)
                skipped_modes[mode] = "already completed; reused existing run_summary.json and best/final checkpoints"
                print(f"[INFO] mode={mode}: skipping completed run in {mode_dir}")
                continue
        wb_group = args.wandb_group or run_name
        wb_name = f"{run_name}_{mode}"
        summary = train_one_variant(
            variant_name=mode,
            train_examples=train_rows,
            val_examples=splits["val"],
            dataset=dataset,
            model_name=args.model_name,
            full_train_size=len(splits["train"]),
            run_dir=mode_dir,
            seed=int(args.seed),
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            eval_batch_size=int(args.eval_batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_accum=int(args.grad_accum),
            warmup_ratio=float(args.warmup_ratio),
            max_grad_norm=float(args.max_grad_norm),
            max_length=int(args.max_length),
            num_workers=int(args.num_workers),
            dtype=args.dtype,
            wandb_use=bool(args.use_wandb),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=wb_group,
            wandb_name=wb_name,
            wandb_log_every=int(args.wandb_log_every),
            save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        )
        summary["trained_with_examples"] = int(len(train_rows))
        run_summaries.append(summary)

    all_run_summaries: List[Dict[str, Any]] = []
    summary_by_mode: Dict[str, Dict[str, Any]] = {}
    for summary in reused_run_summaries:
        summary_by_mode[str(summary.get("variant", ""))] = summary
    for summary in run_summaries:
        summary_by_mode[str(summary.get("variant", ""))] = summary
    for mode in modes:
        if mode in summary_by_mode:
            all_run_summaries.append(summary_by_mode[mode])

    csv_rows: List[Dict[str, Any]] = []
    for s in all_run_summaries:
        n_train = int(s["sizes"]["n_train"])
        n_train_full = int(s["sizes"]["n_train_full"])
        train_seconds = float(s["efficiency"]["total_train_seconds"])
        train_hours = float(train_seconds / 3600.0)
        csv_rows.append(
            {
                "dataset_type": display_name_for_variant(s["variant"]),
                "variant": s["variant"],
                "n_train": n_train,
                "n_train_full": n_train_full,
                "sample_ratio": s["sizes"]["sample_ratio"],
                "best_val_acc": s["metrics"]["best_val_acc"],
                "final_val_acc": s["metrics"]["final_val_acc"],
                "train_seconds": train_seconds,
                "train_minutes": float(train_seconds / 60.0),
                "train_hours": train_hours,
                "steps": s["efficiency"]["global_steps"],
                "steps_per_second": s["efficiency"]["steps_per_second"],
                "tokens_seen": s["efficiency"]["train_tokens_seen_total"],
                "tokens_per_second": s["efficiency"]["tokens_per_second"],
                "avg_tokens_per_sample": s["efficiency"]["avg_tokens_per_sample"],
                "num_params": s["model"]["num_params"],
                "trainable_params": s["model"]["trainable_params"],
                "trainable_ratio": s["model"]["trainable_ratio"],
                "peak_gpu_mem_alloc_bytes": s["efficiency"]["peak_gpu_mem_alloc_bytes"],
                "peak_gpu_mem_reserved_bytes": s["efficiency"]["peak_gpu_mem_reserved_bytes"],
                "peak_gpu_mem_alloc_gib": s["efficiency"]["peak_gpu_mem_alloc_gib"],
                "peak_gpu_mem_reserved_gib": s["efficiency"]["peak_gpu_mem_reserved_gib"],
                "val_acc_per_train_hour": float(s["metrics"]["best_val_acc"] / max(1e-6, train_hours)),
                "best_ckpt": s["checkpoints"]["best"],
                "final_ckpt": s["checkpoints"]["final"],
            }
        )

    summary_csv = exp_dir / "train_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["variant"])
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    manifest = {
        "dataset": dataset,
        "model_name": args.model_name,
        "run_name": run_name,
        "exp_dir": str(exp_dir),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "top_q": float(args.top_q),
        "score_key": args.score_key,
        "requested_modes": requested_modes,
        "trained_modes": [s["variant"] for s in run_summaries],
        "reused_completed_modes": [s["variant"] for s in reused_run_summaries],
        "reported_modes": [s["variant"] for s in all_run_summaries],
        "skipped_modes": skipped_modes,
        "score_csv": score_csv_used,
        "score_csv_input_arg": args.score_csv,
        "auto_score_meta": auto_score_meta,
        "split_sizes": {
            "train_full": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "filter_meta": filter_meta,
        "args": vars(args),
        "run_summaries": all_run_summaries,
        "artifacts": {
            "train_summary_csv": str(summary_csv),
            "val_jsonl": str(exp_dir / "val.jsonl"),
            "test_jsonl": str(exp_dir / "test.jsonl"),
        },
    }
    manifest_path = exp_dir / "experiment_manifest.json"
    save_json(manifest, manifest_path)

    print("[OK] finetuning finished")
    print(f"  - experiment_manifest: {manifest_path}")
    print(f"  - train_summary_csv: {summary_csv}")
    print(f"  - trained_modes: {manifest['trained_modes']}")
    if manifest["reused_completed_modes"]:
        print(f"  - reused_completed_modes: {manifest['reused_completed_modes']}")
    if skipped_modes:
        print(f"  - skipped_modes: {skipped_modes}")
    for s in all_run_summaries:
        print(
            f"  - {s['variant']}: best_val_acc={s['metrics']['best_val_acc']:.4f} | "
            f"sample_ratio={s['sizes']['sample_ratio']:.3f} | best_ckpt={s['checkpoints']['best']}"
        )


if __name__ == "__main__":
    main()
