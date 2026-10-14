import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_gens.mcqa import (MCQATokenDataset, canonical_dataset_name, collate_batch, eval_accuracy, load_checkpoint_state_dict, prepare_task_splits, set_seed,)

DISPLAY_NAME_MAP = {
    "full": "Full",
    "rand_filter": "Rand_Filter",
    "oris_filter": "OriS_Filter",
    "highc_filter": "HighC_Filter",
    "lowc_filter": "LowC_Filter",
    "contc_filter": "ContC_Filter",
    "randc_filter": "RandC_Filter",
}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]

def display_name_for_variant(variant: str) -> str:
    return DISPLAY_NAME_MAP.get(variant, variant)

def resolve_checkpoint_path(
    run_summary: Dict[str, Any],
    checkpoint_type: str,
    *,
    exp_dirs: Optional[Sequence[Path]] = None,
    variant: str = "",
) -> Path:
    ckpts = run_summary.get("checkpoints", {})
    candidates: List[Path] = []
    seen = set()

    def add_candidate(raw: Any) -> None:
        if not raw:
            return
        p = Path(str(raw))
        k = str(p)
        if k and k not in seen:
            seen.add(k)
            candidates.append(p)

    if checkpoint_type in {"best", "final"}:
        filename = f"{checkpoint_type}.pt"
        add_candidate(ckpts.get(checkpoint_type, ""))
    elif checkpoint_type.startswith("epoch_") and checkpoint_type.endswith(".pt"):
        filename = checkpoint_type
    else:
        raise ValueError(f"Unsupported checkpoint_type={checkpoint_type}. Use best/final or epoch_XXX.pt")

    ckpt_dir_raw = run_summary.get("checkpoint_dir", "")
    run_dir_raw = run_summary.get("run_dir", "")
    add_candidate(Path(str(ckpt_dir_raw)) / filename if ckpt_dir_raw else "")
    add_candidate(Path(str(run_dir_raw)) / "checkpoints" / filename if run_dir_raw else "")

    variant_name = (variant or str(run_summary.get("variant", ""))).strip()
    for base_dir in (exp_dirs or []):
        if base_dir is None:
            continue
        exp_dir = Path(base_dir)
        if variant_name:
            add_candidate(exp_dir / variant_name / "checkpoints" / filename)
        if run_dir_raw:
            add_candidate(exp_dir / Path(str(run_dir_raw)).name / "checkpoints" / filename)
        if ckpt_dir_raw:
            ckpt_dir = Path(str(ckpt_dir_raw))
            add_candidate(exp_dir / ckpt_dir.parent.name / ckpt_dir.name / filename)

    for p in candidates:
        if p.is_file():
            return p

    if checkpoint_type in {"best", "final"} and not ckpts.get(checkpoint_type, ""):
        raise FileNotFoundError(f"checkpoint {checkpoint_type} path missing in run_summary")
    attempted = "\n".join(f"  - {p}" for p in candidates) if candidates else "  - (no candidates)"
    raise FileNotFoundError(
        f"checkpoint not found for checkpoint_type={checkpoint_type}. attempted:\n{attempted}"
    )

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate trained checkpoints on downstream test split")
    ap.add_argument("--manifest", type=str, required=True, help="Path to experiment_manifest.json from filter_finetune.py")
    ap.add_argument("--checkpoint_type", type=str, default="best", help="best | final | epoch_XXX.pt")
    ap.add_argument("--variants", type=str, default="", help="Optional comma-separated subset of variants")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_wandb", action="store_false")
    ap.add_argument("--wandb_project", type=str, default="data_mi")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_name", type=str, default="")
    ap.add_argument("--out_json", type=str, default="")
    return ap

def main() -> None:
    args = build_argparser().parse_args()
    set_seed(int(args.seed))

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset = canonical_dataset_name(str(manifest.get("dataset", "")))
    model_name = str(manifest.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("model_name missing in manifest")

    requested_variants = set(parse_csv_list(args.variants))
    run_summaries: List[Dict[str, Any]] = list(manifest.get("run_summaries", []))
    if requested_variants:
        run_summaries = [r for r in run_summaries if str(r.get("variant", "")) in requested_variants]
    if not run_summaries:
        raise RuntimeError("No run_summaries to evaluate")
    manifest_parent_dir = manifest_path.parent
    manifest_exp_dir_raw = str(manifest.get("exp_dir", "")).strip()
    exp_dirs: List[Path] = [manifest_parent_dir]
    if manifest_exp_dir_raw:
        manifest_exp_dir = Path(manifest_exp_dir_raw)
        if str(manifest_exp_dir) != str(manifest_parent_dir):
            exp_dirs.append(manifest_exp_dir)

    splits = prepare_task_splits(dataset_name=dataset, seed=int(manifest.get("seed", args.seed)), train_frac=float(manifest.get("train_frac", 0.8)))
    test_examples: Sequence[Dict[str, Any]] = splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    test_ds = MCQATokenDataset(test_examples, tokenizer=tokenizer, max_length=int(args.max_length))
    if len(test_ds) == 0:
        raise RuntimeError("test dataset is empty after tokenization")
    test_dl = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda b: collate_batch(b, pad_id),
        drop_last=False,
    )

    wb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --use_wandb was set")
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=(args.wandb_name or f"inference_{manifest.get('run_name', 'run')}_{args.checkpoint_type}"),
            config={
                "manifest": str(manifest_path),
                "dataset": dataset,
                "model_name": model_name,
                "checkpoint_type": args.checkpoint_type,
                "n_test": int(len(test_ds)),
            },
        )

    results: List[Dict[str, Any]] = []
    for i, run in enumerate(run_summaries):
        variant = str(run.get("variant", f"run_{i}"))
        ckpt_path = resolve_checkpoint_path(run, args.checkpoint_type, exp_dirs=exp_dirs, variant=variant)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        model = AutoModelForCausalLM.from_pretrained(model_name)
        state_dict, ckpt_obj = load_checkpoint_state_dict(str(ckpt_path), map_location="cpu")
        load_res = model.load_state_dict(state_dict, strict=False)
        model.to(args.device)

        t0 = time.time()
        test_acc = eval_accuracy(model, test_dl, device=args.device)
        eval_sec = float(time.time() - t0)
        train_seconds = float(run.get("efficiency", {}).get("total_train_seconds", float("nan")))
        train_hours = float(train_seconds / 3600.0) if train_seconds == train_seconds else float("nan")
        row = {
            "dataset_type": display_name_for_variant(variant),
            "variant": variant,
            "checkpoint_type": args.checkpoint_type,
            "checkpoint_path": str(ckpt_path),
            "test_acc": float(test_acc),
            "eval_seconds": float(eval_sec),
            "n_test": int(len(test_ds)),
            "n_train": int(run.get("sizes", {}).get("n_train", -1)),
            "n_train_full": int(run.get("sizes", {}).get("n_train_full", -1)),
            "missing_keys": int(len(getattr(load_res, "missing_keys", []))),
            "unexpected_keys": int(len(getattr(load_res, "unexpected_keys", []))),
            "train_sample_ratio": float(run.get("sizes", {}).get("sample_ratio", float("nan"))),
            "best_val_acc": float(run.get("metrics", {}).get("best_val_acc", float("nan"))),
            "final_val_acc": float(run.get("metrics", {}).get("final_val_acc", float("nan"))),
            "train_seconds": train_seconds,
            "train_hours": train_hours,
            "train_global_steps": int(run.get("efficiency", {}).get("global_steps", -1)),
            "train_tokens_seen_total": int(run.get("efficiency", {}).get("train_tokens_seen_total", -1)),
            "train_tokens_per_second": float(run.get("efficiency", {}).get("tokens_per_second", float("nan"))),
            "train_steps_per_second": float(run.get("efficiency", {}).get("steps_per_second", float("nan"))),
        }
        if train_hours == train_hours:
            row["test_acc_per_train_hour"] = float(test_acc / max(1e-6, train_hours))
        if row["n_train"] > 0:
            row["test_acc_per_1k_train_samples"] = float(test_acc / (row["n_train"] / 1000.0))
        if isinstance(ckpt_obj, dict):
            row["checkpoint_epoch"] = int(ckpt_obj.get("epoch", -1)) if "epoch" in ckpt_obj else -1
            row["checkpoint_global_step"] = int(ckpt_obj.get("global_step", -1)) if "global_step" in ckpt_obj else -1

        results.append(row)

        if wb_run is not None:
            wb_run.log({f"eval/{variant}_test_acc": float(test_acc), f"eval/{variant}_seconds": float(eval_sec)}, step=i + 1)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    default_out_dir = manifest_parent_dir
    if len(exp_dirs) > 1 and exp_dirs[1].is_dir():
        default_out_dir = exp_dirs[1]
    out_json = Path(args.out_json) if args.out_json else default_out_dir / f"inference_report_{args.checkpoint_type}.json"
    out_csv = out_json.with_suffix(".csv")
    report = {
        "manifest": str(manifest_path),
        "dataset": dataset,
        "score_key": manifest.get("score_key", ""),
        "model_name": model_name,
        "checkpoint_type": args.checkpoint_type,
        "n_test": int(len(test_ds)),
        "results": results,
    }
    save_json(report, out_json)

    ensure_dir(out_csv.parent)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["variant"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    if wb_run is not None:
        for r in results:
            wb_run.log({"summary/test_acc": float(r["test_acc"]), "summary/variant": r["variant"]})
        wb_run.finish()
    print("[OK] inference finished")
    print(f"  - report_json: {out_json}")
    print(f"  - report_csv: {out_csv}")
    for r in results:
        print(f"  - {r['variant']}: test_acc={r['test_acc']:.4f} ({r['checkpoint_type']})")


if __name__ == "__main__":
    main()
