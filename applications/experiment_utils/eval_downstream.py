"""Standalone downstream evaluator producing the Table-3 metric set.

The base trainer (curriculum_ft.py) only logs val accuracy — NOT ECE / NLL,
and NOT a held-out test set or OOD (ARC-Easy). This script fills that gap
WITHOUT modifying the trainer:

  * loads a trained checkpoint (best.pt / final.pt)
  * evaluates on in-domain SciQ test split      → acc, ece, nll
  * evaluates on OOD ARC-Easy test split        → acc, ece, nll
  * writes eval_metrics.json next to run_summary.json

The aggregator (applications/experiment_utils/aggregate.py) already looks for these keys:
  test_accuracy, ece, nll, test_accuracy_ood, ece_ood, nll_ood

Usage:
  python -m applications.experiment_utils.eval_downstream --run-dir <dir_with_checkpoints/>
  python -m applications.experiment_utils.eval_downstream --scan-root <output_root>   # eval every run under it

ECE = expected calibration error (15 equal-width bins on max-prob).
NLL = mean −log p(gold) over the option distribution.
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
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_gens import mcqa  # type: ignore  # noqa: E402
from data_gens.mcqa import (  # type: ignore
    MCQATokenDataset,
    collate_batch,
    get_option_logits,
    prepare_task_splits,
)

PLAIN_FORMATTER = mcqa.format_mcqa_prompt
from applications.experiment_utils.chat_prompt import format_chat_mcqa_prompt  # noqa: E402


def _dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


EVAL_METRIC_KEYS = (
    "test_accuracy", "ece", "nll", "test_accuracy_ood", "ece_ood", "nll_ood"
)


def valid_existing_eval(path: Path, expected_prompt_format: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("prompt_format") == expected_prompt_format and all(
            isinstance(payload.get(key), (int, float))
            and math.isfinite(float(payload[key]))
            for key in EVAL_METRIC_KEYS
        )
    except Exception:
        return False


@torch.no_grad()
def eval_split(model, tokenizer, examples, device, *, batch_size=8, max_length=256, n_bins=15) -> Dict[str, float]:
    ds = MCQATokenDataset(examples, tokenizer=tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_id))
    model.eval()

    correct = 0
    total = 0
    nll_sum = 0.0
    confidences: List[float] = []
    accuracies: List[int] = []

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        for idx in range(int(logits.shape[0])):
            pos = int(pred_pos[idx].item())
            label_space = list(batch["label_space"][idx])
            label_token_ids = batch["label_token_ids"][idx]
            gold_label = str(batch["gold_label"][idx])
            option_logits = get_option_logits(logits[idx, pos, :], label_space, label_token_ids)
            probs = torch.softmax(option_logits.float(), dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            gold_idx = label_space.index(gold_label)
            is_correct = int(pred_idx == gold_idx)
            correct += is_correct
            total += 1
            p_gold = float(probs[gold_idx].clamp_min(1e-12).item())
            nll_sum += -torch.log(torch.tensor(p_gold)).item()
            confidences.append(float(probs[pred_idx].item()))
            accuracies.append(is_correct)

    acc = correct / max(1, total)
    nll = nll_sum / max(1, total)
    ece = _expected_calibration_error(confidences, accuracies, n_bins=n_bins)
    return {"accuracy": acc, "ece": ece, "nll": nll, "n_eval": total}


def _expected_calibration_error(confidences: List[float], accuracies: List[int], n_bins: int = 15) -> float:
    if not confidences:
        return float("nan")
    import numpy as np
    conf = np.asarray(confidences)
    acc = np.asarray(accuracies, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = acc[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def find_checkpoint(run_dir: Path, prefer: str = "best", *, strict: bool = False) -> Optional[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if strict:
        # No silent fallback: a best/final comparison is meaningless if some rows
        # quietly fall back to the other checkpoint.
        p = ckpt_dir / f"{prefer}.pt"
        return p if p.exists() else None
    for name in ([f"{prefer}.pt", "best.pt", "final.pt"] if prefer == "best" else [f"{prefer}.pt", "final.pt", "best.pt"]):
        p = ckpt_dir / name
        if p.exists():
            return p
    return None


def load_model_for_run(run_dir: Path, device: str, dtype: str, *, prefer: str = "best",
                       strict_checkpoint: bool = False):
    """Read run_summary.json to get the base model id, then load + apply the checkpoint."""
    summary_path = run_dir / "run_summary.json"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt_format = "plain_mcqa"
    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        model_id = obj.get("model_name_or_path") or obj.get("model", {}).get("model_name_or_path") or model_id
        prompt_format = obj.get("prompt_format") or prompt_format
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_dtype(dtype), trust_remote_code=True)
    ckpt = find_checkpoint(run_dir, prefer, strict=strict_checkpoint)
    if ckpt is None:
        raise FileNotFoundError(f"no {prefer}.pt checkpoint under {run_dir}/checkpoints")
    state = torch.load(str(ckpt), map_location="cpu")
    sd = state.get("model_state_dict", state)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint {ckpt} is not an exact state dict for {model_id}: {exc}"
        ) from exc
    model.to(device).eval()
    return model, tokenizer, model_id, str(ckpt), prompt_format


def eval_one_run(run_dir: Path, *, device="cuda", dtype="bfloat16", batch_size=8, max_length=256,
                 indomain="sciq", ood="arc_easy", seed=42, max_test=0,
                 prefer="best", strict_checkpoint=False, out_name="eval_metrics.json") -> Dict:
    model, tokenizer, model_id, ckpt, prompt_format = load_model_for_run(
        run_dir, device, dtype, prefer=prefer, strict_checkpoint=strict_checkpoint
    )
    if prompt_format == "hf_chat_template":
        mcqa.format_mcqa_prompt = format_chat_mcqa_prompt
    elif prompt_format == "plain_mcqa":
        mcqa.format_mcqa_prompt = PLAIN_FORMATTER
    else:
        raise ValueError(f"unsupported prompt_format={prompt_format!r} in {run_dir}")

    in_splits = prepare_task_splits(dataset_name=indomain, seed=seed)
    in_test = in_splits["test"]
    ood_splits = prepare_task_splits(dataset_name=ood, seed=seed)
    ood_test = ood_splits["test"]
    if max_test:
        in_test = in_test[:max_test]
        ood_test = ood_test[:max_test]

    print(f"[eval] {run_dir.name}: model={model_id} prompt_format={prompt_format} ckpt={Path(ckpt).name} "
          f"| in-domain {indomain} n={len(in_test)} | ood {ood} n={len(ood_test)}")
    in_metrics = eval_split(model, tokenizer, in_test, device, batch_size=batch_size, max_length=max_length)
    ood_metrics = eval_split(model, tokenizer, ood_test, device, batch_size=batch_size, max_length=max_length)

    out = {
        "run_dir": str(run_dir),
        "model_name_or_path": model_id,
        "checkpoint": ckpt,
        "prompt_format": prompt_format,
        "indomain_dataset": indomain,
        "ood_dataset": ood,
        # keys the aggregator reads:
        "test_accuracy": in_metrics["accuracy"],
        "ece": in_metrics["ece"],
        "nll": in_metrics["nll"],
        "test_accuracy_ood": ood_metrics["accuracy"],
        "ece_ood": ood_metrics["ece"],
        "nll_ood": ood_metrics["nll"],
        "n_in": in_metrics["n_eval"],
        "n_ood": ood_metrics["n_eval"],
    }
    (run_dir / out_name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] {run_dir.name}: acc={out['test_accuracy']:.4f} ece={out['ece']:.4f} nll={out['nll']:.4f} "
          f"| ood acc={out['test_accuracy_ood']:.4f} ece={out['ece_ood']:.4f} nll={out['nll_ood']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="a single run dir containing checkpoints/")
    ap.add_argument("--scan-root", default="", help="evaluate every run_summary.json under this root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--indomain", default="sciq")
    ap.add_argument("--ood", default="arc_easy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=0, help="cap #test rows (smoke test)")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have eval_metrics.json")
    ap.add_argument("--prefer", default="best", choices=["best", "final"],
                    help="which checkpoint to evaluate (default: best-val)")
    ap.add_argument("--strict-checkpoint", action="store_true",
                    help="fail instead of falling back to the other checkpoint when --prefer is missing")
    ap.add_argument("--out-name", default="",
                    help="output filename next to run_summary.json "
                         "(default: eval_metrics.json for best, eval_metrics_final.json for final)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="with --shard-count, evaluate only run dirs where i %% count == index")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="number of parallel workers splitting the same --scan-root")
    args = ap.parse_args()

    out_name = args.out_name or (
        "eval_metrics.json" if args.prefer == "best" else f"eval_metrics_{args.prefer}.json"
    )

    run_dirs: List[Path] = []
    if args.run_dir:
        run_dirs.append(Path(args.run_dir))
    if args.scan_root:
        for sp in Path(args.scan_root).rglob("run_summary.json"):
            run_dirs.append(sp.parent)
    if not run_dirs:
        print("[ERROR] pass --run-dir or --scan-root"); sys.exit(1)

    run_dirs = sorted(set(path.resolve() for path in run_dirs))
    if args.shard_count > 1:
        total = len(run_dirs)
        run_dirs = [rd for i, rd in enumerate(run_dirs)
                    if i % args.shard_count == args.shard_index]
        print(f"[shard] {args.shard_index}/{args.shard_count}: "
              f"{len(run_dirs)} of {total} run dirs")
    print(f"[eval] {len(run_dirs)} run dir(s) to evaluate "
          f"| checkpoint={args.prefer}.pt -> {out_name}")
    failures = []
    for rd in run_dirs:
        eval_path = rd / out_name
        summary = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
        expected_prompt_format = summary.get("prompt_format") or "plain_mcqa"
        if args.skip_existing and valid_existing_eval(eval_path, expected_prompt_format):
            print(f"[skip] {rd} (complete finite eval_metrics.json exists)")
            continue
        if args.skip_existing and eval_path.exists():
            print(f"[retry] {rd} (existing eval_metrics.json is incomplete/non-finite)")
        try:
            eval_one_run(rd, device=args.device, dtype=args.dtype, batch_size=args.batch_size,
                         max_length=args.max_length, indomain=args.indomain, ood=args.ood,
                         seed=args.seed, max_test=args.max_test, prefer=args.prefer,
                         strict_checkpoint=args.strict_checkpoint, out_name=out_name)
        except Exception as e:
            print(f"[FAIL] {rd}: {e}")
            failures.append((rd, e))
    if failures:
        failed_paths = ", ".join(str(path) for path, _ in failures[:5])
        raise SystemExit(
            f"downstream evaluation failed for {len(failures)}/{len(run_dirs)} run(s): {failed_paths}"
        )


if __name__ == "__main__":
    main()
