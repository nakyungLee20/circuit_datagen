"""Score true GradAlign on the matched full-trajectory checkpoint, no hook.

``cross_metric_eval`` records both GradAlign columns on the *same* ablated
checkpoint (no-hook vs persistent-hook). That acute contrast is small because a
persistently-ablated model has already adapted around the removed edges. This
module supplies the missing half of the two-checkpoint contrast: GradAlign for
every training sample evaluated on the matched full (no-ablation) trajectory
checkpoint, with its own validation reference gradient.

The delta consumed downstream is then

    delta_GA(row) = GA(full_ckpt, no hook) - GA(ablated_ckpt(row), hook)

with the right-hand term read from the existing per-row overlay ``scores``.
Everything except the checkpoint and the absent ablation graph is reused from
``cross_metric_eval`` so the two halves stay pairable.

GPU-heavy; ``--dry-run`` resolves inputs without loading the model.
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
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = _REPO_ROOT
if str(ROOT / "EAP-IG") not in sys.path:
    sys.path.append(str(ROOT / "EAP-IG"))

from selectivity.cross_metric_eval import (  # type: ignore  # noqa: E402
    ApplicationMCQADataset,
    DataLoader as _DataLoader,  # noqa: F401  (kept for import-parity)
    PromptScoreDataset,
    _canonical_path,
    _limit_rows,
    _read_json,
    _sha256,
    build_dataloader,
    collate_prompt_score,
    compute_reference_vector,
    get_target_params,
    load_all_score_rows,
    load_checkpoint_into_model,
    load_model,
    prepare_dataset,
    score_samples_by_dot,
    str_to_dtype,
    validation_metrics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="all_grad_align_sciq.json")
    parser.add_argument(
        "--full-checkpoint", required=True, help="final.pt of the matched full trajectory"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all train samples")
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-reference-items", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-if-valid",
        action="store_true",
        help="skip when an existing output already matches this exact configuration",
    )
    parser.add_argument("--output-json", required=True)
    return parser


def _expected_meta(args: argparse.Namespace, full_checkpoint: Path, input_path: Path) -> Dict[str, Any]:
    return {
        "input_json": _canonical_path(input_path),
        "input_json_sha256": _sha256(input_path),
        "full_checkpoint": _canonical_path(full_checkpoint),
        "full_checkpoint_sha256": _sha256(full_checkpoint),
        "score_source": "full-trajectory checkpoint, no hook, own validation reference",
        "seed": args.seed,
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_items": args.max_items,
        "max_val_items": args.max_val_items,
        "max_reference_items": args.max_reference_items,
    }


def _reusable(existing: Dict[str, Any], expected: Dict[str, Any]) -> tuple[bool, list[str]]:
    meta = existing.get("meta") if isinstance(existing.get("meta"), dict) else {}
    reasons = [
        f"meta.{key}: existing={meta.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if meta.get(key) != value
    ]
    if not isinstance(existing.get("items"), list) or not existing["items"]:
        reasons.append("items missing or empty")
    return (not reasons), reasons


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    input_path = Path(args.input_json).expanduser()
    full_checkpoint = Path(args.full_checkpoint).expanduser()
    output_path = Path(args.output_json).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not full_checkpoint.is_file():
        raise FileNotFoundError(full_checkpoint)

    source_meta = _read_json(input_path).get("meta", {})
    expected = _expected_meta(args, full_checkpoint, input_path)
    if args.dry_run:
        print(json.dumps({"meta": expected}, indent=2))
        return
    if args.reuse_if_valid and output_path.is_file():
        ok, reasons = _reusable(_read_json(output_path), expected)
        if ok:
            print(f"[skip] matching full-checkpoint GradAlign scores: {output_path}")
            return
        print(f"[stale] recomputing {output_path}")
        for reason in reasons[:8]:
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
    load_checkpoint_into_model(model, full_checkpoint, args.device, "matched full trajectory")
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = int(model.tokenizer.pad_token_id)

    all_rows, load_stats = load_all_score_rows(input_path, "grad_align", tokenizer=model.tokenizer)
    eval_rows = _limit_rows(list(all_rows.values()), args.max_items, args.seed + 31)
    eval_dataset = PromptScoreDataset(eval_rows, model.tokenizer, args.max_length)
    if len(eval_dataset) == 0:
        raise RuntimeError("no train samples available for GradAlign scoring")
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
        raise ValueError(
            f"few-shot reconstruction mismatch: expected={expected_fewshot}, actual={actual_fewshot}"
        )
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
    if len(reference_dataset) == 0:
        raise RuntimeError("no SciQ validation samples available for reference gradient")
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda batch: collate_prompt_score(batch, pad_id),
    )

    def forward_full(tokens: torch.Tensor, attention: torch.Tensor, **_: Any) -> torch.Tensor:
        return model(tokens, attention_mask=attention)

    target_params = get_target_params(model)
    reference = compute_reference_vector(
        model, reference_loader, forward_full, target_params, args.device, args.quiet
    )
    scores = score_samples_by_dot(
        model, eval_loader, forward_full, target_params, reference, args.device, 1.0, args.quiet
    )
    if len(scores) != len(eval_dataset):
        raise RuntimeError(
            f"GradAlign output lost samples: scored={len(scores)}, expected={len(eval_dataset)}"
        )
    baseline_eval = validation_metrics(model, validation_loader, forward_full, args.device)

    items = [
        {"hf_id": hf_id, "scores": {"grad_align": float(scores[hf_id])}}
        for hf_id in sorted(scores)
    ]
    metadata = {
        **expected,
        "metric_score_key": "grad_align",
        "definition": "per-sample last-layer loss gradient dot SciQ validation gradient",
        "reference_vector_source": "full checkpoint, no hook",
        "reference_split": "sciq/validation excluding the original few-shot examples",
        "reference_l2_norm": float(torch.linalg.vector_norm(reference.float()).item()),
        "effective_dtype": str(dtype).removeprefix("torch."),
        "n_scored": len(items),
        "n_reference": len(reference_dataset),
        "source_load_stats": load_stats,
        "evaluation": {"baseline": baseline_eval},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"meta": metadata, "items": items}, handle, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)
    print(f"[ok] full-checkpoint GradAlign scores: {output_path}")


if __name__ == "__main__":
    main()
