from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "generation_inputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep score conversion and --help usable without the training dependencies.
SUPPORTED_DATASETS = {"sciq", "arc_easy", "piqa", "winogrande"}


TOPIC_BY_DATASET = {
    "sciq": "science",
    "arc_easy": "science",
    "piqa": "physical reasoning",
    "winogrande": "commonsense reasoning",
}

QUESTION_PREFIX_RE = re.compile(r"^(Question|Goal|Sentence):\s*", flags=re.IGNORECASE)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def normalize_question_text(text: str) -> str:
    return QUESTION_PREFIX_RE.sub("", (text or "").strip())


def resolve_answer_text(example: Mapping[str, Any]) -> str:
    label_space = [str(x) for x in example.get("label_space", [])]
    choices = [str(x) for x in example.get("choices", [])]
    gold_label = str(example.get("gold_label", ""))
    if gold_label in label_space:
        idx = label_space.index(gold_label)
        if 0 <= idx < len(choices):
            return choices[idx]
    return ""


def base_seed_row(example: Mapping[str, Any], dataset: str, split: str) -> Dict[str, Any]:
    hf_id = str(example.get("hf_id", "")).strip()
    return {
        "id": hf_id,
        "hf_id": hf_id,
        "dataset": dataset,
        "split": split,
        "topic": TOPIC_BY_DATASET.get(dataset, dataset),
        "question": normalize_question_text(str(example.get("question_text", ""))),
        "answer": resolve_answer_text(example),
        "choices": [str(x) for x in example.get("choices", [])],
        "gold_label": str(example.get("gold_label", "")),
        "label_space": [str(x) for x in example.get("label_space", [])],
    }


def choose_scores_path(dataset: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--scores-json not found: {path}")
        return path

    candidates = sorted(CHECKPOINTS_ROOT.glob(f"**/all_scores_{dataset}.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find all_scores_{dataset}.json under {CHECKPOINTS_ROOT}. "
            "Pass --scores-json explicitly."
        )

    def priority(path: Path) -> Tuple[int, int, int, str]:
        path_str = str(path)
        return (
            0 if "/static_full/" in path_str else 1,
            0 if "/grad_align/" in path_str else 1,
            len(path_str),
            path_str,
        )

    return sorted(candidates, key=priority)[0]


def _read_score_payload(path: Path) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list) or not payload["items"]:
        raise ValueError(f"{path}: expected a nonempty 'items' list")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: expected a 'meta' object")
    indexed = {}
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict) or not isinstance(item.get("hf_id"), str) or not item["hf_id"].strip():
            raise ValueError(f"{path}: item {index} requires a nonempty string hf_id")
        hf_id = item["hf_id"].strip()
        if hf_id in indexed:
            raise ValueError(f"{path}: duplicate hf_id {hf_id!r}")
        indexed[hf_id] = item
    return indexed, meta


def _score_value(item: Mapping[str, Any], metric: str, context: str) -> float:
    keys = ("grad_align", "gradalign") if metric == "gradalign" else (metric,)
    # Legacy merged inputs used orig_scores for GradAlign; fresh scoring uses scores.
    containers = [item.get("orig_scores", {}), item.get("scores", {}), item] if metric == "gradalign" else [item.get("scores", {}), item]
    value = None
    for container in containers:
        if isinstance(container, Mapping):
            for key in keys:
                if key in container and container[key] is not None:
                    value = container[key]
                    break
        if value is not None:
            break
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: missing or invalid {metric} score; use --grad-align-json for separate GradAlign output") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: non-finite {metric} score")
    return number


def load_scores_index(
    path: Path, dataset: str, split: str, grad_align_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Load legacy merged scores or join fresh dynamic/GradAlign files by identity.

    A two-file join requires identical full hf_id populations and rendered prompts,
    choices, answers, and label token IDs. Mismatched/invalid data is never silently
    dropped or overwritten. Split filtering happens only after the join checks.
    """
    items, meta = _read_score_payload(path)
    grad_items, grad_meta = ({}, {})
    if meta.get("dataset") not in (None, dataset):
        raise ValueError(f"{path}: dataset metadata does not match {dataset!r}")
    if grad_align_path is not None:
        grad_items, grad_meta = _read_score_payload(grad_align_path)
        if items.keys() != grad_items.keys():
            raise ValueError(
                "Score populations differ: "
                f"{len(items.keys() - grad_items.keys())} missing GradAlign IDs, "
                f"{len(grad_items.keys() - items.keys())} extra GradAlign IDs. "
                "Match dataset, split, seed, fewshot settings and max_items."
            )
        for field in ("dataset", "model_name", "seed", "train_frac", "fewshot_k", "fewshot_hf_ids", "use_chat_template", "system_prompt", "answer_instruction"):
            if field in meta and field in grad_meta and meta[field] != grad_meta[field]:
                raise ValueError(f"Score metadata differs: {field}")
        if grad_meta.get("dataset") not in (None, dataset):
            raise ValueError(f"{grad_align_path}: dataset metadata does not match {dataset!r}")
        for hf_id, item in items.items():
            other = grad_items[hf_id]
            for field in ("prompt", "choices", "gold_label", "label_token_ids"):
                if field not in item or field not in other or item[field] != other[field]:
                    raise ValueError(f"Score inputs differ or are missing: {hf_id}, {field}")
                if item[field] is None or item[field] == "" or item[field] == [] or item[field] == {}:
                    raise ValueError(f"Empty score input: {hf_id}, {field}")

    id_dataset = "winogrande_xl" if dataset == "winogrande" else dataset
    prefixes = (f"{id_dataset}/{split}/", f"{dataset}/{split}/")
    by_hf_id = {}
    for hf_id, item in items.items():
        if not hf_id.startswith(prefixes):
            continue
        grad_item = grad_items[hf_id] if grad_align_path is not None else item
        by_hf_id[hf_id] = {
            "aum": _score_value(item, "aum", hf_id),
            "el2n": _score_value(item, "el2n", hf_id),
            "gradalign": _score_value(grad_item, "gradalign", hf_id),
        }
    if not by_hf_id:
        raise ValueError(f"{path}: no scores found for {dataset}/{split}")
    return by_hf_id, {
        "scores_path": str(path), "source_meta": meta,
        "grad_align_path": str(grad_align_path) if grad_align_path is not None else None,
        "grad_align_source_meta": grad_meta,
        "num_items_loaded": len(by_hf_id),
        "duplicate_hf_ids_overwritten": 0,
    }


def maybe_slice(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def build_seed_and_calibration(
    dataset: str,
    split: str,
    scores_by_id: Mapping[str, Mapping[str, float]],
    seed_limit: int,
    calibration_limit: int,
    dataset_seed: int = 42,
    train_frac: float = 0.8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    from data_gens.mcqa import prepare_task_splits

    splits = prepare_task_splits(dataset, seed=dataset_seed, train_frac=train_frac)
    if split not in splits:
        raise ValueError(f"Split {split!r} not found for dataset={dataset}")

    examples = splits[split]
    seed_rows_all = [base_seed_row(ex, dataset=dataset, split=split) for ex in examples]
    unknown_ids = set(scores_by_id) - {row["hf_id"] for row in seed_rows_all}
    if unknown_ids:
        raise ValueError(f"Score IDs do not belong to the loaded dataset split: {sorted(unknown_ids)[:5]}")
    seed_rows = maybe_slice(seed_rows_all, seed_limit)

    calibration_rows_all: List[Dict[str, Any]] = []
    missing_scores = 0
    for row in seed_rows_all:
        hf_id = row["hf_id"]
        metrics = scores_by_id.get(hf_id)
        if metrics is None:
            missing_scores += 1
            continue
        calibration_rows_all.append(
            {
                **row,
                "aum": float(metrics["aum"]),
                "el2n": float(metrics["el2n"]),
                "gradalign": float(metrics["gradalign"]),
            }
        )
    calibration_rows = maybe_slice(calibration_rows_all, calibration_limit)

    stats = {
        "num_seed_rows_all": len(seed_rows_all),
        "num_seed_rows_written": len(seed_rows),
        "num_calibration_rows_all": len(calibration_rows_all),
        "num_calibration_rows_written": len(calibration_rows),
        "num_missing_scores": missing_scores,
    }
    return seed_rows, calibration_rows, stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export generation seed JSONL and activation-calibration JSONL from an application dataset. "
            "The default path and score resolver are set up for SciQ."
        )
    )
    p.add_argument("--dataset", type=str, default="sciq", choices=sorted(SUPPORTED_DATASETS))
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--scores-json",
        type=str,
        default=None,
        help="Dynamic all_scores_<dataset>.json (requires --grad-align-json), or legacy merged AUM/EL2N/GradAlign scores. Auto-resolved when omitted.",
    )
    p.add_argument("--grad-align-json", type=str, help="Separate all_grad_align_<dataset>.json; validates IDs, populations, and identical model inputs before joining.")
    p.add_argument("--seed-limit", type=int, default=0, help="0 means write every seed sample in the split.")
    p.add_argument("--calibration-limit", type=int, default=0, help="0 means write every calibration-capable sample in the split.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    dataset = str(args.dataset).strip().lower()
    split = str(args.split).strip().lower()
    out_dir = Path(args.out_dir).expanduser() / dataset / split

    scores_path = choose_scores_path(dataset, args.scores_json)
    grad_align_path = Path(args.grad_align_json).expanduser() if args.grad_align_json else None
    scores_by_id, score_meta = load_scores_index(scores_path, dataset=dataset, split=split, grad_align_path=grad_align_path)
    seed_rows, calibration_rows, stats = build_seed_and_calibration(
        dataset=dataset,
        split=split,
        scores_by_id=scores_by_id,
        seed_limit=int(args.seed_limit),
        calibration_limit=int(args.calibration_limit),
        dataset_seed=int(score_meta["source_meta"].get("seed", 42)),
        train_frac=float(score_meta["source_meta"].get("train_frac", 0.8)),
    )

    seed_tag = f"n{len(seed_rows)}" if args.seed_limit > 0 else "all"
    calibration_tag = f"n{len(calibration_rows)}" if args.calibration_limit > 0 else "all"

    seed_path = out_dir / f"{dataset}_{split}_seed_{seed_tag}.jsonl"
    calibration_path = out_dir / f"{dataset}_{split}_calibration_{calibration_tag}.jsonl"
    manifest_path = out_dir / f"{dataset}_{split}_seed_manifest.json"

    ensure_dir(out_dir)
    save_jsonl(seed_path, seed_rows)
    save_jsonl(calibration_path, calibration_rows)

    manifest = {
        "dataset": dataset,
        "split": split,
        "scores_json": str(scores_path),
        "grad_align_json": str(grad_align_path) if grad_align_path else None,
        "seed_path": str(seed_path),
        "calibration_path": str(calibration_path),
        "score_meta": score_meta,
        "stats": stats,
        "limits": {
            "seed_limit": int(args.seed_limit),
            "calibration_limit": int(args.calibration_limit),
        },
    }
    save_json(manifest_path, manifest)

    print(f"[DONE] seed_path={seed_path}")
    print(f"[DONE] calibration_path={calibration_path}")
    print(f"[DONE] manifest_path={manifest_path}")
    print(f"[INFO] num_seed_rows={len(seed_rows)} num_calibration_rows={len(calibration_rows)}")
    if stats["num_missing_scores"] > 0:
        print(f"[WARN] missing_scores={stats['num_missing_scores']} rows were excluded from calibration export")


if __name__ == "__main__":
    main()
