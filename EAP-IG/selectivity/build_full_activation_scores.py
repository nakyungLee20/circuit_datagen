"""Build genuine per-sample circuit-attribution scores for all SciQ train rows.

This is a thin, resumable Exp4 launcher around the existing EAP-IG full-split
scorer. It fixes the circuit budget and score definition used by the 3x3
Spearman analysis: top-250, absolute edge ranking, weighted direction-aware
aggregation, and the complete 11,679-example SciQ training split.
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
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


ROOT = _REPO_ROOT
for directory in (ROOT, ROOT / "EAP-IG", ROOT / "EAP-IG", ROOT / "EAP-IG" / "src"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

from applications import circuit_scores as _circuit_scores  # type: ignore  # noqa: E402
from data_gens.mcqa import prepare_task_splits  # type: ignore  # noqa: E402


PROMPT_SCHEMA_KEYS = (
    "dataset",
    "model_name",
    "seed",
    "fewshot_k",
    "fewshot_hf_ids",
    "num_train_examples",
    "num_val_examples",
    "train_frac",
    "use_chat_template",
    "system_prompt",
    "answer_instruction",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-key", choices=("aum", "el2n", "grad_align"), required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--circuits-root",
        default=str(_REPO_ROOT / 'circuit_results/Qwen__Qwen2.5-1.5B-Instruct'),
    )
    parser.add_argument(
        "--buckets-root",
        default=str(_REPO_ROOT / 'data_gens/buckets/application/Qwen__Qwen2.5-1.5B-Instruct'),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument("--expected-samples", type=int, default=11679)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument(
        "--incorrect-agg", choices=("auto", "max", "mean", "min"), default="max"
    )
    parser.add_argument("--noise-sigma", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _object_sha256(obj: Dict[str, Any]) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return obj


def load_exact_utility_rows(
    utility_json: Path,
    train_rows: Sequence[Dict[str, Any]],
    score_key: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], str]:
    obj = _read_json(utility_json)
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
    items = obj.get("items")
    if meta is None or not isinstance(items, list):
        raise ValueError(f"{utility_json}: expected meta object and items list")
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{utility_json}: item {index} is not an object")
        hf_id = str(item.get("hf_id", "")).strip()
        if not hf_id or hf_id in by_id:
            raise ValueError(f"{utility_json}: missing/duplicate hf_id at item {index}")
        by_id[hf_id] = item
    raw_by_id = {str(row.get("hf_id", "")): row for row in train_rows}
    if len(raw_by_id) != len(train_rows) or set(by_id) != set(raw_by_id):
        raise ValueError(
            f"{utility_json}: exact SciQ train coverage mismatch: utility={len(by_id)}, "
            f"raw={len(raw_by_id)}, common={len(set(by_id) & set(raw_by_id))}"
        )

    prompt_digest = hashlib.sha256()
    raw_keys = ("task", "question_text", "choices", "label_space", "gold_label")
    for hf_id in sorted(by_id):
        item, raw = by_id[hf_id], raw_by_id[hf_id]
        for key in raw_keys:
            if item.get(key) != raw.get(key):
                raise ValueError(f"{utility_json}: raw split {key} mismatch at {hf_id}")
        prompt = str(item.get("prompt", ""))
        label_space = item.get("label_space")
        label_token_ids = item.get("label_token_ids")
        gold_label = str(item.get("gold_label", ""))
        if (
            not prompt
            or not isinstance(label_space, list)
            or not isinstance(label_token_ids, dict)
            or gold_label not in label_space
            or gold_label not in label_token_ids
        ):
            raise ValueError(f"{utility_json}: incomplete exact prompt/label data at {hf_id}")
        try:
            normalized_token_ids = {
                str(label): int(label_token_ids[str(label)]) for label in label_space
            }
            gold_token_id = int(item.get("gold_token_id"))
            utility = float((item.get("scores") or {}).get(score_key))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{utility_json}: invalid exact token/utility data at {hf_id}") from exc
        if gold_token_id != normalized_token_ids[gold_label]:
            raise ValueError(f"{utility_json}: gold token map mismatch at {hf_id}")
        if not math.isfinite(utility):
            raise ValueError(f"{utility_json}: non-finite {score_key} at {hf_id}")
        prompt_record = {
            "hf_id": hf_id,
            "prompt": prompt,
            "label_space": [str(value) for value in label_space],
            "gold_label": gold_label,
            "gold_token_id": gold_token_id,
            "label_token_ids": normalized_token_ids,
        }
        prompt_digest.update(
            json.dumps(
                prompt_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        prompt_digest.update(b"\n")
    return by_id, dict(meta), prompt_digest.hexdigest()


def exact_utility_runtime_builder(
    utility_rows: Dict[str, Dict[str, Any]], expected_score_key: str
) -> Callable[..., Tuple[list[Dict[str, Any]], int]]:
    """Replace app1's plain formatter with the exact stored utility prompts."""

    def builder(
        examples: Sequence[Dict[str, Any]],
        dataset: str,
        score_key: str,
        split_name: str,
        tokenizer: Any,
        seed: int,
        corrupt_type: str,
        noise_sigma: float,
    ) -> Tuple[list[Dict[str, Any]], int]:
        del tokenizer
        if dataset != "sciq" or score_key != expected_score_key:
            raise ValueError(
                f"exact utility builder expected sciq/{expected_score_key}, "
                f"got {dataset}/{score_key}"
            )
        example_ids = [str(example.get("hf_id", "")) for example in examples]
        if len(example_ids) != len(set(example_ids)) or set(example_ids) != set(utility_rows):
            raise ValueError("exact utility builder received a different split/hf_id set")
        rows: list[Dict[str, Any]] = []
        for example in examples:
            hf_id = str(example["hf_id"])
            source = utility_rows[hf_id]
            label_space = [str(value) for value in source["label_space"]]
            label_token_ids = {
                label: int(source["label_token_ids"][label]) for label in label_space
            }
            gold_label = str(source["gold_label"])
            gold_token_id = int(source["gold_token_id"])
            corrupt_seed = int(
                hashlib.sha256(
                    f"{seed}::{dataset}::{score_key}::{split_name}::{hf_id}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            ) % (2**31 - 1)
            prompt = str(source["prompt"])
            rows.append(
                {
                    "clean": prompt,
                    "corrupted": prompt,
                    "label": gold_token_id,
                    "hf_id": hf_id,
                    "task": str(source.get("task", dataset)),
                    "score_key": score_key,
                    "bucket": split_name,
                    "gold_label": gold_label,
                    "label_space": label_space,
                    "label_token_ids": label_token_ids,
                    "option_token_ids": [label_token_ids[label] for label in label_space],
                    "corrupt": {
                        "type": corrupt_type,
                        "sigma": float(noise_sigma),
                        "seed": corrupt_seed,
                    },
                }
            )
        return rows, 0

    return builder


def build_request_contract(
    args: argparse.Namespace,
    train_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    circuits_root = Path(args.circuits_root).expanduser()
    buckets_root = Path(args.buckets_root).expanduser()
    high_pt = _circuit_scores._find_latest_circuit_pt(
        circuits_root, "sciq", args.score_key, "high", "gaussian"
    )
    low_pt = _circuit_scores._find_latest_circuit_pt(
        circuits_root, "sciq", args.score_key, "low", "gaussian"
    )
    utility_json = _circuit_scores._find_all_scores_json(
        buckets_root, "sciq", args.score_key
    )
    if high_pt is None or not high_pt.is_file():
        raise FileNotFoundError(f"missing high circuit for {args.score_key}: {circuits_root}")
    if low_pt is None or not low_pt.is_file():
        raise FileNotFoundError(f"missing low circuit for {args.score_key}: {circuits_root}")
    if utility_json is None or not utility_json.is_file():
        raise FileNotFoundError(f"missing utility JSON for {args.score_key}: {buckets_root}")
    utility_rows, source_meta, prompt_sha256 = load_exact_utility_rows(
        utility_json, train_rows, args.score_key
    )
    prompt_schema = {key: source_meta.get(key) for key in PROMPT_SCHEMA_KEYS}
    required_source_meta = {
        "dataset": "sciq",
        "model_name": args.model_name,
        "seed": int(args.seed),
        "fewshot_k": 3,
        "num_train_examples": int(args.expected_samples),
        "train_frac": 0.8,
        "use_chat_template": True,
    }
    source_mismatches = [
        f"{key}: actual={source_meta.get(key)!r}, expected={value!r}"
        for key, value in required_source_meta.items()
        if source_meta.get(key) != value
    ]
    fewshot_ids = source_meta.get("fewshot_hf_ids")
    if not isinstance(fewshot_ids, list) or len(fewshot_ids) != 3:
        source_mismatches.append("fewshot_hf_ids does not contain exactly 3 IDs")
    for key in ("system_prompt", "answer_instruction"):
        if not str(source_meta.get(key, "")).strip():
            source_mismatches.append(f"{key} is empty")
    if source_mismatches:
        raise ValueError(
            f"{utility_json}: utility prompt schema mismatch: "
            + "; ".join(source_mismatches)
        )
    hf_ids = sorted(str(row.get("hf_id", "")) for row in train_rows)
    hf_id_sha256 = hashlib.sha256("\n".join(hf_ids).encode("utf-8")).hexdigest()
    split_payload = [
        {
            key: row.get(key)
            for key in (
                "hf_id", "task", "question_text", "choices", "label_space", "gold_label"
            )
        }
        for row in sorted(train_rows, key=lambda item: str(item.get("hf_id", "")))
    ]
    split_sha256 = hashlib.sha256(
        json.dumps(
            split_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    contract = {
        "schema_version": 2,
        "dataset": "sciq",
        "score_key": args.score_key,
        "expected_samples": int(args.expected_samples),
        "hf_id_sha256": hf_id_sha256,
        "split_content_sha256": split_sha256,
        "prompt_source": "exact utility JSON prompt/gold_token_id/label_token_ids",
        "utility_prompt_sha256": prompt_sha256,
        "utility_source_meta_sha256": _object_sha256(source_meta),
        "prompt_schema": prompt_schema,
        "model_name": args.model_name,
        "circuits_root": str(circuits_root.resolve()),
        "buckets_root": str(buckets_root.resolve()),
        "utility_json": str(utility_json.resolve()),
        "utility_json_sha256": _sha256(utility_json),
        "high_circuit_pt": str(high_pt.resolve()),
        "high_circuit_sha256": _sha256(high_pt),
        "low_circuit_pt": str(low_pt.resolve()),
        "low_circuit_sha256": _sha256(low_pt),
        "corrupt_type": "gaussian",
        "topk_edges": int(args.topk_edges),
        "topk_rank_abs": True,
        "score_mode": "weighted_direction_aware",
        "seed": int(args.seed),
        "device": args.device,
        "dtype": args.dtype,
        "ig_steps_arg": int(args.ig_steps),
        "incorrect_agg_arg": args.incorrect_agg,
        "noise_sigma": float(args.noise_sigma),
        "random_baseline_trials": 0,
    }
    return contract, utility_rows


def validate_completed_artifact(
    output_path: Path,
    metadata: Dict[str, Any],
    contract: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
) -> None:
    expected_ids = {str(row.get("hf_id", "")) for row in train_rows}
    records: Dict[str, Dict[str, str]] = {}
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            hf_id = str(row.get("hf_id", "")).strip()
            if not hf_id:
                raise RuntimeError(f"{output_path}: CSV row {index} has no hf_id")
            if hf_id in records:
                raise RuntimeError(f"{output_path}: duplicate hf_id {hf_id!r}")
            records[hf_id] = dict(row)
    if set(records) != expected_ids:
        raise RuntimeError(
            f"{output_path}: hf_id coverage mismatch: csv={len(records)}, "
            f"expected={len(expected_ids)}, common={len(set(records) & expected_ids)}"
        )
    for hf_id, row in records.items():
        if row.get("dataset") != "sciq" or row.get("score_key") != contract["score_key"]:
            raise RuntimeError(f"{output_path}: dataset/score_key mismatch at {hf_id}")
        for key in ("true_score", "high_circuit_score", "low_circuit_score"):
            try:
                finite = math.isfinite(float(row.get(key, "")))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise RuntimeError(f"{output_path}: non-finite {key} at {hf_id}")

    circuits = metadata.get("circuits")
    runtime = metadata.get("runtime")
    if not isinstance(circuits, dict) or not isinstance(runtime, dict):
        raise RuntimeError("attribution metadata lacks circuit/runtime certification")
    checks = {
        "high circuit": (
            str(Path(str(circuits.get("high_pt", ""))).resolve()),
            contract["high_circuit_pt"],
        ),
        "low circuit": (
            str(Path(str(circuits.get("low_pt", ""))).resolve()),
            contract["low_circuit_pt"],
        ),
        "utility JSON": (
            str(Path(str(metadata.get("all_scores_json", ""))).resolve()),
            contract["utility_json"],
        ),
        "model": (runtime.get("model_name"), contract["model_name"]),
        "score mode": (runtime.get("score_mode"), contract["score_mode"]),
        "top-k": (circuits.get("topk_edges"), contract["topk_edges"]),
        "absolute rank": (circuits.get("topk_rank_abs"), True),
        "high mapped edges": (circuits.get("high_selected_edges"), contract["topk_edges"]),
        "low mapped edges": (circuits.get("low_selected_edges"), contract["topk_edges"]),
        "high missing edges": (circuits.get("high_missing_runtime_edges"), 0),
        "low missing edges": (circuits.get("low_missing_runtime_edges"), 0),
        "weighted mode": (runtime.get("weighted_direction_aware"), True),
        "IG steps": (runtime.get("ig_steps"), contract["ig_steps_arg"]),
        "incorrect aggregation": (
            runtime.get("incorrect_agg"), contract["incorrect_agg_arg"]
        ),
        "random trials": (runtime.get("random_baseline_trials_arg"), 0),
    }
    mismatches = [
        f"{name}: actual={actual!r}, expected={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError("invalid attribution metadata: " + "; ".join(mismatches))
    for key in (
        "dropped_rows_before_scoring", "skipped_dataset_fail", "skipped_attribute_fail"
    ):
        if int(metadata.get(key, -1)) != 0:
            raise RuntimeError(f"invalid attribution metadata: {key}={metadata.get(key)!r}")
    if int(metadata.get("n_original_scores_available", -1)) != len(expected_ids):
        raise RuntimeError(
            "utility score coverage mismatch: "
            f"available={metadata.get('n_original_scores_available')!r}, "
            f"expected={len(expected_ids)}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.topk_edges != 250:
        raise ValueError(f"Exp4 activation score budget must be 250, got {args.topk_edges}")
    if args.ig_steps <= 0:
        raise ValueError("Exp4 requires an explicit positive --ig-steps (default: 5)")
    if args.incorrect_agg == "auto":
        raise ValueError("Exp4 requires an explicit --incorrect-agg (default: max)")
    if not math.isfinite(args.noise_sigma) or args.noise_sigma < 0:
        raise ValueError("--noise-sigma must be a finite non-negative number")
    splits = prepare_task_splits("sciq", seed=args.seed, train_frac=0.8)
    train_rows = list(splits["train"])
    if len(train_rows) != args.expected_samples:
        raise ValueError(
            f"SciQ train size mismatch: expected={args.expected_samples}, actual={len(train_rows)}"
        )
    contract, utility_rows = build_request_contract(args, train_rows)
    output_path = Path(args.output_csv)
    contract_path = output_path.with_suffix(".exp4_request.json")
    completion_path = output_path.with_suffix(".exp4_complete.json")
    contract_sha256 = _object_sha256(contract)
    config = {
        **contract,
        "output_csv": str(output_path),
        "contract": str(contract_path),
        "completion": str(completion_path),
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if output_path.is_file() and not args.force:
        if not contract_path.is_file():
            raise RuntimeError(
                f"existing CSV has no Exp4 request contract: {output_path}. "
                "Use --force once to replace this uncertified artifact."
            )
        existing_contract = _read_json(contract_path)
        if existing_contract != contract:
            raise RuntimeError(
                f"existing CSV request contract does not match current inputs/config: "
                f"{contract_path}. Use --force to recompute it."
            )
        if completion_path.is_file():
            completion = _read_json(completion_path)
            if completion.get("request_contract_sha256") != contract_sha256:
                raise RuntimeError(
                    f"completion certificate does not match request: {completion_path}. "
                    "Use --force to recompute it."
                )
            if completion.get("status") == "complete" and (
                completion.get("csv_sha256") != _sha256(output_path)
            ):
                raise RuntimeError(
                    f"completed attribution CSV changed after certification: {output_path}. "
                    "Use --force to recompute it."
                )
            if completion.get("status") == "complete":
                metadata_path = output_path.with_suffix(".meta.json")
                if not metadata_path.is_file():
                    raise RuntimeError(
                        f"certified CSV lacks scorer metadata: {metadata_path}. "
                        "Use --force to recompute it."
                    )
                cached_metadata = _read_json(metadata_path)
                validate_completed_artifact(
                    output_path, cached_metadata, contract, train_rows
                )
                print(
                    f"[skip] certified full SciQ activation scores: {output_path} "
                    f"(n={len(train_rows)})"
                )
                return
    _write_json_atomic(contract_path, contract)
    _write_json_atomic(
        completion_path,
        {
            "schema_version": 1,
            "status": "in_progress",
            "request_contract_sha256": contract_sha256,
        },
    )
    original_builder = _circuit_scores._build_runtime_rows
    _circuit_scores._build_runtime_rows = exact_utility_runtime_builder(
        utility_rows, args.score_key
    )
    try:
        metadata = _circuit_scores.build_full_split_score_csv(
            dataset="sciq",
            score_key=args.score_key,
            split_examples=train_rows,
            out_csv_path=args.output_csv,
            circuits_root=args.circuits_root,
            buckets_root=args.buckets_root,
            corrupt_type="gaussian",
            topk_edges=args.topk_edges,
            topk_rank_abs=True,
            sum_abs=False,
            direction_aware=False,
            weighted_direction_aware=True,
            random_baseline_trials=0,
            seed=args.seed,
            model_name_override=args.model_name,
            device=args.device,
            dtype=args.dtype,
            ig_steps_override=args.ig_steps,
            incorrect_agg_override=args.incorrect_agg,
            noise_sigma=args.noise_sigma,
            split_name="train",
            resume=not args.force,
            force_recompute=args.force,
            save_every=args.save_every,
            quiet=False,
        )
    finally:
        _circuit_scores._build_runtime_rows = original_builder
    n_records = int(metadata.get("n_records_total", metadata.get("n_records", -1)))
    if n_records != args.expected_samples:
        raise RuntimeError(
            f"full-split circuit scoring incomplete: expected={args.expected_samples}, got={n_records}. "
            "The CSV is resumable; rerun the same command."
        )
    validate_completed_artifact(output_path, metadata, contract, train_rows)
    _write_json_atomic(
        completion_path,
        {
            "schema_version": 1,
            "status": "complete",
            "request_contract_sha256": contract_sha256,
            "csv_sha256": _sha256(output_path),
            "n_records": n_records,
        },
    )
    print(f"[ok] full SciQ activation scores: {args.output_csv} (n={n_records})")


if __name__ == "__main__":
    main()
