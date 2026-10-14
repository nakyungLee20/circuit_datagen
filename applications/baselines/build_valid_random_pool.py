"""Build a 10k trainer-convertible fully-random-circuit pool.

Inputs must carry the rebuttal full-randomization contract: target positions /
heads and activation-vector orientations are both randomized.  Legacy
``fixed_both_random`` assets that only randomized positions are rejected.
Within a compliant 5,000 × 4 raw / top-2 selected asset, valid selected rows
are retained; an invalid selected row is replaced by the next-highest-scoring
valid sibling from the same record. A deterministic global score fallback is
used only if a record has no valid sibling, and that deviation is recorded.
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
from collections import Counter, defaultdict
from pathlib import Path


from applications.curriculum_ft import generated_row_to_example
from applications.baselines.fully_random_circuit_entry import (
    RANDOMIZATION_CONTRACT,
    validate_fully_random_output_dir,
)
from applications.experiment_utils.io import dump_pool_as_mirror, read_jsonl, write_json, write_jsonl


def _key(row) -> str:
    candidate_id = row.get("candidate_id") or row.get("record_id")
    if candidate_id:
        return str(candidate_id)
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _record_id(row) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return str(row.get("record_id") or source.get("id") or source.get("hf_id") or "")


def _score(row) -> float:
    value = row.get("final_score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-input", required=True)
    parser.add_argument("--raw-input", required=True)
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--n-select", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0, help="retained for CLI compatibility; ordering is score-deterministic")
    args = parser.parse_args()

    source_dir = Path(args.selected_input).expanduser().resolve().parent
    if Path(args.raw_input).expanduser().resolve().parent != source_dir:
        raise ValueError("selected/raw fully-random inputs must share one output directory")
    source_manifest = validate_fully_random_output_dir(
        source_dir,
        recipe="easy",
        expected_seed_data_num=5000,
        expected_num_candidates=4,
        expected_keep_per_record=2,
        expected_topk_edges=250,
    )
    selected_rows = read_jsonl(args.selected_input)
    raw_rows = read_jsonl(args.raw_input)
    if not (0 < args.n_select <= len(selected_rows)):
        raise ValueError(
            f"--n-select must be in [1, selected_input_count={len(selected_rows)}]"
        )
    budget_rows = selected_rows[: args.n_select]
    invalid = []
    kept = []
    seen = set()
    selected_keys = {_key(row) for row in budget_rows}
    valid_cache = {}

    def is_valid(row, source_name: str, source_idx: int) -> bool:
        key = _key(row)
        if key not in valid_cache:
            valid_cache[key] = generated_row_to_example(row, dataset_name="sciq") is not None
        if not valid_cache[key]:
            invalid.append(
                {
                    "source": source_name,
                    "source_idx": source_idx,
                    "record_id": _record_id(row),
                    "candidate_id": row.get("candidate_id") or row.get("record_id"),
                    "final_score": row.get("final_score"),
                    "reason": "rejected by curriculum_ft.generated_row_to_example",
                }
            )
            return False
        return True

    def keep(row) -> bool:
        key = _key(row)
        if key in seen:
            return False
        seen.add(key)
        kept.append(row)
        return True

    target_per_record = Counter(_record_id(row) for row in budget_rows)
    kept_per_record = Counter()
    for idx, row in enumerate(budget_rows):
        if is_valid(row, "selected", idx) and keep(row):
            kept_per_record[_record_id(row)] += 1

    n_from_selected = len(kept)
    raw_by_record = defaultdict(list)
    for idx, row in enumerate(raw_rows):
        raw_by_record[_record_id(row)].append((idx, row))
    for record_id in raw_by_record:
        raw_by_record[record_id].sort(
            key=lambda item: (-_score(item[1]), _key(item[1]), item[0])
        )

    same_record_replacements = 0
    deficient_records = []
    for record_id in sorted(target_per_record):
        needed = target_per_record[record_id] - kept_per_record[record_id]
        for raw_idx, row in raw_by_record.get(record_id, []):
            if needed <= 0:
                break
            if _key(row) in selected_keys or _key(row) in seen:
                continue
            if not is_valid(row, "same_record_fill", raw_idx):
                continue
            keep(row)
            kept_per_record[record_id] += 1
            same_record_replacements += 1
            needed -= 1
        if needed > 0:
            deficient_records.append({"record_id": record_id, "missing": needed})

    global_fallback = 0
    if len(kept) < args.n_select:
        global_order = sorted(
            enumerate(raw_rows),
            key=lambda item: (-_score(item[1]), _record_id(item[1]), _key(item[1]), item[0]),
        )
        for raw_idx, row in global_order:
            if _key(row) in selected_keys or _key(row) in seen:
                continue
            if not is_valid(row, "global_fallback", raw_idx):
                continue
            keep(row)
            global_fallback += 1
            if len(kept) >= args.n_select:
                break

    if len(kept) != args.n_select:
        raise RuntimeError(
            f"could only construct {len(kept)} unique trainer-convertible rows; "
            f"requested {args.n_select}"
        )
    out = dump_pool_as_mirror(
        kept,
        variant_root=Path(args.variant_root),
        recipe="easy",
        prompt_mode="fixed",
        gen_mode="both",
        circuit_target_mode="random",
        file_kind="selected",
    )
    invalid_path = out.parent / "easy_invalid_manifest.jsonl"
    write_jsonl(invalid_path, invalid)
    meta_path = out.parent / "easy_selection_meta.json"
    write_json(
        meta_path,
        {
            "selected_input": str(Path(args.selected_input).resolve()),
            "raw_input": str(Path(args.raw_input).resolve()),
            "selected_input_count": len(selected_rows),
            "raw_input_count": len(raw_rows),
            "valid_kept_from_selected": n_from_selected,
            "same_record_score_replacements": same_record_replacements,
            "global_score_fallback": global_fallback,
            "deficient_records_before_global_fallback": deficient_records,
            "replacement_policy": "same-record next-highest final_score; deterministic global final_score fallback only if necessary",
            "randomization_contract": RANDOMIZATION_CONTRACT,
            "fully_random_source_manifest": str(
                source_dir / "easy_fully_random_manifest.json"
            ),
            "random_direction_algorithm": source_manifest["random_direction_algorithm"],
            "random_position_algorithm": source_manifest["random_position_algorithm"],
            "invalid_encountered": len(invalid),
            "selected_count": len(kept),
            "invalid_manifest": str(invalid_path),
            "output": str(out),
        },
    )
    print(
        f"[random-pool] wrote {out}: selected_valid={n_from_selected}, "
        f"same_record_fill={same_record_replacements}, global_fallback={global_fallback}, "
        f"total={len(kept)}; meta={meta_path}"
    )


if __name__ == "__main__":
    main()
