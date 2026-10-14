#!/usr/bin/env python3
"""Apply the frozen steered per-record support cutoff to unsteered candidates."""

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
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


from applications.experiment_utils.io import (  # noqa: E402
    NAIVE_RAW_PATH,
    NAIVE_SELECTED_PATH,
    find_full_steered_pool_dir,
    read_jsonl,
)
from applications.steering_selection.support_scoring import (  # noqa: E402
    candidate_id,
    load_support_index,
    recipe_weights_from_rows,
    support_components,
    validate_score_coverage,
    weighted_support_score,
)

PROFILES = ("easy", "hard", "helpful")
MIX = {"easy": 0.333, "hard": 0.333, "helpful": 0.334}
DEFAULT_SCORES = (
    _SUPPLEMENTAL_ROOT
    / "steering_selection/work_exp2_steering_selection/_support"
    / "naive_support_scores.jsonl"
)
DEFAULT_OUTPUT = _SUPPLEMENTAL_ROOT / "additional/results/frozen_threshold"


def record_id(row: Mapping[str, Any]) -> str:
    value = row.get("record_id")
    if value is None or not str(value):
        raise ValueError(f"candidate {candidate_id(row)!r} has no record_id")
    return str(value)


def group_ids(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[record_id(row)].append(candidate_id(row))
    return dict(groups)


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive-raw", default=str(NAIVE_RAW_PATH))
    parser.add_argument("--naive-eligible", default=str(NAIVE_SELECTED_PATH))
    parser.add_argument("--naive-support-scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # This loader deliberately rejects an in-progress or stale score file.
    score_index = load_support_index(args.naive_support_scores)
    naive_raw = read_jsonl(args.naive_raw)
    validate_score_coverage(naive_raw, score_index, label="naive raw")
    eligible = read_jsonl(args.naive_eligible)
    eligible_ids = [candidate_id(row) for row in eligible]
    if len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("post-dedup unsteered pool contains duplicate candidate IDs")
    missing = sorted(set(eligible_ids) - set(score_index))
    if missing:
        raise ValueError(f"eligible unsteered candidates lack scores: {missing[:3]}")

    eligible_groups = group_ids(eligible)
    profile_rows: list[dict[str, Any]] = []
    counts_by_profile: dict[str, dict[str, dict[str, int]]] = {}
    source_paths: dict[str, dict[str, str]] = {}

    for profile in PROFILES:
        directory = find_full_steered_pool_dir(profile)
        raw_path = directory / f"{profile}_raw_candidates.jsonl"
        selected_path = directory / f"{profile}_selected.jsonl"
        raw = read_jsonl(raw_path)
        selected = read_jsonl(selected_path)
        weights = recipe_weights_from_rows(raw, profile)
        raw_scores = {
            candidate_id(row): weighted_support_score(support_components(row), weights)
            for row in raw
        }
        raw_groups = group_ids(raw)
        selected_groups = group_ids(selected)
        if set(raw_groups) != set(eligible_groups) or set(selected_groups) != set(raw_groups):
            raise ValueError(f"{profile}: steered/unsteered record populations differ")

        thresholds: dict[str, float] = {}
        reconstructed = 0
        selected_ids = {candidate_id(row) for row in selected}
        for rid, ids in raw_groups.items():
            chosen = selected_groups[rid]
            if len(ids) != 4 or len(chosen) != 2:
                raise ValueError(
                    f"{profile}/{rid}: expected 4 raw and 2 selected, "
                    f"got {len(ids)} and {len(chosen)}"
                )
            if not set(chosen).issubset(raw_scores):
                raise ValueError(f"{profile}/{rid}: selected candidate missing from raw pool")
            threshold = min(raw_scores[cid] for cid in chosen)
            thresholds[rid] = threshold
            reconstructed += sum(raw_scores[cid] >= threshold for cid in ids)

        per_record: dict[str, dict[str, int]] = {}
        passed = 0
        total = 0
        for rid, ids in eligible_groups.items():
            threshold = thresholds[rid]
            n_pass = sum(
                weighted_support_score(score_index[cid], weights) >= threshold
                for cid in ids
            )
            per_record[rid] = {"passed": n_pass, "total": len(ids)}
            passed += n_pass
            total += len(ids)

        rate = passed / total
        profile_rows.append(
            {
                "profile": profile,
                "passed": passed,
                "eligible": total,
                "frozen_threshold_pass_rate": rate,
                "steered_reference_retention": len(selected) / len(raw),
                "selected_rule_reconstruction_rate": len(selected) / reconstructed,
            }
        )
        counts_by_profile[profile] = per_record
        source_paths[profile] = {
            "steered_raw": str(raw_path.resolve()),
            "steered_selected": str(selected_path.resolve()),
        }

    record_ids = sorted(eligible_groups)
    rng = np.random.default_rng(args.seed)
    boot = {profile: np.empty(args.bootstrap) for profile in PROFILES}
    boot["uniform_mix"] = np.empty(args.bootstrap)
    pass_arrays = {
        profile: np.asarray(
            [counts_by_profile[profile][rid]["passed"] for rid in record_ids],
            dtype=np.int64,
        )
        for profile in PROFILES
    }
    total_arrays = {
        profile: np.asarray(
            [counts_by_profile[profile][rid]["total"] for rid in record_ids],
            dtype=np.int64,
        )
        for profile in PROFILES
    }
    for index in range(args.bootstrap):
        sample = rng.integers(0, len(record_ids), len(record_ids))
        mixed = 0.0
        for profile in PROFILES:
            rate = float(pass_arrays[profile][sample].sum()) / float(
                total_arrays[profile][sample].sum()
            )
            boot[profile][index] = rate
            mixed += MIX[profile] * rate
        boot["uniform_mix"][index] = mixed

    rates = {row["profile"]: row["frozen_threshold_pass_rate"] for row in profile_rows}
    overall = sum(MIX[profile] * rates[profile] for profile in PROFILES)
    for row in profile_rows:
        row["bootstrap_95ci"] = percentile_ci(boot[row["profile"]])

    summary = {
        "headline": {
            "original_identical_rank_rule_retention": 0.5,
            "frozen_steered_cutoff_unsteered_pass_rate": overall,
            "bootstrap_95ci": percentile_ci(boot["uniform_mix"]),
            "mix": MIX,
        },
        "definition": (
            "For each profile and source record, freeze the lower support score "
            "among the two actually selected steered candidates, then apply that "
            "numeric cutoff to post-dedup unsteered candidates from the same record."
        ),
        "important_distinction": (
            "Re-ranking each unsteered group and retaining its top 2 of 4 is exactly "
            "50% by construction; the frozen-cutoff rate is the non-mechanical diagnostic."
        ),
        "ci_method": (
            f"{args.bootstrap} paired record-level bootstrap replicates; seed={args.seed}"
        ),
        "profiles": profile_rows,
        "sources": {
            "unsteered_raw": str(Path(args.naive_raw).resolve()),
            "unsteered_post_dedup": str(Path(args.naive_eligible).resolve()),
            "unsteered_support_scores": str(Path(args.naive_support_scores).resolve()),
            "steered": source_paths,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_threshold_acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "frozen_threshold_acceptance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    lo, hi = summary["headline"]["bootstrap_95ci"]
    line = (
        "Under the original rank-based rule, retention is 50% by construction. "
        f"Using the steered per-record cutoff unchanged, unsteered generation passes "
        f"at {100 * overall:.2f}% (95% record-bootstrap CI "
        f"{100 * lo:.2f}–{100 * hi:.2f}%)."
    )
    (output_dir / "rebuttal_line.txt").write_text(line + "\n", encoding="utf-8")
    print(line)


if __name__ == "__main__":
    main()
