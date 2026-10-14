#!/usr/bin/env python3
"""Both directions of the frozen-cutoff acceptance diagnostic.

`frozen_threshold_acceptance.py` freezes the *steered* per-record cutoff and
measures how often unsteered candidates clear it.  A reviewer asking "is
steered acceptance substantially higher than unsteered acceptance?" also wants
the mirror image: freeze the *unsteered* per-record cutoff and measure how
often steered candidates clear it.  Under the paper's rank-based rule both
arms retain exactly 50% by construction, so neither number alone is
informative; the pair of frozen-cutoff rates is.

Direction A (steered cutoff -> unsteered candidates): reproduces the existing
script and is recomputed here so both arms come from one pass.
Direction B (unsteered cutoff -> steered candidates): the reviewer's phrasing.

Neither direction has 50% as its no-effect point.  A retain-top-2-of-4 cutoff
transferred to an independent group of 4 exchangeable candidates passes 2/5 of
them in expectation, so the null is 40%.  This script therefore also reports a
within-record label-permutation null and a threshold-free paired AUC,
P(steered score > unsteered score) within a source record, whose null is 0.5.
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
DEFAULT_OUTPUT = _SUPPLEMENTAL_ROOT / "additional/results/symmetric_acceptance"


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


def second_highest(values: list[float]) -> float:
    """The retain-top-2 cutoff: the lower of the two best scores in a group."""

    if len(values) < 2:
        raise ValueError("a top-2 cutoff needs at least two candidates")
    return sorted(values, reverse=True)[1]


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive-raw", default=str(NAIVE_RAW_PATH))
    parser.add_argument("--naive-eligible", default=str(NAIVE_SELECTED_PATH))
    parser.add_argument("--naive-support-scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    auc_records: dict[str, dict[str, tuple[float, float]]] = {}
    score_groups: dict[str, dict[str, tuple[list[float], list[float]]]] = {}

    for profile in PROFILES:
        directory = find_full_steered_pool_dir(profile)
        raw_path = directory / f"{profile}_raw_candidates.jsonl"
        selected_path = directory / f"{profile}_selected.jsonl"
        raw = read_jsonl(raw_path)
        selected = read_jsonl(selected_path)
        weights = recipe_weights_from_rows(raw, profile)

        steered_scores = {
            candidate_id(row): weighted_support_score(support_components(row), weights)
            for row in raw
        }
        unsteered_scores = {
            cid: weighted_support_score(score_index[cid], weights) for cid in eligible_ids
        }
        raw_groups = group_ids(raw)
        selected_groups = group_ids(selected)
        if set(raw_groups) != set(eligible_groups) or set(selected_groups) != set(raw_groups):
            raise ValueError(f"{profile}: steered/unsteered record populations differ")

        a_per_record: dict[str, dict[str, int]] = {}
        b_per_record: dict[str, dict[str, int]] = {}
        for rid, steered_ids in raw_groups.items():
            chosen = selected_groups[rid]
            unsteered_ids = eligible_groups[rid]
            if len(steered_ids) != 4 or len(chosen) != 2:
                raise ValueError(
                    f"{profile}/{rid}: expected 4 raw and 2 selected steered candidates, "
                    f"got {len(steered_ids)} and {len(chosen)}"
                )

            # Direction A: steered cutoff applied to unsteered candidates.
            steered_cutoff = min(steered_scores[cid] for cid in chosen)
            a_pass = sum(unsteered_scores[cid] >= steered_cutoff for cid in unsteered_ids)
            a_per_record[rid] = {"passed": a_pass, "total": len(unsteered_ids)}

            # Direction B: unsteered cutoff applied to steered candidates.
            unsteered_cutoff = second_highest([unsteered_scores[cid] for cid in unsteered_ids])
            b_pass = sum(steered_scores[cid] >= unsteered_cutoff for cid in steered_ids)
            b_per_record[rid] = {"passed": b_pass, "total": len(steered_ids)}

        # Threshold-free effect size: within each record, the fraction of
        # steered x unsteered candidate pairs where the steered one scores
        # higher (ties counted as half).  Null = 0.5.
        auc_num = 0.0
        auc_den = 0.0
        per_record_auc: dict[str, tuple[float, float]] = {}
        for rid, steered_ids in raw_groups.items():
            s_vals = np.asarray([steered_scores[cid] for cid in steered_ids])
            u_vals = np.asarray([unsteered_scores[cid] for cid in eligible_groups[rid]])
            wins = float((s_vals[:, None] > u_vals[None, :]).sum())
            ties = float((s_vals[:, None] == u_vals[None, :]).sum())
            num = wins + 0.5 * ties
            den = float(s_vals.size * u_vals.size)
            per_record_auc[rid] = (num, den)
            auc_num += num
            auc_den += den

        a_passed = sum(v["passed"] for v in a_per_record.values())
        a_total = sum(v["total"] for v in a_per_record.values())
        b_passed = sum(v["passed"] for v in b_per_record.values())
        b_total = sum(v["total"] for v in b_per_record.values())
        profile_rows.append(
            {
                "profile": profile,
                "rank_rule_retention_both_arms": 0.5,
                "unsteered_pass_at_steered_cutoff": a_passed / a_total,
                "unsteered_passed": a_passed,
                "unsteered_eligible": a_total,
                "steered_pass_at_unsteered_cutoff": b_passed / b_total,
                "steered_passed": b_passed,
                "steered_eligible": b_total,
                "paired_auc_steered_gt_unsteered": auc_num / auc_den,
            }
        )
        counts[profile] = {"A": a_per_record, "B": b_per_record}
        auc_records[profile] = per_record_auc
        score_groups[profile] = {
            rid: (
                [steered_scores[cid] for cid in raw_groups[rid]],
                [unsteered_scores[cid] for cid in eligible_groups[rid]],
            )
            for rid in raw_groups
        }

    record_ids = sorted(eligible_groups)
    rng = np.random.default_rng(args.seed)
    arrays = {
        (profile, direction, field): np.asarray(
            [counts[profile][direction][rid][field] for rid in record_ids], dtype=np.int64
        )
        for profile in PROFILES
        for direction in ("A", "B")
        for field in ("passed", "total")
    }
    boot = {
        key: np.empty(args.bootstrap)
        for key in [(p, d) for p in PROFILES for d in ("A", "B")] + [("mix", "A"), ("mix", "B")]
    }
    for index in range(args.bootstrap):
        sample = rng.integers(0, len(record_ids), len(record_ids))
        for direction in ("A", "B"):
            mixed = 0.0
            for profile in PROFILES:
                rate = float(arrays[(profile, direction, "passed")][sample].sum()) / float(
                    arrays[(profile, direction, "total")][sample].sum()
                )
                boot[(profile, direction)][index] = rate
                mixed += MIX[profile] * rate
            boot[("mix", direction)][index] = mixed

    auc_num_arrays = {
        profile: np.asarray([auc_records[profile][rid][0] for rid in record_ids])
        for profile in PROFILES
    }
    auc_den_arrays = {
        profile: np.asarray([auc_records[profile][rid][1] for rid in record_ids])
        for profile in PROFILES
    }
    boot_auc = {profile: np.empty(args.bootstrap) for profile in PROFILES}
    boot_auc["mix"] = np.empty(args.bootstrap)
    rng_auc = np.random.default_rng(args.seed + 1)
    for index in range(args.bootstrap):
        sample = rng_auc.integers(0, len(record_ids), len(record_ids))
        mixed = 0.0
        for profile in PROFILES:
            rate = float(auc_num_arrays[profile][sample].sum()) / float(
                auc_den_arrays[profile][sample].sum()
            )
            boot_auc[profile][index] = rate
            mixed += MIX[profile] * rate
        boot_auc["mix"][index] = mixed

    # Within-record label permutation: pool each record's steered and unsteered
    # scores and re-split at the observed group sizes.  This is the empirical
    # no-effect point for both directions given the real group sizes, and it
    # does not assume the analytic 4-vs-4 exchangeability case.
    rng_perm = np.random.default_rng(args.seed + 2)
    perm_a = {profile: np.empty(args.permutations) for profile in PROFILES}
    perm_b = {profile: np.empty(args.permutations) for profile in PROFILES}
    for profile in PROFILES:
        pooled = {
            rid: (np.asarray(s + u), len(s))
            for rid, (s, u) in score_groups[profile].items()
        }
        for index in range(args.permutations):
            a_pass = a_tot = b_pass = b_tot = 0
            for values, n_steered in pooled.values():
                shuffled = rng_perm.permutation(values)
                pseudo_steered = shuffled[:n_steered]
                pseudo_unsteered = shuffled[n_steered:]
                a_cut = second_highest(list(pseudo_steered))
                b_cut = second_highest(list(pseudo_unsteered))
                a_pass += int((pseudo_unsteered >= a_cut).sum())
                a_tot += pseudo_unsteered.size
                b_pass += int((pseudo_steered >= b_cut).sum())
                b_tot += pseudo_steered.size
            perm_a[profile][index] = a_pass / a_tot
            perm_b[profile][index] = b_pass / b_tot

    for row in profile_rows:
        profile = row["profile"]
        row["unsteered_pass_95ci"] = percentile_ci(boot[(profile, "A")])
        row["steered_pass_95ci"] = percentile_ci(boot[(profile, "B")])
        row["paired_auc_95ci"] = percentile_ci(boot_auc[profile])
        row["permutation_null_unsteered_pass"] = float(perm_a[profile].mean())
        row["permutation_null_steered_pass"] = float(perm_b[profile].mean())

    def mixed(field: str) -> float:
        by_profile = {r["profile"]: r[field] for r in profile_rows}
        return sum(MIX[p] * by_profile[p] for p in PROFILES)

    overall_a = mixed("unsteered_pass_at_steered_cutoff")
    overall_b = mixed("steered_pass_at_unsteered_cutoff")
    overall_auc = mixed("paired_auc_steered_gt_unsteered")

    summary = {
        "headline": {
            "rank_rule_retention_both_arms": 0.5,
            "unsteered_pass_at_steered_cutoff": overall_a,
            "unsteered_pass_95ci": percentile_ci(boot[("mix", "A")]),
            "permutation_null_unsteered_pass": mixed("permutation_null_unsteered_pass"),
            "steered_pass_at_unsteered_cutoff": overall_b,
            "steered_pass_95ci": percentile_ci(boot[("mix", "B")]),
            "permutation_null_steered_pass": mixed("permutation_null_steered_pass"),
            "analytic_exchangeable_null": 0.4,
            "paired_auc_steered_gt_unsteered": overall_auc,
            "paired_auc_95ci": percentile_ci(boot_auc["mix"]),
            "paired_auc_null": 0.5,
            "mix": MIX,
        },
        "definitions": {
            "rank_rule_retention_both_arms": (
                "The deployed filter retains the top 2 of each record's 4 candidates, so "
                "acceptance is 50% for steered and unsteered generation by construction."
            ),
            "unsteered_pass_at_steered_cutoff": (
                "Freeze the lower support score among the two selected steered candidates "
                "for a record, then apply that numeric cutoff to the post-dedup unsteered "
                "candidates of the same record."
            ),
            "steered_pass_at_unsteered_cutoff": (
                "Freeze the second-highest support score among that record's unsteered "
                "candidates (the cutoff a top-2 rule would set on the unsteered pool), then "
                "apply it to the record's four steered candidates."
            ),
            "analytic_exchangeable_null": (
                "If steered and unsteered candidates were exchangeable, a retain-top-2-of-4 "
                "cutoff transferred to an independent group of 4 would pass 2/5 = 40% of "
                "them, not 50%. Rates below 40% in direction A and above 40% in direction B "
                "both indicate an upward shift in steered support scores."
            ),
            "paired_auc_steered_gt_unsteered": (
                "Fraction of within-record steered x unsteered candidate pairs where the "
                "steered candidate has the higher support score (ties = 0.5). Null = 0.5; "
                "threshold-free, so it does not depend on the top-2 rule at all."
            ),
        },
        "ci_method": (
            f"{args.bootstrap} paired record-level bootstrap replicates; seed={args.seed}; "
            f"{args.permutations} within-record label permutations for the null"
        ),
        "profiles": profile_rows,
        "sources": {
            "unsteered_raw": str(Path(args.naive_raw).resolve()),
            "unsteered_post_dedup": str(Path(args.naive_eligible).resolve()),
            "unsteered_support_scores": str(Path(args.naive_support_scores).resolve()),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "symmetric_acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "symmetric_acceptance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)

    lo_a, hi_a = summary["headline"]["unsteered_pass_95ci"]
    lo_b, hi_b = summary["headline"]["steered_pass_95ci"]
    lo_auc, hi_auc = summary["headline"]["paired_auc_95ci"]
    line = (
        "Acceptance under the deployed rank-based filter is 50% for both steered and "
        "unsteered generation by construction, so the two arms cannot differ on that "
        "statistic. Holding the numeric cutoff fixed instead (no-effect point 40%, not "
        f"50%): unsteered candidates clear the steered cutoff at {100 * overall_a:.2f}% "
        f"(95% CI {100 * lo_a:.2f}-{100 * hi_a:.2f}%), while steered candidates clear the "
        f"unsteered cutoff at {100 * overall_b:.2f}% "
        f"(95% CI {100 * lo_b:.2f}-{100 * hi_b:.2f}%). Threshold-free, a steered candidate "
        f"outscores an unsteered candidate from the same source record "
        f"{100 * overall_auc:.2f}% of the time "
        f"(95% CI {100 * lo_auc:.2f}-{100 * hi_auc:.2f}%, null 50%)."
    )
    (output_dir / "rebuttal_line.txt").write_text(line + "\n", encoding="utf-8")
    print(line)
    for row in profile_rows:
        print(
            f"  {row['profile']:<8} "
            f"unsteered@steered-cut={100 * row['unsteered_pass_at_steered_cutoff']:5.2f}% "
            f"(null {100 * row['permutation_null_unsteered_pass']:5.2f}%)  "
            f"steered@unsteered-cut={100 * row['steered_pass_at_unsteered_cutoff']:5.2f}% "
            f"(null {100 * row['permutation_null_steered_pass']:5.2f}%)  "
            f"AUC={row['paired_auc_steered_gt_unsteered']:.4f}"
        )


if __name__ == "__main__":
    main()
