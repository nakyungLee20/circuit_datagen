"""Pre-selection support-distribution analysis for Experiment 2.

For each of easy/hard/helpful, this script applies the exact same generation-
time recipe weights to (a) the explicitly re-scored neutral 20k candidates and
(b) the corresponding clean steered 20k candidates.  It never substitutes
empirical difficulty or a zero fallback for the circuit support score.
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
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from applications.experiment_utils.io import (  # noqa: E402
    NAIVE_RAW_PATH,
    find_full_steered_pool_dir,
    read_jsonl,
    write_json,
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
EXP2_WORK_ROOT = _SUPPLEMENTAL_ROOT / "steering_selection" / "work_exp2_steering_selection"


def cohens_d(steered: Sequence[float], unsteered: Sequence[float]) -> float:
    if len(steered) < 2 or len(unsteered) < 2:
        return float("nan")
    var_s = float(np.var(steered, ddof=1))
    var_u = float(np.var(unsteered, ddof=1))
    denom = len(steered) + len(unsteered) - 2
    pooled = math.sqrt(
        ((len(steered) - 1) * var_s + (len(unsteered) - 1) * var_u) / denom
    )
    return float((np.mean(steered) - np.mean(unsteered)) / pooled) if pooled else float("nan")


def ks_two_sample(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    from scipy import stats

    stat, pvalue = stats.ks_2samp(a, b)
    return float(stat), float(pvalue)


def summarize_pair(unsteered: Sequence[float], steered: Sequence[float]) -> Dict[str, float | int]:
    if not unsteered or not steered:
        raise ValueError("both score arrays must be non-empty")
    ks_stat, ks_pvalue = ks_two_sample(unsteered, steered)
    mean_u = float(np.mean(unsteered))
    mean_s = float(np.mean(steered))
    return {
        "n_unsteered": len(unsteered),
        "n_steered": len(steered),
        "mean_unsteered": mean_u,
        "std_unsteered": float(np.std(unsteered, ddof=1)),
        "mean_steered": mean_s,
        "std_steered": float(np.std(steered, ddof=1)),
        "mean_shift_steered_minus_unsteered": mean_s - mean_u,
        "cohens_d_steered_vs_unsteered": cohens_d(steered, unsteered),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
    }


def save_histograms(
    pairs: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
    for axis, (profile, (unsteered, steered)) in zip(axes[0], pairs.items()):
        low = min(min(unsteered), min(steered))
        high = max(max(unsteered), max(steered))
        bins = np.linspace(low, high, 60) if high > low else np.linspace(low - 1, high + 1, 3)
        axis.hist(unsteered, bins=bins, alpha=0.55, density=True, label="unsteered")
        axis.hist(steered, bins=bins, alpha=0.55, density=True, label="steered")
        axis.set_title(f"{profile} support $s_k$")
        axis.set_xlabel("exact weighted support score")
        axis.set_ylabel("density")
        axis.legend()
    figure.suptitle("Pre-selection support distribution: steered vs unsteered")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"[png] {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive-raw", default=str(NAIVE_RAW_PATH))
    parser.add_argument(
        "--naive-support-scores",
        default=str(EXP2_WORK_ROOT / "_support" / "naive_support_scores.jsonl"),
    )
    parser.add_argument("--profiles", default="easy,hard,helpful")
    parser.add_argument(
        "--result-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/steering_selection/result_exp2_steering_selection')
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    profiles = [piece.strip() for piece in args.profiles.split(",") if piece.strip()]
    if profiles != list(PROFILES):
        raise ValueError(f"profiles must be exactly {','.join(PROFILES)}")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    naive = read_jsonl(args.naive_raw)
    if args.limit:
        naive = naive[: args.limit]
    score_index = load_support_index(args.naive_support_scores)
    if args.limit:
        limited_ids = {candidate_id(row) for row in naive}
        score_index = {cid: vector for cid, vector in score_index.items() if cid in limited_ids}
    validate_score_coverage(naive, score_index, label="naive raw")
    naive_vectors = [score_index[candidate_id(row)] for row in naive]
    print(f"[input] naive={len(naive)} exact support vectors")

    profile_stats: Dict[str, Mapping] = {}
    histogram_pairs: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
    clean_inputs: Dict[str, str] = {}
    for profile in profiles:
        directory = find_full_steered_pool_dir(profile)
        path = directory / f"{profile}_raw_candidates.jsonl"
        steered_rows = read_jsonl(path)
        if args.limit:
            steered_rows = steered_rows[: args.limit]
        weights = recipe_weights_from_rows(steered_rows, profile)
        steered_vectors = [support_components(row) for row in steered_rows]
        unsteered_scores = [weighted_support_score(vector, weights) for vector in naive_vectors]
        steered_scores = [weighted_support_score(vector, weights) for vector in steered_vectors]
        profile_stats[profile] = {
            "recipe": profile,
            "support_definition": (
                "weighted sum of base_logprob/easy_support/hard_support/"
                "helpful_support/shortcut_support using this recipe's stored "
                "selection_score_weights"
            ),
            "source_file_unsteered": str(Path(args.naive_raw).resolve()),
            "source_file_steered": str(path.resolve()),
            "n_raw_unsteered": len(naive),
            "n_raw_steered": len(steered_rows),
            "recipe_weights": weights,
            **summarize_pair(unsteered_scores, steered_scores),
        }
        histogram_pairs[profile] = (unsteered_scores, steered_scores)
        clean_inputs[profile] = str(path)
        print(
            f"[{profile}] naive={len(unsteered_scores)} steered={len(steered_scores)} "
            f"shift={profile_stats[profile]['mean_shift_steered_minus_unsteered']:+.6f}"
        )

    summary = {
        "score_definition": (
            "generation-time circuit support components evaluated with activation addition "
            "+ attention steering, then combined with the corresponding profile's stored "
            "selection_score_weights"
        ),
        "comparison_stage": "raw candidates before any top-half/diversity/random selection",
        "naive_raw": str(Path(args.naive_raw).resolve()),
        "naive_support_scores": str(Path(args.naive_support_scores).resolve()),
        "clean_steered_raw": clean_inputs,
        "profiles": profile_stats,
    }
    output_json = result_dir / "distribution_stats.json"
    write_json(output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    save_histograms(histogram_pairs, result_dir / "distribution_hist.png")


if __name__ == "__main__":
    main()
