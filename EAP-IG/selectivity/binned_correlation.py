"""Exp 4-a(ii) at the group level: bin examples by utility, then correlate.

Motivation: the circuits are built from *bucket-level* (high/low) contrasts, but
the per-example Spearman evaluates them at the individual level. Binning by
utility decile averages away per-example attribution noise and asks the weaker,
better-matched question: does mean circuit attribution move monotonically with
utility level?

Binned correlations are inflated by construction (n drops to the bin count and
within-bin noise is averaged out), so this script also reports:
  * a permutation null (utility shuffled, re-binned) giving a calibrated p-value
    and the inflation baseline;
  * eta^2, the between-bin share of attribution variance, which is a binning-
    robust effect size directly comparable to the per-example r^2.
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
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

UTILITY_KEYS = ("aum", "el2n", "grad_align")
CIRCUIT_ROWS = ("aum_high", "el2n_high", "grad_align_high")
UTILITY_LABEL = {"aum": "AUM", "el2n": "EL2N", "grad_align": "GradAlign"}
CIRCUIT_LABEL = {
    "aum_high": "AUM-high",
    "el2n_high": "EL2N-high",
    "grad_align_high": "GA-high",
}

DEFAULT_SCORE_DIR = Path(
    str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix/activation_scores')
)


def load_metric_csv(
    path: Path, activation_column: str, utility_column: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    activations: Dict[str, float] = {}
    utilities: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(
            {"hf_id", activation_column, utility_column} - set(reader.fieldnames or [])
        )
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for line_no, row in enumerate(reader, start=2):
            hf_id = str(row["hf_id"]).strip()
            act, util = float(row[activation_column]), float(row[utility_column])
            if not (math.isfinite(act) and math.isfinite(util)):
                raise ValueError(f"{path}:{line_no}: non-finite score")
            activations[hf_id] = act
            utilities[hf_id] = util
    return activations, utilities


def rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bins over average ranks.

    Rank-based rather than value-based, so heavy utility tie blocks (AUM has
    only 541 distinct values over 11,679 examples) still yield balanced bins.
    """
    ranks = stats.rankdata(values, method="ordinal")
    return np.minimum(((ranks - 1) * n_bins) // len(values), n_bins - 1)


def trimmed_mean(values: np.ndarray, proportion: float) -> float:
    return float(stats.trim_mean(values, proportion))


def bin_profile(
    activation: np.ndarray, utility: np.ndarray, n_bins: int, trim: float
) -> Dict[str, np.ndarray]:
    bins = rank_bins(utility, n_bins)
    rows = {
        key: np.empty(n_bins, dtype=float)
        for key in ("utility_mean", "act_mean", "act_median", "act_trimmed", "count")
    }
    for b in range(n_bins):
        mask = bins == b
        rows["count"][b] = int(mask.sum())
        rows["utility_mean"][b] = float(utility[mask].mean())
        rows["act_mean"][b] = float(activation[mask].mean())
        rows["act_median"][b] = float(np.median(activation[mask]))
        rows["act_trimmed"][b] = trimmed_mean(activation[mask], trim)
    rows["bins"] = bins
    return rows


def eta_squared(activation: np.ndarray, bins: np.ndarray, n_bins: int) -> float:
    """Between-bin share of attribution variance; binning-robust effect size."""
    grand = activation.mean()
    total_ss = float(((activation - grand) ** 2).sum())
    if total_ss <= 0.0:
        return float("nan")
    between_ss = 0.0
    for b in range(n_bins):
        group = activation[bins == b]
        if group.size:
            between_ss += group.size * (group.mean() - grand) ** 2
    return float(between_ss / total_ss)


def binned_correlations(
    utility_means: np.ndarray, act_summary: np.ndarray
) -> Dict[str, float]:
    pearson = stats.pearsonr(utility_means, act_summary)
    spearman = stats.spearmanr(utility_means, act_summary)
    kendall = stats.kendalltau(utility_means, act_summary, variant="b")
    return {
        "pearson": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_b": float(kendall.statistic),
        "kendall_b_p": float(kendall.pvalue),
    }


def permutation_null(
    activation: np.ndarray,
    utility: np.ndarray,
    n_bins: int,
    summary: str,
    trim: float,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    """Shuffle utility, re-bin, recompute. Calibrates the binning inflation."""
    rng = np.random.default_rng(seed)
    observed = bin_profile(activation, utility, n_bins, trim)
    obs_r = float(
        stats.pearsonr(observed["utility_mean"], observed[summary])[0]
    )
    null_abs: List[float] = []
    extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(utility)
        profile = bin_profile(activation, shuffled, n_bins, trim)
        null_r = float(stats.pearsonr(profile["utility_mean"], profile[summary])[0])
        null_abs.append(abs(null_r))
        if abs(null_r) >= abs(obs_r):
            extreme += 1
    return {
        "observed_r": obs_r,
        "null_mean_abs_r": float(np.mean(null_abs)),
        "null_p95_abs_r": float(np.percentile(null_abs, 95)),
        "perm_p": (extreme + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", default=str(DEFAULT_SCORE_DIR))
    parser.add_argument("--activation-column", default="high_circuit_score")
    parser.add_argument("--utility-column", default="true_score")
    parser.add_argument(
        "--bins", type=int, nargs="+", default=[10, 20, 50],
        help="Bin counts to report. 10 = deciles (headline).",
    )
    parser.add_argument("--headline-bins", type=int, default=10)
    parser.add_argument(
        "--summary", choices=("act_mean", "act_median", "act_trimmed"),
        default="act_trimmed",
        help="Within-bin summary. Trimmed mean resists the kurtosis-170 tail.",
    )
    parser.add_argument("--trim", type=float, default=0.1)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix/binned_correlation')
        ),
    )
    args = parser.parse_args(argv)

    score_dir = Path(args.score_dir)
    activations: Dict[str, Dict[str, float]] = {}
    utilities: Dict[str, Dict[str, float]] = {}
    for metric in UTILITY_KEYS:
        path = score_dir / f"sciq_{metric}_full_train_top250_weighted.csv"
        act, util = load_metric_csv(path, args.activation_column, args.utility_column)
        activations[f"{metric}_high"] = act
        utilities[metric] = util
    common = sorted(set.intersection(*[set(v) for v in utilities.values()]))
    n = len(common)
    print(f"[join] n={n}")

    act_vectors = {
        c: np.array([activations[c][i] for i in common], dtype=float)
        for c in CIRCUIT_ROWS
    }
    util_vectors = {
        m: np.array([utilities[m][i] for i in common], dtype=float)
        for m in UTILITY_KEYS
    }

    cells: List[Dict[str, object]] = []
    profiles: List[Dict[str, object]] = []
    for circuit in CIRCUIT_ROWS:
        x = act_vectors[circuit]
        for metric in UTILITY_KEYS:
            y = util_vectors[metric]
            per_example = {
                "pearson": float(stats.pearsonr(x, y)[0]),
                "spearman": float(stats.spearmanr(x, y).statistic),
                "kendall_b": float(stats.kendalltau(x, y, variant="b").statistic),
            }
            for n_bins in args.bins:
                profile = bin_profile(x, y, n_bins, args.trim)
                record: Dict[str, object] = {
                    "circuit": circuit,
                    "utility": metric,
                    "n_bins": n_bins,
                    "summary": args.summary,
                    "n": n,
                    "eta_squared": eta_squared(x, profile["bins"], n_bins),
                    "per_example_pearson": per_example["pearson"],
                    "per_example_spearman": per_example["spearman"],
                    "per_example_kendall_b": per_example["kendall_b"],
                }
                record.update(
                    binned_correlations(profile["utility_mean"], profile[args.summary])
                )
                if n_bins == args.headline_bins:
                    record.update(
                        permutation_null(
                            x, y, n_bins, args.summary, args.trim,
                            args.n_perm, args.seed,
                        )
                    )
                    for b in range(n_bins):
                        profiles.append({
                            "circuit": circuit,
                            "utility": metric,
                            "bin": b + 1,
                            "count": int(profile["count"][b]),
                            "utility_mean": profile["utility_mean"][b],
                            "act_mean": profile["act_mean"][b],
                            "act_median": profile["act_median"][b],
                            "act_trimmed": profile["act_trimmed"][b],
                        })
                cells.append(record)
                print(
                    f"[cell] {circuit} x {metric} bins={n_bins}: "
                    f"r={record['pearson']:+.3f} rho={record['spearman']:+.3f} "
                    f"tau={record['kendall_b']:+.3f} eta2={record['eta_squared']:.4f}"
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "binned_correlation_long.csv"
    field_names: List[str] = []
    for record in cells:
        for key in record:
            if key not in field_names:
                field_names.append(key)
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, restval="")
        writer.writeheader()
        writer.writerows(cells)
    profile_path = out_dir / "bin_profiles.csv"
    with profile_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0].keys()))
        writer.writeheader()
        writer.writerows(profiles)
    (out_dir / "meta.json").write_text(
        json.dumps({
            "n": n,
            "activation_column": args.activation_column,
            "utility_column": args.utility_column,
            "bins": args.bins,
            "headline_bins": args.headline_bins,
            "summary": args.summary,
            "trim": args.trim,
            "n_perm": args.n_perm,
            "seed": args.seed,
            "score_dir": str(score_dir),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] outputs under {out_dir}")


if __name__ == "__main__":
    main()
