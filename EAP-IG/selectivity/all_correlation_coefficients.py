"""Exp 4-a(ii) robustness: recompute the circuit x utility association matrix
under every reasonable dependence coefficient, not only Spearman rho.

Inputs are exactly the artifacts that ``spearman_and_jaccard.py`` consumes:
per-sample scalar circuit attribution ``high_circuit_score`` joined on
``hf_id`` against the stored ``true_score`` of each utility metric.

Coefficients reported per cell (all on the same n = 11,679 join):
  pearson      linear product-moment r
  spearman     rank product-moment rho (the published number)
  kendall_b    tau-b, tie-corrected concordance
  kendall_c    tau-c, Stuart's variant
  normal       van der Waerden normal-scores r (rank -> Phi^-1 -> Pearson)
  xi           Chatterjee's xi, asymmetric: predictability of utility
               from the circuit score, sensitive to non-monotone dependence
  dcor         distance correlation, zero iff independent, always >= 0
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

COEFFS = (
    ("pearson", "Pearson r", True),
    ("spearman", "Spearman rho", True),
    ("kendall_b", "Kendall tau-b", True),
    ("kendall_c", "Kendall tau-c", True),
    ("normal", "van der Waerden (normal scores) r", True),
    ("xi", "Chatterjee xi", False),
    ("dcor", "Distance correlation", False),
)

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
            if not hf_id:
                raise ValueError(f"{path}:{line_no}: empty hf_id")
            if hf_id in activations:
                raise ValueError(f"{path}:{line_no}: duplicate hf_id {hf_id!r}")
            act = float(row[activation_column])
            util = float(row[utility_column])
            if not (math.isfinite(act) and math.isfinite(util)):
                raise ValueError(f"{path}:{line_no}: non-finite score")
            activations[hf_id] = act
            utilities[hf_id] = util
    return activations, utilities


def normal_scores(values: np.ndarray) -> np.ndarray:
    """van der Waerden transform: average ranks -> inverse normal CDF."""
    ranks = stats.rankdata(values, method="average")
    return stats.norm.ppf(ranks / (len(values) + 1.0))


def chatterjee_xi(x: np.ndarray, y: np.ndarray, *, seed: int, repeats: int = 20):
    """Tie-aware xi_n, averaged over random tie-breaks in x.

    Returns (mean xi, one-sided p-value from sqrt(n) xi -> N(0, 2/5)).
    Asymmetric by construction: measures how well y is a function of x.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    # r_i = #{j : y_j <= y_i}, l_i = #{j : y_j >= y_i}; both tie-aware.
    r_all = stats.rankdata(y, method="max")
    l_all = stats.rankdata(-y, method="max")
    denominator = 2.0 * float(np.sum(l_all * (n - l_all)))
    estimates: List[float] = []
    for _ in range(repeats):
        order = np.lexsort((rng.random(n), x))  # random tie-break within equal x
        r_sorted = r_all[order]
        numerator = n * float(np.sum(np.abs(np.diff(r_sorted))))
        estimates.append(1.0 - numerator / denominator)
    xi = float(np.mean(estimates))
    p_value = float(stats.norm.sf(math.sqrt(n) * xi, scale=math.sqrt(2.0 / 5.0)))
    return xi, p_value, float(np.std(estimates))


def double_centered(values: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - values[None, :])
    row = distances.mean(axis=1, keepdims=True)
    col = distances.mean(axis=0, keepdims=True)
    return distances - row - col + distances.mean()


def distance_correlation(a_centered: np.ndarray, b_centered: np.ndarray) -> float:
    dcov2 = float(np.mean(a_centered * b_centered))
    dvarx = float(np.mean(a_centered * a_centered))
    dvary = float(np.mean(b_centered * b_centered))
    if dvarx <= 0.0 or dvary <= 0.0:
        return float("nan")
    return math.sqrt(max(dcov2, 0.0) / math.sqrt(dvarx * dvary))


def format_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-16:
        return f"<1e-16 ({p:.1e})"
    return f"{p:.2e}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", default=str(DEFAULT_SCORE_DIR))
    parser.add_argument("--activation-column", default="high_circuit_score")
    parser.add_argument("--utility-column", default="true_score")
    parser.add_argument("--expected-samples", type=int, default=11679)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-dcor", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix/correlation_variants')
        ),
    )
    args = parser.parse_args(argv)

    score_dir = Path(args.score_dir)
    activations: Dict[str, Dict[str, float]] = {}
    utilities: Dict[str, Dict[str, float]] = {}
    for metric in UTILITY_KEYS:
        path = score_dir / f"sciq_{metric}_full_train_top250_weighted.csv"
        act, util = load_metric_csv(path, args.activation_column, args.utility_column)
        if len(act) != args.expected_samples:
            raise RuntimeError(f"{path}: n={len(act)}, expected {args.expected_samples}")
        activations[f"{metric}_high"] = act
        utilities[metric] = util
        print(f"[load] {metric}: n={len(act)}")

    common = sorted(set.intersection(*[set(v) for v in utilities.values()]))
    if len(common) != args.expected_samples:
        raise RuntimeError(f"common hf_ids={len(common)}, expected {args.expected_samples}")
    n = len(common)
    print(f"[join] common hf_ids: {n}")

    act_vectors = {
        circuit: np.array([activations[circuit][i] for i in common], dtype=float)
        for circuit in CIRCUIT_ROWS
    }
    util_vectors = {
        metric: np.array([utilities[metric][i] for i in common], dtype=float)
        for metric in UTILITY_KEYS
    }

    centered: Dict[str, np.ndarray] = {}
    if not args.skip_dcor:
        for name, vec in list(act_vectors.items()) + list(util_vectors.items()):
            centered[name] = double_centered(vec)
            print(f"[dcor] double-centered {name}")

    rows: List[Dict[str, object]] = []
    for circuit in CIRCUIT_ROWS:
        x = act_vectors[circuit]
        for metric in UTILITY_KEYS:
            y = util_vectors[metric]
            cell: Dict[str, object] = {
                "circuit": circuit,
                "utility": metric,
                "n": n,
                "n_ties_circuit": int(n - len(np.unique(x))),
                "n_ties_utility": int(n - len(np.unique(y))),
            }
            pearson = stats.pearsonr(x, y)
            cell["pearson"], cell["pearson_p"] = float(pearson[0]), float(pearson[1])
            spearman = stats.spearmanr(x, y)
            cell["spearman"], cell["spearman_p"] = (
                float(spearman.statistic),
                float(spearman.pvalue),
            )
            tau_b = stats.kendalltau(x, y, variant="b")
            cell["kendall_b"], cell["kendall_b_p"] = (
                float(tau_b.statistic),
                float(tau_b.pvalue),
            )
            tau_c = stats.kendalltau(x, y, variant="c")
            cell["kendall_c"], cell["kendall_c_p"] = (
                float(tau_c.statistic),
                float(tau_c.pvalue),
            )
            vdw = stats.pearsonr(normal_scores(x), normal_scores(y))
            cell["normal"], cell["normal_p"] = float(vdw[0]), float(vdw[1])
            xi, xi_p, xi_sd = chatterjee_xi(x, y, seed=args.seed)
            cell["xi"], cell["xi_p"], cell["xi_tiebreak_sd"] = xi, xi_p, xi_sd
            if args.skip_dcor:
                cell["dcor"], cell["dcor_p"] = float("nan"), float("nan")
            else:
                cell["dcor"] = distance_correlation(centered[circuit], centered[metric])
                cell["dcor_p"] = float("nan")  # V-statistic; permutation test not run
            rows.append(cell)
            print(
                f"[cell] {circuit} x {metric}: "
                + " ".join(f"{key}={cell[key]:+.4f}" for key, _, _ in COEFFS)
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "correlation_variants_long.csv"
    field_names = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[csv] {csv_path}")

    lookup = {(row["circuit"], row["utility"]): row for row in rows}
    lines: List[str] = []
    lines.append("# Exp 4-a(ii): circuit x utility association under seven coefficients")
    lines.append("")
    lines.append(
        f"Per-example circuit attribution (`{args.activation_column}`) vs stored utility "
        f"score (`{args.utility_column}`), inner-joined on `hf_id`; n = {n:,} for every cell."
    )
    lines.append("")
    # Master table: every cell under every coefficient, side by side.
    lines.append("## All nine cells, all seven coefficients")
    lines.append("")
    lines.append(
        "| Circuit | Utility | " + " | ".join(title for _, title, _ in COEFFS) + " |"
    )
    lines.append("| --- | --- | " + " | ".join("---:" for _ in COEFFS) + " |")
    for circuit in CIRCUIT_ROWS:
        for metric in UTILITY_KEYS:
            row = lookup[(circuit, metric)]
            own = circuit.rsplit("_", 1)[0] == metric
            cells = []
            for key, _, signed in COEFFS:
                value = float(row[key])
                text = "n/a" if not math.isfinite(value) else (
                    f"{value:+.3f}" if signed else f"{value:.3f}"
                )
                cells.append(f"**{text}**" if own else text)
            label = f"{CIRCUIT_LABEL[circuit]}{' (own)' if own else ''}"
            lines.append(
                f"| {label} | {UTILITY_LABEL[metric]} | " + " | ".join(cells) + " |"
            )
    lines.append("")

    for key, title, signed in COEFFS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Circuit \\ Utility score | " + " | ".join(UTILITY_LABEL[m] for m in UTILITY_KEYS) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in UTILITY_KEYS) + " |")
        for circuit in CIRCUIT_ROWS:
            cells = []
            for metric in UTILITY_KEYS:
                row = lookup[(circuit, metric)]
                value = float(row[key])
                text = "n/a" if not math.isfinite(value) else (
                    f"{value:+.2f}" if signed else f"{value:.3f}"
                )
                p_value = float(row[f"{key}_p"])
                if math.isfinite(p_value) and p_value >= 0.05:
                    text += " (n.s.)"
                if circuit.rsplit("_", 1)[0] == metric:
                    text = f"**{text}**"
                cells.append(text)
            lines.append(f"| {CIRCUIT_LABEL[circuit]} | " + " | ".join(cells) + " |")
        lines.append("")
        p_key = f"{key}_p"
        p_values = [
            float(lookup[(c, m)][p_key]) for c in CIRCUIT_ROWS for m in UTILITY_KEYS
        ]
        if all(math.isfinite(p) for p in p_values):
            lines.append(f"Largest p-value across the nine cells: {format_p(max(p_values))}.")
        else:
            lines.append(
                "Distance correlation is a non-negative V-statistic; no closed-form "
                "p-value is reported (a permutation test was not run)."
            )
        lines.append("")

    # Does each row peak on its own utility score? Coefficient-dependent.
    lines.append("## Diagonal dominance by coefficient")
    lines.append("")
    lines.append("| Coefficient | " + " | ".join(CIRCUIT_LABEL[c] for c in CIRCUIT_ROWS) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in CIRCUIT_ROWS) + " |")
    for key, title, _ in COEFFS:
        cells = []
        for circuit in CIRCUIT_ROWS:
            own_metric = circuit.rsplit("_", 1)[0]
            magnitudes = {
                metric: abs(float(lookup[(circuit, metric)][key]))
                for metric in UTILITY_KEYS
            }
            best = max(magnitudes, key=magnitudes.get)
            cells.append(
                "yes" if best == own_metric else f"no (peaks on {UTILITY_LABEL[best]})"
            )
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "\"yes\" means the largest-magnitude cell in that circuit's row is its own "
        "utility score."
    )
    lines.append("")
    markdown_path = out_dir / "correlation_variants.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[md] {markdown_path}")

    meta = {
        "n": n,
        "score_dir": str(score_dir),
        "activation_column": args.activation_column,
        "utility_column": args.utility_column,
        "coefficients": [key for key, _, _ in COEFFS],
        "seed": args.seed,
        "dcor_computed": not args.skip_dcor,
        "scipy_version": stats.__name__ and __import__("scipy").__version__,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] outputs under {out_dir}")


if __name__ == "__main__":
    main()
