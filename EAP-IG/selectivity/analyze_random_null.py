"""Turn the per-trial random-circuit scores into calibrated p-values.

Consumes the .npz files written by ``random_circuit_null.py``. For every
(circuit row, utility column) cell it compares the real circuit's correlation
against the distribution of the same correlation computed from 200 random
circuits, so the question "is rho = 0.17 large?" gets an empirical answer
instead of an asymptotic one.

Reported per cell:
  rho_real         the statistic under test
  null_mean/sd     centre and spread of the 200 random-circuit correlations
  z                (rho_real - null_mean) / null_sd
  p_perm           two-sided (#{|null| >= |real|} + 1) / (n_trials + 1)
  pct              where rho_real falls in the null, as a percentile

Both nulls are reported: `perm` (same weights, random edges) answers "is this
edge set special", `top15k` (random edges from the circuit's own top-15000)
answers "are the top-250 special among plausible edges".
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
CIRCUIT_ROWS = ("aum", "el2n", "grad_align")
UTILITY_LABEL = {"aum": "AUM", "el2n": "EL2N", "grad_align": "GradAlign"}
CIRCUIT_LABEL = {"aum": "AUM-high", "el2n": "EL2N-high", "grad_align": "GA-high"}

CORRELATIONS = {
    "spearman": lambda a, b: float(stats.spearmanr(a, b).statistic),
    "pearson": lambda a, b: float(stats.pearsonr(a, b)[0]),
    "kendall_b": lambda a, b: float(stats.kendalltau(a, b, variant="b").statistic),
}


def load_null(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def null_stats(real: float, null: np.ndarray) -> Dict[str, float]:
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return {k: float("nan") for k in ("null_mean", "null_sd", "z", "p_perm", "pct")}
    mean, sd = float(finite.mean()), float(finite.std(ddof=1))
    extreme = int((np.abs(finite) >= abs(real)).sum())
    return {
        "null_mean": mean,
        "null_sd": sd,
        "z": (real - mean) / sd if sd > 0 else float("nan"),
        "p_perm": (extreme + 1) / (finite.size + 1),
        "pct": float((finite < real).mean() * 100.0),
        "n_trials_used": float(finite.size),
    }


def format_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    return f"{p:.4f}" if p >= 1e-4 else f"<{1e-4:.0e}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--null-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix/random_null')
        ),
    )
    parser.add_argument("--dataset", default="sciq")
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument(
        "--score-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix/activation_scores')
        ),
        help="Full-population CSVs supplying the utility columns.",
    )
    parser.add_argument(
        "--correlation", choices=tuple(CORRELATIONS) + ("all",), default="all"
    )
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)

    null_dir = Path(args.null_dir)
    out_dir = Path(args.out_dir) if args.out_dir else null_dir

    # Each npz holds one circuit row: its real score, its per-trial null scores,
    # and its own utility. Cross-metric columns need the other files' utilities,
    # joined on hf_id.
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for metric in CIRCUIT_ROWS:
        path = null_dir / f"{args.dataset}_{metric}_top{args.topk_edges}_null.npz"
        if not path.is_file():
            print(f"[skip] missing {path}")
            continue
        data[metric] = load_null(path)
        print(f"[load] {metric}: n={len(data[metric]['hf_id'])} "
              f"nulls={[k for k in data[metric] if k.startswith('null_')]}")
    if not data:
        raise SystemExit("no null artifacts found")

    # Utility columns must cover whatever subset each row was scored on, so read
    # them from the full-population activation CSVs rather than from the npz
    # (whose true_score only spans that row's own bucket). Verified against the
    # npz for the row's own metric below.
    utility_by_id: Dict[str, Dict[str, float]] = {}
    for metric in UTILITY_KEYS:
        path = Path(args.score_dir) / f"{args.dataset}_{metric}_full_train_top250_weighted.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            utility_by_id[metric] = {
                str(row["hf_id"]).strip(): float(row["true_score"])
                for row in csv.DictReader(handle)
            }
        print(f"[utility] {metric}: {len(utility_by_id[metric])} rows from {path.name}")
    for metric, payload in data.items():
        source = utility_by_id[metric]
        drift = max(
            abs(source[str(h)] - float(v))
            for h, v in zip(payload["hf_id"], payload["true_score"])
            if str(h) in source
        )
        if drift > 1e-9:
            raise RuntimeError(f"{metric}: utility CSV disagrees with npz by {drift}")
    print("[check] npz true_score matches the activation CSVs exactly")

    null_names = sorted(
        {k[5:] for payload in data.values() for k in payload if k.startswith("null_")}
    )
    corr_names = list(CORRELATIONS) if args.correlation == "all" else [args.correlation]

    rows: List[Dict[str, object]] = []
    for circuit in CIRCUIT_ROWS:
        if circuit not in data:
            continue
        payload = data[circuit]
        ids = [str(h) for h in payload["hf_id"]]
        real_score = payload["high_circuit_score"]
        for utility in UTILITY_KEYS:
            if utility not in utility_by_id:
                continue
            source = utility_by_id[utility]
            keep = [i for i, h in enumerate(ids) if h in source]
            if len(keep) < 10:
                print(f"[skip] {circuit} x {utility}: only {len(keep)} joined ids")
                continue
            idx = np.array(keep, dtype=int)
            y = np.array([source[ids[i]] for i in keep], dtype=float)
            x = real_score[idx]
            for corr_name in corr_names:
                fn = CORRELATIONS[corr_name]
                real = fn(x, y)
                record: Dict[str, object] = {
                    "circuit": circuit,
                    "utility": utility,
                    "correlation": corr_name,
                    "n": len(keep),
                    "rho_real": real,
                }
                for null_name in null_names:
                    key = f"null_{null_name}"
                    if key not in payload:
                        continue
                    trials = payload[key][idx]  # [n_examples, n_trials]
                    null_rhos = np.array(
                        [fn(trials[:, t], y) for t in range(trials.shape[1])],
                        dtype=float,
                    )
                    for stat_name, value in null_stats(real, null_rhos).items():
                        record[f"{null_name}_{stat_name}"] = value
                rows.append(record)
                summary = "  ".join(
                    f"{nn}: z={record.get(f'{nn}_z', float('nan')):+.1f} "
                    f"p={format_p(float(record.get(f'{nn}_p_perm', float('nan'))))}"
                    for nn in null_names
                )
                print(f"[cell] {circuit} x {utility} [{corr_name}] "
                      f"real={real:+.4f}  {summary}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "random_null_summary.csv"
    field_names: List[str] = []
    for record in rows:
        for key in record:
            if key not in field_names:
                field_names.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] {csv_path}")

    lookup = {(r["circuit"], r["utility"], r["correlation"]): r for r in rows}
    lines = [
        "# Exp 4-a(ii): real circuit vs random-circuit null",
        "",
        "Each cell: observed correlation, then how far it sits from the "
        "distribution of the same correlation over 200 random circuits.",
        "",
    ]
    for corr_name in corr_names:
        for null_name in null_names:
            lines.append(f"## {corr_name} — null: `{null_name}`")
            lines.append("")
            lines.append(
                "| Circuit \\ Utility | "
                + " | ".join(UTILITY_LABEL[m] for m in UTILITY_KEYS) + " |"
            )
            lines.append("| --- | " + " | ".join("---:" for _ in UTILITY_KEYS) + " |")
            for circuit in CIRCUIT_ROWS:
                cells = []
                for utility in UTILITY_KEYS:
                    record = lookup.get((circuit, utility, corr_name))
                    if record is None:
                        cells.append("—")
                        continue
                    real = float(record["rho_real"])
                    z = float(record.get(f"{null_name}_z", float("nan")))
                    p = float(record.get(f"{null_name}_p_perm", float("nan")))
                    text = f"{real:+.3f}<br>z={z:+.1f}, p={format_p(p)}"
                    if circuit == utility:
                        text = f"**{text}**"
                    cells.append(text)
                lines.append(f"| {CIRCUIT_LABEL[circuit]} | " + " | ".join(cells) + " |")
            lines.append("")
            sds = [
                float(lookup[(c, u, corr_name)][f"{null_name}_null_sd"])
                for c in CIRCUIT_ROWS for u in UTILITY_KEYS
                if (c, u, corr_name) in lookup
                and f"{null_name}_null_sd" in lookup[(c, u, corr_name)]
            ]
            if sds:
                lines.append(
                    f"Null spread across cells: SD ranges "
                    f"{min(sds):.4f} – {max(sds):.4f}. A |rho| below roughly "
                    f"{2 * max(sds):.3f} is not separable from a random circuit."
                )
                lines.append("")
    (out_dir / "random_null_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[md] {out_dir / 'random_null_summary.md'}")


if __name__ == "__main__":
    main()
