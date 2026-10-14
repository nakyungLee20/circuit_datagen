"""Exp 4-a(ii) restricted to each circuit's own construction set.

Each *-high circuit was discovered from the top-15% bucket of its own utility
(1,751 of 11,679 SciQ train examples). Evaluating its per-example attribution
against utility over the full population therefore mixes in 10k examples the
circuit was never derived from. This script recomputes the 3x3 tables with each
row restricted to that circuit's own high bucket.

Caveat this script quantifies rather than hides: restricting to a top-15%
bucket truncates the row's own utility range, so the diagonal cell is subject
to range-restriction attenuation that the off-diagonal cells do not face. The
per-row utility SD ratio (bucket vs full) is reported for exactly this reason.
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
from typing import Dict, List, Sequence, Set, Tuple

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
DEFAULT_BUCKETS_ROOT = Path(
    str(_REPO_ROOT / 'data_gens/buckets/application/Qwen__Qwen2.5-1.5B-Instruct')
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


def load_bucket_ids(buckets_root: Path, metric: str, dataset: str, side: str) -> Set[str]:
    path = buckets_root / metric / dataset / f"{metric}_{dataset}_{side}.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    items = obj["items"] if isinstance(obj, dict) and "items" in obj else obj
    ids = {str(row["hf_id"]).strip() for row in items}
    if not ids:
        raise ValueError(f"{path}: empty bucket")
    return ids


def three_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y, variant="b")
    return {
        "pearson": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_b": float(kendall.statistic),
        "kendall_b_p": float(kendall.pvalue),
    }


def render_table(
    lookup: Dict[Tuple[str, str], Dict[str, float]], key: str, digits: int = 3
) -> List[str]:
    lines = [
        "| Circuit \\ Utility | " + " | ".join(UTILITY_LABEL[m] for m in UTILITY_KEYS) + " |",
        "| --- | " + " | ".join("---:" for _ in UTILITY_KEYS) + " |",
    ]
    for circuit in CIRCUIT_ROWS:
        cells = []
        for metric in UTILITY_KEYS:
            row = lookup[(circuit, metric)]
            text = f"{float(row[key]):+.{digits}f}"
            if float(row[f"{key}_p"]) >= 0.05:
                text += " (n.s.)"
            if circuit.rsplit("_", 1)[0] == metric:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {CIRCUIT_LABEL[circuit]} | " + " | ".join(cells) + " |")
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", default=str(DEFAULT_SCORE_DIR))
    parser.add_argument("--buckets-root", default=str(DEFAULT_BUCKETS_ROOT))
    parser.add_argument("--dataset", default="sciq")
    parser.add_argument("--side", default="high", choices=("high", "low"))
    parser.add_argument("--activation-column", default="high_circuit_score")
    parser.add_argument("--utility-column", default="true_score")
    parser.add_argument(
        "--out-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix/high_bucket_correlation')
        ),
    )
    args = parser.parse_args(argv)

    score_dir = Path(args.score_dir)
    buckets_root = Path(args.buckets_root)

    activations: Dict[str, Dict[str, float]] = {}
    utilities: Dict[str, Dict[str, float]] = {}
    for metric in UTILITY_KEYS:
        path = score_dir / f"sciq_{metric}_full_train_top250_weighted.csv"
        act, util = load_metric_csv(path, args.activation_column, args.utility_column)
        activations[f"{metric}_high"] = act
        utilities[metric] = util
    common = sorted(set.intersection(*[set(v) for v in utilities.values()]))
    print(f"[join] full population n={len(common)}")

    bucket_ids = {
        metric: load_bucket_ids(buckets_root, metric, args.dataset, args.side)
        for metric in UTILITY_KEYS
    }
    for metric, ids in bucket_ids.items():
        outside = ids - set(common)
        if outside:
            raise RuntimeError(f"{metric} bucket has {len(outside)} ids outside the scored set")
        print(f"[bucket] {metric}-{args.side}: n={len(ids)}")

    rows: List[Dict[str, object]] = []
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for circuit in CIRCUIT_ROWS:
        own_metric = circuit.rsplit("_", 1)[0]
        subset = sorted(bucket_ids[own_metric])
        x = np.array([activations[circuit][i] for i in subset], dtype=float)
        for metric in UTILITY_KEYS:
            y = np.array([utilities[metric][i] for i in subset], dtype=float)
            y_full = np.array([utilities[metric][i] for i in common], dtype=float)
            x_full = np.array([activations[circuit][i] for i in common], dtype=float)
            record: Dict[str, object] = {
                "circuit": circuit,
                "utility": metric,
                "bucket": f"{own_metric}_{args.side}",
                "n": len(subset),
                "n_full": len(common),
                "utility_sd_bucket": float(y.std(ddof=1)),
                "utility_sd_full": float(y_full.std(ddof=1)),
                "utility_sd_ratio": float(y.std(ddof=1) / y_full.std(ddof=1)),
                "utility_unique_bucket": int(len(np.unique(y))),
            }
            record.update(three_correlations(x, y))
            full = three_correlations(x_full, y_full)
            record.update({f"full_{k}": v for k, v in full.items()})
            rows.append(record)
            lookup[(circuit, metric)] = record
            print(
                f"[cell] {circuit} x {metric} (n={len(subset)}): "
                f"r={record['pearson']:+.3f} rho={record['spearman']:+.3f} "
                f"tau={record['kendall_b']:+.3f} | full rho={full['spearman']:+.3f} "
                f"| util sd ratio={record['utility_sd_ratio']:.3f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.side}_bucket_correlation_long.csv"
    field_names = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Exp 4-a(ii) restricted to each circuit's own {args.side} bucket",
        "",
        f"Row `C-high` uses only the {len(bucket_ids['aum'])} examples in C's "
        f"{args.side} bucket (top-15% by C's utility) — the same examples the "
        "circuit was discovered from. Columns are the three utility scores on "
        "that same subset.",
        "",
    ]
    for key, title in (
        ("pearson", "Pearson r"),
        ("spearman", "Spearman rho"),
        ("kendall_b", "Kendall tau-b"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(render_table(lookup, key))
        lines.append("")
    lines.append("## Range restriction diagnostic")
    lines.append("")
    lines.append("| Circuit row | Utility | SD in bucket | SD full | ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for circuit in CIRCUIT_ROWS:
        for metric in UTILITY_KEYS:
            row = lookup[(circuit, metric)]
            lines.append(
                f"| {CIRCUIT_LABEL[circuit]} | {UTILITY_LABEL[metric]} | "
                f"{float(row['utility_sd_bucket']):.4f} | "
                f"{float(row['utility_sd_full']):.4f} | "
                f"{float(row['utility_sd_ratio']):.3f} |"
            )
    lines.append("")
    (out_dir / f"{args.side}_bucket_correlation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[ok] outputs under {out_dir}")


if __name__ == "__main__":
    main()
