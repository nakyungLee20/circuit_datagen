"""Aggregate run_summary.json files across seeds and configs.

Usage:
  python -m applications.experiment_utils.aggregate --root /path/to/output_root --out-agg result.csv

Walks the {dataset}/{model_key}/{variant}/{run_tag}/ tree, groups
runs by (dataset, model_key, variant), computes mean±std of the
standard metrics, plus a simple paired bootstrap between any two
variants sharing the same set of seeds.
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
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

METRIC_KEYS = [
    "best_val_acc",
    "final_val_acc",
    "test_accuracy",
    "test_accuracy_ood",
    "ece",
    "ece_ood",
    "nll",
    "nll_ood",
]


def _get_deep(obj: Any, path: List[str]) -> Optional[float]:
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def find_metric(summary: Dict[str, Any], key: str) -> Optional[float]:
    """Best-effort lookup: 'metrics.<key>' or top-level or 'eval.<key>'."""
    for path in (["metrics", key], [key], ["eval", key], ["metrics", "eval", key]):
        v = _get_deep(summary, path)
        if v is not None:
            return v
    return None


def scan(root: Path, eval_filename: str = "eval_metrics.json") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in root.rglob("run_summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read {summary_path}: {e}")
            continue
        # Directory template: <root>/.../{dataset}/{model_key}/{variant}/{run_tag}/run_summary.json
        parts = summary_path.parts
        try:
            run_tag = parts[-2]
            variant = parts[-3]
            model_key = parts[-4]
            dataset = parts[-5]
        except IndexError:
            continue
        nested_seed = summary.get("args", {}).get("seed")
        seed = nested_seed if nested_seed is not None else summary.get("seed", "?")
        # Merge in eval_metrics.json (ECE / NLL / OOD live there, not in run_summary).
        eval_path = summary_path.parent / eval_filename
        eval_metrics: Dict[str, Any] = {}
        if eval_path.exists():
            try:
                with eval_path.open("r", encoding="utf-8") as f:
                    eval_metrics = json.load(f)
            except Exception as e:
                print(f"[WARN] failed to read {eval_path}: {e}")
        row = {
            "dataset": dataset,
            "model_key": model_key,
            "variant": variant,
            "run_tag": run_tag,
            "seed": seed,
            "path": str(summary_path),
            "has_eval": bool(eval_metrics),
            "eval_file": eval_filename,
            "eval_checkpoint": eval_metrics.get("checkpoint", ""),
            "summary_mtime_ns": summary_path.stat().st_mtime_ns,
        }
        for k in METRIC_KEYS:
            # eval_metrics.json takes priority for its keys; fall back to run_summary.
            if k in eval_metrics and isinstance(eval_metrics[k], (int, float)):
                row[k] = float(eval_metrics[k])
            else:
                row[k] = find_metric(summary, k)
        rows.append(row)
    return rows


def latest_run_per_variant_seed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one run artifact per (dataset, model, variant, seed).

    Re-running a driver appends timestamped runs under the same result root.
    Treating those as extra independent seeds understates uncertainty, so only
    the newest artifact for each declared seed enters mean/std and bootstrap.
    ``out-per-run`` still records every artifact for auditability.
    """
    selected: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            row["dataset"], row["model_key"], row["variant"], str(row["seed"])
        )
        previous = selected.get(key)
        if previous is None or (
            row.get("summary_mtime_ns", 0), row["path"]
        ) > (
            previous.get("summary_mtime_ns", 0), previous["path"]
        ):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            row["dataset"], row["model_key"], row["variant"], str(row["seed"])
        ),
    )


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def group_and_reduce(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["model_key"], r["variant"])].append(r)
    out: List[Dict[str, Any]] = []
    for (ds, mk, var), rs in sorted(groups.items()):
        seeds = sorted({str(r["seed"]) for r in rs})
        agg = {
            "dataset": ds,
            "model_key": mk,
            "variant": var,
            "n_runs": len(rs),
            "seeds": ",".join(seeds),
        }
        for k in METRIC_KEYS:
            m, s = _mean_std([r[k] for r in rs])
            agg[f"{k}_mean"] = None if m is None else round(m, 4)
            agg[f"{k}_std"] = None if s is None else round(s, 4)
        out.append(agg)
    return out


def paired_bootstrap(
    a: List[float], b: List[float], n_boot: int = 10000, seed: int = 0
) -> Dict[str, float]:
    """Two-sided paired bootstrap for mean(a) − mean(b)."""
    if len(a) != len(b) or not a:
        return {
            "delta_mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_two_sided": float("nan"),
            "n": 0,
        }
    rng = random.Random(seed)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    obs = sum(diffs) / len(diffs)
    boots: List[float] = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    # centered null
    centered = [b_ - obs for b_ in boots]
    ge = sum(1 for c in centered if abs(c) >= abs(obs))
    p = (ge + 1) / (n_boot + 1)
    boots.sort()
    lo_idx = max(0, int(0.025 * (n_boot - 1)))
    hi_idx = min(n_boot - 1, int(0.975 * (n_boot - 1)))
    return {
        "delta_mean": obs,
        "ci95_low": boots[lo_idx],
        "ci95_high": boots[hi_idx],
        "p_two_sided": p,
        "n": n,
    }


def _per_seed_metric(rows: List[Dict[str, Any]], metric: str) -> Dict[str, float]:
    """Collapse accidental repeated runs to one value per seed by their mean."""
    by_seed: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            by_seed[str(row["seed"])].append(value)
    return {
        seed: sum(values) / len(values)
        for seed, values in by_seed.items()
        if values
    }


def paired_bootstrap_table(
    rows: List[Dict[str, Any]], *, n_boot: int = 10000, bootstrap_seed: int = 0
) -> List[Dict[str, Any]]:
    """Pair every variant pair that has exactly the same completed seed set."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model_key"], row["variant"])].append(row)

    dataset_models = sorted({(key[0], key[1]) for key in groups})
    output: List[Dict[str, Any]] = []
    for dataset, model_key in dataset_models:
        variants = sorted(
            key[2] for key in groups if key[0] == dataset and key[1] == model_key
        )
        for variant_a, variant_b in combinations(variants, 2):
            rows_a = groups[(dataset, model_key, variant_a)]
            rows_b = groups[(dataset, model_key, variant_b)]
            seeds_a = {str(row["seed"]) for row in rows_a if str(row["seed"]) != "?"}
            seeds_b = {str(row["seed"]) for row in rows_b if str(row["seed"]) != "?"}
            if not seeds_a or seeds_a != seeds_b:
                continue
            ordered_seeds = sorted(seeds_a)
            for metric_index, metric in enumerate(METRIC_KEYS):
                values_a = _per_seed_metric(rows_a, metric)
                values_b = _per_seed_metric(rows_b, metric)
                paired_seeds = [
                    seed for seed in ordered_seeds
                    if seed in values_a and seed in values_b
                ]
                if len(paired_seeds) < 2:
                    continue
                a = [values_a[seed] for seed in paired_seeds]
                b = [values_b[seed] for seed in paired_seeds]
                stats = paired_bootstrap(
                    a,
                    b,
                    n_boot=n_boot,
                    seed=bootstrap_seed + len(output) * len(METRIC_KEYS) + metric_index,
                )
                output.append(
                    {
                        "dataset": dataset,
                        "model_key": model_key,
                        "variant_a": variant_a,
                        "variant_b": variant_b,
                        "metric": metric,
                        "n_pairs": stats["n"],
                        "seeds": ",".join(paired_seeds),
                        "mean_a": round(sum(a) / len(a), 6),
                        "mean_b": round(sum(b) / len(b), 6),
                        "delta_mean_a_minus_b": round(stats["delta_mean"], 6),
                        "ci95_low": round(stats["ci95_low"], 6),
                        "ci95_high": round(stats["ci95_high"], 6),
                        "p_two_sided": round(stats["p_two_sided"], 6),
                        "n_boot": n_boot,
                    }
                )
    return output


def dump_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[WARN] no rows to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] wrote {len(rows)} rows to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output_root scanned for run_summary.json")
    ap.add_argument("--out-per-run", default="", help="CSV of individual runs")
    ap.add_argument("--out-agg", default="", help="CSV of aggregated (mean±std) rows")
    ap.add_argument("--out-bootstrap", default="", help="CSV of paired-bootstrap variant comparisons")
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--eval-filename", default="eval_metrics.json",
                    help="per-run metrics file to merge (use eval_metrics_final.json "
                         "for the final-checkpoint tables)")
    ap.add_argument("--require-eval", action="store_true",
                    help="drop runs missing --eval-filename instead of falling back to "
                         "run_summary.json (keeps best/final tables from silently mixing)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    rows = scan(root, eval_filename=args.eval_filename)
    print(f"[SCAN] found {len(rows)} run_summary.json files under {root} "
          f"(eval file: {args.eval_filename})")
    if args.require_eval:
        kept = [r for r in rows if r["has_eval"]]
        if len(kept) != len(rows):
            print(f"[SELECT] dropped {len(rows) - len(kept)} run(s) with no {args.eval_filename}")
        rows = kept

    if args.out_per_run:
        dump_csv(Path(args.out_per_run), rows)
    aggregate_rows = latest_run_per_variant_seed(rows)
    if len(aggregate_rows) != len(rows):
        print(
            f"[SELECT] latest per variant+seed: {len(aggregate_rows)} of "
            f"{len(rows)} run artifacts"
        )
    agg = group_and_reduce(aggregate_rows)
    if args.out_agg:
        dump_csv(Path(args.out_agg), agg)
    else:
        for r in agg:
            print(json.dumps(r, ensure_ascii=False))
    if args.out_bootstrap:
        bootstrap_rows = paired_bootstrap_table(
            aggregate_rows,
            n_boot=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        dump_csv(Path(args.out_bootstrap), bootstrap_rows)


if __name__ == "__main__":
    main()
