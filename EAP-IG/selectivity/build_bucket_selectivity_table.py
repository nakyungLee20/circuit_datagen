"""Bucket-restricted Exp4 selectivity table with a two-checkpoint GradAlign column.

Two deliberate departures from ``build_selectivity_table``:

1. Samples are restricted to the high/low buckets rather than the whole SciQ
   train split. ``--sample-set own`` (default) evaluates each utility column on
   that metric's own high-union-low bucket (1751 + 1751 = 3502 samples), which
   is the set the corresponding circuit was discovered from.
2. ``delta_grad_align`` is the two-checkpoint contrast

       GA(full trajectory ckpt, no hook) - GA(ablated trajectory ckpt, hook)

   using ``full_ckpt_gradalign.py`` output for the left term, instead of the
   same-checkpoint no-hook/hook overlay, which is nearly degenerate because a
   persistently-ablated model has already adapted around the removed edges.
   The same-checkpoint value is retained as a diagnostic column.

AUM and EL2N were already two-trajectory contrasts (each run accumulates its own
scores), so this makes all three utility columns structurally comparable.
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

DEFAULT_RESULT_ROOT = Path(
    str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix')
)
DEFAULT_BUCKETS_ROOT = Path(
    str(_REPO_ROOT / 'data_gens/buckets/application/Qwen__Qwen2.5-1.5B-Instruct')
)
METRICS = ("aum", "el2n", "grad_align")
ROW_LABELS = {
    "aum_high": "AUM-high",
    "el2n_high": "EL2N-high",
    "gradalign_high": "GA-high",
    "random_edge_matched": "Random (250-edge matched)",
}
ROWS = ("aum_high", "el2n_high", "gradalign_high", "random_edge_matched")
DIAGONAL_METRIC = {"aum_high": "aum", "el2n_high": "el2n", "gradalign_high": "grad_align"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_items(path: Path, section: str = "scores") -> Dict[str, Dict[str, float]]:
    """Map hf_id -> {metric: value} from an Exp4 run/overlay artifact."""
    payload = _read_json(path)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing items list")
    out: Dict[str, Dict[str, float]] = {}
    for item in items:
        hf_id = str(item.get("hf_id", ""))
        values = item.get(section)
        if not hf_id or not isinstance(values, dict):
            continue
        row: Dict[str, float] = {}
        for metric in METRICS:
            keys = (metric, "gradalign") if metric == "grad_align" else (metric,)
            for key in keys:
                number = _finite(values.get(key))
                if number is not None:
                    row[metric] = number
                    break
        out[hf_id] = row
    return out


def load_bucket_ids(buckets_root: Path, metric: str, dataset: str = "sciq") -> Dict[str, Set[str]]:
    ids: Dict[str, Set[str]] = {}
    for bucket in ("high", "low"):
        path = buckets_root / metric / dataset / f"{metric}_{dataset}_{bucket}.json"
        payload = _read_json(path)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{path}: missing items list")
        ids[bucket] = {str(item.get("hf_id", "")) for item in items if item.get("hf_id")}
    return ids


def resolve_sample_sets(
    buckets_root: Path, policy: str, universe: Set[str]
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """Return {metric: hf_id set} plus a description of how it was built."""
    per_metric = {metric: load_bucket_ids(buckets_root, metric) for metric in METRICS}
    own = {metric: per_metric[metric]["high"] | per_metric[metric]["low"] for metric in METRICS}
    if policy == "own":
        sets = own
    elif policy == "union":
        merged = set().union(*own.values())
        sets = {metric: set(merged) for metric in METRICS}
    elif policy == "intersection":
        shared = set.intersection(*own.values())
        sets = {metric: set(shared) for metric in METRICS}
    elif policy == "all":
        sets = {metric: set(universe) for metric in METRICS}
    else:
        raise ValueError(f"unknown sample-set policy: {policy}")
    sets = {metric: values & universe for metric, values in sets.items()}
    detail = {
        "policy": policy,
        "sizes": {metric: len(values) for metric, values in sets.items()},
        "bucket_sizes": {
            metric: {bucket: len(values) for bucket, values in per_metric[metric].items()}
            for metric in METRICS
        },
    }
    return sets, detail


def population_z(items: Mapping[str, Mapping[str, float]], metric: str) -> Dict[str, float]:
    """Standardize a metric within one checkpoint, over that checkpoint's whole population.

    The raw score scale is not comparable across checkpoints: a damaged model has
    larger gradients (GradAlign) and a shifted margin distribution (AUM/EL2N), so a
    raw full-minus-ablated difference partly measures the ablated checkpoint's own
    spread. Standardizing each side against its own population removes that.
    """
    ids = sorted(hf_id for hf_id, values in items.items() if metric in values)
    values = np.asarray([items[hf_id][metric] for hf_id in ids], dtype=np.float64)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    if not std or not math.isfinite(std) or std <= 0:
        return {hf_id: float("nan") for hf_id in ids}
    mean = float(np.mean(values))
    return {hf_id: (value - mean) / std for hf_id, value in zip(ids, values)}


def summarize(full: np.ndarray, ablated: np.ndarray) -> Dict[str, float]:
    """Positive-drop convention: delta = mean(full - ablated)."""
    if len(full) == 0 or len(full) != len(ablated):
        return {key: float("nan") for key in ("delta", "delta_z", "cohen_dz", "delta_pct",
                                              "full_mean", "ablated_mean", "full_std",
                                              "t_stat")} | {"n": 0}
    paired = full - ablated
    delta = float(np.mean(paired))
    full_mean = float(np.mean(full))
    full_std = float(np.std(full, ddof=1)) if len(full) > 1 else float("nan")
    paired_std = float(np.std(paired, ddof=1)) if len(paired) > 1 else float("nan")
    stderr = paired_std / math.sqrt(len(paired)) if paired_std and paired_std > 0 else float("nan")
    return {
        "delta": delta,
        "delta_z": delta / full_std if full_std and full_std > 0 else float("nan"),
        "cohen_dz": delta / paired_std if paired_std and paired_std > 0 else float("nan"),
        "delta_pct": 100.0 * delta / abs(full_mean) if full_mean != 0 else float("nan"),
        "full_mean": full_mean,
        "ablated_mean": float(np.mean(ablated)),
        "full_std": full_std,
        "t_stat": delta / stderr if stderr and math.isfinite(stderr) and stderr > 0 else float("nan"),
        "n": int(len(paired)),
    }


def paired_arrays(
    full: Mapping[str, Mapping[str, float]],
    ablated: Mapping[str, Mapping[str, float]],
    metric: str,
    hf_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    left: List[float] = []
    right: List[float] = []
    for hf_id in hf_ids:
        a = full.get(hf_id, {}).get(metric)
        b = ablated.get(hf_id, {}).get(metric)
        if a is None or b is None:
            continue
        left.append(a)
        right.append(b)
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def eval_deltas(overlay_meta: Mapping[str, Any]) -> Tuple[float, float]:
    evaluation = overlay_meta.get("evaluation")
    if not isinstance(evaluation, dict):
        return float("nan"), float("nan")
    baseline = evaluation.get("baseline")
    ablated = evaluation.get("ablated")
    if not isinstance(baseline, dict) or not isinstance(ablated, dict):
        return float("nan"), float("nan")
    base_acc, ab_acc = _finite(baseline.get("accuracy")), _finite(ablated.get("accuracy"))
    base_loss, ab_loss = _finite(baseline.get("loss")), _finite(ablated.get("loss"))
    delta_acc = base_acc - ab_acc if base_acc is not None and ab_acc is not None else float("nan")
    delta_loss = ab_loss - base_loss if base_loss is not None and ab_loss is not None else float("nan")
    return delta_acc, delta_loss


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--buckets-root", type=Path, default=DEFAULT_BUCKETS_ROOT)
    parser.add_argument(
        "--sample-set",
        default="own",
        choices=("own", "union", "intersection", "all"),
        help="own: each column uses its own metric's high|low bucket",
    )
    parser.add_argument(
        "--full-gradalign-json",
        type=Path,
        default=None,
        help="two_checkpoint/full_ckpt_gradalign.json (default: under --result-root)",
    )
    parser.add_argument("--output-prefix", default="bucket_selectivity")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    root: Path = args.result_root
    meta = _read_json(root / "selectivity_meta.json")
    diagnostics = meta.get("diagnostics", {})
    full_run_path = Path(str(meta["full_json"]))

    full_scores = load_items(full_run_path)
    full_ga_path = args.full_gradalign_json or (root / "two_checkpoint" / "full_ckpt_gradalign.json")
    full_ga = load_items(full_ga_path) if full_ga_path.is_file() else {}
    if not full_ga:
        raise FileNotFoundError(
            f"missing two-checkpoint full GradAlign scores: {full_ga_path}"
        )

    universe = set(full_scores) & set(full_ga)
    sample_sets, sample_detail = resolve_sample_sets(args.buckets_root, args.sample_set, universe)

    records: List[Dict[str, Any]] = []
    for row in ROWS:
        info = diagnostics.get(row)
        if not isinstance(info, dict):
            raise KeyError(f"selectivity_meta.json lacks diagnostics for {row}")
        ablated_scores = load_items(Path(str(info["run_json"])))
        overlay_path = Path(str(info["gradalign_overlay"]))
        overlay_payload = _read_json(overlay_path)
        overlay_ablated = load_items(overlay_path, "scores")
        overlay_full_same_ckpt = load_items(overlay_path, "full_recomputed_scores")

        record: Dict[str, Any] = {"row": row, "label": ROW_LABELS[row]}
        for metric in ("aum", "el2n"):
            ids = sorted(sample_sets[metric])
            stats = summarize(*paired_arrays(full_scores, ablated_scores, metric, ids))
            for key, value in stats.items():
                record[f"{metric}_{key}"] = value

        ga_ids = sorted(sample_sets["grad_align"])
        two_ckpt = summarize(*paired_arrays(full_ga, overlay_ablated, "grad_align", ga_ids))
        for key, value in two_ckpt.items():
            record[f"grad_align_{key}"] = value

        # Scale-free companion: standardize each side inside its own checkpoint first.
        for metric in METRICS:
            left, right = (
                (full_ga, overlay_ablated) if metric == "grad_align"
                else (full_scores, ablated_scores)
            )
            z_left, z_right = population_z(left, metric), population_z(right, metric)
            ids = sorted(sample_sets[metric] & set(z_left) & set(z_right))
            paired = np.asarray(
                [z_left[hf_id] - z_right[hf_id] for hf_id in ids], dtype=np.float64
            )
            paired = paired[np.isfinite(paired)]
            record[f"{metric}_delta_zpop"] = (
                float(np.mean(paired)) if len(paired) else float("nan")
            )
            record[f"{metric}_zpop_n"] = int(len(paired))
        same_ckpt = summarize(
            *paired_arrays(overlay_full_same_ckpt, overlay_ablated, "grad_align", ga_ids)
        )
        for key in ("delta", "delta_z", "n"):
            record[f"grad_align_same_ckpt_{key}"] = same_ckpt[key]

        delta_acc, delta_loss = eval_deltas(overlay_payload.get("meta", {}))
        record["delta_acc"] = delta_acc
        record["delta_loss"] = delta_loss
        records.append(record)

    columns = [
        "row", "label",
        "aum_delta", "el2n_delta", "grad_align_delta", "delta_loss", "delta_acc",
        "aum_delta_zpop", "el2n_delta_zpop", "grad_align_delta_zpop",
        "aum_cohen_dz", "el2n_cohen_dz", "grad_align_cohen_dz",
        "aum_delta_z", "el2n_delta_z", "grad_align_delta_z",
        "aum_delta_pct", "el2n_delta_pct", "grad_align_delta_pct",
        "aum_t_stat", "el2n_t_stat", "grad_align_t_stat",
        "aum_n", "el2n_n", "grad_align_n",
        "grad_align_same_ckpt_delta", "grad_align_same_ckpt_delta_z",
        "aum_full_mean", "aum_ablated_mean", "el2n_full_mean", "el2n_ablated_mean",
        "grad_align_full_mean", "grad_align_ablated_mean",
        "aum_full_std", "el2n_full_std", "grad_align_full_std",
    ]
    csv_path = root / f"{args.output_prefix}_{args.sample_set}.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for record in records:
            handle.write(
                ",".join(
                    str(record.get(column, "")) if not isinstance(record.get(column), float)
                    else repr(record[column])
                    for column in columns
                )
                + "\n"
            )

    lines = [
        "| Ablated circuit | ΔAUM | ΔEL2N | ΔGA | ΔLoss |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {label} | {aum} | {el2n} | {ga} | {loss} |".format(
                label=record["label"],
                aum=_fmt(record["aum_delta"]),
                el2n=_fmt(record["el2n_delta"]),
                ga=_fmt(record["grad_align_delta"]),
                loss=_fmt(record["delta_loss"]),
            )
        )
    markdown = "\n".join(lines)
    md_path = root / f"{args.output_prefix}_{args.sample_set}.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")

    meta_out = {
        "sample_set": sample_detail,
        "full_run_json": str(full_run_path),
        "full_gradalign_json": str(full_ga_path),
        "definitions": {
            "delta": "mean(full - ablated), paired by hf_id, positive = utility drop",
            "delta_z": (
                "delta / sample std of the full-side scores on the same subset; NOT "
                "comparable across columns because each checkpoint has its own scale"
            ),
            "delta_zpop": (
                "each side standardized inside its own checkpoint over all 11679 samples, "
                "then the bucket mean of the difference; scale-free, but it discards the "
                "population-wide mean shift and reports only movement relative to it"
            ),
            "cohen_dz": (
                "delta / sample std of the paired difference; keeps the direction of the "
                "actual shift and is comparable across columns"
            ),
            "grad_align": (
                "two-checkpoint: GA(full trajectory ckpt, no hook) - "
                "GA(ablated trajectory ckpt, persistent hook); each side uses its own "
                "checkpoint's SciQ validation reference gradient"
            ),
            "grad_align_same_ckpt": (
                "diagnostic: same ablated checkpoint, no-hook vs hook (the previous definition)"
            ),
            "delta_acc/delta_loss": (
                "validation, matched full trajectory vs persistent-ablated trajectory"
            ),
        },
        "rows": records,
    }
    meta_path = root / f"{args.output_prefix}_{args.sample_set}_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_out, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(markdown)
    print()
    print(f"[ok] {csv_path}")
    print(f"[ok] {md_path}")
    print(f"[ok] {meta_path}")


if __name__ == "__main__":
    main()
