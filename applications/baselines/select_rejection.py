"""Experiment 1b-②: rejection-sampling on the 20k naive pool.

Given a scores JSONL produced by compute_utility_scores.py, produce 3 pools
of 10k each — one per metric — selecting top-50% by:
  - AUM proxy  (higher = more learnable)   → mirrors "easy" profile
  - EL2N       (higher = more challenging) → mirrors "hard"  profile
  - GradAlign  (higher = more aligned)     → mirrors "helpful" profile

Each pool is written into a distinct mirror tree the trainer can consume.
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
import math
import sys
from pathlib import Path
from typing import Dict, List

from applications.experiment_utils.io import (
    NAIVE_RAW_PATH,
    dump_pool_as_mirror,
    read_jsonl,
    write_json,
)
from applications.curriculum_ft import generated_row_to_example  # noqa: E402

EXP1_WORK_ROOT = _SUPPLEMENTAL_ROOT / "baselines" / "work_exp1_downstream_seeds"

METRIC_TO_LABEL = {
    "aum_proxy": "learnable",  # informational
    "el2n": "challenging",
    "gradalign": "aligned",
}


def rank_top_half(rows: List[dict], scores: List[dict], key: str, n_select: int) -> List[int]:
    """Return indices (into rows) sorted by score[key] descending, cut at n_select."""
    pairs = []
    for s in scores:
        if s.get("valid") is False:
            continue
        v = s.get(key)
        idx = s.get("idx")
        if not isinstance(idx, int) or not (0 <= idx < len(rows)):
            raise ValueError(f"{key}: invalid score idx={idx!r}")
        try:
            value = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: idx={idx} has invalid score {v!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key}: idx={idx} has non-finite score {value}")
        pairs.append((idx, value))
    if len(pairs) < n_select:
        raise RuntimeError(
            f"{key}: only {len(pairs)} valid scores for requested n_select={n_select}"
        )
    pairs.sort(key=lambda p: -p[1])
    return [i for i, _ in pairs[:n_select]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(NAIVE_RAW_PATH))
    ap.add_argument("--scores", required=True, help="jsonl from compute_utility_scores.py")
    ap.add_argument("--variant-root", default=str(EXP1_WORK_ROOT / "rejection"))
    ap.add_argument("--n-select", type=int, default=10000)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    scores = read_jsonl(args.scores)
    print(f"[load] rows={len(rows)} scores={len(scores)}")
    if not rows:
        raise RuntimeError(f"input pool is empty: {args.input}")
    if not scores:
        raise RuntimeError(f"score file is empty: {args.scores}")
    if not (0 < args.n_select <= len(rows)):
        raise ValueError(f"--n-select must be in [1, {len(rows)}], got {args.n_select}")
    score_indices = [score.get("idx") for score in scores]
    expected_indices = list(range(len(rows)))
    if (
        any(not isinstance(idx, int) for idx in score_indices)
        or len(score_indices) != len(set(score_indices))
        or set(score_indices) != set(expected_indices)
    ):
        raise RuntimeError(
            "score file must contain exactly one row for every input idx; "
            f"expected {len(expected_indices)} indices, got {len(score_indices)}"
        )

    core_valid = [
        generated_row_to_example(row, dataset_name="sciq") is not None
        for row in rows
    ]
    score_by_idx = {score["idx"]: score for score in scores}
    mismatches = [
        idx for idx, valid in enumerate(core_valid)
        if bool(score_by_idx[idx].get("valid")) != valid
    ]
    if mismatches:
        raise RuntimeError(
            f"score valid mask disagrees with core converter at {len(mismatches)} rows; "
            f"first indices={mismatches[:10]}"
        )
    valid_count = sum(core_valid)
    invalid_count = len(rows) - valid_count
    if args.n_select > valid_count:
        raise RuntimeError(
            f"n_select={args.n_select} exceeds trainer-convertible valid_count={valid_count}"
        )

    for key in ("aum_proxy", "el2n", "gradalign"):
        idxs = rank_top_half(rows, scores, key=key, n_select=args.n_select)
        if len(idxs) != args.n_select or len(set(idxs)) != args.n_select:
            raise RuntimeError(f"{key}: selection did not produce {args.n_select} unique rows")
        picked = [rows[i] for i in idxs]
        out = dump_pool_as_mirror(
            picked,
            variant_root=Path(args.variant_root) / key,
            recipe="naive",
            prompt_mode="fixed",
            gen_mode="both",
            circuit_target_mode="discovered",
            file_kind="selected",
        )
        meta_path = out.parent / f"{key}_selection_meta.json"
        write_json(
            meta_path,
            {
                "input": str(Path(args.input).expanduser().resolve()),
                "scores": str(Path(args.scores).expanduser().resolve()),
                "metric": key,
                "direction": "descending",
                "raw_count": len(rows),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "selected_count": len(picked),
                "selection_fraction_of_valid": len(picked) / valid_count,
                "requested_budget": args.n_select,
                "output": str(out),
            },
        )
        print(
            f"[rejection:{key}] wrote {out} (raw={len(rows)}, valid={valid_count}, "
            f"invalid={invalid_count}, selected={len(picked)}, "
            f"fraction={len(picked) / valid_count:.6f}); meta={meta_path}"
        )


if __name__ == "__main__":
    main()
