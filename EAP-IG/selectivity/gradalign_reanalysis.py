#!/usr/bin/env python3
"""Scale-robust AUROC and bootstrap GapRed analysis for GradAlign ablations."""

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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

DEFAULT_INPUT = Path(
    str(_REPO_ROOT / 'results/zero_abl_result/app_zero_abl_top250/sciq/grad_align/static_mean')
)
DEFAULT_OUTPUT = (
    _SUPPLEMENTAL_ROOT / "additional/results/gradalign_reanalysis"
)
TARGETS = ("high", "low", "random")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    required = {
        "hf_id",
        "bucket",
        "q_full",
        "q_full_recomputed",
        "q_ablated",
    }
    for target in TARGETS:
        path = root / f"sciq_grad_align_static_{target}_sample_scores.csv"
        frame = pd.read_csv(path).sort_values("hf_id").reset_index(drop=True)
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if set(frame["bucket"]) != {"high", "low"}:
            raise ValueError(f"{path}: expected high/low buckets")
        if frame["hf_id"].duplicated().any():
            raise ValueError(f"{path}: duplicate hf_id")
        for column in ("q_full", "q_full_recomputed", "q_ablated"):
            values = frame[column].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {column}")
        frames[target] = frame

    reference = frames["high"]
    for target in TARGETS[1:]:
        current = frames[target]
        if not reference[["hf_id", "bucket"]].equals(current[["hf_id", "bucket"]]):
            raise ValueError(f"{target}: sample population differs")
        for column in ("q_full", "q_full_recomputed"):
            if not np.allclose(
                reference[column].to_numpy(float),
                current[column].to_numpy(float),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(f"{target}: {column} differs from matched full values")
    return frames


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = load_inputs(Path(args.input_dir))
    base = frames["high"]
    y = (base["bucket"].to_numpy() == "high").astype(np.int8)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) != len(neg):
        raise ValueError("expected balanced original high/low buckets")

    score_vectors = {
        "original_full_q": base["q_full"].to_numpy(float),
        "matched_full_recomputed_q": base["q_full_recomputed"].to_numpy(float),
        **{
            f"{target}_circuit_ablated_q": frames[target]["q_ablated"].to_numpy(float)
            for target in TARGETS
        },
    }
    rng = np.random.default_rng(args.seed)
    boot_auc = {
        name: np.empty(args.bootstrap, dtype=float) for name in score_vectors
    }
    for index in range(args.bootstrap):
        sample = np.concatenate(
            (
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            )
        )
        sampled_y = y[sample]
        for name, scores in score_vectors.items():
            boot_auc[name][index] = auc(sampled_y, scores[sample])

    matched_name = "matched_full_recomputed_q"
    matched_auc = auc(y, score_vectors[matched_name])
    auc_rows = []
    for name, scores in score_vectors.items():
        point = auc(y, scores)
        delta_boot = boot_auc[name] - boot_auc[matched_name]
        separation_reduction = abs(matched_auc - 0.5) - abs(point - 0.5)
        separation_reduction_boot = (
            np.abs(boot_auc[matched_name] - 0.5)
            - np.abs(boot_auc[name] - 0.5)
        )
        auc_rows.append(
            {
                "score": name,
                "auroc": point,
                "auroc_ci_low": ci(boot_auc[name])[0],
                "auroc_ci_high": ci(boot_auc[name])[1],
                "distance_to_chance": abs(point - 0.5),
                "delta_vs_matched_full": point - matched_auc,
                "delta_ci_low": ci(delta_boot)[0],
                "delta_ci_high": ci(delta_boot)[1],
                "chance_separation_reduction_vs_matched_full": (
                    separation_reduction
                ),
                "chance_separation_reduction_ci_low": ci(
                    separation_reduction_boot
                )[0],
                "chance_separation_reduction_ci_high": ci(
                    separation_reduction_boot
                )[1],
            }
        )

    full = score_vectors[matched_name]
    gap_rows = []
    gap_bootstrap: dict[str, np.ndarray] = {}
    for target in TARGETS:
        ablated = score_vectors[f"{target}_circuit_ablated_q"]
        gap_full = float(full[pos].mean() - full[neg].mean())
        gap_ablated = float(ablated[pos].mean() - ablated[neg].mean())
        ratio = 1.0 - abs(gap_ablated) / abs(gap_full)
        absolute_gap_change = abs(gap_ablated) - abs(gap_full)
        boot = np.empty(args.bootstrap, dtype=float)
        boot_gap_full = np.empty(args.bootstrap, dtype=float)
        boot_gap_ablated = np.empty(args.bootstrap, dtype=float)
        boot_absolute_gap_change = np.empty(args.bootstrap, dtype=float)
        boot_mean_abs_delta = np.empty(args.bootstrap, dtype=float)
        target_rng = np.random.default_rng(args.seed)
        for index in range(args.bootstrap):
            sampled_pos = target_rng.choice(pos, size=len(pos), replace=True)
            sampled_neg = target_rng.choice(neg, size=len(neg), replace=True)
            gf = float(full[sampled_pos].mean() - full[sampled_neg].mean())
            ga = float(ablated[sampled_pos].mean() - ablated[sampled_neg].mean())
            sampled = np.concatenate((sampled_pos, sampled_neg))
            boot_gap_full[index] = gf
            boot_gap_ablated[index] = ga
            boot_absolute_gap_change[index] = abs(ga) - abs(gf)
            boot_mean_abs_delta[index] = float(
                np.mean(np.abs(full[sampled] - ablated[sampled]))
            )
            boot[index] = np.nan if abs(gf) < 1e-15 else 1.0 - abs(ga) / abs(gf)
        finite_boot = boot[np.isfinite(boot)]
        gap_bootstrap[target] = finite_boot
        delta = full - ablated
        gap_rows.append(
            {
                "ablated_circuit": target,
                "n_high": len(pos),
                "n_low": len(neg),
                "gap_full_recomputed": gap_full,
                "gap_full_recomputed_ci_low": ci(boot_gap_full)[0],
                "gap_full_recomputed_ci_high": ci(boot_gap_full)[1],
                "gap_ablated": gap_ablated,
                "gap_ablated_ci_low": ci(boot_gap_ablated)[0],
                "gap_ablated_ci_high": ci(boot_gap_ablated)[1],
                "absolute_gap_change": absolute_gap_change,
                "absolute_gap_change_ci_low": ci(boot_absolute_gap_change)[0],
                "absolute_gap_change_ci_high": ci(boot_absolute_gap_change)[1],
                "gap_reduction_ratio": ratio,
                "gap_reduction_percent_math": 100.0 * ratio,
                "gap_reduction_ci_low": ci(finite_boot)[0],
                "gap_reduction_ci_high": ci(finite_boot)[1],
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "mean_abs_delta_ci_low": ci(boot_mean_abs_delta)[0],
                "mean_abs_delta_ci_high": ci(boot_mean_abs_delta)[1],
                "mean_abs_delta_high": float(np.mean(np.abs(delta[pos]))),
                "mean_abs_delta_low": float(np.mean(np.abs(delta[neg]))),
                "pearson_full_vs_ablated": float(pearsonr(full, ablated).statistic),
                "spearman_full_vs_ablated": float(spearmanr(full, ablated).statistic),
            }
        )

    original = score_vectors["original_full_q"]
    alignment = {
        "pearson_original_vs_recomputed_full": float(
            pearsonr(original, full).statistic
        ),
        "spearman_original_vs_recomputed_full": float(
            spearmanr(original, full).statistic
        ),
    }
    auc_by_name = {row["score"]: row for row in auc_rows}
    gap_by_target = {row["ablated_circuit"]: row for row in gap_rows}
    high_auc = auc_by_name["high_circuit_ablated_q"]["auroc"]
    full_auc = auc_by_name[matched_name]["auroc"]
    random_auc = auc_by_name["random_circuit_ablated_q"]["auroc"]
    interpretation = (
        "The original q_full AUROC is tautological because those scores defined "
        "the high/low buckets. The matched no-hook recomputation is the valid "
        "pre-ablation comparator. GradAlign high-circuit ablation does not collapse "
        "AUROC to 0.5 if it crosses chance; that is an ordering reversal, while a "
        "random-edge result close to matched full indicates preservation."
    )
    summary = {
        "n": int(len(y)),
        "n_high": int(len(pos)),
        "n_low": int(len(neg)),
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed},
        "auroc": auc_rows,
        "gapred": gap_rows,
        "full_score_alignment": alignment,
        "interpretation": interpretation,
        "unit_note": (
            "gap_reduction_ratio is 1-|gap_ablated|/|gap_full|. Multiplying it "
            "by 100 gives gap_reduction_percent_math; the legacy table displayed "
            "the unmultiplied ratio with a percent sign. absolute_gap_change is "
            "|gap_ablated|-|gap_full|, so positive values indicate expansion. "
            "chance_separation_reduction_vs_matched_full is "
            "|AUROC_full-0.5|-|AUROC_score-0.5|, so positive values indicate "
            "reduced high/low separability."
        ),
        "sources": {
            target: str(
                (
                    Path(args.input_dir)
                    / f"sciq_grad_align_static_{target}_sample_scores.csv"
                ).resolve()
            )
            for target in TARGETS
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gradalign_reanalysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(auc_rows).to_csv(output_dir / "gradalign_auroc.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(output_dir / "gradalign_gapred.csv", index=False)
    gap = gap_by_target["high"]
    line = (
        f"Matched GradAlign AUROC changes from {full_auc:.3f} to {high_auc:.3f} "
        f"after high-circuit ablation, versus {random_auc:.3f} after random-edge "
        f"ablation; this is an ordering reversal, not collapse to chance. "
        f"GapRed ratio is {gap['gap_reduction_ratio']:.3f} "
        f"(95% bootstrap CI {gap['gap_reduction_ci_low']:.3f} to "
        f"{gap['gap_reduction_ci_high']:.3f})."
    )
    (output_dir / "rebuttal_line.txt").write_text(line + "\n", encoding="utf-8")
    print(line)


if __name__ == "__main__":
    main()
