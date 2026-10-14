"""Selectivity table from the clean-checkpoint hook on/off contrast.

Reads the per-circuit artifacts written by ``clean_ckpt_hook_delta`` and reports
each utility drop in units of the hook-off standard deviation on the same subset.
Because the weights are identical on both sides, the hook-off distribution is the
same for every row, so all four rows share one denominator per column.
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_RESULT_ROOT = Path(
    str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix')
)
METRICS = ("aum", "el2n", "grad_align")
ROWS = (
    ("aum", "AUM-high"),
    ("el2n", "EL2N-high"),
    ("grad_align", "GA-high"),
    ("random_model", "Random (250-edge matched)"),
)


def _fmt(value: float, digits: int = 4) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-prefix", default="clean_hook_selectivity")
    args = parser.parse_args(argv)
    clean_dir = args.result_root / "clean_hook"

    records: List[Dict[str, Any]] = []
    hook_off_reference: Dict[str, np.ndarray] = {}
    for key, label in ROWS:
        path = clean_dir / f"clean_{key}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload["items"]
        meta = payload["meta"]
        record: Dict[str, Any] = {"circuit": key, "label": label, "n": len(items)}
        for metric in METRICS:
            off = np.asarray([x["hook_off"][metric] for x in items], dtype=np.float64)
            on = np.asarray([x["hook_on"][metric] for x in items], dtype=np.float64)
            paired = off - on
            off_std = float(np.std(off, ddof=1))
            paired_std = float(np.std(paired, ddof=1))
            delta = float(np.mean(paired))
            record[f"{metric}_delta"] = delta
            record[f"{metric}_delta_over_sd_full"] = (
                delta / off_std if off_std > 0 else float("nan")
            )
            record[f"{metric}_cohen_dz"] = (
                delta / paired_std if paired_std > 0 else float("nan")
            )
            record[f"{metric}_t_stat"] = (
                delta / (paired_std / math.sqrt(len(paired)))
                if paired_std > 0 and len(paired) > 1
                else float("nan")
            )
            # Signed means cancel when a circuit scrambles scores without a
            # consistent direction, which is exactly what GradAlign does; the
            # magnitude views below keep that disruption visible.
            record[f"{metric}_mean_abs_delta"] = float(np.mean(np.abs(paired)))
            record[f"{metric}_median_abs_delta"] = float(np.median(np.abs(paired)))
            record[f"{metric}_direction_consistency"] = (
                abs(delta) / float(np.mean(np.abs(paired)))
                if np.mean(np.abs(paired)) > 0
                else float("nan")
            )
            record[f"{metric}_hook_off_mean"] = float(np.mean(off))
            record[f"{metric}_hook_on_mean"] = float(np.mean(on))
            record[f"{metric}_hook_off_sd"] = off_std
            record[f"{metric}_hook_on_sd"] = float(np.std(on, ddof=1))
            hook_off_reference.setdefault(metric, off)
            if not np.allclose(hook_off_reference[metric], off, rtol=1e-9, atol=1e-9):
                record.setdefault("warnings", []).append(
                    f"{metric}: hook-off scores differ from the first row's; "
                    "the clean side should be identical across circuits"
                )
        evaluation = meta.get("evaluation", {})
        off_eval, on_eval = evaluation.get("hook_off", {}), evaluation.get("hook_on", {})
        record["delta_loss"] = float(on_eval.get("loss", float("nan"))) - float(
            off_eval.get("loss", float("nan"))
        )
        record["delta_acc"] = float(off_eval.get("accuracy", float("nan"))) - float(
            on_eval.get("accuracy", float("nan"))
        )
        record["val_loss_hook_off"] = off_eval.get("loss")
        record["val_loss_hook_on"] = on_eval.get("loss")
        record["val_acc_hook_off"] = off_eval.get("accuracy")
        record["val_acc_hook_on"] = on_eval.get("accuracy")
        records.append(record)

    # Every circuit's disruption expressed as a multiple of the random-circuit
    # control, which absorbs each metric's own scale.
    random_record = next(r for r in records if r["circuit"] == "random_model")
    for record in records:
        for metric in METRICS:
            for stat in ("mean_abs_delta", "median_abs_delta"):
                baseline = random_record[f"{metric}_{stat}"]
                record[f"{metric}_{stat}_vs_random"] = (
                    record[f"{metric}_{stat}"] / baseline if baseline > 0 else float("nan")
                )

    columns = ["circuit", "label", "n"] + [
        f"{metric}_{suffix}"
        for suffix in (
            "delta", "delta_over_sd_full", "cohen_dz", "t_stat",
            "mean_abs_delta", "median_abs_delta", "direction_consistency",
            "mean_abs_delta_vs_random", "median_abs_delta_vs_random",
            "hook_off_mean", "hook_on_mean", "hook_off_sd", "hook_on_sd",
        )
        for metric in METRICS
    ] + [
        "delta_loss", "delta_acc",
        "val_loss_hook_off", "val_loss_hook_on", "val_acc_hook_off", "val_acc_hook_on",
    ]
    csv_path = args.result_root / f"{args.output_prefix}.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for record in records:
            handle.write(
                ",".join(
                    repr(record[column]) if isinstance(record.get(column), float)
                    else str(record.get(column, ""))
                    for column in columns
                )
                + "\n"
            )

    lines = [
        "| Ablated circuit | ΔAUM / SD(full) | ΔEL2N / SD(full) | ΔGA / SD(full) | raw ΔLoss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {label} | {aum} | {el2n} | {ga} | {loss} |".format(
                label=record["label"],
                aum=_fmt(record["aum_delta_over_sd_full"]),
                el2n=_fmt(record["el2n_delta_over_sd_full"]),
                ga=_fmt(record["grad_align_delta_over_sd_full"]),
                loss=_fmt(record["delta_loss"]),
            )
        )
    def matrix_block(title: str, suffix: str, digits: int) -> List[str]:
        block = [
            "",
            f"**{title}**",
            "",
            "| Ablated circuit | AUM | EL2N | GA |",
            "| --- | ---: | ---: | ---: |",
        ]
        for item in records:
            block.append(
                "| {label} | {a} | {e} | {g} |".format(
                    label=item["label"],
                    a=_fmt(item[f"aum_{suffix}"], digits),
                    e=_fmt(item[f"el2n_{suffix}"], digits),
                    g=_fmt(item[f"grad_align_{suffix}"], digits),
                )
            )
        return block

    lines += matrix_block("Signed mean(Δ)", "delta", 4)
    lines += matrix_block("Magnitude mean|Δ|", "mean_abs_delta", 4)
    lines += matrix_block("Magnitude median|Δ| (outlier-robust)", "median_abs_delta", 4)
    lines += matrix_block(
        "mean|Δ| as a multiple of the random-circuit control", "mean_abs_delta_vs_random", 2
    )
    lines += matrix_block(
        "Direction consistency |mean(Δ)| / mean|Δ| (1.0 = every sample moves the same way)",
        "direction_consistency",
        3,
    )
    markdown = "\n".join(lines)
    (args.result_root / f"{args.output_prefix}.md").write_text(markdown + "\n", encoding="utf-8")
    with (args.result_root / f"{args.output_prefix}_meta.json").open("w", encoding="utf-8") as h:
        json.dump(
            {
                "design": (
                    "clean full-trajectory checkpoint scored twice per circuit; weights "
                    "identical, hook off vs hook on. No trained-weight damage is involved."
                ),
                "delta": "mean(hook_off - hook_on), paired by hf_id",
                "delta_over_sd_full": "delta / sd of the hook-off scores on the same subset",
                "delta_loss": "validation loss(hook on) - loss(hook off), same checkpoint",
                "rows": records,
            },
            h,
            ensure_ascii=False,
            indent=2,
        )
        h.write("\n")

    print(markdown)
    print()
    for record in records:
        for warning in record.get("warnings", []):
            print(f"[warn] {record['circuit']}: {warning}")
    print(f"[ok] {csv_path}")


if __name__ == "__main__":
    main()
