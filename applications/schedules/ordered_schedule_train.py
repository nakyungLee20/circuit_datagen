"""One common ordered-batch engine for all four Exp3 schedule conditions.

The shared trainer intentionally shuffles each epoch and maps epochs onto
exactly three datasets.  Those semantics are correct for its normal SAMS
experiments but its three reused/shuffled datasets would confound a schedule
ablation against batch-linear/Bengio conditions. This Exp3-only wrapper reuses
the shared model/optimizer/evaluation/checkpoint implementation while supplying
one exact, ordered dataset per epoch for uniform, reverse, linear, and Bengio,
and temporarily forcing its training DataLoader to ``shuffle=False``.

No core trainer source is modified.
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
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from applications import curriculum_ft as ft  # type: ignore  # noqa: E402

from applications.schedules.linear_interp_schedule import (  # noqa: E402
    PROFILES,
    interp_ratios,
    parse_ratios,
)

UNIFORM_TARGET = {"easy": 0.333, "hard": 0.333, "helpful": 0.334}
REVERSE_STAGE_TARGETS = (
    {"easy": 0.10, "hard": 0.65, "helpful": 0.25},
    {"easy": 0.25, "hard": 0.45, "helpful": 0.30},
    {"easy": 0.55, "hard": 0.15, "helpful": 0.30},
)
# SAMS is the easy->hard direction of the very same three stage targets, so the
# ordered-trainer arm is exactly REVERSE read backwards. Defining it that way
# keeps the two arms guaranteed-symmetric instead of two hand-typed tables that
# could drift apart. Table 3's fifth row previously came from exp1's shared
# trainer (shuffle=on), which confounded schedule effect with trainer effect;
# running this arm puts SAMS under the same ordered-batch trainer as the other
# four schedules.
SAMS_STAGE_TARGETS = tuple(reversed(REVERSE_STAGE_TARGETS))


class CyclingSampler:
    def __init__(self, rows: Sequence[Dict[str, Any]], seed: int) -> None:
        if not rows:
            raise ValueError("cannot sample from an empty pool")
        self.rows = [dict(row) for row in rows]
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.rows)))
        self.rng.shuffle(self.order)
        self.cursor = 0

    def draw(self, count: int) -> List[Dict[str, Any]]:
        output = []
        for _ in range(count):
            if self.cursor >= len(self.order):
                self._reshuffle()
            output.append(dict(self.rows[self.order[self.cursor]]))
            self.cursor += 1
        return output


def batch_plan(
    *,
    epochs: int,
    examples_per_epoch: int,
    batch_size: int,
    source_ratio: float,
) -> List[Dict[str, int]]:
    if epochs <= 0 or examples_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("epochs, examples_per_epoch, and batch_size must be positive")
    if not 0.0 <= source_ratio <= 1.0:
        raise ValueError(f"source_ratio must be in [0,1], got {source_ratio}")
    specs: List[Dict[str, int]] = []
    seen_examples = 0
    assigned_source = 0
    global_batch = 0
    for epoch_idx in range(epochs):
        offset = 0
        batch_idx = 0
        while offset < examples_per_epoch:
            size = min(batch_size, examples_per_epoch - offset)
            target_source_total = round((seen_examples + size) * source_ratio)
            n_source = target_source_total - assigned_source
            if not 0 <= n_source <= size:
                raise AssertionError((n_source, size, source_ratio))
            specs.append(
                {
                    "global_batch": global_batch,
                    "epoch_index": epoch_idx,
                    "batch_index_in_epoch": batch_idx,
                    "batch_size": size,
                    "n_source": n_source,
                    "n_generated": size - n_source,
                }
            )
            assigned_source += n_source
            seen_examples += size
            offset += size
            batch_idx += 1
            global_batch += 1
    return specs


def allocate_generated_counts(
    *,
    n_generated: int,
    target: Mapping[str, float],
    cumulative_expected: Dict[str, float],
    cumulative_actual: Dict[str, int],
) -> Dict[str, int]:
    for profile in PROFILES:
        cumulative_expected[profile] += n_generated * float(target[profile])
    counts = {profile: 0 for profile in PROFILES}
    for _ in range(n_generated):
        profile = max(
            PROFILES,
            key=lambda key: (
                cumulative_expected[key] - cumulative_actual[key],
                -PROFILES.index(key),
            ),
        )
        counts[profile] += 1
        cumulative_actual[profile] += 1
    return counts


def load_linear_pools(
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Mapping[str, Any]]:
    pools, metadata = ft.build_generated_recipe_pools(
        args=args,
        recipes=list(PROFILES),
        file_kind=args.ours_file_kind,
        circuit_target_mode="discovered",
    )
    for profile in PROFILES:
        if not pools[profile]:
            raise ValueError(f"empty clean steered pool for {profile}")
    return pools, metadata


def load_ranked_bengio_pool(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Mapping[str, Any]]:
    if args.shuffle_choices:
        raise ValueError(
            "Bengio proxy ranks are defined on the stored normalized choice order; "
            "pass --no-shuffle-choices so training consumes that exact prompt/label order"
        )
    raw_rows: List[Dict[str, Any]] = []
    files: Dict[str, str] = {}
    choice_order_verified = 0
    for recipe in ("easy", "medium", "hard"):
        path = ft.find_latest_generated_file(
            generation_root=Path(args.generation_root),
            generator_model_key=str(args.generator_model_key or ""),
            recipe=recipe,
            file_kind="selected",
            prompt_mode=args.generated_prompt_mode,
            gen_mode=args.generated_gen_mode,
            circuit_target_mode="discovered",
            path_must_contain=str(args.generated_path_contains or ""),
        )
        files[recipe] = str(path)
        for raw in ft.load_jsonl(path):
            converted = ft.generated_row_to_example(raw, args.dataset)
            if converted is None:
                raise ValueError(
                    f"Bengio pool row failed conversion: file={path}, "
                    f"candidate={raw.get('candidate_id')!r}"
                )
            converted["exp3_proxy_margin"] = float(raw["exp3_proxy_margin"])
            converted["exp3_difficulty_rank"] = int(raw["exp3_difficulty_rank"])
            converted["exp3_source_band"] = str(raw.get("exp3_source_band") or "")
            expected_question = str(raw.get("question") or "")
            expected_choices = [str(choice) for choice in (raw.get("choices") or [])]
            expected_gold = str(raw.get("answer") or "")
            if (
                converted["question_text"] != expected_question
                or converted["choices"] != expected_choices
                or converted["gold_label"] != expected_gold
            ):
                raise ValueError(
                    "Bengio consumed example differs from proxy-ranked normalized row: "
                    f"file={path}, rank={converted['exp3_difficulty_rank']}"
                )
            choice_order_verified += 1
            raw_rows.append(converted)
    raw_rows.sort(key=lambda row: int(row["exp3_difficulty_rank"]))
    ranks = [int(row["exp3_difficulty_rank"]) for row in raw_rows]
    if ranks != list(range(len(raw_rows))):
        raise ValueError(
            "Bengio pool ranks are not one contiguous global ordering; "
            f"first={ranks[:5]}, last={ranks[-5:]}"
        )
    margins = [float(row["exp3_proxy_margin"]) for row in raw_rows]
    violations = sum(
        1 for left, right in zip(margins, margins[1:]) if right > left + 1e-8
    )
    if violations:
        raise ValueError(
            f"Bengio pool is not easy->hard (non-increasing proxy margin): "
            f"{violations} adjacent violations"
        )
    return raw_rows, {
        "files": files,
        "n_ranked": len(raw_rows),
        "margin_max_easy": margins[0],
        "margin_min_hard": margins[-1],
        "adjacent_monotonicity_violations": violations,
        "choice_order_preserved_from_proxy_ranking": True,
        "choice_order_verified_rows": choice_order_verified,
        "shuffle_choices": False,
    }


def build_ratio_stages(
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Dict[str, Any]],
    pools: Mapping[str, Sequence[Dict[str, Any]]],
    schedule_kind: str,
    start: Mapping[str, float] | None = None,
    end: Mapping[str, float] | None = None,
) -> Tuple[List[ft.StageDataset], List[Mapping[str, Any]], Mapping[str, Any]]:
    if schedule_kind not in {"uniform", "reverse", "linear", "sams"}:
        raise ValueError(f"unsupported ratio schedule: {schedule_kind}")
    if schedule_kind == "linear" and (start is None or end is None):
        raise ValueError("linear schedule requires start and end ratios")
    specs = batch_plan(
        epochs=args.epochs,
        examples_per_epoch=args.train_examples,
        batch_size=args.batch_size,
        source_ratio=args.source_ratio,
    )
    source_sampler = CyclingSampler(source_rows, args.seed + 70001)
    pool_samplers = {
        profile: CyclingSampler(pools[profile], args.seed + 71000 + idx * 1009)
        for idx, profile in enumerate(PROFILES)
    }
    cumulative_expected = {profile: 0.0 for profile in PROFILES}
    cumulative_actual = {profile: 0 for profile in PROFILES}
    stages_rows: List[List[Dict[str, Any]]] = [[] for _ in range(args.epochs)]
    schedule: List[Mapping[str, Any]] = []
    total_batches = len(specs)
    reverse_epoch_stage_ids = ft.build_epoch_stage_ids(args.epochs, num_stages=3)

    for spec in specs:
        global_batch = spec["global_batch"]
        t = global_batch / max(1, total_batches - 1)
        ratio_stage_id = None
        if schedule_kind == "uniform":
            target = dict(UNIFORM_TARGET)
        elif schedule_kind in {"reverse", "sams"}:
            ratio_stage_id = reverse_epoch_stage_ids[spec["epoch_index"]]
            targets = (REVERSE_STAGE_TARGETS if schedule_kind == "reverse"
                       else SAMS_STAGE_TARGETS)
            target = dict(targets[ratio_stage_id])
        else:  # linear
            assert start is not None and end is not None
            if global_batch == 0:
                target = dict(start)
            elif global_batch == total_batches - 1:
                target = dict(end)
            else:
                target = interp_ratios(t, start, end)
        counts = allocate_generated_counts(
            n_generated=spec["n_generated"],
            target=target,
            cumulative_expected=cumulative_expected,
            cumulative_actual=cumulative_actual,
        )
        batch = source_sampler.draw(spec["n_source"])
        generated = []
        for profile in PROFILES:
            generated.extend(pool_samplers[profile].draw(counts[profile]))
        # Deterministic within-batch interleaving does not alter the batch
        # composition. DataLoader order across batches remains fixed.
        random.Random(args.seed + global_batch * 7919).shuffle(generated)
        batch.extend(generated)
        stages_rows[spec["epoch_index"]].extend(batch)
        schedule_row: Dict[str, Any] = {
            **spec,
            "target_generated_ratios": target,
            "realized_generated_counts": counts,
        }
        if schedule_kind == "linear":
            schedule_row["t"] = t
        if ratio_stage_id is not None:
            schedule_row["ratio_stage_id"] = ratio_stage_id
        schedule.append(schedule_row)

    stages = []
    for epoch_idx, rows in enumerate(stages_rows):
        if len(rows) != args.train_examples:
            raise AssertionError((epoch_idx, len(rows), args.train_examples))
        epoch_specs = [row for row in schedule if row["epoch_index"] == epoch_idx]
        avg_target = {
            profile: float(
                sum(
                    row["target_generated_ratios"][profile] * row["n_generated"]
                    for row in epoch_specs
                )
                / max(1, sum(row["n_generated"] for row in epoch_specs))
            )
            for profile in PROFILES
        }
        summary = ft.summarize_examples(rows)
        summary["exp3_schedule"] = {
            "kind": f"batch_{schedule_kind}",
            "first_global_batch": epoch_specs[0]["global_batch"],
            "last_global_batch": epoch_specs[-1]["global_batch"],
            "mean_target_generated_ratios": avg_target,
            "dataloader_shuffle": False,
        }
        stages.append(
            ft.StageDataset(
                stage_id=epoch_idx,
                stage_name=f"{schedule_kind}_epoch_{epoch_idx + 1}",
                ratios=avg_target,
                examples=rows,
                summary=summary,
            )
        )
    validation = {
        "schedule_kind": schedule_kind,
        "first_target": schedule[0]["target_generated_ratios"],
        "last_target": schedule[-1]["target_generated_ratios"],
        "n_batches": len(schedule),
        "cumulative_realized_generated_counts": cumulative_actual,
        "dataloader_shuffle": False,
        "epoch_stage_mapping": list(range(args.epochs)),
        "ratio_targets_by_epoch": [
            next(
                row["target_generated_ratios"]
                for row in schedule
                if row["epoch_index"] == epoch_idx
            )
            for epoch_idx in range(args.epochs)
        ],
    }
    if schedule_kind == "linear":
        assert start is not None and end is not None
        endpoint_error = max(
            abs(float(validation[which][profile]) - float(reference[profile]))
            for which, reference in (("first_target", start), ("last_target", end))
            for profile in PROFILES
        )
        validation["max_endpoint_abs_error"] = endpoint_error
        if endpoint_error > 1e-12:
            raise AssertionError(validation)
    elif schedule_kind == "uniform":
        if any(
            row["target_generated_ratios"] != UNIFORM_TARGET for row in schedule
        ):
            raise AssertionError("uniform target changed across batches")
    else:
        validation[f"{schedule_kind}_ratio_stage_ids_by_epoch"] = reverse_epoch_stage_ids
        if schedule_kind == "sams":
            # Guard the symmetry the constant relies on: stage k of SAMS must be
            # stage (2-k) of REVERSE, or the two arms are no longer comparable.
            assert all(SAMS_STAGE_TARGETS[i] == REVERSE_STAGE_TARGETS[2 - i]
                       for i in range(3)), "SAMS/REVERSE stage targets desynced"
    return stages, schedule, validation


def build_linear_stages(
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Dict[str, Any]],
    pools: Mapping[str, Sequence[Dict[str, Any]]],
    start: Mapping[str, float],
    end: Mapping[str, float],
) -> Tuple[List[ft.StageDataset], List[Mapping[str, Any]], Mapping[str, Any]]:
    """Backward-compatible entry used by the existing schedule tests."""

    return build_ratio_stages(
        args=args,
        source_rows=source_rows,
        pools=pools,
        schedule_kind="linear",
        start=start,
        end=end,
    )


def build_bengio_stages(
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Dict[str, Any]],
    ranked_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[ft.StageDataset], List[Mapping[str, Any]], Mapping[str, Any]]:
    specs = batch_plan(
        epochs=args.epochs,
        examples_per_epoch=args.train_examples,
        batch_size=args.batch_size,
        source_ratio=args.source_ratio,
    )
    total_generated = sum(spec["n_generated"] for spec in specs)
    if total_generated <= 0:
        raise ValueError("Bengio curriculum needs a positive generated-data fraction")
    # Use the identical source-example stream as the three ratio schedules so
    # the ablation changes only the generated-example schedule.
    source_sampler = CyclingSampler(source_rows, args.seed + 70001)
    stages_rows: List[List[Dict[str, Any]]] = [[] for _ in range(args.epochs)]
    schedule: List[Mapping[str, Any]] = []
    generated_cursor = 0

    for spec in specs:
        generated_rows = []
        margins = []
        ranks = []
        for _ in range(spec["n_generated"]):
            # Repetitions, when necessary, are adjacent rather than cycling
            # easy->hard again. Thus the global generated stream is monotonic.
            index = min(
                len(ranked_rows) - 1,
                int(generated_cursor * len(ranked_rows) / total_generated),
            )
            row = dict(ranked_rows[index])
            generated_rows.append(row)
            margins.append(float(row["exp3_proxy_margin"]))
            ranks.append(int(row["exp3_difficulty_rank"]))
            generated_cursor += 1
        batch = source_sampler.draw(spec["n_source"]) + generated_rows
        stages_rows[spec["epoch_index"]].extend(batch)
        schedule.append(
            {
                **spec,
                "generated_rank_min": min(ranks) if ranks else None,
                "generated_rank_max": max(ranks) if ranks else None,
                "generated_margin_mean": float(sum(margins) / len(margins)) if margins else None,
                "generated_margin_max": max(margins) if margins else None,
                "generated_margin_min": min(margins) if margins else None,
            }
        )

    nonempty = [row for row in schedule if row["generated_margin_mean"] is not None]
    violations = sum(
        1
        for left, right in zip(nonempty, nonempty[1:])
        if right["generated_margin_mean"] > left["generated_margin_mean"] + 1e-8
    )
    rank_violations = sum(
        1
        for left, right in zip(nonempty, nonempty[1:])
        if right["generated_rank_min"] < left["generated_rank_max"]
    )
    if violations or rank_violations:
        raise ValueError(
            "constructed Bengio batches are not monotonic easy->hard: "
            f"margin_violations={violations}, rank_violations={rank_violations}"
        )

    stages = []
    for epoch_idx, rows in enumerate(stages_rows):
        epoch_specs = [row for row in schedule if row["epoch_index"] == epoch_idx]
        summary = ft.summarize_examples(rows)
        summary["exp3_schedule"] = {
            "kind": "bengio_ordered",
            "generated_rank_start": next(
                row["generated_rank_min"]
                for row in epoch_specs
                if row["generated_rank_min"] is not None
            ),
            "generated_rank_end": next(
                row["generated_rank_max"]
                for row in reversed(epoch_specs)
                if row["generated_rank_max"] is not None
            ),
            "dataloader_shuffle": False,
        }
        stages.append(
            ft.StageDataset(
                stage_id=epoch_idx,
                stage_name=f"bengio_epoch_{epoch_idx + 1}",
                ratios={"ordered_synthetic": 1.0},
                examples=rows,
                summary=summary,
            )
        )
    validation = {
        "schedule_kind": "bengio",
        "n_batches": len(schedule),
        "n_ranked_pool_rows": len(ranked_rows),
        "n_generated_draws": total_generated,
        "batch_mean_margin_monotonicity_violations": violations,
        "batch_rank_monotonicity_violations": rank_violations,
        "first_generated_rank": nonempty[0]["generated_rank_min"],
        "last_generated_rank": nonempty[-1]["generated_rank_max"],
        "dataloader_shuffle": False,
        "epoch_stage_mapping": list(range(args.epochs)),
    }
    return stages, schedule, validation


def validate_tokenized_order(args: argparse.Namespace, stages: Sequence[ft.StageDataset]) -> None:
    tokenizer = ft.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for stage in stages:
        tokenized = ft.MCQATokenDataset(
            stage.examples,
            tokenizer=tokenizer,
            max_length=int(args.max_length),
        )
        if len(tokenized) != len(stage.examples):
            raise RuntimeError(
                f"{stage.stage_name}: tokenization dropped "
                f"{len(stage.examples) - len(tokenized)} examples; ordered batch boundaries "
                "would no longer be valid"
            )


def train_ordered(
    args: argparse.Namespace,
    stages: Sequence[ft.StageDataset],
    val_examples: Sequence[Dict[str, Any]],
    data_sources: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_tokenized_order(args, stages)
    original_dataloader = ft.build_dataloader
    original_stage_builder = ft.build_epoch_stage_ids
    original_stage_names = ft.STAGE_NAMES

    def ordered_dataloader(dataset, **kwargs):
        kwargs["shuffle"] = False
        return original_dataloader(dataset, **kwargs)

    ft.build_dataloader = ordered_dataloader
    ft.build_epoch_stage_ids = lambda epochs, num_stages=3: list(range(epochs))
    ft.STAGE_NAMES = tuple(stage.stage_name for stage in stages)
    try:
        mode_tag = "curriculum" if args.curriculum_mode == "on" else "random"
        variant_name = f"ours_{mode_tag}_{args.variant_suffix}"
        model_key = ft.sanitize_name(args.model_name_or_path)
        run_tag = args.run_name or ft.now_str()
        run_dir = ft.ensure_dir(
            Path(args.output_root) / args.dataset / model_key / variant_name / run_tag
        )
        summary = ft.train_one_run(
            args=args,
            variant_name=variant_name,
            run_dir=run_dir,
            stage_datasets=stages,
            val_examples=val_examples,
            data_sources=data_sources,
        )
    finally:
        ft.build_dataloader = original_dataloader
        ft.build_epoch_stage_ids = original_stage_builder
        ft.STAGE_NAMES = original_stage_names
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = ft.build_arg_parser()
    parser.add_argument(
        "--exp3-schedule",
        choices=("uniform", "reverse", "linear", "bengio", "sams"),
        required=True,
    )
    parser.add_argument("--linear-start", default="easy=0.60,hard=0.15,helpful=0.25")
    parser.add_argument("--linear-end", default="easy=0.10,hard=0.60,helpful=0.30")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected_curriculum_mode = "off" if args.exp3_schedule == "uniform" else "on"
    if args.baseline_mode != "ours" or args.curriculum_mode != expected_curriculum_mode:
        raise ValueError(
            "ordered Exp3 trainer requires baseline_mode=ours and "
            f"curriculum_mode={expected_curriculum_mode} for {args.exp3_schedule}"
        )
    if args.shuffle_choices:
        raise ValueError(
            "Exp3 schedule isolation requires --no-shuffle-choices for every ordered condition"
        )
    if not args.variant_suffix:
        raise ValueError("--variant-suffix is required and must not contain the seed")
    if re.search(r"(?:^|_)seed\d+(?:_|$)", args.variant_suffix, flags=re.IGNORECASE):
        raise ValueError(
            f"variant suffix must be seed-independent, got {args.variant_suffix!r}"
        )
    args.dataset = ft.canonical_dataset_name(args.dataset)
    ft.set_seed(int(args.seed))
    splits = ft.prepare_task_splits(
        dataset_name=args.dataset,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
    )
    source_rows = [
        ft.original_example_with_metadata(row, source_name="original")
        for row in splits["train"]
    ]
    val_examples = [
        ft.original_example_with_metadata(row, source_name="val")
        for row in splits["val"]
    ]
    val_examples = ft.maybe_limit_rows(
        val_examples,
        int(args.max_val_examples),
        seed=int(args.seed),
    )
    args.raw_score_key = ""

    if args.exp3_schedule in {"uniform", "reverse", "linear", "sams"}:
        pools, pool_metadata = load_linear_pools(args)
        start = parse_ratios(args.linear_start) if args.exp3_schedule == "linear" else None
        end = parse_ratios(args.linear_end) if args.exp3_schedule == "linear" else None
        stages, batch_schedule, validation = build_ratio_stages(
            args=args,
            source_rows=source_rows,
            pools=pools,
            schedule_kind=args.exp3_schedule,
            start=start,
            end=end,
        )
    else:
        ranked_rows, pool_metadata = load_ranked_bengio_pool(args)
        stages, batch_schedule, validation = build_bengio_stages(
            args=args,
            source_rows=source_rows,
            ranked_rows=ranked_rows,
        )
        validation["choice_order_preserved_from_proxy_ranking"] = True
        validation["choice_order_verified_rows"] = int(
            pool_metadata["choice_order_verified_rows"]
        )

    validation["generated_choice_shuffle"] = False

    data_sources = {
        "baseline_mode": "ours",
        "curriculum_mode": args.curriculum_mode,
        "exp3_ordered_schedule": args.exp3_schedule,
        "pool_metadata": pool_metadata,
        "batch_schedule": batch_schedule,
        "schedule_validation": validation,
        "consumption_contract": {
            "one_dataset_per_epoch": True,
            "epoch_stage_mapping": list(range(args.epochs)),
            "train_dataloader_shuffle": False,
            "generated_choice_shuffle": False,
            "choice_order_preserved_from_proxy_ranking": (
                args.exp3_schedule == "bengio"
            ),
            "batch_size": int(args.batch_size),
        },
    }
    summary = train_ordered(args, stages, val_examples, data_sources)
    summary["exp3_schedule_validation"] = validation
    ft.save_json(Path(summary["run_dir"]) / "run_summary.json", summary)
    print(f"[DONE] ordered schedule={args.exp3_schedule} run_dir={summary['run_dir']}")


if __name__ == "__main__":
    main()
