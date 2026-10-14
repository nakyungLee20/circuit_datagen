from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gens.mcqa import (  # type: ignore
    MCQATokenDataset,
    canonical_dataset_name,
    collate_batch,
    get_option_logits,
    prepare_task_splits,
    set_seed,
)
from applications.seed_dataset import choose_scores_path, load_scores_index  # type: ignore


DEFAULT_GENERATION_ROOT = REPO_ROOT / "results" / "circuit_generation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "curriculum_ft"
BASELINE_MODE_CHOICES = ("original", "prompt_only", "random_circuit", "ours", "raw_score")
CURRICULUM_MODE_CHOICES = ("off", "on")
GENERATED_FILE_KIND_CHOICES = ("raw", "selected", "selected_then_raw")
GENERATED_RECIPE_BANDS = ("easy", "medium", "hard", "helpful")
RAW_SCORE_KEYS = ("aum", "el2n", "gradalign")
RAW_SCORE_EASY_HIGH_DEFAULT = {"aum": True, "el2n": False, "gradalign": True}
STAGE_NAMES = ("warmup", "transition", "challenge")
# Matches leading letter/number prefixes like "A. ", "B) ", "1. ", "2) ", "a. " etc.
# Used to strip generated choices that mistakenly include these prefixes.
# The separator must be followed by whitespace: without that requirement the
# strip is not idempotent and corrupts legitimate tokens such as "S-waves",
# "T-cells", "X-rays" (letter-hyphen) and "1.0g" (digit-period decimals).
CHOICE_LABEL_RE = re.compile(r"^\s*([A-Za-z1-9])[\.\)\:\-]\s+")
QUESTION_PREFIX_RE = re.compile(r"^(Question|Goal|Sentence):\s*", flags=re.IGNORECASE)


@dataclass
class StageDataset:
    stage_id: int
    stage_name: str
    ratios: Dict[str, float]
    examples: List[Dict[str, Any]]
    summary: Dict[str, Any]

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)

def save_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip()).strip("_") or "run"

def maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def avg(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))

def entropy_from_counts(counts: Mapping[str, int]) -> float:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        prob = float(count) / total
        if prob > 0:
            ent -= prob * math.log(prob)
    return float(ent)

def normalize_question_text(text: str) -> str:
    return QUESTION_PREFIX_RE.sub("", (text or "").strip())

def ensure_question_prefix(text: str) -> str:
    clean = normalize_question_text(text)
    return f"Question: {clean}" if clean else ""

def strip_choice_prefix(text: Any) -> str:
    return CHOICE_LABEL_RE.sub("", str(text or "").strip())

def parse_ratio_map(raw: str, allowed_keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for piece in (raw or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Expected key=value in ratio spec, got: {piece!r}")
        key, value = piece.split("=", 1)
        key = key.strip()
        if allowed_keys is not None and key not in allowed_keys:
            raise ValueError(f"Unsupported ratio key: {key}. allowed={list(allowed_keys)}")
        try:
            ratios[key] = float(value)
        except ValueError as e:
            raise ValueError(f"Invalid ratio value in {piece!r}") from e
    if not ratios:
        raise ValueError(f"Empty ratio spec: {raw!r}")
    if all(v <= 0 for v in ratios.values()):
        raise ValueError(f"At least one positive ratio is required: {raw!r}")
    return ratios

def normalize_ratio_map(raw: Mapping[str, float]) -> Dict[str, float]:
    positive = {str(k): float(v) for k, v in raw.items() if float(v) > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError(f"Invalid ratio map: {raw}")
    return {k: float(v / total) for k, v in positive.items()}

def allocate_counts(total: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    ratios = normalize_ratio_map(ratios)
    items = list(ratios.items())
    counts = {k: int(math.floor(total * v)) for k, v in items}
    remainder = int(total - sum(counts.values()))
    ranked = sorted(items, key=lambda kv: (total * kv[1] - counts[kv[0]], kv[0]), reverse=True,)
    for idx in range(remainder):
        counts[ranked[idx % len(ranked)][0]] += 1
    return counts

def parse_csv_set(raw: str) -> List[str]:
    return [piece.strip() for piece in (raw or "").split(",") if piece.strip()]

def load_first_jsonl_row(path: Path) -> Optional[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None

def generated_file_suffix(file_kind: str) -> str:
    if file_kind == "raw":
        return "raw_candidates"
    if file_kind == "selected":
        return "selected"
    raise ValueError(file_kind)

def resolve_generator_model_dir(root: Path, generator_model_key: str) -> Path:
    if generator_model_key:
        path = root / generator_model_key
        if not path.is_dir():
            raise FileNotFoundError(f"generator model dir not found: {path}")
        return path
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(subdirs) == 1:
        return subdirs[0]
    raise ValueError(
        f"Could not infer generator model directory under {root}. "
        f"Found {len(subdirs)} candidates; pass --generator-model-key explicitly."
    )

def match_generated_metadata(row: Mapping[str, Any], *, recipe: str, prompt_mode: str, gen_mode: str, circuit_target_mode: str,) -> bool:
    steering = row.get("steering", {}) if isinstance(row.get("steering"), dict) else {}
    row_recipe = str(row.get("target_band", "")).strip()
    row_prompt_mode = str(steering.get("prompt_mode", "")).strip() or "fixed"
    row_gen_mode = str(steering.get("gen_mode", "")).strip() or "both"
    row_circuit_target_mode = str(steering.get("circuit_target_mode", "")).strip() or "discovered"
    return (
        row_recipe == recipe
        and row_prompt_mode == prompt_mode
        and row_gen_mode == gen_mode
        and row_circuit_target_mode == circuit_target_mode
    )

def find_latest_generated_file(
    *,
    generation_root: Path,
    generator_model_key: str,
    recipe: str,
    file_kind: str,
    prompt_mode: str,
    gen_mode: str,
    circuit_target_mode: str,
    path_must_contain: str = "",
) -> Path:
    model_dir = resolve_generator_model_dir(generation_root, generator_model_key)
    suffix = generated_file_suffix(file_kind)
    recipe_root = model_dir / recipe
    if recipe_root.is_dir():
        candidates = sorted(recipe_root.rglob(f"{recipe}_{suffix}.jsonl"))
    else:
        candidates = sorted(model_dir.rglob(f"{recipe}_{suffix}.jsonl"))
    matched: List[Tuple[float, Path]] = []
    for path in candidates:
        if path_must_contain and path_must_contain not in str(path):
            continue
        first_row = load_first_jsonl_row(path)
        if first_row is None:
            continue
        if match_generated_metadata(
            first_row,
            recipe=recipe,
            prompt_mode=prompt_mode,
            gen_mode=gen_mode,
            circuit_target_mode=circuit_target_mode,
        ):
            matched.append((path.stat().st_mtime, path))
    if not matched:
        raise FileNotFoundError(
            "Could not find generated file matching "
            f"recipe={recipe}, file_kind={file_kind}, prompt_mode={prompt_mode}, "
            f"gen_mode={gen_mode}, circuit_target_mode={circuit_target_mode}, "
            f"path_must_contain={path_must_contain!r} under {model_dir}"
        )
    matched.sort(key=lambda item: (item[0], str(item[1])))
    return matched[-1][1]

def resolve_answer_label_from_row(answer_raw: Any, choices: Sequence[str]) -> Optional[str]:
    """Resolve a raw answer string to a canonical uppercase label (A/B/C/D/...).

    Handles three formats produced by generation, checked in this priority order:
    1. Full choice text (possibly prefixed): strip leading "A. "/"B) "/etc., then
       text-match against choices. Must come first to avoid "Bacteria" → "B" misparse.
    2. Direct alpha label: exactly one character "A"/"b"/"C" → upper-cased label.
    3. Numeric 1-based index: exactly one digit "1"→"A", "2"→"B", etc.
    """
    label_space = [chr(ord("A") + idx) for idx in range(len(choices))]
    answer = str(answer_raw or "").strip()
    if not answer:
        return None
    answer_upper = answer.upper()
    # Case 1: full choice text with optional prefix — strip prefix and text-match.
    # This must precede the single-char checks: a full text like "Bacteria" must not be
    # misread as label "B" just because its first character happens to be in label_space.
    normalized_answer = strip_choice_prefix(answer).lower()
    normalized_choices = [strip_choice_prefix(choice).lower() for choice in choices]
    if normalized_answer in normalized_choices:
        idx = normalized_choices.index(normalized_answer)
        return label_space[idx]
    # Case 2: direct alpha label — only when the entire answer is a single letter.
    if len(answer) == 1 and answer_upper in label_space:
        return answer_upper
    # Case 3: numeric 1-based index — only when the entire answer is a single digit.
    if len(answer) == 1 and answer.isdigit():
        digit_idx = int(answer) - 1
        if 0 <= digit_idx < len(label_space):
            return label_space[digit_idx]
    return None

def generated_row_to_example(row: Mapping[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    parsed = row.get("parsed", {}) if isinstance(row.get("parsed"), dict) else {}
    question = str(row.get("question") or parsed.get("question") or "").strip()
    choices_raw = row.get("choices") or parsed.get("choices") or []
    if not isinstance(choices_raw, list):
        choices_raw = []
    cleaned_choices = [strip_choice_prefix(choice) for choice in choices_raw if str(choice).strip()]
    if not question or len(cleaned_choices) < 2:
        return None

    gold_label = resolve_answer_label_from_row(row.get("answer") or parsed.get("answer"), cleaned_choices)
    if gold_label is None:
        return None
    label_space = [chr(ord("A") + idx) for idx in range(len(cleaned_choices))]
    if gold_label not in label_space:
        return None

    steering = row.get("steering", {}) if isinstance(row.get("steering"), dict) else {}
    internal_scores = row.get("internal_scores", {}) if isinstance(row.get("internal_scores"), dict) else {}
    sanity = row.get("sanity", {}) if isinstance(row.get("sanity"), dict) else {}
    source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}

    candidate_id = str(row.get("candidate_id") or row.get("record_id") or "")
    target_band = str(row.get("target_band") or "generated")
    return {
        "hf_id": f"generated/{target_band}/{candidate_id}",
        "task": dataset_name,
        "question_text": ensure_question_prefix(question),
        "choices": cleaned_choices,
        "label_space": label_space,
        "gold_label": gold_label,
        "origin": "generated",
        "generated_recipe": target_band,
        "generated_candidate_id": candidate_id,
        "source_record_id": str(row.get("record_id") or ""),
        "source_hf_id": str(source.get("hf_id") or source.get("id") or ""),
        "source_question": str(source.get("question") or ""),
        "rationale": str(row.get("rationale") or parsed.get("rationale") or ""),
        "parsed_difficulty": str(parsed.get("difficulty") or ""),
        "parsed_helpful": bool(parsed.get("helpful")) if "helpful" in parsed else None,
        "generated_prompt_mode": str(steering.get("prompt_mode") or ""),
        "generated_gen_mode": str(steering.get("gen_mode") or ""),
        "generated_circuit_target_mode": str(steering.get("circuit_target_mode") or "discovered"),
        "selection_policy": str(row.get("selection_policy") or ""),
        "final_score": maybe_float(row.get("final_score")),
        "difficulty_proxy": maybe_float(internal_scores.get("difficulty_proxy")),
        "helpfulness_proxy": maybe_float(internal_scores.get("helpfulness_proxy")),
        "base_ppl": maybe_float(sanity.get("base_ppl")),
        "json_parse_ok": bool(sanity.get("json_parse_ok", False)),
    }

def original_example_with_metadata(example: Mapping[str, Any], source_name: str) -> Dict[str, Any]:
    row = dict(example)
    row["origin"] = source_name
    row["question_text"] = ensure_question_prefix(str(row.get("question_text") or ""))
    return row

def sample_from_pool(pool: Sequence[Dict[str, Any]], *, n: int, seed: int, allow_replacement: bool,) -> List[Dict[str, Any]]:
    rows = list(pool)
    if n <= 0:
        return []
    if not rows:
        raise ValueError("Cannot sample from an empty pool")
    rng = random.Random(seed)
    if n <= len(rows):
        return rng.sample(rows, n)
    if not allow_replacement:
        raise ValueError(
            f"Requested {n} examples from a pool of size {len(rows)} without replacement. "
            "Either generate a larger pool or enable --allow-pool-oversample."
        )
    print(
        f"[WARN] oversampling: pool_size={len(rows)} < requested={n} "
        f"(x{n / len(rows):.2f}) — sampling with replacement. "
        "Some examples will appear more than once. "
        "Consider generating a larger pool or adjusting --train-examples."
    )
    return [rows[rng.randrange(len(rows))] for _ in range(n)]

def compose_from_pools(
    pools: Mapping[str, Sequence[Dict[str, Any]]],
    ratios: Mapping[str, float],
    *,
    total_examples: int,
    seed: int,
    allow_replacement: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    counts = allocate_counts(total_examples, ratios)
    combined: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for idx, (pool_name, count) in enumerate(counts.items()):
        sampled = sample_from_pool(
            pools[pool_name],
            n=count,
            seed=seed + idx * 1009,
            allow_replacement=allow_replacement,
        )
        combined.extend(sampled)
        source_counts[pool_name] = int(len(sampled))
    shuffler = random.Random(seed + 7919)
    shuffler.shuffle(combined)
    return combined, {"requested_ratios": dict(ratios), "normalized_counts": source_counts}


def summarize_examples(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter(str(row.get("gold_label") or "") for row in rows if row.get("gold_label"))
    origin_counts = Counter(str(row.get("origin") or "") for row in rows if row.get("origin"))
    band_counts = Counter(str(row.get("generated_recipe") or "") for row in rows if row.get("generated_recipe"))
    raw_band_counts = Counter(str(row.get("raw_score_band") or "") for row in rows if row.get("raw_score_band"))
    prompt_mode_counts = Counter(str(row.get("generated_prompt_mode") or "") for row in rows if row.get("generated_prompt_mode"))
    circuit_mode_counts = Counter(
        str(row.get("generated_circuit_target_mode") or "") for row in rows if row.get("generated_circuit_target_mode")
    )
    unique_questions = {str(row.get("question_text") or "").strip() for row in rows if row.get("question_text")}
    unique_source_ids = {str(row.get("source_hf_id") or "") for row in rows if row.get("source_hf_id")}
    return {
        "n": int(len(rows)),
        "label_counts": dict(label_counts),
        "label_entropy": float(entropy_from_counts(label_counts)),
        "origin_counts": dict(origin_counts),
        "generated_band_counts": dict(band_counts),
        "raw_score_band_counts": dict(raw_band_counts),
        "prompt_mode_counts": dict(prompt_mode_counts),
        "circuit_target_mode_counts": dict(circuit_mode_counts),
        "unique_questions": int(len(unique_questions)),
        "unique_source_hf_ids": int(len(unique_source_ids)),
        "mean_num_choices": avg([float(len(row.get("choices", []))) for row in rows]),
        "mean_question_words": avg([float(len(str(row.get("question_text") or "").split())) for row in rows]),
        "mean_rationale_words": avg([float(len(str(row.get("rationale") or "").split())) for row in rows]),
        "json_parse_rate": avg([1.0 if row.get("json_parse_ok") else 0.0 for row in rows]),
        "mean_final_score": avg([maybe_float(row.get("final_score")) for row in rows]),
        "mean_difficulty_proxy": avg([maybe_float(row.get("difficulty_proxy")) for row in rows]),
        "mean_helpfulness_proxy": avg([maybe_float(row.get("helpfulness_proxy")) for row in rows]),
        "mean_base_ppl": avg([maybe_float(row.get("base_ppl")) for row in rows]),
    }

def replicate_constant_stage_datasets(rows: Sequence[Dict[str, Any]], ratios: Mapping[str, float]) -> List[StageDataset]:
    normalized = normalize_ratio_map(ratios)
    base_rows = [dict(row) for row in rows]
    base_summary = summarize_examples(base_rows)
    return [
        StageDataset(
            stage_id=stage_id,
            stage_name=STAGE_NAMES[stage_id],
            ratios=dict(normalized),
            examples=[dict(row) for row in base_rows],
            summary=dict(base_summary),
        )
        for stage_id in range(3)
    ]

def build_anchored_pools_from_flat_ratios(
    pools: Mapping[str, Sequence[Dict[str, Any]]],
    anchor_ratios: Mapping[str, float],
    total_anchor: int,
    seed: int,
    allow_replacement: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pre-sample fixed per-band anchor pools for --anchor-replay mode.

    Each band's anchor is a frozen random subset of the full band pool.
    All curriculum stages sample from these anchors rather than the full pools,
    guaranteeing that later stages only replay examples already seen in stage 1
    (strict subset replay). Stages may oversample within an anchor band (with
    replacement) when a stage ratio exceeds the anchor band size.
    """
    counts = allocate_counts(total_anchor, anchor_ratios)
    anchors: Dict[str, List[Dict[str, Any]]] = {}
    for idx, (band, count) in enumerate(sorted(counts.items())):
        if band not in pools or count <= 0:
            continue
        anchors[band] = sample_from_pool(
            pools[band], n=count, seed=seed + idx * 1009, allow_replacement=allow_replacement
        )
    return anchors


def _inject_source_rows_into_stages(
    stages: List[StageDataset],
    source_pool: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    seed_base: int,
) -> List[StageDataset]:
    """Supplement each stage's generated rows with rows from the real source dataset.

    For each stage:
      - keeps the first (1 - source_ratio) * train_examples generated rows
        (already shuffled by compose_from_pools, so this is an unbiased prefix)
      - samples source_ratio * train_examples rows from source_pool independently
        per stage (different seed each stage for variety)
      - shuffles the combined set and returns a new StageDataset

    source_ratio=0.0 → no-op, original stages returned unchanged.
    Applies only to ours / prompt_only / random_circuit modes; original and
    raw_score already use source data and are never routed here.
    """
    source_ratio = float(getattr(args, "source_ratio", 0.0))
    if source_ratio <= 0.0:
        return stages
    n_src = round(source_ratio * args.train_examples)
    n_gen = args.train_examples - n_src
    if n_src <= 0:
        return stages
    allow = bool(args.allow_pool_oversample) or len(source_pool) < n_src
    print(
        f"[SOURCE MIX] source_ratio={source_ratio:.3f}  "
        f"n_gen={n_gen}  n_src={n_src}  source_pool_size={len(source_pool)}"
    )
    new_stages: List[StageDataset] = []
    for i, stage in enumerate(stages):
        gen_rows = list(stage.examples)[:n_gen]
        src_rows = sample_from_pool(
            list(source_pool), n=n_src,
            seed=seed_base + i * 3571,
            allow_replacement=allow,
        )
        combined = gen_rows + src_rows
        rng = random.Random(seed_base + i * 7331)
        rng.shuffle(combined)
        summary = summarize_examples(combined)
        new_stages.append(StageDataset(stage.stage_id, stage.stage_name, stage.ratios, combined, summary))
        print(f"[SOURCE MIX] stage={stage.stage_id + 1} gen={len(gen_rows)} src={len(src_rows)} total={len(combined)}")
    return new_stages


def _make_flat_stage_datasets(
    pools: Mapping[str, Sequence[Dict[str, Any]]],
    ratios: Mapping[str, float],
    *,
    args: argparse.Namespace,
) -> Tuple[List[StageDataset], Dict[str, Any]]:
    """Build 3 constant-ratio stage datasets for curriculum_mode=off.

    off_resample=False (default): sample once, replicate identical rows to all
        3 stages — same 10 k unique examples, curriculum=off baseline.
    off_resample=True: independently re-draw each stage from the same pool
        using a different seed, so stages may see different examples while the
        pool (and therefore data support) stays the same.
    """
    normalized = normalize_ratio_map(ratios)
    allow = bool(args.allow_pool_oversample)
    if not getattr(args, "off_resample", False):
        rows, sample_meta = compose_from_pools(
            pools, ratios, total_examples=args.train_examples,
            seed=args.seed, allow_replacement=allow,
        )
        return replicate_constant_stage_datasets(rows, ratios), {"composition": sample_meta}
    stages: List[StageDataset] = []
    metas: Dict[str, Any] = {}
    for stage_id in range(3):
        s_seed = args.seed + stage_id
        rows, sample_meta = compose_from_pools(
            pools, ratios, total_examples=args.train_examples,
            seed=s_seed, allow_replacement=allow,
        )
        summary = summarize_examples(rows)
        stages.append(StageDataset(stage_id, STAGE_NAMES[stage_id], normalized, rows, summary))
        metas[f"stage_{stage_id + 1}"] = {"composition": sample_meta, "summary": summary}
    return stages, {"stages": metas}

def flatten_numeric(prefix: str, payload: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}"
        if isinstance(value, Mapping):
            out.update(flatten_numeric(name, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out

def build_epoch_stage_ids(epochs: int, num_stages: int = 3) -> List[int]:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    return [min(num_stages - 1, int(math.floor(epoch_idx * num_stages / epochs))) for epoch_idx in range(epochs)]

def maybe_limit_rows(rows: Sequence[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    arr = list(rows)
    if limit <= 0 or len(arr) <= limit:
        return arr
    rng = random.Random(seed)
    return rng.sample(arr, limit)

def load_generated_pool_from_file(path: Path, dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_rows = load_jsonl(path)
    converted: List[Dict[str, Any]] = []
    dropped = 0
    for row in raw_rows:
        example = generated_row_to_example(row, dataset_name)
        if example is None:
            dropped += 1
            continue
        converted.append(example)
    drop_pct = 100.0 * dropped / max(1, len(raw_rows))
    drop_tag = f" [WARN: high drop rate]" if drop_pct >= 20.0 else ""
    print(
        f"[POOL] {path.name}: loaded={len(raw_rows)} "
        f"converted={len(converted)} dropped={dropped} ({drop_pct:.1f}%){drop_tag}"
    )
    meta = {
        "path": str(path),
        "num_rows_loaded": len(raw_rows),
        "num_rows_converted": len(converted),
        "num_rows_dropped": dropped,
        "summary": summarize_examples(converted),
    }
    return converted, meta

def _merge_selected_with_raw_extras(
    selected_path: Path,
    raw_path: Path,
    dataset_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load selected pool then append unique raw candidates not already in selected.

    Deduplication is done by generated_candidate_id so that every example in
    the merged pool is unique.  The resulting pool is ordered selected-first,
    which means sample_from_pool will prefer quality-filtered examples when it
    does not need to oversample.
    """
    selected_rows, selected_meta = load_generated_pool_from_file(selected_path, dataset_name)
    raw_rows, raw_meta = load_generated_pool_from_file(raw_path, dataset_name)

    selected_ids = {
        ex["generated_candidate_id"]
        for ex in selected_rows
        if ex.get("generated_candidate_id")
    }
    extra_rows = [
        ex for ex in raw_rows
        if not ex.get("generated_candidate_id") or ex["generated_candidate_id"] not in selected_ids
    ]
    merged = selected_rows + extra_rows

    overlap = len(raw_rows) - len(extra_rows)
    print(
        f"[POOL MERGE] selected={len(selected_rows)}  "
        f"raw_unique_extras={len(extra_rows)}  raw_overlap_skipped={overlap}  "
        f"merged_total={len(merged)}"
    )

    # Keep num_rows_* at top level so build_generated_recipe_pools summary still works.
    file_meta = {
        "file_kind": "selected_then_raw",
        "selected": selected_meta,
        "raw_extras": raw_meta,
        "num_rows_loaded": selected_meta["num_rows_loaded"] + raw_meta["num_rows_loaded"],
        "num_rows_converted": len(merged),
        "num_rows_dropped": (
            selected_meta["num_rows_dropped"] + raw_meta["num_rows_dropped"] + overlap
        ),
        "merged_pool_size": len(merged),
    }
    return merged, file_meta


def shuffle_choices_for_row(row: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Randomly permute the answer choices, updating gold_label to match the new position.

    Fixes the A-position bias that arises because LLMs tend to place the correct
    answer first when generating MCQ data (~38% A vs. 25% expected).
    Only acts when choices/label_space are consistent and gold_label is found.
    """
    choices = list(row.get("choices", []))
    label_space = list(row.get("label_space", []))
    gold_label = str(row.get("gold_label", ""))
    if len(choices) < 2 or len(choices) != len(label_space) or gold_label not in label_space:
        return row
    old_idx = label_space.index(gold_label)
    perm = list(range(len(choices)))
    rng.shuffle(perm)
    new_choices = [choices[i] for i in perm]
    new_gold_label = label_space[perm.index(old_idx)]
    return {**row, "choices": new_choices, "gold_label": new_gold_label}


def build_generated_recipe_pools(*, args: argparse.Namespace, recipes: Sequence[str], file_kind: str, circuit_target_mode: str,) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    generation_root = Path(args.generation_root).expanduser()
    pools: Dict[str, List[Dict[str, Any]]] = {}
    meta: Dict[str, Any] = {
        "recipes": list(recipes),
        "file_kind": file_kind,
        "circuit_target_mode": circuit_target_mode,
        "generated_path_contains": str(getattr(args, "generated_path_contains", "") or ""),
        "files": {},
    }

    def _find(fk: str, recipe: str) -> Path:
        return find_latest_generated_file(
            generation_root=generation_root,
            generator_model_key=str(args.generator_model_key or ""),
            recipe=recipe,
            file_kind=fk,
            prompt_mode=args.generated_prompt_mode,
            gen_mode=args.generated_gen_mode,
            circuit_target_mode=circuit_target_mode,
            path_must_contain=str(getattr(args, "generated_path_contains", "") or ""),
        )

    for recipe in recipes:
        if file_kind == "selected_then_raw":
            rows, file_meta = _merge_selected_with_raw_extras(
                selected_path=_find("selected", recipe),
                raw_path=_find("raw", recipe),
                dataset_name=args.dataset,
            )
        else:
            rows, file_meta = load_generated_pool_from_file(
                _find(file_kind, recipe), dataset_name=args.dataset
            )
        if getattr(args, "shuffle_choices", True):
            rows = [
                shuffle_choices_for_row(ex, random.Random(args.seed + i))
                for i, ex in enumerate(rows)
            ]
        pools[recipe] = rows
        meta["files"][recipe] = file_meta

    total_loaded = sum(m["num_rows_loaded"] for m in meta["files"].values())
    total_converted = sum(m["num_rows_converted"] for m in meta["files"].values())
    total_dropped = sum(m["num_rows_dropped"] for m in meta["files"].values())
    print(
        f"[POOL SUMMARY] file_kind={file_kind} target={circuit_target_mode} | "
        f"total: loaded={total_loaded} converted={total_converted} dropped={total_dropped} "
        f"({100.0 * total_dropped / max(1, total_loaded):.1f}%) | "
        + "  ".join(f"{r}={len(pools[r])}" for r in recipes)
    )
    return pools, meta

def build_raw_score_bands(
    *,
    train_examples: Sequence[Dict[str, Any]],
    dataset: str,
    split: str,
    score_key: str,
    easy_high: bool,
    easy_frac: float,
    hard_frac: float,
    scores_json: Optional[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if easy_frac < 0 or hard_frac < 0 or easy_frac + hard_frac >= 1.0:
        raise ValueError(
            "raw score band fractions must satisfy 0 <= easy_frac, hard_frac and easy_frac + hard_frac < 1.0"
        )
    scores_path = choose_scores_path(dataset, scores_json)
    scores_by_id, score_meta = load_scores_index(scores_path, dataset=dataset, split=split)

    scored_rows: List[Tuple[float, Dict[str, Any]]] = []
    missing = 0
    for example in train_examples:
        hf_id = str(example.get("hf_id") or "")
        if hf_id not in scores_by_id or score_key not in scores_by_id[hf_id]:
            missing += 1
            continue
        score_value = float(scores_by_id[hf_id][score_key])
        row = original_example_with_metadata(example, source_name="raw_score")
        row["raw_score_key"] = score_key
        row["raw_score_value"] = score_value
        scored_rows.append((score_value, row))

    scored_rows.sort(key=lambda item: item[0], reverse=bool(easy_high))
    total = len(scored_rows)
    easy_n = int(math.floor(total * easy_frac))
    hard_n = int(math.floor(total * hard_frac))
    middle_n = max(0, total - easy_n - hard_n)

    easy_rows = [dict(row, raw_score_band="easy") for _, row in scored_rows[:easy_n]]
    medium_rows = [dict(row, raw_score_band="medium") for _, row in scored_rows[easy_n : easy_n + middle_n]]
    hard_rows = [dict(row, raw_score_band="hard") for _, row in scored_rows[easy_n + middle_n :]]
    meta = {
        "score_key": score_key,
        "scores_json": str(scores_path),
        "score_meta": score_meta,
        "easy_high": bool(easy_high),
        "easy_frac": float(easy_frac),
        "hard_frac": float(hard_frac),
        "num_missing_scores": int(missing),
        "bands": {
            "easy": summarize_examples(easy_rows),
            "medium": summarize_examples(medium_rows),
            "hard": summarize_examples(hard_rows),
        },
    }
    return {"easy": easy_rows, "medium": medium_rows, "hard": hard_rows}, meta

def build_stage_datasets_for_variant(
    *,
    args: argparse.Namespace,
    train_examples: Sequence[Dict[str, Any]],
    baseline_mode: str,
    curriculum_mode: str,
    raw_score_key: Optional[str] = None,
) -> Tuple[List[StageDataset], Dict[str, Any]]:
    metadata: Dict[str, Any] = {"baseline_mode": baseline_mode, "curriculum_mode": curriculum_mode}

    if baseline_mode == "original":
        if curriculum_mode != "off":
            raise ValueError("baseline_mode=original only supports --curriculum-mode off")
        pool = [original_example_with_metadata(ex, source_name="original") for ex in train_examples]
        flat_stages, flat_meta = _make_flat_stage_datasets(
            {"original": pool}, {"original": 1.0}, args=args
        )
        metadata["pool_meta"] = {"original": summarize_examples(pool), **flat_meta}
        return flat_stages, metadata

    if baseline_mode == "prompt_only":
        if curriculum_mode != "off":
            raise ValueError("baseline_mode=prompt_only only supports --curriculum-mode off")
        pools, pool_meta = build_generated_recipe_pools(
            args=args,
            recipes=["naive"],
            file_kind=args.prompt_only_file_kind,
            circuit_target_mode="discovered",
        )
        flat_stages, flat_meta = _make_flat_stage_datasets(
            {"naive": pools["naive"]}, {"naive": 1.0}, args=args
        )
        metadata["pool_meta"] = {"generated": pool_meta, **flat_meta}
        flat_stages = _inject_source_rows_into_stages(flat_stages, train_examples, args, seed_base=args.seed + 20000)
        return flat_stages, metadata

    if baseline_mode == "random_circuit":
        if curriculum_mode != "off":
            raise ValueError("baseline_mode=random_circuit only supports --curriculum-mode off")
        random_recipes = parse_csv_set(args.random_circuit_recipes)
        if not random_recipes:
            raise ValueError("--random-circuit-recipes must include at least one recipe")
        pools, pool_meta = build_generated_recipe_pools(
            args=args,
            recipes=random_recipes,
            file_kind=args.random_circuit_file_kind,
            circuit_target_mode="random",
        )
        if getattr(args, "global_random_circuit", False):
            # Pool all recipe bands together and sample uniformly — truly global random baseline.
            all_pooled = [row for recipe_rows in pools.values() for row in recipe_rows]
            active_pools: Mapping[str, Sequence[Dict[str, Any]]] = {"all": all_pooled}
            active_ratios: Mapping[str, float] = {"all": 1.0}
        else:
            active_pools = pools
            active_ratios = parse_ratio_map(args.random_circuit_ratios, allowed_keys=random_recipes)
        flat_stages, flat_meta = _make_flat_stage_datasets(active_pools, active_ratios, args=args)
        metadata["pool_meta"] = {
            "generated": pool_meta,
            "global_random_circuit": bool(getattr(args, "global_random_circuit", False)),
            **flat_meta,
        }
        flat_stages = _inject_source_rows_into_stages(flat_stages, train_examples, args, seed_base=args.seed + 30000)
        return flat_stages, metadata

    if baseline_mode == "ours":
        # Collect only the bands actually referenced by any stage or flat ratio.
        # This lets prompt-mode runs (source_none) skip bands like "medium" that
        # were never generated with that prompt/gen_mode combination.
        if curriculum_mode == "off":
            ratio_strs_for_pool_discovery = [args.flat_generated_ratios]
        else:
            ratio_strs_for_pool_discovery = [
                args.flat_generated_ratios,
                args.stage1_ratios,
                args.stage2_ratios,
                args.stage3_ratios,
            ]
        referenced_bands: set = set()
        for rs in ratio_strs_for_pool_discovery:
            referenced_bands.update(parse_ratio_map(rs).keys())
        recipes = [b for b in GENERATED_RECIPE_BANDS if b in referenced_bands]
        if not recipes:
            recipes = list(GENERATED_RECIPE_BANDS)
        pools, pool_meta = build_generated_recipe_pools(
            args=args,
            recipes=recipes,
            file_kind=args.ours_file_kind,
            circuit_target_mode="discovered",
        )
        metadata["pool_meta"] = {"generated": pool_meta}
        if curriculum_mode == "off":
            ratios = parse_ratio_map(args.flat_generated_ratios, allowed_keys=recipes)
            flat_stages, flat_meta = _make_flat_stage_datasets(pools, ratios, args=args)
            metadata["pool_meta"].update(flat_meta)
            flat_stages = _inject_source_rows_into_stages(flat_stages, train_examples, args, seed_base=args.seed + 40000)
            return flat_stages, metadata

        stage_specs = [
            parse_ratio_map(args.stage1_ratios, allowed_keys=recipes),
            parse_ratio_map(args.stage2_ratios, allowed_keys=recipes),
            parse_ratio_map(args.stage3_ratios, allowed_keys=recipes),
        ]
        if getattr(args, "anchor_replay", False):
            # Pre-build fixed per-band anchor pools (sized by flat_generated_ratios).
            # All stages sample from these anchors → later-stage easy/helpful samples
            # are guaranteed to be subsets of what stage-1 already saw (strict replay).
            # Stages may oversample within an anchor (allow_replacement=True) because
            # a stage's band ratio can exceed the anchor band's size.
            anchor_ratios = parse_ratio_map(args.flat_generated_ratios, allowed_keys=recipes)
            active_pools = build_anchored_pools_from_flat_ratios(
                pools, anchor_ratios,
                total_anchor=args.train_examples,
                seed=args.seed,
                allow_replacement=bool(args.allow_pool_oversample),
            )
            stage_allow_replacement = True
            metadata["pool_meta"]["anchor_replay"] = True
            metadata["pool_meta"]["anchor_ratios"] = normalize_ratio_map(anchor_ratios)
            metadata["pool_meta"]["anchor_sizes"] = {k: len(v) for k, v in active_pools.items()}
        else:
            active_pools = pools
            stage_allow_replacement = bool(args.allow_pool_oversample)

        stages: List[StageDataset] = []
        stage_meta: Dict[str, Any] = {}
        for stage_id, ratios in enumerate(stage_specs):
            rows, sample_meta = compose_from_pools(
                active_pools,
                ratios,
                total_examples=args.train_examples,
                seed=args.seed + stage_id ,
                allow_replacement=stage_allow_replacement,
            )
            normalized = normalize_ratio_map(ratios)
            stage_summary = summarize_examples(rows)
            stages.append(StageDataset(stage_id, STAGE_NAMES[stage_id], normalized, rows, stage_summary))
            stage_meta[f"stage_{stage_id + 1}"] = {"ratios": normalized, "composition": sample_meta, "summary": stage_summary}
        metadata["pool_meta"]["stages"] = stage_meta
        stages = _inject_source_rows_into_stages(stages, train_examples, args, seed_base=args.seed + 50000)
        return stages, metadata

    if baseline_mode == "raw_score":
        if raw_score_key is None:
            raise ValueError("raw_score baseline requires raw_score_key")
        # 1) 먼저 동일한 개수의 seed rows를 뽑는다. "sample first, band later"를 보장.
        seed_rows = sample_from_pool(
            train_examples,
            n=args.train_examples,
            seed=args.seed,
            allow_replacement=False,
        )
        easy_high = raw_score_key in set(parse_csv_set(args.raw_score_easy_high))
        bands, band_meta = build_raw_score_bands(
            train_examples=seed_rows,
            dataset=args.dataset,
            split="train",
            score_key=raw_score_key,
            easy_high=easy_high,
            easy_frac=float(args.raw_band_easy_frac),
            hard_frac=float(args.raw_band_hard_frac),
            scores_json=args.scores_json,
        )
        metadata["pool_meta"] = {
            "raw_score": band_meta,
            "seed_rows_count": int(len(seed_rows)),
            "seed_rows_summary": summarize_examples(seed_rows),
        }
        if curriculum_mode == "off":
            all_rows = bands["easy"] + bands["medium"] + bands["hard"]
            flat_stages, flat_meta = _make_flat_stage_datasets(
                {"raw_score": all_rows}, {"raw_score": 1.0}, args=args
            )
            metadata["pool_meta"].update(flat_meta)
            return flat_stages, metadata

        raw_stage_specs = [
            parse_ratio_map(args.raw_stage1_ratios, allowed_keys=("easy", "medium", "hard")),
            parse_ratio_map(args.raw_stage2_ratios, allowed_keys=("easy", "medium", "hard")),
            parse_ratio_map(args.raw_stage3_ratios, allowed_keys=("easy", "medium", "hard")),
        ]
        stages = []
        raw_stage_meta: Dict[str, Any] = {}
        for stage_id, ratios in enumerate(raw_stage_specs):
            rows, sample_meta = compose_from_pools(
                bands,
                ratios,
                total_examples=args.train_examples,
                seed=args.seed + stage_id,
                allow_replacement=bool(args.allow_pool_oversample),
            )
            normalized = normalize_ratio_map(ratios)
            stage_summary = summarize_examples(rows)
            stages.append(StageDataset(stage_id, STAGE_NAMES[stage_id], normalized, rows, stage_summary))
            raw_stage_meta[f"stage_{stage_id + 1}"] = {"ratios": normalized, "composition": sample_meta, "summary": stage_summary}
        metadata["pool_meta"]["stages"] = raw_stage_meta
        return stages, metadata

    raise ValueError(f"Unsupported baseline_mode={baseline_mode}")


def save_checkpoint(path: Path, model: Any, epoch: int, global_step: int, meta: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "meta": dict(meta),
        },
        str(path),
    )


@torch.no_grad()
def eval_mcqa_metrics(model: Any, dataloader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    entropies: List[float] = []
    margins: List[float] = []
    pred_counts: Counter[str] = Counter()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits
        for idx in range(int(logits.shape[0])):
            pos = int(pred_pos[idx].item())
            label_space = list(batch["label_space"][idx])
            label_token_ids = batch["label_token_ids"][idx]
            gold_label = str(batch["gold_label"][idx])
            option_logits = get_option_logits(logits[idx, pos, :], label_space, label_token_ids)
            probs = torch.softmax(option_logits, dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            pred_label = label_space[pred_idx]
            pred_counts[pred_label] += 1
            gold_idx = label_space.index(gold_label)
            correct += int(pred_idx == gold_idx)
            total += 1
            ent = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())
            topk = torch.topk(probs, k=min(2, int(probs.shape[0]))).values
            margin = float(topk[0].item()) if int(topk.shape[0]) == 1 else float(topk[0].item() - topk[1].item())
            entropies.append(ent)
            margins.append(margin)
    model.train()
    return {
        "accuracy": float(correct / max(1, total)),
        "mean_option_entropy": float(sum(entropies) / max(1, len(entropies))),
        "mean_margin": float(sum(margins) / max(1, len(margins))),
        "pred_label_entropy": float(entropy_from_counts(pred_counts)),
        "n_eval": int(total),
    }


def resolve_torch_dtype(dtype_name: str, device: str) -> Optional[torch.dtype]:
    if not device.startswith("cuda"):
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]

def build_dataloader(dataset: MCQATokenDataset, *, batch_size: int, num_workers: int, pad_id: int, device: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=device.startswith("cuda"),
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        drop_last=False,
    )

def train_one_run(
    *,
    args: argparse.Namespace,
    variant_name: str,
    run_dir: Path,
    stage_datasets: Sequence[StageDataset],
    val_examples: Sequence[Dict[str, Any]],
    data_sources: Mapping[str, Any],
) -> Dict[str, Any]:
    ensure_dir(run_dir)
    ensure_dir(run_dir / "data")
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    metrics_path = run_dir / "epoch_metrics.jsonl"

    save_json(run_dir / "run_config.json", vars(args))
    save_json(run_dir / "data" / "data_sources.json", data_sources)
    for stage in stage_datasets:
        save_jsonl(run_dir / "data" / f"stage_{stage.stage_id + 1}_train.jsonl", stage.examples)
        save_json(run_dir / "data" / f"stage_{stage.stage_id + 1}_summary.json", stage.summary)
    save_jsonl(run_dir / "data" / "val.jsonl", val_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    torch_dtype = resolve_torch_dtype(args.dtype, args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    if hasattr(model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(args.device)

    num_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    stage_token_datasets: List[MCQATokenDataset] = []
    stage_token_summaries: Dict[str, Any] = {}
    for stage in stage_datasets:
        token_ds = MCQATokenDataset(stage.examples, tokenizer=tokenizer, max_length=int(args.max_length))
        if len(token_ds) == 0:
            raise RuntimeError(f"Stage {stage.stage_id + 1} produced an empty tokenized dataset")
        stage_token_datasets.append(token_ds)
        stage_token_summaries[f"stage_{stage.stage_id + 1}"] = {
            "n_examples": int(len(token_ds)),
            "total_tokens_once": int(sum(token_ds.lengths)),
            "avg_tokens_per_sample": float(sum(token_ds.lengths) / max(1, len(token_ds))),
        }
    val_ds = MCQATokenDataset(val_examples, tokenizer=tokenizer, max_length=int(args.max_length))
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty after tokenization")

    val_dl = build_dataloader(
        val_ds,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pad_id=pad_id,
        device=args.device,
        shuffle=False,
    )

    epoch_stage_ids = build_epoch_stage_ids(int(args.epochs), num_stages=3)
    total_updates = 0
    for stage_id in epoch_stage_ids:
        stage_n = len(stage_token_datasets[stage_id])
        stage_batches = int(math.ceil(stage_n / max(1, int(args.batch_size))))
        total_updates += int(math.ceil(stage_batches / max(1, int(args.grad_accum))))
    total_updates = max(1, total_updates)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    warmup_steps = int(total_updates * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    use_fp16 = args.device.startswith("cuda") and args.dtype == "float16"
    use_bf16 = args.device.startswith("cuda") and args.dtype == "bfloat16"
    amp_enabled = use_fp16 or use_bf16
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_ckpt = ckpt_dir / "best.pt"
    final_ckpt = ckpt_dir / "final.pt"
    best_val_acc = float("-inf")
    global_step = 0
    seen_batches = 0
    epoch_metrics: List[Dict[str, Any]] = []
    total_train_seconds = 0.0

    wb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --use-wandb was set")
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            group=(args.wandb_group or None),
            name=args.wandb_name or variant_name,
            config={
                "dataset": args.dataset,
                "variant_name": variant_name,
                "baseline_mode": args.baseline_mode,
                "curriculum_mode": args.curriculum_mode,
                "model_name_or_path": args.model_name_or_path,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "grad_accum": int(args.grad_accum),
                "warmup_ratio": float(args.warmup_ratio),
                "max_length": int(args.max_length),
                "dtype": args.dtype,
                "run_dir": str(run_dir),
            },
        )

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        optimizer.zero_grad(set_to_none=True)
        for epoch_idx in range(int(args.epochs)):
            stage_id = epoch_stage_ids[epoch_idx]
            stage = stage_datasets[stage_id]
            train_ds = stage_token_datasets[stage_id]
            train_dl = build_dataloader(
                train_ds,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                pad_id=pad_id,
                device=args.device,
                shuffle=True,
            )

            epoch_t0 = time.time()
            model.train()
            running_loss = 0.0
            running_items = 0
            grad_norm_sum = 0.0
            grad_norm_count = 0

            for step, batch in enumerate(train_dl, start=1):
                seen_batches += 1
                input_ids = batch["input_ids"].to(args.device)
                attn = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                    raw_loss = outputs.loss
                    loss = raw_loss / max(1, int(args.grad_accum))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += float(raw_loss.detach().item()) * int(input_ids.shape[0])
                running_items += int(input_ids.shape[0])

                do_step = (step % max(1, int(args.grad_accum)) == 0) or (step == len(train_dl))
                if do_step:
                    grad_norm = None
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if float(args.max_grad_norm) > 0:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm)).item()
                        )
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if grad_norm is not None:
                        grad_norm_sum += grad_norm
                        grad_norm_count += 1

                    if wb_run is not None and (global_step % max(1, int(args.wandb_log_every)) == 0):
                        wb_run.log(
                            {
                                "train/loss_step": float(raw_loss.detach().item()),
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "train/global_step": float(global_step),
                                "train/epoch": float(epoch_idx + 1),
                                "train/stage_id": float(stage_id + 1),
                            },
                            step=global_step,
                        )

            total_train_seconds += float(time.time() - epoch_t0)
            val_metrics = eval_mcqa_metrics(model, val_dl, device=args.device)
            train_loss_epoch = float(running_loss / max(1, running_items))
            mean_grad_norm = float(grad_norm_sum / max(1, grad_norm_count))

            row = {
                "epoch": int(epoch_idx + 1),
                "stage_id": int(stage_id + 1),
                "stage_name": stage.stage_name,
                "train_loss": float(train_loss_epoch),
                "val_acc": float(val_metrics["accuracy"]),
                "val_option_entropy": float(val_metrics["mean_option_entropy"]),
                "val_margin": float(val_metrics["mean_margin"]),
                "val_pred_label_entropy": float(val_metrics["pred_label_entropy"]),
                "global_step": int(global_step),
                "epoch_seconds": float(time.time() - epoch_t0),
                "mean_grad_norm": float(mean_grad_norm),
                "stage_ratios": dict(stage.ratios),
                "stage_summary": stage.summary,
            }
            epoch_metrics.append(row)
            metrics_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_handle.flush()

            ckpt_meta = {
                "variant_name": variant_name,
                "epoch": int(epoch_idx + 1),
                "stage_id": int(stage_id + 1),
                "val_acc": float(val_metrics["accuracy"]),
                "train_loss": float(train_loss_epoch),
            }
            if float(val_metrics["accuracy"]) > best_val_acc:
                best_val_acc = float(val_metrics["accuracy"])
                save_checkpoint(best_ckpt, model, epoch_idx + 1, global_step, ckpt_meta)

            stage_log = flatten_numeric("stage_summary", stage.summary)
            ratio_log = {f"stage_ratio/{k}": float(v) for k, v in stage.ratios.items()}
            if wb_run is not None:
                wb_run.log(
                    {
                        "epoch/train_loss": float(train_loss_epoch),
                        "epoch/val_acc": float(val_metrics["accuracy"]),
                        "epoch/val_option_entropy": float(val_metrics["mean_option_entropy"]),
                        "epoch/val_margin": float(val_metrics["mean_margin"]),
                        "epoch/val_pred_label_entropy": float(val_metrics["pred_label_entropy"]),
                        "epoch/seconds": float(row["epoch_seconds"]),
                        "epoch/num": float(epoch_idx + 1),
                        "epoch/stage_id": float(stage_id + 1),
                        **stage_log,
                        **ratio_log,
                    },
                    step=global_step,
                )

    save_checkpoint(final_ckpt, model, int(args.epochs), global_step, {"variant_name": variant_name, "final": True},)

    final_val_acc = float(epoch_metrics[-1]["val_acc"]) if epoch_metrics else float("nan")
    total_tokens_seen = 0
    for epoch_idx, stage_id in enumerate(epoch_stage_ids):
        del epoch_idx
        total_tokens_seen += int(sum(stage_token_datasets[stage_id].lengths))

    summary = {
        "dataset": args.dataset,
        "variant_name": variant_name,
        "baseline_mode": args.baseline_mode,
        "curriculum_mode": args.curriculum_mode,
        "raw_score_key": args.raw_score_key or None,
        "model_name_or_path": args.model_name_or_path,
        "seed": int(args.seed),
        "run_dir": str(run_dir),
        "checkpoints": {
            "best": str(best_ckpt),
            "final": str(final_ckpt),
        },
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_accum": int(args.grad_accum),
            "warmup_ratio": float(args.warmup_ratio),
            "max_grad_norm": float(args.max_grad_norm),
            "max_length": int(args.max_length),
            "num_workers": int(args.num_workers),
            "dtype": args.dtype,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
        "model": {
            "num_params": int(num_params),
            "trainable_params": int(trainable_params),
            "trainable_ratio": float(trainable_params / max(1, num_params)),
        },
        "stage_schedule": {
            "epoch_stage_ids": [int(x + 1) for x in epoch_stage_ids],
            "stage_names": list(STAGE_NAMES),
            "stage_token_summaries": stage_token_summaries,
        },
        "data_sources": dict(data_sources),
        "sizes": {
            "n_val": int(len(val_ds)),
            "n_train_requested_per_stage": int(args.train_examples),
            "n_train_tokenized_per_stage": {
                f"stage_{stage.stage_id + 1}": int(len(stage_token_datasets[stage.stage_id]))
                for stage in stage_datasets
            },
        },
        "efficiency": {
            "global_steps": int(global_step),
            "seen_batches": int(seen_batches),
            "total_train_seconds": float(total_train_seconds),
            "steps_per_second": float(global_step / max(1e-6, total_train_seconds)),
            "tokens_seen_total": int(total_tokens_seen),
            "tokens_per_second": float(total_tokens_seen / max(1e-6, total_train_seconds)),
        },
        "metrics": {
            "best_val_acc": float(best_val_acc if best_val_acc > -1e9 else float("nan")),
            "final_val_acc": float(final_val_acc),
            "epochs": epoch_metrics,
        },
    }
    save_json(run_dir / "run_summary.json", summary)

    if wb_run is not None:
        wb_run.log(
            {
                "run/best_val_acc": float(summary["metrics"]["best_val_acc"]),
                "run/final_val_acc": float(summary["metrics"]["final_val_acc"]),
                "run/train_seconds": float(total_train_seconds),
                "run/num_params": int(num_params),
                "run/trainable_params": int(trainable_params),
            },
            step=global_step,
        )
        wb_run.finish()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curriculum-learning finetuning runner for circuit-generated MCQA datasets")
    p.add_argument("--dataset", type=str, default="sciq", choices=["piqa", "winogrande", "sciq", "arc_easy"])
    p.add_argument("--baseline-mode", "--baseline_mode", dest="baseline_mode", type=str, default="ours", choices=BASELINE_MODE_CHOICES)
    p.add_argument("--curriculum-mode", "--curriculum_mode", dest="curriculum_mode", type=str, default="on", choices=CURRICULUM_MODE_CHOICES)
    p.add_argument("--model-name-or-path", "--model_name_or_path", dest="model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-root", "--output_root", dest="output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="")
    p.add_argument(
        "--variant-suffix", "--variant_suffix", dest="variant_suffix", type=str, default="",
        help=(
            "Optional suffix appended to the auto-generated variant_name, e.g. 'easy_only'. "
            "Useful when multiple runs share the same baseline_mode/curriculum_mode but differ "
            "in flat-ratio or recipe knobs (steering-only ablations). "
            "Result: 'ours_random_easy_only/' instead of 'ours_random/'."
        ),
    )

    p.add_argument("--train-examples", "--train_examples", dest="train_examples", type=int, default=10000)
    p.add_argument("--max-val-examples", "--max_val_examples", dest="max_val_examples", type=int, default=0)
    p.add_argument("--train-frac", "--train_frac", dest="train_frac", type=float, default=0.8)

    p.add_argument("--generation-root", "--generation_root", dest="generation_root", type=str, default=str(DEFAULT_GENERATION_ROOT))
    p.add_argument("--generator-model-key", "--generator_model_key", dest="generator_model_key", type=str, default="Qwen__Qwen2.5-1.5B-Instruct")
    p.add_argument("--generated-prompt-mode", "--generated_prompt_mode", dest="generated_prompt_mode", type=str, default="fixed")
    p.add_argument("--generated-gen-mode", "--generated_gen_mode", dest="generated_gen_mode", type=str, default="both")
    p.add_argument(
        "--generated-path-contains", "--generated_path_contains", dest="generated_path_contains", type=str, default="",
        help="Optional substring filter applied to selected/raw generated file paths, useful for ablation subdirectories like 'no_aum'.",
    )
    p.add_argument("--prompt-only-file-kind", "--prompt_only_file_kind", dest="prompt_only_file_kind", type=str, default="raw", choices=GENERATED_FILE_KIND_CHOICES)
    p.add_argument("--random-circuit-file-kind", "--random_circuit_file_kind", dest="random_circuit_file_kind", type=str, default="raw", choices=GENERATED_FILE_KIND_CHOICES)
    p.add_argument("--ours-file-kind", "--ours_file_kind", dest="ours_file_kind", type=str, default="selected", choices=GENERATED_FILE_KIND_CHOICES)
    p.add_argument("--random-circuit-recipes", "--random_circuit_recipes", dest="random_circuit_recipes", type=str, default="easy,medium,hard,helpful")
    p.add_argument("--random-circuit-ratios", "--random_circuit_ratios", dest="random_circuit_ratios", type=str, default="easy=0.25,medium=0.25,hard=0.25,helpful=0.25")
    p.add_argument("--flat-generated-ratios", "--flat_generated_ratios", dest="flat_generated_ratios", type=str, default="easy=0.25,medium=0.25,hard=0.25,helpful=0.25")

    p.add_argument("--stage1-ratios", "--stage1_ratios", dest="stage1_ratios", type=str, default="easy=0.7,helpful=0.3")
    p.add_argument("--stage2-ratios", "--stage2_ratios", dest="stage2_ratios", type=str, default="medium=0.5,easy=0.2,helpful=0.3")
    p.add_argument("--stage3-ratios", "--stage3_ratios", dest="stage3_ratios", type=str, default="hard=0.5,medium=0.2,helpful=0.3")

    p.add_argument("--scores-json", "--scores_json", dest="scores_json", type=str, default="")
    p.add_argument("--raw-score-keys", "--raw_score_keys", dest="raw_score_keys", type=str, default="aum,el2n,gradalign")
    p.add_argument("--raw-score-easy-high", "--raw_score_easy_high", dest="raw_score_easy_high", type=str, default="aum,gradalign")
    p.add_argument("--raw-band-easy-frac", "--raw_band_easy_frac", dest="raw_band_easy_frac", type=float, default=0.33)
    p.add_argument("--raw-band-hard-frac", "--raw_band_hard_frac", dest="raw_band_hard_frac", type=float, default=0.33)
    p.add_argument("--raw-stage1-ratios", "--raw_stage1_ratios", dest="raw_stage1_ratios", type=str, default="easy=1.0")
    p.add_argument("--raw-stage2-ratios", "--raw_stage2_ratios", dest="raw_stage2_ratios", type=str, default="medium=0.7142857,easy=0.2857143")
    p.add_argument("--raw-stage3-ratios", "--raw_stage3_ratios", dest="raw_stage3_ratios", type=str, default="hard=0.7142857,medium=0.2857143")

    p.add_argument("--allow-pool-oversample", "--allow_pool_oversample", dest="allow_pool_oversample", action="store_true")
    p.add_argument(
        "--off-resample", "--off_resample", dest="off_resample", action="store_true",
        help=(
            "curriculum=off: independently re-draw each of the 3 stage datasets from the pool "
            "with a different seed instead of replicating the same subset. "
            "Makes the unique-data support of off comparable to on (B-style diversity)."
        ),
    )
    p.add_argument(
        "--anchor-replay", "--anchor_replay", dest="anchor_replay", action="store_true",
        help=(
            "ours+curriculum=on: pre-build per-band anchor pools sized by --flat-generated-ratios "
            "before constructing stage datasets. All stages then sample from these fixed anchors "
            "(with replacement within each anchor band), so later-stage easy/helpful examples are "
            "strict subsets of what stage-1 already saw. Enables apples-to-apples ordering comparison."
        ),
    )
    p.add_argument(
        "--global-random-circuit", "--global_random_circuit", dest="global_random_circuit", action="store_true",
        help=(
            "random_circuit: ignore --random-circuit-ratios and instead pool all recipe bands "
            "together, sampling uniformly (truly global random baseline). "
            "Equivalent to setting a single recipe=all with ratio=1.0."
        ),
    )
    p.add_argument(
        "--source-ratio", "--source_ratio", dest="source_ratio", type=float, default=0.0,
        help=(
            "Fraction [0.0, 1.0] of train_examples per stage to sample from the real source "
            "dataset. The remainder comes from the generated pool. Applies only to ours / "
            "prompt_only / random_circuit modes; original and raw_score already use source data. "
            "E.g. 0.2 → 80%% generated + 20%% real source rows per stage."
        ),
    )
    p.add_argument(
        "--shuffle-choices", "--shuffle_choices", dest="shuffle_choices", action="store_true", default=True,
        help=(
            "Randomly permute answer choices in generated data pools so the correct answer is "
            "uniformly distributed across positions A/B/C/D. Fixes the ~38%% A-position bias "
            "that arises because LLMs tend to place the correct answer first. "
            "Applied at pool-load time in build_generated_recipe_pools. Default: on."
        ),
    )
    p.add_argument("--no-shuffle-choices", dest="shuffle_choices", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    p.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0)
    p.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=1)
    p.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", "--max_grad_norm", dest="max_grad_norm", type=float, default=1.0)
    p.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=512)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    p.add_argument("--gradient-checkpointing", "--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")

    p.add_argument("--use-wandb", "--use_wandb", dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", type=str, default="data_circuits_curriculum")
    p.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity", type=str, default="")
    p.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", type=str, default="")
    p.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", type=str, default="")
    p.add_argument("--wandb-log-every", "--wandb_log_every", dest="wandb_log_every", type=int, default=20)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.dataset = canonical_dataset_name(args.dataset)
    set_seed(int(args.seed))

    splits = prepare_task_splits(dataset_name=args.dataset, seed=int(args.seed), train_frac=float(args.train_frac))
    train_examples = [original_example_with_metadata(ex, source_name="original") for ex in splits["train"]]
    val_examples = [original_example_with_metadata(ex, source_name="val") for ex in splits["val"]]
    val_examples = maybe_limit_rows(val_examples, int(args.max_val_examples), seed=int(args.seed))

    run_tag = args.run_name or now_str()
    model_key = sanitize_name(args.model_name_or_path)
    output_root = Path(args.output_root).expanduser()

    run_specs: List[Tuple[str, Optional[str]]] = []
    if args.baseline_mode == "raw_score":
        raw_score_keys = parse_csv_set(args.raw_score_keys)
        raw_variant_suffix = f"_{args.variant_suffix}" if getattr(args, "variant_suffix", "") else ""
        if args.curriculum_mode == "on":
            run_specs = [(f"raw_score_curriculum_{score_key}{raw_variant_suffix}", score_key) for score_key in raw_score_keys]
        else:
            score_key = raw_score_keys[0] if raw_score_keys else "aum"
            run_specs = [(f"raw_score_random_{score_key}{raw_variant_suffix}", score_key)]
    else:
        variant_suffix = "curriculum" if args.curriculum_mode == "on" else "random"
        if getattr(args, "variant_suffix", ""):
            variant_suffix = f"{variant_suffix}_{args.variant_suffix}"
        run_specs = [(f"{args.baseline_mode}_{variant_suffix}", None)]

    summaries: List[Dict[str, Any]] = []
    for variant_name, raw_score_key in run_specs:
        args.raw_score_key = raw_score_key or ""
        stage_datasets, data_sources = build_stage_datasets_for_variant(
            args=args,
            train_examples=train_examples,
            baseline_mode=args.baseline_mode,
            curriculum_mode=args.curriculum_mode,
            raw_score_key=raw_score_key,
        )
        run_dir = ensure_dir(output_root / args.dataset / model_key / variant_name / run_tag)
        summary = train_one_run(
            args=args,
            variant_name=variant_name,
            run_dir=run_dir,
            stage_datasets=stage_datasets,
            val_examples=val_examples,
            data_sources=data_sources,
        )
        summaries.append(summary)
        print(f"[DONE] variant={variant_name} run_dir={run_dir}")

    if len(summaries) > 1:
        aggregate = {
            "dataset": args.dataset,
            "model_name_or_path": args.model_name_or_path,
            "baseline_mode": args.baseline_mode,
            "curriculum_mode": args.curriculum_mode,
            "run_tag": run_tag,
            "variants": [
                {
                    "variant_name": summary["variant_name"],
                    "run_dir": summary["run_dir"],
                    "best_val_acc": summary["metrics"]["best_val_acc"],
                    "final_val_acc": summary["metrics"]["final_val_acc"],
                }
                for summary in summaries
            ],
        }
        aggregate_path = ensure_dir(output_root / args.dataset / model_key / f"{args.baseline_mode}_{args.curriculum_mode}") / f"{run_tag}_aggregate.json"
        save_json(aggregate_path, aggregate)
        print(f"[DONE] aggregate_summary={aggregate_path}")


if __name__ == "__main__":
    main()
