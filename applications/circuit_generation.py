from __future__ import annotations
"""
Circuit-aware difficulty/helpful data generation framework.
What this script does
---------------------
1. Loads a Hugging Face causal LM (designed for Qwen2/Qwen2.5 style decoder models,
   but most activation-addition pieces are architecture-agnostic).
2. Builds circuit presets from either:
   - built-in defaults derived from your structural analysis, or
   - a real top-edge CSV/JSONL file (top-k edges are re-summarized into practical targets).
3. Supports two inference-time steering modes:
   - Activation Addition (RepE-style residual steering)
   - Targeted attention-logit steering (Qwen2 eager attention only)
4. Generates candidate MCQA samples for easy / medium / hard / helpful bands.
5. Re-scores candidates contrastively with circuit recipes and exports a filtered dataset.

Important design choice
-----------------------
The implementation deliberately keeps interventions small and sparse:
- few critical layers / heads
- optional head-temperature + weak positional bias instead of arbitrary large mask edits
- activation steering only on selected blocks
This follows the spirit of circuit-aware, low-footprint intervention rather than broad OOD shifts.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_VERBOSE: bool = True  # set False via --no-verbose to suppress per-record/per-candidate noise

def print_rule(title: str, char: str = "=") -> None:
    line = char * 88
    print(f"\n{line}\n{title}\n{line}")

def print_kv(key: str, value: Any) -> None:
    print(f"[info] {key}: {value}")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
    return rows

def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

def append_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())

def mean_safe(xs: Sequence[float], default: float = 0.0) -> float:
    return statistics.mean(xs) if xs else default

def median_safe(xs: Sequence[float], default: float = 0.0) -> float:
    return statistics.median(xs) if xs else default

def normalize_text_for_dedup(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())

def normalize_json_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): normalize_json_like(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_json_like(v) for v in value]
    if isinstance(value, str):
        return normalize_text_for_dedup(value)
    return value

def build_candidate_exact_signature(row: Mapping[str, Any]) -> Optional[str]:
    parsed = row.get("parsed") or {}
    question = normalize_text_for_dedup(parsed.get("question") or row.get("question"))
    answer = normalize_text_for_dedup(parsed.get("answer") or row.get("answer"))
    choices = parsed.get("choices") if parsed else row.get("choices")
    if not question or not answer or not choices:
        return None
    payload = {
        "question": question,
        "choices": normalize_json_like(choices),
        "answer": answer,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest

def dedup_candidates_exact(candidates: Sequence[Dict[str, Any]], selected_signatures: set[str],) -> Tuple[List[Dict[str, Any]], int]:
    local_seen = set(selected_signatures)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in candidates:
        signature = build_candidate_exact_signature(row)
        if not signature:
            kept.append(row)
            continue
        if signature in local_seen:
            dropped += 1
            continue
        local_seen.add(signature)
        kept.append(row)
    return kept, dropped

def summarize_selected_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def avg(vals: Iterable[Any]) -> Optional[float]:
        values = [float(v) for v in vals if v is not None]
        return sum(values) / len(values) if values else None
    return {
        "n": len(rows),
        "json_parse_rate": avg(1.0 if r.get("sanity", {}).get("json_parse_ok") else 0.0 for r in rows),
        "mean_final_score": avg(r.get("final_score") for r in rows),
        "mean_difficulty_proxy": avg(r.get("internal_scores", {}).get("difficulty_proxy") for r in rows),
        "mean_helpfulness_proxy": avg(r.get("internal_scores", {}).get("helpfulness_proxy") for r in rows),
        "mean_empirical_difficulty": avg(r.get("empirical_eval", {}).get("difficulty_empirical") for r in rows),
        "mean_solver_correct_prob_no_rationale": avg(r.get("empirical_eval", {}).get("solver_correct_prob_no_rationale") for r in rows),
        "mean_rationale_gain": avg(r.get("empirical_eval", {}).get("rationale_gain") for r in rows),
    }

def prepare_resume_state(raw_candidates_path: Path, selected_path: Path, progress_path: Path,) -> Dict[str, Any]:
    existing_raw_rows = load_jsonl(raw_candidates_path) if raw_candidates_path.exists() else []
    existing_selected_rows = load_jsonl(selected_path) if selected_path.exists() else []
    progress_rows = load_jsonl(progress_path) if progress_path.exists() else []
    if progress_rows:
        completed_record_ids = {
            str(row.get("record_id"))
            for row in progress_rows
            if row.get("record_id") is not None
        }
    else:
        completed_record_ids = {
            str(row.get("record_id"))
            for row in existing_selected_rows
            if row.get("record_id") is not None
        }
        if completed_record_ids:
            synthesized_progress = [{"record_id": record_id, "status": "completed", "source": "selected_fallback"} for record_id in sorted(completed_record_ids)]
            save_jsonl(progress_path, synthesized_progress)
            progress_rows = synthesized_progress

    filtered_raw_rows = [row for row in existing_raw_rows if str(row.get("record_id")) in completed_record_ids]
    filtered_selected_rows = [row for row in existing_selected_rows if str(row.get("record_id")) in completed_record_ids]
    filtered_progress_rows = [
        row for row in progress_rows
        if str(row.get("record_id")) in completed_record_ids
    ]

    dangling_raw_rows = len(existing_raw_rows) - len(filtered_raw_rows)
    dangling_selected_rows = len(existing_selected_rows) - len(filtered_selected_rows)
    dangling_progress_rows = len(progress_rows) - len(filtered_progress_rows)

    if raw_candidates_path.exists() and dangling_raw_rows:
        save_jsonl(raw_candidates_path, filtered_raw_rows)
    if selected_path.exists() and dangling_selected_rows:
        save_jsonl(selected_path, filtered_selected_rows)
    if progress_path.exists() and dangling_progress_rows:
        save_jsonl(progress_path, filtered_progress_rows)

    selected_signatures = {
        signature
        for row in filtered_selected_rows
        if (signature := build_candidate_exact_signature(row)) is not None
    }

    return {
        "completed_record_ids": completed_record_ids,
        "selected_rows": filtered_selected_rows,
        "selected_signatures": selected_signatures,
        "dangling_raw_rows": dangling_raw_rows,
        "dangling_selected_rows": dangling_selected_rows,
        "dangling_progress_rows": dangling_progress_rows,
    }

# -----------------------------------------------------------------------------
# Circuit schema
# -----------------------------------------------------------------------------
CANONICAL_CIRCUIT_KEYS = (
    "aum_high",
    "aum_low",
    "el2n_high",
    "el2n_low",
    "gradalign_high",
    "gradalign_low",
)

@dataclass
class NodeTarget:
    layer: int
    component: str  # 'mlp' | 'attn'
    heads: Optional[List[int]] = None
    path: Optional[str] = None  # k/q/v if known from edge parsing
    role: str = ""
    weight: float = 1.0
    preferred_token_policy: str = "last"
    attention_mode: str = "none"  # none | prefix | recent | prefix+recent
    recent_window: int = 16
    prefix_fraction: float = 0.30

    def label(self) -> str:
        if self.component == "mlp":
            return f"m{self.layer}"
        if self.heads:
            heads_str = ",".join(str(h) for h in self.heads)
            suffix = f"<{self.path}>" if self.path else ""
            return f"a{self.layer}.h[{heads_str}]{suffix}"
        suffix = f"<{self.path}>" if self.path else ""
        return f"a{self.layer}<all>{suffix}"

@dataclass
class CircuitPreset:
    key: str
    metric: str
    polarity: str
    steering_role: str  # easy_anchor | hard_rerouting | control | short_path | boundary
    description: str
    targets: List[NodeTarget] = field(default_factory=list)
    default_activation_scale: float = 0.0
    default_attention_scale: float = 0.0
    allow_positive_steering: bool = True
    notes: str = ""


@dataclass
class RecipeConfig:
    name: str
    activation_scales: Dict[str, float]
    attention_scales: Dict[str, float]
    score_weights: Dict[str, float]
    description: str


@dataclass
class EdgeRow:
    metric: str
    polarity: str
    source: str
    target: str
    weight: float

def clone_node_target(target: NodeTarget) -> NodeTarget:
    return NodeTarget(
        layer=int(target.layer),
        component=str(target.component),
        heads=list(target.heads) if target.heads is not None else None,
        path=target.path,
        role=str(target.role),
        weight=float(target.weight),
        preferred_token_policy=str(target.preferred_token_policy),
        attention_mode=str(target.attention_mode),
        recent_window=int(target.recent_window),
        prefix_fraction=float(target.prefix_fraction),
    )

def clone_circuit_preset(preset: CircuitPreset) -> CircuitPreset:
    return CircuitPreset(
        key=str(preset.key),
        metric=str(preset.metric),
        polarity=str(preset.polarity),
        steering_role=str(preset.steering_role),
        description=str(preset.description),
        targets=[clone_node_target(target) for target in preset.targets],
        default_activation_scale=float(preset.default_activation_scale),
        default_attention_scale=float(preset.default_attention_scale),
        allow_positive_steering=bool(preset.allow_positive_steering),
        notes=str(preset.notes),
    )


NODE_ATTN_RE = re.compile(r"a(?P<layer>\d+)\.h(?P<head>\d+)(?:<(?P<path>[kvq])>)?")
NODE_ATTN_LAYER_RE = re.compile(r"a(?P<layer>\d+)(?:<(?P<path>[kvq])>)?")
NODE_MLP_RE = re.compile(r"m(?P<layer>\d+)")


# -----------------------------------------------------------------------------
# Default presets from the user's structural summary
# -----------------------------------------------------------------------------
def build_default_presets() -> Dict[str, CircuitPreset]:
    presets: Dict[str, CircuitPreset] = {
        "aum_high": CircuitPreset(
            key="aum_high",
            metric="AUM",
            polarity="high",
            steering_role="easy_anchor",
            description="easy/supportive canonical scoring path",
            targets=[
                NodeTarget(0, "attn", heads=[6, 7, 8, 9, 10, 11], path="v", role="input_interface", attention_mode="prefix+recent", recent_window=12, prefix_fraction=0.35),
                NodeTarget(1, "mlp", role="early_hub", weight=1.2),
                NodeTarget(2, "mlp", role="early_hub", weight=1.1),
                NodeTarget(6, "attn", heads=[6, 7, 8, 9, 10, 11], path="k", role="easy_hub", attention_mode="prefix+recent", recent_window=20, prefix_fraction=0.30),
                NodeTarget(14, "attn", heads=[2], path="k", role="convergence", attention_mode="none", recent_window=32),
                NodeTarget(23, "mlp", role="readout", weight=1.15),
            ],
            default_activation_scale=0.35,
            default_attention_scale=0.25,
            allow_positive_steering=True,
            notes="Primary easy-anchor path. Safe positive steering target.",
        ),
        "aum_low": CircuitPreset(
            key="aum_low",
            metric="AUM",
            polarity="low",
            steering_role="boundary",
            description="boundary / ambiguity / failure-adjacent branch",
            targets=[
                NodeTarget(2, "attn", heads=None, path="k", role="early_boundary", attention_mode="none"),
                NodeTarget(3, "mlp", role="reroute_buffer"),
                NodeTarget(5, "attn", heads=None, path="k", role="boundary_branch", attention_mode="none"),
                NodeTarget(27, "mlp", role="late_boundary_readout"),
            ],
            default_activation_scale=0.00,
            default_attention_scale=0.00,
            allow_positive_steering=False,
            notes="Use mainly as suppressive steering / acceptance score. Direct positive steering is discouraged.",
        ),
        "el2n_low": CircuitPreset(
            key="el2n_low",
            metric="EL2N",
            polarity="low",
            steering_role="easy_anchor",
            description="stable canonical easy route",
            targets=[
                NodeTarget(0, "attn", heads=[6, 7, 8, 9, 10, 11], path="v", role="input_interface", attention_mode="prefix+recent", recent_window=12, prefix_fraction=0.35),
                NodeTarget(1, "mlp", role="early_hub", weight=1.2),
                NodeTarget(2, "mlp", role="early_hub", weight=1.1),
                NodeTarget(3, "mlp", role="stabilizer", weight=0.8),
                NodeTarget(6, "attn", heads=[6, 7, 8, 9, 10, 11], path="k", role="easy_hub", attention_mode="prefix+recent", recent_window=20, prefix_fraction=0.30),
                NodeTarget(7, "attn", heads=None, path="k", role="easy_followthrough", attention_mode="prefix"),
                NodeTarget(23, "mlp", role="readout", weight=1.15),
            ],
            default_activation_scale=0.40,
            default_attention_scale=0.28,
            allow_positive_steering=True,
            notes="Strong stable easy anchor. Good for easy generation and anchor replay.",
        ),
        "el2n_high": CircuitPreset(
            key="el2n_high",
            metric="EL2N",
            polarity="high",
            steering_role="hard_rerouting",
            description="hard-but-useful rerouting circuit",
            targets=[
                NodeTarget(2, "attn", heads=None, path="k", role="rerouting_entry", attention_mode="none"),
                NodeTarget(4, "attn", heads=None, path="k", role="rerouting_mid", attention_mode="none"),
                NodeTarget(5, "attn", heads=None, path="k", role="rerouting_mid", attention_mode="none"),
                NodeTarget(4, "mlp", role="deeper_reroute", weight=1.0),
                NodeTarget(6, "mlp", role="deeper_reroute", weight=1.0),
                NodeTarget(13, "mlp", role="late_control_bridge", weight=1.1),
            ],
            default_activation_scale=0.32,
            default_attention_scale=0.18,
            allow_positive_steering=True,
            notes="Best positive steering target for hard sample generation.",
        ),
        "gradalign_high": CircuitPreset(
            key="gradalign_high",
            metric="GradAlign",
            polarity="high",
            steering_role="control",
            description="validation-aligned control / mid-late routing",
            targets=[
                NodeTarget(13, "mlp", role="control_entry", weight=1.15),
                NodeTarget(14, "attn", heads=[2], path="k", role="control_hub", attention_mode="none"),
                NodeTarget(17, "attn", heads=None, role="late_control", attention_mode="none"),
                NodeTarget(18, "attn", heads=None, role="late_control", attention_mode="none"),
                NodeTarget(19, "attn", heads=None, role="late_control", attention_mode="none"),
                NodeTarget(20, "attn", heads=None, role="late_control", attention_mode="none"),
            ],
            default_activation_scale=0.28,
            default_attention_scale=0.15,
            allow_positive_steering=True,
            notes="Use for helpful/control augmentation, not as the sole hardest-data driver.",
        ),
        "gradalign_low": CircuitPreset(
            key="gradalign_low",
            metric="GradAlign",
            polarity="low",
            steering_role="short_path",
            description="direct readout / evidence extraction / short path",
            targets=[
                NodeTarget(0, "attn", heads=[6, 7, 8, 9, 10, 11], path="v", role="input_interface", attention_mode="prefix+recent", recent_window=12, prefix_fraction=0.35),
                NodeTarget(1, "mlp", role="early_shortcut", weight=1.0),
                NodeTarget(2, "mlp", role="early_shortcut", weight=1.0),
                NodeTarget(3, "mlp", role="early_shortcut", weight=0.9),
                NodeTarget(23, "mlp", role="direct_readout", weight=1.1),
            ],
            default_activation_scale=0.12,
            default_attention_scale=0.15,
            allow_positive_steering=True,
            notes="Use only weakly. Helpful for crisp evidence extraction, but suppress for deep reasoning.",
        ),
    }
    return presets

DEFAULT_RECIPES: Dict[str, RecipeConfig] = {
    "easy": RecipeConfig(
        name="easy",
        activation_scales={
            "aum_high": 0.38,
            "aum_low": -0.23,
            "el2n_high": -0.33,
            "el2n_low": 0.45,
            "gradalign_high": 0.08,
            "gradalign_low": 0.12,
        },
        attention_scales={
            "aum_high": 0.28,
            "aum_low": 0.00,
            "el2n_high": -0.14,
            "el2n_low": 0.30,
            "gradalign_high": 0.01,
            "gradalign_low": 0.12,
        },
        score_weights={
            "easy_support": 1.2,
            "hard_support": -0.8,
            "helpful_support": 0.3,
            "shortcut_support": 0.1,
            "base_logprob": 0.2,
        },
        description="Amplify canonical easy anchors; suppress rerouting/boundary branches.",
    ),
    "medium": RecipeConfig(
        name="medium",
        activation_scales={
            "aum_high": 0.18,
            "aum_low": -0.10,
            "el2n_high": 0.16,
            "el2n_low": 0.16,
            "gradalign_high": 0.20,
            "gradalign_low": -0.05,
        },
        attention_scales={
            "aum_high": 0.10,
            "aum_low": 0.00,
            "el2n_high": 0.08,
            "el2n_low": 0.10,
            "gradalign_high": 0.10,
            "gradalign_low": -0.05,
        },
        score_weights={
            "easy_support": 0.2,
            "hard_support": 0.2,
            "helpful_support": 0.4,
            "shortcut_support": -0.1,
            "base_logprob": 0.3,
        },
        description="Keep easy anchor alive but introduce moderate rerouting and control.",
    ),
    "hard": RecipeConfig(
        name="hard",
        activation_scales={
            "aum_high": -0.32,
            "aum_low": 0.00,
            "el2n_high": 0.40,
            "el2n_low": -0.35,
            "gradalign_high": 0.28,
            "gradalign_low": -0.25,
        },
        attention_scales={
            "aum_high": -0.15,
            "aum_low": 0.00,
            "el2n_high": 0.15,
            "el2n_low": -0.18,
            "gradalign_high": 0.08,
            "gradalign_low": -0.12,
        },
        score_weights={
            "easy_support": -0.9,
            "hard_support": 1.2,
            "helpful_support": 0.5,
            "shortcut_support": -0.6,
            "base_logprob": -0.1,
        },
        description="Suppress short canonical path; amplify rerouting + control. AUM-low remains score-only by default.",
    ),
    "helpful": RecipeConfig(
        name="helpful",
        activation_scales={
            "aum_high": 0.18,
            "aum_low": -0.12,
            "el2n_high": -0.08,
            "el2n_low": 0.16,
            "gradalign_high": 0.40,
            "gradalign_low": -0.24,
        },
        attention_scales={
            "aum_high": 0.08,
            "aum_low": 0.00,
            "el2n_high": -0.05,
            "el2n_low": 0.08,
            "gradalign_high": 0.12,
            "gradalign_low": -0.10,
        },
        score_weights={
            "easy_support": 0.3,
            "hard_support": 0.2,
            "helpful_support": 1.3,
            "shortcut_support": -0.8,
            "base_logprob": 0.2,
        },
        description="Use GradAlign-high as the main helpful/control driver, while suppressing GradAlign-low shortcuts.",
    ),
}

def build_neutral_recipe(name: str = "neutral") -> RecipeConfig:
    return RecipeConfig(
        name=name,
        activation_scales={k: 0.0 for k in CANONICAL_CIRCUIT_KEYS},
        attention_scales={k: 0.0 for k in CANONICAL_CIRCUIT_KEYS},
        score_weights={"base_logprob": 1.0},
        description="No circuit steering; plain prompt-driven generation/scoring.",
    )

NEUTRAL_RECIPE = build_neutral_recipe()
RECIPE_CHOICES = tuple(list(DEFAULT_RECIPES.keys()) + ["naive"])
PROMPT_MODE_CHOICES = ("source", "fixed")
GEN_MODE_CHOICES = ("steering", "selection", "both", "none")
DEDUP_CHOICES = ("off", "exact")
CIRCUIT_TARGET_MODE_CHOICES = ("discovered", "random")

# -----------------------------------------------------------------------------
# Edge parsing and target derivation
# -----------------------------------------------------------------------------
def normalize_metric_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"grad", "grad_align", "gradalign"}:
        return "gradalign"
    if key == "el2n":
        return "el2n"
    if key == "aum":
        return "aum"
    return key

def canonical_circuit_key(metric: str, polarity: str) -> str:
    metric_key = normalize_metric_name(metric)
    pol = polarity.strip().lower()
    if pol in {"h", "high", "top"}:
        pol = "high"
    elif pol in {"l", "low", "bottom"}:
        pol = "low"
    return f"{metric_key}_{pol}"

def parse_node_name(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text in {"input", "embed", "embedding", "embeddings"}:
        return {"kind": "input"}
    if text in {"logits", "unembed", "output"}:
        return {"kind": "output"}
    m = NODE_ATTN_RE.fullmatch(text)
    if m:
        return {
            "kind": "attn",
            "layer": int(m.group("layer")),
            "head": int(m.group("head")),
            "path": m.group("path"),
        }
    m = NODE_ATTN_LAYER_RE.fullmatch(text)
    if m:
        return {
            "kind": "attn_layer",
            "layer": int(m.group("layer")),
            "head": None,
            "path": m.group("path"),
        }
    m = NODE_MLP_RE.fullmatch(text)
    if m:
        return {"kind": "mlp", "layer": int(m.group("layer"))}
    return None

def load_edge_rows(path: str | Path) -> List[EdgeRow]:
    path = Path(path)
    rows: List[EdgeRow] = []
    if path.suffix.lower() == ".jsonl":
        records = load_jsonl(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    for rec in records:
        metric = rec.get("metric") or rec.get("score_key") or rec.get("score_name")
        polarity = rec.get("polarity") or rec.get("circuit") or rec.get("bucket") or rec.get("group")
        source = rec.get("source") or rec.get("src") or rec.get("from") or rec.get("u")
        target = rec.get("target") or rec.get("dst") or rec.get("to") or rec.get("v")
        weight_val = rec.get("weight") or rec.get("score") or rec.get("abs_score") or rec.get("edge_weight") or rec.get("score_value") or rec.get("attr") or rec.get("value")
        if metric is None or polarity is None or source is None or target is None or weight_val is None:
            # skip malformed rows silently; caller will see if nothing was loaded.
            continue
        try:
            weight = float(weight_val)
        except (TypeError, ValueError):
            continue
        rows.append(
            EdgeRow(
                metric=str(metric),
                polarity=str(polarity),
                source=str(source),
                target=str(target),
                weight=weight,
            )
        )
    return rows

# Filename convention: {metric}_{polarity}_top_edges.csv
_CIRCUIT_FILE_RE = re.compile(
    r"(?P<metric>aum|el2n|grad_align|gradalign)_(?P<polarity>high|low)_top_edges\.csv$",
    re.IGNORECASE,
)

def load_circuit_edge_dir(dirpath: str | Path, topk_per_circuit: int = 250,) -> List[EdgeRow]:
    """Load edge rows from a directory of per-circuit CSV files.
    Each CSV filename encodes its circuit identity:
        aum_high_top_edges.csv, el2n_low_top_edges.csv, grad_align_high_top_edges.csv, ...
    This function reads each file, infers metric/polarity from the filename,
    and maps the CSV columns (source, target, score, qkv, signed_role) into EdgeRow objects.
    """
    dirpath = Path(dirpath)
    all_rows: List[EdgeRow] = []
    found_files: List[str] = []

    for csv_path in sorted(dirpath.glob("*_top_edges.csv")):
        m = _CIRCUIT_FILE_RE.search(csv_path.name)
        if m is None:
            print(f"[warn] skipping unrecognized file: {csv_path.name}")
            continue
        raw_metric = m.group("metric").lower().replace("_", "")   # grad_align -> gradalign
        polarity = m.group("polarity").lower()
        circuit_key = canonical_circuit_key(raw_metric, polarity)
        found_files.append(f"{csv_path.name} -> {circuit_key}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            loaded = 0
            for rec in reader:
                if loaded >= topk_per_circuit:
                    break
                src = rec.get("source", "").strip()
                tgt = rec.get("target", "").strip()
                score_str = rec.get("score") or rec.get("abs_score") or rec.get("weight")
                if not src or not tgt or score_str is None:
                    continue
                try:
                    weight = float(score_str)
                except (TypeError, ValueError):
                    continue
                all_rows.append(EdgeRow(
                    metric=raw_metric,
                    polarity=polarity,
                    source=src,
                    target=tgt,
                    weight=weight,
                ))
                loaded += 1
    print(f"[edge-dir] loaded {len(all_rows)} edges from {len(found_files)} files in {dirpath}")
    for fname in found_files:
        print(f"  - {fname}")
    return all_rows

def derive_targets_from_edges(edge_rows: Sequence[EdgeRow], topk_edges: int, max_attn_groups_per_circuit: int = 3, max_mlp_per_circuit: int = 3,) -> Dict[str, List[NodeTarget]]:
    grouped: Dict[str, List[EdgeRow]] = defaultdict(list)
    for row in edge_rows:
        key = canonical_circuit_key(row.metric, row.polarity)
        if key in CANONICAL_CIRCUIT_KEYS:
            grouped[key].append(row)

    derived: Dict[str, List[NodeTarget]] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: abs(r.weight), reverse=True)[:topk_edges]
        attn_scores: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = {}
        mlp_scores: Dict[int, float] = defaultdict(float)

        for row in rows:
            for endpoint in (row.source, row.target):
                parsed = parse_node_name(endpoint)
                if not parsed:
                    continue
                if parsed["kind"] == "attn":
                    score_key = (parsed["layer"], parsed.get("path"))
                    if score_key not in attn_scores:
                        attn_scores[score_key] = {
                            "score": 0.0,
                            "heads": defaultdict(float),
                            "path": parsed.get("path"),
                        }
                    attn_scores[score_key]["score"] += abs(row.weight)
                    attn_scores[score_key]["heads"][parsed["head"]] += abs(row.weight)
                elif parsed["kind"] == "attn_layer":
                    score_key = (parsed["layer"], parsed.get("path"))
                    if score_key not in attn_scores:
                        attn_scores[score_key] = {"score": 0.0, "heads": defaultdict(float), "path": parsed.get("path")}
                    attn_scores[score_key]["score"] += abs(row.weight)
                elif parsed["kind"] == "mlp":
                    mlp_scores[parsed["layer"]] += abs(row.weight)

        targets: List[NodeTarget] = []
        for (layer, path_tag), info in sorted(attn_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)[:max_attn_groups_per_circuit]:
            sorted_heads = sorted(info["heads"].items(), key=lambda kv: kv[1], reverse=True)
            heads = [h for h, _ in sorted_heads[:6]] if sorted_heads else None
            targets.append(
                NodeTarget(
                    layer=layer,
                    component="attn",
                    heads=heads,
                    path=path_tag,
                    role="edge_derived",
                    weight=1.0,
                    preferred_token_policy="last",
                    attention_mode="none",
                )
            )

        for layer, score in sorted(mlp_scores.items(), key=lambda kv: kv[1], reverse=True)[:max_mlp_per_circuit]:
            targets.append(NodeTarget(layer=layer, component="mlp", role="edge_derived", weight=1.0))

        # stable order by layer
        targets = sorted(targets, key=lambda t: (t.layer, 0 if t.component == "attn" else 1))
        derived[key] = targets
    return derived

def merge_presets_with_edge_targets(presets: Dict[str, CircuitPreset], edge_targets: Dict[str, List[NodeTarget]], keep_default_if_missing: bool = True,) -> Dict[str, CircuitPreset]:
    merged = {k: clone_circuit_preset(v) for k, v in presets.items()}
    for key, targets in edge_targets.items():
        if key not in merged:
            continue
        if targets:
            merged[key].targets = [clone_node_target(target) for target in targets]
            merged[key].notes += " | targets derived from edge table"
        elif not keep_default_if_missing:
            merged[key].targets = []
    return merged

# -----------------------------------------------------------------------------
# External sample score aggregation (optional)
# -----------------------------------------------------------------------------
def zscore_map(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mu = mean_safe(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    if sd < 1e-8:
        return [0.0 for _ in values]
    return [(v - mu) / sd for v in values]

class ExternalCircuitScorer:
    """
    Optional post-hoc scorer if you already have sample-level circuit scores.
    Expected columns (JSONL / CSV row):
      id, aum_high, aum_low, el2n_high, el2n_low, gradalign_high, gradalign_low
    Additional aliases like grad_high / grad_low are normalized when possible.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.by_id: Dict[str, Dict[str, float]] = {}
        for rec in records:
            rid = str(rec.get("id") or rec.get("sample_id") or rec.get("uid") or "")
            if not rid:
                continue
            normalized: Dict[str, float] = {}
            for key in CANONICAL_CIRCUIT_KEYS:
                aliases = [
                    key,
                    key.replace("gradalign", "grad"),
                    key.replace("_", "-"),
                    key.upper(),
                ]
                for alias in aliases:
                    if alias in rec:
                        try:
                            normalized[key] = float(rec[alias])
                            break
                        except (TypeError, ValueError):
                            pass
            self.by_id[rid] = normalized

    @classmethod
    def from_path(cls, path: str | Path) -> "ExternalCircuitScorer":
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            records = load_jsonl(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
        return cls(records)

    def get(self, sample_id: str) -> Dict[str, float]:
        return self.by_id.get(str(sample_id), {})

    @staticmethod
    def aggregate(scores: Mapping[str, float], target_band: str) -> float:
        aum_h = float(scores.get("aum_high", 0.0))
        aum_l = float(scores.get("aum_low", 0.0))
        el2n_h = float(scores.get("el2n_high", 0.0))
        el2n_l = float(scores.get("el2n_low", 0.0))
        grad_h = float(scores.get("gradalign_high", 0.0))
        grad_l = float(scores.get("gradalign_low", 0.0))
        if target_band == "easy":
            return 1.0 * aum_h + 1.1 * el2n_l - 0.8 * el2n_h - 0.5 * aum_l + 0.2 * grad_h + 0.2 * grad_l
        if target_band == "hard":
            return 1.1 * el2n_h + 0.6 * grad_h - 0.9 * aum_h - 0.8 * el2n_l - 0.4 * grad_l + 0.2 * aum_l
        if target_band == "helpful":
            return 1.2 * grad_h + 0.5 * aum_h + 0.4 * el2n_l - 0.7 * grad_l - 0.3 * aum_l
        if target_band == "medium":
            return 0.5 * grad_h + 0.3 * aum_h + 0.3 * el2n_l + 0.2 * el2n_h - 0.2 * grad_l
        return 0.0

# -----------------------------------------------------------------------------
# Prompt builders for MCQA generation / rewriting
# -----------------------------------------------------------------------------
def infer_text_fields(record: Mapping[str, Any]) -> Tuple[str, str]:
    question = (
        record.get("question")
        or record.get("prompt")
        or record.get("input")
        or record.get("text")
        or record.get("stem")
        or ""
    )
    answer = record.get("answer") or record.get("completion") or record.get("output") or record.get("rationale") or ""
    return str(question), str(answer)

def build_system_prompt(target_band: str, prompt_mode: str = "source") -> str:
    base = (
        "You are generating high-quality science multiple-choice training data. "
        "Always keep the science factually correct, avoid trivia leakage, and return strict JSON. "
        "CRITICAL FORMAT RULES: "
        "(1) Each element of 'choices' must be plain text only — do NOT include any letter or number prefix. "
        "Write the choice text directly (e.g. 'Coriolis Effect', not 'A. Coriolis Effect' or '1. Coriolis Effect'). "
        "(2) The 'answer' field must be exactly one uppercase letter — A, B, C, or D — "
        "indicating the position of the correct choice in the choices array (A=first, B=second, C=third, D=fourth). "
        "Never use a digit ('1', '2') or full text as the answer value."
    )
    if prompt_mode == "fixed" or target_band == "naive":
        return base + " Create one self-contained science MCQA sample with plausible distractors and a concise rationale."
    if target_band == "easy":
        return base + " Bias the problem toward a single-hop canonical solution with clear but nontrivial distractors."
    if target_band == "hard":
        return base + " Bias the problem toward deeper reasoning, ambiguity resolution, or multi-step evidence combination."
    if target_band == "helpful":
        return base + " Bias the sample toward being pedagogically helpful: crisp rationale, decisive evidence, and distractor explanations."
    return base + " Target medium difficulty with balanced distractor strength and concise rationale."

def build_user_prompt(record: Mapping[str, Any], target_band: str, prompt_mode: str = "source") -> str:
    question, answer = infer_text_fields(record)
    topic = record.get("topic") or record.get("subject") or record.get("domain") or "science"
    # choices must be plain text — no "A. " / "1. " prefix in the example either,
    # as the model tends to mimic the format shown here.
    base_json_spec = (
        '{"question": "...", "choices": ["<choice text>", "<choice text>", "<choice text>", "<choice text>"], '
        '"answer": "A", "rationale": "...", "difficulty": "easy|medium|hard", "helpful": true}'
    )
    if prompt_mode == "fixed" or target_band == "naive":
        if question:
            return (
                f"Source topic: {topic}\n"
                f"Source item (for inspiration only, do not copy wording):\n{question}\n"
                f"Source answer (optional): {answer}\n\n"
                "Create ONE new science MCQA sample in a similar domain. "
                "Keep it self-contained, factually correct, and naturally written. "
                "Do not explicitly mention a target difficulty band in the output.\n"
                f"Return only JSON with schema: {base_json_spec}"
            )
        return (
            f"Topic: {topic}\n"
            "Create ONE new science MCQA sample. Keep it self-contained, factually correct, and naturally written.\n"
            f"Return only JSON with schema: {base_json_spec}"
        )
    if question:
        if target_band == "easy":
            return (
                f"Source topic: {topic}\n"
                f"Source item (for inspiration only, do not copy wording):\n{question}\n"
                f"Source answer (optional): {answer}\n\n"
                "Create ONE new MCQA sample that is easier than the source. "
                "Use a single canonical science fact or one direct concept application. "
                "Distractors should be plausible but clearly eliminable by the decisive concept.\n"
                f"Return only JSON with schema: {base_json_spec}"
            )
        if target_band == "hard":
            return (
                f"Source topic: {topic}\n"
                f"Source item (for inspiration only, do not copy wording):\n{question}\n"
                f"Source answer (optional): {answer}\n\n"
                "Create ONE new MCQA sample that is harder than the source. "
                "Require at least one extra inference step, concept disambiguation, or stronger distractor discrimination.\n"
                f"Return only JSON with schema: {base_json_spec}"
            )
        if target_band == "helpful":
            return (
                f"Source topic: {topic}\n"
                f"Source item (for inspiration only, do not copy wording):\n{question}\n"
                f"Source answer (optional): {answer}\n\n"
                "Create ONE new MCQA sample optimized to be instructional and helpful. "
                "The rationale must explicitly state the decisive evidence and why at least one distractor is wrong.\n"
                f"Return only JSON with schema: {base_json_spec}"
            )
        return (
            f"Source topic: {topic}\n"
            f"Source item (for inspiration only, do not copy wording):\n{question}\n"
            f"Source answer (optional): {answer}\n\n"
            "Create ONE new medium-difficulty MCQA sample with balanced distractors and compact rationale.\n"
            f"Return only JSON with schema: {base_json_spec}"
        )

    # no source question provided -> topic-driven generation
    if target_band == "easy":
        return (
            f"Topic: {topic}\n"
            "Create ONE easy science MCQA sample. Use a single direct concept and clear distractor separation.\n"
            f"Return only JSON with schema: {base_json_spec}"
        )
    if target_band == "hard":
        return (
            f"Topic: {topic}\n"
            "Create ONE hard science MCQA sample. Require at least two linked concepts or nontrivial distractor rejection.\n"
            f"Return only JSON with schema: {base_json_spec}"
        )
    if target_band == "helpful":
        return (
            f"Topic: {topic}\n"
            "Create ONE instructional science MCQA sample with a rationale that teaches the key distinction.\n"
            f"Return only JSON with schema: {base_json_spec}"
        )
    return (
        f"Topic: {topic}\n"
        "Create ONE medium-difficulty science MCQA sample with balanced distractors.\n"
        f"Return only JSON with schema: {base_json_spec}"
    )


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
class TransformerImportError(RuntimeError):
    pass

def require_transformers() -> Any:
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise TransformerImportError(
            "transformers is required to run this script. Install with `pip install transformers accelerate`."
        ) from exc
    return transformers

def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    force_eager_attention: bool = True,
    prefer_single_device: bool = False,
) -> Tuple[Any, Any]:
    transformers = require_transformers()

    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype.lower(), torch.bfloat16)

    extra_kwargs: Dict[str, Any] = {}
    if force_eager_attention:
        # Works for current HF style where attention implementation can be selected.
        extra_kwargs["attn_implementation"] = "eager"

    resolved_device_map: Any = device_map
    if (
        prefer_single_device
        and isinstance(device_map, str)
        and device_map == "auto"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        # Hook-based activation collection / steering is numerically brittle under
        # model-parallel auto sharding. Keep the full model on cuda:0 by default.
        resolved_device_map = {"": 0}
        print("[info] prefer_single_device enabled: overriding device_map=auto -> {'': 0} for steering stability")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=resolved_device_map,
        trust_remote_code=True,
        **extra_kwargs,
    )
    model.eval()

    if force_eager_attention:
        # Some models keep the flag on config rather than constructor-only.
        if hasattr(model, "config"):
            try:
                setattr(model.config, "_attn_implementation", "eager")
            except Exception:
                pass

    return model, tokenizer

def get_base_model(model: Any) -> nn.Module:
    # Common Hugging Face causal LM shapes.
    candidates = [
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "transformer", None),
    ]
    for cand in candidates:
        if cand is not None and hasattr(cand, "layers"):
            return cand
    if hasattr(model, "layers"):
        return model
    raise ValueError("Could not find decoder layers on the supplied model.")

def get_decoder_layers(model: Any) -> Sequence[nn.Module]:
    base = get_base_model(model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("Decoder layers not found.")
    return layers

def get_module_path_map(model: Any) -> Dict[str, nn.Module]:
    layers = get_decoder_layers(model)
    path_map: Dict[str, nn.Module] = {}
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            path_map[f"L{idx:02d}.attn"] = layer.self_attn
        if hasattr(layer, "mlp"):
            path_map[f"L{idx:02d}.mlp"] = layer.mlp
    return path_map

def target_to_module_key(target: NodeTarget) -> str:
    return f"L{target.layer:02d}.{'attn' if target.component == 'attn' else 'mlp'}"

def infer_num_attention_heads(model: Any) -> int:
    for layer in get_decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "num_heads"):
            return int(getattr(attn, "num_heads"))
        head_dim = getattr(attn, "head_dim", None)
        q_proj = getattr(attn, "q_proj", None)
        out_features = getattr(q_proj, "out_features", None)
        if head_dim and out_features:
            return int(out_features // head_dim)
    raise ValueError("Could not infer the model's number of attention heads.")

def _sample_layer_for_random_target(
    rng: random.Random,
    available_layers: Sequence[int],
    used_layers: set[int],
    excluded_layers: set[int],
) -> int:
    preferred = [layer for layer in available_layers if layer not in used_layers and layer not in excluded_layers]
    if not preferred:
        preferred = [layer for layer in available_layers if layer not in used_layers]
    if not preferred:
        preferred = [layer for layer in available_layers if layer not in excluded_layers]
    if not preferred:
        preferred = list(available_layers)
    return int(rng.choice(preferred))

def build_randomized_presets(
    presets: Mapping[str, CircuitPreset],
    model: Any,
    seed: int,
) -> Dict[str, CircuitPreset]:
    attn_layers = [idx for idx, layer in enumerate(get_decoder_layers(model)) if hasattr(layer, "self_attn")]
    mlp_layers = [idx for idx, layer in enumerate(get_decoder_layers(model)) if hasattr(layer, "mlp")]
    if not attn_layers and not mlp_layers:
        raise ValueError("Could not build random circuit presets: no decoder attention/MLP layers found.")
    num_heads = infer_num_attention_heads(model) if attn_layers else 0

    randomized: Dict[str, CircuitPreset] = {}
    for preset_idx, key in enumerate(sorted(presets.keys())):
        preset = presets[key]
        circuit_rng = random.Random(seed + 1009 * (preset_idx + 1))
        random_preset = clone_circuit_preset(preset)
        random_preset.notes += " | matched random target baseline"
        random_preset.description += " [random-target baseline]"

        attn_excluded = {target.layer for target in preset.targets if target.component == "attn"}
        mlp_excluded = {target.layer for target in preset.targets if target.component == "mlp"}
        used_attn_layers: set[int] = set()
        used_mlp_layers: set[int] = set()
        randomized_targets: List[NodeTarget] = []

        for target in preset.targets:
            new_target = clone_node_target(target)
            if target.component == "attn" and attn_layers:
                new_target.layer = _sample_layer_for_random_target(
                    circuit_rng,
                    attn_layers,
                    used_attn_layers,
                    attn_excluded,
                )
                used_attn_layers.add(new_target.layer)
                if target.heads is not None and num_heads > 0:
                    head_count = min(len(target.heads), num_heads)
                    available_heads = list(range(num_heads))
                    if new_target.layer == target.layer:
                        remaining = [head for head in available_heads if head not in set(target.heads)]
                        if len(remaining) >= head_count:
                            available_heads = remaining
                    new_target.heads = sorted(circuit_rng.sample(available_heads, head_count))
            elif target.component == "mlp" and mlp_layers:
                new_target.layer = _sample_layer_for_random_target(
                    circuit_rng,
                    mlp_layers,
                    used_mlp_layers,
                    mlp_excluded,
                )
                used_mlp_layers.add(new_target.layer)
            randomized_targets.append(new_target)

        random_preset.targets = randomized_targets
        randomized[key] = random_preset
    return randomized

# -----------------------------------------------------------------------------
# Activation collection and vector bank
# -----------------------------------------------------------------------------
class MeanActivationCollector:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        module_keys: Sequence[str],
        max_length: int = 512,
        default_token_policy: str = "last_nonpad",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.module_keys = list(dict.fromkeys(module_keys))
        self.max_length = max_length
        self.default_token_policy = default_token_policy
        self.module_map = get_module_path_map(model)
        self.current_attention_mask: Optional[torch.Tensor] = None
        self.batch_cache: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        for key in self.module_keys:
            if key not in self.module_map:
                print(f"[warn] activation collector skipped missing module {key}")
                continue
            module = self.module_map[key]

            def make_hook(module_key: str) -> Callable:
                def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                    tensor = output[0] if isinstance(output, tuple) else output
                    if not isinstance(tensor, torch.Tensor):
                        return
                    selected = self._select_features(tensor, self.current_attention_mask, self.default_token_policy)
                    self.batch_cache[module_key] = selected.detach().cpu()
                return hook

            self.hooks.append(module.register_forward_hook(make_hook(key)))

    @staticmethod
    def _select_features(tensor: torch.Tensor, attention_mask: Optional[torch.Tensor], token_policy: str) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor of shape [B, S, H] or [B, H], got {tuple(tensor.shape)}")
        mask = attention_mask.to(device=tensor.device) if attention_mask is not None else None
        if token_policy == "all_mean":
            if mask is None:
                return tensor.mean(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            return (tensor * mask.unsqueeze(-1)).sum(dim=1) / denom
        if token_policy == "last":
            return tensor[:, -1, :]
        # last_nonpad default
        if mask is None:
            return tensor[:, -1, :]
        idx = mask.long().sum(dim=1).clamp_min(1) - 1
        idx = idx.clamp_max(tensor.size(1) - 1)  # safety guard for any residual mask/sequence mismatch
        batch_idx = torch.arange(tensor.size(0), device=tensor.device)
        return tensor[batch_idx, idx, :]

    def collect_mean(self, texts: Sequence[str], batch_size: int = 4, max_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if max_samples is not None:
            texts = list(texts)[:max_samples]
        sums: Dict[str, Optional[torch.Tensor]] = {k: None for k in self.module_keys}
        counts: Dict[str, int] = {k: 0 for k in self.module_keys}

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if not batch:
                continue
            toks = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.model.device) for k, v in toks.items()}
            self.current_attention_mask = toks.get("attention_mask")
            self.batch_cache = {}
            with torch.no_grad():
                _ = self.model(**toks)
            for key, feats in self.batch_cache.items():
                if sums[key] is None:
                    sums[key] = feats.sum(dim=0)
                else:
                    sums[key] = sums[key] + feats.sum(dim=0)
                counts[key] += feats.shape[0]

        result: Dict[str, torch.Tensor] = {}
        for key in self.module_keys:
            if sums[key] is None or counts[key] == 0:
                continue
            result[key] = sums[key] / counts[key]
        return result

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class VectorBankBuilder:
    """
    Builds circuit-specific steering vectors from sample-level scalar scores.
    Expected calibration JSONL fields:
        - prompt/question/text/input
        - optional completion/answer/output
        - aum, el2n, gradalign (float)
    We compute high-vs-low prototype means per metric and define:
        vector[metric_high] = mean(high group) - mean(low group)
        vector[metric_low]  = -vector[metric_high]
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        presets: Mapping[str, CircuitPreset],
        quantile: float = 0.15,
        max_texts_per_group: int = 256,
        batch_size: int = 4,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.presets = presets
        self.quantile = quantile
        self.max_texts_per_group = max_texts_per_group
        self.batch_size = batch_size
        self.max_length = max_length

    @staticmethod
    def _record_to_text(rec: Mapping[str, Any]) -> str:
        question, answer = infer_text_fields(rec)
        if answer:
            return f"{question}\nAnswer: {answer}"
        return question

    def _top_bottom_texts(self, records: Sequence[Mapping[str, Any]], metric_field: str) -> Tuple[List[str], List[str]]:
        valid: List[Tuple[float, str]] = []
        for rec in records:
            if metric_field not in rec:
                continue
            try:
                score = float(rec[metric_field])
            except (TypeError, ValueError):
                continue
            text = self._record_to_text(rec)
            if text.strip():
                valid.append((score, text))
        if not valid:
            return [], []
        valid.sort(key=lambda x: x[0])
        k = max(4, int(len(valid) * self.quantile))
        low = [txt for _, txt in valid[:k]]
        high = [txt for _, txt in valid[-k:]]
        return high[: self.max_texts_per_group], low[: self.max_texts_per_group]

    def build_from_records(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        module_keys: List[str] = []
        for preset in self.presets.values():
            for target in preset.targets:
                module_keys.append(target_to_module_key(target))
        module_keys = list(dict.fromkeys(module_keys))

        collector = MeanActivationCollector(
            self.model,
            self.tokenizer,
            module_keys=module_keys,
            max_length=self.max_length,
            default_token_policy="last_nonpad",
        )

        bank: Dict[str, Dict[str, torch.Tensor]] = {}
        field_map = {
            "aum": "aum",
            "el2n": "el2n",
            "gradalign": "gradalign",
        }
        try:
            for metric_key, metric_field in field_map.items():
                high_texts, low_texts = self._top_bottom_texts(records, metric_field)
                if _VERBOSE:
                    print_rule(f"Building vectors for {metric_key}", "-")
                    print_kv("high_group_size", len(high_texts))
                    print_kv("low_group_size", len(low_texts))
                if not high_texts or not low_texts:
                    print(f"[warn] skipped {metric_key}: missing high/low calibration texts")
                    continue

                mean_high = collector.collect_mean(high_texts, batch_size=self.batch_size, max_samples=self.max_texts_per_group)
                mean_low = collector.collect_mean(low_texts, batch_size=self.batch_size, max_samples=self.max_texts_per_group)
                diff: Dict[str, torch.Tensor] = {}
                for key in module_keys:
                    if key in mean_high and key in mean_low:
                        vec = mean_high[key] - mean_low[key]
                        diff[key] = vec
                        if _VERBOSE:
                            print(f"[vector] {metric_key}_high {key}: norm={vec.norm().item():.4f}")
                if not diff:
                    print(f"[warn] no usable vectors collected for {metric_key}")
                    continue
                bank[f"{metric_key}_high"] = diff
                bank[f"{metric_key}_low"] = {k: -v for k, v in diff.items()}
        finally:
            collector.close()
        return bank

# -----------------------------------------------------------------------------
# Activation addition steering
# -----------------------------------------------------------------------------
class ActivationAddSteerer:
    def __init__(
        self,
        model: Any,
        presets: Mapping[str, CircuitPreset],
        vector_bank: Mapping[str, Mapping[str, torch.Tensor]],
        activation_scales: Mapping[str, float],
        token_policy: str = "last_nonpad",
        global_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.presets = presets
        self.vector_bank = vector_bank
        self.activation_scales = dict(activation_scales)
        self.token_policy = token_policy
        self.global_scale = global_scale
        self.module_map = get_module_path_map(model)
        self.current_attention_mask: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []
        self.module_deltas = self._build_module_deltas()

    def _build_module_deltas(self) -> Dict[str, torch.Tensor]:
        deltas: Dict[str, Optional[torch.Tensor]] = defaultdict(lambda: None)
        for circuit_key, scale in self.activation_scales.items():
            if abs(scale) < 1e-8:
                continue
            if circuit_key not in self.presets:
                continue
            if circuit_key not in self.vector_bank:
                print(f"[warn] activation steering skipped {circuit_key}: no vector bank entry")
                continue
            preset = self.presets[circuit_key]
            for target in preset.targets:
                module_key = target_to_module_key(target)
                vec = self.vector_bank[circuit_key].get(module_key)
                if vec is None:
                    continue
                weighted = vec * (scale * target.weight * self.global_scale)
                if deltas[module_key] is None:
                    deltas[module_key] = weighted.clone()
                else:
                    deltas[module_key] = deltas[module_key] + weighted
        final: Dict[str, torch.Tensor] = {}
        for key, vec in deltas.items():
            if vec is not None:
                final[key] = vec
        return final

    @staticmethod
    def _apply_delta(
        tensor: torch.Tensor,
        delta: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_policy: str,
    ) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor + delta.to(dtype=tensor.dtype, device=tensor.device)
        if tensor.ndim != 3:
            return tensor
        delta = delta.to(dtype=tensor.dtype, device=tensor.device)
        mask = attention_mask.to(device=tensor.device) if attention_mask is not None else None
        out = tensor.clone()
        if token_policy == "all":
            out = out + delta.view(1, 1, -1)
            return out
        if token_policy == "all_mean":
            out = out + delta.view(1, 1, -1)
            return out
        if token_policy == "last":
            out[:, -1, :] += delta
            return out
        # last_nonpad default
        # During KV-cache decode steps model.generate() feeds only the new token,
        # so tensor.size(1)==1 while attention_mask still spans the full sequence.
        # In that case applying to [:, -1, :] (the only slot) is always correct.
        if mask is None or tensor.size(1) == 1:
            out[:, -1, :] += delta
            return out
        idx = mask.long().sum(dim=1).clamp_min(1) - 1
        idx = idx.clamp_max(out.size(1) - 1)  # safety guard against any remaining mismatch
        batch_idx = torch.arange(out.size(0), device=out.device)
        out[batch_idx, idx, :] += delta
        return out

    def install(self) -> None:
        if _VERBOSE:
            print_rule("Installing activation steering", "-")
        for module_key, delta in self.module_deltas.items():
            if module_key not in self.module_map:
                print(f"[warn] cannot install activation hook: missing {module_key}")
                continue
            module = self.module_map[module_key]
            if _VERBOSE:
                print(f"[hook] {module_key} | delta_norm={delta.norm().item():.4f}")

            def make_hook(key: str, delta_vec: torch.Tensor) -> Callable:
                def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
                    tensor = output[0] if isinstance(output, tuple) else output
                    if not isinstance(tensor, torch.Tensor):
                        return output
                    steered = self._apply_delta(tensor, delta_vec, self.current_attention_mask, self.token_policy)
                    if isinstance(output, tuple):
                        return (steered, *output[1:])
                    return steered
                return hook

            self.hooks.append(module.register_forward_hook(make_hook(module_key, delta)))

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self.current_attention_mask = attention_mask

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# -----------------------------------------------------------------------------
# Qwen2 attention-logit steering
# -----------------------------------------------------------------------------
@dataclass
class LayerAttentionPolicy:
    heads: Optional[List[int]] = None
    temperature_delta: float = 0.0  # positive => sharpen, negative => flatten
    prefix_bias: float = 0.0
    recent_bias: float = 0.0
    recent_window: int = 16

class Qwen2AttentionSteerer:
    """
    Head-level attention steering for HF Qwen2/Qwen2.5 style models.
    This patch mirrors the official eager-attention path from Hugging Face's Qwen2 implementation,
    where q_proj / k_proj / v_proj / o_proj exist and attention_mask is added before softmax.
    If those assumptions fail, the patcher will skip installation and print a warning.
    """

    def __init__(
        self,
        model: Any,
        presets: Mapping[str, CircuitPreset],
        attention_scales: Mapping[str, float],
        temperature_clip: Tuple[float, float] = (0.70, 1.50),
    ) -> None:
        self.model = model
        self.presets = presets
        self.attention_scales = dict(attention_scales)
        self.temperature_clip = temperature_clip
        self.layer_policies = self._build_policies()
        self.patched_modules: List[Tuple[nn.Module, Callable[..., Any]]] = []
        self._hf_helpers = None

    def _load_qwen2_helpers(self) -> Optional[Tuple[Any, Any]]:
        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv  # type: ignore
            return apply_rotary_pos_emb, repeat_kv
        except Exception as exc:
            print(f"[warn] Qwen2 attention steering unavailable: {exc}")
            return None

    def _build_policies(self) -> Dict[int, LayerAttentionPolicy]:
        per_layer: Dict[int, LayerAttentionPolicy] = defaultdict(LayerAttentionPolicy)
        for circuit_key, scale in self.attention_scales.items():
            if abs(scale) < 1e-8:
                continue
            preset = self.presets.get(circuit_key)
            if preset is None:
                continue
            for target in preset.targets:
                if target.component != "attn":
                    continue
                if target.heads is None:
                    # No head-level information => skip fine-grained attention manipulation.
                    continue
                policy = per_layer[target.layer]
                existing_heads = set(policy.heads or [])
                existing_heads.update(target.heads)
                policy.heads = sorted(existing_heads)
                policy.temperature_delta += scale * 0.40 * target.weight
                if target.attention_mode in {"prefix", "prefix+recent"}:
                    policy.prefix_bias += scale * 0.20 * target.weight
                if target.attention_mode in {"recent", "prefix+recent"}:
                    policy.recent_bias += scale * 0.12 * target.weight
                    policy.recent_window = max(policy.recent_window, target.recent_window)
        return dict(per_layer)

    @staticmethod
    def _make_recent_mask(q_len: int, k_len: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q_pos = torch.arange(k_len - q_len, k_len, device=device)  # works for decode as well
        k_pos = torch.arange(k_len, device=device)
        dist = q_pos[:, None] - k_pos[None, :]
        mask = ((dist >= 1) & (dist <= window)).to(dtype=dtype)
        return mask

    def _apply_policy(self, layer_idx: int, attn_logits: torch.Tensor) -> torch.Tensor:
        # attn_logits: [B, H, Q, K]
        policy = self.layer_policies.get(layer_idx)
        if policy is None or not policy.heads:
            return attn_logits
        out = attn_logits
        device = out.device
        dtype = out.dtype
        heads = [h for h in policy.heads if 0 <= h < out.shape[1]]
        if not heads:
            return out
        head_index = torch.tensor(heads, device=device, dtype=torch.long)

        if abs(policy.temperature_delta) > 1e-8:
            temp = clamp(1.0 - policy.temperature_delta, self.temperature_clip[0], self.temperature_clip[1])
            out[:, head_index, :, :] = out[:, head_index, :, :] / temp

        if abs(policy.prefix_bias) > 1e-8:
            prefix_len = max(1, int(out.shape[-1] * 0.30))
            out[:, head_index, :, :prefix_len] = out[:, head_index, :, :prefix_len] + policy.prefix_bias

        if abs(policy.recent_bias) > 1e-8:
            recent_mask = self._make_recent_mask(
                q_len=out.shape[-2],
                k_len=out.shape[-1],
                window=policy.recent_window,
                device=device,
                dtype=dtype,
            )
            out[:, head_index, :, :] = out[:, head_index, :, :] + policy.recent_bias * recent_mask.view(1, 1, out.shape[-2], out.shape[-1])
        return out

    def install(self) -> None:
        self._hf_helpers = self._load_qwen2_helpers()
        if self._hf_helpers is None:
            print("[warn] attention steering not installed")
            return

        apply_rotary_pos_emb, repeat_kv = self._hf_helpers
        layers = get_decoder_layers(self.model)
        if _VERBOSE:
            print_rule("Installing Qwen2 attention steering", "-")
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in self.layer_policies:
                continue
            if not hasattr(layer, "self_attn"):
                continue
            attn_module = layer.self_attn
            required = ["q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "num_key_value_groups", "scaling", "attention_dropout", "layer_idx"]
            if not all(hasattr(attn_module, name) for name in required):
                print(f"[warn] skipped layer {layer_idx} attention patch: incompatible module")
                continue
            if hasattr(attn_module, "_circuit_original_forward"):
                continue

            policy = self.layer_policies[layer_idx]
            if _VERBOSE:
                print(f"[attn-patch] layer={layer_idx} heads={policy.heads} temp_delta={policy.temperature_delta:.3f} prefix_bias={policy.prefix_bias:.3f} recent_bias={policy.recent_bias:.3f}")
            original_forward = attn_module.forward
            steerer = self

            def patched_forward(
                module_self: nn.Module,
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_values: Optional[Any] = None,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module_self.head_dim)
                query_states = module_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = module_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = module_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, module_self.layer_idx)

                key_states_rep = repeat_kv(key_states, module_self.num_key_value_groups)
                value_states_rep = repeat_kv(value_states, module_self.num_key_value_groups)
                attn_logits = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * module_self.scaling
                if attention_mask is not None:
                    attn_logits = attn_logits + attention_mask
                attn_logits = steerer._apply_policy(module_self.layer_idx, attn_logits)
                attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = F.dropout(attn_weights, p=0.0 if not module_self.training else module_self.attention_dropout, training=module_self.training)
                attn_output = torch.matmul(attn_weights, value_states_rep)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = module_self.o_proj(attn_output)
                return attn_output, attn_weights

            attn_module._circuit_original_forward = original_forward  # type: ignore[attr-defined]
            attn_module.forward = types.MethodType(patched_forward, attn_module)
            self.patched_modules.append((attn_module, original_forward))

    def remove(self) -> None:
        for module, original_forward in self.patched_modules:
            module.forward = original_forward
            if hasattr(module, "_circuit_original_forward"):
                delattr(module, "_circuit_original_forward")
        self.patched_modules = []

# -----------------------------------------------------------------------------
# Steering context manager
# -----------------------------------------------------------------------------
class SteeringSession:
    def __init__(
        self,
        model: Any,
        presets: Mapping[str, CircuitPreset],
        vector_bank: Optional[Mapping[str, Mapping[str, torch.Tensor]]],
        activation_scales: Mapping[str, float],
        attention_scales: Mapping[str, float],
        use_activation_addition: bool,
        use_attention_steering: bool,
        token_policy: str = "last_nonpad",
        activation_global_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.presets = presets
        self.vector_bank = vector_bank or {}
        self.activation_scales = dict(activation_scales)
        self.attention_scales = dict(attention_scales)
        self.use_activation_addition = use_activation_addition
        self.use_attention_steering = use_attention_steering
        self.token_policy = token_policy
        self.activation_global_scale = activation_global_scale
        self.activation_steerer: Optional[ActivationAddSteerer] = None
        self.attn_steerer: Optional[Qwen2AttentionSteerer] = None

    def __enter__(self) -> "SteeringSession":
        if self.use_activation_addition and self.vector_bank:
            self.activation_steerer = ActivationAddSteerer(
                self.model,
                self.presets,
                self.vector_bank,
                self.activation_scales,
                token_policy=self.token_policy,
                global_scale=self.activation_global_scale,
            )
            self.activation_steerer.install()
        elif self.use_activation_addition:
            print("[warn] activation steering requested, but vector bank is empty")

        if self.use_attention_steering:
            self.attn_steerer = Qwen2AttentionSteerer(self.model, self.presets, self.attention_scales)
            self.attn_steerer.install()
        return self

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        if self.activation_steerer is not None:
            self.activation_steerer.set_attention_mask(attention_mask)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.activation_steerer is not None:
            self.activation_steerer.remove()
        if self.attn_steerer is not None:
            self.attn_steerer.remove()

# -----------------------------------------------------------------------------
# Generation and scoring
# -----------------------------------------------------------------------------
class CircuitAwareGenerator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        presets: Mapping[str, CircuitPreset],
        vector_bank: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        max_input_length: int = 768,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.presets = presets
        self.vector_bank = vector_bank or {}
        self.max_input_length = max_input_length

    def render_prompt_text(self, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    def _prepare_inputs(self, system_prompt: str, user_prompt: str) -> Dict[str, torch.Tensor]:
        text = self.render_prompt_text(system_prompt, user_prompt)
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        toks = {k: v.to(self.model.device) for k, v in toks.items()}
        return toks

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        activation_scales: Mapping[str, float],
        attention_scales: Mapping[str, float],
        use_activation_addition: bool,
        use_attention_steering: bool,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> str:
        if seed is not None:
            set_seed(seed)
        toks = self._prepare_inputs(system_prompt, user_prompt)
        with SteeringSession(
            self.model,
            self.presets,
            self.vector_bank,
            activation_scales,
            attention_scales,
            use_activation_addition=use_activation_addition,
            use_attention_steering=use_attention_steering,
        ) as session:
            session.set_attention_mask(toks.get("attention_mask"))
            with torch.no_grad():
                do_sample = temperature > 0
                generate_kwargs: Dict[str, Any] = {
                    **toks,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if do_sample:
                    generate_kwargs.update(
                        {
                            "temperature": max(temperature, 1e-5),
                            "top_p": top_p,
                        }
                    )
                    # Always clean up invalid logits to prevent multinomial NaN crashes.
                    # remove_invalid_values replaces NaN/inf logits with -inf before sampling.
                    # renormalize_logits ensures the distribution sums to 1 after all warpers.
                    # top_k is intentionally NOT set: it distorts Qwen2.5 generation quality.
                    generate_kwargs.update(
                        {
                            "remove_invalid_values": True,
                            "renormalize_logits": True,
                        }
                    )
                output_ids = self.model.generate(**generate_kwargs)
        new_tokens = output_ids[0, toks["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def next_token_option_probs(self, prompt_text: str, options: Sequence[str] = ("A", "B", "C", "D"),) -> Dict[str, float]:
        toks = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        toks = {k: v.to(self.model.device) for k, v in toks.items()}

        with torch.no_grad():
            logits = self.model(**toks).logits[0, -1]

        raw = {}
        for opt in options:
            ids = self.tokenizer.encode(" " + opt, add_special_tokens=False)
            if len(ids) == 1:
                raw[opt] = float(logits[ids[0]].item())
            else:
                # tokenizer가 " A"를 1토큰으로 안 쪼개줄 때 fallback
                ids = self.tokenizer.encode(opt, add_special_tokens=False)
                raw[opt] = float(logits[ids[0]].item())

        vals = torch.tensor(list(raw.values()), dtype=torch.float32)
        probs = torch.softmax(vals, dim=-1).tolist()
        return {k: p for k, p in zip(raw.keys(), probs)}
    
    def normalized_logprob(
        self,
        prompt_text: str,
        completion_text: str,
        activation_scales: Mapping[str, float],
        attention_scales: Mapping[str, float],
        use_activation_addition: bool,
        use_attention_steering: bool,
    ) -> float:
        full_text = prompt_text + completion_text
        full = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_input_length)
        prompt = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_input_length)
        full = {k: v.to(self.model.device) for k, v in full.items()}
        prompt_len = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        with SteeringSession(
            self.model,
            self.presets,
            self.vector_bank,
            activation_scales,
            attention_scales,
            use_activation_addition=use_activation_addition,
            use_attention_steering=use_attention_steering,
        ) as session:
            session.set_attention_mask(full.get("attention_mask"))
            with torch.no_grad():
                outputs = self.model(**full, labels=labels)
        loss = float(outputs.loss.item())
        return -loss

# -----------------------------------------------------------------------------
# Candidate parsing / ranking
# -----------------------------------------------------------------------------
def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    # direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # best-effort: locate first and last braces
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return {}
    snippet = text[first : last + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}
    
def build_mcqa_eval_prompt(parsed: Mapping[str, Any], include_rationale: bool = False) -> str:
    choices = parsed.get("choices", [])
    text = (
        "Answer the following multiple-choice question by giving only one option letter.\n"
        f"Question: {parsed.get('question', '')}\n"
        "Choices:\n" + "\n".join(str(c) for c in choices) + "\n"
    )
    if include_rationale and parsed.get("rationale"):
        text += f"Explanation: {parsed.get('rationale', '')}\n"
    text += "Answer:"
    return text

def score_candidate_empirically(generator: CircuitAwareGenerator, parsed: Mapping[str, Any]) -> Dict[str, Any]:
    if not parsed or "choices" not in parsed:
        return {}
    gold = str(parsed.get("answer", "")).strip().upper()[:1]
    if gold not in {"A", "B", "C", "D"}:
        return {}
    p_no = generator.next_token_option_probs(build_mcqa_eval_prompt(parsed, include_rationale=False))
    p_ra = generator.next_token_option_probs(build_mcqa_eval_prompt(parsed, include_rationale=True))

    def margin(probs: Mapping[str, float]) -> float:
        gold_p = float(probs.get(gold, 0.0))
        other = [float(v) for k, v in probs.items() if k != gold]
        return gold_p - (max(other) if other else 0.0)

    def entropy(probs: Mapping[str, float]) -> float:
        ps = [max(1e-8, float(v)) for v in probs.values()]
        return float(-sum(p * math.log(p) for p in ps))

    pred_no = max(p_no, key=p_no.get)
    pred_ra = max(p_ra, key=p_ra.get)
    return {
        "gold_answer": gold,
        "solver_pred_no_rationale": pred_no,
        "solver_pred_with_rationale": pred_ra,
        "solver_correct_prob_no_rationale": float(p_no.get(gold, 0.0)),
        "solver_correct_prob_with_rationale": float(p_ra.get(gold, 0.0)),
        "solver_margin_no_rationale": float(margin(p_no)),
        "solver_margin_with_rationale": float(margin(p_ra)),
        "solver_entropy_no_rationale": float(entropy(p_no)),
        "solver_entropy_with_rationale": float(entropy(p_ra)),
        "difficulty_empirical": 1.0 - float(p_no.get(gold, 0.0)),
        "rationale_gain": float(p_ra.get(gold, 0.0)) - float(p_no.get(gold, 0.0)),
        "helpfulness_empirical": float(p_ra.get(gold, 0.0)) - float(p_no.get(gold, 0.0)),
    }

def score_candidate_from_recipes(
    generator: CircuitAwareGenerator,
    prompt_text: str,
    completion_text: str,
    target_recipe: RecipeConfig,
    base_recipe: RecipeConfig,
    use_activation_addition: bool,
    use_attention_steering: bool,
) -> Dict[str, float]:
    base_lp = generator.normalized_logprob(
        prompt_text,
        completion_text,
        base_recipe.activation_scales,
        base_recipe.attention_scales,
        use_activation_addition,
        use_attention_steering,
    )
    easy_lp = generator.normalized_logprob(
        prompt_text,
        completion_text,
        DEFAULT_RECIPES["easy"].activation_scales,
        DEFAULT_RECIPES["easy"].attention_scales,
        use_activation_addition,
        use_attention_steering,
    )
    hard_lp = generator.normalized_logprob(
        prompt_text,
        completion_text,
        DEFAULT_RECIPES["hard"].activation_scales,
        DEFAULT_RECIPES["hard"].attention_scales,
        use_activation_addition,
        use_attention_steering,
    )
    helpful_lp = generator.normalized_logprob(
        prompt_text,
        completion_text,
        DEFAULT_RECIPES["helpful"].activation_scales,
        DEFAULT_RECIPES["helpful"].attention_scales,
        use_activation_addition,
        use_attention_steering,
    )
    shortcut_lp = generator.normalized_logprob(
        prompt_text,
        completion_text,
        {**DEFAULT_RECIPES["easy"].activation_scales, "gradalign_low": 0.35, "gradalign_high": -0.10},
        {**DEFAULT_RECIPES["easy"].attention_scales, "gradalign_low": 0.20, "gradalign_high": -0.10},
        use_activation_addition,
        use_attention_steering,
    )

    scores = {
        "base_logprob": base_lp,
        "easy_support": easy_lp - base_lp,
        "hard_support": hard_lp - base_lp,
        "helpful_support": helpful_lp - base_lp,
        "shortcut_support": shortcut_lp - base_lp,
    }
    final = 0.0
    for key, weight in target_recipe.score_weights.items():
        final += weight * scores.get(key, 0.0)
    scores["final_score"] = final
    scores["difficulty_proxy"] = scores["hard_support"] - scores["easy_support"]
    scores["helpfulness_proxy"] = scores["helpful_support"] - max(0.0, scores["shortcut_support"])
    return scores

# -----------------------------------------------------------------------------
# CLI assembly helpers
# -----------------------------------------------------------------------------
MANUAL_SCALE_ARG_NAMES = {
    "aum_high": "aum_high",
    "aum_low": "aum_low",
    "el2n_high": "el2n_high",
    "el2n_low": "el2n_low",
    "gradalign_high": "grad_high",
    "gradalign_low": "grad_low",
}

def overlay_manual_scales(recipe: RecipeConfig, args: argparse.Namespace,) -> RecipeConfig:
    act = dict(recipe.activation_scales)
    attn = dict(recipe.attention_scales)
    for circuit_key, arg_base in MANUAL_SCALE_ARG_NAMES.items():
        act_attr = f"{arg_base}_act"
        attn_attr = f"{arg_base}_attn"
        if hasattr(args, act_attr):
            manual = getattr(args, act_attr)
            if manual is not None:
                act[circuit_key] = float(manual)
        if hasattr(args, attn_attr):
            manual = getattr(args, attn_attr)
            if manual is not None:
                attn[circuit_key] = float(manual)
    return RecipeConfig(
        name=recipe.name,
        activation_scales=act,
        attention_scales=attn,
        score_weights=dict(recipe.score_weights),
        description=recipe.description,
    )


def resolve_experiment_settings(args: argparse.Namespace) -> Dict[str, Any]:
    effective_prompt_mode = args.prompt_mode
    if args.recipe == "naive":
        if effective_prompt_mode != "fixed":
            print("[info] recipe=naive forces prompt_mode=fixed")
        print("[info] recipe=naive disables recipe-conditioned steering and recipe-based selection")
        effective_prompt_mode = "fixed"

    target_recipe = overlay_manual_scales(DEFAULT_RECIPES[args.recipe], args) if args.recipe in DEFAULT_RECIPES else build_neutral_recipe("naive")
    generation_steering_enabled = args.recipe in DEFAULT_RECIPES and args.gen_mode in {"steering", "both"}
    selection_enabled = args.recipe in DEFAULT_RECIPES and args.gen_mode in {"selection", "both"}

    generation_use_activation_addition = generation_steering_enabled and args.use_activation_addition
    generation_use_attention_steering = generation_steering_enabled and args.use_attention_steering
    selection_use_activation_addition = selection_enabled and args.use_activation_addition
    selection_use_attention_steering = selection_enabled and args.use_attention_steering

    if generation_steering_enabled and not (generation_use_activation_addition or generation_use_attention_steering):
        print("[warn] gen_mode includes steering, but both steering mechanisms are disabled. Generation will be prompt-only.")
    if selection_enabled and not (selection_use_activation_addition or selection_use_attention_steering):
        print("[warn] gen_mode includes selection, but both steering mechanisms are disabled. Recipe-based re-scoring will collapse toward neutral.")

    return {
        "effective_prompt_mode": effective_prompt_mode,
        "target_recipe": target_recipe,
        "generation_recipe": target_recipe if generation_steering_enabled else NEUTRAL_RECIPE,
        "selection_recipe": target_recipe if selection_enabled else NEUTRAL_RECIPE,
        "generation_steering_enabled": generation_steering_enabled,
        "selection_enabled": selection_enabled,
        "generation_use_activation_addition": generation_use_activation_addition,
        "generation_use_attention_steering": generation_use_attention_steering,
        "selection_use_activation_addition": selection_use_activation_addition,
        "selection_use_attention_steering": selection_use_attention_steering,
    }

def summarize_recipe(recipe: RecipeConfig) -> None:
    print_rule(f"Recipe: {recipe.name}", "-")
    print(recipe.description)
    print("[activation]")
    for k in CANONICAL_CIRCUIT_KEYS:
        print(f"  - {k}: {recipe.activation_scales.get(k, 0.0):+.3f}")
    print("[attention]")
    for k in CANONICAL_CIRCUIT_KEYS:
        print(f"  - {k}: {recipe.attention_scales.get(k, 0.0):+.3f}")

# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------
def run_generation(args: argparse.Namespace) -> None:
    global _VERBOSE
    _VERBOSE = bool(getattr(args, "verbose", True))
    set_seed(args.seed)
    print_rule("Circuit-aware generation start")
    print_kv("model", args.model_name_or_path)
    print_kv("recipe", args.recipe)
    print_kv("gen_mode", args.gen_mode)
    print_kv("prompt_mode", args.prompt_mode)
    print_kv("dedup", args.dedup)
    print_kv("circuit_target_mode", args.circuit_target_mode)
    print_kv("use_activation_addition", args.use_activation_addition)
    print_kv("use_attention_steering", args.use_attention_steering)

    presets = build_default_presets()
    edge_rows: List[EdgeRow] = []
    if args.edge_table_dir:
        edge_rows = load_circuit_edge_dir(args.edge_table_dir, topk_per_circuit=args.topk_edges)
    elif args.edge_table:
        edge_rows = load_edge_rows(args.edge_table)
    if edge_rows:
        print_kv("loaded_edge_rows", len(edge_rows))
        derived_targets = derive_targets_from_edges(
            edge_rows,
            topk_edges=args.topk_edges,
            max_attn_groups_per_circuit=args.max_attn_groups_per_circuit,
            max_mlp_per_circuit=args.max_mlp_per_circuit,
        )
        presets = merge_presets_with_edge_targets(presets, derived_targets)
    else:
        print("[info] no edge table provided; using built-in default presets only")

    settings = resolve_experiment_settings(args)
    effective_prompt_mode = str(settings["effective_prompt_mode"])
    target_recipe = settings["target_recipe"]
    generation_recipe = settings["generation_recipe"]
    selection_recipe = settings["selection_recipe"]
    generation_steering_enabled = bool(settings["generation_steering_enabled"])
    selection_enabled = bool(settings["selection_enabled"])
    generation_use_activation_addition = bool(settings["generation_use_activation_addition"])
    generation_use_attention_steering = bool(settings["generation_use_attention_steering"])
    selection_use_activation_addition = bool(settings["selection_use_activation_addition"])
    selection_use_attention_steering = bool(settings["selection_use_attention_steering"])
    need_activation_vectors = generation_use_activation_addition or selection_use_activation_addition

    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        dtype=args.dtype,
        device_map=args.device_map,
        # Always use eager attention: FA2 can produce NaN logits in bfloat16 for certain inputs,
        # which crashes multinomial sampling. Attention steering also requires eager mode.
        force_eager_attention=True,
        prefer_single_device=(
            need_activation_vectors
            or generation_use_activation_addition
            or generation_use_attention_steering
            or selection_use_activation_addition
            or selection_use_attention_steering
        ),
    )

    if args.circuit_target_mode == "random" and args.recipe in DEFAULT_RECIPES:
        print("[info] using matched random circuit targets instead of discovered targets")
        presets = build_randomized_presets(presets, model=model, seed=args.seed)
    elif args.circuit_target_mode == "random":
        print("[info] circuit_target_mode=random requested, but the resolved recipe does not use circuit steering/selection")

    if _VERBOSE:
        for key, preset in presets.items():
            print(f"[preset] {key} | role={preset.steering_role} | targets={[t.label() for t in preset.targets]}")

    vector_bank: Dict[str, Dict[str, torch.Tensor]] = {}
    if need_activation_vectors and args.calibration_jsonl:
        calibration_records = load_jsonl(args.calibration_jsonl)
        vector_builder = VectorBankBuilder(
            model=model,
            tokenizer=tokenizer,
            presets=presets,
            quantile=args.calibration_quantile,
            max_texts_per_group=args.max_calibration_per_group,
            batch_size=args.calibration_batch_size,
            max_length=args.calibration_max_length,
        )
        vector_bank = vector_builder.build_from_records(calibration_records)
        print_kv("vector_bank_keys", list(vector_bank.keys()))
    elif need_activation_vectors:
        print("[warn] activation addition enabled but no calibration_jsonl provided. Steering will fall back to attention-only.")
    elif args.use_activation_addition:
        print("[info] activation addition flags are set, but the resolved gen_mode/recipe does not use activation steering.")

    generator = CircuitAwareGenerator(model=model, tokenizer=tokenizer, presets=presets, vector_bank=vector_bank)

    print_rule("Resolved experiment mode", "-")
    print_kv("effective_prompt_mode", effective_prompt_mode)
    print_kv("circuit_target_mode", args.circuit_target_mode)
    print_kv("target_recipe", target_recipe.name)
    print_kv("generation_recipe", generation_recipe.name)
    print_kv("selection_recipe", selection_recipe.name if selection_enabled else "disabled")
    print_kv("generation_steering_enabled", generation_steering_enabled)
    print_kv("selection_enabled", selection_enabled)
    print_kv("generation_use_activation_addition", generation_use_activation_addition)
    print_kv("generation_use_attention_steering", generation_use_attention_steering)
    print_kv("selection_use_activation_addition", selection_use_activation_addition)
    print_kv("selection_use_attention_steering", selection_use_attention_steering)
    summarize_recipe(generation_recipe)
    if selection_enabled and selection_recipe.name != generation_recipe.name:
        summarize_recipe(selection_recipe)

    external_scorer = None
    if selection_enabled and args.external_score_file:
        external_scorer = ExternalCircuitScorer.from_path(args.external_score_file)
    elif args.external_score_file:
        print("[info] external_score_file provided, but selection is disabled. Ignoring external scores.")

    input_records = load_jsonl(args.input_jsonl)
    total_input_records = len(input_records)
    if args.seed_data_num is not None and args.seed_data_num > 0:
        input_records = input_records[: args.seed_data_num]
    print_kv("input_records_total", total_input_records)
    print_kv("input_records_used", len(input_records))
    out_dir = ensure_dir(args.output_dir)
    raw_candidates_path = out_dir / f"{args.recipe}_raw_candidates.jsonl"
    selected_path = out_dir / f"{args.recipe}_selected.jsonl"
    progress_path = out_dir / f"{args.recipe}_progress.jsonl"
    summary_path = out_dir / f"{args.recipe}_summary.json"

    resume_state = prepare_resume_state(raw_candidates_path, selected_path, progress_path)
    completed_record_ids = set(resume_state["completed_record_ids"])
    selected_rows: List[Dict[str, Any]] = list(resume_state["selected_rows"])
    selected_signatures = set(resume_state["selected_signatures"]) if args.dedup == "exact" else set()
    if completed_record_ids:
        print_kv("resume_completed_records", len(completed_record_ids))
        print_kv("resume_selected_rows", len(selected_rows))
    if resume_state["dangling_raw_rows"] or resume_state["dangling_selected_rows"] or resume_state["dangling_progress_rows"]:
        print(
            "[info] cleaned dangling partial outputs before resume: "
            f"raw={resume_state['dangling_raw_rows']} "
            f"selected={resume_state['dangling_selected_rows']} "
            f"progress={resume_state['dangling_progress_rows']}"
        )

    with open(raw_candidates_path, "a", encoding="utf-8") as raw_handle, \
         open(selected_path, "a", encoding="utf-8") as selected_handle, \
         open(progress_path, "a", encoding="utf-8") as progress_handle:
        for row_idx, record in enumerate(input_records):
            record_id = str(record.get("id") or record.get("sample_id") or row_idx)
            if record_id in completed_record_ids:
                if _VERBOSE:
                    print(f"[resume] skipping completed record {record_id}")
                continue

            system_prompt = build_system_prompt(args.recipe, effective_prompt_mode)
            user_prompt = build_user_prompt(record, args.recipe, effective_prompt_mode)
            prompt_text_for_scoring = generator.render_prompt_text(system_prompt, user_prompt)
            if _VERBOSE:
                print_rule(f"Generating for record {record_id}", "-")
                print(user_prompt[:500] + ("..." if len(user_prompt) > 500 else ""))

            candidates_for_record: List[Dict[str, Any]] = []
            for cand_idx in range(args.num_candidates):
                seed = args.seed + row_idx * 1000 + cand_idx
                candidate_id = f"{record_id}_{cand_idx}"
                completion = generator.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    activation_scales=generation_recipe.activation_scales,
                    attention_scales=generation_recipe.attention_scales,
                    use_activation_addition=generation_use_activation_addition,
                    use_attention_steering=generation_use_attention_steering,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                )
                parsed = extract_json_object(completion)
                parsed_question = str(parsed.get("question", "")).strip()
                parsed_choices = parsed.get("choices", []) if isinstance(parsed.get("choices", []), list) else []
                parsed_answer = str(parsed.get("answer", "")).strip()
                parsed_rationale = str(parsed.get("rationale", "")).strip()
                if not parsed:
                    print(f"[warn] candidate {cand_idx} is not parseable JSON")

                # Shuffle choices to remove A-position bias from the generator.
                # The correct answer is re-mapped to its new position label.
                # Priority: full text-match first, then single-char alpha, then single-digit.
                # A full text like "Bacteria" must not be misread as label "B"
                # via the old [:1] shortcut.
                if getattr(args, "shuffle_choices", True) and len(parsed_choices) >= 2:
                    label_space = [chr(ord("A") + i) for i in range(len(parsed_choices))]
                    ans_raw = parsed_answer.strip()
                    ans_up = ans_raw.upper()
                    old_idx: Optional[int] = None

                    normed_answer = ans_raw.lower()
                    for choice_idx, choice in enumerate(parsed_choices):
                        if str(choice).strip().lower() == normed_answer:
                            old_idx = choice_idx
                            break

                    if old_idx is None and len(ans_raw) == 1 and ans_up in label_space:
                        old_idx = label_space.index(ans_up)

                    if old_idx is None and len(ans_raw) == 1 and ans_raw.isdigit():
                        digit_idx = int(ans_raw) - 1
                        if 0 <= digit_idx < len(parsed_choices):
                            old_idx = digit_idx

                    if old_idx is not None:
                        perm = list(range(len(parsed_choices)))
                        random.Random(seed ^ (cand_idx * 7919)).shuffle(perm)
                        parsed_choices = [parsed_choices[i] for i in perm]
                        parsed_answer = label_space[perm.index(old_idx)]
                        parsed = dict(parsed)
                        parsed["choices"] = parsed_choices
                        parsed["answer"] = parsed_answer

                # OOD detection: compute base perplexity (no steering)
                base_nll = generator.normalized_logprob(
                    prompt_text_for_scoring, completion,
                    {k: 0.0 for k in CANONICAL_CIRCUIT_KEYS},
                    {k: 0.0 for k in CANONICAL_CIRCUIT_KEYS},
                    use_activation_addition=False,
                    use_attention_steering=False,
                )
                base_ppl = math.exp(-base_nll) if base_nll > -30 else float("inf")
                if base_ppl > 50.0:
                    print(f"[OOD-WARN] candidate {cand_idx}: base_ppl={base_ppl:.1f} (>50) — may be out-of-distribution")

                internal_scores: Dict[str, float] = {}
                empirical_eval = score_candidate_empirically(generator, parsed) if args.compute_empirical_eval else {}
                external_score = 0.0
                external_components: Dict[str, float] = {}
                final_score: Optional[float] = None
                if selection_enabled:
                    internal_scores = score_candidate_from_recipes(
                        generator=generator,
                        prompt_text=prompt_text_for_scoring,
                        completion_text=completion,
                        target_recipe=selection_recipe,
                        base_recipe=NEUTRAL_RECIPE,
                        use_activation_addition=selection_use_activation_addition,
                        use_attention_steering=selection_use_attention_steering,
                    )
                    if external_scorer is not None:
                        external_components = external_scorer.get(record_id)
                        external_score = external_scorer.aggregate(external_components, args.recipe)
                    final_score = internal_scores["final_score"] + args.external_score_weight * external_score

                candidate_row = {
                    "record_id": record_id,
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "target_band": args.recipe,
                    "source": record,
                    "generation_text": completion,
                    "question": parsed_question,
                    "choices": parsed_choices,
                    "answer": parsed_answer,
                    "rationale": parsed_rationale,
                    "parsed": parsed,
                    "internal_scores": internal_scores,
                    "empirical_eval": empirical_eval,
                    "external_scores": external_components,
                    "external_aggregate": external_score,
                    "final_score": final_score,
                    "steering": {
                        "prompt_mode": effective_prompt_mode,
                        "gen_mode": args.gen_mode,
                        "dedup_mode": args.dedup,
                        "circuit_target_mode": args.circuit_target_mode,
                        "generation_recipe": generation_recipe.name,
                        "selection_recipe": selection_recipe.name if selection_enabled else None,
                        "generation_activation_scales": generation_recipe.activation_scales,
                        "generation_attention_scales": generation_recipe.attention_scales,
                        "selection_score_weights": dict(selection_recipe.score_weights) if selection_enabled else {},
                        "generation_use_activation_addition": generation_use_activation_addition,
                        "generation_use_attention_steering": generation_use_attention_steering,
                        "selection_use_activation_addition": selection_use_activation_addition,
                        "selection_use_attention_steering": selection_use_attention_steering,
                    },
                    "sanity": {
                        "question_len": len(parsed_question.split()),
                        "rationale_len": len(parsed_rationale.split()),
                        "json_parse_ok": bool(parsed),
                        "base_ppl": round(base_ppl, 2),
                        "ood_flag": base_ppl > 50.0,
                    },
                }
                append_jsonl_row(raw_handle, candidate_row)
                candidates_for_record.append(candidate_row)
                if _VERBOSE:
                    if selection_enabled and final_score is not None:
                        print(
                            f"[candidate] {candidate_row['candidate_id']} | final={final_score:+.4f} | ppl={base_ppl:.1f} | "
                            f"easy={internal_scores['easy_support']:+.4f} hard={internal_scores['hard_support']:+.4f} "
                            f"helpful={internal_scores['helpful_support']:+.4f} shortcut={internal_scores['shortcut_support']:+.4f}"
                        )
                    else:
                        print(f"[candidate] {candidate_row['candidate_id']} | selection=off | ppl={base_ppl:.1f}")

            dedup_dropped = 0
            if args.dedup == "exact":
                candidates_for_record, dedup_dropped = dedup_candidates_exact(candidates_for_record, selected_signatures)
                if dedup_dropped and _VERBOSE:
                    print(f"[dedup] exact dropped {dedup_dropped} candidates before selection for record {record_id}")

            selection_policy = "score_topk"
            kept: List[Dict[str, Any]] = []
            if not candidates_for_record:
                selection_policy = "empty_after_dedup"
            elif selection_enabled:
                candidates_for_record.sort(
                    key=lambda x: float(x["final_score"]) if x.get("final_score") is not None else float("-inf"),
                    reverse=True,
                )
                kept = candidates_for_record[: args.keep_per_record]
            else:
                # naive recipe or gen_mode=steering: no contrastive scoring — keep all candidates
                if args.recipe == "naive" or args.keep_per_record >= len(candidates_for_record):
                    selection_policy = "keep_all"
                    kept = list(candidates_for_record)
                else:
                    selection_policy = "random_subset"
                    sampler = random.Random(args.seed + row_idx * 1000 + 7919)
                    chosen = sorted(sampler.sample(range(len(candidates_for_record)), args.keep_per_record))
                    kept = [candidates_for_record[i] for i in chosen]

            for row in kept:
                row["selection_policy"] = selection_policy
                row["dedup_mode"] = args.dedup
                append_jsonl_row(selected_handle, row)
                if args.dedup == "exact":
                    signature = build_candidate_exact_signature(row)
                    if signature is not None:
                        selected_signatures.add(signature)
            selected_rows.extend(kept)

            progress_row = {
                "record_id": record_id,
                "status": "completed",
                "num_candidates_generated": args.num_candidates,
                "num_candidates_after_dedup": len(candidates_for_record),
                "num_selected": len(kept),
                "selection_policy": selection_policy,
                "dedup_mode": args.dedup,
            }
            append_jsonl_row(progress_handle, progress_row)
            completed_record_ids.add(record_id)
            print(
                f"[select] policy={selection_policy} kept {len(kept)} of {len(candidates_for_record)} "
                f"for record {record_id}"
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summarize_selected_rows(selected_rows), f, ensure_ascii=False, indent=2)
    print_rule("Generation completed")
    print_kv("raw_candidates", str(raw_candidates_path))
    print_kv("selected", str(selected_path))
    print_kv("progress", str(progress_path))
    print_kv("summary", str(summary_path))
    if selected_rows:
        ppls = [float(r["sanity"]["base_ppl"]) for r in selected_rows if r["sanity"]["base_ppl"] < 1e6]
        ood_count = sum(1 for r in selected_rows if r["sanity"].get("ood_flag", False))
        json_ok = sum(1 for r in selected_rows if r["sanity"]["json_parse_ok"])
        print_kv("selected_mean_ppl", f"{mean_safe(ppls):.2f}" if ppls else "N/A")
        print_kv("selected_ood_flagged", f"{ood_count}/{len(selected_rows)}")
        print_kv("selected_json_parse_ok", f"{json_ok}/{len(selected_rows)}")
        if selection_enabled:
            finals = [float(r["final_score"]) for r in selected_rows if r.get("final_score") is not None]
            print_kv("selected_mean_final", f"{mean_safe(finals):.4f}")
            print_kv("selected_median_final", f"{median_safe(finals):.4f}")
        else:
            print_kv("selected_mean_final", "N/A (selection disabled)")
            print_kv("selected_median_final", "N/A (selection disabled)")
        if ood_count > len(selected_rows) * 0.3:
            print("[WARN] >30% of selected candidates flagged as OOD. Consider reducing steering scales.")

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Circuit-aware difficulty/helpful data generation")
    p.add_argument("--model-name-or-path", type=str, required=True)
    p.add_argument("--input-jsonl", type=str, required=True, help="Input source data (one JSON object per line)")
    p.add_argument("--seed-data-num", "--seed_data_num", dest="seed_data_num", type=int, default=4000, help="Optionally use only the first N rows from input_jsonl.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--recipe", type=str, choices=RECIPE_CHOICES, default="easy", help="Target recipe. Use 'naive' for a fixed-prompt prompt-only baseline.")
    p.add_argument("--gen-mode", "--gen_mode", dest="gen_mode", type=str, choices=GEN_MODE_CHOICES, default="both", help="How recipe information is used: steering-only generation, selection-only filtering, or both.")
    p.add_argument("--prompt-mode", "--prompt_mode", dest="prompt_mode", type=str, choices=PROMPT_MODE_CHOICES, default="fixed", help="source = current recipe-conditioned prompt; fixed = target-independent shared prompt (default).")
    p.add_argument("--dedup", type=str, choices=DEDUP_CHOICES, default="exact", help="Optional exact dedup before selection using normalized question+choices+answer hashing.")
    p.add_argument("--circuit-target-mode", "--circuit_target_mode", dest="circuit_target_mode", type=str, choices=CIRCUIT_TARGET_MODE_CHOICES, default="discovered", help="discovered = use the loaded/derived circuit targets; random = matched random target baseline with the same recipe scales and target counts.")

    # steering controls
    p.add_argument("--use-activation-addition", action="store_true")
    p.add_argument("--use-attention-steering", action="store_true")
    p.add_argument("--edge-table", type=str, default=None, help="Optional single combined top-edge CSV/JSONL")
    p.add_argument("--edge-table-dir", type=str, default=None, help="Directory with per-circuit CSVs (e.g. aum_high_top_edges.csv, el2n_low_top_edges.csv, ...)")
    p.add_argument("--topk-edges", type=int, default=250)
    p.add_argument("--max-attn-groups-per-circuit", type=int, default=3)
    p.add_argument("--max-mlp-per-circuit", type=int, default=3)
    p.add_argument("--compute-empirical-eval", action="store_true")

    # per-circuit manual scales (None => keep recipe default; 0 => off)
    p.add_argument("--aum-high-act", type=float, default=None)
    p.add_argument("--aum-low-act", type=float, default=None)
    p.add_argument("--el2n-high-act", type=float, default=None)
    p.add_argument("--el2n-low-act", type=float, default=None)
    p.add_argument("--grad-high-act", type=float, default=None)
    p.add_argument("--grad-low-act", type=float, default=None)
    p.add_argument("--aum-high-attn", type=float, default=None)
    p.add_argument("--aum-low-attn", type=float, default=None)
    p.add_argument("--el2n-high-attn", type=float, default=None)
    p.add_argument("--el2n-low-attn", type=float, default=None)
    p.add_argument("--grad-high-attn", type=float, default=None)
    p.add_argument("--grad-low-attn", type=float, default=None)
    # activation calibration
    p.add_argument("--calibration-jsonl", type=str, default=None, help="JSONL with prompt/text + aum/el2n/gradalign scores")
    p.add_argument("--calibration-quantile", type=float, default=0.15)
    p.add_argument("--max-calibration-per-group", type=int, default=256)
    p.add_argument("--calibration-batch-size", type=int, default=4)
    p.add_argument("--calibration-max-length", type=int, default=512)

    # generation params
    p.add_argument("--num-candidates", type=int, default=5)
    p.add_argument("--keep-per-record", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    # scoring
    p.add_argument("--external-score-file", type=str, default=None, help="Optional candidate/source-level external circuit score file")
    p.add_argument("--external-score-weight", type=float, default=0.5)
    # runtime
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--shuffle-choices", "--shuffle_choices", dest="shuffle_choices", action="store_true", default=True,
        help=(
            "Randomly permute answer choices after generation so the correct answer is uniformly "
            "distributed across A/B/C/D. Fixes the ~38%% A-position bias from LLM generation. "
            "Default: on."
        ),
    )
    p.add_argument("--no-shuffle-choices", dest="shuffle_choices", action="store_false")
    p.add_argument(
        "--verbose", dest="verbose", action="store_true", default=True,
        help="Print per-record banners, per-candidate scores, and install-time details. Default: on.",
    )
    p.add_argument(
        "--no-verbose", dest="verbose", action="store_false",
        help="Suppress per-record/per-candidate noise; keep warnings and summaries.",
    )
    return p

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_generation(args)


if __name__ == "__main__":
    main()
