"""Exp 4-b: paired cross-metric causal selectivity matrix.

Each causal row is one persistent-ablation trajectory. AUM and EL2N are
read from that trajectory and compared, by ``hf_id``, with a matched full
trajectory. True GradAlign is read only from an explicit cross-metric
forward/backward overlay; ``gradlogits_l2`` is never accepted as GradAlign.

The analyzer validates that all causal and random runs use the same edge
budget. Missing compute is represented as NaN with an explicit note, or can
be made fatal with ``--require-complete``.
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
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_RESULT_DIR = (
    str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix')
)
METRICS = ("aum", "el2n", "grad_align")
CAUSAL_ROWS = ("aum_high", "el2n_high", "gradalign_high")
ALL_ROWS = CAUSAL_ROWS + ("random_edge_matched",)
DIAGONAL_INDEX = {"aum_high": 0, "el2n_high": 1, "gradalign_high": 2}
PROMPT_SCHEMA_KEYS = (
    "dataset",
    "model_name",
    "seed",
    "fewshot_k",
    "fewshot_hf_ids",
    "num_train_examples",
    "num_val_examples",
    "train_frac",
    "use_chat_template",
    "system_prompt",
    "answer_instruction",
)


@dataclass
class RunData:
    path: Path
    meta: Dict[str, Any]
    items: Dict[str, Dict[str, Any]]


def _as_finite(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=16)
def _read_source_meta(path_text: str) -> Optional[Dict[str, Any]]:
    source = _read_json(Path(path_text))
    source_meta = source.get("meta") if isinstance(source, dict) else None
    return dict(source_meta) if isinstance(source_meta, dict) else None


@lru_cache(maxsize=32)
def _sha256_cached(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> RunData:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = _read_json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        raise ValueError(f"{path}: expected an object with an 'items' list")
    items: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(obj["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item {index} is not an object")
        hf_id = str(item.get("hf_id", "")).strip()
        if not hf_id:
            raise ValueError(f"{path}: item {index} has no hf_id")
        if hf_id in items:
            raise ValueError(f"{path}: duplicate hf_id {hf_id!r}")
        items[hf_id] = item
    if not items:
        raise ValueError(f"{path}: no usable items")
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    return RunData(path=path, meta=dict(meta), items=items)


def exact_score(item: Mapping[str, Any], metric: str, section: str = "scores") -> Optional[float]:
    """Read an exact metric key. Grad-logit norm is intentionally excluded."""
    values = item.get(section)
    if not isinstance(values, dict):
        return None
    keys = (metric, "gradalign") if metric == "grad_align" else (metric,)
    for key in keys:
        value = _as_finite(values.get(key))
        if value is not None:
            return value
    return None


def paired_values(
    full: RunData,
    ablated: RunData,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    ids = sorted(set(full.items) & set(ablated.items))
    full_values: List[float] = []
    ablated_values: List[float] = []
    for hf_id in ids:
        left = exact_score(full.items[hf_id], metric, "scores")
        right = exact_score(ablated.items[hf_id], metric, "scores")
        if left is None or right is None:
            continue
        full_values.append(left)
        ablated_values.append(right)
    note = "matched full-run scores"
    return np.asarray(full_values), np.asarray(ablated_values), note


def paired_gradalign(overlay: RunData) -> Tuple[np.ndarray, np.ndarray, str]:
    baseline: List[float] = []
    ablated: List[float] = []
    for hf_id in sorted(overlay.items):
        item = overlay.items[hf_id]
        # Never fall back to the stored, different-checkpoint q_full/orig score.
        left = exact_score(item, "grad_align", "full_recomputed_scores")
        right = exact_score(item, "grad_align", "scores")
        if left is None or right is None:
            continue
        baseline.append(left)
        ablated.append(right)
    return (
        np.asarray(baseline),
        np.asarray(ablated),
        "same-checkpoint q_full_recomputed vs q_ablated exact GradAlign overlay",
    )


def summarize_paired(full: np.ndarray, ablated: np.ndarray) -> Dict[str, float]:
    if len(full) != len(ablated) or len(full) == 0:
        return {
            "delta": float("nan"),
            "delta_z": float("nan"),
            "delta_pct": float("nan"),
            "n": 0,
            "full_mean": float("nan"),
            "ablated_mean": float("nan"),
            "full_std": float("nan"),
        }
    # Utility columns report a positive drop: full - ablated.
    delta_samples = full - ablated
    delta = float(np.mean(delta_samples))
    full_mean = float(np.mean(full))
    ablated_mean = float(np.mean(ablated))
    full_std = float(np.std(full, ddof=1)) if len(full) > 1 else float("nan")
    return {
        "delta": delta,
        "delta_z": delta / full_std if full_std > 0 else float("nan"),
        "delta_pct": 100.0 * delta / abs(full_mean) if full_mean != 0 else float("nan"),
        "n": int(len(full)),
        "full_mean": full_mean,
        "ablated_mean": ablated_mean,
        "full_std": full_std,
    }


def _nested_dict(obj: Mapping[str, Any], *keys: str) -> Optional[Mapping[str, Any]]:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


def load_eval_delta(path: Optional[Path]) -> Tuple[float, float, str]:
    """Read paired validation Acc/Loss from an Exp4 evaluator artifact."""
    if path is None or not path.is_file():
        return float("nan"), float("nan"), "missing paired validation eval"
    obj = _read_json(path)
    if not isinstance(obj, dict):
        return float("nan"), float("nan"), "invalid paired validation eval"
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else obj
    evaluation = meta.get("evaluation") if isinstance(meta, dict) else None
    if not isinstance(evaluation, dict):
        evaluation = obj.get("evaluation") if isinstance(obj.get("evaluation"), dict) else None
    if not isinstance(evaluation, dict):
        return float("nan"), float("nan"), "no evaluation object"
    baseline = evaluation.get("baseline")
    ablated = evaluation.get("ablated")
    if not isinstance(baseline, dict) or not isinstance(ablated, dict):
        return float("nan"), float("nan"), "evaluation lacks baseline/ablated pair"
    base_acc = _as_finite(baseline.get("accuracy"))
    ab_acc = _as_finite(ablated.get("accuracy"))
    base_loss = _as_finite(baseline.get("loss"))
    ab_loss = _as_finite(ablated.get("loss"))
    # Positive means degradation for both columns: accuracy drop and loss increase.
    delta_acc = base_acc - ab_acc if base_acc is not None and ab_acc is not None else float("nan")
    delta_loss = ab_loss - base_loss if base_loss is not None and ab_loss is not None else float("nan")
    note = "matched full trajectory vs persistent-ablated trajectory validation eval"
    if not math.isfinite(delta_acc) or not math.isfinite(delta_loss):
        note = "incomplete paired validation eval"
    return delta_acc, delta_loss, note


def _meta_value(run: RunData, key: str) -> Any:
    return run.meta.get(key)


def input_prompt_schema(run: RunData) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    source_text = str(
        run.meta.get("input_all_scores_json") or run.meta.get("input_json") or ""
    ).strip()
    if not source_text:
        return None, "input score JSON path is missing"
    source_path = Path(source_text).expanduser()
    if not source_path.is_file():
        return None, f"input score JSON is not readable: {source_path}"
    source_meta = _read_source_meta(str(source_path.resolve()))
    if not isinstance(source_meta, dict):
        return None, f"input JSON has no metadata object: {source_path}"
    return {key: source_meta.get(key) for key in PROMPT_SCHEMA_KEYS}, None


@lru_cache(maxsize=16)
def _source_prompt_gold_digest(path_text: str) -> Tuple[str, int]:
    source = _read_json(Path(path_text))
    items = source.get("items") if isinstance(source, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"{path_text}: expected an items list for prompt/gold digest")
    records: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path_text}: item {index} is not an object")
        hf_id = str(item.get("hf_id", "")).strip()
        if not hf_id or hf_id in records:
            raise ValueError(f"{path_text}: missing/duplicate hf_id at item {index}")
        records[hf_id] = {
            "hf_id": hf_id,
            "prompt": item.get("prompt"),
            "gold_token_id": item.get("gold_token_id"),
            "gold_label": item.get("gold_label"),
            "label_space": item.get("label_space"),
            "label_token_ids": item.get("label_token_ids"),
        }
    digest = hashlib.sha256()
    for hf_id in sorted(records):
        digest.update(
            json.dumps(
                records[hf_id], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest(), len(records)


def input_prompt_gold_digest(run: RunData) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    source_text = str(
        run.meta.get("input_all_scores_json") or run.meta.get("input_json") or ""
    ).strip()
    if not source_text:
        return None, None, "input score JSON path is missing"
    source_path = Path(source_text).expanduser()
    if not source_path.is_file():
        return None, None, f"input score JSON is not readable: {source_path}"
    try:
        digest, count = _source_prompt_gold_digest(str(source_path.resolve()))
    except (OSError, ValueError) as exc:
        return None, None, str(exc)
    return digest, count, None


def validate_run(
    run: RunData,
    *,
    row_name: str,
    expected_topk: int,
    expected_samples: int,
    expected_hf_ids: Iterable[str],
    expected_score_key: Optional[str],
) -> List[str]:
    errors: List[str] = []
    if expected_score_key and str(_meta_value(run, "score_key")) != expected_score_key:
        errors.append(
            f"score_key={_meta_value(run, 'score_key')!r}, expected {expected_score_key!r}"
        )
    if row_name in CAUSAL_ROWS:
        if _meta_value(run, "ablation_enabled") is not True:
            errors.append("ablation_enabled is not true")
        if str(_meta_value(run, "ablation_edge_mode")) != "scored":
            errors.append("ablation_edge_mode is not 'scored'")
        if str(_meta_value(run, "circuit_bucket")) != "high":
            errors.append("circuit_bucket is not 'high'")
        if str(_meta_value(run, "corrupt_type")) != "gaussian":
            errors.append("corrupt_type is not 'gaussian'")
    elif row_name == "random_edge_matched":
        if _meta_value(run, "ablation_enabled") is not True:
            errors.append("random control ablation_enabled is not true")
        if str(_meta_value(run, "ablation_edge_mode")) != "random":
            errors.append("random control does not declare ablation_edge_mode='random'")
        if str(_meta_value(run, "random_edge_pool")) != "model":
            errors.append("random control does not use the declared model-edge pool")
    topk = _meta_value(run, "topk_edges")
    try:
        topk_int = int(topk)
    except (TypeError, ValueError):
        topk_int = -1
    if topk_int != expected_topk:
        errors.append(f"topk_edges={topk!r}, expected {expected_topk}")
    edge_meta = run.meta.get("edge_meta")
    mapped = edge_meta.get("mapped_edges_in_runtime") if isinstance(edge_meta, dict) else None
    try:
        mapped_int = int(mapped) if mapped is not None else -1
    except (TypeError, ValueError):
        mapped_int = -1
    if mapped_int != expected_topk:
        errors.append(f"mapped_edges_in_runtime={mapped!r}, expected {expected_topk}")
    expected_ids = set(expected_hf_ids)
    if len(run.items) != expected_samples or set(run.items) != expected_ids:
        errors.append(
            f"hf_id coverage is incomplete/mismatched: run={len(run.items)}, "
            f"expected={expected_samples}, common={len(set(run.items) & expected_ids)}"
        )
    return errors


def validate_config_match(full: RunData, ablated: RunData) -> List[str]:
    differences: List[str] = []
    for key in (
        "dataset", "model_name", "seed", "epochs", "batch_size", "lr",
        "weight_decay", "grad_accum", "el2n_epochs", "max_samples", "max_length",
        "dtype", "resume_state_dict", "resume_state_dict_sha256",
        "circuit_bucket", "corrupt_type", "topk_rank_abs",
    ):
        left, right = full.meta.get(key), ablated.meta.get(key)
        if left != right:
            differences.append(f"{key}: full={left!r}, ablated={right!r}")
    content_mismatches = 0
    for hf_id in set(full.items) & set(ablated.items):
        left_item, right_item = full.items[hf_id], ablated.items[hf_id]
        for key in (
            "task", "question_text", "choices", "prompt", "fewshot_k",
            "gold_token_id", "gold_label", "label_space",
        ):
            if left_item.get(key) != right_item.get(key):
                content_mismatches += 1
                break
    if content_mismatches:
        differences.append(
            f"sample content mismatch for {content_mismatches} shared hf_ids "
            "(prompt/gold/label-space)"
        )
    full_schema, full_schema_error = input_prompt_schema(full)
    ablated_schema, ablated_schema_error = input_prompt_schema(ablated)
    if full_schema_error:
        differences.append(f"full input prompt schema: {full_schema_error}")
    if ablated_schema_error:
        differences.append(f"ablated input prompt schema: {ablated_schema_error}")
    if full_schema is not None and ablated_schema is not None:
        for key in PROMPT_SCHEMA_KEYS:
            if full_schema.get(key) != ablated_schema.get(key):
                differences.append(
                    f"input prompt schema {key}: full={full_schema.get(key)!r}, "
                    f"ablated={ablated_schema.get(key)!r}"
                )
    return differences


def validate_overlay(
    overlay: RunData,
    run: RunData,
    full: RunData,
    *,
    row_name: str,
    expected_topk: int,
) -> List[str]:
    errors: List[str] = []
    expected_circuit = {
        "aum_high": "aum",
        "el2n_high": "el2n",
        "gradalign_high": "grad_align",
        "random_edge_matched": "random_model",
    }[row_name]
    if str(overlay.meta.get("metric_score_key")) != "grad_align":
        errors.append("overlay metric_score_key is not exact grad_align")
    if str(overlay.meta.get("full_score_source")) != "same-checkpoint no-hook q_full_recomputed":
        errors.append("overlay does not certify same-checkpoint q_full_recomputed")
    if str(overlay.meta.get("ablated_score_source")) != "same-checkpoint persistent zero-ablation q_ablated":
        errors.append("overlay does not certify same-checkpoint persistent q_ablated")
    if str(overlay.meta.get("reference_vector_source")) != "full":
        errors.append("overlay reference_vector_source is not the fixed full reference")
    expected_primary = (
        "matched full-trajectory final no-hook vs persistent-ablated "
        "trajectory final with hook"
    )
    if str(overlay.meta.get("evaluation_primary")) != expected_primary:
        errors.append("overlay does not certify the matched-trajectory primary evaluation")
    if str(overlay.meta.get("fingerprint_algorithm")) != "sha256":
        errors.append("overlay does not certify SHA256 input/artifact fingerprints")
    for path_key, hash_key in (
        ("input_json", "input_json_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("full_checkpoint", "full_checkpoint_sha256"),
        ("circuit_pt", "circuit_pt_sha256"),
    ):
        path_text = str(overlay.meta.get(path_key, ""))
        stored_hash = str(overlay.meta.get(hash_key, ""))
        if path_key == "circuit_pt" and expected_circuit == "random_model":
            if path_text or stored_hash:
                errors.append("random-model overlay unexpectedly records a circuit file")
            continue
        path_obj = Path(path_text).expanduser() if path_text else None
        if path_obj is None or not path_obj.is_file():
            errors.append(f"overlay fingerprint target is missing: {path_key}={path_text!r}")
            continue
        actual_hash = _sha256_cached(str(path_obj.resolve()))
        if not stored_hash or stored_hash != actual_hash:
            errors.append(f"overlay {hash_key} does not match the current {path_key}")
    if str(overlay.meta.get("circuit_score_key")) != expected_circuit:
        errors.append(
            f"overlay circuit_score_key={overlay.meta.get('circuit_score_key')!r}, "
            f"expected {expected_circuit!r}"
        )
    for key in ("model_name", "dtype", "seed", "max_length"):
        if overlay.meta.get(key) != run.meta.get(key):
            errors.append(
                f"overlay {key}={overlay.meta.get(key)!r}, "
                f"trajectory {key}={run.meta.get(key)!r}"
            )
    if row_name == "random_edge_matched" and (
        overlay.meta.get("random_edges_seed") != run.meta.get("random_edges_seed")
    ):
        errors.append("overlay random_edges_seed differs from the persistent-ablation run")
    run_schema, run_schema_error = input_prompt_schema(run)
    overlay_schema, overlay_schema_error = input_prompt_schema(overlay)
    if run_schema_error:
        errors.append(f"trajectory input prompt schema: {run_schema_error}")
    if overlay_schema_error:
        errors.append(f"overlay input prompt schema: {overlay_schema_error}")
    if run_schema is not None and overlay_schema is not None and run_schema != overlay_schema:
        errors.append("overlay input prompt schema differs from the persistent-ablation run")
    run_digest, run_count, run_digest_error = input_prompt_gold_digest(run)
    overlay_digest, overlay_count, overlay_digest_error = input_prompt_gold_digest(overlay)
    if run_digest_error:
        errors.append(f"trajectory prompt/gold digest: {run_digest_error}")
    if overlay_digest_error:
        errors.append(f"overlay prompt/gold digest: {overlay_digest_error}")
    if (
        run_digest is not None
        and overlay_digest is not None
        and (run_digest != overlay_digest or run_count != overlay_count)
    ):
        errors.append(
            "overlay per-hf prompt/gold content differs from the persistent-ablation input "
            f"(trajectory_n={run_count}, overlay_n={overlay_count})"
        )
    try:
        overlay_topk = int(overlay.meta.get("topk_edges", -1))
    except (TypeError, ValueError):
        overlay_topk = -1
    if overlay_topk != expected_topk:
        errors.append(
            f"overlay topk_edges={overlay.meta.get('topk_edges')!r}, expected {expected_topk}"
        )
    edge_meta = overlay.meta.get("edge_meta")
    mapped = edge_meta.get("mapped_edges_in_runtime") if isinstance(edge_meta, dict) else None
    try:
        overlay_mapped = int(mapped if mapped is not None else -1)
    except (TypeError, ValueError):
        overlay_mapped = -1
    if overlay_mapped != expected_topk:
        errors.append(f"overlay mapped_edges_in_runtime={mapped!r}, expected {expected_topk}")
    run_edge_meta = run.meta.get("edge_meta")
    if isinstance(run_edge_meta, dict) and isinstance(edge_meta, dict):
        def top_names(meta: Mapping[str, Any]) -> List[str]:
            rows = meta.get("top10_selected_edges")
            if not isinstance(rows, list):
                return []
            return [str(row.get("edge", "")) for row in rows if isinstance(row, dict)]

        run_top, overlay_top = top_names(run_edge_meta), top_names(edge_meta)
        if run_top and overlay_top and run_top != overlay_top:
            errors.append("overlay selected edges do not match the persistent-ablation run")
        run_circuit = str(run_edge_meta.get("circuit_pt", run.meta.get("circuit_pt", "")))
        overlay_circuit = str(edge_meta.get("circuit_pt", ""))
        if run_circuit and overlay_circuit and Path(run_circuit).resolve() != Path(overlay_circuit).resolve():
            errors.append("overlay circuit_pt differs from the persistent-ablation run")
    expected_checkpoint = (run.path.parent / "checkpoints" / "final.pt").resolve()
    checkpoint_text = str(overlay.meta.get("checkpoint", ""))
    if not checkpoint_text or Path(checkpoint_text).resolve() != expected_checkpoint:
        errors.append(
            f"overlay checkpoint={checkpoint_text!r}, expected {str(expected_checkpoint)!r}"
        )
    expected_full_checkpoint = (full.path.parent / "checkpoints" / "final.pt").resolve()
    full_checkpoint_text = str(overlay.meta.get("full_checkpoint", ""))
    if (
        not full_checkpoint_text
        or Path(full_checkpoint_text).resolve() != expected_full_checkpoint
    ):
        errors.append(
            f"overlay full_checkpoint={full_checkpoint_text!r}, "
            f"expected {str(expected_full_checkpoint)!r}"
        )
    if checkpoint_text and full_checkpoint_text and (
        Path(checkpoint_text).resolve() == Path(full_checkpoint_text).resolve()
    ):
        errors.append("overlay ablated and full checkpoints are the same file")
    expected_ids = set(run.items)
    if set(overlay.items) != expected_ids:
        errors.append(
            f"overlay hf_id set differs from trajectory: overlay={len(overlay.items)}, "
            f"trajectory={len(expected_ids)}, common={len(set(overlay.items) & expected_ids)}"
        )
    for key in ("max_items", "max_val_items", "max_reference_items"):
        if overlay.meta.get(key) != 0:
            errors.append(f"overlay {key}={overlay.meta.get(key)!r}, expected full-data value 0")
    if overlay.meta.get("n_scored") != len(expected_ids):
        errors.append(
            f"overlay n_scored={overlay.meta.get('n_scored')!r}, expected {len(expected_ids)}"
        )
    evaluation = overlay.meta.get("evaluation")
    baseline_eval = evaluation.get("baseline") if isinstance(evaluation, dict) else None
    ablated_eval = evaluation.get("ablated") if isinstance(evaluation, dict) else None
    if not isinstance(baseline_eval, dict) or not isinstance(ablated_eval, dict):
        errors.append("overlay lacks paired baseline/ablated validation evaluation")
    else:
        try:
            baseline_n = int(baseline_eval.get("n", -1))
            ablated_n = int(ablated_eval.get("n", -1))
            reference_n = int(overlay.meta.get("n_reference", -1))
        except (TypeError, ValueError):
            baseline_n = ablated_n = reference_n = -1
        if baseline_n <= 0 or baseline_n != ablated_n or baseline_n != reference_n:
            errors.append(
                "overlay validation/reference coverage mismatch: "
                f"baseline={baseline_n}, ablated={ablated_n}, reference={reference_n}"
            )
        acute_eval = evaluation.get("acute_no_hook_same_checkpoint")
        try:
            acute_n = int(acute_eval.get("n", -1)) if isinstance(acute_eval, dict) else -1
        except (TypeError, ValueError):
            acute_n = -1
        if acute_n != baseline_n:
            errors.append(
                f"overlay acute diagnostic coverage={acute_n}, expected {baseline_n}"
            )
    return errors


def selectivity_index(values: Sequence[float], diagonal: int) -> float:
    """Literal signed diagonal / arithmetic mean(signed off-diagonals).

    A zero off-diagonal mean makes the ratio undefined and is reported as NaN;
    signs are never discarded, so negative ratios remain visible.
    """
    if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
        return float("nan")
    diag = float(values[diagonal])
    off = [float(value) for i, value in enumerate(values) if i != diagonal]
    denom = float(np.mean(off))
    if denom == 0.0:
        return float("nan")
    return diag / denom


def heatmap(
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    path: Path,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib unavailable: {exc}")
        return
    values = np.asarray(matrix, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(1e-6, float(finite.max())) if finite.size else 1.0
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.imshow(masked, cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    for (i, j), value in np.ndenumerate(values):
        text = "NA" if not math.isfinite(float(value)) else f"{value:+.2f}"
        ax.text(j, i, text, ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[png] {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-json", required=True, help="Matched no-ablation trajectory")
    parser.add_argument("--aum-json", default="")
    parser.add_argument("--el2n-json", default="")
    parser.add_argument("--ga-json", default="")
    parser.add_argument("--random-json", default="")
    parser.add_argument("--aum-ga-overlay", default="")
    parser.add_argument("--el2n-ga-overlay", default="")
    parser.add_argument("--ga-ga-overlay", default="")
    parser.add_argument("--random-ga-overlay", default="")
    parser.add_argument("--expected-topk", type=int, default=250)
    parser.add_argument("--expected-samples", type=int, default=11679)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    # Resolve through symlinks: the default result dir is a link into /DATA, so
    # staging beside the unresolved path lands on / and the atomic replace below
    # fails with EXDEV instead of publishing the table.
    result_dir = Path(args.result_dir).resolve()
    full = load_run(Path(args.full_json))
    if full.meta.get("ablation_enabled") is not False:
        raise ValueError(f"{full.path}: full baseline does not declare ablation_enabled=false")
    full_coverage_valid = len(full.items) == args.expected_samples
    if not full_coverage_valid and args.require_complete:
        raise ValueError(
            f"full baseline hf_id coverage: got={len(full.items)}, expected={args.expected_samples}"
        )

    run_paths = {
        "aum_high": Path(args.aum_json) if args.aum_json else None,
        "el2n_high": Path(args.el2n_json) if args.el2n_json else None,
        "gradalign_high": Path(args.ga_json) if args.ga_json else None,
        "random_edge_matched": Path(args.random_json) if args.random_json else None,
    }
    overlay_paths = {
        "aum_high": Path(args.aum_ga_overlay) if args.aum_ga_overlay else None,
        "el2n_high": Path(args.el2n_ga_overlay) if args.el2n_ga_overlay else None,
        "gradalign_high": Path(args.ga_ga_overlay) if args.ga_ga_overlay else None,
        "random_edge_matched": Path(args.random_ga_overlay) if args.random_ga_overlay else None,
    }
    expected_keys = {
        "aum_high": "aum", "el2n_high": "el2n", "gradalign_high": "grad_align",
        "random_edge_matched": "aum",
    }

    rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    incomplete: List[str] = []
    for row_name in ALL_ROWS:
        path = run_paths[row_name]
        row: Dict[str, Any] = {"row": row_name}
        notes: List[str] = []
        counts: Dict[str, int] = {}
        summaries: Dict[str, Dict[str, float]] = {}
        run: Optional[RunData] = None
        if path is None or not path.is_file():
            notes.append("missing persistent-ablation run")
        else:
            run = load_run(path)
            errors = validate_run(
                run, row_name=row_name, expected_topk=args.expected_topk,
                expected_samples=args.expected_samples,
                expected_hf_ids=full.items,
                expected_score_key=expected_keys[row_name],
            )
            if not full_coverage_valid:
                errors.append(
                    f"full baseline coverage={len(full.items)}, expected={args.expected_samples}"
                )
            errors.extend(validate_config_match(full, run))
            if errors:
                notes.extend(f"INVALID: {error}" for error in errors)
                run = None

        for metric in ("aum", "el2n"):
            if run is None:
                summary = summarize_paired(np.array([]), np.array([]))
            else:
                baseline, ablated, source = paired_values(full, run, metric)
                summary = summarize_paired(baseline, ablated)
                if summary["n"] == 0:
                    notes.append(f"missing exact {metric} scores")
                else:
                    notes.append(f"{metric}: {source}")
            summaries[metric] = summary
            counts[metric] = int(summary["n"])

        overlay_path = overlay_paths[row_name]
        overlay: Optional[RunData] = None
        valid_overlay_for_eval: Optional[Path] = None
        if run is not None and overlay_path is not None and overlay_path.is_file():
            overlay = load_run(overlay_path)
            overlay_errors = validate_overlay(
                overlay, run, full, row_name=row_name, expected_topk=args.expected_topk
            )
            if overlay_errors:
                notes.extend(f"INVALID overlay: {error}" for error in overlay_errors)
                ga_summary = summarize_paired(np.array([]), np.array([]))
            else:
                valid_overlay_for_eval = overlay_path
                baseline, ablated, source = paired_gradalign(overlay)
                ga_summary = summarize_paired(baseline, ablated)
                if ga_summary["n"]:
                    notes.append(f"grad_align: {source}")
                else:
                    notes.append(
                        "overlay lacks exact scores.grad_align and "
                        "full_recomputed_scores.grad_align pairs"
                    )
        else:
            ga_summary = summarize_paired(np.array([]), np.array([]))
            notes.append("missing true-GradAlign overlay (gradlogits_l2 is not used)")
        summaries["grad_align"] = ga_summary
        counts["grad_align"] = int(ga_summary["n"])

        delta_acc, delta_loss, eval_note = load_eval_delta(valid_overlay_for_eval)
        notes.append(eval_note)
        for metric in METRICS:
            summary = summaries[metric]
            row[f"delta_{metric}"] = summary["delta"]
            row[f"delta_{metric}_z"] = summary["delta_z"]
            row[f"delta_{metric}_pct"] = summary["delta_pct"]
            row[f"n_{metric}"] = int(summary["n"])
        row["delta_acc"] = delta_acc
        row["delta_loss"] = delta_loss
        if row_name in DIAGONAL_INDEX:
            row["selectivity_index"] = selectivity_index(
                [row["delta_aum_z"], row["delta_el2n_z"], row["delta_grad_align_z"]],
                DIAGONAL_INDEX[row_name],
            )
        else:
            row["selectivity_index"] = float("nan")
        row["note"] = "; ".join(dict.fromkeys(notes))
        required = [row[f"delta_{metric}_z"] for metric in METRICS]
        if row_name in CAUSAL_ROWS and (
            not all(math.isfinite(float(value)) for value in required)
            or not all(row[f"n_{metric}"] == args.expected_samples for metric in METRICS)
            or not math.isfinite(float(delta_acc))
            or not math.isfinite(float(delta_loss))
        ):
            incomplete.append(row_name)
        diagnostics[row_name] = {
            "run_json": str(path) if path else "",
            "gradalign_overlay": str(overlay_path) if overlay_path else "",
            "counts": counts,
            "notes": notes,
        }
        rows.append(row)

    fields = [
        "row",
        "delta_aum_full_minus_ablated",
        "delta_el2n_full_minus_ablated",
        "delta_grad_align_full_minus_ablated",
        "delta_aum_z_full_minus_ablated",
        "delta_el2n_z_full_minus_ablated",
        "delta_grad_align_z_full_minus_ablated",
        "delta_aum_pct_full_minus_ablated",
        "delta_el2n_pct_full_minus_ablated",
        "delta_grad_align_pct_full_minus_ablated",
        "n_aum", "n_el2n", "n_grad_align",
        "delta_acc_full_minus_ablated",
        "delta_loss_ablated_minus_full",
        "selectivity_index_signed_z", "note",
    ]
    export_rows: List[Dict[str, Any]] = []
    for row in rows:
        exported = {
            "row": row["row"],
            "n_aum": row["n_aum"],
            "n_el2n": row["n_el2n"],
            "n_grad_align": row["n_grad_align"],
            "note": row["note"],
            "delta_acc_full_minus_ablated": row["delta_acc"],
            "delta_loss_ablated_minus_full": row["delta_loss"],
            "selectivity_index_signed_z": row["selectivity_index"],
        }
        for metric in METRICS:
            exported[f"delta_{metric}_full_minus_ablated"] = row[f"delta_{metric}"]
            exported[f"delta_{metric}_z_full_minus_ablated"] = row[f"delta_{metric}_z"]
            exported[f"delta_{metric}_pct_full_minus_ablated"] = row[f"delta_{metric}_pct"]
        export_rows.append(exported)
    causal = [row for row in rows if row["row"] in CAUSAL_ROWS]
    matrix = np.asarray(
        [
            [
                row["delta_aum_z"], row["delta_el2n_z"], row["delta_grad_align_z"],
                row["delta_acc"], row["delta_loss"],
            ]
            for row in causal
        ],
        dtype=float,
    )
    metadata = {
        "full_json": str(full.path),
        "expected_topk": int(args.expected_topk),
        "expected_samples": int(args.expected_samples),
        "full_hf_ids": len(full.items),
        "definitions": {
            "delta": "utility drop: mean(full - ablated), paired by hf_id",
            "z": "delta / sample std of the matched full metric",
            "grad_align": (
                "exact gradient dot validation-gradient; paired same-checkpoint "
                "q_full_recomputed vs q_ablated; never stored q_full or gradlogits_l2"
            ),
            "accuracy_loss": (
                "positive degradation convention: delta_acc=full-ablated; "
                "delta_loss=ablated-full; matched full-trajectory final no-hook "
                "vs persistent-ablated trajectory final with hook. Same-ablated-"
                "checkpoint no-hook is stored only as an acute diagnostic"
            ),
            "selectivity_index": (
                "literal signed diagonal z / arithmetic mean of the two signed "
                "off-diagonal z values; zero off-diagonal mean is undefined (NaN)"
            ),
            "random": "edge-count-matched random ablation, not degradation-matched",
        },
        "diagnostics": diagnostics,
        "incomplete_causal_rows": incomplete,
    }
    if incomplete and args.require_complete:
        raise RuntimeError(f"incomplete causal rows: {', '.join(incomplete)}")

    result_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{result_dir.name}.stage-", dir=result_dir.parent
    ) as temporary:
        stage = Path(temporary)
        with (stage / "selectivity_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(export_rows)
        heatmap(
            matrix,
            ["delta AUM (z)", "delta EL2N (z)", "delta GA (z)", "delta Acc", "delta Loss"],
            [row["row"] for row in causal],
            stage / "selectivity_3x5.png",
            "Cross-metric causal selectivity (paired by hf_id)",
        )
        with (stage / "selectivity_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        result_dir.mkdir(parents=True, exist_ok=True)
        for staged in stage.iterdir():
            staged.replace(result_dir / staged.name)
    print(f"[csv] {result_dir / 'selectivity_matrix.csv'}")
    if incomplete:
        print(
            f"[WARN] incomplete causal rows: {', '.join(incomplete)}; "
            "NaNs are preserved and labelled NA"
        )


if __name__ == "__main__":
    main()
