import csv
import gc
import hashlib
import json
import re
import zlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
_EAPIG_ROOT = _ROOT / "EAP-IG"
_EAP_SRC = _EAPIG_ROOT / "src"

import sys
for _p in [str(_ROOT), str(_EAPIG_ROOT), str(_EAP_SRC), str(_CUR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gens.mcqa import format_mcqa_prompt
from scen1_eapig import (  # type: ignore
    PairsDatasetRuntimeCorrupt,
    collate_pairs,
    load_model,
    logit_diff_mc_tensor,
    str_to_dtype,
)
from eap.attribute import attribute  # type: ignore
from eap.graph import Graph  # type: ignore

CIRCUIT_ROOT_HINTS = ["train_dynamic", "data_attr"]
BUCKET_ROOT_HINTS = ["train_dynamic", "data_attr"]
RUN_RE = re.compile(r"_m(?P<m>\d+)_top(?P<top>\d+)_")

@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str
    ig_steps: int
    incorrect_agg: str

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(obj: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _stable_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    if "/" in n:
        return n
    if n.startswith("pythia-"):
        return f"EleutherAI/{n}"
    return n

def _score_pt_candidate(path: Path) -> Tuple[int, int, float]:
    base = path.name
    m = RUN_RE.search(base)
    if m:
        ig_steps = int(m.group("m"))
        topn = int(m.group("top"))
    else:
        ig_steps = -1
        topn = -1
    return (topn, ig_steps, path.stat().st_mtime)

def _expand_roots(base_root: Path, hints: Sequence[str]) -> List[Path]:
    roots: List[Path] = []

    def _add(p: Path) -> None:
        if p.is_dir() and p not in roots:
            roots.append(p)

    if base_root.name in set(hints):
        for n in hints:
            _add(base_root.parent / n)
        return roots if roots else [base_root]

    has_hint_child = False
    for n in hints:
        p = base_root / n
        if p.is_dir():
            has_hint_child = True
            _add(p)
    if not has_hint_child:
        _add(base_root)

    return roots if roots else [base_root]

def _iter_matching_files(roots: Sequence[Path], pattern: str) -> List[Path]:
    out: List[Path] = []
    seen: Set[Path] = set()
    for r in roots:
        if not r.is_dir():
            continue
        for p in sorted(r.rglob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out

def _find_latest_circuit_pt(circuits_root: Path, dataset: str, score_key: str, bucket: str, corrupt_type: str,) -> Optional[Path]:
    roots = _expand_roots(circuits_root, CIRCUIT_ROOT_HINTS)
    pts: List[Path] = []
    for root in roots:
        d = root / dataset / score_key / bucket / corrupt_type
        if not d.is_dir():
            continue
        cand = list(d.glob("eapig_*.pt"))
        if not cand:
            cand = list(d.glob("*.pt"))
        pts.extend(cand)
    if not pts:
        return None
    pts = sorted(set(pts), key=_score_pt_candidate, reverse=True)
    return pts[0]

def _find_all_scores_json(buckets_root: Path, dataset: str, score_key: str) -> Optional[Path]:
    roots = _expand_roots(buckets_root, BUCKET_ROOT_HINTS)
    specific_name = f"all_{score_key}_{dataset}.json"
    generic_name = f"all_scores_{dataset}.json"

    for root in roots:
        c = root / "all" / dataset / specific_name
        if c.is_file():
            return c
    found_specific = _iter_matching_files(roots, specific_name)
    if found_specific:
        return found_specific[0]

    for root in roots:
        c = root / "all" / dataset / generic_name
        if c.is_file():
            return c
    found_generic = _iter_matching_files(roots, generic_name)
    if found_generic:
        return found_generic[0]
    return None

def _load_original_score_map(all_scores_json: Path, score_key: str) -> Dict[str, float]:
    obj = _read_json(all_scores_json)
    rows = obj.get("items", obj if isinstance(obj, list) else [])
    if not isinstance(rows, list):
        return {}
    out: Dict[str, float] = {}
    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if not hf_id:
            continue
        score_val: Any = None
        scores = r.get("scores")
        if isinstance(scores, dict):
            score_val = scores.get(score_key)
        if score_val is None:
            score_val = r.get(score_key)
        if score_val is None:
            score_val = r.get("true_score")
        score_f = _safe_float(score_val)
        if np.isfinite(score_f):
            out[hf_id] = float(score_f)
    return out

def _pick_from_run_meta(paths: Sequence[Optional[Path]], key: str) -> Optional[Any]:
    vals: List[Any] = []
    for p in paths:
        if p is None or not p.is_file():
            continue
        try:
            v = _read_json(p).get(key)
        except Exception:
            continue
        if v is None:
            continue
        vals.append(v)
    if not vals:
        return None
    return vals[0]

def _resolve_runtime_config(high_run_json: Optional[Path], low_run_json: Optional[Path], model_name_override: str, ig_steps_override: int, incorrect_agg_override: str,) -> RuntimeConfig:
    run_paths = [high_run_json, low_run_json]
    if ig_steps_override > 0:
        ig_steps = int(ig_steps_override)
    else:
        ig_steps = int(_pick_from_run_meta(run_paths, "ig_steps") or 5)

    if incorrect_agg_override != "auto":
        incorrect_agg = str(incorrect_agg_override)
    else:
        incorrect_agg = str(_pick_from_run_meta(run_paths, "incorrect_agg") or "max")
        if incorrect_agg not in {"max", "mean", "min"}:
            incorrect_agg = "max"

    model_name_raw = (model_name_override or "").strip()
    if model_name_raw:
        model_name = _normalize_model_name(model_name_raw)
    else:
        model_name = _normalize_model_name(str(_pick_from_run_meta(run_paths, "model_name") or ""))
        if not model_name:
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    return RuntimeConfig(model_name=model_name, ig_steps=ig_steps, incorrect_agg=incorrect_agg)

def _dtype_from_args(device: str, dtype_str: str) -> torch.dtype:
    ds = (dtype_str or "").strip().lower()
    if not ds:
        return torch.float16 if device.startswith("cuda") else torch.float32
    dt = str_to_dtype(ds)
    if (not device.startswith("cuda")) and dt in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dt

def select_edge_indices_from_circuit(circuit_pt: Path, topk_edges: int, rank_abs: bool,) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[float], int]:
    g = Graph.from_pt(str(circuit_pt))
    edges = list(g.edges.values())
    total_edges = len(edges)
    if total_edges == 0:
        raise RuntimeError(f"No edges found in graph: {circuit_pt}")

    if topk_edges <= 0:
        selected = [e for e in edges if bool(e.in_graph)]
        if not selected:
            selected = edges
    else:
        k = min(int(topk_edges), total_edges)
        selected = sorted(
            edges,
            key=lambda e: abs(float(e.score)) if rank_abs else float(e.score),
            reverse=True,
        )[:k]

    selected = sorted(
        selected,
        key=lambda e: abs(float(e.score)) if rank_abs else float(e.score),
        reverse=True,
    )
    row_idx = torch.tensor([int(e.matrix_index[0]) for e in selected], dtype=torch.long)
    col_idx = torch.tensor([int(e.matrix_index[1]) for e in selected], dtype=torch.long)
    edge_names = [str(e.name) for e in selected]
    edge_scores = [float(e.score) for e in selected]
    return row_idx, col_idx, edge_names, edge_scores, total_edges

def remap_edge_indices_by_name(edge_names: Sequence[str], runtime_graph: Graph,) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    rows: List[int] = []
    cols: List[int] = []
    missing: List[str] = []
    for edge_name in edge_names:
        e = runtime_graph.edges.get(str(edge_name))
        if e is None:
            missing.append(str(edge_name))
            continue
        rows.append(int(e.matrix_index[0]))
        cols.append(int(e.matrix_index[1]))
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
        missing,
    )

def build_edge_sign_tensor(
    edge_names: Sequence[str],
    edge_scores: Sequence[float],
    runtime_graph: Graph,
    device: torch.device,
) -> torch.Tensor:
    if len(edge_names) != len(edge_scores):
        raise ValueError("edge_names and edge_scores length mismatch")
    signs: List[float] = []
    for edge_name, edge_score in zip(edge_names, edge_scores):
        if runtime_graph.edges.get(str(edge_name)) is None:
            continue
        if float(edge_score) > 0:
            signs.append(1.0)
        elif float(edge_score) < 0:
            signs.append(-1.0)
        else:
            signs.append(0.0)
    return torch.tensor(signs, dtype=torch.float32, device=device)

def build_edge_weight_tensor(
    edge_names: Sequence[str],
    edge_scores: Sequence[float],
    runtime_graph: Graph,
    device: torch.device,
) -> torch.Tensor:
    if len(edge_names) != len(edge_scores):
        raise ValueError("edge_names and edge_scores length mismatch")
    weights: List[float] = []
    for edge_name, edge_score in zip(edge_names, edge_scores):
        if runtime_graph.edges.get(str(edge_name)) is None:
            continue
        weights.append(float(edge_score))
    return torch.tensor(weights, dtype=torch.float32, device=device)

def _sum_selected_scores(
    scores_matrix: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    sum_abs: bool,
    edge_weights: Optional[torch.Tensor] = None,
) -> float:
    if row_idx.numel() == 0 or col_idx.numel() == 0:
        return float("nan")
    vals = scores_matrix[row_idx, col_idx]
    if edge_weights is not None:
        if int(edge_weights.numel()) != int(vals.numel()):
            raise ValueError(
                f"edge_weights size mismatch: weights={int(edge_weights.numel())}, vals={int(vals.numel())}"
            )
        vals = vals * edge_weights.to(device=vals.device, dtype=vals.dtype)
    elif sum_abs:
        vals = vals.abs()
    return float(vals.sum().item())

def reset_graph_scores(graph: Graph) -> None:
    graph.scores.zero_()

def build_random_edge_subsets(real_rows: torch.Tensor, real_cols: torch.Tensor, subset_size: int, num_trials: int, seed: int, device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor]:
    n_edges = int(real_rows.numel())
    k = max(0, min(int(subset_size), n_edges))
    t = max(0, int(num_trials))
    if t == 0 or k == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0, 0), dtype=torch.long, device=device),
        )

    rr = real_rows.detach().cpu().numpy()
    cc = real_cols.detach().cpu().numpy()
    rng = np.random.default_rng(int(seed))
    out_r = np.empty((t, k), dtype=np.int64)
    out_c = np.empty((t, k), dtype=np.int64)
    if k >= n_edges:
        out_r[:] = rr[np.newaxis, :]
        out_c[:] = cc[np.newaxis, :]
    else:
        for i in range(t):
            idx = rng.choice(n_edges, size=k, replace=False)
            out_r[i] = rr[idx]
            out_c[i] = cc[idx]
    return (
        torch.tensor(out_r, dtype=torch.long, device=device),
        torch.tensor(out_c, dtype=torch.long, device=device),
    )

def sum_random_subset_scores(scores_matrix: torch.Tensor, rand_rows: torch.Tensor, rand_cols: torch.Tensor, sum_abs: bool,) -> np.ndarray:
    if rand_rows.numel() == 0:
        return np.zeros((0,), dtype=np.float64)
    vals = scores_matrix[rand_rows, rand_cols]
    if sum_abs:
        vals = vals.abs()
    sums = vals.sum(dim=1)
    return sums.detach().float().cpu().numpy().astype(np.float64, copy=False)

def _load_existing_records(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_path.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hf_id = str(row.get("hf_id", "")).strip()
                if not hf_id:
                    continue
                out[hf_id] = {
                    "hf_id": hf_id,
                    "split": str(row.get("split", "")),
                    "dataset": str(row.get("dataset", "")),
                    "score_key": str(row.get("score_key", "")),
                    "true_score": _safe_float(row.get("true_score")),
                    "original_score": _safe_float(row.get("original_score")),
                    "high_circuit_score": _safe_float(row.get("high_circuit_score")),
                    "low_circuit_score": _safe_float(row.get("low_circuit_score")),
                    "random_mean_score": _safe_float(row.get("random_mean_score")),
                }
    except Exception:
        return {}
    return out

def _record_with_derived(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    high = _safe_float(out.get("high_circuit_score"))
    low = _safe_float(out.get("low_circuit_score"))
    out["high_minus_low"] = high - low if np.isfinite(high) and np.isfinite(low) else float("nan")
    out["abs_high_low"] = abs(high - low) if np.isfinite(high) and np.isfinite(low) else float("nan")
    return out

def _write_records_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    cols = [
        "hf_id",
        "split",
        "dataset",
        "score_key",
        "true_score",
        "original_score",
        "high_circuit_score",
        "low_circuit_score",
        "random_mean_score",
        "high_minus_low",
        "abs_high_low",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in cols})

def _build_runtime_rows(examples: Sequence[Dict[str, Any]], dataset: str, score_key: str, split_name: str, tokenizer: Any, seed: int, corrupt_type: str, noise_sigma: float,) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    dropped = 0
    for ex in examples:
        hf_id = str(ex.get("hf_id", "")).strip()
        label_space = [str(x) for x in ex.get("label_space", [])]
        gold_label = str(ex.get("gold_label", ""))
        if not hf_id or not label_space or not gold_label:
            dropped += 1
            continue
        try:
            prompt, _, label_token_ids = format_mcqa_prompt(
                question_text=str(ex.get("question_text", "")),
                choices=ex.get("choices", []),
                label_space=label_space,
                tokenizer=tokenizer,
                prefix_context="",
            )
        except Exception:
            dropped += 1
            continue
        if gold_label not in label_token_ids:
            dropped += 1
            continue
        try:
            option_token_ids = [int(label_token_ids[lab]) for lab in label_space]
            gold_token_id = int(label_token_ids[gold_label])
        except Exception:
            dropped += 1
            continue

        cseed = _stable_int(f"{seed}::{dataset}::{score_key}::{split_name}::{hf_id}")
        rows.append(
            {
                "clean": prompt,
                "corrupted": prompt,
                "label": gold_token_id,
                "hf_id": hf_id,
                "task": str(ex.get("task", dataset)),
                "score_key": score_key,
                "bucket": split_name,
                "gold_label": gold_label,
                "label_space": label_space,
                "label_token_ids": label_token_ids,
                "option_token_ids": option_token_ids,
                "corrupt": {
                    "type": corrupt_type,
                    "sigma": float(noise_sigma),
                    "seed": int(cseed),
                },
            }
        )
    return rows, dropped

def build_full_split_score_csv(
    *,
    dataset: str,
    score_key: str,
    split_examples: Sequence[Dict[str, Any]],
    out_csv_path: str,
    circuits_root: str,
    buckets_root: str,
    corrupt_type: str = "gaussian",
    topk_edges: int = 500,
    topk_rank_abs: bool = True,
    sum_abs: bool = False,
    direction_aware: bool = False,
    weighted_direction_aware: bool = False,
    random_baseline_trials: int = 150,
    seed: int = 42,
    model_name_override: str = "",
    device: str = "cuda",
    dtype: str = "",
    ig_steps_override: int = 0,
    incorrect_agg_override: str = "auto",
    noise_sigma: float = 0.1,
    split_name: str = "train",
    resume: bool = True,
    force_recompute: bool = False,
    save_every: int = 100,
    quiet: bool = False,
) -> Dict[str, Any]:
    csv_path = Path(out_csv_path)
    meta_path = csv_path.with_suffix(".meta.json")
    _ensure_dir(csv_path.parent)

    circuits_root_p = Path(circuits_root)
    buckets_root_p = Path(buckets_root)
    if not circuits_root_p.is_dir():
        raise FileNotFoundError(f"circuits_root not found: {circuits_root_p}")
    if not buckets_root_p.is_dir():
        raise FileNotFoundError(f"buckets_root not found: {buckets_root_p}")

    high_pt = _find_latest_circuit_pt(
        circuits_root=circuits_root_p,
        dataset=dataset,
        score_key=score_key,
        bucket="high",
        corrupt_type=corrupt_type,
    )
    low_pt = _find_latest_circuit_pt(
        circuits_root=circuits_root_p,
        dataset=dataset,
        score_key=score_key,
        bucket="low",
        corrupt_type=corrupt_type,
    )
    if high_pt is None or low_pt is None:
        raise FileNotFoundError(
            "Failed to locate circuit pt files for dataset/score_key. "
            f"dataset={dataset}, score_key={score_key}, corrupt_type={corrupt_type}, "
            f"high_found={high_pt is not None}, low_found={low_pt is not None}"
        )
    high_run_json = high_pt.with_suffix(".run.json")
    low_run_json = low_pt.with_suffix(".run.json")
    runtime_cfg = _resolve_runtime_config(
        high_run_json=high_run_json if high_run_json.is_file() else None,
        low_run_json=low_run_json if low_run_json.is_file() else None,
        model_name_override=model_name_override,
        ig_steps_override=int(ig_steps_override),
        incorrect_agg_override=incorrect_agg_override,
    )

    all_scores_json = _find_all_scores_json(buckets_root_p, dataset, score_key)
    if all_scores_json is None:
        raise FileNotFoundError(
            f"all score json not found for dataset={dataset}, score_key={score_key}, buckets_root={buckets_root_p}"
        )
    original_score_map = _load_original_score_map(all_scores_json, score_key)

    existing = _load_existing_records(csv_path) if (resume and csv_path.is_file() and not force_recompute) else {}
    if csv_path.is_file() and not force_recompute and not resume:
        records = [_record_with_derived(v) for _, v in sorted(existing.items(), key=lambda kv: kv[0])]
        _write_records_csv(csv_path, records)
        meta = {
            "status": "cached",
            "csv_path": str(csv_path),
            "meta_path": str(meta_path),
            "n_records": int(len(records)),
            "dataset": dataset,
            "score_key": score_key,
            "split_name": split_name,
            "circuits": {"high_pt": str(high_pt), "low_pt": str(low_pt)},
            "runtime": {
                "model_name": runtime_cfg.model_name,
                "ig_steps": int(runtime_cfg.ig_steps),
                "incorrect_agg": runtime_cfg.incorrect_agg,
            },
            "all_scores_json": str(all_scores_json),
        }
        _write_json(meta, meta_path)
        return meta

    actual_device = str(device or "").strip() or "cuda"
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = "cpu"
    score_dtype = _dtype_from_args(actual_device, dtype)

    model = load_model(runtime_cfg.model_name, actual_device, score_dtype)
    model.eval()

    runtime_rows, dropped_rows = _build_runtime_rows(
        examples=split_examples,
        dataset=dataset,
        score_key=score_key,
        split_name=split_name,
        tokenizer=model.tokenizer,
        seed=int(seed),
        corrupt_type=corrupt_type,
        noise_sigma=float(noise_sigma),
    )

    pending_rows: List[Dict[str, Any]] = []
    records_by_id: Dict[str, Dict[str, Any]] = dict(existing)
    for r in runtime_rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if not hf_id:
            continue
        if hf_id in records_by_id:
            continue
        pending_rows.append(r)

    _, _, high_names, high_edge_scores, high_total = select_edge_indices_from_circuit(
        high_pt,
        int(topk_edges),
        bool(topk_rank_abs),
    )
    _, _, low_names, low_edge_scores, low_total = select_edge_indices_from_circuit(
        low_pt,
        int(topk_edges),
        bool(topk_rank_abs),
    )

    graph = Graph.from_model(model)
    high_rows, high_cols, high_missing = remap_edge_indices_by_name(high_names, graph)
    low_rows, low_cols, low_missing = remap_edge_indices_by_name(low_names, graph)
    score_dev = graph.scores.device
    high_rows = high_rows.to(score_dev)
    high_cols = high_cols.to(score_dev)
    low_rows = low_rows.to(score_dev)
    low_cols = low_cols.to(score_dev)
    high_signs = build_edge_sign_tensor(high_names, high_edge_scores, graph, score_dev)
    low_signs = build_edge_sign_tensor(low_names, low_edge_scores, graph, score_dev)
    high_weights = build_edge_weight_tensor(high_names, high_edge_scores, graph, score_dev)
    low_weights = build_edge_weight_tensor(low_names, low_edge_scores, graph, score_dev)

    # Priority requested by experiment setting:
    # weighted_direction_aware > abs_sum > direction_aware > signed_sum
    if bool(weighted_direction_aware):
        score_mode = "weighted_direction_aware"
    elif bool(sum_abs):
        score_mode = "abs_sum"
    elif bool(direction_aware):
        score_mode = "direction_aware"
    else:
        score_mode = "signed_sum"

    if bool(weighted_direction_aware) and bool(sum_abs) and (not quiet):
        print("[WARN] --score_weighted_direction_aware is enabled, so --score_sum_abs is ignored for high/low score.")
    if bool(weighted_direction_aware) and bool(direction_aware) and (not quiet):
        print("[WARN] --score_weighted_direction_aware is enabled, so --score_direction_aware is ignored for high/low score.")
    if (not bool(weighted_direction_aware)) and bool(sum_abs) and bool(direction_aware) and (not quiet):
        print("[WARN] --score_sum_abs is enabled, so --score_direction_aware is ignored for high/low score.")

    random_enabled = int(random_baseline_trials) > 0
    random_rows = torch.empty((0, 0), dtype=torch.long, device=score_dev)
    random_cols = torch.empty((0, 0), dtype=torch.long, device=score_dev)
    random_subset_size = 0
    random_total_edges = 0
    random_trials_effective = 0
    random_mode = "mean_uniform_subsets"
    random_score_mode = "mean_raw_signed_sum"
    if random_enabled:
        real_rows, real_cols = torch.where(graph.real_edge_mask)
        real_rows = real_rows.to(score_dev)
        real_cols = real_cols.to(score_dev)
        n_real = int(real_rows.numel())
        random_total_edges = n_real
        if int(topk_edges) > 0:
            random_subset_size = min(int(topk_edges), n_real)
        else:
            counts = [int(len(high_rows)), int(len(low_rows))]
            valid = [c for c in counts if c > 0]
            random_subset_size = int(round(float(np.mean(valid)))) if valid else 0
            random_subset_size = min(random_subset_size, n_real)
        case_seed = int(seed) + int(zlib.crc32(f"{dataset}/{score_key}".encode("utf-8")) & 0xFFFFFFFF)
        rand_rows_2d, rand_cols_2d = build_random_edge_subsets(
            real_rows=real_rows,
            real_cols=real_cols,
            subset_size=int(random_subset_size),
            num_trials=int(random_baseline_trials),
            seed=case_seed + 17,
            device=score_dev,
        )
        if rand_rows_2d.numel() > 0:
            random_rows = rand_rows_2d
            random_cols = rand_cols_2d
            random_trials_effective = int(rand_rows_2d.shape[0])

    metric_attr = partial(
        logit_diff_mc_tensor,
        loss=True,
        mean=True,
        incorrect_agg=runtime_cfg.incorrect_agg,
    )

    skipped_dataset_fail = 0
    skipped_attribute_fail = 0
    processed_now = 0
    total_pending = len(pending_rows)

    for i, row in enumerate(pending_rows, start=1):
        hf_id = str(row.get("hf_id", "")).strip()
        if not hf_id:
            continue

        try:
            ds = PairsDatasetRuntimeCorrupt(
                [row],
                tokenizer=model.tokenizer,
                dataset=dataset,
                require_equal_length=True,
                max_items=1,
                strict_label_token=True,
                debug_first=False,
            )
            dl = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_pairs,
                num_workers=0,
                pin_memory=False,
            )
        except Exception:
            skipped_dataset_fail += 1
            continue

        model.reset_hooks(including_permanent=True)
        model.zero_grad(set_to_none=True)
        reset_graph_scores(graph)
        try:
            attribute(
                model,
                graph,
                dl,
                metric_attr,
                method="EAP-IG-inputs",
                ig_steps=runtime_cfg.ig_steps,
                quiet=True,
            )
        except Exception:
            skipped_attribute_fail += 1
            model.reset_hooks(including_permanent=True)
            model.zero_grad(set_to_none=True)
            reset_graph_scores(graph)
            continue

        scores = graph.scores
        high_score = _sum_selected_scores(
            scores,
            high_rows,
            high_cols,
            bool(sum_abs) and (score_mode == "abs_sum"),
            edge_weights=(
                high_weights if score_mode == "weighted_direction_aware"
                else (high_signs if score_mode == "direction_aware" else None)
            ),
        )
        low_score = _sum_selected_scores(
            scores,
            low_rows,
            low_cols,
            bool(sum_abs) and (score_mode == "abs_sum"),
            edge_weights=(
                low_weights if score_mode == "weighted_direction_aware"
                else (low_signs if score_mode == "direction_aware" else None)
            ),
        )
        random_mean = float("nan")
        if random_enabled:
            # RandC remains independent from high/low score_mode. We score the
            # sample on each random subset, then average the raw signed sums.
            random_scores = sum_random_subset_scores(
                scores,
                random_rows,
                random_cols,
                False,
            )
            if random_scores.size > 0:
                random_mean = float(np.mean(random_scores))

        true_score = float(original_score_map.get(hf_id, float("nan")))
        records_by_id[hf_id] = _record_with_derived(
            {
                "hf_id": hf_id,
                "split": split_name,
                "dataset": dataset,
                "score_key": score_key,
                "true_score": true_score,
                "original_score": true_score,
                "high_circuit_score": high_score,
                "low_circuit_score": low_score,
                "random_mean_score": random_mean,
            }
        )
        processed_now += 1

        model.reset_hooks(including_permanent=True)
        model.zero_grad(set_to_none=True)
        reset_graph_scores(graph)

        do_save = int(save_every) > 0 and (processed_now % int(save_every) == 0)
        if do_save:
            rows_now = [records_by_id[k] for k in sorted(records_by_id.keys())]
            _write_records_csv(csv_path, rows_now)
            if not quiet:
                print(
                    f"[INFO] score csv checkpoint saved ({processed_now}/{total_pending}) -> {csv_path}"
                )

        if (not quiet) and (i % 100 == 0):
            print(f"[INFO] scoring progress: {i}/{total_pending}")

    rows_final = [records_by_id[k] for k in sorted(records_by_id.keys())]
    _write_records_csv(csv_path, rows_final)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    meta = {
        "status": "ok",
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "dataset": dataset,
        "score_key": score_key,
        "split_name": split_name,
        "n_input_examples": int(len(split_examples)),
        "n_runtime_rows": int(len(runtime_rows)),
        "n_records_total": int(len(rows_final)),
        "n_records_preexisting": int(len(existing)),
        "n_processed_now": int(processed_now),
        "dropped_rows_before_scoring": int(dropped_rows),
        "pending_rows": int(total_pending),
        "skipped_dataset_fail": int(skipped_dataset_fail),
        "skipped_attribute_fail": int(skipped_attribute_fail),
        "all_scores_json": str(all_scores_json),
        "n_original_scores_available": int(len(original_score_map)),
        "runtime": {
            "model_name": runtime_cfg.model_name,
            "ig_steps": int(runtime_cfg.ig_steps),
            "incorrect_agg": runtime_cfg.incorrect_agg,
            "device": actual_device,
            "dtype": str(score_dtype),
            "sum_abs": bool(sum_abs),
            "direction_aware": bool(direction_aware),
            "weighted_direction_aware": bool(weighted_direction_aware),
            "score_mode": score_mode,
            "random_baseline_trials_arg": int(random_baseline_trials),
            "random_baseline_trials_effective": int(random_trials_effective),
            "random_mode": random_mode,
            "random_score_mode": random_score_mode,
            "random_total_edges": int(random_total_edges),
            "random_subset_size": int(random_subset_size),
            "resume": bool(resume),
            "force_recompute": bool(force_recompute),
        },
        "circuits": {
            "high_pt": str(high_pt),
            "low_pt": str(low_pt),
            "high_total_edges": int(high_total),
            "low_total_edges": int(low_total),
            "high_selected_edges": int(len(high_rows)),
            "low_selected_edges": int(len(low_rows)),
            "high_missing_runtime_edges": int(len(high_missing)),
            "low_missing_runtime_edges": int(len(low_missing)),
            "topk_edges": int(topk_edges),
            "topk_rank_abs": bool(topk_rank_abs),
        },
    }
    _write_json(meta, meta_path)
    return meta
