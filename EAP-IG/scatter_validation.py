import argparse
import gc
import json
import os
import random
import re
import sys
import zlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
_EAPIG_ROOT = _ROOT / "EAP-IG"
_EAP_SRC = _EAPIG_ROOT / "src"
for _p in [str(_ROOT), str(_EAPIG_ROOT), str(_EAP_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scen1_eapig import (  # type: ignore  # noqa: E402
    PairsDatasetRuntimeCorrupt,
    collate_pairs,
    load_model,
    logit_diff_mc_tensor,
    str_to_dtype,
)
from eap.attribute import attribute  # type: ignore  # noqa: E402
from eap.graph import Graph  # type: ignore  # noqa: E402

SCORE_KEYS_TD = ["aum", "forget", "el2n", "gradlogits_l2", "grad_align", "tracin"]
CIRCUIT_ROOT_HINTS = ["train_dynamic", "data_attr"]
BUCKET_ROOT_HINTS = ["train_dynamic", "data_attr"]
PAIR_ROOT_HINTS = ["train_dynamic", "data_attr"]
RUN_RE = re.compile(r"_m(?P<m>\d+)_top(?P<top>\d+)_")
PAIR_RE = re.compile(
    r"^pairs_(?P<dataset>.+)_(?P<score_key>[^_]+(?:_[^_]+)*)_(?P<bucket>high|low)_(?P<corrupt_type>[^.]+)\.jsonl$"
)
EVAL_MODES = {"bucket_only", "all_case_pairs", "all_dataset_pairs"}

@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    score_key: str
    corrupt_type: str
    high_pt: Path
    low_pt: Path
    high_bucket_json: Path
    low_bucket_json: Path
    high_pairs_jsonl: Path
    low_pairs_jsonl: Path
    high_run_json: Optional[Path]
    low_run_json: Optional[Path]

@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str
    ig_steps: int
    incorrect_agg: str

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def parse_csv_list(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

def normalize_model_name(name: str) -> str:
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

def _find_bucket_json(buckets_root: Path, dataset: str,score_key: str, bucket: str,) -> Optional[Path]:
    rel = Path(score_key) / dataset / f"{score_key}_{dataset}_{bucket}.json"
    roots = _expand_roots(buckets_root, BUCKET_ROOT_HINTS)
    for root in roots:
        c = root / rel
        if c.is_file():
            return c
    found = _iter_matching_files(roots, f"{score_key}_{dataset}_{bucket}.json")
    return found[0] if found else None

def _find_pairs_jsonl(pairs_root: Path, dataset: str, score_key: str, bucket: str, corrupt_type: str,) -> Optional[Path]:
    name = f"pairs_{dataset}_{score_key}_{bucket}_{corrupt_type}.jsonl"
    roots = _expand_roots(pairs_root, PAIR_ROOT_HINTS)
    for root in roots:
        c = root / name
        if c.is_file():
            return c
    found = _iter_matching_files(roots, name)
    return found[0] if found else None

def _find_all_scores_json(buckets_root: Path, dataset: str, score_key: str,) -> Optional[Path]:
    roots = _expand_roots(buckets_root, BUCKET_ROOT_HINTS)

    # Prefer score-specific files (e.g. data_attr/all_{score_key}_{dataset}.json),
    # then fall back to generic all_scores_{dataset}.json.
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

def _find_dataset_pairs_jsonl(pairs_root: Path, dataset: str, corrupt_type: str) -> List[Tuple[Path, str, str]]:
    roots = _expand_roots(pairs_root, PAIR_ROOT_HINTS)
    out: List[Tuple[Path, str, str]] = []
    seen: Set[Path] = set()
    prefix = f"pairs_{dataset}_"
    suffix = f"_{corrupt_type}.jsonl"
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob(f"{prefix}*{suffix}")):
            if p in seen:
                continue
            seen.add(p)
            name = p.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue

            mid = name[len(prefix) : -len(suffix)]  # "<score_key>_<bucket>"
            try:
                score_key, bucket = mid.rsplit("_", 1)
            except ValueError:
                continue
            if bucket not in {"high", "low"}:
                continue
            out.append((p, score_key, bucket))
    return out

def _source_priority(source_score_key: str, source_bucket: str, target_score_key: str) -> Tuple[int, int, str]:
    return (
        0 if source_score_key == target_score_key else 1,
        0 if source_bucket == "high" else 1,
        source_score_key,
    )

def discover_cases(circuits_root: Path, buckets_root: Path, pairs_root: Path, datasets_filter: Set[str], score_keys_filter: Set[str], corrupt_type: str,) -> Tuple[List[CaseSpec], List[str]]:
    warnings: List[str] = []
    cases: List[CaseSpec] = []
    seen_case: Set[Tuple[str, str, str]] = set()

    if not circuits_root.is_dir():
        raise FileNotFoundError(f"circuits_root not found: {circuits_root}")

    circuit_roots = _expand_roots(circuits_root, CIRCUIT_ROOT_HINTS)
    for croot in circuit_roots:
        dataset_dirs = [p for p in croot.iterdir() if p.is_dir()]
        dataset_dirs.sort(key=lambda p: p.name)
        for dset_dir in dataset_dirs:
            dataset = dset_dir.name
            if datasets_filter and dataset not in datasets_filter:
                continue
            score_dirs = [p for p in dset_dir.iterdir() if p.is_dir()]
            score_dirs.sort(key=lambda p: p.name)
            for sk_dir in score_dirs:
                score_key = sk_dir.name
                if score_keys_filter and score_key not in score_keys_filter:
                    continue
                case_key = (dataset, score_key, corrupt_type)
                if case_key in seen_case:
                    continue
                seen_case.add(case_key)

                high_pt = _find_latest_circuit_pt(circuits_root, dataset, score_key, "high", corrupt_type)
                low_pt = _find_latest_circuit_pt(circuits_root, dataset, score_key, "low", corrupt_type)
                if high_pt is None or low_pt is None:
                    warnings.append(
                        f"Skip {dataset}/{score_key}: missing circuit .pt "
                        f"(high={high_pt is not None}, low={low_pt is not None})"
                    )
                    continue

                high_bucket = _find_bucket_json(buckets_root, dataset, score_key, "high")
                low_bucket = _find_bucket_json(buckets_root, dataset, score_key, "low")
                if high_bucket is None or low_bucket is None:
                    warnings.append(
                        f"Skip {dataset}/{score_key}: missing bucket json "
                        f"(high={high_bucket is not None}, low={low_bucket is not None})"
                    )
                    continue

                high_pairs = _find_pairs_jsonl(pairs_root, dataset, score_key, "high", corrupt_type)
                low_pairs = _find_pairs_jsonl(pairs_root, dataset, score_key, "low", corrupt_type)
                if high_pairs is None or low_pairs is None:
                    warnings.append(
                        f"Skip {dataset}/{score_key}: missing pairs jsonl "
                        f"(high={high_pairs is not None}, low={low_pairs is not None})"
                    )
                    continue

                high_run = high_pt.with_suffix(".run.json")
                low_run = low_pt.with_suffix(".run.json")
                case = CaseSpec(
                    dataset=dataset,
                    score_key=score_key,
                    corrupt_type=corrupt_type,
                    high_pt=high_pt,
                    low_pt=low_pt,
                    high_bucket_json=high_bucket,
                    low_bucket_json=low_bucket,
                    high_pairs_jsonl=high_pairs,
                    low_pairs_jsonl=low_pairs,
                    high_run_json=high_run if high_run.is_file() else None,
                    low_run_json=low_run if low_run.is_file() else None,
                )
                cases.append(case)

    cases.sort(key=lambda c: (c.dataset, c.score_key))
    return cases, warnings

def load_bucket_score_map(bucket_json: Path, score_key: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    obj = read_json(bucket_json)
    rows = obj.get("items", obj if isinstance(obj, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected bucket format: {bucket_json}")

    score_map: Dict[str, Dict[str, Any]] = {}
    missing_hfid = 0
    missing_score = 0
    bad_score = 0
    duplicates = 0
    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if not hf_id:
            missing_hfid += 1
            continue

        score_val: Any = None
        scores = r.get("scores")
        if isinstance(scores, dict):
            score_val = scores.get(score_key)
        if score_val is None:
            score_val = r.get(score_key)
        if score_val is None:
            missing_score += 1
            continue
        try:
            score_float = float(score_val)
        except Exception:
            bad_score += 1
            continue

        if hf_id in score_map:
            duplicates += 1
        score_map[hf_id] = {
            "true_score": score_float,
            "task": str(r.get("task", "")),
        }

    stats = {
        "num_rows": len(rows),
        "num_unique_hf_id": len(score_map),
        "missing_hf_id": missing_hfid,
        "missing_score": missing_score,
        "bad_score": bad_score,
        "duplicates": duplicates,
    }
    return score_map, stats

def build_eval_rows(case: CaseSpec, validation_buckets: str, eval_mode: str, buckets_root: Path, pairs_root: Path,) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    assert validation_buckets in {"both", "high", "low"}
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"Unknown eval_mode={eval_mode}. choices={sorted(EVAL_MODES)}")
    use_buckets = ["high", "low"] if validation_buckets == "both" else [validation_buckets]

    out_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "validation_buckets": validation_buckets,
        "eval_mode": eval_mode,
        "bucket_stats": {},
        "pair_rows": {},
        "joined_rows": {},
        "missing_score_match": {},
    }

    if eval_mode == "bucket_only":
        seen: Set[Tuple[str, str]] = set()
        for bucket in use_buckets:
            bucket_json = case.high_bucket_json if bucket == "high" else case.low_bucket_json
            pairs_jsonl = case.high_pairs_jsonl if bucket == "high" else case.low_pairs_jsonl

            score_map, b_stats = load_bucket_score_map(bucket_json, case.score_key)
            pair_rows = read_jsonl(pairs_jsonl)
            stats["bucket_stats"][bucket] = b_stats
            stats["pair_rows"][bucket] = len(pair_rows)

            joined = 0
            missing = 0
            for r in pair_rows:
                hf_id = str(r.get("hf_id", "")).strip()
                if not hf_id:
                    missing += 1
                    continue
                info = score_map.get(hf_id)
                if info is None:
                    missing += 1
                    continue

                key = (hf_id, bucket)
                if key in seen:
                    continue
                seen.add(key)

                r2 = dict(r)
                r2["_true_score"] = float(info["true_score"])
                r2["_sample_bucket"] = bucket
                r2["_task"] = info["task"] or str(r.get("task", case.dataset))
                out_rows.append(r2)
                joined += 1
            stats["joined_rows"][bucket] = joined
            stats["missing_score_match"][bucket] = missing

        out_rows.sort(key=lambda x: (str(x.get("_sample_bucket", "")), str(x.get("hf_id", ""))))
        stats["num_rows_after_join"] = len(out_rows)
        return out_rows, stats

    all_scores_json = _find_all_scores_json(buckets_root, case.dataset, case.score_key)
    if all_scores_json is None:
        raise FileNotFoundError(
            f"all score json not found for dataset={case.dataset} under buckets_root={buckets_root}"
        )

    score_map, all_stats = load_bucket_score_map(all_scores_json, case.score_key)
    stats["all_scores_json"] = str(all_scores_json)
    stats["all_scores_stats"] = all_stats

    pair_sources: List[Tuple[Path, str, str]] = []
    if eval_mode == "all_case_pairs":
        if "high" in use_buckets:
            pair_sources.append((case.high_pairs_jsonl, case.score_key, "high"))
        if "low" in use_buckets:
            pair_sources.append((case.low_pairs_jsonl, case.score_key, "low"))
    elif eval_mode == "all_dataset_pairs":
        pair_sources = _find_dataset_pairs_jsonl(
            pairs_root=pairs_root,
            dataset=case.dataset,
            corrupt_type=case.corrupt_type,
        )
        if validation_buckets != "both":
            pair_sources = [x for x in pair_sources if x[2] == validation_buckets]
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    if not pair_sources:
        raise RuntimeError(
            f"No pair sources found for dataset={case.dataset}, corrupt_type={case.corrupt_type}, "
            f"eval_mode={eval_mode}, validation_buckets={validation_buckets}"
        )

    stats["num_pair_files"] = len(pair_sources)
    stats["pair_files"] = [str(p) for p, _, _ in pair_sources]
    stats["pair_files_by_source_bucket"] = {}
    for _, src_score_key, src_bucket in pair_sources:
        stats["pair_files_by_source_bucket"].setdefault(src_bucket, 0)
        stats["pair_files_by_source_bucket"][src_bucket] += 1

    best_by_hf_id: Dict[str, Dict[str, Any]] = {}
    pair_rows_total = 0
    missing_hf_id = 0
    missing_score = 0
    replaced_by_better_source = 0

    for pairs_jsonl, src_score_key, src_bucket in pair_sources:
        pair_rows = read_jsonl(pairs_jsonl)
        pair_rows_total += len(pair_rows)
        stats["pair_rows"][f"{src_score_key}:{src_bucket}"] = len(pair_rows)

        for r in pair_rows:
            hf_id = str(r.get("hf_id", "")).strip()
            if not hf_id:
                missing_hf_id += 1
                continue
            info = score_map.get(hf_id)
            if info is None:
                missing_score += 1
                continue

            priority = _source_priority(src_score_key, src_bucket, case.score_key)
            old = best_by_hf_id.get(hf_id)
            if old is not None and old["_source_priority"] <= priority:
                continue
            if old is not None:
                replaced_by_better_source += 1

            r2 = dict(r)
            r2["_true_score"] = float(info["true_score"])
            r2["_sample_bucket"] = "all"
            r2["_task"] = info["task"] or str(r.get("task", case.dataset))
            r2["_source_score_key"] = src_score_key
            r2["_source_bucket"] = src_bucket
            r2["_source_pairs_jsonl"] = str(pairs_jsonl)
            r2["_source_priority"] = priority
            best_by_hf_id[hf_id] = r2

    out_rows = list(best_by_hf_id.values())
    for r in out_rows:
        r.pop("_source_priority", None)
    out_rows.sort(key=lambda x: str(x.get("hf_id", "")))

    stats["pair_rows_total"] = pair_rows_total
    stats["missing_hf_id"] = missing_hf_id
    stats["missing_score_match"]["all"] = missing_score
    stats["joined_rows"]["all"] = len(out_rows)
    stats["replaced_by_better_source"] = replaced_by_better_source
    stats["num_rows_after_join"] = len(out_rows)
    return out_rows, stats

def subsample_rows(rows: List[Dict[str, Any]], max_samples_per_bucket: int, max_samples_per_case: int, seed: int,) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    rng = random.Random(seed)
    sampled = rows

    if max_samples_per_bucket > 0:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in sampled:
            b = str(r.get("_sample_bucket", "unknown"))
            grouped.setdefault(b, []).append(r)
        out: List[Dict[str, Any]] = []
        for b, arr in grouped.items():
            arr_sorted = sorted(arr, key=lambda x: str(x.get("hf_id", "")))
            if len(arr_sorted) > max_samples_per_bucket:
                arr_sorted = rng.sample(arr_sorted, max_samples_per_bucket)
            out.extend(arr_sorted)
        sampled = out

    if max_samples_per_case > 0 and len(sampled) > max_samples_per_case:
        sampled = rng.sample(sampled, max_samples_per_case)

    sampled.sort(key=lambda x: (str(x.get("_sample_bucket", "")), str(x.get("hf_id", ""))))
    return sampled

def _pick_from_run_meta(case: CaseSpec, key: str) -> Optional[Any]:
    vals: List[Any] = []
    for p in [case.high_run_json, case.low_run_json]:
        if p is None:
            continue
        try:
            v = read_json(p).get(key)
        except Exception:
            continue
        if v is None:
            continue
        vals.append(v)
    if not vals:
        return None
    return vals[0]

def resolve_runtime_config(case: CaseSpec, args: argparse.Namespace) -> RuntimeConfig:
    if args.ig_steps > 0:
        ig_steps = int(args.ig_steps)
    else:
        ig_steps = int(_pick_from_run_meta(case, "ig_steps") or 5)

    if args.incorrect_agg != "auto":
        incorrect_agg = args.incorrect_agg
    else:
        incorrect_agg = str(_pick_from_run_meta(case, "incorrect_agg") or "max")
        if incorrect_agg not in {"max", "mean", "min"}:
            incorrect_agg = "max"

    model_name_raw = (args.model_name or "").strip()
    if model_name_raw:
        model_name = normalize_model_name(model_name_raw)
    else:
        model_name = normalize_model_name(str(_pick_from_run_meta(case, "model_name") or ""))
        if not model_name:
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    return RuntimeConfig(model_name=model_name, ig_steps=ig_steps, incorrect_agg=incorrect_agg)

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

def percentile_rank(dist: Sequence[float], observed: float) -> float:
    arr = np.asarray(dist, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(observed):
        return float("nan")
    less = float(np.sum(arr < observed))
    equal = float(np.sum(arr == observed))
    return 100.0 * (less + 0.5 * equal) / float(arr.size)

def summarize_distribution(dist: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(dist, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }

def correlation_distribution(true_scores: np.ndarray, trial_matrix: np.ndarray) -> Dict[str, Any]:
    if trial_matrix.ndim != 2 or trial_matrix.shape[0] != true_scores.shape[0]:
        return {
            "pearson_dist": [],
            "spearman_dist": [],
            "pearson_summary": summarize_distribution([]),
            "spearman_summary": summarize_distribution([]),
        }
    pearsons: List[float] = []
    spearmans: List[float] = []
    for i in range(trial_matrix.shape[1]):
        c = compute_corr_dict(true_scores, trial_matrix[:, i])
        pearsons.append(float(c.get("pearson", np.nan)))
        spearmans.append(float(c.get("spearman", np.nan)))
    return {
        "pearson_dist": pearsons,
        "spearman_dist": spearmans,
        "pearson_summary": summarize_distribution(pearsons),
        "spearman_summary": summarize_distribution(spearmans),
    }

def compute_corr_dict(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    n = int(mask.sum())
    if n < 2:
        return {"n": n, "pearson": float("nan"), "spearman": float("nan")}
    sx = pd.Series(xs[mask])
    sy = pd.Series(ys[mask])
    pearson = float(sx.corr(sy, method="pearson"))
    spearman = float(sx.corr(sy, method="spearman"))
    return {"n": n, "pearson": pearson, "spearman": spearman}

def add_trendline(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, linewidth: float = 2.0) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return
    x_m = x[mask]
    y_m = y[mask]
    if len(np.unique(x_m)) < 2:
        return
    try:
        a, b = np.polyfit(x_m, y_m, deg=1)
    except Exception:
        return
    xs = np.linspace(float(x_m.min()), float(x_m.max()), 120)
    ys = a * xs + b
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.95)

def save_single_scatter(df: pd.DataFrame, y_col: str, title: str, x_label: str, y_label: str, out_path: Path,) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)
    colors = {"high": "#d1495b", "low": "#00798c"}
    if "sample_bucket" in df.columns:
        buckets = sorted(str(x) for x in df["sample_bucket"].dropna().unique().tolist())
        cmap = plt.get_cmap("tab10")
        for i, b in enumerate(buckets):
            sub = df[df["sample_bucket"] == b]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["true_score"].values,
                sub[y_col].values,
                s=18,
                alpha=0.75,
                label=f"samples={b}",
                color=colors.get(b, cmap(i % 10)),
            )
    else:
        ax.scatter(df["true_score"].values, df[y_col].values, s=18, alpha=0.75, color="#2b59c3")

    corr = compute_corr_dict(df["true_score"].values, df[y_col].values)
    x = df["true_score"].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    add_trendline(ax, x, y, color="#222222", linewidth=2.2)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    if len(df["sample_bucket"].unique()) > 1 if "sample_bucket" in df.columns else False:
        ax.legend(loc="best")
    ax.text(0.02, 0.98, f"N={corr['n']}\nPearson={corr['pearson']:.3f}\nSpearman={corr['spearman']:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

def save_both_scatter(df: pd.DataFrame, title: str, x_label: str, y_label: str, out_path: Path,) -> None:
    d_high = df[["hf_id", "true_score", "sample_bucket", "high_circuit_score"]].copy()
    d_high["circuit"] = "high"
    d_high = d_high.rename(columns={"high_circuit_score": "circuit_score"})
    d_low = df[["hf_id", "true_score", "sample_bucket", "low_circuit_score"]].copy()
    d_low["circuit"] = "low"
    d_low = d_low.rename(columns={"low_circuit_score": "circuit_score"})
    both = pd.concat([d_high, d_low], axis=0, ignore_index=True)
    colors = {"high": "#d1495b", "low": "#2b59c3"}

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)
    for c in ["high", "low"]:
        sub = both[both["circuit"] == c]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["true_score"].values,
            sub["circuit_score"].values,
            s=18,
            alpha=0.72,
            label=f"{c} circuit",
            color=colors[c],
        )
        add_trendline(
            ax,
            sub["true_score"].to_numpy(dtype=np.float64),
            sub["circuit_score"].to_numpy(dtype=np.float64),
            color=colors[c],
            linewidth=2.0,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

def save_three_scatter(df: pd.DataFrame, title: str, x_label: str, y_label: str, out_path: Path,) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=140)
    specs = [
        ("high_circuit_score", "high circuit", "#d1495b"),
        ("low_circuit_score", "low circuit", "#2b59c3"),
        ("random_mean_score", "random baseline", "#2a9d8f"),
    ]
    for col, label, color in specs:
        if col not in df.columns:
            continue
        sub = df[np.isfinite(df[col].to_numpy(dtype=np.float64))]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["true_score"].values,
            sub[col].values,
            s=16 if "random" not in col else 14,
            alpha=0.70 if "random" not in col else 0.50,
            label=label,
            color=color,
        )
        add_trendline(
            ax,
            sub["true_score"].to_numpy(dtype=np.float64),
            sub[col].to_numpy(dtype=np.float64),
            color=color,
            linewidth=2.0 if "random" not in col else 1.8,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

def save_random_corr_distribution_plot(random_pearson_dist: Sequence[float], observed_high: float, observed_low: float, out_path: Path,title_prefix: str,) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=140)
    arr = np.asarray(random_pearson_dist, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size > 0:
        bins = max(10, min(60, int(np.sqrt(arr.size) * 2)))
        ax.hist(arr, bins=bins, color="#2a9d8f", alpha=0.35, edgecolor="#2a9d8f", linewidth=0.8, label="random dist")
        ax.axvline(float(np.nanmean(arr)), color="#2a9d8f", linestyle=":", linewidth=1.8, alpha=0.95, label="random mean")
    if np.isfinite(observed_high):
        ax.axvline(float(observed_high), color="#d1495b", linestyle="--", linewidth=2.0, label="high circuit")
    if np.isfinite(observed_low):
        ax.axvline(float(observed_low), color="#2b59c3", linestyle="--", linewidth=2.0, label="low circuit")
    ax.set_title(f"{title_prefix} | Random Pearson corr distribution")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="best")

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

def try_load_existing_records(csv_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not csv_path.is_file():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    required = {"hf_id", "sample_bucket", "true_score", "high_circuit_score", "low_circuit_score"}
    if not required.issubset(set(df.columns)):
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, r in df.iterrows():
        hf_id = str(r["hf_id"])
        bucket = str(r["sample_bucket"])
        out[(hf_id, bucket)] = {
            "hf_id": hf_id,
            "sample_bucket": bucket,
            "true_score": float(r["true_score"]),
            "high_circuit_score": float(r["high_circuit_score"]),
            "low_circuit_score": float(r["low_circuit_score"]),
        }
    return out

def _dtype_from_args(device: str, dtype_str: str) -> torch.dtype:
    ds = (dtype_str or "").strip().lower()
    if not ds:
        return torch.float16 if device.startswith("cuda") else torch.float32
    dt = str_to_dtype(ds)
    if (not device.startswith("cuda")) and dt in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dt

def resolve_highlow_score_mode(args: argparse.Namespace) -> str:
    if bool(args.score_weighted_direction_aware):
        return "weighted_direction_aware"
    return "abs_sum" if bool(args.sum_abs) else "signed_sum"

def process_case(case: CaseSpec, args: argparse.Namespace, model: Any, runtime_cfg: RuntimeConfig, buckets_root: Path, pairs_root: Path, out_root: Path,) -> Dict[str, Any]:
    case_out_dir = out_root / case.dataset / case.score_key
    ensure_dir(case_out_dir)
    highlow_score_mode = resolve_highlow_score_mode(args)
    mode_tag = {
        "weighted_direction_aware": "weighted",
        "abs_sum": "sumabs",
        "signed_sum": "signed",
    }[highlow_score_mode]
    tag = (
        f"mode{args.eval_mode}_"
        f"vb{args.validation_buckets}_"
        f"k{args.topk_edges if args.topk_edges > 0 else 'all'}_"
        f"rank{'abs' if args.topk_rank_abs else 'signed'}_"
        f"score{mode_tag}_"
        f"rb{int(args.random_baseline_trials)}_"
        f"ig{runtime_cfg.ig_steps}_"
        f"agg{runtime_cfg.incorrect_agg}"
    )
    stem = f"{case.dataset}_{case.score_key}_{tag}"
    csv_path = case_out_dir / f"{stem}_sample_scores.csv"
    high_png = case_out_dir / f"{stem}_high_only.png"
    low_png = case_out_dir / f"{stem}_low_only.png"
    both_png = case_out_dir / f"{stem}_both.png"
    random_png = case_out_dir / f"{stem}_random_only.png"
    hlr_png = case_out_dir / f"{stem}_high_low_random.png"
    random_corr_png = case_out_dir / f"{stem}_random_corr_dist.png"
    json_path = case_out_dir / f"{stem}_metrics.json"

    random_enabled = int(args.random_baseline_trials) > 0
    base_ready = csv_path.is_file() and high_png.is_file() and low_png.is_file() and both_png.is_file() and json_path.is_file()
    random_ready = random_png.is_file() and hlr_png.is_file() and random_corr_png.is_file()
    if args.skip_existing and base_ready and ((not random_enabled) or random_ready):
        return read_json(json_path)

    _, _, high_names, high_top_scores, high_total = select_edge_indices_from_circuit(case.high_pt, args.topk_edges, args.topk_rank_abs)
    _, _, low_names, low_top_scores, low_total = select_edge_indices_from_circuit( case.low_pt, args.topk_edges, args.topk_rank_abs)

    eval_rows, join_stats = build_eval_rows(case=case, validation_buckets=args.validation_buckets, eval_mode=args.eval_mode, buckets_root=buckets_root, pairs_root=pairs_root)
    eval_rows = subsample_rows(eval_rows, max_samples_per_bucket=args.max_samples_per_bucket, max_samples_per_case=args.max_samples_per_case, seed=args.seed,)

    resume_for_case = bool(args.resume) and (not random_enabled)
    existing = try_load_existing_records(csv_path) if resume_for_case else {}
    records: List[Dict[str, Any]] = []
    pending_rows: List[Dict[str, Any]] = []
    for r in eval_rows:
        hf_id = str(r.get("hf_id", ""))
        bucket = str(r.get("_sample_bucket", ""))
        key = (hf_id, bucket)
        old = existing.get(key)
        if old is None:
            pending_rows.append(r)
        else:
            records.append(old)

    graph = Graph.from_model(model)
    high_rows, high_cols, high_missing = remap_edge_indices_by_name(high_names, graph)
    low_rows, low_cols, low_missing = remap_edge_indices_by_name(low_names, graph)
    # --- gpu to tensor ---
    score_dev = graph.scores.device
    high_rows = high_rows.to(score_dev)
    high_cols = high_cols.to(score_dev)
    low_rows  = low_rows.to(score_dev)
    low_cols  = low_cols.to(score_dev)
    high_weights = torch.empty((0,), dtype=torch.float32, device=score_dev)
    low_weights = torch.empty((0,), dtype=torch.float32, device=score_dev)
    if highlow_score_mode == "weighted_direction_aware":
        high_weights = build_edge_weight_tensor(high_names, high_top_scores, graph, score_dev)
        low_weights = build_edge_weight_tensor(low_names, low_top_scores, graph, score_dev)

    random_rows = torch.empty((0, 0), dtype=torch.long, device=score_dev)
    random_cols = torch.empty((0, 0), dtype=torch.long, device=score_dev)
    random_subset_size = 0
    random_subset_rule = "disabled"
    if random_enabled:
        real_rows, real_cols = torch.where(graph.real_edge_mask)
        real_rows = real_rows.to(score_dev)
        real_cols = real_cols.to(score_dev)
        real_edge_count = int(real_rows.numel())
        if int(args.topk_edges) > 0:
            random_subset_size = min(int(args.topk_edges), real_edge_count)
            random_subset_rule = "topk_edges_arg"
        else:
            counts = [int(len(high_rows)), int(len(low_rows))]
            valid = [c for c in counts if c > 0]
            random_subset_size = int(round(float(np.mean(valid)))) if valid else 0
            random_subset_size = min(random_subset_size, real_edge_count)
            random_subset_rule = "mean_selected_edges_from_circuits"

        case_seed = int(args.seed) + int(zlib.crc32(f"{case.dataset}/{case.score_key}".encode("utf-8")) & 0xFFFFFFFF)
        random_rows, random_cols = build_random_edge_subsets(
            real_rows=real_rows,
            real_cols=real_cols,
            subset_size=int(random_subset_size),
            num_trials=int(args.random_baseline_trials),
            seed=case_seed + 11,
            device=score_dev,
        )

    metric_attr = partial(logit_diff_mc_tensor,loss=True, mean=True, incorrect_agg=runtime_cfg.incorrect_agg)

    skipped_dataset_fail = 0
    skipped_attribute_fail = 0
    random_trials_by_key: Dict[Tuple[str, str], np.ndarray] = {}
    iterator = pending_rows if args.quiet else tqdm(pending_rows, desc=f"{case.dataset}/{case.score_key}", leave=False)
    for row in iterator:
        hf_id = str(row.get("hf_id", ""))
        sample_bucket = str(row.get("_sample_bucket", ""))
        true_score = float(row.get("_true_score"))
        try:
            ds = PairsDatasetRuntimeCorrupt(
                [row],
                tokenizer=model.tokenizer,
                dataset=case.dataset,
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

        s = graph.scores
        high_score = _sum_selected_scores(
            s,
            high_rows,
            high_cols,
            bool(args.sum_abs) and (highlow_score_mode == "abs_sum"),
            edge_weights=(high_weights if highlow_score_mode == "weighted_direction_aware" else None),
        )
        low_score = _sum_selected_scores(
            s,
            low_rows,
            low_cols,
            bool(args.sum_abs) and (highlow_score_mode == "abs_sum"),
            edge_weights=(low_weights if highlow_score_mode == "weighted_direction_aware" else None),
        )
        random_mean = float("nan")
        key = (hf_id, sample_bucket)
        if random_enabled:
            rand_trials = sum_random_subset_scores(s, random_rows, random_cols, args.sum_abs)
            random_trials_by_key[key] = rand_trials
            if rand_trials.size > 0:
                random_mean = float(np.mean(rand_trials))

        records.append(
            {
                "hf_id": hf_id,
                "sample_bucket": sample_bucket,
                "true_score": true_score,
                "high_circuit_score": high_score,
                "low_circuit_score": low_score,
                "random_mean_score": random_mean,
            }
        )
        model.reset_hooks(including_permanent=True)
        model.zero_grad(set_to_none=True)
        reset_graph_scores(graph)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    df = pd.DataFrame.from_records(records)
    if len(df) == 0:
        report = {
            "dataset": case.dataset,
            "score_key": case.score_key,
            "status": "empty",
            "reason": "no valid records after join/attribution",
            "eval_mode": args.eval_mode,
            "join_stats": join_stats,
            "skipped_dataset_fail": skipped_dataset_fail,
            "skipped_attribute_fail": skipped_attribute_fail,
            "edge_mapping": {
                "high_selected_by_name": int(len(high_names)),
                "high_mapped_to_runtime": int(len(high_rows)),
                "high_missing_in_runtime": int(len(high_missing)),
                "low_selected_by_name": int(len(low_names)),
                "low_mapped_to_runtime": int(len(low_rows)),
                "low_missing_in_runtime": int(len(low_missing)),
            },
            "paths": {
                "high_pt": str(case.high_pt),
                "low_pt": str(case.low_pt),
            },
            "runtime": {
                "model_name": runtime_cfg.model_name,
                "ig_steps": runtime_cfg.ig_steps,
                "incorrect_agg": runtime_cfg.incorrect_agg,
                "resume_used_for_records": resume_for_case,
                "random_baseline_trials": int(args.random_baseline_trials),
            },
        }
        write_json(report, json_path)
        return report

    df = df.sort_values(["sample_bucket", "hf_id"]).reset_index(drop=True)
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)

    true_scores = df["true_score"].to_numpy(dtype=np.float64)
    high_scores = df["high_circuit_score"].to_numpy(dtype=np.float64)
    low_scores = df["low_circuit_score"].to_numpy(dtype=np.float64)

    high_corr = compute_corr_dict(true_scores, high_scores)
    low_corr = compute_corr_dict(true_scores, low_scores)
    high_missing_set = set(high_missing)
    low_missing_set = set(low_missing)
    high_edges_set = {n for n in high_names if n not in high_missing_set}
    low_edges_set = {n for n in low_names if n not in low_missing_set}

    random_baseline_report: Dict[str, Any] = {
        "enabled": bool(random_enabled),
        "num_trials": int(args.random_baseline_trials),
        "resume_compatible": bool(not random_enabled),
        "note": "resume is disabled when random baseline is enabled",
        "subset_size": int(random_subset_size),
        "subset_size_rule": random_subset_rule,
    }
    random_pearson_dist: List[float] = []
    if random_enabled:
        random_keys = [(str(r["hf_id"]), str(r["sample_bucket"])) for _, r in df.iterrows()]
        trial_rows: List[np.ndarray] = []
        missing_trial_rows = 0
        for key in random_keys:
            trials = random_trials_by_key.get(key)
            if trials is None:
                missing_trial_rows += 1
                continue
            trial_rows.append(trials)

        trial_matrix = np.stack(trial_rows, axis=0) if trial_rows else np.zeros((0, 0), dtype=np.float64)
        rand_dist = correlation_distribution(true_scores, trial_matrix)
        random_pearson_dist = [float(x) for x in rand_dist["pearson_dist"]]

        random_baseline_report.update(
            {
                "missing_trial_rows": int(missing_trial_rows),
                "correlation_distribution": {
                    "pearson_dist": [float(x) for x in rand_dist["pearson_dist"]],
                    "spearman_dist": [float(x) for x in rand_dist["spearman_dist"]],
                    "pearson": rand_dist["pearson_summary"],
                    "spearman": rand_dist["spearman_summary"],
                },
                "circuit_vs_random": {
                    "high": {
                        "pearson_percentile_of_circuit": percentile_rank(
                            rand_dist["pearson_dist"], float(high_corr.get("pearson", np.nan))
                        ),
                        "spearman_percentile_of_circuit": percentile_rank(
                            rand_dist["spearman_dist"], float(high_corr.get("spearman", np.nan))
                        ),
                        "pearson_upper_tail_p": float(
                            np.mean(
                                np.asarray(rand_dist["pearson_dist"], dtype=np.float64)
                                >= float(high_corr.get("pearson", np.nan))
                            )
                        )
                        if rand_dist["pearson_dist"]
                        else float("nan"),
                    },
                    "low": {
                        "pearson_percentile_of_circuit": percentile_rank(
                            rand_dist["pearson_dist"], float(low_corr.get("pearson", np.nan))
                        ),
                        "spearman_percentile_of_circuit": percentile_rank(
                            rand_dist["spearman_dist"], float(low_corr.get("spearman", np.nan))
                        ),
                        "pearson_upper_tail_p": float(
                            np.mean(
                                np.asarray(rand_dist["pearson_dist"], dtype=np.float64)
                                >= float(low_corr.get("pearson", np.nan))
                            )
                        )
                        if rand_dist["pearson_dist"]
                        else float("nan"),
                    },
                },
            }
        )

    x_label = f"True {case.score_key} score"
    if highlow_score_mode == "weighted_direction_aware":
        highlow_ylabel = "Weighted direction-aware circuit score"
    elif highlow_score_mode == "abs_sum":
        highlow_ylabel = "Circuit attribution sum (abs)"
    else:
        highlow_ylabel = "Circuit attribution sum (signed)"
    save_single_scatter(
        df=df,
        y_col="high_circuit_score",
        title=f"{case.dataset} / {case.score_key} | High-circuit score vs true score",
        x_label=x_label,
        y_label=highlow_ylabel,
        out_path=high_png,
    )
    save_single_scatter(
        df=df,
        y_col="low_circuit_score",
        title=f"{case.dataset} / {case.score_key} | Low-circuit score vs true score",
        x_label=x_label,
        y_label=highlow_ylabel,
        out_path=low_png,
    )
    save_both_scatter(
        df=df,
        title=f"{case.dataset} / {case.score_key} | High vs Low circuit scores",
        x_label=x_label,
        y_label=highlow_ylabel,
        out_path=both_png,
    )

    if random_enabled:
        save_single_scatter(
            df=df,
            y_col="random_mean_score",
            title=f"{case.dataset} / {case.score_key} | Random baseline score vs true score",
            x_label=x_label,
            y_label="Random baseline attribution sum",
            out_path=random_png,
        )
        save_three_scatter(
            df=df,
            title=f"{case.dataset} / {case.score_key} | High / Low / Random",
            x_label=x_label,
            y_label=f"{highlow_ylabel} / random",
            out_path=hlr_png,
        )
        save_random_corr_distribution_plot(
            random_pearson_dist=random_pearson_dist,
            observed_high=float(high_corr.get("pearson", np.nan)),
            observed_low=float(low_corr.get("pearson", np.nan)),
            out_path=random_corr_png,
            title_prefix=f"{case.dataset}/{case.score_key}",
        )

    per_bucket: Dict[str, Any] = {}
    for b in sorted(df["sample_bucket"].unique().tolist()):
        sub = df[df["sample_bucket"] == b]
        per_bucket[b] = {
            "n": int(len(sub)),
            "high_circuit": compute_corr_dict(sub["true_score"].values, sub["high_circuit_score"].values),
            "low_circuit": compute_corr_dict(sub["true_score"].values, sub["low_circuit_score"].values),
        }
        if random_enabled and "random_mean_score" in sub.columns:
            per_bucket[b]["random_mean"] = compute_corr_dict(sub["true_score"].values, sub["random_mean_score"].values)

    report = {
        "dataset": case.dataset,
        "score_key": case.score_key,
        "corrupt_type": case.corrupt_type,
        "eval_mode": args.eval_mode,
        "status": "ok",
        "n_samples": int(len(df)),
        "counts_by_sample_bucket": {k: int(v) for k, v in df["sample_bucket"].value_counts().to_dict().items()},
        "join_stats": join_stats,
        "skipped_dataset_fail": skipped_dataset_fail,
        "skipped_attribute_fail": skipped_attribute_fail,
        "edge_mapping": {
            "high_selected_by_name": int(len(high_names)),
            "high_mapped_to_runtime": int(len(high_rows)),
            "high_missing_in_runtime": int(len(high_missing)),
            "low_selected_by_name": int(len(low_names)),
            "low_mapped_to_runtime": int(len(low_rows)),
            "low_missing_in_runtime": int(len(low_missing)),
        },
        "runtime": {
            "model_name": runtime_cfg.model_name,
            "ig_steps": runtime_cfg.ig_steps,
            "incorrect_agg": runtime_cfg.incorrect_agg,
            "sum_abs": bool(args.sum_abs),
            "score_weighted_direction_aware": bool(args.score_weighted_direction_aware),
            "score_mode": highlow_score_mode,
            "resume_used_for_records": resume_for_case,
            "random_baseline_trials": int(args.random_baseline_trials),
        },
        "edge_selection": {
            "topk_edges": int(args.topk_edges),
            "rank_abs": bool(args.topk_rank_abs),
            "high_circuit_total_edges": int(high_total),
            "low_circuit_total_edges": int(low_total),
            "high_circuit_selected": int(len(high_rows)),
            "low_circuit_selected": int(len(low_rows)),
            "edge_overlap_count": int(len(high_edges_set & low_edges_set)),
            "high_top10": [{"edge": n, "score": s} for n, s in list(zip(high_names, high_top_scores))[:10]],
            "low_top10": [{"edge": n, "score": s} for n, s in list(zip(low_names, low_top_scores))[:10]],
        },
        "correlation": {
            "high_circuit": high_corr,
            "low_circuit": low_corr,
            "random_baseline": random_baseline_report,
            "per_sample_bucket": per_bucket,
        },
        "artifacts": {
            "sample_csv": str(csv_path),
            "high_scatter_png": str(high_png),
            "low_scatter_png": str(low_png),
            "both_scatter_png": str(both_png),
            "random_scatter_png": str(random_png) if random_enabled else "",
            "high_low_random_scatter_png": str(hlr_png) if random_enabled else "",
            "random_corr_dist_png": str(random_corr_png) if random_enabled else "",
            "metrics_json": str(json_path),
        },
        "paths": {
            "high_pt": str(case.high_pt),
            "low_pt": str(case.low_pt),
            "high_bucket_json": str(case.high_bucket_json),
            "low_bucket_json": str(case.low_bucket_json),
            "high_pairs_jsonl": str(case.high_pairs_jsonl),
            "low_pairs_jsonl": str(case.low_pairs_jsonl),
        },
    }
    write_json(report, json_path)
    return report

def _sanitize_tag(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw).strip())
    s = s.strip("_")
    return s or "all"

def build_run_summary_tag(
    datasets_filter: Set[str],
    score_keys_filter: Set[str],
    eval_mode: str,
    validation_buckets: str,
) -> str:
    ds_tag = "_".join(sorted(datasets_filter)) if datasets_filter else "all"
    sk_tag = "_".join(sorted(score_keys_filter)) if score_keys_filter else "all"
    raw = f"ds_{ds_tag}__sk_{sk_tag}__mode_{eval_mode}__vb_{validation_buckets}"
    return _sanitize_tag(raw)

def build_summary_rows_from_reports(reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_for_csv: List[Dict[str, Any]] = []
    for r in reports:
        if r.get("status") != "ok":
            rows_for_csv.append(
                {
                    "dataset": r.get("dataset"),
                    "score_key": r.get("score_key"),
                    "status": r.get("status"),
                    "n_samples": r.get("n_samples", 0),
                    "pearson_high": np.nan,
                    "spearman_high": np.nan,
                    "pearson_low": np.nan,
                    "spearman_low": np.nan,
                    "random_trials": 0,
                    "pearson_high_pct_vs_random": np.nan,
                    "pearson_low_pct_vs_random": np.nan,
                    "metrics_json": r.get("artifacts", {}).get("metrics_json", ""),
                }
            )
            continue
        corr = r.get("correlation", {})
        hc = corr.get("high_circuit", {})
        lc = corr.get("low_circuit", {})
        rb = corr.get("random_baseline", {})
        rb_cmp = rb.get("circuit_vs_random", {})
        rb_high = rb_cmp.get("high", {})
        rb_low = rb_cmp.get("low", {})
        rows_for_csv.append(
            {
                "dataset": r.get("dataset"),
                "score_key": r.get("score_key"),
                "status": r.get("status"),
                "n_samples": r.get("n_samples", 0),
                "pearson_high": hc.get("pearson"),
                "spearman_high": hc.get("spearman"),
                "pearson_low": lc.get("pearson"),
                "spearman_low": lc.get("spearman"),
                "random_trials": rb.get("num_trials", 0),
                "pearson_high_pct_vs_random": rb_high.get("pearson_percentile_of_circuit"),
                "pearson_low_pct_vs_random": rb_low.get("pearson_percentile_of_circuit"),
                "metrics_json": r.get("artifacts", {}).get("metrics_json", ""),
            }
        )
    return rows_for_csv

def collect_latest_reports_from_metrics(out_root: Path) -> List[Dict[str, Any]]:
    latest_by_case: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
    for p in sorted(out_root.rglob("*_metrics.json")):
        try:
            rep = read_json(p)
        except Exception:
            continue
        dataset = str(rep.get("dataset", "")).strip()
        score_key = str(rep.get("score_key", "")).strip()
        if not dataset or not score_key:
            continue
        key = (dataset, score_key)
        mtime = float(p.stat().st_mtime)
        prev = latest_by_case.get(key)
        if prev is None or mtime > prev[0]:
            latest_by_case[key] = (mtime, rep)
    return [
        v[1]
        for _, v in sorted(latest_by_case.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Correlation sanity-check between true score and per-sample circuit attribution score.")
    ap.add_argument("--root", type=str, default=str(_ROOT))
    ap.add_argument("--circuits_root", type=str, default=str(_ROOT / "circuit_results"))
    ap.add_argument("--buckets_root", type=str, default=str(_ROOT / "data_gens" / "buckets"))
    ap.add_argument("--pairs_root", type=str, default=str(_ROOT / "data_gens" / "pairs"))
    ap.add_argument("--out_root", type=str, default=str(_ROOT / "results" / "validation" / "scatter"))
    ap.add_argument("--datasets", type=str, default="arc_easy,sciq,piqa,winogrande")
    ap.add_argument("--score_keys",type=str, default="aum,grad_align,forget,el2n,gradlogits_l2,tracin")
    ap.add_argument("--corrupt_type", type=str, default="gaussian")
    ap.add_argument("--validation_buckets", type=str, default="both", choices=["both", "high", "low"])
    ap.add_argument("--eval_mode", type=str, default="all_dataset_pairs", choices=["bucket_only", "all_case_pairs", "all_dataset_pairs"], help="bucket_only: 기존 high/low bucket 기반 검증. all_case_pairs: 동일 case의 high/low pairs + all scores. all_dataset_pairs: 같은 dataset의 모든 pairs + all scores (권장).")
    ap.add_argument("--topk_edges", type=int, default=500, help="Top-k edges per circuit. <=0 means saved in_graph edges.")
    ap.add_argument("--topk_rank_signed", dest="topk_rank_abs", action="store_false", help="Rank edges by signed score.")
    ap.add_argument("--sum_abs", action="store_true", help="Use sum(abs(edge score)) instead of signed sum.")
    ap.add_argument(
        "--score_weighted_direction_aware",
        action="store_true",
        help="Use weighted direction-aware high/low score: sum(attribution * circuit edge score). Overrides --sum_abs for high/low score.",
    )
    ap.add_argument("--ig_steps", type=int, default=0, help="0 -> auto from run.json, else override.")
    ap.add_argument("--incorrect_agg", type=str, default="auto", choices=["auto", "max", "mean", "min"])
    ap.add_argument("--model_name", type=str, default="", help="Empty (default) = auto from circuit .run.json;") # "EleutherAI/pythia-1b"
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="", choices=["", "float16", "bfloat16", "float32"])
    ap.add_argument("--max_samples_per_case", type=int, default=500, help="0 means all.")
    ap.add_argument("--max_samples_per_bucket", type=int, default=0, help="0 means all.")
    ap.add_argument("--random_baseline_trials", type=int, default=150, help=("0 disables random baseline. If >0, build one random-edge baseline (subset size from topk setting), sample N trials, and estimate correlation distribution."),)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", dest="resume", action="store_true")
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--skip_existing", action="store_true", help="Skip case if all artifacts already exist.")
    ap.add_argument("--quiet", action="store_true")
    # ap.set_defaults(sum_abs=False)
    ap.set_defaults(resume=True)
    ap.set_defaults(topk_rank_abs=True)
    return ap

def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.score_weighted_direction_aware and args.sum_abs and (not args.quiet):
        print("[WARN] --score_weighted_direction_aware is enabled, so --sum_abs is ignored for high/low score.")

    root = Path(args.root)
    circuits_root = Path(args.circuits_root) if args.circuits_root else root / "circuit_results"
    buckets_root = Path(args.buckets_root) if args.buckets_root else root / "data_gens" / "buckets"
    pairs_root = Path(args.pairs_root) if args.pairs_root else root / "data_gens" / "pairs"
    out_root = Path(args.out_root) if args.out_root else root / "results" / "validation" / "scatter"
    ensure_dir(out_root)

    datasets_filter = set(parse_csv_list(args.datasets))
    score_keys_filter = set(parse_csv_list(args.score_keys))
    if not score_keys_filter:
        score_keys_filter = set(SCORE_KEYS_TD)

    cases, warnings = discover_cases(
        circuits_root=circuits_root,
        buckets_root=buckets_root,
        pairs_root=pairs_root,
        datasets_filter=datasets_filter,
        score_keys_filter=score_keys_filter,
        corrupt_type=args.corrupt_type,
    )
    if not cases:
        msg = {
            "status": "no_case",
            "warnings": warnings,
            "inputs": {
                "circuits_root": str(circuits_root),
                "buckets_root": str(buckets_root),
                "pairs_root": str(pairs_root),
                "datasets_filter": sorted(datasets_filter),
                "score_keys_filter": sorted(score_keys_filter),
                "corrupt_type": args.corrupt_type,
                "eval_mode": args.eval_mode,
                "validation_buckets": args.validation_buckets,
                "random_baseline_trials": args.random_baseline_trials,
            },
        }
        write_json(msg, out_root / "summary.json")
        raise RuntimeError("No valid cases found. Check paths/filters.")

    torch.backends.cuda.matmul.allow_tf32 = True
    device = args.device
    dtype = _dtype_from_args(device, args.dtype)

    highlow_score_mode = resolve_highlow_score_mode(args)
    print(
        f"[INFO] score_mode={highlow_score_mode} (sum_abs={args.sum_abs}, "
        f"weighted_direction_aware={args.score_weighted_direction_aware}), max_num={args.max_samples_per_case}"
    )

    current_model_name: Optional[str] = None
    model = None
    reports: List[Dict[str, Any]] = []
    iterator = cases if args.quiet else tqdm(cases, desc="cases")
    for case in iterator:
        runtime_cfg = resolve_runtime_config(case, args)
        if model is None or runtime_cfg.model_name != current_model_name:
            if model is not None:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            model = load_model(runtime_cfg.model_name, device, dtype)
            model.eval()
            current_model_name = runtime_cfg.model_name

        report = process_case(
            case=case,
            args=args,
            model=model,
            runtime_cfg=runtime_cfg,
            buckets_root=buckets_root,
            pairs_root=pairs_root,
            out_root=out_root,
        )
        reports.append(report)

    summary = {
        "status": "ok",
        "num_cases": len(cases),
        "warnings": warnings,
        "settings": {
            "circuits_root": str(circuits_root),
            "buckets_root": str(buckets_root),
            "pairs_root": str(pairs_root),
            "out_root": str(out_root),
            "device": device,
            "dtype": str(dtype),
            "eval_mode": args.eval_mode,
            "validation_buckets": args.validation_buckets,
            "topk_edges": args.topk_edges,
            "topk_rank_abs": args.topk_rank_abs,
            "sum_abs": args.sum_abs,
            "score_weighted_direction_aware": args.score_weighted_direction_aware,
            "score_mode": highlow_score_mode,
            "ig_steps_arg": args.ig_steps,
            "incorrect_agg_arg": args.incorrect_agg,
            "seed": args.seed,
            "max_samples_per_case": args.max_samples_per_case,
            "max_samples_per_bucket": args.max_samples_per_bucket,
            "random_baseline_trials": args.random_baseline_trials,
        },
        "reports": reports,
    }
    run_tag = build_run_summary_tag(
        datasets_filter=datasets_filter,
        score_keys_filter=score_keys_filter,
        eval_mode=args.eval_mode,
        validation_buckets=args.validation_buckets,
    )
    run_summary_json = out_root / f"summary_{run_tag}.json"
    run_summary_csv = out_root / f"summary_table_{run_tag}.csv"
    write_json(summary, run_summary_json)
    pd.DataFrame(build_summary_rows_from_reports(reports)).to_csv(run_summary_csv, index=False)

    # Backward-compatible aggregate files:
    # Rebuild from all per-case metrics json (latest per dataset/score_key) to avoid
    # "last dataset wins" when jobs are split by dataset and share out_root.
    aggregate_reports = collect_latest_reports_from_metrics(out_root)
    if not aggregate_reports:
        aggregate_reports = reports
    aggregate_summary = dict(summary)
    aggregate_summary["num_cases"] = int(len(aggregate_reports))
    aggregate_summary["reports"] = aggregate_reports
    aggregate_summary["aggregation"] = {
        "method": "latest_metrics_json_per_dataset_score_key",
        "source_glob": "*_metrics.json",
        "num_reports_aggregated": int(len(aggregate_reports)),
        "run_summary_json": str(run_summary_json),
        "run_summary_csv": str(run_summary_csv),
    }
    write_json(aggregate_summary, out_root / "summary.json")
    pd.DataFrame(build_summary_rows_from_reports(aggregate_reports)).to_csv(
        out_root / "summary_table.csv", index=False
    )


if __name__ == "__main__":
    main()
