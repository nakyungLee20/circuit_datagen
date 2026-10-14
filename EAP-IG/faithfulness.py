import argparse
import gc
import glob
import json
import os
import re
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CUR_DIR)
_EAPIG_ROOT = os.path.join(_ROOT, "EAP-IG")
_EAP_SRC = os.path.join(_EAPIG_ROOT, "src")
for _p in (_ROOT, _EAPIG_ROOT, _EAP_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scen1_eapig import (
    PairsDatasetRuntimeCorrupt,
    collate_pairs,
    evaluate_baseline_runtime_corrupt,
    infer_dataset_from_rows,
    load_model,
    logit_diff_mc_tensor,
    read_jsonl,
)
import scen1_eapig as scen1_mod
from eap.evaluate import evaluate_graph
from eap.graph import Graph

DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
RUN_RE = re.compile(r"_m(?P<m>\d+)_top(?P<top>\d+)_")
KNOWN_SCORE_KEYS = ["gradlogits_l2", "grad_align", "forget", "tracin", "el2n", "aum"]
KNOWN_BUCKETS = ["high", "low"]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            v = int(s)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in csv list: {s!r}") from exc
        if v > 0:
            out.append(v)
    return sorted(set(out))

def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if abs(den) < eps:
        return float("nan")
    return float(num / den)

def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _patch_tokenize_plus_maxlen(max_len: int) -> None:
    if max_len <= 0:
        return
    raw_tokenize_plus = scen1_mod._ORIG_TOKENIZE_PLUS

    def _tokenize_plus_limited(model, inputs, max_length=None):
        eff_max = max_length if max_length is not None else max_len
        return raw_tokenize_plus(model, inputs, max_length=eff_max)

    scen1_mod._ORIG_TOKENIZE_PLUS = _tokenize_plus_limited

def _parse_pairs_filename(path: str) -> Optional[Dict[str, str]]:
    base = os.path.basename(path)
    if not (base.startswith("pairs_") and base.endswith(".jsonl")):
        return None
    stem = base[:-6]
    body = stem[len("pairs_") :]

    for bucket in KNOWN_BUCKETS:
        marker = f"_{bucket}_"
        bucket_pos = body.rfind(marker)
        if bucket_pos < 0:
            continue
        left = body[:bucket_pos]
        corrupt_type = body[bucket_pos + len(marker) :]
        if not left or not corrupt_type:
            continue
        for score_key in KNOWN_SCORE_KEYS:
            score_suffix = f"_{score_key}"
            if not left.endswith(score_suffix):
                continue
            dataset = left[: -len(score_suffix)]
            if not dataset:
                continue
            return {
                "dataset": dataset,
                "score_key": score_key,
                "bucket": bucket,
                "corrupt_type": corrupt_type,
                "tag": body,
            }
    return None

def resolve_pairs_files(pairs_dir: str) -> List[Tuple[Dict[str, str], str]]:
    pattern = os.path.join(pairs_dir, "**", "pairs_*.jsonl")
    paths = sorted(glob.glob(pattern, recursive=True))
    unique: Dict[Tuple[str, str, str, str], Tuple[Dict[str, str], str]] = {}
    for path in paths:
        meta = _parse_pairs_filename(path)
        if meta is None:
            continue
        key = (
            meta["dataset"],
            meta["score_key"],
            meta["bucket"],
            meta["corrupt_type"],
        )
        prev = unique.get(key)
        if prev is None or os.path.getmtime(path) >= os.path.getmtime(prev[1]):
            unique[key] = (meta, path)
    return sorted(unique.values(), key=lambda x: x[1])

def _score_graph_candidate(path: str) -> Tuple[int, int, float]:
    base = os.path.basename(path)
    match = RUN_RE.search(base)
    if match is None:
        return (-1, -1, os.path.getmtime(path))
    return (
        int(match.group("top")),
        int(match.group("m")),
        os.path.getmtime(path),
    )

def find_graph_for_meta(meta: Dict[str, str], circuits_root: str) -> Optional[str]:
    search_dirs = [
        os.path.join(
            circuits_root,
            meta["dataset"],
            meta["score_key"],
            meta["bucket"],
            meta["corrupt_type"],
        ),
        os.path.join(
            circuits_root,
            "data_attr",
            meta["dataset"],
            meta["score_key"],
            meta["bucket"],
            meta["corrupt_type"],
        ),
    ]
    candidates: List[str] = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        pts = glob.glob(os.path.join(d, f"eapig_{meta['tag']}_*.pt"))
        if not pts:
            pts = glob.glob(os.path.join(d, "eapig_*.pt"))
        if not pts:
            pts = glob.glob(os.path.join(d, "*.pt"))
        candidates.extend(pts)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=_score_graph_candidate, reverse=True)
    return candidates[0]

def build_targets(args: argparse.Namespace) -> List[Tuple[Dict[str, str], str, str]]:
    if bool(args.pairs_path) != bool(args.graph_path):
        raise ValueError("--pairs_path and --graph_path must be provided together.")

    if args.pairs_path and args.graph_path:
        meta = _parse_pairs_filename(args.pairs_path) or {
            "dataset": args.dataset or "",
            "score_key": args.score_key or "",
            "bucket": args.bucket or "",
            "corrupt_type": args.corrupt_type or "",
            "tag": os.path.splitext(os.path.basename(args.pairs_path))[0].replace("pairs_", ""),
        }
        return [(meta, args.pairs_path, args.graph_path)]

    targets: List[Tuple[Dict[str, str], str, str]] = []
    for meta, pairs_path in resolve_pairs_files(args.pairs_dir):
        if args.dataset and meta["dataset"] != args.dataset:
            continue
        if args.score_key and meta["score_key"] != args.score_key:
            continue
        if args.bucket and meta["bucket"] != args.bucket:
            continue
        if args.corrupt_type and meta["corrupt_type"] != args.corrupt_type:
            continue
        graph_path = find_graph_for_meta(meta, args.circuits_root)
        if graph_path is None:
            continue
        targets.append((meta, pairs_path, graph_path))
    return targets

def validate_args(args: argparse.Namespace) -> None:
    if args.skip_saved_mask and not parse_csv_ints(args.topk_list):
        raise ValueError("No circuit evaluations requested. Remove --skip_saved_mask or provide --topk_list.")
    if int(args.random_trials) > 0 and args.topk_level != "edge":
        raise ValueError("--random_trials currently supports only edge-level circuit comparisons.")

def normalize_gqa_mode(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if s == "mha":
        return "auto"
    if s in {"auto", "native", "ungrouped"}:
        return s
    return None

def infer_gqa_mode_from_graph_path(graph_path: str) -> str:
    run_json = graph_path[:-3] + ".run.json" if graph_path.endswith(".pt") else graph_path + ".run.json"
    if os.path.exists(run_json):
        try:
            with open(run_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            mode = normalize_gqa_mode(obj.get("gqa_mode"))
            if mode is None and isinstance(obj.get("model_circuit_summary"), dict):
                mode = normalize_gqa_mode(obj["model_circuit_summary"].get("eap_actual_gqa_mode"))
            if mode is not None:
                return mode
        except Exception:
            pass

    graph = Graph.from_pt(graph_path)
    graph_n_heads = int(graph.cfg.get("n_heads"))
    graph_n_kv_heads = int(graph.cfg.get("n_kv_heads", graph.cfg.get("n_key_value_heads", graph_n_heads)))
    ungroup = bool(graph.cfg.get("ungroup_grouped_query_attention", False))
    if graph_n_kv_heads != graph_n_heads and not ungroup:
        return "native"
    if ungroup:
        return "ungrouped"
    return "auto"

def resolve_gqa_mode(requested_mode: str, targets: List[Tuple[Dict[str, str], str, str]]) -> str:
    explicit = normalize_gqa_mode(requested_mode)
    if explicit is not None:
        return explicit

    inferred_modes = {}
    for meta, _, graph_path in targets:
        tag = meta.get("tag") or os.path.basename(graph_path)
        inferred_modes[tag] = infer_gqa_mode_from_graph_path(graph_path)

    unique_modes = sorted(set(inferred_modes.values()))
    if len(unique_modes) != 1:
        raise ValueError(f"Mixed GQA modes across selected targets: {inferred_modes}")
    return unique_modes[0]

def load_eval_dataloader(model, pairs_path: str, batch_size: int, max_items: int,) -> Tuple[DataLoader, str]:
    rows = read_jsonl(pairs_path)
    dataset_name = infer_dataset_from_rows(rows, default="piqa")
    ds = PairsDatasetRuntimeCorrupt(
        rows=rows,
        tokenizer=model.tokenizer,
        dataset=dataset_name,
        require_equal_length=True,
        max_items=(max_items if max_items > 0 else None),
        strict_label_token=True,
        debug_first=False,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs, num_workers=0,)
    return dl, dataset_name

def load_or_compute_baseline(baseline_json_path: str, model, dataloader: DataLoader, metric_eval, pairs_path: str,
    dtype: torch.dtype, batch_size: int, max_seq_len: int, quiet: bool, force: bool,) -> Tuple[float, float]:
    baseline_clean: Optional[float] = None
    baseline_corrupt: Optional[float] = None
    if (not force) and os.path.exists(baseline_json_path):
        try:
            with open(baseline_json_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            same_pairs = str(old.get("pairs_path", "")) == str(pairs_path)
            same_seq = int(old.get("max_seq_len", max_seq_len)) == int(max_seq_len)
            same_dtype = str(old.get("dtype", str(dtype))) == str(dtype)
            if same_pairs and same_seq and same_dtype:
                baseline_clean = float(old.get("baseline_clean"))
                baseline_corrupt = float(old.get("baseline_corrupt"))
        except Exception:
            baseline_clean = None
            baseline_corrupt = None

    if baseline_clean is None or baseline_corrupt is None:
        model.reset_hooks(including_permanent=True)
        baseline_clean = evaluate_baseline_runtime_corrupt(model, dataloader, metric_eval, run_corrupted=False, quiet=quiet,).mean().item()
        _cleanup_cuda()
        baseline_corrupt = evaluate_baseline_runtime_corrupt(model, dataloader, metric_eval, run_corrupted=True, quiet=quiet,).mean().item()
        _cleanup_cuda()
        with open(baseline_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pairs_path": pairs_path,
                    "baseline_clean": baseline_clean,
                    "baseline_corrupt": baseline_corrupt,
                    "dtype": str(dtype),
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    return float(baseline_clean), float(baseline_corrupt)

def build_saved_graph(graph_path: str) -> Graph:
    g = Graph.from_pt(graph_path)
    g.prune()
    return g

def build_topk_graph(graph_path: str, topk: int, absolute: bool, level: str,) -> Graph:
    g = Graph.from_pt(graph_path)
    max_edges = int(g.real_edge_mask.sum().item())
    k = min(int(topk), max_edges)
    g.apply_topn(k, absolute=absolute, level=level, reset=True, prune=True)
    return g

def build_random_same_size_graph(graph_path: str, topk: int, seed: int,) -> Graph:
    g = Graph.from_pt(graph_path)
    real_edges = torch.nonzero(g.real_edge_mask, as_tuple=False)
    if real_edges.numel() == 0:
        raise RuntimeError(f"No real edges in graph: {graph_path}")
    k = min(int(topk), int(real_edges.shape[0]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chosen = real_edges[torch.randperm(real_edges.shape[0], generator=generator)[:k]]
    g.reset(empty=True)
    g.in_graph[chosen[:, 0], chosen[:, 1]] = True
    g.nodes_in_graph[:] = True
    if g.neurons_in_graph is not None:
        g.neurons_in_graph[:] = True
    g.prune()
    return g

def build_complement_graph(graph_path: str, circuit_graph: Graph) -> Graph:
    g = Graph.from_pt(graph_path)
    g.in_graph[:] = g.real_edge_mask & (~circuit_graph.in_graph.bool())
    g.nodes_in_graph[:] = True
    if g.neurons_in_graph is not None:
        g.neurons_in_graph[:] = True
    g.prune()
    return g

def evaluate_graph_mean(model, graph: Graph, dataloader: DataLoader, metric_eval, quiet: bool, intervention: str,) -> float:
    model.reset_hooks(including_permanent=True)
    vals = evaluate_graph(model,graph, dataloader, metrics=metric_eval, quiet=quiet, intervention=intervention,)
    out = float(vals.mean().item())
    del vals
    _cleanup_cuda()
    return out

def make_faithfulness_row(
    meta: Dict[str, str],
    pairs_path: str,
    graph_path: str,
    eval_name: str,
    eval_type: str,
    requested_topk: Optional[int],
    trial: Optional[int],
    baseline_clean: float,
    baseline_corrupt: float,
    circuit_value: float,
    included_edges: int,
    included_nodes: int,
    random_parent: str = "",
) -> Dict[str, Any]:
    denom = abs(baseline_clean) if baseline_clean != 0 else 1e-9
    recovery_denom = baseline_clean - baseline_corrupt
    return {
        "dataset": meta.get("dataset", ""),
        "score_key": meta.get("score_key", ""),
        "bucket": meta.get("bucket", ""),
        "corrupt_type": meta.get("corrupt_type", ""),
        "eval_name": eval_name,
        "eval_type": eval_type,
        "random_parent_eval": random_parent,
        "requested_topk": (int(requested_topk) if requested_topk is not None else np.nan),
        "trial": (int(trial) if trial is not None else np.nan),
        "included_nodes": int(included_nodes),
        "included_edges": int(included_edges),
        "F_full_model": float(baseline_clean),
        "F_corrupt": float(baseline_corrupt),
        "F_circuit": float(circuit_value),
        "faithfulness_score": float(1.0 - (abs(baseline_clean - circuit_value) / denom)),
        "recovery_score": float((circuit_value - baseline_corrupt) / recovery_denom) if recovery_denom != 0 else float("nan"),
        "retention_ratio": safe_ratio(circuit_value, baseline_clean),
        "drop_from_full": float(baseline_clean - circuit_value),
        "pairs_path": pairs_path,
        "graph_path": graph_path,
    }

def make_necessity_row(
    meta: Dict[str, str],
    pairs_path: str,
    graph_path: str,
    source_eval_name: str,
    requested_topk: Optional[int],
    baseline_clean: float,
    baseline_corrupt: float,
    complement_value: float,
    included_edges: int,
    included_nodes: int,
    intervention: str,
) -> Dict[str, Any]:
    return {
        "dataset": meta.get("dataset", ""),
        "score_key": meta.get("score_key", ""),
        "bucket": meta.get("bucket", ""),
        "corrupt_type": meta.get("corrupt_type", ""),
        "source_eval_name": source_eval_name,
        "requested_topk": (int(requested_topk) if requested_topk is not None else np.nan),
        "intervention": intervention,
        "included_nodes": int(included_nodes),
        "included_edges": int(included_edges),
        "F_full_model": float(baseline_clean),
        "F_corrupt": float(baseline_corrupt),
        "F_complement": float(complement_value),
        "drop_from_full": float(baseline_clean - complement_value),
        "pairs_path": pairs_path,
        "graph_path": graph_path,
    }

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=("Faithfulness-only evaluation for saved circuits. "
            "Focuses on full-model vs circuit-only performance, with optional random same-size controls and an optional lightweight complement check.")
    )
    ap.add_argument("--pairs_dir", type=str, default=os.path.join(_ROOT, "data_gens", "pairs"))
    ap.add_argument("--circuits_root", type=str, default=os.path.join(_ROOT, "circuit_results"))
    ap.add_argument("--pairs_path", type=str, default="")
    ap.add_argument("--graph_path", type=str, default="")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--score_key", type=str, default="")
    ap.add_argument("--bucket", type=str, default="")
    ap.add_argument("--corrupt_type", type=str, default="")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gqa_mode", type=str, default="", choices=["", "auto", "native", "ungrouped"])
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", type=str, default=os.environ.get("EAP_DTYPE", "bfloat16"))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("EAP_BATCH_SIZE", "4")))
    ap.add_argument("--max_seq_len", type=int, default=int(os.environ.get("EAP_MAX_SEQ_LEN", "768")))
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--incorrect_agg", type=str, default="max", choices=["max", "mean", "min"])

    ap.add_argument("--topk_list", type=str, default="", help="Comma-separated top-k sizes for circuit sweeps.")
    ap.add_argument("--topk_level", type=str, default="edge", choices=["edge", "node", "neuron"])
    ap.add_argument("--no_topk_absolute", dest="topk_absolute", action="store_false")
    ap.set_defaults(topk_absolute=True)
    ap.add_argument("--skip_saved_mask", action="store_true", help="Skip the saved in_graph mask evaluation.")
    ap.add_argument("--random_trials", type=int, default=0, help="Optional random same-size circuit trials per evaluated circuit.")
    ap.add_argument("--random_seed", type=int, default=42)

    ap.add_argument("--include_complement_eval", action="store_true", help="Also evaluate the complement graph (lightweight necessity check).",)
    ap.add_argument("--complement_intervention", type=str, default="patching", choices=["patching", "zero"], help="Intervention used only for the optional complement evaluation.",)
    ap.add_argument("--out_dir", type=str, default=os.path.join(_ROOT, "results", "validation", "faithfulness"))
    ap.add_argument("--force_recompute_baseline", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_dir(args.out_dir)
    torch.manual_seed(0)
    np.random.seed(0)

    dtype_name = str(args.dtype).strip().lower()
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype = DTYPE_MAP[dtype_name]
    if (not args.device.startswith("cuda")) and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    _patch_tokenize_plus_maxlen(int(args.max_seq_len))
    targets = build_targets(args)
    if not targets:
        raise RuntimeError("No evaluation targets found. Check filters, pairs paths, or circuits_root.")

    resolved_gqa_mode = resolve_gqa_mode(args.gqa_mode, targets)
    print(f"[INFO] targets={len(targets)} gqa_mode={resolved_gqa_mode}")
    model = load_model(args.model_name, args.device, dtype, gqa_mode=resolved_gqa_mode)
    model.eval()
    metric_eval = partial(logit_diff_mc_tensor, loss=False, mean=False, incorrect_agg=args.incorrect_agg,)

    all_faithfulness_rows: List[Dict[str, Any]] = []
    all_necessity_rows: List[Dict[str, Any]] = []
    for meta, pairs_path, graph_path in targets:
        tag = meta.get("tag") or os.path.splitext(os.path.basename(pairs_path))[0].replace("pairs_", "")
        target_dir = os.path.join(args.out_dir, tag)
        ensure_dir(target_dir)
        baseline_json = os.path.join(target_dir, "baseline.json")
        faithfulness_csv = os.path.join(target_dir, "faithfulness.csv")
        necessity_csv = os.path.join(target_dir, "necessity.csv")
        summary_json = os.path.join(target_dir, "summary.json")

        dataloader, dataset_name = load_eval_dataloader(model=model, pairs_path=pairs_path, batch_size=int(args.batch_size), max_items=int(args.max_items),)
        baseline_clean, baseline_corrupt = load_or_compute_baseline(
            baseline_json_path=baseline_json,
            model=model,
            dataloader=dataloader,
            metric_eval=metric_eval,
            pairs_path=pairs_path,
            dtype=dtype,
            batch_size=int(args.batch_size),
            max_seq_len=int(args.max_seq_len),
            quiet=bool(args.quiet),
            force=bool(args.force_recompute_baseline),
        )
        saved_graph = build_saved_graph(graph_path)
        saved_edges = int(saved_graph.count_included_edges())
        max_real_edges = int(saved_graph.real_edge_mask.sum().item())
        topk_list = [k for k in parse_csv_ints(args.topk_list) if k <= max_real_edges]

        print(
            f"[RUN] tag={tag} dataset={dataset_name} "
            f"baseline(clean)={baseline_clean:.6f} baseline(corrupt)={baseline_corrupt:.6f} "
            f"saved_edges={saved_edges}"
        )

        target_faithfulness_rows: List[Dict[str, Any]] = []
        target_necessity_rows: List[Dict[str, Any]] = []
        eval_specs: List[Tuple[str, str, Optional[int], Graph]] = []
        if not args.skip_saved_mask:
            eval_specs.append(("saved_mask", "saved_mask", saved_edges, saved_graph))
        for k in topk_list:
            eval_specs.append((f"topk_{k}", "topk", int(k), build_topk_graph(graph_path, k, args.topk_absolute, args.topk_level)))

        for eval_name, eval_type, requested_topk, graph in eval_specs:
            circuit_value = evaluate_graph_mean(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_eval=metric_eval,
                quiet=bool(args.quiet),
                intervention="patching",
            )
            included_edges = int(graph.count_included_edges())
            included_nodes = int(graph.count_included_nodes())
            row = make_faithfulness_row(
                meta=meta,
                pairs_path=pairs_path,
                graph_path=graph_path,
                eval_name=eval_name,
                eval_type=eval_type,
                requested_topk=requested_topk,
                trial=None,
                baseline_clean=baseline_clean,
                baseline_corrupt=baseline_corrupt,
                circuit_value=circuit_value,
                included_edges=included_edges,
                included_nodes=included_nodes,
            )
            target_faithfulness_rows.append(row)
            print(
                f"  [circuit] {eval_name} "
                f"edges={included_edges} nodes={included_nodes} "
                f"F(circuit)={circuit_value:.6f} faithfulness={row['faithfulness_score']:.4f}"
            )

            if args.include_complement_eval:
                complement_graph = build_complement_graph(graph_path, graph)
                complement_value = evaluate_graph_mean(
                    model=model,
                    graph=complement_graph,
                    dataloader=dataloader,
                    metric_eval=metric_eval,
                    quiet=bool(args.quiet),
                    intervention=args.complement_intervention,
                )
                necessity_row = make_necessity_row(
                    meta=meta,
                    pairs_path=pairs_path,
                    graph_path=graph_path,
                    source_eval_name=eval_name,
                    requested_topk=requested_topk,
                    baseline_clean=baseline_clean,
                    baseline_corrupt=baseline_corrupt,
                    complement_value=complement_value,
                    included_edges=int(complement_graph.count_included_edges()),
                    included_nodes=int(complement_graph.count_included_nodes()),
                    intervention=args.complement_intervention,
                )
                target_necessity_rows.append(necessity_row)
                print(
                    f"  [complement] {eval_name} "
                    f"intervention={args.complement_intervention} "
                    f"F(complement)={complement_value:.6f}"
                )

            if int(args.random_trials) > 0 and requested_topk is not None and requested_topk > 0:
                for trial in range(int(args.random_trials)):
                    random_graph = build_random_same_size_graph(
                        graph_path=graph_path,
                        topk=int(requested_topk),
                        seed=int(args.random_seed) + trial,
                    )
                    random_value = evaluate_graph_mean(
                        model=model,
                        graph=random_graph,
                        dataloader=dataloader,
                        metric_eval=metric_eval,
                        quiet=bool(args.quiet),
                        intervention="patching",
                    )
                    random_row = make_faithfulness_row(
                        meta=meta,
                        pairs_path=pairs_path,
                        graph_path=graph_path,
                        eval_name=f"{eval_name}_random_t{trial}",
                        eval_type="random_same_size",
                        requested_topk=requested_topk,
                        trial=trial,
                        baseline_clean=baseline_clean,
                        baseline_corrupt=baseline_corrupt,
                        circuit_value=random_value,
                        included_edges=int(random_graph.count_included_edges()),
                        included_nodes=int(random_graph.count_included_nodes()),
                        random_parent=eval_name,
                    )
                    target_faithfulness_rows.append(random_row)
                    print(
                        f"  [random] parent={eval_name} trial={trial} "
                        f"edges={random_row['included_edges']} "
                        f"F(circuit)={random_value:.6f}"
                    )

        faithfulness_df = pd.DataFrame(target_faithfulness_rows)
        faithfulness_df.to_csv(faithfulness_csv, index=False)
        all_faithfulness_rows.extend(target_faithfulness_rows)
        necessity_path_out: Optional[str] = None
        if target_necessity_rows:
            necessity_df = pd.DataFrame(target_necessity_rows)
            necessity_df.to_csv(necessity_csv, index=False)
            all_necessity_rows.extend(target_necessity_rows)
            necessity_path_out = necessity_csv

        summary = {
            "tag": tag,
            "dataset_name": dataset_name,
            "meta": meta,
            "paths": {
                "pairs_path": pairs_path,
                "graph_path": graph_path,
                "baseline_json": baseline_json,
                "faithfulness_csv": faithfulness_csv,
                "necessity_csv": necessity_path_out,
            },
            "baseline": {
                "F_full_model": baseline_clean,
                "F_corrupt": baseline_corrupt,
            },
            "saved_graph": {
                "included_edges": saved_edges,
                "included_nodes": int(saved_graph.count_included_nodes()),
                "max_real_edges": max_real_edges,
            },
            "config": vars(args),
            "notes": {
                "focus": "faithfulness-only",
                "baseline_definition": (
                    "Uses evaluate_baseline_runtime_corrupt for paired clean/corrupted runtime corruption. "
                    "This is the scen1 analogue of the EAP-IG notebook baseline-vs-circuit check."
                ),
                "omitted_as_redundant": [
                    "td-style checkpoint mediation summaries",
                    "zero_ablation-style scatter and bucket-gap reports",
                    "structure-only outputs",
                ],
            },
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    global_faithfulness_csv = os.path.join(args.out_dir, "global_faithfulness.csv")
    pd.DataFrame(all_faithfulness_rows).to_csv(global_faithfulness_csv, index=False)
    print(f"[OK] saved {global_faithfulness_csv}")
    if all_necessity_rows:
        global_necessity_csv = os.path.join(args.out_dir, "global_necessity.csv")
        pd.DataFrame(all_necessity_rows).to_csv(global_necessity_csv, index=False)
        print(f"[OK] saved {global_necessity_csv}")
    model.reset_hooks(including_permanent=True)


if __name__ == "__main__":
    main()
