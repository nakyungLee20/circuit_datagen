"""Exp 4-a(ii): random-circuit null for the per-example attribution correlation.

Why not reuse the random baseline of ``applications/circuit_scores.py``
(``--score_random_baseline_trials``): it is recorded as
``random_score_mode = "mean_raw_signed_sum"``, which differs from the statistic
under test in two ways that both matter here.

  1. ``sum_random_subset_scores`` sums raw, unweighted EAP-IG
     attributions, while ``high_circuit_score`` is a *weighted* sum with the
     circuit's own edge scores. An unweighted sum is a different estimator, so
     its correlation is not a null for the weighted one.
  2. Only ``np.mean`` over trials is stored. Correlating a trial-averaged score
     destroys exactly the trial-to-trial variance the null is made of; the
     needed object is the distribution of per-trial correlations.

This script keeps the attribution pass identical to the original run and stores
the full per-trial matrix, so the null distribution is computed offline.

Two nulls, both weighted so they are directly comparable to the real statistic:

  perm   weight-preserving permutation. The circuit's own 250 edge weights are
         reattached to 250 uniformly drawn real edges. Isolates *which edges*
         from *how the weights are distributed* -- the null for "this specific
         edge set carries the signal".
  top15k rank-matched. 250 edges are drawn from the same circuit's top-15000
         pool and weighted by their own stored scores. The null for "the top-250
         are special among plausible circuit edges".
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
import json
import sys
import time
import zlib
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "EAP-IG")):
    if _p not in sys.path:
        sys.path.append(_p)

from applications import circuit_scores as A  # noqa: E402
from scen1_eapig import (  # type: ignore  # noqa: E402
    PairsDatasetRuntimeCorrupt,
    collate_pairs,
    load_model,
    logit_diff_mc_tensor,
)
from eap.attribute import attribute  # type: ignore  # noqa: E402
from eap.graph import Graph  # type: ignore  # noqa: E402


def load_target_ids(paths: Sequence[str]) -> Optional[List[str]]:
    """Union of hf_ids from bucket JSONs; None means the full split."""
    if not paths:
        return None
    ids: List[str] = []
    seen = set()
    for text in paths:
        obj = json.loads(Path(text).read_text(encoding="utf-8"))
        items = obj["items"] if isinstance(obj, dict) and "items" in obj else obj
        for row in items:
            hf_id = str(row["hf_id"]).strip()
            if hf_id and hf_id not in seen:
                seen.add(hf_id)
                ids.append(hf_id)
    return ids


def build_null_subsets(
    graph: Graph,
    circuit_weights: torch.Tensor,
    pool_rows: torch.Tensor,
    pool_cols: torch.Tensor,
    pool_weights: Optional[torch.Tensor],
    n_trials: int,
    subset_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (rows, cols, weights) each shaped [n_trials, subset_size]."""
    rng = np.random.default_rng(seed)
    n_pool = int(pool_rows.numel())
    if n_pool < subset_size:
        raise ValueError(f"pool has {n_pool} edges, need {subset_size}")
    rows = torch.empty((n_trials, subset_size), dtype=torch.long, device=device)
    cols = torch.empty((n_trials, subset_size), dtype=torch.long, device=device)
    weights = torch.empty((n_trials, subset_size), dtype=torch.float32, device=device)
    base = circuit_weights.detach().cpu().numpy()
    for t in range(n_trials):
        idx = rng.choice(n_pool, size=subset_size, replace=False)
        rows[t] = pool_rows[idx]
        cols[t] = pool_cols[idx]
        if pool_weights is None:
            # Weight-preserving permutation: same multiset, shuffled assignment.
            weights[t] = torch.tensor(
                rng.permutation(base), dtype=torch.float32, device=device
            )
        else:
            weights[t] = pool_weights[idx]
    return rows, cols, weights


def score_subsets(
    scores_matrix: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    weights: torch.Tensor,
) -> np.ndarray:
    vals = scores_matrix[rows, cols]
    vals = vals * weights.to(device=vals.device, dtype=vals.dtype)
    return vals.sum(dim=1).detach().float().cpu().numpy().astype(np.float64)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sciq")
    parser.add_argument("--score-key", required=True, choices=("aum", "el2n", "grad_align"))
    parser.add_argument("--bucket-json", nargs="*", default=[],
                        help="Restrict to these hf_ids (union). Omit for the full split.")
    parser.add_argument("--circuits-root",
                        default=str(_REPO_ROOT / 'circuit_results/Qwen__Qwen2.5-1.5B-Instruct'))
    parser.add_argument("--buckets-root",
                        default=str(_REPO_ROOT / 'data_gens/buckets/application/Qwen__Qwen2.5-1.5B-Instruct'))
    parser.add_argument("--corrupt-type", default="gaussian")
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--null", choices=("perm", "top15k", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-sigma", type=float, default=0.1)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--incorrect-agg", default="max")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test cap on examples.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.score_key}_top{args.topk_edges}"
    npz_path = out_dir / f"{tag}_null.npz"
    meta_path = out_dir / f"{tag}_null.meta.json"

    circuits_root = Path(args.circuits_root)
    buckets_root = Path(args.buckets_root)
    high_pt = A._find_latest_circuit_pt(
        circuits_root=circuits_root, dataset=args.dataset,
        score_key=args.score_key, bucket="high", corrupt_type=args.corrupt_type,
    )
    if high_pt is None:
        raise FileNotFoundError(f"no high circuit for {args.score_key}")
    print(f"[circuit] {high_pt}")

    runtime_cfg = A._resolve_runtime_config(
        high_run_json=high_pt.with_suffix(".run.json") if high_pt.with_suffix(".run.json").is_file() else None,
        low_run_json=None, model_name_override="",
        ig_steps_override=int(args.ig_steps), incorrect_agg_override=args.incorrect_agg,
    )
    print(f"[runtime] model={runtime_cfg.model_name} ig_steps={runtime_cfg.ig_steps} "
          f"incorrect_agg={runtime_cfg.incorrect_agg}")

    all_scores_json = A._find_all_scores_json(buckets_root, args.dataset, args.score_key)
    if all_scores_json is None:
        raise FileNotFoundError(f"no all-scores json for {args.score_key}")
    original_score_map = A._load_original_score_map(all_scores_json, args.score_key)

    # Identical split construction to build_full_activation_scores.py.
    from data_gens.mcqa import prepare_task_splits  # type: ignore

    splits = prepare_task_splits(args.dataset, seed=int(args.seed), train_frac=0.8)
    split_examples = list(splits[args.split_name])
    print(f"[split] {args.split_name}: {len(split_examples)} examples")

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    score_dtype = A._dtype_from_args(device, args.dtype)
    model = load_model(runtime_cfg.model_name, device, score_dtype)
    model.eval()

    # The published CSVs were produced with app1's row builder monkey-patched to
    # the exact stored utility prompts (request contract: prompt_source =
    # "exact utility JSON prompt/gold_token_id/label_token_ids"). Freshly
    # formatted prompts give different attributions, so reuse that builder here.
    # It validates against the complete train set, hence build first, subset after.
    from selectivity.build_full_activation_scores import (  # type: ignore
        exact_utility_runtime_builder,
        load_exact_utility_rows,
    )

    utility_rows, _source_meta, prompt_sha256 = load_exact_utility_rows(
        Path(all_scores_json), split_examples, args.score_key
    )
    builder = exact_utility_runtime_builder(utility_rows, args.score_key)
    runtime_rows, dropped = builder(
        examples=split_examples, dataset=args.dataset, score_key=args.score_key,
        split_name=args.split_name, tokenizer=model.tokenizer, seed=int(args.seed),
        corrupt_type=args.corrupt_type, noise_sigma=float(args.noise_sigma),
    )
    print(f"[rows] runtime_rows={len(runtime_rows)} dropped={dropped} "
          f"prompt_sha256={prompt_sha256[:16]}")

    keep = load_target_ids(args.bucket_json)
    if keep is not None:
        keep_set = set(keep)
        runtime_rows = [r for r in runtime_rows if str(r.get("hf_id", "")).strip() in keep_set]
        print(f"[subset] restricted to {len(runtime_rows)} rows from {len(args.bucket_json)} bucket file(s)")
    if args.limit > 0:
        runtime_rows = runtime_rows[: args.limit]
        print(f"[limit] smoke test on {len(runtime_rows)} rows")

    # Real circuit: identical selection and weighting to the original run.
    _, _, high_names, high_edge_scores, _ = A.select_edge_indices_from_circuit(
        high_pt, int(args.topk_edges), True
    )
    graph = Graph.from_model(model)
    high_rows, high_cols, _ = A.remap_edge_indices_by_name(high_names, graph)
    score_dev = graph.scores.device
    high_rows, high_cols = high_rows.to(score_dev), high_cols.to(score_dev)
    high_weights = A.build_edge_weight_tensor(high_names, high_edge_scores, graph, score_dev)
    n_edges = int(high_rows.numel())
    print(f"[edges] real circuit edges resolved: {n_edges}")

    case_seed = int(args.seed) + int(zlib.crc32(f"{args.dataset}/{args.score_key}".encode()) & 0xFFFFFFFF)

    nulls: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    if args.null in ("perm", "both"):
        real_r, real_c = torch.where(graph.real_edge_mask)
        nulls["perm"] = build_null_subsets(
            graph, high_weights, real_r.to(score_dev), real_c.to(score_dev), None,
            args.n_trials, n_edges, case_seed + 101, score_dev,
        )
        print(f"[null:perm] pool={int(real_r.numel())} real edges, trials={args.n_trials}")
    if args.null in ("top15k", "both"):
        _, _, pool_names, pool_scores, _ = A.select_edge_indices_from_circuit(
            high_pt, 15000, True
        )
        p_rows, p_cols, _ = A.remap_edge_indices_by_name(pool_names, graph)
        p_w = A.build_edge_weight_tensor(pool_names, pool_scores, graph, score_dev)
        nulls["top15k"] = build_null_subsets(
            graph, high_weights, p_rows.to(score_dev), p_cols.to(score_dev), p_w,
            args.n_trials, n_edges, case_seed + 202, score_dev,
        )
        print(f"[null:top15k] pool={int(p_rows.numel())} edges, trials={args.n_trials}")

    metric_attr = partial(
        logit_diff_mc_tensor, loss=True, mean=True,
        incorrect_agg=runtime_cfg.incorrect_agg,
    )

    hf_ids: List[str] = []
    real_scores: List[float] = []
    true_scores: List[float] = []
    null_scores: Dict[str, List[np.ndarray]] = {k: [] for k in nulls}
    skipped = 0
    started = time.time()

    def flush() -> None:
        payload: Dict[str, Any] = {
            "hf_id": np.array(hf_ids),
            "high_circuit_score": np.array(real_scores, dtype=np.float64),
            "true_score": np.array(true_scores, dtype=np.float64),
        }
        for key, chunks in null_scores.items():
            payload[f"null_{key}"] = np.array(chunks, dtype=np.float32)
        np.savez_compressed(npz_path, **payload)

    for i, row in enumerate(runtime_rows, start=1):
        hf_id = str(row.get("hf_id", "")).strip()
        if not hf_id:
            continue
        try:
            ds = PairsDatasetRuntimeCorrupt(
                [row], tokenizer=model.tokenizer, dataset=args.dataset,
                require_equal_length=True, max_items=1,
                strict_label_token=True, debug_first=False,
            )
            dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pairs,
                            num_workers=0, pin_memory=False)
        except Exception:
            skipped += 1
            continue

        model.reset_hooks(including_permanent=True)
        model.zero_grad(set_to_none=True)
        A.reset_graph_scores(graph)
        try:
            attribute(model, graph, dl, metric_attr, method="EAP-IG-inputs",
                      ig_steps=runtime_cfg.ig_steps, quiet=True)
        except Exception:
            skipped += 1
            model.reset_hooks(including_permanent=True)
            model.zero_grad(set_to_none=True)
            A.reset_graph_scores(graph)
            continue

        scores = graph.scores
        hf_ids.append(hf_id)
        real_scores.append(A._sum_selected_scores(
            scores, high_rows, high_cols, False, edge_weights=high_weights
        ))
        true_scores.append(float(original_score_map.get(hf_id, float("nan"))))
        for key, (r, c, w) in nulls.items():
            null_scores[key].append(score_subsets(scores, r, c, w).astype(np.float32))

        model.reset_hooks(including_permanent=True)
        model.zero_grad(set_to_none=True)
        A.reset_graph_scores(graph)

        if args.save_every > 0 and i % args.save_every == 0:
            flush()
            rate = (time.time() - started) / i
            remaining = rate * (len(runtime_rows) - i)
            print(f"[progress] {i}/{len(runtime_rows)}  {rate:.2f}s/ex  "
                  f"eta={remaining/3600:.2f}h", flush=True)

    flush()
    elapsed = time.time() - started
    meta = {
        "dataset": args.dataset,
        "score_key": args.score_key,
        "circuit_pt": str(high_pt),
        "circuit_pt_bytes": high_pt.stat().st_size,
        "n_examples_requested": len(runtime_rows),
        "n_examples_scored": len(hf_ids),
        "n_skipped": skipped,
        "n_edges": n_edges,
        "n_trials": int(args.n_trials),
        "nulls": sorted(nulls),
        "null_definition": {
            "perm": "circuit's own 250 weights permuted onto 250 uniformly drawn real edges",
            "top15k": "250 edges drawn from the circuit's top-15000 pool, weighted by their own scores",
        },
        "weighted": True,
        "per_trial_stored": True,
        "bucket_json": list(args.bucket_json),
        "model_name": runtime_cfg.model_name,
        "ig_steps": int(runtime_cfg.ig_steps),
        "incorrect_agg": runtime_cfg.incorrect_agg,
        "noise_sigma": float(args.noise_sigma),
        "seed": int(args.seed),
        "elapsed_sec": elapsed,
        "sec_per_example": elapsed / max(1, len(hf_ids)),
        "npz_path": str(npz_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {len(hf_ids)} scored in {elapsed/3600:.2f}h -> {npz_path}")


if __name__ == "__main__":
    main()
