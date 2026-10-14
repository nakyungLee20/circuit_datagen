import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_gens.mcqa import canonical_dataset_name, prepare_task_splits, set_seed

EXPERIMENT_MODES = [
    "full",
    "rand_filter",
    "oris_filter",
    "highc_filter",
    "lowc_filter",
    "bothc_filter",
    "contc_filter",
    "randc_filter",
]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_str() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

def normalize_top_q(raw_q: float) -> float:
    q = float(raw_q)
    if q <= 0:
        raise ValueError(f"top_q must be > 0. got={q}")
    if q > 1.0:
        if q <= 100.0:
            q = q / 100.0
        else:
            raise ValueError(f"top_q should be in (0,1] or (0,100]. got={raw_q}")
    return q

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())

def _resolve_csv_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm_to_actual: Dict[str, str] = {}
    for col in fieldnames:
        key = _normalize_col_name(col)
        if key and key not in norm_to_actual:
            norm_to_actual[key] = col

    for cand in candidates:
        found = norm_to_actual.get(_normalize_col_name(cand))
        if found:
            return found
    return None

def _original_score_candidates(score_key: str) -> List[str]:
    k = (score_key or "").strip()
    out = ["true_score", "original_score", "orig_score", "base_score", "score",]
    if not k:
        return out
    out = [k, f"{k}_score", f"orig_{k}", f"{k}_orig_score", f"{k}_original_score",] + out
    return out

def load_score_csv(path: str, original_score_key: str = "") -> Dict[str, Dict[str, float]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"score csv not found: {p}")
    out: Dict[str, Dict[str, float]] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header columns: {p}")

        hf_id_col = _resolve_csv_col(fieldnames, ["hf_id", "id", "sample_id", "example_id"])
        if not hf_id_col:
            raise ValueError("Could not find id column. expected one of: hf_id/id/sample_id/example_id")
        high_col = _resolve_csv_col(fieldnames, ["high_circuit_score", "highc_score", "high_score", "high_circuit", "high"],)
        low_col = _resolve_csv_col(fieldnames, ["low_circuit_score", "lowc_score", "low_score", "low_circuit", "low"],)
        rand_col = _resolve_csv_col(fieldnames, ["random_mean_score", "randc_score", "rand_circuit_score", "random_circuit_score", "random_score", "rand_score",],)
        orig_col = _resolve_csv_col(fieldnames, _original_score_candidates(original_score_key))

        for row in reader:
            hf_id = str(row.get(hf_id_col, "")).strip()
            if not hf_id:
                continue
            high = _safe_float(row.get(high_col)) if high_col else float("nan")
            low = _safe_float(row.get(low_col)) if low_col else float("nan")
            random_score = _safe_float(row.get(rand_col)) if rand_col else float("nan")
            original_score = _safe_float(row.get(orig_col)) if orig_col else float("nan")
            out[hf_id] = {
                "high_circuit_score": high,
                "low_circuit_score": low,
                "random_mean_score": random_score,
                "original_score": original_score,
                "high_minus_low": high - low,
                "low_minus_high": low - high,
                "abs_high_low": abs(high - low),
            }
    return out

def _select_count(total: int, top_q: float) -> int:
    if total <= 0:
        return 0
    return max(1, int(math.ceil(float(top_q) * float(total))))

def _materialize_by_ids(examples: Sequence[Dict[str, Any]], id_set: set) -> List[Dict[str, Any]]:
    return [x for x in examples if str(x.get("hf_id", "")) in id_set]

def _build_ranked(scored_examples: Sequence[Dict[str, Any]], score_map: Dict[str, Dict[str, float]], score_key: str,) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for ex in scored_examples:
        hf_id = str(ex.get("hf_id", ""))
        v = score_map[hf_id].get(score_key, float("nan"))
        if math.isfinite(v):
            rows.append((hf_id, float(v)))
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows

def _finite_score_count(scored_examples: Sequence[Dict[str, Any]], score_map: Dict[str, Dict[str, float]], score_key: str,) -> int:
    cnt = 0
    for ex in scored_examples:
        hf_id = str(ex.get("hf_id", ""))
        if math.isfinite(float(score_map[hf_id].get(score_key, float("nan")))):
            cnt += 1
    return cnt

def build_training_variants(
    train_examples: Sequence[Dict[str, Any]],
    score_map: Optional[Dict[str, Dict[str, float]]],
    top_q: float,
    seed: int,
    original_score_key: str = "",
    rand_filter_policy: str = "match_scored_q",
    contc_abs_diff: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if rand_filter_policy not in {"match_scored_q", "full_q"}:
        raise ValueError(
            f"Unsupported rand_filter_policy={rand_filter_policy}. "
            "choices=['match_scored_q','full_q']"
        )
    top_q = normalize_top_q(top_q)
    variants: Dict[str, List[Dict[str, Any]]] = {"full": list(train_examples)}
    full_n = len(train_examples)
    n_select_full = _select_count(full_n, top_q)
    rng = random.Random(seed)
    idx = list(range(full_n))
    rng.shuffle(idx)
    n_select_rand = n_select_full
    pick = set(idx[:n_select_rand])
    variants["rand_filter"] = [train_examples[i] for i in range(full_n) if i in pick]

    score_based_modes = ["oris_filter", "highc_filter", "lowc_filter", "bothc_filter", "contc_filter", "randc_filter",]
    for mode in score_based_modes:
        variants.setdefault(mode, [])

    if not score_map:
        meta = {
            "full_train_size": full_n,
            "score_coverage": {"n_with_scores": 0, "n_missing_scores": full_n, "coverage_ratio": 0.0},
            "top_q": top_q,
            "n_select_full": n_select_full,
            "n_select_rand": n_select_rand,
            "n_select_scored": 0,
            "rand_filter_policy": rand_filter_policy,
            "original_score_key": original_score_key,
            "variants": {k: {"n": len(v)} for k, v in variants.items()},
            "notes": [
                "score_map is missing: only full/rand_filter variants were created. To use oris/highc/lowc/bothc/contc/randc filters, provide --score_csv.",
            ],
        }
        return variants, meta

    scored_examples: List[Dict[str, Any]] = []
    missing_score = 0
    for ex in train_examples:
        hf_id = str(ex.get("hf_id", ""))
        if hf_id in score_map:
            scored_examples.append(ex)
        else:
            missing_score += 1
    n_scored = len(scored_examples)
    n_select_scored = _select_count(n_scored, top_q)
    if rand_filter_policy == "match_scored_q":
        # Fair-size comparison: random filter uses the same sample count as score-based filters.
        n_select_rand = n_select_scored if n_scored > 0 else n_select_full
    else:
        n_select_rand = n_select_full
    pick = set(idx[:n_select_rand])
    variants["rand_filter"] = [train_examples[i] for i in range(full_n) if i in pick]

    ranked_high = _build_ranked(scored_examples, score_map, "high_circuit_score")
    ranked_low = _build_ranked(scored_examples, score_map, "low_circuit_score")
    contc_score_key = "abs_high_low" if bool(contc_abs_diff) else "high_minus_low"
    ranked_cont = _build_ranked(scored_examples, score_map, contc_score_key)
    ranked_randc = _build_ranked(scored_examples, score_map, "random_mean_score")
    ranked_oris = _build_ranked(scored_examples, score_map, "original_score")

    ids_high = {hf_id for hf_id, _ in ranked_high[:n_select_scored]}
    ids_low = {hf_id for hf_id, _ in ranked_low[:n_select_scored]}
    n_select_both_half = _select_count(n_scored, top_q / 2.0)
    both_ordered: List[str] = []
    both_seen = set()
    for hf_id, _ in ranked_high[:n_select_both_half]:
        if hf_id in both_seen:
            continue
        both_seen.add(hf_id)
        both_ordered.append(hf_id)
    for hf_id, _ in ranked_low[:n_select_both_half]:
        if hf_id in both_seen:
            continue
        both_seen.add(hf_id)
        both_ordered.append(hf_id)
    if len(both_ordered) < n_select_scored:
        max_ranked = max(len(ranked_high), len(ranked_low))
        for i in range(max_ranked):
            if i < len(ranked_high):
                hf_id = ranked_high[i][0]
                if hf_id not in both_seen:
                    both_seen.add(hf_id)
                    both_ordered.append(hf_id)
            if len(both_ordered) >= n_select_scored:
                break
            if i < len(ranked_low):
                hf_id = ranked_low[i][0]
                if hf_id not in both_seen:
                    both_seen.add(hf_id)
                    both_ordered.append(hf_id)
            if len(both_ordered) >= n_select_scored:
                break
    ids_both = set(both_ordered[:n_select_scored])
    ids_cont = {hf_id for hf_id, _ in ranked_cont[:n_select_scored]}
    ids_randc = {hf_id for hf_id, _ in ranked_randc[:n_select_scored]}
    ids_oris = {hf_id for hf_id, _ in ranked_oris[:n_select_scored]}

    variants["highc_filter"] = _materialize_by_ids(train_examples, ids_high)
    variants["lowc_filter"] = _materialize_by_ids(train_examples, ids_low)
    variants["bothc_filter"] = _materialize_by_ids(train_examples, ids_both)
    variants["contc_filter"] = _materialize_by_ids(train_examples, ids_cont)
    variants["randc_filter"] = _materialize_by_ids(train_examples, ids_randc)
    variants["oris_filter"] = _materialize_by_ids(train_examples, ids_oris)

    meta = {
        "full_train_size": full_n,
        "score_coverage": {
            "n_with_scores": n_scored,
            "n_missing_scores": missing_score,
            "coverage_ratio": float(n_scored / max(1, full_n)),
            "n_with_highc": _finite_score_count(scored_examples, score_map, "high_circuit_score"),
            "n_with_lowc": _finite_score_count(scored_examples, score_map, "low_circuit_score"),
            "n_with_randc": _finite_score_count(scored_examples, score_map, "random_mean_score"),
            "n_with_oris": _finite_score_count(scored_examples, score_map, "original_score"),
        },
        "top_q": top_q,
        "n_select_full": n_select_full,
        "n_select_rand": n_select_rand,
        "n_select_scored": n_select_scored,
        "n_select_both_seed_each": n_select_both_half,
        "n_select_both_target": n_select_scored,
        "n_select_both_actual": int(len(ids_both)),
        "rand_filter_policy": rand_filter_policy,
        "original_score_key": original_score_key,
        "contc_abs_diff": bool(contc_abs_diff),
        "contc_score_key": contc_score_key,
        "variants": {k: {"n": len(v)} for k, v in variants.items()},
        "score_preview": {
            "highc_filter": ranked_high[:10],
            "lowc_filter": ranked_low[:10],
            "bothc_filter": both_ordered[:10],
            "contc_filter": ranked_cont[:10],
            "randc_filter": ranked_randc[:10],
            "oris_filter": ranked_oris[:10],
        },
    }
    if not ranked_oris:
        meta.setdefault("notes", []).append(
            "Could not build oris_filter: original score column is missing or non-finite."
        )
    if not ranked_randc:
        meta.setdefault("notes", []).append(
            "Could not build randc_filter: random circuit column is missing or non-finite."
        )
    if not ranked_high or not ranked_low:
        meta.setdefault("notes", []).append(
            "Could not fully build highc/lowc/bothc filters: high/low circuit columns are missing or non-finite."
        )
    if rand_filter_policy == "match_scored_q" and n_scored == 0:
        meta.setdefault("notes", []).append(
            "rand_filter_policy=match_scored_q requested, but n_with_scores=0 so rand_filter fell back to full_q."
        )
    return variants, meta

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Circuit-aware train split filtering (Exp3-1 baselines)")
    ap.add_argument("--dataset", type=str, required=True, choices=["piqa", "winogrande", "sciq", "arc_easy", "arc"])
    ap.add_argument("--score_csv", type=str, default="", help="CSV containing hf_id and score columns")
    ap.add_argument("--score_key", type=str, default="", help="Original score key (e.g., aum, grad_align)")
    ap.add_argument("--auto_score_from_circuits", action="store_true", help="If --score_csv is empty, compute per-sample circuit scores for a split and save a compatible CSV before filtering.")
    ap.add_argument("--auto_score_force", action="store_true", help="Force re-computation for auto-built score CSV.")
    ap.add_argument("--score_split", type=str, default="train", choices=["train", "val", "test"], help="Split used for auto score CSV generation.")
    ap.add_argument("--circuits_root", type=str, default=str(_ROOT / "circuit_results"))
    ap.add_argument("--buckets_root", type=str, default=str(_ROOT / "data_gens" / "buckets"))
    ap.add_argument("--corrupt_type", type=str, default="gaussian")
    ap.add_argument("--score_model_name", type=str, default="", help="Optional override model name for auto score generation.")
    ap.add_argument("--score_device", type=str, default="cuda")
    ap.add_argument("--score_dtype", type=str, default="", choices=["", "float16", "bfloat16", "float32"])
    ap.add_argument("--score_ig_steps", type=int, default=0, help="0 means auto from circuit run metadata.")
    ap.add_argument("--score_incorrect_agg", type=str, default="auto", choices=["auto", "max", "mean", "min"])
    ap.add_argument("--score_noise_sigma", type=float, default=0.1)
    ap.add_argument("--score_topk_edges", type=int, default=500)
    ap.add_argument("--score_topk_rank_signed", dest="score_topk_rank_abs", action="store_false")
    ap.add_argument("--score_sum_abs", action="store_true")
    ap.add_argument(
        "--score_direction_aware",
        action="store_true",
        help="Use direction-aware circuit score: sum(attribution * sign(circuit edge score)) for high/low circuits.",
    )
    ap.add_argument(
        "--score_weighted_direction_aware",
        action="store_true",
        help="Use weighted direction-aware score: sum(attribution * circuit edge score) for high/low circuits.",
    )
    ap.add_argument(
        "--score_random_baseline_trials",
        type=int,
        default=150,
        help="0 disables randc score. Any value >0 enables a single fixed random-edge subset raw-sum score (independent from high/low score_mode).",
    )
    ap.add_argument("--score_resume", dest="score_resume", action="store_true")
    ap.add_argument("--no_score_resume", dest="score_resume", action="store_false")
    ap.add_argument("--score_save_every", type=int, default=100)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--top_q", type=float, default=0.2, help="e.g., 0.2 or 20")
    ap.add_argument(
        "--contc_abs_diff",
        type=int,
        default=1,
        choices=[0, 1],
        help="ContC ranking score: 1=abs(high-low), 0=(high-low) raw signed difference.",
    )
    ap.add_argument("--rand_filter_policy", type=str,  default="match_scored_q", choices=["match_scored_q", "full_q"],
        help="match_scored_q: rand_filter size matches score-based selected count; full_q: rand_filter uses full train top_q count.",)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=str(_ROOT / "results" / "filtered_data"))
    ap.add_argument("--modes", type=str, default=",".join(EXPERIMENT_MODES))
    ap.set_defaults(score_topk_rank_abs=True)
    ap.set_defaults(score_resume=True)
    return ap

def main() -> None:
    args = build_argparser().parse_args()
    set_seed(int(args.seed))

    dataset = canonical_dataset_name(args.dataset)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in EXPERIMENT_MODES]
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}. allowed={EXPERIMENT_MODES}")

    run_dir = Path(args.out_dir) / dataset / f"filter_{dataset}_{now_str()}"
    ensure_dir(run_dir)

    splits = prepare_task_splits(dataset_name=dataset, seed=int(args.seed), train_frac=float(args.train_frac))
    score_csv_used = str(args.score_csv or "").strip()
    auto_score_meta: Optional[Dict[str, Any]] = None
    if (not score_csv_used) and bool(args.auto_score_from_circuits):
        if not args.score_key:
            raise ValueError("--score_key is required when --auto_score_from_circuits is enabled")
        from applications.circuit_scores import build_full_split_score_csv

        split_name = str(args.score_split)
        split_rows = list(splits[split_name])
        auto_csv = run_dir / f"{split_name}_sample_scores_{dataset}_{args.score_key}.csv"
        auto_score_meta = build_full_split_score_csv(
            dataset=dataset,
            score_key=args.score_key,
            split_examples=split_rows,
            out_csv_path=str(auto_csv),
            circuits_root=str(args.circuits_root),
            buckets_root=str(args.buckets_root),
            corrupt_type=str(args.corrupt_type),
            topk_edges=int(args.score_topk_edges),
            topk_rank_abs=bool(args.score_topk_rank_abs),
            sum_abs=bool(args.score_sum_abs),
            direction_aware=bool(args.score_direction_aware),
            weighted_direction_aware=bool(args.score_weighted_direction_aware),
            random_baseline_trials=int(args.score_random_baseline_trials),
            seed=int(args.seed),
            model_name_override=str(args.score_model_name),
            device=str(args.score_device),
            dtype=str(args.score_dtype),
            ig_steps_override=int(args.score_ig_steps),
            incorrect_agg_override=str(args.score_incorrect_agg),
            noise_sigma=float(args.score_noise_sigma),
            split_name=split_name,
            resume=bool(args.score_resume),
            force_recompute=bool(args.auto_score_force),
            save_every=int(args.score_save_every),
            quiet=False,
        )
        score_csv_used = str(auto_csv)

    score_map = load_score_csv(score_csv_used, original_score_key=args.score_key) if score_csv_used else None
    variants, filter_meta = build_training_variants(
        train_examples=splits["train"],
        score_map=score_map,
        top_q=float(args.top_q),
        seed=int(args.seed),
        original_score_key=args.score_key,
        rand_filter_policy=args.rand_filter_policy,
        contc_abs_diff=bool(args.contc_abs_diff),
    )

    if bool(args.auto_score_from_circuits) and str(args.score_split) != "train":
        filter_meta.setdefault("notes", []).append(
            f"auto score split is '{args.score_split}', but filtering target split is 'train'. "
            "This can reduce score coverage for training selection."
        )

    save_jsonl_rows(splits["val"], run_dir / "val.jsonl")
    save_jsonl_rows(splits["test"], run_dir / "test.jsonl")

    train_files: Dict[str, str] = {}
    for mode in modes:
        if mode not in variants:
            continue
        path = run_dir / f"train_{mode}.jsonl"
        save_jsonl_rows(variants[mode], path)
        train_files[mode] = str(path)

    manifest = {
        "dataset": dataset,
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "top_q": float(args.top_q),
        "score_key": args.score_key,
        "score_csv": score_csv_used,
        "score_csv_input_arg": args.score_csv,
        "requested_modes": modes,
        "saved_train_files": train_files,
        "val_file": str(run_dir / "val.jsonl"),
        "test_file": str(run_dir / "test.jsonl"),
        "split_sizes": {
            "train_full": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "filter_meta": filter_meta,
        "auto_score_meta": auto_score_meta,
    }
    save_json(manifest, run_dir / "filter_manifest.json")

    print("[OK] filtering artifacts saved")
    print(f"  - run_dir: {run_dir}")
    print(f"  - manifest: {run_dir / 'filter_manifest.json'}")
    for mode, fp in train_files.items():
        print(f"  - train[{mode}]: {fp}")


if __name__ == "__main__":
    main()
