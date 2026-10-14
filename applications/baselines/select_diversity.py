"""Experiment 1b-①: diversity-filtered naive-prompt pool.

Reads the 20k naive_raw_candidates, embeds `question` with
sentence-transformers/all-MiniLM-L6-v2, runs greedy farthest-point
selection down to N (default 10 000), writes a mirror pool the trainer
can consume as BASELINE_MODE=prompt_only.

Output (default --variant-root):
        results/supplemental/baselines/work_exp1_downstream_seeds/diversity/
        Qwen__Qwen2.5-1.5B-Instruct/naive/fixed_both_discovered/<ts>/naive_selected.jsonl
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
import sys
from pathlib import Path

import numpy as np

from applications.experiment_utils.io import (
    NAIVE_RAW_PATH,
    dump_pool_as_mirror,
    read_jsonl,
    write_json,
    write_jsonl,
)
from applications.curriculum_ft import generated_row_to_example  # noqa: E402

EXP1_WORK_ROOT = _SUPPLEMENTAL_ROOT / "baselines" / "work_exp1_downstream_seeds"


def greedy_farthest_point(emb: np.ndarray, n_select: int, seed: int = 0) -> list[int]:
    """Greedy max-min (farthest-point) selection on L2-normalized embeddings."""
    n = emb.shape[0]
    if n_select >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    picked = [first]
    # min-distance to picked set for every candidate (cosine distance since normalized)
    dist = 1.0 - emb @ emb[first]
    dist[first] = -np.inf
    for _ in range(n_select - 1):
        j = int(np.argmax(dist))
        picked.append(j)
        new_d = 1.0 - emb @ emb[j]
        dist = np.minimum(dist, new_d)
        dist[j] = -np.inf
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(NAIVE_RAW_PATH))
    ap.add_argument("--variant-root", default=str(EXP1_WORK_ROOT / "diversity"))
    ap.add_argument("--n-select", type=int, default=10000)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap input rows for smoke tests")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    valid_rows = []
    normalized_rows = []
    invalid_rows = []
    for idx, row in enumerate(rows):
        normalized = generated_row_to_example(row, dataset_name="sciq")
        if normalized is None:
            invalid_rows.append(
                {
                    "idx": idx,
                    "candidate_id": row.get("candidate_id") or row.get("record_id"),
                    "reason": "rejected by curriculum_ft.generated_row_to_example",
                }
            )
            continue
        valid_rows.append(row)
        normalized_rows.append(normalized)
    if not (0 < args.n_select <= len(valid_rows)):
        raise RuntimeError(
            f"requested n_select={args.n_select}, but only {len(valid_rows)} of "
            f"{len(rows)} raw rows are trainer-convertible"
        )
    print(
        f"[load] raw={len(rows)} valid={len(valid_rows)} invalid={len(invalid_rows)} "
        f"from {args.input}"
    )

    # Encode question texts
    from sentence_transformers import SentenceTransformer  # deferred import

    model = SentenceTransformer(args.model)
    texts = [str(row["question_text"]) for row in normalized_rows]
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"[embed] {emb.shape}")

    idxs = greedy_farthest_point(emb, n_select=args.n_select, seed=args.seed)
    picked = [valid_rows[i] for i in idxs]

    out = dump_pool_as_mirror(
        picked,
        variant_root=Path(args.variant_root),
        recipe="naive",
        prompt_mode="fixed",
        gen_mode="both",
        circuit_target_mode="discovered",
        file_kind="selected",
    )
    invalid_path = out.parent / "naive_invalid_manifest.jsonl"
    write_jsonl(invalid_path, invalid_rows)
    meta_path = out.parent / "naive_selection_meta.json"
    write_json(
        meta_path,
        {
            "input": str(Path(args.input).expanduser().resolve()),
            "selection": "greedy_farthest_point_on_core_normalized_question",
            "embedding_model": args.model,
            "seed": args.seed,
            "raw_count": len(rows),
            "valid_count": len(valid_rows),
            "invalid_count": len(invalid_rows),
            "selected_count": len(picked),
            "selection_fraction_of_valid": len(picked) / len(valid_rows),
            "invalid_manifest": str(invalid_path),
            "output": str(out),
        },
    )
    print(
        f"[write] {out} (raw={len(rows)}, valid={len(valid_rows)}, "
        f"invalid={len(invalid_rows)}, selected={len(picked)}); meta={meta_path}"
    )


if __name__ == "__main__":
    main()
