#!/usr/bin/env python3
# Per-sample AUM / EL2N / accuracy for one checkpoint, with the ablation hook
# optionally installed -- the snapshot counterpart to the training-dynamics
# averages in all_scores_sciq.json.
#
# The shipped 4a matrix pairs each column differently: ΔAUM and ΔEL2N compare
# training dynamics accumulated over two separate 5-epoch trajectories, ΔGA
# compares hook-on vs hook-off inside one checkpoint, and ΔAcc/ΔLoss compare the
# full trajectory's final weights against the ablated trajectory's. This script
# exists so all four can instead be read off the same two checkpoints.
#
# Definitions are imported from the trainer rather than restated, so AUM stays
# "gold option logit minus best other option logit" and EL2N stays
# "||softmax(option logits) - onehot(gold)||_2" by construction.
#
#   python score_checkpoint_metrics.py --checkpoint <ckpt> --output-json <out>
#   python score_checkpoint_metrics.py --checkpoint <ckpt> --ablate \
#       --circuit-pt <edge.pt> --output-json <out>

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
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE = _REPO_ROOT
for extra in (BASE, BASE / "EAP-IG"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from zero_ablation_finetune import (  # noqa: E402
    AblationTrainDataset,
    build_ablation_graph,
    build_random_ablation_graph,
    collate_train_batch,
    el2n_from_option_probs,
    load_model,
    load_training_rows,
    str_to_dtype,
)
from zero_ablation import forward_with_zero_ablation  # type: ignore # noqa: E402

DEFAULT_INPUT = (
    str(_REPO_ROOT / 'data_gens/buckets/application/Qwen__Qwen2.5-1.5B-Instruct/all/sciq/all_scores_sciq.json')
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--all-scores-json", default=DEFAULT_INPUT)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--circuit-pt", default="")
    ap.add_argument("--ablation-edge-mode", default="scored", choices=("scored", "random"))
    ap.add_argument("--random-edge-pool", default="circuit", choices=("circuit", "model"))
    ap.add_argument("--topk-edges", type=int, default=250)
    ap.add_argument("--topk-rank-abs", action="store_true")
    ap.add_argument("--random-edges-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = str_to_dtype(args.dtype)
    model = load_model(model_name=args.model_name, device=args.device, dtype=dtype)
    model.eval()
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = int(model.tokenizer.pad_token_id)

    obj = torch.load(args.checkpoint, map_location=args.device)
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    bad = model.load_state_dict(state, strict=False)
    if bad.missing_keys or bad.unexpected_keys:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={bad.missing_keys[:5]}, "
            f"unexpected={bad.unexpected_keys[:5]}"
        )

    rows = load_training_rows(
        all_scores_json=Path(args.all_scores_json),
        tokenizer=model.tokenizer,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    ds = AblationTrainDataset(rows=rows, tokenizer=model.tokenizer, max_length=args.max_length)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_train_batch(b, pad_id),
        drop_last=False,
    )

    graph = None
    edge_meta = {}
    if args.ablate:
        if not args.circuit_pt:
            raise SystemExit("--ablate needs --circuit-pt")
        circuit = Path(args.circuit_pt)
        if not circuit.is_file():
            raise SystemExit(f"circuit not found: {circuit}")
        if args.ablation_edge_mode == "random":
            graph, edge_meta = build_random_ablation_graph(
                model=model, circuit_pt=circuit,
                topk_edges=args.topk_edges, random_seed=args.random_edges_seed,
            )
        else:
            graph, edge_meta = build_ablation_graph(
                model=model, circuit_pt=circuit,
                topk_edges=args.topk_edges, rank_abs=bool(args.topk_rank_abs),
            )

    def forward(tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if graph is not None:
            return forward_with_zero_ablation(model, graph, tokens, attn)
        return model(tokens, attention_mask=attn)

    items = []
    n_correct = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="score"):
            tokens = batch["input_ids"].to(args.device)
            attn = batch["attention_mask"].to(args.device)
            pred_pos = batch["pred_pos"].to(args.device)
            logits = forward(tokens, attn)
            last = logits[torch.arange(logits.shape[0]), pred_pos, :]
            for i, hf_id in enumerate(batch["hf_ids"]):
                option_ids = batch["option_token_ids"][i]
                gold_idx = int(batch["gold_option_index"][i])
                if len(option_ids) < 2 or not (0 <= gold_idx < len(option_ids)):
                    continue
                lv = last[i]
                opt = lv[torch.tensor(option_ids, device=lv.device)]
                gold_logit = opt[gold_idx]
                other = torch.cat([opt[:gold_idx], opt[gold_idx + 1:]], dim=0)
                other_max = torch.max(other) if other.numel() > 0 else gold_logit
                correct = int(torch.argmax(opt).item()) == gold_idx
                n_correct += int(correct)
                items.append({
                    "hf_id": str(hf_id),
                    "scores": {
                        "aum": float((gold_logit - other_max).item()),
                        "el2n": el2n_from_option_probs(opt, gold_idx),
                    },
                    "correct": bool(correct),
                })

    out = {
        "meta": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "ablated": bool(args.ablate),
            "circuit_pt": str(Path(args.circuit_pt).resolve()) if args.circuit_pt else "",
            "ablation_edge_mode": args.ablation_edge_mode if args.ablate else "",
            "topk_edges": int(args.topk_edges) if args.ablate else 0,
            "edge_meta": edge_meta,
            "model_name": args.model_name,
            "dtype": args.dtype,
            "all_scores_json": str(Path(args.all_scores_json).resolve()),
            "max_length": int(args.max_length),
            "seed": int(args.seed),
            "n_items": len(items),
            "accuracy": (n_correct / len(items)) if items else float("nan"),
            "definition": {
                "aum": "gold option logit - max other option logit at this checkpoint",
                "el2n": "||softmax(option logits) - onehot(gold)||_2 at this checkpoint",
            },
        },
        "items": items,
    }
    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(outp)
    print(f"[ok] {len(items)} items, acc={out['meta']['accuracy']:.4f} -> {outp}")


if __name__ == "__main__":
    main()
