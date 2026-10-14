import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
_EAPIG_ROOT = _ROOT / "EAP-IG"
_EAP_SRC = _EAPIG_ROOT / "src"
for _p in [str(_ROOT), str(_EAPIG_ROOT), str(_EAP_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scen1_eapig import load_model, str_to_dtype  # type: ignore
from eap.graph import Graph  # type: ignore
from zero_ablation import (
    ALL_SCORE_KEYS,
    DYNAMIC_SCORE_KEYS,
    build_ablation_graph,
    ensure_dir,
    find_all_scores_json,
    find_latest_circuit_pt,
    forward_with_zero_ablation,
    infer_score_group,
    infer_label_token_ids_from_prompt,
    read_json,
    resolve_gold_token_id,
    write_json,
)

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def logits_grad_l2(logits_vec: torch.Tensor, gold_token_id: int) -> float:
    lv = logits_vec.float()
    log_z = torch.logsumexp(lv, dim=-1)
    logsumexp2 = torch.logsumexp(2.0 * lv, dim=-1)
    sum_p2 = torch.exp(logsumexp2 - 2.0 * log_z)
    p_gold = torch.exp(lv[gold_token_id] - log_z)
    val = torch.sqrt(torch.clamp(sum_p2 - 2.0 * p_gold + 1.0, min=0.0))
    return float(val.item())

def el2n_from_option_probs(option_logits: torch.Tensor, gold_index: int) -> float:
    p = torch.softmax(option_logits.float(), dim=-1)
    y = torch.zeros_like(p)
    y[gold_index] = 1.0
    return float(torch.linalg.norm(p - y, ord=2).item())

def global_grad_norm(model: Any) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total_sq += float(torch.sum(p.grad.detach().float() ** 2).item())
    return float(total_sq ** 0.5)

@dataclass
class TDState:
    aum_sum: float = 0.0
    aum_cnt: int = 0
    forget_cnt: int = 0
    prev_correct: Optional[bool] = None
    el2n_sum: float = 0.0
    el2n_cnt: int = 0
    grad_sum: float = 0.0
    grad_cnt: int = 0

class AblationTrainDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows: List[Dict[str, Any]] = []
        for row in rows:
            prompt = str(row.get("prompt", ""))
            if not prompt:
                continue
            gold_id = row.get("gold_token_id", None)
            if gold_id is None:
                continue
            try:
                gold_id = int(gold_id)
            except Exception:
                continue
            ids = tokenizer.encode(prompt, add_special_tokens=False) + [gold_id]
            if max_length > 0 and len(ids) > max_length:
                ids = ids[-max_length:]
            if len(ids) < 2:
                continue
            pred_pos = len(ids) - 2
            rr = dict(row)
            rr["input_ids"] = ids
            rr["pred_pos"] = int(pred_pos)
            rr["gold_token_id"] = int(gold_id)
            self.rows.append(rr)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "hf_id": str(row["hf_id"]),
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "pred_pos": int(row["pred_pos"]),
            "gold_token_id": int(row["gold_token_id"]),
            "option_token_ids": list(row.get("option_token_ids", [])),
            "gold_option_index": int(row.get("gold_option_index", -1)),
            "raw_row": row.get("raw_row", {}),
        }

def collate_train_batch(batch: Iterable[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    items = list(batch)
    bsz = len(items)
    max_len = max(int(x["input_ids"].shape[0]) for x in items)

    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    pred_pos = torch.empty((bsz,), dtype=torch.long)
    gold_ids = torch.empty((bsz,), dtype=torch.long)
    hf_ids: List[str] = []
    option_token_ids: List[List[int]] = []
    gold_option_index: List[int] = []
    raw_rows: List[Dict[str, Any]] = []

    for i, x in enumerate(items):
        ids = x["input_ids"]
        ll = int(ids.shape[0])
        input_ids[i, :ll] = ids
        attention_mask[i, :ll] = 1
        pred_pos[i] = int(x["pred_pos"])
        gold_ids[i] = int(x["gold_token_id"])
        hf_ids.append(str(x["hf_id"]))
        option_token_ids.append([int(v) for v in x.get("option_token_ids", [])])
        gold_option_index.append(int(x.get("gold_option_index", -1)))
        raw_rows.append(dict(x.get("raw_row", {})))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pred_pos": pred_pos,
        "gold_ids": gold_ids,
        "hf_ids": hf_ids,
        "option_token_ids": option_token_ids,
        "gold_option_index": gold_option_index,
        "raw_rows": raw_rows,
    }

def _normalize_label_data(row: Dict[str, Any], tokenizer: Any,) -> Tuple[List[int], int]:
    label_space = row.get("label_space")
    gold_label = row.get("gold_label")
    if not isinstance(label_space, list) or not label_space or gold_label is None:
        return [], -1
    label_space = [str(x) for x in label_space]
    gold_label = str(gold_label)

    label_ids = row.get("label_token_ids")
    if not isinstance(label_ids, dict) or len(label_ids) < len(label_space):
        prompt = str(row.get("prompt", ""))
        if not prompt:
            return [], -1
        label_ids = infer_label_token_ids_from_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            label_space=label_space,
            strict=False,
        )

    try:
        opt_ids = [int(label_ids[lab]) for lab in label_space if lab in label_ids]
    except Exception:
        return [], -1
    if len(opt_ids) != len(label_space):
        return [], -1
    if gold_label not in label_space:
        return [], -1
    gold_idx = int(label_space.index(gold_label))
    return opt_ids, gold_idx

def load_training_rows(all_scores_json: Path, tokenizer: Any, max_samples: int, seed: int,) -> List[Dict[str, Any]]:
    obj = read_json(all_scores_json)
    rows = obj.get("items", obj if isinstance(obj, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected all-scores format: {all_scores_json}")

    out: List[Dict[str, Any]] = []
    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if not hf_id:
            continue
        prompt = str(r.get("prompt", "")).strip()
        if not prompt:
            continue
        gold_token_id = resolve_gold_token_id(r, tokenizer=tokenizer)
        if gold_token_id is None:
            continue

        option_ids, gold_idx = _normalize_label_data(r, tokenizer)
        rr = {
            "hf_id": hf_id,
            "prompt": prompt,
            "gold_token_id": int(gold_token_id),
            "option_token_ids": option_ids,
            "gold_option_index": int(gold_idx),
            "raw_row": dict(r),
        }
        out.append(rr)

    if max_samples > 0 and len(out) > max_samples:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(out), size=max_samples, replace=False).tolist()
        out = [out[i] for i in idx]
        out.sort(key=lambda x: str(x["hf_id"]))
    return out

def save_checkpoint(model: Any, path: Path, epoch: int, step: int, meta: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "global_step": int(step),
            "meta": meta,
        },
        str(path),
    )

def build_random_ablation_graph(
    model: Any,
    circuit_pt: Path,
    topk_edges: int,
    random_seed: int,
) -> Tuple[Graph, Dict[str, Any]]:
    saved_graph = Graph.from_pt(str(circuit_pt))
    saved_edges = list(saved_graph.edges.values())
    total_edges = len(saved_edges)
    if total_edges == 0:
        raise RuntimeError(f"No edges found in graph: {circuit_pt}")

    if int(topk_edges) <= 0:
        target_k = sum(1 for e in saved_edges if bool(e.in_graph))
        if target_k <= 0:
            target_k = total_edges
    else:
        target_k = min(int(topk_edges), total_edges)

    runtime_graph = Graph.from_model(model)
    runtime_graph.reset(empty=False)

    score_by_name: Dict[str, float] = {str(e.name): float(e.score) for e in saved_edges}
    candidate_names = [str(e.name) for e in saved_edges if str(e.name) in runtime_graph.edges]
    if len(candidate_names) == 0:
        raise RuntimeError("No overlapping edge names between saved circuit and runtime graph.")

    k = min(int(target_k), len(candidate_names))
    rng = np.random.default_rng(int(random_seed))
    picked_idx = rng.choice(len(candidate_names), size=k, replace=False).tolist()
    selected_names = [candidate_names[int(i)] for i in picked_idx]

    rows: List[int] = []
    cols: List[int] = []
    for name in selected_names:
        edge = runtime_graph.edges.get(name)
        if edge is None:
            continue
        rows.append(int(edge.matrix_index[0]))
        cols.append(int(edge.matrix_index[1]))

    if rows:
        rr = torch.tensor(rows, dtype=torch.long)
        cc = torch.tensor(cols, dtype=torch.long)
        runtime_graph.in_graph[rr, cc] = False
    runtime_graph.prune()

    edge_meta = {
        "circuit_pt": str(circuit_pt),
        "ablation_edge_mode": "random",
        "random_edges_seed": int(random_seed),
        "random_edges_candidate_pool": int(len(candidate_names)),
        "random_edges_target_k": int(target_k),
        "selected_edges_by_name": int(len(selected_names)),
        "mapped_edges_in_runtime": int(len(rows)),
        "total_edges_in_saved_graph": int(total_edges),
        "remaining_edges_after_ablation": int(runtime_graph.in_graph.sum().item()),
        "top10_selected_edges": [
            {"edge": n, "score": float(score_by_name.get(n, float("nan")))}
            for n in selected_names[:10]
        ],
    }
    return runtime_graph, edge_meta

def build_random_ablation_graph_from_model(
    model: Any,
    topk_edges: int,
    random_seed: int,
) -> Tuple[Graph, Dict[str, Any]]:
    runtime_graph = Graph.from_model(model)
    runtime_graph.reset(empty=False)
    edge_names = list(runtime_graph.edges.keys())
    total_edges = len(edge_names)
    if total_edges == 0:
        raise RuntimeError("No edges found in runtime graph.")
    if int(topk_edges) <= 0:
        raise ValueError("--topk_edges must be > 0 when --random_edge_pool=model.")

    k = min(int(topk_edges), total_edges)
    rng = np.random.default_rng(int(random_seed))
    picked_idx = rng.choice(total_edges, size=k, replace=False).tolist()
    selected_names = [edge_names[int(i)] for i in picked_idx]

    rows: List[int] = []
    cols: List[int] = []
    for name in selected_names:
        edge = runtime_graph.edges.get(name)
        if edge is None:
            continue
        rows.append(int(edge.matrix_index[0]))
        cols.append(int(edge.matrix_index[1]))

    if rows:
        rr = torch.tensor(rows, dtype=torch.long)
        cc = torch.tensor(cols, dtype=torch.long)
        runtime_graph.in_graph[rr, cc] = False
    runtime_graph.prune()

    edge_meta = {
        "circuit_pt": "",
        "ablation_edge_mode": "random",
        "random_edge_pool": "model",
        "random_edges_seed": int(random_seed),
        "random_edges_candidate_pool": int(total_edges),
        "random_edges_target_k": int(k),
        "selected_edges_by_name": int(len(selected_names)),
        "mapped_edges_in_runtime": int(len(rows)),
        "remaining_edges_after_ablation": int(runtime_graph.in_graph.sum().item()),
        "top10_selected_edges": [{"edge": n} for n in selected_names[:10]],
    }
    return runtime_graph, edge_meta

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Finetune under circuit zero-ablation. Saves checkpoints + all_scores JSON.")
    ap.add_argument("--dataset", type=str, required=True, help="arc_easy/sciq/piqa/winogrande")
    ap.add_argument("--score_key", type=str, required=True, choices=sorted(ALL_SCORE_KEYS))
    ap.add_argument("--buckets_root", type=str, default=str(_ROOT / "data_gens" / "buckets"))
    ap.add_argument("--circuits_root", type=str, default=str(_ROOT / "circuit_results"))
    ap.add_argument("--all_scores_json", type=str, default="", help="Override input all_scores json")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--resume_state_dict", type=str, default="", help="Optional local state_dict checkpoint to initialize from")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--disable_ablation", action="store_true", help="Train full model without zero-ablation hooks")
    ap.add_argument("--circuit_bucket", type=str, default="high", choices=["high", "low"])
    ap.add_argument("--corrupt_type", type=str, default="gaussian")
    ap.add_argument("--ablation_edge_mode", type=str, default="scored", choices=["scored", "random"], help="Use top-score edges (scored) or random edges with matched k (random).",)
    ap.add_argument("--random_edges_seed", type=int, default=42, help="Random seed for --ablation_edge_mode=random")
    ap.add_argument("--random_edge_pool", type=str, default="circuit", choices=["circuit", "model"], help="Random edge source: discovered circuit edges or full runtime-model edges.")
    ap.add_argument("--topk_edges", type=int, default=500, help="<=0 uses saved in_graph, >0 uses top-k edges")
    ap.add_argument("--topk_rank_signed", dest="topk_rank_abs", action="store_false")
    ap.set_defaults(topk_rank_abs=True)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--el2n_epochs", type=int, default=2, help="Compute EL2N only for early epochs")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out_root", type=str, default=str(_ROOT / "checkpoints" / "zero_ablation_finetune_runs"))
    ap.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    ap.add_argument("--wandb_project", type=str, default="data_mi")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_log_every", type=int, default=20, help="Log every N optimizer steps")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random_seed = int(args.seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    group = infer_score_group(args.score_key)
    all_scores_json = Path(args.all_scores_json) if args.all_scores_json else find_all_scores_json(
        buckets_root=Path(args.buckets_root),
        dataset=args.dataset,
        score_key=args.score_key,
        group=group,
    )
    if all_scores_json is None or (not all_scores_json.is_file()):
        raise FileNotFoundError(f"all_scores json not found for dataset={args.dataset}, score_key={args.score_key}")

    dtype = str_to_dtype(args.dtype)
    if (not args.device.startswith("cuda")) and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    model = load_model(model_name=args.model_name, device=args.device, dtype=dtype)
    model.train()
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = int(model.tokenizer.pad_token_id)

    if args.resume_state_dict:
        ckpt = Path(args.resume_state_dict)
        if not ckpt.is_file():
            raise FileNotFoundError(f"resume_state_dict not found: {ckpt}")
        obj = torch.load(str(ckpt), map_location=args.device)
        state_dict = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        model.load_state_dict(state_dict, strict=False)

    train_rows = load_training_rows(
        all_scores_json=all_scores_json,
        tokenizer=model.tokenizer,
        max_samples=int(args.max_samples),
        seed=random_seed,
    )
    if len(train_rows) == 0:
        raise RuntimeError("No usable training rows found.")

    train_ds = AblationTrainDataset(rows=train_rows, tokenizer=model.tokenizer, max_length=int(args.max_length))
    if len(train_ds) == 0:
        raise RuntimeError("No usable rows after tokenization.")

    train_dl = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda b: collate_train_batch(b, pad_id),
        drop_last=False,
    )

    ablation_enabled = not bool(args.disable_ablation)
    edge_meta: Dict[str, Any] = {}
    circuit_pt: Optional[Path] = None
    ab_graph = None
    if ablation_enabled:
        use_model_pool = bool(args.ablation_edge_mode == "random" and args.random_edge_pool == "model")
        if use_model_pool:
            ab_graph, edge_meta = build_random_ablation_graph_from_model(
                model=model,
                topk_edges=int(args.topk_edges),
                random_seed=int(args.random_edges_seed),
            )
        else:
            circuit_pt = find_latest_circuit_pt(
                circuits_root=Path(args.circuits_root),
                dataset=args.dataset,
                score_key=args.score_key,
                bucket=args.circuit_bucket,
                corrupt_type=args.corrupt_type,
                group=group,
            )
            if circuit_pt is None:
                raise FileNotFoundError(
                    f"Could not find circuit .pt for {args.dataset}/{args.score_key}/{args.circuit_bucket}/{args.corrupt_type}"
                )
            if args.ablation_edge_mode == "random":
                ab_graph, edge_meta = build_random_ablation_graph(
                    model=model,
                    circuit_pt=circuit_pt,
                    topk_edges=int(args.topk_edges),
                    random_seed=int(args.random_edges_seed),
                )
            else:
                ab_graph, edge_meta = build_ablation_graph(
                    model=model,
                    circuit_pt=circuit_pt,
                    topk_edges=int(args.topk_edges),
                    rank_abs=bool(args.topk_rank_abs),
                )

    def forward_fn(tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if ablation_enabled and (ab_graph is not None):
            return forward_with_zero_ablation(model, ab_graph, tokens, attn)
        return model(tokens, attention_mask=attn)

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    states: Dict[str, TDState] = {str(r["hf_id"]): TDState() for r in train_ds.rows}

    if not ablation_enabled:
        run_mode = "full"
    elif args.ablation_edge_mode == "random":
        run_mode = "ablated_random_model" if args.random_edge_pool == "model" else "ablated_random_circuit"
    else:
        run_mode = "ablated"
    bucket_tag = "none" if (args.ablation_edge_mode == "random" and args.random_edge_pool == "model") else args.circuit_bucket
    run_tag = f"{args.dataset}_{args.score_key}_{run_mode}_{bucket_tag}_{now_str()}"
    run_dir = Path(args.out_root) / args.dataset / args.score_key / run_tag
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    wb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install wandb or run without --use_wandb.")
        wb_cfg = dict(vars(args))
        wb_cfg.update(
            {
                "resolved_all_scores_json": str(all_scores_json),
                "run_mode": run_mode,
                "run_dir": str(run_dir),
                "ablation_enabled": bool(ablation_enabled),
                "circuit_pt": str(circuit_pt) if circuit_pt is not None else "",
            }
        )
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=(args.wandb_run_name or run_tag),
            config=wb_cfg,
        )

    global_step = 0
    seen_batches = 0
    wandb_log_every = max(1, int(args.wandb_log_every))
    for epoch in range(int(args.epochs)):
        iterator = train_dl if args.quiet else tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        model.train()
        for batch in iterator:
            seen_batches += 1
            did_step = False
            gnorm = None
            tokens = batch["input_ids"].to(args.device)
            attn = batch["attention_mask"].to(args.device)
            pred_pos = batch["pred_pos"].to(args.device)
            gold_ids = batch["gold_ids"].to(args.device)
            hf_ids = [str(x) for x in batch["hf_ids"]]
            option_ids_batch = batch["option_token_ids"]
            gold_opt_idx_batch = batch["gold_option_index"]

            model.reset_hooks(including_permanent=True)
            logits = forward_fn(tokens, attn)
            idx = torch.arange(logits.shape[0], device=logits.device)
            last_logits = logits[idx, pred_pos, :]
            loss = F.cross_entropy(last_logits.float(), gold_ids, reduction="mean") / max(1, int(args.grad_accum))
            loss.backward()

            if seen_batches % max(1, int(args.grad_accum)) == 0:
                gnorm = global_grad_norm(model)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                did_step = True

            batch_aum_vals: List[float] = []
            batch_el2n_vals: List[float] = []
            batch_grad_vals: List[float] = []
            batch_acc_vals: List[float] = []
            with torch.no_grad():
                for i, hf_id in enumerate(hf_ids):
                    st = states.get(hf_id)
                    if st is None:
                        continue
                    lv = last_logits[i]
                    gold_token_id = int(gold_ids[i].item())

                    grad_l2 = logits_grad_l2(lv, gold_token_id)
                    st.grad_sum += grad_l2
                    st.grad_cnt += 1
                    batch_grad_vals.append(grad_l2)

                    option_ids = option_ids_batch[i]
                    gold_opt_idx = int(gold_opt_idx_batch[i])
                    if len(option_ids) >= 2 and gold_opt_idx >= 0 and gold_opt_idx < len(option_ids):
                        opt_logits = lv[torch.tensor(option_ids, device=lv.device)]
                        gold_logit = opt_logits[gold_opt_idx]
                        other = torch.cat([opt_logits[:gold_opt_idx], opt_logits[gold_opt_idx + 1 :]], dim=0)
                        other_max = torch.max(other) if other.numel() > 0 else gold_logit

                        margin = float((gold_logit - other_max).item())
                        st.aum_sum += margin
                        st.aum_cnt += 1
                        batch_aum_vals.append(margin)

                        pred_idx = int(torch.argmax(opt_logits).item())
                        correct = pred_idx == gold_opt_idx
                        if st.prev_correct is not None and st.prev_correct and (not correct):
                            st.forget_cnt += 1
                        st.prev_correct = bool(correct)
                        batch_acc_vals.append(1.0 if correct else 0.0)

                        if epoch < int(args.el2n_epochs):
                            e = el2n_from_option_probs(opt_logits, gold_opt_idx)
                            st.el2n_sum += e
                            st.el2n_cnt += 1
                            batch_el2n_vals.append(e)

            if wb_run is not None and did_step and (global_step % wandb_log_every == 0):
                log_dict: Dict[str, float] = {
                    "train/loss": float(loss.item() * max(1, int(args.grad_accum))),
                    "train/lr": float(optim.param_groups[0]["lr"]),
                    "train/epoch": float(epoch + 1),
                    "train/global_step": float(global_step),
                    "train/seen_batches": float(seen_batches),
                }
                if gnorm is not None:
                    log_dict["train/global_grad_norm"] = float(gnorm)
                if batch_grad_vals:
                    log_dict["td/batch_gradlogits_l2_mean"] = float(sum(batch_grad_vals) / len(batch_grad_vals))
                if batch_aum_vals:
                    log_dict["td/batch_aum_mean"] = float(sum(batch_aum_vals) / len(batch_aum_vals))
                if batch_acc_vals:
                    log_dict["td/batch_acc_mean"] = float(sum(batch_acc_vals) / len(batch_acc_vals))
                if batch_el2n_vals:
                    log_dict["td/batch_el2n_mean"] = float(sum(batch_el2n_vals) / len(batch_el2n_vals))
                wb_run.log(log_dict, step=global_step)

        meta_ckpt = {
            "dataset": args.dataset,
            "score_key": args.score_key,
            "epoch": int(epoch + 1),
            "global_step": int(global_step),
            "ablation_enabled": bool(ablation_enabled),
            "ablation_edge_mode": str(args.ablation_edge_mode),
            "random_edge_pool": str(args.random_edge_pool),
            "random_edges_seed": int(args.random_edges_seed),
            "circuit_bucket": args.circuit_bucket,
        }
        save_checkpoint(model, ckpt_dir / f"epoch_{epoch+1:03d}.pt", epoch=epoch + 1, step=global_step, meta=meta_ckpt)

    save_checkpoint(model, ckpt_dir / "final.pt", epoch=int(args.epochs), step=global_step, meta={"final": True})

    rows_out: List[Dict[str, Any]] = []
    for rr in train_ds.rows:
        hf_id = str(rr["hf_id"])
        st = states.get(hf_id, TDState())
        raw = dict(rr.get("raw_row", {}))

        out = {
            "hf_id": hf_id,
            "task": raw.get("task"),
            "question_text": raw.get("question_text"),
            "choices": raw.get("choices"),
            "label_space": raw.get("label_space"),
            "gold_label": raw.get("gold_label"),
            "gold_token_id": int(rr["gold_token_id"]),
            "prompt": raw.get("prompt", rr.get("prompt")),
            "tokens_len": raw.get("tokens_len"),
            "fewshot_k": raw.get("fewshot_k"),
            "scores": {
                "aum": (st.aum_sum / max(1, st.aum_cnt)) if st.aum_cnt > 0 else None,
                "forget": int(st.forget_cnt),
                "el2n": (st.el2n_sum / max(1, st.el2n_cnt)) if st.el2n_cnt > 0 else None,
                "gradlogits_l2": (st.grad_sum / max(1, st.grad_cnt)) if st.grad_cnt > 0 else None,
            },
            "counts": {
                "aum_cnt": int(st.aum_cnt),
                "forget_cnt": int(st.forget_cnt),
                "el2n_cnt": int(st.el2n_cnt),
                "grad_cnt": int(st.grad_cnt),
            },
            "orig_scores": raw.get("scores", {}),
        }
        rows_out.append(out)

    meta = {
        "dataset": args.dataset,
        "score_key": args.score_key,
        "input_all_scores_json": str(all_scores_json),
        "model_name": args.model_name,
        "resume_state_dict": args.resume_state_dict,
        "device": args.device,
        "dtype": args.dtype,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_accum": int(args.grad_accum),
        "max_samples": int(args.max_samples),
        "max_length": int(args.max_length),
        "ablation_enabled": bool(ablation_enabled),
        "ablation_edge_mode": str(args.ablation_edge_mode),
        "random_edge_pool": str(args.random_edge_pool),
        "random_edges_seed": int(args.random_edges_seed),
        "circuit_bucket": args.circuit_bucket,
        "corrupt_type": args.corrupt_type,
        "topk_edges": int(args.topk_edges),
        "topk_rank_abs": bool(args.topk_rank_abs),
        "circuit_pt": str(circuit_pt) if circuit_pt is not None else "",
        "edge_meta": edge_meta,
        "num_rows_out": int(len(rows_out)),
        "global_step": int(global_step),
        "wandb": {
            "enabled": bool(wb_run is not None),
            "project": args.wandb_project if wb_run is not None else "",
            "entity": args.wandb_entity if wb_run is not None else "",
            "run_name": (wb_run.name if wb_run is not None else ""),
            "run_id": (wb_run.id if wb_run is not None else ""),
            "run_url": (getattr(wb_run, "url", "") if wb_run is not None else ""),
        },
    }

    all_scores_out = run_dir / f"all_scores_{args.dataset}.json"
    write_json({"meta": meta, "items": rows_out}, all_scores_out)
    summary_out = run_dir / "run_summary.json"
    write_json(
        {
            "run_dir": str(run_dir),
            "all_scores_json": str(all_scores_out),
            "final_checkpoint": str(ckpt_dir / "final.pt"),
            "checkpoints_dir": str(ckpt_dir),
            "meta": meta,
        },
        summary_out,
    )

    print("[OK] zero-ablation finetune run saved:")
    print(f"  - run_dir: {run_dir}")
    print(f"  - all_scores_json: {all_scores_out}")
    print(f"  - final_checkpoint: {ckpt_dir / 'final.pt'}")
    print(f"  - summary: {summary_out}")

    if wb_run is not None:
        wb_run.log(
            {
                "run/global_step": float(global_step),
                "run/num_rows_out": float(len(rows_out)),
            },
            step=global_step,
        )
        wb_run.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
