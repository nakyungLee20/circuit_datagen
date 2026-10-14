from pathlib import Path
import os, re, json, time, argparse, random, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from tqdm import tqdm

# ----------------------------- Utilities -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stable_hash_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def split_list(items: List[Any], frac: float, seed: int) -> Tuple[List[Any], List[Any]]:
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(frac * len(items))
    a = [items[i] for i in idx[:cut]]
    b = [items[i] for i in idx[cut:]]
    return a, b

def pop_fewshot(items: List[Dict[str, Any]], k: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if k <= 0 or len(items) == 0:
        return [], items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    pick = set(idx[:min(k, len(items))])
    few = [items[i] for i in range(len(items)) if i in pick]
    rest = [items[i] for i in range(len(items)) if i not in pick]
    return few, rest

def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(to_text(y) for y in x)
    if isinstance(x, dict):
        if "text" in x:
            return to_text(x["text"])
        return json.dumps(x, ensure_ascii=False)
    return str(x)

# ----------------------------- Prompt / label token handling -----------------------------
def choose_label_tokenization(tokenizer, label_space):
    """
    Ensure label is exactly 1 token given a prefix (for stable next-token classification).
    """
    candidates = ["\nAnswer:\n", "\nAnswer: ", "\nAnswer:"]
    for prefix in candidates:
        ok = True
        ids = {}
        for lab in label_space:
            a = tokenizer.encode(prefix + lab, add_special_tokens=False)
            b = tokenizer.encode(prefix, add_special_tokens=False)
            if len(a) != len(b) + 1:
                ok = False
                break
            ids[lab] = a[-1]
        if ok:
            return prefix, ids
    raise ValueError(f"Could not find a prefix for labels {label_space} with single-token labels.")

def format_mcqa_prompt(
    question_text: str,
    choices: List[str],
    label_space: List[str],
    tokenizer,
    fewshot_context: str = "",
) -> Tuple[str, str, Dict[str, int]]:
    answer_prefix, label_token_ids = choose_label_tokenization(tokenizer, label_space)
    lines = [question_text.strip(), ""]
    for lab, ch in zip(label_space, choices):
        lines.append(f"{lab}. {str(ch).strip()}")
    prompt = "\n".join(lines) + answer_prefix
    if fewshot_context:
        prompt = fewshot_context + prompt
    return prompt, answer_prefix, label_token_ids

def build_fewshot_context(fewshot_examples: List[Dict[str, Any]], tokenizer) -> str:
    if not fewshot_examples:
        return ""
    blocks = []
    for ex in fewshot_examples:
        p, _, _ = format_mcqa_prompt(
            ex["question_text"], ex["choices"], ex["label_space"], tokenizer, fewshot_context=""
        )
        blocks.append(p + ex["gold_label"] + "\n\n")
    return "".join(blocks)

# ----------------------------- Dataset adapters -----------------------------
def load_piqa_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("piqa", trust_remote_code=True)
    out = []
    for i, ex in enumerate(ds[split]):
        out.append({
            "_hf_id": f"piqa/{split}/{i}",
            "goal": ex.get("goal", ""),
            "sol1": ex.get("sol1", ""),
            "sol2": ex.get("sol2", ""),
            "label": ex.get("label", None),  # 0/1
        })
    return out

def build_piqa_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exs = []
    for r in rows:
        if r["label"] is None:
            continue
        gold = "A" if int(r["label"]) == 0 else "B"
        question_text = f"Goal: {r['goal']}"
        choices = [r["sol1"], r["sol2"]]
        exs.append({
            "hf_id": r["_hf_id"],
            "task": "piqa",
            "question_text": question_text,
            "choices": choices,
            "label_space": ["A", "B"],
            "gold_label": gold,
        })
    return exs

def load_winogrande_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/winogrande", "winogrande_xl")
    out = []
    for i, ex in enumerate(ds[split]):
        out.append({
            "_hf_id": f"winogrande_xl/{split}/{i}",
            "sentence": ex.get("sentence", ""),
            "option1": ex.get("option1", ""),
            "option2": ex.get("option2", ""),
            "answer": ex.get("answer", ex.get("label", None)),  # "1"/"2"
        })
    return out

def build_winogrande_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exs = []
    for r in rows:
        if r["answer"] is None:
            continue
        ans = str(r["answer"])
        gold = "A" if ans in ("1", "0") else "B"
        if ans == "2":
            gold = "B"
        question_text = f"Sentence: {r['sentence']}"
        choices = [r["option1"], r["option2"]]
        exs.append({
            "hf_id": r["_hf_id"],
            "task": "winogrande",
            "question_text": question_text,
            "choices": choices,
            "label_space": ["A", "B"],
            "gold_label": gold,
        })
    return exs

def load_sciq_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/sciq")
    out = []
    for i, ex in enumerate(ds[split]):
        out.append({
            "_hf_id": f"sciq/{split}/{i}",
            "question": ex.get("question", ""),
            "correct": ex.get("correct_answer", ""),
            "d1": ex.get("distractor1", ""),
            "d2": ex.get("distractor2", ""),
            "d3": ex.get("distractor3", ""),
        })
    return out

def build_sciq_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    labels = ["A", "B", "C", "D"]
    exs = []
    for r in rows:
        choices = [r["correct"], r["d1"], r["d2"], r["d3"]]
        idxs = list(range(4))
        rng = random.Random(stable_hash_int(r["_hf_id"]))
        rng.shuffle(idxs)
        shuffled = [choices[j] for j in idxs]
        gold_pos = idxs.index(0)
        gold = labels[gold_pos]
        question_text = f"Question: {r['question']}"
        exs.append({
            "hf_id": r["_hf_id"],
            "task": "sciq",
            "question_text": question_text,
            "choices": shuffled,
            "label_space": labels,
            "gold_label": gold,
        })
    return exs

def load_arc_easy_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    out = []

    def choices_to_map(ch) -> Dict[str, str]:
        if ch is None:
            return {}
        if isinstance(ch, str):
            ch_str = ch.strip()
            if (ch_str.startswith("{") and ch_str.endswith("}")) or (ch_str.startswith("[") and ch_str.endswith("]")):
                try:
                    return choices_to_map(json.loads(ch_str))
                except json.JSONDecodeError:
                    return {}
            return {}
        if isinstance(ch, dict):
            if "label" in ch and "text" in ch:
                labels = ch["label"]
                texts = ch["text"]
                if isinstance(labels, list):
                    if not isinstance(texts, list):
                        texts = [texts] * len(labels)
                    return {str(l): to_text(t) for l, t in zip(labels, texts)}
                return {str(labels): to_text(texts)}
            return {str(k): to_text(v) for k, v in ch.items()}
        if isinstance(ch, list):
            if not ch:
                return {}
            if all(isinstance(c, dict) for c in ch):
                out_map = {c.get("label"): to_text(c.get("text", "")) for c in ch if c.get("label")}
                if out_map:
                    return out_map
            labels = [chr(ord("A") + j) for j in range(len(ch))]
            return {lab: to_text(c.get("text", c) if isinstance(c, dict) else c) for lab, c in zip(labels, ch)}
        return {}

    for i, ex in enumerate(ds[split]):
        q = ex.get("question", {})
        stem = ""
        choices_map: Dict[str, str] = {}
        if isinstance(q, dict):
            stem = q.get("stem", "")
            choices_map = choices_to_map(q.get("choices", []))
        else:
            stem = "" if q is None else str(q)
            choices_map = choices_to_map(ex.get("choices") or ex.get("options") or [])
        answer = (ex.get("answerKey") or "").strip()
        out.append({
            "_hf_id": f"arc_easy/{split}/{i}",
            "stem": stem,
            "choices_map": choices_map,
            "answerKey": answer,
        })
    return out

def build_arc_easy_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exs = []
    for r in rows:
        if not r["choices_map"]:
            continue
        labels = sorted(r["choices_map"].keys())
        choices = [to_text(r["choices_map"][lab]) for lab in labels]
        gold = (r["answerKey"] or "").strip()
        if gold not in labels and gold.isdigit():
            idx = int(gold) - 1
            if 0 <= idx < len(labels):
                gold = labels[idx]
        if gold not in labels:
            continue
        question_text = f"Question: {r['stem']}"
        exs.append({
            "hf_id": r["_hf_id"],
            "task": "arc_easy",
            "question_text": question_text,
            "choices": choices,
            "label_space": labels,
            "gold_label": gold,
        })
    return exs

def prepare_dataset(dataset_name: str, tokenizer, seed: int, train_frac: float, fewshot_k: int):
    """
    Returns: train_examples, val_examples, fewshot_context
    Rules:
      - piqa / winogrande: use HF train split -> split 0.8/0.2, fewshot from train (removed)
      - sciq / arc_easy: use HF train as train, HF validation as val, fewshot from val (removed)
    """
    if dataset_name == "piqa":
        train_rows = load_piqa_split("train")
        exs = build_piqa_examples(train_rows)
        tr, va = split_list(exs, train_frac, seed)
        few, tr = pop_fewshot(tr, fewshot_k, seed + 13)
        ctx = build_fewshot_context(few, tokenizer)
        return tr, va, ctx

    if dataset_name == "winogrande":
        train_rows = load_winogrande_split("train")
        exs = build_winogrande_examples(train_rows)
        tr, va = split_list(exs, train_frac, seed)
        few, tr = pop_fewshot(tr, fewshot_k, seed + 17)
        ctx = build_fewshot_context(few, tokenizer)
        return tr, va, ctx

    if dataset_name == "sciq":
        tr_rows = load_sciq_split("train")
        va_rows = load_sciq_split("validation")
        tr = build_sciq_examples(tr_rows)
        va = build_sciq_examples(va_rows)
        few, va = pop_fewshot(va, fewshot_k, seed + 23)  # remove from val to avoid leakage in eval
        ctx = build_fewshot_context(few, tokenizer)
        return tr, va, ctx

    if dataset_name == "arc_easy":
        tr_rows = load_arc_easy_split("train")
        va_rows = load_arc_easy_split("validation")
        tr = build_arc_easy_examples(tr_rows)
        va = build_arc_easy_examples(va_rows)
        few, va = pop_fewshot(va, fewshot_k, seed + 29)  # remove from val to avoid leakage in eval
        ctx = build_fewshot_context(few, tokenizer)
        return tr, va, ctx

    raise ValueError(dataset_name)

# ----------------------------- Torch Dataset -----------------------------
class MCQATokenDataset(Dataset):
    """
    input_ids = tokenize(prompt_with_fewshot) + [gold_label_token] (single token)
    labels    = -100 for prompt tokens, gold_label_token_id for last token
    pred_pos  = index where logits predict the label token (the token right before the label token)
    """
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, fewshot_context: str = "", max_length: int = 512):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length
        self.fewshot_context = fewshot_context

        for ex in self.examples:
            prompt, answer_prefix, label_token_ids = format_mcqa_prompt(
                ex["question_text"], ex["choices"], ex["label_space"], tokenizer, fewshot_context=fewshot_context
            )
            ex["prompt"] = prompt
            ex["answer_prefix"] = answer_prefix
            ex["label_token_ids"] = label_token_ids
            ex["gold_token_id"] = label_token_ids[ex["gold_label"]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        prompt_ids = self.tok.encode(ex["prompt"], add_special_tokens=False)
        gold_id = ex["gold_token_id"]
        input_ids = prompt_ids + [gold_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            prompt_ids = prompt_ids[-(self.max_length - 1):]

        pred_pos = len(input_ids) - 2
        labels = [-100] * (len(input_ids) - 1) + [gold_id]

        return {
            "hf_id": ex["hf_id"],
            "task": ex["task"],
            "prompt": ex["prompt"],
            "label_space": ex["label_space"],
            "gold_label": ex["gold_label"],
            "label_token_ids": ex["label_token_ids"],
            "gold_token_id": gold_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pred_pos": pred_pos,
        }

def collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, labels, attention_mask = [], [], []
    pred_pos = []
    meta = {k: [] for k in ["hf_id", "task", "prompt", "label_space", "gold_label", "label_token_ids", "gold_token_id"]}

    for x in batch:
        L = x["input_ids"].shape[0]
        pad_n = max_len - L
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)], dim=0))
        labels.append(torch.cat([x["labels"], torch.full((pad_n,), -100, dtype=torch.long)], dim=0))
        attention_mask.append(torch.cat([torch.ones((L,), dtype=torch.long), torch.zeros((pad_n,), dtype=torch.long)], dim=0))
        pred_pos.append(x["pred_pos"])
        for k in meta.keys():
            meta[k].append(x[k])

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "pred_pos": torch.tensor(pred_pos, dtype=torch.long),
        **meta,
    }

# ----------------------------- Debug / eval utils -----------------------------
def get_option_logits(logits_vec: torch.Tensor, label_space: List[str], label_token_ids: Dict[str, int]) -> torch.Tensor:
    ids = [label_token_ids[lab] for lab in label_space]
    return logits_vec[torch.tensor(ids, device=logits_vec.device)]

@torch.no_grad()
def sanity_check_next_token(model, tokenizer, examples: List[Dict[str, Any]], fewshot_context: str, n: int, device: str):
    print("\n[Sanity check] Next-token label distribution at Answer position (pre-train):")
    model.eval()
    for ex in examples[:n]:
        # Build prompt only (no gold appended)
        prompt, ans_prefix, label_token_ids = format_mcqa_prompt(
            ex["question_text"], ex["choices"], ex["label_space"], tokenizer, fewshot_context=fewshot_context
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids = ids[-512:]  # safety
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=input_ids)
        lv = out.logits[0, -1, :]  # next-token logits
        opt_logits = get_option_logits(lv, ex["label_space"], label_token_ids)
        probs = torch.softmax(opt_logits.float(), dim=-1).cpu().tolist()
        top = max(range(len(probs)), key=lambda i: probs[i])
        print(f"- {ex['hf_id']} | top={ex['label_space'][top]} prob={probs[top]:.3f} | probs={dict(zip(ex['label_space'], [round(p,3) for p in probs]))}")
    model.train()
    print()

@torch.no_grad()
def eval_accuracy(model, dataloader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits
        B = logits.shape[0]
        for i in range(B):
            pos = int(pred_pos[i].item())
            lv = logits[i, pos, :]
            label_space = batch["label_space"][i]
            label_token_ids = batch["label_token_ids"][i]
            gold_label = batch["gold_label"][i]
            opt_logits = get_option_logits(lv, label_space, label_token_ids)
            pred_idx = int(torch.argmax(opt_logits).item())
            gold_idx = label_space.index(gold_label)
            correct += int(pred_idx == gold_idx)
            total += 1
    model.train()
    return correct / max(1, total)

# ----------------------------- Attribution State & Logic -----------------------------
@dataclass
class AttrState:
    score_sum: float = 0.0
    count: int = 0

def get_last_layer_params(model):
    """Last Linear Layer Weights Only"""
    if hasattr(model, "embed_out"): return [model.embed_out.weight]
    elif hasattr(model, "lm_head"): return [model.lm_head.weight]
    else: return [list(model.parameters())[-1]]

def compute_grads(model, batch, device):
    """Compute gradients of Loss w.r.t Last Layer Params (Flattened)"""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    loss = outputs.loss
    target_params = get_last_layer_params(model)
    grads = torch.autograd.grad(loss, target_params, create_graph=False, retain_graph=False)
    return torch.cat([g.view(-1) for g in grads]), loss.item()

def compute_per_sample_grads(model, batch, device):
    """Compute per-sample gradients (Inefficient loop but exact)"""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    B = input_ids.shape[0]
    per_sample_grads = []
    for i in range(B):
        outputs = model(input_ids=input_ids[i:i+1], attention_mask=attn[i:i+1], labels=labels[i:i+1])
        target_params = get_last_layer_params(model)
        grads = torch.autograd.grad(outputs.loss, target_params)
        per_sample_grads.append(torch.cat([g.view(-1) for g in grads]).detach().cpu())
    return per_sample_grads

def compute_validation_gradient(model, val_dl, device):
    """Compute average gradient over the validation set"""
    print("Computing Validation Gradient...")
    val_grad_sum = None
    val_cnt = 0
    model.eval() # Ensure eval mode for validation stats
    for batch in tqdm(val_dl, desc="Validation Grad"):
        flat_grad, _ = compute_grads(model, batch, device)
        if val_grad_sum is None:
            val_grad_sum = torch.zeros_like(flat_grad)
        val_grad_sum += flat_grad
        val_cnt += 1
    model.train() # Restore train mode (caller handles this if needed)
    return (val_grad_sum / val_cnt).to(device)

def empirical_fisher_hvp_batch(model, batch, device, v, max_samples=None):
    """Compute (sum_i g_i g_i^T) v over a batch (Empirical Fisher HVP)."""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    B = input_ids.shape[0]
    if max_samples is not None:
        B = min(B, max_samples)
    hvp = torch.zeros_like(v)
    for i in range(B):
        outputs = model(input_ids=input_ids[i:i+1], attention_mask=attn[i:i+1], labels=labels[i:i+1])
        target_params = get_last_layer_params(model)
        grads = torch.autograd.grad(outputs.loss, target_params, create_graph=False, retain_graph=False)
        g = torch.cat([g.view(-1) for g in grads])
        if g.dtype != v.dtype:
            g = g.to(v.dtype)
        alpha = torch.dot(g, v)
        hvp.add_(g, alpha=alpha)
    return hvp, B

def cg_solve(hvp_fn, b, max_iter=100, tol=1e-4):
    """Conjugate Gradient solver for Hx=b using matrix-free HVP."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    if torch.sqrt(rsold) < tol:
        return x
    for _ in range(max_iter):
        Ap = hvp_fn(p)
        denom = torch.dot(p, Ap)
        if denom.abs() < 1e-12:
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def compute_ihvp_cg(model, train_dl, device, v, samples=1000, damping=1e-3, cg_max_iter=100, cg_tol=1e-4):
    """Compute IHVP using Empirical Fisher + CG (matrix-free)."""
    print(f"Computing IHVP via CG using {samples} samples...")
    model.eval()
    v = v.to(device)
    if v.dtype in (torch.float16, torch.bfloat16):
        v = v.float()

    hvp_batches = []
    seen_samples = 0
    for batch in train_dl:
        if seen_samples >= samples:
            break
        remaining = samples - seen_samples
        batch_count = min(remaining, batch["input_ids"].shape[0])
        hvp_batches.append((batch, batch_count))
        seen_samples += batch_count
    if seen_samples == 0:
        raise RuntimeError("Hessian estimation saw 0 samples. Check train_dl or --hessian_samples.")

    def hvp_fn(p):
        hvp_sum = torch.zeros_like(p)
        seen_samples = 0
        for batch, batch_count in hvp_batches:
            hvp_b, cnt = empirical_fisher_hvp_batch(model, batch, device, p, max_samples=batch_count)
            hvp_sum += hvp_b
            seen_samples += cnt
        hvp_avg = hvp_sum / seen_samples
        return hvp_avg + damping * p

    return cg_solve(hvp_fn, v, max_iter=cg_max_iter, tol=cg_tol)

# ----------------------------- Main training + scoring -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="piqa", choices=["piqa","winogrande","sciq","arc_easy"])
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-1b")
    ap.add_argument("--root_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/buckets/data_attribution'))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    # DA Methods
    ap.add_argument("--da_type", type=str, default="grad_align", choices=["grad_align", "tracin", "influence"],  help="Method: grad_align, tracin (requires --do_train), influence (usually static)")
    # Training Controls
    ap.add_argument("--do_train", action="store_true", help="If set, run actual training loop (accumulate scores). Required for TracIn.")
    ap.add_argument("--epochs", type=int, default=5, help="Number of epochs if do_train is True. If False, runs 1 epoch inference.")
    ap.add_argument("--lr", type=float, default=2e-5, help="Optimizer LR for training")
    # Method Params
    ap.add_argument("--if_damping", type=float, default=1e-3, help="Damping for Influence Function")
    ap.add_argument("--hessian_samples", type=int, default=1500, help="Samples for Hessian estimation")
    ap.add_argument("--cg_max_iter", type=int, default=100, help="Max CG iterations for IHVP")
    ap.add_argument("--cg_tol", type=float, default=1e-4, help="CG tolerance for IHVP")
    ap.add_argument("--max_items", type=int, default=0, help="0 means use all")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--score_every", type=int, default=20)
    ap.add_argument("--bucket_frac", type=float, default=0.15)
    ap.add_argument("--bucket_k", type=int, default=0)
    # Few-shot controls
    ap.add_argument("--fewshot_k", type=int, default=3, help="2~3 recommended")
    ap.add_argument("--train_frac", type=float, default=0.8, help="piqa/winogrande train split fraction")
    ap.add_argument("--sanity_check", action="store_false", help="print next-token label distribution before training")
    ap.add_argument("--sanity_n", type=int, default=2)
    # Debug / eval
    ap.add_argument("--eval_each_epoch", action="store_false", help="compute val accuracy each epoch")
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--wandb_project", type=str, default="data_mi")
    ap.add_argument("--wandb_run_name", type=str, default="")
    args = ap.parse_args()

    # default setting
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wb = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.da_type}_{args.dataset}_{now_str()}",
        config=vars(args),
    )

    # load model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Loading Model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,).to(device)
    model.train()

    # Prepare dataset + fewshot context
    train_examples, val_examples, fewshot_ctx = prepare_dataset(args.dataset, tok, args.seed, args.train_frac, args.fewshot_k)
    if args.max_items and args.max_items > 0:
        train_examples = train_examples[:args.max_items]
    print(f"Data: {len(train_examples)} Train, {len(val_examples)} Val")

    # Optional sanity check: does model put mass on label tokens?
    if args.sanity_check:
        sanity_check_next_token(model, tok, train_examples, fewshot_ctx, args.sanity_n, device)

    # Build dataloaders
    train_ds = MCQATokenDataset(train_examples, tok, fewshot_context=fewshot_ctx, max_length=args.max_length)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tok.pad_token_id),
        drop_last=False,
    )

    val_dl = None
    if val_examples:
        val_ds = MCQATokenDataset(val_examples, tok, fewshot_context=fewshot_ctx, max_length=args.max_length)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, tok.pad_token_id),
            drop_last=False,
        )

    if val_dl is None:
        raise RuntimeError("Validation split is required for DA scoring (grad_align/tracin/influence).")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.do_train else None
    # Score Storage
    states: Dict[str, AttrState] = {ex["hf_id"]: AttrState() for ex in train_examples}
    seen_batches = 0
    # -------------------------------------------------------------------------
    # Pre-computation (Static Analysis Phase)
    # -------------------------------------------------------------------------
    target_vector = None # Commonly used for Influence or Static GradAlign

    if not args.do_train: # Static Mode
        print("[Mode] Static Analysis (No Training Loop)")
        
        # 1. Validation Gradient (Common for GradAlign / Influence)
        g_val = compute_validation_gradient(model, val_dl, device)
        
        if args.da_type == "grad_align":
            target_vector = g_val
            print("Target: Static Validation Gradient")
            
        elif args.da_type == "influence":
            # IHVP via Empirical Fisher + CG (matrix-free)
            ihvp = compute_ihvp_cg(
                model,
                train_dl,
                device,
                g_val,
                args.hessian_samples,
                args.if_damping,
                args.cg_max_iter,
                args.cg_tol,
            )
            target_vector = -ihvp
            print("Target: - H^-1 * g_val (IHVP)")
            
        elif args.da_type == "tracin":
            print("Warning: TracIn usually requires training (`--do_train`). Calculating Single-Step TracIn (equivalent to GradAlign * LR).")
            target_vector = g_val * args.lr
        
        # Static scoring: disable dropout etc.
        model.eval()
            
    else: # Dynamic Mode (Training Loop)
        print(f"[Mode] Dynamic Analysis (Training {args.epochs} Epochs)")
        # For Influence, we generally don't recompute Hessian every step.
        # If user forces influence + do_train, we treat it as "Average Influence over trajectory" (very expensive/uncommon).
        # We will disable Influence in dynamic mode for simplicity unless you want it.
        if args.da_type == "influence":
            raise ValueError("Influence Function is typically a static metric. Please run without --do_train.")

    # -------------------------------------------------------------------------
    # Main Loop (Training / Scoring)
    # -------------------------------------------------------------------------
    total_epochs = args.epochs if args.do_train else 1
    
    # If using Training Loader for scoring, we need it ordered or tracked by ID.
    # We use the existing train_dl which is shuffled. We track by hf_id.
    
    for epoch in range(total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        
        # [Dynamic Step] Update Target Vector per Epoch if Training
        if args.do_train:
            print("Computing Validation Gradient for current epoch...")
            g_val_dynamic = compute_validation_gradient(model, val_dl, device)
            
            if args.da_type == "grad_align":
                target_vector = g_val_dynamic
            elif args.da_type == "tracin":
                # TracIn: \eta * (g_train . g_val)
                # We incorporate eta (LR) here or in the dot product.
                current_lr = optim.param_groups[0]['lr']
                target_vector = g_val_dynamic * current_lr
        
        if args.do_train:
            model.train()  # training mode (dropout etc.)
        else:
            model.eval()   # static scoring mode

        for batch in tqdm(train_dl, desc=f"Ep {epoch} Scoring"):
            # 1. Compute Gradients for this batch
            batch_grads = compute_per_sample_grads(model, batch, device)
            
            # 2. Score Calculation
            for j, g_i in enumerate(batch_grads):
                g_i = g_i.to(device)
                score = torch.dot(g_i, target_vector).item()
                
                # Accumulate
                hf_id = batch["hf_id"][j]
                states[hf_id].score_sum += score
                states[hf_id].count += 1
            
            # 3. Optimization Step (Only if Training)
            if args.do_train:
                seen_batches += 1
                # We need to re-forward/backward for the actual update because 
                # compute_per_sample_grads detaches graph or is inefficient for update.
                # Standard Trainer flow:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = outputs.loss / max(1, args.grad_accum)
                loss.backward()
                if seen_batches % max(1, args.grad_accum) == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)

        # Eval each epoch (optional, only meaningful when training)
        if args.do_train and args.eval_each_epoch and val_dl is not None:
            acc = eval_accuracy(model, val_dl, device)
            print(f"[val] epoch={epoch} acc={acc:.4f}")
            wb.log({"val/acc": float(acc), "epoch": epoch})

    # -------------------------------------------------------------------------
    # Final Save
    # -------------------------------------------------------------------------
    rows = []
    for ex in train_examples:
        sid = ex["hf_id"]
        if sid not in states: continue
        st = states[sid]
        
        # If TracIn, we want the Sum. If GradAlign/Influence, usually the Mean (or single shot).
        # But TracIn is also often normalized by number of checkpoints. 
        # Let's save both Sum and Mean for flexibility.
        avg_score = st.score_sum / max(1, st.count)
        
        rows.append({
            "hf_id": sid,
            "task": ex["task"],
            "question_text": ex["question_text"],
            "choices": ex.get("choices"),
            "label_space": ex.get("label_space"),
            "gold_label": ex.get("gold_label"),
            "gold_token_id": ex.get("gold_token_id"),
            "label_token_ids": ex.get("label_token_ids"),
            "prompt": ex.get("prompt", None),
            "tokens_len": len(tok.encode(ex.get("prompt", ""))) + 1,
            "fewshot_k": args.fewshot_k,
            "scores": {
                args.da_type: avg_score,
                f"{args.da_type}_sum": st.score_sum
            },
            "counts": {
                "score_cnt": st.count
            },
        })

    def make_buckets(score_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        vals = []
        for r in rows:
            v = r["scores"].get(score_key, None)
            if v is None:
                continue
            vals.append((v, r))
        vals.sort(key=lambda x: x[0], reverse=True)
        n = len(vals)
        if n == 0:
            return [], []
        k = args.bucket_k if args.bucket_k and args.bucket_k > 0 else max(1, int(args.bucket_frac * n))
        high = [x[1] for x in vals[:k]]
        low = [x[1] for x in vals[-k:]]
        return high, low

    meta = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "seed": args.seed,
        "da_type": args.da_type,
        "do_train": args.do_train,
        "epochs": args.epochs,
        "lr": args.lr,
        "bucket_frac": args.bucket_frac,
        "bucket_k": args.bucket_k,
        "score_every": args.score_every,
        "fewshot_k": args.fewshot_k,
        "train_frac": args.train_frac,
        "hessian_samples": args.hessian_samples,
        "if_damping": args.if_damping,
        "cg_max_iter": args.cg_max_iter,
        "cg_tol": args.cg_tol,
    }
    
    # Save All
    out_base = os.path.join(args.root_dir, "all", args.dataset)
    save_json({"meta": meta, "items": rows}, os.path.join(out_base, f"all_{args.da_type}_{args.dataset}.json"))
    
    # Save Buckets (High=Helpful)
    high, low = make_buckets(args.da_type)
    out_dir = os.path.join(args.root_dir, args.da_type, args.dataset)
    save_json({"meta": meta, "bucket": "high", "score": args.da_type, "items": high},
              os.path.join(out_dir, f"{args.da_type}_{args.dataset}_high.json"))
    save_json({"meta": meta, "bucket": "low", "score": args.da_type, "items": low},
              os.path.join(out_dir, f"{args.da_type}_{args.dataset}_low.json"))
    
    print(f"Done. Saved to {out_dir}")
    wb.finish()

if __name__ == "__main__":
    main()
