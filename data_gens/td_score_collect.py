from pathlib import Path
import os, re, json, math, time, argparse, random, hashlib
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

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def split_list(items: List[Any], frac: float, seed: int) -> Tuple[List[Any], List[Any]]:
    """Deterministic shuffle + split"""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(frac * len(items))
    a = [items[i] for i in idx[:cut]]
    b = [items[i] for i in idx[cut:]]
    return a, b

def pop_fewshot(items: List[Dict[str, Any]], k: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pick k fewshot examples (deterministic) and remove them from the list."""
    if k <= 0 or len(items) == 0:
        return [], items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    pick = set(idx[:min(k, len(items))])
    few = [items[i] for i in range(len(items)) if i in pick]
    rest = [items[i] for i in range(len(items)) if i not in pick]
    return few, rest

# ----------------------------- Prompt / label token handling -----------------------------
def choose_label_tokenization(tokenizer, label_space):
    # Make sure prefix+label adds exactly one token over prefix alone.
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
    raise ValueError(
        f"Could not find a prefix such that each label in {label_space} becomes a single token right after it. "
        f"Try changing label strings (e.g., 'A','B','C','D') or prompt formatting."
    )

def format_mcqa_prompt(question_text: str, choices: List[str], label_space: List[str], tokenizer, fewshot_context: str = "",) -> Tuple[str, str, Dict[str, int]]:
    """
    Returns:
      prompt_text_without_label (ends with answer_prefix)
      answer_prefix
      label_token_ids: map label -> token_id
    """
    answer_prefix, label_token_ids = choose_label_tokenization(tokenizer, label_space)
    lines = [question_text.strip(), ""]
    for lab, ch in zip(label_space, choices):
        lines.append(f"{lab}. {str(ch).strip()}")
    prompt = "\n".join(lines) + answer_prefix
    if fewshot_context:
        prompt = fewshot_context + prompt
    return prompt, answer_prefix, label_token_ids

def build_fewshot_context(fewshot_examples: List[Dict[str, Any]], tokenizer) -> str:
    """
    Build fewshot blocks:
      <Q + choices + Answer prefix><gold_label>\n\n
    """
    if not fewshot_examples:
        return ""
    blocks = []
    for ex in fewshot_examples:
        p, ans_prefix, _ = format_mcqa_prompt(ex["question_text"], ex["choices"], ex["label_space"], tokenizer, fewshot_context="")
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
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pred_pos": pred_pos,
        }

def collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, labels, attention_mask = [], [], []
    pred_pos = []
    meta = {k: [] for k in ["hf_id", "task", "prompt", "label_space", "gold_label", "label_token_ids"]}

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

# ----------------------------- TD score collectors -----------------------------
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

def get_option_logits(logits_vec: torch.Tensor, label_space: List[str], label_token_ids: Dict[str, int]) -> torch.Tensor:
    ids = [label_token_ids[lab] for lab in label_space]
    return logits_vec[torch.tensor(ids, device=logits_vec.device)]

def logits_grad_l2(logits_vec: torch.Tensor, gold_token_id: int) -> float:
    lv = logits_vec.float()
    logZ = torch.logsumexp(lv, dim=-1)
    logsumexp2 = torch.logsumexp(2.0 * lv, dim=-1)
    sum_p2 = torch.exp(logsumexp2 - 2.0 * logZ)
    p_gold = torch.exp(lv[gold_token_id] - logZ)
    val = torch.sqrt(torch.clamp(sum_p2 - 2.0 * p_gold + 1.0, min=0.0))
    return float(val.item())

def el2n_from_option_probs(option_logits: torch.Tensor, gold_index: int) -> float:
    p = torch.softmax(option_logits.float(), dim=-1)
    y = torch.zeros_like(p)
    y[gold_index] = 1.0
    return float(torch.linalg.norm(p - y, ord=2).item())

def global_grad_norm(model: torch.nn.Module) -> float:
    """Global grad norm across all parameters (Trainer-style)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g.float() ** 2).item())
    return float(math.sqrt(total))

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

# ----------------------------- Main training + scoring -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="piqa", choices=["piqa","winogrande","sciq","arc_easy"])
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-1b")
    ap.add_argument("--root_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/buckets'))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_items", type=int, default=0, help="0 means use all")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
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
        name=args.wandb_run_name or f"{args.dataset}_{now_str()}",
        config=vars(args),
    )

    # load model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,).to(device)
    model.train()

    # Prepare dataset + fewshot context
    train_examples, val_examples, fewshot_ctx = prepare_dataset(args.dataset, tok, args.seed, args.train_frac, args.fewshot_k)
    if args.max_items and args.max_items > 0:
        train_examples = train_examples[:args.max_items]

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

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    states: Dict[str, TDState] = {ex["hf_id"]: TDState() for ex in train_examples}
    global_step = 0
    seen_batches = 0

    # training loop
    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            seen_batches += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            pred_pos = batch["pred_pos"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if seen_batches % args.grad_accum == 0:
                # log global grad norm (Trainer-style)
                gnorm = global_grad_norm(model)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                with torch.no_grad():
                    logits = outputs.logits
                    B = logits.shape[0]

                    # Logging용 임시 리스트
                    batch_aum, batch_el2n, batch_grad, batch_acc = [], [], [], []

                    for i in range(B):
                        hid = batch["hf_id"][i]
                        if hid not in states:
                            continue
                        st = states[hid]

                        pos = int(pred_pos[i].item())
                        lv = logits[i, pos, :]

                        label_space = batch["label_space"][i]
                        label_token_ids = batch["label_token_ids"][i]
                        gold_label = batch["gold_label"][i]
                        gold_token_id = label_token_ids[gold_label]

                        opt_logits = get_option_logits(lv, label_space, label_token_ids)

                        gold_idx = label_space.index(gold_label)
                        gold_logit = opt_logits[gold_idx]
                        if len(label_space) > 1:
                            other = torch.cat([opt_logits[:gold_idx], opt_logits[gold_idx+1:]], dim=0)
                            other_max = torch.max(other)
                        else:
                            other_max = gold_logit

                        # 1. AUM Update
                        margin = float((gold_logit - other_max).item())
                        st.aum_sum += margin
                        st.aum_cnt += 1
                        batch_aum.append(margin)

                        # 2. Forgetting Update
                        pred_idx = int(torch.argmax(opt_logits).item())
                        correct = (pred_idx == gold_idx)
                        if st.prev_correct is not None and st.prev_correct and (not correct):
                            st.forget_cnt += 1
                        st.prev_correct = correct
                        batch_acc.append(1.0 if correct else 0.0)

                        # 3. EL2N Update ([수정] epoch < 2 조건으로 변경하여 초기 학습 전체 커버)
                        if epoch < 2:
                            e = el2n_from_option_probs(opt_logits, gold_idx)
                            st.el2n_sum += e
                            st.el2n_cnt += 1
                            batch_el2n.append(e)

                        # 4. Grad Norm Update
                        g = logits_grad_l2(lv, gold_token_id)
                        st.grad_sum += g
                        st.grad_cnt += 1
                        batch_grad.append(g)

                    if (global_step % args.score_every) == 0:
                        log_dict = {
                            "train/loss": float(loss.item() * args.grad_accum),
                            "train/global_grad_norm": float(gnorm),
                            "td/batch_margin_mean": float(sum(batch_aum)/max(1,len(batch_aum))),
                            "td/batch_gradlogits_l2_mean": float(sum(batch_grad)/max(1,len(batch_grad))),
                            "td/batch_acc_mean": float(sum(batch_acc)/max(1,len(batch_acc))),
                            "step": global_step,
                            "epoch": epoch,
                        }
                        if batch_el2n:
                            log_dict["td/batch_el2n_mean"] = float(sum(batch_el2n)/len(batch_el2n))
                        
                        wb.log(log_dict)

                # tqdm postfix / prints
                pbar.set_postfix({"loss": float(loss.item() * args.grad_accum), "gnorm": float(gnorm), "step": global_step})
                if args.print_every > 0 and (global_step % args.print_every) == 0:
                    print(f"[debug] epoch={epoch} step={global_step} loss={float(loss.item()*args.grad_accum):.4f} gnorm={gnorm:.2f}")

        # Eval each epoch (optional)
        if args.eval_each_epoch and val_dl is not None:
            acc = eval_accuracy(model, val_dl, device)
            print(f"[val] epoch={epoch} acc={acc:.4f}")
            wb.log({"val/acc": float(acc), "epoch": epoch})

    # ----------------------------- Finalize scores -----------------------------
    rows = []
    for ex in train_examples:
        hid = ex["hf_id"]
        # Skip if state not found (rare edge case)
        if hid not in states:
            continue
            
        st = states[hid]
        aum = st.aum_sum / max(1, st.aum_cnt)
        el2n = st.el2n_sum / max(1, st.el2n_cnt) if st.el2n_cnt > 0 else None
        grad = st.grad_sum / max(1, st.grad_cnt)
        forget = st.forget_cnt

        rows.append({
            "hf_id": hid,
            "task": ex["task"],
            "question_text": ex["question_text"],
            "choices": ex["choices"],
            "label_space": ex["label_space"],
            "gold_label": ex["gold_label"],
            "gold_token_id": ex.get("gold_token_id"), # [추가] EAP-IG 호환성을 위해 gold_token_id 저장 필수
            "prompt": ex.get("prompt", None),
            "tokens_len": len(tok.encode(ex.get("prompt", ""))) + 1, # [추가] Truncation 여부 확인용 길이 정보 (Approximate)
            "fewshot_k": args.fewshot_k,
            "scores": {
                "aum": aum,
                "forget": forget,
                "el2n": el2n,
                "gradlogits_l2": grad,
            },
            "counts": {
                "aum_cnt": st.aum_cnt,
                "el2n_cnt": st.el2n_cnt,
                "grad_cnt": st.grad_cnt,
                "forget_cnt": st.forget_cnt 
            },
        })

    def make_buckets(score_key: str, higher_is_better: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        vals = []
        for r in rows:
            v = r["scores"].get(score_key, None)
            if v is None:
                continue
            vals.append((v, r))
        vals.sort(key=lambda x: x[0], reverse=higher_is_better)
        n = len(vals)
        if n == 0:
            return [], []
        k = args.bucket_k if args.bucket_k and args.bucket_k > 0 else max(1, int(math.floor(args.bucket_frac * n)))
        high = [x[1] for x in vals[:k]]
        low = [x[1] for x in vals[-k:]]
        return high, low

    metric_defs = [
        ("aum", True),
        ("forget", False),
        ("el2n", True),
        ("gradlogits_l2", True),
    ]

    meta = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "bucket_frac": args.bucket_frac,
        "bucket_k": args.bucket_k,
        "score_every": args.score_every,
        "fewshot_k": args.fewshot_k,
        # "note": {
        #     "prompt": "fewshot_context + (question + choices + Answer prefix) and predict a single label token",
        #     "aum": "mean(option_gold_logit - max_other_option_logit) over observed checkpoints (batch snapshots)",
        #     "forget": "count(correct->incorrect flips) over observed checkpoints (batch snapshots)",
        #     "el2n": "mean L2(p_options - onehot) during early training window (batch snapshots)",
        #     "gradlogits_l2": "mean L2 norm of CE gradient w.r.t vocab logits at answer position (exact, per-example proxy)",
        #     "train/global_grad_norm": "global parameter grad norm per optimizer step (Trainer-style, not per-example)",
        # }
    }

    out_base = os.path.join(args.root_dir, "all", args.dataset)
    save_json({"meta": meta, "items": rows}, os.path.join(out_base, f"all_scores_{args.dataset}.json"))

    for score_key, higher_is_better in metric_defs:
        high, low = make_buckets(score_key, higher_is_better)
        out_dir = os.path.join(args.root_dir, score_key, args.dataset)
        save_json({"meta": meta, "bucket": "high", "score": score_key, "items": high},
                  os.path.join(out_dir, f"{score_key}_{args.dataset}_high.json"))
        save_json({"meta": meta, "bucket": "low", "score": score_key, "items": low},
                  os.path.join(out_dir, f"{score_key}_{args.dataset}_low.json"))
        print(f"[{score_key}] saved high/low buckets to {out_dir} (high={len(high)}, low={len(low)})")

    wb.finish()


if __name__ == "__main__":
    main()
