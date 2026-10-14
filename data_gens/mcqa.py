import hashlib
import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

SUPPORTED_DATASETS = {"piqa", "winogrande", "sciq", "arc_easy"}

def canonical_dataset_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in SUPPORTED_DATASETS:
        return key
    raise ValueError(f"Unsupported dataset: {name}. choices={sorted(SUPPORTED_DATASETS)}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_list(items: List[Any], frac: float, seed: int) -> Tuple[List[Any], List[Any]]:
    if not (0.0 < frac < 1.0):
        raise ValueError(f"frac must be in (0,1). got={frac}")
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(frac * len(items))
    a = [items[i] for i in idx[:cut]]
    b = [items[i] for i in idx[cut:]]
    return a, b

def stable_hash_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def to_text(x: Any) -> str:
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

def load_piqa_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("piqa", trust_remote_code=True)
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds[split]):
        out.append(
            {
                "_hf_id": f"piqa/{split}/{i}",
                "goal": ex.get("goal", ""),
                "sol1": ex.get("sol1", ""),
                "sol2": ex.get("sol2", ""),
                "label": ex.get("label", None),
            }
        )
    return out

def build_piqa_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exs: List[Dict[str, Any]] = []
    for r in rows:
        if r["label"] is None:
            continue
        gold = "A" if int(r["label"]) == 0 else "B"
        exs.append(
            {
                "hf_id": r["_hf_id"],
                "task": "piqa",
                "question_text": f"Goal: {r['goal']}",
                "choices": [r["sol1"], r["sol2"]],
                "label_space": ["A", "B"],
                "gold_label": gold,
            }
        )
    return exs

def load_winogrande_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/winogrande", "winogrande_xl")
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds[split]):
        out.append(
            {
                "_hf_id": f"winogrande_xl/{split}/{i}",
                "sentence": ex.get("sentence", ""),
                "option1": ex.get("option1", ""),
                "option2": ex.get("option2", ""),
                "answer": ex.get("answer", ex.get("label", None)),
            }
        )
    return out

def build_winogrande_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exs: List[Dict[str, Any]] = []
    for r in rows:
        if r["answer"] is None:
            continue
        ans = str(r["answer"])
        gold = "A" if ans in ("1", "0") else "B"
        if ans == "2":
            gold = "B"
        exs.append(
            {
                "hf_id": r["_hf_id"],
                "task": "winogrande",
                "question_text": f"Sentence: {r['sentence']}",
                "choices": [r["option1"], r["option2"]],
                "label_space": ["A", "B"],
                "gold_label": gold,
            }
        )
    return exs

def load_sciq_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/sciq")
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds[split]):
        out.append(
            {
                "_hf_id": f"sciq/{split}/{i}",
                "question": ex.get("question", ""),
                "correct": ex.get("correct_answer", ""),
                "d1": ex.get("distractor1", ""),
                "d2": ex.get("distractor2", ""),
                "d3": ex.get("distractor3", ""),
            }
        )
    return out

def build_sciq_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    labels = ["A", "B", "C", "D"]
    exs: List[Dict[str, Any]] = []
    for r in rows:
        choices = [r["correct"], r["d1"], r["d2"], r["d3"]]
        idxs = list(range(4))
        rng = random.Random(stable_hash_int(r["_hf_id"]))
        rng.shuffle(idxs)
        shuffled = [choices[j] for j in idxs]
        gold_pos = idxs.index(0)
        exs.append(
            {
                "hf_id": r["_hf_id"],
                "task": "sciq",
                "question_text": f"Question: {r['question']}",
                "choices": shuffled,
                "label_space": labels,
                "gold_label": labels[gold_pos],
            }
        )
    return exs

def _choices_to_map(ch: Any) -> Dict[str, str]:
    if ch is None:
        return {}
    if isinstance(ch, str):
        ch_str = ch.strip()
        if (ch_str.startswith("{") and ch_str.endswith("}")) or (ch_str.startswith("[") and ch_str.endswith("]")):
            try:
                return _choices_to_map(json.loads(ch_str))
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
        return {
            lab: to_text(c.get("text", c) if isinstance(c, dict) else c)
            for lab, c in zip(labels, ch)
        }
    return {}

def load_arc_easy_split(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds[split]):
        q = ex.get("question", {})
        stem = ""
        choices_map: Dict[str, str] = {}
        if isinstance(q, dict):
            stem = q.get("stem", "")
            choices_map = _choices_to_map(q.get("choices", []))
        else:
            stem = "" if q is None else str(q)
            choices_map = _choices_to_map(ex.get("choices") or ex.get("options") or [])
        answer = (ex.get("answerKey") or "").strip()
        out.append(
            {
                "_hf_id": f"arc_easy/{split}/{i}",
                "stem": stem,
                "choices_map": choices_map,
                "answerKey": answer,
            }
        )
    return out

def build_arc_easy_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exs: List[Dict[str, Any]] = []
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
        exs.append(
            {
                "hf_id": r["_hf_id"],
                "task": "arc_easy",
                "question_text": f"Question: {r['stem']}",
                "choices": choices,
                "label_space": labels,
                "gold_label": gold,
            }
        )
    return exs

def prepare_task_splits(dataset_name: str, seed: int = 42, train_frac: float = 0.8) -> Dict[str, List[Dict[str, Any]]]:
    dataset_name = canonical_dataset_name(dataset_name)
    if dataset_name == "piqa":
        train_rows = load_piqa_split("train")
        tr_all = build_piqa_examples(train_rows)
        tr, va = split_list(tr_all, train_frac, seed)
        te = build_piqa_examples(load_piqa_split("validation"))
        return {"train": tr, "val": va, "test": te}
    if dataset_name == "winogrande":
        train_rows = load_winogrande_split("train")
        tr_all = build_winogrande_examples(train_rows)
        tr, va = split_list(tr_all, train_frac, seed)
        te = build_winogrande_examples(load_winogrande_split("validation"))
        return {"train": tr, "val": va, "test": te}
    if dataset_name == "sciq":
        tr = build_sciq_examples(load_sciq_split("train"))
        va = build_sciq_examples(load_sciq_split("validation"))
        te = build_sciq_examples(load_sciq_split("test"))
        return {"train": tr, "val": va, "test": te}
    if dataset_name == "arc_easy":
        tr = build_arc_easy_examples(load_arc_easy_split("train"))
        va = build_arc_easy_examples(load_arc_easy_split("validation"))
        te = build_arc_easy_examples(load_arc_easy_split("test"))
        return {"train": tr, "val": va, "test": te}
    raise ValueError(dataset_name)

def choose_label_tokenization(tokenizer: Any, label_space: Sequence[str]) -> Tuple[str, Dict[str, int]]:
    candidates = ["\nAnswer:\n", "\nAnswer: ", "\nAnswer:"]
    for prefix in candidates:
        ok = True
        ids: Dict[str, int] = {}
        for lab in label_space:
            a = tokenizer.encode(prefix + str(lab), add_special_tokens=False)
            b = tokenizer.encode(prefix, add_special_tokens=False)
            if len(a) != len(b) + 1:
                ok = False
                break
            ids[str(lab)] = int(a[-1])
        if ok:
            return prefix, ids
    raise ValueError(
        f"Could not find single-token answer formatting for labels={list(label_space)}. "
        "Try labels like A/B/C/D."
    )

def format_mcqa_prompt(question_text: str, choices: Sequence[str], label_space: Sequence[str], tokenizer: Any, prefix_context: str = "",) -> Tuple[str, str, Dict[str, int]]:
    answer_prefix, label_token_ids = choose_label_tokenization(tokenizer, label_space)
    lines = [str(question_text).strip(), ""]
    for lab, ch in zip(label_space, choices):
        lines.append(f"{lab}. {str(ch).strip()}")
    prompt = "\n".join(lines) + answer_prefix
    if prefix_context:
        prompt = prefix_context + prompt
    return prompt, answer_prefix, label_token_ids

class MCQATokenDataset(Dataset):
    """
    input_ids = tokenize(prompt) + [gold_label_token]
    labels    = -100 for prompt tokens, gold_label_token for last token
    pred_pos  = position where logits predict the label token
    """
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
        prefix_context: str = "",
    ):
        self.rows: List[Dict[str, Any]] = []
        self.lengths: List[int] = []

        for ex in examples:
            label_space = [str(x) for x in ex["label_space"]]
            prompt, _, label_token_ids = format_mcqa_prompt(
                question_text=ex["question_text"],
                choices=ex["choices"],
                label_space=label_space,
                tokenizer=tokenizer,
                prefix_context=prefix_context,
            )
            gold_label = str(ex["gold_label"])
            if gold_label not in label_token_ids:
                continue

            gold_id = int(label_token_ids[gold_label])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = prompt_ids + [gold_id]
            if max_length > 0 and len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
            if len(input_ids) < 2:
                continue

            labels = [-100] * (len(input_ids) - 1) + [gold_id]
            row = {
                "hf_id": str(ex.get("hf_id", "")),
                "task": str(ex.get("task", "")),
                "prompt": prompt,
                "label_space": label_space,
                "gold_label": gold_label,
                "label_token_ids": label_token_ids,
                "gold_token_id": gold_id,
                "input_ids": input_ids,
                "labels": labels,
                "pred_pos": len(input_ids) - 2,
            }
            self.rows.append(row)
            self.lengths.append(len(input_ids))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "hf_id": row["hf_id"],
            "task": row["task"],
            "prompt": row["prompt"],
            "label_space": row["label_space"],
            "gold_label": row["gold_label"],
            "label_token_ids": row["label_token_ids"],
            "pred_pos": int(row["pred_pos"]),
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }

def collate_batch(batch: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, labels, attention_mask = [], [], []
    pred_pos = []
    meta = {k: [] for k in ["hf_id", "task", "prompt", "label_space", "gold_label", "label_token_ids"]}
    for x in batch:
        l_i = int(x["input_ids"].shape[0])
        pad_n = max_len - l_i
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"], torch.full((pad_n,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones((l_i,), dtype=torch.long), torch.zeros((pad_n,), dtype=torch.long)]))
        pred_pos.append(int(x["pred_pos"]))
        for k in meta:
            meta[k].append(x[k])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "pred_pos": torch.tensor(pred_pos, dtype=torch.long),
        **meta,
    }

def get_option_logits(logits_vec: torch.Tensor, label_space: Sequence[str], label_token_ids: Dict[str, int]) -> torch.Tensor:
    ids = [int(label_token_ids[str(lab)]) for lab in label_space]
    return logits_vec[torch.tensor(ids, device=logits_vec.device)]

@torch.no_grad()
def eval_accuracy(model: Any, dataloader: Any, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits
        bsz = int(logits.shape[0])
        for i in range(bsz):
            pos = int(pred_pos[i].item())
            lv = logits[i, pos, :]
            label_space = batch["label_space"][i]
            label_token_ids = batch["label_token_ids"][i]
            gold_label = batch["gold_label"][i]

            opt_logits = get_option_logits(lv, label_space, label_token_ids)
            pred_idx = int(torch.argmax(opt_logits).item())
            gold_idx = list(label_space).index(gold_label)
            correct += int(pred_idx == gold_idx)
            total += 1
    model.train()
    return float(correct / max(1, total))

def load_checkpoint_state_dict(ckpt_path: str, map_location: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj
    if isinstance(obj, dict):
        return obj, {}
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
