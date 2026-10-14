from pathlib import Path
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_gens.td_score_collect import (
    build_arc_easy_examples,
    build_piqa_examples,
    build_sciq_examples,
    build_winogrande_examples,
    el2n_from_option_probs,
    global_grad_norm,
    load_arc_easy_split,
    load_piqa_split,
    load_sciq_split,
    load_winogrande_split,
    logits_grad_l2,
    now_str,
    pop_fewshot,
    save_json,
    set_seed,
    split_list,
)
from data_gens.da_score_collect import compute_ihvp_cg, compute_per_sample_grads, compute_validation_gradient

DEFAULT_SYSTEM_PROMPT = "You are a helpful multiple-choice assistant. Reply with only the correct option letter."
DEFAULT_ANSWER_INSTRUCTION = "Return only the correct option letter."
DYNAMIC_SCORE_KEYS = {"aum", "forget", "el2n", "gradlogits_l2"}
DATA_ATTR_SCORE_KEYS = {"grad_align", "tracin", "influence"}

class NullRun:
    def log(self, *_args, **_kwargs) -> None:
        return

    def finish(self) -> None:
        return

def init_run(args: argparse.Namespace) -> NullRun:
    if not args.use_wandb or wandb is None:
        return NullRun()
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.score_group}_{args.dataset}_{now_str()}",
        config=vars(args),
    )

def sanitize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def compute_label_token_ids_from_prompt(tokenizer, prompt: str, label_space: List[str]) -> Dict[str, int]:
    base = tokenizer.encode(prompt, add_special_tokens=False)
    out: Dict[str, int] = {}
    for lab in label_space:
        found = False
        for suffix in (lab, " " + lab, "\n" + lab):
            ids = tokenizer.encode(prompt + suffix, add_special_tokens=False)
            if len(ids) == len(base) + 1 and ids[: len(base)] == base:
                out[lab] = ids[-1]
                found = True
                break
        if not found:
            raise ValueError(
                f"Label {lab!r} is not a single next-token under the rendered prompt. "
                "Adjust the chat instruction or label space."
            )
    return out

def build_user_content(question_text: str, choices: List[str], label_space: List[str], answer_instruction: str) -> str:
    lines = [question_text.strip(), ""]
    for lab, choice in zip(label_space, choices):
        lines.append(f"{lab}. {str(choice).strip()}")
    lines.extend(["", answer_instruction.strip()])
    return "\n".join(lines)

def build_plain_prompt(question_text: str, choices: List[str], label_space: List[str], fewshot_examples: List[Dict[str, Any]], answer_instruction: str,) -> str:
    fewshot_blocks: List[str] = []
    for ex in fewshot_examples:
        fewshot_blocks.append(
            build_user_content(ex["question_text"], ex["choices"], ex["label_space"], answer_instruction)
            + "\n"
            + ex["gold_label"]
            + "\n\n"
        )
    return "".join(fewshot_blocks) + build_user_content(question_text, choices, label_space, answer_instruction) + "\n"

def apply_chat_template_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    name_or_path = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen3" in name_or_path:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)

def render_prompt(tokenizer, question_text: str, choices: List[str], label_space: List[str], fewshot_examples: List[Dict[str, Any]], *, system_prompt: str, answer_instruction: str, use_chat_template: bool,) -> Tuple[str, str]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        for ex in fewshot_examples:
            messages.append(
                {
                    "role": "user",
                    "content": build_user_content(
                        ex["question_text"],
                        ex["choices"],
                        ex["label_space"],
                        answer_instruction,
                    ),
                }
            )
            messages.append({"role": "assistant", "content": ex["gold_label"]})
        messages.append(
            {
                "role": "user",
                "content": build_user_content(question_text, choices, label_space, answer_instruction),
            }
        )
        prompt = apply_chat_template_prompt(tokenizer, messages)
        return prompt, "chat_template"
    prompt = build_plain_prompt(question_text, choices, label_space, fewshot_examples, answer_instruction)
    return prompt, "plain"

def prepare_dataset(dataset_name: str, seed: int, train_frac: float, fewshot_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if dataset_name == "piqa":
        train_rows = load_piqa_split("train")
        exs = build_piqa_examples(train_rows)
        train_examples, val_examples = split_list(exs, train_frac, seed)
        fewshot_examples, train_examples = pop_fewshot(train_examples, fewshot_k, seed + 13)
        return train_examples, val_examples, fewshot_examples
    if dataset_name == "winogrande":
        train_rows = load_winogrande_split("train")
        exs = build_winogrande_examples(train_rows)
        train_examples, val_examples = split_list(exs, train_frac, seed)
        fewshot_examples, train_examples = pop_fewshot(train_examples, fewshot_k, seed + 17)
        return train_examples, val_examples, fewshot_examples
    if dataset_name == "sciq":
        train_examples = build_sciq_examples(load_sciq_split("train"))
        val_examples = build_sciq_examples(load_sciq_split("validation"))
        fewshot_examples, val_examples = pop_fewshot(val_examples, fewshot_k, seed + 23)
        return train_examples, val_examples, fewshot_examples
    if dataset_name == "arc_easy":
        train_examples = build_arc_easy_examples(load_arc_easy_split("train"))
        val_examples = build_arc_easy_examples(load_arc_easy_split("validation"))
        fewshot_examples, val_examples = pop_fewshot(val_examples, fewshot_k, seed + 29)
        return train_examples, val_examples, fewshot_examples
    raise ValueError(dataset_name)

class ApplicationMCQADataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        *,
        fewshot_examples: List[Dict[str, Any]],
        system_prompt: str,
        answer_instruction: str,
        max_length: int,
        use_chat_template: bool,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.answer_instruction = answer_instruction
        self.use_chat_template = use_chat_template
        self.fewshot_examples = fewshot_examples
        self.fewshot_hf_ids = [str(ex["hf_id"]) for ex in fewshot_examples]

        for ex in self.examples:
            prompt, prompt_mode = render_prompt(
                tokenizer,
                ex["question_text"],
                ex["choices"],
                ex["label_space"],
                fewshot_examples,
                system_prompt=system_prompt,
                answer_instruction=answer_instruction,
                use_chat_template=use_chat_template,
            )
            full_prompt = prompt
            full_prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
            max_prompt_len = max(1, max_length - 1)
            used_prompt_ids = full_prompt_ids[-max_prompt_len:]
            used_prompt = tokenizer.decode(
                used_prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            label_token_ids = compute_label_token_ids_from_prompt(tokenizer, used_prompt, ex["label_space"])
            ex["prompt_full"] = full_prompt
            ex["prompt"] = used_prompt
            ex["prompt_mode"] = prompt_mode
            ex["system_prompt"] = system_prompt if prompt_mode == "chat_template" else None
            ex["answer_instruction"] = answer_instruction
            ex["fewshot_hf_ids"] = self.fewshot_hf_ids
            ex["prompt_was_truncated"] = len(full_prompt_ids) != len(used_prompt_ids)
            ex["prompt_token_len_raw"] = len(full_prompt_ids)
            ex["prompt_token_len_used"] = len(used_prompt_ids)
            ex["label_token_ids"] = label_token_ids
            ex["gold_token_id"] = label_token_ids[ex["gold_label"]]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer.encode(ex["prompt"], add_special_tokens=False)
        gold_id = ex["gold_token_id"]
        input_ids = prompt_ids + [gold_id]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
        pred_pos = len(input_ids) - 2
        labels = [-100] * (len(input_ids) - 1) + [gold_id]
        return {
            "hf_id": ex["hf_id"],
            "task": ex["task"],
            "prompt": ex["prompt"],
            "prompt_mode": ex["prompt_mode"],
            "system_prompt": ex["system_prompt"],
            "answer_instruction": ex["answer_instruction"],
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
    meta = {
        key: []
        for key in [
            "hf_id",
            "task",
            "prompt",
            "prompt_mode",
            "system_prompt",
            "answer_instruction",
            "label_space",
            "gold_label",
            "label_token_ids",
            "gold_token_id",
        ]
    }
    for x in batch:
        length = x["input_ids"].shape[0]
        pad_n = max_len - length
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)], dim=0))
        labels.append(torch.cat([x["labels"], torch.full((pad_n,), -100, dtype=torch.long)], dim=0))
        attention_mask.append(
            torch.cat([torch.ones((length,), dtype=torch.long), torch.zeros((pad_n,), dtype=torch.long)], dim=0)
        )
        pred_pos.append(x["pred_pos"])
        for key in meta.keys():
            meta[key].append(x[key])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "pred_pos": torch.tensor(pred_pos, dtype=torch.long),
        **meta,
    }

def build_dataloader(dataset: Dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        drop_last=False,
    )

@torch.no_grad()
def sanity_check_next_token(model, dataset: ApplicationMCQADataset, n: int, device: str, sanity_chars: int) -> None:
    print("\n[Sanity check] Next-token label distribution at assistant answer position:")
    model.eval()
    for ex in dataset.examples[:n]:
        ids = dataset.tokenizer.encode(ex["prompt"], add_special_tokens=False)[-dataset.max_length :]
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids=input_ids).logits[0, -1, :]
        option_logits = logits[
            torch.tensor([ex["label_token_ids"][lab] for lab in ex["label_space"]], device=logits.device)
        ]
        probs = torch.softmax(option_logits.float(), dim=-1).cpu().tolist()
        top_idx = max(range(len(probs)), key=lambda i: probs[i])
        print(
            f"- {ex['hf_id']} | top={ex['label_space'][top_idx]} prob={probs[top_idx]:.3f} | "
            f"probs={dict(zip(ex['label_space'], [round(p, 3) for p in probs]))}"
        )
        print(
            f"  prompt_mode={ex.get('prompt_mode')} "
            f"prompt_token_len_raw={ex.get('prompt_token_len_raw')} "
            f"prompt_token_len_used={ex.get('prompt_token_len_used')} "
            f"prompt_was_truncated={ex.get('prompt_was_truncated')}"
        )
        print(f"  gold_label={ex.get('gold_label')} gold_token_id={ex.get('gold_token_id')}")
        print(f"  label_token_ids={ex.get('label_token_ids')}")
        prompt = ex.get("prompt", "")
        if sanity_chars == 0:
            print("  prompt:")
            print(prompt)
        else:
            prompt_head = prompt[:sanity_chars]
            print(f"  prompt_head(chars={sanity_chars}):")
            print(prompt_head)
        print()
    model.train()
    print()

@torch.no_grad()
def eval_accuracy(model, dataloader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        pred_pos = batch["pred_pos"].to(device)
        for i in range(logits.shape[0]):
            lv = logits[i, int(pred_pos[i].item()), :]
            label_space = batch["label_space"][i]
            label_token_ids = batch["label_token_ids"][i]
            gold_label = batch["gold_label"][i]
            option_logits = lv[torch.tensor([label_token_ids[lab] for lab in label_space], device=lv.device)]
            pred_idx = int(torch.argmax(option_logits).item())
            gold_idx = label_space.index(gold_label)
            correct += int(pred_idx == gold_idx)
            total += 1
    model.train()
    return correct / max(1, total)

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

@dataclass
class AttrState:
    score_sum: float = 0.0
    count: int = 0

def finalize_dynamic_rows(args: argparse.Namespace, tokenizer,train_examples: List[Dict[str, Any]],states: Dict[str, TDState],) -> List[Dict[str, Any]]:
    rows = []
    for ex in train_examples:
        state = states[ex["hf_id"]]
        rows.append(
            {
                "hf_id": ex["hf_id"],
                "task": ex["task"],
                "question_text": ex["question_text"],
                "choices": ex["choices"],
                "label_space": ex["label_space"],
                "gold_label": ex["gold_label"],
                "gold_token_id": ex["gold_token_id"],
                "label_token_ids": ex["label_token_ids"],
                "prompt": ex["prompt"],
                "prompt_full": ex.get("prompt_full"),
                "prompt_mode": ex["prompt_mode"],
                "system_prompt": ex["system_prompt"],
                "answer_instruction": ex["answer_instruction"],
                "fewshot_hf_ids": ex.get("fewshot_hf_ids", []),
                "prompt_was_truncated": ex.get("prompt_was_truncated", False),
                "prompt_token_len_raw": ex.get("prompt_token_len_raw"),
                "prompt_token_len_used": ex.get("prompt_token_len_used"),
                "tokens_len": len(tokenizer.encode(ex["prompt"], add_special_tokens=False)) + 1,
                "fewshot_k": args.fewshot_k,
                "scores": {
                    "aum": state.aum_sum / max(1, state.aum_cnt),
                    "forget": state.forget_cnt,
                    "el2n": (state.el2n_sum / max(1, state.el2n_cnt)) if state.el2n_cnt > 0 else None,
                    "gradlogits_l2": state.grad_sum / max(1, state.grad_cnt),
                },
                "counts": {
                    "aum_cnt": state.aum_cnt,
                    "forget_cnt": state.forget_cnt,
                    "el2n_cnt": state.el2n_cnt,
                    "grad_cnt": state.grad_cnt,
                },
            }
        )
    return rows

def finalize_attr_rows(args: argparse.Namespace, tokenizer, train_examples: List[Dict[str, Any]], states: Dict[str, AttrState],) -> List[Dict[str, Any]]:
    rows = []
    for ex in train_examples:
        state = states[ex["hf_id"]]
        avg_score = state.score_sum / max(1, state.count)
        rows.append(
            {
                "hf_id": ex["hf_id"],
                "task": ex["task"],
                "question_text": ex["question_text"],
                "choices": ex["choices"],
                "label_space": ex["label_space"],
                "gold_label": ex["gold_label"],
                "gold_token_id": ex["gold_token_id"],
                "label_token_ids": ex["label_token_ids"],
                "prompt": ex["prompt"],
                "prompt_full": ex.get("prompt_full"),
                "prompt_mode": ex["prompt_mode"],
                "system_prompt": ex["system_prompt"],
                "answer_instruction": ex["answer_instruction"],
                "fewshot_hf_ids": ex.get("fewshot_hf_ids", []),
                "prompt_was_truncated": ex.get("prompt_was_truncated", False),
                "prompt_token_len_raw": ex.get("prompt_token_len_raw"),
                "prompt_token_len_used": ex.get("prompt_token_len_used"),
                "tokens_len": len(tokenizer.encode(ex["prompt"], add_special_tokens=False)) + 1,
                "fewshot_k": args.fewshot_k,
                "scores": {
                    args.score_key: avg_score,
                    f"{args.score_key}_sum": state.score_sum,
                },
                "counts": {
                    "score_cnt": state.count,
                },
            }
        )
    return rows

def make_buckets(rows: List[Dict[str, Any]], score_key: str, higher_is_better: bool, bucket_frac: float, bucket_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    vals = []
    for row in rows:
        value = row["scores"].get(score_key)
        if value is None:
            continue
        vals.append((value, row))
    vals.sort(key=lambda x: x[0], reverse=higher_is_better)
    if not vals:
        return [], []
    n = len(vals)
    k = bucket_k if bucket_k and bucket_k > 0 else max(1, int(math.floor(bucket_frac * n)))
    return [x[1] for x in vals[:k]], [x[1] for x in vals[-k:]]

def save_dynamic_outputs(args: argparse.Namespace, rows: List[Dict[str, Any]], model_root: str) -> None:
    meta = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "score_group": "dynamic",
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "score_every": args.score_every,
        "bucket_frac": args.bucket_frac,
        "bucket_k": args.bucket_k,
        "fewshot_k": args.fewshot_k,
        "fewshot_hf_ids": getattr(args, "fewshot_hf_ids", []),
        "num_train_examples": getattr(args, "num_train_examples", None),
        "num_val_examples": getattr(args, "num_val_examples", None),
        "train_frac": args.train_frac,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "answer_instruction": args.answer_instruction,
    }
    out_all = os.path.join(model_root, "all", args.dataset, f"all_scores_{args.dataset}.json")
    save_json({"meta": meta, "items": rows}, out_all)
    metric_defs = [("aum", True), ("forget", False), ("el2n", True), ("gradlogits_l2", True)]
    for score_key, higher_is_better in metric_defs:
        high, low = make_buckets(rows, score_key, higher_is_better, args.bucket_frac, args.bucket_k)
        out_dir = os.path.join(model_root, score_key, args.dataset)
        save_json({"meta": meta, "bucket": "high", "score": score_key, "items": high}, os.path.join(out_dir, f"{score_key}_{args.dataset}_high.json"))
        save_json({"meta": meta, "bucket": "low", "score": score_key, "items": low}, os.path.join(out_dir, f"{score_key}_{args.dataset}_low.json"))
        print(f"[{score_key}] saved high/low buckets to {out_dir} (high={len(high)}, low={len(low)})")

def save_attr_outputs(args: argparse.Namespace, rows: List[Dict[str, Any]], model_root: str) -> None:
    meta = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "score_group": "data_attr",
        "score_key": args.score_key,
        "seed": args.seed,
        "do_train": args.do_train,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "score_every": args.score_every,
        "bucket_frac": args.bucket_frac,
        "bucket_k": args.bucket_k,
        "fewshot_k": args.fewshot_k,
        "fewshot_hf_ids": getattr(args, "fewshot_hf_ids", []),
        "num_train_examples": getattr(args, "num_train_examples", None),
        "num_val_examples": getattr(args, "num_val_examples", None),
        "train_frac": args.train_frac,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "answer_instruction": args.answer_instruction,
        "hessian_samples": args.hessian_samples,
        "if_damping": args.if_damping,
        "cg_max_iter": args.cg_max_iter,
        "cg_tol": args.cg_tol,
    }
    out_all = os.path.join(model_root, "all", args.dataset, f"all_{args.score_key}_{args.dataset}.json")
    save_json({"meta": meta, "items": rows}, out_all)
    high, low = make_buckets(rows, args.score_key, True, args.bucket_frac, args.bucket_k)
    out_dir = os.path.join(model_root, args.score_key, args.dataset)
    save_json({"meta": meta, "bucket": "high", "score": args.score_key, "items": high}, os.path.join(out_dir, f"{args.score_key}_{args.dataset}_high.json"))
    save_json({"meta": meta, "bucket": "low", "score": args.score_key, "items": low}, os.path.join(out_dir, f"{args.score_key}_{args.dataset}_low.json"))
    print(f"[{args.score_key}] saved high/low buckets to {out_dir} (high={len(high)}, low={len(low)})")

def run_dynamic(args: argparse.Namespace, model, tokenizer, train_dl: DataLoader, val_dl: Optional[DataLoader], train_examples: List[Dict[str, Any]], run: NullRun) -> None:
    states: Dict[str, TDState] = {ex["hf_id"]: TDState() for ex in train_examples}
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    global_step = 0
    seen_batches = 0
    device = args.device

    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            seen_batches += 1
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss / max(1, args.grad_accum)
            loss.backward()

            with torch.no_grad():
                logits = outputs.logits
                batch_aum, batch_el2n, batch_grad, batch_acc = [], [], [], []
                for i in range(logits.shape[0]):
                    hid = batch["hf_id"][i]
                    state = states[hid]
                    pos = int(batch["pred_pos"][i].item())
                    lv = logits[i, pos, :]
                    label_space = batch["label_space"][i]
                    label_token_ids = batch["label_token_ids"][i]
                    gold_label = batch["gold_label"][i]
                    gold_idx = label_space.index(gold_label)
                    gold_token_id = label_token_ids[gold_label]
                    option_logits = lv[torch.tensor([label_token_ids[lab] for lab in label_space], device=lv.device)]
                    gold_logit = option_logits[gold_idx]
                    other_logits = torch.cat([option_logits[:gold_idx], option_logits[gold_idx + 1 :]], dim=0)
                    other_max = torch.max(other_logits) if other_logits.numel() else gold_logit
                    margin = float((gold_logit - other_max).item())
                    state.aum_sum += margin
                    state.aum_cnt += 1
                    batch_aum.append(margin)

                    pred_idx = int(torch.argmax(option_logits).item())
                    correct = pred_idx == gold_idx
                    if state.prev_correct is not None and state.prev_correct and not correct:
                        state.forget_cnt += 1
                    state.prev_correct = correct
                    batch_acc.append(1.0 if correct else 0.0)

                    if epoch < min(args.epochs, 2):
                        el2n = el2n_from_option_probs(option_logits, gold_idx)
                        state.el2n_sum += el2n
                        state.el2n_cnt += 1
                        batch_el2n.append(el2n)

                    grad_val = logits_grad_l2(lv, gold_token_id)
                    state.grad_sum += grad_val
                    state.grad_cnt += 1
                    batch_grad.append(grad_val)

            if seen_batches % max(1, args.grad_accum) != 0:
                pbar.set_postfix({"loss": float(loss.item() * max(1, args.grad_accum)), "step": global_step})
                continue

            grad_norm = global_grad_norm(model)
            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.score_every == 0:
                log_dict = {
                    "train/loss": float(loss.item() * max(1, args.grad_accum)),
                    "train/global_grad_norm": float(grad_norm),
                    "td/batch_margin_mean": float(sum(batch_aum) / max(1, len(batch_aum))),
                    "td/batch_gradlogits_l2_mean": float(sum(batch_grad) / max(1, len(batch_grad))),
                    "td/batch_acc_mean": float(sum(batch_acc) / max(1, len(batch_acc))),
                    "epoch": epoch,
                    "step": global_step,
                }
                if batch_el2n:
                    log_dict["td/batch_el2n_mean"] = float(sum(batch_el2n) / len(batch_el2n))
                run.log(log_dict)

            pbar.set_postfix({"loss": float(loss.item() * max(1, args.grad_accum)), "step": global_step})

        if args.eval_each_epoch and val_dl is not None:
            acc = eval_accuracy(model, val_dl, device)
            print(f"[val] epoch={epoch} acc={acc:.4f}")
            run.log({"val/acc": float(acc), "epoch": epoch})

    rows = finalize_dynamic_rows(args, tokenizer, train_examples, states)
    save_dynamic_outputs(args, rows, os.path.join(args.root_dir, args.model_tag))

def run_data_attr(args: argparse.Namespace, model, tokenizer, train_dl: DataLoader, val_dl: DataLoader, train_examples: List[Dict[str, Any]], run: NullRun) -> None:
    if val_dl is None:
        raise RuntimeError("Validation split is required for application data attribution scoring.")

    states: Dict[str, AttrState] = {ex["hf_id"]: AttrState() for ex in train_examples}
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.do_train else None
    if not args.do_train:
        g_val = compute_validation_gradient(model, val_dl, args.device)
        if args.score_key == "grad_align":
            target_vector = g_val
        elif args.score_key == "influence":
            target_vector = -compute_ihvp_cg(
                model,
                train_dl,
                args.device,
                g_val,
                args.hessian_samples,
                args.if_damping,
                args.cg_max_iter,
                args.cg_tol,
            )
        else:
            target_vector = g_val * args.lr
        model.eval()
    else:
        if args.score_key == "influence":
            raise ValueError("Influence scoring is only supported in static mode.")
        target_vector = None
        model.train()

    total_epochs = args.epochs if args.do_train else 1
    seen_batches = 0
    for epoch in range(total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs}")
        if args.do_train:
            g_val_dynamic = compute_validation_gradient(model, val_dl, args.device)
            target_vector = g_val_dynamic if args.score_key == "grad_align" else g_val_dynamic * args.lr
            model.train()
        else:
            model.eval()

        for batch in tqdm(train_dl, desc=f"Ep {epoch} Scoring"):
            per_sample_grads = compute_per_sample_grads(model, batch, args.device)
            for i, grad in enumerate(per_sample_grads):
                score = torch.dot(grad.to(args.device), target_vector).item()
                state = states[batch["hf_id"][i]]
                state.score_sum += score
                state.count += 1

            if args.do_train and optim is not None:
                seen_batches += 1
                outputs = model(
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    labels=batch["labels"].to(args.device),
                )
                loss = outputs.loss / max(1, args.grad_accum)
                loss.backward()
                if seen_batches % max(1, args.grad_accum) == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)

        if args.do_train and args.eval_each_epoch:
            acc = eval_accuracy(model, val_dl, args.device)
            print(f"[val] epoch={epoch} acc={acc:.4f}")
            run.log({"val/acc": float(acc), "epoch": epoch})

    rows = finalize_attr_rows(args, tokenizer, train_examples, states)
    save_attr_outputs(args, rows, os.path.join(args.root_dir, args.model_tag))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="sciq", choices=["piqa", "winogrande", "sciq", "arc_easy"])
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model_tag", type=str, default="")
    ap.add_argument("--root_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/buckets/application'))
    ap.add_argument("--score_group", type=str, default="dynamic", choices=["dynamic", "data_attr"])
    ap.add_argument("--score_key", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--score_every", type=int, default=20)
    ap.add_argument("--bucket_frac", type=float, default=0.15)
    ap.add_argument("--bucket_k", type=int, default=0)
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--answer_instruction", type=str, default=DEFAULT_ANSWER_INSTRUCTION)
    ap.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    ap.add_argument("--sanity_check", action="store_true")
    ap.add_argument("--sanity_n", type=int, default=2)
    ap.add_argument("--sanity_chars", type=int, default=800)
    ap.add_argument("--eval_each_epoch", action="store_true")
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="data_mi")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--hessian_samples", type=int, default=1000)
    ap.add_argument("--if_damping", type=float, default=1e-3)
    ap.add_argument("--cg_max_iter", type=int, default=100)
    ap.add_argument("--cg_tol", type=float, default=1e-4)
    ap.set_defaults(use_chat_template=True)
    args = ap.parse_args()
    if not args.model_tag:
        args.model_tag = sanitize_model_tag(args.model_name)
    if not args.score_key:
        args.score_key = "aum" if args.score_group == "dynamic" else "grad_align"
    if args.score_key not in (DYNAMIC_SCORE_KEYS | DATA_ATTR_SCORE_KEYS):
        raise ValueError(
            f"Unsupported score_key: {args.score_key}. "
            f"Choose one of {sorted(DYNAMIC_SCORE_KEYS | DATA_ATTR_SCORE_KEYS)}."
        )
    if args.score_group == "dynamic" and args.score_key not in DYNAMIC_SCORE_KEYS:
        raise ValueError(f"Unsupported dynamic score_key: {args.score_key}")
    if args.score_group == "data_attr" and args.score_key not in DATA_ATTR_SCORE_KEYS:
        raise ValueError("--score_group data_attr requires --score_key in {grad_align, tracin, influence}")
    if args.score_key == "tracin" and not args.do_train:
        print("[WARN] score_key=tracin without --do_train uses a static approximation (validation-gradient dot product * lr).")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run = init_run(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.train()

    train_examples, val_examples, fewshot_examples = prepare_dataset(args.dataset, args.seed, args.train_frac, args.fewshot_k)
    if args.max_items and args.max_items > 0:
        train_examples = train_examples[: args.max_items]

    train_ds = ApplicationMCQADataset(
        train_examples,
        tokenizer,
        fewshot_examples=fewshot_examples,
        system_prompt=args.system_prompt,
        answer_instruction=args.answer_instruction,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )
    val_ds = ApplicationMCQADataset(
        val_examples,
        tokenizer,
        fewshot_examples=fewshot_examples,
        system_prompt=args.system_prompt,
        answer_instruction=args.answer_instruction,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    ) if val_examples else None

    print(f"Model: {args.model_name}")
    print(f"Model tag: {args.model_tag}")
    print(f"Dataset: train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0} fewshot={len(fewshot_examples)}")
    print(f"Prompt mode: {'chat_template' if args.use_chat_template and getattr(tokenizer, 'chat_template', None) else 'plain'}")
    args.fewshot_hf_ids = [str(ex["hf_id"]) for ex in fewshot_examples]
    args.num_train_examples = len(train_ds)
    args.num_val_examples = len(val_ds) if val_ds is not None else 0

    if args.sanity_check:
        sanity_check_next_token(model, train_ds, args.sanity_n, args.device, args.sanity_chars)

    train_dl = build_dataloader(train_ds, tokenizer, args.batch_size, shuffle=True)
    val_dl = build_dataloader(val_ds, tokenizer, args.batch_size, shuffle=False) if val_ds is not None else None

    ensure_dir(os.path.join(args.root_dir, args.model_tag))
    if args.score_group == "dynamic":
        run_dynamic(args, model, tokenizer, train_dl, val_dl, train_ds.examples, run)
    else:
        run_data_attr(args, model, tokenizer, train_dl, val_dl, train_ds.examples, run)

    run.finish()


if __name__ == "__main__":
    main()
