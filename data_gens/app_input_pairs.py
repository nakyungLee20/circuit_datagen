from pathlib import Path
import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
from transformers import AutoTokenizer

CorruptType = Literal["gaussian", "zero", "mean"]
HookPoint = Literal["hook_embed", "blocks.0.hook_resid_pre"]

@dataclass
class BatchInputCorruptor:
    """
    Apply corruption only on the corrupted forward pass.
    """
    corrupt_type: CorruptType = "gaussian"
    sigma: float = 0.1
    hook_point: HookPoint = "hook_embed"
    seeds: Optional[List[int]] = None
    mean_vec: Optional[torch.Tensor] = None

    def set_batch_seeds(self, seeds: List[int]) -> None:
        self.seeds = seeds

    def _get_mean_vec(self, act: torch.Tensor, model=None) -> torch.Tensor:
        if self.mean_vec is not None:
            mv = self.mean_vec
            if mv.device != act.device:
                mv = mv.to(act.device)
            if mv.dtype != act.dtype:
                mv = mv.to(act.dtype)
            return mv

        if model is not None and hasattr(model, "W_E"):
            return model.W_E.mean(dim=0).to(device=act.device, dtype=act.dtype)
        return act.mean(dim=(0, 1))

    def hook(self, act: torch.Tensor, hook, model=None) -> torch.Tensor:
        if self.corrupt_type == "zero":
            return torch.zeros_like(act)
        if self.corrupt_type == "mean":
            mv = self._get_mean_vec(act, model=model)
            return mv.view(1, 1, -1).expand_as(act)
        if self.sigma <= 0:
            return act

        batch_size = act.shape[0]
        if self.seeds is None:
            return act + torch.randn_like(act) * self.sigma
        if len(self.seeds) != batch_size:
            raise ValueError(f"seeds length {len(self.seeds)} != batch {batch_size}")

        chunks = []
        for i in range(batch_size):
            generator = torch.Generator(device=act.device)
            generator.manual_seed(int(self.seeds[i]))
            noise_i = torch.randn(
                act[i].shape,
                generator=generator,
                device=act.device,
                dtype=act.dtype,
            ) * self.sigma
            chunks.append(noise_i)
        return act + torch.stack(chunks, dim=0)

def make_corrupt_hook_fn(corruptor: BatchInputCorruptor, model=None):
    def _fn(act, hook):
        return corruptor.hook(act, hook, model=model)
    return _fn

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def stable_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def sanitize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")

def normalize_label_token_ids(value: Any) -> Optional[Dict[str, int]]:
    if not isinstance(value, dict):
        return None
    out: Dict[str, int] = {}
    for key, token_id in value.items():
        try:
            out[str(key)] = int(token_id)
        except (TypeError, ValueError):
            return None
    return out

def compute_label_token_ids_from_prompt(tok, prompt: str, label_space: List[str]) -> Dict[str, int]:
    base = tok.encode(prompt, add_special_tokens=False)
    out: Dict[str, int] = {}
    for label in label_space:
        found = False
        for suffix in (label, " " + label, "\n" + label):
            ids = tok.encode(prompt + suffix, add_special_tokens=False)
            if len(ids) == len(base) + 1 and ids[: len(base)] == base:
                out[label] = ids[-1]
                found = True
                break
        if not found:
            raise ValueError(f"Label {label!r} is not a single next-token under the saved prompt.")
    return out

def encode_len(tok, text: str, max_length: int) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    if max_length > 0 and len(ids) > max_length:
        ids = ids[-max_length:]
    return len(ids)

def resolve_bucket_path(args: argparse.Namespace) -> Tuple[str, str]:
    model_tag = args.bucket_model_tag or args.model_tag
    if args.bucket_path:
        return args.bucket_path, model_tag

    fname = f"{args.score_key}_{args.dataset}_{args.bucket}.json"
    candidate_with_tag = os.path.join(
        args.buckets_root_dir,
        model_tag,
        args.score_key,
        args.dataset,
        fname,
    )
    if os.path.exists(candidate_with_tag):
        return candidate_with_tag, model_tag
    candidate_legacy = os.path.join(
        args.buckets_root_dir,
        args.score_key,
        args.dataset,
        fname,
    )
    return candidate_legacy, model_tag

def build_pairs_self_corrupt(
    items: List[Dict[str, Any]],
    tok,
    *,
    dataset: str,
    score_key: str,
    bucket: str,
    seed: int,
    corrupt_type: str,
    noise_sigma: float,
    max_length: int,
    model_name: str,
    model_tag: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dropped = 0
    recomputed_label_ids = 0
    mismatch_option_ids = 0
    mismatch_goldid = 0

    for item in items:
        prompt = item.get("prompt", "")
        if not prompt:
            dropped += 1
            continue

        label_space = item.get("label_space")
        gold_label = item.get("gold_label")
        saved_label_token_ids = normalize_label_token_ids(item.get("label_token_ids"))
        saved_gold_token_id = item.get("gold_token_id")

        if not isinstance(label_space, list) or not label_space or not isinstance(gold_label, str):
            dropped += 1
            continue
        label_space = [str(x) for x in label_space]
        if gold_label not in label_space:
            dropped += 1
            continue

        try:
            label_token_ids = compute_label_token_ids_from_prompt(tok, prompt, label_space)
            recomputed_label_ids += 1
        except Exception:
            if saved_label_token_ids is None:
                dropped += 1
                continue
            label_token_ids = saved_label_token_ids

        if saved_label_token_ids is not None:
            saved_option_ids = [int(saved_label_token_ids[label]) for label in label_space]
            used_option_ids = [int(label_token_ids[label]) for label in label_space]
            if saved_option_ids != used_option_ids:
                mismatch_option_ids += 1

        gold_token_id = int(label_token_ids[gold_label])
        if saved_gold_token_id is not None and int(saved_gold_token_id) != gold_token_id:
            mismatch_goldid += 1

        token_len = encode_len(tok, prompt, max_length)
        hf_id = item.get("hf_id", item.get("_hf_id", ""))
        cseed = stable_int(f"{seed}::{dataset}::{score_key}::{bucket}::{hf_id}")

        row = {
            "clean": prompt,
            "corrupted": prompt,
            "label": gold_token_id,
            "hf_id": hf_id,
            "task": item.get("task", dataset),
            "score_key": score_key,
            "bucket": bucket,
            "model_name": model_name,
            "model_tag": model_tag,
            "prompt_full": item.get("prompt_full"),
            "prompt_mode": item.get("prompt_mode"),
            "system_prompt": item.get("system_prompt"),
            "answer_instruction": item.get("answer_instruction"),
            "fewshot_hf_ids": item.get("fewshot_hf_ids", []),
            "prompt_was_truncated": item.get("prompt_was_truncated", False),
            "prompt_token_len_raw": item.get("prompt_token_len_raw"),
            "prompt_token_len_used": item.get("prompt_token_len_used"),
            "gold_label": gold_label,
            "label_space": label_space,
            "label_token_ids": label_token_ids,
            "saved_label_token_ids": saved_label_token_ids,
            "option_token_ids": [int(label_token_ids[label]) for label in label_space],
            "corrupt": {
                "type": corrupt_type,
                "sigma": float(noise_sigma),
                "seed": int(cseed),
            },
            "token_len": token_len,
            "max_length": int(max_length),
        }
        out.append(row)

    meta = {
        "dataset": dataset,
        "score_key": score_key,
        "bucket": bucket,
        "seed": seed,
        "model_name": model_name,
        "model_tag": model_tag,
        "corrupt_type": corrupt_type,
        "noise_sigma": noise_sigma,
        "max_length": max_length,
        "num_in": len(items),
        "num_out": len(out),
        "dropped": dropped,
        "pair_mode": "self_corrupt",
        "recomputed_label_ids": recomputed_label_ids,
        "mismatch_option_ids": mismatch_option_ids,
        "mismatch_goldid": mismatch_goldid,
    }
    return out, meta

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["piqa", "winogrande", "sciq", "arc_easy"])
    ap.add_argument("--score_key", type=str, required=True, choices=["aum", "forget", "el2n", "gradlogits_l2", "grad_align", "tracin", "influence"],)
    ap.add_argument("--bucket", type=str, required=True, choices=["high", "low"])
    ap.add_argument("--buckets_root_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/buckets/application'))
    ap.add_argument("--bucket_path", type=str, default="")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model_tag", type=str, default="")
    ap.add_argument("--bucket_model_tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--corrupt_type", type=str, default="gaussian", choices=["gaussian", "zero", "mean"])
    ap.add_argument("--noise_sigma", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/pairs/application'))
    ap.add_argument("--out_name", type=str, default="", help="Optional custom filename. If empty, auto-named.")
    args = ap.parse_args()
    if not args.model_tag:
        args.model_tag = sanitize_model_tag(args.model_name)
    return args

def main() -> None:
    args = parse_args()
    bucket_path, bucket_model_tag = resolve_bucket_path(args)
    if not os.path.exists(bucket_path):
        raise FileNotFoundError(
            f"Bucket file not found: {bucket_path}. "
            "Expected application buckets under "
            f"{os.path.join(args.buckets_root_dir, bucket_model_tag, args.score_key, args.dataset)}"
        )
    bucket_obj = read_json(bucket_path)
    items = bucket_obj.get("items", [])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    pairs, meta = build_pairs_self_corrupt(
        items=items,
        tok=tokenizer,
        dataset=args.dataset,
        score_key=args.score_key,
        bucket=args.bucket,
        seed=args.seed,
        corrupt_type=args.corrupt_type,
        noise_sigma=args.noise_sigma,
        max_length=args.max_length,
        model_name=args.model_name,
        model_tag=args.model_tag,
    )

    meta["source_bucket_path"] = bucket_path
    meta["bucket_model_tag"] = bucket_model_tag
    meta["source_bucket_meta"] = bucket_obj.get("meta", {})

    out_dir = os.path.join(args.out_dir, args.model_tag)
    os.makedirs(out_dir, exist_ok=True)
    fname = args.out_name.strip()
    if not fname:
        fname = f"pairs_{args.dataset}_{args.score_key}_{args.bucket}_{args.corrupt_type}.jsonl"
    out_path = os.path.join(out_dir, fname)

    write_jsonl(pairs, out_path)
    meta_path = out_path.replace(".jsonl", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] bucket={bucket_path}")
    print(f"[OK] wrote pairs: {out_path}")
    print(f"[OK] wrote meta : {meta_path}")
    print(
        f"[STATS] in={meta['num_in']} out={meta['num_out']} dropped={meta['dropped']} "
        f"recomputed_label_ids={meta['recomputed_label_ids']} "
        f"mismatch_option_ids={meta['mismatch_option_ids']} mismatch_goldid={meta['mismatch_goldid']}"
    )


if __name__ == "__main__":
    main()
