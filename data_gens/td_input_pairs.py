from pathlib import Path
import os, json, argparse, hashlib
from typing import Any, Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional, List, Literal
import torch

CorruptType = Literal["gaussian", "zero", "mean"]
HookPoint = Literal["hook_embed", "blocks.0.hook_resid_pre"]

@dataclass
class BatchInputCorruptor:
    """
    Apply corruption ONLY on the corrupted forward pass.
    - gaussian: add N(0, sigma^2) noise
    - zero: set activations to 0
    - mean: replace with a fixed mean vector (defaults to vocab-mean of embedding matrix)
    """
    corrupt_type: CorruptType = "gaussian"
    sigma: float = 0.1
    hook_point: HookPoint = "hook_embed"
    # batch-specific seeds (set each batch)
    seeds: Optional[List[int]] = None
    # mean vector (d_model,)
    mean_vec: Optional[torch.Tensor] = None

    def set_batch_seeds(self, seeds: List[int]) -> None:
        self.seeds = seeds

    def _get_mean_vec(self, act: torch.Tensor, model=None) -> torch.Tensor:
        """
        act: (B, S, d_model)
        If mean_vec is not provided, try to use model.W_E.mean(0) when hook at embed.
        Fallback: act.mean over (B,S).
        """
        if self.mean_vec is not None:
            mv = self.mean_vec
            if mv.device != act.device:
                mv = mv.to(act.device)
            if mv.dtype != act.dtype:
                mv = mv.to(act.dtype)
            return mv

        if model is not None and hasattr(model, "W_E"):
            mv = model.W_E.mean(dim=0)
            mv = mv.to(device=act.device, dtype=act.dtype)
            return mv

        mv = act.mean(dim=(0, 1))
        return mv

    def hook(self, act: torch.Tensor, hook, model=None) -> torch.Tensor:
        """
        TransformerLens hook signature: hook_fn(act, hook) -> act
        We'll pass model via closure if needed.
        """
        ct = self.corrupt_type

        if ct == "zero":
            return torch.zeros_like(act)

        if ct == "mean":
            mv = self._get_mean_vec(act, model=model)  # (d_model,)
            return mv.view(1, 1, -1).expand_as(act)

        # gaussian
        if self.sigma <= 0:
            return act

        B = act.shape[0]
        if self.seeds is None:
            # fallback: one generator for the whole batch
            return act + torch.randn_like(act) * self.sigma

        if len(self.seeds) != B:
            raise ValueError(f"seeds length {len(self.seeds)} != batch {B}")

        noise_chunks = []
        for i in range(B):
            g = torch.Generator(device=act.device)
            g.manual_seed(int(self.seeds[i]))
            noise_i = torch.randn(act[i].shape, generator=g, device=act.device, dtype=act.dtype) * self.sigma
            noise_chunks.append(noise_i)
        noise = torch.stack(noise_chunks, dim=0)
        return act + noise

def make_corrupt_hook_fn(corruptor: BatchInputCorruptor, model=None):
    # closure to match TransformerLens hook interface
    def _fn(act, hook):
        return corruptor.hook(act, hook, model=model)
    return _fn

# ---------------- I/O ----------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def stable_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def compute_label_token_ids_from_prompt(tok, prompt: str, label_space: List[str]) -> Dict[str, int]:
    """
    Returns {label_str: token_id} such that the *next token* after `prompt` is that label.
    Assumes prompt ends right before the answer token (e.g., "...Answer:\n").
    """
    base = tok.encode(prompt, add_special_tokens=False)
    out: Dict[str, int] = {}
    for lab in label_space:
        # 가장 기대하는 형태: prompt + lab 가 정확히 한 토큰만 늘어남
        ids = tok.encode(prompt + lab, add_special_tokens=False)
        if len(ids) == len(base) + 1 and ids[:len(base)] == base:
            out[lab] = ids[-1]
            continue
        # fallback: " "+lab, "\n"+lab
        found = False
        for lab2 in [" " + lab, "\n" + lab]:
            ids2 = tok.encode(prompt + lab2, add_special_tokens=False)
            if len(ids2) == len(base) + 1 and ids2[:len(base)] == base:
                out[lab] = ids2[-1]
                found = True
                break
        if not found:
            raise ValueError(f"Label {lab!r} is not a single next-token under this prompt suffix.")
    return out

# ---------------- pairing modes ----------------
def _encode_len(tok, text: str, max_length: int) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    if max_length > 0 and len(ids) > max_length:
        ids = ids[-max_length:]
    return len(ids)

def build_pairs_self_corrupt(items: List[Dict[str, Any]], tok, dataset: str, score_key: str, bucket: str,
    seed: int, corrupt_type: str, noise_sigma: float, max_length: int,) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Each item becomes one pair:
      clean = prompt
      corrupted = prompt (same text)
      BUT we will apply embedding corruption at runtime for the corrupted forward pass.
    """
    out = []
    dropped = 0
    for it in items:
        prompt = it.get("prompt", "")
        if not prompt:
            dropped += 1
            continue

        gold_token_id = it.get("gold_token_id", None)
        label_token_ids = it.get("label_token_ids", None)
        label_space = it.get("label_space", None)

        if gold_token_id is None:
            dropped += 1
            continue

        # gold_label = it.get("gold_label", None)
        # label_space = it.get("label_space", None)
        # gold_token_id_saved = it.get("gold_token_id", None)

        # if not (isinstance(label_space, list) and len(label_space) >= 2 and isinstance(gold_label, str)):
        #     dropped += 1
        #     continue
        # if gold_label not in label_space:
        #     dropped += 1
        #     continue

        # # (핵심) prompt 기반으로 label_token_ids 재구성
        # try:
        #     label_token_ids = compute_label_token_ids_from_prompt(tok, prompt, label_space)
        # except Exception as e:
        #     # prompt 포맷이 깨졌거나, label이 single-token이 아닐 때
        #     dropped += 1
        #     continue

        # option_token_ids = [int(label_token_ids[x]) for x in label_space]
        # gold_token_id = int(label_token_ids[gold_label])

        # # (강력 권장) Step1에서 저장된 gold_token_id와 불일치하면 드롭/경고
        # if gold_token_id_saved is not None and int(gold_token_id_saved) != gold_token_id:
        #     # tokenizer/model mismatch 혹은 Step1 저장값이 다른 prefix 컨텍스트에서 나온 경우
        #     dropped += 1
        #     continue

        # length match check (항상 같아야 함)
        Lc = _encode_len(tok, prompt, max_length)
        Lk = _encode_len(tok, prompt, max_length)
        if Lc != Lk:
            dropped += 1
            continue

        # per-example corruption seed (deterministic)
        hf_id = it.get("hf_id", it.get("_hf_id", ""))
        cseed = stable_int(f"{seed}::{dataset}::{score_key}::{bucket}::{hf_id}")

        row = {
            # EAP-IG에서 보통 필요한 3종 세트(텍스트 pair + label token)
            "clean": prompt,
            "corrupted": prompt,
            "label": int(gold_token_id),
            # "label": gold_token_id,
            # ---- extra metadata (네가 metric/logitdiff 만들 때 유용) ----
            "hf_id": hf_id,
            "task": it.get("task", dataset),
            "score_key": score_key,
            "bucket": bucket,
            "gold_label": it.get("gold_label", None),
            "label_space": label_space,
            "label_token_ids": label_token_ids,
            "option_token_ids": [int(label_token_ids[x]) for x in label_space] if (label_token_ids and label_space) else None,
            # "gold_label": gold_label,
            # "label_space": label_space,
            # "label_token_ids": label_token_ids,
            # "option_token_ids": option_token_ids,
            # ---- corruption config (runtime에서 그대로 사용) ----
            "corrupt": {
                "type": corrupt_type,     # gaussian | zero | mean
                "sigma": float(noise_sigma),
                "seed": int(cseed),
            },
            # ---- length info ----
            "token_len": int(Lc),
            "max_length": int(max_length),
        }
        out.append(row)

    meta = {
        "dataset": dataset,
        "score_key": score_key,
        "bucket": bucket,
        "seed": seed,
        "corrupt_type": corrupt_type,
        "noise_sigma": noise_sigma,
        "max_length": max_length,
        "num_in": len(items),
        "num_out": len(out),
        "dropped": dropped,
        "pair_mode": "self_corrupt",
    }
    return out, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["piqa","winogrande","sciq","arc_easy"])
    ap.add_argument("--score_key", type=str, required=True, choices=["aum","forget","el2n","gradlogits_l2"])
    ap.add_argument("--bucket", type=str, required=True, choices=["high","low"])
    ap.add_argument("--buckets_root_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/buckets'))
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-1b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    # corruption config (saved into jsonl; applied at runtime in EAP-IG)
    ap.add_argument("--corrupt_type", type=str, default="gaussian", choices=["gaussian","zero","mean"])
    ap.add_argument("--noise_sigma", type=float, default=0.1)
    # output
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/pairs'))
    ap.add_argument("--out_name", type=str, default="", help="Optional custom filename. If empty, auto-named.")
    args = ap.parse_args()

    bucket_path = os.path.join(args.buckets_root_dir, args.score_key, args.dataset, f"{args.score_key}_{args.dataset}_{args.bucket}.json")
    obj = read_json(bucket_path)
    items = obj.get("items", [])

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    pairs, meta = build_pairs_self_corrupt(
        items=items,
        tok=tok,
        dataset=args.dataset,
        score_key=args.score_key,
        bucket=args.bucket,
        seed=args.seed,
        corrupt_type=args.corrupt_type,
        noise_sigma=args.noise_sigma,
        max_length=args.max_length,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    fname = args.out_name.strip()
    if not fname:
        fname = f"pairs_{args.dataset}_{args.score_key}_{args.bucket}_{args.corrupt_type}.jsonl"
    out_path = os.path.join(args.out_dir, fname)

    write_jsonl(pairs, out_path)

    # meta json도 같이 저장(디버깅/재현용)
    meta_path = out_path.replace(".jsonl", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] bucket={bucket_path}")
    print(f"[OK] wrote pairs: {out_path}")
    print(f"[OK] wrote meta : {meta_path}")
    print(f"[STATS] in={meta['num_in']} out={meta['num_out']} dropped={meta['dropped']}")

if __name__ == "__main__":
    main()
