from pathlib import Path
import argparse
import glob
import json
import os
import sys
from functools import partial
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from dataclasses import dataclass

# -------------------------
# EAP-IG imports (local clone)
# -------------------------
# EAP-IG imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_EAP_SRC = os.path.join(_CURRENT_DIR, "src")
if _EAP_SRC not in sys.path:
    sys.path.insert(0, _EAP_SRC)

from eap.attribute import attribute
from eap.attribute_node import attribute_node
from eap.evaluate import evaluate_graph
from eap.graph import Graph
import eap.utils as eap_utils
import eap.attribute as eap_attr_mod
import eap.attribute_node as eap_attr_node_mod
import eap.evaluate as eap_eval_mod
from gqa_circuits import load_model_for_circuit_discovery, summarize_model_for_circuit_discovery
from eap.export import edge_export_info

# datasets imports
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

CorruptType = Literal["gaussian", "zero", "mean"]
HookPoint = Literal["hook_embed", "blocks.0.hook_resid_pre"]
from data_gens.td_input_pairs import (BatchInputCorruptor,  make_corrupt_hook_fn,)

# _ROOT = os.path.dirname(os.path.abspath(__file__))
# _SRC = os.path.join(_ROOT, "src")
# if _SRC not in sys.path:
#     sys.path.insert(0, _SRC)
# from eap.attribute import attribute
# from eap.evaluate import evaluate_graph
# from eap.graph import Graph
# import eap.utils as eap_utils
# import eap.attribute as eap_attr_mod
# import eap.evaluate as eap_eval_mod

# -------------------------
# IO utils
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

SCORE_KEYS = ["aum", "forget", "el2n", "gradlogits_l2", "grad_align", "tracin"]
BUCKETS = ["high", "low"]
CORRUPT_TYPES = ["gaussian", "zero", "mean"]

def _parse_pairs_filename(path: str) -> Optional[Dict[str, str]]:
    base = os.path.basename(path)
    if not (base.startswith("pairs_") and base.endswith(".jsonl")):
        return None
    stem = base[:-6]  # strip .jsonl
    for score_key in SCORE_KEYS:
        for bucket in BUCKETS:
            for ct in CORRUPT_TYPES:
                suffix = f"_{score_key}_{bucket}_{ct}"
                if stem.endswith(suffix) and stem.startswith("pairs_"):
                    dataset = stem[len("pairs_"):-len(suffix)]
                    if not dataset:
                        return None
                    return {
                        "dataset": dataset,
                        "score_key": score_key,
                        "bucket": bucket,
                        "corrupt_type": ct,
                        "tag": stem[len("pairs_"):],
                    }
    return None

def resolve_pairs_files(
    pairs_dir: str,
    pairs_path: Optional[str],
    dataset: Optional[str],
    score_key: Optional[str],
    bucket: Optional[str],
    corrupt_type: Optional[str],
    pairs_glob: Optional[str],
) -> List[Tuple[Optional[Dict[str, str]], str]]:
    if pairs_path:
        return [(_parse_pairs_filename(pairs_path), pairs_path)]
    if pairs_glob:
        pattern = pairs_glob if os.path.isabs(pairs_glob) else os.path.join(pairs_dir, pairs_glob)
        recursive = "**" in pattern
    else:
        pattern = os.path.join(pairs_dir, "**", "pairs_*.jsonl")
        recursive = True
    paths = sorted(glob.glob(pattern, recursive=recursive))
    out: List[Tuple[Optional[Dict[str, str]], str]] = []
    for p in paths:
        meta = _parse_pairs_filename(p)
        if meta is None:
            continue
        if dataset and meta["dataset"] != dataset:
            continue
        if score_key and meta["score_key"] != score_key:
            continue
        if bucket and meta["bucket"] != bucket:
            continue
        if corrupt_type and meta["corrupt_type"] != corrupt_type:
            continue
        out.append((meta, p))
    return out

def _build_out_dir(out_root: str, pairs_meta: Optional[Dict[str, str]]) -> str:
    if not pairs_meta:
        return out_root
    return os.path.join(
        out_root,
        pairs_meta.get("dataset", "unknown"),
        pairs_meta.get("score_key", "unknown"),
        pairs_meta.get("bucket", "unknown"),
        pairs_meta.get("corrupt_type", "unknown"),
    )

def str_to_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

@dataclass
class CorruptItem:
    text: str
    corrupt: Dict[str, Any]
    def __str__(self) -> str:
        return self.text

# -------------------------
# tokenizer helpers
# -------------------------
def _label_token_id_from_prefix(tokenizer, prefix: str, label: str, strict: bool) -> int:
    """
    Find single-token id for label under a specific prefix context.
    We require: encode(prefix+label) == encode(prefix) + [token]
    """
    ids_p = tokenizer.encode(prefix, add_special_tokens=False)
    ids_pl = tokenizer.encode(prefix + label, add_special_tokens=False)
    if len(ids_pl) == len(ids_p) + 1 and ids_pl[: len(ids_p)] == ids_p:
        return ids_pl[-1]

    # fallback attempts (common variants)
    variants = [label, " " + label, "\n" + label]
    for lab in variants:
        ids_pl = tokenizer.encode(prefix + lab, add_special_tokens=False)
        if len(ids_pl) == len(ids_p) + 1 and ids_pl[: len(ids_p)] == ids_p:
            return ids_pl[-1]

    msg = f"Label {label!r} not single-token under prefix={prefix!r}."
    if strict:
        raise ValueError(msg)
    print("[WARN]", msg, "Falling back to first token of encode(label).")
    return tokenizer.encode(label, add_special_tokens=False)[0]

def infer_label_space(rows: Sequence[Dict[str, Any]], dataset: str) -> List[str]:
    # row에 label_space가 있으면 그걸 우선 사용
    for r in rows:
        ls = r.get("label_space", None)
        if isinstance(ls, list) and len(ls) >= 2:
            return [str(x) for x in ls]
    # fallback: dataset별 기본값
    if dataset in ["piqa", "winogrande"]:
        return ["A", "B"]
    if dataset in ["sciq", "arc_easy"]:
        return ["A", "B", "C", "D"]
    # unknown
    return ["A", "B"]

def infer_dataset_from_rows(rows: Sequence[Dict[str, Any]], default: str = "piqa") -> str:
    for r in rows:
        t = r.get("task", None)
        if isinstance(t, str) and t:
            return t
    return default

def infer_answer_prefix_candidates(dataset: str) -> List[str]:
    # 네 prompt 포맷(예: "Answer:\n")에 맞춰 후보들을 둠
    return [
        "\nAnswer:\n",
        "\nAnswer: ",
        "\nAnswer:",
        "Answer:\n",
        "Answer: ",
        "Answer:",
        "\n답:\n",
        "\n정답:\n",
    ]

def choose_prefix_and_label_ids(
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    label_space: List[str],
    dataset: str,
    strict: bool,
) -> Tuple[str, Dict[str, int]]:
    """
    file 단위로 prefix 하나를 선택(대부분 동일 포맷이므로).
    조건: 모든 label이 prefix context에서 single-token.
    """
    prefix_cands = infer_answer_prefix_candidates(dataset)

    # rows 중 하나에서 prefix 후보가 실제로 포함되는 것도 우선 시도
    sample_text = None
    for r in rows:
        c = r.get("clean", "")
        if isinstance(c, str) and c:
            sample_text = c
            break

    ordered = []
    if sample_text:
        for p in prefix_cands:
            if p in sample_text:
                ordered.append(p)
    for p in prefix_cands:
        if p not in ordered:
            ordered.append(p)

    for prefix in ordered:
        try:
            label_ids = {
                lab: _label_token_id_from_prefix(tokenizer, prefix, lab, strict=True)
                for lab in label_space
            }
            return prefix, label_ids
        except Exception:
            continue

    # 마지막 fallback: strict=False면 encode(label) single token만 체크
    if not strict:
        label_ids = {}
        ok = True
        for lab in label_space:
            ids = tokenizer.encode(lab, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
            label_ids[lab] = ids[0]
        if ok:
            return ordered[0], label_ids

    raise RuntimeError(
        f"Could not find a usable answer prefix for label_space={label_space}. "
        f"Tried={ordered}"
    )


def compute_label_token_ids_from_prompt(
    tokenizer,
    prompt: str,
    label_space: List[str],
    strict: bool,
) -> Dict[str, int]:
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
            msg = f"Label {lab!r} not single-token under full prompt."
            if strict:
                raise ValueError(msg)
            print("[WARN]", msg, "Falling back to first token of encode(label).")
            ids = tokenizer.encode(lab, add_special_tokens=False)
            if len(ids) == 0:
                raise ValueError(f"Could not tokenize label {lab!r}.")
            out[lab] = ids[0]
    return out

# -------------------------
# Dataset: (clean, corrupted, labels_tensor)
# -------------------------
class PairsDatasetRuntimeCorrupt(Dataset):
    """
    Returns (clean_text, corrupted_text, labels_tensor[K])
    where labels[:,0]=correct, labels[:,1:]=incorrect
    """
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, dataset: str, require_equal_length: bool = True,
        max_items: Optional[int] = None, strict_label_token: bool = True, debug_first: bool = False,):
        self.items: List[Tuple[str, CorruptItem, torch.Tensor]] = []
        self.dropped_len = 0
        self.dropped_label = 0
        self.mismatch_goldid = 0  # 저장된 r["label"] vs gold_label로 복원한 정답 토큰 불일치 카운트
        self.mismatch_option_ids = 0  # pair 파일에 저장된 option ids vs 현재 tokenizer 기반 option ids 불일치 카운트
        self.text_corruption_pairs = 0  # corrupted text itself differs from clean while runtime corruption config is also present
        self.first_meta: Optional[Dict[str, Any]] = None
        self.first_label_ids: Optional[List[int]] = None
        self.dataset = dataset
        self.label_space = infer_label_space(rows, dataset)

        have_row_level_ids = any(
            isinstance(r.get("option_token_ids"), list) or isinstance(r.get("label_token_ids"), dict)
            for r in rows
        )
        if have_row_level_ids:
            self.answer_prefix = None
            self.label_ids = None
        else:
            prefix, inferred_label_ids = choose_prefix_and_label_ids(
                tokenizer=tokenizer,
                rows=rows,
                label_space=self.label_space,
                dataset=dataset,
                strict=strict_label_token,
            )
            self.answer_prefix = prefix
            self.label_ids = inferred_label_ids  # {label_str: token_id}
        self.n_choices = len(self.label_space)

        first_meta = None
        for r in rows:
            clean = r.get("clean", None)
            corrupted = r.get("corrupted", None)
            corrupt_cfg = r.get("corrupt", None)
            # 저장 포맷: label(=gold token id) 있을 수도 있고, 없을 수도 있음
            saved_gold_id = r.get("label", None)
            gold_label = r.get("gold_label", None)
            row_label_token_ids = r.get("label_token_ids", None)  # dict or None
            row_option_ids = r.get("option_token_ids", None)      # list or None

            if not isinstance(clean, str) or not clean:
                continue
            if not isinstance(corrupted, str) or not corrupted:
                corrupted = clean
            if not isinstance(corrupt_cfg, dict):
                continue  # runtime corruption config가 없으면 corrupted pass 불가
            if corrupted != clean:
                self.text_corruption_pairs += 1
            if require_equal_length:
                if token_len(tokenizer, clean) != token_len(tokenizer, corrupted):
                    self.dropped_len += 1
                    continue

            # -------------------------
            # 1) option_ids 결정 (A/B/C/D 각각의 token id)
            # -------------------------
            option_ids: Optional[List[int]] = None
            saved_option_ids: Optional[List[int]] = None
            option_source = "unknown"
            # (a) row에 option_token_ids가 이미 있으면 저장만 해둠
            if isinstance(row_option_ids, list) and len(row_option_ids) == self.n_choices:
                try:
                    saved_option_ids = [int(x) for x in row_option_ids]
                except Exception:
                    saved_option_ids = None

            # (b) row에 label_token_ids dict가 있으면 저장만 해둠
            if saved_option_ids is None and isinstance(row_label_token_ids, dict):
                try:
                    saved_option_ids = [int(row_label_token_ids[lab]) for lab in self.label_space]
                except Exception:
                    saved_option_ids = None

            # (c) 실제 평가에는 row-level ids를 우선 사용
            if saved_option_ids is not None:
                option_ids = list(saved_option_ids)
                option_source = "row_level_ids"

            # (d) row-level ids가 없으면 full prompt 기준으로 직접 재계산
            prompt_label_ids = None
            if option_ids is None:
                try:
                    prompt_label_ids = compute_label_token_ids_from_prompt(
                        tokenizer,
                        clean,
                        self.label_space,
                        strict=strict_label_token,
                    )
                    option_ids = [int(prompt_label_ids[lab]) for lab in self.label_space]
                    option_source = "row_prompt_recomputed"
                except Exception:
                    option_ids = None

            # (e) 마지막 fallback: file-level prefix 기반 추론
            if option_ids is None and self.label_ids is not None:
                try:
                    option_ids = [int(self.label_ids[lab]) for lab in self.label_space]
                    option_source = "file_prefix_fallback"
                except Exception:
                    option_ids = None

            if option_ids is None or len(option_ids) != self.n_choices:
                self.dropped_label += 1
                continue
            if saved_option_ids is not None and saved_option_ids != option_ids:
                self.mismatch_option_ids += 1

            # -------------------------
            # 2) 정답 토큰 id 결정 (안전한 규칙)
            #    - gold_label이 있으면 gold_label을 "정답의 단일 소스"로 사용
            #    - saved_gold_id(r["label"])는 검증용으로만 사용 (불일치여도 drop 안 함)
            # -------------------------
            correct_id: Optional[int] = None
            if isinstance(gold_label, str) and gold_label in self.label_space:
                try:
                    correct_id = int(option_ids[self.label_space.index(gold_label)])
                except Exception:
                    correct_id = None
            else:
                # gold_label이 없으면 saved_gold_id를 fallback으로 사용
                if saved_gold_id is not None:
                    try:
                        correct_id = int(saved_gold_id)
                    except Exception:
                        correct_id = None

            if correct_id is None:
                self.dropped_label += 1
                continue

            # saved_gold_id와 gold_label 기반 correct_id가 다르면 mismatch로 카운트만 하고 진행
            if saved_gold_id is not None:
                try:
                    if int(saved_gold_id) != int(correct_id):
                        self.mismatch_goldid += 1
                except Exception:
                    pass
            # correct_id가 option_ids 안에 없으면 (드문 케이스) -> drop (오답 집합 구성 불가)
            if correct_id not in option_ids:
                self.dropped_label += 1
                continue
            incorrect_ids = [tid for tid in option_ids if tid != correct_id]
            if len(incorrect_ids) != (self.n_choices - 1):
                self.dropped_label += 1
                continue

            labels = torch.tensor([correct_id] + incorrect_ids, dtype=torch.long)
            # corrupted 텍스트는 CorruptItem로 감싸서 corrupt config를 유지
            corrupted_ct = CorruptItem(text=corrupted, corrupt=corrupt_cfg)
            self.items.append((clean, corrupted_ct, labels))

            if first_meta is None:
                first_meta = {
                    "clean": clean[:250],
                    "corrupted": corrupted[:250],
                    "saved_gold_id": int(saved_gold_id) if saved_gold_id is not None else None,
                    "gold_label": gold_label,
                    "correct_id_used": int(correct_id),
                    "label_space": self.label_space,
                    "option_ids_saved": saved_option_ids,
                    "option_ids_used": option_ids,
                    "label_ids_file_inferred": self.label_ids,
                    "label_ids_prompt_recomputed": prompt_label_ids,
                    "answer_prefix": self.answer_prefix,
                    "option_source": option_source,
                    "corrupt_cfg": corrupt_cfg,
                }
            if max_items is not None and len(self.items) >= max_items:
                break

        if len(self.items) == 0:
            raise RuntimeError(
                "No usable pairs found. "
                f"dropped_len={self.dropped_len}, dropped_label={self.dropped_label}, mismatch_goldid={self.mismatch_goldid}"
            )

        if first_meta is not None:
            self.first_meta = first_meta
            self.first_label_ids = list(first_meta.get("option_ids_used", []))

        if debug_first and first_meta is not None:
            print("[DEBUG] First example meta:")
            for k, v in first_meta.items():
                print(f"  - {k}: {v}")
            if self.mismatch_goldid:
                print(f"[WARN] mismatch_goldid={self.mismatch_goldid} (saved label token id != gold_label-derived token id). Proceeding using gold_label when available.")
            if self.mismatch_option_ids:
                print(f"[WARN] mismatch_option_ids={self.mismatch_option_ids} (saved option ids in pair file != current tokenizer-derived option ids). Proceeding using current tokenizer ids.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

def collate_pairs(batch: Iterable[Tuple[str, CorruptItem, torch.Tensor]]):
    clean, corrupted, labels = zip(*batch)
    labels_t = torch.stack(labels, dim=0)
    return list(clean), list(corrupted), labels_t

def collate_pairs_swapped(batch: Iterable[Tuple[str, CorruptItem, torch.Tensor]]):
    clean, corrupted, labels = zip(*batch)
    labels_t = torch.stack(labels, dim=0)
    return list(corrupted), list(clean), labels_t

# -------------------------
# Runtime corruption hooks for EAP-IG
# -------------------------
_CURRENT_CORRUPT_HOOK: Optional[Tuple[str, Any]] = None
_ORIG_TOKENIZE_PLUS = eap_utils.tokenize_plus

def _build_corrupt_hook_from_inputs(inputs: Sequence[Any], model: HookedTransformer) -> Optional[Tuple[str, Any]]:
    if not inputs:
        return None
    cfgs: List[Dict[str, Any]] = []
    any_cfg = False
    for x in inputs:
        cfg = getattr(x, "corrupt", None)
        if cfg is None:
            if any_cfg:
                raise ValueError("Mixed corrupted/clean items in a batch. Use a clean-only batch or fix the dataset.")
            continue
        any_cfg = True
        if not isinstance(cfg, dict):
            raise ValueError("corrupt config must be a dict.")
        cfgs.append(cfg)
    if not any_cfg:
        return None
    if len(cfgs) != len(inputs):
        raise ValueError("Missing corrupt config for some items in the batch.")

    def _norm_type(v: Any) -> str:
        return "gaussian" if v is None else str(v)

    def _norm_sigma(v: Any, default: float) -> float:
        if v is None:
            return float(default)
        return float(v)

    def _norm_hook(v: Any) -> str:
        return "hook_embed" if v is None else str(v)

    ctype = _norm_type(cfgs[0].get("type", "gaussian"))
    sigma = _norm_sigma(cfgs[0].get("sigma", 0.1), 0.1)
    hook_point = _norm_hook(cfgs[0].get("hook_point", "hook_embed"))
    for c in cfgs[1:]:
        if _norm_type(c.get("type", ctype)) != ctype:
            raise ValueError("Mixed corrupt_type in a batch. Use batch_size=1 or regenerate pairs.")
        if _norm_sigma(c.get("sigma", sigma), sigma) != sigma:
            raise ValueError("Mixed noise_sigma in a batch. Use batch_size=1 or regenerate pairs.")
        if _norm_hook(c.get("hook_point", hook_point)) != hook_point:
            raise ValueError("Mixed hook_point in a batch. Use batch_size=1 or regenerate pairs.")

    seeds = []
    for c in cfgs:
        if "seed" not in c:
            raise ValueError("Missing corrupt seed in batch.")
        seeds.append(int(c.get("seed", 0)))

    corruptor = BatchInputCorruptor(corrupt_type=ctype, sigma=sigma, hook_point=hook_point)
    corruptor.set_batch_seeds(seeds)
    hook_fn = make_corrupt_hook_fn(corruptor, model=model)
    return (hook_point, hook_fn)

def _tokenize_plus_with_corrupt(model: HookedTransformer, inputs: Sequence[Any], max_length: Optional[int] = None):
    global _CURRENT_CORRUPT_HOOK
    _CURRENT_CORRUPT_HOOK = None
    hook = _build_corrupt_hook_from_inputs(inputs, model)
    _CURRENT_CORRUPT_HOOK = hook
    str_inputs = [str(x) for x in inputs]
    return _ORIG_TOKENIZE_PLUS(model, str_inputs, max_length=max_length)

def _wrap_make_hooks_and_matrices(orig_fn):
    def _wrapped(*args, **kwargs):
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = orig_fn(*args, **kwargs)
        global _CURRENT_CORRUPT_HOOK
        hook = _CURRENT_CORRUPT_HOOK
        _CURRENT_CORRUPT_HOOK = None
        if hook is not None:
            fwd_hooks_corrupted = [hook] + list(fwd_hooks_corrupted)
        return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference
    return _wrapped

# Monkeypatch EAP-IG internals to inject runtime corruption on corrupted passes.
eap_utils.tokenize_plus = _tokenize_plus_with_corrupt
eap_attr_mod.tokenize_plus = _tokenize_plus_with_corrupt
eap_attr_node_mod.tokenize_plus = _tokenize_plus_with_corrupt
eap_eval_mod.tokenize_plus = _tokenize_plus_with_corrupt
eap_attr_mod.make_hooks_and_matrices = _wrap_make_hooks_and_matrices(eap_attr_mod.make_hooks_and_matrices)
eap_attr_node_mod.make_hooks_and_matrices = _wrap_make_hooks_and_matrices(eap_attr_node_mod.make_hooks_and_matrices)
eap_eval_mod.make_hooks_and_matrices = _wrap_make_hooks_and_matrices(eap_eval_mod.make_hooks_and_matrices)

def _forward_corrupted(model: HookedTransformer, tokens: torch.Tensor, attention_mask: torch.Tensor, corrupted_inputs: Sequence[Any],) -> torch.Tensor:
    hook = _build_corrupt_hook_from_inputs(corrupted_inputs, model)
    if hook is None:
        return model(tokens, attention_mask=attention_mask)
    with model.hooks(fwd_hooks=[hook]):
        return model(tokens, attention_mask=attention_mask)

def evaluate_baseline_runtime_corrupt(model: HookedTransformer, dataloader: DataLoader, metrics, run_corrupted: bool = False, quiet: bool = False,):
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, clean_attention_mask, clean_input_lengths, _ = _ORIG_TOKENIZE_PLUS(model, list(clean))
        corrupted_tokens, corrupted_attention_mask, corrupted_input_lengths, _ = _ORIG_TOKENIZE_PLUS(model, [str(x) for x in corrupted])
        with torch.inference_mode():
            corrupted_logits = _forward_corrupted(model, corrupted_tokens, corrupted_attention_mask, corrupted)
            logits = model(clean_tokens, attention_mask=clean_attention_mask)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, corrupted_input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, clean_input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results

# -------------------------
# Metric: vectorized MCQA logit diff
# signature matches eap's expected: (logits, clean_logits, input_length, labels, ...)
# -------------------------
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length.to(logits.device) - 1]  # [B, V]

def logit_diff_mc_tensor(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,   # [B, K], labels[:,0]=correct, labels[:,1:]=incorrect
    mean: bool = True,
    loss: bool = False,
    incorrect_agg: str = "max",  # max/mean/min
) -> torch.Tensor:
    del clean_logits
    last_logits = get_logit_positions(logits, input_length)  # [B, V]

    correct_ids = labels[:, 0].to(last_logits.device)        # [B]
    incorrect_ids = labels[:, 1:].to(last_logits.device)     # [B, K-1]

    correct_logit = last_logits.gather(1, correct_ids.unsqueeze(1)).squeeze(1)  # [B]
    incorrect_logits = last_logits.gather(1, incorrect_ids)                      # [B, K-1]

    if incorrect_agg == "mean":
        incorrect_logit = incorrect_logits.mean(dim=1)
    elif incorrect_agg == "min":
        incorrect_logit = incorrect_logits.min(dim=1).values
    else:
        incorrect_logit = incorrect_logits.max(dim=1).values

    results = correct_logit - incorrect_logit  # [B]

    if loss:
        results = -results
    return results.mean() if mean else results

# -------------------------
# Model loader (yours + safe flags)
# -------------------------
def load_model(model_name: str, device: str, dtype: torch.dtype, gqa_mode: str = "ungrouped") -> HookedTransformer:
    return load_model_for_circuit_discovery(
        model_name,
        device,
        dtype,
        gqa_mode=gqa_mode,  # type: ignore[arg-type]
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
    )

def _infer_tag_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    if base.startswith("pairs_"):
        return base[len("pairs_"):]
    return base

def dump_top_edges_jsonl(graph: Graph, out_path: str, topn: int = 2000, absolute: bool = True):
    """ 사람이 보기 좋은 edge score 덤프. graph.edges 구조가 버전마다 조금 달라질 수 있어서 방어적으로 추출. """
    edges_obj = getattr(graph, "edges", None)
    if edges_obj is None:
        print("[WARN] graph has no .edges attribute; skip dumping edge scores.")
        return
    if isinstance(edges_obj, dict):
        edges = edges_obj.values()
    else:
        edges = edges_obj

    uses_native_gqa = bool(graph.get_gqa_info().get("uses_native_gqa", False))
    grouped_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    ordered_rows: List[Dict[str, Any]] = []
    for e in edges:
        # score field candidates
        score = None
        for k in ["score", "ie", "indirect_effect", "value"]:
            if hasattr(e, k):
                score = getattr(e, k)
                break
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        name = getattr(e, "name", None) or str(e)
        src = getattr(getattr(e, "parent", None), "name", None)
        dst = getattr(getattr(e, "child", None), "name", None)
        export_info = edge_export_info(e)
        row = {
            "edge": export_info["export_name"],
            "src": src,
            "dst": dst,
            "score": score_f,
            "qkv": export_info["qkv"],
            "query_head": export_info["query_head"],
            "kv_head": export_info["kv_head"],
            "dst_slot_name": export_info["dst_slot_name"],
            "matrix_index": export_info["matrix_index"],
            "shares_backward_slot": export_info["shares_backward_slot"],
            "aliases": [name],
        }
        matrix_index = tuple(export_info["matrix_index"])
        if uses_native_gqa and export_info["shares_backward_slot"]:
            existing = grouped_rows.get(matrix_index)
            if existing is None:
                grouped_rows[matrix_index] = row
                ordered_rows.append(row)
            else:
                existing["aliases"].append(name)
                existing["alias_count"] = len(existing["aliases"])
                existing["dst_aliases"] = sorted(set(existing.get("dst_aliases", [existing["dst"]]) + [dst]))
        else:
            row["alias_count"] = 1
            row["dst_aliases"] = [dst]
            ordered_rows.append(row)

    rows = []
    for row in ordered_rows:
        if "alias_count" not in row:
            row["alias_count"] = len(row["aliases"])
        if "dst_aliases" not in row:
            row["dst_aliases"] = [row["dst"]]
        row["deduped_by_matrix_index"] = bool(uses_native_gqa and row["shares_backward_slot"])
        rows.append(row)

    def key_fn(r):
        s = r["score"]
        if s is None:
            return -1e30
        return abs(s) if absolute else s

    rows.sort(key=key_fn, reverse=True)
    rows = rows[:topn]

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run_one(model: HookedTransformer, pairs_path: str, out_dir: str, ig_steps: int, topn: int, level: str,
    absolute_scores: bool, batch_size: int, num_workers: int, max_items: Optional[int], strict_label_token: bool,
    require_equal_length: bool, incorrect_agg: str, save_json: bool, eval_circuit: bool,
    dump_edges: bool, debug: bool, debug_chars: int, pairs_meta: Optional[Dict[str, str]],) -> None:

    rows = read_jsonl(pairs_path)
    dataset_name = (pairs_meta or {}).get("dataset") or infer_dataset_from_rows(rows, default="piqa")
    dataset = PairsDatasetRuntimeCorrupt(
        rows,
        tokenizer=model.tokenizer,
        dataset=dataset_name,
        require_equal_length=require_equal_length,
        max_items=max_items,
        strict_label_token=strict_label_token,
        debug_first=debug,
    )
    if require_equal_length and dataset.dropped_len:
        print(f"[WARN] Dropped {dataset.dropped_len} pairs with length mismatch from {pairs_path}")
    if dataset.dropped_label:
        print(f"[WARN] Dropped {dataset.dropped_label} pairs with label mismatch from {pairs_path}")
    if dataset.mismatch_option_ids:
        print(f"[WARN] Saved option ids differed from current tokenizer on {dataset.mismatch_option_ids} pairs in {pairs_path}. Using current tokenizer ids.")
    if dataset.text_corruption_pairs:
        print(
            f"[WARN] Found {dataset.text_corruption_pairs} pairs where corrupted text differs from clean in {pairs_path}. "
            "The corrupted pass will use both corrupted text and runtime activation corruption."
        )
    if len(dataset) == 0:
        raise RuntimeError(f"No usable pairs found in {pairs_path}")
    if debug:
        print(f"[DEBUG] pairs_path={pairs_path}")
        print(f"[DEBUG] items={len(dataset)} dropped_len={dataset.dropped_len} dropped_label={dataset.dropped_label} n_choices={dataset.n_choices}")
        if dataset.first_meta:
            meta = dataset.first_meta
            label_ids = dataset.first_label_ids or []
            print(f"[DEBUG] label_space={meta['label_space']} label_ids={label_ids} answer_prefix={meta['answer_prefix']}")
            clean = str(meta["clean"])
            corrupt = str(meta["corrupted"])
            clean_len = token_len(model.tokenizer, clean)
            corrupt_len = token_len(model.tokenizer, corrupt)
            print(f"[DEBUG] prompt_len clean={clean_len} corrupted={corrupt_len}")
            if debug_chars != 0:
                c = clean[:debug_chars]
                d = corrupt[:debug_chars]
                print(f"[DEBUG] prompt_clean_head:\n{c}")
                print(f"[DEBUG] prompt_corrupted_head:\n{d}")
            else:
                print(f"[DEBUG] prompt_clean:\n{clean}")
                print(f"[DEBUG] prompt_corrupted:\n{corrupt}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
        num_workers=num_workers,
        pin_memory=True,
    )

    baseline_clean = None
    # ---- sanity: baseline clean vs corrupted ----
    if eval_circuit:
        metric_eval = partial(
            logit_diff_mc_tensor,
            loss=False,
            mean=False,
            incorrect_agg=incorrect_agg,
        )
        baseline_clean = evaluate_baseline_runtime_corrupt(model, dataloader, metric_eval, run_corrupted=False).mean().item()
        baseline_corrupt = evaluate_baseline_runtime_corrupt(model, dataloader, metric_eval, run_corrupted=True).mean().item()
        print(f"[SANITY] clean_baseline={baseline_clean:.6f}  corrupted_baseline={baseline_corrupt:.6f}  diff={baseline_clean-baseline_corrupt:.6f}")
    elif debug:
        metric_eval = partial(
            logit_diff_mc_tensor,
            loss=False,
            mean=False,
            incorrect_agg=incorrect_agg,
        )
        clean_batch, corrupted_batch, labels_batch = next(iter(dataloader))
        with torch.no_grad():
            clean_tokens, clean_attention_mask, clean_input_lengths, _ = _ORIG_TOKENIZE_PLUS(model, list(clean_batch))
            corrupted_tokens, corrupted_attention_mask, corrupted_input_lengths, _ = _ORIG_TOKENIZE_PLUS(model, [str(x) for x in corrupted_batch])
            clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)
            corrupted_logits = _forward_corrupted(model, corrupted_tokens, corrupted_attention_mask, corrupted_batch)
        clean_metric = metric_eval(clean_logits, corrupted_logits, clean_input_lengths, labels_batch).mean().item()
        corrupt_metric = metric_eval(corrupted_logits, clean_logits, corrupted_input_lengths, labels_batch).mean().item()
        print(f"[DEBUG] first_batch_metric clean={clean_metric:.6f} corrupted={corrupt_metric:.6f}")

    # ---- attribution ----
    model.reset_hooks(including_permanent=True)
    metric_attr = partial(logit_diff_mc_tensor,loss=True, mean=True, incorrect_agg=incorrect_agg)
    if level == "edge":
        graph = Graph.from_model(model)
        attribute(model, graph, dataloader, metric_attr, method="EAP-IG-inputs", ig_steps=ig_steps)
    elif level == "node":
        graph = Graph.from_model(model, node_scores=True)
        attribute_node(
            model,
            graph,
            dataloader,
            metric_attr,
            method="EAP-IG-inputs",
            ig_steps=ig_steps,
            neuron=False,
        )
    elif level == "neuron":
        graph = Graph.from_model(model, neuron_level=True, node_scores=True)
        attribute_node(
            model,
            graph,
            dataloader,
            metric_attr,
            method="EAP-IG-inputs",
            ig_steps=ig_steps,
            neuron=True,
        )
    else:
        raise ValueError(f"Unsupported level: {level}")
    graph.apply_topn(topn, absolute=absolute_scores, level=level)

    tag = (pairs_meta or {}).get("tag") or _infer_tag_from_path(pairs_path)
    out_name = f"eapig_{tag}_m{ig_steps}_top{topn}_{level}.pt"
    out_path = os.path.join(out_dir, out_name)
    graph.to_pt(out_path)
    print(f"[OK] wrote {out_path}")

    if save_json:
        graph.to_json(out_path.replace(".pt", ".json"))

    if dump_edges:
        dump_path = out_path.replace(".pt", ".edge_scores.jsonl")
        dump_top_edges_jsonl(graph, dump_path, topn=min(5000, topn), absolute=absolute_scores)
        print(f"[OK] wrote {dump_path}")

    # Save run metadata for downstream analysis (bucket comparisons, etc.)
    meta_path = out_path.replace(".pt", ".run.json")
    pairs_meta_path = pairs_path.replace(".jsonl", ".meta.json")
    pairs_meta_json = None
    if os.path.exists(pairs_meta_path):
        try:
            with open(pairs_meta_path, "r", encoding="utf-8") as f:
                pairs_meta_json = json.load(f)
        except Exception:
            pairs_meta_json = None
    run_meta = {
        "pairs_path": pairs_path,
        "pairs_meta_path": pairs_meta_path if os.path.exists(pairs_meta_path) else None,
        "pairs_meta": pairs_meta_json,
        "dataset": dataset_name,
        "score_key": (pairs_meta or {}).get("score_key"),
        "bucket": (pairs_meta or {}).get("bucket"),
        "corrupt_type": (pairs_meta or {}).get("corrupt_type"),
        "model_name": getattr(model.cfg, "model_name", None),
        "model_circuit_summary": summarize_model_for_circuit_discovery(model),
        "gqa_mode": getattr(model.cfg, "eap_actual_gqa_mode", None),
        "dtype": str(model.cfg.dtype),
        "device": str(model.cfg.device),
        "ig_steps": ig_steps,
        "topn": topn,
        "level": level,
        "incorrect_agg": incorrect_agg,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_items": max_items,
        "require_equal_length": require_equal_length,
        "strict_label_token": strict_label_token,
        "num_items": len(dataset),
        "dropped_len": dataset.dropped_len,
        "dropped_label": dataset.dropped_label,
        "mismatch_option_ids": dataset.mismatch_option_ids,
        "mismatch_goldid": dataset.mismatch_goldid,
        "text_corruption_pairs": dataset.text_corruption_pairs,
        "answer_prefix": dataset.answer_prefix,
        "label_space": dataset.label_space,
        "label_ids": dataset.first_label_ids,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {meta_path}")

    if eval_circuit:
        metric_eval = partial(logit_diff_mc_tensor, loss=False, mean=False, incorrect_agg=incorrect_agg,)
        if baseline_clean is None:
            baseline_clean = evaluate_baseline_runtime_corrupt(model, dataloader, metric_eval, run_corrupted=False).mean().item()
        result = evaluate_graph(model, graph, dataloader, metric_eval).mean().item()
        print(f"[EVAL] baseline(clean)={baseline_clean:.6f}  circuit(clean)={result:.6f}")

    model.reset_hooks(including_permanent=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_path", type=str, default="", help="Path to a specific pairs jsonl.")
    ap.add_argument("--pairs_dir", type=str, default=str(Path(__file__).resolve().parents[1] / 'data_gens/pairs'))
    ap.add_argument("--pairs_glob", type=str, default="", help="Optional glob under pairs_dir (overrides filters, supports ** for recursion).")
    ap.add_argument("--dataset", type=str, default="piqa", help="Filter by dataset (piqa/winogrande/sciq/arc_easy). Empty=all.")
    ap.add_argument("--score_key", type=str, default="aum", help="Filter by score key (aum/forget/el2n/gradlogits_l2/grad_align/tracin). Empty=all.")
    ap.add_argument("--bucket", type=str, default="", help="Filter by bucket (high/low). Empty=all.")
    ap.add_argument("--corrupt_type", type=str, default="", help="Filter by corruption type (gaussian/zero/mean). Empty=all.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--gqa_mode", type=str, default="native", choices=["auto", "native", "ungrouped"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--ig_steps", type=int, default=5)   # m=5
    ap.add_argument("--topn", type=int, default=15000)
    ap.add_argument("--level", type=str, default="edge", choices=["edge", "node", "neuron"])
    ap.add_argument("--no_absolute_scores", dest="absolute_scores", action="store_false")
    ap.add_argument("--max_items", type=int, default=0, help="Limit number of pairs per file (0 = all).")
    ap.add_argument("--allow_multi_token_labels", dest="strict_label_token", action="store_false")
    ap.add_argument("--allow_length_mismatch", dest="require_equal_length", action="store_false")
    ap.add_argument("--incorrect_agg", type=str, default="max", choices=["max", "mean", "min"])
    ap.add_argument("--save_json", action="store_true", help="Also save graph to json.")
    ap.add_argument("--no_save_json", dest="save_json", action="store_false", help="Skip saving graph json.")
    ap.add_argument("--eval", action="store_true", help="Also print clean/corrupted baseline + circuit eval on clean.")
    ap.add_argument("--no_eval", dest="eval", action="store_false", help="Skip clean/corrupted baseline evaluation.")
    ap.add_argument("--dump_edges", action="store_true", help="Dump top edge scores to jsonl for easy postprocessing.")
    ap.add_argument("--no_dump_edges", dest="dump_edges", action="store_false", help="Skip edge score dump.")
    ap.add_argument("--debug", action="store_true", help="Print first-sample debug info and first-batch metric.")
    ap.add_argument("--no_debug", dest="debug", action="store_false", help="Disable prompt debug prints.")
    ap.add_argument("--debug_chars", type=int, default=400, help="Max chars to show for prompt debug (0 = full).")
    ap.add_argument("--out_root", type=str, default=str(Path(__file__).resolve().parents[1] / 'circuit_results'), help="Base output directory.")
    ap.add_argument("--out_dir", type=str, default="", help="Override output directory; if set, all outputs go here.")
    ap.set_defaults(
        absolute_scores=True,
        strict_label_token=True,
        require_equal_length=True,
        eval=True,
        dump_edges=True,
        save_json=True,
        debug=True,
    )
    args = ap.parse_args()

    pairs_path = args.pairs_path if args.pairs_path else None
    dataset_filter = args.dataset.strip() or None
    score_key_filter = args.score_key.strip() or None
    bucket_filter = args.bucket.strip() or None
    corrupt_type_filter = args.corrupt_type.strip() or None

    # perf niceties
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(args.model_name, args.device, str_to_dtype(args.dtype), gqa_mode=args.gqa_mode)
    pairs_files = resolve_pairs_files(
        pairs_dir=args.pairs_dir,
        pairs_path=pairs_path,
        dataset=dataset_filter,
        score_key=score_key_filter,
        bucket=bucket_filter,
        corrupt_type=corrupt_type_filter,
        pairs_glob=args.pairs_glob if args.pairs_glob else None,
    )
    if not pairs_files:
        raise RuntimeError("No pairs files found. Check --pairs_dir/--pairs_glob/--filters/--pairs_path.")

    pairs_iter = pairs_files
    if len(pairs_files) > 1:
        pairs_iter = tqdm(pairs_files, desc="pairs files")
    for meta, ppath in pairs_iter:
        meta_use = meta or _parse_pairs_filename(ppath)
        out_dir = args.out_dir if args.out_dir else _build_out_dir(args.out_root, meta_use)
        ensure_dir(out_dir)
        run_one(
            model=model,
            pairs_path=ppath,
            out_dir=out_dir,
            ig_steps=args.ig_steps,
            topn=args.topn,
            level=args.level,
            absolute_scores=args.absolute_scores,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=(args.max_items if args.max_items > 0 else None),
            strict_label_token=args.strict_label_token,
            require_equal_length=args.require_equal_length,
            incorrect_agg=args.incorrect_agg,
            save_json=args.save_json,
            eval_circuit=args.eval,
            dump_edges=args.dump_edges,
            debug=args.debug,
            debug_chars=args.debug_chars,
            pairs_meta=meta_use,
        )


if __name__ == "__main__":
    main()
