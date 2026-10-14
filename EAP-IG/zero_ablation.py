import argparse
import gc
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import einsum
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_CUR_DIR = Path(__file__).resolve().parent
_ROOT = _CUR_DIR.parent
_EAPIG_ROOT = _ROOT / "EAP-IG"
_EAP_SRC = _EAPIG_ROOT / "src"
for _p in [str(_ROOT), str(_EAPIG_ROOT), str(_EAP_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scen1_eapig import load_model, str_to_dtype  # type: ignore  # noqa: E402
from eap.graph import AttentionNode, Graph  # type: ignore  # noqa: E402
from eap.utils import make_hooks_and_matrices  # type: ignore  # noqa: E402

STATIC_SCORE_KEYS = {"grad_align", "tracin"}
DYNAMIC_SCORE_KEYS = {"aum", "forget", "el2n", "gradlogits_l2"}
ALL_SCORE_KEYS = sorted(STATIC_SCORE_KEYS | DYNAMIC_SCORE_KEYS)
RUN_RE = re.compile(r"_m(?P<m>\d+)_top(?P<top>\d+)_")
BUCKET_ROOT_HINTS = ["train_dynamic", "data_attr"]
CIRCUIT_ROOT_HINTS = ["train_dynamic", "data_attr"]

@dataclass(frozen=True)
class EvalStats:
    n_eval: int
    n_high: int
    n_low: int

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _target_tag(circuits: Sequence[str]) -> str:
    tags = [str(x).strip() for x in circuits if str(x).strip()]
    if not tags:
        return "none"
    return "_".join(tags)

def save_report_with_target_and_summary(report: Dict[str, Any], out_root: Path, dataset: str, score_key: str, circuits: Sequence[str],) -> Tuple[Path, Path, Path]:
    tag = _target_tag(circuits)
    # Per-target latest path (non-overlapping across targets).
    latest_report = out_root / f"{dataset}_{score_key}_zero_ablation_report_latest_{tag}.json"
    write_json(report, latest_report)
    # Target-specific immutable report path for analysis.
    target_report = out_root / f"{dataset}_{score_key}_zero_ablation_report_{tag}.json"
    write_json(report, target_report)
    # Merge into a cumulative summary json.
    summary_path = out_root / f"{dataset}_{score_key}_zero_ablation_summary.json"
    summary_obj: Dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = read_json(summary_path)
            if isinstance(loaded, dict):
                summary_obj = dict(loaded)
        except Exception:
            summary_obj = {}

    summary_obj["dataset"] = dataset
    summary_obj["score_key"] = score_key
    summary_obj["group"] = report.get("group", summary_obj.get("group", ""))
    summary_obj["paths"] = report.get("paths", summary_obj.get("paths", {}))
    summary_obj["load_stats"] = report.get("load_stats", summary_obj.get("load_stats", {}))
    summary_obj["eval_stats"] = report.get("eval_stats", summary_obj.get("eval_stats", {}))

    reports_by_target = summary_obj.get("reports_by_target", {})
    if not isinstance(reports_by_target, dict):
        reports_by_target = {}
    reports_by_target[tag] = str(target_report)
    summary_obj["reports_by_target"] = reports_by_target

    circuits_merged = summary_obj.get("circuits_merged", {})
    if not isinstance(circuits_merged, dict):
        circuits_merged = {}
    for cb, cb_report in report.get("circuits", {}).items():
        circuits_merged[str(cb)] = cb_report
    summary_obj["circuits_merged"] = circuits_merged
    summary_obj["available_circuits"] = sorted(circuits_merged.keys())

    configs_by_target = summary_obj.get("configs_by_target", {})
    if not isinstance(configs_by_target, dict):
        configs_by_target = {}
    configs_by_target[tag] = report.get("config", {})
    summary_obj["configs_by_target"] = configs_by_target

    history = summary_obj.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append({"target_tag": tag, "report_path": str(target_report)})
    summary_obj["history"] = history

    write_json(summary_obj, summary_path)
    return latest_report, target_report, summary_path

def parse_csv_list(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

def normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    if "/" in n:
        return n
    if n.startswith("pythia-"):
        return f"EleutherAI/{n}"
    return n

def infer_score_group(score_key: str) -> str:
    if score_key in STATIC_SCORE_KEYS:
        return "data_attr"
    if score_key in DYNAMIC_SCORE_KEYS:
        return "train_dynamic"
    raise ValueError(f"Unsupported score_key={score_key}. choices={ALL_SCORE_KEYS}")

def _expand_roots(base_root: Path, hints: Sequence[str]) -> List[Path]:
    roots: List[Path] = []

    def _add(p: Path) -> None:
        if p.is_dir() and p not in roots:
            roots.append(p)

    if base_root.name in set(hints):
        _add(base_root)
        return roots if roots else [base_root]

    has_hint_child = False
    for n in hints:
        p = base_root / n
        if p.is_dir():
            has_hint_child = True
            _add(p)
    if not has_hint_child:
        _add(base_root)
    return roots if roots else [base_root]

def _iter_matching_files(roots: Sequence[Path], pattern: str) -> List[Path]:
    out: List[Path] = []
    seen: Set[Path] = set()
    for r in roots:
        if not r.is_dir():
            continue
        for p in sorted(r.rglob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out

def _score_pt_candidate(path: Path) -> Tuple[int, int, float]:
    base = path.name
    m = RUN_RE.search(base)
    if m:
        ig_steps = int(m.group("m"))
        topn = int(m.group("top"))
    else:
        ig_steps = -1
        topn = -1
    return (topn, ig_steps, path.stat().st_mtime)

def find_latest_circuit_pt(circuits_root: Path, dataset: str, score_key: str, bucket: str, corrupt_type: str, group: str,) -> Optional[Path]:
    roots = _expand_roots(circuits_root, CIRCUIT_ROOT_HINTS)
    candidates: List[Path] = []
    for root in roots:
        if root.name in {"train_dynamic", "data_attr"}:
            search = [root / dataset / score_key / bucket / corrupt_type]
        else:
            search = [
                root / group / dataset / score_key / bucket / corrupt_type,
                root / dataset / score_key / bucket / corrupt_type,
            ]
        for d in search:
            if not d.is_dir():
                continue
            pts = list(d.glob("eapig_*.pt")) or list(d.glob("*.pt"))
            candidates.extend(pts)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=_score_pt_candidate, reverse=True)
    return candidates[0]

def find_bucket_json(buckets_root: Path, dataset: str, score_key: str, bucket: str, group: str,) -> Optional[Path]:
    rel = Path(score_key) / dataset / f"{score_key}_{dataset}_{bucket}.json"
    roots = _expand_roots(buckets_root, BUCKET_ROOT_HINTS)
    # Prefer the group-matched subtree first (e.g., data_attr for static keys).
    roots = sorted(roots, key=lambda p: 0 if p.name == group else 1)
    for root in roots:
        if root.name in {"train_dynamic", "data_attr"}:
            cands = [root / rel]
        else:
            cands = [root / group / rel, root / rel]
        for c in cands:
            if c.is_file():
                return c
    found = _iter_matching_files(roots, f"{score_key}_{dataset}_{bucket}.json")
    return found[0] if found else None

def find_all_scores_json(buckets_root: Path, dataset: str, score_key: str, group: str,) -> Optional[Path]:
    if group == "data_attr":
        names = [f"all_{score_key}_{dataset}.json", f"all_scores_{dataset}.json"]
    else:
        names = [f"all_scores_{dataset}.json", f"all_{score_key}_{dataset}.json"]
    roots = _expand_roots(buckets_root, BUCKET_ROOT_HINTS)
    # Prefer the group-matched subtree first (e.g., avoid picking train_dynamic/all_scores for static keys).
    roots = sorted(roots, key=lambda p: 0 if p.name == group else 1)
    for root in roots:
        if root.name in {"train_dynamic", "data_attr"}:
            base = root
            cands = [base / "all" / dataset / n for n in names]
        else:
            cands = [root / group / "all" / dataset / n for n in names] + [root / "all" / dataset / n for n in names]
        for c in cands:
            if c.is_file():
                return c
    for n in names:
        found = _iter_matching_files(roots, n)
        if found:
            return found[0]
    return None

def load_bucket_ids(bucket_json: Path) -> Set[str]:
    obj = read_json(bucket_json)
    rows = obj.get("items", obj if isinstance(obj, list) else [])
    out: Set[str] = set()
    if not isinstance(rows, list):
        return out
    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if hf_id:
            out.add(hf_id)
    return out

def extract_score_from_row(row: Dict[str, Any], score_key: str) -> Optional[float]:
    score_val: Any = None
    scores = row.get("scores")
    if isinstance(scores, dict):
        score_val = scores.get(score_key)
    if score_val is None:
        score_val = row.get(score_key)
    if score_val is None:
        return None
    try:
        return float(score_val)
    except Exception:
        return None

def infer_label_token_ids_from_prompt(tokenizer: Any, prompt: str, label_space: Sequence[str], strict: bool = False,) -> Dict[str, int]:
    out: Dict[str, int] = {}
    base = tokenizer.encode(prompt, add_special_tokens=False)
    for lab in label_space:
        found: Optional[int] = None
        for cand in [lab, " " + lab, "\n" + lab]:
            ids = tokenizer.encode(prompt + cand, add_special_tokens=False)
            if len(ids) == len(base) + 1 and ids[: len(base)] == base:
                found = int(ids[-1])
                break
        if found is None:
            if strict:
                raise ValueError(f"Cannot infer single label token for {lab!r}")
            ids = tokenizer.encode(str(lab), add_special_tokens=False)
            if not ids:
                continue
            found = int(ids[0])
        out[str(lab)] = found
    return out

def resolve_gold_token_id(row: Dict[str, Any], tokenizer: Optional[Any] = None) -> Optional[int]:
    v = row.get("gold_token_id", row.get("label", None))
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    label_space = row.get("label_space")
    gold_label = row.get("gold_label")
    label_token_ids = row.get("label_token_ids")
    if isinstance(label_token_ids, dict) and gold_label is not None:
        gg = str(gold_label)
        if gg in label_token_ids:
            try:
                return int(label_token_ids[gg])
            except Exception:
                pass
    if tokenizer is None:
        return None
    if not isinstance(label_space, list) or gold_label is None:
        return None
    prompt = str(row.get("prompt", ""))
    if not prompt:
        return None
    inferred = infer_label_token_ids_from_prompt(tokenizer, prompt, [str(x) for x in label_space], strict=False)
    gg = str(gold_label)
    if gg not in inferred:
        return None
    return int(inferred[gg])

def load_all_score_rows(all_scores_json: Path, score_key: str, tokenizer: Optional[Any] = None,) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    obj = read_json(all_scores_json)
    rows = obj.get("items", obj if isinstance(obj, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected all-scores format: {all_scores_json}")
    out: Dict[str, Dict[str, Any]] = {}
    missing_hfid = 0
    missing_score = 0
    bad_score = 0
    missing_prompt = 0
    missing_gold_id = 0
    duplicates = 0
    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if not hf_id:
            missing_hfid += 1
            continue
        score = extract_score_from_row(r, score_key)
        if score is None:
            missing_score += 1
            continue
        prompt = str(r.get("prompt", "")).strip()
        if not prompt:
            missing_prompt += 1
            continue
        gold_token_id = resolve_gold_token_id(r, tokenizer=tokenizer)
        if gold_token_id is None:
            missing_gold_id += 1
            continue
        row = {
            "hf_id": hf_id,
            "prompt": prompt,
            "full_score": float(score),
            "gold_token_id": int(gold_token_id),
            "label_space": [str(x) for x in r.get("label_space", [])] if isinstance(r.get("label_space"), list) else [],
            "gold_label": str(r.get("gold_label", "")),
            "label_token_ids": r.get("label_token_ids", {}),
            "task": str(r.get("task", "")),
        }
        if hf_id in out:
            duplicates += 1
        out[hf_id] = row
    stats = {
        "num_rows": len(rows),
        "num_kept": len(out),
        "missing_hf_id": missing_hfid,
        "missing_score": missing_score,
        "bad_score": bad_score,
        "missing_prompt": missing_prompt,
        "missing_gold_token_id": missing_gold_id,
        "duplicates": duplicates,
    }
    return out, stats

def build_eval_rows(all_rows_by_id: Dict[str, Dict[str, Any]], high_ids: Set[str], low_ids: Set[str], max_samples_per_bucket: int, seed: int,) -> Tuple[List[Dict[str, Any]], EvalStats]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []

    def _take(ids: Set[str], bucket: str) -> List[Dict[str, Any]]:
        rows = []
        for hf_id in sorted(ids):
            r = all_rows_by_id.get(hf_id)
            if r is None:
                continue
            r2 = dict(r)
            r2["bucket"] = bucket
            rows.append(r2)
        if max_samples_per_bucket > 0 and len(rows) > max_samples_per_bucket:
            rows = rng.sample(rows, max_samples_per_bucket)
            rows.sort(key=lambda x: str(x["hf_id"]))
        return rows

    high_rows = _take(high_ids, "high")
    low_rows = _take(low_ids, "low")
    out.extend(high_rows)
    out.extend(low_rows)
    out.sort(key=lambda x: (str(x["bucket"]), str(x["hf_id"])))
    return out, EvalStats(n_eval=len(out), n_high=len(high_rows), n_low=len(low_rows))

class PromptScoreDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int = 512):
        self.rows: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        for r in rows:
            prompt = str(r.get("prompt", ""))
            if not prompt:
                continue
            gold = r.get("gold_token_id", None)
            if gold is None:
                continue
            try:
                gold_id = int(gold)
            except Exception:
                continue
            ids = tokenizer.encode(prompt, add_special_tokens=False) + [gold_id]
            if self.max_length > 0 and len(ids) > self.max_length:
                ids = ids[-self.max_length :]
            if len(ids) < 2:
                continue
            pred_pos = len(ids) - 2
            rr = dict(r)
            rr["input_ids"] = ids
            rr["pred_pos"] = int(pred_pos)
            rr["gold_token_id"] = int(gold_id)
            self.rows.append(rr)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "hf_id": str(r["hf_id"]),
            "bucket": str(r.get("bucket", "")),
            "full_score": float(r.get("full_score", np.nan)),
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "pred_pos": int(r["pred_pos"]),
            "gold_token_id": int(r["gold_token_id"]),
        }

def collate_prompt_score(batch: Iterable[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    items = list(batch)
    bsz = len(items)
    max_len = max(int(x["input_ids"].shape[0]) for x in items)
    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    pred_pos = torch.empty((bsz,), dtype=torch.long)
    gold_ids = torch.empty((bsz,), dtype=torch.long)
    hf_ids: List[str] = []
    buckets: List[str] = []
    full_scores: List[float] = []
    for i, x in enumerate(items):
        ids = x["input_ids"]
        ll = int(ids.shape[0])
        input_ids[i, :ll] = ids
        attention_mask[i, :ll] = 1
        pred_pos[i] = int(x["pred_pos"])
        gold_ids[i] = int(x["gold_token_id"])
        hf_ids.append(str(x["hf_id"]))
        buckets.append(str(x.get("bucket", "")))
        full_scores.append(float(x.get("full_score", np.nan)))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pred_pos": pred_pos,
        "gold_ids": gold_ids,
        "hf_ids": hf_ids,
        "buckets": buckets,
        "full_scores": full_scores,
    }

def _attach_bucket_to_rows(rows: Sequence[Dict[str, Any]], high_ids: Set[str], low_ids: Set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        hf_id = str(rr.get("hf_id", "")).strip()
        if hf_id in high_ids:
            rr["bucket"] = "high"
        elif hf_id in low_ids:
            rr["bucket"] = "low"
        else:
            rr["bucket"] = str(rr.get("bucket", "other") or "other")
        out.append(rr)
    return out

def build_opposite_bucket_patch_lookup(rows: Sequence[Dict[str, Any]], seed: int,) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    rng = random.Random(int(seed))
    ids_by_hf: Dict[str, List[int]] = {}
    high: List[str] = []
    low: List[str] = []
    other: List[str] = []
    duplicate_hf = 0
    bad_rows = 0

    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        raw_ids = r.get("input_ids")
        if (not hf_id) or (not isinstance(raw_ids, list)) or (len(raw_ids) == 0):
            bad_rows += 1
            continue
        if hf_id in ids_by_hf:
            duplicate_hf += 1
            continue
        try:
            ids_by_hf[hf_id] = [int(x) for x in raw_ids]
        except Exception:
            bad_rows += 1
            continue

        bucket = str(r.get("bucket", "")).strip().lower()
        if bucket == "high":
            high.append(hf_id)
        elif bucket == "low":
            low.append(hf_id)
        else:
            other.append(hf_id)

    rng.shuffle(high)
    rng.shuffle(low)
    rng.shuffle(other)

    lookup: Dict[str, List[int]] = {}

    def _assign(targets: Sequence[str], sources: Sequence[str]) -> int:
        if not sources:
            return 0
        mapped = 0
        for i, hf_id in enumerate(targets):
            src_hf_id = sources[i % len(sources)]
            src_ids = ids_by_hf.get(src_hf_id)
            if not src_ids:
                continue
            lookup[hf_id] = src_ids
            mapped += 1
        return mapped

    mapped_high = _assign(high, low)
    mapped_low = _assign(low, high)
    source_pool = high + low
    rng.shuffle(source_pool)
    mapped_other = _assign(other, source_pool)

    stats = {
        "num_rows": int(len(rows)),
        "num_unique_hf_id": int(len(ids_by_hf)),
        "high_count": int(len(high)),
        "low_count": int(len(low)),
        "other_count": int(len(other)),
        "mapped_high_to_low": int(mapped_high),
        "mapped_low_to_high": int(mapped_low),
        "mapped_other": int(mapped_other),
        "num_mapped": int(len(lookup)),
        "num_unmapped": int(max(0, len(ids_by_hf) - len(lookup))),
        "duplicate_hf_id": int(duplicate_hf),
        "bad_rows": int(bad_rows),
    }
    return lookup, stats

def _build_patching_source_batch(
    hf_ids: Sequence[str],
    target_tokens: torch.Tensor,
    target_attention_mask: torch.Tensor,
    patch_source_by_hf_id: Dict[str, List[int]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    source_tokens = target_tokens.clone()
    source_attention_mask = target_attention_mask.clone()
    tlen = int(target_tokens.shape[1])
    dtype_tok = target_tokens.dtype
    dtype_attn = target_attention_mask.dtype
    device = target_tokens.device

    mapped = 0
    missing = 0
    for i, hf_id in enumerate(hf_ids):
        src = patch_source_by_hf_id.get(str(hf_id), None)
        if not src:
            missing += 1
            continue

        src_ids = src[-tlen:] if len(src) > tlen else src
        src_len = int(len(src_ids))
        tok = torch.full((tlen,), int(pad_id), dtype=dtype_tok, device=device)
        attn = torch.zeros((tlen,), dtype=dtype_attn, device=device)
        if src_len > 0:
            tok[:src_len] = torch.tensor(src_ids, dtype=dtype_tok, device=device)
            attn[:src_len] = 1

        source_tokens[i] = tok
        source_attention_mask[i] = attn
        mapped += 1

    return source_tokens, source_attention_mask, {"mapped": int(mapped), "missing": int(missing)}

def compute_mean_activation_template(model: Any, graph: Graph, dataloader: DataLoader, device: str, quiet: bool, max_samples: int = 0,) -> Tuple[torch.Tensor, Dict[str, int]]:
    model.eval()
    running = torch.zeros((graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=torch.float32)
    n_samples = 0
    n_tokens = 0
    n_batches = 0

    iterator = dataloader if quiet else tqdm(dataloader, desc="mean-activation", leave=False)
    stop = False
    for batch in iterator:
        tokens_batch = batch["input_ids"].to(device)
        attn_batch = batch["attention_mask"].to(device)
        n_batches += 1
        bsz = int(tokens_batch.shape[0])
        for i in range(bsz):
            if int(max_samples) > 0 and n_samples >= int(max_samples):
                stop = True
                break

            tokens = tokens_batch[i : i + 1]
            attn = attn_batch[i : i + 1]
            batch_size, n_pos = int(tokens.shape[0]), int(tokens.shape[1])
            (fwd_hooks_corrupted, _, _), acts = make_hooks_and_matrices(
                model,
                graph,
                batch_size=batch_size,
                n_pos=n_pos,
                scores=None,
            )
            with torch.inference_mode():
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(tokens, attention_mask=attn)

            pos_mask = attn.to(dtype=acts.dtype).unsqueeze(-1).unsqueeze(-1)
            running += (acts * pos_mask).sum(dim=(0, 1)).float()
            n_tokens += int(attn.sum().item())
            n_samples += 1
            model.reset_hooks(including_permanent=True)

        if stop:
            break

    if n_tokens <= 0:
        raise RuntimeError("Failed to compute mean activation template (no valid tokens).")

    mean_template = (running / float(n_tokens)).to(dtype=model.cfg.dtype).unsqueeze(0).unsqueeze(0)
    stats = {
        "n_samples": int(n_samples),
        "n_tokens": int(n_tokens),
        "n_batches": int(n_batches),
        "max_samples": int(max_samples),
    }
    return mean_template, stats

def get_target_params(model: Any) -> List[torch.Tensor]:
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return [model.unembed.W_U]
    if hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
        return [model.embed_out.weight]
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return [model.lm_head.weight]
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return [params[-1]]

def flatten_grads(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1).detach().float() for g in grads], dim=0)

def corr_dict(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    n = int(mask.sum())
    if n < 2:
        return {"n": n, "pearson": float("nan"), "spearman": float("nan")}
    sx = pd.Series(xs[mask])
    sy = pd.Series(ys[mask])
    return {
        "n": n,
        "pearson": float(sx.corr(sy, method="pearson")),
        "spearman": float(sx.corr(sy, method="spearman")),
    }

def save_full_vs_ablated_scatter(
    df: pd.DataFrame,
    score_key: str,
    title: str,
    out_path: Path,
    x_col: str = "q_full",
    y_col: str = "q_ablated",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=140)
    colors = {"high": "#d1495b", "low": "#2b59c3"}
    for b in ["high", "low"]:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[x_col].values,
            sub[y_col].values,
            s=20,
            alpha=0.76,
            color=colors[b],
            label=f"{b} bucket",
        )
    arr_x = df[x_col].to_numpy(dtype=np.float64)
    arr_y = df[y_col].to_numpy(dtype=np.float64)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if int(mask.sum()) > 0:
        mn = float(min(arr_x[mask].min(), arr_y[mask].min()))
        mx = float(max(arr_x[mask].max(), arr_y[mask].max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.5, color="#444444", alpha=0.8, label="y=x")
    ax.set_title(title)
    ax.set_xlabel(x_label or f"Original q(x): {score_key}")
    ax.set_ylabel(y_label or f"Ablated q^(-C)(x): {score_key}")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    plt.close(fig)

def select_edge_names_from_circuit(circuit_pt: Path, topk_edges: int, rank_abs: bool,) -> Tuple[List[str], List[float], int]:
    g = Graph.from_pt(str(circuit_pt))
    edges = list(g.edges.values())
    total_edges = len(edges)
    if total_edges == 0:
        raise RuntimeError(f"No edges found in graph: {circuit_pt}")
    if topk_edges <= 0:
        selected = [e for e in edges if bool(e.in_graph)]
        if not selected:
            selected = edges
    else:
        k = min(int(topk_edges), total_edges)
        selected = sorted(
            edges,
            key=lambda e: abs(float(e.score)) if rank_abs else float(e.score),
            reverse=True,
        )[:k]
    selected = sorted(
        selected,
        key=lambda e: abs(float(e.score)) if rank_abs else float(e.score),
        reverse=True,
    )
    names = [str(e.name) for e in selected]
    scores = [float(e.score) for e in selected]
    return names, scores, total_edges

def build_ablation_graph(model: Any, circuit_pt: Path, topk_edges: int, rank_abs: bool,) -> Tuple[Graph, Dict[str, Any]]:
    edge_names, edge_scores, total_edges = select_edge_names_from_circuit(circuit_pt, topk_edges, rank_abs)
    runtime_graph = Graph.from_model(model)
    runtime_graph.reset(empty=False)
    missing: List[str] = []
    rows: List[int] = []
    cols: List[int] = []
    for name in edge_names:
        edge = runtime_graph.edges.get(name)
        if edge is None:
            missing.append(name)
            continue
        rows.append(int(edge.matrix_index[0]))
        cols.append(int(edge.matrix_index[1]))

    if rows:
        rr = torch.tensor(rows, dtype=torch.long)
        cc = torch.tensor(cols, dtype=torch.long)
        runtime_graph.in_graph[rr, cc] = False
    runtime_graph.prune()
    meta = {
        "circuit_pt": str(circuit_pt),
        "selected_edges_by_name": int(len(edge_names)),
        "mapped_edges_in_runtime": int(len(rows)),
        "missing_edges_in_runtime": int(len(missing)),
        "total_edges_in_saved_graph": int(total_edges),
        "remaining_edges_after_ablation": int(runtime_graph.in_graph.sum().item()),
        "top10_selected_edges": [{"edge": n, "score": s} for n, s in list(zip(edge_names, edge_scores))[:10]],
    }
    if missing:
        meta["missing_edge_names_top10"] = missing[:10]
    return runtime_graph, meta

def build_random_ablation_graph_from_circuit(model: Any, circuit_pt: Path, topk_edges: int, random_seed: int,) -> Tuple[Graph, Dict[str, Any]]:
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
        "random_edge_pool": "circuit",
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
    random_seed: int,) -> Tuple[Graph, Dict[str, Any]]:
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

def _make_zero_input_construction_hooks(model: Any, graph: Graph, activation_difference: torch.Tensor, in_graph_matrix: torch.Tensor,) -> List[Tuple[str, Callable[..., torch.Tensor]]]:
    if model.cfg.use_normalization_before_and_after:
        raise RuntimeError("zero_ablation.py currently supports non-Gemma models only.")
    if graph.neurons_in_graph is not None:
        raise RuntimeError("Neuron-level circuits are not supported in zero_ablation.py.")
    def make_input_construction_hook(in_graph_vector: torch.Tensor):
        def input_construction_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            del hook
            update = einsum(
                activation_difference[:, :, : len(in_graph_vector)],
                in_graph_vector,
                "batch pos previous hidden, previous ... -> batch pos ... hidden",
            )
            activations = activations + update
            return activations
        return input_construction_hook

    hooks: List[Tuple[str, Callable[..., torch.Tensor]]] = []
    for layer in range(model.cfg.n_layers):
        has_any_attn = any(graph.nodes[f"a{layer}.h{head}"].in_graph for head in range(model.cfg.n_heads))
        all_parents_on = all(
            parent_edge.in_graph
            for head in range(model.cfg.n_heads)
            for parent_edge in graph.nodes[f"a{layer}.h{head}"].parent_edges
        )
        if has_any_attn and (not all_parents_on):
            node = graph.nodes[f"a{layer}.h0"]
            prev_index = graph.prev_index(node)
            for i, letter in enumerate("qkv"):
                bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                hook_fn = make_input_construction_hook(in_graph_matrix[:prev_index, bwd_index])
                hooks.append((node.qkv_inputs[i], hook_fn))

        mlp_node = graph.nodes[f"m{layer}"]
        if mlp_node.in_graph and (not all(parent_edge.in_graph for parent_edge in mlp_node.parent_edges)):
            prev_index = graph.prev_index(mlp_node)
            bwd_index = graph.backward_index(mlp_node)
            hook_fn = make_input_construction_hook(in_graph_matrix[:prev_index, bwd_index])
            hooks.append((mlp_node.in_hook, hook_fn))

    logits_node = graph.nodes["logits"]
    if not all(parent_edge.in_graph for parent_edge in logits_node.parent_edges):
        prev_index = graph.prev_index(logits_node)
        bwd_index = graph.backward_index(logits_node)
        hook_fn = make_input_construction_hook(in_graph_matrix[:prev_index, bwd_index])
        hooks.append((logits_node.in_hook, hook_fn))
    return hooks

def forward_with_circuit_intervention(
    model: Any,
    ablation_graph: Graph,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    intervention: str = "zero",
    source_tokens: Optional[torch.Tensor] = None,
    source_attention_mask: Optional[torch.Tensor] = None,
    mean_activation_template: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, n_pos = int(tokens.shape[0]), int(tokens.shape[1])
    (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
        model,
        ablation_graph,
        batch_size=batch_size,
        n_pos=n_pos,
        scores=None,
    )
    iv = str(intervention).strip().lower()
    if iv not in {"zero", "mean", "patching"}:
        raise ValueError(f"Unsupported intervention: {intervention}")

    if iv == "patching":
        src_tokens = source_tokens if source_tokens is not None else tokens
        src_attn = source_attention_mask if source_attention_mask is not None else attention_mask
        if tuple(src_tokens.shape) != tuple(tokens.shape):
            raise ValueError(
                f"patching source/target shape mismatch: source={tuple(src_tokens.shape)} target={tuple(tokens.shape)}"
            )
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(src_tokens, attention_mask=src_attn)
    elif iv == "mean":
        if mean_activation_template is None:
            raise ValueError("mean intervention requires mean_activation_template")
        mean_tpl = mean_activation_template.to(device=model.cfg.device, dtype=model.cfg.dtype)
        if mean_tpl.ndim == 2:
            mean_tpl = mean_tpl.unsqueeze(0).unsqueeze(0)
        if mean_tpl.ndim != 4 or mean_tpl.shape[0] != 1 or mean_tpl.shape[1] != 1:
            raise ValueError(f"Invalid mean_activation_template shape: {tuple(mean_tpl.shape)}")
        activation_difference += mean_tpl

    in_graph_matrix = 1 - ablation_graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    input_hooks = _make_zero_input_construction_hooks(model, ablation_graph, activation_difference, in_graph_matrix)
    with model.hooks(fwd_hooks=fwd_hooks_clean + input_hooks):
        logits = model(tokens, attention_mask=attention_mask)
    return logits

def forward_with_zero_ablation(model: Any, ablation_graph: Graph, tokens: torch.Tensor, attention_mask: torch.Tensor,) -> torch.Tensor:
    return forward_with_circuit_intervention(
        model=model,
        ablation_graph=ablation_graph,
        tokens=tokens,
        attention_mask=attention_mask,
        intervention="zero",
    )

def compute_reference_vector(
    model: Any,
    dataloader: DataLoader,
    forward_fn: Callable[..., torch.Tensor],
    target_params: Sequence[torch.Tensor],
    device: str,
    quiet: bool,
    patch_source_by_hf_id: Optional[Dict[str, List[int]]] = None,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    g_sum: Optional[torch.Tensor] = None
    count = 0
    iterator = dataloader if quiet else tqdm(dataloader, desc="reference-grad", leave=False)
    for batch in iterator:
        tokens = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        gold_ids = batch["gold_ids"].to(device)
        hf_ids = [str(x) for x in batch.get("hf_ids", [])]
        model.reset_hooks(including_permanent=True)
        model.zero_grad(set_to_none=True)
        source_tokens = None
        source_attn = None
        if patch_source_by_hf_id is not None:
            if pad_id is None:
                raise ValueError("pad_id must be set when patch_source_by_hf_id is provided")
            source_tokens, source_attn, _ = _build_patching_source_batch(
                hf_ids=hf_ids,
                target_tokens=tokens,
                target_attention_mask=attn,
                patch_source_by_hf_id=patch_source_by_hf_id,
                pad_id=int(pad_id),
            )
        logits = forward_fn(
            tokens,
            attn,
            source_tokens=source_tokens,
            source_attention_mask=source_attn,
        )
        idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[idx, pred_pos, :].float()
        loss = F.cross_entropy(last_logits, gold_ids, reduction="mean")
        grads = torch.autograd.grad(loss, target_params, create_graph=False, retain_graph=False)
        vec = flatten_grads(grads)
        if g_sum is None:
            g_sum = torch.zeros_like(vec)
        g_sum += vec
        count += 1
    if g_sum is None or count == 0:
        raise RuntimeError("Failed to compute reference gradient vector (no valid batches).")
    return g_sum / float(count)

def score_samples_by_dot(
    model: Any,
    dataloader: DataLoader,
    forward_fn: Callable[..., torch.Tensor],
    target_params: Sequence[torch.Tensor],
    target_vector: torch.Tensor,
    device: str,
    scale: float,
    quiet: bool,
    patch_source_by_hf_id: Optional[Dict[str, List[int]]] = None,
    pad_id: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    iterator = dataloader if quiet else tqdm(dataloader, desc="score-eval", leave=False)
    for batch in iterator:
        tokens = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        gold_ids = batch["gold_ids"].to(device)
        hf_ids = [str(x) for x in batch["hf_ids"]]
        for i, hf_id in enumerate(hf_ids):
            model.reset_hooks(including_permanent=True)
            model.zero_grad(set_to_none=True)
            source_tokens = None
            source_attn = None
            if patch_source_by_hf_id is not None:
                if pad_id is None:
                    raise ValueError("pad_id must be set when patch_source_by_hf_id is provided")
                source_tokens, source_attn, _ = _build_patching_source_batch(
                    hf_ids=[hf_id],
                    target_tokens=tokens[i : i + 1],
                    target_attention_mask=attn[i : i + 1],
                    patch_source_by_hf_id=patch_source_by_hf_id,
                    pad_id=int(pad_id),
                )
            logits = forward_fn(
                tokens[i : i + 1],
                attn[i : i + 1],
                source_tokens=source_tokens,
                source_attention_mask=source_attn,
            )
            pos = int(pred_pos[i].item())
            target = gold_ids[i : i + 1]
            last_logits = logits[0, pos, :].float().unsqueeze(0)
            loss = F.cross_entropy(last_logits, target, reduction="mean")
            grads = torch.autograd.grad(loss, target_params, create_graph=False, retain_graph=False)
            g_i = flatten_grads(grads)
            score = float(torch.dot(g_i, target_vector).item() * scale)
            out[hf_id] = score
    return out

def compute_gap_from_df(df: pd.DataFrame, full_col: str = "q_full", ab_col: str = "q_ablated") -> Dict[str, float]:
    hi = df[df["bucket"] == "high"]
    lo = df[df["bucket"] == "low"]
    mean_high_full = float(np.nanmean(hi[full_col].to_numpy(dtype=np.float64))) if len(hi) else float("nan")
    mean_low_full = float(np.nanmean(lo[full_col].to_numpy(dtype=np.float64))) if len(lo) else float("nan")
    mean_high_ab = float(np.nanmean(hi[ab_col].to_numpy(dtype=np.float64))) if len(hi) else float("nan")
    mean_low_ab = float(np.nanmean(lo[ab_col].to_numpy(dtype=np.float64))) if len(lo) else float("nan")

    # Per-bucket shift under ablation (useful for debugging/selectivity checks).
    delta_high = mean_high_full - mean_high_ab
    delta_low = mean_low_full - mean_low_ab
    delta_full = mean_high_full - mean_low_full
    delta_ab = mean_high_ab - mean_low_ab
    denom = max(abs(delta_full), 1e-12)
    collapse_ratio_abs = abs(delta_ab) / denom
    return {
        "mean_high_full": mean_high_full,
        "mean_low_full": mean_low_full,
        "mean_high_ablated": mean_high_ab,
        "mean_low_ablated": mean_low_ab,
        "delta_high": float(delta_high),
        "delta_low": float(delta_low),
        "delta_full": float(delta_full),
        "delta_ablated": float(delta_ab),
        "collapse_ratio_abs": float(collapse_ratio_abs),
        "gap_reduction_ratio": float(1.0 - collapse_ratio_abs),
    }

def build_joined_eval_df(eval_rows: Sequence[Dict[str, Any]], ablated_scores_by_hf_id: Dict[str, float], full_recomputed_by_hf_id: Optional[Dict[str, float]] = None,) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for r in eval_rows:
        hf_id = str(r["hf_id"])
        if hf_id not in ablated_scores_by_hf_id:
            continue
        rr = {
            "hf_id": hf_id,
            "bucket": str(r.get("bucket", "")),
            "q_full": float(r.get("full_score", np.nan)),
            "q_ablated": float(ablated_scores_by_hf_id[hf_id]),
        }
        rr["delta"] = float(rr["q_full"] - rr["q_ablated"])
        rr["abs_delta"] = float(abs(rr["delta"]))
        if full_recomputed_by_hf_id is not None:
            rr["q_full_recomputed"] = float(full_recomputed_by_hf_id.get(hf_id, np.nan))
        recs.append(rr)
    df = pd.DataFrame.from_records(recs)
    if len(df) == 0:
        return df
    df = df.sort_values(["bucket", "hf_id"]).reset_index(drop=True)
    return df

def _score_file_to_map(score_json: Path, score_key: str) -> Dict[str, float]:
    obj = read_json(score_json)
    rows = obj.get("items", obj if isinstance(obj, list) else [])
    out: Dict[str, float] = {}
    if not isinstance(rows, list):
        return out
    for r in rows:
        hf_id = str(r.get("hf_id", "")).strip()
        if not hf_id:
            continue
        v = extract_score_from_row(r, score_key)
        if v is None:
            continue
        out[hf_id] = float(v)
    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Circuit zero-ablation validation: compare Δ=E[q|high]-E[q|low] vs Δ(-C) with fixed original buckets.")
    ap.add_argument("--root", type=str, default=str(_ROOT))
    ap.add_argument("--circuits_root", type=str, default=str(_ROOT / "circuit_results"))
    ap.add_argument("--buckets_root", type=str, default=str(_ROOT / "data_gens" / "buckets"))
    ap.add_argument("--out_dir", type=str, default=str(_ROOT / "results" / "validation" / "zero_ablation"))
    ap.add_argument("--dataset", type=str, required=True, help="arc_easy/sciq/piqa/winogrande")
    ap.add_argument("--score_key", type=str, required=True, choices=ALL_SCORE_KEYS)
    ap.add_argument("--corrupt_type", type=str, default="gaussian")
    ap.add_argument("--circuits_to_ablate", type=str, default="both", choices=["both", "high", "low", "random"])

    ap.add_argument("--all_scores_json", type=str, default="", help="Override full-score JSON path")
    ap.add_argument("--high_bucket_json", type=str, default="", help="Override high bucket JSON path")
    ap.add_argument("--low_bucket_json", type=str, default="", help="Override low bucket JSON path")
    ap.add_argument("--topk_edges", type=int, default=500, help="<=0 uses saved in_graph edges; >0 takes top-k edge scores")
    ap.add_argument("--topk_rank_signed", dest="topk_rank_abs", action="store_false", help="Rank circuit edges by signed score")
    ap.set_defaults(topk_rank_abs=True)
    ap.add_argument("--random_edges_seed", type=int, default=42, help="Seed for random edge selection when circuits_to_ablate=random")
    ap.add_argument("--random_edge_pool", type=str, default="model", choices=["model", "circuit"], help="Random edge source for circuits_to_ablate=random")
    ap.add_argument("--random_circuit_bucket", type=str, default="high", choices=["high", "low"], help="When random_edge_pool=circuit, pick random edges from this bucket's circuit")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Used for static score keys")
    ap.add_argument("--checkpoint_state_dict", type=str, default="", help="Optional local state_dict checkpoint for static scoring model")
    ap.add_argument("--reference_vector_source", type=str, default="full", choices=["full", "ablated"])
    ap.add_argument("--reference_scores_json", type=str, default="", help="Optional JSON (same format as all_scores) used only for reference gradient vector")
    ap.add_argument("--max_reference_samples", type=int, default=512, help="0 means all available")
    ap.add_argument("--tracin_lr", type=float, default=2e-5, help="Scale for tracin score at single checkpoint")
    ap.add_argument("--save_full_recomputed", action="store_true", help="Also recompute q(x) from current full model for sanity checks")
    ap.add_argument(
        "--static_ablation_intervention",
        type=str,
        default="zero",
        choices=["zero", "mean", "patching"],
        help="Intervention used for static score re-computation.",
    )
    ap.add_argument(
        "--mean_source",
        type=str,
        default="reference",
        choices=["reference", "eval"],
        help="Dataset used to compute mean activation template when static_ablation_intervention=mean.",
    )
    ap.add_argument(
        "--mean_max_samples",
        type=int,
        default=512,
        help="Max samples for mean template computation (0 means all).",
    )

    ap.add_argument("--dynamic_ablated_high_json", type=str, default="", help="For dynamic score keys: q^(-C_high) all-scores JSON")
    ap.add_argument("--dynamic_ablated_low_json", type=str, default="", help="For dynamic score keys: q^(-C_low) all-scores JSON")
    ap.add_argument("--dynamic_ablated_random_json", type=str, default="", help="For dynamic score keys: q^(-C_random) all-scores JSON")

    ap.add_argument("--max_samples_per_bucket", type=int, default=0, help="0 means all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    score_key = args.score_key
    dataset = args.dataset
    group = infer_score_group(score_key)
    circuits_root = Path(args.circuits_root)
    buckets_root = Path(args.buckets_root)
    base_out_root = Path(args.out_dir) / dataset / score_key
    static_intervention_arg = str(args.static_ablation_intervention).strip().lower()
    out_root = base_out_root
    if score_key in STATIC_SCORE_KEYS:
        out_root = base_out_root / f"static_{static_intervention_arg}"
    ensure_dir(out_root)

    all_scores_json = Path(args.all_scores_json) if args.all_scores_json else find_all_scores_json(
        buckets_root=buckets_root,
        dataset=dataset,
        score_key=score_key,
        group=group,
    )
    if all_scores_json is None or (not all_scores_json.is_file()):
        raise FileNotFoundError(f"all_scores json not found (dataset={dataset}, score_key={score_key}, group={group})")

    high_bucket_json = Path(args.high_bucket_json) if args.high_bucket_json else find_bucket_json(
        buckets_root=buckets_root,
        dataset=dataset,
        score_key=score_key,
        bucket="high",
        group=group,
    )
    low_bucket_json = Path(args.low_bucket_json) if args.low_bucket_json else find_bucket_json(
        buckets_root=buckets_root,
        dataset=dataset,
        score_key=score_key,
        bucket="low",
        group=group,
    )
    if high_bucket_json is None or not high_bucket_json.is_file():
        raise FileNotFoundError(f"High bucket json not found for {dataset}/{score_key}")
    if low_bucket_json is None or not low_bucket_json.is_file():
        raise FileNotFoundError(f"Low bucket json not found for {dataset}/{score_key}")

    model_for_loading = None
    load_tokenizer = False
    if score_key in STATIC_SCORE_KEYS:
        load_tokenizer = True
    tokenizer = None
    if load_tokenizer:
        dtype = str_to_dtype(args.dtype)
        if (not args.device.startswith("cuda")) and dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32
        model_name = normalize_model_name(args.model_name)
        model_for_loading = load_model(model_name=model_name, device=args.device, dtype=dtype)
        tokenizer = model_for_loading.tokenizer

    all_rows_by_hf_id, all_stats = load_all_score_rows(all_scores_json, score_key, tokenizer=tokenizer)
    high_ids = load_bucket_ids(high_bucket_json)
    low_ids = load_bucket_ids(low_bucket_json)
    eval_rows, eval_stats = build_eval_rows(
        all_rows_by_id=all_rows_by_hf_id,
        high_ids=high_ids,
        low_ids=low_ids,
        max_samples_per_bucket=int(args.max_samples_per_bucket),
        seed=int(args.seed),
    )
    if len(eval_rows) == 0:
        raise RuntimeError("No evaluation rows after joining all_scores with high/low bucket IDs.")

    if args.circuits_to_ablate == "both":
        circuits = ["high", "low"]
    else:
        circuits = [args.circuits_to_ablate]
    report: Dict[str, Any] = {
        "dataset": dataset,
        "score_key": score_key,
        "group": group,
        "paths": {
            "all_scores_json": str(all_scores_json),
            "high_bucket_json": str(high_bucket_json),
            "low_bucket_json": str(low_bucket_json),
            "artifact_root": str(out_root),
            "artifact_root_base": str(base_out_root),
        },
        "load_stats": all_stats,
        "eval_stats": {
            "n_eval": int(eval_stats.n_eval),
            "n_high": int(eval_stats.n_high),
            "n_low": int(eval_stats.n_low),
        },
        "config": vars(args),
        "circuits": {},
    }

    if score_key in DYNAMIC_SCORE_KEYS:
        dyn_paths = {
            "high": Path(args.dynamic_ablated_high_json) if args.dynamic_ablated_high_json else None,
            "low": Path(args.dynamic_ablated_low_json) if args.dynamic_ablated_low_json else None,
            "random": Path(args.dynamic_ablated_random_json) if args.dynamic_ablated_random_json else None,
        }
        for cb in circuits:
            score_path = dyn_paths.get(cb)
            if score_path is None or (not score_path.is_file()):
                report["circuits"][cb] = {
                    "status": "skip",
                    "reason": f"missing dynamic ablated score json for circuit={cb}. set --dynamic_ablated_{cb}_json",
                }
                continue
            ab_scores = _score_file_to_map(score_path, score_key)
            df = build_joined_eval_df(eval_rows, ab_scores)
            if len(df) == 0:
                report["circuits"][cb] = {"status": "empty", "reason": "no overlapping hf_id between eval rows and ablated scores"}
                continue
            gap = compute_gap_from_df(df)
            corr = corr_dict(df["q_full"].values, df["q_ablated"].values)
            csv_path = out_root / f"{dataset}_{score_key}_dynamic_{cb}_sample_scores.csv"
            png_path = out_root / f"{dataset}_{score_key}_dynamic_{cb}_scatter.png"
            df.to_csv(csv_path, index=False)
            save_full_vs_ablated_scatter(
                df=df,
                score_key=score_key,
                title=f"{dataset}/{score_key} | dynamic q^(-C_{cb}) vs q",
                out_path=png_path,
            )
            report["circuits"][cb] = {
                "status": "ok",
                "mode": "dynamic_from_json",
                "score_json": str(score_path),
                "n_used": int(len(df)),
                "gap": gap,
                "corr_full_vs_ablated": corr,
                "artifacts": {"sample_csv": str(csv_path), "scatter_png": str(png_path)},
            }

        latest_report, target_report, summary_json = save_report_with_target_and_summary(
            report=report,
            out_root=out_root,
            dataset=dataset,
            score_key=score_key,
            circuits=circuits,
        )
        print(f"[OK] saved report(latest): {latest_report}")
        print(f"[OK] saved report(target): {target_report}")
        print(f"[OK] saved summary      : {summary_json}")
        return

    # Static score keys: run gradient-based q^(-C)(x) re-computation with configurable intervention.
    assert score_key in STATIC_SCORE_KEYS
    model = model_for_loading
    assert model is not None

    if args.checkpoint_state_dict:
        ckpt_path = Path(args.checkpoint_state_dict)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint_state_dict not found: {ckpt_path}")
        state_obj = torch.load(str(ckpt_path), map_location=args.device)
        state_dict = state_obj.get("model_state_dict", state_obj) if isinstance(state_obj, dict) else state_obj
        model.load_state_dict(state_dict, strict=False)
        report["loaded_state_dict"] = str(ckpt_path)
    else:
        report["loaded_state_dict"] = ""

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    eval_ds = PromptScoreDataset(eval_rows, tokenizer=tokenizer, max_length=int(args.max_length))
    if len(eval_ds) == 0:
        raise RuntimeError("No usable eval dataset rows after tokenization.")
    eval_dl = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda b: collate_prompt_score(b, pad_id),
    )

    if args.reference_scores_json:
        ref_json_path = Path(args.reference_scores_json)
        if not ref_json_path.is_file():
            raise FileNotFoundError(f"reference_scores_json not found: {ref_json_path}")
        ref_rows_by_hf_id, ref_stats = load_all_score_rows(ref_json_path, score_key, tokenizer=tokenizer)
        report["static_reference_scores_json"] = str(ref_json_path)
        report["static_reference_load_stats"] = ref_stats
        ref_candidates = _attach_bucket_to_rows(list(ref_rows_by_hf_id.values()), high_ids=high_ids, low_ids=low_ids)
    else:
        ref_candidates = _attach_bucket_to_rows(list(all_rows_by_hf_id.values()), high_ids=high_ids, low_ids=low_ids)
    if int(args.max_reference_samples) > 0 and len(ref_candidates) > int(args.max_reference_samples):
        rng = random.Random(int(args.seed) + 17)
        ref_candidates = rng.sample(ref_candidates, int(args.max_reference_samples))
    ref_ds = PromptScoreDataset(ref_candidates, tokenizer=tokenizer, max_length=int(args.max_length))
    if len(ref_ds) == 0:
        raise RuntimeError("No usable reference rows for static gradient scoring.")
    ref_dl = DataLoader(
        ref_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lambda b: collate_prompt_score(b, pad_id),
    )

    model.eval()
    target_params = get_target_params(model)
    static_intervention = static_intervention_arg

    eval_patch_lookup: Dict[str, List[int]] = {}
    eval_patch_stats: Dict[str, int] = {}
    ref_patch_lookup: Dict[str, List[int]] = {}
    ref_patch_stats: Dict[str, int] = {}
    mean_activation_template: Optional[torch.Tensor] = None
    mean_template_stats: Dict[str, int] = {}

    if static_intervention == "patching":
        eval_patch_lookup, eval_patch_stats = build_opposite_bucket_patch_lookup(
            rows=eval_ds.rows,
            seed=int(args.seed) + 101,
        )
        ref_patch_lookup, ref_patch_stats = build_opposite_bucket_patch_lookup(
            rows=ref_ds.rows,
            seed=int(args.seed) + 211,
        )
    elif static_intervention == "mean":
        mean_source_dl = ref_dl if args.mean_source == "reference" else eval_dl
        mean_graph = Graph.from_model(model)
        mean_graph.reset(empty=False)
        mean_activation_template, mean_template_stats = compute_mean_activation_template(
            model=model,
            graph=mean_graph,
            dataloader=mean_source_dl,
            device=args.device,
            quiet=bool(args.quiet),
            max_samples=int(args.mean_max_samples),
        )
        del mean_graph

    report["static_ablation"] = {
        "intervention": static_intervention,
        "mean_source": str(args.mean_source),
        "mean_template_stats": mean_template_stats,
        "patching_eval_stats": eval_patch_stats,
        "patching_reference_stats": ref_patch_stats,
    }

    def forward_full(
        tokens: torch.Tensor,
        attn: torch.Tensor,
        source_tokens: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del source_tokens, source_attention_mask
        return model(tokens, attention_mask=attn)

    ref_vec_full = compute_reference_vector(
        model=model,
        dataloader=ref_dl,
        forward_fn=forward_full,
        target_params=target_params,
        device=args.device,
        quiet=bool(args.quiet),
    )
    report["static_reference"] = {
        "source": str(args.reference_vector_source),
        "intervention": static_intervention,
        "n_reference_rows": int(len(ref_ds)),
        "target_vector_dim": int(ref_vec_full.numel()),
    }

    full_recomputed: Optional[Dict[str, float]] = None
    static_scale = float(args.tracin_lr) if score_key == "tracin" else 1.0
    should_recompute_full = bool(args.save_full_recomputed or args.checkpoint_state_dict)
    if should_recompute_full:
        full_recomputed = score_samples_by_dot(
            model=model,
            dataloader=eval_dl,
            forward_fn=forward_full,
            target_params=target_params,
            target_vector=ref_vec_full,
            device=args.device,
            scale=static_scale,
            quiet=bool(args.quiet),
        )

    for cb in circuits:
        circuit_pt: Optional[Path] = None
        edge_meta: Dict[str, Any] = {}
        if cb == "random":
            if args.random_edge_pool == "circuit":
                circuit_pt = find_latest_circuit_pt(
                    circuits_root=circuits_root,
                    dataset=dataset,
                    score_key=score_key,
                    bucket=str(args.random_circuit_bucket),
                    corrupt_type=args.corrupt_type,
                    group=group,
                )
                if circuit_pt is None or (not circuit_pt.is_file()):
                    report["circuits"][cb] = {
                        "status": "skip",
                        "reason": (
                            f"random_edge_pool=circuit but circuit pt not found "
                            f"for bucket={args.random_circuit_bucket}"
                        ),
                    }
                    continue
                ab_graph, edge_meta = build_random_ablation_graph_from_circuit(
                    model=model,
                    circuit_pt=circuit_pt,
                    topk_edges=int(args.topk_edges),
                    random_seed=int(args.random_edges_seed),
                )
            else:
                ab_graph, edge_meta = build_random_ablation_graph_from_model(
                    model=model,
                    topk_edges=int(args.topk_edges),
                    random_seed=int(args.random_edges_seed),
                )
            edge_meta["mode"] = "random_edges"

            def forward_ablated(
                tokens: torch.Tensor,
                attn: torch.Tensor,
                source_tokens: Optional[torch.Tensor] = None,
                source_attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                return forward_with_circuit_intervention(
                    model=model,
                    ablation_graph=ab_graph,
                    tokens=tokens,
                    attention_mask=attn,
                    intervention=static_intervention,
                    source_tokens=source_tokens,
                    source_attention_mask=source_attention_mask,
                    mean_activation_template=mean_activation_template,
                )
        else:
            circuit_pt = find_latest_circuit_pt(
                circuits_root=circuits_root,
                dataset=dataset,
                score_key=score_key,
                bucket=cb,
                corrupt_type=args.corrupt_type,
                group=group,
            )
            if circuit_pt is None or (not circuit_pt.is_file()):
                report["circuits"][cb] = {"status": "skip", "reason": f"circuit pt not found for bucket={cb}"}
                continue
            ab_graph, edge_meta = build_ablation_graph(
                model=model,
                circuit_pt=circuit_pt,
                topk_edges=int(args.topk_edges),
                rank_abs=bool(args.topk_rank_abs),
            )

            def forward_ablated(
                tokens: torch.Tensor,
                attn: torch.Tensor,
                source_tokens: Optional[torch.Tensor] = None,
                source_attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                return forward_with_circuit_intervention(
                    model=model,
                    ablation_graph=ab_graph,
                    tokens=tokens,
                    attention_mask=attn,
                    intervention=static_intervention,
                    source_tokens=source_tokens,
                    source_attention_mask=source_attention_mask,
                    mean_activation_template=mean_activation_template,
                )

        if args.reference_vector_source == "ablated":
            target_vec = compute_reference_vector(
                model=model,
                dataloader=ref_dl,
                forward_fn=forward_ablated,
                target_params=target_params,
                device=args.device,
                quiet=bool(args.quiet),
                patch_source_by_hf_id=(ref_patch_lookup if static_intervention == "patching" else None),
                pad_id=pad_id,
            )
        else:
            target_vec = ref_vec_full

        ab_scores = score_samples_by_dot(
            model=model,
            dataloader=eval_dl,
            forward_fn=forward_ablated,
            target_params=target_params,
            target_vector=target_vec,
            device=args.device,
            scale=static_scale,
            quiet=bool(args.quiet),
            patch_source_by_hf_id=(eval_patch_lookup if static_intervention == "patching" else None),
            pad_id=pad_id,
        )
        df = build_joined_eval_df(eval_rows, ab_scores, full_recomputed_by_hf_id=full_recomputed)
        if len(df) == 0:
            report["circuits"][cb] = {"status": "empty", "reason": "no joined rows after static scoring"}
            continue

        gap = compute_gap_from_df(df)
        gap_full_source = "q_full"
        corr_full_source = "q_full"
        recomputed_gap: Optional[Dict[str, float]] = None
        if "q_full_recomputed" in df.columns and bool(df["q_full_recomputed"].notna().any()):
            recomputed_gap = compute_gap_from_df(df, full_col="q_full_recomputed", ab_col="q_ablated")
            gap = recomputed_gap
            gap_full_source = "q_full_recomputed"
            corr_full_source = "q_full_recomputed"
        corr = corr_dict(df[corr_full_source].values, df["q_ablated"].values)
        csv_path = out_root / f"{dataset}_{score_key}_static_{cb}_sample_scores.csv"
        png_path = out_root / f"{dataset}_{score_key}_static_{cb}_scatter.png"
        df.to_csv(csv_path, index=False)
        save_full_vs_ablated_scatter(
            df=df,
            score_key=score_key,
            title=f"{dataset}/{score_key} | static q^(-C_{cb}) vs q",
            out_path=png_path,
        )
        recomputed_png_path: Optional[Path] = None
        if "q_full_recomputed" in df.columns and bool(df["q_full_recomputed"].notna().any()):
            recomputed_png_path = out_root / f"{dataset}_{score_key}_static_{cb}_scatter_recomputed_vs_ablated.png"
            save_full_vs_ablated_scatter(
                df=df,
                score_key=score_key,
                title=f"{dataset}/{score_key} | static q_full(recomputed) vs q^(-C_{cb})",
                out_path=recomputed_png_path,
                x_col="q_full_recomputed",
                y_col="q_ablated",
                x_label=f"Recomputed full q(x) on loaded model: {score_key}",
                y_label=f"Ablated q^(-C)(x) on loaded model: {score_key}",
            )

        circuit_report: Dict[str, Any] = {
            "status": "ok",
            "mode": ("static_random_edges" if cb == "random" else "static_gradient"),
            "intervention": static_intervention,
            "circuit_pt": str(circuit_pt) if circuit_pt is not None else "",
            "n_used": int(len(df)),
            "gap": gap,
            "gap_full_source": gap_full_source,
            "corr_full_vs_ablated": corr,
            "corr_full_source": corr_full_source,
            "edge_meta": edge_meta,
            "artifacts": {"sample_csv": str(csv_path), "scatter_png": str(png_path)},
        }
        if recomputed_gap is not None:
            circuit_report["gap_from_original_full_json"] = compute_gap_from_df(df, full_col="q_full", ab_col="q_ablated")
        if recomputed_png_path is not None:
            circuit_report["artifacts"]["scatter_recomputed_vs_ablated_png"] = str(recomputed_png_path)
        if "q_full_recomputed" in df.columns:
            circuit_report["corr_original_vs_recomputed_full"] = corr_dict(
                df["q_full"].values, df["q_full_recomputed"].values
            )
            circuit_report["corr_recomputed_full_vs_ablated"] = corr_dict(
                df["q_full_recomputed"].values, df["q_ablated"].values
            )
        report["circuits"][cb] = circuit_report

    latest_report, target_report, summary_json = save_report_with_target_and_summary(
        report=report,
        out_root=out_root,
        dataset=dataset,
        score_key=score_key,
        circuits=circuits,
    )
    print(f"[OK] saved report(latest): {latest_report}")
    print(f"[OK] saved report(target): {target_report}")
    print(f"[OK] saved summary      : {summary_json}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
