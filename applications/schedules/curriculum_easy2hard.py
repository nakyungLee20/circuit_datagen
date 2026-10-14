"""Prepare the ranked pool for the Bengio-style easy → hard baseline.

Rows from the clean easy/hard/helpful selected pools are first normalized with
the downstream trainer's exact converter, then scored by Qwen-0.5B margin.
The requested pool is sampled across the *full* high-margin→low-margin range,
not from an easiest-only prefix. Storage chunks are labelled easy/medium/hard,
but training order is enforced later by ``ordered_schedule_train.py``: it
recombines the global ranks, creates monotonic physical batches, and disables
DataLoader shuffling.
"""

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
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from applications.experiment_utils.io import (
    GENERATOR_MODEL_KEY,
    dump_pool_as_mirror,
    read_jsonl,
    stamp_row_metadata,
)

from applications import curriculum_ft as curriculum_core  # type: ignore  # noqa: E402

EXP3_WORK_ROOT = _SUPPLEMENTAL_ROOT / "schedules" / "work_exp3_sams_schedule"
DEFAULT_STEERED_GEN_ROOT = _REPO_ROOT / 'results/circuit_gen_prev'


def resolve_steered_selected_path(generation_root: Path, profile: str) -> Path:
    """Resolve the exact clean pool consumed by the other Exp3 conditions."""

    return curriculum_core.find_latest_generated_file(
        generation_root=Path(generation_root).expanduser(),
        generator_model_key=GENERATOR_MODEL_KEY,
        recipe=profile,
        file_kind="selected",
        prompt_mode="fixed",
        gen_mode="both",
        circuit_target_mode="discovered",
        path_must_contain="",
    )


def build_prompt(tok, question, choices):
    sys.path.insert(0, str(_REPO_ROOT))
    from data_gens.mcqa import format_mcqa_prompt  # type: ignore
    label_space = [chr(ord("A") + idx) for idx in range(len(choices))]
    prompt, _answer_prefix, label_token_ids = format_mcqa_prompt(
        question_text=question,
        choices=choices,
        label_space=label_space,
        tokenizer=tok,
    )
    return prompt, label_token_ids


def margin_for(model, tok, prompt, label_token_ids, gold_label, device):
    inp = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]
    labels = list(label_token_ids)
    opts = torch.stack([logits[label_token_ids[label]] for label in labels])
    gold_idx = labels.index(gold_label)
    others = torch.cat([opts[:gold_idx], opts[gold_idx + 1:]])
    return float(opts[gold_idx].item() - torch.max(others).item())


def resolve_gold_label(row, choices):
    answer = str(row.get("answer") or row.get("gold_label") or "").strip()
    labels = [chr(ord("A") + idx) for idx in range(len(choices))]
    if len(answer) == 1 and answer.upper() in labels:
        return answer.upper()
    normalized = answer.lower()
    for idx, choice in enumerate(choices):
        if str(choice).strip().lower() == normalized:
            return labels[idx]
    raise ValueError(
        f"cannot resolve answer={answer!r} for candidate={row.get('candidate_id')!r}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-root", default=str(EXP3_WORK_ROOT / "curriculum_e2h"))
    ap.add_argument("--steered-gen-root", default=str(DEFAULT_STEERED_GEN_ROOT))
    ap.add_argument("--proxy-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="forward-only model whose margin defines difficulty")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-total", type=int, default=12000)
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.proxy_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.proxy_model, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(args.device).eval()

    all_rows: List[dict] = []
    dropped_invalid = 0
    for band in ("easy", "hard", "helpful"):
        selected_path = resolve_steered_selected_path(
            Path(args.steered_gen_root), band
        )
        print(f"[input] clean steered {band}: {selected_path}")
        rows = read_jsonl(selected_path)
        if args.limit:
            rows = rows[: args.limit]
        for row in rows:
            converted = curriculum_core.generated_row_to_example(row, args.dataset)
            if converted is None:
                dropped_invalid += 1
                continue
            copied = dict(row)
            # Normalize to precisely the representation that the downstream
            # trainer accepts, so a row scored here cannot be dropped later.
            copied["question"] = converted["question_text"]
            copied["choices"] = list(converted["choices"])
            copied["answer"] = converted["gold_label"]
            copied["exp3_source_band"] = band
            copied["_exp3_proxy_example"] = converted
            all_rows.append(copied)
    print(
        f"[load] valid concatenated synthetic pool = {len(all_rows)} "
        f"(trainer-invalid dropped={dropped_invalid})"
    )
    if len(all_rows) < args.n_total:
        raise ValueError(
            f"after trainer-identical validation only {len(all_rows)} rows remain; "
            f"cannot build n_total={args.n_total}"
        )

    # Score each row with proxy-model margin
    scored: List[tuple] = []
    for i, r in enumerate(all_rows):
        converted = r["_exp3_proxy_example"]
        q = converted["question_text"]
        choices = converted["choices"]
        gold = converted["gold_label"]
        prompt, label_token_ids = build_prompt(tok, q, choices)
        mgn = margin_for(model, tok, prompt, label_token_ids, gold, args.device)
        scored.append((mgn, i))
        if i % 1000 == 0:
            print(f"  scored {i}/{len(all_rows)}  margin={mgn:.3f}")

    # Sort ascending by margin: low margin = hard, high margin = easy
    scored.sort(key=lambda p: p[0])
    ordered_hard_to_easy = [all_rows[i] for _, i in scored]
    # Bengio curriculum wants easy → hard, so reverse
    ordered_all = list(reversed(ordered_hard_to_easy))
    if len(ordered_all) < args.n_total:
        raise ValueError(
            f"requested n_total={args.n_total}, but only {len(ordered_all)} rows are available"
        )
    if len(ordered_all) == args.n_total:
        ordered = ordered_all
    elif args.n_total == 1:
        ordered = [ordered_all[0]]
    else:
        # Preserve the full easy->hard support when reducing ~29k valid rows
        # to the requested 12k. Taking a prefix would silently keep only the
        # easiest portion and invalidate the curriculum baseline.
        indices = [
            round(idx * (len(ordered_all) - 1) / (args.n_total - 1))
            for idx in range(args.n_total)
        ]
        ordered = [ordered_all[idx] for idx in indices]

    margin_by_object_id = {id(all_rows[idx]): float(margin) for margin, idx in scored}
    ranked = []
    for rank, row in enumerate(ordered):
        copied = dict(row)
        copied.pop("_exp3_proxy_example", None)
        copied["exp3_proxy_margin"] = margin_by_object_id[id(row)]
        copied["exp3_difficulty_rank"] = rank
        copied["exp3_difficulty_quantile"] = rank / max(1, len(ordered) - 1)
        ranked.append(copied)
    ordered = ranked

    # Chop into 3 chunks, relabel as easy / medium / hard
    third = len(ordered) // 3
    chunks = {"easy": ordered[:third], "medium": ordered[third: 2 * third], "hard": ordered[2 * third:]}

    for recipe, rows in chunks.items():
        stamped = [
            stamp_row_metadata(r, target_band=recipe, prompt_mode="fixed", gen_mode="both", circuit_target_mode="discovered")
            for r in rows
        ]
        dump_pool_as_mirror(
            stamped, variant_root=Path(args.variant_root),
            recipe=recipe, prompt_mode="fixed", gen_mode="both",
            circuit_target_mode="discovered", file_kind="selected",
            force_stamp_metadata=False,
        )
        print(f"[write] {recipe}: {len(stamped)} rows")


if __name__ == "__main__":
    main()
