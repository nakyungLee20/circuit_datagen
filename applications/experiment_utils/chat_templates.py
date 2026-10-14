"""Smoke-test the exact plain and chat MCQA contracts used by Exp1.

Qwen 1a/1b keeps the core trainer's plain ``format_mcqa_prompt``. Llama/Gemma
1c is launched through ``chat_curriculum_entry.py`` and therefore uses the
student tokenizer's real Hugging Face chat template. This command validates
both one-token answer boundaries before expensive cross-model runs begin.
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
import json
import sys
from pathlib import Path
from typing import List

from transformers import AutoTokenizer


from data_gens.mcqa import format_mcqa_prompt  # type: ignore  # noqa: E402
from applications.experiment_utils.chat_prompt import format_chat_mcqa_prompt  # noqa: E402
from applications.experiment_utils.io import NAIVE_SELECTED_PATH  # noqa: E402

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
]


def peek_first_row():
    with NAIVE_SELECTED_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty jsonl: {NAIVE_SELECTED_PATH}")


def _validate_incremental_ids(tokenizer, prompt: str, label_ids, labels: List[str]) -> None:
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for label in labels:
        full_ids = tokenizer.encode(prompt + label, add_special_tokens=False)
        expected = int(label_ids[label])
        if len(full_ids) != len(base_ids) + 1 or full_ids[:-1] != base_ids or full_ids[-1] != expected:
            raise ValueError(
                f"label={label!r} is not one stable incremental token after prompt; "
                f"returned_id={expected}"
            )


def smoke_model(model_id: str, row, *, local_files_only: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    question = row["question"]
    choices = [str(choice) for choice in row["choices"]]
    labels = [chr(ord("A") + idx) for idx in range(len(choices))]

    plain_prompt, plain_prefix, plain_ids = format_mcqa_prompt(
        question_text=question,
        choices=choices,
        label_space=labels,
        tokenizer=tokenizer,
    )
    _validate_incremental_ids(tokenizer, plain_prompt, plain_ids, labels)

    chat_prompt, chat_prefix, chat_ids = format_chat_mcqa_prompt(
        question_text=question,
        choices=choices,
        label_space=labels,
        tokenizer=tokenizer,
    )
    _validate_incremental_ids(tokenizer, chat_prompt, chat_ids, labels)
    print(
        f"[OK] {model_id}: plain_tokens={len(tokenizer.encode(plain_prompt, add_special_tokens=False))} "
        f"plain_prefix={plain_prefix!r}; chat_tokens={len(tokenizer.encode(chat_prompt, add_special_tokens=False))} "
        f"chat_prefix={chat_prefix!r}; chat_label_ids={chat_ids}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model ids/paths")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any tokenizer/contract fails")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise ValueError("--models resolved to an empty list")
    row = peek_first_row()
    failures = []
    for model_id in models:
        try:
            smoke_model(model_id, row, local_files_only=args.local_files_only)
        except Exception as exc:
            failures.append((model_id, exc))
            print(f"[FAIL] {model_id}: {type(exc).__name__}: {exc}")
    if failures and args.strict:
        raise SystemExit(1)
    if failures:
        print(f"[WARN] {len(failures)}/{len(models)} model smoke tests failed")


if __name__ == "__main__":
    main()
