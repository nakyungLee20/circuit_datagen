"""Opt-in Hugging Face chat-template formatter for rebuttal cross-model SFT.

The core trainer remains unchanged.  Exp1's cross-model entrypoint installs
``format_chat_mcqa_prompt`` into ``data_gens.mcqa`` before loading the trainer, so
MCQATokenDataset uses the student's own chat template for both train and val.
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


import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


def format_chat_mcqa_prompt(
    question_text: str,
    choices: Sequence[str],
    label_space: Sequence[str],
    tokenizer: Any,
    prefix_context: str = "",
) -> Tuple[str, str, Dict[str, int]]:
    """Render an MCQA user turn and leave the assistant at a one-token answer.

    The incremental-tokenization check is essential: MCQATokenDataset appends
    ``gold_id`` to the already-tokenized prompt, so ``prompt + label`` must add
    exactly that one token without re-tokenizing the prompt boundary.
    """
    labels = [str(label) for label in label_space]
    if len(labels) != len(choices) or len(labels) < 2:
        raise ValueError("chat MCQA requires one label per choice and at least two choices")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"tokenizer {getattr(tokenizer, 'name_or_path', '<unknown>')} has no chat_template"
        )

    lines = [f"{prefix_context}{str(question_text).strip()}", ""]
    lines.extend(f"{label}. {str(choice).strip()}" for label, choice in zip(labels, choices))
    lines.extend(["", f"Answer with exactly one label from: {', '.join(labels)}."])
    messages = [{"role": "user", "content": "\n".join(lines)}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer.apply_chat_template returned an empty/non-string prompt")

    # Some templates end at a whitespace-stable assistant boundary; others need
    # an explicit answer prefix before A/B/C/D becomes one incremental token.
    suffixes = ("", " ", "\n", "Answer: ", "\nAnswer: ", "\nAnswer:\n")
    for answer_prefix in suffixes:
        prompt = rendered + answer_prefix
        base_ids = tokenizer.encode(prompt, add_special_tokens=False)
        label_ids: Dict[str, int] = {}
        valid = True
        for label in labels:
            full_ids = tokenizer.encode(prompt + label, add_special_tokens=False)
            if len(full_ids) != len(base_ids) + 1 or full_ids[:-1] != base_ids:
                valid = False
                break
            label_ids[label] = int(full_ids[-1])
        if valid:
            return prompt, answer_prefix, label_ids
    raise ValueError(
        "chat template could not produce a stable single-token answer boundary "
        f"for labels={labels}"
    )


def install_chat_formatter():
    """Patch ``data_gens.mcqa.format_mcqa_prompt`` and return the old formatter."""
    from data_gens import mcqa  # type: ignore

    previous = mcqa.format_mcqa_prompt
    mcqa.format_mcqa_prompt = format_chat_mcqa_prompt
    return previous


def mark_chat_template_runs(output_root: str, run_name: str, model_name_or_path: str) -> int:
    """Persist the prompt contract so post-hoc evaluation uses the same adapter."""
    root = Path(output_root).expanduser()
    marked = 0
    if not root.is_dir():
        return marked
    for summary_path in root.rglob("run_summary.json"):
        if run_name and summary_path.parent.name != run_name:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("model_name_or_path") != model_name_or_path:
            continue
        summary["prompt_format"] = "hf_chat_template"
        summary["prompt_formatter"] = "applications.experiment_utils.chat_prompt.format_chat_mcqa_prompt"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        config_path = summary_path.parent / "run_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["prompt_format"] = "hf_chat_template"
            config["prompt_formatter"] = "applications.experiment_utils.chat_prompt.format_chat_mcqa_prompt"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        marked += 1
    return marked
