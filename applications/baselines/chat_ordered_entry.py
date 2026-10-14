"""Run ordered_schedule_train.py with the rebuttal-only chat formatter installed.

The ordered-batch trainer was written for Exp3, whose student is Qwen and which
therefore runs on the core trainer's plain MCQA formatter. Exp1's 1c students
(Llama/Gemma) are chat models and every existing 1c run went through
``chat_curriculum_entry.py`` so their prompts carry the HF chat template. Running
the ordered trainer on those students without the same adapter would change the
prompt contract, and the resulting numbers would not be comparable to any other
1c row.

This is the ordered-trainer twin of ``chat_curriculum_entry.py``: same formatter
install, same post-run marking so ``eval_downstream`` reads the run back with
the adapter it was trained under. Qwen runs must NOT go through here -- they
stay bit-for-bit on the original formatter.

  python chat_ordered_entry.py --model-name-or-path google/gemma-3-1b-it ...
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


import sys
from pathlib import Path
from typing import List

APPLICATIONS_ROOT = _REPO_ROOT / 'applications'
if str(APPLICATIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPLICATIONS_ROOT))

from applications.experiment_utils.chat_prompt import install_chat_formatter, mark_chat_template_runs


def _arg_value(args: List[str], flag: str, default: str = "") -> str:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def main() -> None:
    trainer_args = sys.argv[1:]
    if not trainer_args:
        raise SystemExit("usage: chat_ordered_entry.py [ordered_schedule_train args ...]")

    output_root = _arg_value(trainer_args, "--output-root")
    run_name = _arg_value(trainer_args, "--run-name")
    model_name_or_path = _arg_value(trainer_args, "--model-name-or-path")
    if not (output_root and run_name and model_name_or_path):
        raise SystemExit(
            "--output-root, --run-name and --model-name-or-path are all required "
            "so the run can be marked with its prompt contract afterwards"
        )
    if "qwen" in model_name_or_path.lower():
        raise SystemExit(
            f"refusing to install the chat formatter for {model_name_or_path!r}: "
            "Qwen runs must stay on the core trainer's plain MCQA formatter"
        )

    # Install before importing the trainer, mirroring chat_curriculum_entry.py.
    # data_gens.mcqa resolves format_mcqa_prompt as a module global at call time, so
    # patching the attribute is what takes effect inside MCQATokenDataset.
    install_chat_formatter()
    print("[CHAT] installed rebuttal HF chat-template MCQA formatter", flush=True)

    from applications.schedules import ordered_schedule_train

    sys.argv = [ordered_schedule_train.__file__, *trainer_args]
    ordered_schedule_train.main()

    marked = mark_chat_template_runs(output_root, run_name, model_name_or_path)
    if marked != 1:
        raise RuntimeError(
            f"training returned but expected exactly one matching run_summary.json; "
            f"marked={marked}, output_root={output_root!r}, run_name={run_name!r}, "
            f"model={model_name_or_path!r}"
        )
    print(f"[CHAT] marked {marked} run(s) with prompt_format=hf_chat_template", flush=True)


if __name__ == "__main__":
    main()
