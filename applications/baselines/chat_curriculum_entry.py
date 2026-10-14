"""Run curriculum_ft.py with the rebuttal-only chat formatter installed.

Pass the trainer path first, followed by its usual arguments::

    python applications/baselines/chat_curriculum_entry.py \
      applications/curriculum_ft.py --model-name-or-path google/gemma-3-1b-it ...

Exp1 uses this entrypoint for Llama/Gemma only, keeping every Qwen/default run
bit-for-bit on the core trainer's original plain MCQA formatter.
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


import runpy
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
    if len(sys.argv) < 2:
        raise SystemExit("usage: chat_curriculum_entry.py /path/to/curriculum_ft.py [args ...]")
    trainer_path = Path(sys.argv[1]).expanduser().resolve()
    trainer_args = sys.argv[2:]
    if not trainer_path.is_file():
        raise FileNotFoundError(trainer_path)

    output_root = _arg_value(trainer_args, "--output-root")
    run_name = _arg_value(trainer_args, "--run-name")
    model_name_or_path = _arg_value(trainer_args, "--model-name-or-path")
    install_chat_formatter()
    print("[CHAT] installed rebuttal HF chat-template MCQA formatter", flush=True)

    sys.argv = [str(trainer_path), *trainer_args]
    runpy.run_path(str(trainer_path), run_name="__main__")

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
