#!/usr/bin/env python3
# Exp4-local integrity wrapper around the shared zero-ablation trainer.

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
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

WRAPPER_PATH = Path(__file__).resolve()
EXP_DIR = WRAPPER_PATH.parent
REBUTTAL_DIR = EXP_DIR.parent
BASE_DIR = _REPO_ROOT
CORE_TRAINER = BASE_DIR / "EAP-IG" / "zero_ablation_finetune.py"
if str(REBUTTAL_DIR) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_DIR))

from selectivity.run_registry import resolve_current_circuit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".integrity.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def snapshot_summaries(root: Path) -> Dict[Path, Tuple[int, int]]:
    if not root.is_dir():
        return {}
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("run_summary.json")
    }


def wrapper_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="sciq")
    parser.add_argument("--score_key", required=True)
    parser.add_argument("--all_scores_json", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--circuits_root", required=True)
    parser.add_argument("--circuit_bucket", default="high")
    parser.add_argument("--corrupt_type", default="gaussian")
    parser.add_argument("--resume_state_dict", default="")
    parser.add_argument("--el2n_epochs", type=int, default=2)
    parser.add_argument("--disable_ablation", action="store_true")
    parser.add_argument("--ablation_edge_mode", default="scored")
    parser.add_argument("--random_edge_pool", default="circuit")
    args, _ = parser.parse_known_args(argv)
    if args.dataset != "sciq":
        raise ValueError(f"Exp4 integrity wrapper only supports dataset=sciq, got {args.dataset!r}")
    return args


def expected_circuit(args: argparse.Namespace) -> Optional[Path]:
    random_model = (
        args.ablation_edge_mode == "random" and args.random_edge_pool == "model"
    )
    if args.disable_ablation or random_model:
        return None
    return resolve_current_circuit(
        canonical(Path(args.circuits_root)),
        args.score_key,
        args.circuit_bucket,
        args.corrupt_type,
    )


def changed_summary(root: Path, before: Dict[Path, Tuple[int, int]]) -> Path:
    after = snapshot_summaries(root)
    changed = [path for path, version in after.items() if before.get(path) != version]
    if len(changed) != 1:
        raise RuntimeError(
            "trainer must create or update exactly one run_summary.json; "
            f"observed {len(changed)} changed summaries under {root}"
        )
    return changed[0]


def main(argv: Optional[Sequence[str]] = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    args = wrapper_args(forwarded)
    input_json = canonical(Path(args.all_scores_json))
    resume = canonical(Path(args.resume_state_dict)) if args.resume_state_dict else None
    circuit = expected_circuit(args)
    for label, path in (("input JSON", input_json), ("resume checkpoint", resume), ("circuit", circuit)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    core_trainer_hash = sha256_file(CORE_TRAINER)
    wrapper_hash = sha256_file(WRAPPER_PATH)
    input_hash = sha256_file(input_json)
    resume_hash = sha256_file(resume) if resume is not None else ""
    circuit_hash = sha256_file(circuit) if circuit is not None else ""
    out_root = canonical(Path(args.out_root))
    before = snapshot_summaries(out_root)

    completed = subprocess.run([sys.executable, "-u", str(CORE_TRAINER), *forwarded])
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    summary_path = changed_summary(out_root, before)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise RuntimeError(f"invalid summary object: {summary_path}")
    score_path = canonical(Path(str(summary.get("all_scores_json", ""))))
    checkpoint = canonical(Path(str(summary.get("final_checkpoint", ""))))
    if not score_path.is_file() or not checkpoint.is_file():
        raise RuntimeError("trainer summary is missing its score JSON or final checkpoint")
    with score_path.open("r", encoding="utf-8") as handle:
        score_obj = json.load(handle)
    if not isinstance(score_obj, dict) or not isinstance(score_obj.get("meta"), dict):
        raise RuntimeError(f"invalid score artifact: {score_path}")
    meta = dict(score_obj["meta"])

    recorded_input = canonical(Path(str(meta.get("input_all_scores_json", ""))))
    if recorded_input != input_json:
        raise RuntimeError(f"trainer used unexpected input JSON: {recorded_input}")
    recorded_circuit_text = str(meta.get("circuit_pt", ""))
    recorded_circuit = canonical(Path(recorded_circuit_text)) if recorded_circuit_text else None
    if recorded_circuit != circuit:
        raise RuntimeError(
            f"trainer used unexpected circuit: recorded={recorded_circuit}, expected={circuit}"
        )
    if sha256_file(input_json) != input_hash:
        raise RuntimeError(f"input JSON changed during training: {input_json}")
    if resume is not None and sha256_file(resume) != resume_hash:
        raise RuntimeError(f"resume checkpoint changed during training: {resume}")
    if circuit is not None and sha256_file(circuit) != circuit_hash:
        raise RuntimeError(f"circuit changed during training: {circuit}")
    if sha256_file(CORE_TRAINER) != core_trainer_hash:
        raise RuntimeError(f"core trainer changed during training: {CORE_TRAINER}")
    if sha256_file(WRAPPER_PATH) != wrapper_hash:
        raise RuntimeError(f"integrity wrapper changed during training: {WRAPPER_PATH}")

    checkpoint_hash = sha256_file(checkpoint)
    meta.update(
        {
            "input_all_scores_json_sha256": input_hash,
            "resume_state_dict_sha256": resume_hash,
            "el2n_epochs": int(args.el2n_epochs),
            "circuit_pt_sha256": circuit_hash,
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": checkpoint_hash,
            "core_trainer": str(CORE_TRAINER.resolve()),
            "core_trainer_sha256": core_trainer_hash,
            "integrity_wrapper": str(WRAPPER_PATH),
            "integrity_wrapper_sha256": wrapper_hash,
        }
    )
    score_obj["meta"] = meta
    atomic_write_json(score_path, score_obj)
    score_hash = sha256_file(score_path)

    summary.update(
        {
            "all_scores_json": str(score_path),
            "all_scores_json_sha256": score_hash,
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": checkpoint_hash,
            "meta": meta,
        }
    )
    atomic_write_json(summary_path, summary)
    print(f"[OK] Exp4 integrity contract attached: {summary_path}")


if __name__ == "__main__":
    main()
