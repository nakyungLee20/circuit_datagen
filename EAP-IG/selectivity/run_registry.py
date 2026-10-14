"""Discover timestamped Exp4 runs and maintain an atomic manifest."""

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
from functools import lru_cache
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


EXP_DIR = Path(__file__).resolve().parent
CORE_TRAINER = _REPO_ROOT / "EAP-IG" / "zero_ablation_finetune.py"
INTEGRITY_WRAPPER = EXP_DIR / "trajectory_with_integrity.py"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return obj if isinstance(obj, dict) else {}


def canonical_path(value: object) -> str:
    """Return one stable spelling for an input/artifact path."""
    return str(Path(str(value)).expanduser().resolve())


@lru_cache(maxsize=64)
def _file_sha256_versioned(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path_text: str) -> str:
    path = Path(path_text).resolve()
    stat = path.stat()
    return _file_sha256_versioned(str(path), stat.st_size, stat.st_mtime_ns)


RUN_RE = re.compile(r"_m(?P<m>\d+)_top(?P<top>\d+)_")

def _score_pt_candidate(path: Path) -> Tuple[int, int, float]:
    match = RUN_RE.search(path.name)
    if match:
        ig_steps, topn = int(match.group("m")), int(match.group("top"))
    else:
        ig_steps = topn = -1
    return topn, ig_steps, path.stat().st_mtime

def resolve_current_circuit(
    circuits_root: Path, score_key: str, bucket: str, corrupt_type: str
) -> Path:
    group = "data_attr" if score_key in {"grad_align", "tracin"} else "train_dynamic"
    hints = ("train_dynamic", "data_attr")
    if circuits_root.name in hints:
        roots = [circuits_root] if circuits_root.is_dir() else []
    else:
        roots = [circuits_root / hint for hint in hints if (circuits_root / hint).is_dir()]
        if not roots and circuits_root.is_dir():
            roots = [circuits_root]
    candidates: List[Path] = []
    for root in roots:
        search = (
            [root / "sciq" / score_key / bucket / corrupt_type]
            if root.name in hints
            else [
                root / group / "sciq" / score_key / bucket / corrupt_type,
                root / "sciq" / score_key / bucket / corrupt_type,
            ]
        )
        for directory in search:
            if not directory.is_dir():
                continue
            candidates.extend(
                list(directory.glob("eapig_*.pt")) or list(directory.glob("*.pt"))
            )
    if not candidates:
        raise FileNotFoundError(
            f"no circuit for sciq/{score_key}/{bucket}/{corrupt_type} under {circuits_root}"
        )
    return sorted(set(candidates), key=_score_pt_candidate, reverse=True)[0].resolve()


def same_float(actual: object, expected: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, float(expected), rel_tol=1e-12, abs_tol=1e-15
    )


def config_matches(
    meta: Dict[str, Any],
    *,
    score_key: str,
    run_kind: str,
    topk: int,
    seed: int,
    input_json: Path,
    model_name: str,
    dtype: str,
    device: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    max_samples: int,
    max_length: int,
    resume_state_dict: str,
    el2n_epochs: int,
    random_edges_seed: int,
    circuit_bucket: str,
    corrupt_type: str,
) -> bool:
    """Match every output-affecting option recorded by the trainer."""
    exact = {
        "dataset": "sciq",
        "score_key": score_key,
        "model_name": model_name,
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "el2n_epochs": el2n_epochs,
        "max_samples": max_samples,
        "max_length": max_length,
        "topk_edges": topk,
        "topk_rank_abs": True,
    }
    if any(meta.get(key) != value for key, value in exact.items()):
        return False
    source = meta.get("input_all_scores_json")
    if not source or canonical_path(source) != canonical_path(input_json):
        return False
    input_path = Path(input_json).expanduser().resolve()
    if (
        not input_path.is_file()
        or meta.get("input_all_scores_json_sha256") != file_sha256(str(input_path))
    ):
        return False
    stored_resume = str(meta.get("resume_state_dict", ""))
    if bool(stored_resume) != bool(resume_state_dict):
        return False
    if stored_resume and canonical_path(stored_resume) != canonical_path(resume_state_dict):
        return False
    expected_resume_hash = (
        file_sha256(canonical_path(resume_state_dict)) if resume_state_dict else ""
    )
    if str(meta.get("resume_state_dict_sha256", "")) != expected_resume_hash:
        return False
    for path_key, hash_key, expected_path in (
        ("core_trainer", "core_trainer_sha256", CORE_TRAINER),
        ("integrity_wrapper", "integrity_wrapper_sha256", INTEGRITY_WRAPPER),
    ):
        if not expected_path.is_file():
            return False
        stored_path = str(meta.get(path_key, ""))
        if not stored_path or canonical_path(stored_path) != canonical_path(expected_path):
            return False
        if str(meta.get(hash_key, "")) != file_sha256(str(expected_path)):
            return False

    if not same_float(meta.get("lr"), lr):
        return False
    if not same_float(meta.get("weight_decay"), weight_decay):
        return False

    if run_kind == "full":
        return meta.get("ablation_enabled") is False
    if run_kind == "scored":
        return (
            meta.get("ablation_enabled") is True
            and str(meta.get("ablation_edge_mode")) == "scored"
            and str(meta.get("circuit_bucket")) == circuit_bucket
            and str(meta.get("corrupt_type")) == corrupt_type
        )
    if run_kind == "random_model":
        try:
            stored_random_seed = int(meta.get("random_edges_seed", -1))
        except (TypeError, ValueError):
            return False
        return (
            meta.get("ablation_enabled") is True
            and str(meta.get("ablation_edge_mode")) == "random"
            and str(meta.get("random_edge_pool")) == "model"
            and stored_random_seed == random_edges_seed
        )
    raise ValueError(run_kind)


def candidate_from_summary(
    summary_path: Path,
    *,
    score_key: str,
    run_kind: str,
    topk: int,
    seed: int,
    expected_samples: int,
    input_json: Path,
    model_name: str,
    dtype: str,
    device: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    max_samples: int,
    max_length: int,
    resume_state_dict: str,
    el2n_epochs: int,
    random_edges_seed: int,
    circuit_bucket: str,
    corrupt_type: str,
    current_circuit: Optional[Path],
) -> Optional[Dict[str, Any]]:
    summary = read_json(summary_path)
    meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
    config = {
        "score_key": score_key,
        "run_kind": run_kind,
        "topk": topk,
        "seed": seed,
        "input_json": input_json,
        "model_name": model_name,
        "dtype": dtype,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_samples": max_samples,
        "max_length": max_length,
        "resume_state_dict": resume_state_dict,
        "el2n_epochs": el2n_epochs,
        "random_edges_seed": random_edges_seed,
        "circuit_bucket": circuit_bucket,
        "corrupt_type": corrupt_type,
    }
    if not config_matches(meta, **config):
        return None

    run_dir = Path(summary.get("run_dir") or summary_path.parent)
    score_json = Path(summary.get("all_scores_json") or run_dir / "all_scores_sciq.json")
    checkpoint = Path(summary.get("final_checkpoint") or run_dir / "checkpoints" / "final.pt")
    if not score_json.is_file() or not checkpoint.is_file():
        return None
    score_json_hash = file_sha256(str(score_json.resolve()))
    checkpoint_hash = file_sha256(str(checkpoint.resolve()))
    if summary.get("all_scores_json_sha256") != score_json_hash:
        return None
    if summary.get("final_checkpoint_sha256") != checkpoint_hash:
        return None
    score_obj = read_json(score_json)
    score_meta = score_obj.get("meta") if isinstance(score_obj.get("meta"), dict) else {}
    if not config_matches(score_meta, **config):
        return None
    score_items = score_obj.get("items")
    expected_rows = (
        min(expected_samples, max_samples) if max_samples > 0 else expected_samples
    )
    if not isinstance(score_items, list) or len(score_items) != expected_rows:
        return None
    try:
        rows_out = int(score_meta.get("num_rows_out", -1))
    except (TypeError, ValueError):
        rows_out = -1
    if rows_out != expected_rows:
        return None
    if score_meta.get("final_checkpoint_sha256") != checkpoint_hash:
        return None
    for key in ("circuit_pt", "edge_meta"):
        if score_meta.get(key) != meta.get(key):
            return None
    circuit_text = str(meta.get("circuit_pt", ""))
    if run_kind == "scored":
        if not circuit_text or not Path(circuit_text).is_file() or current_circuit is None:
            return None
        if canonical_path(circuit_text) != str(current_circuit.resolve()):
            return None
        circuit_hash = file_sha256(str(current_circuit.resolve()))
        if score_meta.get("circuit_pt_sha256") != circuit_hash:
            return None
        edge_meta = meta.get("edge_meta")
        if not isinstance(edge_meta, dict):
            return None
        try:
            mapped_edges = int(edge_meta.get("mapped_edges_in_runtime", -1))
        except (TypeError, ValueError):
            return None
        if mapped_edges != topk:
            return None
    elif run_kind == "random_model":
        edge_meta = meta.get("edge_meta")
        if not isinstance(edge_meta, dict):
            return None
        try:
            mapped_edges = int(edge_meta.get("mapped_edges_in_runtime", -1))
        except (TypeError, ValueError):
            return None
        if mapped_edges != topk:
            return None
    return {
        "run_dir": str(run_dir.resolve()),
        "summary": str(summary_path.resolve()),
        "all_scores_json": str(score_json.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "circuit_pt": canonical_path(circuit_text) if circuit_text else "",
        "score_key": score_key,
        "run_kind": run_kind,
        "topk_edges": topk,
        "seed": seed,
        "expected_samples": expected_samples,
        "input_json": canonical_path(input_json),
        "input_json_sha256": file_sha256(canonical_path(input_json)),
        "all_scores_json_sha256": score_json_hash,
        "checkpoint_sha256": checkpoint_hash,
        "circuit_pt_sha256": (
            file_sha256(canonical_path(circuit_text)) if circuit_text else ""
        ),
        "config": {
            "model_name": model_name,
            "dtype": dtype,
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_samples": max_samples,
            "max_length": max_length,
            "resume_state_dict": canonical_path(resume_state_dict) if resume_state_dict else "",
            "el2n_epochs": el2n_epochs,
            "random_edges_seed": random_edges_seed,
            "circuit_bucket": circuit_bucket,
            "corrupt_type": corrupt_type,
            "topk_rank_abs": True,
        },
        "mtime": summary_path.stat().st_mtime,
    }


def discover(
    root: Path,
    *,
    score_key: str,
    run_kind: str,
    topk: int,
    seed: int,
    expected_samples: int,
    input_json: Path,
    model_name: str,
    dtype: str,
    device: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    max_samples: int,
    max_length: int,
    resume_state_dict: str,
    el2n_epochs: int,
    random_edges_seed: int,
    circuit_bucket: str,
    corrupt_type: str,
    circuits_root: Path,
) -> Dict[str, Any]:
    current_circuit = (
        resolve_current_circuit(circuits_root, score_key, circuit_bucket, corrupt_type)
        if run_kind == "scored"
        else None
    )
    candidates: List[Dict[str, Any]] = []
    if root.is_dir():
        for summary_path in root.rglob("run_summary.json"):
            candidate = candidate_from_summary(
                summary_path,
                score_key=score_key,
                run_kind=run_kind,
                topk=topk,
                seed=seed,
                expected_samples=expected_samples,
                input_json=input_json,
                model_name=model_name,
                dtype=dtype,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                lr=lr,
                weight_decay=weight_decay,
                max_samples=max_samples,
                max_length=max_length,
                resume_state_dict=resume_state_dict,
                el2n_epochs=el2n_epochs,
                random_edges_seed=random_edges_seed,
                circuit_bucket=circuit_bucket,
                corrupt_type=corrupt_type,
                current_circuit=current_circuit,
            )
            if candidate:
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"no complete run under {root} matching score_key={score_key}, "
            f"run_kind={run_kind}, topk={topk}, seed={seed}, input={input_json}, "
            "and the complete requested training configuration"
        )
    candidates.sort(key=lambda item: (float(item["mtime"]), item["run_dir"]), reverse=True)
    return candidates[0]


def update_manifest(path: Path, label: str, record: Dict[str, Any]) -> None:
    manifest: Dict[str, Any] = {}
    if path.is_file():
        manifest = read_json(path)
    runs = manifest.get("runs") if isinstance(manifest.get("runs"), dict) else {}
    runs[label] = record
    manifest["schema_version"] = 2
    manifest["runs"] = runs
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--root", required=True)
    register.add_argument("--manifest", required=True)
    register.add_argument("--label", required=True)
    register.add_argument("--score-key", required=True)
    register.add_argument("--run-kind", choices=("full", "scored", "random_model"), required=True)
    register.add_argument("--topk", type=int, required=True)
    register.add_argument("--seed", type=int, required=True)
    register.add_argument("--expected-samples", type=int, required=True)
    register.add_argument("--input-json", required=True)
    register.add_argument("--model-name", required=True)
    register.add_argument("--dtype", required=True)
    register.add_argument("--device", required=True)
    register.add_argument("--epochs", type=int, required=True)
    register.add_argument("--batch-size", type=int, required=True)
    register.add_argument("--grad-accum", type=int, required=True)
    register.add_argument("--lr", type=float, required=True)
    register.add_argument("--weight-decay", type=float, required=True)
    register.add_argument("--max-samples", type=int, required=True)
    register.add_argument("--max-length", type=int, required=True)
    register.add_argument("--resume-state-dict", default="")
    register.add_argument("--el2n-epochs", type=int, required=True)
    register.add_argument("--circuits-root", required=True)
    register.add_argument("--random-edges-seed", type=int, required=True)
    register.add_argument("--circuit-bucket", default="high")
    register.add_argument("--corrupt-type", default="gaussian")
    resolve = subparsers.add_parser("resolve-circuit")
    resolve.add_argument("--circuits-root", required=True)
    resolve.add_argument("--score-key", required=True)
    resolve.add_argument("--bucket", choices=("high", "low"), required=True)
    resolve.add_argument("--corrupt-type", default="gaussian")
    resolve.add_argument("--edge-jsonl", action="store_true")
    get = subparsers.add_parser("get")
    get.add_argument("--manifest", required=True)
    get.add_argument("--label", required=True)
    get.add_argument(
        "--field",
        choices=("run_dir", "summary", "all_scores_json", "checkpoint", "circuit_pt"),
        required=True,
    )
    subparsers.add_parser("show").add_argument("--manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "register":
        record = discover(
            Path(args.root),
            score_key=args.score_key,
            run_kind=args.run_kind,
            topk=args.topk,
            seed=args.seed,
            expected_samples=args.expected_samples,
            input_json=Path(args.input_json),
            model_name=args.model_name,
            dtype=args.dtype,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_samples=args.max_samples,
            max_length=args.max_length,
            resume_state_dict=args.resume_state_dict,
            el2n_epochs=args.el2n_epochs,
            random_edges_seed=args.random_edges_seed,
            circuit_bucket=args.circuit_bucket,
            corrupt_type=args.corrupt_type,
            circuits_root=Path(args.circuits_root),
        )
        update_manifest(Path(args.manifest), args.label, record)
        print(record["run_dir"])
        return
    if args.command == "resolve-circuit":
        circuit = resolve_current_circuit(
            Path(args.circuits_root), args.score_key, args.bucket, args.corrupt_type
        )
        print(circuit.with_suffix(".edge_scores.jsonl") if args.edge_jsonl else circuit)
        return
    if args.command == "get":
        manifest = read_json(Path(args.manifest))
        runs = manifest.get("runs") if isinstance(manifest.get("runs"), dict) else {}
        record = runs.get(args.label)
        if not isinstance(record, dict) or not record.get(args.field):
            raise KeyError(f"manifest has no {args.label}.{args.field}")
        print(record[args.field])
        return
    print(json.dumps(read_json(Path(args.manifest)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
