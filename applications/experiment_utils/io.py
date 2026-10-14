"""Shared IO helpers for rebuttal experiments.

All rebuttal pool-builder scripts read/write JSONL rows that must be
consumable by curriculum_ft.py (see find_latest_generated_file /
match_generated_metadata). The most important requirement is that every
row we emit carries a 'target_band' string and a nested 'steering' dict
with keys {prompt_mode, gen_mode, circuit_target_mode} matching what the
trainer will look for.
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
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

DATA_ROOT = _REPO_ROOT / 'results'

# Canonical existing pool locations discovered during exploration
NAIVE_RAW_PATH = (
    DATA_ROOT
    / "circuit_gen_prev/Qwen__Qwen2.5-1.5B-Instruct/naive/fixed_both_discovered"
    / "20260408_115405/naive_raw_candidates.jsonl"
)
NAIVE_SELECTED_PATH = (
    DATA_ROOT
    / "circuit_gen_prev/Qwen__Qwen2.5-1.5B-Instruct/naive/fixed_both_discovered"
    / "20260408_115405/naive_selected.jsonl"
)
STEERED_ABLATE_ROOT = DATA_ROOT / "circuit_gen_ablate/Qwen__Qwen2.5-1.5B-Instruct"
STEERED_PREV_ROOT = DATA_ROOT / "circuit_gen_prev/Qwen__Qwen2.5-1.5B-Instruct"
CIRCUIT_CSV_ROOT = DATA_ROOT / "circuit_steering_csvs"
SCORES_JSON_DEFAULT = Path(
    str(_REPO_ROOT / 'checkpoints/static_full/sciq/grad_align/sciq_grad_align_full_high/all_scores_sciq.json')
)
GENERATOR_MODEL_KEY = "Qwen__Qwen2.5-1.5B-Instruct"


def read_jsonl(path: os.PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stream_jsonl(path: os.PathLike) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: os.PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: os.PathLike, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_mirror_dir(
    variant_root: Path,
    recipe: str,
    prompt_mode: str = "fixed",
    gen_mode: str = "both",
    circuit_target_mode: str = "discovered",
) -> Path:
    """Create a mirror directory tree that curriculum_ft.py can discover.

    Result layout:  variant_root/Qwen__Qwen2.5-1.5B-Instruct/<recipe>/
                    <prompt_mode>_<gen_mode>_<circuit_target_mode>/<run_ts>/
    Returns the leaf directory (where <recipe>_selected.jsonl etc. should live).
    """
    leaf = (
        variant_root
        / GENERATOR_MODEL_KEY
        / recipe
        / f"{prompt_mode}_{gen_mode}_{circuit_target_mode}"
        / stamp()
    )
    leaf.mkdir(parents=True, exist_ok=True)
    return leaf


def stamp_row_metadata(
    row: Mapping[str, Any],
    *,
    target_band: str,
    prompt_mode: str = "fixed",
    gen_mode: str = "both",
    circuit_target_mode: str = "discovered",
) -> Dict[str, Any]:
    """Force a row to have the metadata trainer expects.

    Preserves everything else in the row. Returns a shallow-copied dict.
    """
    out = dict(row)
    out["target_band"] = target_band
    steering = dict(out.get("steering") or {})
    steering["prompt_mode"] = prompt_mode
    steering["gen_mode"] = gen_mode
    steering["circuit_target_mode"] = circuit_target_mode
    out["steering"] = steering
    return out


def dump_pool_as_mirror(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant_root: Path,
    recipe: str,
    prompt_mode: str = "fixed",
    gen_mode: str = "both",
    circuit_target_mode: str = "discovered",
    file_kind: str = "selected",  # "selected" or "raw"
    force_stamp_metadata: bool = True,
) -> Path:
    """Write a pool of rows into a mirror dir the trainer can consume."""
    leaf = build_mirror_dir(
        variant_root=variant_root,
        recipe=recipe,
        prompt_mode=prompt_mode,
        gen_mode=gen_mode,
        circuit_target_mode=circuit_target_mode,
    )
    if force_stamp_metadata:
        stamped = [
            stamp_row_metadata(
                r,
                target_band=recipe,
                prompt_mode=prompt_mode,
                gen_mode=gen_mode,
                circuit_target_mode=circuit_target_mode,
            )
            for r in rows
        ]
    else:
        stamped = [dict(r) for r in rows]
    suffix = "raw_candidates" if file_kind == "raw" else "selected"
    out_path = leaf / f"{recipe}_{suffix}.jsonl"
    write_jsonl(out_path, stamped)
    summary = {
        "recipe": recipe,
        "prompt_mode": prompt_mode,
        "gen_mode": gen_mode,
        "circuit_target_mode": circuit_target_mode,
        "file_kind": file_kind,
        "n_rows": len(stamped),
        "written_at": stamp(),
        "path": str(out_path),
    }
    write_json(leaf / f"{recipe}_summary.json", summary)
    return out_path


def link_existing_pool(
    src_jsonl: Path,
    *,
    variant_root: Path,
    recipe: str,
    prompt_mode: str = "fixed",
    gen_mode: str = "both",
    circuit_target_mode: str = "discovered",
    file_kind: str = "selected",
) -> Path:
    """Symlink an existing pool jsonl into a mirror directory."""
    leaf = build_mirror_dir(
        variant_root=variant_root,
        recipe=recipe,
        prompt_mode=prompt_mode,
        gen_mode=gen_mode,
        circuit_target_mode=circuit_target_mode,
    )
    suffix = "raw_candidates" if file_kind == "raw" else "selected"
    dst = leaf / f"{recipe}_{suffix}.jsonl"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(Path(src_jsonl).resolve())
    return dst


def find_latest_steered_pool_dir(
    profile: str,
    *,
    root: Path = STEERED_PREV_ROOT,
    variant_subdir: str = "fixed_both_discovered",
) -> Path:
    """Return the newest {profile}/{variant_subdir}/<run_ts>/ directory.

    The default is the base (non-ablated) full-circuit root. Callers that need
    an ablation must opt in through ``find_ablated_steered_pool_dir`` or pass an
    explicit root; this prevents Exp1--3 from silently consuming Exp4 pools.
    """
    base = root / profile / variant_subdir
    if not base.is_dir():
        # Fall back to the "prev" root that holds the clean full-circuit pool.
        base = STEERED_PREV_ROOT / profile / variant_subdir
    if not base.is_dir():
        raise FileNotFoundError(base)
    # Prefer timestamp-shaped run dirs (YYYYMMDD_HHMMSS) over ablation subdirs.
    ts_dirs = sorted(p for p in base.iterdir() if p.is_dir() and any(ch.isdigit() for ch in p.name) and "_" in p.name and "no_" not in p.name)
    if ts_dirs:
        return ts_dirs[-1]
    # Otherwise use the first dir that is not clearly an ablation subdir.
    generic = sorted(p for p in base.iterdir() if p.is_dir() and "no_" not in p.name)
    if generic:
        return generic[-1]
    # Last resort: whatever exists.
    fallback = sorted(p for p in base.iterdir() if p.is_dir())
    if not fallback:
        raise FileNotFoundError(f"no run dirs under {base}")
    return fallback[-1]


def find_full_steered_pool_dir(profile: str) -> Path:
    """Explicit accessor for the CLEAN (non-ablated) full-circuit steered pool.
    Uses circuit_gen_prev/<profile>/fixed_both_discovered/<latest_ts>/.
    """
    base = STEERED_PREV_ROOT / profile / "fixed_both_discovered"
    if not base.is_dir():
        raise FileNotFoundError(base)
    candidates = sorted(p for p in base.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no run dirs under {base}")
    return candidates[-1]


def find_ablated_steered_pool_dir(profile: str, ablation: str) -> Path:
    """Accessor for the per-metric-ablated pool used by Exp 4.

    ablation ∈ {'no_aum', 'no_el2n', 'no_gradalign'}.
    """
    d = STEERED_ABLATE_ROOT / profile / "fixed_both_discovered" / f"{profile}_{ablation}"
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d
