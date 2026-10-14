"""Rebuttal-only entrypoint for a fully randomized circuit control.

The core generator's ``circuit_target_mode=random`` preserves target counts and
randomizes layer/head locations, but its ``VectorBankBuilder`` still derives
activation directions from metric high-minus-low calibration groups.  This
adapter keeps the core location randomizer and replaces each metric/module
high vector with a seeded Gaussian direction of the same L2 norm. Its paired
low vector remains the exact negative, preserving the core bank's matched
recipe-composition invariant.

Invoke it with the public generator and its usual CLI arguments::

    python applications/baselines/fully_random_circuit_entry.py \
      applications/circuit_generation.py --circuit-target-mode random \
      --use-activation-addition --help

The helper is recipe-agnostic.  Callers choose ``--output-dir`` in the normal
trainer-discoverable layout.  Every raw/selected data row and a sidecar manifest
carry a strict provenance contract so legacy location-only controls cannot be
silently reused.
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


import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "applications") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "applications"))

from applications import circuit_generation as generation


RANDOMIZATION_CONTRACT = "positions_heads_and_activation_directions_v1"
RANDOM_DIRECTION_ALGORITHM = "seeded_gaussian_antithetic_same_l2_norm_v1"
RANDOM_POSITION_ALGORITHM = "core_build_randomized_presets_v1"
SIDECAR_SUFFIX = "_fully_random_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_input_fingerprints(
    edge_table: Optional[str], edge_table_dir: Optional[str]
) -> list[Dict[str, str]]:
    """Fingerprint the deterministic file set consumed by the core loader."""
    paths: list[Path]
    if edge_table:
        paths = [Path(edge_table).expanduser().resolve()]
    elif edge_table_dir:
        directory = Path(edge_table_dir).expanduser().resolve()
        paths = [
            path.resolve()
            for path in sorted(directory.glob("*_top_edges.csv"))
            if generation._CIRCUIT_FILE_RE.search(path.name) is not None
        ]
    else:
        paths = []
    if not paths:
        raise RuntimeError("fully-random control requires a non-empty edge-table input set")
    fingerprints = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        fingerprints.append({"path": str(path), "sha256": _sha256_file(path)})
    return fingerprints


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _derived_seed(base_seed: int, circuit_key: str, module_key: str) -> int:
    payload = (
        f"{RANDOM_DIRECTION_ALGORITHM}|{int(base_seed)}|{circuit_key}|{module_key}"
    ).encode("utf-8")
    # torch.Generator.manual_seed accepts a signed 64-bit-range integer.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _random_same_norm(
    reference: torch.Tensor, *, seed: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    source = reference.detach().float()
    source_norm = float(torch.linalg.vector_norm(source).item())
    if not math.isfinite(source_norm) or source_norm <= 0.0:
        raise RuntimeError(
            f"cannot construct a norm-matched random direction from norm={source_norm}"
        )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    random_cpu = torch.randn(
        tuple(reference.shape), generator=rng, dtype=torch.float32, device="cpu"
    )
    random_norm = float(torch.linalg.vector_norm(random_cpu).item())
    if not math.isfinite(random_norm) or random_norm <= 0.0:
        raise RuntimeError("seeded Gaussian direction has a non-finite/zero norm")
    random_cpu.mul_(source_norm / random_norm)
    randomized = random_cpu.to(device=reference.device, dtype=reference.dtype)

    # Casting to bf16/fp16 introduces tiny norm error; one correction in fp32
    # keeps magnitude matching tight while retaining the generator's dtype.
    cast_norm = float(torch.linalg.vector_norm(randomized.detach().float()).item())
    if not math.isfinite(cast_norm) or cast_norm <= 0.0:
        raise RuntimeError("random direction became non-finite/zero after dtype cast")
    randomized = (
        randomized.detach().float().mul_(source_norm / cast_norm)
        .to(device=reference.device, dtype=reference.dtype)
    )
    final = randomized.detach().float()
    final_norm = float(torch.linalg.vector_norm(final).item())
    relative_norm_error = abs(final_norm - source_norm) / max(source_norm, 1e-12)
    if relative_norm_error > 5e-3:
        raise RuntimeError(
            "random direction failed norm matching: "
            f"source={source_norm}, random={final_norm}, relerr={relative_norm_error}"
        )

    cosine = float(
        torch.dot(source.reshape(-1), final.reshape(-1)).item()
        / max(source_norm * final_norm, 1e-12)
    )
    source_sha = _tensor_sha256(reference)
    random_sha = _tensor_sha256(randomized)
    if source_sha == random_sha:
        raise RuntimeError("random direction is byte-identical to the metric direction")
    return randomized, {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "source_norm": source_norm,
        "random_norm": final_norm,
        "relative_norm_error": relative_norm_error,
        "source_vs_random_cosine": cosine,
        "source_sha256": source_sha,
        "random_sha256": random_sha,
        "derived_seed": int(seed),
    }


def randomize_vector_bank(
    vector_bank: Mapping[str, Mapping[str, torch.Tensor]], *, base_seed: int
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], list[Dict[str, Any]]]:
    """Return same-norm random orientations and an auditable tensor manifest.

    The core bank defines every metric-low vector as the exact negative of its
    metric-high vector.  We draw one random orientation per metric/module and
    preserve that antithetic pair exactly; independently randomizing high/low
    would change recipe-composition norms and cease to be a matched control.
    """
    if not vector_bank:
        raise RuntimeError("core VectorBankBuilder produced an empty vector bank")
    randomized: Dict[str, Dict[str, torch.Tensor]] = {}
    audit: list[Dict[str, Any]] = []
    high_keys = sorted(key for key in vector_bank if key.endswith("_high"))
    expected_keys = {
        paired_key
        for high_key in high_keys
        for paired_key in (high_key, high_key[: -len("_high")] + "_low")
    }
    if set(vector_bank) != expected_keys:
        raise RuntimeError(
            "vector bank must contain complete metric high/low pairs; "
            f"found={sorted(vector_bank)}, expected={sorted(expected_keys)}"
        )
    for high_key in high_keys:
        low_key = high_key[: -len("_high")] + "_low"
        high_map = vector_bank[high_key]
        low_map = vector_bank[low_key]
        if not high_map or set(high_map) != set(low_map):
            raise RuntimeError(
                f"mismatched/empty vector modules for {high_key}/{low_key}"
            )
        randomized[high_key] = {}
        randomized[low_key] = {}
        for module_key in sorted(high_map):
            high_source = high_map[module_key]
            low_source = low_map[module_key]
            source_pair_error = float(
                (low_source.detach().float() + high_source.detach().float())
                .abs()
                .max()
                .item()
            )
            source_scale = max(
                float(high_source.detach().float().abs().max().item()), 1e-12
            )
            if source_pair_error / source_scale > 5e-3:
                raise RuntimeError(
                    f"core vector pair is not antithetic for "
                    f"{high_key}/{low_key}/{module_key}: error={source_pair_error}"
                )
            seed = _derived_seed(base_seed, high_key, module_key)
            high_random, details = _random_same_norm(high_source, seed=seed)
            low_random = -high_random
            randomized[high_key][module_key] = high_random
            randomized[low_key][module_key] = low_random
            random_pair_error = float(
                (low_random.detach().float() + high_random.detach().float())
                .abs()
                .max()
                .item()
            )
            low_source_norm = float(
                torch.linalg.vector_norm(low_source.detach().float()).item()
            )
            low_random_norm = float(
                torch.linalg.vector_norm(low_random.detach().float()).item()
            )
            low_relative_norm_error = abs(low_random_norm - low_source_norm) / max(
                low_source_norm, 1e-12
            )
            if random_pair_error != 0.0 or low_relative_norm_error > 5e-3:
                raise RuntimeError(
                    f"random pair/norm invariant failed for "
                    f"{high_key}/{low_key}/{module_key}"
                )
            audit.append(
                {
                    "high_circuit_key": high_key,
                    "low_circuit_key": low_key,
                    "module_key": module_key,
                    "source_pair_antisymmetry_max_abs": source_pair_error,
                    "random_pair_antisymmetry_max_abs": random_pair_error,
                    "low_source_norm": low_source_norm,
                    "low_random_norm": low_random_norm,
                    "low_relative_norm_error": low_relative_norm_error,
                    "low_random_sha256": _tensor_sha256(low_random),
                    **details,
                }
            )
    return randomized, audit


def _stamp_row(
    row: Mapping[str, Any], *, recipe: str, direction_seed: int
) -> Dict[str, Any]:
    stamped = dict(row)
    steering_value = row.get("steering")
    if not isinstance(steering_value, Mapping):
        return stamped
    steering = dict(steering_value)
    if str(steering.get("circuit_target_mode") or "") != "random":
        raise RuntimeError("fully-random adapter received a non-random generated row")
    steering.update(
        {
            "randomization_contract": RANDOMIZATION_CONTRACT,
            "random_position_algorithm": RANDOM_POSITION_ALGORITHM,
            "random_direction_algorithm": RANDOM_DIRECTION_ALGORITHM,
            "random_direction_seed": int(direction_seed),
            "metric_direction_orientation_used": False,
        }
    )
    stamped["steering"] = steering
    if str(stamped.get("target_band") or "") != str(recipe):
        raise RuntimeError(
            f"generated row target_band={stamped.get('target_band')!r} "
            f"does not match recipe={recipe!r}"
        )
    return stamped


def validate_fully_random_row(
    row: Mapping[str, Any], *, expected_recipe: Optional[str] = None
) -> None:
    """Raise if a generated data row lacks the full-randomization contract."""
    steering = row.get("steering")
    if not isinstance(steering, Mapping):
        raise ValueError("row has no steering mapping")
    expected = {
        "circuit_target_mode": "random",
        "randomization_contract": RANDOMIZATION_CONTRACT,
        "random_position_algorithm": RANDOM_POSITION_ALGORITHM,
        "random_direction_algorithm": RANDOM_DIRECTION_ALGORITHM,
        "metric_direction_orientation_used": False,
    }
    for key, value in expected.items():
        if steering.get(key) != value:
            raise ValueError(
                f"row steering.{key}={steering.get(key)!r}; expected {value!r}"
            )
    seed = steering.get("random_direction_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"invalid steering.random_direction_seed={seed!r}")
    if expected_recipe is not None and str(row.get("target_band") or "") != str(
        expected_recipe
    ):
        raise ValueError(
            f"row target_band={row.get('target_band')!r}; "
            f"expected {expected_recipe!r}"
        )


def _validate_jsonl_rows(
    path: Path, *, recipe: str, expected_seed: Optional[int] = None
) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                validate_fully_random_row(row, expected_recipe=recipe)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if expected_seed is not None:
                actual_seed = row["steering"]["random_direction_seed"]
                if actual_seed != int(expected_seed):
                    raise ValueError(
                        f"{path}:{line_no}: random_direction_seed={actual_seed}; "
                        f"expected {expected_seed}"
                    )
            count += 1
    return count


def validate_fully_random_output_dir(
    path: str | Path,
    *,
    recipe: str,
    expected_seed_data_num: int = 5000,
    expected_num_candidates: int = 4,
    expected_keep_per_record: int = 2,
    expected_topk_edges: int = 250,
) -> Dict[str, Any]:
    """Validate sidecar, budget, output counts, and every raw/selected row."""
    output_dir = Path(path).expanduser().resolve()
    sidecar = output_dir / f"{recipe}{SIDECAR_SUFFIX}"
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    required_manifest = {
        "randomization_contract": RANDOMIZATION_CONTRACT,
        "random_position_algorithm": RANDOM_POSITION_ALGORITHM,
        "random_direction_algorithm": RANDOM_DIRECTION_ALGORITHM,
        "recipe": recipe,
        "prompt_mode": "fixed",
        "gen_mode": "both",
        "circuit_target_mode": "random",
        "metric_direction_orientation_used": False,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{sidecar}: {key}={manifest.get(key)!r}; expected {expected!r}"
            )
    expected_budget = {
        "seed_data_num": int(expected_seed_data_num),
        "num_candidates": int(expected_num_candidates),
        "keep_per_record": int(expected_keep_per_record),
        "topk_edges": int(expected_topk_edges),
    }
    budget = manifest.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError(f"{sidecar}: missing budget mapping")
    for key, expected in expected_budget.items():
        if int(budget.get(key, -1)) != expected:
            raise ValueError(
                f"{sidecar}: budget.{key}={budget.get(key)!r}; expected {expected}"
            )

    direction_seed = manifest.get("random_direction_seed")
    if isinstance(direction_seed, bool) or not isinstance(direction_seed, int):
        raise ValueError(f"{sidecar}: invalid random_direction_seed")
    raw_path = output_dir / f"{recipe}_raw_candidates.jsonl"
    selected_path = output_dir / f"{recipe}_selected.jsonl"
    if not raw_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            f"missing raw/selected output under {output_dir} for recipe={recipe}"
        )
    raw_count = _validate_jsonl_rows(
        raw_path, recipe=recipe, expected_seed=direction_seed
    )
    selected_count = _validate_jsonl_rows(
        selected_path, recipe=recipe, expected_seed=direction_seed
    )
    expected_raw = expected_seed_data_num * expected_num_candidates
    expected_selected = expected_seed_data_num * expected_keep_per_record
    if raw_count != expected_raw or selected_count != expected_selected:
        raise ValueError(
            f"{output_dir}: counts raw={raw_count}/{expected_raw}, "
            f"selected={selected_count}/{expected_selected}"
        )
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    if int(counts.get("raw", -1)) != raw_count or int(
        counts.get("selected", -1)
    ) != selected_count:
        raise ValueError(f"{sidecar}: manifest/output count mismatch")
    if not manifest.get("vector_audit"):
        raise ValueError(f"{sidecar}: empty vector_audit")
    for audit_idx, item in enumerate(manifest["vector_audit"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"{sidecar}: vector_audit[{audit_idx}] is not a mapping")
        source_norm = float(item.get("source_norm", float("nan")))
        source_pair_error = float(
            item.get("source_pair_antisymmetry_max_abs", float("nan"))
        )
        invariants = {
            "relative_norm_error": float(
                item.get("relative_norm_error", float("nan"))
            ),
            "low_relative_norm_error": float(
                item.get("low_relative_norm_error", float("nan"))
            ),
            "random_pair_antisymmetry_max_abs": float(
                item.get("random_pair_antisymmetry_max_abs", float("nan"))
            ),
        }
        if not math.isfinite(source_norm) or source_norm <= 0:
            raise ValueError(f"{sidecar}: vector_audit[{audit_idx}] invalid source norm")
        if (
            not math.isfinite(source_pair_error)
            or source_pair_error > max(1e-6, source_norm * 5e-3)
        ):
            raise ValueError(
                f"{sidecar}: vector_audit[{audit_idx}] source pair is not antithetic"
            )
        if (
            not math.isfinite(invariants["relative_norm_error"])
            or invariants["relative_norm_error"] > 5e-3
            or not math.isfinite(invariants["low_relative_norm_error"])
            or invariants["low_relative_norm_error"] > 5e-3
            or invariants["random_pair_antisymmetry_max_abs"] != 0.0
        ):
            raise ValueError(
                f"{sidecar}: vector_audit[{audit_idx}] norm/anti-symmetry invariant failed"
            )
        if item.get("source_sha256") == item.get("random_sha256"):
            raise ValueError(
                f"{sidecar}: vector_audit[{audit_idx}] retained the metric direction"
            )

    fingerprint_specs = [
        (manifest.get("input_jsonl"), manifest.get("input_jsonl_sha256")),
        (
            manifest.get("calibration_jsonl"),
            manifest.get("calibration_jsonl_sha256"),
        ),
        (manifest.get("generator_path"), manifest.get("generator_sha256")),
        (manifest.get("adapter_path"), manifest.get("adapter_sha256")),
    ]
    edge_files = manifest.get("edge_table_files")
    if not isinstance(edge_files, list) or not edge_files:
        raise ValueError(f"{sidecar}: missing edge_table_files fingerprints")
    current_edge_files = _edge_input_fingerprints(
        manifest.get("edge_table"), manifest.get("edge_table_dir")
    )
    if current_edge_files != edge_files:
        raise ValueError(
            f"{sidecar}: recognized edge-table file set/content changed"
        )
    for item in edge_files:
        if not isinstance(item, Mapping):
            raise ValueError(f"{sidecar}: malformed edge_table_files entry")
        fingerprint_specs.append((item.get("path"), item.get("sha256")))
    for file_path, expected_sha in fingerprint_specs:
        if not isinstance(file_path, str) or not isinstance(expected_sha, str):
            raise ValueError(f"{sidecar}: incomplete content fingerprint")
        actual_path = Path(file_path).expanduser().resolve()
        if not actual_path.is_file():
            raise FileNotFoundError(actual_path)
        actual_sha = _sha256_file(actual_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{sidecar}: content hash changed for {actual_path}: "
                f"found={actual_sha}, expected={expected_sha}"
            )
    if Path(str(manifest.get("generator_path"))).expanduser().resolve() != Path(
        generation.__file__
    ).resolve():
        raise ValueError(f"{sidecar}: generator_path is not the expected core generator")
    if Path(str(manifest.get("adapter_path"))).expanduser().resolve() != Path(
        __file__
    ).resolve():
        raise ValueError(f"{sidecar}: adapter_path is not this common helper")
    return manifest


def _validate_existing_rows(
    output_dir: Path, *, recipe: str, expected_seed: int
) -> None:
    """Allow only resumptions produced by this exact contract and seed."""
    for suffix in ("raw_candidates", "selected"):
        path = output_dir / f"{recipe}_{suffix}.jsonl"
        if path.exists():
            _validate_jsonl_rows(
                path, recipe=recipe, expected_seed=expected_seed
            )


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: fully_random_circuit_entry.py "
            "/path/to/circuit_generation.py [args ...]"
        )
    generator_path = Path(sys.argv[1]).expanduser().resolve()
    expected_generator = Path(generation.__file__).resolve()
    if generator_path != expected_generator:
        raise RuntimeError(
            f"unexpected generator path: {generator_path}; expected {expected_generator}"
        )
    generator_args = sys.argv[2:]
    args = generation.build_arg_parser().parse_args(generator_args)
    if args.circuit_target_mode != "random":
        raise RuntimeError("fully-random adapter requires --circuit-target-mode=random")
    if not args.use_activation_addition:
        raise RuntimeError(
            "fully-random direction control requires --use-activation-addition"
        )
    if not args.calibration_jsonl:
        raise RuntimeError(
            "fully-random direction control requires --calibration-jsonl"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    direction_seed = int(args.seed)
    input_path = Path(args.input_jsonl).expanduser().resolve()
    calibration_path = Path(args.calibration_jsonl).expanduser().resolve()
    generator_sha256 = _sha256_file(generator_path)
    adapter_path = Path(__file__).resolve()
    adapter_sha256 = _sha256_file(adapter_path)
    edge_table_files = _edge_input_fingerprints(
        args.edge_table, args.edge_table_dir
    )
    immutable_input_identity = {
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": _sha256_file(input_path),
        "calibration_jsonl": str(calibration_path),
        "calibration_jsonl_sha256": _sha256_file(calibration_path),
        "edge_table_files": edge_table_files,
        "generator_sha256": generator_sha256,
        "adapter_sha256": adapter_sha256,
    }
    manifest_path = output_dir / f"{args.recipe}{SIDECAR_SUFFIX}"
    if manifest_path.is_file():
        manifest = validate_fully_random_output_dir(
            output_dir,
            recipe=args.recipe,
            expected_seed_data_num=int(args.seed_data_num),
            expected_num_candidates=int(args.num_candidates),
            expected_keep_per_record=int(args.keep_per_record),
            expected_topk_edges=int(args.topk_edges),
        )
        expected_identity = {
            "random_direction_seed": direction_seed,
            "model_name_or_path": args.model_name_or_path,
            "edge_table": str(Path(args.edge_table).expanduser().resolve())
            if args.edge_table
            else None,
            "edge_table_dir": str(
                Path(args.edge_table_dir).expanduser().resolve()
            )
            if args.edge_table_dir
            else None,
            **immutable_input_identity,
        }
        mismatches = {
            key: {"found": manifest.get(key), "expected": expected}
            for key, expected in expected_identity.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"completed fully-random output identity mismatch: {mismatches}"
            )
        print(
            f"[FULL-RANDOM] reusing validated completed output: {manifest_path}",
            flush=True,
        )
        return
    _validate_existing_rows(
        output_dir, recipe=args.recipe, expected_seed=direction_seed
    )

    vector_audit: list[Dict[str, Any]] = []
    original_build = generation.VectorBankBuilder.build_from_records

    def build_fully_random(
        builder: Any, records: Iterable[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        metric_bank = original_build(builder, records)
        random_bank, audit = randomize_vector_bank(
            metric_bank, base_seed=direction_seed
        )
        vector_audit.clear()
        vector_audit.extend(audit)
        print(
            f"[FULL-RANDOM] replaced {len(audit)} metric-derived activation "
            f"directions with {RANDOM_DIRECTION_ALGORITHM}",
            flush=True,
        )
        return random_bank

    original_append = generation.append_jsonl_row

    def append_with_contract(handle: Any, row: Mapping[str, Any]) -> None:
        original_append(
            handle,
            _stamp_row(
                row, recipe=args.recipe, direction_seed=direction_seed
            ),
        )

    generation.VectorBankBuilder.build_from_records = build_fully_random
    generation.append_jsonl_row = append_with_contract
    generation.main(generator_args)
    if not vector_audit:
        raise RuntimeError(
            "generation completed without building any randomized activation vectors"
        )

    raw_path = output_dir / f"{args.recipe}_raw_candidates.jsonl"
    selected_path = output_dir / f"{args.recipe}_selected.jsonl"
    raw_count = _validate_jsonl_rows(
        raw_path, recipe=args.recipe, expected_seed=direction_seed
    )
    selected_count = _validate_jsonl_rows(
        selected_path, recipe=args.recipe, expected_seed=direction_seed
    )
    manifest = {
        "randomization_contract": RANDOMIZATION_CONTRACT,
        "random_position_algorithm": RANDOM_POSITION_ALGORITHM,
        "random_direction_algorithm": RANDOM_DIRECTION_ALGORITHM,
        "metric_direction_orientation_used": False,
        "random_direction_seed": direction_seed,
        "recipe": args.recipe,
        "prompt_mode": args.prompt_mode,
        "gen_mode": args.gen_mode,
        "circuit_target_mode": args.circuit_target_mode,
        "model_name_or_path": args.model_name_or_path,
        **immutable_input_identity,
        "edge_table": str(Path(args.edge_table).expanduser().resolve())
        if args.edge_table
        else None,
        "edge_table_dir": str(Path(args.edge_table_dir).expanduser().resolve())
        if args.edge_table_dir
        else None,
        "budget": {
            "seed_data_num": int(args.seed_data_num),
            "num_candidates": int(args.num_candidates),
            "keep_per_record": int(args.keep_per_record),
            "topk_edges": int(args.topk_edges),
        },
        "counts": {"raw": raw_count, "selected": selected_count},
        "paths": {
            "output_dir": str(output_dir),
            "raw": str(raw_path),
            "selected": str(selected_path),
        },
        "generator_path": str(generator_path),
        "adapter_path": str(adapter_path),
        "vector_audit": vector_audit,
    }
    _write_manifest(manifest_path, manifest)
    validate_fully_random_output_dir(
        output_dir,
        recipe=args.recipe,
        expected_seed_data_num=int(args.seed_data_num),
        expected_num_candidates=int(args.num_candidates),
        expected_keep_per_record=int(args.keep_per_record),
        expected_topk_edges=int(args.topk_edges),
    )
    print(
        f"[FULL-RANDOM] validated contract and outputs: {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
