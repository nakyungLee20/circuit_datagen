"""Exact support-score utilities for Experiment 2.

The unsteered (``naive``) candidates were generated without selection, so
their JSONL intentionally contains no ``internal_scores``.  Experiment 2 must
not replace those missing scores with zero or with empirical difficulty.  This
module provides:

* a resumable CLI that evaluates every naive completion with the same
  generator, discovered circuits, activation vectors, and five-log-probability
  scoring procedure used during steered generation; and
* lightweight helpers used by the pool builder and distribution analysis to
  apply the *same profile-specific recipe weights* to naive and steered support
  components.

The expensive CLI writes a compact candidate-id -> support-vector JSONL.  It
does not modify the original generated pools.
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
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

REQUIRED_COMPONENTS = (
    "base_logprob",
    "easy_support",
    "hard_support",
    "helpful_support",
    "shortcut_support",
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_EDGE_DIR = _REPO_ROOT / 'results/circuit_steering_csvs'
DEFAULT_CALIBRATION = Path(
    str(_REPO_ROOT / 'results/generation_inputs/sciq/train/sciq_train_calibration_all.jsonl')
)
CONTRACT_VERSION = "exp2_support_scoring_request_v1"


def candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id")
    if value is None or not str(value).strip():
        raise ValueError("row is missing a non-empty candidate_id")
    return str(value)


def support_components(row: Mapping[str, Any]) -> Dict[str, float]:
    """Return and validate a support vector from a pool or score-file row."""

    raw = row.get("support_scores")
    if not isinstance(raw, Mapping):
        raw = row.get("internal_scores")
    if not isinstance(raw, Mapping):
        raise ValueError(f"candidate {candidate_id(row)!r} has no support score mapping")

    out: Dict[str, float] = {}
    missing = []
    for key in REQUIRED_COMPONENTS:
        value = raw.get(key)
        if value is None:
            missing.append(key)
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate {candidate_id(row)!r} has non-numeric {key}={value!r}"
            ) from exc
    if missing:
        raise ValueError(
            f"candidate {candidate_id(row)!r} is missing support components: {missing}"
        )
    return out


def recipe_weights_from_rows(rows: Iterable[Mapping[str, Any]], profile: str) -> Dict[str, float]:
    """Read the generation-time recipe weights and verify they are consistent."""

    expected: Dict[str, float] | None = None
    checked = 0
    for row in rows:
        steering = row.get("steering")
        raw = steering.get("selection_score_weights") if isinstance(steering, Mapping) else None
        if not isinstance(raw, Mapping) or not raw:
            continue
        weights = {str(k): float(v) for k, v in raw.items()}
        if expected is None:
            expected = weights
        elif weights != expected:
            raise ValueError(
                f"{profile}: inconsistent selection_score_weights in raw pool; "
                f"first={expected}, candidate={candidate_id(row)!r}, current={weights}"
            )
        checked += 1
    if expected is None:
        raise ValueError(
            f"{profile}: no steering.selection_score_weights found in the clean raw pool"
        )
    if checked == 0:
        raise AssertionError("unreachable")
    unknown = sorted(set(expected) - set(REQUIRED_COMPONENTS))
    if unknown:
        raise ValueError(f"{profile}: unsupported recipe weight keys: {unknown}")
    return expected


def weighted_support_score(
    components: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Apply the exact generation-time recipe weights to support components."""

    missing = [key for key in weights if key not in components]
    if missing:
        raise ValueError(f"support vector is missing weighted keys: {missing}")
    return float(sum(float(weights[key]) * float(components[key]) for key in weights))


def load_support_index(
    path: str | Path, *, require_contract: bool = True
) -> Dict[str, Dict[str, float]]:
    """Load a compact support-score JSONL and reject duplicate candidate IDs."""

    score_path = Path(path)
    if not score_path.is_file():
        raise FileNotFoundError(
            f"naive support-score file not found: {score_path}\n"
            "Create it first: python applications/steering_selection/support_scoring.py "
            f"--input-jsonl <naive_raw_candidates.jsonl> --output-jsonl {score_path}\n"
            "(the naive pool comes from applications/circuit_generation.py --recipe naive)."
        )
    if require_contract:
        contract_sha256, expected_n_scored = _validate_completed_artifact_contract(
            score_path
        )
    else:
        contract_sha256, expected_n_scored = "", -1
    out: Dict[str, Dict[str, float]] = {}
    with score_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = candidate_id(row)
            if cid in out:
                raise ValueError(
                    f"duplicate candidate_id={cid!r} in {score_path} at line {line_no}"
                )
            if require_contract and str(row.get("request_contract_sha256") or "") != contract_sha256:
                raise RuntimeError(
                    f"{score_path}:{line_no}: row is not bound to completed request "
                    f"contract {contract_sha256}"
                )
            out[cid] = support_components(row)
    if not out:
        raise ValueError(f"empty support-score file: {score_path}")
    if require_contract and len(out) != expected_n_scored:
        raise RuntimeError(
            f"{score_path}: completed metadata n_scored={expected_n_scored}, "
            f"actual unique rows={len(out)}"
        )
    return out


def validate_score_coverage(
    rows: Iterable[Mapping[str, Any]],
    score_index: Mapping[str, Mapping[str, float]],
    *,
    label: str,
) -> None:
    ids = [candidate_id(row) for row in rows]
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise ValueError(f"{label}: raw pool contains {duplicate_count} duplicate candidate IDs")
    missing = [cid for cid in ids if cid not in score_index]
    extra = sorted(set(score_index) - set(ids))
    if missing or extra:
        raise ValueError(
            f"{label}: support-score coverage mismatch: "
            f"raw={len(ids)}, scored={len(score_index)}, "
            f"missing={len(missing)}, extra={len(extra)}; "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return _sha256_text(encoded)


def _file_contract(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _meta_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def _read_existing_score_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = candidate_id(row)
            if cid in seen:
                raise ValueError(
                    f"duplicate candidate_id={cid!r} in {path} at line {line_no}"
                )
            support_components(row)
            seen.add(cid)
            rows.append(row)
    return rows


def _index_input_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cid = candidate_id(row)
        if cid in indexed:
            raise ValueError(f"input contains duplicate candidate_id={cid!r}")
        indexed[cid] = row
    return indexed


def _recognized_edge_contract(cdg: Any, edge_dir: Path) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    seen_keys: Dict[str, str] = {}
    for path in sorted(edge_dir.glob("*_top_edges.csv")):
        match = cdg._CIRCUIT_FILE_RE.search(path.name)
        if match is None:
            continue
        raw_metric = match.group("metric").lower().replace("_", "")
        polarity = match.group("polarity").lower()
        key = cdg.canonical_circuit_key(raw_metric, polarity)
        if key in seen_keys:
            raise ValueError(
                f"duplicate recognized edge files for {key}: "
                f"{seen_keys[key]} and {path}"
            )
        seen_keys[key] = str(path.resolve())
        entries.append({"circuit_key": key, **_file_contract(path)})
    expected = set(cdg.CANONICAL_CIRCUIT_KEYS)
    actual = set(seen_keys)
    if actual != expected:
        raise ValueError(
            "recognized edge-file circuit set mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return entries


def _load_prompt_tokenizer(cdg: Any, model_name_or_path: str) -> Any:
    transformers = cdg.require_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenizer_contract(tokenizer: Any) -> Dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    special_tokens = getattr(tokenizer, "special_tokens_map", {})
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", -1)),
        "chat_template_sha256": _canonical_sha256(chat_template),
        "special_tokens_sha256": _canonical_sha256(special_tokens),
    }


def _render_scoring_prompt(cdg: Any, tokenizer: Any, row: Mapping[str, Any]) -> str:
    cid = candidate_id(row)
    source = row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"candidate {cid!r} has no source record")
    system_prompt = cdg.build_system_prompt("easy", "fixed")
    user_prompt = cdg.build_user_prompt(source, "easy", "fixed")
    # Call the core renderer itself without initializing a model/steerer.
    renderer = object.__new__(cdg.CircuitAwareGenerator)
    renderer.tokenizer = tokenizer
    return cdg.CircuitAwareGenerator.render_prompt_text(
        renderer, system_prompt, user_prompt
    )


def _build_request_contract(
    *,
    args: argparse.Namespace,
    cdg: Any,
    input_path: Path,
    edge_dir: Path,
    calibration_path: Path,
    tokenizer: Any,
) -> Dict[str, Any]:
    generator_path = Path(cdg.__file__).resolve()
    scorer_path = Path(__file__).resolve()
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "unknown"
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "unknown"
    return {
        "contract_version": CONTRACT_VERSION,
        "input_jsonl": _file_contract(input_path),
        "calibration_jsonl": _file_contract(calibration_path),
        "recognized_edge_files": _recognized_edge_contract(cdg, edge_dir),
        "code": {
            "generator": _file_contract(generator_path),
            "scorer": _file_contract(scorer_path),
        },
        "model": {
            "model_name_or_path": str(args.model_name_or_path),
            "dtype": str(args.dtype),
            "device_map": str(args.device_map),
            "force_eager_attention": True,
            "prefer_single_device": True,
            "tokenizer": _tokenizer_contract(tokenizer),
            "transformers_version": transformers_version,
            "torch_version": torch_version,
        },
        "scoring": {
            "topk_edges": int(args.topk_edges),
            "max_attn_groups_per_circuit": int(args.max_attn_groups_per_circuit),
            "max_mlp_per_circuit": int(args.max_mlp_per_circuit),
            "calibration_quantile": float(args.calibration_quantile),
            "max_calibration_per_group": int(args.max_calibration_per_group),
            "calibration_batch_size": int(args.calibration_batch_size),
            "calibration_max_length": int(args.calibration_max_length),
            "limit": int(args.limit),
            "prompt_target": "easy",
            "prompt_mode": "fixed",
            "target_recipe": "easy",
            "base_recipe": "neutral",
            "use_activation_addition": True,
            "use_attention_steering": True,
            "support_components": list(REQUIRED_COMPONENTS),
        },
    }


def _write_contract_meta(
    meta_path: Path,
    *,
    contract: Mapping[str, Any],
    status: str,
    output_path: Path,
    n_scored: int,
) -> None:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "request_contract_sha256": _canonical_sha256(contract),
        "request_contract": dict(contract),
        "status": status,
        "output_jsonl": str(output_path.resolve()),
        "n_scored": int(n_scored),
    }
    if status == "complete":
        payload["output_file"] = _file_contract(output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(meta_path)


def _validate_stored_contract(
    meta_path: Path, expected_contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not meta_path.is_file():
        raise RuntimeError(
            "stale support-score output: score JSONL exists without its exact request "
            f"contract sidecar {meta_path}. Remove/move the old output or choose a new "
            "--output-jsonl; it cannot be resumed safely."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = meta.get("request_contract")
    if not isinstance(stored, Mapping):
        raise RuntimeError(
            f"stale support-score metadata has no request_contract: {meta_path}"
        )
    stored_digest = str(meta.get("request_contract_sha256") or "")
    actual_stored_digest = _canonical_sha256(stored)
    if stored_digest != actual_stored_digest:
        raise RuntimeError(
            f"support-score metadata contract digest is corrupt at {meta_path}: "
            f"stored={stored_digest}, recomputed={actual_stored_digest}"
        )
    expected_digest = _canonical_sha256(expected_contract)
    if stored_digest != expected_digest or dict(stored) != dict(expected_contract):
        raise RuntimeError(
            "stale support-score request contract mismatch; refusing to mix resumed "
            f"rows. stored={stored_digest}, current={expected_digest}. "
            "Remove/move the old output or choose a new --output-jsonl."
        )
    return meta


def _validate_completed_artifact_contract(score_path: Path) -> tuple[str, int]:
    """Revalidate a completed artifact before any pool/stat consumer uses it."""

    meta_path = _meta_path(score_path)
    if not meta_path.is_file():
        raise RuntimeError(
            f"support-score artifact has no exact contract sidecar: {meta_path}. "
            "Regenerate it with MODE=scores."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    contract = meta.get("request_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"invalid request_contract in {meta_path}")
    _validate_stored_contract(meta_path, contract)
    if meta.get("status") != "complete":
        raise RuntimeError(
            f"support-score artifact is not complete: status={meta.get('status')!r}"
        )
    if str(meta.get("output_jsonl") or "") != str(score_path.resolve()):
        raise RuntimeError(
            f"support-score sidecar/output mismatch: sidecar={meta.get('output_jsonl')!r}, "
            f"requested={str(score_path.resolve())!r}"
        )
    stored_output = meta.get("output_file")
    if not isinstance(stored_output, Mapping):
        raise RuntimeError(
            f"completed support-score sidecar has no output fingerprint: {meta_path}. "
            "Run MODE=scores once to revalidate and finalize the artifact."
        )
    current_output = _file_contract(score_path)
    comparable_output = {
        "path": stored_output.get("path"),
        "size_bytes": stored_output.get("size_bytes"),
        "sha256": stored_output.get("sha256"),
    }
    if current_output != comparable_output:
        raise RuntimeError(
            "completed support-score output changed after finalization: "
            f"stored={comparable_output}, current={current_output}"
        )

    fingerprint_entries = [
        ("input_jsonl", contract.get("input_jsonl")),
        ("calibration_jsonl", contract.get("calibration_jsonl")),
    ]
    code = contract.get("code")
    if not isinstance(code, Mapping):
        raise RuntimeError(f"contract code fingerprint missing in {meta_path}")
    fingerprint_entries.extend(
        [("code.generator", code.get("generator")), ("code.scorer", code.get("scorer"))]
    )
    edge_entries = contract.get("recognized_edge_files")
    if not isinstance(edge_entries, list) or not edge_entries:
        raise RuntimeError(f"contract edge fingerprints missing in {meta_path}")
    fingerprint_entries.extend(
        (f"recognized_edge_files[{idx}]", entry)
        for idx, entry in enumerate(edge_entries)
    )
    for label, stored_entry in fingerprint_entries:
        if not isinstance(stored_entry, Mapping) or not stored_entry.get("path"):
            raise RuntimeError(f"invalid {label} fingerprint in {meta_path}")
        current = _file_contract(Path(str(stored_entry["path"])))
        comparable = {
            "path": stored_entry.get("path"),
            "size_bytes": stored_entry.get("size_bytes"),
            "sha256": stored_entry.get("sha256"),
        }
        if current != comparable:
            raise RuntimeError(
                f"support-score dependency changed for {label}: "
                f"stored={comparable}, current={current}"
            )

    model = contract.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError(f"contract model settings missing in {meta_path}")
    current_versions = {
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": importlib.metadata.version("torch"),
    }
    for key, current in current_versions.items():
        if str(model.get(key)) != str(current):
            raise RuntimeError(
                f"support-score environment changed for {key}: "
                f"stored={model.get(key)!r}, current={current!r}"
            )
    return str(meta["request_contract_sha256"]), int(meta.get("n_scored", -1))


def _validate_existing_row_content(
    *,
    existing_rows: Sequence[Mapping[str, Any]],
    input_by_id: Mapping[str, Mapping[str, Any]],
    contract_sha256: str,
    prompt_builder: Callable[[Mapping[str, Any]], str],
) -> None:
    for score_row in existing_rows:
        cid = candidate_id(score_row)
        input_row = input_by_id.get(cid)
        if input_row is None:
            raise RuntimeError(
                f"stale support-score row candidate_id={cid!r} is outside current input"
            )
        completion = input_row.get("generation_text")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"candidate {cid!r} has no generation_text completion")
        expected_completion = _sha256_text(completion)
        expected_prompt = _sha256_text(prompt_builder(input_row))
        actual_completion = str(score_row.get("completion_sha256") or "")
        actual_prompt = str(score_row.get("scoring_prompt_sha256") or "")
        actual_contract = str(score_row.get("request_contract_sha256") or "")
        if actual_completion != expected_completion:
            raise RuntimeError(
                f"stale completion for candidate={cid!r}: stored={actual_completion}, "
                f"current={expected_completion}"
            )
        if actual_prompt != expected_prompt:
            raise RuntimeError(
                f"stale scoring prompt for candidate={cid!r}: stored={actual_prompt}, "
                f"current={expected_prompt}"
            )
        if actual_contract != contract_sha256:
            raise RuntimeError(
                f"candidate={cid!r} is not bound to the current request contract: "
                f"stored={actual_contract}, current={contract_sha256}"
            )


def score_naive_pool(args: argparse.Namespace) -> None:
    """Run the original circuit-aware scorer on unsteered completions."""

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    edge_dir = Path(args.edge_table_dir)
    calibration_path = Path(args.calibration_jsonl)
    for required in (input_path, edge_dir, calibration_path):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(_REPO_ROOT / 'applications'))
    # circuit_gen_prev was produced by this released generator. A later,
    # unreleased generator variant used different recipe weights/scales, so
    # scoring with it would silently define a different s_k for the neutral
    # controls.
    from applications import circuit_generation as cdg  # type: ignore

    rows = cdg.load_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    input_by_id = _index_input_rows(rows)

    # Tokenizer-only loading is sufficient to fingerprint/reconstruct scoring
    # prompts. A completed resume can therefore be validated without loading
    # the full 1.5B model onto GPU.
    prompt_tokenizer = _load_prompt_tokenizer(cdg, args.model_name_or_path)
    contract = _build_request_contract(
        args=args,
        cdg=cdg,
        input_path=input_path,
        edge_dir=edge_dir,
        calibration_path=calibration_path,
        tokenizer=prompt_tokenizer,
    )
    contract_sha256 = _canonical_sha256(contract)
    meta_path = _meta_path(output_path)
    if output_path.exists() or meta_path.exists():
        _validate_stored_contract(meta_path, contract)
    else:
        _write_contract_meta(
            meta_path,
            contract=contract,
            status="in_progress",
            output_path=output_path,
            n_scored=0,
        )

    existing_rows = _read_existing_score_rows(output_path)
    _validate_existing_row_content(
        existing_rows=existing_rows,
        input_by_id=input_by_id,
        contract_sha256=contract_sha256,
        prompt_builder=lambda row: _render_scoring_prompt(
            cdg, prompt_tokenizer, row
        ),
    )
    existing = {
        candidate_id(row): support_components(row) for row in existing_rows
    }
    remaining = [row for row in rows if candidate_id(row) not in existing]
    print(
        f"[support-score] input={len(rows)} existing={len(existing)} "
        f"remaining={len(remaining)} output={output_path}"
    )
    if not remaining:
        validate_score_coverage(rows, existing, label="naive")
        _write_contract_meta(
            meta_path,
            contract=contract,
            status="complete",
            output_path=output_path,
            n_scored=len(existing),
        )
        print("[support-score] complete; nothing to do")
        return

    presets = cdg.build_default_presets()
    edge_rows = cdg.load_circuit_edge_dir(
        edge_dir,
        topk_per_circuit=args.topk_edges,
    )
    if not edge_rows:
        raise RuntimeError(f"no circuit edges loaded from {edge_dir}")
    derived = cdg.derive_targets_from_edges(
        edge_rows,
        topk_edges=args.topk_edges,
        max_attn_groups_per_circuit=args.max_attn_groups_per_circuit,
        max_mlp_per_circuit=args.max_mlp_per_circuit,
    )
    presets = cdg.merge_presets_with_edge_targets(presets, derived)

    model, tokenizer = cdg.load_model_and_tokenizer(
        args.model_name_or_path,
        dtype=args.dtype,
        device_map=args.device_map,
        force_eager_attention=True,
        prefer_single_device=True,
    )
    if _tokenizer_contract(tokenizer) != _tokenizer_contract(prompt_tokenizer):
        raise RuntimeError(
            "tokenizer contract changed between resume validation and model loading"
        )
    calibration_rows = cdg.load_jsonl(calibration_path)
    vector_builder = cdg.VectorBankBuilder(
        model=model,
        tokenizer=tokenizer,
        presets=presets,
        quantile=args.calibration_quantile,
        max_texts_per_group=args.max_calibration_per_group,
        batch_size=args.calibration_batch_size,
        max_length=args.calibration_max_length,
    )
    vector_bank = vector_builder.build_from_records(calibration_rows)
    missing_vectors = sorted(set(cdg.CANONICAL_CIRCUIT_KEYS) - set(vector_bank))
    if missing_vectors:
        raise RuntimeError(
            "activation-vector calibration did not produce every circuit key: "
            f"{missing_vectors}"
        )
    generator = cdg.CircuitAwareGenerator(
        model=model,
        tokenizer=tokenizer,
        presets=presets,
        vector_bank=vector_bank,
    )

    # prompt_mode=fixed is target-independent.  Reconstructing it from the
    # source row gives exactly the same prompt schema for naive and steered
    # candidates; "easy" here does not inject an easy-band instruction.
    system_prompt = cdg.build_system_prompt("easy", "fixed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for idx, row in enumerate(remaining, start=1):
            cid = candidate_id(row)
            source = row.get("source")
            if not isinstance(source, Mapping):
                raise ValueError(f"candidate {cid!r} has no source record")
            completion = row.get("generation_text")
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError(f"candidate {cid!r} has no generation_text completion")
            user_prompt = cdg.build_user_prompt(source, "easy", "fixed")
            prompt_text = generator.render_prompt_text(system_prompt, user_prompt)
            scores = cdg.score_candidate_from_recipes(
                generator=generator,
                prompt_text=prompt_text,
                completion_text=completion,
                target_recipe=cdg.DEFAULT_RECIPES["easy"],
                base_recipe=cdg.NEUTRAL_RECIPE,
                use_activation_addition=True,
                use_attention_steering=True,
            )
            compact = {
                "candidate_id": cid,
                "support_scores": {key: float(scores[key]) for key in REQUIRED_COMPONENTS},
                "scoring_prompt_sha256": _sha256_text(prompt_text),
                "completion_sha256": _sha256_text(completion),
                "request_contract_sha256": contract_sha256,
            }
            handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
            handle.flush()
            if idx == 1 or idx % max(1, args.log_every) == 0:
                print(f"[support-score] {idx}/{len(remaining)} candidate={cid}")

    # The sidecar remains in_progress until coverage is checked, so bypass the
    # completed-artifact gate only for this one internal finalization read.
    complete = load_support_index(output_path, require_contract=False)
    validate_score_coverage(rows, complete, label="naive")
    _write_contract_meta(
        meta_path,
        contract=contract,
        status="complete",
        output_path=output_path,
        n_scored=len(complete),
    )
    print(f"[support-score] complete: {len(complete)} rows; metadata={meta_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score unsteered candidates with the exact circuit support scorer"
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--edge-table-dir", default=str(DEFAULT_EDGE_DIR))
    parser.add_argument("--calibration-jsonl", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--topk-edges", type=int, default=250)
    parser.add_argument("--max-attn-groups-per-circuit", type=int, default=3)
    parser.add_argument("--max-mlp-per-circuit", type=int, default=3)
    parser.add_argument("--calibration-quantile", type=float, default=0.15)
    parser.add_argument("--max-calibration-per-group", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=4)
    parser.add_argument("--calibration-max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> None:
    score_naive_pool(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
