"""Build the Experiment-2 Steering × Selection conditions.

Every complete condition is emitted as an easy/hard/helpful three-band mirror
pool so downstream training can use the same stationary Uniform-Mix.  Clean
steered inputs are resolved only from ``circuit_gen_prev``; Exp4 ablation pools
are never searched.
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
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from applications.experiment_utils.io import (  # noqa: E402
    GENERATOR_MODEL_KEY,
    NAIVE_RAW_PATH,
    dump_pool_as_mirror,
    find_full_steered_pool_dir,
    read_jsonl,
)
from applications.baselines.fully_random_circuit_entry import (  # noqa: E402
    validate_fully_random_output_dir,
)
from applications.steering_selection.support_scoring import (  # noqa: E402
    candidate_id,
    load_support_index,
    recipe_weights_from_rows,
    support_components,
    validate_score_coverage,
    weighted_support_score,
)

from applications import curriculum_ft as curriculum_core  # type: ignore  # noqa: E402

PROFILES = ("easy", "hard", "helpful")
CONDITIONS = (
    "unsteered_random",
    "unsteered_filter",
    "random_steering_filter",
    "steered_random",
    "steered_filter",
    "steered_diversity",
)
# Match the stationary Uniform-Mix used by the existing baselines exactly.
UNIFORM_MIX = {"easy": 0.333, "hard": 0.333, "helpful": 0.334}
EXP2_WORK_ROOT = _SUPPLEMENTAL_ROOT / "steering_selection" / "work_exp2_steering_selection"


def parse_csv(raw: str, allowed: Sequence[str]) -> List[str]:
    values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported values {unknown}; allowed={list(allowed)}")
    if not values:
        raise ValueError("at least one value is required")
    return values


def require_selectable(rows: Sequence[Mapping], n: int, label: str) -> None:
    if n <= 0:
        raise ValueError(f"n-select must be positive, got {n}")
    if len(rows) < n:
        raise ValueError(f"{label}: need at least {n} candidates, found {len(rows)}")


def random_sample(rows: Sequence[dict], n: int, seed: int) -> List[dict]:
    require_selectable(rows, n, "random sample")
    rng = random.Random(seed)
    return [rows[idx] for idx in rng.sample(range(len(rows)), n)]


def top_by_scores(rows: Sequence[dict], scores: Mapping[str, float], n: int) -> List[dict]:
    require_selectable(rows, n, "support filtering")
    ranked = sorted(
        enumerate(rows),
        key=lambda pair: (-float(scores[candidate_id(pair[1])]), pair[0]),
    )
    return [rows[idx] for idx, _row in ranked[:n]]


def profile_quotas(total: int) -> Dict[str, int]:
    counts = {profile: int(total * UNIFORM_MIX[profile]) for profile in PROFILES}
    remainder = total - sum(counts.values())
    # Largest-remainder method, deterministic tie break by canonical order.
    order = sorted(
        PROFILES,
        key=lambda profile: (total * UNIFORM_MIX[profile] - counts[profile], -PROFILES.index(profile)),
        reverse=True,
    )
    for idx in range(remainder):
        counts[order[idx % len(order)]] += 1
    return counts


def split_in_order(rows: Sequence[dict], counts: Mapping[str, int]) -> Dict[str, List[dict]]:
    expected = sum(counts.values())
    if len(rows) != expected:
        raise ValueError(f"split expected {expected} rows, got {len(rows)}")
    out: Dict[str, List[dict]] = {}
    offset = 0
    for profile in PROFILES:
        end = offset + counts[profile]
        out[profile] = list(rows[offset:end])
        offset = end
    return out


def disjoint_profile_filter(
    rows: Sequence[dict],
    score_maps: Mapping[str, Mapping[str, float]],
    counts: Mapping[str, int],
) -> Dict[str, List[dict]]:
    """Allocate profile top scores without assigning a neutral candidate twice."""

    ranked: Dict[str, List[dict]] = {
        profile: sorted(
            rows,
            key=lambda row: (-float(score_maps[profile][candidate_id(row)]), candidate_id(row)),
        )
        for profile in PROFILES
    }
    cursors = {profile: 0 for profile in PROFILES}
    selected: Dict[str, List[dict]] = {profile: [] for profile in PROFILES}
    used = set()
    while any(len(selected[profile]) < counts[profile] for profile in PROFILES):
        progressed = False
        for profile in PROFILES:
            if len(selected[profile]) >= counts[profile]:
                continue
            candidates = ranked[profile]
            while cursors[profile] < len(candidates):
                row = candidates[cursors[profile]]
                cursors[profile] += 1
                cid = candidate_id(row)
                if cid in used:
                    continue
                selected[profile].append(row)
                used.add(cid)
                progressed = True
                break
        if not progressed:
            raise ValueError(
                f"could not fill disjoint profile quotas {dict(counts)} from {len(rows)} rows"
            )
    return selected


def namespace_ids(rows: Sequence[dict], profile: str) -> List[dict]:
    """Make local per-generation candidate IDs globally unique across profiles."""

    out = []
    for row in rows:
        copied = dict(row)
        original = candidate_id(row)
        copied["exp2_original_candidate_id"] = original
        copied["candidate_id"] = f"{profile}:{original}"
        out.append(copied)
    return out


def normalize_trainer_valid(rows: Sequence[dict], dataset: str = "sciq") -> List[dict]:
    """Keep only rows accepted by the real trainer and freeze that normalization."""

    output = []
    for row in rows:
        converted = curriculum_core.generated_row_to_example(row, dataset)
        if converted is None:
            continue
        normalized = dict(row)
        normalized["question"] = converted["question_text"]
        normalized["choices"] = list(converted["choices"])
        normalized["answer"] = converted["gold_label"]
        output.append(normalized)
    return output


def validate_condition_pools(
    pools: Mapping[str, Sequence[dict]], total: int, condition: str
) -> Dict[str, object]:
    all_ids: List[str] = []
    sizes = {}
    for profile in PROFILES:
        ids = [candidate_id(row) for row in pools[profile]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{condition}/{profile}: duplicate candidate_id within profile")
        all_ids.extend(ids)
        sizes[profile] = len(ids)
    n_unique = len(set(all_ids))
    n_overlap = len(all_ids) - n_unique
    if len(all_ids) != total or n_overlap:
        raise ValueError(
            f"{condition}: expected total={total}, overlap=0; "
            f"got total={len(all_ids)}, unique={n_unique}, overlap={n_overlap}"
        )
    return {
        "profile_sizes": sizes,
        "n_total": len(all_ids),
        "n_unique_candidate_ids": n_unique,
        "n_overlap_candidate_ids": n_overlap,
    }


def farthest_point(rows: Sequence[dict], n: int, model_id: str, seed: int) -> List[dict]:
    require_selectable(rows, n, "diversity filtering")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    texts = [
        str(row.get("question") or (row.get("source") or {}).get("question_text") or "")
        for row in rows
    ]
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(rows)))
    picked = [first]
    distance = 1.0 - embeddings @ embeddings[first]
    distance[first] = -np.inf
    for _ in range(n - 1):
        idx = int(np.argmax(distance))
        picked.append(idx)
        distance = np.minimum(distance, 1.0 - embeddings @ embeddings[idx])
        distance[idx] = -np.inf
    return [rows[idx] for idx in picked]


def random_circuit_output_dir(
    profile: str, *, asset_root: Path, run_tag: str
) -> Path:
    """Return the one explicit Exp2-local fully-random generation directory.

    There is deliberately no latest-run or legacy ``circuit_gen_prev``
    fallback: an incomplete or stale location-only random control must not be
    accepted as the fully-random direction control required by the rebuttal.
    """

    return (
        Path(asset_root)
        / GENERATOR_MODEL_KEY
        / profile
        / "fixed_both_random"
        / run_tag
    )


def validate_random_circuit_assets(
    *,
    profiles: Sequence[str],
    asset_root: Path,
    run_tag: str,
    seed_data_num: int,
    num_candidates: int,
    keep_per_record: int,
    topk_edges: int,
    allow_missing: bool,
) -> tuple[Dict[str, Path], Dict[str, Mapping[str, object]], str]:
    """Resolve and fully validate all profile assets before pool construction."""

    paths: Dict[str, Path] = {}
    manifests: Dict[str, Mapping[str, object]] = {}
    missing: List[str] = []
    for profile in profiles:
        output_dir = random_circuit_output_dir(
            profile, asset_root=asset_root, run_tag=run_tag
        )
        if not output_dir.is_dir():
            missing.append(f"{profile}: {output_dir}")
            continue
        try:
            manifest = validate_fully_random_output_dir(
                output_dir,
                recipe=profile,
                expected_seed_data_num=seed_data_num,
                expected_num_candidates=num_candidates,
                expected_keep_per_record=keep_per_record,
                expected_topk_edges=topk_edges,
            )
        except Exception as exc:
            # Existing-but-unverifiable data are invalid, not merely missing.
            # ALLOW_MISSING_RANDOM_CIRCUIT must never mask legacy/mixed rows.
            raise ValueError(
                f"invalid fully-random circuit asset for {profile} at "
                f"{output_dir}: {exc}"
            ) from exc
        selected = output_dir / f"{profile}_selected.jsonl"
        paths[profile] = selected
        manifests[profile] = {
            "output_dir": str(output_dir.resolve()),
            "sidecar": str(
                (output_dir / f"{profile}_fully_random_manifest.json").resolve()
            ),
            "randomization_contract": manifest["randomization_contract"],
            "random_direction_algorithm": manifest["random_direction_algorithm"],
            "random_direction_seed": manifest["random_direction_seed"],
            "budget": dict(manifest["budget"]),
            "counts": dict(manifest["counts"]),
        }
        print(f"[random-input] validated {profile}: {selected}")

    missing_message = ""
    if missing:
        missing_message = (
            "random_steering_filter requires Exp2-local fully-random-position/head/"
            "direction assets for every Uniform-Mix profile. Missing:\n  - "
            + "\n  - ".join(missing)
        )
        if not allow_missing:
            raise FileNotFoundError(
                missing_message
                + "\nRun MODE=random_generate first, or explicitly set "
                "ALLOW_MISSING_RANDOM_CIRCUIT=1 to skip this entire condition."
            )
        print(f"[SKIP] {missing_message}")
        paths = {}
        manifests = {}
    return paths, manifests, missing_message


def stamp_provenance(
    rows: Sequence[Mapping],
    *,
    condition: str,
    source_profile: str,
    selection: str,
) -> List[dict]:
    stamped = []
    for row in rows:
        out = dict(row)
        out["exp2_provenance"] = {
            "condition": condition,
            "source_profile": source_profile,
            "selection": selection,
            "source_circuit_target_mode": str(
                (row.get("steering") or {}).get("circuit_target_mode") or ""
            ),
            "source_randomization_contract": str(
                (row.get("steering") or {}).get("randomization_contract") or ""
            ),
            "source_random_direction_algorithm": str(
                (row.get("steering") or {}).get("random_direction_algorithm") or ""
            ),
            "source_random_direction_seed": (
                (row.get("steering") or {}).get("random_direction_seed")
            ),
        }
        stamped.append(out)
    return stamped


def write_band(
    rows: Sequence[Mapping],
    *,
    variant_root: Path,
    condition: str,
    profile: str,
    source_profile: str,
    selection: str,
) -> str:
    output = dump_pool_as_mirror(
        stamp_provenance(
            rows,
            condition=condition,
            source_profile=source_profile,
            selection=selection,
        ),
        variant_root=variant_root / condition,
        recipe=profile,
        prompt_mode="fixed",
        gen_mode="both",
        # Trainer-facing mirror namespace; source provenance remains explicit.
        circuit_target_mode="discovered",
        file_kind="selected",
    )
    print(f"[write] {condition}/{profile}: {len(rows)} rows -> {output}")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive-raw", default=str(NAIVE_RAW_PATH))
    parser.add_argument(
        "--naive-support-scores",
        default=str(EXP2_WORK_ROOT / "_support" / "naive_support_scores.jsonl"),
    )
    parser.add_argument("--profiles", default="easy,hard,helpful")
    parser.add_argument("--variant-root", default=str(EXP2_WORK_ROOT))
    parser.add_argument(
        "--random-asset-root",
        default=str(EXP2_WORK_ROOT / "random_circuit_assets"),
    )
    parser.add_argument("--random-run-tag", default="fully_random_seed42")
    parser.add_argument("--random-seed-data-num", type=int, default=5000)
    parser.add_argument("--random-num-candidates", type=int, default=4)
    parser.add_argument("--random-keep-per-record", type=int, default=2)
    parser.add_argument("--random-topk-edges", type=int, default=250)
    parser.add_argument("--validate-random-assets-only", action="store_true")
    parser.add_argument("--n-select", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minilm", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--allow-missing-random-circuit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    profiles = parse_csv(args.profiles, PROFILES)
    if not args.validate_random_assets_only and profiles != list(PROFILES):
        raise ValueError(
            "Exp2 downstream comparisons require all profiles in order: easy,hard,helpful"
        )
    conditions = parse_csv(args.conditions, CONDITIONS)

    random_paths: Dict[str, Path] = {}
    random_asset_validation: Dict[str, Mapping[str, object]] = {}
    random_missing_message = ""
    if args.validate_random_assets_only or "random_steering_filter" in conditions:
        (
            random_paths,
            random_asset_validation,
            random_missing_message,
        ) = validate_random_circuit_assets(
            profiles=profiles,
            asset_root=Path(args.random_asset_root),
            run_tag=args.random_run_tag,
            seed_data_num=args.random_seed_data_num,
            num_candidates=args.random_num_candidates,
            keep_per_record=args.random_keep_per_record,
            topk_edges=args.random_topk_edges,
            allow_missing=args.allow_missing_random_circuit,
        )
    if args.validate_random_assets_only:
        if random_missing_message:
            print("[random-input] condition unavailable; explicit missing-assets skip enabled")
        else:
            print(
                "[random-input] all fully-random assets validated: "
                + ", ".join(str(random_paths[p]) for p in profiles)
            )
        return

    variant_root = Path(args.variant_root)
    variant_root.mkdir(parents=True, exist_ok=True)

    naive_raw_rows = read_jsonl(args.naive_raw)
    if args.limit:
        naive_raw_rows = naive_raw_rows[: args.limit]
    naive = normalize_trainer_valid(naive_raw_rows)
    require_selectable(naive, args.n_select, "naive raw")
    quotas = profile_quotas(args.n_select)
    print(
        f"[input] naive raw: raw={len(naive_raw_rows)} trainer_valid={len(naive)} "
        f"@ {args.naive_raw}"
    )

    steered_raw: Dict[str, List[dict]] = {}
    steered_selected: Dict[str, List[dict]] = {}
    steered_dirs: Dict[str, str] = {}
    input_validation: Dict[str, Mapping[str, int]] = {
        "naive": {"raw": len(naive_raw_rows), "trainer_valid": len(naive)}
    }
    recipe_weights: Dict[str, Dict[str, float]] = {}
    for profile in profiles:
        directory = find_full_steered_pool_dir(profile)
        path = directory / f"{profile}_raw_candidates.jsonl"
        raw_rows = read_jsonl(path)
        if args.limit:
            raw_rows = raw_rows[: args.limit]
        rows = normalize_trainer_valid(raw_rows)
        require_selectable(rows, quotas[profile], f"clean steered {profile}")
        for row in raw_rows:
            support_components(row)
        recipe_weights[profile] = recipe_weights_from_rows(raw_rows, profile)
        steered_raw[profile] = rows
        selected_path = directory / f"{profile}_selected.jsonl"
        selected_raw_rows = read_jsonl(selected_path)
        if args.limit:
            selected_raw_rows = selected_raw_rows[: args.limit]
        selected_rows = normalize_trainer_valid(selected_raw_rows)
        require_selectable(
            selected_rows, quotas[profile], f"clean steered selected {profile}"
        )
        steered_selected[profile] = selected_rows
        steered_dirs[profile] = str(directory)
        input_validation[profile] = {
            "raw_candidates": len(raw_rows),
            "raw_candidates_trainer_valid": len(rows),
            "selected": len(selected_raw_rows),
            "selected_trainer_valid": len(selected_rows),
        }
        print(
            f"[input] clean steered {profile}: raw={len(raw_rows)} "
            f"valid={len(rows)} @ {path}; selected={len(selected_raw_rows)} "
            f"valid={len(selected_rows)} @ {selected_path}"
        )

    written: Dict[str, Dict[str, str]] = {}
    condition_meta: Dict[str, Dict[str, object]] = {}
    skipped: Dict[str, str] = {}

    if "unsteered_random" in conditions:
        # One neutral random control exposed under each recipe label so all
        # conditions use the exact same Uniform-Mix FT interface.
        selected = random_sample(naive, args.n_select, args.seed)
        pools = split_in_order(selected, quotas)
        condition_meta["unsteered_random"] = validate_condition_pools(
            pools, args.n_select, "unsteered_random"
        )
        written["unsteered_random"] = {}
        for profile in profiles:
            written["unsteered_random"][profile] = write_band(
                pools[profile],
                variant_root=variant_root,
                condition="unsteered_random",
                profile=profile,
                source_profile="naive",
                selection="random_top_half",
            )

    if "unsteered_filter" in conditions:
        score_index = load_support_index(args.naive_support_scores)
        validate_score_coverage(naive_raw_rows, score_index, label="naive raw")
        score_maps = {}
        for profile in profiles:
            score_maps[profile] = {
                cid: weighted_support_score(vector, recipe_weights[profile])
                for cid, vector in score_index.items()
            }
        pools = disjoint_profile_filter(naive, score_maps, quotas)
        condition_meta["unsteered_filter"] = validate_condition_pools(
            pools, args.n_select, "unsteered_filter"
        )
        written["unsteered_filter"] = {}
        for profile in profiles:
            written["unsteered_filter"][profile] = write_band(
                pools[profile],
                variant_root=variant_root,
                condition="unsteered_filter",
                profile=profile,
                source_profile="naive",
                selection=f"top_half_exact_{profile}_support",
            )

    if "random_steering_filter" in conditions:
        if random_missing_message:
            print(f"[SKIP] {random_missing_message}")
            skipped["random_steering_filter"] = random_missing_message
        else:
            written["random_steering_filter"] = {}
            random_condition_pools: Dict[str, List[dict]] = {}
            for profile_idx, (profile, path) in enumerate(random_paths.items()):
                raw_rows = read_jsonl(path)
                if args.limit:
                    raw_rows = raw_rows[: args.limit]
                rows = normalize_trainer_valid(raw_rows)
                require_selectable(rows, quotas[profile], f"random-circuit selected {profile}")
                first_mode = str((rows[0].get("steering") or {}).get("circuit_target_mode"))
                if first_mode != "random":
                    raise ValueError(
                        f"{path}: expected steering.circuit_target_mode='random', got {first_mode!r}"
                    )
                # The source file is already the generation-time top-half.
                # Draw the profile quota uniformly from that selected 10k;
                # taking rows[:quota] would introduce an unintended second rank filter.
                selected = namespace_ids(
                    random_sample(
                        rows, quotas[profile], args.seed + 20000 + 1009 * profile_idx
                    ),
                    profile,
                )
                random_condition_pools[profile] = selected
                written["random_steering_filter"][profile] = write_band(
                    selected,
                    variant_root=variant_root,
                    condition="random_steering_filter",
                    profile=profile,
                    source_profile=profile,
                    selection="generation_time_exact_support_filter",
                )
            condition_meta["random_steering_filter"] = validate_condition_pools(
                random_condition_pools, args.n_select, "random_steering_filter"
            )

    if "steered_random" in conditions:
        written["steered_random"] = {}
        steered_random_pools: Dict[str, List[dict]] = {}
        for profile_idx, profile in enumerate(profiles):
            selected = namespace_ids(
                random_sample(
                    steered_raw[profile], quotas[profile], args.seed + 1009 * profile_idx
                ),
                profile,
            )
            steered_random_pools[profile] = selected
            written["steered_random"][profile] = write_band(
                selected,
                variant_root=variant_root,
                condition="steered_random",
                profile=profile,
                source_profile=profile,
                selection="random_top_half",
            )
        condition_meta["steered_random"] = validate_condition_pools(
            steered_random_pools,
            args.n_select,
            "steered_random",
        )

    if "steered_filter" in conditions:
        written["steered_filter"] = {}
        steered_filter_pools: Dict[str, List[dict]] = {}
        for profile_idx, profile in enumerate(profiles):
            # The clean selected file is already the original support top-half.
            # Random quota sampling keeps the downstream total budget at 10k
            # without applying another score threshold.
            selected = namespace_ids(
                random_sample(
                    steered_selected[profile],
                    quotas[profile],
                    args.seed + 30000 + 1009 * profile_idx,
                ),
                profile,
            )
            steered_filter_pools[profile] = selected
            written["steered_filter"][profile] = write_band(
                selected,
                variant_root=variant_root,
                condition="steered_filter",
                profile=profile,
                source_profile=profile,
                selection="generation_time_exact_support_filter",
            )
        condition_meta["steered_filter"] = validate_condition_pools(
            steered_filter_pools, args.n_select, "steered_filter"
        )

    if "steered_diversity" in conditions:
        written["steered_diversity"] = {}
        diversity_pools: Dict[str, List[dict]] = {}
        for profile_idx, profile in enumerate(profiles):
            selected = farthest_point(
                steered_raw[profile],
                quotas[profile],
                model_id=args.minilm,
                seed=args.seed + 1009 * profile_idx,
            )
            selected = namespace_ids(selected, profile)
            diversity_pools[profile] = selected
            written["steered_diversity"][profile] = write_band(
                selected,
                variant_root=variant_root,
                condition="steered_diversity",
                profile=profile,
                source_profile=profile,
                selection="minilm_greedy_farthest_point",
            )
        condition_meta["steered_diversity"] = validate_condition_pools(
            diversity_pools, args.n_select, "steered_diversity"
        )

    incomplete = {
        condition: sorted(set(profiles) - set(paths))
        for condition, paths in written.items()
        if set(paths) != set(profiles)
    }
    if incomplete:
        raise RuntimeError(f"internal error: incomplete Uniform-Mix conditions: {incomplete}")

    manifest = {
        "profiles": profiles,
        "uniform_mix_ratios": UNIFORM_MIX,
        "n_select_total": int(args.n_select),
        "profile_quotas": quotas,
        "naive_raw": str(Path(args.naive_raw).resolve()),
        "naive_support_scores": str(Path(args.naive_support_scores).resolve()),
        "clean_steered_dirs": steered_dirs,
        "random_asset_root": str(Path(args.random_asset_root).resolve()),
        "random_run_tag": args.random_run_tag,
        "fully_random_asset_validation": random_asset_validation,
        "input_trainer_validation": input_validation,
        "complete_conditions": sorted(written),
        "skipped_conditions": skipped,
        "files": written,
        "condition_pool_validation": condition_meta,
    }
    manifest_path = variant_root / "build_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
