"""Exp 4-a: circuit overlap and correlational selectivity.

Jaccard is computed from the canonical EAP-IG ``*.edge_scores.jsonl``
files, not the 50-row visualization CSV exports. Spearman correlates one
genuine per-sample circuit-attribution score ``S_c(x)`` with each stored
utility score after joining on ``hf_id``. A full-vs-ablated metric delta is
not a circuit activation score and is never used here.
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
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CIRCUITS = (
    "aum_high",
    "aum_low",
    "el2n_high",
    "el2n_low",
    "grad_align_high",
    "grad_align_low",
)

UTILITY_KEYS = ("aum", "el2n", "grad_align")
CIRCUIT_ROWS = ("aum_high", "el2n_high", "grad_align_high")

def _finite_float(value: object, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _object_sha256(obj: Dict[str, object]) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def validate_activation_certificate(
    csv_path: Path,
    *,
    score_key: str,
    expected_samples: int,
    topk_edges: int,
    expected_ig_steps: int,
    expected_incorrect_agg: str,
    expected_noise_sigma: float,
) -> Dict[str, object]:
    request_path = csv_path.with_suffix(".exp4_request.json")
    completion_path = csv_path.with_suffix(".exp4_complete.json")
    request = _read_json_object(request_path)
    completion = _read_json_object(completion_path)
    expected = {
        "schema_version": 2,
        "dataset": "sciq",
        "score_key": score_key,
        "expected_samples": expected_samples,
        "prompt_source": "exact utility JSON prompt/gold_token_id/label_token_ids",
        "topk_edges": topk_edges,
        "topk_rank_abs": True,
        "score_mode": "weighted_direction_aware",
        "ig_steps_arg": expected_ig_steps,
        "incorrect_agg_arg": expected_incorrect_agg,
        "random_baseline_trials": 0,
    }
    mismatches = [
        f"{key}: actual={request.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if request.get(key) != value
    ]
    try:
        noise_matches = math.isclose(
            float(request.get("noise_sigma")),
            expected_noise_sigma,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    except (TypeError, ValueError):
        noise_matches = False
    if not noise_matches:
        mismatches.append(
            f"noise_sigma: actual={request.get('noise_sigma')!r}, "
            f"expected={expected_noise_sigma!r}"
        )
    prompt_schema = request.get("prompt_schema")
    if not isinstance(prompt_schema, dict) or prompt_schema.get("num_train_examples") != expected_samples:
        mismatches.append("prompt_schema does not certify the full SciQ train population")
    for key in (
        "hf_id_sha256", "split_content_sha256", "utility_prompt_sha256",
        "utility_source_meta_sha256",
    ):
        if not str(request.get(key, "")):
            mismatches.append(f"missing request fingerprint {key}")
    for path_key, hash_key in (
        ("utility_json", "utility_json_sha256"),
        ("high_circuit_pt", "high_circuit_sha256"),
        ("low_circuit_pt", "low_circuit_sha256"),
    ):
        path_text = str(request.get(path_key, ""))
        path = Path(path_text).expanduser() if path_text else None
        if path is None or not path.is_file():
            mismatches.append(f"missing current fingerprint target {path_key}={path_text!r}")
        elif str(request.get(hash_key, "")) != _sha256(path):
            mismatches.append(f"{hash_key} no longer matches {path_key}")
    request_sha256 = _object_sha256(request)
    if completion.get("status") != "complete":
        mismatches.append(f"completion status is {completion.get('status')!r}, not complete")
    if completion.get("request_contract_sha256") != request_sha256:
        mismatches.append("completion certificate belongs to a different request")
    if completion.get("csv_sha256") != _sha256(csv_path):
        mismatches.append("CSV hash differs from its completion certificate")
    if completion.get("n_records") != expected_samples:
        mismatches.append(
            f"completion n_records={completion.get('n_records')!r}, expected={expected_samples}"
        )
    if mismatches:
        raise ValueError(
            f"{csv_path}: uncertified/stale full attribution artifact: "
            + "; ".join(mismatches)
        )
    return {
        "request": str(request_path),
        "completion": str(completion_path),
        "request_contract_sha256": request_sha256,
        "csv_sha256": completion.get("csv_sha256"),
        "prompt_source": request.get("prompt_source"),
        "utility_prompt_sha256": request.get("utility_prompt_sha256"),
        "hf_id_sha256": request.get("hf_id_sha256"),
    }


def validate_cross_request_identity(
    certifications: Dict[str, Dict[str, object]]
) -> Tuple[str, str]:
    prompt_digests = {
        str(certifications.get(metric, {}).get("utility_prompt_sha256", ""))
        for metric in UTILITY_KEYS
    }
    hf_id_digests = {
        str(certifications.get(metric, {}).get("hf_id_sha256", ""))
        for metric in UTILITY_KEYS
    }
    if "" in prompt_digests or len(prompt_digests) != 1:
        raise RuntimeError(
            f"activation requests use different per-hf prompt/gold data: {sorted(prompt_digests)}"
        )
    if "" in hf_id_digests or len(hf_id_digests) != 1:
        raise RuntimeError(
            f"activation requests use different hf_id populations: {sorted(hf_id_digests)}"
        )
    return next(iter(prompt_digests)), next(iter(hf_id_digests))


def load_canonical_top_edges(path: Path, topk: int) -> List[str]:
    """Load exactly ``topk`` unique canonical edges, ranked by |EAP score|."""
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not path.is_file():
        raise FileNotFoundError(f"canonical edge file not found: {path}")

    ranked: List[Tuple[float, int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            edge = str(row.get("edge", "")).strip()
            matrix_index = row.get("matrix_index")
            if (
                not isinstance(matrix_index, list)
                or len(matrix_index) != 2
                or not all(isinstance(value, int) for value in matrix_index)
            ):
                raise ValueError(f"{path}:{line_no}: missing canonical matrix_index pair")
            # Matrix index is invariant to GQA query-head aliases, unlike a
            # human-readable edge representative.
            edge_identity = f"matrix:{matrix_index[0]}:{matrix_index[1]}"
            if not edge:
                raise ValueError(f"{path}:{line_no}: missing canonical 'edge'")
            score = _finite_float(row.get("score"), context=f"{path}:{line_no}:score")
            ranked.append((abs(score), line_no, edge_identity))

    # Do not rely on implicit file ordering. Line number is the stable tie break.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    unique: List[str] = []
    seen = set()
    for _, _, edge in ranked:
        if edge in seen:
            continue
        seen.add(edge)
        unique.append(edge)
        if len(unique) == topk:
            break
    if len(unique) != topk:
        raise ValueError(
            f"{path}: requested top-{topk}, but only {len(unique)} unique canonical edges exist"
        )
    return unique


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_sample_score_csv(
    path: Path,
    *,
    activation_column: str,
    utility_column: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return ``hf_id -> S_c(x)`` and ``hf_id -> utility`` from one CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"per-sample circuit-score CSV not found: {path}")
    activations: Dict[str, float] = {}
    utilities: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"hf_id", activation_column, utility_column}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for line_no, row in enumerate(reader, start=2):
            hf_id = str(row.get("hf_id", "")).strip()
            if not hf_id:
                raise ValueError(f"{path}:{line_no}: empty hf_id")
            if hf_id in activations:
                raise ValueError(f"{path}:{line_no}: duplicate hf_id {hf_id!r}")
            activations[hf_id] = _finite_float(
                row.get(activation_column),
                context=f"{path}:{line_no}:{activation_column}",
            )
            utilities[hf_id] = _finite_float(
                row.get(utility_column),
                context=f"{path}:{line_no}:{utility_column}",
            )
    if len(activations) < 2:
        raise ValueError(f"{path}: need at least two samples, found {len(activations)}")
    return activations, utilities


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Tie-aware average ranks used only when SciPy is unavailable."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and arr[order[end]] == arr[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats

        rho, p_value = stats.spearmanr(a, b)
        return float(rho), float(p_value)
    except ImportError:
        ra, rb = _average_ranks(a), _average_ranks(b)
        if np.std(ra) == 0 or np.std(rb) == 0:
            return float("nan"), float("nan")
        return float(np.corrcoef(ra, rb)[0, 1]), float("nan")


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    digits: Optional[int] = 6,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["circuit"] + list(column_labels))
        for i, label in enumerate(row_labels):
            values: List[object] = []
            for value in matrix[i]:
                if not math.isfinite(float(value)):
                    values.append("NaN")
                elif digits is None:
                    values.append(int(value))
                else:
                    values.append(f"{float(value):.{digits}f}")
            writer.writerow([label] + values)


def heatmap(
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    out_path: Path,
    title: str,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "viridis",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib unavailable: {exc}")
        return

    masked = np.ma.masked_invalid(np.asarray(matrix, dtype=float))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#d9d9d9")
    fig, ax = plt.subplots(
        figsize=(max(4.5, len(xlabels) * 1.0), max(4.0, len(ylabels) * 0.75))
    )
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    for (i, j), value in np.ndenumerate(np.asarray(matrix, dtype=float)):
        text = "NA" if not math.isfinite(float(value)) else f"{value:.2f}"
        ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[png] {out_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-aum-high", required=True)
    parser.add_argument("--edge-aum-low", required=True)
    parser.add_argument("--edge-el2n-high", required=True)
    parser.add_argument("--edge-el2n-low", required=True)
    parser.add_argument("--edge-grad-align-high", required=True)
    parser.add_argument("--edge-grad-align-low", required=True)
    parser.add_argument(
        "--topk-edges", type=int, default=250,
        help="Canonical edge budget. Exp4 uses 250.",
    )
    parser.add_argument("--activation-aum", required=True)
    parser.add_argument("--activation-el2n", required=True)
    parser.add_argument(
        "--activation-grad-align", required=True
    )
    parser.add_argument(
        "--activation-column", default="high_circuit_score",
        help="One scalar S_c(x) column in each per-sample circuit-score CSV.",
    )
    parser.add_argument(
        "--utility-column", default="true_score",
        help="Stored utility score column in each corresponding metric CSV.",
    )
    parser.add_argument(
        "--missing-policy", choices=("fail", "nan"), default="fail",
        help="Fail on a missing score CSV, or explicitly emit its row/column as NaN.",
    )
    parser.add_argument(
        "--expected-samples", type=int, default=11679,
        help="Expected complete SciQ training-split size.",
    )
    parser.add_argument("--expected-ig-steps", type=int, default=5)
    parser.add_argument(
        "--expected-incorrect-agg", choices=("max", "mean", "min"), default="max"
    )
    parser.add_argument("--expected-noise-sigma", type=float, default=0.1)
    parser.add_argument(
        "--allow-subset", action="store_true",
        help="Explicitly allow a subset analysis; metadata and plot title will label it as such.",
    )
    parser.add_argument(
        "--result-dir",
        default=(
            str(_REPO_ROOT / 'results/supplemental/selectivity/result_exp4_selectivity_matrix')
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    result_dir = Path(args.result_dir)

    # 4-a(i): canonical top-250 overlap from the exact six resolved artifacts.
    edge_paths = {
        "aum_high": Path(args.edge_aum_high),
        "aum_low": Path(args.edge_aum_low),
        "el2n_high": Path(args.edge_el2n_high),
        "el2n_low": Path(args.edge_el2n_low),
        "grad_align_high": Path(args.edge_grad_align_high),
        "grad_align_low": Path(args.edge_grad_align_low),
    }
    edges = {
        circuit: load_canonical_top_edges(edge_paths[circuit], args.topk_edges)
        for circuit in CIRCUITS
    }
    print("[edges]", {name: len(value) for name, value in edges.items()})
    jaccard_matrix = np.empty((len(CIRCUITS), len(CIRCUITS)), dtype=float)
    for i, left in enumerate(CIRCUITS):
        for j, right in enumerate(CIRCUITS):
            jaccard_matrix[i, j] = jaccard(edges[left], edges[right])
    # Each circuit row uses its own scalar activation map; every utility
    # column comes from its corresponding CSV, yielding all 9 cross cells.
    score_paths = {
        "aum": Path(args.activation_aum),
        "el2n": Path(args.activation_el2n),
        "grad_align": Path(args.activation_grad_align),
    }
    activations: Dict[str, Dict[str, float]] = {}
    utilities: Dict[str, Dict[str, float]] = {}
    missing_metrics: List[str] = []
    certifications: Dict[str, Dict[str, object]] = {}
    for metric in UTILITY_KEYS:
        try:
            if not args.allow_subset:
                certifications[metric] = validate_activation_certificate(
                    score_paths[metric],
                    score_key=metric,
                    expected_samples=args.expected_samples,
                    topk_edges=args.topk_edges,
                    expected_ig_steps=args.expected_ig_steps,
                    expected_incorrect_agg=args.expected_incorrect_agg,
                    expected_noise_sigma=args.expected_noise_sigma,
                )
            activation_map, utility_map = load_sample_score_csv(
                score_paths[metric],
                activation_column=args.activation_column,
                utility_column=args.utility_column,
            )
        except (FileNotFoundError, ValueError) as exc:
            if args.missing_policy == "fail":
                raise
            print(f"[missing] {metric}: {exc}")
            missing_metrics.append(metric)
            continue
        activations[f"{metric}_high"] = activation_map
        utilities[metric] = utility_map
        print(f"[sample-scores] {metric}: n={len(activation_map)} path={score_paths[metric]}")

    if not args.allow_subset:
        validate_cross_request_identity(certifications)

    if args.expected_samples <= 1:
        raise ValueError("--expected-samples must be greater than one")
    counts = {metric: len(utilities.get(metric, {})) for metric in UTILITY_KEYS}
    incomplete_counts = {
        metric: count for metric, count in counts.items() if count != args.expected_samples
    }
    if incomplete_counts and not args.allow_subset:
        raise RuntimeError(
            "Spearman requires the complete SciQ train split. "
            f"expected {args.expected_samples} rows per metric, got {incomplete_counts}. "
            "Run MODE=scores to resume full scoring, or use --allow-subset explicitly."
        )
    available_id_sets = [set(values) for values in utilities.values()]
    common_ids = set.intersection(*available_id_sets) if available_id_sets else set()
    if len(common_ids) != args.expected_samples and not args.allow_subset:
        raise RuntimeError(
            "metric CSVs do not cover the same complete hf_id set: "
            f"common={len(common_ids)}, expected={args.expected_samples}"
        )
    sample_scope = "full_train" if len(common_ids) == args.expected_samples else "explicit_subset"

    rho_matrix = np.full((len(CIRCUIT_ROWS), len(UTILITY_KEYS)), np.nan, dtype=float)
    p_matrix = np.full_like(rho_matrix, np.nan)
    n_matrix = np.zeros_like(rho_matrix, dtype=int)
    for i, circuit in enumerate(CIRCUIT_ROWS):
        activation_map = activations.get(circuit)
        if activation_map is None:
            continue
        for j, utility in enumerate(UTILITY_KEYS):
            utility_map = utilities.get(utility)
            if utility_map is None:
                continue
            matched_ids = sorted(set(activation_map) & set(utility_map))
            if len(matched_ids) < 2:
                print(f"[missing] {circuit} x {utility}: {len(matched_ids)} matched hf_ids")
                continue
            rho, p_value = spearman(
                [activation_map[hf_id] for hf_id in matched_ids],
                [utility_map[hf_id] for hf_id in matched_ids],
            )
            rho_matrix[i, j] = rho
            p_matrix[i, j] = p_value
            n_matrix[i, j] = len(matched_ids)
        print(f"[spearman] {circuit}: {rho_matrix[i].tolist()}")

    metadata = {
        "definition": {
            "jaccard": (
                "canonical matrix-index-deduplicated EAP-IG edges ranked by "
                f"absolute score and truncated to top-{args.topk_edges}"
            ),
            "circuit_score": (
                f"one per-sample scalar from {args.activation_column!r}; "
                "never a full-vs-ablated utility difference"
            ),
            "utility_score": f"column {args.utility_column!r} in each metric CSV",
            "alignment": "pairwise inner join on hf_id for all 9 cross cells",
        },
        "topk_edges": int(args.topk_edges),
        "edge_paths": {name: str(path.resolve()) for name, path in edge_paths.items()},
        "edge_sha256": {name: _sha256(path) for name, path in edge_paths.items()},
        "sample_score_paths": {name: str(path) for name, path in score_paths.items()},
        "sample_score_counts": {
            metric: len(utilities.get(metric, {})) for metric in UTILITY_KEYS
        },
        "expected_samples": int(args.expected_samples),
        "common_hf_ids": len(common_ids),
        "sample_scope": sample_scope,
        "allow_subset": bool(args.allow_subset),
        "activation_certifications": certifications,
        "missing_metrics": missing_metrics,
        "spearman_pair_counts": n_matrix.tolist(),
    }

    # Validate and compute everything before atomically replacing any public artifact.
    # The stage must live inside result_dir: it may be a symlink onto another device,
    # and a rename across devices is not possible.
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{result_dir.name}.stage-", dir=result_dir
    ) as temporary:
        stage = Path(temporary)
        write_matrix_csv(
            stage / "jaccard.csv", jaccard_matrix,
            row_labels=CIRCUITS, column_labels=CIRCUITS,
        )
        heatmap(
            jaccard_matrix, CIRCUITS, CIRCUITS,
            stage / "jaccard_heatmap.png",
            f"Canonical top-{args.topk_edges} edge Jaccard",
            vmin=0.0, vmax=1.0, cmap_name="viridis",
        )
        write_matrix_csv(
            stage / "spearman.csv", rho_matrix,
            row_labels=CIRCUIT_ROWS, column_labels=UTILITY_KEYS,
        )
        write_matrix_csv(
            stage / "spearman_pvalues.csv", p_matrix,
            row_labels=CIRCUIT_ROWS, column_labels=UTILITY_KEYS,
        )
        write_matrix_csv(
            stage / "spearman_n.csv", n_matrix.astype(float),
            row_labels=CIRCUIT_ROWS, column_labels=UTILITY_KEYS, digits=None,
        )
        heatmap(
            rho_matrix, UTILITY_KEYS, CIRCUIT_ROWS,
            stage / "spearman_heatmap.png",
            f"Spearman: circuit attribution vs utility ({sample_scope}, n={len(common_ids)})",
            vmin=-1.0, vmax=1.0, cmap_name="coolwarm",
        )
        with (stage / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        for staged in stage.iterdir():
            staged.replace(result_dir / staged.name)
    print(f"[ok] Exp4-a outputs written under {result_dir}")


if __name__ == "__main__":
    main()
