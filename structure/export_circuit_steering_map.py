"""Export validated structural circuits to JSON and generation-ready edge CSVs.

All six metric/bucket edge and node tables are required. Analysis summaries and
faithfulness measurements are optional metadata; no steering presets or claims
about causal reliability are inferred from a previous experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_KEYS = ("aum", "el2n", "grad_align")
BUCKETS = ("high", "low")
EDGE_FIELDS = (
    "source", "target", "qkv", "score", "abs_score", "signed_role",
    "src_kind", "dst_kind", "src_layer", "dst_layer", "src_head", "dst_head",
)
NODE_FIELDS = ("name", "kind", "layer", "head", "total_score", "signed_role")


def finite_float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return number


def optional_int(value: Any, context: str) -> int | None:
    if value in (None, "", "None"):
        return None
    number = finite_float(value, context)
    if not number.is_integer():
        raise ValueError(f"{context}: expected an integer, got {value!r}")
    return int(number)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: missing or duplicate CSV column names")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: table is empty")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: row width does not match CSV header")
    return rows


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        # Existing pandas-derived summaries use NaN for undefined statistics.
        payload = json.load(handle, parse_constant=lambda value: None)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def node_details(name: str) -> tuple[str, int | None, int | None]:
    attn = re.fullmatch(r"a(\d+)\.(?:h|kv)(\d+)", name)
    if attn:
        return "attn", int(attn[1]), int(attn[2])
    mlp = re.fullmatch(r"m(\d+)", name)
    if mlp:
        return "mlp", int(mlp[1]), None
    if name in ("input", "embed", "hook_embed"):
        return "embed", None, None
    if name in ("logits", "unembed", "lm_head"):
        return "unembed", None, None
    return "other", None, None


def parse_edges(rows: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        context = f"{path}:{line}"
        source = str(row.get("src", row.get("source", ""))).strip()
        target = str(row.get("dst", row.get("target", ""))).strip()
        if not source or not target:
            raise ValueError(f"{context}: source/src and target/dst are required")
        qkv = str(row.get("qkv", "")).strip().lower()
        if qkv not in ("", "q", "k", "v"):
            raise ValueError(f"{context}: qkv must be q, k, v, or empty")
        key = (source, target, qkv)
        if key in seen:
            raise ValueError(f"{context}: duplicate edge {key}")
        seen.add(key)
        score = finite_float(row.get("score"), f"{context} score")
        abs_score = finite_float(row.get("abs_score", abs(score)), f"{context} abs_score")
        if not math.isclose(abs_score, abs(score), rel_tol=1e-5, abs_tol=1e-10):
            raise ValueError(f"{context}: abs_score does not match abs(score)")
        edge = {
            "source": source, "target": target, "qkv": qkv,
            "score": score, "abs_score": abs_score,
            # These are sign labels for the generation loader, not causal claims.
            "signed_role": "driver" if score >= 0 else "suppressor",
        }
        for prefix, name in (("src", source), ("dst", target)):
            kind, layer, head = node_details(name)
            edge[f"{prefix}_kind"] = row.get(f"{prefix}_kind") or kind
            edge[f"{prefix}_layer"] = optional_int(
                row.get(f"{prefix}_layer", row.get(f"{prefix}_plot_layer", layer)),
                f"{context} {prefix}_layer",
            )
            edge[f"{prefix}_head"] = optional_int(row.get(f"{prefix}_head", head), f"{context} {prefix}_head")
        edges.append(edge)
    return edges


def parse_nodes(rows: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    nodes = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        context = f"{path}:{line}"
        name = str(row.get("node", row.get("name", ""))).strip()
        if not name or name in seen:
            raise ValueError(f"{context}: missing or duplicate node name {name!r}")
        seen.add(name)
        kind, layer, head = node_details(name)
        score = finite_float(row.get("total_score", row.get("score")), f"{context} total_score")
        nodes.append({
            "name": name, "kind": row.get("kind") or kind,
            "layer": optional_int(row.get("layer", layer), f"{context} layer"),
            "head": optional_int(row.get("head", head), f"{context} head"),
            "total_score": score, "signed_role": "driver" if score >= 0 else "suppressor",
        })
    return nodes


def summarize_targets(edges: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[int, int] = {}
    qkv_by_layer: dict[int, dict[str, float]] = {}
    for edge in edges:
        for layer in (edge["src_layer"], edge["dst_layer"]):
            if layer is not None and layer >= 0:
                counts[layer] = counts.get(layer, 0) + 1
        layer, qkv = edge["dst_layer"], edge["qkv"].upper()
        if layer is not None and layer >= 0 and qkv:
            masses = qkv_by_layer.setdefault(layer, {"Q": 0.0, "K": 0.0, "V": 0.0})
            masses[qkv] += edge["abs_score"]
    return {
        "top_edges": edges, "top_nodes": nodes,
        "top_attention_heads": [n for n in nodes if n["kind"] in ("attn", "attention")][:10],
        "top_mlp_nodes": [n for n in nodes if n["kind"] == "mlp"][:10],
        "hub_layers": sorted(counts, key=lambda layer: (-counts[layer], layer))[:5],
        "qkv_by_layer": qkv_by_layer,
    }



def read_faithfulness_metadata(
    root: Path, *, dataset: str, metric: str, bucket: str, corrupt_type: str,
) -> dict[str, Any]:
    """Read a complete pair from a nested layout or faithfulness.py's flat <root>/<tag> output."""
    tag = f"{dataset}_{metric}_{bucket}_{corrupt_type}"
    candidates = (root / metric / bucket / tag, root / metric / tag, root / tag)
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        raise FileNotFoundError(
            f"--faith-root has no result directory for {metric}/{bucket}; "
            f"checked {', '.join(str(path) for path in candidates)}"
        )
    result = {}
    for name in ("baseline", "summary"):
        path = directory / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"--faith-root requires baseline.json and summary.json for every circuit; missing {path}")
        result[name] = read_optional_json(path)
        if not result[name]:
            raise ValueError(f"{path}: faithfulness metadata must not be empty")
    return result


def build_steering_map(
    structure_root: Path, *, model: str, dataset: str, topk: int = 250,
    top_nodes: int = 50, corrupt_type: str = "gaussian", faith_root: Path | None = None,
) -> dict[str, Any]:
    """Read and validate every source table before any output is created."""
    if faith_root is not None and not faith_root.is_dir():
        raise FileNotFoundError(f"--faith-root is not an existing directory: {faith_root}")
    if topk < 1 or top_nodes < 1:
        raise ValueError("topk and top_nodes must be positive")
    result: dict[str, Any] = {
        "model": model, "dataset": dataset, "corrupt_type": corrupt_type,
        "topk_for_structure": f"top{topk}", "combos": {},
    }
    for metric in SCORE_KEYS:
        metric_root = structure_root / f"{metric}_{corrupt_type}"
        backbone = read_optional_json(metric_root / "analysis" / "shared_backbone_summary.json")
        for bucket in BUCKETS:
            source = metric_root / "visualizations" / bucket
            edges_path, nodes_path = source / "topk_graph_edges.csv", source / "topk_graph_nodes.csv"
            edges = parse_edges(read_csv(edges_path), edges_path)
            nodes = parse_nodes(read_csv(nodes_path), nodes_path)
            node_names = {node["name"] for node in nodes}
            missing = {e[endpoint] for e in edges for endpoint in ("source", "target")} - node_names
            if missing:
                raise ValueError(f"{nodes_path}: edge endpoints missing from node table: {sorted(missing)}")
            selected_edges = sorted(edges, key=lambda edge: -edge["abs_score"])[:topk]
            # Keep the node-importance ranking produced by visualize_circuits.py.
            selected_nodes = nodes[:top_nodes]
            combo: dict[str, Any] = {
                "score_key": metric, "bucket": bucket,
                "steering_targets": summarize_targets(selected_edges, selected_nodes),
                "structure_stats": read_optional_json(metric_root / "analysis" / bucket / "structure_summary.json"),
                "shared_backbone": backbone,
                "source_tables": {"edges": str(edges_path), "nodes": str(nodes_path)},
            }
            if faith_root is not None:
                combo["faithfulness"] = read_faithfulness_metadata(
                    faith_root, dataset=dataset, metric=metric, bucket=bucket, corrupt_type=corrupt_type,
                )
            result["combos"][f"{metric}_{bucket}"] = combo
    return result


def write_outputs(steering_map: Mapping[str, Any], out_path: Path, csv_dir: Path) -> None:
    serialized = json.dumps(steering_map, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
    for key, combo in steering_map["combos"].items():
        for kind, fields in (("edges", EDGE_FIELDS), ("nodes", NODE_FIELDS)):
            with (csv_dir / f"{key}_top_{kind}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(combo["steering_targets"][f"top_{kind}"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True, help="Directory containing {metric}_{corrupt-type}/visualizations/{high,low}.")
    parser.add_argument("--model", required=True, help="Model identifier recorded in output metadata.")
    parser.add_argument("--dataset", required=True, help="Dataset identifier recorded in output metadata.")
    parser.add_argument("--out-path", type=Path, default=REPO_ROOT / "results" / "circuit_steering_map.json")
    parser.add_argument("--csv-dir", type=Path, help="Default: <out-path parent>/circuit_steering_csvs.")
    parser.add_argument("--faith-root", type=Path, help="Optional faithfulness result root, e.g. faithfulness.py's --out_dir (tag directories), or a root with metric/bucket/tag or metric/tag directories. Requires baseline.json and summary.json for all six circuits.")
    parser.add_argument("--corrupt-type", default="gaussian")
    parser.add_argument("--topk", type=int, default=250, help="Maximum edges per circuit (ranked by absolute attribution).")
    parser.add_argument("--top-nodes", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_steering_map(
            args.structure_root.expanduser(), model=args.model, dataset=args.dataset,
            topk=args.topk, top_nodes=args.top_nodes, corrupt_type=args.corrupt_type,
            faith_root=args.faith_root.expanduser() if args.faith_root else None,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    out_path = args.out_path.expanduser()
    csv_dir = args.csv_dir.expanduser() if args.csv_dir else out_path.parent / "circuit_steering_csvs"
    write_outputs(payload, out_path, csv_dir)
    print(f"Saved {len(payload['combos'])} circuits: {out_path}\nCSV directory: {csv_dir}")


if __name__ == "__main__":
    main()
