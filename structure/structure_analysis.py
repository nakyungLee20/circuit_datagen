import argparse
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _circuit_utils import (
    aggregate_circuit_tables,
    build_tag_from_paths,
    display_layer_label,
    ensure_dir,
    layer_mass_matrix,
    load_circuit_table,
    node_importance_table,
    normalized_entropy,
    resolve_input_paths,
    save_csv,
    save_json,
    slugify,
    top_edge_set,
)


DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parents[1] / 'results/structure')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compact structural analysis for EAP-IG circuits.")
    ap.add_argument("--circuits", nargs="*", default=[], help="Single condition circuit files.")
    ap.add_argument("--compare-a", nargs="*", default=[], help="Condition A circuit files.")
    ap.add_argument("--compare-b", nargs="*", default=[], help="Condition B circuit files.")
    ap.add_argument("--label-a", type=str, default="condition_a")
    ap.add_argument("--label-b", type=str, default="condition_b")
    ap.add_argument("--compare-topk", type=int, default=200, help="Top-k edges used in shared-backbone comparison.")
    ap.add_argument("--out-root", type=str, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def _load_tables(paths: Sequence[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for idx, path in enumerate(paths):
        label = f"{Path(path).stem}_{idx}" if Path(path).stem in tables else Path(path).stem
        tables[label] = load_circuit_table(path, included_only=True)
    return tables


def _max_layer(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    vals = pd.concat([df["src_layer"], df["dst_layer"]], axis=0)
    vals = vals[np.isfinite(vals)]
    vals = [int(v) for v in vals if int(v) >= 0]
    if not vals:
        return 0
    return max(vals)


def _indegree_outdegree(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    indegree: Dict[str, int] = {}
    outdegree: Dict[str, int] = {}
    for _, row in df.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        outdegree[src] = outdegree.get(src, 0) + 1
        indegree[dst] = indegree.get(dst, 0) + 1
        indegree.setdefault(src, indegree.get(src, 0))
        outdegree.setdefault(dst, outdegree.get(dst, 0))
    return indegree, outdegree


def _topological_nodes(df: pd.DataFrame) -> Sequence[str]:
    if df.empty:
        return []
    nodes = set(df["src"]) | set(df["dst"])
    layer_map = {}
    for node in nodes:
        node_rows = df[(df["src"] == node) | (df["dst"] == node)]
        src_layer = node_rows["src_plot_layer"][node_rows["src"] == node]
        dst_layer = node_rows["dst_plot_layer"][node_rows["dst"] == node]
        if not src_layer.empty:
            layer_map[node] = int(src_layer.iloc[0])
        elif not dst_layer.empty:
            layer_map[node] = int(dst_layer.iloc[0])
        else:
            layer_map[node] = 0
    return sorted(nodes, key=lambda node: (layer_map[node], node))


def compute_path_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {
            "shortest_hops_to_logits": float("nan"),
            "longest_hops_to_logits": float("nan"),
            "dominant_path_hops": float("nan"),
            "dominant_path_weight": float("nan"),
            "active_layer_span": float("nan"),
        }

    adjacency: Dict[str, list] = {}
    nodes = set(df["src"]) | set(df["dst"])
    for node in nodes:
        adjacency[node] = []
    for _, row in df.iterrows():
        adjacency[str(row["src"])].append((str(row["dst"]), float(row["abs_score"])))

    indegree, outdegree = _indegree_outdegree(df)
    sources = [node for node in nodes if indegree.get(node, 0) == 0]
    sinks = [node for node in nodes if outdegree.get(node, 0) == 0]
    target = "logits" if "logits" in nodes else (sinks[0] if sinks else None)
    topo = _topological_nodes(df)

    shortest = {node: np.inf for node in nodes}
    longest = {node: -np.inf for node in nodes}
    best_weight = {node: -np.inf for node in nodes}
    dominant_hops = {node: np.inf for node in nodes}
    for source in sources:
        shortest[source] = 0.0
        longest[source] = 0.0
        best_weight[source] = 0.0
        dominant_hops[source] = 0.0

    for node in topo:
        if not np.isfinite(shortest[node]) and not np.isfinite(best_weight[node]):
            continue
        for dst, weight in adjacency.get(node, []):
            shortest[dst] = min(shortest[dst], shortest[node] + 1.0)
            longest[dst] = max(longest[dst], longest[node] + 1.0)
            candidate_weight = best_weight[node] + weight
            candidate_hops = dominant_hops[node] + 1.0
            if candidate_weight > best_weight[dst]:
                best_weight[dst] = candidate_weight
                dominant_hops[dst] = candidate_hops
            elif candidate_weight == best_weight[dst] and candidate_hops < dominant_hops[dst]:
                dominant_hops[dst] = candidate_hops

    if target is None:
        target = max(nodes, key=lambda node: best_weight[node])

    layers = sorted({int(layer) for layer in pd.concat([df["src_plot_layer"], df["dst_plot_layer"]], axis=0).dropna()})
    valid_layers = [layer for layer in layers if layer >= 0]
    active_layer_span = (max(valid_layers) - min(valid_layers)) if valid_layers else float("nan")

    return {
        "shortest_hops_to_logits": float(shortest.get(target, np.nan)),
        "longest_hops_to_logits": float(longest.get(target, np.nan)),
        "dominant_path_hops": float(dominant_hops.get(target, np.nan)),
        "dominant_path_weight": float(best_weight.get(target, np.nan)),
        "active_layer_span": float(active_layer_span),
    }


def classify_localization(nodes: pd.DataFrame) -> Dict[str, float]:
    attn = nodes[nodes["kind"] == "attn"].copy()
    if attn.empty:
        return {
            "label": "unknown",
            "top1_share": float("nan"),
            "top5_share": float("nan"),
            "entropy_norm": float("nan"),
            "late_attn_share": float("nan"),
        }
    values = attn["total_abs"].to_numpy(dtype=float)
    total = max(values.sum(), 1e-8)
    top1_share = float(np.sort(values)[::-1][:1].sum() / total)
    top3_share = float(np.sort(values)[::-1][:3].sum() / total)
    top5_share = float(np.sort(values)[::-1][:5].sum() / total)
    entropy_norm = normalized_entropy(values)
    max_layer = int(attn["layer"].dropna().max()) if attn["layer"].notna().any() else 0
    late_cutoff = int(np.floor((2.0 * max_layer) / 3.0))
    late_mass = attn.loc[attn["layer"].fillna(-1) >= late_cutoff, "total_abs"].sum()
    late_attn_share = float(late_mass / total)
    focus_share = top3_share if len(values) <= 8 else top5_share
    if top1_share >= 0.40 or focus_share >= 0.70 or entropy_norm <= 0.55:
        label = "localized"
    elif top1_share <= 0.18 and entropy_norm >= 0.78:
        label = "distributed"
    else:
        label = "mixed"
    return {
        "label": label,
        "top1_share": top1_share,
        "top3_share": top3_share,
        "top5_share": top5_share,
        "entropy_norm": float(entropy_norm),
        "late_attn_share": late_attn_share,
    }


def classify_attention_vs_mlp(nodes: pd.DataFrame) -> Dict[str, float]:
    attn_mass = float(nodes.loc[nodes["kind"] == "attn", "total_abs"].sum())
    mlp_mass = float(nodes.loc[nodes["kind"] == "mlp", "total_abs"].sum())
    denom = max(attn_mass + mlp_mass, 1e-8)
    attn_ratio = attn_mass / denom
    if attn_ratio >= 0.60:
        label = "attention-heavy"
    elif attn_ratio <= 0.40:
        label = "mlp-heavy"
    else:
        label = "balanced"
    return {
        "label": label,
        "attn_mass": attn_mass,
        "mlp_mass": mlp_mass,
        "attn_ratio": attn_ratio,
    }


def classify_bottleneck(df: pd.DataFrame) -> Dict[str, float]:
    terminal = (
        df[df["dst_kind"] == "unembed"]
        .groupby("src", dropna=False)["abs_score"]
        .sum()
        .sort_values(ascending=False)
    )
    if terminal.empty:
        return {
            "label": "unknown",
            "terminal_top1_share": float("nan"),
            "terminal_top3_share": float("nan"),
            "n_terminal_nodes": 0,
        }
    values = terminal.to_numpy(dtype=float)
    total = max(values.sum(), 1e-8)
    top1_share = float(values[:1].sum() / total)
    top3_share = float(values[:3].sum() / total)
    if top1_share >= 0.55 or top3_share >= 0.86:
        label = "single-bottleneck"
    elif top1_share <= 0.35 and len(values) >= 4:
        label = "multi-path"
    else:
        label = "mixed"
    return {
        "label": label,
        "terminal_top1_share": top1_share,
        "terminal_top3_share": top3_share,
        "n_terminal_nodes": int(len(values)),
    }


def classify_path_depth(df: pd.DataFrame) -> Dict[str, float]:
    metrics = compute_path_metrics(df)
    max_layer = _max_layer(df)
    available_depth = max(max_layer + 2, 1)
    dominant = metrics["dominant_path_hops"]
    span = metrics["active_layer_span"]
    norm_depth = float(dominant / available_depth) if np.isfinite(dominant) else float("nan")
    if np.isfinite(dominant) and (dominant <= 3 or (np.isfinite(span) and span <= 2)):
        label = "short"
    elif np.isfinite(dominant) and (dominant >= max(5, int(round(0.60 * available_depth))) or (np.isfinite(span) and span >= 0.65 * max_layer)):
        label = "long"
    else:
        label = "mixed"
    metrics["label"] = label
    metrics["dominant_path_norm"] = norm_depth
    return metrics


def analyze_single_circuit(df: pd.DataFrame) -> Dict[str, object]:
    nodes = node_importance_table(df)
    save = {
        "n_edges": int(len(df)),
        "n_nodes": int(len(nodes)),
        "max_layer": int(_max_layer(df)),
        "localized_vs_distributed": classify_localization(nodes),
        "attention_vs_mlp": classify_attention_vs_mlp(nodes),
        "single_bottleneck_vs_multi_path": classify_bottleneck(df),
        "short_vs_long_path": classify_path_depth(df),
    }
    return save


def _layer_overlap_table(df_a: pd.DataFrame, df_b: pd.DataFrame, topk: int) -> pd.DataFrame:
    top_a = set(df_a.sort_values("abs_score", ascending=False).head(topk)["edge"])
    top_b = set(df_b.sort_values("abs_score", ascending=False).head(topk)["edge"])
    layers = sorted(set(df_a["dst_plot_layer"]) | set(df_b["dst_plot_layer"]))
    rows = []
    for layer in layers:
        layer_a = set(df_a[(df_a["dst_plot_layer"] == layer) & (df_a["edge"].isin(top_a))]["edge"])
        layer_b = set(df_b[(df_b["dst_plot_layer"] == layer) & (df_b["edge"].isin(top_b))]["edge"])
        union = layer_a | layer_b
        rows.append(
            {
                "dst_plot_layer": int(layer),
                "n_a": len(layer_a),
                "n_b": len(layer_b),
                "n_shared": len(layer_a & layer_b),
                "jaccard": (len(layer_a & layer_b) / len(union)) if union else float("nan"),
                "a_abs_mass": float(df_a.loc[df_a["dst_plot_layer"] == layer, "abs_score"].sum()),
                "b_abs_mass": float(df_b.loc[df_b["dst_plot_layer"] == layer, "abs_score"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("dst_plot_layer").reset_index(drop=True)


def compare_shared_backbone(df_a: pd.DataFrame, df_b: pd.DataFrame, *, topk: int) -> Dict[str, object]:
    set_a = top_edge_set(df_a, topk)
    set_b = top_edge_set(df_b, topk)
    union = set_a | set_b
    overall_jaccard = (len(set_a & set_b) / len(union)) if union else 1.0
    overlap_by_layer = _layer_overlap_table(df_a, df_b, topk=topk)
    max_layer = max(_max_layer(df_a), _max_layer(df_b))
    early_cut = int(np.floor(max_layer / 2.0))
    early = overlap_by_layer[overlap_by_layer["dst_plot_layer"] <= early_cut]
    late = overlap_by_layer[overlap_by_layer["dst_plot_layer"] > early_cut]
    early_jaccard = float(early["jaccard"].mean()) if not early.empty else float("nan")
    late_jaccard = float(late["jaccard"].mean()) if not late.empty else float("nan")
    if np.isfinite(early_jaccard) and np.isfinite(late_jaccard) and early_jaccard >= 0.22 and early_jaccard >= late_jaccard + 0.10:
        label = "shared_backbone_task_specific_suffix"
    elif overall_jaccard >= 0.45:
        label = "largely_shared"
    elif np.isfinite(early_jaccard) and np.isfinite(late_jaccard) and late_jaccard >= early_jaccard + 0.10:
        label = "different_backbones"
    else:
        label = "mixed"
    return {
        "label": label,
        "overall_topk_jaccard": float(overall_jaccard),
        "early_layer_overlap": early_jaccard,
        "late_layer_overlap": late_jaccard,
        "topk": int(topk),
        "overlap_by_layer": overlap_by_layer,
    }


def plot_structure_profile(summary: Dict[str, object], out_path: str, *, title: str) -> None:
    loc = summary["localized_vs_distributed"]
    bal = summary["attention_vs_mlp"]
    bottleneck = summary["single_bottleneck_vs_multi_path"]
    depth = summary["short_vs_long_path"]
    metrics = {
        "top5_share": float(loc["top5_share"]),
        "late_attn_share": float(loc["late_attn_share"]),
        "attn_ratio": float(bal["attn_ratio"]),
        "terminal_top1_share": float(bottleneck["terminal_top1_share"]),
        "path_depth_norm": float(depth["dominant_path_norm"]),
    }
    labels = list(metrics.keys())
    values = [metrics[label] if np.isfinite(metrics[label]) else 0.0 for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color=["#577590", "#4d908e", "#43aa8b", "#f8961e", "#f94144"], edgecolor="black")
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15))
    ax.set_ylabel("normalized score")
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shared_backbone(compare: Dict[str, object], out_path: str, *, label_a: str, label_b: str) -> None:
    overlap = compare["overlap_by_layer"]
    if overlap.empty:
        return
    max_real_layer = max(0, int(overlap["dst_plot_layer"].max()) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    axes[0].plot(overlap["dst_plot_layer"], overlap["jaccard"], color="#277da1", linewidth=2.2)
    axes[0].set_title("Edge overlap by destination layer")
    axes[0].set_xlabel("Destination layer")
    axes[0].set_ylabel("Top-k Jaccard")
    axes[0].set_ylim(0.0, 1.02)

    width = 0.38
    xs = np.arange(len(overlap))
    axes[1].bar(xs - width / 2, overlap["a_abs_mass"], width=width, label=label_a, color="#43aa8b", edgecolor="black")
    axes[1].bar(xs + width / 2, overlap["b_abs_mass"], width=width, label=label_b, color="#f8961e", edgecolor="black")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([display_layer_label(int(layer), max_real_layer) for layer in overlap["dst_plot_layer"]], rotation=40, ha="right")
    axes[1].set_title("Condition mass by destination layer")
    axes[1].set_ylabel("sum(|score|)")
    axes[1].legend(frameon=False)

    fig.suptitle("Shared backbone vs task-specific suffix", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_single_analysis(df: pd.DataFrame, out_dir: str, *, title: str) -> Dict[str, object]:
    ensure_dir(out_dir)
    save_csv(df, os.path.join(out_dir, "aggregate_edges.csv"))
    nodes = node_importance_table(df)
    save_csv(nodes, os.path.join(out_dir, "node_importance.csv"))
    layer_matrix = layer_mass_matrix(df, value_col="abs_score")
    save_csv(layer_matrix.reset_index().rename(columns={"src_plot_layer": "layer"}), os.path.join(out_dir, "layer_mass_matrix.csv"))
    terminal = (
        df[df["dst_kind"] == "unembed"]
        .groupby("src", dropna=False)["abs_score"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"src": "terminal_node", "abs_score": "terminal_abs_mass"})
    )
    save_csv(terminal, os.path.join(out_dir, "terminal_bottleneck.csv"))

    summary = analyze_single_circuit(df)
    summary["title"] = title
    save_json(summary, os.path.join(out_dir, "structure_summary.json"))
    plot_structure_profile(summary, os.path.join(out_dir, "structure_profile.png"), title=title)
    return summary


def main() -> None:
    args = parse_args()
    circuits = resolve_input_paths(args.circuits)
    compare_a = resolve_input_paths(args.compare_a)
    compare_b = resolve_input_paths(args.compare_b)
    if not circuits and not (compare_a and compare_b):
        raise SystemExit("Provide either --circuits or both --compare-a and --compare-b.")

    tag_paths = circuits or (compare_a + compare_b)
    tag = slugify(args.tag) if args.tag else build_tag_from_paths(tag_paths, default="structure")
    root_dir = ensure_dir(os.path.join(args.out_root, tag, "analysis"))

    manifest = {
        "circuits": circuits,
        "compare_a": compare_a,
        "compare_b": compare_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "compare_topk": args.compare_topk,
    }
    save_json(manifest, os.path.join(root_dir, "manifest.json"))

    if circuits:
        tables = _load_tables(circuits)
        aggregate = aggregate_circuit_tables(tables)
        save_single_analysis(aggregate, os.path.join(root_dir, "primary"), title="Primary condition")

    if compare_a and compare_b:
        tables_a = _load_tables(compare_a)
        tables_b = _load_tables(compare_b)
        agg_a = aggregate_circuit_tables(tables_a)
        agg_b = aggregate_circuit_tables(tables_b)
        summary_a = save_single_analysis(agg_a, os.path.join(root_dir, slugify(args.label_a)), title=args.label_a)
        summary_b = save_single_analysis(agg_b, os.path.join(root_dir, slugify(args.label_b)), title=args.label_b)
        compare = compare_shared_backbone(agg_a, agg_b, topk=args.compare_topk)
        save_csv(compare["overlap_by_layer"], os.path.join(root_dir, "shared_backbone_overlap_by_layer.csv"))
        compare_payload = dict(compare)
        compare_payload["overlap_by_layer"] = compare["overlap_by_layer"].to_dict(orient="records")
        compare_payload["summary_a"] = summary_a
        compare_payload["summary_b"] = summary_b
        save_json(compare_payload, os.path.join(root_dir, "shared_backbone_summary.json"))
        plot_shared_backbone(
            compare,
            os.path.join(root_dir, "shared_backbone.png"),
            label_a=args.label_a,
            label_b=args.label_b,
        )


if __name__ == "__main__":
    main()
