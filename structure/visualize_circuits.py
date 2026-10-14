import argparse
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from _circuit_utils import (
    aggregate_circuit_tables,
    build_tag_from_paths,
    display_layer_label,
    edge_presence_table,
    ensure_dir,
    layer_mass_matrix,
    load_circuit_table,
    node_importance_table,
    pairwise_jaccard,
    qkv_layer_table,
    resolve_input_paths,
    save_csv,
    save_json,
    slugify,
    top_edge_set,
    top_edges,
)


DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parents[1] / 'results/structure')
NODE_COLORS = {
    "embed": "#e9c46a",
    "attn": "#2a9d8f",
    "mlp": "#f4a261",
    "unembed": "#264653",
    "resid": "#8ab17d",
    "other": "#c9c9c9",
}
EDGE_COLORS = {
    "q": "#d81b60",
    "k": "#1e88e5",
    "v": "#43a047",
    None: "#4d4d4d",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal EAP-IG circuit visualization toolkit.")
    ap.add_argument("--circuits", nargs="*", default=[], help="Single condition circuits for base plots and optional stability.")
    ap.add_argument("--compare-a", nargs="*", default=[], help="Condition A circuits for delta plots.")
    ap.add_argument("--compare-b", nargs="*", default=[], help="Condition B circuits for delta plots.")
    ap.add_argument("--label-a", type=str, default="condition_a")
    ap.add_argument("--label-b", type=str, default="condition_b")
    ap.add_argument("--topk", type=int, default=40, help="Top-k edges for the circuit graph.")
    ap.add_argument("--stability-topk", type=int, default=100, help="Top-k edges used in stability overlap.")
    ap.add_argument("--delta-topk", type=int, default=25, help="Number of differential edges to show.")
    ap.add_argument("--out-root", type=str, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--tag", type=str, default="", help="Optional result subdirectory name.")
    return ap.parse_args()


def _load_tables(paths: Sequence[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for idx, path in enumerate(paths):
        label = f"{Path(path).stem}_{idx}" if Path(path).stem in tables else Path(path).stem
        tables[label] = load_circuit_table(path, included_only=True)
    return tables


def _max_layer_from_df(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    candidates = pd.concat([df["src_layer"], df["dst_layer"]], axis=0)
    candidates = candidates[np.isfinite(candidates)]
    if candidates.empty:
        return 0
    finite = [int(x) for x in candidates if int(x) >= 0]
    if not finite:
        return 0
    return max(finite)


def _setup_figure(nrows: int = 1, ncols: int = 1, figsize: Tuple[float, float] = (10, 6)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor("white")
    return fig, axes


def _save_figure(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, matrix: pd.DataFrame, *, cmap: str, title: str, cbar_label: str, center: Optional[float] = None) -> None:
    values = matrix.to_numpy(dtype=float)
    if values.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    if center is not None:
        vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
        im.set_clim(vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    annotate = values.shape[0] * values.shape[1] <= 100
    if annotate:
        vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                color = "white" if vmax > 0 and abs(value) > 0.55 * vmax else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def _build_layer_labels(matrix: pd.DataFrame, max_layer: int) -> pd.DataFrame:
    relabeled = matrix.copy()
    relabeled.index = [display_layer_label(int(layer), max_layer) for layer in relabeled.index]
    relabeled.columns = [display_layer_label(int(layer), max_layer) for layer in relabeled.columns]
    return relabeled


def plot_topk_graph(df: pd.DataFrame, out_dir: str, *, title: str, topk: int) -> None:
    top_df = top_edges(df, topk)
    csv_path = os.path.join(out_dir, "topk_graph_edges.csv")
    save_csv(top_df, csv_path)
    if top_df.empty:
        return

    nodes = node_importance_table(top_df)
    node_csv = os.path.join(out_dir, "topk_graph_nodes.csv")
    save_csv(nodes, node_csv)
    max_layer = _max_layer_from_df(top_df)

    layer_groups: Dict[int, List[dict]] = {}
    for _, row in nodes.iterrows():
        layer = int(row["plot_layer"])
        layer_groups.setdefault(layer, []).append(row.to_dict())

    kind_order = {"embed": 0, "attn": 1, "mlp": 2, "unembed": 3, "other": 4, "resid": 5}
    positions: Dict[str, Tuple[float, float]] = {}
    sorted_layers = sorted(layer_groups)
    max_nodes_in_layer = max(len(items) for items in layer_groups.values())
    fig_w = max(11.0, 2.8 * len(sorted_layers) + 2.0)
    fig_h = max(6.0, 0.65 * max_nodes_in_layer + 2.0)
    fig, axes = _setup_figure(figsize=(fig_w, fig_h))
    ax = axes[0][0]

    for layer_idx, layer in enumerate(sorted_layers):
        layer_nodes = sorted(
            layer_groups[layer],
            key=lambda row: (kind_order.get(str(row["kind"]), 99), -float(row["total_abs"])),
        )
        ys = np.linspace(0.92, 0.08, len(layer_nodes)) if len(layer_nodes) > 1 else np.array([0.5])
        x = float(layer_idx)
        for y, row in zip(ys, layer_nodes):
            positions[str(row["node"])] = (x, float(y))

    max_abs = float(top_df["abs_score"].max()) if not top_df.empty else 1.0
    for _, row in top_df.sort_values("abs_score", ascending=True).iterrows():
        src = row["src"]
        dst = row["dst"]
        if src not in positions or dst not in positions:
            continue
        src_pos = positions[src]
        dst_pos = positions[dst]
        linewidth = 0.8 + 4.2 * float(row["abs_score"]) / max(max_abs, 1e-8)
        rad = 0.08 if abs(src_pos[0] - dst_pos[0]) <= 1 else 0.0
        patch = FancyArrowPatch(
            src_pos,
            dst_pos,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=EDGE_COLORS.get(row.get("qkv"), EDGE_COLORS[None]),
            alpha=0.75,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=14,
            shrinkB=14,
        )
        ax.add_patch(patch)

    for node_name, (x, y) in positions.items():
        node_row = nodes[nodes["node"] == node_name].iloc[0]
        facecolor = NODE_COLORS.get(str(node_row["kind"]), NODE_COLORS["other"])
        text_color = "white" if str(node_row["kind"]) == "unembed" else "black"
        ax.text(
            x,
            y,
            node_name,
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=facecolor, edgecolor="black", linewidth=0.9),
        )

    for layer_idx, layer in enumerate(sorted_layers):
        ax.text(layer_idx, 1.02, display_layer_label(layer, max_layer), ha="center", va="bottom", fontsize=10, fontweight="bold")

    node_handles = [Patch(facecolor=color, edgecolor="black", label=kind) for kind, color in NODE_COLORS.items() if kind in {"embed", "attn", "mlp", "unembed"}]
    edge_handles = [
        Line2D([0], [0], color=EDGE_COLORS[qkv], linewidth=2.4, label=f"{qkv or 'direct'} edge")
        for qkv in ["q", "k", "v", None]
    ]
    ax.legend(handles=node_handles + edge_handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    ax.set_title(f"{title}\nTop-{min(topk, len(top_df))} circuit graph", fontsize=13)
    ax.set_xlim(-0.6, len(sorted_layers) - 0.4)
    ax.set_ylim(-0.04, 1.08)
    ax.axis("off")
    _save_figure(fig, os.path.join(out_dir, "topk_graph.png"))


def plot_layer_heatmap(df: pd.DataFrame, out_dir: str, *, title: str) -> None:
    matrix = layer_mass_matrix(df, value_col="abs_score")
    save_csv(matrix.reset_index().rename(columns={"src_plot_layer": "layer"}), os.path.join(out_dir, "layer_to_layer_abs_mass.csv"))
    relabeled = _build_layer_labels(matrix, _max_layer_from_df(df))
    fig, axes = _setup_figure(figsize=(8.5, 6.5))
    _heatmap(
        axes[0][0],
        relabeled,
        cmap="magma",
        title=f"{title}\nLayer-to-layer score mass",
        cbar_label="sum(|score|)",
    )
    _save_figure(fig, os.path.join(out_dir, "layer_to_layer_heatmap.png"))


def plot_node_importance(df: pd.DataFrame, out_dir: str, *, title: str) -> None:
    nodes = node_importance_table(df)
    save_csv(nodes, os.path.join(out_dir, "node_importance.csv"))
    fig, axes = _setup_figure(nrows=1, ncols=3, figsize=(18, 5.5))

    attn = nodes[(nodes["kind"] == "attn") & nodes["head"].notna()].copy()
    if attn.empty:
        axes[0][0].text(0.5, 0.5, "No attention nodes", ha="center", va="center")
        axes[0][0].axis("off")
    else:
        attn_heat = (
            attn.pivot_table(index="layer", columns="head", values="total_abs", aggfunc="sum", fill_value=0.0)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        _heatmap(
            axes[0][0],
            attn_heat,
            cmap="YlGnBu",
            title="Attention head importance",
            cbar_label="incoming + outgoing |score|",
        )

    mlp = nodes[nodes["kind"] == "mlp"].copy()
    if mlp.empty:
        axes[0][1].text(0.5, 0.5, "No MLP nodes", ha="center", va="center")
        axes[0][1].axis("off")
    else:
        mlp = mlp.sort_values("layer")
        axes[0][1].bar(mlp["layer"].astype(int), mlp["total_abs"], color=NODE_COLORS["mlp"], edgecolor="black")
        axes[0][1].set_title("MLP importance by layer")
        axes[0][1].set_xlabel("Layer")
        axes[0][1].set_ylabel("incoming + outgoing |score|")

    axes[0][2].set_title("Bottleneck map: fan-in vs fan-out")
    for kind in ["attn", "mlp", "embed", "unembed", "other"]:
        sub = nodes[nodes["kind"] == kind]
        if sub.empty:
            continue
        axes[0][2].scatter(
            sub["fan_in_abs"],
            sub["fan_out_abs"],
            s=24 + 260 * sub["total_abs"] / max(nodes["total_abs"].max(), 1e-8),
            color=NODE_COLORS.get(kind, NODE_COLORS["other"]),
            alpha=0.72,
            edgecolor="black",
            linewidth=0.5,
            label=kind,
        )
    for _, row in nodes.head(min(10, len(nodes))).iterrows():
        axes[0][2].annotate(str(row["node"]), (row["fan_in_abs"], row["fan_out_abs"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axes[0][2].set_xlabel("fan-in |score|")
    axes[0][2].set_ylabel("fan-out |score|")
    axes[0][2].legend(frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.02)
    _save_figure(fig, os.path.join(out_dir, "node_importance_map.png"))


def plot_qkv_breakdown(df: pd.DataFrame, out_dir: str, *, title: str) -> None:
    table = qkv_layer_table(df)
    save_csv(table, os.path.join(out_dir, "qkv_breakdown.csv"))
    fig, axes = _setup_figure(nrows=1, ncols=2, figsize=(15, 5.5))
    if table.empty:
        for ax in axes[0]:
            ax.text(0.5, 0.5, "No q/k/v edges", ha="center", va="center")
            ax.axis("off")
        fig.suptitle(title, fontsize=13)
        _save_figure(fig, os.path.join(out_dir, "qkv_breakdown.png"))
        return

    pivot = (
        table.pivot_table(index="dst_plot_layer", columns="qkv", values="abs_score", aggfunc="sum", fill_value=0.0)
        .reindex(columns=["q", "k", "v"], fill_value=0.0)
        .sort_index()
    )
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    for qkv in ["q", "k", "v"]:
        values = pivot[qkv].to_numpy(dtype=float)
        axes[0][0].bar(x, values, bottom=bottom, color=EDGE_COLORS[qkv], edgecolor="black", label=qkv)
        bottom += values
    axes[0][0].set_xticks(x)
    axes[0][0].set_xticklabels([display_layer_label(int(layer), _max_layer_from_df(df)) for layer in pivot.index], rotation=45, ha="right")
    axes[0][0].set_title("Per-layer q/k/v mass")
    axes[0][0].set_ylabel("sum(|score|)")
    axes[0][0].legend(frameon=False)

    totals = table.groupby("qkv", dropna=False)["abs_score"].sum().reindex(["q", "k", "v"], fill_value=0.0)
    axes[0][1].bar(totals.index, totals.values, color=[EDGE_COLORS[qkv] for qkv in totals.index], edgecolor="black")
    axes[0][1].set_title("Overall q/k/v mass")
    axes[0][1].set_ylabel("sum(|score|)")

    fig.suptitle(title, fontsize=13, y=1.02)
    _save_figure(fig, os.path.join(out_dir, "qkv_breakdown.png"))


def _build_delta_table(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    a_signed = dict(zip(df_a["edge"], df_a["score"]))
    a_abs = dict(zip(df_a["edge"], df_a["abs_score"]))
    b_signed = dict(zip(df_b["edge"], df_b["score"]))
    b_abs = dict(zip(df_b["edge"], df_b["abs_score"]))
    meta = {}
    for df in [df_a, df_b]:
        for _, row in df.iterrows():
            edge = row["edge"]
            if edge not in meta:
                meta[edge] = {"src": row["src"], "dst": row["dst"], "qkv": row["qkv"]}
    rows = []
    for edge in sorted(meta):
        rows.append(
            {
                "edge": edge,
                "src": meta[edge]["src"],
                "dst": meta[edge]["dst"],
                "qkv": meta[edge]["qkv"],
                "score_a": float(a_signed.get(edge, 0.0)),
                "score_b": float(b_signed.get(edge, 0.0)),
                "abs_a": float(a_abs.get(edge, 0.0)),
                "abs_b": float(b_abs.get(edge, 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    out["delta_score"] = out["score_a"] - out["score_b"]
    out["delta_abs"] = out["abs_a"] - out["abs_b"]
    out["abs_delta_abs"] = out["delta_abs"].abs()
    return out.sort_values("abs_delta_abs", ascending=False).reset_index(drop=True)


def plot_condition_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    *,
    label_a: str,
    label_b: str,
    topk: int,
) -> None:
    delta = _build_delta_table(df_a, df_b)
    save_csv(delta, os.path.join(out_dir, "condition_delta_edges.csv"))
    top_delta = delta.head(max(4, topk)).copy()
    if top_delta.empty:
        return

    matrix_a = layer_mass_matrix(df_a, value_col="abs_score")
    matrix_b = layer_mass_matrix(df_b, value_col="abs_score")
    all_rows = sorted(set(matrix_a.index) | set(matrix_b.index))
    all_cols = sorted(set(matrix_a.columns) | set(matrix_b.columns))
    matrix_a = matrix_a.reindex(index=all_rows, columns=all_cols, fill_value=0.0)
    matrix_b = matrix_b.reindex(index=all_rows, columns=all_cols, fill_value=0.0)
    delta_matrix = matrix_a - matrix_b
    save_csv(delta_matrix.reset_index().rename(columns={"src_plot_layer": "layer"}), os.path.join(out_dir, "condition_delta_layer_matrix.csv"))

    fig, axes = _setup_figure(nrows=1, ncols=2, figsize=(18, 6.5))
    bar_df = top_delta.sort_values("delta_abs")
    colors = ["#c0392b" if value > 0 else "#2166ac" for value in bar_df["delta_abs"]]
    axes[0][0].barh(bar_df["edge"], bar_df["delta_abs"], color=colors, edgecolor="black")
    axes[0][0].axvline(0.0, color="black", linewidth=1.0)
    axes[0][0].set_title(f"Edges stronger in {label_a} vs {label_b}")
    axes[0][0].set_xlabel("mean |score| delta")

    relabeled = _build_layer_labels(delta_matrix, max(_max_layer_from_df(df_a), _max_layer_from_df(df_b)))
    _heatmap(
        axes[0][1],
        relabeled,
        cmap="coolwarm",
        title=f"Layer delta: {label_a} - {label_b}",
        cbar_label="delta sum(|score|)",
        center=0.0,
    )
    fig.suptitle(f"Condition delta: {label_a} vs {label_b}", fontsize=13, y=1.02)
    _save_figure(fig, os.path.join(out_dir, "condition_delta.png"))


def plot_stability(
    tables: Mapping[str, pd.DataFrame],
    out_dir: str,
    *,
    title: str,
    topk: int,
) -> None:
    if len(tables) <= 1:
        return
    edge_sets = {name: top_edge_set(df, topk) for name, df in tables.items()}
    overlap = pairwise_jaccard(edge_sets)
    presence = edge_presence_table(tables, topk)
    save_csv(overlap.reset_index().rename(columns={"index": "run"}), os.path.join(out_dir, "stability_pairwise_jaccard.csv"))
    save_csv(presence, os.path.join(out_dir, "stability_edge_presence.csv"))

    fig, axes = _setup_figure(nrows=1, ncols=2, figsize=(16, 6))
    _heatmap(
        axes[0][0],
        overlap,
        cmap="viridis",
        title=f"{title}\nPairwise top-{topk} overlap",
        cbar_label="Jaccard",
    )

    if presence.empty:
        axes[0][1].text(0.5, 0.5, "No overlap data", ha="center", va="center")
        axes[0][1].axis("off")
    else:
        curve = presence["frac"].to_numpy(dtype=float)
        axes[0][1].plot(np.arange(1, len(curve) + 1), curve, color="#2a9d8f", linewidth=2.2)
        axes[0][1].fill_between(np.arange(1, len(curve) + 1), curve, color="#2a9d8f", alpha=0.18)
        axes[0][1].set_title("Stable edge frequency curve")
        axes[0][1].set_xlabel("Edge rank by recurrence")
        axes[0][1].set_ylabel("Fraction of runs containing edge")
        axes[0][1].set_ylim(0.0, 1.02)
        if len(curve) > 0:
            axes[0][1].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    _save_figure(fig, os.path.join(out_dir, "stability.png"))


def _base_visualizations(df: pd.DataFrame, out_dir: str, *, title: str, topk: int) -> None:
    ensure_dir(out_dir)
    save_csv(df, os.path.join(out_dir, "aggregate_edges.csv"))
    plot_topk_graph(df, out_dir, title=title, topk=topk)
    plot_layer_heatmap(df, out_dir, title=title)
    plot_node_importance(df, out_dir, title=title)
    plot_qkv_breakdown(df, out_dir, title=title)


def main() -> None:
    args = parse_args()
    circuits = resolve_input_paths(args.circuits)
    compare_a = resolve_input_paths(args.compare_a)
    compare_b = resolve_input_paths(args.compare_b)
    if not circuits and not (compare_a and compare_b):
        raise SystemExit("Provide either --circuits or both --compare-a and --compare-b.")

    tag_paths = circuits or (compare_a + compare_b)
    tag = slugify(args.tag) if args.tag else build_tag_from_paths(tag_paths, default="visualization")
    root_dir = ensure_dir(os.path.join(args.out_root, tag, "visualizations"))

    manifest = {
        "circuits": circuits,
        "compare_a": compare_a,
        "compare_b": compare_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "topk": args.topk,
        "stability_topk": args.stability_topk,
        "delta_topk": args.delta_topk,
    }
    save_json(manifest, os.path.join(root_dir, "manifest.json"))

    if circuits:
        tables = _load_tables(circuits)
        aggregate = aggregate_circuit_tables(tables)
        base_dir = ensure_dir(os.path.join(root_dir, "primary"))
        _base_visualizations(aggregate, base_dir, title="Primary condition", topk=args.topk)
        if len(tables) > 1:
            plot_stability(tables, base_dir, title="Primary condition stability", topk=args.stability_topk)

    if compare_a and compare_b:
        tables_a = _load_tables(compare_a)
        tables_b = _load_tables(compare_b)
        agg_a = aggregate_circuit_tables(tables_a)
        agg_b = aggregate_circuit_tables(tables_b)
        dir_a = ensure_dir(os.path.join(root_dir, slugify(args.label_a)))
        dir_b = ensure_dir(os.path.join(root_dir, slugify(args.label_b)))
        _base_visualizations(agg_a, dir_a, title=args.label_a, topk=args.topk)
        _base_visualizations(agg_b, dir_b, title=args.label_b, topk=args.topk)
        if len(tables_a) > 1:
            plot_stability(tables_a, dir_a, title=f"{args.label_a} stability", topk=args.stability_topk)
        if len(tables_b) > 1:
            plot_stability(tables_b, dir_b, title=f"{args.label_b} stability", topk=args.stability_topk)
        plot_condition_delta(
            agg_a,
            agg_b,
            root_dir,
            label_a=args.label_a,
            label_b=args.label_b,
            topk=args.delta_topk,
        )


if __name__ == "__main__":
    main()
