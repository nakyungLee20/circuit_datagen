import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None


_CURRENT_DIR = Path(__file__).resolve().parent
_EAP_SRC = _CURRENT_DIR.parent / "EAP-IG" / "src"

Graph = None
if _EAP_SRC.is_dir():
    import sys

    if str(_EAP_SRC) not in sys.path:
        sys.path.insert(0, str(_EAP_SRC))
    try:
        from eap.graph import Graph  # type: ignore
    except Exception:
        Graph = None


EDGE_RE = re.compile(r"([^>]+)->([^<]+)(?:<([qkv])(?::kv\d+)?>)?")
LAYER_PATTERNS = [
    re.compile(r"^a(\d+)\.h\d+$"),
    re.compile(r"^a(\d+)\.kv\d+$"),
    re.compile(r"^m(\d+)$"),
    re.compile(r"blocks\.(\d+)\."),
    re.compile(r"\blayer[_\.](\d+)"),
    re.compile(r"\bL(\d+)\b"),
]
HEAD_PATTERNS = [
    re.compile(r"\bH(\d+)\b"),
    re.compile(r"head[_\.](\d+)"),
    re.compile(r"\.h(\d+)\b"),
]
SUPPORTED_SUFFIXES = {".pt", ".json", ".jsonl"}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: object, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\-\.]+", "_", (text or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def parse_edge_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(name, str):
        return None, None, None
    match = EDGE_RE.match(name)
    if not match:
        return None, None, None
    return match.group(1), match.group(2), match.group(3)


def parse_layer(name: object) -> Optional[int]:
    if not isinstance(name, str):
        return None
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def parse_head(name: object) -> Optional[int]:
    if not isinstance(name, str):
        return None
    for pattern in HEAD_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def node_kind(name: object) -> str:
    if not isinstance(name, str):
        return "other"
    lower = name.lower()
    if re.match(r"^a\d+\.(h\d+|kv\d+)$", lower):
        return "attn"
    if re.match(r"^m\d+$", lower):
        return "mlp"
    if lower in {"input", "embed", "hook_embed"}:
        return "embed"
    if lower in {"logits", "unembed", "lm_head"}:
        return "unembed"
    if "attn" in lower or "attention" in lower:
        return "attn"
    if "mlp" in lower or "ff" in lower:
        return "mlp"
    if "embed" in lower:
        return "embed"
    if "unembed" in lower or "lm_head" in lower:
        return "unembed"
    if "resid" in lower or "residual" in lower:
        return "resid"
    return "other"


def is_attn_node(name: object) -> bool:
    return node_kind(name) == "attn"


def is_mlp_node(name: object) -> bool:
    return node_kind(name) == "mlp"


def display_layer(name: object, max_layer: Optional[int] = None) -> int:
    kind = node_kind(name)
    layer = parse_layer(name)
    if kind == "embed":
        return -1
    if kind == "unembed":
        return (max_layer if max_layer is not None else -1) + 1
    if layer is not None:
        return layer
    if kind == "other":
        return (max_layer if max_layer is not None else 0) + 1
    return 0


def display_layer_label(layer: int, max_layer: Optional[int]) -> str:
    if layer == -1:
        return "input"
    if max_layer is not None and layer == max_layer + 1:
        return "logits"
    return f"L{layer}"


def resolve_input_paths(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if not item:
            continue
        matches: List[str] = []
        if any(ch in item for ch in "*?[]"):
            matches = sorted(glob.glob(item))
        elif os.path.isdir(item):
            for suffix in sorted(SUPPORTED_SUFFIXES):
                matches.extend(sorted(glob.glob(os.path.join(item, f"**/*{suffix}"), recursive=True)))
        else:
            matches = [item]
        for match in matches:
            if os.path.isfile(match) and Path(match).suffix in SUPPORTED_SUFFIXES:
                out.append(os.path.abspath(match))
    seen = set()
    uniq: List[str] = []
    for path in out:
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


def _load_edges_from_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            edge = record.get("edge")
            src = record.get("src")
            dst = record.get("dst")
            score = safe_float(record.get("score"))
            if (src is None or dst is None) and isinstance(edge, str):
                src2, dst2, qkv = parse_edge_name(edge)
                src = src or src2
                dst = dst or dst2
                if qkv is not None:
                    record["qkv"] = qkv
            if edge is None and src is not None and dst is not None:
                edge = f"{src}->{dst}"
            rows.append(
                {
                    "edge": edge,
                    "src": src,
                    "dst": dst,
                    "score": score,
                    "in_graph": record.get("in_graph", record.get("included", True)),
                    "qkv": record.get("qkv"),
                }
            )
    return pd.DataFrame(rows)


def _load_edges_from_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("edges"), dict):
        rows: List[Dict[str, object]] = []
        for edge_name, info in payload["edges"].items():
            info = info or {}
            src = info.get("parent")
            dst = info.get("child")
            qkv = info.get("qkv")
            if (src is None or dst is None) and isinstance(edge_name, str):
                src2, dst2, qkv2 = parse_edge_name(edge_name)
                src = src or src2
                dst = dst or dst2
                qkv = qkv or qkv2
            rows.append(
                {
                    "edge": edge_name,
                    "src": src,
                    "dst": dst,
                    "score": safe_float(info.get("score")),
                    "in_graph": info.get("in_graph", True),
                    "qkv": qkv,
                }
            )
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported JSON circuit format: {path}")


def _load_edges_from_pt_graph(path: str, included_only: bool) -> pd.DataFrame:
    graph = Graph.from_pt(path)
    rows: List[Dict[str, object]] = []
    edges = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
    for edge in edges:
        included = bool(edge.in_graph)
        if included_only and not included:
            continue
        rows.append(
            {
                "edge": getattr(edge, "name", None) or str(edge),
                "src": getattr(edge.parent, "name", None),
                "dst": getattr(edge.child, "name", None),
                "score": safe_float(edge.score),
                "in_graph": included,
                "qkv": getattr(edge, "qkv", None),
            }
        )
    return pd.DataFrame(rows)


def _dst_slot_base_name(dst_name: str) -> str:
    if not isinstance(dst_name, str):
        return str(dst_name)
    return dst_name.split("<", 1)[0]


def _load_edges_from_pt_raw(path: str, included_only: bool) -> pd.DataFrame:
    if torch is None:
        raise RuntimeError("torch is not installed; cannot load .pt circuit files.")
    payload = torch.load(path, map_location="cpu")
    src_nodes = payload["src_nodes"]
    dst_nodes = payload["dst_nodes"]
    dst_meta = payload.get("dst_nodes_metadata")
    scores = payload["edges_scores"]
    in_graph = payload["edges_in_graph"]
    rows: List[Dict[str, object]] = []
    for src_index, src in enumerate(src_nodes):
        for dst_index, dst_name in enumerate(dst_nodes):
            included = bool(in_graph[src_index, dst_index])
            if included_only and not included:
                continue
            dst = _dst_slot_base_name(dst_name)
            qkv = None
            if dst_meta and dst_index < len(dst_meta):
                qkv = dst_meta[dst_index].get("qkv")
            elif isinstance(dst_name, str) and "<" in dst_name:
                _, _, qkv = parse_edge_name(f"{src}->{dst_name}")
            edge_name = f"{src}->{dst}" if qkv is None else f"{src}->{dst}<{qkv}>"
            rows.append(
                {
                    "edge": edge_name,
                    "src": src,
                    "dst": dst,
                    "score": safe_float(scores[src_index, dst_index]),
                    "in_graph": included,
                    "qkv": qkv,
                }
            )
    return pd.DataFrame(rows)


def load_circuit_table(path: str, included_only: bool = True) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        df = _load_edges_from_jsonl(path)
    elif suffix == ".json":
        df = _load_edges_from_json(path)
    elif suffix == ".pt":
        if Graph is not None:
            try:
                df = _load_edges_from_pt_graph(path, included_only=included_only)
            except Exception:
                df = _load_edges_from_pt_raw(path, included_only=included_only)
        else:
            df = _load_edges_from_pt_raw(path, included_only=included_only)
    else:
        raise ValueError(f"Unsupported circuit suffix: {path}")

    return finalize_edges_df(df, source_path=path)


def finalize_edges_df(df: pd.DataFrame, source_path: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "edge",
                "src",
                "dst",
                "score",
                "abs_score",
                "qkv",
                "src_layer",
                "dst_layer",
                "src_plot_layer",
                "dst_plot_layer",
                "src_head",
                "dst_head",
                "src_kind",
                "dst_kind",
                "source_path",
            ]
        )
    out = df.copy()
    if "src" not in out.columns or "dst" not in out.columns:
        src_dst = out["edge"].map(parse_edge_name)
        out["src"] = out.get("src", pd.Series(index=out.index)).fillna(src_dst.map(lambda x: x[0]))
        out["dst"] = out.get("dst", pd.Series(index=out.index)).fillna(src_dst.map(lambda x: x[1]))
    if "qkv" not in out.columns:
        out["qkv"] = out["edge"].map(lambda x: parse_edge_name(x)[2] if isinstance(x, str) else None)
    else:
        missing_qkv = out["qkv"].isna()
        out.loc[missing_qkv, "qkv"] = out.loc[missing_qkv, "edge"].map(
            lambda x: parse_edge_name(x)[2] if isinstance(x, str) else None
        )
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"]).copy()
    if "abs_score" not in out.columns:
        out["abs_score"] = out["score"].abs()
    else:
        out["abs_score"] = pd.to_numeric(out["abs_score"], errors="coerce")
        out["abs_score"] = out["abs_score"].fillna(out["score"].abs())
    out["src_layer"] = out["src"].map(parse_layer)
    out["dst_layer"] = out["dst"].map(parse_layer)
    max_layer = pd.concat([out["src_layer"], out["dst_layer"]], axis=0).dropna()
    max_layer_value = int(max_layer.max()) if not max_layer.empty else 0
    out["src_plot_layer"] = out["src"].map(lambda x: display_layer(x, max_layer_value))
    out["dst_plot_layer"] = out["dst"].map(lambda x: display_layer(x, max_layer_value))
    out["src_head"] = out["src"].map(parse_head)
    out["dst_head"] = out["dst"].map(parse_head)
    out["src_kind"] = out["src"].map(node_kind)
    out["dst_kind"] = out["dst"].map(node_kind)
    out["source_path"] = source_path or out.get("source_path", "")
    if "in_graph" in out.columns:
        out["in_graph"] = out["in_graph"].fillna(True).astype(bool)
    out = out.sort_values("abs_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def aggregate_circuit_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    zero_fill_missing: bool = True,
) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    names = list(tables.keys())
    meta: Dict[str, Dict[str, object]] = {}
    score_maps: Dict[str, Dict[str, float]] = {}
    abs_maps: Dict[str, Dict[str, float]] = {}
    for name, table in tables.items():
        if table is None or table.empty:
            score_maps[name] = {}
            abs_maps[name] = {}
            continue
        score_maps[name] = dict(zip(table["edge"], table["score"]))
        abs_maps[name] = dict(zip(table["edge"], table["abs_score"]))
        for _, row in table.iterrows():
            edge = row["edge"]
            if edge not in meta:
                meta[edge] = {
                    "src": row.get("src"),
                    "dst": row.get("dst"),
                    "qkv": row.get("qkv"),
                }

    edges = sorted(meta.keys())
    if not edges:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for edge in edges:
        signed_values: List[float] = []
        abs_values: List[float] = []
        present = 0
        for name in names:
            if edge in score_maps[name]:
                present += 1
                signed_values.append(float(score_maps[name][edge]))
                abs_values.append(float(abs_maps[name][edge]))
            elif zero_fill_missing:
                signed_values.append(0.0)
                abs_values.append(0.0)
        rows.append(
            {
                "edge": edge,
                "src": meta[edge]["src"],
                "dst": meta[edge]["dst"],
                "qkv": meta[edge]["qkv"],
                "score": float(np.mean(signed_values)) if signed_values else float("nan"),
                "abs_score": float(np.mean(abs_values)) if abs_values else float("nan"),
                "n_present": present,
                "presence_frac": present / max(len(names), 1),
            }
        )
    out = finalize_edges_df(pd.DataFrame(rows))
    if not out.empty:
        out["n_runs"] = len(names)
    return out


def top_edges(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    topk = max(1, int(topk))
    return df.sort_values("abs_score", ascending=False).head(topk).copy()


def edge_score_map(df: pd.DataFrame, column: str = "score") -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    return {str(edge): float(score) for edge, score in zip(df["edge"], df[column])}


def node_importance_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "node",
                "in_edges",
                "out_edges",
                "fan_in_abs",
                "fan_out_abs",
                "fan_in_score",
                "fan_out_score",
                "total_abs",
                "total_score",
                "layer",
                "head",
                "kind",
            ]
        )
    src = (
        df.groupby("src", dropna=False)
        .agg(
            out_edges=("edge", "count"),
            fan_out_abs=("abs_score", "sum"),
            fan_out_score=("score", "sum"),
        )
        .reset_index()
        .rename(columns={"src": "node"})
    )
    dst = (
        df.groupby("dst", dropna=False)
        .agg(
            in_edges=("edge", "count"),
            fan_in_abs=("abs_score", "sum"),
            fan_in_score=("score", "sum"),
        )
        .reset_index()
        .rename(columns={"dst": "node"})
    )
    out = src.merge(dst, on="node", how="outer")
    for column in ["in_edges", "out_edges", "fan_in_abs", "fan_out_abs", "fan_in_score", "fan_out_score"]:
        out[column] = out[column].fillna(0.0)
    out["in_edges"] = out["in_edges"].astype(int)
    out["out_edges"] = out["out_edges"].astype(int)
    out["total_abs"] = out["fan_in_abs"] + out["fan_out_abs"]
    out["total_score"] = out["fan_in_score"] + out["fan_out_score"]
    out["layer"] = out["node"].map(parse_layer)
    out["head"] = out["node"].map(parse_head)
    out["kind"] = out["node"].map(node_kind)
    out["plot_layer"] = out["layer"]
    if out["layer"].notna().any():
        max_layer_value = int(out["layer"].dropna().max())
    else:
        max_layer_value = 0
    out["plot_layer"] = out["node"].map(lambda x: display_layer(x, max_layer_value))
    out = out.sort_values("total_abs", ascending=False).reset_index(drop=True)
    return out


def layer_mass_matrix(df: pd.DataFrame, value_col: str = "abs_score") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    matrix = (
        df.groupby(["src_plot_layer", "dst_plot_layer"], dropna=False)[value_col]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return matrix


def qkv_layer_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["dst_plot_layer", "qkv", "abs_score"])
    attn = df[df["qkv"].notna()].copy()
    if attn.empty:
        return pd.DataFrame(columns=["dst_plot_layer", "qkv", "abs_score"])
    out = (
        attn.groupby(["dst_plot_layer", "qkv"], dropna=False)["abs_score"]
        .sum()
        .reset_index()
        .sort_values(["dst_plot_layer", "qkv"])
    )
    return out


def pairwise_jaccard(sets: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    names = list(sets.keys())
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, name_i in enumerate(names):
        set_i = set(sets[name_i])
        for j, name_j in enumerate(names):
            set_j = set(sets[name_j])
            denom = len(set_i | set_j)
            matrix[i, j] = 1.0 if denom == 0 else len(set_i & set_j) / denom
    return pd.DataFrame(matrix, index=names, columns=names)


def edge_presence_table(tables: Mapping[str, pd.DataFrame], topk: int) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame(columns=["edge", "count", "frac"])
    sets = {name: set(top_edges(df, topk)["edge"]) for name, df in tables.items()}
    counts: Dict[str, int] = {}
    for edge_set in sets.values():
        for edge in edge_set:
            counts[edge] = counts.get(edge, 0) + 1
    rows = [{"edge": edge, "count": count, "frac": count / len(sets)} for edge, count in counts.items()]
    return pd.DataFrame(rows).sort_values(["count", "edge"], ascending=[False, True]).reset_index(drop=True)


def build_tag_from_paths(paths: Sequence[str], default: str = "circuit") -> str:
    if not paths:
        return default
    stems = [Path(path).stem for path in paths[:3]]
    tag = "_".join(slugify(stem) for stem in stems if stem)
    if len(paths) > 3:
        tag = f"{tag}_{len(paths)}runs"
    return tag or default


def top_edge_set(df: pd.DataFrame, topk: int) -> set:
    if df is None or df.empty:
        return set()
    return set(top_edges(df, topk)["edge"])


def normalized_entropy(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float("nan")
    probs = arr / arr.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / math.log(arr.size)) if arr.size > 1 else 0.0

