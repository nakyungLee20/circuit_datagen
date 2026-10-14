from typing import Any, Dict, List, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from .graph import Edge, Graph


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_gqa_info(graph_or_cfg: Any) -> Dict[str, Any]:
    cfg = graph_or_cfg.cfg if hasattr(graph_or_cfg, "cfg") else graph_or_cfg

    n_heads = int(_cfg_get(cfg, "n_heads"))
    effective_n_kv_heads = int(_cfg_get(cfg, "n_kv_heads", n_heads))
    raw_n_kv_heads = int(_cfg_get(cfg, "n_key_value_heads", effective_n_kv_heads))
    ungroup = bool(_cfg_get(cfg, "ungroup_grouped_query_attention", False))
    has_grouped_query_attention = bool(
        _cfg_get(cfg, "has_grouped_query_attention", raw_n_kv_heads != n_heads)
    )
    uses_native_gqa = bool(
        _cfg_get(
            cfg,
            "uses_native_gqa",
            has_grouped_query_attention and effective_n_kv_heads != n_heads and not ungroup,
        )
    )

    return {
        "n_heads": n_heads,
        "n_kv_heads": effective_n_kv_heads,
        "n_key_value_heads": raw_n_kv_heads,
        "ungroup_grouped_query_attention": ungroup,
        "has_grouped_query_attention": has_grouped_query_attention,
        "uses_native_gqa": uses_native_gqa,
        "heads_per_kv": None if effective_n_kv_heads == 0 else n_heads // effective_n_kv_heads,
    }


def format_dst_slot_name(
    layer: int,
    qkv: str,
    *,
    head: Optional[int] = None,
    kv_head: Optional[int] = None,
    uses_native_gqa: bool = False,
) -> str:
    if qkv not in {"q", "k", "v"}:
        raise ValueError(f"Unsupported qkv value: {qkv}")

    if uses_native_gqa and qkv in {"k", "v"}:
        if kv_head is None:
            raise ValueError("kv_head is required for native-GQA k/v slot export")
        return f"a{layer}.kv{kv_head}<{qkv}>"

    if head is None:
        raise ValueError("head is required for q slot export")
    return f"a{layer}.h{head}<{qkv}>"


def get_dst_nodes_metadata(graph: "Graph") -> List[Dict[str, Any]]:
    info = get_gqa_info(graph)
    n_heads = info["n_heads"]
    n_kv_heads = info["n_kv_heads"]
    uses_native_gqa = info["uses_native_gqa"]

    out: List[Dict[str, Any]] = []
    for layer in range(graph.cfg["n_layers"]):
        for head in range(n_heads):
            out.append(
                {
                    "name": format_dst_slot_name(layer, "q", head=head, uses_native_gqa=uses_native_gqa),
                    "layer": layer,
                    "kind": "attention_input",
                    "qkv": "q",
                    "head": head,
                    "kv_head": None,
                }
            )
        if uses_native_gqa:
            for kv_head in range(n_kv_heads):
                out.append(
                    {
                        "name": format_dst_slot_name(layer, "k", kv_head=kv_head, uses_native_gqa=True),
                        "layer": layer,
                        "kind": "attention_input",
                        "qkv": "k",
                        "head": None,
                        "kv_head": kv_head,
                    }
                )
            for kv_head in range(n_kv_heads):
                out.append(
                    {
                        "name": format_dst_slot_name(layer, "v", kv_head=kv_head, uses_native_gqa=True),
                        "layer": layer,
                        "kind": "attention_input",
                        "qkv": "v",
                        "head": None,
                        "kv_head": kv_head,
                    }
                )
        else:
            for letter in ("k", "v"):
                for head in range(n_heads):
                    out.append(
                        {
                            "name": format_dst_slot_name(layer, letter, head=head, uses_native_gqa=False),
                            "layer": layer,
                            "kind": "attention_input",
                            "qkv": letter,
                            "head": head,
                            "kv_head": head,
                        }
                    )
        out.append(
            {
                "name": f"m{layer}",
                "layer": layer,
                "kind": "mlp_input",
                "qkv": None,
                "head": None,
                "kv_head": None,
            }
        )
    out.append(
        {
            "name": "logits",
            "layer": None,
            "kind": "logits_input",
            "qkv": None,
            "head": None,
            "kv_head": None,
        }
    )
    return out


def edge_export_info(edge: "Edge") -> Dict[str, Any]:
    info = get_gqa_info(edge.graph)
    uses_native_gqa = info["uses_native_gqa"]
    query_head = getattr(edge.child, "head", None)
    kv_head = getattr(edge.child, "kv_head", None) if edge.qkv in {"k", "v"} else None
    dst_slot_name = None
    if edge.qkv is not None:
        dst_slot_name = format_dst_slot_name(
            edge.child.layer,
            edge.qkv,
            head=query_head,
            kv_head=kv_head,
            uses_native_gqa=uses_native_gqa,
        )

    if edge.qkv is None:
        export_name = edge.name
    elif uses_native_gqa and edge.qkv in {"k", "v"}:
        export_name = f"{edge.parent.name}->{edge.child.name}<{edge.qkv}:kv{kv_head}>"
    else:
        export_name = edge.name

    return {
        "name": edge.name,
        "export_name": export_name,
        "parent": edge.parent.name,
        "child": edge.child.name,
        "qkv": edge.qkv,
        "query_head": query_head,
        "kv_head": kv_head,
        "dst_slot_name": dst_slot_name,
        "matrix_index": [int(edge.matrix_index[0]), int(edge.matrix_index[1])],
        "shares_backward_slot": bool(uses_native_gqa and edge.qkv in {"k", "v"}),
    }


def edge_graphviz_label(edge: "Edge") -> str:
    info = edge_export_info(edge)
    if not get_gqa_info(edge.graph)["uses_native_gqa"]:
        return ""
    if info["qkv"] is None:
        return ""
    if info["qkv"] == "q":
        return "q"
    if info["kv_head"] is None:
        return str(info["qkv"])
    return f"{info['qkv']}@kv{info['kv_head']}"
