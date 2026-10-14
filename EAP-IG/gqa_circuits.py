import argparse
import json
import os
import sys
from typing import Dict, Literal, Optional
import torch
from transformer_lens import HookedTransformer

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_EAP_SRC = os.path.join(_CURRENT_DIR, "src")
if _EAP_SRC not in sys.path:
    sys.path.insert(0, _EAP_SRC)

from eap.graph import Graph

GQAMode = Literal["auto", "native", "ungrouped"]

def _resolve_requested_gqa_mode(model: HookedTransformer, gqa_mode: GQAMode) -> str:
    raw_n_kv_heads = getattr(model.cfg, "n_key_value_heads", None)
    has_grouped_query_attention = raw_n_kv_heads is not None and raw_n_kv_heads != model.cfg.n_heads
    if not has_grouped_query_attention:
        return "mha"
    if gqa_mode == "auto":
        return "native"
    return gqa_mode

def configure_model_for_eap(model: HookedTransformer, *, gqa_mode: GQAMode = "auto",) -> str:
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    actual_mode = _resolve_requested_gqa_mode(model, gqa_mode)
    if getattr(model.cfg, "n_key_value_heads", None) is not None:
        model.cfg.ungroup_grouped_query_attention = actual_mode == "ungrouped"
    return actual_mode

def load_model_for_circuit_discovery(model_name: str, device: str, dtype: torch.dtype, *, gqa_mode: GQAMode = "auto", center_writing_weights: bool = False, center_unembed: bool = False, fold_ln: bool = False,) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_ln=fold_ln,
        device=device,
        dtype=dtype,
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    actual_mode = configure_model_for_eap(model, gqa_mode=gqa_mode)
    setattr(model.cfg, "eap_requested_gqa_mode", gqa_mode)
    setattr(model.cfg, "eap_actual_gqa_mode", actual_mode)
    return model

def summarize_model_for_circuit_discovery(model: HookedTransformer) -> Dict[str, Optional[object]]:
    raw_n_kv_heads = getattr(model.cfg, "n_key_value_heads", None)
    effective_n_kv_heads = model.cfg.n_heads
    if raw_n_kv_heads is not None and not bool(getattr(model.cfg, "ungroup_grouped_query_attention", False)):
        effective_n_kv_heads = raw_n_kv_heads
    return {
        "model_name": getattr(model.cfg, "model_name", None),
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "n_key_value_heads": raw_n_kv_heads,
        "effective_n_kv_heads": effective_n_kv_heads,
        "ungroup_grouped_query_attention": bool(getattr(model.cfg, "ungroup_grouped_query_attention", False)),
        "eap_requested_gqa_mode": getattr(model.cfg, "eap_requested_gqa_mode", None),
        "eap_actual_gqa_mode": getattr(model.cfg, "eap_actual_gqa_mode", None),
        "use_attn_result": bool(getattr(model.cfg, "use_attn_result", False)),
        "use_split_qkv_input": bool(getattr(model.cfg, "use_split_qkv_input", False)),
        "use_hook_mlp_in": bool(getattr(model.cfg, "use_hook_mlp_in", False)),
    }

def summarize_graph_for_circuit_discovery(model: HookedTransformer) -> Dict[str, object]:
    graph = Graph.from_model(model)
    return {
        "graph_cfg": dict(graph.cfg),
        "gqa_info": graph.get_gqa_info(),
        "n_forward": int(graph.n_forward),
        "n_backward": int(graph.n_backward),
        "dst_nodes_head": graph.get_dst_nodes()[: min(16, graph.n_backward)],
    }

def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--gqa_mode", type=str, default="native", choices=["auto", "native", "ungrouped"])
    args = ap.parse_args()

    model = load_model_for_circuit_discovery(
        args.model_name,
        args.device,
        _dtype_from_name(args.dtype),
        gqa_mode=args.gqa_mode,  # type: ignore[arg-type]
    )
    print(
        json.dumps(
            {
                "model": summarize_model_for_circuit_discovery(model),
                "graph": summarize_graph_for_circuit_discovery(model),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
