#!/usr/bin/env python3
"""
DeimosStream Universal 2026 Sparse-MoE Architecture Adapter Layer (`src/deimos_stream/model_adapters.py`)
=========================================================================================================
Trait-based adapter primitives for converting and executing 2026 HuggingFace Sparse & Hybrid MoE models:
  1. Mixed Dense + MoE layer discovery (`first_k_dense_replace`, `num_dense_layers`, `moe_layers`,
     `mlp_only_layers`) across `mlp`, `feed_forward`, `block_sparse_moe`, and `ffn` containers.
  2. Block-scaled `FP8 E4M3` & `UE8M0` / `MXFP8` (`weight_scale_inv`, `.scale`) dequantization
     (`DeepSeek-V4`, `GLM-5.3-Flash`, `Qwen3.6/3.8-FP8`, `DeepSeek-V3/V3.2`).
  3. Clamped & alpha-scaled SwiGLU extraction (`swiglu_limit=10.0` on `DeepSeek-V4` / `GLM-5.3`,
     `SwiGLU-OAI` `alpha=1.702, limit=7.0, up+1` on `MiniMax-M3.1-Flash`).
  4. Runtime MoE block binding & generic CUDA Group-64 Q4 backbone quantization (`DeimosQ4Linear`).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Set, Tuple

from deimos_stream.model_catalog import (
    ROADMAP_MOE_FAMILIES,
    SUPPORTED_MOE_FAMILIES,
    MoEFamilySpec,
    list_supported_moe_families,
)
from deimos_stream.triton_q4 import quantize_2d_q4_torch, triton_q4_dense_linear

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _NNModule = nn.Module
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _NNModule = object  # type: ignore[assignment,misc]

_IGNORE_PREFIX_RE = re.compile(
    r"^(model\.visual\.|visual\.|vision_model\.|vision_tower\.|model\.vision_tower\.|"
    r"multi_modal_projector\.|model\.multi_modal_projector\.|patch_merge_mlp\.|aligner\.|mtp\.|model\.mtp\.|image_)"
)
_LAYER_KEY_RE = re.compile(r"^(?:model\.language_model\.|language_model\.model\.|model\.)?layers\.(\d+)\.")


def extract_swiglu_params(raw_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extracts `(swiglu_limit, swiglu_alpha, swiglu_up_offset)` for 2026 clamped/OAI SwiGLU families."""
    t_cfg = raw_cfg.get("text_config", raw_cfg)
    assert isinstance(t_cfg, dict)
    mtype = str(t_cfg.get("model_type") or raw_cfg.get("model_type") or "").lower()
    raw_lim = t_cfg.get("swiglu_limit", raw_cfg.get("swiglu_limit"))
    limit = float(raw_lim) if (raw_lim is not None and 0.0 < float(raw_lim) < 1e6) else 0.0
    alpha = float(t_cfg.get("swiglu_alpha", raw_cfg.get("swiglu_alpha", 1.0)) or 1.0)
    up_offset = 1.0 if mtype in {"minimax_m3_vl", "minimax_m3_vl_text"} else 0.0
    return limit, alpha, up_offset


def should_use_hf_hybrid_converter(local_dir: Path) -> bool:
    """Returns `True` when a local checkpoint requires the universal `hf_hybrid_q4` converter."""
    import json

    cfg_p = Path(local_dir) / "config.json"
    if not cfg_p.exists():
        return False
    raw = json.loads(cfg_p.read_text())
    t_cfg = raw.get("text_config", raw)
    assert isinstance(t_cfg, dict)
    mtype = str(t_cfg.get("model_type") or raw.get("model_type") or "").lower()
    if mtype not in {"olmoe", "mixtral", "qwen2_moe", "qwen3_moe", "phimoe"}:
        return True
    if raw.get("quantization_config") or t_cfg.get("quantization_config"):
        return True
    if int(t_cfg.get("first_k_dense_replace", 0) or 0) > 0:
        return True
    if t_cfg.get("mlp_only_layers"):
        return True
    return False


def validate_convertible_architecture(raw_cfg: Dict[str, object]) -> None:
    """Rejects checkpoints requiring non-standard biased/Engram/looped leaf formats with a clear error."""
    t_cfg = raw_cfg.get("text_config", raw_cfg)
    assert isinstance(t_cfg, dict)
    mtype = str(t_cfg.get("model_type") or raw_cfg.get("model_type") or "").lower()
    if mtype == "deepseek_v41" or t_cfg.get("engram_layer_ids"):
        raise ValueError(
            "DeepSeek-V4.1-Flash (`deepseek_v41` / `engram_layer_ids`) requires 20+20 CED + 196B mmap Engram "
            "n-gram tables; standard `deepseek_v4` (`DeepSeek-V4-Flash`) without Engram layers is supported."
        )
    if mtype == "gpt_oss":
        raise ValueError(
            "OpenAI `gpt_oss` uses MXFP4 experts with per-expert linear biases (`gate_up_proj_bias`, "
            "`down_proj_bias`), which bias-free `DEIMIDX2` Q4 leaves do not store."
        )
    if mtype == "inkling2_loop":
        raise ValueError(
            "Thinking Machines `inkling2_loop` uses multi-turn looped execution with dual-tanh softcapped SwiGLU."
        )
    sw_list = t_cfg.get("swiglu_limits")
    if isinstance(sw_list, list) and any(bool(x) for x in sw_list):
        raise ValueError("Per-layer post-SiLU `swiglu_limits` lists (`step3p7`) are not yet supported.")


def is_ignored_checkpoint_key(key: str, num_layers: int) -> bool:
    """Filters out vision encoders, projectors, MTP heads, and extra MTP draft layers (`l >= num_layers`)."""
    if _IGNORE_PREFIX_RE.match(key) or key.endswith(".gate.bias_vl") or ".engram." in key:
        return True
    m = _LAYER_KEY_RE.match(key)
    if m and int(m.group(1)) >= num_layers:
        return True
    return False


def is_companion_scale_key(key: str, active_keys: Set[str]) -> bool:
    """Returns `True` when `key` is an FP8 / UE8M0 companion scale tensor consumed by `pop_tensor_dequant`."""
    if key.endswith(("_scale_inv", ".scale_inv", ".weight_scale_inv")):
        return True
    if key.endswith(".scale") and f"{key[:-6]}.weight" in active_keys:
        return True
    if key.endswith("_scale") and f"{key[:-6]}_weight" in active_keys:
        return True
    return False


def detect_lm_prefix(active_keys: Set[str]) -> str:
    """Detects top-level language-model key prefix across HF single-modal & multimodal checkpoints."""
    for cand in ("model.language_model", "language_model.model", "model"):
        if any(k.startswith(f"{cand}.layers.") for k in active_keys):
            return cand
    return ""


def canonicalize_hf_key(key: str, lp_root: str) -> str:
    """Normalizes multimodal language-model tensor names while preserving raw HF conversion keys."""
    if lp_root in ("model.language_model", "language_model.model") and key.startswith(f"{lp_root}."):
        return "model." + key[len(lp_root) + 1 :]
    if key.startswith("language_model.lm_head."):
        return key[len("language_model.") :]
    return key


def dequantize_fp8_block_tensor(w_fp8: "torch.Tensor", scale_inv: "torch.Tensor") -> "torch.Tensor":
    """Dequantizes block-scaled `FP8 E4M3` (`float32`, `UE8M0`, or `uint8` `E8M0`) weights to `float16`."""
    w_f32 = w_fp8.to(torch.float32)
    if scale_inv.dtype == torch.uint8:
        s_f32 = (scale_inv.to(device=w_f32.device, dtype=torch.float32) - 127.0).exp2()
    else:
        s_f32 = scale_inv.to(device=w_f32.device, dtype=torch.float32)
    if w_f32.ndim == 2 and s_f32.ndim == 2:
        m, n = w_f32.shape
        sm, sn = s_f32.shape
        if sm == m and (sn == 1 or sn == n):
            return (w_f32 * s_f32).to(torch.float16)
        bm = (m + sm - 1) // sm
        bn = (n + sn - 1) // sn
        s_exp = s_f32.repeat_interleave(bm, dim=0)[:m, :].repeat_interleave(bn, dim=1)[:, :n]
        return (w_f32 * s_exp).to(torch.float16)
    if w_f32.ndim == 3 and s_f32.ndim == 3:
        _, m, n = w_f32.shape
        _, sm, sn = s_f32.shape
        bm = (m + sm - 1) // sm
        bn = (n + sn - 1) // sn
        s_exp = s_f32.repeat_interleave(bm, dim=1)[:, :m, :].repeat_interleave(bn, dim=2)[:, :, :n]
        return (w_f32 * s_exp).to(torch.float16)
    return (w_f32 * s_f32).to(torch.float16)


def pop_tensor_dequant(
    fetcher: Any,
    key: str,
    active_keys: Set[str],
    used_special: Set[str],
    device: str = "cpu",
) -> "torch.Tensor":
    """Loads `key` from `fetcher` and automatically applies companion FP8 / UE8M0 block scales."""
    used_special.add(key)
    t = fetcher.pop_tensor(key, device=device)
    cands = [f"{key}_scale_inv", f"{key}.scale_inv"]
    if key.endswith(".weight"):
        stem = key[: -len(".weight")]
        cands.extend([f"{stem}.weight_scale_inv", f"{stem}.scale"])
    for sk in cands:
        if sk in active_keys and sk not in used_special:
            used_special.add(sk)
            s_inv = fetcher.pop_tensor(sk, device=device)
            return dequantize_fp8_block_tensor(t, s_inv)
    if hasattr(torch, "float8_e4m3fn") and t.dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    }:
        return t.to(torch.float16)
    return t


def discover_hf_moe_layers(
    num_layers: int, lp_root: str, active_keys: Set[str]
) -> List[Tuple[int, str]]:
    """Discovers `(layer_idx, container_attr)` for every layer containing routed MoE expert weights."""
    pfx_base = f"{lp_root}.layers." if lp_root else "layers."
    found: List[Tuple[int, str]] = []
    for l_idx in range(num_layers):
        lp = f"{pfx_base}{l_idx}"
        for attr in ("mlp", "feed_forward", "block_sparse_moe", "ffn"):
            ep = f"{lp}.{attr}.experts."
            if (
                f"{lp}.{attr}.experts.gate_up_proj" in active_keys
                or f"{lp}.{attr}.experts.0.gate_proj.weight" in active_keys
                or f"{lp}.{attr}.experts.0.w1.weight" in active_keys
                or any(k.startswith(ep) for k in active_keys)
            ):
                found.append((l_idx, attr))
                break
    return found


def extract_layer_experts_gu_dn(
    fetcher: Any,
    lp: str,
    container_attr: str,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    active_keys: Set[str],
    used_special: Set[str],
    q_dev: str,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Extracts `(gu_3d [E, 2*I, D], dn_3d [E, D, I])` across 3D, Llama-4 transposed 3D, and 2D shards."""
    ep = f"{lp}.{container_attr}.experts"
    gu_3d_k = f"{ep}.gate_up_proj"
    dn_3d_k = f"{ep}.down_proj"
    if gu_3d_k not in active_keys and f"{gu_3d_k}.weight" in active_keys:
        gu_3d_k = f"{gu_3d_k}.weight"
        dn_3d_k = f"{dn_3d_k}.weight"

    if gu_3d_k in active_keys:
        gu_t = pop_tensor_dequant(fetcher, gu_3d_k, active_keys, used_special, device=q_dev)
        dn_t = pop_tensor_dequant(fetcher, dn_3d_k, active_keys, used_special, device=q_dev)
        if dn_t.shape[1] == inter_size and dn_t.shape[2] == hidden_size and inter_size != hidden_size:
            gu_t = gu_t.transpose(1, 2).contiguous()
            dn_t = dn_t.transpose(1, 2).contiguous()
        return gu_t, dn_t

    gu_list, dn_list = [], []
    for e_idx in range(num_experts):
        if f"{ep}.{e_idx}.gate_proj.weight" in active_keys:
            gk, uk, dk = (
                f"{ep}.{e_idx}.gate_proj.weight",
                f"{ep}.{e_idx}.up_proj.weight",
                f"{ep}.{e_idx}.down_proj.weight",
            )
        else:
            gk, uk, dk = (
                f"{ep}.{e_idx}.w1.weight",
                f"{ep}.{e_idx}.w3.weight",
                f"{ep}.{e_idx}.w2.weight",
            )
        g_w = pop_tensor_dequant(fetcher, gk, active_keys, used_special, device="cpu")
        u_w = pop_tensor_dequant(fetcher, uk, active_keys, used_special, device="cpu")
        d_w = pop_tensor_dequant(fetcher, dk, active_keys, used_special, device="cpu")
        gu_list.append(torch.cat([g_w, u_w], dim=0))
        dn_list.append(d_w)
    del q_dev
    return torch.stack(gu_list, dim=0), torch.stack(dn_list, dim=0)


class DeimosQ4Linear(_NNModule):  # type: ignore[misc,valid-type]
    """Group-64 4-bit `nn.Linear` replacement backed by `__dp4a` CUDA / Triton Q4 GEMV."""

    def __init__(self, lin: "nn.Linear", device: "torch.device", group_size: int = 64):
        super().__init__()
        self.in_features = int(lin.in_features)
        self.out_features = int(lin.out_features)
        self.group_size = int(group_size)
        w_f16 = lin.weight.data.to(device=device, dtype=torch.float16).contiguous()
        self.qw, self.qs, self.qb = quantize_2d_q4_torch(w_f16, group_size=self.group_size)
        del w_f16
        self.bias = (
            nn.Parameter(lin.bias.data.to(device=device, dtype=torch.float16), requires_grad=False)
            if lin.bias is not None
            else None
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        y = triton_q4_dense_linear(
            x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16),
            self.qw, self.qs, self.qb, group_size=self.group_size,
        )
        if self.bias is not None:
            y = y + self.bias
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y.reshape(*orig_shape, self.out_features)


def quantize_generic_hf_backbone_q4(model: "nn.Module", device: "torch.device") -> int:
    """Quantizes qualifying dense `nn.Linear` projections (`in_features % 64 == 0`) to Group-64 Q4."""
    skip_tokens = ("gate", "router", "lm_head", "embed", "norm", "a_proj", "b_proj", "indexer", "hc_")
    replaced = 0

    def _walk(parent: "nn.Module", prefix: str = "") -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            low = full.lower()
            if any(tok in low.split(".")[-1] for tok in skip_tokens) and not low.endswith(
                ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
            ):
                continue
            if isinstance(child, nn.Linear) and child.in_features >= 128 and child.in_features % 64 == 0:
                setattr(parent, name, DeimosQ4Linear(child, device=device, group_size=64))
                replaced += 1
            else:
                _walk(child, full)

    inner = getattr(model, "model", model)
    _walk(inner)
    return replaced


def bind_hf_moe_layer(layer_mod: "nn.Module", container_attr: str, paged_exp: Any) -> Any:
    """Attaches `paged_exp` (`DeimosPagedHFExperts`) to `layer_mod` across 2025–2026 HF MoE families."""
    eff_attr = container_attr if hasattr(layer_mod, container_attr) else next(
        a for a in ("mlp", "feed_forward", "block_sparse_moe") if hasattr(layer_mod, a)
    )
    moe_mod = getattr(layer_mod, eff_attr)

    # Case 1: Llama-4 `Llama4TextMoe` (`router` + `shared_expert`)
    if moe_mod.__class__.__name__ == "Llama4TextMoe" or (
        hasattr(moe_mod, "router") and hasattr(moe_mod, "shared_expert") and not hasattr(moe_mod, "gate")
    ):
        moe_mod.experts = paged_exp
        router_mod = moe_mod.router
        top_k = int(getattr(router_mod, "top_k", 1))

        def _llama4_sparse_forward(hidden_states: "torch.Tensor"):
            x_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            router_logits = F.linear(x_flat, router_mod.weight)
            top_val, top_idx = torch.topk(router_logits, top_k, dim=-1)
            scores = torch.sigmoid(top_val.float()).to(x_flat.dtype)
            y = paged_exp(x_flat, top_idx, scores, shared_fn=lambda: moe_mod.shared_expert(x_flat))
            return y, router_logits

        moe_mod.forward = _llama4_sparse_forward  # type: ignore[method-assign]
        return router_mod

    moe_mod.experts = paged_exp

    # Case 2: Qwen3.8 / Qwen3.6 / Qwen3.5 / Qwen3-Next (`gate` + sigmoid-gated `shared_expert`)
    if hasattr(moe_mod, "gate") and hasattr(moe_mod, "shared_expert") and hasattr(moe_mod, "shared_expert_gate"):
        def _gated_shared_forward(hidden_states: "torch.Tensor", *args: Any, **kwargs: Any) -> "torch.Tensor":
            del args, kwargs
            B, L, D = hidden_states.shape
            x_flat = hidden_states.reshape(-1, D)
            g_out = moe_mod.gate(x_flat)
            rw, se = (g_out[1], g_out[2]) if isinstance(g_out, tuple) and len(g_out) >= 3 else (g_out[0], g_out[1])

            def _sh() -> "torch.Tensor":
                return torch.sigmoid(moe_mod.shared_expert_gate(x_flat)) * moe_mod.shared_expert(x_flat)

            return paged_exp(x_flat, se, rw, shared_fn=_sh).reshape(B, L, D)

        moe_mod.forward = _gated_shared_forward  # type: ignore[method-assign]
        return moe_mod.gate

    # Case 3: DeepSeek-V4 / GLM-5.3 / Kimi-K3 / MiniMax-M3.1 / DeepSeek-V3 (`gate` + additive `shared_experts`)
    if (
        hasattr(moe_mod, "gate")
        and getattr(moe_mod, "shared_experts", None) is not None
        and not getattr(moe_mod, "is_hash", False)
    ):
        flat_gate = moe_mod.__class__.__name__.startswith("MiniMax")
        r_scale = float(moe_mod.__dict__.get("routed_scaling_factor", 1.0))

        def _additive_shared_forward(hidden_states: "torch.Tensor", *args: Any, **kwargs: Any) -> "torch.Tensor":
            del args, kwargs
            orig_shape = hidden_states.shape
            x_flat = hidden_states.reshape(-1, orig_shape[-1])
            g_in = x_flat if flat_gate else hidden_states
            g_out = moe_mod.gate(g_in)
            rw, se = (g_out[1], g_out[2]) if isinstance(g_out, tuple) and len(g_out) >= 3 else (g_out[0], g_out[1])
            if r_scale != 1.0:
                rw = rw * r_scale
            sh_in = x_flat if flat_gate else hidden_states
            y = paged_exp(
                x_flat, se, rw, shared_fn=lambda: moe_mod.shared_experts(sh_in).reshape(-1, orig_shape[-1])
            )
            return y.reshape(*orig_shape)

        moe_mod.forward = _additive_shared_forward  # type: ignore[method-assign]
        return moe_mod.gate

    return getattr(moe_mod, "gate", getattr(moe_mod, "router", None))


def load_hf_meta_skeleton(
    store_dir: Path,
    device: "torch.device",
    compute_dtype: "torch.dtype",
    backbone: str,
) -> Tuple["nn.Module", Any, Path]:
    """Instantiates `AutoModel` on `meta`, applies HF 5.x `conversion_mapping` to `hf_backbone.safetensors`, and materializes rotary buffers."""
    from safetensors.torch import load_file as st_load_torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.modeling_utils import LoadStateDictConfig

    hf_cfg = AutoConfig.from_pretrained(str(store_dir), trust_remote_code=True)
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    m_cls = AutoModelForCausalLM._model_mapping.get(type(text_cfg), None)
    text_cfg._attn_implementation = "sdpa" if getattr(m_cls, "_supports_sdpa", False) else "eager"
    text_cfg.dtype = compute_dtype

    with torch.device("meta"):
        if m_cls is not None:
            model = AutoModelForCausalLM.from_config(text_cfg, dtype=compute_dtype, trust_remote_code=True)
        else:
            src_cfg_p = store_dir / "hf_source_config.json"
            full_cfg = AutoConfig.from_pretrained(str(src_cfg_p if src_cfg_p.exists() else store_dir))
            if hasattr(full_cfg, "quantization_config"):
                delattr(full_cfg, "quantization_config")
            v_cls = AutoModelForImageTextToText._model_mapping.get(type(full_cfg), None)
            full_cfg.text_config._attn_implementation = "sdpa" if getattr(v_cls, "_supports_sdpa", False) else "eager"
            full_cfg.text_config.dtype = compute_dtype
            full_cfg.dtype = compute_dtype
            model = AutoModelForImageTextToText.from_config(full_cfg, dtype=compute_dtype, trust_remote_code=True)
            if hasattr(model.model, "visual"):
                model.model.visual = None
            if hasattr(model.model, "vision_tower"):
                model.model.vision_tower = None
            text_cfg = full_cfg.text_config

    bb_path = store_dir / "hf_backbone.safetensors"
    bb_tensors = st_load_torch(str(bb_path), device="cpu")
    load_dev = torch.device("cpu") if (backbone == "q4" and device.type == "cuda") else device
    is_vlm = hasattr(model.model, "language_model")
    casted_bb = {}
    for k in list(bb_tensors.keys()):
        v = bb_tensors.pop(k)
        if is_vlm and k.startswith(("model.layers.", "model.norm.")):
            k = "model.language_model." + k[len("model.") :]
        casted_bb[k] = (
            v.to(device=load_dev, dtype=compute_dtype)
            if v.is_floating_point() and v.dtype != torch.float32
            else v.to(device=load_dev)
        )
    convert_and_load_state_dict_in_model(
        model,
        casted_bb,
        LoadStateDictConfig(
            device_map={"": load_dev},
            dtype=compute_dtype,
            weight_mapping=get_model_conversion_mapping(model),
        ),
    )
    casted_bb.clear()
    for _, parent_mod in list(model.named_modules()):
        for child_name, child_mod in list(parent_mod.named_children()):
            if any(b.device.type == "meta" for b in child_mod.buffers(recurse=False)):
                setattr(parent_mod, child_name, type(child_mod)(config=text_cfg).to(device))
    return model, text_cfg, bb_path


__all__ = [
    "DeimosQ4Linear",
    "MoEFamilySpec",
    "ROADMAP_MOE_FAMILIES",
    "SUPPORTED_MOE_FAMILIES",
    "bind_hf_moe_layer",
    "canonicalize_hf_key",
    "dequantize_fp8_block_tensor",
    "detect_lm_prefix",
    "discover_hf_moe_layers",
    "extract_layer_experts_gu_dn",
    "extract_swiglu_params",
    "is_companion_scale_key",
    "is_ignored_checkpoint_key",
    "list_supported_moe_families",
    "load_hf_meta_skeleton",
    "pop_tensor_dequant",
    "quantize_generic_hf_backbone_q4",
    "should_use_hf_hybrid_converter",
    "validate_convertible_architecture",
]
