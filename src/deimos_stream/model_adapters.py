#!/usr/bin/env python3
"""
DeimosStream Universal Sparse-MoE Architecture Adapter Registry (`src/deimos_stream/model_adapters.py`)
=======================================================================================================
Provides a single trait-based adapter layer for converting and executing 2025–2026 HuggingFace
Sparse & Hybrid MoE checkpoints without per-model code duplication:
  1. Mixed Dense + MoE layer discovery (`first_k_dense_replace`, `moe_layers`, `mlp_only_layers`,
     `moe_layer_interval`) directly from checkpoint `weight_map`, stripping MTP/vision weights.
  2. Block-scaled `FP8 E4M3` (`weight_scale_inv` `[128,128]` / `[32,32]`) dequantization during
     `deimos-stream convert` (`DeepSeek-V3/R1/V3.2`, `Kimi-K2`, `Qwen3-FP8`, `GLM-4.5-FP8`).
  3. Canonical `[E, 2I, D]` / `[E, D, I]` expert extraction across 3D `gate_up_proj`, transposed
     3D `Llama-4` (`[E, D, 2I]`), and 2D split `gate_proj/up_proj/down_proj` (`w1/w3/w2`) shards.
  4. Runtime MoE block binding (`mlp`, `feed_forward`, `block_sparse_moe`) and generic CUDA Q4
     backbone quantization (`DeimosQ4Linear`) across `transformers` 5.x models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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


@dataclass(frozen=True)
class MoEFamilySpec:
    family: str
    model_types: Tuple[str, ...]
    attention_kind: str
    routing_kind: str
    example_models: Tuple[str, ...]
    backends: Tuple[str, ...] = ("mlx", "cuda", "cpu")
    status: str = "verified"
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


SUPPORTED_MOE_FAMILIES: Tuple[MoEFamilySpec, ...] = (
    MoEFamilySpec(
        family="Qwen3.6 / Qwen3.5 / Qwen3-Next Hybrid",
        model_types=("qwen3_5_moe", "qwen3_next"),
        attention_kind="Hybrid GatedDeltaNet + QK-Norm GQA",
        routing_kind="Softmax Top-K + Gated Shared Expert",
        example_models=("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        notes="81-segment static CUDA Graphs + native Metal gated_delta_update.",
    ),
    MoEFamilySpec(
        family="Qwen2-MoE / Qwen3-MoE",
        model_types=("qwen2_moe", "qwen3_moe", "qwen3_vl_moe"),
        attention_kind="GQA + Per-Head QK-Norm",
        routing_kind="Softmax Top-K (+ optional Shared Expert / mlp_only_layers)",
        example_models=("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B", "Qwen/Qwen2.5-MoE"),
        notes="Supports both BF16/FP16 and FP8 block-scaled checkpoints.",
    ),
    MoEFamilySpec(
        family="DeepSeek-V3 / R1 / V3.2 / Kimi-K2 / Moonlight",
        model_types=("deepseek_v3", "deepseek_v32"),
        attention_kind="MLA (Multi-head Latent Attention + YaRN)",
        routing_kind="NoAux Sigmoid Top-K (e_score_correction_bias) + Shared Experts",
        example_models=("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1", "moonshotai/Moonlight-16B-A3B-Instruct", "moonshotai/Kimi-K2-Instruct"),
        notes="Auto-dequantizes [128,128] FP8 weight_scale_inv shards; preserves first_k_dense_replace layers.",
    ),
    MoEFamilySpec(
        family="DeepSeek-V2 / DeepSeek-V2-Lite",
        model_types=("deepseek_v2",),
        attention_kind="MLA (Multi-head Latent Attention)",
        routing_kind="Softmax / Group-Limited Top-K + Shared Experts",
        example_models=("deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek-ai/DeepSeek-V2.5"),
        notes="Preserves dense layer 0 in backbone and streams MoE layers 1..L-1.",
    ),
    MoEFamilySpec(
        family="GLM-4.5 / GLM-4.7 MoE",
        model_types=("glm4_moe", "glm4_moe_lite"),
        attention_kind="GQA / MLA + Partial RoPE",
        routing_kind="NoAux Sigmoid Top-K + Shared Experts (first_k_dense_replace)",
        example_models=("zai-org/GLM-4.5-Air", "zai-org/GLM-4.5"),
        notes="Strips MTP draft layer automatically during conversion.",
    ),
    MoEFamilySpec(
        family="Llama 4 (Scout & Maverick)",
        model_types=("llama4", "llama4_text"),
        attention_kind="Interleaved Chunked GQA + iRoPE Temperature Tuning",
        routing_kind="Sigmoid Top-K + Shared Expert (interleaved moe_layers)",
        example_models=("meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"),
        notes="Auto-transposes [E, D, 2I] 3D experts and replaces HF O(E) dense BMM with O(top_k) paged execution.",
    ),
    MoEFamilySpec(
        family="OLMoE / Mixtral / Phi-MoE",
        model_types=("olmoe", "mixtral", "phimoe"),
        attention_kind="GQA / SWA (+ Full QK-Norm on OLMoE)",
        routing_kind="Softmax Top-K",
        example_models=("allenai/OLMoE-1B-7B-0125-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", "microsoft/Phi-3.5-MoE-instruct"),
        notes="Supports both native shared_core engine and HF/MLX universal engine.",
    ),
    MoEFamilySpec(
        family="MiniMax-M2 / ERNIE-4.5-MoE / Jamba / Granite-MoE",
        model_types=("minimax", "minimax_m2", "ernie4_5_moe", "jamba", "granitemoe", "granitemoehybrid"),
        attention_kind="GQA or Hybrid Mamba2-SSM + GQA",
        routing_kind="Softmax / Bias-Corrected Top-K (interleaved MoE layers)",
        example_models=("MiniMaxAI/MiniMax-M2", "baidu/ERNIE-4.5-21B-A3B-PT", "ai21labs/AI21-Jamba-1.5-Mini"),
        notes="Automatic interleaved MoE layer discovery from checkpoint weight_map.",
    ),
)

ROADMAP_MOE_FAMILIES: Tuple[Dict[str, str], ...] = (
    {
        "family": "DeepSeek-V4.1-Flash (CED + 196B Engram + NVFP4)",
        "model_type": "deepseek_v41",
        "status": "roadmap",
        "reason": "Requires custom 20+20 Causal Encoder-Decoder (CED), CSA2 sparse indexer, clamped SwiGLU (limit=10.0), and 185 GiB mmap Engram n-gram lookup tables.",
    },
    {
        "family": "OpenAI gpt-oss-20b / gpt-oss-120b (MXFP4 + Biased Clamped SwiGLU)",
        "model_type": "gpt_oss",
        "status": "roadmap",
        "reason": "Uses per-expert gate_up_proj_bias / down_proj_bias and alpha=1.702 clamped SwiGLU (limit=7.0) not yet stored in bias-free DEIMIDX2 Q4 leaves.",
    },
)

_IGNORE_PREFIX_RE = re.compile(
    r"^(model\.visual\.|visual\.|vision_model\.|model\.vision_tower\.|multi_modal_projector\.|aligner\.|mtp\.|image_)"
)
_LAYER_KEY_RE = re.compile(r"^(?:model\.language_model\.|language_model\.model\.|model\.)?layers\.(\d+)\.")


def list_supported_moe_families() -> Dict[str, object]:
    return {
        "supported": [s.as_dict() for s in SUPPORTED_MOE_FAMILIES],
        "roadmap": list(ROADMAP_MOE_FAMILIES),
    }


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
    """Rejects checkpoints requiring non-standard biased/Engram leaf formats with a clear error."""
    t_cfg = raw_cfg.get("text_config", raw_cfg)
    assert isinstance(t_cfg, dict)
    mtype = str(t_cfg.get("model_type") or raw_cfg.get("model_type") or "").lower()
    if mtype == "deepseek_v41" or t_cfg.get("engram_layer_ids"):
        raise ValueError(
            "DeepSeek-V4.1-Flash (`deepseek_v41` / `engram_layer_ids`) requires CED + 196B mmap Engram "
            "tables + clamped SwiGLU (`limit=10.0`) and is not yet supported by `DEIMIDX2`."
        )
    if mtype == "gpt_oss":
        raise ValueError(
            "OpenAI `gpt_oss` uses MXFP4 experts with per-expert linear biases (`gate_up_proj_bias`, "
            "`down_proj_bias`) and `alpha=1.702` clamped SwiGLU (`limit=7.0`), which `DEIMIDX2` leaves do not yet store."
        )
    sw_lim = t_cfg.get("swiglu_limit")
    if sw_lim is not None and float(sw_lim) < 1e6:
        raise ValueError(
            f"Model specifies finite `swiglu_limit={sw_lim}`, which requires clamped SwiGLU expert kernels."
        )


def is_ignored_checkpoint_key(key: str, num_layers: int) -> bool:
    """Filters out vision encoders, projectors, MTP heads, and extra MTP draft layers (`l >= num_layers`)."""
    if _IGNORE_PREFIX_RE.match(key):
        return True
    m = _LAYER_KEY_RE.match(key)
    if m and int(m.group(1)) >= num_layers:
        return True
    return False


def detect_lm_prefix(active_keys: Set[str]) -> str:
    """Detects top-level language-model key prefix across HF single-modal & multimodal checkpoints."""
    for cand in ("model.language_model", "language_model.model", "model"):
        if any(k.startswith(f"{cand}.layers.") for k in active_keys):
            return cand
    return ""


def canonicalize_hf_key(key: str, lp_root: str) -> str:
    """Normalizes multimodal language-model tensor names to `model.*` (and `lm_head.*`)."""
    if lp_root in ("model.language_model", "language_model.model") and key.startswith(f"{lp_root}."):
        return "model." + key[len(lp_root) + 1 :]
    if key.startswith("language_model.lm_head."):
        return key[len("language_model.") :]
    if lp_root == "" and key.startswith(("layers.", "norm.")):
        return f"model.{key}"
    return key


def dequantize_fp8_block_tensor(w_fp8: "torch.Tensor", scale_inv: "torch.Tensor") -> "torch.Tensor":
    """Dequantizes block-scaled (`[128,128]`, `[32,32]`, or per-row) `FP8` weights to `float32`."""
    w_f32 = w_fp8.to(torch.float32)
    s_f32 = scale_inv.to(device=w_f32.device, dtype=torch.float32)
    if w_f32.ndim == 2 and s_f32.ndim == 2:
        m, n = w_f32.shape
        sm, sn = s_f32.shape
        if sm == m and (sn == 1 or sn == n):
            return w_f32 * s_f32
        bm = (m + sm - 1) // sm
        bn = (n + sn - 1) // sn
        s_exp = s_f32.repeat_interleave(bm, dim=0)[:m, :].repeat_interleave(bn, dim=1)[:, :n]
        return (w_f32 * s_exp).to(torch.float16)
    if w_f32.ndim == 3 and s_f32.ndim == 3:
        e, m, n = w_f32.shape
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
    """Loads `key` from `fetcher` and automatically applies companion FP8 `weight_scale_inv` if present."""
    used_special.add(key)
    t = fetcher.pop_tensor(key, device=device)
    for s_sfx in ("_scale_inv", ".scale_inv"):
        sk = key + s_sfx
        if sk not in active_keys and key.endswith(".weight"):
            sk_alt = key[: -len(".weight")] + ".weight_scale_inv"
            if sk_alt in active_keys:
                sk = sk_alt
        if sk in active_keys and sk not in used_special:
            used_special.add(sk)
            s_inv = fetcher.pop_tensor(sk, device=device)
            return dequantize_fp8_block_tensor(t, s_inv)
    if hasattr(torch, "float8_e4m3fn") and t.dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    }:
        return t.to(torch.float32)
    return t


def discover_hf_moe_layers(
    num_layers: int, lp_root: str, active_keys: Set[str]
) -> List[Tuple[int, str]]:
    """Discovers `(layer_idx, container_attr)` for every layer containing routed MoE expert weights."""
    pfx_base = f"{lp_root}.layers." if lp_root else "layers."
    found: List[Tuple[int, str]] = []
    for l_idx in range(num_layers):
        lp = f"{pfx_base}{l_idx}"
        for attr in ("mlp", "feed_forward", "block_sparse_moe"):
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
    return torch.stack(gu_list, dim=0).to(q_dev), torch.stack(dn_list, dim=0).to(q_dev)


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
    skip_tokens = ("gate", "router", "lm_head", "embed", "norm", "a_proj", "b_proj", "indexer")
    replaced = 0

    def _walk(parent: "nn.Module", prefix: str = "") -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            low = full.lower()
            if any(tok in low.split(".")[-1] for tok in skip_tokens) and not low.endswith(
                ("gate_proj", "up_proj", "down_proj")
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
    """Attaches `paged_exp` (`DeimosPagedHFExperts`) to `layer_mod.<container_attr>` across HF MoE families."""
    moe_mod = getattr(layer_mod, container_attr)

    # Case 1: Llama-4 `Llama4TextMoe` (`router` returns dense sigmoid scatter; replace O(E) repeat with O(top_k)!)
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
            y = paged_exp(
                x_flat, top_idx, scores, shared_fn=lambda: moe_mod.shared_expert(x_flat)
            )
            return y, router_logits

        moe_mod.forward = _llama4_sparse_forward  # type: ignore[method-assign]
        return router_mod

    moe_mod.experts = paged_exp

    # Case 2: Qwen3.5 / Qwen3.6 / Qwen3-Next (`gate` + sigmoid-gated `shared_expert`)
    if hasattr(moe_mod, "gate") and hasattr(moe_mod, "shared_expert") and hasattr(moe_mod, "shared_expert_gate"):
        def _gated_shared_forward(hidden_states: "torch.Tensor") -> "torch.Tensor":
            B, L, D = hidden_states.shape
            x_flat = hidden_states.reshape(-1, D)
            g_out = moe_mod.gate(x_flat)
            rw, se = (g_out[1], g_out[2]) if isinstance(g_out, tuple) and len(g_out) >= 3 else (g_out[0], g_out[1])

            def _sh() -> "torch.Tensor":
                return torch.sigmoid(moe_mod.shared_expert_gate(x_flat)) * moe_mod.shared_expert(x_flat)

            return paged_exp(x_flat, se, rw, shared_fn=_sh).reshape(B, L, D)

        moe_mod.forward = _gated_shared_forward  # type: ignore[method-assign]
        return moe_mod.gate

    # Case 3: DeepSeek-V2 / DeepSeek-V3 / V3.2 / GLM-4.5 (`gate` + additive `shared_experts`)
    if (
        hasattr(moe_mod, "gate")
        and getattr(moe_mod, "shared_experts", None) is not None
        and not getattr(moe_mod, "is_hash", False)
    ):
        def _additive_shared_forward(hidden_states: "torch.Tensor") -> "torch.Tensor":
            orig_shape = hidden_states.shape
            x_flat = hidden_states.reshape(-1, orig_shape[-1])
            g_out = moe_mod.gate(hidden_states)
            rw, se = (g_out[1], g_out[2]) if isinstance(g_out, tuple) and len(g_out) >= 3 else (g_out[0], g_out[1])
            y = paged_exp(
                x_flat, se, rw, shared_fn=lambda: moe_mod.shared_experts(hidden_states).reshape(-1, orig_shape[-1])
            )
            return y.reshape(*orig_shape)

        moe_mod.forward = _additive_shared_forward  # type: ignore[method-assign]
        return moe_mod.gate

    return getattr(moe_mod, "gate", getattr(moe_mod, "router", None))
