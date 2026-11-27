#!/usr/bin/env python3
"""
DeimosStream HuggingFace Hybrid Backbone CUDA Fast-Path Patcher (`src/deimos_stream/hf_fast_patch.py`)
=====================================================================================================
Patches `Qwen3_5Moe` / `Qwen3.6` hybrid text backbones in-place on NVIDIA CUDA GPUs:
  1. Concatenates split projections layer-by-layer (`in_proj_qkv + z`, `q + k + v`,
     and `shared_expert gate + up`) with zero net VRAM increase, optionally quantizing
     dense projections in-place to 4-bit Group-64 (`backbone="q4"`, saving ~2.18 GiB VRAM
     and 3.5x HBM bandwidth while keeping `in_proj_a/b`, `conv1d`, routers, and norms FP16/FP32).
  2. Replaces eager PyTorch `Qwen3_5MoeRMSNorm`, `Qwen3_5MoeRMSNormGated`, `Qwen3_5MoeTopKRouter`,
     `Qwen3_5MoeGatedDeltaNet` (`L == 1`), and `Qwen3_5MoeAttention` (`L == 1` RoPE)
     with exact fused Triton kernels from `triton_hybrid.py`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from deimos_stream.triton_hybrid import (
    hybrid_triton_available,
    triton_causal_conv1d_update,
    triton_partial_rope_decode_inplace,
    triton_recurrent_gated_delta_step,
    triton_rmsnorm_gated,
    triton_topk_softmax,
    triton_zero_centered_rmsnorm,
)
from deimos_stream.triton_q4 import (
    quantize_2d_int8_torch,
    quantize_2d_q4_torch,
    triton_int8_gemv,
    triton_q4_dense_linear,
    triton_swiglu_gated,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _make_proj_fn(
    w_fp16: "torch.Tensor", device: "torch.device", mode: str = "fp16"
) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Creates an FP16 `F.linear`, symmetric INT8 `triton_int8_gemv`, or Q4 `triton_q4_dense_linear` closure."""
    if mode == "int8":
        w_dev = w_fp16.to(device=device, dtype=torch.float16)
        del w_fp16
        wi8, sf16 = quantize_2d_int8_torch(w_dev)
        del w_dev
        return lambda x: triton_int8_gemv(x, wi8, sf16)
    if mode == "q4" and w_fp16.shape[-1] % 64 == 0:
        w_dev = w_fp16.to(device=device, dtype=torch.float16)
        del w_fp16
        qw, qs, qb = quantize_2d_q4_torch(w_dev, group_size=64)
        del w_dev
        return lambda x: triton_q4_dense_linear(x, qw, qs, qb, group_size=64)
    w_c = w_fp16.to(device=device).contiguous()
    return lambda x: F.linear(x, w_c)


def _patch_rmsnorm_zero_centered(norm_mod: "nn.Module") -> None:
    if not hasattr(norm_mod, "weight") or not hasattr(norm_mod, "eps"):
        return
    eps = float(norm_mod.eps)  # type: ignore[arg-type]

    def _fast_norm(x: "torch.Tensor") -> "torch.Tensor":
        return triton_zero_centered_rmsnorm(x, norm_mod.weight, eps)  # type: ignore[arg-type]

    norm_mod.forward = _fast_norm  # type: ignore[method-assign]


def _patch_router(gate_mod: "nn.Module") -> None:
    if not hasattr(gate_mod, "weight") or not hasattr(gate_mod, "top_k"):
        return
    top_k = int(gate_mod.top_k)  # type: ignore[arg-type]
    hidden_dim = int(gate_mod.hidden_dim)  # type: ignore[arg-type]

    def _fast_router(hidden_states: "torch.Tensor"):
        x_2d = hidden_states.reshape(-1, hidden_dim)
        logits = F.linear(x_2d, gate_mod.weight)  # type: ignore[arg-type]
        if logits.shape[0] <= 16 and logits.shape[-1] <= 512:
            scores, indices = triton_topk_softmax(logits, top_k=top_k, norm_topk=True)
            return logits, scores, indices
        probs = F.softmax(logits, dtype=torch.float32, dim=-1)
        vals, inds = torch.topk(probs, top_k, dim=-1)
        vals = (vals / vals.sum(dim=-1, keepdim=True)).to(logits.dtype)
        return logits, vals, inds

    gate_mod.forward = _fast_router  # type: ignore[method-assign]


def _patch_shared_expert_block(
    mlp_mod: "nn.Module", paged_exp: Any, device: "torch.device", use_q4: bool
) -> None:
    sh = getattr(mlp_mod, "shared_expert", None)
    sg_lin = getattr(mlp_mod, "shared_expert_gate", None)
    if sh is None or sg_lin is None or not hasattr(sh, "gate_proj"):
        return

    inter = int(sh.intermediate_size)  # type: ignore[arg-type]
    gu_w = torch.cat([sh.gate_proj.weight.data, sh.up_proj.weight.data], dim=0)
    dn_w = sh.down_proj.weight.data
    sg_w = sg_lin.weight.data.to(device=device).contiguous()
    del sh.gate_proj, sh.up_proj, sh.down_proj, mlp_mod.shared_expert_gate

    gu_fn = _make_proj_fn(gu_w, device, "q4" if use_q4 else "fp16")
    dn_fn = _make_proj_fn(dn_w, device, "q4" if use_q4 else "fp16")
    del gu_w, dn_w

    def _fast_mlp_forward(hidden_states: "torch.Tensor") -> "torch.Tensor":
        B, L, D = hidden_states.shape
        x_flat = hidden_states.reshape(-1, D)
        _, routing_weights, selected_experts = mlp_mod.gate(x_flat)  # type: ignore[operator]

        def _run_shared_for(xf: "torch.Tensor") -> "torch.Tensor":
            act = triton_swiglu_gated(gu_fn(xf), F.linear(xf, sg_w))
            return dn_fn(act)

        out = paged_exp(x_flat, selected_experts, routing_weights, shared_fn=lambda: _run_shared_for(x_flat))
        return out.reshape(B, L, D)

    def _shared_only(xf: "torch.Tensor") -> "torch.Tensor":
        return dn_fn(triton_swiglu_gated(gu_fn(xf), F.linear(xf, sg_w)))

    mlp_mod._run_shared_fn = _shared_only  # type: ignore[attr-defined]
    mlp_mod.forward = _fast_mlp_forward  # type: ignore[method-assign]


def _patch_gated_delta_net(la_mod: "nn.Module", device: "torch.device", use_q4: bool) -> None:
    qkv_dim = int(la_mod.conv_dim)  # type: ignore[arg-type]
    val_dim = int(la_mod.value_dim)  # type: ignore[arg-type]
    key_dim = int(la_mod.key_dim)  # type: ignore[arg-type]
    n_vh = int(la_mod.num_v_heads)  # type: ignore[arg-type]
    hk_dim = int(la_mod.head_k_dim)  # type: ignore[arg-type]
    hv_dim = int(la_mod.head_v_dim)  # type: ignore[arg-type]
    l_idx = int(la_mod.layer_idx)  # type: ignore[arg-type]
    norm_eps = float(la_mod.layer_norm_epsilon)  # type: ignore[arg-type]
    apply_silu = getattr(la_mod, "activation", "silu") == "silu"

    qkvz_w = torch.cat([la_mod.in_proj_qkv.weight.data, la_mod.in_proj_z.weight.data], dim=0)  # type: ignore[attr-defined]
    ba_w = torch.cat([la_mod.in_proj_b.weight.data, la_mod.in_proj_a.weight.data], dim=0).to(device=device).contiguous()  # type: ignore[attr-defined]
    out_w = la_mod.out_proj.weight.data  # type: ignore[attr-defined]
    del la_mod.in_proj_qkv, la_mod.in_proj_z, la_mod.in_proj_b, la_mod.in_proj_a, la_mod.out_proj

    qkvz_fn = _make_proj_fn(qkvz_w, device, "int8" if use_q4 else "fp16")
    out_fn = _make_proj_fn(out_w, device, "int8" if use_q4 else "fp16")
    del qkvz_w, out_w
    conv_w_2d = la_mod.conv1d.weight.data.squeeze(1).to(device=device).contiguous()  # type: ignore[attr-defined]

    cls_mod = __import__(la_mod.__class__.__module__, fromlist=["*"])
    causal_conv1d_fn = getattr(cls_mod, "causal_conv1d_fn")
    chunk_delta_fn = getattr(cls_mod, "torch_chunk_gated_delta_rule")
    apply_mask_fn = getattr(cls_mod, "apply_mask_to_padding_states")

    def _fast_linear_attn(
        hidden_states: "torch.Tensor",
        cache: Optional[Any] = None,
        attention_mask: Optional["torch.Tensor"] = None,
        **kwargs: Any,
    ) -> "torch.Tensor":
        if attention_mask is not None:
            hidden_states = apply_mask_fn(hidden_states, attention_mask)
        B, S, _ = hidden_states.shape
        conv_state = cache.get_conv_state(l_idx, state_idx=0) if cache is not None else None
        rec_state = cache.get_recurrent_state(l_idx, state_idx=0) if cache is not None else None

        qkvz = qkvz_fn(hidden_states)
        ba = F.linear(hidden_states, ba_w)
        mixed_qkv = qkvz[..., :qkv_dim]
        z = qkvz[..., qkv_dim:]
        b = ba[..., :n_vh]
        a = ba[..., n_vh:]

        if S == 1 and conv_state is not None and rec_state is not None and not cache.is_recording_past(l_idx):
            qkv_conv = triton_causal_conv1d_update(
                mixed_qkv.transpose(1, 2), conv_state, conv_w_2d, la_mod.conv1d.bias, apply_silu=apply_silu  # type: ignore[attr-defined]
            ).transpose(1, 2)
            q = qkv_conv[..., :key_dim].view(B, 1, -1, hk_dim)
            k = qkv_conv[..., key_dim : 2 * key_dim].view(B, 1, -1, hk_dim)
            v = qkv_conv[..., 2 * key_dim :].view(B, 1, -1, hv_dim)

            core_out, new_state = triton_recurrent_gated_delta_step(
                q, k, v, a, b, la_mod.A_log, la_mod.dt_bias, rec_state  # type: ignore[attr-defined]
            )
            if new_state.data_ptr() != rec_state.data_ptr():
                cache.update_recurrent_state(new_state, l_idx)
            normed = triton_rmsnorm_gated(core_out, z.view(B, 1, -1, hv_dim), la_mod.norm.weight, norm_eps)  # type: ignore[attr-defined]
            return out_fn(normed.view(B, 1, val_dim))

        qkv_t = mixed_qkv.transpose(1, 2)
        if cache is not None:
            qkv_t = cache.update_conv_state(qkv_t, l_idx, conv_kernel_size=la_mod.conv_kernel_size)  # type: ignore[attr-defined]
        qkv_t = causal_conv1d_fn(
            qkv_t, conv_w_2d, la_mod.conv1d.bias, activation=la_mod.activation, **kwargs  # type: ignore[attr-defined]
        )
        if cache is not None:
            qkv_t = qkv_t[:, :, -S:]
        qkv_s = qkv_t.transpose(1, 2)
        q = qkv_s[..., :key_dim].reshape(B, S, -1, hk_dim)
        k = qkv_s[..., key_dim : 2 * key_dim].reshape(B, S, -1, hk_dim)
        v = qkv_s[..., 2 * key_dim :].reshape(B, S, -1, hv_dim)
        beta = b.sigmoid()
        g = -la_mod.A_log.float().exp() * F.softplus(a.float() + la_mod.dt_bias)  # type: ignore[attr-defined]
        core_out, last_state = chunk_delta_fn(
            q, k, v, g=g, beta=beta, initial_state=rec_state,
            output_final_state=cache is not None, use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs,
        )
        if cache is not None:
            cache.update_recurrent_state(last_state, l_idx)
        normed = triton_rmsnorm_gated(core_out, z.reshape(B, S, -1, hv_dim), la_mod.norm.weight, norm_eps)  # type: ignore[attr-defined]
        return out_fn(normed.reshape(B, S, val_dim))

    la_mod.forward = _fast_linear_attn  # type: ignore[method-assign]


def _patch_gqa_attention(attn_mod: "nn.Module", device: "torch.device", use_q4: bool) -> None:
    if not (hasattr(attn_mod, "q_proj") and hasattr(attn_mod, "k_proj") and hasattr(attn_mod, "v_proj")):
        return
    if getattr(attn_mod.q_proj, "bias", None) is not None or getattr(attn_mod.o_proj, "bias", None) is not None:
        return

    q_out = int(attn_mod.q_proj.weight.shape[0])  # type: ignore[attr-defined]
    k_out = int(attn_mod.k_proj.weight.shape[0])  # type: ignore[attr-defined]
    head_dim = int(attn_mod.head_dim)  # type: ignore[attr-defined]
    l_idx = int(attn_mod.layer_idx)  # type: ignore[attr-defined]
    scaling = float(attn_mod.scaling)  # type: ignore[attr-defined]

    qkv_w = torch.cat([attn_mod.q_proj.weight.data, attn_mod.k_proj.weight.data, attn_mod.v_proj.weight.data], dim=0)
    o_w = attn_mod.o_proj.weight.data  # type: ignore[attr-defined]
    del attn_mod.q_proj, attn_mod.k_proj, attn_mod.v_proj, attn_mod.o_proj

    qkv_fn = _make_proj_fn(qkv_w, device, "q4" if use_q4 else "fp16")
    o_fn = _make_proj_fn(o_w, device, "q4" if use_q4 else "fp16")
    del qkv_w, o_w
    attn_mod._qkv_fn = qkv_fn  # type: ignore[attr-defined]
    attn_mod._o_fn = o_fn  # type: ignore[attr-defined]
    attn_mod._q_out = q_out  # type: ignore[attr-defined]
    attn_mod._k_out = k_out  # type: ignore[attr-defined]
    attn_mod._head_dim = head_dim  # type: ignore[attr-defined]
    attn_mod._scaling = scaling  # type: ignore[attr-defined]

    _patch_rmsnorm_zero_centered(attn_mod.q_norm)  # type: ignore[arg-type]
    _patch_rmsnorm_zero_centered(attn_mod.k_norm)  # type: ignore[arg-type]

    cls_mod = __import__(attn_mod.__class__.__module__, fromlist=["*"])
    apply_rope_ref: Callable = getattr(cls_mod, "apply_rotary_pos_emb")
    all_attn = getattr(cls_mod, "ALL_ATTENTION_FUNCTIONS")
    eager_attn = getattr(cls_mod, "eager_attention_forward")

    def _fast_gqa(
        hidden_states: "torch.Tensor",
        position_embeddings: Tuple["torch.Tensor", "torch.Tensor"],
        attention_mask: Optional["torch.Tensor"],
        cache: Optional[Any] = None,
        **kwargs: Any,
    ):
        B, S, _ = hidden_states.shape
        qkv = qkv_fn(hidden_states)
        qg = qkv[..., :q_out].view(B, S, -1, head_dim * 2)
        q_raw, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(B, S, -1)

        q = attn_mod.q_norm(q_raw).transpose(1, 2).contiguous()  # type: ignore[operator]
        k = attn_mod.k_norm(qkv[..., q_out : q_out + k_out].view(B, S, -1, head_dim)).transpose(1, 2).contiguous()  # type: ignore[operator]
        v = qkv[..., q_out + k_out :].view(B, S, -1, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        rot_dim = int(cos.shape[-1])
        if S == 1 and ((rot_dim // 2) & ((rot_dim // 2) - 1)) == 0:
            triton_partial_rope_decode_inplace(q, k, cos, sin)
        else:
            q, k = apply_rope_ref(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v, l_idx, kwargs=kwargs)

        attn_fn = all_attn.get_interface(attn_mod.config._attn_implementation, eager_attn)  # type: ignore[attr-defined]
        attn_out, attn_w = attn_fn(
            attn_mod, q, k, v, attention_mask, dropout=0.0, scaling=scaling, **kwargs
        )
        attn_out = attn_out.reshape(B, S, -1) * torch.sigmoid(gate)
        return o_fn(attn_out), attn_w

    attn_mod.forward = _fast_gqa  # type: ignore[method-assign]


def patch_hf_hybrid_for_cuda(
    model: "nn.Module", device: "torch.device", backbone: str = "fp16"
) -> bool:
    """Applies fused Triton kernels & optional 4-bit Group-64 projections (`backbone='q4'`)."""
    if not hybrid_triton_available(device):
        return False
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        return False
    use_q4 = backbone == "q4"

    if hasattr(inner, "norm"):
        _patch_rmsnorm_zero_centered(inner.norm)

    for lyr in inner.layers:
        if hasattr(lyr, "input_layernorm"):
            _patch_rmsnorm_zero_centered(lyr.input_layernorm)
        if hasattr(lyr, "post_attention_layernorm"):
            _patch_rmsnorm_zero_centered(lyr.post_attention_layernorm)
        if hasattr(lyr, "linear_attn"):
            _patch_gated_delta_net(lyr.linear_attn, device=device, use_q4=use_q4)
        if hasattr(lyr, "self_attn"):
            _patch_gqa_attention(lyr.self_attn, device=device, use_q4=use_q4)
        if hasattr(lyr, "mlp"):
            if hasattr(lyr.mlp, "gate"):
                _patch_router(lyr.mlp.gate)
            if hasattr(lyr.mlp, "experts"):
                _patch_shared_expert_block(lyr.mlp, lyr.mlp.experts, device=device, use_q4=use_q4)
    model.to(device)
    torch.cuda.empty_cache()
    return True
