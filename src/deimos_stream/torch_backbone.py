#!/usr/bin/env python3
"""
DeimosStream PyTorch (`CUDA` / `CPU`) Attention & Backbone Loader (`src/deimos_stream/torch_backbone.py`)
========================================================================================================
Provides:
  1. `rms_norm` & NeoX/Llama split-half RoPE (`apply_split_half_rope`, matching MLX `traditional=False`).
  2. `TorchUniversalAttention` using PyTorch FlashAttention-2 / cuDNN `F.scaled_dot_product_attention`
     with MHA/GQA and full, per-head (`Qwen3-MoE`), or absent (`Mixtral`) QK-Norm.
  3. `load_torch_backbone` zero-copy `mmap` loader for `shared_core.bin` and `lm_head_int8.bin`.
"""

from __future__ import annotations

import gc
import mmap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from deimos_stream.backbone import _open_core_views
from deimos_stream.torch_arena import HAS_TORCH
from deimos_stream.torch_kv_cache import TorchKVCache

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
else:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


TorchSharedExpertTuple = Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]


def rms_norm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    """FP32-variance RMSNorm returning activations in `weight.dtype`."""
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(var + eps)).to(weight.dtype) * weight


def apply_split_half_rope(
    q: "torch.Tensor", k: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Applies NeoX/Llama split-half RoPE (`traditional=False`) matching MLX `initialize_rope`."""

    def _rot(t: torch.Tensor) -> torch.Tensor:
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    q_out = (q.float() * cos + _rot(q.float()) * sin).to(q.dtype)
    k_out = (k.float() * cos + _rot(k.float()) * sin).to(k.dtype)
    return q_out, k_out


class TorchUniversalAttention:
    """Fused-QKV PyTorch SDPA Attention supporting MHA/GQA and full, per-head, or absent QK-Norm."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float,
        inv_freq: "torch.Tensor",
        qkv_w: "torch.Tensor",
        o_w: "torch.Tensor",
        q_norm_w: Optional["torch.Tensor"] = None,
        k_norm_w: Optional["torch.Tensor"] = None,
    ):
        self.n_heads = int(num_heads)
        self.n_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.rms_eps = float(rms_eps)
        self.inv_freq = inv_freq
        self.qkv_w = qkv_w
        self.o_w = o_w
        self.q_norm_w = q_norm_w
        self.k_norm_w = k_norm_w
        self.per_head = (
            q_norm_w is not None and int(q_norm_w.shape[-1]) == self.head_dim and self.n_heads > 1
        )

    def __call__(self, x: "torch.Tensor", cache: Optional[TorchKVCache]) -> "torch.Tensor":
        B, L, _ = x.shape
        qkv = x @ self.qkv_w.T
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        if self.q_norm_w is not None and self.k_norm_w is not None and not self.per_head:
            q = rms_norm(q, self.q_norm_w, self.rms_eps)
            k = rms_norm(k, self.k_norm_w, self.rms_eps)

        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_kv_heads, self.head_dim)
        if self.q_norm_w is not None and self.k_norm_w is not None and self.per_head:
            q = rms_norm(q, self.q_norm_w, self.rms_eps)
            k = rms_norm(k, self.k_norm_w, self.rms_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        pos_offset = cache.offset if cache is not None else 0
        pos = torch.arange(pos_offset, pos_offset + L, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[None, None, :, :]
        q, k = apply_split_half_rope(q, k, emb.cos(), emb.sin())

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_attn, v_attn = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        else:
            k_attn, v_attn = k, v

        is_causal = L > 1 and pos_offset == 0
        attn_mask: Optional[torch.Tensor] = None
        if L > 1 and pos_offset > 0:
            total_k = k_attn.shape[2]
            attn_mask = torch.ones((L, total_k), dtype=torch.bool, device=x.device).tril(
                diagonal=pos_offset
            )
        out = F.scaled_dot_product_attention(
            q, k_attn, v_attn, attn_mask=attn_mask, is_causal=is_causal
        )
        return out.transpose(1, 2).contiguous().view(B, L, self.q_dim) @ self.o_w.T


def load_torch_backbone(
    store_dir: Path,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    rms_eps: float,
    rope_theta: float,
    device: "torch.device",
    compute_dtype: "torch.dtype",
) -> Tuple[
    List["torch.Tensor"],
    List["torch.Tensor"],
    List[TorchUniversalAttention],
    List["torch.Tensor"],
    List[Optional[TorchSharedExpertTuple]],
    "torch.Tensor",
    Tuple["torch.Tensor", "torch.Tensor"],
]:
    """Loads `shared_core.bin` + INT8 `lm_head_int8.bin` into PyTorch CUDA/CPU tensors."""
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    core_mm, core_f, views = _open_core_views(store_dir, num_layers, num_experts, hidden_size)
    try:
        in_norms: List[torch.Tensor] = []
        post_norms: List[torch.Tensor] = []
        attns: List[TorchUniversalAttention] = []
        routers: List[torch.Tensor] = []
        shared_mlps: List[Optional[TorchSharedExpertTuple]] = []

        for l_i in range(num_layers):
            lp = f"layers.{l_i}"
            in_norms.append(torch.from_numpy(views[f"{lp}.in_norm"].copy()).to(device, compute_dtype))
            post_norms.append(
                torch.from_numpy(views[f"{lp}.post_norm"].copy()).to(device, compute_dtype)
            )
            qkv = np.concatenate(
                [views[f"{lp}.q_proj"], views[f"{lp}.k_proj"], views[f"{lp}.v_proj"]], axis=0
            )
            qn = (
                torch.from_numpy(views[f"{lp}.q_norm"].copy()).to(device, compute_dtype)
                if f"{lp}.q_norm" in views
                else None
            )
            kn = (
                torch.from_numpy(views[f"{lp}.k_norm"].copy()).to(device, compute_dtype)
                if f"{lp}.k_norm" in views
                else None
            )
            attns.append(
                TorchUniversalAttention(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rms_eps=rms_eps,
                    inv_freq=inv_freq,
                    qkv_w=torch.from_numpy(qkv).to(device, compute_dtype),
                    o_w=torch.from_numpy(views[f"{lp}.o_proj"].copy()).to(device, compute_dtype),
                    q_norm_w=qn,
                    k_norm_w=kn,
                )
            )
            routers.append(torch.from_numpy(views[f"{lp}.router"].copy()).to(device, compute_dtype))
            if f"{lp}.shared_gate_proj" in views:
                sg = torch.from_numpy(views[f"{lp}.shared_gate_proj"].copy()).to(
                    device, compute_dtype
                )
                su = torch.from_numpy(views[f"{lp}.shared_up_proj"].copy()).to(
                    device, compute_dtype
                )
                sd = torch.from_numpy(views[f"{lp}.shared_down_proj"].copy()).to(
                    device, compute_dtype
                )
                sgate = (
                    torch.from_numpy(views[f"{lp}.shared_gate"].copy()).to(device, compute_dtype)
                    if f"{lp}.shared_gate" in views
                    else None
                )
                shared_mlps.append((sg, su, sd, sgate))
            else:
                shared_mlps.append(None)

        final_norm = torch.from_numpy(views["final_norm"].copy()).to(device, compute_dtype)
    finally:
        views.clear()
        gc.collect()
        core_mm.close()
        core_f.close()  # type: ignore[attr-defined]

    w_nbytes = vocab_size * hidden_size
    with open(store_dir / "lm_head_int8.bin", "rb") as f:
        lm_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            w_i8 = (
                np.frombuffer(lm_mm, dtype=np.int8, count=w_nbytes)
                .reshape(vocab_size, hidden_size)
                .copy()
            )
            s_f16 = (
                np.frombuffer(lm_mm, dtype=np.float16, count=vocab_size, offset=w_nbytes)
                .reshape(vocab_size, 1)
                .copy()
            )
        finally:
            lm_mm.close()

    lm_head = (
        torch.from_numpy(w_i8).to(device),
        torch.from_numpy(s_f16).to(device, compute_dtype),
    )
    return in_norms, post_norms, attns, routers, shared_mlps, final_norm, lm_head
