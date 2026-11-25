#!/usr/bin/env python3
"""
DeimosStream Universal Attention & Backbone Loader (`src/deimos_stream/backbone.py`)
====================================================================================
Provides:
  1. `UniversalAttention`: Fused-QKV Apple Metal Attention supporting MHA/GQA and
     full, per-head (`Qwen3-MoE`), or absent (`Mixtral`) QK-Norm.
  2. `load_backbone`: Zero-copy `mmap` loader for `shared_core.bin` (`core_manifest.json`
     or canonical `OLMoE` layout) and 8-bit `lm_head_int8.bin`.
  3. Mach `task_info` RSS and MLX GPU memory helpers.
"""

from __future__ import annotations

import ctypes
import gc
import json
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
import numpy as np


def mlx_mem_mb() -> Tuple[float, float]:
    """Returns `(active_vram_mb, peak_vram_mb)` from MLX Metal allocator."""
    return (
        float(mx.get_active_memory()) / (1024.0 * 1024.0),
        float(mx.get_peak_memory()) / (1024.0 * 1024.0),
    )


def rss_mb() -> float:
    """Returns current macOS Mach resident set size (`resident_size`) in MiB."""

    class _TaskBasicInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", ctypes.c_uint64),
            ("system_time", ctypes.c_uint64),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    libc = ctypes.CDLL(None)
    info = _TaskBasicInfo()
    cnt = ctypes.c_uint32(ctypes.sizeof(info) // 4)
    if libc.task_info(libc.mach_task_self(), 20, ctypes.byref(info), ctypes.byref(cnt)) == 0:
        return float(info.resident_size) / (1024.0 * 1024.0)
    return 0.0


class UniversalAttention(nn.Module):
    """Fused-QKV Metal Attention supporting MHA/GQA and full, per-head, or absent QK-Norm."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float,
        rope_theta: float,
        qkv_w: mx.array,
        o_w: mx.array,
        q_norm_w: Optional[mx.array] = None,
        k_norm_w: Optional[mx.array] = None,
    ):
        super().__init__()
        self.n_heads = int(num_heads)
        self.n_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_w = mx.contiguous(qkv_w)
        self.o_w = mx.contiguous(o_w)
        self.q_norm: Optional[nn.RMSNorm] = None
        self.k_norm: Optional[nn.RMSNorm] = None
        self.per_head_qk_norm: bool = False

        if q_norm_w is not None and k_norm_w is not None:
            qn_dim = int(q_norm_w.shape[-1])
            kn_dim = int(k_norm_w.shape[-1])
            self.per_head_qk_norm = qn_dim == self.head_dim and self.n_heads > 1
            self.q_norm = nn.RMSNorm(qn_dim, eps=rms_eps)
            self.q_norm.weight = q_norm_w
            self.k_norm = nn.RMSNorm(kn_dim, eps=rms_eps)
            self.k_norm.weight = k_norm_w

        self.rope = initialize_rope(
            self.head_dim,
            rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=8192,
        )

    def __call__(
        self, x: mx.array, mask: Optional[mx.array], cache: Optional[KVCache]
    ) -> mx.array:
        B, L, _ = x.shape
        qkv = x @ self.qkv_w.T
        if self.q_dim == self.kv_dim:
            q, k, v = mx.split(qkv, 3, axis=-1)
        else:
            q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        if self.q_norm is not None and self.k_norm is not None and not self.per_head_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.reshape(B, L, self.n_heads, self.head_dim)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None and self.per_head_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(B, L, -1) @ self.o_w.T


SharedExpertTuple = Tuple[mx.array, mx.array, mx.array, Optional[mx.array]]


def _open_core_views(
    store_dir: Path, num_layers: int, num_experts: int, hidden_size: int
) -> Tuple[mmap.mmap, object, Dict[str, np.ndarray]]:
    core_f = open(store_dir / "shared_core.bin", "rb")
    core_mm = mmap.mmap(core_f.fileno(), 0, access=mmap.ACCESS_READ)
    cm_path = store_dir / "core_manifest.json"
    views: Dict[str, np.ndarray] = {}

    if cm_path.exists():
        meta = json.loads(cm_path.read_text())["tensors"]
        for cname, info in meta.items():
            shape = tuple(int(x) for x in info["shape"])
            numel = 1
            for d in shape:
                numel *= d
            views[cname] = np.frombuffer(
                core_mm, dtype=np.float16, count=numel, offset=int(info["offset"])
            ).reshape(shape)
        return core_mm, core_f, views

    specs: List[Tuple[str, str, Tuple[int, ...]]] = []
    for l_idx in range(num_layers):
        p = f"model.layers.{l_idx}"
        lp = f"layers.{l_idx}"
        specs.extend(
            [
                (f"{p}.input_layernorm.weight", f"{lp}.in_norm", (hidden_size,)),
                (f"{p}.mlp.gate.weight", f"{lp}.router", (num_experts, hidden_size)),
                (f"{p}.post_attention_layernorm.weight", f"{lp}.post_norm", (hidden_size,)),
                (f"{p}.self_attn.k_norm.weight", f"{lp}.k_norm", (hidden_size,)),
                (f"{p}.self_attn.k_proj.weight", f"{lp}.k_proj", (hidden_size, hidden_size)),
                (f"{p}.self_attn.o_proj.weight", f"{lp}.o_proj", (hidden_size, hidden_size)),
                (f"{p}.self_attn.q_norm.weight", f"{lp}.q_norm", (hidden_size,)),
                (f"{p}.self_attn.q_proj.weight", f"{lp}.q_proj", (hidden_size, hidden_size)),
                (f"{p}.self_attn.v_proj.weight", f"{lp}.v_proj", (hidden_size, hidden_size)),
            ]
        )
    specs.append(("model.norm.weight", "final_norm", (hidden_size,)))
    specs.sort(key=lambda it: it[0])
    off = 0
    for _, cname, shape in specs:
        numel = 1
        for d in shape:
            numel *= d
        views[cname] = np.frombuffer(
            core_mm, dtype=np.float16, count=numel, offset=off
        ).reshape(shape)
        off += numel * 2
    return core_mm, core_f, views


def load_backbone(
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
) -> Tuple[
    List[nn.RMSNorm],
    List[nn.RMSNorm],
    List[UniversalAttention],
    List[mx.array],
    List[Optional[SharedExpertTuple]],
    nn.RMSNorm,
    Tuple[mx.array, mx.array, mx.array],
]:
    """Loads `shared_core.bin` + 8-bit `lm_head_int8.bin` cleanly into MLX GPU arrays."""
    core_mm, core_f, views = _open_core_views(store_dir, num_layers, num_experts, hidden_size)
    try:
        in_norms: List[nn.RMSNorm] = []
        post_norms: List[nn.RMSNorm] = []
        attns: List[UniversalAttention] = []
        routers: List[mx.array] = []
        shared_mlps: List[Optional[SharedExpertTuple]] = []

        for l_idx in range(num_layers):
            lp = f"layers.{l_idx}"
            qkv_w = mx.concatenate(
                [
                    mx.array(views[f"{lp}.q_proj"]),
                    mx.array(views[f"{lp}.k_proj"]),
                    mx.array(views[f"{lp}.v_proj"]),
                ],
                axis=0,
            )
            qn = mx.array(views[f"{lp}.q_norm"]) if f"{lp}.q_norm" in views else None
            kn = mx.array(views[f"{lp}.k_norm"]) if f"{lp}.k_norm" in views else None
            attns.append(
                UniversalAttention(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rms_eps=rms_eps,
                    rope_theta=rope_theta,
                    qkv_w=qkv_w,
                    o_w=mx.array(views[f"{lp}.o_proj"]),
                    q_norm_w=qn,
                    k_norm_w=kn,
                )
            )
            in_n = nn.RMSNorm(hidden_size, eps=rms_eps)
            in_n.weight = mx.array(views[f"{lp}.in_norm"])
            post_n = nn.RMSNorm(hidden_size, eps=rms_eps)
            post_n.weight = mx.array(views[f"{lp}.post_norm"])
            in_norms.append(in_n)
            post_norms.append(post_n)
            routers.append(mx.contiguous(mx.array(views[f"{lp}.router"])))

            if f"{lp}.shared_gate_proj" in views:
                sg = mx.array(views[f"{lp}.shared_gate_proj"])
                su = mx.array(views[f"{lp}.shared_up_proj"])
                sd = mx.array(views[f"{lp}.shared_down_proj"])
                sgate = (
                    mx.array(views[f"{lp}.shared_gate"]) if f"{lp}.shared_gate" in views else None
                )
                shared_mlps.append((sg, su, sd, sgate))
            else:
                shared_mlps.append(None)

        final_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        final_norm.weight = mx.array(views["final_norm"])
        mx.eval(
            [n.parameters() for n in in_norms],
            [n.parameters() for n in post_norms],
            [a.parameters() for a in attns],
            routers,
            [x for s in shared_mlps if s is not None for x in s if x is not None],
            final_norm.parameters(),
        )
        mx.synchronize()
    finally:
        views.clear()
        gc.collect()
        core_mm.close()
        core_f.close()  # type: ignore[attr-defined]

    w_nbytes = vocab_size * hidden_size
    with open(store_dir / "lm_head_int8.bin", "rb") as f:
        lm_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            w_i8: Optional[np.ndarray] = np.frombuffer(
                lm_mm, dtype=np.int8, count=w_nbytes
            ).reshape(vocab_size, hidden_size)
            s_f16: Optional[np.ndarray] = np.frombuffer(
                lm_mm, dtype=np.float16, count=vocab_size, offset=w_nbytes
            ).reshape(vocab_size, 1)
            fp16_tmp = mx.array(w_i8).astype(mx.float16) * mx.array(s_f16)
            lm_qw, lm_qs, lm_qb = mx.quantize(fp16_tmp, group_size=64, bits=8)
            mx.eval(lm_qw, lm_qs, lm_qb)
            mx.synchronize()
            del fp16_tmp, w_i8, s_f16
            gc.collect()
        finally:
            lm_mm.close()

    return in_norms, post_norms, attns, routers, shared_mlps, final_norm, (lm_qw, lm_qs, lm_qb)
