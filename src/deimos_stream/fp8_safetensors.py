#!/usr/bin/env python3
"""
DeimosStream Zero-Copy Raw `.safetensors` FP8 Index & `F_NOCACHE` Reader
(`src/deimos_stream/fp8_safetensors.py`)
========================================================================
Parses HuggingFace single-file or multi-shard `.safetensors` headers directly in binary
(`< 50 ms` for 62 shards / 306 GiB) without duplicating weights on disk (`0` extra GB):
  - Indexes exact `[byte_offset, nbytes, dtype, shape]` per tensor across shards.
  - Reads routed `FP8 E4M3` expert weights (`[128, 128]` `weight_scale_inv` or `UE8M0` `.scale`)
    on demand via `os.pread` over `F_NOCACHE` descriptors on macOS (preventing XNU Unified
    Memory file-cache bloat and swap) or `POSIX_FADV_DONTNEED` on Linux.
  - Exposes zero-VRAM `mmap` token embedding lookups and streamed INT8 `lm_head` quantization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from deimos_stream.model_adapters import (
    canonicalize_hf_key,
    dequantize_fp8_block_tensor,
    detect_lm_prefix,
    discover_hf_moe_layers,
    is_companion_scale_key,
    is_ignored_checkpoint_key,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

_MACOS_F_NOCACHE = 48

_E4M3_LUT_F32: np.ndarray = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _sign = -1.0 if (_b & 0x80) else 1.0
    _exp = (_b >> 3) & 0x0F
    _man = _b & 0x07
    if _exp == 0x0F and _man == 0x07:
        _E4M3_LUT_F32[_b] = 0.0
    elif _exp == 0:
        _E4M3_LUT_F32[_b] = _sign * (_man / 8.0) * (2.0**-6)
    else:
        _E4M3_LUT_F32[_b] = _sign * (1.0 + _man / 8.0) * (2.0 ** (_exp - 7))

_POS_E4M3_VALS: np.ndarray = _E4M3_LUT_F32[:127].copy()
_POS_E4M3_MIDS: np.ndarray = 0.5 * (_POS_E4M3_VALS[:-1] + _POS_E4M3_VALS[1:])


@dataclass(frozen=True)
class TensorSpan:
    """Byte address and metadata of a single tensor inside a `.safetensors` shard."""

    shard_idx: int
    offset: int
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]


def is_raw_fp8_moe_checkpoint(store_dir: Path) -> bool:
    """Returns `True` if `store_dir` is an unconverted HuggingFace FP8 Sparse/Hybrid MoE checkpoint."""
    sdir = Path(store_dir).expanduser().resolve()
    if not sdir.is_dir() or (sdir / "merkle_manifest.json").exists():
        return False
    cfg_p = sdir / "config.json"
    if not cfg_p.exists():
        return False
    has_st = (sdir / "model.safetensors.index.json").exists() or any(sdir.glob("*.safetensors"))
    if not has_st:
        return False
    raw = json.loads(cfg_p.read_text())
    t_cfg = raw.get("text_config", raw)
    if not isinstance(t_cfg, dict):
        return False
    has_moe = any(
        int(t_cfg.get(k) or 0) > 1 for k in ("num_experts", "num_local_experts", "n_routed_experts")
    )
    if not has_moe:
        return False
    q_cfg = raw.get("quantization_config") or t_cfg.get("quantization_config") or {}
    if isinstance(q_cfg, dict) and str(q_cfg.get("quant_method", "")).lower() in {"fp8", "mxfp8", "finegrained_fp8"}:
        return True
    idx_p = sdir / "model.safetensors.index.json"
    if idx_p.exists():
        wm = json.loads(idx_p.read_text()).get("weight_map", {})
        return any(k.endswith(("_scale_inv", ".weight_scale_inv", ".scale")) for k in wm)
    return False


def _encode_f32_to_fp8_block(arr_f32: np.ndarray, bm: int, bn: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantizes an unquantized 2D `[M, N]` float32 matrix into `(u8_e4m3, scale_f32 [M/bm, N/bn])`."""
    m, n = arr_f32.shape
    gm, gn = m // bm, n // bn
    blocks = arr_f32.reshape(gm, bm, gn, bn)
    scale = (np.max(np.abs(blocks), axis=(1, 3)).clip(min=1e-6) / 448.0).astype(np.float32)
    normed = np.clip(blocks / scale[:, None, :, None], -448.0, 448.0).reshape(m, n)
    sign_bit = np.where(normed < 0.0, np.uint8(0x80), np.uint8(0))
    codes = np.searchsorted(_POS_E4M3_MIDS, np.abs(normed)).astype(np.uint8)
    return (sign_bit | codes), scale


def _normalize_scale_grid(
    scale_f32: np.ndarray, out_rows: int, in_cols: int, target_bm: int, target_bn: int
) -> np.ndarray:
    """Normalizes scalar, per-row, or coarse block scales to a `[out_rows // bm, in_cols // bn]` float32 grid."""
    gm, gn = out_rows // target_bm, in_cols // target_bn
    if scale_f32.ndim == 0 or scale_f32.size == 1:
        return np.full((gm, gn), float(scale_f32.reshape(-1)[0]), dtype=np.float32)
    if scale_f32.ndim == 1:
        scale_f32 = scale_f32.reshape(-1, 1)
    sm, sn = scale_f32.shape
    if sm == gm and sn == gn:
        return np.ascontiguousarray(scale_f32, dtype=np.float32)
    if sm == out_rows and sn == 1:
        row_blk = scale_f32.reshape(gm, target_bm).mean(axis=1, keepdims=True)
        return np.broadcast_to(row_blk, (gm, gn)).copy().astype(np.float32)
    rep_m = max(1, gm // max(1, sm))
    rep_n = max(1, gn // max(1, sn))
    out = np.repeat(np.repeat(scale_f32, rep_m, axis=0)[:gm, :], rep_n, axis=1)[:, :gn]
    return np.ascontiguousarray(out, dtype=np.float32)


class SafetensorsZeroCopyIndex:
    """Zero-copy `.safetensors` binary index reading FP8 experts directly via `F_NOCACHE` `os.pread`."""

    def __init__(self, checkpoint_dir: Path, use_nocache: bool = True):
        self.root = Path(checkpoint_dir).expanduser().resolve()
        self.raw_cfg: Dict[str, object] = json.loads((self.root / "config.json").read_text())
        t_raw = self.raw_cfg.get("text_config", self.raw_cfg)
        assert isinstance(t_raw, dict)
        self.text_cfg: Dict[str, object] = dict(t_raw)

        self.num_total_layers = int(self.text_cfg["num_hidden_layers"])  # type: ignore[arg-type]
        self.hidden_size = int(self.text_cfg["hidden_size"])  # type: ignore[arg-type]
        self.inter_size = int(
            self.text_cfg.get("moe_intermediate_size")
            or self.text_cfg.get("ffn_hidden_size")
            or self.text_cfg["intermediate_size"]  # type: ignore[arg-type]
        )
        self.num_experts = int(
            self.text_cfg.get("num_experts")
            or self.text_cfg.get("num_local_experts")
            or self.text_cfg["n_routed_experts"]  # type: ignore[arg-type]
        )
        self.vocab_size = int(self.text_cfg["vocab_size"])  # type: ignore[arg-type]
        self.top_k = int(
            self.text_cfg.get("num_experts_per_tok") or self.text_cfg.get("num_experts_per_token") or 8  # type: ignore[arg-type]
        )

        idx_p = self.root / "model.safetensors.index.json"
        if idx_p.exists():
            wm: Dict[str, str] = dict(json.loads(idx_p.read_text())["weight_map"])
            shard_names = sorted(set(wm.values()))
        else:
            shard_names = [p.name for p in sorted(self.root.glob("*.safetensors"))]
            if not shard_names:
                raise FileNotFoundError(f"No .safetensors files found in {self.root}")

        self.shard_paths: List[Path] = [self.root / s for s in shard_names]
        self._fds: List[int] = []
        self.spans: Dict[str, TensorSpan] = {}
        for s_idx, sp in enumerate(self.shard_paths):
            fd = os.open(sp, os.O_RDONLY)
            if use_nocache and sys.platform == "darwin":
                try:
                    fcntl.fcntl(fd, _MACOS_F_NOCACHE, 1)
                except OSError:
                    pass
            self._fds.append(fd)
            hlen = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            hdr = json.loads(os.pread(fd, hlen, 8).decode("utf-8"))
            data_base = 8 + hlen
            for k, meta in hdr.items():
                if k == "__metadata__":
                    continue
                b0, b1 = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
                self.spans[k] = TensorSpan(
                    shard_idx=s_idx,
                    offset=data_base + b0,
                    nbytes=b1 - b0,
                    dtype=str(meta["dtype"]),
                    shape=tuple(int(d) for d in meta["shape"]),
                )

        self.active_keys: Set[str] = {
            k for k in self.spans if not is_ignored_checkpoint_key(k, self.num_total_layers)
        }
        self.lp_root = detect_lm_prefix(self.active_keys)
        self.moe_specs = discover_hf_moe_layers(self.num_total_layers, self.lp_root, self.active_keys)
        if not self.moe_specs:
            raise ValueError(f"No routed MoE layers found in {self.root}")
        self.moe_layer_ids = [li for li, _ in self.moe_specs]
        self.moe_containers = [ca for _, ca in self.moe_specs]
        self.num_moe_layers = len(self.moe_layer_ids)

        self.block_m, self.block_n = self._detect_block_shape()
        self.expert_bytes = (
            (2 * self.inter_size * self.hidden_size)
            + (self.hidden_size * self.inter_size)
            + 4 * ((2 * self.inter_size // self.block_m) * (self.hidden_size // self.block_n))
            + 4 * ((self.hidden_size // self.block_m) * (self.inter_size // self.block_n))
        )
        self._io_pool = ThreadPoolExecutor(max_workers=4)
        self._emb_file: Optional[object] = None
        self._emb_mmap: Optional[mmap.mmap] = None

    def _detect_block_shape(self) -> Tuple[int, int]:
        D, I = self.hidden_size, self.inter_size
        q_cfg = self.raw_cfg.get("quantization_config") or self.text_cfg.get("quantization_config") or {}
        if isinstance(q_cfg, dict):
            wbs = q_cfg.get("weight_block_size") or q_cfg.get("block_size")
            if isinstance(wbs, list) and len(wbs) == 2:
                bm, bn = int(wbs[0]), int(wbs[1])
                if I % bm == 0 and D % bm == 0 and D % bn == 0 and I % bn == 0:
                    return bm, bn
        l0, c0 = self.moe_specs[0]
        ep = f"{self.lp_root + '.' if self.lp_root else ''}layers.{l0}.{c0}.experts"
        for cand, rows, cols in (
            (f"{ep}.0.gate_proj.weight", I, D),
            (f"{ep}.0.w1.weight", I, D),
            (f"{ep}.gate_up_proj", 2 * I, D),
            (f"{ep}.gate_up_proj.weight", 2 * I, D),
        ):
            if cand in self.spans:
                sk = self._find_scale_key(cand)
                if sk is not None:
                    shp = self.spans[sk].shape[-2:]
                    if len(shp) == 2 and shp[0] > 0 and shp[1] > 0:
                        bm, bn = rows // shp[0], cols // shp[1]
                        if bm >= 1 and bn >= 1 and I % bm == 0 and D % bm == 0 and D % bn == 0 and I % bn == 0:
                            return bm, bn
        bs = next(b for b in (128, 64, 32, 16, 8, 4, 1) if I % b == 0 and D % b == 0)
        return bs, bs

    def _find_scale_key(self, weight_key: str) -> Optional[str]:
        cands = [f"{weight_key}_scale_inv", f"{weight_key}.scale_inv"]
        if weight_key.endswith(".weight"):
            stem = weight_key[: -len(".weight")]
            cands.extend([f"{stem}.weight_scale_inv", f"{stem}.scale"])
        return next((sk for sk in cands if sk in self.spans), None)

    def read_raw_bytes(self, span: TensorSpan, sub_offset: int = 0, sub_nbytes: Optional[int] = None) -> bytes:
        nb = span.nbytes if sub_nbytes is None else int(sub_nbytes)
        off = span.offset + int(sub_offset)
        fd = self._fds[span.shard_idx]
        buf = os.pread(fd, nb, off)
        if sys.platform.startswith("linux") and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(fd, off & ~4095, (nb + 8191) & ~4095, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        return buf

    def _raw_to_f32(self, raw: bytes, dtype_str: str, shape: Tuple[int, ...]) -> np.ndarray:
        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            return (u16.astype(np.uint32) << 16).view(np.float32)
        if dtype_str == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        if dtype_str == "F8_E4M3":
            return _E4M3_LUT_F32[np.frombuffer(raw, dtype=np.uint8).reshape(shape)]
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()

    def read_scale_f32(self, scale_key: str, slice_idx: Optional[int] = None) -> np.ndarray:
        sp = self.spans[scale_key]
        elem_b = 1 if sp.dtype in ("U8", "F8_E8M0") else (2 if sp.dtype in ("F16", "BF16") else 4)
        if slice_idx is not None and len(sp.shape) == 3:
            stride = sp.shape[1] * sp.shape[2] * elem_b
            raw = self.read_raw_bytes(sp, sub_offset=slice_idx * stride, sub_nbytes=stride)
            shp = sp.shape[1:]
        else:
            raw = self.read_raw_bytes(sp)
            shp = sp.shape
        if sp.dtype in ("U8", "F8_E8M0"):
            u8 = np.frombuffer(raw, dtype=np.uint8).reshape(shp)
            return np.exp2(u8.astype(np.float32) - 127.0)
        return self._raw_to_f32(raw, sp.dtype, shp)

    def _read_matrix_fp8(
        self, key: str, rows: int, cols: int, slice_idx: Optional[int] = None, transposed: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        sp = self.spans[key]
        bm, bn = self.block_m, self.block_n
        elem_b = 1 if sp.dtype == "F8_E4M3" else (2 if sp.dtype in ("F16", "BF16") else 4)
        raw_shape = (cols, rows) if transposed else (rows, cols)
        stride = rows * cols * elem_b
        raw = self.read_raw_bytes(sp, (slice_idx or 0) * stride, stride) if slice_idx is not None else self.read_raw_bytes(sp)
        sk = self._find_scale_key(key)
        if sp.dtype == "F8_E4M3" and not transposed:
            u8 = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols).copy()
            sc = _normalize_scale_grid(self.read_scale_f32(sk, slice_idx), rows, cols, bm, bn) if sk else np.ones((rows // bm, cols // bn), dtype=np.float32)
            return u8, sc
        arr_f32 = self._raw_to_f32(raw, sp.dtype, raw_shape)
        if sk is not None:
            sc_raw = self.read_scale_f32(sk, slice_idx)
            arr_f32 = arr_f32 * _normalize_scale_grid(sc_raw, raw_shape[0], raw_shape[1], bm, bn).repeat(bm, 0).repeat(bn, 1)
        if transposed:
            arr_f32 = np.ascontiguousarray(arr_f32.T)
        return _encode_f32_to_fp8_block(arr_f32, bm, bn)

    def read_expert_fp8(
        self, moe_idx: int, expert_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reads `(gu_u8 [2I, D], gu_sc [2I/bm, D/bn], dn_u8 [D, I], dn_sc [D/bm, I/bn])` without dequantizing."""
        l_idx, attr = self.moe_layer_ids[moe_idx], self.moe_containers[moe_idx]
        lp = f"{self.lp_root}.layers.{l_idx}" if self.lp_root else f"layers.{l_idx}"
        ep = f"{lp}.{attr}.experts"
        D, I = self.hidden_size, self.inter_size

        gu_3d = f"{ep}.gate_up_proj" if f"{ep}.gate_up_proj" in self.spans else f"{ep}.gate_up_proj.weight"
        if gu_3d in self.spans:
            dn_3d = f"{ep}.down_proj" if f"{ep}.down_proj" in self.spans else f"{ep}.down_proj.weight"
            sp_dn = self.spans[dn_3d]
            trans = sp_dn.shape[1] == I and sp_dn.shape[2] == D and I != D
            gu_u8, gu_sc = self._read_matrix_fp8(gu_3d, 2 * I, D, slice_idx=expert_idx, transposed=trans)
            dn_u8, dn_sc = self._read_matrix_fp8(dn_3d, D, I, slice_idx=expert_idx, transposed=trans)
            return gu_u8, gu_sc, dn_u8, dn_sc

        if f"{ep}.{expert_idx}.gate_proj.weight" in self.spans:
            gk, uk, dk = (
                f"{ep}.{expert_idx}.gate_proj.weight",
                f"{ep}.{expert_idx}.up_proj.weight",
                f"{ep}.{expert_idx}.down_proj.weight",
            )
        else:
            gk, uk, dk = (
                f"{ep}.{expert_idx}.w1.weight",
                f"{ep}.{expert_idx}.w3.weight",
                f"{ep}.{expert_idx}.w2.weight",
            )
        g_u8, gs = self._read_matrix_fp8(gk, I, D)
        u_u8, us = self._read_matrix_fp8(uk, I, D)
        d_u8, ds = self._read_matrix_fp8(dk, D, I)
        return np.concatenate([g_u8, u_u8], axis=0), np.concatenate([gs, us], axis=0), d_u8, ds

    def read_torch_dequant(self, key: str, compute_dtype: "torch.dtype") -> "torch.Tensor":
        """Loads a non-expert tensor into PyTorch, dequantizing FP8 if accompanied by a block scale."""
        sp = self.spans[key]
        raw = bytearray(self.read_raw_bytes(sp))
        if sp.dtype == "F8_E4M3":
            u8 = torch.frombuffer(raw, dtype=torch.uint8).reshape(sp.shape)
            sk = self._find_scale_key(key)
            w_f32 = torch.from_numpy(_E4M3_LUT_F32[u8.numpy()])
            if sk is not None:
                s_t = torch.from_numpy(self.read_scale_f32(sk))
                return dequantize_fp8_block_tensor(w_f32, s_t).to(compute_dtype)
            return w_f32.to(compute_dtype)
        if sp.dtype == "BF16":
            return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(sp.shape).to(compute_dtype).clone()
        if sp.dtype == "F16":
            return torch.frombuffer(raw, dtype=torch.float16).reshape(sp.shape).to(compute_dtype).clone()
        return torch.frombuffer(raw, dtype=torch.float32).reshape(sp.shape).clone()

    def _embedding_key(self) -> str:
        cands = (
            f"{self.lp_root}.embed_tokens.weight" if self.lp_root else "embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "embed.weight",
        )
        return next(k for k in cands if k in self.spans)

    def open_embedding_mmap(self) -> Tuple[np.ndarray, str]:
        """Maps `embed_tokens.weight` directly inside its `.safetensors` shard with zero RAM copy."""
        emb_key = self._embedding_key()
        sp = self.spans[emb_key]
        f_obj = open(self.shard_paths[sp.shard_idx], "rb")
        mm = mmap.mmap(f_obj.fileno(), 0, access=mmap.ACCESS_READ)
        self._emb_file, self._emb_mmap = f_obj, mm
        np_dt = np.uint16 if sp.dtype == "BF16" else (np.float16 if sp.dtype == "F16" else np.float32)
        view = np.frombuffer(mm, dtype=np_dt, count=self.vocab_size * self.hidden_size, offset=sp.offset).reshape(
            self.vocab_size, self.hidden_size
        )
        return view, sp.dtype

    def quantize_lm_head_int8(self, tile_rows: int = 16384) -> Tuple[np.ndarray, np.ndarray]:
        """Streams `lm_head.weight` (or tied embeddings) in `16K`-row tiles into `(w_i8, scale_f16)`."""
        lm_cands = ("lm_head.weight", "language_model.lm_head.weight", "head.weight", self._embedding_key())
        lm_key = next(k for k in lm_cands if k in self.spans)
        sp = self.spans[lm_key]
        sk = self._find_scale_key(lm_key)
        sc_full = self.read_scale_f32(sk) if sk is not None else None
        elem_b = 1 if sp.dtype == "F8_E4M3" else (2 if sp.dtype in ("F16", "BF16") else 4)
        V, D = self.vocab_size, self.hidden_size
        w_i8 = np.empty((V, D), dtype=np.int8)
        s_f16 = np.empty((V, 1), dtype=np.float16)
        for v0 in range(0, V, tile_rows):
            v1 = min(V, v0 + tile_rows)
            raw = self.read_raw_bytes(sp, sub_offset=v0 * D * elem_b, sub_nbytes=(v1 - v0) * D * elem_b)
            ch = self._raw_to_f32(raw, sp.dtype, (v1 - v0, D))
            if sc_full is not None:
                bm = max(1, V // sc_full.shape[0])
                bn = max(1, D // sc_full.shape[1])
                ch = ch * np.repeat(np.repeat(sc_full[v0 // bm : math.ceil(v1 / bm)], bm, axis=0)[: v1 - v0], bn, axis=1)[:, :D]
            row_s = (np.max(np.abs(ch), axis=1, keepdims=True).clip(min=1e-6) / 127.0).astype(np.float16)
            w_i8[v0:v1] = np.clip(np.round(ch / row_s.astype(np.float32)), -127.0, 127.0).astype(np.int8)
            s_f16[v0:v1] = row_s
        return w_i8, s_f16

    def _non_expert_backbone_keys(self) -> List[str]:
        moe_exact = {
            f"{self.lp_root + '.' if self.lp_root else ''}layers.{li}.{ca}.experts" for li, ca in self.moe_specs
        }
        moe_dot = tuple(p + "." for p in moe_exact)
        skip_exact = {
            "lm_head.weight",
            "language_model.lm_head.weight",
            "head.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "embed.weight",
            self._embedding_key(),
        }
        return [
            k
            for k in sorted(self.active_keys)
            if k not in skip_exact
            and not is_companion_scale_key(k, self.active_keys)
            and k not in moe_exact
            and not k.startswith(moe_dot)
        ]

    def fixed_backbone_bytes(self, compute_elem_bytes: int = 2) -> int:
        """Estimates device-resident bytes for non-expert backbone + INT8 `lm_head` (`embed_tokens` stays mmap)."""
        total = self.vocab_size * self.hidden_size + self.vocab_size * 2
        for k in self._non_expert_backbone_keys():
            sp = self.spans[k]
            eb = 4 if sp.dtype == "F32" else compute_elem_bytes
            total += int(math.prod(sp.shape)) * eb
        return total

    def build_backbone_state_dict(self, compute_dtype: "torch.dtype") -> Dict[str, "torch.Tensor"]:
        """Loads all non-expert backbone parameters into CPU tensors ready for `meta`-skeleton injection."""
        return {
            canonicalize_hf_key(k, self.lp_root): self.read_torch_dequant(k, compute_dtype)
            for k in self._non_expert_backbone_keys()
        }

    def close(self) -> None:
        self._io_pool.shutdown(wait=True)
        if self._emb_mmap is not None:
            self._emb_mmap.close()
            self._emb_mmap = None
        if self._emb_file is not None:
            self._emb_file.close()  # type: ignore[attr-defined]
            self._emb_file = None
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
