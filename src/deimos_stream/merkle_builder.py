#!/usr/bin/env python3
"""
DeimosStream (`DEIMIDX2`) Universal Dynamic Merkle Builder & Q4 Layout Engine
(`src/deimos_stream/merkle_builder.py`)
=============================================================================
Responsibilities:
  1. Compiles `deimos_merkle.mm` + `deimos_arena.mm` into `libdeimos_stream.dylib`.
  2. Computes exact 16 KB page-aligned 4-bit affine SwiGLU expert leaf byte layouts.
  3. Builds self-describing `DEIMIDX2` binary Merkle index files (`merkle_tree.idx`)
     supporting arbitrary `(num_layers, num_experts, vocab_size)` topologies.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

PAGE_SIZE: int = 16384
DEIMOS_HEADER_STRUCT = struct.Struct("<8s10I8Q32s32s32s32s32s32sI12s")
DEIMOS_EXPERT_STRUCT = struct.Struct("<32sQIIQQ")
assert DEIMOS_HEADER_STRUCT.size == 320
assert DEIMOS_EXPERT_STRUCT.size == 64

PKG_DIR: Path = Path(__file__).resolve().parent
NATIVE_DIR: Path = PKG_DIR / "native"


def _resolve_include_dir() -> Path:
    """Locates `deimos_stream.h` in either installed wheel (`deimos_stream/include`) or repo root."""
    pkg_inc = PKG_DIR / "include"
    if (pkg_inc / "deimos_stream.h").exists():
        return pkg_inc
    repo_inc = PKG_DIR.parent.parent / "include"
    if (repo_inc / "deimos_stream.h").exists():
        return repo_inc
    raise FileNotFoundError("Could not locate include/deimos_stream.h")


def _resolve_dylib_path() -> Path:
    """Chooses `native/libdeimos_stream.{dylib,so}` when writable, else `~/.cache/deimos-stream/`."""
    ext = "dylib" if sys.platform == "darwin" else "so"
    fname = f"libdeimos_stream.{ext}"
    if os.access(NATIVE_DIR, os.W_OK):
        return NATIVE_DIR / fname
    cache_dir = Path.home() / ".cache" / "deimos-stream"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / fname


def align_up(value: int, alignment: int = PAGE_SIZE) -> int:
    """Rounds `value` up to the nearest multiple of `alignment`."""
    return (value + alignment - 1) & ~(alignment - 1)


def next_power_of_two(n: int) -> int:
    """Returns the smallest power of two >= max(1, n)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def sha256_pair_deimos(left32: bytes, right32: bytes) -> bytes:
    """Computes internal binary Merkle parent hash `SHA256(0x01 || left32 || right32)`."""
    return hashlib.sha256(b"\x01" + left32 + right32).digest()


def compile_deimos_dylib(force: bool = False) -> Path:
    """Compiles `deimos_merkle.mm` + `deimos_arena.mm` into `libdeimos_stream.dylib`."""
    src_merkle = NATIVE_DIR / "deimos_merkle.mm"
    src_arena = NATIVE_DIR / "deimos_arena.mm"
    inc_dir = _resolve_include_dir()
    hdr_path = inc_dir / "deimos_stream.h"
    sha_hdr = inc_dir / "deimos_sha256.h"
    dylib_path = _resolve_dylib_path()

    newest_src = max(
        src_merkle.stat().st_mtime,
        src_arena.stat().st_mtime,
        hdr_path.stat().st_mtime,
        sha_hdr.stat().st_mtime if sha_hdr.exists() else 0.0,
    )
    if not force and dylib_path.exists() and dylib_path.stat().st_mtime >= newest_src:
        return dylib_path

    if sys.platform == "darwin":
        cmd = [
            "clang++",
            "-O3",
            "-std=c++17",
            "-dynamiclib",
            "-fPIC",
            "-I",
            str(inc_dir),
            "-x",
            "objective-c++",
            "-framework",
            "Foundation",
            str(src_merkle),
            str(src_arena),
            "-o",
            str(dylib_path),
        ]
    else:
        cxx = shutil.which("clang++") or shutil.which("c++") or "g++"
        cmd = [
            cxx,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-I",
            str(inc_dir),
            "-x",
            "c++",
            str(src_merkle),
            str(src_arena),
            "-o",
            str(dylib_path),
        ]
    subprocess.run(cmd, check=True)
    return dylib_path


def compute_q4_leaf_layout(
    hidden_size: int, intermediate_size: int, group_size: int = 64
) -> Dict[str, int]:
    """
    Computes exact byte lengths for a 4-bit affine quantized SwiGLU expert leaf:
      `[gu_w || gu_s || gu_b || dn_w || dn_s || dn_b]` padded to `PAGE_SIZE = 16,384`.
    """
    if hidden_size % group_size != 0 or intermediate_size % group_size != 0:
        raise ValueError(
            f"hidden_size={hidden_size} and intermediate_size={intermediate_size} "
            f"must be divisible by group_size={group_size}"
        )
    gu_out = 2 * intermediate_size
    gu_w_bytes = gu_out * (hidden_size // 8) * 4
    gu_s_bytes = gu_out * (hidden_size // group_size) * 2
    dn_w_bytes = hidden_size * (intermediate_size // 8) * 4
    dn_s_bytes = hidden_size * (intermediate_size // group_size) * 2

    payload_bytes = gu_w_bytes + 2 * gu_s_bytes + dn_w_bytes + 2 * dn_s_bytes
    stride_bytes = align_up(payload_bytes, PAGE_SIZE)
    return {
        "gu_w_bytes": gu_w_bytes,
        "gu_s_bytes": gu_s_bytes,
        "dn_w_bytes": dn_w_bytes,
        "dn_s_bytes": dn_s_bytes,
        "payload_bytes": payload_bytes,
        "stride_bytes": stride_bytes,
    }


def build_deimidx2_file(
    idx_path: Path,
    num_layers: int,
    num_experts: int,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    quant_bits: int,
    quant_group_size: int,
    expert_stride_bytes: int,
    expert_payload_bytes: int,
    leaf_hashes: List[bytes],
    leaf_offsets: Optional[List[int]] = None,
    embedding_root: Optional[bytes] = None,
    shared_core_root: Optional[bytes] = None,
    output_head_root: Optional[bytes] = None,
    orig_safetensors_bytes: int = 0,
    kv_table_slots: int = 1024,
) -> Tuple[str, str]:
    """
    Writes a universal self-describing `DEIMIDX2` binary Merkle index for arbitrary `(L, E, V)`.
    Returns `(model_merkle_root_hex, expert_tree_root_hex)`.
    """
    total_experts = num_layers * num_experts
    if len(leaf_hashes) != total_experts:
        raise ValueError(f"Expected {total_experts} leaf hashes, got {len(leaf_hashes)}")

    tree_cap = next_power_of_two(total_experts)
    pad_leaf_hash = hashlib.sha256(b"\x00").digest()

    xor_tree: List[bytes] = [b"\x00" * 32] * (2 * tree_cap)
    for slot in range(total_experts):
        xor_tree[tree_cap + slot] = leaf_hashes[slot]
    for slot in range(total_experts, tree_cap):
        xor_tree[tree_cap + slot] = pad_leaf_hash
    for node_i in range(tree_cap - 1, 0, -1):
        xor_tree[node_i] = sha256_pair_deimos(xor_tree[2 * node_i], xor_tree[2 * node_i + 1])

    expert_tree_root = xor_tree[1]
    emb_root = embedding_root if embedding_root is not None else hashlib.sha256(b"emb").digest()
    core_root = shared_core_root if shared_core_root is not None else hashlib.sha256(b"core").digest()
    head_root = output_head_root if output_head_root is not None else hashlib.sha256(b"head").digest()

    left_b = sha256_pair_deimos(emb_root, core_root)
    right_b = sha256_pair_deimos(expert_tree_root, head_root)
    model_root = sha256_pair_deimos(left_b, right_b)

    offset_xor = DEIMOS_HEADER_STRUCT.size
    offset_exp = offset_xor + (2 * tree_cap * 32)
    offset_tok = 0
    offset_kv = offset_exp + (total_experts * DEIMOS_EXPERT_STRUCT.size)
    total_idx_bytes = offset_kv + (kv_table_slots * 80)

    exp_buf = bytearray(total_experts * DEIMOS_EXPERT_STRUCT.size)
    for slot in range(total_experts):
        b_off = leaf_offsets[slot] if leaf_offsets is not None else (slot * expert_stride_bytes)
        DEIMOS_EXPERT_STRUCT.pack_into(
            exp_buf,
            slot * DEIMOS_EXPERT_STRUCT.size,
            leaf_hashes[slot],
            b_off,
            expert_stride_bytes,
            0,
            0,
            0,
        )

    hdr_bytes = DEIMOS_HEADER_STRUCT.pack(
        b"DEIMIDX2",
        2,
        num_layers,
        num_experts,
        tree_cap,
        vocab_size,
        hidden_size,
        intermediate_size,
        quant_bits,
        quant_group_size,
        0,
        expert_stride_bytes,
        expert_payload_bytes,
        offset_xor,
        offset_exp,
        offset_tok,
        offset_kv,
        total_idx_bytes,
        orig_safetensors_bytes,
        model_root,
        model_root,
        emb_root,
        core_root,
        expert_tree_root,
        head_root,
        kv_table_slots,
        b"\x00" * 12,
    )

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "wb") as f:
        f.write(hdr_bytes)
        for node_h in xor_tree:
            f.write(node_h)
        f.write(exp_buf)
        f.write(b"\x00" * (kv_table_slots * 80))

    return model_root.hex(), expert_tree_root.hex()


def _ensure_symlink(src: Path, dst: Path) -> None:
    src_abs = src.resolve()
    if dst.is_symlink():
        if dst.resolve() == src_abs:
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(str(src_abs), str(dst))


def convert_fp16_cas_store_to_q4(
    fp16_store_dir: Path,
    q4_store_dir: Optional[Path] = None,
    group_size: int = 64,
    force_rebuild: bool = False,
) -> Path:
    """Converts an FP16 `cas_experts.bin` Merkle store into 16 KB page-aligned Q4 `DEIMIDX2`."""
    fp16_dir = Path(fp16_store_dir).resolve()
    out_dir = Path(q4_store_dir).resolve() if q4_store_dir is not None else (fp16_dir / "v3_q4")
    out_dir.mkdir(parents=True, exist_ok=True)
    compile_deimos_dylib(force=False)

    cfg = json.loads((fp16_dir / "config.json").read_text())
    src_manifest = json.loads((fp16_dir / "merkle_manifest.json").read_text())

    num_layers = int(cfg["num_hidden_layers"])
    num_experts = int(cfg["num_experts"])
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    vocab_size = int(cfg["vocab_size"])

    layout = compute_q4_leaf_layout(hidden_size, intermediate_size, group_size=group_size)
    stride_bytes = layout["stride_bytes"]
    payload_bytes = layout["payload_bytes"]
    expected_cas_bytes = num_layers * num_experts * stride_bytes

    cas_q4_path = out_dir / "cas_experts_q4.bin"
    idx_path = out_dir / "merkle_tree.idx"
    manifest_path = out_dir / "merkle_manifest.json"

    for shared_fname in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "shared_core.bin",
        "embed_tokens.bin",
        "lm_head_int8.bin",
    ):
        src_f = fp16_dir / shared_fname
        if src_f.exists():
            _ensure_symlink(src_f, out_dir / shared_fname)

    if (
        not force_rebuild
        and cas_q4_path.exists()
        and cas_q4_path.stat().st_size == expected_cas_bytes
        and idx_path.exists()
        and manifest_path.exists()
    ):
        return out_dir

    t0 = time.perf_counter()
    fp16_leaf_bytes = 3 * hidden_size * intermediate_size * 2
    gu_elems = 2 * intermediate_size * hidden_size
    d_elems = hidden_size * intermediate_size
    gu_fp16_bytes = gu_elems * 2
    pad_tail = b"\x00" * (stride_bytes - payload_bytes)

    leaf_hashes: List[bytes] = []
    tmp_cas_path = out_dir / "cas_experts_q4.bin.tmp"

    with open(fp16_dir / "cas_experts.bin", "rb") as src_f, open(tmp_cas_path, "wb") as dst_f:
        src_mm = mmap.mmap(src_f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for l_idx in range(num_layers):
                gu_batch_np = np.empty(
                    (num_experts, 2 * intermediate_size, hidden_size), dtype=np.float16
                )
                dn_batch_np = np.empty(
                    (num_experts, hidden_size, intermediate_size), dtype=np.float16
                )
                for e_idx in range(num_experts):
                    slot = l_idx * num_experts + e_idx
                    off = slot * fp16_leaf_bytes
                    gu_batch_np[e_idx] = np.frombuffer(
                        src_mm, dtype=np.float16, count=gu_elems, offset=off
                    ).reshape(2 * intermediate_size, hidden_size)
                    dn_batch_np[e_idx] = np.frombuffer(
                        src_mm, dtype=np.float16, count=d_elems, offset=off + gu_fp16_bytes
                    ).reshape(hidden_size, intermediate_size)

                gu_qw, gu_qs, gu_qb = mx.quantize(
                    mx.array(gu_batch_np), group_size=group_size, bits=4
                )
                dn_qw, dn_qs, dn_qb = mx.quantize(
                    mx.array(dn_batch_np), group_size=group_size, bits=4
                )
                mx.eval(gu_qw, gu_qs, gu_qb, dn_qw, dn_qs, dn_qb)

                gu_qw_np = np.asarray(gu_qw)
                gu_qs_np = np.asarray(gu_qs)
                gu_qb_np = np.asarray(gu_qb)
                dn_qw_np = np.asarray(dn_qw)
                dn_qs_np = np.asarray(dn_qs)
                dn_qb_np = np.asarray(dn_qb)

                for e_idx in range(num_experts):
                    leaf_raw = (
                        gu_qw_np[e_idx].tobytes()
                        + gu_qs_np[e_idx].tobytes()
                        + gu_qb_np[e_idx].tobytes()
                        + dn_qw_np[e_idx].tobytes()
                        + dn_qs_np[e_idx].tobytes()
                        + dn_qb_np[e_idx].tobytes()
                        + pad_tail
                    )
                    leaf_hashes.append(hashlib.sha256(leaf_raw).digest())
                    dst_f.write(leaf_raw)

                mx.clear_cache()
                sys.stdout.write(
                    f"[DEIMIDX2-Q4] Quantized & SHA-256 Hashed Layer {l_idx + 1}/{num_layers} "
                    f"({(l_idx + 1) * num_experts}/{num_layers * num_experts} leaves)\n"
                )
                sys.stdout.flush()
        finally:
            src_mm.close()

    tmp_cas_path.replace(cas_q4_path)

    model_root_hex, expert_root_hex = build_deimidx2_file(
        idx_path=idx_path,
        num_layers=num_layers,
        num_experts=num_experts,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_bits=4,
        quant_group_size=group_size,
        expert_stride_bytes=stride_bytes,
        expert_payload_bytes=payload_bytes,
        leaf_hashes=leaf_hashes,
        embedding_root=bytes.fromhex(str(src_manifest["embedding_trie_root"])),
        shared_core_root=bytes.fromhex(str(src_manifest["shared_core_root"])),
        output_head_root=bytes.fromhex(str(src_manifest["output_head_root"])),
        orig_safetensors_bytes=int(src_manifest.get("expert_cas_total_bytes", 12884901888)),
    )

    manifest_out = {
        "format": "DEIMIDX2",
        "version": 2,
        "quant_bits": 4,
        "quant_group_size": group_size,
        "num_layers": num_layers,
        "num_experts_per_layer": num_experts,
        "total_experts": num_layers * num_experts,
        "expert_stride_bytes": stride_bytes,
        "expert_payload_bytes": payload_bytes,
        "cas_q4_total_bytes": expected_cas_bytes,
        "model_merkle_root": model_root_hex,
        "expert_tree_root": expert_root_hex,
        "conversion_time_sec": round(time.perf_counter() - t0, 2),
    }
    manifest_path.write_text(json.dumps(manifest_out, indent=2))
    return out_dir
