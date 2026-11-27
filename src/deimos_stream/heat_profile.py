#!/usr/bin/env python3
"""
DeimosStream Persistent Routing-Heat Profile (`src/deimos_stream/heat_profile.py`)
==================================================================================
Zero-quality-loss (`tau=1.0`) persistent expert frequency profile (`.deimos_heat`):
  - Records cumulative routed expert activations `[num_layers, num_experts]` across runs
    using an atomic exponential-moving-average (`0.75 * prior + session`) binary file.
  - On engine startup, pre-populates physical GPU/UMA arena slots (`0 .. slots_per_layer-1`)
    with each layer's historically hottest experts instead of cold sequential `0..S-1` indices.
  - Strictly preserves exact model weights and router decisions (`0.0` logit / token drift).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import struct
from typing import Any, Optional, Tuple

import numpy as np

HEAT_FILENAME: str = ".deimos_heat"
HEAT_MAGIC: bytes = b"DEIMHEAT"
HEAT_VERSION: int = 1
HEAT_HEADER_STRUCT = struct.Struct("<8sIIIIQ")  # 32 bytes: magic, ver, L, E, runs, total_hits
assert HEAT_HEADER_STRUCT.size == 32


def bind_heat_c_api(lib: ctypes.CDLL) -> None:
    """Idempotently binds C++17 `.deimos_heat` export & warm-slot seeding entrypoints."""
    if getattr(lib, "_deimos_heat_bound", False):
        return
    u32_p = ctypes.POINTER(ctypes.c_uint32)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    lib.deimos_arena_export_expert_hits.argtypes = [ctypes.c_void_p, u32_p, ctypes.c_int]
    lib.deimos_arena_export_expert_hits.restype = ctypes.c_int
    lib.deimos_arena_seed_warm_slots.argtypes = [
        ctypes.c_void_p,
        i32_p,
        i32_p,
        u32_p,
        ctypes.c_int,
    ]
    lib.deimos_arena_seed_warm_slots.restype = ctypes.c_int
    lib._deimos_heat_bound = True  # type: ignore[attr-defined]


def heat_file_path(store_dir: Path) -> Path:
    """Returns canonical `.deimos_heat` path inside a `DEIMIDX2` store directory."""
    return Path(store_dir) / HEAT_FILENAME


def load_heat_counts(
    store_dir: Path, num_layers: int, num_experts: int
) -> Optional[Tuple[np.ndarray, int, int]]:
    """Loads `(counts_u32 [L, E], total_runs, total_hits)` if valid `.deimos_heat` exists."""
    p = heat_file_path(store_dir)
    if not p.is_file():
        return None
    expected_bytes = HEAT_HEADER_STRUCT.size + num_layers * num_experts * 4
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if len(raw) != expected_bytes:
        return None
    magic, ver, l_cnt, e_cnt, n_runs, tot_hits = HEAT_HEADER_STRUCT.unpack_from(raw, 0)
    if magic != HEAT_MAGIC or ver != HEAT_VERSION or l_cnt != num_layers or e_cnt != num_experts:
        return None
    arr = np.frombuffer(raw, dtype="<u4", offset=HEAT_HEADER_STRUCT.size).reshape(
        num_layers, num_experts
    ).copy()
    return arr, int(n_runs), int(tot_hits)


def compute_warm_slot_plan(
    counts_u32: np.ndarray, slots_per_layer: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes `(warm_ids [L, S], init_slot_freqs [L, S], init_expert_freqs [L, E])`.

    Ranks experts per layer by descending historical frequency (ties broken deterministically by
    smaller expert index) and scales initial TinyLFU frequencies into `[1 .. 32]` so warm slots
    resist single-token cold noise while remaining adaptable to topic shifts.
    """
    num_layers, num_experts = counts_u32.shape
    s_cap = max(1, min(num_experts, int(slots_per_layer)))
    warm_ids = np.empty((num_layers, s_cap), dtype=np.int32)
    slot_freqs = np.zeros((num_layers, s_cap), dtype=np.int32)
    exp_freqs = np.zeros((num_layers, num_experts), dtype=np.uint32)

    idx_grid = np.arange(num_experts, dtype=np.int64)
    for l_idx in range(num_layers):
        row = counts_u32[l_idx].astype(np.int64)
        # Stable descending sort on count, ascending on expert index:
        order = np.lexsort((idx_grid, -row))
        chosen = order[:s_cap].astype(np.int32)
        warm_ids[l_idx] = chosen
        max_c = int(row[chosen[0]]) if s_cap > 0 else 0
        if max_c > 0:
            scaled_all = np.ceil((row.astype(np.float64) / float(max_c)) * 24.0).astype(np.int32)
            scaled_all = np.where(row > 0, np.clip(scaled_all, 1, 24), 0)
            exp_freqs[l_idx] = scaled_all.astype(np.uint32)
            slot_freqs[l_idx] = scaled_all[chosen]
    return warm_ids, slot_freqs, exp_freqs


def seed_arena_from_heat(arena: Any, stage_bound_pools: bool = True) -> Optional[np.ndarray]:
    """Seeds `arena` from `<store_dir>/.deimos_heat` when present and `not is_fully_pinned`.

    Returns `warm_ids [L, S]` (`int32`) if a valid non-empty heat profile was applied, else `None`.
    Respects `DEIMOS_DISABLE_HEAT=1` environment override.
    """
    if os.environ.get("DEIMOS_DISABLE_HEAT", "0") == "1":
        return None
    if getattr(arena, "is_fully_pinned", False) or not getattr(arena, "arena_handle", None):
        return None
    store_dir = getattr(arena.bridge, "store_dir", None)
    if store_dir is None:
        return None
    loaded = load_heat_counts(Path(store_dir), arena.num_layers, arena.num_experts)
    if loaded is None:
        return None
    counts_u32, _, _ = loaded
    if int(counts_u32.sum()) <= 0:
        return None

    bind_heat_c_api(arena.bridge.lib)
    warm_ids, slot_freqs, exp_freqs = compute_warm_slot_plan(counts_u32, arena.slots_per_layer)
    rc = arena.bridge.lib.deimos_arena_seed_warm_slots(
        arena.arena_handle,
        warm_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        slot_freqs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        exp_freqs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        1 if stage_bound_pools else 0,
    )
    if rc == -2:
        from deimos_stream.arena import DeimosMerkleIntegrityError

        raise DeimosMerkleIntegrityError("Corrupted Q4 Merkle leaf rejected during warm heat preload!")
    if rc != 0:
        return None
    return warm_ids


def save_arena_heat(arena: Any) -> bool:
    """Exports cumulative session hits from `arena` and atomically updates `.deimos_heat`."""
    if os.environ.get("DEIMOS_DISABLE_HEAT", "0") == "1":
        return False
    handle = getattr(arena, "arena_handle", None)
    store_dir = getattr(getattr(arena, "bridge", None), "store_dir", None)
    if not handle or store_dir is None:
        return False

    bind_heat_c_api(arena.bridge.lib)
    total = int(arena.num_layers) * int(arena.num_experts)
    session_hits = np.zeros((total,), dtype=np.uint32)
    rc = arena.bridge.lib.deimos_arena_export_expert_hits(
        handle,
        session_hits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        total,
    )
    if rc != 0:
        return False
    sess_sum = int(session_hits.sum())
    if sess_sum <= 0:
        return False

    sess_2d = session_hits.reshape(arena.num_layers, arena.num_experts).astype(np.uint64)
    prior = load_heat_counts(Path(store_dir), arena.num_layers, arena.num_experts)
    if prior is not None:
        prev_arr, prev_runs, prev_tot = prior
        merged = np.minimum((prev_arr.astype(np.uint64) * 3) // 4 + sess_2d, 0x3FFFFFFF)
        runs = min(prev_runs + 1, 0xFFFFFFFF)
        tot_hits = prev_tot + sess_sum
    else:
        merged = np.minimum(sess_2d, 0x3FFFFFFF)
        runs = 1
        tot_hits = sess_sum

    out_path = heat_file_path(Path(store_dir))
    tmp_path = out_path.with_suffix(f".heat_tmp_{os.getpid()}")
    hdr = HEAT_HEADER_STRUCT.pack(
        HEAT_MAGIC, HEAT_VERSION, int(arena.num_layers), int(arena.num_experts), runs, tot_hits
    )
    payload = hdr + np.ascontiguousarray(merged, dtype="<u4").tobytes()
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, out_path)
        return True
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
