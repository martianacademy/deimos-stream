#!/usr/bin/env python3
"""
DeimosStream Hardware Backend Resolver & Engine Factory (`src/deimos_stream/factory.py`)
========================================================================================
Routing Priority (`backend="auto"`):
  1. Apple Silicon (`darwin/arm64` with MLX): **`mlx`** (`DeimosPagedMoEEngine` — zero-copy
     Unified Memory + native 4-bit Metal `mx.gather_qmm`). Never falls back to `mps`.
  2. Linux / Windows with NVIDIA/ROCm GPU (`torch.cuda.is_available()`): **`cuda`**
     (`TorchDeimosPagedMoEEngine` — `pin_memory=True` async PCIe DMA + FlashAttention SDPA).
  3. Portable PyTorch fallback: **`cpu`** (`TorchDeimosPagedMoEEngine(device_str="cpu")`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine
from deimos_stream.torch_arena import HAS_TORCH
from deimos_stream.torch_engine import TorchDeimosPagedMoEEngine

if HAS_TORCH:
    import torch
else:
    torch = None  # type: ignore[assignment]

AnyDeimosEngine = Union[
    DeimosPagedMoEEngine, TorchDeimosPagedMoEEngine, DeimosHFHybridMoEEngine
]


def resolve_backend(backend: str = "auto") -> Optional[str]:
    """Resolves `auto | mlx | cuda | cpu` to the active available execution backend."""
    b = backend.lower().strip()
    if b == "auto":
        if HAS_MLX:
            return "mlx"
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        if HAS_TORCH:
            return "cpu"
        return None
    if b == "mlx":
        if not HAS_MLX:
            raise RuntimeError("MLX backend requires Apple Silicon (`pip install mlx mlx-lm`).")
        return "mlx"
    if b == "cuda":
        if not HAS_TORCH or not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requires PyTorch with an active CUDA/ROCm GPU.")
        return "cuda"
    if b == "cpu":
        if not HAS_TORCH:
            raise RuntimeError("CPU backend requires PyTorch (`pip install 'deimos-stream[cuda]'`).")
        return "cpu"
    raise ValueError(f"Unsupported backend={backend!r}; choose from 'auto', 'mlx', 'cuda', 'cpu'.")


def create_paged_moe_engine(
    store_dir: Path,
    backend: str = "auto",
    slots_per_layer: int = 16,
    verify_merkle_on_cold: bool = True,
    use_nocache_pread: bool = False,
    enable_lookahead: bool = True,
    enable_merkle_kv: bool = True,
    miss_tail_tau: float = 1.0,
    min_k: int = 4,
    top_k: Optional[int] = None,
) -> AnyDeimosEngine:
    """Instantiates `DeimosPagedMoEEngine` (`mlx`), `TorchDeimosPagedMoEEngine`, or `DeimosHFHybridMoEEngine`."""
    sdir = Path(store_dir).expanduser().resolve()
    mf_path = sdir / "merkle_manifest.json"
    if mf_path.exists():
        mf = json.loads(mf_path.read_text())
        if mf.get("store_kind") == "hf_hybrid_q4" or (sdir / "hf_backbone.safetensors").exists():
            dev_str = "cuda" if (HAS_TORCH and torch.cuda.is_available() and backend != "cpu") else "cpu"
            return DeimosHFHybridMoEEngine(
                store_dir=sdir,
                device_str=dev_str,
                slots_per_layer=slots_per_layer,
                verify_merkle_on_cold=verify_merkle_on_cold,
                use_nocache_pread=use_nocache_pread,
                enable_lookahead=enable_lookahead,
                enable_merkle_kv=enable_merkle_kv,
                miss_tail_tau=miss_tail_tau,
                min_k=min_k,
                top_k=top_k,
            )

    resolved = resolve_backend(backend)
    if resolved == "mlx":
        return DeimosPagedMoEEngine(
            store_dir=store_dir,
            slots_per_layer=slots_per_layer,
            verify_merkle_on_cold=verify_merkle_on_cold,
            use_nocache_pread=use_nocache_pread,
            enable_lookahead=enable_lookahead,
            enable_merkle_kv=enable_merkle_kv,
            miss_tail_tau=miss_tail_tau,
            min_k=min_k,
            top_k=top_k,
        )
    if resolved in {"cuda", "cpu"}:
        return TorchDeimosPagedMoEEngine(
            store_dir=store_dir,
            device_str=resolved,
            slots_per_layer=slots_per_layer,
            verify_merkle_on_cold=verify_merkle_on_cold,
            use_nocache_pread=use_nocache_pread,
            enable_lookahead=enable_lookahead,
            enable_merkle_kv=enable_merkle_kv,
            miss_tail_tau=miss_tail_tau,
            min_k=min_k,
            top_k=top_k,
        )
    raise RuntimeError("Neither MLX (macOS arm64) nor PyTorch (`deimos-stream[cuda]`) is installed.")
