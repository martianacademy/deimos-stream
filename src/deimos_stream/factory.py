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
import sys
from pathlib import Path
from typing import Optional, Union

from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from deimos_stream.fast_decode import apply_q4_backbone, apply_q4_backbone_mlx_hybrid
from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine
from deimos_stream.memory_budget import SlotPlan, SlotsArg, parse_slots_arg, plan_slots
from deimos_stream.mlx_hybrid import DeimosMLXHybridMoEEngine
from deimos_stream.mlx_universal import DeimosMLXUniversalMoEEngine
from deimos_stream.torch_arena import HAS_TORCH
from deimos_stream.torch_engine import TorchDeimosPagedMoEEngine

if HAS_TORCH:
    import torch
else:
    torch = None  # type: ignore[assignment]

AnyDeimosEngine = Union[
    DeimosPagedMoEEngine,
    DeimosMLXHybridMoEEngine,
    DeimosMLXUniversalMoEEngine,
    TorchDeimosPagedMoEEngine,
    DeimosHFHybridMoEEngine,
]


def resolve_backend(backend: str = "auto") -> Optional[str]:
    """Resolves `auto | mlx | cuda | cpu` to the active available execution backend."""
    b = backend.lower().strip()
    if b == "mlx_hybrid":
        b = "mlx"
    elif b.startswith("hf_"):
        b = b[3:]
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
    slots_per_layer: SlotsArg = "auto",
    verify_merkle_on_cold: bool = True,
    use_nocache_pread: bool = False,
    enable_lookahead: bool = True,
    enable_merkle_kv: bool = True,
    miss_tail_tau: float = 1.0,
    min_k: int = 4,
    top_k: Optional[int] = None,
    max_memory_gb: Optional[float] = None,
    backbone: str = "fp16",
) -> AnyDeimosEngine:
    """Instantiates `DeimosPagedMoEEngine` (`mlx`), `TorchDeimosPagedMoEEngine`, or `DeimosHFHybridMoEEngine`.

    `slots_per_layer="auto"` (default) sizes the expert arena from free memory: all experts are
    pinned when they fit, otherwise as many slots as fit (see `memory_budget.plan_slots`).
    `max_memory_gb` optionally caps the total memory budget.
    `backbone="q4"` quantizes attention + lm_head to 4-bit in memory (MLX dense engine only).
    """
    if backbone not in {"fp16", "q4"}:
        raise ValueError(f"Unsupported backbone={backbone!r}; choose 'fp16' or 'q4'.")
    sdir = Path(store_dir).expanduser().resolve()
    resolved = resolve_backend(backend)
    if resolved is None:
        raise RuntimeError("Neither MLX (macOS arm64) nor PyTorch (`deimos-stream[cuda]`) is installed.")
    plan: Optional[SlotPlan] = None
    slots_arg = parse_slots_arg(slots_per_layer)
    if slots_arg == "auto":
        plan = plan_slots(sdir, resolved, max_memory_gb=max_memory_gb, top_k=top_k, backbone=backbone)
        sys.stderr.write(plan.summary() + "\n")
        slots = plan.slots_per_layer
    else:
        slots = int(slots_arg)
    common = dict(
        store_dir=sdir,
        slots_per_layer=slots,
        verify_merkle_on_cold=verify_merkle_on_cold,
        use_nocache_pread=use_nocache_pread,
        enable_lookahead=enable_lookahead,
        enable_merkle_kv=enable_merkle_kv,
        miss_tail_tau=miss_tail_tau,
        min_k=min_k,
        top_k=top_k,
    )

    mf_path = sdir / "merkle_manifest.json"
    is_hybrid = (sdir / "hf_backbone.safetensors").exists()
    if mf_path.exists() and not is_hybrid:
        is_hybrid = json.loads(mf_path.read_text()).get("store_kind") == "hf_hybrid_q4"

    eng: AnyDeimosEngine
    if is_hybrid:
        if resolved == "mlx":
            mtype = ""
            cfg_p = sdir / "config.json"
            if cfg_p.exists():
                mtype = str(json.loads(cfg_p.read_text()).get("model_type", "")).lower()
            if mtype in {"qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_text"}:
                eng = DeimosMLXHybridMoEEngine(backbone=backbone, **common)  # type: ignore[arg-type]
                if backbone == "q4":
                    apply_q4_backbone_mlx_hybrid(eng)
            else:
                eng = DeimosMLXUniversalMoEEngine(backbone=backbone, **common)  # type: ignore[arg-type]
        else:
            eng = DeimosHFHybridMoEEngine(device_str=resolved, backbone=backbone, **common)  # type: ignore[arg-type]
    elif resolved == "mlx":
        eng = DeimosPagedMoEEngine(**common)  # type: ignore[arg-type]
        if backbone == "q4":
            apply_q4_backbone(eng)
    else:
        eng = TorchDeimosPagedMoEEngine(device_str=resolved, backbone=backbone, **common)  # type: ignore[arg-type]
    eng.slot_plan = plan  # type: ignore[union-attr]
    return eng
