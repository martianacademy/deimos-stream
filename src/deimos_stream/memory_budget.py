#!/usr/bin/env python3
"""
DeimosStream Memory-Aware Slot Planner (`src/deimos_stream/memory_budget.py`)
============================================================================
Chooses `slots_per_layer` from the memory that is actually free (`--slots auto`):

  * If the backbone, the lm_head, every expert and a working reserve all fit, every expert
    is pinned (`slots = num_experts`). This is the fastest path, because it skips paging entirely.
  * Otherwise it uses as many slots as fit:
        slots = floor((usable - fixed) / (num_layers * expert_stride_bytes))
    clamped to `[top_k, num_experts]`.

    usable = min(available * memory_fraction, max_memory) - reserve

Where `available` comes from:
  * `mlx` (Apple unified memory): free + inactive + speculative + purgeable pages (`vm_stat`),
    capped by Metal's `max_recommended_working_set_size` minus MLX's active memory
  * `cuda`: free device memory (`torch.cuda.mem_get_info`)
  * `cpu`: `/proc/meminfo` `MemAvailable` (Linux) or `vm_stat` (macOS)

This module imports nothing heavy at load time, so CI can test it without MLX or a GPU.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

GIB = 1024**3
DEFAULT_FRACTION: Dict[str, float] = {"mlx": 0.75, "cuda": 0.90, "cpu": 0.75}
DEFAULT_RESERVE_BYTES = 1 * GIB  # KV cache, activations and prefill transients

SlotsArg = Union[int, str]


@dataclass
class SlotPlan:
    slots_per_layer: int
    num_experts: int
    fully_pinned: bool
    forced_minimum: bool
    available_bytes: int
    usable_bytes: int
    fixed_bytes: int
    per_slot_bytes: int
    full_expert_bytes: int
    memory_source: str

    def summary(self) -> str:
        mode = "all experts pinned" if self.fully_pinned else "paged"
        msg = (
            f"[DeimosStream] auto slots: {self.slots_per_layer}/{self.num_experts} per layer ({mode}) | "
            f"free {self.available_bytes / GIB:.2f} GiB ({self.memory_source}) -> usable "
            f"{self.usable_bytes / GIB:.2f} GiB | fixed {self.fixed_bytes / GIB:.2f} GiB | "
            f"experts {self.slots_per_layer * self.per_slot_bytes / GIB:.2f} of "
            f"{self.full_expert_bytes / GIB:.2f} GiB"
        )
        if self.forced_minimum:
            msg += " | WARNING: budget is below the top_k minimum, so top_k slots are used and may exceed it"
        return msg

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_slots_arg(value: SlotsArg) -> SlotsArg:
    """Accepts `auto` or a positive integer (CLI `--slots`, chat `/slots`, server config)."""
    if isinstance(value, int):
        if value < 1:
            raise ValueError("slots must be >= 1")
        return value
    v = str(value).strip().lower()
    if v == "auto":
        return "auto"
    n = int(v)
    if n < 1:
        raise ValueError("slots must be >= 1")
    return n


def store_layout(store_dir: Path) -> Tuple[int, int, int, int, int]:
    """Returns (num_layers, num_experts, expert_stride_bytes, fixed_device_bytes, top_k)."""
    sdir = Path(store_dir).expanduser().resolve()
    mf = json.loads((sdir / "merkle_manifest.json").read_text())
    num_layers = int(mf["num_layers"])
    num_experts = int(mf["num_experts_per_layer"])
    stride = int(mf.get("expert_stride_bytes") or mf["expert_payload_bytes"])
    # Device-resident non-expert weights. `embed_tokens.bin` stays a CPU mmap row lookup.
    fixed = 0
    for name in ("shared_core.bin", "hf_backbone.safetensors", "lm_head_int8.bin"):
        p = sdir / name
        if p.exists():
            fixed += p.stat().st_size
    top_k = 8
    cfg_path = sdir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        sub = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        top_k = int(cfg.get("num_experts_per_tok") or sub.get("num_experts_per_tok") or 8)
    return num_layers, num_experts, stride, fixed, top_k


def _macos_available_bytes() -> Optional[int]:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    total = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        km = re.search(rf"{key}:\s+(\d+)", out)
        if km:
            total += int(km.group(1)) * page
    return total or None


def _linux_available_bytes() -> Optional[int]:
    p = Path("/proc/meminfo")
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def available_memory_bytes(backend: str) -> Tuple[int, str]:
    """Free memory the expert arena may use on `backend` (`mlx`, `cuda`, `cpu`)."""
    if backend == "cuda":
        import torch

        free_b, _total = torch.cuda.mem_get_info()
        return int(free_b), "cuda free VRAM"
    host = _macos_available_bytes() if sys.platform == "darwin" else _linux_available_bytes()
    if backend == "mlx":
        import mlx.core as mx

        info_fn = getattr(mx, "device_info", None) or mx.metal.device_info
        info = info_fn()
        get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        metal_cap = int(info["max_recommended_working_set_size"]) - int(get_active())
        if host is None:
            return max(0, metal_cap // 2), "metal working set (vm_stat unavailable)"
        return max(0, min(host, metal_cap)), "unified memory"
    if host is None:
        return 8 * GIB, "assumed 8 GiB (host RAM unknown)"
    return host, "host RAM"


def plan_slots(
    store_dir: Path,
    backend: str,
    max_memory_gb: Optional[float] = None,
    memory_fraction: Optional[float] = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    available_bytes: Optional[int] = None,
    top_k: Optional[int] = None,
    backbone: str = "fp16",
) -> SlotPlan:
    """Plans `slots_per_layer` for `store_dir` on `backend` from the currently free memory."""
    num_layers, num_experts, stride, fixed, cfg_top_k = store_layout(store_dir)
    if backbone == "q4":  # FP16 shared_core and INT8 lm_head shrink to 4.5 bits per weight
        sdir = Path(store_dir).expanduser().resolve()
        core = (sdir / "shared_core.bin").stat().st_size if (sdir / "shared_core.bin").exists() else 0
        head = (sdir / "lm_head_int8.bin").stat().st_size if (sdir / "lm_head_int8.bin").exists() else 0
        fixed -= core + head - int(core * 4.5 / 16) - int(head * 4.5 / 8)
    k = int(top_k or cfg_top_k)
    if available_bytes is None:
        available_bytes, source = available_memory_bytes(backend)
    else:
        source = "given"
    frac = DEFAULT_FRACTION.get(backend, 0.75) if memory_fraction is None else float(memory_fraction)
    usable = int(available_bytes * frac)
    if max_memory_gb is not None:
        usable = min(usable, int(float(max_memory_gb) * GIB))
    usable = max(0, usable - int(reserve_bytes))

    per_slot = num_layers * stride
    full = num_experts * per_slot
    if fixed + full <= usable:
        slots = num_experts
    else:
        slots = int(math.floor(max(0, usable - fixed) / per_slot))
    forced = slots < k
    slots = max(min(k, num_experts), min(num_experts, slots))
    return SlotPlan(
        slots_per_layer=slots,
        num_experts=num_experts,
        fully_pinned=slots == num_experts,
        forced_minimum=forced,
        available_bytes=int(available_bytes),
        usable_bytes=usable,
        fixed_bytes=fixed,
        per_slot_bytes=per_slot,
        full_expert_bytes=full,
        memory_source=source,
    )
