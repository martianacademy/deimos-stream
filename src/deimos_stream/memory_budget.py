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
  * `cuda`: free device memory (`torch.cuda.mem_get_info`), and the slot count is also capped by free
    host RAM (container-aware), because the CUDA arena keeps a host copy of the slot pool
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
    host_available_bytes: Optional[int] = None
    host_limited: bool = False

    def summary(self) -> str:
        mode = "all experts pinned" if self.fully_pinned else "paged"
        msg = (
            f"[DeimosStream] auto slots: {self.slots_per_layer}/{self.num_experts} per layer ({mode}) | "
            f"free {self.available_bytes / GIB:.2f} GiB ({self.memory_source}) -> usable "
            f"{self.usable_bytes / GIB:.2f} GiB | fixed {self.fixed_bytes / GIB:.2f} GiB | "
            f"experts {self.slots_per_layer * self.per_slot_bytes / GIB:.2f} of "
            f"{self.full_expert_bytes / GIB:.2f} GiB"
        )
        if self.host_limited:
            msg += (f" | capped by host RAM ({(self.host_available_bytes or 0) / GIB:.2f} GiB free): "
                    "CUDA keeps a host copy of the slot pool")
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
    if not (sdir / "merkle_manifest.json").exists():
        from deimos_stream.fp8_safetensors import SafetensorsZeroCopyIndex, is_raw_fp8_moe_checkpoint

        if is_raw_fp8_moe_checkpoint(sdir):
            idx = SafetensorsZeroCopyIndex(sdir, use_nocache=False)
            try:
                return (
                    idx.num_moe_layers,
                    idx.num_experts,
                    idx.expert_bytes,
                    idx.fixed_backbone_bytes(compute_elem_bytes=2),
                    idx.top_k,
                )
            finally:
                idx.close()
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


def _cgroup_dirs() -> list[Path]:
    dirs = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    cg_proc = Path("/proc/self/cgroup")
    if cg_proc.exists():
        try:
            for line in cg_proc.read_text().splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3 and parts[2].startswith("/") and len(parts[2]) > 1:
                    rel = parts[2].lstrip("/")
                    dirs.append(Path("/sys/fs/cgroup") / rel)
                    dirs.append(Path("/sys/fs/cgroup/memory") / rel)
        except OSError:
            pass
    return dirs


def _cgroup_available_bytes() -> Optional[int]:
    """Free memory under a container limit (cgroup v2 or v1); None when there is no limit."""
    best: Optional[int] = None
    for d in _cgroup_dirs():
        try:
            v2 = d / "memory.max"
            if v2.exists() and v2.read_text().strip() != "max":
                used = int((d / "memory.current").read_text())
                stat = d / "memory.stat"
                m = re.search(r"^inactive_file (\d+)", stat.read_text(), re.M) if stat.exists() else None
                val = max(0, int(v2.read_text()) - used + (int(m.group(1)) if m else 0))
                best = val if best is None else min(best, val)
            v1 = d / "memory.limit_in_bytes"
            if v1.exists() and int(v1.read_text()) < (1 << 60):
                used = int((d / "memory.usage_in_bytes").read_text())
                stat = d / "memory.stat"
                m = re.search(r"^(?:total_inactive_file|cache) (\d+)", stat.read_text(), re.M) if stat.exists() else None
                val = max(0, int(v1.read_text()) - used + (int(m.group(1)) if m else 0))
                best = val if best is None else min(best, val)
        except (OSError, ValueError):
            continue
    return best


def _linux_total_ram_bytes() -> Optional[int]:
    """Total RAM available to this process, tightened by cgroup v2/v1 container limits."""
    tot = None
    p = Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("MemTotal:"):
                tot = int(line.split()[1]) * 1024
                break
    for d in _cgroup_dirs():
        try:
            v2 = d / "memory.max"
            if v2.exists() and v2.read_text().strip() != "max":
                lim = int(v2.read_text())
                tot = lim if tot is None else min(tot, lim)
            v1 = d / "memory.limit_in_bytes"
            if v1.exists() and int(v1.read_text()) < (1 << 60):
                lim = int(v1.read_text())
                tot = lim if tot is None else min(tot, lim)
        except (OSError, ValueError):
            continue
    return tot


def _linux_available_bytes() -> Optional[int]:
    """`MemAvailable`, tightened by the container's cgroup limit when one is set."""
    avail = None
    p = Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    cg = _cgroup_available_bytes()
    if cg is not None:
        avail = cg if avail is None else min(avail, cg)
    return avail


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
    host_available_bytes: Optional[int] = None,
) -> SlotPlan:
    """Plans `slots_per_layer` for `store_dir` on `backend` from the currently free memory."""
    num_layers, num_experts, stride, fixed, cfg_top_k = store_layout(store_dir)
    if backbone == "q4":  # shared_core -> 4.5b, hybrid hf_backbone -> ~7.5b (INT8 SSM + Q4 Attn), lm_head -> 4.5b
        sdir = Path(store_dir).expanduser().resolve()
        if not (sdir / "merkle_manifest.json").exists():
            fixed = int(fixed * 0.48)
        else:
            core = (sdir / "shared_core.bin").stat().st_size if (sdir / "shared_core.bin").exists() else 0
            hf_bb = (
                (sdir / "hf_backbone.safetensors").stat().st_size
                if (sdir / "hf_backbone.safetensors").exists()
                else 0
            )
            head = (sdir / "lm_head_int8.bin").stat().st_size if (sdir / "lm_head_int8.bin").exists() else 0
            fixed -= (core + hf_bb + head) - int(core * 4.5 / 16) - int(hf_bb * 7.5 / 16) - int(head * 4.5 / 8)
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
        host_available_bytes=host_available_bytes,
        host_limited=False,
    )
