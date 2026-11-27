---
title: "How You Can Run Massive LLMs Even on Consumer Hardware (5 Bottlenecks, Simple Fixes & Step-by-Step Code)"
published: false
description: "Why big open-source LLMs crash normal laptops and 8GB-16GB GPUs—and how fixing 5 simple memory bottlenecks in Python lets you run 35B to 306B models at home."
tags: ai, python, llm, opensource
---

You see an exciting new **35B** or **300B+** open-source AI model drop on HuggingFace.

You open your laptop, your gaming PC with an 8 GB–16 GB GPU, or a free Google Colab T4, run `AutoModelForCausalLM.from_pretrained(...)`—and within seconds your terminal explodes with:

```text
RuntimeError: CUDA out of memory.
```

Or worse, your system RAM hits 100%, swap kicks in, and your entire computer freezes before generating a single word.

Most people assume you *need* $30,000 enterprise GPUs to run big Large Language Models. **You don't.**

Running a massive modern LLM on consumer hardware isn't blocked by one impossible wall—it's blocked by **5 small, surprisingly simple software mistakes** that standard loaders make by default.

In this guide, we'll break down each of those **5 small bottlenecks** one by one:
1. **The Simple Problem** (in plain English + 1-line math)
2. **The Simple Fix** (how to rethink it)
3. **Step-by-Step Python Code** (clean, readable code showing exactly how it works)

At the end, you'll also see how to run all of this out-of-the-box in **3 commands**—running a **35-Billion-parameter** model inside **`2.7 GB` of GPU VRAM at `30 tokens/sec`** (or `43 tokens/sec` on a 15 GB T4), and even a **306-Billion-parameter** model on a single GPU.

---

## Bottleneck 1: Python Crashes *Before* Reading Any Weights

### The Simple Problem
When you initialize a PyTorch model the normal way (`model = AutoModelForCausalLM.from_config(config)`), PyTorch immediately allocates **random 32-bit numbers in your RAM** for every single parameter *before* it even opens your downloaded weights file.

* **The Math:** `35 Billion parameters × 4 bytes (float32) = 140 GB of useless random numbers!`
* Even a `32 GB` PC crashes before loading byte #1 from disk.

### The Simple Fix
Build the model on PyTorch's **`"meta"` device** first. A `meta` model is just an empty architectural blueprint—it knows every layer's shape and connections, while using **literally `0 bytes` of RAM and `0 bytes` of GPU VRAM**. Then you stream real `4-bit` or `16-bit` weights into it piece by piece.

### Step-by-Step Code

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

def create_zero_ram_skeleton(model_id: str):
    # Step 1: Download ONLY the tiny JSON config (~5 KB), not the 70 GB weights
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Step 2: Build the entire neural network inside PyTorch's "meta" device.
    # Memory used on CPU RAM = 0 bytes! Memory used on GPU VRAM = 0 bytes!
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch.float16,   # Keep default blueprint in 16-bit, NOT 32-bit!
            trust_remote_code=True,
        )

    return model
```

**Why this works:** Instead of needing `140 GB` of upfront scratch RAM just to construct the Python object, `create_zero_ram_skeleton()` finishes in **milliseconds using `0 MB`**.

---

## Bottleneck 2: Your Vocabulary Table Wastes 1–2 GB of GPU VRAM

### The Simple Problem
Modern 2026 LLMs support `150,000` to `250,000` tokens (words/subwords across dozens of languages and code).
If `vocab_size = 248,320` and `hidden_size = 4,096`, that first lookup table (`embed_tokens`) takes:

* **The Math:** `248,320 × 4,096 × 2 bytes (FP16) ≈ 2.03 GB of GPU VRAM!`
* **The Irony:** To generate the next word, the GPU only looks up **1 single row (`8 KB`)** out of those `248,320` rows! Keeping `2 GB` parked on a `6 GB` or `8 GB` GPU just to read `8 KB` per token is pure waste.

### The Simple Fix
Keep `embed_tokens.bin` on disk/RAM using operating-system **memory mapping (`mmap`)**. Let the CPU grab just the **`8 KB` row** for the current token ID and copy *only* those `8 KB` to the GPU. **GPU VRAM used = `0 MB`.**

### Step-by-Step Code

```python
import mmap
import numpy as np
import torch
import torch.nn as nn

class ZeroVramMmapEmbedding(nn.Module):
    def __init__(self, bin_path: str, vocab_size: int, hidden_size: int, device: str = "cuda"):
        super().__init__()
        self.device = device
        # Step 1: Open the raw FP16 embedding file WITHOUT loading it into RAM
        self._file = open(bin_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Step 2: Wrap the mmap buffer as a zero-copy 2D NumPy view [vocab_size, hidden_size]
        self.table = np.frombuffer(
            self._mmap, dtype=np.float16, count=vocab_size * hidden_size
        ).reshape(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Step 3: Pull token IDs to CPU (usually just 1 integer during generation!)
        ids_cpu = input_ids.detach().cpu().numpy()

        # Step 4: Slice ONLY the needed 8 KB row(s) and send that tiny slice to the GPU
        rows = np.ascontiguousarray(self.table[ids_cpu])
        return torch.from_numpy(rows).to(self.device, non_blocking=True)
```

**Result:** You instantly save **`1 GB to 2 GB` of GPU VRAM** with virtually zero speed penalty (`8 KB` takes `< 0.002 ms` to copy).

---

## Bottleneck 3: Loading All "Experts" When Each Word Uses Only 3%

### The Simple Problem
Almost every frontier open model today (`Qwen3.6`, `DeepSeek-V3/V4`, `GLM-5`, `Llama-4`, `Mixtral`) uses **Mixture-of-Experts (MoE)**.

Think of an MoE model like a hospital with **256 specialist doctors ("experts")** on every floor, where **each patient (token) only visits 8 doctors (`3.1%`)**:
* A **35B** model (`Qwen3.6-35B-A3B`) only activates **3B parameters** per word!
* Yet standard tools try to load **all 256 experts** onto your GPU—or offload *entire floors* to the CPU (`llama.cpp --n-cpu-moe`), leaving half your GPU sitting idle or crashing with OOM on 288-expert models.

### The Simple Fix
Instead of keeping *all* 256 experts in a layer or *zero* experts in a layer, give **every layer a small "VIP Parking Lot" (`Slot Arena`)** in GPU VRAM—say `16` or `32` slots per layer.

In natural human language (or Python code), **a small group of popular experts gets picked over and over again**. Tracking popularity with a simple frequency counter (**TinyLFU**) lets just `25%–50%` of slots answer **`70% to 88%` of expert lookups directly on the GPU!**

### Step-by-Step Code

```python
class LayerSlotCache:
    """Tracks which experts live inside a fixed-size GPU VRAM slot arena."""

    def __init__(self, num_slots: int, num_experts: int):
        # Step 1: Map expert_id -> gpu_slot (-1 means NOT currently on GPU)
        self.expert_to_slot = [-1] * num_experts
        self.slot_to_expert = [-1] * num_slots
        # Step 2: Count how often each expert is actually requested by the router
        self.freq = [0] * num_experts

    def split_hits_and_misses(self, selected_experts: list[int]):
        gpu_hits, cpu_misses = [], []

        for eid in selected_experts:
            self.freq[eid] += 1                      # Step 3: Record popularity
            slot = self.expert_to_slot[eid]
            if slot != -1:
                gpu_hits.append((eid, slot))         # Fast path: already in GPU VRAM!
            else:
                cpu_misses.append(eid)               # Cold path: not in VRAM right now

        return gpu_hits, cpu_misses

    def maybe_promote(self, cold_eid: int) -> int | None:
        # Step 4: Find the least-used expert currently parked on the GPU
        victim_slot = min(
            range(len(self.slot_to_expert)),
            key=lambda s: self.freq[self.slot_to_expert[s]] if self.slot_to_expert[s] != -1 else -1,
        )
        resident_eid = self.slot_to_expert[victim_slot]

        # Step 5: Only replace it if the new expert is strictly MORE popular!
        if resident_eid == -1 or self.freq[cold_eid] > self.freq[resident_eid]:
            if resident_eid != -1:
                self.expert_to_slot[resident_eid] = -1
            self.slot_to_expert[victim_slot] = cold_eid
            self.expert_to_slot[cold_eid] = victim_slot
            return victim_slot
        return None
```

**Why this works:** Random one-off experts don't kick out your hot coding/conversation experts. Your GPU VRAM usage stays **100% flat and predictable**—whether you give it `2.7 GB` (`8` slots/layer) or `12 GB` (`159` slots/layer).

---

## Bottleneck 4: The PCIe Cable Traffic Jam (`8 KB` vs `13.5 MB`!)

### The Simple Problem
Here is the biggest trap people fall into when building "weight streaming" engines:

> *"Whenever a needed expert isn't in GPU VRAM, I'll just copy its weights from CPU RAM to GPU VRAM, run it, and move on!"*

Let's check the math over a normal PCIe cable (`~12–16 GB/s` real-world speed):
* **1 single large expert's weights (4-bit):** `~2 MB` to `13.5 MB`
* **Across 40 layers (`2–5` misses per layer):** `200 MB` to `3,000 MB` copied **for every single word!**
* Your GPU spends **90% of its life stalled waiting on the PCIe cable**, crawling at `2–4 tokens/sec`.

### The Simple Fix
**Never move missed expert weights to the GPU during token generation. Move the tiny token vector to the CPU instead!**

Look at the size difference:
* Moving the **expert weights** to the GPU = **`13,500 KB`** per miss.
* Moving the **token hidden state** to the CPU = **`8 KB`** (`4096 × 2 bytes`)!

While the GPU computes all the **cached Hit experts**, your multi-core CPU simultaneously multiplies that `8 KB` vector against the **Missed experts** directly over `mmap` in RAM, and sends a single **`8 KB` result vector** back to add on the GPU!

```text
           GPU (VRAM Slot Arena)                        CPU (Multi-Core SIMD over mmap)
   +-----------------------------------+        +-------------------------------------+
   | 1. Attention + Router             |        |                                     |
   | 2. Send 8 KB vector + Miss IDs ---+------->| 3. Run ONLY Missed Experts in RAM   |
   | 4. Run Cached Hit Experts on GPU  |  8 KB  |    (Zero weight copies over PCIe!)  |
   | 5. Final Output = GPU + CPU <-----+--------+ 4. Return 8 KB vector to GPU        |
   +-----------------------------------+  8 KB  +-------------------------------------+
```

### Step-by-Step Code

```python
import torch

def hybrid_moe_forward(
    x_gpu: torch.Tensor,              # Shape: [1, hidden_size] (~8 KB in FP16)
    routed_ids: list[int],            # e.g. 8 chosen experts out of 256
    routed_weights: torch.Tensor,     # Router probabilities
    cache: LayerSlotCache,
    gpu_slot_arena,                   # Fast 4-bit GPU kernel on resident slots
    cpu_mmap_pool,                    # Parallel C++/SIMD threads over mmap file
) -> torch.Tensor:
    # Step 1: Split the 8 chosen experts into GPU Hits vs CPU Misses
    hits, misses = cache.split_hits_and_misses(routed_ids)

    # Step 2: Copy ONLY the tiny 8 KB activation vector to CPU (NOT megabytes of weights!)
    if misses:
        x_cpu = x_gpu.to("cpu", non_blocking=True)

    # Step 3: Launch GPU kernel for all cached Hits immediately (non-blocking)
    y_gpu = gpu_slot_arena.run_4bit_swiglu(x_gpu, hits, routed_weights)

    # Step 4: At the SAME TIME, CPU computes any Misses in-place over mmap
    if misses:
        y_cpu = cpu_mmap_pool.run_4bit_misses(x_cpu, misses, routed_weights)
        # Send just 8 KB back to GPU and sum the two partial answers!
        y_gpu = y_gpu + y_cpu.to(x_gpu.device, non_blocking=True)

        # Optional Step 5: Promote at most 1 newly hot expert in background DMA stream
        cache.maybe_promote(misses[0])

    return y_gpu
```

**Result:** On a 35B MoE model on an NVIDIA T4, eliminating PCIe weight stalls jumped speed from **`5.3 tokens/sec` -> `43.3 tokens/sec` (`8x` faster)** with **zero loss in output quality**!

---

## Bottleneck 5: Running a `21 GB` Model on a `12 GB` RAM Machine (Stopping OS Double-Caching)

### The Simple Problem
What if your quantized model is `21 GB` (or `186 GB`), and your system RAM is only `12 GB` (like free Colab or a budget PC)?

When you read `10 GB` of popular experts from your SSD via `mmap` and upload them into GPU VRAM, your operating system tries to be "helpful" and **keeps a duplicate copy of those exact same `10 GB` in system RAM** (the OS file page cache).
* Now `10 GB` of GPU VRAM + `10 GB` of System RAM hold the **exact same data**!
* Your remaining RAM runs out, and your SSD starts thrashing violently.

*(Bonus Linux gotcha: Python/C's standard `posix_madvise(..., POSIX_MADV_DONTNEED)` is actually a **silent no-op** in Linux `glibc`—it returns `0` without freeing a single byte!)*

### The Simple Fix
Pack each expert aligned to **`16 KB` OS memory pages**. The exact millisecond an expert finishes copying into GPU VRAM, call the raw Linux **`madvise(MADV_DONTNEED)`** syscall on that expert's slice so the OS drops the duplicate RAM pages immediately.

Now your **GPU VRAM** holds hot experts and your **System RAM** holds *only* non-overlapping cold experts!

### Step-by-Step Code

```python
import ctypes
import sys

MADV_DONTNEED = 4  # Linux kernel constant: "drop these clean file pages from RAM now"
_libc = ctypes.CDLL("libc.so.6") if sys.platform.startswith("linux") else None

def drop_duplicate_ram_pages(mmap_base_addr: int, expert_offset: int, expert_bytes: int) -> None:
    """Free Host RAM pages immediately after an expert is uploaded into GPU VRAM."""
    if _libc is None:
        return

    # Step 1: Must be aligned to OS page size (4 KB on x86 Linux, 16 KB on ARM/macOS)
    page_addr = mmap_base_addr + expert_offset

    # Step 2: Call raw Linux madvise() directly (NOT glibc's no-op posix_madvise!)
    # Because cas_experts_q4.bin is read-only on disk, the kernel safely frees the RAM
    # instantly and seamlessly re-reads from NVMe only if needed later.
    _libc.madvise(
        ctypes.c_void_p(page_addr),
        ctypes.c_size_t(expert_bytes),
        ctypes.c_int(MADV_DONTNEED),
    )
```

**Result:** With disjoint RAM + VRAM residency, a **`21 GB` 35B model runs smoothly inside a `12 GB` RAM limit at `29.1 tokens/sec`**, and a **`186 GB` 306B model runs inside a `112 GB` RAM limit (`2.1x` faster than `llama.cpp`)**!

---

## Try It Yourself in 3 Commands (Mac, Windows WSL2, Linux, or Colab)

We combined all 5 of these techniques—plus custom 4-bit CUDA `__dp4a` kernels, static CUDA Graphs, C++17 `AVX2`/`NEON` CPU pools, and native **Apple Silicon Metal (`MLX`)** kernels—into an open-source MIT engine: **[`deimos-stream`](https://github.com/martianacademy/deimos-stream)**.

```bash
# 1. Install (Apple Silicon Mac uses native Metal MLX; Linux/WSL2 uses NVIDIA CUDA)
pip install -U deimos-stream          # macOS (M1 / M2 / M3 / M4)
pip install -U "deimos-stream[cuda]"  # Linux / Windows WSL2 / Colab NVIDIA GPU

# 2. Stream-convert a HuggingFace MoE model into 16KB page-aligned 4-bit format
# (Try LiquidAI/LFM2.5-8B-A1B for laptops, or Qwen/Qwen3.6-35B-A3B for 6GB-16GB GPUs!)
deimos-stream convert \
  --model-dir Qwen/Qwen3.6-35B-A3B \
  --output-dir ./qwen36_35b_q4

# 3. Chat! `--slots auto` automatically sizes the GPU slot cache to fit your free VRAM
deimos-stream chat \
  --store-dir ./qwen36_35b_q4 \
  --backbone q4 \
  --slots auto \
  --tau 1.0 \
  --no-verify
```

Want to force a **35-Billion-parameter** model (`67 GB` original size) into **under `2.8 GB` of GPU VRAM** (less VRAM than a lightweight video game)? Just cap `--slots 8`:

```bash
deimos-stream chat --store-dir ./qwen36_35b_q4 --backbone q4 --slots 8 --tau 1.0 --no-verify
```

### Real Verified Speeds on Everyday & Single-GPU Hardware

| Model (`Original Size`) | Hardware Tested | GPU VRAM Used | Generation Speed |
| :--- | :--- | :---: | :---: |
| **`OLMoE-1B-7B` (`13.2 GB`)** | MacBook (`Apple Silicon` Metal, Ultra-Low Mode) | **`0.66 GB` (`676 MiB`)** | **`75.7 tok/s`** |
| **`OLMoE-1B-7B` (`13.2 GB`)** | MacBook (`Apple Silicon` Metal, `--slots auto`) | **`3.82 GB`** | **`422.0 tok/s`** |
| **`Qwen3.6-35B-A3B` (`67.0 GB`)** | Single `NVIDIA T4` (`--slots 8`, Ultra-Low VRAM) | **`2.71 GB`** | **`29.9 tok/s`** |
| **`Qwen3.6-35B-A3B` (`67.0 GB`)** | Single `NVIDIA T4` (`--slots 64`, 8GB-Class Budget) | **`6.43 GB`** | **`38.8 tok/s`** |
| **`Qwen3.6-35B-A3B` (`67.0 GB`)** | Single `NVIDIA T4` (`--slots auto`, 15GB GPU) | **`12.73 GB`** | **`43.3 tok/s`** |
| **`GLM-5.3-Flash 306B` (`305.8 GB`)** | Single `15 GB T4` / `24 GB A10` GPU | **`10.7 GB` / `17.0 GB`** | **`2.3` – `3.44 tok/s`** |

---

## Key Takeaway

Whenever someone tells you consumer hardware can't run modern LLMs, look at **what the bytes are actually doing**:
1. Don't allocate random `float32` tensors before loading (`torch.device("meta")`).
2. Don't park `2 GB` lookup tables in GPU VRAM to read `8 KB` per word (`mmap`).
3. Cache **popular individual experts** in a fixed VRAM arena instead of whole layers (`TinyLFU`).
4. Never copy megabytes of missed weights over PCIe—send the **`8 KB` activation** to the CPU instead.
5. Drop duplicate OS page-cache copies (`madvise(MADV_DONTNEED)`) as soon as pages hit VRAM.

All source code (`Python`, `C++17`, `CUDA`, and `Apple MLX`) is 100% open-source (MIT) here: **[github.com/martianacademy/deimos-stream](https://github.com/martianacademy/deimos-stream)**.

If there's one specific bottleneck above—like writing your own **4-bit GPU kernel from scratch**, **streaming 300 GB conversions on a tiny SSD**, or **Apple Silicon Unified Memory zero-copy tricks**—that you want a dedicated deep-dive tutorial on next, drop a comment below!
