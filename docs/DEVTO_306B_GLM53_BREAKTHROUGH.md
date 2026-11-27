---
title: "How We Ran a 306B Frontier MoE (GLM-5.3-Flash) on a Single 15 GB T4 & 24 GB GPU — Even With RAM Capped Below Model Size"
published: false
description: "A step-by-step systems breakdown of breaking the PCIe, VRAM, and Host RAM walls for 2026 300B+ Hybrid MoE models using Zero-PCIe CPU/GPU execution, TinyLFU expert slot arenas, and disjoint madvise paging."
tags: ai, llm, cuda, systems
---

Running a **306-billion-parameter** frontier model normally implies an 8×H100 or 4×A100 cluster.

What happens if all you have is a **single 15 GB NVIDIA Tesla T4** or a **single 24 GB PCIe GPU**—or worse, a container where **even Host RAM is hard-capped at `112 GiB`, `74 GiB` smaller than the quantized model itself**?

We put [`deimos-stream`](https://github.com/martianacademy/deimos-stream) to the ultimate live test on **`zai-org/GLM-5.3-Flash`** (`305.78 GiB` across 62 FP8 shards, `45` hybrid layers, `12,096` routed experts), head-to-head against upstream `llama.cpp` (`UD-Q4_K_XL`, `185.99 GiB`).

Here is the exact engineering journey—every wall we hit, how we solved it from first principles, verbatim live reasoning outputs from the 306B model running on a single GPU, and the raw cloud numbers.

---

## TL;DR — What Happened in Live Cloud Tests (`tau = 1.0`, Zero Expert Skipping)

1. **Single `15 GB NVIDIA Tesla T4`:** `DeimosStream` ran **`306B GLM-5.3-Flash` inside `10.71 GiB` PyTorch VRAM** (`11.3 GiB` `nvidia-smi` peak), generating clean structured Markdown. `llama.cpp --n-cpu-moe 42` crashed immediately trying to allocate a `20.88 GiB` CUDA buffer.
2. **Single `24 GB GPU` (`NVIDIA A10`, `224 GiB` RAM):**
   - `DeimosStream (--backbone q4 --slots auto)` hit **`3.44 tok/s`** decode and **`23.0s` TTFT** (`164.8 GiB` RSS)—beating `llama.cpp --n-cpu-moe 45` (**`2.92 tok/s`**, **`81.7s` TTFT**, `187.7 GiB` RSS) by **`+17.8%` on decode** and **`3.55×` on TTFT**, while `llama.cpp --n-cpu-moe 42` crashed with CUDA OOM even on 24 GB VRAM.
   - **Full Unquantized 16-Bit Backbone (`--backbone fp16 --slots 8`):** Kept all 45 layers of non-expert KDA/DSA attention, mHC streams, routers, and shared experts in **16-bit FP16** inside **`19.55–19.93 GiB` VRAM** at **`3.31 tok/s`**—still faster than `llama.cpp`!
3. **Out-of-Core `112 GiB` Hard-Capped Host RAM (`< 61%` of the `186 GiB` Q4 Model Size!):**
   - `DeimosStream` ran cleanly inside **`112.86 GiB` peak RSS** at **`2.14 tok/s`** and **`46.2s` TTFT**—**`2.10×` faster decode**, **`3.17×` faster TTFT**, and **`74.8 GiB` lower RSS** than `llama.cpp` (`1.02 tok/s`, `146.5s` TTFT, `187.66 GiB` RSS).

---

## Step 1: Anatomy of a 2026 306B Hybrid MoE (And Why Whole-Layer Offloading Breaks)

Modern late-2026 frontier models are no longer uniform Transformer stacks. Inspecting `zai-org/GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`) reveals five distinct architectural mechanisms operating simultaneously:

- **`45` Decoder Layers (`hidden_size = 4096`):** Layers `0..2` are dense (`intermediate_size = 12288`); layers `3..44` (`42` layers) are sparse Mixture-of-Experts with **`288` routed experts + `1` shared expert per layer** (`12,096` routed experts total, `top_k = 8`, `moe_intermediate_size = 2048`).
- **Hybrid Token Mixing (`34` KDA + `11` DSA Layers):** `34` layers use **Kimi Delta Attention (`KDA`)** gated linear recurrence; `11` layers use **DeepSeek Sparse Attention (`DSA`)** with compressed MLA latents (`kv_lora_rank = 512`, NoPE) and cross-layer `full`/`shared` k-pool indexers.
- **Manifold-Constrained Hyper-Connections (`mHC`):** Tokens propagate along `hc_mult` parallel residual streams mixed via Sinkhorn-Knopp doubly-stochastic matrices (`attn_hc`, `ffn_hc`).
- **Clamped SwiGLU (`swiglu_limit = 10.0`):** Expert gates clamp `gate <= 10.0` and `up in [-10.0, 10.0]` before `SiLU(gate) * up` to prevent FP8/FP16 outlier explosions.

### Why `llama.cpp --n-cpu-moe` Hits a Granularity Wall

`llama.cpp` offloads MoE experts **one entire layer at a time**:
- Each `GLM-5.3-Flash` MoE layer holds `288 × 13.5 MiB ≈ 3.80 GiB` of expert weights.
- Because layers `0, 1, 2` are dense, `--n-cpu-moe 42` only offloads layers `3..41`, leaving `3` MoE layers (`42, 43, 44` = `~11.4 GiB`) + `~9.5 GiB` of non-expert backbone/buffers = **`20.88 GiB` (`22,423,665,024 B`) on `CUDA0`**.
- Result: `--n-cpu-moe 42` **crashes with CUDA OOM on both a 15 GB T4 and a 23 GB A10**. You are forced all the way to `--n-cpu-moe 45` (**zero** MoE experts on GPU, leaving `~11 GiB` of your 24 GB GPU completely idle during MoE compute!).

---

## Step 2: Carving `305.8 GiB` Into a Permanent Backbone + `12,096` Page-Aligned Leaves

Instead of downloading `306 GiB` to disk up front, DeimosStream's reference-counted converter streams 3 safetensors shards ahead in background threads, dequantizes `[128, 128]` block-scaled `FP8` tensors on GPU in 32-expert chunks (`< 1.5 GiB` VRAM), deletes each shard the moment its reference count hits zero, and emits four purpose-built files:

```text
glm53_flash_deimos_q4/  (175.6 GiB total from 305.78 GiB FP8 source)
├── embed_tokens.bin         1.20 GiB   CPU mmap FP16 lookup table (0 MiB GPU VRAM!)
├── hf_backbone.safetensors 14.30 GiB   34 KDA + 11 DSA + 3 dense MLPs + 42 shared experts + mHC
├── lm_head_int8.bin       605.30 MiB   Symmetric INT8 output head (eliminates 1.2 GiB logit spikes)
├── cas_experts_q4.bin     159.42 GiB   12,096 experts × exact 14,155,776 B (13.500 MiB, 16 KB aligned)
└── merkle_tree.idx          1.80 MiB   DEIMIDX2 header + 12,096 SHA-256 leaf hashes + Tier-1 XOR root
```

Notice two subtle engineering wins during conversion and `meta`-skeleton loading:
1. **Zero-RAM `meta` Instantiation + Strict `FP16/FP32` Split:** Loading a 306B HuggingFace multimodal class (`Glm5NextForConditionalGeneration`) normally materializes `float32` parameters by default. Instantiating on `torch.device("meta")` with explicit `dtype=torch.float16` keeps dense projections in `FP16` (or in-place Group-64 `Q4` via `--backbone q4`, shrinking fixed VRAM to **`~5.4 GiB`**) while preserving sensitive recurrent and router buffers (`A_log`, `dt_bias`, `conv1d`, `gate.weight`, `e_score_correction_bias`, `mHC` scales) in exact `FP32`.
2. **Exact `16 KB` OS Page Alignment:** Every single expert (`gate_up_proj` + `down_proj` + Group-64 FP16 scales & mins) packs into **`14,155,776` bytes (`864 × 16,384 B`)** with **zero padding waste**, enabling direct `mmap` / `madvise` page operations per expert.

---

## Step 3: Sub-Layer TinyLFU GPU Slot Arena + Clamped `__dp4a` CUDA Kernels

Instead of pinning 288 experts in 3 layers and 0 in the other 39, `TorchPagedExpertArena` allocates **`S` physical expert slots inside every single one of the 42 MoE layers**:

- On a `24 GB` GPU (`--slots auto`), the planner budgets **`19–21` slots/layer** (`840` hot experts resident in VRAM simultaneously).
- Real natural-language routing is heavily skewed: caching just `20 / 288` (`6.9%`) of experts per layer via **TinyLFU frequency admission** yielded a **`27.6% – 31.2%` live cache hit rate** (`4.3×` higher than uniform random)!
- Hit slots execute directly on GPU through custom `nvcc` `__dp4a` (`sm_75` / `sm_86` / `sm_89`) Q4 GEMV kernels with fused **`swiglu_limit = 10.0`** clamping—skipping missed slots at zero warp cost.

---

## Step 4: The Core Breakthrough — Zero-PCIe Hybrid Execution (`16 KB` vs `4.75 GB`)

Here is the trap that kills naive "stream weights from RAM to GPU on demand" engines:

> At `top_k = 8` across `42` layers, `GLM-5.3-Flash` touches `336 × 13.5 MiB = 4.43 GiB` of expert weights **per token**. Even at a `30%` GPU cache hit rate, copying `3.1 GiB` of missed expert weights over PCIe Gen4 (`~12–16 GB/s` effective) stalls **every single token by `200–260 ms`** before a single FLOP executes!

So **DeimosStream never copies missed expert weights over PCIe during token decode.** Instead, it moves the tiny activation vector to the weights:

```text
                 GPU Stream (CUDA)                         Host CPU (22 AVX2 Workers)
        ┌──────────────────────────────────┐          ┌──────────────────────────────────┐
        │ 1. KDA / DSA Attn + mHC + Router │          │                                  │
        │ 2. Burst D2H: x_f16 (8 KB) + IDs ├─────────►│ 3. TinyLFU Split (Hits vs Misses)│
        │ 4. Shared Expert + Hit Q4 SwiGLU │  8 KB!   │ 4. Parallel Q8_1 × Q4 AVX2 GEMV  │
        │    (__dp4a on resident slots)    │          │    in-place over zero-copy mmap  │
        │ 5. Residual Add ◄────────────────┼──────────┤    -> Pinned H2D y_cpu (8 KB)    │
        └──────────────────────────────────┘  8 KB!   └──────────────────────────────────┘
        Background _dma_stream: asynchronously promotes <= 1 hot TinyLFU expert via ring buffer
```

1. Only **`~8.1 KB`** (`4096 × FP16` input + `8` expert IDs/weights) crosses PCIe D2H into pinned memory.
2. While the GPU computes the **shared expert + all TinyLFU hit experts** in parallel, C++17 worker threads quantize `x` once to `Q8_1` and run **`AVX2` fused clamped-SwiGLU dot products directly over `cas_mmap`** for the missed experts—sending back a single **`8 KB` FP16 vector**.
3. Meanwhile, an asynchronous CUDA `_dma_stream` promotes at most 1 newly hot expert into the TinyLFU ring buffer in the background without blocking decode.

---

## Step 5: Breaking the Host RAM Ceiling (`112 GiB` Capped RAM for a `186 GiB` Model!)

What happens when Host RAM (`112 GiB`) is **smaller than `cas_experts_q4.bin` (`159.5 GiB`)**?

Normally, staging `11.6 GiB` of hot experts into GPU VRAM leaves those exact same `11.6 GiB` duplicated in the Linux OS page cache, starving CPU-miss experts of RAM and triggering catastrophic NVMe swap thrashing (and note a nasty Linux `glibc` quirk we found: `posix_madvise(..., POSIX_MADV_DONTNEED)` is a **silent no-op** on Linux!).

DeimosStream calls the raw Linux `madvise(addr, 14155776, MADV_DONTNEED)` syscall the instant an expert is staged into VRAM, enforcing **strictly disjoint GPU-VRAM vs Host-RAM residency**, and drops `hf_backbone.safetensors` (`14.3 GiB`) from host page cache via `POSIX_FADV_DONTNEED` immediately after GPU upload.

*(And what about the `DEIMIDX2` SHA-256 + XOR Merkle tree? At runtime on trusted local NVMe, `--no-verify` adds **zero** overhead; on cold untrusted cloud/P2P mounts, enabling verification cryptographically checked **`10,146` SHA-256 expert leaves + `22,242` XOR proofs** on the fly with zero corrupted pages.)*

---

## Live Human Verification: Watching the 306B Model Think & Solve Its Own VRAM Math

Benchmarks mean nothing if logits are broken. Here are two **unedited live transcripts** generated by `306B zai-org/GLM-5.3-Flash` (`tau = 1.0`, greedy `temp = 0`) running on a **single `24 GB NVIDIA A10` GPU** (`17.8–18.3 GiB` PyTorch VRAM):

### Live Test 1 — Systems & Cryptography Design (`960` tokens, `3.27 tok/s`, `27.84%` GPU Hit Rate)

**Prompt:** *"Explain how a two-tier SHA-256 + XOR Merkle tree can cryptographically verify 4-bit Mixture-of-Experts weights streamed on demand into a single 16 GB GPU without stalling token generation, and include a concise Python pseudocode snippet."*

**Live `306B GLM-5.3-Flash` Output (verbatim):**

> `<think>User asks a technical design question. Provide explanation + pseudocode. Keep reasonably concise.</think>`
>
> **# Two-Tier SHA-256 + XOR Merkle Tree for On-Demand MoE Weight Verification**
>
> **## The Problem**
> A MoE model (e.g., 8 experts per layer, 4-bit quantized) may have 100+ GB of weights, but only a few experts per token are needed per forward pass. You stream expert shards from host storage/remote peers into a 16 GB GPU. The security question: **how do you know a streamed shard is authentic without hashing the whole model up front or stalling generation?**
>
> **## Design: Two-Tier Tree**
> - **Tier 1 — Chunk hashes (SHA-256):** Each expert shard (say 32 MB of 4-bit weights) is split into 64 KB chunks. Each chunk gets a SHA-256 digest. This lets you verify *one chunk at a time* as it streams — you never wait for a whole expert.
> - **Tier 2 — XOR aggregation (compact commitment):** All Tier-1 digests of an expert are XOR-folded into a single 32-byte "expert commitment." All expert commitments are XOR-folded into one **root commitment** published when the model is signed... *(Note: XOR alone is malleable, so Tier-1 SHA-256 digests are the collision-resistant layer; XOR is only an aggregation convenience over already-cryptographic digests.)*

```python
# ---- Offline (publisher) ----
def build_commitments(shards):                 # shards: list of expert weight blobs
    expert_commits = []
    for shard in shards:
        chunk_roots = merkle_root([sha256(shard[i:i+65536])
                                   for i in range(0, len(shard), 65536)])
        expert_commits.append(chunk_roots)
    root = xor_fold(expert_commits)            # 32-byte root commitment
    return sign(root, publisher_key)           # Ed25519 signature over root
```

*(Notice the 306B model independently pointed out the exact cryptographic subtlety that XOR is an order-independent aggregator whose collision resistance comes strictly from the underlying SHA-256 leaves!)*

### Live Test 2 — Asking the 306B Model to Calculate Its Own Engine's Exact VRAM Slot Math (`3.31 tok/s`, `31.20%` GPU Hit Rate)

**Prompt:** *"A cloud server has 22.5 GiB of usable GPU VRAM. A 306B-parameter MoE model has a 5.4 GiB quantized non-expert backbone and 42 MoE layers, where each expert takes 13.5 MiB in 4-bit precision. Leaving 1.5 GiB for KV cache and activations, calculate the exact maximum integer number of expert slots per layer that can reside in GPU VRAM simultaneously (note: 1 GiB = 1024 MiB)..."*

**Live `306B GLM-5.3-Flash` Reasoning Trace (verbatim):**

> `<think>...Available for experts = 22.5 - 5.4 - 1.5 = 15.6 GiB. Convert to MiB: 15.6*1024 = 15974.4 MiB. Each expert 13.5 MiB. Total expert slots across all layers = floor(15974.4 / 13.5) = 1183.288888... floor 1183. Then per layer = floor(1183 / 42) = 28.166... floor 28. If all 42 layers have 28 expert slots resident, memory used = 42*28*13.5 MiB = 15876 MiB = 15.50390625 GiB. Leaves 98.4 MiB! If 29 per layer: 42*29*13.5 = 16443 MiB = 16.0576171875 GiB > 15.6 GiB by 468.6 MiB... Since GPU can hold limited experts, remaining experts on CPU. A CPU engine performs matrix-vector multiplication for "cache misses" (experts not in GPU VRAM) using SIMD/AVX...</think>`

Every single step—`15,974.4 MiB` budget, `28` slots/layer (`15,876 MiB` used, `98.4 MiB` spare), and why `29` overflows by `468.6 MiB`—was derived live with zero arithmetic errors on a single GPU.

---

## Full Cloud Head-to-Head Benchmarks (`306B GLM-5.3-Flash`, `64` Greedy Tokens, `tau = 1.0`)

All runs used identical prompts, greedy decoding (`temp = 0`), `24` vCPUs (`-t 22`), and `unsloth/GLM-5.3-Flash-GGUF` (`UD-Q4_K_XL`, `185.99 GiB`) on upstream `llama.cpp` (`master`, CUDA 12.8.1).

### Suite 1: Single `24 GB GPU` (`NVIDIA A10, 23,028 MiB`) + `224 GiB` Host RAM

| Engine & Configuration | Status | Decode (`tok/s`) | TTFT (`s`) | GPU Peak (`MiB`) | Host RSS (`GiB`) | GPU Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DeimosStream (--backbone q4 --slots auto [19])`** | **OK** | **`3.44`** | **`23.00s`** | `17,377` (`18,817` smi) | **`164.81`** | `27.61%` |
| **`DeimosStream (--backbone q4 --slots 16)`** | **OK** | **`3.38`** | **`23.86s`** | `15,611` (`17,061` smi) | **`164.74`** | `21.17%` |
| **`DeimosStream (--backbone fp16 --slots 8)`** *(16-bit BB!)* | **OK** | **`3.31`** | **`29.26s`** | `20,019` (`21,821` smi) | **`164.57`** | `10.08%` |
| **`DeimosStream (--backbone q4 --slots 8)`** | **OK** | **`3.27`** | **`30.40s`** | `11,075` (`12,529` smi) | **`164.82`** | `10.65%` |
| `llama.cpp UD-Q4_K_XL (--n-cpu-moe 42)` | **OOM (exit 1)** | — | — | `22,424 req` (failed) | `187.66` | — |
| `llama.cpp UD-Q4_K_XL (--n-cpu-moe 45)` | OK | `2.92` | `81.65s` | `12,241` | `187.66` | `0.0%` |

### Suite 2: Out-of-Core `112 GiB` Hard-Capped Host RAM (`< 61%` of `186 GiB` Model!) on `24 GB GPU`

| Engine & Configuration | Status | Decode (`tok/s`) | TTFT (`s`) | GPU Peak (`MiB`) | Host RSS (`GiB`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`DeimosStream (--backbone q4 --slots auto [21])`** | **OK** | **`2.14` (`2.10×`)** | **`46.24s` (`3.17×`)** | `18,455` | **`112.86` (`-74.8 GiB`)** |
| **`DeimosStream (--backbone q4 --slots 8)`** | **OK** | **`2.07` (`2.03×`)** | **`52.69s` (`2.78×`)** | `11,075` | **`117.15` (`-70.5 GiB`)** |
| `llama.cpp UD-Q4_K_XL (--n-cpu-moe 42)` | **OOM (exit 1)** | — | — | `21,846` (failed) | `187.66` |
| `llama.cpp UD-Q4_K_XL (--n-cpu-moe 45)` | OK | `1.02` | `146.48s` | `11,176` | `187.66` |

### Suite 3: Single `15 GB NVIDIA Tesla T4` (`15,360 MiB`) + `224 GiB` Host RAM

| Engine & Configuration | Status | Decode (`tok/s`) | TTFT (`s`) | GPU Peak (`MiB`) | Host RSS (`GiB`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`DeimosStream (--backbone q4 --slots 8)`** | **OK** | `2.27` | **`81.45s`** | **`10,963`** (`11,305` smi) | **`164.57` (`-23.1 GiB`)** |
| `llama.cpp UD-Q4_K_XL (--n-cpu-moe 42)` | **OOM (exit 1)** | — | — | `22,424 req` (failed) | `187.66` |
| `llama.cpp UD-Q4_K_XL (--n-cpu-moe 45)` | OK | **`4.02`** | `82.23s` | `11,027` | `187.66` |

> **Engineering Honesty Note on the T4 Row:** Why did `llama.cpp --n-cpu-moe 45` hit `4.02 tok/s` vs `2.27 tok/s` on T4 when `224 GiB` RAM was fully uncapped? At `slots = 8 / 288` (`10.3%` GPU hit rate), `90%` of expert compute is strictly **host-DRAM bandwidth bound** (`~4.3 GB/token`). Unsloth's dynamic `UD-Q4_K_XL` quantizes less-sensitive routed experts down to `Q2_K`/`Q3_K` (`~3.4 bits/weight` on cold experts), reading ~25–30% fewer DRAM bytes per miss than DeimosStream's uniform `4.5 bits/weight` Group-64 affine Q4 across all 12,096 experts. Supporting `Q3_K` / `IQ3_XXS` sub-4-bit cold CAS leaves in `DEIMIDX2` is a straightforward next step that directly closes that pure-CPU bandwidth gap while keeping our `24 GB` and `112 GiB Capped-RAM` leads!

---

## Try It Yourself

```bash
pip install "deimos-stream[cuda]>=0.8.0"

# 1. Stream-convert 306B zai-org/GLM-5.3-Flash (< 25 GiB temporary shard buffer)
python -m deimos_stream convert-hf --repo zai-org/GLM-5.3-Flash --out ./glm53_deimos_q4

# 2. Run on any single 16 GB+ or 24 GB+ NVIDIA GPU (or Apple Silicon Mac via --backend mlx)
python -m deimos_stream chat --store-dir ./glm53_deimos_q4 --backend cuda --backbone q4 --slots auto --no-verify
```

Code, C++17 `AVX2/NEON` + `__dp4a` CUDA kernels, and benchmark scripts: **[github.com/martianacademy/deimos-stream](https://github.com/martianacademy/deimos-stream)**.
