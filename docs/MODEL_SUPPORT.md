# DeimosStream 2026 Frontier MoE Model Support & Hardware Sizing Guide

DeimosStream uses a **trait-based Universal MoE Adapter system** ([`model_adapters.py`](../src/deimos_stream/model_adapters.py), [`model_catalog.py`](../src/deimos_stream/model_catalog.py), and [`mlx_universal.py`](../src/deimos_stream/mlx_universal.py)) purpose-built for **2026 Sparse & Hybrid MoE architectures**. Inspect the live registry anytime via:

```bash
deimos-stream models
```

---

## 1. 2026 Frontier Architectural Primitives Handled Automatically

Across late-2026 HuggingFace `transformers` (`>= 5.24`) and Apple Silicon `mlx-lm`, DeimosStream normalizes five structural variations during `deimos-stream convert` and engine startup:

1. **Clamped & Alpha-Scaled SwiGLU (`swiglu_limit`, `swiglu_alpha`, `up + 1`):**
   - **Standard SwiGLU (`Qwen3.6 / Qwen3.8`, `Kimi-K3 / K2.5`, `LiquidAI LFM2.5-MoE`, `GLM-4.7-Flash`, `Ling-3.2`):** `silu(gate) * up`
   - **Clamped SwiGLU (`DeepSeek-V4-Flash`, `GLM-5.3-Flash`, `HY4-Flash`):** `silu(min(gate, 10.0)) * clamp(up, -10.0, 10.0)`
   - **SwiGLU-OAI (`MiniMax-M3.1-Flash`, `MiniMax-M3`):** `min(gate, 7.0) * sigmoid(1.702 * min(gate, 7.0)) * (clamp(up, -7.0, 7.0) + 1.0)`
   Supported natively across **CUDA `__dp4a` (`deimos_cuda_q4.cu`)**, **Triton (`triton_q4.py`)**, **C++17 `AVX2/NEON` (`deimos_cpu_moe.cpp`)**, and **Apple Silicon `MLX` (`DeimosMLXSwitchGLU`)**.
2. **`UE8M0` (`.scale`) & Block-Scaled `FP8 E4M3` (`.weight_scale_inv`) Dequantization:**
   Automatically decodes power-of-two `float8_e8m0fnu` / `uint8` `E8M0` scales (`DeepSeek-V4-Flash` `layers.*.ffn.experts.*.w1.{weight,scale}`, `MXFP8`) and `float32` block scales (`GLM-5.3-Flash`, `Qwen3.6/3.8-FP8`, `DeepSeek-V3/V3.2`) into Group-64 Q4 `DEIMIDX2` Merkle leaves.
3. **Mixed Dense + MoE Layer Discovery (`moe_layer_ids`):**
   Scans `mlp`, `feed_forward`, `block_sparse_moe`, and `ffn` containers (`first_k_dense_replace`, `num_dense_layers`, `mlp_layer_types`, `moe_layers`), keeping dense prefix layers in `hf_backbone.safetensors` while stripping vision towers and MTP draft heads.
4. **`transformers` 5.x `conversion_mapping` & Multimodal Text Extraction:**
   Uses `convert_and_load_state_dict_in_model` on `meta`-device skeletons (`0` bytes host RAM for unpaged experts) and automatically extracts the language backbone from multimodal MoEs (`GLM-5.3-Flash` `Glm5NextForConditionalGeneration`, `Qwen3.8-Flash-Next`, `MiniMax-M3.1-Flash`).
5. **Dual Native Execution (`CUDA` + Apple Silicon `MLX`):**
   Runs one unified `DEIMIDX2` store on **NVIDIA CUDA** (`__dp4a` Q4 + zero-PCIe C++17 `AVX2` miss engine) and **Apple Silicon Metal** (`DeimosMLXHybridMoEEngine` & `DeimosMLXUniversalMoEEngine` `mx.gather_qmm`).

---

## 2. 2026 Frontier Model Matrix & Hardware Sizing

```bash
deimos-stream convert --model-dir <repo_id_or_local_dir> --output-dir ./my_deimos_store
deimos-stream chat --store-dir ./my_deimos_store --backbone q4 --tau 1.0 --no-verify
```

| 2026 Model Family | `model_type` | Attention & Routing | Representative 2026 Checkpoints | `DEIMIDX2` Store | Min GPU VRAM (`--slots 8..16`) | Recommended Host RAM |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **LiquidAI LFM2.5-MoE** *(Oct 2026)* | `lfm2_moe` | Hybrid `LIV ShortConv` + QK-Norm GQA, Top-4 (`32E`/`64E`) | `LiquidAI/LFM2.5-8B-A1B`, `LiquidAI/LFM2.5-24B-A2B` | `~4.4 GiB` *(8B)* / `~12.5 GiB` *(24B)* | **`1.2 – 2.4 GiB`** | `8 – 24 GB` |
| **GLM-4.7-Flash** *(2026)* | `glm4_moe_lite` | **MLA**, First-1 Dense, NoAux Sigmoid Top-4 (`64E`) | `zai-org/GLM-4.7-Flash` (`30B-A3B`) | `~16.2 GiB` | **`2.5 – 3.8 GiB`** | `16 – 32 GB` |
| **Qwen3.6 / Qwen3.8 Hybrid** *(2026)* | `qwen3_5_moe`, `qwen4_exp_text` | **GatedDeltaNet + QSA/GQA**, Gated Shared Expert (`256E`/`512E`) | `Qwen/Qwen3.6-35B-A3B` (`35B`), `Qwen/Qwen3.8-Flash-Next` | **`21.0 GiB`** *(35B)* / `~92 GiB` *(3.8-Next)* | **`2.7 GiB`** *(35B, verified T4)* / `~9 GiB` | **`12 – 32 GB`** *(35B: `29–43 tok/s` T4)* / `96+ GB` |
| **Kimi-K3 / K2.5 / Kimi-Linear** *(Nov 2026)* | `kimi_linear`, `kimi_k3`, `kimi_k25` | Hybrid **KDA** + **MLA**, NoAux Sigmoid (`256E`–`896E`) | `moonshotai/Kimi-Linear-48B-A3B-Instruct`, `moonshotai/Kimi-K2.5`, `moonshotai/Kimi-K3` | `~26 GiB` *(48B)* / `~310+ GiB` *(K2.5/K3)* | **`3.5 GiB`** *(48B)* / `16–24 GiB` *(K2.5/K3)* | `32 GB` *(48B)* / `128–512 GB` |
| **Ling-3.2 / Bailing-MoE-V3** *(Nov 2026)* | `bailing_hybrid`, `bailing_moe_v3` | Hybrid **KDA** + **MLA**, Sigmoid Top-8 (`512E`) | `inclusionAI/Ling-3.2-air`, `inclusionAI/Ling-3.1-flash` | `~34 GiB` *(Air)* / `~145 GiB` *(Flash)* | **`4.5 – 12 GiB`** | `32 – 192 GB` |
| **DeepSeek-V4-Flash** *(Aug 2026)* | `deepseek_v4` | **MLA + HCA/CSA + mHC**, `sqrtsoftplus` Top-6 (`256E`), `UE8M0`, `limit=10` | `deepseek-ai/DeepSeek-V4-Flash-0731` (`155B`) | `~82 GiB` | **`8 – 12 GiB`** | `64 – 128 GB` *(fits 128 GB M4 Max / Cloud VM)* |
| **GLM-5.3-Flash / GLM-5.2** *(Aug 2026)* | `glm5_next` | Linear Attn + **DSA/MLA + mHC**, Sigmoid Top-8 (`288E`), `limit=10` | `zai-org/GLM-5.3-Flash` (`305B`), `zai-org/GLM-5.2` | `~160 GiB` | **`12 – 16 GiB`** | `64–128 GB` *(paged)* / `192+ GB` *(cached)* |
| **MiniMax-M3.1-Flash / M3** *(Nov 2026)* | `minimax_m3_vl` | Sliding/Full GQA + `GemmaRMSNorm`, `SwiGLU-OAI` (`alpha=1.702, limit=7`) | `MiniMaxAI/MiniMax-M3.1-Flash`, `MiniMaxAI/MiniMax-M3` | `~220 GiB` | **`12 – 18 GiB`** | `128 – 256+ GB` |
| **DeepSeek-V3.2 / R1 & Llama-4** | `deepseek_v32`, `deepseek_v3`, `llama4` | **MLA + YaRN** or Chunked iRoPE GQA | `deepseek-ai/DeepSeek-V3.2`, `Llama-4-Scout-17B-16E` (`109B`) | `~59 GiB` *(Scout)* / `~355 GiB` *(671B)* | **`8 – 20 GiB`** | `64 – 384+ GB` |

*(Legacy 2024–2025 models including `Qwen3-MoE`, `DeepSeek-V2`, `ERNIE-4.5-MoE`, `Jamba`, `Mixtral`, and `OLMoE` remain backward-compatible.)*

---

## 3. Roadmap Formats (Validated Before Download)

`validate_convertible_architecture()` checks `config.json` upfront and gives an immediate explanation on three non-standard formats:

1. **`DeepSeek-V4.1-Flash` w/ `engram_layer_ids` (`deepseek_v41`, `748B`):** Standard `deepseek_v4` (`DeepSeek-V4-Flash-0731`, `155B`) works today; `V4.1` models with `engram_layer_ids` require `20+20` CED + `185 GiB` `mmap` hashed `Engram` tables.
2. **`thinkingmachines/Inkling-2-Mini` (`inkling2_loop`):** Requires multi-turn looped decoder state with dual-tanh `Inkling2LoopSoftcappedSwiGLU`.
3. **`openai/gpt-oss-20b` / `120b` (`gpt_oss`):** Uses per-expert linear biases (`gate_up_proj_bias`, `down_proj_bias`) outside bias-free `DEIMIDX2` Q4 leaves.
