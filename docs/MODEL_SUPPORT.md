# DeimosStream Universal MoE Model Support & Hardware Sizing Guide (`v0.7.0`)

DeimosStream uses a **trait-based Universal MoE Adapter system** (`src/deimos_stream/model_adapters.py` and `src/deimos_stream/mlx_universal.py`) rather than maintaining separate per-model forks. You can inspect the live registry at any time via:

```bash
deimos-stream models
```

---

## 1. How the Universal Adapter Works

Across modern HuggingFace `transformers` (`>= 5.0`) and Apple Silicon `mlx-lm`, 95% of Sparse and Hybrid MoE checkpoints share a common routed SwiGLU structure once four structural variations are normalized during `deimos-stream convert` and engine startup:

1. **Automatic Mixed Dense + MoE Layer Discovery (`moe_layer_ids`):**
   Models such as `DeepSeek-V2/V3/R1` (`first_k_dense_replace = 1..3`), `GLM-4.5` (`first_k_dense_replace = 1`), `Llama-4 Maverick` (`interleave_moe_layer_step = 2`), `ERNIE-4.5-MoE` (`moe_layer_interval`), and `Jamba` mix dense MLP layers with routed MoE layers. `discover_hf_moe_layers()` inspects `weight_map` directly, packs **only** MoE layers into `cas_experts_q4.bin` (`merkle_tree.idx`), keeps dense MLP layers inside `hf_backbone.safetensors`, and automatically strips vision towers (`visual.*`, `vision_model.*`) and extra Multi-Token Prediction draft layers (`layers.{>= num_hidden_layers}`).
2. **Block-Scaled `FP8 E4M3` Dequantization (`weight_scale_inv`):**
   Frontier checkpoints published natively in `FP8` (`DeepSeek-V3`, `DeepSeek-R1`, `DeepSeek-V3.2`, `Kimi-K2`, `Qwen3-FP8`, `GLM-4.5-FP8`) store `float8_e4m3fn` weights paired with `[128, 128]` or `[32, 32]` `weight_scale_inv` tensors. `pop_tensor_dequant()` dequantizes these blocks on the fly during conversion into Group-64 Q4 `DEIMIDX2` Merkle leaves and FP16 `hf_backbone.safetensors`, removing `quantization_config` from the runtime `config.json`.
3. **Canonical Expert Layout & `O(top_k)` Router Binding:**
   Handles 2D per-expert shards (`gate_proj/up_proj/down_proj` or `w1/w3/w2`), 3D stacked `[E, 2I, D]` tensors, and transposed 3D `Llama-4` `[E, D, 2I]` / `[E, I, D]` tensors. On `Llama-4`, `bind_hf_moe_layer()` replaces HuggingFace's `O(num_experts)` dense `torch.bmm` repeat with `O(top_k)` sparse paged execution.
4. **Dual Native Execution (`CUDA` + Apple Silicon `MLX`):**
   - **NVIDIA CUDA / Linux (`DeimosHFHybridMoEEngine`):** Instantiates any `AutoModelForCausalLM` skeleton on `torch.device("meta")` (`0` bytes RAM), binds `DeimosPagedHFExperts` (`__dp4a` CUDA Q4 + zero-PCIe C++17 `AVX2` miss engine), and supports `--backbone q4` via `DeimosQ4Linear`.
   - **Apple Silicon Metal (`DeimosMLXUniversalMoEEngine` & `DeimosMLXHybridMoEEngine`):** Resolves the native `mlx_lm.models.<arch>` class (`MLA`, `GatedDeltaNet`, `Mamba2`, `ChunkedKVCache`, `YaRN`), swaps every `SwitchGLU` layer before `mx.eval` (`0` bytes allocated for unpaged experts) with `DeimosMLXSwitchGLU` (`mx.gather_qmm`), and runs `model.sanitize()` (splitting `MLA` `kv_b_proj` automatically).

---

## 2. Supported Model Families & Hardware Requirements

All models convert with the same command (pass either a HuggingFace repo ID or a local directory):

```bash
deimos-stream convert --model-dir <repo_id_or_local_dir> --output-dir ./my_deimos_store
deimos-stream chat --store-dir ./my_deimos_store --backbone q4 --tau 1.0 --no-verify
```

| Family | `model_type` | Attention & Routing | Representative Models | `DEIMIDX2` Disk Size | Minimum VRAM (`--slots 8..16`) | Recommended Host RAM (Fast Hybrid) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **OLMoE / Phi-MoE** | `olmoe`, `phimoe` | GQA (+ QK-Norm), Softmax Top-8/Top-2 | `allenai/OLMoE-1B-7B-0125-Instruct` (`7B`) | `3.4 GiB` | **`0.7 – 1.1 GiB`** | `8 – 16 GB` |
| **DeepSeek-V2-Lite / Moonlight** | `deepseek_v2`, `deepseek_v3` | **MLA**, First-1 Dense, Softmax / NoAux Sigmoid Top-6 | `deepseek-ai/DeepSeek-V2-Lite-Chat` (`16B`), `moonshotai/Moonlight-16B-A3B-Instruct` | `~9.2 GiB` | **`2.2 – 3.2 GiB`** | `16 – 32 GB` |
| **Qwen3-MoE (`30B`)** | `qwen3_moe`, `qwen2_moe` | GQA + QK-Norm, Softmax Top-8 (`128E`) | `Qwen/Qwen3-30B-A3B` (`30.5B`) | `~16.8 GiB` | **`2.6 – 4.2 GiB`** | `16 – 32 GB` |
| **Qwen3.6 / 3.5 / Next Hybrid** | `qwen3_5_moe`, `qwen3_next` | **GatedDeltaNet + GQA**, Gated Shared Expert (`256E` Top-8) | `Qwen/Qwen3.6-35B-A3B` (`35B`), `Qwen/Qwen3-Next-80B-A3B` | `21.0 GiB` *(35B)* / `~44 GiB` *(80B)* | **`2.7 GiB`** *(35B, verified T4)* / **`~5.5 GiB`** *(80B)* | **`12 – 32 GB`** *(35B: `29–43 tok/s` T4)* / `64 GB` *(80B)* |
| **Mixtral** | `mixtral` | GQA / SWA, Softmax Top-2 (`8E`) | `mistralai/Mixtral-8x7B-Instruct-v0.1` (`46.7B`) | `~23.5 GiB` | **`5.8 – 6.5 GiB`** | `32 – 64 GB` |
| **GLM-4.5 / GLM-4.5-Air** | `glm4_moe`, `glm4_moe_lite` | Partial-RoPE GQA / MLA, First-1 Dense, NoAux Sigmoid Top-8 | `zai-org/GLM-4.5-Air` (`106B-A12B`) | `~58 GiB` | **`6.5 – 9.5 GiB`** | `64 – 128 GB` |
| **Llama 4 (Scout & Maverick)** | `llama4`, `llama4_text` | Chunked GQA + iRoPE, Sigmoid Top-1 + Shared Expert | `Llama-4-Scout-17B-16E` (`109B`), `Llama-4-Maverick-17B-128E` (`400B`) | `~59 GiB` *(Scout)* / `~212 GiB` *(Maverick)* | **`8 – 12 GiB`** *(Scout)* / **`12 – 16 GiB`** *(Maverick)* | `64 – 128 GB` *(Scout)* / `256+ GB` *(Maverick)* |
| **Qwen3-235B-A22B** | `qwen3_moe` | GQA + QK-Norm, Softmax Top-8 (`128E`, `94` layers) | `Qwen/Qwen3-235B-A22B` (`235B`) | `~122 GiB` | **`14 – 20 GiB`** | `128 – 256 GB` |
| **DeepSeek-V3 / R1 / V3.2 / Kimi-K2** | `deepseek_v3`, `deepseek_v32` | **MLA + YaRN**, First-3 Dense, NoAux Sigmoid Top-8 (`256E`/`384E`), `FP8` shards | `deepseek-ai/DeepSeek-V3` (`671B`), `deepseek-ai/DeepSeek-R1`, `Kimi-K2` (`1T`) | `~355 GiB` *(671B)* / `~530 GiB` *(1T)* | **`16 – 24 GiB`** *(backbone + minimal slots)* | `64–128 GB` *(NVMe-paged ~1–3 tok/s)* / **`384–512 GB`** *(RAM-cached ~10–20+ tok/s)* |
| **MiniMax / ERNIE-4.5 / Jamba / Granite** | `minimax_m2`, `ernie4_5_moe`, `jamba`, `granitemoehybrid` | GQA or Hybrid Mamba2-SSM + GQA, Interleaved MoE | `MiniMaxAI/MiniMax-M2`, `baidu/ERNIE-4.5-21B-A3B-PT`, `AI21-Jamba-1.5-Mini` | `12 – 120 GiB` | **`3 – 12 GiB`** | `16 – 128 GB` |

*(Note: `OLMoE-1B-7B` on Apple M4 Max and `Qwen/Qwen3.6-35B-A3B` on NVIDIA Tesla T4 are measured full-checkpoint benchmarks; `DeepSeek-V3` (`FP8` + `MLA`), `GLM-4.5`, and `Llama-4` adapters are verified for forward & KV-cached decode parity (`Pearson r > 0.995`) in `tests/test_model_adapters.py`, with analytical memory envelopes derived from exact tensor dimensions.)*

---

## 3. Roadmap Architectures (Not Yet Supported Out-of-the-Box)

DeimosStream validates `config.json` before downloading shards (`validate_convertible_architecture`) and raises an immediate, descriptive error on checkpoints whose expert math differs from standard bias-free unclamped SwiGLU `DEIMIDX2` leaves:

1. **`deepseek-ai/DeepSeek-V4.1-Flash` (`model_type: "deepseek_v41"` — `~748B` total: `552B` MoE + `196B` Engram):**
   - **Why it needs dedicated extensions:** Uses a custom `20+20` Causal Encoder-Decoder (`CED`) + Compressed Sparse Attention 2 (`CSA2`), `NVFP4`/`UE8M0` `[32, 32]` expert quantization, clamped SwiGLU (`swiglu_limit = 10.0`), and two `384M`-row INT8 `Engram` conditional memory lookup tables (`~185 GiB` on layers `1` & `14`).
   - **Expected Hardware Specs once supported:** `16–24 GB` GPU VRAM + **`1 TB` PCIe Gen4/Gen5 NVMe SSD** (`~500 GB` store, `Engram` `mmap`-streamed in `0 GiB` VRAM) + `64–128 GB` RAM (`~1.5–4 tok/s` SSD-paged) or `384–512 GB` RAM / Mac Studio `512 GB` (`~12–50+ tok/s`).
2. **`openai/gpt-oss-20b` & `openai/gpt-oss-120b` (`model_type: "gpt_oss"`):**
   - **Why it needs dedicated extensions:** Ships `MXFP4` expert blocks with **per-expert linear biases** (`gate_up_proj_bias`, `down_proj_bias`) and `alpha = 1.702` clamped SwiGLU (`swiglu_limit = 7.0`), whereas `DEIMIDX2` v2 expert leaves store bias-free Group-64 Q4 `(weight, scale, zero_bias)`.

---

## 4. Adding a New MoE Adapter (`< 30` Lines)

If a new HuggingFace / `mlx-lm` Sparse MoE model uses standard SwiGLU experts (`silu(gate) * up -> down`) without per-expert linear biases:

1. **Check `discover_hf_moe_layers()` in `src/deimos_stream/model_adapters.py`:** If the model places experts under a new attribute name (beyond `mlp`, `feed_forward`, or `block_sparse_moe`), add that container name to the tuple.
2. **Check `extract_layer_experts_gu_dn()`:** If 3D expert tensors are stored transposed (`[E, D, 2I]`), the shape check `(gu_t.shape[1] == hidden_size and gu_t.shape[2] == 2 * inter_size)` already auto-transposes them.
3. **Check `bind_hf_moe_layer()`:** If `moe_mod.experts` already takes `(hidden_states, top_k_index, top_k_weights)`, no custom wrapper is needed; if the parent MoE block has a custom `shared_expert` or router signature, add a compact branch in `bind_hf_moe_layer()`.
4. **Add an end-to-end parity test in `tests/test_model_adapters.py`** verifying `Pearson r > 0.995` on both `backend="cpu"` (`DeimosHFHybridMoEEngine`) and `backend="mlx"` (`DeimosMLXUniversalMoEEngine`), then register the family in `SUPPORTED_MOE_FAMILIES`.
