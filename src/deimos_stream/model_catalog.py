#!/usr/bin/env python3
"""
DeimosStream 2026 Frontier Sparse & Hybrid-MoE Family Catalog (`src/deimos_stream/model_catalog.py`)
====================================================================================================
Canonical registry of verified late-2026 Sparse, Hybrid, and Multimodal MoE architectures supported
by DeimosStream (`DEIMIDX2` Merkle Q4 paging + `__dp4a` CUDA / `AVX2/NEON` / Apple Silicon `MLX`),
plus explicit roadmap status for non-standard leaf formats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MoEFamilySpec:
    family: str
    model_types: Tuple[str, ...]
    attention_kind: str
    routing_kind: str
    example_models: Tuple[str, ...]
    backends: Tuple[str, ...] = ("mlx", "cuda", "cpu")
    status: str = "verified"
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


SUPPORTED_MOE_FAMILIES: Tuple[MoEFamilySpec, ...] = (
    MoEFamilySpec(
        family="Qwen3.8 / Qwen3.6 / Qwen3-Next Hybrid (2026)",
        model_types=("qwen4_exp", "qwen4_exp_text", "qwen3_5_moe", "qwen3_next"),
        attention_kind="Hybrid GatedDeltaNet + QSA / QK-Norm GQA",
        routing_kind="Softmax Top-K + Sigmoid-Gated Shared Expert",
        example_models=(
            "Qwen/Qwen3.8-Flash-Next",
            "Qwen/Qwen3.6-35B-A3B",
            "Qwen/Qwen3.6-35B-A3B-FP8",
        ),
        notes="81-segment static CUDA Graphs (43.3 tok/s T4 on Qwen3.6-35B-A3B) + native Metal GatedDeltaNet.",
    ),
    MoEFamilySpec(
        family="DeepSeek-V4 / DeepSeek-V3.2 / R1 (2026)",
        model_types=("deepseek_v4", "deepseek_v32", "deepseek_v3"),
        attention_kind="MLA + HCA/CSA Sparse Indexer + mHC HyperConnections",
        routing_kind="NoAux SqrtSoftplus / Sigmoid Top-K + Clamped SwiGLU (limit=10.0)",
        example_models=(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "deepseek-ai/DeepSeek-V3.2",
            "deepseek-ai/DeepSeek-R1",
        ),
        notes="Auto-dequantizes UE8M0 (.scale) & FP8 (.weight_scale_inv) [128,128] blocks + native limit=10.0 SwiGLU.",
    ),
    MoEFamilySpec(
        family="GLM-5.3-Flash / GLM-5 / GLM-4.7-Flash (2026)",
        model_types=("glm5_next", "glm5_next_text", "glm4_moe_lite", "glm4_moe"),
        attention_kind="Hybrid Linear Attention + DSA/MLA + HyperConnections",
        routing_kind="NoAux Sigmoid Top-K + Shared Experts + Clamped SwiGLU (limit=10.0)",
        example_models=(
            "zai-org/GLM-5.3-Flash",
            "zai-org/GLM-5.2",
            "zai-org/GLM-4.7-Flash",
        ),
        notes="Handles text-only extraction from multimodal Glm5Next checkpoints + first_k_dense_replace.",
    ),
    MoEFamilySpec(
        family="Kimi-K3 / Kimi-K2.5 / Kimi-Linear (Nov 2026)",
        model_types=("kimi_linear", "kimi_k3", "kimi_k25", "kimi_k2"),
        attention_kind="Hybrid KimiDeltaAttention (KDA) + MLA",
        routing_kind="NoAux Sigmoid Top-K + Shared Experts",
        example_models=(
            "moonshotai/Kimi-K3",
            "moonshotai/Kimi-K2.5",
            "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        ),
        notes="Supports KDA short-conv / forget-gate conversion mappings on both HF CUDA and Apple Silicon MLX.",
    ),
    MoEFamilySpec(
        family="MiniMax-M3.1-Flash / MiniMax-M3 (Nov 2026)",
        model_types=("minimax_m3_vl", "minimax_m3_vl_text", "minimax_m2", "minimax"),
        attention_kind="Interleaved Sliding / Full GQA + GemmaRMSNorm",
        routing_kind="NoAux Sigmoid Top-K + Shared Expert + SwiGLU-OAI (alpha=1.702, limit=7.0)",
        example_models=(
            "MiniMaxAI/MiniMax-M3.1-Flash",
            "MiniMaxAI/MiniMax-M3",
        ),
        notes="Native (up + 1) * gate * sigmoid(1.702 * gate) clamped SwiGLU across CUDA, CPU SIMD & MLX.",
    ),
    MoEFamilySpec(
        family="LiquidAI LFM2.5-MoE (Oct 2026)",
        model_types=("lfm2_moe",),
        attention_kind="Hybrid LIV ShortConv + QK-Layernorm GQA",
        routing_kind="Bias-Balanced Top-K Sparse MoE (num_dense_layers prefix)",
        example_models=(
            "LiquidAI/LFM2.5-24B-A2B",
            "LiquidAI/LFM2.5-8B-A1B",
        ),
        notes="Ultra-fast 2026 laptop/edge hybrid MoE supported natively on both MLX Metal and CUDA/CPU.",
    ),
    MoEFamilySpec(
        family="Ling-3.2 / Bailing-MoE-V3 (Nov 2026)",
        model_types=("bailing_hybrid", "bailing_moe_v3", "bailing_moe_v3_vl"),
        attention_kind="Hybrid KDA Linear Attention + MLA",
        routing_kind="Bias-Balanced Sigmoid Top-K + Shared Experts",
        example_models=(
            "inclusionAI/Ling-3.2-air",
            "inclusionAI/Ling-3.1-flash",
        ),
        notes="Preserves dense prefix layers (first_k_dense_replace) and streams 512-expert routed layers.",
    ),
    MoEFamilySpec(
        family="Llama 4 / Qwen3-MoE / ERNIE-4.5 / Legacy MoEs",
        model_types=(
            "llama4", "llama4_text", "qwen3_moe", "qwen3_vl_moe", "ernie4_5_moe",
            "deepseek_v2", "granitemoehybrid", "granitemoe", "jamba", "olmoe", "mixtral", "phimoe",
        ),
        attention_kind="Chunked iRoPE GQA / Hybrid Mamba2 / MLA",
        routing_kind="Sigmoid or Softmax Top-K",
        example_models=(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Qwen/Qwen3-235B-A22B-FP8",
            "baidu/ERNIE-4.5-21B-A3B-PT",
        ),
        notes="Full backwards compatibility with 2025 architectures alongside 2026 frontier models.",
    ),
)

ROADMAP_MOE_FAMILIES: Tuple[Dict[str, str], ...] = (
    {
        "family": "DeepSeek-V4.1-Flash w/ 196B Engram (engram_layer_ids)",
        "model_type": "deepseek_v41",
        "status": "roadmap",
        "reason": "Standard DeepSeek-V4 (no engram_layer_ids) works today; V4.1 Engram layers require 185 GiB mmap hashed n-gram embedding tables + 20+20 CED encoder-decoder.",
    },
    {
        "family": "Thinking Machines Inkling-2-Mini (Looped Softcapped MXFP4 MoE)",
        "model_type": "inkling2_loop",
        "status": "roadmap",
        "reason": "Uses multi-turn looped decoder execution with dual-tanh Inkling2LoopSoftcappedSwiGLU and packed MXFP4 loop weights.",
    },
    {
        "family": "OpenAI gpt-oss-20b / gpt-oss-120b (Biased MXFP4 Experts)",
        "model_type": "gpt_oss",
        "status": "roadmap",
        "reason": "Uses per-expert linear biases (gate_up_proj_bias / down_proj_bias) not stored in bias-free DEIMIDX2 Q4 leaves.",
    },
)


def list_supported_moe_families() -> Dict[str, object]:
    """Returns structured dictionary of verified 2026 MoE families and roadmap items."""
    return {
        "supported": [s.as_dict() for s in SUPPORTED_MOE_FAMILIES],
        "roadmap": list(ROADMAP_MOE_FAMILIES),
    }
