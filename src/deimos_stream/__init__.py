"""
DeimosStream — Universal Low-VRAM Sparse MoE & LLM Merkle-Paged Streaming Engine
================================================================================
"""

from deimos_stream.arena import (
    ArenaStats,
    DeimosMerkleBridge,
    DeimosMerkleIntegrityError,
    PagedExpertArena,
)
from deimos_stream.converter import (
    convert_safetensors_moe_to_deimos_q4,
    normalize_moe_config,
)
from deimos_stream.detokenizer import IncrementalStreamingDetokenizer
from deimos_stream.engine import BenchmarkMetrics, DeimosPagedMoEEngine
from deimos_stream.factory import create_paged_moe_engine, resolve_backend
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine
from deimos_stream.heat_profile import compute_warm_slot_plan, load_heat_counts
from deimos_stream.kv_cache import MerkleKVPrefixCache
from deimos_stream.mtp_drafter import attach_mtp_drafter_if_available
from deimos_stream.mtp_store import discover_mtp_spec
from deimos_stream.speculative import SuffixNgramDrafter, iter_speculative_tokens
from deimos_stream.mlx_hybrid import DeimosMLXHybridMoEEngine
from deimos_stream.mlx_universal import DeimosMLXUniversalMoEEngine
from deimos_stream.model_adapters import (
    SUPPORTED_MOE_FAMILIES,
    list_supported_moe_families,
)
from deimos_stream.merkle_builder import (
    PAGE_SIZE,
    build_deimidx2_file,
    compile_deimos_dylib,
    compute_q4_leaf_layout,
    convert_fp16_cas_store_to_q4,
)
from deimos_stream.torch_arena import TorchPagedExpertArena
from deimos_stream.torch_engine import TorchDeimosPagedMoEEngine

__version__ = "0.10.0"


def __getattr__(name: str) -> object:
    if name in {"DeimosChatSession", "inspect_store"}:
        from deimos_stream import cli

        return getattr(cli, name)
    if name == "create_deimos_app":
        from deimos_stream.server import create_deimos_app

        return create_deimos_app
    raise AttributeError(f"module 'deimos_stream' has no attribute {name!r}")


__all__ = [
    "PAGE_SIZE",
    "ArenaStats",
    "BenchmarkMetrics",
    "DeimosChatSession",
    "DeimosHFHybridMoEEngine",
    "DeimosMLXHybridMoEEngine",
    "DeimosMLXUniversalMoEEngine",
    "DeimosMerkleBridge",
    "DeimosMerkleIntegrityError",
    "DeimosPagedMoEEngine",
    "IncrementalStreamingDetokenizer",
    "MerkleKVPrefixCache",
    "PagedExpertArena",
    "SUPPORTED_MOE_FAMILIES",
    "SuffixNgramDrafter",
    "TorchDeimosPagedMoEEngine",
    "TorchPagedExpertArena",
    "__version__",
    "attach_mtp_drafter_if_available",
    "build_deimidx2_file",
    "compile_deimos_dylib",
    "compute_q4_leaf_layout",
    "compute_warm_slot_plan",
    "convert_fp16_cas_store_to_q4",
    "convert_hf_hybrid_moe_to_deimos",
    "convert_safetensors_moe_to_deimos_q4",
    "create_deimos_app",
    "create_paged_moe_engine",
    "discover_mtp_spec",
    "inspect_store",
    "iter_speculative_tokens",
    "list_supported_moe_families",
    "load_heat_counts",
    "normalize_moe_config",
    "resolve_backend",
]
