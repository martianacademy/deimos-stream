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
from deimos_stream.kv_cache import MerkleKVPrefixCache
from deimos_stream.merkle_builder import (
    PAGE_SIZE,
    build_deimidx2_file,
    compile_deimos_dylib,
    compute_q4_leaf_layout,
    convert_fp16_cas_store_to_q4,
)

__version__ = "0.1.0"


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
    "DeimosMerkleBridge",
    "DeimosMerkleIntegrityError",
    "DeimosPagedMoEEngine",
    "IncrementalStreamingDetokenizer",
    "MerkleKVPrefixCache",
    "PagedExpertArena",
    "__version__",
    "build_deimidx2_file",
    "compile_deimos_dylib",
    "compute_q4_leaf_layout",
    "convert_fp16_cas_store_to_q4",
    "convert_safetensors_moe_to_deimos_q4",
    "create_deimos_app",
    "inspect_store",
    "normalize_moe_config",
]
