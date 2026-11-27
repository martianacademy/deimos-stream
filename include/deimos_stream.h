// ============================================================================
// DeimosStream (`DEIMIDX2`) Universal Dynamic Merkle Index & UMA Pool Stager
// (`include/deimos_stream.h`)
// ============================================================================
// Self-describing binary Merkle store supporting arbitrary `(num_layers,
// num_experts, vocab_size, quant_bits)` Sparse MoE topologies, Tier-1 flat XOR
// tree + Tier-2 ARMv8 `CC_SHA256` leaf verification, Robin-Hood 16-token KV
// prefix indexing, and zero-GIL C++17 Unified Memory DMA staging for Q4/Q8
// PagedExpertArena slots.
// ============================================================================

#ifndef DEIMOS_STREAM_H
#define DEIMOS_STREAM_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

enum DeimosError {
    DEIMOS_OK                 =  0,
    DEIMOS_ERR_INVALID_ARG    = -1,
    DEIMOS_ERR_MERKLE_CORRUPT = -2,
    DEIMOS_ERR_BOUNDS         = -3,
    DEIMOS_ERR_ALIGNMENT      = -4,
    DEIMOS_ERR_IO             = -5
};

enum DeimosLeafFlags {
    DEIMOS_FLAG_IS_COW   = 1u << 0,
    DEIMOS_FLAG_VERIFIED = 1u << 1
};

#pragma pack(push, 1)

struct DeimosHeader {
    char     magic[8];               // "DEIMIDX2" (also accepts v3 "SYNPIDX2")
    uint32_t version;                // 2
    uint32_t num_layers;             // Arbitrary L (e.g. 16, 48, 64, 78)
    uint32_t num_experts;            // Arbitrary E per layer (e.g. 8, 60, 64, 128, 256)
    uint32_t tree_leaf_cap;          // Next power-of-2 >= (num_layers * num_experts)
    uint32_t vocab_size;             // Arbitrary V (e.g. 50304, 151936, 248320)
    uint32_t hidden_size;            // Model hidden dim D
    uint32_t intermediate_size;      // Expert SwiGLU intermediate dim I
    uint32_t quant_bits;             // 4, 8, or 16
    uint32_t quant_group_size;       // e.g. 64 (0 for FP16)
    uint32_t num_cow_leaves;         // Active CoW delta leaf count

    uint64_t expert_stride_bytes;    // 16 KB page-aligned byte stride in CAS file
    uint64_t expert_payload_bytes;   // Exact packed tensor payload bytes <= stride
    uint64_t offset_xor_tree;        // Byte offset to 2 * tree_leaf_cap * 32B nodes
    uint64_t offset_expert_table;    // Byte offset to (L * E) * 64B entries
    uint64_t offset_token_table;     // Byte offset to V * 40B entries (0 if omitted)
    uint64_t offset_kv_table;        // Byte offset to kv_slots * 80B entries
    uint64_t total_idx_bytes;        // Total merkle_tree.idx byte length
    uint64_t orig_safetensors_bytes; // Original unquantized/source checkpoint bytes

    uint8_t  base_model_root[32];
    uint8_t  active_model_root[32];
    uint8_t  embedding_root[32];
    uint8_t  shared_core_root[32];
    uint8_t  expert_tree_root[32];
    uint8_t  output_head_root[32];

    uint32_t kv_table_slots;         // Number of Robin-Hood KV prefix slots
    uint8_t  reserved[12];           // Pad header to exact 320 bytes (5 x 64B cache lines)
};

struct DeimosExpertEntry {
    uint8_t  sha256[32];
    uint64_t byte_offset;
    uint32_t byte_length;
    uint32_t flags;
    uint64_t access_freq;
    uint64_t last_step;
};

struct DeimosKVSlot {
    uint8_t  prefix_hash[32];
    uint8_t  parent_hash[32];
    uint32_t occupied;
    uint32_t token_end;
    double   last_used_ts;
};

struct DeimosStats {
    uint64_t xor_proofs_verified;
    uint64_t byte_proofs_verified;
    uint64_t dma_slot_stages;
    uint64_t dma_bytes_staged;
    uint64_t evictions_issued;
    uint64_t prefetches_issued;
};

#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(DeimosHeader) == 320, "DeimosHeader must be 320 bytes");
static_assert(sizeof(DeimosExpertEntry) == 64, "DeimosExpertEntry must be 64 bytes");
static_assert(sizeof(DeimosKVSlot) == 80, "DeimosKVSlot must be 80 bytes");
#endif

void* deimos_open(const char* store_dir, const char* cas_filename, int enable_nocache_fd);
void  deimos_close(void* handle);

int deimos_get_header(void* handle, DeimosHeader* out_header);
int deimos_verify_leaf(void* handle, int layer_idx, int expert_idx, int verify_bytes);

int deimos_stage_q4_batch(
    void*          handle,
    int            layer_idx,
    const int32_t* expert_ids,
    const int32_t* phys_slots,
    int            count,
    uint8_t*       pool_gu_w,
    uint8_t*       pool_gu_s,
    uint8_t*       pool_gu_b,
    uint8_t*       pool_dn_w,
    uint8_t*       pool_dn_s,
    uint8_t*       pool_dn_b,
    size_t         gu_w_bytes,
    size_t         gu_s_bytes,
    size_t         dn_w_bytes,
    size_t         dn_s_bytes,
    int            verify_on_cold,
    int            use_nocache_pread
);

int  deimos_evict_expert(void* handle, int layer_idx, int expert_idx);
void deimos_prefetch_layer(void* handle, int layer_idx, const int32_t* expert_ids, int count);
void deimos_get_stats(void* handle, DeimosStats* out_stats);

const uint8_t* deimos_get_expert_leaf_ptr(
    void* handle,
    int   layer_idx,
    int   expert_idx,
    int   verify_on_cold,
    int*  out_rc
);

int deimos_kv_lookup(void* handle, const uint8_t* prefix_hash32, double now_ts);
int deimos_kv_insert(
    void*          handle,
    const uint8_t* prefix_hash32,
    const uint8_t* parent_hash32,
    uint32_t       token_end,
    double         now_ts,
    uint8_t*       out_evicted_hash32
);

// ============================================================================
// Native C++17 LFU+LRU Page Table, Zero-GIL Residual Lookahead Worker Thread &
// Cache-Aware Miss-Only Nucleus Dynamic-K Resolver (`deimos_arena.mm`)
// ============================================================================

struct DeimosArenaTelemetry {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t evictions;
    uint64_t lookahead_submitted;
    uint64_t lookahead_staged;
    uint64_t lookahead_useful_hits;
    uint64_t tail_misses_skipped;
    uint64_t total_experts_evaluated;
};

void* deimos_arena_create(
    void*  merkle_handle,
    int    num_layers,
    int    num_experts,
    int    slots_per_layer,
    size_t gu_w_bytes,
    size_t gu_s_bytes,
    size_t dn_w_bytes,
    size_t dn_s_bytes,
    int    verify_on_cold,
    int    use_nocache_pread
);

void deimos_arena_destroy(void* arena_handle);

int deimos_arena_bind_layer_ptrs(
    void*    arena_handle,
    int      layer_idx,
    uint8_t* pool_gu_w,
    uint8_t* pool_gu_s,
    uint8_t* pool_gu_b,
    uint8_t* pool_dn_w,
    uint8_t* pool_dn_s,
    uint8_t* pool_dn_b
);

int deimos_arena_preload_all(void* arena_handle);
int deimos_arena_mark_preloaded_prefix(void* arena_handle, int count);

int deimos_arena_resolve_and_prefetch(
    void*          arena_handle,
    int            layer_idx,
    const int32_t* routed_ids,
    const float*   routed_scores,
    int            num_tokens,
    int            top_k,
    float          miss_tail_tau,
    int            min_k,
    int            next_layer_idx,
    const int32_t* next_pred_ids,
    int            next_pred_count,
    int32_t*       out_phys_slots,
    float*         out_eff_scores,
    int*           out_eff_k
);

void deimos_arena_wait_idle(void* arena_handle);
int  deimos_arena_get_slot_gens(void* arena_handle, int layer_idx, uint64_t* out_gens, int max_slots);
void deimos_arena_get_telemetry(void* arena_handle, DeimosArenaTelemetry* out_tel);
void deimos_arena_reset_telemetry(void* arena_handle);

int deimos_arena_resolve_ring_batch(
    void*          arena_handle,
    int            layer_idx,
    const int32_t* routed_ids,
    const float*   routed_scores,
    int            num_tokens,
    int            top_k,
    int32_t*       out_phys_slots,
    int32_t*       out_miss_experts,
    int32_t*       out_miss_phys_slots,
    int*           out_num_misses
);

int deimos_arena_resolve_hybrid_decode(
    void*          arena_handle,
    int            layer_idx,
    const int32_t* routed_ids,
    const float*   routed_scores,
    int            top_k,
    float          miss_tail_tau,
    int            min_k,
    int            max_promos,
    int32_t*       out_hit_phys_slots,
    float*         out_hit_scores,
    int*           out_num_hits,
    int32_t*       out_miss_experts,
    float*         out_miss_scores,
    int*           out_num_misses,
    int32_t*       out_promo_experts,
    int32_t*       out_promo_phys_slots,
    int*           out_num_promos
);

int deimos_cpu_q4_swiglu_forward(
    void*          merkle_handle,
    int            layer_idx,
    const int32_t* expert_ids,
    const float*   expert_scores,
    int            num_experts_active,
    const uint16_t* x_f16,
    uint16_t*      out_f16,
    int            hidden_size,
    int            intermediate_size,
    int            group_size,
    int            verify_on_cold
);

void deimos_warm_cpu_moe_pool(
    void* merkle_handle,
    int   num_layers,
    int   start_expert,
    int   end_expert,
    int   hidden_size,
    int   intermediate_size,
    int   group_size
);

#ifdef __cplusplus
}
#endif

#endif // DEIMOS_STREAM_H
