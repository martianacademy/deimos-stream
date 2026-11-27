// ============================================================================
// DeimosStream (`DEIMIDX2`) Universal Dynamic Merkle & Zero-GIL UMA DMA Core
// (`src/deimos_stream/native/deimos_merkle.mm`)
// ============================================================================
// Implements:
//   1. Dynamic `DEIMIDX2` binary mmap loader supporting arbitrary `(L, E, V)`
//      without fixed leaf-count or vocabulary-size limits.
//   2. Tier-1 startup flat XOR Merkle tree verification (`O(tree_leaf_cap)`)
//      and Tier-2 ARMv8 `CC_SHA256` byte verification on cold expert loads.
//   3. Batched zero-GIL `deimos_stage_q4_batch` copying quantized expert
//      sub-tensors (`gu_w, gu_s, gu_b, dn_w, dn_s, dn_b`) from mmap or physical
//      uncached NVMe (`F_NOCACHE` pread) straight into MLX Unified Memory slots.
// ============================================================================

#include "deimos_stream.h"
#include "deimos_sha256.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

inline void sha256_pair_deimos(const uint8_t* left32, const uint8_t* right32, uint8_t* out32) {
    uint8_t buf[65];
    buf[0] = 0x01;
    std::memcpy(buf + 1, left32, 32);
    std::memcpy(buf + 33, right32, 32);
    deimos_sha256_bytes(buf, 65, out32);
}

inline bool is_power_of_two(uint32_t x) {
    return x > 0u && ((x & (x - 1u)) == 0u);
}

inline bool is_valid_magic(const char magic[8]) {
    return (std::memcmp(magic, "DEIMIDX2", 8) == 0) ||
           (std::memcmp(magic, "SYNPIDX2", 8) == 0);
}

struct DeimosContext {
    int      idx_fd         = -1;
    int      cas_fd         = -1;
    int      cas_nocache_fd = -1;
    size_t   idx_size       = 0;
    size_t   cas_size       = 0;
    uint8_t* idx_mmap       = nullptr;
    uint8_t* cas_mmap       = nullptr;

    DeimosHeader*      header   = nullptr;
    uint8_t*           xor_tree = nullptr;
    DeimosExpertEntry* experts  = nullptr;
    DeimosKVSlot*      kv_table = nullptr;
    std::unique_ptr<std::atomic<uint8_t>[]> verified_ram;

    std::atomic<uint64_t> xor_proofs{0};
    std::atomic<uint64_t> byte_proofs{0};
    std::atomic<uint64_t> dma_stages{0};
    std::atomic<uint64_t> dma_bytes{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> prefetches{0};
    std::atomic<uint64_t> step_clock{0};

    ~DeimosContext() {
        if (cas_mmap && cas_mmap != MAP_FAILED) munmap(cas_mmap, cas_size);
        if (idx_mmap && idx_mmap != MAP_FAILED) munmap(idx_mmap, idx_size);
        if (cas_nocache_fd >= 0) close(cas_nocache_fd);
        if (cas_fd >= 0) close(cas_fd);
        if (idx_fd >= 0) close(idx_fd);
    }
};

int verify_tier1_dynamic(DeimosContext* ctx) {
    const DeimosHeader* h = ctx->header;
    if (!h || !is_valid_magic(h->magic) || h->version != 2u) {
        return DEIMOS_ERR_MERKLE_CORRUPT;
    }
    const uint64_t total_experts = static_cast<uint64_t>(h->num_layers) * static_cast<uint64_t>(h->num_experts);
    if (total_experts == 0u || h->tree_leaf_cap < total_experts || !is_power_of_two(h->tree_leaf_cap)) {
        return DEIMOS_ERR_MERKLE_CORRUPT;
    }
    if (h->total_idx_bytes > ctx->idx_size) {
        return DEIMOS_ERR_BOUNDS;
    }

    uint8_t left_b[32], right_b[32], model_root[32];
    sha256_pair_deimos(h->embedding_root, h->shared_core_root, left_b);
    sha256_pair_deimos(h->expert_tree_root, h->output_head_root, right_b);
    sha256_pair_deimos(left_b, right_b, model_root);
    const uint8_t* expected_model_root = (h->num_cow_leaves > 0u) ? h->active_model_root : h->base_model_root;
    if (std::memcmp(model_root, expected_model_root, 32) != 0) {
        return DEIMOS_ERR_MERKLE_CORRUPT;
    }

    const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uint32_t cap = h->tree_leaf_cap;
    for (uint64_t slot = 0; slot < total_experts; ++slot) {
        const DeimosExpertEntry& e = ctx->experts[slot];
        if (std::memcmp(e.sha256, ctx->xor_tree + ((cap + slot) * 32u), 32) != 0) {
            return DEIMOS_ERR_MERKLE_CORRUPT;
        }
        if ((e.byte_offset % page_sz) != 0u || (e.byte_length % page_sz) != 0u) {
            return DEIMOS_ERR_ALIGNMENT;
        }
        if ((e.flags & DEIMOS_FLAG_IS_COW) == 0u && ctx->cas_size > 0) {
            if (static_cast<size_t>(e.byte_offset) + static_cast<size_t>(e.byte_length) > ctx->cas_size) {
                return DEIMOS_ERR_BOUNDS;
            }
        }
    }

    uint8_t parent_h[32];
    for (uint32_t idx = cap - 1u; idx >= 1u; --idx) {
        const uint8_t* lc = ctx->xor_tree + ((2u * idx) * 32u);
        const uint8_t* rc = ctx->xor_tree + ((2u * idx + 1u) * 32u);
        sha256_pair_deimos(lc, rc, parent_h);
        if (std::memcmp(parent_h, ctx->xor_tree + (idx * 32u), 32) != 0) {
            return DEIMOS_ERR_MERKLE_CORRUPT;
        }
    }
    if (std::memcmp(ctx->xor_tree + 32u, h->expert_tree_root, 32) != 0) {
        return DEIMOS_ERR_MERKLE_CORRUPT;
    }

    ctx->xor_proofs.fetch_add(total_experts, std::memory_order_relaxed);
    return DEIMOS_OK;
}

int verify_single_leaf_internal(
    DeimosContext* ctx,
    uint32_t       slot,
    const uint8_t* leaf_bytes_ptr,
    bool           verify_bytes
) {
    const DeimosExpertEntry& e = ctx->experts[slot];
    uint8_t curr_hash[32];

    if (verify_bytes) {
        if (!leaf_bytes_ptr) return DEIMOS_ERR_BOUNDS;
        deimos_sha256_bytes(leaf_bytes_ptr, static_cast<size_t>(e.byte_length), curr_hash);
        if (std::memcmp(curr_hash, e.sha256, 32) != 0) {
            if (ctx->verified_ram) ctx->verified_ram[slot].store(0u, std::memory_order_relaxed);
            return DEIMOS_ERR_MERKLE_CORRUPT;
        }
        ctx->byte_proofs.fetch_add(1, std::memory_order_relaxed);
    } else {
        std::memcpy(curr_hash, e.sha256, 32);
    }

    uint32_t node = ctx->header->tree_leaf_cap + slot;
    if (std::memcmp(ctx->xor_tree + (node * 32u), curr_hash, 32) != 0) {
        if (ctx->verified_ram) ctx->verified_ram[slot].store(0u, std::memory_order_relaxed);
        return DEIMOS_ERR_MERKLE_CORRUPT;
    }
    while (node > 1u) {
        uint32_t sib = node ^ 1u;
        const uint8_t* sib_h = ctx->xor_tree + (sib * 32u);
        if ((node & 1u) == 0u) sha256_pair_deimos(curr_hash, sib_h, curr_hash);
        else sha256_pair_deimos(sib_h, curr_hash, curr_hash);
        node >>= 1u;
    }
    if (std::memcmp(curr_hash, ctx->header->expert_tree_root, 32) != 0) {
        if (ctx->verified_ram) ctx->verified_ram[slot].store(0u, std::memory_order_relaxed);
        return DEIMOS_ERR_MERKLE_CORRUPT;
    }
    if (verify_bytes && ctx->verified_ram) {
        ctx->verified_ram[slot].store(1u, std::memory_order_release);
    }
    ctx->xor_proofs.fetch_add(1, std::memory_order_relaxed);
    return DEIMOS_OK;
}

} // namespace

extern "C" {

void* deimos_open(const char* store_dir_cstr, const char* cas_filename_cstr, int enable_nocache_fd) {
    if (!store_dir_cstr || !cas_filename_cstr) return nullptr;
    std::string dir(store_dir_cstr);
    std::string idx_path = dir + "/merkle_tree.idx";
    std::string cas_path = dir + "/" + std::string(cas_filename_cstr);

    int idx_fd = open(idx_path.c_str(), O_RDWR);
    if (idx_fd < 0) return nullptr;
    struct stat st_idx{};
    if (fstat(idx_fd, &st_idx) != 0 || static_cast<size_t>(st_idx.st_size) < sizeof(DeimosHeader)) {
        close(idx_fd);
        return nullptr;
    }
    int cas_fd = open(cas_path.c_str(), O_RDONLY);
    struct stat st_cas{};
    if (cas_fd >= 0) fstat(cas_fd, &st_cas);

    void* idx_map = mmap(nullptr, st_idx.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, idx_fd, 0);
    if (idx_map == MAP_FAILED) {
        close(idx_fd);
        if (cas_fd >= 0) close(cas_fd);
        return nullptr;
    }
    void* cas_map = (cas_fd >= 0 && st_cas.st_size > 0)
                        ? mmap(nullptr, st_cas.st_size, PROT_READ, MAP_PRIVATE, cas_fd, 0)
                        : MAP_FAILED;

    auto* ctx     = new DeimosContext();
    ctx->idx_fd   = idx_fd;
    ctx->cas_fd   = cas_fd;
    ctx->idx_size = static_cast<size_t>(st_idx.st_size);
    ctx->cas_size = (cas_map != MAP_FAILED) ? static_cast<size_t>(st_cas.st_size) : 0;
    ctx->idx_mmap = static_cast<uint8_t*>(idx_map);
    ctx->cas_mmap = (cas_map != MAP_FAILED) ? static_cast<uint8_t*>(cas_map) : nullptr;

    ctx->header   = reinterpret_cast<DeimosHeader*>(ctx->idx_mmap);
    ctx->xor_tree = ctx->idx_mmap + ctx->header->offset_xor_tree;
    ctx->experts  = reinterpret_cast<DeimosExpertEntry*>(ctx->idx_mmap + ctx->header->offset_expert_table);
    ctx->kv_table = (ctx->header->kv_table_slots > 0u && ctx->header->offset_kv_table + ctx->header->kv_table_slots * sizeof(DeimosKVSlot) <= ctx->idx_size)
                        ? reinterpret_cast<DeimosKVSlot*>(ctx->idx_mmap + ctx->header->offset_kv_table)
                        : nullptr;

    if (verify_tier1_dynamic(ctx) != DEIMOS_OK) { delete ctx; return nullptr; }
    const size_t total_exp = static_cast<size_t>(ctx->header->num_layers) * static_cast<size_t>(ctx->header->num_experts);
    ctx->verified_ram = std::make_unique<std::atomic<uint8_t>[]>(total_exp);
    for (size_t s = 0; s < total_exp; ++s) {
        ctx->verified_ram[s].store((ctx->experts[s].flags & DEIMOS_FLAG_VERIFIED) ? 1u : 0u, std::memory_order_relaxed);
    }
    if (ctx->cas_mmap) posix_madvise(ctx->cas_mmap, ctx->cas_size, POSIX_MADV_NORMAL);
    if (enable_nocache_fd && cas_fd >= 0) {
        int nc_fd = open(cas_path.c_str(), O_RDONLY);
        if (nc_fd >= 0) {
#ifdef F_NOCACHE
            fcntl(nc_fd, F_NOCACHE, 1);
#endif
            ctx->cas_nocache_fd = nc_fd;
        }
    }
    return static_cast<void*>(ctx);
}

void deimos_close(void* handle) {
    if (handle) delete static_cast<DeimosContext*>(handle);
}

int deimos_get_header(void* handle, DeimosHeader* out_header) {
    if (!handle || !out_header) return DEIMOS_ERR_INVALID_ARG;
    std::memcpy(out_header, static_cast<DeimosContext*>(handle)->header, sizeof(DeimosHeader));
    return DEIMOS_OK;
}

int deimos_verify_leaf(void* handle, int layer_idx, int expert_idx, int verify_bytes) {
    if (!handle) return DEIMOS_ERR_INVALID_ARG;
    auto* ctx = static_cast<DeimosContext*>(handle);
    const int n_l = static_cast<int>(ctx->header->num_layers);
    const int n_e = static_cast<int>(ctx->header->num_experts);
    if (layer_idx < 0 || layer_idx >= n_l || expert_idx < 0 || expert_idx >= n_e) return DEIMOS_ERR_INVALID_ARG;
    const uint32_t slot = static_cast<uint32_t>(layer_idx) * static_cast<uint32_t>(n_e) + static_cast<uint32_t>(expert_idx);
    const DeimosExpertEntry& e = ctx->experts[slot];
    const uint8_t* ptr = nullptr;
    if (verify_bytes) {
        if (!ctx->cas_mmap || static_cast<size_t>(e.byte_offset) + static_cast<size_t>(e.byte_length) > ctx->cas_size) return DEIMOS_ERR_BOUNDS;
        ptr = ctx->cas_mmap + e.byte_offset;
    }
    return verify_single_leaf_internal(ctx, slot, ptr, verify_bytes != 0);
}

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
) {
    if (!handle || !expert_ids || !phys_slots || count < 0) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    if (count == 0) return DEIMOS_OK;

    auto* ctx = static_cast<DeimosContext*>(handle);
    const int n_l = static_cast<int>(ctx->header->num_layers);
    const int n_e = static_cast<int>(ctx->header->num_experts);
    if (layer_idx < 0 || layer_idx >= n_l) return DEIMOS_ERR_INVALID_ARG;

    const size_t payload_req = gu_w_bytes + 2u * gu_s_bytes + dn_w_bytes + 2u * dn_s_bytes;
    if (payload_req > ctx->header->expert_payload_bytes) {
        return DEIMOS_ERR_BOUNDS;
    }

    for (int i = 0; i < count; ++i) {
        const int e_idx = expert_ids[i];
        const int p_slot = phys_slots[i];
        if (e_idx < 0 || e_idx >= n_e || p_slot < 0) {
            return DEIMOS_ERR_INVALID_ARG;
        }
        const uint32_t slot = static_cast<uint32_t>(layer_idx) * static_cast<uint32_t>(n_e) + static_cast<uint32_t>(e_idx);
        const DeimosExpertEntry& entry = ctx->experts[slot];

        if (static_cast<size_t>(entry.byte_offset) + static_cast<size_t>(entry.byte_length) > ctx->cas_size) {
            return DEIMOS_ERR_BOUNDS;
        }

        const uint8_t* src = nullptr;
        thread_local std::vector<uint8_t> tl_pread_scratch;
        if (use_nocache_pread && ctx->cas_nocache_fd >= 0) {
            if (tl_pread_scratch.size() < entry.byte_length) {
                tl_pread_scratch.resize(entry.byte_length);
            }
            ssize_t nread = pread(ctx->cas_nocache_fd, tl_pread_scratch.data(), entry.byte_length, static_cast<off_t>(entry.byte_offset));
            if (nread != static_cast<ssize_t>(entry.byte_length)) {
                return DEIMOS_ERR_IO;
            }
            src = tl_pread_scratch.data();
        } else {
            if (!ctx->cas_mmap) return DEIMOS_ERR_BOUNDS;
            src = ctx->cas_mmap + entry.byte_offset;
        }

        if (verify_on_cold && ctx->verified_ram[slot].load(std::memory_order_acquire) == 0u) {
            int vrc = verify_single_leaf_internal(ctx, slot, src, true);
            if (vrc != DEIMOS_OK) return vrc;
        }

        const size_t ps = static_cast<size_t>(p_slot);
        size_t off = 0;
        std::memcpy(pool_gu_w + ps * gu_w_bytes, src + off, gu_w_bytes); off += gu_w_bytes;
        std::memcpy(pool_gu_s + ps * gu_s_bytes, src + off, gu_s_bytes); off += gu_s_bytes;
        std::memcpy(pool_gu_b + ps * gu_s_bytes, src + off, gu_s_bytes); off += gu_s_bytes;
        std::memcpy(pool_dn_w + ps * dn_w_bytes, src + off, dn_w_bytes); off += dn_w_bytes;
        std::memcpy(pool_dn_s + ps * dn_s_bytes, src + off, dn_s_bytes); off += dn_s_bytes;
        std::memcpy(pool_dn_b + ps * dn_s_bytes, src + off, dn_s_bytes);

        ctx->dma_stages.fetch_add(1, std::memory_order_relaxed);
        ctx->dma_bytes.fetch_add(payload_req, std::memory_order_relaxed);
    }
    return DEIMOS_OK;
}

const uint8_t* deimos_get_expert_leaf_ptr(
    void* handle, int layer_idx, int expert_idx, int verify_on_cold, int* out_rc
) {
    if (out_rc) *out_rc = DEIMOS_OK;
    if (!handle) { if (out_rc) *out_rc = DEIMOS_ERR_INVALID_ARG; return nullptr; }
    auto* ctx = static_cast<DeimosContext*>(handle);
    const int n_l = static_cast<int>(ctx->header->num_layers);
    const int n_e = static_cast<int>(ctx->header->num_experts);
    if (layer_idx < 0 || layer_idx >= n_l || expert_idx < 0 || expert_idx >= n_e || !ctx->cas_mmap) {
        if (out_rc) *out_rc = DEIMOS_ERR_BOUNDS;
        return nullptr;
    }
    const uint32_t slot = static_cast<uint32_t>(layer_idx) * static_cast<uint32_t>(n_e) + static_cast<uint32_t>(expert_idx);
    const DeimosExpertEntry& entry = ctx->experts[slot];
    if (static_cast<size_t>(entry.byte_offset) + static_cast<size_t>(entry.byte_length) > ctx->cas_size) {
        if (out_rc) *out_rc = DEIMOS_ERR_BOUNDS;
        return nullptr;
    }
    const uint8_t* src = ctx->cas_mmap + entry.byte_offset;
    if (verify_on_cold && ctx->verified_ram[slot].load(std::memory_order_acquire) == 0u) {
        int vrc = verify_single_leaf_internal(ctx, slot, src, true);
        if (vrc != DEIMOS_OK) { if (out_rc) *out_rc = vrc; return nullptr; }
    }
    return src;
}

int deimos_evict_expert(void* handle, int layer_idx, int expert_idx) {
    if (!handle) return DEIMOS_ERR_INVALID_ARG;
    auto* ctx = static_cast<DeimosContext*>(handle);
    const int n_l = static_cast<int>(ctx->header->num_layers);
    const int n_e = static_cast<int>(ctx->header->num_experts);
    if (layer_idx < 0 || layer_idx >= n_l || expert_idx < 0 || expert_idx >= n_e || !ctx->cas_mmap) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    const uint32_t slot = static_cast<uint32_t>(layer_idx) * static_cast<uint32_t>(n_e) + static_cast<uint32_t>(expert_idx);
    const DeimosExpertEntry& e = ctx->experts[slot];
    madvise(ctx->cas_mmap + e.byte_offset, e.byte_length, MADV_DONTNEED);
    ctx->evictions.fetch_add(1, std::memory_order_relaxed);
    return DEIMOS_OK;
}

void deimos_prefetch_layer(void* handle, int layer_idx, const int32_t* expert_ids, int count) {
    if (!handle || !expert_ids || count <= 0) return;
    auto* ctx = static_cast<DeimosContext*>(handle);
    if (!ctx->cas_mmap) return;
    const int n_l = static_cast<int>(ctx->header->num_layers);
    const int n_e = static_cast<int>(ctx->header->num_experts);
    if (layer_idx < 0 || layer_idx >= n_l) return;
    for (int i = 0; i < count; ++i) {
        int e_idx = expert_ids[i];
        if (e_idx >= 0 && e_idx < n_e) {
            uint32_t slot = static_cast<uint32_t>(layer_idx) * static_cast<uint32_t>(n_e) + static_cast<uint32_t>(e_idx);
            const DeimosExpertEntry& e = ctx->experts[slot];
            posix_madvise(ctx->cas_mmap + e.byte_offset, e.byte_length, POSIX_MADV_WILLNEED);
            ctx->prefetches.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void deimos_get_stats(void* handle, DeimosStats* out_stats) {
    if (!handle || !out_stats) return;
    auto* ctx = static_cast<DeimosContext*>(handle);
    out_stats->xor_proofs_verified = ctx->xor_proofs.load(std::memory_order_relaxed);
    out_stats->byte_proofs_verified = ctx->byte_proofs.load(std::memory_order_relaxed);
    out_stats->dma_slot_stages = ctx->dma_stages.load(std::memory_order_relaxed);
    out_stats->dma_bytes_staged = ctx->dma_bytes.load(std::memory_order_relaxed);
    out_stats->evictions_issued = ctx->evictions.load(std::memory_order_relaxed);
    out_stats->prefetches_issued = ctx->prefetches.load(std::memory_order_relaxed);
}

int deimos_kv_lookup(void* handle, const uint8_t* prefix_hash32, double now_ts) {
    if (!handle || !prefix_hash32) return 0;
    auto* ctx = static_cast<DeimosContext*>(handle);
    if (!ctx->kv_table || ctx->header->kv_table_slots == 0u) return 0;
    const uint32_t cap = ctx->header->kv_table_slots;
    uint64_t h64 = 0;
    std::memcpy(&h64, prefix_hash32, sizeof(uint64_t));
    const uint32_t base = static_cast<uint32_t>(h64 % cap);
    const uint32_t probe_lim = std::min<uint32_t>(64u, cap);
    for (uint32_t d = 0; d < probe_lim; ++d) {
        DeimosKVSlot& sl = ctx->kv_table[(base + d) % cap];
        if (!sl.occupied) continue;
        if (std::memcmp(sl.prefix_hash, prefix_hash32, 32) == 0) {
            sl.last_used_ts = now_ts;
            return 1;
        }
    }
    return 0;
}

int deimos_kv_insert(
    void* handle, const uint8_t* prefix_hash32, const uint8_t* parent_hash32,
    uint32_t token_end, double now_ts, uint8_t* out_evicted_hash32
) {
    if (!handle || !prefix_hash32 || !parent_hash32) return DEIMOS_ERR_INVALID_ARG;
    auto* ctx = static_cast<DeimosContext*>(handle);
    if (!ctx->kv_table || ctx->header->kv_table_slots == 0u) return DEIMOS_ERR_BOUNDS;
    const uint32_t cap = ctx->header->kv_table_slots;
    uint64_t h64 = 0;
    std::memcpy(&h64, prefix_hash32, sizeof(uint64_t));
    const uint32_t base = static_cast<uint32_t>(h64 % cap);
    const uint32_t probe_lim = std::min<uint32_t>(64u, cap);

    int32_t free_idx = -1;
    uint32_t oldest_idx = base;
    double oldest_ts = 1e300;

    for (uint32_t d = 0; d < probe_lim; ++d) {
        uint32_t idx = (base + d) % cap;
        DeimosKVSlot& sl = ctx->kv_table[idx];
        if (sl.occupied && std::memcmp(sl.prefix_hash, prefix_hash32, 32) == 0) {
            sl.last_used_ts = now_ts;
            sl.token_end = token_end;
            return 0;
        }
        if (!sl.occupied && free_idx < 0) free_idx = static_cast<int32_t>(idx);
        if (sl.occupied && sl.last_used_ts < oldest_ts) {
            oldest_ts = sl.last_used_ts;
            oldest_idx = idx;
        }
    }

    int evicted = 0;
    uint32_t target = (free_idx >= 0) ? static_cast<uint32_t>(free_idx) : oldest_idx;
    DeimosKVSlot& dst = ctx->kv_table[target];
    if (dst.occupied && free_idx < 0) {
        if (out_evicted_hash32) std::memcpy(out_evicted_hash32, dst.prefix_hash, 32);
        evicted = 1;
    }
    std::memcpy(dst.prefix_hash, prefix_hash32, 32);
    std::memcpy(dst.parent_hash, parent_hash32, 32);
    dst.occupied = 1u;
    dst.token_end = token_end;
    dst.last_used_ts = now_ts;
    return evicted;
}

} // extern "C"
