// ============================================================================
// DeimosStream Ring-Buffered Batch & TinyLFU Hybrid Decode Resolvers
// (`src/deimos_stream/native/deimos_arena_hybrid.cpp`)
// ============================================================================

#include "deimos_arena_internal.h"

extern "C" {

int deimos_arena_mark_preloaded_prefix(void* arena_handle, int count) {
    if (!arena_handle || count <= 0) return DEIMOS_ERR_INVALID_ARG;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    const int n = std::min(count, mgr->slots_per_layer);
    for (int l = 0; l < mgr->num_layers; ++l) {
        mgr->free_slots[l].clear();
        for (int s = mgr->slots_per_layer - 1; s >= n; --s) {
            mgr->free_slots[l].push_back(s);
        }
        for (int s = 0; s < n; ++s) {
            mgr->virt_to_phys[l][s] = s;
            mgr->phys_to_virt[l][s] = s;
            mgr->slot_freq[l][s]    = 0;
            mgr->slot_step[l][s]    = 0;
            mgr->slot_gen[l][s]     = ++mgr->gen_clock;
        }
    }
    return DEIMOS_OK;
}

int deimos_arena_resolve_ring_batch(
    void* arena_handle, int layer_idx,
    const int32_t* routed_ids, const float* routed_scores,
    int num_tokens, int top_k,
    int32_t* out_phys_slots, int32_t* out_miss_experts,
    int32_t* out_miss_phys_slots, int* out_num_misses
) {
    (void)routed_scores;
    if (!arena_handle || !routed_ids || !out_phys_slots || !out_miss_experts || !out_miss_phys_slots || !out_num_misses) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    if (layer_idx < 0 || layer_idx >= mgr->num_layers || num_tokens <= 0 || top_k <= 0) return DEIMOS_ERR_INVALID_ARG;
    mgr->wait_idle();
    mgr->tick_decay(layer_idx);

    auto& v2p   = mgr->virt_to_phys[layer_idx];
    auto& freqs = mgr->slot_freq[layer_idx];
    auto& steps = mgr->slot_step[layer_idx];
    auto& efreq = mgr->expert_freq[layer_idx];
    std::vector<uint8_t> protect(static_cast<size_t>(mgr->slots_per_layer), 0);

    const int total = num_tokens * top_k;
    mgr->total_experts_evaluated += static_cast<uint64_t>(total);
    int n_miss = 0;

    for (int i = 0; i < total; ++i) {
        int e = routed_ids[i];
        if (e < 0 || e >= mgr->num_experts) return DEIMOS_ERR_INVALID_ARG;
        efreq[e] += 1u;
        if (mgr->total_expert_hits[layer_idx][e] < 0x3fffffffU) mgr->total_expert_hits[layer_idx][e] += 1u;
        int ps = v2p[e];
        if (ps >= 0) {
            protect[ps] = 1;
            mgr->cache_hits += 1;
        } else {
            mgr->cache_misses += 1;
            bool seen = false;
            for (int m = 0; m < n_miss; ++m) if (out_miss_experts[m] == e) { seen = true; break; }
            if (!seen) {
                if (n_miss >= mgr->slots_per_layer) return DEIMOS_ERR_BOUNDS;
                out_miss_experts[n_miss++] = e;
            }
        }
    }
    for (int m = 0; m < n_miss; ++m) {
        int e = out_miss_experts[m];
        int ps = mgr->allocate_slot_unlocked(layer_idx, e, protect.data(), 0);
        if (ps < 0) return DEIMOS_ERR_BOUNDS;
        protect[ps] = 1;
        out_miss_phys_slots[m] = ps;
    }
    for (int i = 0; i < total; ++i) {
        int e = routed_ids[i];
        int ps = v2p[e];
        out_phys_slots[i] = ps;
        freqs[ps] = static_cast<int32_t>(std::min<uint32_t>(efreq[e], 64u));
        steps[ps] = mgr->step_clock;
    }
    *out_num_misses = n_miss;
    return DEIMOS_OK;
}

int deimos_arena_resolve_hybrid_decode(
    void* arena_handle, int layer_idx,
    const int32_t* routed_ids, const float* routed_scores,
    int top_k, float miss_tail_tau, int min_k, int max_promos,
    int32_t* out_hit_phys_slots, float* out_hit_scores, int* out_num_hits,
    int32_t* out_miss_experts, float* out_miss_scores, int* out_num_misses,
    int32_t* out_promo_experts, int32_t* out_promo_phys_slots, int* out_num_promos
) {
    if (!arena_handle || !routed_ids || !routed_scores || top_k <= 0 || top_k > 32) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    if (layer_idx < 0 || layer_idx >= mgr->num_layers) return DEIMOS_ERR_INVALID_ARG;
    mgr->wait_idle();
    mgr->tick_decay(layer_idx);

    auto& v2p   = mgr->virt_to_phys[layer_idx];
    auto& freqs = mgr->slot_freq[layer_idx];
    auto& steps = mgr->slot_step[layer_idx];
    auto& efreq = mgr->expert_freq[layer_idx];
    auto& frees = mgr->free_slots[layer_idx];

    int32_t work_ids[32];
    float   work_sc[32];
    int eff_k = filter_miss_tail_tau(
        routed_ids, routed_scores, 1, top_k, miss_tail_tau, min_k,
        mgr->num_experts, v2p, work_ids, work_sc, &mgr->tail_misses_skipped
    );
    mgr->total_experts_evaluated += static_cast<uint64_t>(eff_k);

    uint8_t protect_stack[256];
    std::vector<uint8_t> protect_heap;
    uint8_t* protect = protect_stack;
    if (mgr->slots_per_layer > 256) {
        protect_heap.assign(static_cast<size_t>(mgr->slots_per_layer), 0);
        protect = protect_heap.data();
    } else {
        std::memset(protect_stack, 0, static_cast<size_t>(mgr->slots_per_layer));
    }

    int n_hits = 0, n_miss = 0;
    for (int i = 0; i < eff_k; ++i) {
        int e = work_ids[i];
        if (e < 0 || e >= mgr->num_experts) return DEIMOS_ERR_INVALID_ARG;
        if (efreq[e] < 64u) efreq[e] += 1u;
        if (mgr->total_expert_hits[layer_idx][e] < 0x3fffffffU) mgr->total_expert_hits[layer_idx][e] += 1u;
        int ps = v2p[e];
        if (ps >= 0) {
            protect[ps] = 1;
            if (freqs[ps] < 64) freqs[ps] += 2;
            steps[ps] = mgr->step_clock;
            out_hit_phys_slots[n_hits] = ps;
            out_hit_scores[n_hits]     = work_sc[i];
            n_hits += 1;
            mgr->cache_hits += 1;
        } else {
            out_miss_experts[n_miss] = e;
            out_miss_scores[n_miss]  = work_sc[i];
            n_miss += 1;
            mgr->cache_misses += 1;
        }
    }

    int n_promos = 0;
    const bool can_promo_now = (max_promos > 0) && ((mgr->step_clock - mgr->last_promo_step) >= mgr->num_layers);
    const int promo_cap = can_promo_now ? 1 : 0;
    for (int m = 0; m < n_miss && n_promos < promo_cap; ++m) {
        int e = out_miss_experts[m];
        if (v2p[e] >= 0) continue;
        bool allow = (!frees.empty() && efreq[e] >= 3u);
        if (!allow && frees.empty()) {
            int vic = mgr->find_victim_slot(layer_idx, protect);
            if (vic >= 0) {
                if (freqs[vic] == 0 && efreq[e] >= 3u) {
                    allow = true;
                } else if (efreq[e] >= 12u && static_cast<int32_t>(efreq[e]) >= freqs[vic] * 2 + 8) {
                    allow = true;
                }
            }
        }
        if (allow) {
            int ps = mgr->allocate_slot_unlocked(layer_idx, e, protect, static_cast<int32_t>(efreq[e]));
            if (ps >= 0) {
                protect[ps] = 1;
                out_promo_experts[n_promos]    = e;
                out_promo_phys_slots[n_promos] = ps;
                n_promos += 1;
                mgr->last_promo_step = mgr->step_clock;
            }
        }
    }

    *out_num_hits   = n_hits;
    *out_num_misses = n_miss;
    *out_num_promos = n_promos;
    return DEIMOS_OK;
}

int deimos_arena_export_expert_hits(void* arena_handle, uint32_t* out_hits, int max_elements) {
    if (!arena_handle || !out_hits || max_elements <= 0) return DEIMOS_ERR_INVALID_ARG;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    mgr->wait_idle();
    const int total = mgr->num_layers * mgr->num_experts;
    if (max_elements < total) return DEIMOS_ERR_BOUNDS;
    for (int l = 0; l < mgr->num_layers; ++l) {
        std::memcpy(
            out_hits + static_cast<size_t>(l) * static_cast<size_t>(mgr->num_experts),
            mgr->total_expert_hits[l].data(),
            static_cast<size_t>(mgr->num_experts) * sizeof(uint32_t)
        );
    }
    return DEIMOS_OK;
}

int deimos_arena_seed_warm_slots(
    void*           arena_handle,
    const int32_t*  warm_expert_ids,
    const int32_t*  init_slot_freqs,
    const uint32_t* init_expert_freqs,
    int             stage_bound_pools
) {
    if (!arena_handle || !warm_expert_ids || !init_slot_freqs || !init_expert_freqs) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    mgr->wait_idle();

    const int S = mgr->slots_per_layer;
    const int E = mgr->num_experts;
    std::vector<int32_t> slot_seq(static_cast<size_t>(S));
    for (int s = 0; s < S; ++s) slot_seq[s] = s;

    int saved_nc = mgr->use_nocache_pread;
#if defined(__APPLE__)
    if (stage_bound_pools) mgr->use_nocache_pread = 1;
#endif

    for (int l = 0; l < mgr->num_layers; ++l) {
        const int32_t*  w_ids = warm_expert_ids + static_cast<size_t>(l) * static_cast<size_t>(S);
        const int32_t*  s_frq = init_slot_freqs + static_cast<size_t>(l) * static_cast<size_t>(S);
        const uint32_t* e_frq = init_expert_freqs + static_cast<size_t>(l) * static_cast<size_t>(E);

        std::fill(mgr->virt_to_phys[l].begin(), mgr->virt_to_phys[l].end(), -1);
        std::fill(mgr->phys_to_virt[l].begin(), mgr->phys_to_virt[l].end(), -1);
        std::fill(mgr->prefetched_flag[l].begin(), mgr->prefetched_flag[l].end(), 0);
        mgr->free_slots[l].clear();

        for (int e = 0; e < E; ++e) {
            mgr->expert_freq[l][e] = std::min<uint32_t>(e_frq[e], 48u);
        }
        for (int s = 0; s < S; ++s) {
            int e = w_ids[s];
            if (e < 0 || e >= E || mgr->virt_to_phys[l][e] >= 0) {
                mgr->use_nocache_pread = saved_nc;
                return DEIMOS_ERR_INVALID_ARG;
            }
            mgr->virt_to_phys[l][e] = s;
            mgr->phys_to_virt[l][s] = e;
            mgr->slot_freq[l][s]    = std::max<int32_t>(0, std::min<int32_t>(s_frq[s], 48));
            mgr->slot_step[l][s]    = 0;
            mgr->slot_gen[l][s]     = ++mgr->gen_clock;
        }
        if (stage_bound_pools && mgr->ptrs[l].gu_w) {
            int rc = mgr->stage_layer_experts(l, w_ids, slot_seq.data(), S);
            if (rc != DEIMOS_OK) {
                mgr->use_nocache_pread = saved_nc;
                return rc;
            }
        }
    }
    mgr->use_nocache_pread = saved_nc;
    return DEIMOS_OK;
}

} // extern "C"
