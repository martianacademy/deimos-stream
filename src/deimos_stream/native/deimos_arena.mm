// ============================================================================
// DeimosStream Native C++17 Page Table & UMA Lookahead Stager (`deimos_arena.mm`)
// ============================================================================

#include "deimos_arena_internal.h"

extern "C" {

void* deimos_arena_create(
    void* merkle_handle, int num_layers, int num_experts, int slots_per_layer,
    size_t gu_w_bytes, size_t gu_s_bytes, size_t dn_w_bytes, size_t dn_s_bytes,
    int verify_on_cold, int use_nocache_pread
) {
    if (!merkle_handle || num_layers <= 0 || num_experts <= 0 || slots_per_layer <= 0) return nullptr;
    auto* mgr = new DeimosArenaManager();
    mgr->merkle_handle     = merkle_handle;
    mgr->num_layers        = num_layers;
    mgr->num_experts       = num_experts;
    mgr->slots_per_layer   = std::min(num_experts, slots_per_layer);
    mgr->gu_w_bytes        = gu_w_bytes;
    mgr->gu_s_bytes        = gu_s_bytes;
    mgr->dn_w_bytes        = dn_w_bytes;
    mgr->dn_s_bytes        = dn_s_bytes;
    mgr->verify_on_cold    = verify_on_cold;
    mgr->use_nocache_pread = use_nocache_pread;

    const size_t nL = static_cast<size_t>(num_layers);
    const size_t nE = static_cast<size_t>(num_experts);
    const size_t nC = static_cast<size_t>(mgr->slots_per_layer);
    mgr->ptrs.resize(nL);
    mgr->virt_to_phys.assign(nL, std::vector<int32_t>(nE, -1));
    mgr->phys_to_virt.assign(nL, std::vector<int32_t>(nC, -1));
    mgr->slot_freq.assign(nL, std::vector<int32_t>(nC, 0));
    mgr->expert_freq.assign(nL, std::vector<uint32_t>(nE, 0));
    mgr->slot_step.assign(nL, std::vector<int64_t>(nC, 0));
    mgr->slot_gen.assign(nL, std::vector<uint64_t>(nC, 0));
    mgr->prefetched_flag.assign(nL, std::vector<uint8_t>(nE, 0));
    mgr->free_slots.resize(nL);
    for (int l = 0; l < num_layers; ++l) {
        for (int s = mgr->slots_per_layer - 1; s >= 0; --s) mgr->free_slots[l].push_back(s);
    }
    mgr->worker = std::thread(&DeimosArenaManager::worker_loop, mgr);
    return static_cast<void*>(mgr);
}

void deimos_arena_destroy(void* arena_handle) {
    if (arena_handle) delete static_cast<DeimosArenaManager*>(arena_handle);
}

int deimos_arena_bind_layer_ptrs(
    void* arena_handle, int layer_idx,
    uint8_t* gu_w, uint8_t* gu_s, uint8_t* gu_b,
    uint8_t* dn_w, uint8_t* dn_s, uint8_t* dn_b
) {
    if (!arena_handle) return DEIMOS_ERR_INVALID_ARG;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    if (layer_idx < 0 || layer_idx >= mgr->num_layers) return DEIMOS_ERR_INVALID_ARG;
    mgr->ptrs[layer_idx] = LayerPoolPtrs{gu_w, gu_s, gu_b, dn_w, dn_s, dn_b};
    return DEIMOS_OK;
}

int deimos_arena_preload_all(void* arena_handle) {
    if (!arena_handle) return DEIMOS_ERR_INVALID_ARG;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    if (mgr->slots_per_layer < mgr->num_experts) return DEIMOS_ERR_INVALID_ARG;
    std::vector<int32_t> all_ids(static_cast<size_t>(mgr->num_experts));
    for (int e = 0; e < mgr->num_experts; ++e) all_ids[e] = e;
    int saved_nc = mgr->use_nocache_pread;
#if defined(__APPLE__)
    mgr->use_nocache_pread = 1;
#else
    mgr->use_nocache_pread = 0;
#endif
    for (int l = 0; l < mgr->num_layers; ++l) {
        if (mgr->ptrs[l].gu_w) {
            int rc = mgr->stage_layer_experts(l, all_ids.data(), all_ids.data(), mgr->num_experts);
            if (rc != DEIMOS_OK) { mgr->use_nocache_pread = saved_nc; return rc; }
        }
        mgr->virt_to_phys[l] = all_ids;
        mgr->phys_to_virt[l] = all_ids;
        for (int s = 0; s < mgr->slots_per_layer; ++s) mgr->slot_gen[l][s] = ++mgr->gen_clock;
        mgr->free_slots[l].clear();
    }
    mgr->use_nocache_pread = saved_nc;
    return DEIMOS_OK;
}

int deimos_arena_resolve_and_prefetch(
    void* arena_handle, int layer_idx,
    const int32_t* routed_ids, const float* routed_scores,
    int num_tokens, int top_k, float miss_tail_tau, int min_k,
    int next_layer_idx, const int32_t* next_pred_ids, int next_pred_count,
    int32_t* out_phys_slots, float* out_eff_scores, int* out_eff_k
) {
    if (!arena_handle || !routed_ids || !routed_scores || !out_phys_slots || !out_eff_scores || !out_eff_k) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    if (layer_idx < 0 || layer_idx >= mgr->num_layers || num_tokens <= 0 || top_k <= 0) {
        return DEIMOS_ERR_INVALID_ARG;
    }
    mgr->wait_idle();
    if (mgr->last_err != DEIMOS_OK) { int err = mgr->last_err; mgr->last_err = DEIMOS_OK; return err; }
    mgr->tick_decay(layer_idx);

    auto& v2p   = mgr->virt_to_phys[layer_idx];
    auto& freqs = mgr->slot_freq[layer_idx];
    auto& steps = mgr->slot_step[layer_idx];
    auto& pflag = mgr->prefetched_flag[layer_idx];

    std::vector<uint8_t> protect(static_cast<size_t>(mgr->slots_per_layer), 0);
    std::vector<int32_t> work_ids(static_cast<size_t>(num_tokens * top_k));
    std::vector<float>   work_sc(static_cast<size_t>(num_tokens * top_k));
    int eff_k = filter_miss_tail_tau(
        routed_ids, routed_scores, num_tokens, top_k, miss_tail_tau, min_k,
        mgr->num_experts, v2p, work_ids.data(), work_sc.data(), &mgr->tail_misses_skipped
    );

    const int total_eval = num_tokens * eff_k;
    mgr->total_experts_evaluated += static_cast<uint64_t>(total_eval);

    std::vector<int32_t> miss_experts;
    for (int i = 0; i < total_eval; ++i) {
        int e = work_ids[i];
        if (e < 0 || e >= mgr->num_experts) return DEIMOS_ERR_INVALID_ARG;
        int ps = v2p[e];
        if (ps >= 0) {
            protect[ps] = 1;
            mgr->cache_hits += 1;
            if (pflag[e]) { mgr->lookahead_useful_hits += 1; pflag[e] = 0; }
        } else {
            mgr->cache_misses += 1;
            if (std::find(miss_experts.begin(), miss_experts.end(), e) == miss_experts.end()) {
                miss_experts.push_back(e);
            }
        }
    }

    if (!miss_experts.empty()) {
        if (static_cast<int>(miss_experts.size()) > mgr->slots_per_layer) return DEIMOS_ERR_BOUNDS;
        std::vector<int32_t> target_slots(miss_experts.size());
        for (size_t m = 0; m < miss_experts.size(); ++m) {
            int ps = mgr->allocate_slot_unlocked(layer_idx, miss_experts[m], protect.data());
            if (ps < 0) return DEIMOS_ERR_BOUNDS;
            protect[ps] = 1;
            target_slots[m] = ps;
        }
        int rc = mgr->stage_layer_experts(
            layer_idx, miss_experts.data(), target_slots.data(), static_cast<int>(miss_experts.size())
        );
        if (rc != DEIMOS_OK) return rc;
    }

    for (int i = 0; i < total_eval; ++i) {
        int ps = v2p[work_ids[i]];
        out_phys_slots[i] = ps;
        out_eff_scores[i] = work_sc[i];
        freqs[ps] += 1;
        steps[ps] = mgr->step_clock;
    }
    *out_eff_k = eff_k;

    if (next_layer_idx >= 0 && next_layer_idx < mgr->num_layers && next_pred_ids && next_pred_count > 0) {
        std::lock_guard<std::mutex> lk(mgr->mtx);
        mgr->pending_job.layer_idx = next_layer_idx;
        mgr->pending_job.count     = std::min(32, next_pred_count);
        for (int i = 0; i < mgr->pending_job.count; ++i) mgr->pending_job.expert_ids[i] = next_pred_ids[i];
        mgr->has_job = true;
        mgr->lookahead_submitted += 1;
        mgr->cv_work.notify_one();
    }
    return DEIMOS_OK;
}

void deimos_arena_wait_idle(void* arena_handle) {
    if (arena_handle) static_cast<DeimosArenaManager*>(arena_handle)->wait_idle();
}

int deimos_arena_get_slot_gens(void* arena_handle, int layer_idx, uint64_t* out_gens, int max_slots) {
    if (!arena_handle || !out_gens || max_slots <= 0) return DEIMOS_ERR_INVALID_ARG;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    if (layer_idx < 0 || layer_idx >= mgr->num_layers) return DEIMOS_ERR_INVALID_ARG;
    mgr->wait_idle();
    int n = std::min(max_slots, mgr->slots_per_layer);
    std::memcpy(out_gens, mgr->slot_gen[layer_idx].data(), static_cast<size_t>(n) * sizeof(uint64_t));
    return DEIMOS_OK;
}

void deimos_arena_get_telemetry(void* arena_handle, DeimosArenaTelemetry* out_tel) {
    if (!arena_handle || !out_tel) return;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    mgr->wait_idle();
    out_tel->cache_hits              = mgr->cache_hits;
    out_tel->cache_misses            = mgr->cache_misses;
    out_tel->evictions               = mgr->evictions;
    out_tel->lookahead_submitted     = mgr->lookahead_submitted;
    out_tel->lookahead_staged        = mgr->lookahead_staged;
    out_tel->lookahead_useful_hits   = mgr->lookahead_useful_hits;
    out_tel->tail_misses_skipped     = mgr->tail_misses_skipped;
    out_tel->total_experts_evaluated = mgr->total_experts_evaluated;
}

void deimos_arena_reset_telemetry(void* arena_handle) {
    if (!arena_handle) return;
    auto* mgr = static_cast<DeimosArenaManager*>(arena_handle);
    mgr->wait_idle();
    mgr->cache_hits = mgr->cache_misses = mgr->evictions = 0;
    mgr->lookahead_submitted = mgr->lookahead_staged = mgr->lookahead_useful_hits = 0;
    mgr->tail_misses_skipped = mgr->total_experts_evaluated = 0;
}

} // extern "C"
