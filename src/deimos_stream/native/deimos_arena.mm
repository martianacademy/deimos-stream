// ============================================================================
// DeimosStream Native C++17 Page Table, Residual Lookahead Async Stager &
// Miss-Only Nucleus Dynamic-K Resolver (`src/deimos_stream/native/deimos_arena.mm`)
// ============================================================================

#include "deimos_stream.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct LayerPoolPtrs {
    uint8_t* gu_w = nullptr;
    uint8_t* gu_s = nullptr;
    uint8_t* gu_b = nullptr;
    uint8_t* dn_w = nullptr;
    uint8_t* dn_s = nullptr;
    uint8_t* dn_b = nullptr;
};

struct LookaheadJob {
    int     layer_idx = -1;
    int     count     = 0;
    int32_t expert_ids[32];
};

struct DeimosArenaManager {
    void*  merkle_handle     = nullptr;
    int    num_layers        = 0;
    int    num_experts       = 0;
    int    slots_per_layer   = 0;
    size_t gu_w_bytes        = 0;
    size_t gu_s_bytes        = 0;
    size_t dn_w_bytes        = 0;
    size_t dn_s_bytes        = 0;
    int    verify_on_cold    = 1;
    int    use_nocache_pread = 0;

    std::vector<LayerPoolPtrs>        ptrs;
    std::vector<std::vector<int32_t>> virt_to_phys;
    std::vector<std::vector<int32_t>> phys_to_virt;
    std::vector<std::vector<int32_t>> slot_freq;
    std::vector<std::vector<int64_t>> slot_step;
    std::vector<std::vector<uint8_t>> prefetched_flag;
    std::vector<std::vector<int32_t>> free_slots;
    int64_t                           step_clock = 0;

    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv_work;
    std::condition_variable cv_idle;
    bool                    has_job   = false;
    bool                    busy      = false;
    bool                    stop_flag = false;
    int                     last_err  = DEIMOS_OK;
    LookaheadJob            pending_job{};

    uint64_t cache_hits              = 0;
    uint64_t cache_misses            = 0;
    uint64_t evictions               = 0;
    uint64_t lookahead_submitted     = 0;
    uint64_t lookahead_staged        = 0;
    uint64_t lookahead_useful_hits   = 0;
    uint64_t tail_misses_skipped     = 0;
    uint64_t total_experts_evaluated = 0;

    int allocate_slot_unlocked(int l_idx, int e_miss, const uint8_t* protect_phys) {
        auto& frees = free_slots[l_idx];
        auto& v2p   = virt_to_phys[l_idx];
        auto& p2v   = phys_to_virt[l_idx];
        auto& freqs = slot_freq[l_idx];
        auto& steps = slot_step[l_idx];

        int p_slot = -1;
        if (!frees.empty()) {
            p_slot = frees.back();
            frees.pop_back();
        } else {
            int32_t best_f = 0x7fffffff;
            int64_t best_s = 0x7fffffffffffffffLL;
            for (int c = 0; c < slots_per_layer; ++c) {
                if (protect_phys[c]) continue;
                if (freqs[c] < best_f || (freqs[c] == best_f && steps[c] < best_s)) {
                    best_f = freqs[c];
                    best_s = steps[c];
                    p_slot = c;
                }
            }
            if (p_slot < 0) return -1;
            int old_e = p2v[p_slot];
            if (old_e >= 0) {
                v2p[old_e] = -1;
                prefetched_flag[l_idx][old_e] = 0;
                evictions += 1;
                deimos_evict_expert(merkle_handle, l_idx, old_e);
            }
        }
        v2p[e_miss]   = p_slot;
        p2v[p_slot]   = e_miss;
        freqs[p_slot] = 1;
        steps[p_slot] = step_clock;
        return p_slot;
    }

    int stage_layer_experts(int l_idx, const int32_t* e_ids, const int32_t* p_slots, int count) {
        const LayerPoolPtrs& lp = ptrs[l_idx];
        return deimos_stage_q4_batch(
            merkle_handle, l_idx, e_ids, p_slots, count,
            lp.gu_w, lp.gu_s, lp.gu_b, lp.dn_w, lp.dn_s, lp.dn_b,
            gu_w_bytes, gu_s_bytes, dn_w_bytes, dn_s_bytes,
            verify_on_cold, use_nocache_pread
        );
    }

    int execute_lookahead_job(const LookaheadJob& job) {
        const int l_idx = job.layer_idx;
        if (l_idx < 0 || l_idx >= num_layers || job.count <= 0 || !ptrs[l_idx].gu_w) {
            return DEIMOS_OK;
        }
        auto& v2p = virt_to_phys[l_idx];
        std::vector<uint8_t> protect(static_cast<size_t>(slots_per_layer), 0);

        int uniq_pred[32];
        int n_uniq = 0;
        for (int i = 0; i < job.count && n_uniq < 32; ++i) {
            int e = job.expert_ids[i];
            if (e < 0 || e >= num_experts) continue;
            bool dup = false;
            for (int j = 0; j < n_uniq; ++j) {
                if (uniq_pred[j] == e) { dup = true; break; }
            }
            if (!dup) {
                uniq_pred[n_uniq++] = e;
                if (v2p[e] >= 0) protect[v2p[e]] = 1;
            }
        }

        int32_t stage_e[32], stage_s[32];
        int n_stage = 0;
        step_clock += 1;
        for (int i = 0; i < n_uniq && n_stage < slots_per_layer; ++i) {
            int e = uniq_pred[i];
            if (v2p[e] >= 0) continue;
            int p_slot = allocate_slot_unlocked(l_idx, e, protect.data());
            if (p_slot < 0) break;
            protect[p_slot] = 1;
            prefetched_flag[l_idx][e] = 1;
            stage_e[n_stage] = e;
            stage_s[n_stage] = p_slot;
            n_stage += 1;
        }
        if (n_stage == 0) return DEIMOS_OK;
        int rc = stage_layer_experts(l_idx, stage_e, stage_s, n_stage);
        if (rc == DEIMOS_OK) lookahead_staged += static_cast<uint64_t>(n_stage);
        return rc;
    }

    void worker_loop() {
        while (true) {
            LookaheadJob job{};
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_work.wait(lk, [&] { return stop_flag || has_job; });
                if (stop_flag && !has_job) return;
                job = pending_job;
                has_job = false;
                busy = true;
            }
            int rc = execute_lookahead_job(job);
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (rc != DEIMOS_OK && last_err == DEIMOS_OK) last_err = rc;
                busy = false;
                if (!has_job) cv_idle.notify_all();
            }
        }
    }

    void wait_idle() {
        std::unique_lock<std::mutex> lk(mtx);
        cv_idle.wait(lk, [&] { return !has_job && !busy; });
    }

    ~DeimosArenaManager() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop_flag = true;
        }
        cv_work.notify_all();
        if (worker.joinable()) worker.join();
    }
};

} // namespace

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
    mgr->slot_step.assign(nL, std::vector<int64_t>(nC, 0));
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
    mgr->use_nocache_pread = 0;
    for (int l = 0; l < mgr->num_layers; ++l) {
        int rc = mgr->stage_layer_experts(l, all_ids.data(), all_ids.data(), mgr->num_experts);
        if (rc != DEIMOS_OK) { mgr->use_nocache_pread = saved_nc; return rc; }
        mgr->virt_to_phys[l] = all_ids;
        mgr->phys_to_virt[l] = all_ids;
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
    if (mgr->last_err != DEIMOS_OK) {
        int err = mgr->last_err;
        mgr->last_err = DEIMOS_OK;
        return err;
    }

    mgr->step_clock += 1;
    if ((mgr->step_clock & 511LL) == 0LL) {
        for (int s = 0; s < mgr->slots_per_layer; ++s) mgr->slot_freq[layer_idx][s] >>= 1;
    }

    auto& v2p   = mgr->virt_to_phys[layer_idx];
    auto& freqs = mgr->slot_freq[layer_idx];
    auto& steps = mgr->slot_step[layer_idx];
    auto& pflag = mgr->prefetched_flag[layer_idx];

    std::vector<uint8_t> protect(static_cast<size_t>(mgr->slots_per_layer), 0);
    std::vector<int32_t> work_ids(static_cast<size_t>(num_tokens * top_k));
    std::vector<float>   work_sc(static_cast<size_t>(num_tokens * top_k));
    int eff_k = top_k;

    if (num_tokens == 1 && miss_tail_tau < 0.9999f && top_k <= 32) {
        int ord[32];
        float sum_all = 0.0f;
        for (int j = 0; j < top_k; ++j) { ord[j] = j; sum_all += routed_scores[j]; }
        for (int i = 1; i < top_k; ++i) {
            int key_i = ord[i], j = i - 1;
            while (j >= 0 && routed_scores[ord[j]] < routed_scores[key_i]) { ord[j + 1] = ord[j]; --j; }
            ord[j + 1] = key_i;
        }
        int kept = 0;
        float cum_before = 0.0f, kept_sum = 0.0f;
        const float inv_sum = (sum_all > 1e-8f) ? (1.0f / sum_all) : 0.0f;
        for (int rank = 0; rank < top_k; ++rank) {
            int e_id = routed_ids[ord[rank]];
            float sc = routed_scores[ord[rank]];
            bool in_vram = (e_id >= 0 && e_id < mgr->num_experts && v2p[e_id] >= 0);
            float share = cum_before * inv_sum;
            cum_before += sc;
            if (rank < min_k || in_vram || share < miss_tail_tau) {
                work_ids[kept] = e_id;
                work_sc[kept]  = sc;
                kept_sum += sc;
                kept += 1;
            } else {
                mgr->tail_misses_skipped += 1;
            }
        }
        const float rescale = (kept_sum > 1e-8f) ? (sum_all / kept_sum) : 1.0f;
        for (int j = 0; j < kept; ++j) work_sc[j] *= rescale;
        eff_k = std::max(1, kept);
    } else {
        std::memcpy(work_ids.data(), routed_ids, static_cast<size_t>(num_tokens * top_k) * sizeof(int32_t));
        std::memcpy(work_sc.data(), routed_scores, static_cast<size_t>(num_tokens * top_k) * sizeof(float));
    }

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
