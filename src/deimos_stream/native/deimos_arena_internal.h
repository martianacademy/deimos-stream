// ============================================================================
// DeimosStream Internal Arena State & Miss-Tail Filter (`deimos_arena_internal.h`)
// ============================================================================

#ifndef DEIMOS_ARENA_INTERNAL_H
#define DEIMOS_ARENA_INTERNAL_H

#include "deimos_stream.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

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

inline int filter_miss_tail_tau(
    const int32_t* routed_ids, const float* routed_scores,
    int num_tokens, int top_k, float miss_tail_tau, int min_k,
    int num_experts, const std::vector<int32_t>& v2p,
    int32_t* work_ids, float* work_sc, uint64_t* skipped_counter
) {
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
            bool in_vram = (e_id >= 0 && e_id < num_experts && v2p[e_id] >= 0);
            float share = cum_before * inv_sum;
            cum_before += sc;
            if (rank < min_k || in_vram || share < miss_tail_tau) {
                work_ids[kept] = e_id;
                work_sc[kept]  = sc;
                kept_sum += sc;
                kept += 1;
            } else {
                *skipped_counter += 1;
            }
        }
        const float rescale = (kept_sum > 1e-8f) ? (sum_all / kept_sum) : 1.0f;
        for (int j = 0; j < kept; ++j) work_sc[j] *= rescale;
        return std::max(1, kept);
    }
    std::memcpy(work_ids, routed_ids, static_cast<size_t>(num_tokens * top_k) * sizeof(int32_t));
    std::memcpy(work_sc, routed_scores, static_cast<size_t>(num_tokens * top_k) * sizeof(float));
    return top_k;
}

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

    std::vector<LayerPoolPtrs>         ptrs;
    std::vector<std::vector<int32_t>>  virt_to_phys;
    std::vector<std::vector<int32_t>>  phys_to_virt;
    std::vector<std::vector<int32_t>>  slot_freq;
    std::vector<std::vector<uint32_t>> expert_freq;
    std::vector<std::vector<uint32_t>> total_expert_hits;
    std::vector<std::vector<int64_t>>  slot_step;
    std::vector<std::vector<uint64_t>> slot_gen;
    std::vector<std::vector<uint8_t>>  prefetched_flag;
    std::vector<std::vector<int32_t>>  free_slots;
    int64_t                            step_clock = 0;
    int64_t                            last_promo_step = -1000;
    uint64_t                           gen_clock  = 0;

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

    int find_victim_slot(int l_idx, const uint8_t* protect_phys) const {
        const auto& freqs = slot_freq[l_idx];
        const auto& steps = slot_step[l_idx];
        int p_slot = -1;
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
        return p_slot;
    }

    int allocate_slot_unlocked(int l_idx, int e_miss, const uint8_t* protect_phys, int32_t init_freq = 1) {
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
            p_slot = find_victim_slot(l_idx, protect_phys);
            if (p_slot < 0) return -1;
            int old_e = p2v[p_slot];
            if (old_e >= 0) {
                v2p[old_e] = -1;
                prefetched_flag[l_idx][old_e] = 0;
                if (freqs[p_slot] > 0) evictions += 1;
                if (use_nocache_pread) {
                    deimos_evict_expert(merkle_handle, l_idx, old_e);
                }
            }
        }
        v2p[e_miss]   = p_slot;
        p2v[p_slot]   = e_miss;
        freqs[p_slot] = init_freq;
        steps[p_slot] = step_clock;
        slot_gen[l_idx][p_slot] = ++gen_clock;
        return p_slot;
    }

    int stage_layer_experts(int l_idx, const int32_t* e_ids, const int32_t* p_slots, int count) {
        const LayerPoolPtrs& lp = ptrs[l_idx];
        if (!lp.gu_w) return DEIMOS_ERR_INVALID_ARG;
        return deimos_stage_q4_batch(
            merkle_handle, l_idx, e_ids, p_slots, count,
            lp.gu_w, lp.gu_s, lp.gu_b, lp.dn_w, lp.dn_s, lp.dn_b,
            gu_w_bytes, gu_s_bytes, dn_w_bytes, dn_s_bytes,
            verify_on_cold, use_nocache_pread
        );
    }

    int execute_lookahead_job(const LookaheadJob& job) {
        const int l_idx = job.layer_idx;
        if (l_idx < 0 || l_idx >= num_layers || job.count <= 0) return DEIMOS_OK;
        if (!ptrs[l_idx].gu_w) {
            const auto& v2p_la = virt_to_phys[l_idx];
            int32_t warm_e[32];
            int n_warm = 0;
            for (int i = 0; i < job.count && n_warm < 32; ++i) {
                int e = job.expert_ids[i];
                if (e >= 0 && e < num_experts && v2p_la[e] < 0) warm_e[n_warm++] = e;
            }
            if (n_warm > 0) {
                deimos_prefetch_layer(merkle_handle, l_idx, warm_e, n_warm);
                lookahead_staged += static_cast<uint64_t>(n_warm);
            }
            return DEIMOS_OK;
        }
        auto& v2p = virt_to_phys[l_idx];
        std::vector<uint8_t> protect(static_cast<size_t>(slots_per_layer), 0);

        int uniq_pred[32], n_uniq = 0;
        for (int i = 0; i < job.count && n_uniq < 32; ++i) {
            int e = job.expert_ids[i];
            if (e < 0 || e >= num_experts) continue;
            bool dup = false;
            for (int j = 0; j < n_uniq; ++j) if (uniq_pred[j] == e) { dup = true; break; }
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

    void tick_decay(int layer_idx) {
        step_clock += 1;
        if ((step_clock & 511LL) == 0LL) {
            for (int s = 0; s < slots_per_layer; ++s) slot_freq[layer_idx][s] >>= 1;
            for (int e = 0; e < num_experts; ++e) expert_freq[layer_idx][e] >>= 1;
        }
    }

    ~DeimosArenaManager() {
        { std::lock_guard<std::mutex> lk(mtx); stop_flag = true; }
        cv_work.notify_all();
        if (worker.joinable()) worker.join();
    }
};

#endif // DEIMOS_ARENA_INTERNAL_H
