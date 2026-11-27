// ============================================================================
// DeimosStream Zero-PCIe In-Place `cas_mmap` Multi-Threaded SIMD Q4 SwiGLU Core
// (`src/deimos_stream/native/deimos_cpu_moe.cpp`)
// ============================================================================

#include "deimos_stream.h"
#include "deimos_simd_q4.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#endif

namespace {

inline int detect_effective_cpus() {
    if (const char* env_t = std::getenv("DEIMOS_CPU_THREADS")) {
        int v = std::atoi(env_t);
        if (v > 0) return v;
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    int cpus = hw > 0 ? hw : 4;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        int aff = CPU_COUNT(&mask);
        if (aff > 0) cpus = std::min(cpus, aff);
    }
    auto check_v2 = [&](const std::string& p) {
        if (FILE* f = std::fopen(p.c_str(), "r")) {
            long q = 0, per = 0;
            if (std::fscanf(f, "%ld %ld", &q, &per) == 2 && q > 0 && per > 0) {
                int cg = static_cast<int>((q + per - 1) / per);
                if (cg > 0) cpus = std::min(cpus, cg);
            }
            std::fclose(f);
        }
    };
    auto check_v1 = [&](const std::string& dir) {
        FILE* fq = std::fopen((dir + "/cpu.cfs_quota_us").c_str(), "r");
        FILE* fp = std::fopen((dir + "/cpu.cfs_period_us").c_str(), "r");
        if (fq && fp) {
            long q = 0, per = 0;
            if (std::fscanf(fq, "%ld", &q) == 1 && std::fscanf(fp, "%ld", &per) == 1 && q > 0 && per > 0) {
                int cg = static_cast<int>((q + per - 1) / per);
                if (cg > 0) cpus = std::min(cpus, cg);
            }
        }
        if (fq) std::fclose(fq);
        if (fp) std::fclose(fp);
    };
    check_v2("/sys/fs/cgroup/cpu.max");
    check_v1("/sys/fs/cgroup/cpu");
    if (FILE* fc = std::fopen("/proc/self/cgroup", "r")) {
        char line[512];
        while (std::fgets(line, sizeof(line), fc)) {
            char* nl = std::strchr(line, '\n');
            if (nl) *nl = '\0';
            char* c1 = std::strchr(line, ':');
            char* c2 = c1 ? std::strchr(c1 + 1, ':') : nullptr;
            if (!c2 || c2[1] != '/') continue;
            std::string ctrl(c1 + 1, static_cast<size_t>(c2 - (c1 + 1)));
            std::string sub(c2 + 1);
            if (ctrl.empty()) check_v2("/sys/fs/cgroup" + sub + "/cpu.max");
            if (ctrl.find("cpu") != std::string::npos) check_v1("/sys/fs/cgroup/cpu" + sub);
        }
        std::fclose(fc);
    }
#endif
    return cpus;
}

using deimos_simd::cpu_relax;
using deimos_simd::fp16_to_fp32;
using deimos_simd::fp32_to_fp16;
using deimos_simd::q4_dot_row;
using deimos_simd::quantize_halfgroups_q8;

struct CpuMoeTask {
    const uint8_t* const* leaf_ptrs = nullptr;
    const float*          scores    = nullptr;
    int                   M         = 0;
    int                   D         = 0;
    int                   I         = 0;
    int                   G         = 64;
    const float*          xe        = nullptr;
    const float*          xo        = nullptr;
    const int8_t*         q8_xe     = nullptr;
    const int8_t*         q8_xo     = nullptr;
    const float*          sxe       = nullptr;
    const float*          sxo       = nullptr;
    const float*          sum_x     = nullptr;
    float*                act_even  = nullptr;
    float*                act_odd   = nullptr;
    int8_t*               q8_ae     = nullptr;
    int8_t*               q8_ao     = nullptr;
    float*                sae       = nullptr;
    float*                sao       = nullptr;
    float*                act_sum   = nullptr;
    uint16_t*             out_f16   = nullptr;
};

class DeimosCpuMoePool {
public:
    static DeimosCpuMoePool& instance() {
        static DeimosCpuMoePool pool;
        return pool;
    }

    void run(const CpuMoeTask& t) {
        if (num_threads_ <= 1) {
            exec_phase1(t, 0, 1);
            prep_act_q8(t);
            exec_phase2(t, 0, 1);
            return;
        }
        cur_task_ = &t;
        done_p1_.store(0, std::memory_order_relaxed);
        done_p2_.store(0, std::memory_order_relaxed);
        phase_gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_workers_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_.notify_all();
        }

        exec_phase1(t, 0, num_threads_);
        if (done_p1_.fetch_add(1, std::memory_order_acq_rel) + 1 < num_threads_) {
            while (done_p1_.load(std::memory_order_acquire) < num_threads_) { cpu_relax(); }
        }
        prep_act_q8(t);
        phase_gen_.fetch_add(1, std::memory_order_release);

        exec_phase2(t, 0, num_threads_);
        if (done_p2_.fetch_add(1, std::memory_order_acq_rel) + 1 < num_threads_) {
            while (done_p2_.load(std::memory_order_acquire) < num_threads_) { cpu_relax(); }
        }
    }

private:
    int num_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<uint64_t> phase_gen_{0};
    std::atomic<int> sleeping_workers_{0};
    std::atomic<int> done_p1_{0};
    std::atomic<int> done_p2_{0};
    std::atomic<bool> stop_{false};
    const CpuMoeTask* cur_task_ = nullptr;

    DeimosCpuMoePool() {
        int cpus = detect_effective_cpus();
        int target = (cpus >= 6) ? (cpus - 2) : (cpus > 2 ? (cpus - 1) : cpus);
        num_threads_ = std::max(1, std::min(12, target));
        for (int tid = 1; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_main(tid); });
        }
    }

    ~DeimosCpuMoePool() {
        stop_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_.notify_all();
        }
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    static void prep_act_q8(const CpuMoeTask& t) {
        const int half_i = t.I / 2;
        const int ng_dn  = t.I / t.G;
        const int hg     = t.G / 2;
        for (int m = 0; m < t.M; ++m) {
            quantize_halfgroups_q8(
                t.act_even + m * half_i,
                t.act_odd  + m * half_i,
                ng_dn, hg,
                t.q8_ae   + m * half_i,
                t.q8_ao   + m * half_i,
                t.sae     + m * ng_dn,
                t.sao     + m * ng_dn,
                t.act_sum + m * ng_dn
            );
        }
    }

    static void exec_phase1(const CpuMoeTask& t, int tid, int nthreads) {
        const int half_d = t.D / 2;
        const int ng_gu  = t.D / t.G;
        const int hg     = t.G / 2;
        const int half_i = t.I / 2;
        const int total_pairs = t.M * half_i;
        const int p0 = (total_pairs * tid) / nthreads;
        const int p1 = (total_pairs * (tid + 1)) / nthreads;
        if (p0 >= p1) return;

#if defined(__x86_64__) || defined(_M_X64)
        if ((hg & 31) == 0 && deimos_simd::cpu_has_avx2_fma_f16c()) {
            deimos_simd::phase1_avx2_q8_range(
                t.leaf_ptrs, t.I, half_d, ng_gu, hg,
                t.q8_xe, t.q8_xo, t.sxe, t.sxo, t.sum_x,
                t.act_even, t.act_odd, p0, p1
            );
            return;
        }
#endif
        const size_t gu_w_bytes = static_cast<size_t>(2 * t.I) * static_cast<size_t>(half_d);
        const size_t gu_s_bytes = static_cast<size_t>(2 * t.I) * static_cast<size_t>(ng_gu) * 2u;

        for (int p = p0; p < p1; ++p) {
            const int m = p / half_i;
            const int j = p % half_i;
            const uint8_t*  gu_w = t.leaf_ptrs[m];
            const uint16_t* gu_s = reinterpret_cast<const uint16_t*>(gu_w + gu_w_bytes);
            const uint16_t* gu_b = reinterpret_cast<const uint16_t*>(gu_w + gu_w_bytes + gu_s_bytes);

            for (int sub = 0; sub < 2; ++sub) {
                const int r_gate = 2 * j + sub;
                const int r_up   = t.I + r_gate;
                float gate = q4_dot_row(
                    gu_w + static_cast<size_t>(r_gate) * half_d,
                    gu_s + static_cast<size_t>(r_gate) * ng_gu,
                    gu_b + static_cast<size_t>(r_gate) * ng_gu,
                    t.xe, t.xo, t.sum_x, ng_gu, hg
                );
                float up = q4_dot_row(
                    gu_w + static_cast<size_t>(r_up) * half_d,
                    gu_s + static_cast<size_t>(r_up) * ng_gu,
                    gu_b + static_cast<size_t>(r_up) * ng_gu,
                    t.xe, t.xo, t.sum_x, ng_gu, hg
                );
                float val = (gate / (1.0f + std::exp(-gate))) * up;
                if (sub == 0) t.act_even[m * half_i + j] = val;
                else          t.act_odd[m * half_i + j]  = val;
            }
        }
    }

    static void exec_phase2(const CpuMoeTask& t, int tid, int nthreads) {
        const int half_d = t.D / 2;
        const int ng_gu  = t.D / t.G;
        const int half_i = t.I / 2;
        const int ng_dn  = t.I / t.G;
        const int hg     = t.G / 2;
        const int r0 = (t.D * tid) / nthreads;
        const int r1 = (t.D * (tid + 1)) / nthreads;
        if (r0 >= r1) return;

#if defined(__x86_64__) || defined(_M_X64)
        if ((hg & 31) == 0 && deimos_simd::cpu_has_avx2_fma_f16c()) {
            thread_local std::vector<float> tl_acc;
            if (static_cast<int>(tl_acc.size()) < (r1 - r0)) {
                tl_acc.resize(static_cast<size_t>(r1 - r0));
            }
            deimos_simd::phase2_avx2_q8_range(
                t.leaf_ptrs, t.scores, t.M, t.D, t.I,
                half_d, ng_gu, half_i, ng_dn, hg,
                t.q8_ae, t.q8_ao, t.sae, t.sao, t.act_sum,
                tl_acc.data(), t.out_f16, r0, r1
            );
            return;
        }
#endif
        const size_t gu_w_bytes = static_cast<size_t>(2 * t.I) * static_cast<size_t>(half_d);
        const size_t gu_s_bytes = static_cast<size_t>(2 * t.I) * static_cast<size_t>(ng_gu) * 2u;
        const size_t dn_off     = gu_w_bytes + 2u * gu_s_bytes;
        const size_t dn_w_bytes = static_cast<size_t>(t.D) * static_cast<size_t>(half_i);
        const size_t dn_s_bytes = static_cast<size_t>(t.D) * static_cast<size_t>(ng_dn) * 2u;

        for (int r = r0; r < r1; ++r) {
            float acc = 0.0f;
            for (int m = 0; m < t.M; ++m) {
                const uint8_t*  dn_base = t.leaf_ptrs[m] + dn_off;
                const uint8_t*  dn_w = dn_base + static_cast<size_t>(r) * half_i;
                const uint16_t* dn_s = reinterpret_cast<const uint16_t*>(dn_base + dn_w_bytes) + static_cast<size_t>(r) * ng_dn;
                const uint16_t* dn_b = reinterpret_cast<const uint16_t*>(dn_base + dn_w_bytes + dn_s_bytes) + static_cast<size_t>(r) * ng_dn;
                acc += t.scores[m] * q4_dot_row(
                    dn_w, dn_s, dn_b,
                    t.act_even + m * half_i,
                    t.act_odd  + m * half_i,
                    t.act_sum  + m * ng_dn,
                    ng_dn, hg
                );
            }
            t.out_f16[r] = fp32_to_fp16(acc);
        }
    }

    void worker_main(int tid) {
        uint64_t seen_gen = 0;
        while (true) {
            uint64_t g = phase_gen_.load(std::memory_order_acquire);
            if (g == seen_gen) {
                int spins = 0;
                while ((g = phase_gen_.load(std::memory_order_acquire)) == seen_gen && spins < 600) {
                    cpu_relax();
                    ++spins;
                }
                if (g == seen_gen) {
                    std::unique_lock<std::mutex> lk(mtx_);
                    sleeping_workers_.fetch_add(1, std::memory_order_seq_cst);
                    cv_.wait(lk, [&] {
                        return stop_.load(std::memory_order_acquire) ||
                               phase_gen_.load(std::memory_order_seq_cst) != seen_gen;
                    });
                    sleeping_workers_.fetch_sub(1, std::memory_order_seq_cst);
                    if (stop_.load(std::memory_order_acquire)) return;
                    g = phase_gen_.load(std::memory_order_acquire);
                }
            }
            if (stop_.load(std::memory_order_acquire)) return;

            const CpuMoeTask* task = cur_task_;
            exec_phase1(*task, tid, num_threads_);
            done_p1_.fetch_add(1, std::memory_order_acq_rel);

            const uint64_t p2_target = g + 1;
            while (phase_gen_.load(std::memory_order_acquire) < p2_target) {
                if (stop_.load(std::memory_order_acquire)) return;
                cpu_relax();
            }
            exec_phase2(*task, tid, num_threads_);
            done_p2_.fetch_add(1, std::memory_order_acq_rel);
            seen_gen = p2_target;
        }
    }
};

} // namespace

extern "C" int deimos_cpu_q4_swiglu_forward(
    void*           merkle_handle,
    int             layer_idx,
    const int32_t*  expert_ids,
    const float*    expert_scores,
    int             num_experts_active,
    const uint16_t* x_f16,
    uint16_t*       out_f16,
    int             hidden_size,
    int             intermediate_size,
    int             group_size,
    int             verify_on_cold
) {
    if (!merkle_handle || !expert_ids || !expert_scores || !x_f16 || !out_f16 ||
        num_experts_active <= 0 || hidden_size <= 0 || intermediate_size <= 0 ||
        (hidden_size % group_size) != 0 || (intermediate_size % group_size) != 0) {
        return DEIMOS_ERR_INVALID_ARG;
    }

    const uint8_t* ptrs_stack[32];
    std::vector<const uint8_t*> ptrs_heap;
    const uint8_t** leaf_ptrs = ptrs_stack;
    if (num_experts_active > 32) {
        ptrs_heap.resize(static_cast<size_t>(num_experts_active));
        leaf_ptrs = ptrs_heap.data();
    }
    for (int m = 0; m < num_experts_active; ++m) {
        int rc = DEIMOS_OK;
        leaf_ptrs[m] = deimos_get_expert_leaf_ptr(merkle_handle, layer_idx, expert_ids[m], verify_on_cold, &rc);
        if (!leaf_ptrs[m] || rc != DEIMOS_OK) return (rc != DEIMOS_OK) ? rc : DEIMOS_ERR_BOUNDS;
    }

    const int half_d = hidden_size / 2;
    const int ng_gu  = hidden_size / group_size;
    const int hg     = group_size / 2;
    const int half_i = intermediate_size / 2;
    const int ng_dn  = intermediate_size / group_size;

    thread_local std::vector<float>  tl_fbuf;
    thread_local std::vector<int8_t> tl_qbuf;
    const size_t freq = static_cast<size_t>(half_d * 2 + ng_gu * 3) +
                        static_cast<size_t>(num_experts_active) * static_cast<size_t>(half_i * 2 + ng_dn * 3);
    const size_t qreq = static_cast<size_t>(half_d * 2) +
                        static_cast<size_t>(num_experts_active) * static_cast<size_t>(half_i * 2);
    if (tl_fbuf.size() < freq) tl_fbuf.resize(freq);
    if (tl_qbuf.size() < qreq) tl_qbuf.resize(qreq);

    float* xe       = tl_fbuf.data();
    float* xo       = xe + half_d;
    float* sxe      = xo + half_d;
    float* sxo      = sxe + ng_gu;
    float* sum_x    = sxo + ng_gu;
    float* act_even = sum_x + ng_gu;
    float* act_odd  = act_even + static_cast<size_t>(num_experts_active) * half_i;
    float* sae      = act_odd  + static_cast<size_t>(num_experts_active) * half_i;
    float* sao      = sae      + static_cast<size_t>(num_experts_active) * ng_dn;
    float* act_sum  = sao      + static_cast<size_t>(num_experts_active) * ng_dn;

    int8_t* q8_xe = tl_qbuf.data();
    int8_t* q8_xo = q8_xe + half_d;
    int8_t* q8_ae = q8_xo + half_d;
    int8_t* q8_ao = q8_ae + static_cast<size_t>(num_experts_active) * half_i;

    for (int j = 0; j < half_d; ++j) {
        xe[j] = fp16_to_fp32(x_f16[2 * j]);
        xo[j] = fp16_to_fp32(x_f16[2 * j + 1]);
    }
    quantize_halfgroups_q8(xe, xo, ng_gu, hg, q8_xe, q8_xo, sxe, sxo, sum_x);

    CpuMoeTask task{
        leaf_ptrs, expert_scores, num_experts_active,
        hidden_size, intermediate_size, group_size,
        xe, xo, q8_xe, q8_xo, sxe, sxo, sum_x,
        act_even, act_odd, q8_ae, q8_ao, sae, sao, act_sum, out_f16
    };
    DeimosCpuMoePool::instance().run(task);
    return DEIMOS_OK;
}

extern "C" void deimos_warm_cpu_moe_pool(
    void* merkle_handle,
    int   num_layers,
    int   start_expert,
    int   end_expert,
    int   hidden_size,
    int   intermediate_size,
    int   group_size
) {
    if (!merkle_handle || hidden_size <= 0 || intermediate_size <= 0) return;
    std::vector<uint16_t> x_zero(static_cast<size_t>(hidden_size), 0);
    std::vector<uint16_t> y_zero(static_cast<size_t>(hidden_size), 0);
    int32_t eid = 0;
    float esc = 1.0f;
    deimos_cpu_q4_swiglu_forward(
        merkle_handle, 0, &eid, &esc, 1,
        x_zero.data(), y_zero.data(), hidden_size, intermediate_size, group_size, 0
    );
    if (start_expert > 0) {
        deimos_evict_expert(merkle_handle, 0, 0);
    }
#if defined(__linux__)
    if (num_layers <= 0 || start_expert < 0 || start_expert >= end_expert) return;
    DeimosHeader hdr{};
    if (deimos_get_header(merkle_handle, &hdr) != DEIMOS_OK) return;
    const size_t stride = hdr.expert_stride_bytes;
    const int e_hi = std::min(end_expert, static_cast<int>(hdr.num_experts));
    if (start_expert >= e_hi) return;

    const int n_thr = std::min(4, num_layers);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_thr));
    for (int t = 0; t < n_thr; ++t) {
        workers.emplace_back([&, t]() {
            volatile uint8_t sink = 0;
            for (int l = t; l < num_layers; l += n_thr) {
                const uint8_t* p0 = deimos_get_expert_leaf_ptr(merkle_handle, l, start_expert, 0, nullptr);
                const uint8_t* p1 = deimos_get_expert_leaf_ptr(merkle_handle, l, e_hi - 1, 0, nullptr);
                if (p0 && p1 && p1 >= p0) {
                    const size_t span = static_cast<size_t>(p1 - p0) + stride;
                    posix_madvise(const_cast<uint8_t*>(p0), span, POSIX_MADV_WILLNEED);
                    for (size_t off = 0; off < span; off += 4096u) {
                        sink ^= p0[off];
                    }
                }
            }
            (void)sink;
        });
    }
    for (auto& w : workers) w.join();
#else
    (void)num_layers; (void)start_expert; (void)end_expert;
#endif
}
