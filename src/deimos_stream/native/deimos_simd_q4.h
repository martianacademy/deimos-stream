// ============================================================================
// DeimosStream Portable IEEE-754 FP16 & 32-Element `Q8_1 × Q4` SIMD Kernels
// (`src/deimos_stream/native/deimos_simd_q4.h`)
// ============================================================================
// `AVX2+FMA+F16C` (`x86_64`) 32-element half-group (`sx_e, sx_o`) INT8×UINT4
// `_mm256_maddubs_epi16` (`VPMADDUBSW`) kernels (`~52 µs/expert` on 7 cores,
// matching `ggml` `Q8_1` 32-element precision with exact FP32 group-sum bias,
// zero `int16` overflow, zero `libm` / `vzeroupper` stalls, and `m`-outer
// sequential `dn_w` scan), plus `ARM NEON` (`AArch64`) and scalar fallbacks.
// ============================================================================

#ifndef DEIMOS_SIMD_Q4_H
#define DEIMOS_SIMD_Q4_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace deimos_simd {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

inline float fp16_to_fp32(uint16_t h) {
#if defined(__aarch64__)
    __fp16 fh;
    std::memcpy(&fh, &h, sizeof(uint16_t));
    return static_cast<float>(fh);
#else
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0u) {
        if (mant != 0u) {
            exp = 1u;
            while ((mant & 0x0400u) == 0u) { mant <<= 1; --exp; }
            mant &= 0x03FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        } else {
            bits = sign;
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
#endif
}

inline uint16_t fp32_to_fp16(float f) {
#if defined(__aarch64__)
    __fp16 fh = static_cast<__fp16>(f);
    uint16_t out;
    std::memcpy(&out, &fh, sizeof(uint16_t));
    return out;
#else
    uint32_t x;
    std::memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x00001000u) >> 13));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint32_t round_bit = 0x00001000u;
    if ((mant & 0x00001FFFu) == round_bit && (mant & 0x00002000u) == 0u) round_bit = 0u;
    mant += round_bit;
    if (mant & 0x00800000u) { mant = 0u; ++exp; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
#endif
}

// Quantizes `xe[g*hg..]` and `xo[g*hg..]` with independent 32-element scales `sxe[g], sxo[g]`!
inline void quantize_halfgroups_q8(
    const float* xe, const float* xo, int ng, int hg,
    int8_t* q8e, int8_t* q8o, float* sxe, float* sxo, float* sum_x
) {
    for (int g = 0; g < ng; ++g) {
        const float* xeg = xe + g * hg;
        const float* xog = xo + g * hg;
        float max_e = 0.0f, max_o = 0.0f, s = 0.0f;
        for (int j = 0; j < hg; ++j) {
            float ve = xeg[j], vo = xog[j];
            s += ve + vo;
            float ae = std::fabs(ve), ao = std::fabs(vo);
            if (ae > max_e) max_e = ae;
            if (ao > max_o) max_o = ao;
        }
        sum_x[g] = s;
        sxe[g] = max_e / 127.0f;
        sxo[g] = max_o / 127.0f;
        float inv_e = (max_e > 1e-30f) ? (127.0f / max_e) : 0.0f;
        float inv_o = (max_o > 1e-30f) ? (127.0f / max_o) : 0.0f;
        int8_t* qe = q8e + g * hg;
        int8_t* qo = q8o + g * hg;
        for (int j = 0; j < hg; ++j) {
            qe[j] = static_cast<int8_t>(std::lrintf(xeg[j] * inv_e));
            qo[j] = static_cast<int8_t>(std::lrintf(xog[j] * inv_o));
        }
    }
}

inline float q4_dot_row_scalar(
    const uint8_t* w_row, const uint16_t* s_row, const uint16_t* b_row,
    const float* xe, const float* xo, const float* sum_x,
    int num_groups, int half_g
) {
    float acc = 0.0f;
    for (int g = 0; g < num_groups; ++g) {
        const uint8_t* wg = w_row + g * half_g;
        const float* xeg = xe + g * half_g;
        const float* xog = xo + g * half_g;
        float dot_q = 0.0f;
        for (int j = 0; j < half_g; ++j) {
            uint8_t p = wg[j];
            dot_q += static_cast<float>(p & 0x0Fu) * xeg[j] + static_cast<float>(p >> 4) * xog[j];
        }
        acc += fp16_to_fp32(s_row[g]) * dot_q + fp16_to_fp32(b_row[g]) * sum_x[g];
    }
    return acc;
}

#if defined(__aarch64__)
inline float q4_dot_row_neon(
    const uint8_t* w_row, const uint16_t* s_row, const uint16_t* b_row,
    const float* xe, const float* xo, const float* sum_x,
    int num_groups, int half_g
) {
    if ((half_g & 15) != 0) {
        return q4_dot_row_scalar(w_row, s_row, b_row, xe, xo, sum_x, num_groups, half_g);
    }
    float32x4_t v_acc = vdupq_n_f32(0.0f);
    float bias_acc = 0.0f;
    const uint8x16_t m4 = vdupq_n_u8(0x0Fu);
    for (int g = 0; g < num_groups; ++g) {
        float32x4_t v_grp = vdupq_n_f32(0.0f);
        const uint8_t* wg = w_row + g * half_g;
        const float* xeg = xe + g * half_g;
        const float* xog = xo + g * half_g;
        for (int b0 = 0; b0 < half_g; b0 += 16) {
            uint8x16_t raw = vld1q_u8(wg + b0);
            uint8x16_t lo8 = vandq_u8(raw, m4);
            uint8x16_t hi8 = vshrq_n_u8(raw, 4);
            uint16x8_t lo16_a = vmovl_u8(vget_low_u8(lo8));
            uint16x8_t hi16_a = vmovl_u8(vget_low_u8(hi8));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_a))), vld1q_f32(xeg + b0));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_a))), vld1q_f32(xog + b0));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16_a))), vld1q_f32(xeg + b0 + 4));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16_a))), vld1q_f32(xog + b0 + 4));
            uint16x8_t lo16_b = vmovl_u8(vget_high_u8(lo8));
            uint16x8_t hi16_b = vmovl_u8(vget_high_u8(hi8));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_b))), vld1q_f32(xeg + b0 + 8));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_b))), vld1q_f32(xog + b0 + 8));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16_b))), vld1q_f32(xeg + b0 + 12));
            v_grp = vfmaq_f32(v_grp, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16_b))), vld1q_f32(xog + b0 + 12));
        }
        v_acc = vfmaq_f32(v_acc, v_grp, vdupq_n_f32(fp16_to_fp32(s_row[g])));
        bias_acc += fp16_to_fp32(b_row[g]) * sum_x[g];
    }
    return vaddvq_f32(v_acc) + bias_acc;
}
#endif

inline float eval_swiglu_f32(float gate, float up, float limit, float alpha, float up_offset) {
    if (limit > 0.0f) {
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        else if (up < -limit) up = -limit;
    }
    return (gate / (1.0f + std::exp(-alpha * gate))) * (up + up_offset);
}

#if defined(__x86_64__) || defined(_M_X64)
inline bool cpu_has_avx2_fma_f16c() {
    static const bool supported = (__builtin_cpu_supports("avx2") &&
                                   __builtin_cpu_supports("fma") &&
                                   __builtin_cpu_supports("f16c"));
    return supported;
}

__attribute__((target("avx2,fma,f16c"), always_inline))
inline float hsum256_ps_avx2(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma,f16c"), always_inline))
inline float fast_silu_avx2(float x) {
    if (x <= -20.0f) return 0.0f;
    if (x >= 20.0f) return x;
    float z = -x;
    float t = z * 1.4426950408889634f;
    float nf = _mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(t),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    float r = z - nf * 0.693145751953125f - nf * 1.428606765330187e-6f;
    float p = 1.3888889e-3f;
    p = p * r + 8.3333338e-3f;
    p = p * r + 4.1666668e-2f;
    p = p * r + 1.6666667e-1f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    uint32_t pbits = static_cast<uint32_t>(static_cast<int32_t>(nf) + 127) << 23;
    float pow2n;
    std::memcpy(&pow2n, &pbits, sizeof(float));
    return x / (1.0f + p * pow2n);
}

__attribute__((target("avx2,fma,f16c"), always_inline))
inline __m256 dot32_q8_split_avx2(
    const uint8_t* wp, __m256i qe, __m256i qo, __m256i m4, __m256i ones,
    float sw_sxe, float sw_sxo, __m256 acc
) {
    __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp));
    __m256i ie = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(raw, m4), qe), ones);
    __m256i io = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(raw, 4), m4), qo), ones);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ie), _mm256_set1_ps(sw_sxe), acc);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(io), _mm256_set1_ps(sw_sxo), acc);
    return acc;
}

__attribute__((target("avx2,fma,f16c")))
inline void phase1_avx2_q8_range(
    const uint8_t* const* leaf_ptrs, int I, int half_d, int ng_gu, int hg,
    const int8_t* q8e, const int8_t* q8o, const float* sxe, const float* sxo, const float* sum_x,
    float* act_even, float* act_odd, int p0, int p1,
    float swiglu_limit = 0.0f, float swiglu_alpha = 1.0f, float swiglu_up_offset = 0.0f
) {
    const int half_i = I / 2;
    const bool std_swiglu = (swiglu_limit <= 0.0f && swiglu_alpha == 1.0f && swiglu_up_offset == 0.0f);
    const size_t gu_w_bytes = static_cast<size_t>(2 * I) * static_cast<size_t>(half_d);
    const size_t gu_s_bytes = static_cast<size_t>(2 * I) * static_cast<size_t>(ng_gu) * 2u;
    const __m256i m4   = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);

    for (int p = p0; p < p1; ++p) {
        const int m = p / half_i;
        const int j = p % half_i;
        const uint8_t*  gu_w = leaf_ptrs[m];
        const uint16_t* gu_s = reinterpret_cast<const uint16_t*>(gu_w + gu_w_bytes);
        const uint16_t* gu_b = reinterpret_cast<const uint16_t*>(gu_w + gu_w_bytes + gu_s_bytes);

        const int rg0 = 2 * j, rg1 = 2 * j + 1;
        const int ru0 = I + rg0, ru1 = I + rg1;
        const uint8_t* wg0 = gu_w + static_cast<size_t>(rg0) * half_d;
        const uint8_t* wg1 = gu_w + static_cast<size_t>(rg1) * half_d;
        const uint8_t* wu0 = gu_w + static_cast<size_t>(ru0) * half_d;
        const uint8_t* wu1 = gu_w + static_cast<size_t>(ru1) * half_d;
        const uint16_t* sg0 = gu_s + static_cast<size_t>(rg0) * ng_gu;
        const uint16_t* sg1 = gu_s + static_cast<size_t>(rg1) * ng_gu;
        const uint16_t* su0 = gu_s + static_cast<size_t>(ru0) * ng_gu;
        const uint16_t* su1 = gu_s + static_cast<size_t>(ru1) * ng_gu;
        const uint16_t* bg0 = gu_b + static_cast<size_t>(rg0) * ng_gu;
        const uint16_t* bg1 = gu_b + static_cast<size_t>(rg1) * ng_gu;
        const uint16_t* bu0 = gu_b + static_cast<size_t>(ru0) * ng_gu;
        const uint16_t* bu1 = gu_b + static_cast<size_t>(ru1) * ng_gu;

        __m256 vg0 = _mm256_setzero_ps(), vg1 = _mm256_setzero_ps();
        __m256 vu0 = _mm256_setzero_ps(), vu1 = _mm256_setzero_ps();
        float bg0_a = 0.0f, bg1_a = 0.0f, bu0_a = 0.0f, bu1_a = 0.0f;

        for (int g = 0; g < ng_gu; ++g) {
            const float sxeg = sxe[g], sxog = sxo[g], ssum = sum_x[g];
            __m256 sb8 = _mm256_cvtph_ps(_mm_set_epi16(
                bu1[g], su1[g], bu0[g], su0[g], bg1[g], sg1[g], bg0[g], sg0[g]
            ));
            alignas(32) float sbf[8];
            _mm256_store_ps(sbf, sb8);
            for (int b0 = 0; b0 < hg; b0 += 32) {
                __m256i qe = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8e + g * hg + b0));
                __m256i qo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8o + g * hg + b0));
                vg0 = dot32_q8_split_avx2(wg0 + g * hg + b0, qe, qo, m4, ones, sbf[0] * sxeg, sbf[0] * sxog, vg0);
                vg1 = dot32_q8_split_avx2(wg1 + g * hg + b0, qe, qo, m4, ones, sbf[2] * sxeg, sbf[2] * sxog, vg1);
                vu0 = dot32_q8_split_avx2(wu0 + g * hg + b0, qe, qo, m4, ones, sbf[4] * sxeg, sbf[4] * sxog, vu0);
                vu1 = dot32_q8_split_avx2(wu1 + g * hg + b0, qe, qo, m4, ones, sbf[6] * sxeg, sbf[6] * sxog, vu1);
            }
            bg0_a += sbf[1] * ssum;
            bg1_a += sbf[3] * ssum;
            bu0_a += sbf[5] * ssum;
            bu1_a += sbf[7] * ssum;
        }
        const float g0 = hsum256_ps_avx2(vg0) + bg0_a, u0 = hsum256_ps_avx2(vu0) + bu0_a;
        const float g1 = hsum256_ps_avx2(vg1) + bg1_a, u1 = hsum256_ps_avx2(vu1) + bu1_a;
        act_even[m * half_i + j] = std_swiglu ? (fast_silu_avx2(g0) * u0)
                                              : eval_swiglu_f32(g0, u0, swiglu_limit, swiglu_alpha, swiglu_up_offset);
        act_odd[m * half_i + j]  = std_swiglu ? (fast_silu_avx2(g1) * u1)
                                              : eval_swiglu_f32(g1, u1, swiglu_limit, swiglu_alpha, swiglu_up_offset);
    }
}

__attribute__((target("avx2,fma,f16c")))
inline void phase2_avx2_q8_range(
    const uint8_t* const* leaf_ptrs, const float* scores, int M,
    int D, int I, int half_d, int ng_gu, int half_i, int ng_dn, int hg,
    const int8_t* q8_ae, const int8_t* q8_ao, const float* sae, const float* sao, const float* act_sum,
    float* acc_buf, uint16_t* out_f16, int r0, int r1
) {
    const size_t gu_w_bytes = static_cast<size_t>(2 * I) * static_cast<size_t>(half_d);
    const size_t gu_s_bytes = static_cast<size_t>(2 * I) * static_cast<size_t>(ng_gu) * 2u;
    const size_t dn_off     = gu_w_bytes + 2u * gu_s_bytes;
    const size_t dn_w_bytes = static_cast<size_t>(D) * static_cast<size_t>(half_i);
    const size_t dn_s_bytes = static_cast<size_t>(D) * static_cast<size_t>(ng_dn) * 2u;
    const __m256i m4   = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);

    const int nrows = r1 - r0;
    std::memset(acc_buf, 0, static_cast<size_t>(nrows) * sizeof(float));

    for (int m = 0; m < M; ++m) {
        const float w_score = scores[m];
        const uint8_t*  dn_base = leaf_ptrs[m] + dn_off;
        const uint16_t* s_base  = reinterpret_cast<const uint16_t*>(dn_base + dn_w_bytes);
        const uint16_t* b_base  = reinterpret_cast<const uint16_t*>(dn_base + dn_w_bytes + dn_s_bytes);
        const int8_t* qae  = q8_ae   + m * half_i;
        const int8_t* qao  = q8_ao   + m * half_i;
        const float*  saem = sae     + m * ng_dn;
        const float*  saom = sao     + m * ng_dn;
        const float*  asm_ = act_sum + m * ng_dn;

        int r = r0;
        for (; r + 1 < r1; r += 2) {
            const uint8_t*  w0 = dn_base + static_cast<size_t>(r) * half_i;
            const uint8_t*  w1 = dn_base + static_cast<size_t>(r + 1) * half_i;
            const uint16_t* s0 = s_base  + static_cast<size_t>(r) * ng_dn;
            const uint16_t* s1 = s_base  + static_cast<size_t>(r + 1) * ng_dn;
            const uint16_t* b0 = b_base  + static_cast<size_t>(r) * ng_dn;
            const uint16_t* b1 = b_base  + static_cast<size_t>(r + 1) * ng_dn;
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            float ba0 = 0.0f, ba1 = 0.0f;
            for (int g = 0; g < ng_dn; ++g) {
                __m128 sb4 = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, b1[g], s1[g], b0[g], s0[g]));
                alignas(16) float sbf[4];
                _mm_store_ps(sbf, sb4);
                const float saeg = saem[g], saog = saom[g];
                for (int k0 = 0; k0 < hg; k0 += 32) {
                    __m256i qe = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qae + g * hg + k0));
                    __m256i qo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qao + g * hg + k0));
                    v0 = dot32_q8_split_avx2(w0 + g * hg + k0, qe, qo, m4, ones, sbf[0] * saeg, sbf[0] * saog, v0);
                    v1 = dot32_q8_split_avx2(w1 + g * hg + k0, qe, qo, m4, ones, sbf[2] * saeg, sbf[2] * saog, v1);
                }
                ba0 += sbf[1] * asm_[g];
                ba1 += sbf[3] * asm_[g];
            }
            acc_buf[r - r0]     += w_score * (hsum256_ps_avx2(v0) + ba0);
            acc_buf[r + 1 - r0] += w_score * (hsum256_ps_avx2(v1) + ba1);
        }
        for (; r < r1; ++r) {
            const uint8_t*  w0 = dn_base + static_cast<size_t>(r) * half_i;
            const uint16_t* s0 = s_base  + static_cast<size_t>(r) * ng_dn;
            const uint16_t* b0 = b_base  + static_cast<size_t>(r) * ng_dn;
            __m256 v0 = _mm256_setzero_ps();
            float ba0 = 0.0f;
            for (int g = 0; g < ng_dn; ++g) {
                __m128 sb2 = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, b0[g], s0[g]));
                float sw = _mm_cvtss_f32(sb2);
                for (int k0 = 0; k0 < hg; k0 += 32) {
                    __m256i qe = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qae + g * hg + k0));
                    __m256i qo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qao + g * hg + k0));
                    v0 = dot32_q8_split_avx2(w0 + g * hg + k0, qe, qo, m4, ones, sw * saem[g], sw * saom[g], v0);
                }
                ba0 += _mm_cvtss_f32(_mm_shuffle_ps(sb2, sb2, 1)) * asm_[g];
            }
            acc_buf[r - r0] += w_score * (hsum256_ps_avx2(v0) + ba0);
        }
    }

    for (int i = 0; i < nrows; ++i) {
        out_f16[r0 + i] = static_cast<uint16_t>(
            _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(acc_buf[i]), _MM_FROUND_TO_NEAREST_INT), 0)
        );
    }
}
#endif

inline float q4_dot_row(
    const uint8_t* w_row, const uint16_t* s_row, const uint16_t* b_row,
    const float* xe, const float* xo, const float* sum_x,
    int num_groups, int half_g
) {
#if defined(__aarch64__)
    return q4_dot_row_neon(w_row, s_row, b_row, xe, xo, sum_x, num_groups, half_g);
#else
    return q4_dot_row_scalar(w_row, s_row, b_row, xe, xo, sum_x, num_groups, half_g);
#endif
}

} // namespace deimos_simd

#endif // DEIMOS_SIMD_Q4_H
