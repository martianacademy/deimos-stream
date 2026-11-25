// ============================================================================
// DeimosStream Cross-Platform SHA-256 Dispatch (`include/deimos_sha256.h`)
// ============================================================================
// Uses Apple Silicon hardware-accelerated `CC_SHA256` on macOS (`__APPLE__`)
// and a zero-dependency FIPS 180-4 C++17 implementation on Linux / POSIX.
// ============================================================================

#ifndef DEIMOS_SHA256_H
#define DEIMOS_SHA256_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__APPLE__) && !defined(DEIMOS_PORTABLE_SHA256)
#include <CommonCrypto/CommonDigest.h>
inline void deimos_sha256_bytes(const uint8_t* data, size_t len, uint8_t out32[32]) {
    CC_SHA256(data, static_cast<CC_LONG>(len), out32);
}
#else
namespace deimos_detail {
inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }

inline void sha256_transform(uint32_t st[8], const uint8_t blk[64]) {
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(blk[i * 4]) << 24) |
               (static_cast<uint32_t>(blk[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(blk[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(blk[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a=st[0], b=st[1], c=st[2], d=st[3], e=st[4], f=st[5], g=st[6], h=st[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}
} // namespace deimos_detail

inline void deimos_sha256_bytes(const uint8_t* data, size_t len, uint8_t out32[32]) {
    uint32_t st[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    size_t off = 0;
    while (off + 64 <= len) {
        deimos_detail::sha256_transform(st, data + off);
        off += 64;
    }
    uint8_t tail[128];
    std::memset(tail, 0, sizeof(tail));
    size_t rem = len - off;
    if (rem > 0) std::memcpy(tail, data + off, rem);
    tail[rem] = 0x80u;
    size_t blocks = (rem < 56) ? 1u : 2u;
    uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    for (int i = 0; i < 8; ++i) {
        tail[blocks * 64 - 1 - i] = static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFFu);
    }
    for (size_t b = 0; b < blocks; ++b) {
        deimos_detail::sha256_transform(st, tail + b * 64);
    }
    for (int i = 0; i < 8; ++i) {
        out32[i * 4]     = static_cast<uint8_t>((st[i] >> 24) & 0xFFu);
        out32[i * 4 + 1] = static_cast<uint8_t>((st[i] >> 16) & 0xFFu);
        out32[i * 4 + 2] = static_cast<uint8_t>((st[i] >> 8) & 0xFFu);
        out32[i * 4 + 3] = static_cast<uint8_t>(st[i] & 0xFFu);
    }
}
#endif

#endif // DEIMOS_SHA256_H
