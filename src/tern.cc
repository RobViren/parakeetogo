#include <algorithm>
#include <cstdio>
#include <cstring>
#if defined(__x86_64__)
#include <cpuid.h>
#elif defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#include "src/tern.h"

namespace pk {

#define PK_DECL(ns)                                                         \
  namespace ns {                                                            \
  void Gemm(const TernW& w, const QAct& a, float* y, int rb, int R, int t0, int t1);     \
  void QuantRows(const float* x, int C, uint8_t* q, float* sx, int t0, int t1); \
  }
#if defined(__x86_64__)
PK_DECL(vnni)
PK_DECL(avx2)
static const bool kVnni = [] {
  const char* e = std::getenv("PK_ISA");
  return e ? !std::strcmp(e, "vnni") : [] {
    unsigned a, b, c, d;
    return __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a >> 4 & 1);  // AVX-VNNI; clang 18 lacks the builtin name
  }();
}();
static const auto kGemm = kVnni ? vnni::Gemm : avx2::Gemm;
static const auto kQuantRows = kVnni ? vnni::QuantRows : avx2::QuantRows;
const char* TernIsa() { return kVnni ? "avx2+vnni" : "avx2"; }
#elif defined(__aarch64__)
PK_DECL(neon)
static const auto kGemm = [] {
  if (!(getauxval(AT_HWCAP) & HWCAP_ASIMDDP)) fprintf(stderr, "this CPU lacks the NEON dot product extension\n"), exit(1);
  return neon::Gemm;
}();
static const auto kQuantRows = neon::QuantRows;
const char* TernIsa() { return "neon+dotprod"; }
#endif
static const int kChunk = std::getenv("PK_CHUNK") ? atoi(std::getenv("PK_CHUNK")) : 48;

TernW PackTern(const uint8_t* qweight, const uint16_t* scales_f16, int out, int in) {
  TernW p;
  p.out = out, p.in = in, p.G = in / kGroup;
  const int RB = out / kRows, nb = (in + 4) / 5;
  p.w = Alloc<int8_t>(size_t(out) * in);
  p.init = Alloc<int32_t>(size_t(p.G) * out);
  p.scale = Alloc<float>(size_t(p.G) * out);
  static const auto lut = [] {
    std::vector<int8_t> l(256 * 5);
    for (int b = 0; b < 256; ++b)
      for (int i = 0, v = b; i < 5; ++i, v /= 3) l[b * 5 + i] = int8_t(v % 3 - 1);
    return l;
  }();
  std::vector<int8_t> row(size_t(nb) * 5);
  for (int i = 0; i < out; ++i) {
    for (int b = 0; b < nb; ++b) std::memcpy(&row[b * 5], &lut[qweight[size_t(i) * nb + b] * 5], 5);
    for (int g = 0; g < p.G; ++g) {
      const size_t o = size_t(g) * RB + i / kRows;
      int sum = 0;
      for (int s = 0; s < 32; ++s) {
        const int8_t* c = &row[g * kGroup + s * 4];
        std::memcpy(p.w.get() + o * 1024 + s * 32 + (i % kRows) * 4, c, 4);
        sum += c[0] + c[1] + c[2] + c[3];
      }
      p.init[o * 8 + i % kRows] = -128 * sum;
      p.scale[o * 8 + i % kRows] = F16(scales_f16[size_t(i) * p.G + g]);
    }
  }
  return p;
}

void QuantizeRows(const float* x, QAct& a, int t0, int t1) {
  kQuantRows(x, a.C, a.q.get(), a.sx.get(), t0, t1);
}

void Quantize(Pool& pool, const float* x, QAct& a) {
  pool.For((a.T + 15) / 16, [&](int c) { QuantizeRows(x, a, c * 16, std::min(a.T, c * 16 + 16)); }, 1);
}

// Work items are (row tile, frame chunk), tile-major, so the cores sharing a tile pull its weights from L3 once.
void TernGemm(Pool& pool, const TernW& w, const QAct& a, float* y) {
  const int RB = w.out / kRows, n12 = RB / 12, n4 = RB % 12 / 4, n1 = RB % 4, chunks = (a.T + kChunk - 1) / kChunk;
  pool.For((n12 + n4 + n1) * chunks, [&](int j) {
    const int tile = j / chunks, t0 = j % chunks * kChunk, t1 = std::min(a.T, t0 + kChunk);
    const int R = tile < n12 ? 12 : tile < n12 + n4 ? 4 : 1;
    const int rb = tile < n12 ? tile * 12 : tile < n12 + n4 ? n12 * 12 + (tile - n12) * 4 : n12 * 12 + n4 * 4 + tile - n12 - n4;
    kGemm(w, a, y, rb, R, t0, t1);
  }, 1);
}

}  // namespace pk
