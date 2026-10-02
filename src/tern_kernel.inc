#include <immintrin.h>

#include "src/tern.h"

namespace pk::PK_NS {

#define PK_ROWS1(F) F(0)
#define PK_ROWS4(F) F(0) F(1) F(2) F(3)
#define PK_ROWS12(F) PK_ROWS4(F) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11)

#define PK_WV(r) _mm256_load_si256((const __m256i*)(w + r * 1024 + s * 32))
#define PK_INIT(r) _mm256_load_si256((const __m256i*)(init + r * 8))
#ifdef PK_VNNI
#define PK_BEGIN(r) __m256i a##r = PK_INIT(r);
#define PK_STEP(r) a##r = _mm256_dpbusd_avx_epi32(a##r, xb, PK_WV(r));
#define PK_END(r)
#else
#define PK_BEGIN(r) __m256i a##r = _mm256_setzero_si256();
#define PK_STEP(r) a##r = _mm256_add_epi16(a##r, _mm256_maddubs_epi16(xb, PK_WV(r)));
#define PK_END(r) a##r = _mm256_add_epi32(_mm256_madd_epi16(a##r, _mm256_set1_epi16(1)), PK_INIT(r));
#endif
#define PK_OUT(r)                                                                                    \
  {                                                                                                  \
    const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(a##r), _mm256_load_ps(scale + r * 8));         \
    _mm256_storeu_ps(y + r * 8,                                                                      \
                     first ? _mm256_mul_ps(v, sxb) : _mm256_fmadd_ps(v, sxb, _mm256_loadu_ps(y + r * 8))); \
  }

#define PK_TILE(R)                                                                                       \
  static void Tile##R(const int8_t* w, const int32_t* init, const float* scale, const uint8_t* x,        \
                      size_t xstride, const float* sx, size_t sxstride, float* y, size_t ystride, int T, \
                      bool first) {                                                                      \
    for (int t = 0; t < T; ++t, x += xstride, sx += sxstride, y += ystride) {                            \
      PK_ROWS##R(PK_BEGIN)                                                                               \
      for (int s = 0; s < 32; ++s) {                                                                     \
        const __m256i xb = _mm256_set1_epi32(*(const int32_t*)(x + 4 * s));                              \
        PK_ROWS##R(PK_STEP)                                                                              \
      }                                                                                                  \
      PK_ROWS##R(PK_END)                                                                                 \
      const __m256 sxb = _mm256_broadcast_ss(sx);                                                        \
      PK_ROWS##R(PK_OUT)                                                                                 \
    }                                                                                                    \
  }
PK_TILE(12)
PK_TILE(4)
PK_TILE(1)

// Rows [rb, rb + R) of 8-row blocks, frames [t0, t1). R is 12, 4 or 1.
void Gemm(const TernW& w, const QAct& a, float* y, int rb, int R, int t0, int t1) {
  const int RB = w.out / kRows;
  const auto tile = R == 12 ? Tile12 : R == 4 ? Tile4 : Tile1;
  for (int g = 0; g < w.G; ++g) {
    const size_t o = size_t(g) * RB + rb;
    if (g + 1 < w.G)
      for (int i = 0; i < R * 1024; i += 64) _mm_prefetch((const char*)(w.w.get() + (o + RB) * 1024 + i), _MM_HINT_T0);
    tile(w.w.get() + o * 1024, w.init.get() + o * 8, w.scale.get() + o * 8, a.q.get() + size_t(t0) * a.C + g * kGroup,
         a.C, a.sx.get() + size_t(t0) * w.G + g, w.G, y + size_t(t0) * w.out + rb * kRows, w.out, t1 - t0, g == 0);
  }
}

void QuantRows(const float* x, int C, uint8_t* q, float* sx, int t0, int t1) {
  const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256i fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7), off = _mm256_set1_epi8(char(0x80));
  const int G = C / kGroup;
  for (int t = t0; t < t1; ++t)
    for (int g = 0; g < G; ++g) {
      const float* p = x + size_t(t) * C + g * kGroup;
      __m256 m = _mm256_setzero_ps();
      for (int i = 0; i < kGroup; i += 8) m = _mm256_max_ps(m, _mm256_and_ps(absmask, _mm256_loadu_ps(p + i)));
      __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
      m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
      const float mx = _mm_cvtss_f32(_mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1)));
      const float s = mx > 0 ? mx / 127.f : 1.f;
      sx[size_t(t) * G + g] = s;
      const __m256 inv = _mm256_set1_ps(1.f / s);
      auto cv = [&](int i) { return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i), inv)); };
      for (int i = 0; i < kGroup; i += 32) {
        const __m256i b = _mm256_packs_epi16(_mm256_packs_epi32(cv(i), cv(i + 8)), _mm256_packs_epi32(cv(i + 16), cv(i + 24)));
        _mm256_storeu_si256((__m256i*)(q + size_t(t) * C + g * kGroup + i),
                            _mm256_xor_si256(_mm256_permutevar8x32_epi32(b, fix), off));
      }
    }
}

}  // namespace pk::PK_NS
