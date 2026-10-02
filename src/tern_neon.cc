#include <arm_neon.h>

#include "src/tern.h"

namespace pk::neon {

#define PK_ROWS1(F) F(0)
#define PK_ROWS4(F) F(0) F(1) F(2) F(3)
#define PK_ROWS12(F) PK_ROWS4(F) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11)

// The signed dot product takes the activations as q - 128, so the +128 offset and its init term drop out and the
// int32 sums equal the x86 kernels'. A 32-byte weight vector is two registers here: 24 accumulators for 12 blocks.
#define PK_BEGIN(r) int32x4_t a##r = vdupq_n_s32(0), b##r = vdupq_n_s32(0);
#define PK_STEP(r)                                         \
  a##r = vdotq_s32(a##r, vld1q_s8(w + r * 1024 + s * 32), xb); \
  b##r = vdotq_s32(b##r, vld1q_s8(w + r * 1024 + s * 32 + 16), xb);
#define PK_HALF(acc, o)                                                         \
  {                                                                             \
    const float32x4_t v = vmulq_f32(vcvtq_f32_s32(acc), vld1q_f32(scale + o));  \
    vst1q_f32(y + o, first ? vmulq_f32(v, sxb) : vfmaq_f32(vld1q_f32(y + o), v, sxb)); \
  }
#define PK_OUT(r) PK_HALF(a##r, r * 8) PK_HALF(b##r, r * 8 + 4)

#define PK_TILE(R)                                                                                    \
  static void Tile##R(const int8_t* w, const float* scale, const uint8_t* x, size_t xstride, const float* sx, \
                      size_t sxstride, float* y, size_t ystride, int T, bool first) {                 \
    for (int t = 0; t < T; ++t, x += xstride, sx += sxstride, y += ystride) {                         \
      PK_ROWS##R(PK_BEGIN)                                                                            \
      for (int s = 0; s < 32; ++s) {                                                                  \
        uint32_t four;                                                                                \
        __builtin_memcpy(&four, x + 4 * s, 4);                                                        \
        const int8x16_t xb = vreinterpretq_s8_u32(vdupq_n_u32(four ^ 0x80808080u));                   \
        PK_ROWS##R(PK_STEP)                                                                           \
      }                                                                                               \
      const float32x4_t sxb = vdupq_n_f32(*sx);                                                       \
      PK_ROWS##R(PK_OUT)                                                                              \
    }                                                                                                 \
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
      for (int i = 0; i < R * 1024; i += 64) __builtin_prefetch(w.w.get() + (o + RB) * 1024 + i);
    tile(w.w.get() + o * 1024, w.scale.get() + o * 8, a.q.get() + size_t(t0) * a.C + g * kGroup, a.C,
         a.sx.get() + size_t(t0) * w.G + g, w.G, y + size_t(t0) * w.out + rb * kRows, w.out, t1 - t0, g == 0);
  }
}

void QuantRows(const float* x, int C, uint8_t* q, float* sx, int t0, int t1) {
  const int G = C / kGroup;
  for (int t = t0; t < t1; ++t)
    for (int g = 0; g < G; ++g) {
      const float* p = x + size_t(t) * C + g * kGroup;
      float32x4_t m = vdupq_n_f32(0.f);
      for (int i = 0; i < kGroup; i += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(p + i)));
      const float mx = vmaxvq_f32(m);
      const float s = mx > 0 ? mx / 127.f : 1.f;
      sx[size_t(t) * G + g] = s;
      const float32x4_t inv = vdupq_n_f32(1.f / s);
      auto cv = [&](int i) { return vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p + i), inv))); };
      for (int i = 0; i < kGroup; i += 16) {
        const int8x16_t b = vcombine_s8(vqmovn_s16(vcombine_s16(cv(i), cv(i + 4))),
                                        vqmovn_s16(vcombine_s16(cv(i + 8), cv(i + 12))));
        vst1q_u8(q + size_t(t) * C + g * kGroup + i, veorq_u8(vreinterpretq_u8_s8(b), vdupq_n_u8(0x80)));
      }
    }
}

}  // namespace pk::neon
