#pragma once
#include <hwy/highway.h>

#include <algorithm>
#include <cstdint>

#if HWY_ARCH_X86
#include <immintrin.h>
#endif

namespace pk {
namespace hn = hwy::HWY_NAMESPACE;

// Eight floats: one 256-bit vector, or two 128-bit halves where that is the widest there is (NEON). Either way the
// data layouts, the summation order and so every result are the same, and the 16 or 32 registers fill alike.
#if HWY_MAX_BYTES >= 32
using DF = hn::FixedTag<float, 8>;
constexpr int kParts = 1;
#else
using DF = hn::FixedTag<float, 4>;
constexpr int kParts = 2;
#endif
constexpr int kLanes = 8 / kParts;
using VF = hn::Vec<DF>;

struct F8 {
  VF v[kParts];
};

template <class F> HWY_INLINE F8 Each(F&& f) {
  F8 r;
  for (int i = 0; i < kParts; ++i) r.v[i] = f(i);
  return r;
}

HWY_INLINE F8 Zero8() { return Each([](int) { return hn::Zero(DF()); }); }
HWY_INLINE F8 Set8(float x) { return Each([&](int) { return hn::Set(DF(), x); }); }
HWY_INLINE F8 Load8(const float* p) { return Each([&](int i) { return hn::LoadU(DF(), p + i * kLanes); }); }
HWY_INLINE void Store8(F8 a, float* p) {
  for (int i = 0; i < kParts; ++i) hn::StoreU(a.v[i], DF(), p + i * kLanes);
}
// Eight f16 values widened to f32.
HWY_INLINE F8 LoadHalf8(const uint16_t* p) {
  const hn::Rebind<uint16_t, DF> du;
  const hn::Rebind<hwy::float16_t, DF> dh;
  return Each([&](int i) { return hn::PromoteTo(DF(), hn::BitCast(dh, hn::LoadU(du, p + i * kLanes))); });
}

HWY_INLINE F8 operator+(F8 a, F8 b) { return Each([&](int i) { return hn::Add(a.v[i], b.v[i]); }); }
HWY_INLINE F8 operator-(F8 a, F8 b) { return Each([&](int i) { return hn::Sub(a.v[i], b.v[i]); }); }
HWY_INLINE F8 operator*(F8 a, F8 b) { return Each([&](int i) { return hn::Mul(a.v[i], b.v[i]); }); }
HWY_INLINE F8 Max(F8 a, F8 b) { return Each([&](int i) { return hn::Max(a.v[i], b.v[i]); }); }
// a * b + c in one rounding.
HWY_INLINE F8 MulAdd(F8 a, F8 b, F8 c) { return Each([&](int i) { return hn::MulAdd(a.v[i], b.v[i], c.v[i]); }); }

// ((t0 + t4) + (t2 + t6)) + ((t1 + t5) + (t3 + t7)).
HWY_INLINE float Sum8(F8 a) {
  const hn::FixedTag<float, 4> d4;
#if HWY_MAX_BYTES >= 32
  const auto s = hn::Add(hn::LowerHalf(d4, a.v[0]), hn::UpperHalf(d4, a.v[0]));
#else
  const auto s = hn::Add(a.v[0], a.v[1]);
#endif
  alignas(16) float t[4];
  hn::Store(s, d4, t);
  return (t[0] + t[2]) + (t[1] + t[3]);
}

HWY_INLINE float Peak8(F8 a) {
  float m = hn::ReduceMax(DF(), a.v[0]);
  for (int i = 1; i < kParts; ++i) m = std::max(m, hn::ReduceMax(DF(), a.v[i]));
  return m;
}

HWY_INLINE VF Exp(VF x) {
  const DF d;
  const hn::RebindToSigned<DF> di;
  x = hn::Min(hn::Max(x, hn::Set(d, -87.33654f)), hn::Set(d, 88.f));
  const VF n = hn::Round(hn::Mul(x, hn::Set(d, 1.44269504088896341f)));
  VF r = hn::NegMulAdd(n, hn::Set(d, 0.693359375f), x);
  r = hn::NegMulAdd(n, hn::Set(d, -2.12194440e-4f), r);
  VF p = hn::Set(d, 1.9875691500e-4f);
  p = hn::MulAdd(p, r, hn::Set(d, 1.3981999507e-3f));
  p = hn::MulAdd(p, r, hn::Set(d, 8.3334519073e-3f));
  p = hn::MulAdd(p, r, hn::Set(d, 4.1665795894e-2f));
  p = hn::MulAdd(p, r, hn::Set(d, 1.6666665459e-1f));
  p = hn::MulAdd(p, r, hn::Set(d, 5.0000001201e-1f));
  p = hn::Add(hn::MulAdd(p, hn::Mul(r, r), r), hn::Set(d, 1.f));
  const auto e = hn::ShiftLeft<23>(hn::Add(hn::ConvertTo(di, n), hn::Set(di, 127)));
  return hn::Mul(p, hn::BitCast(d, e));
}
HWY_INLINE F8 Exp8(F8 x) { return Each([&](int i) { return Exp(x.v[i]); }); }
HWY_INLINE F8 Sigmoid8(F8 x) {
  const VF one = hn::Set(DF(), 1.f);
  return Each([&](int i) { return hn::Div(one, hn::Add(one, Exp(hn::Neg(x.v[i])))); });
}

// Denormal results and inputs become zero while this lives.
struct FlushDenormals {
#if HWY_ARCH_X86
  unsigned saved = _mm_getcsr();
  FlushDenormals() { _mm_setcsr(saved | 0x8040); }
  ~FlushDenormals() { _mm_setcsr(saved); }
#elif HWY_ARCH_ARM_A64
  uint64_t saved;
  FlushDenormals() {
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved));
    __asm__ volatile("msr fpcr, %0" : : "r"(saved | (1ull << 24)));
  }
  ~FlushDenormals() { __asm__ volatile("msr fpcr, %0" : : "r"(saved)); }
#endif
};

}  // namespace pk
