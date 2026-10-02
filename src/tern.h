#pragma once
#include "src/pk.h"

namespace pk {

constexpr int kGroup = 128;
constexpr int kRows = 8;

// Ternary linear packed for the int8 kernel. Rows go in blocks of 8 so one 32-byte vector holds 4 consecutive
// input columns for each of 8 output rows and a dot-product instruction lands one int32 per output row.
// w: [G][out/8][32][8][4] in {-1,0,1}. init: [G][out/8][8] = -128 * sum of the group's weights, which cancels
// the +128 offset on the unsigned activations. scale: [G][out/8][8].
struct TernW {
  int out = 0, in = 0, G = 0;
  Buf<int8_t> w;
  Buf<int32_t> init;
  Buf<float> scale;
};

// Activations quantized per (frame, 128-column group): q = round(x / sx) + 128.
struct QAct {
  int T = 0, C = 0;
  Buf<uint8_t> q;
  Buf<float> sx;
  QAct() = default;
  QAct(int T, int C) : T(T), C(C), q(Alloc<uint8_t>(size_t(T) * C)), sx(Alloc<float>(size_t(T) * C / kGroup)) {}
};

TernW PackTern(const uint8_t* qweight, const uint16_t* scales_f16, int out, int in);
void Quantize(Pool& pool, const float* x, QAct& a);
void QuantizeRows(const float* x, QAct& a, int t0, int t1);
void TernGemm(Pool& pool, const TernW& w, const QAct& a, float* y);
const char* TernIsa();

}  // namespace pk
