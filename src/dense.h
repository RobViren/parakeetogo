#pragma once
#include "src/pk.h"
#include "src/simd.h"

namespace pk {

// Right operand of c = a * B with B [K][N], stored as ceil(N / 16) panels of [K][16] (zero padded) so the kernel
// streams one panel while 6 rows x 16 columns of c sit in registers. A linear layer is B = weight^T plus bias.
struct Packed {
  int K = 0, N = 0;
  size_t cap = 0;
  Buf<float> p, bias;
  int panels() const { return (N + 15) / 16; }
  void Resize(int k, int n);
};

// B[k][n] = src[k * sk + n * sn]. A linear weight [N][K] is (sk 1, sn K); a matrix [K][N] is (sk N, sn 1).
void Pack(Packed& b, int K, int N, const float* src, size_t sk, size_t sn, const float* bias = nullptr);

// Single thread: c[m][16 * p0 .. 16 * p1) for m < M, clipped to N columns.
void Gemm(const Packed& b, const float* a, size_t astride, int M, float* c, size_t cstride, int p0, int p1);

// c [M][N] = a [M][K] * B + bias, threaded over rows, or over panels when M is small.
void Linear(Pool& pool, const Packed& b, const float* a, int M, float* c);

// A linear layer applied to one row at a time, kept as the f16 the weights ship in. A single row product is bound
// by the bytes it streams, not by its multiplies. Panels and summation order match Packed under the one row kernel.
struct Half {
  int K = 0, N = 0;
  Buf<uint16_t> p;
  Buf<float> bias;
  int panels() const { return (N + 15) / 16; }
};
void PackHalf(Half& b, int K, int N, const uint16_t* weight, const float* bias);  // weight [N][K]

// The input columns (K <= 1280) a row product visits, split by k % 4: one accumulator set each. Zero inputs can be
// left out without changing any sum.
struct Picks {
  int n[4] = {};
  uint16_t k[4][320];
  Picks(const float* x, int K, bool skip_zeros) {
    for (int i = 0; i < K; ++i)
      if (!skip_zeros || x[i] != 0.f) k[i % 4][n[i % 4]++] = uint16_t(i);
  }
};

// c [16 * panels()] = x [K] * B + bias, 32-byte aligned, threaded over panels with a fixed split so each core keeps
// its share of the weights in its own L2 from one step to the next.
void Linear(Pool& pool, const Half& b, const float* x, const Picks& picks, float* c);

// First index of the largest value.
int Argmax(const float* x, int n);

// n is a multiple of 8 in all of these.
void Silu(float* x, size_t n);
void Sigmoid(float* x, size_t n);
void Glu(const float* value, const float* gate, float* y, size_t n);
void Softmax(float* x, size_t n);
void LayerNorm(const float* x, const float* w, const float* b, float* y, size_t n);

// y[c] = bias[c] + the sum over j < taps, in order, of x(j, c) * w[j][c], where x(j, c) is the 8 inputs of tap j at
// channel c. The sum stays in a register; a full 3x3 or 9 tap window gets its own unrolled copy.
template <class X>
[[gnu::always_inline]] static inline void TapsN(const float* bias, const float* const* w, int taps, X&& x, float* y,
                                                int C, bool relu) {
  for (int c = 0; c < C; c += 8) {
    F8 acc = Load8(bias + c);
    for (int j = 0; j < taps; ++j) acc = acc + x(j, c) * Load8(w[j] + c);
    Store8(relu ? Max(Zero8(), acc) : acc, y + c);
  }
}
template <class X>
static inline void Taps(const float* bias, const float* const* w, int taps, X&& x, float* y, int C, bool relu) {
  if (taps == 9) TapsN(bias, w, 9, x, y, C, relu);
  else TapsN(bias, w, taps, x, y, C, relu);
}

}  // namespace pk
