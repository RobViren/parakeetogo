#include <algorithm>
#include <cmath>
#include <cstring>

#include "src/dense.h"

namespace pk {

void Packed::Resize(int k, int n) {
  K = k, N = n;
  const size_t need = size_t(panels()) * K * 16;
  if (need > cap) p = Alloc<float>(cap = need);
}

void Pack(Packed& b, int K, int N, const float* src, size_t sk, size_t sn, const float* bias) {
  b.Resize(K, N);
  float* p = b.p.get();
  for (int n0 = 0; n0 < N; n0 += 16)
    for (int k = 0; k < K; ++k, p += 16)
      for (int j = 0; j < 16; ++j) p[j] = n0 + j < N ? src[k * sk + (n0 + j) * sn] : 0.f;
  if (!bias) return;
  b.bias = Alloc<float>(b.panels() * 16);
  std::memset(b.bias.get(), 0, b.panels() * 64);
  std::memcpy(b.bias.get(), bias, N * 4);
}

// R rows of a against one panel. U independent accumulator sets over k keep a single row from being bound by the
// FMA latency chain.
template <int R, int U>
static inline void Kernel(const float* p, int K, const float* a, size_t astride, const float* bias, float* out) {
  F8 acc[U][R][2];
  for (int u = 0; u < U; ++u)
    for (int r = 0; r < R; ++r) {
      acc[u][r][0] = bias && !u ? Load8(bias) : Zero8();
      acc[u][r][1] = bias && !u ? Load8(bias + 8) : Zero8();
    }
  auto step = [&](int k, int u) {
    const F8 w0 = Load8(p + size_t(k) * 16), w1 = Load8(p + size_t(k) * 16 + 8);
    for (int r = 0; r < R; ++r) {
      const F8 x = Set8(a[r * astride + k]);
      acc[u][r][0] = MulAdd(x, w0, acc[u][r][0]);
      acc[u][r][1] = MulAdd(x, w1, acc[u][r][1]);
    }
  };
  int k = 0;
  for (; k + U <= K; k += U)
    for (int u = 0; u < U; ++u) step(k + u, u);
  for (; k < K; ++k) step(k, 0);
  for (int r = 0; r < R; ++r)
    for (int h = 0; h < 2; ++h) {
      F8 s = acc[0][r][h];
      for (int u = 1; u < U; ++u) s = s + acc[u][r][h];
      Store8(s, out + r * 16 + h * 8);
    }
}

void Gemm(const Packed& b, const float* a, size_t astride, int M, float* c, size_t cstride, int p0, int p1) {
  alignas(32) float out[6 * 16];
  constexpr int kBlock = 48;
  for (int mb = 0; mb < M; mb += kBlock)
    for (int p = p0; p < p1; ++p) {
      const float* w = b.p.get() + size_t(p) * b.K * 16;
      const float* bias = b.bias ? b.bias.get() + p * 16 : nullptr;
      const int n = std::min(16, b.N - p * 16), me = std::min(M, mb + kBlock);
      int m = mb;
      for (; m + 6 <= me; m += 6) {
        Kernel<6, 1>(w, b.K, a + m * astride, astride, bias, out);
        for (int r = 0; r < 6; ++r) std::memcpy(c + (m + r) * cstride + p * 16, out + r * 16, n * 4);
      }
      for (; m < me; ++m) {
        Kernel<1, 4>(w, b.K, a + m * astride, astride, bias, out);
        std::memcpy(c + m * cstride + p * 16, out, n * 4);
      }
    }
}

void Linear(Pool& pool, const Packed& b, const float* a, int M, float* c) {
  const int P = b.panels(), n = pool.n;
  if (M >= 6 * n) {
    auto cut = [&](int i) { return i == n ? M : M * i / n / 6 * 6; };
    pool.Run([&](int i) {
      const int m0 = cut(i);
      Gemm(b, a + size_t(m0) * b.K, b.K, cut(i + 1) - m0, c + size_t(m0) * b.N, b.N, 0, P);
    });
  } else {
    pool.Run([&](int i) { Gemm(b, a, b.K, M, c, b.N, P * i / n, P * (i + 1) / n); });
  }
}

void PackHalf(Half& b, int K, int N, const uint16_t* weight, const float* bias) {
  b.K = K, b.N = N;
  b.p = Alloc<uint16_t>(size_t(b.panels()) * K * 16), b.bias = Alloc<float>(b.panels() * 16);
  uint16_t* p = b.p.get();
  for (int n0 = 0; n0 < N; n0 += 16)
    for (int k = 0; k < K; ++k, p += 16)
      for (int j = 0; j < 16; ++j) p[j] = n0 + j < N ? weight[size_t(n0 + j) * K + k] : 0;
  std::memset(b.bias.get(), 0, b.panels() * 64);
  std::memcpy(b.bias.get(), bias, N * 4);
}

static void Gemv(const Half& b, const float* x, const Picks& s, float* c, int p0, int p1) {
  const int common = *std::min_element(s.n, s.n + 4), most = *std::max_element(s.n, s.n + 4);
  for (int p = p0; p < p1; ++p) {
    const uint16_t* w = b.p.get() + size_t(p) * b.K * 16;
    F8 acc[4][2];
    for (int u = 0; u < 4; ++u) acc[u][0] = acc[u][1] = Zero8();
    acc[0][0] = Load8(b.bias.get() + p * 16), acc[0][1] = Load8(b.bias.get() + p * 16 + 8);
    auto step = [&](int u, int i) {
      const int k = s.k[u][i];
      const F8 v = Set8(x[k]);
      acc[u][0] = MulAdd(v, LoadHalf8(w + k * 16), acc[u][0]);
      acc[u][1] = MulAdd(v, LoadHalf8(w + k * 16 + 8), acc[u][1]);
    };
    for (int i = 0; i < common; ++i)
      for (int u = 0; u < 4; ++u) step(u, i);
    for (int i = common; i < most; ++i)
      for (int u = 0; u < 4; ++u)
        if (i < s.n[u]) step(u, i);
    for (int h = 0; h < 2; ++h)
      Store8(acc[0][h] + acc[1][h] + acc[2][h] + acc[3][h], c + p * 16 + h * 8);
  }
}

void Linear(Pool& pool, const Half& b, const float* x, const Picks& picks, float* c) {
  const int P = b.panels(), n = pool.n;
  pool.Run([&](int i) { Gemv(b, x, picks, c, P * i / n, P * (i + 1) / n); });
}

int Argmax(const float* x, int n) {
  F8 m = Load8(x);
  int i = 8;
  for (; i + 8 <= n; i += 8) m = Max(m, Load8(x + i));
  float peak = Peak8(m);
  for (; i < n; ++i) peak = std::max(peak, x[i]);
  for (i = 0; i + 1 < n && x[i] != peak; ++i) {}
  return i;
}

void Silu(float* x, size_t n) {
  for (size_t i = 0; i < n; i += 8) {
    const F8 v = Load8(x + i);
    Store8(v * Sigmoid8(v), x + i);
  }
}

void Sigmoid(float* x, size_t n) {
  for (size_t i = 0; i < n; i += 8) Store8(Sigmoid8(Load8(x + i)), x + i);
}

void Glu(const float* value, const float* gate, float* y, size_t n) {
  for (size_t i = 0; i < n; i += 8) Store8(Load8(value + i) * Sigmoid8(Load8(gate + i)), y + i);
}

void Softmax(float* x, size_t n) {
  F8 m = Load8(x);
  for (size_t i = 8; i < n; i += 8) m = Max(m, Load8(x + i));
  const F8 peak = Set8(Peak8(m));
  F8 sum = Zero8();
  for (size_t i = 0; i < n; i += 8) {
    const F8 e = Exp8(Load8(x + i) - peak);
    sum = sum + e;
    Store8(e, x + i);
  }
  const F8 inv = Set8(1.f / Sum8(sum));
  for (size_t i = 0; i < n; i += 8) Store8(Load8(x + i) * inv, x + i);
}

void LayerNorm(const float* x, const float* w, const float* b, float* y, size_t n) {
  F8 s = Zero8(), ss = Zero8();
  for (size_t i = 0; i < n; i += 8) s = s + Load8(x + i);
  const F8 mean = Set8(Sum8(s) / n);
  for (size_t i = 0; i < n; i += 8) {
    const F8 d = Load8(x + i) - mean;
    ss = MulAdd(d, d, ss);
  }
  const F8 inv = Set8(1.f / std::sqrt(Sum8(ss) / n + 1e-5f));
  for (size_t i = 0; i < n; i += 8) {
    const F8 d = (Load8(x + i) - mean) * inv;
    Store8(MulAdd(d, Load8(w + i), Load8(b + i)), y + i);
  }
}

}  // namespace pk
