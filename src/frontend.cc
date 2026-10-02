#include <algorithm>
#include <cmath>

#include "src/model.h"

namespace pk {

constexpr int kWin = 400, kFft = 512, kBins = kFft / 2 + 1;

struct MelTables {
  float window[kFft] = {};
  std::vector<float> bank;  // [mel][bin]
  int first[kMels], last[kMels];
  double twiddle_re[kFft], twiddle_im[kFft];  // per stage, contiguous: the stage of half width h starts at h - 1
  int reversed[kFft];
  MelTables() : bank(kMels * kBins) {
    for (int i = 0; i < kWin; ++i) window[(kFft - kWin) / 2 + i] = float(0.5 - 0.5 * std::cos(2 * M_PI * i / (kWin - 1)));
    auto to_hz = [](double mel) { return mel >= 15 ? 1000 * std::exp((mel - 15) * (std::log(6.4) / 27)) : 200 * mel / 3; };
    const double top = 15 + std::log(kSampleRate / 2 / 1000.0) * (27 / std::log(6.4)), step = top / (kMels + 1);
    double edges[kMels + 2];
    for (int i = 0; i < kMels + 2; ++i) edges[i] = to_hz(i == kMels + 1 ? top : i * step);
    for (int m = 0; m < kMels; ++m) {
      first[m] = kBins, last[m] = 0;
      for (int b = 0; b < kBins; ++b) {
        const double hz = b * (kSampleRate / 2.0) / (kBins - 1);
        const double rise = (hz - edges[m]) / (edges[m + 1] - edges[m]), fall = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
        const double v = std::max(0.0, std::min(rise, fall)) * (2 / (edges[m + 2] - edges[m]));
        bank[m * kBins + b] = float(v);
        if (v > 0) first[m] = std::min(first[m], b), last[m] = b + 1;
      }
    }
    for (int half = 1; half < kFft; half *= 2)
      for (int i = 0; i < half; ++i) {
        const double angle = -2 * M_PI * (i * (kFft / 2 / half)) / kFft;
        twiddle_re[half - 1 + i] = std::cos(angle), twiddle_im[half - 1 + i] = std::sin(angle);
      }
    for (int i = 0; i < kFft; ++i) {
      int r = 0;
      for (int b = 0; b < 9; ++b) r |= ((i >> b) & 1) << (8 - b);
      reversed[i] = r;
    }
  }
};

std::vector<float> LogMel(Pool& pool, const float* pcm, size_t n, int& frames, int& valid) {
  Clock clock(kFrontend);
  static const MelTables tables;
  valid = int(n / kHop), frames = valid + 1;
  std::vector<float> x(n + kFft, 0.f), mel(size_t(frames) * kMels);
  float* padded = x.data() + kFft / 2;
  constexpr int kGrain = 1 << 14;
  pool.For(int((n + kGrain - 1) / kGrain), [&](int g) {
    for (size_t i = size_t(g) * kGrain, e = std::min(n, i + kGrain); i < e; ++i)
      padded[i] = i ? pcm[i] - 0.97f * pcm[i - 1] : pcm[0];
  }, 1);
  pool.For(frames, [&](int t) {
    alignas(32) double re[kFft], im[kFft] = {};
    float power[kBins];
    for (int i = 0; i < kFft; ++i) re[tables.reversed[i]] = x[size_t(t) * kHop + i] * tables.window[i];
    for (int half = 1; half < kFft; half *= 2) {
      const double* wr = tables.twiddle_re + half - 1;
      const double* wi = tables.twiddle_im + half - 1;
      for (int at = 0; at < kFft; at += 2 * half) {
        double* ar = re + at;
        double* ai = im + at;
        double* br = ar + half;
        double* bi = ai + half;
        if (half < 4) {
          // clang's vectorization of this 1 or 2 trip loop runs 10x slower than the scalar code.
#pragma clang loop vectorize(disable) interleave(disable)
          for (int i = 0; i < half; ++i) {
            const double tr = br[i] * wr[i] - bi[i] * wi[i], ti = br[i] * wi[i] + bi[i] * wr[i];
            br[i] = ar[i] - tr, bi[i] = ai[i] - ti, ar[i] += tr, ai[i] += ti;
          }
          continue;
        }
        const hn::CappedTag<double, 4> d;
        for (int i = 0; i < half; i += int(hn::Lanes(d))) {
          const auto xr = hn::LoadU(d, br + i), xi = hn::LoadU(d, bi + i);
          const auto cr = hn::LoadU(d, wr + i), ci = hn::LoadU(d, wi + i);
          const auto tr = hn::Sub(hn::Mul(xr, cr), hn::Mul(xi, ci));
          const auto ti = hn::Add(hn::Mul(xr, ci), hn::Mul(xi, cr));
          const auto yr = hn::LoadU(d, ar + i), yi = hn::LoadU(d, ai + i);
          hn::StoreU(hn::Add(yr, tr), d, ar + i), hn::StoreU(hn::Add(yi, ti), d, ai + i);
          hn::StoreU(hn::Sub(yr, tr), d, br + i), hn::StoreU(hn::Sub(yi, ti), d, bi + i);
        }
      }
    }
    for (int b = 0; b < kBins; ++b) {
      const float r = float(re[b]), i = float(im[b]);
      power[b] = r * r + i * i;
    }
    for (int m = 0; m < kMels; ++m) {
      float s = 0;
      for (int b = tables.first[m]; b < tables.last[m]; ++b) s += power[b] * tables.bank[m * kBins + b];
      mel[size_t(t) * kMels + m] = std::log(s + 0x1p-24f);
    }
  }, 32);
  // Per mel statistics accumulate over frames in order; 16 mels per item keeps the walk on one cache line per frame.
  pool.For(kMels / 16, [&](int g) {
    double sum[16] = {}, squares[16] = {}, mean[16];
    float fmean[16], spread[16];
    float* column = mel.data() + g * 16;
    for (int t = 0; t < valid; ++t)
      for (int j = 0; j < 16; ++j) sum[j] += column[size_t(t) * kMels + j];
    for (int j = 0; j < 16; ++j) mean[j] = sum[j] / valid;
    for (int t = 0; t < valid; ++t)
      for (int j = 0; j < 16; ++j) {
        const double d = column[size_t(t) * kMels + j] - mean[j];
        squares[j] += d * d;
      }
    for (int j = 0; j < 16; ++j) {
      fmean[j] = float(mean[j]);
      spread[j] = float(std::sqrt(squares[j] / (valid - 1))) + 1e-5f;
    }
    for (int t = 0; t < valid; ++t) {
      float* row = column + size_t(t) * kMels;
      for (int j = 0; j < 16; ++j) row[j] = (row[j] - fmean[j]) / spread[j];
    }
    std::fill_n(column + size_t(valid) * kMels, 16, 0.f);
  }, 1);
  return mel;
}

}  // namespace pk
