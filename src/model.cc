#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "src/model.h"

namespace pk {

static const char* const kTernNames[kTerns] = {
    "feed_forward1.linear1", "feed_forward1.linear2", "self_attn.q_proj",      "self_attn.k_proj",
    "self_attn.v_proj",      "self_attn.relative_k_proj", "self_attn.o_proj",  "conv.pointwise_conv1",
    "conv.pointwise_conv2",  "feed_forward2.linear1", "feed_forward2.linear2"};

Model::Model(const std::string& dir, int threads) : pool(threads) {
  size_t blob_size;
  const uint8_t* blob = MapFile((dir + "/model.safetensors").c_str(), &blob_size);
  const SafeTensors st(blob);
  LoadTokenizer(dir + "/tokenizer.json");

  pool.For(kLayers * kTerns, [&](int i) {
    const std::string name = "encoder.layers." + std::to_string(i / kTerns) + "." + kTernNames[i % kTerns];
    const Tensor& q = st[name + ".qweight"];
    const Tensor& s = st[name + ".scales"];
    layers[i / kTerns].t[i % kTerns] = PackTern(q.data, s.f16(), q.shape[0], s.shape[1] * kGroup);
  });
  for (int i = 0; i < kLayers; ++i) {
    Layer& L = layers[i];
    const std::string p = "encoder.layers." + std::to_string(i) + ".";
    auto norm = [&](const char* name) { return Norm{st.F32(p + name + ".weight"), st.F32(p + name + ".bias")}; };
    L.norm_ff1 = norm("norm_feed_forward1"), L.norm_att = norm("norm_self_att"), L.norm_conv = norm("norm_conv");
    L.norm_ff2 = norm("norm_feed_forward2"), L.norm_out = norm("norm_out");
    L.bias_u = st.F32(p + "self_attn.bias_u"), L.bias_v = st.F32(p + "self_attn.bias_v");
    const auto w = st.F32(p + "conv.depthwise_conv.weight"), gamma = st.F32(p + "conv.norm.weight");
    const auto beta = st.F32(p + "conv.norm.bias"), mean = st.F32(p + "conv.norm.running_mean");
    const auto var = st.F32(p + "conv.norm.running_var");
    L.dw.resize(kConvKernel * kHidden), L.dw_bias.resize(kHidden);
    for (int c = 0; c < kHidden; ++c) {
      const float scale = gamma[c] / std::sqrt(var[c] + 1e-5f);
      L.dw_bias[c] = beta[c] - mean[c] * scale;
      for (int j = 0; j < kConvKernel; ++j) L.dw[j * kHidden + c] = w[c * kConvKernel + j] * scale;
    }
  }

  auto linear = [&](Packed& b, const std::vector<float>& w, const std::vector<float>& bias) {
    Pack(b, w.size() / bias.size(), bias.size(), w.data(), 1, w.size() / bias.size(), bias.data());
  };
  auto taps = [&](const std::string& name) {
    const auto w = st.F32(name);
    std::vector<float> t(w.size());
    for (int c = 0; c < kSubChannels; ++c)
      for (int k = 0; k < 9; ++k) t[k * kSubChannels + c] = w[c * 9 + k];
    return t;
  };
  const std::string sub = "encoder.subsampling.";
  conv0 = taps(sub + "layers.0.weight"), conv0_bias = st.F32(sub + "layers.0.bias");
  for (int i = 0; i < 2; ++i) {
    const std::string dw = sub + "layers." + std::to_string(2 + 3 * i), pw = sub + "layers." + std::to_string(3 + 3 * i);
    sub_dw[i] = taps(dw + ".weight"), sub_dw_bias[i] = st.F32(dw + ".bias");
    linear(sub_pw[i], st.F32(pw + ".weight"), st.F32(pw + ".bias"));
  }
  {
    // The reference flattens [channel][mel]; activations here are [mel][channel].
    const auto w = st.F32(sub + "linear.weight");
    std::vector<float> t(w.size());
    for (int o = 0; o < kHidden; ++o)
      for (int c = 0; c < kSubChannels; ++c)
        for (int m = 0; m < 16; ++m) t[o * 4096 + m * kSubChannels + c] = w[o * 4096 + c * 16 + m];
    linear(sub_linear, t, st.F32(sub + "linear.bias"));
  }
  linear(enc_proj, st.F32("encoder_projector.weight"), st.F32("encoder_projector.bias"));

  linear(vad_proj, st.F32("vad_head.proj.weight"), st.F32("vad_head.proj.bias"));
  {
    const auto w = st.F32("vad_head.ctx.weight");
    std::vector<float> t(w.size());
    for (int o = 0; o < 128; ++o)
      for (int c = 0; c < 128; ++c)
        for (int j = 0; j < 5; ++j) t[o * 640 + j * 128 + c] = w[(o * 128 + c) * 5 + j];
    linear(vad_ctx, t, st.F32("vad_head.ctx.bias"));
  }
  vad_out = st.F32("vad_head.out.weight"), vad_out_bias = st.F32("vad_head.out.bias")[0];

  auto half = [&](const std::string& name) {
    const Tensor& t = st[name];
    if (t.dtype != 'h') fprintf(stderr, "%s is not F16\n", name.c_str()), exit(1);
    return t.f16();
  };
  embedding = st.F32("decoder.embedding.weight");
  for (int i = 0; i < 2; ++i) {
    const std::string p = "decoder.lstm.", l = "_l" + std::to_string(i);
    const uint16_t* ih = half(p + "weight_ih" + l);
    const uint16_t* hh = half(p + "weight_hh" + l);
    auto bias = st.F32(p + "bias_ih" + l);
    const auto bias_hh = st.F32(p + "bias_hh" + l);
    std::vector<uint16_t> w(bias.size() * 2 * kEncoded);
    for (size_t o = 0; o < bias.size(); ++o) {
      std::memcpy(&w[o * 2 * kEncoded], &ih[o * kEncoded], kEncoded * 2);
      std::memcpy(&w[o * 2 * kEncoded + kEncoded], &hh[o * kEncoded], kEncoded * 2);
      bias[o] += bias_hh[o];
    }
    PackHalf(cell[i], 2 * kEncoded, int(bias.size()), w.data(), bias.data());
  }
  PackHalf(dec_proj, kEncoded, kEncoded, half("decoder.decoder_projector.weight"),
           st.F32("decoder.decoder_projector.bias").data());
  PackHalf(joint, kEncoded, kVocab + kDurations, half("joint.head.weight"), st.F32("joint.head.bias").data());
  munmap(const_cast<uint8_t*>(blob), blob_size);
}

void Model::Reserve(Act& a, const float* x, int T, int C) {
  a.x = x, a.q.T = T, a.q.C = C;
  if (float_path || size_t(T) * C <= a.cap) return;
  a.cap = size_t(T) * C;
  a.q.q = Alloc<uint8_t>(a.cap), a.q.sx = Alloc<float>(a.cap / kGroup);
}

void Model::Prepare(Act& a, const float* x, int T, int C) {
  Reserve(a, x, T, C);
  if (float_path) return;
  Clock clock(kQuantize);
  Quantize(pool, x, a.q);
}

void Model::Apply(const TernW& w, const Act& a, float* y) {
  Clock clock(kTernGemm);
  if (!float_path) return TernGemm(pool, w, a.q, y);
  unpacked.Resize(w.in, w.out);
  const int RB = w.out / kRows;
  pool.For(w.out / 16, [&](int p) {
    float* d = unpacked.p.get() + size_t(p) * w.in * 16;
    for (int k = 0; k < w.in; ++k)
      for (int j = 0; j < 16; ++j) {
        const size_t o = size_t(k / kGroup) * RB + 2 * p + j / 8;
        d[k * 16 + j] = w.w[o * 1024 + (k % kGroup / 4) * 32 + (j % 8) * 4 + k % 4] * w.scale[o * 8 + j % 8];
      }
  });
  Linear(pool, unpacked, a.x, a.q.T, y);
}

// One output row [W][channel] of a stride 2 depthwise 3x3 from its three input rows [2 * W][channel]. A row past
// the valid length is zero.
static void DepthwiseRow(const float* const rows[3], int W, bool live, const std::vector<float>& dw,
                         const std::vector<float>& dw_bias, float* out) {
  const int C = kSubChannels;
  if (!live) return void(std::fill_n(out, W * C, 0.f));
  for (int w = 0; w < W; ++w) {
    const float* x[9];
    const float* t[9];
    int taps = 0;
    for (int ky = 0; ky < 3; ++ky)
      for (int kx = w ? 0 : 1; kx < 3; ++kx) {
        x[taps] = rows[ky] + (2 * w - 1 + kx) * C;
        t[taps++] = &dw[(ky * 3 + kx) * C];
      }
    Taps(dw_bias.data(), t, taps, [&](int j, int c) { return Load8(x[j] + c); }, out + w * C, C, false);
  }
}

// Pointwise conv and ReLU over n rows [W][channel] starting at global row lo. Rows outside [0, total) are padding.
static void Pointwise(const Packed& pw, const float* in, int W, int lo, int n, int total, float* out) {
  const int C = kSubChannels;
  Gemm(pw, in, C, n * W, out, C, 0, pw.panels());
  for (int i = 0; i < n; ++i) {
    float* o = out + size_t(i) * W * C;
    const bool padding = lo + i < 0 || lo + i >= total;
    for (int j = 0; j < W * C; ++j) o[j] = padding ? 0.f : std::max(o[j], 0.f);
  }
}

// Rows past the valid length are zeroed after each stride 2 conv only (reference quirk), so the pointwise bias
// refills them and the next depthwise conv reads that row at its right edge. Each chunk of output rows computes
// its own small pyramid, and the first conv's rows live only in a three row ring feeding the depthwise conv, which
// keeps the 256 channel maps out of RAM and the chunk in L2.
std::vector<float> Model::Subsample(const float* mel, int T0, int valid, int& T3, int& v3) {
  Clock clock(kSubsampler);
  const int C = kSubChannels, T1 = (T0 + 1) / 2, T2 = (T1 + 1) / 2;
  const int v1 = (valid + 1) / 2, v2 = (v1 + 1) / 2;
  T3 = (T2 + 1) / 2, v3 = (v2 + 1) / 2;
  float* flat = scratch[6].Get(size_t(T3) * 16 * C);
  constexpr int kChunk = 8;
  const int chunks = (T3 + kChunk - 1) / kChunk;
  pool.For(chunks, [&](int chunk) {
    static thread_local std::vector<float> ring, s2, tmp;
    const int a = chunk * kChunk, n3 = std::min(T3 - a, kChunk), n2 = 2 * n3 + 1;
    const int lo2 = 2 * a - 1, lo1 = 2 * lo2 - 1, W1 = 64 * C, W2 = 32 * C, W3 = 16 * C;
    ring.resize(3 * W1), tmp.resize(size_t(n2) * W2), s2.resize(size_t(n2) * W2);
    auto conv0_row = [&](int r) {
      float* out = &ring[r % 3 * W1];
      const int row = lo1 + r;
      if (row < 0 || row >= std::min(T1, v1)) return void(std::fill_n(out, W1, 0.f));
      for (int w = 0; w < 64; ++w) {
        float x[9];
        const float* t[9];
        int taps = 0;
        for (int ky = 0; ky < 3; ++ky) {
          const int y = 2 * row - 1 + ky;
          if (y < 0 || y >= T0) continue;
          for (int kx = w ? 0 : 1; kx < 3; ++kx) {
            x[taps] = mel[y * kMels + 2 * w - 1 + kx];
            t[taps++] = &conv0[(ky * 3 + kx) * C];
          }
        }
        Taps(conv0_bias.data(), t, taps, [&](int j, int) { return Set8(x[j]); }, out + w * C, C, true);
      }
    };
    for (int i = 0; i < n2; ++i) {
      for (int r = i ? 2 * i + 1 : 0; r <= 2 * i + 2; ++r) conv0_row(r);
      const float* rows[3] = {&ring[2 * i % 3 * W1], &ring[(2 * i + 1) % 3 * W1], &ring[(2 * i + 2) % 3 * W1]};
      DepthwiseRow(rows, 32, lo2 + i >= 0 && lo2 + i < v2, sub_dw[0], sub_dw_bias[0], &tmp[size_t(i) * W2]);
    }
    Pointwise(sub_pw[0], tmp.data(), 32, lo2, n2, T2, s2.data());
    for (int i = 0; i < n3; ++i) {
      const float* rows[3] = {&s2[size_t(2 * i) * W2], &s2[size_t(2 * i + 1) * W2], &s2[size_t(2 * i + 2) * W2]};
      DepthwiseRow(rows, 16, a + i < v3, sub_dw[1], sub_dw_bias[1], &tmp[size_t(i) * W3]);
    }
    Pointwise(sub_pw[1], tmp.data(), 16, a, n3, T3, &flat[size_t(a) * W3]);
  }, 1);
  std::vector<float> out(size_t(T3) * kHidden);
  Linear(pool, sub_linear, flat, T3, out.data());
  return out;
}

std::vector<float> Model::Vad(const float* hidden, int T) {
  Clock clock(kVad);
  std::vector<float> a(size_t(T) * 128), window(size_t(T) * 640, 0.f), b(size_t(T) * 128), probabilities(T);
  // This head's activations reach thousands, where the sigmoid lands in denormals and every vector takes a
  // microcode assist. Flushed to zero they give the exact sigmoid's result, as the reference does.
  auto silu = [](std::vector<float>& v) {
    FlushDenormals flush;
    Silu(v.data(), v.size());
  };
  Linear(pool, vad_proj, hidden, T, a.data());
  silu(a);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < 5; ++j)
      if (t + j - 2 >= 0 && t + j - 2 < T) std::memcpy(&window[(t * 5 + j) * 128], &a[(t + j - 2) * 128], 128 * 4);
  Linear(pool, vad_ctx, window.data(), T, b.data());
  silu(b);
  for (int t = 0; t < T; ++t) {
    float s = vad_out_bias;
    for (int c = 0; c < 128; ++c) s += b[t * 128 + c] * vad_out[c];
    probabilities[t] = 1.f / (1.f + std::exp(-s));
  }
  return probabilities;
}

// Sized to the first segment, then once more to the 30 s cap, so a process projects the table at most twice.
void Model::Positions(int T) {
  if (T <= rel_T && rel_float == float_path) return;
  rel_T = rel_T ? std::max(T, kSegmentFrames) : T, rel_float = float_path;
  const int R = 2 * rel_T - 1, H = kHidden;
  std::vector<float> positions(size_t(R) * H), r(size_t(R) * H);
  float inverse[kHidden / 2];
  for (int i = 0; i < H / 2; ++i) inverse[i] = 1.f / std::pow(10000.f, float(2 * i) / H);
  pool.For(R, [&](int j) {
    for (int i = 0; i < H / 2; ++i) {
      const float phase = float(rel_T - 1 - j) * inverse[i];
      positions[size_t(j) * H + 2 * i] = std::sin(phase), positions[size_t(j) * H + 2 * i + 1] = std::cos(phase);
    }
  });
  Act a;
  Prepare(a, positions.data(), R, H);
  for (int layer = 0; layer < kLayers; ++layer) {
    Apply(layers[layer].t[kRel], a, r.data());
    Clock clock(kAttention);
    pool.For(kHeads, [&](int h) { Pack(rel[layer][h], kHeadDim, R, r.data() + h * kHeadDim, 1, H); });
  }
}

// Query i reads its T position scores at columns rel_T - 1 - i onward, so a block of query rows needs only the
// union window of T + rows - 1 columns. One work item is (head, row block): its keys, positions, values and scores
// stay in L2 and each 16-column panel is reused from L1 across the block's rows.
void Model::AttentionMix(int layer, int T, const float* q, const float* k, const float* v, float* out) {
  Clock clock(kAttention);
  const Layer& L = layers[layer];
  const Packed* rp = rel[layer];
  const int D = kHeadDim, Tp = (T + 15) & ~15, Wp = Tp + kAttBlock + 32, blocks = (T + kAttBlock - 1) / kAttBlock;
  pool.For(2 * kHeads, [&](int i) {
    const int h = i / 2;
    if (i % 2 == 0) Pack(kp[h], D, T, k + h * D, 1, kHidden);
    if (i % 2 == 1) Pack(vp[h], T, D, v + h * D, kHidden, 1);
  }, 1);
  const float scale = 1.f / std::sqrt(float(D));
  pool.For(kHeads * blocks, [&](int item) {
    static thread_local std::vector<float> query, content, position;
    query.resize(kAttBlock * D), content.resize(size_t(kAttBlock) * Tp), position.resize(size_t(kAttBlock) * Wp);
    const int h = item / blocks, i0 = item % blocks * kAttBlock, m = std::min(kAttBlock, T - i0);
    auto biased = [&](const std::vector<float>& bias) {
      for (int i = 0; i < m; ++i)
        for (int d = 0; d < D; ++d) query[i * D + d] = q[size_t(i0 + i) * kHidden + h * D + d] + bias[h * D + d];
    };
    // Padding and scores far below the row peak become denormal probabilities, each a microcode assist. Flushed
    // to zero they change nothing above 1e-38.
    FlushDenormals flush;
    biased(L.bias_u);
    Gemm(kp[h], query.data(), D, m, content.data(), Tp, 0, kp[h].panels());
    biased(L.bias_v);
    const int first = rel_T - 1 - i0, p0 = (first - m + 1) / 16, p1 = (first + T - 1) / 16 + 1;
    for (int p = p0; p < p1; ++p) {
      // Rows whose window [first - i, first - i + T) meets this panel, widened to whole 6 row kernel groups.
      const int r0 = std::max(0, first - 16 * p - 15) / 6 * 6, r1 = std::min(m, (first + T - 16 * p + 5) / 6 * 6);
      Gemm(rp[h], &query[r0 * D], D, r1 - r0, position.data() + size_t(r0) * Wp - p0 * 16, Wp, p, p + 1);
    }
    for (int i = 0; i < m; ++i) {
      float* s = &content[size_t(i) * Tp];
      const float* p = &position[size_t(i) * Wp + rel_T - 1 - (i0 + i) - p0 * 16];
      for (int j = 0; j < T; ++j) s[j] = (s[j] + p[j]) * scale;
      for (int j = T; j < Tp; ++j) s[j] = -INFINITY;
      Softmax(s, Tp);
    }
    Gemm(vp[h], content.data(), Tp, m, out + size_t(i0) * kHidden + h * D, kHidden, 0, D / 16);
  }, 1);
}

std::vector<float> Model::Encode(const float* pcm, size_t samples, const std::string& tag, int& T) {
  int frames, valid, rows;
  const std::vector<float> mel = LogMel(pool, pcm, samples, frames, valid);
  Dump(tag + ".mel", mel.data(), 'f', {frames, kMels});
  std::vector<float> x = Subsample(mel.data(), frames, valid, rows, T);
  x.resize(size_t(T) * kHidden);
  Dump(tag + ".subsample", x.data(), 'f', {T, kHidden});

  const int H = kHidden;
  auto buffer = [&](int i, int width) { return scratch[i].Get(size_t(T) * width); };
  float* const wide = buffer(0, 4 * H);
  float* const n = buffer(1, H);
  float* const y = buffer(2, H);
  float* const q = buffer(3, H);
  float* const k = buffer(4, H);
  float* const v = buffer(5, H);
  Positions(T);

  // Ops whose output feeds a ternary linear write its int8 codes while the row is still in L1, so only the
  // attention output needs a separate quantize pass. The residual add rides in the LayerNorm that reads it.
  auto codes = [&](int t) {
    if (!float_path) QuantizeRows(act.x, act.q, t, t + 1);
  };
  auto norm = [&](float scale, const Norm& ln, float* out, bool quantize) {
    Clock clock(kNorms);
    if (quantize) Reserve(act, out, T, H);
    pool.For(T, [&](int t) {
      float* row = &x[size_t(t) * H];
      const float* delta = &y[size_t(t) * H];
      if (scale != 0.f)
        for (int i = 0; i < H; ++i) row[i] += scale * delta[i];
      LayerNorm(row, ln.w.data(), ln.b.data(), out + size_t(t) * H, H);
      if (quantize) codes(t);
    });
  };
  auto feed_forward = [&](float scale, const Norm& ln, const TernW& in, const TernW& out) {
    norm(scale, ln, n, true);
    Apply(in, act, wide);
    {
      Clock clock(kNorms);
      Reserve(act, wide, T, 4 * H);
      pool.For(T, [&](int t) { Silu(&wide[size_t(t) * 4 * H], 4 * H), codes(t); });
    }
    Apply(out, act, y);
  };

  for (int layer = 0; layer < kLayers; ++layer) {
    const Layer& L = layers[layer];
    feed_forward(0.f, L.norm_ff1, L.t[kFf1In], L.t[kFf1Out]);

    norm(0.5f, L.norm_att, n, true);
    Apply(L.t[kQ], act, q), Apply(L.t[kK], act, k), Apply(L.t[kV], act, v);
    AttentionMix(layer, T, q, k, v, n);
    Prepare(act, n, T, H);
    Apply(L.t[kO], act, y);

    norm(1.f, L.norm_conv, n, true);
    Apply(L.t[kPw1], act, wide);
    {
      Clock clock(kNorms);
      pool.For(T, [&](int t) { Glu(&wide[size_t(t) * 2 * H], &wide[size_t(t) * 2 * H + H], &q[size_t(t) * H], H); });
    }
    {
      Clock clock(kDepthwise);
      Reserve(act, n, T, H);
      pool.For(T, [&](int t) {
        const int j0 = std::max(0, 4 - t), taps = std::min(kConvKernel, T + 4 - t) - j0;
        const float* w[kConvKernel];
        for (int j = 0; j < taps; ++j) w[j] = &L.dw[(j0 + j) * H];
        const float* in = &q[size_t(t + j0 - 4) * H];
        float* o = &n[size_t(t) * H];
        Taps(L.dw_bias.data(), w, taps, [&](int j, int c) { return Load8(in + size_t(j) * H + c); }, o, H,
             false);
        Silu(o, H), codes(t);
      });
    }
    Apply(L.t[kPw2], act, y);

    feed_forward(1.f, L.norm_ff2, L.t[kFf2In], L.t[kFf2Out]);
    norm(0.5f, L.norm_out, x.data(), false);
    char name[16];
    snprintf(name, sizeof name, ".layer%02d", layer);
    Dump(tag + name, x.data(), 'f', {T, H});
  }
  std::vector<float> encoded(size_t(T) * kEncoded);
  Linear(pool, enc_proj, x.data(), T, encoded.data());
  Dump(tag + ".encoded", encoded.data(), 'f', {T, kEncoded});
  return encoded;
}

}  // namespace pk
