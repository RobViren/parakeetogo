#pragma once
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "src/dense.h"
#include "src/st.h"
#include "src/tern.h"

namespace pk {

constexpr int kSampleRate = 16000, kHop = 160, kMels = 128;
constexpr int kLayers = 24, kHidden = 1024, kHeads = 8, kHeadDim = 128, kConvKernel = 9, kSubChannels = 256;
constexpr int kEncoded = 640, kVocab = 8193, kBlank = 8192, kPad = 2, kDurations = 5;
constexpr int kAttBlock = 48;  // query rows per attention work item, a multiple of the 6 row kernel
constexpr int kSegmentFrames = 376;  // encoder frames in a 30 s segment, the longest the segmenter emits

enum Stage { kFrontend, kSubsampler, kQuantize, kTernGemm, kAttention, kDepthwise, kNorms, kDecode, kVad, kTotal, kStages };

// PK_TIME=1 prints one TSV line per stage (stage, total ms, calls) to stderr at exit.
struct Times {
  double ms[kStages] = {}, audio_seconds = 0;
  long calls[kStages] = {};
  ~Times();
};
extern Times g_times;

struct Clock {
  Stage stage;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Clock(Stage s) : stage(s) {}
  ~Clock() {
    g_times.ms[stage] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++g_times.calls[stage];
  }
};


// [n / 160 + 1][128], normalized over the first valid = n / 160 frames. The trailing frame is zeroed, not dropped.
std::vector<float> LogMel(Pool& pool, const float* pcm, size_t n, int& frames, int& valid);

enum Tern { kFf1In, kFf1Out, kQ, kK, kV, kRel, kO, kPw1, kPw2, kFf2In, kFf2Out, kTerns };

struct Norm {
  std::vector<float> w, b;
};

struct Layer {
  TernW t[kTerns];
  Norm norm_ff1, norm_att, norm_conv, norm_ff2, norm_out;
  std::vector<float> bias_u, bias_v, dw, dw_bias;  // dw [9][1024] with BatchNorm folded in
};

// Input rows of a ternary linear: int8 codes, or the float rows themselves on the oracle path.
struct Act {
  const float* x = nullptr;
  size_t cap = 0;
  QAct q;
};

struct Model {
  Pool pool;
  bool float_path = false;
  std::string dump_dir;

  Layer layers[kLayers];
  std::vector<float> conv0, conv0_bias, sub_dw[2], sub_dw_bias[2];  // [3][3][256]
  Packed sub_pw[2], sub_linear, enc_proj, vad_proj, vad_ctx;
  Half cell[2], dec_proj, joint;
  std::vector<float> vad_out, embedding;
  float vad_out_bias = 0;
  std::vector<std::string> vocab;
  std::vector<char> special;

  Act act;
  Scratch scratch[7];
  Packed unpacked, kp[kHeads], vp[kHeads];
  // relative_k_proj of the position table depends only on the offset, never on the audio, and every row is
  // quantized and projected on its own. Cached per layer and head for rel_T frames; offset d is column rel_T - 1 - d.
  Packed rel[kLayers][kHeads];
  int rel_T = 0;
  bool rel_float = false;

  Model(const std::string& dir, int threads);
  void LoadTokenizer(const std::string& path);
  std::string Transcribe(const float* pcm, size_t n);

  // model.cc
  void Reserve(Act& a, const float* x, int T, int C);
  void Prepare(Act& a, const float* x, int T, int C);
  void Apply(const TernW& w, const Act& a, float* y);
  std::vector<float> Subsample(const float* mel, int frames, int valid, int& rows, int& valid_rows);
  std::vector<float> Vad(const float* hidden, int rows);
  void Positions(int T);
  void AttentionMix(int layer, int T, const float* q, const float* k, const float* v, float* out);
  std::vector<float> Encode(const float* pcm, size_t n, const std::string& tag, int& T);

  // decode.cc
  void Predict(int token, float* h, float* c, float* predicted);
  std::vector<int> Greedy(const float* encoded, int T);
  std::vector<std::pair<double, double>> BlockSpeech(const float* pcm, size_t n, const std::string& tag);
  std::vector<std::pair<long, long>> Segments(const float* pcm, size_t n);
  std::string Detokenize(const std::vector<int>& steps);
  void Dump(const std::string& name, const void* data, char type, std::vector<int> shape);
};

}  // namespace pk
