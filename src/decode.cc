#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "parakeetogo.h"
#include "src/model.h"

namespace pk {

Times g_times;

Times::~Times() {
  if (!std::getenv("PK_TIME")) return;
  static const char* const names[kStages] = {"frontend", "subsampler", "quantize", "tern_gemm", "attention",
                                             "depthwise", "norms_act", "decoder_joint", "vad", "total"};
  double other = 2 * ms[kTotal];
  for (int s = 0; s < kStages; ++s) {
    fprintf(stderr, "%s\t%.1f\t%ld\n", names[s], ms[s], calls[s]);
    other -= ms[s];
  }
  fprintf(stderr, "other\t%.1f\t0\naudio_s\t%.2f\t0\n", other, audio_seconds);
}

// Just enough JSON to walk tokenizer.json: commas and colons count as whitespace.
struct Json {
  const char* p;
  void Space() {
    while (*p && std::strchr(" \n\r\t,:", *p)) ++p;
  }
  std::string String() {
    Space();
    std::string s;
    auto hex = [&] {
      unsigned v = 0;
      for (int i = 0; i < 4; ++i) v = v * 16 + (std::isdigit(*p) ? *p - '0' : (*p | 32) - 'a' + 10), ++p;
      return v;
    };
    for (++p; *p != '"';) {
      if (*p != '\\') {
        s += *p++;
        continue;
      }
      const char e = p[1];
      p += 2;
      if (e != 'u') {
        s += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
        continue;
      }
      unsigned c = hex();
      if (c >= 0xd800 && c < 0xdc00) p += 2, c = 0x10000 + ((c - 0xd800) << 10) + (hex() - 0xdc00);
      if (c < 0x80) s += char(c);
      else if (c < 0x800) s += char(0xc0 | c >> 6);
      else if (c < 0x10000) s += char(0xe0 | c >> 12);
      else s += char(0xf0 | c >> 18), s += char(0x80 | (c >> 12 & 63));
      if (c >= 0x800) s += char(0x80 | (c >> 6 & 63));
      if (c >= 0x80) s += char(0x80 | (c & 63));
    }
    ++p;
    return s;
  }
  double Number() {
    Space();
    char* end;
    const double v = std::strtod(p, &end);
    p = end;
    return v;
  }
  void Skip() {
    Space();
    if (*p == '"') {
      String();
    } else if (*p == '{' || *p == '[') {
      for (++p, Space(); *p != '}' && *p != ']'; Space()) Skip();
      ++p;
    } else {
      while (!std::strchr(",}] \n\r\t", *p)) ++p;
    }
  }
  // f(key) must consume the value.
  template <class F> void Object(F&& f) {
    Space();
    for (++p, Space(); *p != '}'; Space()) f(String());
    ++p;
  }
  template <class F> void Array(F&& f) {
    Space();
    for (++p, Space(); *p != ']'; Space()) f();
    ++p;
  }
};

void Model::LoadTokenizer(const std::string& path) {
  size_t size;
  const uint8_t* file = MapFile(path.c_str(), &size);
  const std::string text(reinterpret_cast<const char*>(file), size);
  vocab.resize(kVocab), special.assign(kVocab, 0);
  Json j{text.c_str()};
  j.Object([&](const std::string& key) {
    if (key == "added_tokens") {
      j.Array([&] {
        int id = 0;
        bool is_special = false;
        j.Object([&](const std::string& field) {
          if (field == "id") id = int(j.Number());
          else if (field == "special") j.Space(), is_special = *j.p == 't', j.Skip();
          else j.Skip();
        });
        special[id] = is_special;
      });
    } else if (key == "model") {
      j.Object([&](const std::string& field) {
        if (field == "vocab") j.Object([&](const std::string& token) { vocab[int(j.Number())] = token; });
        else j.Skip();
      });
    } else {
      j.Skip();
    }
  });
}

std::string Model::Detokenize(const std::vector<int>& steps) {
  std::string text;
  for (size_t i = 1; i < steps.size(); i += 3)
    if (steps[i] != kBlank && steps[i] != kPad && !special[steps[i]]) text += vocab[steps[i]];
  for (size_t at; (at = text.find("▁")) != std::string::npos;) text.replace(at, 3, " ");
  const size_t a = text.find_first_not_of(" \n\t\r"), b = text.find_last_not_of(" \n\t\r");
  return a == std::string::npos ? "" : text.substr(a, b - a + 1);
}

void Model::Dump(const std::string& name, const void* data, char type, std::vector<int> shape) {
  if (dump_dir.empty()) return;
  size_t n = 4;
  std::string dims;
  for (int d : shape) n *= d, dims += std::to_string(d) + ",";
  std::string header = std::string("{'descr': '<") + type + "4', 'fortran_order': False, 'shape': (" + dims + "), }";
  header.resize((10 + header.size() + 64) / 64 * 64 - 10, ' ');
  header.back() = '\n';
  const uint16_t length = uint16_t(header.size());
  FILE* f = fopen((dump_dir + "/" + name + ".npy").c_str(), "wb");
  fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  fwrite(&length, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data, 1, n, f);
  fclose(f);
}

void Model::Predict(int token, float* h, float* c, float* predicted) {
  const int D = kEncoded;
  alignas(32) float in[2 * D], gates[4 * D];
  static const Picks all2(in, 2 * D, false), all(in, D, false);
  for (int i = 0; i < 2; ++i) {
    std::memcpy(in, i ? h : &embedding[size_t(token) * D], D * 4);
    std::memcpy(in + D, h + i * D, D * 4);
    Linear(pool, cell[i], in, all2, gates);
    Sigmoid(gates, 2 * D), Sigmoid(gates + 3 * D, D);
    for (int j = 0; j < D; ++j) {
      c[i * D + j] = gates[D + j] * c[i * D + j] + gates[j] * std::tanh(gates[2 * D + j]);
      h[i * D + j] = gates[3 * D + j] * std::tanh(c[i * D + j]);
    }
  }
  Linear(pool, dec_proj, h + D, all, predicted);
}

// Rows of (frame, token, duration), one per joint evaluation, blanks included. The step budget is 10 per frame
// for the whole segment with no per frame symbol cap: a non blank token with duration 0 stays on its frame.
// About three steps in four emit a token and change the predictor state, so there is no run of frames to batch
// under one state; each step is one sparse row product over the joint head (the ReLU zeroes over a third of it).
std::vector<int> Model::Greedy(const float* encoded, int T) {
  Clock clock(kDecode);
  const int D = kEncoded;
  std::vector<float> h(2 * D, 0.f), c(2 * D, 0.f);
  alignas(32) float predicted[D], in[D], logits[(kVocab + kDurations + 15) / 16 * 16];
  std::vector<int> steps;
  Predict(kBlank, h.data(), c.data(), predicted);
  for (int frame = 0, tokens = 0; frame < T && int(steps.size()) < 3 * 10 * T && tokens < 4096;) {
    for (int i = 0; i < D; ++i) in[i] = std::max(encoded[size_t(frame) * D + i] + predicted[i], 0.f);
    Linear(pool, joint, in, Picks(in, D, true), logits);
    const int token = Argmax(logits, kVocab);
    int duration = int(std::max_element(logits + kVocab, logits + kVocab + kDurations) - logits) - kVocab;
    if (token == kBlank && duration == 0) duration = 1;
    steps.insert(steps.end(), {frame, token, duration});
    frame += duration;
    if (token != kBlank) Predict(token, h.data(), c.data(), predicted), ++tokens;
  }
  return steps;
}

using Spans = std::vector<std::pair<double, double>>;
constexpr double kFrameSeconds = 0.08, kVadThreshold = 0.5, kVadMinSpeech = 0.1, kVadMinGap = 0.1;
constexpr double kSegmentSeconds = 30.0, kMinSegmentSeconds = 1.0, kMinPause = 0.2, kPauseEps = 1e-6;
constexpr long kBlockSamples = 120 * kSampleRate, kMinTailSamples = kSampleRate / 2, kMinSamples = 320;

// The VAD head sees every subsampler row, including the ones past the valid length (reference quirk).
Spans Model::BlockSpeech(const float* pcm, size_t n, const std::string& tag) {
  int frames, mel_valid, rows, valid;
  const std::vector<float> mel = LogMel(pool, pcm, n, frames, mel_valid);
  const std::vector<float> hidden = Subsample(mel.data(), frames, mel_valid, rows, valid);
  const std::vector<float> probabilities = Vad(hidden.data(), rows);
  Dump(tag + ".probabilities", probabilities.data(), 'f', {valid});
  const double seconds = double(n) / kSampleRate;
  Spans regions, kept;
  for (int i = 0, start = -1; i <= valid; ++i) {
    const bool speaking = i < valid && probabilities[i] >= kVadThreshold;
    if (speaking && start < 0) start = i;
    if (speaking || start < 0) continue;
    const double a = start * kFrameSeconds, b = i * kFrameSeconds;
    if (!regions.empty() && a - regions.back().second < kVadMinGap) regions.back().second = b;
    else regions.push_back({a, b});
    start = -1;
  }
  for (auto [a, b] : regions)
    if (b - a >= kVadMinSpeech) kept.push_back({a, std::min(b, seconds)});
  return kept;
}

static Spans Pauses(Spans regions, double seconds) {
  std::sort(regions.begin(), regions.end());
  Spans pauses;
  double previous = 0;
  for (auto [a, b] : regions) {
    if (a - previous >= kMinPause - kPauseEps) pauses.push_back({previous, a});
    previous = std::max(previous, b);
  }
  if (seconds - previous >= kMinPause - kPauseEps) pauses.push_back({previous, seconds});
  return pauses;
}

static double NextCut(const Spans& pauses) {
  const std::pair<double, double>* pick = nullptr;
  for (auto& p : pauses)
    if (p.first >= kMinSegmentSeconds && p.second <= kSegmentSeconds) pick = &p;
  if (!pick)
    for (auto& p : pauses)
      if (const double mid = (p.first + p.second) / 2; mid >= kMinSegmentSeconds && mid <= kSegmentSeconds) pick = &p;
  return pick ? (pick->first + pick->second) / 2 : kSegmentSeconds;
}

static bool HasSpeech(const Spans& regions, double seconds) {
  return std::any_of(regions.begin(), regions.end(), [&](auto& r) { return r.second > 0 && r.first < seconds; });
}

// Sample ranges to transcribe. Audio up to 30 s is one segment and never reaches the VAD head.
std::vector<std::pair<long, long>> Model::Segments(const float* pcm, size_t n) {
  const long cap = long(kSegmentSeconds * kSampleRate), total = long(n);
  if (total <= cap) return {{0, total}};
  std::vector<std::pair<long, long>> out;
  Spans regions;
  long start = 0, read = 0;
  for (int block_index = 0; read < total; ++block_index) {
    const long remaining = total - read;
    const long block = remaining <= kBlockSamples ? remaining : std::min(kBlockSamples, remaining - kMinTailSamples);
    const double offset = double(read - start) / kSampleRate;
    char tag[16];
    snprintf(tag, sizeof tag, "vad%03d", block_index);
    for (auto [a, b] : BlockSpeech(pcm + read, block, tag)) regions.push_back({a + offset, b + offset});
    read += block;
    while (read - start > cap) {
      const long cut = std::lrint(NextCut(Pauses(regions, double(read - start) / kSampleRate)) * kSampleRate);
      const double seconds = double(cut) / kSampleRate;
      if (HasSpeech(regions, seconds)) out.push_back({start, start + cut});
      start += cut;
      Spans rest;
      for (auto [a, b] : regions)
        if (b > seconds) rest.push_back({std::max(a - seconds, 0.0), b - seconds});
      regions = rest;
    }
  }
  if (read - start >= kMinSamples && HasSpeech(regions, double(read - start) / kSampleRate)) out.push_back({start, read});
  return out;
}

std::string Model::Transcribe(const float* pcm, size_t n) {
  Clock clock(kTotal);
  g_times.audio_seconds += double(n) / kSampleRate;
  if (n < size_t(kMinSamples)) return "";
  const auto segments = Segments(pcm, n);
  std::vector<int> flat;
  for (auto [a, b] : segments) flat.insert(flat.end(), {int(a), int(b)});
  Dump("segments", flat.data(), 'i', {int(segments.size()), 2});
  std::string text;
  for (size_t i = 0; i < segments.size(); ++i) {
    char tag[16];
    snprintf(tag, sizeof tag, "seg%03zu", i);
    int T;
    const std::vector<float> encoded = Encode(pcm + segments[i].first, segments[i].second - segments[i].first, tag, T);
    const std::vector<int> steps = Greedy(encoded.data(), T);
    Dump(std::string(tag) + ".steps", steps.data(), 'i', {int(steps.size() / 3), 3});
    const std::string part = Detokenize(steps);
    if (!part.empty()) text += (text.empty() ? "" : " ") + part;
  }
  return text;
}

}  // namespace pk

extern "C" {

pk_model* pk_load(const char* model_dir, int threads) {
  return reinterpret_cast<pk_model*>(new pk::Model(model_dir, threads));
}

void pk_debug(pk_model* model, int float_path, const char* dump_dir) {
  auto* m = reinterpret_cast<pk::Model*>(model);
  m->float_path = float_path;
  m->dump_dir = dump_dir ? dump_dir : "";
  if (dump_dir) std::filesystem::create_directories(dump_dir);
}

char* pk_transcribe(pk_model* model, const float* pcm, size_t samples) {
  return strdup(reinterpret_cast<pk::Model*>(model)->Transcribe(pcm, samples).c_str());
}

void pk_free(pk_model* model) { delete reinterpret_cast<pk::Model*>(model); }
}
