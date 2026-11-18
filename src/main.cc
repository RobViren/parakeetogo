#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "parakeetogo.h"

static std::vector<uint8_t> ReadAll(FILE* f) {
  std::vector<uint8_t> data;
  uint8_t chunk[1 << 16];
  for (size_t n; (n = fread(chunk, 1, sizeof chunk, f)) > 0;) data.insert(data.end(), chunk, chunk + n);
  return data;
}

static void Fail(const std::string& message) {
  fprintf(stderr, "%s\n", message.c_str());
  exit(1);
}

static std::vector<float> FromS16(const uint8_t* s, size_t n) {
  std::vector<float> pcm(n);
  for (size_t i = 0; i < n; ++i) pcm[i] = int16_t(s[2 * i] | s[2 * i + 1] << 8) / 32768.f;
  return pcm;
}

static std::vector<float> ReadWav(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) Fail("cannot open " + path);
  const std::vector<uint8_t> d = ReadAll(f);
  fclose(f);
  if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) Fail(path + " is not a wav file");
  auto u16 = [&](size_t at) { return d[at] | d[at + 1] << 8; };
  auto u32 = [&](size_t at) { return uint32_t(u16(at)) | uint32_t(u16(at + 2)) << 16; };
  bool ok = false;
  for (size_t at = 12; at + 8 <= d.size();) {
    const size_t size = std::min<size_t>(u32(at + 4), d.size() - at - 8), body = at + 8;
    if (!std::memcmp(&d[at], "fmt ", 4)) ok = u16(body) == 1 && u16(body + 2) == 1 && u32(body + 4) == 16000 && u16(body + 14) == 16;
    else if (!std::memcmp(&d[at], "data", 4) && ok) return FromS16(&d[body], size / 2);
    at = body + size + (size & 1);
  }
  Fail(path + ": need 16 kHz mono PCM16 wav (ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav)");
  return {};
}

static void Json(const char* text, FILE* out) {
  fputs("{\"text\":\"", out);
  for (const char* c = text; *c; ++c) {
    if (*c == '"' || *c == '\\') fputc('\\', out), fputc(*c, out);
    else if (uint8_t(*c) < 0x20) fprintf(out, "\\u%04x", *c);
    else fputc(*c, out);
  }
  fputs("\"}\n", out);
  fflush(out);
}

// --serve keeps the model resident: after "ready", each request on stdin is <u32 sample count><float32 pcm>, and
// each reply on stdout is one {"text":"..."} line.
static void Serve(pk_model* model) {
  puts("ready");
  fflush(stdout);
  for (uint32_t n; fread(&n, 4, 1, stdin) == 1;) {
    std::vector<float> pcm(n);
    if (fread(pcm.data(), 4, n, stdin) != n) break;
    char* text = pk_transcribe(model, pcm.data(), pcm.size());
    Json(text, stdout);
    free(text);
  }
}

int main(int argc, char** argv) {
  std::string model_dir = "model";
  std::vector<std::string> inputs;
  const char* dump = nullptr;
  int threads = 0, float_path = 0, serve = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-m" && i + 1 < argc) model_dir = argv[++i];
    else if (a == "-t" && i + 1 < argc) threads = atoi(argv[++i]);
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--float") float_path = 1;
    else if (a == "--serve") serve = 1;
    else inputs.push_back(a);
  }
  if (inputs.empty() && !serve) Fail("usage: pk [-m model_dir] [-t threads] [--float] [--dump DIR] file.wav... | - | --serve\n16 kHz mono PCM16 wav, one transcript line per file; - reads raw s16le 16 kHz mono from stdin");
  pk_model* model = pk_load(model_dir.c_str(), threads);
  if (serve) Serve(model);
  if (float_path || dump) pk_debug(model, float_path, dump);
  for (const std::string& input : inputs) {
    std::vector<float> pcm;
    if (input == "-") {
      const std::vector<uint8_t> raw = ReadAll(stdin);
      pcm = FromS16(raw.data(), raw.size() / 2);
    } else {
      pcm = ReadWav(input);
    }
    const auto t0 = std::chrono::steady_clock::now();
    char* text = pk_transcribe(model, pcm.data(), pcm.size());
    if (getenv("PK_TIME"))
      fprintf(stderr, "file\t%s\t%.2f\t%.3f\n", input.c_str(), pcm.size() / 16000.0,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    puts(text);
    fflush(stdout);
    free(text);
  }
  pk_free(model);
}
