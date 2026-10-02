# parakeetogo

## Background
After using audio.cpp, CrispASR, and other minimal dependency targeting projects I was surprised to find that the ggml/gguf performance actually lagged behind the transformer model equivalent despite being natively compiled. The kernels were doing more work than needed and a targeted analysis with my good buddy Claude was able to roll in a lot of the amazing work moondream put into the ternary tune of parakeet. I'm working on a voice interface for Claude Code and wanted something that didn't soak up VRAM or could be deployed on the potato speed server that lives in my dresser drawer (back taken off, don't worry). Highway worked incredible as a portable solution for the SIMD acceleration. As fun as CUDA is I want to push the CPU angle because my server does not have a GPU of any kind, and I am curious to see if I can host this on my Pixel. Anyway, the model is fast, performant, and small. Sharing if anyone else needs this for their use cases.

The rest of this is largely LLM drafted for full disclosure. I obligate myself to be clear about what is and is not AI generated.

## What is it
A standalone C++ inference engine for [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), the ternary (1.58-bit) version of parakeet-tdt-0.6b-v3, on CPU (x86 and ARM). Audio in, text out, from one binary with no Python and no runtime downloads. Sister project to [kokoallovero](https://github.com/RobViren/kokoallovero), which does the same for Kokoro text to speech.

It runs about 2x faster than Photon, moondream's own runtime, on the same CPU. i7-14700F, 56 clips totalling 4.8 hours of speech:

| | pk | Photon | speedup |
| --- | --- | --- | --- |
| all 56 clips, sustained | 142 s (122x real time) | 300 s (58x) | 2.1x |
| one 109 s clip, from idle | 0.66 s (166x) | 1.54 s | 2.3x |

Transcripts differ from Photon's by 0.23% of words, which is the noise between two int8 implementations of the same model. `ref/photon_run.py` is the Photon side of that comparison (it needs the `moondream` package); `tools/wer.py` scores two transcript files against each other.

## Build

Needs CMake 3.24+ and Clang (GCC builds but spills the ternary kernel's accumulators). On x86 the binary needs AVX2, FMA, F16C and BMI2 (Haswell or later), and uses AVX-VNNI when the CPU has it. Highway is fetched at configure time.

```
cmake -B build -G Ninja
cmake --build build
```

Options:
- `-DPK_EMBED=ON` embeds the model and tokenizer from `PK_ASSET_DIR` (default `model/`), giving a single executable that needs no files.
- `-DPK_STATIC=ON` links fully static, for a binary that runs on older glibc.

## ARM

The same source builds for aarch64. SIMD outside the ternary kernel goes through [Google Highway](https://github.com/google/highway) as eight-float vectors: one 256-bit register on x86, two 128-bit registers on NEON, with the same layouts and summation order. The ternary kernel has a NEON variant using the dot product extension (ARMv8.2 `dotprod`; required). The result is bit-identical to x86: every `--dump` stage and the transcript match byte for byte, int8 and float paths, checked under qemu.

Cross-compiling from x86 with Clang and an aarch64 sysroot (Debian: `g++-aarch64-linux-gnu clang`):

```
cmake -B build-arm64 -G Ninja -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu -DPK_STATIC=ON -DPK_EMBED=ON
```

ARM speed is untuned: tile sizes and thread defaults were swept on Intel. The default thread count is the cores within 10% of the fastest one, which on a phone is often a single prime core, so pass `-t`.

## Weights

Configuring downloads `model.safetensors` and `tokenizer.json` into `model/` from [rodbiren/kokoro_parakeet_aio_models](https://huggingface.co/rodbiren/kokoro_parakeet_aio_models) if they are missing (`-DPK_FETCH_ASSETS=OFF` to skip, `-DPK_ASSET_URL=` to point elsewhere). Nothing is downloaded at runtime.

They are unmodified copies of moondream's files, which the engine reads in place with no conversion step. To take them from the source instead:

```
uvx --from huggingface_hub hf download moondream/parakeet-redux model.safetensors tokenizer.json --local-dir model
```

## Use

```
build/pk talk.wav
ffmpeg -i talk.mp3 -ar 16000 -ac 1 -f s16le - | build/pk -
```

- Input is 16 kHz mono PCM16 wav, one transcript line per file. `-` reads raw s16le 16 kHz mono from stdin.
- `-m dir` (model directory, default `model`), `-t threads` (default: the fast physical cores).
- `--float` runs the ternary layers on unquantized activations, the parity oracle. `--dump dir` writes one .npy per stage.
- `PK_TIME=1` prints per-stage timings to stderr. `PK_ISA=avx2` forces the non-VNNI kernel.

`parakeetogo.h` is the C API (`pk_load`, `pk_transcribe`, `pk_free`); the CLI is a thin wrapper over `libpk.a`.

## How it works

- `src/tern*`: the ternary GEMM, where two thirds of the time goes. The encoder's 604M ternary weights are unpacked at load to int8, and activations are quantized to int8 per frame and 128-column group, the same grouping the weight scales use. The inner step is one `vpdpbusd` per 8 output rows by 4 input columns, 12 accumulators deep, with tiles sized to keep weights in L1 across frames.
- `src/model.cc`, `src/dense.cc`: subsampler, encoder and VAD head. Relative position projections are cached per process, attention is blocked by head and query rows, and the ops that produce activations write their int8 codes in the same pass.
- `src/decode.cc`: tokenizer, TDT greedy decode, and VAD segmentation so long audio is handled inside the engine.
- `src/frontend.cc`: log-mel features.
- `ref/parakeet.py`: a plain PyTorch reimplementation of the model that matches Photon's float path token for token. It is the spec the C++ was ported from (`uv run ref/parakeet.py file.wav [--dump dir] [--int8]`).

Two correct int8 implementations decorrelate to the quantization noise level within a few layers, so parity is checked in three parts: the float path against the reference's dumps (`tools/parity.py`), the int8 kernel against the float kernel on the same codes, and int8 drift on transcripts of real audio.

## License

Apache 2.0 for the code. The model weights are moondream's, under CC-BY-4.0.
