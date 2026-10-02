#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pk_model pk_model;

/* model_dir holds model.safetensors and tokenizer.json. A build with PK_EMBED falls back to its embedded copies
   when the files are absent. Exits with a message if neither exists. threads <= 0 picks the fast physical cores. */
pk_model* pk_load(const char* model_dir, int threads);

/* 16 kHz mono float PCM in [-1, 1] (int16 / 32768). Returns UTF-8 text the caller releases with free(). */
char* pk_transcribe(pk_model* model, const float* pcm, size_t samples);

/* float_path runs the ternary layers on unquantized activations (the parity oracle). dump_dir, if not NULL,
   receives one .npy per stage of every following pk_transcribe call. */
void pk_debug(pk_model* model, int float_path, const char* dump_dir);

void pk_free(pk_model* model);

#ifdef __cplusplus
}
#endif
