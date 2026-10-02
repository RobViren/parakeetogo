#include <cstring>

#include "src/st.h"

#define PK_INCBIN(sym, file)                                                              \
  __asm__(".section .rodata\n.balign 64\n" #sym ":\n.incbin \"" PK_ASSET_DIR "/" file "\"\n" #sym "_end:\n.previous\n"); \
  extern "C" const uint8_t sym[], sym##_end[];
PK_INCBIN(pk_asset_model, "model.safetensors")
PK_INCBIN(pk_asset_tokenizer, "tokenizer.json")

namespace pk {
const uint8_t* Embedded(const char* name, size_t* size) {
  const bool model = !std::strcmp(name, "model.safetensors");
  if (!model && std::strcmp(name, "tokenizer.json")) return nullptr;
  if (size) *size = model ? pk_asset_model_end - pk_asset_model : pk_asset_tokenizer_end - pk_asset_tokenizer;
  return model ? pk_asset_model : pk_asset_tokenizer;
}
}  // namespace pk
