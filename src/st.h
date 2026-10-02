#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pk {

struct Tensor {
  char dtype = 0;  // 'h' F16, 'f' F32, 'b' U8, 'l' I64
  std::vector<int> shape;
  const uint8_t* data = nullptr;
  size_t n = 1;
  const uint16_t* f16() const { return reinterpret_cast<const uint16_t*>(data); }
  const float* f32() const { return reinterpret_cast<const float*>(data); }
};

// A safetensors blob (mmapped file or embedded bytes), read in place.
struct SafeTensors {
  std::unordered_map<std::string, Tensor> t;
  explicit SafeTensors(const uint8_t* blob);
  const Tensor& operator[](const std::string& name) const;
  std::vector<float> F32(const std::string& name) const;
};

// Falls back to the embedded asset of the same basename when the file does not exist.
const uint8_t* MapFile(const char* path, size_t* size = nullptr);
const uint8_t* Embedded(const char* name, size_t* size);

}  // namespace pk
