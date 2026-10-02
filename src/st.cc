#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/pk.h"
#include "src/simd.h"
#include "src/st.h"

namespace pk {

const uint8_t* MapFile(const char* path, size_t* size) {
  const int fd = open(path, O_RDONLY);
  struct stat s;
  if (fd < 0 || fstat(fd, &s)) {
    const char* slash = std::strrchr(path, '/');
    if (const uint8_t* p = Embedded(slash ? slash + 1 : path, size)) return p;
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  void* p = mmap(nullptr, s.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (size) *size = s.st_size;
  return static_cast<const uint8_t*>(p);
}

// The header is flat JSON: {"name":{"dtype":"F16","shape":[..],"data_offsets":[a,b]},...}. No escapes in names.
SafeTensors::SafeTensors(const uint8_t* blob) {
  uint64_t hn;
  std::memcpy(&hn, blob, 8);
  const char* p = reinterpret_cast<const char*>(blob) + 8;
  const char* end = p + hn;
  const uint8_t* base = blob + 8 + hn;
  auto str = [&]() {
    p = static_cast<const char*>(std::memchr(p, '"', end - p)) + 1;
    const char* e = static_cast<const char*>(std::memchr(p, '"', end - p));
    std::string s(p, e);
    p = e + 1;
    return s;
  };
  auto ints = [&]() {
    std::vector<long> v;
    p = static_cast<const char*>(std::memchr(p, '[', end - p)) + 1;
    while (*p != ']') {
      char* e;
      v.push_back(std::strtol(p, &e, 10));
      p = e;
      while (*p == ',' || *p == ' ') ++p;
    }
    ++p;
    return v;
  };
  while (true) {
    const char* q = static_cast<const char*>(std::memchr(p, '"', end - p));
    if (!q) break;
    const std::string name = str();
    p = static_cast<const char*>(std::memchr(p, '{', end - p)) + 1;
    if (name == "__metadata__") {
      p = static_cast<const char*>(std::memchr(p, '}', end - p)) + 1;
      continue;
    }
    Tensor t;
    for (int k = 0; k < 3; ++k) {
      const std::string key = str();
      if (key == "dtype") {
        const std::string d = str();
        t.dtype = d == "F16" ? 'h' : d == "F32" ? 'f' : d == "U8" ? 'b' : 'l';
      } else if (key == "shape") {
        for (long v : ints()) t.shape.push_back(int(v)), t.n *= v;
      } else {
        t.data = base + ints()[0];
      }
    }
    this->t[name] = std::move(t);
    p = static_cast<const char*>(std::memchr(p, '}', end - p)) + 1;
  }
}

const Tensor& SafeTensors::operator[](const std::string& name) const {
  auto it = t.find(name);
  if (it == t.end()) {
    fprintf(stderr, "missing tensor %s\n", name.c_str());
    exit(1);
  }
  return it->second;
}

std::vector<float> SafeTensors::F32(const std::string& name) const {
  const Tensor& x = (*this)[name];
  std::vector<float> v(x.n);
  if (x.dtype == 'f') std::memcpy(v.data(), x.data, x.n * 4);
  size_t i = 0;
  if (x.dtype == 'h') {
    for (; i + 8 <= x.n; i += 8)
      Store8(LoadHalf8(x.f16() + i), &v[i]);
    for (; i < x.n; ++i) v[i] = F16(x.f16()[i]);
  }
  return v;
}

}  // namespace pk
