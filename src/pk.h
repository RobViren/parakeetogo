#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace pk {

inline void Pause() {
#if defined(__x86_64__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ volatile("isb");
#endif
}

struct AlignedFree {
  void operator()(void* p) const { std::free(p); }
};
template <class T> using Buf = std::unique_ptr<T[], AlignedFree>;
template <class T> Buf<T> Alloc(size_t n) {
  void* p = nullptr;
  posix_memalign(&p, 64, (n * sizeof(T) + 63) / 64 * 64);
  return Buf<T>(static_cast<T*>(p));
}

// Grow-only uninitialized storage reused across segments.
struct Scratch {
  Buf<float> p;
  size_t cap = 0;
  float* Get(size_t n) {
    if (n > cap) p = Alloc<float>(cap = n);
    return p.get();
  }
};

float F16(uint16_t h);

// Persistent workers pinned one per logical CPU within the process affinity mask: physical cores fastest first,
// then hyperthread siblings. n <= 0 uses the fast physical cores, which is where a power-capped desktop peaks.
struct Pool {
  int n;
  explicit Pool(int n);
  ~Pool();
  template <class F> void Run(F&& f) {
    call_ = [](void* c, int i) { (*static_cast<F*>(c))(i); };
    ctx_ = &f;
    pending_.store(n - 1);
    gen_.fetch_add(1);
    gen_.notify_all();
    f(0);
    for (int spin = 0; pending_.load() != 0 && spin < kSpin; ++spin) Pause();
    for (int p; (p = pending_.load()) != 0;) pending_.wait(p);
  }
  // Work is claimed in grains from a shared counter, so fast cores take more of it. Slow cores stop claiming once
  // the fast ones can finish the rest in one grain each; otherwise every call ends waiting on an E-core.
  template <class F> void For(int count, F&& f, int grain = 0) {
    if (grain <= 0) grain = std::max(1, count / (8 * n));
    std::atomic<int> next{0};
    Run([&](int i) {
      for (;;) {
        if (slow_[i] && count - next.load(std::memory_order_relaxed) <= grain * fast_) return;
        const int j = next.fetch_add(grain, std::memory_order_relaxed);
        if (j >= count) return;
        for (int k = j, e = std::min(count, j + grain); k < e; ++k) f(k);
      }
    });
  }

 private:
  // Calls arrive back to back (hundreds per segment), so a futex sleep between them costs more than the work it
  // saves. Workers and the caller spin this many pauses before sleeping (PK_SPIN overrides).
  static const int kSpin;
  void Work(int i, int cpu);
  void (*call_)(void*, int) = nullptr;
  void* ctx_ = nullptr;
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> pending_{0};
  std::atomic<bool> stop_{false};
  std::vector<char> slow_;
  int fast_ = 0;
  std::vector<std::thread> workers_;
};

}  // namespace pk
