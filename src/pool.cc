#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <string>

#include "src/pk.h"

namespace pk {

float F16(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m + 1024), e - 25);
  return h & 0x8000 ? -v : v;
}

struct Cpu {
  int id;
  bool slow, sibling;
};

// One entry per logical CPU in the affinity mask: physical cores fastest first, then their hyperthread siblings.
static std::vector<Cpu> FastCores() {
  cpu_set_t mask;
  sched_getaffinity(0, sizeof mask, &mask);
  struct Core {
    long freq;
    int cpu;
  };
  std::map<std::pair<int, int>, Core> cores;
  std::vector<Core> spare;
  auto read = [](int cpu, const char* f) {
    std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + f);
    long v = 0;
    in >> v;
    return v;
  };
  long top = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) continue;
    const auto key = std::pair(int(read(cpu, "topology/physical_package_id")), int(read(cpu, "topology/core_id")));
    const Core c{read(cpu, "cpufreq/cpuinfo_max_freq"), cpu};
    top = std::max(top, c.freq);
    if (!cores.count(key)) cores[key] = c;
    else spare.push_back(c);
  }
  std::vector<Core> order;
  for (auto& [k, c] : cores) order.push_back(c);
  std::stable_sort(order.begin(), order.end(), [](const Core& a, const Core& b) { return a.freq > b.freq; });
  std::vector<Cpu> cpus;
  for (auto& c : order) cpus.push_back({c.cpu, c.freq * 10 < top * 9, false});
  for (auto& c : spare) cpus.push_back({c.cpu, c.freq * 10 < top * 9, true});
  return cpus;
}

static void PinTo(int cpu) {
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(cpu, &one);
  pthread_setaffinity_np(pthread_self(), sizeof one, &one);
}

const int Pool::kSpin = std::getenv("PK_SPIN") ? atoi(std::getenv("PK_SPIN")) : 3000;

Pool::Pool(int threads) : n(threads) {
  const std::vector<Cpu> cpus = FastCores();
  if (n <= 0) {
    n = 0;
    for (auto& c : cpus) n += !c.slow && !c.sibling;
    n = std::max(1, n);
  }
  slow_.assign(n, 0);
  auto cpu = [&](int i) { return cpus.empty() ? -1 : cpus[i % cpus.size()].id; };
  for (int i = 0; i < n; ++i) fast_ += !(slow_[i] = !cpus.empty() && cpus[i % cpus.size()].slow);
  if (cpu(0) >= 0) PinTo(cpu(0));
  for (int i = 1; i < n; ++i) workers_.emplace_back(&Pool::Work, this, i, cpu(i));
}

Pool::~Pool() {
  stop_ = true;
  gen_.fetch_add(1);
  gen_.notify_all();
  for (auto& t : workers_) t.join();
}

void Pool::Work(int i, int cpu) {
  if (cpu >= 0) PinTo(cpu);
  for (uint64_t seen = 0;;) {
    for (int spin = 0; gen_.load() == seen && spin < kSpin; ++spin) Pause();
    gen_.wait(seen);
    seen = gen_.load();
    if (stop_) return;
    call_(ctx_, i);
    if (pending_.fetch_sub(1) == 1) pending_.notify_one();
  }
}

}  // namespace pk
