#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

struct Fixture {
  std::vector<float> coords, demand;
  float capacity = 1;
};

inline Fixture fixture(int customers, uint32_t seed) {
  std::mt19937 rng(seed);
  Fixture result;
  result.coords.resize(2 * (customers + 1));
  for (auto &v : result.coords)
    v = static_cast<float>(rng() >> 8) * (1.0f / (1u << 24));
  const float capacity = customers >= 50000   ? 2000
                         : customers >= 10000 ? 1000
                         : customers >= 5000  ? 500
                         : customers >= 1000  ? 250
                                              : 50;
  result.demand.resize(customers + 1);
  for (int i = 1; i <= customers; ++i)
    result.demand[i] = (1 + rng() % 9) / capacity;
  return result;
}

inline uint64_t hash_word(uint64_t hash, uint32_t word) {
  for (int shift = 0; shift < 32; shift += 8)
    hash = (hash ^ ((word >> shift) & 255)) * 1099511628211ULL;
  return hash;
}

inline uint64_t route_hash(const std::vector<int32_t> &route) {
  uint64_t hash = 14695981039346656037ULL;
  for (auto node : route)
    hash = hash_word(hash, static_cast<uint32_t>(node));
  return hash;
}

inline uint64_t cost_hash(uint64_t hash, const std::vector<float> &costs) {
  for (float cost : costs) {
    uint32_t bits;
    std::memcpy(&bits, &cost, sizeof(bits));
    hash = hash_word(hash, bits);
  }
  return hash;
}

inline void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Independent coverage, scaled capacity and double-precision distance check.
inline void verify(const Fixture &f, const std::vector<int32_t> &route,
                   float cost) {
  require(route.size() >= 3 && route.front() == 0 && route.back() == 0,
          "missing depot endpoints");
  std::vector<int> seen(f.demand.size());
  int64_t load = 0, capacity = std::llround(f.capacity * 100000);
  double measured = 0;
  int32_t prev = 0;
  for (int32_t u : route) {
    require(u >= 0 && static_cast<size_t>(u) < seen.size(), "invalid node");
    if (u) {
      require(++seen[u] == 1, "duplicate customer");
      load += std::llround(f.demand[u] * 100000);
      require(load <= capacity, "capacity violation");
    } else
      load = 0;
    const double dx = double(f.coords[2 * u]) - f.coords[2 * prev];
    const double dy = double(f.coords[2 * u + 1]) - f.coords[2 * prev + 1];
    measured += std::hypot(dx, dy);
    prev = u;
  }
  for (size_t i = 1; i < seen.size(); ++i)
    require(seen[i] == 1, "missing customer");
  require(std::isfinite(cost) &&
              std::abs(measured - cost) <= 2e-5 * std::max(1.0, measured),
          "incorrect solution cost");
}
