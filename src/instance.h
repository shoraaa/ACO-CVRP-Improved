#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace cvrp {
struct Instance {
  std::string name;
  std::vector<float> coords;
  std::vector<float> demand;
  float capacity = 1;
  double reference_cost = 0;
};

Instance generate_instance(int32_t customers, uint64_t seed);
// Read one comma-separated/list-style CVRP record per line.
bool read_instance(std::istream &input, Instance &instance);
// Compact standalone format: n capacity, then n lines of x y demand (depot
// first).
Instance read_plain_instance(std::istream &input);
} // namespace cvrp
