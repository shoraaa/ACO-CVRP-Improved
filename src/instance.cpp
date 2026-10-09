#include "instance.h"
#include "cvrp.h"
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cvrp {
namespace {
std::string trim(const std::string &s) {
  const auto first = s.find_first_not_of(" \t\r\n[]\"'");
  if (first == std::string::npos)
    return {};
  const auto last = s.find_last_not_of(" \t\r\n[]\"'");
  return s.substr(first, last - first + 1);
}

bool keyword(const std::string &s) {
  return s == "name" || s == "depot" || s == "customer" || s == "demand" ||
         s == "capacity" || s == "cost" || s == "node_flag" || s == "end";
}

float number(const std::string &s) {
  size_t consumed = 0;
  float value = std::stof(s, &consumed);
  if (consumed != s.size() || !std::isfinite(value))
    throw std::invalid_argument("invalid numeric field: " + s);
  return value;
}
} // namespace

Instance generate_instance(int32_t customers, uint64_t seed) {
  if (customers < 1 || customers > 1000000)
    throw std::invalid_argument(
        "generated customer count must be in [1, 1000000]");
  Instance instance;
  instance.name = "synthetic-" + std::to_string(customers);
  const float capacity = customers >= 50000   ? 2000
                         : customers >= 10000 ? 1000
                         : customers >= 5000  ? 500
                         : customers >= 1000  ? 250
                                              : 50;
  Random rng;
  rng.seed(seed);
  instance.coords.resize(2 * (customers + 1));
  for (auto &value : instance.coords)
    value = rng.next_float();
  instance.demand.resize(customers + 1);
  for (int32_t i = 1; i <= customers; ++i)
    instance.demand[i] = (1 + rng.next_uint(9)) / capacity;
  return instance;
}

bool read_instance(std::istream &input, Instance &instance) {
  std::string line;
  while (std::getline(input, line) && trim(line).empty()) {
  }
  if (!input && trim(line).empty())
    return false;
  instance = {};
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ','))
    fields.push_back(trim(field));
  std::vector<float> depot, customers;
  bool has_capacity = false;
  for (size_t i = 0; i < fields.size();) {
    const auto key = fields[i++];
    const size_t first = i;
    while (i < fields.size() && !keyword(fields[i]))
      ++i;
    if (key == "name") {
      if (i != first + 1)
        throw std::invalid_argument("name must contain one field");
      instance.name = fields[first];
    } else if (key == "depot" || key == "customer" || key == "demand") {
      auto &values = key == "depot"      ? depot
                     : key == "customer" ? customers
                                         : instance.demand;
      for (size_t j = first; j < i; ++j)
        values.push_back(number(fields[j]));
    } else if (key == "capacity" || key == "cost") {
      if (i != first + 1)
        throw std::invalid_argument(key + " must contain one number");
      if (key == "capacity") {
        instance.capacity = number(fields[first]);
        has_capacity = true;
      } else {
        instance.reference_cost = number(fields[first]);
      }
    } else if (first == 1 && !keyword(key)) {
      instance.name = key;
    }
  }
  if (depot.size() != 2 || customers.empty() || customers.size() % 2 ||
      !has_capacity)
    throw std::invalid_argument(
        "expected depot x,y, customer x,y,..., capacity C, demand d,...");
  instance.coords = std::move(depot);
  instance.coords.insert(instance.coords.end(), customers.begin(),
                         customers.end());
  const size_t n = instance.coords.size() / 2;
  if (instance.demand.size() == n - 1)
    instance.demand.insert(instance.demand.begin(), 0);
  if (instance.demand.size() != n)
    throw std::invalid_argument("demand count does not match coordinates");
  return true;
}

Instance read_plain_instance(std::istream &input) {
  Instance instance;
  int64_t nodes = 0;
  if (!(input >> nodes >> instance.capacity) || nodes < 2 || nodes > 1000001)
    throw std::invalid_argument(
        "expected node count (including depot) and capacity");
  instance.coords.resize(2 * nodes);
  instance.demand.resize(nodes);
  for (int64_t i = 0; i < nodes; ++i)
    if (!(input >> instance.coords[2 * i] >> instance.coords[2 * i + 1] >>
          instance.demand[i]))
      throw std::invalid_argument("expected one x y demand row per node");
  std::string extra;
  if (input >> extra)
    throw std::invalid_argument("unexpected data after last node");
  return instance;
}
} // namespace cvrp
