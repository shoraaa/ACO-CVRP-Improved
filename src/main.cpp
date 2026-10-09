#include "cvrp.h"
#include "instance.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace {
struct Options {
  std::string input, dataset, demand_file, initial_file;
  int32_t nodes = 1000, iterations = 1000, ants = 100, threads = 4, limit = 1,
          skip = 0;
  int32_t candidates = 48, backup = 32, new_edges = 4, fixed_steps = 0;
  uint64_t seed = 1234;
  float evaporation = 0.1f, alpha = 1.0f;
  float capacity_scale = 1.0f;
  double time_limit = -1;
  int32_t max_non_impr = 0;
  bool iterations_given = false;
  bool route = false, local_search = true, smooth = true, spatial = true,
       sweep = true, depot_heuristic = false;
  float gbest = 0, tau_min = 0;
  int32_t force_change = 0, grow_after = 0, max_copy_run = -1;
  int32_t escape_budget = 0;
  bool initial_ls = true, extended = true, merge = true;
};

void usage() {
  std::cout
      << "CVRP ACO+ILS (Euclidean distances, depot at node 0)\n"
         "Usage: cvrp_solver [--input instance.txt | --dataset dataset.txt | "
         "--nodes N]\n"
         "  --input FILE        n capacity, followed by n rows of x y demand\n"
         "  --dataset FILE      comma-separated/list-style CVRP "
         "records\n"
         "  --nodes N           generated customers (default 1000)\n"
         "  --iterations N      ACO iterations (default 1000)\n"
         "  --time-limit S      search seconds instead of an iteration budget\n"
         "  --max-non-impr N    consecutive stale iterations (0 disables)\n"
         "  --capacity-scale X multiply planning capacity (default 1)\n"
         "  --demand FILE       whitespace-separated customer planning "
         "demands\n"
         "  --ants N            ants per iteration (default 100)\n"
         "  --threads N         OpenMP threads (default 4)\n"
         "  --seed N            search/generation seed (default 1234)\n"
         "  --limit N           dataset records to solve (default 1)\n"
         "  --skip N            dataset records to skip first (default 0)\n"
         "  --candidates N      candidate neighbors, 1..64 (default 48)\n"
         "  --backup N          backup neighbors (default 32)\n"
         "  --new-edges N       cross-route perturbation budget (default 4)\n"
         "  --fixed-steps N     override perturbation budget (default 0)\n"
         "  --evaporation X     pheromone evaporation in (0,1) (default 0.1)\n"
         "  --alpha X           pheromone exponent (default 1)\n"
         "  --classic-mmas      use classic pheromone bounds\n"
         "  --no-local-search   disable route improvement\n"
         "  --no-initial-ls     skip the global local search of the start\n"
         "  --basic-neighborhood  original operators only (no Or-opt, intra\n"
         "                      relocate, or edge-creating 2-opt* variants)\n"
         "  --no-merge          keep one ant per iteration (no merging of\n"
         "                      improving ants with disjoint routes)\n"
         "  --greedy-start      nearest-neighbour start instead of the sweep\n"
         "  --no-spatial-order  keep the input numbering (default: renumber\n"
         "                      customers along a Hilbert curve internally)\n"
         "  --route             include depot-separated route in JSON output\n";
}

uint64_t integer(const std::string &value, uint64_t maximum) {
  if (value.empty() ||
      value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected a nonnegative integer: " + value);
  const auto result = std::stoull(value);
  if (result > maximum)
    throw std::invalid_argument("integer out of range: " + value);
  return result;
}

float real(const std::string &value) {
  size_t used = 0;
  const auto result = std::stof(value, &used);
  if (used != value.size() || !std::isfinite(result))
    throw std::invalid_argument("expected a finite number: " + value);
  return result;
}

std::string json_string(const std::string &value) {
  std::string result = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += c;
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += c;
    }
  }
  return result + '"';
}

void solve(const cvrp::Instance &instance, const Options &options,
           uint64_t seed) {
  auto demand = instance.demand;
  if (!options.demand_file.empty()) {
    std::ifstream input(options.demand_file);
    if (!input)
      throw std::invalid_argument("cannot open demand file");
    for (size_t i = 1; i < demand.size(); ++i)
      if (!(input >> demand[i]))
        throw std::invalid_argument(
            "demand file must contain one number per customer");
    std::string extra;
    if (input >> extra)
      throw std::invalid_argument("extra values in demand file");
  }
  const auto start = std::chrono::steady_clock::now();
  cvrp::Solver solver(instance.coords.data(), demand.data(),
                      static_cast<int32_t>(instance.demand.size()),
                      instance.capacity * options.capacity_scale, options.ants,
                      options.candidates, options.backup, options.new_edges,
                      options.evaporation, options.alpha, 0.05f,
                      options.local_search, false, true, options.smooth,
                      options.fixed_steps, options.initial_ls,
                      options.extended, options.merge, options.spatial,
                      options.sweep);
  solver.seed_rng(seed);
  solver.global_best_source = options.gbest;
  solver.tau_min_override = options.tau_min;
  solver.force_change = options.force_change;
  solver.max_copy_run = options.max_copy_run;
  solver.escape_budget = options.escape_budget;
  solver.grow_after = options.grow_after;
  if (options.depot_heuristic)
    solver.use_neutral_depot_heuristic();
  if (!options.initial_file.empty()) {
    // Start the search from a given plan (whitespace-separated, depot 0).
    std::ifstream plan_input(options.initial_file);
    std::vector<int32_t> plan;
    for (int32_t node; plan_input >> node;)
      plan.push_back(node);
    solver.set_solution(plan);
  }
  const double initialization_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (options.time_limit >= 0)
    solver.run_for(options.time_limit, options.max_non_impr);
  else
    solver.run(options.iterations);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  const auto best = solver.best_solution();
  solver.validate_route(best);
  int routes_used = 0;
  for (size_t i = 1; i < best.size(); ++i)
    routes_used += best[i - 1] == 0 && best[i] != 0;
  std::cout << std::setprecision(10)
            << "{\"instance\":" << json_string(instance.name)
            << ",\"customers\":" << solver.m << ",\"cost\":" << solver.best_cost
            << ",\"seconds\":" << seconds
            << ",\"iterations\":" << solver.iterations_run
            << ",\"phases\":{\"probabilities\":" << solver.phase_probabilities
            << ",\"links\":" << solver.phase_links
            << ",\"ants\":" << solver.phase_ants
            << ",\"select\":" << solver.phase_select
            << ",\"pheromone\":" << solver.phase_pheromone << "}"
            << ",\"ant_thread_seconds\":{\"copy\":" << solver.ant_copy
            << ",\"construct\":" << solver.ant_construct
            << ",\"inter\":" << solver.ant_inter
            << ",\"intra\":" << solver.ant_intra
            << ",\"pack\":" << solver.ant_pack << "}"
            << ",\"seed\":" << seed << ",\"time_to_best\":"
            << initialization_seconds + solver.time_to_best
            << ",\"search_seconds\":" << solver.search_seconds
            << ",\"initialization_seconds\":" << initialization_seconds
            << ",\"routes_used\":" << routes_used
            << ",\"planning_capacity\":" << solver.capacity;
  if (instance.reference_cost > 0)
    std::cout << ",\"gap_percent\":"
              << 100 * (solver.best_cost / instance.reference_cost - 1);
  if (options.route) {
    std::cout << ",\"route\":[";
    for (size_t i = 0; i < best.size(); ++i) {
      if (i)
        std::cout << ',';
      std::cout << best[i];
    }
    std::cout << ']';
  }
  std::cout << "}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    Options options;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help" || key == "-h") {
        usage();
        return 0;
      }
      if (key == "--depot-heuristic") {
        options.depot_heuristic = true;
        continue;
      }
      if (key == "--greedy-start") {
        options.sweep = false;
        continue;
      }
      if (key == "--no-spatial-order") {
        options.spatial = false;
        continue;
      }
      if (key == "--route") {
        options.route = true;
        continue;
      }
      if (key == "--no-local-search") {
        options.local_search = false;
        continue;
      }
      if (key == "--no-initial-ls") {
        options.initial_ls = false;
        continue;
      }
      if (key == "--no-merge") {
        options.merge = false;
        continue;
      }
      if (key == "--basic-neighborhood") {
        options.extended = false;
        continue;
      }
      if (key == "--classic-mmas") {
        options.smooth = false;
        continue;
      }
      if (++i >= argc)
        throw std::invalid_argument("missing value for " + key);
      const std::string value = argv[i];
      if (key == "--input")
        options.input = value;
      else if (key == "--dataset")
        options.dataset = value;
      else if (key == "--skip") {
        options.skip = std::stoi(value);
        if (options.skip < 0)
          throw std::invalid_argument("skip must be nonnegative");
      }
      else if (key == "--demand")
        options.demand_file = value;
      else if (key == "--initial")
        options.initial_file = value;
      else if (key == "--gbest-prob")
        options.gbest = std::stof(value);
      else if (key == "--tau-min")
        options.tau_min = std::stof(value);
      else if (key == "--force-change")
        options.force_change = value == "all" ? 2 : value == "first" ? 1 : 0;
      else if (key == "--max-copy-run")
        options.max_copy_run = std::stoi(value);
      else if (key == "--escape")
        options.escape_budget = std::stoi(value);
      else if (key == "--grow-ants")
        options.grow_after = std::stoi(value);
      else if (key == "--time-limit") {
        options.time_limit = real(value);
        if (options.time_limit < 0)
          throw std::invalid_argument("negative time limit");
      } else if (key == "--capacity-scale")
        options.capacity_scale = real(value);
      else if (key == "--seed")
        options.seed = integer(value, UINT64_MAX);
      else if (key == "--evaporation")
        options.evaporation = real(value);
      else if (key == "--alpha")
        options.alpha = real(value);
      else {
        const auto number = static_cast<int32_t>(integer(value, INT32_MAX));
        if (key == "--nodes")
          options.nodes = number;
        else if (key == "--iterations") {
          options.iterations = number;
          options.iterations_given = true;
        } else if (key == "--max-non-impr")
          options.max_non_impr = number;
        else if (key == "--ants")
          options.ants = number;
        else if (key == "--threads")
          options.threads = number;
        else if (key == "--limit")
          options.limit = number;
        else if (key == "--candidates")
          options.candidates = number;
        else if (key == "--backup")
          options.backup = number;
        else if (key == "--new-edges")
          options.new_edges = number;
        else if (key == "--fixed-steps")
          options.fixed_steps = number;
        else
          throw std::invalid_argument("unknown option: " + key);
      }
    }
    if (options.threads < 1 || options.limit < 1 || options.ants < 1)
      throw std::invalid_argument("threads, limit and ants must be positive");
    if (options.capacity_scale <= 0)
      throw std::invalid_argument("capacity scale must be positive");
    if (options.time_limit >= 0 && options.iterations_given)
      throw std::invalid_argument("choose either --time-limit or --iterations");
    if (options.time_limit < 0 && options.max_non_impr)
      throw std::invalid_argument("--max-non-impr requires --time-limit");
    if (!options.demand_file.empty() && options.limit != 1)
      throw std::invalid_argument("a demand override requires --limit 1");
    if (!options.input.empty() && !options.dataset.empty())
      throw std::invalid_argument("choose either --input or --dataset");
    omp_set_num_threads(options.threads);
    if (!options.input.empty() || !options.dataset.empty()) {
      const std::string path =
          options.input.empty() ? options.dataset : options.input;
      std::ifstream input(path);
      if (!input)
        throw std::runtime_error("cannot open " + path);
      if (!options.input.empty()) {
        auto instance = cvrp::read_plain_instance(input);
        instance.name = path;
        solve(instance, options, options.seed);
      } else {
        int solved = 0;
        cvrp::Instance instance;
        for (int32_t skipped = 0; skipped < options.skip; ++skipped)
          if (!cvrp::read_instance(input, instance))
            throw std::invalid_argument("dataset has fewer records than --skip");
        instance = cvrp::Instance{};
        while (solved < options.limit && cvrp::read_instance(input, instance)) {
          if (instance.name.empty())
            instance.name = path + ':' + std::to_string(solved + 1);
          solve(instance, options, options.seed + solved);
          ++solved;
        }
        if (!solved)
          throw std::invalid_argument("dataset contains no instances");
      }
    } else {
      solve(cvrp::generate_instance(options.nodes, options.seed), options,
            options.seed);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cvrp_solver: " << error.what() << '\n';
    return 1;
  }
}
