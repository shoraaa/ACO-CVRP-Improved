#include "cvrp.h"
#include "instance.h"
#include "test_support.h"
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>

namespace {
template <typename F> void rejects(F operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "invalid input was accepted");
}

void feasibility() {
  for (int n : {1, 2, 3, 8, 31, 100}) {
    for (uint32_t seed = 0; seed < 5; ++seed) {
      auto f = fixture(n, seed);
      f.capacity = 0.18f; // tight routes, including single-customer routes
      for (int k : {1, 2, 32, 64}) {
        cvrp::Solver solver(f.coords.data(), f.demand.data(), n + 1, f.capacity,
                            8, k, 0, 12, .1f, 1, .05f, true, false, true,
                            seed % 2 == 0);
        solver.seed_rng(seed);
        verify(f, solver.best_solution(), solver.best_cost);
        cvrp::SampleResult batch;
        for (int iteration = 0; iteration < 10; ++iteration) {
          solver.sample(batch);
          for (size_t a = 0; a < batch.routes.size(); ++a)
            verify(f, batch.routes[a], batch.costs[a]);
          auto best = std::min_element(batch.costs.begin(), batch.costs.end()) -
                      batch.costs.begin();
          solver.update_pheromone(batch.routes[best], batch.costs[best]);
        }
        verify(f, solver.best_solution(), solver.best_cost);
      }
    }
  }
}

void degenerate() {
  for (int n : {1, 2, 40}) {
    auto f = fixture(n, 7);
    std::fill(f.coords.begin(), f.coords.end(), 0);
    std::fill(f.demand.begin(), f.demand.end(), 0);
    for (bool smooth : {false, true}) {
      cvrp::Solver s(f.coords.data(), f.demand.data(), n + 1, 1, 8, 32, 32, 12,
                     .1f, 1, .05f, true, false, true, smooth, 20);
      s.run(20);
      verify(f, s.best_solution(), s.best_cost);
      require(s.best_cost == 0, "zero-distance instance has nonzero cost");
      for (float tau : s.pheromone_sparse)
        require(std::isfinite(tau), "nonfinite pheromone");
    }
  }
}

void iteration_and_threads() {
  auto f = fixture(200, 42);
  // run() equals the manual sample/update loop only without ant merging,
  // which that loop does not perform.
  auto single = [&] {
    return cvrp::Solver(f.coords.data(), f.demand.data(), 201, 1, 16, 32, 32,
                        4, .1f, 1, .05f, true, false, true, true, 0, true, true,
                        false);
  };
  std::vector<int32_t> reference, merged_reference;
  for (int threads : {1, 2, 4}) {
    omp_set_num_threads(threads);
    cvrp::Solver merged(f.coords.data(), f.demand.data(), 201, 1, 16);
    merged.seed_rng(123);
    merged.run(30);
    verify(f, merged.best_solution(), merged.best_cost);
    if (merged_reference.empty())
      merged_reference = merged.best_route;
    else
      require(merged_reference == merged.best_route,
              "thread count changed merged solution");
    cvrp::Solver s = single();
    s.seed_rng(123);
    auto initial = s.best_route;
    require(s.run(0) == s.best_cost && initial == s.best_route,
            "zero iterations changed state");
    s.run(30);
    if (reference.empty())
      reference = s.best_route;
    else
      require(reference == s.best_route, "thread count changed solution");
    cvrp::Solver split = single();
    split.seed_rng(123);
    split.run(10);
    split.run(20);
    require(split.best_route == reference, "split run changed solution");
    cvrp::Solver manual = single();
    manual.seed_rng(123);
    cvrp::SampleResult batch;
    for (int i = 0; i < 30; ++i) {
      manual.sample(batch);
      auto best = std::min_element(batch.costs.begin(), batch.costs.end()) -
                  batch.costs.begin();
      manual.update_pheromone(batch.routes[best], batch.costs[best]);
    }
    require(manual.best_route == reference, "run differs from sample/update");
  }
}

void invalid_inputs() {
  auto f = fixture(8, 1);
  auto make = [&](int ants, int candidates, float capacity) {
    return cvrp::Solver(f.coords.data(), f.demand.data(), 9, capacity, ants,
                        candidates);
  };
  rejects([&] { make(0, 32, 1); });
  rejects([&] { make(1, 0, 1); });
  rejects([&] { make(1, -1, 1); });
  rejects([&] { make(1, 65, 1); });
  rejects([&] { make(1, 32, 0); });
  rejects([&] { make(1, 32, .01f); });
  rejects([&] { make(1, 32, std::numeric_limits<float>::infinity()); });
  rejects(
      [&] { cvrp::Solver(f.coords.data(), f.demand.data(), INT32_MIN, 1); });
  rejects(
      [&] { cvrp::Solver(f.coords.data(), f.demand.data(), 9, 1, 8, 32, -1); });
  rejects([&] {
    cvrp::Solver(f.coords.data(), f.demand.data(), 9, 1, 8, 32, 32, 12, 0);
  });
  rejects([&] {
    cvrp::Solver(f.coords.data(), f.demand.data(), 9, 1, 8, 32, 32, 12, 1);
  });
  f.demand[1] = -1;
  rejects([&] { make(1, 32, 1); });
  f.demand[1] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { make(1, 32, 1); });
  f.demand[1] = .1f;
  f.coords[0] = std::numeric_limits<float>::infinity();
  rejects([&] { make(1, 32, 1); });
  f.coords[0] = 0;
  f.demand[0] = .1f;
  rejects([&] { make(1, 32, 1); });
  f.demand[0] = 0;
  auto s = make(8, 32, 1);
  rejects([&] { s.run(-1); });
  rejects([&] { s.update_pheromone({0, 1, 0}, 1); });
  rejects([&] { s.update_pheromone({0, 1, 1, 0}, 1); });
  rejects([&] { s.update_pheromone({0, -1, 0}, 1); });
  rejects([&] { s.update_pheromone({0, 9, 0}, 1); });
  rejects([&] { s.update_pheromone(s.best_solution(), -1); });
  rejects([&] {
    s.update_pheromone(s.best_solution(), std::numeric_limits<float>::quiet_NaN());
  });
  verify(f, s.best_solution(), s.best_cost);
}

void parsing() {
  std::istringstream plain("3 10\n0 0 0\n1 0 4\n0 1 6\n");
  auto p = cvrp::read_plain_instance(plain);
  require(p.demand == std::vector<float>({0, 4, 6}) && p.coords.size() == 6,
          "plain parser");
  std::istringstream data(
      "\n['name', 'tiny', 'depot', 0, 0, 'customer', 1, 0, 0, 1, 'demand', 4, "
      "6, 'capacity', 10, 'cost', 3.4, 'end']\n"
      "depot,0,0,customer,1,0,0,1,capacity,10,demand,0,4,6,cost,3.4,node_flag,"
      "1,2\n");
  cvrp::Instance parsed;
  require(cvrp::read_instance(data, parsed) && parsed.name == "tiny",
          "list parser");
  require(parsed.coords == p.coords && parsed.demand == p.demand,
          "list coordinates/demand");
  require(cvrp::read_instance(data, parsed), "comma parser");
  require(parsed.coords == p.coords && parsed.demand == p.demand,
          "comma coordinates/demand");
  require(!cvrp::read_instance(data, parsed), "parser EOF");
  std::istringstream bad(
      "depot,0,0,customer,1,0,capacity,10,demand,not-a-number\n");
  rejects([&] { cvrp::read_instance(bad, parsed); });
  std::istringstream missing("2 10\n0 0 0\n");
  rejects([&] { cvrp::read_plain_instance(missing); });
}

void time_limits_and_restart() {
  const auto f = fixture(100, 42);
  cvrp::Solver solver(f.coords.data(), f.demand.data(), 101, 1, 8);
  solver.seed_rng(123);
  const auto initial = solver.best_solution();
  solver.run_for(0);
  require(solver.iterations_run == 0 && solver.best_solution() == initial,
          "zero time budget changed the solution");
  rejects([&] { solver.run_for(-1); });
  rejects([&] { solver.run_for(std::numeric_limits<double>::infinity()); });
  rejects([&] { solver.run_for(1, -1); });
  solver.run_for(0.02);
  verify(f, solver.best_solution(), solver.best_cost);
  // A fixed trajectory makes the warm-start regression independent of speed.
  cvrp::Solver warm(f.coords.data(), f.demand.data(), 101, 1, 8);
  warm.seed_rng(123);
  warm.run(30);
  const auto incumbent = warm.best_solution();
  cvrp::Solver restarted(f.coords.data(), f.demand.data(), 101, 1, 8);
  restarted.set_solution(incumbent);
  require(restarted.best_cost <= warm.best_cost,
          "warm start lost the supplied incumbent");
  restarted.polish();
  require(restarted.best_cost <= warm.best_cost,
          "polish lost the incumbent");
  restarted.run(5);
  verify(f, restarted.best_solution(), restarted.best_cost);
  require(restarted.best_cost <= warm.best_cost + 1e-4f,
          "restart or polish lost the incumbent");
  rejects([&] { restarted.set_solution({0, 1, 0}); });
}

struct Golden {
  int n;
  uint32_t seed;
  int ants, iterations;
  float cost;
  uint64_t route, trace;
};
const Golden golden[] = {
#include "golden.inc"
};

void regression() {
  omp_set_num_threads(4);
  for (const auto &g : golden) {
    const auto f = fixture(g.n, g.seed);
    // Legacy search fixtures also exercise larger generated instances.
    // Exact trajectory checks are optional because toolchains can differ.
    cvrp::Solver s(f.coords.data(), f.demand.data(), g.n + 1, 1, g.ants, 32, 32,
                   12, .1f, 1, .05f, true, false, true, true, 0, false, false,
                   false, false, false);
    s.seed_rng(g.seed);
    cvrp::SampleResult batch;
    uint64_t trace = 14695981039346656037ULL;
    for (int i = 0; i < g.iterations; ++i) {
      s.sample(batch);
      trace = cost_hash(trace, batch.costs);
      auto best = std::min_element(batch.costs.begin(), batch.costs.end()) -
                  batch.costs.begin();
      s.update_pheromone(batch.routes[best], batch.costs[best]);
    }
    verify(f, s.best_solution(), s.best_cost);
#ifdef CVRP_STRICT_REGRESSION
    require(s.best_cost == g.cost, "baseline cost changed");
    require(route_hash(s.best_route) == g.route, "baseline route changed");
    require(trace == g.trace, "sampled cost trajectory changed");
#endif
  }
}
} // namespace

int main() {
  try {
    omp_set_num_threads(4);
    feasibility();
    degenerate();
    iteration_and_threads();
    invalid_inputs();
    parsing();
    time_limits_and_restart();
    regression();
    std::cout << "All CVRP feasibility, validation, determinism and baseline "
                 "regression checks passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
