#include "cvrp.h"
#include "test_support.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>

int main() {
  omp_set_num_threads(4);
  std::cout
      << "customers,seed,ants,iterations,median_seconds,cost,route_hash\n";
  for (int n : {100, 1000, 10000, 100000}) {
    const int ants = n == 100 ? 32 : 100;
    const int iterations = n == 100000 ? 20 : n == 10000 ? 100 : 200;
    for (uint32_t seed : {1, 42, 1234}) {
      auto f = fixture(n, seed);
      std::vector<double> times;
      float cost = 0;
      uint64_t hash = 0;
      for (int repeat = 0; repeat < 3; ++repeat) {
        cvrp::Solver solver(f.coords.data(), f.demand.data(), n + 1, 1, ants);
        solver.seed_rng(seed);
        const auto start = std::chrono::steady_clock::now();
        solver.run(iterations);
        times.push_back(std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start)
                            .count());
        verify(f, solver.best_solution(), solver.best_cost);
        if (repeat)
          require(hash == route_hash(solver.best_route),
                  "nondeterministic search");
        cost = solver.best_cost;
        hash = route_hash(solver.best_route);
      }
      std::sort(times.begin(), times.end());
      std::cout << n << ',' << seed << ',' << ants << ',' << iterations << ','
                << std::setprecision(10) << times[1] << ',' << cost << ','
                << hash << std::endl;
    }
  }
}
