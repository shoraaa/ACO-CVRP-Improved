#include "cvrp.h"
#include <iostream>
#include <omp.h>

int main() {
  const float coords[] = {0, 0, 1, 0, 0, 1, 1, 1};
  const float demand[] = {0, 4, 6, 5};
  omp_set_num_threads(2);
  cvrp::Solver solver(coords, demand, 4, 10, /*ants=*/16);
  solver.seed_rng(1234);
  solver.run(100);
  const auto route = solver.best_solution(); // Original input node ids.
  solver.validate_route(route);
  std::cout << "cost: " << solver.best_cost << '\n';
  for (int node : route)
    std::cout << node << ' ';
  std::cout << '\n';
}
