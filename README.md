# CVRP ACO Core

A standalone C++17 solver for the deterministic capacitated vehicle routing
problem, designed for fast improvement under short search budgets. It combines
sparse ant colony optimization with focused iterated local search and parallel
ants using OpenMP.

The solver uses a sweep initialization, candidate neighborhoods, cached edge
distances, and local search concentrated around each ant's changes. Improving
ants that modify disjoint routes can be merged into one solution. Operators
include relocate, swap, Or-opt, 2-opt*, and intra-route 2-opt.

## Quick start

Requires CMake 3.20+, a C++17 compiler, and OpenMP. Linux with GCC is the tested
platform. The solver and tests need no Python, GPU, Boost, or external datasets.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# A small included example.
./build/cvrp_solver --input examples/tiny.txt --iterations 100 --route

# A generated instance with 1,000 customers and a five-second search budget.
./build/cvrp_solver --nodes 1000 --time-limit 5 --threads 4 --seed 1234
```

For a binary tuned to your own CPU, configure with `-DCVRP_NATIVE=ON`. That binary
may require CPU instructions unavailable on another machine. The default is OFF.

## Inputs and outputs

For `--input FILE`, the first line is `node_count capacity`, followed by one
`x y demand` row per node. The node count **includes the depot**, which is row 0
and must have zero demand:

```text
4 10
0.0 0.0 0
1.0 0.0 4
0.0 1.0 6
1.0 1.0 5
```

Demands are nonnegative and each customer must fit in a vehicle. Routes begin and
end at the depot. The objective is total **unrounded Euclidean distance**, with
no fixed vehicle count or vehicle-opening cost. Capacity checks use integer
scaling, `round(value * 100000)`. Coordinates and search costs use single precision.

`--dataset FILE` reads one comma-separated or list-style record per line, as
shown in [examples/tiny-dataset.txt](examples/tiny-dataset.txt):

```sh
./build/cvrp_solver --dataset examples/tiny-dataset.txt \
  --limit 2 --iterations 100 --threads 4 --seed 1234 --route
```

This record format is different from raw CVRPLIB `.vrp` files; those files need
conversion before use. `--limit` selects how many records to solve and `--skip`
skips records at the start. Dataset record seeds increment from `--seed`.
Downloaded datasets can be kept in the ignored `data/` directory.

The CLI prints one JSON object per solved instance. Main fields are:

| Field | Meaning |
| --- | --- |
| `cost` | Best route length in input coordinate units |
| `seconds` | Initialization plus search wall time, excluding input parsing |
| `initialization_seconds` | Solver construction and initial improvement |
| `search_seconds` | Search wall time |
| `time_to_best` | Initialization plus search time until the incumbent was found |
| `iterations` | Completed search iterations |
| `routes_used` | Number of nonempty vehicle routes |
| `route` | Optional depot-separated route, enabled by `--route` |

Routes use the original input ids: `[0, customer, ..., 0, customer, ..., 0]`.
Every customer appears exactly once. A dataset reference cost enables
`gap_percent`; compare it only with a reference using the same distance convention
and constraints.

## Search budgets and reproducibility

`--time-limit S` is a **search-only** budget. Initialization runs first, and the
deadline is checked between complete iterations, so actual time may exceed the
requested budget. Use `seconds` when comparing total solve times.

For repeatable search trajectories, use `--iterations N` with a fixed instance,
seed, configuration, and compiler. Time-limited runs can complete different
numbers of iterations. Ant RNG streams are independent of thread scheduling;
tests check matching routes across one, two, and four threads. Floating-point
differences between toolchains or architectures can change trajectories.

The defaults are 100 ants, 48 candidate neighbors, 32 backup neighbors, four
cross-route perturbation edges, evaporation 0.1, and pheromone exponent 1.
`--help` lists the main CLI options. Examples of search variants:

```sh
./build/cvrp_solver --nodes 1000 --iterations 1000 --no-merge
./build/cvrp_solver --nodes 1000 --iterations 1000 --greedy-start
./build/cvrp_solver --nodes 1000 --time-limit 5 --max-non-impr 100
```

See [BENCHMARKS.md](BENCHMARKS.md) for a reproducible benchmarking protocol and
[docs/VALIDATION.md](docs/VALIDATION.md) for release validation. This release does
not attach historical research results to the extracted source without rerunning
the corresponding protocol.

## C++ library

Link the CMake target `cvrp` and include `cvrp.h`. Coordinates are interleaved
`float` values, depot first; the constructor copies the input arrays.

```cpp
#include "cvrp.h"
#include <omp.h>

int main() {
    const float coords[] = {0, 0, 1, 0, 0, 1, 1, 1};
    const float demand[] = {0, 4, 6, 5};
    omp_set_num_threads(4);
    cvrp::Solver solver(coords, demand, 4, 10);
    solver.seed_rng(1234);
    solver.run_for(1.0);
    auto route = solver.best_solution(); // Original input ids.
    solver.validate_route(route);
}
```

Use `best_solution()` to retrieve external ids; `best_route` is internal state
whose ids may be spatially reordered. `run(iterations)` and `run_for(seconds)`
continue the existing search. `set_solution(route)` supplies a feasible start.
`sample()` and `update_pheromone()` allow manual iteration control, but manual
sample/update does not perform the ant merging in `run()`. A solver object must
not be accessed concurrently.

A complete example is in [examples/library.cpp](examples/library.cpp). CMake
builds and tests it when `BUILD_TESTING=ON`, the default.

## Repository contents

- `src/`: deterministic solver, instance readers, and CLI.
- `examples/`: small input fixtures and a library example.
- `tests/`: feasibility, parsing, reproducibility, and regression checks.
- `.github/workflows/ci.yml`: Linux builds and sanitizer checks.
- `docs/PUBLISHING.md`: commands to create a new GitHub repository.

## License and acknowledgments

MIT; see [LICENSE](LICENSE). The k-d tree derives from Rafał Skinderowicz's
Focused ACO / ACO-TSP-Adaptive-Tuning implementation. Attribution and the upstream
MIT notice are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[licenses/](licenses/). The algorithm also builds on ideas from focused ACO,
iterated local search, candidate neighborhoods, and MAX-MIN ant systems.
