# Benchmarking the deterministic core

Use a Release build, record the compiler version and CPU, and keep thread counts,
seeds, solver parameters, instance files, and distance conventions with the results.
For this solver, `--time-limit` measures search only; JSON `seconds` includes
initialization. Report both when claiming performance under a short time limit.

## Quality at a fixed budget

Use the same instance for every search seed. With `--nodes`, changing the seed also
changes the generated instance, so use `--input` or `--dataset` for repeated runs
on a fixed problem.

```sh
mkdir -p results
for seed in 1 2 3 4 5; do
  ./build/cvrp_solver --input examples/tiny.txt \
    --time-limit 1 --threads 4 --seed "$seed" --route
done > results/tiny-1s.jsonl
```

The included tiny instance checks the workflow; meaningful quality measurements
require larger instances. For a publication table, replace it with the intended
benchmark instances and include:

- Dataset source, instance names, file checksums, and any coordinate conversion.
- CPU, operating system, compiler, optimization options, and OpenMP thread count.
- Search budget, actual initialization and total time, seeds, and all nondefault options.
- Mean and best cost or gap, variability across seeds, and route feasibility checks.
- Reference costs with their provenance and identical distance and fleet conventions.

The solver minimizes unrounded Euclidean length. Standard rounded CVRPLIB
references may use a different objective. Reprice saved routes under the reference
convention before reporting those gaps, and label the convention explicitly.

## Synthetic throughput

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCVRP_BUILD_BENCHMARK=ON
cmake --build build --parallel
mkdir -p results
OMP_PROC_BIND=close OMP_PLACES=cores ./build/cvrp_benchmark > results/throughput.csv
```

The benchmark covers 100, 1,000, 10,000, and 100,000 customers; three instance/search
seeds; and three repetitions. It reports median search time, cost, and route hash
with four OpenMP threads. It uses the core defaults, except 32 ants for 100
customers. Iteration budgets are 200, 200, 100, and 20, respectively. Initialization
is excluded. The test independently checks feasibility and route cost.

These measurements describe throughput and reproducibility. They do not establish
competitive solution quality or optimality. Curated results can be committed under
`benchmarks/`; generated `results/` files are ignored.
