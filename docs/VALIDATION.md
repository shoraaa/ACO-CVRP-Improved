# Release validation

The release was extracted from the working source on 2026-10-09. All preparation
was performed in this separate folder; original source, documentation, and Git
configuration were left in place.

## Extraction

The copied solver retains the deterministic construction, pheromone updates,
candidate lists, ant merging, local search, random streams, and default settings.
Custom stochastic route objectives, recourse dynamic programming, scenario
evaluation, research instrumentation, and experiment executables were removed.
The release library compiles only `src/cvrp.cpp` and `src/instance.cpp`.

During packaging, a warm-start test found that `set_solution()`/`polish()` could
replace a supplied incumbent with a worse solution after local search. The public
polish path now keeps the incumbent when its recomputed route cost is better.
This guard is limited to explicit warm starts and polishing; it does not change
the default constructor or search loop. A deterministic regression covers it.

## Comparing with the research solver

Both versions were compiled from source with GCC 16.2.1 on x86-64, using `-O3`,
`-DNDEBUG`, `-fopenmp`, `-fno-math-errno`, and `-ffp-contract=off`.

In 27 configurations (28 solved records), route arrays, costs, iteration counts,
route counts, and planning capacities matched exactly. The comparisons covered:

- Generated instances with 100, 1,000, and 10,000 customers and seeds 1, 42, 1234.
- One, two, and four OpenMP threads; 16 ants and a run with the default 100 ants.
- Ant merging disabled, greedy start, original numbering, classic MMAS,
  no local search, basic neighborhoods, a nondefault pheromone exponent, and
  fixed perturbation steps.
- Escape moves, forced changes, bounded copy runs, global-best reinforcement,
  and neutral depot heuristics.
- Two local CVRPLIB records from the research dataset, not redistributed here.

These are finite regression comparisons, not a proof of equivalence for every
possible input. The original working-source hashes and comparison configurations
are recorded in [extraction-checks.json](extraction-checks.json).

## Tests

The test suite checks independent route feasibility and cost, invalid parameters,
degenerate inputs, parsing, thread-count reproducibility, continued runs,
time-budget handling, warm starts, CLI inputs, and the C++ library example.
All fixtures needed to run the suite are included.

Exact historical trajectory hashes in `tests/golden.inc` are toolchain dependent.
Enable them with `-DCVRP_STRICT_REGRESSION=ON` on a matching toolchain. Portable
tests still run the same legacy fixtures and verify feasibility and recomputed
cost when strict checks are OFF.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-O1 -fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

## Verified results

Local validation on 2026-10-09, Intel Core i5-13400F, Linux x86-64, GCC 16.2.1,
CMake 4.4.2:

| Configuration | Result |
| --- | --- |
| Release, portable CPU flags, strict trajectory checks ON | 8/8 tests passed |
| Release, native CPU flags, strict trajectory checks ON | 8/8 tests passed |
| Fresh independent source copy, default settings | 8/8 tests passed |
| Debug with `-O1`, AddressSanitizer, UndefinedBehaviorSanitizer, and leak checking | 8/8 tests passed |
| Final source versus research solver | 27 configurations / 28 records matched |

LeakSanitizer initially failed under sandbox process tracing; the complete
sanitizer suite was then rerun successfully outside the sandbox with leak
checking enabled. No sanitizer suppression was added to the release.

An independent temporary Git repository staged all 25 intended files. Whitespace
checks, relative documentation links, and ignore checks for build output,
datasets, results, credentials, and source backups passed. Building that copy
did not add any generated files to its staged set. The original 41 inspected
files retained their hashes, and the parent repository's tracked status was
unchanged.

The GitHub workflow is included but must run on GitHub after publication. The
full throughput campaign and comparative quality benchmarks were not run as
part of packaging.
