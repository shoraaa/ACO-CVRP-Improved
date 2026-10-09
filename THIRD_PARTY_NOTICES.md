# Third-party notices

## K-d tree

`src/kd_tree.h` is adapted from code by **Rafał Skinderowicz**. The existing
FocusedACO copyright attribution is preserved in the file.

- Original attribution: <https://github.com/RSkinderowicz/FocusedACO>
- MIT-licensed distribution of the same implementation:
  <https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning>
- Inspected distribution revision: `a904e6a8786d48593ef1cac975edef5ed8920af3`.
- License text: [licenses/ACO-TSP-Adaptive-Tuning-MIT.txt](licenses/ACO-TSP-Adaptive-Tuning-MIT.txt).

The local adaptation uses unrounded Euclidean distances instead of rounded TSP
distances, and removes the corresponding `+1` margin in neighborhood bounds.
The upstream implementation and the local version were compared during packaging.
No other files from the upstream project are bundled.

## Algorithm background

Focused ACO is described in Rafał Skinderowicz, “Improving Ant Colony Optimization
efficiency for solving large TSP instances,” *Applied Soft Computing*, 2022,
article 108653: <https://doi.org/10.1016/j.asoc.2022.108653>.

The separate adaptive-tuning implementation and its paper are documented in the
[upstream repository](https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning).

OpenMP and the C++ standard library are linked through the user's compiler
toolchain. Compiler and runtime binaries are not included in this source release.
