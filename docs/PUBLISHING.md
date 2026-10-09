# Publish this folder as a new repository

This is a standalone deterministic CVRP release. Use this folder as the repository
root. It contains no parent Git history, downloaded datasets, neural checkpoints,
stochastic workflow, or research manuscripts.

## Validate

Run from this folder:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/cvrp_solver --input examples/tiny.txt --time-limit 1 --route
```

The project uses MIT, as chosen for this release. Keep `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and the `licenses/` directory together. The copyright line
uses “CVRP ACO Core contributors”; it can be replaced with the preferred author or
organization credit.

## Create the first commit

Suggested repository name: `cvrp-aco-core`.

Suggested description: “Fast deterministic CVRP solver with sparse ant colony
optimization, focused local search, and OpenMP parallelism.”

Suggested topics: `ant-colony-optimization`, `cvrp`, `vehicle-routing`,
`combinatorial-optimization`, `cpp`, `openmp`.

If this folder is still inside the research checkout, enter it explicitly before
running Git commands. `git init` creates its own repository; do not stage the
parent research repository to publish this release.

```sh
cd /path/to/aco-core-release
git init -b main
git add .
git diff --cached --stat
git diff --cached --check
git status --short
git commit -m "Release standalone deterministic CVRP ACO core"
```

Review the staged file list before committing. `.gitignore` excludes local build
products, results, downloaded data, credentials, and backup files. Ignore rules
do not remove files already tracked in an existing repository, which is why this
folder starts with fresh history.

## Push to GitHub

Using an authenticated GitHub CLI:

```sh
gh repo create cvrp-aco-core --public --source=. --remote=origin --push
```

Or create an empty repository in the GitHub UI, then substitute your account below:

```sh
git remote add origin https://github.com/YOUR_ACCOUNT/cvrp-aco-core.git
git push -u origin main
```

After GitHub Actions passes, tag the first release if desired:

```sh
git tag -a v1.0.0 -m "Deterministic CVRP ACO core"
git push origin v1.0.0
```

For research citations, cite the published repository URL and exact commit/tag.
Add publication metadata only once its authors, title, and identifier are final.
