#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace cvrp {
constexpr int32_t MAX_CAND_LIST_SIZE = 64;
constexpr float EPS = 1e-12f;
constexpr int64_t DEMAND_SCALE = 100000;

class Random {
public:
  void seed(uint64_t s) {
    // SplitMix64 initialization
    auto splitmix = [](uint64_t &x) {
      x += 0x9e3779b97f4a7c15ULL;
      uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
      return z ^ (z >> 31);
    };
    state_[0] = splitmix(s);
    state_[1] = splitmix(s);
  }

  uint32_t next_u32() {
    const uint64_t s0 = state_[0];
    uint64_t s1 = state_[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    state_[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    state_[1] = rotl(s1, 36);
    return static_cast<uint32_t>(result >> 32);
  }

  // Uniform [0, max_exclusive)
  uint32_t next_uint(uint32_t max_exclusive) {
    uint32_t x = next_u32();
    uint64_t m =
        static_cast<uint64_t>(x) * static_cast<uint64_t>(max_exclusive);
    uint32_t l = static_cast<uint32_t>(m);
    if (l < max_exclusive) {
      uint32_t t = -max_exclusive % max_exclusive;
      while (l < t) {
        x = next_u32();
        m = static_cast<uint64_t>(x) * static_cast<uint64_t>(max_exclusive);
        l = static_cast<uint32_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 32);
  }

  // Uniform [0, 1)
  float next_float() {
    return static_cast<float>(next_u32() >> 8) * (1.0f / (1u << 24));
  }

private:
  static constexpr uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  uint64_t state_[2] = {1, 2};
};

// Node order along a Hilbert curve over the coordinates (depot stays first):
// order[new_id] = old_id. Solving a renumbered instance keeps nodes that are
// close in space close in memory.
std::vector<int32_t> spatial_order(const float *coords, int32_t n);

// Routes use depot separators: [0, customer, ..., 0, customer, ..., 0].
struct SampleResult {
  std::vector<float> costs;
  std::vector<std::vector<int32_t>> routes;
};

class Solver {
public:
  const int32_t n, m;
  int32_t n_ants; // may grow during run_for (see grow_after)
  const int32_t k, bl, min_new_edges, fixed_steps;
  const float rho, alpha, p_best;
  const bool use_local_search, disable_heuristic, extend_ls, smooth_mmas,
      initial_local_search, extended_neighborhood, merge_ants;
  const float capacity;
  int64_t capacity_int;
  std::vector<float> coords, demand, d0, heuristic_sparse, pheromone_sparse;
  std::vector<int64_t> demand_int;
  // Internally customers are renumbered along a Hilbert curve (when
  // spatial_order is on) so that nearby customers are close in memory.
  // source_route and best_route use that internal numbering; every public
  // function below takes and returns the caller's original ids.
  std::vector<int32_t> nn_list, backup_list, source_route, best_route;
  std::vector<float> nn_dist; // dist(u, nn_list[u * k + j])
  float source_cost = 0, best_cost = 0, tau_min = 0, tau_max = 0;
  // Search-only timings, accumulated across calls; the initial solution is at t=0.
  double time_to_best = 0, search_seconds = 0;
  int64_t iterations_run = 0;
  // Wall seconds per search phase, accumulated over iterations.
  double phase_probabilities = 0, phase_links = 0, phase_ants = 0,
         phase_select = 0, phase_pheromone = 0;
  // Thread-seconds inside ants: copying the source, construction, local search.
  double ant_copy = 0, ant_construct = 0, ant_inter = 0, ant_intra = 0,
         ant_pack = 0;

  Solver(const float *coords, const float *demand, int32_t n, float capacity,
         int32_t n_ants = 100, int32_t candidates = 48, int32_t backup = 32,
         int32_t min_new_edges = 4, float decay = 0.1f, float alpha = 1.0f,
         float p_best = 0.05f, bool local_search = true,
         bool disable_heuristic = false, bool extend_ls = true,
         bool smooth_mmas = true, int32_t fixed_steps = 0,
         bool initial_local_search = true, bool extended_neighborhood = true,
         bool merge_ants = true, bool spatial_order = true,
         bool sweep_start = true);

  void seed_rng(uint64_t seed) { rng_.seed(seed); }
  void sample(SampleResult &result);
  float run(int32_t iterations);
  // The deadline is checked between complete iterations. Zero stagnation disables it.
  float run_for(double seconds, int32_t max_non_impr = 0);
  void update_pheromone(const std::vector<int32_t> &route, float cost);
  // The best solution in the caller's ids.
  std::vector<int32_t> best_solution() const { return to_external(best_route); }
  // Validate external solutions before they enter the unchecked search loop.
  float validate_route(const std::vector<int32_t> &route) const {
    return check_route(to_internal(route));
  }

  // Restarts from a feasible solution using the caller's node ids.
  void set_solution(const std::vector<int32_t> &route);
  // Full local search on the current solution.
  void polish();
  // Probability that an iteration continues from (and reinforces) the global
  // best instead of the iteration's best, as in Focused ACO. Zero draws no
  // random number, so the default search is unchanged.
  float global_best_source = 0;
  // Smooth MAX-MIN lower bound; 0 keeps the default 1/k.
  float tau_min_override = 0;
  // Forced modification (after AACO): 1 forbids an ant's first choice from
  // repeating the source's edge, 2 forbids it at every step, so every choice
  // creates a new edge. 0 is off.
  int32_t force_change = 0;
  // Bounded walk: after max_copy_run consecutive re-picked source edges, the
  // next choice must create a new edge. Re-picking lets one ant reach several
  // separate regions linked by walked source edges; the bound keeps it from
  // spending its steps copying. -1 is off; 0 is the same as forcing every step.
  int32_t max_copy_run = -1;
  // Escape (after PRISM's random_escape): once an ant's local search has
  // converged, up to escape_budget random feasible inter-route moves (relocate,
  // else swap) around the ant's changed nodes, each followed by a new descent.
  // The ant returns the best state it saw, so escapes never make it worse.
  // 0 is off.
  int32_t escape_budget = 0;
  // Doubles the ant count after every grow_after consecutive iterations
  // without improvement, up to max_ants (0: eight times the initial count).
  int32_t grow_after = 0, max_ants = 0;
  // Gives edges at the depot the mean heuristic weight of their row instead
  // of the savings value, which is exactly zero there.
  void use_neutral_depot_heuristic();
  // Ants open no route beyond this many (0: no limit), so that a search
  // started from a plan within a fleet limit stays within it.
  int32_t route_limit = 0;
private:
  // Scratch buffers belong to the solver and are reused by each OpenMP worker.
  // Per-node and per-route flags are epoch stamps (flag set <=> stamp ==
  // epoch), so starting an ant costs O(1) instead of clearing O(n) arrays.
  struct Workspace {
    std::vector<int32_t> next, prev, node_route, positions, checklist;
    std::vector<int32_t> route_lengths, touched_routes;
    std::vector<int32_t> checklist_scratch_a, checklist_scratch_b, region;
    std::vector<int64_t> loads, cumulative;
    // 16-bit stamps keep the per-node flags small enough to stay in cache at
    // 30K nodes; they are cleared in full when the counters wrap.
    using Stamp = uint16_t;
    std::vector<Stamp> visited, marked, dlb;
    std::vector<Stamp> cum_valid, pos_valid, touched;
    Stamp epoch = 0, mark_epoch = 0;
    double t_copy = 0, t_construct = 0, t_inter = 0, t_intra = 0, t_pack = 0;
  };
  // An ant's solution relative to the iteration's source: only the routes it
  // changed are stored; everything else is the source route with that id.
  struct AntResult {
    float cost = 0;
    double delta = 0; // cost change vs the source, over the changed routes
    int32_t route_count = 0;
    std::vector<int32_t> ids, offsets, nodes; // sorted ids; nodes per id
  };
  std::vector<int32_t> order_, rank_; // internal -> caller id, and inverse
  std::vector<int32_t> to_internal(const std::vector<int32_t> &route) const;
  std::vector<int32_t> to_external(const std::vector<int32_t> &route) const;
  float check_route(const std::vector<int32_t> &route) const;
  float distance_scale = 1;
  Random rng_;
  Workspace source_links;
  int32_t source_route_count = 0;
  std::vector<float> source_route_costs;
  std::vector<Workspace> workers;
  std::vector<AntResult> ants;
  std::vector<float> probmat, log_heuristic, heuristic_weight;
  std::vector<int32_t> start_nodes, positions, selected;
  // Per-source-route owner while merging ants (stamped by merge_epoch).
  std::vector<uint32_t> claim_stamp;
  std::vector<int32_t> claim_ant, claim_slot;
  uint32_t merge_epoch = 0;
  std::vector<uint64_t> ant_seeds;
  void reinforce(const std::vector<int32_t> &route, float cost);
  bool iterate();
  // Combines improving ants with disjoint changed source routes (in order of
  // improvement) into `route`; true when more than one ant was combined.
  bool merge_improving(double noise,
                       std::vector<int32_t> &route, double &merged_delta);
  void build_nn_lists();
  void build_heuristic();
  void build_d0();
  void build_initial_solution();
  // Sweep start: customers by angle around the depot, cut into routes at
  // capacity, each route ordered by nearest neighbour from the depot; the
  // cheapest of several starting angles is kept. With an off-centre depot this
  // gives petal-shaped routes, which local search cannot create from compact
  // clusters once routes are full.
  void build_sweep_solution();
  void polish_source();
  int32_t build_links(const std::vector<int32_t> &route, Workspace &work) const;
  void prepare(Workspace &work) const;
  void begin_ant(Workspace &work) const;
  void compute_probmat();
  void sample_ants();
  void materialize(const AntResult &ant, std::vector<int32_t> &route) const;
  void sample_ant_direct(const float *probmat, int32_t start, AntResult &out,
                         Workspace &work, Random &rng);
  float intra_route_ls(std::vector<int32_t> &checklist, Workspace &work);
  float inter_route_ls(std::vector<int32_t> &checklist, Workspace &work);
  float dist(int32_t u, int32_t v) const;
  float route_cost(const std::vector<int32_t> &route) const;
  std::pair<float, float> calc_trail_limits_cl(float cost) const;
  std::pair<float, float> calc_trail_limits_smooth(float cost) const;
};
} // namespace cvrp
