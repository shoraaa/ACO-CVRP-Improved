// Sparse CVRP ant colony search with focused iterated local search.
#include "cvrp.h"
#include "kd_tree.h"
#include <chrono>
#include <cstring>
#include <limits>
#include <omp.h>
#include <stdexcept>

namespace cvrp {

std::vector<int32_t> spatial_order(const float *coords, int32_t n) {
  float min_x = coords[0], max_x = coords[0], min_y = coords[1],
        max_y = coords[1];
  for (int32_t i = 1; i < n; ++i) {
    min_x = std::min(min_x, coords[2 * i]);
    max_x = std::max(max_x, coords[2 * i]);
    min_y = std::min(min_y, coords[2 * i + 1]);
    max_y = std::max(max_y, coords[2 * i + 1]);
  }
  const float span = std::max({max_x - min_x, max_y - min_y, 1e-9f});
  constexpr uint32_t side = 1u << 16;
  // Hilbert index of a cell (x, y) on a side x side grid.
  auto hilbert = [&](uint32_t x, uint32_t y) {
    uint64_t d = 0;
    for (uint32_t s = side / 2; s > 0; s /= 2) {
      const uint32_t rx = (x & s) > 0, ry = (y & s) > 0;
      d += uint64_t(s) * s * ((3 * rx) ^ ry);
      if (!ry) {
        if (rx) {
          x = s - 1 - x;
          y = s - 1 - y;
        }
        std::swap(x, y);
      }
    }
    return d;
  };
  std::vector<std::pair<uint64_t, int32_t>> keys;
  keys.reserve(n > 0 ? n - 1 : 0);
  for (int32_t i = 1; i < n; ++i) {
    auto cell = [&](float value, float low) {
      return std::min<uint32_t>(side - 1,
                                uint32_t((value - low) / span * (side - 1)));
    };
    keys.push_back({hilbert(cell(coords[2 * i], min_x),
                            cell(coords[2 * i + 1], min_y)),
                    i});
  }
  std::sort(keys.begin(), keys.end());
  std::vector<int32_t> order{0};
  for (const auto &key : keys)
    order.push_back(key.second);
  return order;
}

float Solver::route_cost(const std::vector<int32_t> &route) const {
  float cost = 0;
  for (size_t i = 1; i < route.size(); ++i)
    cost += dist(route[i - 1], route[i]);
  return cost;
}

std::vector<int32_t>
Solver::to_internal(const std::vector<int32_t> &route) const {
  if (rank_.empty())
    return route;
  std::vector<int32_t> mapped(route.size());
  for (size_t i = 0; i < route.size(); ++i)
    mapped[i] = route[i] >= 0 && route[i] < n ? rank_[route[i]] : route[i];
  return mapped; // invalid ids pass through for check_route to reject
}

std::vector<int32_t>
Solver::to_external(const std::vector<int32_t> &route) const {
  if (order_.empty())
    return route;
  std::vector<int32_t> mapped(route.size());
  for (size_t i = 0; i < route.size(); ++i)
    mapped[i] = order_[route[i]];
  return mapped;
}

float Solver::check_route(const std::vector<int32_t> &route) const {
  if (route.size() < 3 || route.front() != 0 || route.back() != 0)
    throw std::invalid_argument("route must start and end at depot 0");
  std::vector<uint8_t> seen(n, 0);
  int64_t load = 0;
  int32_t count = 0;
  for (int32_t u : route) {
    if (u < 0 || u >= n)
      throw std::invalid_argument("route contains an invalid customer");
    if (u == 0) {
      load = 0;
      continue;
    }
    if (seen[u]++)
      throw std::invalid_argument("route visits a customer more than once");
    ++count;
    load += demand_int[u];
    if (load > capacity_int)
      throw std::invalid_argument("route exceeds vehicle capacity");
  }
  if (count != m)
    throw std::invalid_argument("route must visit every customer");
  return route_cost(route);
}

void Solver::update_pheromone(const std::vector<int32_t> &external,
                              float cost) {
  const auto route = to_internal(external);
  const float actual = check_route(route);
  if (!std::isfinite(cost) ||
      std::abs(cost - actual) > 1e-5f * std::max(1.0f, actual))
    throw std::invalid_argument("cost does not match route");
  reinforce(route, actual);
}

int32_t Solver::build_links(const std::vector<int32_t> &route,
                            Workspace &work) const {
  work.next.resize(2 * n);
  work.prev.resize(2 * n);
  work.node_route.resize(2 * n);
  work.loads.resize(n);
  int32_t count = 0;
  int32_t prev = n;
  int64_t load = 0;
  for (int32_t u : route) {
    if (u == 0) {
      if (prev >= n)
        continue;
      work.next[prev] = n + count;
      work.prev[n + count] = prev;
      work.loads[count++] = load;
      prev = n + count;
      load = 0;
    } else {
      work.next[prev] = u;
      work.prev[u] = prev;
      work.node_route[u] = count;
      load += demand_int[u];
      prev = u;
    }
  }
  return count;
}

// Per-node loops use dynamic chunks: threads on slower (efficiency) cores
// take fewer chunks instead of holding everyone back. Nodes are independent,
// so results do not depend on the schedule.
void Solver::compute_probmat() {
  probmat.resize(pheromone_sparse.size());
  if (alpha == 1.0f) {
    // With alpha = 1 the weight is a plain product (as in FACO), with no
    // log/exp per edge; tiny weights keep the same relative floor.
#pragma omp parallel for schedule(dynamic, 256)
    for (int32_t u = 0; u < n; ++u) {
      float *row = probmat.data() + size_t(u) * k;
      float maximum = 0;
      for (int32_t j = 0; j < k; ++j) {
        const int32_t i = u * k + j;
        row[j] = (pheromone_sparse[i] + EPS) * heuristic_weight[i];
        maximum = std::max(maximum, row[j]);
      }
      const float floor = EPS * maximum;
      for (int32_t j = 0; j < k; ++j)
        row[j] = std::max(row[j], floor);
    }
    return;
  }
#pragma omp parallel for schedule(dynamic, 256)
  for (int32_t u = 0; u < n; ++u) {
    float logits[MAX_CAND_LIST_SIZE];
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t j = 0; j < k; ++j) {
      const int32_t i = u * k + j;
      logits[j] =
          alpha * std::log(pheromone_sparse[i] + EPS) + log_heuristic[i];
      maximum = std::max(maximum, logits[j]);
    }
    for (int32_t j = 0; j < k; ++j)
      probmat[u * k + j] = std::max(std::exp(logits[j] - maximum), EPS);
  }
}

void Solver::prepare(Workspace &work) const {
  work.next.resize(2 * n);
  work.prev.resize(2 * n);
  work.node_route.resize(2 * n);
  work.cumulative.resize(2 * n);
  work.loads.resize(n);
  work.positions.resize(n);
  work.route_lengths.resize(n);
  for (auto *stamps : {&work.visited, &work.marked, &work.dlb, &work.cum_valid,
                       &work.pos_valid, &work.touched})
    stamps->resize(n, 0);
}

// Starts a new ant: all per-node and per-route flags become unset in O(1).
void Solver::begin_ant(Workspace &work) const {
  // Escapes advance the mark counter up to 2 * escape_budget more per ant.
  const int32_t wrap =
      std::numeric_limits<Workspace::Stamp>::max() - 4 - 2 * escape_budget;
  if (work.epoch >= wrap || work.mark_epoch >= wrap) {
    for (auto *stamps : {&work.visited, &work.marked, &work.dlb,
                         &work.cum_valid, &work.pos_valid, &work.touched})
      std::fill(stamps->begin(), stamps->end(), 0);
    work.epoch = work.mark_epoch = 0;
  }
  ++work.epoch;
  ++work.mark_epoch;
  work.touched_routes.clear();
}

static double seconds_between(std::chrono::steady_clock::time_point a,
                              std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void Solver::sample_ants() {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  ants.resize(n_ants);
  compute_probmat();
  const auto t1 = Clock::now();
  phase_probabilities += seconds_between(t0, t1);
  source_route_count = build_links(source_route, source_links);
  // Per-route source costs let every ant price only the routes it changed.
  source_route_costs.assign(source_route_count, 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (int32_t r = 0; r < source_route_count; ++r) {
    float cost = 0;
    int32_t prev = 0;
    for (int32_t x = source_links.next[n + r]; x < n;
         x = source_links.next[x]) {
      cost += dist(prev, x);
      prev = x;
    }
    source_route_costs[r] = cost + dist(prev, 0);
  }
  start_nodes.resize(n_ants);
  ant_seeds.resize(n_ants);
  // Draw in the original order so ant scheduling cannot affect the search.
  for (int32_t a = 0; a < n_ants; ++a)
    start_nodes[a] = 1 + rng_.next_uint(m);
  for (int32_t a = 0; a < n_ants; ++a) {
    uint64_t hi = rng_.next_u32(), lo = rng_.next_u32();
    ant_seeds[a] = (hi << 32) ^ lo ^ (0x9e3779b97f4a7c15ULL + uint64_t(a));
  }
  const auto t2 = Clock::now();
  phase_links += seconds_between(t1, t2);
  const int threads = std::min(n_ants, omp_get_max_threads());
  workers.resize(threads);
#pragma omp parallel num_threads(threads)
  {
    auto &work = workers[omp_get_thread_num()];
    prepare(work);
    // Each ant has its own seed and starts from a reset workspace, so which
    // thread runs it does not matter: balance the uneven local-search work.
#pragma omp for schedule(dynamic, 1)
    for (int32_t a = 0; a < n_ants; ++a) {
      Random rng;
      rng.seed(ant_seeds[a]);
      sample_ant_direct(probmat.data(), start_nodes[a], ants[a], work, rng);
    }
  }
  phase_ants += seconds_between(t2, Clock::now());
  for (auto &w : workers) {
    ant_copy += w.t_copy, ant_construct += w.t_construct;
    ant_inter += w.t_inter, ant_intra += w.t_intra, ant_pack += w.t_pack;
    w.t_copy = w.t_construct = w.t_inter = w.t_intra = w.t_pack = 0;
  }
}

// Expands an ant into the full depot-separated route, in route-id order.
void Solver::materialize(const AntResult &ant,
                         std::vector<int32_t> &route) const {
  route.clear();
  route.reserve(m + ant.route_count + 2);
  size_t t = 0;
  for (int32_t r = 0; r < ant.route_count; ++r) {
    if (t < ant.ids.size() && ant.ids[t] == r) {
      if (ant.offsets[t] < ant.offsets[t + 1]) {
        route.push_back(0);
        route.insert(route.end(), ant.nodes.begin() + ant.offsets[t],
                     ant.nodes.begin() + ant.offsets[t + 1]);
      }
      ++t;
    } else if (r < source_route_count) {
      int32_t x = source_links.next[n + r];
      if (x < n)
        route.push_back(0);
      for (; x < n; x = source_links.next[x])
        route.push_back(x);
    }
  }
  if (!route.empty())
    route.push_back(0);
}

void Solver::sample(SampleResult &result) {
  sample_ants();
  result.costs.resize(n_ants);
  result.routes.resize(n_ants);
#pragma omp parallel for schedule(dynamic, 256)
  for (int32_t a = 0; a < n_ants; ++a) {
    materialize(ants[a], result.routes[a]);
    result.costs[a] = route_cost(result.routes[a]);
    result.routes[a] = to_external(result.routes[a]);
  }
}

bool Solver::iterate() {
  sample_ants();
  const auto select_start = std::chrono::steady_clock::now();
  // Ant costs are incremental (source cost plus changed routes). Only ants
  // within a tight tolerance of the minimum are expanded and priced exactly,
  // which selects the same ant as an exact argmin with first-index ties.
  float approx = std::numeric_limits<float>::infinity();
  for (const auto &ant : ants)
    approx = std::min(approx, ant.cost);
  const float tolerance = 1e-4f * std::max(1.0f, std::abs(approx));
  std::vector<int32_t> route;
  float exact = std::numeric_limits<float>::infinity();
  for (int32_t a = 0; a < n_ants; ++a) {
    if (ants[a].cost > approx + tolerance)
      continue;
    materialize(ants[a], route);
    const float cost = route_cost(route);
    if (cost < exact) {
      exact = cost;
      selected.swap(route);
    }
  }
  // Ant merging (after FILO-style many-local-changes acceptance): ants change
  // few routes, and at scale most improving ants touch disjoint routes. Their
  // changes are independent, so all improving ants whose changed source routes
  // are unclaimed are combined, in order of improvement (index breaks ties, so
  // the result does not depend on thread count). New routes are ant-private.
  if (merge_ants && exact < source_cost) {
    // Improvements are tiny relative to the total at scale, so compare the
    // double-precision deltas against float noise, not against `tolerance`.
    const double noise = 1e-7 * std::max(1.0f, source_cost);
    double merged_delta = 0;
    if (merge_improving(noise, route, merged_delta)) {
      const float cost = route_cost(route);
      if (cost < exact) {
        exact = cost;
        selected.swap(route);
      }
    }
  }
  const auto pheromone_start = std::chrono::steady_clock::now();
  phase_select += seconds_between(select_start, pheromone_start);
  const float previous = best_cost;
  if (global_best_source > 0 && rng_.next_float() < global_best_source) {
    selected = best_route;
    exact = best_cost;
  }
  reinforce(selected, exact);
  phase_pheromone +=
      seconds_between(pheromone_start, std::chrono::steady_clock::now());
  ++iterations_run;
  return best_cost < previous;
}

bool Solver::merge_improving(double noise,
                             std::vector<int32_t> &route,
                             double &merged_delta) {
  merged_delta = 0;
  std::vector<int32_t> order;
  auto delta_of = [&](int32_t a) {
    return ants[a].delta;
  };
  for (int32_t a = 0; a < n_ants; ++a)
    if (delta_of(a) < -noise)
      order.push_back(a);
  std::sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
    return delta_of(x) < delta_of(y) || (delta_of(x) == delta_of(y) && x < y);
  });
  if (claim_stamp.size() < size_t(source_route_count)) {
    claim_stamp.resize(source_route_count, 0);
    claim_ant.resize(source_route_count);
    claim_slot.resize(source_route_count);
  }
  if (++merge_epoch == 0) {
    std::fill(claim_stamp.begin(), claim_stamp.end(), 0);
    merge_epoch = 1;
  }
  std::vector<int32_t> accepted;
  // Under a route limit, an ant joins only if the merged plan stays within
  // it: every ant respects the limit, but the routes opened by several ants
  // add up.
  int64_t routes = 0;
  if (route_limit > 0)
    for (int32_t r = 0; r < source_route_count; ++r)
      routes += source_links.next[n + r] < n;
  for (int32_t a : order) {
    const auto &ids = ants[a].ids;
    bool free = true;
    for (int32_t r : ids)
      if (r < source_route_count && claim_stamp[r] == merge_epoch) {
        free = false;
        break;
      }
    if (!free)
      continue;
    if (route_limit > 0) {
      int64_t change = 0;
      for (size_t t = 0; t < ids.size(); ++t)
        change += int64_t(ants[a].offsets[t] < ants[a].offsets[t + 1]) -
                  int64_t(ids[t] < source_route_count &&
                          source_links.next[n + ids[t]] < n);
      if (routes + change > route_limit)
        continue;
      routes += change;
    }
    for (size_t t = 0; t < ids.size(); ++t)
      if (ids[t] < source_route_count) {
        claim_stamp[ids[t]] = merge_epoch;
        claim_ant[ids[t]] = a;
        claim_slot[ids[t]] = static_cast<int32_t>(t);
      }
    accepted.push_back(a);
    merged_delta += delta_of(a);
  }
  if (accepted.size() > 1) {
    route.clear();
    route.reserve(m + source_route_count + 16);
    auto append = [&](const AntResult &ant, size_t t) {
      if (ant.offsets[t] < ant.offsets[t + 1]) {
        route.push_back(0);
        route.insert(route.end(), ant.nodes.begin() + ant.offsets[t],
                     ant.nodes.begin() + ant.offsets[t + 1]);
      }
    };
    for (int32_t r = 0; r < source_route_count; ++r) {
      if (claim_stamp[r] == merge_epoch) {
        append(ants[claim_ant[r]], claim_slot[r]);
      } else {
        int32_t x = source_links.next[n + r];
        if (x < n)
          route.push_back(0);
        for (; x < n; x = source_links.next[x])
          route.push_back(x);
      }
    }
    for (int32_t a : accepted)
      for (size_t t = 0; t < ants[a].ids.size(); ++t)
        if (ants[a].ids[t] >= source_route_count)
          append(ants[a], t);
    route.push_back(0);
    return true;
  }
  return false;
}

void Solver::use_neutral_depot_heuristic() {
  double total = 0;
  size_t count = 0;
  for (int32_t u = 1; u < n; ++u)
    for (int32_t j = 0; j < k; ++j)
      if (nn_list[u * k + j] != 0) {
        total += heuristic_weight[u * k + j];
        ++count;
      }
  const float global = count ? float(total / count) : 1.0f;
  for (int32_t u = 0; u < n; ++u) {
    double row = 0;
    int32_t customers = 0;
    for (int32_t j = 0; j < k; ++j)
      if (u != 0 && nn_list[u * k + j] != 0) {
        row += heuristic_weight[u * k + j];
        ++customers;
      }
    const float value = customers ? float(row / customers) : global;
    for (int32_t j = 0; j < k; ++j)
      if (u == 0 || nn_list[u * k + j] == 0) {
        heuristic_weight[u * k + j] = value;
        log_heuristic[u * k + j] = std::log(value);
      }
  }
}

void Solver::set_solution(const std::vector<int32_t> &external) {
  const auto route = to_internal(external);
  const float cost = check_route(route);
  source_route = best_route = route;
  source_cost = best_cost = cost;
  // A given start gets the same global local search as the built-in one.
  if (use_local_search && initial_local_search)
    polish();
}

void Solver::polish() {
  const auto incumbent = best_route;
  const float incumbent_cost = best_cost;
  polish_source();
  // Recompute-based protection: a public polish/warm start must retain the
  // incumbent even if the move deltas disagree with the final route length.
  if (best_cost > incumbent_cost) {
    source_route = best_route = incumbent;
    source_cost = best_cost = incumbent_cost;
  }
}

float Solver::run(int32_t iterations) {
  if (iterations < 0)
    throw std::invalid_argument("iterations must be nonnegative");
  const auto start = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < iterations; ++i) {
    if (iterate())
      time_to_best = search_seconds + std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  }
  search_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  return best_cost;
}

float Solver::run_for(double seconds, int32_t max_non_impr) {
  if (!std::isfinite(seconds) || seconds < 0 || max_non_impr < 0)
    throw std::invalid_argument("time limit and stagnation limit must be nonnegative");
  const auto start = std::chrono::steady_clock::now();
  int32_t stale = 0;
  auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };
  const int32_t ant_cap = max_ants > 0 ? max_ants : 8 * n_ants;
  while (elapsed() < seconds && (!max_non_impr || stale < max_non_impr)) {
    if (iterate()) {
      stale = 0;
      time_to_best = search_seconds + elapsed();
    } else {
      ++stale;
      // Stagnation-triggered colony growth (instead of a fixed schedule).
      if (grow_after > 0 && stale % grow_after == 0 && n_ants < ant_cap)
        n_ants = std::min(2 * n_ants, ant_cap);
    }
  }
  search_seconds += elapsed();
  return best_cost;
}
Solver::Solver(const float *coords_ptr, const float *demand_ptr, int32_t n_,
               float capacity_, int32_t n_ants_, int32_t cand_list_size,
               int32_t backup_list_size, int32_t min_new_edges_, float decay,
               float alpha_, float p_best_, bool use_local_search_,
               bool disable_heuristic_, bool extend_ls_, bool smooth_mmas_,
               int32_t fixed_steps_, bool initial_local_search_,
               bool extended_neighborhood_, bool merge_ants_,
               bool spatial_order_, bool sweep_start)
    : n(n_), m(n_ > 1 ? n_ - 1 : 0), n_ants(n_ants_),
      k(std::clamp(cand_list_size, 0, m)),
      bl(std::min(backup_list_size, m - k)), min_new_edges(min_new_edges_),
      fixed_steps(fixed_steps_), rho(decay), alpha(alpha_), p_best(p_best_),
      use_local_search(use_local_search_),
      disable_heuristic(disable_heuristic_), extend_ls(extend_ls_),
      smooth_mmas(smooth_mmas_), initial_local_search(initial_local_search_),
      extended_neighborhood(extended_neighborhood_), merge_ants(merge_ants_),
      capacity(capacity_), capacity_int(0) {
  if (!coords_ptr || !demand_ptr || n < 2)
    throw std::invalid_argument("expected depot and at least one customer");
  if (n > std::numeric_limits<int32_t>::max() / (2 * MAX_CAND_LIST_SIZE) ||
      int64_t(n) * bl > std::numeric_limits<int32_t>::max())
    throw std::invalid_argument("too many nodes");
  if (n_ants <= 0 || cand_list_size <= 0 ||
      cand_list_size > MAX_CAND_LIST_SIZE || backup_list_size < 0 ||
      min_new_edges < 0 || fixed_steps < 0)
    throw std::invalid_argument(
        "invalid ant count, candidate size, or perturbation budget");
  if (!std::isfinite(rho) || rho <= 0 || rho >= 1 || !std::isfinite(alpha) ||
      alpha < 0 || !std::isfinite(p_best) || p_best <= 0 || p_best >= 1)
    throw std::invalid_argument(
        "require 0 < decay,p_best < 1 and finite alpha >= 0");
  if (!std::isfinite(capacity) || capacity < 1.0f / DEMAND_SCALE ||
      static_cast<double>(capacity) * DEMAND_SCALE >=
          static_cast<double>(std::numeric_limits<int64_t>::max()) / n)
    throw std::invalid_argument(
        "capacity is outside supported scaled integer range");
  capacity_int = static_cast<int64_t>(std::round(capacity * DEMAND_SCALE));
  for (int32_t i = 0; i < n; ++i) {
    if (!std::isfinite(coords_ptr[2 * i]) ||
        !std::isfinite(coords_ptr[2 * i + 1]) ||
        std::abs(coords_ptr[2 * i]) > 1e15f ||
        std::abs(coords_ptr[2 * i + 1]) > 1e15f)
      throw std::invalid_argument(
          "coordinates must be finite and within +/-1e15");
    if (!std::isfinite(demand_ptr[i]) || demand_ptr[i] < 0 ||
        demand_ptr[i] > capacity)
      throw std::invalid_argument(
          "each demand must be finite and between zero and capacity");
  }
  if (demand_ptr[0] != 0)
    throw std::invalid_argument("depot demand must be zero");

  if (spatial_order_) {
    order_ = spatial_order(coords_ptr, n);
    rank_.resize(n);
    for (int32_t i = 0; i < n; ++i)
      rank_[order_[i]] = i;
  }
  coords.resize(static_cast<size_t>(n) * 2);
  demand.resize(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const int32_t from = order_.empty() ? i : order_[i];
    coords[2 * i] = coords_ptr[2 * from];
    coords[2 * i + 1] = coords_ptr[2 * from + 1];
    demand[i] = demand_ptr[from];
  }

  demand_int.resize(n);
  for (int32_t i = 0; i < n; ++i) {
    demand_int[i] = (int64_t)std::round(demand[i] * DEMAND_SCALE);
  }

  // Scale move tolerances with the geometry to avoid roundoff-only cycles
  // when input coordinates are supplied in original benchmark units.
  float min_x = coords[0], max_x = coords[0];
  float min_y = coords[1], max_y = coords[1];
  for (int32_t i = 1; i < n; ++i) {
    min_x = std::min(min_x, coords[2 * i]);
    max_x = std::max(max_x, coords[2 * i]);
    min_y = std::min(min_y, coords[2 * i + 1]);
    max_y = std::max(max_y, coords[2 * i + 1]);
  }
  distance_scale = std::max({1.0f, max_x - min_x, max_y - min_y});
  build_nn_lists();
  nn_dist.resize(nn_list.size());
  for (int32_t u = 0; u < n; ++u)
    for (int32_t j = 0; j < k; ++j)
      nn_dist[u * k + j] = dist(u, nn_list[u * k + j]);

  build_d0();
  build_heuristic();

  source_route.reserve(n * 2);
  best_route.reserve(n * 2);

  if (sweep_start)
    build_sweep_solution();
  else
    build_initial_solution();
  // Converge the greedy start under the full local search once. Afterwards
  // every ant only re-optimizes around its own perturbation, so the incumbent
  // stays converged without paying for global passes.
  if (use_local_search && initial_local_search)
    polish_source();

  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(source_cost)
                                  : calc_trail_limits_cl(source_cost);
  tau_min = tmin;
  tau_max = tmax;
  pheromone_sparse.assign(n * k, tau_max);

  rng_.seed(42);
  log_heuristic.resize(heuristic_sparse.size());
  for (size_t i = 0; i < log_heuristic.size(); ++i)
    log_heuristic[i] =
        disable_heuristic ? 0.0f : std::log(heuristic_sparse[i] + EPS);
  heuristic_weight.resize(log_heuristic.size());
  for (size_t i = 0; i < log_heuristic.size(); ++i)
    heuristic_weight[i] =
        disable_heuristic ? 1.0f : heuristic_sparse[i] + EPS;
}

float Solver::dist(int32_t u, int32_t v) const {
  const size_t ou = static_cast<size_t>(u) * 2;
  const size_t ov = static_cast<size_t>(v) * 2;
  float dx = coords[ou] - coords[ov];
  float dy = coords[ou + 1] - coords[ov + 1];
  return std::sqrt(dx * dx + dy * dy);
}

void Solver::build_nn_lists() {
  nn_list.resize(n * k);
  backup_list.resize(n * bl);

  const int32_t total = k + bl;
  if (total <= 0)
    return;

  std::vector<Vec2d> pts(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const size_t off = static_cast<size_t>(i) * 2;
    pts[static_cast<size_t>(i)] =
        Vec2d{(double)coords[off], (double)coords[off + 1]};
  }

  KDTree shared_kdtree(pts);

#pragma omp parallel default(none) shared(shared_kdtree, nn_list, backup_list) \
    firstprivate(n, k, bl, total)
  {
    KDTree kdtree = shared_kdtree;
    std::vector<int32_t> deleted_nodes;
    deleted_nodes.reserve(total + 8);

#pragma omp for schedule(static)
    for (int32_t u = 0; u < n; ++u) {
      deleted_nodes.clear();
      int32_t current_k = 0;
      int32_t current_bl = 0;

      if (u > 0 && k > 0) {
        nn_list[u * k + 0] = 0;
        current_k = 1;
      }

      int32_t search_limit = total + 5;

      for (int32_t step = 0; step < search_limit; ++step) {
        if (current_k >= k && current_bl >= bl)
          break;

        uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
        int32_t v = static_cast<int32_t>(pt_idx);

        kdtree.delete_point(pt_idx);
        deleted_nodes.push_back(v);

        if (v == u)
          continue; // Skip self
        if (u > 0 && v == 0)
          continue; // Skip depot for customers (already added)

        if (current_k < k) {
          nn_list[u * k + current_k] = v;
          current_k++;
        } else if (current_bl < bl) {
          backup_list[u * bl + current_bl] = v;
          current_bl++;
        }
      }

      for (int32_t v_del : deleted_nodes) {
        kdtree.undelete_point(static_cast<uint32_t>(v_del));
      }
    }
  }
}

void Solver::build_heuristic() {
  heuristic_sparse.resize(n * k);
  if (disable_heuristic) {
    std::fill(heuristic_sparse.begin(), heuristic_sparse.end(), 1.0f);
    return;
  }
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float d = dist(u, v);
      float depot_u = d0[u];
      float depot_v = d0[v];

      d = std::max(0.0f, depot_u + depot_v - d);
      heuristic_sparse[u * k + j] = d;
    }
  }
}

void Solver::build_d0() {
  d0.resize(n);
  for (int32_t v = 0; v < n; ++v) {
    d0[v] = dist(0, v);
  }
}

void Solver::build_sweep_solution() {
  std::vector<int32_t> order(m);
  std::vector<float> angle(n, 0);
  for (int32_t i = 1; i < n; ++i) {
    order[i - 1] = i;
    angle[i] = std::atan2(coords[2 * i + 1] - coords[1], coords[2 * i] - coords[0]);
  }
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return angle[a] < angle[b] || (angle[a] == angle[b] && a < b);
  });
  // Route membership stamps for the nearest-neighbour ordering.
  std::vector<int32_t> member(n, -1), used(n, -1);
  std::vector<int32_t> route, plan, best_plan;
  float best = std::numeric_limits<float>::max();
  const int32_t rotations = std::min<int32_t>(12, std::max<int32_t>(1, m));
  int32_t stamp = 0;
  for (int32_t rotation = 0; rotation < rotations; ++rotation) {
    const int32_t first = int32_t(int64_t(rotation) * m / rotations);
    plan.assign(1, 0);
    float cost = 0;
    auto close = [&]() {
      // Order the collected customers by nearest neighbour from the depot:
      // candidates first, a scan of the route when none is left there.
      ++stamp;
      for (int32_t x : route)
        member[x] = stamp;
      int32_t curr = 0;
      for (size_t placed = 0; placed < route.size(); ++placed) {
        int32_t next = -1;
        if (curr != 0)
          for (int32_t j = 0; j < k; ++j) {
            const int32_t v = nn_list[curr * k + j];
            if (v > 0 && member[v] == stamp && used[v] != stamp) {
              next = v;
              break;
            }
          }
        if (next < 0) {
          float nearest = std::numeric_limits<float>::max();
          for (int32_t x : route)
            if (used[x] != stamp && dist(curr, x) < nearest) {
              nearest = dist(curr, x);
              next = x;
            }
        }
        used[next] = stamp;
        cost += dist(curr, next);
        plan.push_back(next);
        curr = next;
      }
      cost += dist(curr, 0);
      plan.push_back(0);
      route.clear();
    };
    int64_t load = 0;
    for (int32_t t = 0; t < m; ++t) {
      const int32_t x = order[(first + t) % m];
      if (!route.empty() && load + demand_int[x] > capacity_int) {
        close();
        load = 0;
      }
      route.push_back(x);
      load += demand_int[x];
    }
    if (!route.empty())
      close();
    if (cost < best) {
      best = cost;
      best_plan.swap(plan);
    }
  }
  best_route = source_route = best_plan;
  best_cost = source_cost = route_cost(best_route);
}

void Solver::build_initial_solution() {

  float best_c = std::numeric_limits<float>::max();
  best_route.clear();

  int32_t num_starts = std::min(8, n);

  std::vector<int32_t> current_route;
  current_route.reserve(n * 2);
  std::vector<uint8_t> visited(n, 0);

  for (int32_t s = 0; s < num_starts; ++s) {
    current_route.clear();
    std::fill(visited.begin(), visited.end(), 0);
    visited[0] = 1;

    int32_t start_node = 1 + (s % (n - 1));

    current_route.push_back(start_node);
    visited[start_node] = 1;

    int32_t curr = start_node;
    int64_t cur_cap = capacity_int - demand_int[start_node];
    float cost = dist(0, start_node);

    int32_t visited_cnt = 1;
    while (visited_cnt < n - 1) {
      int32_t best_nb = -1;
      float min_d = std::numeric_limits<float>::max();

      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[curr * k + j];
        if (v > 0 && !visited[v]) {
          if (demand_int[v] <= cur_cap) {
            float d = dist(curr, v);
            if (d < min_d) {
              min_d = d;
              best_nb = v;
              break; // NN is sorted
            }
          }
        }
      }

      if (best_nb == -1) {
        bool can_fit = false;
        for (int32_t v = 1; v < n; ++v) {
          if (!visited[v] && demand_int[v] <= cur_cap) {
            can_fit = true;
            if (dist(curr, v) < min_d) {
              min_d = dist(curr, v);
              best_nb = v;
            }
          }
        }
        if (!can_fit) {

          cost += dist(curr, 0);
          current_route.push_back(0); // Depot
          curr = 0;
          cur_cap = capacity_int;
          continue;
        }
      }

      if (best_nb != -1) {
        visited[best_nb] = 1;
        visited_cnt++;
        current_route.push_back(best_nb);
        cost += dist(curr, best_nb);
        cur_cap -= demand_int[best_nb];
        curr = best_nb;
      }
    }

    cost += dist(curr, 0);

    if (cost < best_c) {
      best_c = cost;
      best_route = current_route;
    }
  }

  if (!best_route.empty()) {
    std::vector<int32_t> full;
    full.reserve(best_route.size() + 2);
    full.push_back(0);
    for (int32_t x : best_route)
      full.push_back(x);
    full.push_back(0);
    best_route = full;

    source_cost = best_c;
    best_cost = best_c;
    source_route = best_route;
  }
}

// Full local search on the source solution: every customer starts in the
// checklist, and inter/intra passes alternate until neither improves.
void Solver::polish_source() {
  if (!use_local_search || source_route.size() < 3)
    return;
  Workspace work;
  prepare(work);
  const int32_t route_count = build_links(source_route, work);
  std::vector<int32_t> checklist, order(m);
  // A fixed pseudo-random scan order: first-improvement results should not
  // depend on how customers happen to be numbered.
  Random shuffle;
  shuffle.seed(0x5eed);
  for (int32_t i = 0; i < m; ++i)
    order[i] = i + 1;
  for (int32_t i = m - 1; i > 0; --i)
    std::swap(order[i], order[shuffle.next_uint(uint32_t(i + 1))]);
  for (int pass = 0; pass < 50; ++pass) {
    begin_ant(work);
    checklist.clear();
    for (int32_t v : order) {
      checklist.push_back(v);
      work.marked[v] = work.mark_epoch;
    }
    float gain = inter_route_ls(checklist, work);
    checklist.clear();
    for (int32_t v = 1; v < n; ++v)
      checklist.push_back(v);
    gain += intra_route_ls(checklist, work);
    if (gain <= 1e-6f * distance_scale)
      break;
  }
  source_route.clear();
  for (int32_t r = 0; r < route_count; ++r) {
    int32_t x = work.next[n + r];
    if (x < n)
      source_route.push_back(0);
    for (; x < n; x = work.next[x])
      source_route.push_back(x);
  }
  source_route.push_back(0);
  source_cost = route_cost(source_route);
  best_route = source_route;
  best_cost = source_cost;
}

std::pair<float, float>
Solver::calc_trail_limits_cl(float solution_cost) const {
  float tau_max_ = 1.0f / (solution_cost * (1.0f - rho) + EPS);
  float avg = static_cast<float>(std::max(2, k));
  float p = std::pow(p_best, 1.0f / avg);
  float tau_min_ =
      std::min(tau_max_, tau_max_ * (1.0f - p) / ((avg - 1.0f) * p + EPS));
  return {tau_min_, tau_max_};
}

std::pair<float, float>
Solver::calc_trail_limits_smooth(float solution_cost) const {
  (void)solution_cost;
  float tau_max_ = 1.0f;
  float denom = static_cast<float>(std::max<int32_t>(1, k));
  float tau_min_ = tau_min_override > 0 ? tau_min_override : 1.0f / denom;
  return {tau_min_, tau_max_};
}

void Solver::reinforce(const std::vector<int32_t> &best_route_in,
                       float new_best_cost) {

  if (new_best_cost < best_cost) {
    best_cost = new_best_cost;
    best_route = best_route_in;
  }

  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(best_cost)
                                  : calc_trail_limits_cl(best_cost);
  tau_min = tmin;
  tau_max = tmax;

  int32_t R = (int32_t)best_route_in.size();
  positions.assign(n, -1);
  auto &pos = positions;
  for (int32_t i = 0; i < R; ++i) {
    int32_t v = best_route_in[i];
    if (v != 0) {
      pos[v] = i;
    }
  }

  auto in_route_edge = [&](int32_t u, int32_t v) -> bool {
    if (u != 0 && v != 0) {
      int32_t pu = pos[u];
      int32_t pv = pos[v];
      return std::abs(pu - pv) == 1;
    }

    if (u == 0 && v == 0)
      return false; // Loop (0,0) not relevant

    int32_t cust = (u != 0) ? u : v;
    int32_t p = pos[cust];

    if (p > 0 && best_route_in[p - 1] == 0)
      return true;
    if (p < R - 1 && best_route_in[p + 1] == 0)
      return true;

    return false;
  };

  const float decay_factor = 1.0f - rho;
  const float deposit = (!smooth_mmas) ? (1.0f / (new_best_cost + EPS)) : 0.0f;

#pragma omp parallel for schedule(dynamic, 256)
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float &tau = pheromone_sparse[u * k + j];
      bool is_in = in_route_edge(u, v);

      if (smooth_mmas) {
        float target = is_in ? tau_max : tau_min;
        tau = decay_factor * tau + rho * target;
      } else {
        tau *= decay_factor;
        if (is_in) {
          tau += deposit;
        }
        tau = std::max(tau_min, std::min(tau_max, tau));
      }
    }
  }

  source_route = best_route_in;
  source_cost = new_best_cost;
}

// Neighbour-list 2-opt inside routes, in place on the workspace links.
// Positions are computed lazily for the routes the checklist reaches.
float Solver::intra_route_ls(std::vector<int32_t> &checklist, Workspace &work) {
  float total_improvement = 0.0f;
  auto &next_node = work.next;
  auto &prev_node = work.prev;
  auto &node_route = work.node_route;
  auto &pos_in_route = work.positions;
  auto &lengths = work.route_lengths;
  const uint32_t epoch = work.epoch;

  auto ensure_positions = [&](int32_t r) {
    if (work.pos_valid[r] == epoch)
      return;
    work.pos_valid[r] = epoch;
    int32_t count = 0;
    for (int32_t x = next_node[n + r]; x < n; x = next_node[x])
      pos_in_route[x] = count++;
    lengths[r] = count;
  };
  auto customer = [&](int32_t x) { return x < n ? x : 0; };

  // A fresh mark epoch: exactly the nodes currently in the checklist.
  const uint32_t mark = ++work.mark_epoch;
  auto &marked = work.marked;
  for (int32_t node : checklist)
    if (node > 0 && node < n)
      marked[node] = mark;

  std::vector<int32_t> &segment = work.checklist_scratch_a;
  size_t checklist_pos = 0;
  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a <= 0 || a >= n)
      continue;

    int32_t r = node_route[a];
    if (r < 0 || r >= n)
      continue;
    ensure_positions(r);
    const int32_t size = lengths[r];
    if (size < 2)
      continue;

    const int32_t a_pos = pos_in_route[a];
    const int32_t a_prev = customer(prev_node[a]);
    const int32_t a_next = customer(next_node[a]);

    float dist_a_prev = dist(a_prev, a);
    float dist_a_next = dist(a, a_next);

    float max_diff = 0.0f;
    int32_t best_i = -1, best_j = -1, first = -1, last = -1;

    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue; // Must be same route

      float dist_ab = dist(a, b);
      if (dist_a_prev <= dist_ab)
        break; // NN list is sorted

      int32_t b_pos = pos_in_route[b];
      if (b_pos == a_pos)
        continue;

      int32_t b_prev = customer(prev_node[b]);

      if (b_pos > a_pos) {
        float d_old = dist_a_prev + dist(b_prev, b);
        float d_new = dist(a_prev, b_prev) + dist_ab;
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = a_pos, best_j = b_pos, first = a, last = b_prev;
        }
      } else {
        float d_old = dist(b_prev, b) + dist_a_prev;
        float d_new = dist(b_prev, a) + dist(b, a_prev);
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = b_pos, best_j = a_pos, first = b, last = a_prev;
        }
      }
    }

    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue;

      float dist_ab = dist(a, b);
      if (dist_a_next <= dist_ab)
        break;

      int32_t b_pos = pos_in_route[b];
      if (b_pos == a_pos)
        continue;

      int32_t b_next = customer(next_node[b]);

      if (b_pos > a_pos) {
        float d_old = dist_a_next + dist(b, b_next);
        float d_new = dist_ab + dist(a_next, b_next);
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = a_pos + 1, best_j = b_pos + 1, first = a_next, last = b;
        }
      }
    }

    if (max_diff > 1e-6f * distance_scale && best_i >= 0 && best_j > best_i) {
      // Reverse positions [best_i, best_j): nodes first..last.
      const int32_t before = prev_node[first], after = next_node[last];
      segment.clear();
      for (int32_t x = first;; x = next_node[x]) {
        segment.push_back(x);
        if (x == last)
          break;
      }
      int32_t cursor = before;
      for (auto it = segment.rbegin(); it != segment.rend(); ++it) {
        next_node[cursor] = *it;
        prev_node[*it] = cursor;
        cursor = *it;
      }
      next_node[cursor] = after;
      prev_node[after] = cursor;
      for (int32_t i = 0; i < static_cast<int32_t>(segment.size()); ++i)
        pos_in_route[segment[segment.size() - 1 - i]] = best_i + i;
      total_improvement += max_diff;
      if (work.touched[r] != epoch) {
        work.touched[r] = epoch;
        work.touched_routes.push_back(r);
      }

      if (extend_ls) {
        auto add = [&](int32_t node) {
          if (node > 0 && node < n && marked[node] != mark) {
            checklist.push_back(node);
            marked[node] = mark;
          }
        };
        // Positions best_i-1 .. min(size, best_j+1)-1, in route order.
        if (best_i > 0)
          add(before);
        for (auto it = segment.rbegin(); it != segment.rend(); ++it)
          add(*it);
        if (best_j < size)
          add(after);
      }
    }
  }

  return total_improvement;
}

// Operates in place on the workspace links. Route loads and node_route are
// always current; cumulative loads are computed lazily per route, so the cost
// is proportional to the routes the search actually reads or changes.
float Solver::inter_route_ls(std::vector<int32_t> &checklist, Workspace &work) {
  float total_improvement = 0.0f;
  auto &next_node = work.next;
  auto &prev_node = work.prev;
  auto &node_route = work.node_route;
  auto &cum_demand = work.cumulative;
  auto &route_loads = work.loads;
  const uint32_t epoch = work.epoch, mark = work.mark_epoch;
  auto &marked = work.marked;
  auto &dlb_stamp = work.dlb;
  // dlb[u] set <=> stamp == epoch; a fresh ant has every bit cleared.
  struct Bit {
    Workspace::Stamp &stamp;
    uint32_t epoch;
    operator bool() const { return stamp == epoch; }
    Bit &operator=(bool on) {
      stamp = Workspace::Stamp(on ? epoch : 0);
      return *this;
    }
  };
  auto dlb_at = [&](int32_t u) { return Bit{dlb_stamp[u], epoch}; };
  auto in_checklist_at = [&](int32_t u) { return Bit{marked[u], mark}; };

  auto dist = [&](int32_t u, int32_t v) {
    int32_t ru = (u >= n) ? 0 : u;
    int32_t rv = (v >= n) ? 0 : v;
    float dx = coords[ru * 2] - coords[rv * 2];
    float dy = coords[ru * 2 + 1] - coords[rv * 2 + 1];
    return std::sqrt(dx * dx + dy * dy);
  };

  auto touch = [&](int32_t u) {
    if (u < n) {
      dlb_at(u) = false;
      if (!in_checklist_at(u)) {
        checklist.push_back(u);
        in_checklist_at(u) = true;
      }
    }
  };

  // Recomputes one route's cumulative loads; `changed` records the route as
  // modified by this ant (a pure read does not).
  auto update_route_state = [&](int32_t start_node, int32_t route_id,
                                bool changed = true) {
    int32_t curr = start_node;
    if (curr < n)
      return;
    work.cum_valid[route_id] = epoch;
    if (changed && work.touched[route_id] != epoch) {
      work.touched[route_id] = epoch;
      work.touched_routes.push_back(route_id);
    }

    cum_demand[curr] = 0;
    node_route[curr] = route_id;
    int64_t load = 0;

    curr = next_node[curr];
    while (curr < n) {
      load += demand_int[curr];
      cum_demand[curr] = load;
      node_route[curr] = route_id;
      curr = next_node[curr];
    }
    route_loads[route_id] = load;
  };

  auto ensure_cumulative = [&](int32_t route_id) {
    if (work.cum_valid[route_id] != epoch)
      update_route_state(n + route_id, route_id, false);
  };

  if (checklist.empty()) {
    return 0.0f; // Nothing to do
  }

  const float EPS = 1e-5f * distance_scale;

  // Extended localized neighbourhood around a candidate pair (u, v) from u's
  // neighbour list. Every move places u next to v: intra-route relocate,
  // Or-opt of 2-3 consecutive customers (inter and intra, both orientations),
  // and the four 2-opt* variants that create edge (u, v). A route r is the
  // chain following separator node n + r, so 2-opt* is executed by relinking
  // the four chain pieces between the separators.
  auto &piece_a = work.checklist_scratch_a, &piece_b = work.checklist_scratch_b;
  auto relink = [&](int32_t start, const std::vector<int32_t> &nodes,
                    int32_t end) {
    int32_t last = start;
    for (int32_t x : nodes) {
      next_node[last] = x;
      prev_node[x] = last;
      last = x;
    }
    next_node[last] = end;
    prev_node[end] = last;
  };
  // Appends customers from..to (inclusive) of one route; an endpoint that is a
  // separator denotes an empty piece.
  auto chain = [&](int32_t from, int32_t to_inclusive,
                   std::vector<int32_t> &out, bool append) {
    if (!append)
      out.clear();
    if (from >= n || to_inclusive >= n)
      return;
    for (int32_t x = from; x < n; x = next_node[x]) {
      out.push_back(x);
      if (x == to_inclusive)
        break;
    }
  };
  auto route_end = [&](int32_t x) {
    while (x < n)
      x = next_node[x];
    return x;
  };
  auto route_last = [&](int32_t x) {
    while (next_node[x] < n)
      x = next_node[x];
    return x;
  };
  // Edge lengths shared by every move type. u's are set once per scanned
  // node, v's once per candidate pair; u's Or-opt segments (ignoring v) and
  // their removal terms once per node. Same expressions, fewer square roots.
  float d_pu_u = 0, d_u_nu = 0, d_pu_nu = 0, d_uv = 0, d_v_nv = 0, d_pv_v = 0;
  int32_t seg_u[3], seg_u_length = 0;
  float removed_u[3], bridge_u[3];

  auto extended_move = [&](int32_t u, int32_t v) -> bool {
    const int32_t r_u = node_route[u], r_v = node_route[v];
    const int32_t prev_u = prev_node[u], next_u = next_node[u];
    const int32_t prev_v = prev_node[v], next_v = next_node[v];
    // Or-opt: segment u..last of 1..3 consecutive customers not containing v
    // (length 1 only intra; inter single relocation is handled above).
    const int32_t *seg = seg_u;
    int32_t length = 0;
    while (length < seg_u_length && seg[length] != v)
      ++length;
    int64_t load = 0;
    for (int32_t len = 1; len <= length; ++len) {
      const int32_t last = seg[len - 1];
      load += demand_int[last];
      if (r_u != r_v && (len == 1 || route_loads[r_v] + load > capacity_int))
        continue;
      const int32_t after = next_node[last];
      const float removed = removed_u[len - 1];
      const float bridge = bridge_u[len - 1];
      // Forward, after v: v -> u .. last -> next_v.
      if (v != prev_u) {
        float delta = bridge + d_uv + dist(last, next_v) - removed - d_v_nv;
        if (delta < -EPS) {
          next_node[prev_u] = after;
          prev_node[after] = prev_u;
          const int32_t nv = next_node[v];
          next_node[v] = u;
          prev_node[u] = v;
          next_node[last] = nv;
          prev_node[nv] = last;
          update_route_state(n + r_u, r_u);
          if (r_v != r_u)
            update_route_state(n + r_v, r_v);
          touch(u), touch(last), touch(prev_u), touch(after), touch(v),
              touch(nv);
          total_improvement -= delta;
          return true;
        }
      }
      // Reversed, before v: prev_v -> last .. u -> v.
      if (v != after) {
        float delta = bridge + dist(prev_v, last) + d_uv - removed - d_pv_v;
        if (delta < -EPS) {
          next_node[prev_u] = after;
          prev_node[after] = prev_u;
          const int32_t pv = prev_node[v];
          int32_t cursor = pv;
          for (int32_t t = len - 1; t >= 0; --t) {
            next_node[cursor] = seg[t];
            prev_node[seg[t]] = cursor;
            cursor = seg[t];
          }
          next_node[cursor] = v;
          prev_node[v] = cursor;
          update_route_state(n + r_u, r_u);
          if (r_v != r_u)
            update_route_state(n + r_v, r_v);
          touch(u), touch(last), touch(prev_u), touch(after), touch(v),
              touch(pv);
          total_improvement -= delta;
          return true;
        }
      }
    }
    if (r_u == r_v)
      return false;
    // 2-opt* variants creating edge (u, v); loads from cumulative demand.
    ensure_cumulative(r_u);
    ensure_cumulative(r_v);
    const int64_t head_u = cum_demand[u], head_v = cum_demand[v];
    const int64_t load_u = route_loads[r_u], load_v = route_loads[r_v];
    const int64_t du = demand_int[u], dv = demand_int[v];
    // Loads first: a variant's length change is computed only if both new
    // routes fit (same expressions and order as before, so same results).
    const int64_t loads[4][2] = {
        // A: (u,next_u),(prev_v,v) -> (u,v),(prev_v,next_u)
        {head_u + load_v - (head_v - dv), (head_v - dv) + load_u - head_u},
        // B: (prev_u,u),(v,next_v) -> (v,u),(prev_u,next_v)
        {(head_u - du) + load_v - head_v, head_v + load_u - (head_u - du)},
        // C: (u,next_u),(v,next_v) -> (u,v),(next_u,next_v), reversing heads
        {head_u + head_v, (load_u - head_u) + (load_v - head_v)},
        // D: (prev_u,u),(prev_v,v) -> (u,v),(prev_u,prev_v), reversing tails
        {(head_u - du) + (head_v - dv),
         (load_u - head_u + du) + (load_v - head_v + dv)}};
    for (int32_t variant = 0; variant < 4; ++variant) {
      if (loads[variant][0] > capacity_int || loads[variant][1] > capacity_int)
        continue;
      const float base = d_uv;
      const float delta =
          variant == 0   ? base + dist(prev_v, next_u) - d_u_nu - d_pv_v
          : variant == 1 ? base + dist(prev_u, next_v) - d_pu_u - d_v_nv
          : variant == 2 ? base + dist(next_u, next_v) - d_u_nu - d_v_nv
                         : base + dist(prev_u, prev_v) - d_pu_u - d_pv_v;
      if (!(delta < -EPS))
        continue;
      const int32_t d_u = n + r_u, d_v = n + r_v;
      const int32_t e_u = route_end(u), e_v = route_end(v);
      const int32_t first_u = next_node[d_u], first_v = next_node[d_v];
      const int32_t last_u = route_last(u), last_v = route_last(v);
      if (variant == 0) {
        chain(first_u, u, piece_a, false);
        chain(v, last_v, piece_a, true);
        chain(first_v, prev_v, piece_b, false);
        chain(next_u, last_u, piece_b, true);
        relink(d_u, piece_a, e_v);
        relink(d_v, piece_b, e_u);
      } else if (variant == 1) {
        chain(first_u, prev_u, piece_a, false);
        chain(next_v, last_v, piece_a, true);
        chain(first_v, v, piece_b, false);
        chain(u, last_u, piece_b, true);
        relink(d_u, piece_a, e_v);
        relink(d_v, piece_b, e_u);
      } else if (variant == 2) {
        chain(first_u, u, piece_a, false);
        const size_t mark = piece_a.size();
        chain(first_v, v, piece_a, true);
        std::reverse(piece_a.begin() + mark, piece_a.end());
        chain(next_u, last_u, piece_b, false);
        std::reverse(piece_b.begin(), piece_b.end());
        chain(next_v, last_v, piece_b, true);
        relink(d_u, piece_a, e_u);
        relink(d_v, piece_b, e_v);
      } else {
        chain(first_u, prev_u, piece_a, false);
        const size_t mark = piece_a.size();
        chain(first_v, prev_v, piece_a, true);
        std::reverse(piece_a.begin() + mark, piece_a.end());
        chain(u, last_u, piece_b, false);
        std::reverse(piece_b.begin(), piece_b.end());
        chain(v, last_v, piece_b, true);
        relink(d_u, piece_a, e_u);
        relink(d_v, piece_b, e_v);
      }
      update_route_state(d_u, r_u);
      update_route_state(d_v, r_v);
      touch(u), touch(v), touch(prev_u), touch(next_u), touch(prev_v),
          touch(next_v);
      total_improvement -= delta;
      return true;
    }
    return false;
  };

  int32_t head = 0;
  while (head < (int32_t)checklist.size()) {
    int32_t u = checklist[head++];
    in_checklist_at(u) = false;

    if (u == 0 || dlb_at(u))
      continue;

    bool improved = false;
    int32_t r_u = node_route[u];
    // u's neighbourhood is fixed until a move is kept (then the scan stops),
    // so its edge lengths are computed once; same values, fewer square roots.
    d_pu_u = dist(prev_node[u], u);
    d_u_nu = dist(u, next_node[u]);
    d_pu_nu = dist(prev_node[u], next_node[u]);
    seg_u_length = 0;
    for (int32_t x = u; seg_u_length < 3 && x < n; x = next_node[x]) {
      seg_u[seg_u_length] = x;
      removed_u[seg_u_length] = d_pu_u + dist(x, next_node[x]);
      bridge_u[seg_u_length] = dist(prev_node[u], next_node[x]);
      ++seg_u_length;
    }
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      if (v == 0)
        continue;
      d_uv = nn_dist[u * k + j];
      d_v_nv = dist(v, next_node[v]);
      d_pv_v = dist(prev_node[v], v);
      // Cross edges used by several move types, computed on first use.
      float c_u_nv = -1, c_pv_u = -1, c_v_nu = -1;
      auto d_u_nv = [&] {
        return c_u_nv < 0 ? (c_u_nv = dist(u, next_node[v])) : c_u_nv;
      };
      auto d_pv_u = [&] {
        return c_pv_u < 0 ? (c_pv_u = dist(prev_node[v], u)) : c_pv_u;
      };
      auto d_v_nu = [&] {
        return c_v_nu < 0 ? (c_v_nu = dist(v, next_node[u])) : c_v_nu;
      };

      int32_t r_v = node_route[v];

      int32_t next_u = next_node[u];
      int32_t prev_v = prev_node[v];

      if (r_u != r_v) {
        if (route_loads[r_v] + demand_int[u] <= capacity_int) {

          int32_t prev_u = prev_node[u];
          float delta =
              d_pu_nu + d_uv + d_u_nv() - d_pu_u - d_u_nu - d_v_nv;

          if (delta < -EPS) {
            next_node[prev_u] = next_u;
            prev_node[next_u] = prev_u;

            int32_t old_next_v = next_node[v];
            next_node[v] = u;
            prev_node[u] = v;
            next_node[u] = old_next_v;
            prev_node[old_next_v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(v);
            touch(prev_u);
            touch(next_u);
            touch(old_next_v);
            improved = true;
            total_improvement -= delta; // delta is negative
            break;
          }

          float delta2 =
              d_pu_nu + d_pv_u() + d_uv - d_pu_u - d_u_nu - d_pv_v;

          if (delta2 < -EPS) {
            next_node[prev_u] = next_u;
            prev_node[next_u] = prev_u;

            next_node[prev_v] = u;
            prev_node[u] = prev_v;
            next_node[u] = v;
            prev_node[v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(prev_v);
            touch(v);
            touch(prev_u);
            touch(next_u);
            improved = true;
            total_improvement -= delta2;
            break;
          }
        }
      }

      if (r_u != r_v) {
        int64_t load_u_new = route_loads[r_u] - demand_int[u] + demand_int[v];
        int64_t load_v_new = route_loads[r_v] - demand_int[v] + demand_int[u];

        if (load_u_new <= capacity_int && load_v_new <= capacity_int) {
          int32_t prev_u = prev_node[u];
          float delta = dist(prev_u, v) + d_v_nu() + d_pv_u() + d_u_nv() -
                        d_pu_u - d_u_nu - d_pv_v - d_v_nv;
          if (delta < -EPS) {
            int32_t nu = next_node[u], pu = prev_node[u];
            int32_t nv = next_node[v], pv = prev_node[v];
            next_node[pu] = v;
            prev_node[nu] = v;
            next_node[pv] = u;
            prev_node[nv] = u;
            next_node[u] = nv;
            prev_node[u] = pv;
            next_node[v] = nu;
            prev_node[v] = pu;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(v);
            touch(pu);
            touch(nu);
            touch(pv);
            touch(nv);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }

      if (r_u != r_v) {
        ensure_cumulative(r_u);
        ensure_cumulative(r_v);
        int64_t head_u = cum_demand[u];
        int64_t tail_u = route_loads[r_u] - head_u;
        int64_t head_v = cum_demand[v];
        int64_t tail_v = route_loads[r_v] - head_v;

        if (head_u + tail_v <= capacity_int &&
            head_v + tail_u <= capacity_int) {
          float delta = d_u_nv() + d_v_nu() - d_u_nu - d_v_nv;
          if (delta < -EPS) {
            int32_t nu = next_node[u];
            int32_t nv = next_node[v];
            next_node[u] = nv;
            prev_node[nv] = u;
            next_node[v] = nu;
            prev_node[nu] = v;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            touch(u);
            touch(v);
            touch(nu);
            touch(nv);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }
      if (extended_neighborhood && extended_move(u, v)) {
        improved = true;
        break;
      }
    }
    if (!improved)
      dlb_at(u) = true;
  }

  return total_improvement;
}

void Solver::sample_ant_direct(const float *probmat, int32_t start_node,
                               AntResult &out, Workspace &work, Random &rng) {
  auto &next_node = work.next;
  auto &prev_node = work.prev;
  auto &node_route = work.node_route;
  auto &route_loads = work.loads;
  using Clock = std::chrono::steady_clock;
  const auto t_begin = Clock::now();
  begin_ant(work);
  const uint32_t epoch = work.epoch;
  auto touch_route = [&](int32_t r) {
    if (work.touched[r] != epoch) {
      work.touched[r] = epoch;
      work.touched_routes.push_back(r);
    }
  };
  std::copy_n(source_links.next.begin(), n + source_route_count,
              next_node.begin());
  std::copy_n(source_links.prev.begin(), n + source_route_count,
              prev_node.begin());
  std::copy_n(source_links.node_route.begin(), n, node_route.begin());
  std::copy_n(source_links.loads.begin(), source_route_count,
              route_loads.begin());
  int32_t num_routes = source_route_count;
  const int32_t max_routes = route_limit > 0 ? std::min(n, route_limit) : n;
  const auto t_copied = Clock::now();
  work.t_copy += seconds_between(t_begin, t_copied);

  auto is_source_edge = [&](int32_t u, int32_t v) {
    if (u == 0)
      return source_links.next[v] >= n || source_links.prev[v] >= n;
    if (v == 0)
      return source_links.next[u] >= n || source_links.prev[u] >= n;
    return source_links.next[u] == v || source_links.prev[u] == v;
  };

  auto &visited = work.visited;
  auto &checklist = work.checklist;
  int32_t visited_count = 0;

  int32_t curr = start_node;
  visited[curr] = epoch;
  visited_count++;

  checklist.clear();
  checklist.push_back(curr);
  auto &in_checklist = work.marked;
  const uint32_t mark = work.mark_epoch;
  in_checklist[curr] = mark;

  int32_t new_edges_cross = 0;
  int32_t copy_run = 0; // consecutive re-picked source edges
  int32_t steps = 0;
  int32_t max_steps = m * 4;
  if (fixed_steps > 0)
    max_steps = fixed_steps;

  while (true) {

    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges_cross >= min_new_edges || visited_count >= m)
        break;
    }
    if (steps > max_steps)
      break;

    int32_t r_curr;
    if (curr >= n) {
      r_curr = curr - n;
    } else {
      r_curr = node_route[curr];
    }

    int32_t chosen = -1;

    {
      int32_t candidates[MAX_CAND_LIST_SIZE + 1];
      float probs[MAX_CAND_LIST_SIZE + 1];
      int32_t c_size = 0;
      float sum_prob = 0.0f;

      int32_t lookup_node = (curr >= n) ? 0 : curr;
      const float *row_prob = probmat + (size_t)lookup_node * k;

      for (int32_t j = 0; j < k; ++j) {
        int32_t v = nn_list[lookup_node * k + j];

        if (v == 0) {

          if (curr >= n)
            continue;
        } else {
          if (v < 0 || v >= n)
            continue;
          if (visited[v] == epoch)
            continue;

          int32_t r_v = node_route[v];
          if (r_v != r_curr) {
            if (route_loads[r_curr] + demand_int[v] > capacity_int)
              continue;
          }
          // Forced modification: do not spend the step on a source edge.
          const bool forced =
              (force_change && (force_change == 2 || steps == 0)) ||
              (max_copy_run >= 0 && copy_run >= max_copy_run);
          if (forced && is_source_edge(curr >= n ? 0 : curr, v))
            continue;
        }

        candidates[c_size] = v;
        probs[c_size] = row_prob[j];
        sum_prob += row_prob[j];
        c_size++;
      }

      if (c_size > 0) {
        float r = rng.next_float() * sum_prob;
        float run = 0.0f;
        chosen = candidates[c_size - 1];
        for (int32_t i = 0; i < c_size; ++i) {
          run += probs[i];
          if (r <= run) {
            chosen = candidates[i];
            break;
          }
        }
      }
    }

    if (chosen == -1) {
      int32_t lookup_node = (curr >= n) ? 0 : curr;
      for (int32_t j = 0; j < bl; ++j) {
        int32_t v = backup_list[lookup_node * bl + j];
        if (v == 0) {
          if (curr >= n)
            continue;
          chosen = 0;
          break;
        }
        if (v > 0 && v < n && visited[v] != epoch) {
          int32_t r_v = node_route[v];
          if (r_curr == r_v ||
              route_loads[r_curr] + demand_int[v] <= capacity_int) {
            chosen = v;
            break;
          }
        }
      }
    }

    if (chosen == -1) {
      float min_d = std::numeric_limits<float>::max();
      int32_t best_global = -1;
      int32_t lookup_node = (curr >= n) ? 0 : curr;

      for (int32_t v = 1; v < n; ++v) {
        if (visited[v] != epoch) {
          int32_t r_v = node_route[v];
          if (r_curr == r_v ||
              route_loads[r_curr] + demand_int[v] <= capacity_int) {
            float d = dist(lookup_node, v);
            if (d < min_d) {
              min_d = d;
              best_global = v;
            }
          }
        }
      }

      if (best_global == -1) {
        if (curr < n)
          chosen = 0; // Force close/split if customer
      } else {

        float d_depot = dist(lookup_node, 0);
        if (curr < n && d_depot < min_d) {
          chosen = 0;
        } else {
          chosen = best_global;
        }
      }
    }

    if (chosen == -1)
      break;

    copy_run = chosen > 0 && prev_node[chosen] == curr ? copy_run + 1 : 0;
    if (chosen == 0) {
      int32_t next_c = next_node[curr];

      if (next_c >= n) {

        ++new_edges_cross;

        curr = next_c;
        steps++;
        continue;
      }

      if (num_routes >= max_routes) {
        break;
      }

      int32_t r_new = num_routes++;
      int32_t new_depot = n + r_new;
      int32_t old_depot = n + r_curr;

      int32_t route_end = prev_node[old_depot];

      next_node[curr] = old_depot;
      prev_node[old_depot] = curr;

      next_node[new_depot] = next_c;
      prev_node[next_c] = new_depot;

      next_node[route_end] = new_depot;
      prev_node[new_depot] = route_end;

      int64_t load_shift = 0;
      int32_t w = next_c;
      while (w != new_depot && w < n) {
        node_route[w] = r_new;
        load_shift += demand_int[w];
        w = next_node[w];
      }
      route_loads[r_curr] -= load_shift;
      route_loads[r_new] = load_shift;
      touch_route(r_curr);
      touch_route(r_new);

      if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
        if (in_checklist[curr >= n ? 0 : curr] != mark) {
          checklist.push_back(curr >= n ? 0 : curr);
          in_checklist[curr >= n ? 0 : curr] = mark;
        }
        new_edges_cross++;
      }

      curr = new_depot; // We are now at the start of new route
      steps++;
      continue;
    }

    int32_t v = chosen;
    int32_t r_v = node_route[v];

    int32_t prev_v = prev_node[v];
    int32_t next_v = next_node[v];

    if (prev_v == curr) {

      visited[v] = epoch;
      visited_count++;
      curr = v;
      steps++;
      continue;
    }

    next_node[prev_v] = next_v;
    prev_node[next_v] = prev_v;

    int32_t next_c = next_node[curr];
    next_node[curr] = v;
    prev_node[v] = curr;
    next_node[v] = next_c;
    prev_node[next_c] = v;

    if (r_curr != r_v) {
      if (r_v != -1) {
        route_loads[r_v] -= demand_int[v];
      }
      route_loads[r_curr] += demand_int[v];
      node_route[v] = r_curr;
    }
    touch_route(r_curr);
    if (r_v >= 0)
      touch_route(r_v);

    visited[v] = epoch;
    visited_count++;

    int32_t u_idx = (curr >= n) ? 0 : curr;
    if (!is_source_edge(u_idx, v)) {
      if (in_checklist[u_idx] != mark) {
        checklist.push_back(u_idx);
        in_checklist[u_idx] = mark;
      }
      if (in_checklist[v] != mark) {
        checklist.push_back(v);
        in_checklist[v] = mark;
      }

      bool is_cross = false;
      if (u_idx == 0)
        is_cross = true; // Depot->Node is boundary edge
      else if (source_links.node_route[u_idx] != source_links.node_route[v])
        is_cross = true;

      if (is_cross)
        new_edges_cross++;
    }

    curr = v;
    steps++;
  }

  const auto t_built = Clock::now();
  work.t_construct += seconds_between(t_copied, t_built);
  const bool escaping = use_local_search && escape_budget > 0;
  if (escaping)
    work.region.assign(checklist.begin(), checklist.end());
  if (use_local_search && !checklist.empty()) {
    inter_route_ls(checklist, work);
    const auto t_inter = Clock::now();
    work.t_inter += seconds_between(t_built, t_inter);
    intra_route_ls(checklist, work);
    work.t_intra += seconds_between(t_inter, Clock::now());
  }

  // Store only the changed routes and price them against the source.
  auto pack = [&](AntResult &result) {
    const auto t_packing = Clock::now();
    auto &ids = work.touched_routes;
    std::sort(ids.begin(), ids.end());
    result.route_count = num_routes;
    result.ids.assign(ids.begin(), ids.end());
    result.offsets.clear();
    result.nodes.clear();
    double delta = 0;
    for (int32_t r : ids) {
      result.offsets.push_back(static_cast<int32_t>(result.nodes.size()));
      float route = 0;
      int32_t prev = 0;
      for (int32_t x = next_node[n + r]; x < n; x = next_node[x]) {
        result.nodes.push_back(x);
        route += dist(prev, x);
        prev = x;
      }
      if (prev)
        route += dist(prev, 0);
      delta += double(route) -
               (r < source_route_count ? source_route_costs[r] : 0.0f);
    }
    result.offsets.push_back(static_cast<int32_t>(result.nodes.size()));
    result.delta = delta;
    result.cost = static_cast<float>(source_cost + delta);
    work.t_pack += seconds_between(t_packing, Clock::now());
  };
  pack(out);
  if (!escaping || work.region.empty())
    return;

  // Escapes: a random feasible move near the ant's changes, then descent; the
  // ant keeps the best packed state (the champion).
  thread_local AntResult trial;
  const float EPS = 1e-5f * distance_scale;
  auto place = [&](int32_t x, int32_t after) {
    const int32_t p = prev_node[x], q = next_node[x];
    next_node[p] = q, prev_node[q] = p;
    const int32_t a = next_node[after];
    next_node[after] = x, prev_node[x] = after;
    next_node[x] = a, prev_node[a] = x;
  };
  auto invalidate = [&](int32_t r) {
    work.cum_valid[r] = 0;
    work.pos_valid[r] = 0;
    touch_route(r);
  };
  for (int32_t e = 0; e < escape_budget; ++e) {
    const uint32_t fresh = ++work.mark_epoch;
    checklist.clear();
    auto touch = [&](int32_t x) {
      if (x > 0 && x < n && work.marked[x] != fresh) {
        work.dlb[x] = 0;
        work.marked[x] = fresh;
        checklist.push_back(x);
      }
    };
    bool moved = false;
    for (int32_t attempt = 0; attempt < 16 && !moved; ++attempt) {
      const int32_t u =
          work.region[rng.next_uint(uint32_t(work.region.size()))];
      if (u <= 0 || u >= n)
        continue;
      const int32_t v = nn_list[size_t(u) * k + rng.next_uint(k)];
      if (v <= 0 || v >= n)
        continue;
      const int32_t ru = node_route[u], rv = node_route[v];
      if (ru == rv)
        continue;
      const int32_t pu = prev_node[u], nu = next_node[u];
      const int32_t pv = prev_node[v], nv = next_node[v];
      if (route_loads[rv] + demand_int[u] <= capacity_int &&
          (pu < n || nu < n)) {
        place(u, v); // relocate u after v
        route_loads[ru] -= demand_int[u], route_loads[rv] += demand_int[u];
        node_route[u] = rv;
        touch(u), touch(pu), touch(nu), touch(v), touch(nv);
      } else if (route_loads[ru] - demand_int[u] + demand_int[v] <=
                     capacity_int &&
                 route_loads[rv] - demand_int[v] + demand_int[u] <=
                     capacity_int &&
                 nu != v && nv != u) {
        place(u, v); // swap u and v: u after v, then v where u was
        place(v, pu);
        route_loads[ru] += demand_int[v] - demand_int[u];
        route_loads[rv] += demand_int[u] - demand_int[v];
        node_route[u] = rv, node_route[v] = ru;
        touch(u), touch(v), touch(pu), touch(nu), touch(pv), touch(nv);
      } else {
        continue;
      }
      invalidate(ru), invalidate(rv);
      moved = true;
    }
    if (!moved)
      break;
    // Each pass caches loads (inter) or positions (intra) per route for the
    // whole ant; the other pass has since changed the touched routes.
    for (int32_t r : work.touched_routes)
      work.cum_valid[r] = 0;
    const auto t_escape = Clock::now();
    inter_route_ls(checklist, work);
    const auto t_inter = Clock::now();
    work.t_inter += seconds_between(t_escape, t_inter);
    for (int32_t r : work.touched_routes)
      work.pos_valid[r] = 0;
    intra_route_ls(checklist, work);
    work.t_intra += seconds_between(t_inter, Clock::now());
    pack(trial);
    if (trial.cost < out.cost - EPS)
      std::swap(out, trial);
  }
}

} // namespace cvrp
