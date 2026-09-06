#include "fabric/expert_balance.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rocket::fabric {
namespace {

void fill(ExpertPartition& p, const std::vector<std::uint64_t>& counts) {
  p.load[0] = p.load[1] = 0.0;
  for (std::size_t e = 0; e < counts.size(); ++e)
    p.load[p.owner[e]] += static_cast<double>(counts[e]);
  const double lo = std::min(p.load[0], p.load[1]);
  const double hi = std::max(p.load[0], p.load[1]);
  p.imbalance = lo > 0.0 ? hi / lo : 0.0;
}

}  // namespace

std::vector<int> ExpertPartition::experts_of(int rank) const {
  std::vector<int> out;
  for (std::size_t e = 0; e < owner.size(); ++e)
    if (owner[e] == rank) out.push_back(static_cast<int>(e));
  return out;
}

std::vector<std::uint64_t> load_expert_histogram(const std::filesystem::path& path,
                                                 int n_experts) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("expert histogram not readable: " + path.string());
  std::vector<std::uint64_t> counts(static_cast<std::size_t>(n_experts), 0);
  std::vector<char> seen(static_cast<std::size_t>(n_experts), 0);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    long long id = -1;
    unsigned long long c = 0;
    if (!(ls >> id >> c)) continue;
    if (id < 0 || id >= n_experts)
      throw std::runtime_error("expert histogram " + path.string() + " has out-of-range id " +
                               std::to_string(id));
    counts[static_cast<std::size_t>(id)] = c;
    seen[static_cast<std::size_t>(id)] = 1;
  }
  for (int e = 0; e < n_experts; ++e)
    if (seen[static_cast<std::size_t>(e)] == 0)
      throw std::runtime_error("expert histogram " + path.string() + " is missing expert " +
                               std::to_string(e));
  return counts;
}

ExpertPartition balance_experts(const std::vector<std::uint64_t>& counts) {
  const int n = static_cast<int>(counts.size());
  if (n % 2 != 0) throw std::runtime_error("routed experts must split evenly across the pair");
  const int cap = n / 2;

  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  // Heaviest first; expert id breaks ties so the result does not depend on
  // the sort's stability or on the order the histogram was written in.
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (counts[static_cast<std::size_t>(a)] != counts[static_cast<std::size_t>(b)])
      return counts[static_cast<std::size_t>(a)] > counts[static_cast<std::size_t>(b)];
    return a < b;
  });

  ExpertPartition p;
  p.owner.assign(static_cast<std::size_t>(n), -1);
  double load[2] = {0.0, 0.0};
  int taken[2] = {0, 0};
  for (const int e : order) {
    int r;
    if (taken[0] == cap) {
      r = 1;
    } else if (taken[1] == cap) {
      r = 0;
    } else {
      r = (load[0] <= load[1]) ? 0 : 1;  // rank 0 wins ties, so both ranks agree
    }
    p.owner[static_cast<std::size_t>(e)] = r;
    load[r] += static_cast<double>(counts[static_cast<std::size_t>(e)]);
    ++taken[r];
  }
  fill(p, counts);
  return p;
}

ExpertPartition contiguous_partition(int n_experts, const std::vector<std::uint64_t>& counts) {
  ExpertPartition p;
  p.owner.assign(static_cast<std::size_t>(n_experts), 0);
  for (int e = n_experts / 2; e < n_experts; ++e) p.owner[static_cast<std::size_t>(e)] = 1;
  if (!counts.empty()) fill(p, counts);
  return p;
}

}  // namespace rocket::fabric
