// The balanced expert partition (src/fabric/expert_balance.h). Pure host code
// over a synthetic histogram, so it needs neither a checkpoint nor a peer and
// never skips.
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "fabric/expert_balance.h"

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& detail) {
  std::printf("  %-58s %s%s%s\n", what, ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ",
              detail.c_str());
  if (!ok) ++failures;
}

}  // namespace

int main() {
  using rocket::fabric::balance_experts;
  using rocket::fabric::contiguous_partition;

  // A histogram whose load is entirely on the low ids: the id split is the
  // worst possible partition, the balanced one has to fix it.
  const int n = 16;
  std::vector<std::uint64_t> skewed(static_cast<std::size_t>(n), 1);
  for (int e = 0; e < n / 2; ++e) skewed[static_cast<std::size_t>(e)] = 100;

  const auto id_split = contiguous_partition(n, skewed);
  const auto balanced = balance_experts(skewed);

  check("id split on a skewed histogram is badly imbalanced", id_split.imbalance > 10.0,
        std::to_string(id_split.imbalance) + "x");
  check("balanced partition is within 1% of even", balanced.imbalance < 1.01,
        std::to_string(balanced.imbalance) + "x");

  int owned[2] = {0, 0};
  for (const int r : balanced.owner) ++owned[r];
  check("both ranks own the same number of experts", owned[0] == n / 2 && owned[1] == n / 2,
        std::to_string(owned[0]) + " / " + std::to_string(owned[1]));

  // Every expert is owned exactly once, by rank 0 or rank 1, and the two
  // ranks' sets partition the id space; a rank that computed a different
  // partition would deadlock the exchange rather than fail a check.
  const std::vector<int> a = balanced.experts_of(0);
  const std::vector<int> b = balanced.experts_of(1);
  std::vector<int> seen(static_cast<std::size_t>(n), 0);
  for (const int e : a) ++seen[static_cast<std::size_t>(e)];
  for (const int e : b) ++seen[static_cast<std::size_t>(e)];
  bool once = true;
  for (const int c : seen) once = once && (c == 1);
  check("the two owned sets partition every expert exactly once", once, "");

  // Determinism: the same histogram must give the same partition on both
  // ranks, which is the only reason neither has to send the other its set.
  const auto again = balance_experts(skewed);
  check("the partition is a pure function of the histogram", again.owner == balanced.owner, "");

  // A flat histogram is already balanced, and the pack must not make it worse.
  const std::vector<std::uint64_t> flat(static_cast<std::size_t>(n), 7);
  check("a flat histogram packs to exactly even", balance_experts(flat).imbalance == 1.0, "");

  // Round trip through the on-disk format the telemetry run writes.
  const std::string path = "/tmp/rocket-test-expert-histogram.txt";
  {
    std::ofstream out(path);
    out << "# comment line\n";
    for (int e = 0; e < n; ++e) out << e << " " << skewed[static_cast<std::size_t>(e)] << "\n";
  }
  const auto read_back = rocket::fabric::load_expert_histogram(path, n);
  check("histogram round trips through the file format", read_back == skewed, "");

  bool threw = false;
  try {
    rocket::fabric::load_expert_histogram(path, n + 2);  // file does not cover every expert
  } catch (const std::exception&) {
    threw = true;
  }
  check("a histogram missing an expert is rejected", threw, "");

  std::printf("\n%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
