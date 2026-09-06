// Balancing the routed-expert split by measured firing frequency.
//
// The 144/144 split by expert id (src/fabric/expert_parallel.h) divides the
// experts evenly and the work unevenly: a rank's per-step cost is the number
// of (stream, expert) rows it computes plus the expert weights it has to
// stream in for them, and neither is uniform across expert ids because the
// router is not uniform across expert ids.
//
// The input is one number per routed expert: how often it fired, summed over
// every layer and every step of a telemetry run (model.h::expert_fire_counts,
// written by rocket-static-fire --expert-histogram-out). Summing over layers
// is what makes the histogram 288 numbers instead of 42 * 288, and it is the
// right granularity because the split assigns an expert id for every layer at
// once, not per layer.
//
// The partition is a greedy longest-processing-time bin pack over two bins,
// constrained to equal cardinality. Equal cardinality is not cosmetic: a
// rank's expert cache has to hold that rank's own experts, 91 GiB for 144 of
// them against about 82 GiB of headroom, so a partition that gave one rank
// more experts would push that rank's working set further past its cache
// while the other rank's sat idle.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rocket::fabric {

struct ExpertPartition {
  // owner[e] is the rank that owns expert e.
  std::vector<int> owner;
  // Predicted load of each rank, in fired rows, from the histogram.
  double load[2] = {0.0, 0.0};
  // load[busier] / load[idler]; 1.0 is a perfectly balanced partition.
  double imbalance = 1.0;

  std::vector<int> experts_of(int rank) const;
};

// Reads "<expert_id> <count>" lines, '#' comments ignored. Throws
// std::runtime_error if the file is missing or does not cover [0, n_experts).
std::vector<std::uint64_t> load_expert_histogram(const std::filesystem::path& path, int n_experts);

// Equal-cardinality greedy bin pack. Ties and equal counts resolve by expert
// id, so the partition is a pure function of the histogram and both ranks
// compute the same one without exchanging it.
ExpertPartition balance_experts(const std::vector<std::uint64_t>& counts);

// The id-contiguous split this replaces, for measuring one against the other.
ExpertPartition contiguous_partition(int n_experts, const std::vector<std::uint64_t>& counts);

}  // namespace rocket::fabric
