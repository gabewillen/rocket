// Teacher-forced GLM-5.3 accuracy and throughput measurement.
// Kept separate from rocket-decode so scoring I/O and corpus parsing cannot
// alter the production serving executable.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fabric/expert_balance.h"
#include "fabric/expert_parallel.h"
#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
const char* arg_value(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return fallback;
}
std::vector<int> parse_token_line(const std::string& line, int line_no) {
  std::vector<int> ids;
  std::istringstream in(line);
  long long id = 0;
  while (in >> id) {
    if (id < 0 || id > 0x7fffffffll)
      throw std::runtime_error("token id out of range on line " + std::to_string(line_no));
    ids.push_back(static_cast<int>(id));
  }
  if (!in.eof()) throw std::runtime_error("non-integer token field on line " + std::to_string(line_no));
  return ids;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const char* token_file = arg_value(argc, argv, "--token-file", nullptr);
    const char* text_file = arg_value(argc, argv, "--text-file", nullptr);
    const char* output_file = arg_value(argc, argv, "--output", nullptr);
    const char* logits_file = arg_value(argc, argv, "--logits", nullptr);
    const int max_score_tokens = std::atoi(arg_value(argc, argv, "--max-score-tokens", "0"));
    const int batch = std::atoi(arg_value(argc, argv, "--batch", "8"));
    const int score_chunk = std::atoi(arg_value(argc, argv, "--score-chunk", "7"));
    const int max_tokens = std::atoi(arg_value(argc, argv, "--max-tokens", "128"));
    const double cache_gib = std::atof(arg_value(argc, argv, "--expert-cache-gib", "90"));
    const int rank = std::atoi(arg_value(argc, argv, "--rank", "-1"));
    if ((!token_file && !text_file) || (token_file && text_file) || !output_file ||
        max_score_tokens < 0 || batch <= 0 || max_tokens <= 0 || score_chunk < 1 ||
        score_chunk > rocket::engine::DecodeEngine::kSpecMax)
      throw std::runtime_error("use exactly one input and --output; batch and sizing must be positive; score chunk must be in [1,8]");

    const auto snapshot = rocket::fuel::default_nvfp4_snapshot_dir();
    const auto attention = rocket::fuel::default_attention_yaml();
    if (snapshot.empty() || attention.empty()) throw std::runtime_error("fuel paths are not configured");
    const auto cfg = rocket::fuel::load_model_config(attention, snapshot);
    const rocket::fuel::Tokenizer tokenizer(snapshot / "tokenizer.json");

    std::unique_ptr<rocket::fabric::ExpertParallel> ep;
    if (rank >= 0) {
      rocket::fabric::Config fc;
      fc.rank = rank;
      fc.bootstrap_host = arg_value(argc, argv, "--host", "192.168.100.10");
      fc.bootstrap_port = std::atoi(arg_value(argc, argv, "--port", "18779"));
      const auto part = rocket::fabric::contiguous_partition(cfg.n_routed_experts, {});
      ep = std::make_unique<rocket::fabric::ExpertParallel>(
          fc, part.owner, batch * rocket::engine::DecodeEngine::kSpecMax * cfg.num_experts_per_tok,
          cfg.hidden_size);
    }
    rocket::engine::DecodeEngine engine(
        cfg, snapshot, static_cast<std::size_t>(cache_gib * (1ull << 30)), max_tokens, batch);
    if (ep) {
      engine.weights().set_expert_set(ep->owned_experts());
      engine.set_expert_parallel(ep.get());
      engine.weights().preload_owned_experts(nullptr);
      ep->barrier();
    }

    const bool emit = rank <= 0;
    std::ofstream json;
    std::ofstream binary;
    if (emit) {
      json.open(output_file);
      if (!json) throw std::runtime_error(std::string("cannot write ") + output_file);
      json << "{\"type\":\"metadata\",\"schema\":\"rocket.glm53.teacher-score.v1\","
           << "\"vocab_size\":" << cfg.vocab_size << ",\"batch\":" << batch
           << ",\"score_chunk\":" << score_chunk << "}\n";
      if (logits_file) {
        binary.open(logits_file, std::ios::binary | std::ios::trunc);
        if (!binary) throw std::runtime_error(std::string("cannot write ") + logits_file);
        const char magic[8] = {'R','K','T','L','O','G','1','\0'};
        const std::uint32_t vocab = static_cast<std::uint32_t>(cfg.vocab_size);
        const std::uint64_t rows = 0;
        binary.write(magic, sizeof(magic));
        binary.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
        binary.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
      }
    }

    std::ifstream input(token_file ? token_file : text_file);
    if (!input) throw std::runtime_error("cannot read score input");
    std::string line;
    std::uint64_t row = 0;
    int line_no = 0, sequence = 0;
    std::vector<int> next;
    std::vector<double> inference_ms;
    auto emit_row = [&](const std::vector<float>& logits, int sequence_id, std::size_t position,
                        int input_id, int target) {
      if (!emit) return;
      const float max_logit = *std::max_element(logits.begin(), logits.end());
      double exp_sum = 0.0;
      for (float x : logits) exp_sum += std::exp(static_cast<double>(x - max_logit));
      const double logsumexp = static_cast<double>(max_logit) + std::log(exp_sum);
      if (target < 0 || target >= cfg.vocab_size) throw std::runtime_error("target outside vocabulary");
      std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                          std::greater<std::pair<float, int>>> top;
      for (int v = 0; v < cfg.vocab_size; ++v) {
        const std::pair<float, int> item{logits[v], v};
        if (top.size() < 5) top.push(item);
        else if (item > top.top()) { top.pop(); top.push(item); }
      }
      std::vector<std::pair<float, int>> top5;
      while (!top.empty()) { top5.push_back(top.top()); top.pop(); }
      std::reverse(top5.begin(), top5.end());
      json << "{\"type\":\"token\",\"sequence\":" << sequence_id
           << ",\"position\":" << position << ",\"input_id\":" << input_id
           << ",\"target_id\":" << target << ",\"target_logprob\":"
           << (static_cast<double>(logits[target]) - logsumexp)
           << ",\"logsumexp\":" << logsumexp << ",\"top5\":[";
      for (std::size_t i = 0; i < top5.size(); ++i) {
        if (i) json << ',';
        json << "{\"id\":" << top5[i].second << ",\"logprob\":"
             << (static_cast<double>(top5[i].first) - logsumexp) << '}';
      }
      json << ']';
      if (binary) {
        json << ",\"logit_row\":" << row;
        binary.write(reinterpret_cast<const char*>(logits.data()),
                     static_cast<std::streamsize>(logits.size() * sizeof(float)));
      }
      json << "}\n";
      ++row;
    };
    const auto wall_start = Clock::now();
    while (std::getline(input, line)) {
      ++line_no;
      if (line.empty() || line[0] == '#') continue;
      std::vector<int> ids = token_file ? parse_token_line(line, line_no) : tokenizer.encode(line);
      if (max_score_tokens > 0 && ids.size() > static_cast<std::size_t>(max_score_tokens))
        ids.resize(static_cast<std::size_t>(max_score_tokens));
      if (ids.size() < 2) continue;
      ++sequence;
      engine.reset();
      for (std::size_t pos = 0; pos + 1 < ids.size();) {
        const int count = std::min<int>(score_chunk, ids.size() - 1 - pos);
        const auto token_start = Clock::now();
        std::vector<int> chunk(static_cast<std::size_t>(count) * batch);
        for (int j = 0; j < count; ++j)
          for (int m = 0; m < batch; ++m)
            chunk[static_cast<std::size_t>(j) * batch + m] = ids[pos + j];
        engine.step_spec(chunk, count, next, false);
        if (cudaDeviceSynchronize() != cudaSuccess)
          throw std::runtime_error("CUDA synchronize after score step failed");
        inference_ms.push_back(ms_since(token_start));
        if (emit)
          for (int j = 0; j < count; ++j)
            emit_row(engine.last_logits_row(j * batch), sequence, pos + j, ids[pos + j],
                     ids[pos + j + 1]);
        engine.commit_positions(std::vector<int>(batch, count));
        pos += count;
      }
    }
    if (emit) {
      const double wall_ms = ms_since(wall_start);
      const double model_ms = std::accumulate(inference_ms.begin(), inference_ms.end(), 0.0);
      std::sort(inference_ms.begin(), inference_ms.end());
      const double median_ms = inference_ms.empty() ? 0.0 : inference_ms[inference_ms.size() / 2];
      const double stream_tps = model_ms > 0 ? row * 1000.0 / model_ms : 0.0;
      const double aggregate_tps = stream_tps * batch;
      json << "{\"type\":\"summary\",\"sequences\":" << sequence
           << ",\"tokens\":" << row << ",\"batch\":" << batch
           << ",\"score_chunk\":" << score_chunk << ",\"model_ms\":" << model_ms
           << ",\"model_tokens_per_second\":"
           << aggregate_tps << ",\"model_tokens_per_second_per_stream\":" << stream_tps
           << ",\"median_step_ms\":" << median_ms << ",\"wall_ms\":" << wall_ms
           << ",\"wall_tokens_per_second\":"
           << (wall_ms > 0 ? row * batch * 1000.0 / wall_ms : 0.0) << "}\n";
      std::printf("score %llu positions x batch %d: %.2f aggregate tok/s, %.2f tok/s/stream, %.2f ms median\n",
                  static_cast<unsigned long long>(row), batch, aggregate_tps, stream_tps, median_ms);
      if (binary) {
        binary.seekp(8 + sizeof(std::uint32_t));
        binary.write(reinterpret_cast<const char*>(&row), sizeof(row));
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
