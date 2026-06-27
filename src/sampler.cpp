#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cpullm {

Sampler::Sampler(GenerationConfig config) : config_(config), state_(config.seed ? config.seed : 0x9E3779B97F4A7C15ull) {}

static std::uint64_t next_u64(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint32_t Sampler::sample(std::span<const float> logits) {
  if (logits.empty()) throw std::invalid_argument("empty logits");
  if (config_.temperature <= 0.0f) {
    return static_cast<std::uint32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  std::vector<std::uint32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  const std::size_t k = std::min<std::size_t>(config_.top_k == 0 ? logits.size() : config_.top_k, logits.size());
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](auto a, auto b) { return logits[a] > logits[b]; });
  ids.resize(k);

  const float max_logit = logits[ids.front()];
  std::vector<float> probs;
  probs.reserve(ids.size());
  float sum = 0.0f;
  for (auto id : ids) {
    const float p = std::exp((logits[id] - max_logit) / config_.temperature);
    probs.push_back(p);
    sum += p;
  }
  for (auto& p : probs) p /= sum;

  if (config_.top_p < 1.0f) {
    float cumulative = 0.0f;
    std::size_t keep = probs.size();
    for (std::size_t i = 0; i < probs.size(); ++i) {
      cumulative += probs[i];
      if (cumulative >= config_.top_p) { keep = i + 1; break; }
    }
    ids.resize(std::max<std::size_t>(1, keep));
    probs.resize(ids.size());
    const float renorm = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (auto& p : probs) p /= renorm;
  }

  const double u = static_cast<double>(next_u64(state_) >> 11) * (1.0 / 9007199254740992.0);
  double cdf = 0.0;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    cdf += probs[i];
    if (u <= cdf) return ids[i];
  }
  return ids.back();
}

} // namespace cpullm
