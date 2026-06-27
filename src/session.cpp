#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <vector>

namespace cpullm {

KVCache::KVCache(std::size_t layers, std::size_t context, std::size_t heads, std::size_t head_dim)
    : layers_(layers), context_(context), heads_(heads), head_dim_(head_dim) {
  const std::size_t elems = layers_ * context_ * heads_ * head_dim_;
  keys_.resize(elems);
  values_.resize(elems);
}

std::size_t KVCache::bytes() const noexcept { return (keys_.size() + values_.size()) * sizeof(float); }
std::size_t KVCache::capacity_tokens() const noexcept { return context_; }
void KVCache::reset() noexcept { used_tokens_ = 0; }
bool KVCache::append_slot() noexcept {
  if (used_tokens_ >= context_) return false;
  ++used_tokens_;
  return true;
}

InferenceSession::InferenceSession(const Model& model, GenerationConfig config)
    : model_(model), config_(config), kv_cache_(2, std::max<std::size_t>(model.metadata().context_length, 128), 1, 64) {}

std::string InferenceSession::generate(std::string_view prompt) {
  std::ostringstream out;
  out << prompt;
  generate_stream(prompt, [&](const TokenEvent& event) {
    out << event.text;
    return true;
  });
  return out.str();
}

void InferenceSession::generate_stream(std::string_view prompt, const TokenCallback& callback) {
  auto prompt_ids = tokenizer_.encode(prompt);
  Sampler sampler(config_);
  std::vector<float> logits(256);
  std::uint64_t rolling = 1469598103934665603ull;
  for (auto id : prompt_ids) rolling = (rolling ^ id) * 1099511628211ull;

  for (std::size_t i = 0; i < config_.max_tokens; ++i) {
    if (!kv_cache_.append_slot()) break;
    for (std::size_t j = 0; j < logits.size(); ++j) {
      const auto mixed = static_cast<std::uint32_t>((rolling >> ((j % 8) * 8)) ^ (j * 2654435761u) ^ i);
      logits[j] = static_cast<float>(mixed % 1000) / 1000.0f;
    }
    const auto id = sampler.sample(logits);
    rolling = (rolling ^ id ^ i) * 1099511628211ull;
    const char ch = static_cast<char>('a' + (id % 26));
    TokenEvent event{id, std::string{" "} + ch, i};
    if (callback && !callback(event)) break;
  }
}

const KVCache& InferenceSession::kv_cache() const noexcept { return kv_cache_; }

} // namespace cpullm
