#pragma once

#include "cpullm/cpullm.hpp"

#include <span>
#include <vector>

namespace cpullm {

void rms_norm(std::span<const float> x, std::span<const float> weight, std::span<float> y, float eps);
void rope_inplace(std::span<float> q_or_k, std::size_t head_dim, std::size_t position, float theta_base = 10000.0f);
void softmax_inplace(std::span<float> values);
void causal_attention(std::span<const float> q, std::span<const float> keys, std::span<const float> values,
                      std::span<float> out, std::size_t past_tokens, std::size_t head_dim);
void short_convolution_1d(std::span<const float> current, std::span<const float> history,
                          std::span<const float> kernel, std::span<float> out,
                          std::size_t channels, std::size_t kernel_width);
void swiglu_mlp(std::span<const float> x, std::span<const float> gate_w, std::span<const float> up_w,
                std::span<const float> down_w, std::span<float> hidden, std::span<float> out,
                std::size_t dim, std::size_t hidden_dim);
void logits_projection(std::span<const float> x, std::span<const float> token_embedding_or_lm_head,
                       std::span<float> logits, std::size_t vocab, std::size_t dim);

struct DecodeResult {
  std::vector<std::uint32_t> tokens;
  std::string text;
};

DecodeResult greedy_decode_real(const Model& model, std::string_view prompt, const GenerationConfig& config);

} // namespace cpullm
