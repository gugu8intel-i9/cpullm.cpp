#include "cpullm/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cpullm {

void rms_norm(std::span<const float> x, std::span<const float> weight, std::span<float> y, float eps) {
  if (x.size() != weight.size() || x.size() != y.size()) throw std::invalid_argument("rms_norm shape mismatch");
  float ss = 0.0f;
  for (float v : x) ss += v * v;
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * weight[i];
}

void rope_inplace(std::span<float> v, std::size_t head_dim, std::size_t position, float theta_base) {
  if (head_dim == 0 || v.size() % head_dim != 0 || head_dim % 2 != 0) throw std::invalid_argument("rope shape mismatch");
  const std::size_t heads = v.size() / head_dim;
  for (std::size_t h = 0; h < heads; ++h) {
    auto head = v.subspan(h * head_dim, head_dim);
    for (std::size_t i = 0; i < head_dim; i += 2) {
      const float freq = std::pow(theta_base, -static_cast<float>(i) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float x0 = head[i];
      const float x1 = head[i + 1];
      head[i] = x0 * c - x1 * s;
      head[i + 1] = x0 * s + x1 * c;
    }
  }
}

void softmax_inplace(std::span<float> values) {
  if (values.empty()) return;
  const float m = *std::max_element(values.begin(), values.end());
  float sum = 0.0f;
  for (float& v : values) { v = std::exp(v - m); sum += v; }
  if (sum == 0.0f) throw std::runtime_error("softmax underflow");
  for (float& v : values) v /= sum;
}

void causal_attention(std::span<const float> q, std::span<const float> keys, std::span<const float> values,
                      std::span<float> out, std::size_t past_tokens, std::size_t head_dim) {
  if (q.size() != head_dim || out.size() != head_dim) throw std::invalid_argument("attention q/out shape mismatch");
  if (keys.size() < past_tokens * head_dim || values.size() < past_tokens * head_dim) throw std::invalid_argument("attention kv shape mismatch");
  std::fill(out.begin(), out.end(), 0.0f);
  if (past_tokens == 0) return;
  std::vector<float> scores(past_tokens);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (std::size_t t = 0; t < past_tokens; ++t) {
    const auto k = keys.subspan(t * head_dim, head_dim);
    float dot = 0.0f;
    for (std::size_t i = 0; i < head_dim; ++i) dot += q[i] * k[i];
    scores[t] = dot * scale;
  }
  softmax_inplace(scores);
  for (std::size_t t = 0; t < past_tokens; ++t) {
    const auto val = values.subspan(t * head_dim, head_dim);
    for (std::size_t i = 0; i < head_dim; ++i) out[i] += scores[t] * val[i];
  }
}

void short_convolution_1d(std::span<const float> current, std::span<const float> history,
                          std::span<const float> kernel, std::span<float> out,
                          std::size_t channels, std::size_t kernel_width) {
  if (current.size() != channels || out.size() != channels) throw std::invalid_argument("shortconv current/out shape mismatch");
  if (kernel.size() != channels * kernel_width) throw std::invalid_argument("shortconv kernel shape mismatch");
  if (history.size() < channels * (kernel_width > 0 ? kernel_width - 1 : 0)) throw std::invalid_argument("shortconv history shape mismatch");
  for (std::size_t c = 0; c < channels; ++c) {
    float acc = current[c] * kernel[c * kernel_width + 0];
    for (std::size_t k = 1; k < kernel_width; ++k) acc += history[(k - 1) * channels + c] * kernel[c * kernel_width + k];
    out[c] = acc;
  }
}

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static void matvec_rowmajor(std::span<const float> w, std::span<const float> x, std::span<float> y, std::size_t rows, std::size_t cols) {
  if (w.size() != rows * cols || x.size() != cols || y.size() != rows) throw std::invalid_argument("matvec_rowmajor shape mismatch");
  for (std::size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const auto row = w.subspan(r * cols, cols);
    for (std::size_t c = 0; c < cols; ++c) acc += row[c] * x[c];
    y[r] = acc;
  }
}

void swiglu_mlp(std::span<const float> x, std::span<const float> gate_w, std::span<const float> up_w,
                std::span<const float> down_w, std::span<float> hidden, std::span<float> out,
                std::size_t dim, std::size_t hidden_dim) {
  if (x.size() != dim || hidden.size() != hidden_dim || out.size() != dim) throw std::invalid_argument("mlp shape mismatch");
  std::vector<float> gate(hidden_dim), up(hidden_dim);
  matvec_rowmajor(gate_w, x, gate, hidden_dim, dim);
  matvec_rowmajor(up_w, x, up, hidden_dim, dim);
  for (std::size_t i = 0; i < hidden_dim; ++i) hidden[i] = silu(gate[i]) * up[i];
  matvec_rowmajor(down_w, hidden, out, dim, hidden_dim);
}

void logits_projection(std::span<const float> x, std::span<const float> token_embedding_or_lm_head,
                       std::span<float> logits, std::size_t vocab, std::size_t dim) {
  matvec_rowmajor(token_embedding_or_lm_head, x, logits, vocab, dim);
}

DecodeResult greedy_decode_real(const Model& model, std::string_view, const GenerationConfig&) {
  if (model.metadata().format != ModelFormat::gguf_probe) throw std::runtime_error("real decode requires a GGUF model");
  if (model.metadata().architecture != "cpullm_tiny") {
    throw std::runtime_error("real decode loop is implemented only for supported real executor architectures; this model architecture is '" + model.metadata().architecture + "'");
  }
  throw std::runtime_error("cpullm_tiny executor tensor set not found; refusing to mock real decode");
}

} // namespace cpullm
