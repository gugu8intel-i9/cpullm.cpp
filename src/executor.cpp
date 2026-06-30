#include "cpullm/executor.hpp"
#include "cpullm/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cpullm {
namespace {

void require_linear(LinearView l) {
  if (l.rows == 0 || l.cols == 0 || l.weight.size() != l.rows * l.cols) {
    throw std::invalid_argument("invalid LinearView");
  }
}

static float silu_local(float x) { return x / (1.0f + std::exp(-x)); }

void resize_at_least(std::vector<float>& v, std::size_t n) {
  if (v.size() < n) v.resize(n);
}

} // namespace

void linear_f32(LinearView linear, std::span<const float> x, std::span<float> y) {
  require_linear(linear);
  if (x.size() != linear.cols || y.size() != linear.rows) throw std::invalid_argument("linear_f32 shape mismatch");
  for (std::size_t r = 0; r < linear.rows; ++r) {
    const auto row = linear.weight.subspan(r * linear.cols, linear.cols);
    float acc = 0.0f;
    for (std::size_t c = 0; c < linear.cols; ++c) acc += row[c] * x[c];
    y[r] = acc;
  }
}

std::vector<ExpertSelection> route_topk(LinearView router, std::span<const float> x, std::size_t top_k,
                                        ExecutorWorkspace& ws) {
  require_linear(router);
  if (x.size() != router.cols) throw std::invalid_argument("router input shape mismatch");
  top_k = std::min(top_k, router.rows);
  if (top_k == 0) throw std::invalid_argument("top_k must be non-zero");

  resize_at_least(ws.logits, router.rows);
  linear_f32(router, x, std::span<float>(ws.logits.data(), router.rows));

  ws.indices.resize(router.rows);
  std::iota(ws.indices.begin(), ws.indices.end(), 0);
  std::partial_sort(ws.indices.begin(), ws.indices.begin() + static_cast<std::ptrdiff_t>(top_k), ws.indices.end(),
                    [&](std::size_t a, std::size_t b) { return ws.logits[a] > ws.logits[b]; });

  float max_logit = ws.logits[ws.indices[0]];
  float sum = 0.0f;
  std::vector<ExpertSelection> selected;
  selected.reserve(top_k);
  for (std::size_t i = 0; i < top_k; ++i) {
    const std::size_t idx = ws.indices[i];
    const float w = std::exp(ws.logits[idx] - max_logit);
    selected.push_back({idx, w});
    sum += w;
  }
  for (auto& s : selected) s.weight /= sum;
  return selected;
}

void execute_dense_block(std::span<const float> x, const DenseBlockWeights& w, std::span<float> y,
                         ExecutorWorkspace& ws, std::size_t position) {
  const std::size_t dim = x.size();
  if (dim == 0 || y.size() != dim || w.attn_norm.size() != dim || w.ffn_norm.size() != dim) {
    throw std::invalid_argument("dense block shape mismatch");
  }
  require_linear(w.q_proj); require_linear(w.k_proj); require_linear(w.v_proj); require_linear(w.o_proj);
  require_linear(w.gate_proj); require_linear(w.up_proj); require_linear(w.down_proj);
  if (w.q_proj.rows != dim || w.k_proj.rows != dim || w.v_proj.rows != dim || w.o_proj.rows != dim ||
      w.q_proj.cols != dim || w.k_proj.cols != dim || w.v_proj.cols != dim || w.o_proj.cols != dim) {
    throw std::invalid_argument("dense attention projection shape mismatch");
  }
  const std::size_t hidden = w.gate_proj.rows;
  if (w.up_proj.rows != hidden || w.down_proj.rows != dim || w.down_proj.cols != hidden ||
      w.gate_proj.cols != dim || w.up_proj.cols != dim) {
    throw std::invalid_argument("dense MLP projection shape mismatch");
  }

  resize_at_least(ws.a, dim);
  resize_at_least(ws.b, dim);
  resize_at_least(ws.c, dim);
  resize_at_least(ws.d, dim);
  resize_at_least(ws.hidden, hidden * 3);

  auto normed = std::span<float>(ws.a.data(), dim);
  auto q = std::span<float>(ws.b.data(), dim);
  auto k = std::span<float>(ws.c.data(), dim);
  auto v = std::span<float>(ws.d.data(), dim);
  rms_norm(x, w.attn_norm, normed, w.norm_eps);
  linear_f32(w.q_proj, normed, q);
  linear_f32(w.k_proj, normed, k);
  linear_f32(w.v_proj, normed, v);
  rope_inplace(q, dim, position, w.rope_base);
  rope_inplace(k, dim, position, w.rope_base);

  // Single-token causal attention: q attends to the current key/value. This is the decode-step primitive;
  // full executors pass accumulated KV pages here once graph wiring is complete.
  auto attn = normed;
  causal_attention(q, k, v, attn, 1, dim);
  linear_f32(w.o_proj, attn, q);
  for (std::size_t i = 0; i < dim; ++i) ws.c[i] = x[i] + q[i];

  auto ffn_in = normed;
  rms_norm(std::span<const float>(ws.c.data(), dim), w.ffn_norm, ffn_in, w.norm_eps);
  auto gate = std::span<float>(ws.hidden.data(), hidden);
  auto up = std::span<float>(ws.hidden.data() + hidden, hidden);
  auto act = std::span<float>(ws.hidden.data() + 2 * hidden, hidden);
  linear_f32(w.gate_proj, ffn_in, gate);
  linear_f32(w.up_proj, ffn_in, up);
  for (std::size_t i = 0; i < hidden; ++i) act[i] = silu_local(gate[i]) * up[i];
  linear_f32(w.down_proj, act, q);
  for (std::size_t i = 0; i < dim; ++i) y[i] = ws.c[i] + q[i];
}

void execute_moe_block(std::span<const float> x, const MoeBlockWeights& w, std::span<float> y,
                       ExecutorWorkspace& ws, std::size_t position) {
  const std::size_t dim = x.size();
  if (dim == 0 || y.size() != dim || w.experts.empty()) throw std::invalid_argument("MoE block shape mismatch");

  DenseBlockWeights dense_prefix;
  dense_prefix.attn_norm = w.attn_norm;
  dense_prefix.q_proj = w.q_proj;
  dense_prefix.k_proj = w.k_proj;
  dense_prefix.v_proj = w.v_proj;
  dense_prefix.o_proj = w.o_proj;
  dense_prefix.ffn_norm = w.ffn_norm;
  // Attention part is executed manually below so MoE can replace the FFN.
  require_linear(w.q_proj); require_linear(w.k_proj); require_linear(w.v_proj); require_linear(w.o_proj); require_linear(w.router);
  if (w.router.cols != dim || w.router.rows != w.experts.size()) throw std::invalid_argument("MoE router shape mismatch");

  resize_at_least(ws.a, dim);
  resize_at_least(ws.b, dim);
  resize_at_least(ws.c, dim);
  resize_at_least(ws.d, dim);

  auto normed = std::span<float>(ws.a.data(), dim);
  auto q = std::span<float>(ws.b.data(), dim);
  auto k = std::span<float>(ws.c.data(), dim);
  auto v = std::span<float>(ws.d.data(), dim);
  rms_norm(x, w.attn_norm, normed, w.norm_eps);
  linear_f32(w.q_proj, normed, q);
  linear_f32(w.k_proj, normed, k);
  linear_f32(w.v_proj, normed, v);
  rope_inplace(q, dim, position, w.rope_base);
  rope_inplace(k, dim, position, w.rope_base);
  causal_attention(q, k, v, normed, 1, dim);
  linear_f32(w.o_proj, normed, q);
  for (std::size_t i = 0; i < dim; ++i) ws.c[i] = x[i] + q[i];

  rms_norm(std::span<const float>(ws.c.data(), dim), w.ffn_norm, normed, w.norm_eps);
  const auto selected = route_topk(w.router, normed, w.top_k, ws);
  std::fill(q.begin(), q.end(), 0.0f);
  for (const auto& sel : selected) {
    const auto& e = w.experts[sel.index];
    const std::size_t hidden = e.gate_proj.rows;
    resize_at_least(ws.hidden, hidden * 3);
    auto gate = std::span<float>(ws.hidden.data(), hidden);
    auto up = std::span<float>(ws.hidden.data() + hidden, hidden);
    auto act = std::span<float>(ws.hidden.data() + 2 * hidden, hidden);
    linear_f32(e.gate_proj, normed, gate);
    linear_f32(e.up_proj, normed, up);
    for (std::size_t i = 0; i < hidden; ++i) act[i] = silu_local(gate[i]) * up[i];
    linear_f32(e.down_proj, act, v);
    for (std::size_t i = 0; i < dim; ++i) q[i] += sel.weight * v[i];
  }
  for (std::size_t i = 0; i < dim; ++i) y[i] = ws.c[i] + q[i];
}

} // namespace cpullm
