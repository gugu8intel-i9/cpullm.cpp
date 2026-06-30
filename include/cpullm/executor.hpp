#pragma once

#include "cpullm/cpullm.hpp"

#include <span>
#include <vector>

namespace cpullm {

struct LinearView {
  std::span<const float> weight;
  std::size_t rows = 0;
  std::size_t cols = 0;
};

struct DenseBlockWeights {
  std::span<const float> attn_norm;
  LinearView q_proj;
  LinearView k_proj;
  LinearView v_proj;
  LinearView o_proj;
  std::span<const float> ffn_norm;
  LinearView gate_proj;
  LinearView up_proj;
  LinearView down_proj;
  float norm_eps = 1e-5f;
  float rope_base = 10000.0f;
};

struct MoeExpertView {
  LinearView gate_proj;
  LinearView up_proj;
  LinearView down_proj;
};

struct MoeBlockWeights {
  std::span<const float> attn_norm;
  LinearView q_proj;
  LinearView k_proj;
  LinearView v_proj;
  LinearView o_proj;
  std::span<const float> ffn_norm;
  LinearView router;
  std::span<const MoeExpertView> experts;
  std::size_t top_k = 2;
  float norm_eps = 1e-5f;
  float rope_base = 10000.0f;
};

struct ExecutorWorkspace {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;
  std::vector<float> d;
  std::vector<float> hidden;
  std::vector<float> logits;
  std::vector<std::size_t> indices;
};

struct ExpertSelection {
  std::size_t index = 0;
  float weight = 0.0f;
};

void linear_f32(LinearView linear, std::span<const float> x, std::span<float> y);
std::vector<ExpertSelection> route_topk(LinearView router, std::span<const float> x, std::size_t top_k,
                                        ExecutorWorkspace& ws);
void execute_dense_block(std::span<const float> x, const DenseBlockWeights& weights, std::span<float> y,
                         ExecutorWorkspace& ws, std::size_t position = 0);
void execute_moe_block(std::span<const float> x, const MoeBlockWeights& weights, std::span<float> y,
                       ExecutorWorkspace& ws, std::size_t position = 0);

} // namespace cpullm
