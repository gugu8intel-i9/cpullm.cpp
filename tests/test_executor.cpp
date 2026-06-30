#include "cpullm/executor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static cpullm::LinearView lv(const std::vector<float>& w, std::size_t r, std::size_t c) {
  return {std::span<const float>(w.data(), w.size()), r, c};
}

int main() {
  const std::size_t dim = 4;
  const std::size_t hidden = 8;
  std::vector<float> x = {0.2f, -0.1f, 0.4f, 0.7f};
  std::vector<float> y(dim);
  std::vector<float> norm(dim, 1.0f);
  std::vector<float> ident(dim * dim, 0.0f);
  for (std::size_t i = 0; i < dim; ++i) ident[i * dim + i] = 1.0f;
  std::vector<float> gate(hidden * dim), up(hidden * dim), down(dim * hidden);
  for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = 0.01f * static_cast<float>((i % 7) + 1);
  for (std::size_t i = 0; i < up.size(); ++i) up[i] = 0.02f * static_cast<float>((i % 5) + 1);
  for (std::size_t i = 0; i < down.size(); ++i) down[i] = 0.015f * static_cast<float>((i % 3) + 1);

  cpullm::DenseBlockWeights dense{
      .attn_norm = norm,
      .q_proj = lv(ident, dim, dim),
      .k_proj = lv(ident, dim, dim),
      .v_proj = lv(ident, dim, dim),
      .o_proj = lv(ident, dim, dim),
      .ffn_norm = norm,
      .gate_proj = lv(gate, hidden, dim),
      .up_proj = lv(up, hidden, dim),
      .down_proj = lv(down, dim, hidden),
  };
  cpullm::ExecutorWorkspace ws;
  cpullm::execute_dense_block(x, dense, y, ws, 0);
  for (float v : y) assert(std::isfinite(v));

  std::vector<float> router = {1, 0, 0, 0, 0, 1, 0, 0}; // 2 experts x dim
  cpullm::MoeExpertView experts_arr[2] = {
      {lv(gate, hidden, dim), lv(up, hidden, dim), lv(down, dim, hidden)},
      {lv(gate, hidden, dim), lv(up, hidden, dim), lv(down, dim, hidden)},
  };
  cpullm::MoeBlockWeights moe{
      .attn_norm = norm,
      .q_proj = lv(ident, dim, dim),
      .k_proj = lv(ident, dim, dim),
      .v_proj = lv(ident, dim, dim),
      .o_proj = lv(ident, dim, dim),
      .ffn_norm = norm,
      .router = lv(router, 2, dim),
      .experts = std::span<const cpullm::MoeExpertView>(experts_arr, 2),
      .top_k = 2,
  };
  cpullm::execute_moe_block(x, moe, y, ws, 0);
  for (float v : y) assert(std::isfinite(v));
  auto selected = cpullm::route_topk(lv(router, 2, dim), x, 1, ws);
  assert(selected.size() == 1);
  assert(selected[0].weight > 0.99f);

  std::cout << "cpullm executor tests passed\n";
}
