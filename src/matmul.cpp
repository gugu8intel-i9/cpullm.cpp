#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <stdexcept>

namespace cpullm {

void matmul_f32(const float* a, const float* b, float* c, MatmulPlan plan) {
  if (!a || !b || !c || plan.m == 0 || plan.n == 0 || plan.k == 0) {
    throw std::invalid_argument("invalid matmul_f32 arguments");
  }

  constexpr std::size_t block_m = 4;
  constexpr std::size_t block_n = 8;

  std::fill(c, c + plan.m * plan.n, 0.0f);
  for (std::size_t ii = 0; ii < plan.m; ii += block_m) {
    for (std::size_t jj = 0; jj < plan.n; jj += block_n) {
      const std::size_t imax = std::min(ii + block_m, plan.m);
      const std::size_t jmax = std::min(jj + block_n, plan.n);
      for (std::size_t kk = 0; kk < plan.k; ++kk) {
        for (std::size_t i = ii; i < imax; ++i) {
          const float av = a[i * plan.k + kk];
          for (std::size_t j = jj; j < jmax; ++j) {
            const float bv = plan.transpose_b ? b[j * plan.k + kk] : b[kk * plan.n + j];
            c[i * plan.n + j] += av * bv;
          }
        }
      }
    }
  }
}

std::size_t TensorView::elements() const {
  std::size_t total = 1;
  for (const auto dim : shape) total *= dim;
  return shape.empty() ? 0 : total;
}

} // namespace cpullm
