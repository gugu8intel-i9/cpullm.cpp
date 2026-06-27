#include "cpullm/cpullm.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
  constexpr std::size_t rows = 256;
  constexpr std::size_t cols = 256;
  std::vector<float> w(rows * cols), x(cols), y(rows);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 5.0f;
  auto qw = cpullm::quantize_q4_0(w);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) cpullm::matvec_q4_0_f32(qw, x, y, cols);
  const auto end = std::chrono::steady_clock::now();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "q4_0 matvec rows=" << rows << " cols=" << cols << " iters=1000 us=" << us
            << " checksum=" << y[0] << '\n';
}
