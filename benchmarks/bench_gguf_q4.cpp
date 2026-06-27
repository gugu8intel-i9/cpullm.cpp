#include "cpullm/cpullm.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: cpullm-gguf-q4-bench <model.gguf> [tensor-name]\n";
    return 2;
  }
  auto gguf = cpullm::GgufFile::open(argv[1]);
  const cpullm::GgufTensorInfo* tensor = nullptr;
  if (argc >= 3) tensor = gguf.find_tensor(argv[2]);
  if (!tensor) {
    for (const auto& t : gguf.tensors()) {
      if (t.dtype == cpullm::DataType::q4_0 && t.shape.size() >= 2 && t.shape[0] % 32 == 0) {
        tensor = &t;
        break;
      }
    }
  }
  if (!tensor) {
    std::cerr << "no q4_0 matrix tensor found\n";
    return 1;
  }
  const std::size_t cols = static_cast<std::size_t>(tensor->shape[0]);
  const std::size_t rows = static_cast<std::size_t>(tensor->shape[1]);
  auto bytes = gguf.tensor_bytes(*tensor);

  std::vector<float> x(cols), y(rows);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;

  const auto start = std::chrono::steady_clock::now();
  constexpr int iters = 200;
  for (int i = 0; i < iters; ++i) {
    cpullm::matvec_gguf_q4_0_f32(bytes, x, y, cols);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "model=" << gguf.summary() << '\n'
            << "tensor=" << tensor->name << " rows=" << rows << " cols=" << cols
            << " bytes=" << tensor->bytes << " iters=" << iters << " us=" << us
            << " matvec_per_sec=" << (iters * 1000000.0 / static_cast<double>(us))
            << " checksum=" << (y.empty() ? 0.0f : y[0]) << '\n';
  return 0;
}
