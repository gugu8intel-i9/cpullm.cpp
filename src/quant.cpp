#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cpullm {

std::size_t dtype_size(DataType dtype) {
  switch (dtype) {
    case DataType::f32: return sizeof(float);
    case DataType::q4_0: return sizeof(Q4Block);
  }
  return 0;
}

std::vector<Q4Block> quantize_q4_0(std::span<const float> values) {
  const std::size_t blocks = (values.size() + 31) / 32;
  std::vector<Q4Block> out(blocks);
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t base = b * 32;
    float amax = 0.0f;
    for (std::size_t i = 0; i < 32 && base + i < values.size(); ++i) {
      amax = std::max(amax, std::fabs(values[base + i]));
    }
    const float scale = amax == 0.0f ? 1.0f : amax / 7.0f;
    out[b].scale = scale;
    for (std::size_t i = 0; i < 32; ++i) {
      const float v = base + i < values.size() ? values[base + i] : 0.0f;
      int q = static_cast<int>(std::nearbyint(v / scale));
      q = std::clamp(q, -8, 7);
      const std::uint8_t nibble = static_cast<std::uint8_t>(q + 8);
      if ((i & 1) == 0) out[b].packed[i / 2] = nibble;
      else out[b].packed[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
    }
  }
  return out;
}

float dot_q4_0_f32(std::span<const Q4Block> quantized, std::span<const float> x) {
  if (x.size() > quantized.size() * 32) throw std::invalid_argument("q4 dot shape mismatch");
  float sum = 0.0f;
  for (std::size_t b = 0; b < quantized.size(); ++b) {
    for (std::size_t i = 0; i < 32; ++i) {
      const std::size_t idx = b * 32 + i;
      if (idx >= x.size()) return sum;
      const std::uint8_t byte = quantized[b].packed[i / 2];
      const int q = static_cast<int>(((i & 1) == 0 ? byte & 0x0F : byte >> 4)) - 8;
      sum += (static_cast<float>(q) * quantized[b].scale) * x[idx];
    }
  }
  return sum;
}

void matvec_q4_0_f32(std::span<const Q4Block> rows, std::span<const float> x,
                     std::span<float> y, std::size_t cols) {
  if (cols == 0 || x.size() != cols) throw std::invalid_argument("q4 matvec shape mismatch");
  const std::size_t blocks_per_row = (cols + 31) / 32;
  if (rows.size() < y.size() * blocks_per_row) throw std::invalid_argument("q4 row buffer too small");
  for (std::size_t r = 0; r < y.size(); ++r) {
    y[r] = dot_q4_0_f32(rows.subspan(r * blocks_per_row, blocks_per_row), x);
  }
}

} // namespace cpullm
