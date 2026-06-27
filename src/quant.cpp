#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cpullm {

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

namespace cpullm {

float fp16_to_f32(std::uint16_t h) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u)) << 16;
  std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x03ffu;
  std::uint32_t f = 0;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x03ffu;
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

float dot_gguf_q4_0_f32(std::span<const std::byte> row, std::span<const float> x) {
  constexpr std::size_t block_bytes = 18;
  const std::size_t blocks = (x.size() + 31) / 32;
  if (row.size() < blocks * block_bytes) throw std::invalid_argument("GGUF q4_0 row too small");
  float sum = 0.0f;
  for (std::size_t b = 0; b < blocks; ++b) {
    const auto* ptr = row.data() + b * block_bytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, ptr, sizeof(scale_bits));
    const float d = fp16_to_f32(scale_bits);
    const auto* qs = reinterpret_cast<const std::uint8_t*>(ptr + 2);
    for (std::size_t i = 0; i < 32; ++i) {
      const std::size_t idx = b * 32 + i;
      if (idx >= x.size()) return sum;
      const std::uint8_t byte = qs[i / 2];
      const int q = static_cast<int>(((i & 1) == 0 ? byte & 0x0F : byte >> 4)) - 8;
      sum += static_cast<float>(q) * d * x[idx];
    }
  }
  return sum;
}

void matvec_gguf_q4_0_f32(std::span<const std::byte> rows, std::span<const float> x,
                          std::span<float> y, std::size_t cols) {
  constexpr std::size_t block_bytes = 18;
  if (cols == 0 || x.size() != cols) throw std::invalid_argument("GGUF q4_0 matvec shape mismatch");
  const std::size_t blocks_per_row = (cols + 31) / 32;
  const std::size_t row_bytes = blocks_per_row * block_bytes;
  if (rows.size() < y.size() * row_bytes) throw std::invalid_argument("GGUF q4_0 buffer too small");
  for (std::size_t r = 0; r < y.size(); ++r) {
    y[r] = dot_gguf_q4_0_f32(rows.subspan(r * row_bytes, row_bytes), x);
  }
}

} // namespace cpullm
