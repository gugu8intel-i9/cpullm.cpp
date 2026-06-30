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

namespace cpullm {

static std::uint16_t read_u16(const std::byte* p) {
  std::uint16_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

static float read_f32(const std::byte* p) {
  float v = 0.0f;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

void matvec_gguf_f16_f32(std::span<const std::byte> rows, std::span<const float> x,
                         std::span<float> y, std::size_t cols) {
  if (cols == 0 || x.size() != cols) throw std::invalid_argument("GGUF f16 matvec shape mismatch");
  const std::size_t row_bytes = cols * 2;
  if (rows.size() < y.size() * row_bytes) throw std::invalid_argument("GGUF f16 buffer too small");
  for (std::size_t r = 0; r < y.size(); ++r) {
    const auto row = rows.subspan(r * row_bytes, row_bytes);
    float acc = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) acc += fp16_to_f32(read_u16(row.data() + c * 2)) * x[c];
    y[r] = acc;
  }
}

void matvec_gguf_q4_1_f32(std::span<const std::byte> rows, std::span<const float> x,
                          std::span<float> y, std::size_t cols) {
  constexpr std::size_t block_bytes = 20;
  if (cols == 0 || x.size() != cols) throw std::invalid_argument("GGUF q4_1 matvec shape mismatch");
  const std::size_t blocks_per_row = (cols + 31) / 32;
  const std::size_t row_bytes = blocks_per_row * block_bytes;
  if (rows.size() < y.size() * row_bytes) throw std::invalid_argument("GGUF q4_1 buffer too small");
  for (std::size_t r = 0; r < y.size(); ++r) {
    const auto row = rows.subspan(r * row_bytes, row_bytes);
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
      const auto* ptr = row.data() + b * block_bytes;
      const float d = fp16_to_f32(read_u16(ptr));
      const float m = fp16_to_f32(read_u16(ptr + 2));
      const auto* qs = reinterpret_cast<const std::uint8_t*>(ptr + 4);
      for (std::size_t i = 0; i < 32; ++i) {
        const std::size_t idx = b * 32 + i;
        if (idx >= cols) break;
        const std::uint8_t byte = qs[i / 2];
        const int q = static_cast<int>((i & 1) == 0 ? byte & 0x0F : byte >> 4);
        sum += (static_cast<float>(q) * d + m) * x[idx];
      }
    }
    y[r] = sum;
  }
}

void matvec_gguf_q8_0_f32(std::span<const std::byte> rows, std::span<const float> x,
                          std::span<float> y, std::size_t cols) {
  constexpr std::size_t block_bytes = 34;
  if (cols == 0 || x.size() != cols) throw std::invalid_argument("GGUF q8_0 matvec shape mismatch");
  const std::size_t blocks_per_row = (cols + 31) / 32;
  const std::size_t row_bytes = blocks_per_row * block_bytes;
  if (rows.size() < y.size() * row_bytes) throw std::invalid_argument("GGUF q8_0 buffer too small");
  for (std::size_t r = 0; r < y.size(); ++r) {
    const auto row = rows.subspan(r * row_bytes, row_bytes);
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
      const auto* ptr = row.data() + b * block_bytes;
      const float d = fp16_to_f32(read_u16(ptr));
      const auto* qs = reinterpret_cast<const std::int8_t*>(ptr + 2);
      for (std::size_t i = 0; i < 32; ++i) {
        const std::size_t idx = b * 32 + i;
        if (idx >= cols) break;
        sum += static_cast<float>(qs[i]) * d * x[idx];
      }
    }
    y[r] = sum;
  }
}

bool matvec_gguf_any_f32(DataType dtype, std::span<const std::byte> rows, std::span<const float> x,
                         std::span<float> y, std::size_t cols) {
  switch (dtype) {
    case DataType::f32: {
      if (rows.size() < y.size() * cols * sizeof(float)) throw std::invalid_argument("GGUF f32 buffer too small");
      for (std::size_t r = 0; r < y.size(); ++r) {
        float acc = 0.0f;
        for (std::size_t c = 0; c < cols; ++c) acc += read_f32(rows.data() + (r * cols + c) * 4) * x[c];
        y[r] = acc;
      }
      return true;
    }
    case DataType::f16: matvec_gguf_f16_f32(rows, x, y, cols); return true;
    case DataType::q4_0: matvec_gguf_q4_0_f32(rows, x, y, cols); return true;
    case DataType::q4_1: matvec_gguf_q4_1_f32(rows, x, y, cols); return true;
    case DataType::q8_0: matvec_gguf_q8_0_f32(rows, x, y, cols); return true;
    default: return false;
  }
}

const char* dtype_name(DataType dtype) noexcept {
  switch (dtype) {
    case DataType::f32: return "F32"; case DataType::f16: return "F16"; case DataType::q4_0: return "Q4_0";
    case DataType::q4_1: return "Q4_1"; case DataType::q5_0: return "Q5_0"; case DataType::q5_1: return "Q5_1";
    case DataType::q8_0: return "Q8_0"; case DataType::q2_k: return "Q2_K"; case DataType::q3_k: return "Q3_K";
    case DataType::q4_k: return "Q4_K"; case DataType::q5_k: return "Q5_K"; case DataType::q6_k: return "Q6_K";
    case DataType::q8_k: return "Q8_K"; case DataType::iq2_xxs: return "IQ2_XXS"; case DataType::iq2_xs: return "IQ2_XS";
    case DataType::iq3_xxs: return "IQ3_XXS"; case DataType::iq1_s: return "IQ1_S"; case DataType::iq4_nl: return "IQ4_NL";
    case DataType::iq3_s: return "IQ3_S"; case DataType::iq2_s: return "IQ2_S"; case DataType::iq4_xs: return "IQ4_XS";
    case DataType::iq1_m: return "IQ1_M"; case DataType::bf16: return "BF16"; case DataType::tq1_0: return "TQ1_0";
    case DataType::tq2_0: return "TQ2_0"; case DataType::unknown: return "UNKNOWN";
  }
  return "UNKNOWN";
}

bool dtype_has_low_ram_matvec(DataType dtype) noexcept {
  return dtype == DataType::f32 || dtype == DataType::f16 || dtype == DataType::q4_0 || dtype == DataType::q4_1 || dtype == DataType::q8_0;
}

} // namespace cpullm
