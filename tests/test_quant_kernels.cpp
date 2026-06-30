#include "cpullm/cpullm.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static std::uint16_t f32_to_f16_bits(float f) {
  // Minimal test helper for small positive finite values.
  std::uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const int exp = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  const std::uint32_t mant = (bits >> 13) & 0x3ffu;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
}

static void put_u16(std::byte* p, std::uint16_t v) { std::memcpy(p, &v, sizeof(v)); }

int main() {
  assert(cpullm::dtype_size(cpullm::DataType::q4_k) == 144);
  assert(cpullm::dtype_has_low_ram_matvec(cpullm::DataType::q8_0));
  assert(!cpullm::dtype_has_low_ram_matvec(cpullm::DataType::q4_k));

  const float x[32] = {1.0f};
  float y[1]{};

  std::vector<std::byte> f16(32 * 2);
  put_u16(f16.data(), f32_to_f16_bits(2.0f));
  cpullm::matvec_gguf_f16_f32(f16, x, y, 32);
  assert(std::fabs(y[0] - 2.0f) < 1e-3f);

  std::vector<std::byte> q8(34);
  put_u16(q8.data(), f32_to_f16_bits(0.5f));
  reinterpret_cast<std::int8_t*>(q8.data() + 2)[0] = 4;
  cpullm::matvec_gguf_q8_0_f32(q8, x, y, 32);
  assert(std::fabs(y[0] - 2.0f) < 1e-3f);

  std::vector<std::byte> q41(20);
  put_u16(q41.data(), f32_to_f16_bits(0.5f));
  put_u16(q41.data() + 2, f32_to_f16_bits(1.0f));
  reinterpret_cast<std::uint8_t*>(q41.data() + 4)[0] = 0x02;
  cpullm::matvec_gguf_q4_1_f32(q41, x, y, 32);
  assert(std::fabs(y[0] - 2.0f) < 1e-3f);

  assert(cpullm::matvec_gguf_any_f32(cpullm::DataType::q8_0, q8, x, y, 32));
  assert(!cpullm::matvec_gguf_any_f32(cpullm::DataType::q4_k, q8, x, y, 32));

  std::cout << "cpullm quant kernel tests passed\n";
}
