#pragma once

#include "cpullm/cpullm.hpp"

#include <cstdint>
#include <string>

namespace cpullm {

enum class AccelerationProfile : std::uint8_t { off, balanced, turbo };

struct ResidencyConfig {
  AccelerationProfile profile = AccelerationProfile::off;
  bool prefetch = false;
  bool touch_pages = false;
};

struct ResidencyReport {
  std::uint64_t bytes_considered = 0;
  bool prefetched = false;
  bool touched = false;
  std::string note;
};

AccelerationProfile parse_acceleration_profile(std::string_view value);
const char* acceleration_profile_name(AccelerationProfile profile) noexcept;
ResidencyConfig residency_config_for_profile(AccelerationProfile profile);
ResidencyReport apply_residency_policy(const Model& model, const ResidencyConfig& config);

} // namespace cpullm
