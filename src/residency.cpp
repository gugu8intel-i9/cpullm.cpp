#include "cpullm/residency.hpp"

#include <stdexcept>

namespace cpullm {

AccelerationProfile parse_acceleration_profile(std::string_view value) {
  if (value == "off" || value == "none") return AccelerationProfile::off;
  if (value == "balanced") return AccelerationProfile::balanced;
  if (value == "turbo" || value == "max") return AccelerationProfile::turbo;
  throw std::invalid_argument("unknown acceleration profile: " + std::string(value));
}

const char* acceleration_profile_name(AccelerationProfile profile) noexcept {
  switch (profile) {
    case AccelerationProfile::off: return "off";
    case AccelerationProfile::balanced: return "balanced";
    case AccelerationProfile::turbo: return "turbo";
  }
  return "unknown";
}

ResidencyConfig residency_config_for_profile(AccelerationProfile profile) {
  ResidencyConfig cfg;
  cfg.profile = profile;
  switch (profile) {
    case AccelerationProfile::off:
      break;
    case AccelerationProfile::balanced:
      cfg.prefetch = true;
      cfg.touch_pages = false;
      break;
    case AccelerationProfile::turbo:
      cfg.prefetch = true;
      cfg.touch_pages = true;
      break;
  }
  return cfg;
}

ResidencyReport apply_residency_policy(const Model& model, const ResidencyConfig& config) {
  ResidencyReport report;
  report.prefetched = config.prefetch;
  report.touched = config.touch_pages;
  const auto* gguf = model.gguf_file();
  if (!gguf) {
    report.note = "non-GGUF model: residency policy skipped";
    return report;
  }
  if (!config.prefetch && !config.touch_pages) {
    report.note = "residency profile off";
    return report;
  }
  report.bytes_considered = gguf->prefetch_tensors(config.touch_pages);
  report.note = config.touch_pages
      ? "tensor pages requested and touched into OS page cache"
      : "tensor pages requested via OS prefetch advice";
  return report;
}

} // namespace cpullm
