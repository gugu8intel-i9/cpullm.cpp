#pragma once

#include "cpullm/cpullm.hpp"

#include <span>
#include <string>
#include <vector>

namespace cpullm {

enum class PlanStatus : std::uint8_t { runnable, blocked };

struct PlanIssue {
  std::string code;
  std::string message;
};

struct KernelCoverage {
  bool f32 = false;
  bool f16 = false;
  bool q4_0 = false;
  bool q4_1 = false;
  bool q8_0 = false;
  bool unknown_types = false;
};

struct ArchitectureProfile {
  std::string name;
  std::string family;
  bool transformer = true;
  bool hybrid = false;
  bool moe = false;
  bool uses_rope = true;
  bool uses_rms_norm = true;
  std::vector<std::string> required_ops;
  std::vector<std::string> notes;
};

struct ExecutionPlan {
  PlanStatus status = PlanStatus::blocked;
  std::string architecture;
  std::string family;
  bool recognized_architecture = false;
  bool transformer = false;
  bool hybrid = false;
  bool moe = false;
  std::size_t layers = 0;
  std::size_t tensors = 0;
  std::size_t tokenizer_tokens = 0;
  KernelCoverage kernels;
  std::vector<std::string> required_ops;
  std::vector<PlanIssue> issues;

  bool runnable() const noexcept { return status == PlanStatus::runnable; }
  std::string to_text() const;
  std::string to_json() const;
};

std::span<const ArchitectureProfile> architecture_profiles() noexcept;
const ArchitectureProfile* find_architecture_profile(std::string_view architecture) noexcept;
ExecutionPlan build_execution_plan(const Model& model);

} // namespace cpullm
