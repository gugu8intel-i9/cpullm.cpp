#pragma once

#include "cpullm/cpullm.hpp"

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
  bool q4_0 = false;
  bool q4_1 = false;
  bool q8_0 = false;
  bool unknown_types = false;
};

struct ExecutionPlan {
  PlanStatus status = PlanStatus::blocked;
  std::string architecture;
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

ExecutionPlan build_execution_plan(const Model& model);

} // namespace cpullm
