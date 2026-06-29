#include "cpullm/residency.hpp"

#include <cassert>
#include <iostream>

int main() {
  assert(cpullm::parse_acceleration_profile("off") == cpullm::AccelerationProfile::off);
  assert(cpullm::parse_acceleration_profile("balanced") == cpullm::AccelerationProfile::balanced);
  assert(cpullm::parse_acceleration_profile("turbo") == cpullm::AccelerationProfile::turbo);
  auto cfg = cpullm::residency_config_for_profile(cpullm::AccelerationProfile::turbo);
  assert(cfg.prefetch);
  assert(cfg.touch_pages);
  auto model = cpullm::Model::load_manifest("examples/toy-model.yml");
  auto report = cpullm::apply_residency_policy(model, cfg);
  assert(report.bytes_considered == 0);
  assert(!report.note.empty());
  std::cout << "cpullm residency tests passed\n";
}
