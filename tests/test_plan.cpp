#include "cpullm/plan.hpp"

#include <cassert>
#include <iostream>

int main() {
  auto model = cpullm::Model::load_manifest("examples/toy-model.yml");
  auto plan = cpullm::build_execution_plan(model);
  assert(!plan.runnable());
  assert(!plan.issues.empty());
  assert(plan.to_text().find("execution_plan_status=blocked") != std::string::npos);
  assert(plan.to_json().find("\"status\": \"blocked\"") != std::string::npos);
  std::cout << "cpullm plan tests passed\n";
}
