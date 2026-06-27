#include "cpullm/cpullm.hpp"

#include <iostream>

int main() {
  std::cout << cpullm::detect_cpu_features().summary() << '\n';
  cpullm::Arena arena(1024 * 1024);
  auto* scratch = static_cast<float*>(arena.allocate(16 * sizeof(float)));
  scratch[0] = 1.0f;
  std::cout << "arena_used=" << arena.used() << '\n';
}
