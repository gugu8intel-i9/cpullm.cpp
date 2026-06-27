#include "cpullm/cpullm.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  cpullm::Arena arena(4096);
  auto* ptr = arena.allocate(128);
  assert(ptr != nullptr);
  assert(arena.used() >= 128);
  arena.reset();
  assert(arena.used() == 0);

  const float a[] = {1, 2, 3, 4, 5, 6};       // 2x3
  const float b_t[] = {7, 8, 9, 10, 11, 12};  // transposed 2x3 for B=3x2
  float c[4] = {};
  cpullm::matmul_f32(a, b_t, c, {.m = 2, .n = 2, .k = 3, .transpose_b = true});
  assert(std::fabs(c[0] - 50.0f) < 1e-5f);
  assert(std::fabs(c[1] - 68.0f) < 1e-5f);
  assert(std::fabs(c[2] - 122.0f) < 1e-5f);
  assert(std::fabs(c[3] - 167.0f) < 1e-5f);

  cpullm::Tokenizer tok({"<unk>", "hello", "world"});
  auto ids = tok.encode("hello fast world");
  assert(ids.size() == 3);
  assert(tok.decode(ids) == "hello <unk> world");

  cpullm::Graph g;
  assert(g.add_node("matmul") == 0);
  assert(g.size() == 1);

  std::cout << "cpullm core tests passed\n";
  return 0;
}
