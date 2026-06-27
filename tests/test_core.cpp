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

  auto q = cpullm::quantize_q4_0(std::span<const float>(a, 6));
  const float dot = cpullm::dot_q4_0_f32(q, std::span<const float>(a, 6));
  assert(dot > 80.0f);

  cpullm::Sampler greedy({.max_tokens = 1, .temperature = 0.0f});
  const float logits[] = {0.1f, 2.0f, 0.3f};
  assert(greedy.sample(logits) == 1);

  cpullm::KVCache cache(2, 4, 1, 8);
  assert(cache.capacity_tokens() == 4);
  assert(cache.bytes() == 2 * 4 * 1 * 8 * 2 * sizeof(float));
  assert(cache.append_slot());

  cpullm::Model model = cpullm::Model::load_manifest("examples/toy-model.yml");
  cpullm::InferenceSession session(model, {.max_tokens = 3, .temperature = 0.0f});
  std::size_t streamed = 0;
  session.generate_stream("hello", [&](const cpullm::TokenEvent&) { ++streamed; return true; });
  assert(streamed == 3);

  cpullm::Graph g;
  assert(g.add_node("matmul") == 0);
  assert(g.size() == 1);

  std::cout << "cpullm core tests passed\n";
  return 0;
}
