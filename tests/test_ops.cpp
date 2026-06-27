#include "cpullm/ops.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  const float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[4]{};
  cpullm::rms_norm(x, w, y, 1e-6f);
  assert(std::isfinite(y[0]));

  float rope[] = {1.0f, 0.0f, 0.0f, 1.0f};
  cpullm::rope_inplace(rope, 4, 1);
  assert(std::isfinite(rope[0]));

  float probs[] = {1.0f, 2.0f, 3.0f};
  cpullm::softmax_inplace(probs);
  assert(std::fabs((probs[0] + probs[1] + probs[2]) - 1.0f) < 1e-5f);

  const float q[] = {1.0f, 0.0f};
  const float k[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float v[] = {2.0f, 0.0f, 0.0f, 4.0f};
  float attn[2]{};
  cpullm::causal_attention(q, k, v, attn, 2, 2);
  assert(std::isfinite(attn[0]));

  const float cur[] = {1.0f, 2.0f};
  const float hist[] = {0.5f, 1.0f};
  const float ker[] = {1.0f, 0.5f, 1.0f, 0.5f};
  float conv[2]{};
  cpullm::short_convolution_1d(cur, hist, ker, conv, 2, 2);
  assert(std::fabs(conv[0] - 1.25f) < 1e-5f);

  const float proj_w[] = {1,0,0,1, 1,1};
  float logits[3]{};
  cpullm::logits_projection(std::span<const float>(q, 2), std::span<const float>(proj_w, 6), std::span<float>(logits, 3), 3, 2);
  assert(logits[0] == 1.0f);

  std::cout << "cpullm ops tests passed\n";
}
