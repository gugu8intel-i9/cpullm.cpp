#include "cpullm/cpullm.hpp"

#include <sstream>

namespace cpullm {

Engine::Engine(Model model) : model_(std::move(model)) {}

std::string Engine::generate(std::string_view prompt, const GenerationConfig& config) {
  const auto ids = tokenizer_.encode(prompt);
  std::ostringstream out;
  out << prompt;
  out << "\n\n[cpullm foundation runtime: loaded " << model_.metadata().name
      << ", prompt_tokens=" << ids.size()
      << ", max_tokens=" << config.max_tokens
      << ", temperature=" << config.temperature
      << "]";
  return out.str();
}

const Model& Engine::model() const noexcept { return model_; }

} // namespace cpullm
