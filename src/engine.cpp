#include "cpullm/cpullm.hpp"

#include <sstream>

namespace cpullm {

Engine::Engine(Model model) : model_(std::move(model)) {}

std::string Engine::generate(std::string_view prompt, const GenerationConfig& config) {
  InferenceSession session(model_, config);
  std::ostringstream out;
  out << session.generate(prompt);
  out << "\n\n[cpullm inference engine: loaded " << model_.metadata().name
      << ", format=" << (model_.metadata().format == ModelFormat::gguf_probe ? "gguf" : "manifest")
      << ", max_tokens=" << config.max_tokens
      << ", temperature=" << config.temperature
      << ", top_k=" << config.top_k
      << ", top_p=" << config.top_p
      << ", spec=" << (config.speculative.mode == SpeculativeMode::mtp ? "mtp" : (config.speculative.mode == SpeculativeMode::draft_model ? "draft" : "off"))
      << ", draft_n_max=" << config.speculative.draft_n_max
      << ", kv_bytes=" << session.kv_cache().bytes()
      << "]";
  return out.str();
}

void Engine::generate_stream(std::string_view prompt, const GenerationConfig& config, const TokenCallback& callback) {
  InferenceSession session(model_, config);
  session.generate_stream(prompt, callback);
}

const Model& Engine::model() const noexcept { return model_; }

} // namespace cpullm
