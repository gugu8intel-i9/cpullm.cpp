#include "cpullm/llama_compat.h"
#include "cpullm/cpullm.hpp"

#include <memory>
#include <string>

struct llama_model {
  cpullm::Model model;
  explicit llama_model(cpullm::Model m) : model(std::move(m)) {}
};

struct llama_context {
  llama_model* owner = nullptr;
  llama_context_params params{};
};

extern "C" llama_context_params llama_context_default_params(void) {
  llama_context_params params{};
  params.n_ctx = 2048;
  params.n_threads = 0;
  return params;
}

extern "C" llama_model* llama_load_model_from_file(const char* path, llama_context_params) {
  if (!path) return nullptr;
  try {
    return new llama_model(cpullm::Model::load_manifest(path));
  } catch (...) {
    return nullptr;
  }
}

extern "C" llama_context* llama_new_context_with_model(llama_model* model, llama_context_params params) {
  if (!model) return nullptr;
  try {
    auto* ctx = new llama_context{};
    ctx->owner = model;
    ctx->params = params;
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

extern "C" void llama_free(llama_context* ctx) { delete ctx; }
extern "C" void llama_free_model(llama_model* model) { delete model; }

extern "C" const char* llama_print_system_info(void) {
  static std::string info;
  info = cpullm::detect_cpu_features().summary();
  return info.c_str();
}
