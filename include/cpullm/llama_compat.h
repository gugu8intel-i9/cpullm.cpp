#pragma once

/*
 * Minimal llama.cpp-style C ABI scaffold.
 *
 * This header is intentionally tiny: it gives embedders stable names to start
 * integrating against while cpullm.cpp grows toward fuller llama.cpp source and
 * binary compatibility. Functions are prefixed with llama_ for drop-in migration
 * experiments, but implemented clean-room by cpullm.cpp.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int llama_token;

struct llama_context_params {
  int n_ctx;
  int n_threads;
};

struct llama_model;
struct llama_context;

struct llama_context_params llama_context_default_params(void);
struct llama_model* llama_load_model_from_file(const char* path, struct llama_context_params params);
struct llama_context* llama_new_context_with_model(struct llama_model* model, struct llama_context_params params);
void llama_free(struct llama_context* ctx);
void llama_free_model(struct llama_model* model);
const char* llama_print_system_info(void);

#ifdef __cplusplus
}
#endif
