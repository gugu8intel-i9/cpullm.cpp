# cpullm.cpp

A high-performance, lightweight, CPU-first LLM runtime foundation written in modern C++.

> Status: early foundation. The project currently provides the runtime skeleton, warning-clean CPU feature detection, arena allocation, a portable blocked F32 matmul baseline, tokenizer scaffolding, model manifest loading, CLI, example, tests, and architecture notes.

## Goals

- **Lightweight:** small C++20 core, no required third-party dependencies.
- **Fast by design:** scratch arenas, cache-aware kernels, runtime CPU feature dispatch, and a path toward fused graph execution.
- **Innovative:** format adapters, paged KV cache planning, kernel registry, and quantization experiments without inheriting llama.cpp internals.
- **Practical:** simple CLI and stable library API as the foundation grows.
- **Apache-2.0:** permissive licensing for research and production use.

## Repository layout

```text
apps/              cpullm-cli entry point
include/cpullm/    public C++ API
src/               runtime implementation
examples/          minimal embedding and toy manifest
tests/             dependency-free core tests
docs/              architecture and roadmap notes
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Enable native CPU tuning for local benchmarking:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPULLM_NATIVE=ON
cmake --build build -j
```

## Try it

```bash
./build/cpullm-minimal
./build/cpullm-cli examples/toy-model.yml "Hello from cpullm"
```

## Current components

- `cpullm::Arena` — cache-line aligned scratch allocator for hot-path temporary memory.
- `cpullm::detect_cpu_features()` — portable runtime feature discovery for x86 and ARM targets.
- `cpullm::matmul_f32()` — blocked scalar baseline that future SIMD microkernels can replace through dispatch.
- `cpullm::Tokenizer` — minimal whitespace tokenizer placeholder for API development.
- `cpullm::Model::load_manifest()` — tiny manifest loader for early CLI/runtime wiring.
- `cpullm::Engine` — generation facade ready for sampler, graph, and KV-cache integration.

## Roadmap

1. Add quantized tensor block types and dequantizing matmul kernels.
2. Introduce a runtime kernel registry for AVX2, AVX-512, NEON, and portable fallbacks.
3. Implement paged KV cache and streaming token generation API.
4. Add model format adapters, starting with a clean-room GGUF reader.
5. Add reproducible benchmarks and profiling scripts.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
