# cpullm.cpp

A high-performance, lightweight, CPU-first LLM runtime foundation designed to become a drop-in alternative to llama.cpp.

> Status: early foundation. The project currently provides llama.cpp-style CLI entry points, common llama.cpp flag parsing, a minimal llama-compatible C ABI scaffold, warning-clean CPU feature detection, arena allocation, a portable blocked F32 matmul baseline, tokenizer scaffolding, model manifest loading, CLI, example, tests, and architecture notes.

## Goals

- **Drop-in llama.cpp ergonomics:** build a `llama-cli` target and accept common flags like `-m`, `-p`, `-n`, `--temp`, `-c`, and `-t`.
- **Lightweight:** small C++20 core, no required third-party dependencies.
- **Fast by design:** scratch arenas, cache-aware kernels, runtime CPU feature dispatch, and a path toward fused graph execution.
- **Innovative:** format adapters, paged KV cache planning, kernel registry, and quantization experiments without inheriting llama.cpp internals.
- **Practical:** simple CLI and stable library/API shims as the foundation grows.
- **Apache-2.0:** permissive licensing for research and production use.

## Repository layout

```text
apps/              cpullm-cli / llama-cli entry point
include/cpullm/    public C++ API and llama-compatible C ABI scaffold
src/               runtime implementation
examples/          minimal embedding and toy manifest
tests/             dependency-free core tests
docs/              architecture, compatibility plan, and roadmap notes
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

Native name:

```bash
./build/cpullm-minimal
./build/cpullm-cli -m examples/toy-model.yml -p "Hello from cpullm" -n 32 --temp 0.8
```

llama.cpp-style name:

```bash
./build/llama-cli -m examples/toy-model.yml -p "Hello from cpullm" -n 32 --temp 0.8 -c 2048 -t 0
```

## Current components

- `cpullm::Arena` — cache-line aligned scratch allocator for hot-path temporary memory.
- `cpullm::detect_cpu_features()` — portable runtime feature discovery for x86 and ARM targets.
- `cpullm::matmul_f32()` — blocked scalar baseline that future SIMD microkernels can replace through dispatch.
- `cpullm::Tokenizer` — minimal whitespace tokenizer placeholder for API development.
- `cpullm::Model::load_manifest()` — tiny manifest loader for early CLI/runtime wiring.
- `cpullm::Engine` — generation facade ready for sampler, graph, and KV-cache integration.
- `include/cpullm/llama_compat.h` — minimal llama.cpp-style C ABI scaffold for migration experiments.

## llama.cpp compatibility

See [docs/llama_compatibility.md](docs/llama_compatibility.md).

Today, cpullm.cpp accepts common llama.cpp-style CLI invocations for YAML manifests. Full GGUF loading and real token generation are planned next.

## Roadmap

1. Add a clean-room GGUF reader and tensor metadata loader.
2. Add quantized tensor block types and dequantizing matmul kernels.
3. Introduce a runtime kernel registry for AVX2, AVX-512, NEON, and portable fallbacks.
4. Implement paged KV cache and streaming token generation API.
5. Broaden llama.cpp CLI and C API compatibility.
6. Add reproducible benchmarks and profiling scripts.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
