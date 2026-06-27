# cpullm.cpp

A high-performance, lightweight, CPU-first LLM runtime foundation designed to become a drop-in alternative to llama.cpp.

> Status: active inference-engine foundation. The project now includes llama.cpp-style CLI entry points, common llama.cpp flag parsing, a minimal llama-compatible C ABI scaffold, streaming generation APIs, session/KV-cache management, sampler stack, q4_0 quantized matvec primitives, GGUF probing, tensor storage, benchmark scaffolding with finite deterministic inputs, warning-clean CPU feature detection, arena allocation, tokenizer scaffolding, model manifest loading, tests, and architecture notes.

## Goals

- **Drop-in llama.cpp ergonomics:** build a `llama-cli` target and accept common flags like `-m`, `-p`, `-n`, `--temp`, `-c`, `-t`, and `--stream`.
- **Lightweight:** small C++20 core, no required third-party dependencies.
- **Fast by design:** scratch arenas, q4_0 primitives, cache-aware kernels, runtime CPU feature dispatch, and a path toward fused graph execution.
- **Novel architecture:** clean separation between model format probing, tensor storage, session state, sampling, KV cache, and kernels.
- **Production-oriented:** testable layers, deterministic seeded sampling, streaming callback API, benchmark target, and compatibility wrappers at the edge.
- **Apache-2.0:** permissive licensing for research and production use.

## Repository layout

```text
apps/              cpullm-cli / llama-cli entry point
benchmarks/        microbenchmarks for kernel work
include/cpullm/    public C++ API and llama-compatible C ABI scaffold
src/               runtime implementation
docs/              architecture, inference engine, compatibility plan, and roadmap notes
examples/          minimal embedding and toy manifest
tests/             dependency-free core tests
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
./build/cpullm-cli -m examples/toy-model.yml -p "Hello from cpullm" -n 32 --temp 0.8 --stream
```

llama.cpp-style name:

```bash
./build/llama-cli -m examples/toy-model.yml -p "Hello from cpullm" -n 32 --temp 0.8 -c 2048 -t 0 --stream
```

Benchmark:

```bash
./build/cpullm-bench
```

## Current components

- `cpullm::Arena` — cache-line aligned scratch allocator for hot-path temporary memory.
- `cpullm::detect_cpu_features()` — portable runtime feature discovery for x86 and ARM targets.
- `cpullm::matmul_f32()` — blocked scalar baseline that future SIMD microkernels can replace through dispatch.
- `cpullm::quantize_q4_0()` / `matvec_q4_0_f32()` — compact quantized kernel primitives for lightweight inference.
- `cpullm::Sampler` — top-k, top-p, temperature, greedy mode, and deterministic seed support.
- `cpullm::KVCache` — explicit decoder cache capacity and memory accounting.
- `cpullm::InferenceSession` — reusable generation state with streaming token callbacks.
- `cpullm::TensorStore` — typed tensor registry for future memory-mapped model weights.
- `cpullm::probe_gguf()` — lightweight GGUF header validation and metadata counts.
- `cpullm::Tokenizer` — minimal whitespace tokenizer placeholder for API development.
- `cpullm::Model::load()` — model loading front door for YAML manifests and GGUF probing.
- `cpullm::Engine` — generation facade ready for graph execution and real transformer layers.
- `include/cpullm/llama_compat.h` — minimal llama.cpp-style C ABI scaffold for migration experiments, routed through the same `Model::load()` front door as the native CLI.

## llama.cpp compatibility

See [docs/llama_compatibility.md](docs/llama_compatibility.md).

Today, cpullm.cpp accepts common llama.cpp-style CLI invocations for YAML manifests and can probe GGUF headers. Full GGUF tensor loading and transformer execution are the next major milestones.

## Inference engine

See [docs/inference_engine.md](docs/inference_engine.md).

The engine is now structured around reusable sessions, explicit KV cache, quantized kernels, deterministic sampling, and streaming callbacks. The generated text is still synthetic until real model tensor loading and transformer graph execution are connected.

## Roadmap

1. Memory-map GGUF tensor data and register tensors in `TensorStore`.
2. Implement RMSNorm, RoPE, attention, MLP, and logits projection operators.
3. Add AVX2, AVX-512, and NEON q4_0/q8_0 microkernels behind runtime dispatch.
4. Implement paged KV cache blocks and batch/decode scheduling.
5. Broaden llama.cpp CLI and C API compatibility.
6. Add reproducible benchmarks and profiling scripts against public model families.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
