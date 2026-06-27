# cpullm.cpp

A high-performance, lightweight, CPU-first LLM runtime foundation designed to become a drop-in alternative to llama.cpp.

> Status: active inference-engine foundation. The project now includes llama.cpp-style CLI entry points, common llama.cpp flag parsing, a minimal llama-compatible C ABI scaffold, streaming generation APIs, session/KV-cache management, sampler stack, internal q4_0 primitives, GGUF-native 18-byte q4_0 mapped matvec, hardened memory-mapped GGUF metadata/tensor-directory loading, tensor storage, mapped-weight benchmark scaffolding with finite deterministic inputs, warning-clean CPU feature detection, arena allocation, tokenizer scaffolding, model manifest loading, tests, and architecture notes.

## Goals

- **Drop-in llama.cpp ergonomics:** build a `llama-cli` target and accept common flags like `-m`, `-p`, `-n`, `--temp`, `-c`, `-t`, and `--stream`.
- **Lightweight:** small C++20 core, no required third-party dependencies.
- **Fast by design:** scratch arenas, zero-copy memory-mapped GGUF tensor access, q4_0 primitives, cache-aware kernels, runtime CPU feature dispatch, and a path toward fused graph execution.
- **Novel architecture:** clean separation between model format probing, tensor storage, session state, sampling, KV cache, and kernels.
- **Production-oriented:** testable layers, deterministic seeded sampling, streaming callback API, benchmark target, and compatibility wrappers at the edge.
- **Apache-2.0:** permissive licensing for research and production use.

## Repository layout

```text
apps/              cpullm-cli / llama-cli entry point
benchmarks/        microbenchmarks for kernel and mapped GGUF tensor work
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
./build/cpullm-gguf-q4-bench LFM2.5-230M-Q4_0.gguf
```

## Current components

- `cpullm::Arena` — cache-line aligned scratch allocator for hot-path temporary memory.
- `cpullm::detect_cpu_features()` — portable runtime feature discovery for x86 and ARM targets.
- `cpullm::matmul_f32()` — blocked scalar baseline that future SIMD microkernels can replace through dispatch.
- `cpullm::quantize_q4_0()` / `matvec_q4_0_f32()` — compact internal quantized kernel primitives for lightweight inference.
- `cpullm::matvec_gguf_q4_0_f32()` — GGUF-native q4_0 matvec over 18-byte GGML blocks with FP16 scales, avoiding copies/conversion.
- `cpullm::Sampler` — top-k, top-p, temperature, greedy mode, and deterministic seed support.
- `cpullm::KVCache` — explicit decoder cache capacity and memory accounting.
- `cpullm::InferenceSession` — reusable generation state with streaming token callbacks.
- `cpullm::TensorStore` — typed tensor registry for future memory-mapped model weights.
- `cpullm::GgufFile` — zero-copy memory-mapped GGUF metadata and tensor-directory loader with tensor byte views and robust numeric metadata extraction for LFM2/LFM2.5.
- `cpullm::probe_gguf()` — lightweight GGUF validation and metadata counts.
- `cpullm::Tokenizer` — minimal whitespace tokenizer placeholder for API development.
- `cpullm::Model::load()` — model loading front door for YAML manifests and GGUF probing.
- `cpullm::Engine` — generation facade ready for graph execution and real transformer layers.
- `include/cpullm/llama_compat.h` — minimal llama.cpp-style C ABI scaffold for migration experiments, routed through the same `Model::load()` front door as the native CLI.

## llama.cpp compatibility

See [docs/llama_compatibility.md](docs/llama_compatibility.md).

Today, cpullm.cpp accepts common llama.cpp-style CLI invocations for YAML manifests and can memory-map GGUF metadata/tensor directories. Full LFM2.5 transformer operator execution is the next major milestone; the benchmark report now records llama.cpp as the real CPU baseline cpullm must beat.

## Benchmarks

The first real baseline report is available at [docs/benchmarks/lfm25_230m_q4_cpu.md](docs/benchmarks/lfm25_230m_q4_cpu.md). On the 2-vCPU AVX-512 sandbox, llama.cpp `b9821` runs `LFM2.5-230M-Q4_0.gguf` at **547.11 ± 2.01 pp512 tokens/sec** and **87.54 ± 0.72 tg128 tokens/sec** on CPU. cpullm.cpp is not fairly comparable yet because it can load mapped GGUF metadata/tensors but does not yet execute the full LFM2.5 transformer graph.

Reproduce with:

```bash
python3 scripts/benchmark_lfm25_230m_q4_cpu.py --threads 2
```

## Inference engine

See [docs/inference_engine.md](docs/inference_engine.md) and [docs/gguf_loader.md](docs/gguf_loader.md).

The engine is now structured around reusable sessions, explicit KV cache, quantized kernels, deterministic sampling, streaming callbacks, and zero-copy mapped GGUF tensor access. LFM2.5 metadata and Q4_0 tensors can be loaded and benchmarked directly from GGUF; generated text is still synthetic until the transformer operators are connected.

## Roadmap

1. Connect memory-mapped GGUF tensors to LFM2.5 operator execution.
2. Implement LFM2.5 RMSNorm, RoPE, attention, short-convolution/hybrid blocks, MLP, and logits projection operators.
3. Add AVX2, AVX-512, and NEON q4_0/q8_0 microkernels behind runtime dispatch.
4. Implement paged KV cache blocks and batch/decode scheduling.
5. Broaden llama.cpp CLI and C API compatibility.
6. Add reproducible benchmarks and profiling scripts against public model families.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
