# cpullm.cpp

A high-performance, lightweight, CPU-first LLM runtime foundation designed to become a drop-in alternative to llama.cpp.

> Status: active inference-engine foundation. The project now includes llama.cpp-style CLI entry points, common llama.cpp flag parsing plus model-family-agnostic production-readiness validation including warning-clean MTP/speculative options with graceful fallback by default and strict mode, a minimal llama-compatible C ABI scaffold, streaming generation APIs, session/KV-cache management, sampler stack, real RMSNorm/RoPE/attention/short-convolution/SwiGLU/logits operators, GGUF tokenizer loading, real decode fail-fast boundary, internal q4_0 primitives, GGUF-native 18-byte q4_0 mapped matvec, hardened memory-mapped GGUF metadata/tensor-directory loading, tensor storage, mapped-weight benchmark scaffolding with finite deterministic inputs, warning-clean CPU feature detection, arena allocation, tokenizer scaffolding, model manifest loading, tests, and architecture notes.

## Goals

- **Drop-in llama.cpp ergonomics:** build a `llama-cli` target and accept common flags like `-m`, `-p`, `-n`, `--temp`, `-c`, `-t`, `--stream`, `--check`, `--dump-plan`, `--list-architectures`, `--accel`, `--spec-type mtp`, `--spec-type draft`, `--draft-model`, `--spec-draft-n-max`, and `--spec-strict`.
- **Lightweight:** small C++20 core, no required third-party dependencies.
- **Fast by design:** scratch arenas, zero-copy memory-mapped GGUF tensor access, real operator primitives, real MTP/speculative accept/reject core, q4_0 primitives, cache-aware kernels, runtime CPU feature dispatch, and a path toward fused graph execution.
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


Universal acceleration:

```bash
./build/cpullm-cli -m model.gguf --accel balanced -p "Hello"
./build/cpullm-cli -m model.gguf --accel turbo -p "Hello"
```

Production readiness check:

```bash
./build/cpullm-cli -m model.gguf --check
./build/cpullm-cli -m model.gguf --dump-plan
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
- `cpullm::speculative_greedy_decode()` — real callback-based draft/verify speculative accept/reject loop with stats; no mock token path.
- `cpullm::mtp_greedy_decode()` — MTP wrapper over the same production accept/reject core for built-in draft heads.
- `cpullm::KVCache` — explicit decoder cache capacity and memory accounting.
- `cpullm::InferenceSession` — reusable generation state with streaming token callbacks.
- `cpullm::TensorStore` — typed tensor registry for future memory-mapped model weights.
- `cpullm::rms_norm`, `rope_inplace`, `causal_attention`, `short_convolution_1d`, `swiglu_mlp`, and `logits_projection` — real lightweight decoder operator primitives.
- `cpullm::Tokenizer::from_gguf()` — loads tokenizer token arrays from GGUF metadata.
- `cpullm::greedy_decode_real()` — real decode boundary that refuses unsupported GGUF architectures instead of mocking output.
- `cpullm::architecture_profiles()` — registry for common GGUF families including LLaMA, Qwen, Mistral/Mixtral, Gemma, Phi, DeepSeek, Granite, Falcon, MPT, StarCoder, LFM, and more.
- `cpullm::build_execution_plan()` — production-readiness planner for GGUF compatibility, kernel coverage, tensor requirements, and architecture blockers across model families.
- `cpullm::apply_residency_policy()` — universal GGUF tensor residency/prefetch acceleration that can use more RAM/startup compute to reduce cold page faults.
- `cpullm::GgufFile` — zero-copy memory-mapped GGUF metadata and tensor-directory loader with tensor byte views and robust numeric metadata extraction for LFM2/LFM2.5.
- `cpullm::probe_gguf()` — lightweight GGUF validation and metadata counts.
- `cpullm::Tokenizer` — minimal whitespace tokenizer placeholder for API development.
- `cpullm::Model::load()` — model loading front door for YAML manifests and GGUF probing.
- `cpullm::Engine` — generation facade ready for graph execution and real transformer layers.
- `include/cpullm/llama_compat.h` — minimal llama.cpp-style C ABI scaffold for migration experiments, routed through the same `Model::load()` front door as the native CLI.

## llama.cpp compatibility

See [docs/llama_compatibility.md](docs/llama_compatibility.md), [docs/mtp.md](docs/mtp.md), [docs/speculative_decoding.md](docs/speculative_decoding.md), and [docs/speculative_fallback.md](docs/speculative_fallback.md).

Today, cpullm.cpp accepts common llama.cpp-style CLI invocations for YAML manifests, can memory-map GGUF metadata/tensor directories, and exposes MTP and classic draft-model speculative flags that gracefully fall back to normal decoding unless real verified speculative execution is possible; `--spec-strict` restores fail-fast behavior. Full LFM2.5 transformer operator execution is the next major milestone; the benchmark report now records llama.cpp as the real CPU baseline cpullm must beat.

## Latest LFM2.5 comparison

After the real-operator/no-mock upgrade, llama.cpp `b9821` runs `LFM2.5-230M-Q4_0.gguf` on the 2-vCPU AVX-512 sandbox at **519.19 ± 1.48 pp512 tokens/sec** and **85.78 ± 0.88 tg128 tokens/sec**. cpullm.cpp now loads/inspects the GGUF, loads tokenizer token arrays, runs real operator tests, and benchmarks a mapped GGUF Q4_0 tensor at **311.122 matvec/sec** for `blk.0.ffn_gate.weight`, but it correctly refuses full LFM2.5 decode because the `lfm2` graph is not wired yet. See [docs/benchmarks/lfm25_230m_q4_cpu_after_real_ops.md](docs/benchmarks/lfm25_230m_q4_cpu_after_real_ops.md).

## Benchmarks

The first real baseline report is available at [docs/benchmarks/lfm25_230m_q4_cpu.md](docs/benchmarks/lfm25_230m_q4_cpu.md). On the 2-vCPU AVX-512 sandbox, llama.cpp `b9821` runs `LFM2.5-230M-Q4_0.gguf` at **547.11 ± 2.01 pp512 tokens/sec** and **87.54 ± 0.72 tg128 tokens/sec** on CPU. cpullm.cpp is not fairly comparable yet because it can load mapped GGUF metadata/tensors but does not yet execute the full LFM2.5 transformer graph.

Reproduce with:

```bash
python3 scripts/benchmark_lfm25_230m_q4_cpu.py --threads 2
```

## Inference engine

See [docs/inference_engine.md](docs/inference_engine.md), [docs/gguf_loader.md](docs/gguf_loader.md), [docs/real_executor.md](docs/real_executor.md), [docs/production_readiness.md](docs/production_readiness.md), [docs/architecture_registry.md](docs/architecture_registry.md), and [docs/universal_acceleration.md](docs/universal_acceleration.md).

The engine is now structured around reusable sessions, explicit KV cache, quantized kernels, deterministic sampling, streaming callbacks, speculative fallback reporting, and zero-copy mapped GGUF tensor access. LFM2.5 metadata and Q4_0 tensors can be loaded and benchmarked directly from GGUF. GGUF generation now enters a real decode boundary and fails for unsupported architectures instead of producing synthetic text.

## Roadmap

1. Wire generic dense-transformer graph execution first, then family-specific LFM hybrid and MoE paths.
2. Implement LFM2.5 RMSNorm, RoPE, attention, short-convolution/hybrid blocks, MLP, and logits projection operators.
3. Add AVX2, AVX-512, and NEON q4_0/q8_0 microkernels behind runtime dispatch.
4. Implement paged KV cache blocks and batch/decode scheduling.
5. Wire MTP draft-head tensors and separate draft-model speculative decoding to the verifier/drafter executor for supported architectures.
6. Broaden llama.cpp CLI and C API compatibility.
7. Add reproducible benchmarks and profiling scripts against public model families.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
