# Inference Engine

cpullm.cpp now has a production-oriented inference scaffold rather than a CLI-only placeholder.

## Implemented layers

- **Model loading front door:** `Model::load()` dispatches between YAML manifests and GGUF probing.
- **GGUF probing:** validates GGUF magic and reads version, tensor count, and metadata count without pulling the whole file into memory.
- **Tensor store:** central typed tensor registry for future memory-mapped model weights.
- **Quantized kernels:** q4_0 packing plus q4_0 x f32 dot/matvec primitives.
- **Sampler:** temperature, top-k, top-p, deterministic seedable PRNG, and greedy mode.
- **KV cache:** explicit capacity and allocation accounting for decoder sessions.
- **Session API:** reusable `InferenceSession` with string generation and streaming token callbacks.
- **CLI streaming:** `--stream` emits token events immediately.
- **Benchmark harness:** `cpullm-bench` gives a simple q4_0 matvec timing target.

## Production direction

The engine is intentionally split into small independently testable pieces. Real model execution will connect GGUF tensor loading to transformer graph execution through this path:

1. Memory-map GGUF tensor data.
2. Register tensors in `TensorStore`.
3. Select quantized kernels through CPU feature dispatch.
4. Execute decoder layers with paged KV-cache slots.
5. Stream sampler output through `InferenceSession`.

This is a clean-room path toward llama.cpp-style ergonomics with a smaller core and sharper extension points.
