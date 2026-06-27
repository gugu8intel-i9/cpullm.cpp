# cpullm.cpp Architecture

cpullm.cpp starts from a small, CPU-first runtime core rather than a monolithic inference binary.

## Design pillars

- **Tiny public API:** one umbrella header while the internals are free to evolve.
- **Scratch-first memory:** arena allocation for hot-path temporary tensors and graph execution.
- **Kernel dispatch:** portable scalar kernels first, then AVX2/AVX-512/NEON implementations behind runtime CPU feature detection.
- **Format adapters:** keep model file parsing isolated from execution so GGUF, SafeTensors, and custom packed formats can coexist.
- **Composable graph:** operator nodes are intentionally simple today; scheduling, fusion, and KV-cache paging can be layered in without rewriting the CLI.

## Near-term roadmap

1. Quantized tensor blocks: q4_0 baseline, then q4_K/q5_K-style research variants.
2. Paged KV cache with cache-line aligned blocks.
3. Runtime kernel registry with feature-specific microkernels.
4. Streaming token API and sampler stack.
5. Benchmarks against llama.cpp-compatible model families using independently written code.
