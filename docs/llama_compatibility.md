# llama.cpp Compatibility Plan

cpullm.cpp is being shaped as a lightweight, high-performance alternative that can be adopted with minimal changes by llama.cpp users.

## Compatibility targets

| Area | Current status | Target |
| --- | --- | --- |
| CLI name | `cpullm-cli` and `llama-cli` CMake targets | Accept common llama.cpp flags by default |
| CLI flags | `-m`, `-p`, `-n`, `--temp`, `-c`, `-t`, `--version`, `--help` | Broaden coverage for batch, GPU-ignore, seed, sampler, and prompt-cache flags |
| C API | Minimal `include/cpullm/llama_compat.h` scaffold | Practical subset of the llama.cpp embedding API |
| Model files | YAML manifest for foundation tests | Clean-room GGUF reader and tensor loader |
| Runtime | Arena, CPU features, scalar blocked matmul | Quantized kernels, dispatch registry, paged KV cache, streaming decode |

## Compatibility philosophy

1. **Drop-in ergonomics:** common command lines should work or fail with clear messages.
2. **Clean-room internals:** no dependency on llama.cpp implementation details.
3. **Small core:** compatibility wrappers live at the edges; the runtime remains compact.
4. **Performance first:** every compatibility layer should preserve a fast path to native cpullm kernels.

## Example

```bash
./build/llama-cli -m examples/toy-model.yml -p "Hello" -n 32 --temp 0.7 -c 2048 -t 0
```

This works today for cpullm YAML manifests. GGUF inference is planned next.
