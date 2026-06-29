# Universal Acceleration Profiles

cpullm.cpp now has a model-family-agnostic residency acceleration layer. It works with any GGUF model because it operates below model architecture semantics: on memory-mapped tensor pages.

## CLI

```bash
./build/cpullm-cli -m model.gguf --accel off      -p "Hello"
./build/cpullm-cli -m model.gguf --accel balanced -p "Hello"
./build/cpullm-cli -m model.gguf --accel turbo    -p "Hello"
```

## Profiles

- `off`: no residency work.
- `balanced`: issue OS prefetch advice for mapped tensor pages.
- `turbo`: issue prefetch advice and touch tensor pages to pull them into the OS page cache.

## Why this is compatible with all LLMs

The policy does not depend on LLaMA, Qwen, Gemma, LFM, MoE, or any model-specific graph. It walks the GGUF tensor directory and operates on mapped tensor byte ranges. That makes it universally applicable to GGUF LLMs.

## Trade-off

`turbo` uses more startup compute and memory bandwidth. It may improve cold-start latency by reducing page faults later, but it can be counterproductive if RAM is constrained or the model is much larger than available memory.

## No mock policy

This is a real OS/page-cache residency optimization, not fake inference acceleration. The CLI reports how many tensor bytes were considered and what action was taken.
