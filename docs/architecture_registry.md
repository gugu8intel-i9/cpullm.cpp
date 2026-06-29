# Architecture Registry

cpullm.cpp is not intended to be optimized for one model family. The production planner now uses an architecture profile registry covering common GGUF model families.

## CLI

```bash
./build/cpullm-cli --list-architectures
./build/cpullm-cli -m model.gguf --check
./build/cpullm-cli -m model.gguf --dump-plan
```

## Recognized profile examples

- `llama`
- `qwen2`
- `qwen3`
- `qwen3moe`
- `mistral`
- `mixtral`
- `gemma`, `gemma2`, `gemma3`
- `phi2`, `phi3`
- `deepseek2`
- `granite`
- `gptneox`
- `falcon`
- `baichuan`
- `mpt`
- `starcoder`
- `lfm2`

Each profile records whether the architecture is dense transformer, hybrid, or MoE, and the required operator family.

## Purpose

The registry is a compatibility contract. It lets cpullm report whether a model is blocked by:

- missing tokenizer implementation
- missing quantized kernel
- missing tensor names
- missing dense transformer graph scheduling
- missing hybrid block scheduling
- missing MoE router/expert scheduling
- unrecognized architecture

This keeps the runtime model-family-agnostic while retaining honest no-mock behavior.
