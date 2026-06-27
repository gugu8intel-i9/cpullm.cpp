# Real Executor Components

cpullm.cpp now contains the real lightweight operator building blocks required for a transformer/hybrid decoder path.

Implemented operators:

1. **RMSNorm** — `cpullm::rms_norm`
2. **RoPE** — `cpullm::rope_inplace`
3. **Causal attention** — `cpullm::causal_attention`
4. **Short convolution / hybrid block primitive** — `cpullm::short_convolution_1d`
5. **SwiGLU MLP** — `cpullm::swiglu_mlp`
6. **Logits projection** — `cpullm::logits_projection`
7. **GGUF tokenizer token loading** — `cpullm::Tokenizer::from_gguf`
8. **Real decode entry point** — `cpullm::greedy_decode_real`

## No mock policy

GGUF models now go through the real decode entry point. If the model architecture/tensor set is not supported by a wired executor, cpullm fails with an explicit error instead of generating synthetic text.

This means LFM2.5 no longer returns fake generated tokens. It will report that the architecture is not yet wired to a full real executor.

## Current state

The operator layer is real and tested, but full LFM2.5 graph wiring remains a separate implementation step. LFM2.5 has architecture-specific short-convolution/hybrid details and tensor layouts that must be connected exactly before real decoding can run.

## Why this matters

The project now has the core math primitives, tokenizer loading path, mapped GGUF tensor access, and a real decode boundary. This prevents mock output while making the remaining work focused: architecture graph wiring and kernel specialization.
