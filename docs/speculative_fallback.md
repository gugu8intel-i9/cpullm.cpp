# Speculative Fallback Policy

cpullm.cpp supports MTP and classic draft-model speculative decoding as opt-in acceleration modes. The model must still run when the acceleration is unavailable.

## Default behavior: graceful fallback

By default, if a user requests MTP or speculative decoding and cpullm cannot execute it for the loaded model/runtime, cpullm falls back to normal decoding.

Examples:

- `--spec-type mtp` on a GGUF with no MTP heads falls back to normal decoding.
- `--draft-model draft.gguf` before the target/draft executor is wired falls back to normal decoding.
- missing `--spec-draft-n-max` falls back to normal decoding.

The runtime records the reason in:

```cpp
InferenceSession::speculative_state().fallback_reason
```

Non-streaming CLI output includes:

```text
spec_active=false
spec_fallback="..."
```

This is **not** mock speculative decoding. It is normal decoding with an explicit fallback report.

## Strict behavior

Use strict mode when a deployment requires speculative acceleration and should fail otherwise:

```bash
./build/cpullm-cli \
  -m model.gguf \
  --spec-type mtp \
  --spec-draft-n-max 3 \
  --spec-strict
```

In strict mode, the same unsupported condition becomes an error.

## Why fallback by default?

A drop-in llama.cpp-style runtime should not make a valid model unusable just because an optional acceleration path is unavailable. Fallback keeps the model usable while still making the acceleration status visible and auditable.
