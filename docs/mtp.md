# Multi-Token Prediction (MTP)

cpullm.cpp now exposes a real MTP option, but it deliberately refuses to fake MTP output.

## CLI

```bash
./build/cpullm-cli \
  -m model-with-mtp-heads.gguf \
  -p "Hello" \
  --spec-type mtp \
  --spec-draft-n-max 3
```

The flags mirror current llama.cpp-style MTP usage:

- `--spec-type mtp` enables built-in multi-token prediction heads.
- `--spec-draft-n-max N` caps the number of draft tokens proposed per verifier step.

`draft-mtp` is accepted as an alias for compatibility.

## No mock policy

If MTP is requested and the model does not expose MTP/draft-head metadata or tensors, cpullm exits with an error.

If MTP heads are detected but the full cpullm verifier/drafter executor is not wired for that architecture yet, cpullm also exits with an error. It does **not** silently fall back to synthetic tokens or normal decoding while claiming MTP.

## Implemented core

The repository includes a real greedy MTP speculative-decoding core:

```cpp
cpullm::mtp_greedy_decode(...)
```

It performs the production accept/reject loop:

1. Ask the MTP draft head for up to `draft_n_max` future tokens.
2. Verify the draft sequence with the main model path.
3. Accept matching draft tokens.
4. On first mismatch, emit the verifier token and resume from the corrected context.
5. Track draft, accept, reject, and verifier-step statistics.

The core is callback-based so it remains lightweight and architecture-independent. The next step is wiring the callbacks to the LFM2.5 executor once the transformer operators are complete.

## GGUF capability detection

`Model::load()` scans GGUF metadata keys and tensor names for MTP/draft-head indicators and exposes:

```cpp
ModelMetadata::mtp.present
ModelMetadata::mtp.head_count
ModelMetadata::mtp.tensor_names
```

Inspect with:

```bash
./build/cpullm-cli -m model.gguf --inspect
```

## Why this shape?

MTP must be verified by the target model. A command-line flag alone is not MTP. This implementation keeps the option honest while putting the high-performance, lightweight accept/reject machinery in place for real model execution.
