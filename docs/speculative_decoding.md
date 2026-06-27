# Speculative Decoding

cpullm.cpp now exposes classic draft-model speculative decoding as a real no-mock option.

## CLI

```bash
./build/cpullm-cli \
  -m target.gguf \
  --draft-model draft.gguf \
  --spec-type draft \
  --spec-draft-n-max 4 \
  -p "Hello"
```

Compatibility aliases:

- `--spec-type speculative`
- `--spec-type draft-model`
- `--draft-model`, `--model-draft`, or `-md`

## No mock policy

Speculative decoding requires two real executors:

1. a target/verifier model
2. a smaller draft model

If the user requests speculative decoding before cpullm has a real verifier/draft executor wired for the loaded architecture, cpullm exits with an error. It does not silently generate synthetic tokens and call them speculative.

## Implemented core

The repository includes the real greedy speculative accept/reject loop:

```cpp
cpullm::speculative_greedy_decode(...)
```

It is the same production algorithmic skeleton used by MTP:

1. Generate a draft sequence from the draft model.
2. Verify that sequence with the target model.
3. Accept matching draft tokens.
4. On first mismatch, emit the target token and continue from the corrected context.
5. Track verifier steps, drafted tokens, accepted tokens, and rejected tokens.

The implementation is callback-based so it stays lightweight and can be wired to any future architecture executor without pulling in heavyweight dependencies.

## Difference from MTP

- **MTP** uses draft heads inside the same GGUF/model.
- **Classic speculative decoding** uses a separate draft model.

Both share the same accept/reject core, but their draft callbacks come from different sources.
