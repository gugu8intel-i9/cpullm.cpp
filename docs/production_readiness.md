# Production Readiness Validator

cpullm.cpp now includes a real execution planner instead of guessing whether a GGUF can run.

## Commands

```bash
./build/cpullm-cli -m model.gguf --check
./build/cpullm-cli -m model.gguf --dump-plan
```

`--check` prints a text report and exits with:

- `0` when the model is runnable by the current real executor
- `3` when the model is valid but blocked by missing kernels, unsupported architecture wiring, tokenizer gaps, or missing tensors

`--dump-plan` emits JSON for CI and automated compatibility dashboards.

## What it validates

- GGUF presence and metadata
- tokenizer token array availability
- tensor count and required tensor names for LFM2
- quantization/kernel coverage (`f32`, `q4_0`, `q4_1`, `q8_0`, unknown types)
- required operator set
- architecture-specific execution blockers

## No mock policy

The planner does not make unsupported models appear runnable. If LFM2 graph wiring is incomplete, it reports `executor.lfm2.graph_unwired`. This keeps the project production-honest while making the remaining work explicit.

## Why this is useful

A production drop-in runtime needs deterministic compatibility reporting. Users and CI systems should be able to distinguish:

1. malformed models,
2. valid models blocked by missing kernels,
3. valid models blocked by architecture graph wiring,
4. fully runnable models.

The planner is the foundation for that compatibility contract.
