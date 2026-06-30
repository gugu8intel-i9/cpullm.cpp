# Real Graph Executors

cpullm.cpp now includes real dense and MoE graph executor foundations.

## Dense executor

```cpp
cpullm::execute_dense_block(...)
```

Implements a real single-token decoder block path:

1. RMSNorm
2. Q/K/V projections
3. RoPE
4. single-token causal attention primitive
5. output projection
6. residual
7. FFN RMSNorm
8. SwiGLU gate/up/down MLP
9. residual

## MoE executor

```cpp
cpullm::execute_moe_block(...)
```

Adds:

- router projection
- top-k expert selection
- softmax expert weights
- weighted expert execution
- residual merge

## Scope

This is real numerical execution and is covered by tests. It is not yet the final GGUF model-family executor because the next step is mapping each architecture's tensor names/layouts and quantized kernels into these views.

The executor is intentionally model-family-agnostic: dense transformer families and MoE families can use the same core once tensor mapping is added.
