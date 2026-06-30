# GGUF Tensor Type Support

cpullm.cpp now recognizes the common GGML/GGUF tensor type IDs and computes byte sizes without full dequantization.

## Low-RAM zero-copy matvec kernels

These formats have mapped-weight matvec kernels that operate directly on GGUF bytes:

- `F32`
- `F16`
- `Q4_0`
- `Q4_1`
- `Q8_0`

The kernels avoid allocating a full dequantized weight matrix. They stream/dequantize blocks while multiplying, which uses much less RAM than materializing F32 weights.

## Recognized formats with pending kernels

cpullm recognizes sizes and planner coverage for:

- `Q5_0`, `Q5_1`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`
- `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_NL`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, `IQ1_M`
- `BF16`
- `TQ1_0`, `TQ2_0`

These are reported as recognized-but-pending by the production planner rather than unknown.

## API

```cpp
cpullm::dtype_name(dtype)
cpullm::dtype_size(dtype)
cpullm::dtype_has_low_ram_matvec(dtype)
cpullm::matvec_gguf_any_f32(dtype, bytes, x, y, cols)
```

## Why this reduces RAM

Traditional dequantization can expand a Q4 model by roughly 8x into F32 memory. cpullm's mapped kernels keep the quantized bytes in the memory-mapped file and dequantize only the values needed for the current dot product.

This is slower than a heavily optimized packed SIMD path today, but it is memory efficient, simple, portable, and provides a correct foundation for later SIMD fused-dequant kernels.
