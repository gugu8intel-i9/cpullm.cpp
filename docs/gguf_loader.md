# GGUF Loader

cpullm.cpp now includes a lightweight memory-mapped GGUF loader. This is a production prerequisite for real llama.cpp-style inference because it avoids copying model weights into heap-owned buffers and gives kernels direct typed views over tensor bytes.

## Features

- Cross-platform mmap-style file ownership (`mmap` on POSIX, file mapping on Windows).
- GGUF magic/version/count validation.
- Metadata parser for scalar and array values.
- Tensor directory parser with shapes, GGML type IDs, byte sizes, and aligned offsets.
- LFM2/LFM2.5 metadata extraction through `Model::load()`.
- Tensor lookup by name.
- Zero-copy tensor byte views.
- Q4_0 matrix benchmark over actual mapped GGUF tensor data.

## Inspect LFM2.5

```bash
./build/cpullm-cli -m LFM2.5-230M-Q4_0.gguf --inspect
```

## Benchmark a mapped Q4_0 tensor

```bash
./build/cpullm-gguf-q4-bench LFM2.5-230M-Q4_0.gguf
```

This benchmark does not claim full model inference. It isolates one required hot path: reading a Q4_0 matrix directly from the mapped GGUF and running cpullm's q4_0 x f32 matvec primitive.

## Next operator work

The loader enables the next stage: connect mapped LFM2.5 tensors to RMSNorm, RoPE, short convolution/hybrid blocks, attention, MLP, and logits projection.

## LFM2.5-230M Q4_0 validation result

Validated against `LFM2.5-230M-Q4_0.gguf` from `unsloth/LFM2.5-230M-GGUF`:

```text
name=Lfm2.5-230M
architecture=lfm2
context_length=128000
blocks=14
embedding_length=1024
feed_forward_length=2560
attention_heads=16
vocab_size=65536
GGUF v3 tensors=132 metadata=45 alignment=32 arch=lfm2 name="Lfm2.5-230M"
```

Mapped GGUF-native Q4_0 tensor benchmark on the 2-vCPU AVX-512 sandbox:

```text
tensor=blk.0.ffn_gate.weight rows=2560 cols=1024 bytes=1474560 iters=200 us=639809 matvec_per_sec=312.593 checksum=-0.254247
```

This is a real mapped-weight kernel benchmark over bytes from the GGUF file, not synthetic generated data.
