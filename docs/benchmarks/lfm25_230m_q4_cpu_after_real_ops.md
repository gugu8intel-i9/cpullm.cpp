# LFM2.5-230M Q4_0 CPU comparison after real-operator upgrades

Date: 2026-06-27

## Environment

- CPU: Intel Xeon Processor @ 2.60GHz
- vCPUs: 2
- CPU features observed by system: AVX2, AVX-512F, AVX-512 VNNI and related extensions
- Model: `unsloth/LFM2.5-230M-GGUF`, file `LFM2.5-230M-Q4_0.gguf`
- Model file size: 143 MB locally, 140.05 MiB as reported by llama.cpp
- llama.cpp release: `b9821`, build `050ee92d0 (9821)`
- llama.cpp backend: CPU icelake

## llama.cpp: full real inference

Command:

```bash
./llama/llama-bench \
  -m models/LFM2.5-230M-Q4_0.gguf \
  -ngl 0 \
  -pg 512,128 \
  -t 2 \
  -r 3 \
  -o md
```

Result:

| model          | size       | params   | backend | threads | test        | t/s           |
| -------------- | ---------: | -------: | ------- | ------: | ----------: | ------------: |
| lfm2 230M Q4_0 | 140.05 MiB | 229.69 M | CPU     | 2       | pp512       | 519.19 ± 1.48 |
| lfm2 230M Q4_0 | 140.05 MiB | 229.69 M | CPU     | 2       | tg128       | 85.78 ± 0.88  |
| lfm2 230M Q4_0 | 140.05 MiB | 229.69 M | CPU     | 2       | pp512+tg128 | 236.28 ± 1.40 |

## cpullm.cpp after upgrades

Commit under test includes:

- memory-mapped GGUF metadata and tensor directory loading
- GGUF tokenizer token-array loading
- GGUF-native Q4_0 mapped matvec kernel
- RMSNorm
- RoPE
- causal attention
- short-convolution primitive
- SwiGLU MLP
- logits projection
- MTP/speculative accept-reject cores
- no-mock real GGUF decode boundary

Validation:

```text
cpullm core tests passed
cpullm ops tests passed
```

### LFM2.5 GGUF inspection

```text
name=Lfm2.5-230M
architecture=lfm2
context_length=128000
blocks=14
embedding_length=1024
feed_forward_length=2560
attention_heads=16
attention_kv_heads=0
vocab_size=65536
mtp_present=false
mtp_head_count=0
GGUF v3 tensors=132 metadata=45 alignment=32 arch=lfm2 name="Lfm2.5-230M"
```

### cpullm mapped GGUF Q4_0 kernel benchmark

Command:

```bash
/tmp/cpullm-gguf-q4-bench .cache/bench/models/LFM2.5-230M-Q4_0.gguf
```

Result:

```text
model=GGUF v3 tensors=132 metadata=45 alignment=32 arch=lfm2 name="Lfm2.5-230M"
tensor=blk.0.ffn_gate.weight rows=2560 cols=1024 bytes=1474560 iters=200 us=642834 matvec_per_sec=311.122 checksum=-0.254247
```

This is a real mapped-weight kernel benchmark over the GGUF file. It is **not** full model inference tokens/sec.

### cpullm full decode attempt

Command:

```bash
/tmp/cpullm-cli \
  -m .cache/bench/models/LFM2.5-230M-Q4_0.gguf \
  -p test \
  -n 1
```

Result:

```text
exit code: 1
cpullm error: real decode loop is implemented only for supported real executor architectures; this model architecture is 'lfm2'
```

This is intentional. After the no-mock upgrade, cpullm refuses to generate synthetic text for unsupported GGUF architectures. LFM2.5 graph wiring is not complete, so there is no valid cpullm tokens/sec number for LFM2.5 full decode yet.

## Comparison

| Runtime | Full LFM2.5 Q4_0 CPU inference | Prompt processing | Token generation | Notes |
| --- | --- | ---: | ---: | --- |
| llama.cpp b9821 | Yes | 519.19 ± 1.48 t/s | 85.78 ± 0.88 t/s | Complete LFM2 executor in llama.cpp |
| cpullm.cpp upgraded | No, fails honestly | N/A | N/A | Has real mapped GGUF loading, tokenizer tokens, operators, and Q4 mapped-kernel benchmark; LFM2 graph wiring remains |

## Interpretation

The upgrades moved cpullm from synthetic GGUF output to a no-mock architecture boundary. That is a quality improvement, not a throughput win yet.

The current baseline to beat remains llama.cpp's CPU `tg128` score of **85.78 ± 0.88 tokens/sec** and `pp512` score of **519.19 ± 1.48 tokens/sec** on this 2-vCPU AVX-512 sandbox.

cpullm's next required milestone for a fair apples-to-apples benchmark is exact LFM2 graph execution wiring:

1. token embedding lookup, including quantized GGUF types beyond Q4_0
2. per-block RMSNorm and residual ordering
3. LFM2 short-convolution/hybrid block semantics
4. attention q/k/v/o projection tensor mapping
5. RoPE placement and head layout
6. MLP gate/up/down tensor mapping across Q4_0/Q4_1/Q8/etc.
7. final norm and logits projection
8. tokenizer encode/decode parity with GGUF GPT-2/LFM2 tokenizer metadata
9. threaded kernel dispatch
