# LFM2.5-230M Q4_0 CPU benchmark

Date: 2026-06-27

## Model

- Repository: `unsloth/LFM2.5-230M-GGUF`
- File: `LFM2.5-230M-Q4_0.gguf`
- Download URL: <https://huggingface.co/unsloth/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_0.gguf>
- Local file size during this run: 143 MB

## Hardware / runtime

- Sandbox CPU: Intel Xeon Processor @ 2.60GHz
- vCPU count: 2
- Observed CPU features: SSE2, AVX2, AVX-512F
- llama.cpp release: `b9821`, build `050ee92d0 (9821)`
- llama.cpp backend loaded: CPU icelake
- llama.cpp command:

```bash
./llama/llama-bench \
  -m models/LFM2.5-230M-Q4_0.gguf \
  -ngl 0 \
  -pg 512,128 \
  -t 2 \
  -r 3 \
  -o md
```

## llama.cpp result: real GGUF inference

| model          | size       | params   | backend | threads | test        | t/s            |
| -------------- | ---------: | -------: | ------- | ------: | ----------: | -------------: |
| lfm2 230M Q4_0 | 140.05 MiB | 229.69 M | CPU     | 2       | pp512       | 547.11 ± 2.01  |
| lfm2 230M Q4_0 | 140.05 MiB | 229.69 M | CPU     | 2       | tg128       | 87.54 ± 0.72   |
| lfm2 230M Q4_0 | 140.05 MiB | 229.69 M | CPU     | 2       | pp512+tg128 | 238.51 ± 3.48  |

## cpullm.cpp result: not comparable yet

cpullm.cpp does **not** yet run real GGUF transformer inference. It currently:

1. Opens/probes the GGUF header.
2. Builds a session and KV-cache.
3. Emits deterministic synthetic tokens through the streaming callback path.

The current cpullm smoke command was:

```bash
.cache/bench/cpullm-bin/cpullm-cli \
  -m .cache/bench/models/LFM2.5-230M-Q4_0.gguf \
  -p "Benchmark prompt" \
  -n 128 \
  --temp 0 \
  --stream
```

Observed synthetic stream timing over five process launches:

```text
runs seconds: 0.002485, 0.002184, 0.002067, 0.002074, 0.002093
mean seconds: 0.002181
synthetic tokens/sec: 58699.33
```

This number is useful only as a session/CLI overhead smoke test. It must not be compared with llama.cpp's real model inference tokens/sec.

## Interpretation

The honest baseline to beat is llama.cpp's CPU `tg128` score of **87.54 ± 0.72 tokens/sec** and `pp512` score of **547.11 ± 2.01 tokens/sec** on this 2-vCPU AVX-512 sandbox.

cpullm.cpp needs the following before a fair benchmark can be produced:

1. Full GGUF metadata parsing.
2. Memory-mapped GGUF tensor loading.
3. LFM2/LFM2.5 architecture support.
4. RMSNorm, RoPE, attention, MLP, convolution/hybrid blocks if required by the LFM architecture, and logits projection.
5. Real tokenizer loading from GGUF metadata.
6. Q4_0/Q4_K CPU kernel dispatch and threading.

## Reproduce

```bash
python3 scripts/benchmark_lfm25_230m_q4_cpu.py --threads 2
```

The script downloads the llama.cpp CPU release, downloads the Q4_0 GGUF model, builds cpullm with `g++ -O3 -march=native`, runs llama.cpp `llama-bench`, and records cpullm's current GGUF-probe/synthetic-stream smoke measurement separately.
