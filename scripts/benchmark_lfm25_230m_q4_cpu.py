#!/usr/bin/env python3
"""Benchmark LFM2.5-230M Q4_0 on CPU with llama.cpp and cpullm.cpp.

This script intentionally reports cpullm separately until cpullm can execute
real GGUF transformer inference. Today cpullm can probe GGUF and run its own
synthetic/session + q4 microbench path; llama.cpp is the real inference baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "bench"
MODEL_URL = "https://huggingface.co/unsloth/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_0.gguf"
MODEL_NAME = "LFM2.5-230M-Q4_0.gguf"
LLAMA_RELEASE = "b9821"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz"


def run(cmd: list[str], cwd: Path = ROOT, timeout: int = 900, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"downloading {url} -> {dst}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dst)


def ensure_llama() -> Path:
    llama_dir = CACHE / "llama"
    bench = llama_dir / "llama-bench"
    if bench.exists():
        return bench
    archive = CACHE / "llama-ubuntu-x64.tar.gz"
    download(LLAMA_URL, archive)
    llama_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(llama_dir)
    candidates = list(llama_dir.rglob("llama-bench"))
    if not candidates:
        raise RuntimeError("llama-bench not found after extracting release archive")
    return candidates[0]


def ensure_model() -> Path:
    model = CACHE / "models" / MODEL_NAME
    download(MODEL_URL, model)
    return model


def build_cpullm() -> Path:
    out = CACHE / "cpullm-bin" / "cpullm-cli"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++", "-std=c++20", "-O3", "-march=native", "-DNDEBUG", "-Iinclude",
        *[str(p) for p in sorted((ROOT / "src").glob("*.cpp"))],
        "apps/cpullm-cli.cpp", "-o", str(out),
    ]
    run(cmd, timeout=180)
    return out


def cpu_info() -> str:
    if Path("/proc/cpuinfo").exists():
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def benchmark_llama(llama_bench: Path, model: Path, threads: int) -> str:
    p = run([
        str(llama_bench), "-m", str(model), "-ngl", "0", "-pg", "512,128",
        "-t", str(threads), "-r", "3", "-o", "md",
    ], cwd=CACHE, timeout=900)
    return p.stdout


def benchmark_cpullm(cpullm: Path, model: Path) -> dict[str, object]:
    cmd = [str(cpullm), "-m", str(model), "-p", "Benchmark prompt", "-n", "128", "--temp", "0", "--stream"]
    times: list[float] = []
    stderr = ""
    stdout_prefix = ""
    for _ in range(5):
        t0 = time.perf_counter()
        p = run(cmd, timeout=60)
        times.append(time.perf_counter() - t0)
        stderr = p.stderr.strip()
        stdout_prefix = p.stdout[:180].replace("\n", "\\n")
    return {
        "note": "cpullm currently probes GGUF and emits synthetic session tokens; this is not real model inference.",
        "runs_seconds": times,
        "mean_seconds": statistics.mean(times),
        "synthetic_tokens_per_second": 128 / statistics.mean(times),
        "stderr": stderr,
        "stdout_prefix": stdout_prefix,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--json", type=Path, default=CACHE / "lfm25_230m_q4_cpu_results.json")
    args = ap.parse_args()

    llama_bench = ensure_llama()
    model = ensure_model()
    cpullm = build_cpullm()

    llama_md = benchmark_llama(llama_bench, model, args.threads)
    cpullm_result = benchmark_cpullm(cpullm, model)

    result = {
        "model": MODEL_NAME,
        "model_url": MODEL_URL,
        "llama_release": LLAMA_RELEASE,
        "threads": args.threads,
        "cpu": cpu_info(),
        "llama_bench_markdown": llama_md,
        "cpullm": cpullm_result,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2))
    print("\n=== llama.cpp ===")
    print(llama_md)
    print("=== cpullm.cpp current status ===")
    print(json.dumps(cpullm_result, indent=2))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
