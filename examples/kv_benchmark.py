"""Measure long-context forward peaks and decode tails on the NumPy CPU model."""

import argparse
from dataclasses import asdict
import json
import os
import platform
from time import perf_counter_ns
import tracemalloc

import numpy as np

from src.numpy_gpt.model import Config, GPT


def payload(cache):
    return sum(k.nbytes + v.nbytes for k, v in cache)


# region request
def request(model, ids, prompt, cached, *, profile=False):
    """Prefill once, then supply the same known ID to both paths at each step."""
    start = perf_counter_ns()
    if cached:
        out, cache = model.forward(ids[:, :prompt], track=False, use_cache=True)
    else:
        out = model.forward(ids[:, :prompt], track=False)
        cache = ()
    prefill_ms = (perf_counter_ns() - start) / 1e6
    logits = [out.data[:, -1].copy()]
    del out

    if profile:
        prefill_current, prefill_peak = tracemalloc.get_traced_memory()
        prefill_kv = payload(cache)
        transition = 0
        tracemalloc.reset_peak()

    decode_ms = []
    for position in range(prompt, ids.shape[1]):
        if profile:
            old_bytes = payload(cache)
        start = perf_counter_ns()
        if cached:
            out, cache = model.forward(
                ids[:, position:position + 1], track=False, cache=cache, use_cache=True
            )
        else:
            out = model.forward(ids[:, :position + 1], track=False)
        decode_ms.append((perf_counter_ns() - start) / 1e6)
        logits.append(out.data[:, -1].copy())
        del out
        if profile:
            # The old cache is still live while forward constructs and returns the new one.
            transition = max(transition, old_bytes + payload(cache))

    memory = None
    if profile:
        _, decode_peak = tracemalloc.get_traced_memory()
        memory = {
            "prefill_peak_bytes": prefill_peak,
            "decode_start_bytes": prefill_current,
            "decode_peak_bytes": decode_peak,
            "request_peak_bytes": max(prefill_peak, decode_peak),
            "prefill_kv_bytes": prefill_kv,
            "final_kv_bytes": payload(cache),
            "transition_kv_lower_bound_bytes": transition,
        }
    return prefill_ms, decode_ms, np.stack(logits, axis=1), memory
# endregion request


# region memory_pass
def measure_memory(model, ids, prompt, cached):
    if tracemalloc.is_tracing():
        raise RuntimeError("Run the benchmark with allocation tracing initially disabled")
    tracemalloc.start()
    try:
        return request(model, ids, prompt, cached, profile=True)[-1]
    finally:
        tracemalloc.stop()
# endregion memory_pass


def trace_calibration():
    if tracemalloc.is_tracing():
        raise RuntimeError("Run the benchmark with allocation tracing initially disabled")
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        sample = np.empty(1024 * 1024, dtype=np.uint8)
        after, _ = tracemalloc.get_traced_memory()
        observed = after - before
        if observed < sample.nbytes:
            raise RuntimeError("The tracer did not observe the NumPy data allocation")
        return {"array_bytes": sample.nbytes, "observed_bytes": observed}
    finally:
        tracemalloc.stop()


def percentiles(samples):
    return {key: float(value) for key, value in zip(
        ("p50_ms", "p95_ms", "p99_ms"),
        np.percentile(samples, [50, 95, 99], method="linear"), strict=True,
    )}


def check_logits(reference, actual):
    error = float(np.max(np.abs(reference - actual)))
    if not np.isfinite(error) or error > 1e-12:
        raise AssertionError(f"Cached and recomputed logits differ by {error}")
    return error


# region benchmark
def benchmark(prompt_lengths, *, decode_tokens, repetitions, warmups=1):
    if not prompt_lengths or min(prompt_lengths) < 1 or min(decode_tokens, repetitions, warmups) < 1:
        raise ValueError("Use positive lengths and at least one warm-up and measurement")
    config = Config(vocab=32, context=max(prompt_lengths) + decode_tokens,
                    channels=32, heads=4, hidden=96, layers=2)
    seed = 42
    model = GPT(config, seed=seed)
    ids = np.random.default_rng(seed).integers(config.vocab, size=(1, config.context))
    calibration = trace_calibration()
    cases = []
    for prompt in prompt_lengths:
        inputs = ids[:, :prompt + decode_tokens]
        prefill = {mode: [] for mode in ("recompute", "cache")}
        decode = {mode: [] for mode in prefill}
        max_error = 0.
        for repeat in range(warmups + repetitions):
            # Alternate which mode runs first. Tracing has stopped before this loop.
            modes = ("recompute", "cache") if repeat % 2 == 0 else ("cache", "recompute")
            runs = {mode: request(model, inputs, prompt, mode == "cache") for mode in modes}
            max_error = max(max_error, check_logits(runs["recompute"][2], runs["cache"][2]))
            if repeat >= warmups:
                for mode, result in runs.items():
                    prefill[mode].append(result[0])
                    decode[mode].append(result[1])

        conditions = {}
        for mode in prefill:
            conditions[mode] = {
                "prefill": percentiles(prefill[mode]),
                "decode": percentiles(np.asarray(decode[mode]).ravel()),
                "prefill_samples_ms": prefill[mode],
                "decode_samples_ms": decode[mode],
                "memory": measure_memory(model, inputs, prompt, mode == "cache"),
            }
        cases.append({"prompt_tokens": prompt, "max_logit_error": max_error, "modes": conditions})
    return {
        "config": asdict(config), "batch": 1, "dtype": "float64", "seed": seed,
        "decode_tokens": decode_tokens, "warmups": warmups, "repetitions": repetitions,
        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "trace_calibration": calibration, "cases": cases,
    }
# endregion benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", action="store_true", help="measure 128, 512 and 1024 prompt tokens")
    args = parser.parse_args()
    result = (benchmark((128, 512, 1024), decode_tokens=16, repetitions=16, warmups=2)
              if args.long else benchmark((16, 32), decode_tokens=3, repetitions=2))
    print(json.dumps(result, indent=2))
