import json
from pathlib import Path

import pytest

from examples.kv_benchmark import benchmark, percentiles
from src.modern.efficiency import cache_bytes


def check_case(run, case):
    prompt = case["prompt_tokens"]
    steps = run["decode_tokens"]
    repetitions = run["repetitions"]
    config = run["config"]
    per_token = cache_bytes(config["layers"], run["batch"], config["heads"], 1,
                            config["channels"] // config["heads"], 8)

    assert case["max_logit_error"] <= 1e-12
    for mode, results in case["modes"].items():
        assert len(results["prefill_samples_ms"]) == repetitions
        assert len(results["decode_samples_ms"]) == repetitions
        assert all(len(row) == steps for row in results["decode_samples_ms"])
        samples = [duration for row in results["decode_samples_ms"] for duration in row]
        assert min(samples) > 0
        assert results["decode"] == percentiles(samples)
        assert results["prefill"] == percentiles(results["prefill_samples_ms"])

        memory = results["memory"]
        assert memory["request_peak_bytes"] == max(memory["prefill_peak_bytes"],
                                                   memory["decode_peak_bytes"])
        assert memory["decode_peak_bytes"] >= memory["decode_start_bytes"]
        if mode == "cache":
            assert memory["prefill_kv_bytes"] == prompt * per_token
            assert memory["final_kv_bytes"] == (prompt + steps) * per_token
            assert memory["transition_kv_lower_bound_bytes"] == (2 * (prompt + steps) - 1) * per_token
            assert memory["decode_start_bytes"] >= memory["prefill_kv_bytes"]
            assert memory["decode_peak_bytes"] >= memory["transition_kv_lower_bound_bytes"]
        else:
            assert all(memory[key] == 0 for key in
                       ("prefill_kv_bytes", "final_kv_bytes", "transition_kv_lower_bound_bytes"))


def test_percentiles_use_linear_interpolation():
    assert percentiles([1, 2, 3, 4, 10]) == pytest.approx(
        {"p50_ms": 3, "p95_ms": 8.8, "p99_ms": 9.76}
    )


def test_kv_benchmark_smoke_checks_both_paths():
    run = benchmark((8, 16), decode_tokens=3, repetitions=2)
    assert run["trace_calibration"]["observed_bytes"] >= 1024 * 1024
    for case in run["cases"]:
        check_case(run, case)


def test_archived_long_measurement_matches_its_raw_samples():
    run = json.loads(Path("reports/kv-long-context.json").read_text())
    assert [case["prompt_tokens"] for case in run["cases"]] == [128, 512, 1024]
    assert (run["warmups"], run["repetitions"], run["decode_tokens"]) == (2, 16, 16)
    assert run["openblas_num_threads"] == run["omp_num_threads"] == "1"
    for case in run["cases"]:
        check_case(run, case)
