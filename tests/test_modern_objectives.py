import numpy as np
import pytest

from examples.modern_objectives import policy_objectives, routing_budget, streaming_check


def test_gqa_projection_and_cache_counts_and_router_vjp():
    report = routing_budget()
    assert [r["projection_weights"] for r in report["resources"]] == [1024, 768, 640]
    assert [r["cache_bytes"] for r in report["resources"]] == [16384, 8192, 4096]
    assert report["load"] == [2/3, 1/3]
    assert report["auxiliary"] == pytest.approx(19/18)
    np.testing.assert_array_equal(report["task_router_gradient"], np.zeros((2, 2)))
    np.testing.assert_allclose(report["aux_router_gradient"],
                               [[1/12, -1/12], [1/24, -1/24]], atol=1e-15, rtol=0)


def test_streaming_rescale_and_speculative_residual():
    report = streaming_check()
    assert report["old_mass"] == pytest.approx(4/3)
    assert report["old_accumulator"] == pytest.approx(2)
    assert report["rescale"] == pytest.approx(0.5)
    assert report["new_mass"] == pytest.approx(5/3)
    assert report["new_accumulator"] == pytest.approx(5)
    assert report["streamed"] == pytest.approx(3, abs=1e-14, rel=0)
    assert report["shifted"] == pytest.approx(3, abs=1e-12, rel=0)
    assert report["acceptance"] == pytest.approx(.7)
    assert report["rejection_distribution"] == [1, 0, 0]
    assert report["combined"] == [.6, .3, .1]


def test_answer_only_loss_and_policy_objective_branches():
    report = policy_objectives()
    assert report["sft_loss"] == pytest.approx(np.log(4))
    np.testing.assert_array_equal(report["sft_gradients"],
                                  [[0, 0, 0, 0], [.125, .125, -.375, .125], [.125, .125, .125, -.375]])
    np.testing.assert_allclose(report["ppo"], [1.2, .7, -1.3, -.8], atol=1e-15, rtol=0)
    assert report["ppo_mean"] == pytest.approx(-.05)
    np.testing.assert_allclose(report["ppo_slopes"], [0, 1, -1, 0], atol=1e-10, rtol=0)
    assert report["group_advantage"] == [1, -1, 1, -1]
    assert report["equal_group_advantage"] == [0, 0, 0, 0]
    assert report["dpo_loss"] == pytest.approx(.644397, abs=5e-7, rel=0)
    assert report["dpo_slope"] == pytest.approx(-.047502, abs=5e-7, rel=0)
    assert report["dpo_slope_error"] < 1e-10
