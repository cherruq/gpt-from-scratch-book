import numpy as np
import pytest

from examples.training_bridge import batch_backprop, experiment
from src.autograd.tensor import Tensor


@pytest.fixture(scope="module")
def report():
    return experiment()


def test_lm_head_gradient_reduction_clipping_and_update(report):
    result = report["optimizer"]
    raw = np.array([[.5, -.5], [1., -1.]])
    np.testing.assert_array_equal(result["gradient"], raw)
    first, last = map(np.asarray, result["microbatch_gradients"])
    np.testing.assert_array_equal((first + last) / 2, np.zeros_like(raw))
    np.testing.assert_allclose(result["full_gradient"], (first + 2 * last) / 3, atol=1e-16)
    assert result["preclip_norm"] == pytest.approx(np.sqrt(2.5))
    clipped = raw / np.sqrt(2.5)
    np.testing.assert_allclose(result["m"], .1 * clipped, atol=1e-16)
    np.testing.assert_allclose(result["v"], .001 * clipped**2, atol=1e-16)
    np.testing.assert_allclose(result["weights"], -.1 * clipped / (abs(clipped) + 1e-8), atol=1e-15)
    assert result["final"] == pytest.approx(0.437488, abs=5e-7, rel=0)


def test_single_binary_output_logit(report):
    neuron = report["neuron"]
    assert neuron["gradient"] == -.25
    assert neuron["weight"] == .025
    np.testing.assert_array_equal(neuron["logits"], [[0., .025], [0., .05]])
    assert neuron["final"] == pytest.approx((np.logaddexp(0, .025) +
                                             np.logaddexp(0, -.05)) / 2)


def test_xor_initialization_and_activation_controls(report):
    result = report["xor"]
    assert result["designed_output"] == [0, 1, 1, 0]
    assert result["affine_mse"] == pytest.approx(.25)
    assert result["zero_init_final"] == .25 and result["zero_init_nonbias_weights"] == 0
    np.testing.assert_allclose(result["zero_init_output"], .5, atol=1e-15)
    assert result["random_init"][1] < 1e-20


def test_full_batch_backprop_manual_and_finite_difference(report):
    result = report["backprop"]
    assert result["loss"] == 1
    g = result["gradients"]
    a = np.log(3) / 2
    np.testing.assert_allclose(g["x"], [[1.5 * a, -a], [-2 * a, .75 * a]], atol=1e-15)
    np.testing.assert_array_equal(g["w1"], [[-2, 1.5], [3, -4]])
    np.testing.assert_array_equal(g["b1"], [-.5, -.5])
    np.testing.assert_array_equal(g["w2"], [[.5], [-.5]])
    assert g["b2"] == [0]
    assert result["difference_max_error"] < 2e-9
    assert result["final"] == pytest.approx(.968985, abs=5e-7, rel=0)


def test_same_gradients_distinct_step_sizes():
    small, large = batch_backprop(.02), batch_backprop(.3)
    assert small["gradients"] == large["gradients"]
    assert small["final"] < 1 < large["final"]
    assert small["final"] == pytest.approx(.493039, abs=5e-7, rel=0)
    assert large["final"] == pytest.approx(5.736078, abs=5e-7, rel=0)


def test_shared_scalar_gets_two_contributions():
    w = Tensor(2., requires_grad=True)
    (w * (w * 3.)).backward()
    assert w.grad == 12.
