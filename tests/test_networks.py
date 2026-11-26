import itertools

import numpy as np

from examples.networks import EXPERIMENTS
from src.autograd.tensor import Tensor
from src.math_basics.normalization import dropout, norm_vjp


def test_neuron_and_xor():
    weight, initial, final = EXPERIMENTS[12]()
    assert final < initial * 1e-6
    np.testing.assert_allclose(weight, 2, atol=4e-4)
    initial, final, predictions = EXPERIMENTS[13]()
    assert final < initial * 1e-7
    np.testing.assert_allclose(predictions.ravel(), [0, 1, 1, 0], atol=1e-3)


def test_manual_backward_and_tensor_graph():
    analytical, numerical = EXPERIMENTS[14]()
    np.testing.assert_allclose(analytical, numerical, atol=1e-9, rtol=1e-7)
    value, grad, loss, x_grad, bias_grad = EXPERIMENTS[15]()
    assert (value, grad, loss) == (6, 5, 36)
    np.testing.assert_array_equal(x_grad, [[3, 4], [3, 4]])
    np.testing.assert_array_equal(bias_grad, [4, 6])


def test_tensor_required_operators_and_accumulated_indices():
    p = Tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]], True)
    out = p[[1, 1, 0]].reshape(1, 3, 2).transpose(-2, 0, -1).exp().log().sum(axis=(0, 2)).sum()
    out.backward()
    np.testing.assert_allclose(p.grad, [[1, 1], [2, 2], [0, 0]], atol=1e-14)
    out.backward()
    np.testing.assert_allclose(p.grad, [[2, 2], [4, 4], [0, 0]], atol=1e-14)


def test_norm_and_dropout():
    EXPERIMENTS[16]()
    x = np.array([[0.1, 0.3, -0.2], [1., 3., -2.]])
    gain = np.array([1., 1.5, 2.])
    g = np.array([[1., -2., 3.], [1., -2., -3.]])
    for center in (True, False):
        expected, _, _ = norm_vjp(x, gain, g, center=center)
        def loss(data):
            u = data - data.mean(axis=-1, keepdims=True) if center else data
            return np.sum(u / np.sqrt(np.mean(u ** 2, axis=-1, keepdims=True) + 1e-5) * gain * g)
        for index in itertools.product(range(2), range(3)):
            perturb = np.zeros_like(x)
            perturb[index] = 1e-5
            numerical = (loss(x + perturb) - loss(x - perturb)) / 2e-5
            np.testing.assert_allclose(expected[index], numerical, atol=1e-8, rtol=2e-7)
    rng = np.random.default_rng(42)
    ones = np.ones(10_000)
    assert dropout(ones, 0.25, rng, training=False) is ones
    values = dropout(ones, 0.25, rng, training=True)
    assert set(values) == {0., 4 / 3}
    assert abs(values.mean() - 1) < 0.02
