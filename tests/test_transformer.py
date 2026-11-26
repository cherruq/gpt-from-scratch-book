import numpy as np
import pytest

from examples.transformer import EXPERIMENTS, scaled
from src.autograd.tensor import Tensor
from src.numpy_gpt.model import Config, GPT, attention, cross_entropy


def test_attention_weights_and_mask():
    fixed, scores = EXPERIMENTS[22]()
    np.testing.assert_allclose(fixed, [[0.5, 3]])
    np.testing.assert_allclose(scores, [[np.sqrt(0.5), 0]])
    a = 1 / (1 + np.exp(-1 / np.sqrt(2)))
    weights, y = scaled()
    np.testing.assert_allclose(weights, [[a, 1-a], [1-a, a]], atol=1e-14)
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(y, [[2*a, 4*(1-a)], [2*(1-a), 4*a]])
    weights, y = scaled(with_mask=True)
    np.testing.assert_allclose(weights, [[1, 0], [1-a, a]])
    np.testing.assert_array_equal(y[0], [2, 0])
    q = Tensor(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="visible"):
        attention(q, q, q, [[False, False], [True, True]])


def test_multihead_ffn_and_causality():
    _, changes = EXPERIMENTS[24]()
    np.testing.assert_array_equal(changes[0, :2], 0)
    assert np.all(changes[0, 2:] > 0)
    axes, row, output, grad = EXPERIMENTS[25]()
    assert (axes, output, grad.shape) == ((1, 2, 4, 2), (1, 4, 4), (1, 4, 4))
    np.testing.assert_array_equal(row, [10, 11])
    assert np.all(np.isfinite(grad))
    z, h, y, _ = EXPERIMENTS[26]()
    np.testing.assert_array_equal(z, [[[1, -1, -1]]])
    np.testing.assert_allclose(y, [[[0.682384, -0.317616]]], atol=1e-6)


def test_attention_full_gradient():
    rng = np.random.default_rng(42)
    q, k, v = [Tensor(rng.normal(size=(1, 2, 3, 2)), True) for _ in range(3)]
    mask = np.tril(np.ones((3, 3), dtype=bool))
    def objective():
        return (attention(q, k, v, mask) ** 2).sum()
    objective().backward()
    for operand in (q, k, v):
        for index in np.ndindex(operand.shape):
            old = operand.data[index]
            operand.data[index] = old + 1e-5
            plus = objective().data
            operand.data[index] = old - 1e-5
            minus = objective().data
            operand.data[index] = old
            np.testing.assert_allclose(operand.grad[index], (plus - minus) / 2e-5,
                                       atol=2e-9, rtol=2e-7)


def test_residual_and_shared_parameter():
    _, _, identity, gradient = EXPERIMENTS[27]()
    np.testing.assert_array_equal(identity, np.arange(8.).reshape(1, 2, 4) / 8)
    np.testing.assert_array_equal(gradient, np.ones((1, 2, 4)))

    count, shape, loss, gradients = EXPERIMENTS[28]()
    assert (count, shape) == (544, (2, 4, 8))
    assert loss == pytest.approx(2.090428, abs=1e-6)
    assert gradients["token"] == (8, 4)
    model = GPT(Config(vocab=8))
    ids, targets = [[1, 1, 2]], [[1, 2, 3]]
    p = model.parameters["token"]
    cross_entropy(model.forward(ids), targets).backward()
    assert np.linalg.norm(p.grad[7]) > 0  # Token 7 is absent in the input; the head still uses it.
    for index in ((1, 0), (7, 0)):
        start = p.data[index]
        p.data[index] = start + 1e-6
        plus = cross_entropy(model.forward(ids, track=False), targets).data
        p.data[index] = start - 1e-6
        minus = cross_entropy(model.forward(ids, track=False), targets).data
        p.data[index] = start
        np.testing.assert_allclose(p.grad[index], (plus - minus) / 2e-6, atol=5e-8, rtol=5e-7)
