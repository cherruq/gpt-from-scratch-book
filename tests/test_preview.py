import numpy as np

from examples.preview import run
from src.autograd.tensor import Tensor
from src.numpy_gpt.model import Config, GPT, cross_entropy, softmax


def test_preview():
    shape, initial, final, row, output = run()
    assert shape == (2, 4, 8)
    np.testing.assert_allclose([initial, final], [2.090428, 0.015276], atol=1e-6, rtol=0)
    np.testing.assert_allclose(row.sum(), 1, atol=1e-14)
    np.testing.assert_allclose(row[1], 0.984031, atol=1e-6, rtol=0)
    assert output == "春夏秋冬春夏秋冬"


def test_causality_and_parameters():
    model = GPT(Config(vocab=8))
    assert model.parameter_count() == 544
    first = model.forward([[1, 2, 3, 4]], track=False).data
    second = model.forward([[1, 2, 7, 6]], track=False).data
    np.testing.assert_allclose(first[:, :2], second[:, :2], atol=1e-13, rtol=0)
    assert not np.allclose(first[:, 2:], second[:, 2:])


def test_stable_cross_entropy_and_softmax():
    logits = Tensor([[1000, 999, -1000]], True)
    loss = cross_entropy(logits, [1])
    loss.backward()
    expected = softmax(logits).data - [[0, 1, 0]]
    np.testing.assert_allclose(loss.data, np.logaddexp(0, 1), atol=1e-14)
    np.testing.assert_allclose(logits.grad, expected, atol=1e-14)
    assert np.isfinite(cross_entropy(logits, [2]).data)


def test_chapter_one_hand_calculation():
    scores = [0, np.log(2), 0, 0, 0, 0, 0, 0]
    probs = softmax(Tensor(scores)).data
    np.testing.assert_allclose(probs, np.array([1, 2, 1, 1, 1, 1, 1, 1]) / 9)
    np.testing.assert_allclose(cross_entropy(Tensor([scores]), [1]).data, np.log(4.5))
