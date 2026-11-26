import numpy as np
import pytest

from examples.language import EXPERIMENTS
from src.tokenizer.bpe import ByteBPE, merge_pair


def test_bpe():
    assert EXPERIMENTS[17]() == (((97, 98), (258, 258)), [259], 280)
    assert merge_pair([97, 97, 97], (97, 97), 258) == [258, 97]
    empty = ByteBPE.train([], 258)
    assert empty.encode("", boundaries=True) == [256, 257]
    assert empty.decode(empty.encode("é中🐍")) == "é中🐍"
    with pytest.raises(ValueError):
        empty.decode([-1])


def test_embedding_and_training():
    rows, gradient = EXPERIMENTS[18]()
    np.testing.assert_array_equal(rows, [[3, 4], [3, 4], [1, 2]])
    np.testing.assert_array_equal(gradient, [[4, 5], [3, 4], [0, 0]])
    counts, losses, similarities = EXPERIMENTS[19]()
    assert counts[0, 2] == counts[1, 2] == counts[2, 0] == 3
    assert losses[1] < losses[0] * 0.85
    assert similarities[0, 1] > similarities[0, 3]
    one, two, ppl, initial, final, sample = EXPERIMENTS[20]()
    np.testing.assert_allclose(one, [1/3, 2/3])
    np.testing.assert_allclose(two, [[0.2, 0.8], [2/3, 1/3]])
    assert ppl == pytest.approx((0.8 ** 2 * (2 / 3)) ** (-1/3))
    assert final < initial * 0.05
    assert sample == [0, 1, 0, 1, 0]


def test_sinusoidal_relative_dot():
    zero, one, a, b = EXPERIMENTS[21]()
    np.testing.assert_array_equal(zero, [0, 1, 0, 1])
    np.testing.assert_allclose(one, [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)])
    assert a == pytest.approx(b)
