import numpy as np
import pytest

from examples import loss_lab, norm_lab
from src.autograd.tensor import Tensor


def test_masked_ce_hand_calculation():
    result = loss_lab.hand_case()
    assert result["valid_counts"] == [1, 2]
    assert result["mean_nll"] == pytest.approx(np.log(9) / 3)
    np.testing.assert_allclose(result["batch_means"], [np.log(3), np.log(3) / 2], atol=1e-14)
    assert result["unweighted_mean"] == pytest.approx(3 * np.log(3) / 4)
    expected = np.array([[[2, -8, 6], [0, 0, 0]], [[3, 3, -6], [2, -4, 2]]]) / 36
    np.testing.assert_allclose(result["gradients"], expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("shape,seed", [((2, 3, 5), 42), ((3, 2, 7), 19), ((1, 1, 1), 8)])
def test_masked_ce_with_replaced_inputs(shape, seed):
    result = loss_lab.check(*shape, seed)
    assert tuple(result["shape"]) == shape
    assert result["difference_error"] < 1e-8
    assert result["translation_error"] < 2e-12
    assert result["max_row_gradient_sum"] < 1e-14


def test_masked_ce_rejects_no_targets_and_wrong_mask():
    x = Tensor(np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="at least one valid"):
        loss_lab.masked_loss(x, [[-1, -1]], [[False, False]])
    with pytest.raises(ValueError, match="boolean"):
        loss_lab.masked_loss(x, [[1, 2]], [[0, 1]])
    with pytest.raises(ValueError, match="shape|mask"):
        loss_lab.masked_loss(x, [[1, 2]], [True, True])


def test_layer_norm_hand_gain_bias_and_input():
    hand = norm_lab.hand_case()
    np.testing.assert_array_equal(hand["y"], [[[0, 0]]])
    np.testing.assert_array_equal(hand["dx"], [[[15 / 16, -15 / 16]]])
    np.testing.assert_array_equal(hand["dgain"], [-1 / 2, 3 / 2])
    np.testing.assert_array_equal(hand["dbias"], [1, 3])


@pytest.mark.parametrize("shape,epsilon,seed", [((2, 3, 4), 0.01, 42),
                                              ((1, 4, 3), 0.05, 19),
                                              ((2, 2, 1), 0.03, 8)])
def test_layer_norm_with_replaced_inputs(shape, epsilon, seed):
    result = norm_lab.check(*shape, epsilon, seed)
    assert tuple(result["shape"]) == shape
    assert max(result["finite_diff_max_errors"].values()) < 1e-8
    assert max(result["autograd_max_errors"].values()) < 1e-12
    assert result["max_input_gradient_row_sum"] < 1e-12
