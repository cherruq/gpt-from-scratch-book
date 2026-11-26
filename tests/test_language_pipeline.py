import numpy as np
import pytest

from examples.language_pipeline import experiment, repeated_negative
from src.autograd.tensor import Tensor
from src.numpy_gpt.model import attention, causal_mask


@pytest.fixture(scope="module")
def default():
    return experiment(["春春", "春"], ["春春春", ""])


def test_boundary_local_cooccurrence_and_noise(default):
    assert default["train_ids"] == [[256, 259, 259, 257], [256, 259, 257]]
    assert default["vocab"] == 260
    context = default["context"]
    assert context["directed_pairs"] == 10
    assert context["counts"] == [
        [256, 259, 2], [257, 259, 2], [259, 256, 2], [259, 257, 2], [259, 259, 2],
    ]
    raw = np.array([2, 2, 6]) ** 0.75
    np.testing.assert_allclose([p for _, p in context["noise"]], raw / raw.sum(), atol=1e-15, rtol=0)
    assert (context["initial_loss"], context["final_loss"]) == pytest.approx(
        (0.688452, 0.452672), abs=5e-7, rel=0,
    )


def test_wider_context_counts_same_id_in_distinct_positions():
    bigger = experiment(["春春", "春"], ["春春春"], radius=2)
    assert bigger["context"]["directed_pairs"] == 16
    assert [256, 257, 1] in bigger["context"]["counts"]
    assert [257, 256, 1] in bigger["context"]["counts"]
    assert [259, 259, 2] in bigger["context"]["counts"]


def test_positive_and_duplicate_negative_scatter_gradient():
    report = repeated_negative()
    assert report["loss"] == pytest.approx(np.log(2))
    assert report["score_grad"] == [-1/6, 1/6, 1/6]
    np.testing.assert_array_equal(report["center_grad"], [0, -1/6])
    np.testing.assert_array_equal(report["table_grad"], [[0, 0], [1/6, 0]])
    assert report["difference_max_error"] < 1e-10


def test_bigram_held_out_smoothing_and_weighted_nll(default):
    report = default["bigram"]
    assert report["counts"] == [[256, 259, 2], [259, 257, 2], [259, 259, 1]]
    assert report["targets"] == 5
    assert report["unseen_transitions"] == 1 and report["unseen_contexts"] == 0
    assert report["mle_nll"] is None
    probabilities = np.array([201/460, 101/560, 101/560, 201/560, 1/460])
    np.testing.assert_allclose([row["p"] for row in report["held_out"]], probabilities,
                               atol=1e-15, rtol=0)
    assert report["bigram_nll"] == pytest.approx(-np.log(probabilities).mean(), abs=1e-14, rel=0)
    assert report["bigram_ppl"] == pytest.approx(np.prod(probabilities) ** (-1/5), abs=1e-13, rel=0)
    assert report["unigram_nll"] == pytest.approx(-(3*np.log(301/760) + 2*np.log(201/760))/5)


def test_mle_with_no_unseen_evaluation_pair():
    result = experiment(["春春", "春"], ["春春春"])["bigram"]
    assert result["unseen_transitions"] == 0
    assert result["mle_nll"] == pytest.approx(-np.log((1/3)**2 * (2/3)) / 4)


def test_new_records_and_no_merges_keep_byte_fallback():
    result = experiment(["ab", "ba"], ["aa", "🐍"], merges=0, radius=2, alpha=0.1, seed=7)
    assert result["vocab"] == 258
    assert result["held_out_ids"][1] == [256, 240, 159, 144, 141, 257]
    assert result["bigram"]["unseen_contexts"] == 4
    assert result["bigram"]["unseen_transitions"] == 6
    np.testing.assert_allclose([row["p"] for row in result["bigram"]["held_out"][-4:]],
                               1/258, atol=1e-15, rtol=0)
    assert result["positions"]["repeat"] is None


def test_position_table_and_fixed_slot_control(default):
    report = default["positions"]
    assert report["repeat"]["id"] == 259 and report["repeat"]["slots"] == [1, 2]
    assert report["repeat"]["token_delta"] == [0, 0, 0, 0]
    np.testing.assert_allclose(report["repeat"]["embedding_delta"],
                               [-0.003866, 0.025934, -0.014082, -0.014933], atol=5e-7, rtol=0)
    np.testing.assert_allclose([row["output"] for row in report["control"]],
                               [[0, 0.5], [0, 0.5], [0, 0.75], [0, 0.25]], atol=1e-15)
    np.testing.assert_allclose([row["weights"] for row in report["control"]],
                               [[0.5, 0.5], [0.5, 0.5], [0.25, 0.75], [0.25, 0.75]], atol=1e-15)
    np.testing.assert_allclose(report["sinusoidal_one_step"], np.cos(1) + np.cos(0.01), atol=1e-15)


def test_fixed_causal_mask_changes_which_slots_can_contribute():
    qk = Tensor(np.zeros((1, 1, 2, 1)))
    v = np.array([[[[0.0], [1.0]]]])
    permutation = [1, 0]
    np.testing.assert_array_equal(attention(qk, qk, Tensor(v)).data.ravel(), [0.5, 0.5])
    mask = causal_mask(np.arange(2))
    baseline = attention(qk, qk, Tensor(v), mask).data.ravel()
    permuted = attention(qk, qk, Tensor(v[:, :, permutation]), mask).data.ravel()
    np.testing.assert_array_equal(baseline, [0, 0.5])
    np.testing.assert_array_equal(permuted, [1, 0.5])
    assert not np.array_equal(baseline[permutation], permuted)
