import numpy as np

from examples.gradient_labs import direction, shared_affine, table_seed


def test_ch07_direction_and_roundoff():
    result = direction()
    np.testing.assert_array_equal(result["gradient"], [-1, 0.5])
    np.testing.assert_allclose([result["directional"], result["difference"]], -0.2,
                               rtol=0, atol=1e-10)
    large, middle, tiny = result["absolute_errors"]
    assert large > 1e-6 and middle < 1e-9 and tiny > middle


def test_ch08_batched_affine_and_two_weight_paths():
    result = shared_affine()
    np.testing.assert_array_equal(result["output"], [[[3, 4]], [[7, 8]]])
    np.testing.assert_array_equal(result["dx"], [[[2, 0]], [[-4, 6]]])
    np.testing.assert_array_equal(result["dw"], [[-1, 5.5], [-2, 9.5]])
    np.testing.assert_array_equal(result["db"], [0, 2])
    assert result["weight_fd_error"] < 1e-8


def test_ch15_seeded_gather_and_shared_leaf():
    result = table_seed()
    np.testing.assert_array_equal(result["output"], [[6, 6, 2], [12, 12, 6]])
    np.testing.assert_array_equal(result["table"], [[6, 18], [6, 27], [0, 0]])
    np.testing.assert_array_equal(result["scale"], [12, 48])
    np.testing.assert_array_equal(result["second_table"], 2 * np.asarray(result["table"]))
    assert result["seeded_scalar"] == 168


def test_ch23_rectangular_scores_weights_and_values():
    from examples.attention_labs import rectangular

    result = rectangular()
    np.testing.assert_allclose(result["scores_in_log2"], [[0, 1, 2], [2, 1, 0]], atol=1e-14)
    np.testing.assert_allclose(result["weights"], np.array([[1, 2, 4], [4, 2, 1]]) / 7, atol=1e-14)
    np.testing.assert_allclose(result["output"], np.array([[13, 8], [7, 5]]) / 7, atol=1e-14)


def test_ch24_causal_and_padding_are_independent():
    from examples.attention_labs import padded

    result = padded()
    np.testing.assert_array_equal(result["mask"], [[1, 0, 0], [1, 0, 1]])
    np.testing.assert_allclose(result["weights"], [[1, 0, 0], [0.8, 0, 0.2]])
    np.testing.assert_allclose(result["output"], [[1, 0], [1.4, 0.2]])
    np.testing.assert_allclose(result["value_grad"], [[1.8, 1.8], [0, 0], [0.2, 0.2]])
    np.testing.assert_array_equal(result["changed_pad"], result["output"])
    np.testing.assert_allclose(result["changed_future"], [[1, 0], [3.4, 2.2]])


def test_ch25_hand_gradient_and_projected_two_head_backward():
    from examples.attention_labs import attention_backward

    result = attention_backward()
    scores = np.array([[2, -38, 36], [-20, -10, 30]]) / 49
    np.testing.assert_allclose(result["ds"], scores, atol=1e-14)
    np.testing.assert_allclose(result["dq"], np.array([[34, -34], [50, -50]]) / (49 * np.sqrt(2)))
    np.testing.assert_allclose(result["dk"], np.log(2) * scores.T)
    np.testing.assert_allclose(result["dv"], np.array([[9, 3], [6, 0], [6, -3]]) / 7, atol=1e-14)
    assert max(result["output_error"], result["input_error"], result["weight_error"]) < 1e-12
    np.testing.assert_allclose(result["projection_gradient"], -0.12471628, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["projection_difference"], result["projection_gradient"], atol=1e-8, rtol=0)


def test_ch26_gelu_tanh_derivative_and_ffn():
    from examples.attention_labs import ffn_backward
    from examples.transformer import ffn

    result = ffn_backward()
    np.testing.assert_allclose(result["slopes"], [-0.08296408, 0.5, 1.08296408], atol=1e-8)
    np.testing.assert_allclose(result["dz"], [1.08296408, -0.08296408, -0.16592817], atol=1e-8)
    np.testing.assert_allclose(result["dx"], [1.24889225, -0.08296408], atol=1e-8)
    np.testing.assert_allclose(result["dx"], ffn()[-1][0, 0], atol=1e-14)
    assert result["autograd_error"] < 1e-12 and result["difference_error"] < 1e-9


def test_ch27_layer_norm_boundary_and_complete_block_difference():
    from examples.attention_labs import norm_boundary

    result = norm_boundary()
    constant, variable = result["rows"]
    np.testing.assert_array_equal(constant["y"], [0, 0])
    np.testing.assert_array_equal(constant["dx"], [2, -2])
    np.testing.assert_array_equal(constant["gain"], [0, 0])
    np.testing.assert_array_equal(constant["bias"], [1, -1])
    np.testing.assert_allclose(variable["dx"], np.array([1, -1]) * 2 / (5 * np.sqrt(5)))
    np.testing.assert_allclose(result["block_dx"], 0.19708119, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["block_difference"], result["block_dx"], atol=1e-8, rtol=0)


def test_ch28_tied_input_and_head_gradients_recombine():
    from examples.attention_labs import tied_paths

    result = tied_paths()
    assert result["logits_error"] < 1e-14 and result["tied_gradient_error"] < 1e-12
    np.testing.assert_allclose([result["row1_input"], result["row1_head"], result["row1_tied"]],
                               [-0.03089878, 0.32966901, 0.29877023], atol=1e-8, rtol=0)
    np.testing.assert_allclose([result["row7_input"], result["row7_head"]],
                               [0, -0.07082213], atol=1e-8, rtol=0)


def test_ch29_windows_tail_coverage_and_exact_overlap():
    from examples.training_labs import window_audit

    result = window_audit()
    np.testing.assert_array_equal(result["toy_inputs"], [[0, 3], [3, 4], [4, 5], [5, 6]])
    np.testing.assert_array_equal(result["toy_targets"], [[3, 4], [4, 5], [5, 6], [6, 1]])
    np.testing.assert_array_equal(result["toy_validation"], [[3, 4], [5, 6]])
    assert result["train_windows"] == 7504 and result["vocab"] == 124
    assert result["validation_windows"] == 72 and result["validation_windows_in_train"] == 56
    assert result["validation_total_targets"] == 1192
    assert result["validation_evaluated"] == 1152 and result["validation_tail"] == 40


def test_ch30_clipping_target_offset_and_resume():
    from examples.training_labs import training_audit

    result = training_audit()
    clipping = result["clipping"]
    assert clipping["raw_norm"] == 5
    np.testing.assert_allclose(clipping["m"], [0.06, 0.08], atol=1e-15)
    np.testing.assert_allclose(clipping["v"], [0.00036, 0.00064], atol=1e-15)
    np.testing.assert_allclose(clipping["updated"], [1.86, -1.08], atol=2e-9, rtol=0)
    right, wrong = (result["targets"][key] for key in ("next_token", "copy_input"))
    np.testing.assert_allclose([right["initial"], right["next_token_nll"], wrong["own_target_nll"],
                                wrong["next_token_nll"]],
                               [2.090428, 0.015276, 0.012340, 9.617911], atol=1e-6, rtol=0)
    assert right["own_target_nll"] == right["next_token_nll"]
    np.testing.assert_array_equal(right["predictions"], [[1, 2, 3, 0], [5, 6, 7, 4]])
    np.testing.assert_array_equal(wrong["predictions"], [[0, 1, 2, 3], [4, 5, 6, 7]])
    resume = result["resume"]
    assert all(resume[key] for key in ("same_weights", "same_moments", "same_rng"))
    assert resume["first_restored_batch"] == [5521, 5711, 5383, 5898]
    assert resume["fresh_seed_batch"] == [669, 5807, 4911, 3293]
    groups = result["validation_groups"]
    assert (groups["seen_tokens"], groups["unseen_tokens"]) == (896, 256)
    combined = (896 * groups["seen_nll"] + 256 * groups["unseen_nll"]) / 1152
    np.testing.assert_allclose(groups["overall_nll"], combined, atol=1e-14, rtol=0)
    np.testing.assert_allclose([groups["seen_nll"], groups["unseen_nll"], groups["overall_nll"]],
                               [1.283947, 2.393287, 1.530467], atol=1e-6, rtol=0)
