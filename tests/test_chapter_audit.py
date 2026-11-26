"""Extra numerical assertions found during the chapter-by-chapter audit."""

import json
from pathlib import Path

import numpy as np
import pytest

from examples import language, math_basics, modern, networks, training, transformer
from src.autograd.tensor import Tensor
from src.math_basics.mlp import MLP, xor_data
from src.modern.architecture import rms_norm, rotary, swiglu
from src.modern.efficiency import cache_bytes, forward_flops, quantize_int8
from src.modern.post_training import parity_policy
from src.numpy_gpt.generate import distribution
from src.numpy_gpt.model import Config, GPT, cross_entropy
from src.numpy_gpt.optim import AdamW
from src.numpy_gpt.train import read_config
from src.tokenizer.bpe import ByteBPE


def test_ch01_to_06_published_hand_values():
    np.testing.assert_allclose(distribution([0, 2, 1, -1]), [0.08714, 0.64391, 0.23688, 0.03206],
                               atol=5e-6, rtol=0)
    assert np.empty((2, 4, 8), dtype=np.float64).nbytes == 512
    x = np.arange(32, dtype=np.int64).reshape(2, 4, 4)
    address = x[1:2, 2:3, 3:4].ctypes.data - x.ctypes.data
    assert (address, x[1, 2, 3]) == (216, 27)
    assert x[:, ::2, :].strides == (128, 64, 8) and np.shares_memory(x, x[:, ::2, :])
    basis = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    u = np.array([3, 2]) @ basis
    assert (np.linalg.norm(u, 1), np.linalg.norm(u, np.inf)) == (10, 3)
    _, _, projection, _ = math_basics.dot()
    np.testing.assert_allclose((np.array([2., 1.]) - projection) @ np.array([1., 2.]), 0, atol=1e-14)
    matrix, _ = math_basics.matrix()
    np.testing.assert_array_equal(matrix + [1, 0, -2], [[5, 6, 1], [11, 12, 3]])
    np.testing.assert_array_equal(np.array([[3, 2]]) @ [[2, 0, -1], [1, 3, 2]], [[8, 6, 1]])
    assert math_basics.activate()[-1][1] == 0


def test_ch07_to_11_published_hand_values():
    def ratio(a, b):
        return a * b / (a*a + b*b) if a or b else 0

    assert ratio(1e-4, 0) == ratio(0, 1e-4) == 0
    assert ratio(1e-4, 1e-4) == 0.5
    xy = Tensor([[2., 3.]], True)
    (xy @ Tensor([[2, 4], [1, -3]])).backward([[1., 2.]])
    np.testing.assert_array_equal(xy.grad, [[10, -5]])
    joint, p, _ = math_basics.probability()
    assert joint[0, 1] / p == pytest.approx(15/17)
    extreme = cross_entropy(Tensor([[1000., 999., -1000.]]), [2]).data.item()
    assert extreme == pytest.approx(2000.31326, abs=5e-6, rel=0)
    w = Tensor([2.], True)
    w.grad = np.array([4.])
    AdamW({"w": w}, lr=0.1, decay=0, max_norm=None).step()
    np.testing.assert_allclose(w.data, [1.9], atol=1e-9, rtol=0)


def test_ch12_to_16_published_hand_values():
    x = np.array([1., 2.])
    y = 2 * x
    weights = [0.]
    for _ in range(2):
        weights.append(weights[-1] - 0.1 * np.mean((x * weights[-1] - y) * x))
    np.testing.assert_array_equal(weights, [0, 0.5, 0.875])
    w, initial, final = networks.neuron()
    np.testing.assert_allclose([w, initial], [1.999648, 9.697604], atol=5e-7, rtol=0)
    assert final == pytest.approx(3.09e-7, abs=5e-10, rel=0)
    xor_points, expected = xor_data()
    pre = xor_points @ np.ones((2, 2)) + [0, -1]
    np.testing.assert_array_equal(np.maximum(pre, 0) @ [[1], [-2]], expected)
    tiny = MLP(hidden=1)
    tiny.up.weight.fill(0)
    tiny.down.weight.fill(3)
    x = np.array([[2., 0.]])
    _, h, r = tiny.objective(x, [[1.]])
    _, dw1, db1, dw2, db2 = tiny.backward(x, h, r)
    np.testing.assert_array_equal([dw1[0, 0], db1[0], dw2[0, 0], db2[0]], [-12, -6, 0, -2])
    z = Tensor([[-1., 2.]], True)
    z.exp().log().sum().backward()
    np.testing.assert_allclose(z.grad, [[1, 1]], atol=1e-14)
    out, (dx, _, _) = networks.norm_check()
    np.testing.assert_allclose(out, [[-0.894427, 0.894427]], atol=5e-7, rtol=0)
    np.testing.assert_allclose(dx, [[-0.089443, 0.089443]], atol=5e-7, rtol=0)


def test_ch17_to_22_published_hand_values():
    byte = ByteBPE.train([], 258)
    assert len(byte.encode("春")) == 3
    whitespace = "春\r\n\t风  "
    assert byte.decode(byte.encode(whitespace)) == whitespace
    e = Tensor([[1, 2], [3, 4], [5, 6]], True)
    (e[[1, 0, 0]] * [[1, 1], [2, 3], [4, 5]]).sum().backward()
    np.testing.assert_array_equal(e.grad, [[6, 8], [1, 1], [0, 0]])
    central = Tensor([[1, 0]], True)
    scores = central @ Tensor([[0, 0], [1, -1]])
    # At score zero, the gradient of binary logistic loss is 0.5 − target.
    scores.backward([[-0.25, 0.25]])
    np.testing.assert_array_equal(central.grad, [[0, -0.5]])
    _, losses, similarities = language.word2vec()
    np.testing.assert_allclose(losses, [0.69702, 0.32296], atol=5e-6, rtol=0)
    np.testing.assert_allclose(similarities[0, [1, 3]], [0.89308, 0.32674], atol=5e-6, rtol=0)
    _, _, _, initial, final, _ = language.counts_model()
    np.testing.assert_allclose([initial, final], [0.684873, 0.000202], atol=5e-7, rtol=0)
    _, _, a, b = language.position_demo()
    np.testing.assert_allclose([a, b], [1.540252, 1.540252], atol=5e-7, rtol=0)
    fixed, score = transformer.retrieve()
    np.testing.assert_array_equal(fixed, [[0.5, 3]])
    np.testing.assert_allclose(score, [[0.707107, 0]], atol=5e-7, rtol=0)


def test_ch23_to_30_supplementary_hand_values():
    from src.numpy_gpt.model import attention_weights

    q = Tensor([[[[1, 0], [2, 0]]]])
    k = Tensor(np.eye(2)[None, None])
    np.testing.assert_allclose(attention_weights(q, k).data[0, 0, :, 0], [0.66976, 0.80443],
                               atol=5e-6, rtol=0)
    _, changes = transformer.causal()
    np.testing.assert_allclose(changes, [[0, 0, 0.044690, 0.041261]], atol=5e-7, rtol=0)
    gate = Tensor([[0.]], True)
    up = Tensor([[4.]], True)
    h = swiglu(Tensor([[1.]]), gate, up, Tensor([[1.]]))
    h.sum().backward()
    np.testing.assert_array_equal([h.data.item(), gate.grad.item(), up.grad.item()], [0, 2, 0])
    loss = cross_entropy(Tensor(np.log([[0.25, 0.75], [0.5, 0.5]])), [0, 0]).data.item()
    assert loss == pytest.approx(1.03972, abs=5e-6, rel=0)
    assert np.exp(loss) == pytest.approx(2.82843, abs=5e-6, rel=0)


def test_ch31_to_32_published_hand_values():
    original = np.array([0.5, 0.3, 0.2])
    logits = np.log(original)
    np.testing.assert_allclose(distribution(logits, top_p=0.85), original)
    np.testing.assert_allclose(distribution(logits, top_p=0.79), [0.625, 0.375, 0])
    assert np.exp(np.average([1., 3.], weights=[3, 1])) == pytest.approx(4.48169, abs=5e-6, rel=0)
    gradient = training.gradient_report()
    np.testing.assert_allclose([gradient["nll_before"], gradient["nll_after"]],
                               [2.078860, 2.038937], atol=5e-7, rtol=0)


def test_ch33_two_rows_average_and_recombine():
    torch = pytest.importorskip("torch")
    initial = np.tile([[0, np.log(2)]], (2, 1))
    numpy = Tensor(initial, True)
    loss = cross_entropy(numpy, [1, 1])
    loss.backward()
    use_torch = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    torch_loss = torch.nn.functional.cross_entropy(use_torch, torch.tensor([1, 1]))
    torch_loss.backward()
    np.testing.assert_allclose([loss.data, torch_loss.item()], np.log(1.5), atol=1e-14, rtol=0)
    np.testing.assert_allclose(numpy.grad.sum(axis=0), [1/3, -1/3], atol=1e-14)
    np.testing.assert_allclose(use_torch.grad.numpy(), numpy.grad, atol=1e-14)


def test_ch34_to_36_memory_and_architecture_constants():
    quick, large = (read_config(f"configs/{name}.json") for name in ("quick", "larger"))
    n = []
    for config in (quick, large):
        corpus = config.corpus()
        assert corpus.tokenizer.vocab == 124
        n.append(GPT(config.model_config(corpus.tokenizer.vocab)).parameter_count())
    assert n == [7776, 825600] and np.empty(n[1], dtype="float32").nbytes == 3302400
    assert large.context * large.batch_size == 512 and 512 * large.steps == 76800
    assert cache_bytes(2, 2, 2, 128, 2, 8) == 32768
    assert cache_bytes(2, 2, 2, 4, 2, 4) == 512
    unit = Tensor([[[[1, 0]]]])
    values = [(rotary(unit, [p]) * rotary(unit, [p+1])).sum().data for p in (0, 6)]
    np.testing.assert_allclose(values, np.cos(1), atol=1e-14, rtol=0)
    np.testing.assert_allclose(rms_norm(Tensor([[1, 3]]), Tensor([1, 1]), eps=0).data,
                               np.array([[1, 3]]) / np.sqrt(5))
    assert modern.modern_parts()["router_auxiliary"] == pytest.approx(1.761594, abs=5e-7, rel=0)


def test_ch37_to_38_efficiency_and_reward_constants():
    codes, scale = quantize_int8(np.array([[1.], [-0.4]]))
    np.testing.assert_array_equal(codes, [[127], [-51]])
    assert (codes * scale)[1, 0] == pytest.approx(-0.401575, abs=5e-7, rel=0)
    assert forward_flops(1, 1, 4, 16, 2, 8) == 864
    result = modern.efficiency()
    np.testing.assert_allclose(result["decode_frequency"], [0.5926, 0.3086, 0.0988], atol=1e-12)
    np.testing.assert_array_equal(parity_policy(steps=0)[1], [0.5, 0.5])
    result = modern.post_training()
    np.testing.assert_allclose(result["probability_of_correct"], [0.98786, 0.98961], atol=5e-6, rtol=0)
    assert result["dpo_policy_log_odds_1"] == pytest.approx(0.644397, abs=5e-7, rel=0)


def test_ch39_to_41_probe_and_graduation_budget():
    z = np.array([[1.], [-1.]])
    w = np.linalg.solve(z.T @ z + np.eye(1), z.T @ np.array([1., -1.]))
    np.testing.assert_allclose(z @ w, [2/3, -2/3], atol=1e-14)
    interpretation = json.loads(Path("reports/interpretability.json").read_text())
    assert interpretation["probe_accuracy"] * interpretation["probe_validation_count"] == 7
    np.testing.assert_allclose([interpretation[key] for key in ("original", "other_context", "ablation")],
                               [0.004619, 0.004871, 0.005134], atol=5e-7, rtol=0)
    np.testing.assert_allclose(interpretation["attention_row"][-1], 0.18718, atol=5e-6, rtol=0)
    logits = np.array([[1., 0.]])
    cache = np.array([[1.00000000001, 0.]])
    assert np.max(np.abs(logits - cache)) > 1e-12 and logits.argmax() == cache.argmax()
    model = GPT(Config(vocab=8))
    assert model.config.channels // model.config.heads == 2
    assert model.forward(np.zeros((2, 4), dtype=int), track=False).shape == (2, 4, 8)
    assert model.parameter_count() == 544
