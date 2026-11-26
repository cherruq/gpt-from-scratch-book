import numpy as np
import pytest

from examples.modern import cached, efficiency, modern_parts, post_training
from src.autograd.tensor import Tensor
from src.modern.architecture import grouped_attention, mixture, rotary
from src.modern.efficiency import cache_bytes, online_attention
from src.numpy_gpt.generate import generate
from src.numpy_gpt.model import Config, GPT, attention, causal_mask


def test_cache_equivalence_and_window_rebuild():
    data = cached()
    assert data["logits_max_abs"] < 1e-12
    assert data["cache_bytes"] == cache_bytes(2, 2, 2, 4, 2, 8) == 1024
    model = GPT(Config(vocab=8))
    seed = [0, 1, 2, 3, 4]
    for prompt in ([0], seed):
        settings = {"seed": 29, "temperature": 0.8, "top_k": 5, "top_p": 0.9}
        assert generate(model, prompt, 12, **settings) == generate(model, prompt, 12, use_cache=True, **settings)
    _, prefix = model.forward([[0]], track=False, use_cache=True)
    extension, _ = model.forward([[1, 2]], track=False, use_cache=True, cache=prefix)
    np.testing.assert_allclose(extension.data, model.forward([[0, 1, 2]], track=False).data[:, 1:], atol=1e-12)


def test_rope_relative_rotation_and_vjp():
    q = Tensor(np.array([1., 0., 0.5, 0.25]).reshape(1, 1, 1, 4), True)
    k = Tensor(np.array([0., 1., 0.2, -0.3]).reshape(1, 1, 1, 4))
    a = rotary(q, [2])
    b = rotary(k, [6])
    np.testing.assert_allclose((a * b).sum().data,
                               (rotary(q, [11]) * rotary(k, [15])).sum().data, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(a.data), np.linalg.norm(q.data), atol=1e-14)
    (a * b).sum().backward()
    np.testing.assert_allclose(q.grad, rotary(k, [4]).data, atol=1e-14)


def test_gqa_and_router():
    rng = np.random.default_rng(42)
    q = Tensor(rng.normal(size=(1, 4, 3, 2)))
    k, v = [Tensor(rng.normal(size=(1, 2, 3, 2))) for _ in range(2)]
    mask = causal_mask([0, 1, 2])
    expected = attention(q, Tensor(k.data.repeat(2, axis=1)), Tensor(v.data.repeat(2, axis=1)), mask)
    np.testing.assert_allclose(grouped_attention(q, k, v, mask).data, expected.data, atol=1e-14)
    info = modern_parts()
    assert info["block_shape"] == (1, 2, 4) and info["block_input_gradient_finite"]
    assert info["moe"] == [[[2.0, 0.0]]] and info["load"] == [1.0, 0.0]
    data = Tensor([[[1.0, 0], [0, 1.0]]])
    identity = Tensor(np.eye(2), True)
    result, auxiliary, load = mixture(data, identity, [identity, 2*identity], 2)
    assert auxiliary.data == pytest.approx(1)
    np.testing.assert_allclose(load, [0.5, 0.5])
    result.sum().backward()
    assert np.isfinite(identity.grad).all()


def test_efficiency_and_reward():
    result = efficiency()
    assert result["online_error"] < 1e-12
    assert result["quantization_error"] <= 0.5 / 127
    np.testing.assert_allclose(result["decode_frequency"], [0.6, 0.3, 0.1], atol=0.025, rtol=0)
    with pytest.raises(ValueError, match="finite"):
        online_attention(np.array([-np.inf]), np.ones((1, 1)))
    policy = post_training()
    assert policy["answers"] == [1, 0]
    assert min(policy["probability_of_correct"]) > 0.95
    assert policy["dpo_policy_log_odds_1"] < policy["dpo_zero_margin"]
