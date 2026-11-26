"""M1–M2 experiments. Run python -m examples.networks 12..16."""

import argparse

import numpy as np

from examples.math_basics import central_difference
from src.autograd.scalar import Value
from src.autograd.tensor import Tensor
from src.math_basics.mlp import MLP, xor_data, xor_train
from src.math_basics.normalization import norm_vjp
from src.numpy_gpt.model import layer_norm


# region neuron
def neuron():
    x, y = np.array([1., 2.]), np.array([2., 4.])
    w = float(np.random.default_rng(42).normal(0, 0.1))
    initial = np.mean((x * w - y) ** 2)
    for _ in range(30):
        residual = x * w - y
        w -= 0.1 * np.mean(residual * x)
    return w, float(initial), float(np.mean((x * w - y) ** 2))
# endregion neuron


# region backward_check
def backward_check():
    x, y = xor_data()
    model = MLP()
    _, hidden, residual = model.objective(x, y)
    analytical = model.backward(x, hidden, residual)[1][0, 0]
    original = model.up.weight[0, 0]

    def evaluate(value):
        model.up.weight[0, 0] = value
        return model.objective(x, y)[0]

    try:
        numerical = central_difference(evaluate, original)
    finally:
        model.up.weight[0, 0] = original
    return float(analytical), float(numerical)
# endregion backward_check


# region tensor_dag
def tensor_dag():
    scalar = Value(2)
    result = scalar * scalar + scalar
    result.backward()
    x = Tensor([[1., 2.], [3., 4.]], True)
    bias = Tensor([2., 3.], True)
    loss = (x * bias + x).sum()
    loss.backward()
    return result.data, scalar.grad, float(loss.data), x.grad, bias.grad
# endregion tensor_dag


# region norm_check
def norm_check():
    x = Tensor([[1., 3.]], True)
    gain = Tensor([1., 1.], True)
    bias = Tensor([0., 0.], True)
    upstream = np.array([[1., 2.]])
    y = layer_norm(x, gain, bias, eps=0.25)
    (y * upstream).sum().backward()
    manual = norm_vjp(x.data, gain.data, upstream, eps=0.25)
    for actual, expected in zip((x.grad, gain.grad, bias.grad), manual, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    return y.data, manual
# endregion norm_check


EXPERIMENTS = {12: neuron, 13: xor_train, 14: backward_check, 15: tensor_dag, 16: norm_check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", choices=EXPERIMENTS, type=int)
    args = parser.parse_args()
    print(EXPERIMENTS[args.chapter]())
