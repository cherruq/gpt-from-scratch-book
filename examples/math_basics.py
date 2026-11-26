"""M0 experiments. Run one with python -m examples.math_basics 2..11."""

import argparse
import math

import numpy as np

from src.autograd.tensor import Tensor
from src.numpy_gpt.model import cross_entropy, gelu, softmax
from src.numpy_gpt.optim import AdamW


# region layout
def layout():
    x = np.arange(32, dtype=np.int64).reshape(2, 4, 4)
    bias = np.array([10, 20, 30, 40])
    heads = x.reshape(2, 4, 2, 2).transpose(0, 2, 1, 3)
    return x.strides, heads.strides, (x + bias)[1, 2].tolist(), heads[1, 0, 2].tolist()
# endregion layout


# region span
def span():
    basis = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    vector = np.array([2., -1.]) @ basis
    return vector, np.linalg.matrix_rank(basis), np.linalg.norm(vector)
# endregion span


# region dot
def dot():
    q, k = np.array([2., 1.]), np.array([1., 2.])
    score = q @ k
    cosine = score / (np.linalg.norm(q) * np.linalg.norm(k))
    return score, cosine, score / (k @ k) * k, np.linalg.norm(q - k)
# endregion dot


# region matrix
def matrix():
    x = np.array([[1, 2], [3, 4]])
    weights = np.array([[2, 0, -1], [1, 3, 2]])
    return x @ weights, np.linalg.matrix_rank(weights)
# endregion matrix


# region activate
def activate():
    x = np.array([-1., 0., 1.])
    sigmoid = 1 / (1 + np.exp(-x))
    return np.maximum(x, 0), sigmoid, gelu(Tensor(x)).data, x * sigmoid
# endregion activate


# region difference
def central_difference(function, value, step=1e-5):
    return (function(value + step) - function(value - step)) / (2 * step)


def difference():
    def loss(weight):
        return np.logaddexp(0, -2 * weight)

    return loss(0), central_difference(loss, 0), -1.0
# endregion difference


# region vjp
def vjp():
    x = np.array([[1., 2.], [3., 4.]])
    w = np.array([[2., -1.], [0., 3.]])
    g = np.array([[1., 0.], [-1., 2.]])
    return x @ w, g @ w.T, x.T @ g, g.sum(axis=0)
# endregion vjp


# region probability
def probability():
    first = np.array([0.5, 0.5])
    conditional = np.array([[0.25, 0.75], [0.9, 0.1]])
    joint = first[:, None] * conditional
    probability_summer = joint[:, 1].sum()
    return joint, probability_summer, probability_summer * (1 - probability_summer)
# endregion probability


# region information
def information():
    logits = Tensor([[0, np.log(2), np.log(3)]], requires_grad=True)
    p = softmax(logits).data[0]
    loss = cross_entropy(logits, [1])
    loss.backward()
    entropy = -p @ np.log(p)
    return p, entropy, float(loss.data), math.exp(loss.data), logits.grad[0]
# endregion information


# region optimization
def optimization():
    weight = Tensor([2.0], True)
    weight.grad = np.array([4.0])
    optimizer = AdamW({"w": weight}, lr=0.1, decay=0.2, max_norm=None)
    optimizer.step()
    trajectories = {
        rate: [2.0 * (1 - 2 * rate) ** n for n in range(5)]
        for rate in (0.1, 1.0, 1.1)
    }
    return float(weight.data[0]), trajectories
# endregion optimization


EXPERIMENTS = {
    2: layout, 3: span, 4: dot, 5: matrix, 6: activate, 7: difference,
    8: vjp, 9: probability, 10: information, 11: optimization,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=EXPERIMENTS)
    args = parser.parse_args()
    print(EXPERIMENTS[args.chapter]())
