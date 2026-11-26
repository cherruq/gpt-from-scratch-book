"""Worked forward and backward checks for chapters 23–28."""

import argparse
import json
from math import sqrt

import numpy as np

from examples.math_basics import central_difference
from src.autograd.tensor import Tensor
from src.math_basics.normalization import norm_vjp
from src.numpy_gpt.model import (
    Config, GPT, attention, attention_weights, causal_mask, cross_entropy, gelu,
    layer_norm, merge_heads, split_heads, transformer_block,
)


def rectangular_inputs():
    q = np.eye(2) * (sqrt(2) * np.log(2))
    k = np.array([[0., 2.], [1., 1.], [2., 0.]])
    v = np.array([[1., 0.], [0., 2.], [3., 1.]])
    return q, k, v


# region rectangular
def rectangular():
    q, k, v = [Tensor(a[None, None]) for a in rectangular_inputs()]
    scores = q.data[0, 0] @ k.data[0, 0].T / sqrt(2)
    return {"scores_in_log2": (scores / np.log(2)).tolist(),
            "weights": attention_weights(q, k).data[0, 0].tolist(),
            "output": attention(q, k, v).data[0, 0].tolist()}
# endregion rectangular


# region padded
def padded():
    q, k, v = [Tensor(a[None, None], True) for a in rectangular_inputs()]
    positions = np.array([0, 2])
    real_keys = np.array([True, False, True])
    mask = causal_mask(positions) & real_keys[None, :]
    output = attention(q, k, v, mask)
    output.sum().backward()
    moved_pad = v.data.copy()
    moved_pad[0, 0, 1] = [500, -500]
    moved_future = v.data.copy()
    moved_future[0, 0, 2] += 10
    return {"mask": mask.tolist(),
            "weights": attention_weights(q, k, mask).data[0, 0].tolist(),
            "output": output.data[0, 0].tolist(), "value_grad": v.grad[0, 0].tolist(),
            "changed_pad": attention(q, k, Tensor(moved_pad), mask).data[0, 0].tolist(),
            "changed_future": attention(q, k, Tensor(moved_future), mask).data[0, 0].tolist()}
# endregion padded


# region manual_attention_vjp
def manual_attention_vjp(q, k, v, upstream, mask=None):
    """Works on NumPy arrays [B,H,T,D], using the key axis for Softmax."""
    scale = 1 / sqrt(q.shape[-1])
    scores = (q @ k.swapaxes(-1, -2)) * scale
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    a = exp / exp.sum(axis=-1, keepdims=True)
    da = upstream @ v.swapaxes(-1, -2)
    dv = a.swapaxes(-1, -2) @ upstream
    ds = a * (da - (a * da).sum(axis=-1, keepdims=True))
    dq = (ds @ k) * scale
    dk = (ds.swapaxes(-1, -2) @ q) * scale
    return a @ v, (dq, dk, dv), ds
# endregion manual_attention_vjp


# region projected_mha
def projected_mha():
    rng = np.random.default_rng(42)
    x = Tensor(rng.normal(size=(2, 3, 4)), True)
    weights = [Tensor(rng.normal(scale=0.4, size=(4, 4)), True) for _ in range(4)]
    upstream = rng.normal(size=x.shape)
    mask = causal_mask(np.arange(3))

    def split(a):
        b, t, c = a.shape
        return a.reshape(b, t, 2, c // 2).swapaxes(1, 2)

    def merge(a):
        b, h, t, d = a.shape
        return a.swapaxes(1, 2).reshape(b, t, h * d)

    q, k, v = [split(x.data @ w.data) for w in weights[:3]]
    head_upstream = split(upstream @ weights[3].data.T)
    output, partials, _ = manual_attention_vjp(q, k, v, head_upstream, mask)
    input_parts = []
    matrix_grads = []
    for w, partial in zip(weights[:3], partials, strict=True):
        g = merge(partial)
        input_parts.append(g @ w.data.T)
        matrix_grads.append(x.data.reshape(-1, 4).T @ g.reshape(-1, 4))
    matrix_grads.append(merge(output).reshape(-1, 4).T @ upstream.reshape(-1, 4))

    def forward():
        qkv = [split_heads(x @ w, 2) for w in weights[:3]]
        return merge_heads(attention(*qkv, mask)) @ weights[3]

    y = forward()
    (y * upstream).sum().backward()
    manual_output = merge(output) @ weights[3].data
    manual_input = sum(input_parts)
    old = weights[0].data[1, 2]

    def perturbed(value):
        weights[0].data[1, 2] = value
        return (forward().data * upstream).sum()

    try:
        numeric = central_difference(perturbed, old, 1e-5)
    finally:
        weights[0].data[1, 2] = old
    return {
        "output_error": float(np.max(np.abs(manual_output - y.data))),
        "input_error": float(np.max(np.abs(manual_input - x.grad))),
        "weight_error": float(max(np.max(np.abs(g - w.grad))
                                  for g, w in zip(matrix_grads, weights, strict=True))),
        "projection_gradient": float(matrix_grads[0][1, 2]),
        "projection_difference": float(numeric),
    }
# endregion projected_mha


def attention_backward():
    q, k, v = [a[None, None] for a in rectangular_inputs()]
    upstream = np.array([[[[1., -1.], [2., 1.]]]])
    _, (dq, dk, dv), ds = manual_attention_vjp(q, k, v, upstream)
    return {"ds": ds[0, 0].tolist(), "dq": dq[0, 0].tolist(),
            "dk": dk[0, 0].tolist(), "dv": dv[0, 0].tolist(), **projected_mha()}


# region gelu_prime
def gelu_prime(z):
    factor = sqrt(2 / np.pi)
    cubic = 0.044715
    t = np.tanh(factor * (z + cubic * z**3))
    return (1 + t) / 2 + z * (1 - t**2) * factor * (1 + 3 * cubic * z**2) / 2


def ffn_backward():
    z = np.array([1., -1., -1.])
    w = np.array([[1., 0., -1.], [0., 1., 0.]])
    dgelu = gelu_prime(z)
    dz = np.array([1., 1., 2.]) * dgelu
    dx = dz @ w.T
    points = np.array([-1., 0., 1.])
    direct = Tensor(points, True)
    gelu(direct).sum().backward()
    numeric = central_difference(lambda a: gelu(Tensor(a)).data, points)
    return {"slopes": gelu_prime(points).tolist(), "dz": dz.tolist(), "dx": dx.tolist(),
            "autograd_error": float(np.max(np.abs(gelu_prime(points) - direct.grad))),
            "difference_error": float(np.max(np.abs(gelu_prime(points) - numeric)))}
# endregion gelu_prime


# region norm_boundary
def norm_boundary():
    upstream = np.array([[1., -1.]])
    cases = []
    for row in ([2., 2.], [1., 3.]):
        x = Tensor([row], True)
        gain = Tensor([1., 1.], True)
        bias = Tensor([0., 0.], True)
        y = layer_norm(x, gain, bias, eps=0.25)
        (y * upstream).sum().backward()
        expected = norm_vjp(x.data, gain.data, upstream, eps=0.25)
        for actual, analytic in zip((x.grad, gain.grad, bias.grad), expected, strict=True):
            np.testing.assert_allclose(actual, analytic, rtol=0, atol=1e-14)
        cases.append({"y": y.data[0].tolist(), "dx": x.grad[0].tolist(),
                      "gain": gain.grad.tolist(), "bias": bias.grad.tolist()})
    model = GPT(Config(vocab=8))
    x = Tensor(np.arange(12.).reshape(1, 3, 4) / 10, True)
    mask = causal_mask(np.arange(3))
    g = np.linspace(-0.4, 0.7, 12).reshape(x.shape)

    def objective(a):
        y, _ = transformer_block(a, model.parameters, "block.0", 2, mask)
        return (y * g).sum()

    objective(x).backward()
    unit = np.zeros_like(x.data)
    unit[0, 1, 2] = 1
    numeric = central_difference(lambda h: objective(Tensor(x.data + h * unit)).data, 0.)
    return {"rows": cases, "block_dx": float(x.grad[0, 1, 2]), "block_difference": float(numeric)}
# endregion norm_boundary


# region tied_paths
def tied_paths():
    model = GPT(Config(vocab=8))
    ids = np.array([[1, 1, 2]])
    targets = [[1, 2, 3]]
    tied_logits = model.forward(ids)
    loss = cross_entropy(tied_logits, targets)
    loss.backward()
    tied = model.parameters["token"]

    # Same values, with separate leaf variables for the two use sites.
    input_table = Tensor(tied.data.copy(), True)
    head_table = Tensor(tied.data.copy(), True)
    p = model.parameters
    pos = np.arange(ids.shape[1])
    x = input_table[ids] + p["position"][pos]
    for i in range(model.config.layers):
        x, _ = transformer_block(x, p, f"block.{i}", model.config.heads, causal_mask(pos))
    z = layer_norm(x, p["final.gain"], p["final.bias"])
    logits = z @ head_table.transpose(1, 0)
    cross_entropy(logits, targets).backward()
    return {"logits_error": float(np.max(np.abs(tied_logits.data - logits.data))),
            "tied_gradient_error": float(np.max(np.abs(tied.grad - input_table.grad - head_table.grad))),
            "row1_input": float(input_table.grad[1, 0]), "row1_head": float(head_table.grad[1, 0]),
            "row1_tied": float(tied.grad[1, 0]), "row7_input": float(input_table.grad[7, 0]),
            "row7_head": float(head_table.grad[7, 0])}
# endregion tied_paths


LABS = {23: rectangular, 24: padded, 25: attention_backward, 26: ffn_backward, 27: norm_boundary, 28: tied_paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=LABS)
    print(json.dumps(LABS[parser.parse_args().chapter](), indent=2))
