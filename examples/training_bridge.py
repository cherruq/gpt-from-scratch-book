"""M1. Audit gradients, moments and updates before building a GPT block."""

import argparse
import json

import numpy as np

from src.autograd.tensor import Tensor
from src.math_basics.mlp import MLP, xor_data, xor_train
from src.numpy_gpt.model import cross_entropy
from src.numpy_gpt.optim import AdamW


# region output_head_step
def output_head_step(rate):
    h = np.array([[1., 2.]])
    w = Tensor(np.zeros((2, 2)), requires_grad=True)

    def run(labels):
        features = np.repeat(h, len(labels), axis=0)
        loss = cross_entropy(Tensor(features) @ w, np.array(labels))
        w.grad = None
        loss.backward()
        return float(loss.data), w.grad.copy()

    initial, gradient = run([1])
    _, full = run([0, 1, 1])
    _, first = run([0])
    _, last = run([1, 1])
    w.grad = gradient.copy()
    optimizer = AdamW({"head": w}, lr=rate, decay=0., max_norm=1.)
    norm = optimizer.step()
    new_logits = h @ w.data
    return {"initial": initial, "gradient": gradient.tolist(),
            "microbatch_gradients": [first.tolist(), last.tolist()], "full_gradient": full.tolist(),
            "preclip_norm": norm, "m": optimizer.m["head"].tolist(), "v": optimizer.v["head"].tolist(),
            "weights": w.data.tolist(), "logits": new_logits.tolist(),
            "final": float(cross_entropy(Tensor(new_logits), np.array([1])).data)}
# endregion output_head_step


# region scalar_head
def scalar_head(rate):
    inputs = Tensor([[1.], [2.]])
    labels = np.array([0, 1])
    w = Tensor([[0.]], requires_grad=True)
    loss = cross_entropy((inputs @ w) * [0., 1.], labels)
    loss.backward()
    gradient = float(w.grad[0, 0])
    w.data -= rate * w.grad
    new_logits = (inputs @ w) * [0., 1.]
    return {"gradient": gradient, "weight": float(w.data[0, 0]), "initial": float(loss.data),
            "logits": new_logits.data.tolist(),
            "final": float(cross_entropy(new_logits, labels).data)}
# endregion scalar_head


def parameters(model):
    return (model.up.weight, model.up.bias, model.down.weight, model.down.bias)


# region symmetry
def xor_controls():
    x, y = xor_data()
    relu = np.maximum(x @ np.ones((2, 2)) + [0., -1.], 0.)
    designed = relu @ np.array([[1.], [-2.]])
    model = MLP()
    for parameter in parameters(model):
        parameter.fill(0.)
    for _ in range(800):
        _, hidden, residual = model.objective(x, y)
        model.step(model.backward(x, hidden, residual), .15)
    final, _, residual = model.objective(x, y)
    affine_inputs = np.column_stack((x, np.ones(len(x))))
    affine, _, _, _ = np.linalg.lstsq(affine_inputs, y, rcond=None)
    before, learned, _ = xor_train()
    return {"designed_output": designed[:, 0].tolist(),
            "zero_init_output": (residual + y)[:, 0].tolist(),
            "zero_init_final": float(final),
            "zero_init_nonbias_weights": int(np.count_nonzero(model.up.weight) +
                                             np.count_nonzero(model.down.weight)),
            "affine_mse": float(np.mean((affine_inputs @ affine - y) ** 2)),
            "random_init": [before, learned]}
# endregion symmetry


# region batch_backprop
def batch_backprop(rate):
    a = np.arctanh(.5)
    x, y = np.diag([1., 2.]), np.zeros((2, 1))
    model = MLP(hidden=2)
    model.up.weight[...] = [[0., a], [a/2, 0.]]
    model.down.weight[...] = [[2.], [-2.]]
    loss, hidden, residual = model.objective(x, y)
    gradients = model.backward(x, hidden, residual)
    errors = []
    step = 1e-5
    for array, gradient in zip((x, *parameters(model)), gradients, strict=True):
        for idx in np.ndindex(array.shape):
            saved = array[idx]
            array[idx] = saved + step
            plus = model.objective(x, y)[0]
            array[idx] = saved - step
            minus = model.objective(x, y)[0]
            array[idx] = saved
            errors.append(abs((plus - minus) / (2 * step) - gradient[idx]))
    model.step(gradients, rate)
    updated, _, next_residual = model.objective(x, y)
    return {"hidden": hidden.tolist(), "prediction": (residual + y).tolist(), "loss": float(loss),
            "gradients": dict(zip(("x", "w1", "b1", "w2", "b2"),
                                  (array.tolist() for array in gradients), strict=True)),
            "difference_max_error": float(max(errors)), "final": float(updated),
            "next_prediction": (next_residual + y).tolist()}
# endregion batch_backprop


def experiment(rate=0.1):
    if not np.isfinite(rate) or not 0 < rate <= 10:
        raise ValueError("rate must be positive and at most 10 for this small experiment")
    return {"rate": rate, "optimizer": output_head_step(rate), "neuron": scalar_head(rate),
            "xor": xor_controls(), "backprop": batch_backprop(rate)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=.1)
    args = parser.parse_args()
    try:
        result = experiment(args.rate)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
