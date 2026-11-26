"""Extra worked examples for chapters 7, 8 and 15."""

import argparse
import json

import numpy as np

from examples.math_basics import central_difference
from src.autograd.tensor import Tensor


# region direction
def direction():
    x = np.array([2., -1.])
    u = np.array([3., 4.]) / 5
    w = np.zeros(2)

    def loss(weights):
        return np.logaddexp(0, -(weights @ x))

    gradient = -0.5 * x
    analytic = gradient @ u
    numeric = central_difference(lambda h: loss(w + h * u), 0., 1e-5)

    def scalar_loss(value):
        return np.logaddexp(0, -2 * value)

    point = 0.3
    slope = -2 / (1 + np.exp(2 * point))
    steps = (1e-2, 1e-5, 1e-12)
    errors = [abs(central_difference(scalar_loss, point, h) - slope) for h in steps]
    return {"gradient": gradient.tolist(), "directional": float(analytic),
            "difference": float(numeric), "steps": steps, "absolute_errors": errors}
# endregion direction


# region shared_affine
def shared_affine():
    x = Tensor([[[1, 2]], [[3, 4]]], True)
    w = Tensor([[2, -1], [0, 3]], True)
    b = Tensor([1, -1], True)
    upstream = np.array([[[1., 0.]], [[-1., 2.]]])
    y = x @ w + b
    data_loss = (y * upstream).sum()
    penalty = (w * w).sum() / 4
    (data_loss + penalty).backward()

    # This finite difference evaluates the scalar function using plain NumPy.
    def loss(weight):
        return np.sum((x.data @ weight + b.data) * upstream) + np.sum(weight**2) / 4

    finite = np.empty_like(w.data)
    for index in np.ndindex(w.shape):
        unit = np.zeros_like(w.data)
        unit[index] = 1
        finite[index] = central_difference(lambda h: loss(w.data + h * unit), 0.)
    return {"output": y.data.tolist(), "dx": x.grad.tolist(), "dw": w.grad.tolist(),
            "db": b.grad.tolist(), "weight_fd_error": float(np.max(np.abs(finite - w.grad)))}
# endregion shared_affine


# region table_seed
def table_seed():
    table = Tensor([[1, 2], [3, 4], [5, 6]], True)
    scale = Tensor([2, 3], True)
    ids = [1, 1, 0]
    output = (table[ids] * scale).transpose(1, 0)
    seed = np.arange(1., 7.).reshape(2, 3)
    output.backward(seed)
    grad_table = table.grad.copy()
    grad_scale = scale.grad.copy()
    output.backward(seed)
    return {"output": output.data.tolist(), "table": grad_table.tolist(),
            "scale": grad_scale.tolist(), "second_table": table.grad.tolist(),
            "seeded_scalar": float(np.sum(output.data * seed))}
# endregion table_seed


LABS = {7: direction, 8: shared_affine, 15: table_seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=LABS)
    print(json.dumps(LABS[parser.parse_args().chapter](), indent=2))
