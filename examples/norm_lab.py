"""Chapter 16: unpack LayerNorm and check all of its gradients."""

import argparse
import json

import numpy as np

from examples.math_basics import central_difference
from src.autograd.tensor import Tensor
from src.math_basics.normalization import norm_vjp
from src.numpy_gpt.model import layer_norm


# region same_loss
def plain_layer_norm(x, gain, bias, epsilon):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    scale = np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + epsilon)
    return centered / scale * gain + bias


def derivative(objective, args, argument):
    """Differentiate a scalar objective through one input at a time."""
    base = args[argument]
    result = np.empty_like(base)
    for index in np.ndindex(base.shape):
        def perturbed(step):
            trial = list(args)
            trial[argument] = base.copy()
            trial[argument][index] += step
            return objective(*trial)

        result[index] = central_difference(perturbed, 0.)
    return result
# endregion same_loss


def hand_case():
    x = np.array([[[0., 2.]]])
    gain, bias = np.array([2., -1.]), np.array([1., 0.5])
    incoming = np.array([[[1., 3.]]])
    dx, dgain, dbias = norm_vjp(x, gain, incoming, eps=3.)
    return {"y": plain_layer_norm(x, gain, bias, 3.).tolist(),
            "dx": dx.tolist(), "dgain": dgain.tolist(), "dbias": dbias.tolist()}


# region norm_check
def check(batch, time, channels, epsilon, seed):
    if min(batch, time, channels) < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Use positive dimensions and a finite, positive epsilon")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, time, channels))
    x[0, 0] = 1.  # Include one exactly constant row.
    gain, bias = rng.normal(size=(2, channels))
    incoming = rng.normal(size=x.shape)
    args = (x, gain, bias)
    tensors = [Tensor(a, requires_grad=True) for a in args]
    y = layer_norm(*tensors, eps=epsilon)
    (y * incoming).sum().backward()
    formula = norm_vjp(x, gain, incoming, eps=epsilon)

    def objective(x_value, gain_value, bias_value):
        return np.sum(plain_layer_norm(x_value, gain_value, bias_value, epsilon) * incoming)

    numeric = [derivative(objective, args, i) for i in range(len(args))]
    names = ("x", "gain", "bias")
    return {"shape": list(x.shape), "epsilon": epsilon,
            "finite_diff_max_errors": {
                name: float(np.max(np.abs(analytic - approximated)))
                for name, analytic, approximated in zip(names, formula, numeric, strict=True)
            },
            "autograd_max_errors": {
                name: float(np.max(np.abs(analytic - tensor.grad)))
                for name, analytic, tensor in zip(names, formula, tensors, strict=True)
            },
            "max_input_gradient_row_sum": float(np.max(np.abs(formula[0].sum(axis=-1))))}
# endregion norm_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--time", type=int, default=3)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps({"hand": hand_case(), "check": check(**vars(args))}, indent=2))
