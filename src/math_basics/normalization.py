"""The hand-derived LayerNorm/RMSNorm gradient and inverted dropout."""

import numpy as np


# region norms
def norm_vjp(x, gain, gradient, *, eps=1e-5, center=True):
    u = x - x.mean(axis=-1, keepdims=True) if center else x
    reciprocal = (np.mean(u ** 2, axis=-1, keepdims=True) + eps) ** -0.5
    normalized = u * reciprocal
    g = gradient * gain
    core = g - normalized * (g * normalized).mean(axis=-1, keepdims=True)
    if center:
        core -= g.mean(axis=-1, keepdims=True)
    leading = tuple(range(x.ndim - 1))
    return reciprocal * core, (gradient * normalized).sum(axis=leading), gradient.sum(axis=leading)


def dropout(x, probability, rng, *, training):
    if not 0 <= probability < 1:
        raise ValueError("Drop probability must lie in [0,1)")
    if not training or probability == 0:
        return x
    keep = rng.random(x.shape) >= probability
    return x * keep / (1 - probability)
# endregion norms
