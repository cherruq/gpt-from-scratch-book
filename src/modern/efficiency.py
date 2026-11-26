"""M8 reference experiments for memory traffic, quantization and decoding."""

import numpy as np


def cache_bytes(layers, batch, kv_heads, length, width, itemsize):
    return 2 * layers * batch * kv_heads * length * width * itemsize


def forward_flops(batch, time, channels, hidden, layers, vocab):
    linear = 2 * batch * time * (layers * (4 * channels ** 2 + 2 * channels * hidden) + channels * vocab)
    attention = 4 * batch * layers * time ** 2 * channels
    return linear + attention


# region online
def online_attention(scores, values, block=2):
    """One attention row; stream scores and values without a full softmax row."""
    maximum = -np.inf
    mass = 0.0
    accumulator = np.zeros(values.shape[-1])
    for start in range(0, len(scores), block):
        s = scores[start:start + block]
        v = values[start:start + block]
        if np.all(np.isneginf(s)):
            continue
        new_max = max(maximum, s.max())
        scale = np.exp(maximum - new_max)
        weights = np.exp(s - new_max)
        mass = mass * scale + weights.sum()
        accumulator = accumulator * scale + weights @ v
        maximum = new_max
    if mass == 0:
        raise ValueError("At least one score must be finite")
    return accumulator / mass
# endregion online


def quantize_int8(weight):
    max_abs = np.abs(weight).max(axis=0)
    scale = np.where(max_abs > 0, max_abs / 127, 1.0)
    return np.rint(weight / scale).clip(-127, 127).astype(np.int8), scale


def speculative_choice(target, draft, rng):
    """One proposal; accept at min(1, target/draft), otherwise sample residual."""
    candidate = rng.choice(len(draft), p=draft)
    if rng.random() <= min(1.0, target[candidate] / draft[candidate]):
        return candidate, True
    residual = np.maximum(target - draft, 0.0)
    return rng.choice(len(target), p=residual / residual.sum()), False
