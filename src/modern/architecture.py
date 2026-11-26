"""M8 illustrative components separate from the baseline GPT."""

import numpy as np

from src.autograd.tensor import Tensor
from src.numpy_gpt.model import attention, causal_mask, merge_heads, softmax, split_heads


# region rope
def rotary(x, positions, base=10_000):
    d = x.shape[-1]
    if d % 2 or len(positions) != x.shape[-2]:
        raise ValueError("RoPE needs paired channels and one position per token")
    frequencies = base ** (-np.arange(0, d, 2) / d)
    angles = np.asarray(positions)[:, None] * frequencies
    cos, sin = np.cos(angles), np.sin(angles)

    def rotate(data, direction):
        first, second = data[..., ::2], data[..., 1::2]
        output = np.empty_like(data)
        output[..., ::2] = cos * first - direction * sin * second
        output[..., 1::2] = direction * sin * first + cos * second
        return output

    return Tensor.from_op(rotate(x.data, 1), (x, lambda g: rotate(g, -1)))
# endregion rope


def rms_norm(x, gain, eps=1e-5):
    return x * ((x * x).mean(axis=-1, keepdims=True) + eps) ** -0.5 * gain


def swiglu(x, gate, up, down):
    projected = x @ gate
    silu = projected * (1 + (0.5 * projected).tanh()) * 0.5
    return (silu * (x @ up)) @ down


# region grouped_attention
def grouped_attention(q, k, v, mask=None):
    hq, hkv = q.shape[1], k.shape[1]
    if hq % hkv or k.shape != v.shape:
        raise ValueError("Use matching key/value heads dividing the query head count")
    kv_head = np.repeat(np.arange(hkv), hq // hkv)
    return attention(q, k[:, kv_head], v[:, kv_head], mask)
# endregion grouped_attention


def modern_block(x, parameters, query_heads, kv_heads):
    """Tiny pre-RMS block with RoPE, GQA and SwiGLU, using explicit weights."""
    p = parameters
    positions = np.arange(x.shape[1])
    z = rms_norm(x, p["attention_gain"])
    q = rotary(split_heads(z @ p["q"], query_heads), positions)
    k = rotary(split_heads(z @ p["k"], kv_heads), positions)
    v = split_heads(z @ p["v"], kv_heads)
    x = x + merge_heads(grouped_attention(q, k, v, causal_mask(positions))) @ p["out"]
    return x + swiglu(rms_norm(x, p["ffn_gain"]), p["gate"], p["up"], p["down"])


# region routing
def mixture(x, router, experts, top_k=1):
    """Evaluate all small experts, then mix only top-k outputs for each token."""
    logits = x @ router
    count = len(experts)
    if not 1 <= top_k <= count or logits.shape[-1] != count:
        raise ValueError("Router width and top_k must match expert count")
    ranked = np.argsort(-logits.data, axis=-1, kind="stable")
    selected = np.zeros(logits.shape, dtype=bool)
    np.put_along_axis(selected, ranked[..., :top_k], True, axis=-1)
    gates = softmax(logits.where(selected, -np.inf))
    out = sum((x @ weight) * gates[..., i:i+1] for i, weight in enumerate(experts))
    leading = tuple(range(len(logits.shape) - 1))
    load = selected.mean(axis=leading) / top_k
    average_probability = softmax(logits).mean(axis=leading)
    auxiliary = count * (average_probability * load).sum()
    return out, auxiliary, load
# endregion routing
