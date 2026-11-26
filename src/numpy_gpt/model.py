"""A decoder-only GPT composed from operations in src.autograd.tensor."""

from dataclasses import dataclass

import numpy as np

from src.autograd.tensor import Tensor


@dataclass(frozen=True)
class Config:
    vocab: int
    context: int = 4
    channels: int = 4
    heads: int = 2
    hidden: int = 16
    layers: int = 2

    def __post_init__(self):
        dimensions = (self.vocab, self.context, self.channels, self.heads,
                      self.hidden, self.layers)
        if any(not isinstance(dim, int) or dim < 1 for dim in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if self.channels % self.heads:
            raise ValueError("channels must be divisible by heads")


def softmax(x: Tensor) -> Tensor:
    shifted = x - np.max(x.data, axis=-1, keepdims=True)
    exp = shifted.exp()
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits: Tensor, labels) -> Tensor:
    """Mean negative log likelihood, using the fused stable value and VJP."""
    labels = np.asarray(labels)
    if logits.shape[:-1] != labels.shape or labels.size == 0:
        raise ValueError("Expected one target per nonempty logits row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Targets must be integer IDs")
    vocabulary = logits.shape[-1]
    if np.any(labels < 0) or np.any(labels >= vocabulary):
        raise ValueError("Target outside vocabulary")
    targets = labels.ravel()
    rows = np.arange(targets.size)
    flat = logits.data.reshape(-1, vocabulary)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    total = exponential.sum(axis=-1, keepdims=True)
    loss = (np.log(total[:, 0]) - shifted[rows, targets]).mean()
    gradient = exponential / total
    gradient[rows, targets] -= 1
    gradient = gradient.reshape(logits.shape) / targets.size
    return Tensor.from_op(loss, (logits, lambda g: g * gradient))


def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1 + (np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)).tanh())


def layer_norm(x: Tensor, gain: Tensor, bias: Tensor, eps=1e-5) -> Tensor:
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    return centered * (variance + eps) ** -0.5 * gain + bias


# region attention
def attention_weights(q: Tensor, k: Tensor, mask=None) -> Tensor:
    """Q [B,H,Tq,D], K [B,H,Tk,D], mask broadcasts over B and H."""
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(q.shape[-1])
    if mask is not None:
        visible = np.broadcast_to(mask, scores.shape).any(axis=-1)
        if not np.all(visible):
            raise ValueError("Every query needs at least one visible key")
        scores = scores.where(mask, -np.inf)
    return softmax(scores)


def attention(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    return attention_weights(q, k, mask) @ v
# endregion attention


# region head_axes
def split_heads(x: Tensor, heads: int) -> Tensor:
    b, t, c = x.shape
    return x.reshape(b, t, heads, c // heads).transpose(0, 2, 1, 3)


def merge_heads(x: Tensor) -> Tensor:
    b, h, t, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)
# endregion head_axes


def causal_mask(positions):
    positions = np.asarray(positions)
    return np.arange(positions[-1] + 1)[None, :] <= positions[:, None]


# region transformer_block
def transformer_block(x, parameters, prefix, heads, mask, previous=None, observer=None):
    """Shared block for full sequences, prefill and one-token decoding."""
    def linear(value, name):
        return value @ parameters[prefix + name + ".weight"] + parameters[prefix + name + ".bias"]

    def norm(value, name):
        return layer_norm(value, parameters[prefix + name + ".gain"], parameters[prefix + name + ".bias"])

    z = norm(x, ".norm1")
    q, k, v = (split_heads(linear(z, "." + name), heads) for name in ("q", "k", "v"))
    if previous is not None:
        past_k, past_v = previous
        k = Tensor(np.concatenate((past_k, k.data), axis=2))
        v = Tensor(np.concatenate((past_v, v.data), axis=2))
    weights = attention_weights(q, k, mask)
    if observer is not None:
        observer(prefix + ".attention", weights.data.copy())
    x = x + linear(merge_heads(weights @ v), ".out")
    z = norm(x, ".norm2")
    x = x + linear(gelu(linear(z, ".up")), ".down")
    return x, (k.data, v.data)
# endregion transformer_block


class GPT:
    def __init__(self, config: Config, seed=42):
        self.config = config
        self.parameters: dict[str, Tensor] = {}
        generator = np.random.default_rng(seed)
        c, f = config.channels, config.hidden

        def create(name, shape, fill=None):
            data = generator.normal(0, 0.02, shape) if fill is None else np.full(shape, fill)
            self.parameters[name] = Tensor(data, requires_grad=True)

        def linear(name, in_features, out_features):
            create(name + ".weight", (in_features, out_features))
            create(name + ".bias", (out_features,), 0)

        def norm(name):
            create(name + ".gain", (c,), 1)
            create(name + ".bias", (c,), 0)

        create("token", (config.vocab, c))
        create("position", (config.context, c))
        for i in range(config.layers):
            tag = f"block.{i}"
            norm(tag + ".norm1")
            for projection in ("q", "k", "v", "out"):
                linear(tag + "." + projection, c, c)
            norm(tag + ".norm2")
            linear(tag + ".up", c, f)
            linear(tag + ".down", f, c)
        norm("final")

    def forward(self, ids, *, track=True, cache=None, use_cache=False, observer=None):
        ids = np.asarray(ids)
        cfg = self.config
        if ids.ndim != 2 or not all(ids.shape) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("ids must be a nonempty integer array [B,T]")
        if np.any(ids < 0) or np.any(ids >= cfg.vocab):
            raise ValueError("Token outside vocabulary")
        if (use_cache or cache is not None) and track:
            raise ValueError("KV cache is for inference")
        if cache is not None and (not use_cache or len(cache) != cfg.layers):
            raise ValueError("Expected one KV pair per layer")
        offset = 0 if cache is None else cache[0][0].shape[2]
        length = ids.shape[1]
        if offset + length > cfg.context:
            raise ValueError("Sequence exceeds the learned position table")
        if cache is not None:
            expected = (ids.shape[0], cfg.heads, offset, cfg.channels // cfg.heads)
            if any(k.shape != expected or v.shape != expected for k, v in cache):
                raise ValueError("Inconsistent KV cache shape")
        p = self.parameters if track else {
            name: Tensor(value.data) for name, value in self.parameters.items()
        }

        positions = np.arange(offset, offset + length)
        mask = causal_mask(positions)
        x = p["token"][ids] + p["position"][positions]
        if observer is not None:
            observer("embedding", x.data.copy())
        updated = []
        for i in range(cfg.layers):
            previous = None if cache is None else cache[i]
            tag = f"block.{i}"
            x, pair = transformer_block(x, p, tag, cfg.heads, mask, previous, observer)
            if observer is not None:
                observer(tag, x.data.copy())
            if use_cache:
                updated.append(pair)
        logits = layer_norm(x, p["final.gain"], p["final.bias"]) @ p["token"].transpose(1, 0)
        return (logits, tuple(updated)) if use_cache else logits

    def state_dict(self):
        return {name: tensor.data.copy() for name, tensor in self.parameters.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.parameters):
            raise ValueError("Parameter names do not match model")
        for name, tensor in self.parameters.items():
            if state[name].shape != tensor.shape:
                raise ValueError(f"Incompatible shape for {name}")
        for name, tensor in self.parameters.items():
            tensor.data[...] = state[name]

    def zero_grad(self):
        for tensor in self.parameters.values():
            tensor.grad = None

    def parameter_count(self):
        return sum(tensor.data.size for tensor in self.parameters.values())
