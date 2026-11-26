"""M4–M5 experiments. Run python -m examples.transformer 22..28."""

import argparse

import numpy as np

from src.autograd.tensor import Tensor
from src.numpy_gpt.model import (Config, GPT, attention, attention_weights, causal_mask,
                                 cross_entropy, gelu, merge_heads, split_heads, transformer_block)


# region retrieve
def retrieve():
    values = np.array([[2., 0.], [0., 4.]])
    fixed = np.array([[0.25, 0.75]])
    q, keys = np.array([[1., 0.]]), np.eye(2)
    score = (q @ keys.T) / np.sqrt(2)
    return fixed @ values, score
# endregion retrieve


# region scaled
def scaled(with_mask=False):
    q, k = (Tensor(np.eye(2).reshape(1, 1, 2, 2)) for _ in range(2))
    v = Tensor(np.array([[2., 0.], [0., 4.]]).reshape(1, 1, 2, 2))
    mask = causal_mask([0, 1]) if with_mask else None
    return attention_weights(q, k, mask).data[0, 0], attention(q, k, v, mask).data[0, 0]
# endregion scaled


# region causal
def causal():
    model = GPT(Config(vocab=8))
    a = model.forward([[0, 1, 2, 3]], track=False).data
    b = model.forward([[0, 1, 6, 7]], track=False).data
    return causal_mask(np.arange(4)), np.max(np.abs(a - b), axis=-1)
# endregion causal


# region multihead
def multihead():
    x = Tensor(np.arange(16.).reshape(1, 4, 4), True)
    heads = split_heads(x, 2)
    mixed = merge_heads(attention(heads, heads, heads, causal_mask(np.arange(4))))
    (mixed ** 2).sum().backward()
    return heads.shape, heads.data[0, 1, 2], mixed.shape, x.grad
# endregion multihead


# region ffn
def ffn():
    x = Tensor([[[1., -1.]]], True)
    expansion = Tensor([[1., 0., -1.], [0., 1., 0.]])
    contraction = Tensor([[1., 0.], [0., 1.], [1., 1.]])
    z = x @ expansion
    h = gelu(z)
    y = h @ contraction
    y.sum().backward()
    return z.data, h.data, y.data, x.grad
# endregion ffn


# region block
def block():
    model = GPT(Config(vocab=8, layers=1))
    x = Tensor(np.arange(8.).reshape(1, 2, 4) / 8, True)
    output, _ = transformer_block(x, model.parameters, "block.0", 2, causal_mask([0, 1]))
    output.sum().backward()
    residual_gradient = x.grad.copy()
    model.zero_grad()
    x.grad = None
    for suffix in ("out.weight", "out.bias", "down.weight", "down.bias"):
        model.parameters["block.0." + suffix].data.fill(0)
    identity, _ = transformer_block(x, model.parameters, "block.0", 2, causal_mask([0, 1]))
    identity.sum().backward()
    return output.data, residual_gradient, identity.data, x.grad
# endregion block


# region complete_gpt
def complete_gpt():
    model = GPT(Config(vocab=8))
    ids = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = np.array([[1, 2, 3, 0], [5, 6, 7, 4]])
    logits = model.forward(ids)
    loss = cross_entropy(logits, targets)
    loss.backward()
    return model.parameter_count(), logits.shape, float(loss.data), {
        name: tensor.grad.shape for name, tensor in model.parameters.items()
    }
# endregion complete_gpt


EXPERIMENTS = {22: retrieve, 23: scaled, 24: causal, 25: multihead,
               26: ffn, 27: block, 28: complete_gpt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", choices=EXPERIMENTS, type=int)
    args = parser.parse_args()
    print(EXPERIMENTS[args.chapter]())
