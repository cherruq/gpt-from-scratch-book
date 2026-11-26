"""The same decoder using PyTorch Tensor operations and autograd."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.numpy_gpt.model import Config, GPT


class TorchGPT(nn.Module):
    def __init__(self, config: Config, seed=42, *, device=None, dtype=torch.float64, sdpa=False):
        super().__init__()
        self.config = config
        self.sdpa = sdpa
        weights = GPT(config, seed).state_dict()
        self.weights = nn.ParameterDict({
            self.key(name): nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
            for name, value in weights.items()
        })

    @staticmethod
    def key(name):
        return name.replace(".", "__")

    def named_weights(self):
        return ((name.replace("__", "."), value) for name, value in self.weights.items())

    def numpy_state(self):
        return {name: value.detach().cpu().numpy().copy() for name, value in self.named_weights()}

    @torch.no_grad()
    def import_numpy(self, weights):
        current = dict(self.named_weights())
        if set(weights) != set(current):
            raise ValueError("Parameter names do not match")
        for name, value in current.items():
            if value.shape != weights[name].shape:
                raise ValueError(f"Incompatible {name}")
        for name, value in current.items():
            value.copy_(torch.as_tensor(weights[name], device=value.device, dtype=value.dtype))

# region torch_forward
    def forward(self, ids):
        cfg = self.config
        if ids.ndim != 2 or not all(ids.shape) or ids.shape[1] > cfg.context:
            raise ValueError("Expected nonempty [B,T] within configured context")
        batch, time = ids.shape
        dimensions = cfg.channels // cfg.heads

        def param(name):
            return self.weights[self.key(name)]

        def linear(x, name):
            return x @ param(name + ".weight") + param(name + ".bias")

        def norm(x, name):
            return F.layer_norm(x, (cfg.channels,), param(name + ".gain"), param(name + ".bias"), 1e-5)

        def split(x):
            return x.reshape(batch, time, cfg.heads, dimensions).transpose(1, 2)

        mask = torch.ones((time, time), device=ids.device, dtype=torch.bool).tril() if not self.sdpa else None
        x = param("token").index_select(0, ids.reshape(-1)).reshape(batch, time, cfg.channels)
        x = x + param("position")[:time]
        for i in range(cfg.layers):
            tag = f"block.{i}"
            z = norm(x, tag + ".norm1")
            q, k, v = (split(linear(z, tag + "." + proj)) for proj in ("q", "k", "v"))
            if self.sdpa:
                mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(dimensions)
                weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
                mixed = weights @ v
            x = x + linear(mixed.transpose(1, 2).reshape(batch, time, cfg.channels), tag + ".out")
            z = norm(x, tag + ".norm2")
            x = x + linear(F.gelu(linear(z, tag + ".up"), approximate="tanh"), tag + ".down")
        return norm(x, "final") @ param("token").T
# endregion torch_forward


def objective(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
