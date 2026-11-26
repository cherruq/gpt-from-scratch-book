"""AdamW with explicit moment buffers and global gradient clipping."""

import numpy as np


class AdamW:
    def __init__(self, parameters, lr=0.003, beta1=0.9, beta2=0.999,
                 eps=1e-8, decay=0.01, max_norm=1.0):
        if lr <= 0 or eps <= 0 or decay < 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("Invalid AdamW hyperparameters")
        if max_norm is not None and max_norm <= 0:
            raise ValueError("max_norm must be positive")
        self.parameters = parameters
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.eps, self.decay, self.max_norm = eps, decay, max_norm
        self.m = {name: np.zeros_like(p.data) for name, p in parameters.items()}
        self.v = {name: np.zeros_like(p.data) for name, p in parameters.items()}
        self.steps = 0

    def options(self):
        return {key: getattr(self, key) for key in
                ("lr", "beta1", "beta2", "eps", "decay", "max_norm")}

    def step(self):
        active = [(name, p) for name, p in self.parameters.items() if p.grad is not None]
        if not active:
            raise ValueError("Run backward before optimizer step")
        norm = np.sqrt(sum(np.sum(p.grad ** 2) for _, p in active))
        if not np.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        scale = 1.0 if self.max_norm is None or norm <= self.max_norm else self.max_norm / norm
        self.steps += 1
        for name, p in active:
            grad = p.grad * scale
            m, v = self.m[name], self.v[name]
            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * grad ** 2
            corrected_m = m / (1 - self.beta1 ** self.steps)
            corrected_v = v / (1 - self.beta2 ** self.steps)
            p.data[...] *= 1 - self.lr * self.decay
            p.data[...] -= self.lr * corrected_m / (np.sqrt(corrected_v) + self.eps)
        return float(norm)
