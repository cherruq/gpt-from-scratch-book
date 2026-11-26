"""M1. Two-layer network with explicitly written gradients."""

import numpy as np


# region linear
class Linear:
    def __init__(self, inputs, outputs, rng):
        self.weight = rng.normal(0, 1 / np.sqrt(inputs), (inputs, outputs))
        self.bias = np.zeros(outputs)

    def forward(self, x):
        return x @ self.weight + self.bias

    def vjp(self, x, gradient):
        return gradient @ self.weight.T, x.T @ gradient, gradient.sum(axis=0)
# endregion linear


class MLP:
    def __init__(self, seed=42, hidden=8):
        rng = np.random.default_rng(seed)
        self.up = Linear(2, hidden, rng)
        self.down = Linear(hidden, 1, rng)

    def objective(self, x, y):
        hidden = np.tanh(self.up.forward(x))
        residual = self.down.forward(hidden) - y
        return np.mean(residual ** 2), hidden, residual

# region manual_backward
    def backward(self, x, hidden, residual):
        dl = 2 * residual / residual.size
        dh, dw2, db2 = self.down.vjp(hidden, dl)
        dz = dh * (1 - hidden ** 2)
        dx, dw1, db1 = self.up.vjp(x, dz)
        return dx, dw1, db1, dw2, db2
# endregion manual_backward

    def step(self, gradients, rate):
        parameters = self.up.weight, self.up.bias, self.down.weight, self.down.bias
        for parameter, gradient in zip(parameters, gradients[1:], strict=True):
            parameter -= rate * gradient


def xor_data():
    return np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]), np.array([[0.], [1.], [1.], [0.]])


# region xor_train
def xor_train(steps=800, rate=0.15):
    x, y = xor_data()
    model = MLP()
    initial, _, _ = model.objective(x, y)
    for _ in range(steps):
        _, hidden, residual = model.objective(x, y)
        model.step(model.backward(x, hidden, residual), rate)
    final, _, residual = model.objective(x, y)
    return float(initial), float(final), residual + y
# endregion xor_train
