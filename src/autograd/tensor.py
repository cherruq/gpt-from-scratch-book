"""Reverse-mode autodiff supporting the tensor operations used by the baseline."""

from collections.abc import Callable
from typing import Self

import numpy as np


def unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo the axis insertion and size-one expansion done by NumPy."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.edges: list[tuple[Tensor, Callable]] = []

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def from_op(data, *edges) -> "Tensor":
        active = [(parent, vjp) for parent, vjp in edges if parent.requires_grad]
        result = Tensor(data, bool(active))
        result.edges = active
        return result

    def backward(self, seed=None) -> None:
        if seed is None:
            if self.shape != ():
                raise ValueError("A non-scalar output needs an explicit backward seed")
            seed = np.ones_like(self.data)
        seed = np.asarray(seed, dtype=np.float64)
        if seed.shape != self.shape:
            raise ValueError("Backward seed must match output shape")

        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent, _ in node.edges)

        gradients = {self: seed}
        for node in reversed(order):
            if not node.requires_grad:
                continue
            gradient = gradients.pop(node)
            if not node.edges:
                node.grad = gradient.copy() if node.grad is None else node.grad + gradient
            for parent, vjp in node.edges:
                contribution = vjp(gradient)
                if parent in gradients:
                    gradients[parent] = gradients[parent] + contribution
                else:
                    gradients[parent] = contribution

    @staticmethod
    def wrap(value) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    def __add__(self, other) -> Self:
        other = self.wrap(other)
        return self.from_op(self.data + other.data,
                            (self, lambda g: unbroadcast(g, self.shape)),
                            (other, lambda g: unbroadcast(g, other.shape)))

    __radd__ = __add__

    def __neg__(self) -> Self:
        return self.from_op(-self.data, (self, lambda g: -g))

    def __sub__(self, other) -> Self:
        return self + -self.wrap(other)

    def __rsub__(self, other) -> Self:
        return self.wrap(other) + -self

    def __mul__(self, other) -> Self:
        other = self.wrap(other)
        return self.from_op(self.data * other.data,
                            (self, lambda g: unbroadcast(g * other.data, self.shape)),
                            (other, lambda g: unbroadcast(g * self.data, other.shape)))

    __rmul__ = __mul__

    def __pow__(self, power: float) -> Self:
        data = self.data ** power
        if power == 0:
            return self.from_op(data, (self, lambda g: np.zeros_like(self.data)))
        return self.from_op(data, (self, lambda g: g * power * self.data ** (power - 1)))

    def __truediv__(self, other) -> Self:
        return self * self.wrap(other) ** -1.0

    def __matmul__(self, other) -> Self:
        other = self.wrap(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("matmul expects explicit row and column axes")
        return self.from_op(
            self.data @ other.data,
            (self, lambda g: unbroadcast(g @ other.data.swapaxes(-1, -2), self.shape)),
            (other, lambda g: unbroadcast(self.data.swapaxes(-1, -2) @ g, other.shape)),
        )

    def sum(self, axis=None, keepdims=False) -> Self:
        data = self.data.sum(axis=axis, keepdims=keepdims)

        def vjp(gradient):
            if axis is not None and not keepdims:
                gradient = np.expand_dims(gradient, axis=axis)
            return np.broadcast_to(gradient, self.shape)

        return self.from_op(data, (self, vjp))

    def mean(self, axis=None, keepdims=False) -> Self:
        dimensions = range(self.data.ndim) if axis is None else np.atleast_1d(axis)
        count = int(np.prod([self.shape[dimension] for dimension in dimensions]))
        return self.sum(axis, keepdims) / count

    def reshape(self, *shape) -> Self:
        return self.from_op(self.data.reshape(*shape), (self, lambda g: g.reshape(self.shape)))

    def transpose(self, *axes) -> Self:
        data = self.data.transpose(axes)
        inverse = np.argsort([axis % self.data.ndim for axis in axes])
        return self.from_op(data, (self, lambda g: g.transpose(inverse)))

    def __getitem__(self, index) -> Self:
        def vjp(gradient):
            buffer = np.zeros_like(self.data)
            np.add.at(buffer, index, gradient)
            return buffer

        return self.from_op(self.data[index], (self, vjp))

    def exp(self) -> Self:
        value = np.exp(self.data)
        return self.from_op(value, (self, lambda g: g * value))

    def log(self) -> Self:
        return self.from_op(np.log(self.data), (self, lambda g: g / self.data))

    def tanh(self) -> Self:
        value = np.tanh(self.data)
        return self.from_op(value, (self, lambda g: g * (1 - value * value)))

    def where(self, mask, otherwise) -> Self:
        """Keep entries selected by mask, replacing others by a constant."""
        return self.from_op(np.where(mask, self.data, otherwise),
                            (self, lambda g: unbroadcast(np.where(mask, g, 0), self.shape)))
