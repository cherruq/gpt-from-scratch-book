"""M2. Scalar reverse mode with identity-based DAG traversal."""


class Value:
    def __init__(self, data, edges=()):
        self.data = float(data)
        self.edges = edges
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, ((self, 1.0), (other, 1.0)))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, ((self, other.data), (other, self.data)))

    def backward(self):
        ordered, visited = [], set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent, _ in node.edges:
                visit(parent)
            ordered.append(node)

        visit(self)
        for node in ordered:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(ordered):
            for parent, slope in node.edges:
                parent.grad += node.grad * slope


if __name__ == "__main__":
    x = Value(2)
    y = x * x + x
    y.backward()
    print(y.data, x.grad)
