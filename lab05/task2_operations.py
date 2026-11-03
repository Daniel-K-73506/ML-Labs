"""
Task 2 [20]: Implement addition, multiplication, powers and ReLU.
Using them builds a computation graph during the forward pass.

Each operation creates a new Value whose _prev holds its operands and whose
_backward closure holds the local derivative rule. Gradients are ACCUMULATED
with += (never overwritten), which matters when a node is used twice.

Extra ops (exp, log, sigmoid, neg, sub, div) are built from the same
machinery and are needed for the sigmoid + BCE model in Task 5.
"""

import math

from task1_value import Value, graph_nodes  # noqa: F401  (graph_nodes re-exported)


def _wrap(x):
    return x if isinstance(x, Value) else Value(x)


def _add(self, other):
    other = _wrap(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
        self.grad += out.grad      # d(a+b)/da = 1
        other.grad += out.grad     # d(a+b)/db = 1

    out._backward = _backward
    return out


def _mul(self, other):
    other = _wrap(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad   # d(ab)/da = b
        other.grad += self.data * out.grad   # d(ab)/db = a

    out._backward = _backward
    return out


def _pow(self, p):
    assert isinstance(p, (int, float)), "only constant powers supported"
    out = Value(self.data ** p, (self,), f"**{p}")

    def _backward():
        self.grad += p * (self.data ** (p - 1)) * out.grad

    out._backward = _backward
    return out


def _relu(self):
    out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

    def _backward():
        self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

    out._backward = _backward
    return out


def _exp(self):
    out = Value(math.exp(self.data), (self,), "exp")

    def _backward():
        self.grad += out.data * out.grad

    out._backward = _backward
    return out


def _log(self):
    out = Value(math.log(self.data), (self,), "log")

    def _backward():
        self.grad += (1.0 / self.data) * out.grad

    out._backward = _backward
    return out


def _sigmoid(self):
    return (1 + (-self).exp()) ** -1


def _neg(self):
    return self * -1.0


def _sub(self, other):
    return self + (-other if isinstance(other, Value) else -float(other))


def _rsub(self, other):
    return (-self) + other


def _truediv(self, other):
    return self * _wrap(other) ** -1


Value.__add__ = _add
Value.__radd__ = _add
Value.__mul__ = _mul
Value.__rmul__ = _mul
Value.__pow__ = _pow
Value.relu = _relu
Value.exp = _exp
Value.log = _log
Value.sigmoid = _sigmoid
Value.__neg__ = _neg
Value.__sub__ = _sub
Value.__rsub__ = _rsub
Value.__truediv__ = _truediv


if __name__ == "__main__":
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = ((a * b + c) ** 2).relu()
    print("forward value:", out.data, "(expected 16.0)")
    print("graph built during forward pass:")
    for n in reversed(graph_nodes(out)):
        parents = ", ".join(f"{p._op or 'leaf'}={p.data:g}" for p in n._prev)
        print(f"  op={n._op or 'leaf':7s} data={n.data:8.3f}  parents=[{parents}]")
