"""
Task 1 [25]: Design a Value node that stores data, gradient, parent nodes,
and the local derivative rule.

This file defines only the node itself. Operations are added in
task2_operations.py and the backward pass in task3_backward.py.
"""


class Value:
    """A scalar node in a computation graph.

    data       - forward value
    grad       - d(output)/d(this node); accumulated during backward()
    _prev      - parent nodes (the operands that produced this node)
    _backward  - closure holding the LOCAL derivative rule for this node
    _op        - label of the operation that produced the node
    """

    __slots__ = ("data", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._backward = lambda: None  # leaves have no local rule
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}, op='{self._op}')"


def graph_nodes(root):
    """Return every node reachable from root (shows the size of the graph)."""
    seen, stack, out = set(), [root], []
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        out.append(n)
        stack.extend(n._prev)
    return out


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(-3.0, _children=(a,), _op="demo")
    print(a)
    print(b)
    print("parents of b:", b._prev)
    print("local rule stored as a callable:", callable(b._backward))
