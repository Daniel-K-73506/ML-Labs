"""
Task 3 [25]: Implement the backward pass.

  1. Topologically sort the graph
  2. Set output gradient to 1
  3. Visit nodes in reverse order
  4. Accumulate gradients, do not overwrite

Importing this module gives a complete engine: Value + operations + backward().
"""

from task2_operations import Value, graph_nodes  # noqa: F401


def backward(self):
    # 1) topological sort (iterative DFS: safe for deep graphs, no recursion limit)
    topo, visited = [], set()
    stack = [(self, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            topo.append(node)          # all parents were appended before this node
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for parent in node._prev:
            if id(parent) not in visited:
                stack.append((parent, False))

    # 2) output gradient = 1
    self.grad = 1.0

    # 3) reverse topological order; each local rule uses += so gradients accumulate
    for node in reversed(topo):
        node._backward()


Value.backward = backward


if __name__ == "__main__":
    # x is used twice -> its gradient must be the SUM of both paths
    x = Value(3.0)
    y = x * x + x                       # dy/dx = 2x + 1 = 7
    y.backward()
    print(f"d(x*x + x)/dx at x=3 -> {x.grad} (expected 7.0)")
    assert abs(x.grad - 7.0) < 1e-12

    # diamond graph: z = (a+b) * (a*b)
    a, b = Value(2.0), Value(5.0)
    z = (a + b) * (a * b)               # dz/da = ab + (a+b)b = 10+35 = 45 ; dz/db = ab + (a+b)a = 10+14 = 24
    z.backward()
    print(f"dz/da = {a.grad} (expected 45.0), dz/db = {b.grad} (expected 24.0)")
    assert abs(a.grad - 45.0) < 1e-12 and abs(b.grad - 24.0) < 1e-12
    print("backward pass OK")
