"""
Task 4 [10]: Verify against PyTorch autograd and numerical gradients on
((a*b + c)**2).relu().sum().  Max absolute difference must be < 1e-6.
"""

import numpy as np
import torch

from task3_backward import Value, graph_nodes


def main():
    print("=" * 70)
    print("TASK 4: gradient verification on ((a*b + c)**2).relu().sum()")
    print("=" * 70)
    rng = np.random.default_rng(42)
    a0, b0, c0 = (rng.normal(size=6) for _ in range(3))

    # --- our engine: elementwise scalar graph, then sum ---
    av = [Value(x) for x in a0]
    bv = [Value(x) for x in b0]
    cv = [Value(x) for x in c0]
    total = Value(0.0)
    for i in range(len(av)):
        total = total + ((av[i] * bv[i] + cv[i]) ** 2).relu()
    total.backward()
    mine = [np.array([v.grad for v in vs]) for vs in (av, bv, cv)]
    print(f"graph nodes built during forward pass: {len(graph_nodes(total))}")

    # --- PyTorch autograd (float64) ---
    ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True)
                  for x in (a0, b0, c0))
    ((ta * tb + tc) ** 2).relu().sum().backward()
    torch_g = [t.grad.numpy() for t in (ta, tb, tc)]

    # --- numerical central differences ---
    def f(a, b, c):
        return np.sum(np.maximum((a * b + c) ** 2, 0.0))

    eps = 1e-6
    base = [a0.copy(), b0.copy(), c0.copy()]
    num = []
    for k in range(3):
        g = np.zeros_like(base[k])
        for i in range(base[k].size):
            plus = [x.copy() for x in base]
            minus = [x.copy() for x in base]
            plus[k][i] += eps
            minus[k][i] -= eps
            g[i] = (f(*plus) - f(*minus)) / (2 * eps)
        num.append(g)

    worst = 0.0
    for name, m, t, n in zip("abc", mine, torch_g, num):
        d_t = np.max(np.abs(m - t))
        d_n = np.max(np.abs(m - n))
        worst = max(worst, d_t, d_n)
        print(f"  d/d{name}: max|mine-torch| = {d_t:.3e}   max|mine-numeric| = {d_n:.3e}")
    print(f"  worst difference = {worst:.3e}  (requirement: < 1e-6)")
    assert worst < 1e-6, "gradient check failed"
    print("  PASS")


if __name__ == "__main__":
    main()
