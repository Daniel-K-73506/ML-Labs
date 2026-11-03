"""
Task 5 [15]: Train a tiny model on make_moons.
Architecture: 2 -> 8 -> 1, sigmoid output, BCE loss.
Compare speed with PyTorch. Median of 5 runs.

Both implementations use identical initial weights, full-batch SGD and the
same loss formula, so their losses should agree to ~1e-16.
"""

import statistics
import time

import numpy as np
import torch
from sklearn.datasets import make_moons

from task3_backward import Value

N_SAMPLES = 100
EPOCHS = 50
LR = 0.5
EPS = 1e-12


def init_params():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 0.5, (8, 2))
    b1 = np.zeros(8)
    W2 = rng.normal(0, 0.5, (1, 8))
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def train_value(X, y, params):
    """2 -> 8 (ReLU) -> 1 (sigmoid), BCE loss, full-batch SGD with our engine."""
    W1, b1, W2, b2 = params
    W1v = [[Value(w) for w in row] for row in W1]
    b1v = [Value(b) for b in b1]
    W2v = [Value(w) for w in W2[0]]
    b2v = Value(b2[0])
    allp = [p for row in W1v for p in row] + b1v + W2v + [b2v]

    final_loss = None
    for _ in range(EPOCHS):
        losses = []
        for xi, yi in zip(X, y):
            h = []
            for j in range(8):
                z = W1v[j][0] * xi[0] + W1v[j][1] * xi[1] + b1v[j]
                h.append(z.relu())
            out = b2v
            for j in range(8):
                out = out + W2v[j] * h[j]
            p = out.sigmoid()
            # BCE: -(y log p + (1-y) log(1-p))
            loss_i = -((p + EPS).log() * float(yi) + ((1 - p) + EPS).log() * float(1 - yi))
            losses.append(loss_i)
        loss = Value(0.0)
        for l in losses:
            loss = loss + l
        loss = loss * (1.0 / len(X))

        for p in allp:
            p.grad = 0.0
        loss.backward()
        for p in allp:
            p.data -= LR * p.grad
        final_loss = loss.data

    W1n = np.array([[w.data for w in row] for row in W1v])
    b1n = np.array([b.data for b in b1v])
    W2n = np.array([w.data for w in W2v])
    h = np.maximum(X @ W1n.T + b1n, 0)
    pred = 1 / (1 + np.exp(-(h @ W2n + b2v.data))) > 0.5
    return final_loss, float(np.mean(pred == y))


def train_torch(X, y, params):
    W1, b1, W2, b2 = (torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in params)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    final_loss = None
    for _ in range(EPOCHS):
        h = torch.relu(Xt @ W1.T + b1)
        p = torch.sigmoid(h @ W2.T + b2).squeeze(1)
        loss = -(yt * torch.log(p + EPS) + (1 - yt) * torch.log(1 - p + EPS)).mean()
        for t in (W1, b1, W2, b2):
            t.grad = None
        loss.backward()
        with torch.no_grad():
            for t in (W1, b1, W2, b2):
                t -= LR * t.grad
        final_loss = loss.item()
    with torch.no_grad():
        h = torch.relu(Xt @ W1.T + b1)
        pred = torch.sigmoid(h @ W2.T + b2).squeeze(1) > 0.5
        acc = (pred.double() == yt).double().mean().item()
    return final_loss, acc


def main():
    print("=" * 70)
    print(f"TASK 5: make_moons training (N={N_SAMPLES}, epochs={EPOCHS}, lr={LR}), median of 5 runs")
    print("=" * 70)
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.15, random_state=0)

    def timed(fn):
        times, res = [], None
        for _ in range(5):
            t0 = time.perf_counter()
            res = fn(X, y, init_params())
            times.append(time.perf_counter() - t0)
        return statistics.median(times), res

    t_mine, (l_mine, a_mine) = timed(train_value)
    t_torch, (l_torch, a_torch) = timed(train_torch)

    print(f"  our Value engine : median {t_mine * 1000:9.1f} ms | final loss {l_mine:.6f} | acc {a_mine:.2%}")
    print(f"  PyTorch autograd : median {t_torch * 1000:9.1f} ms | final loss {l_torch:.6f} | acc {a_torch:.2%}")
    print(f"  PyTorch is {t_mine / t_torch:.0f}x faster (vectorised tensor kernels vs "
          f"one Python object per scalar)")
    print(f"  loss difference between implementations: {abs(l_mine - l_torch):.2e}")


if __name__ == "__main__":
    main()
