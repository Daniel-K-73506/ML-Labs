"""
Task 6 [20]: In PyTorch, inspect the graph of a small model. Compare eager
execution against torch.compile or torch.jit.trace after warm-up (falls back
to tracing if compilation is unsupported). Inspect tensor shape, dtype,
contiguity. Measure how a non-contiguous view changes operation speed.

Also prints the short note on when compilation helps.
"""

import statistics
import time
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def walk_graph(fn, depth=0, seen=None, lines=None):
    """Collect the autograd graph (grad_fn chain) of a tensor as text lines."""
    if seen is None:
        seen, lines = set(), []
    if fn is None or id(fn) in seen:
        return lines
    seen.add(id(fn))
    lines.append("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk_graph(nxt, depth + 1, seen, lines)
    return lines


def bench(fn, *args, iters=200, repeats=5, warmup=10):
    """Median time per call in microseconds, after warm-up."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        meds = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(*args)
            meds.append((time.perf_counter() - t0) / iters)
    return statistics.median(meds) * 1e6


def try_compile(fn, example):
    """Return a compiled function, or None if torch.compile is unsupported here."""
    try:
        compiled = torch.compile(fn)
        with torch.no_grad():
            for _ in range(3):  # first call triggers compilation
                compiled(example)
        return compiled
    except Exception as e:  # e.g. missing C++ compiler on Windows
        print(f"  [torch.compile unavailable: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:90]}] -> using torch.jit.trace only")
        return None


NOTE = """
NOTE: when does compilation help?
---------------------------------
Eager mode runs one operator at a time. Each op pays a fixed Python /
dispatcher launch cost and writes its result to memory before the next op
reads it. Compilation (torch.compile) or tracing (torch.jit.trace) captures
the whole graph, which lets the framework fuse chains of element-wise ops into
one kernel, remove intermediate memory traffic, and skip Python overhead.

It helps most when: (1) the model has many small / memory-bound element-wise
ops (see the pointwise chain), (2) the same shapes are run many times so the
one-time compile cost is amortised, and (3) the workload is launch-overhead
bound (tiny models, small batches, especially on GPUs).

It helps little, or can hurt, when: the time is dominated by large matmuls /
convolutions that are already highly tuned library calls (see the wide MLP),
the code runs only a few times (compile time > savings), shapes change often
(recompilation), or the code has data-dependent Python control flow
(tracing records only the path taken, so it silently bakes it in).

Non-contiguous views (transposes, strided slices) are free to create but make
later operations slower because memory is no longer read sequentially, which
hurts cache use and vectorisation; calling .contiguous() pays for one copy up
front and is worthwhile only if the tensor is reused several times.
"""


class WideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 10),
        )

    def forward(self, z):
        return self.net(z)


def pointwise_chain(z):  # many cheap element-wise ops -> fusion opportunity
    for _ in range(8):
        z = torch.sin(z) * z + torch.cos(z) - z * 0.5
    return z


def main():
    print("=" * 70)
    print("TASK 6: PyTorch graph inspection, eager vs compiled/traced, contiguity")
    print("=" * 70)
    torch.manual_seed(0)
    print(f"torch version: {torch.__version__}\n")

    # ---- 6a. inspect the graph of a small model ----
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
    x = torch.randn(100, 2)
    out = model(x)
    print("6a. autograd graph of 2->8->1 model (output towards inputs):")
    for line in walk_graph(out.grad_fn):
        print("   ", line)
    print("\n    tensor properties:")
    for name, t in [("input x", x), ("fc1.weight", model[0].weight),
                    ("fc1.bias", model[0].bias), ("output", out)]:
        print(f"      {name:11s} shape={tuple(t.shape)!s:10s} dtype={t.dtype} "
              f"stride={t.stride()!s:9s} contiguous={t.is_contiguous()}")
    traced_small = torch.jit.trace(model, x)
    print("\n    TorchScript graph of the traced model:")
    for line in str(traced_small.graph).splitlines():
        print("     ", line)

    # ---- 6b. eager vs traced vs compiled ----
    print("\n6b. eager vs torch.jit.trace vs torch.compile (median of 5, after warm-up)")
    cases = [
        ("tiny 2->8->1 model (batch 100)", model.eval(), x),
        ("wide MLP 256-512-512-10 (batch 128)", WideMLP().eval(), torch.randn(128, 256)),
        ("pointwise chain (1M elements)", pointwise_chain, torch.randn(1_000_000)),
    ]
    print(f"    {'workload':38s} {'eager us':>10s} {'trace us':>10s} {'compile us':>11s}")
    for name, fn, inp in cases:
        t_eager = bench(fn, inp)
        with torch.no_grad():
            traced = torch.jit.trace(fn, inp)
        t_trace = bench(traced, inp)
        compiled = try_compile(fn, inp)
        comp_txt = f"{bench(compiled, inp):11.1f}" if compiled is not None else f"{'n/a':>11s}"
        print(f"    {name:38s} {t_eager:10.1f} {t_trace:10.1f} {comp_txt}")

    # ---- 6c. non-contiguous tensors ----
    print("\n6c. contiguous vs non-contiguous operands (2048 x 2048, float32)")
    A = torch.randn(2048, 2048)
    B = torch.randn(2048, 2048)
    Bt = B.t()              # transposed view: same storage, swapped strides
    S = B[:, ::2]           # strided slice view
    A_half = A[:, ::2].contiguous()
    print(f"    B  : shape={tuple(B.shape)} stride={B.stride()} contiguous={B.is_contiguous()}")
    print(f"    B.t: shape={tuple(Bt.shape)} stride={Bt.stride()} contiguous={Bt.is_contiguous()}")
    print(f"    B[:, ::2]: shape={tuple(S.shape)} stride={S.stride()} contiguous={S.is_contiguous()}")

    t_c = bench(torch.add, A, B, iters=20)
    t_nc = bench(torch.add, A, Bt, iters=20)
    t_conv = bench(lambda a, b: torch.add(a, b.contiguous()), A, Bt, iters=20)
    print(f"\n    A + B                  (contiguous)      : {t_c:10.0f} us")
    print(f"    A + B.t                (non-contiguous)  : {t_nc:10.0f} us   ({t_nc / t_c:.2f}x)")
    print(f"    A + B.t.contiguous()   (copy first)      : {t_conv:10.0f} us   ({t_conv / t_c:.2f}x)")

    t_rc = bench(torch.relu, A_half, iters=20)
    t_rn = bench(torch.relu, S, iters=20)
    print(f"\n    relu(contiguous 2048x1024)               : {t_rc:10.0f} us")
    print(f"    relu(B[:, ::2] strided view)             : {t_rn:10.0f} us   ({t_rn / t_rc:.2f}x)")

    print(NOTE)


if __name__ == "__main__":
    main()
