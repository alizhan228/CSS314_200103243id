import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = 30_000_000
MIN_RUNS = 5
MIN_BUDGET_S = 1.0
MAX_RUNS = 50

# Verdict thresholds (efficiency at max cores)
STALL_EFF = 0.60    # below this the kernel is considered to have stalled
GAP = 0.15          # the other kernel must be at least this much more efficient


def cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def bench(fn) -> float:
    """Median wall time of fn() in milliseconds."""
    fn()  # warm-up (page faults, thread start)
    times, spent = [], 0.0
    while (len(times) < MIN_RUNS or spent < MIN_BUDGET_S) and len(times) < MAX_RUNS:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        spent += dt
    return statistics.median(times) * 1000.0


def bounds_for(p: int) -> np.ndarray:
    return np.linspace(0, N, p + 1, dtype=np.int64)


def make_copy_kernel(A, B, p, pool):
    bd = bounds_for(p)

    def chunk(k):
        lo, hi = bd[k], bd[k + 1]
        np.copyto(B[lo:hi], A[lo:hi])

    return lambda: list(pool.map(chunk, range(p)))


def make_trig_kernel(A, B, p, pool):
    bd = bounds_for(p)
    s1 = [np.empty(bd[k + 1] - bd[k]) for k in range(p)]
    s2 = [np.empty(bd[k + 1] - bd[k]) for k in range(p)]

    def chunk(k):
        lo, hi = bd[k], bd[k + 1]
        a = A[lo:hi]
        np.sin(a, out=s1[k])
        np.cos(a, out=s2[k])
        np.multiply(s1[k], s2[k], out=s1[k])
        np.tan(a, out=s2[k])
        np.add(s1[k], s2[k], out=B[lo:hi])

    return lambda: list(pool.map(chunk, range(p)))


def run_kernel(name, factory, A, B, counts, check):
    results = []  # (p, time_ms)
    for p in counts:
        B.fill(0.0)
        with ThreadPoolExecutor(max_workers=p) as pool:
            fn = factory(A, B, p, pool)
            t = bench(fn)
        if not check():
            print(f"  [FAIL] {name}: result mismatch at {p} workers")
        results.append((p, t))
    return results


def table(name, results, bytes_moved=None):
    t1 = results[0][1]
    hdr = f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10}"
    if bytes_moved:
        hdr += f" | {'Bandwidth (GB/s)':>16}"
    print(name)
    print(hdr)
    print("-" * len(hdr))
    out = []
    for p, t in results:
        s = t1 / t
        e = s / p
        line = f"{p:>6} | {t:>10.2f} | {s:>7.2f}x | {e:>10.2f}"
        if bytes_moved:
            line += f" | {bytes_moved / (t / 1000.0) / 1e9:>16.2f}"
        print(line)
        out.append((p, t, s, e))
    print()
    return out


def main() -> None:
    max_cores = os.cpu_count() or 1
    counts = list(dict.fromkeys([1, 2, max_cores]))

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {cpu_model()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()} "
          f"| NumPy {np.__version__}")
    print(f"Logical cores (os.cpu_count): {max_cores} | worker counts tested: {counts}")
    print(f"Array size: {N:,} float64 elements ({N * 8 / 1e6:.0f} MB per array)")
    print(f"Timing: median of >= {MIN_RUNS} runs after warm-up\n")

    rng = np.random.default_rng(0)
    A = rng.uniform(0.1, 1.4, N)   # keeps tan() well-behaved
    B = np.empty(N)
    B.fill(0.0)

    # --- Kernel A: copy ---
    res_a = run_kernel("A", make_copy_kernel, A, B, counts,
                       check=lambda: np.array_equal(A, B))
    # --- Kernel B: trig ---
    probe = slice(0, 1_000_000)
    expected = np.sin(A[probe]) * np.cos(A[probe]) + np.tan(A[probe])
    res_b = run_kernel("B", make_trig_kernel, A, B, counts,
                       check=lambda: np.allclose(B[probe], expected, rtol=1e-12))

    # Copy moves N*8 bytes read + N*8 bytes written (write-allocate not counted).
    rows_a = table("Kernel A (memory-bound): B[i] = A[i]", res_a, bytes_moved=2 * N * 8)
    rows_b = table("Kernel B (compute-bound): B[i] = sin(A[i])*cos(A[i]) + tan(A[i])", res_b)

    # --- Verdict ---
    pa, ta, sa, ea = rows_a[-1]
    pb, tb, sb, eb = rows_b[-1]
    print("=== Verdict ===")
    print(f"At {pa} cores: Kernel A speedup {sa:.2f}x (efficiency {ea:.2f}); "
          f"Kernel B speedup {sb:.2f}x (efficiency {eb:.2f})")
    if pa == 1:
        print("Only 1 logical core available; scaling cannot be assessed.")
    elif ea < STALL_EFF and eb - ea >= GAP:
        print("RESULT: Kernel A (vector copy) STALLED in scaling due to memory "
              "channel / bandwidth saturation.")
        print("        Kernel B (sin/cos/tan) kept scaling: it is bound by CPU compute.")
    elif eb < STALL_EFF and ea - eb >= GAP:
        print("RESULT: Kernel B stalled more than Kernel A. This is unexpected for a "
              "compute-bound kernel; likely causes are SMT (logical cores sharing "
              "physical cores), thermal or power limits, or background load.")
    elif ea < STALL_EFF and eb < STALL_EFF:
        print("RESULT: Both kernels scaled poorly. Kernel A is memory-bandwidth "
              "saturated; Kernel B is probably limited by SMT (logical cores sharing "
              "physical cores) or power/thermal limits.")
    else:
        print("RESULT: Neither kernel stalled below the efficiency threshold "
              f"({STALL_EFF:.2f}); memory bandwidth was not saturated at this core count.")
    print(f"(Rule: a kernel 'stalls' if its efficiency at max cores is < {STALL_EFF:.2f} "
          f"and the other kernel's is at least {GAP:.2f} higher.)")
    print("Note: efficiency is relative to LOGICAL cores; on SMT CPUs even "
          "compute-bound code tops out below 1.0.")


if __name__ == "__main__":
    main()