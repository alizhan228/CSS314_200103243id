import math
import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

A, B = 2.5, 1.0
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
MIN_RUNS = 7
MIN_BUDGET_S = 0.25     # keep sampling each measurement for at least this long
MAX_RUNS = 5000
REFINE_TOLERANCE = 0.01  # stop bisecting when hi - lo <= 1% of lo


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
    """Median wall time of fn() in seconds (adaptive number of runs)."""
    fn(); fn()  # warm-up
    times, spent = [], 0.0
    while (len(times) < MIN_RUNS or spent < MIN_BUDGET_S) and len(times) < MAX_RUNS:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        spent += dt
    return statistics.median(times)


def measure(n: int, pool: ThreadPoolExecutor, workers: int):
    """Return (serial_s, parallel_s, results_match) for array size n."""
    rng = np.random.default_rng(0)
    x = rng.random(n)
    y_ser = np.empty(n)
    y_par = np.empty(n)

    def serial():
        np.multiply(x, A, out=y_ser)
        np.add(y_ser, B, out=y_ser)

    w = min(workers, n)
    bounds = np.linspace(0, n, w + 1, dtype=np.int64)

    def chunk(k):
        lo, hi = bounds[k], bounds[k + 1]
        out = y_par[lo:hi]
        np.multiply(x[lo:hi], A, out=out)
        np.add(out, B, out=out)

    def parallel():
        list(pool.map(chunk, range(w)))

    t_ser = bench(serial)
    t_par = bench(parallel)
    return t_ser, t_par, bool(np.array_equal(y_ser, y_par))


def main() -> None:
    workers = os.cpu_count() or 1
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {cpu_model()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()} "
          f"| NumPy {np.__version__}")
    print(f"Logical cores used (os.cpu_count): {workers}")
    print(f"Kernel: y[i] = {A} * x[i] + {B} (float64)")
    print(f"Timing: median of >= {MIN_RUNS} runs, >= {MIN_BUDGET_S}s sampling per measurement\n")

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        header = (f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>13} | "
                  f"{'Speedup':>8} | {'Winner':>8}")
        print(header)
        print("-" * len(header))
        all_ok = True
        for n in SIZES:
            t_ser, t_par, ok = measure(n, pool, workers)
            all_ok &= ok
            s = t_ser / t_par
            rows.append((n, t_ser, t_par, s))
            print(f"{n:>12,} | {t_ser * 1000:>12.4f} | {t_par * 1000:>13.4f} | "
                  f"{s:>7.2f}x | {'parallel' if s > 1 else 'serial':>8}")
        print(f"\nResult check (parallel output == serial output): {'PASS' if all_ok else 'FAIL'}")

        # ---- Crossover analysis ----
        print("\n=== Crossover analysis ===")
        slower = [n for n, _, _, s in rows if s <= 1.0]
        if not slower:
            print(f"Parallel was FASTER than serial at every tested N "
                  f"(from N = {SIZES[0]:,}); no crossover in the tested range.")
            return
        last_slower = max(slower)
        if last_slower == SIZES[-1]:
            print(f"Parallel was never faster than serial up to N = {SIZES[-1]:,} "
                  f"on this machine.")
            return
        first_faster = SIZES[SIZES.index(last_slower) + 1]
        print(f"Tested grid: parallel SLOWER at N = {last_slower:,}, "
              f"FASTER at N = {first_faster:,} (and all larger tested N).")
        print(f"Crossover lies in the range: {last_slower:,} < N <= {first_faster:,}")

        # ---- Refine by bisection (geometric midpoint) ----
        lo, hi = last_slower, first_faster
        for _ in range(40):
            if hi - lo <= max(1, int(lo * REFINE_TOLERANCE)):
                break
            mid = int(math.sqrt(lo * hi))
            if mid <= lo or mid >= hi:
                break
            t_ser, t_par, _ = measure(mid, pool, workers)
            if t_ser / t_par > 1.0:
                hi = mid
            else:
                lo = mid
        print(f"Refined tipping point (bisection, +/-{REFINE_TOLERANCE:.0%}): "
              f"parallel slower at N = {lo:,}, faster at N = {hi:,}")
        print(f"==> Multi-threading becomes faster than serial at N ~= {hi:,} "
              f"on this hardware.")
        print("Note: measurements are noisy near the crossover; "
              "re-run on an idle machine to confirm.")


if __name__ == "__main__":
    main()