import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
 
import numpy as np
 
# ---- EDIT THIS: same Student ID as Task 1 (RNG seed) ----
STUDENT_ID = 200103243
# ---------------------------------------------------------
 
N = 20_000_000
P_PARALLEL = 0.95   # parallelizable fraction for Amdahl's Law
REPEATS = 7         # timed runs per worker count (median is reported)
 
 
def cpu_model() -> str:
    """Best-effort CPU model string across Linux / macOS / Windows."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"
 
 
def amdahl(p: int, parallel_fraction: float = P_PARALLEL) -> float:
    """Theoretical max speedup: 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p)
 
 
def parallel_sum(data: np.ndarray, pool: ThreadPoolExecutor, workers: int) -> float:
    """Split the array into `workers` contiguous chunks, sum each in its own
    thread (NumPy releases the GIL inside np.sum), then reduce the partials."""
    bounds = np.linspace(0, data.size, workers + 1, dtype=np.int64)
    chunks = [data[bounds[i]:bounds[i + 1]] for i in range(workers)]
    partials = pool.map(np.sum, chunks)
    return float(sum(partials))
 
 
def benchmark(data: np.ndarray, workers: int) -> tuple[float, float]:
    """Return (median wall time in ms, computed sum) for a given worker count."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = parallel_sum(data, pool, workers)  # warm-up (spawns threads)
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            result = parallel_sum(data, pool, workers)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return statistics.median(times), result
 
 
def main() -> None:
    max_cores = os.cpu_count() or 1
 
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU: {cpu_model()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    print(f"Logical cores (os.cpu_count): {max_cores}")
    print(f"Array size: {N:,} float64 elements")
    print(f"Amdahl parallel fraction P = {P_PARALLEL}")
    print(f"Timing: median of {REPEATS} runs per worker count (after 1 warm-up)")
 
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    reference = float(data.sum())
 
    # [1, 2, 4, max_cores, max_cores * 2], de-duplicated, order preserved
    counts = list(dict.fromkeys([1, 2, 4, max_cores, max_cores * 2]))
 
    rows = []
    t1_ms = None
    for p in counts:
        t_ms, result = benchmark(data, p)
        if p == 1:
            t1_ms = t_ms
        speedup = t1_ms / t_ms
        ok = np.isclose(result, reference, rtol=1e-9)
        rows.append((p, t_ms, speedup, amdahl(p), ok))
 
    print()
    header = f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>9} | {'Amdahl Upper Bound':>19}"
    print(header)
    print("-" * len(header))
    for p, t_ms, s, a, ok in rows:
        print(f"{p:>6} | {t_ms:>15.3f} | {s:>8.2f}x | {a:>18.2f}x")
 
    print()
    print(f"Serial reference sum: {reference:.6f}")
    print("Result check (parallel sum matches serial): "
          + ("PASS" if all(r[4] for r in rows) else "FAIL"))
 
 
if __name__ == "__main__":
    main()
 