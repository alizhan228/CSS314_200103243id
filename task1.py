
# Force any underlying native libraries to stay single-threaded.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import platform
import subprocess
import time

import numpy as np

# ---- EDIT THIS: your Student ID (used as the RNG seed) ----
STUDENT_ID = 200103243
# -----------------------------------------------------------

N = 20_000_000


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


def main() -> None:
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU: {cpu_model()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Array size: {N:,} float64 elements")

    # Generate data (not included in the timed region).
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    # ---- Timed serial section: sum + standard deviation ----
    t0 = time.perf_counter()

    total = data.sum()
    mean = total / N
    # Two-pass population standard deviation (ddof = 0)
    std = np.sqrt(np.sum((data - mean) ** 2) / N)

    t1 = time.perf_counter()
    # --------------------------------------------------------

    print(f"Sum: {total:.6f}")
    print(f"Standard deviation (population): {std:.10f}")
    print(f"Execution time: {(t1 - t0) * 1000:.3f} ms")


if __name__ == "__main__":
    main()