

import multiprocessing as mp
import os
import platform
import sys
import time
from multiprocessing import shared_memory

ITERS = 50_000_000
CACHE_LINE = 64
SLOT = 8                       # bytes per int64 counter
PAD_STRIDE = CACHE_LINE // SLOT  # 8 slots = 64 bytes
REPEATS = 3


def worker(shm_name: str, slot: int, mode: str, barrier) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    mv = shm.buf.cast("q")  # int64 view over the shared block
    barrier.wait()          # all workers + parent start together
    if mode == "local":
        local = 0
        for _ in range(ITERS):
            local += 1
        mv[slot] = local    # the only shared write
    else:
        for _ in range(ITERS):
            mv[slot] += 1   # read-modify-write on shared memory
    mv.release()
    shm.close()


def run(mode: str, n: int) -> tuple[float, bool]:
    """Run one variant; return (wall time in ms, all counters correct)."""
    stride = PAD_STRIDE if mode == "padded" else 1
    shm = shared_memory.SharedMemory(create=True, size=n * stride * SLOT)
    mv = shm.buf.cast("q")
    for i in range(n * stride):
        mv[i] = 0

    barrier = mp.Barrier(n + 1)
    procs = [mp.Process(target=worker, args=(shm.name, i * stride, mode, barrier))
             for i in range(n)]
    for p in procs:
        p.start()

    barrier.wait()
    t0 = time.perf_counter()
    for p in procs:
        p.join()
    t1 = time.perf_counter()

    ok = all(mv[i * stride] == ITERS for i in range(n))
    mv.release()
    shm.close()
    shm.unlink()
    return (t1 - t0) * 1000.0, ok


def detect_workers() -> tuple[int, str]:
    if len(sys.argv) > 1:
        return max(1, int(sys.argv[1])), "from command line"
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys:
            return phys, "physical cores via psutil"
    except ImportError:
        pass
    return os.cpu_count() or 1, ("defaulted to LOGICAL cores; pass the physical "
                                 "core count as argv[1] or pip install psutil")


def main() -> None:
    n, source = detect_workers()
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Worker processes (N): {n} ({source})")
    print(f"Increments per worker: {ITERS:,}")
    print(f"Cache line assumed: {CACHE_LINE} bytes | padded stride: {PAD_STRIDE} slots")
    print(f"Timing: best of {REPEATS} runs per variant\n")

    best = {"unpadded": float("inf"), "padded": float("inf"), "local": float("inf")}
    all_ok = True
    for _ in range(REPEATS):
        for mode in best:
            t, ok = run(mode, n)
            all_ok &= ok
            best[mode] = min(best[mode], t)

    print(f"{'Variant':<22} | {'Wall Time (ms)':>14}")
    print("-" * 23 + "+" + "-" * 16)
    print(f"{'Unpadded Shared':<22} | {best['unpadded']:>14.3f}")
    print(f"{'Padded (64-byte)':<22} | {best['padded']:>14.3f}")
    print(f"{'Local variable':<22} | {best['local']:>14.3f}")

    print(f"\nExecution ratio (Unpadded / Padded): {best['unpadded'] / best['padded']:.2f}x slower")
    print(f"Execution ratio (Unpadded / Local):  {best['unpadded'] / best['local']:.2f}x slower")
    print(f"Result check (all counters == {ITERS:,}): {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()