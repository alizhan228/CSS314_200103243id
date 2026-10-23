import os
import platform
import statistics
import sys
import threading
import time

THREADS = 4
PER_THREAD = 2_000_000
EXPECTED = THREADS * PER_THREAD
REPEATS = 3


class Counter:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


def _touch(v):
    # A Python-level call: CPython can hand the GIL to another thread here.
    return v


def naive_worker(c: Counter, barrier) -> None:
    barrier.wait()
    for _ in range(PER_THREAD):
        c.value += 1


def racy_worker(c: Counter, barrier) -> None:
    barrier.wait()
    for _ in range(PER_THREAD):
        tmp = c.value            # read
        tmp = _touch(tmp) + 1    # compute (switch point)
        c.value = tmp            # write back, may overwrite another thread's update


def locked_worker(c: Counter, barrier, lock) -> None:
    barrier.wait()
    for _ in range(PER_THREAD):
        with lock:
            tmp = c.value
            tmp = _touch(tmp) + 1
            c.value = tmp


def run(worker, *extra) -> tuple[float, int]:
    """Run THREADS workers; return (wall time in ms, final counter value)."""
    c = Counter()
    barrier = threading.Barrier(THREADS + 1)
    threads = [threading.Thread(target=worker, args=(c, barrier, *extra))
               for _ in range(THREADS)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    return (time.perf_counter() - t0) * 1000.0, c.value


def report_unsafe(label: str, worker) -> float:
    print(f"--- {label} ---")
    times = []
    for r in range(1, REPEATS + 1):
        t_ms, actual = run(worker)
        lost = EXPECTED - actual
        pct = lost / EXPECTED * 100.0
        times.append(t_ms)
        print(f"  run {r}: expected {EXPECTED:,} | actual {actual:,} | "
              f"lost {lost:,} updates ({pct:.4f}%) | {t_ms:.1f} ms")
    med = statistics.median(times)
    print(f"  median time: {med:.1f} ms\n")
    return med


def main() -> None:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    sys.setswitchinterval(1e-6)  # encourage frequent thread switches

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"GIL enabled: {gil} | switch interval: {sys.getswitchinterval()} s")
    print(f"Threads: {THREADS} | increments per thread: {PER_THREAD:,} | "
          f"expected final value: {EXPECTED:,}")
    print(f"Timing: {REPEATS} runs per variant, medians reported\n")

    report_unsafe("1. Naive, no locks: c.value += 1", naive_worker)
    t_racy = report_unsafe("2. Racy, no locks: read -> call -> write", racy_worker)

    print("--- 3. Correct: same body as (2) protected by threading.Lock ---")
    lock = threading.Lock()
    times, all_ok = [], True
    for r in range(1, REPEATS + 1):
        t_ms, actual = run(locked_worker, lock)
        ok = actual == EXPECTED
        all_ok &= ok
        times.append(t_ms)
        print(f"  run {r}: expected {EXPECTED:,} | actual {actual:,} | "
              f"lost {EXPECTED - actual:,} | {t_ms:.1f} ms | {'PASS' if ok else 'FAIL'}")
    t_lock = statistics.median(times)
    print(f"  median time: {t_lock:.1f} ms\n")

    penalty = t_lock - t_racy
    print("=== Synchronisation cost (median locked vs median racy) ===")
    print(f"Racy (no lock): {t_racy:.1f} ms")
    print(f"Locked        : {t_lock:.1f} ms")
    print(f"Wall-clock penalty from the lock: {penalty:+.1f} ms "
          f"({t_lock / t_racy:.2f}x, {penalty / t_racy * 100:+.1f}%)")
    print(f"Locked version correct in all runs: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()