import os
import platform
import queue
import statistics
import threading
import time

N_ITEMS = 1000
WORKERS = 4
UNIT = 0.0001          # seconds per unit of (i % 20)
REPEATS = 5


def work(i: int) -> None:
    d = UNIT * (i % 20)
    if d > 0:
        time.sleep(d)


# ---- chunk sources: each worker calls its source until it returns None ----

def make_static():
    per = N_ITEMS // WORKERS

    def source_for(w):
        given = [False]

        def src():
            if given[0]:
                return None
            given[0] = True
            return (w * per, (w + 1) * per if w < WORKERS - 1 else N_ITEMS)
        return src
    return [source_for(w) for w in range(WORKERS)]


def make_dynamic(chunk: int):
    q = queue.Queue()
    for s in range(0, N_ITEMS, chunk):
        q.put((s, min(s + chunk, N_ITEMS)))

    def src():
        try:
            return q.get_nowait()
        except queue.Empty:
            return None
    return [src] * WORKERS


def run(make_sources, workers: int = WORKERS):
    """Run one schedule; return (total_ms, finish_offsets_ms, items_per_worker)."""
    sources = make_sources()
    finishes = [0.0] * workers
    counts = [0] * workers
    barrier = threading.Barrier(workers + 1)

    def worker(w):
        src = sources[w]
        barrier.wait()
        while (c := src()) is not None:
            for i in range(c[0], c[1]):
                work(i)
                counts[w] += 1
        finishes[w] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    offsets = [(f - t0) * 1000.0 for f in finishes]
    return max(offsets), offsets, counts


def run_serial():
    t0 = time.perf_counter()
    for i in range(N_ITEMS):
        work(i)
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    total_work_ms = sum(UNIT * (i % 20) for i in range(N_ITEMS)) * 1000.0
    ideal_ms = total_work_ms / WORKERS

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Items: {N_ITEMS} | Workers: {WORKERS} | cost(i) = {UNIT} * (i % 20) s")
    print(f"Sum of nominal sleep time: {total_work_ms:.1f} ms | "
          f"ideal balanced time (total/{WORKERS}): {ideal_ms:.1f} ms")
    print(f"Timing: {REPEATS} runs per strategy; the run with the median total is shown\n")

    serial_ms = statistics.median(run_serial() for _ in range(3))
    print(f"Serial reference (1 worker, median of 3): {serial_ms:.1f} ms\n")

    strategies = [
        ("A: Static blocks", make_static),
        ("B: Dynamic, chunk=1", lambda: make_dynamic(1)),
        ("B: Dynamic, chunk=5", lambda: make_dynamic(5)),
    ]

    header = (f"{'Strategy':<22} | {'Total (ms)':>10} | {'First done (ms)':>15} | "
              f"{'Last done (ms)':>14} | {'Delta (ms)':>10}")
    print(header)
    print("-" * len(header))

    details = []
    for name, factory in strategies:
        runs = [run(factory) for _ in range(REPEATS)]
        runs.sort(key=lambda r: r[0])
        total, offsets, counts = runs[len(runs) // 2]   # median-total run
        first, last = min(offsets), max(offsets)
        print(f"{name:<22} | {total:>10.2f} | {first:>15.2f} | "
              f"{last:>14.2f} | {last - first:>10.2f}")
        details.append((name, offsets, counts))

    print("\nPer-worker finish times (ms after start) and items processed:")
    for name, offsets, counts in details:
        fin = ", ".join(f"{o:7.2f}" for o in offsets)
        cnt = ", ".join(f"{c:4d}" for c in counts)
        print(f"  {name:<22} finish: [{fin}]  items: [{cnt}]")

    print("\nDelta = time between the first and the last worker finishing "
          "(load imbalance).")


if __name__ == "__main__":
    main()