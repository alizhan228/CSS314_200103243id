import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError:
    print("This script needs Numba:  pip install numba")
    sys.exit(1)


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


# ======================================================================
# GPU MODE: vector dot product, block-size sweep
# ======================================================================
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
GPU_N = 10_000_000
GPU_MAX_BLOCKS = 8192     # grid-stride loop covers the rest
GPU_RUNS = 20


def make_dot_kernel(tpb: int):
    from numba import cuda

    @cuda.jit
    def dot(x, y, out):
        sh = cuda.shared.array(tpb, dtype=numba.float64)  # tpb is a compile-time constant
        tid = cuda.threadIdx.x
        i = cuda.grid(1)
        stride = cuda.gridsize(1)
        acc = 0.0
        while i < x.size:
            acc += x[i] * y[i]
            i += stride
        sh[tid] = acc
        cuda.syncthreads()
        s = tpb // 2
        while s > 0:
            if tid < s:
                sh[tid] += sh[tid + s]
            cuda.syncthreads()
            s //= 2
        if tid == 0:
            cuda.atomic.add(out, 0, sh[0])

    return dot


def gpu_mode() -> None:
    from numba import cuda

    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    print(f"GPU: {name} | compute capability {dev.compute_capability} | "
          f"SMs: {dev.MULTIPROCESSOR_COUNT} | max threads/block: {dev.MAX_THREADS_PER_BLOCK}")
    print(f"Vector length: {GPU_N:,} float64 | max blocks launched: {GPU_MAX_BLOCKS} "
          f"(grid-stride loop) | timing: median of {GPU_RUNS} runs (CUDA events)\n")

    rng = np.random.default_rng(0)
    x = rng.random(GPU_N)
    y = rng.random(GPU_N)
    reference = float(np.dot(x, y))
    d_x, d_y = cuda.to_device(x), cuda.to_device(y)
    d_out = cuda.to_device(np.zeros(1))
    zero = np.zeros(1)

    hdr = (f"{'Block size':>10} | {'Blocks':>7} | {'Latency (ms)':>12} | "
           f"{'Eff. BW (GB/s)':>14} | {'Check':>5}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for tpb in BLOCK_SIZES:
        kernel = make_dot_kernel(tpb)
        blocks = min((GPU_N + tpb - 1) // tpb, GPU_MAX_BLOCKS)

        d_out.copy_to_device(zero)
        kernel[blocks, tpb](d_x, d_y, d_out)   # warm-up / JIT compile
        cuda.synchronize()

        times = []
        ok = True
        for _ in range(GPU_RUNS):
            d_out.copy_to_device(zero)
            start, end = cuda.event(), cuda.event()
            start.record()
            kernel[blocks, tpb](d_x, d_y, d_out)
            end.record()
            end.synchronize()
            times.append(cuda.event_elapsed_time(start, end))
            ok &= abs(float(d_out.copy_to_host()[0]) - reference) <= 1e-9 * abs(reference)
        t = statistics.median(times)
        bw = 2 * GPU_N * 8 / (t / 1000.0) / 1e9
        results.append((tpb, t))
        print(f"{tpb:>10} | {blocks:>7} | {t:>12.4f} | {bw:>14.1f} | "
              f"{'PASS' if ok else 'FAIL':>5}")

    best = min(results, key=lambda r: r[1])
    worst = max(results, key=lambda r: r[1])
    print(f"\nBest geometry : {best[0]} threads/block ({best[1]:.4f} ms)")
    print(f"Worst geometry: {worst[0]} threads/block ({worst[1]:.4f} ms) "
          f"= {worst[1] / best[1]:.2f}x slower than best")
    print("Latency is kernel time only (CUDA events); host<->device copies excluded.")


# ======================================================================
# CPU MODE: tiled matrix multiplication
# ======================================================================
MAT_N = 1024
TILES = [16, 64]
CPU_RUNS = 3


@njit(parallel=True)
def matmul_naive(A, B, C, n):
    # ijk order: the inner loop walks B down a column (stride n * 8 bytes),
    # which is cache-hostile. Parallel over rows of C.
    for i in prange(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit(parallel=True)
def matmul_tiled(A, B, C, n, bs):
    # Blocked: three bs x bs tiles are reused while hot in cache; inner loop is
    # unit-stride on B and C. Parallel over tile-rows of C (no write conflicts).
    nb = (n + bs - 1) // bs
    for bi in prange(nb):
        i0 = bi * bs
        i1 = min(i0 + bs, n)
        for i in range(i0, i1):
            for j in range(n):
                C[i, j] = 0.0
        for bk in range(nb):
            k0 = bk * bs
            k1 = min(k0 + bs, n)
            for bj in range(nb):
                j0 = bj * bs
                j1 = min(j0 + bs, n)
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        a = A[i, k]
                        for j in range(j0, j1):
                            C[i, j] += a * B[k, j]


def time_ms(fn) -> float:
    ts = []
    for _ in range(CPU_RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def cpu_mode() -> None:
    max_threads = numba.config.NUMBA_NUM_THREADS
    thread_counts = list(dict.fromkeys([1, max_threads]))

    print(f"Matrix size: {MAT_N} x {MAT_N} float64 | tiles tested: "
          f"{', '.join(f'{b}x{b}' for b in TILES)} | thread counts: {thread_counts}")
    print(f"Timing: median of {CPU_RUNS} runs after JIT warm-up\n")

    rng = np.random.default_rng(0)
    A = rng.random((MAT_N, MAT_N))
    B = rng.random((MAT_N, MAT_N))
    C = np.empty((MAT_N, MAT_N))
    ref = A @ B

    # JIT warm-up on a small problem (compiles both kernels)
    a_s, b_s, c_s = A[:64, :64].copy(), B[:64, :64].copy(), np.empty((64, 64))
    matmul_naive(a_s, b_s, c_s, 64)
    matmul_tiled(a_s, b_s, c_s, 64, 16)

    macs = MAT_N ** 3
    all_ok = True
    for t in thread_counts:
        numba.set_num_threads(t)
        print(f"--- {t} thread(s) ---")
        hdr = (f"{'Variant':<14} | {'Time (ms)':>10} | {'GFLOP/s':>8} | "
               f"{'ns/MAC':>7} | {'vs naive':>8} | {'Tile set (3 tiles)':>18}")
        print(hdr)
        print("-" * len(hdr))

        variants = [("Non-tiled", lambda: matmul_naive(A, B, C, MAT_N), "-")]
        for bs in TILES:
            variants.append((f"Tiled {bs}x{bs}",
                             (lambda bs=bs: matmul_tiled(A, B, C, MAT_N, bs)),
                             f"{3 * bs * bs * 8 / 1024:.1f} KiB"))

        t_naive = None
        for name, fn, footprint in variants:
            ms = time_ms(fn)
            fn()  # leave C holding this variant's result for the check below
            ok = np.allclose(C, ref, rtol=1e-8)
            all_ok &= ok
            if t_naive is None:
                t_naive = ms
            gflops = 2 * macs / (ms / 1000.0) / 1e9
            ns_mac = ms * 1e6 / macs
            print(f"{name:<14} | {ms:>10.1f} | {gflops:>8.2f} | {ns_mac:>7.3f} | "
                  f"{t_naive / ms:>7.2f}x | {footprint:>18}")
        print()

    numba.set_num_threads(max_threads)
    blas_ms = time_ms(lambda: np.dot(A, B, out=C))
    print(f"Reference: NumPy/BLAS A @ B ({max_threads} threads available): "
          f"{blas_ms:.1f} ms")
    print(f"Result check (all variants match NumPy): {'PASS' if all_ok else 'FAIL'}\n")

    print("Reading the results:")
    print("  - ns/MAC is time per multiply-add. The extra ns/MAC of the non-tiled")
    print("    variant over the best tiled one is, in effect, the time lost to cache-miss")
    print("    stalls (B is read down a column with an 8 KiB stride).")
    print("  - Tile set = bytes touched by one tile step (3 tiles of 8-byte doubles).")
    print("    16x16 (6 KiB) fits in a typical 32-48 KiB L1d; 64x64 (96 KiB) spills")
    print("    to L2, so it can be slower than 16x16 on some CPUs and faster on others.")
    print("  - Hardware miss counters are not accessible from Python. On Linux you can")
    print("    confirm with:  perf stat -e cache-misses,cache-references python task8.py --cpu")


def main() -> None:
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {cpu_model()}")
    print(f"Platform: {platform.platform()} | Python {platform.python_version()} "
          f"| NumPy {np.__version__} | Numba {numba.__version__}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")

    use_gpu = False
    if "--cpu" not in sys.argv:
        try:
            from numba import cuda
            use_gpu = cuda.is_available()
        except Exception:
            use_gpu = False

    if use_gpu:
        print("Mode: GPU (CUDA vector dot product, block-size sweep)\n")
        gpu_mode()
    else:
        reason = "forced with --cpu" if "--cpu" in sys.argv else "no usable CUDA GPU detected"
        print(f"Mode: CPU (tiled matrix multiplication; {reason})\n")
        cpu_mode()


if __name__ == "__main__":
    main()
