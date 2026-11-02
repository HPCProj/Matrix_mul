# Performance Analysis: Shared-Memory Matrix Multiplication (Versions 0 & 1)

**Experiment Name:** `exp_chabane`  
**Platform:** Linux (`x86_64`), 64 Logical CPU Cores  
**Software Stack:** Python 3.10.12, NumPy 2.2.6, Numba 0.68.1 (`omp` threading layer)  
**Test Parameters:** $n \in \{1024, 2048, 4096\}$, $p \in \{1, 2, 4, 8, 16, 32, 64, 128\}$, 4 repetitions per run.

---

## 1. Numerical Correctness and Warm-up
Before timing, a JIT warm-up call was executed on small $16 \times 16$ matrices. This isolated the compilation overhead from the benchmark measurements. 

Every parallel execution (`Version 1`) was verified against the serial baseline (`Version 0`) using `np.allclose`. All configurations returned `True`, confirming that parallelizing the outer row loop (`prange(n)`) introduces no race conditions or numerical inaccuracies.

---

## 2. Summary of Experimental Results

The table below highlights the median runtime, peak speedup, and efficiency across the tested matrix sizes.

| Matrix Size ($n$) | Serial Time $T_0$ (s) | Best Parallel Time $T_p$ (s) | Optimal Threads ($p$) | Peak Speedup ($S$) | Efficiency at Peak ($E$) | NumPy `A @ B` (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1024** | 0.1730 | 0.0498 | 4 | 3.47x | 86.8% | 0.6050 |
| **2048** | 2.8189 | 0.8068 | 4 | 3.49x | 87.3% | 1.0514 |
| **4096** | 24.7483 | 7.1938 | 4 | 3.44x | 86.0% | 3.0585 |

---

## 3. Scaling Behavior (Speedup and Efficiency)

### Strong Scaling Region ($p = 1$ to $p = 4$)
Across all matrix sizes, performance scales well from 1 to 4 threads:
* At **$p = 1$**, `Version 1` matches `Version 0` closely ($S \approx 1.00\text{x}$), showing minimal OpenMP overhead for a single thread.
* At **$p = 2$**, speedup reaches **$1.77\text{x} - 2.01\text{x}$** with high efficiency ($88\% - 100\%$).
* At **$p = 4$**, all three matrix sizes reach their peak performance, achieving a consistent speedup of **$\sim 3.44\text{x} - 3.49\text{x}$** and an efficiency of **$\sim 86\% - 87\%$**.

### Saturation and Degradation ($p \ge 8$)
Beyond 4 threads, adding more threads does not reduce execution time:
* For **$n = 2048$** and **$n = 4096$**, runtime plateaus and slowly degrades from $7.19\text{s}$ ($p=4$) to $9.89\text{s}$ ($p=128$).
* For **$n = 1024$**, runtime degrades sharply at $p = 64$ ($0.207\text{s}$, slower than serial), because the computation is too small to amortize the cost of waking and synchronizing 64 OpenMP threads.
* Because speedup plateaus at $\sim 3.5\text{x}$, parallel efficiency ($E = S/p$) drops steeply after $p = 4$, falling below $4\%$ at $p = 64$.

---

## 4. Bottleneck Analysis: Why Does Speedup Peak at $p = 4$?

Even though the system reports 64 logical cores, speedup saturates at 4 threads. Three hardware and algorithmic bottlenecks explain this behavior:

1. **Memory Bandwidth Saturation:**  
   The kernel uses the `i-k-j` loop order: `C[i, j] += a * B[k, j]`. While this order provides good spatial locality (reading rows of $B$ and $C$ contiguously), it does not use cache blocking (tiling). For large $n$, the matrices exceed the CPU cache capacity. Each thread streams data continuously from main RAM. Around 4 to 8 threads, the memory bus reaches maximum bandwidth, and extra cores simply wait for data.

2. **NUMA (Non-Uniform Memory Access) Effects:**  
   On a 64-core server, cores are typically divided across multiple CPU sockets or NUMA nodes. In our code, the main thread initializes matrices $A$, $B$, and $C$ before the parallel region. Due to the Linux "first-touch" memory policy, all matrix pages are allocated on a single NUMA node. When $p > 4$, threads scheduled on distant sockets must fetch data across the inter-socket interconnect, increasing memory latency.

3. **Thread Oversubscription ($p = 128$):**  
   At $p = 128$, the number of threads exceeds the 64 logical cores. The operating system must perform frequent context switches, adding overhead without increasing compute throughput.

---

## 5. Comparison with NumPy BLAS (`A @ B`)

Comparing our Numba kernel against NumPy's built-in `A @ B` reveals two distinct regimes:

* **Small Matrix ($n = 1024$):** Our serial Numba code ($0.173\text{s}$) and 4-thread Numba code ($0.050\text{s}$) are significantly faster than NumPy ($0.605\text{s}$). By default, NumPy spawns up to 64 BLAS threads on this machine. For a $1024 \times 1024$ matrix, thread synchronization overhead across 64 cores dominates the runtime.
* **Large Matrix ($n = 4096$):** NumPy ($3.059\text{s}$) outperforms our best Numba run ($7.194\text{s}$) by **$2.35\text{x}$**. Underlying BLAS libraries (such as OpenBLAS or MKL) use **cache-blocked (tiled) algorithms** and explicit SIMD vectorization. By keeping small sub-blocks in L1/L2 cache, BLAS avoids the memory-bandwidth wall that limits our unblocked Numba loop.

---

## 6. Conclusion
1. The `i-k-j` OpenMP loop in Numba provides accurate results and achieves **$\sim 86\%$ parallel efficiency up to 4 threads**.
2. Simple row-parallel matrix multiplication is **memory-bound**, not compute-bound, on multi-core hardware.
3. Scaling beyond 4 threads requires **cache blocking (loop tiling)** to reuse data in L1/L2 cache and reduce main memory traffic.
