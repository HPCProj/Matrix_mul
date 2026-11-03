# Performance Analysis: Distributed-Memory Matrix Multiplication with MPI (Single-Host Evaluation)

**Experiment Name:** `exp_chabane` (MPI Local Cluster Simulation)  
**Platform:** Linux (`6.17.0-35-generic`), 64 Logical CPU Cores  
**Software Stack:** Python 3.10.12, `mpi4py` (Open MPI), NumPy, Numba (`@njit` serial local kernel)  
**Test Parameters:** $n \in \{1024, 2048, 4096\}$, $p \in \{1, 2, 4, 8\}$, 4 repetitions per run.

---

## 1. Parallel Design and Correctness
This implementation uses a **1D row-block decomposition**:
1. **Broadcast (`MPI.Bcast`):** Rank 0 broadcasts matrix $B$ ($n \times n$) to all $p$ processes.
2. **Scatter (`MPI.Scatter`):** Rank 0 divides matrix $A$ into contiguous blocks of $n/p$ rows and distributes them across ranks.
3. **Local Compute:** Each rank computes its local block $C_{\text{loc}} = A_{\text{loc}} \times B$ using the single-threaded Numba kernel (`OMP_NUM_THREADS=1`).
4. **Gather (`MPI.Gather`):** Rank 0 collects all `$C_loc$` blocks into the final matrix $C$.

All configurations returned `Correct: True` when checked against `A @ B` with `np.allclose`.

---

## 2. Summary of Experimental Results

The table below reports the median total time, computation time, communication time (`Bcast + Scatter + Gather`), speedup, and efficiency at the optimal process count ($p = 4$) and at $p = 8$.

| Matrix Size ($n$) | Serial Baseline $T_0$ (s) | Processes ($p$) | Compute Time (s) | Comm Time (s) | Total MPI Time $T_p$ (s) | Speedup ($S$) | Efficiency ($E$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1024** | 0.1401 | **4** | 0.0500 | 0.0285 | **0.0783** | **1.79x** | **45.0%** |
| **1024** | 0.1401 | 8 | 0.0780 | 0.0237 | 0.1253 | 1.12x | 14.0% |
| **2048** | 2.5733 | **4** | 0.7799 | 0.0929 | **0.8962** | **2.87x** | **72.0%** |
| **2048** | 2.5733 | 8 | 0.9251 | 0.2421 | 1.1816 | 2.18x | 27.0% |
| **4096** | 24.2768 | **4** | 6.8214 | 0.4433 | **7.2739** | **3.34x** | **83.0%** |
| **4096** | 24.2768 | 8 | 8.2257 | 2.5409 | 11.4228 | 2.13x | 27.0% |

---

## 3. Scaling Behavior and Problem Size Effect

As shown in `speedup.png` and `efficiency.png`, scaling improves significantly as the matrix size $n$ increases:

* **Small Workload ($n = 1024$):** Speedup peaks at **$1.79\text{x}$** ($E = 45\%$) for $p = 4$. At this small size, communication takes $0.0285\text{s}$, which accounts for **36.4%** of the total execution time ($0.0783\text{s}$).
* **Medium Workload ($n = 2048$):** Peak speedup rises to **$2.87\text{x}$** ($E = 72\%$) at $p = 4$. Communication overhead drops to **10.4%** of the total runtime.
* **Large Workload ($n = 4096$):** Peak speedup reaches **$3.34\text{x}$** ($E = 83\%$) at $p = 4$. Here, communication ($0.4433\text{s}$) represents only **6.1%** of the total runtime ($7.2739\text{s}$).

**Theoretical Explanation:**  
Matrix multiplication requires $2n^3$ floating-point operations, while the data transferred per process grows only as $O(n^2)$. Therefore, larger matrices increase the computation-to-communication ratio, hiding message-passing overhead.

---

## 4. Bottleneck Analysis: Computation vs. Communication (`comm_comp_breakdown.png`)

The stacked bar chart (`comm_comp_breakdown.png` for $n = 4096$) explains why runtime decreases from $p = 1$ to $p = 4$, but degrades at $p = 8$ ($7.27\text{s} \rightarrow 11.42\text{s}$):

1. **Communication Surge at $p = 8$:**  
   Between $p = 4$ and $p = 8$, communication time jumps nearly **$5.7\text{x}$** (from $0.4433\text{s}$ to $2.5409\text{s}$). During `MPI.Bcast(B)`, rank 0 must copy a $128\text{ MB}$ matrix ($4096 \times 4096 \times 8\text{ bytes}$) into the private memory space of every rank. At $p = 8$, simultaneous shared-memory copies (`Scatter`, `Bcast`, and `Gather`) saturate the system memory bus.

2. **Local Compute Slowdown Beyond 4 Processes:**  
   Even though each process handles half as many rows at $p = 8$ ($512$ rows) compared to $p = 4$ ($1024$ rows), local compute time increases from $6.82\text{s}$ to $8.23\text{s}$. This matches our earlier OpenMP finding: the environment reaches its hardware memory-bandwidth or physical vCPU allocation limit at 4 active workers. Running 8 processes simultaneously causes memory contention and CPU time-sharing.
