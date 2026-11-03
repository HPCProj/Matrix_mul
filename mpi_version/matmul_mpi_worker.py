import os
import sys
import json
import time
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from numba import njit

@njit(parallel=False)
def matmul_local_kernel(A, B, C):
    n, m = A.shape
    p = B.shape[1]
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = args.size

    # Numba JIT Warmup
    dummy_a = np.ones((8, 8), dtype=np.float64)
    dummy_b = np.ones((8, 8), dtype=np.float64)
    dummy_c = np.zeros((8, 8), dtype=np.float64)
    matmul_local_kernel(dummy_a, dummy_b, dummy_c)

    if args.warmup:
        sys.exit(0)

    if n % size != 0:
        if rank == 0:
            print(json.dumps({"error": f"Size {n} is not divisible by rank count {size}"}))
        sys.exit(1)

    rows_per_rank = n // size

    if rank == 0:
        np.random.seed(42)
        A = np.random.rand(n, n).astype(np.float64)
        B = np.random.rand(n, n).astype(np.float64)
    else:
        A = None
        B = np.empty((n, n), dtype=np.float64)

    A_loc = np.empty((rows_per_rank, n), dtype=np.float64)
    C_loc = np.zeros((rows_per_rank, n), dtype=np.float64)

    comm.Barrier()

    # 1. Broadcast B
    t_bcast_start = MPI.Wtime()
    comm.Bcast(B, root=0)
    t_bcast = MPI.Wtime() - t_bcast_start

    # 2. Scatter A
    t_scatter_start = MPI.Wtime()
    comm.Scatter(A, A_loc, root=0)
    t_scatter = MPI.Wtime() - t_scatter_start

    # 3. Local Computation
    t_comp_start = MPI.Wtime()
    matmul_local_kernel(A_loc, B, C_loc)
    t_comp = MPI.Wtime() - t_comp_start

    # 4. Gather C
    C = np.empty((n, n), dtype=np.float64) if rank == 0 else None
    t_gather_start = MPI.Wtime()
    comm.Gather(C_loc, C, root=0)
    t_gather = MPI.Wtime() - t_gather_start

    comm.Barrier()

    max_t_bcast = comm.reduce(t_bcast, op=MPI.MAX, root=0)
    max_t_scatter = comm.reduce(t_scatter, op=MPI.MAX, root=0)
    max_t_comp = comm.reduce(t_comp, op=MPI.MAX, root=0)
    max_t_gather = comm.reduce(t_gather, op=MPI.MAX, root=0)

    if rank == 0:
        total_time = max_t_bcast + max_t_scatter + max_t_comp + max_t_gather
        is_correct = bool(np.allclose(C, A @ B))
        payload = {
            "bcast_time_s": max_t_bcast,
            "scatter_time_s": max_t_scatter,
            "compute_time_s": max_t_comp,
            "gather_time_s": max_t_gather,
            "total_time_s": total_time,
            "correct": is_correct
        }
        print(f"JSON_RESULT:{json.dumps(payload)}")

if __name__ == "__main__":
    main()
