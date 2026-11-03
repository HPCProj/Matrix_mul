import os
import sys
import time
import json
import logging
import platform
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

def parse_args():
    parser = argparse.ArgumentParser(description="HPC MPI Matrix Multiplication Benchmark")
    parser.add_argument("--experiments_name", "-experiments_name", type=str, required=True)
    parser.add_argument("--results_folder", "-results_folder", type=str, default="results")
    parser.add_argument("--matrix_size", "-matrix_size", type=str, default="1024,2048,4096")
    parser.add_argument("--p_list", "-p_list", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--run_repetition", "-run_repetition", type=int, default=4)
    return parser.parse_args()

args = parse_args()

p_values = [int(p.strip()) for p in args.p_list.split(",") if p.strip()]
matrix_sizes = [int(n.strip()) for n in args.matrix_size.split(",") if n.strip()]

target_dir = os.path.join(args.results_folder, args.experiments_name)
os.makedirs(target_dir, exist_ok=True)

logger = logging.getLogger("MPI_Benchmark")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(os.path.join(target_dir, "experiment.log"), mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

@njit(parallel=False)
def serial_matmul(A, B, C):
    n, m = A.shape
    p = B.shape[1]
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]

def get_serial_baseline(n, repetitions):
    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    serial_matmul(A[:4, :4], B[:4, :4], C[:4, :4])
    
    times = []
    for _ in range(repetitions):
        C.fill(0.0)
        t0 = time.perf_counter()
        serial_matmul(A, B, C)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def run_mpi_process(p, n):
    cmd = ["mpirun", "--oversubscribe", "-n", str(p), sys.executable, "matmul_mpi_worker.py", "--size", str(n)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"mpirun error (p={p}, n={n}): {result.stderr.strip()}")
        return None
    for line in result.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line.replace("JSON_RESULT:", ""))
    return None

def save_config():
    config = {
        "experiment_name": args.experiments_name,
        "results_folder": args.results_folder,
        "matrix_sizes": matrix_sizes,
        "p_list": p_values,
        "run_repetition": args.run_repetition,
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_cores_logical": os.cpu_count(),
            "python": platform.python_version()
        }
    }
    with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration metadata stored in config.json")

def generate_plots(df):
    logical_cores = os.cpu_count() or 1

    # 1. Speedup Plot
    plt.figure(figsize=(8, 6))
    for n, grp in df.groupby("n"):
        plt.plot(grp["p"], grp["speedup"], marker="o", linewidth=2, label=f"n = {n}")
    plt.axvline(x=logical_cores, color="red", linestyle="--", alpha=0.7, label=f"Cores ({logical_cores})")
    plt.title(f"MPI Speedup vs Processes - {args.experiments_name}")
    plt.xlabel("Processes (p)")
    plt.ylabel("Speedup (S)")
    plt.xticks(p_values)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "speedup.png"), dpi=300)
    plt.close()

    # 2. Efficiency Plot
    plt.figure(figsize=(8, 6))
    for n, grp in df.groupby("n"):
        plt.plot(grp["p"], grp["efficiency"], marker="s", linewidth=2, label=f"n = {n}")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Efficiency")
    plt.axvline(x=logical_cores, color="red", linestyle="--", alpha=0.7, label=f"Cores ({logical_cores})")
    plt.title(f"MPI Efficiency vs Processes - {args.experiments_name}")
    plt.xlabel("Processes (p)")
    plt.ylabel("Efficiency (E)")
    plt.xticks(p_values)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "efficiency.png"), dpi=300)
    plt.close()

    # 3. Communication vs Computation Breakdown (Max matrix size)
    max_n = max(matrix_sizes)
    df_max = df[df["n"] == max_n]
    if not df_max.empty:
        plt.figure(figsize=(10, 6))
        p_str = [str(p) for p in df_max["p"]]
        comp = df_max["compute_time_s"].values
        comm = (df_max["bcast_time_s"] + df_max["scatter_time_s"] + df_max["gather_time_s"]).values
        
        plt.bar(p_str, comp, label="Computation (Local Kernel)", color="#2b5c8f")
        plt.bar(p_str, comm, bottom=comp, label="Communication (Bcast+Scatter+Gather)", color="#d95f02")
        plt.title(f"Time Breakdown: Comp vs Comm (n = {max_n})")
        plt.xlabel("Processes (p)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "comm_comp_breakdown.png"), dpi=300)
        plt.close()

    logger.info("Generated plots: speedup.png, efficiency.png, comm_comp_breakdown.png")

def main():
    save_config()
    subprocess.run(["mpirun", "-n", "1", sys.executable, "matmul_mpi_worker.py", "--size", "16", "--warmup"])

    records = []
    for n in matrix_sizes:
        logger.info(f"Computing Serial Baseline for n = {n}...")
        t_serial = get_serial_baseline(n, args.run_repetition)
        logger.info(f"[Baseline] Serial median runtime: {t_serial:.4f}s")

        for p in p_values:
            if n % p != 0:
                logger.warning(f"Skipping n={n}, p={p} because {n} is not divisible by {p}.")
                continue

            runs = []
            all_valid = True
            for _ in range(args.run_repetition):
                res = run_mpi_process(p, n)
                if res and res["correct"]:
                    runs.append(res)
                else:
                    all_valid = False

            if not runs:
                logger.error(f"Failed all executions for n={n}, p={p}")
                continue

            med_total = float(np.median([r["total_time_s"] for r in runs]))
            med_comp = float(np.median([r["compute_time_s"] for r in runs]))
            med_bcast = float(np.median([r["bcast_time_s"] for r in runs]))
            med_scatter = float(np.median([r["scatter_time_s"] for r in runs]))
            med_gather = float(np.median([r["gather_time_s"] for r in runs]))

            speedup = t_serial / med_total
            efficiency = speedup / p

            logger.info(
                f"[MPI] n={n:<4d} | p={p:<3d} | Total: {med_total:.4f}s | "
                f"Comp: {med_comp:.4f}s | Comm: {(med_bcast+med_scatter+med_gather):.4f}s | "
                f"Speedup: {speedup:.2f}x | Eff: {efficiency:.2f} | Correct: {all_valid}"
            )

            records.append({
                "n": n,
                "p": p,
                "serial_time_s": t_serial,
                "total_time_s": med_total,
                "compute_time_s": med_comp,
                "bcast_time_s": med_bcast,
                "scatter_time_s": med_scatter,
                "gather_time_s": med_gather,
                "speedup": speedup,
                "efficiency": efficiency,
                "correct": all_valid
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(target_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    generate_plots(df)
    logger.info("MPI Experiment completed successfully.")

if __name__ == "__main__":
    main()
