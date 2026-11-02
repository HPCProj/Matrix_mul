import os
import sys
import time
import json
import logging
import argparse
import platform
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="HPC Matrix Multiplication Benchmark")
    parser.add_argument("--experiments_name", "-experiments_name", type=str, required=True)
    parser.add_argument("--matrix_size", "-matrix_size", type=str, default="1024,2048,4096")
    parser.add_argument("--p_list", "-p_list", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--run_repetition", "-run_repetition", type=int, default=4)
    parser.add_argument("--threading_layer", type=str, default="omp", choices=["omp", "default", "workqueue", "tbb"])
    return parser.parse_args()


def load_or_build_preliminary_config(args):

    matrix_sizes = [int(x.strip()) for x in args.matrix_size.split(",") if x.strip()]
    p_list = [int(x.strip()) for x in args.p_list.split(",") if x.strip()]
    repetitions = args.run_repetition
    threading_layer = args.threading_layer
    return {
        "experiment_name": args.experiments_name,
        "base_results_folder": "results",
        "experiment_dir": os.path.join("results", args.experiments_name),
        "matrix_sizes": matrix_sizes,
        "p_list": p_list,
        "run_repetition": repetitions,
	"requested_threading_layer": threading_layer,
     }


args = parse_arguments()
prelim_config = load_or_build_preliminary_config(args)

max_threads_requested = max(prelim_config["p_list"])
logical_cores = os.cpu_count() or 1
os.environ["NUMBA_NUM_THREADS"] = str(max(max_threads_requested, logical_cores))
if prelim_config["requested_threading_layer"] != "default":
    os.environ["NUMBA_THREADING_LAYER"] = prelim_config["requested_threading_layer"]

import numpy as np
import pandas as pd
import numba
from numba import njit, prange
import matplotlib.pyplot as plt


def setup_logger(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, "experiment.log")

    logger = logging.getLogger("HPC_Experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@njit(parallel=False)
def matmul_serial(A, B, C):
    n, m = A.shape
    p = B.shape[1]
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]


@njit(parallel=True)
def matmul_parallel(A, B, C):
    n, m = A.shape
    p = B.shape[1]
    for i in prange(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]


def warmup_kernels(logger, requested_layer):
    logger.info("Starting Numba JIT warm-up...")
    dummy_A = np.ones((16, 16), dtype=np.float64)
    dummy_B = np.ones((16, 16), dtype=np.float64)
    dummy_C = np.zeros((16, 16), dtype=np.float64)

    matmul_serial(dummy_A, dummy_B, dummy_C)

    try:
        numba.set_num_threads(1)
        matmul_parallel(dummy_A, dummy_B, dummy_C)
        active_layer = numba.threading_layer()
        logger.info(f"Numba warm-up complete. Active threading layer: {active_layer}")
    except ValueError as err:
        logger.warning(f"Requested threading layer '{requested_layer}' unavailable ({err}). Falling back to 'default'.")
        os.environ["NUMBA_THREADING_LAYER"] = "default"
        numba.config.THREADING_LAYER = "default"
        matmul_parallel(dummy_A, dummy_B, dummy_C)
        active_layer = numba.threading_layer()
        logger.info(f"Fallback warm-up complete. Active threading layer: {active_layer}")

    return active_layer


def save_configuration(prelim_cfg, active_layer, logger):
    config_data = {
        "experiment_name": prelim_cfg["experiment_name"],
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": prelim_cfg["experiment_dir"],
        "matrix_sizes": prelim_cfg["matrix_sizes"],
        "p_list": prelim_cfg["p_list"],
        "run_repetition": prelim_cfg["run_repetition"],
        "hardware_and_env": {
            "logical_cpu_cores": os.cpu_count() or 1,
            "max_allocated_numba_threads": int(os.environ["NUMBA_NUM_THREADS"]),
            "requested_threading_layer": prelim_cfg["requested_threading_layer"],
            "active_threading_layer": active_layer,
            "os_system": platform.system(),
            "os_release": platform.release(),
            "machine_arch": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "numba_version": numba.__version__,
            "pandas_version": pd.__version__,
        }
    }

    config_path = os.path.join(prelim_cfg["experiment_dir"], "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    logger.info(f"Configuration saved to {config_path}")
    return config_data


def run_benchmarks(config, logger):
    records = []
    np.random.seed(42)
    logical_cores = config["hardware_and_env"]["logical_cpu_cores"]
    repetitions = config["run_repetition"]

    for n in config["matrix_sizes"]:
        logger.info(f"Initializing matrices for size n = {n} x {n}")
        try:
            A = np.random.rand(n, n).astype(np.float64)
            B = np.random.rand(n, n).astype(np.float64)
            C_serial = np.zeros((n, n), dtype=np.float64)
            C_parallel = np.zeros((n, n), dtype=np.float64)
        except MemoryError:
            logger.error(f"MemoryError for matrix size n = {n}. Skipping.")
            continue

        numpy_runs = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            C_ref = A @ B
            numpy_runs.append(time.perf_counter() - t0)
        median_numpy = float(np.median(numpy_runs))
        logger.info(f"[n={n}] NumPy Reference (A @ B) | Median Time: {median_numpy:.6f}s")

        serial_runs = []
        for _ in range(repetitions):
            C_serial.fill(0.0)
            t0 = time.perf_counter()
            matmul_serial(A, B, C_serial)
            serial_runs.append(time.perf_counter() - t0)

        t_serial = float(np.median(serial_runs))
        serial_correct = bool(np.allclose(C_ref, C_serial))
        logger.info(f"[n={n}] Version 0 (Serial)      | Median Time: {t_serial:.6f}s | Correct: {serial_correct}")

        records.append({
            "experiment_name": config["experiment_name"],
            "version": "NumPy Reference",
            "n": n,
            "p": logical_cores,
            "repetitions": repetitions,
            "median_time_s": median_numpy,
            "min_time_s": float(np.min(numpy_runs)),
            "max_time_s": float(np.max(numpy_runs)),
            "std_time_s": float(np.std(numpy_runs)),
            "speedup": t_serial / median_numpy,
            "efficiency": (t_serial / median_numpy) / logical_cores,
            "correct_allclose": True,
            "raw_times_s": ";".join(f"{t:.6f}" for t in numpy_runs),
        })

        records.append({
            "experiment_name": config["experiment_name"],
            "version": "Version 0 (Serial)",
            "n": n,
            "p": 1,
            "repetitions": repetitions,
            "median_time_s": t_serial,
            "min_time_s": float(np.min(serial_runs)),
            "max_time_s": float(np.max(serial_runs)),
            "std_time_s": float(np.std(serial_runs)),
            "speedup": 1.0,
            "efficiency": 1.0,
            "correct_allclose": serial_correct,
            "raw_times_s": ";".join(f"{t:.6f}" for t in serial_runs),
        })

        for p in config["p_list"]:
            numba.set_num_threads(p)
            parallel_runs = []
            is_correct = False

            for r in range(repetitions):
                C_parallel.fill(0.0)
                t0 = time.perf_counter()
                matmul_parallel(A, B, C_parallel)
                parallel_runs.append(time.perf_counter() - t0)
                if r == 0:
                    is_correct = bool(np.allclose(C_serial, C_parallel))

            t_p = float(np.median(parallel_runs))
            speedup = t_serial / t_p
            efficiency = speedup / p

            logger.info(
                f"[n={n}] Version 1 (Parallel)    | p={p:<3d} | Median Time: {t_p:.6f}s | "
                f"Speedup: {speedup:.4f}x | Efficiency: {efficiency:.4f} | Correct: {is_correct}"
            )

            records.append({
                "experiment_name": config["experiment_name"],
                "version": "Version 1 (Parallel)",
                "n": n,
                "p": p,
                "repetitions": repetitions,
                "median_time_s": t_p,
                "min_time_s": float(np.min(parallel_runs)),
                "max_time_s": float(np.max(parallel_runs)),
                "std_time_s": float(np.std(parallel_runs)),
                "speedup": speedup,
                "efficiency": efficiency,
                "correct_allclose": is_correct,
                "raw_times_s": ";".join(f"{t:.6f}" for t in parallel_runs),
            })

    df = pd.DataFrame(records)
    numeric_cols = ["median_time_s", "min_time_s", "max_time_s", "std_time_s", "speedup", "efficiency"]
    df[numeric_cols] = df[numeric_cols].round(6)
    return df


def generate_plots(df, config, logger):
    parallel_df = df[df["version"] == "Version 1 (Parallel)"].copy()
    if parallel_df.empty:
        logger.warning("No parallel results available to plot.")
        return

    exp_dir = config["experiment_dir"]
    exp_name = config["experiment_name"]
    p_values = config["p_list"]
    cpu_cores = config["hardware_and_env"]["logical_cpu_cores"]

    plt.figure(figsize=(8, 6))
    for n, group in parallel_df.groupby("n"):
        plt.plot(group["p"], group["speedup"], marker="o", linewidth=2, label=f"n = {n}")
    plt.axvline(x=cpu_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({cpu_cores})")
    plt.title(f"Speedup vs. Thread Count ({exp_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Speedup (S = T_serial / T_p)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    speedup_path = os.path.join(exp_dir, "speedup.png")
    plt.savefig(speedup_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for n, group in parallel_df.groupby("n"):
        plt.plot(group["p"], group["efficiency"], marker="s", linewidth=2, label=f"n = {n}")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Efficiency (1.0)")
    plt.axvline(x=cpu_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({cpu_cores})")
    plt.title(f"Efficiency vs. Thread Count ({exp_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Efficiency (E = S / p)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    efficiency_path = os.path.join(exp_dir, "efficiency.png")
    plt.savefig(efficiency_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for n, group in parallel_df.groupby("n"):
        plt.plot(group["p"], group["median_time_s"], marker="^", linewidth=2, label=f"n = {n}")
    plt.axvline(x=cpu_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({cpu_cores})")
    plt.title(f"Median Runtime vs. Thread Count ({exp_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Median Time (s) [Log Scale]")
    plt.yscale("log")
    plt.xticks(p_values)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    runtime_path = os.path.join(exp_dir, "runtime.png")
    plt.savefig(runtime_path, dpi=300)
    plt.close()

    logger.info(f"Plots saved in {exp_dir} (speedup.png, efficiency.png, runtime.png)")


def main():
    logger = setup_logger(prelim_config["experiment_dir"])
    logger.info(f"Starting experiment '{prelim_config['experiment_name']}'")

    active_layer = warmup_kernels(logger, prelim_config["requested_threading_layer"])
    full_config = save_configuration(prelim_config, active_layer, logger)

    results_df = run_benchmarks(full_config, logger)

    csv_path = os.path.join(full_config["experiment_dir"], "results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results DataFrame saved to {csv_path}")

    summary_cols = ["version", "n", "p", "median_time_s", "speedup", "efficiency", "correct_allclose"]
    logger.info("Experiment Summary Table:\n" + results_df[summary_cols].to_string(index=False))

    generate_plots(results_df, full_config, logger)
    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
