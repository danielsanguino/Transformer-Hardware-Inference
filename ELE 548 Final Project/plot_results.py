#!/usr/bin/env python3
"""
Plot CPU vs GPU inference benchmark results.

Usage examples:

# If you saved CPU and GPU results separately:
python plot_results.py \
  --cpu_csv results/results_cpu.csv \
  --gpu_csv results/results_gpu.csv \
  --out_dir results/graphs

# If you only have one combined CSV:
python plot_results.py \
  --combined_csv results/aslfr_inference_results.csv \
  --out_dir results/graphs
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Argument parsing
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots for ASLFR CPU vs GPU inference benchmarks."
    )
    parser.add_argument(
        "--cpu_csv",
        type=str,
        default=None,
        help="Path to CSV with CPU-only results (optional if using --combined_csv).",
    )
    parser.add_argument(
        "--gpu_csv",
        type=str,
        default=None,
        help="Path to CSV with GPU-only results (optional if using --combined_csv).",
    )
    parser.add_argument(
        "--combined_csv",
        type=str,
        default=None,
        help="Path to CSV with both devices mixed in a single file (optional).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/graphs",
        help="Directory to save generated plots.",
    )
    return parser.parse_args()


# ---------------------------
# Data loading / merging
# ---------------------------

def load_results(
    cpu_csv: Optional[str],
    gpu_csv: Optional[str],
    combined_csv: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (cpu_df, gpu_df), each possibly empty if missing.
    Priority:
      1) If combined_csv is provided, split on device column.
      2) Else, use cpu_csv and gpu_csv separately.
    """

    if combined_csv is not None:
        combined_path = Path(combined_csv)
        if not combined_path.exists():
            raise FileNotFoundError(f"combined_csv not found: {combined_path}")
        df = pd.read_csv(combined_path)
        if "device" not in df.columns:
            raise ValueError("combined_csv must have a 'device' column (cpu/gpu).")

        cpu_df = df[df["device"] == "cpu"].copy()
        gpu_df = df[df["device"] == "gpu"].copy()
        print(f"[INFO] Loaded combined CSV: {combined_path}")
        print(f"[INFO] CPU rows: {len(cpu_df)}, GPU rows: {len(gpu_df)}")
        return cpu_df, gpu_df

    # Else, use separate files
    if cpu_csv is None and gpu_csv is None:
        raise ValueError("You must provide either --combined_csv OR at least one of --cpu_csv/--gpu_csv.")

    if cpu_csv is not None:
        cpu_path = Path(cpu_csv)
        if not cpu_path.exists():
            raise FileNotFoundError(f"cpu_csv not found: {cpu_path}")
        cpu_df = pd.read_csv(cpu_path)
        cpu_df["device"] = "cpu"
        print(f"[INFO] Loaded CPU CSV: {cpu_path} ({len(cpu_df)} rows)")
    else:
        cpu_df = pd.DataFrame(columns=["batch_size"])  # empty placeholder
        print("[INFO] No CPU CSV provided.")

    if gpu_csv is not None:
        gpu_path = Path(gpu_csv)
        if not gpu_path.exists():
            raise FileNotFoundError(f"gpu_csv not found: {gpu_path}")
        gpu_df = pd.read_csv(gpu_path)
        gpu_df["device"] = "gpu"
        print(f"[INFO] Loaded GPU CSV: {gpu_path} ({len(gpu_df)} rows)")
    else:
        gpu_df = pd.DataFrame(columns=["batch_size"])  # empty placeholder
        print("[INFO] No GPU CSV provided.")

    return cpu_df, gpu_df


# ---------------------------
# Plot helpers
# ---------------------------

def _ensure_out_dir(out_dir: str) -> Path:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_throughput(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame, out_dir: Path):
    """
    Plot throughput (samples/sec) vs batch size for CPU and GPU.
    """
    plt.figure()
    title = "Throughput vs Batch Size"

    # Sort by batch_size if available
    if not cpu_df.empty and "batch_size" in cpu_df.columns:
        cpu_df_sorted = cpu_df.sort_values("batch_size")
        plt.plot(
            cpu_df_sorted["batch_size"],
            cpu_df_sorted["throughput_samples_per_sec"],
            marker="o",
            linestyle="-",
            label="CPU",
        )

    if not gpu_df.empty and "batch_size" in gpu_df.columns:
        gpu_df_sorted = gpu_df.sort_values("batch_size")
        plt.plot(
            gpu_df_sorted["batch_size"],
            gpu_df_sorted["throughput_samples_per_sec"],
            marker="o",
            linestyle="-",
            label="GPU",
        )

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (samples / second)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = out_dir / "throughput_vs_batch_size.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_latency(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame, out_dir: Path):
    """
    Plot average batch latency (ms) vs batch size for CPU and GPU.
    """
    plt.figure()
    title = "Average Batch Latency vs Batch Size"

    if not cpu_df.empty and "batch_size" in cpu_df.columns:
        cpu_df_sorted = cpu_df.sort_values("batch_size")
        plt.plot(
            cpu_df_sorted["batch_size"],
            cpu_df_sorted["avg_batch_latency_ms"],
            marker="o",
            linestyle="-",
            label="CPU",
        )

    if not gpu_df.empty and "batch_size" in gpu_df.columns:
        gpu_df_sorted = gpu_df.sort_values("batch_size")
        plt.plot(
            gpu_df_sorted["batch_size"],
            gpu_df_sorted["avg_batch_latency_ms"],
            marker="o",
            linestyle="-",
            label="GPU",
        )

    plt.xlabel("Batch size")
    plt.ylabel("Avg batch latency (ms)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = out_dir / "latency_vs_batch_size.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_cpu_utilization(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame, out_dir: Path):
    """
    Plot CPU utilization vs batch size for CPU and GPU runs.
    """
    plt.figure()
    title = "CPU Utilization vs Batch Size"

    if not cpu_df.empty and "batch_size" in cpu_df.columns:
        cpu_df_sorted = cpu_df.sort_values("batch_size")
        plt.plot(
            cpu_df_sorted["batch_size"],
            cpu_df_sorted["cpu_util_percent"],
            marker="o",
            linestyle="-",
            label="CPU",
        )

    if not gpu_df.empty and "batch_size" in gpu_df.columns and "cpu_util_percent" in gpu_df.columns:
        gpu_df_sorted = gpu_df.sort_values("batch_size")
        plt.plot(
            gpu_df_sorted["batch_size"],
            gpu_df_sorted["cpu_util_percent"],
            marker="o",
            linestyle="-",
            label="GPU-run CPU util",
        )

    plt.xlabel("Batch size")
    plt.ylabel("CPU utilization (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = out_dir / "cpu_utilization_vs_batch_size.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_gpu_utilization(gpu_df: pd.DataFrame, out_dir: Path):
    """
    Plot GPU utilization vs batch size for GPU runs (if data available).
    """
    if gpu_df.empty or "gpu_util_percent" not in gpu_df.columns:
        print("[INFO] Skipping GPU utilization plot (no GPU data).")
        return

    plt.figure()
    title = "GPU Utilization vs Batch Size"

    gpu_df_sorted = gpu_df.sort_values("batch_size")
    plt.plot(
        gpu_df_sorted["batch_size"],
        gpu_df_sorted["gpu_util_percent"],
        marker="o",
        linestyle="-",
        label="GPU",
    )

    plt.xlabel("Batch size")
    plt.ylabel("GPU utilization (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = out_dir / "gpu_utilization_vs_batch_size.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    out_dir = _ensure_out_dir(args.out_dir)

    cpu_df, gpu_df = load_results(
        cpu_csv=args.cpu_csv,
        gpu_csv=args.gpu_csv,
        combined_csv=args.combined_csv,
    )

    # If both are empty, nothing to do
    if cpu_df.empty and gpu_df.empty:
        print("[WARN] No data to plot. Check your CSV paths.")
        return

    # Generate plots
    plot_throughput(cpu_df, gpu_df, out_dir)
    plot_latency(cpu_df, gpu_df, out_dir)
    plot_cpu_utilization(cpu_df, gpu_df, out_dir)
    plot_gpu_utilization(gpu_df, out_dir)

    print(f"[INFO] All plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
