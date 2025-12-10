#!/usr/bin/env python3
"""
ASLFR CPU vs GPU inference benchmark.

Usage examples (from repo root):

# CPU run (uses defaults in results/)
python aslfr_inference_benchmark.py --device cpu --data_csv results/aslfr_test.csv

# GPU run with custom output path
python aslfr_inference_benchmark.py \
  --device gpu \
  --data_csv results/aslfr_test.csv \
  --output_csv results/results_gpu.csv
"""

import os
import argparse
import time
import json
import subprocess
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


# ---------------------------
# Argument parsing
# ---------------------------

def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"

    parser = argparse.ArgumentParser(
        description="Benchmark ASLFR Transformer inference on CPU vs GPU."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        required=True,
        help="Which device to benchmark on."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(results_dir / "model.h5"),
        help=(
            "Path to model. This can be either:\n"
            "- A legacy .h5 / .keras Keras model file, or\n"
            "- A SavedModel directory (e.g., results/model_full_tf)."
        ),
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="Path to CSV file with preprocessed ASLFR features."
    )
    parser.add_argument(
        "--inference_args",
        type=str,
        default=str(results_dir / "inference_args.json"),
        help="Path to inference_args.json (with selected_columns, n_frames, n_cols). "
             "Defaults to results/inference_args.json next to this script."
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help="List of batch sizes to test."
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=50,
        help="Number of timed batches per batch size."
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup batches (not timed)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(results_dir / "aslfr_inference_results.csv"),
        help="Where to save the results table. "
             "Defaults to results/aslfr_inference_results.csv next to this script."
    )
    return parser.parse_args()


# ---------------------------
# Path helpers
# ---------------------------

def resolve_existing_path(path_str: str, *fallback_dirs: Path) -> Path:
    """
    Resolve a user-supplied path, trying CWD first and then fallbacks.
    Raises FileNotFoundError with a helpful message if not found.
    """
    candidate = Path(path_str).expanduser()

    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.append(Path.cwd() / candidate)
        for base in fallback_dirs:
            search_paths.append(base / candidate)

    for p in search_paths:
        if p.exists():
            return p.resolve()

    tried = ", ".join(str(p) for p in search_paths)
    raise FileNotFoundError(f"Could not find path for '{path_str}'. Tried: {tried}")


# ---------------------------
# System metrics helpers
# ---------------------------

def get_cpu_metrics() -> Dict[str, float]:
    """
    Approximate CPU utilization and memory usage.
    """
    cpu_util = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    mem_used_gb = vm.used / (1024 ** 3)
    return {
        "cpu_util_percent": cpu_util,
        "cpu_mem_used_gb": mem_used_gb,
    }


def get_gpu_metrics() -> Dict[str, float]:
    """
    Query GPU utilization and memory usage via nvidia-smi, if available.
    Returns NaNs if not available.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        first_line = out.splitlines()[0]
        util_str, mem_str = [x.strip() for x in first_line.split(",")]
        gpu_util = float(util_str)
        gpu_mem_used_mb = float(mem_str)
        gpu_mem_used_gb = gpu_mem_used_mb / 1024.0
    except Exception:
        gpu_util = float("nan")
        gpu_mem_used_gb = float("nan")

    return {
        "gpu_util_percent": gpu_util,
        "gpu_mem_used_gb": gpu_mem_used_gb,
    }


# ---------------------------
# Data loading
# ---------------------------

def load_features(
    data_csv: Union[str, Path],
    inference_args_path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load preprocessed features from CSV using selected_columns in inference_args.json.

    Supports:
      - 2D flattened features (N, D)
      - 3D features serialized as flat but with (n_frames, n_cols) metadata

    If inference_args.json contains "n_frames" and "n_cols", we will reshape
    from (N, D_flat) to (N, n_frames, n_cols) where D_flat = n_frames * n_cols.
    """
    with open(inference_args_path, "r") as f:
        inf_args = json.load(f)

    selected_columns: List[str] = inf_args["selected_columns"]
    n_frames = inf_args.get("n_frames", None)
    n_cols = inf_args.get("n_cols", None)

    print(f"[INFO] Loading data from {data_csv}")
    df = pd.read_csv(data_csv)

    missing = [c for c in selected_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following selected_columns are missing in {data_csv}: {missing}"
        )

    X_flat = df[selected_columns].to_numpy(dtype="float32")
    print(f"[INFO] Loaded flat feature matrix X_flat with shape {X_flat.shape}")

    if n_frames is not None and n_cols is not None:
        expected_dim = n_frames * n_cols
        if X_flat.shape[1] != expected_dim:
            raise ValueError(
                f"[ERROR] Flat feature dim {X_flat.shape[1]} does not match "
                f"n_frames * n_cols = {n_frames} * {n_cols} = {expected_dim}"
            )
        X = X_flat.reshape(X_flat.shape[0], n_frames, n_cols)
        print(f"[INFO] Reshaped features to 3D tensor with shape {X.shape}")
    else:
        X = X_flat

    return X, inf_args


# ---------------------------
# Model loading helper
# ---------------------------

def load_aslfr_inference_callable(model_path: str):
    """
    Returns a callable `infer_fn(batch_np)` that runs the ASLFR model on a batch.

    - If `model_path` is a SavedModel directory, we use `tf.saved_model.load` and
      call its `serving_default` signature with (frames, phrase).
    - If it's a regular .h5/.keras file, we load it with `tf.keras.models.load_model`.

    The callable expects a NumPy array of shape:
      - (batch_size, T, D) for sequence input (e.g., 128 frames x 164 features).
    """
    from pathlib import Path as _Path
    import tensorflow as tf

    path = _Path(model_path)

    # Case 1: SavedModel directory (your `results/model_full_tf`)
    if path.is_dir():
        print(f"[INFO] Detected SavedModel directory at {path}. Using tf.saved_model.load().")
        loaded = tf.saved_model.load(str(path))

        # Inspect available signatures
        sigs = dict(loaded.signatures)
        print(f"[INFO] Available signatures: {list(sigs.keys())}")

        if "serving_default" in sigs:
            fn = sigs["serving_default"]
            print("[INFO] Using 'serving_default' signature for inference.")
        elif "serve" in sigs:
            fn = sigs["serve"]
            print("[INFO] Using 'serve' signature for inference.")
        else:
            # Fallback: pick the first available signature
            key0 = next(iter(sigs.keys()))
            fn = sigs[key0]
            print(f"[WARN] 'serving_default' not found. Using '{key0}' signature instead.")

        # From the original notebook: MAX_PHRASE_LENGTH = 31 + 1
        MAX_PHRASE_LENGTH = 32

        def infer_fn(batch_np: np.ndarray):
            # batch_np: shape (batch_size, T, D)
            frames_batch = tf.convert_to_tensor(batch_np, dtype=tf.float32)
            batch_size = tf.shape(frames_batch)[0]

            # Dummy phrase input – we don't care about the contents, just shape
            phrase_dummy = tf.zeros((batch_size, MAX_PHRASE_LENGTH), dtype=tf.int32)

            # Call the SavedModel concrete function via named arguments
            _ = fn(frames=frames_batch, phrase=phrase_dummy)
            return None

        return infer_fn

    # Case 2: Regular Keras model file (.h5 / .keras)
    else:
        print(f"[INFO] Detected Keras model file at {path}. Using keras.models.load_model().")
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
        model.trainable = False

        def infer_fn(batch_np: np.ndarray):
            frames_batch = tf.convert_to_tensor(batch_np, dtype=tf.float32)
            _ = model(frames_batch, training=False)
            return None

        return infer_fn


# ---------------------------
# Benchmark core
# ---------------------------

def set_device_before_tf_import(device: str):
    """
    MUST be called before importing tensorflow.
    For CPU runs, disable CUDA so TF cannot see GPUs.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")
        print("[INFO] Forcing TensorFlow to use CPU only (CUDA_VISIBLE_DEVICES=-1)")
    else:
        print("[INFO] Allowing TensorFlow to use GPU (if available)")


def benchmark_inference(
    device: str,
    model_path: str,
    X: np.ndarray,
    batch_sizes: List[int],
    num_iters: int,
    num_warmup: int,
) -> List[Dict[str, Any]]:
    """
    Run inference on the given device for multiple batch sizes.
    Supports X with shape:
      - (N, D)
      - (N, T, D)
    """

    import tensorflow as tf  # noqa: E402

    print("[INFO] TensorFlow version:", tf.__version__)
    print("[INFO] Available physical devices:", tf.config.list_physical_devices())

    print(f"[INFO] Loading model from {model_path}")
    infer_fn = load_aslfr_inference_callable(model_path)

    n_samples = X.shape[0]
    X_ndim = X.ndim
    print(f"[INFO] Input tensor ndim = {X_ndim}, shape = {X.shape}")

    results: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed=42)

    for batch_size in batch_sizes:
        if batch_size > n_samples:
            print(
                f"[WARN] Batch size {batch_size} > number of samples {n_samples}. "
                f"Skipping this batch size."
            )
            continue

        print(f"\n[INFO] ==== Benchmarking batch_size = {batch_size} on {device.upper()} ====")

        indices = rng.choice(
            n_samples,
            size=batch_size * (num_warmup + num_iters),
            replace=True,
        )

        if X_ndim == 2:
            X_batched = X[indices].reshape(num_warmup + num_iters, batch_size, -1)
        elif X_ndim == 3:
            T = X.shape[1]
            D = X.shape[2]
            X_batched = X[indices].reshape(num_warmup + num_iters, batch_size, T, D)
        else:
            raise ValueError(f"Unsupported input ndim={X_ndim}; expected 2 or 3.")

        print(f"[INFO] Warmup: {num_warmup} batches (not timed)")
        for i in range(num_warmup):
            infer_fn(X_batched[i])

        cpu_metrics_before = get_cpu_metrics()
        if device == "gpu":
            gpu_metrics_before = get_gpu_metrics()
        else:
            gpu_metrics_before = {
                "gpu_util_percent": float("nan"),
                "gpu_mem_used_gb": float("nan"),
            }

        print(f"[INFO] Timed runs: {num_iters} batches")
        start = time.perf_counter()
        for i in range(num_warmup, num_warmup + num_iters):
            infer_fn(X_batched[i])
        end = time.perf_counter()

        total_time = end - start
        total_batches = num_iters
        total_samples = num_iters * batch_size
        avg_batch_latency_ms = (total_time / total_batches) * 1000.0
        throughput_sps = total_samples / total_time

        cpu_metrics_after = get_cpu_metrics()
        if device == "gpu":
            gpu_metrics_after = get_gpu_metrics()
        else:
            gpu_metrics_after = {
                "gpu_util_percent": float("nan"),
                "gpu_mem_used_gb": float("nan"),
            }

        cpu_util_avg = (
            cpu_metrics_before["cpu_util_percent"] + cpu_metrics_after["cpu_util_percent"]
        ) / 2.0
        cpu_mem_avg = (
            cpu_metrics_before["cpu_mem_used_gb"] + cpu_metrics_after["cpu_mem_used_gb"]
        ) / 2.0

        gpu_util_avg = (
            gpu_metrics_before["gpu_util_percent"] + gpu_metrics_after["gpu_util_percent"]
        ) / 2.0
        gpu_mem_avg = (
            gpu_metrics_before["gpu_mem_used_gb"] + gpu_metrics_after["gpu_mem_used_gb"]
        ) / 2.0

        print(
            f"[RESULT] batch_size={batch_size}, "
            f"total_time={total_time:.4f} s, "
            f"throughput={throughput_sps:.2f} samples/s, "
            f"avg_batch_latency={avg_batch_latency_ms:.3f} ms"
        )
        print(
            f"[RESULT] CPU util ~{cpu_util_avg:.1f}%, "
            f"CPU mem ~{cpu_mem_avg:.2f} GB"
        )
        if device == "gpu":
            print(
                f"[RESULT] GPU util ~{gpu_util_avg:.1f}%, "
                f"GPU mem ~{gpu_mem_avg:.2f} GB"
            )

        results.append(
            {
                "device": device,
                "batch_size": batch_size,
                "num_iters": num_iters,
                "num_warmup": num_warmup,
                "n_samples_available": n_samples,
                "total_samples_processed": total_samples,
                "total_time_sec": total_time,
                "avg_batch_latency_ms": avg_batch_latency_ms,
                "throughput_samples_per_sec": throughput_sps,
                "cpu_util_percent": cpu_util_avg,
                "cpu_mem_used_gb": cpu_mem_avg,
                "gpu_util_percent": gpu_util_avg,
                "gpu_mem_used_gb": gpu_mem_avg,
            }
        )

    return results


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"

    model_path = resolve_existing_path(args.model_path, results_dir, script_dir)
    inference_args_path = resolve_existing_path(args.inference_args, results_dir, script_dir)
    data_csv_path = resolve_existing_path(args.data_csv, results_dir, script_dir)

    output_csv_path = Path(args.output_csv).expanduser()
    if not output_csv_path.is_absolute():
        output_csv_path = (Path.cwd() / output_csv_path).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    set_device_before_tf_import(args.device)

    X, inf_args = load_features(data_csv_path, inference_args_path)

    results = benchmark_inference(
        device=args.device,
        model_path=str(model_path),
        X=X,
        batch_sizes=args.batch_sizes,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
    )

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv_path, index=False)
        print(f"\n[INFO] Saved results to {output_csv_path}")
        print(df_results)
    else:
        print("[WARN] No results (maybe all batch sizes were larger than dataset).")


if __name__ == "__main__":
    main()
