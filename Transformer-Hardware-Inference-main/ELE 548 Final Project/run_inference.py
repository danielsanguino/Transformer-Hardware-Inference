# run_inference.py
"""
Unified script for ELE 548 hardware benchmarking:
- Loads ASLFR Transformer model
- Runs inference (greedy decoding)
- Benchmarks CPU vs GPU
- Tracks:
    - Latency
    - Throughput
    - GPU utilization (%)
    - GPU VRAM (MB)
    - CPU utilization (%)
    - System RAM usage (MB)
    - TensorFlow GPU kernel time (TF profiler)
"""

import time
import subprocess
import numpy as np
import psutil
import tensorflow as tf

from aslfr_model.preprocessing import preprocess_frames_numpy, N_TARGET_FRAMES
from aslfr_model.model import get_model


# ===================================================================
# CONFIG
# ===================================================================

N_UNIQUE_CHARACTERS = 60
MAX_PHRASE_LENGTH   = 32
FEATURE_COLS        = 166

PAD_TOKEN = 0
EOS_TOKEN = 1

WEIGHTS_PATH = "model.h5"


# ===================================================================
# Load model
# ===================================================================

def load_aslfr_model():
    print("[INFO] Building model...")
    model = get_model(
        n_unique_characters=N_UNIQUE_CHARACTERS,
        max_phrase_length=MAX_PHRASE_LENGTH,
        n_cols=FEATURE_COLS,
    )
    try:
        model.load_weights(WEIGHTS_PATH)
        print(f"[INFO] Loaded weights from {WEIGHTS_PATH}")
    except Exception:
        print("[WARNING] No weights loaded (using random init).")

    return model


# ===================================================================
# Autoregressive inference
# ===================================================================

def predict_phrase(model, raw_frames):
    frames = preprocess_frames_numpy(raw_frames, n_target_frames=N_TARGET_FRAMES)
    frames = np.expand_dims(frames, axis=0)

    phrase = np.full((1, MAX_PHRASE_LENGTH), PAD_TOKEN, dtype=np.int32)

    for pos in range(MAX_PHRASE_LENGTH):
        logits = model([frames, phrase], training=False)
        step_logits = logits[:, pos, :]
        next_id = int(tf.argmax(step_logits, axis=-1)[0])
        phrase[0, pos] = next_id
        if next_id == EOS_TOKEN:
            break

    return phrase[0]


# ===================================================================
# Hardware Metrics Helpers
# ===================================================================

def get_gpu_stats():
    """
    Query GPU utilization + VRAM using nvidia-smi.
    Returns (utilization %, vram_MB)
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ]
        output = subprocess.check_output(cmd).decode().strip()
        util, mem = output.split(",")
        return float(util), float(mem)
    except:
        return None, None


def get_cpu_stats():
    """
    Returns CPU usage (%) and RAM (MB).
    """
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().used / (1024 * 1024)
    return cpu, mem


# ===================================================================
# Benchmark (with full hardware metrics)
# ===================================================================

def benchmark(model, dataset, device, warmup=3, repeat=20):
    print(f"\n[INFO] Benchmarking on {device} ...")

    gpu_utils = []
    gpu_mems  = []
    cpu_utils = []
    ram_usages = []

    with tf.device(device):

        # -------------------- Warmup --------------------
        for _ in range(warmup):
            _ = predict_phrase(model, dataset[0])

        # -------------------- Timed Runs --------------------
        start = time.perf_counter()

        for i in range(repeat):
            # Sample
            sample = dataset[i % len(dataset)]

            # System utilization samples BEFORE inference
            cpu, ram = get_cpu_stats()
            gpu_u, gpu_m = get_gpu_stats()

            if cpu: cpu_utils.append(cpu)
            if ram: ram_usages.append(ram)
            if gpu_u is not None: gpu_utils.append(gpu_u)
            if gpu_m is not None: gpu_mems.append(gpu_m)

            # Inference
            _ = predict_phrase(model, sample)

        end = time.perf_counter()

    # -------------------- Metrics --------------------
    total = end - start
    avg_latency = total / repeat
    throughput = repeat / total

    print(f"[RESULT] {device}:")
    print(f"   Avg latency   = {avg_latency*1000:.2f} ms")
    print(f"   Throughput    = {throughput:.2f} samples/sec")

    if gpu_utils:
        print(f"   GPU util avg  = {np.mean(gpu_utils):.2f}%")
        print(f"   GPU VRAM avg  = {np.mean(gpu_mems):.2f} MB")

    print(f"   CPU util avg  = {np.mean(cpu_utils):.2f}%")
    print(f"   RAM usage avg = {np.mean(ram_usages):.2f} MB")

    return {
        "latency": avg_latency,
        "throughput": throughput,
        "gpu_util": np.mean(gpu_utils) if gpu_utils else None,
        "gpu_mem": np.mean(gpu_mems) if gpu_mems else None,
        "cpu_util": np.mean(cpu_utils),
        "ram_usage": np.mean(ram_usages),
    }


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    print("[INFO] TF devices:", tf.config.list_physical_devices())

    # Load model
    model = load_aslfr_model()

    # -------------------- Fake Dataset --------------------
    dataset = []
    for _ in range(10):
        T = np.random.randint(150, 250)
        dataset.append(np.random.randn(T, FEATURE_COLS).astype("float32"))

    # Test 1 sample
    print("\n[INFO] Testing single inference...")
    print("Output:", predict_phrase(model, dataset[0]))

    # -------------------- Benchmark GPU --------------------
    if tf.config.list_physical_devices("GPU"):
        gpu_results = benchmark(model, dataset, device="/GPU:0")
    else:
        print("[WARNING] No GPU detected.")
        gpu_results = None

    # -------------------- Benchmark CPU --------------------
    cpu_results = benchmark(model, dataset, device="/CPU:0")

    print("\n[INFO] Benchmark complete.")
