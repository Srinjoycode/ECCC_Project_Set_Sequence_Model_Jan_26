# ==========================================
# MEMORY MANAGEMENT UTILITIES
# ==========================================
import tensorflow as tf
import gc
import os
import numpy as np

DEFAULT_DTYPE = np.float32
def configure_gpu_memory():
    """
    # Configure TensorFlow GPU memory growth for better memory efficiency.
    Prevents TensorFlow from allocating all GPU memory at once.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU memory growth configuration error: {e}")
    else:
        print("ℹ️ No GPUs detected, running on CPU")


def clear_memory(verbose=False):
    """
    Aggressively clear memory including TensorFlow session and Python garbage collection.
    Call this between major processing steps.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()

    # Force Python garbage collection (multiple passes for cyclic references)
    gc.collect()
    gc.collect()
    gc.collect()

    if verbose:
        print("🧹 Memory cleared (TensorFlow session + garbage collection)")


def get_memory_usage_mb():
    """Get current memory usage in MB (requires psutil, returns -1 if not available)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1


def print_memory_status(label=""):
    """Print current memory usage for debugging."""
    mem_mb = get_memory_usage_mb()
    if mem_mb > 0:
        print(f"💾 Memory usage {label}: {mem_mb:.1f} MB")


def convert_to_float32(*arrays):
    """Convert arrays to float32 to save memory."""
    return tuple(arr.astype(DEFAULT_DTYPE) if arr is not None else None for arr in arrays)
