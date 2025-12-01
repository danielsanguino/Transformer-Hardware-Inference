# preprocessing.py
"""
Preprocessing utilities for ASL fingerspelling transformer model.

This module provides:
- Constants that describe the input frame shape
- A simple TensorFlow layer that resamples variable-length sequences
  of frame features to a fixed length N_TARGET_FRAMES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Global shape / data constants
# ---------------------------------------------------------------------------

# Number of frames we resample each recording to (matches Kaggle notebook)
N_TARGET_FRAMES: int = 128

# Default number of feature columns per frame.
# In the original notebook this is derived from the selected landmark columns
# (COLUMNS0). Here we keep it configurable and detect it from data at runtime
# when possible.
N_COLS_DEFAULT: int = 166  # <- you can override this in your code if needed


@dataclass
class PreprocessConfig:
    """
    Configuration for the PreprocessLayer.

    Attributes
    ----------
    n_target_frames : int
        Number of frames to resample each sequence to.
    n_cols : Optional[int]
        Feature dimension. If None, will be inferred from inputs.
    """
    n_target_frames: int = N_TARGET_FRAMES
    n_cols: Optional[int] = None


class PreprocessLayer(tf.keras.layers.Layer):
    """
    TensorFlow layer that converts a variable-length sequence of frame features
    into a fixed [N_TARGET_FRAMES, N_COLS] tensor.

    This is a *simplified* version of the preprocessing in the Kaggle notebook:
    - It assumes you already selected the landmark columns you want.
    - It only does temporal resampling + padding, not hand-flipping tricks.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None, name: str = "preprocess", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config or PreprocessConfig()
        self.n_target_frames = self.config.n_target_frames
        self._n_cols = self.config.n_cols  # can be None until first call

    def build(self, input_shape):
        # If n_cols not provided, infer from input_shape on first build
        if self._n_cols is None:
            # input_shape: (batch, time, features) OR (time, features)
            if len(input_shape) == 2:
                self._n_cols = int(input_shape[-1])
            elif len(input_shape) == 3:
                self._n_cols = int(input_shape[-1])
            else:
                raise ValueError(f"Unexpected input shape for PreprocessLayer: {input_shape}")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : tf.Tensor
            Either [time, features] or [batch, time, features].

        Returns
        -------
        tf.Tensor
            [time, features] if input was [time, features],
            [batch, time, features] if input was [batch, time, features],
            with time == n_target_frames.
        """
        x = tf.convert_to_tensor(inputs)
        orig_rank = tf.rank(x)

        # Ensure shape is [batch, time, features]
        if orig_rank == 2:
            x = tf.expand_dims(x, axis=0)  # [1, T, F]
        elif orig_rank != 3:
            raise ValueError(f"PreprocessLayer expects rank-2 or rank-3, got rank {orig_rank}")

        batch_size = tf.shape(x)[0]
        n_frames = tf.shape(x)[1]
        n_cols = tf.shape(x)[2]

        # If sequence is shorter than target length, pad with zeros on the time axis.
        # If longer, we still pass through tf.image.resize which will downsample.
        x = tf.reshape(x, [batch_size, n_frames, n_cols, 1])  # fake spatial dim for tf.image.resize

        # Resize along the time dimension to n_target_frames
        x = tf.image.resize(
            images=x,
            size=(self.n_target_frames, 1),
            method=tf.image.ResizeMethod.BILINEAR,
        )  # [batch, n_target_frames, 1, features?] – but we used features as channels

        # Reshape back to [batch, time, features]
        x = tf.reshape(x, [batch_size, self.n_target_frames, n_cols])

        # Restore original rank (squeeze batch if necessary)
        if orig_rank == 2:
            x = tf.squeeze(x, axis=0)  # [time, features]

        return x


# ---------------------------------------------------------------------------
# Convenience function for NumPy → Tensor preprocessing
# ---------------------------------------------------------------------------

def preprocess_frames_numpy(
    frames: np.ndarray,
    n_target_frames: int = N_TARGET_FRAMES,
) -> np.ndarray:
    """
    Convenience function to preprocess a single sample in NumPy.

    Parameters
    ----------
    frames : np.ndarray
        Shape [time, features]. Can be variable-length in time.
    n_target_frames : int
        Target number of frames.

    Returns
    -------
    np.ndarray
        Resampled frames with shape [n_target_frames, features].
    """
    if frames.ndim != 2:
        raise ValueError(f"Expected frames shape [time, features], got {frames.shape}")

    layer = PreprocessLayer(PreprocessConfig(n_target_frames=n_target_frames, n_cols=frames.shape[-1]))
    # Build the layer by calling it once
    out = layer(tf.convert_to_tensor(frames, dtype=tf.float32))
    return out.numpy()


# Expose a default N_COLS symbol for imports from model.py
N_COLS: int = N_COLS_DEFAULT
