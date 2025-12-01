# model.py
"""
ASL fingerspelling Transformer model (cleaned up from Kaggle notebook).

This defines:
- A simple frame embedding layer with positional encodings
- Transformer Encoder and Decoder blocks
- A get_model(...) function that returns a compiled tf.keras.Model
  suitable for training or inference.

The focus is *clean structure* and *easy CPU vs GPU benchmarking*.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf

from .preprocessing import N_TARGET_FRAMES, N_COLS


# ---------------------------------------------------------------------------
# Hyperparameters (mirroring the Kaggle notebook)
# ---------------------------------------------------------------------------

LAYER_NORM_EPS: float = 1e-6

UNITS_ENCODER: int = 384
UNITS_DECODER: int = 256

NUM_BLOCKS_ENCODER: int = 4
NUM_BLOCKS_DECODER: int = 2
NUM_HEADS: int = 4
MLP_RATIO: int = 2

# Dropout (can be tuned / disabled)
EMBEDDING_DROPOUT: float = 0.0
MLP_DROPOUT_RATIO: float = 0.30
MHA_DROPOUT_RATIO: float = 0.20
CLASSIFIER_DROPOUT_RATIO: float = 0.10

# Initializers and activations
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform()
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform()
INIT_ZEROS = tf.keras.initializers.Zeros()
GELU = tf.keras.activations.gelu


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Standard multi-head self/cross attention.

    This is slightly cleaner than the fused version in the original notebook,
    but mathematically equivalent in spirit.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.scale = tf.math.rsqrt(tf.cast(self.depth, tf.float32))

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=INIT_HE_UNIFORM)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x: [batch, time, d_model] -> [batch, num_heads, time, depth]
        batch_size = tf.shape(x)[0]
        time = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, time, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        # x: [batch, num_heads, time, depth] -> [batch, time, d_model]
        batch_size = tf.shape(x)[0]
        time = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, [batch_size, time, self.d_model])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # Linear projections
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        # Split into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True) * self.scale  # [B, H, T_q, T_k]

        if mask is not None:
            # mask expected shape [B, 1, T_q, T_k] or broadcastable
            scores += (mask * -1e9)

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)

        context = tf.matmul(weights, v)  # [B, H, T_q, depth]
        context = self._combine_heads(context)  # [B, T_q, d_model]
        out = self.wo(context)
        out = self.dropout(out, training=training)
        return out


class FrameEmbedding(tf.keras.layers.Layer):
    """
    Simple frame embedding:
    - Normalize non-zero values
    - Linear projection to UNITS_ENCODER
    - Add learnable positional embedding
    """

    def __init__(
        self,
        units: int = UNITS_ENCODER,
        n_target_frames: int = N_TARGET_FRAMES,
        dropout: float = EMBEDDING_DROPOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.n_target_frames = n_target_frames
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def build(self, input_shape):
        # Positional embeddings for each frame index
        self.positional_embedding = self.add_weight(
            name="embedding_positional_encoder",
            shape=[self.n_target_frames, self.units],
            initializer=INIT_ZEROS,
            trainable=True,
        )
        # Simple dense embedding for the full frame feature vector
        self.dense = tf.keras.layers.Dense(
            self.units,
            activation=GELU,
            use_bias=False,
            kernel_initializer=INIT_GLOROT_UNIFORM,
            name="frame_dense",
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        x: [batch, time, features]
        """
        # Simple heuristic: treat all-zero frames as padding
        # (we don't mask here; masking is handled in encoder attention)
        embedded = self.dense(x)  # [B, T, units]

        # Add positional encodings
        pos = tf.expand_dims(self.positional_embedding, axis=0)  # [1, T, units]
        x = embedded + pos

        x = self.dropout(x, training=training)
        return x


class Encoder(tf.keras.layers.Layer):
    """
    Transformer encoder: stacks NUM_BLOCKS_ENCODER residual attention + MLP blocks.
    """

    def __init__(self, num_blocks: int = NUM_BLOCKS_ENCODER, **kwargs):
        super().__init__(name="encoder", **kwargs)
        self.num_blocks = num_blocks
        self.supports_masking = True

        self.layers_norm1 = []
        self.attention_layers = []
        self.layers_norm2 = []
        self.mlp_layers = []

        for i in range(num_blocks):
            self.layers_norm1.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS, name=f"enc_ln1_{i}")
            )
            self.attention_layers.append(
                MultiHeadAttention(UNITS_ENCODER, NUM_HEADS, dropout=MHA_DROPOUT_RATIO, name=f"enc_mha_{i}")
            )
            self.layers_norm2.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS, name=f"enc_ln2_{i}")
            )
            self.mlp_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            UNITS_ENCODER * MLP_RATIO,
                            activation=GELU,
                            use_bias=False,
                            kernel_initializer=INIT_GLOROT_UNIFORM,
                        ),
                        tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
                        tf.keras.layers.Dense(
                            UNITS_ENCODER,
                            use_bias=False,
                            kernel_initializer=INIT_HE_UNIFORM,
                        ),
                    ],
                    name=f"enc_mlp_{i}",
                )
            )

    def _build_padding_mask(self, frames_inp: tf.Tensor) -> tf.Tensor:
        """
        Build a [batch, 1, 1, time] mask where 1 indicates padded (all-zero) frames.
        """
        # frames_inp: [B, T, F]
        mask = tf.math.equal(tf.reduce_sum(tf.abs(frames_inp), axis=-1), 0.0)  # [B, T]
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=1)  # [B, 1, T]
        mask = tf.expand_dims(mask, axis=1)  # [B, 1, 1, T]
        return mask

    def call(self, x: tf.Tensor, frames_inp: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        x: [batch, time, UNITS_ENCODER]
        frames_inp: [batch, time, features] (for building padding mask)
        """
        attn_mask = self._build_padding_mask(frames_inp)

        for ln1, mha, ln2, mlp in zip(
            self.layers_norm1, self.attention_layers, self.layers_norm2, self.mlp_layers
        ):
            # Self-attention with padding mask
            y = ln1(x)
            y = mha(y, y, y, mask=attn_mask, training=training)
            x = x + y

            # MLP
            y = ln2(x)
            y = mlp(y, training=training)
            x = x + y

        return x


class Decoder(tf.keras.layers.Layer):
    """
    Transformer decoder that attends to encoder outputs given an input phrase.

    - Input: phrase token IDs [batch, L]
    - Context: encoder outputs [batch, T_enc, UNITS_ENCODER]
    - Output: [batch, L, UNITS_DECODER]
    """

    def __init__(self, num_blocks: int = NUM_BLOCKS_DECODER, **kwargs):
        super().__init__(name="decoder", **kwargs)
        self.num_blocks = num_blocks
        self.supports_masking = True

        # Token embedding
        self.char_emb = tf.keras.layers.Embedding(
            input_dim=1,  # placeholder; real vocab size is set in build()
            output_dim=UNITS_DECODER,
            embeddings_initializer=INIT_GLOROT_UNIFORM,
            name="char_embedding",
        )

        self.self_attn_layers = []
        self.cross_attn_layers = []
        self.ln_self_attn = []
        self.ln_cross_attn = []
        self.mlp_layers = []

        for i in range(num_blocks):
            self.self_attn_layers.append(
                MultiHeadAttention(UNITS_DECODER, NUM_HEADS, dropout=MHA_DROPOUT_RATIO, name=f"dec_self_mha_{i}")
            )
            self.cross_attn_layers.append(
                MultiHeadAttention(UNITS_DECODER, NUM_HEADS, dropout=MHA_DROPOUT_RATIO, name=f"dec_cross_mha_{i}")
            )
            self.ln_self_attn.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS, name=f"dec_ln_self_{i}")
            )
            self.ln_cross_attn.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS, name=f"dec_ln_cross_{i}")
            )
            self.mlp_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            UNITS_DECODER * MLP_RATIO,
                            activation=GELU,
                            use_bias=False,
                            kernel_initializer=INIT_GLOROT_UNIFORM,
                        ),
                        tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
                        tf.keras.layers.Dense(
                            UNITS_DECODER,
                            use_bias=False,
                            kernel_initializer=INIT_HE_UNIFORM,
                        ),
                    ],
                    name=f"dec_mlp_{i}",
                )
            )

    def build(self, input_shape):
        # Set real vocab size from the phrase input shape if available.
        # input_shape[0] -> phrase, e.g. (batch, L)
        super().build(input_shape)

    @staticmethod
    def _causal_mask(seq_len: tf.Tensor) -> tf.Tensor:
        """
        Create a [1, 1, L, L] mask for causal self-attention.
        """
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        mask = tf.cast(j > i, tf.float32)  # 1 where future positions should be masked
        mask = tf.reshape(mask, [1, 1, seq_len, seq_len])
        return mask

    def call(
        self,
        encoder_out: tf.Tensor,
        phrase_inp: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        encoder_out: [batch, T_enc, UNITS_ENCODER]
        phrase_inp: [batch, L] integer token ids
        """
        batch_size = tf.shape(phrase_inp)[0]
        seq_len = tf.shape(phrase_inp)[1]

        # Token embedding + positional encoding
        x = self.char_emb(phrase_inp)  # [B, L, UNITS_DECODER]

        pos_emb = self.add_weight(
            "positional_embedding_decoder",
            shape=[seq_len, UNITS_DECODER],
            initializer=INIT_ZEROS,
            trainable=True,
        )
        x = x + tf.expand_dims(pos_emb, axis=0)  # [B, L, D]

        # Masks
        self_mask = self._causal_mask(seq_len)  # [1, 1, L, L]
        # No padding mask for encoder_out here for simplicity; you can add it.

        for ln_self, ln_cross, self_mha, cross_mha, mlp in zip(
            self.ln_self_attn,
            self.ln_cross_attn,
            self.self_attn_layers,
            self.cross_attn_layers,
            self.mlp_layers,
        ):
            # Masked self-attention
            y = ln_self(x)
            y = self_mha(y, y, y, mask=self_mask, training=training)
            x = x + y

            # Encoder-decoder cross-attention
            y = ln_cross(x)
            # Key/value from encoder, query from decoder
            y = cross_mha(y, encoder_out, encoder_out, training=training)
            x = x + y

            # MLP
            y = mlp(x, training=training)
            x = x + y

        return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(
    n_unique_characters: int,
    max_phrase_length: int,
    n_target_frames: int = N_TARGET_FRAMES,
    n_cols: int = N_COLS,
    compile_model: bool = True,
) -> tf.keras.Model:
    """
    Build the ASL Transformer model.

    Parameters
    ----------
    n_unique_characters : int
        Size of the output vocabulary (including PAD/SOS/EOS).
    max_phrase_length : int
        Maximum output sequence length (including EOS).
    n_target_frames : int
        Number of frames per sequence (after preprocessing).
    n_cols : int
        Feature dimension of each frame.
    compile_model : bool
        If True, compile the model with a default optimizer/loss/metrics.

    Returns
    -------
    tf.keras.Model
        Keras model with inputs [frames, phrase] and output logits
        of shape [batch, max_phrase_length, n_unique_characters].
    """
    # Inputs
    frames_inp = tf.keras.layers.Input(
        shape=(n_target_frames, n_cols),
        dtype=tf.float32,
        name="frames",
    )
    phrase_inp = tf.keras.layers.Input(
        shape=(max_phrase_length,),
        dtype=tf.int32,
        name="phrase",
    )

    # Masking (treat all-zero frames as padding)
    x = tf.keras.layers.Masking(mask_value=0.0, name="frame_mask")(frames_inp)

    # Embedding + Encoder
    x = FrameEmbedding(units=UNITS_ENCODER, n_target_frames=n_target_frames, name="embedding")(x)
    enc_out = Encoder(num_blocks=NUM_BLOCKS_ENCODER, name="encoder")(x, frames_inp)

    # Project encoder output to decoder dimension if needed
    if UNITS_ENCODER != UNITS_DECODER:
        enc_out = tf.keras.layers.Dense(
            UNITS_DECODER,
            use_bias=False,
            kernel_initializer=INIT_GLOROT_UNIFORM,
            name="enc_to_dec_proj",
        )(enc_out)

    # Decoder (returns [batch, L, UNITS_DECODER])
    dec_out = Decoder(num_blocks=NUM_BLOCKS_DECODER, name="decoder")(enc_out, phrase_inp)

    # Final classifier over characters
    logits = tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(CLASSIFIER_DROPOUT_RATIO),
            tf.keras.layers.Dense(
                n_unique_characters,
                activation=None,
                use_bias=False,
                kernel_initializer=INIT_GLOROT_UNIFORM,
            ),
        ],
        name="classifier",
    )(dec_out)

    model = tf.keras.Model(
        inputs=[frames_inp, phrase_inp],
        outputs=logits,
        name="aslfr_transformer",
    )

    if compile_model:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy"),
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
