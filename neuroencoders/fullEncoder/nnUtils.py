# Load libs
import contextlib
import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import keras
import keras.utils as keras_utils
import numpy as np
import psutil
import tensorflow as tf
from denseweight import DenseWeight
from keras import ops as kops
from scipy.ndimage import gaussian_filter

from neuroencoders.utils.global_classes import Params, SpatialConstraintsMixin


def get_device_context(device):
    """
    Return a context manager that controls TensorFlow device or strategy scope.

    Parameters
    ----------
    device : None, str, or tf.distribute.Strategy
        - ``None``:
            No explicit device placement is requested. A no-op context manager
            (``contextlib.nullcontext``) is returned, so the caller's existing
            device/strategy configuration is left unchanged.
        - ``tf.distribute.Strategy``:
            A distribution strategy instance. The strategy's scope
            (``strategy.scope()``) is returned so that any model or variable
            creation inside the ``with`` block is placed under the strategy.
        - ``str``:
            A TensorFlow device specification string, e.g. ``"/CPU:0"``,
            ``"/GPU:0"``, or ``"/device:GPU:0"``. If no distribution strategy
            is active, this is passed to :func:`tf.device`. If a distribution
            strategy is already active, a no-op context is returned instead so
            that the strategy can manage placement automatically.

    Returns
    -------
    context manager
        A context manager suitable for use in a ``with`` statement:

        - ``contextlib.nullcontext()`` when:
            * ``device is None``,
            * a distribution strategy is active and ``device`` is a ``str``,
            * or ``device`` is an invalid device specification (caught
              ``ValueError`` or ``TypeError`` from :func:`tf.device`).
        - ``strategy.scope()`` when ``device`` is an instance of
          :class:`tf.distribute.Strategy`.
        - ``tf.device(device)`` when ``device`` is a valid device string and
          no distribution strategy is active.

    Notes
    -----
    This helper centralizes the logic for safely handling device strings and
    distribution strategies. It intentionally falls back to a no-op context
    manager for unsupported or invalid configurations instead of raising,
    making it easier to write higher-level utilities that accept an optional
    ``device`` argument.

    When a distribution strategy is active and a device string is provided,
    the device string is ignored with a warning, allowing the strategy to
    manage device placement automatically.

    Examples
    --------
    Use default (no explicit device):

    >>> with get_device_context(None):
    ...     x = tf.constant(1.0)

    Use a specific device:

    >>> with get_device_context("/GPU:0"):
    ...     x = tf.constant(1.0)

    Use with a distribution strategy:

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> with get_device_context(strategy):
    ...     model = tf.keras.Sequential(...)

    Active strategy with a device string (string is ignored, strategy wins):

    >>> strategy = tf.distribute.MirroredStrategy()
    >>> with strategy.scope():
    ...     # Strategy is active here
    ...     with get_device_context("/GPU:0"):
    ...         # This inner scope is effectively a no-op; placement is still
    ...         # controlled by the active strategy.
    ...         x = tf.constant(1.0)

    Invalid device string (falls back to no-op context):

    >>> with get_device_context("INVALID_DEVICE"):
    ...     # Falls back to a no-op context; no exception is raised here.
    ...     x = tf.constant(1.0)
    """
    logger = logging.getLogger(__name__)

    if device is None:
        return contextlib.nullcontext()
    if isinstance(device, tf.distribute.Strategy):
        return device.scope()
    if tf.distribute.has_strategy() and isinstance(device, str):
        # If a distribution strategy is active, it's generally better to let it
        # handle device placement automatically.
        logger.warning(
            f"Distribution strategy is active, ignoring explicit device placement request: {device}"
        )
        return contextlib.nullcontext()
    try:
        return tf.device(device)
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Invalid device specification '{device}': {e}. Falling back to default device placement."
        )
        return contextlib.nullcontext()


@keras.saving.register_keras_serializable(package="neuroencoders")
class AddNullSpike(tf.keras.layers.Layer):
    """
    The "Null Spike" Trick:
    Add a row of zeros at index 0 for each example in the batch.
    When index_inputs[g] is 0 (padding), tf.gather will pick up these zeros.
    """

    def call(self, e):
        null_spike = kops.zeros_like(e[:, :1, :])  # shape (batch, 1, nFeatures)
        return kops.concatenate([null_spike, e], axis=1)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            return (input_shape[0], input_shape[1] + 1, input_shape[2])
        return (input_shape[0], None, input_shape[2])


@keras.saving.register_keras_serializable(package="neuroencoders")
class GatherSpikes(tf.keras.layers.Layer):
    """
    Gather spikes into the global sequence: (batch, seqLen, nFeatures)
    batch_dims=1 enables parallel gathering across the batch
    """

    def call(self, inputs):
        full_emb, indices = inputs
        return tf.gather(full_emb, indices, batch_dims=1)

    def compute_output_shape(self, input_shapes):
        # input_shapes = [(batch, time_in, feat), (batch, seqLen)]
        return (input_shapes[0][0], input_shapes[1][1], input_shapes[0][2])


@keras.saving.register_keras_serializable(package="neuroencoders")
class MaskingLayer(tf.keras.layers.Layer):
    """
    Apply mask to features, setting masked values to zero.
    """

    def call(self, inputs):
        mask, features = inputs
        mask_expanded = kops.expand_dims(mask, axis=-1)
        return kops.where(mask_expanded, features, kops.zeros_like(features))

    def compute_output_shape(self, input_shapes):
        # input_shapes = [(batch, seqLen), (batch, seqLen, feat)]
        return input_shapes[1]


########### CONVOLUTIONAL NETWORK CLASS #####################
@keras.saving.register_keras_serializable(package="neuroencoders")
class spikeNet(tf.keras.layers.Layer):
    """
    This class is a convolutional network that takes as input a spike sequence from nChannels and returns a feature vector of size nFeatures.

    args:
    -----
    nChannels: number of channels in the input
    device: the device on which the network is run
    nFeatures: the size of the output feature vector
    number: a number to identify the network

    Details of the default network:
    -----
    The network is composed of 3 convolutional layers each followed by a max pooling layer.
    The convolutional layers have 8, 16 and 32 filters of size 2x3. The max pooling layers have a pool size of 1x2.
    The convolutional layers are followed by 3 dense layers with a ReLU activation function. The dense layers have a size of nFeatures and the
        last dense layer has a size of nFeatures and is named "outputCNN{number}".

    One filter of size (2,3) would roughly mean that the first filters "see" 2 channels at a time and 3 bins of a 32 timesteps sampling,
        i.e. (3/20000) ~= 0.15 ms for a sampling rate of 20 000Hz. The whole 32 window corresponds to 1.6ms of data.
    """

    def __init__(
        self,
        nChannels=4,
        device: str = "/cpu:0",
        nFeatures=128,
        number="",
        reduce_dense=False,
        no_cnn=False,
        **kwargs,
    ):
        name = kwargs.pop("name", "spikeNet{}".format(number))
        self.reduce_dense = reduce_dense
        self.batch_normalization = kwargs.pop("batch_normalization", True)
        self.no_cnn = no_cnn
        super().__init__(name=name, **kwargs)
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        self.device = device
        self.number = number
        with get_device_context(self.device):
            self.convLayer1 = tf.keras.layers.Conv2D(8, [2, 3], padding="same")
            self.convLayer2 = tf.keras.layers.Conv2D(16, [2, 3], padding="same")
            self.convLayer3 = tf.keras.layers.Conv2D(32, [2, 3], padding="same")

            self.maxPoolLayer1 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="same"
            )
            self.maxPoolLayer2 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="same"
            )
            self.maxPoolLayer3 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="same"
            )
            if self.batch_normalization:
                self.bn1 = tf.keras.layers.BatchNormalization()
                self.bn2 = tf.keras.layers.BatchNormalization()
                self.bn3 = tf.keras.layers.BatchNormalization()

            self.dropoutLayer = tf.keras.layers.Dropout(0.2)
            self.denseLayer1 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer2 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer3 = tf.keras.layers.Dense(
                self.nFeatures, activation="relu", name="outputCNN{}".format(number)
            )

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "nChannels": self.nChannels,
                "device": self.device,
                "nFeatures": self.nFeatures,
                "number": self.number,
                "batch_normalization": self.batch_normalization,
            }
        )
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        nChannels = config.get("nChannels", 4)
        device = config.get("device", "/cpu:0")
        nFeatures = config.get("nFeatures", 128)
        number = config.get("number", "")
        batch_normalization = config.get("batch_normalization", True)
        reduce_dense = config.get("reduce_dense", False)
        no_cnn = config.get("no_cnn", False)
        return cls(
            nChannels=nChannels,
            device=device,
            nFeatures=nFeatures,
            number=number,
            batch_normalization=batch_normalization,
            reduce_dense=reduce_dense,
            no_cnn=no_cnn,
        )

    def __call__(self, input):
        return self.apply(input)

    def apply(self, input):
        with get_device_context(self.device):
            if self.no_cnn:
                # reshape input directly to dense layer
                # must be of shape (batch_size, -1)
                x = tf.keras.layers.Flatten()(input)
                x = self.denseLayer3(x)
                x = self.dropoutLayer(x)
                print("Skipping CNN layers")
                return x
            x = kops.expand_dims(input, axis=3)
            x = self.convLayer1(x)
            if self.batch_normalization:
                x = self.bn1(x)
            x = self.maxPoolLayer1(x)
            x = self.convLayer2(x)
            if self.batch_normalization:
                x = self.bn2(x)
            x = self.maxPoolLayer2(x)
            x = self.convLayer3(x)
            if self.batch_normalization:
                x = self.bn3(x)
            x = self.maxPoolLayer3(x)

            # or we could simply tf.keras.layers.Flatten() the output of the conv layers - leaves batch size unchanged
            x = tf.keras.layers.Flatten()(x)
            if not self.reduce_dense:
                x = self.denseLayer1(x)
                x = self.dropoutLayer(x)
                x = self.denseLayer2(x)
                x = self.denseLayer3(x)
            else:
                x = self.denseLayer3(x)
                x = self.dropoutLayer(x)

            return x

    @property
    def variables(self):
        vars_list = (
            self.convLayer1.variables
            + self.convLayer2.variables
            + self.convLayer3.variables
            + self.maxPoolLayer1.variables
            + self.maxPoolLayer2.variables
            + self.maxPoolLayer3.variables
            + self.denseLayer3.variables
        )
        if not self.reduce_dense:
            vars_list += self.denseLayer1.variables + self.denseLayer2.variables
        if self.batch_normalization:
            vars_list += self.bn1.variables + self.bn2.variables + self.bn3.variables
        if self.no_cnn:
            return self.denseLayer3.variables + self.dropoutLayer.variables
        return vars_list

    def layers(self):
        layers_list = (
            self.convLayer1,
            self.convLayer2,
            self.convLayer3,
            self.maxPoolLayer1,
            self.maxPoolLayer2,
            self.maxPoolLayer3,
            self.denseLayer3,
        )
        if not self.reduce_dense:
            layers_list += (self.denseLayer1, self.denseLayer2)
        if self.batch_normalization:
            layers_list += (self.bn1, self.bn2, self.bn3)
        if self.no_cnn:
            return (self.denseLayer3, self.dropoutLayer)
        return layers_list

    def build(self, input_shape):
        super().build(input_shape)
        # validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch, time, channels), got {input_shape}"
            )
        if input_shape[1] != self.nChannels:
            raise ValueError(
                f"Expected input shape with {self.nChannels} channels, got {input_shape[1]}"
            )


@keras.saving.register_keras_serializable(package="neuroencoders")
class SpikeNet1D(tf.keras.layers.Layer):
    """
    Refined Spike Encoder.
    Input shape: (Batch, Channels, Time) -> e.g., (128, 6, 32)
    This version transposes the input so Conv1D operates on the Time axis (32)
    while keeping the 6 channels separate.

    Output shape: (Batch, nFeatures) -> e.g., (128, 128)
    """

    def __init__(
        self,
        nChannels=4,
        device: str = "/cpu:0",
        nFeatures=128,
        number="",
        dropout_rate=0.2,
        batch_normalization=True,
        **kwargs,
    ):
        name = kwargs.pop("name", "spikeNet1D{}".format(number))
        # TODO: implement reduce_dense and no_cnn options if needed
        self.reduce_dense = kwargs.pop("reduce_dense", False)
        self.no_cnn = kwargs.pop("no_cnn", False)
        super().__init__(name=name, **kwargs)
        self.nChannels = nChannels
        self.nFeatures = nFeatures
        self.batch_normalization = batch_normalization
        self.device = device
        self.number = number

        with get_device_context(self.device):
            # Layer 1: Convolves over time bins
            self.conv1 = tf.keras.layers.Conv1D(16, 3, padding="same", activation=None)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.act1 = tf.keras.layers.Activation("relu")
            self.pool1 = tf.keras.layers.MaxPool1D(2, padding="same")  # Pool time only

            # Layer 2
            self.conv2 = tf.keras.layers.Conv1D(32, 3, padding="same", activation=None)
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.act2 = tf.keras.layers.Activation("relu")
            self.pool2 = tf.keras.layers.MaxPool1D(2, padding="same")

            # Layer 3
            self.nConvChannels = (
                64  # number of output channels from conv layers (before global pool)
            )
            self.conv3 = tf.keras.layers.Conv1D(
                self.nConvChannels, 3, padding="same", activation=None
            )
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.act3 = tf.keras.layers.Activation("relu")
            self.GlobalPool = tf.keras.layers.GlobalAveragePooling1D()

            self.temporal_extractor = tf.keras.Sequential(
                [
                    self.conv1,
                    self.bn1,
                    self.act1,
                    self.pool1,
                    self.conv2,
                    self.bn2,
                    self.act2,
                    self.pool2,
                    self.conv3,
                    self.bn3,
                    self.act3,
                    self.GlobalPool,  # result (Batch * nChannels, 64)
                ]
            )

            # Aggregation
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
            self.flatten = tf.keras.layers.Flatten()
            # Input to dense will be (6 channels * 64 features) = 384
            self.dense_fusion = tf.keras.layers.Dense(nFeatures * 2, activation="relu")
            self.dense_out = tf.keras.layers.Dense(
                nFeatures, activation=None, name=f"outputCNN{number}"
            )

    def apply(self, x, training=False):
        return self.call(x, training=training)

    def call(self, x, training=False):
        with get_device_context(self.device):
            # 1. Input is (Batch, Channels, Time) -> (128, 6, 32)
            shape = tf.shape(x)
            B, _C, T = shape[0], shape[1], shape[2]

            # Step 1: Reshape to process all channels through the SAME backbone
            # New shape: (Batch * 6, 32, 1)
            # T is hardcoded to be 32 for now (see tfrec creation via julia)
            x = tf.reshape(
                x, [B * self.nChannels, T, 1]
            )  # channels_last format for conv1d and global average pooling

            # Step 2: Extract temporal features (Shared Weights)
            # Output shape: (Batch * 6, self.nConvChannels = 64)
            x = self.temporal_extractor(x)

            # Step 3: Concatenate channels back together
            # New shape: (Batch, 6 * nConvChannels) -> (128, 384)
            x = tf.reshape(x, [B, self.nChannels * self.nConvChannels])

            # Step 4: Spatial Interaction (Learning relationships between fixed channels)
            x = self.dense_fusion(x)
            x = self.dropout(x, training=training)
            return self.dense_out(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nChannels": self.nChannels,
                "device": self.device,
                "nFeatures": self.nFeatures,
                "number": self.number,
                "batch_normalization": self.batch_normalization,
                "reduce_dense": self.reduce_dense,
                "no_cnn": self.no_cnn,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        nChannels = config.get("nChannels", 4)
        device = config.get("device", "/cpu:0")
        nFeatures = config.get("nFeatures", 128)
        number = config.get("number", "")
        batch_normalization = config.get("batch_normalization", True)
        reduce_dense = config.get("reduce_dense", False)
        no_cnn = config.get("no_cnn", False)
        return cls(
            nChannels=nChannels,
            device=device,
            nFeatures=nFeatures,
            number=number,
            batch_normalization=batch_normalization,
            reduce_dense=reduce_dense,
            no_cnn=no_cnn,
        )

    def build(self, input_shape):
        super().build(input_shape)
        # validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch, time, channels), got {input_shape}"
            )
        if input_shape[1] != self.nChannels:
            raise ValueError(
                f"Expected input shape with {self.nChannels} channels, got {input_shape[1]}"
            )

        # ---- Build sub-layers ----
        # 1. Temporal Extractor
        # Input to this is (Batch * Channels, Time, 1)
        time_steps = input_shape[2]
        self.temporal_extractor.build((None, time_steps, 1))

        # 2. Dense Fusion
        # The extractor outputs 64 features per channel (from the 3rd conv/global pool)
        # Input shape becomes (Batch, Channels * 64)
        fusion_input_dim = (
            self.nChannels * self.nConvChannels
        )  # 6 channels * 64 convChannels = 384
        self.dense_fusion.build((None, fusion_input_dim))

        # 3. Dense Out
        # Dense fusion outputs (Batch, nFeatures * 2)
        self.dense_out.build((None, self.nFeatures * 2))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nFeatures)


########### CONVOLUTIONAL NETWORK CLASS #####################


########### TRANSFORMER ENCODER CLASS #####################
@keras.saving.register_keras_serializable(package="neuroencoders")
class GroupAttentionFusion(tf.keras.layers.Layer):
    """
    Fuses features from multiple spike groups using Self-Attention.
    Instead of concatenating [G1, G2, ...], this layer allows groups to
    contextualize each other before flattening.
    """

    def __init__(self, n_groups, embed_dim, num_heads=4, device="/cpu:0", **kwargs):
        super().__init__(**kwargs)
        self.n_groups = n_groups
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device

        with get_device_context(self.device):
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )
            self.norm = tf.keras.layers.LayerNormalization()
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.flatten = tf.keras.layers.Flatten()
            self.supports_masking = True  # To indicate that this layer supports masking

    def call(self, inputs, mask=None):
        # inputs: List of tensors, each shape (Batch, Time, Features)
        with get_device_context(self.device):
            # 1. Stack groups: (Batch, Time, Groups, Features)
            x = tf.stack(inputs, axis=2)

            # 2. Add Group Embeddings
            # Broadcast embeddings across Batch and Time
            x = x + self.group_embeddings

            shape = tf.shape(x)
            B, T = shape[0], shape[1]

            # 3. Merge Batch and Time for Attention
            # Attention operates on the 'Groups' dimension (axis 2)
            # Shape becomes (Batch*Time, Groups, Features)
            x_reshaped = tf.reshape(x, (B * T, self.n_groups, self.embed_dim))

            # handle mask if provided
            mha_mask = None
            if mask is not None:
                # mask comes in as shape (Batch, max(nSpikes), n_groups)
                # we need to reshape it to match x_reshape ie (Batch*Time, n_groups)
                mask_reshaped = tf.reshape(mask, (B * T, self.n_groups))
                # now expands dims to (Batch*Time, 1, n_groups) for mha
                # The shape (B, 1, S) allows broadcasting the mask over the query dimension (dim 1).
                # Meaning: "For every querying group, here are the keys (target groups) you can attend to."
                mha_mask = tf.expand_dims(mask_reshaped, axis=1)

            # 4. Self-Attention over groups
            attn_out = self.mha(
                query=x_reshaped,
                value=x_reshaped,
                key=x_reshaped,
                attention_mask=mha_mask,
            )
            x_reshaped = self.norm(x_reshaped + self.dropout(attn_out))

            # 5. Restore temporal dimension & Flatten groups
            # Reshape to (Batch, Time, Groups * Features)
            # This prepares the tensor for the downstream LSTM/Transformer
            output = tf.reshape(x_reshaped, (B, T, self.n_groups * self.embed_dim))

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_groups": self.n_groups,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "device": self.device,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        n_groups = config.get("n_groups", 2)
        embed_dim = config.get("embed_dim", 64)
        num_heads = config.get("num_heads", 4)
        device = config.get("device", "/cpu:0")
        return cls(
            n_groups=n_groups, embed_dim=embed_dim, num_heads=num_heads, device=device
        )

    def build(self, input_shape):
        """
        Create learnable group embeddings.
        input_shape is a list of shapes [(B, T, F), (B, T, F), ...]
        """
        # Learnable positional embedding for each group ID
        # Shape: (1, 1, n_groups, embed_dim) for broadcasting over Batch and Time
        self.group_embeddings = self.add_weight(
            name="group_embeddings",
            shape=(1, 1, self.n_groups, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32",
        )
        # build sublayers
        # attention operates on (batch*time, n_groups, embed_dim)
        self.mha.build(
            query_shape=(None, self.n_groups, self.embed_dim),
            value_shape=(None, self.n_groups, self.embed_dim),
            key_shape=(None, self.n_groups, self.embed_dim),
        )
        self.norm.build(input_shape=(None, self.n_groups, self.embed_dim))

        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        # Reduce mask from (Batch, Time, nGroups) to (Batch, Time)
        # using any() because a time step is valid if any group has valid data
        return kops.any(mask, axis=-1)

    def compute_output_shape(self, input_shape):
        # Output shape: (Batch, Time, n_groups * embed_dim)
        batch_size, time_steps = input_shape[0][0], input_shape[0][1]
        return (batch_size, time_steps, self.n_groups * self.embed_dim)


@keras.saving.register_keras_serializable(package="neuroencoders")
class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
    """Global Average Pooling that respects masking"""

    def __init__(self, device="/cpu:0", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.supports_masking = True

    def call(self, inputs, mask=None):
        with get_device_context(self.device):
            if mask is not None:
                # Convert mask to float and add dimension for broadcasting
                # expand mask to match input dimensions [batch, seq_len, 1]
                mask_expanded = kops.expand_dims(mask, axis=-1)
                # Cast mask to input dtype to avoid multiplication errors with bool
                mask_expanded = kops.cast(
                    mask_expanded, self.compute_dtype or inputs.dtype
                )

                # Apply mask to inputs
                masked_inputs = inputs * mask_expanded

                # Calculate sum and count of non-masked elements
                sum_inputs = kops.sum(masked_inputs, axis=1)  # [batch, features]
                count_inputs = kops.sum(mask_expanded, axis=1)  # [batch, features]

                # Avoid division by zero
                count_inputs = kops.maximum(count_inputs, 1.0)

                # Calculate average
                return sum_inputs / count_inputs
            else:
                return kops.mean(inputs, axis=1)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"device": self.device})
        return base_config

    def compute_mask(self, inputs, mask=None):
        # No mask to pass on after pooling
        return None

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        device = config.get("device", "/cpu:0")
        return cls(device=device)


def create_attention_mask_from_padding_mask(padding_mask):
    """
    Convert padding mask to attention mask for transformer
    Args:
        padding_mask: Boolean mask where True indicates valid positions
    Returns:
        attention_mask: Boolean mask for attention weights (3D: [batch, q_len, k_len])
    """
    if padding_mask is None:
        return None

    # Cast to boolean if it's float (multiplicative mask)
    if kops.dtype(padding_mask) != "bool":
        padding_mask = kops.cast(padding_mask, "bool")

    # Create 3D mask for attention: [batch_size, seq_len, seq_len]
    mask_shape = kops.shape(padding_mask)
    seq_len = mask_shape[1]

    # Expand to [batch_size, 1, seq_len] for broadcasting across queries
    padding_mask = kops.expand_dims(padding_mask, 1)

    # Tile to [batch_size, seq_len, seq_len]
    padding_mask = kops.tile(padding_mask, [1, seq_len, 1])

    return padding_mask


@keras.saving.register_keras_serializable(package="neuroencoders")
class PositionalEncoding(tf.keras.layers.Layer):
    # increase max_len if you have longer sequences
    def __init__(self, max_len=200, d_model=128, **kwargs):
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.supports_masking = True

    def build(self, input_shape):
        # Create positional encoding matrix
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe)

    def call(self, x):
        with get_device_context(self.device):
            seq_len = tf.shape(x)[1]
            pe = kops.cast(self.pe[:seq_len, :], x.dtype)
        return x + pe

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "max_len": self.max_len,
            "d_model": self.d_model,
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        layer_config = {
            "max_len": config.get("max_len", 500),
            "d_model": config.get("d_model", 128),
            "device": config.get("device", "/cpu:0"),
        }
        return cls(**layer_config)

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable(package="neuroencoders")
class SafeMaskCreation(tf.keras.layers.Layer):
    """
    Create boolean mask without unnecessary casting.
    True where valid (not equal to -1).
    """

    def call(self, batchedInputGroups):
        pad_value = -1
        # If input is sparse, convert to dense immediately
        if isinstance(batchedInputGroups, tf.SparseTensor):
            batchedInputGroups = tf.sparse.to_dense(
                batchedInputGroups,
                default_value=pad_value,
            )

        # Create boolean mask
        padding_mask = kops.not_equal(batchedInputGroups, pad_value)
        return padding_mask

    def compute_output_shape(self, input_shape):
        return input_shape


def safe_mask_creation(batchedInputGroups, pad_value=-1):
    """
    Create boolean mask without unnecessary casting.
    True where valid (not equal to pad_value).
    """
    # If input is sparse, convert to dense immediately
    if isinstance(batchedInputGroups, tf.SparseTensor):
        batchedInputGroups = tf.sparse.to_dense(
            batchedInputGroups,
            default_value=pad_value,
        )

    # Create boolean mask
    padding_mask = kops.not_equal(batchedInputGroups, pad_value)
    return padding_mask


@keras.saving.register_keras_serializable(package="neuroencoders")
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
        A custom Transformer Encoder Block layer with multi-head attention and feedforward network.
        Adapted from
    Wairagkar, M. et al. (2025) ‘An instantaneous voice-synthesis neuroprosthesis’, Nature, pp. 1–8. Available at: https://doi.org/10.1038/s41586-025-09127-3.
    """

    def __init__(
        self,
        d_model=64,
        num_heads=8,
        ff_dim1=256,
        ff_dim2=64,
        dropout_rate=0.5,
        device="/cpu:0",
        **kwargs,
    ):
        self.residual = kwargs.pop("residual", True)
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim1 = ff_dim1
        self.ff_dim2 = ff_dim2
        self.dropout_rate = dropout_rate
        self.device = device

        with get_device_context(self.device):
            # Layer normalization at the beginning
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            # Multi-head attention
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )

            # Dropout after attention
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

            # Feedforward network
            self.ff_layer1 = tf.keras.layers.Dense(self.ff_dim1, activation="relu")
            self.ff_layer2 = tf.keras.layers.Dense(self.ff_dim2, activation="relu")

            # Final layer normalization
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True  # To indicate that this layer supports masking

    def build(self, input_shape):
        """
        Build all child layers with proper input shapes.

        Args:
            input_shape: Expected to be (batch_size, sequence_length, d_model)
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch, seq, features), got {input_shape}"
            )

        batch_size, seq_length, feature_dim = input_shape

        # Ensure feature dimension matches d_model
        if feature_dim != self.d_model:
            raise ValueError(
                f"Input feature dimension {feature_dim} doesn't match d_model {self.d_model}"
            )

        with get_device_context(self.device):
            # Build layer normalization layers
            self.norm1.build(input_shape)
            self.norm2.build(input_shape)

            # Build multi-head attention
            # MHA expects (query_shape, key_shape, value_shape)
            self.mha.build(input_shape, input_shape, input_shape)

            # Build dropout (doesn't need explicit build but good practice)
            self.dropout1.build(input_shape)

            # Build feedforward layers
            self.ff_layer1.build(input_shape)

            # ff_layer2 input shape depends on ff_layer1 output
            ff1_output_shape = (batch_size, seq_length, self.ff_dim1)
            self.ff_layer2.build(ff1_output_shape)

        # Mark this layer as built
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the transformer encoder block.
        The output maintains the same shape as input but with ff_dim2 features.
        """
        batch_size, seq_length, _ = input_shape
        return (batch_size, seq_length, self.ff_dim2)

    def call(self, x, mask=None, training=False):
        with get_device_context(self.device):
            # Layer norm at the beginning
            x_norm = self.norm1(x)

            # create attention mask if needed
            attention_mask = None
            if mask is not None:
                attention_mask = create_attention_mask_from_padding_mask(mask)

            # Multi-head attention with residual connection
            attn_output = self.mha(
                query=x_norm,
                value=x_norm,
                attention_mask=attention_mask,
                query_mask=None,
                key_mask=None,
                training=training,
            )
            attn_output = self.dropout1(attn_output, training=training)

            if self.residual:
                x = x + attn_output  # Residual connection

            # Feedforward network
            ff_output = self.ff_layer1(x)
            ff_output = self.ff_layer2(ff_output)

            # Final layer norm and residual connection
            x = self.norm2(x + ff_output)

        return x

    def compute_mask(self, inputs, mask=None):
        """
        Propagate the input mask to the output.
        """
        return mask

    def get_config(self):
        """Return the config of the layer for serialization."""
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim1": self.ff_dim1,
                "ff_dim2": self.ff_dim2,
                "dropout_rate": self.dropout_rate,
                "device": self.device,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        # The deserialized layers are not used; just pass config values to the constructor.
        layer_config = {
            "d_model": config.get("d_model", 64),
            "num_heads": config.get("num_heads", 8),
            "ff_dim1": config.get("ff_dim1", 256),
            "ff_dim2": config.get("ff_dim2", 64),
            "dropout_rate": config.get("dropout_rate", 0.5),
            "device": config.get("device", "/cpu:0"),
        }
        return cls(**layer_config)


########### END OF TRANSFORMER ENCODER CLASS #####################


########### SPIKE STORAGE AND PARCING FUNCTIONS #####################
def get_spike_sequences(params, generator):
    # WARNING: This function is actually not used in the code, it might be a helper function to understand the pipeline of the spike sequence??
    """
    Warning: This function is not used in the code.
    Could be used in the main neuroEncoder function to get the Spike sequence from the spike generator
    and cast it into an "example" format that will then be decoded by tensorflow inputs system tf.io as the key word yield is used, this function effectively returns a generator.

    The goal of the function is to bin the set of spikes with respect to times, gather spikes in time windows of fix length.

    args:
    params: the parameters of the network
    generator: the generator that yields the spikes
    """

    windowStart = None

    length = 0
    times = []
    groups = []
    allSpikes = [
        [] for _ in range(params.nGroups)
    ]  # nGroups of array each containing the spike of a group
    for pos_index, grp, time, spike, pos in generator:
        if windowStart is None:
            windowStart = (
                time  # at the first pass: initialize the windowStart on "time"
            )

        if time > windowStart + params.windowLength:
            # if we got over the window-length
            allSpikes = [
                np.zeros([0, params.nChannelsPerGroup[g], 32])
                if allSpikes[g] == []
                else np.stack(allSpikes[g], axis=0)
                for g in range(params.nGroups)
            ]  # stacks each list of array in allSpikes
            # allSpikes then is composed of nGroups array of stacked "spike"
            res = {
                "pos_index": pos_index,
                "pos": pos,
                "groups": groups,
                "length": length,
                "times": times,
            }
            res.update({"spikes" + str(g): allSpikes[g] for g in range(params.nGroups)})
            yield res
            # increase the windowStart by one window length
            length = 0
            groups = []
            times = []
            allSpikes = [
                [] for _ in range(params.nGroups)
            ]  # The all Spikes is reset so that we stop gathering the spikes in this window
            windowStart += params.windowLength
            # Pierre: Then we increment the windowStart until it is above the last seen spike time
            while time > windowStart + params.windowLength:
                # res = {"train": train, "pos": pos, "groups": [], "length": 0, "times": []}
                # res.update({"spikes"+str(g): np.zeros([0, params.nChannels[g], 32]) for g in range(params.nGroups)})
                # yield res
                windowStart += params.windowLength
        # Pierre: While we have not entered a new window, we start to gather spikes, time and group
        # of each input.
        times.append(time)
        groups.append(grp)
        # Pierre: so here we understand that groups indicate for each spikes array
        # obtained from the generator the groups from which they belong to !
        # But the spike array are well mapped separately to different groups:
        allSpikes[grp].append(spike)
        length += 1
        # --> so length correspond to the number of spike sequence obtained from the generator for each window considered


def serialize_spike_sequence(params, pos_index, pos, groups, length, times, *spikes):
    """
    Moves from the info obtained via the SpikeDetector -> spikeGenerator -> getSpikeSequences pipeline toward the tensorflow storing file.
    This take a specific format, which is here declared through the dict+tf.train.Feature organisation. We see that groups now correspond to the "spikes" we had before....
    """

    feat = {
        "pos_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[pos_index])),
        "pos": tf.train.Feature(float_list=tf.train.FloatList(value=pos)),
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[length])),
        "groups": tf.train.Feature(int64_list=tf.train.Int64List(value=groups)),
        "time": tf.train.Feature(float_list=tf.train.FloatList(value=[np.mean(times)])),
    }
    # Pierre: convert the spikes dict into a tf.train.Feature, used for the tensorflow protocol.
    # their is no reason to change the key name but still done here.
    for g in range(params.nGroups):
        feat.update(
            {
                "group" + str(g): tf.train.Feature(
                    float_list=tf.train.FloatList(value=spikes[g].ravel())
                )
            }
        )

    example_proto = tf.train.Example(features=tf.train.Features(feature=feat))
    return example_proto.SerializeToString()  # to string


def serialize_single_spike(clu, spike):
    feat = {
        "clu": tf.train.Feature(int64_list=tf.train.Int64List(value=[clu])),
        "spike": tf.train.Feature(float_list=tf.train.FloatList(value=spike.ravel())),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feat))
    return example_proto.SerializeToString()


@tf.function
def parse_serialized_sequence(
    params: Params,
    tensors: Dict[str, tf.Tensor],
    count_spikes: bool = False,
    sorted_indices: bool = None,
    max_spikes: int = None,
    max_spikes_per_group: int = None,
):
    # TODO: add sorted indices to the function, in order to filter by indexInDat (eg spike sorting)
    tensors = dict(tensors)

    if max_spikes is not None:
        actual_total = tf.shape(
            tf.sparse.to_dense(tensors["groups"], default_value=-1)
        )[0]
        tf.debugging.assert_less_equal(
            actual_total,
            max_spikes,
            message=f"A sample exceeded {max_spikes} total spikes. Actual: {actual_total}. Consider increasing max_spikes or filtering your dataset.",
        )
    # 1. Handle Metadata (Vectorized to avoid CPU overhead)
    lengths = []
    for key in ["pos", "groups", "indexInDat"]:
        if isinstance(tensors[key], tf.SparseTensor):
            default = -1.0 if key == "pos" else -1
            tensors[key] = tf.sparse.to_dense(tensors[key], default_value=default)
        if key == "pos":
            tensors[key] = tf.reshape(tensors[key], [params.dimOutput])
        else:
            tensors[key] = tf.reshape(tensors[key], [-1])

    for g in range(params.nGroups):
        group_key = f"group{g}"

        # 2. Convert to Dense once
        tensors[group_key] = tf.sparse.to_dense(tensors[group_key], default_value=-1.0)

        # 3. Reshape [Total_Values] -> [Num_Spikes, Channels, 32]
        tensors[group_key] = tf.reshape(
            tensors[group_key], [-1, params.nChannelsPerGroup[g], 32]
        )

        # 4. Optimized "Valid Spike" Detection
        # Instead of heavy reduce_sum + cast, we check if the max value is > -1.0
        # If a spike is all -1.0 (default), it's a padding spike.
        isValid = tf.reduce_any(tf.not_equal(tensors[group_key], -1.0), axis=[1, 2])

        # 5. Use boolean_mask instead of gather(where)
        tensors[group_key] = tf.boolean_mask(tensors[group_key], isValid)
        if max_spikes_per_group is not None:
            tf.debugging.assert_less_equal(
                tf.shape(tensors[group_key])[0],
                max_spikes_per_group,
                message=f"Group {g} exceeded {max_spikes_per_group} spikes. Actual: {tf.shape(tensors[group_key])[0]}. Consider increasing max_spikes_per_group or filtering your dataset.",
            )

        lengths.append(tf.shape(tensors[group_key])[0])

        if count_spikes:
            tensors[f"group{g}_spikes_count"] = tf.shape(tensors[group_key])[0]
    tensors["max_spikes"] = tf.reduce_max(lengths)

    return tensors


@tf.function
def parse_serialized_spike(featDesc, ex_proto):
    """
    Parse a serialized spike example.
    Args:
        featDesc: Feature description for parsing
        ex_proto: Serialized TFRecord example

    Returns:
        Parsed tensors
    """
    tensors = tf.io.parse_single_example(serialized=ex_proto, features=featDesc)
    return tensors


########### SPIKE STORAGE AND PARCING FUNCTIONS #####################


def import_true_pos(feature):
    """
    Returns a function that adds true position (the feature array) to the parsed tensors.
    """
    feature_tensor = tf.convert_to_tensor(feature)

    def change_feature(vals):
        idx = tf.cast(vals["pos_index"], tf.int32)
        vals["pos"] = tf.gather(feature_tensor, idx)
        vals["pos"] = tf.reshape(
            vals["pos"], [feature_tensor.shape[1]]
        )  # reshape to (2,) for consistency
        return vals

    return change_feature


def squeeze_or_expand_to_same_rank(x1, x2, expand_rank_1=True):
    """Squeeze/expand last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = tf.expand_dims(x2, axis=-1)
            else:
                x1 = tf.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = tf.expand_dims(x1, axis=-1)
            else:
                x2 = tf.squeeze(x2, axis=-1)
    return x1, x2


class NeuralDataAugmentation:
    """Neural data augmentation pipeline for TFRecord datasets."""

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize augmentation parameters.

        kwargs:
            keep_original: Whether to keep the original trial (default: True)
            num_augmentations: Number of augmented copies per trial (4-20 range)
            white_noise_std: Standard deviation for white noise (default: 5.0)
            offset_noise_std: Standard deviation for constant offset (default: 1.6)
            offset_scale_factor: Scale factor for threshold crossings offset (default: 0.67)
            cumulative_noise_std: Standard deviation for cumulative noise (default: 0.02)
            spike_band_channels: List of spike-band channel indices (if None, assumes all channels)
        """
        self.keep_original = kwargs.get("keep_original", True)
        self.num_augmentations = kwargs.get("num_augmentations", 11)
        self.white_noise_std = kwargs.get("white_noise_std", 2.0)
        self.offset_noise_std = kwargs.get("offset_noise_std", 1.0)
        self.offset_scale_factor = kwargs.get("offset_scale_factor", 0.67)
        self.cumulative_noise_std = kwargs.get("cumulative_noise_std", 0.02)
        spike_band_channels = kwargs.get("spike_band_channels", None)
        self.spike_band_channels = (
            spike_band_channels if spike_band_channels is not None else []
        )
        self.device = kwargs.get("device", "/cpu:0")

    def add_white_noise(self, neural_data: tf.Tensor) -> tf.Tensor:
        """
        Add white noise to all time points of all channels independently.

        Args:
            neural_data: Tensor of any shape

        Returns:
            Augmented neural data with white noise
        """
        noise = tf.random.normal(
            shape=tf.shape(neural_data),
            mean=0.0,
            stddev=self.white_noise_std,
            dtype=neural_data.dtype,
        )
        return neural_data + noise

    def add_constant_offset(self, neural_data: tf.Tensor, axis: int = -2) -> tf.Tensor:
        """
        Add constant offset to channels along specified axis.

        Args:
            neural_data: Input tensor
            axis: Axis along which to apply offset (default: -2, second-to-last dimension)

        Returns:
            Augmented neural data with constant offset
        """
        # Create offset shape - same as neural_data but with 1 along time dimension
        shape = tf.shape(neural_data)
        offset_shape = tf.concat(
            [
                shape[: axis + 1],  # Keep dimensions up to and including channel axis
                [1],  # Make time dimension 1 for broadcasting
                shape[axis + 2 :],  # Keep remaining dimensions
            ],
            axis=0,
        )

        # Generate offset noise
        offset = tf.random.normal(
            shape=offset_shape,
            mean=0.0,
            stddev=self.offset_noise_std,
            dtype=neural_data.dtype,
        )

        # Apply offset to neural data
        augmented_data = neural_data + offset

        return augmented_data

    def add_cumulative_noise(
        self, neural_data: tf.Tensor, time_axis: int = -1
    ) -> tf.Tensor:
        """
        Add cumulative (random walk) noise along the specified time axis.

        Args:
            neural_data: Input tensor
            time_axis: Axis along which to apply cumulative noise (default: -1, last dimension)

        Returns:
            Augmented neural data with cumulative noise
        """
        # Generate random noise for each time step
        noise_increments = tf.random.normal(
            shape=tf.shape(neural_data),
            mean=0.0,
            stddev=self.cumulative_noise_std,
            dtype=neural_data.dtype,
        )

        # Compute cumulative sum along time axis to create random walk
        cumulative_noise = tf.cumsum(noise_increments, axis=time_axis)

        return neural_data + cumulative_noise

    @tf.function
    def augment_sample(
        self, neural_data: tf.Tensor, time_axis: int = -1, channel_axis: int = -2
    ) -> tf.Tensor:
        """
        Apply all augmentation strategies to a sample.

        Args:
            neural_data: Neural features tensor
            time_axis: Axis representing time dimension
            channel_axis: Axis representing channel dimension

        Returns:
            Augmented neural data
        """
        # Apply white noise
        augmented_data = self.add_white_noise(neural_data)

        # Apply constant offset
        augmented_data = self.add_constant_offset(augmented_data, axis=channel_axis)

        # Apply cumulative noise
        augmented_data = self.add_cumulative_noise(augmented_data, time_axis=time_axis)

        return augmented_data

    @tf.function
    def augment_spike_group_vectorized(self, group_data: tf.Tensor) -> tf.Tensor:
        # 1. Prepare shapes for broadcasting
        # result shape: [num_augs, spikes, channels, time]
        num_spikes = tf.shape(group_data)[0]
        channels = tf.shape(group_data)[1]
        time_bins = tf.shape(group_data)[2]

        output_shape = [self.num_augmentations, num_spikes, channels, time_bins]

        # 2. Add White Noise (Independently for every augmentation)
        # Total randomness in one go
        white_noise = tf.random.normal(output_shape, stddev=self.white_noise_std)

        # 3. Constant Offset (One per channel per augmentation)
        # Shape: [num_augs, 1, channels, 1]
        offset_noise = tf.random.normal(
            [self.num_augmentations, 1, channels, 1], stddev=self.offset_noise_std
        )

        # 4. Cumulative Noise (Random walk per augmentation)
        # Shape: [num_augs, num_spikes, channels, time]
        cum_noise = tf.cumsum(
            tf.random.normal(output_shape, stddev=self.cumulative_noise_std), axis=-1
        )

        # 5. Combine everything using broadcasting
        # group_data [None, spikes, channels, time] + noises
        return group_data[tf.newaxis, ...] + white_noise + offset_noise + cum_noise

    def augment_spike_group(self, group_data: tf.Tensor) -> tf.Tensor:
        """
        Apply augmentation to spike group data with shape [num_spikes, channels, time_bins].

        Args:
            group_data: Tensor of shape [num_spikes, channels, time_bins]

        Returns:
            Augmented group data
        """
        return self.augment_sample(group_data, time_axis=2, channel_axis=1)

    def create_augmented_copies(
        self, neural_data: tf.Tensor, time_axis: int = -1, channel_axis: int = -2
    ) -> Dict[str, tf.Tensor]:
        """
        Create multiple augmented copies of a single trial.

        Args:
            neural_data: Neural features tensor
            time_axis: Axis representing time dimension
            channel_axis: Axis representing channel dimension

        Returns:
            Dictionary containing stacked augmented data
        """
        augmented_samples = []

        for _ in range(self.num_augmentations):
            aug_data = self.augment_sample(neural_data, time_axis, channel_axis)
            augmented_samples.append(aug_data)

        # Stack all augmented samples
        result = {"neural_data": tf.stack(augmented_samples, axis=0)}

        return result

    def __repr__(self):
        return (
            f"NeuralDataAugmentation(num_augmentations={self.num_augmentations}, "
            f"keep_original={self.keep_original}, "
            f"white_noise_std={self.white_noise_std}, "
            f"offset_noise_std={self.offset_noise_std}, "
            f"offset_scale_factor={self.offset_scale_factor}, "
            f"cumulative_noise_std={self.cumulative_noise_std}, "
            f"spike_band_channels={self.spike_band_channels})"
        )

    def __call__(
        self, neural_data: tf.Tensor, time_axis: int = -1, channel_axis: int = -2
    ):
        return self.augment_sample(neural_data, time_axis, channel_axis)


@tf.function
def parse_serialized_sequence_with_augmentation(
    params, tensors, augmentation_config=None, count_spikes=False
):
    tensors = dict(tensors)
    lengths = []
    # 1. Clean up Metadata Metadata efficiently
    for key in ["groups", "indexInDat", "pos"]:
        if key in tensors:
            if isinstance(tensors[key], tf.SparseTensor):
                tensors[key] = tf.sparse.to_dense(
                    tensors[key], default_value=-1.0 if key == "pos" else -1
                )
            if key == "pos":
                tensors[key] = tf.reshape(tensors[key], [params.dimOutput])
            else:
                tensors[key] = tf.reshape(tensors[key], [-1])

    original_groups = {}
    for g in range(params.nGroups):
        g_key = f"group{g}"
        # Convert to dense and reshape once
        tensors[g_key] = tf.sparse.to_dense(tensors[g_key], default_value=-1.0)
        tensors[g_key] = tf.reshape(
            tensors[g_key], [-1, params.nChannelsPerGroup[g], 32]
        )

        # Optimized filtering: skip spikes that are all default (-1.0)
        isValid = tf.reduce_any(tf.not_equal(tensors[g_key], -1.0), axis=[1, 2])
        tensors[g_key] = tf.boolean_mask(tensors[g_key], isValid)
        lengths.append(tf.shape(tensors[g_key])[0])

        if count_spikes:
            tensors[f"group{g}_spikes_count"] = tf.shape(tensors[g_key])[0]

        original_groups[g_key] = tensors[g_key]

    tensors["max_spikes"] = tf.reduce_max(lengths)

    if augmentation_config is not None:
        # Pass only the necessary data to the stacking function
        return apply_group_augmentation(
            tensors, original_groups, params, augmentation_config
        )

    return tensors


@tf.function
def apply_group_augmentation(
    tensors: Dict[str, tf.Tensor],
    original_groups: Dict[str, tf.Tensor],
    params: Params,
    augmentation_config: NeuralDataAugmentation,
    count_spikes: bool = False,
):
    """
    Apply augmentation to each group and replicate metadata efficiently.
    """
    num_augs = augmentation_config.num_augmentations
    keep_original = getattr(augmentation_config, "keep_original", False)

    result_tensors = {}
    # --- 1. Vectorized Spike Augmentation ---
    for g in range(params.nGroups):
        g_key = f"group{g}"
        group_data = original_groups[g_key]  # Shape: [Spikes, Chan, Time]

        # Generate ALL noise for ALL augmentations in 3 big API calls
        # Shape: [num_augs, Spikes, Chan, Time]
        noise_shape = tf.concat([[num_augs], tf.shape(group_data)], axis=0)

        white = tf.random.normal(
            noise_shape, stddev=augmentation_config.white_noise_std
        )
        offset = tf.random.normal(
            [num_augs, 1, tf.shape(group_data)[1], 1],
            stddev=augmentation_config.offset_noise_std,
        )
        cum_noise = tf.cumsum(
            tf.random.normal(
                noise_shape, stddev=augmentation_config.cumulative_noise_std
            ),
            axis=-1,
        )

        # Broadcast the original data to match noise shape
        augmented_versions = group_data[tf.newaxis, ...] + white + offset + cum_noise

        if keep_original:
            result_tensors[g_key] = tf.concat(
                [group_data[tf.newaxis, ...], augmented_versions], axis=0
            )
        else:
            result_tensors[g_key] = augmented_versions

    # --- 2. Lightning Fast Metadata Replication ---
    # We don't re-calculate indices! We just repeat what Step 1 produced.
    metadata_keys = [
        "pos_index",
        "pos",
        "groups",
        "length",
        "time",
        "time_behavior",
        "indexInDat",
        "max_spikes",
    ] + [f"indices{g}" for g in range(params.nGroups)]
    if count_spikes:
        metadata_keys += [f"group{g}_spikes_count" for g in range(params.nGroups)]
    n_total = num_augs + (1 if keep_original else 0)

    for key in metadata_keys:
        if key in tensors:
            # Repeat the pre-calculated tensor N times
            result_tensors[key] = tf.repeat(
                tensors[key][tf.newaxis, ...], n_total, axis=0
            )

    return result_tensors


def parse_tfrecord_with_augmentation(
    example_proto: tf.Tensor,
    feature_description: Dict[str, tf.io.FixedLenFeature],
    augmentation_config: NeuralDataAugmentation,
) -> Dict[str, tf.Tensor]:
    """
    Parse TFRecord example and apply data augmentation.

    Args:
        example_proto: Serialized TFRecord example
        feature_description: Feature description for parsing
        augmentation_config: Augmentation configuration object

    Returns:
        Dictionary of parsed and augmented features
    """
    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Extract neural data (reshape as needed based on your data format)
    neural_data = parsed_features["neural_data"]  # Adjust key name as needed
    neural_data = tf.reshape(
        neural_data, [-1, tf.shape(neural_data)[-1]]
    )  # [time_steps, channels]

    # Extract labels
    parsed_features.get("labels", None)

    # Apply augmentation
    augmented_data = augmentation_config.create_augmented_copies(neural_data)

    return augmented_data


@keras.saving.register_keras_serializable(package="neuroencoders")
class LinearizationLayer(tf.keras.layers.Layer):
    """
    A simple layer to linearize Euclidean data into a maze-like linear track.
    Follows the same logic as the linearizer pykeops code.
    """

    def __init__(self, maze_points, ts_proj, **kwargs):
        """
        Args:
            maze_points : numpy array of shape (J,2) that represents some (x,y) anchor coordinates in the maze, that the euclidean data will be projected to. J is the number os spatial bins (default = 100)
            ts_proj : numpy array of shape (J,) that represents the linear position corresponding to each maze point.
            device : device to run the layer on, default is "/cpu:0"
        """
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)
        # Convert to TensorFlow constants
        if maze_points is None or ts_proj is None:
            raise ValueError("maze_points and ts_proj cannot be None")
        maze_points = np.array(maze_points).reshape(-1, 2)
        ts_proj = np.array(ts_proj).reshape(-1)
        self.maze_points = tf.constant(maze_points)
        self.ts_proj = tf.constant(ts_proj)

    def call(self, euclidean_data):
        """
        Project euclidean_data to the closest maze point and return the corresponding linear position.

        Args:
        euclidean_data : tensor of shape (batch, 2) that represents (x,y) coordinates in the Aligned maze (0,1)^2 coordinates.

        Returns a list of two tensors:
        projected_pos : the maze_points the euclidean_data was projected to, i.e. the closest anchor for linearization shape (batch_size, 2).
        linear_pos : a tensor of shape (N,) that represents linear position.

        """
        with get_device_context(self.device):
            # Expand dimensions for broadcasting
            # euclidean_data: [batch_size, features] -> [batch_size, 1, features]
            # maze_points: [num_points, features] -> [1, num_points, features]
            euclidean_expanded = kops.expand_dims(euclidean_data, axis=1)
            maze_expanded = kops.cast(
                kops.expand_dims(self.maze_points, axis=0), euclidean_data.dtype
            )

            # Calculate squared distances
            distance_matrix = kops.sum(
                kops.square(maze_expanded - euclidean_expanded), axis=-1
            )

            # Find argmin
            best_points = kops.cast(kops.argmin(distance_matrix, axis=1), tf.int32)

            # Gather results
            projected_pos = kops.take(self.maze_points, best_points)
            linear_pos = kops.take(self.ts_proj, best_points)

        return [projected_pos, linear_pos]

    def get_config(self):
        base_config = super().get_config()
        try:
            maze_points_list = self.maze_points.numpy().tolist()
            ts_proj_list = self.ts_proj.numpy().tolist()
        except AttributeError:
            # Handle case where these aren't TensorFlow tensors
            maze_points_list = (
                self.maze_points.tolist()
                if hasattr(self.maze_points, "tolist")
                else self.maze_points
            )
            ts_proj_list = (
                self.ts_proj.tolist()
                if hasattr(self.ts_proj, "tolist")
                else self.ts_proj
            )

        return {
            **base_config,
            "maze_points": maze_points_list,
            "ts_proj": ts_proj_list,
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        maze_points = config.get("maze_points", None)
        ts_proj = config.get("ts_proj", None)
        device = config.get("device", "/cpu:0")
        return cls(
            maze_points=maze_points,
            ts_proj=ts_proj,
            device=device,
        )

    def build(self, input_shape):
        """
        Build is called the first time the layer is used.
        No trainable weights are needed here, but we check input shape.
        """
        # input_shape: (batch_size, 2)
        if len(input_shape) != 2 or input_shape[-1] != 2:
            raise ValueError(
                f"Input to LinearizationLayer must be of shape (batch, 2), got {input_shape}"
            )
        super().build(input_shape)


@keras.saving.register_keras_serializable(package="neuroencoders")
class LinearPosWeighting(tf.keras.layers.Layer):
    """
    A layer to weight the first 2 dimensions of position outputs by
    the linearized positions before computing the loss.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Args:
            inputs: list or tuple of two tensors
                - myoutputPos: shape (batch_size, D)
                - lin_truePos: shape (batch_size,)  (linearized position)

        Returns:
            Weighted myoutputPos: shape (batch_size, D)
        """
        myoutputPos, lin_truePos = inputs

        # Expand lin_truePos to match first 2 dims of myoutputPos
        lin_truePos_exp = tf.expand_dims(lin_truePos, axis=-1)  # (batch_size, 1)

        # Weight first 2 dimensions
        weighted_output = tf.concat(
            [myoutputPos[:, :2] * lin_truePos_exp, myoutputPos[:, 2:]], axis=-1
        )
        return weighted_output


@keras.saving.register_keras_serializable(package="neuroencoders")
class DynamicDenseWeightLayer(tf.keras.layers.Layer):
    """Layer that calls fitted DenseWeight for each batch dynamically"""

    def __init__(self, fitted_denseweight, **kwargs):
        self.training_data = kwargs.pop("training_data", None)
        self.alpha = kwargs.pop("fitted_dw_alpha", 1.0)
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)
        self.fitted_dw = fitted_denseweight  # Pre-fitted DenseWeight object

    def _compute_batch_weights(self, linearized_pos):
        """Compute weights for a batch using fitted DenseWeight"""
        # Convert tensor to numpy for DenseWeight
        with get_device_context(self.device):
            if hasattr(linearized_pos, "numpy"):
                linearized_np = linearized_pos.numpy()
            else:
                linearized_np = np.array(linearized_pos)

            # Call the fitted DenseWeight to get weights for this batch
            # This uses the fitted model but computes weights for current samples
            batch_weights = self.fitted_dw.eval(linearized_np)

        return batch_weights.astype(np.float32)

    def call(self, linearized_pos):
        """
        Dynamically compute weights for current batch using fitted DenseWeight
        """
        with get_device_context(self.device):
            # Use tf.py_function to call the fitted DenseWeight
            weights = tf.py_function(
                func=self._compute_batch_weights, inp=[linearized_pos]
            )

            # Set shape (tf.py_function loses shape info)
            tf.shape(linearized_pos)[0]
            weights.set_shape([None])

        return weights

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "fitted_dw_alpha": self.alpha,
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        fitted_dw_config = config.get("fitted_dw_alpha")
        training_data = config.get("training_data")
        fitted_dw = DenseWeight(fitted_dw_config)
        config.get("device", "/cpu:0")
        if training_data is not None:
            fitted_dw.fit(training_data)
        # return cls(fitted_denseweight=fitted_dw, device=device)
        raise NotImplementedError(
            "Deserialization of DynamicDenseWeightLayer is not fully implemented. You must recreate it with the fitted DenseWeight instance."
        )


@keras.saving.register_keras_serializable(package="neuroencoders")
class UMazeProjectionLayer(tf.keras.layers.Layer, SpatialConstraintsMixin):
    def __init__(self, grid_size, smoothing_factor=0.01, maze_params=None, **kwargs):
        """
        Differentiable projection layer that softly constrains (x,y) predictions
        to lie within a U-shaped maze.

        Args:
            maze_params (dict): Defines maze geometry.
            smoothing_factor (float): Controls softness of constraints.
        """
        super().__init__(**kwargs)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        self.smoothing_factor = smoothing_factor

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x, y = inputs[..., 0], inputs[..., 1]
        x_proj, y_proj = self._project_points(x, y)

        proj = kops.stack([x_proj, y_proj], axis=-1)  # (batch, ..., 2)

        rest = inputs[..., 2:]  # if no extra dims, this is shape (..., 0)
        return kops.concatenate([proj, rest], axis=-1)

    def _project_points(self, x, y):
        dtype = x.dtype
        gap_x_min = tf.constant(self.maze_params_dict["gap_x_min"], dtype=dtype)
        gap_x_max = tf.constant(self.maze_params_dict["gap_x_max"], dtype=dtype)
        gap_y_min = tf.constant(self.maze_params_dict["gap_y_min"], dtype=dtype)

        # Define constraint lines
        lines = tf.stack(
            [
                [gap_x_min, 0.0, gap_x_min, gap_y_min],  # left vertical
                [gap_x_max, 0.0, gap_x_max, gap_y_min],  # right vertical
                [gap_x_min, gap_y_min, gap_x_max, gap_y_min],  # top horizontal
            ],
            axis=0,
        )  # (3,4)

        # Expand predictions (N,1)
        px, py = tf.expand_dims(x, -1), tf.expand_dims(y, -1)

        # Unpack line endpoints
        x1, y1, x2, y2 = [lines[:, i][tf.newaxis, :] for i in range(4)]  # (1,3)

        # Project onto each line
        dx, dy = x2 - x1, y2 - y1
        t = tf.clip_by_value(
            ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2 + 1e-8), 0.0, 1.0
        )
        proj_x, proj_y = x1 + t * dx, y1 + t * dy  # (N,3)

        # Distances
        dist = tf.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)  # (N,3)

        # Find closest projection
        min_idx = tf.argmin(dist, axis=-1, output_type=tf.int32)  # (N,)
        closest_proj_x = tf.gather(proj_x, min_idx, axis=1, batch_dims=1)
        closest_proj_y = tf.gather(proj_y, min_idx, axis=1, batch_dims=1)
        closest_dist = tf.gather(dist, min_idx, axis=1, batch_dims=1)

        batch_size = tf.shape(x)[0]

        # --- Noise (scaled by distance) ---
        left_noise_x = (
            -tf.random.uniform((batch_size,), 0.0, 0.5, dtype=dtype) * closest_dist
        )
        right_noise_x = (
            tf.random.uniform((batch_size,), 0.0, 0.5, dtype=dtype) * closest_dist
        )
        global_noise_y = (
            tf.random.normal((batch_size,), mean=0.0, stddev=0.3, dtype=dtype)
            * closest_dist
        )
        top_noise_x = (
            tf.random.normal((batch_size,), mean=0.0, stddev=0.2, dtype=dtype)
            * closest_dist
        )
        top_noise_y = (
            tf.random.uniform((batch_size,), 0.0, 0.5, dtype=dtype) * closest_dist
        )

        noise_x = tf.stack([left_noise_x, right_noise_x, top_noise_x], axis=1)  # (N,3)
        noise_y = tf.stack([global_noise_y, global_noise_y, top_noise_y], axis=1)

        chosen_noise_x = tf.gather(noise_x, min_idx, axis=1, batch_dims=1)
        chosen_noise_y = tf.gather(noise_y, min_idx, axis=1, batch_dims=1)

        proj_x_noisy = closest_proj_x + chosen_noise_x
        proj_y_noisy = closest_proj_y + chosen_noise_y

        # Soft inside indicator
        inside_soft = (
            tf.sigmoid((gap_x_max - x) / self.smoothing_factor)
            * tf.sigmoid((x - gap_x_min) / self.smoothing_factor)
            * tf.sigmoid((gap_y_min - y) / self.smoothing_factor)
        )

        x_final = (1 - inside_soft) * x + inside_soft * proj_x_noisy
        y_final = (1 - inside_soft) * y + inside_soft * proj_y_noisy

        # Clip to maze corridor
        x_final = tf.clip_by_value(
            x_final, self.maze_params_dict["x_min"], self.maze_params_dict["x_max"]
        )
        y_final = tf.clip_by_value(
            y_final, self.maze_params_dict["y_min"], self.maze_params_dict["y_max"]
        )

        return x_final, y_final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "maze_params": self.maze_params_dict,
                "smoothing_factor": self.smoothing_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        maze_params = config.get("maze_params", None)
        smoothing_factor = config.get("smoothing_factor", 0.01)
        grid_size = config.get("grid_size", 50)
        return cls(
            maze_params=maze_params,
            smoothing_factor=smoothing_factor,
            grid_size=grid_size,
        )


# Custom layer that combines feature_output and UMazeProjectionLayer
@keras.saving.register_keras_serializable(package="neuroencoders")
class FeatureOutputWithUMaze(tf.keras.layers.Layer):
    def __init__(self, orig_layer_config, maze_params=None, **kwargs):
        super().__init__(**kwargs)
        # Rebuild the original layer (Dense in your case)
        self.orig = tf.keras.layers.Dense.from_config(orig_layer_config)
        self.proj = UMazeProjectionLayer(maze_params=maze_params)

    def call(self, inputs, **kwargs):
        x = self.orig(inputs, **kwargs)
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "orig_layer_config": self.orig.get_config(),
                "maze_params": self.proj.maze_params_dict,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        orig_layer_config = config.get("orig_layer_config")
        maze_params = config.get("maze_params", None)
        return cls(orig_layer_config=orig_layer_config, maze_params=maze_params)

    # ---- Weight management ----
    def get_weights(self):
        # Only the Dense has trainable weights
        return self.orig.get_weights()

    def set_weights(self, weights):
        # Load into the Dense
        self.orig.set_weights(weights)

    @property
    def trainable_weights(self):
        # Expose only Dense's trainable weights
        return self.orig.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.orig.non_trainable_weights


def clone_model_with_custom_layer(layer):
    if layer.name == "feature_output":
        print(" --> Replacing with custom stack")
        return FeatureOutputWithUMaze(
            layer.get_config(), name=layer.name + "_with_proj"
        )
    # TODO: at some points, implement maze_coords
    return layer


def get_last_dense_layers_before_output(
    model, output_layer_name="feature_output_with_proj", k=2
):
    """
    Finds the last k Dense layers that feed into the given output layer.

    Args:
        model: Keras Functional model
        output_layer_name: name of the custom output layer
        k: number of Dense layers to return (default 2)

    Returns:
        List of Keras layer objects
    """
    output_layer = model.get_layer(output_layer_name)

    # Get all layers connected to it (recursively)
    visited = set()
    stack = [output_layer]
    dense_layers = []

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        # Collect Dense layers
        if isinstance(current, tf.keras.layers.Dense):
            dense_layers.append(current)

        # Add inbound layers to stack
        for node in current._inbound_nodes:
            inbound_layers = node.inbound_layers
            if not isinstance(inbound_layers, list):
                inbound_layers = [inbound_layers]
            stack.extend(inbound_layers)

    # Return the last k Dense layers in order of appearance
    return dense_layers[:k][::-1]  # reverse so closest layers come last


@keras.saving.register_keras_serializable(package="neuroencoders")
class GaussianHeatmapLayer(tf.keras.layers.Layer, SpatialConstraintsMixin):
    """
    Layer that generates Gaussian heatmaps for given true positions.
    This layer computes a Gaussian heatmap based on the true positions
    """

    def __init__(
        self,
        training_positions,
        grid_size,
        eps=1e-8,
        sigma=0.03,
        neg=-100,
        maze_params=None,
        **kwargs,
    ):
        tf.keras.layers.Layer.__init__(self, **kwargs)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        # Convert numpy array to Python list immediately for serialization
        if hasattr(training_positions, "tolist"):
            self.training_positions = (
                training_positions  # Keep original for computation
            )
            self._training_positions_serializable = (
                training_positions.tolist()
            )  # For config
        else:
            self.training_positions = training_positions
            self._training_positions_serializable = training_positions
        self.grid_size = grid_size
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.neg = float(neg)
        self.maze_params = maze_params

        self._initialize_computed_attributes()

        # final dense layer to map features to logits
        self.feature_to_logits_map = tf.keras.layers.Dense(self.GRID_H * self.GRID_W)

    def _initialize_computed_attributes(self):
        self.EPS = self.common_eps
        self.NEG = self.common_neg
        if self.training_positions is not None:
            self._validate_training_positions()
            self.occ = self.occupancy_map(self.training_positions)
            self.WMAP = self.weight_map_from_occ(self.occ, alpha=0.5)
        elif not hasattr(self, "WMAP"):
            self.WMAP = tf.ones((self.GRID_H, self.GRID_W), dtype=tf.float32)
        self.gaussian_kernel = self._create_gaussian_kernel(self.sigma)

    def _create_gaussian_kernel(self, sigma):
        """Create a 2D Gaussian kernel for smoothing logits"""
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return tf.constant(kernel)

    def _validate_training_positions(self):
        """Validate that training positions don't fall in forbidden regions"""
        bins = self.positions_to_bins(self.training_positions)
        x_indices = bins % self.GRID_W
        y_indices = bins // self.GRID_W
        forbidden_positions = self.forbid_mask_np[y_indices, x_indices] > 0
        # Filters out forbidden positions and warns user
        n_forbidden = np.sum(forbidden_positions)
        if n_forbidden > 0:
            self.training_positions = self.training_positions[~forbidden_positions]

    def call(self, inputs, flatten=True):
        """
        Forward pass through the layer.

        Args:
            inputs: Tensor of shape [B, feature_dim]

        Returns:
            logits_hw: Tensor of shape [B, H*W] representing unnormalized logits
        """
        logits_flat = self.feature_to_logits_map(inputs)
        logits_hw = kops.reshape(logits_flat, (-1, self.GRID_H, self.GRID_W))
        # smooth logits with a gaussian kernel to avoid spiky predictions
        kernel = kops.cast(self.gaussian_kernel[:, :, None, None], logits_hw.dtype)
        logits_hw = tf.nn.conv2d(
            logits_hw[:, :, :, None],
            kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )[:, :, :, 0]
        if not flatten:
            return logits_hw
        logits_flat = kops.reshape(logits_hw, (-1, self.GRID_H * self.GRID_W))
        return logits_flat

    def gaussian_heatmap_targets(self, pos_batch, sigma=None):
        """Unified method for generating Gaussian targets using Mixin logic"""
        if sigma is None:
            sigma = self.sigma
        return self.gaussian_heatmap_targets_tf(pos_batch, sigma=sigma)

    def decode_and_uncertainty(self, logits_hw, mode="argmax", return_probs=False):
        """Unified decoding logic using Mixin's method"""
        return self.decode_and_uncertainty_tf(
            logits_hw, mode=mode, return_probs=return_probs
        )

    def positions_to_bins(self, pos):
        xs = np.clip((pos[:, 0] * self.GRID_W).astype(int), 0, self.GRID_W - 1)
        ys = np.clip((pos[:, 1] * self.GRID_H).astype(int), 0, self.GRID_H - 1)
        return ys * self.GRID_W + xs

    def occupancy_map(self, positions):
        occ = np.zeros((self.GRID_H, self.GRID_W), np.float32)
        idx = self.positions_to_bins(positions)
        for k in idx:
            occ[k // self.GRID_W, k % self.GRID_W] += 1

        allowed_mask = self.get_allowed_mask(use_tensorflow=False)
        allowed_mask = allowed_mask.astype(np.float32)
        return occ * allowed_mask

    def weight_map_from_occ(
        self,
        occ,
        alpha=0.05,
        eps=None,
        smooth_sigma=1.0,
        max_weight=15.0,
        log_scale=False,
        remove_isolated_zeros=True,
    ):
        """
        Compute weight map from occupancy counts, ignoring forbidden and zero-count bins.

        - Forbidden bins: always weight=0
        - Zero-count bins: always weight=0 (ignored)
        """
        if eps is None:
            eps = self.EPS

        # Mask forbidden regions early
        allowed_mask = self.get_allowed_mask(use_tensorflow=False)
        forbid_mask = ~allowed_mask.astype(bool)
        occ = occ.copy()
        occ[forbid_mask] = 0.0

        # Optional smoothing (but ignore forbid bins!)
        if smooth_sigma is not None and smooth_sigma > 0:
            # 1. Create a float mask (1.0 inside, 0.0 outside)
            mask_weights = allowed_mask.astype(float)

            # 2. Smooth the data (zeros outside are treated as "missing" by step 4)
            occ_smoothed = gaussian_filter(
                occ, sigma=smooth_sigma, mode="constant", cval=0.0
            )

            # 3. Smooth the mask (calculates the "validity" weight of each pixel)
            mask_smoothed = gaussian_filter(
                mask_weights, sigma=smooth_sigma, mode="constant", cval=0.0
            )

            # 4. Normalize: Divide smoothed data by smoothed mask
            # We use np.divide with a 'where' clause to avoid dividing by zero outside the shape
            occ_normalized = np.zeros_like(occ)
            np.divide(
                occ_smoothed,
                mask_smoothed,
                out=occ_normalized,
                where=mask_smoothed > 1e-6,
            )

            # 5. Apply the result
            occ = occ_normalized

            # 6. Re-apply the hard mask to ensure the outside is perfectly zero
            occ[forbid_mask] = 0.0

        if remove_isolated_zeros:
            forbid_mask, occ = self.remove_isolated_zeros(forbid_mask, occ)

        # Define weights only on bins with occupancy > 0
        valid_mask = (occ > 0) & (~forbid_mask)

        if log_scale:
            inv = np.zeros_like(occ)
            inv[valid_mask] = 1.0 / np.log1p(occ[valid_mask] + eps)
        else:
            inv = np.zeros_like(occ)
            inv[valid_mask] = (1.0 / (occ[valid_mask] + eps)) ** alpha

        # Normalize mean weight on valid bins ≈ 1
        if np.any(valid_mask):
            inv[valid_mask] /= np.mean(inv[valid_mask])

        # Clip excessively large weights (only on valid bins)
        if max_weight is not None:
            inv[valid_mask] = np.clip(inv[valid_mask], 0.0, max_weight)

        # Forbidden + zero-count bins remain 0
        return tf.constant(inv)

    def project_out_of_forbid(self, xy, forbid_box=None):
        """
        Project decoded positions back into allowed space if inside forbidden region.

        Args:
            xy: [B, 2] predicted positions
            forbid_box: (xmin, xmax, ymin, ymax)

        Returns:
            xy_projected: [B, 2] corrected positions
        """
        if forbid_box is None:
            forbid_box = (
                self.maze_params_dict["gap_x_min"],
                self.maze_params_dict["gap_x_max"],
                0.0,
                self.maze_params_dict["gap_y_min"],
            )
        xmin, xmax, ymin, ymax = forbid_box
        x, y = xy[:, 0], xy[:, 1]

        inside_x = tf.logical_and(x >= xmin, x <= xmax)
        inside_y = tf.logical_and(y >= ymin, y <= ymax)
        inside = tf.logical_and(inside_x, inside_y)

        # If inside forbidden region, snap to closest edge of the rectangle
        x_clamped = tf.where(x < xmin, xmin, tf.where(x > xmax, xmax, x))
        y_clamped = tf.where(y < ymin, ymin, tf.where(y > ymax, ymax, y))

        # Distance to each edge
        dx_left = tf.abs(x - xmin)
        dx_right = tf.abs(x - xmax)
        dy_bottom = tf.abs(y - ymin)
        dy_top = tf.abs(y - ymax)

        # Pick closest edge
        move_x_left = dx_left <= tf.minimum(dx_right, tf.minimum(dy_bottom, dy_top))
        move_x_right = dx_right <= tf.minimum(dx_left, tf.minimum(dy_bottom, dy_top))
        move_y_bot = dy_bottom <= tf.minimum(dy_top, tf.minimum(dx_left, dx_right))
        move_y_top = dy_top <= tf.minimum(dy_bottom, tf.minimum(dx_left, dx_right))

        # New coordinates
        new_x = tf.where(move_x_left, xmin, tf.where(move_x_right, xmax, x_clamped))
        new_y = tf.where(move_y_bot, ymin, tf.where(move_y_top, ymax, y_clamped))

        corrected = tf.stack([new_x, new_y], axis=-1)
        return tf.where(inside[:, None], corrected, xy)

    def fit_temperature(self, val_logits, val_targets, iters=200, lr=1e-2):
        """
        Fit temperature scaling parameter on validation set to minimize NLL.
        Args:
            val_logits: [N, H, W] logits from validation set
            val_targets: [N, H, W] target heatmaps from validation set
            iters: number of optimization steps
            lr: learning rate for optimizer
        Returns:
            T_cal: fitted temperature scalar
        """
        logT = tf.Variable(0.0, trainable=True)
        opt = tf.keras.optimizers.Adam(lr)
        for step in range(iters):
            with tf.GradientTape() as t:
                scaled = val_logits / tf.exp(logT)
                B, H, W = tf.shape(scaled)[0], tf.shape(scaled)[1], tf.shape(scaled)[2]
                scaled_flat = tf.reshape(
                    tf.where(self.forbid_mask_tf[None] > 0, self.NEG, scaled),
                    [B, H * W],
                )
                logp_flat = tf.nn.log_softmax(scaled_flat, axis=-1)
                logp = tf.reshape(logp_flat, [B, H, W])
                nll = -tf.reduce_mean(tf.reduce_sum(val_targets * logp, [1, 2]))

            opt.apply_gradients([(t.gradient(nll, logT), logT)])
            if step % 50 == 0 or step == iters - 1:
                print(
                    f"Temp fit step {step}: NLL={nll.numpy():.4f}, T={tf.exp(logT).numpy():.4f}"
                )
        # inference: probs = softmax(mask_logits / T_cal)
        return float(tf.exp(logT).numpy())

    def get_config(self):
        """Return the config dict for serialization"""
        config = tf.keras.layers.Layer.get_config(self)

        # Convert TensorFlow tensors to Python scalars
        neg_value = self.neg
        if hasattr(neg_value, "numpy"):
            neg_value = float(neg_value.numpy())

        config.update(
            {
                "training_positions": None,  # Avoid storing large arrays
                "grid_size": self.grid_size,
                "eps": float(self.eps),
                "sigma": float(self.sigma),
                "neg": neg_value,
                "maze_params": self.maze_params,
                "WMAP": self.WMAP.numpy().tolist() if hasattr(self, "WMAP") else None,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config dict"""
        wmap = config.pop("WMAP", None)
        obj = cls(**config)
        if wmap is not None:
            obj.WMAP = tf.constant(wmap, dtype=tf.float32)
            # Re-initialize computed attributes to ensure consistency
            obj._initialize_computed_attributes()
        return obj

    def build(self, input_shape):
        """Build the layer - called automatically by Keras"""
        super().build(input_shape)
        # Ensure computed attributes are initialized after build
        if not hasattr(self, "EPS"):
            self._initialize_computed_attributes()


@keras.saving.register_keras_serializable(package="neuroencoders")
class GaussianHeatmapLosses(tf.keras.layers.Layer, SpatialConstraintsMixin):
    """
    A simple wrapup class to hold various loss functions and handle keras symbols. Inherits from
    GaussianHeatmapLayer to access masks and constants.
    """

    def __init__(
        self,
        training_positions,
        grid_size,
        l_function_layer_params,
        eps=1e-8,
        sigma=0.03,
        neg=-100,
        maze_params=None,
        sinkhorn_eps=0.4,
        **kwargs,
    ):
        """
        Args:
            heatmap_layer: An instance of GaussianHeatmapLayer to provide masks and constants.
        """
        tf.keras.layers.Layer.__init__(self, **kwargs)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        # Convert numpy array to Python list immediately for serialization
        if hasattr(training_positions, "tolist"):
            self.training_positions = (
                training_positions  # Keep original for computation
            )
            self._training_positions_serializable = (
                training_positions.tolist()
            )  # For config
        else:
            self.training_positions = training_positions
            self._training_positions_serializable = training_positions

        self.grid_size = grid_size
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.neg = float(neg)
        self.maze_params = maze_params
        self.l_function_layer_params = l_function_layer_params
        self.l_function_layer = LinearizationLayer(
            maze_points=self.l_function_layer_params["maze_points"],
            ts_proj=self.l_function_layer_params["ts_proj"],
            device=self.l_function_layer_params.pop("device", None),
            name=self.l_function_layer_params.get("name", "linearization_layer"),
        )
        self.sinkhorn_eps = sinkhorn_eps

        self.allowed_mask_tf = tf.cast(
            self.get_allowed_mask(use_tensorflow=True), tf.float32
        )
        self.forbid_mask_tf = tf.cast(1.0 - self.allowed_mask_tf, tf.float32)
        self.NEG = tf.cast(tf.constant(self.neg), tf.float32)
        self.EPS = tf.cast(tf.constant(self.eps), tf.float32)

        self.allowed_mask_flat = kops.reshape(self.allowed_mask_tf, (-1,))  # [H*W]

        # keep only allowed coordinates
        mask_indices = kops.where(self.allowed_mask_flat > 0)[0]  # [N_allowed]
        mask_indices = kops.reshape(mask_indices, (-1,))  # ensure 1D
        # store to map [H,W] to allowed indices
        self.allowed_indices = self.add_variable(
            shape=mask_indices.shape,
            initializer=lambda shape, dtype: mask_indices,
            dtype="int32",
            trainable=False,
            name="allowed_indices",
        )

        self.N_valid = kops.shape(mask_indices)[0]
        self._precompute_cost_matrix()

    def call(self, inputs, loss_type="safe_kl", **kwargs):
        """
        Compute loss in a Keras symbolic-safe way.

        Args:
            inputs: Dictionary with keys 'logits' and 'targets', or tuple/list of (y_true, y_pred)
            loss_type: 'weighted' or 'kl' or 'safe_kl'

        Returns:
            loss: Scalar loss tensor
        """
        if not isinstance(inputs, dict):
            # assume (targets, logits) or [targets, logits] ie y_true, y_pred
            if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
                target_hw, logits_hw = (
                    kops.cast(inputs[0], "float32"),
                    kops.cast(inputs[1], "float32"),
                )
            else:
                raise ValueError("Expected dict or (logits, targets) pair")
        else:
            logits_hw = kops.cast(inputs["logits"], "float32")
            target_hw = kops.cast(inputs["targets"], "float32")

        if loss_type == "weighted":
            return self._weighted_heatmap_loss(logits_hw, target_hw, **kwargs)
        elif loss_type == "kl":
            return self._kl_heatmap_loss(logits_hw, target_hw, **kwargs)
        elif loss_type == "safe_kl":
            return self._safe_kl_heatmap_loss(logits_hw, target_hw, **kwargs)
        elif loss_type == "wasserstein":
            return self._safe_kl_wasserstein_heatmap_loss(
                logits_hw, target_hw, **kwargs
            )
        else:
            raise ValueError("Unknown loss_type:" + str(loss_type))

    def get_config(self):
        """Return the config dict for serialization"""
        config = tf.keras.layers.Layer.get_config(self)

        # Convert TensorFlow tensors to Python scalars
        neg_value = self.neg
        if hasattr(neg_value, "numpy"):
            neg_value = float(neg_value.numpy())

        # handle l function layer params serialization
        l_function_layer_params_serializable = self.l_function_layer_params.copy()
        for k, v in l_function_layer_params_serializable.items():
            try:
                v = v.numpy().tolist()
            except AttributeError:
                # Handle case where these aren't TensorFlow tensors
                v = v.tolist() if hasattr(v, "tolist") else v
            l_function_layer_params_serializable[k] = v

        config.update(
            {
                "training_positions": self._training_positions_serializable,
                "grid_size": self.grid_size,
                "eps": float(self.eps),
                "sigma": float(self.sigma),
                "neg": neg_value,
                "maze_params": self.maze_params,
                "l_function_layer_params": l_function_layer_params_serializable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config dict"""
        layer_config = {
            "training_positions": config.get("training_positions"),
            "grid_size": config.get("grid_size", (45, 45)),
            "eps": config.get("eps", 1e-8),
            "sigma": config.get("sigma", 0.03),
            "neg": config.get("neg", -100),
            "l_function_layer_params": config.get("l_function_layer_params", None),
            "maze_params": config.get("maze_params", None),
            "name": config.get("name", "gaussian_heatmap_losses"),
        }
        return cls(**layer_config)

    def build(self, input_shape):
        """Build the layer"""
        super().build(input_shape)

    @tf.function
    def _weighted_heatmap_loss(self, logits_hw, target_hw, wmap=None):
        batch_size = kops.shape(logits_hw)[0]

        masked_logits = kops.where(
            kops.expand_dims(self.forbid_mask_tf, 0) > 0, self.NEG, logits_hw
        )
        # Flatten for softmax
        logits_flat = kops.reshape(
            masked_logits, (batch_size, self.GRID_H * self.GRID_W)
        )
        probs_flat = kops.softmax(logits_flat, axis=-1)
        probs = kops.reshape(probs_flat, (batch_size, self.GRID_H, self.GRID_W))

        # apply weights
        if wmap is None:
            wmap = self.WMAP
        weights = kops.expand_dims(wmap, 0) * kops.expand_dims(self.allowed_mask_tf, 0)

        se = kops.square(probs - target_hw)
        # normalize by sum of weights to keep scale stable
        # Compute weighted loss
        weighted_se = se * weights
        loss_per_sample = kops.sum(weighted_se, axis=[1, 2]) / (
            kops.sum(weights) + self.EPS
        )

        return kops.mean(loss_per_sample)

    @tf.function
    def _kl_heatmap_loss(self, logits_hw, target_hw, wmap=None, scale=False):
        """
        Numerically stable KL divergence loss between target heatmap (P) and predicted (Q).
        """
        batch_size = kops.shape(logits_hw)[0]

        # Safety clipping to prevent extreme logits
        logits_hw = tf.clip_by_value(logits_hw, -20.0, 20.0)

        # Mask forbidden bins in logits
        safe_neg = self.NEG
        masked_logits = kops.where(
            kops.expand_dims(self.forbid_mask_tf, 0) > 0, safe_neg, logits_hw
        )

        # Get predicted probabilities (not log probabilities)
        probs_flat = kops.softmax(
            kops.reshape(masked_logits, (batch_size, self.GRID_H * self.GRID_W)),
            axis=-1,
        )
        probs = kops.reshape(probs_flat, (batch_size, self.GRID_H, self.GRID_W))

        # Process targets with safety checks
        allowed_mask = kops.expand_dims(self.allowed_mask_tf, 0)
        P = target_hw * allowed_mask
        P_sum = kops.sum(P, axis=[1, 2], keepdims=True)
        safe_eps = kops.maximum(self.EPS, 1e-8)

        uniform_fallback = allowed_mask / kops.sum(self.allowed_mask_tf)
        P = kops.where(
            P_sum > safe_eps,
            P / (P_sum + safe_eps),
            uniform_fallback,
        )

        # Ensure final normalization
        P_sum_final = kops.sum(P, axis=[1, 2], keepdims=True)
        P = P / (P_sum_final + safe_eps)

        # Define threshold for meaningful probability mass
        threshold = safe_eps * 10
        safe_probs = kops.maximum(probs, safe_eps)

        # CORRECTED KL FORMULA: Only compute KL where P has meaningful mass
        # KL(P(  )Q) = sum P * log(P/Q) only where P > threshold
        kl = kops.where(
            P > threshold,
            P * kops.log(P / safe_probs),
            0.0,  # Zero contribution where P is negligible
        )

        # Apply weighting
        if wmap is not None:
            weights = kops.expand_dims(wmap, 0)
            valid_mask = weights > 0
            kl = kl * weights
            loss_per_sample = kops.sum(kl, axis=[1, 2]) / (
                kops.sum(weights * valid_mask) + safe_eps
            )
            final_loss = kops.mean(loss_per_sample)

        else:
            # Compute final loss with scaling to prevent gradient explosion
            final_loss = kops.mean(kops.mean(kl, axis=[1, 2]))

        if scale:
            # Scale down the loss to prevent gradient explosion (divide by 100)
            return final_loss / 100.0

        return final_loss

    @tf.function
    def _safe_kl_heatmap_loss(
        self,
        logits_hw,
        target_hw,
        wmap=None,
        scale=False,
        return_batch=False,
        reduction=None,
    ):
        """
        Numerically stable KL divergence loss between target heatmap (P) and predicted (Q).
        Equivalent to KL(P||Q), but implemented using TensorFlow cross-entropy ops.
        """
        batch_size = tf.shape(logits_hw)[0]
        allowed_mask = self.allowed_mask_tf

        # Clip logits for stability and apply forbid mask
        masked_logits = kops.where(
            kops.expand_dims(self.forbid_mask_tf, 0) > 0,
            self.NEG,
            logits_hw,
        )
        logits_flat = kops.reshape(
            masked_logits, (batch_size, self.GRID_H * self.GRID_W)
        )

        # Normalize target distribution P
        P = target_hw * allowed_mask
        P_sum = kops.sum(P, axis=[1, 2], keepdims=True)
        P = kops.where(
            P_sum > self.EPS,
            P / (P_sum + self.EPS),
            allowed_mask / kops.sum(allowed_mask),
        )
        P = P / (kops.sum(P, axis=[1, 2], keepdims=True) + self.EPS)
        P_flat = kops.reshape(P, (batch_size, self.GRID_H * self.GRID_W))

        # --- KL(P||Q) = cross_entropy(P,Q) - entropy(P) ---
        q_probs = kops.softmax(logits_flat, axis=-1)  # Convert logits to probabilities
        ce = -kops.sum(P_flat * kops.log(q_probs + self.EPS), axis=-1)  # [B]
        entropy = -kops.sum(P_flat * tf.math.log(P_flat + self.EPS), axis=-1)  # [B]
        kl = ce - entropy

        # Apply weighting map if provided
        if wmap is not None:
            weights = wmap[None]
            valid_mask = weights > 0
            wsum = kops.sum(weights * valid_mask) + self.EPS
            kl = kl * (kops.sum(weights) / wsum)

        if return_batch or reduction == "none":
            return kl  # [B]
        else:
            loss = kops.mean(kl)
            if scale:
                loss /= 100.0
            return loss

    def _precompute_linear_cost_matrix(self):
        """
        Refactored: Instead of a matrix, we precompute a sorted indexing map
        to allow O(N) Wasserstein via 1D CDF.
        """
        # 1. Build coordinate grid [H*W, 2]
        xs = kops.linspace(0.0, 1.0, self.GRID_W)
        ys = kops.linspace(0.0, 1.0, self.GRID_H)
        xs, ys = kops.meshgrid(xs, ys)
        self.all_grid_coords = kops.stack(
            [kops.reshape(xs, (-1,)), kops.reshape(ys, (-1,))], axis=-1
        )

        # 2. Get coords for allowed cells
        coords_allowed = kops.take(self.all_grid_coords, self.allowed_indices, axis=0)

        # 3. Get 1D Maze Positions (linearized) if possible
        if self.l_function_layer is not None:
            _, lin_coords = self.l_function_layer(coords_allowed)
            lin_coords = kops.reshape(lin_coords, (-1,))

            # 4. CRITICAL: Sort allowed_indices by their 1D position
            # This ensures cumsum() follows the maze path
            lin_coords_np = tf.keras.backend.get_value(lin_coords)
            sort_idx = np.argsort(lin_coords_np)

            # Update allowed_indices to be in maze-order
            original_allowed_np = tf.keras.backend.get_value(self.allowed_indices)
            sorted_allowed_np = original_allowed_np[sort_idx]

            # Store as constant
            self.allowed_indices_sorted = tf.constant(sorted_allowed_np, dtype=tf.int32)
        else:
            # Fallback to standard order
            self.allowed_indices_sorted = self.allowed_indices

        # Optional: Store distances between steps if your grid isn't uniform
        # step_sizes = np.diff(np.sort(lin_coords_np))
        # self.maze_step_sizes = tf.constant(step_sizes, dtype=tf.float32)

    def _precompute_cost_matrix(self):
        """
        Precompute cost matrix using only keras.ops so it works symbolically
        across backends (TF, JAX, Torch).
        """

        # Build coordinate grid [H, W, 2] in symbolic form
        xs = kops.linspace(0.0, 1.0, self.GRID_W)  # [W]
        ys = kops.linspace(0.0, 1.0, self.GRID_H)  # [H]
        xs = kops.broadcast_to(xs[None, :], (self.GRID_H, self.GRID_W))
        ys = kops.broadcast_to(ys[:, None], (self.GRID_H, self.GRID_W))
        coords = kops.stack([xs, ys], axis=-1)  # [H, W, 2]
        coords = kops.reshape(coords, (-1, 2))  # [N, 2] where N=H*W

        coords_allowed = kops.take(
            coords, self.allowed_indices, axis=0
        )  # [N_allowed, 2]
        # Apply your linearization function (kept symbolic)
        _, lin_coords = self.l_function_layer(coords_allowed)  # [N_valid, 1]
        lin_coords = kops.reshape(lin_coords, (-1,))  # [N]

        # Build cost matrix |li - lj|
        li = kops.expand_dims(lin_coords, 0)  # [1, N_valid]
        lj = kops.expand_dims(lin_coords, 1)  # [N_valid, 1]
        diff = li - lj  # [N_valid, N_valid]
        C = kops.abs(diff)

        # --- Make sure cost_matrix and lin_coords are stored as eager tensors
        # Try to evaluate them to numpy (this will succeed when called eagerly during __init__).
        # If that fails (rare), fall back to storing them as non-trainable tf.Variable

        C_np = tf.keras.backend.get_value(C)
        C_rescaled, info = rescale_cost_matrix(C_np)
        lin_coords_np = tf.keras.backend.get_value(lin_coords)
        # store as eager tf.constant so they won't be graph-captured later
        self.cost_matrix = tf.constant(C_rescaled, dtype=tf.float32)
        self.lin_coords = tf.constant(lin_coords_np, dtype=tf.float32)

        # compute kernel once and store as CPU-side constant; don't keep gradient tracking
        eps_tf = tf.cast(self.sinkhorn_eps, tf.float32)
        with tf.device("/CPU:0"):
            kernel_np = tf.keras.backend.get_value(
                kops.exp(-tf.cast(self.cost_matrix, tf.float32) / eps_tf)
            )
            M_np = kernel_np * tf.keras.backend.get_value(self.cost_matrix)
            self.kernel = tf.constant(kernel_np, dtype=tf.float32)
            self.M = tf.constant(M_np, dtype=tf.float32)

        # make sure these won't be part of gradients
        self.cost_matrix = tf.stop_gradient(tf.cast(self.cost_matrix, tf.float32))
        self.kernel = tf.stop_gradient(tf.cast(self.kernel, tf.float32))
        self.M = tf.stop_gradient(tf.cast(self.M, tf.float32))

    def _safe_kl_wasserstein_heatmap_loss(
        self,
        logits_hw,
        target_hw,
        alpha=None,
        sinkhorn_iters=20,
        return_batch=False,
        reduction=None,
    ):
        """
        KL divergence + optional Wasserstein distance (Sinkhorn)
        using precomputed linearized maze cost matrix.
        """
        if alpha is None:
            alpha = 1  # default weight for Wasserstein penalty

        batch_size = kops.shape(logits_hw)[0]
        allowed_mask = kops.cast(self.allowed_mask_tf, "float32")

        # Mask + logits flatten
        masked_logits = kops.where(
            kops.expand_dims(kops.cast(self.forbid_mask_tf, "float32"), 0) > 0,
            self.NEG,
            logits_hw,
        )
        logits_flat = kops.reshape(
            masked_logits, (batch_size, self.GRID_H * self.GRID_W)
        )

        # Normalize target P
        P = target_hw * allowed_mask
        P_sum = kops.sum(P, axis=[1, 2], keepdims=True)
        P = kops.where(
            P_sum > self.EPS,
            P / (P_sum + self.EPS),
            allowed_mask / kops.sum(allowed_mask),
        )
        P = P / (kops.sum(P, axis=[1, 2], keepdims=True) + self.EPS)
        P_flat = kops.reshape(P, (batch_size, self.GRID_H * self.GRID_W))

        # --- KL(P||Q) ---
        # q_probs = kops.softmax(logits_flat/(3*1e-2), axis=-1)
        q_probs = kops.softmax(logits_flat, axis=-1)

        P_allowed = tf.gather(P_flat, self.allowed_indices, axis=1)  # [B, N_valid]
        q_allowed = tf.gather(q_probs, self.allowed_indices, axis=1)  # [B, N_valid]

        ce = -kops.sum(P_allowed * kops.log(q_allowed + self.EPS), axis=-1)
        entropy = -kops.sum(P_allowed * tf.math.log(P_allowed + self.EPS), axis=-1)
        kl = ce - entropy  # [B]

        # --- Wasserstein penalty ---
        if alpha > 0.0:
            # use precomputed CPU-side constants (self.kernel, self.cost_matrix)
            P_allowed = P_allowed / (kops.sum(P_allowed, axis=1, keepdims=True) + 1e-9)
            q_allowed = q_allowed / (kops.sum(q_allowed, axis=1, keepdims=True) + 1e-9)
            temp = kops.matmul(P_allowed, self.cost_matrix)  # [batch, N]
            W = kops.sum(temp * q_allowed, axis=1)  # [batch]
            loss = kl + alpha * W
        else:
            loss = kl

        if return_batch or reduction == "none":
            return loss
        else:
            return kops.mean(loss)

    def _safe_linear_kl_wasserstein_heatmap_loss(
        self,
        logits_hw,
        target_hw,
        alpha=1.0,
        return_batch=False,
        reduction=None,
    ):
        batch_size = kops.shape(logits_hw)[0]

        # 1. Flatten and extract maze-ordered valid cells
        logits_flat = kops.reshape(logits_hw, (batch_size, -1))
        target_flat = kops.reshape(target_hw, (batch_size, -1))

        # Pull out only the allowed cells in their 1D-maze-order
        p_valid = kops.take(target_flat, self.allowed_indices_sorted, axis=1)
        q_logits_valid = kops.take(logits_flat, self.allowed_indices_sorted, axis=1)

        # 2. Probability Normalization
        # Q: Softmax only over the valid maze path
        q_valid = kops.softmax(q_logits_valid, axis=-1)

        # P: Normalize targets to ensure they sum to 1.0 (True distribution)
        p_valid = p_valid / (kops.sum(p_valid, axis=-1, keepdims=True) + self.EPS)

        # 3. KL Divergence (Spatial-agnostic point comparison)
        # Equivalent to 2D KL because mass is only in the allowed cells
        kl = kops.sum(
            p_valid * (kops.log(p_valid + self.EPS) - kops.log(q_valid + self.EPS)),
            axis=-1,
        )

        # 4. 1D Wasserstein (Maze-aware spatial comparison)
        if alpha > 0.0:
            # Earth Mover Distance in 1D = Integral of |CDF_P - CDF_Q|
            cdf_p = kops.cumsum(p_valid, axis=-1)
            cdf_q = kops.cumsum(q_valid, axis=-1)

            # This penalizes mass the further it has to move along the maze path
            w_dist = kops.sum(kops.abs(cdf_p - cdf_q), axis=-1)

            loss = kl + (alpha * w_dist)
        else:
            loss = kl

        if return_batch or reduction == "none":
            return loss
        return kops.mean(loss)

    def compute_output_shape(self, input_shape):
        """
        Return output shape as a tuple (batch_size,).
        Accepts input_shape as:
          - dict: {'logits': shape, 'targets': shape}
          - tuple/list: (shape1, shape2, ...)
          - tf.TensorShape or tuple representing a single tensor shape
        Always returns a 1-tuple (batch_dim,) where batch_dim may be None.
        """

        # Helper to read batch dim from a single shape representation
        def _batch_from_shape(shp):
            # tf.TensorShape -> tuple or list of dims
            try:
                # If it's a tf.TensorShape, convert to tuple
                if hasattr(shp, "as_list"):
                    dims = shp.as_list()
                else:
                    dims = tuple(shp)
                # dims might be [] for scalar tensors; be safe
                if len(dims) == 0:
                    return None
                return dims[0]
            except Exception:
                # Fallback: unknown shape -> None
                return None

        # If dict, pick first value (logits or targets)
        if isinstance(input_shape, dict):
            # Prefer 'logits' key if present
            if "logits" in input_shape:
                first_shape = input_shape["logits"]
            else:
                # fallback to first value
                first_shape = next(iter(input_shape.values()))
            batch = _batch_from_shape(first_shape)
            return (batch,)

        # If list/tuple, inspect first element
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 0:
                return (None,)
            first = input_shape[0]
            # In some Keras usages the list element may itself be a dict
            if isinstance(first, dict):
                if "logits" in first:
                    batch = _batch_from_shape(first["logits"])
                else:
                    batch = _batch_from_shape(next(iter(first.values())))
                return (batch,)
            batch = _batch_from_shape(first)
            return (batch,)

        # Otherwise assume a single shape-like object
        batch = _batch_from_shape(input_shape)
        return (batch,)


def rescale_cost_matrix(
    C_orig,  # numpy array shape [N, N] original cost matrix (CPU)
    allowed_indices=None,  # optional list/array of allowed indices (subset of 0..N-1)
    sample_true_indices=None,  # optional sample of true indices to check "local" costs
    global_target=5.0,
    local_target=0.8,
    local_radius=2,  # neighborhood radius in grid steps (Manhattan or index-based – choose consistent with how C was built)
    max_gamma=8.0,
    gamma_step=1.25,
    max_iters=10,
    verbose=True,
):
    """
    Returns C_rescaled (numpy float32).
    C_orig expected >=0.
    The function will:
      - linear normalize C to [0,1]
      - raise to power gamma (>=1) to compress small values if needed
      - multiply by global_target so max ~ global_target
    It tries to ensure the mean cost within `local_radius` of sample_true_indices is <= local_target.
    """
    C = np.array(C_orig, dtype=np.float64)
    N = C.shape[0]
    assert C.shape[0] == C.shape[1]

    # basic linear normalize to [0,1]
    C_min = C.min()
    C_max = C.max()
    if C_max <= C_min + 1e-12:
        raise ValueError("cost matrix is constant; cannot rescale usefully")

    C_norm = (C - C_min) / (C_max - C_min)  # in [0,1]

    # choose sample indices to inspect local costs
    if sample_true_indices is None:
        # if allowed_indices provided, sample a few of those, otherwise sample some indices
        pool = (
            np.array(allowed_indices) if allowed_indices is not None else np.arange(N)
        )
        rng = np.random.default_rng(0)
        # sample up to 20 indices
        sample_true_indices = rng.choice(pool, size=min(20, pool.size), replace=False)

    # helper: function to compute local mean for an index
    # This assumes C rows correspond to distances from that "true" index to all target indices.
    # We need a way to define neighbors within `local_radius`. If C_orig was built from grid coords,
    # the neighbor selection should be computed from the grid coordinates; here we approximate by
    # selecting the K smallest distances (a cheap proxy). If you can map indices -> (x,y), better: use manhattan.
    def local_mean_from_row(Crow_norm, radius=local_radius, approx_k=None):
        # Quick approach: take the smallest K distances as "neighbors".
        # If grid coords are available, replace this with manhattan neighborhood selection.
        if approx_k is None:
            # approximate number of cells within Manhattan radius r on a grid:
            # K ≈ 1 + 2*r*(r+1)  (diamond shape). For r=2 -> 1 + 2*2*3 = 13
            approx_k = 1 + 2 * radius * (radius + 1)
        smallest = np.partition(Crow_norm, approx_k)[:approx_k]
        return smallest.mean()

    # iterate gamma to compress small values until local_mean <= local_target (after scaling by global_target)
    gamma = 1.0
    it = 0
    while it < max_iters:
        C_try = (C_norm**gamma) * global_target  # in [0, global_target]
        # compute mean local cost across sample indices
        local_means = []
        for idx in sample_true_indices:
            row = C_try[idx, :]  # cost from idx to all
            lm = local_mean_from_row(row, radius=local_radius)
            local_means.append(lm)
        avg_local = float(np.mean(local_means))
        max_val = float(C_try.max())
        if verbose:
            print(
                f"iter {it}: gamma={gamma:.3f}, max={max_val:.4f}, avg_local={avg_local:.4f}"
            )
        # Check targets:
        if (
            avg_local <= local_target
            and abs(max_val - global_target) / global_target < 1e-6
        ):
            break
        # if local mean too large, increase gamma to compress small distances
        if avg_local > local_target and gamma < max_gamma:
            gamma = min(max_gamma, gamma * gamma_step)
            it += 1
            continue
        # if max deviates (shouldn't because we always multiply by global_target), break
        break

    # final matrix
    C_rescaled = (C_norm**gamma) * global_target
    # final safety-clamp (avoid negative / numerical issues)
    C_rescaled = np.clip(C_rescaled, 0.0, None).astype(np.float32)
    return C_rescaled, {
        "gamma": gamma,
        "iters": it,
        "avg_local": avg_local,
        "global_max": float(C_rescaled.max()),
    }


@keras.saving.register_keras_serializable(package="neuroencoders")
class KLHeatmapLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        gaussian_loss_layer_config: Dict,
        wmap=None,
        scale=False,
        **kwargs,
    ):
        self.loss_type = kwargs.pop("loss_type", "safe_kl")
        super().__init__(reduction="none", **kwargs)
        # only serialize the config of the layer, not the layer itself
        self.wmap = wmap
        self.scale = scale

        self.gaussian_loss_layer_config = gaussian_loss_layer_config
        self.gaussian_loss_layer = GaussianHeatmapLosses.from_config(
            self.gaussian_loss_layer_config
        )

    def call(self, y_true, y_pred):
        # y_pred should be logits in shape [B, H, W]
        # y_true should be target heatmap in shape [B, H, W]
        loss_inputs = {"logits": y_pred, "targets": y_true}
        return self.gaussian_loss_layer(
            loss_inputs,
            loss_type=self.loss_type,
            wmap=self.wmap,
            scale=self.scale,
            return_batch=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gaussian_loss_config": self.gaussian_loss_config,
                "wmap": self.wmap,
                "scale": self.scale,
                "loss_type": self.loss_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the layer from saved config
        gaussian_loss_layer = GaussianHeatmapLosses.from_config(
            config["gaussian_loss_config"]
        )
        return cls(
            gaussian_loss_layer=gaussian_loss_layer,
            wmap=config.get("wmap"),
            scale=config.get("scale", False),
            loss_type=config.get("loss_type", "safe_kl"),
        )


def bin_class(example, GRID_W, GRID_H, stride, FORBID):
    """
    Map true (x,y) position to discrete bin class, -1 if forbidden.
    """
    pos = example["pos"]
    x = tf.cast(tf.clip_by_value(pos[0] * GRID_W, 0, GRID_W - 1), tf.int32)
    y = tf.cast(tf.clip_by_value(pos[1] * GRID_H, 0, GRID_H - 1), tf.int32)

    # downscale to coarser grid
    x_coarse = x // stride
    y_coarse = y // stride

    bin_cls = y * GRID_W + x
    # Map forbidden bins to a dummy class that we'll exclude by giving it zero target mass:
    coarse_W = GRID_W // stride
    bin_cls = y_coarse * coarse_W + x_coarse  # ✅ FIXED: was y * GRID_W + x

    # Check if forbidden
    forbidden_here = tf.greater(FORBID[y_coarse * stride, x_coarse * stride], 0)
    return tf.where(forbidden_here, -1, bin_cls)


@keras.saving.register_keras_serializable(package="neuroencoders")
class DenseLossProcessor:
    """Processor for Dense Loss with dynamic weight computation"""

    def __init__(self, maze_points, ts_proj, alpha=1.0, verbose=False, device="/cpu:0"):
        self.maze_points = maze_points
        self.ts_proj = ts_proj
        self.alpha = alpha
        self.linearization_layer = LinearizationLayer(
            maze_points, ts_proj, device=device
        )
        self.fitted_dw = None
        self.weights_layer = None
        self.verbose = verbose
        self.device = device

    def fit_dense_weight_model(self, full_training_positions):
        """
        Step 1: Fit DenseWeight ONCE on full dataset to learn imbalance patterns
        Call this ONCE before training with your complete training dataset
        """
        if self.verbose:
            print("Fitting DenseWeight model on full dataset for imbalance analysis...")

        with get_device_context(self.device):
            # Convert to numpy if needed
            if hasattr(full_training_positions, "numpy"):
                training_pos_np = full_training_positions.numpy()
            else:
                training_pos_np = np.array(full_training_positions)

            # Create temporary model for linearization
            temp_input = tf.keras.Input(shape=training_pos_np.shape[1:])
            _, self.linearized_output = self.linearization_layer(temp_input)
            temp_model = tf.keras.Model(
                inputs=temp_input, outputs=self.linearized_output
            )

            # Get linearized positions for full training dataset
            linearized_training = temp_model.predict(training_pos_np, verbose=0)
            self.linearized_training = linearized_training

            # Fit DenseWeight model on full dataset
            self.fitted_dw = DenseWeight(alpha=self.alpha)
            self.training_weights = self.fitted_dw.fit(linearized_training)

            # Create dynamic weights layer that uses the fitted model
            self.weights_layer = DynamicDenseWeightLayer(
                self.fitted_dw,
                training_data=linearized_training,
                fitted_dw_alpha=self.alpha,
                device=self.device,
            )

            if self.verbose:
                print(
                    "✓ DenseWeight model fitted on {} samples".format(
                        len(training_pos_np)
                    )
                )
                print("✓ Ready for dynamic weight computation during training")

        return self.fitted_dw

    def get_weights_layer(self):
        """Get the dynamic weights layer for use in your model"""
        if self.weights_layer is None:
            raise ValueError("Must call fit_dense_weight_model() first!")
        return self.weights_layer

    def get_config(self):
        """
        Get the configuration of the DenseLossProcessor.
        This is necessary for serialization/deserialization.
        """
        return {
            "maze_points": self.maze_points,
            "ts_proj": self.ts_proj,
            "alpha": self.alpha,
            "fitted_dw": self.fitted_dw if self.fitted_dw else None,
            "verbose": self.verbose,
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the DenseLossProcessor from its config.
        This is necessary for serialization/deserialization.
        """
        maze_points = tf.constant(config.pop("maze_points"))
        ts_proj = tf.constant(config.pop("ts_proj"))
        alpha = config.pop("alpha", 1.0)

        fitted_dw_config = config.pop("fitted_dw", None)
        fitted_dw = fitted_dw_config if fitted_dw_config else None

        processor = cls(maze_points=maze_points, ts_proj=ts_proj, alpha=alpha)
        processor.fitted_dw = fitted_dw
        processor.verbose = config.pop("verbose", False)
        processor.device = config.pop("device", "/cpu:0")
        # return processor
        raise NotImplementedError(
            "Deserialization of DenseLossProcessor not fully implemented yet."
        )


class ContrastiveMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total = logs.get("posLoss")
        raw = logs.get("rawPosLoss")

        if total is not None and raw is not None:
            diff = total - raw
            print(
                f"\n[Epoch {epoch + 1}] Contrastive Contribution: {diff:.4f} "
                f"({(diff / total) * 100:.1f}% of total loss)"
            )


# memory garbage collection class
class MemoryUsageCallbackExtended(tf.keras.callbacks.Callback):
    """Monitor memory usage during training, collect garbage."""

    def __init__(self, log_every_n_epochs=1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.log_every_n_epochs == 0:
            print("**Epoch {}**".format(epoch))
            print(
                f"Memory usage on epoch begin: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.1f}GB"
            )

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Memory usage on epoch end: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.1f}GB"
        )
        if epoch % self.log_every_n_epochs == 0:
            gc.collect()
        # deleted the clear_session() call to avoid issues with custom layers


def _get_loss_function(
    loss_name: str,
    alpha: float,
    delta: float,
    gaussian_loss_layer_config: Optional[Dict] = None,
) -> tf.keras.losses.Loss:
    """Helper function to get loss function by name with reduction='none'"""
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError(reduction="none")
    elif loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError(reduction="none")
    elif loss_name == "huber":
        return tf.keras.losses.Huber(delta=delta, reduction="none")
    elif loss_name == "msle":
        return tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")
    elif loss_name == "logcosh":
        return tf.keras.losses.LogCosh(reduction="none")
    elif loss_name == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy(reduction="none")
    elif loss_name == "categorical_crossentropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    elif loss_name == "mse_plus_msle":

        def combined_loss_mse(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError(reduction="none")(y_true, y_pred)
            msle = tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")(
                y_true, y_pred
            )
            return mse + alpha * msle

        return combined_loss_mse
    elif loss_name == "cyclic_mae":  # for head direction in radians

        def cyclical_mae_rad(y_true, y_pred):
            # Ensure float32 for numerical stability in mixed precision
            y_true = kops.cast(y_true, "float32")
            y_pred = kops.cast(y_pred, "float32")
            return tf.keras.backend.minimum(
                tf.keras.backend.abs(y_pred - y_true),
                tf.keras.backend.minimum(
                    tf.keras.backend.abs(y_pred - y_true + 2 * np.pi),
                    tf.keras.backend.abs(y_pred - y_true - 2 * np.pi),
                ),
            )

        return cyclical_mae_rad
    elif loss_name == "kl_heatmap":
        if gaussian_loss_layer_config is None:
            raise ValueError(
                "gaussian_layer config must be provided for kl_heatmap loss"
            )
        return KLHeatmapLoss(gaussian_loss_layer_config, scale=False)
        # reduction is already none

    else:
        raise ValueError(f"Loss function {loss_name} not recognized")


@keras.saving.register_keras_serializable(package="neuroencoders")
class ContrastiveLossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        temperature=0.1,
        sigma=0.1,
        eps=1e-8,
        name="contrastive_loss_layer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.sigma = sigma
        self.eps = tf.cast(eps, "float32")

    def call(self, inputs):
        """
        Args:
            inputs: A dictionary with keys:
                - 'z': (N, D) latent vectors (from LSTM or Transformer output)
                - 'pos': (N, dimOutput) ground-truth positions. Will be sliced to (N, 2).
        Returns:
            Scalar contrastive loss value.
        """
        if isinstance(inputs, dict):
            z = inputs["z"]
            pos = inputs["pos"]
        else:
            pos, z = inputs  # assume tuple/list input (pos, z) ie y_true, y_pred order
        return self.distance_weighted_nt_xent(z, pos)

    def distance_weighted_nt_xent(self, z, pos):
        """
        Contrastive loss based on spatial distance in XY space.

        Args:
            z: (N, D) latent vectors (from LSTM or Transformer output)
            pos: (N, 1) ground-truth linearized positions.
            sigma: Spatial kernel width.
                   NOTE: If positions are normalized [0,1], sigma should be small (e.g., 0.1).
                   If positions are normalized between 0-1 (as they should), a value for sigma like 0.05-0.2 is recommended.
            temperature: NT-Xent scaling parameter.
        """

        # Ensure we operate on float32 for numerical stability (mixed precision)
        dtype = "float32"
        z = kops.cast(z, dtype)
        pos = kops.cast(pos, dtype)

        # 1. Get dynamic batch size (Crucial Fix for "None" error)
        N = tf.shape(z)[0]

        # 2. Normalize latents to unit length
        # This ensures dot product = cosine similarity
        z = tf.math.l2_normalize(z, axis=1)

        # 3. Compute pairwise Cosine Similarity Logits
        # Shape: (N, N)
        logits = tf.matmul(z, z, transpose_b=True)
        logits = logits / tf.cast(self.temperature, dtype)

        # 4. Compute Pairwise Spatial Distances
        # pos can be (N, 1) or (N, D) or even just (N,)
        pos = tf.convert_to_tensor(pos)
        if len(pos.shape) == 1:
            pos = tf.reshape(pos, [-1, 1])

        if pos.shape[-1] > 1:
            # Compute pairwise Euclidean distance for (N, D) positions
            r = tf.reduce_sum(tf.square(pos), axis=1, keepdims=True)
            d2 = r - 2 * tf.matmul(pos, pos, transpose_b=True) + tf.transpose(r)
            d = tf.sqrt(tf.maximum(d2, self.eps))
        else:
            # Already (N, 1) or reshaped to (N, 1)
            d = tf.abs(pos - tf.transpose(pos))  # (N, N)

        # 5. Compute Distance Weights (Soft Positives)
        # w_ij = exp( - d_ij^2 / (2*sigma^2) )
        sigma_sq = tf.cast(self.sigma**2, dtype)
        w = tf.exp(-0.5 * tf.square(d) / sigma_sq)

        # 6. Mask Self-Similarity (Diagonal)
        # We don't want the network to learn "I am similar to myself" (trivial)
        mask_diag = tf.eye(N, dtype=dtype)
        w = w * (1.0 - mask_diag)  # Zero out diagonal weights

        # Normalize weights per row so they sum to 1 (conceptually similar to soft labels)
        w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + self.eps
        w_norm = w / w_sum

        # 7. Compute Softmax Log-Probabilities
        # Standard NT-Xent/InfoNCE denominator includes all negatives AND positives
        # We mask the self-similarity (diagonal) from the logits with a large negative number
        # so it doesn't affect the softmax denominator.
        logits_masked = logits + (-1e9) * mask_diag
        log_prob = tf.nn.log_softmax(logits_masked, axis=1)

        # 8. Compute Loss
        # Cross entropy with soft targets w_norm
        loss_per_anchor = -tf.reduce_sum(w_norm * log_prob, axis=1)

        return tf.reduce_mean(loss_per_anchor)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "temperature": self.temperature,
                "sigma": self.sigma,
                "eps": self.eps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            temperature=config.get("temperature", 0.1),
            sigma=config.get("sigma", 0.1),
            eps=config.get("eps", 1e-8),
        )


@keras.saving.register_keras_serializable(package="neuroencoders")
class MultiColumnLossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        column_losses: Optional[Dict[str, str]] = None,
        column_weights: Optional[Dict[str, float]] = None,
        alpha: float = 1.0,
        delta: float = 1.0,
        name: str = "multi_output_loss_layer",
        gaussian_params: Optional[Dict] = None,
        merge_columns: Optional[List[List[int]]] = None,
        merge_losses: Optional[List[str]] = None,
        merge_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Args:
            column_losses (dict): Dictionary mapping column specifications to loss names.
                Example: {"0": "mse", "1,2": "huber"} means column 0 uses MSE and columns 1 and 2 use Huber loss.
            column_weights (dict): Dictionary mapping column specifications to weights.
                Example: {"0": 1.0, "1,2": 0.5} means column 0 has weight 1.0 and columns 1 and 2 have weight 0.5.
            alpha (float): Hyperparameter for losses like Huber or MSLE.
            name (str): Name of the layer.
            gaussian_layer: Optional Gaussian layer for certain loss functions.
            merge_columns (List[List[int]]): List of column groups to process together.
                Example: [[0, 1], [2, 3]] means columns 0,1 are processed together and columns 2,3 are processed together.
            merge_losses (List[str]): Loss function names for each merged column group.
                Must have same length as merge_columns if provided.
            merge_weights (List[float]): Weights for each merged column group.
                Must have same length as merge_columns if provided.
            **kwargs: Additional keyword arguments for the Layer constructor.
        """
        super().__init__(name=name, **kwargs)
        self.column_losses = column_losses or {}
        self.column_weights = column_weights or {}
        self.alpha = alpha
        self.delta = delta
        self.merge_columns = merge_columns or []
        self.merge_losses = merge_losses or []
        self.merge_weights = merge_weights or []
        if gaussian_params:
            if "name" not in gaussian_params:
                gaussian_params["name"] = "gaussian_loss_layer"
        self.gaussian_loss_layer_config = gaussian_params

        # Validate merge parameters
        if self.merge_columns:
            if self.merge_losses and len(self.merge_losses) != len(self.merge_columns):
                raise ValueError("merge_losses must have same length as merge_columns")
            if self.merge_weights and len(self.merge_weights) != len(
                self.merge_columns
            ):
                raise ValueError("merge_weights must have same length as merge_columns")

            # Fill in defaults for merge_losses and merge_weights if not provided
            if not self.merge_losses:
                self.merge_losses = ["mse"] * len(self.merge_columns)
            if not self.merge_weights:
                self.merge_weights = [1.0] * len(self.merge_columns)

        # Create loss functions for merged columns
        self.merged_loss_functions = []
        for loss_name in self.merge_losses:
            self.merged_loss_functions.append(
                _get_loss_function(
                    loss_name, self.alpha, self.delta, self.gaussian_loss_layer_config
                )
            )

        # Create individual column loss functions (for backwards compatibility)
        self.individual_losses = {}
        self.individual_weights = {}

        for col_spec, loss_name in self.column_losses.items():
            self.individual_losses[col_spec] = _get_loss_function(
                loss_name, self.alpha, self.delta, self.gaussian_loss_layer_config
            )
            self.individual_weights[col_spec] = self.column_weights.get(col_spec, 1.0)

    def _add_losses_to_model(self, model):
        """
        Utility to add all individual losses to model for tracking.
        """
        for col_spec, loss_fn in self.individual_losses.items():
            loss_name = "loss_col_{}".format(col_spec)
            model.add_loss(
                lambda y_true, y_pred: loss_fn(y_true, y_pred), name=loss_name
            )

    def _parse_column_spec(self, col_spec: str) -> List[int]:
        """Parse column specification like '0' or '1,2' into list of integers."""
        if "," in col_spec:
            return [int(x.strip()) for x in col_spec.split(",")]
        else:
            return [int(col_spec)]

    def call(self, y_true, y_pred):
        """
        Compute the combined loss.
        Args:
            y_true: True values with shape (batch_size, num_columns)
            y_pred: Predicted values with shape (batch_size, num_columns)
        Returns:
            Combined loss tensor with shape (batch_size,)
        """
        # Ensure float32 for numerical stability in mixed precision
        y_true = kops.cast(y_true, "float32")
        y_pred = kops.cast(y_pred, "float32")
        total_loss = tf.zeros(tf.shape(y_true)[0], dtype="float32")

        # Track which columns have been processed to avoid double-counting
        processed_columns = set()

        # Process merged columns first
        for i, (col_group, loss_fn, weight) in enumerate(
            zip(self.merge_columns, self.merged_loss_functions, self.merge_weights)
        ):
            # Extract the merged columns
            y_true_merged = tf.gather(
                y_true, col_group, axis=1
            )  # Shape: (batch_size, len(col_group))
            y_pred_merged = tf.gather(
                y_pred, col_group, axis=1
            )  # Shape: (batch_size, len(col_group))

            # Compute loss for merged columns
            merged_loss = loss_fn(y_true_merged, y_pred_merged)
            total_loss += weight * merged_loss

            # Mark these columns as processed
            processed_columns.update(col_group)

        # Process individual columns that weren't part of any merged group
        for col_spec, loss_fn in self.individual_losses.items():
            columns = self._parse_column_spec(col_spec)

            # Skip if any of these columns were already processed in merged groups
            if any(col in processed_columns for col in columns):
                continue

            if len(columns) == 1:
                # Single column
                col_idx = columns[0]
                y_true_col = y_true[:, col_idx]
                y_pred_col = y_pred[:, col_idx]
                loss_val = loss_fn(y_true_col, y_pred_col)
            else:
                # Multiple columns (legacy support for comma-separated specs)
                y_true_cols = tf.gather(y_true, columns, axis=1)
                y_pred_cols = tf.gather(y_pred, columns, axis=1)
                loss_val = loss_fn(y_true_cols, y_pred_cols)

            weight = self.individual_weights[col_spec]
            total_loss += weight * loss_val

            # Mark these columns as processed
            processed_columns.update(columns)

        return total_loss

    def get_config(self):
        """Return the config of the layer for serialization."""
        config = super().get_config()
        config.update(
            {
                "column_losses": self.column_losses,
                "column_weights": self.column_weights,
                "alpha": self.alpha,
                "delta": self.delta,
                "merge_columns": self.merge_columns,
                "merge_losses": self.merge_losses,
                "merge_weights": self.merge_weights,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        layer_config = {
            "column_losses": config.get("column_losses", {}),
            "column_weights": config.get("column_weights", {}),
            "alpha": config.get("alpha", 1.0),
            "delta": config.get("delta", 1.0),
            "merge_columns": config.get("merge_columns", []),
            "merge_losses": config.get("merge_losses", []),
            "merge_weights": config.get("merge_weights", []),
        }
        return cls(**layer_config)

    def build(self, input_shape):
        """
        Build the layer. This method is called once the input shape is known.

        Args:
            input_shape: Can be a single shape tuple or a list of two shape tuples
                        [y_true_shape, y_pred_shape] if called with two inputs,
                        or a single shape if called with one input (assuming both have same shape)
        """
        # Handle both single input shape and list of input shapes
        if isinstance(input_shape, list) and len(input_shape) == 2:
            # Two inputs: y_true and y_pred
            y_true_shape, y_pred_shape = input_shape

            # Validate that both inputs have the same shape
            if y_true_shape != y_pred_shape:
                raise ValueError(
                    f"y_true and y_pred must have the same shape. "
                    f"Got y_true: {y_true_shape}, y_pred: {y_pred_shape}"
                )

            self.input_shape_value = y_true_shape
        else:
            # Single input shape (both inputs assumed to have same shape)
            self.input_shape_value = input_shape

        # Validate input shape
        if len(self.input_shape_value) < 2:
            raise ValueError(
                f"Input must be at least 2D (batch_size, num_columns). "
                f"Got shape: {self.input_shape_value}"
            )

        self.num_columns = self.input_shape_value[-1]

        # Validate column indices in merge_columns
        for i, col_group in enumerate(self.merge_columns):
            for col_idx in col_group:
                if col_idx >= self.num_columns or col_idx < 0:
                    raise ValueError(
                        f"Column index {col_idx} in merge_columns[{i}] is out of range. "
                        f"Input has {self.num_columns} columns (indices 0-{self.num_columns - 1})"
                    )

        # Validate column indices in individual losses
        for col_spec in self.column_losses.keys():
            columns = self._parse_column_spec(col_spec)
            for col_idx in columns:
                if col_idx >= self.num_columns or col_idx < 0:
                    raise ValueError(
                        f"Column index {col_idx} in column_losses['{col_spec}'] is out of range. "
                        f"Input has {self.num_columns} columns (indices 0-{self.num_columns - 1})"
                    )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Can be a single shape tuple or a list of two shape tuples
                        [y_true_shape, y_pred_shape] if called with two inputs,
                        or a single shape if called with one input

        Returns:
            tuple: Output shape (batch_size,) - returns a scalar loss per sample
        """
        # Handle both single input shape and list of input shapes
        if isinstance(input_shape, list) and len(input_shape) == 2:
            # Two inputs: y_true and y_pred
            batch_size = input_shape[0][0]  # Get batch size from first input
        else:
            # Single input shape
            batch_size = input_shape[0]  # Get batch size

        # Return shape: (batch_size,) - one scalar loss per sample in the batch
        return (batch_size,)


def get_output_shape_for(self, input_shape):
    """
    Alternative method name used by some Keras versions.
    """
    return self.compute_output_shape(input_shape)


@keras.saving.register_keras_serializable(package="neuroencoders")
class PositionError2D(tf.keras.metrics.Metric, SpatialConstraintsMixin):
    """
    Keras Metric to calculate 2D position error (Euclidean distance)
    by decoding heatmap logits.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (45, 45),
        maze_params: Optional[Dict] = None,
        name="pos_error_2d",
        **kwargs,
    ):
        # Handle the case where a config dictionary is passed directly (legacy support)
        if isinstance(grid_size, dict) and "grid_size" in grid_size:
            config = grid_size
            grid_size = config.get("grid_size", (45, 45))
            maze_params = config.get("maze_params", None)
        elif isinstance(grid_size, dict) and "gaussian_heatmap_layer" in grid_size:
            # Another variant of legacy support
            config = grid_size.get("gaussian_heatmap_layer", {})
            grid_size = config.get("grid_size", (45, 45))
            maze_params = config.get("maze_params", None)

        tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )

        # We store them for get_config reconstruction
        self._grid_size = grid_size
        self._maze_params = maze_params

        self.total_dist = self.add_weight(name="total_dist", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Decode heatmap logits using unified mixin method
        rank = y_pred.shape.rank
        if rank is None:
            rank = tf.rank(y_pred)

        if rank == 2:
            logits_hw = tf.reshape(y_pred, [-1, self.GRID_H, self.GRID_W])
        else:
            logits_hw = y_pred

        # Decode using the mixin's unified method
        # Using expectation for smoothness in training logs
        xy, _, _, _ = self.decode_and_uncertainty_tf(logits_hw, mode="expectation")

        # Ensure xy is float32 for metric calculation
        xy = tf.cast(xy, tf.float32)
        y_true_coords = tf.cast(y_true[:, :2], tf.float32)
        dist = tf.sqrt(tf.reduce_sum(tf.square(xy - y_true_coords), axis=-1))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            dist = dist * sample_weight
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

        self.total_dist.assign_add(tf.reduce_sum(dist))

    def result(self):
        return self.total_dist / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total_dist.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = tf.keras.metrics.Metric.get_config(self)
        config.update(
            {
                "grid_size": list(self._grid_size),
                "maze_params": self._maze_params,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="neuroencoders")
class AngularErrorMetric(tf.keras.metrics.Metric):
    """
    Keras Metric for angular error.
    Handles both radians (1 col) and unit vectors (2 cols).
    """

    def __init__(self, name="angular_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name="total_error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_shape = tf.shape(y_true)
        error = tf.cond(
            tf.equal(y_true_shape[1], 1),
            # True branch: Radians
            lambda: tf.abs((y_pred - y_true + np.pi) % (2 * np.pi) - np.pi),
            # False branch: Unit Vectors
            lambda: tf.acos(
                tf.clip_by_value(
                    tf.reduce_sum(y_pred * y_true, axis=-1)
                    / (tf.norm(y_pred, axis=-1) * tf.norm(y_true, axis=-1) + 1e-8),
                    -1.0,
                    1.0,
                )
            ),
        )
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            error = error * sample_weight
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

        self.total_error.assign_add(tf.reduce_sum(error))

    def result(self):
        return self.total_error / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable(package="neuroencoders")
class CyclicMAE(tf.keras.losses.Loss):
    def __init__(self, high=2 * np.pi, name="cyclic_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.high = high

    def call(self, y_true, y_pred):
        # Calculate angular difference: (y_pred - y_true) mod high
        delta = tf.abs(y_true - y_pred)
        # handle circularity
        delta = tf.where(delta > self.high / 2, self.high - delta, delta)
        return tf.reduce_mean(delta, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"high": self.high})
        return config


@keras.saving.register_keras_serializable(package="neuroencoders")
class GaussianHeatmapLoss(tf.keras.losses.Loss, SpatialConstraintsMixin):
    """
    Gaussian Heatmap Loss that converts 2D coordinates to target heatmaps on-the-fly
    and computes KL divergence directly.
    """

    def __init__(
        self,
        gaussian_params: Dict,
        l_function_params: Optional[Dict] = None,
        loss_type: str = "safe_kl",
        name: str = "gaussian_heatmap_loss",
        **kwargs,
    ):
        # Initialize Keras Loss
        tf.keras.losses.Loss.__init__(self, name=name, **kwargs)

        # Initialize Spatial Mixin
        grid_size = gaussian_params.get("grid_size", (45, 45))
        maze_params = gaussian_params.get("maze_params", None)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )

        # Core parameters
        self.gaussian_params = gaussian_params
        self.l_function_params = l_function_params
        self.loss_type = loss_type
        self.sigma = float(gaussian_params.get("sigma", 0.03))
        self.eps = float(gaussian_params.get("eps", 1e-8))
        self.neg = float(gaussian_params.get("neg", -100))

        # Precompute occupancy-based weights if training_positions available
        training_positions = gaussian_params.get("training_positions")
        if training_positions is not None:
            occ = self.occupancy_map(training_positions)
            self.WMAP = tf.constant(
                self.weight_map_from_occ(occ, alpha=0.5), dtype=tf.float32
            )
        else:
            self.WMAP = tf.ones((self.GRID_H, self.GRID_W), dtype=tf.float32)

    def call(self, y_true, y_pred):
        # y_true: [B, 2+] coordinates, y_pred: [B, H, W] or [B, H*W] logits
        # Ensure float32 for loss calculation (mixed precision safety)
        y_true = kops.cast(y_true, "float32")
        y_pred = kops.cast(y_pred, "float32")

        # 1. Generate target heatmap from coordinates using Mixin method
        targets_hw = self.gaussian_heatmap_targets_tf(y_true[:, :2], sigma=self.sigma)

        # 2. Reshape predictions if flattened
        rank = y_pred.shape.rank
        if rank == 2:
            y_pred = kops.reshape(y_pred, (-1, self.GRID_H, self.GRID_W))

        # 3. Compute loss based on type
        if self.loss_type == "safe_kl":
            return self._compute_safe_kl(y_pred, targets_hw)
        else:
            # Fallback to direct MSE/MAE if specified
            return tf.reduce_mean(tf.square(y_pred - targets_hw))

    def _compute_safe_kl(self, logits_hw, target_hw):
        batch_size = kops.shape(logits_hw)[0]

        # Mask forbidden bins in logits to avoid mass leak
        forbid_mask = kops.expand_dims(self.forbid_mask_tf, 0)
        masked_logits = kops.where(forbid_mask > 0, self.common_neg, logits_hw)

        # softmax probability over the grid
        probs_flat = kops.softmax(
            kops.reshape(masked_logits, (batch_size, self.GRID_H * self.GRID_W)),
            axis=-1,
        )
        probs = kops.reshape(probs_flat, (batch_size, self.GRID_H, self.GRID_W))

        # KL calculation with stability thresholding
        threshold = self.common_eps * 10
        safe_probs = kops.maximum(probs, self.common_eps)

        # Weighted KL
        weights = kops.expand_dims(self.WMAP, 0)
        kl = kops.where(
            target_hw > threshold,
            target_hw * kops.log(target_hw / safe_probs),
            0.0,
        )

        # Normalized weighted sum
        weighted_kl = kl * weights
        loss_per_sample = kops.sum(weighted_kl, axis=[1, 2]) / (
            kops.sum(weights) + self.common_eps
        )
        return kops.mean(loss_per_sample)

    def occupancy_map(self, positions):
        occ = np.zeros((self.GRID_H, self.GRID_W), np.float32)
        xs = np.clip((positions[:, 0] * self.GRID_W).astype(int), 0, self.GRID_W - 1)
        ys = np.clip((positions[:, 1] * self.GRID_H).astype(int), 0, self.GRID_H - 1)
        for k in range(len(positions)):
            occ[ys[k], xs[k]] += 1
        return occ

    def weight_map_from_occ(self, occ, alpha=0.5):
        # simplified version of the weight map logic
        # weight = 1 / (occ + 1)^alpha
        weights = 1.0 / (occ + 1.0) ** alpha
        # apply spatial mask
        allowed_mask = self.get_allowed_mask(use_tensorflow=False)
        return weights * allowed_mask

    def get_config(self):
        config = tf.keras.losses.Loss.get_config(self)
        # Handle numpy serialization for training_positions
        g_params = self.gaussian_params.copy()
        if hasattr(g_params.get("training_positions"), "tolist"):
            g_params["training_positions"] = g_params["training_positions"].tolist()

        # handle l function layer params serialization
        l_function_layer_params_serializable = self.l_function_params.copy()
        for k, v in l_function_layer_params_serializable.items():
            try:
                v = v.numpy().tolist()
            except AttributeError:
                # Handle case where these aren't TensorFlow tensors
                v = v.tolist() if hasattr(v, "tolist") else v
            l_function_layer_params_serializable[k] = v

        config.update(
            {
                "gaussian_params": g_params,
                "l_function_params": l_function_layer_params_serializable,
                "loss_type": self.loss_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        g_params = config.get("gaussian_params", {})
        # Handle deserialization of training_positions if it was a list
        if "training_positions" in g_params and isinstance(
            g_params["training_positions"], list
        ):
            g_params["training_positions"] = np.array(g_params["training_positions"])
        return cls(
            gaussian_params=g_params,
            l_function_params=config.get("l_function_params"),
            loss_type=config.get("loss_type", "safe_kl"),
        )


@keras.saving.register_keras_serializable(package="neuroencoders")
class ContrastiveRegressionLoss(tf.keras.losses.Loss):
    """
    Contrastive Loss based on NT-Xent with spatial weighting,
    computing the NT-Xent loss directly without internal layers.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sigma: float = 0.1,
        l_function_params: Optional[Dict] = None,
        name: str = "contrastive_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.l_function_params = l_function_params
        self.eps = 1e-8

        self.l_function = None
        if l_function_params is not None:
            self.l_function = LinearizationLayer(**l_function_params)

    def call(self, y_true, y_pred):
        # y_true: (batch, 2+) coords, y_pred: (batch, D) latent
        # Ensure float32 for numerical stability (mixed precision)
        z = kops.cast(y_pred, "float32")
        y_true = kops.cast(y_true, "float32")

        # 1. Get positions for weighting (linearized or 2D)
        if self.l_function is not None:
            _, linearized_pos = self.l_function(y_true[:, :2])
            pos = linearized_pos
        else:
            pos = y_true[:, :2] if y_true.shape[-1] >= 2 else y_true

        # 2. Normalize latents to unit length for cosine similarity
        N = tf.shape(z)[0]
        z = tf.math.l2_normalize(z, axis=1)

        # 3. Compute pairwise Cosine Similarity Logits
        logits = tf.matmul(z, z, transpose_b=True) / self.temperature

        # 4. Compute Pairwise Spatial Distances
        if len(pos.shape) == 1:
            pos = tf.reshape(pos, [-1, 1])

        if pos.shape[-1] > 1:
            r = tf.reduce_sum(tf.square(pos), axis=1, keepdims=True)
            d2 = r - 2 * tf.matmul(pos, pos, transpose_b=True) + tf.transpose(r)
            d = tf.sqrt(tf.maximum(d2, self.eps))
        else:
            d = tf.abs(pos - tf.transpose(pos))

        # 5. Distance weighting (soft positives)
        w = kops.cast(tf.exp(-0.5 * tf.square(d) / (self.sigma**2)), "float32")
        mask_diag = tf.eye(N, dtype="float32")
        w = w * (1.0 - mask_diag)  # Mask self-similarity

        w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + self.eps
        w_norm = w / w_sum

        # 6. Compute Softmax Log-Probabilities masking diagonal
        logits_masked = logits + (-1e9) * mask_diag
        log_prob = tf.nn.log_softmax(logits_masked, axis=1)

        # 7. Cross entropy with soft targets
        loss_per_anchor = -tf.reduce_sum(w_norm * log_prob, axis=1)
        return tf.reduce_mean(loss_per_anchor)

    def get_config(self):
        config = super().get_config()

        if self.l_function_params is not None:
            # handle l function layer params serialization
            l_function_layer_params_serializable = self.l_function_params.copy()
            for k, v in l_function_layer_params_serializable.items():
                try:
                    v = v.numpy().tolist()
                except AttributeError:
                    # Handle case where these aren't TensorFlow tensors
                    v = v.tolist() if hasattr(v, "tolist") else v
                l_function_layer_params_serializable[k] = v
        else:
            l_function_layer_params_serializable = None

        config.update(
            {
                "temperature": self.temperature,
                "sigma": self.sigma,
                "l_function_params": l_function_layer_params_serializable,
            }
        )
        return config


# Register custom layers and losses for Keras serialization
keras_utils.get_custom_objects()["DenseLossProcessor"] = DenseLossProcessor
keras_utils.get_custom_objects()["SpikeNet1D"] = SpikeNet1D
keras_utils.get_custom_objects()["spikeNet"] = spikeNet
keras_utils.get_custom_objects()["DynamicDenseWeightLayer"] = DynamicDenseWeightLayer
keras_utils.get_custom_objects()["GaussianHeatmapLayer"] = GaussianHeatmapLayer
keras_utils.get_custom_objects()["GaussianHeatmapLosses"] = GaussianHeatmapLosses
keras_utils.get_custom_objects()["KLHeatmapLoss"] = KLHeatmapLoss
keras_utils.get_custom_objects()["DenseWeight"] = DenseWeight
keras_utils.get_custom_objects()["LinearizationLayer"] = LinearizationLayer
keras_utils.get_custom_objects()["MemoryUsageCallbackExtended"] = (
    MemoryUsageCallbackExtended
)
keras_utils.get_custom_objects()["MultiColumnLossLayer"] = MultiColumnLossLayer
keras_utils.get_custom_objects()["ContrastiveLossLayer"] = ContrastiveLossLayer
keras_utils.get_custom_objects()["GroupAttentionFusion"] = GroupAttentionFusion
keras_utils.get_custom_objects()["ContrastiveMonitor"] = ContrastiveMonitor
keras_utils.get_custom_objects()["CyclicMAE"] = CyclicMAE
keras_utils.get_custom_objects()["PositionError2D"] = PositionError2D
keras_utils.get_custom_objects()["GaussianHeatmapLoss"] = GaussianHeatmapLoss
keras_utils.get_custom_objects()["ContrastiveRegressionLoss"] = (
    ContrastiveRegressionLoss
)
