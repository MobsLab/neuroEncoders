# Load libs
import contextlib
import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import keras
import keras.utils as keras_utils
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


@tf.keras.utils.register_keras_serializable(package="neuroencoders")
class MaskedBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        # The last dimension is the number of channels/filters
        dim = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma", shape=(dim,), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(dim,), initializer="zeros", trainable=True
        )

        # Moving statistics for inference
        self.moving_mean = self.add_weight(
            name="moving_mean", shape=(dim,), initializer="zeros", trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance", shape=(dim,), initializer="ones", trainable=False
        )

    def call(self, x, mask=None, training=False):
        if mask is None:
            # During training without mask: compute batch statistics normally
            # During inference without mask: use moving statistics
            if training:
                # Compute batch statistics (standard BN behavior)
                # x shape: (Batch, Time, Channels) or (Batch, Channels, Time, 1)
                reduction_axes = list(range(len(x.shape) - 1))

                mean = tf.reduce_mean(x, axis=reduction_axes)
                variance = tf.reduce_mean(tf.square(x - mean), axis=reduction_axes)

                # Update moving statistics
                self.moving_mean.assign(
                    self.moving_mean * self.momentum + mean * (1 - self.momentum)
                )
                self.moving_variance.assign(
                    self.moving_variance * self.momentum
                    + variance * (1 - self.momentum)
                )
            else:
                # Inference without mask: use moving statistics
                mean = self.moving_mean
                variance = self.moving_variance

            # Normalize and apply Gamma/Beta
            return tf.nn.batch_normalization(
                x, mean, variance, self.beta, self.gamma, self.epsilon
            )

        # Masked case: use masked statistics
        mask = tf.cast(mask, x.dtype)
        # Ensure mask is broadcastable to x
        while mask.shape.ndims is not None and mask.shape.ndims < x.shape.ndims:
            mask = tf.expand_dims(mask, axis=-1)

        # x shape: (Batch, Time, Channels) or (Batch, Channels, Time, 1)
        # We need to reduce over all axes EXCEPT the last one (Channels)
        reduction_axes = list(range(len(x.shape) - 1))

        if training:
            # 1. Calculate Mean (sum of values / sum of mask)
            # We multiply by mask to ensure padding is 0, then sum
            count = tf.reduce_sum(mask, axis=reduction_axes) + self.epsilon
            mean = tf.reduce_sum(x * mask, axis=reduction_axes) / count

            # 2. Calculate Variance: sum((x - mean)^2 * mask) / count
            squared_diff = tf.square(x - mean) * mask
            variance = tf.reduce_sum(squared_diff, axis=reduction_axes) / count

            # 3. Update moving statistics
            self.moving_mean.assign(
                self.moving_mean * self.momentum + mean * (1 - self.momentum)
            )
            self.moving_variance.assign(
                self.moving_variance * self.momentum + variance * (1 - self.momentum)
            )
        else:
            mean = self.moving_mean
            variance = self.moving_variance

        # 4. Normalize and apply Gamma/Beta
        x_norm = tf.nn.batch_normalization(
            x, mean, variance, self.beta, self.gamma, self.epsilon
        )

        # 5. RE-APPLY MASK: This is critical to ensure padding remains exactly 0.0
        return x_norm * mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_structure": self.target_structure,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "sigma": self.sigma,
                "l_function_params": self.l_function_params,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        epsilon = config.pop("epsilon", 1e-3)
        momentum = config.pop("momentum", 0.99)
        target_structure = config.pop("target_structure", None)
        sigma = config.pop("sigma", None)
        l_function_params = config.pop("l_function_params", None)
        return cls(
            target_structure=target_structure,
            sigma=sigma,
            l_function_params=l_function_params,
            epsilon=epsilon,
            momentum=momentum,
            **config,
        )


@tf.keras.utils.register_keras_serializable(package="neuroencoders")
class ChannelwiseFixedNormalization(tf.keras.layers.Layer):
    """Deterministic channel-wise normalization with externally set statistics.

    This layer mirrors the 3-weight contract often used with
    ``tf.keras.layers.Normalization``: ``[mean, variance, count]``.
    """

    def __init__(self, axis=1, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        rank = len(input_shape)
        axis = self.axis if self.axis >= 0 else rank + self.axis
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"Invalid normalization axis {self.axis} for shape {input_shape}"
            )

        channels = input_shape[axis]
        if channels is None:
            raise ValueError(
                "Channel dimension must be known to build ChannelwiseFixedNormalization"
            )

        self.mean = self.add_weight(
            name="mean",
            shape=(channels,),
            initializer="zeros",
            trainable=False,
        )
        self.variance = self.add_weight(
            name="variance",
            shape=(channels,),
            initializer="ones",
            trainable=False,
        )
        self.count = self.add_weight(
            name="count",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        rank = len(x.shape)
        axis = self.axis if self.axis >= 0 else rank + self.axis
        broadcast_shape = [1] * rank
        broadcast_shape[axis] = tf.shape(self.mean)[0]

        mean = tf.reshape(self.mean, broadcast_shape)
        variance = tf.reshape(self.variance, broadcast_shape)
        return (x - mean) / tf.sqrt(variance + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "epsilon": self.epsilon})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="neuroencoders")
class AddNullSpike(tf.keras.layers.Layer):
    """
    The "Null Spike" Trick:
    Add a row of zeros at index 0 for each example in the batch.
    When index_inputs[g] is 0 (padding), tf.gather will pick up these zeros.
    """

    def __init__(self, n_features, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.supports_masking = True

    def call(self, e, mask=None, training=False):
        # e shape : (batch, max_spikes_per_group, nFeatures)
        dtype = self.compute_dtype
        batch_size = tf.shape(e)[0]
        n_features = self.n_features
        null_spike = kops.zeros((batch_size, 1, n_features), dtype=dtype)
        e = kops.cast(e, dtype)  # Ensure e is the same dtype as null_spike
        return kops.concatenate([null_spike, e], axis=1)

    def compute_mask(self, inputs, mask=None):
        # When propagating a per-spike mask through AddNullSpike, prepend a
        # valid mask entry for the synthetic null spike at index 0.
        if mask is None:
            return None
        null_mask = tf.ones((tf.shape(mask)[0], 1), dtype=mask.dtype)
        return tf.concat([null_mask, mask], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        # make sure inputs is a tuple of (mask, features)
        mask, features = inputs
        mask_expanded = kops.expand_dims(mask, axis=-1)
        return kops.where(mask_expanded, features, kops.zeros_like(features))

    def compute_mask(self, inputs, mask=None):
        # this layer already has a mask in the inputs, so we can just pass it through
        return inputs[0]  # the mask is the first element of the inputs

    def compute_output_shape(self, input_shapes):
        # input_shapes = [(batch, seqLen), (batch, seqLen, feat)]
        return input_shapes[1]


########### CONVOLUTIONAL NETWORK CLASS #####################
@keras.saving.register_keras_serializable(package="neuroencoders")
class SpikeNet2D(tf.keras.layers.Layer):
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
        self.batch_normalization = kwargs.pop("batch_normalization", False)
        self.no_cnn = no_cnn
        super().__init__(name=name, **kwargs)
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        self.device = device
        self.number = number
        self.supports_masking = True
        with get_device_context(self.device):
            # Input normalization setup (axis=1 for channels)
            self.input_normalization = ChannelwiseFixedNormalization(
                axis=1, name="input_norm"
            )

            self.convLayer1 = tf.keras.layers.Conv2D(8, [2, 3], padding="same")
            self.convLayer2 = tf.keras.layers.Conv2D(16, [2, 3], padding="same")
            self.convLayer3 = tf.keras.layers.Conv2D(32, [2, 3], padding="same")

            self.pool = tf.keras.layers.MaxPool2D([1, 2], padding="same")

            if self.batch_normalization:
                self.bn1 = MaskedBatchNormalization()
                self.bn2 = MaskedBatchNormalization()
                self.bn3 = MaskedBatchNormalization()

            self.dropoutLayer = tf.keras.layers.Dropout(0.2)
            self.denseLayer1 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer2 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer3 = tf.keras.layers.Dense(
                self.nFeatures, activation="relu", name="outputCNN{}".format(number)
            )
            self.flatten = tf.keras.layers.Flatten()

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
        batch_normalization = config.get("batch_normalization", False)
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

    def call(self, input, mask=None, training=False):
        dtype = self.compute_dtype
        with get_device_context(self.device):
            # Apply Normalization FIRST
            # Padding (0.0) becomes (0 - mean) / std. This is effectively noise.
            # But the mask will be re-applied by initial_batchNorm later.
            x = self.input_normalization(input)

            x = kops.expand_dims(x, axis=-1)
            x = kops.cast(x, dtype)  # Ensure correct dtype for the conv layers
            if mask is not None:
                # broadcast mask to match x shape for batch normalization
                m = tf.cast(mask, dtype)
                while len(m.shape) < len(x.shape):
                    m = tf.expand_dims(m, axis=-1)
            else:
                m = None

            if self.no_cnn:
                # reshape input directly to dense layer
                # must be of shape (batch_size, -1)
                x = self.flatten(input)
                x = self.denseLayer3(x)
                x = self.dropoutLayer(x, training=training)
                print("Skipping CNN layers")
                if mask is not None:
                    x = x * tf.cast(tf.expand_dims(mask, axis=-1), dtype)
                return x

            x = self.convLayer1(x)
            if self.batch_normalization:
                x = self.bn1(x, mask=m, training=training)
            x = self.pool(x)

            x = self.convLayer2(x)
            if self.batch_normalization:
                x = self.bn2(x, mask=m, training=training)
            x = self.pool(x)

            x = self.convLayer3(x)
            if self.batch_normalization:
                x = self.bn3(x, mask=m, training=training)
            x = self.pool(x)

            # or we could simply tf.keras.layers.Flatten() the output of the conv layers - leaves batch size unchanged
            x = self.flatten(x)
            if not self.reduce_dense:
                x = self.denseLayer1(x)
                x = self.dropoutLayer(x, training=training)
                x = self.denseLayer2(x)
                x = self.denseLayer3(x)
            else:
                x = self.denseLayer3(x)
                x = self.dropoutLayer(x, training=training)

            if mask is not None:
                x = x * tf.cast(tf.expand_dims(mask, axis=-1), dtype)

            return x

    @property
    def variables(self):
        vars_list = (
            self.convLayer1.variables
            + self.convLayer2.variables
            + self.convLayer3.variables
            + self.pool.variables
            + self.pool.variables
            + self.pool.variables
            + self.denseLayer3.variables,
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
            self.input_normalization,
            self.convLayer1,
            self.convLayer2,
            self.convLayer3,
            self.pool,
            self.pool,
            self.pool,
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
        # validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch, time, channels), got {input_shape}"
            )
        if input_shape[1] != self.nChannels:
            raise ValueError(
                f"Expected input shape with {self.nChannels} channels, got {input_shape[1]}"
            )

        # Force build of input normalization
        # shape passed to input_normalization is (batch, channels, time)
        # It normalizes along axis=1 (channels) -> means shape (channels,)
        if not self.input_normalization.built:
            self.input_normalization.build(input_shape)

        # We construct the shapes layer by layer to ensure correct build
        # Input to network pipeline: (Batch, Channels, Time)
        # Inside call, it is expanded to (Batch, Channels, Time, 1)
        batch_dim = input_shape[0]
        channels_dim = input_shape[1]
        time_dim = input_shape[2]

        current_shape = (batch_dim, channels_dim, time_dim, 1)

        if self.no_cnn:
            # handle no_cnn path: flatten -> dense3
            # flatten input: (Batch, Channels, Time) -> (Batch, Channels*Time)
            flat_shape = (batch_dim, channels_dim * time_dim)
            if not self.denseLayer3.built:
                self.denseLayer3.build(flat_shape)
            return

        # Layer 1
        if not self.convLayer1.built:
            self.convLayer1.build(current_shape)
        current_shape = self.convLayer1.compute_output_shape(current_shape)

        if self.batch_normalization:
            if not self.bn1.built:
                self.bn1.build(current_shape)

        if not self.pool.built:
            self.pool.build(current_shape)
        current_shape = self.pool.compute_output_shape(current_shape)

        # Layer 2
        if not self.convLayer2.built:
            self.convLayer2.build(current_shape)
        current_shape = self.convLayer2.compute_output_shape(current_shape)

        if self.batch_normalization:
            if not self.bn2.built:
                self.bn2.build(current_shape)

        if not self.pool.built:
            self.pool.build(current_shape)
        current_shape = self.pool.compute_output_shape(current_shape)

        # Layer 3
        if not self.convLayer3.built:
            self.convLayer3.build(current_shape)
        current_shape = self.convLayer3.compute_output_shape(current_shape)

        if self.batch_normalization:
            if not self.bn3.built:
                self.bn3.build(current_shape)

        if not self.pool.built:
            self.pool.build(current_shape)
        current_shape = self.pool.compute_output_shape(current_shape)

        # Flatten
        # current_shape is (Batch, NewH, NewW, NewFilters)
        # Flattened size = NewH * NewW * NewFilters
        flat_size = current_shape[1] * current_shape[2] * current_shape[3]
        flat_shape_tensor = (batch_dim, flat_size)

        if not self.reduce_dense:
            if not self.denseLayer1.built:
                self.denseLayer1.build(flat_shape_tensor)
            if not self.denseLayer2.built:
                self.denseLayer2.build((batch_dim, self.nFeatures))
            if not self.denseLayer3.built:
                self.denseLayer3.build((batch_dim, self.nFeatures))
        else:
            if not self.denseLayer3.built:
                self.denseLayer3.build(flat_shape_tensor)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nFeatures)


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
        batch_normalization=False,
        **kwargs,
    ):
        name = kwargs.pop("name", "spikeNet1D{}".format(number))
        # TODO: implement reduce_dense and no_cnn options if needed
        self.reduce_dense = kwargs.pop("reduce_dense", False)
        self.nConvChannels = kwargs.pop(
            "nConvChannels", 64
        )  # Number of filters in the last conv layer (before global pooling)
        self.no_cnn = kwargs.pop("no_cnn", False)
        super().__init__(name=name, **kwargs)
        self.nChannels = nChannels
        self.nFeatures = nFeatures
        self.batch_normalization = batch_normalization
        self.device = device
        self.number = number
        self.supports_masking = True

        with get_device_context(self.device):
            # Add normalization layer (axis=1 because input is (Batch, Channels, Time))
            self.input_normalization = ChannelwiseFixedNormalization(
                axis=1, name="input_norm"
            )

            layers = []

            # Layer 1
            layers.append(tf.keras.layers.Conv1D(16, 3, padding="same"))
            print(f"batch norm set to {self.batch_normalization}")
            if self.batch_normalization:
                layers.append(MaskedBatchNormalization())
            layers.append(tf.keras.layers.Activation("relu"))
            layers.append(tf.keras.layers.MaxPool1D(2, padding="same"))

            # Layer 2
            layers.append(tf.keras.layers.Conv1D(32, 3, padding="same"))
            if self.batch_normalization:
                layers.append(MaskedBatchNormalization())
            layers.append(tf.keras.layers.Activation("relu"))
            layers.append(tf.keras.layers.MaxPool1D(2, padding="same"))

            # Layer 3
            layers.append(tf.keras.layers.Conv1D(self.nConvChannels, 3, padding="same"))
            if self.batch_normalization:
                layers.append(MaskedBatchNormalization())
            layers.append(tf.keras.layers.Activation("relu"))
            layers.append(tf.keras.layers.GlobalAveragePooling1D())

            # Define the backbone
            self.temporal_extractor_layers = layers

            # Spatial info / channels
            self.channel_interactor = tf.keras.layers.Conv1D(
                filters=self.nConvChannels, kernel_size=3, padding="same"
            )

            # Aggregation
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
            self.flatten = tf.keras.layers.Flatten()
            # Input to dense will be (6 channels * 64 features) = 384
            self.dense_fusion = tf.keras.layers.Dense(nFeatures * 2, activation="relu")
            self.dense_out = tf.keras.layers.Dense(
                nFeatures, activation=None, name=f"outputCNN{number}"
            )

    def call(self, x, mask=None, training=False):
        """
        Forward pass through the SpikeNet1D architecture.

        Batch is supposed to be Batch * nb_spikes, where each spike is treated as a separate example. The temporal extractor processes each channel independently with shared weights, then we learn interactions across channels before the final dense fusion.

        Steps:
        1. Reshape input to (Batch * Channels, Time, 1) for Conv1D processing.
        2. Pass through shared temporal extractor (Conv1D + pooling).
        3. Reshape back to (Batch, Channels, nConvChannels).
        4. Apply channel interaction (Conv1D across channels).
        5. Flatten and pass through dense layers for final feature fusion.

        """
        dtype = self.compute_dtype
        with get_device_context(self.device):
            # Apply Normalization FIRST
            x = self.input_normalization(x)

            # 1. Input is (Batch = Batch*totalNbSpikes, Channels, Time) -> (128, 6, 32)
            total_nb_spikes = tf.shape(x)[0]
            T = 32
            C = self.nChannels

            # Step 1: Reshape to process all channels through the SAME backbone
            # New shape: (Batch * 6, 32, 1)
            # T is hardcoded to be 32 for now (see tfrec creation via julia)
            x = tf.reshape(
                x, [-1, T, 1]
            )  # channels_last format for conv1d and global average pooling
            x = kops.cast(x, dtype)  # Ensure correct dtype for the conv layers

            # Step 2: Extract temporal features (Shared Weights)
            # Output shape: (Batch * 6, self.nConvChannels = 64)
            expanded_mask = None
            if mask is not None:
                expanded_mask = tf.repeat(
                    mask, repeats=C, axis=0
                )  # Repeat mask for each channel
                expanded_mask = tf.expand_dims(
                    expanded_mask, axis=-1
                )  # shape (Batch * 6, 1)

                if len(expanded_mask.shape) == 2:
                    expanded_mask = tf.expand_dims(
                        expanded_mask, axis=-1
                    )  # shape (Batch * 6, 1, 1)

            for layer in self.temporal_extractor_layers:
                # instead of calling the whole sequential, we loop through layers to pass the mask to
                if isinstance(layer, MaskedBatchNormalization):
                    x = layer(x, mask=expanded_mask, training=training)
                else:
                    x = layer(x, training=training)

            # Step 3: Concatenate channels back together
            # New shape: (Batch, 6, nConvChannels) -> (128, 6, 64)
            x_unfolded = tf.reshape(x, [total_nb_spikes, C, self.nConvChannels])

            # Step 4: Spatial Interaction (Learning relationships between fixed channels)
            x = self.channel_interactor(x_unfolded)  # shape still (Batch, 6, 32)

            # Step 5: Flatten and Dense Fusion
            x = self.flatten(x)  # shape (Batch, 6*32 = 192)
            x = self.dense_fusion(x)
            x = self.dropout(x, training=training)
            out = self.dense_out(x)
            if mask is not None:
                out = out * tf.cast(
                    tf.expand_dims(mask, axis=-1), dtype
                )  # Apply mask to final output

            return out

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
                "nConvChannels": self.nConvChannels,
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
            nConvChannels=config.get("nConvChannels", 64),
        )

    def build(self, input_shape):
        # validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch, time, channels), got {input_shape}"
            )
        if input_shape[1] != self.nChannels:
            raise ValueError(
                f"Expected input shape with {self.nChannels} channels, got {input_shape[1]}"
            )

        # Force build of input normalization
        if not self.input_normalization.built:
            self.input_normalization.build(input_shape)

        # ---- Build sub-layers ----
        # 1. Temporal Extractor
        # Input to this is (Batch * Channels, Time, 1)
        # We assume 32 time steps as per the call logic (reshaped to [-1, 32, 1])
        time_steps = 32

        # Propagate shapes through the sequential-like list
        current_shape = (None, time_steps, 1)

        for layer in self.temporal_extractor_layers:
            if not layer.built:
                layer.build(current_shape)
            # Update shape for next layer
            current_shape = layer.compute_output_shape(current_shape)

        # Output of temporal extractor is (Batch*Channels, NewFilteredDim)
        # where NewFilteredDim is nConvChannels (due to GlobalAveragePooling)

        # 2. Channel Interaction
        # Access shape from reshape logic in call: (Batch, Channels, nConvChannels)
        if not self.channel_interactor.built:
            self.channel_interactor.build((None, self.nChannels, self.nConvChannels))

        # Output of interactor is (Batch, Channels, nConvChannels) (same, just mixing)

        # 3. Dense Fusion
        # Input shape becomes (Batch, Channels * nConvChannels)
        # Flattened
        fusion_input_dim = self.nChannels * self.nConvChannels

        # Explicitly build the dense layers so they create their weight variables
        if not self.dense_fusion.built:
            self.dense_fusion.build((None, fusion_input_dim))

        # The output of dense_fusion is nFeatures * 2
        if not self.dense_out.built:
            self.dense_out.build((None, self.nFeatures * 2))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nFeatures)


SpikeNet: TypeAlias = SpikeNet1D | SpikeNet2D


@keras.saving.register_keras_serializable(package="neuroencoders")
class SpikeEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        spikeNets: List[SpikeNet],
        params: Params,
        max_nb_spikes: int,
        max_spikes_per_group: int,
        conv_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spikeNets = spikeNets
        if not isinstance(self.spikeNets, list):
            self.spikeNets = [self.spikeNets]  # Ensure it's always a list

        self.params = params
        self.max_nb_spikes = max_nb_spikes
        self.max_spikes_per_group = max_spikes_per_group
        self.conv_dim = conv_dim
        self.supports_masking = True

    def call(self, inputs, mask=None, training=False):
        dtype = self.compute_dtype
        # IMPORTANT: 'inputs' is a LIST of tensors: [group0, group1, ...]
        # Not a dictionary!
        encoded_groups = []

        for g in range(len(inputs)):
            group_input = inputs[g]
            group_mask = mask[g] if mask is not None else None

            max_spikes = self.max_spikes_per_group
            n_ch = self.params.nChannelsPerGroup[g]

            # Reshape to 3D for Conv: (Batch*MaxSpks, Channels, Time)
            x = tf.reshape(group_input, [-1, n_ch, 32])
            x = tf.cast(x, dtype)  # Ensure correct dtype for the spikeNet

            # When we reshape x to (Batch * MaxSpks, Channels, Time)
            # We MUST reshape mask to (Batch * MaxSpks, 1)
            folded_mask = None
            if group_mask is not None:
                # Keep folded mask rank-1 so downstream dense output masking in
                # SpikeNet broadcasts to (N, 1) rather than introducing a 3rd axis.
                folded_mask = tf.reshape(group_mask, [-1])

            # Forward pass through the specific tower
            x = self.spikeNets[g](
                x, mask=folded_mask, training=training
            )  # Output shape: (Batch*MaxSpks, nFeatures)

            # Reshape back to (Batch, MaxSpks, Features)
            x = tf.reshape(x, [-1, max_spikes, self.params.nFeatures])
            encoded_groups.append(x)

        return encoded_groups

    def compute_output_shape(self, input_shape):
        # Help Keras infer the shapes since there's a loop
        # input_shape is a list of shapes
        return [
            (shape[0], self.max_spikes_per_group, self.params.nFeatures)
            for shape in input_shape
        ]

    def build(self, input_shape):
        # No trainable weights in this layer, but we need to call build on sub-layers
        for g in range(len(input_shape)):
            self.spikeNets[g].build((None, self.params.nChannelsPerGroup[g], 32))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        # turn params in serializable dict with only necessary info
        params_dict = {
            "nGroups": self.params.nGroups,
            "nChannelsPerGroup": self.params.nChannelsPerGroup,
            "nFeatures": self.params.nFeatures,
        }
        # turns spikeNets into a list of their configs (assuming they are serializable)
        serialized_nets = [tf.keras.layers.serialize(net) for net in self.spikeNets]
        config.update(
            {
                "params": params_dict,
                "spikeNets": serialized_nets,
                "conv_dim": self.conv_dim,
                "max_nb_spikes": self.max_nb_spikes,
                "max_spikes_per_group": self.max_spikes_per_group,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct params object from dict
        params_dict = config.pop("params")

        class Params:
            pass

        params = Params()
        for key, value in params_dict.items():
            setattr(params, key, value)

        # Reconstruct spikeNets from their configs
        spikeNets_configs = config.pop("spikeNets", [])
        spikeNets = [
            tf.keras.layers.deserialize(net_config) for net_config in spikeNets_configs
        ]
        return cls(spikeNets=spikeNets, params=params, **config)


@keras.saving.register_keras_serializable(package="neuroencoders")
class SpikeSequenceProcessor(tf.keras.layers.Layer):
    """
    Encapsulates the logic for processing spike groups into a single sequence.
    Typically involves:
    1. SpikeEncoder (per group) -> latents
    2. AddNullSpike (per group) -> to handle padding
    3. Concatenation
    4. GlobalSequenceGather -> reorder to time sequence
    5. Masking (for the whole sequence, will be passed downstream to LSTM/Transformer)
    """

    def __init__(
        self,
        spike_encoder: SpikeEncoder,
        n_groups: int,
        n_features: int,
        max_spikes_per_group: int,
        max_nb_spikes: int,
        device="/cpu:0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spike_encoder = spike_encoder
        self.n_groups = n_groups
        self.n_features = n_features
        self.max_spikes_per_group = max_spikes_per_group
        self.max_nb_spikes = max_nb_spikes
        self.device = device

        # Compute offsets statically
        self.offsets = []
        curr_offset = 0
        for g in range(self.n_groups):
            self.offsets.append(curr_offset)
            # +1 because of the Null Spike added to each group
            curr_offset += self.max_spikes_per_group + 1

        self.add_null_spike_layers = [
            AddNullSpike(n_features=self.n_features, name=f"add_null_spike_group{g}")
            for g in range(self.n_groups)
        ]

        self.sequence_reconstructor = GlobalSequenceGather(
            n_groups=self.n_groups,
            group_dim=self.n_features,
            offsets=self.offsets,
            max_nb_spikes=self.max_nb_spikes,
        )

        self.safe_mask_creation = SafeMaskCreation(name="safe_mask_creation")
        self.masking_layer = MaskingLayer(name="masking_layer_before_rnn")
        self.supports_masking = (
            True  # This layer will produce a mask for downstream layers
        )

    def build(self, input_shape):
        # input_shape is a list of shapes
        # inputsToSpikeNets shapes + indices shapes + inputGroups shape
        # We don't need to do much here as sub-layers are built on call or have fixed logic

        # TODO
        self.spike_encoder.build(
            input_shape[: self.n_groups]
        )  # Build spike encoder with the first n_groups input shapes

        for g in range(self.n_groups):
            # they each receive (Batch, MaxSpikesPerGroup, nFeatures) after the spike encoder and before the gather
            self.add_null_spike_layers[g].build(
                (None, self.max_spikes_per_group, self.n_features)
            )

        # receives the concatenated groups of shape (Batch, n_groups * (MaxSpikesPerGroup + 1), nFeatures) and indices of shape (Batch, SeqLen) and inputGroups of shape (Batch, SeqLen)
        self.sequence_reconstructor.build(
            [
                (
                    None,
                    self.n_groups * (self.max_spikes_per_group + 1),
                    self.n_features,
                ),
                (None, self.max_nb_spikes),
                (None, self.max_nb_spikes),
            ]
        )

        # receives inputGroups of shape (Batch, SeqLen) to create the mask, and also receives the features of shape (Batch, SeqLen, nFeatures) to apply the mask
        self.safe_mask_creation.build(
            [(None, self.max_nb_spikes), (None, self.max_nb_spikes, self.n_features)]
        )

        self.masking_layer.build(
            [(None, self.max_nb_spikes), (None, self.max_nb_spikes, self.n_features)]
        )

        super().build(input_shape)

    def call(self, inputs, mask=None, training=False):
        """
        inputs: List containing:
         - [inputsToSpikeNets...] (nGroups tensors)
         - [indices...] (nGroups tensors)
         - inputGroups (1 tensor)
        """

        inputs_to_spike_nets = inputs[: self.n_groups]
        indices = inputs[self.n_groups : 2 * self.n_groups]
        input_groups = inputs[-1]

        # Optional upstream masks can be propagated by Keras as a list aligned
        # with inputs. We combine them with the explicit validity mask computed
        # from waveform padding.
        incoming_group_masks = None
        if isinstance(mask, (list, tuple)) and len(mask) >= self.n_groups:
            incoming_group_masks = list(mask[: self.n_groups])

        all_group_masks = []
        for g in range(self.n_groups):
            # Create a mask for this specific group: (Batch, MaxSpikesPerGroup)
            # We look at the first channel/time bin; if it's 0 (or your pad value), it's a mask
            # Alternatively, if you have a spike_count tensor, use tf.sequence_mask
            group_data = inputs_to_spike_nets[g]
            # Shape: (Batch, MaxSpikes, Channels, Time) -> Mask: (Batch, MaxSpikes)
            g_mask = tf.reduce_any(tf.not_equal(group_data, 0.0), axis=[-1, -2])
            if incoming_group_masks is not None and incoming_group_masks[g] is not None:
                g_mask = tf.logical_and(
                    g_mask,
                    tf.cast(incoming_group_masks[g], tf.bool),
                )
            all_group_masks.append(g_mask)

        with get_device_context(self.device):
            # 1. ENCODE
            group_latents_raw = self.spike_encoder(
                inputs_to_spike_nets, mask=all_group_masks, training=training
            )

            all_group_latents = []
            for g, latent in enumerate(group_latents_raw):
                full_emb = self.add_null_spike_layers[g](
                    latent, mask=all_group_masks[g]
                )
                all_group_latents.append(full_emb)

            # 2. RECONSTRUCT: Interleave groups back into the temporal sequence
            pool = kops.concatenate(all_group_latents, axis=1)

            # 3. GATHER
            all_features = self.sequence_reconstructor([pool, indices, input_groups])

            # 4. MASKING
            mymask = self.safe_mask_creation(input_groups)
            masked_features = self.masking_layer([mymask, all_features])

            # Sum inputs for legacy/diagnostics
            sum_features = kops.sum(masked_features, axis=1)

            # The layer returns the processed features sequence and the mask
            # But standard Keras layers return one tensor usually, or tuple.
            # We return (masked_features, mymask, sum_features)
            # However, Keras functional model might prefer single tensor output or list.
            # Let's return the main features, and maybe attached mask?
            # Actually downstream logic expects (allFeatures, mymask, sumFeatures) roughly.

            return masked_features, mymask, sum_features, all_features

    def compute_mask(self, inputs, mask=None):
        # The mask is based on the inputGroups tensor, which is the last element in inputs
        input_groups = inputs[-1]
        return self.safe_mask_creation(input_groups)

    def compute_output_shape(self, input_shape):
        # input_shape is a list of shapes:
        # [spike_nets_shapes..., indices_shapes..., group_sequence_shape]
        # group_sequence is the last element
        group_seq_shape = input_shape[-1]
        batch_size = group_seq_shape[0]
        seq_len = group_seq_shape[1]  # same as max nb spikes

        # Returns: (masked_features, mymask, sum_features, all_features)
        return [
            (batch_size, seq_len, self.n_features),  # masked_features
            (batch_size, seq_len),  # mymask
            (batch_size, self.n_features),  # sum_features
            (batch_size, seq_len, self.n_features),  # all_features
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spike_encoder": tf.keras.utils.serialize_keras_object(
                    self.spike_encoder
                ),
                "n_groups": self.n_groups,
                "n_features": self.n_features,
                "max_spikes_per_group": self.max_spikes_per_group,
                "max_nb_spikes": self.max_nb_spikes,
                "device": self.device,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        spike_encoder_config = config.pop("spike_encoder")
        spike_encoder = tf.keras.utils.deserialize_keras_object(spike_encoder_config)
        return cls(spike_encoder=spike_encoder, **config)


########### CONVOLUTIONAL NETWORK CLASS #####################
@keras.saving.register_keras_serializable(package="neuroencoders")
class MaskedSequential(tf.keras.Sequential):
    def __init__(self, layers=None, name=None):
        super().__init__(layers=layers, name=name)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        # This tells the Sequential block to pass the mask
        # to the layers inside it instead of destroying it.
        return mask


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
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                name="group_attention",
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
class GlobalSequenceGather(tf.keras.layers.Layer):
    """
    Gathers features from the concatenated pool of all groups back into the original temporal sequence order.
    """

    def __init__(self, n_groups, group_dim, offsets, max_nb_spikes, **kwargs):
        super().__init__(**kwargs)
        self.n_groups = n_groups
        self.group_dim = group_dim
        self.max_nb_spikes = max_nb_spikes
        self.offsets = kops.convert_to_tensor(offsets, dtype="int32")

    def build(self, input_shape):
        # learnable embedding for each group
        self.group_embeddings = self.add_weight(
            name="group_embeddings",
            shape=(self.n_groups, self.group_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.null_identity = self.add_weight(
            name="null_identity",
            shape=(1, self.group_dim),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        pool, indices_list, group_sequence = inputs
        dtype = self.compute_dtype

        # 1. Vectorized Global Index Calculation
        # pool shape: (Batch, Total_Spikes, Features)
        batch_size = kops.shape(pool)[0]
        seq_len = kops.shape(group_sequence)[1]

        # Stack indices: (n_groups, Batch, SeqLen)
        stacked_indices = kops.stack(indices_list, axis=0)
        offsets_expanded = self.offsets[:, None, None]

        # options: (n_groups, Batch, SeqLen) -> global positions within the pool
        options = stacked_indices + offsets_expanded

        # Pick the correct group's index for every time step
        # Since kops.take doesn't do batch_dims, we pick based on the group_sequence ID
        safe_group_seq = kops.where(
            group_sequence == -1, 0, kops.cast(group_sequence, "int32")
        )

        # We use a masking strategy to flatten the options into the final global_indices
        # This replaces the 'for' loop and 'kops.where' chain
        group_mask = kops.one_hot(
            safe_group_seq, num_classes=self.n_groups
        )  # (B, Seq, nGroups)
        group_mask = kops.transpose(group_mask, [2, 0, 1])  # (nGroups, B, Seq)

        # Extract the specific indices for each group and sum them
        # (This is a standard XLA trick for 'gathering from a list of tensors')
        global_indices = kops.sum(kops.cast(options, "float32") * group_mask, axis=0)
        global_indices = kops.cast(global_indices, "int32")  # (Batch, SeqLen)

        # 2. BATCH TAKE: Flattening for kops.take
        # kops.take usually works on a single axis. To do batch take:
        # We shift global_indices by the batch offset
        batch_offsets = kops.arange(batch_size, dtype="int32") * kops.shape(pool)[1]
        flat_global_indices = kops.reshape(
            global_indices + batch_offsets[:, None], [-1]
        )

        # Flatten the pool: (Batch * Total_Spikes, Features)
        flat_pool = kops.reshape(pool, [-1, self.group_dim])

        # Perform the take
        sequence_features = kops.take(flat_pool, flat_global_indices, axis=0)
        sequence_features = kops.reshape(
            sequence_features, [batch_size, seq_len, self.group_dim]
        )

        # 3. Anatomical Shank Identity
        lookup_table = kops.concatenate(
            [
                self.group_embeddings,
                kops.cast(self.null_identity, self.group_embeddings.dtype),
            ],
            axis=0,
        )

        safe_ids = kops.where(
            group_sequence == -1, self.n_groups, kops.cast(group_sequence, "int32")
        )
        identities = kops.take(lookup_table, kops.reshape(safe_ids, [-1]), axis=0)
        identities = kops.reshape(identities, [batch_size, seq_len, self.group_dim])

        return kops.cast(sequence_features, dtype) + kops.cast(identities, dtype)

    def compute_output_shape(self, input_shape):
        # input_shape[2] is the shape of 'group_sequence' (Batch, SeqLen)
        batch_size = input_shape[2][0]
        seq_len = self.max_nb_spikes  # or input_shape[2][1]
        return (batch_size, seq_len, self.group_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_groups": self.n_groups,
                "group_dim": self.group_dim,
                "offsets": self.offsets,
                "max_nb_spikes": self.max_nb_spikes,
                "device": self.device,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
                outputs = sum_inputs / count_inputs
                outputs._keras_mask = None  # Clear mask after pooling
                return outputs
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

    # Expand to [batch_size, 1, seq_len] for broadcasting across queries
    padding_mask = kops.expand_dims(padding_mask, 1)

    return padding_mask


@keras.saving.register_keras_serializable(package="neuroencoders")
class PositionalEncoding(tf.keras.layers.Layer):
    # increase max_len if you have longer sequences
    def __init__(self, max_len=512, d_model=128, **kwargs):
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.supports_masking = True

    def call(self, x, mask=None):
        with get_device_context(self.device):
            target_dtype = self.compute_dtype

            # 1. Scale input: pre-calculate sqrt(d_model) as a float first
            # This becomes a single constant in the XLA graph
            sqrt_d_model = kops.cast(np.sqrt(self.d_model), target_dtype)
            x = x * sqrt_d_model

            # 2. Static Addition: No slicing!
            # XLA will now 'bake' self.pe into the GPU kernel
            pe = kops.cast(self.pe, target_dtype)
            weight = kops.cast(self.pe_weight, target_dtype)

        return x + (pe * weight)

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

    def compute_mask(self, inputs, mask=None):
        return mask  # Pass through the mask unchanged

    def build(self, input_shape):
        # Create positional encoding matrix
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = self.add_weight(
            name="positional_encoding",
            shape=(self.max_len, self.d_model),
            initializer=tf.keras.initializers.Constant(pe),
            trainable=False,
        )
        self.pe_weight = self.add_weight(
            name="pe_weight",
            shape=(),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)


@keras.saving.register_keras_serializable(package="neuroencoders")
class SafeMaskCreation(tf.keras.layers.Layer):
    """
    Create boolean mask without unnecessary casting.
    True where valid (not equal to -1).
    """

    def call(self, batchedInputGroups):
        pad_value = -1
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
    # Create boolean mask
    padding_mask = kops.not_equal(batchedInputGroups, pad_value)
    return padding_mask


@keras.saving.register_keras_serializable(package="neuroencoders")
class ResidualWrapper(tf.keras.layers.Layer):
    """
    Wraps a layer to add a residual connection: output = layer(input) + input
    """

    def __init__(self, layer_to_wrap: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.layer_to_wrap = layer_to_wrap
        self.supports_masking = True  # Keep masking support if the wrapped layer supports it (eg transformer encoder block)

    def call(self, inputs, mask=None, training=False):
        output = self.layer_to_wrap(inputs, mask=mask, training=training)
        return output + inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return self.layer_to_wrap.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"layer_to_wrap": tf.keras.utils.serialize_keras_object(self.layer_to_wrap)}
        )
        return config

    @classmethod
    def from_config(cls, config):
        layer_config = config.pop("layer_to_wrap")
        layer = tf.keras.utils.deserialize_keras_object(layer_config)
        return cls(layer, **config)

    def build(self, input_shape):
        super().build(input_shape)
        self.layer_to_wrap.build(input_shape)


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
        dropout_rate=0.5,
        device="/cpu:0",
        **kwargs,
    ):
        self.residual = kwargs.pop("residual", True)
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim1 = ff_dim1
        self.ff_dim2 = (
            self.d_model
        )  # Output dimension matches input for residual connection
        self.dropout_rate = dropout_rate
        self.device = device
        self.supports_masking = True  # To indicate that this layer supports masking

        with get_device_context(self.device):
            # Layer normalization at the beginning
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            # Multi-head attention
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                name="mha",
            )

            # Dropout after attention
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

            # Feedforward network
            self.ff_layer1 = tf.keras.layers.Dense(self.ff_dim1, activation="gelu")
            self.ff_layer2 = tf.keras.layers.Dense(self.ff_dim2)

            # Final layer normalization
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True  # To indicate that this layer supports masking

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
            x_norm2 = self.norm2(x)
            ff_output = self.ff_layer1(x_norm2)
            ff_output = self.ff_layer2(ff_output)

            # Final layer norm and residual connection
            x = x + ff_output

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
            "dropout_rate": config.get("dropout_rate", 0.5),
            "device": config.get("device", "/cpu:0"),
        }
        return cls(**layer_config)

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


def validate_padding_contract(
    tensors: Dict[str, tf.Tensor], validate_type: str = "parsed"
):
    """
    Validate that tensors follow the established padding contract:
    - Index tensors (groups, indexInDat) use -1 for padding
    - Spike waveform tensors (group{g}) use 0.0 for padding

    This helps catch subtle bugs where padding conventions are violated.

    Args:
        tensors: Dictionary of tensors to validate
        validate_type: "parsed" (after parse_serialized_sequence) or "raw" (before)
    """
    # Check index tensors use -1 padding
    for key in ["groups", "indexInDat"]:
        if key in tensors:
            t = tensors[key]
            # The minimum value should be -1 (or close, for valid indices >= 0)
            min_val = tf.reduce_min(t)
            # Assert that no value is less than -1 (which would indicate corruption)
            tf.debugging.assert_greater_equal(
                min_val,
                -1,
                message=f"Tensor '{key}' has values < -1, violating padding contract",
            )

    # Check spike waveforms use 0.0 padding (only for dense tensors)
    for g in range(32):  # Max nGroups
        key = f"group{g}"
        if key in tensors:
            t = tensors[key]
            # Should be float type; check that padded regions are 0.0
            # A simple heuristic: rows/spikes that are all 0.0 are padding
            # This is a soft check--no assertion here by design.
            _ = tf.reduce_all(tf.equal(t, 0.0), axis=list(range(1, len(t.shape))))


@tf.function
def parse_serialized_sequence(
    params: Params,
    tensors: Dict[str, tf.Tensor],
    count_spikes: bool = False,
    sorted_indices: Optional[bool] = None,
    max_spikes: Optional[int] = None,
    max_spikes_per_group: Optional[int] = None,
):
    """
    Parse sparse tensors into dense padded tensors following strict padding contract:
    - groups, indexInDat padded with -1
    - group{g} spike waveforms padded with 0.0
    """
    # TODO: add sorted indices to the function, in order to filter by indexInDat (eg spike sorting)
    tensors = dict(tensors)
    if max_spikes is None:
        max_spikes = getattr(params, "max_nb_spikes", 512)
    if max_spikes_per_group is None:
        max_spikes_per_group = getattr(params, "max_nb_spikes_per_group", 128)

    # Track total sparse group entries before densification.
    num_groups = tf.shape(tensors["groups"].indices)[0]

    if max_spikes is not None:
        actual_total = tf.shape(tensors["groups"].indices)[0]

        # 2. Determine the truncation limit (the smaller of the two)
        # This prevents errors if actual_total is already smaller than max_spikes
        limit = tf.minimum(actual_total, max_spikes)

        # 3. Slice the indices and values to the limit
        tensors["groups"] = tf.sparse.SparseTensor(
            indices=tensors["groups"].indices[:limit],
            values=tensors["groups"].values[:limit],
            dense_shape=tf.cast(tf.stack([limit]), tf.int64),
        )

        # 4. Optional: Add your "Simple Warning" here
        tf.cond(
            actual_total > max_spikes,
            lambda: tf.print("⚠️ Truncating sample:", actual_total, "->", max_spikes),
            lambda: tf.no_op(),
        )
    # 1. Handle Metadata (Vectorized to avoid CPU overhead)
    # Padding contract: use -1 for index/metadata tensors
    lengths = []
    default = -1
    for key in ["pos", "groups", "indexInDat"]:
        if isinstance(tensors[key], tf.SparseTensor):
            padded_sparse = tf.sparse.reset_shape(tensors[key], new_shape=[max_spikes])
            tensors[key] = tf.sparse.to_dense(padded_sparse, default_value=default)
        if key == "pos":
            tensors[key] = tf.reshape(tensors[key], [params.dimOutput])

    # 3. Process each group to fixed-size dense blocks
    # Padding contract: use 0.0 for spike waveforms
    for g in range(params.nGroups):
        group_key = f"group{g}"

        spike_size = params.nChannelsPerGroup[g] * 32
        flat_max_size = max_spikes_per_group * spike_size
        total_entries = tf.shape(tensors[group_key].indices)[0]
        actual_spike_count = total_entries // spike_size

        lengths.append(actual_spike_count)  # Keep track of actual spike count per group
        limit = tf.minimum(actual_spike_count, max_spikes_per_group)
        new_flat_len = limit * spike_size
        tensors[group_key] = tf.sparse.SparseTensor(
            indices=tensors[group_key].indices[:new_flat_len],
            values=tensors[group_key].values[:new_flat_len],
            dense_shape=tf.cast(tf.stack([new_flat_len]), tf.int64),
        )
        tf.cond(
            actual_spike_count > max_spikes_per_group,
            lambda: tf.print(
                f"⚠️ Truncating group {g} spikes:",
                actual_spike_count,
                "->",
                max_spikes_per_group,
            ),
            lambda: tf.no_op(),
        )

        # Convert Sparse to Dense: use 0.0 padding for waveforms
        padded_spikes = tf.sparse.reset_shape(
            tensors[group_key], new_shape=[flat_max_size]
        )
        raw_flat = tf.sparse.to_dense(padded_spikes, default_value=0.0)

        tensors[group_key] = tf.reshape(
            raw_flat, [max_spikes_per_group, params.nChannelsPerGroup[g], 32]
        )

        if count_spikes:
            tensors[f"group{g}_spikes_count"] = actual_spike_count

    # 5. Length and Masking
    # Keep track of actual length for SafeMaskCreation
    tensors["total_nb_spikes"] = tf.cast(num_groups, tf.int32)
    tensors["max_spikes_in_groups"] = tf.reduce_max(tf.stack(lengths))

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
            normalize: Whether to normalize data (default: False)
            normalization_stats: Tuple of (means, stds) for normalization. means and stds are lists of arrays per group.
        """
        self.keep_original = kwargs.get("keep_original", True)
        self.num_augmentations = kwargs.get("num_augmentations", 11)
        self.white_noise_std = kwargs.get("white_noise_std", 0.05)
        self.offset_noise_std = kwargs.get("offset_noise_std", 0.05)
        self.offset_scale_factor = kwargs.get("offset_scale_factor", 0.67)
        self.cumulative_noise_std = kwargs.get("cumulative_noise_std", 0.02)
        spike_band_channels = kwargs.get("spike_band_channels", None)
        self.spike_band_channels = (
            spike_band_channels if spike_band_channels is not None else []
        )
        self.device = kwargs.get("device", "/cpu:0")
        self.normalize = kwargs.get("normalize", False)
        self.normalization_stats = kwargs.get("normalization_stats", None)

        if self.normalize:
            # If normalization is enabled, we assume the data will be ~unit variance.
            # We scale the default noise levels down if they appear to be at the "raw" scale.
            # Heuristic: if white_noise_std > 1.0, it's probably for raw data.
            if self.white_noise_std > 1.0:
                print(
                    f"Scaling down noise levels for normalized data (was {self.white_noise_std})"
                )
                self.white_noise_std /= 50.0  # e.g. 2.0 -> 0.04
                self.offset_noise_std /= 50.0
                print(f"New white_noise_std: {self.white_noise_std}")

    def normalize_group(self, group_data: tf.Tensor, group_idx: int) -> tf.Tensor:
        """
        Normalize group data using stored stats.
        group_data: (Batch, Channels, Time)
        """
        if not self.normalize or self.normalization_stats is None:
            return group_data

        print(
            "Normalization enabled from withing NeuralDataAugmentation. Checking noise levels..."
        )
        means, stds = self.normalization_stats
        # data checks
        if group_idx >= len(means) or group_idx >= len(stds):
            return group_data

        # shape of means[g]: (Channels,) or (1, Channels, 1) or similar
        # data is (Spikes, Channels, Time)
        m = tf.convert_to_tensor(means[group_idx], dtype=group_data.dtype)
        s = tf.convert_to_tensor(stds[group_idx], dtype=group_data.dtype)

        # Ensure shapes for broadcasting: (1, Channels, 1)
        if len(m.shape) == 1:
            m = m[tf.newaxis, :, tf.newaxis]
        if len(s.shape) == 1:
            s = s[tf.newaxis, :, tf.newaxis]

        # Avoid division by zero
        s = tf.where(tf.equal(s, 0), tf.ones_like(s), s)

        return (group_data - m) / s

        if group_idx >= len(means) or group_idx >= len(stds):
            return group_data

        # shape of means[g]: (Channels,) or (1, Channels, 1) or similar
        # data is (Spikes, Channels, Time)
        m = tf.convert_to_tensor(means[group_idx], dtype=group_data.dtype)
        s = tf.convert_to_tensor(stds[group_idx], dtype=group_data.dtype)

        # Ensure shapes for broadcasting: (1, Channels, 1)
        if len(m.shape) == 1:
            m = m[tf.newaxis, :, tf.newaxis]
        if len(s.shape) == 1:
            s = s[tf.newaxis, :, tf.newaxis]

        # Avoid division by zero
        s = tf.where(tf.equal(s, 0), tf.ones_like(s), s)

        return (group_data - m) / s

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
        # Convert negative axis to positive to ensure slicing works correctly
        rank = tf.rank(neural_data)
        if axis < 0:
            axis = rank + axis

        # Create offset shape - same as neural_data but with 1 along time dimension
        shape = tf.shape(neural_data)

        # We assume time dimension is AFTER channel dimension (axis + 1)
        # If axis is the last dimension, this logic fails, but usually constant offset
        # is across time for channels.

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

    def call(self, neural_data: tf.Tensor, time_axis: int = -1, channel_axis: int = -2):
        return self.augment_sample(neural_data, time_axis, channel_axis)


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

        if augmentation_config.normalize:
            group_data = augmentation_config.normalize_group(group_data, g)

        # now that ours groups are fixed size and already padded, we need a mask to avoid augmenting the padded part
        is_real_spike = tf.reduce_any(
            tf.not_equal(group_data, 0.0), axis=[1, 2]
        )  # Shape: [Spikes]
        # broadcast to match group_data shape
        is_real_spike = is_real_spike[
            tf.newaxis, :, tf.newaxis, tf.newaxis
        ]  # Shape: [1, Spikes, 1, 1]

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

        # zero out the padded spikes to ensure they remain unchanged after augmentations
        augmented_versions = tf.where(is_real_spike, augmented_versions, 0.0)

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
        "total_nb_spikes",
        "time",
        "time_behavior",
        "indexInDat",
        "max_spikes_in_groups",
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
        self.maze_points = tf.constant(maze_points, dtype=tf.float32)
        self.ts_proj = tf.constant(ts_proj, dtype=tf.float32)

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
            projected_pos = kops.cast(
                kops.take(self.maze_points, best_points), euclidean_data.dtype
            )
            linear_pos = kops.cast(
                kops.take(self.ts_proj, best_points), euclidean_data.dtype
            )

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
                "grid_size": self.grid_size,
                "maze_params": self.maze_params_dict,
                "smoothing_factor": self.smoothing_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        maze_params = config.pop("maze_params", None)
        smoothing_factor = config.pop("smoothing_factor", 0.01)
        grid_size = config.pop("grid_size", 50)
        return cls(
            grid_size=grid_size,
            maze_params=maze_params,
            smoothing_factor=smoothing_factor,
            **config,
        )


# Custom layer that combines feature_output and UMazeProjectionLayer
@keras.saving.register_keras_serializable(package="neuroencoders")
class FeatureOutputWithUMaze(tf.keras.layers.Layer):
    def __init__(
        self, orig_layer_config, grid_size=(45, 45), maze_params=None, **kwargs
    ):
        super().__init__(**kwargs)
        # Rebuild the original layer (Dense in your case)
        self.orig = tf.keras.layers.Dense.from_config(orig_layer_config)
        self.proj = UMazeProjectionLayer(grid_size=grid_size, maze_params=maze_params)

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
        self.deviceName = kwargs.pop("device", None)
        tf.keras.layers.Layer.__init__(self, **kwargs)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        self.training_positions = training_positions
        self.eps = eps
        self.sigma = sigma
        self.neg = neg
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
                "device": self.deviceName,
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
        self.feature_to_logits_map.build(input_shape)
        # Ensure computed attributes are initialized after build
        if not hasattr(self, "EPS"):
            self._initialize_computed_attributes()

        super().build(input_shape)


@keras.saving.register_keras_serializable(package="neuroencoders")
class GaussianHeatmapLosses(tf.keras.losses.Loss, SpatialConstraintsMixin):
    """
    A simple wrapup class to hold various loss functions and handle keras symbols. Inherits from
    GaussianHeatmapLayer to access masks and constants.
    """

    def __init__(
        self,
        l_function_layer_params,
        training_positions=None,
        grid_size=(45, 45),
        eps=1e-8,
        sigma=0.03,
        neg=-100,
        maze_params=None,
        sinkhorn_eps=0.4,
        loss_type="safe_kl",
        WMAP=None,
        scale=False,
        name="gaussian_heatmap_losses",
        **kwargs,
    ):
        """
        Args:
            heatmap_layer: An instance of GaussianHeatmapLayer to provide masks and constants.
        """
        policy = tf.keras.mixed_precision.global_policy()
        self.storage_dtype = policy.compute_dtype

        self.deviceName = kwargs.pop("device", None)
        self.return_batch = kwargs.pop("return_batch", False)
        tf.keras.losses.Loss.__init__(self, name=name, **kwargs)
        SpatialConstraintsMixin.__init__(
            self, grid_size=grid_size, maze_params=maze_params
        )
        self.loss_type = loss_type
        if WMAP is not None:
            self.WMAP = tf.cast(tf.constant(WMAP), tf.float32)
        else:
            self.WMAP = None

        self.scale = scale
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
            device=self.l_function_layer_params.pop("device", self.deviceName),
            name=self.l_function_layer_params.get("name", "linearization_layer"),
        )
        self.sinkhorn_eps = sinkhorn_eps

        with get_device_context(self.deviceName):
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
            self.allowed_indices = tf.constant(
                mask_indices,
                dtype=tf.int32,
                name="allowed_indices",
            )

            self.N_valid = kops.shape(mask_indices)[0]
            self._precompute_cost_matrix()

    def call(self, y_true, y_pred):
        """
        Compute loss in a Keras symbolic-safe way.
        """
        with get_device_context(self.deviceName):
            pred_shape = kops.shape(y_pred)
            true_shape = kops.shape(y_true)
            if len(pred_shape) == 2:
                y_pred = kops.reshape(
                    y_pred,
                    (-1, self.GRID_H, self.GRID_W),
                )

            if true_shape[1] == 2:
                # If input is (B, 2), assume it's (x,y) and convert to heatmap targets
                y_true = self.gaussian_heatmap_targets_tf(y_true)

            y_true = kops.cast(y_true, y_pred.dtype)
            if self.loss_type == "weighted":
                return self._weighted_heatmap_loss(y_pred, y_true, wmap=self.WMAP)
            elif self.loss_type == "kl":
                return self._kl_heatmap_loss(
                    y_pred, y_true, wmap=self.WMAP, scale=self.scale
                )
            elif self.loss_type == "safe_kl":
                return self._safe_kl_heatmap_loss(
                    y_pred, y_true, wmap=self.WMAP, scale=self.scale
                )
            elif self.loss_type == "wasserstein":
                return self._safe_kl_wasserstein_heatmap_loss(y_pred, y_true)
            else:
                raise ValueError("Unknown loss_type:" + str(self.loss_type))

    def get_config(self):
        """Return the config dict for serialization"""
        config = tf.keras.losses.Loss.get_config(self)

        # Convert TensorFlow tensors to Python scalars
        neg_value = self.neg
        if hasattr(neg_value, "numpy"):
            neg_value = float(neg_value.numpy())

        if self.l_function_layer_params is not None:
            # handle l function layer params serialization
            l_function_layer_params_serializable = self.l_function_layer_params.copy()
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
                "training_positions": self._training_positions_serializable,
                "grid_size": self.grid_size,
                "l_function_layer_params": l_function_layer_params_serializable,
                "eps": self.eps,
                "sigma": self.sigma,
                "neg": neg_value,
                "maze_params": self.maze_params,
                "sinkhorn_eps": self.sinkhorn_eps,
                "loss_type": self.loss_type,
                "scale": self.scale,
                "WMAP": self.WMAP,
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
    ):
        """
        Numerically stable KL divergence loss between target heatmap (P) and predicted (Q).
        Equivalent to KL(P||Q), but implemented using TensorFlow cross-entropy ops.
        """
        batch_size = tf.shape(logits_hw)[0]
        allowed_mask = self.allowed_mask_tf

        # Accept both flattened [B, H*W] and image-shaped [B, H, W] targets.
        if len(target_hw.shape) == 2:
            target_hw = kops.reshape(target_hw, (batch_size, self.GRID_H, self.GRID_W))

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
            weights = kops.cast(wmap[None], "float32")
            valid_mask = kops.cast(weights > 0, "float32")
            wsum = kops.sum(weights * valid_mask) + self.EPS
            kl = kl * (kops.sum(weights) / wsum)

        loss = kops.mean(kl)
        if scale:
            loss /= 100.0
        return loss

    def _precompute_cost_matrix(self):
        """
        Precompute cost matrix using only keras.ops so it works symbolically
        across backends (TF, JAX, Torch).
        """
        with get_device_context(self.deviceName):
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
            C_raw = C.numpy()  # Evaluate to numpy for rescaling

            # --- Make sure cost_matrix and lin_coords are stored as eager tensors

            C_rescaled, info = rescale_cost_matrix(C_raw)
            C_fixed = tf.constant(C_rescaled, dtype=tf.float32)
            # store as eager tf.constant so they won't be graph-captured later
            self.cost_matrix = kops.cast(C_fixed, self.storage_dtype)
            self.lin_coords = tf.constant(lin_coords.numpy(), self.storage_dtype)

            ####
            # Sinkhorn kernel
            ####
            # # compute kernel once and store as CPU-side constant; don't keep gradient tracking
            # eps_tf = tf.cast(self.sinkhorn_eps, self.storage_dtype)
            # self.kernel = tf.constant(
            #     kops.exp(-self.cost_matrix / eps_tf), self.storage_dtype
            # )
            # self.M = self.kernel * self.cost_matrix

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

    @tf.function
    def _safe_kl_wasserstein_heatmap_loss(
        self,
        logits_hw,
        target_hw,
        alpha=None,
        sinkhorn_iters=20,
        **kwargs,
    ):
        """
        KL divergence + optional Wasserstein distance (Sinkhorn)
        using precomputed linearized maze cost matrix.
        """
        dtype = logits_hw.dtype
        if alpha is None:
            alpha = 1  # default weight for Wasserstein penalty

        batch_size = kops.shape(logits_hw)[0]
        allowed_mask = kops.cast(self.allowed_mask_tf, dtype)
        forbid_mask = kops.cast(self.forbid_mask_tf, dtype)
        cost_matrix = kops.cast(self.cost_matrix, dtype)
        target_hw = kops.cast(target_hw, dtype)
        NEG = kops.cast(self.NEG, dtype)
        EPS = kops.cast(self.EPS, dtype)

        # Mask + logits flatten
        masked_logits = kops.where(
            kops.expand_dims(forbid_mask, 0) > 0,
            NEG,
            logits_hw,
        )
        logits_flat = kops.reshape(
            masked_logits, (batch_size, self.GRID_H * self.GRID_W)
        )

        # Normalize target P
        P = target_hw * allowed_mask
        P_sum = kops.sum(P, axis=[1, 2], keepdims=True)
        P = kops.where(
            P_sum > EPS,
            P / (P_sum + EPS),
            allowed_mask / kops.sum(allowed_mask),
        )
        P = P / (kops.sum(P, axis=[1, 2], keepdims=True) + EPS)
        P_flat = kops.reshape(P, (batch_size, self.GRID_H * self.GRID_W))

        # --- KL(P||Q) ---
        q_probs = kops.softmax(logits_flat, axis=-1)

        P_allowed = tf.gather(P_flat, self.allowed_indices, axis=1)  # [B, N_valid]
        q_allowed = tf.gather(q_probs, self.allowed_indices, axis=1)  # [B, N_valid]

        ce = -kops.sum(P_allowed * kops.log(q_allowed + EPS), axis=-1)
        entropy = -kops.sum(P_allowed * tf.math.log(P_allowed + EPS), axis=-1)
        kl = ce - entropy  # [B]

        # --- Wasserstein penalty ---
        if alpha > 0.0:
            # use precomputed CPU-side constants (self.kernel, self.cost_matrix)
            P_allowed = P_allowed / (kops.sum(P_allowed, axis=1, keepdims=True) + 1e-9)
            q_allowed = q_allowed / (kops.sum(q_allowed, axis=1, keepdims=True) + 1e-9)
            temp = kops.matmul(P_allowed, cost_matrix)  # [batch, N]
            W = kops.sum(temp * q_allowed, axis=1)  # [batch]
            loss = kl + alpha * kops.log(W + kops.expand_dims(EPS, 0))
        else:
            loss = kl

        return kops.reshape(loss, (-1, 1))

    def _safe_linear_kl_wasserstein_heatmap_loss(
        self,
        logits_hw,
        target_hw,
        alpha=1.0,
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
    C = np.array(C_orig, dtype=np.float32)
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


def bin_class(example, GRID_W, GRID_H, FORBID, stride=None):
    """
    Map true (x,y) position to discrete bin class, -1 if forbidden.
    """
    pos = example["pos"]
    x = tf.cast(tf.clip_by_value(pos[0] * GRID_W, 0, GRID_W - 1), tf.int32)
    y = tf.cast(tf.clip_by_value(pos[1] * GRID_H, 0, GRID_H - 1), tf.int32)

    if stride is not None:
        # downscale to coarser grid
        x = x // stride
        y = y // stride

    bin_cls = y * GRID_W + x

    if stride is not None:
        # Check if forbidden
        forbidden_here = tf.greater(FORBID[y * stride, x * stride], 0)
    else:
        forbidden_here = tf.gather_nd(FORBID, tf.stack([y, x], axis=-1))

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


class ContrastiveVisualizer(tf.keras.callbacks.Callback):
    def __init__(
        self,
        viz_x,
        viz_y,
        encoder_model: tf.keras.models.Model,
        params: Params,
        epoch_freq=1,
        save_dir: str = "log_viz",
        trial_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.viz_x = viz_x
        self.viz_y = viz_y
        self.encoder_model = encoder_model
        self.epoch_freq = epoch_freq
        self.save_dir = save_dir if save_dir is not None else "log_viz"
        self.trial_idx = trial_idx if trial_idx is not None else 0
        self.params_class = params

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_freq != 0:
            return

        # 1. Get Latents for PCA
        try:
            # We want the output of the full encoder (after Transformer blocks)
            # but before any final projection/pooling if you stopped at MHA
            latents = self.encoder_model.predict(self.viz_x, verbose=0)

            # If output is (batch, seq, dim), average across spikes for cleaner clusters
            if len(latents.shape) > 2:
                latents = np.mean(latents, axis=1)
        except Exception as e:
            print(f"\n[Visualizer] Latent extraction failed: {e}")
            return

        # 2. PCA Projection
        proj = PCA(n_components=2).fit_transform(latents)

        # 3. Plot PCA Scatter
        plt.figure(figsize=(10, 7))
        sc = plt.scatter(
            proj[:, 0], proj[:, 1], c=self.viz_y, cmap="plasma", s=20, alpha=0.6
        )
        plt.colorbar(sc, label="Target Value (LinPos)")
        plt.title(f"Latent Space - Epoch {epoch + 1}")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True, linestyle="--", alpha=0.3)

        plt.savefig(os.path.join(self.save_dir, f"latent_epoch_{epoch + 1:03d}.png"))
        plt.close()

        # 4. Save Attention Map Snapshot (using your working logic)
        self._save_attn_snapshot(epoch)
        self._combined_save_attn_snapshot(epoch)
        if epoch == 0:  # Only plot raw spikes on the first epoch to save time
            self._plot_raw_spikes(epoch)

    def _save_attn_snapshot(self, epoch):
        try:
            # Mirror your working function indexing
            transformer_encoder = self.encoder_model.get_layer("transformer_encoder")
            internal_block = transformer_encoder.layers[1]

            # Sub-model for prefix
            prefix_model = tf.keras.Model(
                inputs=self.encoder_model.input, outputs=transformer_encoder.input
            )
            x = prefix_model(self.viz_x, training=False)

            # Handle mask/tensor list
            mask = None
            if isinstance(x, (list, tuple)):
                mask = x[1] if len(x) > 1 else None
                x = x[0]

            # Apply internal norm1
            x_norm = internal_block.norm1(x)

            # Create attention mask
            attention_mask = None
            if mask is not None:
                attention_mask = create_attention_mask_from_padding_mask(mask)

            # Get weights (Trial 0, Head 0)
            _, weights = internal_block.mha(
                query=x_norm,
                value=x_norm,
                attention_mask=attention_mask,
                return_attention_scores=True,
                training=False,
            )
            trial_weights = weights[self.trial_idx]  # [heads, seq, seq]
            attn_map = tf.cast(
                tf.reduce_mean(trial_weights, axis=0), tf.float32
            ).numpy()  # average over heads -> [seq, seq]

            # Plot and save map
            plt.figure(figsize=(8, 6))
            plt.imshow(attn_map, cmap="viridis", aspect="equal")
            plt.title(f"Attention Snapshot - Epoch {epoch + 1}")
            plt.colorbar(label="Weight")
            plt.savefig(os.path.join(self.save_dir, f"attn_epoch_{epoch + 1:03d}.png"))
            plt.close()

            # Create a grouped colorbar/labels for the heatmap axes
            groups_for_trial = self.viz_x["groups"][self.trial_idx]
            if hasattr(groups_for_trial, "numpy"):
                groups_for_trial = groups_for_trial.numpy()

            valid_len = np.sum(groups_for_trial != -1)

            # Slice the map to remove padding from the visual
            clean_attn = attn_map[:valid_len, :valid_len]

            plt.figure(figsize=(10, 8))
            sns.heatmap(clean_attn, cmap="viridis", xticklabels=5, yticklabels=5)
            plt.title(f"Mean Attention - Epoch {epoch + 1} (Non-padded spikes only)")
            plt.savefig(
                os.path.join(self.save_dir, f"sns_attn_epoch_{epoch + 1:03d}.png")
            )
            plt.close()

        except Exception as e:
            print(f"\n[Visualizer] Attention snapshot failed: {e}")

    def _plot_raw_spikes(self, epoch):
        # Extract the specific best trial
        # Note: self.viz_x must contain the keys used in your example
        example = {k: v[self.trial_idx] for k, v in self.viz_x.items()}

        n_groups = self.params_class.nGroups  # Or params.nGroups
        fig, axs = plt.subplots(n_groups, 1, figsize=(12, 2 * n_groups), sharex=True)

        cmap_pool = [
            "tab10",
            "Set1",
            "Set2",
            "Set3",
            "Dark2",
            "Paired",
            "Accent",
            "tab20",
            "tab20b",
            "tab20c",
            "Pastel1",
            "Pastel2",
        ]
        # Pre-config groups
        cmaps = [plt.get_cmap(name) for name in cmap_pool[:n_groups]]

        # Plot limited spikes for clarity (e.g., first 100)
        num_spikes = min(100, int(example["length"]))

        for i in range(num_spikes):
            idx_group = int(example["groups"][i])
            if idx_group == -1:
                continue  # Skip padding

            # Identify spike data
            target_data = example[f"group{idx_group}"]
            spike_idx = int(example[f"indices{idx_group}"][i])

            if spike_idx == 0:
                continue  # Padded spike

            spike_to_plot = target_data[spike_idx - 1].astype(np.float64)
            start_of_spike = example["indexInDat"][i] - 16
            time_axis = np.arange(start_of_spike, start_of_spike + 32).astype(np.int64)

            ax = axs[idx_group]
            for ch in range(spike_to_plot.shape[0]):
                ax.plot(
                    time_axis,
                    spike_to_plot[ch, :],
                    c=cmaps[idx_group](ch),
                    alpha=0.5,
                    lw=1,
                )

        axs[-1].set_xlabel("Sample Index")
        fig.suptitle(f"Input Spikes (Trial {self.trial_idx}) - Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"spikes_epoch_{epoch + 1:03d}.png"))
        plt.close()

    def _combined_save_attn_snapshot(self, epoch):
        try:
            cmap_pool = [
                "tab10",
                "Set1",
                "Set2",
                "Set3",
                "Dark2",
                "Paired",
                "Accent",
                "tab20",
                "tab20b",
                "tab20c",
                "Pastel1",
                "Pastel2",
            ]
            # 1. Extraction Logic (Same as before)
            transformer_encoder = self.encoder_model.get_layer("transformer_encoder")
            internal_block = transformer_encoder.layers[1]
            prefix_model = tf.keras.Model(
                inputs=self.encoder_model.input, outputs=transformer_encoder.input
            )
            x_raw = prefix_model(self.viz_x, training=False)

            mask = x_raw[1] if isinstance(x_raw, (list, tuple)) else None
            x_raw = x_raw[0] if isinstance(x_raw, (list, tuple)) else x_raw

            x_norm = internal_block.norm1(x_raw)
            _, weights = internal_block.mha(
                query=x_norm,
                value=x_norm,
                attention_mask=create_attention_mask_from_padding_mask(mask)
                if mask is not None
                else None,
                return_attention_scores=True,
                training=False,
            )

            # 2. Prepare Data
            # Average heads and slice to valid (non-padded) sequence length
            groups_for_trial = self.viz_x["groups"][self.trial_idx]
            if hasattr(groups_for_trial, "numpy"):
                groups_for_trial = groups_for_trial.numpy()
            valid_len = np.sum(groups_for_trial != -1)
            # Use first 100 max for visual clarity
            plot_len = min(200, valid_len)

            attn_map = tf.cast(
                tf.reduce_mean(weights[self.trial_idx], axis=0), tf.float32
            )
            attn_map = attn_map[:plot_len, :plot_len].numpy()
            example = {k: v[self.trial_idx] for k, v in self.viz_x.items()}

            # 3. Setup GridSpec (1 row for spikes, 1 row for attention)
            fig = plt.figure(figsize=(12, 14))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)

            ax_spikes = fig.add_subplot(gs[0])
            ax_attn = fig.add_subplot(gs[1])

            # 4. Plot Spikes (Aligned to the attention columns)
            cmaps = [
                plt.get_cmap(name) for name in cmap_pool[: self.params_class.nGroups]
            ]

            for i in range(plot_len):
                g_idx = int(groups_for_trial[i])
                spike_idx = int(example[f"indices{g_idx}"][i])
                if spike_idx == 0:
                    continue

                wave = example[f"group{g_idx}"][spike_idx - 1]  # [channels, samples]
                # Center the waveform at the column index 'i'
                time_axis = np.linspace(i - 0.4, i + 0.4, wave.shape[1])

                time_axis = time_axis.astype(np.float64)
                wave = wave.astype(np.float64)

                for ch in range(wave.shape[0]):
                    ax_spikes.plot(
                        time_axis,
                        wave[ch, :],
                        color=cmaps[g_idx](ch),
                        lw=0.8,
                        alpha=0.7,
                    )

            ax_spikes.set_title(f"Aligned Spikes & Attention - Epoch {epoch + 1}")
            ax_spikes.set_ylabel("Voltage")
            ax_spikes.set_xlim(-0.5, plot_len - 0.5)
            ax_spikes.axis("off")  # Cleaner look

            # 5. Plot Attention Map
            im = ax_attn.imshow(attn_map, cmap="viridis", aspect="auto", origin="upper")
            ax_attn.set_xlabel("Key Spike Index")
            ax_attn.set_ylabel("Query Spike Index")

            # Colorbar
            plt.colorbar(
                im,
                ax=ax_attn,
                orientation="horizontal",
                fraction=0.05,
                pad=0.1,
                label="Mean Attention Weight",
            )

            plt.savefig(
                os.path.join(self.save_dir, f"combined_viz_epoch_{epoch + 1:03d}.png"),
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            print(f"\n[Visualizer] Combined snapshot failed: {e}")


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
        policy = tf.keras.mixed_precision.global_policy()
        self.storage_dtype = policy.compute_dtype
        self.return_batch = kwargs.pop("return_batch", False)
        super().__init__(name=name, **kwargs)
        self.high = high

    def call(self, y_true, y_pred):
        # Ensure types match
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Calculate raw difference
        delta = y_pred - y_true

        # Replace % with tf.math.mod
        # Formula: ((delta + half) % high) - half
        half = self.high / 2.0

        # This handles the wrapping around the 'high' boundary (e.g., 2*pi)
        dist = tf.math.mod(delta + half, self.high) - half
        dist = tf.abs(dist)
        dist = kops.reshape(dist, (-1, 1))  # Ensure output is (batch_size,)

        return dist

    def get_config(self):
        config = super().get_config()
        config.update({"high": self.high})
        return config


@tf.keras.utils.register_keras_serializable(package="neuroencoders")
class ScaledSigmoid(tf.keras.layers.Layer):
    def __init__(self, high=2 * np.pi, **kwargs):
        super(ScaledSigmoid, self).__init__(**kwargs)
        self.high = high

    def call(self, inputs):
        return tf.math.sigmoid(inputs) * self.high

    def get_config(self):
        config = super(ScaledSigmoid, self).get_config()
        config.update({"high": self.high})
        return config


@tf.keras.utils.register_keras_serializable(package="neuroencoders")
def scaled_sigmoid(x, high=2 * np.pi):
    # Scales 0->1 to 0->2pi
    return tf.math.sigmoid(x) * high


@keras.saving.register_keras_serializable(package="neuroencoders")
class ContrastiveRegressionLoss(tf.keras.losses.Loss):
    """
    Contrastive Loss based on NT-Xent with spatial weighting (linearized positions) and other target weighting functions.
    computing the NT-Xent loss directly without internal layers.
    """

    def __init__(
        self,
        target_structure: Dict[str, Dict[str, Any]],
        temperature: float = 0.4,
        sigma: float = 0.1,
        l_function_params: Optional[Dict] = None,
        name: str = "contrastive_loss",
        **kwargs,
    ):
        policy = tf.keras.mixed_precision.global_policy()
        self.storage_dtype = policy.compute_dtype
        self.target_structure = target_structure
        self.return_batch = kwargs.pop("return_batch", False)
        self.deviceName = kwargs.pop("device", None)
        super().__init__(name=name, **kwargs)
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.l_function_params = l_function_params
        self.eps = 1e-8

        self.l_function = None
        if l_function_params is not None:
            self.l_function = LinearizationLayer(**l_function_params)

    def __call__(self, y_true, y_pred=None, sample_weight=None):
        # Legacy call style in this codebase uses loss_layer([y_true, y_pred]).
        if y_pred is None and isinstance(y_true, (list, tuple)) and len(y_true) == 2:
            y_true, y_pred = y_true
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

    def call(self, y_true, y_pred=None):
        """
        Compute the contrastive regression loss.
        inputs: y_true: (batch, 2+) coords + others, y_pred: (batch, D) latent
        outputs: scalar loss value (or batch of losses if return_batch=True or reduction='none')
        """
        # Backward compatibility: allow layer([y_true, y_pred]) invocation style.
        if y_pred is None and isinstance(y_true, (list, tuple)) and len(y_true) == 2:
            y_true, y_pred = y_true
        if y_pred is None:
            raise ValueError(
                "ContrastiveRegressionLoss expects both y_true and y_pred."
            )

        # y_true: (batch, 2+) coords, y_pred: (batch, D) latent
        # Ensure float32 for numerical stability (mixed precision)
        dtype = y_pred.dtype
        y_true = kops.cast(y_true, dtype)
        z = y_pred

        # Normalize latents to unit length for cosine similarity
        N = kops.shape(z)[0]
        z = tf.math.l2_normalize(z, axis=1)
        # Similarity matrix
        w = tf.ones((N, N), dtype=dtype)

        # Use target_structure to apply the correct distance logic for each component
        for name, info in self.target_structure.items():
            raw_slice = info.get("slice")
            if isinstance(raw_slice, (tuple, list)):
                start, end = raw_slice
            else:
                start = int(raw_slice)
                end = start + int(info.get("dim", 1))
            val = y_true[:, start:end]

            if name in ["pos_2d", "pos_lin"]:
                # Euclidean Kernel

                # 1. Get positions for weighting (linearized or 2D)
                if self.l_function is not None:
                    _, linearized_pos = self.l_function(val)
                    pos = tf.reshape(linearized_pos, [-1])
                elif kops.shape(val)[1] == 1:
                    pos = tf.reshape(val, [-1])
                else:
                    raise ValueError(
                        "ContrastiveRegressionLoss requires l_function_params to compute position weights. "
                        "Please provide l_function_params for the LinearizationLayer."
                    )

                # 3. Compute pairwise Cosine Similarity Logits
                logits = tf.matmul(z, z, transpose_b=True) / self.temperature

                # 4. Compute Pairwise Spatial Distances
                if len(pos.shape) == 1:
                    pos = tf.reshape(pos, [-1, 1])

                if pos.shape[-1] > 1:
                    r = tf.reduce_sum(tf.square(pos), axis=1, keepdims=True)
                    d2 = r - 2 * tf.matmul(pos, pos, transpose_b=True) + tf.transpose(r)
                    d2 = tf.maximum(d2, self.eps)
                else:
                    d2 = tf.abs(pos - tf.transpose(pos))

                # 5. Distance weighting (soft positives)
                kernel = tf.exp(-0.5 * d2 / (self.sigma**2))
                w *= kops.cast(kernel, dtype)

            elif name == "hd":  # head direction in radians
                # Circular Kernel (Cosine similarity)
                # If val is 1D (radians), use cos(theta1 - theta2)
                cos_sim = tf.cos(val[:, None] - val[None, :])
                # Normalize to [0, 1] range
                w *= kops.cast((cos_sim + 1.0) / 2.0, dtype)
            elif name == "direction":  # bool towards/away from shock
                matches = tf.equal(val[:, 0][:, None], val[:, 0][None, :])
                w *= kops.cast(matches, dtype)

            elif name in ["speed", "thigmo"]:
                # Simple Linear Difference Kernel
                d_lin = tf.abs(val[:, None] - val[None, :])
                w *= kops.cast(kops.exp(-d_lin / self.sigma), dtype)

        mask_diag = tf.eye(N, dtype=dtype)
        w = w * (1.0 - mask_diag)  # Mask self-similarity

        w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + self.eps
        w_norm = w / w_sum

        # 6. Compute Softmax Log-Probabilities masking diagonal
        logits = tf.matmul(z, z, transpose_b=True) / self.temperature
        logits_masked = logits + -1e9 * mask_diag
        log_prob = tf.nn.log_softmax(logits_masked, axis=1)

        # 7. Cross entropy with soft targets
        loss_per_anchor = -tf.reduce_sum(w_norm * log_prob, axis=1)

        return tf.cast(kops.reshape(loss_per_anchor, (-1, 1)), tf.float32)


def _get_loss_function(loss_name, alpha=1.0, delta=1.0, **kwargs):
    """Backward-compatible loss factory used by legacy code paths."""
    name = (loss_name or "").lower()
    if name in {"cyclic_mae", "cyclical_mae", "cyclic"}:
        return CyclicMAE()
    if name == "huber":
        return tf.keras.losses.Huber(delta=delta)
    if name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    if name == "mse":
        return tf.keras.losses.MeanSquaredError()
    raise ValueError(f"Unknown loss function: {loss_name}")

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
                "target_structure": self.target_structure,
            }
        )
        return config


# Register custom layers and losses for Keras serialization
keras_utils.get_custom_objects()["DenseLossProcessor"] = DenseLossProcessor
keras_utils.get_custom_objects()["SpikeNet1D"] = SpikeNet1D
keras_utils.get_custom_objects()["SpikeNet2D"] = SpikeNet2D
keras_utils.get_custom_objects()["SpikeEncoder"] = SpikeEncoder
keras_utils.get_custom_objects()["SpikeSequenceProcessor"] = SpikeSequenceProcessor
keras_utils.get_custom_objects()["MaskedSequential"] = MaskedSequential
keras_utils.get_custom_objects()["GroupAttentionFusion"] = GroupAttentionFusion
keras_utils.get_custom_objects()["GlobalSequenceGather"] = GlobalSequenceGather
keras_utils.get_custom_objects()["MaskedGlobalAveragePooling1D"] = (
    MaskedGlobalAveragePooling1D
)
keras_utils.get_custom_objects()["PositionalEncoding"] = PositionalEncoding
keras_utils.get_custom_objects()["SafeMaskCreation"] = SafeMaskCreation
keras_utils.get_custom_objects()["ResidualWrapper"] = ResidualWrapper
keras_utils.get_custom_objects()["TransformerEncoderBlock"] = TransformerEncoderBlock
keras_utils.get_custom_objects()["DynamicDenseWeightLayer"] = DynamicDenseWeightLayer
keras_utils.get_custom_objects()["UMazeProjectionLayer"] = UMazeProjectionLayer
keras_utils.get_custom_objects()["FeatureOutputWithUMaze"] = FeatureOutputWithUMaze
keras_utils.get_custom_objects()["GaussianHeatmapLayer"] = GaussianHeatmapLayer
keras_utils.get_custom_objects()["GaussianHeatmapLosses"] = GaussianHeatmapLosses
keras_utils.get_custom_objects()["DenseLossProcessor"] = DenseLossProcessor
keras_utils.get_custom_objects()["PositionError2D"] = PositionError2D
keras_utils.get_custom_objects()["AngularErrorMetric"] = AngularErrorMetric
keras_utils.get_custom_objects()["CyclicMAE"] = CyclicMAE
keras_utils.get_custom_objects()["ContrastiveRegressionLoss"] = (
    ContrastiveRegressionLoss
)
keras_utils.get_custom_objects()["LinearizationLayer"] = LinearizationLayer
keras_utils.get_custom_objects()["MemoryUsageCallbackExtended"] = (
    MemoryUsageCallbackExtended
)
keras_utils.get_custom_objects()["ContrastiveMonitor"] = ContrastiveMonitor
keras_utils.get_custom_objects()["scaled_sigmoid"] = scaled_sigmoid
keras_utils.get_custom_objects()["ScaledSigmoid"] = ScaledSigmoid
keras_utils.get_custom_objects()["ContrastiveVisualizer"] = ContrastiveVisualizer
keras_utils.get_custom_objects()["MaskedBatchNormalization"] = MaskedBatchNormalization
