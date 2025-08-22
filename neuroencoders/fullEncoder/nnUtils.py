# Load libs
import os
from typing import Dict, Optional, Tuple

import numpy as np
from denseweight import DenseWeight

from neuroencoders.utils.global_classes import MAZE_COORDS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import tensorflow as tf
from tensorflow import keras


########### CONVOLUTIONAL NETWORK CLASS #####################
class spikeNet:
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

    One filter of size (2,3) would roughly mean that the first filters "see" half of the channels and 3 time bins,
        i.e. from (3*0.036/32) ~= 3 ms for 36 ms-based windows to ~100 ms for 1.08s-based windows.
    """

    def __init__(
        self,
        nChannels=4,
        device: str = "/cpu:0",
        nFeatures=128,
        number="",
        **kwargs,
    ):
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        self.device = device
        self.batch_normalization = kwargs.get("batch_normalization", True)
        with tf.device(self.device):
            self.convLayer1 = tf.keras.layers.Conv2D(8, [2, 3], padding="SAME")
            self.convLayer2 = tf.keras.layers.Conv2D(16, [2, 3], padding="SAME")
            self.convLayer3 = tf.keras.layers.Conv2D(32, [2, 3], padding="SAME")

            self.maxPoolLayer1 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="SAME"
            )
            self.maxPoolLayer2 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="SAME"
            )
            self.maxPoolLayer3 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="SAME"
            )

            self.dropoutLayer = tf.keras.layers.Dropout(0.5)
            self.denseLayer1 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer2 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer3 = tf.keras.layers.Dense(
                self.nFeatures, activation="relu", name=f"outputCNN{number}"
            )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "nChannels": self.nChannels,
            "device": self.device,
            "nFeatures": self.nFeatures,
            "number": self.number,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        nChannels = config.pop("nChannels", 4)
        device = config.pop("device", "/cpu:0")
        nFeatures = config.pop("nFeatures", 128)
        number = config.pop("number", "")
        return cls(
            nChannels=nChannels,
            device=device,
            nFeatures=nFeatures,
            number=number,
        )

    def __call__(self, input):
        return self.apply(input)

    def apply(self, input):
        with tf.device(self.device):
            x = tf.expand_dims(input, axis=3)
            x = self.convLayer1(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = self.maxPoolLayer1(x)
            x = self.convLayer2(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = self.maxPoolLayer2(x)
            x = self.convLayer3(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = self.maxPoolLayer3(x)

            x = tf.reshape(
                x, [-1, self.nChannels * 8 * 16]
            )  # change from 32 to 16 and 4 to 8
            # by pooling we moved from 32 bins to 4. By convolution we generated 32 channels
            x = self.denseLayer1(x)
            x = self.dropoutLayer(x)
            x = self.denseLayer2(x)
            x = self.denseLayer3(x)
        return x

    def variables(self):
        return (
            self.convLayer1.variables
            + self.convLayer2.variables
            + self.convLayer3.variables
            + self.maxPoolLayer1.variables
            + self.maxPoolLayer2.variables
            + self.maxPoolLayer3.variables
            + self.denseLayer1.variables
            + self.denseLayer2.variables
            + self.denseLayer3.variables
        )

    def layers(self):
        return (
            self.convLayer1,
            self.convLayer2,
            self.convLayer3,
            self.maxPoolLayer1,
            self.maxPoolLayer2,
            self.maxPoolLayer3,
            self.denseLayer1,
            self.denseLayer2,
            self.denseLayer3,
        )


########### CONVOLUTIONAL NETWORK CLASS #####################


########### TRANSFORMER ENCODER CLASS #####################
class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
    """Global Average Pooling that respects masking"""

    def __init__(self, **kwargs):
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        with tf.device(self.device):
            if mask is not None:
                # Convert mask to float and add dimension for broadcasting
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, axis=-1)

                # Apply mask to inputs
                masked_inputs = inputs * mask

                # Calculate sum and count of non-masked elements
                sum_inputs = tf.reduce_sum(masked_inputs, axis=1)
                count_inputs = tf.reduce_sum(mask, axis=1)

                # Avoid division by zero
                count_inputs = tf.maximum(count_inputs, 1.0)

                # Calculate average
                return sum_inputs / count_inputs
            else:
                return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "name": self.__class__.__name__, "device": self.device}

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        return cls(**config)


def create_attention_mask_from_padding_mask(padding_mask):
    """
    Convert padding mask to attention mask for transformer
    Args:
        padding_mask: Boolean mask where True indicates valid positions
    Returns:
        attention_mask: Float mask for attention weights
    """
    if padding_mask is None:
        return None

    # Convert to float (1.0 for valid, 0.0 for padded)
    attention_mask = tf.cast(padding_mask, tf.float32)

    # Create 4D mask for attention: [batch_size, 1, seq_len, seq_len]
    seq_len = tf.shape(attention_mask)[1]

    # Expand to [batch_size, 1, 1, seq_len] for broadcasting
    attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 1)

    # Create mask for key/value positions [batch_size, 1, seq_len, seq_len]
    attention_mask = tf.tile(attention_mask, [1, 1, seq_len, 1])

    return attention_mask


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=10000, d_model=128, **kwargs):
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def build(self, input_shape):
        # Create positional encoding matrix
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        with tf.device(self.device):
            seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

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
        d_model = config.pop("d_model", 128)
        max_len = config.pop("max_len", 10000)
        return cls(
            d_model=d_model, max_len=max_len, device=config.get("device", "/cpu:0")
        )


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model=64,
        num_heads=8,
        ff_dim1=256,
        ff_dim2=64,
        dropout_rate=0.5,
        device="/cpu:0",
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim1 = ff_dim1
        self.ff_dim2 = ff_dim2
        self.dropout_rate = dropout_rate
        self.device = device

        with tf.device(self.device):
            # Layer normalization at the beginning
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            # Multi-head attention
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )

            # Dropout after attention
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

            # Feedforward network
            self.ff_layer1 = tf.keras.layers.Dense(ff_dim1, activation="relu")
            self.ff_layer2 = tf.keras.layers.Dense(ff_dim2, activation="relu")

            # Final layer normalization
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask=None, training=False):
        with tf.device(self.device):
            # Layer norm at the beginning
            x_norm = self.norm1(x)

            # create attention mask if needed
            attention_mask = None
            if mask is not None:
                attention_mask = create_attention_mask_from_padding_mask(mask)

            # Multi-head attention with residual connection
            attn_output = self.mha(
                x_norm, x_norm, attention_mask=attention_mask, training=training
            )
            attn_output = self.dropout1(attn_output, training=training)
            x = x + attn_output  # Residual connection

            # Feedforward network
            ff_output = self.ff_layer1(x)
            ff_output = self.ff_layer2(ff_output)

            # Final layer norm and residual connection
            x = self.norm2(x + ff_output)

        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "norm1": tf.keras.saving.serialize_keras_object(self.norm1),
            "mha": tf.keras.saving.serialize_keras_object(self.mha),
            "dropout1": tf.keras.saving.serialize_keras_object(self.dropout1),
            "ff_layer1": tf.keras.saving.serialize_keras_object(self.ff_layer1),
            "ff_layer2": tf.keras.saving.serialize_keras_object(self.ff_layer2),
            "norm2": tf.keras.saving.serialize_keras_object(self.norm2),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim1": self.ff_dim1,
            "ff_dim2": self.ff_dim2,
            "dropout_rate": self.dropout_rate,
            "device": self.device,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        # The deserialized layers are not used; just pass config values to the constructor.
        return cls(
            d_model=config.get("d_model", 64),
            num_heads=config.get("num_heads", 8),
            ff_dim1=config.get("ff_dim1", 256),
            ff_dim2=config.get("d_model", 64),
            dropout_rate=config.get("dropout_rate", 0.5),
            device=config.get("device", "/cpu:0"),
        )


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


# @tf.function
def parse_serialized_sequence(params, tensors, batched=False):  # featDesc, ex_proto,
    tensors["groups"] = tf.sparse.to_dense(tensors["groups"], default_value=-1)
    # Pierre 13/02/2021: Why use sparse.to_dense, and not directly a FixedLenFeature?
    # Probably because he wanted a variable length <> inputs sequences
    tensors["groups"] = tf.reshape(tensors["groups"], [-1])

    tensors["indexInDat"] = tf.sparse.to_dense(tensors["indexInDat"], default_value=-1)
    tensors["indexInDat"] = tf.reshape(tensors["indexInDat"], [-1])

    for g in range(params.nGroups):
        # here 32 correspond to the number of discretized time bin for a spike
        zeros = tf.constant(np.zeros([params.nChannelsPerGroup[g], 32]), tf.float32)
        tensors["group" + str(g)] = tf.sparse.reshape(tensors["group" + str(g)], [-1])
        tensors["group" + str(g)] = tf.sparse.to_dense(tensors["group" + str(g)])
        tensors["group" + str(g)] = tf.reshape(tensors["group" + str(g)], [-1])
        if batched:
            tensors["group" + str(g)] = tf.reshape(
                tensors["group" + str(g)],
                [params.batchSize, -1, params.nChannelsPerGroup[g], 32],
            )
        # even if batched: gather all together
        tensors["group" + str(g)] = tf.reshape(
            tensors["group" + str(g)], [-1, params.nChannelsPerGroup[g], 32]
        )
        # Pierre 12/03/2021: the batchSize and timesteps are gathered together
        nonZeros = tf.logical_not(
            tf.equal(
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        tf.equal(tensors["group" + str(g)], zeros), tf.int32
                    ),
                    axis=[1, 2],
                ),
                32 * params.nChannelsPerGroup[g],
            )
        )
        # nonZeros: control that the voltage measured is not 0, at all channels and time bin inside the detected spike
        tensors["group" + str(g)] = tf.gather(
            tensors["group" + str(g)], tf.where(nonZeros)
        )[:, 0, :, :]
        # I don't understand why it can then call [:,0,:,:] as the output tensor of gather should have the same
        # shape as tensors["group"+str(g)"], [-1,params.nChannels[g],32] ...

    return tensors


def parse_serialized_spike(featDesc, ex_proto, batched=False):
    if batched:
        tensors = tf.io.parse_example(serialized=ex_proto, features=featDesc)
    else:
        tensors = tf.io.parse_single_example(serialized=ex_proto, features=featDesc)
    return tensors


########### SPIKE STORAGE AND PARCING FUNCTIONS #####################


def import_true_pos(feature):
    def change_feature(vals):
        vals["pos"] = tf.gather(feature, vals["pos_index"])
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
            num_augmentations: Number of augmented copies per trial (4-20 range)
            white_noise_std: Standard deviation for white noise (default: 1.2)
            offset_noise_std: Standard deviation for constant offset (default: 0.6)
            offset_scale_factor: Scale factor for threshold crossings (default: 0.67)
            cumulative_noise_std: Standard deviation for cumulative noise (default: 0.02)
            spike_band_channels: List of spike-band channel indices (if None, assumes all channels)
        """
        self.num_augmentations = kwargs.get("num_augmentations", 11)
        self.white_noise_std = kwargs.get("white_noise_std", 5.0)
        self.offset_noise_std = kwargs.get("offset_noise_std", 1.6)
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
            neural_data: Tensor of shape [time_steps, channels] or [batch, time_steps, channels]

        Returns:
            Augmented neural data with white noise
        """
        with tf.device(self.device):
            noise = tf.random.normal(
                shape=tf.shape(neural_data),
                mean=0.0,
                stddev=self.white_noise_std,
                dtype=neural_data.dtype,
            )
        return neural_data + noise

    def add_constant_offset(self, neural_data: tf.Tensor) -> tf.Tensor:
        """
        Add constant offset to spike-band channels and scaled version to threshold crossings.

        Args:
            neural_data: Tensor of shape [time_steps, channels] or [batch, time_steps, channels]
            threshold_crossings: Optional threshold crossing data for scaled offset

        Returns:
            Augmented neural data with constant offset
        """
        # Generate constant offset for each channel
        with tf.device(self.device):
            if len(neural_data.shape) == 3:  # Batched data
                batch_size = tf.shape(neural_data)[0]
                num_channels = tf.shape(neural_data)[2]
                offset_shape = [batch_size, 1, num_channels]
            else:  # Single sample
                num_channels = tf.shape(neural_data)[1]
                offset_shape = [1, num_channels]

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

    def add_cumulative_noise(self, neural_data: tf.Tensor) -> tf.Tensor:
        """
        Add cumulative (random walk) noise to all channels along the time course.

        Args:
            neural_data: Tensor of shape [time_steps, channels] or [batch, time_steps, channels]

        Returns:
            Augmented neural data with cumulative noise
        """
        with tf.device(self.device):
            if len(neural_data.shape) == 3:  # Batched data
                batch_size = tf.shape(neural_data)[0]
                time_steps = tf.shape(neural_data)[1]
                num_channels = tf.shape(neural_data)[2]
                noise_shape = [batch_size, time_steps, num_channels]
            else:  # Single sample
                time_steps = tf.shape(neural_data)[0]
                num_channels = tf.shape(neural_data)[1]
                noise_shape = [time_steps, num_channels]

            # Generate random noise for each time step
            noise_increments = tf.random.normal(
                shape=noise_shape,
                mean=0.0,
                stddev=self.cumulative_noise_std,
                dtype=neural_data.dtype,
            )

            # Compute cumulative sum along time axis to create random walk
            axis = 1 if len(neural_data.shape) == 3 else 0
            cumulative_noise = tf.cumsum(noise_increments, axis=axis)

        return neural_data + cumulative_noise

    def augment_sample(
        self, neural_data: tf.Tensor, threshold_crossings: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Apply all augmentation strategies to a single sample.

        Args:
            neural_data: Neural features tensor
            threshold_crossings: Optional threshold crossings tensor

        Returns:
            Augmented neural data and threshold crossings (if provided)
        """
        # Apply white noise
        augmented_data = self.add_white_noise(neural_data)

        # Apply constant offset
        augmented_data = self.add_constant_offset(augmented_data)

        # Apply cumulative noise
        augmented_data = self.add_cumulative_noise(augmented_data)

        return augmented_data

    def create_augmented_copies(
        self,
        neural_data: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Create multiple augmented copies of a single trial.

        Args:
            neural_data: Neural features tensor
            threshold_crossings: Optional threshold crossings tensor
            labels: Optional labels tensor

        Returns:
            Dictionary containing stacked augmented data
        """
        augmented_samples = []

        for _ in range(self.num_augmentations):
            aug_data = self.augment_sample(neural_data)
            augmented_samples.append(aug_data)

        # Stack all augmented samples
        result = {"neural_data": tf.stack(augmented_samples, axis=0)}

        return result


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
    labels = parsed_features.get("labels", None)

    # Apply augmentation
    augmented_data = augmentation_config.create_augmented_copies(neural_data, labels)

    return augmented_data


def parse_serialized_sequence_with_augmentation(
    params,
    tensors,
    augmentation_config: Optional[NeuralDataAugmentation] = None,
    batched=False,
):
    """
    Parse serialized sequence with optional data augmentation.

    Args:
        params: Parameters object containing nGroups, nChannelsPerGroup, etc.
        tensors: Dictionary of parsed tensors from TFRecord
        augmentation_config: Optional augmentation configuration
        batched: Whether data is batched

    Returns:
        Dictionary of parsed and optionally augmented tensors
    """
    # Original parsing logic
    tensors["groups"] = tf.sparse.to_dense(tensors["groups"], default_value=-1)
    tensors["groups"] = tf.reshape(tensors["groups"], [-1])
    tensors["indexInDat"] = tf.sparse.to_dense(tensors["indexInDat"], default_value=-1)
    tensors["indexInDat"] = tf.reshape(tensors["indexInDat"], [-1])

    # Store original group data for potential augmentation
    original_groups = {}

    for g in range(params.nGroups):
        # Original processing logic
        zeros = tf.constant(np.zeros([params.nChannelsPerGroup[g], 32]), tf.float32)
        tensors["group" + str(g)] = tf.sparse.reshape(tensors["group" + str(g)], [-1])
        tensors["group" + str(g)] = tf.sparse.to_dense(tensors["group" + str(g)])
        tensors["group" + str(g)] = tf.reshape(tensors["group" + str(g)], [-1])

        if batched:
            tensors["group" + str(g)] = tf.reshape(
                tensors["group" + str(g)],
                [params.batchSize, -1, params.nChannelsPerGroup[g], 32],
            )

        # Reshape for processing
        tensors["group" + str(g)] = tf.reshape(
            tensors["group" + str(g)], [-1, params.nChannelsPerGroup[g], 32]
        )

        # Filter non-zero entries
        nonZeros = tf.logical_not(
            tf.equal(
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        tf.equal(tensors["group" + str(g)], zeros), tf.int32
                    ),
                    axis=[1, 2],
                ),
                32 * params.nChannelsPerGroup[g],
            )
        )

        tensors["group" + str(g)] = tf.gather(
            tensors["group" + str(g)], tf.where(nonZeros)
        )[:, 0, :, :]

        # Store original data for augmentation
        original_groups[f"group{g}"] = tensors["group" + str(g)]

    # Apply augmentation if configured
    if augmentation_config is not None:
        augmented_tensors = apply_group_augmentation(
            tensors, original_groups, params, augmentation_config
        )
        return augmented_tensors

    return tensors


def apply_group_augmentation(
    tensors: Dict[str, tf.Tensor],
    original_groups: Dict[str, tf.Tensor],
    params,
    augmentation_config: NeuralDataAugmentation,
) -> Dict[str, tf.Tensor]:
    """
    Apply augmentation to group-based neural data.

    Args:
        tensors: Original parsed tensors
        original_groups: Dictionary of group data tensors
        params: Parameters object
        augmentation_config: Augmentation configuration

    Returns:
        Dictionary with augmented data
    """
    augmented_copies = []

    for aug_idx in range(augmentation_config.num_augmentations):
        aug_tensors = tensors.copy()

        for g in range(params.nGroups):
            group_key = f"group{g}"
            if group_key in original_groups:
                # Get original group data: [num_spikes, channels, time_bins]
                group_data = original_groups[group_key]

                # Apply augmentation to each spike in the group
                augmented_group = augment_spike_group(group_data, augmentation_config)

                # Update the augmented tensor
                aug_tensors["group" + str(g)] = augmented_group

        # Replicate metadata for each augmentation
        aug_tensors["groups"] = tensors["groups"]
        aug_tensors["pos"] = tensors["pos"]
        aug_tensors["indexInDat"] = tensors["indexInDat"]

        augmented_copies.append(aug_tensors)

    # Stack augmented copies
    result_tensors = {}

    # Stack group data
    for g in range(params.nGroups):
        group_key = "group" + str(g)
        if group_key in tensors:
            stacked_groups = tf.stack(
                [aug[group_key] for aug in augmented_copies], axis=0
            )
            result_tensors[group_key] = stacked_groups

    # Replicate metadata
    result_tensors["groups"] = tf.repeat(
        tf.expand_dims(tensors["groups"], 0),
        augmentation_config.num_augmentations,
        axis=0,
    )
    result_tensors["indexInDat"] = tf.repeat(
        tf.expand_dims(tensors["indexInDat"], 0),
        augmentation_config.num_augmentations,
        axis=0,
    )
    result_tensors["pos"] = tf.repeat(
        tf.expand_dims(tensors["pos"], 0),
        augmentation_config.num_augmentations,
        axis=0,
    )

    return result_tensors


def augment_spike_group(
    group_data: tf.Tensor, augmentation_config: NeuralDataAugmentation
) -> tf.Tensor:
    """
    Apply augmentation to a single spike group.

    Args:
        group_data: Tensor of shape [num_spikes, channels, time_bins]
        augmentation_config: Augmentation configuration

    Returns:
        Augmented group data
    """
    # Apply white noise to all time points and channels
    augmented_data = augmentation_config.add_white_noise(group_data)

    # Apply constant offset per channel
    num_spikes = tf.shape(group_data)[0]
    num_channels = tf.shape(group_data)[1]

    # Generate offset for each spike and channel
    offset = tf.random.normal(
        shape=[num_spikes, num_channels, 1],  # Broadcast across time bins
        mean=0.0,
        stddev=augmentation_config.offset_noise_std,
        dtype=group_data.dtype,
    )

    augmented_data = augmented_data + offset

    # Apply cumulative noise along time dimension (axis=2)
    time_bins = tf.shape(group_data)[2]
    noise_increments = tf.random.normal(
        shape=[num_spikes, num_channels, time_bins],
        mean=0.0,
        stddev=augmentation_config.cumulative_noise_std,
        dtype=group_data.dtype,
    )

    cumulative_noise = tf.cumsum(noise_increments, axis=2)
    augmented_data = augmented_data + cumulative_noise

    return augmented_data


def create_flatten_augmented_groups_fn(params, num_augmentations):
    """
    Create a function for flattening augmented groups that can be used with flat_map.
    This is needed for tf.experimental graph compatibility (no lambda functions).

    Args:
        params: Parameters object
        num_augmentations: Number of augmentations per original sample

    Returns:
        Function that can be used with dataset.flat_map()
    """

    def map_flatten_augmented_groups(data_dict):
        """
        Flatten augmented groups to create individual samples from each augmentation.

        Args:
            data_dict: Dictionary with augmented tensors

        Returns:
            tf.data.Dataset with flattened individual samples
        """
        return flatten_augmented_groups(data_dict, params, num_augmentations)

    return map_flatten_augmented_groups


def flatten_augmented_groups(data_dict, params, num_augmentations):
    """
    Flatten augmented groups to create individual samples from each augmentation.

    Args:
        data_dict: Dictionary with augmented tensors
        params: Parameters object
        num_augmentations: Number of augmentations per original sample

    Returns:
        tf.data.Dataset with flattened individual samples
    """
    # Create a dictionary where each key maps to a tensor containing all augmented samples
    flattened_dict = {}

    # Handle group data
    for g in range(params.nGroups):
        group_key = f"group{g}"
        if group_key in data_dict:
            # data_dict[group_key] has shape [num_augmentations, num_spikes, channels, time_bins]
            # We want to flatten the first dimension
            flattened_dict[group_key] = data_dict[group_key]

    # Handle metadata (already stacked with shape [num_augmentations, ...])
    flattened_dict["groups"] = data_dict["groups"]
    flattened_dict["indexInDat"] = data_dict["indexInDat"]
    flattened_dict["pos"] = data_dict["pos"]

    return tf.data.Dataset.from_tensor_slices(flattened_dict)


class LinearizationLayer(tf.keras.layers.Layer):
    """
    A simple layer to linearize Euclidean data into a maze-like structure.
    Follows the same logic as the linearizer pykeops code.
    """

    def __init__(self, maze_points, ts_proj, **kwargs):
        self.device = kwargs.pop("device", "/cpu:0")
        super().__init__(**kwargs)
        # Convert to TensorFlow constants
        self.maze_points = tf.constant(maze_points, dtype=tf.float32)
        self.ts_proj = tf.constant(ts_proj, dtype=tf.float32)

    def call(self, euclidean_data):
        with tf.device(self.device):
            # Ensure consistent dtype
            euclidean_data = tf.cast(euclidean_data, self.maze_points.dtype)

            # Expand dimensions for broadcasting
            # euclidean_data: [batch_size, features] -> [batch_size, 1, features]
            # maze_points: [num_points, features] -> [1, num_points, features]
            euclidean_expanded = tf.expand_dims(euclidean_data, axis=1)
            maze_expanded = tf.expand_dims(self.maze_points, axis=0)

            # Calculate squared distances
            distance_matrix = tf.reduce_sum(
                tf.square(maze_expanded - euclidean_expanded), axis=-1
            )

            # Find argmin
            best_points = tf.argmin(distance_matrix, axis=1)

            # Gather results
            projected_pos = tf.gather(self.maze_points, best_points)
            linear_pos = tf.gather(self.ts_proj, best_points)

        return projected_pos, linear_pos

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "maze_points": self.maze_points.numpy().tolist(),
            "ts_proj": self.ts_proj.numpy().tolist(),
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        maze_points = tf.constant(config.pop("maze_points"), dtype=tf.float32)
        ts_proj = tf.constant(config.pop("ts_proj"), dtype=tf.float32)
        return cls(
            maze_points=maze_points,
            ts_proj=ts_proj,
            device=config.pop("device", "/cpu:0"),
        )


@keras.saving.register_keras_serializable(
    package="Custom_Layers", name="DynamicDenseWeightLayer"
)
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
        with tf.device(self.device):
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
        with tf.device(self.device):
            # Use tf.py_function to call the fitted DenseWeight
            weights = tf.py_function(
                func=self._compute_batch_weights, inp=[linearized_pos], Tout=tf.float32
            )

            # Set shape (tf.py_function loses shape info)
            batch_size = tf.shape(linearized_pos)[0]
            weights.set_shape([None])

        return weights

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "fitted_dw_alpha": self.alpha,
            "training_data": self.training_data,
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create a new instance of the layer from its config.
        This is necessary for serialization/deserialization.
        """
        fitted_dw_config = config.pop("fitted_dw_alpha")
        training_data = config.pop("training_data", None)
        fitted_dw = DenseWeight(fitted_dw_config)
        if training_data is not None:
            fitted_dw.fit(training_data)
        return cls(fitted_denseweight=fitted_dw, device=config.pop("device", "/cpu:0"))


class UMazeProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, smoothing_factor=0.01, maze_params=None, **kwargs):
        """
        Differentiable projection layer that softly constrains (x,y) predictions
        to lie within a U-shaped maze.

        Args:
            maze_params (dict): Defines maze geometry.
            smoothing_factor (float): Controls softness of constraints.
        """
        super().__init__(**kwargs)

        # Default parameters based on your maze image
        if maze_params is None or not isinstance(maze_params, dict):
            if maze_params is not None:
                maze_coords = np.array(maze_params)
            else:
                maze_coords = MAZE_COORDS
            maze_params = {
                "x_min": maze_coords[:, 0].min(),
                "x_max": maze_coords[:, 0].max(),
                "y_min": maze_coords[:, 1].min(),
                "y_max": maze_coords[:, 1].max(),
                "gap_x_min": maze_coords[-2, 0],
                "gap_x_max": maze_coords[-4, 0],
                "gap_y_min": maze_coords[-3, 1],
            }
        self.maze_params = maze_params
        self.smoothing_factor = smoothing_factor
        print(f"UMazeProjectionLayer initialized with params: {self.maze_params}")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x, y = inputs[..., 0], inputs[..., 1]
        x_proj, y_proj = self._project_points(x, y)
        return tf.stack([x_proj, y_proj], axis=1)

    def _project_points(self, x, y):
        dtype = x.dtype
        gap_x_min = tf.constant(self.maze_params["gap_x_min"], dtype=dtype)
        gap_x_max = tf.constant(self.maze_params["gap_x_max"], dtype=dtype)
        gap_y_min = tf.constant(self.maze_params["gap_y_min"], dtype=dtype)

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
            x_final, self.maze_params["x_min"], self.maze_params["x_max"]
        )
        y_final = tf.clip_by_value(
            y_final, self.maze_params["y_min"], self.maze_params["y_max"]
        )

        return x_final, y_final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "maze_params": self.maze_params,
                "smoothing_factor": self.smoothing_factor,
            }
        )
        return config


# Test the projection layer
def test_projection_layer(
    test_predictions=None, maze_params=None, smoothing_factor=0.01
):
    """Test the projection layer with sample data"""

    # Create some test predictions (some in valid regions, some in gap)
    if test_predictions is None:
        test_predictions = tf.constant(
            [
                [0, 0],
                [1, 1],
                [0.2, 0.5],  # Valid - left arm
                [0.8, 0.3],  # Valid - right arm
                [0.65, 0.9],
                [0.75, 0.9],
                [0.5, 0.8],  # Invalid - in gap, should be projected
                [0.5, 0.5],  # Invalid - in gap, should be projected
                [0.1, 0.5],  # Invalid - in gap, should be projected
                [0.45, 0.9],  # Invalid - in gap, should snap to left
                [0.55, 0.7],  # Invalid - in gap, should snap to right
            ]
        )
    else:
        test_predictions = tf.constant(test_predictions, dtype=tf.float32)

    # Create and test the layer
    projection_layer = UMazeProjectionLayer(
        smoothing_factor=smoothing_factor, maze_params=maze_params
    )
    projected = projection_layer(test_predictions)
    error = tf.sqrt(tf.reduce_sum((test_predictions - projected) ** 2, axis=1))
    mean_error = tf.reduce_mean(error)

    print(f"Mean Error on {test_predictions.shape[0]} predictions: {mean_error:.4f}")

    # plot the result
    import matplotlib.pyplot as plt

    if maze_params is None:
        maze_params = MAZE_COORDS
        print("Using default maze coordinates for plotting")

    plt.figure()
    plt.plot(maze_params[:, 0], maze_params[:, 1], "k-", label="Maze Path")
    # plot by matching each point in test_predictions to its projected point
    plt.scatter(
        test_predictions[:, 0],
        test_predictions[:, 1],
        c="blue",
        label="Original Predictions",
        alpha=0.5,
    )
    plt.scatter(
        projected[:, 0],
        projected[:, 1],
        c="red",
        label="Projected Predictions",
        alpha=0.5,
    )
    # now plot lines between pred and projected points
    for i in range(max(len(test_predictions), 1000)):
        plt.plot(
            [test_predictions[i, 0], projected[i, 0]],
            [test_predictions[i, 1], projected[i, 1]],
            "r--",
            alpha=0.3,
        )

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.title("UMaze Projection Layer Test")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

    return projected


# Custom layer that combines feature_output and UMazeProjectionLayer
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
                "maze_params": self.proj.maze_params,
            }
        )
        return config

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

        with tf.device(self.device):
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
                print(f"✓ DenseWeight model fitted on {len(training_pos_np)} samples")
                print(f"✓ Ready for dynamic weight computation during training")

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
        maze_points = tf.constant(config.pop("maze_points"), dtype=tf.float32)
        ts_proj = tf.constant(config.pop("ts_proj"), dtype=tf.float32)
        alpha = config.pop("alpha", 1.0)

        fitted_dw_config = config.pop("fitted_dw", None)
        fitted_dw = fitted_dw_config if fitted_dw_config else None

        processor = cls(maze_points=maze_points, ts_proj=ts_proj, alpha=alpha)
        processor.fitted_dw = fitted_dw
        processor.verbose = config.pop("verbose", False)
        processor.device = config.pop("device", "/cpu:0")
        return processor
