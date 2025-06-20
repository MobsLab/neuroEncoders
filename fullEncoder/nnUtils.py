# Load libs
import os
from typing import Dict, Optional, Tuple

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import tensorflow as tf


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
    ):
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        self.device = device
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

    def __call__(self, input):
        return self.apply(input)

    def apply(self, input):
        with tf.device(self.device):
            x = tf.expand_dims(input, axis=3)
            x = self.convLayer1(x)
            x = self.maxPoolLayer1(x)
            x = self.convLayer2(x)
            x = self.maxPoolLayer2(x)
            x = self.convLayer3(x)
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


########### CONVOLUTIONAL NETWORK CLASS #####################


########### TRANSFORMER ENCODER CLASS #####################
class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
    """Global Average Pooling that respects masking"""

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
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
    def __init__(self, max_len=10000, d_model=128):
        super(PositionalEncoding, self).__init__()
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
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self, d_model=128, num_heads=4, ff_dim1=256, ff_dim2=128, dropout_rate=0.5
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim1 = ff_dim1
        self.ff_dim2 = ff_dim2
        self.dropout_rate = dropout_rate

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
        num_augmentations: int = 7,
        white_noise_std: float = 5,
        offset_noise_std: float = 1.6,
        offset_scale_factor: float = 0.67,
        cumulative_noise_std: float = 0.02,
        spike_band_channels: Optional[list] = None,
    ):
        """
        Initialize augmentation parameters.

        Args:
            num_augmentations: Number of augmented copies per trial (4-20 range)
            white_noise_std: Standard deviation for white noise (default: 1.2)
            offset_noise_std: Standard deviation for constant offset (default: 0.6)
            offset_scale_factor: Scale factor for threshold crossings (default: 0.67)
            cumulative_noise_std: Standard deviation for cumulative noise (default: 0.02)
            spike_band_channels: List of spike-band channel indices (if None, assumes all channels)
        """
        self.num_augmentations = num_augmentations
        self.white_noise_std = white_noise_std
        self.offset_noise_std = offset_noise_std
        self.offset_scale_factor = offset_scale_factor
        self.cumulative_noise_std = cumulative_noise_std
        self.spike_band_channels = spike_band_channels

    def add_white_noise(self, neural_data: tf.Tensor) -> tf.Tensor:
        """
        Add white noise to all time points of all channels independently.

        Args:
            neural_data: Tensor of shape [time_steps, channels] or [batch, time_steps, channels]

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
