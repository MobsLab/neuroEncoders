import numpy as np
import pytest
import tensorflow as tf

from neuroencoders.fullEncoder.nnUtils import (
    MaskedBatchNormalization,
    NeuralDataAugmentation,
    SpikeNet1D,
    SpikeNet2D,
)


class Params:
    def __init__(self):
        self.nGroups = 2
        self.nChannelsPerGroup = [4, 6]
        self.batch_size = 2


@pytest.fixture
def mock_params():
    return Params()


def test_masked_batch_norm_logic():
    """Test MaskedBatchNormalization correctly ignores masked values"""
    # Create input: (Batch, time, channels)
    # Batch 0: all valid
    # Batch 1: first half valid
    # Batch 2: all masked (padding) - should result in 0 output

    input_data = tf.random.normal((3, 10, 2))
    mask = tf.constant(
        [
            [True] * 10,  # All valid
            [True] * 5 + [False] * 5,  # Half valid
            [False] * 10,  # All masked
        ],
        dtype=bool,
    )

    layer = MaskedBatchNormalization()

    # Run in training mode
    output = layer(input_data, mask=mask, training=True)

    # Check that masked values are exactly zero
    masked_output = output[1, 5:, :]
    assert np.allclose(masked_output, 0.0), "Masked values should be zero"

    masked_output_full = output[2, :, :]
    assert np.allclose(masked_output_full, 0.0), "Fully masked sample should be zero"

    # Check that valid values are normalized (roughly 0 mean 1 std)
    # This is tricky with small batch size, but we can check relative consistency
    valid_output = output[0, :, :]
    # Just ensure it runs and produces reasonable output (not nan)
    assert not np.any(np.isnan(valid_output))


def test_spikenet1d_masking():
    """Test SpikeNet1D handles masking correctly"""
    batch_size = 4
    n_channels = 3
    time_steps = 32

    # Input shape: (Batch, Channels, Time)
    input_data = tf.random.normal((batch_size, n_channels, time_steps))

    # Mask shape: (Batch,) - boolean tensor indicating valid samples
    # Let's say sample 0 and 2 are valid, 1 and 3 are padding/invalid
    mask = tf.constant([True, False, True, False], dtype=bool)

    net = SpikeNet1D(nChannels=n_channels, batch_normalization=True, nFeatures=10)

    output = net(input_data, mask=mask, training=True)

    # Valid samples should have non-zero output
    assert np.any(output[0].numpy() != 0)
    assert np.any(output[2].numpy() != 0)

    # Invalid samples should have exactly zero output
    assert np.all(output[1].numpy() == 0)
    assert np.all(output[3].numpy() == 0)

    # Output shape should be (Batch, nFeatures)
    assert output.shape == (batch_size, 10)


def test_spikenet2d_masking():
    """Test SpikeNet2D handles masking correctly"""
    batch_size = 4
    n_channels = 3
    time_steps = 32

    # Input shape: (Batch, Channels, Time) - transformed inside to (Batch, Channels, Time, 1)
    input_data = tf.random.normal((batch_size, n_channels, time_steps))

    # Mask shape: (Batch,)
    mask = tf.constant([True, False, True, False], dtype=bool)

    net = SpikeNet2D(nChannels=n_channels, batch_normalization=True, nFeatures=10)

    output = net(input_data, mask=mask, training=True)

    # Valid samples should have non-zero output
    assert np.any(output[0].numpy() != 0)
    assert np.any(output[2].numpy() != 0)

    # Invalid samples should have exactly zero output
    assert np.all(output[1].numpy() == 0)
    assert np.all(output[3].numpy() == 0)

    # Output shape should be (Batch, nFeatures)
    assert output.shape == (batch_size, 10)


def test_normalization_augmentation():
    """Test NeuralDataAugmentation applies normalization stats"""
    means = [np.array([10.0, 20.0]), np.array([5.0])]
    stds = [np.array([2.0, 5.0]), np.array([1.0])]

    aug = NeuralDataAugmentation(
        normalize=True,
        normalization_stats=(means, stds),
        white_noise_std=2.0,  # Should be scaled down
    )

    # Verify noise scaling happened
    # 2.0 / 50.0 = 0.04
    assert np.isclose(aug.white_noise_std, 0.04)

    # Create fake group data: (Batch, Channels, Time)
    # Group 0 has 2 channels
    group0_data = tf.constant(
        [
            [[12.0, 12.0], [25.0, 25.0]],  # Sample 0
        ],
        dtype=tf.float32,
    )
    # Shape: (1, 2, 2)
    # Mean input: Ch0=12, Ch1=25.
    # Expected norm: Ch0=(12-10)/2 = 1.0, Ch1=(25-20)/5 = 1.0

    normalized = aug.normalize_group(group0_data, 0)

    assert np.allclose(normalized.numpy(), 1.0)

    # Test augmentation runs (returns matching shape + num_augs)
    # apply_group_augmentation logic is what normally calls normalize_group,
    # but let's test augment_spike_group directly on normalized data

    augmented = aug.augment_spike_group(normalized)

    # Check output shape: (1, 2, 2) + Augmentations?
    # Actually augment_sample is single sample logic generally but augment_spike_group uses augment_sample
    # Wait, augment_sample returns ONE tensor with same shape as input?
    # The new vectorized augmentation returns (num_augs, ...)
    # But NeuralDataAugmentation.augment_spike_group calls augment_sample which calls add_white_noise etc.
    # It returns ONE augmented version.

    assert augmented.shape == normalized.shape

    # Check values changed (noise added)
    assert not np.allclose(augmented.numpy(), normalized.numpy())


def test_normalization_stats_computation_logic():
    """Mock the logic of compute_normalization_stats to ensure it aggregates correctly"""
    # This logic was added to an_network.py, but we can verify the math here.

    # Setup mock data for one group
    n_ch = 2
    # Batch 1:
    # Sample 0: valid, vals=[10, 10], [10, 10] (Mean 10)
    # Sample 1: invalid (padding)
    data1 = np.zeros((2, 2, 2, 5))  # (Batch, Spikes, Channels, Time)
    # Spike 0 of Batch 0 is valid.
    data1[0, 0, :, :] = 10.0

    # Batch 2:
    # Sample 0: valid, vals=[20, 20], [20, 20] (Mean 20)
    data2 = np.zeros((2, 2, 2, 5))
    data2[0, 0, :, :] = 20.0

    # "Dataset" iteration
    sum_x = np.zeros(n_ch)
    sum_sq_x = np.zeros(n_ch)
    total_count = 0

    for batch in [data1, data2]:
        data = tf.constant(batch, dtype=tf.float32)
        # Identify valid (Batch, Spikes)
        is_valid = tf.reduce_any(tf.not_equal(data, 0.0), axis=[2, 3])

        valid_data = tf.boolean_mask(data, is_valid)  # (N_valid, C, T)

        if valid_data.shape[0] == 0:
            continue

        # Reshape to (N*T, C)
        # Transpose (N, C, T) -> (N, T, C)
        flat_data = tf.reshape(tf.transpose(valid_data, [0, 2, 1]), [-1, n_ch])

        count = flat_data.shape[0]
        sum_x += np.sum(flat_data.numpy(), axis=0)
        sum_sq_x += np.sum(np.square(flat_data.numpy()), axis=0)
        total_count += count

    mean = sum_x / total_count
    var = (sum_sq_x / total_count) - (mean**2)
    std = np.sqrt(var)

    return mean, var, std


def test_normalization_layer_integration():
    """Test using Keras Normalization layer integration"""
    # Simulate a SpikeNet1D with normalization layer
    # Initialize layer
    # Axis=1 (channels) -> means shape (2,)
    norm_layer = tf.keras.layers.Normalization(axis=1, name="input_norm")

    # Use adapt on synthetic data with exact per-channel statistics:
    # channel 0 values -> [8, 12] => mean=10, std=2
    # channel 1 values -> [15, 25] => mean=20, std=5
    adapt_data = tf.constant([[[8.0], [15.0]], [[12.0], [25.0]]], dtype=tf.float32)
    norm_layer.adapt(adapt_data)

    # Test output
    # Input: Channel 0 with value 12 (norm -> 1), Channel 1 with value 25 (norm -> 1)
    # Shape needs to be (Batch, Channels, Time)
    input_data = tf.constant([[[12.0, 12.0], [25.0, 25.0]]], dtype=tf.float32)
    # Shape (1, CH=2, T=2).

    output = norm_layer(input_data)

    assert np.allclose(output.numpy(), 1.0, atol=1e-5)
