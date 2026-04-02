import pytest
import tensorflow as tf

from neuroencoders.fullEncoder.nnUtils import GroupAttentionFusion, SpikeNet1D


def test_spike_net_1d():
    nChannels = 4
    nFeatures = 64
    batch_size = 8
    seq_len = 10

    layer = SpikeNet1D(nChannels=nChannels, nFeatures=nFeatures)

    # Input shape: (Batch * SeqLen, Channels, Time)
    # With TimeDistributed, the layer sees (Batch, Channels, Time) if wrapped,
    # but SpikeNet1D expects (Batch, Channels, Time) and handles it.

    x = tf.random.normal((batch_size * seq_len, nChannels, 32))
    output = layer(x)

    assert output.shape == (batch_size * seq_len, nFeatures)


def test_group_attention_fusion():
    nGroups = 4
    nFeatures = 64
    batch_size = 8
    seq_len = 10

    layer = GroupAttentionFusion(n_groups=nGroups, embed_dim=nFeatures)

    # Input: list of tensors, each (Batch, SeqLen, Features)
    inputs = [
        tf.random.normal((batch_size, seq_len, nFeatures)) for _ in range(nGroups)
    ]

    # Mask: (Batch, SeqLen, nGroups)
    mask = tf.cast(tf.random.uniform((batch_size, seq_len, nGroups)) > 0.5, tf.float32)

    output = layer(inputs, mask=mask)

    # Output should be (Batch, SeqLen, Groups * Features)
    assert output.shape == (batch_size, seq_len, nGroups * nFeatures)


def test_spike_net_1d_with_time_distributed():
    nChannels = 4
    nFeatures = 64
    batch_size = 8
    seq_len = 10

    layer = tf.keras.layers.TimeDistributed(
        SpikeNet1D(nChannels=nChannels, nFeatures=nFeatures)
    )

    # Input: (Batch, SeqLen, Channels, Time)
    x = tf.random.normal((batch_size, seq_len, nChannels, 32))
    output = layer(x)

    assert output.shape == (batch_size, seq_len, nFeatures)


if __name__ == "__main__":
    pytest.main([__file__])
