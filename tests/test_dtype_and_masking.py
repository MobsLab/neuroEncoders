import numpy as np
import pytest
import tensorflow as tf
from keras import ops as kops
from keras.layers import Lambda

from neuroencoders.fullEncoder.nnUtils import GaussianHeatmapLayer, PositionalEncoding


def test_gaussian_heatmap_dtype_mismatch():
    # Setup mixed precision policy for bfloat16
    policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
    tf.keras.mixed_precision.set_global_policy(policy)

    # Mock data
    grid_size = (10, 10)
    training_positions = np.random.rand(100, 2).astype(np.float32)
    layer = GaussianHeatmapLayer(
        training_positions=training_positions, grid_size=grid_size
    )

    # Input with bfloat16
    inputs = tf.random.normal((1, 128), dtype=tf.bfloat16)

    # This should not raise TypeError
    try:
        output = layer(inputs)
        assert output.dtype == tf.bfloat16
    except TypeError as e:
        pytest.fail(f"GaussianHeatmapLayer failed with dtype mismatch: {e}")
    finally:
        # Reset policy
        tf.keras.mixed_precision.set_global_policy("float32")


def test_positional_encoding_masking():
    layer = PositionalEncoding(d_model=128)
    assert layer.supports_masking is True


def test_lambda_masking():
    # Example of Lambda layer used in an_network.py
    mymask = tf.constant([[True, False], [True, True]])
    allFeatures = tf.random.normal((2, 2, 128))

    # Wrap in a layer that supports masking
    lambda_layer = Lambda(
        lambda t: kops.where(
            kops.expand_dims(t[0], axis=-1), t[1], kops.zeros_like(t[1])
        )
    )
    lambda_layer.supports_masking = True

    # This should pass without warning about destroying mask if implemented correctly
    output = lambda_layer([mymask, allFeatures])
    assert output.shape == allFeatures.shape
