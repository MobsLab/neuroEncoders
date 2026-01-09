import pytest
import numpy as np
import tensorflow as tf
from neuroencoders.fullEncoder.nnUtils import (
    _get_loss_function,
    ContrastiveLossLayer,
    MultiColumnLossLayer,
)


def test_get_loss_function_standard():
    # Test standard Keras losses via the helper
    y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y_pred = tf.constant([[1.1, 1.9], [3.2, 3.8]])

    # MSE
    loss_fn = _get_loss_function("mse", alpha=1.0, delta=1.0)
    loss = loss_fn(y_true, y_pred)
    assert loss.shape == (2,)
    # (0.1^2 + 0.1^2) / 2 = 0.01
    assert np.allclose(loss[0], 0.01)

    # MAE
    loss_fn = _get_loss_function("mae", alpha=1.0, delta=1.0)
    loss = loss_fn(y_true, y_pred)
    assert loss.shape == (2,)
    # (0.1 + 0.1) / 2 = 0.1
    assert np.allclose(loss[0], 0.1)

    # Huber
    loss_fn = _get_loss_function("huber", alpha=1.0, delta=1.0)
    loss = loss_fn(y_true, y_pred)
    assert loss.shape == (2,)


def test_get_loss_function_combined():
    # Test mse_plus_msle
    alpha = 2.0
    y_true = tf.constant([[1.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.1]], dtype=tf.float32)

    loss_fn = _get_loss_function("mse_plus_msle", alpha=alpha, delta=1.0)
    loss = loss_fn(y_true, y_pred)

    mse = tf.keras.losses.MeanSquaredError(reduction="none")(y_true, y_pred)
    msle = tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")(y_true, y_pred)
    expected = mse + alpha * msle

    assert np.allclose(loss, expected)


def test_get_loss_function_cyclic():
    # Test cyclic_mae (radians)
    y_true = tf.constant([[0.1]], dtype=tf.float32)
    # distance across the 2pi boundary
    y_pred = tf.constant([[2 * np.pi - 0.1]], dtype=tf.float32)

    loss_fn = _get_loss_function("cyclic_mae", alpha=1.0, delta=1.0)
    loss = loss_fn(y_true, y_pred)

    # |0.1 - (2pi - 0.1)| = 2pi - 0.2
    # |0.1 - (2pi - 0.1) + 2pi| = 0.2  <-- this should be picked
    assert np.allclose(loss, 0.2)


def test_contrastive_loss_layer():
    layer = ContrastiveLossLayer(temperature=0.1, sigma=0.1)

    # Identical positions and latents
    z = tf.random.normal((4, 128))
    pos = tf.constant([[0.1], [0.2], [0.3], [0.4]], dtype=tf.float32)

    loss = layer([z, pos])
    assert loss.shape == ()
    assert loss >= 0


def test_multi_column_loss_layer():
    # Test grouping columns
    # Column 0: MSE, Columns 1,2: Huber
    column_losses = {"0": "mse", "1,2": "huber"}
    column_weights = {"0": 1.0, "1,2": 0.5}

    layer = MultiColumnLossLayer(
        column_losses=column_losses, column_weights=column_weights
    )

    y_true = tf.random.normal((8, 3))
    y_pred = tf.random.normal((8, 3))

    # We need to build/call to initialize
    loss = layer(y_true, y_pred)

    assert loss.shape == (8,)

    # Test merge_columns
    layer_merged = MultiColumnLossLayer(
        merge_columns=[[0, 1]], merge_losses=["mse"], merge_weights=[1.0]
    )
    loss_merged = layer_merged(y_true, y_pred)
    assert loss_merged.shape == (8,)


def test_get_loss_function_invalid():
    with pytest.raises(ValueError, match="not recognized"):
        _get_loss_function("non_existent_loss", alpha=1.0, delta=1.0)
