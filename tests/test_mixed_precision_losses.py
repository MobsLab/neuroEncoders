import numpy as np
import pytest
import tensorflow as tf

from neuroencoders.fullEncoder import nnUtils
from neuroencoders.fullEncoder.an_network import LinearizationLayer

# Set mixed precision policy if possible
policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
tf.keras.mixed_precision.set_global_policy(policy)


@pytest.fixture
def mock_l_layer():
    maze_points = np.random.randn(45, 2).astype(np.float32)
    ts_proj = np.random.randn(45).astype(np.float32)
    return LinearizationLayer(maze_points=maze_points, ts_proj=ts_proj)


@pytest.fixture
def gaussian_params():
    return {
        "training_positions": np.random.rand(100, 2).astype(np.float32),
        "grid_size": (45, 45),
        "sigma": 0.03,
        "neg": -100,
    }


def test_gaussian_heatmap_losses_mp(gaussian_params, mock_l_layer):
    losses_layer = nnUtils.GaussianHeatmapLosses(
        **gaussian_params, l_function_layer=mock_l_layer
    )

    # Create bfloat16 inputs
    batch_size = 4
    logits = tf.random.normal((batch_size, 45, 45), dtype=tf.bfloat16)
    targets = tf.random.uniform((batch_size, 45, 45), dtype=tf.bfloat16)

    # Call the layer
    loss = losses_layer({"logits": logits, "targets": targets}, loss_type="safe_kl")

    # Assertions
    assert loss.dtype == tf.float32
    assert not tf.math.is_nan(loss)


def test_contrastive_loss_layer_mp():
    contrastive_layer = nnUtils.ContrastiveLossLayer()

    batch_size = 8
    z = tf.random.normal((batch_size, 64), dtype=tf.bfloat16)
    pos = tf.random.uniform((batch_size, 1), dtype=tf.bfloat16)

    loss = contrastive_layer([pos, z])

    assert loss.dtype == tf.float32
    assert not tf.math.is_nan(loss)


def test_multi_column_loss_layer_mp(gaussian_params, mock_l_layer):
    gh_losses = nnUtils.GaussianHeatmapLosses(
        **gaussian_params, l_function_layer=mock_l_layer
    )

    multi_loss = nnUtils.MultiColumnLossLayer(
        column_losses={"0": "mse", "1": "kl_heatmap"}, gaussian_loss_layer=gh_losses
    )
    print(f"managed to instantiate MultiColumnLossLayer with kl_heatmap: {multi_loss}")

    batch_size = 4
    y_true = tf.random.uniform((batch_size, 2), dtype=tf.bfloat16)
    # y_pred for kl_heatmap expects logits, we just need the right shape
    y_pred = tf.random.normal((batch_size, 2), dtype=tf.bfloat16)

    # MultiColumnLossLayer with kl_heatmap expects (B, H, W) for heatmap column
    # This is handled by _get_loss_function(kl_heatmap)
    # Actually, MultiColumnLossLayer slices y_true, y_pred.

    # Let's test a simpler MSE aggregation first
    multi_loss_simple = nnUtils.MultiColumnLossLayer(
        column_losses={"0": "mse", "1": "mae"}
    )
    y_true = tf.random.uniform((batch_size, 2), dtype=tf.bfloat16)
    y_pred = tf.random.uniform((batch_size, 2), dtype=tf.bfloat16)

    loss = multi_loss_simple(y_true, y_pred)

    assert loss.dtype == tf.float32
    assert not tf.math.is_nan(tf.reduce_mean(loss))


def test_cyclical_mae_rad_mp():
    loss_fn = nnUtils._get_loss_function("cyclic_mae", alpha=1.0, delta=1.0)

    y_true = tf.constant([0.1, 6.2], dtype=tf.bfloat16)
    y_pred = tf.constant([0.2, 0.1], dtype=tf.bfloat16)

    loss = loss_fn(y_true, y_pred)

    assert loss.dtype == tf.float32
    # Check cyclical logic roughly (abs(0.1-6.2) vs abs(0.1-(6.2-2pi)) etc)
    # 6.2 rad is almost 2pi. Expected diff is small.
    # 0.1 and 6.2 => diff is ~0.2 rad in cycle.
    assert tf.reduce_mean(loss) < 0.2
