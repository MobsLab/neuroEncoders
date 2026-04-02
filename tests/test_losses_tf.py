import numpy as np
import tensorflow as tf

from neuroencoders.fullEncoder.nnUtils import ContrastiveRegressionLoss, CyclicMAE


def test_get_loss_function_cyclic():
    # Test cyclic_mae (radians)
    y_true = tf.constant([[0.1]], dtype=tf.float32)
    # distance across the 2pi boundary
    y_pred = tf.constant([[2 * np.pi - 0.1]], dtype=tf.float32)

    loss_fn = CyclicMAE()
    loss = loss_fn(y_true, y_pred)

    # |0.1 - (2pi - 0.1)| = 2pi - 0.2
    # |0.1 - (2pi - 0.1) + 2pi| = 0.2  <-- this should be picked
    assert np.allclose(loss, 0.2)


def test_contrastive_loss_layer():
    layer = ContrastiveRegressionLoss(
        target_structure={"pos_2d": {"dim": 1, "slice": 0, "activation": None}},
        temperature=0.1,
        sigma=0.1,
    )

    # Identical positions and latents
    z = tf.random.normal((4, 128))
    pos = tf.constant([[0.1], [0.2], [0.3], [0.4]], dtype=tf.float32)

    loss = layer([pos, z])  # respect the input format of [y_true, y_pred]
    assert loss.shape == ()
    assert loss >= 0
