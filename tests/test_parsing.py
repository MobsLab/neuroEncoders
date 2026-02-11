import numpy as np
import pytest
import tensorflow as tf

from neuroencoders.fullEncoder import nnUtils


class MockParams:
    def __init__(self):
        self.nGroups = 2
        self.nChannelsPerGroup = [2, 4]
        self.nFeatures = 64
        self.batch_size = 4


def create_dummy_parsed_tensors(params):
    tensors = {
        "pos": tf.constant(np.random.rand(2).astype(np.float32)),
        "groups": tf.SparseTensor(
            indices=[[0], [1], [2]], values=[0, 1, 0], dense_shape=[3]
        ),
        "indexInDat": tf.SparseTensor(
            indices=[[0], [1], [2]], values=[10, 20, 30], dense_shape=[3]
        ),
    }

    for g in range(params.nGroups):
        # Create non-zero spike for group g
        num_spikes = 2
        spike_data = (
            np.random.rand(num_spikes * params.nChannelsPerGroup[g] * 32)
            .astype(np.float32)
            .reshape(-1)
        )

        # We need to simulate the VarLenFeature structure
        tensors[f"group{g}"] = tf.SparseTensor(
            indices=[[i] for i in range(len(spike_data))],
            values=spike_data,
            dense_shape=[len(spike_data)],
        )

    return tensors


def test_parse_serialized_sequence():
    params = MockParams()
    tensors = create_dummy_parsed_tensors(params)

    # Call the refactored function
    parsed = nnUtils.parse_serialized_sequence(params, tensors, count_spikes=True)

    # Check pos
    assert isinstance(parsed["pos"], tf.Tensor)
    assert parsed["pos"].shape == (2,)

    # Check groups
    assert parsed["groups"].shape == (3,)

    # Check group tensors
    for g in range(params.nGroups):
        group_key = f"group{g}"
        assert group_key in parsed
        # parse_serialized_sequence reshapes to [-1, channels, 32] and filters non-zeros
        # Since we added non-zero data, there should be 2 spikes
        assert len(parsed[group_key].shape) == 3
        assert parsed[group_key].shape[1] == params.nChannelsPerGroup[g]
        assert parsed[group_key].shape[2] == 32

        # Check spike counts
        count_key = f"group{g}_spikes_count"
        assert count_key in parsed
        assert parsed[count_key] == parsed[group_key].shape[0]


def test_parse_serialized_sequence_with_augmentation():
    params = MockParams()
    tensors = create_dummy_parsed_tensors(params)

    # Without augmentation config, it should just parse
    parsed = nnUtils.parse_serialized_sequence_with_augmentation(
        params, tensors, count_spikes=True
    )

    assert "group0" in parsed
    assert "group1" in parsed
    assert parsed["group0_spikes_count"] == parsed["group0"].shape[0]


if __name__ == "__main__":
    pytest.main([__file__])
