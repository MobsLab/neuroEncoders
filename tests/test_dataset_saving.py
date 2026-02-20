import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from neuroencoders.fullEncoder.an_network import LSTMandSpikeNetwork as TFNet


@pytest.fixture
def mock_project(tmp_path):
    # Using tmp_path fixture from pytest
    project = MagicMock()
    project.folder = str(tmp_path)
    project.folderModels = os.path.join(project.folder, "models")
    os.makedirs(project.folderModels, exist_ok=True)
    return project


@pytest.fixture
def mock_params():
    params = MagicMock()
    params.batch_size = 2
    params.nGroups = 1
    params.nChannelsPerGroup = [32]
    params.stride = 36
    params.windowSizeMS = 100
    params.dimOutput = 2
    params.GaussianHeatmap = False
    params.usingMixedPrecision = False
    params.max_nb_spikes = 512
    params.max_nb_spikes_per_group = 512
    return params


def test_save_datasets_isolated(mock_params, mock_project):
    # This test verifies the saving methods in isolation
    # We create a mock network object to avoid complex __init__
    model_obj = MagicMock(spec=TFNet)
    # Re-bind the real methods to the mock object
    model_obj._save_datasets_to_tfrec = TFNet._save_datasets_to_tfrec.__get__(
        model_obj, TFNet
    )
    model_obj._save_datasets_to_parquet = TFNet._save_datasets_to_parquet.__get__(
        model_obj, TFNet
    )
    model_obj.convert_tfrec_to_pandas = TFNet.convert_tfrec_to_pandas.__get__(
        model_obj, TFNet
    )
    model_obj.params = mock_params
    model_obj.featDesc = {
        "pos": tf.io.FixedLenFeature([2], tf.float32),
        "group0": tf.io.VarLenFeature(tf.float32),
        "pos_index": tf.io.FixedLenFeature([], tf.int64),
        "groups": tf.io.VarLenFeature(tf.int64),
        "indexInDat": tf.io.VarLenFeature(tf.int64),
    }

    # Mock data with matching dimensions for samples
    # For one sample: pos (2,), groups (N,), group0 (N*32*32)
    data = {
        "pos": tf.constant(np.random.rand(2), dtype=tf.float32),
        "group0": tf.constant(np.random.rand(10 * 5 * 32), dtype=tf.float32),
        "pos_index": tf.constant(1, dtype=tf.int64),
        "groups": tf.constant([0] * 10, dtype=tf.int64),
        "indexInDat": tf.constant([123] * 10, dtype=tf.int64),
    }

    dataset = tf.data.Dataset.from_tensors(data)
    datasets = {"train": dataset}

    base_tfrec = os.path.join(mock_project.folder, "isolated_saved")
    base_parquet = os.path.join(mock_project.folder, "isolated_saved")

    # Test TFRecord saving
    model_obj._save_datasets_to_tfrec(datasets, base_tfrec)
    assert os.path.exists(f"{base_tfrec}_train.tfrec")

    # Verify TFRecord can be read back
    raw_dataset = tf.data.TFRecordDataset(f"{base_tfrec}_train.tfrec")
    feature_description = {
        "pos": tf.io.FixedLenFeature([2], tf.float32),
        "group0": tf.io.VarLenFeature(tf.float32),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    for parsed in parsed_dataset.take(1):
        assert "pos" in parsed
        assert "group0" in parsed

    # Test Parquet saving
    model_obj._save_datasets_to_parquet(datasets, base_parquet)
    assert os.path.exists(f"{base_parquet}_train.parquet")

    # Verify Parquet content
    df = pd.read_parquet(f"{base_parquet}_train.parquet")
    assert "pos" in df.columns
    assert len(df) == 1
    # Verify flattening (group0 should be a 1D array/list in the cell)
    assert len(df["group0"].iloc[0]) == 10 * 5 * 32


def test_load_parsed_dataset(mock_params, mock_project):
    # This test verifies the loading method with the suggested VarLenFeature featDesc
    model_obj = MagicMock(spec=TFNet)
    model_obj._save_datasets_to_tfrec = TFNet._save_datasets_to_tfrec.__get__(
        model_obj, TFNet
    )
    model_obj.load_parsed_dataset = TFNet.load_parsed_dataset.__get__(model_obj, TFNet)
    model_obj.params = mock_params
    model_obj.params.nGroups = 2  # Test with 2 groups
    model_obj.params.nChannelsPerGroup = [18, 5]  # define for both groups

    # Mock data for 2 groups
    data = {
        "pos_index": tf.constant(1, dtype=tf.int64),
        "pos": tf.constant([0.1, 0.2, 0.3], dtype=tf.float32),  # 3D position
        "length": tf.constant(5, dtype=tf.int64),
        "groups": tf.constant([0, 1, 0, 1, 0], dtype=tf.int64),
        "time": tf.constant(100.5, dtype=tf.float32),
        "time_behavior": tf.constant(100.6, dtype=tf.float32),
        "indexInDat": tf.constant([10, 20, 30, 40, 50], dtype=tf.int64),
        "group0": tf.constant(np.random.rand(3 * 18 * 32), dtype=tf.float32),
        "group1": tf.constant(np.random.rand(2 * 5 * 32), dtype=tf.float32),
    }

    dataset = tf.data.Dataset.from_tensors(data)
    datasets = {"train": dataset}

    base_path = os.path.join(mock_project.folder, "loading_test")

    # Save it first
    model_obj._save_datasets_to_tfrec(datasets, base_path)

    featDesc = {
        "pos_index": tf.io.FixedLenFeature([], tf.int64),
        "pos": tf.io.FixedLenFeature([3], tf.float32),
        "length": tf.io.FixedLenFeature([], tf.int64),
        "groups": tf.io.VarLenFeature(tf.int64),
        "time": tf.io.FixedLenFeature([], tf.float32),
        "time_behavior": tf.io.FixedLenFeature([], tf.float32),
        "indexInDat": tf.io.VarLenFeature(tf.int64),
    }
    for g in range(2):
        featDesc[f"group{g}"] = tf.io.VarLenFeature(tf.float32)

    # Load it back using the new load_parsed_dataset
    # It should use its internal default featDesc if pos is really a FixedLenFeature of 2, otherwise we need to provide
    loaded_datasets = model_obj.load_parsed_dataset(
        base_path, keys=["train"], featDesc=featDesc, dimOutput=3
    )

    assert "train" in loaded_datasets
    for batch in loaded_datasets["train"].take(1):
        # After parse_serialized_sequence, pos should be dense
        assert not isinstance(batch["pos"], tf.SparseTensor)
        assert batch["pos"].shape == (3,)
        assert np.allclose(batch["pos"].numpy(), [0.1, 0.2, 0.3])

        # Verify other fields
        assert batch["length"] == 5
        assert len(batch["groups"]) == 512
        assert "group0" in batch
        assert "group1" in batch
        # Reshaped group0: [num_spikes, channels, 32] -> [3, 32, 32]
        assert batch["group0"].shape == (512, 18, 32)
        assert batch["group1"].shape == (512, 5, 32)
