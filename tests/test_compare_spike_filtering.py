import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

# Ensure module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuroencoders.importData.compareSpikeFiltering import WaveFormComparator


class MockParams:
    def __init__(self):
        self.nGroups = 2
        self.nChannelsPerGroup = [2, 2]  # small channels
        self.dimOutput = 2
        self.batch_size = 2
        self.max_nb_spikes = 10
        self.max_spikes_per_group = 5


class MockProject:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.fil = os.path.join(dataPath, "mock.fil")
        self.dat = os.path.join(dataPath, "mock.dat")
        self.xml = os.path.join(dataPath, "mock.xml")


@pytest.fixture
def mock_env():
    temp_dir = tempfile.mkdtemp()
    params = MockParams()
    project = MockProject(temp_dir)

    # Create dummy dat/fil files
    n_samples = 1000
    n_channels = sum(params.nChannelsPerGroup)
    dummy_data = np.zeros((n_samples, n_channels), dtype=np.int16)
    dummy_data.tofile(project.dat)
    dummy_data.tofile(project.fil)

    # Mock data needed for WaveFormComparator
    behavior_data = {
        "Positions": np.zeros((100, 2)),
        "positionTime": np.zeros((100, 2)),
        "Times": {
            "testEpochs": [[0, 100]],
            "trainEpochs": [[0, 100]],
            "sleepNames": [],
            "sleepEpochs": [],
        },
    }

    # Create dummy TFRecord
    tfrec_path = os.path.join(temp_dir, "dataset_stride36.tfrec")
    with tf.io.TFRecordWriter(tfrec_path) as writer:
        for i in range(5):
            # Create a simple example
            feature = {
                "pos_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                "pos": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[0.0, 0.0])
                ),
                "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                "groups": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[0])
                ),  # Group 0 has a spike
                "time": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                "time_behavior": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[0.0])
                ),
                "indexInDat": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[i * 10])
                ),  # index
                "group0": tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.zeros(2 * 32).flatten())
                ),  # 2 channels * 32 samples
                "group1": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[])
                ),  # Empty
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    yield project, params, behavior_data

    shutil.rmtree(temp_dir)


def test_waveform_comparator_init(mock_env):
    project, params, behavior_data = mock_env

    with patch(
        "neuroencoders.importData.compareSpikeFiltering.get_params",
        return_value=(None, 20000, 4),
    ):
        comparator = WaveFormComparator(
            projectPath=project,
            params=params,
            behavior_data=behavior_data,
            windowSizeMS=36,
            useTrain=True,
            useTest=True,
            useAll=True,
            phase="train",
        )

        assert comparator is not None
        assert comparator.dataset is not None


def test_get_nndataset_spikepos(mock_env):
    project, params, behavior_data = mock_env

    # We need to make sure parsing works correctly
    # mocking inEpochsMask to return all True to avoid filtering everything out
    with (
        patch(
            "neuroencoders.importData.compareSpikeFiltering.get_params",
            return_value=(None, 20000, 4),
        ),
        patch(
            "neuroencoders.importData.compareSpikeFiltering.inEpochsMask",
            return_value=np.ones(100, dtype=bool),
        ),
        patch(
            "neuroencoders.importData.compareSpikeFiltering.get_epochs_mask",
            return_value=np.ones(100, dtype=bool),
        ),
    ):
        comparator = WaveFormComparator(
            projectPath=project,
            params=params,
            behavior_data=behavior_data,
            windowSizeMS=36,
            useTrain=True,
            useTest=False,
            phase="train",
        )

        # Test getting spike positions
        indices, posIndex = comparator.get_NNdataset_spikepos()
        assert len(indices) > 0
        # Check that we don't have -1 (padding)
        for idx_array in indices:
            assert not np.any(idx_array == -1)


def test_save_alignment_tools(mock_env):
    project, params, behavior_data = mock_env

    with (
        patch(
            "neuroencoders.importData.compareSpikeFiltering.get_params",
            return_value=(None, 20000, 4),
        ),
        patch(
            "neuroencoders.importData.compareSpikeFiltering.get_epochs_mask",
            return_value=np.ones(100, dtype=bool),
        ),
    ):
        comparator = WaveFormComparator(
            projectPath=project,
            params=params,
            behavior_data=behavior_data,
            windowSizeMS=36,
            useTrain=True,
            useTest=False,
            phase="train",
        )

        # Mock trainerBayes
        trainerBayes = MagicMock()
        trainerBayes.spikeMatTimes = np.array([[0.1], [0.2]])
        trainerBayes.spikeMatLabels = np.array([[1, 0], [0, 1]])  # 2 clusters

        # Mock linearization function
        linearizationFunction = MagicMock()

        comparator.save_alignment_tools(
            trainerBayes=trainerBayes,
            linearizationFunction=linearizationFunction,
            windowSizeMS=36,
        )

        save_path = os.path.join(comparator.alignedDataPath, "train")
        assert os.path.isdir(save_path)
        assert os.path.isfile(os.path.join(save_path, "startTimeWindow_train.csv"))
