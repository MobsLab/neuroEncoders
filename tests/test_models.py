import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

# Assuming original TF class is importable
try:
    from neuroencoders.fullEncoder.an_network import LSTMandSpikeNetwork as TFNet
except ImportError:
    TFNet = None


def get_mock_inputs(
    backend, batch_size=2, n_groups=2, n_channels=[2, 2], seq_len=10, max_spikes=20
):
    """Generates mock inputs for the specified backend."""
    inputs = {}

    if backend == "tensorflow":
        for g in range(n_groups):
            # Group voltage inputs: (Batch, MaxSpikes, Channels, Time)
            inputs[f"group{g}"] = np.random.normal(
                size=(batch_size, max_spikes, n_channels[g], 32)
            ).astype(np.float32)
            # Indices for gathering: (Batch, SeqLen)
            # Indices should be between 0 and max_spikes (0 is null spike)
            inputs[f"indices{g}"] = np.random.randint(
                0, max_spikes + 1, size=(batch_size, seq_len)
            ).astype(np.int32)

        # Groups indicator: (Batch, SeqLen)
        inputs["groups"] = np.random.randint(
            0, n_groups, size=(batch_size, seq_len)
        ).astype(np.int32)
        # Ground truth pos for internal loss: (Batch, 2)
        inputs["pos"] = np.random.normal(size=(batch_size, 2)).astype(np.float32)

    return inputs


def get_mock_behavior_data(n_samples=100):
    return {
        "Times": {
            "speedFilter": np.ones((n_samples,), dtype=bool),
            "trainEpochs": np.array([[0, n_samples]]),
        },
        "positionTime": np.linspace(0, n_samples, n_samples)[:, None],
        "Positions": np.random.rand(n_samples, 2),
    }


@pytest.fixture
def mock_project(temp_project_dir):
    project_dir, xml_path = temp_project_dir
    project = MagicMock()
    project.experimentPath = project_dir
    project.xml = xml_path
    project.folderResult = os.path.join(project_dir, "Network")
    project.folderResultSleep = os.path.join(project_dir, "Network", "results_Sleep")
    return project


def test_model_instantiation(mock_params, mock_project):
    behavior_data = get_mock_behavior_data()
    if TFNet is None:
        pytest.skip("TFNet not available")
    model = TFNet(
        projectPath=mock_project, params=mock_params, behaviorData=behavior_data
    )
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)


def test_model_forward(mock_params, mock_project):
    behavior_data = get_mock_behavior_data()
    backend = "tensorflow"
    inputs = get_mock_inputs(
        backend,
        batch_size=mock_params.batchSize,
        n_groups=mock_params.nGroups,
        n_channels=mock_params.nChannelsPerGroup,
    )

    if TFNet is None:
        pytest.skip("TFNet not available")
    model_obj = TFNet(
        projectPath=mock_project, params=mock_params, behaviorData=behavior_data
    )
    output = model_obj.model(inputs)

    if isinstance(output, (list, tuple)):
        pos_out = output[0]
    else:
        pos_out = output

    assert pos_out.shape == (mock_params.batchSize, mock_params.dimOutput)


def test_train_step(mock_params, mock_project):
    behavior_data = get_mock_behavior_data()
    backend = "tensorflow"
    inputs = get_mock_inputs(
        backend,
        batch_size=mock_params.batchSize,
        n_groups=mock_params.nGroups,
        n_channels=mock_params.nChannelsPerGroup,
    )

    if TFNet is None:
        pytest.skip("TFNet not available")
    model_obj = TFNet(
        projectPath=mock_project, params=mock_params, behaviorData=behavior_data
    )

    targets = {
        "myoutputPos": np.random.randn(mock_params.batchSize, mock_params.dimOutput),
        "posLoss": np.zeros((mock_params.batchSize,)),
    }
    for name in model_obj.outNames[2:]:
        targets[name] = np.zeros((mock_params.batchSize,))

    loss = model_obj.model.train_on_batch(inputs, targets)
    assert loss is not None
