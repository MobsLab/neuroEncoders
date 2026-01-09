import os
import pytest
import numpy as np
import torch
import tensorflow as tf
from unittest.mock import MagicMock, patch

# Import implementation classes
from neuroencoders.fullEncoder.an_network_torch import LSTMandSpikeNetwork as TorchNet

# Assuming original TF class is importable
try:
    from neuroencoders.fullEncoder.an_network import LSTMandSpikeNetwork as TFNet
except ImportError:
    TFNet = None


def get_mock_inputs(
    backend, batch_size=2, n_groups=2, n_channels=[2, 2], n_features=64
):
    """Generates mock inputs for the specified backend."""
    inputs = {}

    if backend == "pytorch":
        for g in range(n_groups):
            indices = torch.arange(batch_size).repeat_interleave(n_channels[g])
            inputs[f"indices{g}"] = indices
            inputs[f"group{g}"] = torch.randn(len(indices), n_channels[g], 32)
        inputs["batch_size"] = batch_size

    elif backend == "tensorflow":
        for g in range(n_groups):
            # Group voltage inputs: (Batch, Channels, Time)
            inputs[f"group{g}"] = tf.random.normal((batch_size, n_channels[g], 32))
            # Indices for gathering: (Batch,)
            inputs[f"indices{g}"] = tf.zeros((batch_size,), dtype=tf.int32)

        inputs["groups"] = tf.zeros((batch_size,), dtype=tf.int32)
        inputs["zeroForGather"] = tf.zeros((batch_size, n_features))
        inputs["pos"] = tf.zeros((batch_size, 2))

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


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
def test_model_instantiation(backend, mock_params, mock_project):
    behavior_data = get_mock_behavior_data()
    if backend == "pytorch":
        model = TorchNet(projectPath=mock_project, params=mock_params)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    elif backend == "tensorflow":
        if TFNet is None:
            pytest.skip("TFNet not available")
        model = TFNet(
            projectPath=mock_project, params=mock_params, behaviorData=behavior_data
        )
        assert model.model is not None
        assert isinstance(model.model, tf.keras.Model)


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
def test_model_forward(backend, mock_params, mock_project):
    behavior_data = get_mock_behavior_data()
    inputs = get_mock_inputs(
        backend,
        batch_size=mock_params.batchSize,
        n_groups=mock_params.nGroups,
        n_channels=mock_params.nChannelsPerGroup,
        n_features=mock_params.nFeatures,
    )

    if backend == "pytorch":
        model = TorchNet(projectPath=mock_project, params=mock_params)
        output = model(inputs)
        assert output.shape == (mock_params.batchSize, mock_params.dimOutput)
        assert not torch.isnan(output).any()
    elif backend == "tensorflow":
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


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
def test_train_step(backend, mock_params, mock_project):
    behavior_data = get_mock_behavior_data()
    inputs = get_mock_inputs(
        backend,
        batch_size=mock_params.batchSize,
        n_groups=mock_params.nGroups,
        n_channels=mock_params.nChannelsPerGroup,
        n_features=mock_params.nFeatures,
    )

    if backend == "pytorch":
        model = TorchNet(projectPath=mock_project, params=mock_params)
        targets = {"pos": torch.randn(mock_params.batchSize, 2)}
        model.train()
        model.optimizer.zero_grad()
        preds = model(inputs)
        loss = torch.nn.functional.mse_loss(preds, targets["pos"])
        loss.backward()
        model.optimizer.step()
        assert not torch.isnan(loss).any()
    elif backend == "tensorflow":
        if TFNet is None:
            pytest.skip("TFNet not available")
        model_obj = TFNet(
            projectPath=mock_project, params=mock_params, behaviorData=behavior_data
        )

        targets = {
            "myoutputPos": np.random.randn(
                mock_params.batchSize, mock_params.dimOutput
            ),
            "posLoss": np.zeros((mock_params.batchSize,)),
        }
        for name in model_obj.outNames[2:]:
            targets[name] = np.zeros((mock_params.batchSize,))

        loss = model_obj.model.train_on_batch(inputs, targets)
        assert loss is not None
