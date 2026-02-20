import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import tables
import tensorflow as tf

from neuroencoders.transformData.linearizer import UMazeLinearizer

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
def mock_linearizer(tmp_path):
    """Creates a mock linearizer for testing."""
    mat_path = tmp_path / "nnBehavior.mat"
    with tables.open_file(str(mat_path), mode="w") as f:
        f.create_group("/", "behavior")

    # UMazeLinearizer will generate canonical path automatically
    return UMazeLinearizer(folder=str(tmp_path), nb_bins=45)


@pytest.fixture
def mock_project(temp_project_dir):
    project_dir, xml_path = temp_project_dir
    project = MagicMock()
    project.experimentPath = project_dir
    project.xml = xml_path
    project.folderResult = os.path.join(project_dir, "Network")
    project.folderResultSleep = os.path.join(project_dir, "Network", "results_Sleep")
    return project


def test_model_instantiation(mock_params, mock_project, mock_linearizer):
    behavior_data = get_mock_behavior_data()
    if TFNet is None:
        pytest.skip("TFNet not available")
    model = TFNet(
        projectPath=mock_project,
        params=mock_params,
        behaviorData=behavior_data,
        linearizer=mock_linearizer,
        jit_compile=False,
        max_spikes_per_group=mock_params.max_nb_spikes_per_group,
    )
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)


def test_model_forward(mock_params, mock_project, mock_linearizer):
    behavior_data = get_mock_behavior_data()
    backend = "tensorflow"
    inputs = get_mock_inputs(
        backend,
        batch_size=mock_params.batch_size,
        n_groups=mock_params.nGroups,
        n_channels=mock_params.nChannelsPerGroup,
        seq_len=mock_params.max_nb_spikes,
        max_spikes=mock_params.max_nb_spikes_per_group,
    )

    if TFNet is None:
        pytest.skip("TFNet not available")
    model_obj = TFNet(
        projectPath=mock_project,
        params=mock_params,
        behaviorData=behavior_data,
        linearizer=mock_linearizer,
        jit_compile=False,
        max_spikes_per_group=mock_params.max_nb_spikes_per_group,
    )
    output = model_obj.model(inputs)

    assert isinstance(output, dict)

    # Verify all expected targets from target_structure are present
    for target_name in model_obj.target_structure.keys():
        assert target_name in output

    if getattr(mock_params, "contrastive_loss", False):
        assert "latent" in output

    # Check shapes
    for target_name, spec in model_obj.target_structure.items():
        if target_name == "pos_2d" and getattr(mock_params, "GaussianHeatmap", False):
            H, W = mock_params.GaussianGridSize
            # For heatmap, shape should be (B, H*W)
            assert output[target_name].shape == (mock_params.batch_size, H * W)
        else:
            assert output[target_name].shape == (mock_params.batch_size, spec["dim"])

    if "latent" in output:
        assert output["latent"].shape == (mock_params.batch_size, mock_params.nFeatures)


def test_train_step(mock_params, mock_project, mock_linearizer):
    behavior_data = get_mock_behavior_data()
    backend = "tensorflow"
    inputs = get_mock_inputs(
        backend,
        batch_size=mock_params.batch_size,
        n_groups=mock_params.nGroups,
        n_channels=mock_params.nChannelsPerGroup,
        seq_len=mock_params.max_nb_spikes,
        max_spikes=mock_params.max_nb_spikes_per_group,
    )

    if TFNet is None:
        pytest.skip("TFNet not available")
    model_obj = TFNet(
        projectPath=mock_project,
        params=mock_params,
        behaviorData=behavior_data,
        linearizer=mock_linearizer,
        jit_compile=False,
        max_spikes_per_group=mock_params.max_nb_spikes_per_group,
    )

    targets = {}
    for name, meta in model_obj.target_structure.items():
        targets[name] = np.random.randn(mock_params.batch_size, meta["dim"]).astype(
            np.float32
        )

    if getattr(mock_params, "contrastive_loss", False):
        targets["latent"] = np.zeros(
            (mock_params.batch_size, mock_params.nFeatures)
        ).astype(np.float32)

    loss = model_obj.model.train_on_batch(inputs, targets)
    assert loss is not None


def test_model_fit(mock_params, mock_project, mock_linearizer):
    """Verifies that model.fit works correctly with the custom loss and multi-output."""
    behavior_data = get_mock_behavior_data()
    backend = "tensorflow"

    if TFNet is None:
        pytest.skip("TFNet not available")

    model_obj = TFNet(
        projectPath=mock_project,
        params=mock_params,
        behaviorData=behavior_data,
        linearizer=mock_linearizer,
        jit_compile=False,
        max_spikes_per_group=mock_params.max_nb_spikes_per_group,
    )

    def generate_data():
        for _ in range(3):
            batch_inputs = get_mock_inputs(
                backend,
                batch_size=mock_params.batch_size,
                n_groups=mock_params.nGroups,
                n_channels=mock_params.nChannelsPerGroup,
                seq_len=mock_params.max_nb_spikes,
                max_spikes=mock_params.max_nb_spikes_per_group,
            )
            batch_targets = {}
            for name, meta in model_obj.target_structure.items():
                batch_targets[name] = np.random.randn(
                    mock_params.batch_size, meta["dim"]
                ).astype(np.float32)

            if getattr(mock_params, "contrastive_loss", False):
                batch_targets["latent"] = np.zeros(
                    (mock_params.batch_size, mock_params.nFeatures)
                ).astype(np.float32)
            yield (batch_inputs, batch_targets)

    history = model_obj.model.fit(
        generate_data(), epochs=1, steps_per_epoch=3, verbose=0
    )

    assert "loss" in history.history
    # Check that individual losses and metrics are reported
    keys = history.history.keys()
    for target_name in model_obj.target_structure.keys():
        assert any(target_name in k for k in keys)
