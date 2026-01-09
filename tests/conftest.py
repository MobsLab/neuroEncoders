import pytest
import numpy as np
import os
import tensorflow as tf


class MockParams:
    def __init__(self):
        self.nGroups = 2
        self.nChannelsPerGroup = [2, 2]  # 2 groups, 2 channels each
        self.nFeatures = 64
        self.nHeads = 4
        self.ff_dim1 = 128
        self.ff_dim2 = 128
        self.dropoutLSTM = 0.1
        self.lstmLayers = 2
        self.lstmSize = 64
        self.TransformerDenseSize1 = 64
        self.TransformerDenseSize2 = 32
        self.dimOutput = 2
        self.batchSize = 4
        self.windowLength = 0.2
        self.dim_factor = 2
        self.project_transformer = True
        self.use_group_attention_fusion = True
        self.weight_decay = 1e-4
        self.learningRates = [1e-3, 1e-4]
        self.target = "pos"
        self.usingMixedPrecision = False
        self.windowSize = 0.036
        self.windowSizeMS = 36
        self.denseweight = False
        self.GaussianHeatmap = False
        self.OversamplingResampling = False

        self.resultsPath = "test_results"
        self.use_conv2d = False
        self.isTransformer = True
        self.GaussianGridSize = (45, 45)
        self.GaussianSigma = 0.05
        self.GaussianEps = 1e-6
        self.nDenseLayers = 2
        self.featureActivation = None
        self.dropoutCNN = 0.35
        self.reduce_dense = False
        self.no_cnn = False
        self.loss = "mse"
        self.contrastive_loss = False
        self.alpha = 1.3
        self.delta = 0.5
        self.transform_w_log = False
        self.mixed_loss = False
        self.dataAugmentation = False
        self.nSteps = 10
        self.heatmap_weight = 1.0
        self.others_weight = 1.0


@pytest.fixture
def mock_params():
    return MockParams()


@pytest.fixture
def temp_project_dir(tmp_path):
    """Creates a temporary project structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create necessary subdirectories
    (project_dir / "dataset").mkdir()
    (project_dir / "Network").mkdir()
    (project_dir / "Network" / "models").mkdir()

    # Create fake XML
    xml_path = project_dir / "test.xml"
    xml_path.write_text("<xml>Mock XML</xml>")

    # Create fake data file
    dat_path = project_dir / "test.dat"
    dat_path.write_text("fake binary content")

    return str(project_dir), str(xml_path)
