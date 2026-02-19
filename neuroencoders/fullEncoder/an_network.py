"""
Neuroencoders: LSTMandSpikeNetwork
an_network module for training and managing LSTM and spiking neural networks.
"""
# Pierre 14/02/21:
# Reorganization of the code:
# One class for the network
# One function for the training boom nahui
# We save the model every epoch during the training
# Dima 21/01/22:
# Cleanining and rewriting of the module

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import warnings
from typing import Dict, List, Optional, Tuple

# Get common libraries
import dill as pickle
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops as kops
from tqdm import tqdm

import wandb

# Get utility functions
from neuroencoders.fullEncoder import nnUtils
from neuroencoders.fullEncoder.nnUtils import (
    AddNullSpike,
    ContrastiveMonitor,
    ContrastiveRegressionLoss,
    CyclicMAE,
    GatherSpikes,
    GaussianHeatmapLayer,
    GaussianHeatmapLoss,
    GroupAttentionFusion,
    MaskedGlobalAveragePooling1D,
    MaskingLayer,
    MemoryUsageCallbackExtended,
    NeuralDataAugmentation,
    PositionError2D,
    PositionalEncoding,
    SafeMaskCreation,
    SpikeNet1D,
    TransformerEncoderBlock,
    UMazeProjectionLayer,
    spikeNet,
)
from neuroencoders.importData.epochs_management import get_epochs_mask, inEpochsMask
from neuroencoders.utils.global_classes import (
    DataHelper,
    Params,
    Project,
    SpatialConstraintsMixin,
)
from wandb.integration.keras import WandbMetricsLogger


# We generate a model with the functional Model interface in tensorflow
########### START OF FULL NETWORK CLASS #####################
class LSTMandSpikeNetwork(SpatialConstraintsMixin):
    """
    LSTMandSpikeNetwork class, the main ann Class.

    Parameters
    ----------
    projectPath : Project object
        Contains the path to the project, the xml file, the dat file, the positions...

    params : Params object
        Contains the parameters of the network (nb of Groups, nb of channels per group, nb of features...)

    deviceName : str, optional, default to CPU
    debug : bool, optional, default to False (whether to use tf profiler with tensorboard)
    phase : str, optional, default to None (if the nnBehavior is used in for a specific session (pre, post...))

    **kwargs : dict, optional
        Additional parameters for the network, such as Transformer vs LSTM, dropout rates, learning rates, activation functions, etc.
    """

    def __init__(
        self,
        projectPath: Project,
        params: Params,
        deviceName: str = "/device:CPU:0",
        debug: bool = False,
        phase: Optional[str] = None,
        **kwargs,
    ):
        # Initialize SpatialConstraintsMixin
        grid_size = getattr(params, "GaussianGridSize", (45, 45))
        maze_params = getattr(params, "GaussianHeatmapMaze", None)
        super(LSTMandSpikeNetwork, self).__init__(
            grid_size=grid_size, maze_params=maze_params, **kwargs
        )

        self.clear_session()
        ### Main parameters here
        self.projectPath = projectPath  # Project object containing the path to the project, the xml file, the dat file, the positions...
        self.params = params  # Params object containing the parameters of the network (nb of Groups, nb of channels per group, nb of features...)
        self.deviceName = deviceName
        self.debug = debug
        self.target = params.target
        self.phase = phase
        self.suffix = "_" + str(phase) if phase is not None else ""
        self._setup_folders()
        self._setup_feature_description()

        self.zeroForGather = tf.zeros([1, self.params.nFeatures])
        self.max_nb_spikes = kwargs.get(
            "max_nb_spikes", 128
        )  # maximum number of spikes per group to consider in the window, for batching purposes
        self.max_spikes_per_group = kwargs.get(
            "max_spikes_per_group", self.max_nb_spikes / self.params.nGroups
        )

        if self.params.usingMixedPrecision:
            print("Using mixed precision with float16")
            policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Compute dtype:", policy.compute_dtype)
            print("Variable dtype:", policy.variable_dtype)
        else:
            print("Not using mixed precision, using float32")

        # Moved the initialization of the DataHelper here
        if kwargs.get("linearizer", None) is not None:
            Linearizer = kwargs["linearizer"]
            self.fix_linearizer(Linearizer.mazePoints, Linearizer.tsProj)
        else:
            self.maze_points = None
            self.ts_proj = None
            self.mazePoints_tensor = None
            self.tsProjTensor = None

        self.target_structure = self._parse_target_structure()

        if params.denseweight:
            if kwargs.get("behaviorData", None) is None:
                warnings.warn(
                    '"behaviorData" not provided, using default setup WITHOUT Dense Weight. Is your code version deprecated?'
                )
            else:
                self.setup_dynamic_dense_loss(**kwargs)
        else:
            self.setup_training_data(**kwargs)
            # just for sake of compatibility

        if getattr(params, "GaussianHeatmap", False) or getattr(
            params, "OversamplingResampling", False
        ):
            assert not params.denseweight, (
                "Cannot use both GaussianHeatmap and DenseWeight"
            )
            if kwargs.get("behaviorData", None) is None:
                warnings.warn(
                    '"behaviorData" not provided, using default setup WITHOUT Gaussian Heatmap layering. Is your code version deprecated?'
                )
            else:
                self.lfunction_layer_params = {
                    "maze_points": self.maze_points,
                    "ts_proj": self.ts_proj,
                    "device": self.deviceName,
                }
                self.setup_gaussian_heatmap(**kwargs)
        else:
            self.gaussian_heatmap_params = None
            self.lfunction_layer_params = None

        self._build_model(**kwargs)
        # if kwargs.get("extractTransformer", False):
        #     self.create_separable_models(
        #         self.model,
        #         spikeNetsoutputsName="dropoutCNN",
        #         transformer_start_layer_name="feature_projection_transformer",
        #         save=True,
        #     )

    def _setup_folders(self):
        self.folderResult = self.projectPath.folderResult
        try:
            self.folderResultSleep = self.projectPath.folderResultSleep
        except AttributeError:
            self.folderResultSleep = os.path.join(
                self.projectPath.experimentPath, "results_Sleep"
            )
            self.projectPath.folderResultSleep = self.folderResultSleep
        self.folderModels = os.path.join(self.projectPath.experimentPath, "models")
        os.makedirs(self.folderResult, exist_ok=True)
        os.makedirs(self.folderResultSleep, exist_ok=True)
        os.makedirs(self.folderModels, exist_ok=True)

    def _setup_feature_description(self):
        # The featDesc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.featDesc = {
            # index of the position in the position array
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            # target position: current value of the environmental correlate
            "pos": tf.io.FixedLenFeature([2], tf.float32),
            # number of spike sequence gathered in the window
            "length": tf.io.FixedLenFeature([], tf.int64),
            # the index of the groups having spike sequences in the window
            "groups": tf.io.VarLenFeature(tf.int64),
            # the mean time-steps of each spike measured in the various groups.
            # Question: should the time not be a VarLenFeature??
            "time": tf.io.FixedLenFeature([], tf.float32),
            # the exact time step from behaviorData["Times"]
            "time_behavior": tf.io.FixedLenFeature([], tf.float32),
            # sample of the spike
            "indexInDat": tf.io.VarLenFeature(tf.int64),
        }
        for g in range(self.params.nGroups):
            # the voltage values (discretized over 32 time bins) of each channel (4 most of the time)
            # of each spike of a given group in the window
            self.featDesc.update(
                {"group" + str(g): tf.io.VarLenFeature(tf.float32)}
            )  # of length nSpikes * nChannels * 32

        # Loss obtained during training
        self.trainLosses = {}

    def _parse_target_structure(self):
        """
        Parses the target string from self.params.target and returns a dictionary
        mapping output names to their dimensions, slices, and ideal activation.
        This is based on the logic in DataHelper.get_true_target().
        """
        target = self.params.target.lower()
        use_heatmap = getattr(self.params, "GaussianHeatmap", False)

        # Pos 2D dimensions: 2 for raw regression or grid size for heatmap
        pos_dim_out = 2
        if use_heatmap:
            pos_dim_out = (
                self.params.GaussianGridSize[0] * self.params.GaussianGridSize[1]
            )

        # Define structure based on the concatenation order in DataHelper.get_true_target
        structure = {}

        if target == "pos":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
        elif target in ["lin", "linear"]:
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "linear"}
        elif target == "linandthigmo":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "linear"}
            structure["thigmo"] = {"dim": 1, "slice": (1, 2), "activation": "linear"}
        elif target == "linanddirection":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "linear"}
            structure["direction"] = {
                "dim": 1,
                "slice": (1, 2),
                "activation": "sigmoid",
            }
        elif target == "direction":
            structure["direction"] = {
                "dim": 1,
                "slice": (0, 1),
                "activation": "sigmoid",
            }
        elif target == "linandheaddirection":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "linear"}
            structure["hd"] = {"dim": 1, "slice": (1, 2), "activation": "linear"}
        elif target == "linandspeed":
            structure["pos_lin"] = {"dim": 1, "slice": (0, 1), "activation": "linear"}
            structure["speed"] = {"dim": 1, "slice": (1, 2), "activation": "linear"}
        elif target == "posanddirection":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
            structure["direction"] = {
                "dim": 1,
                "slice": (2, 3),
                "activation": "sigmoid",
            }
        elif target == "posandheaddirection":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
            structure["hd"] = {"dim": 1, "slice": (2, 3), "activation": "linear"}
        elif target == "posandspeed":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
            structure["speed"] = {"dim": 1, "slice": (2, 3), "activation": "linear"}
        elif target == "posanddirectionandthigmo":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
            structure["direction"] = {
                "dim": 1,
                "slice": (2, 3),
                "activation": "sigmoid",
            }
            structure["thigmo"] = {"dim": 1, "slice": (3, 4), "activation": "linear"}
        elif target == "posandheaddirectionandspeed":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
            structure["hd"] = {"dim": 1, "slice": (2, 3), "activation": "linear"}
            structure["speed"] = {"dim": 1, "slice": (3, 4), "activation": "linear"}
        elif target == "posandheaddirectionandthigmo":
            structure["pos_2d"] = {
                "dim": pos_dim_out,
                "slice": (0, 2),
                "activation": "linear",
            }
            structure["hd"] = {"dim": 1, "slice": (2, 3), "activation": "linear"}
            structure["thigmo"] = {"dim": 1, "slice": (3, 4), "activation": "linear"}
        else:
            # Fallback for complex unknown target
            structure["main_pred"] = {
                "dim": self.params.dimOutput,
                "slice": (0, self.params.dimOutput),
                "activation": "linear",
            }

        return structure

    def _parse_loss_and_metrics_dict(self, **loss_kwargs):
        loss_dict = {}
        loss_weights = {}
        metrics_dict = {}

        for name in self.outNames:
            if name == "pos_2d":
                if getattr(self.params, "GaussianHeatmap", False):
                    assert self.gaussian_heatmap_params is not None, (
                        "Gaussian heatmap parameters not set up"
                    )
                    loss_dict[name] = GaussianHeatmapLoss(
                        gaussian_params=self.gaussian_heatmap_params,
                        l_function_params=self.lfunction_layer_params,
                        loss_type=getattr(self.params, "loss_type", "safe_kl"),
                        **loss_kwargs,
                    )
                    loss_weights[name] = getattr(self.params, "heatmap_weight", 1.5)
                    metrics_dict[name] = [
                        PositionError2D(
                            self.GaussianHeatmap.get_config(), name="dist_2d"
                        )
                    ]
                else:
                    loss_dict[name] = "mae"
                    loss_weights[name] = 1.2
                    metrics_dict[name] = ["mae"]

            elif name == "pos_lin":
                loss_dict[name] = "mae"
                loss_weights[name] = 1.0
                metrics_dict[name] = ["mae"]

            elif name == "thigmo":
                loss_dict[name] = "mae"
                loss_weights[name] = getattr(self.params, "thigmo_weight", 0.8)
                metrics_dict[name] = ["mae"]

            elif name == "direction":
                loss_dict[name] = "binary_crossentropy"
                loss_weights[name] = getattr(self.params, "direction_weight", 0.5)
                metrics_dict[name] = ["accuracy"]

            elif name == "hd":
                # cyclic mae for head direction (radians)
                loss_dict[name] = CyclicMAE(high=2 * np.pi)
                loss_weights[name] = getattr(self.params, "hd_weight", 1.0)
                metrics_dict[name] = [CyclicMAE(high=2 * np.pi, name="hd_mae")]

            elif name == "speed":
                loss_dict[name] = "mae"
                loss_weights[name] = getattr(self.params, "speed_weight", 0.5)
                metrics_dict[name] = ["mae"]

            elif name == "latent":
                loss_dict[name] = ContrastiveRegressionLoss(
                    temperature=getattr(self.params, "temperature", 0.1),
                    sigma=getattr(self.params, "sigma_contrastive", 0.1),
                    l_function_params=self.lfunction_layer_params,
                    **loss_kwargs,
                )
                loss_weights[name] = getattr(self.params, "contrastive_weight", 0.7)

        return loss_dict, loss_weights, metrics_dict

    def _build_model(self, **kwargs):
        ### Description of layers here
        with nnUtils.get_device_context(self.deviceName):
            self.inputsToSpikeNets = [
                tf.keras.layers.Input(
                    shape=(
                        None,  # maxNbOfSpikes in group
                        self.params.nChannelsPerGroup[group],
                        32,
                    ),
                    name="group" + str(group),
                )
                for group in range(self.params.nGroups)
            ]

            self.inputGroups = tf.keras.layers.Input(
                shape=(None,), name="groups", dtype=tf.int32
            )
            self.indices = [
                tf.keras.layers.Input(
                    shape=(None,), name="indices" + str(group), dtype=tf.int32
                )
                for group in range(self.params.nGroups)
            ]

            # Declare spike nets for the different groups:
            spikeNetClass = (
                SpikeNet1D
                if not getattr(self.params, "use_conv2d", False)
                else spikeNet
            )
            self.spikeNets = [
                tf.keras.layers.TimeDistributed(
                    spikeNetClass(
                        nChannels=self.params.nChannelsPerGroup[group],
                        device=self.deviceName,
                        nFeatures=self.params.nFeatures,
                        number=str(group),
                        batch_normalization=True,
                        reduce_dense=getattr(self.params, "reduce_dense", False),
                        no_cnn=getattr(self.params, "no_cnn", False),
                        name=f"spikeNet_{group}",
                    ),
                    name=f"timedist_spikeNet_{group}",
                )
                for group in range(self.params.nGroups)
            ]

            if getattr(self.params, "use_group_attention_fusion", True):
                # 2. Initialize the Group Attention Fusion layer
                self.group_fusion = GroupAttentionFusion(
                    n_groups=self.params.nGroups,
                    embed_dim=self.params.nFeatures,
                    num_heads=4,
                    device=self.deviceName,
                    name="group_fusion",
                )
            self.dropoutLayer = tf.keras.layers.Dropout(
                kwargs.get("dropoutCNN", self.params.dropoutCNN)
            )
            self.lstmdropOutLayer = tf.keras.layers.Dropout(
                kwargs.get("dropoutLSTM", self.params.dropoutLSTM)
            )

            # LSTMs
            self.isTransformer = kwargs.get(
                "isTransformer", getattr(self.params, "isTransformer", False)
            )
            if not self.isTransformer:
                self.lstmsNets = [
                    tf.keras.layers.LSTM(
                        self.params.lstmSize,
                        # if the layer is the last one, we return only the last output from the sequence (will be passed to a Dense layer afterwards)
                        # otherwise, we return the full sequence of outputs
                        return_sequences=(ilayer != self.params.lstmLayers - 1),
                    )
                    for ilayer in range(self.params.lstmLayers)
                ]
            else:
                self.isTransformer = True

                self.dim_factor = getattr(
                    self.params, "dim_factor", 1
                )  # factor to increase the dimension of the transformer if needed
                print("dim_factor:", self.dim_factor)
                print(
                    "project transformer:",
                    getattr(self.params, "project_transformer", True),
                )

                self.lstmsNets = (
                    [
                        PositionalEncoding(
                            d_model=self.params.nFeatures * self.dim_factor
                            if getattr(self.params, "project_transformer", True)
                            else self.params.nFeatures * self.params.nGroups,
                            # if we dont shrink the feature dimension before feeding to the transformer, we need to account for the nGroups factor
                            device=self.deviceName,
                        )
                    ]
                    + [
                        TransformerEncoderBlock(
                            d_model=self.params.nFeatures * self.dim_factor
                            if getattr(self.params, "project_transformer", True)
                            else self.params.nFeatures * self.params.nGroups,
                            num_heads=self.params.nHeads,
                            ff_dim1=self.params.ff_dim1,
                            ff_dim2=self.params.ff_dim2,
                            dropout_rate=self.params.dropoutLSTM,
                            device=self.deviceName,
                            residual=kwargs.get("transformer_residual", True),
                        )
                        for _ in range(self.params.lstmLayers)
                    ]
                    + [
                        MaskedGlobalAveragePooling1D(
                            device=self.deviceName, name="masking_pooling"
                        ),
                        # removed the activations in dense layers for better scaleability
                        tf.keras.layers.Dense(
                            int(self.params.TransformerDenseSize1),
                            kernel_regularizer="l2",
                        ),  # custom loss for heatmaps
                        tf.keras.layers.Dense(
                            int(self.params.TransformerDenseSize2),
                            kernel_regularizer="l2",
                        ),
                    ]
                )

            if getattr(self.params, "contrastive_loss", False):
                self.projector_head = tf.keras.layers.Dense(
                    self.params.nFeatures,
                    activation="relu",
                    name="contrastive_projector",
                    dtype="float32",  # ensure all output layers output float32, regardless of mixed precision policy, to avoid issues with loss scaling in mixed precision
                )

            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            self.epsilon = tf.constant(10 ** (-8))

            # Define named outputs heads based on target structure
            self.heads = {}
            for name, spec in self.target_structure.items():
                if name == "pos_2d" and getattr(self.params, "GaussianHeatmap", False):
                    # GaussianHeatmap is already defined in setup_gaussian_heatmap
                    continue
                else:
                    self.heads[name] = tf.keras.layers.Dense(
                        spec["dim"],
                        activation=spec["activation"],
                        name=name + "_dense",
                        kernel_regularizer="l2",
                        dtype="float32",
                    )

            # Outputs
            print("Output dimension:", self.params.dimOutput)

            self.dim_factor = getattr(
                self.params, "dim_factor", 1
            )  # factor to increase the dimension of the transformer if needed

            if getattr(self.params, "project_transformer", True):
                self.transformer_projection_layer = tf.keras.layers.Dense(
                    self.params.nFeatures * self.dim_factor,
                    activation="relu",
                    name="feature_projection_transformer",
                    dtype="float32",
                )
            self.ProjectionInMazeLayer = UMazeProjectionLayer(
                grid_size=kwargs.get(
                    "grid_size", getattr(self.params, "GaussianGridSize", (40, 40))
                ),
                dtype="float32",
            )

            # Gather the full model
            self.generate_kwargs = kwargs
            outputs = self.generate_model(**kwargs)
            # Build two models
            # One just described, with two objective functions corresponding
            # to both position and predicted losses
            self.model = self.compile_model(
                outputs, modelName="FullModel.pdf", **kwargs
            )
            # TODO: add option to add independent losses to the model ?

            # In theory, the predicted loss could be not learning enough in the first network (optional)
            # Second only with loss corresponding to predicted loss
            if kwargs.get("isPredLoss", False):
                self.predLossModel = self.compile_model(
                    outputs, predLossOnly=True, modelName="predLossModel.pdf", **kwargs
                )

    def rebuild_model(self, **kwargs):
        """
        Regenerate the model with the current parameters.

        Returns
        -------
        None
        """
        self.clear_session()
        outputs = self.generate_model(**kwargs)
        self.model = self.compile_model(outputs, modelName="FullModel.pdf", **kwargs)
        if kwargs.get("isPredLoss", False):
            self.predLossModel = self.compile_model(
                outputs, predLossOnly=True, modelName="predLossModel.pdf", **kwargs
            )

    def change_batch_size(self, new_batch_size, **kwargs):
        """
        Change the batch size of the model.

        Parameters
        ----------
        new_batch_size : int
            The new batch size to set.

        Returns
        -------
        None
        """
        default_kwargs = self.generate_kwargs.copy()
        default_kwargs["batch_size"] = new_batch_size
        default_kwargs.update(kwargs)
        self.rebuild_model(**default_kwargs)

    def apply_transformer_architecture(
        self, allFeatures, allFeatures_raw, mymask, **kwargs
    ):
        """
        Shared transformer logic that can be called from both generate_model and extract_transformer_model.
        This ensures the transformer architecture is defined only once.

        Args:
            allFeatures: Features after dropout (batch_size, seq_len (ie maxNbOfSpikes), feature_dim * nGroups)
            allFeatures_raw: Raw features before dropout (for sumFeatures calculation)
            mymask: Attention mask (batch_size, seq_len)
            **kwargs: Additional arguments

        Returns:
            tuple: (myoutputPos, output, sumFeatures)
            myoutputPos: Final output positions or heatmaps (batch_size, dimOutput) or (batch_size, GaussianGridSize[0], GaussianGridSize[1]) or (batch_size, flattened heatmap + dimOutput - 2)
            output: Output before final dense layers (batch_size, TransformerDenseSize2)
            sumFeatures: Sum of masked raw features (batch_size, feature_dim * nGroups)
        """

        output = None
        outputSeq = None

        masked_features_layer = MaskingLayer(name="masking_layer_transformer")
        masked_features_layer.supports_masking = True
        masked_features = masked_features_layer([mymask, allFeatures_raw])

        if (
            getattr(self.params, "project_transformer", True)
            and self.params.nFeatures * self.dim_factor
            != self.params.nFeatures * self.params.nGroups
        ):
            # 1. Projection layer
            allFeatures = self.transformer_projection_layer(allFeatures)
            sumFeatures = kops.sum(
                self.transformer_projection_layer(masked_features), axis=1
            )
        else:
            sumFeatures = kops.sum(masked_features, axis=1)

        # 2. Positional encoding
        allFeatures = self.lstmsNets[0](allFeatures)
        output = allFeatures

        # 3. Transformer blocks with residual connections
        for ilstm, transformerLayer in enumerate(self.lstmsNets[1:-3]):
            if ilstm == 0:
                if (
                    len(self.lstmsNets) == 5
                ):  # num of transformer layers + one positional encoding + one pooling layer + 2 dense layers == 4 + #(transformer layers)
                    output = transformerLayer(allFeatures, mask=mymask)
                else:
                    outputSeq = transformerLayer(allFeatures, mask=mymask)
                    # residual connections between transformer layers
                    outputSeq = outputSeq + allFeatures
            elif (
                ilstm == len(self.lstmsNets) - 5
            ):  # last transformer layer before pooling
                prevSeq = outputSeq
                output = transformerLayer(outputSeq, mask=mymask)
                # residual connections between transformer layers
                output = output + prevSeq
            else:
                prevSeq = outputSeq
                outputSeq = transformerLayer(outputSeq, mask=mymask)
                outputSeq = outputSeq + prevSeq

        # 4. Pooling and final dense layers
        # size [batch, max_nspikes, nFeatures]
        output = self.lstmsNets[-3](
            output, mask=mymask
        )  # pooling (size [batch, nFeatures*nGroups])

        h = self.lstmsNets[-2](
            output
        )  # dense layer after pooling (size [batch, TransformerDenseSize1])

        if getattr(self.params, "contrastive_loss", False):
            final_output = self.projector_head(
                h
            )  # projection for contrastive loss (size [batch, nFeatures])
        else:
            final_output = None

        x = self.lstmsNets[-1](
            h
        )  # another dense layer after pooling (size [batch, TransformerDenseSize2])

        return x, final_output, sumFeatures

    def apply_lstm_architecture(self, allFeatures, sumFeatures, mymask, **kwargs):
        """
        Shared lstm logic that can be called from generate_model.
        This ensures the lstm architecture is defined only once.

        Args:
            allFeatures: Features after dropout (batch_size, seq_len, feature_dim)
            mymask: Attention mask (batch_size, seq_len)
            **kwargs: Additional arguments

        Returns:
            tuple: (myoutputPos, outputPredLoss, output, sumFeatures)
        """

        output = None
        outputSeq = None

        # Ensure output is assigned if lstmLayers is 0
        output = allFeatures

        # LSTM blocks with masking and dropout
        for ilstm, lstmLayer in enumerate(self.lstmsNets):
            if ilstm == 0:
                if len(self.lstmsNets) == 1:
                    output = lstmLayer(allFeatures, mask=mymask)
                else:
                    outputSeq = lstmLayer(allFeatures, mask=mymask)
                    outputSeq = self.lstmdropOutLayer(outputSeq)
            elif ilstm == len(self.lstmsNets) - 1:
                output = lstmLayer(outputSeq, mask=mymask)
            else:
                outputSeq = lstmLayer(outputSeq, mask=mymask)
                outputSeq = self.lstmdropOutLayer(outputSeq)

        final_output = None

        return output, final_output, sumFeatures

    def generate_model(self, **kwargs):
        """
        Generate the full model with the CNN, LSTM and Dense layers.

        Returns
        -------
        myoutputPos, outputPredLoss, posLoss, uncertaintyLoss
        """
        # CNN plus dense on every group independently
        with nnUtils.get_device_context(self.deviceName):
            allFeatures = []  # store the result of the CNN computation for each group
            for group in range(self.params.nGroups):
                # Process all spikes in the group through the spike net
                x = self.spikeNets[group](self.inputsToSpikeNets[group])
                # x shape: (batch, n_spikes_in_group, nFeatures)

                # The "Null Spike" Trick:
                full_emb = AddNullSpike(
                    name=f"null_spike_g{group}",
                )(x)

                # Gather spikes into the global sequence: (batch, seqLen, nFeatures)
                gather_layer = GatherSpikes(
                    name=f"gather_g{group}",
                )
                gather_layer.supports_masking = True
                gathered = gather_layer([full_emb, self.indices[group]])

                allFeatures.append(gathered)

            if not getattr(self.params, "use_group_attention_fusion", True):
                # OLD: concatenation over features
                allFeatures = kops.concatenate(allFeatures, axis=2)
            else:
                print("Using Group Attention Fusion for feature fusion")
                # NEW: Group Attention Fusion over groups
                # Create masks based on indices (0 means no spike)
                group_masks = [
                    kops.not_equal(self.indices[group], 0)
                    for group in range(self.params.nGroups)
                ]
                fusion_mask = kops.stack(group_masks, axis=-1)
                allFeatures = self.group_fusion(allFeatures, mask=fusion_mask)

            # now the shape of allfeatures is (NbBatch, NbTotSpikeDetected, nGroups*nFeatures)
            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            mymask_layer = SafeMaskCreation(name="safe_mask_creation")
            mymask = mymask_layer(self.inputGroups)

            masking_layer = MaskingLayer(name="masking_layer_before_rnn")
            masking_layer.supports_masking = True
            masked_features = masking_layer([mymask, allFeatures])
            # size is (NbBatch, NbTotSpikeDetected, nGroups*nFeatures)
            sumFeatures = kops.sum(
                masked_features, axis=1
            )  # This var will be used in the predLoss loss

            allFeatures_raw = allFeatures
            allFeatures = self.dropoutLayer(allFeatures)
            # size is (NbBatch, NbTotSpikeDetected, nGroups*nFeatures)

            # LSTM / Transformer
            if not self.isTransformer:
                x, latent, sumFeatures = self.apply_lstm_architecture(
                    allFeatures, sumFeatures, mymask, **kwargs
                )
            else:
                x, latent, sumFeatures = self.apply_transformer_architecture(
                    allFeatures, allFeatures_raw, mymask, **kwargs
                )

            # 5. Create final heads branching from x
            outputs = {}

            for name, layer in self.heads.items():
                out = layer(x)
                if name == "pos_2d" and "pos" in self.params.target.lower():
                    # Check if heatmap or raw regression
                    if not getattr(self.params, "GaussianHeatmap", False):
                        out = self.ProjectionInMazeLayer(out)

                outputs[name] = keras.layers.Identity(name=name, dtype="float32")(out)

            # Special case for GaussianHeatmap if enabled for pos_2d
            if (
                getattr(self.params, "GaussianHeatmap", False)
                and "pos_2d" in self.target_structure
            ):
                # Use the special GaussianHeatmap layer
                # simply a kernel convolution with a fixed gaussian kernel, applied to the output of the dense layer for pos_2d
                # before it also had a dense layer
                out_heatmap = self.GaussianHeatmap(x)
                outputs["pos_2d"] = keras.layers.Identity(
                    name="pos_2d", dtype="float32"
                )(out_heatmap)

            if latent is not None:
                latent = keras.layers.Identity(name="latent", dtype="float32")(latent)
                outputs["latent"] = latent

        return outputs

    def compile_model(
        self,
        outputs,
        modelName="FullModel.pdf",
        predLossOnly=False,
        jit_compile=False,
        **kwargs,
    ):
        """
        Compile the model with the desired losses and optimizer.
        The model is then plotted and saved in the results folder.

        Parameters
        ----------
        outputs : dict of tensors
        modelName : str (default "FullModel.png")
        predLossOnly : bool (default False)

        Returns
        -------
        model : tf.keras.Model
        """
        # Initialize and plot the model
        model = tf.keras.Model(
            inputs=self.inputsToSpikeNets + self.indices + [self.inputGroups],
            outputs=outputs,
        )

        # Compile the model
        # TODO: use params.optimizer instead of hardcoding RMSprop
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=kwargs.get("lr", self.params.learningRates[0]),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-04,
            weight_decay=getattr(self.params, "weight_decay", 1e-4),
            global_clipnorm=getattr(self.params, "global_clipnorm", 1.0),
        )
        # TODO: something with mixed precision and keras policy ?

        # Full model
        self.outNames = list(outputs.keys())

        # Filter kwargs for loss initialization - remove everything except name and reduction
        known_loss_args = ["name", "reduction"]
        loss_kwargs = {k: v for k, v in kwargs.items() if k in known_loss_args}

        loss_dict, loss_weights, metrics_dict = self._parse_loss_and_metrics_dict(
            **loss_kwargs
        )

        if predLossOnly:
            # compile for predLoss only if requested
            # For now, we reuse the same logic but filters could be applied if needed
            model.compile(
                optimizer=self.optimizer,
                loss=loss_dict,
                loss_weights=loss_weights,
                metrics=metrics_dict,
                jit_compile=jit_compile,
            )
        else:
            model.compile(
                optimizer=self.optimizer,
                loss=loss_dict,
                loss_weights=loss_weights,
                metrics=metrics_dict,
                jit_compile=jit_compile,
            )
            # Get internal names of losses
        if (
            not os.path.exists(os.path.join(self.projectPath.experimentPath, modelName))
            or 1 < 2
        ):
            try:
                tf.keras.utils.plot_model(
                    model,
                    to_file=(os.path.join(self.projectPath.experimentPath, modelName)),
                    show_shapes=True,
                    show_layer_names=True,
                )
            except Exception as e:
                print("Could not plot the model:", e)
        return model

    def train(
        self,
        behaviorData,
        **kwargs,
    ):
        """
        Train the network on the dataset.
        The training is done in two steps:
        - First we train the full model on the position loss and the uncertainty loss
        - Then we train the loss predictor model on the predicted loss

        Parameters
        ----------
        behaviorData : dict of arrays containing the times, the feature True...
        onTheFlyCorrection : bool (default False) : normalize the position data on the fly
        windowSizeMS : int (default 36) : size of the window in milliseconds
        scheduler : str (default "decay") : scheduler type to use for the learning rate
        isPredLoss : bool (default True) : whether to train the loss predictor model
        earlyStop : bool (default False) : whether to use early stopping during training
        load_model : bool (default False) : whether to load a previously trained model if it exists
        **kwargs : dict, optional
            Additional parameters for the training, such as batch size, scheduler, learning rate, load_model etc.
            l_function : func, needed for dense weight regularization

        Returns
        -------
        None
        """

        ### Create neccessary arrays
        windowSizeMS = kwargs.pop("windowSizeMS", 36)
        scheduler = kwargs.get("scheduler", "cosine")
        isPredLoss = kwargs.get("isPredLoss", False)
        earlyStop = kwargs.get("earlyStop", False)
        strideFactor = kwargs.get("strideFactor", 1)

        load_model = kwargs.get("load_model", False)
        if not isinstance(windowSizeMS, int):
            windowSizeMS = int(windowSizeMS)

        epochMask = {}
        totMask = {}
        csvLogger = {}
        checkpointPath = {}
        new_checkpointPath = {}

        # Manage folders
        os.makedirs(os.path.join(self.folderModels, str(windowSizeMS)), exist_ok=True)
        os.makedirs(os.path.join(self.folderResult, str(windowSizeMS)), exist_ok=True)
        os.makedirs(
            os.path.join(self.folderResultSleep, str(windowSizeMS)), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.folderModels, str(windowSizeMS), "full"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.folderModels, str(windowSizeMS), "savedModels"),
            exist_ok=True,
        )
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            os.makedirs(
                os.path.join(self.folderModels, str(windowSizeMS), "predLoss"),
                exist_ok=True,
            )
            csvLogger["predLoss"] = tf.keras.callbacks.CSVLogger(
                os.path.join(
                    self.folderModels,
                    str(windowSizeMS),
                    "predLoss",
                    "predLossmodel.log",
                )
            )
        # Manage callbacks
        csvLogger["full"] = tf.keras.callbacks.CSVLogger(
            os.path.join(self.folderModels, str(windowSizeMS), "full", "fullmodel.log")
        )
        for key in csvLogger.keys():
            checkpointPath[key] = os.path.join(
                self.folderModels, str(windowSizeMS), key + "/cp.ckpt"
            )
            new_checkpointPath[key] = os.path.join(
                self.folderModels,
                str(windowSizeMS),
                key + "/cp.weights.h5",
            )

        ## Get speed filter:
        speedMask = behaviorData["Times"]["speedFilter"]

        ## Get datasets
        if strideFactor > 1:
            filename = (
                f"dataset_stride{str(windowSizeMS)}_factor{str(strideFactor)}.tfrec"
            )
        else:
            filename = f"dataset_stride{str(windowSizeMS)}.tfrec"

        # Manage masks
        epochMask["train"] = inEpochsMask(
            behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
        )
        epochMask["test"] = inEpochsMask(
            behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
        )
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            epochMask["predLoss"] = inEpochsMask(
                behaviorData["positionTime"][:, 0],
                behaviorData["Times"]["lossPredSetEpochs"],
            )
        for key in epochMask.keys():
            totMask[key] = speedMask * epochMask[key]

        augmentation_config = NeuralDataAugmentation(device=self.deviceName, **kwargs)
        datasets, counts = self._dataset_loading_pipeline(
            filename, windowSizeMS, behaviorData, totMask, augmentation_config, **kwargs
        )
        if kwargs.get("return_datasets", False):
            return datasets, counts

        if counts is not None:
            # means we are augmenting the data on the fly, so we can visualize the distribution of the data and compute the balanced size after augmentation
            import termplotlib as tpl

            count_x, bin_edges = np.histogram(counts["train"], bins=40)
            fig = tpl.figure()
            fig.hist(
                count_x,
                bin_edges,
                grid=[15, 25],
                force_ascii=False,
            )
            fig.show()
            max_count = counts["train"].max()
            # we compute the balanced size, ie the size of the dataset after resampling if it was perfectly uniform
            print("Max count per bin in training set:", max_count)
            num_allowed_bins = np.sum(counts["train"] > 0)
            print("total num of allowed bins in training set:", num_allowed_bins)
            balanced_size = max_count * num_allowed_bins
            print("Balanced dataset size would be:", balanced_size)
            print(
                "Original training dataset size:",
                self.GaussianHeatmap.training_positions.shape[0],
            )
            n_aug = (
                kwargs.get("num_augmentations", 1)
                if kwargs.get("use_augmentation", False) or self.params.dataAugmentation
                else 1
            )

            # If keeping original, we have n_aug + 1 samples total
            n_total_aug = n_aug + 1 if kwargs.get("keep_original", True) else n_aug
            # In your main pipeline where you calculate steps_per_epoch:
            actual_rep_factors = np.minimum(
                max_count / np.maximum(counts["train"], 1e-8), 20.0
            )
            actual_balanced_size = np.sum(counts["train"] * actual_rep_factors)

            steps_per_epoch = np.floor(
                (actual_balanced_size * n_total_aug) / (self.params.batch_size)
            )
        else:
            print(
                "no data augmentation or class balancing, using original dataset size for steps per epoch calculation"
            )
            num_train_samples = np.sum(epochMask["train"])
            steps_per_epoch = np.floor(num_train_samples / self.params.batch_size)

        print("Steps per epoch:", steps_per_epoch)

        ### Train the model(s)
        # Train
        for key in checkpointPath.keys():
            print("Training the", key, "model")
            nb_epochs_already_trained = 0
            loaded = False
            managed_to_convert = (
                True  # this way new checkpoints will be saved in keras3 format
            )

            if load_model and os.path.exists(os.path.dirname(checkpointPath[key])):
                if key != "predLoss":
                    print(
                        "Loading the weights of the loss training model from",
                        checkpointPath[key],
                    )
                    try:
                        self.model.load_weights(
                            new_checkpointPath[key]
                            if managed_to_convert
                            else checkpointPath[key]
                        )
                        csv_hist = pd.read_csv(
                            os.path.join(
                                self.folderModels,
                                str(windowSizeMS),
                                "full",
                                "fullmodel.log",
                            )
                        )
                        nb_epochs_already_trained = csv_hist["epoch"].max() + 1
                        print("nb_epochs_already_trained =", nb_epochs_already_trained)
                        loaded = True
                    except Exception:
                        try:
                            self.model.load_weights(checkpointPath[key])
                            csv_hist = pd.read_csv(
                                os.path.join(
                                    self.folderModels,
                                    str(windowSizeMS),
                                    "full",
                                    "fullmodel.log",
                                )
                            )
                            nb_epochs_already_trained = csv_hist["epoch"].max() + 1
                            print(
                                "nb_epochs_already_trained =", nb_epochs_already_trained
                            )
                            loaded = True
                        except Exception as e:
                            print(
                                "Error loading weights for",
                                key,
                                "from",
                                checkpointPath[key],
                                "or",
                                new_checkpointPath[key],
                                ":",
                                e,
                            )

            if loaded:
                print(
                    "loaded weights for",
                    key,
                    "model. Fine tune is set to",
                    kwargs.get("fine_tune", False),
                )
                if (
                    os.path.exists(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "full",
                            "fullModelLosses.png",
                        )
                    )
                    or os.path.exists(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "predLoss",
                            "predLossModelLosses.png",
                        )
                    )
                ) and not kwargs.get("fine_tune", False):
                    print(
                        "Loading previous losses from",
                        os.path.join(self.folderModels, str(windowSizeMS)),
                    )
                    continue
                if not kwargs.get("fine_tune", False):
                    print(f"Model loaded for {key}, skipping directly to next.")
                    continue

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=new_checkpointPath[key]
                if managed_to_convert
                else checkpointPath[key],
                save_weights_only=True,
                verbose=1,
            )
            # Manage learning rates schedule
            if loaded and kwargs.get("fine_tune", False):
                print("Fine-tuning the model with a lower learning rate, set to 0.0005")
                self.model.optimizer.learning_rate.assign(0.0005)
            elif loaded:
                print("Loading the model with the initial learning rate")
                self.model.optimizer.learning_rate.assign(self.params.learningRates[0])
            LRScheduler = self.LRScheduler(
                lrs=self.params.learningRates,
                total_epochs=self.params.nEpochs,
                warmup_epochs=kwargs.get("warmup_epochs", 6),
                min_lr=kwargs.get("min_lr", 1e-6),
            )
            if scheduler == "fixed":
                schedule = tf.keras.callbacks.LearningRateScheduler(
                    LRScheduler.schedule_fixed
                )
            elif scheduler == "decay":
                schedule = tf.keras.callbacks.LearningRateScheduler(
                    LRScheduler.schedule_decay
                )
            elif scheduler == "cosine":
                schedule = tf.keras.callbacks.LearningRateScheduler(
                    LRScheduler.schedule_cosine_warmup
                )
            else:
                raise ValueError(
                    'Learning rate schedule is either "fixed", "decay" or "cosine"'
                )

            # NOTE: In case you need debugging, toggle this profiling line to True
            is_tbcallback = kwargs.get("tensorboard_callback", True)
            if self.debug:
                print("Debugging mode is ON")
                if is_tbcallback:
                    print("enabling TensorBoard callback and device placement logging")
                    tf.debugging.set_log_device_placement(True)
                if key != "predLoss":
                    ann_config = {
                        k: v
                        for k, v in self.params.__dict__.items()
                        if not k.startswith("_")
                        and not callable(v)
                        and not isinstance(v, (list, dict, set))
                        and not isinstance(v, np.ndarray)
                        and not isinstance(v, tf.Tensor)
                        and not isinstance(v, DataHelper)
                        and len(str(v)) < 1000
                    }

                    ann_config["loaded"] = loaded

                    prefix = "LOADED_" if loaded else ""
                    if is_tbcallback:
                        # wandb.tensorboard.patch(
                        #     root_logdir=os.path.join(self.folderResult, "logs")
                        # )
                        # tf.profiler.experimental.start(
                        #     os.path.join(self.folderResult, "logs")
                        # )

                        from datetime import datetime

                        log_dir = os.path.join(
                            self.folderResult,
                            "logs",
                            str(windowSizeMS),
                            key,
                            datetime.now().strftime("%Y%m%d-%H%M%S"),
                        )
                        tb_callbacks = tf.keras.callbacks.TensorBoard(
                            log_dir=log_dir,
                            histogram_freq=1,
                            profile_batch=(10, 100),
                        )
                        print(f"starting tensorboard at {log_dir}")
                    run = wandb.init(
                        entity="touseul",
                        project="ContrastiveLossNew",
                        name=f"{prefix}{os.path.basename(os.path.dirname(self.projectPath.xml))}_{os.path.basename(self.projectPath.experimentPath)}_{key}_{windowSizeMS}ms",
                        notes=f"{os.path.basename(self.projectPath.experimentPath)}_{key}",
                        # sync_tensorboard=True,
                        config=ann_config,
                    )

                    wandb_callback = WandbMetricsLogger()
            if key != "predLoss":
                if earlyStop:
                    start_from_epoch = (
                        max(self.params.earlyStop_start - nb_epochs_already_trained, 2),
                    )
                    print(
                        f"will use early stopping starting from epoch {start_from_epoch}"
                    )
                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=2,
                        min_delta=0.05,
                        verbose=1,
                        restore_best_weights=True,
                        start_from_epoch=max(
                            self.params.earlyStop_start - nb_epochs_already_trained, 2
                        ),
                    )
                    callbacks = [
                        csvLogger[key],
                        cp_callback,
                        schedule,
                        es_callback,
                        MemoryUsageCallbackExtended(),
                        ContrastiveMonitor(),
                    ]
                else:
                    callbacks = [
                        csvLogger[key],
                        cp_callback,
                        schedule,
                        MemoryUsageCallbackExtended(),
                        ContrastiveMonitor(),
                    ]

                if self.params.reduce_lr_on_plateau:
                    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.8,
                        patience=10,
                        verbose=1,
                        start_from_epoch=40 - nb_epochs_already_trained,
                    )
                    callbacks.append(reduce_lr_callback)

                if self.debug:
                    if is_tbcallback:
                        callbacks.append(tb_callbacks)
                    callbacks.append(wandb_callback)

                hist = self.model.fit(
                    datasets["train"],
                    epochs=self.params.nEpochs - nb_epochs_already_trained,
                    callbacks=callbacks,  # , tb_callback,cp_callback
                    validation_data=datasets["test"],
                    steps_per_epoch=int(steps_per_epoch),
                )
                self.trainLosses[key] = np.transpose(
                    np.stack(
                        [
                            hist.history["loss"],  # tf_op_layer_lossOfManifold
                        ]
                    )
                )  # tf_op_layer_lossOfLossPredictor_loss
                valLosses = np.transpose(
                    np.stack(
                        [
                            hist.history["val_loss"],  # tf_op_layer_lossOfManifold
                        ]
                    )
                )
                self.losses_fig(
                    self.trainLosses[key],
                    os.path.join(self.folderModels, str(windowSizeMS)),
                    valLosses=valLosses,
                )
                self.model.save_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "full_cp.weights.h5",
                    ),
                )
                try:
                    self.model.save(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "savedModels",
                            "full_model.keras",
                        )
                    )
                except Exception as e:
                    print("Could not save the full model:", e)
                if self.debug:
                    # wandb.tensorboard.unpatch()
                    run.finish()

    def _dataset_loading_pipeline(
        self,
        filename: str,
        windowSizeMS: int,
        behaviorData: Dict,
        totMask,
        augmentation_config: Optional[NeuralDataAugmentation] = None,
        **kwargs,
    ) -> Tuple[Dict[str, tf.data.Dataset], Optional[Dict[str, np.ndarray]]]:
        """
        Create the dataset loading pipeline. It includes parsing, filtering, batching, data augmentation and prefetching. If oversampling resampling is enabled, it is also applied here and returns the counts for each position bin.

        Parameters
        ----------
        filename : str
            The name of the TFRecord file containing the dataset.
        windowSizeMS : int
            The size of the window in milliseconds.
        behaviorData : dict
            Dictionary containing behavioral data such as positions and times.
        totMask : dict or list
            Mask to filter the dataset based on speed and epochs.
        augmentation_config : NeuralDataAugmentation, Optional
            Configuration for data augmentation.
        **kwargs : dict, Optional
            Additional parameters such as onTheFlyCorrection, shuffle, batch_size, inference_mode, extract_spikes_counts.

        Returns
        -------
        datasets : dict
            Dictionary containing the training and testing datasets.
        counts : dict of np.ndarray or None
            Dict of arrays containing counts for oversampling resampling, or None if not applicable.
        """
        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)
        shuffle = kwargs.get("shuffle", True)
        random_spiking = kwargs.get("random_spiking", False)
        batch_size = kwargs.get("batch_size", self.params.batch_size)
        speedMask = kwargs.get("speedMask", None)
        inference_mode = kwargs.get("inference_mode", False)
        if inference_mode and shuffle:
            raise ValueError(
                "Shuffle should be set to False in inference mode to ensure deterministic outputs."
            )

        # Create datasets
        if not isinstance(totMask, dict):
            # it means we have just one set of keys
            totMask_backup = totMask.copy()
            totMask = (
                {"test": totMask_backup}
                if inference_mode
                else {"train": totMask_backup}
            )
        if speedMask is not None and not isinstance(speedMask, dict):
            # it means we have just one set of keys
            speedMask_backup = speedMask.copy()
            speedMask = (
                {"test": speedMask_backup}
                if inference_mode
                else {"train": speedMask_backup}
            )

        def get_mask_filter(totMask_for_key):
            mask_tensor = tf.constant(totMask_for_key, dtype=tf.float32)

            @tf.function
            def filter_by_pos_index(x):
                # return tf.equal(table.lookup(x["pos_index"]), 1.0)
                pos_index = x["pos_index"]
                return tf.equal(tf.gather(mask_tensor, pos_index), 1.0)

            return filter_by_pos_index

        def filter_nan_pos(x):
            pos_data = x["pos"]

            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        @tf.function
        def _parse_function(*vals):
            with nnUtils.get_device_context(self.deviceName):
                return nnUtils.parse_serialized_spike(self.featDesc, *vals)

        @tf.function
        def map_outputs(vals):
            # Move 'pos' to targets, rest stay in inputs
            inputs_dict = {k: v for k, v in vals.items() if k != "pos"}
            # Structured targets matching model outputs
            targets_dict = {}
            for name, spec in self.target_structure.items():
                start_idx, end_idx = spec["slice"]
                targets_dict[name] = vals["pos"][:, start_idx:end_idx]

            # latent targets are the 2D position for contrastive regression
            # TODO: ensure that contrastive regression is always wrt the 2D position
            if "latent" in self.outNames:
                targets_dict["latent"] = vals["pos"][:, :2]

            return (inputs_dict, targets_dict)

        def create_indices(vals):
            return self.create_indices(vals, shuffle=random_spiking)

        ndataset = tf.data.TFRecordDataset(
            os.path.join(self.projectPath.dataPath, filename),
            buffer_size=100 * 1024 * 1024,  # 100MB read buffer
        )

        if shuffle:
            print("Shuffling the dataset")
            ndataset = ndataset.shuffle(
                10000
            )  # move shuffle before parsing for better randomness, we can afford a bigger buffer because we are not yet in the batched stage

        # Parse the record into tensors - simply attribute a name to every tensor from featDesc
        ndataset = ndataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ndataset = ndataset.prefetch(tf.data.AUTOTUNE)

        datasets = {}
        counts = {}
        for key in totMask.keys():
            # This is just max normalization to use if the behavioral data have not been normalized yet
            if onTheFlyCorrection:
                maxPos = np.nanmax(
                    behaviorData["Positions"][
                        np.logical_not(
                            np.isnan(np.sum(behaviorData["Positions"], axis=1))
                        )
                    ]
                )
                posFeature = behaviorData["Positions"] / maxPos
            else:
                posFeature = behaviorData["Positions"]

            # posFeature is already of shape (N,dimOutput) because we ran data_helper.get_true_target before.
            filter_op = get_mask_filter(totMask[key])
            dataset = ndataset.filter(filter_op)
            dataset = dataset.map(nnUtils.import_true_pos(posFeature))
            dataset = dataset.filter(filter_nan_pos).prefetch(tf.data.AUTOTUNE)

            # now that we have clean positions, we can resample if needed
            if self.params.OversamplingResampling and key == "train":
                dataset, count_tmp = self._apply_oversampling_resampling(
                    dataset, windowSizeMS=windowSizeMS, shuffle=shuffle
                )

            def parse_serialized_sequence(vals):
                return nnUtils.parse_serialized_sequence(
                    self.params,
                    vals,
                    count_spikes=kwargs.get("extract_spikes_counts", False),
                    max_spikes=300,
                    max_spikes_per_group=100,
                    # sorted_indices = #TODO: at some point
                )

            # Map create_indices BEFORE batching and data augmentation (per-example)
            dataset = dataset.map(
                parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(create_indices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            # 7. Optimized Detailed Parsing / Augmentation
            is_aug_active = (
                self.params.dataAugmentation
                and key != "test"
                and not kwargs.get("inference_mode", False)
            )

            if is_aug_active:
                optimized_fn = self.create_optimized_parse_function(
                    augmentation=is_aug_active,
                    augmentation_config=augmentation_config if is_aug_active else None,
                    count_spikes=kwargs.get("extract_spikes_counts", False),
                )
                print(f"Applying data augmentation to {key} dataset")
                dataset = dataset.interleave(
                    lambda x: tf.data.Dataset.from_tensor_slices(optimized_fn(x)),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    cycle_length=64,
                    block_length=11,
                    deterministic=False,
                )

            padded_shapes, padding_values = self._get_padding_shapes_values(
                extract_spikes_counts=kwargs.get("extract_spikes_counts", False),
            )

            boundaries = [30, 100]  # TODO: adapt on winMS
            batch_sizes = [batch_size] * (len(boundaries) + 1)

            dataset = dataset.prefetch(buffer_size=2000)
            dataset = dataset.bucket_by_sequence_length(
                element_length_func=lambda x: x["max_spikes"],
                bucket_boundaries=boundaries,
                bucket_batch_sizes=batch_sizes,
                padded_shapes=padded_shapes,
                padding_values=padding_values,
                drop_remainder=True,
            )

            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)

            if key != "test":
                # handle dataset ran out of data by repeating it
                dataset = dataset.repeat()

            options = self._get_dataset_options()
            dataset = dataset.with_options(options).prefetch(tf.data.AUTOTUNE)

            datasets[key] = dataset
            counts[key] = (
                count_tmp
                if self.params.OversamplingResampling and key == "train"
                else None
            )

        # Save parsed datasets to new TFRecord files if requested
        save_parsed_tfrec = kwargs.get("save_parsed_tfrec", None)
        save_parsed_parquet = kwargs.get("save_parsed_parquet", None)

        # The user mentioned useSpeedMask, but the code often uses useSpeedFilter.
        # We handle both, defaulting to the condition that saving only happens when speed masking is NOT applied during loading (i.e. we want the "raw" but cropped data).
        useSpeedMask = kwargs.get("useSpeedMask", kwargs.get("useSpeedFilter", False))
        should_save = not useSpeedMask

        if save_parsed_tfrec is not None:
            if should_save:
                print(
                    f"Saving parsed datasets to TFRecord files with base path: {save_parsed_tfrec}"
                )
                self._save_datasets_to_tfrec(datasets, save_parsed_tfrec)
            else:
                print(
                    f"Skipping TFRecord saving because speed masking is active (useSpeedMask/Filter={useSpeedMask})"
                )

        if save_parsed_parquet is not None:
            if should_save:
                print(
                    f"Saving parsed datasets to Parquet files with base path: {save_parsed_parquet}"
                )
                self._save_datasets_to_parquet(datasets, save_parsed_parquet)
            else:
                print(
                    f"Skipping Parquet saving because speed masking is active (useSpeedMask/Filter={useSpeedMask})"
                )

        return datasets, counts if self.params.OversamplingResampling else None

    def _get_padding_shapes_values(self, extract_spikes_counts=False):
        # Pad and Batch logic
        padded_shapes = {
            "pos_index": [],
            "pos": [self.params.dimOutput],
            "length": [],
            "groups": [self.max_nb_spikes],
            "time": [],
            "time_behavior": [],
            "indexInDat": [self.max_nb_spikes],
            "max_spikes": [],
        }
        padding_values = {
            "pos_index": tf.constant(-1, dtype=tf.int64),
            "pos": tf.constant(-1.0, dtype=tf.float64),
            "length": tf.constant(-1, dtype=tf.int64),
            "groups": tf.constant(-1, dtype=tf.int64),
            "time": tf.constant(-1.0, dtype=tf.float32),
            "time_behavior": tf.constant(-1.0, dtype=tf.float32),
            "indexInDat": tf.constant(-1, dtype=tf.int64),
            "max_spikes": tf.constant(-1, dtype=tf.int32),
        }
        if extract_spikes_counts:
            for g in range(self.params.nGroups):
                padded_shapes[f"group{g}_spikes_count"] = []
                padding_values[f"group{g}_spikes_count"] = tf.constant(
                    0, dtype=tf.int32
                )

        for g in range(self.params.nGroups):
            padded_shapes[f"group{g}"] = [
                self.max_spikes_per_group,  # spikes
                self.params.nChannelsPerGroup[g],
                32,
            ]
            padded_shapes[f"indices{g}"] = [self.max_spikes_per_group]
            padding_values[f"group{g}"] = tf.constant(-1.0, dtype=tf.float32)
            padding_values[f"indices{g}"] = tf.constant(0, dtype=tf.int32)

        return padded_shapes, padding_values

    def _get_dataset_options(self):
        # We shuffle the datasets and cache it - this way the training samples are randomized for each epoch
        # and each mini-batch contains a representative sample of the training set.
        # nSteps represent the buffer size of the shuffle operation - 10 seconds worth of buffer starting
        # from the 0-timepoint of the dataset.
        # once an element is selected, its space in the buffer is replaced by the next element (right after the 10s window...)
        # At each epoch, the shuffle order is different.
        # smaller buffer for batched data
        # were talking in number of batches here, not time (so it does not make sense to use params.nSteps)
        # prefetch entire batches
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.filter_fusion = True
        options.experimental_optimization.noop_elimination = True
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        # get max num of cpu cores minus 1 for data loading
        options.threading.private_threadpool_size = max(1, os.cpu_count() - 1)
        options.threading.max_intra_op_parallelism = 1
        return options

    def load_parsed_dataset(
        self,
        base_path: str,
        keys: List[str] = ["train", "test"],
        featDesc: Optional[Dict] = None,
        dimOutput: Optional[int] = None,
    ) -> Dict[str, tf.data.Dataset]:
        """
        Load datasets that were previously saved using _save_datasets_to_tfrec.

        Parameters
        ----------
        base_path : str
            Base path for the TFRecord files.
        keys : list of str
            The dataset keys to load (e.g., ['train', 'test']).
        featDesc : dict, optional
            The feature description to use for parsing. If None, uses a default that
            handles variable position dimensions.

        Returns
        -------
        datasets : dict
            Dictionary of loaded tf.data.Dataset objects.
        """
        if dimOutput is not None:
            self.params.dimOutput = dimOutput

        if featDesc is None:
            # Default featDesc that handles variable length pos and groups
            featDesc = {
                "pos_index": tf.io.FixedLenFeature([], tf.int64),
                "pos": tf.io.FixedLenFeature([2], tf.float32),
                "length": tf.io.FixedLenFeature([], tf.int64),
                "groups": tf.io.VarLenFeature(tf.int64),
                "time": tf.io.FixedLenFeature([], tf.float32),
                "time_behavior": tf.io.FixedLenFeature([], tf.float32),
                "indexInDat": tf.io.VarLenFeature(tf.int64),
            }
            for g in range(self.params.nGroups):
                featDesc[f"group{g}"] = tf.io.VarLenFeature(tf.float32)

        datasets = {}
        for key in keys:
            file_path = f"{base_path}_{key}.tfrec"
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue

            print(f"Loading {key} dataset from {file_path}...")

            raw_dataset = tf.data.TFRecordDataset(file_path)

            def _parse_function(example_proto):
                return tf.io.parse_single_example(example_proto, featDesc)

            dataset = raw_dataset.map(
                _parse_function, num_parallel_calls=tf.data.AUTOTUNE
            )

            def map_parse_serialized_sequence(*vals):
                return nnUtils.parse_serialized_sequence(self.params, *vals)

            # Re-apply the parsing logic to get the correct shapes (e.g., reshaping groups)
            dataset = dataset.map(
                map_parse_serialized_sequence,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            datasets[key] = dataset

        return datasets

    def create_optimized_parse_function(
        self,
        augmentation: bool = False,
        augmentation_config: Optional[NeuralDataAugmentation] = None,
        count_spikes: bool = False,
    ):
        if augmentation and augmentation_config:

            @tf.function
            def optimized_parse_with_augmentation(tensors):
                # 1. Identify Spike Groups
                original_groups = {}
                for g in range(self.params.nGroups):
                    g_key = f"group{g}"
                    if g_key in tensors:
                        original_groups[g_key] = tensors[g_key]

                # 2. Call the Vectorized Augmentation logic
                # This returns a dict where every tensor has a new leading 'augmentation' dimension
                return nnUtils.apply_group_augmentation(
                    tensors,
                    original_groups,
                    self.params,
                    augmentation_config,
                    count_spikes,
                )

            return optimized_parse_with_augmentation
        else:
            # If no augmentation, it's just a pass-through because
            # parse_serialized_sequence already ran.
            raise ValueError(
                "Augmentation must be enabled and config provided to create optimized parse function."
            )

    def _save_datasets_to_tfrec(self, datasets, base_path):
        """
        Save parsed datasets to new TFRecord files.

        Parameters
        ----------
        datasets : dict
            Dictionary of tf.data.Dataset objects to save (e.g., {'train': dataset, 'test': dataset})
        base_path : str
            Base path for saved TFRecord files. Files will be saved as:
            {base_path}_train.tfrec, {base_path}_test.tfrec, etc.
        """

        def serialize_example(example_dict):
            """Serialize a single example to TFRecord format"""
            # Create feature dict for serialization
            feature_dict = {}

            for key, value in example_dict.items():
                if isinstance(value, tf.Tensor):
                    value = value.numpy()

                # Handle different data types
                if isinstance(value, np.ndarray):
                    if value.dtype in [np.float32, np.float64]:
                        feature_dict[key] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=value.flatten())
                        )
                    elif value.dtype in [np.int32, np.int64]:
                        feature_dict[key] = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=value.flatten())
                        )
                    else:
                        # For other types, convert to bytes
                        feature_dict[key] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[value.tobytes()])
                        )
                elif isinstance(value, (int, np.integer)):
                    feature_dict[key] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(value)])
                    )
                elif isinstance(value, (float, np.floating)):
                    feature_dict[key] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(value)])
                    )
                elif isinstance(value, (str, bytes)):
                    if isinstance(value, str):
                        value = value.encode()
                    feature_dict[key] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value])
                    )

            # Create an Example proto
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature_dict)
            )
            return example_proto.SerializeToString()

        # Save each dataset
        for key, dataset in datasets.items():
            output_path = f"{base_path}_{key}.tfrec"
            print(f"Saving {key} dataset to {output_path}...")

            writer = tf.io.TFRecordWriter(os.path.abspath(output_path))

            try:
                for batch in tqdm(dataset, desc=f"Writing {key} to TFRecord"):
                    if isinstance(batch, tuple):
                        inputs, _ = batch  # Usually (inputs, targets)
                    else:
                        inputs = batch

                    serialized = serialize_example(inputs)
                    writer.write(serialized)
            finally:
                writer.close()

            print(f"Successfully saved {key} dataset to {output_path}")

    def _save_datasets_to_parquet(self, datasets, base_path):
        """
        Save parsed datasets to Parquet files using pandas logic.

        Parameters
        ----------
        datasets : dict
            Dictionary of tf.data.Dataset objects to save (e.g., {'train': dataset, 'test': dataset})
        base_path : str
            Base path for saved Parquet files. Files will be saved as:
            {base_path}_{key}.parquet
        """

        for key, dataset in datasets.items():
            output_path = f"{base_path}_{key}.parquet"
            print(f"Saving {key} dataset to {output_path}...")

            df = self.convert_tfrec_to_pandas(
                dataset, desc=f"Converting {key} to Pandas"
            )

            if df.shape[0] > 0:
                # Some columns might still be lists, which Parquet handles fine as nesting or objects.
                df.to_parquet(output_path)
                print(f"Successfully saved {key} dataset to {output_path}")
            else:
                print(f"No data to save for {key}")

    def convert_tfrec_to_pandas(
        self, dataset, flatten=True, desc="Converting to Pandas"
    ):
        all_data = []
        for example in tqdm(dataset, desc=desc):
            if isinstance(example, tuple):
                inputs, _ = example
            else:
                inputs = example

            row_data = {}
            for k, v in inputs.items():
                val = v.numpy()
                # Parquet (via Arrow) doesn't like multidimensional arrays in object columns.
                # We flatten arrays with ndim > 1 to ensure compatibility.
                if val.ndim > 1 and flatten:
                    # Store as a flattened 1D array inside the cell
                    row_data[k] = [val.reshape(-1)]
                else:
                    # For 1D or scalars, wrap in list for 1-row DataFrame construction
                    row_data[k] = [val] if val.ndim == 1 else val

            all_data.append(pd.DataFrame(row_data))
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def decode_predictions(
        self,
        preds,
        y_true=None,
        fit_temperature=False,
        T_scaling=None,
        l_function=None,
        **kwargs,
    ):
        """
        Consolidated decoding and post-processing of model predictions.

        Args:
            preds (dict): Dictionary of predictions from model (indexed by target_structure keys).
            y_true (np.ndarray, optional): Concatenated ground truth (matches behavioral data).
            fit_temperature (bool): Whether to fit temperature scaling.
            T_scaling (float, optional): Temperature scaling factor.
            l_function (callable, optional): Linearization function.
            **kwargs: Additional parameters.

        Returns:
            dict: Decoded predictions and metadata.
        """
        results = {}
        use_heatmap = getattr(self.params, "GaussianHeatmap", False)

        # 1. Handle main position head if it's a heatmap
        if use_heatmap and "pos_2d" in preds:
            output_logits = preds["pos_2d"]
            # Ensure 3D (Batch, H, W)
            if len(output_logits.shape) == 2:
                H, W = self.params.GaussianGridSize
                output_logits = tf.reshape(output_logits, [-1, H, W])

            # Calibration if requested
            if fit_temperature and y_true is not None:
                # Get index slice for pos_2d (usually 0:2)
                start, end = self.target_structure["pos_2d"]["slice"]
                y_pos_2d = y_true[:, start:end]
                # Heatmap targets from Mixin
                val_targets = self.gaussian_heatmap_targets_tf(y_pos_2d)
                # Calibrate via Layer
                T_cal = self.GaussianHeatmap.fit_temperature(
                    output_logits, val_targets, iters=400
                )
                return T_cal

            if T_scaling is not None:
                output_logits = output_logits / T_scaling

            # Decode via Mixin
            xy, maxp, Hn, var_total = self.decode_and_uncertainty_tf(output_logits)

            results.update(
                {
                    "pos_2d": xy.numpy(),
                    "logits_hw": output_logits.numpy()
                    if hasattr(output_logits, "numpy")
                    else output_logits,
                    "var_total": var_total.numpy()
                    if hasattr(var_total, "numpy")
                    else var_total,
                    "Hn": Hn.numpy() if hasattr(Hn, "numpy") else Hn,
                    "maxp": maxp.numpy() if hasattr(maxp, "numpy") else maxp,
                    "T_scaling": T_scaling,
                }
            )

        # 2. Reconstruct featurePred by looping through target_structure
        # This ensures the output matrix matches expectations of legacy code
        reconstructed_parts = []
        for name, spec in self.target_structure.items():
            if name == "pos_2d" and use_heatmap:
                reconstructed_parts.append(results["pos_2d"])
            elif name in preds:
                pred_val = (
                    preds[name].numpy()
                    if hasattr(preds[name], "numpy")
                    else preds[name]
                )
                reconstructed_parts.append(pred_val)

        if reconstructed_parts:
            # Concatenate all parts (2d pos + HD + etc)
            results["featurePred"] = np.concatenate(reconstructed_parts, axis=-1)

        # 3. Handle latent explicitly (not in reconstructed list because it's auxiliary)
        if "latent" in preds:
            results["latent"] = (
                preds["latent"].numpy()
                if hasattr(preds["latent"], "numpy")
                else preds["latent"]
            )

        # Classification handling
        if results.get("featurePred") is not None:
            target_str = str(self.target).lower()
            if "classification" in target_str or "int" in target_str:
                results["featurePred"] = np.round(results["featurePred"]).astype(int)

        # Linear projections / ID score
        if l_function and results.get("featurePred") is not None:
            projPredPos, linearPred = l_function(results["featurePred"][:, :2])
            results["projPred"] = projPredPos
            results["linearPred"] = linearPred
            if y_true is not None:
                projTruePos, linearTrue = l_function(y_true[:, :2])
                results["projTruePos"] = projTruePos
                results["linearTrue"] = linearTrue

        return results

    def test(self, behaviorData, **kwargs):
        """
        Test the model on a given behaviorData.

        Args
        ----------
        behaviorData : dict
            Dictionary containing the behavioral data, including 'Times', 'Speed', and 'Positions'.
        l_function : callable, optional
            Function to apply to the predicted and true positions, by default None.
        windowSizeMS : int, optional
            Size of the window in milliseconds, by default 36.
        useSpeedFilter : bool, optional
            Whether to use the speed filter, by default False.
        useTrain : bool, optional
            Whether to use the training epochs, by default False.
        useTest : bool, optional
            Whether to use the testing epochs, by default True.
        onTheFlyCorrection : bool, optional
            Whether to apply on-the-fly correction to the positions, by default False.
        isPredLoss : bool, optional
            Whether to use the prediction loss model, by default False.
        speedValue : float, optional
            Custom speed value to filter the data, by default None.
        phase : str, optional
            Phase of the experiment (e.g., 'train', 'test'), by default None.
        template : str, optional
            Template for the data, by default None.

        """

        # Unpack kwargs
        l_function = kwargs.get("l_function", [])
        windowSizeMS = kwargs.pop("windowSizeMS", 36)
        useSpeedFilter = kwargs.get("useSpeedFilter", False)
        useTrain = kwargs.get("useTrain", False)
        useTest = kwargs.get("useTest", True)
        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)
        isPredLoss = kwargs.get("isPredLoss", False)
        speedValue = kwargs.get("speedValue", None)
        phase = kwargs.get("phase", None)
        template = kwargs.get("template", None)
        fit_temperature = kwargs.get("fit_temperature", False)
        T_scaling = kwargs.get("T_scaling", None)
        strideFactor = kwargs.get("strideFactor", 1)
        extract_spikes_counts = kwargs.get("extract_spikes_counts", False)

        # TODO: change speed filter with custom speed
        # Create the folder
        os.makedirs(os.path.join(self.folderResult, str(windowSizeMS)), exist_ok=True)
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowSizeMS), "savedModels", "predLoss"
                ),
            )
        else:
            try:
                self.model.load_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "full_cp.weights.h5",
                    ),
                    skip_mismatch=True,
                )
            except FileNotFoundError:
                print("loading from savedModels failed, trying full checkpoint ")
                try:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels, str(windowSizeMS), "full" + "/cp.ckpt"
                        ),
                    )
                except (FileNotFoundError, ValueError):
                    print("loading from full checkpoint failed, trying weights.h5")
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "full",
                            "cp.weights.h5",
                        ),
                    )

        # Manage the behavior
        if speedValue is None:
            speedMask = behaviorData["Times"]["speedFilter"]
        else:
            speed = behaviorData["Speed"]
            speedMask = speedValue > speed
        if speedMask.shape[0] != behaviorData["Times"]["speedFilter"].shape[0]:
            warnings.warn("The speed mask must be the same length as the speed filter")
        # Manage epoch mask
        epochMask = get_epochs_mask(
            behaviorData=behaviorData, useTrain=useTrain, useTest=useTest
        )
        if useSpeedFilter:
            totMask = speedMask * epochMask
        else:
            totMask = epochMask

        if speedMask.shape[0] != totMask.shape[0]:
            warnings.warn(
                f"""The speed mask must be the same length as the speed filter?
                Trying to fix it with a new speed filter
                for sessions {phase} Relaunch the test function after.
                """
            )
            from importData import rawdata_parser

            rawdata_parser.speed_filter(
                self.projectPath.folder, phase=phase, template=template, overWrite=True
            )
            raise ValueError(
                """The speed mask must be the same length as the speed filter.
                """
            )

        # Load the and imfer dataset
        ## Get datasets
        if strideFactor > 1:
            filename = (
                f"dataset_stride{str(windowSizeMS)}_factor{str(strideFactor)}.tfrec"
            )
        else:
            filename = f"dataset_stride{str(windowSizeMS)}.tfrec"

        datasets, _ = self._dataset_loading_pipeline(
            filename,
            windowSizeMS,
            behaviorData,
            totMask,
            inference_mode=True,
            onTheFlyCorrection=onTheFlyCorrection,
            shuffle=False,
            speedMask=speedMask,
            **kwargs,
        )
        dataset = datasets["test"]

        save_parsed_tfrec = kwargs.get("save_parsed_tfrec", None)
        if save_parsed_tfrec is not None:
            assert not useSpeedFilter, (
                "Cannot use speed filter when saving parsed TFRecord"
            )
            # save final speedMask
            pos_index = np.arange(len(behaviorData["Positions"]))
            # final speedMask is an 2D array with shape (N,2) where N is the number of timepoints
            final_speedMask = np.zeros((len(pos_index), 2), dtype=np.float32)
            final_speedMask[:, 0] = pos_index
            final_speedMask[:, 1] = speedMask
            np.save(f"{save_parsed_tfrec}_speedMask_{phase}.npy", final_speedMask)

            return

        # 1. Run model prediction
        print(f"Inferring values for {phase} dataset...")
        # dataset yields (inputs, targets)
        preds_dict = self.model.predict(dataset, verbose=1)
        # Model returns a dictionary of outputs {"heatmap": ..., "others": ..., "latent": ...}

        full_pos_loss = (
            None  # Position loss is not explicitly returned by the model anymore
        )

        # 2. Extract metadata in a single pass
        print("Extracting metadata...")
        list_pos = []
        list_times = []
        list_times_behavior = []
        list_pos_index = []
        list_groups = []
        list_speed_filter = []
        list_index_in_dat = []

        # Spike Counts (Dynamic dict to handle variable groups)
        dict_spike_counts = {
            f"group{g}_spikes_count": [] for g in range(self.params.nGroups)
        }

        for inputs, targets in tqdm(dataset, desc="Gathering metadata"):
            # Reconstruct full Y ground truth from individual target heads
            max_idx = 0
            for spec in self.target_structure.values():
                max_idx = max(max_idx, spec["slice"][1])

            batch_size = next(iter(targets.values())).shape[0]
            batch_y_true = np.zeros((batch_size, max_idx), dtype=np.float32)

            for name, spec in self.target_structure.items():
                if name in targets:
                    start, end = spec["slice"]
                    batch_y_true[:, start:end] = targets[name].numpy()

            list_pos.append(batch_y_true)
            list_times.append(inputs["time"].numpy())
            list_times_behavior.append(inputs["time_behavior"].numpy())
            list_pos_index.append(inputs["pos_index"].numpy())
            list_index_in_dat.append(inputs["indexInDat"].numpy())
            list_groups.append(inputs["groups"].numpy())

            # Optional keys (use .get or check)
            if "speedFilter" in inputs:
                list_speed_filter.append(inputs["speedFilter"].numpy())

            if extract_spikes_counts:
                for g in range(self.params.nGroups):
                    key = f"group{g}_spikes_count"
                    if key in inputs:
                        dict_spike_counts[key].append(inputs[key].numpy())

        # 3. Concatenate all batches into single arrays
        print("Concatenating results...")
        # full_pred_features and full_pos_loss are already arrays/None from predict

        full_feature_true = np.concatenate(list_pos, axis=0)
        full_times = np.concatenate(list_times, axis=0).flatten()
        full_times_behavior = np.concatenate(list_times_behavior, axis=0).flatten()
        full_pos_index = np.concatenate(list_pos_index, axis=0).flatten()

        # Handle Speed Mask
        # If speedFilter was in dataset, use it. Otherwise compute via lookup
        if len(list_speed_filter) > 0:
            windowmaskSpeed = np.concatenate(list_speed_filter, axis=0).flatten()
        else:
            # Fallback to your original lookup method
            print("Looking up speed mask from original array...")
            windowmaskSpeed = speedMask[full_pos_index]

        # -------------------------------------------------------------------------
        # 3. CONSOLIDATED POST-PROCESSING
        # -------------------------------------------------------------------------
        decoded_results = self.decode_predictions(
            preds=preds_dict,
            y_true=full_feature_true,
            fit_temperature=fit_temperature,
            T_scaling=T_scaling,
            l_function=l_function,
        )

        # If we were just fitting temperature, return the scaling factor
        if fit_temperature:
            return decoded_results

        # Ensure Feature True shape is correct
        featureTrue = np.reshape(
            full_feature_true, [decoded_results["featurePred"].shape[0], -1]
        )

        # -------------------------------------------------------------------------
        # PACKAGING OUTPUTS
        # -------------------------------------------------------------------------

        testOutput = {
            "featurePred": decoded_results["featurePred"],
            "featureTrue": featureTrue,
            "times": full_times,
            "times_behavior": full_times_behavior,
            "posLoss": full_pos_loss,
            "posIndex": full_pos_index,
            "speedMask": windowmaskSpeed,
            "latent": decoded_results["latent"],
        }

        # Merge other metrics from decoding
        for k in [
            "projPred",
            "projTruePos",
            "linearPred",
            "linearTrue",
            "logits_hw",
            "var_total",
            "Hn",
            "maxp",
            "T_scaling",
        ]:
            if k in decoded_results:
                testOutput[k] = decoded_results[k]

        # -------------------------------------------------------------------------
        # CSV GENERATION (Spike Counts)
        # -------------------------------------------------------------------------

        if extract_spikes_counts:
            csv_path = os.path.join(
                self.folderResult, str(windowSizeMS), f"spikes_count_{phase}.csv"
            )

            if not os.path.exists(csv_path) and not useSpeedFilter:
                print("Processing spike counts for CSV...")

                # Concatenate the raw indices
                full_index_raw = [
                    row.tolist() for batch in list_index_in_dat for row in batch
                ]

                # Construct DataFrame directly from the arrays (Much faster than row-loop)
                data_dict = {
                    "posIndex": full_pos_index,
                    # Convert list of arrays/lists to string or keep as object for indexInDat
                    "indexInDat": full_index_raw,
                }

                # Add group counts
                for g in range(self.params.nGroups):
                    key = f"group{g}_spikes_count"
                    if len(dict_spike_counts[key]) > 0:
                        data_dict[key] = np.concatenate(
                            dict_spike_counts[key], axis=0
                        ).astype(int)

                df = pd.DataFrame(data_dict)

                print(f"Saving CSV to {csv_path}")
                df.to_csv(csv_path, index=False)

        # -------------------------------------------------------------------------
        # LINEAR FUNCTION METRICS
        # -------------------------------------------------------------------------
        if l_function:
            projPredPos, linearPred = l_function(decoded_results["featurePred"][:, :2])
            projTruePos, linearTrue = l_function(featureTrue[:, :2])
            testOutput["projPred"] = projPredPos
            testOutput["projTruePos"] = projTruePos
            testOutput["linearPred"] = linearPred
            testOutput["linearTrue"] = linearTrue

        # -------------------------------------------------------------------------
        # ADDITIONAL METRICS & VISUALIZATIONS
        # -------------------------------------------------------------------------
        self._compute_metrics_and_plots(
            featurePred=decoded_results["featurePred"],
            featureTrue=featureTrue,
            phase=phase,
            windowSizeMS=windowSizeMS,
            testOutput=testOutput,
            sleep=False,
        )

        # -------------------------------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------------------------------
        self.saveResults(testOutput, folderName=windowSizeMS, phase=phase)

        return testOutput

    def _compute_metrics_and_plots(
        self, featurePred, featureTrue, phase, windowSizeMS, testOutput, sleep=False
    ):
        """Helper to compute additional metrics and generate visualizations"""
        print(f"Calculating additional metrics for {phase}...")
        target = str(self.params.target).lower()

        metrics = {}
        # 2D Position Metrics
        if "pos" in target or "lin" in target:
            # MSE on 2D positions
            dist_sq = np.sum((featurePred[:, :2] - featureTrue[:, :2]) ** 2, axis=1)
            metrics["mse_2d"] = np.mean(dist_sq)
            metrics["rmse_2d"] = np.sqrt(metrics["mse_2d"])
            metrics["max_error"] = np.max(np.sqrt(dist_sq))
            testOutput["residuals"] = featurePred - featureTrue

        # Classification Metrics
        if "classification" in target or "int" in target:
            metrics["accuracy"] = np.mean(featurePred == featureTrue)

        # Polar/Direction Metrics
        if "direction" in target or "head" in target:
            # Assume angles are in radians if single column, or unit vectors if 2 columns
            if featurePred.shape[1] == 1:
                # Circular MAE
                diff = (featurePred - featureTrue + np.pi) % (2 * np.pi) - np.pi
                metrics["circular_mae"] = np.mean(np.abs(diff))
                metrics["bias"] = np.mean(diff)
            elif featurePred.shape[1] == 2:
                # Dot product for unit vectors
                cos_sim = np.sum(featurePred * featureTrue, axis=1) / (
                    np.linalg.norm(featurePred, axis=1)
                    * np.linalg.norm(featureTrue, axis=1)
                    + 1e-8
                )
                metrics["angular_error"] = np.mean(np.arccos(np.clip(cos_sim, -1, 1)))

        testOutput["metrics"] = metrics
        print(f"Metrics: {metrics}")

        # VISUALIZATIONS
        if ("pos" in target or "lin" in target) and featurePred.shape[1] >= 2:
            print("Generating quiver plot...")
            plt.figure(figsize=(10, 10))
            # Sample for clarity if too many points
            n = len(featurePred)
            step = max(1, n // 500)
            p_true = featureTrue[::step, :2]
            res = (featurePred - featureTrue)[::step, :2]
            plt.quiver(
                p_true[:, 0],
                p_true[:, 1],
                res[:, 0],
                res[:, 1],
                scale_units="xy",
                angles="xy",
                scale=1,
                alpha=0.6,
                color="red",
            )
            plt.title(f"Position Residuals - {phase} ({windowSizeMS}ms)")
            plt.xlabel("X")
            plt.ylabel("Y")

            # Save plot path
            if sleep:
                plot_folder = os.path.join(
                    self.folderResultSleep, str(windowSizeMS), phase
                )
            else:
                plot_folder = os.path.join(self.folderResult, str(windowSizeMS))

            os.makedirs(plot_folder, exist_ok=True)
            plot_path = os.path.join(plot_folder, f"quiver_residuals_{phase}.png")
            plt.savefig(plot_path)
            plt.close()
            testOutput["quiver_plot_path"] = plot_path

    def testSleep(self, behaviorData, **kwargs):
        """
        Test the network on sleep epochs.

        parameters:
        ______________________________________________________
        behaviorData : dict
            dictionary containing the behavioral data. In particular, it needs to contain the following keys:
            - Times : dict with sleepNames and sleepEpochs keys
        l_function : list
        windowSizeDecoder : int
        windowSizeMS : int
        isPredLoss : bool
        """
        # TODO: add option for windowSizeMS vs windowSizeDecoder consistency check, with striding as well
        # Unpack kwargs
        l_function = kwargs.get("l_function", [])
        windowSizeDecoder = kwargs.get("windowSizeDecoder", None)
        windowSizeMS = kwargs.get("windowSizeMS", 36)
        isPredLoss = kwargs.get("isPredLoss", False)
        strideFactor = kwargs.get("strideFactor", 1)
        T_scaling = kwargs.get("T_scaling", None)

        # Create the folder
        if windowSizeDecoder is None:
            folderName = str(windowSizeMS)
            if not os.path.isdir(os.path.join(self.folderResultSleep, folderName)):
                os.makedirs(os.path.join(self.folderResultSleep, folderName))
        else:
            folderName = f"{str(windowSizeMS)}_by_{str(windowSizeDecoder)}"
            if not os.path.isdir(os.path.join(self.folderResultSleep, folderName)):
                os.makedirs(os.path.join(self.folderResultSleep, folderName))

        if windowSizeDecoder is None:
            windowSizeDecoder = windowSizeMS

        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowSizeMS), "savedModels", "predLoss"
                ),
            )
        else:
            try:
                self.model.load_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "full_cp.weights.h5",
                    ),
                    skip_mismatch=True,
                )
            except FileNotFoundError:
                print("loading from savedModels failed, trying full checkpoint ")
                try:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels, str(windowSizeMS), "full" + "/cp.ckpt"
                        ),
                    )
                except (FileNotFoundError, ValueError):
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "full",
                            "cp.weights.h5",
                        ),
                    )

        print("decoding sleep epochs")
        predictions = {}
        for idsleep, sleepName in enumerate(behaviorData["Times"]["sleepNames"]):
            timeSleepStart = behaviorData["Times"]["sleepEpochs"][2 * idsleep][0]
            timeSleepStop = behaviorData["Times"]["sleepEpochs"][2 * idsleep + 1][0]

            if strideFactor > 1:
                sleepFilename = f"datasetSleep_stride{str(windowSizeMS)}_factor{str(strideFactor)}.tfrec"
            else:
                sleepFilename = f"datasetSleep_stride{str(windowSizeMS)}.tfrec"
            # Get the dataset
            dataset = tf.data.TFRecordDataset(
                os.path.join(self.projectPath.dataPath, sleepFilename)
            )

            def _parse_function(*vals):
                return nnUtils.parse_serialized_spike(self.featDesc, *vals)

            dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

            def filter_by_time(x):
                return tf.math.logical_and(
                    tf.squeeze(tf.math.less_equal(x["time"], timeSleepStop)),
                    tf.squeeze(tf.math.greater_equal(x["time"], timeSleepStart)),
                )

            def map_parse_serialized_sequence(*vals):
                return nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                )

            @tf.function
            def map_outputs(vals):
                # Move 'pos' to targets, rest stay in inputs
                inputs_dict = {k: v for k, v in vals.items() if k != "pos"}
                # Structured targets matching model outputs
                targets_dict = {}
                for name, spec in self.target_structure.items():
                    start_idx, end_idx = spec["slice"]
                    targets_dict[name] = vals["pos"][:, start_idx:end_idx]

                # latent targets are the 2D position for contrastive regression
                if "latent" in self.outNames:
                    targets_dict["latent"] = vals["pos"][:, :2]

                return (inputs_dict, targets_dict)

            dataset = dataset.filter(filter_by_time)
            dataset = dataset.batch(self.params.batch_size, drop_remainder=True)

            dataset = dataset.map(
                map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)
            dataset.cache()
            dataset.prefetch(tf.data.AUTOTUNE)
            # Infer
            print(f"Inferring {sleepName} values")
            preds_dict = self.model.predict(dataset, verbose=1)

            # -------------------------------------------------------------------------
            # CONSOLIDATED POST-PROCESSING
            # -------------------------------------------------------------------------
            decoded_results = self.decode_predictions(
                preds=preds_dict,
                T_scaling=T_scaling,
                l_function=l_function,
            )

            output_preds = decoded_results["featurePred"]

            # output is used for predictions[sleepName] packaging
            output = (output_preds, None)

            # Post-infer management: Gather metadata efficiently in a single pass
            print(f"gathering metadata for {sleepName}")
            list_times = []
            list_posIndex = []
            list_IDdat = []
            list_pos = []

            for inputs, targets in tqdm(
                dataset, desc=f"Gathering metadata {sleepName}"
            ):
                list_times.append(inputs["time"].numpy())
                list_posIndex.append(inputs["pos_index"].numpy())
                list_IDdat.append(inputs["indexInDat"].numpy())

                # Reconstruct full Y ground truth from individual target heads
                max_idx = 0
                for spec in self.target_structure.values():
                    max_idx = max(max_idx, spec["slice"][1])

                batch_size = next(iter(targets.values())).shape[0]
                batch_y_true = np.zeros((batch_size, max_idx), dtype=np.float32)

                for name, spec in self.target_structure.items():
                    if name in targets:
                        start, end = spec["slice"]
                        batch_y_true[:, start:end] = targets[name].numpy()

                list_pos.append(batch_y_true)

            times = np.concatenate(list_times, axis=0).flatten()
            posIndex = np.concatenate(list_posIndex, axis=0).flatten()
            IDdat = [batch for batch in list_IDdat]

            # featureTrue if targets were present (some sleep recordings might have it)
            featureTrue = None
            if list_pos:
                featureTrue = np.concatenate(list_pos, axis=0)
                featureTrue = np.reshape(featureTrue, [output[0].shape[0], -1])

            predictions[sleepName] = {
                "featurePred": output[0],
                "featureTrue": featureTrue,  # Still add even if None for consistency
                "times": times,
                "posIndex": posIndex,
                "indexInDat": IDdat,
            }

            # If we have targets, compute metrics for this sleep epoch
            if featureTrue is not None:
                self._compute_metrics_and_plots(
                    featurePred=output[0],
                    featureTrue=featureTrue,
                    phase=sleepName,
                    windowSizeMS=windowSizeMS,
                    testOutput=predictions[sleepName],
                    sleep=True,
                )
            if l_function:
                projPredPos, linearPred = l_function(output[0][:, :2])
                predictions[sleepName]["projPred"] = projPredPos
                predictions[sleepName]["linearPred"] = linearPred

            if getattr(self.params, "GaussianHeatmap", False):
                # add uncertainty and confidence metrics to output dict
                print("Not implemented yet")

        # Save the results
        for key in predictions.keys():
            self.saveResults(
                predictions[key], folderName=folderName, sleep=True, sleepName=key
            )

    def convert_checkpoint_to_keras3(
        self, model, old_checkpoint_path, new_checkpoint_path
    ):
        """Convert old .ckpt format to new .weights.h5 format"""
        try:
            # Try to load old weights
            model.load_weights(old_checkpoint_path)

            # Save in new format
            model.save_weights(new_checkpoint_path)
            print(
                f"Successfully converted {old_checkpoint_path} to {new_checkpoint_path}"
            )

        except Exception as e:
            print(f"Failed to convert checkpoint: {e}")
            return False
        return True

    def get_theweights(self, behaviorData, windowSizeMS, isPredLoss=0):
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowSizeMS), "predLoss" + "/cp.ckpt"
                )
            )
        else:
            self.model.load_weights(
                os.path.join(self.folderModels, str(windowSizeMS), "full" + "/cp.ckpt")
            )
        wdata = []
        for layer in self.model.layers:
            if hasattr(layer, "get_weights"):
                wdata.extend(layer.get_weights())
        # reshaped_w = [tf.reshape(w,(2,3,1,8)) if w.shape == (2,3,8,16) else w for w in wdata]
        # return reshaped_w
        return wdata

    def _apply_oversampling_resampling(self, dataset, windowSizeMS, shuffle=True):
        """
        Apply oversampling resampling to the training dataset to balance the samples.

        Args:
            dataset : tf.data.Dataset
                The training dataset to be resampled.
            windowSizeMS : int
                The window size in milliseconds.
            shuffle : bool, optional
                Whether to shuffle the dataset after resampling, by default True.

        Returns:
            tf.data.Dataset
                The resampled training dataset.
            counts
                np.ndarray
                The counts of samples in each bin before resampling.
        """
        print("Using oversampling resampling on the training set")

        # FIX: make sure GaussianHeatmap is initialized - force it ?

        GRID_H, GRID_W = (
            self.GaussianHeatmap.GRID_H,
            self.GaussianHeatmap.GRID_W,
        )
        # instead of oversampling on such tiny grid, we take a coarser grid mesh
        stride = 5
        self.coarse_H, self.coarse_W = GRID_H // stride, GRID_W // stride

        def map_bin_class(ex):
            return nnUtils.bin_class(
                ex,
                GRID_H,
                GRID_W,
                stride,
                self.GaussianHeatmap.forbid_mask_tf,
            )

        def filter_for_oversampling(ex):
            return tf.greater_equal(map_bin_class(ex), 0)

        # should not be useful as true positions are already allowed!
        dataset = dataset.filter(filter_for_oversampling)

        positions = self.GaussianHeatmap.training_positions
        # filter position by map_bin_class
        bins = self.GaussianHeatmap.positions_to_bins(positions)

        # Convert fine bins → coarse bins
        # Map to coarse bins
        # Step 1: compute fine x,y indices
        x_fine = bins % GRID_W
        y_fine = bins // GRID_W

        # Step 2: downscale to coarse grid
        x_coarse = x_fine // stride
        y_coarse = y_fine // stride

        # Step 3: coarse bin index
        coarse_bins = y_coarse * self.coarse_W + x_coarse  # shape same as positions
        counts = np.bincount(
            coarse_bins, minlength=self.coarse_H * self.coarse_W
        ).astype(np.float32)

        # Flatten FORBID for easy masking
        # Forbidden bins in coarse space (if you want to respect FORBID also at coarse level)
        FORBID_coarse = np.zeros((self.coarse_H, self.coarse_W), dtype=bool)
        for y in range(self.coarse_H):
            for x in range(self.coarse_W):
                # If any fine bin inside coarse cell is forbidden, mark whole cell forbidden
                if np.any(
                    self.GaussianHeatmap.forbid_mask_tf[
                        y * stride : (y + 1) * stride,
                        x * stride : (x + 1) * stride,
                    ]
                    > 0
                ):
                    FORBID_coarse[y, x] = True
        FORBID_flat = FORBID_coarse.flatten()
        counts[FORBID_flat] = 0  # set forbidden bins to 0 count

        allowed_bins = (counts > 0) & (~FORBID_flat)

        # compute oversampling ratios relative to max count among allowed bins
        max_count = counts.max()
        rep_factors = max_count / np.maximum(counts[allowed_bins], 1e-8)
        rep_factors = np.minimum(rep_factors, 20.0)  # clip to avoid extreme repeats
        rep_factors_tf = tf.constant(rep_factors, tf.float32)

        allowed_idx = np.where(allowed_bins)[0]
        bin_to_allowed_idx = -np.ones_like(allowed_bins, dtype=int)
        bin_to_allowed_idx[allowed_idx] = np.arange(len(allowed_idx))
        # convert to tensor
        allowed_bins = tf.constant(allowed_bins)
        bin_to_allowed_idx = tf.constant(bin_to_allowed_idx)

        # Map each example to repeated dataset
        def map_repeat(ex):
            mapped_cls = map_bin_class(ex)
            allowed_idx_val = tf.gather(bin_to_allowed_idx, mapped_cls)

            safe_idx = tf.maximum(allowed_idx_val, 0)  # -1 becomes 0
            # find idx in rep_factors (only allowed bins)
            repeats = tf.cast(
                tf.math.ceil(tf.gather(rep_factors_tf, safe_idx)), tf.int64
            )
            repeats = tf.where(
                allowed_idx_val >= 0, repeats, tf.constant(0, dtype=tf.int64)
            )

            def repeat_fn():
                return tf.data.Dataset.from_tensors(ex).repeat(repeats)

            def empty_fn():
                return tf.data.Dataset.from_tensors(ex).take(0)

            return tf.cond(repeats > 0, repeat_fn, empty_fn)

        dataset_before_oversampling = dataset
        # Save this before the oversampling block
        dataset = dataset.flat_map(map_repeat)
        # shuffle after repeating to mix repeated samples
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
        dataset_after_oversampling = dataset  # Save this before the oversampling block
        from neuroencoders.importData.gui_elements import OversamplingVisualizer

        if not os.path.exists(
            os.path.join(
                self.folderResult, str(windowSizeMS), "oversampling_effect.png"
            )
        ):
            visualizer = OversamplingVisualizer(self.GaussianHeatmap)
            visualizer.visualize_oversampling_effect(
                dataset_before_oversampling,
                dataset_after_oversampling,
                stride=stride,  # Match your stride
                max_samples=30000,
                path=os.path.join(
                    self.folderResult,
                    str(windowSizeMS),
                    "oversampling_effect.png",
                ),
            )
        return dataset, counts

    def get_artificial_spikes(
        self,
        behaviorData: dict,
        windowSizeMS: int = 36,
        useSpeedFilter: bool = False,
        useTrain: bool = False,
        useTest: bool = True,
        isPredLoss: bool = False,
        strideFactor: int = 1,
        phase: str = "test",
        extract_waveforms: bool = False,
        layer_name="outputCNN",
        save: bool = True,
        pad_shanks: bool = False,
        groups_list=None,
        file_path=None,
    ):
        """
        Extract CNN-level embeddings for every spike from the inference dataset,
        using the SAME preprocessing pipeline as `test()`.

        Returns:
            dict with:
                cnn_features  : (N, feature_dim)
                group_ids     : (N,)
                posIndex      : (N,)
                indexInDat    : (N,)
        """
        if groups_list is None:
            groups_list = [g for g in range(self.params.nGroups)]
        if not isinstance(groups_list, list):
            groups_list = [groups_list]

        nGroups = len(groups_list)

        if isinstance(layer_name, str):
            layer_name = [f"{layer_name}{g}" for g in groups_list]

        print("Loading trained weights...")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowSizeMS), "savedModels", "predLoss"
                ),
            )
        else:
            try:
                self.model.load_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "full_cp.weights.h5",
                    ),
                    skip_mismatch=True,
                )
            except FileNotFoundError:
                print("fallback loading full/cp.ckpt")
                try:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels, str(windowSizeMS), "full", "cp.ckpt"
                        ),
                        skip_mismatch=True,
                    )
                except (FileNotFoundError, ValueError):
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "full",
                            "cp.weights.h5",
                        ),
                        skip_mismatch=True,
                    )

        # --- Build the same total mask used in test() ---
        epochMask = get_epochs_mask(
            behaviorData=behaviorData, useTrain=useTrain, useTest=useTest
        )

        if useSpeedFilter:
            speedMask = behaviorData["Times"]["speedFilter"]
        else:
            speedMask = np.ones_like(epochMask, dtype=bool)

        totMask = speedMask * epochMask

        # --- Load dataset using SAME pipeline as test() ---
        filename = (
            f"dataset_stride{windowSizeMS}_factor{strideFactor}.tfrec"
            if strideFactor > 1
            else f"dataset_stride{windowSizeMS}.tfrec"
        )

        datasets, _ = self._dataset_loading_pipeline(
            filename,
            windowSizeMS,
            behaviorData,
            totMask,
            inference_mode=True,
            extract_spikes_counts=False,
            shuffle=False,
            phase=phase,
        )
        dataset = datasets["test"]

        # Build multi-output model once
        layers = [self.model.get_layer(name=l_name).output for l_name in layer_name]

        multi_model = tf.keras.Model(inputs=self.model.input, outputs=layers)

        @tf.function
        def forward(batch, model):
            return model(batch, training=False)

        print(
            f"Extracting CNN embeddings for spikes (output of layers {layer_name[0]}) for {nGroups} groups..."
        )

        all_features = []
        all_group_ids = []
        all_posIndex = []
        all_indexInDat = []
        all_inds = []
        max_nChan = max(self.params.nChannelsPerGroup[g] for g in groups_list)

        if extract_waveforms:
            all_waveforms = []  # GLOBAL list matching all_features
            zero_pad_waveform = {
                g: np.zeros((self.params.nChannelsPerGroup[g], 32)) for g in groups_list
            }
            global_pad = np.zeros((max_nChan, 32))
        else:
            all_waveforms = None

        for batch_inputs, _ in dataset:
            # fetch posIndex and indexInDat for this batch
            batch_posIndex = batch_inputs["pos_index"].numpy()  # size (batch_size,)
            batch_indexInDat = batch_inputs[
                "indexInDat"
            ].numpy()  # size (max_n_spikes,)
            batch_outputs = forward(
                batch_inputs, multi_model
            )  # shape (nGroups, n_spikes_g, dim)

            for g in groups_list:
                # Get raw CNN output for this group
                raw_features = (
                    batch_outputs[g] if nGroups > 1 else batch_outputs
                )  # shape (n_spikes_g, dim)
                zero_pad = np.zeros_like(raw_features[0:1, :])  # shape (1, dim)
                raw_features = np.concatenate(
                    [zero_pad, raw_features], 0
                )  # add 0-vector

                # Indices mapping to original spike order
                inds = batch_inputs[f"indices{g}"].numpy()  # shape (N,)

                # reorder to match spike stream
                ordered_feats = np.take(raw_features, inds, axis=0)

                if pad_shanks:
                    n_chan_g = ordered_feats.shape[1]
                    if n_chan_g < max_nChan:
                        # pad to max channels with zeros
                        # the spike dimension (0) is left untouched, as well as all the next dims (2+)
                        # we need to compute dynamically as ordered_feats.shape is layer dependant
                        pad_shape = (
                            ordered_feats.shape[0],
                            max_nChan - n_chan_g,
                        ) + ordered_feats.shape[2:]
                        pad = np.zeros(pad_shape)
                        ordered_feats = np.concatenate([ordered_feats, pad], axis=1)

                if extract_waveforms:
                    wf_batch = batch_inputs[
                        f"group{g}"
                    ].numpy()  # (n_spikes_g, nChan_g, 32)
                    n_chan_g = wf_batch.shape[1]

                    # pad index 0 (like feature zero_pad)
                    wf_padded = np.concatenate(
                        [zero_pad_waveform[g][None, :, :], wf_batch], axis=0
                    )

                    # reorder exactly like features
                    wf_ordered = np.take(wf_padded, inds, axis=0)

                    # pad to max channels with zeros
                    if n_chan_g < max_nChan:
                        pad = global_pad[
                            n_chan_g:max_nChan, :
                        ]  # (max_nChan - n_chan_g, 32)
                        pad_expanded = np.broadcast_to(
                            pad[None, :, :], (wf_ordered.shape[0], pad.shape[0], 32)
                        )
                        wf_ordered = np.concatenate([wf_ordered, pad_expanded], axis=1)

                    # append to global list (not per-group)
                    all_waveforms.append(wf_ordered)

                all_features.append(ordered_feats)
                all_group_ids.append(np.full(ordered_feats.shape[0], g))
                all_posIndex.append(batch_posIndex)
                all_indexInDat.append(batch_indexInDat)
                all_inds.append(inds)

        # --- Concatenate all groups ---
        cnn_features = np.concatenate(all_features, axis=0)
        group_ids = np.concatenate(all_group_ids, axis=0)
        posIndex = np.concatenate(all_posIndex, axis=0)
        indexInDat = np.concatenate(all_indexInDat, axis=0)
        inds = np.concatenate(all_inds, axis=0)
        if extract_waveforms:
            all_waveforms = np.concatenate(all_waveforms, axis=0)

        result = {
            "cnn_features": cnn_features,
            "group_ids": group_ids,
            "posIndex": posIndex,
            "indexInDat": indexInDat,
            "indices": inds,
        }

        # optional save
        if save:
            out_file = (
                os.path.join(
                    self.folderResult, str(windowSizeMS), "artificial_spikes.pkl"
                )
                if file_path is None
                else file_path
            )
            print(f"Saving artificial spikes to {out_file}...")
            import dill as pickle

            with open(out_file, "wb") as f:
                pickle.dump(result, f)

            if extract_waveforms:
                out_file = os.path.join(
                    self.folderResult,
                    str(windowSizeMS),
                    "artificial_waveforms.pkl",
                )
                print(f"Saving artificial spikes waveforms to {out_file}...")
                with open(
                    out_file,
                    "wb",
                ) as f:
                    pickle.dump(all_waveforms, f)

        return result, all_waveforms

    ########### END OF FULL NETWORK CLASS #####################

    ########### START OF HELPING LSTMandSpikeNetwork FUNCTIONS#####################
    class LRScheduler:
        def __init__(self, lrs, total_epochs=100, warmup_epochs=10, min_lr=1e-6):
            """
            Args:
                lrs: list of learning rates (lrs[0] is used as initial base LR)
                total_epochs: total number of training epochs
                warmup_epochs: number of epochs for linear warmup
                min_lr: minimum learning rate at the end of training
            """
            self.lrs = lrs
            self.initial_lr = lrs[0]
            self.total_epochs = total_epochs
            self.warmup_epochs = warmup_epochs
            self.min_lr = min_lr

        def schedule_fixed(self, epoch, lr):
            if len(self.lrs) == 1:
                return self.lrs[0]
            elif len(self.lrs) == 2:
                return self.lrs[0] if epoch < 10 else self.lrs[1]
            elif len(self.lrs) == 3:
                if epoch < 10:
                    return self.lrs[0]
                elif epoch < 50:
                    return self.lrs[1]
                else:
                    return self.lrs[2]
            else:
                return lr

        def schedule_decay(self, epoch, lr):
            if epoch < 10:
                return lr
            else:
                new_lr = lr * np.exp(-0.01)
                print(f"Epoch {epoch}: learning rate is {new_lr}")
                return float(new_lr)

        def schedule_cosine_warmup(self, epoch, lr):
            """
            Linear warmup followed by Cosine Decay.
            """
            # 1. Linear Warmup
            if epoch < self.warmup_epochs:
                # Linearly increase from approx 0 to initial_lr
                alpha = (epoch + 1) / self.warmup_epochs
                new_lr = self.initial_lr * alpha

            # 2. Cosine Decay
            else:
                # Progress from 0.0 to 1.0 during the decay phase
                decay_steps = self.total_epochs - self.warmup_epochs
                current_step = min(epoch - self.warmup_epochs, decay_steps)

                # Cosine function varies from 1 to -1, mapped to 1 to 0
                cosine_decay = 0.5 * (1 + np.cos(np.pi * current_step / decay_steps))

                # Scale between initial_lr and min_lr
                new_lr = (self.initial_lr - self.min_lr) * cosine_decay + self.min_lr

            print(f"Epoch {epoch}: learning rate is {new_lr:.6f}")
            return float(new_lr)

    def fix_linearizer(self, mazePoints, tsProj):
        ## For the linearization we define two fixed inputs:
        self.maze_points = mazePoints
        self.ts_proj = tsProj
        self.mazePoints_tensor = tf.convert_to_tensor(mazePoints[None, :])
        self.tsProjTensor = tf.convert_to_tensor(tsProj[None, :])

    # used in the data pipepline
    def create_indices(self, vals, shuffle=False):
        """
        Create relative indices for gathering spikes from each group.
        The i-th spike of the group should be positioned at spikePosition[i] in the final tensor.

        Args:
            vals (dict): A dictionary containing the input tensors, including "groups" and "group{n}" for each group.
            addLinearizationTensor (bool): Whether to add linearization tensors to the output.
            shuffle (bool): Whether to shuffle the indices within each group for null hypothesis/control.
        Returns:
            dict: Updated dictionary with indices for each group. The indices are stored under the keys "indices{n}" for each group.
        """
        if shuffle:
            print(
                "Shuffling spike indices within each group for null hypothesis/control."
            )

        groups = vals["groups"]
        for group_id in range(self.params.nGroups):
            # Find positions of spikes belonging to this group
            is_in_group = tf.equal(groups, group_id)

            # 2. Use cumsum to generate sequential IDs (1, 2, 3...) for these spikes
            # This replaces the SparseTensor logic entirely.
            # Example: [0, 1, 0, 1] -> [0, 1, 1, 2]
            relative_indices = tf.cast(
                tf.cumsum(tf.cast(is_in_group, tf.int32)), tf.int32
            )

            # 3. Apply the mask so only spikes in this group have a non-zero index
            # Example: [0, 1, 1, 2] -> [0, 1, 0, 2]
            indices_tensor = tf.where(is_in_group, relative_indices, 0)

            # 4. Handle Shuffling (Null Hypothesis Control)
            if shuffle:
                # We only want to shuffle the non-zero indices
                # Extract values, shuffle them, and put them back
                non_zero_indices = tf.boolean_mask(indices_tensor, is_in_group)
                shuffled_values = tf.random.shuffle(non_zero_indices)

                # Use scatter_nd to put shuffled values back into a zero-filled tensor
                # We need the positions for scattering
                positions = tf.where(is_in_group)
                indices_tensor = tf.scatter_nd(
                    indices=positions,
                    updates=shuffled_values,
                    shape=tf.cast(tf.shape(groups), tf.int64),
                )
            vals[f"indices{group_id}"] = indices_tensor

        return vals

    # used in the data pipepline some day?
    def create_indices_w_temporal_sequence(self, vals, addLinearizationTensor=False):
        """
        Create indices for gathering spikes from each group, respecting actual temporal structure.

        Args:
            vals (dict): A dictionary containing the input tensors, including "groups", "group{n}",
                         "indexInDat" (actual spike times), and "time_behavior".
            addLinearizationTensor (bool): Whether to add linearization tensors to the output.

        Returns:
            dict: Updated dictionary with:
                - groups: reshaped to temporal bins (with -1 for empty bins)
                - indices{n}: positions to gather spikes from each group
                - temporal_mask{n}: mask indicating valid time bins
                - time_bins: actual time bins for the sequence
        """
        # Extract actual timing information
        spike_times = tf.sparse.to_dense(vals["indexInDat"])  # Actual sample indices
        original_groups = tf.sparse.to_dense(
            vals["groups"], default_value=-1
        )  # -1 for padding

        # Define temporal resolution (bin size in samples or time units)
        temporal_bin_size = self.params.get("temporalBinSize", 1.0)

        # Calculate relative times for ALL spikes across all groups
        min_time = tf.reduce_min(spike_times)
        relative_times = spike_times - min_time
        max_time = tf.reduce_max(relative_times)

        # Determine total number of temporal bins needed
        n_temporal_bins = tf.cast(tf.math.ceil(max_time / temporal_bin_size), tf.int32)
        n_temporal_bins = tf.maximum(n_temporal_bins, 1)  # At least one bin

        # Map each spike to its temporal bin
        temporal_bin_indices = tf.cast(relative_times / temporal_bin_size, tf.int32)
        temporal_bin_indices = tf.minimum(
            temporal_bin_indices, n_temporal_bins - 1
        )  # Clamp to valid range

        # Extract batch information (assuming groups tensor is already batched)
        # If groups is 1D: [total_spikes], we need batch size from elsewhere
        # If groups is 2D: [batch, max_spikes], extract batch dimension
        original_shape = tf.shape(original_groups)
        is_batched = len(original_groups.shape) > 1

        if is_batched:
            batch_size = original_shape[0]
            max_spikes_per_batch = original_shape[1]
            # Flatten for processing
            original_groups_flat = tf.reshape(original_groups, [-1])
            temporal_bin_indices_flat = tf.reshape(temporal_bin_indices, [-1])

            # Create batch indices
            batch_indices = tf.repeat(tf.range(batch_size), max_spikes_per_batch)
        else:
            # Assume batch_size is set in params
            batch_size = self.params.batch_size
            total_spikes = original_shape[0]
            max_spikes_per_batch = total_spikes // batch_size

            original_groups_flat = original_groups
            temporal_bin_indices_flat = temporal_bin_indices
            batch_indices = tf.repeat(tf.range(batch_size), max_spikes_per_batch)

        # Create new groups tensor with temporal structure
        # Shape: [batch_size, n_temporal_bins]
        # Initialize with -1 (padding/no spike)
        new_groups = tf.fill([batch_size, n_temporal_bins], -1)

        # For each spike, place its group ID at the corresponding temporal bin
        valid_spike_mask = tf.not_equal(original_groups_flat, -1)
        valid_indices = tf.where(valid_spike_mask)[:, 0]

        valid_batch_indices = tf.gather(batch_indices, valid_indices)
        valid_temporal_bins = tf.gather(temporal_bin_indices_flat, valid_indices)
        valid_groups = tf.gather(original_groups_flat, valid_indices)

        # Create linear indices for scatter: batch * n_temporal_bins + temporal_bin
        linear_indices = valid_batch_indices * n_temporal_bins + valid_temporal_bins

        # Handle collisions: if multiple spikes map to same temporal bin
        # Strategy 1: Keep first occurrence (using sparse tensor)
        # Strategy 2: Keep last occurrence (using scatter_nd with updates overwriting)
        # Strategy 3: Mark as special "multi-spike" bin (value could be max(nGroups))

        # Using sparse tensor to keep first occurrence (consistent with original behavior)
        groups_sparse = tf.sparse.SparseTensor(
            indices=tf.expand_dims(linear_indices, 1),
            values=valid_groups,
            dense_shape=[batch_size * n_temporal_bins],
        )
        new_groups_flat = tf.sparse.to_dense(groups_sparse, default_value=-1)
        new_groups = tf.reshape(new_groups_flat, [batch_size, n_temporal_bins])

        # Update vals with the temporally-structured groups
        vals.update({"groups": new_groups})

        # Now create indices for each group with the temporal structure
        for group in range(self.params.nGroups):
            # Find positions where this group has spikes in the NEW temporal structure
            group_mask = tf.equal(new_groups, group)
            spikePosition = tf.where(
                group_mask
            )  # [num_group_spikes, 2] where 2 = [batch_idx, temporal_bin]

            # Get the original spike indices for this group
            original_group_mask = tf.equal(original_groups_flat, group)
            original_group_indices = tf.where(original_group_mask)[:, 0]

            # Map temporal positions to original spike indices
            # For each temporal bin with this group's spike, find which original spike it corresponds to
            group_temporal_bins = tf.gather(
                temporal_bin_indices_flat, original_group_indices
            )
            group_batch_indices = tf.gather(batch_indices, original_group_indices)

            rangeIndices = tf.range(tf.shape(vals["group" + str(group)])[0]) + 1

            # Create linear indices for the temporal structure
            # Total size is now batch_size * n_temporal_bins
            (spikePosition[:, 0] * n_temporal_bins + spikePosition[:, 1])

            # Map: for each position in spikePosition, which original spike index to use
            # Build lookup: (batch, temporal_bin) -> original_spike_index
            group_linear_keys = (
                group_batch_indices * n_temporal_bins + group_temporal_bins
            )

            lookup_sparse = tf.sparse.SparseTensor(
                indices=tf.expand_dims(group_linear_keys, 1),
                values=rangeIndices,
                dense_shape=[batch_size * n_temporal_bins],
            )
            lookup_dense = tf.cast(
                tf.sparse.to_dense(lookup_sparse, default_value=0), dtype=tf.int32
            )

            # The indices tensor: for each position in flattened [batch, temporal_bin],
            # which spike index to gather (0 means use zeroForGather)
            vals.update(
                {
                    "indices" + str(group): lookup_dense,
                    "n_temporal_bins": n_temporal_bins,  # Same for all groups
                }
            )

        # Create zero tensor for gathering
        zeroForGather = tf.zeros([1, self.params.nFeatures])

        vals.update(
            {
                "zeroForGather": zeroForGather,
                "spike_times": spike_times,  # Keep original times for reference
                "temporal_bin_size": temporal_bin_size,
                "n_temporal_bins": n_temporal_bins,
            }
        )

        if addLinearizationTensor:
            vals.update(
                {"mazePoints": self.mazePoints_tensor, "tsProj": self.tsProjTensor}
            )

        return vals

    def losses_fig(self, trainLosses, folderModels, fullModel=True, valLosses=[]):
        if fullModel:
            # Save the data
            df = pd.DataFrame(trainLosses)
            df.to_csv(os.path.join(folderModels, "full", "fullModelLosses.csv"))
            # Plot the figure'
            fig, ax = plt.subplots()
            ax.plot(trainLosses[:, 0], label="train losses")
            ax.set_title("position loss")
            ax.plot(valLosses[:, 0], label="validation position loss", c="orange")
            # ax[1].plot(trainLosses[:, 1], label="train loss prediction loss")
            # ax[1].set_title("loss predictor loss")
            # ax[1].plot(valLosses[:, 1], label="validation loss prediction loss")
            fig.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(folderModels, "full", "fullModelLosses.png"))
        else:
            # Save the data
            df = pd.DataFrame(trainLosses)
            df.to_csv(os.path.join(folderModels, "predLoss", "predLossModelLosses.csv"))
            # Plot the figure
            fig, ax = plt.subplots()
            ax.plot(trainLosses[:, 0])
            if list(valLosses):
                ax.plot(valLosses)
            fig.tight_layout()
            fig.savefig(
                os.path.join(folderModels, "predLoss", "predLossModelLosses.png")
            )

    def saveResults(
        self,
        test_output,
        folderName=36,
        sleep=False,
        sleepName="Sleep",
        phase=None,
        save_as_pickle=True,
    ):
        # Manage folders to save
        if sleep:
            folderToSave = os.path.join(
                self.folderResultSleep, str(folderName), sleepName
            )
            phase = ""
            if not os.path.isdir(folderToSave):
                os.makedirs(folderToSave)
        else:
            folderToSave = os.path.join(self.folderResult, str(folderName))

        if phase is not None:
            suffix = f"_{phase}" if phase != "" else ""
        else:
            suffix = self.suffix

        # predicted coordinates
        df = pd.DataFrame(test_output["featurePred"])
        df.to_csv(os.path.join(folderToSave, f"featurePred{suffix}.csv"))

        if "Hn" in test_output:
            df = pd.DataFrame(test_output["Hn"])
            df.to_csv(os.path.join(folderToSave, f"Hn{suffix}.csv"))
        if "maxp" in test_output:
            df = pd.DataFrame(test_output["maxp"])
            df.to_csv(os.path.join(folderToSave, f"maxp{suffix}.csv"))
        # True coordinates
        if not sleep:
            df = pd.DataFrame(test_output["featureTrue"])
            df.to_csv(os.path.join(folderToSave, f"featureTrue{suffix}.csv"))
            # Position loss
            df = pd.DataFrame(test_output["posLoss"])
            df.to_csv(os.path.join(folderToSave, f"posLoss{suffix}.csv"))
        # Times of prediction
        df = pd.DataFrame(test_output["times"])
        df.to_csv(os.path.join(folderToSave, f"timeStepsPred{suffix}.csv"))
        # Index of spikes relative to positions
        df = pd.DataFrame(test_output["posIndex"])
        df.to_csv(os.path.join(folderToSave, f"posIndex{suffix}.csv"))

        # Save additional metrics
        if "metrics" in test_output:
            import json

            # Convert numpy types to native python types for JSON serialization
            metrics_serializable = {
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in test_output["metrics"].items()
            }
            with open(os.path.join(folderToSave, f"metrics{suffix}.json"), "w") as f:
                json.dump(metrics_serializable, f, indent=4)

        # Save residuals if present
        if "residuals" in test_output:
            df = pd.DataFrame(test_output["residuals"])
            df.to_csv(os.path.join(folderToSave, f"residuals{suffix}.csv"))
        df = pd.DataFrame(test_output["posIndex"])
        df.to_csv(os.path.join(folderToSave, f"posIndex{suffix}.csv"))
        # Speed mask
        if not sleep:
            df = pd.DataFrame(test_output["speedMask"])
            df.to_csv(os.path.join(folderToSave, f"speedMask{suffix}.csv"))

        if "indexInDat" in test_output:
            df = pd.DataFrame(test_output["indexInDat"])
            df.to_csv(os.path.join(folderToSave, f"indexInDat{suffix}.csv"))
        if "projPred" in test_output:
            df = pd.DataFrame(test_output["projPred"])
            df.to_csv(os.path.join(folderToSave, f"projPredFeature{suffix}.csv"))
        if "linearPred" in test_output:
            df = pd.DataFrame(test_output["linearPred"])
            df.to_csv(os.path.join(folderToSave, f"linearPred{suffix}.csv"))
        if not sleep:
            if "projTruePos" in test_output:
                df = pd.DataFrame(test_output["projTruePos"])
                df.to_csv(os.path.join(folderToSave, f"projTrueFeature{suffix}.csv"))
            if "linearTrue" in test_output:
                df = pd.DataFrame(test_output["linearTrue"])
                df.to_csv(os.path.join(folderToSave, f"linearTrue{suffix}.csv"))

        if save_as_pickle:
            # save the whole results dictionary
            filename = os.path.join(folderToSave, f"decoding_results{suffix}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(test_output, f, pickle.HIGHEST_PROTOCOL)

    def setup_training_data(self, **kwargs):
        # Unpack kwargs
        behaviorData = kwargs.get("behaviorData", None)

        if behaviorData is None:
            raise ValueError(
                "You must provide behaviorData to setup dynamic dense loss."
            )

        speedMask = behaviorData["Times"]["speedFilter"]
        epochMask = inEpochsMask(
            behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
        )
        totMask = speedMask * epochMask
        full_training_true_positions = behaviorData["Positions"][totMask, :2]
        self.training_data = full_training_true_positions

    def setup_dynamic_dense_loss(self, **kwargs):
        """
        Call this ONCE before training to fit the DenseWeight model
        """
        from neuroencoders.fullEncoder.nnUtils import DenseLossProcessor

        alpha = kwargs.get("alpha", 1.3)
        verbose = kwargs.get("verbose", False)
        self.dynamicdense_verbose = verbose

        if verbose:
            print("Setting up Dynamic Dense Loss...")

        # Create the processor
        self.dense_loss_processor = DenseLossProcessor(
            maze_points=self.maze_points,
            ts_proj=self.ts_proj,
            alpha=alpha,
            verbose=verbose,
            device=self.deviceName,
        )
        # Fit DenseWeight model on full training dataset
        self.setup_training_data(**kwargs)
        self.dense_loss_processor.fit_dense_weight_model(self.training_data)
        self.training_weights = self.dense_loss_processor.training_weights
        self.linearized_training = self.dense_loss_processor.linearized_training

        # Set up components for your existing code
        self.linearization_layer = self.dense_loss_processor.linearization_layer
        self.weights_layer = self.dense_loss_processor.get_weights_layer()
        import termplotlib as tpl

        # Store the fitted dynamic weights

        self.dw = self.dense_loss_processor.fitted_dw

        if verbose:
            print("✓ Dynamic Dense Loss ready!")
            fig = tpl.figure()
            fig.plot(
                self.linearized_training,
                self.training_weights,
                label="weight of linearized position due to imbalance",
            )
            fig.show()

    # Your existing loss computation (now works with dynamic weights):
    def apply_dynamic_dense_loss(self, temp_pos_loss, true_pos):
        """
        Your existing code - now dynamically computes weights for each batch
        """
        if hasattr(self, "dw") and hasattr(self, "linearization_layer"):
            print("Applying Dynamic Dense Loss reweighting...")

            # Get linearized position for current batch
            _, linearized_pos = self.linearization_layer(true_pos[:, :2])
            if self.dynamicdense_verbose:
                print(f"Loss shape: {temp_pos_loss.shape}")
                print(f"Linearized pos shape: {linearized_pos.shape}")

            # Dynamically compute weights using fitted DenseWeight model
            # This calls the fitted model with current batch samples
            weightings = self.weights_layer(linearized_pos)
            if self.dynamicdense_verbose:
                print(f"Dynamic weights shape: {weightings.shape}")

            # Apply Dense Loss: f_w(α, current_batch) * M(ŷ_i, y_i)
            temp_pos_loss = kops.multiply(temp_pos_loss, weightings[:, tf.newaxis])

            if self.dynamicdense_verbose:
                print("✓ Applied Dynamic Dense Loss reweighting")

        return temp_pos_loss

    def setup_gaussian_heatmap(self, **kwargs):
        # Unpack kwargs
        behaviorData = kwargs.get("behaviorData", None)
        if behaviorData is None:
            raise ValueError(
                "You must provide behaviorData to setup Gaussian Heatmap Layer."
            )
        grid_size = kwargs.get(
            "grid_size", getattr(self.params, "GaussianGridSize", (40, 40))
        )
        eps = kwargs.get("eps", getattr(self.params, "GaussianEps", 1e-8))
        sigma = kwargs.get("sigma", getattr(self.params, "GaussianSigma", 0.03))
        neg = kwargs.get("neg", getattr(self.params, "GaussianNeg", -100))
        name = kwargs.get("name", "gaussian_heatmap")

        print("Setting up GaussianHeatmapLayer...")
        speedMask = behaviorData["Times"]["speedFilter"]
        epochMask = inEpochsMask(
            behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
        )
        totMask = speedMask * epochMask
        full_training_true_positions = behaviorData["Positions"][totMask, :2]

        self.gaussian_heatmap_params = {
            "training_positions": full_training_true_positions,
            "grid_size": grid_size,
            "eps": eps,
            "sigma": sigma,
            "neg": neg,
        }
        self.GaussianHeatmap = GaussianHeatmapLayer(
            **self.gaussian_heatmap_params,
            name=name,
            dtype="float32",
        )

    def extract_cnn_model(self):
        """
        Extract CNN feature extractor from the complete model.
        Uses existing CNN layers from the class.

        Returns:
            cnn_model: Model that extracts CNN features from group inputs
        """

        # Get CNN input layers
        cnn_inputs = [self.inputsToSpikeNets[i] for i in range(self.params.nGroups)]

        # Get CNN output layers - the outputs of your spikeNets
        cnn_outputs = []
        for group in range(self.params.nGroups):
            x = self.inputsToSpikeNets[group]
            cnn_output = self.spikeNets[group](x)
            cnn_outputs.append(cnn_output)

        # Create CNN model
        cnn_model = tf.keras.Model(
            inputs=cnn_inputs, outputs=cnn_outputs, name="cnn_feature_extractor"
        )

        print(f"CNN Model created with {len(cnn_model.layers)} layers")
        print("CNN Model summary:")
        cnn_model.summary()

        return cnn_model

    def extract_transformer_model(self):
        """
        Extract transformer part using the shared transformer logic.
        Creates new model that takes CNN features as input.

        Returns:
            transformer_model: Model that processes CNN features through transformer
        """

        with nnUtils.get_device_context(self.deviceName):
            # Create new inputs for transformer model
            cnn_feature_inputs = [
                tf.keras.layers.Input(
                    shape=(self.params.nFeatures,), name=f"cnn_features_{i}"
                )
                for i in range(self.params.nGroups)
            ]

            # Other inputs needed by transformer
            groups_input = tf.keras.layers.Input(shape=(), name="groups", dtype="int32")
            pos_input = tf.keras.layers.Input(shape=(2,), name="pos")

            # Create indices inputs (these would normally come from self.indices)
            indices_inputs = [
                tf.keras.layers.Input(shape=(), name=f"indices_{i}", dtype="int32")
                for i in range(self.params.nGroups)
            ]

            # Recreate the feature gathering and concatenation logic
            allFeatures = []
            for group in range(self.params.nGroups):
                filledFeatureTrain = tf.gather(
                    tf.concat([self.zeroForGather, cnn_feature_inputs[group]], axis=0),
                    indices_inputs[group],
                    axis=0,
                )

                filledFeatureTrain = tf.reshape(
                    filledFeatureTrain,
                    [self.params.batch_size, -1, self.params.nFeatures],
                )
                allFeatures.append(filledFeatureTrain)

            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2, name="concat_CNNs")

            # Create mask
            batchedInputGroups = tf.reshape(groups_input, [self.params.batch_size, -1])
            mymask = tf.not_equal(batchedInputGroups, -1)

            # Store raw features and apply dropout
            allFeatures_raw = allFeatures
            allFeatures = self.dropoutLayer(allFeatures)

            # Use the shared transformer logic
            myoutputPos, outputPredLoss, sumFeatures = (
                self.apply_transformer_architecture(
                    allFeatures, allFeatures_raw, mymask
                )
            )

            # Create all inputs for the transformer model
            all_transformer_inputs = (
                cnn_feature_inputs + indices_inputs + [groups_input, pos_input]
            )

            # Create transformer model
            transformer_model = tf.keras.Model(
                inputs=all_transformer_inputs,
                outputs=[myoutputPos, outputPredLoss],
                name="transformer_model",
            )

            print(
                f"Transformer Model created with {len(transformer_model.layers)} layers"
            )
            print("Transformer Model summary:")
            transformer_model.summary()

            return transformer_model

    def create_separated_models(self):
        """
        Main method to create separated CNN and Transformer models.

        Returns:
            tuple: (cnn_model, transformer_model)
        """

        print("=" * 60)
        print("EXTRACTING CNN MODEL")
        print("=" * 60)
        cnn_model = self.extract_cnn_model()

        print("\n" + "=" * 60)
        print("EXTRACTING TRANSFORMER MODEL")
        print("=" * 60)
        transformer_model = self.extract_transformer_model()

        print("\n" + "=" * 60)
        print("MODELS CREATED SUCCESSFULLY")
        print("=" * 60)

        return cnn_model, transformer_model

    def fine_tune_transformer(
        self, transformer_model, train_data, val_data, epochs=20, learning_rate=1e-4
    ):
        """
        Fine-tune transformer model with pre-extracted CNN features

        Args:
            transformer_model: Extracted transformer model
            train_data: Training data tuple (inputs, targets)
            val_data: Validation data tuple (inputs, targets)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """

        print("Fine-tuning Transformer model...")

        # Compile model
        transformer_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=["mse", "mse"],  # For position and loss prediction
            loss_weights=[1.0, 0.1],  # Adjust based on your needs
            metrics=["mae"],
        )

        # Train
        history = transformer_model.fit(
            train_data[0],  # inputs
            train_data[1],  # targets
            validation_data=val_data,
            epochs=epochs,
            verbose=1,
            batch_size=self.params.batch_size,
        )

        return history

    def train_subject_cnn(
        self, cnn_model, train_data, val_data, epochs=10, learning_rate=1e-5
    ):
        """
        Train CNN model for specific subject

        Args:
            cnn_model: Extracted CNN model
            train_data: Training data tuple (inputs, targets)
            val_data: Validation data tuple (inputs, targets)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """

        print("Training CNN for specific subject...")

        # Compile CNN model
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=["mse"] * self.params.nGroups,  # One loss per group
            metrics=["mae"],
        )

        # Train
        history = cnn_model.fit(
            train_data[0],  # inputs
            train_data[1],  # targets
            validation_data=val_data,
            epochs=epochs,
            verbose=1,
            batch_size=self.params.batch_size,
        )

        return history

    def save_separated_models(
        self, cnn_model, transformer_model, base_name="separated"
    ):
        """Save the separated models"""

        cnn_path = f"{base_name}_cnn.h5"
        transformer_path = f"{base_name}_transformer.h5"

        cnn_model.save(cnn_path)
        transformer_model.save(transformer_path)

        print(f"CNN model saved: {cnn_path}")
        print(f"Transformer model saved: {transformer_path}")

        return cnn_path, transformer_path

    def load_separated_models(self, cnn_path, transformer_path):
        """Load separated models"""

        cnn_model = tf.keras.models.load_model(cnn_path)
        transformer_model = tf.keras.models.load_model(transformer_path)

        print(f"CNN model loaded: {cnn_path}")
        print(f"Transformer model loaded: {transformer_path}")

        return cnn_model, transformer_model

    def inference_with_separated_models(
        self,
        cnn_model,
        transformer_model,
        group_data,
        indices_data,
        groups_data,
        pos_data,
    ):
        """
        Perform inference using separated models

        Args:
            cnn_model: CNN feature extractor
            transformer_model: Transformer model
            group_data: List of group input data
            indices_data: List of indices for each group
            groups_data: Groups data
            pos_data: Position data

        Returns:
            predictions: [position_pred, loss_pred]
        """

        # Extract CNN features
        cnn_features = cnn_model.predict(group_data)

        # Combine with other inputs for transformer
        transformer_inputs = (
            list(cnn_features) + list(indices_data) + [groups_data, pos_data]
        )

        # Get final predictions
        predictions = transformer_model.predict(transformer_inputs)
        return predictions

    @classmethod
    def clear_session(cls):
        tf.keras.backend.clear_session()


########### END OF HELPING LSTMandSpikeNetwork FUNCTIONS#####################
