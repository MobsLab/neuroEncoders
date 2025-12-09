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
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
from typing import Dict, Optional

# Get common libraries
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops as kops
from keras.layers import Lambda
from tqdm import tqdm

import wandb

# Get utility functions
from neuroencoders.fullEncoder import nnUtils
from neuroencoders.fullEncoder.nnUtils import (
    GaussianHeatmapLayer,
    GaussianHeatmapLosses,
    LinearPosWeighting,
    LinearizationLayer,
    MemoryUsageCallbackExtended,
    MultiColumnLossLayer,
    NeuralDataAugmentation,
    UMazeProjectionLayer,
    _get_loss_function,
    create_flatten_augmented_groups_fn,
)
from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.utils.global_classes import DataHelper, Params, Project
from wandb.integration.keras import WandbMetricsLogger


# We generate a model with the functional Model interface in tensorflow
########### START OF FULL NETWORK CLASS #####################
class LSTMandSpikeNetwork:
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
        phase: str = None,
        **kwargs,
    ):
        super(LSTMandSpikeNetwork, self).__init__()
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

        # Moved the initialization of the DataHelper here
        if kwargs.get("linearizer", None) is not None:
            Linearizer = kwargs["linearizer"]
            self.fix_linearizer(Linearizer.mazePoints, Linearizer.tsProj)
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
            # assert not params.denseweight, (
            #     "Cannot use both GaussianHeatmap and DenseWeight"
            # )
            if kwargs.get("behaviorData", None) is None:
                warnings.warn(
                    '"behaviorData" not provided, using default setup WITHOUT Gaussian Heatmap layering. Is your code version deprecated?'
                )
            else:
                self.l_function_layer = LinearizationLayer(
                    maze_points=self.maze_points,
                    ts_proj=self.ts_proj,
                    device=self.deviceName,
                    name="l_function",
                )
                self.setup_gaussian_heatmap(**kwargs)

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
            # WARNING: if the target is not position, this might change
            # this is very dirty, but we need to hardcode the position dimension to 2 for the TFRecord parsing
            # then it will be modified by the DataHelper.get_true_target method to actually match the target.
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

    def _build_model(self, **kwargs):
        ### Description of layers here
        with tf.device(self.deviceName):
            self.inputsToSpikeNets = [
                tf.keras.layers.Input(
                    shape=(
                        self.params.nChannelsPerGroup[group],
                        32,
                    ),  # + batch size, which will be batchSize * maxNbOfSpikes
                    # the shape is N, 32 bc the voltage values are discretized over 32 time bins for each channel (4 most of the time)
                    # of each spike of a given group in the window
                    # we measure the voltage in 32 time steps windows. See the julia code.
                    name="group" + str(group),
                    dtype=tf.float16 if self.params.usingMixedPrecision else tf.float32,
                    # If we use mixed precision, we need to specify the type of the inputs
                    # We use float16 for the inputs to the spike nets
                )
                for group in range(self.params.nGroups)
            ]

            self.inputGroups = tf.keras.layers.Input(shape=(), name="groups")
            self.indices = [
                tf.keras.layers.Input(
                    shape=(), name="indices" + str(group), dtype=tf.int32
                )
                for group in range(self.params.nGroups)
            ]

            # The spike nets acts on each group separately; to reorganize all these computations we use
            # an identity matrix which shape is the total number of spike measured (over all groups)
            self.zeroForGather = tf.keras.layers.Input(
                shape=(self.params.nFeatures,),
                name="zeroForGather",
                dtype=tf.float16 if self.params.usingMixedPrecision else tf.float32,
            )  # the actual zero tensor is created in self.create_indices in train/test calls.

            # Declare spike nets for the different groups:
            self.spikeNets = [
                nnUtils.spikeNet(
                    nChannels=self.params.nChannelsPerGroup[group],
                    device=self.deviceName,
                    nFeatures=self.params.nFeatures,
                    number=str(group),
                    batch_normalization=False,
                    reduce_dense=getattr(self.params, "reduce_dense", False),
                    no_cnn=getattr(self.params, "no_cnn", False),
                )
                for group in range(self.params.nGroups)
            ]
            self.dropoutLayer = tf.keras.layers.Dropout(
                kwargs.get("dropoutCNN", self.params.dropoutCNN)
            )
            self.lstmdropOutLayer = tf.keras.layers.Dropout(
                kwargs.get("dropoutLSTM", self.params.dropoutLSTM)
            )

            # LSTMs
            if not kwargs.get("isTransformer", False):
                self.isTransformer = False
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
                from neuroencoders.fullEncoder.nnUtils import (
                    MaskedGlobalAveragePooling1D,
                    PositionalEncoding,
                    TransformerEncoderBlock,
                )

                dim_factor = getattr(
                    self.params, "dim_factor", 1
                )  # factor to increase the dimension of the transformer if needed
                print("dim_factor:", dim_factor)
                print(
                    "project transformer:",
                    getattr(self.params, "project_transformer", True),
                )

                self.lstmsNets = (
                    [
                        PositionalEncoding(
                            d_model=self.params.nFeatures * dim_factor
                            if getattr(self.params, "project_transformer", True)
                            else self.params.nFeatures * self.params.nGroups,
                            # if we dont shrink the feature dimension before feeding to the transformer, we need to account for the nGroups factor
                            device=self.deviceName,
                        )
                    ]
                    + [
                        TransformerEncoderBlock(
                            d_model=self.params.nFeatures * dim_factor
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

            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Pierre
            self.truePos = tf.keras.layers.Input(
                shape=(self.params.dimOutput,), name="pos"
            )
            self.epsilon = tf.constant(10 ** (-8))
            # Outputs
            print("Output dimension:", self.params.dimOutput)
            self.denseFeatureOutput = tf.keras.layers.Dense(
                self.params.dimOutput - 2
                if getattr(
                    self.params, "GaussianHeatmap", False
                )  # if we have a heatmap and other vars, only the others are predicted here
                and self.params.dimOutput > 2
                else self.params.dimOutput,
                activation=getattr(
                    self.params, "featureActivation", None
                ),  # ensures output is in [0,1]
                dtype=tf.float32,
                name="feature_output",
                kernel_regularizer="l2",
            )
            dim_factor = getattr(
                self.params, "dim_factor", 1
            )  # factor to increase the dimension of the transformer if needed

            if getattr(self.params, "project_transformer", True):
                self.transformer_projection_layer = tf.keras.layers.Dense(
                    self.params.nFeatures * dim_factor,
                    activation="relu",
                    dtype=tf.float32,
                    name="feature_projection_transformer",
                )
            self.ProjectionInMazeLayer = UMazeProjectionLayer(
                grid_size=kwargs.get(
                    "grid_size", getattr(self.params, "GaussianGridSize", (40, 40))
                ),
            )

            # Gather the full model
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
            self.predLossModel = self.compile_model(
                outputs, predLossOnly=True, modelName="predLossModel.pdf", **kwargs
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

        print("Using Transformer architecture !")
        masked_features = Lambda(
            lambda t: kops.where(
                kops.expand_dims(t[0], axis=-1), t[1], kops.zeros_like(t[1])
            )
        )([mymask, allFeatures_raw])

        if getattr(self.params, "project_transformer", True):
            # 1. Projection layer
            allFeatures = self.transformer_projection_layer(allFeatures)
            sumFeatures = kops.sum(
                self.transformer_projection_layer(masked_features), axis=1
            )
        else:
            sumFeatures = kops.sum(masked_features, axis=1)

        # 2. Positional encoding
        allFeatures = self.lstmsNets[0](allFeatures)

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
        mymask = kops.cast(mymask, dtype=tf.float32)
        output = self.lstmsNets[-3](
            output, mask=mymask
        )  # pooling (size [batch, nFeatures*nGroups])

        # finally, normalize the output on the unit-hypersphere
        output = tf.keras.layers.UnitNormalization(axis=-1)(output)

        x = self.lstmsNets[-2](
            output
        )  # dense layer after pooling (size [batch, TransformerDenseSize1])
        x = self.lstmsNets[-1](
            x
        )  # another dense layer after pooling (size [batch, TransformerDenseSize2])

        if not getattr(self.params, "GaussianHeatmap", False):
            x = kops.cast(x, dtype=tf.float32)
            myoutputPos = self.denseFeatureOutput(x)
            if "pos" in self.params.target.lower():
                myoutputPos = self.ProjectionInMazeLayer(
                    myoutputPos
                )  # size [batch, dimOutput]
        else:
            if self.params.dimOutput > 2:
                myoutputPos = self.GaussianHeatmap(
                    x
                )  # outputs a flattened heatmap for better concatenation afterwards
                x = kops.cast(x, dtype=tf.float32)  # size [batch, GRID_H * GRID_W]
                others = self.denseFeatureOutput(x)  # size [batch, dimOutput - 2]
                # this way we have 2 different outputs in the model: a heatmap and some scalars
                myoutputPos = tf.keras.layers.Concatenate(name="heatmap_cat_others")(
                    [myoutputPos, others]
                )
                myoutputPos = kops.cast(myoutputPos, dtype=tf.float32)
            else:
                myoutputPos = self.GaussianHeatmap(
                    x, flatten=False
                )  # we dont need to worry about flattening, it's the only output
                myoutputPos = kops.cast(
                    myoutputPos, dtype=tf.float32
                )  # size [batch, GRID_H, GRID_W]

        return myoutputPos, output, sumFeatures

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

        print("Using LSTM architecture !")
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

        ### Outputs
        myoutputPos = self.denseFeatureOutput(self.dropoutLayer(output))  # positions
        # Projection in UMaze space
        myoutputPos = self.ProjectionInMazeLayer(myoutputPos)

        return myoutputPos, output, sumFeatures

    def generate_model(self, **kwargs):
        """
        Generate the full model with the CNN, LSTM and Dense layers.

        Returns
        -------
        myoutputPos, outputPredLoss, posLoss, uncertaintyLoss
        """
        # CNN plus dense on every group independently
        with tf.device(self.deviceName):
            allFeatures = []  # store the result of the CNN computation for each group
            batchSize = kwargs.get("batchSize", self.params.batchSize)
            for group in range(self.params.nGroups):
                # FIX: otherwise just use TimeDistributed layer
                # keep self.inputsToSpikeNets[group] in [batch, nSpikes, nChannels, 32] and then
                # x = TimeDistributed(self.spikeNets[group])(x)  # [batch, max_nSpikes, cnn_dim] --> get rid of the gather, reshape...
                x = self.inputsToSpikeNets[group]
                # --> [NbKeptSpike = batchSize * maxNbOfSpikes,nbChannels,31] tensors, created my nnUtils.parse_serialized_sequence(batched = True) in train/test calls.
                x = self.spikeNets[group].apply(x)
                # outputs a [NbSpikeOfTheGroup = batchSize * maxNbOfSpikes,nFeatures=self.params.nFeatures(default 64)] tensor.
                # The gather strategy:
                #   extract the final position of the spikes
                # Note: inputGroups is already filled with -1 at position that correspond to filling
                # for batch issues
                # The i-th spike of the group should be positioned at spikePosition[i] in the final tensor
                # We therefore need to    indices[spikePosition[i]] to i  so that it is effectively gather
                # We then gather either a value of
                filledFeatureTrain = kops.take(
                    kops.concatenate([self.zeroForGather, x], axis=0),
                    self.indices[
                        group
                    ],  # self.indices[group] contains the indices where to put the spikes of the group in the final tensor - created by self.create_indices in train/test calls.
                    axis=0,
                )  # give time sense
                # At this point; filledFeatureTrain is a tensor of size (NbBatch*max(nbSpikeInBatch),self.params.nFeatures)
                # where we have filled lines corresponding to spike time of the group
                # with the feature computed by the spike net; and let other time with a value of 0:
                # The index of spike detected then become similar to a time value...
                filledFeatureTrain = kops.reshape(
                    filledFeatureTrain,
                    (int(batchSize), -1, self.params.nFeatures),
                )
                # Reshaping the result of the spike net as batchSize:MaxNbTotSpikeDetected:nFeatures
                # this allow to separate spikes from the same window or from the same batch.
                # if use_time: will be reshaped as batchSize:num_idx(max_spikes):nFeatures
                allFeatures.append(filledFeatureTrain)
            allFeatures = tf.tuple(tensors=allFeatures)
            # synchronizes the computation of all features (like a join)
            # The concatenation is made over axis 2, which is the Feature axis
            # So we reserve columns to each output of the spiking networks...
            allFeatures = kops.concatenate(allFeatures, axis=2)  # , name="concat1"
            # now the shape of allfeatures is (NbBatch, NbTotSpikeDetected, nGroups*nFeatures)
            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = kops.reshape(
                self.inputGroups,
                (
                    batchSize,
                    -1,
                ),  # self.inputGroups has shape (BatchSize * NbTotalSpikes) and is filled with group indices or -1
            )
            mymask = nnUtils.safe_mask_creation(
                batchedInputGroups
            )  # [batch, max_n_spikes] (True for real spikes, False for padded values)

            masked_features = Lambda(
                lambda t: kops.where(
                    kops.expand_dims(t[0], axis=-1),
                    t[1],
                    kops.zeros_like(t[1], dtype=t[1].dtype),
                )
            )([mymask, allFeatures])
            # size is (NbBatch, NbTotSpikeDetected, nGroups*nFeatures)
            sumFeatures = kops.sum(
                masked_features, axis=1
            )  # This var will be used in the predLoss loss

            allFeatures_raw = allFeatures
            allFeatures = self.dropoutLayer(allFeatures)
            # size is (NbBatch, NbTotSpikeDetected, nGroups*nFeatures)

            # LSTM
            # TODO: at some point, analyse the output of lstm/transformer ?
            if not kwargs.get("isTransformer", False):
                myoutputPos, output, sumFeatures = self.apply_lstm_architecture(
                    allFeatures, sumFeatures, mymask, **kwargs
                )
            else:
                # Use shared transformer logic
                myoutputPos, output, sumFeatures = self.apply_transformer_architecture(
                    allFeatures, allFeatures_raw, mymask, **kwargs
                )  # shape (batch_size, dimOutput) or (batch_size, GaussianGridSize[0], GaussianGridSize[1]) or (batch_size, flattened heatmap + dimOutput - 2)

            loss_function = self._parse_loss_function_from_params(myoutputPos)

            #### note
            # bayesian loss function  = sum ((y_true - y_pred)^2 / sigma^2 + log(sigma^2))
            # we assume myoutputPos is in cm x cm,
            # as self.truePos (modulo [0,1] normalization)
            # in ~cm2 as no loss is sqrt or log
            if getattr(self.params, "GaussianHeatmap", False):
                if self.params.dimOutput > 2:
                    # we separate the heatmap from the other variables
                    targets_hw = self.GaussianHeatmap.gaussian_heatmap_targets(
                        self.truePos[:, :2]  # already batched
                    )
                    logits_hw = myoutputPos[
                        :,  # all batch
                        : self.params.GaussianGridSize[0]
                        * self.params.GaussianGridSize[1],  # only the heatmap part
                    ]
                    logits_hw = kops.reshape(
                        logits_hw,
                        (
                            batchSize,
                            self.params.GaussianGridSize[0],
                            self.params.GaussianGridSize[1],
                        ),
                    )
                    loss_inputs = {
                        "logits": logits_hw,
                        "targets": targets_hw,
                    }
                    tempPosLoss_logits = self.GaussianLoss_layer(
                        loss_inputs,
                        loss_type=getattr(self.params, "loss_type", "safe_kl"),
                        return_batch=True,
                    )
                    others = myoutputPos[
                        :,
                        self.params.GaussianGridSize[0]
                        * self.params.GaussianGridSize[1] :,
                    ]
                    tempPosLoss_others = loss_function(others, self.truePos[:, 2:])
                    tempPosLoss = (
                        self.params.heatmap_weight * tempPosLoss_logits
                        + self.params.others_weight * tempPosLoss_others
                    )
                else:
                    targets_hw = self.GaussianHeatmap.gaussian_heatmap_targets(
                        self.truePos
                    )
                    loss_inputs = {
                        "logits": myoutputPos,
                        "targets": targets_hw,
                    }
                    tempPosLoss = self.GaussianLoss_layer(
                        loss_inputs,
                        loss_type=getattr(self.params, "loss_type", "safe_kl"),
                        return_batch=True,
                    )
                tempPosLoss = kops.cast(tempPosLoss, tf.float32)

            else:
                if getattr(self.params, "mixed_loss", False):
                    proj, lin_truePos = self.l_function_layer(self.truePos[:, :2])
                    # weight the first 2 dimensions (x,y positions)  by linPos before computing the loss

                    myoutputPos_weighted = LinearPosWeighting()(
                        [myoutputPos, lin_truePos]
                    )
                    truePos_weighted = LinearPosWeighting()([self.truePos, lin_truePos])
                else:
                    myoutputPos_weighted = myoutputPos
                    truePos_weighted = self.truePos

                tempPosLoss = loss_function(myoutputPos_weighted, truePos_weighted)[
                    :, tf.newaxis
                ]
                tempPosLoss = kops.cast(tempPosLoss, tf.float32)
            # for main loss functions:
            # if loss function is mse
            # tempPosLoss is in cm2
            # if loss function is logcosh
            # tempPosLoss is in cm2

            if self.params.denseweight:
                tempPosLoss = self.apply_dynamic_dense_loss(tempPosLoss, self.truePos)
            if self.params.transform_w_log:
                posLoss = tf.keras.layers.Identity(name="posLoss")(tempPosLoss)
            elif not getattr(self.params, "GaussianHeatmap", False):
                posLoss = tf.keras.layers.Identity(name="posLoss")(tempPosLoss)
            else:
                posLoss = tf.keras.layers.Identity(name="posLoss")(tempPosLoss)

            myoutputPos_named = tf.keras.layers.Identity(name="myoutputPos")(
                myoutputPos
            )

        return myoutputPos_named, posLoss

    def _parse_loss_function_from_params(self, myoutputPos):
        # TODO: change
        ### Multi-column loss configuration
        column_losses = getattr(
            self.params,
            "column_losses",
            {str(i): self.params.loss for i in range(myoutputPos.shape[-1])},
        )
        column_weights = getattr(self.params, "column_weights", {})
        merge_columns = getattr(self.params, "merge_columns", [])
        merge_losses = getattr(self.params, "merge_losses", [])
        merge_weights = getattr(self.params, "merge_weights", [])

        # Create the loss instance
        if len(column_weights) > 1 or any("," in spec for spec in column_losses.keys()):
            print("Using multi-column loss")
            self.loss_function = MultiColumnLossLayer(
                column_losses=column_losses,
                column_weights=column_weights,
                merge_columns=merge_columns,
                merge_losses=merge_losses,
                merge_weights=merge_weights,
                alpha=self.params.alpha,
                delta=self.params.delta,
                gaussian_layer=getattr(self, "GaussianLoss_layer", None),
            )  # actually it's more of a layer than a function

        else:
            # Single loss case
            self.loss_function = _get_loss_function(
                self.params.loss,
                alpha=self.params.alpha,
                delta=self.params.delta,
                gaussian_loss_layer=getattr(self, "GaussianLoss_layer", None),
            )

        return self.loss_function

    def compile_model(
        self, outputs, modelName="FullModel.pdf", predLossOnly=False, **kwargs
    ):
        """
        Compile the model with the desired losses and optimizer.
        The model is then plotted and saved in the results folder.

        Parameters
        ----------
        outputs : list of tensors ( myoutputPos, outputPredLoss, posLoss, uncertaintyLoss )
        modelName : str (default "FullModel.png")
        predLossOnly : bool (default False)

        Returns
        -------
        model : tf.keras.Model
        """
        # Initialize and plot the model
        model = tf.keras.Model(
            inputs=self.inputsToSpikeNets
            + self.indices
            + [self.truePos, self.inputGroups, self.zeroForGather],
            outputs=outputs,
        )

        def pos_loss(x, y):
            return y

        # Compile the model
        # TODO: use params.optimizer instead of hardcoding RMSprop
        # self.optimizer = tf.keras.optimizers.RMSprop(
        #     learning_rate=kwargs.get("lr", self.params.learningRates[0])
        # )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=kwargs.get("lr", self.params.learningRates[0]),
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-09,
        )
        # TODO: something with mixed precision and keras policy ?
        if not predLossOnly:
            # Full model
            self.outNames = ["myoutputPos", "posLoss"]
            # TODO: Adam or AdaGrad?
            # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
            model.compile(
                optimizer=self.optimizer,
                loss={
                    # tf_op_layer_ position loss (eucledian distance between predicted and real coordinates)
                    self.outNames[0]: None,
                    self.outNames[1]: pos_loss,
                },
                jit_compile=False,
                # steps_per_execution=4,
            )
            # Get internal names of losses
        if not os.path.exists(os.path.join(self.projectPath.experimentPath, modelName)):
            try:
                tf.keras.utils.plot_model(
                    model,
                    to_file=(os.path.join(self.projectPath.experimentPath, modelName)),
                    show_shapes=True,
                )
            except Exception as e:
                print("Could not plot the model:", e)
        return model

    def generate_model_Cplusplus(self):
        ### Describe
        with tf.device(self.deviceName):
            allFeatures = []
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                x = self.spikeNets[group].apply(x)
                filledFeatureTrain = tf.gather(
                    tf.concat([self.zeroForGather, x], axis=0),
                    self.indices[group],
                    axis=0,
                )
                filledFeatureTrain = tf.reshape(
                    filledFeatureTrain, [1, -1, self.params.nFeatures]
                )
                allFeatures.append(filledFeatureTrain)
            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2)

            sumFeatures = tf.math.reduce_sum(allFeatures, axis=1)
            # This var will be used in the predLoss loss
            allFeatures = self.dropoutLayer(allFeatures, training=True)
            # LSTM
            for ilstm, lstmLayer in enumerate(self.lstmsNets):
                if ilstm == 0:
                    if len(self.lstmsNets) == 1:
                        output = lstmLayer(allFeatures)
                    else:
                        outputSeq = lstmLayer(allFeatures, training=True)
                        outputSeq = self.dropoutLayer(outputSeq)
                elif ilstm == len(self.lstmsNets) - 1:
                    output = lstmLayer(outputSeq)
                else:
                    outputSeq = lstmLayer(outputSeq, training=True)
                    outputSeq = self.dropoutLayer(outputSeq)
                    output_seq = self.lstmsNets[0](allFeatures)
                    # output_seq = tf.ensure_shape(output_seq, [self.params.batchSize,None, self.params.lstmSize])
                    output_seq = self.dropoutLayer(output_seq, training=True)
                    output_seq = self.lstmsNets[1](output_seq)
                    # output_seq = tf.ensure_shape(output_seq, [self.params.batchSize,None, self.params.lstmSize])
                    output_seq = self.dropoutLayer(output_seq, training=True)
                    output_seq = self.lstmsNets[2](output_seq)
                    # output_seq = tf.ensure_shape(output_seq, [self.params.batchSize,None, self.params.lstmSize])
                    output_seq = self.dropoutLayer(output_seq, training=True)
                    output = self.lstmsNets[3](output_seq)
            output = tf.ensure_shape(output, [batchSize, self.params.lstmSize])
            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(
                self.denseLoss3(
                    self.denseLoss4(
                        self.denseLoss5(
                            self.denseLoss1(
                                tf.stop_gradient(
                                    tf.concat([output, sumFeatures], axis=1)
                                )
                            )
                        )
                    )
                )
            )
        ### Initialize
        self.cplusplusModel = tf.keras.Model(
            inputs=self.inputsToSpikeNets + self.indices + [self.zeroForGather],
            outputs=[myoutputPos, outputLoss],
        )
        tf.keras.utils.plot_model(
            self.cplusplusModel,
            to_file=(
                os.path.join(self.projectPath.experimentPath, "FullModel_Cplusplus.png")
            ),
            show_shapes=True,
        )

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
        onTheFlyCorrection : bool (default False) : normaliize the position data on the fly
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

        def pos_loss(x, y):
            return y

        ### Create neccessary arrays
        windowSizeMS = kwargs.pop("windowSizeMS", 36)
        scheduler = kwargs.get("scheduler", "decay")
        isPredLoss = kwargs.get("isPredLoss", True)
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
        datasets = self._dataset_loading_pipeline(
            filename, windowSizeMS, behaviorData, totMask, augmentation_config, **kwargs
        )

        ### Train the model(s)
        # Train
        for key in checkpointPath.keys():
            print("Training the", key, "model")
            nb_epochs_already_trained = 10
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
                    except Exception as e:
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
            LRScheduler = self.LRScheduler(self.params.learningRates)
            if scheduler == "fixed":
                schedule = tf.keras.callbacks.LearningRateScheduler(
                    LRScheduler.schedule_fixed
                )
            elif scheduler == "decay":
                schedule = tf.keras.callbacks.LearningRateScheduler(
                    LRScheduler.schedule_decay
                )
            else:
                raise ValueError('Learning rate schedule is either "fixed" or "decay"')

            # NOTE: In case you need debugging, toggle this profiling line to True
            is_tbcallback = False
            if self.debug:
                print(
                    "Debugging mode is ON, enabling TensorBoard callback and device placement loggin"
                )
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
                    }
                    ann_config["loaded"] = loaded

                    prefix = "LOADED_" if loaded else ""
                    # wandb.tensorboard.patch(
                    #     root_logdir=os.path.join(self.folderResult, "logs")
                    # )
                    # print(f"starting tensorboard at {self.folderResult}/logs ")
                    run = wandb.init(
                        entity="touseul",
                        project="encore",
                        name=f"{prefix}{os.path.basename(os.path.dirname(self.projectPath.xml))}_{os.path.basename(self.projectPath.experimentPath)}_{key}_{windowSizeMS}ms",
                        notes=f"{os.path.basename(self.projectPath.experimentPath)}_{key}",
                        # sync_tensorboard=True,
                        config=ann_config,
                    )
                    # tf.profiler.experimental.start(
                    #     os.path.join(self.folderResult, "logs")
                    # )
                    # tb_callbacks = tf.keras.callbacks.TensorBoard(
                    #     log_dir=os.path.join(self.folderResult, "logs"),
                    #     histogram_freq=1,
                    #     profile_batch=(2, 4) if is_tbcallback else 0,
                    # )

                    wandb_callback = WandbMetricsLogger()
            if key != "predLoss":
                if earlyStop:
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
                    ]
                else:
                    callbacks = [
                        csvLogger[key],
                        cp_callback,
                        schedule,
                        MemoryUsageCallbackExtended(),
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
                    # callbacks.append(tb_callbacks)
                    callbacks.append(wandb_callback)

                hist = self.model.fit(
                    datasets["train"],
                    epochs=self.params.nEpochs - nb_epochs_already_trained + 10,
                    callbacks=callbacks,  # , tb_callback,cp_callback
                    validation_data=datasets["test"],
                )  # steps_per_epoch = int(self.params.nSteps / self.params.nEpochs)

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
    ) -> Dict[str, tf.data.Dataset]:
        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)
        shuffle = kwargs.get("shuffle", True)
        batchSize = kwargs.get("batch_size", self.params.batchSize)

        def filter_by_pos_index(x):
            return tf.equal(table.lookup(x["pos_index"]), 1.0)

        def filter_nan_pos(x):
            pos_data = x["pos"]
            # convert to float if it's a binary pred
            if pos_data.dtype in [tf.int32, tf.int64]:
                pos_data = tf.cast(pos_data, tf.float64)

            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        def _parse_function(*vals):
            with tf.device(self.deviceName):
                return nnUtils.parse_serialized_spike(self.featDesc, *vals)

        dim_output = (
            self.params.dimOutput
            if not getattr(self.params, "GaussianHeatmap", False)
            else (
                self.params.dimOutput
                - 2
                + self.params.GaussianGridSize[0] * self.params.GaussianGridSize[1]
            )
        )

        @tf.function
        def map_outputs(vals):
            return (
                vals,
                {
                    self.outNames[0]: tf.zeros(
                        (batchSize, dim_output), dtype=tf.float32
                    ),
                    self.outNames[1]: tf.zeros(batchSize, dtype=tf.float32),
                },
            )

        ndataset = tf.data.TFRecordDataset(
            os.path.join(self.projectPath.dataPath, filename)
        )
        # Parse the record into tensors - simply attribute a name to every tensor from featDesc
        ndataset = ndataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ndataset = ndataset.prefetch(tf.data.AUTOTUNE)

        # Create datasets
        if not isinstance(totMask, dict):
            # it means we have just one set of keys
            totMask_backup = totMask.copy()
            totMask = (
                {"test": totMask_backup}
                if kwargs.get("inference_mode", False)
                else {"train": totMask_backup}
            )
        datasets = {}
        for key in totMask.keys():
            # creates a lookup table to filter by pos index, and by totMask (speed + epoch)
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(totMask[key])), dtype=tf.int64),
                    tf.constant(totMask[key], dtype=tf.float32),
                ),
                default_value=0,
            )
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
            dataset = ndataset.filter(filter_by_pos_index)
            dataset = dataset.map(nnUtils.import_true_pos(posFeature))
            dataset = dataset.filter(filter_nan_pos)

            # now that we have clean positions, we can resample if needed
            if self.params.OversamplingResampling and key == "train":
                dataset = self._apply_oversampling_resampling(
                    dataset, windowSizeMS=windowSizeMS, shuffle=shuffle
                )

            if shuffle:
                dataset = dataset.shuffle(100000, reshuffle_each_iteration=True)

            dataset = dataset.batch(batchSize, drop_remainder=True)

            if (
                not self.params.dataAugmentation
                or key == "test"
                or kwargs.get("inference_mode", False)
            ):
                print("No data augmentation for", key, "dataset")
                optimized_parse_fn = self.create_optimized_parse_function(
                    augmentation=False,
                    count_spikes=kwargs.get("extract_spikes_counts", False),
                )
                dataset = dataset.map(
                    optimized_parse_fn,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )  # self.featDesc, *
            else:
                # this way the data augmentation is applied after resampling and batching.
                print("Applying data augmentation to", key, "dataset with config:")
                print(augmentation_config)
                optimized_aug_fn = self.create_optimized_parse_function(
                    augmentation=True,
                    augmentation_config=augmentation_config,
                    count_spikes=kwargs.get("extract_spikes_counts", False),
                )

                dataset = dataset.map(
                    optimized_aug_fn,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )  # self.featDesc, *
                flatten_fn = create_flatten_augmented_groups_fn(
                    self.params, augmentation_config.num_augmentations
                )
                dataset = dataset.flat_map(flatten_fn)  # Flatten the augmented groups

            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            dataset = dataset.map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)
            # cache only once, after all the preprocessing
            print(
                f"finalizing the {key} dataset pipeline... Will move data to {self.deviceName}"
            )
            dataset = dataset.cache().apply(
                tf.data.experimental.copy_to_device(self.deviceName)
            )
            # WARN: this seems to be cached on CPU RAM, not GPU RAM...

            # now dataset is a tuple (inputsDict, outputsDict)
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

            dataset = dataset.with_options(options).prefetch(tf.data.AUTOTUNE)

            datasets[key] = dataset

        return datasets

    def create_optimized_parse_function(
        self,
        augmentation: bool = False,
        augmentation_config: Optional[NeuralDataAugmentation] = None,
        count_spikes=False,
    ):
        """
        Create optimized parsing function that respects spike data structure
        """

        if augmentation:

            @tf.function(experimental_relax_shapes=True)
            def optimized_parse_with_augmentation(batch_data):
                # Create a COPY to avoid the mutation error
                processed_batch = {}

                # Copy all keys to new dict (avoid mutation)
                for key in batch_data.keys():
                    processed_batch[key] = batch_data[key]

                # Call your existing function but on the copy
                return nnUtils.parse_serialized_sequence_with_augmentation(
                    self.params,
                    processed_batch,  # Pass the copy
                    augmentation_config=augmentation_config,
                    batched=True,
                    count_spikes=count_spikes,
                )

            return optimized_parse_with_augmentation
        else:

            @tf.function(experimental_relax_shapes=True)
            def optimized_parse_standard(batch_data):
                # Same pattern for standard parsing
                processed_batch = {}
                for key in batch_data.keys():
                    processed_batch[key] = batch_data[key]

                return nnUtils.parse_serialized_sequence(
                    self.params,
                    processed_batch,
                    batched=True,
                    count_spikes=count_spikes,
                )

            return optimized_parse_standard

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
        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)
        isPredLoss = kwargs.get("isPredLoss", False)
        speedValue = kwargs.get("speedValue", None)
        phase = kwargs.get("phase", None)
        template = kwargs.get("template", None)
        fit_temperature = kwargs.get("fit_temperature", False)
        T_scaling = kwargs.get("T_scaling", None)
        epochKey = kwargs.get("epochKey", "testEpochs")
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
                )
            except:
                print("loading from savedModels failed, trying full checkpoint ")
                try:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels, str(windowSizeMS), "full" + "/cp.ckpt"
                        ),
                    )
                except:
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
        if useTrain:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
            ) + inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
            )
        else:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"][epochKey]
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

        dataset = self._dataset_loading_pipeline(
            filename,
            windowSizeMS,
            behaviorData,
            totMask,
            inference_mode=True,
            onTheFlyCorrection=onTheFlyCorrection,
            shuffle=False,
            **kwargs,
        )["test"]
        # -------------------------------------------------------------------------
        # CUSTOM SYNCHRONIZED INFERENCE LOOP
        # -------------------------------------------------------------------------
        print("Starting synchronized inference loop...")

        # 1. Initialize accumulators
        # Predictions
        list_pred_features = []
        list_pred_loss = []

        # Metadata / Ground Truth
        list_pos = []
        list_times = []
        list_times_behavior = []
        list_pos_index = []
        list_speed_filter = []
        list_index_in_dat = []
        list_index_in_dat_raw = []

        # Spike Counts (Dynamic dict to handle variable groups)
        dict_spike_counts = {
            f"group{g}_spikes_count": [] for g in range(self.params.nGroups)
        }

        # 2. Iterate ONCE over the dataset
        # This locks inputs and metadata together for every batch
        for batch in tqdm(dataset, desc="Inferring"):
            # Determine structure: Dataset usually yields (inputs, targets)
            # Based on your map functions, 'inputs' is a dictionary containing 'pos', 'time', etc.
            if isinstance(batch, tuple):
                inputs = batch[0]
                # targets = batch[1] # Unused here, but available
            else:
                inputs = batch

            # A. Run Model Prediction
            # training=False ensures Dropout/BatchNorm behave correctly for inference
            preds = self.model(inputs, training=False)

            # Handle model returning tuple (Features, Loss) vs single output
            if isinstance(preds, (list, tuple)):
                pred_feat_batch = preds[0]
                pred_loss_batch = preds[1]
            else:
                pred_feat_batch = preds
                pred_loss_batch = None

            # B. Store Predictions (move to CPU/numpy immediately to save GPU mem)
            list_pred_features.append(pred_feat_batch.numpy())
            if pred_loss_batch is not None:
                list_pred_loss.append(pred_loss_batch.numpy())

            # C. Store Metadata from the 'inputs' dict
            # We look for keys directly in the input batch that generated the prediction
            list_pos.append(inputs["pos"].numpy())
            list_times.append(inputs["time"].numpy())
            list_times_behavior.append(inputs["time_behavior"].numpy())
            list_pos_index.append(inputs["pos_index"].numpy())
            list_index_in_dat.append(inputs["indexInDat"].numpy())

            # Optional keys (use .get or check)
            if "speedFilter" in inputs:
                list_speed_filter.append(inputs["speedFilter"].numpy())

            if extract_spikes_counts:
                list_index_in_dat_raw.append(inputs["indexInDat_raw"].numpy())
                for g in range(self.params.nGroups):
                    key = f"group{g}_spikes_count"
                    if key in inputs:
                        dict_spike_counts[key].append(inputs[key].numpy())

        # 3. Concatenate all batches into single arrays
        print("Concatenating results...")
        full_pred_features = np.concatenate(list_pred_features, axis=0)

        # Handle Pos Loss
        if len(list_pred_loss) > 0:
            full_pos_loss = np.concatenate(list_pred_loss, axis=0)
        else:
            full_pos_loss = None  # Or empty array depending on downstream needs

        full_feature_true = np.concatenate(list_pos, axis=0)
        full_times = np.concatenate(list_times, axis=0).flatten()
        full_times_behavior = np.concatenate(list_times_behavior, axis=0).flatten()
        full_pos_index = np.concatenate(list_pos_index, axis=0).flatten()
        full_index_in_dat = np.concatenate(list_index_in_dat, axis=0)

        # Handle Speed Mask
        # If speedFilter was in dataset, use it. Otherwise compute via lookup
        if len(list_speed_filter) > 0:
            windowmaskSpeed = np.concatenate(list_speed_filter, axis=0).flatten()
        else:
            # Fallback to your original lookup method
            print("Looking up speed mask from original array...")
            windowmaskSpeed = speedMask[full_pos_index]

        # -------------------------------------------------------------------------
        # POST-PROCESSING (Heatmaps, etc.)
        # -------------------------------------------------------------------------

        outputTest = (full_pred_features, full_pos_loss)

        # Handle Direction classification target
        if self.target.lower() == "direction":
            outputTest = (tf.cast(outputTest[0] > 0.5, tf.int32),)

        # Handle Gaussian Heatmap Decoding
        if getattr(self.params, "GaussianHeatmap", False):
            print("Decoding Gaussian Heatmaps...")
            if self.params.dimOutput > 2:
                # Split logits vs other outputs
                grid_size = self.GaussianHeatmap.GRID_H * self.GaussianHeatmap.GRID_H
                output_logits = outputTest[0][:, :grid_size]
                output_logits = tf.reshape(
                    output_logits,
                    (-1, self.GaussianHeatmap.GRID_H, self.GaussianHeatmap.GRID_W),
                )
                others = outputTest[0][:, grid_size:]
            else:
                output_logits = outputTest[0]
                others = None

            # Initial Decode
            xy, maxp, Hn, var_total = self.GaussianHeatmap.decode_and_uncertainty(
                output_logits
            )

            # Temperature Scaling Logic
            if fit_temperature:
                featureTrue = np.reshape(
                    full_feature_true, [xy.shape[0], self.params.dimOutput]
                )
                featureTruePos = featureTrue[:, :2]
                val_targets = self.GaussianHeatmap.gaussian_heatmap_targets(
                    featureTruePos
                )
                T_scaling = self.GaussianHeatmap.fit_temperature(
                    output_logits, val_targets, iters=400
                )
                return T_scaling

            if T_scaling is not None:
                output_logits = output_logits / T_scaling
                xy, maxp, Hn, var_total = self.GaussianHeatmap.decode_and_uncertainty(
                    output_logits
                )

            # Reassemble output
            if self.params.dimOutput > 2:
                total_output = tf.concat([xy, others], axis=-1)
            else:
                total_output = xy

            # Update outputTest with decoded XY
            outputTest = (total_output.numpy(), outputTest[1])

        # Ensure Feature True shape is correct
        featureTrue = np.reshape(full_feature_true, [outputTest[0].shape[0], -1])

        # -------------------------------------------------------------------------
        # PACKAGING OUTPUTS
        # -------------------------------------------------------------------------

        testOutput = {
            "featurePred": outputTest[0],
            "featureTrue": featureTrue,
            "times": full_times,
            "times_behavior": full_times_behavior,
            "posLoss": full_pos_loss,
            "posIndex": full_pos_index,
            "speedMask": windowmaskSpeed,
        }

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
                full_index_raw = np.concatenate(list_index_in_dat_raw, axis=0)

                # Construct DataFrame directly from the arrays (Much faster than row-loop)
                data_dict = {
                    "posIndex": full_pos_index,
                    # Convert list of arrays/lists to string or keep as object for indexInDat
                    "indexInDat": [x.tolist() for x in full_index_raw],
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
            projPredPos, linearPred = l_function(outputTest[0][:, :2])
            projTruePos, linearTrue = l_function(featureTrue[:, :2])
            testOutput["projPred"] = projPredPos
            testOutput["projTruePos"] = projTruePos
            testOutput["linearPred"] = linearPred
            testOutput["linearTrue"] = linearTrue

        if getattr(self.params, "GaussianHeatmap", False):
            # add uncertainty and confidence metrics to output dict
            testOutput["logits_hw"] = (
                output_logits.numpy()
                if hasattr(output_logits, "numpy")
                else output_logits
            )
            testOutput["var_total"] = (
                var_total.numpy() if hasattr(var_total, "numpy") else var_total
            )
            testOutput["Hn"] = Hn.numpy() if hasattr(Hn, "numpy") else Hn
            testOutput["maxp"] = maxp.numpy() if hasattr(maxp, "numpy") else maxp
            testOutput["T_scaling"] = (
                (T_scaling.numpy() if hasattr(T_scaling, "numpy") else T_scaling)
                if T_scaling is not None
                else None
            )

        # Save the results
        self.saveResults(testOutput, folderName=windowSizeMS, phase=phase)

        return testOutput

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
                )
            except:
                print("loading from savedModels failed, trying full checkpoint ")
                try:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels, str(windowSizeMS), "full" + "/cp.ckpt"
                        ),
                    )
                except:
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

            @tf.autograph.experimental.do_not_convert
            def map_parse_serialized_sequence(*vals):
                return nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                )

            dim_output = (
                self.params.dimOutput
                if not getattr(self.params, "GaussianHeatmap", False)
                else (
                    self.params.dimOutput
                    - 2
                    + self.params.GaussianGridSize[0] * self.params.GaussianGridSize[1]
                )
            )

            @tf.function
            def map_outputs(vals):
                return (
                    vals,
                    {
                        self.outNames[0]: tf.zeros(
                            (self.params.batchSize, dim_output), dtype=tf.float32
                        ),
                        self.outNames[1]: tf.zeros(
                            self.params.batchSize, dtype=tf.float32
                        ),
                    },
                )

            dataset = dataset.filter(filter_by_time)
            dataset = dataset.batch(self.params.batchSize, drop_remainder=True)

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
            output = self.model.predict(dataset, verbose=1)

            if self.target.lower() == "direction":
                output = (tf.cast(output[0] > 0.5, tf.int32),)

            if getattr(self.params, "GaussianHeatmap", False):
                if self.params.dimOutput > 2:
                    output_logits = output[0][
                        :, : self.GaussianHeatmap.GRID_H * self.GaussianHeatmap.GRID_H
                    ]
                    output_logits = tf.reshape(
                        output_logits,
                        (
                            output_logits.shape[0],
                            self.GaussianHeatmap.GRID_H,
                            self.GaussianHeatmap.GRID_W,
                        ),
                    )  # logits with shape (batch, H, W)
                    others = output[0][
                        :, self.GaussianHeatmap.GRID_H * self.GaussianHeatmap.GRID_H :
                    ]  # other outputs (speed, ...) with shape (batch, dimOutput-2)
                else:
                    # output_logits is already (batch, H, W) because flatten = False
                    output_logits = output[0]

                xy, maxp, Hn, var_total = self.GaussianHeatmap.decode_and_uncertainty(
                    output_logits
                )
                if T_scaling is not None:
                    output_logits = output_logits / T_scaling

                if self.params.dimOutput > 2:
                    # concatenate xy with the other outputs (speed, ...)
                    total_output = tf.concat([xy, others], axis=-1)
                else:
                    total_output = xy
                # reconstruct output tuple with xy pred instead of heatmap
                output = (total_output.numpy(), output[1])

            # Post-infer management
            print(f"gathering times of the centre in the time window for {sleepName}")

            @tf.autograph.experimental.do_not_convert
            def map_time(x, y):
                return x["time"]

            datasetTimes = dataset.map(map_time, num_parallel_calls=tf.data.AUTOTUNE)
            times = list(datasetTimes.as_numpy_iterator())
            times = np.ravel(times)
            print(
                f"gathering indices of spikes relative to coordinates for {sleepName}"
            )

            @tf.autograph.experimental.do_not_convert
            def map_pos_index(x, y):
                return x["pos_index"]

            @tf.autograph.experimental.do_not_convert
            def map_index_in_dat(x, y):
                return x["indexInDat"]

            datasetPosIndex = dataset.map(
                map_pos_index, num_parallel_calls=tf.data.AUTOTUNE
            )
            posIndex = list(datasetPosIndex.as_numpy_iterator())
            posIndex = np.ravel(np.array(posIndex))
            #
            datasetIndexInDat = dataset.map(
                map_index_in_dat, num_parallel_calls=tf.data.AUTOTUNE
            )
            IDdat = list(datasetIndexInDat.as_numpy_iterator())

            predictions[sleepName] = {
                "featurePred": output[0],
                "times": times,
                "posIndex": posIndex,
                "indexInDat": IDdat,
            }
            if l_function:
                projPredPos, linearPred = l_function(output[0][:, :2])
                predictions[sleepName]["projPred"] = projPredPos
                predictions[sleepName]["linearPred"] = linearPred

            if getattr(self.params, "GaussianHeatmap", False):
                # add uncertainty and confidence metrics to output dict
                predictions[sleepName]["logits_hw"] = output_logits
                predictions[sleepName]["var_total"] = var_total
                predictions[sleepName]["Hn"] = Hn
                predictions[sleepName]["maxp"] = maxp
                predictions[sleepName]["T_scaling"] = T_scaling

        # Save the results
        for key in predictions.keys():
            self.saveResults(
                predictions[key], folderName=folderName, sleep=True, sleepName=key
            )

    def _apply_oversampling_resampling(self, dataset, windowSizeMS, shuffle=True):
        print("Using oversampling resampling on the training set")

        # FIX: make sure GaussianHeatmap is initialized - force it ?

        GRID_H, GRID_W = (
            self.GaussianHeatmap.GRID_H,
            self.GaussianHeatmap.GRID_W,
        )
        # instead of oversampling on such tiny grid, we take a coarser grid mesh
        stride = 3
        coarse_H, coarse_W = GRID_H // stride, GRID_W // stride

        @tf.autograph.experimental.do_not_convert
        def map_bin_class(ex):
            return nnUtils.bin_class(
                ex,
                GRID_H,
                GRID_W,
                stride,
                self.GaussianHeatmap.forbid_mask_tf,
            )

        # should not be useful as true positions are already allowed!
        dataset = dataset.filter(lambda ex: tf.greater_equal(map_bin_class(ex), 0))

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
        coarse_bins = y_coarse * coarse_W + x_coarse  # shape same as positions
        counts = np.bincount(coarse_bins, minlength=coarse_H * coarse_W).astype(
            np.float32
        )

        # Flatten FORBID for easy masking
        # Forbidden bins in coarse space (if you want to respect FORBID also at coarse level)
        FORBID_coarse = np.zeros((coarse_H, coarse_W), dtype=bool)
        for y in range(coarse_H):
            for x in range(coarse_W):
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

        # Normalize - empirical proba for each allowed bin
        initial_dist = counts / np.maximum(1.0, counts.sum())
        initial_dist = initial_dist[allowed_bins]
        target_dist = np.ones(
            coarse_H * coarse_W,
            np.float32,
        )
        target_dist[FORBID_flat] = 0  # forbid forbidden bins
        target_dist /= target_dist.sum()  # normalize to sum=1
        target_dist = target_dist[allowed_bins]

        # compute oversampling ratios
        rep_factors = target_dist / np.maximum(initial_dist, 1e-8)
        rep_factors = np.minimum(rep_factors, 20.0)  # clip to avoid extreme repeats
        rep_factors_tf = tf.constant(rep_factors, tf.float32)

        allowed_idx = np.where(allowed_bins)[0]
        bin_to_allowed_idx = -np.ones_like(allowed_bins, dtype=int)
        bin_to_allowed_idx[allowed_idx] = np.arange(len(allowed_idx))
        # convert to tensor
        allowed_bins = tf.constant(allowed_bins)
        bin_to_allowed_idx = tf.constant(bin_to_allowed_idx)

        # Map each example to repeated dataset
        @tf.autograph.experimental.do_not_convert
        def map_repeat(ex):
            mapped_cls = map_bin_class(ex)
            allowed_idx_val = tf.gather(bin_to_allowed_idx, mapped_cls)

            safe_idx = tf.maximum(allowed_idx_val, 0)  # -1 becomes 0
            # find idx in rep_factors (only allowed bins)
            repeats = tf.cast(
                tf.math.ceil(tf.gather(rep_factors_tf, safe_idx)), tf.int64
            )
            repeats = tf.where(allowed_idx_val >= 0, repeats, 0)
            return tf.cond(
                repeats > 0,
                lambda: tf.data.Dataset.from_tensors(ex).repeat(repeats),
                lambda: tf.data.Dataset.from_tensors(ex).take(0),
            )

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
                max_samples=20000,
                path=os.path.join(
                    self.folderResult,
                    str(windowSizeMS),
                    "oversampling_effect.png",
                ),
            )
        return dataset

    def get_artificial_spikes(
        self,
        behaviorData: dict,
        windowSizeMS: int = 36,
        useSpeedFilter: bool = False,
        useTrain: bool = False,
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
                )
            except:
                print("fallback loading full/cp.ckpt")
                try:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels, str(windowSizeMS), "full", "cp.ckpt"
                        ),
                    )
                except:
                    self.model.load_weights(
                        os.path.join(
                            self.folderModels,
                            str(windowSizeMS),
                            "full",
                            "cp.weights.h5",
                        ),
                    )

        # --- Build the same total mask used in test() ---
        if useTrain:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
            ) + inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
            )
        else:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
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

        datasets = self._dataset_loading_pipeline(
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
                out_file = (
                    os.path.join(
                        self.folderResult,
                        str(windowSizeMS),
                        "artificial_waveforms.pkl",
                    ),
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
        def __init__(self, lrs):
            self.lrs = lrs

        def schedule_fixed(self, epoch, lr):
            if len(self.lrs) == 1:
                print(f"learning rate is {lr}")
                return self.lrs[0]
            elif len(self.lrs) == 2:
                if epoch < 10:
                    return self.lrs[0]
                else:
                    return self.lrs[1]
            elif len(self.lrs) == 3:
                if epoch < 10:
                    return self.lrs[0]
                elif 10 <= epoch < 50:
                    return self.lrs[1]
                else:
                    return self.lrs[2]
            else:
                raise ValueError(
                    "You cannot have more than 3 learning rate values for schedule_fixed"
                )

        def schedule_decay(self, epoch, lr):
            if epoch < 10:
                print(f"learning rate is {lr}")
                return lr
            else:
                new_lr = lr * tf.math.exp(-0.01)
                new_lr = float(new_lr.numpy())
                print(f"learning rate is {new_lr}")
                return new_lr

    def fix_linearizer(self, mazePoints, tsProj):
        ## For the linearization we define two fixed inputs:
        self.maze_points = mazePoints
        self.ts_proj = tsProj
        self.mazePoints_tensor = tf.convert_to_tensor(
            mazePoints[None, :], dtype=tf.float32
        )
        self.tsProjTensor = tf.convert_to_tensor(tsProj[None, :], dtype=tf.float32)

    # used in the data pipepline
    def create_indices(self, vals, addLinearizationTensor=False):
        """
        Create indices for gathering spikes from each group.
        The i-th spike of the group should be positioned at spikePosition[i] in the final tensor.

        Args:
            vals (dict): A dictionary containing the input tensors, including "groups" and "group{n}" for each group.
            addLinearizationTensor (bool): Whether to add linearization tensors to the output.
        Returns:
            dict: Updated dictionary with indices for each group and optional linearization tensors. The indices are stored under the keys "indices{n}" for each group and represent the positions to gather spikes from each group.

        See self.indices in the model definition for more details on usage.
        """
        for group in range(self.params.nGroups):
            spikePosition = tf.where(tf.equal(vals["groups"], group))
            # Note: inputGroups is already filled with -1 at position that correspond to filling
            # for batch issues
            # The i-th spike of the group should be positioned at spikePosition[i] in the final tensor
            # We therefore need to set indices[spikePosition[i]] to i so that it is effectively gathered
            # We need to wrap the use of sparse tensor (tensorflow error otherwise)
            # The sparse tensor allows us to get the list of indices for the gather quite easily
            rangeIndices = tf.range(tf.shape(vals["group" + str(group)])[0]) + 1
            indices = tf.sparse.SparseTensor(
                spikePosition, rangeIndices, [tf.shape(vals["groups"])[0]]
            )
            indices = tf.cast(tf.sparse.to_dense(indices), dtype=tf.int32)
            vals.update({"indices" + str(group): indices})

            if self.params.usingMixedPrecision:
                zeroForGather = tf.zeros([1, self.params.nFeatures], dtype=tf.float16)
            else:
                zeroForGather = tf.zeros([1, self.params.nFeatures])
            vals.update({"zeroForGather": zeroForGather})

            # changing the dtype to allow faster computations
            if self.params.usingMixedPrecision:
                vals.update(
                    {
                        "group" + str(group): tf.cast(
                            vals["group" + str(group)], dtype=tf.float16
                        )
                    }
                )

            if addLinearizationTensor:
                vals.update({"mazePoints": self.mazePoints_tensor})
                vals.update({"tsProj": self.tsProjTensor})

        if self.params.usingMixedPrecision:
            vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})
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
            spike_times_flat = tf.reshape(spike_times, [-1])
            temporal_bin_indices_flat = tf.reshape(temporal_bin_indices, [-1])

            # Create batch indices
            batch_indices = tf.repeat(tf.range(batch_size), max_spikes_per_batch)
        else:
            # Assume batchSize is set in params
            batch_size = self.params.batchSize
            total_spikes = original_shape[0]
            max_spikes_per_batch = total_spikes // batch_size

            original_groups_flat = original_groups
            spike_times_flat = spike_times
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
            linear_temporal_positions = (
                spikePosition[:, 0] * n_temporal_bins + spikePosition[:, 1]
            )

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

            if self.params.usingMixedPrecision:
                vals.update(
                    {
                        "group" + str(group): tf.cast(
                            vals["group" + str(group)], dtype=tf.float16
                        )
                    }
                )

        # Create zero tensor for gathering
        if self.params.usingMixedPrecision:
            zeroForGather = tf.zeros([1, self.params.nFeatures], dtype=tf.float16)
        else:
            zeroForGather = tf.zeros([1, self.params.nFeatures])

        vals.update(
            {
                "zeroForGather": zeroForGather,
                "spike_times": spike_times,  # Keep original times for reference
                "temporal_bin_size": temporal_bin_size,
                "n_temporal_bins": n_temporal_bins,
            }
        )

        if self.params.usingMixedPrecision:
            vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})

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

        self.GaussianHeatmap = GaussianHeatmapLayer(
            training_positions=full_training_true_positions,
            grid_size=grid_size,
            eps=eps,
            sigma=sigma,
            neg=neg,
            name=name,
        )
        self.GaussianLoss_layer = GaussianHeatmapLosses(
            training_positions=full_training_true_positions,
            grid_size=grid_size,
            eps=eps,
            sigma=sigma,
            neg=neg,
            name="gaussianLoss",
            l_function_layer=self.l_function_layer
            if hasattr(self, "l_function_layer")
            else None,
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
            cnn_output = self.spikeNets[group].apply(x)
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

        with tf.device(self.deviceName):
            # Create new inputs for transformer model
            cnn_feature_inputs = [
                tf.keras.Input(shape=(self.params.nFeatures,), name=f"cnn_features_{i}")
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
                    [self.params.batchSize, -1, self.params.nFeatures],
                )
                allFeatures.append(filledFeatureTrain)

            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2, name="concat_CNNs")

            # Create mask
            batchedInputGroups = tf.reshape(groups_input, [self.params.batchSize, -1])
            mymask = tf.not_equal(batchedInputGroups, -1)

            # Store raw features and apply dropout
            allFeatures_raw = allFeatures
            allFeatures = self.dropoutLayer(allFeatures)

            # Use the shared transformer logic
            myoutputPos, outputPredLoss, output, sumFeatures = (
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
            batch_size=self.params.batchSize,
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
            batch_size=self.params.batchSize,
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

    @classmethod
    def clear_session(cls):
        tf.keras.backend.clear_session()


########### END OF HELPING LSTMandSpikeNetwork FUNCTIONS#####################
