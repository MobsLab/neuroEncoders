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
import gc

import matplotlib.pyplot as plt

# Get common libraries
import dill as pickle
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import wandb

# Get utility functions
from neuroencoders.fullEncoder import nnUtils
from neuroencoders.fullEncoder.nnUtils import UMazeProjectionLayer
from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.utils.global_classes import DataHelper, Params, Project
from wandb import keras as wandbkeras

WandbMetricsLogger = wandbkeras.WandbMetricsLogger


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
        self.suffix = f"_{phase}" if phase is not None else ""
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
            assert not params.denseweight, (
                "Cannot use both GaussianHeatmap and DenseWeight"
            )
            if kwargs.get("behaviorData", None) is None:
                warnings.warn(
                    '"behaviorData" not provided, using default setup WITHOUT Gaussian Heatmap layering. Is your code version deprecated?'
                )
            else:
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
            # the exact time-steps of each spike measured in the various groups.
            # Question: should the time not be a VarLenFeature??
            "time": tf.io.FixedLenFeature([], tf.float32),
            # sample of the spike
            "indexInDat": tf.io.VarLenFeature(tf.int64),
        }
        for g in range(self.params.nGroups):
            # the voltage values (discretized over 32 time bins) of each channel (4 most of the time)
            # of each spike of a given group in the window
            self.featDesc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})

        # Loss obtained during training
        self.trainLosses = {}

    def _build_model(self, **kwargs):
        ### Description of layers here
        with tf.device(self.deviceName):
            self.inputsToSpikeNets = [
                tf.keras.layers.Input(
                    shape=(self.params.nChannelsPerGroup[group], 32),
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
            zeroForGather = tf.constant(
                tf.zeros(
                    [1, self.params.nFeatures],
                    dtype=tf.float16 if self.params.usingMixedPrecision else tf.float32,
                )
            )
            self.zeroForGather = tf.keras.layers.Input(
                tensor=zeroForGather, name="zeroForGather"
            )

            # Declare spike nets for the different groups:
            self.spikeNets = [
                nnUtils.spikeNet(
                    nChannels=self.params.nChannelsPerGroup[group],
                    device=self.deviceName,
                    nFeatures=self.params.nFeatures,
                    number=str(group),
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

                self.lstmsNets = (
                    [
                        PositionalEncoding(
                            d_model=self.params.nFeatures, device=self.deviceName
                        )
                    ]
                    + [
                        TransformerEncoderBlock(
                            d_model=self.params.nFeatures,
                            num_heads=self.params.nHeads,
                            ff_dim1=self.params.ff_dim1,
                            ff_dim2=self.params.ff_dim2,
                            dropout_rate=self.params.dropoutLSTM,
                            device=self.deviceName,
                        )
                        for _ in range(self.params.lstmLayers)
                    ]
                    + [
                        MaskedGlobalAveragePooling1D(device=self.deviceName),
                        # removed the activations in dense layers for better scaleability
                        tf.keras.layers.Dense(
                            self.params.TransformerDenseSize1,
                            kernel_regularizer="l2",
                        ),
                        tf.keras.layers.Dense(
                            self.params.TransformerDenseSize2,
                            kernel_regularizer="l2",
                        ),
                    ]
                )

            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Pierre
            self.truePos = tf.keras.layers.Input(
                shape=(self.params.dimOutput), name="pos"
            )
            self.denseLoss1 = tf.keras.layers.Dense(
                self.params.lstmSize - 0.5 * self.params.lstmSize,
                activation=tf.nn.silu,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
            self.denseLoss3 = tf.keras.layers.Dense(
                self.params.lstmSize - 0.75 * self.params.lstmSize,
                activation=tf.nn.silu,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
            self.denseLoss4 = tf.keras.layers.Dense(
                self.params.lstmSize - 0.75 * self.params.lstmSize,
                activation=tf.nn.silu,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
            self.denseLoss5 = tf.keras.layers.Dense(
                self.params.lstmSize - 0.75 * self.params.lstmSize,
                activation=tf.nn.silu,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
            self.denseLoss2 = tf.keras.layers.Dense(
                1,
                activation=kwargs.get("lossActivation", self.params.lossActivation),
                name="predicted_loss",
            )
            self.denseLossLayers = [
                self.denseLoss1,
                self.denseLoss3,
                self.denseLoss4,
                self.denseLoss5,
            ]
            self.epsilon = tf.constant(10 ** (-8))
            # Outputs
            self.denseFeatureOutput = tf.keras.layers.Dense(
                self.params.dimOutput,
                activation=tf.keras.activations.sigmoid,  # ensures output is in [0,1]
                dtype=tf.float32,
                name="feature_output",
                kernel_regularizer="l2",
            )

            self.projection_layer = tf.keras.layers.Dense(
                self.params.nFeatures,
                activation="relu",
                dtype=tf.float32,
                name="feature_projection_transformer",
            )
            self.ProjectionInMazeLayer = UMazeProjectionLayer(
                grid_size=kwargs.get("grid_size", self.params.GaussianGridSize)
            )

            # Gather the full model
            outputs = self.generate_model(**kwargs)
            # Build two models
            # One just described, with two objective functions corresponding
            # to both position and predicted losses
            self.model = self.compile_model(
                outputs, modelName="FullModel.png", **kwargs
            )
            # In theory, the predicted loss could be not learning enough in the first network (optional)
            # Second only with loss corresponding to predicted loss
            self.predLossModel = self.compile_model(
                outputs, predLossOnly=True, modelName="predLossModel.png", **kwargs
            )

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
            allFeatures: Features after dropout (batch_size, seq_len, feature_dim)
            allFeatures_raw: Raw features before dropout (for sumFeatures calculation)
            mymask: Attention mask (batch_size, seq_len)
            **kwargs: Additional arguments

        Returns:
            tuple: (myoutputPos, outputPredLoss, output, sumFeatures)
        """

        print("Using Transformer architecture !")

        # 1. Projection layer
        allFeatures = self.projection_layer(allFeatures)
        masked_features = tf.where(
            tf.expand_dims(mymask, axis=-1),
            allFeatures_raw,
            tf.zeros_like(allFeatures_raw, dtype=allFeatures_raw.dtype),
        )
        sumFeatures = tf.math.reduce_sum(self.projection_layer(masked_features), axis=1)

        # 2. Positional encoding
        allFeatures = self.lstmsNets[0](allFeatures)

        # 3. Transformer blocks with residual connections
        for ilstm, transformerLayer in enumerate(self.lstmsNets[1:-3]):
            if ilstm == 0:
                if (
                    len(self.lstmsNets) == 5
                ):  # num of transformer layers + one positional encoding + one pooling layer + 2 dense layers == 4 + ##transformer layers
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
        output = self.lstmsNets[-3](output, mask=mymask)  # pooling
        x = self.lstmsNets[-2](output)  # dense layer after pooling
        x = self.lstmsNets[-1](x)  # another dense layer after pooling

        if not getattr(self.params, "GaussianHeatmap", False):
            myoutputPos = self.denseFeatureOutput(x)
            myoutputPos = self.ProjectionInMazeLayer(myoutputPos)
        else:
            myoutputPos = self.GaussianHeatmap(x)

        # 5. Loss prediction

        output_norm = tf.nn.l2_normalize(output, axis=1)
        sumFeatures_norm = tf.nn.l2_normalize(sumFeatures, axis=1)

        output_features = tf.stop_gradient(
            tf.concat([output_norm, sumFeatures_norm], axis=1)
        )

        for i, denseLayer in enumerate(self.denseLossLayers):
            if i == self.params.nDenseLayers - 1:
                break
            else:
                output_features = denseLayer(output_features)
                output_features = self.dropoutLayer(output_features)
        # output_features is now the output of the forelast dense layer
        outputPredLoss = self.denseLoss2(output_features)

        return myoutputPos, outputPredLoss, output, sumFeatures

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

        # normalize the lstm output and the cnn features
        output_norm = tf.nn.l2_normalize(output, axis=1)
        sumFeatures_norm = tf.nn.l2_normalize(sumFeatures, axis=1)

        output_features = tf.stop_gradient(
            tf.concat([output_norm, sumFeatures_norm], axis=1)
        )

        for i, denseLayer in enumerate(self.denseLossLayers):
            if i == self.params.nDenseLayers - 1:
                break
            else:
                output_features = denseLayer(output_features)
                output_features = self.dropoutLayer(output_features)
        # output_features is now the output of the forelast dense layer
        outputPredLoss = self.denseLoss2(output_features)

        return myoutputPos, outputPredLoss, output, sumFeatures

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
            self.params.batchSize = kwargs.get("batchSize", self.params.batchSize)
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                # --> [NbKeptSpike,nbChannels,31] tensors
                x = self.spikeNets[group].apply(x)
                # outputs a [NbSpikeOfTheGroup,nFeatures=self.params.nFeatures(default 128)] tensor.
                # The gather strategy:
                #   extract the final position of the spikes
                # Note: inputGroups is already filled with -1 at position that correspond to filling
                # for batch issues
                # The i-th spike of the group should be positioned at spikePosition[i] in the final tensor
                # We therefore need to    indices[spikePosition[i]] to i  so that it is effectively gather
                # We then gather either a value of
                filledFeatureTrain = tf.gather(
                    tf.concat([self.zeroForGather, x], axis=0),
                    self.indices[group],
                    axis=0,
                )
                # At this point; filledFeatureTrain is a tensor of size (NbBatch*max(nbSpikeInBatch),self.params.nFeatures)
                # where we have filled lines corresponding to spike time of the group
                # with the feature computed by the spike net; and let other time with a value of 0:
                # The index of spike detected then become similar to a time value...
                filledFeatureTrain = tf.reshape(
                    filledFeatureTrain,
                    [self.params.batchSize, -1, self.params.nFeatures],
                )
                # Reshaping the result of the spike net as batchSize:NbTotSpikeDetected:nFeatures
                # this allow to separate spikes from the same window or from the same batch.
                allFeatures.append(filledFeatureTrain)
            allFeatures = tf.tuple(tensors=allFeatures)
            # synchronizes the computation of all features (like a join)
            # The concatenation is made over axis 2, which is the Feature axis
            # So we reserve columns to each output of the spiking networks...
            allFeatures = tf.concat(
                allFeatures, axis=2, name="concat_CNNs"
            )  # , name="concat1"
            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(
                self.inputGroups, [self.params.batchSize, -1]
            )
            mymask = tf.not_equal(batchedInputGroups, -1)

            masked_features = tf.where(
                tf.expand_dims(mymask, axis=-1),
                allFeatures,
                tf.zeros_like(allFeatures, dtype=allFeatures.dtype),
            )
            sumFeatures = tf.math.reduce_sum(
                masked_features, axis=1
            )  # This var will be used in the predLoss loss

            allFeatures_raw = allFeatures
            allFeatures = self.dropoutLayer(allFeatures)

            # LSTM
            if not kwargs.get("isTransformer", False):
                myoutputPos, outputPredLoss, output, sumFeatures = (
                    self.apply_lstm_architecture(
                        allFeatures, sumFeatures, mymask, **kwargs
                    )
                )
            else:
                # Use shared transformer logic
                myoutputPos, outputPredLoss, output, sumFeatures = (
                    self.apply_transformer_architecture(
                        allFeatures, allFeatures_raw, mymask, **kwargs
                    )
                )

            # TODO: change
            ### Multi-column loss configuration
            column_losses = getattr(
                self.params,
                "column_losses",
                {str(i): self.params.loss for i in range(myoutputPos.shape[-1])},
            )
            column_weights = getattr(self.params, "column_weights", {})

            # Create the loss instance
            if len(column_weights) > 1 or any(
                "," in spec for spec in column_losses.keys()
            ):
                loss_function = MultiColumnLossLayer(
                    column_losses=column_losses,
                    column_weights=column_weights,
                    alpha=self.params.alpha,
                )  # actually it's more of a layer than a function

            else:
                # Single loss case
                loss_function = _get_loss_function(
                    self.params.loss, alpha=self.params.alpha
                )

            #### note
            # bayesian loss function  = sum ((y_true - y_pred)^2 / sigma^2 + log(sigma^2))

            # we assume myoutputPos is in cm x cm,
            # as self.truePos (modulo [0,1] normalization)
            # in ~cm2 as no loss is sqrt or log
            if getattr(self.params, "GaussianHeatmap", False):
                targets_hw = self.GaussianHeatmap.gaussian_heatmap_targets(self.truePos)
                tempPosLoss = self.GaussianHeatmap.safe_kl_heatmap_loss(
                    myoutputPos, targets_hw
                )

            else:
                tempPosLoss = loss_function(myoutputPos, self.truePos)[:, tf.newaxis]
            # for main loss functions:
            # if loss function is mse
            # tempPosLoss is in cm2
            # if loss function is logcosh
            # tempPosLoss is in cm2

            if self.params.denseweight:
                tempPosLoss = self.apply_dynamic_dense_loss(tempPosLoss, self.truePos)

            if self.params.transform_w_log:
                posLoss = tf.identity(
                    tf.math.log(tf.math.reduce_mean(tempPosLoss)),
                    name="posLoss",  # log cm2 or ~ 2 * log cm
                )
            elif not getattr(self.params, "GaussianHeatmap", False):
                posLoss = tf.identity(
                    tf.math.reduce_mean(tf.math.sqrt(tempPosLoss)),
                    name="posLoss",  # cm
                )
            else:
                posLoss = tf.identity(
                    tf.math.reduce_mean(tempPosLoss),
                    name="posLoss",  # unitless
                )

            # remark: we need to also stop the gradient to propagate from posLoss to the network at the stage of
            # the computations for the loss of the loss predictor
            # still ~ in cm2
            # # outputPredLoss is supposed to be in cm2 and predict the MSE loss.
            # preUncertaintyLoss is in cm2^2 as it's the MSE between the predicted loss and the posLoss
            if self.params.transform_w_log:
                logPosLoss = tf.math.log(
                    tf.add(tempPosLoss, self.epsilon)
                )  # log cm2 or ~ 2 * log cm
                preUncertaintyLoss = tf.losses.mean_squared_error(
                    outputPredLoss, tf.stop_gradient(logPosLoss)
                )  # in (log cm2)^2 or ~ 4 (log cm)^2
                uncertaintyLoss = tf.identity(
                    tf.math.log(
                        tf.add(tf.math.reduce_mean(preUncertaintyLoss), self.epsilon)
                    ),
                    name="uncertaintyLoss",
                )  # log (log(cm2)^2) or ~ log(4) + log 2(log cm))
            elif not getattr(self.params, "GaussianHeatmap", False):
                preUncertaintyLoss = tf.math.sqrt(
                    tf.math.sqrt(
                        tf.losses.mean_squared_error(
                            outputPredLoss, tf.stop_gradient(tempPosLoss)
                        )  # in cm2^2 (MSE between predicted loss and posLoss)
                    )  # now in cm2
                )  # now in cm

                # back to cm to compute the uncertainty loss as the MSE between the predicted loss and the posLoss
                uncertaintyLoss = tf.identity(
                    tf.math.reduce_mean(preUncertaintyLoss, name="uncertaintyLoss")
                )
            else:
                # TODO: temperature scaling of the loss predictor for the GaussianHeatmap case
                preUncertaintyLoss = tf.losses.mean_squared_error(
                    outputPredLoss, tf.stop_gradient(tempPosLoss)
                )
                uncertaintyLoss = tf.identity(
                    tf.math.reduce_mean(preUncertaintyLoss, name="uncertaintyLoss")
                )

        return myoutputPos, outputPredLoss, posLoss, uncertaintyLoss

    def compile_model(
        self, outputs, modelName="FullModel.png", predLossOnly=False, **kwargs
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
        tf.keras.utils.plot_model(
            model,
            to_file=(os.path.join(self.projectPath.experimentPath, modelName)),
            show_shapes=True,
        )

        @keras.saving.register_keras_serializable()
        def pos_loss(x, y):
            return y

        @keras.saving.register_keras_serializable()
        def uncertainty_loss(x, y):
            return y

        # Compile the model
        # TODO: use params.optimizer instead of hardcoding RMSprop
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=kwargs.get("lr", self.params.learningRates[0])
        )
        if not predLossOnly:
            # Full model
            model.compile(
                # TODO: Adam or AdaGrad?
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
                optimizer=self.optimizer,
                loss={
                    # tf_op_layer_ position loss (eucledian distance between predicted and real coordinates)
                    outputs[2].name.split("/Identity")[0]: pos_loss,
                    # # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss)
                    outputs[3].name.split("/Identity")[0]: uncertainty_loss,
                },
            )
            # Get internal names of losses
            self.outNames = [
                outputs[2].name.split("/Identity")[0],
                outputs[3].name.split("/Identity")[0],
            ]
        else:
            # set all non trainable layers
            # first the spike nets
            for group in range(self.params.nGroups):
                for layer in self.spikeNets[group].layers():
                    if hasattr(layer, "trainable"):
                        layer.trainable = False
            # then the lstm layers
            if hasattr(self, "projection_layer") and hasattr(
                self.projection_layer, "trainable"
            ):
                self.projection_layer.trainable = False
            for lstmLayer in self.lstmsNets:
                if hasattr(lstmLayer, "trainable"):
                    lstmLayer.trainable = False

            # finally, the densefeatureoutput
            if hasattr(self.denseFeatureOutput, "trainable"):
                self.denseFeatureOutput.trainable = False

            # Only used to create the self.predLossModel
            model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]),
                optimizer=self.optimizer,
                loss={
                    outputs[3].name.split("/Identity")[0]: uncertainty_loss
                },  # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss) },
            )
            self.outlossPredNames = [outputs[3].name.split("/Identity")[0]]
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
            output = tf.ensure_shape(
                output, [self.params.batchSize, self.params.lstmSize]
            )
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

        from neuroencoders.fullEncoder.nnUtils import (
            NeuralDataAugmentation,
            create_flatten_augmented_groups_fn,
        )

        ### Create neccessary arrays
        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)
        windowSizeMS = kwargs.get("windowSizeMS", 36)
        scheduler = kwargs.get("scheduler", "decay")
        isPredLoss = kwargs.get("isPredLoss", True)
        earlyStop = kwargs.get("earlyStop", False)

        load_model = kwargs.get("load_model", False)
        if not isinstance(windowSizeMS, int):
            windowSizeMS = int(windowSizeMS)

        if self.debug:
            tf.profiler.experimental.start(logdir=self.folderResult)
        epochMask = {}
        totMask = {}
        csvLogger = {}
        checkpointPath = {}
        # Manage folders
        if not os.path.isdir(os.path.join(self.folderModels, str(windowSizeMS))):
            os.makedirs(os.path.join(self.folderModels, str(windowSizeMS)))

        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowSizeMS), "full")
        ):
            os.makedirs(os.path.join(self.folderModels, str(windowSizeMS), "full"))
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0:
            if not os.path.isdir(
                os.path.join(self.folderModels, str(windowSizeMS), "predLoss")
            ):
                os.makedirs(
                    os.path.join(self.folderModels, str(windowSizeMS), "predLoss")
                )
        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowSizeMS), "savedModels")
        ):
            os.makedirs(
                os.path.join(self.folderModels, str(windowSizeMS), "savedModels")
            )
        # Manage callbacks
        csvLogger["full"] = tf.keras.callbacks.CSVLogger(
            os.path.join(self.folderModels, str(windowSizeMS), "full", "fullmodel.log")
        )
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            csvLogger["predLoss"] = tf.keras.callbacks.CSVLogger(
                os.path.join(
                    self.folderModels,
                    str(windowSizeMS),
                    "predLoss",
                    "predLossmodel.log",
                )
            )
        for key in csvLogger.keys():
            checkpointPath[key] = os.path.join(
                self.folderModels, str(windowSizeMS), key + "/cp.ckpt"
            )

        ## Get speed filter:
        speedMask = behaviorData["Times"]["speedFilter"]

        ## Get datasets
        ndataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )
        )

        def _parse_function(*vals):
            return nnUtils.parse_serialized_spike(self.featDesc, *vals)

        ndataset = ndataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ndataset = ndataset.prefetch(tf.data.AUTOTUNE)
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

        # Create datasets
        datasets = {}

        def filter_by_pos_index(x):
            return tf.equal(table.lookup(x["pos_index"]), 1.0)

        def filter_nan_pos(x):
            pos_data = x["pos"]
            # convert to float if it's a binary pred
            if pos_data.dtype in [tf.int32, tf.int64]:
                pos_data = tf.cast(pos_data, tf.float64)

            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        @tf.autograph.experimental.do_not_convert
        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

        @keras.saving.register_keras_serializable()
        def pos_loss(x, y):
            return y

        @keras.saving.register_keras_serializable()
        def uncertainty_loss(x, y):
            return y

        augmentation_config = NeuralDataAugmentation(device=self.deviceName, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def map_parse_serialized_sequence_with_augmentation(*vals):
            return nnUtils.parse_serialized_sequence_with_augmentation(
                self.params,
                *vals,
                augmentation_config=augmentation_config,
                batched=True,
            )

        flatten_fn = create_flatten_augmented_groups_fn(
            self.params, augmentation_config.num_augmentations
        )

        @tf.autograph.experimental.do_not_convert
        def map_outputs(vals):
            return (
                vals,
                {
                    self.outNames[0]: tf.zeros(self.params.batchSize),
                    self.outNames[1]: tf.zeros(self.params.batchSize),
                },
            )

        for key in totMask.keys():
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(totMask[key])), dtype=tf.int64),
                    tf.constant(totMask[key], dtype=tf.float64),
                ),
                default_value=0,
            )
            datasets[key] = ndataset.filter(filter_by_pos_index)
            # This is just max normalization to use if the behavioral data have not been normalized yet
            # TODO: make a way to input 1D target, different 2D targets...
            # coherent with the --target arg from main
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
            datasets[key] = datasets[key].map(nnUtils.import_true_pos(posFeature))
            datasets[key] = datasets[key].filter(filter_nan_pos)

            # now that we have clean positions, we can resample if needed
            if self.params.OversamplingResampling and key == "train":
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
                datasets[key] = datasets[key].filter(
                    lambda ex: tf.greater_equal(map_bin_class(ex), 0)
                )

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
                rep_factors = np.minimum(
                    rep_factors, 20.0
                )  # clip to avoid extreme repeats
                rep_factors_tf = tf.constant(rep_factors, tf.float32)

                allowed_idx = np.where(allowed_bins)[0]
                bin_to_allowed_idx = -np.ones_like(allowed_bins, dtype=int)
                bin_to_allowed_idx[allowed_idx] = np.arange(len(allowed_idx))
                # convert to tensor
                allowed_bins = tf.constant(allowed_bins)
                bin_to_allowed_idx = tf.constant(bin_to_allowed_idx)

                # Map each example to repeated datasets
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

                datasets_before_oversampling = datasets[
                    key
                ]  # Save this before the oversampling block
                datasets[key] = datasets[key].flat_map(map_repeat)
                # shuffle after repeating to mix repeated samples
                datasets[key] = datasets[key].shuffle(buffer_size=10000, seed=42)
                datasets_after_oversampling = datasets[
                    key
                ]  # Save this before the oversampling block
                from neuroencoders.importData.gui_elements import OversamplingVisualizer

                if not os.path.exists(
                    os.path.join(
                        self.folderResult, str(windowSizeMS), "oversampling_effect.png"
                    )
                ):
                    visualizer = OversamplingVisualizer(self.GaussianHeatmap)
                    visualizer.visualize_oversampling_effect(
                        datasets_before_oversampling,
                        datasets_after_oversampling,
                        stride=stride,  # Match your stride
                        max_samples=20000,
                        path=os.path.join(
                            self.folderResult,
                            str(windowSizeMS),
                            "oversampling_effect.png",
                        ),
                    )

            datasets[key] = (
                datasets[key].batch(self.params.batchSize, drop_remainder=True).cache()
            )

            if not self.params.dataAugmentation or key == "test":
                print("No data augmentation for", key, "dataset")
                datasets[key] = datasets[key].map(
                    map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
                )  # self.featDesc, *
            else:
                print("Applying data augmentation to", key, "dataset")
                datasets[key] = datasets[key].map(
                    map_parse_serialized_sequence_with_augmentation,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )  # self.featDesc, *
                datasets[key] = datasets[key].flat_map(
                    flatten_fn
                )  # Flatten the augmented groups

            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            datasets[key] = datasets[key].map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = (
                datasets[key]
                .map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
            )
            # We shuffle the datasets and cache it - this way the training samples are randomized for each epoch
            # and each mini-batch contains a representative sample of the training set.
            # nSteps represent the buffer size of the shuffle operation - 10 seconds worth of buffer starting
            # from the 0-timepoint of the dataset.
            # once an element is selected, its space in the buffer is replaced by the next element (right after the 10s window...)
            # At each epoch, the shuffle order is different.
            datasets[key] = datasets[key].shuffle(
                buffer_size=10000,
                reshuffle_each_iteration=True,
            )  # were talking in number of batches here, not time (so it does not make sense to use params.nSteps)
            datasets[key] = datasets[key].prefetch(
                tf.data.AUTOTUNE
            )  # prefetch entire batches

        # memory garbage collection class class
        class MemoryUsageCallbackExtended(tf.keras.callbacks.Callback):
            """Monitor memory usage during training, collect garbage."""

            def on_epoch_begin(self, epoch, logs=None):
                print("**Epoch {}**".format(epoch))
                print(
                    "Memory usage on epoch begin: {}".format(
                        psutil.Process(os.getpid()).memory_info().rss
                    )
                )

            def on_epoch_end(self, epoch, logs=None):
                print(
                    "Memory usage on epoch end:   {}".format(
                        psutil.Process(os.getpid()).memory_info().rss
                    )
                )
                gc.collect()
                tf.keras.backend.clear_session()

        ### Train the model(s)
        # Initialize the model for C++ decoder
        # self.generate_model_Cplusplus()
        # Train
        for key in checkpointPath.keys():
            print("Training the", key, "model")
            nb_epochs_already_trained = 10
            loaded = False
            if load_model and os.path.exists(os.path.dirname(checkpointPath[key])):
                if key != "predLoss":
                    print(
                        "Loading the weights of the loss training model from",
                        checkpointPath[key],
                    )
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
                        print("nb_epochs_already_trained =", nb_epochs_already_trained)
                        loaded = True
                    except Exception as e:
                        print(
                            "Error loading weights for",
                            key,
                            "from",
                            checkpointPath[key],
                            ":",
                            e,
                        )

            if loaded and nb_epochs_already_trained >= self.params.nEpochs / 2:
                if not kwargs.get("fine_tune", False):
                    print(f"Model loaded for {key}, skipping directly to next.")
                    continue

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpointPath[key], save_weights_only=True, verbose=1
            )
            # Manage learning rates schedule
            if loaded and kwargs.get("fine_tune", False):
                print("Fine-tuning the model with a lower learning rate, set to 0.0001")
                self.optimizer.learning_rate.assign(0.0001)
                self.model.optimizer.learning_rate.assign(0.0001)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.0001)
                # according to some sources we need to recompile the model after changing the learning rate
                self.model.compile(
                    optimizer=self.optimizer,
                    loss={
                        self.outNames[0]: pos_loss,
                        self.outNames[1]: uncertainty_loss,
                    },
                )
            elif loaded:
                print("Loading the model with the initial learning rate")
                self.optimizer.learning_rate.assign(self.params.learningRates[0])
                self.model.optimizer.learning_rate.assign(self.params.learningRates[0])
                tf.keras.backend.set_value(
                    self.model.optimizer.learning_rate, self.params.learningRates[0]
                )
                self.model.compile(
                    optimizer=self.optimizer,
                    loss={
                        self.outNames[0]: pos_loss,
                        self.outNames[1]: uncertainty_loss,
                    },
                )
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
                print("Debugging mode is ON, enabling TensorBoard callback")
                is_tbcallback = True
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
                    wandb.init(
                        entity="touseul",
                        project="projected rien de rien",
                        name=f"{prefix}{os.path.basename(os.path.dirname(self.projectPath.xml))}_{os.path.basename(self.projectPath.experimentPath)}_{key}_{windowSizeMS}ms",
                        notes=f"{os.path.basename(self.projectPath.experimentPath)}_{key}",
                        # sync_tensorboard=True,
                        config=ann_config,
                    )

                    wandb_callback = WandbMetricsLogger()
            if key == "predLoss":
                # if we need to train the loss predictor model on its own
                self.predLossModel.load_weights(
                    checkpointPath["full"]
                )  # Load weights from full network if we train
                if earlyStop:
                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_tf.identity_1_loss",
                        patience=10,
                        min_delta=0.01,
                        restore_best_weights=True,
                    )
                    callbacks = [csvLogger[key], cp_callback, schedule, es_callback]
                else:
                    callbacks = [csvLogger[key], cp_callback, schedule]

                # if is_tbcallback:
                #     callbacks.append(tb_callbacks)

                hist = self.predLossModel.fit(
                    datasets["predLoss"],
                    epochs=self.params.nEpochs - nb_epochs_already_trained + 10,
                    callbacks=callbacks,
                    validation_data=datasets["test"],
                )
                self.trainLosses[key] = np.transpose(
                    np.stack([hist.history["loss"]])
                )  # tf_op_layer_lossOfLossPredictor_loss
                valLosses = np.transpose(
                    hist.history["val_" + self.outNames[1] + "_loss"]
                )
                self.losses_fig(
                    self.trainLosses[key],
                    os.path.join(self.folderModels, str(windowSizeMS)),
                    fullModel=False,
                    valLosses=valLosses,
                )
                self.predLossModel.save_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "predLoss",
                    ),
                )
                # Save model for C++ decoder
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels,
                #                     str(windowSizeMS), "savedModels","predLossModel"))
                # self.predLossModel.save(
                #     os.path.join(
                #         self.folderModels,
                #         str(windowSizeMS),
                #         "savedModels",
                #         "predLossModel.keras",
                #     )
                # )
            else:
                if earlyStop:
                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        min_delta=0.05,
                        verbose=1,
                        restore_best_weights=True,
                        start_from_epoch=50 - nb_epochs_already_trained,
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
                        monitor="val_tf.identity_1_loss",
                        factor=0.8,
                        patience=10,
                        verbose=1,
                        start_from_epoch=40 - nb_epochs_already_trained,
                    )
                    callbacks.append(reduce_lr_callback)

                if is_tbcallback:
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
                            hist.history[
                                self.outNames[0] + "_loss"
                            ],  # tf_op_layer_lossOfManifold
                            hist.history[self.outNames[1] + "_loss"],
                        ]
                    )
                )  # tf_op_layer_lossOfLossPredictor_loss
                valLosses = np.transpose(
                    np.stack(
                        [
                            hist.history[
                                "val_" + self.outNames[0] + "_loss"
                            ],  # tf_op_layer_lossOfManifold
                            hist.history["val_" + self.outNames[1] + "_loss"],
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
                        self.folderModels, str(windowSizeMS), "savedModels", "full"
                    ),
                )
                # Save model for C++ decoder
                # self.cplusplusModel.predict(datasets['train'])
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels, str(windowSizeMS), "savedModels","fullModel"))
                # self.model.save(
                #     os.path.join(
                #         self.folderModels,
                #         str(windowSizeMS),
                #         "savedModels",
                #         "fullModel.keras",
                #     )
                # )
                if self.debug:
                    # wandb.tensorboard.unpatch()
                    wandb.finish()

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
        # l_function=[],
        # windowSizeMS=36,
        # useSpeedFilter=False,
        # useTrain=False,
        # onTheFlyCorrection=False,
        # isPredLoss=False,
        # speedValue=None,
        # phase=None,
        # template=None,

        # Unpack kwargs
        l_function = kwargs.get("l_function", [])
        windowSizeMS = kwargs.get("windowSizeMS", 36)
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

        # TODO: change speed filter with custom speed
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowSizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowSizeMS)))
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
                        self.folderModels, str(windowSizeMS), "savedModels", "full"
                    ),
                )
            except:
                print("loading from savedModels failed, trying full checkpoint ")
                self.model.load_weights(
                    os.path.join(
                        self.folderModels, str(windowSizeMS), "full" + "/cp.ckpt"
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
        dataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )
        )

        def _parse_function(*vals):
            return nnUtils.parse_serialized_spike(self.featDesc, *vals)

        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(np.arange(len(totMask)), dtype=tf.int64),
                tf.constant(totMask, dtype=tf.float64),
            ),
            default_value=0,
        )

        def filter_by_pos_index(x):
            # check previous commits for this line
            return tf.equal(table.lookup(x["pos_index"]), 1.0)

        def filter_nan_pos(x):
            pos_data = x["pos"]
            # convert to float if it's a binary pred
            if pos_data.dtype in [tf.int32, tf.int64]:
                pos_data = tf.cast(pos_data, tf.float64)
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        @tf.autograph.experimental.do_not_convert
        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

        @tf.autograph.experimental.do_not_convert
        def map_outputs(vals):
            return (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_uncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            )

        dataset = dataset.filter(filter_by_pos_index)
        if onTheFlyCorrection:
            maxPos = np.nanmax(
                behaviorData["Positions"][
                    np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))
                ]
            )
            posFeature = behaviorData["Positions"] / maxPos
        else:
            posFeature = behaviorData["Positions"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(filter_nan_pos)
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE).cache()

        print("INFERRING")
        outputTest = self.model.predict(dataset, verbose=1)

        ### Post-inferring management
        print("gathering true feature")

        @tf.autograph.experimental.do_not_convert
        def map_true_feature(x, y):
            return x["pos"]

        datasetPos = dataset.map(map_true_feature, num_parallel_calls=tf.data.AUTOTUNE)
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)

        if self.target.lower() == "direction":
            outputTest = (tf.cast(outputTest[0] > 0.5, tf.int32),)

        if getattr(self.params, "GaussianHeatmap", False):
            output_logits = outputTest[0]  # logits with shape (batch, H, W)

            xy, maxp, Hn, var_total = self.GaussianHeatmap.decode_and_uncertainty(
                output_logits
            )
            # small hack to have the right shape for featureTrue (batch, 2)
            featureTrue = np.reshape(fullFeatureTrue, [xy.shape[0], xy.shape[-1]])
            if fit_temperature:
                val_targets = self.GaussianHeatmap.gaussian_heatmap_targets(featureTrue)
                T_scaling = self.GaussianHeatmap.fit_temperature(
                    output_logits, val_targets, iters=400
                )
                return T_scaling

            if T_scaling is not None:
                output_logits = output_logits / T_scaling

            xy, maxp, Hn, var_total = self.GaussianHeatmap.decode_and_uncertainty(
                output_logits
            )
            # reconstruct outputTest tuple with xy pred instead of heatmap
            outputTest = (xy.numpy(), outputTest[1], outputTest[2], outputTest[3])

        featureTrue = np.reshape(
            fullFeatureTrue, [outputTest[0].shape[0], outputTest[0].shape[-1]]
        )

        print("gathering times of the centre in the time window")

        @tf.autograph.experimental.do_not_convert
        def map_time(x, y):
            return x["time"]

        datasetTimes = dataset.map(map_time, num_parallel_calls=tf.data.AUTOTUNE)
        times = list(datasetTimes.as_numpy_iterator())
        times = np.reshape(times, [outputTest[0].shape[0]])
        print("gathering indices of spikes relative to coordinates")

        @tf.autograph.experimental.do_not_convert
        def map_pos_index(x, y):
            return x["pos_index"]

        @tf.autograph.experimental.do_not_convert
        def map_index_in_dat(x, y):
            return x["indexInDat"]

        datasetPos_index = dataset.map(
            map_pos_index, num_parallel_calls=tf.data.AUTOTUNE
        )
        datasetIndexInDat = dataset.map(
            map_index_in_dat, num_parallel_calls=tf.data.AUTOTUNE
        )
        IDdat = list(datasetIndexInDat.as_numpy_iterator())
        posIndex = list(datasetPos_index.as_numpy_iterator())
        posIndex = np.ravel(np.array(posIndex))
        print("gathering speed mask")
        windowmaskSpeed = speedMask[
            posIndex
        ]  # the speedMask used in the table lookup call
        posLoss = (
            outputTest[2].numpy() if hasattr(outputTest[2], "numpy") else outputTest[2]
        )
        uncertaintyLoss = (
            outputTest[3].numpy() if hasattr(outputTest[3], "numpy") else outputTest[3]
        )

        testOutput = {
            "featurePred": outputTest[0].numpy()
            if hasattr(outputTest[0], "numpy")
            else outputTest[0],
            "featureTrue": featureTrue,
            "times": times,
            "predLoss": outputTest[1].numpy()
            if hasattr(outputTest[1], "numpy")
            else outputTest[1],
            "posLoss": posLoss,
            "posIndex": posIndex,
            "speedMask": windowmaskSpeed,
            "uncertaintyLoss": uncertaintyLoss,
            "indexInDat": IDdat,
        }

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
        # l_function = ([],)
        # windowSizeDecoder = (None,)
        # windowSizeMS = (36,)
        # isPredLoss = (False,)
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
        # Unpack kwargs
        l_function = kwargs.get("l_function", [])
        windowSizeDecoder = kwargs.get("windowSizeDecoder", None)
        windowSizeMS = kwargs.get("windowSizeMS", 36)
        isPredLoss = kwargs.get("isPredLoss", False)
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
            try:
                self.model.load_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeDecoder),
                        "savedModels",
                        "predLoss",
                    )
                )
            except:
                print(
                    "loading from predLoss savedModels failed, trying checkpoint for sleep epochs"
                )
                self.model.load_weights(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeDecoder),
                        "predLoss" + "/cp.ckpt",
                    )
                )
        else:
            try:
                self.model.load_weights(
                    os.path.join(
                        self.folderModels, str(windowSizeDecoder), "savedModels", "full"
                    )
                )
            except:
                print(
                    "loading from full savedModels failed, trying checkpoint for sleep epochs"
                )
                self.model.load_weights(
                    os.path.join(
                        self.folderModels, str(windowSizeDecoder), "full" + "/cp.ckpt"
                    )
                )

        print("decoding sleep epochs")
        predictions = {}
        for idsleep, sleepName in enumerate(behaviorData["Times"]["sleepNames"]):
            timeSleepStart = behaviorData["Times"]["sleepEpochs"][2 * idsleep][0]
            timeSleepStop = behaviorData["Times"]["sleepEpochs"][2 * idsleep + 1][0]

            # Get the dataset
            dataset = tf.data.TFRecordDataset(
                os.path.join(
                    self.projectPath.dataPath,
                    ("datasetSleep" + "_stride" + str(windowSizeMS) + ".tfrec"),
                )
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

            @tf.autograph.experimental.do_not_convert
            def map_outputs(vals):
                return (
                    vals,
                    {
                        "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                        "tf_op_layer_uncertaintyLoss": tf.zeros(self.params.batchSize),
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
                output_logits = output[0]  # logits with shape (batch, H, W)
                xy, maxp, Hn, var_total = self.GaussianHeatmap.decode_and_uncertainty(
                    output_logits
                )
                if T_scaling is not None:
                    output_logits = output_logits / T_scaling

                output = (xy.numpy(), output[1], output[2], output[3])

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
                "predLoss": output[1],
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

    def get_artificial_spikes(
        self,
        behaviorData,
        windowSizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowSizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowSizeMS)))
        # Loading the weights
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

        # Manage the behavior
        if useSpeedFilter:
            speedMask = behaviorData["Times"]["speedFilter"]
        else:
            speedMask = np.ones_like(behaviorData["Times"]["speedFilter"], dtype=bool)
        if useTrain:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
            )
        else:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
            )
        totMask = speedMask * epochMask

        # Load the and imfer dataset
        dataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )
        )

        def _parse_function(*vals):
            return nnUtils.parse_serialized_spike(self.featDesc, *vals)

        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(np.arange(len(totMask)), dtype=tf.int64),
                tf.constant(totMask, dtype=tf.float64),
            ),
            default_value=0,
        )

        def filter_by_pos_index(x):
            return tf.math.greater(table.lookup(x["pos_index"]), 0)

        def import_true_pos(x):
            return nnUtils.import_true_pos(posFeature)(x)

        def filter_nan_pos(x):
            pos_data = x["pos"]
            # convert to float if it's a binary pred
            if pos_data.dtype in [tf.int32, tf.int64]:
                pos_data = tf.cast(pos_data, tf.float64)

            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

        def map_outputs(vals):
            return (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_uncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            )

        dataset = dataset.filter(filter_by_pos_index)
        posFeature = behaviorData["Positions"]
        dataset = dataset.map(import_true_pos)
        dataset = dataset.filter(filter_nan_pos)
        # dataset = dataset.batch(
        #     self.params.batchSize, drop_remainder=True
        # )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)

        def spike_detaching_factory(groupId):
            detach_function = tf.keras.backend.function(
                [self.model.input], [self.model.get_layer(f"outputCNN{groupId}").output]
            )
            return detach_function

        # TODO: finish this function. You'll get a matrix of artificial spikes
        #  then, you'll need to slice it by windows to get pop spikes and convert
        #  it to bayesian spike trains
        # TODO: to decode sorted spikes with LSTM, you'll need to create a new model
        #  without CNN. Convert spikes to sparse tensor like popSpikes (nbSpikes*nbClusters)
        #  here, reshape it to (128*nbSpikes*nbClusters) this would be the input of the LSTM.
        print("Getting artificial spikes")
        aSpikes = []
        detachingFunctions = []
        zeroForGather = np.zeros((1, 64))
        for i, batch in enumerate(tqdm(dataset)):
            for igroup in range(self.params.nGroups):
                detachingFunctions.append(spike_detaching_factory(igroup))
                spikes = detachingFunctions[igroup](batch)[0]
                spikesWithZero = np.concatenate([zeroForGather, spikes], axis=0)
                popSpikes = np.take(
                    spikesWithZero, batch[0][f"indices{igroup}"], axis=0
                )
                aSpikes.append(popSpikes)
        aSpikes = np.concatenate(aSpikes, axis=0)

        df = pd.DataFrame(aSpikes)
        df.to_csv(os.path.join(self.folderResult, str(windowSizeMS), "aspikes.csv"))

        return aSpikes

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
                print(f"learning rate is {new_lr}")
                return new_lr

    def fix_linearizer(self, mazePoints, tsProj):
        ## For the linearization we define two fixed inputs:
        self.maze_points = mazePoints
        self.ts_proj = tsProj
        self.mazePoints_tensor = tf.convert_to_tensor(
            mazePoints[None, :], dtype=tf.float32
        )
        self.mazePoints = tf.keras.layers.Input(
            tensor=self.mazePoints_tensor, name="mazePoints"
        )
        self.tsProjTensor = tf.convert_to_tensor(tsProj[None, :], dtype=tf.float32)
        self.tsProj = tf.keras.layers.Input(tensor=self.tsProjTensor, name="tsProj")

    # used in the data pipepline
    def create_indices(self, vals, addLinearizationTensor=False):
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

    def losses_fig(self, trainLosses, folderModels, fullModel=True, valLosses=[]):
        if fullModel:
            # Save the data
            df = pd.DataFrame(trainLosses)
            df.to_csv(os.path.join(folderModels, "full", "fullModelLosses.csv"))
            # Plot the figure'
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(trainLosses[:, 0], label="train losses")
            ax[0].set_title("position loss")
            ax[0].plot(valLosses[:, 0], label="validation position loss", c="orange")
            ax[1].plot(trainLosses[:, 1], label="train loss prediction loss")
            ax[1].set_title("loss predictor loss")
            ax[1].plot(valLosses[:, 1], label="validation loss prediction loss")
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
        # Predicted loss
        df = pd.DataFrame(test_output["predLoss"])
        df.to_csv(os.path.join(folderToSave, f"lossPred{suffix}.csv"))
        # True coordinates
        if not sleep:
            df = pd.DataFrame(test_output["featureTrue"])
            df.to_csv(os.path.join(folderToSave, f"featureTrue{suffix}.csv"))
            # Position loss
            df = pd.DataFrame(test_output["posLoss"])
            df.to_csv(os.path.join(folderToSave, f"posLoss{suffix}.csv"))
            # Uncertainty loss
            df = pd.DataFrame(test_output["uncertaintyLoss"])
            df.to_csv(os.path.join(folderToSave, f"uncertaintyLoss{suffix}.csv"))
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
            temp_pos_loss = tf.math.multiply(temp_pos_loss, weightings[:, tf.newaxis])

            if self.dynamicdense_verbose:
                print("✓ Applied Dynamic Dense Loss reweighting")

        return temp_pos_loss

    def setup_gaussian_heatmap(self, **kwargs):
        from neuroencoders.fullEncoder.nnUtils import GaussianHeatmapLayer

        # Unpack kwargs
        behaviorData = kwargs.get("behaviorData", None)
        if behaviorData is None:
            raise ValueError(
                "You must provide behaviorData to setup Gaussian Heatmap Layer."
            )
        grid_size = kwargs.get("grid_size", self.params.GaussianGridSize)
        eps = kwargs.get("eps", self.params.GaussianEps)
        sigma = kwargs.get("sigma", self.params.GaussianSigma)
        neg = kwargs.get("neg", self.params.GaussianNeg)
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


def _get_loss_function(loss_name: str, alpha: float) -> tf.keras.losses.Loss:
    """Helper function to get loss function by name with reduction='none'"""
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError(reduction="none")
    elif loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError(reduction="none")
    elif loss_name == "huber":
        return tf.keras.losses.Huber(delta=alpha, reduction="none")
    elif loss_name == "msle":
        return tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")
    elif loss_name == "logcosh":
        return tf.keras.losses.LogCosh(reduction="none")
    elif loss_name == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy(reduction="none")
    elif loss_name == "categorical_crossentropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    elif loss_name == "mse_plus_msle":

        def combined_loss_mse(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError(reduction="none")(y_true, y_pred)
            msle = tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")(
                y_true, y_pred
            )
            return mse + alpha * msle

        return combined_loss_mse
    elif loss_name == "cyclic_mae":

        def cyclical_mae_rad(y_true, y_pred):
            return tf.keras.backend.minimum(
                tf.keras.backend.abs(y_pred - y_true),
                tf.keras.backend.minimum(
                    tf.keras.backend.abs(y_pred - y_true + 2 * np.pi),
                    tf.keras.backend.abs(y_pred - y_true - 2 * np.pi),
                ),
            )

        return cyclical_mae_rad
    else:
        raise ValueError(f"Loss function {loss_name} not recognized")


########### END OF HELPING LSTMandSpikeNetwork FUNCTIONS#####################


class MultiColumnLossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        column_losses=None,
        column_weights=None,
        alpha=1.0,
        name="multi_output_loss_layer",
        **kwargs,
    ):
        """
        Args:
            column_losses (dict): Dictionary mapping column specifications to loss names.
                Example: {"0": "mse", "1,2": "huber"} means column 0 uses MSE and columns 1 and 2 use Huber loss.
            column_weights (dict): Dictionary mapping column specifications to weights.
                Example: {"0": 1.0, "1,2": 0.5} means column 0 has weight 1.0 and columns 1 and 2 have weight 0.5.
            alpha (float): Hyperparameter for losses like Huber or MSLE.
            name (str): Name of the layer.
            **kwargs: Additional keyword arguments for the Layer constructor.

        """
        super().__init__(name=name, **kwargs)
        self.column_losses = column_losses or {}
        self.column_weights = column_weights or {}
        self.alpha = alpha

        self.loss1 = _get_loss_function(self.column_losses.get("0", "mse"), self.alpha)
        self.loss2 = _get_loss_function(self.column_losses.get("1", "mse"), self.alpha)
        if "2" in self.column_losses:
            self.loss3 = _get_loss_function(self.column_losses.get("2"), self.alpha)
        if "3" in self.column_losses:
            self.loss4 = _get_loss_function(self.column_losses.get("3"), self.alpha)
        self.weight1 = self.column_weights.get("0", 1.0)
        self.weight2 = self.column_weights.get("1", 1.0)
        if "2" in self.column_weights:
            self.weight3 = self.column_weights.get("2", 1.0)
        if "3" in self.column_weights:
            self.weight4 = self.column_weights.get("3", 1.0)

    def call(self, y_true, y_pred):
        """
        Compute the combined loss.

        Args:
            inputs: List [y_true, y_pred] where both have shape (batch_size, 2)

        Returns:
            Combined loss tensor with shape (batch_size,)
        """
        # Extract columns
        y_true_col1 = y_true[:, 0]  # First column for MSE
        y_true_col2 = y_true[:, 1]  # Second column for BCE

        y_pred_col1 = y_pred[:, 0]  # First column for MSE
        y_pred_col2 = y_pred[:, 1]  # Second column for BCE

        # Calculate individual losses (per sample)
        mse_loss_val = self.loss1(y_true_col1, y_pred_col1)
        bce_loss_val = self.loss2(y_true_col2, y_pred_col2)

        # Combine losses with weights
        total_loss = (self.weight1 * mse_loss_val) + (self.weight2 * bce_loss_val)

        if "2" in self.column_losses:
            y_true_col3 = y_true[:, 2]
            y_pred_col3 = y_pred[:, 2]
            loss3_val = self.loss3(y_true_col3, y_pred_col3)
            total_loss += self.weight3 * loss3_val
        if "3" in self.column_losses:
            y_true_col4 = y_true[:, 3]
            y_pred_col4 = y_pred[:, 3]
            loss4_val = self.loss4(y_true_col4, y_pred_col4)
            total_loss += self.weight4 * loss4_val

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "column_losses": self.column_losses,
                "column_weights": self.column_weights,
                "alpha": self.alpha,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


tf.keras.utils.get_custom_objects()["MultiColumnLossLayer"] = MultiColumnLossLayer
