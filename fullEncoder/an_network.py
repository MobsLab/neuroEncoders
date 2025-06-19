# Pierre 14/02/21:
# Reorganization of the code:
# One class for the network
# One function for the training boom nahui
# We save the model every epoch during the training
# Dima 21/01/22:
# Cleanining and rewriting of the module

import os
import warnings

from utils.global_classes import DataHelper

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import matplotlib.pyplot as plt

# Get common libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Get utility functions
from fullEncoder import nnUtils
from importData.epochs_management import inEpochsMask


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
        projectPath,
        params,
        deviceName="/device:CPU:0",
        debug=False,
        phase=None,
        **kwargs,
    ):
        super(LSTMandSpikeNetwork, self).__init__()
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
        self._build_model(**kwargs)

    def _setup_folders(self):
        self.folderResult = os.path.join(self.projectPath.experimentPath, "results")
        self.folderResultSleep = os.path.join(
            self.projectPath.experimentPath, "results_Sleep"
        )
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
                from fullEncoder.nnUtils import (
                    MaskedGlobalAveragePooling1D,
                    PositionalEncoding,
                    TransformerEncoderBlock,
                )

                self.lstmsNets = (
                    [PositionalEncoding(d_model=self.params.nFeatures)]
                    + [
                        TransformerEncoderBlock(
                            d_model=self.params.nFeatures,
                            num_heads=self.params.nHeads,
                            ff_dim1=self.params.ff_dim1,
                            ff_dim2=self.params.ff_dim2,
                            dropout_rate=self.params.dropoutLSTM,
                        )
                        for _ in range(self.params.lstmLayers)
                    ]
                    + [
                        MaskedGlobalAveragePooling1D(),
                        tf.keras.layers.Dense(
                            1024,
                            activation=tf.nn.relu,
                            kernel_regularizer="l2",
                        ),
                        tf.keras.layers.Dense(
                            512,
                            activation=tf.nn.relu,
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
                self.params.lstmSize, activation=tf.nn.silu, kernel_regularizer="l2"
            )
            self.denseLoss3 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.silu, kernel_regularizer="l2"
            )
            self.denseLoss4 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.silu, kernel_regularizer="l2"
            )
            self.denseLoss5 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.silu, kernel_regularizer="l2"
            )
            self.denseLoss2 = tf.keras.layers.Dense(
                1,
                activation=kwargs.get("lossActivation", self.params.lossActivation),
                bias_initializer="ones",
                name="predicted_loss",
            )
            self.epsilon = tf.constant(10 ** (-8))
            # Outputs
            self.denseFeatureOutput = tf.keras.layers.Dense(
                self.params.dimOutput,
                activation=tf.keras.activations.hard_sigmoid,  # ensures output is in [0,1]
                dtype=tf.float32,
                name="feature_output",
                # kernel_regularizer="l2",
            )

            self.projection_layer = tf.keras.layers.Dense(
                self.params.nFeatures,
                activation="relu",
                dtype=tf.float32,
                name="feature_projection",
            )

            # Gather the full model
            outputs = self.generate_model(**kwargs)
            # Build two models
            # One just described, with two objective functions corresponding
            # to both position and predicted losses
            self.model = self.compile_model(outputs, **kwargs)
            # In theory, the predicted loss could be not learning enough in the first network (optional)
            # Second only with loss corresponding to predicted loss
            self.predLossModel = self.compile_model(
                outputs, predLossOnly=True, **kwargs
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
            allFeatures = tf.concat(allFeatures, axis=2)  # , name="concat1"
            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(
                self.inputGroups, [self.params.batchSize, -1]
            )
            mymask = tf.not_equal(batchedInputGroups, -1)

            sumFeatures = tf.math.reduce_sum(
                allFeatures, axis=1
            )  # This var will be used in the predLoss loss

            allFeatures_raw = allFeatures
            allFeatures = self.dropoutLayer(allFeatures)

            # LSTM
            if not kwargs.get("isTransformer", False):
                print("using LSTM architecture !")
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
                myoutputPos = self.denseFeatureOutput(
                    self.dropoutLayer(output)
                )  # positions
            else:
                print("Using Transformer architecture !")
                allFeatures = self.projection_layer(allFeatures)
                sumFeatures = tf.math.reduce_sum(
                    self.projection_layer(allFeatures_raw), axis=1
                )
                allFeatures = self.lstmsNets[0](allFeatures)
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

                output = self.lstmsNets[-3](output, mask=mymask)
                x = self.lstmsNets[-2](output)
                x = self.lstmsNets[-1](x)
                myoutputPos = self.denseFeatureOutput(x)

            print("myoutputPos =", myoutputPos)

            outputPredLoss = self.denseLoss2(
                # WARNING: might have an activation function here.
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
                loss_function = combined_loss(self.params)

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

            tempPosLoss = loss_function(myoutputPos, self.truePos)[:, tf.newaxis]
            # for main loss functions:
            # if loss function is mse
            # tempPosLoss is in cm2
            # if loss function is logcosh
            # tempPosLoss is in cm2

            posLoss = tf.identity(tf.math.reduce_mean(tempPosLoss), name="posLoss")

            # remark: we need to also stop the gradient to propagate from posLoss to the network at the stage of
            # the computations for the loss of the loss predictor
            # still ~ in cm2
            loss_function_PredLoss = tf.keras.losses.mean_squared_error
            # # outputPredLoss is supposed to be in cm2 and predict the MSE loss.
            # outputPredLoss = tf.math.scalar_mul(cst_predLoss, outputPredLoss)

            # preUncertaintyLoss is in cm2^2 as it's the MSE between the predicted loss and the posLoss
            preUncertaintyLoss = loss_function_PredLoss(
                outputPredLoss, tf.stop_gradient(tempPosLoss)
            )

            # back to cm to compute the uncertainty loss as the MSE between the predicted loss and the posLoss
            uncertaintyLoss = tf.identity(preUncertaintyLoss, name="uncertaintyLoss")
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
        if not predLossOnly:
            # Full model
            model.compile(
                # TODO: Adam or AdaGrad?
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
                optimizer=tf.keras.optimizers.Adagrad(
                    learning_rate=kwargs.get("lr", self.params.learningRates[0])
                ),  # Initially compile with first lr.
                loss={
                    # tf_op_layer_ position loss (eucledian distance between predicted and real coordinates)
                    outputs[2].name.split("/Identity")[0]: pos_loss,
                    # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss)
                    outputs[3].name.split("/Identity")[0]: uncertainty_loss,
                },
            )
            # Get internal names of losses
            self.outNames = [
                outputs[2].name.split("/Identity")[0],
                outputs[3].name.split("/Identity")[0],
            ]
        else:
            # Only used to create the self.predLossModel
            model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]),
                optimizer=tf.keras.optimizers.Adagrad(
                    learning_rate=kwargs.get("lr", self.params.learningRates[0])
                ),
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

        Returns
        -------
        None
        """

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

        from fullEncoder.nnUtils import (
            NeuralDataAugmentation,
            create_flatten_augmented_groups_fn,
        )

        augmentation_config = NeuralDataAugmentation()

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
            datasets[key] = datasets[key].batch(
                self.params.batchSize, drop_remainder=True
            )
            if key != "test":
                print("Applying data augmentation to", key, "dataset")
                datasets[key] = datasets[key].map(
                    map_parse_serialized_sequence_with_augmentation,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )  # self.featDesc, *
                datasets[key] = datasets[key].flat_map(
                    flatten_fn
                )  # Flatten the augmented groups
            else:
                print("No data augmentation for", key, "dataset")
                datasets[key] = datasets[key].map(
                    map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
                )  # self.featDesc, *

            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            datasets[key] = datasets[key].map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = datasets[key].map(
                map_outputs, num_parallel_calls=tf.data.AUTOTUNE
            )
            # We shuffle the datasets and cache it - this way the training samples are randomized for each epoch
            # and each mini-batch contains a representative sample of the training set.
            # nSteps represent the buffer size of the shuffle operation - 10 seconds worth of buffer starting
            # from the 0-timepoint of the dataset.
            # once an element is selected, its space in the buffer is replaced by the next element (right after the 10s window...)
            # At each epoch, the shuffle order is different.

            # datasets[key] = (
            #     datasets[key]
            #     .shuffle(
            #         self.params.nSteps, reshuffle_each_iteration=True
            #     )  # NOTE: nSteps = int(10000 * 0.036 / windowSize), 10s worth of buffer
            #     .cache()
            #     .repeat()
            # )  #

            datasets[key] = datasets[key].prefetch(tf.data.AUTOTUNE)  #

        ### Train the model(s)
        # Initialize the model for C++ decoder
        # self.generate_model_Cplusplus()
        # Train
        for key in checkpointPath.keys():
            print("Training the", key, "model")
            if load_model and os.path.exists(os.path.dirname(checkpointPath[key])):
                if key != "predLoss":
                    print(
                        "Loading the weights of the loss training model from",
                        checkpointPath[key],
                    )
                    try:
                        self.model.load_weights(checkpointPath[key])
                        continue
                    except Exception as e:
                        print(
                            "Error loading weights for",
                            key,
                            "from",
                            checkpointPath[key],
                            ":",
                            e,
                        )
                elif key == "predLoss":
                    print(
                        "Loading the weights of the loss predictor model from",
                        checkpointPath[key],
                    )
                    try:
                        self.predLossModel.load_weights(checkpointPath[key])
                        continue
                    except Exception as e:
                        print(
                            "Error loading weights for",
                            key,
                            "from",
                            checkpointPath[key],
                            ":",
                            e,
                        )

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpointPath[key], save_weights_only=True, verbose=1
            )
            # Manage learning rates schedule
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
                log_dir = os.path.join(
                    "logs",
                    os.path.basename(self.projectPath.experimentPath),
                    str(windowSizeMS),
                    os.path.basename(os.path.dirname(self.projectPath.xml)),
                    key,
                )
                tb_callbacks = tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                )

                # tf.debugging.experimental.enable_dump_debug_info(
                #     self.folderResult,
                #     tensor_debug_mode="FULL_HEALTH",
                #     circular_buffer_size=-1,
                # )               )

                if key != "predLoss":
                    from wandb import keras as wandbkeras

                    WandbMetricsLogger = wandbkeras.WandbMetricsLogger

                    import wandb

                    wandb.tensorboard.patch(
                        root_logdir=os.path.join(
                            "logs",
                            os.path.basename(self.projectPath.experimentPath),
                        )
                    )
                    # ann config
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

                    wandb.init(
                        entity="touseul",
                        project="neuroEncoder",
                        notes=f"{os.path.basename(self.projectPath.experimentPath)}_{key}",
                        sync_tensorboard=True,
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
                        monitor="val_tf.identity_1_loss", patience=5, min_delta=0.01
                    )
                    callbacks = [csvLogger[key], cp_callback, schedule, es_callback]
                else:
                    callbacks = [csvLogger[key], cp_callback, schedule]

                if is_tbcallback:
                    callbacks.append(tb_callbacks)

                hist = self.predLossModel.fit(
                    datasets["predLoss"],
                    epochs=self.params.nEpochs,
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
                # Save model for C++ decoder
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels,
                #                     str(windowSizeMS), "savedModels","predLossModel"))
                self.predLossModel.save(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "predLossModel.keras",
                    )
                )
            else:
                if earlyStop:
                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        min_delta=0.01,
                        verbose=1,
                        restore_best_weights=True,
                        start_from_epoch=5,
                    )
                    callbacks = [csvLogger[key], cp_callback, schedule, es_callback]
                else:
                    callbacks = [csvLogger[key], cp_callback, schedule]

                if is_tbcallback:
                    callbacks.append(tb_callbacks)
                    callbacks.append(wandb_callback)

                hist = self.model.fit(
                    datasets["train"],
                    epochs=self.params.nEpochs,
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
                # Save model for C++ decoder
                # self.cplusplusModel.predict(datasets['train'])
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels, str(windowSizeMS), "savedModels","fullModel"))
                self.model.save(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "fullModel.keras",
                    )
                )
        if self.debug:
            wandb.tensorboard.unpatch()
            wandb.finish()

    def train_binary(
        self,
        behaviorData,
        onTheFlyCorrection=False,
        windowSizeMS=36,
        scheduler="decay",
        isPredLoss=True,
        earlyStop=False,
    ):
        ### Create neccessary arrays
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

        def import_true_pos(x):
            return nnUtils.import_true_pos(posFeature)(x)

        def filter_nan_pos(x):
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))

        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

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
            if onTheFlyCorrection:
                maxPos = np.nanmax(
                    behaviorData["pos"][
                        np.logical_not(np.isnan(np.sum(behaviorData["pos"], axis=1)))
                    ]
                )
                # WARNING: where is this "pos" index coming from ?
                # could be nowhere except for nnBehavior.mat - but never created
                # TODO: change to Positions - might be because it is the binary target??
                # TODO: implement if target is binary in main
                posFeature = behaviorData["pos"] / maxPos
            else:
                posFeature = behaviorData["pos"]
            datasets[key] = datasets[key].map(import_true_pos)
            datasets[key] = datasets[key].filter(filter_nan_pos)
            datasets[key] = datasets[key].batch(
                self.params.batchSize, drop_remainder=True
            )
            datasets[key] = datasets[key].map(
                map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
            )  # self.featDesc, *
            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            datasets[key] = datasets[key].map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = datasets[key].map(
                map_outputs, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = (
                datasets[key]
                .shuffle(self.params.nSteps, reshuffle_each_iteration=True)
                .cache()
            )  # .repeat() #
            datasets[key] = datasets[key].prefetch(tf.data.AUTOTUNE)  #

        ### Train the model(s)
        # Initialize the model for C++ decoder
        # self.generate_model_Cplusplus()
        # Train
        for key in checkpointPath.keys():
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpointPath[key], save_weights_only=True, verbose=1
            )
            # Manage learning rates schedule
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
            # # In case you need debugging, uncomment this profiling line
            # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.folderResult)
            if key == "predLoss":
                self.predLossModel.load_weights(
                    checkpointPath["full"]
                )  # Load weights from full network if we train
                if earlyStop:
                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="tf.identity_1_loss", patience=1
                    )
                    callbacks = [csvLogger[key], cp_callback, schedule, es_callback]
                else:
                    callbacks = [csvLogger[key], cp_callback, schedule]
                hist = self.predLossModel.fit(
                    datasets["predLoss"],
                    epochs=self.params.nEpochs,
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
                # Save model for C++ decoder
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels,
                #                     str(windowSizeMS), "savedModels","predLossModel"))
                self.predLossModel.save(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "predLossModel.keras",
                    )
                )
            else:
                if earlyStop:
                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=1
                    )
                    callbacks = [csvLogger[key], cp_callback, schedule, es_callback]
                else:
                    callbacks = [csvLogger[key], cp_callback, schedule]
                hist = self.model.fit(
                    datasets["train"],
                    epochs=self.params.nEpochs,
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
                # Save model for C++ decoder
                # self.cplusplusModel.predict(datasets['train'])
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels, str(windowSizeMS), "savedModels","fullModel"))
                self.model.save(
                    os.path.join(
                        self.folderModels,
                        str(windowSizeMS),
                        "savedModels",
                        "fullModel.keras",
                    )
                )

    def test_binary(
        self,
        behaviorData,
        l_function=[],
        windowSizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=False,
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
        speedMask = behaviorData["Times"]["speedFilter"]
        if useTrain:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
            )
        else:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
            )
        if useSpeedFilter:
            totMask = speedMask * epochMask
        else:
            totMask = epochMask

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
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))

        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

        def map_outputs(vals):
            return (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            )

        dataset = dataset.filter(filter_by_pos_index)
        if onTheFlyCorrection:
            maxPos = np.nanmax(
                behaviorData["pos"][
                    np.logical_not(np.isnan(np.sum(behaviorData["pos"], axis=1)))
                ]
            )
            posFeature = behaviorData["pos"] / maxPos
        else:
            posFeature = behaviorData["pos"]
        dataset = dataset.map(import_true_pos)
        dataset = dataset.filter(filter_nan_pos)
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)
        print("INFERRING")
        outputTest = self.model.predict(dataset, verbose=1)

        ### Post-inferring management
        print("gathering true feature")
        datasetPos = dataset.map(
            lambda x, y: x["pos"], num_parallel_calls=tf.data.AUTOTUNE
        )
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        featureTrue = np.reshape(
            fullFeatureTrue, [outputTest[0].shape[0], outputTest[0].shape[-1]]
        )
        print("gathering times of the centre in the time window")
        datasetTimes = dataset.map(
            lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE
        )
        times = list(datasetTimes.as_numpy_iterator())
        times = np.reshape(times, [outputTest[0].shape[0]])
        print("gathering indices of spikes relative to coordinates")
        datasetPos_index = dataset.map(
            lambda x, y: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE
        )
        posIndex = list(datasetPos_index.as_numpy_iterator())
        posIndex = np.ravel(np.array(posIndex))
        print("gathering speed mask")
        windowmaskSpeed = speedMask[posIndex]
        outLoss = np.expand_dims(outputTest[2], axis=1)

        testOutput = {
            "featurePred": outputTest[0],
            "featureTrue": featureTrue,
            "times": times,
            "predLoss": outputTest[1],
            "lossFromOutputLoss": outLoss,
            "posIndex": posIndex,
            "speedMask": windowmaskSpeed,
        }

        if l_function:
            projPredPos, linearPred = l_function(outputTest[0][:, :2])
            projTruePos, linearTrue = l_function(featureTrue)
            testOutput["projPred"] = projPredPos
            testOutput["projTruePos"] = projTruePos
            testOutput["linearPred"] = linearPred
            testOutput["linearTrue"] = linearTrue

        # Save the results
        self.saveResults(testOutput, folderName=windowSizeMS)

        return testOutput

    def testControl(
        self,
        behaviorData,
        modelPath,
        l_function=[],
        windowSizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowSizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowSizeMS)))
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(modelPath, str(windowSizeMS), "predLoss" + "/cp.ckpt")
            )
        else:
            self.model.load_weights(
                os.path.join(modelPath, str(windowSizeMS), "full" + "/cp.ckpt")
            )

        # Manage the behavior
        speedMask = behaviorData["Times"]["speedFilter"]
        if useTrain:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
            )
        else:
            epochMask = inEpochsMask(
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
            )
        if useSpeedFilter:
            totMask = speedMask * epochMask
        else:
            totMask = epochMask

        # Load the and imfer dataset
        dataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )
        )
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_spike(self.featDesc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))

        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

        def map_outputs(vals):
            return (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
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
        dataset = dataset.map(import_true_pos)
        dataset = dataset.filter(filter_nan_pos)
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)
        print("INFERRING")
        outputTest = self.model.predict(dataset, verbose=1)

        ### Post-inferring management
        print("gathering true feature")
        datasetPos = dataset.map(
            lambda x, y: x["pos"], num_parallel_calls=tf.data.AUTOTUNE
        )
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        featureTrue = np.reshape(
            fullFeatureTrue, [outputTest[0].shape[0], outputTest[0].shape[-1]]
        )
        print("gathering times of the centre in the time window")
        datasetTimes = dataset.map(
            lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE
        )
        times = list(datasetTimes.as_numpy_iterator())
        times = np.reshape(times, [outputTest[0].shape[0]])
        print("gathering indices of spikes relative to coordinates")
        datasetPos_index = dataset.map(
            lambda x, y: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE
        )
        posIndex = list(datasetPos_index.as_numpy_iterator())
        posIndex = np.ravel(np.array(posIndex))
        print("gathering speed mask")
        windowmaskSpeed = speedMask[posIndex]
        outLoss = np.expand_dims(outputTest[2], axis=1)

        testOutput = {
            "featurePred": outputTest[0],
            "featureTrue": featureTrue,
            "times": times,
            "predLoss": outputTest[1],
            "lossFromOutputLoss": outLoss,
            "posIndex": posIndex,
            "speedMask": windowmaskSpeed,
        }

        if l_function:
            projPredPos, linearPred = l_function(outputTest[0][:, :2])
            projTruePos, linearTrue = l_function(featureTrue)
            testOutput["projPred"] = projPredPos
            testOutput["projTruePos"] = projTruePos
            testOutput["linearPred"] = linearPred
            testOutput["linearTrue"] = linearTrue

        # Save the results
        self.saveResults(testOutput, folderName=windowSizeMS)

        return testOutput

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

        # TODO: change speed filter with custom speed
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
                behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
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
            return tf.math.greater(table.lookup(x["pos_index"]), 0)

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
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
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
        dataset = dataset.map(map_outputs, num_parallel_calls=tf.data.AUTOTUNE)

        final_dataset = self.full_dataset_inspection(
            windowSizeMS, behaviorData, totMask, onTheFlyCorrection
        )
        print("INFERRING")
        outputTest = self.model.predict(final_dataset, verbose=1)

        # Inspect outputs
        print("\nMODEL OUTPUT INSPECTION:")
        if isinstance(outputTest, dict):
            for key, value in outputTest.items():
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                print(f"  Min/Max: {np.min(value):.4f} / {np.max(value):.4f}")
                print(f"  Mean/Std: {np.mean(value):.4f} / {np.std(value):.4f}")
        elif isinstance(outputTest, (list, tuple)):
            for i, output in enumerate(outputTest):
                print(f"Output {i}: shape={output.shape}, dtype={output.dtype}")
                print(f"  Min/Max: {np.min(output):.4f} / {np.max(output):.4f}")
        else:
            print(f"Output shape: {outputTest.shape}, dtype={outputTest.dtype}")
            print(f"Min/Max: {np.min(outputTest):.4f} / {np.max(outputTest):.4f}")
            print(f"Mean/Std: {np.mean(outputTest):.4f} / {np.std(outputTest):.4f}")

        if self.target.lower() == "direction":
            outputTest = (tf.cast(outputTest[0] > 0.5, tf.int32),)

        ### Post-inferring management
        print("gathering true feature")
        datasetPos = dataset.map(
            lambda x, y: x["pos"], num_parallel_calls=tf.data.AUTOTUNE
        )
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        featureTrue = np.reshape(
            fullFeatureTrue, [outputTest[0].shape[0], outputTest[0].shape[-1]]
        )
        print("gathering times of the centre in the time window")
        datasetTimes = dataset.map(
            lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE
        )
        times = list(datasetTimes.as_numpy_iterator())
        times = np.reshape(times, [outputTest[0].shape[0]])
        print("gathering indices of spikes relative to coordinates")
        datasetPos_index = dataset.map(
            lambda x, y: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE
        )
        posIndex = list(datasetPos_index.as_numpy_iterator())
        posIndex = np.ravel(np.array(posIndex))
        print("gathering speed mask")
        windowmaskSpeed = speedMask[posIndex]
        outLoss = np.expand_dims(outputTest[2], axis=1)
        loss_from_output_loss = np.expand_dims(outputTest[3], axis=1)

        testOutput = {
            "featurePred": outputTest[0],
            "featureTrue": featureTrue,
            "times": times,
            "predLoss": outputTest[1],
            "lossFromOutputLoss": outLoss,
            "posIndex": posIndex,
            "speedMask": windowmaskSpeed,
            "LossFromUncertaintyLoss": loss_from_output_loss,
        }

        if l_function:
            projPredPos, linearPred = l_function(outputTest[0][:, :2])
            projTruePos, linearTrue = l_function(featureTrue)
            testOutput["projPred"] = projPredPos
            testOutput["projTruePos"] = projTruePos
            testOutput["linearPred"] = linearPred
            testOutput["linearTrue"] = linearTrue

        # Save the results
        self.saveResults(testOutput, folderName=windowSizeMS, phase=phase)

        return testOutput

    def testSleep(
        self,
        behaviorData,
        l_function=[],
        windowSizeDecoder=None,
        windowSizeMS=36,
        isPredLoss=False,
    ):
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
                    self.folderModels, str(windowSizeDecoder), "predLoss" + "/cp.ckpt"
                )
            )
        else:
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

            def map_parse_serialized_sequence(*vals):
                return nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                )

            def map_outputs(vals):
                return (
                    vals,
                    {
                        "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                        "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
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

            # Post-infer management
            print(f"gathering times of the centre in the time window for {sleepName}")
            datasetTimes = dataset.map(
                lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE
            )
            times = list(datasetTimes.as_numpy_iterator())
            times = np.ravel(times)
            print(
                f"gathering indices of spikes relative to coordinates for {sleepName}"
            )
            datasetPosIndex = dataset.map(
                lambda x, y: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE
            )
            posIndex = list(datasetPosIndex.as_numpy_iterator())
            posIndex = np.ravel(np.array(posIndex))
            #
            IDdat = dataset.map(
                lambda vals, y: vals["indexInDat"], num_parallel_calls=tf.data.AUTOTUNE
            )
            IDdat = list(IDdat.as_numpy_iterator())
            outLoss = np.expand_dims(output[2], axis=1)

            predictions[sleepName] = {
                "featurePred": output[0],
                "predLoss": output[1],
                "times": times,
                "posIndex": posIndex,
                "lossFromOutputLoss": outLoss,
                "indexInDat": IDdat,
            }
            if l_function:
                projPredPos, linearPred = l_function(output[0][:, :2])
                predictions[sleepName]["projPred"] = projPredPos
                predictions[sleepName]["linearPred"] = linearPred

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
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))

        def map_parse_serialized_sequence(*vals):
            return nnUtils.parse_serialized_sequence(self.params, *vals, batched=True)

        def map_outputs(vals):
            return (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
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
            ax[1].set_title("log loss prediction loss")
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
        self, test_output, folderName=36, sleep=False, sleepName="Sleep", phase=None
    ):
        # Manage folders to save
        if sleep:
            folderToSave = os.path.join(
                self.folderResultSleep, str(folderName), sleepName
            )
            if not os.path.isdir(folderToSave):
                os.makedirs(folderToSave)
        else:
            folderToSave = os.path.join(self.folderResult, str(folderName))

        if phase is not None:
            suffix = f"_{phase}"
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
        if "projTruePos" in test_output:
            df = pd.DataFrame(test_output["projTruePos"])
            df.to_csv(os.path.join(folderToSave, f"projTrueFeature{suffix}.csv"))
        if "linearPred" in test_output:
            df = pd.DataFrame(test_output["linearPred"])
            df.to_csv(os.path.join(folderToSave, f"linearPred{suffix}.csv"))
        if "linearTrue" in test_output:
            df = pd.DataFrame(test_output["linearTrue"])
            df.to_csv(os.path.join(folderToSave, f"linearTrue{suffix}.csv"))

    @classmethod
    def clear_session(cls):
        tf.keras.backend.clear_session()

    # Dataset and Model Inspection Tools
    # =============================================================================
    # 1. INSPECT DATASET AT DIFFERENT STAGES
    # =============================================================================

    def inspect_dataset_stages(
        self, windowSizeMS, behaviorData, totMask, onTheFlyCorrection=False
    ):
        """Inspect dataset at each processing stage"""

        print("=" * 60)
        print("DATASET INSPECTION AT EACH STAGE")
        print("=" * 60)

        # Stage 1: Raw TFRecord loading
        print("\n1. RAW TFRECORD DATA:")
        dataset_raw = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )
        )

        print(f"Raw dataset type: {type(dataset_raw)}")
        # Take first few raw records to see structure
        for i, raw_record in enumerate(dataset_raw.take(2)):
            print(f"Raw record {i} type: {type(raw_record)}")
            print(f"Raw record {i} shape: {raw_record.shape}")
            if i >= 1:
                break

        # Stage 2: After parsing
        print("\n2. AFTER PARSING:")

        def parse_function(vals):
            return nnUtils.parse_serialized_spike(self.featDesc, vals)

        dataset_parsed = dataset_raw.map(
            parse_function, num_parallel_calls=tf.data.AUTOTUNE
        )

        for i, parsed_data in enumerate(dataset_parsed.take(2)):
            for key, value in parsed_data.items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

                # Handle SparseTensor
                if isinstance(value, tf.SparseTensor):
                    print(f"    SparseTensor - indices shape: {value.indices.shape}")
                    print(f"    SparseTensor - values shape: {value.values.shape}")
                    print(
                        f"    SparseTensor - dense shape: {value.dense_shape.numpy()}"
                    )
                    print(f"    SparseTensor - non-zero count: {value.values.shape[0]}")

                    if value.values.shape[0] < 20:  # Small sparse tensors
                        print(f"    indices: {value.indices.numpy()}")
                        print(f"    values: {value.values.numpy()}")
                    else:
                        print(f"    first 5 indices: {value.indices.numpy()[:5]}")
                        print(f"    first 5 values: {value.values.numpy()[:5]}")

                    # Convert to dense for statistics (if not too large)
                    if tf.reduce_prod(value.dense_shape) < 10000:  # Avoid memory issues
                        dense_tensor = tf.sparse.to_dense(value)
                        print(
                            f"    Dense min/max: {tf.reduce_min(dense_tensor):.4f} / {tf.reduce_max(dense_tensor):.4f}"
                        )
                        print(
                            f"    Sparsity: {1 - value.values.shape[0] / tf.reduce_prod(value.dense_shape).numpy():.4f}"
                        )

                # Handle regular tensors
                elif hasattr(value, "numpy"):
                    if value.shape.num_elements() < 20:
                        print(f"    values: {value.numpy()}")
                    else:
                        print(f"    sample values: {value.numpy().flat[:5]}...")
                        print(
                            f"    min/max: {tf.reduce_min(value):.4f} / {tf.reduce_max(value):.4f}"
                        )

                # Handle other tensor types
                else:
                    print(
                        f"    tensor type: {type(value)} - limited inspection available"
                    )

        else:
            print(f"  Type: {type(parsed_data)}")
            if hasattr(parsed_data, "shape"):
                print(f"  Shape: {parsed_data.shape}")

        # Stage 3: After filtering
        print("\n3. AFTER FILTERING:")
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(np.arange(len(totMask)), dtype=tf.int64),
                tf.constant(totMask, dtype=tf.float64),
            ),
            default_value=0,
        )

        def filter_by_pos_index(x):
            return tf.math.greater(table.lookup(x["pos_index"]), 0)

        def filter_nan_pos(x):
            pos_data = x["pos"]
            if pos_data.dtype in [tf.int32, tf.int64]:
                pos_data = tf.cast(pos_data, tf.float64)
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        dataset_filtered = dataset_parsed.filter(filter_by_pos_index)

        # # Count before and after filtering
        # count_before = sum(1 for _ in dataset_parsed)
        # count_after_pos_filter = sum(1 for _ in dataset_filtered)

        dataset_filtered = dataset_filtered.map(
            nnUtils.import_true_pos(
                behaviorData["Positions"] / np.nanmax(behaviorData["Positions"])
                if onTheFlyCorrection
                else behaviorData["Positions"]
            )
        )
        dataset_filtered = dataset_filtered.filter(filter_nan_pos)

        # count_final = sum(1 for _ in dataset_filtered)

        # print(f"  Samples before filtering: {count_before}")
        # print(f"  Samples after pos_index filter: {count_after_pos_filter}")
        # print(f"  Samples after NaN filter: {count_final}")
        # print(
        #     f"  Filtering removed: {count_before - count_final} samples ({100 * (count_before - count_final) / count_before:.1f}%)"
        # )

        # Stage 4: After batching and final processing
        print("\n4. FINAL PROCESSED DATASET (Model Input):")
        dataset_final = dataset_filtered.batch(
            self.params.batchSize, drop_remainder=True
        )

        def map_parse_serialized_sequence(vals):
            return nnUtils.parse_serialized_sequence(self.params, vals, batched=True)

        def map_outputs(vals):
            return (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            )

        dataset_final = dataset_final.map(
            map_parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_final = dataset_final.map(
            self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_final = dataset_final.map(
            map_outputs, num_parallel_calls=tf.data.AUTOTUNE
        )

        print(f"  Batch size: {self.params.batchSize}")
        print(f"  Number of batches: {sum(1 for _ in dataset_final)}")

        # Inspect first batch
        for batch_idx, (inputs, targets) in enumerate(dataset_final.take(1)):
            print(f"\n  BATCH {batch_idx} STRUCTURE:")
            print("  Inputs:")
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(inputs, (list, tuple)):
                for i, inp in enumerate(inputs):
                    print(f"    input_{i}: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"    single input: shape={inputs.shape}, dtype={inputs.dtype}")

            print("  Targets:")
            if isinstance(targets, dict):
                for key, value in targets.items():
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")

        return dataset_final

    # =============================================================================
    # 2. DETAILED SAMPLE INSPECTION
    # =============================================================================

    def inspect_sample_details(self, dataset, num_samples=2):
        """Inspect individual samples in detail"""

        print("\n" + "=" * 60)
        print("DETAILED SAMPLE INSPECTION")
        print("=" * 60)

        for batch_idx, (inputs, targets) in enumerate(dataset.take(1)):
            batch_size = (
                inputs[list(inputs.keys())[0]].shape[0]
                if isinstance(inputs, dict)
                else inputs.shape[0]
            )

            for sample_idx in range(min(num_samples, batch_size)):
                print(f"\n--- SAMPLE {sample_idx} ---")

                if isinstance(inputs, dict):
                    for key, value in inputs.items():
                        sample_data = value[sample_idx]
                        print(f"{key}:")
                        print(f"  Shape: {sample_data.shape}")
                        print(f"  Dtype: {sample_data.dtype}")
                        print(
                            f"  Min/Max: {tf.reduce_min(sample_data):.4f} / {tf.reduce_max(sample_data):.4f}"
                        )
                        print(
                            f"  Mean/Std: {tf.reduce_mean(sample_data):.4f} / {tf.math.reduce_std(sample_data):.4f}"
                        )

                        # Show some actual values for small tensors
                        if sample_data.shape.num_elements() <= 10:
                            print(f"  Values: {sample_data.numpy()}")
                        else:
                            print(f"  First few values: {sample_data.numpy().flat[:5]}")

    # =============================================================================
    # 3. MODEL INPUT/OUTPUT INSPECTION
    # =============================================================================

    def inspect_model_io(self, model, dataset, num_samples=1):
        """Inspect what the model receives and produces"""

        print("\n" + "=" * 60)
        print("MODEL INPUT/OUTPUT INSPECTION")
        print("=" * 60)

        # Model architecture summary
        print("\n1. MODEL ARCHITECTURE:")
        print(f"Model type: {type(model)}")
        print(f"Number of layers: {len(model.layers)}")

        # Input specifications
        print("\n2. MODEL INPUT SPECIFICATION:")
        if hasattr(model, "input_spec") and model.input_spec:
            for i, spec in enumerate(model.input_spec):
                print(f"  Input {i}: {spec}")

        if hasattr(model, "input_shape"):
            print(f"  Input shape: {model.input_shape}")

        # Output specifications
        print("\n3. MODEL OUTPUT SPECIFICATION:")
        if hasattr(model, "output_shape"):
            print(f"  Output shape: {model.output_shape}")

        # Test with actual data
        print("\n4. ACTUAL INPUT/OUTPUT WITH SAMPLE DATA:")

        for batch_idx, (inputs, targets) in enumerate(dataset.take(num_samples)):
            print(f"\n--- BATCH {batch_idx} ---")

            # Show input details
            print("INPUT TO MODEL:")
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(inputs, (list, tuple)):
                for i, inp in enumerate(inputs):
                    print(f"  input_{i}: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"  shape={inputs.shape}, dtype={inputs.dtype}")

            # Get model predictions
            print("\nMODEL PREDICTION:")
            try:
                predictions = model(inputs, training=False)

                if isinstance(predictions, dict):
                    for key, value in predictions.items():
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        print(
                            f"    Min/Max: {tf.reduce_min(value):.4f} / {tf.reduce_max(value):.4f}"
                        )
                        print(
                            f"    Mean/Std: {tf.reduce_mean(value):.4f} / {tf.math.reduce_std(value):.4f}"
                        )
                elif isinstance(predictions, (list, tuple)):
                    for i, pred in enumerate(predictions):
                        print(f"  output_{i}: shape={pred.shape}, dtype={pred.dtype}")
                        print(
                            f"    Min/Max: {tf.reduce_min(pred):.4f} / {tf.reduce_max(pred):.4f}"
                        )
                else:
                    print(f"  shape={predictions.shape}, dtype={predictions.dtype}")
                    print(
                        f"  Min/Max: {tf.reduce_min(predictions):.4f} / {tf.reduce_max(predictions):.4f}"
                    )
                    print(
                        f"  Mean/Std: {tf.reduce_mean(predictions):.4f} / {tf.math.reduce_std(predictions):.4f}"
                    )

            except Exception as e:
                print(f"  ERROR during prediction: {e}")

            # Show target details
            print("\nTARGET VALUES:")
            if isinstance(targets, dict):
                for key, value in targets.items():
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    if not tf.reduce_all(
                        tf.equal(value, 0)
                    ):  # Only show non-zero targets
                        print(
                            f"    Min/Max: {tf.reduce_min(value):.4f} / {tf.reduce_max(value):.4f}"
                        )

    # =============================================================================
    # 4. VISUALIZATION HELPERS
    # =============================================================================

    def visualize_sample_data(self, dataset, sample_idx=0):
        """Visualize neural data from a sample"""

        print("\n" + "=" * 60)
        print("SAMPLE DATA VISUALIZATION")
        print("=" * 60)

        for batch_idx, (inputs, targets) in enumerate(dataset.take(1)):
            if isinstance(inputs, dict):
                # Try to find neural/spike data to plot
                for key, value in inputs.items():
                    if (
                        "spike" in key.lower()
                        or "neural" in key.lower()
                        or "signal" in key.lower()
                    ):
                        sample_data = value[sample_idx].numpy()

                        plt.figure(figsize=(12, 6))

                        if len(sample_data.shape) == 1:
                            plt.plot(sample_data)
                            plt.title(f"Sample {sample_idx} - {key}")
                            plt.xlabel("Time")
                            plt.ylabel("Value")
                        elif len(sample_data.shape) == 2:
                            plt.imshow(sample_data.T, aspect="auto", origin="lower")
                            plt.title(f"Sample {sample_idx} - {key}")
                            plt.xlabel("Time")
                            plt.ylabel("Channel/Feature")
                            plt.colorbar()

                        plt.tight_layout()
                        plt.show()

                        print(f"Plotted {key} data for sample {sample_idx}")
                        break
            break

    # =============================================================================
    # 5. MAIN INSPECTION FUNCTION
    # =============================================================================

    def full_dataset_inspection(
        self, windowSizeMS, behaviorData, totMask, onTheFlyCorrection=False
    ):
        """Run complete dataset inspection pipeline"""

        # 1. Inspect dataset stages
        final_dataset = self.inspect_dataset_stages(
            windowSizeMS, behaviorData, totMask, onTheFlyCorrection
        )

        # 2. Inspect sample details
        self.inspect_sample_details(num_samples=2)

        # 3. Inspect model I/O
        if hasattr(self, "model") and self.model:
            self.inspect_model_io(self.model, final_dataset, num_samples=1)

        # 4. Optional visualization
        # visualize_sample_data(final_dataset, sample_idx=0)

        return final_dataset


def _get_loss_function(loss_name: str, alpha: int) -> tf.keras.losses.Loss:
    """Helper function to get loss function by name with reduction='none'"""
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError(reduction="none")
    elif loss_name == "huber":
        return tf.keras.losses.Huber(delta=alpha, reduction="none")
    elif loss_name == "msle":
        return tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")
    elif loss_name == "logcosh":
        return tf.keras.losses.LogCosh(reduction="none")
    elif loss_name == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy(reduction="none")
    elif loss_name == "categorical_crossentropy":
        # TODO
        return tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    elif loss_name == "mse_plus_msle":

        def combined_loss_mse(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError(reduction="none")(y_true, y_pred)
            msle = tf.keras.losses.MeanSquaredLogarithmicError(reduction="none")(
                y_true, y_pred
            )
            return mse + alpha * msle

        return combined_loss_mse
    else:
        raise ValueError(f"Loss function {loss_name} not recognized")


########### END OF HELPING LSTMandSpikeNetwork FUNCTIONS#####################


def combined_loss(params):
    """
    Creates a custom loss function combining MSE (default) for position and
    categorical crossentropy (also default) for classification.

    Args:
        params: Parameters object containing model configuration and loss names
        position_weight: Weight for the position (MSE) loss component
        non_position_weight: Weight for the non position loss component

    Returns:
        Custom loss function
    """

    column_losses = params.column_losses
    pos_loss_func = _get_loss_function(column_losses["0"], params.alpha)
    class_loss_func = _get_loss_function(column_losses["1"], params.alpha)
    column_weights = params.column_weights
    position_weight = column_weights["0"]
    non_position_weight = column_weights["1"]
    print(f"Using position loss: {column_losses['0']} with weight {position_weight}")
    print(
        f"Using classification loss: {column_losses['1']} with weight {non_position_weight}"
    )

    def loss_function(y_true, y_pred):
        # Split the true and predicted values
        # Assuming y_true and y_pred have shape (batch_size, 2)
        # where [:, 0] is position and [:, 1] is class label

        # Position component (continuous) - MSE loss
        position_true = y_true[:, 0:1]  # Keep dimension for broadcasting
        position_pred = y_pred[:, 0:1]
        position_loss = pos_loss_func(position_true, position_pred)

        # Classification component - categorical crossentropy
        # If using integer labels (sparse), use sparse_categorical_crossentropy
        class_true = tf.cast(y_true[:, 1], tf.int32)
        class_pred = y_pred[:, 1:]  # Remaining dimensions are class probabilities

        # Use sparse categorical crossentropy if you have integer labels
        classification_loss = class_loss_func(class_true, class_pred)

        # Combine losses with weights
        total_loss = (
            position_weight * position_loss + non_position_weight * classification_loss
        )

        return total_loss

    return loss_function


@tf.keras.utils.register_keras_serializable(package="Custom")
class MultiColumnLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        column_losses=None,
        column_weights=None,
        alpha=1.0,
        name="multi_column_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.column_losses = column_losses or {}
        self.column_weights = column_weights or {}
        self.alpha = alpha

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

    def call(self, y_true, y_pred):
        """Custom loss function for model.compile() with proper batch handling"""
        total_loss = 0.0

        for column_spec, loss_name in self.column_losses.items():
            # Parse column specification
            if "," in column_spec:
                columns = [int(c.strip()) for c in column_spec.split(",")]
            else:
                columns = [int(column_spec)]

            # Extract the specified columns
            if len(columns) == 1:
                y_true_cols = y_true[:, columns[0] : columns[0] + 1]
                y_pred_cols = y_pred[:, columns[0] : columns[0] + 1]
            else:
                y_true_cols = tf.gather(y_true, columns, axis=1)
                y_pred_cols = tf.gather(y_pred, columns, axis=1)

            # Get the loss function and compute per-sample loss
            loss_fn = _get_loss_function(loss_name, alpha=self.alpha)
            column_loss = loss_fn(y_true_cols, y_pred_cols)

            # Reduce to per-sample scalar if needed
            if len(column_loss.shape) > 1:
                column_loss = tf.reduce_mean(column_loss, axis=-1)

            # Apply weight
            weight = self.column_weights.get(column_spec, 1.0)
            total_loss += weight * column_loss

        return total_loss

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
