# Pierre 14/02/21:
# Reorganization of the code:
# One class for the network
# One function for the training boom nahui
# We save the model every epoch during the training
# Dima 21/01/22:
# Cleanining and rewriting of the module

import os

import matplotlib.pyplot as plt

# Get common libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Get utility functions
from fullEncoder import nnUtils
from importData.epochs_management import inEpochsMask


# We generate a model with the functional Model interface in tensorflow
########### FULL NETWORK CLASS #####################
class LSTMandSpikeNetwork:
    def __init__(self, projectPath, params, deviceName="/device:CPU:0"):
        super(LSTMandSpikeNetwork, self).__init__()
        ### Main parameters here
        self.projectPath = projectPath
        self.params = params
        self.deviceName = deviceName
        # Folders
        self.folderResult = os.path.join(self.projectPath.resultsPath, "results")
        self.folderResultSleep = os.path.join(
            self.projectPath.resultsPath, "results_Sleep"
        )
        self.folderModels = os.path.join(self.projectPath.resultsPath, "models")
        if not os.path.isdir(self.folderResult):
            os.makedirs(self.folderResult)
        if not os.path.isdir(self.folderResultSleep):
            os.makedirs(self.folderResultSleep)
        if not os.path.isdir(self.folderModels):
            os.makedirs(self.folderModels)

        # The featDesc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.featDesc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            # target position: current value of the environmental correlate
            "pos": tf.io.FixedLenFeature([self.params.dimOutput], tf.float32),
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

        ### Description of layers here
        with tf.device(self.deviceName):
            if self.params.usingMixedPrecision:
                # If we use mixed precision, we need to specify the type of the inputs
                # We use float16 for the inputs to the spike nets
                self.inputsToSpikeNets = [
                    tf.keras.layers.Input(
                        shape=(self.params.nChannelsPerGroup[group], 32),
                        name="group" + str(group),
                        dtype=tf.float16,
                    )
                    for group in range(self.params.nGroups)
                ]
            else:
                self.inputsToSpikeNets = [
                    tf.keras.layers.Input(
                        shape=(self.params.nChannelsPerGroup[group], 32),
                        name="group" + str(group),
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
            if self.params.usingMixedPrecision:
                zeroForGather = tf.constant(
                    tf.zeros([1, self.params.nFeatures], dtype=tf.float16)
                )
            else:
                zeroForGather = tf.constant(tf.zeros([1, self.params.nFeatures]))
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
            self.dropoutLayer = tf.keras.layers.Dropout(params.dropoutCNN)

            # LSTMs
            self.lstmsNets = []
            for ilayer in range(params.lstmLayers):
                if ilayer == params.lstmLayers - 1:
                    self.lstmsNets.append(tf.keras.layers.LSTM(self.params.lstmSize))
                else:
                    self.lstmsNets.append(
                        tf.keras.layers.LSTM(
                            self.params.lstmSize, return_sequences=True
                        )
                    )

            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Pierre
            self.truePos = tf.keras.layers.Input(
                shape=(self.params.dimOutput), name="pos"
            )
            self.denseLoss1 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss3 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss4 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss5 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss2 = tf.keras.layers.Dense(
                1, activation=self.params.lossActivation, name="predicted_loss"
            )
            self.epsilon = tf.constant(10 ** (-8))
            # Outputs
            self.denseFeatureOutput = tf.keras.layers.Dense(
                self.params.dimOutput,
                activation=tf.keras.activations.hard_sigmoid,
                dtype=tf.float32,
                name="feature_output",
            )
            self.predAbsoluteLinearErrorLayer = tf.keras.layers.Dense(
                1, name="PredLoss"
            )

            # Gather the full model
            outputs = self.generate_model()
            # Build two models
            # One just described, with two objective functions corresponding
            # to both position and predicted losses
            self.model = self.compile_model(outputs)
            # In theory, the predicted loss could be not learning enough in the first network (optional)
            # Second only with loss corresponding to predicted loss
            self.predLossModel = self.compile_model(outputs, predLossOnly=True)

    def get_theweights(self, behaviorData, windowsizeMS, isPredLoss=0):
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowsizeMS), "predLoss" + "/cp.ckpt"
                )
            )
        else:
            self.model.load_weights(
                os.path.join(self.folderModels, str(windowsizeMS), "full" + "/cp.ckpt")
            )
        wdata = []
        for layer in self.model.layers:
            if hasattr(layer, "get_weights"):
                wdata.extend(layer.get_weights())
        # reshaped_w = [tf.reshape(w,(2,3,1,8)) if w.shape == (2,3,8,16) else w for w in wdata]
        # return reshaped_w
        return wdata

    def generate_model(self):
        # CNN plus dense on every group independently
        with tf.device(self.deviceName):
            allFeatures = []  # store the result of the CNN computation for each group
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[
                    group
                ]  # --> [NbKeptSpike,nbChannels,32] tensors
                x = self.spikeNets[group].apply(
                    x
                )  # outputs a [NbSpikeOfTheGroup,nFeatures=self.params.nFeatures(default 128)] tensor.
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
            allFeatures = tf.tuple(
                tensors=allFeatures
            )  # synchronizes the computation of all features (like a join)
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
            allFeatures = self.dropoutLayer(allFeatures)
            # LSTM
            for ilstm, lstmLayer in enumerate(self.lstmsNets):
                if ilstm == 0:
                    if len(self.lstmsNets) == 1:
                        output = lstmLayer(allFeatures, mask=mymask)
                    else:
                        outputSeq = lstmLayer(allFeatures, mask=mymask)
                        outputSeq = self.dropoutLayer(outputSeq)
                elif ilstm == len(self.lstmsNets) - 1:
                    output = lstmLayer(outputSeq, mask=mymask)
                else:
                    outputSeq = lstmLayer(outputSeq, mask=mymask)
                    outputSeq = self.dropoutLayer(outputSeq)
            ### Outputs
            myoutputPos = self.denseFeatureOutput(output)  # positions
            print("myoutputPos =", myoutputPos)
            outputPredLoss = self.denseLoss2(
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
            ### Losses
            tempPL = tf.losses.mean_squared_error(myoutputPos, self.truePos)[
                :, tf.newaxis
            ]
            posLoss = tf.identity(
                tf.math.log(tf.math.reduce_mean(tempPL), name="posLoss")
            )
            # remark: we need to also stop the gradient to progagate from posLoss to the network at the stage of
            # the computations for the loss of the loss predictor
            logposLoss = tf.math.log(
                tf.add(tempPL, self.epsilon)
            )  # minimizing difference between losposLoss and outpredloss
            preUncertaintyLoss = tf.math.reduce_mean(
                tf.losses.mean_squared_error(
                    outputPredLoss, tf.stop_gradient(logposLoss)
                )
            )
            uncertaintyLoss = tf.identity(
                tf.math.log(tf.add(preUncertaintyLoss, self.epsilon)),
                name="UncertaintyLoss",
            )

        return myoutputPos, outputPredLoss, posLoss, uncertaintyLoss

    def compile_model(self, outputs, modelName="FullModel.png", predLossOnly=False):
        # Initialize and plot the model
        model = tf.keras.Model(
            inputs=self.inputsToSpikeNets
            + self.indices
            + [self.truePos, self.inputGroups, self.zeroForGather],
            outputs=outputs,
        )
        tf.keras.utils.plot_model(
            model,
            to_file=(os.path.join(self.projectPath.resultsPath, modelName)),
            show_shapes=True,
        )

        # Compile the model
        if not predLossOnly:
            model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
                optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=self.params.learningRates[0]
                ),  # Initially compile with first lr.
                loss={
                    # tf_op_layer_ position loss (eucledian distance between predicted and real coordinates)
                    outputs[2].name.split("/Identity")[0]: lambda x, y: y,
                    # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss)
                    outputs[3].name.split("/Identity")[0]: lambda x, y: y,
                },
            )
            # Get internal names of losses
            self.outNames = [
                outputs[2].name.split("/Identity")[0],
                outputs[3].name.split("/Identity")[0],
            ]
        else:
            model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]),
                optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=self.params.learningRates[0]
                ),
                loss={
                    outputs[3].name.split("/Identity")[
                        0
                    ]: lambda x, y: y,  # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss)
                },
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
                os.path.join(self.projectPath.resultsPath, "FullModel_Cplusplus.png")
            ),
            show_shapes=True,
        )

    def train(
        self,
        behaviorData,
        onTheFlyCorrection=False,
        windowsizeMS=36,
        scheduler="decay",
        isPredLoss=True,
        earlyStop=False,
    ):
        """
        Train the network on the dataset.
        The training is done in two steps:
        - First we train the full model on the position loss and the uncertainty loss
        - Then we train the loss predictor model on the predicted loss

        Parameters
        ----------
        behaviorData : dict
        onTheFlyCorrection : bool
        windowsizeMS : int
        scheduler : str
        isPredLoss : bool
        earlyStop : bool

        Returns
        -------
        None
        """
        ### Create neccessary arrays
        epochMask = {}
        totMask = {}
        csvLogger = {}
        checkpointPath = {}
        # Manage folders
        if not os.path.isdir(os.path.join(self.folderModels, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderModels, str(windowsizeMS)))

        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowsizeMS), "full")
        ):
            os.makedirs(os.path.join(self.folderModels, str(windowsizeMS), "full"))
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0:
            if not os.path.isdir(
                os.path.join(self.folderModels, str(windowsizeMS), "predLoss")
            ):
                os.makedirs(
                    os.path.join(self.folderModels, str(windowsizeMS), "predLoss")
                )
        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
        ):
            os.makedirs(
                os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
            )
        # Manage callbacks
        csvLogger["full"] = tf.keras.callbacks.CSVLogger(
            os.path.join(self.folderModels, str(windowsizeMS), "full", "fullmodel.log")
        )
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            csvLogger["predLoss"] = tf.keras.callbacks.CSVLogger(
                os.path.join(
                    self.folderModels,
                    str(windowsizeMS),
                    "predLoss",
                    "predLossmodel.log",
                )
            )
        for key in csvLogger.keys():
            checkpointPath[key] = os.path.join(
                self.folderModels, str(windowsizeMS), key + "/cp.ckpt"
            )

        ## Get speed filter:
        speedMask = behaviorData["Times"]["speedFilter"]

        ## Get datasets
        ndataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        for key in totMask.keys():
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(totMask[key])), dtype=tf.int64),
                    tf.constant(totMask[key], dtype=tf.float64),
                ),
                default_value=0,
            )
            datasets[key] = ndataset.filter(
                lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0)
            )
            # This is just max normalization to use if the behavioral data have not been normalized yet
            # TODO: make a way to input 1D target, different 2D targets...
            # coherent with the --target arg from main
            if onTheFlyCorrection:
                maxPos = np.max(
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
            datasets[key] = datasets[key].filter(
                lambda x: tf.math.logical_not(
                    tf.math.is_nan(tf.math.reduce_sum(x["pos"]))
                )
            )
            datasets[key] = datasets[key].batch(
                self.params.batchSize, drop_remainder=True
            )
            datasets[key] = datasets[key].map(
                lambda *vals: nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )  # self.featDesc, *
            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            datasets[key] = datasets[key].map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = datasets[key].map(
                lambda vals: (
                    vals,
                    {
                        self.outNames[0]: tf.zeros(self.params.batchSize),
                        self.outNames[1]: tf.zeros(self.params.batchSize),
                    },
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
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
                hist = self.model.fit(
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
                    os.path.join(self.folderModels, str(windowsizeMS)),
                    fullModel=False,
                    valLosses=valLosses,
                )
                # Save model for C++ decoder
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels,
                #                     str(windowsizeMS), "savedModels","predLossModel"))
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
                    os.path.join(self.folderModels, str(windowsizeMS)),
                    valLosses=valLosses,
                )
                # Save model for C++ decoder
                # self.cplusplusModel.predict(datasets['train'])
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels, str(windowsizeMS), "savedModels","fullModel"))
            self.model.save(
                os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
            )

    def train_binary(
        self,
        behaviorData,
        onTheFlyCorrection=False,
        windowsizeMS=36,
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
        if not os.path.isdir(os.path.join(self.folderModels, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderModels, str(windowsizeMS)))

        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowsizeMS), "full")
        ):
            os.makedirs(os.path.join(self.folderModels, str(windowsizeMS), "full"))
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0:
            if not os.path.isdir(
                os.path.join(self.folderModels, str(windowsizeMS), "predLoss")
            ):
                os.makedirs(
                    os.path.join(self.folderModels, str(windowsizeMS), "predLoss")
                )
        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
        ):
            os.makedirs(
                os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
            )
        # Manage callbacks
        csvLogger["full"] = tf.keras.callbacks.CSVLogger(
            os.path.join(self.folderModels, str(windowsizeMS), "full", "fullmodel.log")
        )
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            csvLogger["predLoss"] = tf.keras.callbacks.CSVLogger(
                os.path.join(
                    self.folderModels,
                    str(windowsizeMS),
                    "predLoss",
                    "predLossmodel.log",
                )
            )
        for key in csvLogger.keys():
            checkpointPath[key] = os.path.join(
                self.folderModels, str(windowsizeMS), key + "/cp.ckpt"
            )

        ## Get speed filter:
        speedMask = behaviorData["Times"]["speedFilter"]

        ## Get datasets
        ndataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
            )
        )
        ndataset = ndataset.map(
            lambda *vals: nnUtils.parse_serialized_spike(self.featDesc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
        for key in totMask.keys():
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(totMask[key])), dtype=tf.int64),
                    tf.constant(totMask[key], dtype=tf.float64),
                ),
                default_value=0,
            )
            datasets[key] = ndataset.filter(
                lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0)
            )
            # This is just max normalization to use if the behavioral data have not been normalized yet
            if onTheFlyCorrection:
                maxPos = np.max(
                    behaviorData["pos"][
                        np.logical_not(np.isnan(np.sum(behaviorData["pos"], axis=1)))
                    ]
                )
                # WARNING: where is this "pos" index coming from ?
                # could be nowhere except for nnBehavior.mat - but never created
                # TODO: implement if target is binary in main
                posFeature = behaviorData["pos"] / maxPos
            else:
                posFeature = behaviorData["pos"]
            datasets[key] = datasets[key].map(nnUtils.import_true_pos(posFeature))
            datasets[key] = datasets[key].filter(
                lambda x: tf.math.logical_not(
                    tf.math.is_nan(tf.math.reduce_sum(x["pos"]))
                )
            )
            datasets[key] = datasets[key].batch(
                self.params.batchSize, drop_remainder=True
            )
            datasets[key] = datasets[key].map(
                lambda *vals: nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )  # self.featDesc, *
            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            datasets[key] = datasets[key].map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = datasets[key].map(
                lambda vals: (
                    vals,
                    {
                        self.outNames[0]: tf.zeros(self.params.batchSize),
                        self.outNames[1]: tf.zeros(self.params.batchSize),
                    },
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
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
                hist = self.model.fit(
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
                    os.path.join(self.folderModels, str(windowsizeMS)),
                    fullModel=False,
                    valLosses=valLosses,
                )
                # Save model for C++ decoder
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels,
                #                     str(windowsizeMS), "savedModels","predLossModel"))
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
                    os.path.join(self.folderModels, str(windowsizeMS)),
                    valLosses=valLosses,
                )
                # Save model for C++ decoder
                # self.cplusplusModel.predict(datasets['train'])
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels, str(windowsizeMS), "savedModels","fullModel"))
            self.model.save(
                os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
            )

    def test_binary(
        self,
        behaviorData,
        l_function=[],
        windowsizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowsizeMS)))
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowsizeMS), "predLoss" + "/cp.ckpt"
                )
            )
        else:
            self.model.load_weights(
                os.path.join(self.folderModels, str(windowsizeMS), "full" + "/cp.ckpt")
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
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        dataset = dataset.filter(
            lambda x: tf.math.greater(table.lookup(x["pos_index"]), 0)
        )  # Check previous commits for this line
        if onTheFlyCorrection:
            maxPos = np.max(
                behaviorData["pos"][
                    np.logical_not(np.isnan(np.sum(behaviorData["pos"], axis=1)))
                ]
            )
            posFeature = behaviorData["pos"] / maxPos
        else:
            posFeature = behaviorData["pos"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
        self.saveResults(testOutput, folderName=windowsizeMS)

        return testOutput

    def testControl(
        self,
        behaviorData,
        modelPath,
        l_function=[],
        windowsizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowsizeMS)))
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(modelPath, str(windowsizeMS), "predLoss" + "/cp.ckpt")
            )
        else:
            self.model.load_weights(
                os.path.join(modelPath, str(windowsizeMS), "full" + "/cp.ckpt")
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
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        dataset = dataset.filter(
            lambda x: tf.math.greater(table.lookup(x["pos_index"]), 0)
        )  # Check previous commits for this line
        if onTheFlyCorrection:
            maxPos = np.max(
                behaviorData["Positions"][
                    np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))
                ]
            )
            posFeature = behaviorData["Positions"] / maxPos
        else:
            posFeature = behaviorData["Positions"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
        self.saveResults(testOutput, folderName=200)

        return testOutput

    def test(
        self,
        behaviorData,
        l_function=[],
        windowsizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowsizeMS)))
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowsizeMS), "predLoss" + "/cp.ckpt"
                )
            )
        else:
            self.model.load_weights(
                os.path.join(self.folderModels, str(windowsizeMS), "full" + "/cp.ckpt")
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
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        dataset = dataset.filter(
            lambda x: tf.math.greater(table.lookup(x["pos_index"]), 0)
        )  # Check previous commits for this line
        if onTheFlyCorrection:
            maxPos = np.max(
                behaviorData["Positions"][
                    np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))
                ]
            )
            posFeature = behaviorData["Positions"] / maxPos
        else:
            posFeature = behaviorData["Positions"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
        self.saveResults(testOutput, folderName=windowsizeMS)

        return testOutput

    def testSleep(
        self,
        behaviorData,
        l_function=[],
        windowSizeDecoder=None,
        windowsizeMS=36,
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
        windowsizeMS : int
        isPredLoss : bool
        """
        # Create the folder
        if windowSizeDecoder is None:
            folderName = str(windowsizeMS)
            if not os.path.isdir(os.path.join(self.folderResultSleep, folderName)):
                os.makedirs(os.path.join(self.folderResultSleep, folderName))
        else:
            folderName = f"{str(windowsizeMS)}_by_{str(windowSizeDecoder)}"
            if not os.path.isdir(os.path.join(self.folderResultSleep, folderName)):
                os.makedirs(os.path.join(self.folderResultSleep, folderName))

        if windowSizeDecoder is None:
            windowSizeDecoder = windowsizeMS

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
                    ("datasetSleep" + "_stride" + str(windowsizeMS) + ".tfrec"),
                )
            )
            dataset = dataset.map(
                lambda *vals: nnUtils.parse_serialized_spike(self.featDesc, *vals),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            dataset = dataset.filter(
                lambda x: tf.math.logical_and(
                    tf.squeeze(tf.math.less_equal(x["time"], timeSleepStop)),
                    tf.squeeze(tf.math.greater_equal(x["time"], timeSleepStart)),
                )
            )
            dataset = dataset.batch(self.params.batchSize, drop_remainder=True)

            dataset = dataset.map(
                lambda *vals: nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            dataset = dataset.map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(
                lambda vals: (
                    vals,
                    {
                        "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                        "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                    },
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
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
        windowsizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowsizeMS)))
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowsizeMS), "predLoss" + "/cp.ckpt"
                )
            )
        else:
            self.model.load_weights(
                os.path.join(self.folderModels, str(windowsizeMS), "full" + "/cp.ckpt")
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
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        dataset = dataset.filter(
            lambda x: tf.math.greater(table.lookup(x["pos_index"]), 0)
        )  # Check previous commits for this line
        posFeature = behaviorData["Positions"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

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
        df.to_csv(os.path.join(self.folderResult, str(windowsizeMS), "aspikes.csv"))

        return aSpikes

    ########### FULL NETWORK CLASS #####################

    ########### HELPING LSTMandSpikeNetwork FUNCTIONS#####################
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
                        "group"
                        + str(group): tf.cast(
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
            ax[0].plot(trainLosses[:, 0])
            ax[0].set_title("position loss")
            ax[0].plot(valLosses[:, 0], label="validation position loss", c="orange")
            ax[1].plot(trainLosses[:, 1])
            ax[1].set_title("log loss prediction loss")
            ax[1].plot(valLosses[:, 1], label="validation log loss prediction loss")
            fig.legend()
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
            fig.savefig(
                os.path.join(folderModels, "predLoss", "predLossModelLosses.png")
            )

    def saveResults(self, test_output, folderName=36, sleep=False, sleepName="Sleep"):
        # Manage folders to save
        if sleep:
            folderToSave = os.path.join(
                self.folderResultSleep, str(folderName), sleepName
            )
            if not os.path.isdir(folderToSave):
                os.makedirs(folderToSave)
        else:
            folderToSave = os.path.join(self.folderResult, str(folderName))

        # predicted coordinates
        df = pd.DataFrame(test_output["featurePred"])
        df.to_csv(os.path.join(folderToSave, "featurePred.csv"))
        # Predicted loss
        df = pd.DataFrame(test_output["predLoss"])
        df.to_csv(os.path.join(folderToSave, "lossPred.csv"))
        # True coordinates
        if not sleep:
            df = pd.DataFrame(test_output["featureTrue"])
            df.to_csv(os.path.join(folderToSave, "featureTrue.csv"))
        # Times of prediction
        df = pd.DataFrame(test_output["times"])
        df.to_csv(os.path.join(folderToSave, "timeStepsPred.csv"))
        # Index of spikes relative to positions
        df = pd.DataFrame(test_output["posIndex"])
        df.to_csv(os.path.join(folderToSave, "posIndex.csv"))
        # Speed mask
        if not sleep:
            df = pd.DataFrame(test_output["speedMask"])
            df.to_csv(os.path.join(folderToSave, "speedMask.csv"))

        if "indexInDat" in test_output:
            df = pd.DataFrame(test_output["indexInDat"])
            df.to_csv(os.path.join(folderToSave, "indexInDat.csv"))
        if "projPred" in test_output:
            df = pd.DataFrame(test_output["projPred"])
            df.to_csv(os.path.join(folderToSave, "projPredFeature.csv"))
        if "projTruePos" in test_output:
            df = pd.DataFrame(test_output["projTruePos"])
            df.to_csv(os.path.join(folderToSave, "projTrueFeature.csv"))
        if "linearPred" in test_output:
            df = pd.DataFrame(test_output["linearPred"])
            df.to_csv(os.path.join(folderToSave, "linearPred.csv"))
        if "linearTrue" in test_output:
            df = pd.DataFrame(test_output["linearTrue"])
            df.to_csv(os.path.join(folderToSave, "linearTrue.csv"))


########### HELPING LSTMandSpikeNetwork FUNCTIONS#####################


class LSTMandSpikeNetwork_control:
    def __init__(self, projectPath, params, otherModelPath, deviceName="/device:CPU:0"):
        super(LSTMandSpikeNetwork_control, self).__init__()
        ### Main parameters here
        self.projectPath = projectPath
        self.params = params
        self.deviceName = deviceName
        # Folders
        self.folderResult = os.path.join(self.projectPath.resultsPath, "results")
        self.folderResultSleep = os.path.join(
            self.projectPath.resultsPath, "results_Sleep"
        )
        self.folderModels = otherModelPath
        if not os.path.isdir(self.folderResult):
            os.makedirs(self.folderResult)
        if not os.path.isdir(self.folderResultSleep):
            os.makedirs(self.folderResultSleep)
        if not os.path.isdir(self.folderModels):
            os.makedirs(self.folderModels)

        # The featDesc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.featDesc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            # target position: current value of the environmental correlate
            "pos": tf.io.FixedLenFeature([self.params.dimOutput], tf.float32),
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

        ### Description of layers here
        with tf.device(self.deviceName):
            if self.params.usingMixedPrecision:
                self.inputsToSpikeNets = [
                    tf.keras.layers.Input(
                        shape=(self.params.nChannelsPerGroup[group], 32),
                        name="group" + str(group),
                        dtype=tf.float16,
                    )
                    for group in range(self.params.nGroups)
                ]
            else:
                self.inputsToSpikeNets = [
                    tf.keras.layers.Input(
                        shape=(self.params.nChannelsPerGroup[group], 32),
                        name="group" + str(group),
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
            if self.params.usingMixedPrecision:
                zeroForGather = tf.constant(
                    tf.zeros([1, self.params.nFeatures], dtype=tf.float16)
                )
            else:
                zeroForGather = tf.constant(tf.zeros([1, self.params.nFeatures]))
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
            self.dropoutLayer = tf.keras.layers.Dropout(params.dropoutCNN)

            # LSTMs
            self.lstmsNets = []
            for ilayer in range(params.lstmLayers):
                if ilayer == params.lstmLayers - 1:
                    self.lstmsNets.append(tf.keras.layers.LSTM(self.params.lstmSize))
                else:
                    self.lstmsNets.append(
                        tf.keras.layers.LSTM(
                            self.params.lstmSize, return_sequences=True
                        )
                    )

            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Pierre
            self.truePos = tf.keras.layers.Input(
                shape=(self.params.dimOutput), name="pos"
            )
            self.denseLoss1 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss3 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss4 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss5 = tf.keras.layers.Dense(
                self.params.lstmSize, activation=tf.nn.relu
            )
            self.denseLoss2 = tf.keras.layers.Dense(
                1, activation=self.params.lossActivation, name="predicted_loss"
            )
            self.epsilon = tf.constant(10 ** (-8))
            # Outputs
            self.denseFeatureOutput = tf.keras.layers.Dense(
                self.params.dimOutput,
                activation=tf.keras.activations.hard_sigmoid,
                dtype=tf.float32,
                name="feature_output",
            )
            self.predAbsoluteLinearErrorLayer = tf.keras.layers.Dense(
                1, name="PredLoss"
            )

            # Gather the full model
            outputs = self.generate_model()
            # Build two models
            # One just desctibed, with two oibjective funcitons corresponding
            # to both position and predicted loss
            self.model = self.compile_model(outputs)
            # In theory, the predicted loss could be not learning enough in the first network (optional)
            # Second only with loss corresponding to predicted loss
            self.predLossModel = self.compile_model(outputs, predLossOnly=True)

    def generate_model(self):
        # CNN plus dense on every group independently
        with tf.device(self.deviceName):
            allFeatures = []  # store the result of the CNN computation for each group
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[
                    group
                ]  # --> [NbKeptSpike,nbChannels,32] tensors
                x = self.spikeNets[group].apply(
                    x
                )  # outputs a [NbSpikeOfTheGroup,nFeatures=self.params.nFeatures(default 128)] tensor.
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
            allFeatures = tf.tuple(
                tensors=allFeatures
            )  # synchronizes the computation of all features (like a join)
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
            allFeatures = self.dropoutLayer(allFeatures)
            # LSTM
            for ilstm, lstmLayer in enumerate(self.lstmsNets):
                if ilstm == 0:
                    if len(self.lstmsNets) == 1:
                        output = lstmLayer(allFeatures, mask=mymask)
                    else:
                        outputSeq = lstmLayer(allFeatures, mask=mymask)
                        outputSeq = self.dropoutLayer(outputSeq)
                elif ilstm == len(self.lstmsNets) - 1:
                    output = lstmLayer(outputSeq, mask=mymask)
                else:
                    outputSeq = lstmLayer(outputSeq, mask=mymask)
                    outputSeq = self.dropoutLayer(outputSeq)
            ### Outputs
            myoutputPos = self.denseFeatureOutput(output)  # positions
            print("myoutputPos =", myoutputPos)
            outputPredLoss = self.denseLoss2(
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
            ### Losses
            tempPL = tf.losses.mean_squared_error(myoutputPos, self.truePos)[
                :, tf.newaxis
            ]
            posLoss = tf.identity(
                tf.math.log(tf.math.reduce_mean(tempPL), name="posLoss")
            )
            # remark: we need to also stop the gradient to progagate from posLoss to the network at the stage of
            # the computations for the loss of the loss predictor
            logposLoss = tf.math.log(
                tf.add(tempPL, self.epsilon)
            )  # minimizing difference between losposLoss and outpredloss
            preUncertaintyLoss = tf.math.reduce_mean(
                tf.losses.mean_squared_error(
                    outputPredLoss, tf.stop_gradient(logposLoss)
                )
            )
            uncertaintyLoss = tf.identity(
                tf.math.log(tf.add(preUncertaintyLoss, self.epsilon)),
                name="UncertaintyLoss",
            )

        return myoutputPos, outputPredLoss, posLoss, uncertaintyLoss

    def compile_model(self, outputs, modelName="FullModel.png", predLossOnly=False):
        # Initialize and plot the model
        model = tf.keras.Model(
            inputs=self.inputsToSpikeNets
            + self.indices
            + [self.truePos, self.inputGroups, self.zeroForGather],
            outputs=outputs,
        )
        tf.keras.utils.plot_model(
            model,
            to_file=(os.path.join(self.projectPath.resultsPath, modelName)),
            show_shapes=True,
        )

        # Compile the model
        if not predLossOnly:
            model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
                optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=self.params.learningRates[0]
                ),  # Initially compile with first lr.
                loss={
                    # tf_op_layer_ position loss (eucledian distance between predicted and real coordinates)
                    outputs[2].name.split("/Identity")[0]: lambda x, y: y,
                    # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss)
                    outputs[3].name.split("/Identity")[0]: lambda x, y: y,
                },
            )
            # Get internal names of losses
            self.outNames = [
                outputs[2].name.split("/Identity")[0],
                outputs[3].name.split("/Identity")[0],
            ]
        else:
            model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]),
                optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=self.params.learningRates[0]
                ),
                loss={
                    outputs[3].name.split("/Identity")[
                        0
                    ]: lambda x, y: y,  # tf_op_layer_ uncertainty loss (MSE between uncertainty and posLoss)
                },
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
                os.path.join(self.projectPath.resultsPath, "FullModel_Cplusplus.png")
            ),
            show_shapes=True,
        )

    def train(
        self,
        behaviorData,
        onTheFlyCorrection=False,
        windowsizeMS=36,
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
        if not os.path.isdir(os.path.join(self.folderModels, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderModels, str(windowsizeMS)))

        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowsizeMS), "full")
        ):
            os.makedirs(os.path.join(self.folderModels, str(windowsizeMS), "full"))
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0:
            if not os.path.isdir(
                os.path.join(self.folderModels, str(windowsizeMS), "predLoss")
            ):
                os.makedirs(
                    os.path.join(self.folderModels, str(windowsizeMS), "predLoss")
                )
        if not os.path.isdir(
            os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
        ):
            os.makedirs(
                os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
            )
        # Manage callbacks
        csvLogger["full"] = tf.keras.callbacks.CSVLogger(
            os.path.join(self.folderModels, str(windowsizeMS), "full", "fullmodel.log")
        )
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            csvLogger["predLoss"] = tf.keras.callbacks.CSVLogger(
                os.path.join(
                    self.folderModels,
                    str(windowsizeMS),
                    "predLoss",
                    "predLossmodel.log",
                )
            )
        for key in csvLogger.keys():
            checkpointPath[key] = os.path.join(
                self.folderModels, str(windowsizeMS), key + "/cp.ckpt"
            )

        ## Get speed filter:
        speedMask = behaviorData["Times"]["speedFilter"]

        ## Get datasets
        ndataset = tf.data.TFRecordDataset(
            os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
            )
        )
        ndataset = ndataset.map(
            lambda *vals: nnUtils.parse_serialized_spike(self.featDesc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
        for key in totMask.keys():
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(totMask[key])), dtype=tf.int64),
                    tf.constant(totMask[key], dtype=tf.float64),
                ),
                default_value=0,
            )
            datasets[key] = ndataset.filter(
                lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0)
            )
            # This is just max normalization to use if the behavioral data have not been normalized yet
            if onTheFlyCorrection:
                maxPos = np.max(
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
            datasets[key] = datasets[key].filter(
                lambda x: tf.math.logical_not(
                    tf.math.is_nan(tf.math.reduce_sum(x["pos"]))
                )
            )
            datasets[key] = datasets[key].batch(
                self.params.batchSize, drop_remainder=True
            )
            datasets[key] = datasets[key].map(
                lambda *vals: nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )  # self.featDesc, *
            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            datasets[key] = datasets[key].map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets[key] = datasets[key].map(
                lambda vals: (
                    vals,
                    {
                        self.outNames[0]: tf.zeros(self.params.batchSize),
                        self.outNames[1]: tf.zeros(self.params.batchSize),
                    },
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
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
                hist = self.model.fit(
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
                    os.path.join(self.folderModels, str(windowsizeMS)),
                    fullModel=False,
                    valLosses=valLosses,
                )
                # Save model for C++ decoder
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels,
                #                     str(windowsizeMS), "savedModels","predLossModel"))
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
                    os.path.join(self.folderModels, str(windowsizeMS)),
                    valLosses=valLosses,
                )
                # Save model for C++ decoder
                # self.cplusplusModel.predict(datasets['train'])
                # print("saving full model in savedmodel format, for c++")
                # tf.saved_model.save(self.cplusplusModel, os.path.join(self.folderModels, str(windowsizeMS), "savedModels","fullModel"))
            self.model.save(
                os.path.join(self.folderModels, str(windowsizeMS), "savedModels")
            )

    def testControl(
        self,
        behaviorData,
        wdata,
        l_function=[],
        windowsizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowsizeMS)))
        # Loading the weights
        self.model.set_weights(wdata)

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
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        dataset = dataset.filter(
            lambda x: tf.math.greater(table.lookup(x["pos_index"]), 0)
        )  # Check previous commits for this line
        if onTheFlyCorrection:
            maxPos = np.max(
                behaviorData["Positions"][
                    np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))
                ]
            )
            posFeature = behaviorData["Positions"] / maxPos
        else:
            posFeature = behaviorData["Positions"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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
        self.saveResults(testOutput, folderName=200)

        return testOutput

    def testSleep(
        self,
        behaviorData,
        l_function=[],
        windowSizeDecoder=None,
        windowsizeMS=36,
        isPredLoss=False,
    ):
        # Create the folder
        if windowSizeDecoder is None:
            folderName = str(windowsizeMS)
            if not os.path.isdir(os.path.join(self.folderResultSleep, folderName)):
                os.makedirs(os.path.join(self.folderResultSleep, folderName))
        else:
            folderName = f"{str(windowsizeMS)}_by_{str(windowSizeDecoder)}"
            if not os.path.isdir(os.path.join(self.folderResultSleep, folderName)):
                os.makedirs(os.path.join(self.folderResultSleep, folderName))

        if windowSizeDecoder is None:
            windowSizeDecoder = windowsizeMS

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
                    ("datasetSleep" + "_stride" + str(windowsizeMS) + ".tfrec"),
                )
            )
            dataset = dataset.map(
                lambda *vals: nnUtils.parse_serialized_spike(self.featDesc, *vals),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            dataset = dataset.filter(
                lambda x: tf.math.logical_and(
                    tf.squeeze(tf.math.less_equal(x["time"], timeSleepStop)),
                    tf.squeeze(tf.math.greater_equal(x["time"], timeSleepStart)),
                )
            )
            dataset = dataset.batch(self.params.batchSize, drop_remainder=True)

            dataset = dataset.map(
                lambda *vals: nnUtils.parse_serialized_sequence(
                    self.params, *vals, batched=True
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            dataset = dataset.map(
                self.create_indices, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(
                lambda vals: (
                    vals,
                    {
                        "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                        "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                    },
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
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
        windowsizeMS=36,
        useSpeedFilter=False,
        useTrain=False,
        isPredLoss=False,
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowsizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowsizeMS)))
        # Loading the weights
        print("Loading the weights of the trained network")
        if len(behaviorData["Times"]["lossPredSetEpochs"]) > 0 and isPredLoss:
            self.model.load_weights(
                os.path.join(
                    self.folderModels, str(windowsizeMS), "predLoss" + "/cp.ckpt"
                )
            )
        else:
            self.model.load_weights(
                os.path.join(self.folderModels, str(windowsizeMS), "full" + "/cp.ckpt")
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
                ("dataset" + "_stride" + str(windowsizeMS) + ".tfrec"),
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
        dataset = dataset.filter(
            lambda x: tf.math.greater(table.lookup(x["pos_index"]), 0)
        )  # Check previous commits for this line
        posFeature = behaviorData["Positions"]
        dataset = dataset.map(nnUtils.import_true_pos(posFeature))
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(
            self.params.batchSize, drop_remainder=True
        )  # remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_posLoss": tf.zeros(self.params.batchSize),
                    "tf_op_layer_UncertaintyLoss": tf.zeros(self.params.batchSize),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

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
        df.to_csv(os.path.join(self.folderResult, str(windowsizeMS), "aspikes.csv"))

        return aSpikes

    ########### FULL NETWORK CLASS #####################

    ########### HELPING LSTMandSpikeNetwork FUNCTIONS#####################
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
                        "group"
                        + str(group): tf.cast(
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
            ax[0].plot(trainLosses[:, 0])
            ax[0].set_title("position loss")
            ax[0].plot(valLosses[:, 0], label="validation position loss", c="orange")
            ax[1].plot(trainLosses[:, 1])
            ax[1].set_title("log loss prediction loss")
            ax[1].plot(valLosses[:, 1], label="validation log loss prediction loss")
            fig.legend()
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
            fig.savefig(
                os.path.join(folderModels, "predLoss", "predLossModelLosses.png")
            )

    def saveResults(self, test_output, folderName=36, sleep=False, sleepName="Sleep"):
        # Manage folders to save
        if sleep:
            folderToSave = os.path.join(
                self.folderResultSleep, str(folderName), sleepName
            )
            if not os.path.isdir(folderToSave):
                os.makedirs(folderToSave)
        else:
            folderToSave = os.path.join(self.folderResult, str(folderName))

        # predicted coordinates
        df = pd.DataFrame(test_output["featurePred"])
        df.to_csv(os.path.join(folderToSave, "featurePred.csv"))
        # Predicted loss
        df = pd.DataFrame(test_output["predLoss"])
        df.to_csv(os.path.join(folderToSave, "lossPred.csv"))
        # True coordinates
        if not sleep:
            df = pd.DataFrame(test_output["featureTrue"])
            df.to_csv(os.path.join(folderToSave, "featureTrue.csv"))
        # Times of prediction
        df = pd.DataFrame(test_output["times"])
        df.to_csv(os.path.join(folderToSave, "timeStepsPred.csv"))
        # Index of spikes relative to positions
        df = pd.DataFrame(test_output["posIndex"])
        df.to_csv(os.path.join(folderToSave, "posIndex.csv"))
        # Speed mask
        if not sleep:
            df = pd.DataFrame(test_output["speedMask"])
            df.to_csv(os.path.join(folderToSave, "speedMask.csv"))

        if "indexInDat" in test_output:
            df = pd.DataFrame(test_output["indexInDat"])
            df.to_csv(os.path.join(folderToSave, "indexInDat.csv"))
        if "projPred" in test_output:
            df = pd.DataFrame(test_output["projPred"])
            df.to_csv(os.path.join(folderToSave, "projPredFeature.csv"))
        if "projTruePos" in test_output:
            df = pd.DataFrame(test_output["projTruePos"])
            df.to_csv(os.path.join(folderToSave, "projTrueFeature.csv"))
        if "linearPred" in test_output:
            df = pd.DataFrame(test_output["linearPred"])
            df.to_csv(os.path.join(folderToSave, "linearPred.csv"))
        if "linearTrue" in test_output:
            df = pd.DataFrame(test_output["linearTrue"])
            df.to_csv(os.path.join(folderToSave, "linearTrue.csv"))


########### HELPING LSTMandSpikeNetwork FUNCTIONS#####################
