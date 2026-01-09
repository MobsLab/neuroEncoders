# Pierre 14/02/21:
# Reorganization of the code:
# One class for the network
# One function for the training boom nahui
# We save the model every epoch during the training
# Dima 21/01/22:
# Cleanining and rewriting of the module

import os

# Get common libraries
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors, not warnings
import tensorflow as tf

# Get utility functions
from neuroencoders.fullEncoder import nnUtils
from neuroencoders.utils.global_classes import Params, Project

# We generate a model with the functional Model interface in tensorflow
########### FULL NETWORK CLASS #####################


class Decoder:
    def __init__(
        self,
        projectPath: Project,
        params: Params,
        windowSizeMS: int = 36,
        deviceName: int = "/device:CPU:0",
    ):
        # Main parameters here
        self.projectPath = projectPath
        self.params = params
        self.deviceName = deviceName
        # Folders
        self.folderResult = os.path.join(
            self.projectPath.resultsPath, "results_decoder"
        )
        self.folderResultSleep = os.path.join(
            self.projectPath.resultsPath, "results_decoderSleep"
        )
        if not os.path.isdir(self.folderResult):
            os.makedirs(self.folderResult)
        if not os.path.isdir(self.folderResultSleep):
            os.makedirs(self.folderResultSleep)

        # Load model
        with tf.device(self.deviceName):
            self.model = tf.keras.models.load_model(
                os.path.join(
                    self.projectPath.graph,
                    str(windowSizeMS),
                    "savedModels",
                    "fullModel.keras",
                )
            )
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
            # the exact time-steps of each spike measured in the various groups. Question: should the time not be a VarLenFeature??
            "time": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64),
        }  # sample of the spike
        for g in range(self.params.nGroups):
            # the voltage values (discretized over 32 time bins) of each channel (4 most of the time)
            # of each spike of a given group in the window
            self.featDesc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})
        # Loss obtained during training
        self.trainLosses = {}

    def train(self):
        pass

    def test(
        self, behaviorData, l_function=[], windowSizeMS=36, onTheFlyCorrection=False
    ):
        # Create the folder
        if not os.path.isdir(os.path.join(self.folderResult, str(windowSizeMS))):
            os.makedirs(os.path.join(self.folderResult, str(windowSizeMS)))

        # Manage the behavior
        tot_mask = np.isfinite(np.sum(behaviorData["Positions"][0:-1], axis=1))

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
                tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                tf.constant(tot_mask, dtype=tf.float64),
            ),
            default_value=0,
        )
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
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
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        # dataset = dataset.batch(1, drop_remainder=True) #remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=False
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.create_indices, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.map(lambda vals: ( vals, {"tf_op_layer_posLoss": tf.zeros(1),
        #                                             "tf_op_layer_UncertaintyLoss": tf.zeros(1)}),
        #                         num_parallel_calls=tf.data.AUTOTUNE)
        print("INFERRING")
        outputTest = self.model.predict(dataset, verbose=1)

        # Post-inferring management
        print("gathering true feature")
        datasetPos = dataset.map(
            lambda x: x["pos"], num_parallel_calls=tf.data.AUTOTUNE
        )
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        featureTrue = np.squeeze(
            np.reshape(
                fullFeatureTrue, [outputTest[0].shape[0], outputTest[0].shape[-1]]
            )
        )
        print("gathering exact time of spikes")
        datasetTimes = dataset.map(
            lambda x: x["time"], num_parallel_calls=tf.data.AUTOTUNE
        )
        times = list(datasetTimes.as_numpy_iterator())
        times = np.reshape(times, [outputTest[0].shape[0]])
        print("gathering indices of spikes relative to coordinates")
        datasetPosIndex = dataset.map(
            lambda x: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE
        )
        posIndex = list(datasetPosIndex.as_numpy_iterator())
        posIndex = np.ravel(np.array(posIndex))

        testOutput = {
            "featurePred": outputTest[0],
            "featureTrue": featureTrue,
            "times": times,
            "predLoss": outputTest[1],
            "posIndex": posIndex,
        }

        if l_function:
            projPredPos, linearPred = l_function(outputTest[0][:, :2])
            projTruePos, linearTrue = l_function(featureTrue)
            testOutput["projPred"] = projPredPos
            testOutput["projTruePos"] = projTruePos
            testOutput["linearPred"] = linearPred
            testOutput["linearTrue"] = linearTrue

        # Save the results
        self.saveResults(testOutput, windowSizeMS=windowSizeMS)

        return testOutput

    ########### FULL NETWORK CLASS #####################

    ########### HELPING LSTMandSpikeNetwork FUNCTIONS#####################

    def fix_linearizer(self, mazePoints, tsProj):
        # For the linearization we define two fixed inputs:
        self.mazePoints_tensor = tf.convert_to_tensor(
            mazePoints[None, :], dtype=tf.float32
        )
        self.mazePoints = tf.keras.layers.Input(
            tensor=self.mazePoints_tensor, name="mazePoints"
        )
        self.tsProj_tensor = tf.convert_to_tensor(tsProj[None, :], dtype=tf.float32)
        self.tsProj = tf.keras.layers.Input(tensor=self.tsProj_tensor, name="tsProj")

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
                vals.update({"tsProj": self.tsProj_tensor})

        if self.params.usingMixedPrecision:
            vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})
        return vals

    def saveResults(self, testOutput, windowSizeMS=36, sleep=False, sleepName="Sleep"):
        # Manage folders to save
        if sleep:
            folderToSave = os.path.join(
                self.folderResultSleep, str(windowSizeMS), sleepName
            )
            if not os.path.isdir(folderToSave):
                os.makedirs(folderToSave)
        else:
            folderToSave = os.path.join(self.folderResult, str(windowSizeMS))
        # predicted coordinates
        df = pd.DataFrame(testOutput["featurePred"])
        df.to_csv(os.path.join(folderToSave, "featurePred.csv"))
        # Predicted loss
        df = pd.DataFrame(testOutput["predLoss"])
        df.to_csv(os.path.join(folderToSave, "lossPred.csv"))
        # True coordinates
        if not sleep:
            df = pd.DataFrame(testOutput["featureTrue"])
            df.to_csv(os.path.join(folderToSave, "featureTrue.csv"))
        # Times of prediction
        df = pd.DataFrame(testOutput["times"])
        df.to_csv(os.path.join(folderToSave, "timeStepsPred.csv"))
        # Index of spikes relative to positions
        df = pd.DataFrame(testOutput["posIndex"])
        df.to_csv(os.path.join(folderToSave, "posIndex.csv"))
        # Speed mask
        if not sleep and "speedMask" in testOutput.keys():
            df = pd.DataFrame(testOutput["speedMask"])
            df.to_csv(os.path.join(folderToSave, "speedMask.csv"))
        if "indexInDat" in testOutput:
            df = pd.DataFrame(testOutput["indexInDat"])
            df.to_csv(os.path.join(folderToSave, "indexInDat.csv"))
        if "projPred" in testOutput:
            df = pd.DataFrame(testOutput["projPred"])
            df.to_csv(os.path.join(folderToSave, "projPredFeature.csv"))
        if "projTruePos" in testOutput:
            df = pd.DataFrame(testOutput["projTruePos"])
            df.to_csv(os.path.join(folderToSave, "projTrueFeature.csv"))
        if "linearPred" in testOutput:
            df = pd.DataFrame(testOutput["linearPred"])
            df.to_csv(os.path.join(folderToSave, "linearPred.csv"))
        if "linearTrue" in testOutput:
            df = pd.DataFrame(testOutput["linearTrue"])
            df.to_csv(os.path.join(folderToSave, "linearTrue.csv"))


########### HELPING LSTMandSpikeNetwork FUNCTIONS#####################
