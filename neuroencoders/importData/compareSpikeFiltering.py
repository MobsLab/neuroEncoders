# Load libs
import os

import numpy as np
import pandas as pd
import pykeops
import tensorflow as tf
from tqdm import tqdm

from neuroencoders.fullEncoder import nnUtils
from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params

## Different strategies are used for spike filtering in the case of the NN and of spike sorting.
# To make sure we end up with a fair comparison between the bayesian algorithm
# and the NN, one needs to give the same spike input to either decoding algorithm,
#
# This is done to have clear population-spike trains for the NN as a file:
# we translate the times of spikes found by the manual spike sorting algo
# (including the noise cluster) to a dataset for the NN

# Clarification from Dima: what is not done really is a bayesian decoder without noise

# TODO: another important idea: get detected spikes from the NN and use them to
#  do bayesian decoding. This would be a fair comparison of the two methods.
#  And, inversly, to input spike sorting results to the NN decoder.
#  However, none of this is done here.

pykeops.set_verbose(False)


class WaveFormComparator:
    def __init__(
        self,
        projectPath,
        params,
        behavior_data,
        windowSizeMS=36,
        useTrain=True,
        sleepName=[],
        **kwargs,
    ):  # todo allow for speed filtering
        self.projectPath = projectPath
        self.params = params
        self.behavior_data = behavior_data
        self.useTrain = useTrain
        self.sleepName = sleepName
        self.windowSizeMS = windowSizeMS
        phase = kwargs.get("phase", None)
        assert phase == params.phase, (
            "The phase of the WaveFormComparator must be the same as the one of the params"
        )
        self.phase = phase
        self.suffix = f"_{phase}" if phase is not None else ""
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            "pos": tf.io.FixedLenFeature(
                [self.params.dimOutput], tf.float32
            ),  # target position: current value of the environmental correlate
            "length": tf.io.FixedLenFeature(
                [], tf.int64
            ),  # number of spike sequence gathered in the window
            "groups": tf.io.VarLenFeature(
                tf.int64
            ),  # the index of the groups having spike sequences in the window
            "time": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64),
        }
        for g in range(self.params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})

        # Manage folder
        self.alignedDataPath = os.path.join(
            self.projectPath.dataPath, "aligned", str(windowSizeMS)
        )
        if not os.path.isdir(self.alignedDataPath):
            os.makedirs(self.alignedDataPath)

        # Manage epochs
        if self.useTrain:
            epochMask = ep.inEpochsMask(
                behavior_data["positionTime"][:, 0],
                behavior_data["Times"]["trainEpochs"],
            )
        else:
            if bool(self.sleepName) and not self.useTrain:
                idsleep = behavior_data["Times"]["sleepNames"].index(self.sleepName)
                timeSleepStart = behavior_data["Times"]["sleepEpochs"][2 * idsleep][0]
                timeSleepStop = behavior_data["Times"]["sleepEpochs"][2 * idsleep + 1][
                    0
                ]
            else:
                epochMask = ep.inEpochsMask(
                    behavior_data["positionTime"][:, 0],
                    behavior_data["Times"]["testEpochs"],
                )

        # Load dataset
        if bool(self.sleepName):
            dataset = tf.data.TFRecordDataset(
                os.path.join(
                    self.projectPath.dataPath,
                    ("datasetSleep" + "_stride" + str(windowSizeMS) + ".tfrec"),
                )
            )
        else:
            dataset = tf.data.TFRecordDataset(
                os.path.join(
                    self.projectPath.dataPath,
                    ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
                )
            )
        self.dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_spike(self.feat_desc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if bool(self.sleepName):
            self.dataset = self.dataset.filter(
                lambda x: tf.math.logical_and(
                    tf.math.less_equal(x["time"], timeSleepStop),
                    tf.math.greater_equal(x["time"], timeSleepStart),
                )
            )
        else:
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(epochMask)), dtype=tf.int64),
                    tf.constant(epochMask, dtype=tf.float64),
                ),
                default_value=0,
            )
            self.dataset = self.dataset.filter(
                lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0)
            )
            self.dataset = self.dataset.map(
                nnUtils.import_true_pos(behavior_data["Positions"])
            )
            self.dataset = self.dataset.filter(
                lambda x: tf.math.logical_not(
                    tf.math.is_nan(tf.math.reduce_sum(x["pos"]))
                )
            )

        self.dataset = self.dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=False
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    def save_alignment_tools(
        self, trainerBayes, linearizationFunction, windowSizeMS=36
    ):
        # Manage folder
        if self.useTrain:
            foldertosave = os.path.join(self.alignedDataPath, "train")
        else:
            if bool(self.sleepName) and not self.useTrain:
                foldertosave = os.path.join(self.alignedDataPath, self.sleepName)
            else:
                foldertosave = os.path.join(self.alignedDataPath, "test")
        if not os.path.isdir(foldertosave):
            os.makedirs(foldertosave)
        if os.path.isfile(
            os.path.join(foldertosave, f"spikeMat_times_window{self.suffix}.csv")
        ):
            return

        # Get data
        self.get_data()
        if not hasattr(trainerBayes, "linearPreferredPos"):
            _ = trainerBayes.train_order_by_pos(
                self.behavior_data, l_function=linearizationFunction
            )
        # gather all windows in the tensorflow dataset
        inputNN = self.get_NNdataset_spikepos()

        ### Mapping spikes from automatic ANN pipeline to windows
        lenInputNN = []  # Number of spikes per window
        meanTimeWindow = []  # Mean time of spikes in the window
        startTimeWindow = []  # Start of windows
        startTimeWindowInSamples = []  # Start of windows in samples
        for _, startTime in tqdm(enumerate(inputNN)):
            if len(startTime) > 0:
                startTimeWindow += [startTime[0] / self.samplingRate]
                startTimeWindowInSamples += [startTime[0]]
            else:
                startTimeWindow += [
                    np.nan
                ]  # we make sure these windows are never selected
            lenInputNN += [len(startTime)]
            timeWindowInSec = [sample / self.samplingRate for sample in startTime]
            meanTimeWindow += [np.mean(timeWindowInSec)]
        lenInputNN = np.array(lenInputNN)
        meanTimeWindow = np.array(meanTimeWindow)
        startTimeWindow = np.array(startTimeWindow)
        startTimeWindowInSamples = np.array(startTimeWindowInSamples)
        # Get rid of empty windows
        goodStartTimeWindowInSamples = startTimeWindowInSamples[
            np.logical_not(np.isnan(startTimeWindowInSamples))
        ]
        stopTimeWindowInSamples = goodStartTimeWindowInSamples + int(
            windowSizeMS / 1000 * self.samplingRate
        )

        ### Mapping spike sorted spike times to windows
        spikeMat_times_window = np.zeros([trainerBayes.spikeMatTimes.shape[0], 2])
        spikeMat_times_window[:, 0] = trainerBayes.spikeMatTimes[:, 0]
        spikeTime_lazy = pykeops.numpy.LazyTensor(
            trainerBayes.spikeMatTimes[:, 0][:, None] * self.samplingRate, axis=0
        )
        startTimeWindow_lazy = pykeops.numpy.Vj(
            goodStartTimeWindowInSamples[:, None].astype(dtype=np.float64)
        )
        stopTimeWindow_lazy = pykeops.numpy.Vj(
            stopTimeWindowInSamples[:, None].astype(dtype=np.float64)
        )
        ans = (spikeTime_lazy - startTimeWindow_lazy).relu().sign() * (
            (stopTimeWindow_lazy - spikeTime_lazy).relu().sign()
        )
        ans2 = ans.max_argmax_reduction(dim=1)
        ans2[1][np.equal(ans2[0], 0)] = -1
        spikeMat_times_window[:, 1] = ans2[1][:, 0]
        # for the pop vector we add one label for the noisy cluster
        spikeMat_window_popVector = np.zeros(
            [len(inputNN), trainerBayes.spikeMatLabels.shape[1] + 1]
        )
        for idSpike, window in tqdm(enumerate(spikeMat_times_window[:, 1])):
            if window != -1:
                cluster = np.where(
                    np.equal(trainerBayes.spikeMatLabels[idSpike, :], 1)
                )[0]
                if len(cluster) > 0:
                    spikeMat_window_popVector[int(window), 1 + cluster[0]] += 1
                else:
                    spikeMat_window_popVector[int(window), 0] += 1  # noisy cluster

        ### Saving
        df = pd.DataFrame(spikeMat_window_popVector)
        df.to_csv(
            os.path.join(foldertosave, f"spikeMat_window_popVector{self.suffix}.csv")
        )
        df = pd.DataFrame(meanTimeWindow)
        df.to_csv(os.path.join(foldertosave, f"meanTimeWindow{self.suffix}.csv"))
        df = pd.DataFrame(spikeMat_times_window)
        df.to_csv(os.path.join(foldertosave, f"spikeMat_times_window{self.suffix}.csv"))
        df = pd.DataFrame(startTimeWindow)
        df.to_csv(os.path.join(foldertosave, f"startTimeWindow{self.suffix}.csv"))
        df = pd.DataFrame(lenInputNN)
        df.to_csv(os.path.join(foldertosave, f"lenInputNN{self.suffix}.csv"))

    def get_NNdataset_spikepos(self):
        resData = self.dataset.map(lambda vals: vals["indexInDat"])
        return list(resData.as_numpy_iterator())

    def get_data(self):
        # Get names
        filPath = self.projectPath.fil
        datPath = self.projectPath.dat
        xmlPath = self.projectPath.xml
        # Map the data
        _, self.samplingRate, nChannels = get_params(xmlPath)
        self.number_timeSteps = os.stat(datPath).st_size // (2 * nChannels)
        self.memmapData = np.memmap(
            datPath, dtype=np.int16, mode="r", shape=(self.number_timeSteps, nChannels)
        )
        self.memmapFil = np.memmap(
            filPath, dtype=np.int16, mode="r", shape=(self.number_timeSteps, nChannels)
        )
