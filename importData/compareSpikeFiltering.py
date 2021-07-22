## Different strategies are used for spike filtering
# in the case of the NN and of spike sorting.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp

from importData.ImportClusters import getBehavior
from importData.rawDataParser import  inEpochsMask
from importData import rawDataParser
from tqdm import tqdm

import pykeops

# To make sure we end up with a fair comparison between the bayesian algorithm
# and the NN, here we look at the waveforms found (including the noise cluster)
# by the algorithm for spike sorting and use them to generate a dataset for the NN
# goal: prove that by using more information (no human biased intervention), the NN
# can extract more information
# We also compare the shape of waveforms extracted by the different filtering strategies.

class waveFormComparator():
    def __init__(self, projectPath, params,behavior_data,useTrain=True,useSleep=False):
        self.projectPath = projectPath
        self.params = params
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos_index" : tf.io.FixedLenFeature([], tf.int64),
            "pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), #target position: current value of the environmental correlate
            "length": tf.io.FixedLenFeature([], tf.int64), #number of spike sequence gathered in the window
            "groups": tf.io.VarLenFeature(tf.int64), # the index of the groups having spike sequences in the window
            "time": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64)}
        for g in range(self.params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})

        self.useTrain = useTrain
        self.useSleep = useSleep

        if useTrain:
            dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
            self.dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                                       num_parallel_calls=tf.data.AUTOTUNE)
            epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['trainEpochs'])
            tot_mask = epochMask
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                    tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
            self.dataset = self.dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
            maxPos = np.max(
                behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            self.dataset = self.dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
            self.dataset = self.dataset.filter(
                lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))
        else:
            if useSleep:
                timeSleepStart = behavior_data["Times"]["sleepEpochs"][0]
                timeSleepStop = behavior_data["Times"]["sleepEpochs"][1] #todo: change
                self.dataset = tf.data.TFRecordDataset(self.projectPath.tfrecSleep)
                self.dataset = self.dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                                      num_parallel_calls=tf.data.AUTOTUNE)
                self.dataset = self.dataset.filter(lambda x: tf.math.logical_and(tf.math.less_equal(x["time"], timeSleepStop),
                                                                       tf.math.greater_equal(x["time"], timeSleepStart)))
            else:
                dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
                self.dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                                           num_parallel_calls=tf.data.AUTOTUNE)
                epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['testEpochs'])
                tot_mask = epochMask
                table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                        tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
                self.dataset = self.dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
                maxPos = np.max(
                    behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
                self.dataset = self.dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
                self.dataset = self.dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

        self.dataset = self.dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals,batched=False),
                              num_parallel_calls=tf.data.AUTOTUNE)

        #todo allow for speed filtering


    def get_NNdataset_spike(self,timeinDat):
        indexInDat= np.array(np.ravel(self.samplingRate*timeinDat),dtype=np.int64)
        filterData = self.dataset.filter(lambda x: tf.greater(tf.math.reduce_sum(tf.math.reduce_sum(tf.cast(tf.equal(x["indexInDat"][None,:],indexInDat[:,None]),
                                                                                         dtype=tf.float32),axis=1)),0))
        ans = list(filterData.as_numpy_iterator())
        return ans

    def get_NNdataset_spikepos(self):
        # indexInDat= np.array(np.ravel(self.samplingRate*timeinDat),dtype=np.int64)
        # filterData = self.dataset.filter(lambda x: tf.greater(tf.math.reduce_sum(tf.math.reduce_sum(tf.cast(tf.equal(x["indexInDat"][None,:],indexInDat[:,None]),
        #                                                                                  dtype=tf.float32),axis=1)),0))
        resData = self.dataset.map(lambda vals: vals["indexInDat"])
        ans = list(resData.as_numpy_iterator())
        return ans

    def compareWaveform(self,spikeSorted,behavior_data):


        spikeMat_labels, spikeMat_times, linearPreferredPos = spikeSorted
        spikeMat_labels = spikeMat_labels[:, :]
        spikeMat_times = spikeMat_times[:]

        # timeNN = pykeops.numpy.Vi(timePreds.astype(dtype=np.float64)[:, None])
        # timeSpike = pykeops.numpy.Vj(spikeMat_times)
        # linkSpikeTimeToNNtime = (timeSpike - timeNN).abs().argmin(axis=0)

    def load_dat(self):
        filPath = "/media/nas6/ProjetERC3/M1199/Reversal/M1199_20210416_Reversal.fil"
        datPath = "/media/nas6/ProjetERC3/M1199/Reversal/M1199_20210416_Reversal.dat"
        xmlPath = "/media/nas6/ProjetERC3/M1199/Reversal/M1199_20210416_Reversal.xml"

        filFileLengthInByte = os.stat(filPath).st_size
        datFileLengthInByte = os.stat(datPath).st_size
        list_channels, self.samplingRate, nChannels = rawDataParser.get_params(xmlPath)
        self.number_timeSteps = os.stat(datPath).st_size // (2 * nChannels)
        self.memmapData = np.memmap(datPath, dtype=np.int16, mode='r', shape=(self.number_timeSteps, nChannels))
        self.memmapFil = np.memmap(filPath, dtype=np.int16, mode='r', shape=(self.number_timeSteps, nChannels))

    def get_spike_waveform(self,spikeTime):
        # Given a spike time, we extract from the .fil and .dat
        # the waveforms of all channels at this spike time.
        # spikes are assumed to be sampled exactly according to sampling_rate.
        waveformsDat = []
        waveformsFil = []
        for s in np.ravel(spikeTime):
            measureStepOfSpikeTime = int(s * int(self.samplingRate))
            waveformsDat += [self.memmapData[measureStepOfSpikeTime:measureStepOfSpikeTime + 15, :]]
            waveformsFil += [self.memmapFil[measureStepOfSpikeTime:measureStepOfSpikeTime + 15, :]]
        waveformsDat = np.array(waveformsDat)
        waveformsFil = np.array(waveformsFil)
        return waveformsDat, waveformsFil

    def fromClusterToSpikeMat(self,cluster_data):
        # let us bin the spike times by 36 ms windows:
        nbSpikes = [a.shape[0] for a in cluster_data["Spike_labels"]]
        nbNeurons = [a.shape[1] for a in cluster_data["Spike_labels"]]
        spikeMat_labels = np.zeros([np.sum(nbSpikes), np.sum(nbNeurons)])
        spikeMat_times = np.zeros([np.sum(nbSpikes), 1])
        cnbSpikes = np.cumsum(nbSpikes)
        cnbNeurons = np.cumsum(nbNeurons)
        for id, n in enumerate(nbSpikes):
            if id > 0:
                spikeMat_labels[cnbSpikes[id - 1]:cnbSpikes[id], cnbNeurons[id - 1]:cnbNeurons[id]] = \
                    cluster_data["Spike_labels"][id]
                spikeMat_times[cnbSpikes[id - 1]:cnbSpikes[id], :] = cluster_data["Spike_times"][id]
            else:
                spikeMat_labels[0:cnbSpikes[id], 0:cnbNeurons[id]] = cluster_data["Spike_labels"][id]
                spikeMat_times[0:cnbSpikes[id], :] = cluster_data["Spike_times"][id]
        return spikeMat_times,spikeMat_labels

    def save_alignment_tools(self,trainerBayes,linearizationFunction):
        if (self.useTrain and not os.path.exists(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain","startTimeWindow.csv")))\
            or ((not self.useTrain and not self.useSleep) and not os.path.exists(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest","startTimeWindow.csv")))\
            or (self.useSleep and not os.path.exists(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","startTimeWindow.csv"))):
            self.load_dat()
            spikeMat_labels, spikeMat_times, linearPreferredPos,_ = trainerBayes.get_spike_ordered_by_prefPos(
                linearizationFunction)

            # gather all windows in the tensorflow dataset
            inputNN = self.get_NNdataset_spikepos()
            # noWindowinputNN = np.concatenate(inputNN)

            # dat id to NN windows id (or 0):
            lenInputNN = []
            meanTimeWindow = []
            startTimeWindow = []
            for id, startTime in tqdm(enumerate(inputNN)):
                if len(startTime) > 0:
                    startTimeWindow += [startTime[0]]
                else:
                    startTimeWindow += [np.nan] # we make sure these windows are never selected
                lenInputNN += [len(startTime)]
                meanTimeWindow += [np.mean(startTime / self.samplingRate)]
            lenInputNN = np.array(lenInputNN)
            meanTimeWindow = np.array(meanTimeWindow)
            startTimeWindow = np.array(startTimeWindow)

            goodStartTimeWindow = startTimeWindow[np.logical_not(np.isnan(startTimeWindow))]
            stopTimeWindow = goodStartTimeWindow+int(self.params.windowLength*self.samplingRate)
            # mapping spike sorted spike times to windows
            spikeMat_times_window = np.zeros([spikeMat_times.shape[0], 2])
            spikeMat_times_window[:, 0] = spikeMat_times[:, 0]

            spikeTime_lazy = pykeops.numpy.LazyTensor(spikeMat_times[:,0][:,None]*self.samplingRate,axis=0)
            startTimeWindow_lazy = pykeops.numpy.Vj(goodStartTimeWindow[:,None].astype(dtype=np.float64))
            stopTimeWindow_lazy = pykeops.numpy.Vj(stopTimeWindow[:,None].astype(dtype=np.float64))
            ans = (spikeTime_lazy-startTimeWindow_lazy).relu().sign() * ((stopTimeWindow_lazy-spikeTime_lazy).relu().sign())
            ans2 = ans.max_argmax_reduction(dim=1)
            ans2[1][np.equal(ans2[0],0)] = -1
            spikeMat_times_window[:, 1] = ans2[1][:,0]

            # for the pop vector we add one label for the noisy cluster
            spikeMat_window_popVector = np.zeros([len(inputNN), spikeMat_labels.shape[1] + 1])
            for idSpike, window in tqdm(enumerate(spikeMat_times_window[:, 1])):
                if window != -1:
                    cluster = np.where(np.equal(spikeMat_labels[idSpike, :], 1))[0]
                    if len(cluster) > 0:
                        spikeMat_window_popVector[int(window), 1 + cluster[0]] += 1
                    else:
                        spikeMat_window_popVector[int(window), 0] += 1  # noisy cluster

            if not os.path.exists(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain")):
                os.makedirs(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain"))
            if not os.path.exists(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest")):
                os.makedirs(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest"))
            if not os.path.exists(os.path.join(self.projectPath.resultsPath, "dataset", "alignment", "sleep")):
                os.makedirs(os.path.join(self.projectPath.resultsPath, "dataset", "alignment", "sleep"))

            if self.useTrain:
                df = pd.DataFrame(spikeMat_window_popVector)
                df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain","spikeMat_window_popVector.csv"))
                df = pd.DataFrame(meanTimeWindow)
                df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain","meanTimeWindow.csv"))
                df = pd.DataFrame(spikeMat_times_window)
                df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain","spikeMat_times_window.csv"))
                df = pd.DataFrame(startTimeWindow)
                df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain","startTimeWindow.csv"))
                df = pd.DataFrame(lenInputNN)
                df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketrain","lenInputNN.csv"))
            else:
                if self.useSleep:
                    df = pd.DataFrame(spikeMat_window_popVector)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","spikeMat_window_popVector.csv"))
                    df = pd.DataFrame(meanTimeWindow)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","meanTimeWindow.csv"))
                    df = pd.DataFrame(spikeMat_times_window)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","spikeMat_times_window.csv"))
                    df = pd.DataFrame(startTimeWindow)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","startTimeWindow.csv"))
                    df = pd.DataFrame(lenInputNN)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","lenInputNN.csv"))
                else:
                    df = pd.DataFrame(spikeMat_window_popVector)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest","spikeMat_window_popVector.csv"))
                    df = pd.DataFrame(meanTimeWindow)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest","meanTimeWindow.csv"))
                    df = pd.DataFrame(spikeMat_times_window)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest","spikeMat_times_window.csv"))
                    df = pd.DataFrame(startTimeWindow)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest","startTimeWindow.csv"))
                    df = pd.DataFrame(lenInputNN)
                    df.to_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","waketest","lenInputNN.csv"))
        # fig, ax = plt.subplots()
        # ax.imshow(spikeMat_window_popVector[100:200, :])
        # ax.set_aspect(67 / 100)
        # fig.show()
        # fig, ax = plt.subplots()
        # ax.scatter(range(spikeMat_window_popVector.shape[0]), np.sum(spikeMat_window_popVector, axis=1), s=1, c="black")
        # fig.show()
        # fig, ax = plt.subplots()
        # ax.scatter(lenInputNN, np.sum(spikeMat_window_popVector, axis=1), s=1, c="black")
        # ax.set_xlabel("number of spike fed to NN in window")
        # ax.set_ylabel("number of spike fed to bayesian in same window")
        # ax.set_title("wake set")
        # fig.show()



