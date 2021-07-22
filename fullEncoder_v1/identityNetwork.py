import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp
from tqdm import tqdm


class identityNetwork():

    def __init__(self, projectPath, params, device_name="/device:gpu:0"):
        super(identityNetwork, self).__init__()
        self.projectPath = projectPath
        self.params = params
        self.device_name = device_name
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            "pos": tf.io.FixedLenFeature([params.dim_output], tf.float32),
            # target position: current value of the environmental correlate
            "groups": tf.io.FixedLenFeature([], tf.int64),
            # the index of the groups having spike sequences in the window
            "time": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.FixedLenFeature([], tf.int64)}
        for g in range(params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.FixedLenFeature([params.nChannels[g],32],tf.float32)})

        self.inputsToSpikeNets = [
            tf.keras.layers.Input(shape=(self.params.nChannels[group], 32), name="group" + str(group)) for group in
            range(self.params.nGroups)]
        x = self.inputsToSpikeNets
        out = tf.concat(x,axis=1)
        self.model = tf.keras.Model(
                inputs=self.inputsToSpikeNets,
                outputs=out)

        self.model.compile()

    def observe_input(self):
        dic_padding = {"group0":0.0,
                        "group1":0.0,
                        "group2":0.0,
                        "group3":0.0,
                        "groups": np.array(-1,np.int64),
                        "indexInDat":np.array(-1,np.int64),
                        "pos_index": np.array(-1,np.int64),
                        "pos":-1.0,
                        "time":-1.0}

        def clean_window(tensors):
            # Time is set to mean time in the window
            # Position is set to be the last position
            # For each spike we filled other groups input with 0, here these are effectively removed
            tensors["time"] = tf.reduce_mean(tensors["time"])
            tensors["pos"] = tensors["pos"][-1,:]
            for g in range(self.params.nGroups):
                zeros = tf.constant(np.zeros([self.params.nChannels[g], 32]), tf.float32)
                nonZeros = tf.logical_not(tf.equal(tf.reduce_sum(input_tensor=tf.cast(tf.equal(
                    tensors["group" + str(g)], zeros), tf.int32), axis=[1, 2]), 32 * self.params.nChannels[g]))
                # nonZeros: control that the voltage measured is not 0, at all channels and time bin inside the detected spike
                tensors["group" + str(g)] = tf.gather(tensors["group" + str(g)], tf.where(nonZeros))[:, 0, :, :]
            return tensors

        def map_to_feature(tensors):
            return {"group0":tf.train.Feature(float_list = tf.train.FloatList(value=tensors["group0"].ravel())),
                    "group1":tf.train.Feature(float_list = tf.train.FloatList(value=tensors["group1"].ravel())),
                    "group2":tf.train.Feature(float_list = tf.train.FloatList(value=tensors["group2"].ravel())),
                    "group3":tf.train.Feature(float_list = tf.train.FloatList(value=tensors["group3"].ravel())),
                    "groups": tf.train.Feature(int64_list =  tf.train.Int64List(value=tensors["groups"])),
                    "indexInDat":tf.train.Feature(int64_list =  tf.train.Int64List(value=tensors["indexInDat"])),
                    "pos_index": tf.train.Feature(int64_list =  tf.train.Int64List(value=tensors["pos_index"])),
                    "pos":tf.train.Feature(float_list = tf.train.FloatList(value=tensors["pos"])),
                    "time":tf.train.Feature(float_list = tf.train.FloatList(value=[tensors["time"]]))}

        ndataset = tf.data.TFRecordDataset(
            os.path.join(self.projectPath.folder, "dataset", "dataset_singleSpike.tfrec"))
        ndataset = ndataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                                num_parallel_calls=tf.data.AUTOTUNE)

        #to access via memmap??
        timeSpikes = pd.read_csv(os.path.join(self.projectPath.folder,"dataset","spikeData_fromJulia.csv")).values[:,[0,3]]
        sleepSpikes = np.equal(timeSpikes[:,1],-2)
        wakeSpikes = np.greater(timeSpikes[:,1],-1)
        timeSpikesWake = timeSpikes[wakeSpikes]
        timeSpikeSleep = timeSpikes[sleepSpikes]

        window_stride = 0.036 #in ms
        window_length = 7*window_stride
        filename = os.path.join(self.projectPath.folder,"dataset","testwindow_7_bis.tfrec")
        writer = tf.io.TFRecordWriter(filename)
        #efficient strategy, using take and skip:
        currentFirstSpikeTime = timeSpikesWake[0,0]
        currentFirstSpikeId = 0
        startWind = []
        for i in tqdm(range(timeSpikesWake.shape[0])): #timeSpikesWake.shape[0]
            if(timeSpikesWake[i,0]>currentFirstSpikeTime+window_stride): # we find the beginning of the next window
                startWind+=[currentFirstSpikeId]
                currentFirstSpikeId = i
                currentFirstSpikeTime = timeSpikesWake[i,0]
        startWinds = np.array(startWind)



        stopWindow = []
        #iterate through all startWinds to find the end of the window
        currentWindow = 0
        for i in tqdm(range(timeSpikesWake.shape[0])): #timeSpikesWake.shape[0]
            if(timeSpikesWake[i,0]>timeSpikesWake[startWinds[currentWindow],0]+window_length): # we find the beginning of the next window
                stopWindow+=[i]
                currentWindow+=1
        stopWindow = np.array(stopWindow)

        for ids in tqdm(range(startWinds.shape[0])):
            s = startWinds[ids]
            dataWindow = (ndataset.take(stopWindow[ids]-s)).padded_batch(stopWindow[ids]-s,padding_values=dic_padding)
            dataWindowClean = dataWindow.map(clean_window)
            b = list(dataWindowClean.as_numpy_iterator())
            writer.write(tf.train.Example(features=tf.train.Features(feature=map_to_feature(b[0]))).SerializeToString())
            if(ids<startWinds.shape[0]-1):
                ndataset = ndataset.skip(startWinds[ids+1] - s)



        # r0 =  (ndataset.take(100)).padded_batch(100,padding_values=dic_padding)
        # r1 = (ndataset.take(5)).padded_batch(5,padding_values=dic_padding)
        # rWind = r0.concatenate(r1)
        # rWindClean = rWind.map(clean_window)
        # #Note: the only way to move from a dictionary of tensor like dataset to
        # # a Example proto like dataset, is to call the numpy iterator function :/
        # bs = list(rWindClean.as_numpy_iterator())
        # rWindClean = [map_to_feature(b) for b in bs]
        # rWindExamples = [tf.train.Example(features=tf.train.Features(feature=rw)) for rw in rWindClean]
        #
        #
        # [writer.write(rw.SerializeToString()) for rw in rWindExamples ]

        ndataset = tf.data.TFRecordDataset(os.path.join(self.projectPath.folder, "dataset", "testwindow.tfrec"))



        #
        #
        # # rWind = rWind.padded_batch(2,drop_remainder=True,padding_values={"group0":0.0,
        # #                                                                  "group1":0.0,
        # #                                                                  "group2":0.0,
        # #                                                                  "group3":0.0,
        # #                                                                  "groups": np.array(-1,np.int64),
        # #                                                                  "indexInDat":np.array(-1,np.int64),
        # #                                                                  "pos_index": np.array(-1,np.int64),
        # #                                                                  "pos":-1.0,
        # #                                                                  "time":-1.0})
        #
        #
        # robs = list(rWind.map(lambda vals: vals["time"]).as_numpy_iterator())
        # rWindBatch = rWind.batch(2, drop_remainder=True)
        # b = rWindBatch.take(1)
        # robs = list(b.map(lambda vals: vals["group2"]).as_numpy_iterator())
        #
        # r1 =  ndataset.take(1)
        # rcat = r0.concatenate(r1)
        # rwind = rcat.window(1,2,1)
        # rwind = rwind.batch(self.params.batch_size, drop_remainder=True)
        # outputs = self.model.predict(rwind)
        # # def change_zero(tensors):
        # #     for g in range(self.params.nGroups):
        # #         if tf.shape(tensors["group"+str(g)])[0]==0:
        # #             print("not ok",tf.shape(tensors["group"+str(g)])[0])
        # #             tensors["group" + str(g)] = zero_fill[g]
        # #         else:
        # #             print("ok",tf.shape(tensors["group"+str(g)])[0])
        # #             tensors["group"+str(g)] = tensors["group" + str(g)]
        # #     return tensors
        # # ndataset = ndataset.map(lambda vals: change_zero(vals))
        # dataset = ndataset.batch(self.params.batch_size, drop_remainder=True)
        # dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence_singleSpike(self.params, *vals,batched=True),
        #              num_parallel_calls=tf.data.AUTOTUNE)
        # d = dataset.take(2).as_numpy_iterator()
        #
        # outputs=  self.model.predict(dataset.take(10))
        #
        # fig,ax = plt.subplots()
        # for i in range(22):
        #     ax.plot(np.zeros(32)+i,outputs[0,i,:])
        # fig.show()
        #
        # print(outputs)

