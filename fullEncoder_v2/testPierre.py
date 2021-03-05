#!env/bin/python3
# test file....

import sys
import os.path
import subprocess
import numpy as np
import tensorflow as tf
from contextlib import ExitStack
import matplotlib.pyplot as plt


class Project():
    def __init__(self, xmlPath, datPath='', jsonPath=None):
        if xmlPath[-3:] != "xml": #change the last character to xml if it was not and checks that it exists
            if os.path.isfile(xmlPath[:-3]+"xml"):
                xmlPath = xmlPath[:-3]+"xml"
            else:
                raise ValueError("the path "+xmlPath+" doesn't match a .xml file")
        self.xml = xmlPath
        self.baseName = xmlPath[:-4]
        if datPath == '':
            self.dat = self.baseName + '.dat'
        else:
            self.dat = datPath
        findFolder = lambda path: path if path[-1]=='/' or len(path)==0 else findFolder(path[:-1])
        self.folder = findFolder(self.dat)
        self.fil = self.dat[:-4] + '.fil'
        if jsonPath == None:
            self.json = self.baseName + '.json'
            self.graph = self.folder + 'graph/decoder'
            self.graphMeta = self.folder + 'graph/decoder.meta'
        else:
            print('using file:',jsonPath)
            self.json = jsonPath
            self.thresholds, self.graph = self.getThresholdsAndGraph()
            self.graphMeta = self.graph + '.meta'

        self.tfrec = {
            "train": self.folder + 'dataset/trainingDataset.tfrec',
            "test": self.folder + 'dataset/testingDataset.tfrec'}

        self.resultsNpz = self.folder + 'results/inferring.npz'
        self.resultsMat = self.folder + 'results/inferring.mat'

        if not os.path.isdir(self.folder + 'dataset'):
            os.makedirs(self.folder + 'dataset')
        if not os.path.isdir(self.folder + 'graph'):
            os.makedirs(self.folder + 'graph')
        if not os.path.isdir(self.folder + 'results'):
            os.makedirs(self.folder + 'results')

    def clu(self, g):
        return self.baseName + ".clu." + str(g+1)

    def res(self, g):
        return self.baseName + ".res." + str(g+1)

    def pos(self, g):
        return self.folder + "dataset/pos." + str(g+1) + ".npz"

    def getThresholdsAndGraph(self):
        import json
        with open(self.json, 'r') as f:
            info = json.loads(f.read())
        return [[abs(info[d][f]) for f in ['threshold'+str(c) for c in range(info[d]['nChannels'])]] \
                for d in ['group'+str(g) for g in range(info['nGroups'])]], \
            info['encodingPrefix']


class Params:
    def __init__(self, detector, windowSize):
        self.nGroups = detector.nGroups()
        self.dim_output = detector.dim_output()
        self.nChannels = detector.numChannelsPerGroup()
        self.length = 0

        self.nSteps = int(10000 * 0.036 / windowSize)
        self.nEpochs = 10
        self.learningTime = detector.learningTime()
        self.windowLength = windowSize # in seconds, as all things should be

        ### from units encoder params
        self.validCluWindow = 0.0005
        self.kernel = 'epanechnikov'
        self.bandwidth = 0.1
        self.masking = 20

        ### full encoder params
        self.nFeatures = 128 #Pierre: test with much lower nb of feature; before it was 128
        self.lstmLayers = 3
        self.lstmSize = 128 #to change back to 128
        self.lstmDropout = 0.3

        self.batch_size = 52 #previously 52
        self.timeMajor = True

        self.learningRates = [0.0003] #, 0.00003, 0.00001]
        self.lossLearningRate = 0.00003
        self.lossActivation = None

        self.usingMixedPrecision = True # this boolean indicates weither tensorflow uses mixed precision
        # ie enforcing float16 computations whenever possible
        # According to tf tutorials, we can allow that in most layer except the output for unclear reasons linked to gradient computations



def main():
    from importData import rawDataParser
    from fullEncoder_v2 import nnUtils
    from fullEncoder_v2 import nnNetwork2
    import tensorflow.keras.mixed_precision as mixed_precision

    # to set as env variable: TF_GPU_THREAD_MODE=gpu_private

    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/RatCatanese/rat122-20090731.xml"
    datPath = xmlPath

    useOpenEphysFilter = False # false if we don't have a .fil file
    windowSize = 0.036
    mode = "full"
    print('NEUROENCODER: MODE', mode)
    split = 0.1
    print('NEUROENCODER: TEST SPLIT', split)
    print()

    #tf.debugging.set_log_device_placement(True)

    projectPath = Project(os.path.expanduser(xmlPath), datPath=os.path.expanduser(datPath), jsonPath=None)
    spikeDetector = rawDataParser.SpikeDetector(projectPath, useOpenEphysFilter, mode)
    params = Params(spikeDetector, windowSize)

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Disable first GPU
        #tf.config.set_visible_devices([], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("FAILED")
        pass

    # OPTIMIZATION of tensorflow
    tf.config.optimizer.set_jit(True) # activate the XLA compilation
    mixed_precision.experimental.set_policy('mixed_float16')
    params.usingMixedPrecision = True

    #tf.distribute.OneDeviceStrategy(device=tf.config.list_logical_devices('GPU')[0])



    if mode == "decode":
        spikeDetector.setThresholds(projectPath.thresholds)

    # Create data files if not present
    if not os.path.isfile(projectPath.tfrec["test"]):
        # setup data readers and writers meta data
        spikeGen = nnUtils.spikeGenerator(projectPath, spikeDetector, maxPos=spikeDetector.maxPos())
        spikeSequenceGen = nnUtils.getSpikeSequences(params, spikeGen())
        readers = {}
        writers = {"testSequences": tf.io.TFRecordWriter(projectPath.tfrec["test"])}
        writers.update({"trainSequences": tf.io.TFRecordWriter(projectPath.tfrec["train"])})

        with ExitStack() as stack:
            # Open data files
            for k,v in writers.items():
                writers[k] = stack.enter_context(v)
            for k,v in readers.items():
                readers[k] = stack.enter_context(v)

            # generate spike sequences in windows of size params.windowLength
            for example in spikeSequenceGen:
                if example["train"] == None:
                    continue
                if example["train"]:
                    writers["trainSequences"].write(nnUtils.serializeSpikeSequence(
                        params,
                        *tuple(example[k] for k in ["pos", "groups", "length", "times"]+["spikes"+str(g) for g in range(params.nGroups)])))
                else:
                    writers["testSequences"].write(nnUtils.serializeSpikeSequence(
                        params,
                        *tuple(example[k] for k in ["pos", "groups", "length", "times"]+["spikes"+str(g) for g in range(params.nGroups)])))

    # Training, testing, and preparing network for online setup
    if mode=="full":
        from fullEncoder_v2 import nnNetwork2 as Training
    elif mode=="decode":
        from decoder import decodeTraining as Training
    # todo: modify here as we changed names!!
    trainer = Training.LSTMandSpikeNetwork(projectPath, params)
    trainLosses = trainer.train()
    outputs = trainer.test()

    inferedPos = np.stack(outputs["inferring"])
    realPos = np.reshape(outputs["pos"],inferedPos.shape)
    fig,ax = plt.subplots(3,1)
    ax[0].scatter(inferedPos[:,0],realPos[:,0])
    ax[0].scatter(inferedPos[:, 1], realPos[:, 1])
    ax[1].hist(realPos[:,0],bins=100)
    ax[1].hist(realPos[:,1], bins=100)
    ax[2].scatter(realPos[:, 0], realPos[:, 1])
    ax[2].scatter(inferedPos[:, 0], inferedPos[:, 1])
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(inferedPos[:, 0], c="black")
    ax.plot(realPos[:, 0], c="red")
    fig.show()
    fig,ax = plt.subplots()
    ax.plot(trainLosses)


    # Saving files
    np.savez(projectPath.resultsNpz, trainLosses=trainLosses, **outputs)
    import scipy.io
    scipy.io.savemat(projectPath.resultsMat, np.load(projectPath.resultsNpz, allow_pickle=True))

    from fullEncoder_v2 import printResults
    printResults.printResults(projectPath.folder)

if __name__=="__main__":
    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/RatCatanese/rat122-20090731.xml"
    subprocess.run(["./getTsdFeature.sh", os.path.expanduser(xmlPath.strip('\'')), "\"" + "pos" + "\"",
                    "\"" + str(0.1) + "\"", "\"" + "end" + "\""])

    main()

    print()
    print()
    print('Encoding over.')
