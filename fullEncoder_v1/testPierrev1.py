#!env/bin/python3
# test file....

import sys
import os.path
import subprocess
import numpy as np
import tensorflow as tf
from contextlib import ExitStack
import matplotlib.pyplot as plt
import pandas as pd
import transformData.linearizer
from importData.rawDataParser import modify_feature_forBestTestSet


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

        #TO change at every experiment:
        self.resultsPath = self.folder + 'results_TF20_NoDropoutSlowRNNhardsigmoid_Euclidean_blockloss'
        self.resultsNpz = self.resultsPath + '/inferring.npz'
        self.resultsMat = self.resultsPath + '/inferring.mat'

        if not os.path.isdir(self.folder + 'dataset'):
            os.makedirs(self.folder + 'dataset')
        if not os.path.isdir(self.folder + 'graph'):
            os.makedirs(self.folder + 'graph')
        if not os.path.isdir(self.resultsPath):
            os.makedirs(self.resultsPath )
        if not os.path.isdir(os.path.join(self.resultsPath, "resultInference")):
            os.makedirs(os.path.join(self.resultsPath, "resultInference"))

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
        self.nEpochs = 120
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

        # To speed up computation we might reduce the number of step done
        # by the lstm.
        self.fasterRNN = False

        self.batch_size = 52 #previously 52

        self.learningRates = [0.0003] #  0.00003  ,    0.00003, 0.00001]
        self.lossLearningRate = 0.00003
        self.lossActivation = None

        self.usingMixedPrecision = True # this boolean indicates weither tensorflow uses mixed precision
        # ie enforcing float16 computations whenever possible
        # According to tf tutorials, we can allow that in most layer except the output for unclear reasons linked to gradient computations

        self.visualizeSet = False


def main():
    from importData import rawDataParser
    from fullEncoder_v1 import nnUtils
    from fullEncoder_v1 import nnNetwork2
    import tensorflow.keras.mixed_precision as mixed_precision

    # to set as env variable: TF_GPU_THREAD_MODE=gpu_private
    # tf.compat.v1.enable_eager_execution()

    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K168/_encoders_testV1/amplifier.xml"
    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/RatCataneseOld/rat122-20090731.xml"
    datPath = ''
    useOpenEphysFilter = False # false if we don't have a .fil file
    windowSize = 0.036
    mode = "full"
    split = 0.1

    #tf.debugging.set_log_device_placement(True)

    projectPath = Project(os.path.expanduser(xmlPath), datPath=os.path.expanduser(datPath), jsonPath=None)
    spikeDetector = rawDataParser.SpikeDetector(projectPath, useOpenEphysFilter, mode)
    params = Params(spikeDetector, windowSize)

    modify_feature_forBestTestSet(projectPath.folder,1000)

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
    #tf.config.optimizer.set_jit(True) # activate the XLA compilation
    #mixed_precision.experimental.set_policy('mixed_float16')
    params.usingMixedPrecision = False

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
                    print("None example found")
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
        from fullEncoder_v1 import nnNetwork2 as Training
    elif mode=="decode":
        from decoder import decodeTraining as Training
    # todo: modify this loading of code files as we changed names!!
    trainer = Training.LSTMandSpikeNetwork(projectPath, params)
    # The data are now saved into a tfrec file,
    # next we provide an efficient tool for selecting the training and testing step

    if params.visualizeSet:
        # we load training and testing data plotting the env variable,
        # and the histrogram over the test set
        datasetTrain = tf.data.TFRecordDataset(projectPath.tfrec["train"])
        datasetTrain = datasetTrain.map(
            lambda *vals: nnUtils.parseSerializedSequence(params, trainer.feat_desc, *vals, batched=False))
        datasetPos = datasetTrain.map(lambda x: x["pos"])
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        datasetTest = tf.data.TFRecordDataset(projectPath.tfrec["test"])
        datasetTest = datasetTest.map(
            lambda *vals: nnUtils.parseSerializedSequence(params, trainer.feat_desc, *vals, batched=False))
        datasetTestPos = datasetTest.map(lambda x: x["pos"])
        fullFeatureTrueTest = list(datasetTestPos.as_numpy_iterator())
        fullFeatureTrueTest = np.array(fullFeatureTrueTest)

        # next we use our linearization tool:
        from transformData.linearizer import  doubleArmMazeLinearization
        path_to_code = os.path.join(projectPath.folder, "../../neuroEncoders/transformData")
        projPosTrue = doubleArmMazeLinearization(fullFeatureTrue,scale=True,path_to_folder=path_to_code)
        projPosTest= doubleArmMazeLinearization(fullFeatureTrueTest,scale=True,path_to_folder=path_to_code)

        fig,ax = plt.subplots(2,2)
        ax[0,0].plot(projPosTrue[:,0])
        ax[0,1].hist(projPosTrue[:, 0],bins=100)
        ax[0,0].set_title("training data")
        ax[1,0].plot(projPosTest[:, 0])
        ax[1, 0].set_title("testing data")
        ax[1,1].hist(projPosTest[:,0],bins=100)
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(projectPath.resultsPath,"linearPosHist.png"))

    # trainLosses = trainer.train()
    # df = pd.DataFrame(trainLosses)
    # df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "lossTraining.csv"))
    # fog,ax = plt.subplots()
    # ax.plot(trainLosses[:,0])
    # plt.show()

    outputs = trainer.test()
    predPos = outputs["featurePred"][:, 0:2]
    truePos = outputs["featureTrue"]
    predLoss = outputs["predofLoss"]
    timeStepsPred = outputs["times"]

    #project the prediction and true data into a line fit on the maze:
    from transformData.linearizer import doubleArmMazeLinearization
    path_to_code = os.path.join(projectPath.folder, "../../neuroEncoders/transformData")
    projPredPos = doubleArmMazeLinearization(predPos,scale=True,path_to_folder=path_to_code)
    projTruePos = doubleArmMazeLinearization(truePos,scale=True,path_to_folder=path_to_code)

    # Saving files
    df = pd.DataFrame(predPos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featurePred.csv"))
    df = pd.DataFrame(truePos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featureTrue.csv"))
    df = pd.DataFrame(projPredPos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "projPredFeature.csv"))
    df = pd.DataFrame(projTruePos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "projTrueFeature.csv"))
    df = pd.DataFrame(predLoss[:,0,0])
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "lossPred.csv"))
    df = pd.DataFrame(timeStepsPred)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "timeStepsPred.csv"))

    # linearize the results:
    linearPred = transformData.linearizer.uMazeLinearization(predPos)
    linearTrue = transformData.linearizer.uMazeLinearization(truePos)


    maxPos = spikeDetector.maxPos()
    euclideanDistance = np.sqrt(np.sum(np.square(predPos * maxPos - truePos * maxPos), axis=1))
    np.mean(euclideanDistance)
    np.std(euclideanDistance)

    fig, ax = plt.subplots()
    ax.scatter(truePos[:, 0], truePos[:, 1], c="black", alpha=0.1, label="true Position")
    ax.scatter(predPos[:, 0], predPos[:, 1], c="red", alpha=0.1, label="predicted Position")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("prediction with TF2.0's architecture")
    fig.legend()
    fig.show()
    fig, ax = plt.subplots()
    ax.scatter(linearTrue[:, 0], linearTrue[:, 1], c="black", alpha=0.1, label="true Position")
    ax.scatter(linearPred[:, 0], linearPred[:, 1], c="red", alpha=0.1, label="predicted Position")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("prediction with TF2.0's architecture")
    fig.legend()
    fig.show()
    fig.savefig(os.path.join(projectPath.resultsPath, "testSetPrediciton.png"))
    fig, ax = plt.subplots()
    ax.plot(trainLosses[:, 0], c="red")
    ax2 = ax.twinx()
    ax2.plot(trainLosses[:, 1], c="black")
    ax.set_title("1 epochs = 1018 time steps")
    ax.set_xlabel("epoch")
    ax.set_ylabel("position loss", c="red")
    ax2.set_ylabel("loss of loss", c="black")
    fig.show()
    fig, ax = plt.subplots()
    ax.hist(euclideanDistance, bins=100)
    fig.show()


    # TO do : create a python file for automatic projection of the results...
    # so that Dima can use it easily.
    bestPred = predLoss < np.quantile(predLoss,0.9)
    fig,ax = plt.subplots()
    ax.plot(predPos[bestPred[:,0,0],1],c="orange")
    ax.plot(truePos[bestPred[:,0,0],1],c="black")
    fig.show()

    fig,ax = plt.subplots(2,1)
    ax[0].plot(predPos[:, 0], c="orange")
    ax[0].plot(truePos[:, 0], c="black")
    ax[1].plot(linearPred[:,0],c="orange")
    ax[1].plot(linearTrue[:,0],c="black")
    fig.show()


if __name__=="__main__":
    # In this architecture we use a 2.0 tensorflow backend, predicting solely the position.
    # I.E without using the simpler feature strategy based on stratified spaces.

    main()
