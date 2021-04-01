#!env/bin/python3
# neuroEncoder bys the MOBS team.
# may 2020
# t.balenbois@gmail.com

import sys
import os.path
import subprocess
import numpy as np
import tensorflow as tf
from contextlib import ExitStack
import matplotlib.pyplot as plt
import pandas as pd


class Project():
    def __init__(self, xmlPath, datPath='', jsonPath=None):
        if xmlPath[-3:] != "xml":
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

        self.resultsPath = self.folder + 'resultsTrainingNewTestSet'
        self.resultsNpz = self.resultsPath + '/inferring.npz'
        self.resultsMat = self.resultsPath + '/inferring.mat'

        if not os.path.isdir(self.folder + 'dataset'):
            os.makedirs(self.folder + 'dataset')
        if not os.path.isdir(self.folder + 'graph'):
            os.makedirs(self.folder + 'graph')
        if not os.path.isdir(self.resultsPath):
            os.makedirs(self.resultsPath)
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
        self.nEpochs = 20
        self.learningTime = detector.learningTime()
        self.windowLength = windowSize # in seconds, as all things should be

        ### from units encoder params
        self.validCluWindow = 0.0005
        self.kernel = 'epanechnikov'
        self.bandwidth = 0.1
        self.masking = 20

        ### full encoder params
        self.nFeatures = 128
        self.lstmLayers = 3
        self.lstmSize = 128
        self.lstmDropout = 0.3

        self.batch_size = 52
        self.timeMajor = True

        self.learningRates = [0.0003, 0.00003, 0.00001]
        self.lossLearningRate = 0.00003
        self.lossActivation = None



def main():
    from importData import rawDataParser
    from fullEncoder import nnUtils
    from unitClassifier import bayesUtils

    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K168/_encoders_test/amplifier.xml"
    datPath = ''
    useOpenEphysFilter = False
    # false if we don't have a .fil file
    # will then use the IntanFilter, except if a .npz file is present (old version of dataset)
    windowSize = 0.036
    mode = "full"
    split = 0.1
    gpu = False

    if gpu:
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    trainLosses = []
    projectPath = Project(os.path.expanduser(xmlPath), datPath=os.path.expanduser(datPath), jsonPath=None)
    spikeDetector = rawDataParser.SpikeDetector(projectPath, useOpenEphysFilter, mode)
    params = Params(spikeDetector, windowSize)
    if mode == "decode":
        spikeDetector.setThresholds(projectPath.thresholds)

    # Create data files if not present
    if not os.path.isfile(projectPath.tfrec["test"]):

        # setup data readers and writers meta data
        spikeGen = nnUtils.spikeGenerator(projectPath, spikeDetector, maxPos=spikeDetector.maxPos())
        spikeSequenceGen = nnUtils.getSpikeSequences(params, spikeGen())
        readers = {}
        writers = {"testSequences": tf.python_io.TFRecordWriter(projectPath.tfrec["test"])}
        if mode=="full":
            writers.update({"trainSequences": tf.python_io.TFRecordWriter(projectPath.tfrec["train"])})
        elif mode=="fromUnits":
            readers.update({"clu"+str(g): open(projectPath.clu(g), 'r') for g in range(params.nGroups)})
            readers.update({"res"+str(g): open(projectPath.res(g), 'r') for g in range(params.nGroups)})
            writers.update({"trainGroup"+str(g): tf.python_io.TFRecordWriter(projectPath.tfrec["train"]+"."+str(g)) for g in range(params.nGroups)})

        with ExitStack() as stack:
            # Open data files
            for k,v in writers.items():
                writers[k] = stack.enter_context(v)
            for k,v in readers.items():
                readers[k] = stack.enter_context(v)

            if mode=="fromUnits":
                nClusters = [int(readers["clu"+str(g)].readline()) for g in range(params.nGroups)]
                params.nClusters = nClusters
                clusterPositions = [{"clu"+str(n):[] for n in range(params.nClusters[g])} for g in range(params.nGroups)]
                clusterReaders = [bayesUtils.ClusterReader(readers["clu"+str(g)], readers["res"+str(g)], spikeDetector.samplingRate) \
                    for g in range(params.nGroups)]
                [clusterReaders[g].getNext() for g in range(params.nGroups)]

            # generate spike sequences in windows of size params.windowLength
            for example in spikeSequenceGen:
                if example["train"] == None:
                    continue
                if example["train"]:
                    if mode=="full":
                        writers["trainSequences"].write(nnUtils.serializeSpikeSequence(
                            params, 
                            *tuple(example[k] for k in ["pos", "groups", "length", "times"]+["spikes"+str(g) for g in range(params.nGroups)])))
                    else:
                        # If decoding from units, we need to find a sorted spike with corresponding timestamp
                        for spk in range(example["length"]):
                            group = example["groups"][spk]
                            while clusterReaders[group].res < example["times"][spk] - params.validCluWindow:
                                clusterReaders[group].getNext()
                            if clusterReaders[group].res > example["times"][spk] + params.validCluWindow:
                                continue
                            clusterPositions[group]["clu"+str(clusterReaders[group].clu)].append(example["pos"])
                            writers["trainGroup"+str(group)].write(nnUtils.serializeSingleSpike(
                                params,
                                clusterReaders[group].clu,
                                example["spikes"+str(group)][(np.array(example["groups"])==group)[:spk+1].sum()-1]))
                            clusterReaders[group].getNext()

                else:
                    writers["testSequences"].write(nnUtils.serializeSpikeSequence(
                        params, 
                        *tuple(example[k] for k in ["pos", "groups", "length", "times"]+["spikes"+str(g) for g in range(params.nGroups)])))

        if mode=="fromUnits":
            for g in range(params.nGroups):
                for c in range(params.nClusters[g]):
                    clusterPositions[g]["clu"+str(c)] = np.array(clusterPositions[g]["clu"+str(c)])
                np.savez(projectPath.pos(g), **clusterPositions[g])


    # Training, testing, and preparing network for online setup
    if mode=="full":
        from fullEncoder import nnTraining as Training
    elif mode=="fromUnits":
        from unitClassifier import bayesTraining as Training
    elif mode=="decode":
        from decoder import decodeTraining as Training
    trainer = Training.Trainer(projectPath, params, spikeDetector, device_name=device_name)
    trainLosses = trainer.train()
    df = pd.DataFrame(trainLosses)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "lossTraining.csv"))

    outputs = trainer.test()
    predPos = outputs["inferring"][:, 0:2]
    truePos = outputs["pos"]
    predLoss = outputs["inferring"][:,2]
    timeStepPred = outputs["times"]

    # Saving files
    df = pd.DataFrame(predPos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featurePred.csv"))
    df = pd.DataFrame(truePos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featureTrue.csv"))
    df = pd.DataFrame(predLoss)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "lossPred.csv"))
    df = pd.DataFrame(timeStepPred)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "timeStepPred.csv"))
    # Saving files
    #np.savez(projectPath.resultsNpz, trainLosses=trainLosses, **outputs)


    maxPos = spikeDetector.maxPos()
    euclideanDistance = np.sqrt(np.sum(np.square(predPos*maxPos - truePos*maxPos),axis=1))
    np.mean(euclideanDistance)
    np.std(euclideanDistance)
    fig,ax = plt.subplots()
    ax.scatter(truePos[:, 0], truePos[:, 1], c="black", alpha=0.1,label="true Position")
    ax.scatter(predPos[:,0],predPos[:,1],c="red",alpha=0.1,label="predicted Position")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("prediction with Thibault's architecture")
    fig.legend()
    fig.show()
    fig.savefig(os.path.join(projectPath.folder,"results","testSetPrediciton.png"))

    fig,ax = plt.subplots()
    ax.plot(trainLosses[:,0],c="red")
    ax2 = ax.twinx()
    ax2.plot(trainLosses[:,1],c="black")
    ax.set_title("1 epochs = 50 time steps")
    ax.set_xlabel("epoch")
    ax.set_ylabel("position loss",c="red")
    ax2.set_ylabel("loss of loss", c="black")
    fig.show()

    fig,ax = plt.subplots()
    ax.hist(euclideanDistance,bins=100)
    fig.show()

    fig,ax = plt.subplots(2,1)
    ax[0].plot(truePos[:,0])
    ax[0].plot(predPos[:,0])
    ax[1].plot(truePos[:,1])
    ax[1].plot(predPos[:,1])
    fig.show()



    import scipy.io
    scipy.io.savemat(projectPath.resultsMat, np.load(projectPath.resultsNpz, allow_pickle=True))

    from fullEncoder import printResults
    printResults.printResults(projectPath.folder)

if __name__=="__main__":

    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K168/_encoders_test/amplifier.xml"
    subprocess.run(["./getTsdFeature.sh", os.path.expanduser(xmlPath.strip('\'')), "\"" + "pos" + "\"",
                    "\"" + str(0.1) + "\"", "\"" + "end" + "\""])
    main()

