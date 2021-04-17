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
from importData.rawDataParser import modify_feature_forBestTestSet,speed_filter
from resultAnalysis import performancePlots



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

        self.tfrec =  self.folder + 'dataset/dataset.tfrec'

        #To change at every experiment:
        self.resultsPath = self.folder + 'results_tf20_pythondata'
        # self.resultsNpz = self.resultsPath + '/inferring.npz'
        # self.resultsMat = self.resultsPath + '/inferring.mat'

        if not os.path.isdir(self.folder + 'dataset'):
            os.makedirs(self.folder + 'dataset')
        # if not os.path.isdir(self.folder + 'graph'):
        #     os.makedirs(self.folder + 'graph')
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
    def __init__(self,list_channels,dim_output, windowSize):
        self.nGroups = len(list_channels)
        self.dim_output = dim_output
        self.nChannels = [len(list_channels[n]) for n in range(self.nGroups)]
        self.length = 0

        self.nSteps = int(10000 * 0.036 / windowSize)
        self.nEpochs = 150
        # self.learningTime = detector.learningTime()
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

        ##Test param:
        self.shuffle_spike_order = False
        self.shuffle_convnets_outputs = False

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

    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K168/M1168_20210122_UMaze.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K168/_encoders_test/amplifier.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199/amplifier.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-monday/continuous.xml"
    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/continuous.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1604/signal.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1404/signal.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/RatCataneseOld/rat122-20090731.xml"
    # subprocess.run(["./getTsdFeature.sh", os.path.expanduser(xmlPath.strip('\'')), "\"" + "pos" + "\"",
    #                  "\"" + str(0.1) + "\"", "\"" + "end" + "\""])

    datPath = ''
    windowSize = 0.036
    mode = "full"

    #tf.debugging.set_log_device_placement(True)

    projectPath = Project(os.path.expanduser(xmlPath), datPath=os.path.expanduser(datPath), jsonPath=None)
    # f = np.load(os.path.join(projectPath.folder, "results", "inferring.npz"), allow_pickle=True)
    # outputs = {}
    # outputs["featurePred"]  = f["inferring"][:,0:2]
    # outputs["featureTrue"] = f["pos"]
    # outputs["times"] = f["times"]
    # from transformData.linearizer import uMazeLinearization2
    # linearizationFunction = uMazeLinearization2
    # projPredTF10,linearPredTF10 = uMazeLinearization2(outputs["featurePred"][:,0:2])
    # projTrueTF10,linearTrueTF10 = uMazeLinearization2(outputs["featureTrue"][:,0:2])
    # outputs["predofLoss"] = f["inferring"][:,2].reshape([f["inferring"].shape[0],1])
    # outputs["projPred"] = projPredTF10
    # outputs["projTruePos"] = projTrueTF10
    # outputs["linearPred"] = linearPredTF10
    # outputs["linearTrue"] = linearTrueTF10
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.folder,"results","nofilter"),1)
    # performancePlots.linear_performance(outputs, os.path.join(projectPath.folder, "results","filter"),0.1)

    list_channels, samplingRate, nChannels = rawDataParser.get_params(projectPath.xml)
    positions,_ = rawDataParser.get_position(projectPath.folder)
    params = Params(list_channels,positions.shape[1], windowSize)

    # OPTIMIZATION of tensorflow
    #tf.config.optimizer.set_jit(True) # activate the XLA compilation
    #mixed_precision.experimental.set_policy('mixed_float16')
    params.usingMixedPrecision = False

    # Training, testing, and preparing network for online setup
    if mode=="full":
        from fullEncoder_v1 import nnNetwork2 as Training
    elif mode=="decode":
        from decoder import decodeTraining as Training
    # todo: modify this loading of code files as we changed names!!

    # params.shuffle_spike_order = True
    trainer = Training.LSTMandSpikeNetwork(projectPath, params)
    # The data are now saved into a tfrec file,
    # next we provide an efficient tool for selecting the training and testing step

    #first we find the first and last position index in the recording dataset
    # to set as limits in for the nnbehavior.mat
    # dataset = tf.data.TFRecordDataset(projectPath.tfrec)
    # dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(trainer.feat_desc, *vals))
    # pos_index = list(dataset.take(10).map(lambda x: x["times"]).as_numpy_iterator())
    #
    speed_filter(projectPath.folder,overWrite=False)
    # modify_feature_forBestTestSet(projectPath.folder) #plimits=[np.min(pos_index),np.max(pos_index)]

    trainLosses = trainer.train()


    #project the prediction and true data into a line fit on the maze:
    from transformData.linearizer import doubleArmMazeLinearization
    from transformData.linearizer import uMazeLinearization2,verifyLinearization
    from importData.ImportClusters import getBehavior
    path_to_code = os.path.join(projectPath.folder, "../../neuroEncoders/transformData")
    # linearizationFunction = lambda x: doubleArmMazeLinearization(x,scale=True,path_to_folder=path_to_code)
    behave_data = getBehavior(projectPath.folder,getfilterSpeed=False)
    verifyLinearization(behave_data["positions"])
    linearizationFunction = uMazeLinearization2

    name_save = "resultTest"
    outputs = trainer.test(linearizationFunction,name_save,useTrain=False)
    performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1)
    performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1)

    name_save = "resultTrain"
    outputs = trainer.test(linearizationFunction,name_save,useTrain=True)
    performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1)
    performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1)

    name_save = "resultTest_full"
    outputs = trainer.test(linearizationFunction,name_save,useTrain=False)
    performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1)
    performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1)

    ## Test with shuflling of spikes in the window
    # We ask here if the algorithm uses the order of the spikes in the window
    params.shuffle_spike_order = True
    trainer.model = trainer.mybuild(trainer.get_Model(),modelName="spikeOrdershufflemodel.png")
    name_save = "result_spikeorderinwindow_shuffle"
    output_shuffle_spikeorder = trainer.test(linearizationFunction,name_save)
    performancePlots.linear_performance(output_shuffle_spikeorder,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1)
    performancePlots.linear_performance(output_shuffle_spikeorder,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1)

    params.shuffle_spike_order = False
    params.shuffle_convnets_outputs = True
    trainer.model = trainer.mybuild(trainer.get_Model(),modelName="convNetOutputshufflemodel.png")
    name_save = "result_convOutputs_shuffle"
    outputs_shuffle_convoutputs = trainer.test(linearizationFunction,name_save)
    performancePlots.linear_performance(outputs_shuffle_convoutputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1)
    performancePlots.linear_performance(outputs_shuffle_convoutputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1)


    # f = np.load(os.path.join(projectPath.folder,"resultsMondayOne","inferring.npz"),allow_pickle=True)
    # predFeature  = f["inferring"]
    # truefeature = f["pos"]
    # timeold = f["times"]
    # # Quickly let us make a confusion matrix:
    # # projPredTF10,linearPredTF10 = uMazeLinearization2(predFeature[bestPred_tf1,0:2])
    # # projTrueTF10,linearTrueTF10 = uMazeLinearization2(truefeature[bestPred_tf1,0:2])


if __name__=="__main__":
    # In this architecture we use a 2.0 tensorflow backend, predicting solely the position.
    # I.E without using the simpler feature strategy based on stratified spaces
    main()
