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
from importData.rawDataParser import modify_feature_forBestTestSet,speed_filter,select_sleep_gui
from resultAnalysis import performancePlots
from importData.JuliaData.juliaDataParser import julia_spike_filter
from importData import ImportClusters
from SimpleBayes import decodebayes
from fullEncoder_v1 import nnNetwork2 as Training
from tqdm import  tqdm

# from transformData.linearizer import doubleArmMazeLinearization
from transformData.linearizer import UMazeLinearizer
from importData.ImportClusters import getBehavior

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
        self.tfdata = self.folder+'dataset/dataset'
        self.tfrecSleep = self.folder + 'dataset/datasetSleep.tfrec'

        #To change at every experiment:
        self.resultsPath = self.folder + 'dataForKarim_withOldFilter'
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

        self.nSteps = int(10000 * 0.036 / windowSize) #not important anymore
        self.nEpochs = 30 #
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

        self.batch_size = 52 #5 #52 #previously 52

        self.nb_eval_dropout = 100

        self.learningRates = [0.0003] #  0.00003  ,    0.00003, 0.00001]
        self.lossActivation = None #tf.nn.relu

        self.usingMixedPrecision = True # this boolean indicates weither tensorflow uses mixed precision
        # ie enforcing float16 computations whenever possible
        # According to tf tutorials, we can allow that in most layer except the output for unclear reasons linked to gradient computations


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
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/continuous.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/LisaRouxDataset/M007_S07_07222015.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1604/signal.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1404/signal.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/RatCataneseOld/rat122-20090731.xml"
    # xmlPath = "/media/nas6/ProjetERC2/Mouse-K199/20210408/_Concatenated/M1199_20210408_UMaze_SpikeRef.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K199-2/M1199_20210408_UMaze_SpikeRef.xml"
    xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-reversal/M1199_20210416_Reversal.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-Reversal-1sWindow//M1199_20210416_Reversal.xml"
    # xmlPath = "/media/nas6/ProjetERC3/M1199/Reversal/M1199_20210416_Reversal.xml"
    # xmlPath = "/home/mobs/Documents/PierreCode/dataTest/LisaRouxNewDataset/M007_S07_07222015.xml"

    # subprocess.run(["./getTsdFeature.sh", os.path.expanduser(xmlPath.strip('\'')), "\"" + "pos" + "\"",
    #                  "\"" + str(0.1) + "\"", "\"" + "end" + "\""])

    datPath = ''
    windowSize = 7*0.036 #1 #corresponds to 30 windows of 36 ms # 0.036

    #tf.debugging.set_log_device_placement(True)

    projectPath = Project(os.path.expanduser(xmlPath), datPath=os.path.expanduser(datPath), jsonPath=None)

    # A matlab program is used to extract the BehaveRessources info into a nn.behavior.
    if not os.path.exists(os.path.join(projectPath.folder,"nnBehavior.mat")):
        subprocess.run(["./getTsdFeature.sh", projectPath.folder])
    # We still need to select the sleep sessions manually
    select_sleep_gui(projectPath.folder, overWrite=False)

    list_channels, samplingRate, nChannels = rawDataParser.get_params(projectPath.xml)
    positions,_ = rawDataParser.get_position(projectPath.folder)
    params = Params(list_channels,positions.shape[1], windowSize)

    julia_spike_filter(projectPath,window_length=windowSize,singleSpike=False)

    # OPTIMIZATION of tensorflow
    #tf.config.optimizer.set_jit(True) # activate the XLA compilation
    #mixed_precision.experimental.set_policy('mixed_float16')
    params.usingMixedPrecision = False

    #Since TF 2.4.1, too much memory seems to be allocated for the LSTM, and the programm would
    # throw an error if we don't manually prevent it from full memory growth. Before it was an issue
    # that only existed for windows, but it seems to have now spread to linux as well...
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # params.shuffle_spike_order = True
    trainer = Training.LSTMandSpikeNetwork(projectPath, params)
    # The data are now saved into a tfrec file,

    # fig,ax = plt.subplots()
    # ax.scatter(noWindowinputNN/wfc.samplingRate,noWindowinputNN/wfc.samplingRate,c="black",s=1,label="NN input spikes",alpha=0.5)
    # # separate cluster 0 and other
    # cluster0mask = np.equal(np.sum(spikeMat_labels,axis=1),0)
    # ax.scatter(spikeMat_times[wakeMask[:,0]*cluster0mask],spikeMat_times[wakeMask[:,0]*cluster0mask]+1,c="red",s=1,label="spike sorting, cluster 0")
    # otherclustermask = np.logical_not(cluster0mask)
    # ax.scatter(spikeMat_times[wakeMask[:,0] * otherclustermask], spikeMat_times[wakeMask[:,0] * otherclustermask] + 0.75, c="orange", s=1,label="spike sorting, other cluster")
    # ax.scatter(spikeMat_times[wakeMask],spikeMat_times[wakeMask]+0.5,c="grey",s=1,label="spike sorting, all")
    # fig.legend(loc="center left")
    # fig.show()
    # next we provide an efficient tool for selecting the training and testing step

    #GUI selections:
    speed_filter(projectPath.folder,overWrite=False)
    modify_feature_forBestTestSet(projectPath.folder,overWrite=False)

    windowSizeMS = 7*36
    trainLosses = trainer.train(onTheFlyCorrection=True,windowsizeMS=252)

    path_to_code = os.path.join(projectPath.folder, "../../neuroEncoders/transformData")
    # linearizationFunction = lambda x: doubleArmMazeLinearization(x,scale=True,path_to_folder=path_to_code)
    behave_data = getBehavior(projectPath.folder,getfilterSpeed=False)
    maxPos = np.max(behave_data["Positions"][np.logical_not(np.isnan(np.sum(behave_data["Positions"],axis=1)))])
    behavePos = behave_data["Positions"]
    umazeLinearizer = UMazeLinearizer(projectPath.folder)
    umazeLinearizer.verifyLinearization(behavePos/maxPos,projectPath.folder)
    linearizationFunction = umazeLinearizer.pykeopsLinearization
    trainer.fix_linearizer(umazeLinearizer.mazepoints,umazeLinearizer.tsProj)
    #
    # #
    # name_save = "resultTrain"
    # outputs = trainer.test(linearizationFunction,name_save,useTrain=True,onTheFlyCorrection=True)
    # # TODO: change the filtering done here!!!! <--------
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)
    # name_save = "resultTrain_full"
    # outputs = trainer.test(linearizationFunction,name_save,useTrain=True,useSpeedFilter=False,onTheFlyCorrection=True)
    # # TODO: change the filtering done here!!!! <--------
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)
    # #
    # name_save = "resultTest-Full-NoLossPredTraining"
    # outputs = trainer.test(linearizationFunction,name_save,useSpeedFilter=False,useTrain=False,onTheFlyCorrection=True,forceFirstTrainingWeight=True)
    # behave_data = getBehavior(projectPath.folder, getfilterSpeed=True)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)
    #
    # name_save = "resultTest-NoLossPredTraining"
    # outputs = trainer.test(linearizationFunction,name_save,useTrain=False,onTheFlyCorrection=True,forceFirstTrainingWeight=True)
    # behave_data = getBehavior(projectPath.folder, getfilterSpeed=True)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)
    #
    # name_save = "resultTest"
    # outputs = trainer.test(linearizationFunction,name_save,useTrain=False,onTheFlyCorrection=True)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)
    # # #
    # name_save = "resultTest_full"
    # outputs = trainer.test(linearizationFunction,name_save,useTrain=False,useSpeedFilter=False,onTheFlyCorrection=True)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # performancePlots.linear_performance(outputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)
    #

    #Study outputs of neural networks:
    # trainer.study_CNN_outputs(batch=True, forceFirstTrainingWeight=False,
    #                   useSpeedFilter=False, useTrain=False, onTheFlyCorrection=True)

    # trainer.study_uncertainty_estimate(batch=True, forceFirstTrainingWeight=False,
    #                                              useSpeedFilter=True, useTrain=False, onTheFlyCorrection=True)
    # trainer.study_uncertainty_estimate(linearizationFunction,batch=True, forceFirstTrainingWeight=False,
    #                                              useSpeedFilter=False, useTrain=False, onTheFlyCorrection=True)
    # trainer.fit_uncertainty_estimate(linearizationFunction,batch=True, forceFirstTrainingWeight=False,
    #                                              useSpeedFilter=False, useTrain=True, onTheFlyCorrection=True)

    # predPos = outputs["featurePred"]
    # truePos =  outputs["featureTrue"]
    #
    # fig,ax = plt.subplots(2,1)
    # ax[0].plot(truePos[:,0],truePos[:,1])
    # ax[1].plot(predPos[:, 0], predPos[:, 1])
    # fig.show()

    # ## Bayesian decoding

    #decodebayes.bayesian_parameter_grid_search(projectPath, cluster_data, behavior_data)

    # print("training Bayesian decoder from spike sorting ")
    trainerBayes = decodebayes.Trainer(projectPath)
    print("ordering cluster spike by prefered position")
    trainerBayes.get_spike_ordered_by_prefPos(linearizationFunction)
    #
    # # we can compare the waveforms resulting in the offline (before spike sorting) filtering
    # # and the filtering for the online strategy:
    from importData import  ImportClusters
    behavior_data = ImportClusters.getBehavior(projectPath.folder,getfilterSpeed=True)
    #
    from importData.compareSpikeFiltering import waveFormComparator
    wfc = waveFormComparator(projectPath,params,behavior_data, useTrain=True)
    wfc.save_alignment_tools(trainerBayes,linearizationFunction)
    wfc = waveFormComparator(projectPath,params,behavior_data, useTrain=False)
    wfc.save_alignment_tools(trainerBayes,linearizationFunction)
    wfc = waveFormComparator(projectPath,params,behavior_data, useTrain=False,useSleep=True)
    wfc.save_alignment_tools(trainerBayes,linearizationFunction)
    # trainer.fit_uncertainty_estimate(linearizationFunction,batch=True, forceFirstTrainingWeight=False,
    #                                              useSpeedFilter=False, useTrain=False, onTheFlyCorrection=True)

    behavior_data = getBehavior(projectPath.folder, getfilterSpeed=True)
    # trainer.sleep_decoding(linearizationFunction, [], behavior_data, saveFolder="resultSleep", batch=True,
    #                        batch_size=52)

    from resultAnalysis import paperFigures
    from resultAnalysis import paperFigure_sleep
    # paperFigure_sleep.paperFigure_sleep(projectPath, params, linearizationFunction, behavior_data, "PreSleep")
    paperFigures.paperFigure(trainerBayes, projectPath, params, linearizationFunction, usetrain=False)


    # trainer.sleep_decoding(linearizationFunction,[],behavior_data,saveFolder="resultSleep",batch=True,batch_size=52)

    # trainerBayes.bandwidth = 0.05
    # bayesMatrices = trainerBayes.train(behavior_data, cluster_data)
    # performancePlots.quick_place_field_display(bayesMatrices)

    # trainerBayes = decodebayes.Trainer(projectPath)
    # spikeMat_labels,spikeMat_times,linearPreferredPos = trainerBayes.get_spike_ordered_by_prefPos(linearizationFunction)
    # from importData.compareSpikeFiltering import waveFormComparator
    # wfc = waveFormComparator(projectPath, params, behavior_data)
    # wfc.save_alignment_tools(trainerBayes, linearizationFunction)

    # from resultAnalysis.NNpredAnalysis import compare_nn_popVector
    # compare_nn_popVector(trainerBayes,projectPath, params, linearizationFunction, usetrain=False)
    #
    # name_save = "resultSleep"
    # outputs_sleep_decoding = trainer.sleep_decoding(linearizationFunction,[],behavior_data,saveFolder=name_save,batch=True,batch_size=52)
    #
    # trainer.fit_uncertainty_estimate(linearizationFunction,batch=True, forceFirstTrainingWeight=False,
    #                                              useSpeedFilter=False, useTrain=True, onTheFlyCorrection=True) #spikeMat_labels,spikeMat_times,trainerBayes.linearPreferredPos
    #
    # print("loading spike sorting")


    # for idGroup,rateGroup in enumerate(bayesMatrices["Spike_positions"]):
    #     fig,ax = plt.subplots(len(rateGroup)//4+1,4)
    #     for id in range(len(rateGroup)//2+1):
    #         for idy in range(4):
    #             if 4*id+idy<len(rateGroup):
    #                 ax[id, idy].scatter(behavior_data["Positions"][1:-1:100,0],behavior_data["Positions"][1:-1:100,1],c="grey",alpha=0.1,s=1)
    #                 ax[id,idy].scatter(rateGroup[4*id+idy][:,0],rateGroup[4*id+idy][:,1],s=1,c="red")
    #                 ax[id,idy].set_title("mutual_info:"+str(round(bayesMatrices["Mutual_info"][idGroup][4*id+idy],2)))
    #     [[a2.axes.get_yaxis().set_visible(False) for a2 in a] for a in ax]
    #     [[a2.axes.get_xaxis().set_visible(False) for a2 in a] for a in ax]
    #     fig.suptitle("Group: " +str(idGroup))
    #     fig.tight_layout()
    #     fig.show()

    # outputsBayes = trainerBayes.test_Pierre(bayesMatrices,behavior_data,cluster_data,windowSize=wsBest)
    # name_save = "bayesianDecoding"
    # performancePlots.linear_performance_bayes(outputsBayes, linearizationFunction, os.path.join(projectPath.resultsPath,name_save,"filter"), behavior_data, probaLim=0.7)
    # performancePlots.linear_performance_bayes(outputsBayes, linearizationFunction,
    #                                           os.path.join(projectPath.resultsPath, name_save, "nofilter"),
    #                                           behavior_data, probaLim=0)
    # performancePlots.compare_bayes_network(outputs, outputsBayes, behavior_data)

    # First of, we compare the sleep decoding using the same temporal window for the bayesian network and the neural network
    # outputsBayesSleep = trainerBayes.sleep_decoding(bayesMatrices,behavior_data,cluster_data,windowSize=windowSize)
    #
    # sleep decoding
    # name_save = "resultSleep"
    # outputs_sleep_decoding = trainer.sleep_decoding(linearizationFunction,[],behave_data,saveFolder=name_save,batch=True,batch_size=52)
    ##Next up: we compare the decoding of the bayesian network and the neural network during sleep.
    # performancePlots.compare_linear_sleep_predictions(outputs_sleep_decoding,outputsBayesSleep)


    ## Let us compare the training of 2 bayesian algorithm, with slightly different window size:
    # trainerBayes2 = decodebayes.Trainer(projectPath)
    # trainerBayes2.bandwidth = 0.06
    # bayesMatrices2 = trainerBayes2.train(behavior_data, cluster_data)
    # outputsBayes = trainerBayes2.test_Pierre(bayesMatrices2,behavior_data,cluster_data,windowSize=20)
    # proba = outputsBayes["inferring"][:,2]
    # fig,ax = plt.subplots()
    # ax.hist(proba,bins=100)
    # fig.show()

    #
    # outputsBayes2sleep = trainerBayes2.sleep_decoding(bayesMatrices2, behavior_data, cluster_data, windowSize=0.1)
    # performancePlots.compare_sleep_predictions(outputsBayes2sleep, outputsBayesSleep,"bayesian 0.3","bayesian 0.1")


    ### Next: We compare the convnets feature outputs to the bayesian network decoding.



    ## Test with shuflling of spikes in the window
    # We ask here if the algorithm uses the order of the spikes in the window
    params.shuffle_spike_order = True
    trainer.model = trainer.mybuild(trainer.get_Model(),modelName="spikeOrdershufflemodel.png")
    name_save = "result_spikeorderinwindow_shuffle"
    output_shuffle_spikeorder = trainer.test(linearizationFunction,name_save,onTheFlyCorrection=True)
    performancePlots.linear_performance(output_shuffle_spikeorder,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    performancePlots.linear_performance(output_shuffle_spikeorder,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)

    # name_save = "result_spikeorderinwindow_shuffle_train"
    # output_shuffle_spikeorder = trainer.test(linearizationFunction,name_save,useTrain=True,onTheFlyCorrection=True)
    # performancePlots.linear_performance(output_shuffle_spikeorder,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1,behave_data=behave_data)
    # performancePlots.linear_performance(output_shuffle_spikeorder,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1,behave_data=behave_data)

    # params.shuffle_spike_order = False
    # params.shuffle_convnets_outputs = True
    # trainer.model = trainer.mybuild(trainer.get_Model(),modelName="convNetOutputshufflemodel.png")
    # name_save = "result_convOutputs_shuffle"
    # outputs_shuffle_convoutputs = trainer.test(linearizationFunction,name_save,onTheFlyCorrection=True)
    # performancePlots.linear_performance(outputs_shuffle_convoutputs,os.path.join(projectPath.resultsPath,name_save,"nofilter"),filter=1)
    # performancePlots.linear_performance(outputs_shuffle_convoutputs,os.path.join(projectPath.resultsPath,name_save,"filter"), filter=0.1)


if __name__=="__main__":
    # In this architecture we use a 2.0 tensorflow backend, predicting solely the position.
    # I.E without using the simpler feature strategy based on stratified spaces
    main()
