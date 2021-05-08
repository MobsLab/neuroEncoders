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

        self.resultsPath = os.path.join(self.folder,"result_Transformer_test")
        if not os.path.isdir(self.folder + 'dataset'):
            os.makedirs(self.folder + 'dataset')
        if not os.path.isdir(self.folder + 'graph'):
            os.makedirs(self.folder + 'graph')
        if not os.path.isdir(self.folder + 'results'):
            os.makedirs(self.folder + 'results')
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
        self.nEpochs = 100
        self.learningTime = detector.learningTime()
        self.windowLength = windowSize # in seconds, as all things should be

        ### from units encoder params
        self.validCluWindow = 0.0005
        self.kernel = 'epanechnikov'
        self.bandwidth = 0.1
        self.masking = 20

        ## Model dimensionality of the convnet output
        # We want d_model to be a n_output_variable-root of an integer.
        # i.e d_model**(1/n_output_variable) integer, integer correspond to the number of bin
        # over each axis of the output_variable space (but together the discretization form a grid mesh)
        # If we want bin = 20 for example we can use:

        self.d_model = 32**(self.dim_output)
        # assert self.d_model//self.nGroups == self.d_model/self.nGroups
        self.nFeatures = 128 #int(self.d_model/self.nGroups)
        # The output of each convernet is concatenated to form the input to the model.
        ### full Tranformer params
        self.num_layers_encoder_transformer = 4
        self.num_heads_transformer = 8 #to change back to 128
        self.dff = 512
        self.transformerDropout = 0.1
        self.num_layers_decoder_transformer = 8

        # While the transformer will operate with a wider language, in a higher dimensional space
        # we can force the loss to be a cross-entropy loss by using a restricted vocabulary
        # if this voc is too large, the network does not seem to be able to learn.
        self.placeCellVocSize = 32**(self.dim_output)

        #For the positional encoding: the maximal sequence length
        # I.e the maximal number of spike we could detect
        # Spike detection is made by block of 32 timestep
        # But each group is detected in parallel so we have to multiply this by the nnumber of groups
        self.max_nb_spike_in_window = np.ceil(self.windowLength*20000/32*self.nGroups)
        # TODO: detect sampling rate in xml file (here 20000)

        self.batch_size = 52 #previously 52

        self.learningRates = [0.001] #  0.00003  ,    0.00003, 0.00001]
        self.lossLearningRate = 0.00003
        self.lossActivation = None

        self.usingMixedPrecision = True # this boolean indicates weither tensorflow uses mixed precision
        # ie enforcing float16 computations whenever possible
        # According to tf tutorials, we can allow that in most layer except the output for unclear reasons linked to gradient computations



def main():
    from importData import rawDataParser
    import  nnUtils
    import tensorflow.keras.mixed_precision as mixed_precision

    # to set as env variable: TF_GPU_THREAD_MODE=gpu_private

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

    # OPTIMIZATION of tensorflow
    #tf.config.optimizer.set_jit(True) # activate the XLA compilation
    # mixed_precision.experimental.set_policy('mixed_float16')
    params.usingMixedPrecision = False

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

    # Let us first find a simplification of the network topology: we gather all positions point and
    # fit splines




    # Training, testing, and preparing network for online setup
    if mode=="full":
        import waveformTranslator as Training
    elif mode=="decode":
        from decoder import decodeTraining as Training
    # todo: modify this loading of code files as we changed names!!
    trainer = Training.WaveformTranslator(projectPath, params)
    # trainLosses = trainer.train()
    # df = pd.DataFrame(trainLosses)
    # df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "lossTraining.csv"))
    # fig,ax = plt.subplots()
    # ax.plot(trainLosses[:,0])
    # plt.show()

    # trainer.outputEmbeding.sigma *= 0.1
    # res = trainer.outputEmbeding(tf.squeeze(trainer.outputEmbeding.prefered_feature_tensor,axis=1))
    # res_ex = res[0,:,:]
    # res_ex_meanless = res_ex   #tf.reduce_mean(res_ex,axis=1)
    # testCosine = tf.keras.losses.CosineSimilarity(axis=-1,reduction='none')
    # res_loss = testCosine(res_ex_meanless[-1,:],res_ex_meanless[:,:])
    #
    # pos_pref = trainer.outputEmbeding.prefered_feature_tensor[0,0,:,:]
    # res_loss_sorted = np.sort(res_loss)
    # fig,ax = plt.subplots(3,1)
    # # ax[0].scatter(pos_pref[:, 0], pos_pref[:, 1], c=(tf.reduce_mean(res_ex, axis=1)))
    # # ax[1].plot(tf.reduce_mean(res_ex, axis=1))
    # ax[0].scatter(pos_pref[:,0],pos_pref[:,1],c=(res_loss+1))
    # ax[1].plot(res_loss_sorted)
    # ax[2].imshow(res_ex)
    # plt.show()
    #
    #
    # res_ex_meanless = (res_ex  - tf.reduce_mean(res_ex,axis=0))/tf.math.reduce_std(res_ex,axis=0)
    # corr = np.transpose(res_ex_meanless)*res_ex_meanless
    # corr_0 = corr[210,:]
    # fig, ax = plt.subplots(3, 1)
    # ax[0].scatter(pos_pref[:, 0], pos_pref[:, 1], c=corr_0)
    # ax[1].plot(np.sort(corr_0))
    # ax[2].imshow(corr)
    # plt.show()



    outputs = trainer.test()

    featurePred = outputs["featurePred"]
    featureTrue= outputs["featureTrue"]
    probaTrue = outputs["manifoldProbaTrue"]
    probaPred = outputs["manifoldProbaPred"]
    lossPred = outputs["lossFromOutputLoss"]

    featurePredSelec = np.where((probaTrue[:,0])>0.5,featurePred[:,0],featurePred[:,1])
    featureTrueSelec = np.where((probaTrue[:,0])>0.5,featureTrue[:,0],featureTrue[:,1])

    errorProba = np.where((probaTrue[:,0])>0.5,probaPred[:,0]>0.5,probaPred[:,0]<=0.5)
    np.sum(errorProba)/errorProba.shape[0]

    fig,ax = plt.subplots(4,1)
    ax[0].scatter(featurePredSelec,featureTrueSelec,alpha=0.03)
    #ax[0].scatter(featurePred[:, 1], featureTrue[:, 1])
    ax[1].hist(featureTrueSelec,bins=100,density=True,histtype="step")
    ax[1].hist(featurePredSelec, bins=100,density=True,histtype="step")
    ax[2].plot(featureTrueSelec,alpha=0.5)
    ax[2].plot(featurePredSelec,alpha=0.5)
    # ax[2].plot(featureTrueSelec-featurePredSelec,c="black",alpha=0.5)
    # ax[2].plot(np.square(np.sin(np.pi*(featureTrueSelec - featurePredSelec))), c="red", alpha=0.5)
    #ax[2].plot(featurePredSelec, c="red",alpha=0.5)
    ax[3].plot(probaTrue[:,0], c="black",alpha=0.5)
    ax[3].plot(probaPred[:,0], c="red",alpha=0.5)
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "ExamplePrediction.png"))

    #We then save the output feature so that they are loaded and converted into position in Julia.
    df = pd.DataFrame(featurePred)
    try:
        os.mkdir(os.path.join(projectPath.resultsPath,"resultInference"))
    except:
        pass
    df.to_csv(os.path.join(projectPath.resultsPath,"resultInference","featurePred.csv"))
    df = pd.DataFrame(featureTrue)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featureTrue.csv"))
    df = pd.DataFrame(probaPred)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "probaPred.csv"))
    df = pd.DataFrame(probaTrue)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "probaTrue.csv"))
    df = pd.DataFrame(lossPred)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "lossPred.csv"))

    fig,ax = plt.subplots(2,1)
    ax[0].plot(trainLosses[:,0],c="blue")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Manifold loss",c="blue")
    ax2 = ax[0].twinx()
    ax2.plot(trainLosses[:, 1],c="red")
    ax2.set_ylabel("KL divergence of manifold \n predicted and true proba", c= "red")
    ax[1].plot(trainLosses[:,2],c="black")
    ax[1].set_ylabel("Manifold loss Prediction Error")
    fig.show()
    plt.savefig(os.path.join(projectPath.folder, "results", "loss.png"))

if __name__=="__main__":
    main()

    print()
    print()
    print('Encoding over.')
