
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp
from importData.ImportClusters import getBehavior
from importData.rawDataParser import  inEpochsMask
import pykeops
from tqdm import tqdm
from matplotlib.colors import  LinearSegmentedColormap

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#ffffff'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def paperFigure(trainerBayes,projectPath,params,linearizationFunction,usetrain=True):
    # Predictions of the neural network with uncertainty estimate are provided

    euclidData_train  = np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_fit",
                                                   "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
    timePreds_train  = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "timePreds.csv")).values[:, 1],
                         dtype=np.float32)
    truePosFed_train  = pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "truePosFed.csv")).values[:, 1:]
    windowmask_speed_train = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "windowmask_speed.csv")).values[:, 1:],
                          dtype=np.float32)
    windowmask_speed_train = windowmask_speed_train[:,0].astype(np.bool)

    euclidData = np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_test",
                                                   "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
    timePreds = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_test", "timePreds.csv")).values[:, 1],
                         dtype=np.float32)
    truePosFed = pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_test", "truePosFed.csv")).values[:, 1:]
    windowmask_speed = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_test", "windowmask_speed.csv")).values[:, 1:],
                          dtype=np.float32)
    windowmask_speed = windowmask_speed[:,0].astype(np.bool)
    output_test = [
        np.reshape(euclidData, [-1, params.nb_eval_dropout, params.batch_size, params.dim_output])]
    projectedPos, linearPos = linearizationFunction(euclidData.astype(np.float64))
    linearPos = np.reshape(linearPos, output_test[0].shape[0:3])
    medianLinearPos = np.median(linearPos, axis=1)
    medianLinearPos = np.reshape(medianLinearPos, [np.prod(medianLinearPos.shape[0:2])])
    trueProjPos, trueLinearPos = linearizationFunction(truePosFed)
    trueProjPos_train, trueLinearPos_train = linearizationFunction(truePosFed_train)

    cm = plt.get_cmap('turbo')
    fig,ax = plt.subplots()
    ax.scatter(truePosFed[:,0],truePosFed[:,1],c=cm(trueLinearPos))
    fig.show()



    linearTranspose = np.transpose(linearPos, axes=[0, 2, 1])
    linearTranspose = linearTranspose.reshape(
        [linearTranspose.shape[0] * linearTranspose.shape[1], linearTranspose.shape[2]])
    histPosPred = np.stack([np.histogram(np.abs(linearTranspose[id, :] - np.median(linearTranspose[id, :])),
                                         bins=np.arange(0, stop=1, step=0.01))[0] for id in
                            range(linearTranspose.shape[0])])

    # CSV files helping to align the pop vector from spike used in spike sorting
    # with predictions from spike used by the NN are also provided.
    spikeMat_window_popVector_train = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath,"dataset", "alignment", "waketrain", "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32)
    spikeMat_times_window_train = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "spikeMat_times_window.csv")).values[:,1:],dtype=np.float32)
    meanTimeWindow_train = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "meanTimeWindow.csv")).values[:,1:],dtype=np.float32)
    startTimeWindow_train = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "startTimeWindow.csv")).values[:,1:],dtype=np.float32)
    lenInputNN_train = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "lenInputNN.csv")).values[:,1:],dtype=np.float32)

    spikeMat_window_popVector = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath,"dataset", "alignment", "waketest", "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32)
    spikeMat_times_window = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketest", "spikeMat_times_window.csv")).values[:,1:],dtype=np.float32)
    meanTimeWindow = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketest", "meanTimeWindow.csv")).values[:,1:],dtype=np.float32)
    startTimeWindow = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketest", "startTimeWindow.csv")).values[:,1:],dtype=np.float32)
    lenInputNN = np.array(
        pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketest", "lenInputNN.csv")).values[:,1:],dtype=np.float32)

    ### Load the NN prediction without using noise:

    linearNoNoisePos_train = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTrain_full", "linearPred.csv")).values[:, 1:],dtype=np.float32)
    featureNoNoisePosPred_train = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTrain_full", "featurePred.csv")).values[:, 1:],dtype=np.float32)
    featureNoNoisePosTrue_train = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTrain_full", "featureTrue.csv")).values[:, 1:],dtype=np.float32)
    timePredsNoNoise_train = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTrain_full", "timeStepsPred.csv")).values[:, 1:],dtype=np.float32)
    lossPredNoNoise_train = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTrain_full", "lossPred.csv")).values[:, 1:],dtype=np.float32)

    linearNoNoisePos = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTest_full", "linearPred.csv")).values[:,1:],dtype=np.float32)
    featureNoNoisePosPred = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTest_full", "featurePred.csv")).values[:, 1:],dtype=np.float32)
    featureNoNoisePosTrue = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTest_full", "featureTrue.csv")).values[:, 1:],dtype=np.float32)
    timePredsNoNoise = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTest_full", "timeStepsPred.csv")).values[:, 1:],dtype=np.float32)
    lossPredNoNoise = np.array(pd.read_csv(
        os.path.join(projectPath.resultsPath, "resultTest_full", "lossPred.csv")).values[:, 1:],dtype=np.float32)


    behavior_data = getBehavior(projectPath.folder, getfilterSpeed=True)

    trainerBayes.bandwidth = 0.05
    from importData import  ImportClusters
    cluster_data = ImportClusters.load_spike_sorting(projectPath)
    bayesMatrices = trainerBayes.train(behavior_data, cluster_data)

    from importData.rawDataParser import  get_params
    _, samplingRate, _ = get_params(projectPath.xml)

    placeFieldSort = trainerBayes.linearPosArgSort
    prefPos = trainerBayes.linearPreferredPos

    #quickly obtain bayesian decoding:
    linearpos_bayes_varying_window = []
    proba_bayes_varying_window = []
    for window_size in [1,3,7,14]:
        outputsBayes = trainerBayes.test_as_NN(
            startTimeWindow[:trueProjPos.shape[0]].astype(dtype=np.float64) / samplingRate,
            bayesMatrices, behavior_data,cluster_data, windowSize=window_size * 0.036, masking_factor=1000, useTrain=usetrain)  # using window size of 36 ms!
        pos = outputsBayes["inferring"][:, 0:2]
        _, linearBayesPos = linearizationFunction(pos)
        linearpos_bayes_varying_window +=[linearBayesPos]
        proba = outputsBayes["inferring"][:, 2]
        proba_bayes_varying_window += [proba]

    # Get histogram of predictions for every window:
    histlinearPosPred = np.stack(
        [np.histogram(linearTranspose[id, :], bins=np.arange(0, stop=1, step=0.01), density=True)[0] for id in
         range(linearTranspose.shape[0])])
    histlinearPosPred_density = histlinearPosPred / (np.sum(histlinearPosPred, axis=1)[:, None])
    def xlogx(x):
        y = np.zeros_like(x)
        y[np.greater(x, 0)] = np.log(x[np.greater(x, 0)]) * (x[np.greater(x, 0)])
        return y
    # let us compute the absolute error over each test Epochs:
    # absError_epochs = []
    # absError_epochs_mean = []
    # names = []
    # entropies_epochs_mean = []
    # entropies_epochs = []
    # proba_epochs = []
    # proba_epochs_mean = []
    # error_bayes_epochs = []
    # testEpochs = behavior_data["Times"]["testEpochs"].copy()
    # keptSession = behavior_data["Times"]["keptSession"]
    # sessNames = behavior_data["Times"]["sessionNames"].copy()
    # for idk, k in enumerate(keptSession.astype(np.bool)):
    #     if not k:
    #         sessNames.remove(behavior_data["Times"]["sessionNames"][idk])
    # for x in behavior_data["Times"]["sleepNames"]:
    #     for id2, x2 in enumerate(sessNames):
    #         if x == x2:
    #             sessNames.remove(x2)
    #             # testEpochs[id2 * 2] = -1
    #             # testEpochs[id2 * 2 + 1] = -1
    # # testEpochs = testEpochs[np.logical_not(np.equal(testEpochs, -1))]
    # for i in range(int(len(testEpochs) / 2)):
    #     epoch = testEpochs[2 * i:2 * i + 2]
    #     maskEpoch = inEpochsMask(timePreds, epoch)
    #     maskTot = maskEpoch*(windowmask_speed)
    #     if np.sum(maskTot) > 0:
    #         absError_epochs_mean += [
    #             np.mean(np.abs(trueLinearPos[maskTot] - np.mean(linearTranspose[maskTot], axis=1)))]
    #         absError_epochs += [np.abs(trueLinearPos[maskTot] - np.mean(linearTranspose[maskTot], axis=1))]
    #         names += [sessNames[i]]
    #         entropies_epochs_mean += [np.mean(np.sum(-xlogx(histlinearPosPred_density[maskTot, :]), axis=1))]
    #         entropies_epochs += [np.sum(-xlogx(histlinearPosPred_density[maskTot, :]), axis=1)]
    #         proba_epochs += [proba[maskTot]]
    #         proba_epochs_mean += [np.mean(proba[maskTot])]
    #         error_bayes_epochs += [np.mean(np.abs(linearBayesPos[maskTot]-trueLinearPos[maskTot]))]
    #     else:
    #         print(sessNames[i])
    habEpochMask = inEpochsMask(timePreds,behavior_data["Times"]["testEpochs"][0:4])
    habEpochMaskandSpeed = (habEpochMask) * (windowmask_speed)
    nothabEpochMask = np.logical_not(habEpochMask)
    nothabEpochMaskandSpeed = np.logical_not(habEpochMask)* (windowmask_speed)

    # entropy normalization:
    ### ================ Let us Normalize the entropy given a predicted position ================
    ent = np.sum(-xlogx(histlinearPosPred_density), axis=1)
    trainingEntropies = ent[habEpochMaskandSpeed]
    trainingLinearPos = medianLinearPos[habEpochMaskandSpeed]
    binsLinearPos = np.arange(0,stop=np.max(linearPos)+0.1,step=0.1)
    trainingEntropies_givenPos = [trainingEntropies[np.greater_equal(trainingLinearPos,binsLinearPos[id])
                                                    *np.less(trainingLinearPos,binsLinearPos[id+1])] for id in range(len(binsLinearPos)-1)]
    pos_given_pos = [trainingLinearPos[np.greater_equal(trainingLinearPos,binsLinearPos[id])
                                                    *np.less(trainingLinearPos,binsLinearPos[id+1])] for id in range(len(binsLinearPos)-1)]
    trainingEntropies_givenPos_normalize = [(np.array(t)-np.mean(t))/np.std(t) for t in trainingEntropies_givenPos]
    normalized_entropied = np.zeros_like(ent)
    for id in range(len(binsLinearPos)-1):
        ent_given_pos = ent[np.greater_equal(medianLinearPos,binsLinearPos[id])
                                                    *np.less(medianLinearPos,binsLinearPos[id+1])]
        normalized_entropied[np.greater_equal(medianLinearPos,binsLinearPos[id])
                            *np.less(medianLinearPos,binsLinearPos[id+1])] =  (ent_given_pos - np.mean(trainingEntropies_givenPos[id]))/np.std(trainingEntropies_givenPos[id])


    from tqdm import tqdm

    linearpos_NN_varying_window_median = []
    linearpos_NN_varying_window_argmax = []
    histlinearPosPred_varying_window = []
    binsLinearPosHist  =np.arange(0, stop=1, step=0.01)
    for i in tqdm([1,3,7,14]):
        histlinearPosPred_varying_window += [np.array([np.mean(histlinearPosPred_density[id:id+i,:],axis=0) for id in range(histlinearPosPred_density.shape[0])])]
        linearpos_NN_varying_window_median += [np.array([np.median(np.ravel(linearTranspose[id:id+i,:])) for id in range(histlinearPosPred_density.shape[0])])]
        linearpos_NN_varying_window_argmax += [np.array([ binsLinearPosHist[np.argmax(np.mean(histlinearPosPred_density[id:id+i,:],axis=0))] for id in range(histlinearPosPred_density.shape[0])])]

    max_proba_NN = np.max(histlinearPosPred_density, axis=1)
    ## Let us normalize the probability:
    trainingProba_givenPos = [max_proba_NN[np.greater_equal(linearpos_NN_varying_window_argmax[0], binsLinearPos[id])
                                           * np.less(linearpos_NN_varying_window_argmax[0], binsLinearPos[id + 1])] for
                              id in range(len(binsLinearPos) - 1)]
    normalized_proba = np.zeros_like(max_proba_NN)
    for id in range(len(binsLinearPos) - 1):
        proba_given_pos = max_proba_NN[np.greater_equal(linearpos_NN_varying_window_argmax[0], binsLinearPos[id])
                                       * np.less(linearpos_NN_varying_window_argmax[0], binsLinearPos[id + 1])]
        normalized_proba[np.greater_equal(linearpos_NN_varying_window_argmax[0], binsLinearPos[id])
                         * np.less(linearpos_NN_varying_window_argmax[0], binsLinearPos[id + 1])] = (
                                                                                                                proba_given_pos - np.mean(
                                                                                                            trainingProba_givenPos[
                                                                                                                id])) / np.std(
            trainingProba_givenPos[id])



    fig,ax =plt.subplots(9,9)
    for i in range(79):
        pc_spiking_train = np.greater_equal(spikeMat_window_popVector_train[:timePreds_train.shape[0], i+1], 1)
        pc_spiking = np.greater_equal(spikeMat_window_popVector[:timePreds.shape[0], i + 1], 1)
        # pos = np.where(np.equal(placeFieldSort,i))[0][0]
        pos = placeFieldSort[i] #the old position of the place cell now at position i
        ax[pos//9,pos%9].scatter(truePosFed[:, 0], truePosFed[:, 1], c="grey", alpha=0.05, s=0.5)
        ax[pos//9,pos%9].scatter(truePosFed_train[pc_spiking_train,0],truePosFed_train[pc_spiking_train,1],c="red",s=0.5)
        ax[pos//9,pos%9].scatter(truePosFed[pc_spiking, 0], truePosFed[pc_spiking, 1],
                                      c="red", s=0.5)
    fig.show()

    # fig,ax =plt.subplots()
    # spikings = [np.equal(spikeMat_window_popVector_train[:timePreds_train.shape[0], pcId+1],e) for e in np.arange(1,5,1)]
    # spikTimes = [timePreds_train[spiking] for spiking in spikings]
    # ax.scatter(timePreds_train,trueLinearPos_train,c="grey",s=1,alpha=0.2)
    # cm = plt.get_cmap("tab20")
    # [ax.scatter(spikTimes[id],trueLinearPos_train[spikings[id]],c=cm(id),s=4) for id in range(4)]
    # fig.show()

    spikeMat_popVector_hab = np.zeros([timePreds_train.shape[0]+timePreds.shape[0],
                                       spikeMat_window_popVector.shape[1]])
    spikeMat_popVector_hab[0:timePreds_train.shape[0],:] = spikeMat_window_popVector_train[:timePreds_train.shape[0]]
    spikeMat_popVector_hab[timePreds_train.shape[0]:, :] = spikeMat_window_popVector[:timePreds.shape[0]]
    timePreds_hab = np.concatenate([timePreds_train,timePreds])
    trueLinearPos_hab = np.concatenate([trueLinearPos_train,trueLinearPos])

    for target in [5,16,49,46,68,71]:
        pcId = np.where(np.equal(placeFieldSort,target))[0][0] # newID_PlaceToStudy[0]
        prefPosPC = prefPos[pcId]
        #let us compute the tuning curve of the neuron for the linear variable:
        binsLinearPos = np.arange(0,1,step=0.01)
        pcFiring = spikeMat_popVector_hab[:, pcId+1]
        firing = np.array([np.sum(pcFiring[np.greater_equal(trueLinearPos_hab,binsLinearPos[id])*
                                           np.less(trueLinearPos_hab,binsLinearPos[id+1])])/np.sum(np.greater_equal(trueLinearPos_hab,binsLinearPos[id])*
                                           np.less(trueLinearPos_hab,binsLinearPos[id+1]))
            for id in range(len(binsLinearPos)-1)])
        tc_pc = firing/(0.032)

        pcFiring_test = spikeMat_window_popVector[:timePreds.shape[0], pcId + 1]
        #When making a prediction around the pcFiring, the proba should be reflecting the firing rate of the cell...
        predAroundPrefPos =np.greater(pcFiring_test,0)
        # cm = plt.get_cmap("binary")
        # fig,ax = plt.subplots(1,2)
        # ax[0].scatter(linearpos_NN_varying_window_argmax[0][predAroundPrefPos],
        #            (pcFiring_test/np.sum(spikeMat_window_popVector[:timePreds.shape[0],:],axis=1))[predAroundPrefPos],s=12,
        #            c=cm(max_proba_NN[predAroundPrefPos]/np.max(max_proba_NN[predAroundPrefPos])),edgecolors="black",linewidths=0.2)
        # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,np.max(max_proba_NN[predAroundPrefPos])),cmap=cm),label="NN probability \n estimate")
        # ax[0].set_xlabel("predicted position")
        # ax[0].set_ylabel("Number of spike \n relative to total number of spike \n in 36ms window")
        # ax[1].plot(binsLinearPos[:-1],tc_pc)
        # ax[1].set_xlabel("linear position")
        # ax[1].set_ylabel("firing rate")
        # fig.tight_layout()
        # fig.show()

        error = np.abs(linearNoNoisePos-trueLinearPos)
        #using predicted Error:
        normalize01 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
        cm = plt.get_cmap("gray")
        fig,ax = plt.subplots()
        ax.scatter(linearpos_NN_varying_window_argmax[0][predAroundPrefPos],
                   (pcFiring_test/np.sum(spikeMat_window_popVector[:timePreds.shape[0],:],axis=1))[predAroundPrefPos],s=12,
                   c=cm(normalize01(lossPredNoNoise[predAroundPrefPos])),edgecolors="black",linewidths=0.2)
        ax.scatter(linearpos_NN_varying_window_argmax[0][predAroundPrefPos*np.greater(error,0.5)],
                   (pcFiring_test/np.sum(spikeMat_window_popVector[:timePreds.shape[0],:],axis=1))[predAroundPrefPos*np.greater(error,0.5)],s=24,
                   c="white",alpha=0.1,edgecolors="red",linewidths=0.4)

        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(lossPredNoNoise[predAroundPrefPos]),np.max(lossPredNoNoise[predAroundPrefPos])),cmap=cm)
                     ,label="NN log error \n estimate")
        ax.set_xlabel("predicted position")
        ax.set_ylabel("Number of spike \n relative to total number of spike \n in 36ms window")
        at = ax.twinx()
        at.plot(binsLinearPos[:-1],tc_pc,c="navy",alpha=0.5)
        # ax[0].set_xlabel("linear position")
        at.set_ylabel("firing rate",color="navy")
        fig.tight_layout()
        fig.show()


    ## Figure 1: on habituation set, speed filtered, we plot an example of bayesian and neural network decoding
    cm = plt.get_cmap("tab20c")
    time_filter = np.greater(timePreds,950)*np.less(timePreds,1100)
    time_filter = np.ones(timePreds.shape[0]).astype(np.bool)#*np.less(normalized_entropied,0)
    fig,ax = plt.subplots(4,2,sharex=True,figsize=(15,10),sharey=True)

    [a.plot(timePreds[time_filter],trueLinearPos[time_filter],c="black",alpha=0.3) for a in ax[:,0]]
    ax[0,0].scatter(timePreds[time_filter],linearpos_NN_varying_window_argmax[0][time_filter],c=cm(4),alpha=0.9,label="36ms non-overlapping window",s=1)
    ax[0,0].set_title("Neural network decoder \n 36ms non-overlapping window")
    ax[1,0].scatter(timePreds[time_filter], linearpos_NN_varying_window_argmax[1][time_filter], c=cm(5), alpha=0.9,label="108ms overlapping window",s=1)
    ax[1,0].set_title("108ms overlapping window")
    ax[2,0].scatter(timePreds[time_filter], linearpos_NN_varying_window_argmax[2][time_filter], c=cm(6), alpha=0.9,label="216ms overlapping window",s=1)
    ax[2,0].set_title("252ms overlapping window")
    ax[3,0].scatter(timePreds[time_filter], linearpos_NN_varying_window_argmax[3][time_filter], c=cm(7), alpha=0.9,label="504ms overlapping window",s=1)
    ax[3,0].set_title("504ms overlapping window")

    [a.plot(timePreds[time_filter],trueLinearPos[time_filter],c="black",alpha=0.3) for a in ax[:,1]]
    ax[0,1].scatter(timePreds[time_filter],linearpos_bayes_varying_window[0][time_filter],c=cm(0),alpha=0.9,label="36ms non-overlapping window",s=1)
    ax[0,1].set_title("Bayesian decoder \n 36ms non-overlapping window")
    ax[1,1].scatter(timePreds[time_filter], linearpos_bayes_varying_window[1][time_filter], c=cm(1), alpha=0.9,label="108ms overlapping window",s=1)
    ax[1,1].set_title("108ms overlapping window")
    ax[2,1].scatter(timePreds[time_filter], linearpos_bayes_varying_window[2][time_filter], c=cm(2), alpha=0.9,label="216ms overlapping window",s=1)
    ax[2,1].set_title("252ms overlapping window")
    ax[3,1].scatter(timePreds[time_filter], linearpos_bayes_varying_window[3][time_filter], c=cm(3), alpha=0.9,label="504ms overlapping window",s=1)
    ax[3,1].set_title("504ms overlapping window")
    [a.set_ylabel("linear position") for a in ax[:,0]]
    # [a.set_ylabel("linear position") for a in ax[:,1]]
    ax[3,1].set_xlabel("time (s)")
    ax[3,0].set_xlabel("time (s)")
    # [a.set_aspect(50/2) for a in ax]
    fig.tight_layout()
    fig.show()
    if not os.path.exists(os.path.join(projectPath.resultsPath,"paperFigure")):
        os.mkdir(os.path.join(projectPath.resultsPath,"paperFigure"))
    plt.savefig(os.path.join(projectPath.resultsPath,"paperFigure","fig1.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig1.svg"))

    ## Figure 2: we plot the histograms of errors
    fig,ax = plt.subplots(4,1,sharex=True)
    window_size = []
    for i in range(4):
        error = np.abs(trueLinearPos-linearpos_bayes_varying_window[i])[habEpochMask]
        # ax[i].hist(error,color=cm(i),bins=binsLinearPosHist,alpha=0.3,density=True)
        ax[i].hist(error, color=cm(i), bins=binsLinearPosHist,histtype="step",density=True)
        ax[i].vlines(np.mean(error),0,15, color=cm(i))
        # ax[i].set_ylim(0,15)
        # ax[i].set_yscale("log")
    for j in range(4):
        error = np.abs(trueLinearPos - linearpos_NN_varying_window_argmax[j])[habEpochMask]
        # ax[j].hist(error, color=cm(j+4),bins=binsLinearPosHist,alpha=0.3,density=True)
        ax[j].hist(error, color=cm(j+4), bins=binsLinearPosHist, histtype="step",density=True)
        ax[j].vlines(np.mean(error), 0, 15 , color=cm(j+4))
        # ax[j].set_yscale("log")
    ax[0].set_title("36ms non-overlapping windows")
    ax[1].set_title("108ms overlapping windows")
    ax[2].set_title("252ms overlapping windows")
    ax[3].set_title("504ms overlapping windows")
    fig.tight_layout()
    [a.set_ylabel("histogram") for a in ax]
    ax[3].set_xlabel("absolute linear error")
    # [a.set_aspect(0.02) for a in ax]
    fig.show()

    fig,ax = plt.subplots(4,1,sharex=True,sharey=True)
    window_size = []
    for i in range(4):
        error = np.abs(trueLinearPos-linearpos_bayes_varying_window[i])[habEpochMask]

        e1 = np.sum(np.less(error,0.05))/error.shape[0]
        e2 = (error.shape[0]-np.sum(np.less(error,0.05)))/error.shape[0]
        e1_high = np.sum(np.less(error,0.15))/error.shape[0]
        e2_high = (error.shape[0]-np.sum(np.less(error,0.15)))/error.shape[0]
        e1_low = np.sum(np.less(error,0.05))/error.shape[0]
        e2_low = (error.shape[0] - np.sum(np.less(error,0.05)))/error.shape[0]

        # ax[i].hist(error,color=cm(i),bins=binsLinearPosHist,alpha=0.3,density=True)
        # ax[i].hist(error, color=cm(i), bins=binsLinearPosHist,histtype="step",density=True)
        ax[i].vlines(0,0,e1, color=cm(i),linewidth=10)
        # ax[i].vlines(0,e1_low,e1_high,color="black",linewidth=3)
        ax[i].vlines(0.3,0,e2, color=cm(i),linewidth=10)
        # ax[i].vlines(0.3, e2_low, e2_high, color="black", linewidth=3)
        ax[i].set_xlim(-1,2)
        # ax[i].set_yscale("log")
    for j in range(4):
        error = np.abs(trueLinearPos - linearpos_NN_varying_window_argmax[j])[habEpochMask]
        e1 = np.sum(np.less(error,0.05))/error.shape[0]
        e2 = (error.shape[0]-np.sum(np.less(error,0.05)))/error.shape[0]
        e1_high = np.sum(np.less(error,0.15))/error.shape[0]
        e2_high = (error.shape[0]- np.sum(np.less(error,0.15)))/error.shape[0]
        e1_low = np.sum(np.less(error,0.05))/error.shape[0]
        e2_low = (error.shape[0] - np.sum(np.less(error,0.05)))/error.shape[0]
        # ax[j].hist(error, color=cm(j+4),bins=binsLinearPosHist,alpha=0.3,density=True)
        # ax[j].hist(error, color=cm(j+4), bins=binsLinearPosHist, histtype="step",density=True)
        ax[j].vlines(0.1, 0, e1, color=cm(j+4),linewidth =10)
        # ax[j].vlines(0.1, e1_low, e1_high, color="black", linewidth=3)
        ax[j].vlines(0.4, 0, e2, color=cm(j+4), linewidth=10)
        # ax[j].vlines(0.4, e2_low, e2_high, color="black", linewidth=3)
        # ax[j].set_xlim(-1,2)
        # ax[j].set_yscale("log")
    ax[0].set_title("36ms non-overlapping windows")
    ax[1].set_title("108ms overlapping windows")
    ax[2].set_title("252ms overlapping windows")
    ax[3].set_title("504ms overlapping windows")
    fig.tight_layout()
    [a.set_ylabel("histogram") for a in ax]
    ax[3].set_xlabel("absolute linear error")
    # [a.set_aspect(0.02) for a in ax]
    fig.show()


    fig,ax = plt.subplots()
    meanNN_error = np.array([np.mean(np.abs(trueLinearPos - linearpos_NN_varying_window_argmax[j])[habEpochMask] )for j in range(4)])
    stdNN_error = np.array([np.std(np.abs(trueLinearPos - linearpos_NN_varying_window_argmax[j])[habEpochMask] ) for j in range(4)])
    ax.plot([36,108,252,504],meanNN_error,c="red",label="neural network")
    # ax.fill_between([36,108,252,504],meanNN_error-stdNN_error,meanNN_error+stdNN_error,color="red",alpha=0.5)
    mean_bayes_error = np.array([np.mean(np.abs(trueLinearPos - linearpos_bayes_varying_window[j])[habEpochMask] ) for j in range(4)])
    std_bayes_error = np.array([np.std(np.abs(trueLinearPos - linearpos_bayes_varying_window[j])[habEpochMask] ) for j in range(4)])
    ax.plot([36,108,252,504],mean_bayes_error,c="blue",label="bayesian")
    ax.set_xlabel("window size (ms)")
    ax.set_xticks([36,108,252,504])
    ax.set_xticklabels([36,108,252,504])
    ax.set_ylabel("mean linear error")
    fig.legend(loc=(0.6,0.7))
    # ax.fill_between([36, 108, 252, 504], mean_bayes_error - std_bayes_error, mean_bayes_error + std_bayes_error, color="blue", alpha=0.5)
    fig.show()

    fig,ax = plt.subplots()
    meanNN_error = np.array([np.mean(np.abs(trueLinearPos - linearpos_NN_varying_window_argmax[j])[habEpochMaskandSpeed] )for j in range(4)])
    stdNN_error = np.array([np.std(np.abs(trueLinearPos - linearpos_NN_varying_window_argmax[j])[habEpochMaskandSpeed] ) for j in range(4)])
    ax.plot([36,108,252,504],meanNN_error,c="red",label="neural network")
    # ax.fill_between([36,108,252,504],meanNN_error-stdNN_error,meanNN_error+stdNN_error,color="red",alpha=0.5)
    mean_bayes_error = np.array([np.mean(np.abs(trueLinearPos - linearpos_bayes_varying_window[j])[habEpochMaskandSpeed] ) for j in range(4)])
    std_bayes_error = np.array([np.std(np.abs(trueLinearPos - linearpos_bayes_varying_window[j])[habEpochMaskandSpeed] ) for j in range(4)])
    ax.plot([36,108,252,504],mean_bayes_error,c="blue",label="bayesian")
    ax.set_xlabel("window size (ms)")
    ax.set_xticks([36,108,252,504])
    ax.set_xticklabels([36,108,252,504])
    ax.set_ylabel("mean linear error")
    fig.legend(loc=(0.6,0.7))
    ax.set_title("high speed")
    # ax.fill_between([36, 108, 252, 504], mean_bayes_error - std_bayes_error, mean_bayes_error + std_bayes_error, color="blue", alpha=0.5)
    fig.show()

    ## Figure: comparing prediction of NN and Bayesian

    from SimpleBayes import butils
    # density_bayes_nn = butils.hist2D(np.array(
    #     [linearpos_bayes_varying_window[0][habEpochMask], linearpos_NN_varying_window_argmax[0][habEpochMask]]),nbins=[100,100])
    fig, ax = plt.subplots()
    ax.scatter(linearpos_bayes_varying_window[3][habEpochMaskandSpeed],linearpos_NN_varying_window_argmax[3][habEpochMaskandSpeed],s=1,alpha=0.03,c="black")
    fig.show()
    Ms = []
    Mslow = []
    for l in range(4):
        M = np.zeros([20, 20])
        for idi, i in enumerate(np.arange(0, stop=1, step=0.05)):
            for idj, j in enumerate(np.arange(0, stop=1, step=0.05)):
                M[idi, idj] = np.sum(
                    np.greater_equal(linearpos_bayes_varying_window[l][habEpochMaskandSpeed], i) * np.less(
                        linearpos_bayes_varying_window[l][habEpochMaskandSpeed], i + 0.05) *
                    np.greater_equal(linearpos_NN_varying_window_argmax[l][habEpochMaskandSpeed], j) * np.less(
                        linearpos_NN_varying_window_argmax[l][habEpochMaskandSpeed], j + 0.05))
        Ms += [M]
        M = np.zeros([20, 20])
        for idi, i in enumerate(np.arange(0, stop=1, step=0.05)):
            for idj, j in enumerate(np.arange(0, stop=1, step=0.05)):
                M[idi, idj] = np.sum(
                    np.greater_equal(linearpos_bayes_varying_window[l][habEpochMask*np.logical_not(windowmask_speed)], i) * np.less(
                        linearpos_bayes_varying_window[l][habEpochMask*np.logical_not(windowmask_speed)], i + 0.05) *
                    np.greater_equal(linearpos_NN_varying_window_argmax[l][habEpochMask*np.logical_not(windowmask_speed)], j) * np.less(
                        linearpos_NN_varying_window_argmax[l][habEpochMask*np.logical_not(windowmask_speed)], j + 0.05))
        Mslow += [M]

    cm = plt.get_cmap("terrain")
    fig,ax = plt.subplots(4,4,figsize=(10,10))
    for i in range(4):
        ax[i,0].scatter(linearpos_bayes_varying_window[i][habEpochMaskandSpeed],
                        linearpos_NN_varying_window_argmax[i][habEpochMaskandSpeed],s=1,c="grey")
        ax[i,0].hist2d(linearpos_bayes_varying_window[i][habEpochMaskandSpeed],
                        linearpos_NN_varying_window_argmax[i][habEpochMaskandSpeed],(45,45),cmap=white_viridis,
                       alpha=0.8)
        ax[i,1].scatter(linearpos_bayes_varying_window[i][habEpochMaskandSpeed*np.greater_equal(normalized_proba,1)],
                        linearpos_NN_varying_window_argmax[i][habEpochMaskandSpeed*np.greater_equal(normalized_proba,1)],s=1,c="grey")
        ax[i,1].hist2d(linearpos_bayes_varying_window[i][habEpochMaskandSpeed*np.greater_equal(normalized_proba,1)],
                        linearpos_NN_varying_window_argmax[i][habEpochMaskandSpeed*np.greater_equal(normalized_proba,1)],(45,45),cmap=white_viridis,
                       alpha=0.8)
    for i in range(4):
        ax[i,2].scatter(linearpos_bayes_varying_window[i][habEpochMask*np.logical_not(windowmask_speed)],
                        linearpos_NN_varying_window_argmax[i][habEpochMask*np.logical_not(windowmask_speed)],s=1,c="grey")
        ax[i,2].hist2d(linearpos_bayes_varying_window[i][habEpochMask*np.logical_not(windowmask_speed)],
                        linearpos_NN_varying_window_argmax[i][habEpochMask*np.logical_not(windowmask_speed)],(45,45),cmap=white_viridis,
                       alpha=0.8)
        ax[i,3].scatter(linearpos_bayes_varying_window[i][habEpochMask*np.logical_not(windowmask_speed)*np.greater_equal(normalized_proba,1)],
                        linearpos_NN_varying_window_argmax[i][habEpochMask*np.logical_not(windowmask_speed)*np.greater_equal(normalized_proba,1)],s=1,c="grey")
        ax[i,3].hist2d(linearpos_bayes_varying_window[i][habEpochMask*np.logical_not(windowmask_speed)*np.greater_equal(normalized_proba,1)],
                        linearpos_NN_varying_window_argmax[i][habEpochMask*np.logical_not(windowmask_speed)*np.greater_equal(normalized_proba,1)],(45,45),cmap=white_viridis,
                       alpha=0.8)
    ax[0,0].set_title("high speed")
    ax[0,1].set_title("probability filtering \n high speed")
    ax[0,2].set_title("slow speed")
    ax[0, 3].set_title("probability filtering \n slow speed")
    [a.set_xlabel("Bayesian decoding") for a in ax[3,:]]
    [a.set_ylabel("NN decoding") for a in ax[:,0]]
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "NNvsBayesian.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "NNvsBayesian.svg"))

    fig,ax = plt.subplots(4,1)
    for i in range(4):
        ax[i].scatter(linearpos_bayes_varying_window[i][habEpochMask],linearpos_NN_varying_window_argmax[i][habEpochMask],s=1,alpha=0.03,c="black")
        density_bayes_nn = butils.hist2D(np.array(
            [linearpos_bayes_varying_window[i][habEpochMask], linearpos_NN_varying_window_argmax[i][habEpochMask]]))
        ax[i].imshow(density_bayes_nn)
        ax[i].set_xlabel("bayesian prediction")
        ax[i].set_ylabel("NN predictions")
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "entropy_bias.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "entropy_bias.svg"))

    fig,ax = plt.subplots(3,1,sharex=True,figsize=(5,7))
    ax[0].scatter(medianLinearPos[habEpochMask*np.logical_not(windowmask_speed)],ent[habEpochMask*np.logical_not(windowmask_speed)],s=1,
                  c="black",alpha=0.1)
    ax[0].set_xlabel("predicted linear position")
    ax[0].set_ylabel("entropy" )
    ax[1].hist(trueLinearPos[habEpochMask],bins=50)
    ax[1].set_title("Animal position, histogram")
    # ax[1].set_xlabel("bayesian prediction")
    # ax[1].set_ylabel("NN predictions")
    ax[2].hist(medianLinearPos[habEpochMask*np.logical_not(windowmask_speed)],bins=50)
    ax[2].set_title("Animal position, slow speed, histogram")
    # ax[2].set_ylabel("NN predictions")
    ax[2].set_xlabel("position")
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "entropy_bias.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "entropy_bias.svg"))

    from SimpleBayes.butils import hist2D

    fig,ax = plt.subplots()
    ax.hist(lossPredNoNoise_train[windowmask_speed_train],bins=100)
    fig.show()
    errors = np.log(np.sqrt(np.sum(np.square(featureNoNoisePosTrue-featureNoNoisePosPred),axis=1)))
    fig,ax = plt.subplots(2,2)
    ax[0,0].scatter(lossPredNoNoise[windowmask_speed], errors[windowmask_speed],
                             c="grey",s=1)
    ax[0,0].hist2d(lossPredNoNoise[windowmask_speed][:,0],errors[windowmask_speed],(30,30),
                   cmap=white_viridis, alpha=0.4) #,c="red",alpha=0.4
    ax[0,1].scatter(lossPredNoNoise[np.logical_not(windowmask_speed)], errors[np.logical_not(windowmask_speed)],
                    s=1,c="grey")
    ax[0,1].hist2d(lossPredNoNoise[np.logical_not(windowmask_speed)][:,0], errors[np.logical_not(windowmask_speed)]
                   ,(30,30), cmap=white_viridis,alpha=0.4)
    ax[1,0].scatter(np.exp(lossPredNoNoise[windowmask_speed]),np.exp(errors[windowmask_speed]),s=1,c="grey")
    ax[1,0].hist2d(np.exp(lossPredNoNoise[windowmask_speed])[:,0],np.exp(errors[windowmask_speed])
                   ,(30,30), cmap=white_viridis,alpha=0.4)
    ax[1,1].scatter(np.exp(lossPredNoNoise[np.logical_not(windowmask_speed)]),
                    np.exp(errors[np.logical_not(windowmask_speed)]), s=1,c="grey")
    ax[1,1].hist2d(np.exp(lossPredNoNoise[np.logical_not(windowmask_speed)])[:,0],
                    np.exp(errors[np.logical_not(windowmask_speed)]),(30,30), cmap=white_viridis,alpha=0.4)
    ax[0,0].set_xlabel("error predicted (log)")
    ax[0,0].set_ylabel("true error (log)")
    ax[0,0].set_title("high speed \n testing set")
    ax[0,1].set_xlabel("error predicted (log)")
    ax[0,1].set_ylabel("true error (log)")
    ax[0,1].set_title("slow speed \n testing set")
    ax[1,0].set_xlabel("error predicted ")
    ax[1,0].set_ylabel("true error")
    ax[1,1].set_xlabel("error predicted")
    ax[1,1].set_ylabel("true error")
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "link_true_vs_inferredError.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "link_true_vs_inferredError.svg"))
    #todo: Linear regression

    mazeBorder = np.array([[0,0,1,1,0.63,0.63,0.35,0.35,0],[0,1,1,0,0,0.75,0.75,0,0]])

    ### Figure sur les low vs high speed
    time_filter_traj = inEpochsMask(timePredsNoNoise,[1625,1645])[:,0]  #*np.less(lossPredNoNoise[:,0],-4)
    cm = plt.get_cmap("turbo")
    fig,ax = plt.subplots()
    ts = timePredsNoNoise[time_filter_traj]
    ax.scatter(featureNoNoisePosPred[time_filter_traj,0],featureNoNoisePosPred[time_filter_traj,1],c=cm((ts-np.min(ts))/(np.max(ts)-np.min(ts))),s=3,label="NN")
    ax.plot(featureNoNoisePosTrue[time_filter_traj,0],featureNoNoisePosTrue[time_filter_traj,1],label="true trajectory")
    plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=np.min(ts),vmax=np.max(ts)),cmap=cm),label="prediction time (s)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.plot(mazeBorder.transpose()[:,0],mazeBorder.transpose()[:,1],c="black")
    fig.legend()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "decoded_trajectories.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "decoded_trajectories.svg"))


    fig,ax = plt.subplots()
    time_filter  = np.less(lossPredNoNoise[:,0],-4)
    ax.plot(timePredsNoNoise[time_filter],featureNoNoisePosTrue[:,0][time_filter],c="black",alpha=0.6)
    ax.scatter(timePredsNoNoise[time_filter],featureNoNoisePosPred[:,0][time_filter],c="red",s=1)
    fig.show()

    ### Figure 2: scatter the link between proba and error for the NN


    fig,ax = plt.subplots()
    ax.scatter(linearNoNoisePos,lossPredNoNoise,s=1,color="grey")
    ax.hist2d(linearNoNoisePos[:,0],lossPredNoNoise[:,0], (50, 50), cmap=white_viridis, alpha=0.4)
    fig.show()

    fig,ax = plt.subplots(1,3)
    ax[0].scatter(max_proba_NN[windowmask_speed],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[windowmask_speed],s=1,c="grey")
    ax[0].hist2d(max_proba_NN[windowmask_speed], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[windowmask_speed], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[0].set_title("high speed \n testing set")
    ax[0].set_xlabel("Neural network probability")
    ax[0].set_ylabel("Absolute linear error")
    ax[1].scatter(max_proba_NN[np.logical_not(windowmask_speed)],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[np.logical_not(windowmask_speed)],s=1,c="grey")
    ax[1].hist2d(max_proba_NN[np.logical_not(windowmask_speed)], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[np.logical_not(windowmask_speed)], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[1].set_xlabel("Neural network probability")
    ax[1].set_ylabel("Absolute linear error")
    ax[1].set_title("slow speed \n testing set")
    ax[2].scatter(max_proba_NN,linearpos_NN_varying_window_argmax[0],s=1,c="grey")
    ax[2].hist2d(max_proba_NN, linearpos_NN_varying_window_argmax[0], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[2].set_title("testing set \n")
    ax[2].set_xlabel("Neural network probability")
    ax[2].set_ylabel("predicted linear position")
    [a.set_xlim(0,1) for a in ax]
    [a.set_ylim(0, 1) for a in ax]
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_probaVSLinearError.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_probaVSLinearError.svg"))


    fig,ax = plt.subplots(1,3)
    ax[0].scatter(normalized_proba[windowmask_speed],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[windowmask_speed],s=1,c="grey")
    ax[0].hist2d(normalized_proba[windowmask_speed], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[windowmask_speed], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[0].set_title("high speed \n testing set")
    ax[0].set_xlabel("Neural network normalized probability")
    ax[0].set_ylabel("Absolute linear error")
    ax[1].scatter(normalized_proba[np.logical_not(windowmask_speed)],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[np.logical_not(windowmask_speed)],s=1,c="grey")
    ax[1].hist2d(normalized_proba[np.logical_not(windowmask_speed)], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[np.logical_not(windowmask_speed)], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[1].set_xlabel("Neural network normalized probability")
    ax[1].set_ylabel("Absolute linear error")
    ax[1].set_title("slow speed \n testing set")
    ax[2].scatter(normalized_proba,linearpos_NN_varying_window_argmax[0],s=1,c="grey")
    ax[2].hist2d(normalized_proba, linearpos_NN_varying_window_argmax[0], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[2].set_title("testing set \n")
    ax[2].set_xlabel("Neural network normalized probability")
    ax[2].set_ylabel("predicted linear position")
    # [a.set_xlim(0,1) for a in ax]
    # [a.set_ylim(0, 1) for a in ax]
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_probaVSLinearError_normalized.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_probaVSLinearError_normalized.svg"))



    fig,ax = plt.subplots(1,3)
    ax[0].scatter(ent[windowmask_speed],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[windowmask_speed],s=1,c="grey")
    ax[0].hist2d(ent[windowmask_speed], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[windowmask_speed], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[0].set_title("high speed \n testing set")
    ax[0].set_xlabel("Neural network entropy")
    ax[0].set_ylabel("Absolute linear error")
    ax[1].scatter(ent[np.logical_not(windowmask_speed)],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[np.logical_not(windowmask_speed)],s=1,c="grey")
    ax[1].hist2d(ent[np.logical_not(windowmask_speed)], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[np.logical_not(windowmask_speed)], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[1].set_xlabel("Neural network entropy")
    ax[1].set_ylabel("Absolute linear error")
    ax[1].set_title("slow speed \n testing set")
    ax[2].scatter(ent,linearpos_NN_varying_window_argmax[0],s=1,c="grey")
    ax[2].hist2d(ent, linearpos_NN_varying_window_argmax[0], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[2].set_title("testing set \n")
    ax[2].set_xlabel("Neural network entropy")
    ax[2].set_ylabel("predicted linear position")
    # [a.set_xlim(-4,) for a in ax]
    [a.set_ylim(0, 1) for a in ax]
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_entropyVSLinearError.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_entropyVSLinearError.svg"))

    nanFilter_speed_high  = np.logical_not(np.isnan(normalized_entropied))*windowmask_speed
    nanFilter_speed_slow = np.logical_not(np.isnan(normalized_entropied))*np.logical_not(windowmask_speed)
    nan_filter= np.logical_not(np.isnan(normalized_entropied))
    fig,ax = plt.subplots(1,3)
    ax[0].scatter(normalized_entropied[nanFilter_speed_high],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[nanFilter_speed_high],s=1,c="grey")
    ax[0].hist2d(normalized_entropied[nanFilter_speed_high], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[nanFilter_speed_high], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[0].set_title("high speed \n testing set")
    ax[0].set_xlabel("Neural network entropy, \n normalized given predicted position")
    ax[0].set_ylabel("Absolute linear error")
    ax[0].set_yscale("log")
    ax[1].scatter(normalized_entropied[nanFilter_speed_slow],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[nanFilter_speed_slow],s=1,c="grey")
    ax[1].hist2d(normalized_entropied[nanFilter_speed_slow], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[nanFilter_speed_slow], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[1].set_xlabel("Neural network entropy, \n normalized given predicted position")
    ax[1].set_ylabel("Absolute linear error")
    ax[1].set_title("slow speed \n testing set")
    ax[2].scatter(normalized_entropied[nan_filter],linearpos_NN_varying_window_argmax[0][nan_filter],s=1,c="grey")
    ax[2].hist2d(normalized_entropied[nan_filter], linearpos_NN_varying_window_argmax[0][nan_filter], (50, 50), cmap=white_viridis, alpha=0.4)
    ax[2].set_title("testing set \n")
    ax[2].set_xlabel("Neural network entropy, \n normalized given predicted position")
    ax[2].set_ylabel("predicted linear position")
    # [a.set_xlim(-4,) for a in ax]
    [a.set_ylim(0, 1) for a in ax]
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_entropyVSLinearError_normalized.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_entropyVSLinearError_normalized.svg"))


    ## Figure: decrease of the mean absolute linear error as a function of the filtering value

    error_highspeed_proba_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.greater_equal(max_proba_NN,pfilt)*windowmask_speed])
                       for pfilt in np.arange(np.min(max_proba_NN),np.max(max_proba_NN),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_highspeed_normalizedproba_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.greater_equal(normalized_proba,pfilt)*windowmask_speed])
                       for pfilt in np.arange(np.min(normalized_proba),np.max(normalized_proba),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_highspeed_NN_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.less_equal(lossPredNoNoise[:,0],pfilt)*windowmask_speed])
                       for pfilt in np.arange(np.min(lossPredNoNoise[:,0]),np.max(lossPredNoNoise[:,0]),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_highspeed_entropy_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.less_equal(ent,pfilt)*windowmask_speed])
                       for pfilt in np.arange(np.nanmin(ent),np.nanmax(ent),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_highspeed_normalized_entropy_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.less_equal(normalized_entropied,pfilt)*windowmask_speed])
                       for pfilt in np.arange(np.nanmin(normalized_entropied),np.nanmax(normalized_entropied),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]

    cm = plt.get_cmap("tab20c")
    labelNames = ["36ms","108ms","252ms","504ms"]
    fig,ax = plt.subplots(2,3)
    [ax[0,0].plot(np.arange(np.min(max_proba_NN),np.max(max_proba_NN),0.01),error_highspeed_proba_filter[i],c=cm(i+4),label=labelNames[i]) for i in range(4)]
    ax[0,0].set_xlabel("neural network \n probability filtering value")
    ax[0,0].set_ylabel("mean absolute linear error")
    ax[0,0].set_title("high speed \n testing set")
    [ax[1,0].plot(np.arange(np.min(normalized_proba),np.max(normalized_proba),0.01),error_highspeed_normalizedproba_filter[i],c=cm(i+4)) for i in range(4)]
    ax[1,0].set_xlabel("neural network \n probability filtering value")
    ax[1,0].set_ylabel("mean absolute linear error")
    ax[1,0].set_title("high speed \n testing set")
    [ax[1,1].plot(np.arange(np.min(lossPredNoNoise[:,0]),np.max(lossPredNoNoise[:,0]),0.01),error_highspeed_NN_filter[i],c=cm(i+4)) for i in range(4)]
    ax[1,1].set_xlabel("neural network \n prediction filtering value")
    ax[1,1].set_ylabel("mean absolute linear error")
    ax[1,1].set_title("high speed \n testing set")
    [ax[0,2].plot(np.arange(np.nanmin(ent),np.nanmax(ent),0.01),error_highspeed_entropy_filter[i],c=cm(i+4)) for i in range(4)]
    ax[0,2].set_xlabel("neural network \n entropy filtering value")
    ax[0,2].set_ylabel("mean absolute linear error")
    ax[0,2].set_title("high speed \n testing set")
    [ax[1,2].plot(np.arange(np.nanmin(normalized_entropied),np.nanmax(normalized_entropied),0.01),error_highspeed_normalized_entropy_filter[i],c=cm(i+4)) for i in range(4)]
    ax[1,2].set_xlabel("neural network \n entropy filtering value")
    ax[1,2].set_ylabel("mean absolute linear error")
    ax[1,2].set_title("high speed \n testing set")
    ax[0,1].set_visible(False)
    ax[0,1].set_title("Warning bad figure \n because entropy, proba or NN pred filtering \n  are always based on the first 36 ms window")
    fig.legend(loc=(0.42,0.7))
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_filtering_3types_high.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_filtering_3types_high.svg"))

    error_lowspeed_proba_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.greater_equal(max_proba_NN,pfilt)*np.logical_not(windowmask_speed)])
                       for pfilt in np.arange(np.min(max_proba_NN),np.max(max_proba_NN),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_lowspeed_normalizedproba_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.greater_equal(normalized_proba,pfilt)*np.logical_not(windowmask_speed)])
                       for pfilt in np.arange(np.min(normalized_proba),np.max(normalized_proba),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_lowspeed_NN_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.less_equal(lossPredNoNoise[:,0],pfilt)*np.logical_not(windowmask_speed)])
                       for pfilt in np.arange(np.min(lossPredNoNoise[:,0]),np.max(lossPredNoNoise[:,0]),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_lowspeed_entropy_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.less_equal(ent,pfilt)*np.logical_not(windowmask_speed)])
                       for pfilt in np.arange(np.nanmin(ent),np.nanmax(ent),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]
    error_lowspeed_normalized_entropy_filter = [[np.mean(np.abs(linearpos_NN_varying_window_argmax[i]-trueLinearPos)[np.less_equal(normalized_entropied,pfilt)*np.logical_not(windowmask_speed)])
                       for pfilt in np.arange(np.nanmin(normalized_entropied),np.nanmax(normalized_entropied),0.01)] for i in range(len(linearpos_NN_varying_window_argmax))]

    cm = plt.get_cmap("tab20c")
    fig,ax = plt.subplots(2,3)
    [ax[0,0].plot(np.arange(np.min(max_proba_NN),np.max(max_proba_NN),0.01),error_lowspeed_proba_filter[i],c=cm(i+4),label=labelNames[i]) for i in range(4)]
    ax[0,0].set_xlabel("neural network \n probability filtering value")
    ax[0,0].set_ylabel("mean absolute linear error")
    ax[0,0].set_title("slow speed \n testing set")
    [ax[1,0].plot(np.arange(np.min(normalized_proba),np.max(normalized_proba),0.01),error_lowspeed_normalizedproba_filter[i],c=cm(i+4)) for i in range(4)]
    ax[1,0].set_xlabel("neural network \n probability filtering value")
    ax[1,0].set_ylabel("mean absolute linear error")
    ax[1,0].set_title("slow speed \n testing set")
    [ax[1,1].plot(np.arange(np.min(lossPredNoNoise[:,0]),np.max(lossPredNoNoise[:,0]),0.01),error_lowspeed_NN_filter[i],c=cm(i+4)) for i in range(4)]
    ax[1,1].set_xlabel("neural network \n prediction filtering value")
    ax[1,1].set_ylabel("mean absolute linear error")
    ax[1,1].set_title("slow speed \n testing set")
    [ax[0,2].plot(np.arange(np.nanmin(ent),np.nanmax(ent),0.01),error_lowspeed_entropy_filter[i],c=cm(i+4)) for i in range(4)]
    ax[0,2].set_xlabel("neural network \n entropy filtering value")
    ax[0,2].set_ylabel("mean absolute linear error")
    ax[0,2].set_title("slow speed \n testing set")
    [ax[1,2].plot(np.arange(np.nanmin(normalized_entropied),np.nanmax(normalized_entropied),0.01),error_lowspeed_normalized_entropy_filter[i],c=cm(i+4)) for i in range(4)]
    ax[1,2].set_xlabel("neural network \n entropy filtering value")
    ax[1,2].set_ylabel("mean absolute linear error")
    ax[1,2].set_title("slow speed \n testing set")
    fig.legend(loc=(0.42,0.7))
    ax[0,1].set_visible(False)
    ax[0,1].set_title("Warning bad figure \n because entropy, proba or NN pred filtering \n  are always based on the first 36 ms window")
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_filtering_3types_slow.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_filtering_3types_slow.svg"))

    #Finally we scatter plot the link between these different statistics:

    fig,ax = plt.subplots(2,3)
    ax[0,0].scatter(max_proba_NN,lossPredNoNoise,s=1,color="grey")
    ax[0,0].hist2d(max_proba_NN,lossPredNoNoise[:,0],(50,50), cmap=white_viridis,alpha=0.4)
    ax[0,0].set_xlabel("max probability")
    ax[0,0].set_ylabel("predicted error")
    ax[0,1].scatter(ent,lossPredNoNoise,s=1,color="grey")
    ax[0,1].hist2d(ent,lossPredNoNoise[:,0],(50,50), cmap=white_viridis,alpha=0.4)
    ax[0,1].set_xlabel("entropy")
    ax[0,1].set_ylabel("predicted error")
    ax[0,2].scatter(max_proba_NN,ent,s=1,color="grey")
    ax[0,2].hist2d(max_proba_NN,ent,(50,50), cmap=white_viridis,alpha=0.4)
    ax[0,2].set_xlabel("max probability")
    ax[0,2].set_ylabel("entropy")
    ax[1,0].scatter(normalized_proba,lossPredNoNoise,s=1,color="grey")
    ax[1,0].hist2d(normalized_proba,lossPredNoNoise[:,0],(50,50), cmap=white_viridis,alpha=0.4)
    ax[1,0].set_xlabel("normalized max probability")
    ax[1,0].set_ylabel("predicted error")
    ax[1,1].scatter(normalized_entropied[nan_filter],lossPredNoNoise[nan_filter],s=1,color="grey")
    ax[1,1].hist2d(normalized_entropied[nan_filter],lossPredNoNoise[nan_filter,0],(50,50), cmap=white_viridis,alpha=0.4)
    ax[1,1].set_xlabel("normalized entropy")
    ax[1,1].set_ylabel("predicted error")
    ax[1,2].scatter(normalized_proba[nan_filter],normalized_entropied[nan_filter],s=1,color="grey")
    ax[1,2].hist2d(normalized_proba[nan_filter],normalized_entropied[nan_filter],(50,50), cmap=white_viridis,alpha=0.4)
    ax[1,2].set_xlabel("normalized max probability")
    ax[1,2].set_ylabel("normalized entropy")
    fig.suptitle("test set, all speed")
    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_comparing3filter.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_comparing3filter.svg"))


    #We then replot figure 1 using filtering:

    ## Figure 1: on habituation set, speed filtered, we plot an example of bayesian and neural network decoding

    cm = plt.get_cmap("tab20c")
    time_filter = np.greater(timePreds,950)*np.less(timePreds,1100)
    time_filter_noproba = np.ones(timePreds.shape[0]).astype(np.bool)
    time_filter = np.ones(timePreds.shape[0]).astype(np.bool)*np.greater_equal(normalized_proba,1)
    fig,ax = plt.subplots(4,2,sharex=True,figsize=(15,10),sharey=True)

    [a.plot(timePreds[time_filter],trueLinearPos[time_filter],c="black",alpha=0.3) for a in ax[:,0]]
    ax[0,0].scatter(timePreds[time_filter_noproba],linearpos_NN_varying_window_argmax[0][time_filter_noproba],c=cm(4),alpha=0.9,label="36ms non-overlapping window",s=1)
    ax[0,0].set_title("Neural network decoder \n 36ms non-overlapping window")
    ax[1,0].scatter(timePreds[time_filter_noproba], linearpos_NN_varying_window_argmax[1][time_filter_noproba], c=cm(5), alpha=0.9,label="108ms overlapping window",s=1)
    ax[1,0].set_title("108ms overlapping window")
    ax[2,0].scatter(timePreds[time_filter_noproba], linearpos_NN_varying_window_argmax[2][time_filter_noproba], c=cm(6), alpha=0.9,label="216ms overlapping window",s=1)
    ax[2,0].set_title("252ms overlapping window")
    ax[3,0].scatter(timePreds[time_filter_noproba], linearpos_NN_varying_window_argmax[3][time_filter_noproba], c=cm(7), alpha=0.9,label="504ms overlapping window",s=1)
    ax[3,0].set_title("504ms overlapping window")

    [a.plot(timePreds[time_filter],trueLinearPos[time_filter],c="black",alpha=0.3) for a in ax[:,1]]
    ax[0,1].scatter(timePreds[time_filter],linearpos_NN_varying_window_argmax[0][time_filter],c=cm(4),alpha=0.9,label="36ms non-overlapping window",s=1)
    ax[0,1].set_title("Neural network  \n 36ms non-overlapping window \n probability filtered")
    ax[1,1].scatter(timePreds[time_filter], linearpos_NN_varying_window_argmax[1][time_filter], c=cm(5), alpha=0.9,label="108ms overlapping window",s=1)
    ax[1,1].set_title("108ms overlapping window")
    ax[2,1].scatter(timePreds[time_filter], linearpos_NN_varying_window_argmax[2][time_filter], c=cm(6), alpha=0.9,label="216ms overlapping window",s=1)
    ax[2,1].set_title("252ms overlapping window")
    ax[3,1].scatter(timePreds[time_filter], linearpos_NN_varying_window_argmax[3][time_filter], c=cm(7), alpha=0.9,label="504ms overlapping window",s=1)
    ax[3,1].set_title("504ms overlapping window")
    [a.set_ylabel("linear position") for a in ax[:,0]]
    # [a.set_ylabel("linear position") for a in ax[:,1]]
    ax[3,1].set_xlabel("time (s)")
    ax[3,0].set_xlabel("time (s)")
    # [a.set_aspect(50/2) for a in ax]
    fig.tight_layout()
    fig.show()


    ### Let us pursue on comparing NN and Bayesian:

    import tqdm
    ## We will compare the NN with bayesian, random and shuffled bayesian
    errors = []
    errorsRandomMean = []
    errorsRandomStd = []
    errorsShuffleMean = []
    errorsShuffleStd = []
    for nproba in tqdm.tqdm(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1)):
        bayesPred = linearpos_bayes_varying_window[3][
            np.greater_equal(normalized_proba, nproba)]
        NNpred = linearpos_NN_varying_window_argmax[3][np.greater_equal(normalized_proba, nproba)]
        if (NNpred.shape[0] > 0):
            randomPred = np.random.uniform(0, 1, [NNpred.shape[0], 100])
            errors += [np.mean(np.abs(bayesPred - NNpred))]
            errRand = np.mean(np.abs(NNpred[:, None] - randomPred), axis=0)
            errorsRandomMean += [np.mean(errRand)]
            errorsRandomStd += [np.std(errRand)]
        shuffles = []
        for id in range(100):
            b = np.copy(bayesPred)
            np.random.shuffle(b)
            shuffles += [np.mean(np.abs(NNpred - b))]
        errorsShuffleMean += [np.mean(shuffles)]
        errorsShuffleStd += [np.std(shuffles)]
    errorsRandomMean = np.array(errorsRandomMean)
    errorsRandomStd = np.array(errorsRandomStd)
    errorsShuffleMean = np.array(errorsShuffleMean)
    errorsShuffleStd = np.array(errorsShuffleStd)
    fig, ax = plt.subplots()
    ax.plot(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1), errors, label="bayesian")
    ax.plot(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1), errorsRandomMean, color="red",
            label="random Prediction")
    ax.fill_between(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1),
                    errorsRandomMean + errorsRandomStd, errorsRandomMean - errorsRandomStd, color="orange")
    ax.plot(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1), errorsShuffleMean, color="purple",
            label="shuffle bayesian")
    ax.fill_between(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1),
                    errorsShuffleMean + errorsShuffleStd, errorsShuffleMean - errorsShuffleStd, color="violet")
    ax.set_ylabel("linear distance from NN predictions to Bayesian \n or random predictions")
    ax.set_xlabel("probability filtering value")
    ax.set_title("Wake")
    fig.legend(loc=[0.2, 0.2])
    fig.show()
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_lineardiffBayesNN_wake.png"))
    plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_lineardiffBayesNN_wake.svg"))

    ### supp Figure 2/3: link between NN predicted loss and the predicted position ?
    _, linearNoNoisePosPred_train = linearizationFunction(featureNoNoisePosPred_train)
    fig,ax = plt.subplots()
    ax.scatter(linearNoNoisePos,lossPredNoNoise,s=1,c="grey")
    ax.hist2d(linearNoNoisePos[:,0],lossPredNoNoise[:,0],(30,30),cmap=white_viridis,alpha=0.4)
    ax.set_xlabel("predicted linear position")
    ax.set_ylabel("NN predicted loss")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(linearNoNoisePos[habEpochMaskandSpeed],lossPredNoNoise[habEpochMaskandSpeed],s=1,c="grey")
    ax.hist2d(linearNoNoisePos[habEpochMaskandSpeed,0],lossPredNoNoise[habEpochMaskandSpeed,0],(30,30),cmap=white_viridis,alpha=0.4)
    ax.set_xlabel("predicted linear position")
    ax.set_ylabel("NN predicted loss")
    fig.show()
    fig,ax = plt.subplots()
    ax.scatter(linearNoNoisePos[habEpochMask*np.logical_not(windowmask_speed)],lossPredNoNoise[habEpochMask*np.logical_not(windowmask_speed)],s=1,c="grey")
    ax.hist2d(linearNoNoisePos[habEpochMask*np.logical_not(windowmask_speed),0],lossPredNoNoise[habEpochMask*np.logical_not(windowmask_speed),0],(30,30),cmap=white_viridis,alpha=0.4)
    ax.set_xlabel("predicted linear position")
    ax.set_ylabel("NN predicted loss")
    fig.show()


    mask_right_arm_pred_argmax = np.greater(linearpos_NN_varying_window_argmax[0],0.7)

    error_rightarm = np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[mask_right_arm_pred_argmax*habEpochMaskandSpeed]
    error_OtherArm = np.abs(linearpos_NN_varying_window_argmax[0] - trueLinearPos)[
        np.logical_not(mask_right_arm_pred_argmax) * habEpochMaskandSpeed]

    fig,ax = plt.subplots()
    ax.scatter(linearpos_NN_varying_window_argmax[0][mask_right_arm_pred_argmax*habEpochMaskandSpeed],error_rightarm,s=1)
    fig.show()
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].scatter(ent[habEpochMaskandSpeed],np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[habEpochMaskandSpeed],s=1,c="grey")
    ax[0].hist2d(ent[habEpochMaskandSpeed], np.abs(linearpos_NN_varying_window_argmax[0]-trueLinearPos)[habEpochMaskandSpeed], (30,30),alpha=0.5,cmap=white_viridis)

    ax[1].scatter(ent[mask_right_arm_pred_argmax*habEpochMaskandSpeed],error_rightarm,s=1,c="grey")
    ax[1].hist2d(ent[mask_right_arm_pred_argmax*habEpochMaskandSpeed], error_rightarm, (30,30),alpha=0.5,cmap=white_viridis)
    ax[2].scatter(ent[np.logical_not(mask_right_arm_pred_argmax)*habEpochMaskandSpeed],error_OtherArm,s=1,c="grey")
    ax[2].hist2d(ent[np.logical_not(mask_right_arm_pred_argmax)*habEpochMaskandSpeed], error_OtherArm, (30,30),alpha=0.5,cmap=white_viridis)
    ax[0].set_xlim(0,4)
    fig.show()


    mask_middle_arm_pred_argmax = np.greater(linearpos_NN_varying_window_argmax[0], 0.3)*np.less(linearpos_NN_varying_window_argmax[0], 0.7)
    error_MiddleArm = np.abs(linearpos_NN_varying_window_argmax[0] - trueLinearPos)[
        mask_middle_arm_pred_argmax * habEpochMaskandSpeed]
    fig,ax = plt.subplots()
    ax.hist(error_rightarm,color="blue",histtype="step",density=True,bins=50)
    # ax.vlines(np.mean(error_rightarm),ymin=0,ymax=16,color="blue")
    ax.vlines(np.median(error_rightarm), ymin=0, ymax=16, color="blue")
    ax.hist(error_MiddleArm,color="red",histtype="step",density=True,bins=50)
    # ax.vlines(np.mean(error_MiddleArm),ymin=0,ymax=16,color="red")
    ax.vlines(np.median(error_MiddleArm), ymin=0, ymax=16, color="red")
    fig.show()




    ### Figure 4: we take an example place cell,
    # and we scatter plot a link between its firing rate and the decoding.


