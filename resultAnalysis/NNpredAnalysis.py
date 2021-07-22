# In this file we extensively compare and study the difference
# between the neural network predictions
# and the result of spike sorting, as well as the bayesian decoding.

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


def compare_nn_popVector(trainerBayes,projectPath,params,linearizationFunction,usetrain=True):
    # Predictions of the neural network with uncertainty estimate are provided
    if usetrain:
        euclidData = np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_fit",
                                                       "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
        timePreds = np.array(pd.read_csv(
            os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "timePreds.csv")).values[:, 1],
                             dtype=np.float32)
        truePosFed = pd.read_csv(
            os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "truePosFed.csv")).values[:, 1:]
        windowmask_speed = np.array(pd.read_csv(
            os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "windowmask_speed.csv")).values[:, 1:],
                              dtype=np.float32)
    else:
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

    output_test = [
        np.reshape(euclidData, [-1, params.nb_eval_dropout, params.batch_size, params.dim_output])]

    projectedPos, linearPos = linearizationFunction(euclidData.astype(np.float64))

    linearPos = np.reshape(linearPos, output_test[0].shape[0:3])
    medianLinearPos = np.median(linearPos, axis=1)
    medianLinearPos = np.reshape(medianLinearPos, [np.prod(medianLinearPos.shape[0:2])])

    behavior_data = getBehavior(projectPath.folder, getfilterSpeed=True)
    trueProjPos, trueLinearPos = linearizationFunction(truePosFed)
    habEpochMask = inEpochsMask(timePreds,behavior_data["Times"]["testEpochs"][0:4])
    habEpochMaskandSpeed = (habEpochMask) * (windowmask_speed[:,0].astype(np.bool))
    nothabEpochMask = np.logical_not(habEpochMask)
    nothabEpochMaskandSpeed = np.logical_not(habEpochMask)* (windowmask_speed[:, 0].astype(np.bool))

    # fig,ax = plt.subplots()
    # ax.hist(trueLinearPos[habEpochMaskandSpeed],bins=100)
    # fig.show()
    # randomPos = np.random.uniform(size=[1000,trueLinearPos.shape[0]])
    # delta_random = np.mean(np.abs(randomPos-trueLinearPos)[:,habEpochMaskandSpeed],axis=1)
    # values, indices = np.histogram(trueLinearPos[ habEpochMaskandSpeed], bins=100)
    # values = values.astype(np.float32)
    # weights = values / np.sum(values)
    # new_random_pos = np.random.choice(indices[1:], [100,medianLinearPos.shape[0]], p=weights)
    # delta_random2 = np.mean(np.abs(new_random_pos - trueLinearPos)[:, habEpochMaskandSpeed], axis=1)
    # fig,ax = plt.subplots()
    # ax.hist(delta_random,density=True,bins=50,color="blue")
    # ax.hist(delta_random2, density=True, bins=50,color="green")
    # ax.vlines(np.mean(np.abs(medianLinearPos-trueLinearPos)[habEpochMaskandSpeed]),0,300,color="red")
    # fig.show()



    linearTranspose = np.transpose(linearPos, axes=[0, 2, 1])
    linearTranspose = linearTranspose.reshape(
        [linearTranspose.shape[0] * linearTranspose.shape[1], linearTranspose.shape[2]])
    histPosPred = np.stack([np.histogram(np.abs(linearTranspose[id, :] - np.median(linearTranspose[id, :])),
                                         bins=np.arange(0, stop=1, step=0.01))[0] for id in
                            range(linearTranspose.shape[0])])

    # CSV files helping to align the pop vector from spike used in spike sorting
    # with predictions from spike used by the NN are also provided.
    if usetrain:
        spikeMat_window_popVector = np.array(
            pd.read_csv(os.path.join(projectPath.resultsPath,"dataset", "alignment", "waketrain", "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32)
        spikeMat_times_window = np.array(
            pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "spikeMat_times_window.csv")).values[:,1:],dtype=np.float32)
        meanTimeWindow = np.array(
            pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "meanTimeWindow.csv")).values[:,1:],dtype=np.float32)
        startTimeWindow = np.array(
            pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "startTimeWindow.csv")).values[:,1:],dtype=np.float32)
        lenInputNN = np.array(
            pd.read_csv(os.path.join(projectPath.resultsPath, "dataset", "alignment", "waketrain", "lenInputNN.csv")).values[:,1:],dtype=np.float32)
    else:
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

    import tables
    import subprocess
    if not os.path.exists(os.path.join(projectPath.folder, "nnSWR.mat")):
        subprocess.run(["./getRipple.sh", projectPath.folder])
    with tables.open_file(projectPath.folder + 'nnSWR.mat', "a") as f:
        ripples = f.root.ripple[:, :].transpose()

        ## Let us create a simplifying .mat file for Karim:
        from scipy.io import savemat
        stackedlinearPos = np.transpose(linearPos,axes=[0,2,1])
        stackedlinearos = np.reshape(stackedlinearPos,[stackedlinearPos.shape[0]*stackedlinearPos.shape[1],stackedlinearPos.shape[2]])

        linearPreferredPos = trainerBayes.linearPreferredPos
        reorderdePlaceFields = trainerBayes.reorderdePlaceFields

        # todo : to change ....
        fig,ax = plt.subplots(8,8,figsize=(8,8))
        for i in range(8):
            for j in range(8):
                ax[i,j].imshow(reorderdePlaceFields[i*8+j,:,:])
                ax[i,j].set_title(str(i*8+j)+" | "+str(np.round(np.mean(reorderdePlaceFields[i*8+j,:,:]),2)))
        fig.tight_layout()
        fig.show()

        goodPlaceCells = np.array([3,6,11,14,15,23,43,51,59,63])

        gatheringDict = {"ripples":ripples,"NN_Decoded_Pos":stackedlinearos,"truePos":trueLinearPos,
               "spikesorted_popVector":spikeMat_window_popVector[:trueLinearPos.shape[0],:],
               "placeField":reorderdePlaceFields,"goodPlaceCells":goodPlaceCells,
                "putativeNeuronPreferedLinearPos":linearPreferredPos}
        if usetrain:
            savemat(os.path.join(projectPath.resultsPath, "uncertainty_network_fit","NNvsPopVector.mat"),gatheringDict)
        else:
            savemat(os.path.join(projectPath.resultsPath, "uncertainty_network_test", "NNvsPopVector.mat"),gatheringDict)

    # due to batching we need to remove the remainder:

    spikeMat_window_popVector = spikeMat_window_popVector[:trueProjPos.shape[0],1:]
    linearPreferredPos = trainerBayes.linearPreferredPos

    goodPlaceCells = [43]
    cm = plt.get_cmap("turbo")
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(timePreds,trueLinearPos,c="grey",alpha=0.8,label="true linear position")
    ax[0].scatter(timePreds,medianLinearPos,c="red",alpha=0.2,s=1,label="NN decoding")
    for i in goodPlaceCells:
        isSpiking =spikeMat_window_popVector[:,i]>0
        ax[0].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_popVector[isSpiking,i]/(np.max(spikeMat_window_popVector)),alpha=1)
    ax[0].set_title("NN decoding and raw firing rate")
    ax[0].set_ylabel("linear position \n cells ordered by prefered linear position")
    spikeMat_window_normalizedVector = (spikeMat_window_popVector-np.mean(spikeMat_window_popVector,axis=0))/np.std(spikeMat_window_popVector,axis=0)
    # remove the variance of the spiking
    ax[1].plot(timePreds,trueLinearPos,c="grey",alpha=0.8)
    ax[1].scatter(timePreds,medianLinearPos,c="red",alpha=0.2,s=1,label="NN decoding")
    for i in goodPlaceCells:
        # then scale between 0 and 1:
        spikeMat_window_normalizedVector_01 = (spikeMat_window_normalizedVector[:,i] - np.min(spikeMat_window_normalizedVector[:,i]))/np.max(spikeMat_window_normalizedVector[:,i])
        isSpiking = spikeMat_window_normalizedVector_01 > 0
        ax[1].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_normalizedVector_01[isSpiking] ,alpha=1)
    ax[1].set_ylabel("linear position \n cells ordered by prefered linear position")
    ax[1].set_title("NN decoding and normalized firing rate")
    fig.legend()
    fig.show()
    #
    # fig,ax = plt.subplots(2,1,sharex=True)
    # # ax[0].plot(timePreds,trueLinearPos,c="grey",alpha=0.1,label="true linear position")
    # # ax[0].scatter(timePreds,medianLinearPos,c="red",alpha=0.2,s=1,label="NN decoding")
    # for i in range(spikeMat_window_popVector.shape[1]):
    #     isSpiking =spikeMat_window_popVector[:,i]>0
    #     ax[0].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_popVector[isSpiking,i]/(np.max(spikeMat_window_popVector)),alpha=1)
    # ax[0].set_title("NN decoding and raw firing rate")
    # ax[0].set_ylabel("linear position \n cells ordered by prefered linear position")
    # spikeMat_window_normalizedVector = (spikeMat_window_popVector-np.mean(spikeMat_window_popVector,axis=0))/np.std(spikeMat_window_popVector,axis=0)
    # # remove the variance of the spiking
    # # ax[1].plot(timePreds,trueLinearPos,c="grey",alpha=0.1)
    # # ax[1].scatter(timePreds,medianLinearPos,c="red",alpha=0.2,s=1,label="NN decoding")
    # for i in range(spikeMat_window_popVector.shape[1]):
    #
    #     # then scale between 0 and 1:
    #     spikeMat_window_normalizedVector_01 = (spikeMat_window_normalizedVector[:,i] - np.min(spikeMat_window_normalizedVector[:,i]))/np.max(spikeMat_window_normalizedVector[:,i])
    #     isSpiking = spikeMat_window_normalizedVector_01 > 0
    #     ax[1].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_normalizedVector_01[isSpiking] ,alpha=1)
    # ax[1].set_ylabel("linear position \n cells ordered by prefered linear position")
    # ax[1].set_title("NN decoding and normalized firing rate")
    # fig.legend()
    # fig.show()




    # # Can we predict the population vector from the histogram of NN predictions?
    # from sklearn.linear_model import Ridge
    # from sklearn.model_selection import train_test_split
    # clf = Ridge(alpha=0.1)
    # X_train, X_test, Y_train, Y_test = train_test_split(histPosPred, spikeMat_window_popVector, train_size=0.5)
    # clf.fit(X_train, Y_train)
    # r2train = clf.score(X_train,Y_train)
    # r2test = clf.score(X_test,Y_test)
    # predfromTrain2 = clf.predict(X_test)
    # # fig, ax = plt.subplots()
    # # ax.scatter(predFromNN, predfromTrain2)
    # # fig.show()

    trainerBayes.bandwidth = 0.05
    from importData import  ImportClusters
    cluster_data = ImportClusters.load_spike_sorting(projectPath)
    bayesMatrices = trainerBayes.train(behavior_data, cluster_data)

    # TODO: Plot where we use speed filtering
    speedMask = behavior_data["Times"]["speedFilter"]
    speedTime = behavior_data["Position_time"]
    # align speedMask to NN time:
    st_lazy = pykeops.numpy.LazyTensor(speedTime[:, None])
    timepred_lazy = pykeops.numpy.Vj(timePreds.astype(dtype=np.float64)[:, None])
    res = (st_lazy - timepred_lazy).abs()
    bestSpeed = res.argmin_reduction(axis=0)
    speedFilter_window = speedMask[bestSpeed][:, 0]

    from importData.rawDataParser import  get_params
    _, samplingRate, _ = get_params(projectPath.xml)

    # bayeserror_on_movement = []
    # for n in tqdm(np.arange(0,stop=100,step=10)):
    #     outputsBayes = trainerBayes.test_as_NN(startTimeWindow[:trueProjPos.shape[0]].astype(dtype=np.float64)/samplingRate,bayesMatrices,behavior_data,cluster_data,windowSize=n * 0.036,masking_factor=1000,useTrain=usetrain) #using window size of 36 ms!
    #     pos =  outputsBayes["inferring"][:,0:2]
    #     proba = outputsBayes["inferring"][:, 2]
    #     bayesprojectedPos, bayeslinearPos = linearizationFunction(pos)
    #
    #     # fig,ax = plt.subplots()
    #     # ax.plot(timePreds,trueLinearPos,c="grey")
    #     # ax.scatter(timePreds,bayeslinearPos,c="navy",s=1,alpha=0.2)
    #     # ax.scatter(timePreds,medianLinearPos, c="red", s=1)
    #     # fig.show()
    #     bayeserror_on_movement += [np.mean(np.abs(trueLinearPos[speedFilter_window] - bayeslinearPos[speedFilter_window]))]
    #
    # nnerror_on_movement = np.mean(np.abs(trueLinearPos[speedFilter_window] - medianLinearPos[speedFilter_window]))
    #
    # fig,ax = plt.subplots()
    # ax.plot(np.arange(0,stop=100,step=10)*0.036,bayeserror_on_movement,c="navy",label="bayesian decoder")
    # ax.hlines(nnerror_on_movement,0,90*0.036,color="red",label="NN")
    # ax.set_xlabel("window size")
    # ax.set_ylabel("mean absolute linear decoding error (during movement)")
    # fig.legend()
    # fig.show()

    outputsBayes = trainerBayes.test_as_NN(
        startTimeWindow[:trueProjPos.shape[0]].astype(dtype=np.float64) / samplingRate, bayesMatrices, behavior_data,
        cluster_data, windowSize=30 * 0.036,masking_factor=1000,useTrain=usetrain)  # using window size of 36 ms!
    pos = outputsBayes["inferring"][:, 0:2]
    _,linearBayesPos = linearizationFunction(pos)
    proba = outputsBayes["inferring"][:, 2]

    fig,ax = plt.subplots()
    ax.scatter(timePreds,linearBayesPos,s=1,c="green")
    ax.plot(timePreds,trueLinearPos,c="grey",alpha=0.8)
    fig.show()


    # Is the entropy predictions correlated with the probability ?
    histlinearPosPred = np.stack(
        [np.histogram(linearTranspose[id, :], bins=np.arange(0, stop=1, step=0.01), density=True)[0] for id in
         range(linearTranspose.shape[0])])
    histlinearPosPred_density = histlinearPosPred / (np.sum(histlinearPosPred, axis=1)[:, None])
    def xlogx(x):
        y = np.zeros_like(x)
        y[np.greater(x, 0)] = np.log(x[np.greater(x, 0)]) * (x[np.greater(x, 0)])
        return y
    # let us compute the absolute error over each test Epochs:
    absError_epochs = []
    absError_epochs_mean = []
    names = []
    entropies_epochs_mean = []
    entropies_epochs = []
    proba_epochs = []
    proba_epochs_mean = []
    error_bayes_epochs = []
    testEpochs = behavior_data["Times"]["testEpochs"].copy()
    keptSession = behavior_data["Times"]["keptSession"]
    sessNames = behavior_data["Times"]["sessionNames"].copy()
    for idk, k in enumerate(keptSession.astype(np.bool)):
        if not k:
            sessNames.remove(behavior_data["Times"]["sessionNames"][idk])
    for x in behavior_data["Times"]["sleepNames"]:
        for id2, x2 in enumerate(sessNames):
            if x == x2:
                sessNames.remove(x2)
                # testEpochs[id2 * 2] = -1
                # testEpochs[id2 * 2 + 1] = -1
    # testEpochs = testEpochs[np.logical_not(np.equal(testEpochs, -1))]
    for i in range(int(len(testEpochs) / 2)):
        epoch = testEpochs[2 * i:2 * i + 2]
        maskEpoch = inEpochsMask(timePreds, epoch)
        maskTot = maskEpoch*(windowmask_speed[:,0].astype(np.bool))
        if np.sum(maskTot) > 0:
            absError_epochs_mean += [
                np.mean(np.abs(trueLinearPos[maskTot] - np.mean(linearTranspose[maskTot], axis=1)))]
            absError_epochs += [np.abs(trueLinearPos[maskTot] - np.mean(linearTranspose[maskTot], axis=1))]
            names += [sessNames[i]]
            entropies_epochs_mean += [np.mean(np.sum(-xlogx(histlinearPosPred_density[maskTot, :]), axis=1))]
            entropies_epochs += [np.sum(-xlogx(histlinearPosPred_density[maskTot, :]), axis=1)]
            proba_epochs += [proba[maskTot]]
            proba_epochs_mean += [np.mean(proba[maskTot])]
            error_bayes_epochs += [np.mean(np.abs(linearBayesPos[maskTot]-trueLinearPos[maskTot]))]
        else:
            print(sessNames[i])

    #Note: was old entropy!!
    # fig,ax = plt.subplots()
    # # ax.scatter(proba[np.logical_not(habEpochMaskandSpeed)], np.sum(-xlogx(histlinearPosPred_density), axis=1)[np.logical_not(habEpochMaskandSpeed)], s=1, alpha=0.2,c="red")
    # ax.scatter( proba[nothabEpochMaskandSpeed] ,np.sum(-xlogx(histlinearPosPred_density), axis=1)[nothabEpochMaskandSpeed],s=1,alpha=0.5)
    # ax.set_xlabel("bayesian proba")
    # ax.set_ylabel("entropy (lower is more confident)")
    # # ax.set_xscale("log")
    # # ax.set_xscale("log")
    # # ax.set_yscale("log")
    # fig.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(trueLinearPos[habEpochMaskandSpeed], np.mean(linearTranspose, axis=1)[habEpochMaskandSpeed], s=1, alpha=0.9)
    ax[1].scatter(trueLinearPos[habEpochMaskandSpeed], np.median(linearTranspose, axis=1)[habEpochMaskandSpeed], s=1, alpha=0.9)
    ax[2].scatter(trueLinearPos[habEpochMaskandSpeed], np.argmax(histlinearPosPred, axis=1)[habEpochMaskandSpeed], s=1,
                  alpha=0.9)
    ax[0].set_xlabel("mean linear pos")
    ax[1].set_xlabel("median linear pos")
    ax[2].set_xlabel("argmax linear pos")
    [a.set_ylabel("true linear pos") for a in ax]
    ax[0].set_title("habituation, high speed")
    fig.show()

    fig,ax = plt.subplots(2,1)
    ax[0].plot(timePreds[habEpochMask],trueLinearPos[habEpochMask],c="black")
    ax[0].scatter(timePreds[habEpochMask],medianLinearPos[habEpochMask],c="red",s=3)
    ax[0].set_title("habituation, all speed")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("linear position")
    ax[1].plot(timePreds[habEpochMaskandSpeed],trueLinearPos[habEpochMaskandSpeed],c="black")
    ax[1].scatter(timePreds[habEpochMaskandSpeed],medianLinearPos[habEpochMaskandSpeed],c="red",s=3)
    ax[1].set_title("habituation, high speed")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("linear position")
    fig.show()

    fig,ax = plt.subplots()
    ax.violinplot(np.array(proba_epochs),positions=range(len(proba_epochs_mean)))
    ax.plot(proba_epochs_mean)
    ax.set_ylabel("likelihood")
    ax.vlines(1.5,0,1,color="black")
    ax.text(1.5,1+0.01,"putative electrode movemement onset")
    ax.set_xticks(range(len(absError_epochs)))
    ax.set_xticklabels(names)
    # # ax.plot(error_bayes_epochs,c="red")
    # ax.vlines(1.5,0,np.max(error_bayes_epochs),color="black")
    # ax.text(1.5,np.max(error_bayes_epochs)+0.01,"putative electrode movemement onset")
    # # ax.set_ylabel("mean absolute error, bayes")
    fig.show()

    windowMask_speed = windowmask_speed[:,0].astype(np.bool)
    fig,ax = plt.subplots(2,1)
    ax[0].scatter(np.abs(linearBayesPos[nothabEpochMaskandSpeed]-trueLinearPos[nothabEpochMaskandSpeed]),
               proba[nothabEpochMaskandSpeed],s=1,alpha=0.3)
    ax[0].set_xlabel("linear error, bayesian decoder")
    ax[0].set_ylabel("proba, bayesian decoder")
    ax[0].set_title("after electrode move, speed filtered")
    ax[1].scatter(np.abs(linearBayesPos[nothabEpochMask]-trueLinearPos[nothabEpochMask]),
               proba[nothabEpochMask],s=1,alpha=0.3)
    ax[1].set_xlabel("linear error, bayesian decoder")
    ax[1].set_ylabel("proba, bayesian decoder")
    ax[1].set_title("after electrode move, not speed filtered")
    fig.tight_layout()
    fig.show()


    fig, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].scatter(range(len(absError_epochs)), absError_epochs_mean)
    ax[0].plot(absError_epochs_mean)
    ax[0].set_ylabel("absolute decoding error")
    ax[0].set_xticks(range(len(absError_epochs)))
    ax[0].set_xticklabels(names)
    ax[0].vlines(1.5,0,0.4,color="black")
    ax[0].text(1.5,0.4+0.01,"putative electrode movemement onset")

    ax[1].scatter(range(len(absError_epochs)), entropies_epochs_mean)
    ax[1].plot(entropies_epochs_mean)
    ax[1].violinplot(entropies_epochs, positions=range(len(absError_epochs)))
    ax[1].set_ylabel("mean entropies")
    ax[1].set_xticks(range(len(absError_epochs)))
    ax[1].set_xticklabels(names)
    ax[1].vlines(1.5,0,4,color="black")
    ax[1].text(1.5,4+0.01,"putative electrode movemement onset")
    fig.tight_layout()
    fig.show()


    fig,ax = plt.subplots()
    # ax.scatter(linearBayesPos,trueLinearPos,s=1,alpha = 0.2)
    ax.scatter(timePreds,linearBayesPos,c="green",s=1,alpha=0.5)
    ax.scatter(timePreds,trueLinearPos,c="grey",s=1,alpha=0.5)
    fig.show()


    cmR = plt.get_cmap("Reds")
    cmG = plt.get_cmap("Greens")
    entropies = np.sum(-xlogx(histlinearPosPred_density), axis=1)
    fig,ax = plt.subplots(2,1,sharex = True)
    ax[0].scatter(timePreds,medianLinearPos,c=cmR(entropies/np.max(entropies)),s=1,label="pos")
    ax[0].scatter(timePreds,linearBayesPos,c=cmG(proba),marker="x",label="proba",s=1)
    ax[0].scatter(timePreds,trueLinearPos,c="grey",alpha=0.5,s=1)
    ax[1].scatter(timePreds,-entropies,c="red",s=2)
    ax[1].twinx().scatter(timePreds, proba,c="green",s=2)
    fig.show()

    bayes_abserror = np.abs(trueLinearPos-linearBayesPos)[np.logical_not(habEpochMaskandSpeed)]
    nn_abserror = np.abs(trueLinearPos-medianLinearPos)[np.logical_not(habEpochMaskandSpeed)]
    corr = np.mean(((bayes_abserror-np.mean(bayes_abserror))/np.std(bayes_abserror))*(nn_abserror-np.mean(nn_abserror))/np.std(nn_abserror))
    fig,ax = plt.subplots()
    ax.scatter(np.abs(trueLinearPos-linearBayesPos),np.abs(trueLinearPos-medianLinearPos),s=1,alpha=0.1)
    fig.show()

    # ## Is the neural network dropout error histogram predictive of the bayesian decoding likelihood?
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import train_test_split
    clf = KernelRidge(alpha=1)
    X_train, X_test, Y_train, Y_test = train_test_split(histPosPred, proba, train_size=0.01)
    clf.fit(X_train, Y_train)
    r2train = clf.score(X_train,Y_train)
    r2test = clf.score(X_test,Y_test)
    predfromTrain2 = clf.predict(X_train)
    predfromTest = clf.predict(X_test)

    fig,ax = plt.subplots()
    # ax.scatter(Y_train,predfromTrain2,s=1)
    ax.scatter(Y_test, predfromTest, s=1)
    fig.show()

    habEpochsMask = inEpochsMask(timePreds,behavior_data["Times"]["testEpochs"][0:5])

    probaDecoding = trainerBayes.full_proba_decoding(startTimeWindow[:trueProjPos.shape[0]][habEpochsMask].astype(dtype=np.float64) / samplingRate, bayesMatrices, behavior_data,
        cluster_data, windowSize=30 * 0.036,masking_factor=1000,useTrain=usetrain)

    # probaDecoding = trainerBayes.full_proba_decoding(startTimeWindow[:trueProjPos.shape[0]].astype(dtype=np.float64) / samplingRate, bayesMatrices, behavior_data,
    #     cluster_data, windowSize=1 * 0.036,masking_factor=1000,useTrain=usetrain)

    # The bayesian prediction is made over a 80*80 grid, which can be projected onto the linear grid
    # We can obtain the mapping from the 80*80 grid to the linear coordinate:
    bins = bayesMatrices["Bins"]
    matPos = np.transpose([np.repeat(bins[1], len(bins[0])),np.tile(bins[0], len(bins[1]))])
    fig,ax = plt.subplots()
    ax.scatter(matPos[:,0],matPos[:,1],s=1,c=cm(np.arange(matPos.shape[0])/matPos.shape[0]))
    fig.show()

    _,mappingLinearPos = linearizationFunction(matPos)
    basisChangeMat = np.zeros([matPos.shape[0],np.unique(mappingLinearPos).shape[0]])
    for id,linearCoord in enumerate(np.unique(mappingLinearPos)):
        basisChangeMat[np.equal(mappingLinearPos,linearCoord),id] = 1
    probaDecoding = np.array(probaDecoding)
    probaDecoding_cat = probaDecoding.reshape([probaDecoding.shape[0],basisChangeMat.shape[0]])

    linearProbaDecoding = np.matmul(probaDecoding_cat,basisChangeMat)

    # fig,ax = plt.subplots()
    # ax.matshow(mappingLinearPos.reshape([80,80]))
    # fig.show()
    #
    # fig,ax = plt.subplots()
    # ax.matshow(basisChangeMat)
    # ax.set_aspect(basisChangeMat.shape[1]/basisChangeMat.shape[0])
    # fig.show()
    #
    # fig,ax = plt.subplots(3,1)
    # ax[0].imshow(probaDecoding[1000,:,:])
    # ax[1].plot(probaDecoding_cat[1000, :])
    # ax[2].plot(linearProbaDecoding[1000,:])
    # fig.show()

    linearProbaDecoding_best = np.unique(mappingLinearPos)[np.argmax(linearProbaDecoding,axis=1)]
    allProba = [np.unravel_index(np.argmax(probaDecoding[bin]), probaDecoding[bin].shape) for bin in
    			range(probaDecoding.shape[0])]
    allProba2 = [np.argmax(probaDecoding[bin]) for bin in
    			range(probaDecoding.shape[0])]
    allProba = np.array(allProba)
    allProba2 = np.array(allProba2)
    fig,ax = plt.subplots()
    ax.scatter(allProba[:,0],allProba[:,1],c=cm(allProba2/(80.0*80.0)))
    fig.show()


    bestProba = [np.max(probaDecoding[bin]) for bin in range(probaDecoding.shape[0])]
    position_guessed = [[bayesMatrices['Bins'][i][allProba[bin][i]] for i in range(len(bayesMatrices['Bins']))]
    					for bin in range(probaDecoding.shape[0])]
    position_guessed = np.array(position_guessed)
    _,linearpos_guessed = linearizationFunction(position_guessed)

    # fig,ax = plt.subplots()
    # ax.scatter(position_guessed[:,0],position_guessed[:,1])
    # fig.show()
    # fig,ax = plt.subplots()
    # ax.scatter(linearpos_guessed,linearProbaDecoding_best,s=1)
    # fig.show()

    fig,ax = plt.subplots()
    ax.scatter(timePreds,linearpos_guessed,c="green",s=1)
    ax.scatter(timePreds,linearProbaDecoding_best,c="navy",s=1)
    ax.plot(timePreds,trueLinearPos)
    fig.show()


    histLinearPosPred = np.stack([np.histogram(linearTranspose[id, :],
                                         bins=np.arange(0, stop=1, step=0.01),density=True)[0] for id in
                            range(linearTranspose.shape[0])])
    clf = Ridge(alpha=0)
    X_train, X_test, Y_train, Y_test = train_test_split(histLinearPosPred, linearProbaDecoding, train_size=0.8)
    clf.fit(X_train, Y_train)
    r2train = clf.score(X_train,Y_train)
    r2test = clf.score(X_test,Y_test)
    predfromTrain2 = clf.predict(X_train)
    predfromTest = clf.predict(X_test)

    CC = np.matmul(np.transpose(histPosPred),linearProbaDecoding)

    rlinearProbaDecoding = linearProbaDecoding
    rhistPosPred = histLinearPosPred
    fig,ax = plt.subplots(1,2,sharey=True)
    ax[0].imshow(rlinearProbaDecoding)
    ax[0].set_title("Bayesian decoding")
    ax[1].imshow(rhistPosPred)
    ax[1].set_title("NN decoding")
    ax[0].set_aspect(rlinearProbaDecoding.shape[1]/rlinearProbaDecoding.shape[0])
    ax[1].set_aspect(rhistPosPred.shape[1]/rhistPosPred.shape[0])
    ax[0].plot(trueLinearPos,np.arange(rhistPosPred.shape[0]),c="red")
    ax[0].set_ylabel("time step")
    ax[0].set_xlabel("linear position")
    ax[1].set_xlabel("linear position")
    fig.show()

    fig,ax = plt.subplots(2,1,sharex=True)
    cm = plt.get_cmap("Reds")
    for i in range(rlinearProbaDecoding.shape[1]):
        ax[0].scatter(timePreds,np.unique(mappingLinearPos)[i]+np.zeros_like(timePreds),
                      c=cm(rlinearProbaDecoding[:,i]),s=1)
    ax[0].set_title("Bayesian decoding")
    linearvariable = np.arange(0,stop=1,step=0.01)
    for i in range(rhistPosPred.shape[1]):
        ax[1].scatter(timePreds,linearvariable[i]+np.zeros_like(timePreds),
                      c=cm(rhistPosPred[:,i]/(np.max(rhistPosPred[:,i]))),s=1)
    ax[1].set_title("NN decoding")
    # ax[0].set_aspect(rlinearProbaDecoding.shape[1]/rlinearProbaDecoding.shape[0])
    # ax[1].set_aspect(rhistPosPred.shape[1]/rhistPosPred.shape[0])
    ax[0].plot(timePreds,trueLinearPos,c="grey")
    ax[1].plot(timePreds,trueLinearPos, c="grey")
    ax[0].set_xlabel("time step")
    ax[0].set_ylabel("linear position")
    ax[1].set_xlabel("time step")
    fig.show()

    toDisplay = np.arange(4000,stop=5000)
    fig,ax = plt.subplots(2,1,sharex=True)
    cm = plt.get_cmap("Reds")
    for i in range(rlinearProbaDecoding.shape[1]):
        ax[0].scatter(timePreds[toDisplay],np.unique(mappingLinearPos)[i]+np.zeros_like(timePreds[toDisplay]),
                      c=cm(rlinearProbaDecoding[toDisplay,i]/(np.max(rlinearProbaDecoding[toDisplay,i]))),s=1)
    ax[0].set_title("Bayesian decoding, window size= 0.036s")
    linearvariable = np.arange(0,stop=1,step=0.01)
    for i in range(rhistPosPred.shape[1]):
        ax[1].scatter(timePreds[toDisplay],linearvariable[i]+np.zeros_like(timePreds[toDisplay]),
                      c=cm(rhistPosPred[toDisplay,i]/(np.max(rhistPosPred[toDisplay,i]))),s=1)
    ax[1].set_title("NN decoding, window size = 0.036 s")
    # ax[0].set_aspect(rlinearProbaDecoding.shape[1]/rlinearProbaDecoding.shape[0])
    # ax[1].set_aspect(rhistPosPred.shape[1]/rhistPosPred.shape[0])
    ax[0].plot(timePreds[toDisplay],trueLinearPos[toDisplay],c="grey")
    ax[1].plot(timePreds[toDisplay],trueLinearPos[toDisplay], c="grey")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("linear position")
    ax[1].set_xlabel("time (s)")
    fig.show()


    # Let us compute the cross-correlation of each density vector, in the prediction of
    # # the neural network:
    # rhistlinearPosPred_density = histlinearPosPred_density[nothabEpochMask]
    # cross_corr = np.matmul((rhistlinearPosPred_density-np.mean(rhistlinearPosPred_density,axis=1)[:,None])/(np.std(rhistlinearPosPred_density,axis=1)[:,None]),
    #                        np.transpose((rhistlinearPosPred_density-np.mean(rhistlinearPosPred_density,axis=1)[:,None])/(np.std(rhistlinearPosPred_density,axis=1)[:,None])))
    # N = 100
    # z = np.zeros([cross_corr.shape[0],N])
    # def modInd(x):
    #     y = np.zeros_like(x)
    #     y[np.less(x,0)] = x[np.less(x,0)]
    #     y[np.logical_not(np.less(x,0))] = np.mod(x[np.logical_not(np.less(x,0))],cross_corr.shape[1])
    #     return y
    # for idd in range(cross_corr.shape[0]):
    #     z[idd,:] = cross_corr[idd,modInd(np.arange(idd-int(N/2),idd+int(N/2)))]
    #
    # fig,ax = plt.subplots()
    # ax.matshow(cross_corr)
    # ax.set_aspect(z.shape[1]/z.shape[0])
    # fig.show()
    #
    # fig,ax = plt.subplots()
    # ax.matshow(cross_corr)
    # fig.show()
    #
    # cross_corr_linearPos = np.matmul(np.transpose((rhistPosPred-np.mean(rhistPosPred,axis=0))/(np.std(rhistPosPred,axis=0))),
    #                        (rhistPosPred-np.mean(rhistPosPred,axis=0))/(np.std(rhistPosPred,axis=0)))
    # fig,ax = plt.subplots()
    # ax.matshow(cross_corr_linearPos)
    # fig.show()


    fig,ax = plt.subplots()
    # ax.vlines(-1800,ymin=0,ymax=100,color="orange")
    # ax.vlines(1800, ymin=0, ymax=100,color="orange")
    ax.scatter(np.arange(-int(N/2),int(N/2)),np.mean(z,axis=0),s=1)
    ax.set_xlabel("window delay (ms)")
    ax.set_ylabel("cross-correlation")
    fig.show()



    fig,ax = plt.subplots()
    ax.plot(rlinearProbaDecoding[10,:])
    ax.plot((rhistPosPred)[10,:],c="red")
    fig.show()


    ### ================ Let us Normalize the entropy given a predicted position ================

    ent = np.sum(-xlogx(histlinearPosPred_density), axis=1)
    trainingEntropies = ent[habEpochMaskandSpeed]
    trainingLinearPos = medianLinearPos[habEpochMaskandSpeed]
    binsLinearPos = np.arange(0,stop=np.max(linearPos)+0.1,step=0.01)
    trainingEntropies_givenPos = [trainingEntropies[np.greater_equal(trainingLinearPos,binsLinearPos[id])
                                                    *np.less(trainingLinearPos,binsLinearPos[id+1])] for id in range(len(binsLinearPos)-1)]
    pos_given_pos = [trainingLinearPos[np.greater_equal(trainingLinearPos,binsLinearPos[id])
                                                    *np.less(trainingLinearPos,binsLinearPos[id+1])] for id in range(len(binsLinearPos)-1)]
    trainingEntropies_givenPos_normalize = [(np.array(t)-np.mean(t))/np.std(t) for t in trainingEntropies_givenPos]

    fig,ax = plt.subplots(1,2)
    for t,b in zip(pos_given_pos,trainingEntropies_givenPos_normalize):
        ax[1].scatter(t,b,s=1)
    for t,b in zip(pos_given_pos,trainingEntropies_givenPos):
        ax[0].scatter(t,b,s=1)
    ax[0].set_xlabel("linear position")
    ax[0].set_ylabel("entropy")
    ax[1].set_xlabel("linear position")
    ax[1].set_ylabel("entropy, normalized given position")
    fig.show()



    normalized_entropied = np.zeros_like(ent)
    for id in range(len(binsLinearPos)-1):
        ent_given_pos = ent[np.greater_equal(medianLinearPos,binsLinearPos[id])
                                                    *np.less(medianLinearPos,binsLinearPos[id+1])]
        normalized_entropied[np.greater_equal(medianLinearPos,binsLinearPos[id])
                            *np.less(medianLinearPos,binsLinearPos[id+1])] =  (ent_given_pos - np.mean(trainingEntropies_givenPos[id]))/np.std(trainingEntropies_givenPos[id])
    fig,ax = plt.subplots()
    ax.scatter(medianLinearPos,normalized_entropied,s=1,c="black",alpha=0.2)
    fig.show()

    fig,ax = plt.subplots()
    # ax.scatter(proba[np.logical_not(habEpochMaskandSpeed)], np.sum(-xlogx(histlinearPosPred_density), axis=1)[np.logical_not(habEpochMaskandSpeed)], s=1, alpha=0.2,c="red")
    ax.scatter( proba[habEpochMaskandSpeed] ,normalized_entropied[habEpochMaskandSpeed],s=2,alpha=0.2)
    #todo: density plot here...
    ax.set_xlabel("bayesian proba")
    ax.set_ylabel("entropy (lower is more confident)")
    ax.set_title("habituation, speed filtered")
    fig.show()

    fig,ax = plt.subplots()
    # ax.scatter(proba[np.logical_not(habEpochMaskandSpeed)], np.sum(-xlogx(histlinearPosPred_density), axis=1)[np.logical_not(habEpochMaskandSpeed)], s=1, alpha=0.2,c="red")
    ax.scatter( proba[habEpochMask] ,normalized_entropied[habEpochMask] ,s=2,alpha=1)
    ax.set_xlabel("bayesian proba")
    ax.set_ylabel("entropy (lower is more confident)")
    ax.set_title("habituation, not speed filtered")
    fig.show()

    # let us compute a moving average of the entropy:
    ent_moveAvg = np.array([np.mean(normalized_entropied[id:id + 10]) for id in range(normalized_entropied.shape[0])])
    ent_moveAvg20 = np.array([np.mean(normalized_entropied[id:id + 20]) for id in range(normalized_entropied.shape[0])])
    ent_moveAvg30 = np.array([np.mean(normalized_entropied[id:id + 30]) for id in range(normalized_entropied.shape[0])])

    cm = plt.get_cmap("Reds")
    # lookAt = np.arange(50000,stop=51000,step=1)
    lookAt = np.arange(1000, stop=2000, step=1)
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].scatter(timePreds[lookAt],proba[lookAt],s=1,c="green",alpha=0.5)
    ax[0].twinx().plot(timePreds[lookAt],trueLinearPos[lookAt],c="black")
    ax[0].set_ylabel("proba")
    ax[0].set_xlabel("time")
    ax[1].scatter(timePreds[lookAt],normalized_entropied[lookAt],s=1,c="red",alpha=0.5)
    ax[1].plot(timePreds[lookAt],ent_moveAvg[lookAt],c="purple",label="mva, 10 window")
    ax[1].plot(timePreds[lookAt], ent_moveAvg20[lookAt], c="violet",label="mva, 20 window")
    ax[1].plot(timePreds[lookAt], ent_moveAvg30[lookAt], c="navy",alpha=0.7,label="mva, 30 window")
    # ax[1].twinx().plot(timePreds[lookAt],trueLinearPos[lookAt],c="black")
    # ax[1].twinx().plot(timePreds[lookAt], medianLinearPos[lookAt], c="orange")
    ax[1].set_ylabel("entropy")
    ax[1].set_xlabel("time")
    fig.legend()
    for i in range(histlinearPosPred_density[lookAt].shape[0]):
        ax[2].scatter(np.zeros(99)+timePreds[lookAt][i],np.arange(0, stop=1, step=0.01)[:-1],s=3,c=cm(histlinearPosPred_density[lookAt][i,:]))
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("histograms of predictions")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(ent_moveAvg30,proba,alpha=0.4)
    fig.show()


    lookAt = np.arange(50000,stop=54000,step=1)
    cm = plt.get_cmap("Reds")
    fig,ax = plt.subplots(2,1)
    filter = np.less(normalized_entropied[lookAt],0)
    filterHigh = np.greater_equal(normalized_entropied[lookAt], 0)
    for i in range(histlinearPosPred_density[lookAt][filter,:].shape[0]):
        ax[0].scatter(np.zeros(99) + timePreds[lookAt][filter][i], np.arange(0, stop=1, step=0.01)[:-1], s=3,
                      c=cm(histlinearPosPred_density[lookAt][filter,:][i, :]))
    for i in range(histlinearPosPred_density[lookAt][filterHigh,:].shape[0]):
        ax[1].scatter(np.zeros(99) + timePreds[lookAt][filterHigh][i], np.arange(0, stop=1, step=0.01)[:-1], s=3,
                      c=cm(histlinearPosPred_density[lookAt][filterHigh,:][i, :]))
    ax[0].plot(timePreds[lookAt],trueLinearPos[lookAt],c="black",alpha=0.5)
    fig.show()


    cm = plt.get_cmap("tab10")
    fig,ax = plt.subplots(5,2)
    for i in range(10):
        ax[int(i/2),i%2].plot(np.arange(0, stop=1, step=0.01)[:-1],histlinearPosPred_density[60000+i,:],c=cm(i))
        b = ax[int(i/2),i%2].twinx()
        b.hlines(normalized_entropied[60000+i],0,1,color=cm(i),linestyle="--")
        b.set_ylim(-3,3)
        b.set_ylabel("entropy")
        ax[int(i/2),i%2].set_xlabel("linear pos")
        ax[int(i / 2), i % 2].set_ylabel("density")
        ax[int(i / 2), i % 2].set_ylim(0,0.5)
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied[lookAt],np.abs(linearBayesPos-trueLinearPos)[lookAt])
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied[nothabEpochMaskandSpeed],np.abs(linearBayesPos-trueLinearPos)[nothabEpochMaskandSpeed],s=1)
    fig.show()

    error_filtered=[]
    npPoints = []
    for filter in np.arange(-2,2,step=0.1):
        filtered_ent = np.less(normalized_entropied,filter)
        # filtered_ent = np.greater(normalized_entropied,1.5)
        mask_nothabSpeedFilter = filtered_ent*nothabEpochMask*np.logical_not(windowMask_speed)
        error_filtered += [np.mean(np.abs(medianLinearPos[mask_nothabSpeedFilter]-trueLinearPos[mask_nothabSpeedFilter]))]
        npPoints += [np.sum(mask_nothabSpeedFilter)]
    fig,ax=  plt.subplots()
    ax.plot(np.arange(-2,2,step=0.1),error_filtered)
    # ax.twinx().plot(np.arange(-2, 2, step=0.1), npPoints,c="red")
    ax.set_xlabel("entropy filtering value")
    ax.set_ylabel("mean absolute error")
    ax.set_title("post electrode move (where network is trained), test set, low speed")
    fig.show()

    error_filtered=[]
    npPoints = []
    for filter in np.arange(-2,2,step=0.1):
        filtered_ent = np.less(normalized_entropied,filter)
        # filtered_ent = np.greater(normalized_entropied,1.5)
        mask_nothabSpeedFilter = filtered_ent*nothabEpochMask*np.logical_not(windowMask_speed)
        error_filtered += [np.mean(np.abs(medianLinearPos[mask_nothabSpeedFilter]-trueLinearPos[mask_nothabSpeedFilter]))]
        npPoints += [np.sum(mask_nothabSpeedFilter)]
    fig,ax=  plt.subplots()
    ax.plot(np.arange(-2,2,step=0.1),error_filtered)
    # ax.twinx().plot(np.arange(-2, 2, step=0.1), npPoints,c="red")
    ax.set_xlabel("entropy filtering value")
    ax.set_ylabel("mean absolute error")
    ax.set_title("post electrode move (where network is trained), test set, high speed")
    fig.show()


    error_filtered=[]
    npPoints = []
    for filter in np.arange(-2,3,step=0.1):
        filtered_ent = np.less(normalized_entropied,filter)
        # filtered_ent = np.greater(normalized_entropied,1.5)
        mask_habSpeedFilter = filtered_ent*habEpochMaskandSpeed
        error_filtered += [np.mean(np.abs(medianLinearPos[mask_habSpeedFilter]-trueLinearPos[mask_habSpeedFilter]))]
        npPoints += [np.sum(mask_habSpeedFilter)]
    fig,ax=  plt.subplots()
    ax.plot(np.arange(-2,3,step=0.1),error_filtered)
    axt = ax.twinx()
    axt.plot(np.arange(-2,3,step=0.1), npPoints,c="red")
    axt.set_ylabel("number of window",color="red")
    ax.set_xlabel("entropy filtering value")
    ax.set_ylabel("mean absolute error")
    ax.set_title("before electrode move (where network is trained), test set, high speed")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied[nothabEpochMaskandSpeed],np.abs(medianLinearPos[nothabEpochMaskandSpeed]-trueLinearPos[nothabEpochMaskandSpeed]),s=2)
    fig.show()

    filtered_ent = np.less(normalized_entropied, 0.2)
    # filtered_ent = np.greater(normalized_entropied,1.5)
    mask_nothabSpeedFilter = filtered_ent * habEpochMaskandSpeed

    fig,ax = plt.subplots()
    ax.scatter(medianLinearPos[mask_nothabSpeedFilter],trueLinearPos[mask_nothabSpeedFilter],s=1)
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(timePreds[mask_nothabSpeedFilter],medianLinearPos[mask_nothabSpeedFilter],alpha=0.6,c="red")
    ax.plot(timePreds[mask_nothabSpeedFilter],trueLinearPos[mask_nothabSpeedFilter],alpha=0.6,c="black")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(proba[nothabEpochMaskandSpeed],ent_moveAvg30[nothabEpochMaskandSpeed],s=1)
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(linearBayesPos[mask_nothabSpeedFilter],medianLinearPos[mask_nothabSpeedFilter],s=1)
    fig.show()


    ## AUTOCORR OF THE ENTROPY
    autocorrs = np.zeros(400)
    for i in tqdm(np.arange(200,stop=0,step=-1)):
        autocorrs[200-i] = np.mean(normalized_entropied[200:]*normalized_entropied[200-i:normalized_entropied.shape[0]-i])
        autocorrs[200+i-1] = np.mean(normalized_entropied[:normalized_entropied.shape[0]-200] * normalized_entropied[i:normalized_entropied.shape[0]-200+i])
    fig,ax = plt.subplots()
    ax.plot(autocorrs)
    fig.show()



    ## using the maxima rather than the entropy:
    proba_NN = histlinearPosPred_density.max(axis=1)
    fig,ax = plt.subplots()
    ax.scatter(proba_NN[nothabEpochMaskandSpeed],proba[nothabEpochMaskandSpeed],s=1,alpha=0.5)
    ax.set_xlabel("max proba, NN, not normalized")
    ax.set_ylabel("max proba, bayesian")
    ax.set_title("post electrode movement")
    fig.show()
    fig,ax = plt.subplots()
    ax.scatter(proba_NN[habEpochMask],proba[habEpochMask],s=1,alpha=0.5)
    ax.set_xlabel("max proba, NN, not normalized")
    ax.set_ylabel("max proba, bayesian")
    ax.set_title("habituation")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(medianLinearPos[nothabEpochMaskandSpeed],proba_NN[nothabEpochMaskandSpeed])
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied[nothabEpochMaskandSpeed],proba_NN[nothabEpochMaskandSpeed])
    ax.set_xlabel("")
    fig.show()



    from SimpleBayes import butils
    ### =========== Let us try to link the decoding probability with the number of spikes in the window:
    fig,ax = plt.subplots(2,1)
    ax[0].scatter(normalized_entropied,lenInputNN[:ent.shape[0]][:,0],s=1,alpha=0.2)
    normalize = lambda x : (x-np.mean(x))/np.std(x)
    _, plotDensity = butils.kdenD(np.stack([normalized_entropied,normalize(lenInputNN[:ent.shape[0]][:,0])]).transpose(),
         0.1, kernel="gaussian", nbins=[20,20])
    ax[1].matshow(plotDensity.transpose(),cmap=plt.get_cmap("Reds"),origin="lower")
    ax[0].set_ylabel("number of spike (raw filter)")
    ax[0].set_xlabel("entropy, normalized given position")
    fig.tight_layout()
    fig.show()


    lenInputBayes = np.sum(spikeMat_window_popVector,axis=1)
    outputsBayes_oneWindow = trainerBayes.test_as_NN(
        startTimeWindow[:trueProjPos.shape[0]].astype(dtype=np.float64) / samplingRate, bayesMatrices, behavior_data,
        cluster_data, windowSize=0.036,masking_factor=1000,useTrain=usetrain)  # using window size of 36 ms!
    proba_oneWindow = outputsBayes_oneWindow["inferring"][:, 2]
    pos_oneWindow = outputsBayes_oneWindow["inferring"][:, :2]
    _,linear_bayes_one_window = linearizationFunction(pos_oneWindow)

    ### MSE error without ensemble averaging:
    NN_linear_pos = linearTranspose[:,0]
    print(np.mean(np.abs(medianLinearPos-trueLinearPos)[nothabEpochMaskandSpeed]))
    print(np.mean(np.abs(linear_bayes_one_window-trueLinearPos)[nothabEpochMaskandSpeed]))
    print(np.mean(np.abs(medianLinearPos-trueLinearPos)[habEpochMask]))
    print(np.mean(np.abs(linear_bayes_one_window-trueLinearPos)[habEpochMask]))


    lenInputBayes_30window  = np.array([np.sum(lenInputBayes[id:id+30]) for id in range(len(lenInputBayes))])
    fig,ax = plt.subplots(1,2)
    ax[0].scatter(normalized_entropied[nothabEpochMaskandSpeed], lenInputBayes[nothabEpochMaskandSpeed], s=1, alpha=0.1)
    ax[0].set_title("36ms windows")
    ax[1].scatter(normalized_entropied[nothabEpochMaskandSpeed],lenInputBayes_30window[nothabEpochMaskandSpeed],s=1,alpha=0.1)
    ax[1].set_title("30 * 36ms windows")
    ax[0].set_ylabel("number of spike (sorted)")
    ax[0].set_xlabel("entropy")
    ax[1].set_xlabel("entropy")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(proba[nothabEpochMaskandSpeed],np.abs(linearBayesPos-trueLinearPos)[nothabEpochMaskandSpeed],s=1,alpha=0.2)
    ax.set_xlabel("proba")
    fig.show()
    fig,ax = plt.subplots()
    ax.scatter(proba,np.abs(linearBayesPos-trueLinearPos),s=1,alpha=0.2)
    ax.set_xlabel("proba")
    fig.show()


    lenInputNN_30window = [np.sum(lenInputNN[id:id + 30]) for id in range(len(lenInputNN))]
    fig,ax = plt.subplots()
    ax.scatter(lenInputNN_30window[:ent.shape[0]],lenInputBayes_30window,s=1,alpha=0.1)
    ax.set_aspect(np.max(lenInputBayes_30window)/np.max(lenInputNN_30window[:ent.shape[0]]))
    fig.show()

    fig,ax = plt.subplots(1,2)
    ax[0].scatter(proba_oneWindow[nothabEpochMaskandSpeed], lenInputBayes[nothabEpochMaskandSpeed], s=1, alpha=0.6)
    ax[0].set_title("36ms windows")
    ax[1].scatter(proba[nothabEpochMaskandSpeed],lenInputBayes_30window[nothabEpochMaskandSpeed],s=1,alpha=0.6)
    ax[1].set_title("30 * 36ms windows")
    ax[0].set_ylabel("number of spike (sorted)")
    ax[0].set_xlabel("bayesian probability")
    ax[1].set_xlabel("bayesian probability")
    fig.show()

    fig,ax = plt.subplots(1,2)
    ax[0].scatter(np.abs(linearBayesPos-trueLinearPos)[nothabEpochMaskandSpeed], lenInputBayes[nothabEpochMaskandSpeed], s=1, alpha=0.6)
    ax[0].set_title("36ms windows")
    ax[1].scatter(np.abs(linearBayesPos-trueLinearPos)[nothabEpochMaskandSpeed],lenInputBayes_30window[nothabEpochMaskandSpeed],s=1,alpha=0.6)
    ax[1].set_title("30 * 36ms windows")
    ax[0].set_ylabel("number of spike (sorted)")
    ax[0].set_xlabel("bayesian error")
    ax[1].set_xlabel("bayesian error")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(linearBayesPos[np.logical_not(habEpochMask)],proba[np.logical_not(habEpochMask)],s=1,alpha=0.5)
    fig.show()
    fig, ax = plt.subplots()
    ax.hist(trueLinearPos[np.logical_not(habEpochMask)],bins=50)
    fig.show()



    fig,ax = plt.subplots(1,2,figsize=(10,5),sharey=True)
    ax[0].hist(ent[habEpochMask], bins=100, color="black", density=True, histtype="step",label="habituation")
    ax[0].hist(ent[habEpochMaskandSpeed],bins=100,color="red",alpha=0.4,density=True,label="habituation, high speed")
    ax[0].hist(ent[habEpochMask][np.logical_not(windowmask_speed[habEpochMask][:,0].astype(np.bool))], bins=100,color="blue",alpha=0.5,density=True,label="habituation low speed")
    ax[0].set_ylabel("density")
    ax[0].set_xlabel("entropy")
    ax[1].hist(normalized_entropied[habEpochMask], bins=100, color="black", density=True,histtype="step")
    ax[1].hist(normalized_entropied[habEpochMaskandSpeed],bins=100,color="red",alpha=0.4,density=True)
    ax[1].hist(normalized_entropied[habEpochMask][np.logical_not(windowmask_speed[habEpochMask][:,0].astype(np.bool))], bins=100,color="blue",alpha=0.5,density=True)
    ax[1].set_xlabel("entropy, modified \n by normalizing entropy given predicted position ")
    fig.legend()
    fig.show()

    fig,ax = plt.subplots(1,2,figsize=(10,5),sharey=True)
    ax[0].hist(ent[np.logical_not(habEpochMask)], bins=100, color="black", density=True, histtype="step",label="not hab")
    ax[0].hist(ent[nothabEpochMaskandSpeed],bins=100,color="red",alpha=0.4,density=True,label="not hab, high speed")
    ax[0].hist(ent[np.logical_not(habEpochMask)][np.logical_not(windowmask_speed[np.logical_not(habEpochMask)][:,0].astype(np.bool))], bins=100,color="blue",alpha=0.5,density=True,label="not hab low speed")
    ax[0].set_ylabel("density")
    ax[0].set_xlabel("entropy")
    ax[1].hist(normalized_entropied[np.logical_not(habEpochMask)], bins=100, color="black", density=True,histtype="step")
    ax[1].hist(normalized_entropied[nothabEpochMaskandSpeed],bins=100,color="red",alpha=0.4,density=True)
    ax[1].hist(normalized_entropied[np.logical_not(habEpochMask)][np.logical_not(windowmask_speed[np.logical_not(habEpochMask)][:,0].astype(np.bool))], bins=100,color="blue",alpha=0.5,density=True)
    ax[1].set_xlabel("entropy, modified \n by normalizing entropy given predicted position ")
    fig.legend()
    fig.show()

    cm = plt.get_cmap("turbo")
    fig,ax = plt.subplots()
    minEnt = np.mean(normalized_entropied[np.logical_not(np.isnan(normalized_entropied))])
    maxEnt = np.std(normalized_entropied[np.logical_not(np.isnan(normalized_entropied))])
    ax.scatter(timePreds,medianLinearPos,c=cm((normalized_entropied-minEnt)/maxEnt),s=1)
    fig.show()

    fig,ax = plt.subplots(1,3)
    ax[0].scatter(ent[habEpochMaskandSpeed],
               np.abs(trueLinearPos - medianLinearPos)[habEpochMaskandSpeed], s=1, alpha=0.2)
    ax[0].set_xlabel("entropy")
    ax[0].set_ylabel("linear error")
    ax[2].scatter(normalized_entropied[habEpochMask][np.logical_not(windowmask_speed[habEpochMask][:,0].astype(np.bool))],
                  np.abs(trueLinearPos-medianLinearPos)[habEpochMask][np.logical_not(windowmask_speed[habEpochMask][:,0].astype(np.bool))],s=1,alpha=0.2,c="red")
    ax[1].scatter(normalized_entropied[habEpochMaskandSpeed],
                  np.abs(trueLinearPos - medianLinearPos)[habEpochMaskandSpeed], s=1, alpha=0.2)
    ax[1].set_xlabel("entropy, normalized conditioned on pos")
    ax[1].set_ylabel("linear error")
    ax[2].set_xlabel("entropy, normalized conditioned on pos")
    ax[2].set_title("low speed, hab")
    ax[1].set_title("high speed, hab")
    fig.show()

    fig,ax = plt.subplots(1,3)
    ax[0].scatter(ent[nothabEpochMaskandSpeed],
               np.abs(trueLinearPos - medianLinearPos)[nothabEpochMaskandSpeed], s=1, alpha=0.2)
    ax[0].set_xlabel("entropy")
    ax[0].set_ylabel("linear error")
    ax[2].scatter(normalized_entropied[np.logical_not(habEpochMask)][np.logical_not(windowmask_speed[np.logical_not(habEpochMask)][:,0].astype(np.bool))],
                  np.abs(trueLinearPos-medianLinearPos)[np.logical_not(habEpochMask)][np.logical_not(windowmask_speed[np.logical_not(habEpochMask)][:,0].astype(np.bool))],s=1,alpha=0.2,c="red")
    ax[1].scatter(normalized_entropied[nothabEpochMaskandSpeed],
                  np.abs(trueLinearPos - medianLinearPos)[nothabEpochMaskandSpeed], s=1, alpha=0.2)
    ax[1].set_xlabel("entropy, normalized conditioned on pos")
    ax[1].set_ylabel("linear error")
    ax[2].set_xlabel("entropy, normalized conditioned on pos")
    ax[2].set_title("low speed, not hab")
    ax[1].set_title("high speed, not hab")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied[np.logical_not(habEpochMask)*(windowmask_speed[:,0].astype(np.bool))],
               np.abs(trueLinearPos - medianLinearPos)[np.logical_not(habEpochMask)*(windowmask_speed[:,0].astype(np.bool))], s=4, alpha=0.2)
    ax.set_xlabel("entropy, normalized conditioned on pos")
    ax.set_ylabel("linear error")
    ax.set_title("high speed; after electrode movement")
    fig.show()

    minEnt = np.min(normalized_entropied[np.logical_not(np.isnan(normalized_entropied))])
    maxEnt = np.max(normalized_entropied[np.logical_not(np.isnan(normalized_entropied))])
    binsEntropy = np.arange(minEnt,stop=maxEnt,step=1)
    delta = np.abs(trueLinearPos - medianLinearPos)
    error = [delta[np.greater(normalized_entropied,binsEntropy[id])*
                   np.less(normalized_entropied,binsEntropy[id+1]) ] for id in range(len(binsEntropy)-1)]
    fig,ax = plt.subplots()
    ax.violinplot(error)
    fig.show()

    df_old_ent = pd.read_csv('/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-reversal/dataForKarim/normalize_entropy.csv')
    oldEnt = df_old_ent.values[:,1:]
    timePredsOld = pd.read_csv('/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-reversal/dataForKarim/uncertainty_network_test/timePreds.csv')
    timesPredsOld =  timePredsOld.values[:,1:][:,0]

    fig,ax = plt.subplots()
    ax.scatter(timesPredsOld,oldEnt-np.mean(oldEnt[np.logical_not(np.isnan(oldEnt))]),s=2,alpha=0.5)
    ax.scatter(timePreds, normalized_entropied-np.mean(normalized_entropied),s=1)
    fig.show()

    oldEnt_cut = oldEnt[np.greater(timesPredsOld,np.min(timesPredsOld[np.logical_not(np.isnan(timesPredsOld))]))*
                        np.less(timesPredsOld,np.max(timePreds[np.logical_not(np.isnan(timePreds))]))]
    newEnt_cut = normalized_entropied[np.greater(timePreds,np.min(timesPredsOld[np.logical_not(np.isnan(timesPredsOld))]))*
                        np.less(timePreds,np.max(timePreds[np.logical_not(np.isnan(timePreds))]))]
    fig,ax = plt.subplots()
    ax.scatter(newEnt_cut,oldEnt_cut,s=1)
    ax.set_xlabel("entropy, NN trained on cond")
    ax.set_ylabel("entropy, NN trained on hab")
    fig.show()

    df = pd.DataFrame(normalized_entropied)
    df.to_csv(os.path.join(projectPath.resultsPath,"normalize_entropy.csv"))


    df_emove_ent_1 = pd.read_csv('/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-reversal/dataForKarim2/normalize_entropy.csv')
    emove_ent_1 = df_emove_ent_1.values[:,1:][:,0]
    timePredsOld = pd.read_csv('/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-reversal/dataForKarim2/uncertainty_network_test/timePreds.csv')
    timesPredsOld =  timePredsOld.values[:,1:][:,0]

    df_emove_pos_pred_1 = pd.read_csv('/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-reversal/dataForKarim2/uncertainty_network_test/networkPosPred.csv')
    emove_pos_pred_1 = df_emove_pos_pred_1.values[:,1:]
    o_1 = np.reshape(emove_pos_pred_1, [-1, params.nb_eval_dropout, params.batch_size, params.dim_output])
    _, linearPos_1 = linearizationFunction(emove_pos_pred_1.astype(np.float64))
    linearPos_1 = np.reshape(linearPos_1, o_1.shape[0:3])
    med_linearPos_1 = np.median(linearPos_1,axis=1)
    med_linearPos_1 = np.reshape(med_linearPos_1, [np.prod(med_linearPos_1.shape[0:2])])


    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied,emove_ent_1,s=1,alpha=1)
    ax.set_xlabel("entropy, 2nd NN trained on cond")
    ax.set_ylabel("entropy, 1st NN trained on cond")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(med_linearPos_1[nothabEpochMaskandSpeed],medianLinearPos[nothabEpochMaskandSpeed],s=1,alpha=0.1)
    ax.set_xlabel("linear pos, 2nd NN trained on cond")
    ax.set_ylabel("linear pos, 1st NN trained on cond")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(med_linearPos_1[np.logical_not(habEpochMask)],medianLinearPos[np.logical_not(habEpochMask)],s=1,alpha=0.1)
    ax.set_xlabel("linear pos, 2nd NN trained on cond")
    ax.set_ylabel("linear pos, 1st NN trained on cond")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(normalized_entropied[np.logical_not(habEpochMask)],emove_ent_1[np.logical_not(habEpochMask)],s=1,alpha=0.8)
    ax.set_xlabel("entropy pos, 2nd NN trained on cond")
    ax.set_ylabel("entropy pos, 1st NN trained on cond")
    fig.show()

    fig, ax = plt.subplots()
    ax.scatter(medianLinearPos[nothabEpochMaskandSpeed], linearBayesPos[nothabEpochMaskandSpeed], s=1,
               alpha=0.8)
    ax.set_xlabel("med_linearPos_1 pos, 2nd NN trained on cond")
    ax.set_ylabel("linearBayesPos")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(proba[np.logical_not(habEpochMask)],np.abs(linearBayesPos-trueLinearPos)[np.logical_not(habEpochMask)],s=1)
    fig.show()


     ## Making the link between a place cells firing and high entropy, filtering by entropy value:

    goodPlaceCells = [43]
    entropyFilter = np.less(normalized_entropied,-1)
    cm = plt.get_cmap("turbo")
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(timePreds,trueLinearPos,c="grey",alpha=0.8,label="true linear position")
    ax[0].scatter(timePreds,medianLinearPos,c="red",alpha=0.2,s=1,label="NN decoding")
    for i in goodPlaceCells:
        isSpiking =spikeMat_window_popVector[:,i]>0
        ax[0].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_popVector[isSpiking,i]/(np.max(spikeMat_window_popVector)),alpha=1)
    ax[0].set_title("NN decoding and raw firing rate")
    ax[0].set_ylabel("linear position \n cells ordered by prefered linear position")
    spikeMat_window_normalizedVector = (spikeMat_window_popVector-np.mean(spikeMat_window_popVector,axis=0))/np.std(spikeMat_window_popVector,axis=0)
    # remove the variance of the spiking
    ax[1].plot(timePreds,trueLinearPos,c="grey",alpha=0.8)
    ax[1].scatter(timePreds[entropyFilter],medianLinearPos[entropyFilter],c="red",alpha=0.6,s=1,label="NN decoding")
    for i in goodPlaceCells:
        # then scale between 0 and 1:
        spikeMat_window_normalizedVector_01 = (spikeMat_window_normalizedVector[:,i] - np.min(spikeMat_window_normalizedVector[:,i]))/np.max(spikeMat_window_normalizedVector[:,i])
        isSpiking = spikeMat_window_normalizedVector_01 > 0
        ax[1].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_normalizedVector_01[isSpiking] ,alpha=1)
    ax[1].set_ylabel("linear position \n cells ordered by prefered linear position")
    ax[1].set_title("NN decoding and normalized firing rate")
    fig.legend()
    fig.show()

    # high entropy is indicative of positions where the network make similar predictions

    fig,ax = plt.subplots()
    # ax.scatter(linearBayesPos[habEpochMask], medianLinearPos[habEpochMask], s=1,alpha=0.3)
    ax.scatter(linearBayesPos[entropyFilter*habEpochMaskandSpeed],medianLinearPos[entropyFilter*habEpochMaskandSpeed],s=1)
    fig.show()

    habSlowMask = np.logical_not(windowmask_speed[:,0].astype(dtype=np.bool))*habEpochMask
    fig,ax = plt.subplots()
    # ax.scatter(linearBayesPos[habEpochMask], medianLinearPos[habEpochMask], s=1,alpha=0.3)
    ax.scatter(linearBayesPos[entropyFilter*habSlowMask],medianLinearPos[entropyFilter*habSlowMask],s=1,alpha=0.5)
    ax.set_xlabel("bayesian linear pos prediction")
    ax.set_ylabel("NN linear pos prediction")
    ax.set_title("Entropy filtering, slow speed.")
    fig.show()

    fig,ax = plt.subplots()
    # ax.scatter(linearBayesPos[habEpochMask], medianLinearPos[habEpochMask], s=1,alpha=0.3)
    ax.scatter(np.abs(trueLinearPos-linearBayesPos)[entropyFilter*habSlowMask],np.abs(trueLinearPos-medianLinearPos)[entropyFilter*habSlowMask],s=1,alpha=0.5)
    ax.set_xlabel("bayesian linear pos prediction")
    ax.set_ylabel("NN linear pos prediction")
    ax.set_title("Entropy filtering, slow speed.")
    fig.show()

    cm = plt.get_cmap("turbo")
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(timePreds,trueLinearPos,c="grey",alpha=0.8,label="true linear position")
    ax[0].scatter(timePreds,medianLinearPos,c="red",alpha=0.2,s=1,label="NN decoding")
    for i in goodPlaceCells:
        isSpiking =spikeMat_window_popVector[:,i]>0
        ax[0].scatter(timePreds[isSpiking],np.zeros_like(timePreds[isSpiking])+linearPreferredPos[i],c="navy",s=spikeMat_window_popVector[isSpiking,i]/(np.max(spikeMat_window_popVector)),alpha=1)
    ax[0].set_title("NN decoding and raw firing rate")
    ax[0].set_ylabel("linear position \n cells ordered by prefered linear position")
    spikeMat_window_normalizedVector = (spikeMat_window_popVector-np.mean(spikeMat_window_popVector,axis=0))/np.std(spikeMat_window_popVector,axis=0)
    # remove the variance of the spiking
    ax[1].plot(timePreds,trueLinearPos,c="grey",alpha=0.8)
    ax[1].scatter(timePreds[entropyFilter],medianLinearPos[entropyFilter],c="red",alpha=0.6,s=4,label="NN decoding")
    ax[1].scatter(timePreds[entropyFilter],linearBayesPos[entropyFilter],c="green",s=3)
    ax[1].set_ylabel("linear position \n cells ordered by prefered linear position")
    ax[1].set_title("NN decoding and normalized firing rate")
    fig.legend()
    fig.show()



    cum_histLinearPosPredDensity = np.zeros_like(histlinearPosPred_density)
    for i in tqdm(range(cum_histLinearPosPredDensity.shape[0])):
        cum_histLinearPosPredDensity[i,:] = np.mean(histlinearPosPred_density[i:i+30,:],axis=0)


    #decoding proba only on habituations:
    probaDecoding = trainerBayes.full_proba_decoding(
        startTimeWindow[:trueProjPos.shape[0]][habEpochMask].astype(dtype=np.float64) / samplingRate, bayesMatrices,
        behavior_data,
        cluster_data, windowSize=30 * 0.036, masking_factor=1000, useTrain=usetrain)

    bins = bayesMatrices["Bins"]
    matPos = np.transpose([np.repeat(bins[1], len(bins[0])),np.tile(bins[0], len(bins[1]))])
    _,mappingLinearPos = linearizationFunction(matPos)
    basisChangeMat = np.zeros([matPos.shape[0],99])
    for id,linearCoord in enumerate(np.arange(0,stop=0.99,step=0.01)):
        basisChangeMat[np.equal(mappingLinearPos,linearCoord),id] = 1
    probaDecoding = np.array(probaDecoding)
    probaDecoding_cat = probaDecoding.reshape([probaDecoding.shape[0],basisChangeMat.shape[0]])
    linearProbaDecoding = np.matmul(probaDecoding_cat,basisChangeMat)

    df = pd.DataFrame(linearProbaDecoding)
    df.to_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_test",
                                                       "linearProbaDecodingBayes.csv"))

    if usetrain:
        linearProbaDecoding2= np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_fit",
                                                       "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
    else:
        linearProbaDecoding2= np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_test",
                                                       "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
    histlinearPosPred = np.stack(
        [np.histogram(linearTranspose[id, :], bins=np.arange(0, stop=1, step=0.01), density=True)[0] for id in
         range(linearTranspose.shape[0])])
    histlinearPosPred_density = histlinearPosPred / (np.sum(histlinearPosPred, axis=1)[:, None])
    def xlogx(x):
        y = np.zeros_like(x)
        y[np.greater(x, 0)] = np.log(x[np.greater(x, 0)]) * (x[np.greater(x, 0)])
        return y
    histCorrelations = np.sum(np.multiply(linearProbaDecoding,histlinearPosPred_density[habEpochMask]),axis=1)
    histCorrelations = np.mean(np.multiply((linearProbaDecoding-np.mean(linearProbaDecoding,axis=1)[:,None])/(np.std(linearProbaDecoding,axis=1)[:,None]),
                                          (histlinearPosPred_density[habEpochMask]-np.mean(histlinearPosPred_density[habEpochMask],axis=1)[:,None])/
                                          (np.std(histlinearPosPred_density[habEpochMask],axis=1)[:,None])),
                                          axis=1)

    fig,ax = plt.subplots()
    ax.scatter(histCorrelations,normalized_entropied[habEpochMask],alpha=0.2)
    fig.show()

    histCorrelations_largewindow = np.sum(np.multiply(linearProbaDecoding, cum_histLinearPosPredDensity[habEpochMask]), axis=1)
    histCorrelations_largewindow = np.mean(np.multiply((linearProbaDecoding-np.mean(linearProbaDecoding,axis=1)[:,None])/(np.std(linearProbaDecoding,axis=1)[:,None]),
                                          (cum_histLinearPosPredDensity[habEpochMask]-np.mean(cum_histLinearPosPredDensity[habEpochMask],axis=1)[:,None])/
                                          (np.std(cum_histLinearPosPredDensity[habEpochMask],axis=1)[:,None])),
                                          axis=1)
    entropy_largewindow = np.mean(-xlogx(cum_histLinearPosPredDensity),axis=1)
    # fig,ax = plt.subplots()
    # ax.scatter()
    # fig.show()

    # renormalize the large window entropy:
    trainingEntropies_largeWindow = entropy_largewindow[habEpochMaskandSpeed]
    binsLinearPos = np.arange(0,stop=np.max(linearPos)+0.1,step=0.01)
    trainingEntropies_givenPos = [trainingEntropies_largeWindow[np.greater_equal(trainingLinearPos,binsLinearPos[id])
                                                    *np.less(trainingLinearPos,binsLinearPos[id+1])] for id in range(len(binsLinearPos)-1)]
    pos_given_pos = [trainingLinearPos[np.greater_equal(trainingLinearPos,binsLinearPos[id])
                                                    *np.less(trainingLinearPos,binsLinearPos[id+1])] for id in range(len(binsLinearPos)-1)]
    trainingEntropies_givenPos_normalize = [(np.array(t)-np.mean(t))/np.std(t) for t in trainingEntropies_givenPos]

    fig,ax = plt.subplots(1,2)
    for t,b in zip(pos_given_pos,trainingEntropies_givenPos_normalize):
        ax[1].scatter(t,b,s=1)
    for t,b in zip(pos_given_pos,trainingEntropies_givenPos):
        ax[0].scatter(t,b,s=1)
    ax[0].set_xlabel("linear position")
    ax[0].set_ylabel("entropy")
    ax[1].set_xlabel("linear position")
    ax[1].set_ylabel("entropy, normalized given position")
    fig.show()

    normalized_entropied_largeWindow = np.zeros_like(ent)
    for id in range(len(binsLinearPos)-1):
        ent_given_pos = entropy_largewindow[np.greater_equal(medianLinearPos,binsLinearPos[id])
                                                    *np.less(medianLinearPos,binsLinearPos[id+1])]
        normalized_entropied_largeWindow[np.greater_equal(medianLinearPos,binsLinearPos[id])
                            *np.less(medianLinearPos,binsLinearPos[id+1])] =  (ent_given_pos - np.mean(trainingEntropies_givenPos[id]))/np.std(trainingEntropies_givenPos[id])

    windowspeed_inhab = windowmask_speed[habEpochMask][:,0].astype(np.bool)
    fig, ax = plt.subplots()
    ax.scatter(histCorrelations_largewindow[windowspeed_inhab],normalized_entropied_largeWindow[habEpochMask][windowspeed_inhab] , alpha=0.2,s=2)
    ax.set_xlabel("Correlations between predictions histograms \n of NN and bayes")
    ax.set_ylabel("normalized entropy of NN histrogram")
    ax.set_title("high speed")
    fig.show()
    windowspeed_inhab_slow = np.logical_not(windowmask_speed[habEpochMask][:,0].astype(np.bool))
    fig, ax = plt.subplots()
    ax.scatter(histCorrelations_largewindow[windowspeed_inhab_slow],normalized_entropied_largeWindow[habEpochMask][windowspeed_inhab_slow] , alpha=0.4,s=2)
    ax.set_xlabel("Correlations between predictions histograms \n of NN and bayes")
    ax.set_ylabel("normalized entropy of NN histrogram")
    ax.set_title("slow speed")
    fig.show()

    cm  = plt.get_cmap("turbo")
    fig,ax = plt.subplots()
    entFilt = normalized_entropied_largeWindow[habEpochMask][windowspeed_inhab_slow]
    ax.scatter(histCorrelations_largewindow[windowspeed_inhab_slow],np.abs(medianLinearPos-trueLinearPos)[habEpochMask][windowspeed_inhab_slow],s=1,
               c = cm((entFilt+3)/6))
    maxEnt= np.max(entFilt[np.logical_not(np.isnan(entFilt))])
    minEnt = np.min(entFilt[np.logical_not(np.isnan(entFilt))])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=minEnt, vmax=maxEnt), cmap=cm),
                 label="entropy", ax=ax)
    fig.show()

    cum_linear_pos = np.arange(0, stop=1, step=0.01)[np.argmax(cum_histLinearPosPredDensity,axis=1)]

    proba_CumNNWindow = np.max(cum_histLinearPosPredDensity,axis=1)

    cm  = plt.get_cmap("turbo")
    fig,ax = plt.subplots()
    entFilt = normalized_entropied_largeWindow[habEpochMask][windowspeed_inhab_slow]
    maxEnt= np.max(entFilt[np.logical_not(np.isnan(entFilt))])
    minEnt = np.min(entFilt[np.logical_not(np.isnan(entFilt))])
    window_lowspeed_lowent = windowspeed_inhab_slow *(np.less(proba_CumNNWindow[habEpochMask],0.01))
    ax.scatter(histCorrelations_largewindow[window_lowspeed_lowent],np.abs(trueLinearPos-cum_linear_pos)[habEpochMask][window_lowspeed_lowent],s=2)
    #              c = cm((entFilt-minEnt)/(maxEnt-minEnt))
    # ax.scatter(histCorrelations_largewindow[windowspeed_inhab_slow],np.abs(trueLinearPos-cum_linear_pos)[habEpochMask][windowspeed_inhab_slow],s=1,
    #            c = cm((entFilt-minEnt)/(maxEnt-minEnt)),alpha=0.1)
    # # fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=minEnt, vmax=maxEnt), cmap=cm),
    ax.set_title("all points with entropy less than 0, habituation, slow speed")
    ax.set_xlabel("Correlations between predictions histograms \n of NN and bayes")
    ax.set_ylabel("Linear error of the NN predictions")
    # #              label="entropy", ax=ax)
    ax.set_ylim(-0.1,1)
    fig.show()

    fig,ax = plt.subplots()
    ax.hist(proba_CumNNWindow[habEpochMask][windowspeed_inhab_slow],bins=100)
    ax.hist(proba_CumNNWindow[habEpochMask][windowspeed_inhab],bins=100,color="red",alpha=0.2)
    fig.show()

    fig,ax = plt.subplots()
    window_lowspeed_lowent = windowspeed_inhab_slow*np.less(normalized_entropied[habEpochMask],0)
    ax.scatter(histCorrelations[window_lowspeed_lowent],np.abs(trueLinearPos-medianLinearPos)[habEpochMask][window_lowspeed_lowent],s=2)
    ax.set_title("Habituation, slow speed")
    ax.set_xlabel("Correlations between predictions histograms \n of NN (36 ms) and bayes (30*36ms)")
    ax.set_ylabel("Linear error of the NN predictions")
    ax.set_ylim(-0.1,1)
    fig.show()



    ##Let us look at some example
    entPlot = normalized_entropied_largeWindow[habEpochMask][windowspeed_inhab_slow]
    corrPlot = histCorrelations_largewindow[windowspeed_inhab_slow]
    entCorrErrorFilter = np.less(entPlot,0)*np.greater(corrPlot,0.5)*np.greater(np.abs(trueLinearPos-cum_linear_pos)[habEpochMask][windowspeed_inhab_slow],0.3)
    histCumNN_filter = cum_histLinearPosPredDensity[habEpochMask][windowspeed_inhab_slow][entCorrErrorFilter]
    histNN_filter = histlinearPosPred_density[habEpochMask][windowspeed_inhab_slow][entCorrErrorFilter]
    histBayes_filter = linearProbaDecoding[windowspeed_inhab_slow][entCorrErrorFilter]
    fig,ax = plt.subplots(4,2,figsize=(10,10))
    for idp in range(8):
        ax[int(idp/2),idp%2].plot(histCumNN_filter[idp+10,:],c="red") #/np.sum(histCumNN_filter[idp+10,:])
        ax[int(idp / 2), idp % 2].plot(histNN_filter[idp + 10, :], c="violet")
        ax[int(idp/2),idp%2].plot(histBayes_filter[idp+10,:],c="blue")
        ax[int(idp/2),idp%2].set_title("correlation: "+str(np.round(corrPlot[entCorrErrorFilter][idp+10],2))+" \n ent:"+
                                           str(np.round(entPlot[entCorrErrorFilter][idp+10],2)))
    fig.tight_layout()
    fig.show()

    # Create linear regression object
    import sklearn .linear_model as linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(histCorrelations_largewindow,normalized_entropied_largeWindow[habEpochMask])
    # Make predictions using the testing set
    entropy_pred = regr.predict(histCorrelations_largewindow)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(histCorrelations_largewindow, entropy_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(histCorrelations_largewindow, entropy_pred))


    fig,ax = plt.subplots(2,1)
    ax.plot(histCorrelations)
    fig.show()


