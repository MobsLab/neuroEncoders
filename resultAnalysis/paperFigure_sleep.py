
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


def paperFigure_sleep(projectPath, params, linearizationFunction,behavior_data,sleepName):
    predsNN = pd.read_csv(os.path.join(projectPath.resultsPath, sleepName + "_allPreds.csv")).values[:, 1:]
    timePreds = pd.read_csv(os.path.join(projectPath.resultsPath, sleepName + "_timePreds.csv")).values[:, 1]
    predsLossNN = pd.read_csv(os.path.join(projectPath.resultsPath, sleepName + "_all_loss_Preds.csv")).values[:, 1:]

    proba_bayes_varying_window = []
    linearpos_bayes_varying_window =[]
    for id, window_size in enumerate([1, 3, 7, 14]):
        proba_bayes_varying_window += [pd.read_csv(os.path.join(projectPath.resultsPath, sleepName +
                                                                "_proba_bayes" + str(window_size) + ".csv")).values[:, 1:]]
        linearpos_bayes_varying_window += [pd.read_csv(os.path.join(projectPath.resultsPath, sleepName +
                                                                    "_linear_bayes" + str(window_size) + ".csv")).values[:,
                                           1:]]

    binsHistlinearPos = np.arange(0, stop=1, step=0.02)
    histlinearPosPred = np.stack(
        [np.histogram(predsNN[id, :], bins=binsHistlinearPos, density=True)[0]
         for id in range(predsNN.shape[0])])
    sleepPos = np.median(predsNN, axis=1)
    proba_NN = (histlinearPosPred / (np.sum(histlinearPosPred, axis=1)[:, None]))
    maxProba = np.max(proba_NN, axis=1)

    sleepProba_givenPos = [
        maxProba[np.greater_equal(sleepPos, binsHistlinearPos[id])
                 * np.less(sleepPos, binsHistlinearPos[id + 1])] for id in
        range(len(binsHistlinearPos) - 1)]
    normalized_proba = np.zeros_like(maxProba)
    for id in range(len(binsHistlinearPos) - 1):
        proba_given_pos = maxProba[np.greater_equal(sleepPos, binsHistlinearPos[id])
                                   * np.less(sleepPos, binsHistlinearPos[id + 1])]
        normalized_proba[np.greater_equal(sleepPos, binsHistlinearPos[id])
                         * np.less(sleepPos, binsHistlinearPos[id + 1])] = (proba_given_pos - np.mean(
            sleepProba_givenPos[id])) / np.std(sleepProba_givenPos[id])

    linearpos_NN_varying_window_argmax = []
    for i in tqdm([1, 3, 7, 14]):
        linearpos_NN_varying_window_argmax += [np.array(
            [binsHistlinearPos[np.argmax(np.mean(proba_NN[id:id + i, :], axis=0))] for id in
             range(proba_NN.shape[0])])]

    fig, ax = plt.subplots()
    ax.hist(normalized_proba, bins=100)
    fig.show()

    ##Let us compare NN predictions and linear position during sleep:
    linearPos_by_maxproba_NN = binsHistlinearPos[np.argmax(histlinearPosPred, axis=1)]
    fig, ax = plt.subplots()
    ax.scatter(linearpos_bayes_varying_window[0][:linearPos_by_maxproba_NN.shape[0]], linearPos_by_maxproba_NN, s=1,
               c="grey")
    ax.hist2d(linearpos_bayes_varying_window[0][:linearPos_by_maxproba_NN.shape[0], 0], linearPos_by_maxproba_NN, (30, 30),
              cmap=white_viridis, alpha=0.4)
    fig.show()

    idEnd = maxProba.shape[0]
    fig, ax = plt.subplots()
    ax.scatter(linearpos_bayes_varying_window[2][:maxProba.shape[0]][:idEnd][np.greater(normalized_proba[:idEnd], 4)],
               linearPos_by_maxproba_NN[:idEnd][np.greater(normalized_proba[:idEnd], 4)], s=1, c="grey")
    ax.hist2d(linearpos_bayes_varying_window[2][:maxProba.shape[0], 0][:idEnd][np.greater(normalized_proba[:idEnd], 4)],
              linearPos_by_maxproba_NN[:idEnd][np.greater(normalized_proba[:idEnd], 4)], (30, 30), cmap=white_viridis,
              alpha=0.4)
    fig.show()

    ## We will compare the NN with bayesian, random and shuffled bayesian
    errors = []
    errorsRandomMean = []
    errorsRandomStd = []
    errorsShuffleMean = []
    errorsShuffleStd = []
    for nproba in tqdm(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1)):
        bayesPred = linearpos_bayes_varying_window[3][:maxProba.shape[0], 0][:idEnd][
            np.greater_equal(normalized_proba[:idEnd], nproba)]
        NNpred = linearpos_NN_varying_window_argmax[3][:idEnd][np.greater_equal(normalized_proba[:idEnd], nproba)]
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
    ax.set_ylabel("linead distance from NN predictions to Bayesian \n or random predictions")
    ax.set_xlabel("probability filtering value")
    ax.set_title("Pre sleep")
    fig.legend(loc=[0.2, 0.2])
    fig.show()
    # plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_NNvsBayesianSleep_36.png"))

    # Are training position more replayed during sleep?
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].scatter(linearpos_NN_varying_window_argmax[0], maxProba, s=1, c="grey")
    ax[0, 0].hist2d(linearpos_NN_varying_window_argmax[0], maxProba, (20, 20), cmap=white_viridis,
                    alpha=0.8)
    ax[1, 0].scatter(linearpos_NN_varying_window_argmax[0], normalized_proba, s=1, c="grey")
    ax[1, 0].hist2d(linearpos_NN_varying_window_argmax[0], normalized_proba, (20, 20), cmap=white_viridis,
                    alpha=0.8)

    filter = np.greater(maxProba, 0.5)
    ax[0, 1].scatter(linearpos_NN_varying_window_argmax[0][filter], maxProba[filter], s=1, c="grey")
    ax[0, 1].hist2d(linearpos_NN_varying_window_argmax[0][filter], maxProba[filter], (20, 20), cmap=white_viridis,
                    alpha=0.8)
    filter_renormal = np.greater(normalized_proba, 2)
    ax[1, 1].scatter(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], s=1,
                     c="grey")
    ax[1, 1].hist2d(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], (20, 20),
                    cmap=white_viridis,
                    alpha=0.8)
    filter = np.logical_not(filter)
    ax[0, 2].scatter(linearpos_NN_varying_window_argmax[0][filter], maxProba[filter], s=1, c="grey")
    ax[0, 2].hist2d(linearpos_NN_varying_window_argmax[0][filter], maxProba[filter], (20, 20), cmap=white_viridis,
                    alpha=0.8)
    filter_renormal = np.logical_not(filter_renormal)
    ax[1, 2].scatter(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], s=1,
                     c="grey")
    ax[1, 2].hist2d(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], (20, 20),
                    cmap=white_viridis,
                    alpha=0.8)
    fig.show()

    truePosFed_train = pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "truePosFed.csv")).values[:, 1:]
    truePosFed_test = pd.read_csv(
        os.path.join(projectPath.resultsPath, "uncertainty_network_test", "truePosFed.csv")).values[:, 1:]
    truePos_wakebeforeSleep = np.concatenate([truePosFed_train, truePosFed_test])
    _, lineartruePos_wakebeforeSleep = linearizationFunction(truePos_wakebeforeSleep)

    fig, ax = plt.subplots(2, 3)
    nbins = 30
    ax[0, 0].hist(linearpos_NN_varying_window_argmax[0], bins=nbins, density=True, color="red",
                  label="predicted position in sleep", cumulative=True)
    ax[1, 0].hist(linearpos_NN_varying_window_argmax[0], bins=nbins, density=True, color="red", cumulative=True)
    filter = np.greater(maxProba, 0.5)
    ax[0, 1].hist(linearpos_NN_varying_window_argmax[0][filter], bins=nbins, density=True, color="red", cumulative=True)
    filter_renormal = np.greater(normalized_proba, 2)
    ax[1, 1].hist(linearpos_NN_varying_window_argmax[0][filter_renormal], bins=nbins, density=True, color="red",
                  cumulative=True)
    filter = np.logical_not(filter)
    ax[0, 2].hist(linearpos_NN_varying_window_argmax[0][filter], bins=nbins, density=True, color="red", cumulative=True)
    filter_renormal = np.logical_not(filter_renormal)
    ax[1, 2].hist(linearpos_NN_varying_window_argmax[0][filter_renormal], bins=nbins, density=True, color="red",
                  cumulative=True)
    ax[0, 1].set_ylabel("histogram of position \n (probability filtering)")
    ax[1, 1].set_ylabel("histogram of position \n (renormalized probability filtering)")
    ax[0, 1].set_title("filtering (p>.5, renormalized p>2) \n at high probability")
    ax[0, 2].set_title("filtering (p<=.5, renormalized p<=2) \n at low probability")
    ax[0, 0].hist(lineartruePos_wakebeforeSleep, bins=nbins, density=True, histtype="step", color="black",
                  label="histogram of \n wake position", cumulative=True)
    [a.hist(lineartruePos_wakebeforeSleep, bins=nbins, density=True, histtype="step", color="black", cumulative=True) for a
     in ax[0, 1:]]
    [a.hist(lineartruePos_wakebeforeSleep, bins=nbins, density=True, histtype="step", color="black", cumulative=True) for a
     in ax[1, :]]
    [a.set_xlabel("linear position") for a in ax[0, :]]
    [a.set_xlabel("linear position") for a in ax[1, :]]
    fig.legend(loc=[0.05, 0.9])
    fig.show()



    import tables
    import subprocess

    if not os.path.exists(os.path.join(projectPath.folder, "nnSWR.mat")):
        subprocess.run(["./getRipple.sh", projectPath.folder])
    with tables.open_file(projectPath.folder + 'nnSWR.mat', "a") as f:
        ripples = f.root.ripple[:, :].transpose()

        # cm = plt.get_cmap("turbo")
        # fig, ax = plt.subplots()
        # predConfidence = output_test[1][:,0]
        # # ax.plot(timePreds,medianLinearPos,c="red",alpha=0.3)
        # ax.scatter(timePreds, output_test[0], s=1, c=cm(predConfidence / np.max(predConfidence)))
        # ax.vlines(ripples[ripples[:, 1] <= np.max(timePreds), 1], ymin=0, ymax=1, color="grey", linewidths=1)
        # fig.show()
        #
        # # # let us find the closest timePreds to each ripple time:
        # # rippleTime =ripples.astype(dtype=np.float32)[:,1][:,None]
        # # predTime  = timePreds[:,None]
        # # timeAbsdiff = np.abs(rippleTime-np.transpose(predTime))
        # # bestTime = np.argmin(timeAbsdiff,axis=1)
        # # print(bestTime.shape)
        #
        # linearPos_by_maxproba_NN = binsHistlinearPos[np.argmax(histlinearPosPred, axis=1)]
        # maxproba_NN = np.max(histlinearPosPred / np.sum(histlinearPosPred, axis=1)[:, None], axis=1)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].scatter(linearPos_by_maxproba_NN, maxproba_NN, s=1)
        # M = np.zeros([50, 50])
        # for idi, i in enumerate(np.arange(0, stop=1, step=0.02)):
        #     for idj, j in enumerate(np.arange(0, stop=1, step=0.02)):
        #         M[idi, idj] = np.sum(np.greater_equal(linearPos_by_maxproba_NN, i) *
        #                              np.less(linearPos_by_maxproba_NN, i + 0.02) * np.greater_equal(maxproba_NN, j)
        #                              * np.less(maxproba_NN, j + 0.02))
        # ax[1].matshow(np.transpose(M), origin="lower")
        # ax[0].set_ylabel("proba")
        # ax[0].set_xlabel("decoded position")
        # fig.show()
        #
        # # training Data:
        # euclidData_train = np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_fit",
        #                                                      "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
        # _, linearPos_train = linearizationFunction(euclidData_train.astype(np.float64))
        # linearPos_train = np.reshape(linearPos_train, [-1, params.nb_eval_dropout, params.batch_size])
        # linearPos_train = np.transpose(linearPos_train, axes=[0, 2, 1]).reshape(
        #     [linearPos_train.shape[0] * params.batch_size, params.nb_eval_dropout])
        # histlinearPosPred_train = np.stack(
        #     [np.histogram(linearPos_train[id, :], bins=binsHistlinearPos, density=True)[0]
        #      for id in range(linearPos_train.shape[0])])
        # linearPos_by_maxproba_NN_train = binsHistlinearPos[np.argmax(histlinearPosPred_train, axis=1)]
        # timePreds_train = np.array(pd.read_csv(
        #     os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "timePreds.csv")).values[:, 1],
        #                            dtype=np.float32)
        # truePosFed_train = pd.read_csv(
        #     os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "truePosFed.csv")).values[:, 1:]
        # maxProbaNN_train = np.max(histlinearPosPred_train, axis=1)
        #
        # # test Data:
        # euclidData_test = np.array(pd.read_csv(os.path.join(projectPath.resultsPath, "uncertainty_network_test",
        #                                                     "networkPosPred.csv")).values[:, 1:], dtype=np.float32)
        # _, linearPos_test = linearizationFunction(euclidData_test.astype(np.float64))
        # linearPos_test = np.reshape(linearPos_test, [-1, params.nb_eval_dropout, params.batch_size])
        # linearPos_test = np.transpose(linearPos_test, axes=[0, 2, 1]).reshape(
        #     [linearPos_test.shape[0] * params.batch_size, params.nb_eval_dropout])
        # histlinearPosPred_test = np.stack(
        #     [np.histogram(linearPos_test[id, :], bins=binsHistlinearPos, density=True)[0]
        #      for id in range(linearPos_test.shape[0])])
        # maxProbaNN_test = np.max(histlinearPosPred_test, axis=1)
        # linearPos_by_maxproba_NN_test = binsHistlinearPos[np.argmax(histlinearPosPred_test, axis=1)]
        #
        # maxProba_given_visitedPos_test_hab = [
        #     np.mean(maxProbaNN_test[habEpochMask][np.greater_equal(linearTest[habEpochMask], binsHistlinearPos[id]) *
        #                                           np.less(linearTest[habEpochMask], binsHistlinearPos[id + 1])]) for id in
        #     range(len(binsHistlinearPos) - 1)]
        # maxProba_given_predictedPos_test_hab = [np.mean(
        #     maxProbaNN_test[habEpochMask][
        #         np.greater_equal(linearPos_by_maxproba_NN_test[habEpochMask], binsHistlinearPos[id]) *
        #         np.less(linearPos_by_maxproba_NN_test[habEpochMask], binsHistlinearPos[id + 1])]) for id
        #     in range(len(binsHistlinearPos) - 1)]
        #
        # maxProba_given_visitedPos_train_hab = [
        #     np.mean(maxProbaNN_train[np.greater_equal(linearTrain, binsHistlinearPos[id]) *
        #                              np.less(linearTrain, binsHistlinearPos[id + 1])]) for id in
        #     range(len(binsHistlinearPos) - 1)]
        # maxProba_given_predictedPos_train_hab = [
        #     np.mean(maxProbaNN_train[np.greater_equal(linearPos_by_maxproba_NN_train, binsHistlinearPos[id]) *
        #                              np.less(linearPos_by_maxproba_NN_train, binsHistlinearPos[id + 1])]) for id in
        #     range(len(binsHistlinearPos) - 1)]
        # maxProba_given_predictedPos_sleep = [
        #     np.mean(maxproba_NN[np.greater_equal(linearPos_by_maxproba_NN, binsHistlinearPos[id]) *
        #                         np.less(linearPos_by_maxproba_NN, binsHistlinearPos[id + 1])]) for id
        #     in range(len(binsHistlinearPos) - 1)]
        #
        # timePreds_test = np.array(pd.read_csv(
        #     os.path.join(projectPath.resultsPath, "uncertainty_network_test", "timePreds.csv")).values[:, 1],
        #                           dtype=np.float32)
        # truePosFed_test = pd.read_csv(
        #     os.path.join(projectPath.resultsPath, "uncertainty_network_test", "truePosFed.csv")).values[:, 1:]
        #
        # _, linearTest = linearizationFunction(truePosFed_test)
        # fig, ax = plt.subplots()
        # ax.hist(linearTest, bins=50)
        # fig.show()
        #
        # habEpochMask = inEpochsMask(timePreds_test, behavior_data["Times"]["testEpochs"][0:4])
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].hist(linearPos_by_maxproba_NN_test[habEpochMask], bins=100, density=True, histtype="step", color="orange")
        # ax[0, 0].hist(linearPos_by_maxproba_NN_train, bins=100, density=True, histtype="step", color="navy")
        # ax[0, 0].hist(linearPos_by_maxproba_NN, bins=100, density=True, histtype="step", color="pink")
        # ax[0, 0].set_ylabel("predicted position")
        # ax[1, 0].hist(linearTest[habEpochMask], bins=100, density=True, histtype="step", color="orange")
        # ax[1, 0].hist(linearTrain, bins=100, density=True, histtype="step", color="navy")
        # ax[1, 0].set_ylabel("visited position")
        # ax[0, 1].plot(binsHistlinearPos[:-1],
        #               maxProba_given_predictedPos_test_hab / np.nansum(maxProba_given_predictedPos_test_hab),
        #               color="orange")
        # ax[0, 1].plot(binsHistlinearPos[:-1],
        #               maxProba_given_predictedPos_train_hab / np.nansum(maxProba_given_predictedPos_train_hab),
        #               color="navy")
        # ax[0, 1].plot(binsHistlinearPos[:-1],
        #               maxProba_given_predictedPos_sleep / np.nansum(maxProba_given_predictedPos_sleep), color="pink")
        # ax[0, 1].set_ylabel("mean proba given predicted position")
        # ax[1, 1].plot(binsHistlinearPos[:-1],
        #               maxProba_given_visitedPos_test_hab / np.nansum(maxProba_given_visitedPos_test_hab), color="orange")
        # ax[1, 1].plot(binsHistlinearPos[:-1],
        #               maxProba_given_visitedPos_train_hab / np.nansum(maxProba_given_visitedPos_train_hab), color="navy")
        # ax[1, 1].set_ylabel("mean proba given visited position")
        # fig.show()
        #
        # # Let us compute the correlations matrix of the histograms!
        # normalized_hist_train = (
        #             (histlinearPosPred_train[:, :-5] - np.nanmean(histlinearPosPred_train[:, :-5], axis=0)) / np.nanstd(
        #         histlinearPosPred_train[:, :-5], axis=0))
        # corr_hist_train = np.matmul(np.transpose(normalized_hist_train), normalized_hist_train)
        #
        # normalized_hist_sleep = ((histlinearPosPred[:, :-5] - np.nanmean(histlinearPosPred[:, :-5], axis=0)) / np.nanstd(
        #     histlinearPosPred[:, :-5], axis=0))
        # corr_hist_sleep = np.matmul(np.transpose(normalized_hist_sleep), normalized_hist_sleep)
        #
        # corr_hist_train[np.diag_indices(corr_hist_train.shape[0])] = np.nan
        # corr_hist_sleep[np.diag_indices(corr_hist_sleep.shape[0])] = np.nan
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(corr_hist_train)
        # ax[0].set_title("habituation (training) \n correlations of predicted position time series")
        # ax[1].imshow(corr_hist_sleep)
        # [a.set_ylabel("linear bins") for a in ax]
        # [a.set_xlabel("linear bins") for a in ax]
        # ax[1].set_title("sleep \n correlations of predicted position time series")
        # fig.show()
        #
        # fig, ax = plt.subplots()
        # ax.scatter(timePreds, linearPos_by_maxproba_NN, s=1)
        # fig.show()
        #
        # ## Let us normalize the probability knowing the predicted position
        # # We obtain the normalization term using the training set:
        # ### The only way for us to know about
        #
        # from sklearn.decomposition import PCA
        #
        # pca = PCA(n_components=histlinearPosPred_train.shape[1] - 10)
        # pca.fit(histlinearPosPred_train)
        #
        # sing_val_train = pca.singular_values_
        # fig, ax = plt.subplots()
        # ax.plot(sing_val_train)
        # fig.show()
        #
        # eigenVect = pca.components_
        # fig, ax = plt.subplots()
        # ax.plot(np.transpose(eigenVect[0:10, :]))
        # # ax.set_aspect(99/10)
        # fig.show()
        #
        # _, linearTrain = linearizationFunction(truePosFed_train)
        # fig, ax = plt.subplots()
        # ax.hist(linearTrain, bins=50)
        # fig.show()



        rippleChoice = 1  # choose if we use start, beginning or end of ripple
        predTime = pykeops.numpy.Vi(timePreds.astype(dtype=np.float64)[:, None])
        rippleTime = pykeops.numpy.Vj(ripples[:, rippleChoice].astype(dtype=np.float64)[:, None])
        bestTime = ((predTime - rippleTime).abs().argmin(axis=0))[:, 0]
        bestTimeInsleep = bestTime[inEpochsMask(ripples[:, 1], [np.min(timePreds), np.max(timePreds)])]

        mvaMaxProba = np.array([np.mean(normalized_proba[id:id + 1]) for id in range(maxProba.shape[0])])
        fig, ax = plt.subplots()
        ax.hist(normalized_proba[bestTimeInsleep], bins=50, color="green", alpha=0.2, density=True)
        ax.vlines(np.mean(normalized_proba[bestTimeInsleep]), 0, 0.25, color="green")
        ax.hist(normalized_proba, bins=50, color="red", alpha=0.2, density=True)
        ax.vlines(np.mean(normalized_proba), 0, 0.25, color="red")
        fig.show()

        fig, ax = plt.subplots()
        ax.hist(sleepPos[bestTimeInsleep], bins=100)
        fig.show()
        fig, ax = plt.subplots()
        ax.scatter(sleepPos, normalized_proba, s=1)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(timePreds[bestTimeInsleep], normalized_proba[bestTimeInsleep], s=1)
        fig.show()

        # another way to look at it: let us find the temporal distance to a ripple:
        predTime = pykeops.numpy.Vj(timePreds.astype(dtype=np.float64)[:, None])
        rippleTime = pykeops.numpy.Vi(ripples[:, rippleChoice].astype(dtype=np.float64)[:, None])
        timeDist = ((predTime - rippleTime).abs().min(axis=0))[:, 0]

        # with Proba:
        fig, ax = plt.subplots()
        ax.scatter(timeDist[np.less(timeDist, 1)], maxProba[np.less(timeDist, 1)], s=1, alpha=0.01)
        ax.set_xlabel("distance to ripple")
        ax.set_ylabel("max probability of NN predictions")
        fig.show()
        from SimpleBayes import butils

        timeDist_probaScatter = np.stack([maxProba[np.less(timeDist, 1)], timeDist[np.less(timeDist, 1)]])
        res = butils.hist2D(timeDist_probaScatter.transpose(), nbins=[50, 100])
        fig, ax = plt.subplots()
        ax.imshow(res, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
        ax.set_yticks(np.arange(0, 50, step=10))
        ax.set_yticklabels(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=20)])
        ax.set_xticks(np.arange(0, 100, step=10))
        ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
        ax.set_xlabel("time to ripple")
        ax.set_ylabel("probability")
        fig.show()

        fig, ax = plt.subplots()
        r = ax.hist2d(timeDist[np.less(timeDist, 0.4)], normalized_proba[np.less(timeDist, 0.4)], (1000, 100),
                      cmap=white_viridis)

        means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
        stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
        ax.plot(r[1][:-1], means, c="red", label="mean normalized probability \n given time to ripple", alpha=0.3)
        ax.fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
        ax.set_xlabel("time to ripple (s)")
        ax.set_ylabel("normalized probability")
        fig.legend(loc=(0.3, 0.75))
        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]), np.max(r[0])), cmap=white_viridis),
                     label="Number of windows")
        fig.show()

        deltaT = ripples[1:, rippleChoice] - ripples[:-1, rippleChoice]
        maskSleep = inEpochsMask(ripples[:, rippleChoice], behavior_data["Times"]["sleepEpochs"][:2])
        ecart = np.ravel(deltaT[maskSleep[:-1]])
        fig, ax = plt.subplots(2, 1, sharex=True)
        r = ax[0].hist2d(timeDist[np.less(timeDist, 0.4)], normalized_proba[np.less(timeDist, 0.4)], (1000, 100),
                         cmap=white_viridis)
        means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
        stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
        ax[0].plot(r[1][:-1], means, c="red", label="mean normalized probability \n given time to ripple", alpha=0.3)
        ax[0].fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
        ax[0].set_xlabel("time to ripple (s)")
        ax[0].set_ylabel("normalized probability")
        fig.legend(loc=(0.3, 0.85))
        # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]),np.max(r[0])),cmap=white_viridis),label="Number of windows",cax=ax[0])
        ax[1].set_title("distribution of ripple inter-time")
        fig.tight_layout()
        ax[1].hist(ecart[np.less(np.abs(ecart), 0.4)], bins=100, color="darkorange")
        fig.show()

        ##same figure with the NN loss prediction

        deltaT = ripples[1:, rippleChoice] - ripples[:-1, rippleChoice]
        maskSleep = inEpochsMask(ripples[:, rippleChoice], behavior_data["Times"]["sleepEpochs"][:2])
        ecart = np.ravel(deltaT[maskSleep[:-1]])
        fig, ax = plt.subplots(2, 1, sharex=True)
        r = ax[0].hist2d(timeDist[np.less(timeDist, 0.4)], np.mean(predsLossNN,axis=1)[np.less(timeDist, 0.4)], (1000, 100),
                         cmap=white_viridis)
        means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
        stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
        ax[0].plot(r[1][:-1], means, c="red", label="mean loss prediction \n given time to ripple", alpha=0.3)
        ax[0].fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
        ax[0].set_xlabel("time to ripple (s)")
        ax[0].set_ylabel("NN loss prediction")
        fig.legend(loc=(0.5, 0.65))
        # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]),np.max(r[0])),cmap=white_viridis),label="Number of windows",cax=ax[0])
        ax[1].set_title("distribution of ripple inter-time")
        fig.tight_layout()
        ax[1].hist(ecart[np.less(np.abs(ecart), 0.4)], bins=100, color="darkorange")
        fig.show()


        res2 = res / np.sum(res, axis=0)
        fig, ax = plt.subplots()
        ax.imshow(res2, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
        ax.set_yticks(np.arange(0, 50, step=10))
        ax.set_yticklabels(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=20)])
        ax.set_xticks(np.arange(0, 100, step=10))
        ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
        ax.set_xlabel("time to ripple")
        ax.set_ylabel("probability")
        ax.set_title("density is scaled so that for each time bin \n the density of NN probabilities sum to 1")
        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(res2), np.max(res2)), cmap=plt.get_cmap("gist_rainbow")),
                     label="density")
        fig.show()

        # let us focus on beginning of sleep
        filter_sleep_beginning = np.less(timePreds, 10000)

        # with normalized probability:
        fig, ax = plt.subplots()
        ax.scatter(timeDist[np.less(timeDist, 1) * filter_sleep_beginning],
                   normalized_proba[np.less(timeDist, 1) * filter_sleep_beginning], s=1, alpha=0.1)
        ax.set_xlabel("distance to ripple")
        ax.set_ylabel("max probability of NN predictions")
        fig.show()
        from SimpleBayes import butils

        timeDist_probaScatter = np.stack([normalized_proba[np.less(timeDist, 1) * filter_sleep_beginning],
                                          timeDist[np.less(timeDist, 1) * filter_sleep_beginning]])
        res = butils.hist2D(timeDist_probaScatter.transpose(), nbins=[50, 100])
        fig, ax = plt.subplots()
        ax.imshow(res, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
        ax.set_yticks(np.arange(0, 50, step=10))
        ax.set_yticklabels(
            np.round(np.linspace(np.min(normalized_proba), np.max(normalized_proba), 50)[np.arange(0, 50, step=10)], 2))
        ax.set_xticks(np.arange(0, 100, step=10))
        ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
        ax.set_xlabel("time to ripple")
        ax.set_ylabel("normalized probability")
        fig.show()

        res2 = res / np.sum(res, axis=0)
        fig, ax = plt.subplots()
        ax.imshow(res2, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
        ax.set_yticks(np.arange(0, 50, step=10))
        ax.set_yticklabels(
            np.round(np.linspace(np.min(normalized_proba), np.max(normalized_proba), 50)[np.arange(0, 50, step=10)], 2))
        ax.set_xticks(np.arange(0, 100, step=10))
        ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
        ax.set_xlabel("time to ripple")
        ax.set_ylabel("normalized probability")
        ax.set_title("density is scaled so that for each time bin \n the density of NN probabilities sum to 1")
        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(res2), np.max(res2)), cmap=plt.get_cmap("gist_rainbow")),
                     label="density")
        fig.show()

        fig, ax = plt.subplots()
        ax.vlines(ripples[:, 1], 0, 100, color="black")
        ax.scatter(timePreds, timeDist, s=1, alpha=0.4)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(histlinearPosPred[np.where(np.greater(normalized_proba, 1))[0][100], :])
        fig.show()

        # fig,ax = plt.subplots()
        # ax.hist(predConfidence[bestTime[ripples[:,rippleChoice]<np.max(timePreds)]],bins=50,density=True,alpha=0.4,label="ripple")
        # # ax.hist(predConfidence,bins=50,density=True,alpha=0.4,label="all time")

        #
        # ax.hist(predConfidence[np.logical_not(isRipple)], bins=50, density=True,
        #         alpha=0.4, label="no ripple")
        # ax.set_xlabel("predicted confidence")
        # ax.legend()
        # fig.show()
        #
        # #Let us build a density estimate of the number of ripples
        # N= 200
        # mvaIsRipple    = np.mean(np.stack([isRipple[i:(isRipple.shape[0]-N+i)] for i in range(N)]),axis=0)

        #TODO: for each ripple we have the start and top time,
        # --> use these start and stop time instead of the ripple peak time.

        # predConfidence = maxProba
        predConfidence = np.mean(predsLossNN,axis=1)
        #predConfidence = predsLossNN[:,0]
        # predConfidence = normalized_proba
        isRipple = np.isin(range(predConfidence.shape[0]), bestTimeInsleep)

        from scipy.ndimage import gaussian_filter1d
        gaussRippleDensity = gaussian_filter1d(isRipple.astype(dtype=np.float),30)
        fig,ax = plt.subplots()
        ax.plot(timePreds,isRipple,c="black")
        ax.plot(timePreds, gaussRippleDensity, c="red")
        fig.show()

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(predConfidence[:,None],gaussRippleDensity[:,None])
        print(reg.score(predConfidence[:,None],gaussRippleDensity[:,None]))

        filter = np.greater_equal(gaussRippleDensity,0)
        fig,ax = plt.subplots()
        ax.scatter(predConfidence[filter],(gaussRippleDensity[filter]),c="grey",s=1)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("Ripple density - gaussian filtered")
        # binPredConf = ax.twinx().hist(predConfidence,bins=50,color="orange",alpha=0.2,label="confidence histogram")
        r = ax.hist2d(predConfidence[filter],(gaussRippleDensity[filter]),(500,500),cmap=white_viridis,alpha=0.4)
        meanRippleDensity  = np.array([ np.mean((gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
        stdRippleDensity  = np.array([ np.std((gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
        # ax.plot(r[1][:-1],meanRippleDensity,c="red",label="mean ripple density")
        e = np.logical_not(np.isnan(meanRippleDensity))  # * np.not_equal(meanRippleDensity,0)
        # ax.fill_between(r[1][:-1][e], (meanRippleDensity-stdRippleDensity)[e],(meanRippleDensity+stdRippleDensity)[e], color="violet",alpha=0.5)
        ax.plot(np.arange(np.min(predConfidence),np.max(predConfidence),step=0.1),
                reg.coef_[0,0]*np.arange(np.min(predConfidence),np.max(predConfidence),step=0.1)+reg.intercept_[0],c="black")
        ax.set_title("R2= "+str(np.round(reg.score(predConfidence[:,None],gaussRippleDensity[:,None]),3)))
        fig.legend()
        fig.show()


        filter = np.greater(gaussRippleDensity,0)
        fig,ax = plt.subplots()
        ax.scatter(predConfidence[filter],np.log(gaussRippleDensity[filter]),c="grey",s=1)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("Ripple density - gaussian filtered")
        # binPredConf = ax.twinx().hist(predConfidence,bins=50,color="orange",alpha=0.2,label="confidence histogram")
        r = ax.hist2d(predConfidence[filter],np.log(gaussRippleDensity[filter]),(100,100),cmap=white_viridis,alpha=0.4)
        meanRippleDensity  = np.array([ np.mean(np.log(gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
        stdRippleDensity  = np.array([ np.std(np.log(gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
        ax.plot(r[1][:-1],meanRippleDensity,c="red",label="mean ripple density")
        e = np.logical_not(np.isnan(meanRippleDensity)) * np.not_equal(meanRippleDensity,0)
        ax.fill_between(r[1][:-1][e], (meanRippleDensity-stdRippleDensity)[e],(meanRippleDensity+stdRippleDensity)[e], color="violet",alpha=0.5)
        fig.legend()
        fig.show()

        # fig,ax = plt.subplots(5,1)
        # sigma_gauss = [10,50,100,200,1000]
        # for e in range(5):
        #     gaussRippleDensity = gaussian_filter1d(isRipple.astype(dtype=np.float), sigma_gauss[e])
        #     ax[e].scatter(predConfidence,gaussRippleDensity,s=1,alpha=0.05)
        #     # ax[e].set_aspect(np.max(gaussRippleDensity)/0.015)
        # fig.show()

        if not os.path.exists(os.path.join(projectPath.folder, "nnREMEpochs.mat")):
            subprocess.run(["./getSleepState.sh", projectPath.folder])
        with tables.open_file(projectPath.folder + 'nnREMEpochs.mat', "a") as f2:
            startRem = f2.root.rem.remStart[:, :][0, :]
            stopRem = f2.root.rem.remStop[:, :][0, :]

            # we compare the predicted confidence in REM and outside of REM:
            epochsRem = np.ravel(np.array([[startRem[i], stopRem[i]] for i in range(len(startRem))]))
            maskREM = inEpochsMask(timePreds, epochsRem)
            maskNonRem = np.logical_not(maskREM)

            predConfidence = predsNN[:, 0]

            fig, ax = plt.subplots()
            ax.hist(predConfidence[maskREM], color="red", label="REM", alpha=0.5, density=True, bins=200)
            ax.hist(predConfidence[maskNonRem], color="grey", label="Non-REM", alpha=0.5, density=True, bins=200)
            fig.legend()
            ax.set_xlabel("predicted confidence (trained to predict absolute linear error)")
            fig.show()

            cm = plt.get_cmap("turbo")
            fig, ax = plt.subplots()
            ax.hlines(np.zeros_like(startRem), startRem, stopRem, color="black")
            # ax.scatter(stopRem,np.zeros_like(stopRem),c="red",s=1)
            # ax.plot(timePreds,medianLinearPos,c="red",alpha=0.3)
            ax.scatter(timePreds, predsNN, s=1, c=cm(predConfidence / np.max(predConfidence)))
            fig.show()

    #
    # fig,ax = plt.subplots(len(outputDic.keys()),2)
    # for id,k in enumerate(outputDic.keys()):
    #     ax[id,0].hist(outputDic[k][1][:],bins=1000)
    #     ax[id,0].set_title(k)
    #     ax[id,0].set_xlabel("decoded loss")
    #     ax[id,0].set_ylabel("histogram")
    #     ax[id,1].hist(outputDic[k][1][:],bins=1000)
    #     ax[id,1].set_title(k)
    #     ax[id,1].set_xlabel("decoded loss")
    #     ax[id,1].set_ylabel("histogram")
    #     ax[id,1].set_yscale("log")
    # fig.tight_layout()
    # fig.show()
    #
    # fig,ax = plt.subplots(len(outputDic.keys()),2,figsize=(5,9))
    # for id, k in enumerate(outputDic.keys()):
    #     ax[id,0].scatter(outputDic[k][0][:,0],outputDic[k][0][:,1],alpha=0.1,s=0.1)
    #     errorPred = outputDic[k][1][:,0]
    #     thresh = np.quantile(errorPred,0.1)
    #     ax[id,1].scatter(outputDic[k][0][errorPred<thresh,0],outputDic[k][0][errorPred<thresh,1],alpha=1,s=0.1)
    #     ax[id,0].set_xlabel("predicted X")
    #     ax[id,0].set_ylabel("predicted Y")
    #     ax[id,1].set_xlabel("predicted X")
    #     ax[id,1].set_ylabel("predicted Y")
    #     ax[id,0].set_title(k+ " ;all predictions" )
    #     ax[id,1].set_title(k + " ;filtered prediction \n by predicted loss")
    #     ax[id,0].set_aspect(1)
    #     ax[id,1].set_aspect(1)
    # fig.tight_layout()
    # fig.show()
    #
    # # let us plot the prediction in time...

    # cm = plt.get_cmap("turbo")
    # fig, ax = plt.subplots(len(outputDic.keys()), 3, figsize=(30,20))
    # for id, k in enumerate(outputDic.keys()):
    #     delta = 10
    #     maxLossPred = np.max(outputDic[k][1])
    #     minLossPred = np.min(outputDic[k][1])
    #     ax[id,0].scatter(outputDic[k][2][1:-1:delta],outputDic[k][0][1:-1:delta,0],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
    #     ax[id,1].scatter(outputDic[k][2][1:-1:delta],outputDic[k][0][1:-1:delta,1],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
    #     ax[id,2].scatter(outputDic[k][2][1:-1:delta],outputDic[k][1][1:-1:delta,0],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
    #     ax[id,1].set_xlabel("time")
    #     ax[id,1].set_ylabel("predicted Y")
    #     ax[id,0].set_ylabel("predicted X")
    #     ax[id,2].set_ylabel("predicted loss")
    # fig.show()
    #
    # fig, ax = plt.subplots(len(outputDic.keys()), figsize=(5, 9))
    # for id, k in enumerate(outputDic.keys()):
    #     delta = 10
    #     myfilter = (outputDic[k][1] < np.quantile(outputDic[k][1], 1))[:, 0]
    #     maxLossPred = np.max(np.clip(outputDic[k][1][myfilter,0],-10,1))
    #     minLossPred = np.min(np.clip(outputDic[k][1][myfilter,0],-10,1))
    #     normedLogLoss = (np.clip(outputDic[k][1][myfilter,0][1:-1:delta],-10,1)-minLossPred)/(maxLossPred-minLossPred)
    #     ax[id].scatter(outputDic[k][0][myfilter,0][1:-1:delta],outputDic[k][0][myfilter,1][1:-1:delta],alpha=0.5,s=1,c=cm(normedLogLoss))
    #     ax[id].set_xlabel("predicted X")
    #     ax[id].set_ylabel("predicted Y")
    #     ax[id].set_title(k)
    #     fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=minLossPred,vmax=maxLossPred),cmap=cm), label="Log Loss Pred; clipped" ,ax=ax[id])
    # fig.tight_layout()
    # fig.show()

    print("Ended sleep analysis")