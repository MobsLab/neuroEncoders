
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp
from importData.rawDataParser import getBehavior, get_params
from importData.epochs_management import inEpochsMask
import pykeops
from tqdm import tqdm
from matplotlib.colors import  LinearSegmentedColormap
import matplotlib.gridspec as gridspec


# TODO: this code does not work - REDO!

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#ffffff'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

class PaperFiguresSleep():
    def __init__(self, projectPath, behavior_data, trainerBayes, linearizationFunction, bayesMatrices={}, timewindows=[36], sleepName='PreSleep'):
        self.projectPath = projectPath
        self.trainerBayes = trainerBayes
        self.behavior_data = behavior_data
        self.linearizationFunction = linearizationFunction
        self.bayesMatrices = bayesMatrices
        self.timewindows = timewindows
        self.sleepName = sleepName
        _, self.samplingRate, _ = get_params(self.projectPath.xml)
        
        self.binsLinearPosHist = np.arange(0, stop=1, step=0.01) # discretisation of the linear variable to help in some plots
        self.cm = plt.get_cmap("tab20b")
        # Manage folders
        self.folderFigures = os.path.join(self.projectPath.resultsPath, 'figures')
        if not os.path.exists(self.folderFigures):
            os.mkdir(self.folderFigures)
        self.folderAligned = os.path.join(self.projectPath.dataPath, 'aligned')
        
    def load_data(self):
        # CSV files helping to align the pop vector from spike used in spike sorting
        # with predictions from spike used by the NN are also provided.
        sspikes_train = []
        swspikes_train = []
        sspikes_test = []
        swspikes_test = []

        for ws in self.timewindows:
            sspikes_train.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned, str(ws), self.sleepName, "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32))
            swspikes_train.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned, str(ws), self.sleepName, "startTimeWindow.csv")).values[:,1:],dtype=np.float32))
            sspikes_test.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned ,str(ws), self.sleepName, "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32))
            swspikes_test.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned ,str(ws), self.sleepName, "startTimeWindow.csv")).values[:,1:],dtype=np.float32))


        ### Load the NN prediction without using noise:
        lpredpos = []
        fpredpos = []
        truepos = []
        time = []
        losspred = []
        speedmask = []
        for ws in self.timewindows:
            lpredpos.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results", str(ws), "linearPred.csv")).values[:,1:],dtype=np.float32)))
            fpredpos.append(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "featurePred.csv")).values[:, 1:],dtype=np.float32))
            truepos.append(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "featureTrue.csv")).values[:, 1:],dtype=np.float32))
            time.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "timeStepsPred.csv")).values[:, 1:],dtype=np.float32)))
            losspred.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "lossPred.csv")).values[:, 1:],dtype=np.float32)))
            speedmask.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "windowmask_speed.csv")).values[:, 1:],dtype=np.float32)))

        speedmask = [ws.astype(np.bool) for ws in speedmask]
        ltruepos = [self.linearizationFunction(f)[1] for f in truepos]
        
        # Output
        self.spikes_aligned = {
            'spikes_train': sspikes_train, 'windows_train': swspikes_train, 'spikes_test': sspikes_test, 'windows_test': swspikes_test
        }
        self.resultsNN = {
            'time': time, 'speedmask': speedmask, 'linPred': lpredpos, 'fullPred': fpredpos, 'truePos': truepos, 'linTruePos': ltruepos, 'predLoss': losspred
        }
        
        return self.spikes_aligned, self.resultsNN
        
        
        
# def paperFigure_sleep(projectPath, params, linearizationFunction,behavior_data,sleepName,windowsizeMS=36,saveFolder="resultSleep"):
#     predsNN_varying_wind = []
#     timePreds_varying_wind = []
#     predsLossNN_varying_wind = []

#     proba_bayes_varying_window = []
#     linearpos_bayes_varying_window =[]
#     for id, ws in enumerate([36, 3*36]):
#         predsNN_varying_wind += [pd.read_csv(os.path.join(projectPath.resultsPath,saveFolder,str(ws),sleepName+"_allPreds.csv")).values[:, 1:]]
#         timePreds_varying_wind += [pd.read_csv(os.path.join(projectPath.resultsPath,saveFolder,str(ws),sleepName+"_timePreds.csv")).values[:, 1]]
#         predsLossNN_varying_wind += [pd.read_csv(os.path.join(projectPath.resultsPath,saveFolder,str(ws),sleepName+"_all_loss_Preds.csv")).values[:, 1:]]

#         proba_bayes_varying_window += [pd.read_csv(os.path.join(projectPath.resultsPath,saveFolder,str(ws), sleepName +
#                                                                 "_proba_bayes.csv")).values[:, 1:]]
#         linearpos_bayes_varying_window += [pd.read_csv(os.path.join(projectPath.resultsPath,saveFolder,str(ws), sleepName +
#                                                                     "_linear_bayes.csv")).values[:,1:]]

#     # binsHistlinearPos = np.arange(0, stop=1, step=0.02)
#     # histlinearPosPred_varying_wind =  [np.stack(
#     #     [np.histogram(p[id, :], bins=binsHistlinearPos, density=True)[0]
#     #      for id in range(p.shape[0])])  for p in predsNN_varying_wind]
#     # sleepPos_varying_wind  = np.median(predsNN, axis=1)
#     # proba_NN_varying_wind = [(hl / (np.sum(hl, axis=1)[:, None])) for hl in histlinearPosPred_varying_wind]
#     # maxProba_varying_wind = [np.max(proba_NN, axis=1) for proba_NN in proba_NN_varying_wind]

#     # sleepProba_givenPos = [
#     #     maxProba[np.greater_equal(sleepPos, binsHistlinearPos[id])
#     #              * np.less(sleepPos, binsHistlinearPos[id + 1])] for id in
#     #     range(len(binsHistlinearPos) - 1)]
#     # normalized_proba = np.zeros_like(maxProba)
#     # for id in range(len(binsHistlinearPos) - 1):
#     #     proba_given_pos = maxProba[np.greater_equal(sleepPos, binsHistlinearPos[id])
#     #                                * np.less(sleepPos, binsHistlinearPos[id + 1])]
#     #     normalized_proba[np.greater_equal(sleepPos, binsHistlinearPos[id])
#     #                      * np.less(sleepPos, binsHistlinearPos[id + 1])] = (proba_given_pos - np.mean(
#     #         sleepProba_givenPos[id])) / np.std(sleepProba_givenPos[id])
#     #
#     # linearpos_NN_varying_window_argmax = []
#     # for proba_NN in proba_NN_varying_wind:
#     #     linearpos_NN_varying_window_argmax += [binsHistlinearPos[np.argmax(proba_NN,axis=1)]]

#     # fig, ax = plt.subplots()
#     # ax.hist(normalized_proba, bins=100)
#     # fig.show()

#     linearpos_NN_varying_window = [linearizationFunction(predpos)[1] for predpos in predsNN_varying_wind]

#     ##Let us compare NN predictions and linear position during sleep:
#     fig, ax = plt.subplots()
#     ax.scatter(linearpos_bayes_varying_window[0][:linearpos_NN_varying_window[0].shape[0],0], linearpos_NN_varying_window[0], s=1,
#                c="grey")
#     ax.hist2d(linearpos_bayes_varying_window[0][:linearpos_NN_varying_window[0].shape[0], 0], linearpos_NN_varying_window[0], (30, 30),
#               cmap=white_viridis, alpha=0.4)
#     fig.show()
#     #
#     # idEnd = maxProba.shape[0]
#     # fig, ax = plt.subplots()
#     # ax.scatter(linearpos_bayes_varying_window[2][:maxProba.shape[0]][:idEnd][np.greater(normalized_proba[:idEnd], 4)],
#     #            linearPos_by_maxproba_NN[:idEnd][np.greater(normalized_proba[:idEnd], 4)], s=1, c="grey")
#     # ax.hist2d(linearpos_bayes_varying_window[2][:maxProba.shape[0], 0][:idEnd][np.greater(normalized_proba[:idEnd], 4)],
#     #           linearPos_by_maxproba_NN[:idEnd][np.greater(normalized_proba[:idEnd], 4)], (30, 30), cmap=white_viridis,
#     #           alpha=0.4)
#     # fig.show()

#     #TODO: adapt to loss pred
#     ## We will compare the NN with bayesian, random and shuffled bayesian
#     # the NN filtering is done using loss prediction
#     errors = []
#     errorsRandomMean = []
#     errorsRandomStd = []
#     errorsShuffleMean = []
#     errorsShuffleStd = []
#     for nproba in tqdm(np.arange(np.min(normalized_proba), np.max(normalized_proba), step=0.1)):
#         bayesPred = linearpos_bayes_varying_window[3][:maxProba.shape[0], 0][:idEnd][
#             np.greater_equal(normalized_proba[:idEnd], nproba)]
#         NNpred = linearpos_NN_varying_window_argmax[3][:idEnd][np.greater_equal(normalized_proba[:idEnd], nproba)]
#         if (NNpred.shape[0] > 0):
#             randomPred = np.random.uniform(0, 1, [NNpred.shape[0], 100])
#             errors += [np.mean(np.abs(bayesPred - NNpred))]
#             errRand = np.mean(np.abs(NNpred[:, None] - randomPred), axis=0)
#             errorsRandomMean += [np.mean(errRand)]
#             errorsRandomStd += [np.std(errRand)]
#         shuffles = []
#         for id in range(100):
#             b = np.copy(bayesPred)
#             np.random.shuffle(b)
#             shuffles += [np.mean(np.abs(NNpred - b))]
#         errorsShuffleMean += [np.mean(shuffles)]
#         errorsShuffleStd += [np.std(shuffles)]
#     errorsRandomMean = np.array(errorsRandomMean)
#     errorsRandomStd = np.array(errorsRandomStd)
#     errorsShuffleMean = np.array(errorsShuffleMean)
#     errorsShuffleStd = np.array(errorsShuffleStd)
#     fig, ax = plt.subplots()
#     ax.plot(np.arange(np.min(predsLossNN_varying_wind), np.max(predsLossNN_varying_wind), step=0.1), errors, label="bayesian")
#     ax.plot(np.arange(np.min(predsLossNN_varying_wind), np.max(predsLossNN_varying_wind), step=0.1), errorsRandomMean, color="red",
#             label="random Prediction")
#     ax.fill_between(np.arange(np.min(predsLossNN_varying_wind), np.max(predsLossNN_varying_wind), step=0.1),
#                     errorsRandomMean + errorsRandomStd, errorsRandomMean - errorsRandomStd, color="orange")
#     ax.plot(np.arange(np.min(predsLossNN_varying_wind), np.max(predsLossNN_varying_wind), step=0.1), errorsShuffleMean, color="purple",
#             label="shuffle bayesian")
#     ax.fill_between(np.arange(np.min(predsLossNN_varying_wind), np.max(predsLossNN_varying_wind), step=0.1),
#                     errorsShuffleMean + errorsShuffleStd, errorsShuffleMean - errorsShuffleStd, color="violet")
#     ax.set_ylabel("linead distance from NN predictions to Bayesian \n or random predictions")
#     ax.set_xlabel("probability filtering value")
#     ax.set_title("Pre sleep")
#     fig.legend(loc=[0.2, 0.2])
#     fig.show()
#     # plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure", "fig_NNvsBayesianSleep_36.png"))

#     # Are training position more replayed during sleep?
#     fig, ax = plt.subplots(1, 3)
#     ax[0, 0].scatter(linearpos_NN_varying_window[0], predsLossNN_varying_wind[0], s=1, c="grey")
#     ax[0, 0].hist2d(linearpos_NN_varying_window[0],  predsLossNN_varying_wind[0], (20, 20), cmap=white_viridis,
#                     alpha=0.8)
#     # ax[1, 0].scatter(linearpos_NN_varying_window[0],  normalized_proba, s=1, c="grey")
#     # ax[1, 0].hist2d(linearpos_NN_varying_window[0],  normalized_proba, (20, 20), cmap=white_viridis,
#     #                 alpha=0.8)

#     filter = np.less(predsLossNN_varying_wind[0], -2) #todo: consider fitlering value....
#     ax[0, 1].scatter(linearpos_NN_varying_window[0][filter],  predsLossNN_varying_wind[0][filter], s=1, c="grey")
#     ax[0, 1].hist2d(linearpos_NN_varying_window[0][filter],  predsLossNN_varying_wind[0][filter], (20, 20), cmap=white_viridis,
#                     alpha=0.8)
#     # filter_renormal = np.greater(normalized_proba, 2)
#     # ax[1, 1].scatter(linearpos_NN_varying_window[0][filter_renormal], normalized_proba[filter_renormal], s=1,
#     #                  c="grey")
#     # ax[1, 1].hist2d(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], (20, 20),
#     #                 cmap=white_viridis,
#     #                 alpha=0.8)
#     filter = np.logical_not(filter)
#     ax[0, 2].scatter(linearpos_NN_varying_window[0][filter], predsLossNN_varying_wind[0][filter], s=1, c="grey")
#     ax[0, 2].hist2d(linearpos_NN_varying_window[0][filter], predsLossNN_varying_wind[0][filter], (20, 20), cmap=white_viridis,
#                     alpha=0.8)
#     # filter_renormal = np.logical_not(filter_renormal)
#     # ax[1, 2].scatter(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], s=1,
#     #                  c="grey")
#     # ax[1, 2].hist2d(linearpos_NN_varying_window_argmax[0][filter_renormal], normalized_proba[filter_renormal], (20, 20),
#     #                 cmap=white_viridis,
#     #                 alpha=0.8)
#     fig.show()

#     truePosFed_train = pd.read_csv(
#         os.path.join(projectPath.resultsPath, "uncertainty_network_fit", "truePosFed.csv")).values[:, 1:]
#     truePosFed_test = pd.read_csv(
#         os.path.join(projectPath.resultsPath, "uncertainty_network_test", "truePosFed.csv")).values[:, 1:]
#     truePos_wakebeforeSleep = np.concatenate([truePosFed_train, truePosFed_test])
#     _, lineartruePos_wakebeforeSleep = linearizationFunction(truePos_wakebeforeSleep)

#     #TODO: adapt proba to loss_pred
#     # histograms of predicted positions during sleep, filtered by (proba before) now loss pred
#     fig, ax = plt.subplots(2, 3)
#     nbins = 30
#     ax[0, 0].hist(linearpos_NN_varying_window[0], bins=nbins, density=True, color="red",
#                   label="predicted position in sleep", cumulative=True)
#     ax[1, 0].hist(linearpos_NN_varying_window[0], bins=nbins, density=True, color="red", cumulative=True)
#     filter = np.greater(maxProba, 0.5)
#     ax[0, 1].hist(linearpos_NN_varying_window[0][filter], bins=nbins, density=True, color="red", cumulative=True)
#     filter_renormal = np.greater(normalized_proba, 2)
#     ax[1, 1].hist(linearpos_NN_varying_window[0][filter_renormal], bins=nbins, density=True, color="red",
#                   cumulative=True)
#     filter = np.logical_not(filter)
#     ax[0, 2].hist(linearpos_NN_varying_window[0][filter], bins=nbins, density=True, color="red", cumulative=True)
#     filter_renormal = np.logical_not(filter_renormal)
#     ax[1, 2].hist(linearpos_NN_varying_window[0][filter_renormal], bins=nbins, density=True, color="red",
#                   cumulative=True)
#     ax[0, 1].set_ylabel("histogram of position \n (probability filtering)")
#     ax[1, 1].set_ylabel("histogram of position \n (renormalized probability filtering)")
#     ax[0, 1].set_title("filtering (p>.5, renormalized p>2) \n at high probability")
#     ax[0, 2].set_title("filtering (p<=.5, renormalized p<=2) \n at low probability")
#     ax[0, 0].hist(lineartruePos_wakebeforeSleep, bins=nbins, density=True, histtype="step", color="black",
#                   label="histogram of \n wake position", cumulative=True)
#     [a.hist(lineartruePos_wakebeforeSleep, bins=nbins, density=True, histtype="step", color="black", cumulative=True) for a
#      in ax[0, 1:]]
#     [a.hist(lineartruePos_wakebeforeSleep, bins=nbins, density=True, histtype="step", color="black", cumulative=True) for a
#      in ax[1, :]]
#     [a.set_xlabel("linear position") for a in ax[0, :]]
#     [a.set_xlabel("linear position") for a in ax[1, :]]
#     fig.legend(loc=[0.05, 0.9])
#     fig.show()



#     import tables
#     import subprocess

#     if not os.path.exists(os.path.join(projectPath.folder, "nnSWR.mat")):
#         subprocess.run(["./getRipple.sh", projectPath.folder])
#     with tables.open_file(projectPath.folder + 'nnSWR.mat', "a") as f:
#         ripples = f.root.ripple[:, :].transpose()


#         # We use pykeops to find the closest prediction time to each ripple time.
#         rippleChoice = 1  # choose if we use start, beginning or end of ripple
#         predTime = pykeops.numpy.Vi(timePreds_varying_wind[0].astype(dtype=np.float64)[:, None])
#         rippleTime = pykeops.numpy.Vj(ripples[:, rippleChoice].astype(dtype=np.float64)[:, None])
#         bestTime = ((predTime - rippleTime).abs().argmin(axis=0))[:, 0]
#         # We remove ripple time tat are not inside tjhe predictio time (for exmaple in sleep)
#         bestTimeInsleep = bestTime[inEpochsMask(ripples[:, 1], [np.min(timePreds_varying_wind[0]), np.max(timePreds_varying_wind[0])])]
#         #TODO adapt the code above for the varying_window (we have a problem in the varying_window pipeline,
#         # the window are not exactly the same... can't figure out why)
#         # You just need to do a for loop and use the time_preds conrrespondign to each window size, same for evrything bellow

#         mvaMaxProba = np.array([np.mean(normalized_proba[id:id + 1]) for id in range(maxProba.shape[0])])
#         fig, ax = plt.subplots()
#         ax.hist(normalized_proba[bestTimeInsleep], bins=50, color="green", alpha=0.2, density=True)
#         ax.vlines(np.mean(normalized_proba[bestTimeInsleep]), 0, 0.25, color="green")
#         ax.hist(normalized_proba, bins=50, color="red", alpha=0.2, density=True)
#         ax.vlines(np.mean(normalized_proba), 0, 0.25, color="red")
#         fig.show()
#         # We plot an histogram of probability in sleep and out of sleep
#         #TODO: change from proba to loss_pred


#         fig, ax = plt.subplots()
#         ax.hist(sleepPos[bestTimeInsleep], bins=100)
#         fig.show()
#         fig, ax = plt.subplots()
#         ax.scatter(sleepPos, normalized_proba, s=1)
#         fig.show()
#         fig, ax = plt.subplots()
#         ax.plot(timePreds[bestTimeInsleep], normalized_proba[bestTimeInsleep], s=1)
#         fig.show()

#         # another way to look at it: let us find the temporal distance to a ripple:
#         # TODO: adapt to a varying_window
#         predTime = pykeops.numpy.Vj(timePreds_varying_wind[0].astype(dtype=np.float64)[:, None])
#         rippleTime = pykeops.numpy.Vi(ripples[:, rippleChoice].astype(dtype=np.float64)[:, None])
#         timeDist = ((predTime - rippleTime).abs().min(axis=0))[:, 0]

#         # with Proba:
#         fig, ax = plt.subplots()
#         ax.scatter(timeDist[np.less(timeDist, 1)], maxProba[np.less(timeDist, 1)], s=1, alpha=0.01)
#         ax.set_xlabel("distance to ripple")
#         ax.set_ylabel("max probability of NN predictions")
#         fig.show()
#         from SimpleBayes import butils

#         #TODO: change from proba to loss_prediction
#         timeDist_probaScatter = np.stack([maxProba[np.less(timeDist, 1)], timeDist[np.less(timeDist, 1)]])
#         res = butils.hist2D(timeDist_probaScatter.transpose(), nbins=[50, 100])
#         fig, ax = plt.subplots()
#         ax.imshow(res, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
#         ax.set_yticks(np.arange(0, 50, step=10))
#         ax.set_yticklabels(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=20)])
#         ax.set_xticks(np.arange(0, 100, step=10))
#         ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
#         ax.set_xlabel("time to ripple")
#         ax.set_ylabel("probability")
#         fig.show()
#         #TODO: change from proba to loss_prediction
#         fig, ax = plt.subplots()
#         r = ax.hist2d(timeDist[np.less(timeDist, 0.4)], normalized_proba[np.less(timeDist, 0.4)], (1000, 100),
#                       cmap=white_viridis)
#         means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
#         stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
#         ax.plot(r[1][:-1], means, c="red", label="mean normalized probability \n given time to ripple", alpha=0.3)
#         ax.fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
#         ax.set_xlabel("time to ripple (s)")
#         ax.set_ylabel("normalized probability")
#         fig.legend(loc=(0.3, 0.75))
#         plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]), np.max(r[0])), cmap=white_viridis),
#                      label="Number of windows")
#         fig.show()

#         # old Ffigure
#         # deltaT = ripples[1:, rippleChoice] - ripples[:-1, rippleChoice]
#         # maskSleep = inEpochsMask(ripples[:, rippleChoice], behavior_data["Times"]["sleepEpochs"][:2])
#         # ecart = np.ravel(deltaT[maskSleep[:-1]])
#         # fig, ax = plt.subplots(2, 1, sharex=True)
#         # r = ax[0].hist2d(timeDist[np.less(timeDist, 0.4)], normalized_proba[np.less(timeDist, 0.4)], (1000, 100),
#         #                  cmap=white_viridis)
#         # means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
#         # stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
#         # ax[0].plot(r[1][:-1], means, c="red", label="mean normalized probability \n given time to ripple", alpha=0.3)
#         # ax[0].fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
#         # ax[0].set_xlabel("time to ripple (s)")
#         # ax[0].set_ylabel("normalized probability")
#         # fig.legend(loc=(0.3, 0.85))
#         # # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]),np.max(r[0])),cmap=white_viridis),label="Number of windows",cax=ax[0])
#         # ax[1].set_title("distribution of ripple inter-time")
#         # fig.tight_layout()
#         # ax[1].hist(ecart[np.less(np.abs(ecart), 0.4)], bins=100, color="darkorange")
#         # fig.show()

#         ##same figure with the NN loss prediction
#         # TODO: adapt to varying_window
#         # deltaT = ripples[1:, rippleChoice] - ripples[:-1, rippleChoice]
#         # maskSleep = inEpochsMask(ripples[:, rippleChoice], behavior_data["Times"]["sleepEpochs"][:2])
#         # ecart = np.ravel(deltaT[maskSleep[:-1]])
#         # fig, ax = plt.subplots(2, 1, sharex=True)
#         # r = ax[0].hist2d(timeDist[np.less(timeDist, 0.4)], np.mean(predsLossNN_varying_wind[0],axis=1)[np.less(timeDist, 0.4)], (1000, 100),
#         #                  cmap=white_viridis)
#         # means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
#         # stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
#         # ax[0].plot(r[1][:-1], means, c="red", label="mean loss prediction \n given time to ripple", alpha=0.3)
#         # ax[0].fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
#         # ax[0].set_xlabel("time to ripple (s)")
#         # ax[0].set_ylabel("NN loss prediction")
#         # fig.legend(loc=(0.5, 0.65))
#         # # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]),np.max(r[0])),cmap=white_viridis),label="Number of windows",cax=ax[0])
#         # ax[1].set_title("distribution of ripple inter-time")
#         # fig.tight_layout()
#         # ax[1].hist(ecart[np.less(np.abs(ecart), 0.4)], bins=100, color="darkorange")
#         # fig.show()

#         deltaT = ripples[1:, rippleChoice] - ripples[:-1, rippleChoice]
#         maskSleep = inEpochsMask(ripples[:, rippleChoice], behavior_data["Times"]["sleepEpochs"][:2])
#         ecart = np.ravel(deltaT[maskSleep[:-1]])
#         fig, ax = plt.subplots(1, 1, figsize=(10,7.5))
#         r = ax.hist2d(timeDist[np.less(timeDist, 0.4)], np.mean(predsLossNN_varying_wind[0],axis=1)[np.less(timeDist, 0.4)], (1000, 100),
#                          cmap=white_viridis)
#         means = np.sum(r[0] * r[2][:-1][None, :], axis=1)/np.sum(r[0],axis=1)
#         stds = np.sqrt(np.sum(r[0] * np.power(r[2][:-1][None, :] - means[:, None], 2),axis=1)/np.sum(r[0],axis=1))
#         ax.plot(r[1][:-1], means, c="red", label="mean loss prediction \n given time to ripple", alpha=0.3)
#         ax.fill_between(r[1][:-1], means - stds, means + stds, color="orange", alpha=0.3)
#         ax.set_xlabel("Time to ripple (s)", fontsize='xx-large')
#         ax.tick_params(axis='x', labelsize='x-large')
#         ax.set_ylabel("NN loss prediction", fontsize='xx-large')
#         ax.tick_params(axis='y', labelsize='x-large')
#         ax.set_ylim(-8,1)
#         fig.legend(loc=(0.65, 0.13), fontsize="large")
#         # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(r[0]),np.max(r[0])),cmap=white_viridis),label="Number of windows",cax=ax[0])
#         fig.show()
#         plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure_lossPred", "lossSleepRipples.png"))
#         plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure_lossPred", "lossSleepRipples.svg"))

#         # ===========================================
#         # density normalization ....
#         # res2 = res / np.sum(res, axis=0)
#         # fig, ax = plt.subplots()
#         # ax.imshow(res2, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
#         # ax.set_yticks(np.arange(0, 50, step=10))
#         # ax.set_yticklabels(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=20)])
#         # ax.set_xticks(np.arange(0, 100, step=10))
#         # ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
#         # ax.set_xlabel("time to ripple")
#         # ax.set_ylabel("probability")
#         # ax.set_title("density is scaled so that for each time bin \n the density of NN probabilities sum to 1")
#         # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(res2), np.max(res2)), cmap=plt.get_cmap("gist_rainbow")),
#         #              label="density")
#         # fig.show()
#         # ===============================================
#         # # let us focus on beginning of sleep
#         # filter_sleep_beginning = np.less(timePreds, 10000)
#         #
#         # # with normalized probability:
#         # fig, ax = plt.subplots()
#         # ax.scatter(timeDist[np.less(timeDist, 1) * filter_sleep_beginning],
#         #            normalized_proba[np.less(timeDist, 1) * filter_sleep_beginning], s=1, alpha=0.1)
#         # ax.set_xlabel("distance to ripple")
#         # ax.set_ylabel("max probability of NN predictions")
#         # fig.show()
#         # from SimpleBayes import butils
#         #
#         # timeDist_probaScatter = np.stack([normalized_proba[np.less(timeDist, 1) * filter_sleep_beginning],
#         #                                   timeDist[np.less(timeDist, 1) * filter_sleep_beginning]])
#         # res = butils.hist2D(timeDist_probaScatter.transpose(), nbins=[50, 100])
#         # fig, ax = plt.subplots()
#         # ax.imshow(res, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
#         # ax.set_yticks(np.arange(0, 50, step=10))
#         # ax.set_yticklabels(
#         #     np.round(np.linspace(np.min(normalized_proba), np.max(normalized_proba), 50)[np.arange(0, 50, step=10)], 2))
#         # ax.set_xticks(np.arange(0, 100, step=10))
#         # ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
#         # ax.set_xlabel("time to ripple")
#         # ax.set_ylabel("normalized probability")
#         # fig.show()
#         #
#         # res2 = res / np.sum(res, axis=0)
#         # fig, ax = plt.subplots()
#         # ax.imshow(res2, origin="lower", cmap=plt.get_cmap("gist_rainbow"))
#         # ax.set_yticks(np.arange(0, 50, step=10))
#         # ax.set_yticklabels(
#         #     np.round(np.linspace(np.min(normalized_proba), np.max(normalized_proba), 50)[np.arange(0, 50, step=10)], 2))
#         # ax.set_xticks(np.arange(0, 100, step=10))
#         # ax.set_xticklabels(np.round(np.arange(0, stop=1, step=0.01)[np.arange(0, 100, step=10)], 2))
#         # ax.set_xlabel("time to ripple")
#         # ax.set_ylabel("normalized probability")
#         # ax.set_title("density is scaled so that for each time bin \n the density of NN probabilities sum to 1")
#         # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(res2), np.max(res2)), cmap=plt.get_cmap("gist_rainbow")),
#         #              label="density")
#         # fig.show()
#         #
#         # fig, ax = plt.subplots()
#         # ax.vlines(ripples[:, 1], 0, 100, color="black")
#         # ax.scatter(timePreds, timeDist, s=1, alpha=0.4)
#         # fig.show()
#         #
#         # fig, ax = plt.subplots()
#         # ax.plot(histlinearPosPred[np.where(np.greater(normalized_proba, 1))[0][100], :])
#         # fig.show()
#         # =============================
#         # fig,ax = plt.subplots()
#         # ax.hist(predConfidence[bestTime[ripples[:,rippleChoice]<np.max(timePreds)]],bins=50,density=True,alpha=0.4,label="ripple")
#         # # ax.hist(predConfidence,bins=50,density=True,alpha=0.4,label="all time")

#         #
#         # ax.hist(predConfidence[np.logical_not(isRipple)], bins=50, density=True,
#         #         alpha=0.4, label="no ripple")
#         # ax.set_xlabel("predicted confidence")
#         # ax.legend()
#         # fig.show()
#         #
#         # #Let us build a density estimate of the number of ripples
#         # N= 200
#         # mvaIsRipple    = np.mean(np.stack([isRipple[i:(isRipple.shape[0]-N+i)] for i in range(N)]),axis=0)

#         #TODO: for each ripple we have the start and top time,
#         # --> use these start and stop time instead of the ripple peak time.


#         # Figure concernign the link between ripple density and predicted loss... TODO: fix it for varying window
#         # predConfidence = maxProba
#         predConfidence = np.mean(predsLossNN_varying_wind[0],axis=1)
#         #predConfidence = predsLossNN[:,0]
#         # predConfidence = normalized_proba
#         isRipple = np.isin(range(predConfidence.shape[0]), bestTimeInsleep)
#         from scipy.ndimage import gaussian_filter1d
#         gaussRippleDensity = gaussian_filter1d(isRipple.astype(dtype=np.float),30)
#         fig,ax = plt.subplots()
#         ax.plot(timePreds_varying_wind[0],isRipple,c="black")
#         ax.plot(timePreds_varying_wind[0], gaussRippleDensity, c="red")
#         fig.show()

#         from sklearn.linear_model import LinearRegression
#         reg = LinearRegression().fit(predConfidence[:,None],gaussRippleDensity[:,None])
#         print(reg.score(predConfidence[:,None],gaussRippleDensity[:,None]))

#         filter = np.greater_equal(gaussRippleDensity,0)
#         fig,ax = plt.subplots()
#         ax.scatter(predConfidence[filter],(gaussRippleDensity[filter]),c="grey",s=1)
#         ax.set_xlabel("predicted confidence")
#         ax.set_ylabel("Ripple density - gaussian filtered")
#         # binPredConf = ax.twinx().hist(predConfidence,bins=50,color="orange",alpha=0.2,label="confidence histogram")
#         r = ax.hist2d(predConfidence[filter],(gaussRippleDensity[filter]),(500,500),cmap=white_viridis,alpha=0.4)
#         meanRippleDensity  = np.array([ np.mean((gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
#         stdRippleDensity  = np.array([ np.std((gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
#         # ax.plot(r[1][:-1],meanRippleDensity,c="red",label="mean ripple density")
#         e = np.logical_not(np.isnan(meanRippleDensity))  # * np.not_equal(meanRippleDensity,0)
#         # ax.fill_between(r[1][:-1][e], (meanRippleDensity-stdRippleDensity)[e],(meanRippleDensity+stdRippleDensity)[e], color="violet",alpha=0.5)
#         ax.plot(np.arange(np.min(predConfidence),np.max(predConfidence),step=0.1),
#                 reg.coef_[0,0]*np.arange(np.min(predConfidence),np.max(predConfidence),step=0.1)+reg.intercept_[0],c="black")
#         ax.set_title("R2= "+str(np.round(reg.score(predConfidence[:,None],gaussRippleDensity[:,None]),3)))
#         fig.legend()
#         fig.show()


#         filter = np.greater(gaussRippleDensity,0)
#         fig,ax = plt.subplots()
#         ax.scatter(predConfidence[filter],np.log(gaussRippleDensity[filter]),c="grey",s=1)
#         ax.set_xlabel("predicted confidence")
#         ax.set_ylabel("Ripple density - gaussian filtered")
#         # binPredConf = ax.twinx().hist(predConfidence,bins=50,color="orange",alpha=0.2,label="confidence histogram")
#         r = ax.hist2d(predConfidence[filter],np.log(gaussRippleDensity[filter]),(100,100),cmap=white_viridis,alpha=0.4)
#         meanRippleDensity  = np.array([ np.mean(np.log(gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
#         stdRippleDensity  = np.array([ np.std(np.log(gaussRippleDensity[filter*(predConfidence>=r[1][e]) * (predConfidence<r[1][e+1])]))   for e in range(len(r[1])-1)])
#         ax.plot(r[1][:-1],meanRippleDensity,c="red",label="mean ripple density")
#         e = np.logical_not(np.isnan(meanRippleDensity)) * np.not_equal(meanRippleDensity,0)
#         ax.fill_between(r[1][:-1][e], (meanRippleDensity-stdRippleDensity)[e],(meanRippleDensity+stdRippleDensity)[e], color="violet",alpha=0.5)
#         fig.legend()
#         fig.show()


#         # focus on REM epochs:
#         #TODO: fix it for varying window size
#         #
#         # if not os.path.exists(os.path.join(projectPath.folder, "nnREMEpochs.mat")):
#         #     subprocess.run(["./getSleepState.sh", projectPath.folder])
#         # with tables.open_file(projectPath.folder + 'nnREMEpochs.mat', "a") as f2:
#         #     startRem = f2.root.rem.remStart[:, :][0, :]
#         #     stopRem = f2.root.rem.remStop[:, :][0, :]
#         #
#         #     # we compare the predicted confidence in REM and outside of REM:
#         #     epochsRem = np.ravel(np.array([[startRem[i], stopRem[i]] for i in range(len(startRem))]))
#         #     maskREM = inEpochsMask(timePreds, epochsRem)
#         #     maskNonRem = np.logical_not(maskREM)
#         #
#         #     predConfidence = predsNN[:, 0]
#         #
#         #     fig, ax = plt.subplots()
#         #     ax.hist(predConfidence[maskREM], color="red", label="REM", alpha=0.5, density=True, bins=200)
#         #     ax.hist(predConfidence[maskNonRem], color="grey", label="Non-REM", alpha=0.5, density=True, bins=200)
#         #     fig.legend()
#         #     ax.set_xlabel("predicted confidence (trained to predict absolute linear error)")
#         #     fig.show()
#         #
#         #     cm = plt.get_cmap("turbo")
#         #     fig, ax = plt.subplots()
#         #     ax.hlines(np.zeros_like(startRem), startRem, stopRem, color="black")
#         #     # ax.scatter(stopRem,np.zeros_like(stopRem),c="red",s=1)
#         #     # ax.plot(timePreds,medianLinearPos,c="red",alpha=0.3)
#         #     ax.scatter(timePreds, predsNN, s=1, c=cm(predConfidence / np.max(predConfidence)))
#         #     fig.show()

#     #
#     # fig,ax = plt.subplots(len(outputDic.keys()),2)
#     # for id,k in enumerate(outputDic.keys()):
#     #     ax[id,0].hist(outputDic[k][1][:],bins=1000)
#     #     ax[id,0].set_title(k)
#     #     ax[id,0].set_xlabel("decoded loss")
#     #     ax[id,0].set_ylabel("histogram")
#     #     ax[id,1].hist(outputDic[k][1][:],bins=1000)
#     #     ax[id,1].set_title(k)
#     #     ax[id,1].set_xlabel("decoded loss")
#     #     ax[id,1].set_ylabel("histogram")
#     #     ax[id,1].set_yscale("log")
#     # fig.tight_layout()
#     # fig.show()
#     #
#     # fig,ax = plt.subplots(len(outputDic.keys()),2,figsize=(5,9))
#     # for id, k in enumerate(outputDic.keys()):
#     #     ax[id,0].scatter(outputDic[k][0][:,0],outputDic[k][0][:,1],alpha=0.1,s=0.1)
#     #     errorPred = outputDic[k][1][:,0]
#     #     thresh = np.quantile(errorPred,0.1)
#     #     ax[id,1].scatter(outputDic[k][0][errorPred<thresh,0],outputDic[k][0][errorPred<thresh,1],alpha=1,s=0.1)
#     #     ax[id,0].set_xlabel("predicted X")
#     #     ax[id,0].set_ylabel("predicted Y")
#     #     ax[id,1].set_xlabel("predicted X")
#     #     ax[id,1].set_ylabel("predicted Y")
#     #     ax[id,0].set_title(k+ " ;all predictions" )
#     #     ax[id,1].set_title(k + " ;filtered prediction \n by predicted loss")
#     #     ax[id,0].set_aspect(1)
#     #     ax[id,1].set_aspect(1)
#     # fig.tight_layout()
#     # fig.show()
#     #
#     # # let us plot the prediction in time...

#     # cm = plt.get_cmap("turbo")
#     # fig, ax = plt.subplots(len(outputDic.keys()), 3, figsize=(30,20))
#     # for id, k in enumerate(outputDic.keys()):
#     #     delta = 10
#     #     maxLossPred = np.max(outputDic[k][1])
#     #     minLossPred = np.min(outputDic[k][1])
#     #     ax[id,0].scatter(outputDic[k][2][1:-1:delta],outputDic[k][0][1:-1:delta,0],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
#     #     ax[id,1].scatter(outputDic[k][2][1:-1:delta],outputDic[k][0][1:-1:delta,1],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
#     #     ax[id,2].scatter(outputDic[k][2][1:-1:delta],outputDic[k][1][1:-1:delta,0],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
#     #     ax[id,1].set_xlabel("time")
#     #     ax[id,1].set_ylabel("predicted Y")
#     #     ax[id,0].set_ylabel("predicted X")
#     #     ax[id,2].set_ylabel("predicted loss")
#     # fig.show()
#     #
#     # fig, ax = plt.subplots(len(outputDic.keys()), figsize=(5, 9))
#     # for id, k in enumerate(outputDic.keys()):
#     #     delta = 10
#     #     myfilter = (outputDic[k][1] < np.quantile(outputDic[k][1], 1))[:, 0]
#     #     maxLossPred = np.max(np.clip(outputDic[k][1][myfilter,0],-10,1))
#     #     minLossPred = np.min(np.clip(outputDic[k][1][myfilter,0],-10,1))
#     #     normedLogLoss = (np.clip(outputDic[k][1][myfilter,0][1:-1:delta],-10,1)-minLossPred)/(maxLossPred-minLossPred)
#     #     ax[id].scatter(outputDic[k][0][myfilter,0][1:-1:delta],outputDic[k][0][myfilter,1][1:-1:delta],alpha=0.5,s=1,c=cm(normedLogLoss))
#     #     ax[id].set_xlabel("predicted X")
#     #     ax[id].set_ylabel("predicted Y")
#     #     ax[id].set_title(k)
#     #     fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=minLossPred,vmax=maxLossPred),cmap=cm), label="Log Loss Pred; clipped" ,ax=ax[id])
#     # fig.tight_layout()
#     # fig.show()

#     print("Ended sleep analysis")