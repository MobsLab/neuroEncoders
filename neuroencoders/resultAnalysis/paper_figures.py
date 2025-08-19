# Load libs
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from scipy.stats import sem

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.utils.viz_params import EC, white_viridis


class PaperFigures:
    def __init__(
        self,
        projectPath,
        behaviorData,
        trainerBayes,
        l_function,
        bayesMatrices={},
        timeWindows=[36],
        phase=None,
        sleep=False,
    ):
        self.phase = phase
        suffix = f"_{phase}" if phase is not None else ""
        self.suffix = suffix
        self.projectPath = projectPath
        self.trainerBayes = trainerBayes
        self.behaviorData = behaviorData
        self.l_function = l_function
        self.bayesMatrices = bayesMatrices
        self.timeWindows = timeWindows
        _, self.samplingRate, _ = get_params(self.projectPath.xml)

        self.binsLinearPosHist = np.arange(
            0, stop=1, step=0.01
        )  # discretisation of the linear variable to help in some plots
        self.cm = plt.get_cmap("tab20b")
        # Manage folders
        self.folderFigures = os.path.join(self.projectPath.experimentPath, "figures")
        if not os.path.exists(self.folderFigures):
            os.mkdir(self.folderFigures)
        self.folderAligned = os.path.join(self.projectPath.dataPath, "aligned")
        self.resultsNN_phase = dict()
        self.resultsBayes_phase = dict()
        if sleep:
            from resultAnalysis.paper_figures_sleep import PaperFiguresSleep

            self.sleepFigures = PaperFiguresSleep(
                projectPath,
                behaviorData,
                trainerBayes,
                l_function,
                bayesMatrices=bayesMatrices,
                timeWindows=timeWindows,
            )

    def load_data(self, suffixes=None):
        """
        Method to load the results of the neural network prediction.
        It loads the results from the csv files saved in the results folder of the experiment path.
        It prepares the data in a dictionary format for further analysis and should be called first.

        Parameters
        ----------
        suffix : str, optional
            Suffix to add to the file names, by default None. If None, it uses the class attribute self.suffix.

        Returns
        -------
        None
        """
        ### Load the NN prediction without using noise:
        if suffixes is None:
            self.suffixes = [self.suffix]
        else:
            self.suffixes = suffixes
        if not isinstance(self.suffixes, list):
            self.suffixes = [suffixes]
        for suffix in self.suffixes:
            lPredPos = []
            fPredPos = []
            truePos = []
            lTruePos = []
            time = []
            lossPred = []
            speedMask = []
            posIndex = []
            for ws in self.timeWindows:
                lPredPos.append(
                    np.squeeze(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"linearPred{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                )
                fPredPos.append(
                    np.array(
                        pd.read_csv(
                            os.path.join(
                                self.projectPath.experimentPath,
                                "results",
                                str(ws),
                                f"featurePred{suffix}.csv",
                            )
                        ).values[:, 1:],
                        dtype=np.float32,
                    )
                )
                truePos.append(
                    np.array(
                        pd.read_csv(
                            os.path.join(
                                self.projectPath.experimentPath,
                                "results",
                                str(ws),
                                f"featureTrue{suffix}.csv",
                            )
                        ).values[:, 1:],
                        dtype=np.float32,
                    )
                )
                lTruePos.append(
                    np.squeeze(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"linearTrue{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                )
                time.append(
                    np.squeeze(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"timeStepsPred{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                )
                lossPred.append(
                    np.squeeze(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"lossPred{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                )
                speedMask.append(
                    np.squeeze(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"speedMask{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                )
                posIndex.append(
                    np.squeeze(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"posIndex{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.int32,
                        )
                    )
                )

            speedMask = [ws.astype(np.bool) for ws in speedMask]

            # Output
            if suffix == self.suffix or len(self.suffixes) == 1:
                self.resultsNN = {
                    "time": time,
                    "speedMask": speedMask,
                    "linPred": lPredPos,
                    "fullPred": fPredPos,
                    "truePos": truePos,
                    "linTruePos": lTruePos,
                    "predLoss": lossPred,
                    "posIndex": posIndex,
                }
            self.resultsNN_phase[suffix] = {
                "time": time,
                "speedMask": speedMask,
                "linPred": lPredPos,
                "fullPred": fPredPos,
                "truePos": truePos,
                "linTruePos": lTruePos,
                "predLoss": lossPred,
                "posIndex": posIndex,
            }

    def load_bayes(self, suffixes=None, **kwargs):
        """
        Quickly load the bayesian decoding on the data, using the trainerBayes.
        If the bayesMatrices are not provided, it will train them using the
        trainerBayes.train_order_by_pos method.
        If the testing results are already saved, it will load them - otherwise it will perform a decoding.

        Returns
        -------
        self.resultsBayes : dict
            A dictionary containing the results of the bayesian decoding.
            It contains:
                - linPred: list of linear predicted positions for each time window
                - fullPred: list of full predicted positions for each time window
                - probaBayes: list of probabilities for each time window
                - time: list of time arrays for each time window
        """
        if not hasattr(self.trainerBayes, "linearPreferredPos"):
            self.bayesMatrices = self.trainerBayes.train_order_by_pos(
                self.behaviorData,
                l_function=self.l_function,
                bayesMatrices=self.bayesMatrices
                if (
                    (isinstance(self.bayesMatrices, dict))
                    and ("Occupation" in self.bayesMatrices.keys())
                )
                else None,
                **kwargs,
            )

        # quickly obtain bayesian decoding:
        if suffixes is None:
            self.suffixes = [self.suffix]
        else:
            self.suffixes = suffixes
        if not isinstance(self.suffixes, list):
            self.suffixes = [suffixes]

        for suffix in self.suffixes:
            lPredPosBayes = []
            probaBayes = []
            fPredBayes = []
            posLossBayes = []
            for i, ws in enumerate(self.timeWindows):
                try:
                    lPredPosBayes.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.projectPath.experimentPath,
                                        "results",
                                        str(ws),
                                        f"bayes_linearPred{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        )
                    )
                    probaBayes.append(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"bayes_proba{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                    fPredBayes.append(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"bayes_featurePred{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                    posLossBayes.append(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"bayes_posLoss{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )

                    if (
                        fPredBayes[i].shape
                        != self.resultsNN_phase[suffix]["fullPred"][i].shape
                    ):
                        raise ValueError(
                            "Bayesian and NN results do not have the same shape for "
                            + str(ws)
                            + " ms window."
                        )
                except (FileNotFoundError, ValueError) as e:
                    print(
                        f"""
                        Did not find bayesian results in folder, will test now:
                        {e}
                        """
                    )

                    timesToPredict = self.resultsNN_phase[suffix]["time"][i][
                        :, np.newaxis
                    ].astype(np.float64)
                    outputsBayes = self.trainerBayes.test_as_NN(
                        self.behaviorData,
                        self.bayesMatrices,
                        timesToPredict,
                        windowSizeMS=ws,
                        useTrain=kwargs.get("useTrain", "training" in suffix),
                        l_function=self.l_function,
                        phase=suffix.strip("_"),
                    )
                    infPos = outputsBayes["featurePred"]

                    if "linearPred" in outputsBayes:
                        lPredPosBayes.append(outputsBayes["linearPred"])
                    else:
                        _, linearBayesPos = self.l_function(infPos)
                        lPredPosBayes.append(linearBayesPos)

                    fPredBayes.append(infPos)
                    probaBayes.append(outputsBayes["proba"])
                    posLossBayes.append(outputsBayes["posLoss"])

            # Output
            if suffix == self.suffix or len(self.suffixes) == 1:
                self.resultsBayes = {
                    "linPred": lPredPosBayes,
                    "fullPred": fPredBayes,
                    "probaBayes": probaBayes,
                    "posLossBayes": posLossBayes,
                    "time": self.resultsNN_phase[suffix]["time"],
                }
            self.resultsBayes_phase[suffix] = {
                "linPred": lPredPosBayes,
                "fullPred": fPredBayes,
                "probaBayes": probaBayes,
                "posLossBayes": posLossBayes,
                "time": self.resultsNN_phase[suffix]["time"],
            }

        return self.resultsBayes

    def fig_example_XY(self, timeWindow, suffix=None, phase=None):
        idWindow = self.timeWindows.index(timeWindow)
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(19, 8))
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        for idim in range(2):
            ax[idim, 0].plot(
                self.resultsNN_phase[suffix]["time"][idWindow],
                self.resultsNN_phase[suffix]["truePos"][idWindow][:, idim],
                c="black",
                alpha=0.3,
            )
            ax[idim, 0].scatter(
                self.resultsNN_phase[suffix]["time"][idWindow],
                self.resultsNN_phase[suffix]["fullPred"][idWindow][:, idim],
                c=self.cm(12 + idWindow),
                alpha=0.9,
                label=(str(self.timeWindows[idWindow]) + " ms"),
                s=1,
            )
            ax[idim, 1].plot(
                self.resultsNN_phase[suffix]["time"][idWindow],
                self.resultsNN_phase[suffix]["truePos"][idWindow][:, idim],
                c="black",
                alpha=0.3,
            )
            ax[idim, 1].scatter(
                self.resultsNN_phase[suffix]["time"][idWindow],
                self.resultsBayes_phase[suffix]["fullPred"][idWindow][:, idim],
                c=self.cm(idWindow),
                alpha=0.9,
                label=(str(self.timeWindows[idWindow]) + " ms"),
                s=1,
            )
        ax[0, 0].set_title(
            "Neural network decoder \n " + str(self.timeWindows[idWindow]) + " window",
            fontsize="xx-large",
        )
        ax[0, 1].set_title(
            "Bayesian decoder \n " + str(self.timeWindows[idWindow]) + " window",
            fontsize="xx-large",
        )
        ax[0, 0].set_ylabel("X", fontsize="xx-large")
        ax[1, 0].set_ylabel("Y", fontsize="xx-large")
        ax[1, 0].set_xlabel("Time (s)", fontsize="xx-large")
        ax[1, 1].set_xlabel("Time (s)", fontsize="xx-large")
        # set suffix
        fig.suptitle(f"2D decoding for phase {suffix.strip('_')}", fontsize="xx-large")
        # Save figure
        fig.tight_layout()
        fig.show()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"example2D_nn_bayes_{timeWindow}ms{suffix}.png",
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"example2D_nn_bayes_{timeWindow}ms{suffix}.svg",
            )
        )

    def fig_example_linear(self, suffix=None, phase=None):
        ## Figure 1: on habituation set, speed filtered, we plot an example of bayesian and neural network decoding
        # ANN results
        # TODO: why is it speed filtered?
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        fig, ax = plt.subplots(
            len(self.timeWindows), 2, sharex=True, figsize=(15, 10), sharey=True
        )
        if len(self.timeWindows) == 1:
            ax[0].plot(
                self.resultsNN_phase[suffix]["time"][0],
                self.resultsNN_phase[suffix]["linTruePos"][0],
                c="black",
                alpha=0.3,
            )
            ax[0].scatter(
                self.resultsNN_phase[suffix]["time"][0],
                self.resultsNN_phase[suffix]["linPred"][0],
                c=self.cm(12 + 0),
                alpha=0.9,
                label=(str(self.timeWindows[0]) + " ms"),
                s=1,
            )
            ax[0].set_title(
                "Neural network decoder \n " + str(self.timeWindows[0]) + " window",
                fontsize="xx-large",
            )
            ax[0].set_ylabel("linear position", fontsize="xx-large")
            ax[0].set_yticks([0, 0.4, 0.8])
        else:
            [
                a.plot(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsNN_phase[suffix]["linTruePos"][i],
                    c="black",
                    alpha=0.3,
                )
                for i, a in enumerate(ax[:, 0])
            ]
            for i in range(len(self.timeWindows)):
                ax[i, 0].scatter(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsNN_phase[suffix]["linPred"][i],
                    c=self.cm(12 + i),
                    alpha=0.9,
                    label=(str(self.timeWindows[i]) + " ms"),
                    s=1,
                )
                if i == 0:
                    ax[i, 0].set_title(
                        "Neural network decoder \n "
                        + str(self.timeWindows[i])
                        + " window",
                        fontsize="xx-large",
                    )
                else:
                    ax[i, 0].set_title(
                        str(self.timeWindows[i]) + " window", fontsize="xx-large"
                    )

        # Bayes
        if len(self.timeWindows) == 1:
            ax[1].plot(
                self.resultsNN_phase[suffix]["time"][0],
                self.resultsNN_phase[suffix]["linTruePos"][0],
                c="black",
                alpha=0.3,
            )
            ax[1].scatter(
                self.resultsNN_phase[suffix]["time"][0],
                self.resultsBayes_phase[suffix]["linPred"][0],
                c=self.cm(0),
                alpha=0.9,
                label=(str(self.timeWindows[0]) + " ms"),
                s=1,
            )
            ax[1].set_title(
                "Bayesian decoder \n" + str(self.timeWindows[0]) + " window",
                fontsize="xx-large",
            )
            ax[1].set_xlabel("time (s)", fontsize="xx-large")
        else:
            [
                a.plot(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsNN_phase[suffix]["linTruePos"][i],
                    c="black",
                    alpha=0.3,
                )
                for i, a in enumerate(ax[:, 1])
            ]
            for i in range(len(self.timeWindows)):
                ax[i, 1].scatter(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsBayes_phase[suffix]["linPred"][i],
                    c=self.cm(i),
                    alpha=0.9,
                    label=(str(self.timeWindows[i]) + " ms"),
                    s=1,
                )
                if i == 0:
                    ax[i, 1].set_title(
                        "Bayesian decoder \n" + str(self.timeWindows[i]) + " window",
                        fontsize="xx-large",
                    )
                else:
                    ax[i, 1].set_title(
                        str(self.timeWindows[i]) + " window", fontsize="xx-large"
                    )
            ax[len(self.timeWindows) - 1, 0].set_xlabel("time (s)", fontsize="xx-large")
            ax[len(self.timeWindows) - 1, 1].set_xlabel("time (s)", fontsize="xx-large")
            [a.set_ylabel("linear position", fontsize="xx-large") for a in ax[:, 0]]
            [ax[i, 0].set_yticks([0, 0.4, 0.8]) for i in range(len(self.timeWindows))]
        # give suffix in title
        fig.suptitle(
            f"Linear position decoding for phase {suffix.strip('_')}",
            fontsize="xx-large",
        )
        # Save figure
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folderFigures, f"example_nn_bayes{suffix}.png"))
        fig.savefig(os.path.join(self.folderFigures, f"example_nn_bayes{suffix}.svg"))

    def compare_nn_bayes(
        self, timeWindow, suffix=None, phase=None, isCM=False, isShow=False
    ):
        idWindow = self.timeWindows.index(timeWindow)
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Data
        if isCM:
            nnD = self.resultsNN_phase[suffix]["fullPred"][idWindow] * EC
            bayesD = self.resultsBayes_phase[suffix]["fullPred"][idWindow] * EC
            title = "Euclidian distance (cm)"
        else:
            nnD = self.resultsNN_phase[suffix]["fullPred"][idWindow]
            bayesD = self.resultsBayes_phase[suffix]["fullPred"][idWindow]
            title = "Euclidian distance"
        distMean = np.linalg.norm(nnD - bayesD, axis=1)

        # find the best polynomial fit of euclidian error = f(time)
        poly = np.polyfit(
            self.resultsNN_phase[suffix]["time"][idWindow], distMean, deg=3
        )
        polyFit = np.polyval(poly, self.resultsNN_phase[suffix]["time"][idWindow])

        # Plot euclidian distance between fullPred of resultsNN_phase[suffix] and resultsBayes
        if isShow:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            ax.scatter(
                self.resultsNN_phase[suffix]["time"][idWindow],
                distMean,
                c="black",
                alpha=0.9,
                label=(str(self.timeWindows[idWindow]) + " ms"),
                s=1,
            )
            ax.plot(
                self.resultsNN_phase[suffix]["time"][idWindow],
                polyFit,
                c="xkcd:cherry red",
            )
            ax.set_title(
                "Euclidian distance between neural network and bayesian decoder \n"
                + str(self.timeWindows[idWindow])
                + " window, phase "
                + suffix.strip("_"),
                fontsize="xx-large",
            )
            ax.set_xlabel("time (s)", fontsize="xx-large")
            ax.set_ylabel(title, fontsize="xx-large")
            fig.show()
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    f"nn_bayes_eucledian_distance_{self.timeWindows[idWindow]}_ms{suffix}.png",
                )
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    f"nn_bayes_eucledian_distance_{self.timeWindows[idWindow]}_ms{suffix}.svg",
                )
            )

        return np.mean(distMean)

    def hist_linerrors(
        self, suffix=None, phase=None, speed="all", mask=None, use_mask=False
    ):
        ### Prepare the data
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask

        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]
        # Data
        lErrorNN = [
            np.abs(
                self.resultsNN_phase[suffix]["linTruePos"][i]
                - self.resultsNN_phase[suffix]["linPred"][i]
            )
            for i in range(len(self.timeWindows))
        ]
        lErrorBayes = [
            np.abs(
                self.resultsNN_phase[suffix]["linTruePos"][i]
                - self.resultsBayes_phase[suffix]["linPred"][i]
            )
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            lErrorNN = [lErrorNN[i][habMask[i]] for i in range(len(self.timeWindows))]
            lErrorBayes = [
                lErrorBayes[i][habMask[i]] for i in range(len(self.timeWindows))
            ]
        elif speed == "fast":
            lErrorNN = [
                lErrorNN[i][habMaskFast[i]] for i in range(len(self.timeWindows))
            ]
            lErrorBayes = [
                lErrorBayes[i][habMaskFast[i]] for i in range(len(self.timeWindows))
            ]
        elif speed == "slow":
            lErrorNN = [
                lErrorNN[i][habMaskSlow[i]] for i in range(len(self.timeWindows))
            ]
            lErrorBayes = [
                lErrorBayes[i][habMaskSlow[i]] for i in range(len(self.timeWindows))
            ]
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')

        ## Figure 2: we plot the histograms of errors
        fig, axes = plt.subplots(
            2, 2, sharex=True, figsize=(8, 6.5), constrained_layout=True
        )
        ax = axes.flatten()
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0.025, hspace=0.0001)
        for iw in range(len(self.timeWindows)):
            if iw == 0:
                ax[iw].hist(
                    lErrorNN[iw],
                    color=self.cm(iw + 12),
                    bins=self.binsLinearPosHist,
                    histtype="step",
                    density=True,
                    cumulative=True,
                )  # NN hist
                ax[iw].vlines(
                    np.mean(lErrorNN[iw]), 0, 1, color=self.cm(iw + 12), label="NN"
                )  # NN mean
                ax[iw].hist(
                    lErrorBayes[iw],
                    color=self.cm(iw),
                    bins=self.binsLinearPosHist,
                    histtype="step",
                    density=True,
                    cumulative=True,
                )  # Bayes hist
                ax[iw].vlines(
                    np.mean(lErrorBayes[iw]), 0, 1, color=self.cm(iw), label="Bayesian"
                )  # Bayes mean
            else:
                ax[iw].hist(
                    lErrorNN[iw],
                    color=self.cm(iw + 12),
                    bins=self.binsLinearPosHist,
                    histtype="step",
                    density=True,
                    cumulative=True,
                )  # NN hist
                ax[iw].vlines(
                    np.mean(lErrorNN[iw]), 0, 1, color=self.cm(iw + 12)
                )  # NN mean
                ax[iw].hist(
                    lErrorBayes[iw],
                    color=self.cm(iw),
                    bins=self.binsLinearPosHist,
                    histtype="step",
                    density=True,
                    cumulative=True,
                )  # Bayes hist
                ax[iw].vlines(
                    np.mean(lErrorBayes[iw]), 0, 1, color=self.cm(iw)
                )  # Bayes mean
            ax[iw].set_ylim(0, 1)
            ax[iw].set_title(str(self.timeWindows[iw]) + " window", fontsize="x-large")
        # Tune graph
        [a.set_aspect("auto") for a in ax]
        [a.set_xticks([0, 0.4, 0.8]) for a in ax]
        [a.set_xlim(0, 0.99) for a in ax]
        [a.set_yticks([0.25, 0.5, 0.75, 1]) for a in ax]
        fig.legend(loc=(0.85, 0.57))
        ax[0].set_ylabel("cumulative \n histogram", fontsize="x-large")
        ax[2].set_ylabel("cumulative \n histogram", fontsize="x-large")
        ax[2].set_xlabel("absolute linear error", fontsize="x-large")
        ax[3].set_xlabel("absolute linear error", fontsize="x-large")
        fig.suptitle(
            f"Cumulative histograms of linear position errors for phase {suffix.strip('_')}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"cumulativeHist_{str(speed)}{suffix}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"cumulativeHist_{str(speed)}{suffix}.svg"),
            )
        )

    def error_matrix_linerrors_by_speed(self, suffixes=None, nbins=40, normalized=True):
        if suffixes is None:
            suffixes = self.suffixes

        nrows = len(suffixes)
        ncols = 2 * len(self.timeWindows)

        fig, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(20, 8), sharex=True, sharey=True
        )

        # Handle single row/column cases
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for i, suffix in enumerate(suffixes):
            for iw, winms in enumerate(self.timeWindows):
                # All speed subplot
                H, xedges, yedges = np.histogram2d(
                    self.resultsNN_phase[suffix]["linPred"][iw].reshape(-1),
                    self.resultsNN_phase[suffix]["linTruePos"][iw].reshape(-1),
                    bins=(nbins, nbins),
                    density=True,
                )
                if normalized:
                    H = H / H.max(axis=1)  # the max value of the histogram is 1
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                ax_all = axes[i, 2 * iw]
                ax_all.set_xlim(0, 1)
                ax_all.set_ylim(0, 1)
                im = ax_all.imshow(
                    H,
                    extent=extent,
                    cmap=white_viridis,
                    interpolation="none",
                    origin="lower",
                )
                fig.colorbar(im, ax=ax_all)

                # Fast speed subplot
                H, xedges, yedges = np.histogram2d(
                    self.resultsNN_phase[suffix]["linPred"][iw][
                        self.resultsNN_phase[suffix]["speedMask"][iw]
                    ].reshape(-1),
                    self.resultsNN_phase[suffix]["linTruePos"][iw][
                        self.resultsNN_phase[suffix]["speedMask"][iw]
                    ].reshape(-1),
                    bins=(nbins, nbins),
                    density=True,
                )
                if normalized:
                    H = H / H.max(axis=1)  # the max value of the histogram is 1
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                ax_fast = axes[i, 2 * iw + 1]
                ax_fast.set_xlim(0, 1)
                ax_fast.set_ylim(0, 1)
                im = ax_fast.imshow(
                    H,
                    extent=extent,
                    cmap=white_viridis,
                    interpolation="none",
                    origin="lower",
                )
                fig.colorbar(im, ax=ax_fast)

        # Add multi-level column labels
        for iw, winms in enumerate(self.timeWindows):
            # Top level: winms labels
            x_center = (2 * iw + 0.5) / ncols
            fig.text(
                x_center,
                0.95,
                f"{winms}ms",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

            # Bottom level: all/fast labels
            x_all = (2 * iw) / ncols + 0.5 / ncols
            x_fast = (2 * iw + 1) / ncols + 0.5 / ncols
            fig.text(x_all, 0.92, "all", ha="center", va="bottom", fontsize=10)
            fig.text(x_fast, 0.92, "fast", ha="center", va="bottom", fontsize=10)

        # Add row labels (minimal suffix labels)
        for i, suffix in enumerate(suffixes):
            y_center = (nrows - i - 0.5) / nrows
            fig.text(
                0.02,
                y_center,
                f"{suffix}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                rotation=90,
            )

        fig.text(0.5, 0.04, "predicted linPos", ha="center")
        fig.text(0.04, 0.5, "true linPos", va="center", rotation="vertical")

        # Adjust layout to make room for labels
        fig.subplots_adjust(left=0.08, top=0.88, right=0.98, bottom=0.1)
        fig.suptitle(
            "Error matrix of linear position prediction by speed and time window",
            y=1,
        )

        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"errorMatrix_{suffix}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"errorMatrix_{suffix}.svg"),
            )
        )
        return fig

    def mean_linerrors(
        self,
        suffix=None,
        phase=None,
        speed="all",
        filtProp=None,
        errorType="sem",
        mask=None,
        use_mask=False,
    ):
        ### Prepare the data
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask
        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]
        if filtProp is not None:
            # Calculate filtering values
            sortedLPred = [
                np.argsort(self.resultsNN_phase[suffix]["predLoss"][iw])
                for iw in range(len(self.timeWindows))
            ]
            thresh = [
                np.squeeze(
                    self.resultsNN_phase[suffix]["predLoss"][iw][
                        sortedLPred[iw][int(len(sortedLPred[iw]) * filtProp)]
                    ]
                )
                for iw in range(len(self.timeWindows))
            ]
            filters_lpred = [
                np.ones(self.resultsNN_phase[suffix]["time"][iw].shape).astype(np.bool)
                * np.less_equal(
                    self.resultsNN_phase[suffix]["predLoss"][iw], thresh[iw]
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            filters_lpred = [
                np.ones(habMask[i].shape).astype(np.bool)
                for i in range(len(self.timeWindows))
            ]
        finalMasks = [
            habMask[i] * filters_lpred[i] for i in range(len(self.timeWindows))
        ]
        finalMasksFast = [
            habMaskFast[i] * filters_lpred[i] for i in range(len(self.timeWindows))
        ]
        finalMasksSlow = [
            habMaskSlow[i] * filters_lpred[i] for i in range(len(self.timeWindows))
        ]

        # Data
        lErrorNN = [
            np.abs(
                self.resultsNN_phase[suffix]["linTruePos"][i]
                - self.resultsNN_phase[suffix]["linPred"][i]
            )
            for i in range(len(self.timeWindows))
        ]
        lErrorBayes = [
            np.abs(
                self.resultsNN_phase[suffix]["linTruePos"][i]
                - self.resultsBayes_phase[suffix]["linPred"][i]
            )
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            lErrorNN_mean = np.array(
                [
                    np.mean(lErrorNN[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorNN_std = np.array(
                [
                    np.std(lErrorNN[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorNN_se = np.array(
                [sem(lErrorNN[i][finalMasks[i]]) for i in range(len(self.timeWindows))]
            )
            lErrorBayes_mean = np.array(
                [
                    np.mean(lErrorBayes[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_std = np.array(
                [
                    np.std(lErrorBayes[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_se = np.array(
                [
                    sem(lErrorBayes[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
        elif speed == "fast":
            lErrorNN_mean = np.array(
                [
                    np.mean(lErrorNN[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorNN_std = np.array(
                [
                    np.std(lErrorNN[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorNN_se = np.array(
                [
                    sem(lErrorNN[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_mean = np.array(
                [
                    np.mean(lErrorBayes[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_std = np.array(
                [
                    np.std(lErrorBayes[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_se = np.array(
                [
                    sem(lErrorBayes[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
        elif speed == "slow":
            lErrorNN_mean = np.array(
                [
                    np.mean(lErrorNN[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorNN_std = np.array(
                [
                    np.std(lErrorNN[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorNN_se = np.array(
                [
                    sem(lErrorNN[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_mean = np.array(
                [
                    np.mean(lErrorBayes[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_std = np.array(
                [
                    np.std(lErrorBayes[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            lErrorBayes_se = np.array(
                [
                    sem(lErrorBayes[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')

        if errorType == "std":
            lerrorNN_err = lErrorNN_std
            lerrorBayes_err = lErrorBayes_std
        elif errorType == "sem":
            lerrorNN_err = lErrorNN_se
            lerrorBayes_err = lErrorBayes_se
        else:
            raise ValueError('errorType argument could be only "std" or "sem"')

        # Fig mean error from window size - total
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.timeWindows, lErrorNN_mean, c="red", label="neural network")
        ax.fill_between(
            self.timeWindows,
            lErrorNN_mean - lerrorNN_err,
            lErrorNN_mean + lerrorNN_err,
            color="red",
            alpha=0.5,
        )
        ax.plot(self.timeWindows, lErrorBayes_mean, c="blue", label="bayesian")
        ax.fill_between(
            self.timeWindows,
            lErrorBayes_mean - lerrorBayes_err,
            lErrorBayes_mean + lerrorBayes_err,
            color="blue",
            alpha=0.5,
        )
        ax.set_xlabel("window size (ms)")
        ax.set_xticks(self.timeWindows)
        ax.set_xticklabels(self.timeWindows)
        ax.set_yticks(
            np.unique(
                np.concatenate(
                    [np.round(lErrorNN_mean, 2), np.round(lErrorBayes_mean, 2)]
                )
            )
        )

        ax.set_ylabel("mean linear error")
        fig.legend()
        fig.suptitle(
            f"Mean linear position error for phase {suffix.strip('_')}, speed: {speed}"
        )
        fig.show()
        if filtProp is None:
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanError_{str(speed)}{suffix}.png"),
                )
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanError_{str(speed)}{suffix}.svg"),
                )
            )
        else:
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanError_{str(speed)}_filt{suffix}.png"),
                )
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanError_{str(speed)}_filt{suffix}.svg"),
                )
            )

        return lErrorNN_mean, lerrorBayes_err, lErrorBayes_mean, lerrorBayes_err

    def mean_euclerrors(
        self,
        suffix=None,
        phase=None,
        speed="all",
        filtProp=None,
        errorType="sem",
        isCM=False,
        mask=None,
        use_mask=False,
    ):
        ### Prepare the data
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask

        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]
        if filtProp is not None:
            # Calculate filtering values
            sortedLPred = [
                np.argsort(self.resultsNN_phase[suffix]["predLoss"][iw])
                for iw in range(len(self.timeWindows))
            ]
            thresh = [
                np.squeeze(
                    self.resultsNN_phase[suffix]["predLoss"][iw][
                        sortedLPred[iw][int(len(sortedLPred[iw]) * filtProp)]
                    ]
                )
                for iw in range(len(self.timeWindows))
            ]
            filters_lpred = [
                np.ones(self.resultsNN_phase[suffix]["time"][iw].shape).astype(np.bool)
                * np.less_equal(
                    self.resultsNN_phase[suffix]["predLoss"][iw], thresh[iw]
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            filters_lpred = [
                np.ones(habMask[i].shape).astype(np.bool)
                for i in range(len(self.timeWindows))
            ]
        finalMasks = [
            habMask[i] * filters_lpred[i] for i in range(len(self.timeWindows))
        ]
        finalMasksFast = [
            habMaskFast[i] * filters_lpred[i] for i in range(len(self.timeWindows))
        ]
        finalMasksSlow = [
            habMaskSlow[i] * filters_lpred[i] for i in range(len(self.timeWindows))
        ]

        # Data
        nnD = {}
        bayesD = {}
        if isCM:
            nnD["pred"] = [
                self.resultsNN_phase[suffix]["fullPred"][i] * EC
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                self.resultsNN_phase[suffix]["truePos"][i] * EC
                for i in range(len(self.timeWindows))
            ]
            bayesD["pred"] = [
                self.resultsBayes_phase[suffix]["fullPred"][i] * EC
                for i in range(len(self.timeWindows))
            ]
        else:
            nnD["pred"] = [
                self.resultsNN_phase[suffix]["fullPred"][i]
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                self.resultsNN_phase[suffix]["truePos"][i]
                for i in range(len(self.timeWindows))
            ]
            bayesD["pred"] = [
                self.resultsBayes_phase[suffix]["fullPred"][i]
                for i in range(len(self.timeWindows))
            ]
        errorNN = [
            np.linalg.norm(nnD["true"][i] - nnD["pred"][i], axis=1, ord=2)
            for i in range(len(self.timeWindows))
        ]
        errorBayes = [
            np.linalg.norm(nnD["true"][i] - bayesD["pred"][i], axis=1, ord=2)
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            errorNN_mean = np.array(
                [
                    np.mean(errorNN[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorNN_std = np.array(
                [
                    np.std(errorNN[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorNN_se = np.array(
                [sem(errorNN[i][finalMasks[i]]) for i in range(len(self.timeWindows))]
            )
            errorBayes_mean = np.array(
                [
                    np.mean(errorBayes[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_std = np.array(
                [
                    np.std(errorBayes[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_se = np.array(
                [
                    sem(errorBayes[i][finalMasks[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
        elif speed == "fast":
            errorNN_mean = np.array(
                [
                    np.mean(errorNN[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorNN_std = np.array(
                [
                    np.std(errorNN[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorNN_se = np.array(
                [
                    sem(errorNN[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_mean = np.array(
                [
                    np.mean(errorBayes[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_std = np.array(
                [
                    np.std(errorBayes[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_se = np.array(
                [
                    sem(errorBayes[i][finalMasksFast[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
        elif speed == "slow":
            errorNN_mean = np.array(
                [
                    np.mean(errorNN[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorNN_std = np.array(
                [
                    np.std(errorNN[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorNN_se = np.array(
                [
                    sem(errorNN[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_mean = np.array(
                [
                    np.mean(errorBayes[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_std = np.array(
                [
                    np.std(errorBayes[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
            errorBayes_se = np.array(
                [
                    sem(errorBayes[i][finalMasksSlow[i]])
                    for i in range(len(self.timeWindows))
                ]
            )
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')

        if errorType == "std":
            errorNN_err = errorNN_std
            errorBayes_err = errorBayes_std
        elif errorType == "sem":
            errorNN_err = errorNN_se
            errorBayes_err = errorBayes_se
        else:
            raise ValueError('errorType argument could be only "std" or "sem"')

        # Fig mean error from window size - total
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.timeWindows, errorNN_mean, c="red", label="neural network")
        ax.fill_between(
            self.timeWindows,
            errorNN_mean - errorNN_err,
            errorNN_mean + errorNN_err,
            color="red",
            alpha=0.5,
        )
        ax.plot(self.timeWindows, errorBayes_mean, c="blue", label="bayesian")
        ax.fill_between(
            self.timeWindows,
            errorBayes_mean - errorBayes_err,
            errorBayes_mean + errorBayes_err,
            color="blue",
            alpha=0.5,
        )
        ax.set_xlabel("window size (ms)", fontsize="xx-large")
        ax.set_xticks(self.timeWindows)
        ax.set_xticklabels(self.timeWindows, fontsize="xx-large")
        ax.set_yticks(
            np.unique(
                np.concatenate(
                    [np.round(errorNN_mean, 2), np.round(errorBayes_mean, 2)]
                )
            )
        )
        ax.ticklabel_format(axis="y", style="plain", useOffset=True, useMathText=True)

        ax.set_ylabel("mean euclidian error", fontsize="xx-large")

        fig.legend()
        fig.suptitle(
            f"Mean euclidian position error for phase {suffix.strip('_')}, speed: {speed}",
            fontsize="xx-large",
        )
        fig.show()
        if filtProp is None:
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanEuclError_{str(speed)}{suffix}.png"),
                )
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanEuclError_{str(speed)}{suffix}.svg"),
                )
            )
        else:
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanEuclError_{str(speed)}_filt{suffix}.png"),
                )
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    (f"meanEuclError_{str(speed)}_filt{suffix}.svg"),
                )
            )

        return errorNN_mean, errorNN_err, errorBayes_mean, errorBayes_err

    def nnVSbayes(
        self, suffix=None, phase=None, speed="all", mask=None, use_mask=False
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask

        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            masks = habMask
        elif speed == "fast":
            masks = habMaskFast
        elif speed == "slow":
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')

        # Figure 4:
        cols = plt.get_cmap("terrain")
        fig, ax = plt.subplots(1, len(self.timeWindows), figsize=(10, 6))
        if len(self.timeWindows) == 1:
            ax = [ax]  # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].scatter(
                self.resultsBayes_phase[suffix]["linPred"][iw][masks[iw]],
                self.resultsNN_phase[suffix]["linPred"][iw][masks[iw]],
                s=1,
                c="grey",
            )
            ax[iw].hist2d(
                self.resultsBayes_phase[suffix]["linPred"][iw][masks[iw]],
                self.resultsNN_phase[suffix]["linPred"][iw][masks[iw]],
                (45, 45),
                cmap=white_viridis,
                alpha=0.8,
                density=True,
            )
            ax[iw].set_yticks([])
            if iw < len(self.timeWindows):
                ax[iw].set_xticks([])
        # Tune ticks
        [
            a.set_xlabel((str(self.timeWindows[iw]) + " ms"), fontsize="x-large")
            for iw, a in enumerate(ax)
        ]
        ax[len(self.timeWindows) - 1].set_xlabel(
            (
                str(self.timeWindows[len(self.timeWindows) - 1])
                + " ms \n Bayesian decoding"
            ),
            fontsize="x-large",
        )
        [a.set_ylabel("NN decoding", fontsize="x-large") for a in ax]
        [a.set_aspect("auto") for a in ax]
        [
            plt.colorbar(
                plt.cm.ScalarMappable(plt.Normalize(0, 1), cmap=white_viridis),
                ax=a,
                label="density",
            )
            for a in ax
        ]
        plt.suptitle(
            (
                f"Position decoded during \n{str(speed)} speed periods for phase {suffix.strip('_')}"
            ),
            fontsize="xx-large",
        )
        fig.show()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"NNvsBayesian_{str(speed)}{suffix}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"NNvsBayesian_{str(speed)}{suffix}.svg"),
            )
        )

    def predLoss_vs_trueLoss(self, suffix=None, phase=None, speed="all", mode="2d"):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Calculate error
        if mode == "2d":
            errors = [
                np.sqrt(
                    np.sum(
                        np.square(
                            self.resultsNN_phase[suffix]["truePos"][iw]
                            - self.resultsNN_phase[suffix]["fullPred"][iw]
                        ),
                        axis=1,
                    )
                )
                for iw in range(len(self.timeWindows))
            ]
        elif mode == "1d":
            errors = [
                np.abs(
                    self.resultsNN_phase[suffix]["linTruePos"][iw]
                    - self.resultsNN_phase[suffix]["linPred"][iw]
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            raise ValueError('mode argument could be only "2d" or "1d"')

        # Masks
        habMask = [
            inEpochsMask(
                self.resultsNN_phase[suffix]["time"][i],
                self.behaviorData["Times"]["testEpochs"],
            )
            for i in range(len(self.timeWindows))
        ]
        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            masks = habMask
        elif speed == "fast":
            masks = habMaskFast
        elif speed == "slow":
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')

        # Figure
        fig, ax = plt.subplots(1, len(self.timeWindows), figsize=(10, 4))
        if len(self.timeWindows) == 1:
            ax = [ax]  # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].scatter(
                self.resultsNN_phase[suffix]["predLoss"][iw][masks[iw]],
                errors[iw][masks[iw]],
                c="grey",
                s=1,
            )
            ax[iw].hist2d(
                self.resultsNN_phase[suffix]["predLoss"][iw][masks[iw]],
                errors[iw][masks[iw]],
                (30, 30),
                cmap=white_viridis,
                alpha=0.4,
                density=True,
            )  # ,c="red",alpha=0.4
            ax[iw].set_xlabel("Predicted loss")
            if mode == "2d":
                ax[iw].set_ylabel("True error")
            elif mode == "1d":
                ax[iw].set_ylabel("Linear error")
            ax[iw].set_title((str(self.timeWindows[iw]) + " ms"), fontsize="x-large")

            # modify xticks
            ax[iw].tick_params(axis="x", which="major", labelsize=15, rotation=45)
            ax[iw].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        fig.suptitle(
            f"Predicted loss vs true error during \n{str(speed)} speed periods for phase {suffix.strip('_')}"
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"predLoss_vs_trueLoss{str(speed)}{suffix}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"predLoss_vs_trueLoss{str(speed)}{suffix}.svg"),
            )
        )

    def fig_example_2d(
        self, suffix=None, phase=None, speed="all", mask=None, use_mask=False
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask
        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]

        if speed == "all":
            mask = habMask
        elif speed == "fast":
            mask = habMaskFast
        elif speed == "slow":
            mask = habMaskSlow
        else:
            raise ValueError('speed argument could be only "all", "fast" or "slow"')

        mazeBorder = np.array(
            [[0, 0, 1, 1, 0.63, 0.63, 0.35, 0.35, 0], [0, 1, 1, 0, 0, 0.75, 0.75, 0, 0]]
        )
        ts = [
            self.resultsNN_phase[suffix]["time"][iw][mask[iw]]
            for iw in range(len(self.timeWindows))
        ]
        # Trajectory figure
        cm = plt.get_cmap("turbo")
        fig, ax = plt.subplots(1, len(self.timeWindows), figsize=(10, 4))
        if len(self.timeWindows) == 1:
            ax = [ax]  # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].plot(
                self.resultsNN_phase[suffix]["truePos"][iw][mask[iw], 0],
                self.resultsNN_phase[suffix]["truePos"][iw][mask[iw], 1],
                color="black",
                label="true traj",
                zorder=2,
            )
            ax[iw].scatter(
                self.resultsNN_phase[suffix]["fullPred"][iw][mask[iw], 0],
                self.resultsNN_phase[suffix]["truePos"][iw][mask[iw], 1],
                c="red",
                s=3,
                label="predicted traj",
                zorder=1,
            )
            # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=np.min(ts),vmax=np.max(ts)),cmap=cm),label="prediction time (s)")
            ax[iw].set_xlabel("X")
            ax[iw].set_ylabel("Y")
            ax[iw].plot(
                mazeBorder.transpose()[:, 0], mazeBorder.transpose()[:, 1], c="black"
            )
            ax[iw].set_title((str(self.timeWindows[iw]) + " ms"), fontsize="x-large")
        fig.legend()
        fig.suptitle(
            f"Example of decoded trajectories during \n{str(speed)} speed periods for phase {suffix.strip('_')}",
        )
        fig.show()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"decoded_trajectories_{str(speed)}{suffix}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"decoded_trajectories_{str(speed)}{suffix}.svg"),
            )
        )

    def predLoss_linError(
        self, suffix=None, phase=None, speed="all", step=0.1, mask=None, use_mask=False
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask
        habMaskFast = [
            (habMask[i]) * (self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i])
            * np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i][i])
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            masks = habMask
        elif speed == "fast":
            masks = habMaskFast
        elif speed == "slow":
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "all", "fast" or "slow"')

        ## Calculate errors at each level of predLoss
        errors = [
            np.abs(
                self.resultsNN_phase[suffix]["linTruePos"][iw][masks[iw]]
                - self.resultsNN_phase[suffix]["linPred"][iw][masks[iw]]
            )
            for iw in range(len(self.timeWindows))
        ]
        predLoss_ticks = [
            np.arange(
                np.min(self.resultsNN_phase[suffix]["predLoss"][iw][masks[iw]]),
                np.max(self.resultsNN_phase[suffix]["predLoss"][iw][masks[iw]]),
                step,
            )
            for iw in range(len(self.timeWindows))
        ]
        errors_filtered = []
        for iw in range(len(self.timeWindows)):
            errors_filtered.append(
                [
                    np.mean(
                        errors[iw][
                            np.less_equal(
                                self.resultsNN_phase[suffix]["predLoss"][iw][masks[iw]],
                                pfilt,
                            )
                        ]
                    )
                    for pfilt in predLoss_ticks[iw]
                ]
            )

        ## Figure 6: decrease of the mean absolute linear error as a function of the filtering value
        labelNames = [
            (str(self.timeWindows[iw]) + " ms") for iw in range(len(self.timeWindows))
        ]
        fig, ax = plt.subplots(figsize=(10, 5.3), constrained_layout=True)
        [
            ax.plot(
                predLoss_ticks[iw],
                errors_filtered[iw],
                c=self.cm(12 + iw),
                label=labelNames[iw],
            )
            for iw in range(len(self.timeWindows))
        ]
        ax.set_xlabel(
            "Neural network \n prediction filtering value", fontsize="x-large"
        )
        ax.set_ylabel("mean absolute linear err4or", fontsize="x-large")
        ax.set_title(
            (speed + " speed" + "phase " + suffix.strip("_")), fontsize="x-large"
        )
        fig.legend(loc=(0.87, 0.17), fontsize=12)
        fig.show()

        fig.savefig(os.path.join(self.folderFigures, f"predLoss_vs_error{suffix}.png"))
        fig.savefig(os.path.join(self.folderFigures, f"predLoss_vs_error{suffix}.svg"))

    def predLoss_euclError(
        self,
        suffix=None,
        phase=None,
        speed="all",
        step=0.01,
        isCM=False,
        scaled=False,
        mask=None,
        use_mask=False,
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        REMOVED_PERCENTAGE = 1
        # Data
        nnD = {}
        if isCM:
            nnD["pred"] = [
                self.resultsNN_phase[suffix]["fullPred"][i] * EC
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                self.resultsNN_phase[suffix]["truePos"][i] * EC
                for i in range(len(self.timeWindows))
            ]
        else:
            nnD["pred"] = [
                self.resultsNN_phase[suffix]["fullPred"][i]
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                self.resultsNN_phase[suffix]["truePos"][i]
                for i in range(len(self.timeWindows))
            ]

        # Scale predicted loss between 0 and 1
        predLoss = [
            self.resultsNN_phase[suffix]["predLoss"][iw]
            for iw in range(len(self.timeWindows))
        ]
        if scaled:
            predLoss_scaled = [
                np.divide(
                    np.subtract(predLoss[iw], np.min(predLoss[iw])),
                    np.subtract(np.max(predLoss[iw]), np.min(predLoss[iw])),
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            predLoss_scaled = predLoss
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        self.resultsNN_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(
                        np.bool
                    )
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask
        habMaskFast = [
            self.resultsNN_phase[suffix]["speedMask"][i]
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            np.logical_not(self.resultsNN_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        if speed == "all":
            masks = habMask
        elif speed == "fast":
            masks = habMaskFast
        elif speed == "slow":
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')

        ## Calculate errors at each level of predLoss
        euclErrors = [
            np.linalg.norm(nnD["true"][iw] - nnD["pred"][iw], axis=1)
            for iw in range(len(self.timeWindows))
        ]

        if scaled:
            predLoss_ticks = [
                np.arange(
                    np.min(predLoss_scaled[iw]), np.max(predLoss_scaled[iw]), step
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            predLoss_ticks = [
                np.linspace(np.min(predLoss[iw]), np.max(predLoss[iw]), 1000)
                for iw in range(len(self.timeWindows))
            ]

        errors_filtered = np.zeros((len(self.timeWindows), len(predLoss_ticks[0])))
        for iw in range(len(self.timeWindows)):
            percFiltered = np.array(
                [
                    np.sum([np.less_equal(predLoss_scaled[iw], pfilt)])
                    / predLoss_scaled[iw].shape[0]
                    * 100
                    for pfilt in predLoss_ticks[iw]
                ]
            )
            # I've arbitrarly decided that 1% of the cut off data are not represetative
            maskFilterout = percFiltered < REMOVED_PERCENTAGE
            errors_filtered[iw, :] = np.array(
                [
                    np.mean(
                        euclErrors[iw][masks[iw]][
                            np.less_equal(predLoss_scaled[iw][masks[iw]], pfilt)
                        ]
                    )
                    for pfilt in predLoss_ticks[iw]
                ]
            )
            errors_filtered[iw][maskFilterout] = np.nan

        labelNames = [
            (str(self.timeWindows[iw]) + " ms") for iw in range(len(self.timeWindows))
        ]
        fig, ax = plt.subplots(figsize=(10, 5.3), constrained_layout=True)
        [
            ax.plot(
                predLoss_ticks[iw],
                errors_filtered[iw, :],
                c=self.cm(12 + iw),
                label=labelNames[iw],
            )
            for iw in range(len(self.timeWindows))
        ]
        ax.set_xlabel(
            "Neural network \n prediction filtering value", fontsize="x-large"
        )
        ax.set_ylabel("Euclidean error (cm)", fontsize="x-large")
        ax.set_title(
            (speed + " speed" + " and phase " + suffix.strip("_")), fontsize="x-large"
        )
        fig.legend(loc=(0.87, 0.17), fontsize=12)
        fig.show()

        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"predLossScaled_vs_euclError_{speed}{suffix}.png",
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"predLossScaled_vs_euclError_{speed}{suffix}.svg",
            )
        )

        return predLoss_ticks[0], errors_filtered

    def fig_example_linear_filtered(self, suffix=None, phase=None, fprop=0.3):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Calculate filtering values
        sortedLPred = [
            np.argsort(self.resultsNN_phase[suffix]["predLoss"][iw])
            for iw in range(len(self.timeWindows))
        ]
        sortedprobaBayes = [
            np.argsort(self.resultsBayes_phase[suffix]["probaBayes"][iw])
            for iw in range(len(self.timeWindows))
        ]

        thresh = [
            np.squeeze(
                self.resultsNN_phase[suffix]["predLoss"][iw][
                    sortedLPred[iw][int(len(sortedLPred[iw]) * fprop)]
                ]
            )
            for iw in range(len(self.timeWindows))
        ]
        threshBayes = [
            np.squeeze(
                self.resultsBayes_phase[suffix]["probaBayes"][iw][
                    sortedprobaBayes[iw][int(len(sortedprobaBayes[iw]) * fprop)]
                ]
            )
            for iw in range(len(self.timeWindows))
        ]

        filters_lpred = [
            np.ones(self.resultsNN_phase[suffix]["time"][iw].shape).astype(np.bool)
            * np.less_equal(self.resultsNN_phase[suffix]["predLoss"][iw], thresh[iw])
            for iw in range(len(self.timeWindows))
        ]
        filters_bayes = [
            np.ones(self.resultsBayes_phase[suffix]["time"][iw].shape).astype(np.bool)
            * np.greater_equal(
                self.resultsBayes_phase[suffix]["probaBayes"][iw], threshBayes[iw]
            )
            for iw in range(len(self.timeWindows))
        ]

        fig, ax = plt.subplots(
            len(self.timeWindows), 2, sharex=True, figsize=(15, 10), sharey=True
        )
        # All points
        if len(self.timeWindows) == 1:
            ax[0].plot(
                self.resultsNN_phase[suffix]["time"][0],
                self.resultsNN_phase[suffix]["linTruePos"][0],
                c="black",
                alpha=0.3,
            )
            ax[0].scatter(
                self.resultsNN_phase[suffix]["time"][0][filters_lpred[0]],
                self.resultsNN_phase[suffix]["linPred"][0][filters_lpred[0]],
                c=self.cm(12 + 0),
                alpha=0.9,
                label=(str(self.timeWindows[0]) + " ms"),
                s=1,
            )
            ax[0].set_title(
                "Neural network decoder \n "
                + str(self.timeWindows[0])
                + " window for phase "
                + suffix.strip("_"),
                fontsize="xx-large",
            )
            ax[0].set_ylabel("linear position", fontsize="xx-large")
            ax[0].set_yticks([0, 0.4, 0.8])
        else:
            [
                a.plot(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsNN_phase[suffix]["linTruePos"][i],
                    c="black",
                    alpha=0.3,
                )
                for i, a in enumerate(ax[:, 0])
            ]
            for i in range(len(self.timeWindows)):
                ax[i, 0].scatter(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsNN_phase[suffix]["linPred"][i],
                    c=self.cm(12 + i),
                    alpha=0.9,
                    label=(str(self.timeWindows[i]) + " ms"),
                    s=1,
                )
            if i == 0:
                ax[i, 0].set_title(
                    "Neural network decoder \n "
                    + str(self.timeWindows[i])
                    + " window for phase "
                    + suffix.strip("_"),
                    fontsize="xx-large",
                )
            else:
                ax[i, 0].set_title(
                    str(self.timeWindows[i]) + " window", fontsize="xx-large"
                )

        # Filtered data
        if len(self.timeWindows) == 1:
            ax[1].plot(
                self.resultsNN_phase[suffix]["time"][0],
                self.resultsNN_phase[suffix]["linTruePos"][0],
                c="black",
                alpha=0.3,
            )
            ax[1].scatter(
                self.resultsNN_phase[suffix]["time"][0][filters_lpred[0]],
                self.resultsNN_phase[suffix]["linPred"][0][filters_lpred[0]],
                c=self.cm(12 + 0),
                alpha=0.9,
                label=(str(self.timeWindows[0]) + " ms"),
                s=1,
            )
            ax[1].set_title(
                "Best "
                + str(fprop * 100)
                + "% of predicitons \n"
                + str(self.timeWindows[0])
                + " ms window for phase "
                + suffix.strip("_"),
                fontsize="xx-large",
            )
            ax[1].set_xlabel("time (s)", fontsize="xx-large")
        else:
            [
                a.plot(
                    self.resultsNN_phase[suffix]["time"][i],
                    self.resultsNN_phase[suffix]["linTruePos"][i],
                    c="black",
                    alpha=0.3,
                )
                for i, a in enumerate(ax[:, 1])
            ]
            for i in range(len(self.timeWindows)):
                ax[i, 1].scatter(
                    self.resultsNN_phase[suffix]["time"][i][filters_lpred[i]],
                    self.resultsNN_phase[suffix]["linPred"][i][filters_lpred[i]],
                    c=self.cm(12 + i),
                    alpha=0.9,
                    label=(str(self.timeWindows[i]) + " ms"),
                    s=1,
                )
                if i == 0:
                    ax[i, 1].set_title(
                        "Best "
                        + str(fprop * 100)
                        + "% of predicitons \n"
                        + str(self.timeWindows[0])
                        + " ms window for phase "
                        + suffix.strip("_"),
                        fontsize="xx-large",
                    )
                else:
                    ax[i, 1].set_title(
                        str(self.timeWindows[i]) + " window", fontsize="xx-large"
                    )
            ax[len(self.timeWindows) - 1, 0].set_xlabel("time (s)", fontsize="xx-large")
            ax[len(self.timeWindows) - 1, 1].set_xlabel("time (s)", fontsize="xx-large")
            [a.set_ylabel("linear position", fontsize="xx-large") for a in ax[:, 0]]
            [ax[i, 0].set_yticks([0, 0.4, 0.8]) for i in range(len(self.timeWindows))]
        # Save figure
        fig.tight_layout()
        fig.show()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"example_nn_bayes_filtered_{str(fprop * 100)}%{suffix}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"example_nn_bayes_filtered_{str(fprop * 100)}%{suffix}.svg"),
            )
        )

    # ------------------------------------------------------------------------------------------------------------------------------
    ## Figure 4: we take an example place cell,
    # and we scatter plot a link between its firing rate and the decoding.

    def plot_pc_tuning_curve_and_predictions(self, ws=36):
        dirSave = os.path.join(self.folderFigures, "tuningCurves")
        if not os.path.isdir(dirSave):
            os.mkdir(dirSave)

        iwindow = self.timeWindows.index(ws)
        # Calculate the tuning curve of all place cells
        linearTuningCurves, binEdges = self.trainerBayes.calculate_linear_tuning_curve(
            self.l_function, self.behaviorData
        )
        placeFieldSort = self.trainerBayes.linearPosArgSort
        loadName = os.path.join(
            self.projectPath.dataPath,
            "aligned",
            str(ws),
            "test",
            f"spikeMat_window_popVector{self.suffix}.csv",
        )
        try:
            spikePopAligned = np.array(
                pd.read_csv(loadName).values[:, 1:], dtype=np.float32
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"""File {loadName} not found. Please run the spike alignment first in the WaveFormComparator class.
                If you're using Mouse_Results, you can run the following command:

                Mouse_Results.run_spike_alignment()

                """
            )
        spikePopAligned = spikePopAligned[
            : len(self.resultsNN["linTruePos"][iwindow]), :
        ]
        predLoss = self.resultsNN["predLoss"][iwindow]
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        for icell, tuningCurve in enumerate(linearTuningCurves):
            pcId = np.where(np.equal(placeFieldSort, icell))[0][0]
            spikeHist = spikePopAligned[:, pcId + 1][
                : len(self.resultsNN["linTruePos"][iwindow])
            ]
            spikeMask = np.greater(spikeHist, 0)

            if spikeMask.any():  # some neurons do not spike here
                cm = plt.get_cmap("gray")
                fig, ax = plt.subplots(figsize=(18, 12))
                ax.scatter(
                    self.resultsNN["linPred"][iwindow][spikeMask],
                    (spikeHist / np.sum(spikePopAligned, axis=1))[spikeMask],
                    s=12,
                    c=cm(normalize(predLoss[spikeMask])),
                    edgecolors="black",
                    linewidths=0.2,
                )

                errors = np.ones_like(binEdges[:-1]) * np.nan
                for i, linbin in enumerate(binEdges[:-1]):
                    errors[i] = np.mean(
                        np.abs(
                            self.resultsNN["linTruePos"][iwindow][
                                np.logical_and(
                                    spikeMask,
                                    np.logical_and(
                                        self.resultsNN["linPred"][iwindow] >= linbin,
                                        self.resultsNN["linPred"][iwindow]
                                        < binEdges[i + 1],
                                    ),
                                )
                            ]
                            - self.resultsNN["linPred"][iwindow][
                                np.logical_and(
                                    spikeMask,
                                    np.logical_and(
                                        self.resultsNN["linPred"][iwindow] >= linbin,
                                        self.resultsNN["linPred"][iwindow]
                                        < binEdges[i + 1],
                                    ),
                                )
                            ]
                        )
                    )

                ax.set_xlim(0, 1)

                cbar = plt.colorbar(
                    plt.cm.ScalarMappable(
                        plt.Normalize(
                            np.min(predLoss[spikeMask]), np.max(predLoss[spikeMask])
                        ),
                        cmap=cm,
                    ),
                    label="Predicted loss",
                    ax=ax,
                )
                # decrease colorbar ticks fontsize
                cbar.ax.tick_params(labelsize=12, rotation=-45)

                at = ax.twinx()
                at.spines["right"].set_visible(True)
                at.spines["right"].set_color("navy")
                at.spines["right"].set_linewidth(2.0)
                at.tick_params(axis="y", colors="navy")

                ax.set_xlabel("predicted linear position")
                ax.set_ylabel(
                    f"Number of spikes \n relative to total number of spike \n in {ws}ms window"
                )
                # show the yline in navy color
                # at.plot(binEdges[1:], tuningCurve, c="navy", alpha=0.5)

                at.errorbar(
                    binEdges[1:],
                    tuningCurve,
                    yerr=errors,
                    fmt="o-",
                    color="navy",
                    alpha=0.5,
                    label="tuning curve",
                )

                at.set_ylabel("firing rate w prediction error", color="navy")

                fig.tight_layout()
                fig.show()

                fig.savefig(
                    os.path.join(
                        dirSave, (f"{ws}_tc_pred_cluster{pcId}{self.suffix}.png")
                    )
                )
                # fig.savefig(os.path.join(dirSave, (f'{ws}_tc_pred_cluster{pcId}.svg')))
                plt.close()

    def boxplot_linError(self, timeWindows, dirSave=None, phase=None):
        """
        Boxplot of linear errors for NN and Bayes
        :param timeWindows: time windows used for decoding
        :param dirSave: directory to save the figure
        :param suffix: suffix to add to the figure name

        Will compute:
        :param lErrorNN_mean: mean linear error for NN
        :param lErrorBayes_mean: mean linear error for Bayes
        """
        from resultAnalysis.hyper_paper_figures import boxplot_linError

        if dirSave is None:
            dirSave = self.folderFigures
        if phase is None:
            phase = self.phase
        suffix = f"_{phase}" if phase else ""
        lErrorNN_mean = [
            np.mean(
                np.abs(
                    self.resultsNN_phase[suffix]["linTruePos"][
                        self.timeWindows.index(ws)
                    ]
                    - self.resultsNN_phase[suffix]["linPred"][
                        self.timeWindows.index(ws)
                    ]
                )
            )
            for ws in timeWindows
        ]
        lErrorBayes_mean = [
            np.mean(
                np.abs(
                    self.resultsNN_phase[suffix]["linTruePos"][
                        self.timeWindows.index(ws)
                    ]
                    - self.resultsBayes_phase[suffix]["linPred"][
                        self.timeWindows.index(ws)
                    ]
                )
            )
            for ws in timeWindows
        ]
        return boxplot_linError(
            lErrorNN_mean,
            lErrorBayes_mean,
            timeWindows,
            dirSave=dirSave,
            suffix=suffix,
        )

    def boxplot_euclError(self, timeWindows, dirSave=None, phase=None):
        """
        Boxplot of linear errors for NN and Bayes
        :param timeWindows: time windows used for decoding
        :param dirSave: directory to save the figure
        :param suffix: suffix to add to the figure name

        Will compute:
        :param lErrorNN_mean: mean linear error for NN
        :param lErrorBayes_mean: mean linear error for Bayes
        """
        from resultAnalysis.hyper_paper_figures import boxplot_euclError

        if dirSave is None:
            dirSave = self.folderFigures
        if phase is None:
            phase = self.phase
        suffix = f"_{phase}" if phase else ""
        euclErrorNN_mean = [
            np.mean(
                np.abs(
                    self.resultsNN_phase[suffix]["truePos"][self.timeWindows.index(ws)]
                    - self.resultsNN_phase[suffix]["fullPred"][
                        self.timeWindows.index(ws)
                    ]
                )
            )
            for ws in timeWindows
        ]
        euclErrorBayes_mean = [
            np.mean(
                np.linalg.norm(
                    self.resultsNN_phase[suffix]["truePos"][self.timeWindows.index(ws)]
                    - self.resultsBayes_phase[suffix]["fullPred"][
                        self.timeWindows.index(ws)
                    ],
                    axis=1,
                )
            )
            for ws in timeWindows
        ]
        return boxplot_euclError(
            euclErrorNN_mean,
            euclErrorBayes_mean,
            timeWindows,
            dirSave=dirSave,
            suffix=suffix,
        )

    # def fft_pc():
    #     #Compute Fourier transform of predicted positions:
    #     from scipy.fft import fft, fftfreq
    #     from scipy.interpolate import  interp1d
    #     # First interpolate in time the signal so that we sample them well:
    #     filters = [np.less(timePredsNoNoise_varyingWind[i][:,0],1645)*np.greater(timePredsNoNoise_varyingWind[i][:,0],1627) for i in range(4)]
    #     itps = [interp1d(timePredsNoNoise_varyingWind[i][filters[i],0],linearNoNoisePos_varyingWind[i][filters[i],0]) for i in range(4)]
    #     itpLast = np.min([np.max(timePredsNoNoise_varyingWind[i][filters[i],0]) for i in range(4)])
    #     itpFirst = np.max([np.min(timePredsNoNoise_varyingWind[i][filters[i],0]) for i in range(4)])
    #     x = np.arange(itpFirst,itpLast,0.003)
    #     discrete_linearPos = [itp(x) for itp in itps]
    #     fig,ax = plt.subplots()
    #     ax.scatter(x,discrete_linearPos[3])
    #     fig.show()
    #     spectrums = [fft(dlp) for dlp in discrete_linearPos]
    #     xf = fftfreq(x.shape[0], 0.003)
    #     fig,ax =plt.subplots()
    #     [ax.plot(xf[:1000],2.0/(x.shape[0]) * np.abs(spectrums[i][0:1000]),c=cm(i+4)) for i in [3]]
    #     ax.set_xlabel("frequency, Hz")
    #     ax.set_ylabel("Fourrier Power")
    #     fig.show()


if __name__ == "__main__":
    import warnings

    warnings.warn("Main process not implemented yet.")
