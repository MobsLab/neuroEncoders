# Load libs
import os

import dill as pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from scipy import stats
from scipy.stats import sem, zscore
from statannotations.Annotator import Annotator

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.simpleBayes.decode_bayes import Trainer as TrainerBayes
from neuroencoders.simpleBayes.decode_bayes import (
    extract_spike_counts,
    extract_spike_counts_from_matrix,
)
from neuroencoders.utils.global_classes import Project
from neuroencoders.utils.viz_params import (
    EC,
    MIDDLE_COLOR,
    SAFE_COLOR,
    SHOCK_COLOR,
    white_viridis,
)


class PaperFigures:
    def __init__(
        self,
        projectPath: Project,
        behaviorData: dict,
        trainerBayes: TrainerBayes,
        l_function,
        bayesMatrices: dict = {},
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
        self.resultsNN_phase_pkl = dict()
        self.resultsBayes_phase_pkl = dict()
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

    def load_data(self, suffixes=None, **kwargs):
        """
        Method to load the results of the neural network prediction.
        It loads the results from the csv files saved in the results folder of the experiment path.
        It prepares the data in a dictionary format for further analysis and should be called first.

        Parameters
        ----------
        suffix : str, optional
            Suffix to add to the file names, by default None. If None, it uses the class attribute self.suffix.
        **kwargs : dict, optional such as:
                extract_spike_counts: bool, whether to extract spike counts from csv files if they exist.

        Returns
        -------
        None
        """
        ### Load the NN prediction without using noise:
        if suffixes is None:
            self.suffixes = (
                [self.suffix] if not hasattr(self, "suffixes") else self.suffixes
            )
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
            resultsNN_phase_pkl = []
            spikes_count = []
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
                    ).flatten()
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
                    ).flatten()
                )
                try:
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
                        ).flatten()
                    )
                except FileNotFoundError:
                    print("Adding entropy as lossPred")
                    lossPred.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.projectPath.experimentPath,
                                        "results",
                                        str(ws),
                                        f"Hn{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        ).flatten()
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
                    ).flatten()
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
                    ).flatten()
                )

                if kwargs.get("load_pickle", False):
                    try:  # load pkl files if they exist
                        with open(
                            os.path.join(
                                self.projectPath.experimentPath,
                                "results",
                                str(ws),
                                f"decoding_results{suffix}.pkl",
                            ),
                            "rb",
                        ) as f:
                            results = pickle.load(f)
                            resultsNN_phase_pkl.append(results)
                    except FileNotFoundError:
                        print(
                            f"No pkl file found for resultsNN_phase{suffix} and window {str(ws)}, skipping loading it."
                        )
                        resultsNN_phase_pkl.append(None)
                if kwargs.get("extract_spikes_count", False):
                    try:
                        spikes_count.append(
                            pd.read_csv(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(ws),
                                    f"spikes_count{suffix}.csv",
                                )
                            )
                        )
                    except FileNotFoundError:
                        print(
                            f"No spikes_count file found for resultsNN_phase{suffix} and window {str(ws)}, skipping loading it."
                        )
                        spikes_count.append(None)

            speedMask = [ws.astype(bool) for ws in speedMask]

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
                if kwargs.get("extract_spikes_count", False):
                    self.resultsNN["spikes_count"] = spikes_count

            self.resultsNN_phase[suffix] = {
                "time": time,
                "speedMask": speedMask,
                "linPred": lPredPos,
                "fullPred": fPredPos,
                "truePos": truePos,
                "linTruePos": lTruePos,
                "predLoss": lossPred,
                "posIndex": posIndex,
                "spikes_count": spikes_count,
            }
            if kwargs.get("extract_spikes_count", False):
                self.resultsNN_phase[suffix]["spikes_count"] = spikes_count
            if kwargs.get("load_pickle", False):
                self.resultsNN_phase_pkl[suffix] = resultsNN_phase_pkl

    def load_bayes(self, suffixes=None, **kwargs):
        """
        Quickly load the bayesian decoding on the data, using the trainerBayes.
        If the bayesMatrices are not provided, it will train them using the
        trainerBayes.train_order_by_pos method.
        If the testing results are already saved, it will load them - otherwise it will perform a decoding.

        Parameters
        -------
        suffixes : str or list of str, optional
            Suffixes to add to the file names, by default None. If None, it uses the class attribute self.suffix.
        **kwargs : dict, optional such as:
            onTheFlyCorrection, bayesMatrices, redo.

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
        if not hasattr(self.trainerBayes, "linearPreferredPos") and kwargs.get(
            "load_bayesMatrices", False
        ):
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
            lTruePosBayes = []
            probaBayes = []
            fPredBayes = []
            fTruePosBayes = []
            posLossBayes = []
            timesBayes = []
            resultsBayes_phase_pkl = []
            total_spikes_count = []
            matrix_spikes_count = []
            for i, ws in enumerate(self.timeWindows):
                try:
                    lPredPosBayes.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.trainerBayes.folderResult,
                                        str(ws),
                                        f"bayes_linearPred{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        ).flatten()
                    )
                    lTruePosBayes.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.trainerBayes.folderResult,
                                        str(ws),
                                        f"bayes_linearTrue{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        ).flatten()
                    )
                    probaBayes.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.trainerBayes.folderResult,
                                        str(ws),
                                        f"bayes_proba{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        ).flatten()
                    )
                    fPredBayes.append(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.trainerBayes.folderResult,
                                    str(ws),
                                    f"bayes_featurePred{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                    fTruePosBayes.append(
                        np.array(
                            pd.read_csv(
                                os.path.join(
                                    self.trainerBayes.folderResult,
                                    str(ws),
                                    f"bayes_featureTrue{suffix}.csv",
                                )
                            ).values[:, 1:],
                            dtype=np.float32,
                        )
                    )
                    posLossBayes.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.trainerBayes.folderResult,
                                        str(ws),
                                        f"bayes_posLoss{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        ).flatten()
                    )
                    timesBayes.append(
                        np.squeeze(
                            np.array(
                                pd.read_csv(
                                    os.path.join(
                                        self.trainerBayes.folderResult,
                                        str(ws),
                                        f"bayes_timeStepsPred{suffix}.csv",
                                    )
                                ).values[:, 1:],
                                dtype=np.float32,
                            )
                        ).flatten()
                    )
                    if kwargs.get("extract_spikes_count", False):
                        total_count, _ = extract_spike_counts(
                            timesBayes[-1], self.trainerBayes.spikeMatTimes, ws / 1000
                        )
                        total_spikes_count.append(total_count)
                        matrix_count, _ = extract_spike_counts_from_matrix(
                            timesBayes[-1],
                            self.trainerBayes.spikeMat,
                            self.trainerBayes.spikeMatTimes,
                            ws / 1000,
                        )
                        matrix_spikes_count.append(matrix_count)

                    if (
                        fPredBayes[i].shape[0]
                        != self.resultsNN_phase[suffix]["fullPred"][i].shape[0]
                    ):
                        raise ValueError(
                            f"""
                            Bayesian and NN results do not have the same shape for
                            {str(ws)} ms window.
                            Found shapes {fPredBayes[i].shape} and {self.resultsNN_phase[suffix]["fullPred"][i].shape}.
                            """
                        )
                except (FileNotFoundError, ValueError) as e:
                    print(
                        f"""
                        Trouble finding bayesian results in folder, will test now because:
                        {e}
                        """
                    )

                    timesToPredict = self.resultsNN_phase[suffix]["time"][i][
                        :, np.newaxis
                    ].astype(np.float64)
                    useTrain_default = suffix != f"_{self.phase}"
                    useTrain = kwargs.get("useTrain", useTrain_default)
                    if useTrain:
                        print(f"Using training data for {suffix.strip('_')} phase!")
                    outputsBayes = self.trainerBayes.test_as_NN(
                        self.behaviorData,
                        self.bayesMatrices,
                        timesToPredict,
                        windowSizeMS=ws,
                        useTrain=useTrain,
                        l_function=self.l_function,
                        phase=suffix.strip("_"),
                        folderResult=os.path.join(
                            self.projectPath.experimentPath, "results"
                        ),  # here we choose base folder instead of trainerBayes.folderResult to save all results in the same place - bayesMatrices is already loaded anyway
                    )
                    infPos = outputsBayes["featurePred"]

                    if "linearPred" in outputsBayes:
                        lPredPosBayes.append(outputsBayes["linearPred"].flatten())
                    else:
                        _, linearBayesPos = self.l_function(infPos)
                        lPredPosBayes.append(linearBayesPos)

                    fPredBayes.append(infPos)
                    fTruePosBayes.append(outputsBayes["featureTrue"])
                    probaBayes.append(outputsBayes["proba"].flatten())
                    posLossBayes.append(outputsBayes["posLoss"].flatten())
                    lTruePosBayes.append(outputsBayes["linearTrue"].flatten())
                    timesBayes.append(outputsBayes["times"].flatten())

                if kwargs.get("load_pickle", False):
                    try:  # load pkl files if they exist
                        with open(
                            os.path.join(
                                self.trainerBayes.folderResult,
                                str(ws),
                                f"bayes_decoding_results{suffix}.pkl",
                            ),
                            "rb",
                        ) as f:
                            results = pickle.load(f)
                            resultsBayes_phase_pkl.append(results)
                    except FileNotFoundError:
                        print(
                            f"No pkl file found for resultsBayes_phase{suffix} and window {str(ws)}, skipping loading it."
                        )
                        resultsBayes_phase_pkl.append(None)

            # Output
            if suffix == self.suffix or len(self.suffixes) == 1:
                self.resultsBayes = {
                    "linPred": lPredPosBayes,
                    "linTruePos": lTruePosBayes,
                    "fullPred": fPredBayes,
                    "truePos": fTruePosBayes,
                    "predLoss": probaBayes,  # for compatibility with NN results but in reality corresponds to probaBayes
                    "posLossBayes": posLossBayes,
                    "timeNN": self.resultsNN_phase[suffix]["time"],
                    "time": timesBayes,  # should be exactly the same as timeNN
                    "speedMask": self.resultsNN_phase[suffix]["speedMask"],
                }
                if kwargs.get("extract_spikes_count", False):
                    self.resultsBayes.update(
                        {
                            "total_spikes_count": total_spikes_count,
                            "matrix_spikes_count": matrix_spikes_count,
                        }
                    )
            self.resultsBayes_phase[suffix] = {
                "linPred": lPredPosBayes,
                "linTruePos": lTruePosBayes,
                "fullPred": fPredBayes,
                "truePos": fTruePosBayes,
                "predLoss": probaBayes,
                "posLossBayes": posLossBayes,
                "timeNN": self.resultsNN_phase[suffix]["time"],
                "time": timesBayes,  # should be exactly the same as timeNN
                "speedMask": self.resultsNN_phase[suffix]["speedMask"],
            }
            if kwargs.get("extract_spikes_count", False):
                self.resultsBayes_phase[suffix].update(
                    {
                        "total_spikes_count": total_spikes_count,
                        "matrix_spikes_count": matrix_spikes_count,
                    }
                )
            if kwargs.get("load_pickle", False):
                self.resultsBayes_phase_pkl[suffix] = resultsBayes_phase_pkl

    def fig_example_XY(self, timeWindow, suffix=None, phase=None, block=True):
        idWindow = self.timeWindows.index(timeWindow)
        fig, ax = plt.subplots(
            2,
            2,
            sharex=True,
            sharey=True,
        )
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
        plt.show(block=block)
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

    def fig_example_linear(self, suffix=None, phase=None, block=True):
        ## Figure 1: on habituation set, speed filtered, we plot an example of bayesian and neural network decoding
        # ANN results
        # TODO: why is it speed filtered?
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        fig, ax = plt.subplots(len(self.timeWindows), 2, sharex=True, sharey=True)
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
        plt.show(block=block)
        fig.savefig(os.path.join(self.folderFigures, f"example_nn_bayes{suffix}.png"))
        fig.savefig(os.path.join(self.folderFigures, f"example_nn_bayes{suffix}.svg"))

    def compare_nn_bayes(
        self, timeWindow, suffix=None, phase=None, isCM=False, isShow=False, block=True
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
            fig, ax = plt.subplots(1, 1)
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
            plt.show(block=block)
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
        self,
        suffix=None,
        phase=None,
        speed="all",
        mask=None,
        use_mask=False,
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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
            np.floor(len(self.timeWindows) / 2).astype(int),
            2,
            sharex=True,
            constrained_layout=True,
        )
        ax = axes.flatten()
        # gs1 = gridspec.GridSpec(4, 4)
        # gs1.update(wspace=0.025, hspace=0.0001)
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
        [
            ax[2 * i].set_ylabel("cumulative \n histogram", fontsize="x-large")
            for i in range(len(self.timeWindows) // 2)
        ]
        [
            ax[i].set_xlabel("absolute linear error", fontsize="x-large")
            for i in range(len(self.timeWindows) // 2)
        ]
        fig.suptitle(
            f"Cumulative histograms of linear position errors for phase {suffix.strip('_')}"
        )
        fig.tight_layout()
        plt.show(block=block)
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

        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True)

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
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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
                np.ones(self.resultsNN_phase[suffix]["time"][iw].shape).astype(bool)
                * np.less_equal(
                    self.resultsNN_phase[suffix]["predLoss"][iw], thresh[iw]
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            filters_lpred = [
                np.ones(habMask[i].shape).astype(bool)
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
        fig, ax = plt.subplots()
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
        plt.show(block=block)
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
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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
                np.ones(self.resultsNN_phase[suffix]["time"][iw].shape).astype(bool)
                * np.less_equal(
                    self.resultsNN_phase[suffix]["predLoss"][iw], thresh[iw]
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            filters_lpred = [
                np.ones(habMask[i].shape).astype(bool)
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
        fig, ax = plt.subplots()
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
        plt.show(block=block)
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
        self,
        suffix=None,
        phase=None,
        speed="all",
        mask=None,
        use_mask=False,
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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
        fig, ax = plt.subplots(1, len(self.timeWindows))
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
        plt.show(block=block)
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

    def predLoss_vs_trueLoss(
        self, suffix=None, phase=None, speed="all", mode="2d", block=True, typeDec="ann"
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix

        if typeDec == "ann":
            results_phase = self.resultsNN_phase
        elif typeDec == "bayes":
            results_phase = self.resultsBayes_phase
        else:
            raise ValueError('typeDec argument could be only "NN" or "Bayes"')
        # Calculate error
        if mode == "2d":
            errors = [
                np.sqrt(
                    np.sum(
                        np.square(
                            results_phase[suffix]["truePos"][iw]
                            - results_phase[suffix]["fullPred"][iw]
                        ),
                        axis=1,
                    )
                )
                for iw in range(len(self.timeWindows))
            ]
        elif mode == "1d":
            errors = [
                np.abs(
                    results_phase[suffix]["linTruePos"][iw]
                    - results_phase[suffix]["linPred"][iw]
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            raise ValueError('mode argument could be only "2d" or "1d"')

        # Masks
        habMask = [
            inEpochsMask(
                results_phase[suffix]["time"][i],
                self.behaviorData["Times"]["testEpochs"],
            )
            for i in range(len(self.timeWindows))
        ]
        habMaskFast = [
            (habMask[i]) * (results_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i]) * np.logical_not(results_phase[suffix]["speedMask"][i][i])
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
        fig, ax = plt.subplots(1, len(self.timeWindows))
        if len(self.timeWindows) == 1:
            ax = [ax]  # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].scatter(
                results_phase[suffix]["predLoss"][iw][masks[iw]],
                errors[iw][masks[iw]],
                c="grey",
                s=1,
            )
            ax[iw].hist2d(
                results_phase[suffix]["predLoss"][iw][masks[iw]],
                errors[iw][masks[iw]],
                (30, 30),
                cmap=white_viridis,
                alpha=0.4,
                density=True,
            )  # ,c="red",alpha=0.4
            ax[iw].set_xlabel("Predicted loss" if typeDec == "NN" else "Bayes Proba")
            if mode == "2d":
                ax[iw].set_ylabel("True error")
            elif mode == "1d":
                ax[iw].set_ylabel("Linear error")
            ax[iw].set_title((str(self.timeWindows[iw]) + " ms"), fontsize="x-large")

            # modify xticks
            ax[iw].tick_params(axis="x", which="major", labelsize=15, rotation=45)
            ax[iw].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        fig.suptitle(
            f"{'Predicted loss' if typeDec == 'NN' else 'Bayes Proba'} vs true error during \n{str(speed)} speed periods for phase {suffix.strip('_')}"
        )
        fig.tight_layout()
        plt.show(block=block)
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"predLoss_vs_trueLoss{str(speed)}{suffix}_{typeDec}.png"),
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                (f"predLoss_vs_trueLoss{str(speed)}{suffix}_{typeDec}.svg"),
            )
        )

    def fig_example_2d(
        self,
        suffix=None,
        phase=None,
        speed="all",
        mask=None,
        use_mask=False,
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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
        fig, ax = plt.subplots(1, len(self.timeWindows))
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
        plt.show(block=block)
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
        self,
        suffix=None,
        phase=None,
        speed="all",
        num_steps=200,
        mask=None,
        use_mask=False,
        block=True,
        typeDec="ann",
        scaled=True,
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if typeDec == "ann":
            results_phase = self.resultsNN_phase
        elif typeDec == "bayes":
            results_phase = self.resultsBayes_phase
            if scaled:
                print("Warning: scaling is not allowed for Bayes decoder")
                scaled = False
        else:
            raise ValueError('typeDec argument could be only "NN" or "Bayes"')
        # Masks
        if mask is None:
            if use_mask:
                habMask = [
                    inEpochsMask(
                        results_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(results_phase[suffix]["time"][i].shape).astype(bool)
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask
        habMaskFast = [
            (habMask[i]) * (results_phase[suffix]["speedMask"][i])
            for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            (habMask[i]) * np.logical_not(results_phase[suffix]["speedMask"][i][i])
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
                results_phase[suffix]["linTruePos"][iw][masks[iw]]
                - results_phase[suffix]["linPred"][iw][masks[iw]]
            )
            for iw in range(len(self.timeWindows))
        ]
        predLoss = [
            results_phase[suffix]["predLoss"][iw][masks[iw]]
            for iw in range(len(self.timeWindows))
        ]
        if scaled:
            predLoss = [
                np.divide(
                    np.subtract(predLoss[iw], np.min(predLoss[iw])),
                    np.subtract(np.max(predLoss[iw]), np.min(predLoss[iw])),
                )
                for iw in range(len(self.timeWindows))
            ]

        predLoss_ticks = [
            np.linspace(
                np.min(predLoss[iw]),
                np.max(predLoss[iw]),
                num_steps,
            )
            for iw in range(len(self.timeWindows))
        ]
        errors_filtered = []
        filtering_func = np.less_equal if typeDec == "ann" else np.greater_equal
        for iw in range(len(self.timeWindows)):
            errors_filtered.append(
                [
                    np.mean(
                        errors[iw][
                            filtering_func(
                                predLoss[iw],
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
        fig, ax = plt.subplots(constrained_layout=True)
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
            f"{'Neural network' if typeDec == 'ann' else 'Bayesian decoder'} \n prediction filtering value",
            fontsize="x-large",
        )
        ax.set_ylabel("mean absolute linear error", fontsize="x-large")
        ax.set_title(
            (speed + " speed\n" + "phase " + suffix.strip("_")), fontsize="x-large"
        )
        fig.legend(loc=(0.87, 0.17), fontsize=12)
        plt.show(block=block)

        fig.savefig(
            os.path.join(
                self.folderFigures, f"predLoss_vs_Linerror{suffix}_{typeDec}.png"
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures, f"predLoss_vs_Linerror{suffix}_{typeDec}.svg"
            )
        )

        return predLoss_ticks[0], errors_filtered

    def predLoss_euclError(
        self,
        suffix=None,
        phase=None,
        speed="all",
        typeDec="ann",
        num_steps=200,
        isCM=False,
        scaled=True,
        mask=None,
        use_mask=False,
        block=True,
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if typeDec == "ann":
            results_phase = self.resultsNN_phase
        elif typeDec == "bayes":
            results_phase = self.resultsBayes_phase
            if scaled:
                print("Warning: scaling is not allowed for Bayes decoder")
                scaled = False
        else:
            raise ValueError('typeDec argument could be only "NN" or "Bayes"')

        REMOVED_PERCENTAGE = 1
        # Data
        nnD = {}
        if isCM:
            nnD["pred"] = [
                results_phase[suffix]["fullPred"][i] * EC
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                results_phase[suffix]["truePos"][i] * EC
                for i in range(len(self.timeWindows))
            ]
        else:
            nnD["pred"] = [
                results_phase[suffix]["fullPred"][i]
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                results_phase[suffix]["truePos"][i]
                for i in range(len(self.timeWindows))
            ]

        # Scale predicted loss between 0 and 1
        predLoss = [
            results_phase[suffix]["predLoss"][iw] for iw in range(len(self.timeWindows))
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
                        results_phase[suffix]["time"][i],
                        self.behaviorData["Times"]["testEpochs"],
                    )
                    for i in range(len(self.timeWindows))
                ]
            else:
                habMask = [
                    np.ones(results_phase[suffix]["time"][i].shape).astype(bool)
                    for i in range(len(self.timeWindows))
                ]
        else:
            habMask = mask
        habMaskFast = [
            results_phase[suffix]["speedMask"][i] for i in range(len(self.timeWindows))
        ]
        habMaskSlow = [
            np.logical_not(results_phase[suffix]["speedMask"][i])
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
                np.linspace(
                    np.min(predLoss_scaled[iw]), np.max(predLoss_scaled[iw]), num_steps
                )
                for iw in range(len(self.timeWindows))
            ]
        else:
            predLoss_ticks = [
                np.linspace(np.min(predLoss[iw]), np.max(predLoss[iw]), 1000)
                for iw in range(len(self.timeWindows))
            ]

        errors_filtered = np.zeros((len(self.timeWindows), len(predLoss_ticks[0])))
        filtering_func = np.less_equal if typeDec == "ann" else np.greater_equal
        for iw in range(len(self.timeWindows)):
            percFiltered = np.array(
                [
                    np.sum([filtering_func(predLoss_scaled[iw], pfilt)])
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
                            filtering_func(predLoss_scaled[iw][masks[iw]], pfilt)
                        ]
                    )
                    for pfilt in predLoss_ticks[iw]
                ]
            )
            errors_filtered[iw][maskFilterout] = np.nan

        labelNames = [
            (str(self.timeWindows[iw]) + " ms") for iw in range(len(self.timeWindows))
        ]
        fig, ax = plt.subplots(constrained_layout=True)
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
            f"{'Neural network' if typeDec == 'ann' else 'Bayesian decoder'} \n prediction filtering value",
            fontsize="x-large",
        )
        ax.set_ylabel("Euclidean error (cm)", fontsize="x-large")
        ax.set_title(
            (speed + " speed" + " and phase " + suffix.strip("_")), fontsize="x-large"
        )
        fig.legend(loc=(0.87, 0.17), fontsize=12)
        plt.show(block=block)

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

    def plot_boxplot_error(
        self, results_df, logscale=True, speed="fast", confidence=False, threshold=0.6
    ):
        # for every phase, plot the whisker plot of fast_filtered_error, with a hue on winMS
        if "all_se_error" not in results_df.columns:
            results_df["all_se_error"] = results_df.apply(
                lambda row: np.array(
                    [
                        np.linalg.norm(
                            row["fullPred"][i, :2] - row["truePos"][i, :2],
                        )
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )
            # Now get the filtered MSE error (ie apply nan on no speedMask from the full all_se_error array)
            results_df["fast_filtered_se_error"] = results_df.apply(
                lambda row: np.array(
                    [
                        row["all_se_error"][i] if row["speedMask"][i] else np.nan
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )
            results_df["slow_filtered_se_error"] = results_df.apply(
                lambda row: np.array(
                    [
                        row["all_se_error"][i] if not row["speedMask"][i] else np.nan
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )

            # Get the filtered MSE error (ie only on fast epochs, do the mse of fullPred[speedMask,:2] and truePos[speedMask, :2])
        if speed == "fast":
            column = "fast_filtered_se_error"
        elif speed == "slow":
            column = "slow_filtered_se_error"
        elif speed == "all":
            column = "all_se_error"
        else:
            raise ValueError("Speed value must be fast, slow, or all.i")

        if confidence:
            results_df[f"confidence_{column}"] = results_df.apply(
                lambda row: np.array(
                    [
                        row[column][i] if row["predLoss"][i] < threshold else np.nan
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )
            column = f"confidence_{column}"

        plt.figure()
        results_exploded = results_df.explode(column).reset_index(drop=True)
        sns.boxplot(
            data=results_exploded,
            x="phase",
            y=column,
            hue="winMS",
            order=["training", "pre", "cond", "post"],
        )
        plt.title(f"{speed.capitalize()} epochs squared error by phase and window size")
        if logscale:
            plt.yscale("log")
            plt.ylabel("Squared Error (log)")
        else:
            plt.ylabel("Squared Error")
        plt.xlabel("Phase")
        plt.legend(title="Window Size (ms)")
        plt.savefig(
            os.path.join(
                self.folderFigures,
                f"boxplot_{speed}_filtered_se_error.png",
            )
        )
        plt.show()
        return results_df

    def lin_boxplot_error(
        self, results_df, logscale=True, speed="fast", confidence=False, threshold=0.6
    ):
        # for every phase, plot the whisker plot of fast_filtered_error, with a hue on winMS
        if "lin_all_se_error" not in results_df.columns:
            results_df["lin_all_se_error"] = results_df.apply(
                lambda row: np.abs(row["linPred"] - row["linTruePos"]),
                axis=1,
            )
            # Now get the filtered MSE error (ie apply nan on no speedMask from the full all_se_error array)
            results_df["lin_fast_filtered_se_error"] = results_df.apply(
                lambda row: np.array(
                    [
                        row["lin_all_se_error"][i] if row["speedMask"][i] else np.nan
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )
            results_df["lin_slow_filtered_se_error"] = results_df.apply(
                lambda row: np.array(
                    [
                        row["lin_all_se_error"][i]
                        if not row["speedMask"][i]
                        else np.nan
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )

            # Get the filtered MSE error (ie only on fast epochs, do the mse of fullPred[speedMask,:2] and truePos[speedMask, :2])
        if speed == "fast":
            column = "lin_fast_filtered_se_error"
        elif speed == "slow":
            column = "lin_slow_filtered_se_error"
        elif speed == "all":
            column = "lin_all_se_error"
        else:
            raise ValueError("Speed value must be fast, slow, or all.i")

        if confidence:
            results_df[f"confidence_{column}"] = results_df.apply(
                lambda row: np.array(
                    [
                        row[column][i] if row["predLoss"][i] < threshold else np.nan
                        for i in range(row["truePos"].shape[0])
                    ]
                ),
                axis=1,
            )
            column = f"confidence_{column}"

        plt.figure()
        sns.boxplot(
            data=results_df.explode(column).reset_index(drop=True),
            x="phase",
            y=column,
            hue="winMS",
            order=["training", "pre", "cond", "post"],
        )
        plt.title(f"{speed.capitalize()} epochs linear error by phase and window size")
        if logscale:
            plt.yscale("log")
            plt.ylabel("Absolute Lin Error (log)")
        else:
            plt.ylabel("Absolute Lin Error")
        plt.xlabel("Phase")
        plt.legend(title="Window Size (ms)")
        plt.savefig(
            os.path.join(
                self.folderFigures,
                f"linboxplot_{speed}_filtered_se_error_{confidence=}_{threshold=}.png",
            )
        )
        plt.show()
        return results_df

    def fig_proba_heatmap_error(
        self,
        winMS,
        normalized_by="true",
        plot_bias=False,
        plot_surprise=False,
        show=True,
    ):
        phases = ["training", "pre", "cond", "post"]
        speeds = ["all", "slow", "fast"]
        fig, axs = plt.subplots(
            len(phases),
            len(speeds) * (3 if plot_bias else 2),
            figsize=(20 if plot_bias else 15, 10),
            sharex=True,
            sharey=True,
        )
        idWindow = self.timeWindows.index(winMS)
        for i, phase in enumerate(phases):
            phase = "_" + phase
            for j, speed in enumerate(speeds):
                if speed == "all":
                    speedMask = np.ones_like(
                        self.resultsNN_phase[phase]["speedMask"][idWindow],
                        dtype=bool,
                    )
                elif speed == "slow":
                    speedMask = np.logical_not(
                        self.resultsNN_phase[phase]["speedMask"][idWindow]
                    )
                elif speed == "fast":
                    speedMask = self.resultsNN_phase[phase]["speedMask"][idWindow]
                else:
                    raise ValueError("Speed must be 'all', 'slow' or 'fast'")

                logits_hw = self.resultsNN_phase_pkl[phase][idWindow]["logits_hw"][
                    speedMask
                ]
                truePos = self.resultsNN_phase[phase]["truePos"][idWindow][speedMask]
                predPos = self.resultsNN_phase[phase]["fullPred"][idWindow][speedMask]
                target_hw = self.ann[
                    str(winMS)
                ].GaussianHeatmap.gaussian_heatmap_targets(truePos)
                probs = (
                    self.ann[str(winMS)]
                    .GaussianHeatmap.decode_and_uncertainty(
                        logits_hw, return_probs=True
                    )[-1]
                    .numpy()
                )
                error = np.linalg.norm(truePos[:, :2] - predPos[:, :2], axis=1)
                mean_probs = np.mean(probs, axis=0)
                hist2d, xedges, yedges = np.histogram2d(
                    truePos[:, 0], truePos[:, 1], bins=50, weights=error
                )
                if normalized_by == "true":
                    hist2d_counts, _, _ = np.histogram2d(
                        truePos[:, 0], truePos[:, 1], bins=50
                    )
                elif normalized_by == "pred":
                    hist2d_counts, _, _ = np.histogram2d(
                        predPos[:, 0], predPos[:, 1], bins=50
                    )
                else:
                    raise ValueError("normalized_by must be 'true', or 'pred'")

                hist2d_mean_error = np.divide(
                    hist2d,
                    hist2d_counts,
                    out=np.zeros_like(hist2d),
                    where=hist2d_counts != 0,
                )  # avoid division by zero
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                if plot_surprise:
                    # probs: [N, H, W], truePos: [N, 2]
                    bin_indices_x = np.digitize(truePos[:, 1], yedges) - 1
                    bin_indices_y = np.digitize(truePos[:, 0], xedges) - 1
                    # Clip indices to valid range
                    bin_indices_x = np.clip(bin_indices_x, 0, probs.shape[1] - 1)
                    bin_indices_y = np.clip(bin_indices_y, 0, probs.shape[2] - 1)
                    surprise = -np.log(
                        probs[np.arange(len(truePos)), bin_indices_x, bin_indices_y]
                    )
                    hist2d_surprise, _, _ = np.histogram2d(
                        truePos[:, 0],
                        truePos[:, 1],
                        bins=[xedges, yedges],
                        weights=surprise,
                    )
                    hist2d_mean_error = np.divide(
                        hist2d_surprise,
                        hist2d_counts,
                        out=np.zeros_like(hist2d_surprise),
                        where=hist2d_counts != 0,
                    )

                ax1 = axs[i, j * (3 if plot_bias else 2)]
                im1 = ax1.imshow(
                    mean_probs,
                    origin="lower",
                    extent=extent,
                    vmin=0,
                    vmax=mean_probs.max(),
                )
                ax1.set_title(f"{phase[1:]}-{speed}-Proba")
                plt.colorbar(im1, ax=ax1)
                ax2 = axs[i, j * (3 if plot_bias else 2) + 1]
                im2 = ax2.imshow(
                    hist2d_mean_error.T,
                    origin="lower",
                    extent=extent,
                    vmin=0,
                    vmax=np.nanmax(hist2d_mean_error),
                )
                ax2.set_title(f"{phase[1:]}-{speed}-Error")
                plt.colorbar(im2, ax=ax2)

                # --- Bias heatmap ---
                if plot_bias:
                    # Compute mean bias vector in each bin
                    bias_x = np.zeros((50, 50))
                    bias_y = np.zeros((50, 50))
                    for xi in range(50):
                        for yi in range(50):
                            mask_bin = (
                                (truePos[:, 0] >= xedges[xi])
                                & (truePos[:, 0] < xedges[xi + 1])
                                & (truePos[:, 1] >= yedges[yi])
                                & (truePos[:, 1] < yedges[yi + 1])
                            )
                            if np.any(mask_bin):
                                bias_vec = np.mean(
                                    predPos[mask_bin, :2] - truePos[mask_bin, :2],
                                    axis=0,
                                )
                                bias_x[xi, yi] = bias_vec[0]
                                bias_y[xi, yi] = bias_vec[1]
                            else:
                                bias_x[xi, yi] = np.nan
                                bias_y[xi, yi] = np.nan
                    ax3 = axs[i, j * (3 if plot_bias else 2) + 2]
                    bias_mag = np.sqrt(bias_x.T**2 + bias_y.T**2)
                    im3 = ax3.imshow(
                        bias_mag,
                        origin="lower",
                        extent=extent,
                        vmin=0,
                        vmax=np.nanmax(bias_mag),
                    )
                    ax3.set_title(f"{phase[1:]}-{speed}-Bias")
                    plt.colorbar(im3, ax=ax3)
                    # Optionally, overlay quiver arrows for direction
                    skip = 5  # reduce arrow density
                    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
                    ax3.quiver(
                        X[::skip, ::skip],
                        Y[::skip, ::skip],
                        bias_x.T[::skip, ::skip],
                        bias_y.T[::skip, ::skip],
                        scale=0.05,
                        color="black",
                        alpha=0.7,
                    )
        plt.suptitle(
            f"Heatmap of mean probability and mean euclidean error for window size {winMS} ms\n normalized by {normalized_by} position. Surprise: {plot_surprise}. Bias: {plot_bias}"
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"heatmap_proba_error_{normalized_by}_surprise_{plot_surprise}_bias_{plot_bias}_{winMS}.png",
            ),
            bbox_inches="tight",
        )
        if show:
            plt.show(block=True)
        plt.close(fig)

    def fig_proba_heatmap_vs_true(
        self, winMS, plot_kl=False, per_trial=True, show=True
    ):
        phases = ["training", "pre", "cond", "post"]
        speeds = ["all", "slow", "fast"]
        fig, axs = plt.subplots(
            len(phases),
            len(speeds) * (3 if plot_kl else 2),
            figsize=(20 if plot_kl else 15, 10),
            sharex=True,
            sharey=True,
        )
        idWindow = self.timeWindows.index(winMS)
        for i, phase in enumerate(phases):
            phase = "_" + phase
            for j, speed in enumerate(speeds):
                if speed == "all":
                    speedMask = np.ones_like(
                        self.resultsNN_phase[phase]["speedMask"][idWindow],
                        dtype=bool,
                    )
                elif speed == "slow":
                    speedMask = np.logical_not(
                        self.resultsNN_phase[phase]["speedMask"][idWindow]
                    )
                elif speed == "fast":
                    speedMask = self.resultsNN_phase[phase]["speedMask"][idWindow]
                else:
                    raise ValueError("Speed must be 'all', 'slow' or 'fast'")

                logits_hw = self.resultsNN_phase_pkl[phase][idWindow]["logits_hw"][
                    speedMask
                ]
                truePos = self.resultsNN_phase[phase]["truePos"][idWindow][speedMask]
                target_hw = (
                    self.ann[str(winMS)]
                    .GaussianHeatmap.gaussian_heatmap_targets(truePos)
                    .numpy()
                )
                probs = (
                    self.ann[str(winMS)]
                    .GaussianHeatmap.decode_and_uncertainty(
                        logits_hw, return_probs=True
                    )[-1]
                    .numpy()
                )
                mean_probs = np.mean(probs, axis=0)
                target_mean = np.mean(target_hw, axis=0)
                if per_trial:
                    flat_err = (probs - target_hw).flatten()
                    zmap = zscore(flat_err).reshape(probs.shape).mean(axis=0)
                else:
                    flat_err = (mean_probs - target_mean).flatten()
                    zmap = zscore(flat_err).reshape(mean_probs.shape)

                extent = (0, 1, 0, 1)
                ax1 = axs[i, j * (3 if plot_kl else 2)]
                im1 = ax1.imshow(
                    mean_probs,
                    origin="lower",
                    extent=extent,
                    vmin=0,
                    vmax=mean_probs.max(),
                )
                ax1.set_title(f"{phase[1:]}-{speed}-Proba")
                # plt.colorbar(im1, ax=ax1)
                ax2 = axs[i, j * (3 if plot_kl else 2) + 1]
                im2 = ax2.imshow(
                    zmap,
                    origin="lower",
                    cmap="coolwarm",
                    extent=extent,
                )
                ax2.set_title(f"{speed}-Error (z-scored)")
                # plt.colorbar(im2, ax=ax2)

                if plot_kl:
                    ax3 = axs[i, j * (3 if plot_kl else 2) + 2]
                    if per_trial:
                        P = np.divide(target_hw, target_mean.sum() + 1e-12, axis=0)
                        Q = np.divide(mean_probs, mean_probs.sum() + 1e-12, axis=0)
                        kl_map = np.where(
                            Q > 0, Q * np.log((Q + 1e-12) / (P + 1e-12)), 0
                        ).mean(axis=0)
                    else:
                        P = target_mean / (target_mean.sum() + 1e-12)
                        Q = mean_probs / (mean_probs.sum() + 1e-12)
                        kl_map = np.where(
                            Q > 0, Q * np.log((Q + 1e-12) / (P + 1e-12)), 0
                        )
                    # Compute mean bias vector in each bin
                    im3 = ax3.imshow(
                        kl_map,
                        cmap="magma",
                        origin="lower",
                        extent=extent,
                    )
                    ax3.set_title(f"{speed}-KL Divergence")
                    # plt.colorbar(im3, ax=ax3)
        plt.suptitle(
            f"Heatmap of mean probability and mean euclidean error for window size {winMS} ms.\n Abs: {plot_kl}"
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"heatmap_vs_target_abs_{plot_kl}_{winMS}.png",
            ),
            bbox_inches="tight",
        )
        if show:
            plt.show(block=True)
        plt.close(fig)

    def fig_example_linear_filtered(
        self, suffix=None, phase=None, fprop=0.3, block=True
    ):
        # TODO: add filtering AND plots for bayesian decoder
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
            np.argsort(self.resultsBayes_phase[suffix]["predLoss"][iw])
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
                self.resultsBayes_phase[suffix]["predLoss"][iw][
                    sortedprobaBayes[iw][int(len(sortedprobaBayes[iw]) * fprop)]
                ]
            )
            for iw in range(len(self.timeWindows))
        ]

        filters_lpred = [
            np.ones(self.resultsNN_phase[suffix]["time"][iw].shape).astype(bool)
            * np.less_equal(self.resultsNN_phase[suffix]["predLoss"][iw], thresh[iw])
            for iw in range(len(self.timeWindows))
        ]
        filters_bayes = [
            np.ones(self.resultsBayes_phase[suffix]["time"][iw].shape).astype(bool)
            * np.greater_equal(
                self.resultsBayes_phase[suffix]["predLoss"][iw], threshBayes[iw]
            )
            for iw in range(len(self.timeWindows))
        ]

        fig, ax = plt.subplots(len(self.timeWindows), 2, sharex=True, sharey=True)
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
                    [
                        ax[i, col].set_title(
                            "Best "
                            + str(fprop * 100)
                            + "% of predictions \n"
                            + str(self.timeWindows[0])
                            + " ms window for phase "
                            + suffix.strip("_"),
                            fontsize="xx-large",
                        )
                        for col in range(ax[i].shape[0])
                    ]
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
        plt.show(block=block)
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

    def error_right_left_arm(
        self,
        suffix=None,
        phase=None,
        speed="fast",
        mask=None,
        use_mask=False,
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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
        if speed == "fast":
            mask = habMaskFast
        elif speed == "slow":
            mask = habMaskSlow
        else:
            mask = habMask

        trueLinearPos = [
            self.resultsNN_phase[suffix]["linTruePos"][i]
            for i in range(len(self.timeWindows))
        ]

        mask_right_arm_pred_argmax = [
            np.greater_equal(self.resultsNN_phase[suffix]["linPred"][iw], 0.7)
            for iw in range(len(self.timeWindows))
        ]

        error_rightarm = [
            np.abs(self.resultsNN_phase[suffix]["linPred"][i] - trueLinearPos[i])[
                mask_right_arm_pred_argmax[i] * mask[i]
            ]
            for i in range(len(self.timeWindows))
        ]
        error_OtherArm = [
            np.abs(self.resultsNN_phase[suffix]["linPred"][i] - trueLinearPos[i])[
                np.logical_not(mask_right_arm_pred_argmax[i]) * mask[i]
            ]
            for i in range(len(self.timeWindows))
        ]

        mask_middle_arm_pred_argmax = [
            np.greater_equal(self.resultsNN_phase[suffix]["linPred"][i], 0.3)
            * np.less(self.resultsNN_phase[suffix]["linPred"][i], 0.7)
            for i in range(len(self.timeWindows))
        ]
        error_MiddleArm = [
            np.abs(self.resultsNN_phase[suffix]["linPred"][i] - trueLinearPos[i])[
                mask_middle_arm_pred_argmax[i] * mask[i]
            ]
            for i in range(len(self.timeWindows))
        ]

        error_LeftArm = [
            np.abs(self.resultsNN_phase[suffix]["linPred"][i] - trueLinearPos[i])[
                np.logical_not(mask_middle_arm_pred_argmax[i])
                * np.logical_not(mask_right_arm_pred_argmax[i])
                * mask[i]
            ]
            for i in range(len(self.timeWindows))
        ]
        fig, _axs = plt.subplots(2, len(self.timeWindows) // 2)
        axs = _axs.flatten()
        for i, ax in enumerate(axs):
            ax.hist(
                error_rightarm[i],
                color=SAFE_COLOR,
                histtype="step",
                density=True,
                bins=50,
                label="Right Arm",
            )
            ax.vlines(np.median(error_rightarm[i]), ymin=0, ymax=16, color=SAFE_COLOR)
            ax.hist(
                error_MiddleArm[i],
                color=MIDDLE_COLOR,
                histtype="step",
                density=True,
                bins=50,
                label="Middle Arm",
            )
            ax.vlines(
                np.median(error_MiddleArm[i]), ymin=0, ymax=16, color=MIDDLE_COLOR
            )
            ax.hist(
                error_LeftArm[i],
                color=SHOCK_COLOR,
                histtype="step",
                density=True,
                bins=50,
                label="Left Arm",
            )
            ax.vlines(np.median(error_LeftArm[i]), ymin=0, ymax=16, color=SHOCK_COLOR)
            ax.hist(
                error_OtherArm[i],
                color="gray",
                histtype="step",
                density=True,
                bins=50,
                label="Non-Right Arm",
            )
            ax.vlines(np.median(error_OtherArm[i]), ymin=0, ymax=16, color="gray")
            ax.set_xlabel("error distrib")
            ax.set_title(f"{self.timeWindows[i]} ms")
        axs[-1].legend()
        fig.suptitle(f"Histogramms of error for phase {suffix.strip('_')}")
        plt.show(block=block)
        fig.savefig(os.path.join(self.folderFigures, f"error_hist_by_arm{suffix}.png"))
        fig.savefig(os.path.join(self.folderFigures, f"error_hist_by_arm{suffix}.svg"))
        plt.close()

        fig, _axs = plt.subplots(2, len(self.timeWindows) // 2)
        axs = _axs.flatten()
        for i, ax in enumerate(axs):
            ax.scatter(
                self.resultsNN_phase[suffix]["linPred"][i][
                    mask_right_arm_pred_argmax[i] * mask[i]
                ],
                error_rightarm[i],
                c=SAFE_COLOR,
                s=10,
            )
            ax.scatter(
                self.resultsNN_phase[suffix]["linPred"][i][
                    np.logical_not(mask_middle_arm_pred_argmax[i])
                    * np.logical_not(mask_right_arm_pred_argmax[i])
                    * mask[i]
                ],
                error_LeftArm[i],
                c=SHOCK_COLOR,
                s=10,
            )
            ax.scatter(
                self.resultsNN_phase[suffix]["linPred"][i][
                    mask_middle_arm_pred_argmax[i] * mask[i]
                ],
                error_MiddleArm[i],
                c=MIDDLE_COLOR,
                s=10,
            )
            ax.set_xlabel("Linear Predicted")
            ax.set_ylabel("Linear Error")
            ax.set_title(f"{self.timeWindows[i]} ms")
        fig.suptitle(f"Differential errors for phase {suffix.strip('_')}")
        plt.show(block=block)
        fig.savefig(
            os.path.join(self.folderFigures, f"error_scatter_by_arm{suffix}.png")
        )
        fig.savefig(
            os.path.join(self.folderFigures, f"error_scatter_by_arm{suffix}.svg")
        )
        plt.close()

    def compare_nn_bayes_with_random_pred(
        self, timeWindow, suffix=None, phase=None, block=True
    ):
        # TODO: multi time windows ?
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix

        idWindow = self.timeWindows.index(timeWindow)

        errors = []
        errorsRandomMean = []
        errorsRandomStd = []
        errorsShuffleMean = []
        errorsShuffleStd = []
        predLoss = self.resultsNN_phase[suffix]["predLoss"][idWindow]
        probaBayes = -self.resultsBayes_phase[suffix]["predLoss"][idWindow][
            ::-1
        ]  # rescale between -1 and 0

        timeNN = self.resultsNN_phase[suffix]["time"][idWindow].flatten()
        timeBayes = self.resultsBayes_phase[suffix]["time"][idWindow].flatten()
        if timeNN.shape != timeBayes.shape:
            raise ValueError("Time vectors for NN and Bayes do not match in shape.")

        # Define quantile levels
        quantiles = np.linspace(0, 100, 21)  # 0%, 5%, ..., 100%
        used_quantile = np.zeros_like(quantiles, dtype=bool)
        # Get quantile values for each array
        predLoss_quantiles = np.percentile(predLoss, quantiles)
        probaBayes_quantiles = np.percentile(probaBayes, quantiles)

        for i, (pl, pb) in enumerate(
            tqdm.tqdm(
                zip(predLoss_quantiles, probaBayes_quantiles),
                total=len(quantiles),
                desc="Processing quantiles",
            )
        ):
            nn_mask = np.less(predLoss, pl).flatten()
            bayes_mask = np.less(probaBayes, pb).flatten()
            # Find the intersection of masks (same indices)
            common_mask = nn_mask & bayes_mask
            NNpred = self.resultsNN_phase[suffix]["linPred"][idWindow][common_mask]
            bayesPred = self.resultsBayes_phase[suffix]["linPred"][idWindow][
                common_mask
            ]

            if NNpred.shape[0] > 0:
                used_quantile[i] = True
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
        ax.plot(
            quantiles[used_quantile],
            errors,
            label="nn vs bayesian",
        )
        ax.plot(
            quantiles[used_quantile],
            errorsRandomMean,
            color="red",
            label="nn vs random Prediction",
        )

        ax.fill_between(
            quantiles[used_quantile],
            errorsRandomMean + errorsRandomStd,
            errorsRandomMean - errorsRandomStd,
            color="orange",
        )
        ax.plot(
            quantiles[used_quantile],
            errorsShuffleMean,
            color="purple",
            label="nn vs shuffle bayesian",
        )
        ax.fill_between(
            quantiles[used_quantile],
            errorsShuffleMean + errorsShuffleStd,
            errorsShuffleMean - errorsShuffleStd,
            color="violet",
        )

        ax.set_ylabel("Absolute Error (NN vs Bayesian/Random Prediction)")
        ax.set_xlabel("Quantile (% of best predictions ordered by confidence)")
        ax.set_title(
            f"Comparison of NN and Bayesian Decoder Errors Across Quantiles\nWindow Size: {self.timeWindows[idWindow]} ms, Phase: {suffix.strip('_')}",
            fontsize="xx-large",
        )

        fig.legend(loc=[0.2, 0.2])
        plt.show(block=block)
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"fig_lineardiffBayesNN_{timeWindow}_ms{suffix}.png",
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"fig_lineardiffBayesNN_{timeWindow}_ms{suffix}.svg",
            )
        )
        plt.close()
        return np.mean(errors), np.mean(errorsShuffleMean), np.mean(errorsShuffleStd)

    # ------------------------------------------------------------------------------------------------------------------------------
    ## Figure 4: we take an example place cell,
    # and we scatter plot a link between its firing rate and the decoding.

    def plot_pc_tuning_curve_and_predictions(
        self,
        suffix=None,
        phase=None,
        ws=None,
        block=True,
        show=False,
        useTrain=False,
        useAll=True,
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if ws is None:
            ws = self.timeWindows[0]  # default to the first time window

        dirSave = os.path.join(self.folderFigures, "tuningCurves")
        if not os.path.isdir(dirSave):
            os.mkdir(dirSave)

        iwindow = self.timeWindows.index(ws)
        # Calculate the tuning curve of all place cells
        linearTuningCurves, binEdges = self.trainerBayes.calculate_linear_tuning_curve(
            self.l_function, self.behaviorData
        )
        try:
            placeFieldSort = self.trainerBayes.linearPosArgSort
        except AttributeError:
            print(
                "linearPosArgSort not found in Trainer Bayes, will try to order by position."
            )
            self.bayesMatrices = self.trainerBayes.train_order_by_pos(
                self.behaviorData,
                l_function=self.l_function,
                bayesMatrices=self.bayesMatrices
                if (
                    (isinstance(self.bayesMatrices, dict))
                    and ("Occupation" in self.bayesMatrices.keys())
                )
                else None,
            )
            placeFieldSort = self.trainerBayes.linearPosArgSort

        loadName = os.path.join(
            self.projectPath.dataPath,
            f"aligned{suffix}{'_all' if useAll else ''}",
            str(ws),
            "test" if not useTrain else "train",
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
            : len(self.resultsNN_phase[suffix]["predLoss"]["linTruePos"][iwindow]), :
        ]
        predLoss = self.resultsNN_phase[suffix]["predLoss"]["predLoss"][iwindow]
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        for icell, tuningCurve in enumerate(linearTuningCurves):
            pcId = np.where(np.equal(placeFieldSort, icell))[0][0]
            spikeHist = spikePopAligned[:, pcId + 1][
                : len(self.resultsNN_phase[suffix]["predLoss"]["linTruePos"][iwindow])
            ]
            spikeMask = np.greater(spikeHist, 0)

            if spikeMask.any():  # some neurons do not spike here
                cm = plt.get_cmap("gray")
                fig, ax = plt.subplots()
                ax.scatter(
                    self.resultsNN_phase[suffix]["predLoss"]["linPred"][iwindow][
                        spikeMask
                    ],
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
                            self.resultsNN_phase[suffix]["predLoss"]["linTruePos"][
                                iwindow
                            ][
                                np.logical_and(
                                    spikeMask,
                                    np.logical_and(
                                        self.resultsNN_phase[suffix]["predLoss"][
                                            "linPred"
                                        ][iwindow]
                                        >= linbin,
                                        self.resultsNN_phase[suffix]["predLoss"][
                                            "linPred"
                                        ][iwindow]
                                        < binEdges[i + 1],
                                    ),
                                )
                            ]
                            - self.resultsNN_phase[suffix]["predLoss"]["linPred"][
                                iwindow
                            ][
                                np.logical_and(
                                    spikeMask,
                                    np.logical_and(
                                        self.resultsNN_phase[suffix]["predLoss"][
                                            "linPred"
                                        ][iwindow]
                                        >= linbin,
                                        self.resultsNN_phase[suffix]["predLoss"][
                                            "linPred"
                                        ][iwindow]
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
                if show:
                    plt.show(block=block)

                fig.savefig(
                    os.path.join(dirSave, (f"{ws}_tc_pred_cluster{pcId}{suffix}.png"))
                )
                plt.close()

    def boxplot_linError(
        self, timeWindows, dirSave=None, suffix=None, phase=None, block=True
    ):
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
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        suffix = f"_{phase}" if phase else ""
        lErrorNN_mean = [
            np.mean(
                np.abs(
                    self.resultsNN_phase[suffix]["predLoss"]["linTruePos"][
                        self.timeWindows.index(ws)
                    ]
                    - self.resultsNN_phase[suffix]["predLoss"]["linPred"][
                        self.timeWindows.index(ws)
                    ]
                )
            )
            for ws in timeWindows
        ]
        lErrorBayes_mean = [
            np.mean(
                np.abs(
                    self.resultsNN_phase[suffix]["predLoss"]["linTruePos"][
                        self.timeWindows.index(ws)
                    ]
                    - self.resultsBayes_phase[suffix]["predLoss"]["linPred"][
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

    def boxplot_euclError(
        self, timeWindows, dirSave=None, suffix=None, phase=None, block=True
    ):
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
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        euclErrorNN_mean = [
            np.mean(
                np.abs(
                    self.resultsNN_phase[suffix]["predLoss"]["truePos"][
                        self.timeWindows.index(ws)
                    ]
                    - self.resultsNN_phase[suffix]["predLoss"]["fullPred"][
                        self.timeWindows.index(ws)
                    ]
                )
            )
            for ws in timeWindows
        ]
        euclErrorBayes_mean = [
            np.mean(
                np.linalg.norm(
                    self.resultsNN_phase[suffix]["predLoss"]["truePos"][
                        self.timeWindows.index(ws)
                    ]
                    - self.resultsBayes_phase[suffix]["predLoss"]["fullPred"][
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

    def fft_pc(
        self,
        winValues=None,
        masks=None,
        suffix=None,
        phase=None,
        decoding="ann",
        block=True,
    ):
        # Compute Fourier transform of predicted positions:
        from scipy.fft import fft, fftfreq
        from scipy.interpolate import interp1d

        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix

        if winValues is None:
            winValues = self.timeWindows

        if decoding == "ann":
            results_list = self.resultsNN_phase
        elif decoding == "bayes":
            results_list = self.resultsBayes_phase
        else:
            raise ValueError("decoding does not exist")

        if masks is None:
            masks = [
                np.ones_like(
                    results_list[suffix]["time"][self.timeWindows.index(win_value)],
                    dtype=bool,
                )
                for win_value in winValues
            ]

        # First interpolate in time the signal so that we sample them well:
        itps_pred = [
            interp1d(
                results_list[suffix]["time"][self.timeWindows.index(win_value)],
                results_list[suffix]["linPred"][self.timeWindows.index(win_value)],
            )
            for win_value in winValues
        ]
        itpLast_pred = np.min(
            [
                np.max(
                    results_list[suffix]["time"][self.timeWindows.index(win_value)][
                        masks[i]
                    ]
                )
                for i, win_value in enumerate(winValues)
            ]
        )
        itpFirst_pred = np.max(
            [
                np.min(
                    results_list[suffix]["time"][self.timeWindows.index(win_value)][
                        masks[i]
                    ]
                )
                for i, win_value in enumerate(winValues)
            ]
        )
        x_pred = np.arange(itpFirst_pred, itpLast_pred, 0.003)
        discrete_linearPos_pred = [itp(x_pred) for itp in itps_pred]

        spectrums_pred = [fft(dlp) for dlp in discrete_linearPos_pred]
        xf_pred = fftfreq(x_pred.shape[0], 0.003)

        itps_TruePos = [
            interp1d(
                results_list[suffix]["time"][self.timeWindows.index(win_value)],
                results_list[suffix]["linTruePos"][self.timeWindows.index(win_value)],
            )
            for win_value in winValues
        ]
        itpLast_TruePos = np.min(
            [
                np.max(
                    results_list[suffix]["time"][self.timeWindows.index(win_value)][
                        masks[i]
                    ]
                )
                for i, win_value in enumerate(winValues)
            ]
        )
        itpFirst_TruePos = np.max(
            [
                np.min(
                    results_list[suffix]["time"][self.timeWindows.index(win_value)][
                        masks[i]
                    ]
                )
                for i, win_value in enumerate(winValues)
            ]
        )
        x_TruePos = np.arange(itpFirst_TruePos, itpLast_TruePos, 0.003)
        discrete_linearPos_TruePos = [itp(x_TruePos) for itp in itps_TruePos]

        spectrums_TruePos = [fft(dlp) for dlp in discrete_linearPos_TruePos]
        xf_TruePos = fftfreq(x_TruePos.shape[0], 0.003)
        fig, ax = plt.subplots()
        [
            ax.plot(
                xf_TruePos[:5000],
                2.0 / (x_TruePos.shape[0]) * np.abs(spectrums_TruePos[i][0:5000]),
                label=f"true values, {win_value} ms",
            )
            for i, win_value in enumerate(winValues)
        ]
        [
            ax.plot(
                xf_pred[:5000],
                2.0 / (x_pred.shape[0]) * np.abs(spectrums_pred[i][0:5000]),
                label=f"predicted values, {win_value} ms",
            )
            for i, win_value in enumerate(winValues)
        ]
        ax.set_xlabel("frequency, Hz")
        ax.set_ylabel("Fourier Power")
        fig.legend()
        fig.suptitle(
            f"Fourier Transform of Linear Position (both predicted and true) for {decoding} decoder."
        )
        plt.show(block=block)
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"fft_linearPos_{decoding}_decoder{suffix}.png",
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"fft_linearPos_{decoding}_decoder{suffix}.svg",
            )
        )
        plt.close()

    def correlate_predLoss_and_bayesProba(
        self,
        speed="all",
        suffix=None,
        phase=None,
        mask=None,
        use_mask=False,
        block=True,
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
                    np.ones(self.resultsNN_phase[suffix]["time"][i].shape).astype(bool)
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

        if speed == "fast":
            mask = habMaskFast
        elif speed == "slow":
            mask = habMaskSlow
        else:
            mask = habMask

        # Data
        predLoss = [
            -self.resultsNN_phase[suffix]["predLoss"][i][mask[i]]
            for i in range(len(self.timeWindows))
        ]  # we take the negative st high predLoss = higher confidence.

        bayesProba = [
            self.resultsBayes_phase[suffix]["predLoss"][i][mask[i]]
            for i in range(len(self.timeWindows))
        ]
        # normalize both probas between 0 and 1 to be "confidence values"
        predLoss = [
            np.divide(
                np.subtract(pl, np.min(pl)),
                np.subtract(np.max(pl), np.min(pl)),
            )
            for pl in predLoss
        ]

        # Collect correlations per window
        correlations = []
        for i in range(len(self.timeWindows)):
            r, _ = stats.pearsonr(predLoss[i], bayesProba[i])
            correlations.append(r)

        # Prepare figure
        fig, axes = plt.subplots(
            np.floor(len(self.timeWindows) / 2).astype(int),
            2,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if len(self.timeWindows) == 1:
            axes = [axes]

        correlations = []
        p_values = []

        ax = axes.flatten()

        for iw in range(len(self.timeWindows)):
            x = predLoss[iw]
            y = bayesProba[iw]

            # Scatter plot
            ax[iw].scatter(x, y, alpha=0.7, label=f"{self.timeWindows[iw]} ms")

            # Correlation
            r, p = stats.pearsonr(x, y)
            correlations.append(r)
            p_values.append(p)

            # Fit line
            slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
            ax[iw].plot(
                x, intercept + slope * x, color="red", label=f"r={r:.2f}, p={p:.3f}"
            )

            # Labels
            ax[iw].set_title(f"{self.timeWindows[iw]} ms")
            ax[iw].set_xlabel("predLoss")
            ax[iw].set_ylabel("bayesProba")
            ax[iw].legend()

        fig.suptitle(
            f"Correlation between predicted loss and Bayesian probability for phase {suffix.strip('_')}",
            fontsize="xx-large",
        )
        plt.tight_layout()
        plt.show(block=block)
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"predLoss_bayesProba_correlation{suffix}.png",
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"predLoss_bayesProba_correlation{suffix}.svg",
            )
        )
        plt.close(fig)

    def bayesian_neurons_summary(self, block=True, **kwargs):
        """
        Summary of the Bayesian neurons:
        - Identify interesting neurons based on mutual Information
        - Explore the ordered data
        - Visualize population place fields and linear tuning curves
        - Show position coverage in training data
        - Display quality metrics of the neurons
        - Show the best linear tuning curves
        - Display first and last ordered place fields

        self.trainerBayes must have been trained before calling this method.
        """
        if getattr(self, "bayesMatrices", None) is None:
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

        ordered_mi = np.array(
            [mi for tetrode_mi in self.bayesMatrices["mutualInfo"] for mi in tetrode_mi]
        )[self.trainerBayes.linearPosArgSort]

        # 1. Identify interesting neurons
        high_quality = self.trainerBayes.linearPosArgSort[
            ordered_mi > np.percentile(ordered_mi, 80)
        ]
        high_quality_mask = ordered_mi > np.percentile(ordered_mi, 80)
        print(f"High-quality place cells: {len(high_quality)} neurons")

        # 2. Explore the ordered data
        print(f"Found {len(self.trainerBayes.linearPosArgSort)} neurons")
        print(
            f"Position range: {self.trainerBayes.linearPreferredPos.min():.2f} - {self.trainerBayes.linearPreferredPos.max():.2f}"
        )

        # 3. Visualize population
        fig = plt.figure()

        # Population place fields
        plt.subplot(2, 3, 1)
        plt.imshow(
            self.trainerBayes.orderedPlaceFields[1].T, aspect="auto", origin="lower"
        )
        plt.xticks([])
        plt.yticks([])
        plt.title("First Ordered Place Field")

        # Linear tuning curves (if computed)
        if hasattr(self.trainerBayes, "orderedLinearPlaceFields"):
            plt.subplot(2, 3, 2)
            norm_fields = self.trainerBayes.orderedLinearPlaceFields / np.maximum(
                np.mean(
                    self.trainerBayes.orderedLinearPlaceFields, axis=1, keepdims=True
                ),
                1e-8,
            )
            plt.imshow(norm_fields, aspect="auto", origin="lower")
            linearBins = np.linspace(0, 1, norm_fields.shape[1])
            plt.xticks(
                ticks=np.linspace(0, norm_fields.shape[1], norm_fields.shape[1])[::20],
                labels=np.round(linearBins[::20], 2),
            )
            plt.xlabel("Linear Position")
            plt.ylabel("Neuron Index")
            plt.colorbar(label="Deviation from unit Mean Firing Rate")
            plt.title("Linear Tuning Curves")

        # Position coverage
        plt.subplot(2, 3, 3)
        plt.hist(self.trainerBayes.linearPreferredPos, bins=20, alpha=0.7)
        plt.xlabel("Linear Position")
        plt.title("Pos Coverage in Training Data")

        # Quality metrics
        plt.subplot(2, 3, 4)
        colors = np.array(["blue"] * len(ordered_mi))
        colors[high_quality_mask] = "red"
        plt.plot(ordered_mi, "o-", alpha=0.6, zorder=0)

        plt.scatter(
            np.arange(len(ordered_mi)), ordered_mi, c=colors, alpha=0.6, zorder=1
        )
        plt.title("Mutual Information (ordered)")
        plt.xlabel("Neuron Index")
        plt.ylabel("Mutual Information")

        # Linear tuning curves (if computed)
        if hasattr(self.trainerBayes, "orderedLinearPlaceFields"):
            plt.subplot(2, 3, 5)
            plt.imshow(norm_fields[high_quality_mask], aspect="auto", origin="lower")
            linearBins = np.linspace(0, 1, norm_fields.shape[1])
            neuron_idx = np.arange(norm_fields.shape[0])[high_quality_mask]
            plt.xticks(
                ticks=np.linspace(0, norm_fields.shape[1], norm_fields.shape[1])[::20],
                labels=np.round(linearBins[::20], 2),
            )
            plt.yticks(ticks=np.arange(len(neuron_idx)), labels=neuron_idx)
            plt.xlabel("Linear Position")
            plt.ylabel("Neuron Index")
            plt.colorbar(label="Deviation from unit Mean FR\n" + "- Only HQ cells")
            plt.title("Best Linear Tuning Curves")

        plt.tight_layout()
        plt.show()

        # Population place fields
        plt.subplot(2, 3, 6)
        plt.imshow(
            self.trainerBayes.orderedPlaceFields[-1].T, aspect="auto", origin="lower"
        )
        plt.xticks([])
        plt.yticks([])
        plt.title("Last Ordered Place Field")

        plt.tight_layout()
        plt.show(block=block)
        fig.savefig(
            os.path.join(
                self.folderFigures, f"bayesian_neurons_summary{self.suffix}.png"
            )
        )
        fig.savefig(
            os.path.join(
                self.folderFigures, f"bayesian_neurons_summary{self.suffix}.svg"
            )
        )
        plt.close(fig)


if __name__ == "__main__":
    import warnings

    import tqdm

    from neuroencoders.importData.rawdata_parser import get_behavior
    from neuroencoders.utils.MOBS_Functions import (
        Mouse_Results,
        path_for_experiments_df,
    )

    jsonPath = None
    windowSizeMS = [108, 252]
    mode = "ann"
    target = "pos"
    phase = "pre"
    nEpochs = 200
    mouse = "1199"
    manipe = "PAG"
    nameExp = "current_LogLoss_Transformer_Dense_Transformer"

    Dir = path_for_experiments_df("Sub", nameExp)
    sample_results = Mouse_Results(
        Dir,
        mouse_name=mouse,
        manipe=manipe,
        target=target,
        nameExp=nameExp,
        nEpochs=nEpochs,
        phase=phase,
        deviceName="cpu",
        windows=windowSizeMS,
        isTransformer="LSTM" not in nameExp,
        denseweight=True,
        transform_w_log=True,
        which="both",
        isPredLoss=False,
    )
    sample_results.load_data(suffixes=["_pre", "_cond", "_training"])
    sample_results.load_bayes(suffixes=["_pre", "_cond", "_training"])
    suffix = f"_{phase}"

    ### Let us pursue on comparing NN and Bayesian:
    warnings.warn("Main process not fully implemented yet.")
    sample_results.compare_nn_bayes_with_random_pred(252)
    sample_results.fig_example_XY(252, block=True)
    sample_results.compare_nn_bayes(252, block=True)
    sample_results.error_right_left_arm(block=True)
