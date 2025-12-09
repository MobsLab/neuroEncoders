# Load libs
import os
import platform
import subprocess

import dill as pickle
import matplotlib as mplt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns
import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr, sem, zscore
from statsmodels.stats.proportion import proportions_ztest

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.resultAnalysis.print_results import overview_fig
from neuroencoders.simpleBayes.decode_bayes import Trainer as TrainerBayes
from neuroencoders.simpleBayes.decode_bayes import (
    extract_spike_counts,
    extract_spike_counts_from_matrix,
)
from neuroencoders.utils.PlaceField_dB import _run_place_field_analysis
from neuroencoders.utils.global_classes import (
    MAZE_COORDS,
    ZONEDEF,
    ZONELABELS,
    DataHelper,
    Project,
    is_in_zone,
)
from neuroencoders.utils.viz_params import (
    DELTA_COLOR_FORWARD,
    DELTA_COLOR_REVERSE,
    EC,
    MIDDLE_COLOR,
    PREDICTED_LINE_COLOR,
    SAFE_COLOR,
    SHOCK_COLOR,
    TRUE_LINE_COLOR,
    get_pvalue_stars,
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

        if "_training" in self.suffixes:
            self.suffixes.remove("_training")
            self.suffixes.insert(0, "_training")  # load training first if present

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
                if os.path.exists(
                    os.path.join(
                        self.projectPath.experimentPath,
                        "results",
                        str(ws),
                        f"linearTrue{suffix}.csv",
                    )
                ):
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
                else:
                    lPredPos.append(self.l_function(fPredPos[-1][:, :2])[1].flatten())
                    lTruePos.append(self.l_function(truePos[-1][:, :2])[1].flatten())
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
                if kwargs.get("extract_spikes_count", False) or kwargs.get(
                    "extract_spike_counts", False
                ):
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
                        warnings.warn(
                            f"No spikes_count file found for resultsNN_phase{suffix} and window {str(ws)}, skipping loading it.\n"
                            f"You should export it when testing the model. See `extract_spike_counts` argument from `trainerNN.test`"
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
                if kwargs.get("extract_spikes_count", False) or kwargs.get(
                    "extract_spike_counts", False
                ):
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
            if kwargs.get("extract_spikes_count", False) or kwargs.get(
                "extract_spike_counts", False
            ):
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
        if not hasattr(self.trainerBayes, "linearPreferredPos") and (
            kwargs.get("load_bayesMatrices", False)
            or self.trainerBayes.config.extra_kwargs.get("load_bayesMatrices", False)
        ):
            # load bayesMatrices if not already done
            # first, we mix kwargs with trainerBayes config extra_kwargs, with kwargs having priority
            combined_kwargs = {**self.trainerBayes.config.extra_kwargs, **kwargs}

            self.bayesMatrices = self.trainerBayes.train_order_by_pos(
                self.behaviorData,
                l_function=self.l_function,
                bayesMatrices=self.bayesMatrices
                if (
                    (isinstance(self.bayesMatrices, dict))
                    and ("Occupation" in self.bayesMatrices.keys())
                )
                else None,
                **combined_kwargs,
            )

        # quickly obtain bayesian decoding:
        if suffixes is None:
            self.suffixes = [self.suffix]
        else:
            self.suffixes = suffixes
        if not isinstance(self.suffixes, list):
            self.suffixes = [suffixes]
        if "_training" in self.suffixes:
            self.suffixes.remove("_training")
            self.suffixes.insert(0, "_training")  # load training first if present

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
                    if kwargs.get("extract_spikes_count", False) or kwargs.get(
                        "extract_spike_counts", False
                    ):
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
                if kwargs.get("extract_spikes_count", False) or kwargs.get(
                    "extract_spike_counts", False
                ):
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
            if kwargs.get("extract_spikes_count", False) or kwargs.get(
                "extract_spike_counts", False
            ):
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

    def summary_behavior(
        self,
        DataHelper: DataHelper,
        axs=None,
        show: bool = True,
        block: bool = False,
        save: bool = False,
        extended_zone: bool = False,
        **kwargs,
    ):
        num_sess = len(DataHelper.fullBehavior["Times"]["SessionEpochs"])

        # --- Figure Setup ---
        num_cols = num_sess
        if axs is None:
            fig, axs = plt.subplots(2, num_cols, figsize=(15, 6))
        else:
            axs = np.array(axs).reshape(2, -1)
            fig = axs[0, 0].get_figure()

        map_axs = axs[0, :].flatten()
        bar_axs = axs[1, :].flatten()

        # --- Data Structure to hold occupancy values for plotting and stars ---
        session_occupancy_data = []

        # Find positions when stims happened
        PosMat = DataHelper.fullBehavior["Times"]["PosMat"]
        stim_mask = PosMat[:, 3] == 1

        # Define custom colormap once
        cmap_custom = plt.cm.get_cmap("Reds")
        cmap_custom.set_under("white")

        # --- Loop 1: Data Calculation & Map Plotting (First Row) ---
        # first, sort session epochs by name key to have consistent order
        sort_map = ["hab", "pre", "cond", "post", "extinction"]
        DataHelper.fullBehavior["Times"]["SessionEpochs"] = dict(
            sorted(
                DataHelper.fullBehavior["Times"]["SessionEpochs"].items(),
                key=lambda item: sort_map.index(item[0])
                if item[0] in sort_map
                else len(sort_map),
            )
        )
        after_cond = False
        for i, (sess_name, sess_time) in enumerate(
            DataHelper.fullBehavior["Times"]["SessionEpochs"].items()
        ):
            if sess_name.lower() == "cond":
                after_cond = True
            # 1. Data Filtering
            mask = inEpochsMask(
                DataHelper.fullBehavior["positionTime"], sess_time
            ).flatten()
            pos = DataHelper.fullBehavior["Positions"][mask, :2]
            nan_mask = ~np.any(np.isnan(pos), axis=1)
            pos = pos[nan_mask]
            total_time_points = len(pos)

            # 2. Occupancy Calculation and Z-Test for Proportions
            shock_mask = is_in_zone(pos, ZONEDEF[ZONELABELS.index("Shock")])
            safe_mask = is_in_zone(pos, ZONEDEF[ZONELABELS.index("Safe")])
            if extended_zone:
                shock_mask = shock_mask | is_in_zone(
                    pos, ZONEDEF[ZONELABELS.index("ShockCenter")]
                )
                safe_mask = safe_mask | is_in_zone(
                    pos, ZONEDEF[ZONELABELS.index("SafeCenter")]
                )

            shock_count = np.sum(shock_mask)
            safe_count = np.sum(safe_mask)

            shock_occupancy = (
                shock_count / total_time_points if total_time_points > 0 else 0
            )
            safe_occupancy = (
                safe_count / total_time_points if total_time_points > 0 else 0
            )

            # Perform Two-Sample Z-Test for Proportions (Shock count vs Safe count)
            # This is the statistically correct way to compare two proportions based on counts.
            if total_time_points > 0:
                counts = np.array([safe_count, shock_count])
                # N must be total time points for both groups
                nobs = np.array([total_time_points, total_time_points])

                # Check if there is enough variance to run the test
                if np.sum(nobs) > 0 and np.all(counts > 0):
                    z_stat, p_value = proportions_ztest(
                        counts, nobs=nobs, alternative="two-sided"
                    )
                    p_stars = get_pvalue_stars(p_value)
                else:
                    p_stars = None  # Not enough data/variance for test
            else:
                p_stars = None  # No data in session

            # Store data
            session_occupancy_data.append(
                {
                    "Session": sess_name,
                    "Shock Occupancy": shock_occupancy,
                    "Safe Occupancy": safe_occupancy,
                    "Stars": p_stars,  # Store star string
                }
            )

            # 3. Map Plotting (First Row)
            map_ax = map_axs[i]
            H_true, xedges, yedges = np.histogram2d(
                pos[:, 0], pos[:, 1], bins=40, range=[[0, 1], [0, 1]]
            )
            occupancy_map = gaussian_filter(H_true.T, sigma=2)

            map_ax.plot(
                pos[:, 0], pos[:, 1], c="xkcd:dark grey", alpha=0.4, zorder=1
            )  # Traces
            map_ax.imshow(
                occupancy_map,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=cmap_custom,
                vmin=1,
                zorder=0,
            )
            map_ax.plot(
                MAZE_COORDS[:, 0], MAZE_COORDS[:, 1], color="black", lw=2, zorder=2
            )

            if after_cond:
                pos_stim = DataHelper.fullBehavior["Positions"][stim_mask, :2]
                map_ax.scatter(
                    pos_stim[:, 0],
                    pos_stim[:, 1],
                    c=SHOCK_COLOR,
                    marker="*",
                    s=20,
                    alpha=0.6,
                    label="Stimulations",
                    zorder=3,
                )

            map_ax.set_title(f"{sess_name}")
            if sess_name.lower() == "cond":
                map_ax.set_title(f"{sess_name} (n_stims = {np.sum(stim_mask)})")
            map_ax.set_aspect("equal", adjustable="box")
            map_ax.set_xlim(0, 1)
            map_ax.set_ylim(0, 1)
            map_ax.set_xticks([])
            map_ax.set_yticks([])

        # --- Loop 2: Bar Plotting (Second Row) ---
        # Find the maximum occupancy value across all sessions/zones for consistent Y-axis limits
        max_occupancy = (
            max(
                [d["Shock Occupancy"] for d in session_occupancy_data]
                + [d["Safe Occupancy"] for d in session_occupancy_data]
            )
            * 1.25  # Increased buffer for stars
        )

        for i, data in enumerate(session_occupancy_data):
            bar_ax = bar_axs[i]

            # Data points for this session
            zones = ["Shock", "Safe"]
            occupancy_values = [data["Shock Occupancy"], data["Safe Occupancy"]]

            # Assuming SHOCK_COLOR and SAFE_COLOR are defined globally
            colors = [SHOCK_COLOR, SAFE_COLOR]

            # Plot the bar chart
            bar_ax.bar(zones, occupancy_values, color=colors)
            bar_ax.axhline(
                0.215
                if not extended_zone
                else 0.465,  # expected occupancy if exploration is uniform
                linestyle="--",
                color="gray",
                lw=1.6,
            )  # Add reference line at y=0.215

            # Add value labels on top of bars
            for j, val in enumerate(occupancy_values):
                bar_ax.text(
                    j,
                    val + 0.005 * max_occupancy,
                    f"{val:.2f}",
                    ha="center",
                    fontsize=8,
                )

            # --- ADD STATISTICAL STAR (Simulating statannotations output) ---
            if data["Stars"] is not None:
                # Determine where to place the star (above the highest bar)
                star_y_pos = max(occupancy_values) + 0.05 * max_occupancy
                # Use a line to connect the two bars, similar to statannotations
                bar_ax.plot([0, 1], [star_y_pos, star_y_pos], color="k", lw=0.8)
                bar_ax.text(
                    0.5,  # Center position between the two bars
                    star_y_pos + 0.005 * max_occupancy,  # Slightly above the line
                    data["Stars"],
                    ha="center",
                    color="k",
                )

            bar_ax.set_title(f"Occ: {data['Session']}")
            bar_ax.set_ylim(0, max_occupancy)  # Constant Y-limit for comparison
            bar_ax.set_yticks(np.linspace(0, max_occupancy, 3))  # Set few ticks
            bar_ax.tick_params(axis="x", rotation=0)

            # Only label the Y-axis on the first bar plot for clarity
            if i == 0:
                bar_ax.set_ylabel("Occupancy Fraction")
            else:
                bar_ax.set_yticklabels([])  # Hide Y-labels on subsequent plots

        if show:
            # Final layout adjustments
            fig.suptitle(
                "Behavioral Analysis Summary: Occupancy Maps and Zone Occupancy per Session",
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show(block=block)
        if save:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(
                os.path.join(self.folderFigures, "summary_behavior.png"),
                dpi=300,
            )
            fig.savefig(
                os.path.join(self.folderFigures, "summary_behavior.svg"),
            )

    def error_map(
        self,
        timeWindow: int,
        phase_list=None,
        suffix_list=None,
        axs=None,
        show: bool = True,
        block: bool = False,
        save: bool = False,
        error_type: str = "lin",
        **kwargs,
    ):
        if phase_list is not None:
            suffix_list = [f"_{ph}" for ph in phase_list]
        if suffix_list is None:
            suffix_list = self.suffixes
        if not isinstance(suffix_list, list):
            suffix_list = [suffix_list]
        try:
            winIdx = self.timeWindows.index(timeWindow)
        except ValueError:
            raise ValueError(
                f"Time window {timeWindow}ms not found in self.timeWindows: {self.timeWindows}"
            )

        if axs is None:
            fig, axs = plt.subplots(
                1,
                len(suffix_list),
            )
            if len(suffix_list) == 1:
                axs = np.array([[axs]])
        else:
            axs = np.array(axs).flatten()
            fig = axs[0].get_figure()
        axs = axs.flatten()

        for i, suffix in enumerate(suffix_list):
            lin_true = self.resultsNN_phase[suffix]["linTruePos"][winIdx]
            lin_pred = self.resultsNN_phase[suffix]["linPred"][winIdx]
            x_true = self.resultsNN_phase[suffix]["truePos"][winIdx][:, 0]
            y_true = self.resultsNN_phase[suffix]["truePos"][winIdx][:, 1]
            if error_type == "lin":
                error = np.abs(lin_true - lin_pred)
            elif error_type in {"xy", "euclidean"}:
                error = np.linalg.norm(
                    self.resultsNN_phase[suffix]["fullPred"][winIdx][:, :2]
                    - self.resultsNN_phase[suffix]["truePos"][winIdx][:, :2],
                    axis=1,
                )
            else:
                raise ValueError(
                    f"Unknown error_type {error_type}, should be 'lin' or 'xy'/'euclidean'."
                )
            speed_mask = self.resultsNN_phase[suffix]["speedMask"][winIdx].astype(bool)
            bins = 40
            mean_error_matrix, xedges, yedges, binnumber = binned_statistic_2d(
                x_true[speed_mask],
                y_true[speed_mask],
                error[speed_mask],
                statistic="mean",
                bins=bins,
                range=[[0, 1], [0, 1]],  # Set your min/max range here
            )
            # small smooth interpolation
            mean_error_matrix = gaussian_filter(mean_error_matrix, 1e-1)

            cmap_name = kwargs.get("cmap", "coolwarm")
            cmap = plt.get_cmap(cmap_name)

            axs[i].imshow(
                mean_error_matrix.T,
                origin="lower",
                cmap=cmap,
                extent=[0, 1, 0, 1],
                # interpolation="nearest",
            )
            axs[i].plot(MAZE_COORDS[:, 0], MAZE_COORDS[:, 1], color="black", lw=2)
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            axs[i].set_aspect("equal", adjustable="box")
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            for spine in axs[i].spines.values():
                spine.set_visible(False)
            axs[i].set_title(f"{suffix.strip('_')}")
        if show:
            fig.suptitle(
                f"Error map ({error_type}) for {timeWindow}ms window",
            )
            plt.tight_layout()
            plt.show(block=block)
        if save:
            plt.tight_layout()
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    f"error_map_{error_type}_{timeWindow}ms.png",
                ),
                dpi=300,
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    f"error_map_{error_type}_{timeWindow}ms.svg",
                )
            )

    def fig_summary_id_card(
        self,
        timeWindow: int,
        DataHelper: DataHelper,
        suffix: str = None,
        save: bool = True,
        dimOutput: int = 1,
        **kwargs,
    ):
        """
        Summary figure saved as a multipage PDF.
        Page 1: Behavior & Error Maps
        Page 2+: Trajectories (2 suffixes per page)
        """
        idWindow = self.timeWindows.index(timeWindow)

        # 1. Setup PDF Path
        filename = f"summary_id_card_{timeWindow}ms.pdf"
        save_path = os.path.join(self.folderFigures, filename)

        print(f"Generating PDF at: {save_path}")

        with PdfPages(save_path) as pdf:
            # =========================================================
            # PAGE 1: Summary Behavior + Error Maps
            # =========================================================
            fig1 = plt.figure(
                figsize=(8.27, 11.69)
            )  # Landscape A4-ish, or use (8.27, 11.69) for Portrait
            # Let's use Portrait for vertical stacking
            fig1.set_size_inches(10, 14)

            gs = gridspec.GridSpec(12, 3, figure=fig1)

            # --- Top: Summary Behavior (Rows 0-1) ---
            nrows_beh = 2
            ncols_beh = len(DataHelper.fullBehavior["Times"]["SessionEpochs"])
            gs_summary = gs[0 : nrows_beh + 1, :].subgridspec(nrows_beh, ncols_beh)

            target_axs = [
                fig1.add_subplot(gs_summary[i, j])
                for i in range(nrows_beh)
                for j in range(ncols_beh)
            ]

            self.summary_behavior(
                DataHelper,
                axs=target_axs,
                show=False,
                block=False,
                save=False,
                extended_zone=kwargs.get("extended_zone", False),
            )

            # On row 3, add a text box for error map title
            ax_text = fig1.add_subplot(gs[3, :])
            ax_text.axis("off")  # Hide the axis
            ax_text.text(
                0.5,
                0.5,
                "Error Maps on movement epochs",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
                fontweight="bold",
            )

            # --- Middle: Error Maps (Rows 4-5) ---
            # Leaving row 3 empty as spacing
            start_row_map = 4
            nrows_map = 2
            ncols_map = len(self.suffixes)
            gs_error_maps = gs[
                start_row_map : start_row_map + nrows_map + 1, :
            ].subgridspec(nrows_map, ncols_map)

            target_axs = [
                fig1.add_subplot(gs_error_maps[0, j]) for j in range(ncols_map)
            ]

            self.error_map(
                timeWindow,
                suffix_list=self.suffixes,
                axs=target_axs,
                show=False,
                block=False,
                save=False,
                error_type=kwargs.get("error_type", "lin"),
            )

            fig1.suptitle(
                f"Summary: {os.path.basename(self.projectPath.experimentPath)} ({timeWindow}ms)",
                fontsize=14,
            )
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # =========================================================
            # PAGES 2+: Decoding Performance (Trajectories)
            # =========================================================
            # We want ~4 rows per page.
            # Since 1 suffix = 2 rows (dim 0 and dim 1), we can fit 2 suffixes per page.

            suffixes_per_page = 3 if dimOutput == 1 else 2
            with_hist_distribution = kwargs.get("with_hist_distribution", True)

            # Chunk the suffixes list
            for i in range(0, len(self.suffixes), suffixes_per_page):
                page_suffixes = self.suffixes[i : i + suffixes_per_page]

                fig_page = plt.figure(figsize=(14, 11))  # Paysage

                # Each suffix needs 2 rows.
                # Total rows needed = len(page_suffixes) * 2.
                # We allocate a generic GridSpec with ample rows
                total_page_rows = (
                    3 if dimOutput == 1 else 4
                )  # Fixed to 4 rows (2 suffixes) for consistency
                gs_page = gridspec.GridSpec(
                    total_page_rows, 1, figure=fig_page, hspace=0.4
                )

                current_row_idx = 0

                for suffix in page_suffixes:
                    # Create subgridspec for this specific suffix (2 rows)
                    # Columns: 5 if hist, 4 if not
                    ncols = 6 if with_hist_distribution else 5

                    # Select the 2 rows for this suffix
                    gs_suffix = gs_page[
                        current_row_idx : current_row_idx + dimOutput, 0
                    ].subgridspec(2, ncols)

                    axs_list = []
                    first_ax_ref = None

                    for r in range(dimOutput):  # 2 Dimensions
                        # Main Trajectory Plot
                        if r == 0:
                            ax_main = fig_page.add_subplot(gs_suffix[r, 0:4])
                            first_ax_ref = ax_main
                        else:
                            ax_main = fig_page.add_subplot(
                                gs_suffix[r, 0:4], sharex=first_ax_ref
                            )
                        axs_list.append(ax_main)

                        # Histogram Plot
                        if with_hist_distribution:
                            ax_dist = fig_page.add_subplot(
                                gs_suffix[r, 4], sharey=ax_main
                            )
                            ax_dist.tick_params(axis="y", left=False, labelleft=False)
                            axs_list.append(ax_dist)

                    # Plot Data
                    posIndex = self.resultsNN_phase[suffix]["posIndex"][idWindow]
                    timeStepsPred = self.resultsNN_phase[suffix]["time"][idWindow]
                    speedMask = self.resultsNN_phase[suffix]["speedMask"][idWindow]
                    linpos = self.resultsNN_phase[suffix]["linTruePos"][idWindow]
                    lininferring = self.resultsNN_phase[suffix]["linPred"][idWindow]

                    if dimOutput == 1:
                        pos = self.resultsNN_phase[suffix]["linTruePos"][idWindow]
                        inferring = self.resultsNN_phase[suffix]["linPred"][idWindow]
                        training_data = self.resultsNN_phase["_training"]["linTruePos"][
                            idWindow
                        ]
                    elif dimOutput == 2:
                        pos = self.resultsNN_phase[suffix]["truePos"][idWindow][:, :2]
                        inferring = self.resultsNN_phase[suffix]["fullPred"][idWindow][
                            :, :2
                        ]
                        training_data = self.resultsNN_phase["_training"]["truePos"][
                            idWindow
                        ][:, :2]
                    else:
                        raise ValueError("dimOutput must be 1 or 2.")
                    overview_fig(
                        pos=pos,
                        inferring=inferring,
                        selection=np.ones_like(
                            self.resultsNN_phase[suffix]["speedMask"][idWindow],
                            dtype=bool,
                        ),
                        posIndex=posIndex,
                        timeStepsPred=timeStepsPred,
                        speedMask=speedMask,
                        useSpeedMask=True,
                        concat_epochs=True,
                        dimOutput=dimOutput,
                        show=False,
                        save=False,
                        close=False,
                        training_data=training_data,
                        join_points=False,
                        axs=np.array(axs_list),
                        fig=fig_page,
                    )
                    axs_list[0].set_title(
                        f"Phase: {suffix.strip('_')}",
                    )
                    freeze = inEpochsMask(
                        self.behaviorData["positionTime"][
                            self.resultsNN_phase[suffix]["posIndex"][idWindow]
                        ],
                        self.behaviorData["Times"]["FreezeEpoch"],
                    )
                    if dimOutput == 2:
                        error = np.linalg.norm(inferring - pos, axis=1)
                    else:
                        error = np.abs(inferring - pos)
                    mean_error = np.mean(error[speedMask])
                    median_error = np.median(error[speedMask])
                    random_pred = np.random.permutation(pos)
                    chance_medianerror = np.median(np.abs(random_pred - pos)[speedMask])
                    chance_meanerror = np.mean(np.abs(random_pred - pos)[speedMask])
                    text = (
                        f"Mean Error : {mean_error:.3f} (random {chance_meanerror:.3f})\n"
                        f"Median Error : {median_error:.3f} (random {chance_medianerror:.3f})\n"
                        f"Freeze Fraction: {np.mean(freeze):.3f}"
                    )
                    axs_list[0].text(
                        0.75,
                        -0.3,
                        text,
                        horizontalalignment="right",
                        verticalalignment="top",
                        transform=axs_list[0].transAxes,
                        fontsize="small",
                    )

                    last_col_ax = fig_page.add_subplot(
                        gs_suffix[:, -1]
                    )  # all columns, first row
                    last_col_ax.axis("off")  # Hide the axis
                    self._plot_single_error_matrix(
                        linpos[speedMask],
                        lininferring[speedMask],
                        last_col_ax,
                    )

                    current_row_idx += dimOutput  # Move down 2 rows for next suffix

                plt.tight_layout()
                pdf.savefig(fig_page)
                plt.close(fig_page)
            # =========================================================
            # PAGES after+: Link with bayesian decoder and spike sorting
            # =========================================================
            bayesian_page = plt.figure(figsize=(11.69, 8.27))  # Portrait A4
            bayesian_page.set_size_inches(14, 10)
            # get (2,3) gridspec axes
            gs_bayes = gridspec.GridSpec(2, 3, figure=bayesian_page)
            bayesian_axs = [
                bayesian_page.add_subplot(gs_bayes[i, j])
                for i in range(2)
                for j in range(3)
            ]
            self.bayesian_neurons_summary(
                fig=bayesian_page, axs=bayesian_axs, show=False, block=False, save=False
            )
            plt.tight_layout()
            pdf.savefig(bayesian_page)
            plt.close(bayesian_page)

        print("PDF generation complete.")

        # =========================================================
        # OPEN THE PDF
        # =========================================================
        if save:  # Using the 'save' flag to decide if we open it, or you can add an 'open_pdf' arg
            try:
                if platform.system() == "Darwin":  # macOS
                    subprocess.call(("open", save_path))
                elif platform.system() == "Windows":  # Windows
                    os.startfile(save_path)
                else:  # linux variants
                    subprocess.call(("xdg-open", save_path))
            except Exception as e:
                print(f"Could not open PDF automatically: {e}")

    def fig_summary_id_card_olddd(
        self,
        timeWindow: int,
        DataHelper: DataHelper,
        suffix: str = None,
        show: bool = True,
        block: bool = True,
        save: bool = True,
        **kwargs,
    ):
        """
        Summary figure of decoding performance and behavioral metrics for a given time window.
        """

        # training + test together
        idWindow = self.timeWindows.index(timeWindow)

        x_true = np.concatenate(
            [
                self.resultsNN_phase[suffix]["truePos"][idWindow][:, 0]
                for suffix in self.suffixes
            ]
        )
        y_true = np.concatenate(
            [
                self.resultsNN_phase[suffix]["truePos"][idWindow][:, 1]
                for suffix in self.suffixes
            ]
        )
        x_pred = np.concatenate(
            [
                self.resultsNN_phase[suffix]["fullPred"][idWindow][:, 0]
                for suffix in self.suffixes
            ]
        )
        y_pred = np.concatenate(
            [
                self.resultsNN_phase[suffix]["fullPred"][idWindow][:, 1]
                for suffix in self.suffixes
            ]
        )
        lin_true = np.concatenate(
            [
                self.resultsNN_phase[suffix]["linTruePos"][idWindow]
                for suffix in self.suffixes
            ]
        )
        lin_pred = np.concatenate(
            [
                self.resultsNN_phase[suffix]["linPred"][idWindow]
                for suffix in self.suffixes
            ]
        )
        time = np.concatenate(
            [self.resultsNN_phase[suffix]["time"][idWindow] for suffix in self.suffixes]
        )
        posIndex = [
            self.resultsNN_phase[suffix]["posIndex"][idWindow]
            for suffix in self.suffixes
        ]
        # extract spikes count per group if not already loaded
        if (
            "spikes_count" not in self.resultsNN_phase["_training"]
            or self.resultsNN_phase["_training"]["spikes_count"][idWindow] is None
        ):
            self.load_data(
                suffixes=self.suffixes,
                extract_spikes_count=True,
                load_pickle=True,
            )

        spikes_count = np.concatenate(
            [
                self.resultsNN_phase[suffix]["spikes_count"][idWindow]
                for suffix in self.suffixes
            ]
        )

        # # train mask not defined based on datahelper (can change if epochs were modified after loading data)
        test_mask = (time >= min(self.resultsNN_phase[suffix]["time"][idWindow])) & (
            time <= max(self.resultsNN_phase[suffix]["time"][idWindow])
        )
        train_mask = ~test_mask
        speed_mask = np.concatenate(
            [
                self.resultsNN_phase["_training"]["speedMask"][idWindow],
                self.resultsNN_phase[suffix]["speedMask"][idWindow],
            ]
        ).astype(bool)
        sampling_rate = 1 / max(0.036, int(timeWindow) / 4000)

        # add chance level
        random_pred = np.random.permutation(lin_true)
        chance_mae = np.mean(np.abs(random_pred[train_mask] - lin_true[train_mask]))
        chance_mae_speed = np.mean(
            np.abs(
                random_pred[train_mask & speed_mask] - lin_true[train_mask & speed_mask]
            )
        )

        # ----------------------------------------------------------------------
        # 1. BEHAVIORAL MEASURES
        # ----------------------------------------------------------------------

        # --- Speed ---
        speed_train = DataHelper.fullBehavior["Speed"][posIndex[0]].flatten()
        speed_test = DataHelper.fullBehavior["Speed"][posIndex[1]].flatten()
        speed = np.concatenate([speed_train, speed_test])

        # --- Freezing ---
        freeze_train = inEpochsMask(
            DataHelper.fullBehavior["positionTime"][posIndex[0]],
            DataHelper.fullBehavior["Times"]["FreezeEpoch"],
        )
        freeze_fraction_train = np.mean(freeze_train)
        freeze_duration_train = np.sum(
            DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 1]
            - DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 0]
        )  # total freeze duration in seconds
        freeze_duration_train /= sampling_rate  # convert to seconds

        freeze_test = inEpochsMask(
            DataHelper.fullBehavior["positionTime"][posIndex[1]],
            DataHelper.fullBehavior["Times"]["FreezeEpoch"],
        )
        freeze_fraction_test = np.mean(freeze_test)
        freeze_duration_test = np.sum(
            DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 1]
            - DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 0]
        )  # total freeze duration in seconds
        freeze_duration_test /= sampling_rate  # convert to seconds
        freeze_duration_total = freeze_duration_train + freeze_duration_test

        # ----------------------------------------------------------------------
        # 1b. Reordering in time and removing "false" train epochs (ie test epochs)
        # ----------------------------------------------------------------------
        time_order = np.argsort(np.concatenate([posIndex[0], posIndex[1]]).flatten())
        x_true = x_true[time_order]
        y_true = y_true[time_order]
        x_pred = x_pred[time_order]
        y_pred = y_pred[time_order]
        lin_true = lin_true[time_order]
        lin_pred = lin_pred[time_order]
        time = time[time_order]
        train_mask = train_mask[time_order]
        test_mask = test_mask[time_order]
        speed_mask = speed_mask[time_order]
        speed = speed[time_order]

        # --- Occupancy heatmap (true behavior) ---
        # look only for test set to avoid train bias
        bins = 40
        H_true, xedges, yedges = np.histogram2d(
            x_true[test_mask & speed_mask], y_true[test_mask & speed_mask], bins=bins
        )
        H_true_smooth = gaussian_filter(H_true, sigma=1)
        H_true_norm = H_true_smooth / np.sum(H_true_smooth)

        H_true_train, _, _ = np.histogram2d(
            x_true[train_mask & speed_mask], y_true[train_mask & speed_mask], bins=bins
        )
        H_true_train_smooth = gaussian_filter(H_true_train, sigma=1)
        H_true_train_norm = H_true_train_smooth / np.sum(H_true_train_smooth)

        # ----------------------------------------------------------------------
        # 2. DECODING METRICS
        # ----------------------------------------------------------------------

        # --- Decoding error ---
        # either euclidean
        error = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)
        # or linearized

        error = np.abs(lin_pred - lin_true)
        mae_train = np.mean(error[speed_mask & train_mask])
        mae_train_raw = np.mean(error[train_mask])
        mae_test = np.mean(error[speed_mask & test_mask])
        mae_test_raw = np.mean(error[test_mask])

        # Rolling MAE (5 s window)
        win = int(5 * sampling_rate)
        rolling_mae = np.convolve(error, np.ones(win) / win, mode="same")

        # --- Spatial error heatmap ---
        H_err_sum = np.zeros_like(H_true, dtype=float)
        H_count = np.zeros_like(H_true, dtype=float)

        # assign each sample to a bin
        x_bin = np.digitize(x_true[test_mask & speed_mask], xedges) - 1
        y_bin = np.digitize(y_true[test_mask & speed_mask], yedges) - 1

        valid = (x_bin >= 0) & (x_bin < bins) & (y_bin >= 0) & (y_bin < bins)
        for xb, yb, e in zip(
            x_bin[valid], y_bin[valid], error[test_mask & speed_mask][valid]
        ):
            H_err_sum[xb, yb] += e
            H_count[xb, yb] += 1

        H_err_mean = np.divide(H_err_sum, H_count, where=H_count > 0)
        H_err_smooth = gaussian_filter(H_err_mean, sigma=1)

        # --- Occupancy map from decoded position ---
        H_decoded, _, _ = np.histogram2d(
            x_pred[test_mask & speed_mask],
            y_pred[test_mask & speed_mask],
            bins=[xedges, yedges],
        )
        H_decoded_smooth = gaussian_filter(H_decoded, sigma=1)
        H_decoded_norm = H_decoded_smooth / np.sum(H_decoded_smooth)

        H_decoded_train, _, _ = np.histogram2d(
            x_pred[train_mask & speed_mask],
            y_pred[train_mask & speed_mask],
            bins=[xedges, yedges],
        )
        H_decoded_train_smooth = gaussian_filter(H_decoded_train, sigma=1)
        H_decoded_train_norm = H_decoded_train_smooth / np.sum(H_decoded_train_smooth)

        # --- Spatial correlation between true and decoded occupancy ---
        mask_nonzero = (H_true_norm > 0) | (H_decoded_norm > 0)
        r_occ, _ = pearsonr(
            H_true_norm[mask_nonzero].flatten(), H_decoded_norm[mask_nonzero].flatten()
        )

        # ----------------------------------------------------------------------
        # 3. PLOTTING
        # ----------------------------------------------------------------------

        # create a long vertical figure with gridspec
        fig = plt.figure(figsize=(10, 18))
        gs = gridspec.GridSpec(6, 3, figure=fig)  # 6 rows, 3 columns

        # Plot summary behavior at the top, different gridspec
        nrows = 2
        ncols = len(DataHelper.fullBehavior["Times"]["SessionEpochs"])
        gs_summary = gs[0:nrows, :].subgridspec(nrows, ncols)  # occupy first 4 rows
        target_axs = [
            fig.add_subplot(gs_summary[i, j])
            for i in range(nrows)
            for j in range(ncols)
        ]

        self.summary_behavior(
            DataHelper,
            axs=target_axs,
            show=False,
            block=False,
            save=False,
            extended_zone=kwargs.get("extended_zone", False),
        )

        ax = []

        # --- first 3 rows (3x3 grid) ---
        for row in range(3):
            for col in range(3):
                ax.append(fig.add_subplot(gs[row + 2, col]))

        # ax[0]..ax[8] exist normally

        # --- 4th row: one big subplot spanning all columns ---
        ax_big = fig.add_subplot(gs[3, :])  # span all 3 columns
        ax.append(ax_big)  # this will be ax[9]
        # --- 5th row: classic subplots ---
        for col in range(3):
            ax.append(fig.add_subplot(gs[4, col]))  # ax[10], ax[11], ax[12]

        # (A) Occupancy heatmap (true)
        # add maze outline in black
        ax[0].plot(
            xedges[0] + MAZE_COORDS[:, 0] * (xedges[-1] - xedges[0]),
            yedges[0] + MAZE_COORDS[:, 1] * (yedges[-1] - yedges[0]),
            color="black",
            lw=2,
        )
        im0 = ax[0].imshow(
            H_true_norm.T,
            origin="lower",
            cmap="hot",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax[0].set_xlabel("Test")
        # remove ticks and labels for cleaner look
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # remove spines for cleaner look
        for spine in ax[0].spines.values():
            spine.set_visible(False)

        fig.colorbar(im0, ax=ax[0], label="Occupancy prob")
        divider = make_axes_locatable(ax[0])
        ax_cb = divider.append_axes("left", size="100%", pad=0.6)
        ax_cb.plot(
            xedges[0] + MAZE_COORDS[:, 0] * (xedges[-1] - xedges[0]),
            yedges[0] + MAZE_COORDS[:, 1] * (yedges[-1] - yedges[0]),
            color="black",
            lw=2,
        )
        im0 = ax_cb.imshow(
            H_true_train_norm.T,
            origin="lower",
            cmap="hot",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax_cb.set_title("True Occupancy")
        ax_cb.set_xlabel("Train")
        # remove ticks and labels for cleaner look
        ax_cb.set_xticks([])
        ax_cb.set_yticks([])
        # remove spines for cleaner look
        for spine in ax_cb.spines.values():
            spine.set_visible(False)

        # (B) Speed distribution
        ax[1].hist(
            speed[train_mask],
            bins=30,
            label="Train",
            alpha=0.7,
            density=True,
            color="green",
        )
        ax[1].hist(
            speed[test_mask],
            bins=30,
            label="Test",
            alpha=0.7,
            density=True,
            color="orange",
        )
        ax[1].set_title("Speed Distribution")
        ax[1].set_xlabel("Speed (cm/s)")
        ax[1].set_ylabel("Density")
        ax[1].grid(True, alpha=0.3)
        # removed per-axis legend here

        # (C) Freezing fraction
        ax[2].bar(
            ["Train", "Test"],
            [freeze_fraction_train, freeze_fraction_test],
            color=["green", "orange"],
        )
        ax[2].set_title("Freezing Fraction")
        ax[2].set_ylabel("Proportion of time\nin freezing")
        ax[2].set_ylim(0, 1)

        # (D) Error over time with speed mask - + speed on a twin axis
        ax[3].plot(
            time[speed_mask],
            error[speed_mask],
            lw=0.5,
            alpha=0.6,
            label="Instantaneous",
        )
        ax[3].plot(
            time[speed_mask], rolling_mae[speed_mask], lw=2, label="Rolling MAE (5s)"
        )
        ax[3].axvspan(
            min(time[~test_mask]),
            max(time[~test_mask]),
            color="green",
            alpha=0.2,
            label="Train",
        )
        ax[3].axvspan(
            min(time[test_mask]),
            max(time[test_mask]),
            color="orange",
            alpha=0.5,
            label="Test",
        )
        ax[3].set_title("Decoding Error Over Time (With Speed Mask)")
        ax[3].set_ylabel("Error (lin. dist.)")
        # removed per-axis legend here
        # add chance level as a horizontal line
        ax[3].axhline(
            chance_mae_speed,
            color="pink",
            linestyle="--",
            lw=1,
            label="Chance level on movement",
        )
        ax[3].set_ylim(0, 1)

        # (E) Error over time without speed mask
        ax[4].plot(time, error, lw=0.5, alpha=0.6, label="Instantaneous")
        ax[4].plot(time, rolling_mae, lw=2, label="Rolling MAE (5s)")
        ax[4].axvspan(
            min(time[~test_mask]),
            max(time[~test_mask]),
            color="green",
            alpha=0.2,
            label="Train",
        )
        ax[4].axvspan(
            min(time[test_mask]),
            max(time[test_mask]),
            color="orange",
            alpha=0.5,
            label="Test",
        )
        ax[4].set_title("Decoding Error Over Time (No Speed Mask)")
        ax[4].set_ylabel("Error (lin. dist.)")
        # removed per-axis legend here
        # add chance level as a horizontal line
        ax[4].axhline(
            chance_mae, color="red", linestyle="--", lw=1, label="Chance level"
        )
        ax[4].set_ylim(0, 1)

        # (F) Spatial decoding error heatmap
        # add maze outline in white
        ax[5].plot(
            xedges[0] + MAZE_COORDS[:, 0] * (xedges[-1] - xedges[0]),
            yedges[0] + MAZE_COORDS[:, 1] * (yedges[-1] - yedges[0]),
            color="white",
            lw=2,
        )
        im2 = ax[5].imshow(
            H_err_smooth.T,
            origin="lower",
            cmap="coolwarm",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax[5].set_title("Mean Spatial Decoding Error (lin. dist.)")
        ax[5].set_xticks([])
        ax[5].set_yticks([])
        for spine in ax[5].spines.values():
            spine.set_visible(False)
        fig.colorbar(im2, ax=ax[5], label="Error (lin. dist.)")

        # (G) Occupancy map (decoded)
        ax[6].plot(
            xedges[0] + MAZE_COORDS[:, 0] * (xedges[-1] - xedges[0]),
            yedges[0] + MAZE_COORDS[:, 1] * (yedges[-1] - yedges[0]),
            color="black",
            lw=2,
        )
        im3 = ax[6].imshow(
            H_decoded_norm.T,
            origin="lower",
            cmap="hot",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax[6].set_xlabel("Test")
        ax[6].set_xticks([])
        ax[6].set_yticks([])
        for spine in ax[6].spines.values():
            spine.set_visible(False)
        fig.colorbar(im3, ax=ax[6], label="Occupancy probability")
        divider = make_axes_locatable(ax[6])
        ax_cb = divider.append_axes("left", size="100%", pad=0.6)
        ax_cb.plot(
            xedges[0] + MAZE_COORDS[:, 0] * (xedges[-1] - xedges[0]),
            yedges[0] + MAZE_COORDS[:, 1] * (yedges[-1] - yedges[0]),
            color="black",
            lw=2,
        )
        im0 = ax_cb.imshow(
            H_decoded_train_norm.T,
            origin="lower",
            cmap="hot",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax_cb.set_title("Decoded Occupancy")
        ax_cb.set_xlabel("Train")
        # remove ticks and labels for cleaner look
        ax_cb.set_xticks([])
        ax_cb.set_yticks([])
        for spine in ax_cb.spines.values():
            spine.set_visible(False)

        # (H) True vs Decoded Occupancy comparison
        ax[7].scatter(H_true_norm.flatten(), H_decoded_norm.flatten(), s=8, alpha=0.5)
        ax[7].plot(
            [0, max(H_true_norm.max(), H_decoded_norm.max())],
            [0, max(H_true_norm.max(), H_decoded_norm.max())],
            "k--",
            lw=1,
        )
        ax[7].set_xlabel("True occupancy")
        ax[7].set_ylabel("Decoded occupancy")
        ax[7].set_title(f"Occupancy correlation r = {r_occ:.2f}")

        # (I) Summary metrics + legend consolidation
        text = (
            f"Train MAE: {mae_train:.2f} lin. dist. (~{mae_train * 100:.1f} cm) vs {mae_train_raw:.2f} with immobility.\n"
            f"Test MAE: {mae_test:.2f} lin. dist. (~{mae_test * 100:.1f} cm) vs {mae_test_raw:.2f} with immobility.\n"
            f"Freeze duration: {freeze_duration_total:.1f}s\n"
            f"Occupancy corr (r): {r_occ:.2f}\n"
            f"Train freeze: {freeze_fraction_train * 100:.1f}%\n"
            f"Test freeze: {freeze_fraction_test * 100:.1f}%"
        )
        ax[8].text(0.02, 0.48, text, fontsize=12, va="center", wrap=True)

        ax[8].axis("off")
        ax[8].set_title("Summary Metrics")

        # the last subplot line is an ax that spans the whole row for better layout
        # it just shows an example of true and decoded trajectory in the linear space
        ax[9].plot(
            time[test_mask], lin_true[test_mask], alpha=0.7, color=TRUE_LINE_COLOR
        )
        ax[9].plot(
            time[test_mask],
            lin_pred[test_mask],
            "--",
            alpha=0.5,
            color=PREDICTED_LINE_COLOR,
        )
        ax[9].scatter(
            time[test_mask & speed_mask],
            lin_pred[test_mask & speed_mask],
            label="Decoded Position on movement",
            s=20,
            alpha=0.7,
            color=DELTA_COLOR_FORWARD,
            marker="o",
        )
        ax[9].scatter(
            time[test_mask & ~speed_mask],
            lin_pred[test_mask & ~speed_mask],
            label="Decoded Position while immobile",
            s=6,
            alpha=0.7,
            color=DELTA_COLOR_REVERSE,
            marker="o",
        )
        ax[9].set_xlabel("Time (s)")
        ax[9].set_ylabel("Linearized Position")
        ax[9].set_title("Performance on test set")

        # Collect handles & labels from axes that used legends and put a single legend box into ax[8]
        source_axes = [ax[1], ax[3], ax[4], ax[9]]
        handles_map = {}
        for a in source_axes:
            h, l = a.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in handles_map:  # keep first handle for each unique label
                    handles_map[ll] = hh
        if len(handles_map) > 0:
            unique_labels = list(handles_map.keys())
            unique_handles = [handles_map[k] for k in unique_labels]
            # place legend on the right side of ax[8] so it doesn't overlap the text
            # + move it even more to the right
            legend = ax[8].legend(
                unique_handles,
                unique_labels,
                loc="lower left",
                bbox_to_anchor=(0.57, -1.3),
                fontsize="x-small",
            )
            ax[8].add_artist(legend)

        # bayesian and "neuron" analysis
        # from bayesian_neurons_summary method
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
        # Linear tuning curves (if computed)
        if hasattr(self.trainerBayes, "orderedLinearPlaceFields"):
            norm_fields = self.trainerBayes.orderedLinearPlaceFields / np.maximum(
                np.mean(
                    self.trainerBayes.orderedLinearPlaceFields, axis=1, keepdims=True
                ),
                1e-8,
            )
            lTc = ax[10].imshow(norm_fields, aspect="auto", origin="lower")
            linearBins = np.linspace(0, 1, norm_fields.shape[1])
            ax[10].set_xticks(
                ticks=np.linspace(0, norm_fields.shape[1], norm_fields.shape[1])[::20],
                labels=np.round(linearBins[::20], 2),
            )
            ax[10].set_xlabel("Linear Position")
            ax[10].set_ylabel("Neuron Index")
            ax[10].set_xticks([])
            ax[10].set_yticks([])
            divider = make_axes_locatable(ax[10])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(lTc, label="Deviation from unit Mean Firing Rate", cax=cax)
            ax[10].set_title("Linear Tuning Curves")
            # summary metrics about high-quality neurons

            text = (
                f"High-quality place cells: {len(high_quality)} neurons.\n"
                f"Found {len(self.trainerBayes.linearPosArgSort)} neurons.\n"
                f"Position range: {self.trainerBayes.linearPreferredPos.min():.2f} - {self.trainerBayes.linearPreferredPos.max():.2f}.\n"
            )
            new_ax = make_axes_locatable(ax[10]).append_axes(
                "right", size="30%", pad=0.1
            )
            new_ax.axis("off")
            new_ax.text(0.1, 0.5, text, fontsize=10, va="center", wrap=True)

        ax[11].plot([1, 2], [1, 2])  # empty plot to be removed
        ax[12].plot([1, 2], [1, 2])  # empty plot to be removed

        # show
        plt.tight_layout()
        fig.suptitle(
            f"Decoding ID Card for {os.path.basename(os.path.dirname(self.projectPath.baseName))} - Experiment: {os.path.basename(self.projectPath.experimentPath)} - Phase: {suffix.strip('_')}"
        )

        if show:
            if mplt.get_backend().lower() == "QtAgg".lower():
                plt.get_current_fig_manager().window.showMaximized()
            elif mplt.get_backend().lower() == "TkAgg".lower():
                plt.get_current_fig_manager().resize(
                    *plt.get_current_fig_manager().window.maxsize()
                )
            plt.show(block=block)

        if save:
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    f"summary_id_card_{timeWindow}ms{suffix}.png",
                ),
                dpi=300,
            )
            fig.savefig(
                os.path.join(
                    self.folderFigures,
                    f"summary_id_card_{timeWindow}ms{suffix}.svg",
                ),
            )
        plt.close(fig)

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

    def _plot_single_error_matrix(
        self,
        true_pos,
        pred_pos,
        ax=None,
        nbins=40,
        normalized=True,
        cmap="viridis",  # Replace with white_viridis if defined elsewhere
    ):
        """
        Helper function to plot a single error matrix.
        If ax is None, creates a new figure.
        """
        # 1. Create Figure/Axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure

        # 2. Compute 2D Histogram
        # x=Pred, y=True based on your original logic
        H, xedges, yedges = np.histogram2d(
            pred_pos.reshape(-1),
            true_pos.reshape(-1),
            bins=(nbins, nbins),
            density=True,
        )

        # 3. Normalize (optional)
        if normalized:
            # Axis 1 is the 'y' input to histogram2d (TruePos)
            # We add a small epsilon or handle 0 to avoid NaNs
            max_vals = H.max(axis=1)
            max_vals[max_vals == 0] = 1.0
            H = H / max_vals[:, None]

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # 4. Plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Note: Transpose H.T is used to align axes correctly with origin="lower"
        im = ax.imshow(
            H.T,
            extent=extent,
            cmap=cmap,
            interpolation="none",
            origin="lower",
        )

        # 5. Add Colorbar
        fig.colorbar(im, ax=ax)

        return fig, ax

    def error_matrix_linerrors_by_speed(
        self, suffixes=None, nbins=40, normalized=True, show=False
    ):
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
                    H.T,
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
        if show:
            plt.show()
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
        normalized=True,
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
            # ax[iw].scatter(
            #     self.resultsBayes_phase[suffix]["linPred"][iw][masks[iw]],
            #     self.resultsNN_phase[suffix]["linPred"][iw][masks[iw]],
            #     s=1,
            #     c="grey",
            # )
            H, xedges, yedges = np.histogram2d(
                self.resultsBayes_phase[suffix]["linPred"][iw][masks[iw]],
                self.resultsNN_phase[suffix]["linPred"][iw][masks[iw]],
                bins=(40, 40),
                density=True,
            )
            if normalized:
                with np.errstate(invalid="ignore"):
                    H = H / H.max(axis=1, keepdims=True)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax[iw].imshow(
                H.T,
                extent=extent,
                cmap="viridis",
                interpolation="none",
                origin="lower",
                aspect="auto",
            )
            ax[iw].set_yticks([])
            fig.colorbar(im, ax=ax[iw], label="density")
            if iw < len(self.timeWindows):
                ax[iw].set_xticks([])
            ax[iw].set_aspect("equal")
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
        # [a.set_aspect("auto") for a in ax]
        # [
        #     plt.colorbar(
        #         plt.cm.ScalarMappable(plt.Normalize(0, 1), cmap=white_viridis),
        #         ax=a,
        #         label="density",
        #     )
        #     for a in ax
        # ]
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

        return predLoss_ticks, errors_filtered

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
        strideFactor=1,
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

        useAll_suffix = "_all" if useAll else ""
        strideFactor_suffix = f"_factor{strideFactor}" if strideFactor > 1 else ""

        loadName = os.path.join(
            self.projectPath.dataPath,
            f"aligned_{phase}{useAll_suffix}{strideFactor_suffix}",
            str(ws),
            "test" if not useTrain else "train",
            f"spikeMat_window_popVector{suffix}.csv",
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
            : len(self.resultsNN_phase[suffix]["linTruePos"][iwindow]), :
        ]
        predLoss = self.resultsNN_phase[suffix]["predLoss"][iwindow]
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        for icell, tuningCurve in enumerate(linearTuningCurves):
            pcId = np.where(np.equal(placeFieldSort, icell))[0][0]
            spikeHist = spikePopAligned[:, pcId + 1][
                : len(self.resultsNN_phase[suffix]["linTruePos"][iwindow])
            ]
            spikeMask = np.greater(spikeHist, 0)

            if spikeMask.any():  # some neurons do not spike here
                cm = plt.get_cmap("gray")
                fig, ax = plt.subplots()
                ax.scatter(
                    self.resultsNN_phase[suffix]["linPred"][iwindow][spikeMask],
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
                            self.resultsNN_phase[suffix]["linTruePos"][iwindow][
                                np.logical_and(
                                    spikeMask,
                                    np.logical_and(
                                        self.resultsNN_phase[suffix]["linPred"][iwindow]
                                        >= linbin,
                                        self.resultsNN_phase[suffix]["linPred"][iwindow]
                                        < binEdges[i + 1],
                                    ),
                                )
                            ]
                            - self.resultsNN_phase[suffix]["linPred"][iwindow][
                                np.logical_and(
                                    spikeMask,
                                    np.logical_and(
                                        self.resultsNN_phase[suffix]["linPred"][iwindow]
                                        >= linbin,
                                        self.resultsNN_phase[suffix]["linPred"][iwindow]
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

    def _plot_single_place_field(self, ax, neuron_idx, pos_x, pos_y, epoch, title):
        """Helper to plot a single place field on a specific axis."""
        spike_time = nap.Ts(
            self.trainerBayes.spikeMatTimes[
                self.trainerBayes.spikeMatLabels[:, neuron_idx] == 1
            ].flatten()
        )

        results = _run_place_field_analysis(
            spike_time,
            pos_x,
            pos_y,
            epoch=nap.IntervalSet(epoch),
            smoothing=3,
            freq_video=30,
            threshold=0.7,
            size_map=50,
            limit_maze=(0, 1, 0, 1),
            large_matrix=True,
        )

        # Plot Heatmap
        field_data = results["map"]["rate"]
        ax.imshow(field_data, aspect="auto", origin="lower")

        # Plot Peak
        peak_x = results["stats"]["x"]
        peak_y = results["stats"]["y"]
        ax.plot(peak_x, peak_y, "rx", markersize=10, markeredgewidth=2, label="Peak FR")
        ax.text(
            peak_x + 2,
            peak_y + 2,
            f"{results['stats']['peak']:.2f}Hz",
            color="magenta",
            fontsize=10,
            fontweight="bold",
        )

        # Plot Contours
        if "field" in results["stats"]:
            field_mask = results["stats"]["field"]
            ax.contour(
                field_mask,
                levels=[0.5],
                colors="b",
                linewidths=1,
                origin="lower",
                extent=(0, field_mask.shape[1], 0, field_mask.shape[0]),
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def bayesian_neurons_summary(self, axs=None, fig=None, block=False, **kwargs):
        """
        Summary of the Bayesian neurons.
        Can create its own figure or plot on provided axes.
        """
        # kwargs processing
        plot_high_quality = kwargs.get("plot_high_quality", False)
        save = kwargs.get("save", True if axs is None else False)
        show = kwargs.get("show", True if axs is None else False)

        # --- 1. Train/Load Data ---
        if getattr(self, "bayesMatrices", None) is None:
            existing_bayes = (
                self.bayesMatrices
                if (
                    isinstance(self.bayesMatrices, dict)
                    and "Occupation" in self.bayesMatrices
                )
                else None
            )

            self.bayesMatrices = self.trainerBayes.train_order_by_pos(
                self.behaviorData,
                l_function=self.l_function,
                bayesMatrices=existing_bayes,
                **kwargs,
            )

        # Extract and sort Mutual Information
        flat_mi = [
            mi for tetrode_mi in self.bayesMatrices["mutualInfo"] for mi in tetrode_mi
        ]
        ordered_mi = np.array(flat_mi)[self.trainerBayes.linearPosArgSort]

        # --- 2. Identify High-Quality Neurons ---
        thresh = 80
        percentile_val = np.percentile(ordered_mi, thresh)
        high_quality_mask = ordered_mi > percentile_val
        high_quality_indices = self.trainerBayes.linearPosArgSort[high_quality_mask]

        print(
            f"High-quality place cells: {len(high_quality_indices)} neurons (top {100 - thresh}%)"
        )
        print(f"Total neurons: {len(self.trainerBayes.linearPosArgSort)}")
        print(
            f"Position range: {self.trainerBayes.linearPreferredPos.min():.2f} - {self.trainerBayes.linearPreferredPos.max():.2f}"
        )

        # --- 3. Visualization Setup ---
        if axs is None:
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()
        else:
            axs = np.array(axs).flatten()
            if fig is None:
                fig = axs[0].figure

        # Validate we have enough axes
        if len(axs) < 6:
            raise ValueError(
                f"Provided 'axs' must have at least 6 subplots, got {len(axs)}."
            )

        # --- Data Prep for Place Fields (Do once) ---
        pos_x = nap.Tsd(
            d=self.behaviorData["Positions"][:, 0],
            t=self.behaviorData["positionTime"].flatten(),
        )
        pos_y = nap.Tsd(
            d=self.behaviorData["Positions"][:, 1],
            t=self.behaviorData["positionTime"].flatten(),
        )
        epoch = np.concatenate(
            [
                self.behaviorData["Times"]["trainEpochs"],
                self.behaviorData["Times"]["testEpochs"],
            ]
        ).reshape(-1)

        # --- Panel 0: First Ordered Place Field ---
        neuron_first = (
            self.trainerBayes.linearPosArgSort[1]
            if not plot_high_quality
            else high_quality_indices[0]
        )
        self._plot_single_place_field(
            axs[0], neuron_first, pos_x, pos_y, epoch, "First Ordered Place Field"
        )

        # --- Pre-calculate Linear Fields ---
        has_linear = hasattr(self.trainerBayes, "orderedLinearPlaceFields")
        norm_fields = None
        if has_linear:
            raw_fields = self.trainerBayes.orderedLinearPlaceFields
            # Safe Z-score normalization
            std_val = np.std(raw_fields, axis=0)
            norm_fields = (raw_fields - np.mean(raw_fields, axis=0)) / (std_val + 1e-8)

            linear_bins = np.linspace(0, 1, norm_fields.shape[1])
            x_ticks = np.arange(0, norm_fields.shape[1], 20)
            x_labels = np.round(linear_bins[::20], 2)

        # --- Panel 1: All Linear Tuning Curves ---
        ax = axs[1]
        if has_linear and norm_fields is not None:
            norm = mcolors.TwoSlopeNorm(
                vmin=norm_fields.min(), vcenter=0, vmax=norm_fields.max()
            )
            im = ax.imshow(
                norm_fields, aspect="auto", origin="lower", cmap="coolwarm", norm=norm
            )
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Linear Position")
            ax.set_ylabel("Neuron Index")
            ax.set_title("Linear Tuning Curves")
            fig.colorbar(im, ax=ax, label="Deviation (Z-score)")
        else:
            ax.axis("off")

        # --- Panel 2: Position Coverage ---
        ax = axs[2]
        ax.hist(self.trainerBayes.linearPreferredPos, bins=20, alpha=0.7, color="teal")
        ax.set_xlabel("Linear Position")
        ax.set_title("Pos Coverage in Training Data")

        # --- Panel 3: Quality Metrics (Mutual Info) ---
        ax = axs[3]
        colors = np.array(["blue"] * len(ordered_mi))
        colors[high_quality_mask] = "red"

        ax.plot(ordered_mi, "o-", alpha=0.3, zorder=0, color="gray", linewidth=0.5)
        ax.scatter(
            np.arange(len(ordered_mi)), ordered_mi, c=colors, alpha=0.8, zorder=1, s=15
        )
        ax.set_title("Mutual Information (ordered)")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Mutual Information")

        # --- Panel 4: Best Linear Tuning Curves (High Quality Only) ---
        ax = axs[4]
        if has_linear and norm_fields is not None and high_quality_mask.sum() > 0:
            hq_fields = norm_fields[high_quality_mask]
            norm = mcolors.TwoSlopeNorm(
                vmin=hq_fields.min(), vcenter=0, vmax=hq_fields.max()
            )
            im = ax.imshow(
                hq_fields, aspect="auto", origin="lower", cmap="coolwarm", norm=norm
            )

            hq_indices = np.arange(norm_fields.shape[0])[high_quality_mask]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)

            # Intelligent Y-tick labeling
            if len(hq_indices) > 20:
                y_display_idx = np.linspace(0, len(hq_indices) - 1, 10, dtype=int)
                ax.set_yticks(y_display_idx)
                ax.set_yticklabels(hq_indices[y_display_idx])
            else:
                ax.set_yticks(np.arange(len(hq_indices)))
                ax.set_yticklabels(hq_indices)

            ax.set_xlabel("Linear Position")
            ax.set_ylabel("Original Index")
            ax.set_title(f"Best Linear Tuning Curves (Top {100 - thresh}%)")
            fig.colorbar(im, ax=ax, label="Deviation")
        else:
            ax.text(0.5, 0.5, "No High Quality Fields", ha="center", va="center")
            ax.axis("off")

        # --- Panel 5: Last Ordered Place Field ---
        neuron_last = (
            self.trainerBayes.linearPosArgSort[-1]
            if not plot_high_quality
            else high_quality_indices[-1]
        )
        self._plot_single_place_field(
            axs[5], neuron_last, pos_x, pos_y, epoch, "Last Ordered Place Field"
        )

        # --- Finalize and Save ---
        if save or show:
            plt.tight_layout()

        if save:
            filename = f"bayesian_neurons_summary{self.suffix}"
            fig.savefig(os.path.join(self.folderFigures, f"{filename}.png"), dpi=300)
            fig.savefig(os.path.join(self.folderFigures, f"{filename}.svg"))

        if show:
            plt.show(block=block)
        elif save:
            # If we saved but didn't show, close the figure to free memory
            plt.close(fig)

        return fig


if __name__ == "__main__":
    import warnings

    import tqdm

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
