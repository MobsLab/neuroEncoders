import logging
import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykeops
import tables
from pynapple import IntervalSet

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.resultAnalysis import ripple_analysis_utils
from neuroencoders.resultAnalysis.hyper_paper_figures import barplot_sleep_predLoss
from neuroencoders.simpleBayes.decode_bayes import Trainer as TrainerBayes
from neuroencoders.utils.global_classes import Project
from neuroencoders.utils.viz_params import white_viridis

plt.style.use("neuroencoders.mobs")
pykeops.set_verbose(False)


class PaperFiguresSleep:
    def __init__(
        self,
        projectPath: Project,
        behavior_data: dict,
        bayes: Optional[TrainerBayes],
        linearizationFunction: Optional[Callable] = None,
        bayesMatrices: dict = {},
        timeWindows=[36],
        sleepNames=["PreSleep", "PostSleep"],
        rippleChoice="start",
        folderFigures=None,
        verbose=True,
    ):
        self.projectPath = projectPath
        self.bayes = bayes
        self.behavior_data = behavior_data
        self.behaviorData = behavior_data
        self.l_function = linearizationFunction
        self.bayesMatrices = bayesMatrices
        self.timeWindows = timeWindows
        self.sleepNames = sleepNames
        _, self.samplingRate, _ = get_params(self.projectPath.xml)
        match rippleChoice:
            case "start":
                self.ripCol = 1
            case "center":
                self.ripCol = 2
            case "end":
                self.ripCol = 2

        self.binsLinearPosHist = np.arange(
            0, stop=1, step=0.01
        )  # discretisation of the linear variable to help in some plots
        self.cm = plt.get_cmap("tab20b")
        # Manage folders
        if folderFigures is None:
            self.folderFigures = os.path.join(
                self.projectPath.experimentPath, "figures"
            )
        else:
            self.folderFigures = os.path.join(
                self.projectPath.experimentPath, folderFigures
            )
        if not os.path.exists(self.folderFigures):
            os.mkdir(self.folderFigures)
        try:
            self.folderResultSleep = self.projectPath.folderResultSleep
        except AttributeError:
            self.folderResultSleep = os.path.join(
                self.projectPath.experimentPath, "results_Sleep"
            )
            self.projectPath.folderResultSleep = self.folderResultSleep

        self.folderAligned = os.path.join(self.projectPath.dataPath, "aligned")
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.resultsNN = {
            "times": {},
            "linPred": {},
            "fullPred": {},
            "featurePred": {},
            "predLoss": {},
            "Hn": {},
            "maxp": {},
            "posIndex": {},
            "indexInDat": {},
        }
        self.resultsNN_phase = {}
        self.resultsBayes = {
            "times": {},
            "linPred": {},
            "fullPred": {},
            "featurePred": {},
            "predLoss": {},
            "posIndex": {},
            "indexInDat": {},
        }
        self.resultsBayes_phase = {}
        self.resultsNN_phase_pkl = {}
        self.resultsBayes_phase_pkl = {}
        self.find_session_epochs()

    def _load_csv_result(
        self, base_path: str, prefix: str, dtype=np.float32
    ) -> Optional[np.ndarray]:
        filepath = os.path.join(base_path, f"{prefix}.csv")
        if not os.path.exists(filepath):
            return None
        try:
            return np.array(pd.read_csv(filepath).values[:, 1:], dtype=dtype)
        except Exception as exc:
            self.logger.warning(f"Failed to load {filepath}: {exc}")
            return None

    def find_session_epochs(self):
        """Load awake session masks when the behavior dictionary contains them."""
        self.training = None
        self.trainMask = None
        self.testing = None
        self.testMask = None

        times = (
            self.behavior_data.get("Times", {})
            if isinstance(self.behavior_data, dict)
            else {}
        )
        try:
            self.training = IntervalSet(np.array(times["trainEpochs"]).reshape(-1, 2))
            self.trainMask = inEpochsMask(
                self.behavior_data["positionTime"][:, 0], self.training
            )
        except Exception:
            pass
        try:
            self.testing = IntervalSet(np.array(times["testEpochs"]).reshape(-1, 2))
            self.testMask = inEpochsMask(
                self.behavior_data["positionTime"][:, 0], self.testing
            )
        except Exception:
            pass

    def _load_sleep_results(
        self,
        prefix: str = "",
        sleepNames: Optional[List[str]] = None,
        proxy_file: Optional[str] = "Hn",
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        sleep_names = sleepNames or list(self.sleepNames)
        results = {
            "times": {},
            "linPred": {},
            "fullPred": {},
            "featurePred": {},
            "predLoss": {},
            "posIndex": {},
            "indexInDat": {},
        }
        if proxy_file == "Hn":
            results["Hn"] = {}
            results["maxp"] = {}

        for sleepName in sleep_names:
            results["times"][sleepName] = []
            results["linPred"][sleepName] = []
            results["fullPred"][sleepName] = []
            results["featurePred"][sleepName] = []
            results["predLoss"][sleepName] = []
            results["posIndex"][sleepName] = []
            results["indexInDat"][sleepName] = []
            if proxy_file == "Hn":
                results["Hn"][sleepName] = []
                results["maxp"][sleepName] = []

            for ws in self.timeWindows:
                pathToSleep = os.path.join(self.folderResultSleep, str(ws), sleepName)
                feature_pred = self._load_csv_result(
                    pathToSleep, f"{prefix}featurePred"
                )
                linear_pred = self._load_csv_result(pathToSleep, f"{prefix}linearPred")
                time_pred = self._load_csv_result(pathToSleep, f"{prefix}timeStepsPred")
                pos_index = self._load_csv_result(
                    pathToSleep, f"{prefix}posIndex", dtype=np.int64
                )
                index_in_dat = self._load_csv_result(
                    pathToSleep, f"{prefix}indexInDat", dtype=np.int64
                )

                proxy_loss = None
                if proxy_file is not None:
                    proxy_loss = self._load_csv_result(pathToSleep, proxy_file)
                    if proxy_loss is None and proxy_file == "Hn":
                        proxy_loss = self._load_csv_result(pathToSleep, "lossPred")
                    if proxy_file == "Hn":
                        maxp = self._load_csv_result(pathToSleep, "maxp")
                        results["Hn"][sleepName].append(
                            np.squeeze(proxy_loss).flatten()
                            if proxy_loss is not None
                            else None
                        )
                        results["maxp"][sleepName].append(
                            np.squeeze(maxp).flatten() if maxp is not None else None
                        )

                results["times"][sleepName].append(
                    np.squeeze(time_pred).flatten() if time_pred is not None else None
                )
                results["linPred"][sleepName].append(
                    np.squeeze(linear_pred).flatten()
                    if linear_pred is not None
                    else None
                )
                results["featurePred"][sleepName].append(feature_pred)
                results["fullPred"][sleepName].append(feature_pred)
                results["predLoss"][sleepName].append(
                    np.squeeze(proxy_loss).flatten() if proxy_loss is not None else None
                )
                results["posIndex"][sleepName].append(
                    np.squeeze(pos_index).flatten() if pos_index is not None else None
                )
                results["indexInDat"][sleepName].append(
                    np.squeeze(index_in_dat).flatten()
                    if index_in_dat is not None
                    else None
                )

        return results

    def load_data(self, sleepNames=None):
        """Load sleep ANN decoding results and ripple-aligned metadata."""
        ann_results = self._load_sleep_results(prefix="", sleepNames=sleepNames)
        self.resultsNN = ann_results
        self.resultsNN_phase = {
            sleepName: {
                "times": ann_results["times"][sleepName],
                "linearPred": ann_results["linPred"][sleepName],
                "featurePred": ann_results["featurePred"][sleepName],
                "fullPred": ann_results["fullPred"][sleepName],
                "predLoss": ann_results["predLoss"][sleepName],
                "Hn": ann_results.get("Hn", {}).get(sleepName, []),
                "maxp": ann_results.get("maxp", {}).get(sleepName, []),
                "posIndex": ann_results["posIndex"][sleepName],
                "indexInDat": ann_results["indexInDat"][sleepName],
            }
            for sleepName in (sleepNames or self.sleepNames)
        }

        # Load ripples
        # TODO: add maskSleep maskSleep = inEpochsMask(ripples[:, rippleChoice], behavior_data["Times"]["sleepEpochs"][:2])
        # TODO: should I normalize lossPred?
        with tables.open_file(self.projectPath.folder + "nnSWR.mat", "a") as f:
            ripples = f.root.ripple[:, :].transpose()

        timesRipples = {}
        idCloseRipples = {}
        idCloseRipplesInSleep = {}
        timeDistToRipples = {}
        rippleTimeJ = pykeops.numpy.Vj(
            ripples[:, self.ripCol].astype(dtype=np.float64)[:, None]
        )
        rippleTimeI = pykeops.numpy.Vi(
            ripples[:, self.ripCol].astype(dtype=np.float64)[:, None]
        )
        for isleep, sleepName in enumerate(sleepNames or self.sleepNames):
            timesRipples[sleepName] = []
            idCloseRipples[sleepName] = []
            idCloseRipplesInSleep[sleepName] = []
            timeDistToRipples[sleepName] = []
            for i in range(len(self.timeWindows)):
                # Calculating ids of timesteps that are closest to ripple times
                timesRipples[sleepName].append(ripples[:, self.ripCol])
                sleep_times = self.resultsNN["times"][sleepName][i]
                predTime = pykeops.numpy.Vi(
                    sleep_times.astype(dtype=np.float64)[:, None]
                )
                idCloseRipples[sleepName].append(
                    ((predTime - rippleTimeJ).abs().argmin(axis=0))[:, 0]
                )
                # We remove ripple time tat are not inside tjhe predictio time (for exmaple in sleep)
                idCloseRipplesInSleep[sleepName].append(
                    idCloseRipples[sleepName][i][
                        inEpochsMask(
                            ripples[:, self.ripCol],
                            [np.min(sleep_times), np.max(sleep_times)],
                        )
                    ]
                )  # aka ripple time
                # Calculating the distance between the closest ripple time and everytimesteps
                predTime = pykeops.numpy.Vj(
                    sleep_times.astype(dtype=np.float64)[:, None]
                )
                timeDistToRipples[sleepName].append(
                    ((predTime - rippleTimeI).abs().min(axis=0))[:, 0]
                )

        # Output
        self.ripples = {
            "times": ripples[:, self.ripCol],
            "idCloseRipples": idCloseRipples,
            "idCloseRipplesInSleep": idCloseRipplesInSleep,
            "timeDistToRipples": timeDistToRipples,
        }

    def load_bayes(self, sleepNames=None):
        """Load sleep Bayes decoding results when they are present on disk."""
        bayes_results = self._load_sleep_results(
            prefix="bayes_", sleepNames=sleepNames, proxy_file="bayes_proba"
        )
        self.resultsBayes = bayes_results
        self.resultsBayes_phase = {
            sleepName: {
                "times": bayes_results["times"][sleepName],
                "linearPred": bayes_results["linPred"][sleepName],
                "featurePred": bayes_results["featurePred"][sleepName],
                "fullPred": bayes_results["fullPred"][sleepName],
                "predLoss": bayes_results["predLoss"][sleepName],
                "posIndex": bayes_results["posIndex"][sleepName],
                "indexInDat": bayes_results["indexInDat"][sleepName],
            }
            for sleepName in (sleepNames or self.sleepNames)
        }

    def fig_example_sleep_linear(self):
        fig, ax = plt.subplots(
            len(self.timeWindows),
            len(self.sleepNames),
            figsize=(18, 10),
            sharex="col",
            sharey=True,
        )
        ax = ax.reshape(len(self.timeWindows), len(self.sleepNames))
        for isleep, sleepName in enumerate(self.sleepNames):
            for i in range(len(self.timeWindows)):
                ax[i, isleep].scatter(
                    self.resultsNN["times"][sleepName][i],
                    self.resultsNN["linPred"][sleepName][i],
                    c=self.cm(12 + 0),
                    alpha=0.9,
                    label=(str(self.timeWindows[i]) + " ms"),
                    s=1,
                )
                if i == 0:
                    ax[i, isleep].set_title(
                        f"{sleepName} linear decoded position for {self.timeWindows[i]} ms window",
                    )
                if i == len(self.timeWindows) - 1:
                    ax[i, isleep].set_xlabel("samples", fontsize="xx-large")
                ax[i, isleep].set_ylabel("linear position", fontsize="xx-large")
                ax[i, isleep].set_yticks([0, 0.4, 0.8])
        # Save figure
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folderFigures, "example_sleep_nn.png"))
        fig.savefig(os.path.join(self.folderFigures, "example_sleep_nn.svg"))

    def fig_sleep_distribution_linear(self):
        fig, ax = plt.subplots(
            len(self.timeWindows),
            len(self.sleepNames),
            figsize=(10, 10),
            sharex="col",
            sharey=True,
        )
        ax = ax.reshape(len(self.timeWindows), len(self.sleepNames))
        for isleep, sleepName in enumerate(self.sleepNames):
            for i in range(len(self.timeWindows)):
                ax[i, isleep].hist(
                    self.resultsNN["linPred"][sleepName][i], bins=50, color="black"
                )
                ax[i, isleep].set_title(
                    f"{sleepName} linear decoded distribution for {self.timeWindows[i]} ms window",
                )
                ax[i, isleep].set_xlabel("linear position", fontsize="xx-large")
                ax[i, isleep].set_ylabel("count", fontsize="xx-large")
        # Save figure
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folderFigures, "distr_sleep_nn.png"))
        fig.savefig(os.path.join(self.folderFigures, "distr_sleep_nn.svg"))

    def fig_sleep_distribution_lossPred(self):
        fig, ax = plt.subplots(
            len(self.timeWindows),
            len(self.sleepNames),
            figsize=(10, 10),
            sharex=True,
            sharey=True,
        )
        ax = ax.reshape(len(self.timeWindows), len(self.sleepNames))
        for isleep, sleepName in enumerate(self.sleepNames):
            for i in range(len(self.timeWindows)):
                ax[i, isleep].hist(
                    self.resultsNN["predLoss"][sleepName][i], bins=50, color="black"
                )
                ax[i, isleep].set_title(
                    f"{sleepName} predicted loss distribution for {self.timeWindows[i]} ms window",
                )
                ax[i, isleep].set_xlabel("predicted loss", fontsize="xx-large")
                ax[i, isleep].set_ylabel("count", fontsize="xx-large")
        # Save figure
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folderFigures, "distr_sleep_pred_loss_nn.png"))
        fig.savefig(os.path.join(self.folderFigures, "distr_sleep_pred_loss_nn.svg"))

    def fig_sleep_barplot_lossPred(self):
        return self.barplot_sleep_predLoss()

    def barplot_sleep_predLoss(self, sleepNames=None, dirSave=None, suffix=""):
        """Delegate sleep pred-loss barplotting to the shared helper."""
        if dirSave is None:
            dirSave = self.folderFigures
        return barplot_sleep_predLoss(
            self.resultsNN["predLoss"],
            timeWindows=self.timeWindows,
            sleepNames=tuple(sleepNames or self.sleepNames),
            dirSave=dirSave,
            suffix=suffix,
        )

    def fig_ripples_hist_sleep_and_out(self):
        """Plot histogram of predicted loss during ripples vs all times."""
        ripple_analysis_utils.plot_ripple_losspredict_distribution(
            predloss_dict=self.resultsNN["predLoss"],
            ripple_indices_dict=self.ripples["idCloseRipplesInSleep"],
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
            filename_prefix="distr_lossPred_sleep_during_ripples",
        )

    def fig_ripples_hist_pred_during_ripples(self, duringRipples=True):
        """Plot histogram of linear predicted position during ripples."""
        ripple_analysis_utils.plot_ripple_linear_pred_distribution(
            linpred_dict=self.resultsNN["linPred"],
            ripple_indices_dict=self.ripples["idCloseRipplesInSleep"],
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
            during_ripples=duringRipples,
        )

    def fig_ripples_scatter_position_lossPred_during_ripples(self):
        """Scatter plot of linear predicted position vs predicted loss during ripples."""
        ripple_analysis_utils.plot_ripple_position_vs_losspredict(
            linpred_dict=self.resultsNN["linPred"],
            predloss_dict=self.resultsNN["predLoss"],
            ripple_indices_dict=self.ripples["idCloseRipplesInSleep"],
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
        )

    def fig_ripples_scatter_time_lossPred_during_ripples(self):
        """Scatter plot of time vs predicted loss during ripples."""
        ripple_analysis_utils.plot_ripple_time_vs_losspredict(
            time_dict=self.resultsNN["times"],
            predloss_dict=self.resultsNN["predLoss"],
            ripple_indices_dict=self.ripples["idCloseRipplesInSleep"],
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
        )

    def fig_ripples_final_figure(self, window=0.4):
        """Plot predicted loss as function of time-to-ripple for sleep decoding."""
        ripple_analysis_utils.plot_ripple_time_distance_vs_losspredict(
            time_distance_dict=self.ripples["timeDistToRipples"],
            predloss_dict=self.resultsNN["predLoss"],
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
            window=window,
            filename_prefix="lossSleepRipples",
        )

    def fig_position_replay_analysis(self, loss_threshold=0.65):
        """
        Analyze if training positions are more replayed during sleep.
        Compares NN predicted positions with loss predictions, showing
        filtered (high confidence) vs unfiltered predictions.
        """
        fig, ax = plt.subplots(2, 3, figsize=(15, 5))
        for isleep, sleepName in enumerate(self.sleepNames):
            if isleep >= 2:
                break  # Focus on first sleep period
            for i in range(len(self.timeWindows)):
                linearpos = self.resultsNN["linPred"][sleepName][i]
                predloss = self.resultsNN["predLoss"][sleepName][i]

                # All predictions
                ax[isleep, 0].scatter(linearpos, predloss, s=1, c="grey", alpha=0.5)
                ax[isleep, 0].hist2d(
                    linearpos, predloss, (20, 20), cmap=white_viridis, alpha=0.8
                )
                ax[isleep, 0].set_title(f"{sleepName} - All predictions")
                ax[isleep, 0].set_xlabel("Linear position")
                ax[isleep, 0].set_ylabel("Predicted loss")

                # High confidence predictions (loss < threshold)
                filter_high = np.less(predloss, loss_threshold)
                ax[isleep, 1].scatter(
                    linearpos[filter_high],
                    predloss[filter_high],
                    s=1,
                    c="grey",
                    alpha=0.5,
                )
                ax[isleep, 1].hist2d(
                    linearpos[filter_high],
                    predloss[filter_high],
                    (20, 20),
                    cmap=white_viridis,
                    alpha=0.8,
                )
                ax[isleep, 1].set_title(
                    f"{sleepName} - High confidence (loss < {loss_threshold})"
                )
                ax[isleep, 1].set_xlabel("Linear position")
                ax[isleep, 1].set_ylabel("Predicted loss")

                # Low confidence predictions
                filter_low = np.logical_not(filter_high)
                ax[isleep, 2].scatter(
                    linearpos[filter_low],
                    predloss[filter_low],
                    s=1,
                    c="grey",
                    alpha=0.5,
                )
                ax[isleep, 2].hist2d(
                    linearpos[filter_low],
                    predloss[filter_low],
                    (20, 20),
                    cmap=white_viridis,
                    alpha=0.8,
                )
                ax[isleep, 2].set_title(
                    f"{sleepName} - Low confidence (loss >= {loss_threshold})"
                )
                ax[isleep, 2].set_xlabel("Linear position")
                ax[isleep, 2].set_ylabel("Predicted loss")

        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folderFigures, "position_replay_analysis.png"))
        fig.savefig(os.path.join(self.folderFigures, "position_replay_analysis.svg"))

    def fig_position_cumulative_distribution(self, nbins=30, thresh=0.65):
        """
        Compare cumulative distributions of sleep predicted positions
        with wake (true) positions, with optional confidence filtering.
        """
        # Load true wake positions if available
        lineartruePos_wakebeforeSleep = []
        has_wake_data = 1
        for suffix in ["training", "pre", "cond", "post"]:
            try:
                lineartruePosFed = pd.read_csv(
                    os.path.join(
                        self.projectPath.folderResult,
                        str(self.timeWindows[-1]),
                        f"linearTrue_{suffix}.csv",
                    )
                ).values[:, 1:]
                lineartruePos_wakebeforeSleep.append(lineartruePosFed)
                has_wake_data *= 2
            except Exception as e:
                self.logger.warning(f"Could not load wake position data: {e}")
                has_wake_data = -np.abs(has_wake_data)

        has_wake_data = np.abs(has_wake_data) > 1

        lineartruePos_wakebeforeSleep = (
            np.concatenate(lineartruePos_wakebeforeSleep, axis=0)
            if has_wake_data
            else np.array([])
        )

        fig, ax = plt.subplots(
            len(self.sleepNames), 3, figsize=(15, 5 * len(self.sleepNames))
        )
        if len(self.sleepNames) == 1:
            ax = ax[np.newaxis, :]

        for isleep, sleepName in enumerate(self.sleepNames):
            for i in range(len(self.timeWindows)):
                linearpos = self.resultsNN["linPred"][sleepName][i]

                # All sleep positions
                ax[isleep, 0].hist(
                    linearpos,
                    bins=nbins,
                    density=True,
                    label=f"Predicted sleep position (win {self.timeWindows[i]} ms)",
                    cumulative=True,
                    alpha=0.7,
                )
                if has_wake_data and i == 0:
                    ax[isleep, 0].hist(
                        lineartruePos_wakebeforeSleep,
                        bins=nbins,
                        density=True,
                        histtype="step",
                        color="black",
                        cumulative=True,
                        label="Wake position",
                    )
                ax[isleep, 0].set_xlabel("Linear position")
                ax[isleep, 0].set_ylabel("Cumulative probability")
                ax[isleep, 0].set_title(
                    f"{sleepName} - All predictions ({self.timeWindows} ms)"
                )
                ax[isleep, 0].legend()

                # High confidence filter (loss < 0.5)
                predloss = self.resultsNN["predLoss"][sleepName][i]
                filter_high = np.greater(np.max(predloss) - predloss, thresh)
                ax[isleep, 1].hist(
                    linearpos[filter_high],
                    bins=nbins,
                    density=True,
                    label=f"High confidence predictions (loss < {thresh}, win {self.timeWindows[i]} ms)",
                    cumulative=True,
                    alpha=0.7,
                )
                if has_wake_data and i == 0:
                    ax[isleep, 1].hist(
                        lineartruePos_wakebeforeSleep,
                        bins=nbins,
                        density=True,
                        histtype="step",
                        color="black",
                        cumulative=True,
                        label="Wake position",
                    )
                ax[isleep, 1].set_xlabel("Linear position")
                ax[isleep, 1].set_ylabel("Cumulative probability")
                ax[isleep, 1].set_title(
                    f"{sleepName} - High confidence ({self.timeWindows} ms)"
                )
                ax[isleep, 1].legend()

                # Low confidence filter
                filter_low = np.logical_not(filter_high)
                ax[isleep, 2].hist(
                    linearpos[filter_low],
                    bins=nbins,
                    density=True,
                    label=f"Low confidence predictions (loss >= {thresh}, win {self.timeWindows[i]} ms)",
                    cumulative=True,
                    alpha=0.7,
                )
                if has_wake_data and i == 0:
                    ax[isleep, 2].hist(
                        lineartruePos_wakebeforeSleep,
                        bins=nbins,
                        density=True,
                        histtype="step",
                        color="black",
                        cumulative=True,
                        label="Wake position",
                    )
                ax[isleep, 2].set_xlabel("Linear position")
                ax[isleep, 2].set_ylabel("Cumulative probability")
                ax[isleep, 2].set_title(
                    f"{sleepName} - Low confidence ({self.timeWindows} ms)"
                )
                ax[isleep, 2].legend()

        fig.tight_layout()
        fig.show()
        fig.savefig(
            os.path.join(self.folderFigures, "position_cumulative_distribution.png")
        )
        fig.savefig(
            os.path.join(self.folderFigures, "position_cumulative_distribution.svg")
        )

    def fig_ripple_density_vs_confidence(self):
        """Analyze correlation between predicted confidence and ripple density using linear regression."""
        # Create binary ripple indicators for each sleep period and window
        ripple_indicators = {}
        for sleepName in self.sleepNames:
            ripple_indicators[sleepName] = []
            for i in range(len(self.timeWindows)):
                idCloseRipplesInSleep = self.ripples["idCloseRipplesInSleep"][
                    sleepName
                ][i]
                isRipple = np.zeros(
                    len(self.resultsNN["predLoss"][sleepName][i]), dtype=bool
                )
                isRipple[idCloseRipplesInSleep] = True
                ripple_indicators[sleepName].append(isRipple)

        ripple_analysis_utils.plot_ripple_density_vs_confidence(
            predloss_dict=self.resultsNN["predLoss"],
            ripple_indicators_dict=ripple_indicators,
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
        )

    def fig_ripple_density_log_vs_confidence(self):
        """Analyze correlation between predicted confidence and log ripple density."""
        # Create binary ripple indicators for each sleep period and window
        ripple_indicators = {}
        for sleepName in self.sleepNames:
            ripple_indicators[sleepName] = []
            for i in range(len(self.timeWindows)):
                idCloseRipplesInSleep = self.ripples["idCloseRipplesInSleep"][
                    sleepName
                ][i]
                isRipple = np.zeros(
                    len(self.resultsNN["predLoss"][sleepName][i]), dtype=bool
                )
                isRipple[idCloseRipplesInSleep] = True
                ripple_indicators[sleepName].append(isRipple)

        ripple_analysis_utils.plot_ripple_density_log_vs_confidence(
            predloss_dict=self.resultsNN["predLoss"],
            ripple_indicators_dict=ripple_indicators,
            time_windows=self.timeWindows,
            epoch_labels=self.sleepNames,
            folder_figures=self.folderFigures,
        )


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
#
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
#     #     ax[id,1].set_xlabel("times")
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
