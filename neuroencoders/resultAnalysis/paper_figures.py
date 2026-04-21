# Load libs
import logging
import os
import platform
import subprocess
import warnings
from typing import Callable, Dict, List, Optional, Union

import dill as pickle
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns
import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d, sem, zscore
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from statsmodels.stats.proportion import proportions_ztest

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.resultAnalysis.print_results import overview_fig
from neuroencoders.simpleBayes.decode_bayes import (
    Trainer as TrainerBayes,
)
from neuroencoders.simpleBayes.decode_bayes import (
    extract_spike_counts_keops,
    extract_spike_counts_matrix_keops,
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
    EC,
    MIDDLE_COLOR,
    SAFE_COLOR,
    SAFE_COLOR_PREDICTED,
    SHOCK_COLOR,
    SHOCK_COLOR_PREDICTED,
    get_pvalue_stars,
    white_viridis,
)


class PaperFigures:
    def __init__(
        self,
        projectPath: Project,
        behaviorData: dict,
        trainerBayes: Optional[TrainerBayes],
        l_function: Callable,
        bayesMatrices: Optional[dict] = {},
        timeWindows=[36],
        phase=None,
        sleep=False,
        verbose=True,
        **kwargs,
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

        # Verbosity and logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.resultsNN_phase = dict()
        self.resultsBayes_phase = dict()
        self.resultsNN_phase_pkl = dict()
        self.resultsBayes_phase_pkl = dict()
        # Define consistent figure geometry across pages for homogeneous PDF output.
        self.PDF_FIGSIZE = tuple(kwargs.get("pdf_figsize", (12, 16)))
        self.PDF_DPI = int(kwargs.get("pdf_dpi", 300))
        if sleep:
            from neuroencoders.resultAnalysis.paper_figures_sleep import (
                PaperFiguresSleep,
            )

            self.sleepFigures = PaperFiguresSleep(
                projectPath,
                behaviorData,
                trainerBayes,
                l_function,
                bayesMatrices=bayesMatrices,
                timeWindows=timeWindows,
            )

    def _load_csv_result(
        self, base_path: str, ws: int, prefix: str, suffix: str, dtype=np.float32
    ) -> Optional[np.ndarray]:
        """Helper to load a CSV result file and return as numpy array."""
        filepath = os.path.join(base_path, str(ws), f"{prefix}{suffix}.csv")
        if not os.path.exists(filepath):
            return None
        try:
            data = pd.read_csv(filepath).values[:, 1:]
            return np.array(data, dtype=dtype)
        except Exception as e:
            self.logger.warning(f"Error loading {filepath}: {e}")
            return None

    def _prepare_suffixes(self, suffixes: Optional[Union[str, List[str]]]) -> List[str]:
        """Unified suffix list preparation."""
        if suffixes is None:
            suffixes = [self.suffix] if not hasattr(self, "suffixes") else self.suffixes

        if isinstance(suffixes, str):
            suffixes = [suffixes]

        if "_training" in suffixes:
            suffixes.remove("_training")
            suffixes.insert(0, "_training")

        self.suffixes = suffixes
        return suffixes

    def _extract_bayes_spike_counts(self, times, ws):
        """Internal helper for spike count extraction in load_bayes."""
        if not hasattr(self.trainerBayes, "spikeMatTimes") or not hasattr(
            self.trainerBayes, "spikeMatLabels"
        ):
            raise ValueError(
                "trainerBayes missing spike data. Run train_order_by_pos with extract_spike_counts=True."
            )

        total_count, _ = extract_spike_counts_keops(
            times, self.trainerBayes.spikeMatTimes, ws / 1000
        )
        matrix_count, _ = extract_spike_counts_matrix_keops(
            times,
            self.trainerBayes.spikeMatLabels,
            self.trainerBayes.spikeMatTimes,
            ws / 1000,
        )
        return total_count, matrix_count

    def _perform_bayes_test_fallback(self, suffix, ws, i, kwargs):
        """Helper to perform Bayesian testing if results are missing."""
        timesToPredict = self.resultsNN_phase[suffix]["time"][i][:, np.newaxis].astype(
            np.float64
        )
        useTrain = kwargs.get("useTrain", suffix != f"_{self.suffix.strip('_')}")
        useTest = kwargs.get("useTest", suffix != "_training")

        outputsBayes = self.trainerBayes.test_as_NN(
            self.behaviorData,
            self.bayesMatrices,
            timesToPredict,
            windowSizeMS=ws,
            useTrain=useTrain,
            useTest=useTest,
            l_function=self.l_function,
            phase=suffix.strip("_"),
            folderResult=os.path.join(self.projectPath.experimentPath, "results"),
        )

        f_pred = outputsBayes["featurePred"]
        f_true = outputsBayes["featureTrue"]
        l_pred = outputsBayes.get(
            "linearPred", self.l_function(f_pred[:, :2])[1]
        ).flatten()
        l_true = outputsBayes.get(
            "linearTrue", self.l_function(f_true[:, :2])[1]
        ).flatten()

        return {
            "lPredPos": l_pred,
            "lTruePos": l_true,
            "proba": outputsBayes["proba"].flatten(),
            "fPred": f_pred,
            "fTrue": f_true,
            "posLoss": outputsBayes["posLoss"].flatten(),
            "time": outputsBayes["times"].flatten(),
        }

    def _plot_prediction(self, ax, time, true_pos, pred_pos, color, label):
        """Helper to plot ground truth lines and scattered predictions."""
        ax.plot(time, true_pos, c="black", alpha=0.3)
        ax.scatter(time, pred_pos, c=color, alpha=0.9, label=label, s=1)

    def _save_fig(self, fig, filename_base, block=False):
        """Helper to save figure in PNG and SVG formats."""
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        plt.show(block=block)
        for ext in [".png", ".svg"]:
            fig.savefig(os.path.join(self.folderFigures, f"{filename_base}{ext}"))

    def load_data(self, suffixes=None, **kwargs):
        """
        Method to load the results of the neural network prediction.
        """

        if not hasattr(self, "suffixes"):
            self._prepare_suffixes(suffixes)

        base_results_path = os.path.join(self.projectPath.experimentPath, "results")

        for suffix in self.suffixes:
            phase_results = {
                "lPredPos": [],
                "fPredPos": [],
                "truePos": [],
                "lTruePos": [],
                "time": [],
                "lossPred": [],
                "speedMask": [],
                "posIndex": [],
                "spikes_count": [],
            } or []
            resultsNN_phase_pkl = []

            for ws in self.timeWindows:
                # Load standard position and time data
                f_pred = self._load_csv_result(
                    base_results_path, ws, "featurePred", suffix
                )
                f_true = self._load_csv_result(
                    base_results_path, ws, "featureTrue", suffix
                )
                time_steps = self._load_csv_result(
                    base_results_path, ws, "timeStepsPred", suffix
                )
                speed_mask = self._load_csv_result(
                    base_results_path, ws, "speedMask", suffix
                )
                pos_index = self._load_csv_result(
                    base_results_path, ws, "posIndex", suffix, dtype=np.int32
                )

                # Add to lists (with fallback for missing files if needed)
                phase_results["fPredPos"].append(f_pred)
                phase_results["truePos"].append(f_true)
                phase_results["time"].append(
                    np.squeeze(time_steps).flatten() if time_steps is not None else None
                )
                phase_results["speedMask"].append(
                    np.squeeze(speed_mask).flatten() if speed_mask is not None else None
                )
                phase_results["posIndex"].append(
                    np.squeeze(pos_index).flatten() if pos_index is not None else None
                )

                # Linearized positions
                l_true = self._load_csv_result(
                    base_results_path, ws, "linearTrue", suffix
                )
                l_pred = self._load_csv_result(
                    base_results_path, ws, "linearPred", suffix
                )

                if l_true is not None and l_pred is not None:
                    phase_results["lTruePos"].append(np.squeeze(l_true).flatten())
                    phase_results["lPredPos"].append(np.squeeze(l_pred).flatten())
                elif f_true is not None and f_pred is not None:
                    phase_results["lTruePos"].append(
                        self.l_function(f_true[:, :2])[1].flatten()
                    )
                    phase_results["lPredPos"].append(
                        self.l_function(f_pred[:, :2])[1].flatten()
                    )
                else:
                    phase_results["lTruePos"].append(None)
                    phase_results["lPredPos"].append(None)

                # Loss / Entropy
                loss = self._load_csv_result(base_results_path, ws, "lossPred", suffix)
                if loss is None:
                    loss = self._load_csv_result(base_results_path, ws, "Hn", suffix)
                phase_results["lossPred"].append(
                    np.squeeze(loss).flatten() if loss is not None else None
                )

                # Optional Pickle and Spike Counts
                if kwargs.get("load_pickle", False):
                    pkl_path = os.path.join(
                        base_results_path, str(ws), f"decoding_results{suffix}.pkl"
                    )
                    if os.path.exists(pkl_path):
                        with open(pkl_path, "rb") as f:
                            resultsNN_phase_pkl.append(pickle.load(f))
                    else:
                        resultsNN_phase_pkl.append(None)

                if kwargs.get("extract_spike_counts", False):
                    spikes = (
                        pd.read_csv(
                            os.path.join(
                                base_results_path, str(ws), f"spikes_count{suffix}.csv"
                            )
                        )
                        if os.path.exists(
                            os.path.join(
                                base_results_path, str(ws), f"spikes_count{suffix}.csv"
                            )
                        )
                        else None
                    )
                    phase_results["spikes_count"].append(spikes)

            # Post-process speed mask
            phase_results["speedMask"] = [
                sm.astype(bool) if sm is not None else None
                for sm in phase_results["speedMask"]
            ]

            # Store results
            if suffix == self.suffix or len(self.suffixes) == 1:
                self.resultsNN = {
                    "time": phase_results["time"],
                    "speedMask": phase_results["speedMask"],
                    "linPred": phase_results["lPredPos"],
                    "fullPred": phase_results["fPredPos"],
                    "truePos": phase_results["truePos"],
                    "linTruePos": phase_results["lTruePos"],
                    "predLoss": phase_results["lossPred"],
                    "posIndex": phase_results["posIndex"],
                }

            self.resultsNN_phase[suffix] = {
                "time": phase_results["time"],
                "speedMask": phase_results["speedMask"],
                "linPred": phase_results["lPredPos"],
                "fullPred": phase_results["fPredPos"],
                "truePos": phase_results["truePos"],
                "linTruePos": phase_results["lTruePos"],
                "predLoss": phase_results["lossPred"],
                "posIndex": phase_results["posIndex"],
                "spikes_count": phase_results["spikes_count"],
            }
            if kwargs.get("load_pickle", False):
                self.resultsNN_phase_pkl[suffix] = resultsNN_phase_pkl

    def load_bayes(self, suffixes=None, **kwargs):
        """
        Quickly load the bayesian decoding on the data, using the trainerBayes.
        """
        if self.trainerBayes is None:
            raise ValueError(
                "Bayes trainer not loaded. Please load the bayes trainer first."
            )

        if kwargs.get(
            "load_bayesMatrices", False
        ) or self.trainerBayes.config.extra_kwargs.get("load_bayesMatrices", False):
            combined_kwargs = {**self.trainerBayes.config.extra_kwargs, **kwargs}
            self.bayesMatrices = self.trainerBayes.train_order_by_pos(
                self.behaviorData,
                l_function=self.l_function,
                bayesMatrices=self.bayesMatrices
                if (
                    isinstance(self.bayesMatrices, dict)
                    and "Occupation" in self.bayesMatrices
                )
                else None,
                **combined_kwargs,
            )
            if kwargs.get("load_decoded_bayes", False):
                kwargs["redo"] = True
                self._create_decoding_bayes_matrices(**kwargs)

        if not hasattr(self, "suffixes"):
            self._prepare_suffixes(suffixes)

        base_results_path = self.trainerBayes.folderResult

        for suffix in self.suffixes:
            phase_results = {
                "lPredPos": [],
                "lTruePos": [],
                "proba": [],
                "fPred": [],
                "fTrue": [],
                "posLoss": [],
                "time": [],
                "total_spikes": [],
                "matrix_spikes": [],
            } or []
            resultsBayes_phase_pkl = []

            for i, ws in enumerate(self.timeWindows):
                l_pred = self._load_csv_result(
                    base_results_path, ws, "bayes_linearPred", suffix
                )
                if l_pred is not None:
                    phase_results["lPredPos"].append(l_pred.flatten())
                    phase_results["lTruePos"].append(
                        self._load_csv_result(
                            base_results_path, ws, "bayes_linearTrue", suffix
                        ).flatten()
                    )
                    phase_results["proba"].append(
                        self._load_csv_result(
                            base_results_path, ws, "bayes_proba", suffix
                        ).flatten()
                    )
                    phase_results["fPred"].append(
                        self._load_csv_result(
                            base_results_path, ws, "bayes_featurePred", suffix
                        )
                    )
                    phase_results["fTrue"].append(
                        self._load_csv_result(
                            base_results_path, ws, "bayes_featureTrue", suffix
                        )
                    )
                    phase_results["posLoss"].append(
                        self._load_csv_result(
                            base_results_path, ws, "bayes_posLoss", suffix
                        ).flatten()
                    )
                    phase_results["time"].append(
                        self._load_csv_result(
                            base_results_path, ws, "bayes_timeStepsPred", suffix
                        ).flatten()
                    )

                    if kwargs.get("extract_spike_counts", False):
                        total_count, matrix_count = self._extract_bayes_spike_counts(
                            phase_results["time"][-1], ws
                        )
                        phase_results["total_spikes"].append(total_count)
                        phase_results["matrix_spikes"].append(matrix_count)

                    if (
                        phase_results["fPred"][-1].shape[0]
                        != self.resultsNN_phase[suffix]["fullPred"][i].shape[0]
                    ):
                        self.logger.warning(
                            f"Shape mismatch for window {ws}ms. Bayesian: {phase_results['fPred'][-1].shape}, NN: {self.resultsNN_phase[suffix]['fullPred'][i].shape}"
                        )
                else:
                    self.logger.info(
                        f"Bayesian results not found for {ws}ms, testing now..."
                    )
                    test_results = self._perform_bayes_test_fallback(
                        suffix, ws, i, kwargs
                    )
                    phase_results["lPredPos"].append(test_results["lPredPos"])
                    phase_results["lTruePos"].append(test_results["lTruePos"])
                    phase_results["proba"].append(test_results["proba"])
                    phase_results["fPred"].append(test_results["fPred"])
                    phase_results["fTrue"].append(test_results["fTrue"])
                    phase_results["posLoss"].append(test_results["posLoss"])
                    phase_results["time"].append(test_results["time"])

                if kwargs.get("load_pickle", False):
                    pkl_path = os.path.join(
                        base_results_path,
                        str(ws),
                        f"bayes_decoding_results{suffix}.pkl",
                    )
                    if os.path.exists(pkl_path):
                        with open(pkl_path, "rb") as f:
                            resultsBayes_phase_pkl.append(pickle.load(f))
                    else:
                        resultsBayes_phase_pkl.append(None)

            # Store results
            result_dict = {
                "linPred": phase_results["lPredPos"],
                "linTruePos": phase_results["lTruePos"],
                "fullPred": phase_results["fPred"],
                "truePos": phase_results["fTrue"],
                "predLoss": phase_results[
                    "proba"
                ],  # ProbaBayes mapped to predLoss for NN compatibility
                "posLossBayes": phase_results["posLoss"],
                "timeNN": self.resultsNN_phase[suffix]["time"],
                "time": phase_results["time"],
                "speedMask": self.resultsNN_phase[suffix]["speedMask"],
            }
            if kwargs.get("extract_spike_counts", False):
                result_dict.update(
                    {
                        "total_spikes_count": phase_results["total_spikes"],
                        "matrix_spikes_count": phase_results["matrix_spikes"],
                    }
                )

            self.resultsBayes_phase[suffix] = result_dict
            if suffix == self.suffix or len(self.suffixes) == 1:
                self.resultsBayes = result_dict
            if kwargs.get("load_pickle", False):
                self.resultsBayes_phase_pkl[suffix] = resultsBayes_phase_pkl

    def _create_decoding_bayes_matrices(
        self, winMS=None, suffix=None, phase=None, **kwargs
    ):
        """
        Run the bayes trainer, not with the true positions but rather the predictions of the ANN, and create the bayes matrices from those predictions. This allows to have a fair comparison between the two methods, as they will be based on the same input data (the ANN predictions) rather than the true positions, which may be more accurate than what the ANN can achieve.
        """
        if not self.trainerBayes:
            raise ValueError(
                "Bayes trainer not loaded. Please load the bayes trainer first."
            )
        if not self.bayesMatrices or self.bayesMatrices is None:
            combined_kwargs = {**self.trainerBayes.config.extra_kwargs, **kwargs}
            self.bayesMatrices = self.trainerBayes.train_order_by_pos(
                self.behaviorData,
                l_function=self.l_function,
                bayesMatrices=self.bayesMatrices
                if (
                    isinstance(self.bayesMatrices, dict)
                    and "Occupation" in self.bayesMatrices
                )
                else None,
                **combined_kwargs,
            )

        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if isinstance(suffix, str) and not suffix.startswith("_"):
            suffix = f"_{suffix}"
        if not isinstance(suffix, list):
            suffix = [suffix]

        if winMS is None:
            winMS = self.timeWindows[-1]
        idWindow = self.timeWindows.index(int(winMS))

        # get the ANN predictions for the specified window size in the training phase
        self.load_data(winMS=winMS, suffixes=suffix, which="ann", **kwargs)
        true_pos_list = []
        predicted_list = []
        time_step_pred_list = []
        speed_mask_list = []
        for suff in suffix:
            true_pos = self.resultsNN_phase[suff]["truePos"][idWindow]
            predicted = self.resultsNN_phase[suff]["fullPred"][idWindow]
            time_step_pred = self.resultsNN_phase[suff]["time"][idWindow].reshape(-1, 1)
            speed_mask = self.resultsNN_phase[suff]["speedMask"][idWindow].flatten()
            true_pos_list.append(true_pos)
            predicted_list.append(predicted)
            time_step_pred_list.append(time_step_pred)
            speed_mask_list.append(speed_mask)

        true_pos = np.array(true_pos_list).reshape(-1, true_pos.shape[1])
        predicted = np.array(predicted_list).reshape(-1, predicted.shape[1])
        time_step_pred = np.array(time_step_pred_list).reshape(-1, 1)
        speed_mask = np.array(speed_mask_list).reshape(-1)

        trainEpochs = [time_step_pred.min(), time_step_pred.max()]

        # Get learning time and if needed speedFilter
        samplingWindowPosition = (time_step_pred[1:] - time_step_pred[0:-1])[:, 0]
        samplingWindowPosition[np.isnan(np.sum(predicted[0:-1], axis=1))] = 0
        lEpochIndex = [
            [
                np.argmin(np.abs(time_step_pred - trainEpochs[2 * i + 1])),
                np.argmin(np.abs(time_step_pred - trainEpochs[2 * i])),
            ]
            for i in range(len(trainEpochs) // 2)
        ]
        learningTime = [
            np.sum(
                np.multiply(
                    speed_mask[lEpochIndex[i][1] : lEpochIndex[i][0]],
                    samplingWindowPosition[lEpochIndex[i][1] : lEpochIndex[i][0]],
                )
            )
            for i in range(len(lEpochIndex))
        ]
        learningTime = np.sum(learningTime)

        ## create a fake fullBehavior dict to feed to bayes.train_order_by_pos
        self.decoded_fullBehavior = {
            "positionTime": time_step_pred,
            "Positions": predicted,
            "groundTruth": true_pos,
            "Times": {
                "speedFilter": speed_mask,
                "trainEpochs": trainEpochs,
                "testEpochs": trainEpochs,
                "learning": learningTime,
            },
        }

        self.decoded_bayesMatrices = self.trainerBayes.train_order_by_pos(
            self.decoded_fullBehavior,
            l_function=self.l_function,
            is_predicted=True,
            winMS=winMS,
            **kwargs,
        )
        return self.decoded_bayesMatrices

    def plot_linear_tuning_curves(
        self,
        fullBehavior: Optional[Dict] = None,
        l_function: Optional[Callable] = None,
        use_speed_filter: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        sort_map: Optional[List[int]] = None,
        **kwargs,
    ):
        if fullBehavior is None:
            fullBehavior = self.behaviorData
        if l_function is None:
            l_function = self.l_function
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        else:
            fig = ax.get_figure()

        calc_kwargs = dict(kwargs)
        cax = calc_kwargs.pop("cax", None)
        add_colorbar = calc_kwargs.pop("add_colorbar", True)
        normalize = calc_kwargs.pop("normalize", True)
        scaling_method = calc_kwargs.pop("scaling", "minmax")
        mask = calc_kwargs.pop("mask", None)
        title = calc_kwargs.pop("title", "Linear Tuning Curves")

        lin_place_fields, bin_edges = self.trainerBayes.calculate_linear_tuning_curve(
            l_function=l_function,
            behaviorData=fullBehavior,
            use_speed_filter=use_speed_filter,
            **calc_kwargs,
        )
        if sort_map is None:
            preferred_linear_positions = []
            for tuning_curve in lin_place_fields:
                if np.any(tuning_curve > 0):
                    peak_idx = np.argmax(tuning_curve)
                    preferred_pos = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                    preferred_linear_positions.append(preferred_pos)
                else:
                    preferred_linear_positions.append(bin_edges[0])

            preferred_linear_positions = np.array(preferred_linear_positions)
            linear_pos_argsort = np.argsort(preferred_linear_positions)
        else:
            linear_pos_argsort = sort_map

        ordered_lin_place_fields = np.array(lin_place_fields)[linear_pos_argsort]

        if normalize:
            if scaling_method == "z-score":
                # Safe Z-score normalization per neuron (row-wise)
                mean_vals = np.mean(ordered_lin_place_fields, axis=1, keepdims=True)
                std_val = np.std(ordered_lin_place_fields, axis=1, keepdims=True)
                fields = (ordered_lin_place_fields - mean_vals) / (std_val + 1e-8)

                # Plotting Setup for Z-Score
                cmap = "RdBu_r"
                v_lim = min(np.percentile(np.abs(fields), 99), 4)
                norm = mcolors.TwoSlopeNorm(vmin=-v_lim, vcenter=0, vmax=v_lim)
                cb_label = "Z-Scored FR"
            elif scaling_method == "minmax":
                # Min-Max normalization per neuron (row-wise)
                min_vals = np.min(ordered_lin_place_fields, axis=1, keepdims=True)
                max_vals = np.max(ordered_lin_place_fields, axis=1, keepdims=True)
                fields = (ordered_lin_place_fields - min_vals) / (
                    max_vals - min_vals + 1e-8
                )

                # Plotting Setup for Min-Max
                cmap = "viridis"
                norm = mcolors.Normalize(vmin=0, vmax=1)
                cb_label = "Normalized Firing Rate (0-1)"
            else:
                raise ValueError(
                    f"Unknown scaling method: {scaling_method}. Use 'z-score' or 'minmax'."
                )
        else:
            fields = ordered_lin_place_fields
            cmap = "viridis"
            norm = mcolors.Normalize(vmin=np.min(fields), vmax=np.max(fields))
            cb_label = "Firing Rate"

        fields = fields[mask] if mask is not None else fields
        im = ax.imshow(fields, cmap, norm, origin="lower")
        linear_bins = np.linspace(0, 1, fields.shape[1])
        x_ticks = np.arange(0, fields.shape[1], 20)
        x_labels = np.round(linear_bins[x_ticks], 2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Linear Position")
        ax.set_ylabel(f"Neuron Index ({len(ordered_lin_place_fields)} neurons)")
        ax.set_title(title)
        if add_colorbar:
            if cax is not None:
                cbar = fig.colorbar(
                    im,
                    cax=cax,
                    label=cb_label,
                    orientation="horizontal",
                    location="bottom",
                )
            else:
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    label=cb_label,
                    location="bottom",
                    orientation="horizontal",
                )

            cbar.outline.set_visible(False)

        return im, cb_label

    def fig_example_XY(self, timeWindow, suffix=None, phase=None, block=False):
        idWindow = self.timeWindows.index(timeWindow)
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        suffix = f"_{phase}" if phase is not None else (suffix or self.suffix)

        time = self.resultsNN_phase[suffix]["time"][idWindow]
        true_pos = self.resultsNN_phase[suffix]["truePos"][idWindow]
        nn_pred = self.resultsNN_phase[suffix]["fullPred"][idWindow]
        bayes_pred = self.resultsBayes_phase[suffix]["fullPred"][idWindow]

        for idim, label in enumerate(["X", "Y"]):
            self._plot_prediction(
                ax[idim, 0],
                time,
                true_pos[:, idim],
                nn_pred[:, idim],
                self.cm(12 + idWindow),
                f"{timeWindow} ms",
            )
            self._plot_prediction(
                ax[idim, 1],
                time,
                true_pos[:, idim],
                bayes_pred[:, idim],
                self.cm(idWindow),
                f"{timeWindow} ms",
            )
            ax[idim, 0].set_ylabel(label, fontsize="xx-large")

        ax[0, 0].set_title(
            f"Neural network decoder \n {timeWindow} window", fontsize="xx-large"
        )
        ax[0, 1].set_title(
            f"Bayesian decoder \n {timeWindow} window", fontsize="xx-large"
        )
        ax[1, 0].set_xlabel("Time (s)", fontsize="xx-large")
        ax[1, 1].set_xlabel("Time (s)", fontsize="xx-large")
        fig.suptitle(f"2D decoding for phase {suffix.strip('_')}", fontsize="xx-large")

        self._save_fig(fig, f"example2D_nn_bayes_{timeWindow}ms{suffix}", block=block)

    def fig_example_linear(self, suffix=None, phase=None, block=False):
        suffix = f"_{phase}" if phase is not None else (suffix or self.suffix)
        fig, ax = plt.subplots(
            len(self.timeWindows), 2, sharex=True, sharey=True, squeeze=False
        )

        for i, ws in enumerate(self.timeWindows):
            time = self.resultsNN_phase[suffix]["time"][i]
            true_pos = self.resultsNN_phase[suffix]["linTruePos"][i]
            nn_pred = self.resultsNN_phase[suffix]["linPred"][i]
            bayes_pred = self.resultsBayes_phase[suffix]["linPred"][i]

            # Neural Network plot
            self._plot_prediction(
                ax[i, 0], time, true_pos, nn_pred, self.cm(12 + i), f"{ws} ms"
            )
            title_nn = (
                f"Neural network decoder \n {ws} window" if i == 0 else f"{ws} window"
            )
            ax[i, 0].set_title(title_nn, fontsize="xx-large")
            ax[i, 0].set_ylabel("linear position", fontsize="xx-large")
            ax[i, 0].set_yticks([0, 0.4, 0.8])

            # Bayesian plot
            self._plot_prediction(
                ax[i, 1], time, true_pos, bayes_pred, self.cm(i), f"{ws} ms"
            )
            title_bayes = (
                f"Bayesian decoder \n {ws} window" if i == 0 else f"{ws} window"
            )
            ax[i, 1].set_title(title_bayes, fontsize="xx-large")

        ax[-1, 0].set_xlabel("time (s)", fontsize="xx-large")
        ax[-1, 1].set_xlabel("time (s)", fontsize="xx-large")
        fig.suptitle(
            f"Linear position decoding for phase {suffix.strip('_')}",
            fontsize="xx-large",
        )

        self._save_fig(fig, f"example_nn_bayes{suffix}", block=block)

    def compare_nn_bayes(
        self, timeWindow, suffix=None, phase=None, isCM=False, isShow=False, block=False
    ):
        idWindow = self.timeWindows.index(timeWindow)
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        # Data
        if isCM:
            nnD = self.resultsNN_phase[suffix]["fullPred"][idWindow][:, :2] * EC
            bayesD = self.resultsBayes_phase[suffix]["fullPred"][idWindow][:, :2] * EC
            title = "Euclidian distance (cm)"
        else:
            nnD = self.resultsNN_phase[suffix]["fullPred"][idWindow][:, :2]
            bayesD = self.resultsBayes_phase[suffix]["fullPred"][idWindow][:, :2]
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
        save: bool = True,
        extended_zone: bool = True,
        convex_hull: bool = True,
        **kwargs,
    ):
        num_sess_post = kwargs.get("num_sess_post", 3)
        num_sess = sum(
            [
                1
                for _, sess_time in DataHelper.fullBehavior["Times"][
                    "SessionEpochs"
                ].items()
                if sum(sess_time) > 0
            ]
        )

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
                key=lambda item: (
                    sort_map.index(item[0]) if item[0] in sort_map else len(sort_map)
                ),
            )
        )
        after_cond = False
        for i, (sess_name, sess_time) in enumerate(
            DataHelper.fullBehavior["Times"]["SessionEpochs"].items()
        ):
            if not sess_time:
                warnings.warn(
                    f"Session {sess_name} has no time epochs, skipping plotting and analysis for this session."
                )
                continue  # skip empty sessions
            if sess_name.lower() == "cond":
                after_cond = True
            # 1. Data Filtering
            if sess_name.lower() == "post" or sess_name.lower() == "extinction":
                sess_time = np.array(sess_time).reshape(-1, 2)
                sess_time = sess_time[:num_sess_post]

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

            shock_zone = DataHelper.create_zone_polygon_from_borders(
                ZONEDEF[ZONELABELS.index("Shock")]
            )
            safe_zone = DataHelper.create_zone_polygon_from_borders(
                ZONEDEF[ZONELABELS.index("Safe")]
            )

            shock_count = np.sum(shock_mask)
            safe_count = np.sum(safe_mask)

            shock_occupancy = (
                (DataHelper.polygon.area * shock_count)
                / (total_time_points * shock_zone.area)
                if total_time_points > 0
                else 0
            )
            safe_occupancy = (
                (DataHelper.polygon.area * safe_count)
                / (total_time_points * safe_zone.area)
                if total_time_points > 0
                else 0
            )

            if extended_zone:
                safe_mask_extended = safe_mask | is_in_zone(
                    pos, ZONEDEF[ZONELABELS.index("SafeCenter")]
                )
                safe_zone_extended = unary_union(
                    [
                        safe_zone,
                        DataHelper.create_zone_polygon_from_borders(
                            ZONEDEF[ZONELABELS.index("SafeCenter")]
                        ),
                    ]
                )
                safe_count_extended = np.sum(safe_mask_extended)
                safe_occupancy_extended = (
                    (DataHelper.polygon.area * safe_count_extended)
                    / (safe_zone_extended.area * total_time_points)
                    if total_time_points > 0
                    else 0
                )

            if convex_hull and stim_mask.sum() > 0:
                stim_pos_array = DataHelper.fullBehavior["Positions"][stim_mask, :2]
                # clustering
                clustering = DBSCAN(eps=0.15, min_samples=1).fit(
                    DataHelper.fullBehavior["Positions"][stim_mask, :2]
                )
                labels = clustering.labels_
                hulls = []
                unique_labels = set(labels)

                for label in unique_labels:
                    cluster_points = stim_pos_array[labels == label]
                    points_geom = MultiPoint(cluster_points)
                    buffered_hull = points_geom.buffer(0.04).convex_hull
                    hulls.append(buffered_hull)

                total_shock_zone = unary_union(hulls)
                pos_shapely = [Point(xy) for xy in pos]
                shock_mask_convex_mask = [
                    total_shock_zone.contains(p) for p in pos_shapely
                ]
                shock_convex_count = np.sum(shock_mask_convex_mask)
                shock_occupancy_convex = (
                    (DataHelper.polygon.area * shock_convex_count)
                    / (total_shock_zone.area * total_time_points)
                    if total_time_points > 0
                    else 0
                )

                # as a control/comparison, also calculate occupancy for the convex hull of the safe zone (create symetric hull w small jitter)
                # take all shock hulls, and simply do a symmetry wrt x coordinates (symmetric = 1 - x) with a small jitter to avoid perfect overlap
                safe_hulls = []
                for h in hulls:
                    x_hull, y_hull = h.exterior.xy
                    x_hull, y_hull = np.array(x_hull), np.array(y_hull)
                    x_hull_sym = (
                        np.ones_like(x_hull)
                        - x_hull
                        + np.random.uniform(-0.02, 0.02, size=x_hull.shape)
                    )
                    y_hull_sym = y_hull + np.random.uniform(
                        -0.02, 0.02, size=y_hull.shape
                    )
                    safe_hulls.append(Polygon(zip(x_hull_sym, y_hull_sym)).convex_hull)
                total_safe_zone = unary_union(safe_hulls)
                safe_mask_convex_mask = [
                    total_safe_zone.contains(p) for p in pos_shapely
                ]
                safe_convex_count = np.sum(safe_mask_convex_mask)
                safe_occupancy_convex = (
                    (DataHelper.polygon.area * safe_convex_count)
                    / (total_safe_zone.area * total_time_points)
                    if total_time_points > 0
                    else 0
                )

            # Perform Two-Sample Z-Test for Proportions (Shock count vs Safe count)
            # This is the statistically correct way to compare two proportions based on counts.
            if total_time_points > 0:
                counts = np.array([safe_count, shock_count])
                # N must be total time points for both groups
                nobs = np.array([total_time_points, total_time_points])

                if extended_zone:
                    counts = np.concatenate((counts, [safe_count_extended]))
                    nobs = np.concatenate((nobs, [total_time_points]))

                if convex_hull and stim_mask.sum() > 0:
                    counts = np.concatenate(
                        (counts, [shock_convex_count, safe_convex_count])
                    )
                    nobs = np.concatenate(
                        (nobs, [total_time_points, total_time_points])
                    )

                # Check if there is enough variance to run the test
                if np.sum(nobs) > 0 and np.all(counts > 0) and not extended_zone:
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

            if extended_zone:
                session_occupancy_data[-1]["ExtSafe Occupancy"] = (
                    safe_occupancy_extended
                )
            if convex_hull and stim_mask.sum() > 0:
                session_occupancy_data[-1]["CvxShock Occupancy"] = (
                    shock_occupancy_convex
                )
                session_occupancy_data[-1]["CvxSafe Occupancy"] = safe_occupancy_convex

            # 3. Map Plotting (First Row)
            map_ax = map_axs[i]
            H_true, xedges, yedges = np.histogram2d(
                pos[:, 0], pos[:, 1], bins=40, range=[[0, 1], [0, 1]]
            )
            occupancy_map = gaussian_filter(H_true.T, sigma=2)

            map_ax.plot(
                pos[:, 0],
                pos[:, 1],
                c="xkcd:grey",
                alpha=0.6,
                zorder=2,
                linewidth=0.5,
            )  # Traces
            map_ax.imshow(
                occupancy_map,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=cmap_custom,
                vmin=1,
                zorder=0,
                alpha=0.6,
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

                if convex_hull and stim_mask.sum() > 0:
                    for h in hulls:
                        x_hull, y_hull = h.exterior.xy
                        map_ax.plot(x_hull, y_hull, color=SHOCK_COLOR, linewidth=2)
                        map_ax.fill(
                            x_hull,
                            y_hull,
                            facecolor=SHOCK_COLOR,
                            alpha=0.2,
                        )
                    for h in safe_hulls:
                        x_hull, y_hull = h.exterior.xy
                        map_ax.plot(x_hull, y_hull, color=SAFE_COLOR, linewidth=2)
                        map_ax.fill(
                            x_hull,
                            y_hull,
                            facecolor=SAFE_COLOR,
                            alpha=0.2,
                        )

            map_ax.set_title(f"{sess_name}")
            if sess_name.lower() == "cond":
                map_ax.set_title(f"{sess_name} (n_stims = {np.sum(stim_mask)})")
            if sess_name.lower() == "post" or sess_name.lower() == "extinction":
                map_ax.set_title(f"{sess_name} (n_sess = {len(sess_time)})")
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

        if extended_zone:
            max_occupancy = max(
                max_occupancy,
                max([d.get("ExtSafe Occupancy", 0) for d in session_occupancy_data])
                * 1.25,
            )

        if convex_hull and stim_mask.sum() > 0:
            max_occupancy = max(
                max_occupancy,
                max([d.get("CvxShock Occupancy", 0) for d in session_occupancy_data])
                * 1.25,
            )
            max_occupancy = max(
                max_occupancy,
                max([d.get("CvxSafe Occupancy", 0) for d in session_occupancy_data])
                * 1.25,
            )

        for i, data in enumerate(session_occupancy_data):
            bar_ax = bar_axs[i]

            # Data points for this session
            zones = ["Shock", "Safe"]
            occupancy_values = [data["Shock Occupancy"], data["Safe Occupancy"]]

            colors = [SHOCK_COLOR, SAFE_COLOR]

            if extended_zone:
                zones.append("ExtSafe")
                occupancy_values.append(data["ExtSafe Occupancy"])
                colors.append(SAFE_COLOR_PREDICTED)

            if convex_hull and stim_mask.sum() > 0:
                zones.append("CvxShock")
                occupancy_values.append(data["CvxShock Occupancy"])
                colors.append(SHOCK_COLOR_PREDICTED)
                zones.append("CvxSafe")
                occupancy_values.append(data["CvxSafe Occupancy"])
                colors.append(SAFE_COLOR_PREDICTED)

            sorted_list = ["Shock", "CvxShock", "CvxSafe", "Safe", "Extsafe"]
            order_map = {zone: i for i, zone in enumerate(sorted_list)}
            combined = zip(zones, occupancy_values, colors)
            sorted_combined = sorted(
                combined,
                key=lambda x: order_map.get(x[0], 999),
            )
            zones, occupancy_values, colors = map(list, zip(*sorted_combined))

            # Plot the bar chart
            bar_ax.bar(zones, occupancy_values, color=colors)
            bar_ax.axhline(
                1,  # expected occupancy if exploration is uniform
                linestyle="--",
                color="gray",
                lw=1.6,
            )

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
            bar_ax.set_xticks(range(len(zones)))
            bar_ax.set_xticklabels(zones, rotation=45, ha="right")

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
            if (
                fig.get_tight_layout() is None
            ):  # Only adjust if tight_layout hasn't been called yet
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show(block=block)
        if save:
            if (
                fig.get_tight_layout() is None
            ):  # Only adjust if tight_layout hasn't been called yet
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        cax=None,
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
                layout="constrained",
            )
            if len(suffix_list) == 1:
                axs = np.array([[axs]])
        else:
            axs = np.array(axs).flatten()
            fig = None

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

            errormat = axs[i].imshow(
                mean_error_matrix.T,
                origin="lower",
                cmap=cmap,
                extent=[0, 1, 0, 1],
                norm=mcolors.Normalize(
                    vmin=0,
                    vmax=1 if error_type == "lin" else np.sqrt(2),
                ),
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

        curr_fig = axs[0].get_figure()
        if cax is not None:
            cbar = curr_fig.colorbar(
                errormat,
                cax=cax,
                orientation="horizontal",
            )
        else:
            cbar = curr_fig.colorbar(
                errormat,
                ax=axs[:],
                shrink=0.5,
                location="top",
                orientation="horizontal",
                pad=0.1,
            )

        cbar.outline.set_visible(False)

        if show:
            fig.suptitle(
                f"Error map ({error_type}) for {timeWindow}ms window",
            )
            if (
                fig.get_tight_layout() is None
            ):  # Only adjust if tight_layout hasn't been called yet
                fig.tight_layout()
            plt.show(block=block)
        if save:
            if (
                fig.get_tight_layout() is None
            ):  # Only adjust if tight_layout hasn't been called yet
                fig.tight_layout()
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
        open: bool = True,
        dimOutput: int = 1,
        **kwargs,
    ):
        """
        Summary figure saved as a multipage PDF.
        Page 1: Behavior, Error Maps, Correlations, Summary Stats
        Page 2+: Trajectories (1 suffix per page for detailed view)
        """

        if kwargs.get("mouse_name", None) is not None:
            title_content = f"M{kwargs.get('mouse_name')} - "
        else:
            title_content = f"{os.path.basename(self.projectPath.experimentPath)} - "
        idWindow = self.timeWindows.index(timeWindow)

        # 1. Setup PDF Path
        filename = f"summary_id_card_{timeWindow}ms.pdf"
        save_path = os.path.join(self.folderFigures, filename)
        print(f"Generating PDF at: {save_path}")

        # Compute Entropy Threshold from Training Data
        thresh_entropy = self._get_entropy_threshold(idWindow, kwargs)

        # Define consistent figure geometry across pages for homogeneous PDF output.
        self.PDF_FIGSIZE = tuple(kwargs.get("pdf_figsize", (12, 16)))
        self.PDF_DPI = int(kwargs.get("pdf_dpi", 300))

        with PdfPages(save_path) as pdf:
            # =========================================================
            # PAGE 1: Summary Behavior + Error Maps
            # =========================================================
            self._plot_summary_page_1(
                timeWindow, DataHelper, title_content, kwargs, pdf=pdf
            )

            # =========================================================
            # PAGE 2: Correlations, Histograms & Boxplots
            # =========================================================
            self._plot_summary_page_2(
                timeWindow, idWindow, thresh_entropy, kwargs, pdf=pdf
            )

            # =========================================================
            # PAGES 2+: Trajectories
            # =========================================================
            self._plot_summary_trajectories(
                timeWindow,
                idWindow,
                dimOutput,
                thresh_entropy,
                kwargs,
                pdf=pdf,
            )

            # =========================================================
            # PAGES after+: Link with bayesian decoder and spike sorting
            # =========================================================
            self._plot_summary_bayesian(timeWindow, pdf=pdf)

        print("PDF generation complete.")

        # =========================================================
        # OPEN THE PDF
        # =========================================================
        if open:
            try:
                if platform.system() == "Darwin":
                    subprocess.call(["open", save_path])
                elif platform.system() == "Windows":
                    os.startfile(save_path)
                else:
                    subprocess.call(["xdg-open", save_path])
            except Exception as e:
                print(f"Could not open PDF: {e}")

    def _get_entropy_threshold(self, idWindow, kwargs):
        try:
            train_entropy = self.resultsNN_phase["_training"]["predLoss"][idWindow]
            if train_entropy is not None:
                return np.percentile(train_entropy, kwargs.get("threshold", 20))
        except KeyError:
            print(
                "Warning: No training entropy found. Certainty plots will be skipped or empty."
            )
        return None

    def _plot_summary_page_1(
        self, timeWindow, DataHelper, title_content, kwargs, pdf=None
    ):
        fig1 = plt.figure(figsize=self.PDF_FIGSIZE, constrained_layout=False)
        fig1.set_size_inches(*self.PDF_FIGSIZE, forward=True)
        fig1.set_rasterized(True)

        gs = gridspec.GridSpec(7, 3, figure=fig1)

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
            extended_zone=kwargs.get("extended_zone", True),
            convex_hull=kwargs.get("convex_hull", True),
        )

        # --- Row 3: Text ---
        ax_text = fig1.add_subplot(gs[3, :])
        ax_text.axis("off")
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
        start_row_map = 4
        nrows_map = 2
        ncols_map = len(self.suffixes)
        gs_error_maps = gs[start_row_map : start_row_map + nrows_map, :].subgridspec(
            nrows_map, ncols_map, height_ratios=[0.1, 0.9]
        )
        cax = fig1.add_subplot(
            gs_error_maps[0, ncols_map // 2 - 1 : ncols_map // 2 + 1]
        )

        gs_inner_maps = gs_error_maps[1:, :].subgridspec(1, ncols_map)
        target_axs = [fig1.add_subplot(gs_inner_maps[:, j]) for j in range(ncols_map)]

        self.error_map(
            timeWindow,
            suffix_list=self.suffixes,
            axs=target_axs,
            cax=cax,
            show=False,
            block=False,
            save=False,
            error_type=kwargs.get("error_type", "lin"),
        )

        fig1.suptitle(
            f"Summary: {title_content} ({timeWindow}ms)",
            fontsize=14,
        )
        fig1.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.04,
            top=0.94,
            wspace=0.22,
            hspace=0.55,
        )
        if pdf is not None:
            pdf.savefig(fig1, dpi=self.PDF_DPI, bbox_inches=None)
        else:
            plt.show()
        plt.close(fig1)

    def _plot_summary_page_2(
        self, timeWindow, idWindow, thresh_entropy, kwargs, pdf=None
    ):
        fig1_bis = plt.figure(figsize=self.PDF_FIGSIZE, constrained_layout=False)
        fig1_bis.set_size_inches(*self.PDF_FIGSIZE, forward=True)
        fig1_bis.set_rasterized(True)

        paired = kwargs.get("paired", False)
        num_suffixes = len(self.suffixes)
        num_rows = (num_suffixes + 1) // 4
        if num_rows == 0:
            num_rows = 1

        n_total_rows = 3 + num_rows
        unif_height_ratio = 1 / n_total_rows
        gs_bis = gridspec.GridSpec(
            n_total_rows,
            4,
            figure=fig1_bis,
            height_ratios=[unif_height_ratio / num_rows] * num_rows
            + [unif_height_ratio * 2]
            + [unif_height_ratio] * 2,
            # oversampling or correlation plot,
            # histograms
            # barplot + violinplot
        )

        # --- Row 0: Oversampling summary from training positions ---
        succeeded = self._plot_oversampling_summary(
            timeWindow, idWindow, fig1_bis, gs_bis, num_rows
        )
        if not succeeded:
            # --- Row 0-n: Correlation Plots ---
            gs_corr = gs_bis[0:num_rows, :].subgridspec(num_rows, 4)
            for i, suff in enumerate(self.suffixes):
                row_idx = i % num_rows
                col_idx = i // num_rows
                ax = fig1_bis.add_subplot(gs_corr[row_idx, col_idx])
                self._plot_correlation_scatter(
                    suff, idWindow, ax=ax, show_legend=(i == 0)
                )

        # --- Prepare Data for Bar/Violin Plots ---
        data_box, labels_box, colors_box, phase_box, cond_box = (
            self._prepare_boxplot_data(idWindow, thresh_entropy)
        )

        # --- Row 4: Training Histograms ---
        self._plot_prediction_histograms(
            timeWindow, idWindow, fig1_bis, gs_bis, num_rows
        )

        # --- Row 5+: Barplot & Violinplot ---
        gs_barplot = gs_bis[1 + num_rows : 3 + num_rows, :].subgridspec(2, 1)

        # Barplot
        ax_mean_barplot = fig1_bis.add_subplot(gs_barplot[0, 0])
        if data_box:
            df_box = pd.DataFrame(
                {
                    "Linear Error": np.concatenate(data_box),
                    "Phase": np.concatenate(
                        [[suff] * len(data) for suff, data in zip(phase_box, data_box)]
                    ),
                    "Speeds/Certainty": np.concatenate(
                        [[cond] * len(data) for cond, data in zip(cond_box, data_box)]
                    ),
                    "Color": np.concatenate(
                        [
                            [color] * len(data)
                            for color, data in zip(colors_box, data_box)
                        ]
                    ),
                }
            )
            sns.barplot(
                data=df_box,
                errorbar="se",
                x="Phase",
                y="Linear Error",
                hue="Speeds/Certainty",
                palette=df_box.drop_duplicates("Speeds/Certainty")
                .set_index("Speeds/Certainty")["Color"]
                .to_dict(),
                alpha=0.7,
                ax=ax_mean_barplot,
            )
            ax_mean_barplot.set_ylabel("Mean Linear Error")
            ax_mean_barplot.set_xlabel("")
            ax_mean_barplot.set_title("Mean Linear Error across Phases")
            # Move legend to upper right outside the plot
            ax_mean_barplot.legend(
                title="Condition",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )

        # Violinplot
        ax_box = fig1_bis.add_subplot(gs_barplot[1, 0])
        if data_box:
            self._plot_violin(
                data_box, labels_box, colors_box, ax=ax_box, paired=paired
            )

        fig1_bis.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.04,
            top=0.95,
            wspace=0.32,
            hspace=0.8,
        )
        if pdf is not None:
            pdf.savefig(fig1_bis, dpi=self.PDF_DPI, bbox_inches=None)
        else:
            plt.show()
        plt.close(fig1_bis)

    def _plot_correlation_scatter(self, suff, idWindow, ax=None, show_legend=False):
        if (
            self.resultsNN_phase[suff]["linPred"][idWindow] is None
            or self.resultsNN_phase[suff]["predLoss"][idWindow] is None
        ):
            return

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        lin_err = np.abs(
            self.resultsNN_phase[suff]["linPred"][idWindow]
            - self.resultsNN_phase[suff]["linTruePos"][idWindow]
        )
        entropy = self.resultsNN_phase[suff]["predLoss"][idWindow]
        maxp = self.resultsNN_phase_pkl[suff][idWindow]["maxp"]

        saut = 40
        ax.plot(
            entropy[::saut],
            lin_err[::saut],
            "o",
            color="blue",
            alpha=0.6,
            markersize=3,
        )
        new_ax = ax.twiny()
        new_ax.plot(
            maxp[::saut],
            lin_err[::saut],
            "o",
            color="red",
            alpha=0.6,
            markersize=3,
        )

        legends, handles = [], []
        for metric, color, ax_to_plot in zip(
            [entropy, maxp], ["blue", "red"], [ax, new_ax]
        ):
            idx = np.isfinite(metric) & np.isfinite(lin_err)
            if np.any(idx):
                m, b = np.polyfit(metric[idx], lin_err[idx], 1)
                x_range = np.array([np.min(metric[idx]), np.max(metric[idx])])
                ax_to_plot.plot(x_range, m * x_range + b, color=color, lw=2)
                handles.append(Line2D([0], [0], color=color, lw=2))
                legends.append(
                    f"{'Entropy' if color == 'blue' else 'MaxP'} Fit: y={m:.2f}x"
                )
        if show_legend and handles:
            ax.legend(handles, legends, fontsize=8, loc="lower right")

        ax.set_xlabel("Entropy")
        new_ax.set_xlabel("Max Probability")
        ax.set_ylabel("Linear Error")
        ax.set_title(f"Certainty - {suff.strip('_')}")

        return fig

    def _prepare_boxplot_data(self, idWindow, thresh_entropy):
        data_box, labels_box, colors_box = [], [], []
        phase_box, cond_box = [], []

        colors_map = {
            "All": "lightblue",
            "All+Cert": "darkblue",
            "Fast": "salmon",
            "Fast+Cert": "darkred",
        }
        conditions_box = [
            ("All", False, None),
            ("All+Cert", False, thresh_entropy),
            ("Fast", True, None),
            ("Fast+Cert", True, thresh_entropy),
        ]

        for suff in self.suffixes:
            if self.resultsNN_phase[suff]["linPred"][idWindow] is None:
                continue
            lin_err = np.abs(
                self.resultsNN_phase[suff]["linPred"][idWindow]
                - self.resultsNN_phase[suff]["linTruePos"][idWindow]
            )
            speed_mask = self.resultsNN_phase[suff]["speedMask"][idWindow]
            entropy = self.resultsNN_phase[suff]["predLoss"][idWindow]

            for cond_name, use_speed, thresh in conditions_box:
                mask = np.ones(len(lin_err), dtype=bool)
                if use_speed and speed_mask is not None:
                    mask = mask & speed_mask
                elif use_speed:
                    raise ValueError(f"Speed mask missing for '{cond_name}'")

                if thresh is not None:
                    if entropy is not None:
                        mask = mask & (entropy <= thresh)
                    else:
                        mask = np.zeros(len(lin_err), dtype=bool)

                if np.any(mask):
                    data_box.append(lin_err[mask])
                    labels_box.append(f"{suff}\n{cond_name}")
                    colors_box.append(colors_map[cond_name])
                    phase_box.append(suff.strip("_"))
                    cond_box.append(cond_name)

        return data_box, labels_box, colors_box, phase_box, cond_box

    def _plot_oversampling_summary(
        self, timeWindow, idWindow, fig=None, gs=None, num_row=None, bins=40
    ):
        """Plot before/after oversampling occupancy and linearized distributions."""
        if fig is None or gs is None or num_row is None:
            fig = plt.figure(figsize=self.PDF_FIGSIZE, constrained_layout=False)
            fig.set_size_inches(*self.PDF_FIGSIZE, forward=True)
            gs = gridspec.GridSpec(1, 4, figure=fig)
            num_rows = 1
        else:
            num_rows = num_row

        gs_oversampling = gs[0:num_rows, :].subgridspec(1, 4)
        ax_before = fig.add_subplot(gs_oversampling[0, 0])
        ax_after = fig.add_subplot(gs_oversampling[0, 1])
        ax_diff = fig.add_subplot(gs_oversampling[0, 2])
        ax_hist = fig.add_subplot(gs_oversampling[0, 3])
        succeeded = True

        try:
            pos_before = self.resultsNN_phase["_training"]["truePos"][idWindow][:, :2]
            speed_mask = self.resultsNN_phase["_training"]["speedMask"][
                idWindow
            ].astype(bool)
            pos_before = pos_before[speed_mask]
        except Exception:
            pos_before = None
            succeeded = False

        pos_after_path = os.path.join(
            self.projectPath.folderResult,
            str(timeWindow),
            "pos_after.npy",
        )
        pos_after = (
            np.load(pos_after_path)[:, :2] if os.path.exists(pos_after_path) else None
        )

        def _hist2d_density(positions):
            if positions is None or len(positions) == 0:
                return None
            hist, xedges, yedges = np.histogram2d(
                positions[:, 0],
                positions[:, 1],
                bins=bins,
                range=[[0, 1], [0, 1]],
            )
            return hist.T, [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        before_pack = _hist2d_density(pos_before)
        after_pack = _hist2d_density(pos_after)

        if before_pack is None:
            succeeded = False
            for ax in [ax_before, ax_after, ax_diff, ax_hist]:
                ax.axis("off")
            ax_before.text(
                0.5,
                0.5,
                "No training data available",
                ha="center",
                va="center",
                transform=ax_before.transAxes,
            )
            return

        before_hist, extent = before_pack
        before_density = (
            before_hist / np.sum(before_hist)
            if np.sum(before_hist) > 0
            else before_hist
        )
        im0 = ax_before.imshow(
            before_density,
            origin="lower",
            extent=extent,
            cmap="Blues",
            interpolation="none",
        )
        fig.colorbar(im0, ax=ax_before, fraction=0.046, pad=0.04)
        ax_before.set_aspect("equal", adjustable="box")
        ax_before.set_xticks([])
        ax_before.set_yticks([])
        for spine in ax_before.spines.values():
            spine.set_visible(False)
        ax_before.set_title("Training occupancy")
        ax_before.set_xlabel("X")
        ax_before.set_ylabel("Y")

        if after_pack is not None:
            after_hist, _ = after_pack
            after_density = (
                after_hist / np.sum(after_hist)
                if np.sum(after_hist) > 0
                else after_hist
            )

            im1 = ax_after.imshow(
                after_density,
                origin="lower",
                extent=extent,
                cmap="Blues",
                interpolation="none",
            )
            fig.colorbar(im1, ax=ax_after, fraction=0.046, pad=0.04)
            ax_after.set_title("Oversampled occupancy")
            ax_after.set_xlabel("X")
            ax_after.set_ylabel("Y")
            ax_after.set_aspect("equal", adjustable="box")
            ax_after.set_xticks([])
            ax_after.set_yticks([])
            for spine in ax_after.spines.values():
                spine.set_visible(False)

            diff = after_density - before_density
            vmax = np.nanmax(np.abs(diff)) if np.any(np.isfinite(diff)) else 1.0
            vmax = vmax if vmax > 0 else 1.0
            im2 = ax_diff.imshow(
                diff,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="none",
            )
            fig.colorbar(im2, ax=ax_diff, fraction=0.046, pad=0.04)
            ax_diff.set_title("Density diff (after-before)")
            ax_diff.set_xlabel("X")
            ax_diff.set_ylabel("Y")
            ax_diff.set_aspect("equal", adjustable="box")
            ax_diff.set_xticks([])
            ax_diff.set_yticks([])
            for spine in ax_diff.spines.values():
                spine.set_visible(False)
        else:
            succeeded = False
            ax_after.axis("off")
            ax_after.text(
                0.5,
                0.5,
                "No pos_after.npy",
                ha="center",
                va="center",
                transform=ax_after.transAxes,
            )
            ax_diff.axis("off")

        lin_before = self.l_function(pos_before)[1].flatten()
        ax_hist.hist(
            lin_before,
            bins=30,
            density=True,
            alpha=0.55,
            color="gray",
            label="Training",
        )
        if pos_after is not None:
            lin_after = self.l_function(pos_after)[1].flatten()
            ax_hist.hist(
                lin_after,
                bins=30,
                density=True,
                alpha=0.55,
                color="tab:orange",
                label="Oversampled",
            )
        else:
            succeeded = False
        ax_hist.set_title("Linearized distribution")
        ax_hist.set_xlabel("Linear Position")
        # show one xticks every 0.25 of the linearized space
        ax_hist.set_xticks(np.arange(0, 1.01, 0.25))
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8, loc="upper right")

        return succeeded

    def _plot_prediction_histograms(
        self, timeWindow, idWindow, fig=None, gs=None, row=None
    ):
        if fig is None or gs is None or row is None:
            fig = plt.figure(figsize=self.PDF_FIGSIZE, constrained_layout=False)
            fig.set_size_inches(*self.PDF_FIGSIZE, forward=True)
            gs = gridspec.GridSpec(3, 4, figure=fig)
            row = 0

        gs_histogram = gs[row : row + 1, :].subgridspec(1, len(self.suffixes))
        oversampled = os.path.exists(
            os.path.join(
                self.projectPath.folderResult,
                str(timeWindow),
                "oversampling_effect.png",
            )
        )

        for i, suff in enumerate(self.suffixes):
            ax_hist = fig.add_subplot(gs_histogram[0, i])
            pos_after_path = os.path.join(
                self.projectPath.folderResult,
                str(timeWindow),
                "pos_after.npy",
            )
            pos_after = False
            if not os.path.exists(pos_after_path):
                warnings.warn(
                    "Expected oversampling data not found. Showing vanilla training set."
                )
                training_pos = self.resultsNN_phase["_training"]["featureTrue"][
                    idWindow
                ][:, :2]
            else:
                pos_after = True
                training_pos = np.load(pos_after_path)[:, :2]

            lin_training = self.l_function(training_pos)[1]
            lin_suff_true = self.resultsNN_phase[suff]["linTruePos"][idWindow]
            lin_suff_pred = self.resultsNN_phase[suff]["linPred"][idWindow]

            ax_hist.hist(
                lin_training,
                bins=30,
                alpha=0.5,
                label=(
                    f"Training {'Oversampled' if oversampled and pos_after else 'Original'}"
                ),
                color="xkcd:pale purple",
                density=True,
            )
            ax_hist.hist(
                lin_suff_true,
                bins=30,
                alpha=0.5,
                label=f"{suff} True",
                color="xkcd:dark pink",
                density=True,
            )
            ax_hist.hist(
                lin_suff_pred,
                bins=30,
                alpha=0.5,
                label=f"{suff} Pred",
                color="tab:blue",
                density=True,
            )
            ax_hist.set_title(f"Distrib - {suff.strip('_')}")
            ax_hist.set_xlabel("Linear Position")

        handles = [
            Line2D([0], [0], color="xkcd:pale purple", lw=4, alpha=0.5),
            Line2D([0], [0], color="xkcd:dark pink", lw=4, alpha=0.5),
            Line2D([0], [0], color="tab:blue", lw=4, alpha=0.5),
        ]
        training_label = f"Training {'(Oversampled)' if oversampled else '(Original)'}"
        labels = [
            training_label,
            "True Position",
            "Predicted Position",
        ]
        if len(self.suffixes) > 0:
            legend_ax = fig.axes[-1]
            legend_ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                fontsize=8,
            )

    def _plot_violin(self, data_box, labels_box, colors_box, ax=None, paired=True):
        if ax is None:
            fig, ax = plt.subplots()
        positions = []
        current_pos = 1
        for i in range(len(data_box)):
            positions.append(current_pos)
            if (i + 1) % 4 == 0:
                current_pos += 2
            else:
                current_pos += 1

        vp = ax.violinplot(
            data_box,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.7,
        )

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors_box[i])
            body.set_edgecolor("white")
            body.set_alpha(0.2)
            body.set_linewidth(1)

        for i, data in enumerate(data_box):
            x_pos = positions[i]
            med = np.median(data)
            if med > 0:
                ax.hlines(
                    med,
                    x_pos - 0.35,
                    x_pos + 0.35,
                    colors=colors_box[i],
                    linewidth=10,
                    zorder=10,
                )
            # Scatter points
            jitter = np.random.normal(0, 0.05, size=len(data))
            ax.scatter(
                x_pos + jitter,
                data,
                color=colors_box[i],
                s=3,
                alpha=0.05,
                zorder=5,
                edgecolors="none",
            )

        if paired and len(data_box) > 1:
            for i in range(len(data_box) - 1):
                if (i + 1) % 4 != 0 and len(data_box[i]) == len(data_box[i + 1]):
                    for j in range(len(data_box[i])):
                        ax.plot(
                            [positions[i] + 0.1, positions[i + 1] - 0.1],
                            [data_box[i][j], data_box[i + 1][j]],
                            color="gray",
                            alpha=0.1,
                            linewidth=0.5,
                            zorder=1,
                        )

        ax.set_yscale("log")
        # Use ScalarFormatter for non-scientific notation ticks
        # First only show a few ticks, then format them with FuncFormatter
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs="auto", numticks=10)
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: "{:g}".format(y))
        )
        # hide the minor tick labels but keep the ticks
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: ""))

        ax.set_ylabel("Linear Error (log scale)", fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels("")

        for i in range(0, len(positions), 4):
            group_start = positions[i] - 0.5
            group_end = positions[min(i + 3, len(positions) - 1)] + 0.5
            if (i // 4) % 2 == 0:
                ax.axvspan(group_start, group_end, color="gray", alpha=0.05, zorder=0)
            if i > 0:
                ax.axvline(
                    x=group_start - 0.5,
                    color="black",
                    linestyle=":",
                    alpha=0.2,
                    zorder=1,
                )
        ax.set_xlim(positions[0] - 1, positions[-1] + 1)

    def _plot_summary_trajectories(
        self, timeWindow, idWindow, dimOutput, thresh_entropy, kwargs, pdf=None
    ):
        with_hist_distribution = kwargs.get("with_hist_distribution", True)
        all_conditions = [
            ("All Preds", False, None),
            ("All + Certainty", False, thresh_entropy),
            ("Speed Thresholded", True, None),
            ("Speed + Certainty", True, thresh_entropy),
        ]

        for suffix in self.suffixes:
            posIndex = self.resultsNN_phase[suffix]["posIndex"][idWindow]
            # Use real timestamps if available to show time in seconds
            if hasattr(self, "behaviorData") and "positionTime" in self.behaviorData:
                try:
                    timeStepsPred = self.behaviorData["positionTime"][posIndex]
                except Exception:
                    timeStepsPred = self.resultsNN_phase[suffix]["time"][idWindow]
            else:
                timeStepsPred = self.resultsNN_phase[suffix]["time"][idWindow]
            speedMask = self.resultsNN_phase[suffix]["speedMask"][idWindow]
            entropy = self.resultsNN_phase[suffix]["predLoss"][idWindow]

            if dimOutput == 1:
                pos = self.resultsNN_phase[suffix]["linTruePos"][idWindow]
                inferring = self.resultsNN_phase[suffix]["linPred"][idWindow]
                training_data = self.resultsNN_phase["_training"]["linTruePos"][
                    idWindow
                ]
            elif dimOutput == 2:
                pos = self.resultsNN_phase[suffix]["truePos"][idWindow][:, :2]
                inferring = self.resultsNN_phase[suffix]["fullPred"][idWindow][:, :2]
                training_data = self.resultsNN_phase["_training"]["truePos"][idWindow][
                    :, :2
                ]
            else:
                raise ValueError("dimOutput must be 1 or 2.")

            conditions_per_page = 2
            chunked_conditions = [
                all_conditions[i : i + conditions_per_page]
                for i in range(0, len(all_conditions), conditions_per_page)
            ]

            for page_idx, conditions_chunk in enumerate(chunked_conditions):
                fig_page = plt.figure(
                    figsize=self.PDF_FIGSIZE, constrained_layout=False
                )
                fig_page.set_size_inches(*self.PDF_FIGSIZE, forward=True)
                fig_page.set_rasterized(True)

                nrows_per_cond = dimOutput
                total_rows = len(conditions_chunk) * nrows_per_cond
                gs_page = gridspec.GridSpec(total_rows, 1, figure=fig_page, hspace=0.4)
                current_row = 0

                for cond_name, use_speed, thresh in conditions_chunk:
                    selection = np.ones(len(pos), dtype=bool)
                    if use_speed and speedMask is not None:
                        selection = selection & speedMask
                    if thresh is not None and entropy is not None:
                        selection = selection & (entropy <= thresh)

                    ncols = 6 if with_hist_distribution else 5
                    gs_cond = gs_page[
                        current_row : current_row + dimOutput, 0
                    ].subgridspec(dimOutput, ncols)

                    axs_list = []
                    first_ax_ref = None

                    for r in range(dimOutput):
                        if r == 0:
                            ax_main = fig_page.add_subplot(gs_cond[r, 0:4])
                            first_ax_ref = ax_main
                        else:
                            ax_main = fig_page.add_subplot(
                                gs_cond[r, 0:4], sharex=first_ax_ref
                            )
                        axs_list.append(ax_main)
                        if with_hist_distribution:
                            ax_dist = fig_page.add_subplot(
                                gs_cond[r, 4], sharey=ax_main
                            )
                            ax_dist.tick_params(axis="y", left=False, labelleft=False)
                            axs_list.append(ax_dist)

                    overview_fig(
                        pos=pos,
                        inferring=inferring,
                        selection=selection,
                        posIndex=posIndex,
                        timeStepsPred=timeStepsPred,
                        speedMask=None,
                        useSpeedMask=False,
                        concat_epochs=True,
                        dimOutput=dimOutput,
                        show=False,
                        save=False,
                        close=False,
                        training_data=training_data,
                        join_points=False,
                        axs=np.array(axs_list),
                        fig=fig_page,
                        show_legend=current_row == total_rows - dimOutput,
                    )

                    axs_list[0].set_title(f"Phase: {suffix.strip('_')} - {cond_name}")
                    self._annotate_trajectory_error(
                        axs_list[0],
                        dimOutput,
                        inferring,
                        pos,
                        selection,
                        suffix,
                        idWindow,
                    )

                    # Error Matrix
                    last_col_ax = fig_page.add_subplot(gs_cond[:, -1])
                    last_col_ax.axis("off")
                    linpos = self.resultsNN_phase[suffix]["linTruePos"][idWindow]
                    linpred = self.resultsNN_phase[suffix]["linPred"][idWindow]

                    if len(selection) > 0 and np.sum(selection) > 0:
                        self._plot_single_error_matrix(
                            linpos[selection],
                            linpred[selection],
                            last_col_ax,
                        )

                    current_row += dimOutput

                fig_page.subplots_adjust(
                    left=0.05,
                    right=0.98,
                    bottom=0.04,
                    top=0.95,
                    wspace=0.35,
                    hspace=0.5,
                )
                if pdf is not None:
                    pdf.savefig(fig_page, dpi=self.PDF_DPI, bbox_inches=None)
                else:
                    plt.show()
                plt.close(fig_page)

    def _annotate_trajectory_error(
        self, ax, dimOutput, inferring, pos, selection, suffix, idWindow
    ):
        if dimOutput == 2:
            error = np.linalg.norm(inferring - pos, axis=1)
        else:
            error = np.abs(inferring - pos)

        masked_error = error[selection]
        mean_error = np.nan
        median_error = np.nan
        if len(masked_error) > 0:
            mean_error = np.mean(masked_error)
            median_error = np.median(masked_error)

        text = (
            f"Mean Error: {mean_error:.3f}\n"
            f"Median Error: {median_error:.3f}\n"
            f"Samples: {len(masked_error)}"
        )
        ax.text(
            0.75,
            -0.1,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize="small",
        )

    def _plot_summary_bayesian(self, timeWindow, pdf=None):
        bayesian_page = plt.figure(
            figsize=(self.PDF_FIGSIZE[1], self.PDF_FIGSIZE[0]), constrained_layout=False
        )
        # Use dedicated cbar rows so shared colorbars are controlled by GridSpec.
        gs_bayes = gridspec.GridSpec(
            4,
            3,
            figure=bayesian_page,
            height_ratios=[0.08, 1.0, 0.08, 1.0],
        )
        cax_train = bayesian_page.add_subplot(gs_bayes[0, 1:3])
        cax_pred = bayesian_page.add_subplot(gs_bayes[2, 1:3])

        bayesian_axs = [bayesian_page.add_subplot(gs_bayes[1, j]) for j in range(3)] + [
            bayesian_page.add_subplot(gs_bayes[3, j]) for j in range(3)
        ]

        try:
            self._create_decoding_bayes_matrices(winMS=timeWindow, suffix="_training")
            self.bayesian_neurons_summary(
                fig=bayesian_page,
                axs=bayesian_axs,
                cax_train=cax_train,
                cax_pred=cax_pred,
                show=False,
                block=False,
                save=True,
                winMS=timeWindow,
            )
        except Exception as e:
            print(f"Could not generate bayesian summary page: {e}")
            import traceback

            traceback.print_exc()

        bayesian_page.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.04,
            top=0.96,
            wspace=0.25,
            hspace=0.45,
        )

        if pdf is not None:
            pdf.savefig(bayesian_page, dpi=self.PDF_DPI, bbox_inches=None)
        else:
            bayesian_page.show()
        plt.close(bayesian_page)

        if pdf is None:
            return bayesian_page

    def hist_linerrors(
        self,
        suffix=None,
        phase=None,
        speed="all",
        mask=None,
        use_mask=False,
        block=False,
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
        if (
            fig.get_tight_layout() is None
        ):  # Only adjust if tight_layout hasn't been called yet
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
        ax.set_aspect("equal")

        # Note: Transpose H.T is used to align axes correctly with origin="lower"
        im = ax.imshow(
            H.T,
            extent=extent,
            cmap=cmap,
            interpolation="none",
            origin="lower",
        )

        # 5. Add Smaller Colorbar

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        # 6. Add diagonal stats
        error = np.abs(true_pos.reshape(-1) - pred_pos.reshape(-1))
        in_diag = np.mean(error <= 0.1) * 100
        out_diag = 100 - in_diag

        ax.text(
            0.5,
            -0.15,
            f"Diagonal (±0.1): {in_diag:.1f}%\nOff-Diagonal: {out_diag:.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

        return fig, ax

    def error_matrix_linerrors_by_speed(
        self, suffixes=None, nbins=40, normalized=True, show=False
    ):
        if suffixes is None:
            suffixes = self.suffixes
        if not isinstance(suffixes, list):
            suffixes = [suffixes]
        suffixes = ["_" + s.strip("_") for s in suffixes]

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
                    aspect="auto",
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
                    aspect="auto",
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
        block=False,
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
        block=False,
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
                self.resultsNN_phase[suffix]["fullPred"][i][:, :2] * EC
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                self.resultsNN_phase[suffix]["truePos"][i][:, :2] * EC
                for i in range(len(self.timeWindows))
            ]
            bayesD["pred"] = [
                self.resultsBayes_phase[suffix]["fullPred"][i][:, :2] * EC
                for i in range(len(self.timeWindows))
            ]
        else:
            nnD["pred"] = [
                self.resultsNN_phase[suffix]["fullPred"][i][:, :2]
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                self.resultsNN_phase[suffix]["truePos"][i][:, :2]
                for i in range(len(self.timeWindows))
            ]
            bayesD["pred"] = [
                self.resultsBayes_phase[suffix]["fullPred"][i][:, :2]
                for i in range(len(self.timeWindows))
            ]
        errorNN = [
            np.linalg.norm(nnD["true"][i] - nnD["pred"][i][:, :2], axis=1, ord=2)
            for i in range(len(self.timeWindows))
        ]
        errorBayes = [
            np.linalg.norm(nnD["true"][i] - bayesD["pred"][i][:, :2], axis=1, ord=2)
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
        block=False,
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
        plt.get_cmap("terrain")
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
                cmap=white_viridis,
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
        self,
        suffix=None,
        phase=None,
        speed="fast",
        mode="2d",
        block=False,
        typeDec="ann",
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

        loss_name = "Predicted loss" if typeDec == "ann" else "Bayes Proba"

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
                # aspect="auto",
                alpha=0.4,
                density=True,
            )  # ,c="red",alpha=0.4
            ax[iw].set_xlabel(loss_name, fontsize="x-large")
            if mode == "2d":
                ax[iw].set_ylabel("True error")
            elif mode == "1d":
                ax[iw].set_ylabel("Linear error")
            ax[iw].set_title((str(self.timeWindows[iw]) + " ms"), fontsize="x-large")

            # modify xticks
            ax[iw].tick_params(axis="x", which="major", labelsize=15, rotation=45)
            ax[iw].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        fig.suptitle(
            f"{loss_name} vs true error during \n{str(speed)} speed periods for phase {suffix.strip('_')}"
        )
        if fig.get_layout_engine() is None:
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
        block=False,
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
        [
            self.resultsNN_phase[suffix]["time"][iw][mask[iw]]
            for iw in range(len(self.timeWindows))
        ]
        # Trajectory figure
        plt.get_cmap("turbo")
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
        speed="fast",
        num_steps=200,
        mask=None,
        use_mask=False,
        block=False,
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
        speed="fast",
        typeDec="ann",
        num_steps=200,
        isCM=False,
        scaled=True,
        mask=None,
        use_mask=False,
        block=False,
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
                results_phase[suffix]["fullPred"][i][:, :2] * EC
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                results_phase[suffix]["truePos"][i][:, :2] * EC
                for i in range(len(self.timeWindows))
            ]
        else:
            nnD["pred"] = [
                results_phase[suffix]["fullPred"][i][:, :2]
                for i in range(len(self.timeWindows))
            ]
            nnD["true"] = [
                results_phase[suffix]["truePos"][i][:, :2]
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

    def plot_barplot_error(
        self,
        results_df=None,
        logscale=True,
        speed="fast",
        confidence=False,
        threshold=0.6,
    ):
        if results_df is None:
            try:
                results_df = self.results_df
            except AttributeError:
                raise ValueError(
                    "results_df must be provided if self.results_df not set. If using Mouse_Results object, have you tried convert_to_df() yet?"
                )
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
        sns.barplot(
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
                f"barplot_{speed}_filtered_se_error.png",
            )
        )
        plt.show()
        return results_df

    def lin_barplot_error(
        self,
        results_df=None,
        logscale=True,
        speed="fast",
        confidence=False,
        threshold=0.6,
    ):
        if results_df is None:
            try:
                results_df = self.results_df
            except AttributeError:
                raise ValueError(
                    "results_df must be provided if self.results_df not set. If using Mouse_Results object, have you tried convert_to_df() yet?"
                )
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
        sns.barplot(
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
                f"linbarplot_{speed}_filtered_se_error_{confidence=}_{threshold=}.png",
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

                try:
                    logits_hw = self.resultsNN_phase_pkl[phase][idWindow]["logits_hw"][
                        speedMask
                    ]
                except KeyError:
                    raise ValueError(
                        f"Logits not found for phase {phase} and window {winMS} ms. Make sure to load with load_pickle."
                    )
                truePos = self.resultsNN_phase[phase]["truePos"][idWindow][speedMask][
                    :, :2
                ]
                predPos = self.resultsNN_phase[phase]["fullPred"][idWindow][speedMask][
                    :, :2
                ]
                self.ann.GaussianHeatmap.gaussian_heatmap_targets(truePos)
                probs = self.ann.GaussianHeatmap.decode_and_uncertainty(
                    logits_hw, return_probs=True
                )[-1].numpy()
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
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"heatmap_proba_error_{normalized_by}_surprise_{plot_surprise}_bias_{plot_bias}_{winMS}.png",
            ),
            bbox_inches="tight",
        )
        if show:
            plt.show(block=False)

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
                truePos = self.resultsNN_phase[phase]["truePos"][idWindow][speedMask][
                    :, :2
                ]
                target_hw = self.ann.GaussianHeatmap.gaussian_heatmap_targets(
                    truePos
                ).numpy()
                probs = self.ann.GaussianHeatmap.decode_and_uncertainty(
                    logits_hw, return_probs=True
                )[-1].numpy()
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
                ax1.imshow(
                    mean_probs,
                    origin="lower",
                    extent=extent,
                    vmin=0,
                    vmax=mean_probs.max(),
                )
                ax1.set_title(f"{phase[1:]}-{speed}-Proba")
                # plt.colorbar(im1, ax=ax1)
                ax2 = axs[i, j * (3 if plot_kl else 2) + 1]
                ax2.imshow(
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
                    ax3.imshow(
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
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.folderFigures,
                f"heatmap_vs_target_abs_{plot_kl}_{winMS}.png",
            ),
            bbox_inches="tight",
        )
        if show:
            plt.show(block=False)

    def fig_example_linear_filtered(
        self, suffix=None, phase=None, fprop=0.3, block=False
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
        [
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
        if fig.get_layout_engine() is None:
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
        block=False,
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
        self, timeWindow, suffix=None, phase=None, block=False
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
        return np.mean(errors), np.mean(errorsShuffleMean), np.mean(errorsShuffleStd)

    # ------------------------------------------------------------------------------------------------------------------------------
    ## Figure 4: we take an example place cell,
    # and we scatter plot a link between its firing rate and the decoding.

    def plot_pc_tuning_curve_and_predictions(
        self,
        suffix=None,
        phase=None,
        ws=None,
        block=False,
        show=False,
        useTrain=False,
        useTest=False,
        useAll=True,
        strideFactor=1,
    ):
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if ws is None:
            ws = self.timeWindows[0]  # default to the first time window

        if useTrain and useTest:
            useAll = True

        dirSave = os.path.join(self.folderFigures, "tuningCurves")
        if not os.path.isdir(dirSave):
            os.mkdir(dirSave)

        iwindow = self.timeWindows.index(ws)
        # Calculate the tuning curve of all place cells
        linearTuningCurves, binEdges = self.trainerBayes.calculate_linear_tuning_curve(
            l_function=self.l_function, behaviorData=self.behaviorData
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

        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

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

                if fig.get_layout_engine() is None:
                    fig.tight_layout()
                if show:
                    plt.show(block=block)

                fig.savefig(
                    os.path.join(dirSave, (f"{ws}_tc_pred_cluster{pcId}{suffix}.png"))
                )
                plt.close()

    def barplot_linError(
        self, timeWindows, dirSave=None, suffix=None, phase=None, block=False
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
        from neuroencoders.resultAnalysis.hyper_paper_figures import barplot_linError

        if dirSave is None:
            dirSave = self.folderFigures
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if not isinstance(timeWindows, list):
            timeWindows = [timeWindows]

        lErrorNN_mean = np.array(
            [
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
        )
        lErrorBayes_mean = np.array(
            [
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
        )
        return barplot_linError(
            lErrorNN_mean,
            lErrorBayes_mean,
            timeWindows,
            dirSave=dirSave,
            suffix=suffix,
        )

    def barplot_euclError(
        self, timeWindows, dirSave=None, suffix=None, phase=None, block=False
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
        from neuroencoders.resultAnalysis.hyper_paper_figures import barplot_euclError

        if dirSave is None:
            dirSave = self.folderFigures
        if phase is not None:
            suffix = f"_{phase}"
        if suffix is None:
            suffix = self.suffix
        if not isinstance(timeWindows, list):
            timeWindows = [timeWindows]
        euclErrorNN_mean = np.array(
            [
                np.mean(
                    np.abs(
                        self.resultsNN_phase[suffix]["truePos"][
                            self.timeWindows.index(ws)
                        ][:, :2]
                        - self.resultsNN_phase[suffix]["fullPred"][
                            self.timeWindows.index(ws)
                        ][:, :2]
                    )
                )
                for ws in timeWindows
            ]
        )
        euclErrorBayes_mean = np.array(
            [
                np.mean(
                    np.linalg.norm(
                        self.resultsNN_phase[suffix]["truePos"][
                            self.timeWindows.index(ws)
                        ][:, :2]
                        - self.resultsBayes_phase[suffix]["fullPred"][
                            self.timeWindows.index(ws)
                        ][:, :2],
                        axis=1,
                    )
                )
                for ws in timeWindows
            ]
        )
        return barplot_euclError(
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
        block=False,
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

    def correlate_predLoss_and_bayesProba(
        self,
        speed="all",
        suffix=None,
        phase=None,
        mask=None,
        use_mask=False,
        block=False,
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
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        fig.show(block=block)
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

    def barplot_error_across_suffixes(
        self,
        timeWindow,
        speed="fast",
        type_error="lin",
        suffixes=None,
        dirSave=None,
        block=False,
        ax=None,
    ):
        if suffixes is None:
            suffixes = self.suffixes
        if dirSave is None:
            dirSave = self.folderFigures
        errors_dict = {}
        for suffix in suffixes:
            if type_error == "lin":
                errors_dict[suffix] = np.mean(
                    np.abs(
                        self.resultsNN_phase[suffix]["linTruePos"][
                            self.timeWindows.index(timeWindow)
                        ]
                        - self.resultsNN_phase[suffix]["linPred"][
                            self.timeWindows.index(timeWindow)
                        ]
                    )
                )
            elif type_error == "eucl":
                errors_dict[suffix] = np.mean(
                    np.linalg.norm(
                        self.resultsNN_phase[suffix]["truePos"][
                            self.timeWindows.index(timeWindow)
                        ][:, :2]
                        - self.resultsNN_phase[suffix]["fullPred"][
                            self.timeWindows.index(timeWindow)
                        ][:, :2],
                        axis=1,
                    )
                )
            else:
                raise ValueError("type_error should be 'lin' or 'eucl'")
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if ax is None:
            fig, ax = plt.subplots()

        ax.bar(errors_dict.keys(), errors_dict.values(), color="skyblue")
        ax.set_ylabel(f"Mean {type_error} error")
        ax.set_title(
            f"Mean {type_error} error across suffixes for {timeWindow} ms window, speed: {speed}"
        )
        plt.xticks(rotation=45)
        if fig.get_layout_engine() is None:
            fig.tight_layout()

    def plot_ann_pred_by_phase(
        self,
        winMS_list: Optional[Union[int, List[int]]] = None,
        suffixes: Optional[List[str]] = None,
        speeds: Union[str, List[str]] = "fast",
        type_error: str = "lin",
        reduce_fn: str = "median",
        threshold_pct: Optional[float] = None,
        by: str = "entropy",
        add_bayes: bool = False,
        ax: Optional[matplotlib.axes.Axes] = None,
        show: bool = True,
        save: bool = False,
        **kwargs,
    ):
        """
        Plot the evolution of median/mean lin/eucl error across phases.
        Allows filtering by speed and predLoss (entropy) thresholding.
        """
        if winMS_list is None:
            winMS_list = self.timeWindows
        if isinstance(winMS_list, (int, float)):
            winMS_list = [int(winMS_list)]

        if suffixes is None:
            suffixes = self.suffixes

        if isinstance(speeds, str):
            speeds = [speeds]

        plot_data = []

        # Threshold calculation from training phase if needed
        thresholds = {}
        if threshold_pct is not None:
            training_suffix = (
                "_training" if "_training" in self.resultsNN_phase else None
            )
            if training_suffix is None and suffixes:
                training_suffix = suffixes[0]

            if training_suffix:
                for ws in winMS_list:
                    if ws not in self.timeWindows:
                        continue
                    idx = self.timeWindows.index(ws)
                    results = self.resultsNN_phase[training_suffix]
                    sm = results["speedMask"][idx]

                    for speed in speeds:
                        if speed == "fast":
                            s_mask = sm
                        elif speed == "slow":
                            s_mask = ~sm if sm is not None else None
                        else:
                            s_mask = np.ones_like(results["time"][idx], dtype=bool)

                        if s_mask is None:
                            continue

                        if by == "entropy":
                            val = results["predLoss"][idx]
                            if val is not None:
                                thresholds[(ws, speed)] = np.nanpercentile(
                                    val[s_mask], threshold_pct
                                )
                        elif by == "maxp":
                            if (
                                hasattr(self, "resultsNN_phase_pkl")
                                and training_suffix in self.resultsNN_phase_pkl
                            ):
                                pkl = self.resultsNN_phase_pkl[training_suffix][idx]
                                if pkl is not None and "maxp" in pkl:
                                    thresholds[(ws, speed)] = np.nanpercentile(
                                        pkl["maxp"][s_mask], 100 - threshold_pct
                                    )

        for suffix in suffixes:
            if suffix not in self.resultsNN_phase:
                continue
            results_nn = self.resultsNN_phase[suffix]
            results_bayes = self.resultsBayes_phase.get(suffix) if add_bayes else None

            for ws in winMS_list:
                if ws not in self.timeWindows:
                    continue
                idx = self.timeWindows.index(ws)
                sm = results_nn["speedMask"][idx]

                for speed in speeds:
                    if speed == "fast":
                        s_mask = sm
                    elif speed == "slow":
                        s_mask = ~sm if sm is not None else None
                    else:
                        s_mask = np.ones_like(results_nn["time"][idx], dtype=bool)

                    if s_mask is None:
                        continue

                    # Threshold Mask
                    if threshold_pct is not None and (ws, speed) in thresholds:
                        if by == "entropy":
                            t_mask = (
                                results_nn["predLoss"][idx] <= thresholds[(ws, speed)]
                            )
                            final_mask = s_mask & t_mask
                        elif by == "maxp":
                            if (
                                hasattr(self, "resultsNN_phase_pkl")
                                and suffix in self.resultsNN_phase_pkl
                            ):
                                pkl = self.resultsNN_phase_pkl[suffix][idx]
                                if pkl is not None and "maxp" in pkl:
                                    t_mask = pkl["maxp"] >= thresholds[(ws, speed)]
                                    final_mask = s_mask & t_mask
                                else:
                                    final_mask = s_mask
                        else:
                            final_mask = s_mask
                    else:
                        final_mask = s_mask

                    # ANN Error
                    if results_nn["linPred"][idx] is not None:
                        if type_error == "lin":
                            err = np.abs(
                                results_nn["linTruePos"][idx]
                                - results_nn["linPred"][idx]
                            )
                        else:  # eucl
                            err = np.linalg.norm(
                                results_nn["truePos"][idx][:, :2]
                                - results_nn["fullPred"][idx][:, :2],
                                axis=1,
                            )

                        err = err[final_mask]
                        if len(err) > 0:
                            val = (
                                np.nanmean(err)
                                if reduce_fn == "mean"
                                else np.nanmedian(err)
                            )
                            plot_data.append(
                                {
                                    "Phase": suffix.lstrip("_"),
                                    "Error": val,
                                    "Window (ms)": ws,
                                    "Speed": speed,
                                    "Method": "ANN",
                                }
                            )

                    # Bayes Error
                    if (
                        add_bayes
                        and results_bayes is not None
                        and results_bayes["linPred"][idx] is not None
                    ):
                        if type_error == "lin":
                            err_b = np.abs(
                                results_bayes["linTruePos"][idx]
                                - results_bayes["linPred"][idx]
                            )
                        else:
                            err_b = np.linalg.norm(
                                results_bayes["truePos"][idx][:, :2]
                                - results_bayes["fullPred"][idx][:, :2],
                                axis=1,
                            )

                        err_b = err_b[final_mask]
                        if len(err_b) > 0:
                            val_b = (
                                np.nanmean(err_b)
                                if reduce_fn == "mean"
                                else np.nanmedian(err_b)
                            )
                            plot_data.append(
                                {
                                    "Phase": suffix.lstrip("_"),
                                    "Error": val_b,
                                    "Window (ms)": ws,
                                    "Speed": speed,
                                    "Method": "Bayes",
                                }
                            )

        if not plot_data:
            print("No data found for the specified criteria.")
            return

        df_plot = pd.DataFrame(plot_data)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Determine sorting for Phases
        phase_order = ["training", "pre", "cond", "post", "extinction"]
        unique_phases = df_plot["Phase"].unique()
        order = [p for p in phase_order if p in unique_phases]
        # Append any remaining phases not in our predefined list
        order += [p for p in unique_phases if p not in phase_order]

        hue_val = "Window (ms)" if len(winMS_list) > 1 else "Speed"

        if add_bayes:
            # Combine Window/Speed and Method for hue
            df_plot["Group"] = (
                df_plot[hue_val].astype(str) + " (" + df_plot["Method"] + ")"
            )
            hue_to_use = "Group"
        else:
            hue_to_use = hue_val

        sns.pointplot(
            data=df_plot, x="Phase", y="Error", hue=hue_to_use, order=order, ax=ax
        )

        title = f"Evolution of {reduce_fn} {type_error} error across phases\n"
        if threshold_pct:
            title += f"Threshold: {threshold_pct}% {by}, "
        title += f"Speeds: {', '.join(speeds)}"
        ax.set_title(title)
        ax.set_ylabel(f"{reduce_fn.capitalize()} {type_error} error")
        plt.setp(ax.get_xticklabels(), rotation=45)

        if save:
            self._save_fig(ax.get_figure(), f"error_evolution_{type_error}_{reduce_fn}")
        if show:
            plt.show()
        return df_plot

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
        # if they are arrays, take the first element
        if isinstance(peak_x, np.ndarray):
            peak_x = peak_x[0]
            peak_x = int(peak_x)
        if isinstance(peak_y, np.ndarray):
            peak_y = peak_y[0]
            peak_y = int(peak_y)
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

    def _compute_linear_pf_correlation(
        self, true_behavior, predicted_behavior, timeWindow, use_speed_filter=True
    ):
        """Compute mean Pearson correlation across neurons for linear tuning curves."""
        try:
            true_fields, _ = self.trainerBayes.calculate_linear_tuning_curve(
                l_function=self.l_function,
                behaviorData=true_behavior,
                use_speed_filter=use_speed_filter,
            )
            pred_fields, _ = self.trainerBayes.calculate_linear_tuning_curve(
                l_function=self.l_function,
                behaviorData=predicted_behavior,
                use_speed_filter=use_speed_filter,
                is_predicted=True,
                winMS=timeWindow,
            )
        except Exception:
            return np.nan

        if len(true_fields) == 0 or len(pred_fields) == 0:
            return np.nan

        corr_values = []
        n_cells = min(len(true_fields), len(pred_fields))
        for i in range(n_cells):
            true_pf = np.asarray(true_fields[i]).reshape(-1)
            pred_pf = np.asarray(pred_fields[i]).reshape(-1)
            if (
                true_pf.shape != pred_pf.shape
                or np.std(true_pf) == 0
                or np.std(pred_pf) == 0
            ):
                continue
            corr_values.append(stats.pearsonr(true_pf, pred_pf)[0])

        if len(corr_values) == 0:
            return np.nan
        return float(np.nanmean(corr_values))

    def bayesian_neurons_summary(self, axs=None, fig=None, block=False, **kwargs):
        """
        Summary of the Bayesian neurons.
        Can create its own figure or plot on provided axes.

        Args:
            axs (array-like, optional): Array of matplotlib axes to plot on. If None, a new figure with 6 subplots will be created.
            fig (matplotlib.figure.Figure, optional): Figure object to associate with the axes. If None and axs is provided, fig will be inferred from axs.
            block (bool, optional): Whether to block execution when showing the plot. Default is False.
            **kwargs: Additional keyword arguments for training and plotting.
                Supported kwargs:
                - plot_high_quality (bool): If True, plots only high-quality neurons based on Mutual Information. Default is False.
                - save (bool): If True, saves the figure. Default is True if axs is None, else False.
                - show (bool): If True, displays the figure. Default is True if axs is None, else False.
                - scaling (str): Scaling method for linear place fields ('minmax' or 'z-score'). Default is 'minmax'.
        """
        if not hasattr(self, "trainerBayes") or self.trainerBayes is None:
            raise ValueError(
                "trainerBayes is not available. Please run Bayesian training first."
            )

        # kwargs processing
        plot_high_quality = kwargs.get("plot_high_quality", False)
        save = kwargs.get("save", True if axs is None else False)
        show = kwargs.get("show", True if axs is None else False)
        is_predicted = kwargs.get("is_predicted", False)
        winMS = kwargs.get("winMS", self.timeWindows[0] if self.timeWindows else 100)
        cax_train = kwargs.pop("cax_train", None)
        cax_pred = kwargs.pop("cax_pred", None)

        # --- 1. Train/Load Data ---
        if kwargs.get("bayesMatrices", None) is not None:
            bayes_mat = kwargs["bayesMatrices"]
        else:
            if getattr(self, "bayesMatrices", None) is None:
                existing_bayes = (
                    self.bayesMatrices
                    if (
                        isinstance(self.bayesMatrices, dict)
                        and "Occupation" in self.bayesMatrices
                    )
                    else None
                )

                bayes_mat = self.trainerBayes.train_order_by_pos(
                    self.behaviorData,
                    l_function=self.l_function,
                    bayesMatrices=existing_bayes,
                    **kwargs,
                )
            else:
                bayes_mat = self.bayesMatrices

        # Extract and sort Mutual Information
        flat_mi = [mi for tetrode_mi in bayes_mat["mutualInfo"] for mi in tetrode_mi]
        ordered_mi = np.array(flat_mi)[bayes_mat["linearPosArgSort"]]

        # FIX: check that every neuron fires at least once in the spikeMatLabels (only in training data)
        # --- 2. Identify High-Quality Neurons ---
        thresh = 80
        percentile_val = np.percentile(ordered_mi, thresh)
        high_quality_mask = ordered_mi > percentile_val
        high_quality_indices = bayes_mat["linearPosArgSort"][high_quality_mask]

        print(
            f"High-quality place cells: {len(high_quality_indices)} neurons (top {100 - thresh}%)"
        )
        print(f"Total neurons: {len(bayes_mat['linearPosArgSort'])}")
        print(
            f"Position range: {bayes_mat['linearPreferredPos'].min():.2f} - {bayes_mat['linearPreferredPos'].max():.2f}"
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
        for i in range(len(bayes_mat["linearPosArgSort"])):
            neuron_first = (
                bayes_mat["linearPosArgSort"][i]
                if not plot_high_quality
                else high_quality_indices[i]
            )
            # check if neuron_first has at least 50 spikes in the wake/training epoch and 100 spikes overall
            spike_count = np.sum(self.trainerBayes.spikeMatLabels[:, neuron_first])
            if (
                spike_count > 100
                and nap.Ts(
                    self.trainerBayes.spikeMatTimes[
                        self.trainerBayes.spikeMatLabels[:, neuron_first] == 1
                    ].flatten()
                )
                .restrict(nap.IntervalSet(epoch))
                .shape[0]
                > 50
            ):
                break
        self._plot_single_place_field(
            axs[0], neuron_first, pos_x, pos_y, epoch, "First Ordered Place Field"
        )

        # --- Pre-calculate Linear Fields ---
        has_linear = hasattr(self.trainerBayes, "orderedLinearPlaceFields")
        train_lt_axes = []
        pred_lt_axes = []
        train_lt_im = None
        pred_lt_im = None
        train_cb_label = None
        pred_cb_label = None

        # --- Panel 1: All Linear Tuning Curves ---
        ax = axs[1]
        if has_linear:
            train_lt_im, train_cb_label = self.plot_linear_tuning_curves(
                ax=ax,
                add_colorbar=False,
                **kwargs,
            )
            train_lt_axes.append(ax)
        else:
            ax.axis("off")

        # --- Panel 2: Position Coverage ---
        ax = axs[3] if has_linear and hasattr(self, "decoded_fullBehavior") else axs[2]
        ax.hist(bayes_mat["linearPreferredPos"], bins=20, alpha=0.7, color="teal")
        ax.set_xlabel("Linear Position")
        ax.set_title("Pos Coverage in Training Data")

        # --- Panel 3: Predicted Linear Tuning Curves (mov epochs) or Quality Metrics (Mutual Info) ---
        ax = axs[4] if has_linear and hasattr(self, "decoded_fullBehavior") else axs[3]
        if has_linear and hasattr(self, "decoded_fullBehavior"):
            corr_speed = self._compute_linear_pf_correlation(
                self.behaviorData,
                self.decoded_fullBehavior,
                use_speed_filter=True,
                timeWindow=winMS,
            )
            title = (
                f"Predicted LT Curves (r={corr_speed:.3f})"
                if np.isfinite(corr_speed)
                else "Predicted LT Curves"
            )
            pred_lt_im, pred_cb_label = self.plot_linear_tuning_curves(
                ax=ax,
                fullBehavior=self.decoded_fullBehavior,
                title=title,
                is_predicted=True,
                sort_map=self.trainerBayes.linearPosArgSort,
                add_colorbar=False,
                **kwargs,
            )
            pred_lt_axes.append(ax)
        else:
            colors = np.array(["blue"] * len(ordered_mi))
            colors[high_quality_mask] = "red"

            ax.plot(ordered_mi, "o-", alpha=0.3, zorder=0, color="gray", linewidth=0.5)
            ax.scatter(
                np.arange(len(ordered_mi)),
                ordered_mi,
                c=colors,
                alpha=0.8,
                zorder=1,
                s=15,
            )
            ax.set_title("Mutual Information (ordered)")
            ax.set_xlabel("Neuron Index")
            ax.set_ylabel("Mutual Information")

        # --- Panel 4: Predicted Linear Tuning Curves (all speeds) ---
        ax = axs[5] if has_linear and hasattr(self, "decoded_fullBehavior") else axs[4]
        if has_linear:
            if hasattr(self, "decoded_fullBehavior"):
                corr_all = self._compute_linear_pf_correlation(
                    self.behaviorData,
                    self.decoded_fullBehavior,
                    use_speed_filter=False,
                    timeWindow=winMS,
                )
                title = (
                    f"Predicted LT Curves (All Speeds, r={corr_all:.3f})"
                    if np.isfinite(corr_all)
                    else "Predicted LT Curves (All Speeds)"
                )
                pred_lt_im, pred_cb_label = self.plot_linear_tuning_curves(
                    ax=ax,
                    fullBehavior=self.decoded_fullBehavior,
                    title=title,
                    is_predicted=True,
                    sort_map=self.trainerBayes.linearPosArgSort,
                    use_speed_filter=False,
                    add_colorbar=False,
                    **kwargs,
                )
                pred_lt_axes.append(ax)
            elif high_quality_mask.sum() > 0:
                print(
                    "No decoded bayes matrix provided, plotting original linear fields for high-quality neurons."
                )
                title = f"Best Linear Tuning Curves (Top {100 - thresh}%)"
                self.plot_linear_tuning_curves(
                    ax=ax, mask=high_quality_mask, title=title, **kwargs
                )
            else:
                ax.text(0.5, 0.5, "No High Quality Fields", ha="center", va="center")
                ax.axis("off")

        if has_linear and hasattr(self, "decoded_fullBehavior"):
            train_lt_im, train_cb_label = self.plot_linear_tuning_curves(
                ax=axs[2],
                fullBehavior=self.behaviorData,
                title="LT Curves (All Speeds)",
                is_predicted=False,
                sort_map=self.trainerBayes.linearPosArgSort,
                use_speed_filter=False,
                add_colorbar=False,
                **kwargs,
            )
            train_lt_axes.append(axs[2])
        else:
            # --- Panel 5: Last Ordered Place Field ---
            neuron_last = (
                bayes_mat["linearPosArgSort"][-1]
                if not plot_high_quality
                else high_quality_indices[-1]
            )
            self._plot_single_place_field(
                axs[5], neuron_last, pos_x, pos_y, epoch, "Last Ordered Place Field"
            )

        # Shared colorbars for paired linear tuning curves (same logic as error_map).
        if train_lt_im is not None and len(train_lt_axes) > 0:
            if cax_train is not None:
                train_cbar = fig.colorbar(
                    train_lt_im,
                    cax=cax_train,
                    orientation="horizontal",
                    location="bottom",
                )
            else:
                train_cbar = fig.colorbar(
                    train_lt_im,
                    ax=train_lt_axes,
                    shrink=0.5,
                    location="bottom",
                    orientation="horizontal",
                    pad=0.08,
                )
            if train_cb_label is not None:
                train_cbar.set_label(train_cb_label)
            train_cbar.outline.set_visible(False)
        elif cax_train is not None:
            cax_train.axis("off")

        if pred_lt_im is not None and len(pred_lt_axes) > 0:
            if cax_pred is not None:
                pred_cbar = fig.colorbar(
                    pred_lt_im,
                    cax=cax_pred,
                    orientation="horizontal",
                    location="bottom",
                )
            else:
                pred_cbar = fig.colorbar(
                    pred_lt_im,
                    ax=pred_lt_axes,
                    shrink=0.5,
                    location="bottom",
                    orientation="horizontal",
                    pad=0.08,
                )
            if pred_cb_label is not None:
                pred_cbar.set_label(pred_cb_label)
            pred_cbar.outline.set_visible(False)
        elif cax_pred is not None:
            cax_pred.axis("off")

        # --- Finalize and Save ---
        if save or show:
            if fig.get_layout_engine() is None:
                fig.tight_layout()

        if save:
            filename = f"bayesian_neurons_summary{self.suffix}{'_predicted' if is_predicted else ''}"
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
