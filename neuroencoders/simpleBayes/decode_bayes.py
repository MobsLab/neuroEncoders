# Load lib
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Literal, Optional, Tuple

import dill as pickle
import numpy as np
import pykeops as pykeops

# Pykeops
from pykeops.numpy import LazyTensor as LazyTensor_np
from tqdm import tqdm

from neuroencoders.importData import import_clusters
from neuroencoders.importData.epochs_management import inEpochs, inEpochsMask

# Load custom code
from neuroencoders.simpleBayes import butils
from neuroencoders.utils.global_classes import Project

# !!!! TODO: all train-test in one function, too much repetition
# TODO: option to remove zero cluster from training and testing

pykeops.set_verbose(False)


@dataclass
class DecoderConfig:
    """Configuration class for decoder parameters
    args:
        bandwidth: float, optional, bandwidth for kernel density estimation.
        kernel: str, optional, kernel type for KDE (default is "gaussian").
        masking_factor: float, optional, factor to mask occupation map (default is 20.0).
        min_spikes_threshold: int, optional, minimum number of spikes to consider a cluster (default is 10).
        regularization_factor: float, optional, regularization factor for rate functions (default is 1e-6).
        maxPos: tuple of float, optional, bounds for position normalization (default is None).
    """

    bandwidth: Optional[float] = None
    kernel: str = "gaussian"
    masking_factor: float = 20.0
    min_spikes_threshold: int = 5
    regularization_factor: float = 1e-8  # for numerical stability.
    empty_unit_value: float = 1e-5  # value for empty units in rate functions
    maxPos: Optional[Tuple[float, float]] = None
    fullBehaviorBandwidth: Optional[float] = None

    # Store unexpected kwargs here if needed
    extra_kwargs: dict = field(default_factory=dict, init=False, repr=False)

    def __init__(self, **kwargs):
        # Get all valid field names except extra_kwargs
        valid_fields = {f.name for f in self.__dataclass_fields__.values() if f.init}

        # Assign provided values for known fields
        for key in valid_fields:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
            else:
                setattr(self, key, self.__dataclass_fields__[key].default)

        # Store any unknown arguments without raising an error
        self.extra_kwargs = kwargs


# trainer class
class Trainer:
    def __init__(
        self,
        projectPath: Project,
        config: DecoderConfig = None,
        phase: Literal[
            "all",
            "pre",
            "preNoHab",
            "hab",
            "cond",
            "post",
            "postNoExtinction",
            "extinction",
            None,
        ] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the Trainer with project path and configuration.
        For config, you can pass a DecodDonfig object or keyword arguments that will be used to create a DecoderConfig object.
        Args:
            projectPath: Project object, with proper params and directories.
            config: DecoderConfig, optional, configuration for the decoder.
            phase: str, optional, phase of the experiment (default is None).
            **kwargs: additional keyword arguments for DecoderConfig.
        Raises:
            ValueError: If the projectPath is not provided or if the phase is not valid.
        """
        self.phase = phase
        self.projectPath = projectPath
        self.config = config or DecoderConfig(**kwargs)
        self.clusterData = import_clusters.load_spike_sorting(
            self.projectPath, phase=phase
        )
        self.verbose = verbose

        # Verbosity and logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Folders
        try:
            self.folderResult = os.path.join(self.projectPath.folderResult)
            self.folderResultSleep = os.path.join(self.projectPath.folderResultSleep)
        except AttributeError:
            self.folderResult = os.path.join(self.projectPath.experimentPath, "results")
            self.folderResultSleep = os.path.join(
                self.projectPath.experimentPath, "results_Sleep"
            )
        os.makedirs(self.folderResult, exist_ok=True)
        os.makedirs(self.folderResultSleep, exist_ok=True)

        # Initialize containers
        self.spike_matrices = None
        self.ordered_neurons = None
        self.place_fields = None

    def train(self, behaviorData: Dict, onTheFlyCorrection=False):
        """
        Main training function to build the Bayesian matrices.
        """
        self.logger.info("Starting Bayesian training process...")

        # look for bandwidth in behaviorData. If it is found compare it with the one in config
        if "Bandwidth" in behaviorData:
            self.config.fullBehaviorBandwidth = behaviorData["Bandwidth"]
        # Work with position coordinates
        speed_filtered_positions = behaviorData["Positions"][
            reduce(
                np.intersect1d,
                (
                    np.where(behaviorData["Times"]["speedFilter"]),
                    inEpochs(
                        behaviorData["positionTime"][:, 0],
                        behaviorData["Times"]["trainEpochs"],
                    ),
                ),
            )
        ]  # Get speed-filtered coordinates from train epoch

        maxPos = np.max(
            behaviorData["Positions"][
                np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))
            ]
        )
        if (
            onTheFlyCorrection
        ):  # setting the position to be between 0 and 1 if necessary
            speed_filtered_positions = speed_filtered_positions / maxPos

        positions = speed_filtered_positions[
            ~np.isnan(speed_filtered_positions).any(axis=1)
        ]

        ### Build global occupation map
        final_occupation, occupation, gridFeature = self._build_occupation_map(
            positions
        )

        ### Align the positions time with the spike_times so we can speed filter each spike time (long step)
        speed_filters = self._align_speed_filters(behaviorData)

        nTetrodes = len(self.clusterData["Spike_labels"])
        assert len(speed_filters) == nTetrodes, (
            "Number of tetrodes in speed filters does not match number of tetrodes in cluster data."
        )

        # Marginal rate is the rate function for the whole tetrode
        marginal_rates = []
        # local rate functions are the rate functions for each cluster
        local_rates = []
        spike_positions_all = []
        mutual_info_all = []

        ### Build marginal rate functions
        self.logger.info(
            f"Building marginal and local rate functions for {nTetrodes} tetrodes..."
        )
        for tetrode_idx in tqdm(range(nTetrodes)):
            # Get tetrode-wise data
            tetrode_results = self._process_tetrode(
                tetrode_idx,
                speed_filters[tetrode_idx],
                behaviorData,
                gridFeature,
                final_occupation,
                occupation,
                onTheFlyCorrection,
                maxPos,
            )

            # construct the matrix for this tetrode
            marginal_rates.append(tetrode_results["marginal_rate"])
            local_rates.append(tetrode_results["local_rates"])
            spike_positions_all.append(tetrode_results["spike_positions"])
            mutual_info_all.append(tetrode_results["mutual_info"])

        bayesMatrices = {
            "occupation": occupation,
            "marginalRateFunctions": marginal_rates,
            "rateFunctions": local_rates,
            "bins": [np.unique(gridFeature[i]) for i in range(len(gridFeature))],
            "spikePositions": spike_positions_all,
            "mutualInfo": mutual_info_all,
            "diagnostics": {
                "config": self.config,
                "nTetrodes": nTetrodes,
            },
        }

        # save the bayes matrices
        with open(os.path.join(self.folderResult, "bayesMatrices.pkl"), "wb") as f:
            pickle.dump(bayesMatrices, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info("Bayesian matrices saved successfully.")

        self.logger.info("Training completed successfully.")
        return bayesMatrices

    def train_order_by_pos(
        self, behaviorData: Dict, l_function=None, use_linear_tuning=False, **kwargs
    ) -> Dict:
        """
        Train the model and order the clusters by their preferred position.
        Args:
            behaviorData: dict, containing the position and time data.
            l_function: callable, linearization function.
            use_linear_tuning: bool, whether to use linear tuning for ordering.
            **kwargs: additional arguments including onTheFlyCorrection

        Returns:
            bayesMatrices: dict, containing the trained matrices for Bayesian inference.
        """
        self.logger.info("Training and ordering neurons by position preference...")

        # Get normalization setting from kwargs
        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)

        ### Perform training (build marginal and local rate functions)
        bayesMatrices = self.train(behaviorData, onTheFlyCorrection=onTheFlyCorrection)

        if use_linear_tuning and l_function is not None:
            # Use linear tuning curves for more accurate ordering
            self.logger.info("Computing linear tuning curves for ordering...")
            linear_place_fields, bin_edges = self.calculate_linear_tuning_curve(
                l_function, behaviorData
            )

            # Find preferred position for each neuron from linear tuning curve
            preferred_linear_positions = []
            for tuning_curve in linear_place_fields:
                if np.any(tuning_curve > 0):
                    # Find peak of tuning curve
                    peak_idx = np.argmax(tuning_curve)
                    # Convert bin index to position (center of bin)
                    preferred_pos = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                    preferred_linear_positions.append(preferred_pos)
                else:
                    # No spikes - assign to beginning
                    preferred_linear_positions.append(bin_edges[0])

            preferred_linear_positions = np.array(preferred_linear_positions)

            # Create ordering
            self.linearPosArgSort = np.argsort(preferred_linear_positions)
            self.linearPreferredPos = preferred_linear_positions[self.linearPosArgSort]

            # Store ordered place fields (both 2D and linear)
            place_fields = []
            for rate_group in bayesMatrices["rateFunctions"]:
                place_fields.extend(rate_group)
            self.orderedPlaceFields = np.array(place_fields)[self.linearPosArgSort]
            self.orderedLinearPlaceFields = np.array(linear_place_fields)[
                self.linearPosArgSort
            ]

            self.logger.info(
                f"Ordered {len(self.linearPosArgSort)} neurons by linear tuning curve"
            )
        else:
            # Extract preferred positions
            preferred_positions = self._extract_preferred_positions(bayesMatrices)

            # Apply linearization function
            if l_function is not None:
                _, linear_positions = l_function(preferred_positions)
            else:
                # Default linearization (just use x-coordinate)
                linear_positions = preferred_positions[:, 0]

            # Create ordering
            self.linearPosArgSort = np.argsort(linear_positions)
            self.linearPreferredPos = linear_positions[self.linearPosArgSort]

            # Store ordered place fields
            place_fields = []
            for rate_group in bayesMatrices["rateFunctions"]:
                place_fields.extend(rate_group)
            self.orderedPlaceFields = np.array(place_fields)[self.linearPosArgSort]

            if self.verbose:
                self.logger.info(
                    f"Ordered {len(self.linearPosArgSort)} neurons by position preference"
                )

        return bayesMatrices

    def _extract_preferred_positions(self, bayesMatrices: Dict) -> np.ndarray:
        """Extract preferred positions from rate functions"""
        preferred_pos = []

        for rate_group in bayesMatrices["rateFunctions"]:
            for rate_function in rate_group:
                # Find position of maximum firing rate
                max_pos_idx = np.unravel_index(
                    np.argmax(rate_function), rate_function.shape
                )
                preferred_pos.append(
                    [
                        bayesMatrices["bins"][0][max_pos_idx[1]],
                        bayesMatrices["bins"][1][max_pos_idx[0]],
                    ]
                )

        return np.array(preferred_pos)

    def _build_occupation_map(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Build occupation map with improved masking
        Args:
            positions: np.ndarray, 2D array of positions (x, y) for KDE.
        Returns:
            Tuple containing:
                - occupation_inverse: np.ndarray, inverse occupation map with regularization.
                - occupation: np.ndarray, original occupation map.
                - gridFeature: List, grid features used for KDE.
        """
        # Use provided bandwidth or compute adaptive bandwidth
        # Previously, default was fullBehaviorData["Bandwidth"]
        if self.config.bandwidth is None:
            # Scott's rule with some adjustment for 2D
            n_samples = len(positions)
            bandwidth = n_samples ** (-1 / (2 + 4)) * np.std(positions, axis=0)
            self.config.bandwidth = np.mean(bandwidth)
            self.logger.info(
                f"Using adaptive bandwidth: {self.config.bandwidth:.4f} (based on {n_samples} samples). To be compared with fullBehaviorData['Bandwidth']: {self.config.fullBehaviorBandwidth:.4f}"
            )

        gridFeature, occupation = butils.kdenD(
            positions, self.config.bandwidth, kernel=self.config.kernel
        )

        # Improved masking strategy - logistic thresholding
        occupation_threshold = np.max(occupation) / self.config.masking_factor
        sigma = 0.25 * occupation_threshold  # Adjust sigma for smoother transition
        mask = 1 / (1 + np.exp(-(occupation - occupation_threshold) / sigma))

        # Add regularization instead of just replacing zeros
        self.logger.info(
            f"Occupation map regularization factor: {self.config.regularization_factor / occupation.size:.4e}. before would have been changed by {np.min(occupation[occupation != 0])}"
        )
        eps = self.config.regularization_factor / occupation.size
        occupation_reg = occupation + eps
        occupation_reg /= np.sum(occupation_reg)  # renormalize to sum to 1
        occupation_inverse = 1 / occupation_reg
        occupation_inverse = np.multiply(occupation_inverse, mask)

        if self.verbose:
            self.logger.info(
                f"Occupation map: max={np.max(occupation):.4f}, "
                f"threshold={occupation_threshold:.4f}, "
                f"effective masking of {100 * (1 - np.mean(mask)):.2f}%"
            )

        return occupation_inverse, occupation, gridFeature

    def _compute_rate_function(
        self,
        spike_positions: np.ndarray,
        gridFeature: List,
        final_occupation: np.ndarray,
        n_spikes: int,
        learning_time: float,
    ) -> np.ndarray:
        """
        Compute rate function with better numerical stability
        """
        if n_spikes < self.config.min_spikes_threshold:
            if n_spikes == 0:
                warnings.warn("No spikes found, using uniform rate")
                rate_map = np.zeros_like(final_occupation)
                eps = self.config.regularization_factor / final_occupation.size
                rate_map += eps  # almost zero
                rate_map /= np.sum(rate_map)
                return rate_map
            else:
                warnings.warn(f"Only {len(spike_positions)} spikes, using uniform rate")
                # stronger uniform rate than above
                return np.ones_like(final_occupation) * self.config.empty_unit_value

        _, rate_map = butils.kdenD(
            spike_positions,
            self.config.bandwidth,
            edges=gridFeature,
            kernel=self.config.kernel,
        )

        # Normalize and apply occupation correction
        eps = self.config.empty_unit_value / rate_map.size
        rate_map = rate_map + eps
        rate_map = rate_map / np.sum(rate_map)
        rate_map = (n_spikes * np.multiply(rate_map, final_occupation)) / learning_time

        return rate_map

    def _align_speed_filters(self, behaviorData: Dict) -> List[np.ndarray]:
        """
        Align speed filters with spike times using PyKeops for efficient computation.
        """
        # convert position times to pykeops symbolic tensor (row vectors)
        pos_times = pykeops.numpy.Vj(behaviorData["positionTime"][:, 0][:, None])
        speed_filters = []

        self.logger.info("Aligning speed-filter with spike times using PyKeops")

        for tetrode_idx in tqdm(range(len(self.clusterData["Spike_labels"]))):
            # Convert spike times to PyKeops LazyTensor (column vectors with axis=0)
            spike_times = pykeops.numpy.LazyTensor(
                self.clusterData["Spike_times"][tetrode_idx][:, 0][:, None], axis=0
            )
            # find nearest position time for each spike time
            # matrix nb_spikes x nb_pos_times
            matching_pos_indices = (
                (pos_times - spike_times).abs().argmin_reduction(axis=1)
            )
            # get corresponding speed filter values
            speed_mask = behaviorData["Times"]["speedFilter"][matching_pos_indices]
            speed_filters += [speed_mask]

        self.logger.info("Speed filters aligned successfully")

        return speed_filters

    def _get_filtered_positions(
        self,
        tetrode_idx: int,
        speed_filter: np.ndarray,
        behaviorData: Dict,
        onTheFlyCorrection: bool,
        maxPos: Optional[float] = None,
    ) -> np.ndarray:
        """Get speed and train epoch filtered positions for a tetrode"""

        valid_indices = reduce(
            np.intersect1d,
            (
                np.where(speed_filter),
                inEpochs(
                    self.clusterData["Spike_times"][tetrode_idx][:, 0],
                    behaviorData["Times"]["trainEpochs"],
                ),
            ),
        )

        positions = self.clusterData["Spike_positions"][tetrode_idx][valid_indices]

        if onTheFlyCorrection and maxPos is not None:
            positions = positions / maxPos

        # Remove NaN positions
        positions = positions[~np.isnan(positions).any(axis=1)]
        return positions

    def _process_cluster(
        self,
        tetrode_idx: int,
        cluster_idx: int,
        speed_filter: np.ndarray,
        behaviorData: Dict,
        gridFeature: List,
        final_occupation: np.ndarray,
        raw_occupation: np.ndarray,
        onTheFlyCorrection: bool,
        maxPos: Optional[float] = None,
    ) -> Dict:
        """Process a single cluster: compute rate function and mutual information for train and speed filtered epochs.
        Args:
            tetrode_idx: int, index of the tetrode.
            cluster_idx: int, index of the cluster.
            speed_filter: np.ndarray, speed filter for this tetrode.
            behaviorData: dict, containing position and time data.
            gridFeature: List, grid features used for KDE.
            final_occupation: np.ndarray, final occupation map (1/occupation + regularization).
            raw_occupation: np.ndarray, original occupation map.
            onTheFlyCorrection: bool, whether to normalize positions on-the-fly.
            maxPos: float, optional, maximum position value for normalization.
        Returns:
            Dict containing:
                - rate: np.ndarray, computed rate function for the cluster.
                - positions: np.ndarray, positions of spikes in the cluster.
                - mutual_info: float, mutual information between rate and position.
        """

        # Get cluster-specific spike indices
        cluster_spikes = (
            self.clusterData["Spike_labels"][tetrode_idx][:, cluster_idx] == 1
        )

        valid_indices = reduce(
            np.intersect1d,
            (
                np.where(speed_filter),
                np.where(cluster_spikes),
                inEpochs(
                    self.clusterData["Spike_times"][tetrode_idx][:, 0],
                    behaviorData["Times"]["trainEpochs"],
                ),
            ),
        )

        cluster_positions = self.clusterData["Spike_positions"][tetrode_idx][
            valid_indices
        ]

        if onTheFlyCorrection and maxPos is not None:
            cluster_positions = cluster_positions / maxPos

        cluster_positions = cluster_positions[~np.isnan(cluster_positions).any(axis=1)]

        # Compute rate function
        if len(cluster_positions) >= self.config.min_spikes_threshold:
            rate_function = self._compute_rate_function(
                cluster_positions,
                gridFeature,
                final_occupation,
                len(cluster_positions),
                behaviorData["Times"]["learning"],
            )
        else:
            self.logger.warning(
                f"Cluster {cluster_idx} in tetrode {tetrode_idx} has only {len(cluster_positions)} spikes, using uniform rate."
            )
            if len(cluster_positions) != 0:
                rate_function = (
                    np.ones_like(final_occupation) * self.config.empty_unit_value
                )
            else:
                rate_function = np.zeros_like(final_occupation)
                eps = self.config.regularization_factor / final_occupation.size
                rate_function += eps
                rate_function /= np.sum(rate_function)

        # Compute mutual information
        mutual_info = self._compute_mutual_info(rate_function, raw_occupation)

        return {
            "rate": rate_function,
            "positions": cluster_positions,
            "mutual_info": mutual_info,
        }

    def _compute_mutual_info(
        self,
        rate_function: np.ndarray,
        occupation: np.ndarray,
        method: str = "Skaggs",
        n_rate_bins=5,
    ) -> float:
        """
        Compute mutual information between rate and position

        Args:
            rate_function: np.ndarray, local rate function for the cluster.
            occupation: np.ndarray, occupation map NOT inverse occupation map.

        Returns:
            float, mutual information value.

        TODO: handles other info measures, like entropy, I_sec or I_spike
        see (Souza et al., 2018).
        """
        valid_mask = rate_function > 0
        if not np.any(valid_mask):
            return 0.0

        mean_rate = np.mean(rate_function)
        if mean_rate <= 0:
            return 0.0

        if method.lower() == "skaggs" or method.lower() == "i_spike":  # bits/spike
            mi = np.sum(
                occupation[valid_mask]
                * rate_function[valid_mask]
                / mean_rate
                * np.log2(rate_function[valid_mask] / mean_rate)
            )
        elif method.lower == "i_sec":  # bits/sec
            mi = np.sum(
                occupation[valid_mask]
                * rate_function[valid_mask]
                * np.log2(rate_function[valid_mask] / mean_rate)
            )
        elif method == "shannon":  # raw-sample Shannon MI
            rate_flat = rate_function.flatten()
            occ_flat = occupation.flatten()
            valid_mask = (rate_flat > 0) & (occ_flat > 0)
            if not np.any(valid_mask):
                return 0.0
            rate_flat = rate_flat[valid_mask]
            # Bin firing rates into discrete categories
            rate_bins = np.linspace(
                np.min(rate_flat), np.max(rate_flat), n_rate_bins + 1
            )
            rate_digitized = np.digitize(rate_flat, rate_bins) - 1  # 0..n_rate_bins-1

            # Compute joint distribution p(position_bin, rate_bin)
            # Here each spatial bin contributes equally (raw sample, not weighted by occupation)
            joint_hist, _, _ = np.histogram2d(
                np.arange(len(rate_flat)),  # each bin index = position sample
                rate_digitized,
                bins=[len(rate_flat), n_rate_bins],
            )

            p_ij = joint_hist / np.sum(joint_hist)
            p_i = np.sum(p_ij, axis=1, keepdims=True)  # marginal over rate
            p_j = np.sum(p_ij, axis=0, keepdims=True)  # marginal over position

            mask = p_ij > 0
            mi = np.sum(p_ij[mask] * np.log2(p_ij[mask] / (p_i @ p_j)[mask]))
        else:
            raise ValueError(f"Unknown method {method}")

        return mi

    def _process_tetrode(
        self,
        tetrode_idx: int,
        speed_filter: np.ndarray,
        behaviorData: Dict,
        gridFeature: List,
        final_occupation: np.ndarray,
        occupation: np.ndarray,
        onTheFlyCorrection: bool,
        maxPos: Optional[float] = None,
    ) -> Dict:
        """Process a single tetrode.
        Args:
            tetrode_idx: int, index of the tetrode to process.
            speed_filter: np.ndarray, speed filter for this tetrode.
            behaviorData: dict, containing position and time data.
            gridFeature: List, grid features used for KDE.
            final_occupation: np.ndarray, final occupation map (1/occupation + regularization).
            occupation: np.ndarray, original occupation map.
            onTheFlyCorrection: bool, whether to normalize positions on-the-fly.
            maxPos: float, optional, maximum position value for normalization.
        """

        # Get tetrode positions
        tetrode_positions = self._get_filtered_positions(
            tetrode_idx,
            speed_filter,
            behaviorData,
            onTheFlyCorrection,
            maxPos,
        )

        # Compute marginal rate, which is the rate function for the whole tetrode
        marginal_rate = self._compute_rate_function(
            tetrode_positions,
            gridFeature,
            final_occupation,
            len(tetrode_positions),
            behaviorData["Times"]["learning"],
        )

        # Process each cluster in this tetrode
        n_clusters = self.clusterData["Spike_labels"][tetrode_idx].shape[1]
        local_rates = []
        spike_positions = []
        mutual_info = []

        for cluster_idx in range(n_clusters):
            # now process each cluster and find the local rate, which would be the rate function for each cluster
            cluster_results = self._process_cluster(
                tetrode_idx,
                cluster_idx,
                speed_filter,
                behaviorData,
                gridFeature,
                final_occupation,
                occupation,
                onTheFlyCorrection,
                maxPos,
            )

            local_rates.append(cluster_results["rate"])
            spike_positions.append(cluster_results["positions"])
            mutual_info.append(cluster_results["mutual_info"])

        return {
            "marginal_rate": marginal_rate,
            "local_rates": local_rates,
            "spike_positions": spike_positions,
            "mutual_info": mutual_info,
        }

    def test_legacy(
        self,
        bayesMatrices: Dict = None,
        behaviorData: Dict = None,
        windowSizeMS: int = 36,
        useTrain: bool = False,
        # New parameters for compatibility with new code (fallback from parallel_pred_as_NN)
        timeStepPred=None,
        all_poisson=None,
        clusters=None,
        clusters_time=None,
        log_rf=None,
        occupancy=None,
        return_full_posteriors=False,
        spatial_shape=None,
    ):
        """
        Legacy decoding method that can work in two modes:
        1. Original mode: With bayesMatrices and behavior_data
        2. Fallback mode: With parallel_pred_as_NN compatible arguments

        Args (Original mode):
            bayesMatrices: dict, Bayesian matrices
            behavior_data: dict, behavior data
            windowSizeMS: float, window size in milliseconds
            useTrain: bool, whether to use training data

        Args (Fallback mode):
            timeStepPred: array, time steps for prediction
            all_poisson: array, Poisson terms
            clusters: list, cluster data
            clusters_time: list, spike times
            log_rf: list, log rate functions
            occupancy: array, occupancy map
            return_full_posteriors: bool, return full posterior maps
            spatial_shape: tuple, spatial dimensions

        Returns:
            If fallback mode and return_full_posteriors=False: (max_probs, max_indices)
            If fallback mode and return_full_posteriors=True: (max_probs, max_indices, full_posteriors)
            If original mode: dict with full results
        """
        fallback_mode = (bayesMatrices is None or behaviorData is None) and (
            timeStepPred is not None
        )

        if fallback_mode:
            return self._test_legacy_fallback_mode(
                timeStepPred,
                windowSizeMS,
                all_poisson,
                clusters,
                clusters_time,
                log_rf,
                occupancy,
                return_full_posteriors,
                spatial_shape,
            )
        else:
            return self._test_legacy_original_mode(
                bayesMatrices, behaviorData, windowSizeMS, useTrain
            )

    def _test_legacy_fallback_mode(
        self,
        timeStepPred,
        windowSizeMS,
        all_poisson,
        clusters,
        clusters_time,
        log_rf,
        occupancy,
        return_full_posteriors,
        spatial_shape,
    ):
        """
        Legacy decoding in fallback mode - compatible with parallel_pred_as_NN arguments
        """

        self.logger.info("Running legacy decoding in fallback mode")

        # Convert window size back to seconds for internal logic
        windowSize = windowSizeMS / 1000

        n_bins = len(timeStepPred)

        # Reconstruct mask from occupancy
        mask = occupancy > (np.max(occupancy) / self.config.masking_factor)

        # Convert log_rf to the format expected by legacy code
        Rate_functions = []
        for tetrode_idx, tetrode_log_rf in enumerate(log_rf):
            tetrode_rates = []
            for cluster_log_rf in tetrode_log_rf:
                # Convert from log space back to linear space
                rate_function = np.exp(cluster_log_rf)
                tetrode_rates.append(rate_function)
            Rate_functions.append(tetrode_rates)

        ### Decoding loop (adapted from original legacy code)
        position_proba = [np.ones(occupancy.shape)] * n_bins
        max_probs = np.zeros(n_bins)
        max_indices = np.zeros(n_bins, dtype=int)

        for bin_idx in range(n_bins):
            bin_start_time = timeStepPred[bin_idx]
            bin_stop_time = bin_start_time + windowSize

            # Initialize with Poisson baseline
            tetrodes_contributions = [all_poisson]

            for tetrode in range(len(clusters)):
                # Find spikes in current window
                time_mask = (clusters_time[tetrode][:, 0] > bin_start_time) & (
                    clusters_time[tetrode][:, 0] < bin_stop_time
                )

                if np.any(time_mask):
                    bin_probas = clusters[tetrode][time_mask]
                    bin_clusters = np.sum(bin_probas, axis=0)

                    # Terms that come from spike information
                    if np.sum(bin_clusters) > 0.5:
                        spike_pattern = reduce(
                            np.multiply,
                            [
                                np.exp(log_rf[tetrode][cluster] * bin_clusters[cluster])
                                for cluster in range(len(bin_clusters))
                            ],
                        )
                    else:
                        spike_pattern = np.multiply(np.ones(occupancy.shape), mask)
                else:
                    spike_pattern = np.multiply(np.ones(occupancy.shape), mask)

                tetrodes_contributions.append(spike_pattern)

            # Compute posterior probability map
            position_proba[bin_idx] = reduce(np.multiply, tetrodes_contributions)

            # Add occupancy prior (convert from log space)
            occupancy_linear = np.exp(occupancy)
            position_proba[bin_idx] = position_proba[bin_idx] * occupancy_linear

            # Normalize
            total_prob = np.sum(position_proba[bin_idx])
            if total_prob > 0:
                position_proba[bin_idx] = position_proba[bin_idx] / total_prob
            else:
                position_proba[bin_idx] = np.ones_like(
                    position_proba[bin_idx]
                ) / np.prod(position_proba[bin_idx].shape)

            # Find maximum probability and index
            max_idx_flat = np.argmax(position_proba[bin_idx])
            max_probs[bin_idx] = position_proba[bin_idx].flat[max_idx_flat]
            max_indices[bin_idx] = max_idx_flat

            # Progress indicator
            if bin_idx % 50 == 0 and bin_idx > 0:
                self.logger.debug(
                    f"Legacy fallback progress: {bin_idx}/{n_bins} ({100 * bin_idx / n_bins:.1f}%)"
                )

        self.logger.info("Legacy fallback decoding completed")

        # Return in format compatible with parallel_pred_as_NN
        if return_full_posteriors:
            if spatial_shape is not None:
                full_posteriors = np.array(
                    [prob.reshape(spatial_shape) for prob in position_proba]
                )
            else:
                full_posteriors = np.array(position_proba)
            return (max_probs, max_indices, full_posteriors)
        else:
            return (max_probs, max_indices)

    def _test_legacy_original_mode(
        self, bayesMatrices, behaviorData, windowSizeMS, useTrain
    ):
        """
        Original legacy decoding implementation (unchanged)
        """
        # back to seconds
        windowSize = windowSizeMS / 1000

        print("\nBUILDING POSITION PROBAS")
        # find the spikes times in the test epochs
        if useTrain:
            epochsTrain = inEpochs(
                self.clusterData["Spike_times"][0][:, 0],
                behaviorData["Times"]["trainEpochs"],
            )
            epochsTest = inEpochs(
                self.clusterData["Spike_times"][0][:, 0],
                behaviorData["Times"]["testEpochs"],
            )
            epochs = np.sort(np.concatenate([epochsTrain[0], epochsTest[0]]))
        else:
            epochs = inEpochs(
                self.clusterData["Spike_times"][0][:, 0],
                behaviorData["Times"]["testEpochs"],
            )
        guessed_clusters_time = [
            self.clusterData["Spike_times"][tetrode][epochs]
            for tetrode in range(len(self.clusterData["Spike_times"]))
        ]
        # find the clusters/mua in the test epochs
        guessed_clusters = [
            self.clusterData["Spike_labels"][tetrode][epochs]
            for tetrode in range(len(self.clusterData["Spike_times"]))
        ]

        # load Bayes matrices
        Occupation, Marginal_rate_functions, Rate_functions = [
            bayesMatrices[key]
            for key in ["occupation", "marginalRateFunctions", "rateFunctions"]
        ]
        mask = Occupation > (np.max(Occupation) / self.config.masking_factor)

        ### Build Poisson term
        # first we bin the time
        testEpochs = behaviorData["Times"]["testEpochs"]
        # the total time of the test epochs
        Ttest = np.sum(
            [
                testEpochs[2 * i + 1] - testEpochs[2 * i]
                for i in range(len(testEpochs) // 2)
            ]
        )
        n_bins = math.floor(Ttest / windowSize)
        # for each bin we will need to now the test epoch it belongs to, so that we can then
        # set the time correctly to select the corresponding spikes
        timeEachTestEpoch = [
            testEpochs[2 * i + 1] - testEpochs[2 * i]
            for i in range(len(testEpochs) // 2)
        ]
        cumTimeEachTestEpoch = np.cumsum(timeEachTestEpoch)
        cumTimeEachTestEpoch = np.concatenate([[0], cumTimeEachTestEpoch])
        # a function that given the bin indicates the bin index:
        binToEpoch = lambda x: np.where(
            ((x * windowSize - cumTimeEachTestEpoch[0:-1]) >= 0)
            * ((x * windowSize - cumTimeEachTestEpoch[1:]) < 0)
        )[0][0]
        binToEpochArray = [binToEpoch(bins) for bins in range(n_bins)]
        firstBinEpoch = [
            np.min(np.where(np.equal(binToEpochArray, epochId))[0])
            for epochId in range(len(timeEachTestEpoch))
        ]
        All_Poisson_term = [
            np.exp((-windowSize) * Marginal_rate_functions[tetrode])
            for tetrode in range(len(guessed_clusters))
        ]
        All_Poisson_term = reduce(np.multiply, All_Poisson_term)

        ### Log of rate functions
        log_RF = []
        for tetrode in range(np.shape(Rate_functions)[0]):
            temp = []
            for cluster in range(np.shape(Rate_functions[tetrode])[0]):
                temp.append(
                    np.log(
                        Rate_functions[tetrode][cluster]
                        + np.min(
                            Rate_functions[tetrode][cluster][
                                Rate_functions[tetrode][cluster] != 0
                            ]
                        )
                    )
                )
            log_RF.append(temp)

        ### Decoding loop
        position_proba = [np.ones(np.shape(Occupation))] * n_bins
        position_true = [np.ones(2)] * n_bins
        nSpikes = []
        times = []
        for bin in range(n_bins):
            # Trouble: the test Epochs is discretized in continuous bin
            # whereas we forbid the use of some time steps b filtering them according to speed.
            bin_start_time = (
                testEpochs[2 * binToEpoch(bin)]
                + (bin - firstBinEpoch[binToEpoch(bin)]) * windowSize
            )
            bin_stop_time = bin_start_time + windowSize
            times.append(bin_start_time)

            binSpikes = 0
            tetrodes_contributions = []
            tetrodes_contributions.append(All_Poisson_term)

            for tetrode in range(len(guessed_clusters)):
                # Clusters inside our window
                bin_probas = guessed_clusters[tetrode][
                    np.intersect1d(
                        np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
                        np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time),
                    )
                ]
                bin_clusters = np.sum(bin_probas, 0)
                binSpikes = binSpikes + np.sum(bin_clusters)

                # Terms that come from spike information
                if np.sum(bin_clusters) > 0.5:
                    spike_pattern = reduce(
                        np.multiply,
                        [
                            np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
                            for cluster in range(np.shape(bin_clusters)[0])
                        ],
                    )
                else:
                    spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

                tetrodes_contributions.append(spike_pattern)

            nSpikes.append(binSpikes)

            # Guessed probability map
            position_proba[bin] = reduce(np.multiply, tetrodes_contributions)
            position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])
            # True position
            position_true_mean = np.nanmean(
                behaviorData["Positions"][
                    reduce(
                        np.intersect1d,
                        (
                            np.where(
                                behaviorData["positionTime"][:, 0] > bin_start_time
                            ),
                            np.where(
                                behaviorData["positionTime"][:, 0] < bin_stop_time
                            ),
                        ),
                    )
                ],
                axis=0,
            )
            position_true[bin] = (
                position_true[bin - 1]
                if np.isnan(position_true_mean).any()
                else position_true_mean
            )

            if bin % 50 == 0:
                sys.stdout.write(
                    "[%-30s] : %.3f %%"
                    % ("=" * (bin * 30 // n_bins), bin * 100 / n_bins)
                )
                sys.stdout.write("\r")
                sys.stdout.flush()
        sys.stdout.write(
            "[%-30s] : %.3f %%"
            % ("=" * ((bin + 1) * 30 // n_bins), (bin + 1) * 100 / n_bins)
        )
        sys.stdout.write("\r")
        sys.stdout.flush()

        position_true[0] = position_true[1]
        print("\nDecoding finished")

        # Guessed X and Y
        allProba = [
            np.unravel_index(np.argmax(position_proba[bin]), position_proba[bin].shape)
            for bin in range(len(nSpikes))
        ]
        bestProba = [np.max(position_proba[bin]) for bin in range(len(nSpikes))]
        position_guessed = [
            [
                bayesMatrices["bins"][i][allProba[bin][i]]
                for i in range(len(bayesMatrices["bins"]))
            ]
            for bin in range(len(nSpikes))
        ]
        inferResults = np.concatenate(
            [np.array(position_guessed), np.array(bestProba).reshape([-1, 1])], axis=-1
        )

        # Update the dict to comply to neuroencoders v2, new nomenclature.
        outputResults = {
            "inferring": inferResults,
            "pos": np.array(position_true),
            "featureTrue": np.array(position_true),
            "probaMaps": position_proba,
            "times": np.array(times),
            "nSpikes": np.array(nSpikes),
            "featurePred": inferResults[:, :2],
            "proba": inferResults[:, 2],
            "speed_mask": behaviorData["Times"]["speedFilter"],
        }
        return outputResults

    def test_as_NN(
        self,
        behaviorData: Dict,
        bayesMatrices: Dict,
        timeStepPred: np.ndarray,
        windowSizeMS: float = 36,
        useTrain: bool = False,
        sleepEpochs: List = [],
        l_function=None,
        cross_validate: bool = False,
        cv_folds: int = 5,
        save_posteriors: bool = False,
        save_as_pickle: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Test the model using the neural network approach with improved evaluation and cross-validation.

        Args:
            behaviorData: dict, containing the position and time data.
            bayesMatrices: dict, containing the precomputed matrices for Bayesian inference.
            timeStepPred: array, time steps for prediction.
            windowSizeMS: int, size of the window in milliseconds.
            useTrain: bool, whether to use training epochs for prediction.
            sleepEpochs: list, epochs to consider for sleep decoding.
            l_function: callable, optional linearization function.
            cross_validate: bool, whether to perform cross-validation.
            cv_folds: int, number of folds for cross-validation.
            save_posteriors: bool, whether to save full posterior maps.
            save_as_pickle: bool, whether to save results as a pickle file.

        Returns:
            outputResults: dict, containing predictions, performance metrics, and optionally CV results.
        """

        windowSize = windowSizeMS / 1000
        self.logger.info(f"Starting Bayesian decoding with {windowSizeMS}ms windows")

        # Prepare spike data for the specified epochs
        clusters_time, clusters = self._prepare_spike_data(
            behaviorData, useTrain, sleepEpochs
        )

        # Extract and preprocess Bayesian matrices
        log_terms = self._prepare_bayesian_terms(bayesMatrices, windowSize)

        # Main decoding using the efficient PyKeOps implementation
        self.logger.info("Running parallel PyKeOps Bayesian decoding...")
        decoded_output = self._run_parallel_decoding(
            timeStepPred,
            windowSize,
            clusters_time,
            clusters,
            log_terms,
            save_posteriors,
        )
        self.logger.info("Finished bayesien guess, will now process and clean.")

        # Process and clean the decoded results
        processed_results = self._process_decoding_output(
            decoded_output,
            timeStepPred,
            windowSize,
            clusters_time,
            clusters,
            log_terms,
            bayesMatrices,
            save_posteriors,
        )

        # Get ground truth positions
        featureTrue = self._get_ground_truth_positions(
            timeStepPred, behaviorData, useTrain
        )

        # Compute comprehensive performance metrics
        performance_metrics = self._compute_detailed_performance(
            processed_results, featureTrue, l_function
        )

        # Perform cross-validation if requested
        cv_results = None
        if cross_validate:
            self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
            cv_results = self._perform_cross_validation(
                behaviorData,
                bayesMatrices,
                windowSizeMS,
                l_function,
                cv_folds,
                useTrain,
                sleepEpochs,
            )

        # Compile final results
        outputResults = {
            "featurePred": processed_results["positions"],
            "proba": processed_results["confidence"],
            "times": timeStepPred,
            "featureTrue": featureTrue,
            "posLoss": performance_metrics["posLoss"],
            "performance": performance_metrics,
            "speed_mask": behaviorData["Times"]["speedFilter"],
            "decoding_params": {
                "windowSizeMS": windowSizeMS,
                "useTrain": useTrain,
                "n_time_steps": len(timeStepPred),
                "n_nan_fixed": processed_results.get("n_nan_fixed", 0),
            },
        }

        # Add posterior maps if requested
        if save_posteriors and "posterior_maps" in processed_results:
            outputResults["probaMaps"] = processed_results["posterior_maps"]

        # Add cross-validation results
        if cv_results is not None:
            outputResults["cross_validation"] = cv_results

        # Apply linearization if provided
        if l_function is not None:
            outputResults = self._apply_linearization_transform(
                outputResults, l_function
            )

        # Save results
        self.saveResults(
            outputResults,
            folderName=windowSizeMS,
            phase=kwargs.get("phase", self.phase),
            cross_validate=cross_validate,
            save_as_pickle=save_as_pickle,
        )

        # Log summary statistics
        self._log_decoding_summary(outputResults)

        return outputResults

    def _prepare_spike_data(
        self, behaviorData: Dict, useTrain: bool, sleepEpochs: List
    ) -> Tuple[List, List]:
        """Prepare spike data for decoding epochs.
        Args:
            behaviorData: dict, containing the position and time data.
            useTrain: bool, whether to use training epochs for prediction.
            sleepEpochs: list, epochs to consider for sleep decoding.
        """

        n_tetrodes = len(self.clusterData["Spike_times"])
        clusters_time = []
        clusters = []

        for tetrode in range(n_tetrodes):
            if useTrain:
                epochs_train = inEpochsMask(
                    self.clusterData["Spike_times"][tetrode][:, 0],
                    behaviorData["Times"]["trainEpochs"],
                )
                epochs_test = inEpochsMask(
                    self.clusterData["Spike_times"][tetrode][:, 0],
                    behaviorData["Times"]["testEpochs"],
                )
                combined_epochs = epochs_train + epochs_test
            elif len(sleepEpochs) > 0:
                combined_epochs = inEpochs(
                    self.clusterData["Spike_times"][tetrode][:, 0], sleepEpochs
                )
            else:
                combined_epochs = inEpochs(
                    self.clusterData["Spike_times"][tetrode][:, 0],
                    behaviorData["Times"]["testEpochs"],
                )

            clusters_time.append(
                self.clusterData["Spike_times"][tetrode][combined_epochs]
            )
            clusters.append(self.clusterData["Spike_labels"][tetrode][combined_epochs])

        return clusters_time, clusters

    def _prepare_bayesian_terms(self, bayesMatrices: Dict, windowSize: float) -> Dict:
        """Prepare and validate Bayesian terms for decoding"""

        occupation = bayesMatrices["occupation"]
        marginal_rate_functions = bayesMatrices["marginalRateFunctions"]
        rate_functions = bayesMatrices["rateFunctions"]

        # Create mask for valid positions
        mask = occupation > (np.max(occupation) / self.config.masking_factor)

        # Log occupation with regularization
        log_occupation = np.log(occupation + np.min(occupation[mask]))

        # Poisson terms (probability of no spikes)
        all_poisson = [
            np.exp(-windowSize * marginal_rate_functions[tetrode])
            for tetrode in range(len(marginal_rate_functions))
        ]
        all_poisson = np.log(reduce(np.multiply, all_poisson))

        # Log rate functions with improved numerical stability
        log_rf = []
        for tetrode in range(len(rate_functions)):
            tetrode_log_rf = []
            for cluster in range(len(rate_functions[tetrode])):
                rate_map = rate_functions[tetrode][cluster]
                min_nonzero = (
                    np.min(rate_map[rate_map != 0])
                    if np.any(rate_map != 0)
                    else self.config.regularization_factor
                )
                log_rate = np.log(rate_map + min_nonzero)
                tetrode_log_rf.append(log_rate)
            log_rf.append(tetrode_log_rf)

        return {
            "all_poisson": all_poisson,
            "log_rf": log_rf,
            "log_occupation": log_occupation,
            "mask": mask,
        }

    def _run_parallel_decoding(
        self,
        timeStepPred: np.ndarray,
        windowSize: float,
        clusters_time: List,
        clusters: List,
        log_terms: Dict,
        save_posteriors: bool = False,
    ) -> Tuple:
        """Run the parallel PyKeOps decoding (unchanged core algorithm)"""

        spatial_shape = log_terms["all_poisson"].shape if save_posteriors else None

        output_pos = parallel_pred_as_NN(
            timeStepPred,
            windowSize,
            log_terms["all_poisson"],
            clusters,
            clusters_time,
            log_terms["log_rf"],
            log_terms["log_occupation"],
            return_full_posteriors=save_posteriors,
            spatial_shape=spatial_shape,
        )
        # TODO: add error handling and non-parallel fallback
        return output_pos

    def _process_decoding_output(
        self,
        decoded_output: Tuple,
        timeStepPred: np.ndarray,
        windowSize: float,
        clusters_time: List,
        clusters: List,
        log_terms: Dict,
        bayesMatrices: Dict,
        save_posteriors: bool = False,
    ) -> Dict:
        """Process the raw decoding output with improved NaN handling"""

        # Extract positions and probabilities from PyKeOps output
        # Check if we got full posteriors
        if save_posteriors and len(decoded_output) == 3:
            max_probs, max_indices, full_posteriors = decoded_output
        else:
            max_probs, max_indices = decoded_output
            full_posteriors = None

        # Convert indices back to spatial coordinates
        id_pos = np.unravel_index(max_indices, shape=log_terms["all_poisson"].shape)
        featurePred = np.array(
            [
                bayesMatrices["bins"][i][id_pos[i][:, 0]]
                for i in range(len(bayesMatrices["bins"]))
            ]
        ).T

        # Get confidence values (convert back from log scale)
        confidences = max_probs

        # Handle NaN values with improved fallback
        nan_mask = np.isnan(confidences)
        n_nan_fixed = 0

        if np.any(nan_mask):
            self.logger.warning(
                f"Found {np.sum(nan_mask)} NaN values, fixing with manual computation..."
            )
            n_nan_fixed = np.sum(nan_mask)

            for idx in np.where(nan_mask)[0]:
                manual_result = self._manual_decode_single_window(
                    timeStepPred[idx],
                    windowSize,
                    clusters_time,
                    clusters,
                    log_terms,
                    bayesMatrices,
                )
                featurePred[idx] = manual_result["position"]
                confidences[idx] = manual_result["confidence"]

                # Fix posterior map if available
                if save_posteriors and full_posteriors is not None:
                    uniform_posterior = np.ones_like(log_terms["all_poisson"])
                    uniform_posterior = uniform_posterior / np.sum(uniform_posterior)
                    full_posteriors[idx] = uniform_posterior

        # Ensure no remaining NaNs
        confidences[np.isnan(confidences)] = 0.0

        result = {
            "positions": featurePred,
            "confidence": confidences,
            "n_nan_fixed": n_nan_fixed,
        }
        # Add posterior maps if available
        if full_posteriors is not None:
            result["posterior_maps"] = full_posteriors
            self.logger.info(f"Full posterior maps computed: {full_posteriors.shape}")
            # Validate the posterior maps
            self._validate_posterior_maps(full_posteriors)

        return result

    def _validate_posterior_maps(self, posterior_maps: np.ndarray):
        """Validate the computed posterior maps for common issues"""

        validation_results = {
            "shape": posterior_maps.shape,
            "total_memory_gb": posterior_maps.nbytes / (1024**3),
            "has_nan": np.any(np.isnan(posterior_maps)),
            "has_inf": np.any(np.isinf(posterior_maps)),
            "has_negative": np.any(posterior_maps < 0),
            "min_sum": np.min(np.sum(posterior_maps, axis=(1, 2))),
            "max_sum": np.max(np.sum(posterior_maps, axis=(1, 2))),
            "mean_max_prob": np.mean(
                np.max(posterior_maps.reshape(len(posterior_maps), -1), axis=1)
            ),
        }

        # Log validation results
        self.logger.info("=== Posterior Map Validation ===")
        self.logger.info(f"Shape: {validation_results['shape']}")
        self.logger.info(f"Memory: {validation_results['total_memory_gb']:.2f} GB")
        self.logger.info(
            f"Probability sum range: [{validation_results['min_sum']:.3f}, {validation_results['max_sum']:.3f}]"
        )
        self.logger.info(
            f"Mean max probability: {validation_results['mean_max_prob']:.3f}"
        )

        # Check for issues
        issues = []
        if validation_results["has_nan"]:
            issues.append("Contains NaN values")
        if validation_results["has_inf"]:
            issues.append("Contains infinite values")
        if validation_results["has_negative"]:
            issues.append("Contains negative probabilities")
        if validation_results["min_sum"] < 0.95:
            issues.append("Some posteriors don't sum to ~1.0")
        if validation_results["mean_max_prob"] < 0.01:
            issues.append("Very low confidence predictions")

        if issues:
            self.logger.warning("Posterior validation issues found:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("Posterior maps passed validation")

        self.logger.info("================================")

    def _manual_decode_single_window(
        self,
        time_point: float,
        windowSize: float,
        clusters_time: List,
        clusters: List,
        log_terms: Dict,
        bayesMatrices: Dict,
    ) -> Dict:
        """Manual decoding for a single time window (fallback for NaN cases)"""

        bin_start_time = time_point
        bin_stop_time = time_point + windowSize

        # Start with Poisson and occupation terms
        tetrode_contributions = [log_terms["all_poisson"]]

        # Add each tetrode's contribution
        for tetrode in range(len(clusters)):
            # Find spikes in time window
            time_mask = np.logical_and(
                clusters_time[tetrode][:, 0] > bin_start_time,
                clusters_time[tetrode][:, 0] < bin_stop_time,
            )

            if np.any(time_mask):
                bin_spikes = clusters[tetrode][time_mask]
                bin_clusters = np.sum(bin_spikes, axis=0)

                # Compute spike contribution
                if np.sum(bin_clusters) > 0.5:
                    spike_pattern = reduce(
                        lambda a, b: a + b,
                        [
                            log_terms["log_rf"][tetrode][cluster]
                            * bin_clusters[cluster]
                            for cluster in range(len(bin_clusters))
                        ],
                    )
                else:
                    spike_pattern = np.zeros_like(log_terms["log_occupation"])
            else:
                spike_pattern = np.zeros_like(log_terms["log_occupation"])

            tetrode_contributions.append(spike_pattern)

        # Combine all contributions
        position_posterior = reduce(lambda a, b: a + b, tetrode_contributions)
        position_posterior += log_terms["log_occupation"]

        # Apply mask and normalize
        position_posterior = np.where(log_terms["mask"], position_posterior, -np.inf)

        # Convert to probabilities
        position_posterior_stable = position_posterior - np.max(position_posterior)
        position_probs = np.exp(position_posterior_stable)
        position_probs = position_probs / np.sum(position_probs)

        # Find maximum
        max_idx = np.unravel_index(np.argmax(position_probs), position_probs.shape)
        predicted_position = [
            bayesMatrices["bins"][0][max_idx[1]],
            bayesMatrices["bins"][1][max_idx[0]],
        ]

        return {"position": predicted_position, "confidence": np.max(position_probs)}

    def _get_ground_truth_positions(
        self, timeStepPred: np.ndarray, behaviorData: Dict, useTrain: bool
    ) -> np.ndarray:
        """Get ground truth positions for evaluation"""

        # Get position data for relevant epochs
        if useTrain:
            train_mask = inEpochsMask(
                behaviorData["positionTime"][:, 0],
                behaviorData["Times"]["trainEpochs"],
            )
            test_mask = inEpochsMask(
                behaviorData["positionTime"][:, 0],
                behaviorData["Times"]["testEpochs"],
            )
            epoch_mask = train_mask + test_mask
        else:
            epoch_mask = inEpochsMask(
                behaviorData["positionTime"][:, 0],
                behaviorData["Times"]["testEpochs"],
            )

        real_positions = behaviorData["Positions"][epoch_mask]
        real_times = behaviorData["positionTime"][epoch_mask]

        # Find nearest position for each prediction time
        featureTrue = np.zeros((len(timeStepPred), 2))
        for i, time_stamp in enumerate(timeStepPred):
            nearest_idx = np.abs(real_times - time_stamp).argmin()
            featureTrue[i] = real_positions[nearest_idx]

        return featureTrue

    def _compute_detailed_performance(
        self, processed_results: Dict, featureTrue: np.ndarray, l_function=None
    ) -> Dict:
        """Compute comprehensive performance metrics"""

        featurePred = processed_results["positions"].reshape(-1, 2)
        confidences = processed_results["confidence"].reshape(-1)

        # Basic position errors
        position_errors = np.linalg.norm(featurePred - featureTrue, axis=1)

        # Core metrics
        metrics = {
            "posLoss": position_errors,
            "mean_error": np.mean(position_errors),
            "median_error": np.median(position_errors),
            "std_error": np.std(position_errors),
            "rmse": np.sqrt(np.mean(position_errors**2)),
            "max_error": np.max(position_errors),
            "min_error": np.min(position_errors),
            "error_percentiles": {
                "25th": np.percentile(position_errors, 25),
                "75th": np.percentile(position_errors, 75),
                "90th": np.percentile(position_errors, 90),
                "95th": np.percentile(position_errors, 95),
                "99th": np.percentile(position_errors, 99),
            },
        }

        # Confidence-based metrics
        metrics.update(
            {
                "mean_confidence": np.mean(confidences),
                "median_confidence": np.median(confidences),
                "std_confidence": np.std(confidences),
                "confidence_error_correlation": np.corrcoef(
                    confidences.reshape(-1), position_errors.reshape(-1)
                )[0, 1],
            }
        )

        # Binned performance analysis
        metrics["binned_performance"] = self._analyze_binned_performance(
            position_errors, confidences
        )

        # Linear track specific metrics
        if l_function is not None:
            linear_metrics = self._compute_linear_performance(
                featurePred, featureTrue, l_function
            )
            metrics["linear_track"] = linear_metrics

        return metrics

    def _analyze_binned_performance(
        self, position_errors: np.ndarray, confidences: np.ndarray
    ) -> Dict:
        """Analyze performance in different confidence bins"""

        # Bin by confidence levels
        conf_bins = np.percentile(confidences, [0, 25, 50, 75, 90, 100])
        binned_errors = []

        for i in range(len(conf_bins) - 1):
            mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1])
            if np.any(mask):
                binned_errors.append(np.mean(position_errors[mask]))
            else:
                binned_errors.append(np.nan)

        return {
            "confidence_bins": conf_bins,
            "mean_errors_per_bin": binned_errors,
            "n_samples_per_bin": [
                np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1]))
                for i in range(len(conf_bins) - 1)
            ],
        }

    def _compute_linear_performance(
        self, featurePred: np.ndarray, featureTrue: np.ndarray, l_function
    ) -> Dict:
        """Compute performance metrics for linear track"""

        try:
            _, linear_pred = l_function(featurePred)
            _, linearTrue = l_function(featureTrue)

            linear_errors = np.abs(linear_pred - linearTrue)

            return {
                "mean_error": np.mean(linear_errors),
                "median_error": np.median(linear_errors),
                "std_error": np.std(linear_errors),
                "rmse": np.sqrt(np.mean(linear_errors**2)),
                "correlation": np.corrcoef(linear_pred, linearTrue)[0, 1],
            }
        except Exception as e:
            self.logger.warning(f"Linear performance computation failed: {e}")
            return {"error": str(e)}

    def _perform_cross_validation(
        self,
        behaviorData: Dict,
        bayesMatrices: Dict,
        windowSizeMS: float,
        l_function,
        cv_folds: int,
        useTrain: bool,
        sleepEpochs: List,
    ) -> Dict:
        """Perform k-fold cross-validation of the decoder"""

        if useTrain or len(sleepEpochs) > 0:
            self.logger.warning(
                "Cross-validation not implemented for train or sleep epochs"
            )
            return {"error": "CV not available for train/sleep epochs"}

        try:
            # Get test epoch data
            test_epochs = behaviorData["Times"]["testEpochs"]

            if len(test_epochs) == 0:
                return {"error": "No test epochs available for cross-validation"}

            # create time-based folds from test epochs
            fold_epochs = self._create_temporal_folds(test_epochs, cv_folds)
            cv_results = {
                "fold_errors": [],
                "fold_confidences": [],
                "fold_correlations": [],
                "fold_linear_errors": [],
                "fold_details": [],
            }

            for fold in range(cv_folds):
                self.logger.info(f"Processing fold {fold + 1}/{cv_folds}")

                try:
                    # Split data into train and validation for this fold
                    train_epochs_cv = []
                    val_epochs_cv = []

                    for f_idx, fold_epoch in enumerate(fold_epochs):
                        if f_idx == fold:
                            val_epochs_cv.extend(fold_epoch)  # This fold is validation
                        else:
                            train_epochs_cv.extend(
                                fold_epoch
                            )  # Other folds are training

                    if len(train_epochs_cv) == 0 or len(val_epochs_cv) == 0:
                        self.logger.warning(
                            f"Skipping fold {fold + 1}: insufficient data"
                        )
                        continue

                    # Create modified behavior data for this fold
                    fold_behavior_data = behaviorData.copy()
                    fold_behavior_data["Times"] = behaviorData["Times"].copy()
                    fold_behavior_data["Times"]["trainEpochs"] = train_epochs_cv
                    fold_behavior_data["Times"]["testEpochs"] = val_epochs_cv

                    # Retrain the decoder on the training folds
                    self.logger.debug(f"Retraining decoder for fold {fold + 1}")
                    fold_bayes_matrices = self.train(
                        fold_behavior_data, onTheFlyCorrection=True
                    )

                    # Generate test time points for validation fold
                    val_times = []
                    for epoch in val_epochs_cv:
                        epoch_duration = epoch[1] - epoch[0]
                        n_steps = int(epoch_duration / (windowSizeMS / 1000))
                        if n_steps > 0:
                            times = np.linspace(
                                epoch[0], epoch[1] - windowSizeMS / 1000, n_steps
                            )
                            val_times.extend(times)

                    if len(val_times) == 0:
                        self.logger.warning(f"No validation times for fold {fold + 1}")
                        continue

                    val_times = np.array(val_times)

                    # Test on validation fold (without cross-validation to avoid recursion)
                    fold_results = self.test_as_NN(
                        fold_behavior_data,
                        fold_bayes_matrices,
                        val_times,
                        windowSizeMS=windowSizeMS,
                        use_train=False,
                        sleep_epochs=[],
                        l_function=l_function,
                        cross_validate=False,  # Important: no nested CV
                    )

                    # Extract performance metrics for this fold
                    fold_perf = fold_results["performance"]

                    cv_results["fold_errors"].append(fold_perf["mean_error"])
                    cv_results["fold_confidences"].append(fold_perf["mean_confidence"])

                    # Correlation between predicted and true positions
                    pred_pos = fold_results["predicted_positions"]
                    true_pos = fold_results["true_positions"]

                    if len(pred_pos) > 1 and len(true_pos) > 1:
                        # Compute correlation for each dimension
                        corr_x = np.corrcoef(pred_pos[:, 0], true_pos[:, 0])[0, 1]
                        corr_y = np.corrcoef(pred_pos[:, 1], true_pos[:, 1])[0, 1]
                        mean_corr = (
                            np.mean([corr_x, corr_y])
                            if not np.isnan([corr_x, corr_y]).any()
                            else 0.0
                        )
                        cv_results["fold_correlations"].append(mean_corr)
                    else:
                        cv_results["fold_correlations"].append(0.0)

                    # Linear track performance if available
                    if (
                        "linear_track" in fold_perf
                        and "mean_error" in fold_perf["linear_track"]
                    ):
                        cv_results["fold_linear_errors"].append(
                            fold_perf["linear_track"]["mean_error"]
                        )
                    else:
                        cv_results["fold_linear_errors"].append(np.nan)

                    # Store detailed results for this fold
                    cv_results["fold_details"].append(
                        {
                            "fold": fold + 1,
                            "train_epochs": len(train_epochs_cv),
                            "val_epochs": len(val_epochs_cv),
                            "n_val_points": len(val_times),
                            "performance": fold_perf,
                        }
                    )

                    self.logger.info(
                        f"Fold {fold + 1} completed - Error: {fold_perf['mean_error']:.2f}"
                    )

                except Exception as e:
                    self.logger.error(f"Fold {fold + 1} failed: {e}")
                    cv_results["fold_errors"].append(np.nan)
                    cv_results["fold_confidences"].append(np.nan)
                    cv_results["fold_correlations"].append(np.nan)
                    cv_results["fold_linear_errors"].append(np.nan)
                    continue

            # Compute summary statistics
            valid_errors = [e for e in cv_results["fold_errors"] if not np.isnan(e)]
            valid_confidences = [
                c for c in cv_results["fold_confidences"] if not np.isnan(c)
            ]
            valid_correlations = [
                r for r in cv_results["fold_correlations"] if not np.isnan(r)
            ]

            if len(valid_errors) > 0:
                cv_summary = {
                    "mean_cv_error": np.mean(valid_errors),
                    "std_cv_error": np.std(valid_errors),
                    "median_cv_error": np.median(valid_errors),
                    "min_cv_error": np.min(valid_errors),
                    "max_cv_error": np.max(valid_errors),
                    "mean_cv_confidence": np.mean(valid_confidences)
                    if valid_confidences
                    else np.nan,
                    "mean_cv_correlation": np.mean(valid_correlations)
                    if valid_correlations
                    else np.nan,
                    "successful_folds": len(valid_errors),
                    "total_folds": cv_folds,
                    "fold_success_rate": len(valid_errors) / cv_folds,
                }

                # Add linear track CV results if available
                valid_linear = [
                    e for e in cv_results["fold_linear_errors"] if not np.isnan(e)
                ]
                if valid_linear:
                    cv_summary["mean_cv_linear_error"] = np.mean(valid_linear)
                    cv_summary["std_cv_linear_error"] = np.std(valid_linear)

            else:
                cv_summary = {"error": "All folds failed"}

            cv_results.update(cv_summary)

            self.logger.info(
                f"Cross-validation completed: {len(valid_errors)}/{cv_folds} folds successful"
            )
            if len(valid_errors) > 0:
                self.logger.info(
                    f"CV Performance: {cv_summary['mean_cv_error']:.2f} ± {cv_summary['std_cv_error']:.2f}"
                )

            return cv_results

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {"error": str(e)}

    def _create_temporal_folds(self, test_epochs: List, cv_folds: int) -> List[List]:
        """
        Create temporal folds from test epochs.
        Splits epochs temporally to ensure good coverage across time.
        """

        # Flatten all test epochs into continuous time segments
        all_segments = []
        for epoch in test_epochs:
            all_segments.append(epoch)

        # Sort by start time
        all_segments.sort(key=lambda x: x[0])

        if len(all_segments) < cv_folds:
            self.logger.warning(
                f"Only {len(all_segments)} epochs available for {cv_folds} folds"
            )
            # Assign each epoch to a different fold
            fold_epochs = [[] for _ in range(cv_folds)]
            for i, segment in enumerate(all_segments):
                fold_epochs[i % cv_folds].append(segment)
            return fold_epochs

        # Calculate total duration
        total_duration = sum(segment[1] - segment[0] for segment in all_segments)
        target_duration_per_fold = total_duration / cv_folds

        # Assign segments to folds based on cumulative duration
        fold_epochs = [[] for _ in range(cv_folds)]
        current_fold = 0
        current_fold_duration = 0

        for segment in all_segments:
            segment_duration = segment[1] - segment[0]

            # If adding this segment would exceed target duration and we're not on the last fold
            if (
                current_fold_duration + segment_duration > target_duration_per_fold
                and current_fold < cv_folds - 1
                and len(fold_epochs[current_fold]) > 0
            ):
                current_fold += 1
                current_fold_duration = 0

            fold_epochs[current_fold].append(segment)
            current_fold_duration += segment_duration

        # Log fold information
        for i, fold_segs in enumerate(fold_epochs):
            if fold_segs:
                fold_dur = sum(seg[1] - seg[0] for seg in fold_segs)
                self.logger.debug(
                    f"Fold {i + 1}: {len(fold_segs)} epochs, {fold_dur:.1f}s duration"
                )

        return fold_epochs

    def _apply_linearization_transform(self, results: Dict, l_function) -> Dict:
        """Apply linearization function to results"""

        try:
            # Apply to predictions
            proj_pred, linear_pred = l_function(results["featurePred"])
            proj_true, linearTrue = l_function(results["featureTrue"])

            results.update(
                {
                    "projPred": proj_pred,
                    "projTruePos": proj_true,
                    "linearPred": linear_pred,
                    "linearTrue": linearTrue,
                }
            )

        except Exception as e:
            self.logger.error(f"Linearization failed: {e}")
            results["linearization_error"] = str(e)

        return results

    def _log_decoding_summary(self, outputResults: Dict):
        """Log a summary of decoding performance"""

        perf = outputResults["performance"]
        params = outputResults["decoding_params"]

        self.logger.info("=== Decoding Summary ===")
        self.logger.info(f"Window size: {params['windowSizeMS']}ms")
        self.logger.info(f"Time steps: {params['n_time_steps']}")
        self.logger.info(f"Mean error: {perf['mean_error']:.2f} units")
        self.logger.info(f"Median error: {perf['median_error']:.2f} units")
        self.logger.info(f"RMSE: {perf['rmse']:.2f} units")
        self.logger.info(f"Mean confidence: {perf['mean_confidence']:.3f}")

        if params.get("n_nan_fixed", 0) > 0:
            self.logger.warning(f"Fixed {params['n_nan_fixed']} NaN predictions")

        if "cross_validation" in outputResults:
            cv = outputResults["cross_validation"]
            if "mean_cv_error" in cv:
                self.logger.info(
                    f"CV error: {cv['mean_cv_error']:.2f} ± {cv['std_cv_error']:.2f}"
                )

        self.logger.info("========================")

    # def test_as_NN(
    #     self,
    #     behaviorData,
    #     bayesMatrices,
    #     timeStepPred,
    #     windowSizeMS=36,
    #     useTrain=False,
    #     sleepEpochs=[],
    #     l_function=None,
    #     **kwargs,
    # ):
    #     """
    #     Test the model using the neural network approach.
    #
    #     Args:
    #     --------
    #         behaviorData: dict, containing the position and time data.
    #         bayesMatrices: dict, containing the precomputed matrices for Bayesian inference.
    #         timeStepPred: array, time steps for prediction.
    #         windowSizeMS: int, size of the window in milliseconds.
    #         useTrain: bool, whether to use training epochs for prediction.
    #         sleepEpochs: list, epochs to consider for sleep decoding.
    #         l_function: callable, optional linearization function.
    #
    #     Returns:
    #     --------
    #         outputResults: dict, containing the predicted positions, probabilities, and true positions.
    #
    #     """
    #     windowSize = windowSizeMS / 1000
    #
    #     if useTrain:
    #         epochsTrain = [
    #             inEpochsMask(
    #                 self.clusterData["Spike_times"][tetrode][:, 0],
    #                 behaviorData["Times"]["trainEpochs"],
    #             )
    #             for tetrode in range(len(self.clusterData["Spike_times"]))
    #         ]
    #         epochsTest = [
    #             inEpochsMask(
    #                 self.clusterData["Spike_times"][tetrode][:, 0],
    #                 behaviorData["Times"]["testEpochs"],
    #             )
    #             for tetrode in range(len(self.clusterData["Spike_times"]))
    #         ]
    #         clustersTime = [
    #             self.clusterData["Spike_times"][tetrode][
    #                 epochsTrain[tetrode] + epochsTest[tetrode]
    #             ]
    #             for tetrode in range(len(self.clusterData["Spike_times"]))
    #         ]
    #         clusters = [
    #             self.clusterData["Spike_labels"][tetrode][
    #                 epochsTrain[tetrode] + epochsTest[tetrode]
    #             ]
    #             for tetrode in range(len(self.clusterData["Spike_times"]))
    #         ]
    #     else:
    #         if len(sleepEpochs) > 0:
    #             clustersTime = [
    #                 self.clusterData["Spike_times"][tetrode][
    #                     inEpochs(
    #                         self.clusterData["Spike_times"][tetrode][:, 0], sleepEpochs
    #                     )
    #                 ]
    #                 for tetrode in range(len(self.clusterData["Spike_times"]))
    #             ]
    #             clusters = [
    #                 self.clusterData["Spike_labels"][tetrode][
    #                     inEpochs(
    #                         self.clusterData["Spike_times"][tetrode][:, 0], sleepEpochs
    #                     )
    #                 ]
    #                 for tetrode in range(len(self.clusterData["Spike_times"]))
    #             ]
    #         else:
    #             clustersTime = [
    #                 self.clusterData["Spike_times"][tetrode][
    #                     inEpochs(
    #                         self.clusterData["Spike_times"][tetrode][:, 0],
    #                         behaviorData["Times"]["testEpochs"],
    #                     )
    #                 ]
    #                 for tetrode in range(len(self.clusterData["Spike_times"]))
    #             ]
    #             clusters = [
    #                 self.clusterData["Spike_labels"][tetrode][
    #                     inEpochs(
    #                         self.clusterData["Spike_times"][tetrode][:, 0],
    #                         behaviorData["Times"]["testEpochs"],
    #                     )
    #                 ]
    #                 for tetrode in range(len(self.clusterData["Spike_times"]))
    #             ]
    #
    #     print("\nBUILDING POSITION PROBAS")
    #     occupation, marginalRateFunctions, rateFunctions = [
    #         bayesMatrices[key]
    #         for key in ["occupation", "marginalRateFunctions", "rateFunctions"]
    #     ]
    #     mask = occupation > (np.max(occupation) / self.maskingFactor)
    #     logOccupation = np.log(occupation + np.min(occupation[mask]))
    #     ### Build Poisson term
    #     allPoisson = [
    #         np.exp((-windowSize) * marginalRateFunctions[tetrode])
    #         for tetrode in range(len(clusters))
    #     ]
    #     allPoisson = np.log(reduce(np.multiply, allPoisson))
    #     ### Log of rate functions
    #     logRF = []
    #     for tetrode in range(np.shape(rateFunctions)[0]):
    #         temp = []
    #         for cluster in range(np.shape(rateFunctions[tetrode])[0]):
    #             temp.append(
    #                 np.log(
    #                     rateFunctions[tetrode][cluster]
    #                     + np.min(
    #                         rateFunctions[tetrode][cluster][
    #                             rateFunctions[tetrode][cluster] != 0
    #                         ]
    #                     )
    #                 )
    #             )
    #         logRF.append(temp)
    #
    #     ### Decoding loop
    #     print("Parallel pykeops bayesian test")
    #     outputPOps = parallel_pred_as_NN(
    #         timeStepPred,
    #         windowSize,
    #         allPoisson,
    #         clusters,
    #         clustersTime,
    #         logRF,
    #         logOccupation,
    #     )
    #     print("finished bayesian guess")
    #
    #     idPos = np.unravel_index(outputPOps[1], shape=allPoisson.shape)
    #     inferredPos = np.array(
    #         [
    #             bayesMatrices["bins"][i][idPos[i][:, 0]]
    #             for i in range(len(bayesMatrices["bins"]))
    #         ]
    #     )
    #     # probability moved back to linear scale
    #     # inferredProba = np.exp(outputPOps[0]) / np.sum(np.exp(outputPOps[0]), axis=0)
    #     inferredProba = outputPOps[0]
    #     inferResults = np.concatenate(
    #         [np.transpose(inferredPos), inferredProba], axis=-1
    #     )
    #
    #     # NOTE: A few values of probability predictions present NaN in pykeops
    #     print("Resolving nan issue from pykeops over a few bins")
    #     badBins = np.where(np.isnan(inferResults[:, 2]))[0]
    #     for bin in badBins:
    #         binStartTime = timeStepPred[bin]
    #         binStopTime = binStartTime + windowSize
    #         tetrodesContributions = []
    #         tetrodesContributions.append(allPoisson)
    #         for tetrode in range(len(clusters)):
    #             binProbas = clusters[tetrode][
    #                 np.intersect1d(
    #                     np.where(clustersTime[tetrode][:, 0] > binStartTime),
    #                     np.where(clustersTime[tetrode][:, 0] < binStopTime),
    #                 )
    #             ]
    #             # Note:  we would lose some spikes if we used the clusterData[Spike_pos_index]
    #             # because some spike might be closest to one position further away than windowSize,
    #             # yet themselves be close to the spike time
    #             binClusters = np.sum(binProbas, 0)
    #             # Terms that come from spike information
    #             if np.sum(binClusters) > 0.5:
    #                 spikePattern = reduce(
    #                     lambda a, b: a + b,
    #                     [
    #                         (logRF[tetrode][cluster] + binClusters[cluster])
    #                         for cluster in range(np.shape(binClusters)[0])
    #                     ],
    #                 )
    #             else:
    #                 spikePattern = np.multiply(np.zeros(np.shape(logOccupation)), mask)
    #             tetrodesContributions.append(spikePattern)
    #         # Guessed probability map
    #         positionProba = reduce(lambda a, b: a + b, tetrodesContributions)
    #         positionProba = (
    #             positionProba + logOccupation
    #         )  # prior: Occupation deduced from training!!
    #         # probability moved back to linear scale
    #         positionProba = np.exp(positionProba) / np.sum(np.exp(positionProba))
    #         inferResults[bin, 2] = np.max(positionProba)
    #         inferResults[np.isnan(inferResults[:, 2]), 2] = 0  # to correct for overflow
    #
    #     # Get the true position
    #     idTestEpoch = inEpochsMask(
    #         behaviorData["positionTime"][:, 0], behaviorData["Times"]["testEpochs"]
    #     )
    #     if useTrain:
    #         idTrainEpoch = inEpochsMask(
    #             behaviorData["positionTime"][:, 0], behaviorData["Times"]["trainEpochs"]
    #         )
    #         idTestEpoch = idTrainEpoch + idTestEpoch
    #     realPos = behaviorData["Positions"][idTestEpoch]
    #     realTimes = behaviorData["positionTime"][idTestEpoch]
    #     idsNN = []
    #     for timeStamp in timeStepPred:
    #         idsNN.append(np.abs(realTimes - timeStamp).argmin())
    #     idsNN = np.array(idsNN)
    #     featTrue = realPos[idsNN]
    #
    #     outputResults = {
    #         "featurePred": inferResults[:, :2],
    #         "proba": inferResults[:, 2],
    #         "times": timeStepPred,
    #         "featureTrue": featTrue,
    #         "speed_mask": behaviorData["Times"]["speedFilter"],
    #     }  # , "probaMaps": position_proba
    #
    #     if l_function:
    #         projPredPos, linearPred = l_function(inferResults[:, :2])
    #         projTruePos, linearTrue = l_function(featTrue)
    #         outputResults["projPred"] = projPredPos
    #         outputResults["projTruePos"] = projTruePos
    #         outputResults["linearPred"] = linearPred
    #         outputResults["linearTrue"] = linearTrue
    #
    #     self.saveResults(outputResults, folderName=windowSizeMS, phase=phase)
    #
    #     return outputResults

    def sleep_decoding(
        self,
        behaviorData,
        bayesMatrices,
        windowSizeMS=36,
        l_function=None,
        save_as_pickle=False,
    ):
        windowSize = windowSizeMS / 1000

        clustersTime = {}
        clusters = {}
        inferResultsDic = {}
        for id, sleepName in enumerate(behaviorData["Times"]["sleepNames"]):
            clustersTime[sleepName] = [
                self.clusterData["Spike_times"][tetrode][
                    inEpochs(
                        self.clusterData["Spike_times"][tetrode][:, 0],
                        behaviorData["Times"]["sleepEpochs"][2 * id : 2 * id + 2],
                    )
                ]
                for tetrode in range(len(self.clusterData["Spike_times"]))
            ]
            clusters[sleepName] = [
                self.clusterData["Spike_labels"][tetrode][
                    inEpochs(
                        self.clusterData["Spike_times"][tetrode][:, 0],
                        behaviorData["Times"]["sleepEpochs"][2 * id : 2 * id + 2],
                    )
                ]
                for tetrode in range(len(self.clusterData["Spike_times"]))
            ]

            print("\nBUILDING POSITION PROBAS")
            occupation, marginalRateFunctions, rateFunctions = [
                bayesMatrices[key]
                for key in ["occupation", "marginalRateFunctions", "rateFunctions"]
            ]
            mask = occupation > (np.max(occupation) / self.config.masking_factor)
            logOccupation = np.log(occupation + np.min(occupation[mask]))
            ### Build Poisson term
            allPoisson = [
                np.exp((-windowSize) * marginalRateFunctions[tetrode])
                for tetrode in range(len(clusters[sleepName]))
            ]
            allPoisson = np.log(reduce(np.multiply, allPoisson))

            ### Log of rate functions
            logRF = []
            for tetrode in range(np.shape(rateFunctions)[0]):
                temp = []
                for cluster in range(np.shape(rateFunctions[tetrode])[0]):
                    temp.append(
                        np.log(
                            rateFunctions[tetrode][cluster]
                            + np.min(
                                rateFunctions[tetrode][cluster][
                                    rateFunctions[tetrode][cluster] != 0
                                ]
                            )
                        )
                    )
                logRF.append(temp)

            timeStepPred = behaviorData["positionTime"][
                inEpochs(
                    behaviorData["positionTime"][:, 0],
                    behaviorData["Times"]["sleepEpochs"][2 * id : 2 * id + 2],
                )
            ]
            # For prediction time step we use the time step of measured speed and feature of the animal.
            # A prediction is made over all time steps in the test set
            # and the prediction results can be latter on filter out.

            print("Parallel pykeops bayesian test")
            outputPOps = parallel_pred_as_NN(
                timeStepPred,
                windowSize,
                allPoisson,
                clusters[sleepName],
                clustersTime[sleepName],
                logRF,
                logOccupation,
            )
            print("finished bayesian guess")

            idPos = np.unravel_index(outputPOps[1], shape=allPoisson.shape)
            inferredPos = np.array(
                [
                    bayesMatrices["bins"][i][idPos[i][:, 0]]
                    for i in range(len(bayesMatrices["bins"]))
                ]
            )
            # probability moved back to linear scale
            # No need to normalize the prob here as it's now done in the parallel_pred_as_NN
            # inferredProba = np.exp(outputPOps[0]) / np.sum(
            #     np.exp(outputPOps[0]), axis=0
            # )
            inferredProba = outputPOps[0]
            inferResults = np.concatenate(
                [np.transpose(inferredPos), inferredProba], axis=-1
            )

            # NOTE: A few values of probability predictions present NaN in pykeops....
            print("Resolving nan issue from pykeops over a few bins")
            badBins = np.where(np.isnan(inferResults[:, 2]))[0]
            for bin in badBins:
                binStartTime = timeStepPred[bin] - windowSize / 2
                binStopTime = binStartTime + windowSize / 2
                tetrodesContributions = []
                tetrodesContributions.append(allPoisson)
                for tetrode in range(len(clusters[sleepName])):
                    binProbas = clusters[sleepName][tetrode][
                        np.intersect1d(
                            np.where(
                                clustersTime[sleepName][tetrode][:, 0] > binStartTime
                            ),
                            np.where(
                                clustersTime[sleepName][tetrode][:, 0] < binStopTime
                            ),
                        )
                    ]
                    # Note:  we would lose some spikes if we used the clusterData[Spike_pos_index]
                    # because some spike might be closest to one position further away than windowSize,
                    # yet themselves be close to the spike time
                    binClusters = np.sum(binProbas, 0)
                    # Terms that come from spike information
                    if np.sum(binClusters) > 0.5:
                        spikePattern = reduce(
                            lambda a, b: a + b,
                            [
                                (logRF[tetrode][cluster] + binClusters[cluster])
                                for cluster in range(np.shape(binClusters)[0])
                            ],
                        )
                    else:
                        spikePattern = np.multiply(
                            np.zeros(np.shape(logOccupation)), mask
                        )
                    tetrodesContributions.append(spikePattern)
                # Guessed probability map
                positionProba = reduce(lambda a, b: a + b, tetrodesContributions)
                positionProba = (
                    positionProba + logOccupation
                )  # prior: Occupation deduced from training!!
                # probability moved back to linear scale
                positionProba = np.exp(positionProba) / np.sum(np.exp(positionProba))
                inferResults[bin, 2] = np.max(positionProba)
                inferResults[np.isnan(inferResults[:, 2]), 2] = (
                    0  # to correct for overflow
                )

            inferResultsDic[sleepName] = {
                "featurePred": inferResults[:, :2],
                "proba": inferResults[:, 2],
                "times": timeStepPred,
                "speed_mask": behaviorData["Times"]["speedFilter"],
            }

            if l_function:
                projPredPos, linearPred = l_function(inferResults[:, :2])
                inferResultsDic[sleepName]["projPred"] = projPredPos
                inferResultsDic[sleepName]["linearPred"] = linearPred

        # Save the results
        for key in inferResultsDic.keys():
            self.saveResults(
                inferResultsDic[key],
                folderName=windowSizeMS,
                sleep=True,
                sleepName=key,
                save_as_pickle=save_as_pickle,
            )
        return inferResultsDic

    def calculate_linear_tuning_curve(self, linearization_function, behaviorData):
        """
        Calculate the linear tuning curve for each cell based on spike times and position data.
        """
        linearPlaceFields = []
        # Create one large epoch that comprises both train and test dataset
        minTime = np.min(
            np.concatenate(
                (
                    behaviorData["Times"]["trainEpochs"],
                    behaviorData["Times"]["testEpochs"],
                )
            )
        )
        maxTime = np.max(
            np.concatenate(
                (
                    behaviorData["Times"]["trainEpochs"],
                    behaviorData["Times"]["testEpochs"],
                )
            )
        )
        epochForField = np.array([minTime, maxTime])
        _, linearTraj = linearization_function(behaviorData["Positions"])
        timesMask = inEpochs(np.squeeze(behaviorData["positionTime"]), epochForField)[0]
        timeLinear = np.squeeze(behaviorData["positionTime"][timesMask, :])
        linearTraj = linearTraj[timesMask]
        linSpace = np.arange(min(linearTraj), max(linearTraj), step=0.01)
        histPos, binEdges = np.histogram(linearTraj, bins=linSpace)

        for tetrode in range(len(self.clusterData["Spike_times"])):
            spikeTimesTetrode = np.squeeze(self.clusterData["Spike_times"][tetrode])
            for icell in range(self.clusterData["Spike_labels"][tetrode].shape[1]):
                cellMask = self.clusterData["Spike_labels"][tetrode][:, icell] == 1
                spikeTimes = spikeTimesTetrode[cellMask]
                spikeTimes = spikeTimes[inEpochs(spikeTimes, epochForField)]

                # Find position of the animal at the time of each spike
                if spikeTimes.any():
                    spikeTimesLazy = pykeops.numpy.Vj(
                        spikeTimes[:, None].astype(dtype=np.float64)
                    )
                    timeLinearLazy = pykeops.numpy.Vi(
                        timeLinear[:, None].astype(dtype=np.float64)
                    )
                    idPosInSpikes = (
                        (spikeTimesLazy - timeLinearLazy).abs().argmin(axis=0)
                    )[:, 0]
                    spikePos = linearTraj[idPosInSpikes]
                    # Create histogram of spike position (find P(spikes|position))
                    histSpikes, binEdges = np.histogram(spikePos, bins=linSpace)
                    # Find tuning curve
                    histTuning = histSpikes / histPos
                    linearPlaceFields.append(histTuning)  # save
                else:
                    linearPlaceFields.append(np.zeros(len(linSpace) - 1))

        return linearPlaceFields, binEdges

    def saveResults(
        self,
        test_output: Dict,
        folderName: float = 36,
        sleep: bool = False,
        sleepName: str = "Sleep",
        phase=None,
        cross_validate: bool = False,
        save_as_pickle: bool = True,
    ) -> None:
        import pandas as pd

        # Manage folders to save
        if sleep:
            folderToSave = os.path.join(
                self.folderResultSleep, str(folderName), sleepName
            )
            if not os.path.isdir(folderToSave):
                os.makedirs(folderToSave)
        else:
            folderToSave = os.path.join(self.folderResult, str(folderName))

        if phase is not None:
            suffix = f"_{phase}"
        else:
            suffix = f"_{self.suffix}"
        if cross_validate:
            suffix += "_cv"

        # predicted coordinates
        df = pd.DataFrame(test_output["featurePred"])
        df.to_csv(os.path.join(folderToSave, f"bayes_featurePred{suffix}.csv"))
        df = pd.DataFrame(test_output["proba"])
        df.to_csv(os.path.join(folderToSave, f"bayes_proba{suffix}.csv"))

        # True coordinates
        if not sleep:
            df = pd.DataFrame(test_output["featureTrue"])
            df.to_csv(os.path.join(folderToSave, f"bayes_featureTrue{suffix}.csv"))
            # Position loss
            df = pd.DataFrame(test_output["posLoss"])
            df.to_csv(os.path.join(folderToSave, f"bayes_posLoss{suffix}.csv"))

        # Full posterior distribution
        if "probaMaps" in test_output:
            df = pd.DataFrame(test_output["probaMaps"])
            df.to_csv(os.path.join(folderToSave, f"bayes_probaMaps{suffix}.csv"))

        # Times of prediction
        df = pd.DataFrame(test_output["times"])
        df.to_csv(os.path.join(folderToSave, f"bayes_timeStepsPred{suffix}.csv"))
        # Speed mask
        if not sleep:
            df = pd.DataFrame(test_output["speed_mask"])
            df.to_csv(os.path.join(folderToSave, f"bayes_speedMask{suffix}.csv"))

        if "indexInDat" in test_output:
            df = pd.DataFrame(test_output["indexInDat"])
            df.to_csv(os.path.join(folderToSave, f"bayes_indexInDat{suffix}.csv"))
        if "projPred" in test_output:
            df = pd.DataFrame(test_output["projPred"])
            df.to_csv(os.path.join(folderToSave, f"bayes_projPredFeature{suffix}.csv"))
        if "linearPred" in test_output:
            df = pd.DataFrame(test_output["linearPred"])
            df.to_csv(os.path.join(folderToSave, f"bayes_linearPred{suffix}.csv"))

        if not sleep:
            if "projTruePos" in test_output:
                df = pd.DataFrame(test_output["projTruePos"])
                df.to_csv(
                    os.path.join(folderToSave, f"bayes_projTrueFeature{suffix}.csv")
                )
            if "linearTrue" in test_output:
                df = pd.DataFrame(test_output["linearTrue"])
                df.to_csv(os.path.join(folderToSave, f"bayes_linearTrue{suffix}.csv"))

        if save_as_pickle:
            # save the whole results dictionary
            filename = os.path.join(folderToSave, f"bayes_decoding_results{suffix}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(test_output, f, pickle.HIGHEST_PROTOCOL)


############## Utils ##############
def find_next_bin(times, clusters, start, stop, start_time, stop_time):
    # times: array of times
    # start: last start index
    # stop: last stop index
    # start_time: bin start time
    # stop_time: bin stop time
    newStartID = stop
    while times[newStartID] < start_time:
        newStartID += 1
    newStopId = newStartID + 1
    while times[newStopId] < stop_time and newStopId <= len(times) - 1:
        newStopId += 1
    newStopId = newStopId - 1
    return newStartID, newStopId, clusters[newStartID : newStopId + 1]


# TODO: le passer en test_parallel??
def parallel_pred_as_NN(
    firstSpikeNNtime,
    windowSize,
    allPoisson,
    clusters,
    clustersTime,
    logRF,
    occupancy,
    return_full_posteriors=False,
    spatial_shape=None,
):
    """
    Predict the position of the animal using a Bayesian approach, in parallel over all bins.
    Optionally returns the full posterior distribution for all timesteps.

    Args:
    --------
        firstSpikeNNtime: array, time of the first spike in the neural network.
        windowSize: float, size of the window in seconds.
        allPoisson: array, Poisson term for each tetrode.
        clusters: list of arrays, clusters for each tetrode.
        clustersTime: list of arrays, time of the spikes for each tetrode.
        logRF: list of arrays, log rate functions for each tetrode.
        occupancy: array, occupancy probability for the position.
        return_full_posteriors: bool, whether to return the full posterior distribution.
        spatial_shape: tuple, shape of the spatial grid (required if return_full_posteriors, for reshaping output).

    Returns:
    --------
        outputPos: tuple, predicted position and probability.
        if return_full_posteriors is True:
            full_posteriors: array, full posterior distribution for all timesteps
                                (reshaped to (n_time_steps, *spatial_shape)).


    # Use pykeops library to perform an efficient computation of the predicted position, in parallel over all bins.
    # Note: here achieved on the CPU, could also be ported to the GPU by using torch tensor....
    # Here everything in log scale to avoid numerical overflow
    """

    binStartTime = firstSpikeNNtime
    binStopTime = binStartTime + windowSize

    # we will progressively add each tetrode contribution
    tetrodeContribs = 0
    for tetrode in range(len(clusters)):
        gctLazy = LazyTensor_np(clustersTime[tetrode][:, None])
        binStartTimeLazy = LazyTensor_np(binStartTime[None, :])
        binStopTimeLazy = LazyTensor_np(binStopTime[None, :])
        goodStart = (
            (gctLazy - binStartTimeLazy).relu().sign()
        )  # similar to gct_lazy > bin_start_times.lazy
        goodStop = (binStopTimeLazy - gctLazy).relu().sign()
        # size: (Number of signal time step,Number of prediction bin,1), indicate for each bin the time step in the bin.
        goodBins = goodStart * goodStop
        gcLazy = LazyTensor_np(clusters[tetrode][:, None, :])
        # For each bin, we gather for each cluster in the tetrode the number of spike detected in signal measurements inside this bin.
        # gathering can be effectively implemented by a element wise matrix multiplication with the mask good_bins
        binClusters = (gcLazy * goodBins).sum(axis=0)
        # # transform into an array of size (Nb bin,Nb cluster in tetrode)

        # Prepare for pykeops operations:
        logRF_r = np.transpose(np.array(logRF[tetrode]), axes=[1, 2, 0])
        logRF_r = np.reshape(
            logRF_r, newshape=[np.prod(logRF_r.shape[0:-1]), logRF_r.shape[-1]]
        )
        logRFLazy = LazyTensor_np(logRF_r[None, :, :])
        binClustersLazy = LazyTensor_np(binClusters[:, None, :])

        # the Log firing rate of each cluster is multiplied by the number of bin cluster, and the sum is performed over the
        # number of cluster in the tetrode
        res = (logRFLazy * binClustersLazy).sum(dim=-1)
        tetrodeContribs = tetrodeContribs + res

    # Finally we need to add the Poisson terms common to all tetrode finalS
    # position posterior estimation:
    poisson_r = np.reshape(allPoisson, newshape=[np.prod(allPoisson.shape)])[:, None]
    poissonContribVj = pykeops.numpy.Vj(poisson_r)
    tetrodeContribs = tetrodeContribs + poissonContribVj

    # The probability need to be weighted by the position probabilities:
    occupancy_r = np.reshape(occupancy, newshape=[np.prod(occupancy.shape)])[:, None]
    occupancyContrib = pykeops.numpy.Vj(occupancy_r)
    tetrodeContribs = tetrodeContribs + occupancyContrib

    # If we had only one electrode:
    # ... but we need to sum over the different electrodes.
    # first, we need to normalize the contributions
    tetrodeContribs_stable = tetrodeContribs - pykeops.numpy.Vi(
        tetrodeContribs.max(axis=1)
    )
    tetrodeContribs_linear = tetrodeContribs_stable.exp()
    posterior = tetrodeContribs_linear / pykeops.numpy.Vi(
        tetrodeContribs_linear.sum(axis=1)
    )

    outputPos = posterior.max_argmax_reduction(axis=1)
    max_probs, max_indices = outputPos[0], outputPos[1]

    if return_full_posteriors:
        if spatial_shape is None:
            raise ValueError(
                "spatial_shape must be provided when return_full_posteriors=True"
            )

        # Convert LazyTensor to numpy array and reshape
        # posterior shape: (n_positions, n_time_steps)
        posterior_array = np.array(
            posterior
        ).T  # Transpose to (n_time_steps, n_positions)

        # Reshape to original spatial dimensions
        n_time_steps = len(firstSpikeNNtime)
        full_posteriors = posterior_array.reshape((n_time_steps, *spatial_shape))

        return (max_probs, max_indices, full_posteriors)
    else:
        # Original return format
        return (max_probs, max_indices)


############## Utils ##############


############## Legacy trainer ##############
class LegacyTrainer:
    def __init__(
        self, projectPath, bandwidth=None, kernel="gaussian", masking_factor=20
    ):  # 'epanechnikov' - TODO?
        self.projectPath = projectPath
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.masking_factor = masking_factor
        self.clusterData = import_clusters.load_spike_sorting(self.projectPath)

    def train_order_by_pos(self, behaviorData, l_function=None, **kwargs):
        """
        Train the model and order the clusters by their preferred position.
        Args:
            behaviorData: dict, containing the position and time data.
            l_function: callable, optional linearization function.
            onTheFlyCorrection: bool, whether to correct the positions on the fly.

        Returns:
            bayesMatrices: dict, containing the trained matrices for Bayesian inference.
        """

        onTheFlyCorrection = kwargs.get("onTheFlyCorrection", False)

        # Gather all spikes in large array and sort it in time
        nbSpikes = [a.shape[0] for a in self.clusterData["Spike_labels"]]
        nbNeurons = [a.shape[1] for a in self.clusterData["Spike_labels"]]
        spikeMatLabels = np.zeros([np.sum(nbSpikes), np.sum(nbNeurons)])
        spikeMatTimes = np.zeros([np.sum(nbSpikes), 1])
        cnbSpikes = np.cumsum(nbSpikes)
        cnbNeurons = np.cumsum(nbNeurons)
        for id in range(len(nbSpikes)):
            if id > 0:
                spikeMatLabels[
                    cnbSpikes[id - 1] : cnbSpikes[id],
                    cnbNeurons[id - 1] : cnbNeurons[id],
                ] = self.clusterData["Spike_labels"][id]
                spikeMatTimes[cnbSpikes[id - 1] : cnbSpikes[id], :] = self.clusterData[
                    "Spike_times"
                ][id]
            else:
                spikeMatLabels[0 : cnbSpikes[id], 0 : cnbNeurons[id]] = (
                    self.clusterData["Spike_labels"][id]
                )
                spikeMatTimes[0 : cnbSpikes[id], :] = self.clusterData["Spike_times"][
                    id
                ]

        spikeorder = np.argsort(spikeMatTimes[:, 0])
        self.spikeMatLabels = spikeMatLabels[
            spikeorder, :
        ]  ######################################################### CRASH !!!!!!!!!!!
        self.spikeMatTimes = spikeMatTimes[spikeorder, :]
        ### Perform training (build marginal and local rate functions)
        if onTheFlyCorrection:
            bayesMatrices = self.train(behaviorData, onTheFlyCorrection=True)
        else:
            bayesMatrices = self.train(behaviorData, onTheFlyCorrection=False)
        ### Get preferred position for each cluster
        preferredPos = []
        # TODO: simplify this loop
        for _, rateGroup in enumerate(bayesMatrices["rateFunctions"]):
            for id in range(len(rateGroup) // 2 + 1):
                for idy in range(4):
                    if 4 * id + idy < len(rateGroup):
                        trRateGroup = np.transpose(rateGroup[4 * id + idy][:, :])
                        posX = np.unravel_index(
                            np.argmax(trRateGroup), shape=trRateGroup.shape
                        )
                        preferredPos = preferredPos + [
                            [
                                bayesMatrices["bins"][0][posX[1]],
                                bayesMatrices["bins"][1][posX[0]],
                            ]
                        ]
        preferredPos = np.array(preferredPos)
        _, linearPreferredPos = l_function(preferredPos)
        self.linearPosArgSort = np.argsort(linearPreferredPos)
        self.linearPreferredPos = linearPreferredPos[self.linearPosArgSort]
        self.spikeMatLabels = self.spikeMatLabels[:, self.linearPosArgSort]
        bs = [np.stack(b) for b in bayesMatrices["rateFunctions"]]
        placefields = np.concatenate(bs)
        self.orderedPlaceFields = placefields[self.linearPosArgSort, :]

        return bayesMatrices

    def train(self, behaviorData, onTheFlyCorrection=False):
        Marginal_rate_functions = []
        Rate_functions = []
        Spike_positions = []
        Mutual_info = []
        n_tetrodes = len(self.clusterData["Spike_labels"])
        maxPos = np.max(
            behaviorData["Positions"][
                np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))
            ]
        )
        ### Align the positions time with the spike_times so we can speed filter each spike time (long step)
        pos_times = pykeops.numpy.Vj(behaviorData["positionTime"][:, 0][:, None])
        tetrode_speed_filter_spiketimes = []
        print("Aligning speed-filter with spike times")
        for tetrode in tqdm(range(n_tetrodes)):
            spike_times = pykeops.numpy.LazyTensor(
                self.clusterData["Spike_times"][tetrode][:, 0][:, None], axis=0
            )
            matching_pos_time = (pos_times - spike_times).abs().argmin_reduction(axis=1)
            speed_mask = behaviorData["Times"]["speedFilter"][matching_pos_time]
            tetrode_speed_filter_spiketimes += [speed_mask]

        # Check for bandwidth
        if self.bandwidth is None:
            self.bandwidth = behaviorData["Bandwidth"]
        # Work with position coordinates
        selected_positions = behaviorData["Positions"][
            reduce(
                np.intersect1d,
                (
                    np.where(behaviorData["Times"]["speedFilter"]),
                    inEpochs(
                        behaviorData["positionTime"][:, 0],
                        behaviorData["Times"]["trainEpochs"],
                    ),
                ),
            )
        ]  # Get speed-filtered coordinates from train epoch
        if (
            onTheFlyCorrection
        ):  # setting the position to be between 0 and 1 if necessary
            selected_positions = selected_positions / maxPos
        selected_positions = selected_positions[
            np.logical_not(np.isnan(np.sum(selected_positions, axis=1))), :
        ]  # Remove NaN positions

        ### Build global occupation map
        # xEdges, yEdges, Occupation = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, kernel=self.kernel)
        gridFeature, Occupation = butils.kdenD(
            selected_positions, self.bandwidth, kernel=self.kernel
        )  # 0.07s
        Occupation[Occupation == 0] = np.min(
            Occupation[Occupation != 0]
        )  # We want to avoid having zeros
        mask = Occupation > (
            np.max(Occupation) / self.masking_factor
        )  # Trick to highlight the differences in occupation map
        Occupation_inverse = 1 / Occupation
        Occupation_inverse[Occupation_inverse == np.inf] = 0
        Occupation_inverse = np.multiply(Occupation_inverse, mask)

        ### Build marginal rate functions
        print("Building marginal rate and local rate functions")
        for tetrode in tqdm(range(n_tetrodes)):
            tetrodewisePos = self.clusterData["Spike_positions"][tetrode][
                reduce(
                    np.intersect1d,
                    (
                        np.where(tetrode_speed_filter_spiketimes[tetrode]),
                        inEpochs(
                            self.clusterData["Spike_times"][tetrode][:, 0],
                            behaviorData["Times"]["trainEpochs"],
                        ),
                    ),
                )
            ]
            if (
                onTheFlyCorrection
            ):  # setting the position to be between 0 and 1 if necessary
                tetrodewisePos = tetrodewisePos / maxPos
            tetrodewisePos = tetrodewisePos[
                np.logical_not(np.isnan(np.sum(tetrodewisePos, axis=1))), :
            ]  # Remove NaN: i:e problem with feature recording
            gridFeature, MRF = butils.kdenD(
                tetrodewisePos, self.bandwidth, edges=gridFeature, kernel=self.kernel
            )
            MRF[MRF == 0] = np.min(MRF[MRF != 0])
            MRF = MRF / np.sum(MRF)
            MRF = (
                np.shape(tetrodewisePos)[0]
                * np.multiply(MRF, Occupation_inverse)
                / behaviorData["Times"]["learning"]
            )
            Marginal_rate_functions.append(MRF)
            # Allocate for local rate functions
            Local_rate_functions = []
            Local_Spike_positions = []
            LocalMutualInfo = []

            ### Build local rate functions (one per cluster)
            for label in range(np.shape(self.clusterData["Spike_labels"][tetrode])[1]):
                clusterwisePos = self.clusterData["Spike_positions"][tetrode][
                    reduce(
                        np.intersect1d,
                        (
                            np.where(tetrode_speed_filter_spiketimes[tetrode]),
                            np.where(
                                self.clusterData["Spike_labels"][tetrode][:, label] == 1
                            ),
                            inEpochs(
                                self.clusterData["Spike_times"][tetrode][:, 0],
                                behaviorData["Times"]["trainEpochs"],
                            ),
                        ),
                    )
                ]
                if onTheFlyCorrection:
                    clusterwisePos = clusterwisePos / maxPos
                clusterwisePos = clusterwisePos[
                    np.logical_not(np.isnan(np.sum(clusterwisePos, axis=1))), :
                ]
                if np.shape(clusterwisePos)[0] != 0:
                    gridFeature, LRF = butils.kdenD(
                        clusterwisePos,
                        self.bandwidth,
                        edges=gridFeature,
                        kernel=self.kernel,
                    )
                    LRF[LRF == 0] = np.min(LRF[LRF != 0])
                    LRF = LRF / np.sum(LRF)
                    LRF = (
                        np.shape(clusterwisePos)[0]
                        * np.multiply(LRF, Occupation_inverse)
                        / behaviorData["Times"]["learning"]
                    )
                    Local_rate_functions.append(LRF)
                else:
                    Local_rate_functions.append(np.ones(np.shape(Occupation)))
                Local_Spike_positions.append(clusterwisePos)
                # Let us compute the mutual information with the positions:
                LRF = Local_rate_functions[-1]
                mutualInfo = np.sum(
                    Occupation[LRF > 0]
                    * LRF[LRF > 0]
                    / (np.mean(LRF))
                    * np.log(LRF[LRF > 0] / (np.mean(LRF)))
                    / np.log(2)
                )
                LocalMutualInfo.append(mutualInfo)

            Rate_functions.append(Local_rate_functions)
            Spike_positions.append(Local_Spike_positions)
            Mutual_info.append(LocalMutualInfo)

        bayesMatrices = {
            "Occupation": Occupation,
            "Marginal rate functions": Marginal_rate_functions,
            "Rate functions": Rate_functions,
            "Bins": [np.unique(gridFeature[i]) for i in range(len(gridFeature))],
            "Spike_positions": Spike_positions,
            "Mutual_info": Mutual_info,
        }
        return bayesMatrices

    def test(self, bayesMatrices, behaviorData, windowSize=36):
        windowSize = windowSize / 1000

        print("\nBUILDING POSITION PROBAS")
        guessed_clusters_time = [
            self.clusterData["Spike_times"][tetrode][
                inEpochs(
                    self.clusterData["Spike_times"][tetrode][:, 0],
                    behaviorData["Times"]["testEpochs"],
                )
            ]
            for tetrode in range(len(self.clusterData["Spike_times"]))
        ]
        guessed_clusters = [
            self.clusterData["Spike_labels"][tetrode][
                inEpochs(
                    self.clusterData["Spike_times"][tetrode][:, 0],
                    behaviorData["Times"]["testEpochs"],
                )
            ]
            for tetrode in range(len(self.clusterData["Spike_times"]))
        ]

        Occupation, Marginal_rate_functions, Rate_functions = [
            bayesMatrices[key]
            for key in ["Occupation", "Marginal rate functions", "Rate functions"]
        ]
        mask = Occupation > (np.max(Occupation) / self.masking_factor)

        ### Build Poisson term
        # first we bin the time
        testEpochs = behaviorData["Times"]["testEpochs"]
        Ttest = np.sum(
            [
                testEpochs[2 * i + 1] - testEpochs[2 * i]
                for i in range(len(testEpochs) // 2)
            ]
        )
        n_bins = math.floor(Ttest / windowSize)
        # for each bin we will need to now the test epoch it belongs to, so that we can then
        # set the time correctly to select the corresponding spikes
        timeEachTestEpoch = [
            testEpochs[2 * i + 1] - testEpochs[2 * i]
            for i in range(len(testEpochs) // 2)
        ]
        cumTimeEachTestEpoch = np.cumsum(timeEachTestEpoch)
        cumTimeEachTestEpoch = np.concatenate([[0], cumTimeEachTestEpoch])
        # a function that given the bin indicates the bin index:
        binToEpoch = lambda x: np.where(
            ((x * windowSize - cumTimeEachTestEpoch[0:-1]) >= 0)
            * ((x * windowSize - cumTimeEachTestEpoch[1:]) < 0)
        )[0][0]
        binToEpochArray = [binToEpoch(bins) for bins in range(n_bins)]
        firstBinEpoch = [
            np.min(np.where(np.equal(binToEpochArray, epochId))[0])
            for epochId in range(len(timeEachTestEpoch))
        ]
        All_Poisson_term = [
            np.exp((-windowSize) * Marginal_rate_functions[tetrode])
            for tetrode in range(len(guessed_clusters))
        ]
        All_Poisson_term = reduce(np.multiply, All_Poisson_term)

        ### Log of rate functions
        log_RF = []
        for tetrode in range(np.shape(Rate_functions)[0]):
            temp = []
            for cluster in range(np.shape(Rate_functions[tetrode])[0]):
                temp.append(
                    np.log(
                        Rate_functions[tetrode][cluster]
                        + np.min(
                            Rate_functions[tetrode][cluster][
                                Rate_functions[tetrode][cluster] != 0
                            ]
                        )
                    )
                )
            log_RF.append(temp)

        ### Decoding loop
        position_proba = [np.ones(np.shape(Occupation))] * n_bins
        position_true = [np.ones(2)] * n_bins
        nSpikes = []
        times = []
        for bin in range(n_bins):
            # Trouble: the test Epochs is discretized in continuous bin
            # whereas we forbid the use of some time steps b filtering them according to speed.
            bin_start_time = (
                testEpochs[2 * binToEpoch(bin)]
                + (bin - firstBinEpoch[binToEpoch(bin)]) * windowSize
            )
            bin_stop_time = bin_start_time + windowSize
            times.append(bin_start_time)

            binSpikes = 0
            tetrodes_contributions = []
            tetrodes_contributions.append(All_Poisson_term)

            for tetrode in range(len(guessed_clusters)):
                # Clusters inside our window
                bin_probas = guessed_clusters[tetrode][
                    np.intersect1d(
                        np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
                        np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time),
                    )
                ]
                bin_clusters = np.sum(bin_probas, 0)
                binSpikes = binSpikes + np.sum(bin_clusters)

                # Terms that come from spike information
                if np.sum(bin_clusters) > 0.5:
                    spike_pattern = reduce(
                        np.multiply,
                        [
                            np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
                            for cluster in range(np.shape(bin_clusters)[0])
                        ],
                    )
                else:
                    spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

                tetrodes_contributions.append(spike_pattern)

            nSpikes.append(binSpikes)

            # Guessed probability map
            position_proba[bin] = reduce(np.multiply, tetrodes_contributions)
            position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])
            # True position
            position_true_mean = np.nanmean(
                behaviorData["Positions"][
                    reduce(
                        np.intersect1d,
                        (
                            np.where(
                                behaviorData["positionTime"][:, 0] > bin_start_time
                            ),
                            np.where(
                                behaviorData["positionTime"][:, 0] < bin_stop_time
                            ),
                        ),
                    )
                ],
                axis=0,
            )
            position_true[bin] = (
                position_true[bin - 1]
                if np.isnan(position_true_mean).any()
                else position_true_mean
            )

            if bin % 50 == 0:
                sys.stdout.write(
                    "[%-30s] : %.3f %%"
                    % ("=" * (bin * 30 // n_bins), bin * 100 / n_bins)
                )
                sys.stdout.write("\r")
                sys.stdout.flush()
        sys.stdout.write(
            "[%-30s] : %.3f %%"
            % ("=" * ((bin + 1) * 30 // n_bins), (bin + 1) * 100 / n_bins)
        )
        sys.stdout.write("\r")
        sys.stdout.flush()

        position_true[0] = position_true[1]
        print("\nDecoding finished")

        # Guessed X and Y
        allProba = [
            np.unravel_index(np.argmax(position_proba[bin]), position_proba[bin].shape)
            for bin in range(len(nSpikes))
        ]
        bestProba = [np.max(position_proba[bin]) for bin in range(len(nSpikes))]
        position_guessed = [
            [
                bayesMatrices["Bins"][i][allProba[bin][i]]
                for i in range(len(bayesMatrices["Bins"]))
            ]
            for bin in range(len(nSpikes))
        ]
        inferResults = np.concatenate(
            [np.array(position_guessed), np.array(bestProba).reshape([-1, 1])], axis=-1
        )

        outputResults = {
            "inferring": inferResults,
            "pos": np.array(position_true),
            "probaMaps": position_proba,
            "times": np.array(times),
            "nSpikes": np.array(nSpikes),
        }
        return outputResults

    def full_proba_decoding(
        self, behaviorData, bayesMatrices, timeStepPred, windowSize=36, useTrain=True
    ):
        windowSize = windowSize / 1000

        if useTrain:
            guessed_clusters_time = [
                self.clusterData["Spike_times"][tetrode][
                    inEpochs(
                        self.clusterData["Spike_times"][tetrode][:, 0],
                        behaviorData["Times"]["trainEpochs"],
                    )
                ]
                for tetrode in range(len(self.clusterData["Spike_times"]))
            ]
            guessed_clusters = [
                self.clusterData["Spike_labels"][tetrode][
                    inEpochs(
                        self.clusterData["Spike_times"][tetrode][:, 0],
                        behaviorData["Times"]["trainEpochs"],
                    )
                ]
                for tetrode in range(len(self.clusterData["Spike_times"]))
            ]
        else:
            guessed_clusters_time = [
                self.clusterData["Spike_times"][tetrode][
                    inEpochs(
                        self.clusterData["Spike_times"][tetrode][:, 0],
                        behaviorData["Times"]["testEpochs"],
                    )
                ]
                for tetrode in range(len(self.clusterData["Spike_times"]))
            ]
            guessed_clusters = [
                self.clusterData["Spike_labels"][tetrode][
                    inEpochs(
                        self.clusterData["Spike_times"][tetrode][:, 0],
                        behaviorData["Times"]["testEpochs"],
                    )
                ]
                for tetrode in range(len(self.clusterData["Spike_times"]))
            ]
        # print('\nBUILDING POSITION PROBAS')
        Occupation, Marginal_rate_functions, Rate_functions = [
            bayesMatrices[key]
            for key in ["Occupation", "Marginal rate functions", "Rate functions"]
        ]
        mask = Occupation > (np.max(Occupation) / self.maskingFactor)

        ### Build Poisson term
        All_Poisson_term = [
            np.exp((-windowSize) * Marginal_rate_functions[tetrode])
            for tetrode in range(len(guessed_clusters))
        ]
        All_Poisson_term = reduce(np.multiply, All_Poisson_term)

        ### Log of rate functions
        log_RF = []
        for tetrode in range(np.shape(Rate_functions)[0]):
            temp = []
            for cluster in range(np.shape(Rate_functions[tetrode])[0]):
                temp.append(
                    np.log(
                        Rate_functions[tetrode][cluster]
                        + np.min(
                            Rate_functions[tetrode][cluster][
                                Rate_functions[tetrode][cluster] != 0
                            ]
                        )
                    )
                )
            log_RF.append(temp)

        n_bins = timeStepPred.shape[0]
        ### Decoding loop
        position_probas = []
        nSpikes = []
        for bin in tqdm(timeStepPred):
            bin_start_time = bin
            bin_stop_time = bin_start_time + windowSize
            binSpikes = 0
            tetrodes_contributions = []
            tetrodes_contributions.append(All_Poisson_term)
            for tetrode in range(len(guessed_clusters)):
                bin_probas = guessed_clusters[tetrode][
                    np.intersect1d(
                        np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
                        np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time),
                    )
                ]
                # Note:  we would lose some spikes if we used the clusterData[Spike_pos_index]
                # because some spike might be closest to one position further away than windowSize, yet themselves be close to the spike time
                bin_clusters = np.sum(bin_probas, 0)
                # Terms that come from spike information
                if np.sum(bin_clusters) > 0.5:
                    spike_pattern = reduce(
                        np.multiply,
                        [
                            np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
                            for cluster in range(np.shape(bin_clusters)[0])
                        ],
                    )
                else:
                    spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)
                tetrodes_contributions.append(spike_pattern)
            # Guessed probability map
            position_proba = reduce(np.multiply, tetrodes_contributions)
            position_proba = np.multiply(position_proba, Occupation)
            position_proba = position_proba / np.sum(position_proba)
            position_probas.append(position_proba)
        return position_probas.reshape(-1)
