#!/usr/bin/env python3

import re
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import matplotlib as matplotlib
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, NoNorm, Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from neuroencoders.importData import epochs_management as ep
from neuroencoders.utils.viz_params import (
    ALL_STIMS_COLOR,
    ALPHA_DELTA_LINE,
    ALPHA_TRAIL_LINE,
    ALPHA_TRAIL_POINTS,
    BINARY_COLORS,
    COLORMAP,
    CURRENT_POINT_COLOR,
    CURRENT_PREDICTED_POINT_COLOR,
    DELTA_COLOR,
    DELTA_COLOR_FORWARD,
    DELTA_COLOR_REVERSE,
    FREEZING_LINE_COLOR,
    HLINES,
    MAX_NUM_STARS,
    PREDICTED_CMAP,
    PREDICTED_COLOR,
    PREDICTED_LINE_COLOR,
    REMOVE_TICKS,
    RIPPLES_COLOR,
    SAFE_COLOR,
    SAFE_COLOR_PREDICTED,
    SHOCK_COLOR,
    SHOCK_COLOR_PREDICTED,
    TRUE_COLOR,
    TRUE_LINE_COLOR,
    VLINES,
    WITH_REF_BG,
)


def time_formatter(x, pos):
    td = timedelta(seconds=x)
    h, rem = divmod(int(td.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    ms = int((x % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


class AnimatedPositionPlotter:
    """
    Animated plotter for DataHelper class showing position trajectory with (direction|distance to the wall|whatever) color coding.
    """

    def __init__(
        self,
        data_helper,
        trail_length: int = 30,
        lin_movie_duration: int = 500,
        figsize: Tuple[float, float] = (16, 9),
        fourD_analysis_mode: bool = False,
        linear_position_mode: bool = False,
        with_posMat: bool = True,
        **kwargs,
    ):
        """
        Initialize the animated position plotter.

        Args:
            data_helper: DataHelper instance with .positions and .dim attributes
            trail_length: Number of recent points to show in the trail (default: 40)
            dim: Dimension to use for color coding (default: None, auto-detected as pos, direction, or distance)
            figsize: Figure size as (width, height)
            fourD_analysis_mode: Whether to use 4D analysis mode (3-panel, HD and speed) or just trajectory in maze (default: False)
            linear_position_mode: Whether to use and plot linearized positions around the maze (default: False)
            with_posMat: Whether to use position matrix for the Stimulation times (otherwise will use StimEpoch, but seems less trustworthy) (default: True)
            **kwargs: Additional keyword arguments for customization
                speedMask : Boolean, use speed mask for the trajectory (default: False)
                speedMaskArray: Pre-computed speed mask array instead of simple boolean (optional)
                predicted : Pre-computed predicted positions (optional)
                /!\ when using predicted positions, you must provide a posIndex and a prediction time to match the positions with the predictions.
        """
        self.data_helper = data_helper
        start_stim = self.data_helper.fullBehavior["Times"].get("start_stim", None)
        self.usePosMat = with_posMat
        if start_stim is not None:
            self.plot_stims = kwargs.get("plot_stims", True)
        self.trail_length = trail_length
        self.lin_movie_duration = lin_movie_duration
        self.figsize = figsize
        self.dim_name = self.data_helper.target.capitalize()  # Default dimension name
        self.fourD_analysis_mode = fourD_analysis_mode
        self.l_function = kwargs.get("l_function", None)
        self.linear_position_mode = linear_position_mode
        self.be_fast = kwargs.get("be_fast", False)
        self.axes_names = list()  # for now only used in multipanel plot for trajectories panels, but could be extended to other panels if needed

        self.extract_data(**kwargs)

        # Validate data
        self._validate_data()

        # Setup figure and animation components
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.animation = None

        # Animation parameters
        self.current_frame = 0
        self.total_frames = len(self.positions)
        self.output_dir = kwargs.get("output_dir", Path.cwd() / "output_plots")

        if kwargs.get("setup_plot", False):
            # If setup_plot is True, we immediately setup the plot
            self.setup_plot(**kwargs)

        if kwargs.get("init_animation", False):
            # If init_animation is True, we immediately initialize the animation
            self.init_animation()

    def extract_data(self, **kwargs):
        """
        Extract and process data from data_helper.
        """
        # Get positions
        if kwargs.get("positions_from_NN", None) is not None:
            self.positions_from_NN = np.array(kwargs["positions_from_NN"])
            # self.og_positions_from_NN = self.positions_from_NN.copy()
            if kwargs.get("prediction_time", None) is not None:
                self.prediction_positionTime = np.array(kwargs["prediction_time"])
            else:
                self.prediction_positionTime = None
        else:
            self.positions_from_NN = None
            self.prediction_positionTime = None

        if kwargs.get("predicted_heatmap", None) is not None:
            self.predicted_heatmap = np.array(kwargs["predicted_heatmap"])
        else:
            self.predicted_heatmap = None

        if kwargs.get("predicted_dim_please", None) is not None:
            self.predicted_dim_please = kwargs.pop("predicted_dim_please")

        self.positions = self.positions_from_NN
        self.positionTime = self.prediction_positionTime
        self.plot_all_stims = kwargs.get("plot_all_stims", False)
        # we setup a "true" positionTime to use for precise plotting such as stims...
        # same for "true" positions, which are the original positions for each and every timepoint

        # Get predicted positions if available
        if kwargs.get("predicted", None) is not None:
            self.predicted = np.array(kwargs["predicted"])
            self.posIndex = kwargs.get("posIndex", None)
            if self.posIndex is None:
                raise ValueError(
                    "You must provide a posIndex when using predicted positions."
                )
        else:
            self.predicted = None
            self.posIndex = None

        # Get linearized positions/predictions if available
        if kwargs.get("linearized_true", None) is not None:
            self.linpositions = np.array(kwargs["linearized_true"])
        elif self.linear_position_mode:
            if not self.l_function:
                raise ValueError(
                    "Linear position mode requires either a linearization function (l_function) or linearized_true data."
                )
            _, self.linpositions = self.l_function(self.positions[:, :2])
        else:
            self.linpositions = None

        if kwargs.get("linearized_pred", None) is not None:
            self.linpredicted = np.array(kwargs["linearized_pred"])
        elif self.linear_position_mode and self.predicted is not None:
            if not self.l_function:
                raise ValueError(
                    "Linear position mode requires either a linearization function (l_function) or linearized_pred data."
                )
            _, self.linpredicted = self.l_function(self.predicted[:, :2])
        else:
            self.linpredicted = None

        if self.predicted is not None:
            # sort predicted positions by posIndex
            self.sort_idx = np.argsort(self.prediction_positionTime)
            self.positions_from_NN = self.positions_from_NN[self.sort_idx]
            self.prediction_positionTime = self.prediction_positionTime[self.sort_idx]
            self.predicted_heatmap = (
                self.predicted_heatmap[self.sort_idx]
                if self.predicted_heatmap is not None
                else None
            )
            self.positions = self.positions[self.sort_idx]
            self.positionTime = self.positionTime[self.sort_idx]
            self.predicted = (
                self.predicted[self.sort_idx] if self.predicted is not None else None
            )
            self.posIndex = self.posIndex[self.sort_idx]
            self.linpositions = (
                self.linpositions[self.sort_idx]
                if self.linpositions is not None
                else None
            )
            self.linpredicted = (
                self.linpredicted[self.sort_idx]
                if self.linpredicted is not None
                else None
            )
            self.predicted_dim_please = (
                self.predicted_dim_please[self.sort_idx]
                if self.predicted_dim_please is not None
                else None
            )
        else:
            self.sort_idx = np.arange(len(self.positions))

        if self.plot_stims:
            self.start_stim = self.data_helper.fullBehavior["Times"]["start_stim"]
            self.stop_stim = self.data_helper.fullBehavior["Times"]["stop_stim"]
            self.start_freeze = self.data_helper.fullBehavior["Times"].get(
                "start_freeze", None
            )
            self.stop_freeze = self.data_helper.fullBehavior["Times"].get(
                "stop_freeze", None
            )
            if self.usePosMat:
                self.PosMat = self.data_helper.fullBehavior["Times"]["PosMat"]
                if self.PosMat is not None:
                    self.PosMatStimMask = self.PosMat[:, 3] == 1

            self.tRipples = self.data_helper.fullBehavior["Times"].get("tRipples", None)

        if self.plot_all_stims:
            if self.usePosMat:
                assert self.data_helper.fullBehavior["Positions"] is not None
                # assert they have the same length
                assert (
                    self.data_helper.fullBehavior["Positions"].shape[0]
                    == self.PosMatStimMask.shape[0]
                )
                self.stims_positions = self.data_helper.fullBehavior["Positions"][
                    self.PosMatStimMask, :2
                ]
            else:
                raise NotImplementedError(
                    "Due to inconsistencies in StimEpoch, you need PosMat to plot all the stims at the perfectly right time/positions."
                )
        # Apply masks and clean data
        self._apply_masks(**kwargs)

        # Extract dimension data for color coding (can be called now that we have valid positions indexing)
        self._extract_dim_data(**kwargs)

        # Calculate derived quantities
        self._calculate_derived_data(**kwargs)

    def _apply_masks(self, **kwargs):
        """Apply epoch and speed masks to filter data."""

        if kwargs.get("predLossMask", None) is None:
            if self.predicted is None:
                print("skipping prediction and predloss")
                self.predLossMask = None
            else:
                self.predLossMask = np.ones(len(self.predicted), dtype=bool)
        else:
            self.predLossMask = np.array(kwargs["predLossMask"])
        self.predLossMask = (
            self.predLossMask[self.sort_idx]
            if self.predicted is not None
            else self.predLossMask
        )

        if self.predicted is None:
            epochMask = ep.inEpochsMask(
                self.positionTime,
                self.data_helper.fullBehavior["Times"]["trainEpochs"],
            ) + ep.inEpochsMask(
                self.positionTime,
                self.data_helper.fullBehavior["Times"]["testEpochs"],
            )
            # restrict to best epochs no prediction given
        else:
            epochMask = (
                self.positionTime >= np.nanmin(self.prediction_positionTime)
            ) & (self.positionTime <= np.nanmax(self.prediction_positionTime))
            # restrict to rough prediction time range

        try:
            self.positions[epochMask.flatten()]
            # should work if there is no prediction - totMask is very long
            self.totMask = epochMask.flatten()
        except (IndexError, ValueError):
            warn(
                "Epoch mask does not match positions length. Check your position and prediction time arrays."
            )
            self.totMask = np.ones(len(self.positions), dtype=bool)
            # dummy mask of length positionTime, prediction_positionTime SHOULD NOT HAPPEN
            raise ValueError(
                "this should not happen, epochMask does not match positions length. Check your position and prediction time arrays."
            )

        if kwargs.get("speedMaskArray", None) is not None:
            print("Using provided speed mask array.")
            self.speed_mask = np.array(kwargs["speedMaskArray"])
        elif kwargs.get("speedMask", False):
            print("Using speed mask from data_helper.")
            try:
                self.speed_mask = np.array(
                    self.data_helper.fullBehavior["Times"]["speedFilter"]
                )
                # size of positionTime, not prediction_positionTime
            except AttributeError:
                warn("No speed mask found in self.data_helper. Using all positions.")
                self.speed_mask = np.ones(len(self.totMask), dtype=bool)
                # size of prediction_positionTime
        else:
            self.speed_mask = None
            # size of prediction_positionTime
        self.speed_mask = (
            self.speed_mask[self.sort_idx]
            if self.speed_mask is not None
            else self.speed_mask
        )

        print(f"Speed mask applied: {self.speed_mask is not None}.")
        if self.speed_mask is not None:
            print(
                f"{self.speed_mask.sum() / len(self.speed_mask) * 100:.2f}% of positions kept."
            )

        # apply mask to positionTime now - others follow at the end
        self.positionTime = self.positionTime[self.totMask]
        # now we have masked indexing

        if self.plot_stims:
            if not self.usePosMat:
                # should not be available
                self.stimEpochsIndex = np.array(
                    [
                        [
                            ep.find_closest_index(
                                self.positionTime, start, tolerance=True
                            ),
                            ep.find_closest_index(
                                self.positionTime, stop, tolerance=True
                            ),
                        ]
                        for start, stop in zip(self.start_stim, self.stop_stim)
                    ]
                )
                index_set = set()
                for start, stop in self.stimEpochsIndex:
                    index_set.update(range(start, stop + 1))
                self.stim_indices = np.array(sorted(index_set))
                self.stim_indices = self.stim_indices[self.stim_indices != -1]
            else:
                stim_indices = np.where(self.PosMatStimMask)[0]
                self.stim_indices = np.where(np.isin(self.posIndex, stim_indices))[0]
                print(self.stim_indices)

            if self.tRipples is not None:
                # same as stim indices, give a bit of room for the time matching
                self.tRipples = np.array(
                    [
                        t
                        for t in self.tRipples
                        if t <= self.positionTime[-1] + 0.2  # in seconds
                        and t >= self.positionTime[0] - 0.2
                    ]
                )
                self.ripples_indices = np.array(
                    sorted(
                        [
                            ep.find_closest_index(self.positionTime, t, tolerance=True)
                            for t in self.tRipples
                        ]
                    )
                )
                self.ripples_indices = self.ripples_indices[self.ripples_indices != -1]

            # same reasoning for freezing epochs
            self.FreezeEpochs = np.array(
                [
                    [start, stop]
                    for start, stop in zip(self.start_freeze, self.stop_freeze)
                    if start <= self.positionTime[-1] + 0.2
                    and start >= self.positionTime[0] - 0.2
                    and stop <= self.positionTime[-1] + 0.2
                    and stop >= self.positionTime[0] - 0.2
                ]
            ).reshape(-1, 2)
            self.start_freeze = self.FreezeEpochs[:, 0]
            self.stop_freeze = self.FreezeEpochs[:, 1]

            self.freezingEpochsIndex = np.array(
                [
                    [
                        ep.find_closest_index(self.positionTime, start, tolerance=True),
                        ep.find_closest_index(self.positionTime, stop, tolerance=True),
                    ]
                    for start, stop in zip(self.start_freeze, self.stop_freeze)
                ]
            )
            index_set = set()
            for start, stop in self.freezingEpochsIndex:
                index_set.update(range(start, stop + 1))
            self.freezing_indices = np.array(sorted(index_set))
            self.freezing_indices = self.freezing_indices[self.freezing_indices > 0]

        self.positions = self.positions[self.totMask]

        if self.predLossMask is not None:
            # same reasoning
            try:
                self.predLossMask = self.predLossMask[self.totMask]
            except IndexError:
                self.predLossMask = self.predLossMask

        if self.linpositions is not None:
            self.linpositions = self.linpositions[self.totMask]
        if self.linpredicted is not None:
            self.linpredicted = self.linpredicted

        if self.predicted_heatmap is not None:
            self.predicted_heatmap = self.predicted_heatmap

        # Remove NaN values
        # WARNING: I'm afraid this will remove too much data, and would prefer plotting NaN values instead of removing them.
        # self.valid_indices = ~np.isnan(self.positions).any(axis=1)
        # set valid_indices to True for all positions
        self.true_valid_indices = np.ones(len(self.positions), dtype=bool)
        self.positions = self.positions[self.true_valid_indices]

        if self.predicted is not None:
            self.prediction_valid_indices = np.ones(len(self.predicted), dtype=bool)
            self.predicted = self.predicted[self.prediction_valid_indices]
            self.posIndex = self.posIndex[self.prediction_valid_indices]

            self.predLossMask = self.predLossMask[self.prediction_valid_indices]

            # give NaN values to positions where prediction must not be shown
            self.predicted[~self.predLossMask] = (
                np.nan
            )  # Set positions with predLossMask to NaN
            if self.speed_mask is not None:
                # Set positions with speed_mask to NaN - this way we still get to see the whole true trajectory, but with no predictions plotted where the speed is NaN
                self.predicted[~self.speed_mask] = np.nan

        if self.linpositions is not None:
            self.linpositions = self.linpositions[self.true_valid_indices]

        if self.linpredicted is not None:
            self.linpredicted = self.linpredicted[self.prediction_valid_indices]
            self.linpredicted[~self.predLossMask] = (
                np.nan
            )  # Set positions with predLossMask to NaN
            if self.speed_mask is not None:
                # give NaN values to positions where speed_mask is False
                self.linpredicted[~self.speed_mask] = np.nan

        if self.predicted_heatmap is not None:
            self.predicted_heatmap = self.predicted_heatmap[
                self.prediction_valid_indices
            ]
            self.predicted_heatmap[~self.predLossMask] = np.nan
            if self.speed_mask is not None:
                self.predicted_heatmap[~self.speed_mask] = np.nan

    def _extract_dim_data(self, **kwargs):
        """Extract dimension data for color coding (from original class)."""

        # First, look for dim in kwargs
        dim = kwargs.get("dim", None)
        self.dim_name = getattr(self.data_helper, "target", "position").capitalize()

        if dim is None:
            if self.data_helper.target == "pos" and self.positions.shape[1] == 2:
                dim = np.ones_like(self.true_valid_indices)
                self.dim_name = "dummy"
            elif (
                self.data_helper.target == "lin" or self.data_helper.target == "linear"
            ):
                dim = self.data_helper.linearized
                self.dim_name = "Dist2Threat"
            elif self.data_helper.target.lower() == "linandthigmo":
                dim = "thigmo"
                self.dim_name = "Dist2Wall"
            elif self.data_helper.target.lower() == "posandheaddirectionandspeed":
                dim = "PosHDSpeed"
                self.dim_name = "PosHDSpeed"
            elif (
                self.data_helper.target.lower() == "posandheaddirectionandthigmo"
                or self.positions.shape[1] > 2
            ):
                dim = "Head Direction"
                self.dim_name = "Head Direction"
            elif self.data_helper.target.lower() == "direction":
                dim = "direction"
                self.dim_name = "Direction"
            else:
                dim = np.ones_like(self.true_valid_indices)

        # check if dim is a string or a np array
        if isinstance(dim, str):
            if dim == "direction":
                self.dim = np.array(self.data_helper.direction)
                self.dim_name = "direction"
                # get rid of NaN values in directions
                self.dim = self.dim[self.posIndex]
            elif dim == "distance" or dim == "thigmo":
                if not hasattr(self.data_helper, "thigmo"):
                    self.data_helper.thigmo = np.array(
                        self.data_helper.dist2wall(self.positions)
                    )
                self.dim = np.array(self.data_helper.thigmo)
                self.dim_name = "dist2wall"
                self.dim = self.dim[self.posIndex]
            elif dim == "PosHDSpeed":
                self.dim = np.array(self.data_helper.positions)
                self.dim_name = "PosHDSpeed"
                self.dim = self.dim[self.posIndex]
            elif dim == "Head Direction":
                self.dim = np.array(self.data_helper.positions[:, 3])
                self.dim_name = "Head Direction"
                # self.lin_dim = self.data_helper.positions[:, 2]
                self.dim = self.dim[self.posIndex]
                # self.lin_dim = self.lin_dim[self.totMask][self.true_valid_indices]
                self.positions = self.positions[:, :2]
        elif isinstance(dim, np.ndarray):
            self.dim = dim
            try:
                self.dim = self.dim[self.true_valid_indices]
            except IndexError:
                self.dim = self.dim[self.totMask][self.true_valid_indices]

        if self.predicted is not None:
            # Now look for predicted_dim in kwargs
            predicted_dim = kwargs.get("predicted_dim", None)
            if predicted_dim == "no_dim":
                self.predicted_dim = None
                self.predicted_dim_name = "No Dimension"
            else:
                if isinstance(predicted_dim, str):
                    self.predicted_dim_name = predicted_dim.capitalize()
                else:
                    self.predicted_dim_name = self.dim_name  # Default to dim_name

                if predicted_dim is None:
                    if self.predicted_dim_name.lower() == "dummy":
                        predicted_dim = np.ones_like(self.predicted)
                    elif self.predicted_dim_name == "Dist2Threat":
                        predicted_dim = self.data_helper.linearized
                    elif self.predicted_dim_name == "Dist2Wall":
                        predicted_dim = "thigmo"
                    elif self.predicted_dim_name == "Direction":
                        predicted_dim = "direction"
                    elif self.predicted_dim_name == "PosHDSpeed":
                        predicted_dim = "PosHDSpeed"
                    elif self.predicted_dim_name == "Head Direction":
                        predicted_dim = "Head Direction"
                    else:
                        predicted_dim = np.ones_like(self.predicted)

                # check if predicted_dim is a string or a np array
                if isinstance(predicted_dim, str):
                    if predicted_dim == "direction":
                        if self.linpredicted is None:
                            raise ValueError(
                                "Linearized predictions are required for direction."
                            )
                        self.predicted_dim = np.array(
                            self.data_helper._get_traveling_direction(self.linpredicted)
                        )
                        self.predicted_dim_name = "direction"
                    elif predicted_dim == "distance" or predicted_dim == "thigmo":
                        if self.predicted.shape[1] != 2:
                            raise ValueError(
                                "Predicted positions must have at least 2 dimensions for distance to wall."
                            )
                        self.pred_thigmo = np.array(
                            self.data_helper.dist2wall(self.predicted)
                        )
                        self.predicted_dim_name = "dist2wall"
                    elif predicted_dim == "PosHDSpeed":
                        self.predicted_dim = self.predicted.copy()
                        self.predicted_dim_name = "PosHDSpeed"
                    elif predicted_dim == "Head Direction":
                        if self.predicted.shape[1] < 4:
                            raise ValueError(
                                "Predicted positions must have at least 4 dimensions for head direction."
                            )
                        self.predicted_dim = self.predicted[:, 3]
                        # self.lin_dim_pred = self.predicted[:, 2]
                        self.predicted_dim_name = "Head Direction"
                        self.predicted = self.predicted[:, :2]
                        self.predicted_dim = self.predicted_dim[
                            self.prediction_valid_indices
                        ]
                elif isinstance(predicted_dim, np.ndarray):
                    self.predicted_dim = predicted_dim

    def _calculate_derived_data(self, **kwargs):
        if self.l_function is not None:
            if self.linpositions is None:
                _, lpositions = self.l_function(self.positions[:, :2])
                lpositions = lpositions.reshape(-1)
                self.linpositions = lpositions
            if self.predicted is not None:
                if self.linpredicted is None:
                    _, lpredicted = self.l_function(self.predicted[:, :2])
                    lpredicted = lpredicted.reshape(-1)
                    self.linpredicted = lpredicted

        try:
            # instead of simple {0,1} direction, we compute the full velocity vector
            velocity = np.stack(
                (
                    np.diff(self.positions[:, 0]),
                    np.diff(self.positions[:, 1]),
                ),
                axis=1,
            )
            norm = np.linalg.norm(velocity, axis=1)
            self.velocity_true = velocity / (
                norm[:, None] + 1e-10
            )  # avoid division by zero
            # for the first timepoint, simply set the velocity to zero
            zeros_for_first = np.zeros((1, 2))
            self.velocity_true = np.vstack((zeros_for_first, self.velocity_true))
        except Exception as e:
            warn(f"Error setting velocity True: {e}")
            self.velocity_true = None

        if self.predicted is not None:
            try:
                velocity = np.stack(
                    (
                        np.diff(self.predicted[:, 0]),
                        np.diff(self.predicted[:, 1]),
                    ),
                    axis=1,
                )
                norm = np.linalg.norm(velocity, axis=1)
                self.velocity_predicted = velocity / (
                    norm[:, None] + 1e-10
                )  # avoid division by zero
                # for the first timepoint, simply set the velocity to zero
                zeros_for_first = np.zeros((1, 2))
                self.velocity_predicted = np.vstack(
                    (zeros_for_first, self.velocity_predicted)
                )
            except Exception as e:
                warn(f"Error setting velocity predicted: {e}")
                self.velocity_predicted = None

        self.head_direction = self.data_helper._get_head_direction(
            self.positions, return_as_deg=True
        )
        self.speeds = self.data_helper._get_speed(self.positions, interval=1 / (15))
        if self.predicted is not None:
            if self.predicted.shape[1] == 4:
                self.predicted_head_direction = np.degrees(
                    self.predicted[:, 2]
                )  # this prediction is in radians, so we convert it to degrees
                self.predicted_speeds = self.predicted[:, 3]
            else:
                warn(
                    "Predicted positions do not contain head direction or speed. Computing them from positions."
                )
                self.predicted_head_direction = self.data_helper._get_head_direction(
                    self.predicted, return_as_deg=True
                )
                self.predicted_speeds = self.data_helper._get_speed(
                    self.predicted, interval=1 / (15)
                )

    def _validate_data(self):
        """Validate that the data helper has the required attributes with correct shapes."""
        if not hasattr(self.data_helper, "positions"):
            raise ValueError("DataHelper must have 'positions' attribute")

        positions = np.array(self.positions)
        directions = np.array(self.dim)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "Positions must be a 2D array with shape (n_timepoints, 2)"
            )
        if directions.ndim != 1:
            raise ValueError("Directions must be a 1D array")

    def setup_plot(self, **kwargs):
        """
        Setup the animation plot either in 4D analysis mode (3-panel, HD and speed) or just trajectory in maze.
        """
        if self.fourD_analysis_mode:
            return self._setup_fourD_plot(**kwargs)
        else:
            if kwargs.get("very_simple_plot", False):
                print("Setting up very simple plot (trajectory only)...")
                self.very_simple_plot = True
                self.simple_plot = False
                return self._setup_simple_plot(**kwargs)
            elif kwargs.get("simple_plot", True):
                print(
                    "Setting up multipanel plot (trajectory + linear position movie)..."
                )
                self.very_simple_plot = False
                self.simple_plot = True
                return self._setup_dual_plot(**kwargs)
            else:
                print(
                    "Setting up multipanel plot (trajectory, forward/reverse, and linear position movie)..."
                )
                self.very_simple_plot = False
                self.simple_plot = False
                return self._setup_multipanel_plot(**kwargs)

    def _setup_fourD_plot(self, **kwargs):
        """
        Setup the matplotlib figure and axes for 4D analysis mode (3-panel, HD and speed).
        """
        usedark = kwargs.pop("dark_theme", True)
        plt.style.use("dark_background") if usedark else plt.style.use("default")

        self.fig = plt.figure(figsize=self.figsize)
        self.fig.patch.set_facecolor("#1a1a2e" if usedark else "white")

        # Create grid layout
        gs = self.fig.add_gridspec(
            2, 3, width_ratios=[3, 0.5, 0.5], height_ratios=[1.5, 0.5]
        )

        # Setup trajectory plot (left side, spans both rows)
        self._setup_trajectory_panel(gs, dark_theme=usedark, **kwargs)

        # Setup polar heading plot (top right)
        self._setup_polar_panel(gs, dark_theme=usedark, **kwargs)

        # Setup speed histogram (bottom right)
        self._setup_speed_panel(gs, dark_theme=usedark, **kwargs)

        return self.fig, self.axes

    def _setup_dual_plot(self, **kwargs):
        """
        Setup the matplotlib figure and axes for multipanel viewing mode (1 panel for Shock/Safe; 1 for forward/reverse prediction, 1 for a movie of linear position).
        """
        usedark = kwargs.get("dark_theme", False)
        plt.style.use("dark_background") if usedark else plt.style.use("default")

        if self.linpositions is None:
            raise ValueError(
                "Linear positions are required for multipanel plot. Either use linear_position_mode=True or provide linearized_true data."
            )

        self.fig = plt.figure(figsize=self.figsize)
        self.fig.patch.set_facecolor("#1a1a2e" if usedark else "white")

        # Create grid layout
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 0.5])

        # Setup trajectory plot (top side, spans both columns)
        self._setup_trajectory_panel(
            gs,
            dark_theme=usedark,
            loc_in_gs="top",
            colors_style="speed",
            **kwargs,
        )

        if self.predicted is not None:
            # Setup trajectory plot (top side, spans both columns)
            handles1, labels1 = (
                self.handles_labels["top"]["handles"],
                self.handles_labels["top"]["labels"],
            )
            # remove the axis legends
            self.axes["top"].legend().remove()

            all_handles = handles1
            all_labels = labels1

            label_to_handle = {}
            for label, handle in zip(all_labels, all_handles):
                if label not in label_to_handle:
                    label_to_handle[label] = handle  # keep first occurrence
            labels = list(label_to_handle.keys())
            handles = list(label_to_handle.values())

            self.fig.legend(
                handles,
                labels,
                loc="center left",
                ncol=1,
                frameon=True,
                fontsize=10,
                framealpha=0.5,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            )

        # Setup linearized position movie (bottom)
        self._setup_linpos_movie(gs, dark_theme=usedark, colors_style="speed", **kwargs)

        plt.tight_layout()

        return self.fig, self.axes

    def _setup_multipanel_plot(self, **kwargs):
        """
        Setup the matplotlib figure and axes for multipanel viewing mode (1 panel for Shock/Safe; 1 for forward/reverse prediction, 1 for a movie of linear position).
        """
        usedark = kwargs.get("dark_theme", False)
        plt.style.use("dark_background") if usedark else plt.style.use("default")

        if self.linpositions is None:
            raise ValueError(
                "Linear positions are required for multipanel plot. Either use linear_position_mode=True or provide linearized_true data."
            )

        self.fig = plt.figure(figsize=self.figsize)
        self.fig.patch.set_facecolor("#1a1a2e" if usedark else "white")

        # Create grid layout
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 0.5])

        # Setup trajectory plot (top side, spans both columns if predicted is None, else spans only one)
        self._setup_trajectory_panel(
            gs,
            dark_theme=usedark,
            loc_in_gs="top left" if self.predicted is not None else "top",
            colors_style="Shock/Safe",
            **kwargs,
        )

        if self.predicted is not None:
            # Setup trajectory plot (top side, spans both columns if predicted is None, else spans only one)
            self._setup_trajectory_panel(
                gs,
                dark_theme=usedark,
                loc_in_gs="top right",
                colors_style="Concordant/Discordant",
                **kwargs,
            )

            # if that's the case, also move the legend to the middle of the two plots.
            handles1, labels1 = (
                self.handles_labels["top left"]["handles"],
                self.handles_labels["top left"]["labels"],
            )
            handles2, labels2 = (
                self.handles_labels["top right"]["handles"],
                self.handles_labels["top right"]["labels"],
            )

            # remove the axis legends
            self.axes["top left"].legend().remove()
            self.axes["top right"].legend().remove()

            all_handles = handles1 + handles2
            all_labels = labels1 + labels2

            label_to_handle = {}
            for label, handle in zip(all_labels, all_handles):
                if label not in label_to_handle:
                    label_to_handle[label] = handle  # keep first occurrence
            labels = list(label_to_handle.keys())
            handles = list(label_to_handle.values())

            self.fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=1,
                frameon=True,
                bbox_to_anchor=(0.5, 0.73),
                fontsize=10,
                framealpha=0.5,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            )

        # Setup linearized position movie (bottom)
        self._setup_linpos_movie(gs, dark_theme=usedark, **kwargs)

        plt.tight_layout()

        return self.fig, self.axes

    def _setup_trajectory_panel(self, gs, **kwargs):
        """Setup main trajectory panel using the same routine as simple setup plot."""

        dim = None
        predicted_dim = None
        if kwargs.get("colors_style", None) == "Shock/Safe":
            try:
                dim = self.data_helper.direction
                dim = dim = dim[self.totMask][self.true_valid_indices]
            except:
                dim = self.data_helper._get_traveling_direction(self.linpositions)
            if self.predicted is not None:
                predicted_dim = self.data_helper._get_traveling_direction(
                    self.linpredicted
                )
            kwargs["pair_points"] = False
            kwargs["alpha_trail_line"] = 0.9
            kwargs["alpha_delta_line"] = 0.9
            kwargs["alpha_trail_points"] = 1
            kwargs["delta_color"] = "xkcd:vivid purple"

        elif kwargs.get("colors_style", None) == "speed":
            self.dim_name = "speed_mask"
            self.predicted_dim_name = "speed_mask"
            print("Using speed for color coding!")

            try:
                dim = self.predicted_dim_please
                dim = dim[self.totMask][self.true_valid_indices]
            except:
                print("Using speed mask from random.")
                dim = self.speed_mask

            if self.predicted is not None:
                predicted_dim = np.ones_like(self.linpredicted)
            kwargs["pair_points"] = True
            kwargs["alpha_trail_line"] = 0.9
            kwargs["alpha_delta_line"] = 0.9
            kwargs["alpha_trail_points"] = 1
            kwargs["delta_color"] = "xkcd:vivid purple"
            kwargs["binary_colors"] = True

        elif kwargs.get("colors_style", None) == "Concordant/Discordant":
            kwargs["pair_points"] = True  # force pairing for this panel

        if kwargs.get("loc_in_gs", None) == "top left":
            self.axes[kwargs["loc_in_gs"]] = self.fig.add_subplot(gs[0, 0])
            custom_ax = self.axes[kwargs["loc_in_gs"]]
        elif kwargs.get("loc_in_gs", None) == "top right":
            self.axes[kwargs["loc_in_gs"]] = self.fig.add_subplot(gs[0, 1])
            custom_ax = self.axes[kwargs["loc_in_gs"]]
        elif kwargs.get("loc_in_gs", None) == "top":
            self.axes[kwargs["loc_in_gs"]] = self.fig.add_subplot(gs[0, :])
            self.ax = self.axes[
                kwargs["loc_in_gs"]
            ]  # For compatibility with simple setup
            custom_ax = self.axes[kwargs["loc_in_gs"]]
        else:
            kwargs["loc_in_gs"] = "left"
            self.axes[kwargs["loc_in_gs"]] = self.fig.add_subplot(gs[:, 0])
            self.ax = self.axes[kwargs["loc_in_gs"]]
            custom_ax = self.axes[kwargs["loc_in_gs"]]

        # Use the shared trajectory panel logic
        self._setup_trajectory_panel_logic(
            ax=custom_ax, dim=dim, predicted_dim=predicted_dim, **kwargs
        )

    def _setup_linpos_movie(self, gs, **kwargs):
        """
        Classic movie plotting of linearized positions using a sliding window of winSize.
        If available, will plot Freeze and Stimulation times as arrows just on top.

        Possibility to provide additional arguments such as the cmap for the points, size of the points, and alpha.
        """
        true_color = kwargs.get("true_color", TRUE_COLOR)
        true_line_color = kwargs.get("true_line_color", TRUE_LINE_COLOR)
        predicted_color = kwargs.get("predicted_color", PREDICTED_COLOR)
        predicted_line_color = kwargs.get("predicted_line_color", PREDICTED_LINE_COLOR)
        colors_style = kwargs.get("colors_style", None)

        self.axes["linpos_movie"] = self.fig.add_subplot(gs[1, :])
        ax = self.axes["linpos_movie"]

        if kwargs.get("dark_theme", False):
            ax.set_facecolor("#0f0f23")
        else:
            ax.set_facecolor("grey")
        ax.tick_params(colors="white" if kwargs.get("dark_theme", False) else "black")
        self.artists["linpos_line"] = ax.plot(
            [],
            [],
            color=true_line_color,
            linewidth=2,
        )[0]

        self.artists["current_point"] = ax.scatter(
            [],
            [],
            c=CURRENT_POINT_COLOR,
            s=200,
            marker="o",
            edgecolor="black",
            linewidth=2.5,
            zorder=10,
        )

        if self.predicted is not None:
            # we'll do one scatter for the predicted positions
            # TODO: possibility to add a cmap with predLoss values
            # and one line artist to link the dots between the predicted positions
            self.scatterpoints = []
            if hasattr(self, "lin_dim_pred") and self.lin_dim_pred is not None:
                # create a cmap for the predict lin dim (either predLoss, head direction, thigmo, speed, etc)
                if self.predicted_dim_name == "Head Direction":
                    # means lin_pred_dim is actually head direction !
                    self.predicted_lin_cmap = plt.get_cmap("hsv")
                    self.predicted_lin_norm = Normalize(
                        vmin=np.nanmin(self.lin_dim_pred),
                        vmax=np.nanmax(self.lin_dim_pred),
                    )
                elif self.predicted_dim_name == "speed_mask":
                    # means we dont really plot the predicted dim, but rather show whether the speed was below or above threshold
                    self.predicted_lin_cmap = ListedColormap(
                        [DELTA_COLOR_FORWARD, DELTA_COLOR_REVERSE]
                    )
            self.artists["linpos_pred_points"] = ax.scatter(
                [],
                [],
                cmap=self.predicted_lin_cmap
                if hasattr(self, "predicted_lin_cmap")
                else None,
                c=predicted_color,
                s=100,
                marker="o",
                label=f"Predicted points (colored by {self.dim_name})"
                if hasattr(self, "lin_dim_pred")
                else "Predicted points",
            )
            # add the colorbar for the predicted points
            if hasattr(self, "predicted_lin_cmap"):
                cbar = self.fig.colorbar(
                    plt.cm.ScalarMappable(
                        norm=self.predicted_lin_norm, cmap=self.predicted_lin_cmap
                    ),
                    ax=ax,
                    orientation="vertical",
                    fraction=0.046,
                    pad=0.04,
                )
                cbar.set_label(self.predicted_dim_name)
            self.artists["linpos_pred_line"] = ax.plot(
                [],
                [],
                color=predicted_line_color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )[0]
            self.artists["current_predicted_point"] = ax.plot(
                [],
                [],
                color=CURRENT_PREDICTED_POINT_COLOR,
            )[0]

        self.linpoints = {}
        self.lintimes = {}
        self.linpoints["stims_stars"] = []
        self.linpoints["ripples_stars"] = []
        self.linpoints["freezing_stars"] = {}
        self.lintimes["stims_stars"] = []
        self.lintimes["freezing_stars"] = []
        self.lintimes["ripples_stars"] = []

        self.artists["stims_stars"] = ax.scatter(
            [],
            [],
            color=kwargs.get("shock_color", SHOCK_COLOR),
            s=200,
            marker="*",
            label="Stimulation",
            zorder=10,
        )
        self.artists["ripples_stars"] = ax.scatter(
            [],
            [],
            color=kwargs.get("ripples_color", RIPPLES_COLOR),
            s=200,
            marker="*",
            label="Ripples",
            zorder=10,
        )
        # instead of a star, freezing epochs will be shown as horizontal lines above the trajectory
        (self.artists["freezing_stars"],) = ax.plot(
            [],
            [],
            color=kwargs.get("freeze_color", FREEZING_LINE_COLOR),
            linewidth=4,
            label="Freezing",
            zorder=10,
        )

        ax_handles, ax_labels = ax.get_legend_handles_labels()
        if self.plot_all_stims:
            # if plot_all_stims, create a dummy legend entry for old stims
            stim_handle = Line2D(
                [0],
                [0],
                marker="*",
                color="w",  # no line color
                markerfacecolor=ALL_STIMS_COLOR,
                markersize=20,
                linestyle="None",
            )
            stim_label = "All stimulations"
            # append that to current handles and labels
            ax_handles.append(stim_handle)
            ax_labels.append(stim_label)

        ax.set_title("Linearized position across time")
        ax.set_xlabel(
            "Time (hh:mm:ss.ms)",
        )
        ax.set_ylim(-0.05, 1.15)  # leave some space for the freezing epochs
        ax.set_ylabel(
            "Linearized position (u.a.)",
            color="white" if kwargs.get("dark_theme", False) else "black",
        )
        ax.legend(ax_handles, ax_labels, loc="lower left", fontsize=10, framealpha=0.5)
        ax.xaxis.set_major_formatter(
            FuncFormatter(time_formatter)
        )  # Format x-axis as time

    def _setup_polar_panel(self, gs, **kwargs):
        """Setup polar heading direction panel."""
        self.axes["polar"] = self.fig.add_subplot(gs[0, 1], projection="polar")
        ax = self.axes["polar"]

        if kwargs.get("dark_theme", True):
            ax.set_facecolor("#0f0f23")

        # Configure polar plot
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids(range(0, 360, 45))
        ax.tick_params(colors="white" if kwargs.get("dark_theme", True) else "black")
        ax.grid(True, alpha=0.3)

        # Initialize arrows (will be updated during animation)
        self.artists["gt_arrow"] = None
        self.artists["pred_arrow"] = None

        # Title for heading error
        self.artists["hd_title"] = ax.set_title(
            "HD error: -- °",
            color="white" if kwargs.get("dark_theme", True) else "black",
            pad=20,
        )

    def _setup_speed_panel(self, gs, **kwargs):
        """Setup speed histogram panel."""
        self.axes["speed"] = self.fig.add_subplot(gs[1, 1])
        ax = self.axes["speed"]

        if kwargs.get("dark_theme", True):
            ax.set_facecolor("#0f0f23")

        # Create background histogram of all speeds
        if self.predicted is not None:
            ax.hist(
                self.predicted_speeds[~np.isnan(self.predicted_speeds)],
                bins=40,
                color="lightgray",
                alpha=0.3,
                density=True,
            )

        # Initialize speed indicator lines
        self.artists["gt_speed_line"] = ax.axvline(
            0, color=self.true_color, linewidth=3, alpha=0.8
        )
        if self.predicted is not None:
            self.artists["pred_speed_line"] = ax.axvline(
                0, color=self.predicted_color, linewidth=3, alpha=0.8
            )

        ax.set_xlabel(
            "Speed (u.a.)", color="white" if kwargs.get("dark_theme", True) else "black"
        )
        ax.set_ylabel(
            "Density", color="white" if kwargs.get("dark_theme", True) else "black"
        )
        ax.tick_params(colors="white" if kwargs.get("dark_theme", True) else "black")
        ax.grid(True, alpha=0.3)

        # Set speed limits
        if self.predicted is not None:
            speed_range = [
                min(np.nanmin(self.speeds), np.nanmin(self.predicted_speeds)),
                max(np.nanmax(self.speeds), np.nanmax(self.predicted_speeds)) + 1,
            ]
        else:
            speed_range = [self.speeds.min(), self.speeds.max() + 1]
        ax.set_xlim(speed_range)

        # Title for speed error
        self.artists["speed_title"] = ax.set_title(
            "Speed error: -- u.a.",
            color="white" if kwargs.get("dark_theme", True) else "black",
        )

    def _setup_simple_plot(self, **kwargs):
        """Setup simple single-panel trajectory plot using same logic as trajectory panel."""

        usedark = kwargs.pop("dark_theme", False)
        plt.style.use("dark_background") if usedark else plt.style.use("default")

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.patch.set_facecolor("#1a1a2e" if usedark else "white")
        self.axes["trajectory"] = self.ax  # For consistency

        # Use the same setup logic as the trajectory panel
        self._setup_trajectory_panel_logic(
            ax=self.ax, loc_in_gs="trajectory", dark_theme=usedark, **kwargs
        )

        return self.fig, self.axes

    def _setup_normalizer(self, name_axis, **kwargs):
        """Matplotliib normalization function for color mapping."""
        if self.binary_colors[name_axis]:
            return mcolors.BoundaryNorm([0, 1], 2)
        else:
            # return blank norm for continuous data
            return NoNorm()

    def _setup_trajectory_panel_logic(
        self,
        ax: plt.Axes,
        **kwargs,
    ):
        """
        Setup the matplotlib figure and axes.

        Args:
            ax: Matplotlib Axes object to use for plotting
            colormap: Colormap for direction coding (default: 'hsv')
            alpha_trail_line: Transparency for trail lines (default: 0.6)
            alpha_trail_points: Transparency for trail points (default: 0.95)
            alpha_delta_line: Transparency for delta line (default: 0.6)
            pair_points: Whether to pair predicted and true points (default: False)
            binary_colors: Use binary coloring (auto-detected if None)
            shock_color: Color for shock zone direction (1 values, default: 'xkcd:hot pink')
            safe_color: Color for safe zone direction (0 values, default: 'cornflowerblue')
            hlines: List of y-values for horizontal lines (default: None)
            vlines: List of x-values for vertical lines (default: None)
            line_colors: Color(s) for reference lines (default: 'black')
            line_styles: Style(s) for reference lines (default: '--')
            line_widths: Width(s) for reference lines (default: 1.0)
            line_alpha: Transparency for reference lines (default: 0.7)
            custom_lines: List of line segments as [(x1,y1), (x2,y2), ...] or numpy array (default: None)
            custom_line_colors: Color(s) for custom lines (default: 'black')
            custom_line_styles: Style(s) for custom lines (default: '-')
            custom_line_widths: Width(s) for custom lines (default: 2.0)
            custom_line_alpha: Transparency for custom lines (default: 0.8)
            with_ref_bg: Whether to use a reference background image (default: True)
        """
        colormap = kwargs.get("colormap", COLORMAP)
        predicted_cmap = kwargs.get("predicted_cmap", PREDICTED_CMAP)
        alpha_trail_line = kwargs.get("alpha_trail_line", ALPHA_TRAIL_LINE)
        alpha_trail_points = kwargs.get("alpha_trail_points", ALPHA_TRAIL_POINTS)
        alpha_delta_line = kwargs.get("alpha_delta_line", ALPHA_DELTA_LINE)
        binary_colors = kwargs.get("binary_colors", BINARY_COLORS)
        shock_color = kwargs.get("shock_color", SHOCK_COLOR)
        safe_color = kwargs.get("safe_color", SAFE_COLOR)
        shock_color_predicted = kwargs.get(
            "shock_color_predicted", SHOCK_COLOR_PREDICTED
        )
        safe_color_predicted = kwargs.get("safe_color_predicted", SAFE_COLOR_PREDICTED)
        hlines = kwargs.get("hlines", HLINES)
        vlines = kwargs.get("vlines", VLINES)
        self.remove_ticks = kwargs.get("remove_ticks", REMOVE_TICKS)
        self.with_ref_bg = kwargs.get("with_ref_bg", WITH_REF_BG)
        self.true_color = kwargs.get("true_color", TRUE_COLOR)
        self.true_line_color = kwargs.get("true_line_color", TRUE_LINE_COLOR)
        self.predicted_color = kwargs.get("predicted_color", PREDICTED_COLOR)
        self.predicted_line_color = kwargs.get(
            "predicted_line_color", PREDICTED_LINE_COLOR
        )
        self.delta_color = kwargs.get("delta_color", DELTA_COLOR)
        self.delta_color_forward = kwargs.get(
            "delta_color_forward", DELTA_COLOR_FORWARD
        )
        self.delta_color_reverse = kwargs.get(
            "delta_color_reverse", DELTA_COLOR_REVERSE
        )
        self.max_num_stars = kwargs.get(
            "max_num_stars", MAX_NUM_STARS
        )  # Maximum number of stars to plot

        if not hasattr(self, "dims"):
            self.dims = {}

        if not hasattr(self, "predicted_dims"):
            self.predicted_dims = {}

        dim_to_use = kwargs.pop("dim", None)
        if dim_to_use is None:
            dim_to_use = self.dim

        predicted_dim_to_use = kwargs.pop("predicted_dim", None)
        if predicted_dim_to_use is None and self.predicted is not None:
            predicted_dim_to_use = self.predicted_dim

        name_axis = kwargs.get("loc_in_gs", "left")

        if name_axis not in self.axes_names:
            self.axes_names.append(name_axis)
        self.dims[name_axis] = dim_to_use
        if self.predicted is not None:
            self.predicted_dims[name_axis] = predicted_dim_to_use

        if not hasattr(self, "pair_points"):
            # Initialize pair_points if not already set
            self.pair_points = {}

        if not hasattr(self, "xypoints"):
            self.xypoints = {}
            self.xytimes = {}

        if not hasattr(self, "handles_labels"):
            self.handles_labels = {}

        self.pair_points[name_axis] = kwargs.get("pair_points", False)

        if not hasattr(self, "line_colors"):
            self.line_colors = {}
            self.line_styles = {}
            self.line_widths = {}
            self.line_alpha = {}
            self.custom_lines = {}
            self.custom_line_colors = {}
            self.custom_line_styles = {}
            self.custom_line_widths = {}
            self.custom_line_alpha = {}
            self.hlines = {}
            self.vlines = {}

        self.line_colors[name_axis] = kwargs.get("line_colors", "black")
        self.line_styles[name_axis] = kwargs.get("line_styles", "--")
        self.line_widths[name_axis] = kwargs.get("line_widths", 1.0)
        self.line_alpha[name_axis] = kwargs.get("line_alpha", 1)
        self.custom_line_colors[name_axis] = kwargs.get("custom_line_colors", "black")
        self.custom_line_styles[name_axis] = kwargs.get("custom_line_styles", "-")
        self.custom_line_widths[name_axis] = kwargs.get("custom_line_widths", 2.0)
        self.custom_line_alpha[name_axis] = kwargs.get("custom_line_alpha", 0.8)

        # Store line parameters
        self.hlines[name_axis] = hlines or []
        self.vlines[name_axis] = vlines or []
        self.custom_lines[name_axis] = kwargs.get("custom_lines", None) or [
            self.data_helper.maze_coords,
            self.data_helper.shock_zone,
            self.data_helper.safe_zone,
        ]
        if not kwargs.get("custom_lines", None):
            self.custom_line_colors[name_axis] = (
                ["black", SHOCK_COLOR, SAFE_COLOR]
                if not self.with_ref_bg
                else ["white", SHOCK_COLOR, SAFE_COLOR]
            )
            self.custom_line_styles[name_axis] = ["-", "-", "-"]
            self.custom_line_widths[name_axis] = [4, 2, 2]

        # Auto-detect binary data if not specified
        if binary_colors is None:
            unique_values = np.unique(self.dims[name_axis])
            binary_colors = len(unique_values) == 2 and set(unique_values) == {0, 1}
            if binary_colors:
                print(f"Binary direction data detected (0s and 1s) for {name_axis}.")
                print(f"0 (shock zone) -> {shock_color}")
                print(f"1 (safe zone) -> {safe_color}")

        if not hasattr(self, "binary_colors"):
            self.binary_colors = {}
        self.binary_colors[name_axis] = binary_colors  # boolean flag for binary colors
        self.shock_color = shock_color
        self.safe_color = safe_color
        if not hasattr(self, "norm"):
            self.norm = {}
        if name_axis not in self.norm:
            self.norm[name_axis] = self._setup_normalizer(name_axis, **kwargs)

        if not hasattr(self, "colormap"):
            self.colormap = {}
        if not hasattr(self, "predicted_colormap"):
            self.predicted_colormap = {}

        if self.binary_colors[name_axis]:
            # Binary color mapping
            self.colormap[name_axis] = None
            self.predicted_colormap[name_axis] = None
            # Create custom colors for binary data
            self.colormap[name_axis] = {0: shock_color, 1: safe_color}
            self.predicted_colormap[name_axis] = {
                0: shock_color_predicted,
                1: safe_color_predicted,
            }
        else:
            # Continuous color mapping
            if isinstance(colormap, str):
                self.colormap[name_axis] = cm.get_cmap(colormap)
                self.predicted_colormap[name_axis] = cm.get_cmap(predicted_cmap)
            else:
                self.colormap[name_axis] = colormap
                self.predicted_colormap[name_axis] = predicted_cmap

            if self.dim_name == "Dist2Threat":
                colormap = LinearSegmentedColormap.from_list(
                    "direction_cmap", [shock_color, safe_color], N=256
                )
                predicted_colormap = LinearSegmentedColormap.from_list(
                    "predicted_direction_cmap",
                    [shock_color_predicted, safe_color_predicted],
                    N=256,
                )
                self.colormap[name_axis] = colormap
                self.predicted_colormap[name_axis] = predicted_colormap
            if self.dim_name == "Head Direction":
                colormap = cm.get_cmap("hsv")
                predicted_colormap = cm.get_cmap("hsv")
                self.colormap[name_axis] = colormap
                self.predicted_colormap[name_axis] = predicted_colormap
            # self.norm = Normalize(vmin=np.min(dim_to_use), vmax=np.max(dim_to_use))

        if name_axis not in self.artists:
            self.artists[name_axis] = {}

        if name_axis not in self.xypoints:
            self.xypoints[name_axis] = {}
            self.xytimes[name_axis] = {}
            self.xypoints[name_axis]["stims_stars"] = []
            self.xypoints[name_axis]["ripples_stars"] = []
            self.xytimes[name_axis]["stims_stars"] = []
            self.xytimes[name_axis]["ripples_stars"] = []

        if name_axis not in self.handles_labels:
            self.handles_labels[name_axis] = {
                "handles": [],
                "labels": [],
            }

        if self.plot_stims:
            # If plotting all stimulations, we use and directly plot all stars
            # Will be called in init_animation()
            self.artists[name_axis]["stims_stars"] = ax.scatter(
                [],
                [],
                c=kwargs.get("shock_color", SHOCK_COLOR),
                s=400,
                marker="*",
                label="Stimulation",
                zorder=10,
            )
            self.artists[name_axis]["ripples_stars"] = ax.scatter(
                [],
                [],
                c=kwargs.get("ripples_color", RIPPLES_COLOR),
                s=300,
                marker="*",
                label="Ripples",
                zorder=10,
            )
        if self.predicted_heatmap is not None:
            self.artists[name_axis]["predicted_heatmap"] = ax.imshow(
                np.zeros_like(self.predicted_heatmap[0]),
                extent=(0, 1, 0, 1),
                origin="lower",
                # cmap=cm.get_cmap("Reds"),
                cmap="Greys" if not self.with_ref_bg else "Greys_r",
                alpha=0.7,
                zorder=1,
            )

        (self.artists[name_axis]["line"],) = ax.plot(
            [], [], "-", alpha=alpha_trail_line, linewidth=4, color=self.true_line_color
        )

        if not self.pair_points[name_axis]:
            # if not pairing points, we create a nice line in between each point of its class (predicted or true, otherwise, we'll link each point to its predicted counterpart)
            (self.artists[name_axis]["predicted_line"],) = ax.plot(
                [],
                [],
                "-",
                alpha=alpha_trail_line,
                linewidth=4,
                color=self.predicted_line_color,
            )
            (self.artists[name_axis]["delta_predicted_true"],) = ax.plot(
                [], [], "-", alpha=alpha_delta_line, linewidth=4, color=self.delta_color
            )
        else:
            segments = np.empty((0, 2, 2), dtype=float)
            self.artists[name_axis]["delta_predicted_true"] = ax.add_collection(
                LineCollection(
                    segments,
                    linewidths=4,
                    alpha=alpha_delta_line,
                    zorder=5,
                )
            )

        if self.linpositions is not None or self.linpredicted is not None:
            # If linearized positions are provided, we use them for the maze path
            self.linear_path = np.array(
                [
                    [-0.05, 0],
                    [-0.05, 0.55],
                    [-0.05, 1.05],
                    [0.55, 1.05],
                    [1.05, 1.05],
                    [1.05, 0.55],
                    [1.05, 0],
                ]
            )
            points = self.linear_path.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_lengths = np.sqrt(
                np.sum(np.diff(self.linear_path, axis=0) ** 2, axis=1)
            )
            self.cum_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
            self.midpoints = 0.5 * (self.cum_lengths[:-1] + self.cum_lengths[1:])
            colors = self.midpoints / self.cum_lengths[-1]
            self.total_length = self.cum_lengths[-1]

            colormap = LinearSegmentedColormap.from_list(
                "direction_cmap", [shock_color, safe_color], N=256
            )
            lc = LineCollection(
                segments,
                array=colors,
                cmap=colormap,
                linewidths=4,
                alpha=1,
                zorder=5,
            )

            n_ticks = 10
            tick_positions = np.linspace(0, 1, n_ticks)
            tick_coords = []

            # Interpolate positions along the linear path
            for pos in tick_positions:
                dist_along = pos * self.total_length
                for i in range(len(self.cum_lengths) - 1):
                    if self.cum_lengths[i] <= dist_along <= self.cum_lengths[i + 1]:
                        segment_ratio = (dist_along - self.cum_lengths[i]) / (
                            self.cum_lengths[i + 1] - self.cum_lengths[i]
                        )
                        point = (1 - segment_ratio) * self.linear_path[
                            i
                        ] + segment_ratio * self.linear_path[i + 1]
                        tick_coords.append((point, pos))
                        break

            # Plot ticks and labels
            points_for_legend = []
            for point, pos in tick_coords:
                ax.plot(
                    point[0],
                    point[1],
                    "|" if 0.3 < pos and pos < 0.7 else "-",
                    color=colormap(pos),
                    markersize=13,
                )
                (a_bar,) = ax.plot(
                    point[0],
                    point[1],
                    "-",
                    color=colormap(pos),
                    markersize=11,
                    zorder=-1,
                )
                ax.text(
                    point[0] if 0.3 < pos and pos < 0.7 else point[0] - 0.022,
                    point[1] + 0.01 if 0.3 < pos and pos < 0.7 else point[1],
                    f"{pos:.1f}",
                    color=colormap(pos),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
                points_for_legend.append(a_bar)
            self.artists[name_axis]["linearized_line"] = ax.add_collection(lc)
            ax.plot(
                self.linear_path[-1, 0],
                self.linear_path[-1, 1],
                "v",
                ms=12,
                color=colormap(1.0),
            )
            ax.axis("off")

            # Arrow marker for current linearized position
            if self.linpositions is not None:
                (self.artists[name_axis]["linearized_arrow_true"],) = ax.plot(
                    [],
                    [],
                    "o",
                    markersize=10,
                    color=self.true_color,
                    alpha=1,
                    zorder=100,
                )
            if self.linpredicted is not None:
                (self.artists[name_axis]["linearized_arrow_predicted"],) = ax.plot(
                    [],
                    [],
                    "o",
                    markersize=10,
                    color=self.predicted_color,
                    alpha=1,
                    zorder=100,
                )

        # Initialize scatter plot for trail points
        if self.binary_colors[name_axis]:
            # For binary data, we'll update colors manually
            self.artists[name_axis]["points"] = ax.scatter(
                [], [], s=50, alpha=alpha_trail_points
            )
            if self.predicted is not None:
                self.artists[name_axis]["predicted_points"] = ax.scatter(
                    [], [], s=50, alpha=alpha_trail_points
                )
            else:
                self.artists[name_axis]["predicted_points"] = None
        else:
            # For continuous data, use colormap
            self.artists[name_axis]["points"] = ax.scatter(
                [],
                [],
                c=[],
                s=50,
                alpha=alpha_trail_points,
                cmap=self.colormap[name_axis],
                norm=self.norm[name_axis],
            )
            if self.predicted is not None:
                self.artists[name_axis]["predicted_points"] = ax.scatter(
                    [],
                    [],
                    s=50,
                    c=[],
                    alpha=alpha_trail_points,
                    cmap=self.predicted_colormap[name_axis],
                    norm=self.norm[name_axis],
                )
            else:
                self.artists[name_axis]["predicted_points"] = None

        # Initialize current position marker
        self.artists[name_axis]["current_point"] = ax.scatter(
            [],
            [],
            c=CURRENT_POINT_COLOR,
            s=200,
            marker="o",
            edgecolors="black",
            linewidth=2.5,
            zorder=10,
        )
        if self.predicted is not None:
            self.artists[name_axis]["current_predicted_point"] = ax.scatter(
                [],
                [],
                c=CURRENT_PREDICTED_POINT_COLOR,
                s=200,
                marker="x",
                linewidth=5,
                zorder=10,
            )

        handles_for_legend = []
        labels_for_legend = []

        # Add true trajectory legend_handler
        true_handle = Line2D(
            [0],
            [0],
            color=self.true_line_color,
            linewidth=2,
            label="True Trajectory",
            marker="o",
        )
        handles_for_legend.append(true_handle)
        labels_for_legend.append("True Trajectory")

        if not self.binary_colors[name_axis]:
            current_point_handle = Line2D(
                [0], [0], marker="o", color="red", markersize=10
            )
            current_point_label = "Current True Position"
            handles_for_legend.append(current_point_handle)
            labels_for_legend.append(current_point_label)
        else:
            # Add custom legend for binary colors
            legend_elements = [
                Patch(facecolor=shock_color, label="Shock Zone (0)"),
                Patch(facecolor=safe_color, label="Safe Zone (1)"),
            ]
            label_elements = [
                "Shock Zone (0)",
                "Safe Zone (1)",
            ]
            handles_for_legend.extend(legend_elements)
            labels_for_legend.extend(label_elements)

        if self.predicted is not None:
            current_predicted_point_handle = Line2D(
                [0],
                [0],
                marker="x",
                color=CURRENT_PREDICTED_POINT_COLOR,
                markersize=10,
                linestyle="None",
            )
            current_predicted_point_label = "Current Predicted Position"
            handles_for_legend.append(current_predicted_point_handle)
            labels_for_legend.append(current_predicted_point_label)

            if not self.pair_points[name_axis]:
                # Add predicted trajectory legend_handler
                predicted_handle = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=self.predicted_line_color,
                    linewidth=2,
                    label="Predicted Trajectory",
                )
                handles_for_legend.append(predicted_handle)
                labels_for_legend.append("Predicted Trajectory")
                delta_handle = Line2D(
                    [0],
                    [0],
                    color=self.delta_color,
                    linewidth=4,
                )
                delta_label = ("Delta (Predicted - True)",)
                handles_for_legend.append(delta_handle)
                labels_for_legend.append(delta_label)
            else:
                # add pair points legend with direction info
                delta_forward_handle = Line2D(
                    [0],
                    [0],
                    color=self.delta_color_forward,
                    linewidth=4,
                )
                delta_forward_label = (
                    "Forward Direction"
                    if not self.dim_name == "speed_mask"
                    else "Faster Speed"
                )
                handles_for_legend.append(delta_forward_handle)
                labels_for_legend.append(delta_forward_label)
                delta_discordant_handle = Line2D(
                    [0],
                    [0],
                    color=self.delta_color_reverse,
                    linewidth=4,
                )
                delta_discordant_label = (
                    "Reverse Direction"
                    if not self.dim_name == "speed_mask"
                    else "Slower Speed"
                )
                handles_for_legend.append(delta_discordant_handle)
                labels_for_legend.append(delta_discordant_label)

        if self.linpositions is not None or self.linpredicted is not None:
            handles_for_legend.append(tuple(points_for_legend))
            labels_for_legend.append("Distance to Shock")

        if self.plot_stims and self.very_simple_plot:
            stims_handle = self.artists[name_axis]["stims_stars"]
            ripples_handle = self.artists[name_axis]["ripples_stars"]
            handles_for_legend.append(stims_handle)
            labels_for_legend.append("Stimulation")
            handles_for_legend.append(ripples_handle)
            labels_for_legend.append("Ripples")

        self.handles_labels[name_axis]["handles"] = handles_for_legend
        self.handles_labels[name_axis]["labels"] = labels_for_legend

        ax.legend(
            handles=handles_for_legend,
            labels=labels_for_legend,
            loc=[0.32, 0.05] if not self.simple_plot else "upper left",
            handlelength=1.5,
            handleheight=1.2,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            fontsize=10,
        )
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        title_text = "Position Trajectory"

        if not self.very_simple_plot:
            self.artists[name_axis]["pos_title"] = ax.set_title(
                title_text,
                color="white" if kwargs.get("dark_theme", True) else "black",
                pad=30,
                fontsize="large",
            )
        else:
            self.artists["fig_title"] = self.fig.suptitle(
                title_text,
                color="white" if kwargs.get("dark_theme", True) else "black",
                fontsize="large",
            )

        return self.fig, ax

    def _add_reference_lines(self, ax, name_axis, colors, styles, widths, alpha):
        """Add permanent horizontal and vertical reference lines."""

        # Ensure parameters are lists for consistent handling
        def ensure_list(param, default_length):
            if isinstance(param, (list, tuple)):
                return list(param)
            else:
                return [param] * default_length

        total_lines = len(self.hlines[name_axis]) + len(self.vlines[name_axis])
        if total_lines == 0:
            return

        colors = ensure_list(colors, total_lines)
        styles = ensure_list(styles, total_lines)
        widths = ensure_list(widths, total_lines)
        alphas = ensure_list(alpha, total_lines)

        line_idx = 0

        # Add horizontal lines
        for y_val in self.hlines[name_axis]:
            ax.axhline(
                y=y_val,
                color=colors[line_idx % len(colors)],
                linestyle=styles[line_idx % len(styles)],
                linewidth=widths[line_idx % len(widths)],
                alpha=alphas[line_idx % len(alphas)],
                zorder=1,  # Behind the trajectory
            )
            line_idx += 1

        # Add vertical lines
        for x_val in self.vlines[name_axis]:
            ax.axvline(
                x=x_val,
                color=colors[line_idx % len(colors)],
                linestyle=styles[line_idx % len(styles)],
                linewidth=widths[line_idx % len(widths)],
                alpha=alphas[line_idx % len(alphas)],
                zorder=1,  # Behind the trajectory
            )

    def get_linearized_point(self, pos_frac):
        """Given a linearized position (0 to 1), return x, y on path."""
        distance_along = pos_frac * self.total_length
        for i in range(len(self.cum_lengths) - 1):
            if self.cum_lengths[i] <= distance_along <= self.cum_lengths[i + 1]:
                t = (distance_along - self.cum_lengths[i]) / (
                    self.cum_lengths[i + 1] - self.cum_lengths[i]
                )
                point = (1 - t) * self.linear_path[i] + t * self.linear_path[i + 1]
                return point
        return self.linear_path[-1]

    def _add_custom_lines(self, ax, name_axis, colors, styles, widths, alpha):
        """Add custom line segments (like maze walls, boundaries, etc.)."""
        if not self.custom_lines[name_axis]:
            return

        # Handle different input formats
        lines_to_plot = []

        for line_data in self.custom_lines[name_axis]:
            # Convert to numpy array for easier handling
            line_array = np.array(line_data)

            if line_array.ndim == 2 and line_array.shape[1] == 2:
                # Single line segment as array of points
                lines_to_plot.append(line_array)
            elif line_array.ndim == 1 and len(line_array) == 4:
                # Single line as [x1, y1, x2, y2]
                lines_to_plot.append(
                    np.array(
                        [[line_array[0], line_array[1]], [line_array[2], line_array[3]]]
                    )
                )
            else:
                warn(f"Warning: Unrecognized line format: {line_data}")

        if not lines_to_plot:
            return

        # Ensure parameters are lists
        def ensure_list(param, default_length):
            if isinstance(param, (list, tuple)):
                return list(param)
            else:
                return [param] * default_length

        colors = ensure_list(colors, len(lines_to_plot))
        styles = ensure_list(styles, len(lines_to_plot))
        widths = ensure_list(widths, len(lines_to_plot))
        alphas = ensure_list(alpha, len(lines_to_plot))

        # Plot each line segment
        for i, line_points in enumerate(lines_to_plot):
            ax.plot(
                line_points[:, 0],  # X coordinates
                line_points[:, 1],  # Y coordinates
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)],
                linewidth=widths[i % len(widths)],
                alpha=alphas[i % len(alphas)],
                zorder=1,  # Behind the trajectory
            )

    def animate_frame(self, frame: int, **kwargs) -> list:
        """Animation function for a single frame."""
        if frame == 0:
            return self.flatten_artists()

        # Calculate trail window
        start_idx = max(0, frame + 1 - self.trail_length)
        end_idx = frame + 1

        if end_idx <= start_idx:
            return self.flatten_artists()

        # Update trajectory panel (both simple and analysis modes)
        self._update_trajectory_panel(frame, start_idx, end_idx)
        if not self.very_simple_plot:
            # Calculate trail window
            start_idx = max(0, frame + 1 - self.lin_movie_duration)
            end_idx = frame + 1
            self._update_linpos_movie(frame, start_idx, end_idx)

        # Update analysis panels only in analysis mode
        if self.fourD_analysis_mode:
            self._update_polar_panel(frame)
            self._update_speed_panel(frame)

        if kwargs.get("save_path", None) is not None:
            save_path = kwargs["save_path"]
            bbox_inches = kwargs.get("bbox_inches", "tight")
            dpi = kwargs.get("dpi", 300)
            transparent = kwargs.get("transparent", True)
            pad_inches = kwargs.get("pad_inches", 0.1)
            if not save_path.endswith(".png"):
                warn("The frame will be and should only be saved as a PNG file.")
                save_path += ".png"

            self.fig.savefig(
                save_path,
                bbox_inches=bbox_inches,
                dpi=dpi,
                transparent=transparent,
                pad_inches=pad_inches,
            )
            plt.close(self.fig)

        return self.flatten_artists()

    def flatten_artists(self, artists_dict=None):
        """Recursively flatten nested artist dict into a list."""
        if artists_dict is None:
            artists_dict = self.artists
        flat_list = []
        for value in artists_dict.values():
            if (
                hasattr(value, "set_figure")
                and getattr(value, "figure", None) is not None
                and getattr(value, "axes", None) is not None
            ):  # It's an Artist object
                flat_list.append(value)
            elif isinstance(value, dict):  # Nested dictionary
                flat_list.extend(self.flatten_artists(value))
        return flat_list

    def init_animation(self):
        # Clear all artists
        for artist in self.flatten_artists():
            # check it's not an imshow artist
            from matplotlib.image import AxesImage

            if hasattr(artist, "set_data") and not isinstance(artist, AxesImage):
                artist.set_data([], [])
            elif hasattr(artist, "set_offsets"):
                artist.set_offsets(np.column_stack(([], [])))

        # plot permanent lines
        for name_axis in self.axes_names:
            ax = self.axes[name_axis]
            if self.linpositions is not None:
                ax.set_xlim(-0.06, 1.06)
                ax.set_ylim(-0.01, 1.06)
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            # remove ticks
            if self.remove_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            # Add permanent reference lines
            self._add_reference_lines(
                ax=ax,
                name_axis=name_axis,
                colors=self.line_colors[name_axis],
                styles=self.line_styles[name_axis],
                widths=self.line_widths[name_axis],
                alpha=self.line_alpha[name_axis],
            )

            # Add custom line segments
            self._add_custom_lines(
                ax=ax,
                name_axis=name_axis,
                colors=self.custom_line_colors[name_axis],
                styles=self.custom_line_styles[name_axis],
                widths=self.custom_line_widths[name_axis],
                alpha=self.custom_line_alpha[name_axis],
            )

            if self.with_ref_bg:
                try:
                    reference_image = self.data_helper.aligned_ref
                    ax.imshow(
                        reference_image,
                        cmap="gray",
                        extent=[0, 1, 0, 1],
                        vmin=reference_image.min(),
                        vmax=reference_image.max(),
                        origin="lower",
                    )
                except AttributeError:
                    raise ValueError(
                        "No reference image found. Please provide a reference image."
                    )
            if self.plot_all_stims:
                ax.scatter(
                    self.stims_positions[:, 0],
                    self.stims_positions[:, 1],
                    color=ALL_STIMS_COLOR,
                    s=200,
                    marker="*",
                    label="All stimulations",
                    zorder=2,
                )
            if self.predicted_heatmap is not None:
                self.artists[name_axis]["predicted_heatmap"].set_data(
                    self.predicted_heatmap[0]
                )
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
        return self.flatten_artists()

    def _update_trajectory_panel(self, frame, start_idx, end_idx):
        """Update the main trajectory panel using the same logic as simple animate_frame."""
        if end_idx <= start_idx:
            return

        for name_axis in self.axes_names:
            # Get trail data
            trail_positions = self.positions[start_idx:end_idx]
            trail_directions = self.dims[name_axis][start_idx:end_idx]
            trail_times = self.positionTime[start_idx:end_idx]

            if self.predicted is not None:
                trail_predicted = self.predicted[start_idx:end_idx]
                trail_directions_predicted = self.predicted_dims[name_axis][
                    start_idx:end_idx
                ]
                trail_predicted_times = self.positionTime[start_idx:end_idx]
                assert trail_predicted.shape[0] == trail_directions_predicted.shape[0]
            else:
                trail_predicted = None
                trail_directions_predicted = None

            self.artists[name_axis]["line"].set_data(
                trail_positions[:, 0], trail_positions[:, 1]
            )

            if not self.pair_points[name_axis]:
                # Update trail line
                if self.predicted is not None:
                    self.artists[name_axis]["predicted_line"].set_data(
                        trail_predicted[:, 0], trail_predicted[:, 1]
                    )

            if self.linpositions is not None or self.linpredicted is not None:
                # Update linearized positions if available
                if self.linpositions is not None:
                    current_lin_pos = self.linpositions[frame : frame + 1]
                    x, y = self.get_linearized_point(current_lin_pos[0])
                    self.artists[name_axis]["linearized_arrow_true"].set_data(
                        ([x], [y])
                    )
                if self.linpredicted is not None:
                    current_lin_predicted = self.linpredicted[frame : frame + 1]

                    x, y = self.get_linearized_point(current_lin_predicted[0])
                    if (
                        x
                        == self.artists[name_axis][
                            "linearized_arrow_predicted"
                        ].get_xdata()
                        and y
                        == self.artists[name_axis][
                            "linearized_arrow_predicted"
                        ].get_ydata()
                    ):
                        pass
                    else:
                        self.artists[name_axis]["linearized_arrow_predicted"].set_data(
                            ([x], [y])
                        )

            n_trail_points = len(trail_positions)
            alphas = np.linspace(0.4, 1, n_trail_points)

            # Update trail points with direction colors AND alpha gradients
            if self.binary_colors[name_axis]:
                # Binary color mapping
                trail_colors = [
                    self.colormap[name_axis][int(d)] for d in trail_directions
                ]
                # apply lighten_color to trail colors
                trail_colors = [
                    lighten_color(color, factor)
                    for color, factor in zip(
                        trail_colors, np.linspace(0.2, 1, len(trail_colors))
                    )
                ]
                # switch to rgba to handle alpha
                trail_colors_rgba = []
                for i, d in enumerate(trail_directions):
                    base_color = trail_colors[i]
                    # Convert to RGBA with alpha
                    rgba = (*mcolors.to_rgb(base_color), alphas[i])
                    trail_colors_rgba.append(rgba)

                self.artists[name_axis]["points"].set_offsets(trail_positions)
                self.artists[name_axis]["points"].set_color(trail_colors_rgba)
                if (
                    self.predicted is not None
                    and self.artists[name_axis]["predicted_points"] is not None
                ):
                    trail_colors_predicted = [
                        self.predicted_colormap[name_axis][int(d)]
                        for d in trail_directions_predicted
                    ]
                    trail_colors_predicted = [
                        lighten_color(color, factor)
                        for color, factor in zip(
                            trail_colors_predicted,
                            np.linspace(0.2, 1, len(trail_colors_predicted)),
                        )
                    ]
                    trail_colors_predicted_rgba = []
                    for i, d in enumerate(trail_directions_predicted):
                        if not np.isnan(d).any():
                            base_color = trail_colors_predicted[i]
                            # Convert to RGBA with alpha
                            predicted_rgba = (*mcolors.to_rgb(base_color), alphas[i])
                            trail_colors_predicted_rgba.append(predicted_rgba)
                        else:
                            trail_colors_predicted_rgba.append(
                                (0.5, 0.5, 0.5, alphas[i])
                            )

                    self.artists[name_axis]["predicted_points"].set_offsets(
                        trail_predicted
                    )
                    self.artists[name_axis]["predicted_points"].set_color(
                        trail_colors_predicted_rgba
                    )
            else:
                # Continuous color mapping
                if self.dim_name != "dummy":
                    trail_colors = self.colormap[name_axis](
                        self.norm[name_axis](trail_directions)
                    )
                    trail_colors = [
                        lighten_color(color, factor)
                        for color, factor in zip(
                            trail_colors, np.linspace(0.2, 1, len(trail_colors))
                        )
                    ]
                else:
                    # If dim_name is dummy, use a gradient color from pale red to intense red
                    trail_colors = [
                        (
                            1.0,
                            0.8 * (1 - i / (n_trail_points - 1)),
                            0.8 * (1 - i / (n_trail_points - 1)),
                        )
                        for i in range(n_trail_points)
                    ]
                trail_colors_rgba = []
                for i, d in enumerate(trail_directions):
                    base_color = trail_colors[i]
                    # Convert to RGBA with alpha
                    rgba = (base_color[0], base_color[1], base_color[2], alphas[i])
                    trail_colors_rgba.append(rgba)

                self.artists[name_axis]["points"].set_offsets(trail_positions)
                self.artists[name_axis]["points"].set_color(trail_colors_rgba)

                if (
                    self.predicted is not None
                    and self.artists[name_axis]["predicted_points"] is not None
                ):
                    if self.dim_name != "dummy":
                        trail_predicted_colors = self.predicted_colormap[name_axis](
                            self.norm[name_axis](trail_directions_predicted)
                        )
                        trail_predicted_colors = [
                            lighten_color(color, factor)
                            for color, factor in zip(
                                trail_predicted_colors,
                                np.linspace(0.2, 1, len(trail_colors)),
                            )
                        ]
                    else:
                        # If dim_name is dummy, use a gradient color from pale green to intense green
                        trail_predicted_colors = [
                            (
                                0.8 * (1 - i / (n_trail_points - 1)),
                                1.0,
                                0.8 * (1 - i / (n_trail_points - 1)),
                            )
                            for i in range(n_trail_points)
                        ]
                    predicted_colors_rgba = []
                    for i, d in enumerate(trail_directions_predicted):
                        if not np.isnan(d).any():
                            base_color = trail_predicted_colors[i]

                            rgba = (
                                base_color[0],
                                base_color[1],
                                base_color[2],
                                alphas[i],
                            )
                            predicted_colors_rgba.append(rgba)
                        else:
                            predicted_colors_rgba.append(
                                (0.5, 0.5, 0.5, alphas[i])
                            )  # Gray for NaN

                    self.artists[name_axis]["predicted_points"].set_offsets(
                        trail_predicted
                    )
                    self.artists[name_axis]["predicted_points"].set_color(
                        predicted_colors_rgba
                    )

            # Update current position with color based on direction
            current_pos = self.positions[frame : frame + 1]
            current_dir = self.dims[name_axis][frame]
            if self.predicted is not None and trail_predicted is not None:
                current_predicted_pos = self.predicted[frame : frame + 1]
                current_predicted_dir = self.dims[name_axis][frame]
                if not self.pair_points[name_axis]:
                    self.artists[name_axis]["delta_predicted_true"].set_data(
                        [current_pos[0, 0], current_predicted_pos[0, 0]],
                        [current_pos[0, 1], current_predicted_pos[0, 1]],
                    )
                elif self.dim_name != "speed_mask":
                    # For paired points, we need to create segments
                    # first, we find common timepoints
                    segments = np.stack([trail_positions, trail_predicted], axis=1)

                    if self.velocity_true is not None and (
                        self.speed_mask is None
                        or np.logical_not(
                            np.isnan(self.speed_mask[start_idx:end_idx])
                        ).sum()
                        >= 0
                    ):
                        velocity_true = self.velocity_true[start_idx:end_idx]

                        position_diff = (
                            self.predicted[start_idx:end_idx]
                            - self.positions[start_idx:end_idx]
                        )
                        # Create a mask for concordant and discordant directions by using dot product
                        mask = np.sum(velocity_true * position_diff, axis=1) >= 0
                        mask = self.dims[name_axis][start_idx:end_idx].astype(bool)
                        colors = np.where(
                            mask,
                            self.delta_color_forward,
                            self.delta_color_reverse,
                        )
                    else:
                        colors = np.full(segments.shape[0], self.delta_color)
                    self.artists[name_axis]["delta_predicted_true"].set_segments(
                        segments
                    )
                    self.artists[name_axis]["delta_predicted_true"].set_color(colors)
                elif self.dim_name == "speed_mask":
                    segments = np.stack([trail_positions, trail_predicted], axis=1)
                    self.artists[name_axis]["delta_predicted_true"].set_segments(
                        segments
                    )
                    colors = np.where(
                        trail_directions,
                        self.delta_color_forward,
                        self.delta_color_reverse,
                    )
                    self.artists[name_axis]["delta_predicted_true"].set_color(colors)
                else:
                    raise ValueError(
                        "Error: paired points logic only works for speed_mask or when velocity_true is provided."
                    )

            if self.predicted_heatmap is not None:
                current_heatmap = self.predicted_heatmap[frame]
                masked = np.ma.masked_where(
                    current_heatmap <= current_heatmap.mean(), current_heatmap
                )
                self.artists[name_axis]["predicted_heatmap"].set_data(masked)
                vmin = masked.min()
                vmax = masked.max()
                if vmin == vmax:
                    vmax = vmin + 1e-10
                self.artists[name_axis]["predicted_heatmap"].set_clim(
                    vmin=vmin, vmax=vmax
                )

            if self.binary_colors[name_axis]:
                current_color = self.colormap[name_axis][int(current_dir)]
                self.artists[name_axis]["current_point"].set_offsets(current_pos)
                self.artists[name_axis]["current_point"].set_color(current_color)
                if (
                    self.predicted is not None
                    and "current_predicted_point" in self.artists[name_axis]
                ):
                    self.artists[name_axis]["current_predicted_point"].set_offsets(
                        current_predicted_pos
                    )
                    self.artists[name_axis]["current_predicted_point"].set_color(
                        self.predicted_colormap[name_axis][int(current_predicted_dir)]
                    )
            else:
                self.artists[name_axis]["current_point"].set_offsets(current_pos)
                if (
                    self.predicted is not None
                    and "current_predicted_point" in self.artists[name_axis]
                ):
                    self.artists[name_axis]["current_predicted_point"].set_offsets(
                        current_predicted_pos
                    )
                # Keep current point red for continuous data for visibility

            if self.plot_stims:
                # update stims, freezing, and ripples markers
                for name, indices in [
                    ("stims_stars", self.stim_indices),
                    ("ripples_stars", self.ripples_indices),
                ]:
                    if name not in self.artists[name_axis]:
                        continue
                    try:
                        mask = indices == frame
                    except:
                        continue
                    if np.any(mask):
                        x_data = self.positions[indices[mask], 0]
                        y_data = self.positions[indices[mask], 1]
                        try:
                            if (
                                self.xypoints[name_axis][name] is None
                                or self.xypoints[name_axis][name].size == 0
                            ):
                                self.xypoints[name_axis][name] = np.column_stack(
                                    (x_data, y_data)
                                ).reshape(-1, 2)
                                self.xytimes[name_axis][name] = self.positionTime[frame]
                            else:
                                if (
                                    x_data in self.xypoints[name_axis][name][:, 0]
                                    and y_data in self.xypoints[name_axis][name][:, 1]
                                ):
                                    # skip if already exists
                                    continue
                                self.xypoints[name_axis][name] = np.vstack(
                                    (
                                        self.xypoints[name_axis][name],
                                        np.column_stack((x_data, y_data)).reshape(
                                            -1, 2
                                        ),
                                    )
                                )
                                self.xytimes[name_axis][name] = np.hstack(
                                    (
                                        self.xytimes[name_axis][name],
                                        self.positionTime[frame],
                                    )
                                )
                        except (NameError, AttributeError):
                            self.xypoints[name_axis][name] = np.column_stack(
                                (x_data, y_data)
                            ).reshape(-1, 2)
                            self.xytimes[name_axis][name] = self.positionTime[frame]
                        # if more than max_num_stars different timepoints, keep only the last ones
                        if (
                            len(np.unique(self.xytimes[name_axis][name]))
                            > self.max_num_stars
                        ):
                            unique_times = np.unique(self.xytimes[name_axis][name])
                            times_to_keep = sorted(unique_times)[-5:]
                            mask = np.isin(self.xytimes[name_axis][name], times_to_keep)
                            self.xypoints[name_axis][name] = self.xypoints[name_axis][
                                name
                            ][mask]
                            self.xytimes[name_axis][name] = self.xytimes[name_axis][
                                name
                            ][mask]
                        too_long = (
                            self.xytimes[name_axis][name]
                            < self.positionTime[
                                max(0, frame + 1 - self.lin_movie_duration)
                            ]
                        )
                        if np.any(too_long):
                            self.xypoints[name_axis][name] = self.xypoints[name_axis][
                                name
                            ][~too_long]
                            self.xytimes[name_axis][name] = self.xytimes[name_axis][
                                name
                            ][~too_long]
                        self.artists[name_axis][name].set_offsets(
                            self.xypoints[name_axis][name]
                        )

            # Update title with current frame info and direction
            if not self.be_fast:
                if self.binary_colors[name_axis]:
                    zone_name = "Shock Zone" if current_dir == 0 else "Safe Zone"
                    title_text = f"Position Trajectory - Frame {frame + 1}/{self.total_frames} - {zone_name} @{timedelta(seconds=self.positionTime[frame].astype(float))}"
                else:
                    title_text = f"Position Trajectory - Frame {frame + 1}/{self.total_frames} @{timedelta(seconds=self.positionTime[frame].astype(float))}"
                    # nice but very slow  @{timedelta(seconds=time[-1])}

                # In analysis mode, also show position error
                if self.predicted is not None:
                    pos_error = np.nanmean(
                        np.linalg.norm(
                            self.predicted[: frame + 1] - self.positions[: frame + 1],
                            axis=1,
                        )
                    )
                    title_text = f"Position error: {pos_error:.2f} cm | " + title_text
                if not self.very_simple_plot:
                    self.artists[name_axis]["pos_title"].set_text(title_text)
                else:
                    self.artists["fig_title"].set_text(title_text)

    def _update_polar_panel(self, frame):
        """Update the polar heading panel."""
        if "polar" not in self.axes:
            return

        ax = self.axes["polar"]

        # Remove previous arrows
        if self.artists["gt_arrow"]:
            self.artists["gt_arrow"].remove()
        if self.artists["pred_arrow"]:
            self.artists["pred_arrow"].remove()

        # Get current head_direction
        gt_heading = np.radians(self.head_direction[frame])

        # Draw ground truth arrow
        self.artists["gt_arrow"] = ax.arrow(
            gt_heading,
            0,
            0,
            0.8,
            head_width=0.1,
            head_length=0.1,
            fc=self.true_color,
            ec=self.true_color,
            linewidth=3,
            alpha=0.8,
        )

        # Draw predicted arrow if available
        if self.predicted_head_direction is not None:
            idx = np.where(self.posIndex == self.true_posIndex[frame])[0][0]
            pred_heading = np.radians(self.predicted_head_direction[idx])
            self.artists["pred_arrow"] = ax.arrow(
                pred_heading,
                0,
                0,
                0.6,
                head_width=0.1,
                head_length=0.1,
                fc=self.predicted_color,
                ec=self.predicted_color,
                linewidth=3,
                alpha=0.8,
            )

            # Calculate heading error
            hd_error = np.abs(np.degrees(pred_heading - gt_heading))
            if hd_error > 180:
                hd_error = 360 - hd_error
            self.artists["hd_title"].set_text(f"HD error: {hd_error:.2f}°")

    def _update_speed_panel(self, frame):
        """Update the speed histogram panel."""
        if "speed" not in self.axes:
            return

        # Update speed indicator lines
        current_gt_speed = self.speeds[frame]
        self.artists["gt_speed_line"].set_xdata([current_gt_speed])

        if self.predicted_speeds is not None and self.posIndex_in_true_posIndex[frame]:
            idx = np.where(self.posIndex == self.true_posIndex[frame])[0][0]
            current_pred_speed = self.predicted_speeds[idx]
            self.artists["pred_speed_line"].set_xdata([current_pred_speed])

            # Calculate speed error
            speed_error = np.nanmean(
                np.abs(self.predicted_speeds[: frame + 1] - self.speeds[: frame + 1])
            )
            self.artists["speed_title"].set_text(f"Speed error: {speed_error:.2f} u.a.")

    def _update_linpos_movie(self, frame, start_idx, end_idx):
        """
        Update the linearized position movie panel.
        """
        time = self.positionTime[start_idx:end_idx]
        linpositions = self.linpositions[start_idx:end_idx]

        self.artists["linpos_line"].set_data(time, linpositions)
        self.artists["current_point"].set_offsets(
            np.array([self.positionTime[frame], self.linpositions[frame]])
        )

        if self.linpredicted is not None:
            linpredicted = self.linpredicted[start_idx:end_idx]
            prediction_time = self.positionTime[start_idx:end_idx]

            self.artists["linpos_pred_line"].set_data(prediction_time, linpredicted)
            self.artists["linpos_pred_points"].set_offsets(
                np.column_stack((prediction_time, linpredicted))
            )
            self.artists["current_predicted_point"].set_data(
                [[self.positionTime[frame]], [self.linpredicted[frame]]]
            )
            if hasattr(self, "lin_dim_pred"):
                values = self.lin_dim_pred[start_idx:end_idx]
                color = self.predicted_lin_cmap(self.predicted_lin_norm(values))
                self.artists["linpos_pred_points"].set_facecolors(color)
            if self.dim_name == "speed_mask":
                dim = (
                    self.dims["top"][start_idx:end_idx]
                    if "top" in self.dims
                    else self.dims[list(self.dims.keys())[0]][start_idx:end_idx]
                )
                colors = np.where(
                    dim,
                    self.delta_color_forward,
                    self.delta_color_reverse,
                )
                self.artists["linpos_pred_points"].set_color(colors)

        self.axes["linpos_movie"].set_xlim(
            self.positionTime[start_idx],
            max(
                self.positionTime[min(end_idx + 20, self.positionTime.size - 1)],
                self.positionTime[
                    min(
                        start_idx + self.lin_movie_duration // 2,
                        self.positionTime.size - 1,
                    )
                ],
            ),
        )

        # update stims, freezing, and ripples markers
        for name, indices in [
            ("stims_stars", self.stim_indices),
            ("freezing_stars", self.freezing_indices),
            ("ripples_stars", self.ripples_indices),
        ]:
            if name not in self.artists:
                continue
            try:
                mask = indices == frame
            except:
                continue

            if np.any(mask):
                if name != "freezing_stars":
                    x_data = self.positionTime[indices[mask]]
                    y_data = self.linpositions[indices[mask]]
                    try:
                        if (
                            self.linpoints[name] is None
                            or self.linpoints[name].size == 0
                        ):
                            self.linpoints[name] = np.column_stack(
                                (x_data, y_data)
                            ).reshape(-1, 2)
                            self.lintimes[name] = self.positionTime[frame]
                        else:
                            # ensure it doesnt exist already
                            if (
                                x_data in self.linpoints[name][:, 0]
                                and y_data in self.linpoints[name][:, 1]
                            ):
                                # skip if already exists
                                pass
                                # INFO: cant simplu do return because of the for loop in trajectory axes (see animate_frame)
                            else:
                                self.linpoints[name] = np.vstack(
                                    (
                                        self.linpoints[name].reshape(-1, 2),
                                        np.column_stack((x_data, y_data)).reshape(
                                            -1, 2
                                        ),
                                    )
                                )
                                self.lintimes[name] = np.hstack(
                                    (self.lintimes[name], self.positionTime[frame])
                                )
                    except (NameError, AttributeError):
                        self.linpoints[name] = np.column_stack(
                            (x_data, y_data)
                        ).reshape(-1, 2)
                        self.lintimes[name] = self.positionTime[frame]
                    self.artists[name].set_offsets(self.linpoints[name])
                else:
                    time = self.positionTime[frame]
                    # For freezing epochs, we need to find the row where time is in FreezeEpochs
                    xdata, ydata = [], []
                    for idx, (start, stop) in enumerate(self.FreezeEpochs):
                        if start > time:
                            continue
                        if time <= stop:  # current epoch
                            xdata += [start, time, np.nan]
                        else:
                            xdata += [start, stop, np.nan]

                        ydata += [
                            1.05,
                            1.05,
                            np.nan,
                        ]  # y position for the freeze epoch line
                        self.artists[name].set_xdata(xdata)
                        self.artists[name].set_ydata(ydata)

    def create_animation(
        self,
        interval: int = 50,
        repeat: bool = True,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Create and optionally save the animation.

        Args:
            interval: Time between frames in milliseconds (default: 50)
            repeat: Whether to repeat the animation (default: True)
            save_path: Path to save animation as MP4 (optional)
            **kwargs: Additional arguments for FuncAnimation

        Returns:
            matplotlib.animation.FuncAnimation object
        """
        if self.fig is None:
            self.setup_plot(**kwargs)

        # Default kwargs for better Qt compatibility
        anim_kwargs = {
            "blit": kwargs.get(
                "blit", True
            ),  # Better compatibility with Qt if set False
            "cache_frame_data": kwargs.get(
                "cache_frame_data", False
            ),  # Reduce memory usage
        }

        self.animation = animation.FuncAnimation(
            fig=self.fig,
            func=self.animate_frame,
            frames=self.total_frames,
            interval=interval,
            repeat=repeat,
            init_func=self.init_animation,
            **anim_kwargs,
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            try:
                if save_path.endswith(".gif"):
                    self.animation.save(
                        save_path, writer="pillow", fps=1000 // interval
                    )
                else:
                    self.animation.save(
                        save_path, writer="ffmpeg", fps=1000 // interval
                    )
                print("Animation saved!")
            except Exception as e:
                print(f"Failed to save animation: {e}")

        return self.animation

    def show(self, interval: int = 50, repeat: bool = True, block=False, **kwargs):
        """
        Show the animation with Qt backend support.

        Args:
            interval: Time between frames in milliseconds (default: 50)
            repeat: Whether to repeat the animation (default: True)
            block: Whether to block execution (auto-detected based on environment)
        """
        self.create_animation(interval=interval, repeat=repeat, **kwargs)

        # Auto-detect blocking behavior
        # Show the plot
        plt.show(block=block)

        return self.animation

    def deduplicate_and_merge(
        self, arr, unique_indices, inverse_indices, reduce_fn=np.mean
    ):
        """
        Deduplicate an array based on unique indices and inverse indices.

        Args:
            arr: The array to deduplicate
            unique_indices: Indices of unique elements
            inverse_indices: Indices to map back to original array

        Returns:
            deduplicated array
        """
        deduplicated = np.array(
            [
                reduce_fn(arr[inverse_indices == i], axis=0)
                for i in range(len(unique_indices))
            ]
        )
        return deduplicated


class ModelPerformanceVisualizer:
    """
    Comprehensive visualization tool for binary classification model performance over time.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ):
        """
        Initialize the performance visualizer.

        Args:
            predictions: Binary predictions (0/1) over time
            ground_truth: True labels (0/1) over time
            timestamps: Optional timestamps for x-axis (default: range(len(predictions)))
        """
        self.predictions = np.array(predictions).astype(int)
        self.ground_truth = np.array(ground_truth).astype(int)

        if len(self.predictions) != len(self.ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        self.timestamps = (
            timestamps if timestamps is not None else np.arange(len(predictions))
        )
        self.n_points = len(self.predictions)

        # Calculate basic metrics
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        self.accuracy = accuracy_score(self.ground_truth, self.predictions)
        self.precision = precision_score(
            self.ground_truth, self.predictions, zero_division=0
        )
        self.recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        self.f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)

        # Calculate error types
        self.correct = self.predictions == self.ground_truth
        self.false_positives = (self.predictions == 1) & (self.ground_truth == 0)
        self.false_negatives = (self.predictions == 0) & (self.ground_truth == 1)
        self.true_positives = (self.predictions == 1) & (self.ground_truth == 1)
        self.true_negatives = (self.predictions == 0) & (self.ground_truth == 0)

    def plot_timeline_comparison(
        self,
        figsize: Tuple[int, int] = (15, 8),
        title: str = "Model Predictions vs Ground Truth Over Time",
    ):
        """
        Plot predictions and ground truth over time with error highlighting.

        Args:
            figsize: Figure size
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 2, 1])

        # Plot 1: Stacked comparison
        ax1 = axes[0]

        # Plot ground truth
        ax1.fill_between(
            self.timestamps,
            0,
            self.ground_truth,
            alpha=0.3,
            color="green",
            label="Ground Truth",
            step="pre",
        )

        # Plot predictions with error highlighting
        colors = ["red" if not correct else "blue" for correct in self.correct]
        ax1.scatter(
            self.timestamps,
            self.predictions + 0.1,
            c=colors,
            alpha=0.7,
            s=20,
            label="Predictions",
        )

        ax1.set_ylabel("State")
        ax1.set_title(title)
        ax1.set_ylim(-0.1, 1.3)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error types
        ax2 = axes[1]

        # Create error type visualization
        y_pos = np.zeros(self.n_points)
        colors = []
        labels = []

        for i in range(self.n_points):
            if self.true_positives[i]:
                colors.append("darkgreen")
                y_pos[i] = 1
            elif self.true_negatives[i]:
                colors.append("lightgreen")
                y_pos[i] = 0
            elif self.false_positives[i]:
                colors.append("red")
                y_pos[i] = 0.5
            elif self.false_negatives[i]:
                colors.append("orange")
                y_pos[i] = 0.5

        ax2.scatter(self.timestamps, y_pos, c=colors, alpha=0.8, s=30)
        ax2.set_ylabel("Prediction Type")
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_yticklabels(["True Neg", "Error", "True Pos"])
        ax2.set_title("Prediction Types Over Time")
        ax2.grid(True, alpha=0.3)

        # Create custom legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="darkgreen",
                markersize=8,
                label="True Positive",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightgreen",
                markersize=8,
                label="True Negative",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="False Positive",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=8,
                label="False Negative",
            ),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        # Plot 3: Rolling accuracy
        ax3 = axes[2]
        window_size = max(10, self.n_points // 20)  # Adaptive window size
        rolling_accuracy = self._calculate_rolling_metric(self.correct, window_size)

        ax3.plot(
            self.timestamps,
            rolling_accuracy,
            "purple",
            linewidth=2,
            label=f"Rolling Accuracy (window={window_size})",
        )
        ax3.axhline(
            y=self.accuracy,
            color="red",
            linestyle="--",
            label=f"Overall Accuracy: {self.accuracy:.3f}",
        )
        ax3.set_ylabel("Accuracy")
        ax3.set_xlabel("Time")
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_confusion_matrix_over_time(
        self, window_size: int = 50, figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot confusion matrix metrics over time.

        Args:
            window_size: Window size for rolling calculations
            figsize: Figure size
        """
        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_confusion_metrics(window_size)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        metrics = ["accuracy", "precision", "recall", "f1"]
        colors = ["blue", "green", "red", "purple"]

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i // 2, i % 2]

            # Plot rolling metric
            ax.plot(
                self.timestamps,
                rolling_metrics[metric],
                color=color,
                linewidth=2,
                label=f"Rolling {metric.capitalize()}",
            )

            # Plot overall metric as horizontal line
            overall_value = getattr(self, metric)
            ax.axhline(
                y=overall_value,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"Overall: {overall_value:.3f}",
            )

            ax.set_title(f"{metric.capitalize()} Over Time")
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("Time")
        axes[1, 0].set_xlabel("Time")

        plt.tight_layout()
        return fig, axes

    def plot_error_heatmap(
        self, window_size: int = 20, figsize: Tuple[int, int] = (15, 6)
    ):
        """
        Create a heatmap showing error density over time.

        Args:
            window_size: Size of time windows for aggregation
            figsize: Figure size
        """
        # Create time windows
        n_windows = max(1, self.n_points // window_size)
        window_errors = np.zeros((4, n_windows))  # TP, TN, FP, FN
        window_labels = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, self.n_points)

            window_slice = slice(start_idx, end_idx)
            window_errors[0, i] = np.sum(self.true_positives[window_slice])
            window_errors[1, i] = np.sum(self.true_negatives[window_slice])
            window_errors[2, i] = np.sum(self.false_positives[window_slice])
            window_errors[3, i] = np.sum(self.false_negatives[window_slice])

            # Create window label
            start_time = self.timestamps[start_idx]
            end_time = self.timestamps[end_idx - 1]
            window_labels.append(f"{start_time:.1f}-{end_time:.1f}")

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize by window size for fair comparison
        window_errors_norm = window_errors / window_size

        im = ax.imshow(window_errors_norm, cmap="RdYlBu_r", aspect="auto")

        # Set labels
        ax.set_xticks(range(n_windows))
        ax.set_xticklabels(window_labels, rotation=45)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["True Pos", "True Neg", "False Pos", "False Neg"])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion of Predictions")

        # Add text annotations
        for i in range(4):
            for j in range(n_windows):
                text = ax.text(
                    j,
                    i,
                    f"{window_errors_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        ax.set_title("Prediction Types Heatmap Over Time")
        ax.set_xlabel("Time Windows")
        plt.tight_layout()
        return fig, ax

    def plot_precision_recall_timeline(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot precision-recall curve and timeline.

        Args:
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Precision-Recall Curve (if we have probability scores)
        # For binary predictions, we'll show the single point
        ax1.scatter(
            self.recall,
            self.precision,
            s=100,
            c="red",
            label=f"Model (F1={self.f1:.3f})",
        )
        ax1.plot([0, 1], [0.5, 0.5], "k--", alpha=0.5, label="Random")
        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.set_title("Precision-Recall Point")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Plot 2: Precision and Recall over time
        window_size = max(10, self.n_points // 20)
        rolling_precision = self._calculate_rolling_precision(window_size)
        rolling_recall = self._calculate_rolling_recall(window_size)

        ax2.plot(
            self.timestamps,
            rolling_precision,
            "green",
            linewidth=2,
            label=f"Rolling Precision (window={window_size})",
        )
        ax2.plot(
            self.timestamps,
            rolling_recall,
            "blue",
            linewidth=2,
            label=f"Rolling Recall (window={window_size})",
        )

        ax2.axhline(
            y=self.precision,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Overall Precision: {self.precision:.3f}",
        )
        ax2.axhline(
            y=self.recall,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Overall Recall: {self.recall:.3f}",
        )

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Score")
        ax2.set_title("Precision & Recall Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_comprehensive_dashboard(self, figsize: Tuple[int, int] = (20, 12)):
        """
        Create a comprehensive dashboard with all key visualizations.

        Args:
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)

        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 1, 1])

        # Main timeline plot
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_timeline(ax_main)

        # Rolling metrics
        ax_rolling = fig.add_subplot(gs[1, 0])
        self._plot_rolling_metrics(ax_rolling)

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[1, 1])
        self._plot_confusion_matrix(ax_cm)

        # Error distribution
        ax_errors = fig.add_subplot(gs[1, 2])
        self._plot_error_distribution(ax_errors)

        # Precision-Recall over time
        ax_pr = fig.add_subplot(gs[2, 0])
        self._plot_precision_recall_over_time(ax_pr)

        # Performance summary
        ax_summary = fig.add_subplot(gs[2, 1:])
        self._plot_performance_summary(ax_summary)

        plt.tight_layout()
        return fig

    def _calculate_rolling_metric(
        self, metric_array: np.ndarray, window_size: int
    ) -> np.ndarray:
        """Calculate rolling metric."""
        rolling_metric = np.zeros(self.n_points)
        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)
            rolling_metric[i] = np.mean(metric_array[start_idx:end_idx])
        return rolling_metric

    def _calculate_rolling_confusion_metrics(self, window_size: int) -> dict:
        """Calculate rolling confusion matrix metrics."""
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)

            window_pred = self.predictions[start_idx:end_idx]
            window_truth = self.ground_truth[start_idx:end_idx]

            if len(window_pred) > 0:
                metrics["accuracy"].append(accuracy_score(window_truth, window_pred))
                metrics["precision"].append(
                    precision_score(window_truth, window_pred, zero_division=0)
                )
                metrics["recall"].append(
                    recall_score(window_truth, window_pred, zero_division=0)
                )
                metrics["f1"].append(
                    f1_score(window_truth, window_pred, zero_division=0)
                )
            else:
                for key in metrics:
                    metrics[key].append(0)

        return {key: np.array(values) for key, values in metrics.items()}

    def _calculate_rolling_precision(self, window_size: int) -> np.ndarray:
        """Calculate rolling precision."""
        rolling_precision = np.zeros(self.n_points)
        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)

            window_pred = self.predictions[start_idx:end_idx]
            window_truth = self.ground_truth[start_idx:end_idx]
            rolling_precision[i] = precision_score(
                window_truth, window_pred, zero_division=0
            )

        return rolling_precision

    def _calculate_rolling_recall(self, window_size: int) -> np.ndarray:
        """Calculate rolling recall."""
        rolling_recall = np.zeros(self.n_points)
        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)

            window_pred = self.predictions[start_idx:end_idx]
            window_truth = self.ground_truth[start_idx:end_idx]
            rolling_recall[i] = recall_score(window_truth, window_pred, zero_division=0)

        return rolling_recall

    def _plot_main_timeline(self, ax):
        """Plot main timeline comparison."""
        # Ground truth as filled area
        ax.fill_between(
            self.timestamps,
            0,
            self.ground_truth,
            alpha=0.3,
            color="green",
            label="Ground Truth",
            step="pre",
        )

        # Predictions with error coloring
        correct_mask = self.correct
        ax.scatter(
            self.timestamps[correct_mask],
            self.predictions[correct_mask] + 0.1,
            c="blue",
            alpha=0.7,
            s=15,
            label="Correct",
        )
        ax.scatter(
            self.timestamps[~correct_mask],
            self.predictions[~correct_mask] + 0.1,
            c="red",
            alpha=0.7,
            s=15,
            label="Error",
        )

        ax.set_title("Model Predictions vs Ground Truth")
        ax.set_ylabel("State")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_rolling_metrics(self, ax):
        """Plot rolling accuracy and F1."""
        window_size = max(10, self.n_points // 20)
        rolling_acc = self._calculate_rolling_metric(self.correct, window_size)
        rolling_f1 = self._calculate_rolling_confusion_metrics(window_size)["f1"]

        ax.plot(self.timestamps, rolling_acc, "blue", label="Accuracy")
        ax.plot(self.timestamps, rolling_f1, "red", label="F1")
        ax.set_title("Rolling Metrics")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix(self, ax):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.ground_truth, self.predictions)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title("Confusion Matrix")

    def _plot_error_distribution(self, ax):
        """Plot error type distribution."""
        error_counts = [
            np.sum(self.true_negatives),
            np.sum(self.false_positives),
            np.sum(self.false_negatives),
            np.sum(self.true_positives),
        ]
        labels = ["TN", "FP", "FN", "TP"]
        colors = ["lightgreen", "red", "orange", "darkgreen"]

        ax.pie(error_counts, labels=labels, colors=colors, autopct="%1.1f%%")
        ax.set_title("Prediction Distribution")

    def _plot_precision_recall_over_time(self, ax):
        """Plot precision and recall over time."""
        window_size = max(10, self.n_points // 20)
        rolling_precision = self._calculate_rolling_precision(window_size)
        rolling_recall = self._calculate_rolling_recall(window_size)

        ax.plot(self.timestamps, rolling_precision, "green", label="Precision")
        ax.plot(self.timestamps, rolling_recall, "blue", label="Recall")
        ax.set_title("Precision & Recall")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_summary(self, ax):
        """Plot performance summary table."""
        ax.axis("off")

        # Create summary table
        metrics_data = [
            ["Accuracy", f"{self.accuracy:.3f}"],
            ["Precision", f"{self.precision:.3f}"],
            ["Recall", f"{self.recall:.3f}"],
            ["F1-Score", f"{self.f1:.3f}"],
            ["True Positives", f"{np.sum(self.true_positives)}"],
            ["True Negatives", f"{np.sum(self.true_negatives)}"],
            ["False Positives", f"{np.sum(self.false_positives)}"],
            ["False Negatives", f"{np.sum(self.false_negatives)}"],
        ]

        table = ax.table(
            cellText=metrics_data,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        ax.set_title("Performance Summary", pad=20)

    def print_summary(self):
        """Print performance summary."""
        print("Model Performance Summary")
        print("=" * 30)
        print(f"Overall Accuracy: {self.accuracy:.3f}")
        print(f"Precision: {self.precision:.3f}")
        print(f"Recall: {self.recall:.3f}")
        print(f"F1-Score: {self.f1:.3f}")
        print("\nError Breakdown:")
        print(f"True Positives: {np.sum(self.true_positives)}")
        print(f"True Negatives: {np.sum(self.true_negatives)}")
        print(f"False Positives: {np.sum(self.false_positives)}")
        print(f"False Negatives: {np.sum(self.false_negatives)}")


class OversamplingVisualizer:
    """Visualize the effect of oversampling resampling on spatial distribution"""

    def __init__(self, gaussian_heatmap_model):
        self.GaussianHeatmap = gaussian_heatmap_model
        self.GRID_H = gaussian_heatmap_model.GRID_H
        self.GRID_W = gaussian_heatmap_model.GRID_W

    def extract_positions_from_dataset(self, dataset, max_samples=None):
        """Extract positions from a tf.data.Dataset"""
        positions = []
        count = 0

        for example in dataset:
            if max_samples and count >= max_samples:
                break

            # Assuming 'pos' is the key for positions in your dataset dict
            if isinstance(example, dict):
                pos = example["pos"].numpy()
            else:
                # If dataset yields just positions
                pos = example.numpy()

            positions.append(pos)
            count += 1

        return np.array(positions)

    def positions_to_coarse_bins(self, positions, stride=3):
        """Convert positions to coarse bin indices (matching your oversampling logic)"""
        # Convert positions to fine bins first
        bins = self.GaussianHeatmap.positions_to_bins(positions)

        # Convert to coarse bins (same logic as your code)
        x_fine = bins % self.GRID_W
        y_fine = bins // self.GRID_W
        x_coarse = x_fine // stride
        y_coarse = y_fine // stride

        coarse_H, coarse_W = self.GRID_H // stride, self.GRID_W // stride
        coarse_bins = y_coarse * coarse_W + x_coarse

        return coarse_bins, coarse_H, coarse_W

    def create_distribution_heatmap(self, positions, stride=3, title="Distribution"):
        """Create a 2D heatmap showing spatial distribution of positions"""
        coarse_bins, coarse_H, coarse_W = self.positions_to_coarse_bins(
            positions, stride
        )

        # Count samples per bin
        counts = np.bincount(coarse_bins, minlength=coarse_H * coarse_W)
        distribution = counts.reshape(coarse_H, coarse_W)

        return distribution

    def visualize_oversampling_effect(
        self,
        dataset_before,
        dataset_after,
        stride=3,
        max_samples=10000,
        figsize=(15, 5),
        path=None,
    ):
        """
        Main visualization function comparing before/after oversampling

        Args:
            dataset_before: Dataset before oversampling
            dataset_after: Dataset after oversampling
            stride: Coarse grid stride (should match your oversampling)
            max_samples: Max samples to analyze (for performance)
            figsize: Figure size
        """

        print("Extracting positions from datasets...")

        # Extract positions
        pos_before = self.extract_positions_from_dataset(dataset_before, max_samples)
        pos_after = self.extract_positions_from_dataset(dataset_after, max_samples)

        print(f"Before oversampling: {len(pos_before)} samples")
        print(f"After oversampling: {len(pos_after)} samples")
        print(f"Oversampling ratio: {len(pos_after) / len(pos_before):.2f}x")

        # Create distribution heatmaps
        dist_before = self.create_distribution_heatmap(pos_before, stride)
        dist_after = self.create_distribution_heatmap(pos_after, stride)

        # Calculate difference
        # Normalize by total samples for fair comparison
        dist_before_norm = dist_before / dist_before.sum()
        dist_after_norm = dist_after / dist_after.sum()
        dist_diff = dist_after_norm - dist_before_norm

        # Create forbidden mask for coarse grid
        coarse_H, coarse_W = self.GRID_H // stride, self.GRID_W // stride
        FORBID_coarse = np.zeros((coarse_H, coarse_W), dtype=bool)
        for y in range(coarse_H):
            for x in range(coarse_W):
                if np.any(
                    self.GaussianHeatmap.forbid_mask_tf[
                        y * stride : (y + 1) * stride,
                        x * stride : (x + 1) * stride,
                    ]
                    > 0
                ):
                    FORBID_coarse[y, x] = True

        # Create the visualization
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Raw counts
        im1 = axes[0, 0].imshow(dist_before, cmap="Blues", origin="lower")
        axes[0, 0].set_title(f"Before Oversampling\n({len(pos_before)} samples)")
        axes[0, 0].contour(FORBID_coarse, levels=[0.5], colors="red", linewidths=2)
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(dist_after, cmap="Blues", origin="lower")
        axes[0, 1].set_title(f"After Oversampling\n({len(pos_after)} samples)")
        axes[0, 1].contour(FORBID_coarse, levels=[0.5], colors="red", linewidths=2)
        plt.colorbar(im2, ax=axes[0, 1])

        # Difference in raw counts
        im3 = axes[0, 2].imshow(dist_after - dist_before, cmap="RdBu_r", origin="lower")
        axes[0, 2].set_title("Difference (Raw Counts)")
        axes[0, 2].contour(FORBID_coarse, levels=[0.5], colors="black", linewidths=2)
        plt.colorbar(im3, ax=axes[0, 2])

        # Normalized distributions
        im4 = axes[1, 0].imshow(dist_before_norm, cmap="Blues", origin="lower")
        axes[1, 0].set_title("Before (Normalized)")
        axes[1, 0].contour(FORBID_coarse, levels=[0.5], colors="red", linewidths=2)
        plt.colorbar(im4, ax=axes[1, 0])

        im5 = axes[1, 1].imshow(dist_after_norm, cmap="Blues", origin="lower")
        axes[1, 1].set_title("After (Normalized)")
        axes[1, 1].contour(FORBID_coarse, levels=[0.5], colors="red", linewidths=2)
        plt.colorbar(im5, ax=axes[1, 1])

        # Normalized difference
        vmax = np.abs(dist_diff).max()
        im6 = axes[1, 2].imshow(
            dist_diff, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax
        )
        axes[1, 2].set_title("Difference (Normalized)")
        axes[1, 2].contour(FORBID_coarse, levels=[0.5], colors="black", linewidths=2)
        plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()
        if path is not None:
            fig.savefig(path)
            print(f"Saved figure to {path}")
        plt.close(fig)

        # Print statistics
        self.print_distribution_stats(dist_before_norm, dist_after_norm, FORBID_coarse)

        return dist_before, dist_after, dist_diff

    def print_distribution_stats(self, dist_before, dist_after, forbid_mask):
        """Print statistical comparison of distributions"""

        # Only consider allowed bins
        allowed_mask = ~forbid_mask

        before_allowed = dist_before[allowed_mask]
        after_allowed = dist_after[allowed_mask]

        print("\n=== Distribution Statistics ===")
        print(f"Allowed bins: {allowed_mask.sum()} / {forbid_mask.size}")
        print(f"Forbidden bins: {forbid_mask.sum()} / {forbid_mask.size}")

        print("\nBefore oversampling:")
        print(f"  Min density: {before_allowed.min():.6f}")
        print(f"  Max density: {before_allowed.max():.6f}")
        print(f"  Mean density: {before_allowed.mean():.6f}")
        print(f"  Std density: {before_allowed.std():.6f}")
        print(f"  CV (std/mean): {before_allowed.std() / before_allowed.mean():.4f}")

        print("\nAfter oversampling:")
        print(f"  Min density: {after_allowed.min():.6f}")
        print(f"  Max density: {after_allowed.max():.6f}")
        print(f"  Mean density: {after_allowed.mean():.6f}")
        print(f"  Std density: {after_allowed.std():.6f}")
        print(f"  CV (std/mean): {after_allowed.std() / after_allowed.mean():.4f}")

        # Uniformity metrics
        uniform_target = 1.0 / allowed_mask.sum()  # Perfect uniform density

        kl_before = self.kl_divergence_to_uniform(before_allowed, uniform_target)
        kl_after = self.kl_divergence_to_uniform(after_allowed, uniform_target)

        print("\nUniformity (KL divergence from uniform):")
        print(f"  Before: {kl_before:.6f}")
        print(f"  After: {kl_after:.6f}")
        print(
            f"  Improvement: {kl_before - kl_after:.6f} {'✓' if kl_after < kl_before else '✗'}"
        )

    def kl_divergence_to_uniform(self, distribution, uniform_target):
        """Compute KL divergence from distribution to uniform"""
        eps = 1e-8
        distribution = np.clip(distribution, eps, 1.0)
        return np.sum(distribution * np.log(distribution / (uniform_target + eps)))

    def plot_histogram_comparison(
        self, dataset_before, dataset_after, stride=3, max_samples=10000
    ):
        """Plot histograms of bin occupancy"""

        pos_before = self.extract_positions_from_dataset(dataset_before, max_samples)
        pos_after = self.extract_positions_from_dataset(dataset_after, max_samples)

        coarse_bins_before, coarse_H, coarse_W = self.positions_to_coarse_bins(
            pos_before, stride
        )
        coarse_bins_after, _, _ = self.positions_to_coarse_bins(pos_after, stride)

        counts_before = np.bincount(coarse_bins_before, minlength=coarse_H * coarse_W)
        counts_after = np.bincount(coarse_bins_after, minlength=coarse_H * coarse_W)

        # Filter out forbidden bins
        FORBID_coarse = np.zeros((coarse_H, coarse_W), dtype=bool)
        for y in range(coarse_H):
            for x in range(coarse_W):
                if np.any(
                    self.GaussianHeatmap.forbid_mask_tf[
                        y * stride : (y + 1) * stride,
                        x * stride : (x + 1) * stride,
                    ]
                    > 0
                ):
                    FORBID_coarse[y, x] = True

        allowed_mask = ~FORBID_coarse.flatten()
        counts_before_allowed = counts_before[allowed_mask]
        counts_after_allowed = counts_after[allowed_mask]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(
            counts_before_allowed, bins=30, alpha=0.7, label="Before", color="blue"
        )
        ax1.hist(counts_after_allowed, bins=30, alpha=0.7, label="After", color="red")
        ax1.set_xlabel("Samples per bin")
        ax1.set_ylabel("Number of bins")
        ax1.set_title("Histogram of Bin Occupancy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        data_to_plot = [counts_before_allowed, counts_after_allowed]
        ax2.boxplot(data_to_plot, labels=["Before", "After"])
        ax2.set_ylabel("Samples per bin")
        ax2.set_title("Box Plot Comparison")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Usage example:
"""
# Initialize the visualizer
visualizer = OversamplingVisualizer(your_gaussian_heatmap_model)

# Compare before and after oversampling
# You'll need to create/save your datasets before and after the oversampling step
dist_before, dist_after, diff = visualizer.visualize_oversampling_effect(
    dataset_before=train_dataset_before_oversampling,
    dataset_after=train_dataset_after_oversampling,
    stride=3,  # Should match your oversampling stride
    max_samples=10000
)

# Additional histogram comparison
visualizer.plot_histogram_comparison(
    dataset_before_oversampling,
    dataset_after_oversampling,
    stride=3
)
"""


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    crgb = np.array(list(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])))
    crgb[crgb < 0] = 0  # Ensure no negative values
    crgb[crgb > 1] = 1  # Ensure no values greater than 1
    crgb = tuple(crgb)  # Convert to tuple for consistency
    return crgb


# Helper function for circular statistics
def circular_mean_error(pred_angles, true_angles):
    """Calculate circular mean error in radians"""
    diff = pred_angles - true_angles
    # Wrap to [-pi, pi]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff


def plot_circular_comparison(ax, pred_angles, true_angles, title="Head Direction"):
    """Plot circular comparison of predicted vs true angles"""
    # Create unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", alpha=0.3, linewidth=0.5)

    # Plot predictions and true values
    ax.scatter(
        np.cos(pred_angles),
        np.sin(pred_angles),
        c="red",
        alpha=0.6,
        s=20,
        label="Predicted",
    )
    ax.scatter(
        np.cos(true_angles),
        np.sin(true_angles),
        c="blue",
        alpha=0.6,
        s=20,
        label="True",
    )

    # Add arrows from true to predicted
    for i in range(min(50, len(pred_angles))):  # Limit to 50 arrows for clarity
        ax.annotate(
            "",
            xy=(np.cos(pred_angles[i]), np.sin(pred_angles[i])),
            xytext=(np.cos(true_angles[i]), np.sin(true_angles[i])),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.5),
        )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()

    # Add cardinal direction labels
    ax.text(1.05, 0, "0°", ha="left", va="center")
    ax.text(0, 1.05, "90°", ha="center", va="bottom")
    ax.text(-1.05, 0, "180°", ha="right", va="center")
    ax.text(0, -1.05, "270°", ha="center", va="top")


def create_polar_colorbar(
    fig, mappable, ax_pos, angle_range=(-np.pi, np.pi), title="Head\nDirection"
):
    """
    Create a circular colorbar showing head direction angles that matches arctan2 output

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to add the colorbar to
    mappable : matplotlib.cm.ScalarMappable
        The scatter plot or other mappable object (used for color mapping)
    ax_pos : matplotlib.transforms.Bbox
        Position of the main axis (use ax.get_position())
    angle_range : tuple
        Range of angles (min, max) - use (-π, π) for arctan2 output or (0, 2π) for wrapped

    Returns:
    --------
    cax : matplotlib.axes.Axes
        The colorbar axis object
    """
    # Create small circular axis for colorbar
    cax = fig.add_axes(
        [
            ax_pos.x0 + ax_pos.width + 0.02,
            ax_pos.y0 + ax_pos.height * 0.6,
            0.1,
            0.1,
        ]
    )
    cax.set_xlim(-1.1, 1.1)
    cax.set_ylim(-1.1, 1.1)
    cax.set_aspect("equal")

    # Create angles that match the HSV colormap mapping
    # HSV maps [0, 1] to full color wheel, so we need to map our angle range to [0, 1]
    n_segments = 256

    if angle_range == (-np.pi, np.pi):
        # For arctan2 output range [-π, π]
        # Map to HSV: -π -> 0.5 (cyan), 0 -> 0 (red), π -> 0.5 (cyan)
        # But we want: -π -> 0.5, 0 -> 0, π -> 1.0
        angles = np.linspace(-np.pi, np.pi, n_segments)
        hsv_values = (angles + np.pi) / (2 * np.pi)  # Map [-π, π] to [0, 1]
    else:
        # For range [0, 2π]
        angles = np.linspace(0, 2 * np.pi, n_segments)
        hsv_values = angles / (2 * np.pi)

    colors = plt.cm.hsv(hsv_values)

    for i, (angle, color) in enumerate(zip(angles[:-1], colors[:-1])):
        # Draw small arc segments - note: we plot at the actual mathematical angle
        arc_angles = np.linspace(angle, angles[i + 1], 10)
        x_inner = 0.7 * np.cos(arc_angles)
        y_inner = 0.7 * np.sin(arc_angles)
        x_outer = 1.0 * np.cos(arc_angles)
        y_outer = 1.0 * np.sin(arc_angles)

        # Create polygon for arc segment
        x_poly = np.concatenate([x_inner, x_outer[::-1]])
        y_poly = np.concatenate([y_inner, y_outer[::-1]])
        cax.fill(x_poly, y_poly, color=color, edgecolor="none")

    # Add angle labels at correct mathematical positions
    if angle_range == (-np.pi, np.pi):
        label_angles = [0, np.pi / 2, np.pi, -np.pi / 2]  # Right, Up, Left, Down
        label_texts = ["0°", "90°", "±180°", "-90°"]
    else:
        label_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        label_texts = ["0°", "90°", "180°", "270°"]

    for angle, text in zip(label_angles, label_texts):
        x_label = 1.2 * np.cos(angle)
        y_label = 1.2 * np.sin(angle)
        cax.text(
            x_label,
            y_label,
            text,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Add directional arrows
    arrow_length = 0.15
    for angle in label_angles:
        x_start = 0.85 * np.cos(angle)
        y_start = 0.85 * np.sin(angle)
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        cax.arrow(
            x_start,
            y_start,
            dx,
            dy,
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
        )

    cax.set_xticks([])
    cax.set_yticks([])
    cax.set_title(title, fontsize=11)

    # Remove axes spines
    for spine in cax.spines.values():
        spine.set_visible(False)

    return cax


if __name__ == "__main__":
    # Run demonstration
    try:
        print("=" * 60)
        print("ANIMATED POSITION PLOTTER DEMO")
        print("=" * 60)
        print(f"Backend in use: {matplotlib.get_backend()}")

        plotter, anim = demo_animated_plot()
        print("\nAnimation created successfully!")

        # Keep the plot alive for Qt backends
        backend = matplotlib.get_backend()
        if "Qt" in backend:
            print("\nQt backend detected - plot window should stay interactive")
            print("Close the plot window to continue...")
            try:
                # For Qt backends, ensure event loop runs
                if hasattr(plotter.fig.canvas, "start_main_loop"):
                    plotter.fig.canvas.start_main_loop()
            except:
                pass

    except Exception as e:
        print(f"Error running demonstration: {e}")
        import traceback

        traceback.print_exc()

    # Show usage examples
    print("\n" + "=" * 60)
    print("USAGE WITH YOUR DATA:")
    print("=" * 60)
    print(f"Current backend: {matplotlib.get_backend()}")
    print("""
# Plot your MATLAB maze shape:
maze_coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0],
               [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]

plotter = AnimatedPositionPlotter(your_data_helper)
plotter.setup_plot(
    custom_lines=[maze_coords],     # Your maze shape
    custom_line_colors='black',     # Maze color
    custom_line_styles='-',         # Solid lines
    custom_line_widths=3            # Thick walls
)
plotter.show()

# Multiple custom shapes:
maze = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0], [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]
shock_zone = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]]  # Square shock zone

plotter.setup_plot(
    custom_lines=[maze, shock_zone],
    custom_line_colors=['black', 'red'],    # Different colors
    custom_line_styles=['-', '--'],         # Different styles
    custom_line_widths=[3, 2]               # Different widths
)

# Alternative: Use helper function
maze_coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0],
               [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]
custom_lines = create_maze_from_matlab(maze_coords)

plotter = create_plotter_for_data(
    your_data_helper,
    custom_lines=custom_lines
)

# Mix with other reference lines:
plotter.setup_plot(
    binary_colors=True,
    custom_lines=[maze_coords],          # Maze walls
    hlines=[0.5],                        # Center horizontal line
    vlines=[0.5],                        # Center vertical line
    custom_line_colors='black',          # Maze color
    line_colors='gray'                   # Reference line color
)

# Force Qt backend:
plotter = create_qt_plotter(your_data_helper, trail_length=40)

# In Jupyter notebook:
%matplotlib widget  # or %matplotlib qt
plotter = AnimatedPositionPlotter(your_data_helper)
plotter.show()

# Save as video:
plotter = AnimatedPositionPlotter(your_data_helper)
anim = plotter.create_animation(save_path='trajectory.mp4')

# Manual backend control:
import matplotlib
matplotlib.use('Qt5Agg')  # Before importing pyplot
plotter = AnimatedPositionPlotter(your_data_helper)
plotter.show()
""")

    print("\nAvailable backends on your system:")
    try:
        import matplotlib.backend_bases

        backends = []
        for backend in ["Qt5Agg", "Qt4Agg", "TkAgg", "GTK3Agg", "WXAgg"]:
            try:
                matplotlib.use(backend, force=False)
                backends.append(backend)
            except:
                pass
        print(f"Compatible backends: {backends}")
    except:
        print("Could not detect available backends")
