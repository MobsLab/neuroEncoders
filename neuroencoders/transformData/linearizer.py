## Created by Pierre 01/04/2021

import os
from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itp
import tables
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykeops import set_verbose as pykeopsset_verbose
from pykeops.numpy import LazyTensor as LazyTensor_np

from neuroencoders.utils.global_classes import MAZE_COORDS

pykeopsset_verbose(False)  # Disable verbose output from PyKeOps


class UMazeLinearizer:
    """
    A class to define a linearization function of the data.
    Depending on the maze shape, user might want to change this class
    to fit to their maze shape.

    args:
    folder: str, the folder where the linearization points are saved
    nb_bins: int, the number of bins to use for the linearization, defaults to 100
    n_path_points: int, the number of points to use for the interpolated path, defaults to 25
    phase: str, the phase of the experiment, defaults to None
    data_helper: object, a helper object to provide custom lines (like maze walls, boundaries, etc.), defaults to None
    redo: bool, if True, regenerate the linearization points, defaults to False
    """

    def __init__(self, *args, **kwargs):
        # Extract folder/path parameter (can be called either way)
        if len(args) >= 1:
            folder = args[0]
            args = args[1:]
        else:
            folder = kwargs.pop("folder", None)
            if folder is None:
                folder = kwargs.pop("path", None)

        if folder is None:
            raise ValueError("folder (or path) parameter is required")

        # Extract nb_bins parameter
        if len(args) >= 1:
            nb_bins = args[0]
            args = args[1:]
        else:
            nb_bins = kwargs.pop("nb_bins", 100)

        # Extract phase parameter
        if len(args) >= 1:
            phase = args[0]
            args = args[1:]
        else:
            phase = kwargs.pop("phase", None)

        # Initialize attributes
        self.folder = folder
        self.nb_bins = nb_bins
        self.n_path_points = kwargs.pop("n_path_points", 25)
        self.phase = phase

        # Initialize linearization points
        filename = os.path.join(folder, "nnBehavior.mat")
        if not os.path.exists(filename):
            raise ValueError("this file does not exist :" + folder + "nnBehavior.mat")
        if phase is not None:
            filename = os.path.join(folder, "nnBehavior_" + phase + ".mat")
            if not os.path.exists(filename):
                assert tables.is_hdf5_file(folder + "nnBehavior.mat")
                import shutil

                warn("weird to copy that file now")

                shutil.copyfile(
                    folder + "nnBehavior.mat",
                    folder + "nnBehavior_" + phase + ".mat",
                    follow_symlinks=True,
                )
        self.filename = filename
        # Extract basic behavior
        with tables.open_file(filename, "a") as f:
            children = [c.name for c in f.list_nodes("/behavior")]
            if (
                "linearizationPoints" in children
                and "targetLinearValues" in children
                and kwargs.get("nnPoints", None) is None
                and kwargs.get("redo", False) is False
            ):
                self.nnPoints = f.root.behavior.linearizationPoints[:]
                self.target_linear_values = f.root.behavior.targetLinearValues[:]
            elif (
                kwargs.get("nnPoints", None) is not None
                and kwargs.get("redo", False) is False
            ):
                self.nnPoints = kwargs["nnPoints"]
            else:
                self._generate_canonical_path()

            if "aligned_ref" in children:
                self.aligned_ref = f.root.behavior.aligned_ref[:]
            else:
                self.aligned_ref = None

        self._create_interpolation()

        self.data_helper = kwargs.pop("data_helper", None)
        custom_lines = kwargs.pop("custom_lines", None)
        if self.data_helper is not None:
            print("found data_helper, using its custom lines")
            self.custom_lines = custom_lines or [
                self.data_helper.maze_coords,
                self.data_helper.shock_zone,
                self.data_helper.safe_zone,
            ]
        else:
            self.custom_lines = None
        self.custom_line_colors = ["black", "hotpink", "cornflowerblue"]
        self.custom_line_styles = ["-", "-", "-"]
        self.custom_line_widths = [4, 2, 2]

    def _generate_canonical_path(self):
        """Generate the canonical U-shaped path through the maze center."""
        # old waypoints looked like this:
        # [     [0.15, 0],     [0.15, 0.1],     [0.15, 0.2],     [0.15, 0.3],     [0.15, 0.4],     [0.15, 0.5],     [0.15, 0.6],     [0.15, 0.7],     [0.15, 0.8],     [0.15, 0.9],     [0.25, 0.9],     [0.35, 0.9],     [0.5, 0.9],     [0.65, 0.9],     [0.75, 0.9],     [0.85, 0.9],     [0.85, 0.8],     [0.85, 0.7],     [0.85, 0.6],     [0.85, 0.5],     [0.85, 0.4],     [0.85, 0.3],     [0.85, 0.2],     [0.85, 0.1],     [0.85, 0], ]

        # Outer boundaries
        # TODO: change with custom maze coordinates ?
        # WARNING: for now i don't use it as the data_helper.maze_coords are not super trustworthy # Theotime 31.07.2025
        self.x_min = MAZE_COORDS[:, 0].min()
        self.x_max = MAZE_COORDS[:, 0].max()
        self.y_min = MAZE_COORDS[:, 1].min()
        self.y_max = MAZE_COORDS[:, 1].max()

        # Inner corridor boundaries (assuming indices 4-7 are inner boundaries)
        inner_coords = MAZE_COORDS[4:8]
        self.inner_x_min = inner_coords[:, 0].min()
        self.inner_x_max = inner_coords[:, 0].max()
        self.inner_y_max = inner_coords[:, 1].max()

        # Calculate corridor width
        self.arm_width = self.inner_x_min - self.x_min
        self.top_width = self.y_max - self.inner_y_max

        # Path goes through the center of the arms
        left_center_x = self.x_min + self.arm_width / 2
        right_center_x = self.x_max - self.arm_width / 2
        top_center_y = self.inner_y_max + self.top_width / 2
        center_x = 0.5

        # Define key waypoints with their target linearization values
        waypoints = [
            # Left arm (bottom to top): 0.0 to 0.3
            (left_center_x, self.y_min, 0.0),  # Bottom left
            (left_center_x, top_center_y, 0.3),  # Top left corner
            # Center point
            (center_x, top_center_y, 0.5),  # Center between arms
            # Top connection: 0.3 to 0.7
            (right_center_x, top_center_y, 0.7),  # Top right corner
            # Right arm (top to bottom): 0.7 to 1.0
            (right_center_x, self.y_min, 1.0),  # Bottom right
        ]

        # Generate intermediate points between waypoints
        nnPoints = []
        target_values = []

        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]

            # Number of points for this segment (proportional to linearization distance)
            segment_length = end_point[2] - start_point[2]
            n_segment_points = max(2, int(segment_length * self.n_path_points))

            # Generate points along this segment
            for j in range(n_segment_points):
                if i < len(waypoints) - 2 and j == n_segment_points - 1:
                    continue  # Skip last point to avoid duplication

                t = j / (n_segment_points - 1)
                x = start_point[0] + t * (end_point[0] - start_point[0])
                y = start_point[1] + t * (end_point[1] - start_point[1])
                linear_val = start_point[2] + t * (end_point[2] - start_point[2])

                nnPoints.append([x, y])
                target_values.append(linear_val)

        self.og_nnPoints = np.array(nnPoints)
        self.nnPoints = np.array(nnPoints)
        self.og_target_linear_values = np.array(target_values)
        self.target_linear_values = np.array(target_values)

    def _create_interpolation(self, clip=True):
        """Create interpolation objects for the path."""
        # Create parameter values for spline interpolation
        self.n_points = len(self.nnPoints)
        ts = np.linspace(0, 1, self.n_points)
        # equally spaced linear points. As many as the number of points
        # pu in the verify_linearization function (by default 25 anchor points)

        # Create spline interpolation for the 2D path
        self.path_spline = itp.make_interp_spline(ts, self.nnPoints, k=2)
        # path_spline is the interpolating object that finds a fit between
        # the anchor points and the equally spaced 2D points

        # Create spline interpolation for the linearization values
        self.linear_spline = itp.make_interp_spline(ts, self.target_linear_values, k=2)
        # linear_spline is the interpolating object that finds a fit between
        # the anchor points and the equally spaced linear values

        # Generate final discretized path
        self.tsProj = np.linspace(0, 1, self.nb_bins)
        self.mazePoints = self.path_spline(self.tsProj)
        self.linear_values = self.linear_spline(self.tsProj)

        # Ensure linear values are exactly 0 and 1 at endpoints
        if clip:
            self.linear_values[0] = 0.0
            self.linear_values[-1] = 1.0

    def apply_linearization(self, euclideanData, keops=True):
        """
        Project 2D points onto the linearized maze path.

        Args:
            euclideanData: np.array of shape (N, 2) with 2D coordinates
            keops: bool, if True, use PyKeOps for efficient computation, defaults to True

        Returns:
            projectedPos: np.array of shape (N, 2) with projected coordinates
            linearPos: np.array of shape (N,) with linearized positions
        """
        if keops:
            return self.pykeops_linearization(euclideanData)
        else:
            projectedPos = np.full([euclideanData.shape[0], 2], np.nan)
            linearFeature = np.full([euclideanData.shape[0]], np.nan)
            for idp in range(euclideanData.shape[0]):
                point = euclideanData[idp, :]
                if np.any(np.isnan(point)):
                    continue
                bestPoint = np.argmin(
                    np.sum(
                        np.square(
                            np.reshape(point, [1, euclideanData.shape[1]])
                            - self.mazePoints
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                projectedPos[idp, :] = self.mazePoints[bestPoint, :]
                linearFeature[idp] = self.linear_values[bestPoint]

            return projectedPos, linearFeature

    def pykeops_linearization(self, euclideanData):
        """
        Project 2D points onto the linearized maze path.

        Args:
            euclideanData: np.array of shape (N, 2) with 2D coordinates

        Returns:
            projectedPos: np.array of shape (N, 2) with projected coordinates
            linearPos: np.array of shape (N,) with linearized positions
        """
        if euclideanData.dtype != self.mazePoints.dtype:
            euclideanData = euclideanData.astype(self.mazePoints.dtype)

        N = euclideanData.shape[0]

        # prefill with nan
        projectedPos = np.full([N, 2], np.nan, dtype=self.mazePoints.dtype)
        linearPos = np.full([N], np.nan, dtype=self.mazePoints.dtype)
        valid_mask = np.logical_not(np.any(np.isnan(euclideanData), axis=1))
        valid_indices = np.where(valid_mask)[0]

        if valid_indices.size > 0:
            valid_points = euclideanData[valid_mask]
            euclidData_lazy = LazyTensor_np(valid_points[None, :, :])
            mazePoint_lazy = LazyTensor_np(self.mazePoints[:, None, :])

            distance_matrix_lazy = (
                (mazePoint_lazy - euclidData_lazy).square().sum(axis=-1)
            )
            # find the argmin
            bestPoints = distance_matrix_lazy.argmin_reduction(axis=0)
            projectedPos[valid_indices, :] = self.mazePoints[bestPoints[:, 0], :]
            linearPos[valid_indices] = self.linear_values[bestPoints[:, 0]]

        return projectedPos, linearPos

    def verify_linearization(
        self,
        ExampleEuclideanData,
        folder,
        overwrite=False,
        training=False,
        nnPoints=None,
    ):
        """
        A function to verify and possibly change the linearization.
        This function will plot the data and allow the user to change the linearization points. The new linearization points will be saved in the folder.

        args:
        ExampleEuclideanData: np.array, the data to be linearized
        folder: str, the folder where the linearization points are saved
        overwrite: bool, if True, the linearization points will be overwritten, defaults to False
        """

        if nnPoints is not None:
            self.nnPoints = nnPoints

        def try_linearization(ax, l0s):
            _, linearTrue = self.apply_linearization(euclidData)
            binIndex = [
                np.where((linearTrue >= projBin[id]) * (linearTrue < projBin[id + 1]))[
                    0
                ]
                for id in range(len(projBin) - 1)
            ]
            cm = plt.get_cmap("tab20")
            for tpl in enumerate(binIndex):
                id, bId = tpl
                try:
                    l0s[id].remove()
                except:
                    None
                l0s[id] = ax[0].scatter(
                    euclidData[bId, 0], euclidData[bId, 1], c=[cm(id)]
                )
            return l0s

        def b1update(n):
            """
            Reset the linearization points to the original ones.
            """
            self.nnPoints = self.og_nnPoints.tolist()
            self.target_linear_values = self.og_target_linear_values.tolist()
            # create the interpolating object
            self.n_points = len(self.nnPoints)
            ts = np.linspace(0, 1, self.n_points)
            self.path_spline = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.linear_spline = itp.make_interp_spline(
                ts, np.array(self.target_linear_values), k=2
            )
            # generate final discretized path
            self.tsProj = np.linspace(0, 1, self.nb_bins)
            self.mazePoints = self.path_spline(self.tsProj)
            self.linear_values = self.linear_spline(self.tsProj)
            self.linear_values[0] = 0.0
            self.linear_values[-1] = 1.0

            try:
                self.lPoints.remove()
                fig.canvas.draw()
            except:
                pass
            self.l0s = try_linearization(ax, self.l0s)
            self.lPoints = ax[0].scatter(
                np.array(self.nnPoints)[:, 0], np.array(self.nnPoints)[:, 1], c="black"
            )
            fig.canvas.draw()

        def b2update(n):
            """
            Remove the last linearization point.
            """
            if len(self.nnPoints) > 0:
                self.nnPoints = self.nnPoints[0 : len(self.nnPoints) - 1]
                self.target_linear_values = self.target_linear_values[
                    0 : len(self.target_linear_values) - 1
                ]
                # create the interpolating object
                self.n_points = len(self.nnPoints)
                ts = np.linspace(0, 1, self.n_points)
                self.path_spline = itp.make_interp_spline(
                    ts, np.array(self.nnPoints), k=2
                )
                self.linear_spline = itp.make_interp_spline(
                    ts, np.array(self.target_linear_values), k=2
                )
                self.tsProj = np.linspace(0, 1, self.nb_bins)
                self.mazePoints = self.path_spline(self.tsProj)
                self.linear_values = self.linear_spline(self.tsProj)

                self.lPoints.remove()
                fig.canvas.draw()
                if len(self.nnPoints) > 2:
                    self.l0s = try_linearization(ax, self.l0s)
                else:
                    self.l0s[1] = ax[0].scatter(
                        euclidData[:, 0], euclidData[:, 1], c="blue"
                    )
                self.lPoints = ax[0].scatter(
                    np.array(self.nnPoints)[:, 0],
                    np.array(self.nnPoints)[:, 1],
                    c="black",
                )
                fig.canvas.draw()

        def b3update(n):
            """
            Empty the linearization points.
            """
            if len(self.nnPoints) > 0:
                self.nnPoints = []
                self.target_linear_values = []
                self.lPoints.remove()
                self.l0s[1] = ax[0].scatter(
                    euclidData[:, 0], euclidData[:, 1], c="blue"
                )
                fig.canvas.draw()

        def onclick(event):
            """
            Handle mouse click events to add new linearization points.
            """
            if event.inaxes == self.l0s[1].axes:
                new_point = np.array([event.xdata, event.ydata])
                insertion_index, interpolated_value = _find_insertion_point(new_point)
                self.nnPoints = np.insert(
                    self.nnPoints, insertion_index, new_point, axis=0
                )
                self.target_linear_values = np.insert(
                    self.target_linear_values, insertion_index, interpolated_value
                )
                self._create_interpolation(clip=False)

                try:
                    self.lPoints.remove()
                    fig.canvas.draw()
                except:
                    pass
                if len(self.nnPoints) > 2:
                    self.n_points = len(self.nnPoints)
                    # create the interpolating object
                    ts = np.linspace(0, 1, self.n_points)
                    self.path_spline = itp.make_interp_spline(
                        ts, np.array(self.nnPoints), k=2
                    )
                    self.linear_spline = itp.make_interp_spline(
                        ts, np.array(self.target_linear_values), k=2
                    )
                    self.tsProj = np.linspace(0, 1, self.nb_bins)
                    self.mazePoints = self.path_spline(self.tsProj)
                    self.linear_values = self.linear_spline(self.tsProj)
                    self.linear_values[0] = 0.0
                    self.linear_values[-1] = 1.0

                    self.l0s = try_linearization(ax, self.l0s)
                else:
                    self.l0s[1] = ax[0].scatter(
                        euclidData[:, 0], euclidData[:, 1], c="blue"
                    )
                self.lPoints = ax[0].scatter(
                    np.array(self.nnPoints)[:, 0],
                    np.array(self.nnPoints)[:, 1],
                    c="black",
                )
                fig.canvas.draw()

        def _find_insertion_point(new_point):
            """
            Find where to insert the new point along the path and what its linear value should be.

            Args:
                new_point: np.array of shape (2,) with [x, y] coordinates

            Returns:
                insertion_index: Index where to insert the point
                interpolated_value: Linear value for the new point
            """
            # Calculate distances from new point to all existing path points
            distances = np.linalg.norm(self.nnPoints - new_point, axis=1)
            closest_idx = np.argmin(distances)
            n_points = len(self.nnPoints)

            if closest_idx == 0:
                # Near the start
                if n_points > 1:
                    dist_to_next = np.linalg.norm(self.nnPoints[1] - new_point)
                    if distances[0] < dist_to_next:
                        insertion_index = 0
                        interpolated_value = max(
                            0.0, self.target_linear_values[0] - 0.01
                        )
                    else:
                        insertion_index = 1
                        interpolated_value = (
                            self.target_linear_values[0] + self.target_linear_values[1]
                        ) / 2
                else:
                    insertion_index = 0
                    interpolated_value = 0.0

            elif closest_idx == n_points - 1:
                # Near the end
                dist_to_prev = np.linalg.norm(self.nnPoints[-2] - new_point)
                if distances[-1] < dist_to_prev:
                    insertion_index = n_points
                    interpolated_value = min(1.0, self.target_linear_values[-1] + 0.01)
                else:
                    insertion_index = n_points - 1
                    interpolated_value = (
                        self.target_linear_values[-2] + self.target_linear_values[-1]
                    ) / 2

            else:
                # In the middle - find the best segment
                prev_idx = closest_idx - 1
                next_idx = closest_idx + 1

                # Project onto both possible segments
                proj_to_prev = _project_point_on_segment(
                    new_point, self.nnPoints[prev_idx], self.nnPoints[closest_idx]
                )
                proj_to_next = _project_point_on_segment(
                    new_point, self.nnPoints[closest_idx], self.nnPoints[next_idx]
                )

                dist_to_prev_seg = np.linalg.norm(new_point - proj_to_prev)
                dist_to_next_seg = np.linalg.norm(new_point - proj_to_next)

                if dist_to_prev_seg < dist_to_next_seg:
                    # Insert between prev_idx and closest_idx
                    insertion_index = closest_idx
                    t = _get_interpolation_parameter(
                        new_point,
                        self.nnPoints[prev_idx],
                        self.nnPoints[closest_idx],
                    )
                    interpolated_value = (1 - t) * self.target_linear_values[
                        prev_idx
                    ] + t * self.target_linear_values[closest_idx]
                else:
                    # Insert between closest_idx and next_idx
                    insertion_index = next_idx
                    t = _get_interpolation_parameter(
                        new_point,
                        self.nnPoints[closest_idx],
                        self.nnPoints[next_idx],
                    )
                    interpolated_value = (1 - t) * self.target_linear_values[
                        closest_idx
                    ] + t * self.target_linear_values[next_idx]

            # Ensure the value stays within bounds
            interpolated_value = np.clip(interpolated_value, 0.0, 1.0)
            return insertion_index, interpolated_value

        def _project_point_on_segment(point, seg_start, seg_end):
            """Project a point onto a line segment."""
            seg_vec = seg_end - seg_start
            point_vec = point - seg_start

            # Handle degenerate case
            seg_length_sq = np.dot(seg_vec, seg_vec)
            if seg_length_sq < 1e-10:
                return seg_start

            # Project onto the line and clamp to segment
            t = np.clip(np.dot(point_vec, seg_vec) / seg_length_sq, 0.0, 1.0)
            return seg_start + t * seg_vec

        def _get_interpolation_parameter(point, seg_start, seg_end):
            """Get the interpolation parameter t for a point projected onto a segment."""
            seg_vec = seg_end - seg_start
            point_vec = point - seg_start

            seg_length_sq = np.dot(seg_vec, seg_vec)
            if seg_length_sq < 1e-10:
                return 0.0

            t = np.dot(point_vec, seg_vec) / seg_length_sq
            return np.clip(t, 0.0, 1.0)

        # Check existence of linearized data

        filename = os.path.join(folder, "nnBehavior.mat")
        if not os.path.exists(filename):
            raise ValueError("this file does not exist :" + folder + "nnBehavior.mat")
        if self.phase is not None:
            filename = os.path.join(folder, "nnBehavior_" + self.phase + ".mat")
            if not os.path.exists(filename):
                assert tables.is_hdf5_file(folder + "nnBehavior.mat")
                import shutil

                print("weird to copy that file now")

                shutil.copyfile(
                    folder + "nnBehavior.mat",
                    folder + "nnBehavior_" + phase + ".mat",
                    follow_symlinks=True,
                )
        # Extract basic behavior
        with tables.open_file(filename, "a") as f:
            children = [c.name for c in f.list_nodes("/behavior")]
            if "linearizationPoints" in children and "targetLinearValues" in children:
                print("Linearization points have been created before")
                if overwrite:
                    f.remove_node("/behavior", "linearizationPoints")
                    f.remove_node("/behavior", "targetLinearValues")
                    print("Overwriting linearization")
                else:
                    return
            # Body
            euclidData = ExampleEuclideanData[
                np.logical_not(np.isnan(np.sum(ExampleEuclideanData, axis=1))), :
            ]
            euclidData = euclidData[1:-1:10, :]  # down sample a bit
            projBin = np.arange(0, stop=1.2, step=0.2)
            self.l0s = [None for _ in projBin]

            # Figure
            fig = plt.figure()
            gs = plt.GridSpec(3, 2, figure=fig)
            ax = [
                fig.add_subplot(gs[:, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[2, 1]),
            ]
            self.l0s = try_linearization(ax, self.l0s)
            self.lPoints = ax[0].scatter(
                np.array(self.nnPoints)[:, 0], np.array(self.nnPoints)[:, 1], c="black"
            )
            ax[0].set_aspect(1)
            b1 = plt.Button(ax[1], "reset", color="grey")
            b1.on_clicked(b1update)
            b2 = plt.Button(ax[2], "remove last", color="orange")
            b2.on_clicked(b2update)
            b3 = plt.Button(ax[3], "empty", color="red")
            b3.on_clicked(b3update)
            # Next we obtain user click to create a new set of linearization points
            [a.set_aspect(1) for a in ax]
            fig.canvas.mpl_connect("button_press_event", onclick)
            plt.show(block=True)

            # create the interpolating object
            self.n_points = len(self.nnPoints)
            ts = np.linspace(0, 1, self.n_points)
            self.path_spline = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.linear_spline = itp.make_interp_spline(
                ts, np.array(self.target_linear_values), k=2
            )
            self.tsProj = np.linspace(0, 1, self.nb_bins)
            self.mazePoints = self.path_spline(self.tsProj)
            self.linear_values = self.linear_spline(self.tsProj)
            self.linear_values[0] = 0.0
            self.linear_values[-1] = 1.0
            # plot the exact linearization variable:
            _, linearTrue = self.apply_linearization(euclidData)

            self.plot_linearization_variable(euclidData, linearTrue, training, folder)
            # Save
            if "linearizationPoints" in children:
                f.remove_node("/behavior", "linearizationPoints")
            if "targetLinearValues" in children:
                f.remove_node("/behavior", "targetLinearValues")
            f.create_array("/behavior", "linearizationPoints", self.nnPoints)
            f.create_array("/behavior", "targetLinearValues", self.target_linear_values)
            f.flush()
            f.close()

    def plot_linearization_variable(
        self, euclidData, linearTrue=None, training=False, folder=None, show=True
    ):
        """
        Plot the linearization variable with a color map.

        Args:
            euclidData (np.ndarray): The Euclidean data to plot.
            linearTrue (np.ndarray): The linearization variable.
            training (bool): Whether the plot is for training data.
            folder (str): The folder to save the plot.
        """
        if linearTrue is None:
            print("No linearization variable provided, computing it.")
            _, linearTrue = self.apply_linearization(euclidData)
        cm = plt.get_cmap("Spectral")
        norm = mcolors.Normalize(vmin=0, vmax=1)
        fig, self.axScatter = plt.subplots()
        self._add_custom_lines(
            colors=self.custom_line_colors,
            styles=self.custom_line_styles,
            widths=self.custom_line_widths,
            alpha=0.5,
        )
        scatter_plot = self.axScatter.scatter(
            euclidData[:, 0], euclidData[:, 1], c=linearTrue, cmap=cm, norm=norm
        )
        corners = np.array([[0, 1], [1, 1], [0.5, 0.875]])
        _, linearCorner = self.apply_linearization(corners)
        self.axScatter.scatter(
            corners[:, 0],
            corners[:, 1],
            c=linearCorner,
            cmap=cm,
            norm=norm,
            marker="x",
            s=100,
            label="Corners & center",
        )
        # annotate the corners
        for i, corner in enumerate(corners):
            self.axScatter.annotate(
                f"Value: {linearCorner[i]:.2f}",
                (corner[0], corner[1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        plt.colorbar(scatter_plot, ax=self.axScatter, norm=norm)
        fig.suptitle(
            f"Linearization variable, Spectral colormap for mask {training} and phase {self.phase}. min={linearTrue.min():.2f}, max={linearTrue.max():.2f}",
        )
        # Create new axes on the right and on the top of the current axes
        divider = make_axes_locatable(self.axScatter)
        axHistX = divider.append_axes("bottom", 1.2, pad=0.1, sharex=self.axScatter)
        axHistY = divider.append_axes("right", 1.2, pad=0.1, sharey=self.axScatter)

        # Make some labels invisible
        axHistX.xaxis.set_tick_params(labelbottom=False)
        axHistY.yaxis.set_tick_params(labelleft=False)

        # Plot histograms
        axHistX.hist(euclidData[:, 0], bins=30, color="gray")
        axHistY.hist(euclidData[:, 1], bins=30, orientation="horizontal", color="gray")

        # Set labels
        axHistX.set_ylabel("Frequency")
        axHistY.set_xlabel("Frequency")
        fig.savefig(
            os.path.join(folder, f"linearizationVariable_{self.phase}_{training=}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        if show:
            plt.show(block=True)
        plt.close(fig)

    def _add_custom_lines(self, colors, styles, widths, alpha):
        """Add custom line segments (like maze walls, boundaries, etc.)."""
        if not self.custom_lines:
            return

        # Handle different input formats
        lines_to_plot = []

        for line_data in self.custom_lines:
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
                print(f"Warning: Unrecognized line format: {line_data}")

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
            self.axScatter.plot(
                line_points[:, 0],  # X coordinates
                line_points[:, 1],  # Y coordinates
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)],
                linewidth=widths[i % len(widths)],
                alpha=alphas[i % len(alphas)],
                zorder=10,  # Behind the trajectory
            )
