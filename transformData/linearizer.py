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

pykeopsset_verbose(False)  # Disable verbose output from PyKeOps


class UMazeLinearizer:
    """
    A class to define a linearization function of the data.
    Depending on the maze shape, user might want to change this class
    to fit to their maze shape.

    args:
    folder: str, the folder where the linearization points are saved
    nb_bins: int, the number of bins to use for the linearization, defaults to 100
    phase: str, the phase of the experiment, defaults to None
    data_helper: object, a helper object to provide custom lines (like maze walls, boundaries, etc.), defaults to None
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
        # Extract basic behavior
        with tables.open_file(filename, "a") as f:
            children = [c.name for c in f.list_nodes("/behavior")]
            if "linearizationPoints" in children:
                self.nnPoints = f.root.behavior.linearizationPoints[:]
            else:
                self.nnPoints = [
                    [0.15, 0.1],
                    [0.15, 0.5],
                    [0.15, 0.9],
                    [0.5, 0.9],
                    [0.9, 0.9],
                    [0.85, 0.5],
                    [0.85, 0.1],
                ]

            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            # equally spaced linear points. As many as the number of points
            # pu in the verify_linearization function (by default 7 anchor points)
            self.itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            # itpObject is the interpolating object that finds a fit between
            # the anchor points and the equally spaced 2D points
            self.nb_bins = nb_bins
            self.tsProj = np.arange(0, stop=1, step=1 / self.nb_bins)
            self.mazePoints = self.itpObject(self.tsProj)  # from 1D to 2D

            if "aligned_ref" in children:
                self.aligned_ref = f.root.behavior.aligned_ref[:]
            else:
                self.aligned_ref = None

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

    def apply_linearization(self, euclideanData, keops=True):
        if keops:
            return self.pykeops_linearization(euclideanData)
        else:
            projectedPos = np.zeros([euclideanData.shape[0], 2])
            linearFeature = np.zeros([euclideanData.shape[0]])
            for idp in range(euclideanData.shape[0]):
                bestPoint = np.argmin(
                    np.sum(
                        np.square(
                            np.reshape(
                                euclideanData[idp, :], [1, euclideanData.shape[1]]
                            )
                            - self.mazePoints
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                projectedPos[idp, :] = self.mazePoints[bestPoint, :]
                linearFeature[idp] = self.tsProj[bestPoint]

            return projectedPos, linearFeature

    def pykeops_linearization(self, euclideanData):
        if euclideanData.dtype != self.mazePoints.dtype:
            euclideanData = euclideanData.astype(self.mazePoints.dtype)

        euclidData_lazy = LazyTensor_np(euclideanData[None, :, :])
        mazePoint_lazy = LazyTensor_np(self.mazePoints[:, None, :])

        distance_matrix_lazy = (mazePoint_lazy - euclidData_lazy).square().sum(axis=-1)
        # find the argmin
        bestPoints = distance_matrix_lazy.argmin_reduction(axis=0)
        projectedPos = self.mazePoints[bestPoints[:, 0], :]
        linearPos = self.tsProj[bestPoints[:, 0]]

        return projectedPos, linearPos

    def verify_linearization(
        self, ExampleEuclideanData, folder, overwrite=False, training=False
    ):
        """
        A function to verify and possibly change the linearization.
        This function will plot the data and allow the user to change the linearization points. The new linearization points will be saved in the folder.

        args:
        ExampleEuclideanData: np.array, the data to be linearized
        folder: str, the folder where the linearization points are saved
        overwrite: bool, if True, the linearization points will be overwritten, defaults to False
        """

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
            self.nnPoints = [
                [0.15, 0.1],
                [0.15, 0.5],
                [0.15, 0.9],
                [0.5, 0.9],
                [0.9, 0.9],
                [0.85, 0.5],
                [0.85, 0.1],
            ]
            # create the interpolating object
            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            self.itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.tsProj = np.arange(0, stop=1, step=1 / 100)
            self.mazePoints = self.itpObject(self.tsProj)
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
            if len(self.nnPoints) > 0:
                self.nnPoints = self.nnPoints[0 : len(self.nnPoints) - 1]
                # create the interpolating object
                ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
                self.itpObject = itp.make_interp_spline(
                    ts, np.array(self.nnPoints), k=2
                )
                self.tsProj = np.arange(0, stop=1, step=1 / 100)
                self.mazePoints = self.itpObject(self.tsProj)
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
            if len(self.nnPoints) > 0:
                self.nnPoints = []
                self.lPoints.remove()
                self.l0s[1] = ax[0].scatter(
                    euclidData[:, 0], euclidData[:, 1], c="blue"
                )
                fig.canvas.draw()

        def onclick(event):
            if event.inaxes == self.l0s[1].axes:
                self.nnPoints += [[event.xdata, event.ydata]]
                try:
                    self.lPoints.remove()
                    fig.canvas.draw()
                except:
                    pass
                if len(self.nnPoints) > 2:
                    # create the interpolating object
                    ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
                    self.itpObject = itp.make_interp_spline(
                        ts, np.array(self.nnPoints), k=2
                    )
                    self.tsProj = np.arange(0, stop=1, step=1 / 100)
                    self.mazePoints = self.itpObject(self.tsProj)

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
            if "linearizationPoints" in children:
                print("Linearization points have been created before")
                if overwrite:
                    f.remove_node("/behavior", "linearizationPoints")
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
            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            self.itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.tsProj = np.arange(0, stop=1, step=1 / 100)
            self.mazePoints = self.itpObject(self.tsProj)
            # plot the exact linearization variable:
            _, linearTrue = self.apply_linearization(euclidData)

            self.plot_linearization_variable(euclidData, linearTrue, training, folder)
            # Save
            f.create_array("/behavior", "linearizationPoints", self.nnPoints)
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
        norm = mcolors.Normalize(vmin=linearTrue.min(), vmax=linearTrue.max())
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
        plt.colorbar(scatter_plot, ax=self.axScatter, norm=norm)
        fig.suptitle(
            f"Linearization variable, Spectral colormap for mask {training} and phase {self.phase}"
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
