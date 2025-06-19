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

    def verify_linearization(self, ExampleEuclideanData, folder, overwrite=False):
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
            cm = plt.get_cmap("Spectral")
            norm = mcolors.Normalize(vmin=linearTrue.min(), vmax=linearTrue.max())
            fig, axScatter = plt.subplots()
            scatter_plot = axScatter.scatter(
                euclidData[:, 0], euclidData[:, 1], c=linearTrue, cmap=cm, norm=norm
            )
            # Display the color bar for the scatter plot using the Spectral colormap
            plt.colorbar(scatter_plot, ax=axScatter, norm=norm)
            fig.suptitle("Linearization variable, Spectral colormap")

            # Create new axes on the right and on the top of the current axes
            divider = make_axes_locatable(axScatter)
            axHistX = divider.append_axes("bottom", 1.2, pad=0.1, sharex=axScatter)
            axHistY = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

            # Make some labels invisible
            axHistX.xaxis.set_tick_params(labelbottom=False)
            axHistY.yaxis.set_tick_params(labelleft=False)

            # Plot histograms
            axHistX.hist(euclidData[:, 0], bins=30, color="gray")
            axHistY.hist(
                euclidData[:, 1], bins=30, orientation="horizontal", color="gray"
            )

            # Set labels
            axHistX.set_ylabel("Frequency")
            axHistY.set_xlabel("Frequency")
            plt.show(block=True)
            # Save
            f.create_array("/behavior", "linearizationPoints", self.nnPoints)
            f.flush()
            f.close()
