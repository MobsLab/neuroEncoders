## Created by Pierre 01/04/2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp
import tables
from pykeops.numpy import LazyTensor as LazyTensor_np


class UMazeLinearizer:
    """
    A class to define a linearization function of the data.
    Depending on the maze shape, user might want to change this class
    to fit to their maze shape.

    args:
    folder: str, the folder where the linearization points are saved
    nb_bins: int, the number of bins to use for the linearization, defaults to 100
    """

    def __init__(self, folder: str, nb_bins: int = 100):
        with tables.open_file(folder + "nnBehavior.mat", "a") as f:
            children = [c.name for c in f.list_nodes("/behavior")]
            if "linearizationPoints" in children:
                self.nnPoints = f.root.behavior.linearizationPoints[:]
            else:
                self.nnPoints = [
                    [0.45, 0.40],
                    [0.45, 0.65],
                    [0.45, 0.9],
                    [0.7, 0.9],
                    [0.9, 0.9],
                    [0.9, 0.7],
                    [0.9, 0.4],
                ]

            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.nb_bins = nb_bins
            self.tsProj = np.arange(0, stop=1, step=1 / self.nb_bins)
            self.mazePoints = itpObject(self.tsProj)

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
                [0.45, 0.40],
                [0.45, 0.65],
                [0.45, 0.9],
                [0.7, 0.9],
                [0.9, 0.9],
                [0.9, 0.7],
                [0.9, 0.4],
            ]
            # create the interpolating object
            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.tsProj = np.arange(0, stop=1, step=1 / 100)
            self.mazePoints = itpObject(self.tsProj)
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
                itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
                self.tsProj = np.arange(0, stop=1, step=1 / 100)
                self.mazePoints = itpObject(self.tsProj)
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
                    itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
                    self.tsProj = np.arange(0, stop=1, step=1 / 100)
                    self.mazePoints = itpObject(self.tsProj)

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
        with tables.open_file(folder + "nnBehavior.mat", "a") as f:
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
            plt.show()
            # create the interpolating object
            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.tsProj = np.arange(0, stop=1, step=1 / 100)
            self.mazePoints = itpObject(self.tsProj)
            # plot the exact linearization variable:
            _, linearTrue = self.apply_linearization(euclidData)
            cm = plt.get_cmap("Spectral")
            fig, ax = plt.subplots()
            ax.scatter(euclidData[:, 0], euclidData[:, 1], c=cm(linearTrue))
            ax.set_title("Linearization variable, Spectral colormap")
            plt.show()
            # Save
            f.create_array("/behavior", "linearizationPoints", self.nnPoints)
            f.flush()
            f.close()
