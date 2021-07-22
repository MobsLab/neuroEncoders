## Created by Pierre 01/04/2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp
import os
import tables
from pykeops.numpy import LazyTensor as LazyTensor_np
import pykeops as pykeops

class UMazeLinearizer:
    # A class to define a linearization function of the data.
    # Depending on the maze shape, user might want to change this class
    # to fit to their maze shape.
    def __init__(self,folder):
        with tables.open_file(folder + 'nnBehavior.mat', "a") as f:
            children = [c.name for c in f.list_nodes("/behavior")]
            if "linearizationPoints" in children:
                self.nnPoints =f.root.behavior.linearizationPoints[:]
            else:
                self.nnPoints = [[0.45, 0.40], [0.45, 0.65], [0.45, 0.9], [0.7, 0.9], [0.9, 0.9], [0.9, 0.7], [0.9, 0.4]]

            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.tsProj = np.arange(0, stop=1, step=1 / 100)
            self.mazepoints = itpObject(self.tsProj)

    def verifyLinearization(self,ExampleEuclideanData,folder,overwrite=True):
        ## A function to verify and possibly change the linearization.

        #down sample a bit
        euclidData = ExampleEuclideanData[np.logical_not(np.isnan(np.sum(ExampleEuclideanData, axis=1))), :]
        euclidData = euclidData[1:-1:10,:]
        projBin = np.arange(0, stop=1.2, step=0.2)

        self.l0s = [None for _ in projBin]
        def tryLinearization(ax,l0s):
            projTruePos, linearTrue = self.applyLinearization(euclidData)

            binIndex = [np.where((linearTrue >= projBin[id]) * (linearTrue < projBin[id + 1]))[0] for id in
                        range(len(projBin) - 1)]
            cm = plt.get_cmap("tab20")
            for tpl in enumerate(binIndex):
                id, bId = tpl
                try:
                    l0s[id].remove()
                except:
                    None
                l0s[id] = ax[0].scatter(euclidData[bId, 0], euclidData[bId, 1], c=[cm(id)])
            return l0s

        fig = plt.figure()
        gs = plt.GridSpec(3, 2, figure=fig)
        ax =[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,1])]
        # l0 = ax[0].scatter(ExampleEuclideanData[:, 0], ExampleEuclideanData[:, 1], alpha=0.2)
        self.l0s = tryLinearization(ax,self.l0s)
        self.lPoints = ax[0].scatter(np.array(self.nnPoints)[:, 0], np.array(self.nnPoints)[:, 1], c="black")
        ax[0].set_aspect(1)
        b1 = plt.Button(ax[1],"reset",color="grey")
        def b1update(n):
            self.nnPoints = [[0.45, 0.40], [0.45, 0.65], [0.45, 0.9], [0.7, 0.9], [0.9, 0.9], [0.9, 0.7], [0.9, 0.4]]
            # create the interpolating object
            ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
            itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
            self.tsProj = np.arange(0, stop=1, step=1 / 100)
            self.mazepoints = itpObject(self.tsProj)
            try:
                self.lPoints.remove()
                fig.canvas.draw()
            except:
                pass
            self.l0s = tryLinearization(ax, self.l0s)
            self.lPoints = ax[0].scatter(np.array(self.nnPoints)[:, 0], np.array(self.nnPoints)[:, 1], c="black")
            fig.canvas.draw()
        b1.on_clicked(b1update)

        b2 = plt.Button(ax[2],"remove last",color="orange")
        def b2update(n):
            if len(self.nnPoints)>0:
                self.nnPoints = self.nnPoints[0:len(self.nnPoints)-1]
                # create the interpolating object
                ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
                itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
                self.tsProj = np.arange(0, stop=1, step=1 / 100)
                self.mazepoints = itpObject(self.tsProj)

                self.lPoints.remove()
                fig.canvas.draw()

                if (len(self.nnPoints)>2):
                    self.l0s = tryLinearization(ax, self.l0s)
                else:
                    self.l0s[1] = ax[0].scatter(euclidData[:,0],euclidData[:,1], c="blue")
                self.lPoints = ax[0].scatter(np.array(self.nnPoints)[:, 0], np.array(self.nnPoints)[:, 1], c="black")
                fig.canvas.draw()
        b2.on_clicked(b2update)

        b3 = plt.Button(ax[3], "empty", color="red")
        def b3update(n):
            if len(self.nnPoints) > 0:
                self.nnPoints = []
                self.lPoints.remove()
                self.l0s[1] = ax[0].scatter(euclidData[:,0],euclidData[:,1], c="blue")
                fig.canvas.draw()
        b3.on_clicked(b3update)

        # Next we obtain user click to create a new set of linearization points
        def onclick(event):
            if event.inaxes==self.l0s[1].axes:
                self.nnPoints += [[event.xdata,event.ydata]]
                try:
                    self.lPoints.remove()
                    fig.canvas.draw()
                except:
                    pass
                if (len(self.nnPoints) > 2):
                    # create the interpolating object
                    ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
                    itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
                    self.tsProj = np.arange(0, stop=1, step=1 / 100)
                    self.mazepoints = itpObject(self.tsProj)

                    self.l0s = tryLinearization(ax, self.l0s)
                else:
                    self.l0s[1] = ax[0].scatter(euclidData[:, 0], euclidData[:, 1], c="blue")
                self.lPoints = ax[0].scatter(np.array(self.nnPoints)[:, 0], np.array(self.nnPoints)[:, 1], c="black")
                fig.canvas.draw()
        [a.set_aspect(1) for a in ax]
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # create the interpolating object
        ts = np.arange(0, stop=1, step=1 / np.array(self.nnPoints).shape[0])
        itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
        self.tsProj = np.arange(0, stop=1, step=1 / 100)
        self.mazepoints = itpObject(self.tsProj)

        # plot the exact linearization variable:
        projTruePos, linearTrue = self.applyLinearization(euclidData)
        cm = plt.get_cmap("Spectral")
        fig,ax = plt.subplots()
        ax.scatter(euclidData[:,0],euclidData[:,1],c=cm(linearTrue))
        ax.set_title("Linearization variable, Spectral colormap")
        plt.show()

        if overwrite:
            with tables.open_file(folder + 'nnBehavior.mat', "a") as f:
                children = [c.name for c in f.list_nodes("/behavior")]
                if "linearizationPoints" in children:
                    f.remove_node("/behavior", "linearizationPoints")
                f.create_array("/behavior", "linearizationPoints", self.nnPoints)
                f.flush()

        # create the interpolating object
        ts = np.arange(0, stop=1, step=1/np.array(self.nnPoints).shape[0])
        itpObject = itp.make_interp_spline(ts, np.array(self.nnPoints), k=2)
        self.tsProj = np.arange(0, stop=1, step=1/100)
        self.mazepoints = itpObject(self.tsProj)

    def applyLinearization(self, euclideanData):
        return self.pykeopsLinearization(euclideanData)
        projectedPos = np.zeros([euclideanData.shape[0], 2])
        linearFeature = np.zeros([euclideanData.shape[0]])
        for idp in range(euclideanData.shape[0]):
            bestPoint = np.argmin(np.sum(np.square(np.reshape(euclideanData[idp,:],[1,euclideanData.shape[1]]) - self.mazepoints),axis=1),axis=0)
            projectedPos[idp,:] = self.mazepoints[bestPoint,:]
            linearFeature[idp] = self.tsProj[bestPoint]
        return projectedPos,linearFeature

    def pykeopsLinearization(self,euclideanData):
        if euclideanData.dtype != self.mazepoints.dtype:
            euclideanData = euclideanData.astype(self.mazepoints.dtype)

        euclidData_lazy = LazyTensor_np(euclideanData[None,:,:])
        mazePoint_lazy = LazyTensor_np(self.mazepoints[:,None,:])

        distance_matrix_lazy = (mazePoint_lazy-euclidData_lazy).square().sum(axis=-1)
        #find the argmin
        bestPoints = distance_matrix_lazy.argmin_reduction(axis=0)
        # #extract a mask by using the property of the min and a relu filtering:
        # #Note: the assumption is that the argmin gives only one result
        # bestPoints_Lazy = LazyTensor_np(bestPoints[None,:,:])
        # mask = (((mazePoint_lazy-bestPoints_Lazy).square().sum(axis=-1)).relu().sign()-1).sign()

        projectedPos = self.mazepoints[bestPoints[:,0],:]
        linearPos = self.tsProj[bestPoints[:,0]]

        return projectedPos,linearPos









def doubleArmMazeLinearization(euclideanData,scale=True,path_to_folder=""):
    # We find a simpler parametrization of the maze,
    # project each prediction on this parametrization (forcing it to be on the maze)
    # and project back into euclidean values.
    #Inputs:
    # euclideanData: (n,2) array, n: number of position to project
    # nnPointsFilePath: string, location of the .csv file where the nearestneighbor points
    # describing the maze are located.

    #Note: the euclidean data need to be in the original coordinate scale, i.e cm in [0,261]
    # approximately

    nnPointsFilePath1 = os.path.join(path_to_folder,"interpolationKmeanCenter1.csv")
    nnPointsFilePath2 = os.path.join(path_to_folder,"interpolationKmeanCenter1.csv")

    df = pd.read_csv(nnPointsFilePath1)
    nnPoints1 = df.values
    df = pd.read_csv(nnPointsFilePath2)
    nnPoints2 = df.values

    if scale:
        posMax = np.max([np.max(nnPoints1),np.max(nnPoints2)])
        nnPoints1 = nnPoints1/posMax
        nnPoints2 = nnPoints2/posMax

    #create the interpolating object
    ts = np.arange(0,stop=1,step=1/nnPoints1.shape[0])
    itpObject1 = itp.make_interp_spline(ts,nnPoints1,k=2)
    ts = np.arange(0,stop=1,step=1/nnPoints2.shape[0])
    itpObject2 = itp.make_interp_spline(ts,nnPoints2,k=2)
    tsProj = np.arange(0,stop=1,step=1/10000)
    mazepoints1 = itpObject1(tsProj)
    mazepoints2 = itpObject2(tsProj)
    mazepoints = np.concatenate([mazepoints1,mazepoints2],axis=0)

    projectedPos = np.zeros([euclideanData.shape[0],2])
    for idp in range(euclideanData.shape[0]):
        bestPoint = np.argmin(np.sum(np.square(np.reshape(euclideanData[idp,:],[1,euclideanData.shape[1]]) - mazepoints),axis=1),axis=0)
        projectedPos[idp,:] = mazepoints[bestPoint,:]
    return projectedPos

## Example code:
#
# x = np.arange(0,stop=250,step=1)
# euclideanData = np.transpose(np.stack([x,x]))
# euclideanData[:,1] =150
# nnPointsFilePath1 = "../../SimplerFeature/data/manifoldParam/interpolationKmeanCenter1.csv"
# nnPointsFilePath2 = "../../SimplerFeature/data/manifoldParam/interpolationKmeanCenter2.csv"
#
# df = pd.read_csv(nnPointsFilePath1)
# nnPoints1 = df.values
# df = pd.read_csv(nnPointsFilePath2)
# nnPoints2 = df.values
# #create the interpolating object
# ts = np.arange(0,stop=1,step=1/nnPoints1.shape[0])
# itpObject1 = itp.make_interp_spline(ts,nnPoints1,k=2)
# ts = np.arange(0,stop=1,step=1/nnPoints2.shape[0])
# itpObject2 = itp.make_interp_spline(ts,nnPoints2,k=2)
# tsProj = np.arange(0,stop=1,step=1/10000)
# mazepoints1 = itpObject1(tsProj)
# mazepoints2 = itpObject2(tsProj)
# mazepoints = np.concatenate([mazepoints1,mazepoints2],axis=0)
#
# projectedPos = np.zeros([euclideanData.shape[0],2])
# for idp in range(euclideanData.shape[0]):
#     bestPoint = np.argmin(np.sum(np.square(np.reshape(euclideanData[idp,:],[1,euclideanData.shape[1]]) - mazepoints),axis=1),axis=0)
#     projectedPos[idp,:] = mazepoints[bestPoint,:]
#
# fig,ax = plt.subplots()
# ax.scatter(euclideanData[:,0],euclideanData[:,1])
# ax.scatter(mazepoints[:,0],mazepoints[:,1])
# ax.scatter(projectedPos[:,0],projectedPos[:,1])
# fig.show()

















