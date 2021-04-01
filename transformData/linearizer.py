## Created by Pierre 01/04/2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp


def uMazeLinearization(euclideanData):
    nnPoints = np.array([[0.175,0],[0.175,0.4],[0.195,0.82],[0.4,0.875],[0.6,0.875],[0.805,0.82],[0.815,0.4],[0.815,0]])
    # create the interpolating object
    ts = np.arange(0, stop=1, step=1/nnPoints.shape[0])
    itpObject = itp.make_interp_spline(ts, nnPoints, k=2)

    tsProj = np.arange(0, stop=1, step=1/10000)
    mazepoints = itpObject(tsProj)

    projectedPos = np.zeros([euclideanData.shape[0], 2])
    for idp in range(euclideanData.shape[0]):
        bestPoint = np.argmin(np.sum(np.square(np.reshape(euclideanData[idp,:],[1,euclideanData.shape[1]]) - mazepoints),axis=1),axis=0)
        projectedPos[idp,:] = mazepoints[bestPoint,:]
    return projectedPos


def doubleArmMazeLinearization(euclideanData,scale=True):
    # We find a simpler parametrization of the maze,
    # project each prediction on this parametrization (forcing it to be on the maze)
    # and project back into euclidean values.
    #Inputs:
    # euclideanData: (n,2) array, n: number of position to project
    # nnPointsFilePath: string, location of the .csv file where the nearestneighbor points
    # describing the maze are located.

    #Note: the euclidean data need to be in the original coordinate scale, i.e cm in [0,261]
    # approximately

    nnPointsFilePath1 = "interpolationKmeanCenter1.csv"
    nnPointsFilePath2 = "interpolationKmeanCenter2.csv"

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

















