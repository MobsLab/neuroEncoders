import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp
from importData.ImportClusters import getBehavior
from importData.rawDataParser import  inEpochsMask
import pykeops
from tqdm import tqdm


def analyze_cnn_output(params,behavior_data,projectPath):
    import sklearn.decomposition
    from sklearn.manifold import TSNE
    groupFeatures = []
    svalues = []
    explainedVariances = []
    transformedFeatures = []
    tsneFeatures = []
    for idg in range(params.nGroups):
        groupFeatures.append(
            np.reshape(output_test[2][:, :, idg * 128:(idg + 1) * 128], [np.prod(output_test[2].shape[0:2]), 128]))
        res = sklearn.decomposition.PCA()
        res.fit(groupFeatures[idg][np.sum(np.abs(groupFeatures[idg]), axis=1) > 0, :])
        svalues.append(res.singular_values_)
        explainedVariances.append(res.explained_variance_ratio_)
        transformedFeatures.append(res.transform(groupFeatures[idg][np.sum(np.abs(groupFeatures[idg]), axis=1) > 0, :]))

        tsne = TSNE(n_components=2)
        tsneFeatures.append(tsne.fit_transform(transformedFeatures[idg]))

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(np.argmax(transformedFeatures[0], axis=1), bins=100)  # [np.sum(np.abs(groupFeatures[3]),axis=1)>0,:]
    # ax.set_aspect(transformedFeatures[0].shape[1]/transformedFeatures[0].shape[0])
    fig.show()

    fig, ax = plt.subplots(len(tsneFeatures), figsize=(4, 20))
    cm = plt.get_cmap("tab20")
    for idg in range(params.nGroups):
        bestsvd = np.argmax(transformedFeatures[idg][:, 0:20], axis=1)
        ax[idg].scatter(tsneFeatures[idg][:, 0], tsneFeatures[idg][:, 1], s=1, c=cm(bestsvd))
        ax[idg].set_aspect(1)
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(3, 2)
    for idg in range(params.nGroups):
        ax[0, 0].plot(range(128), svalues[idg])
        ax[1, 0].plot(range(128), explainedVariances[idg])
        ax[2, 0].plot(range(128), explainedVariances[idg])
        ax[0, 1].plot(range(12), svalues[idg][:12])
        ax[1, 1].plot(range(12), explainedVariances[idg][:12])
        ax[2, 1].plot(range(12), explainedVariances[idg][:12])
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    ax[2, 0].set_yscale("log")
    ax[2, 1].set_yscale("log")
    fig.show()

    fig, ax = plt.subplots()
    gf = np.ravel(groupFeatures[0])
    ax.hist(gf[np.not_equal(gf, 0)], bins=100)
    fig.show()

    norm = lambda x: (x - np.min(x, axis=1)) / (np.max(x, axis=1) - np.min(x, axis=1))

    gf = groupFeatures[0][np.sum(np.abs(groupFeatures[0]), axis=1) > 0, :]

    # let us reorganize all the spikes feature by their PCA argmax:
    # TODO

    corrMat = np.matmul(norm(gf), np.transpose(norm(gf)))
    fig, ax = plt.subplots()
    # cm = plt.get_cmap("Reds")
    ax.imshow(corrMat)
    ax.set_aspect(corrMat.shape[1] / corrMat.shape[0])
    fig.show()

    fig, ax = plt.subplots()
    cm = plt.get_cmap("Reds")
    ax.matshow(transformedFeatures[0][:, 0:12], cmap=cm)
    ax.set_aspect(transformedFeatures[0][:, 0:12].shape[1] / transformedFeatures[0].shape[0])
    fig.show()

    # More interestingly: let us compare the cluster_data results with the spike being considered
    # todo: confront the two filtering by making sure the time step difference is less than 15/sampling_rate.
    from importData import ImportClusters
    cluster_data = ImportClusters.load_spike_sorting(projectPath)
    spike_times = cluster_data["Spike_times"]
    maskTime = inEpochsMask(spike_times[0][:, 0], behavior_data['Times']['testEpochs'])
    spike_labels = cluster_data["Spike_labels"]

    sl = spike_labels[0][maskTime]
    st = spike_times[0][maskTime]
    # some spikes are not assigned to any clusters (noisy spikes). we give them the label 0
    cluster_labels = np.zeros(spike_labels[0].shape[0])
    cluster_labels[np.sum(sl, axis=1) > 0] = np.argmax(sl[np.sum(sl, axis=1) > 0, :], axis=1) + 1
    fig, ax = plt.subplots()
    ax.hist(cluster_labels, width=0.5, bins=range(spike_labels[0].shape[1] + 1))
    ax.set_yscale("log")
    fig.show()

    # next we extract spike time from what is feeded to tensorflow:
    datasetTimes = datasetOneBatch.map(lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE)
    times = list(datasetTimes.as_numpy_iterator())
    times = np.array(times)[0, :]

    print("gathering true feature")
    datasetPos = dataset.map(lambda x, y: x["pos"], num_parallel_calls=tf.data.AUTOTUNE)
    fullFeatureTrue = list(datasetPos.as_numpy_iterator())
    fullFeatureTrue = np.array(fullFeatureTrue)
    print("gathering exact time of spikes")
    datasetTimes = dataset.map(lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE)
    times = list(datasetTimes.as_numpy_iterator())
    times = np.array(times)
    datasetPos_index = dataset.map(lambda x, y: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE)
    pos_index = list(datasetPos_index.as_numpy_iterator())

    outLoss = np.expand_dims(output_test[1], axis=1)
    featureTrue = np.reshape(fullFeatureTrue, [output_test[0].shape[0], output_test[0].shape[-1]])
    times = np.reshape(times, [output_test[0].shape[0]])

    convNetFeature = np.stack(output_test[2])

    pca(output_test[2])

    import sklearn.manifold.t_sne as tsne

    tsne()

    # projPredPos, linearPred = linearizationFunction(output_test[0][:, :2])
    # projTruePos, linearTrue = linearizationFunction(featureTrue)

    # What we would like to do:
    # Step0: PCA?
    # Step1: try a TSNE to see if clear clusters
    # Note: stack over all spikes....
    # Step2: k-nn cluster with k=the number of cluster from PCA