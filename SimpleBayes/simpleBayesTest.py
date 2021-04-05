##Pierre 01/04
# to test the bayesian decoder.

import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import transformData.linearizer
from  importData import ImportClusters
import decodebayes

class Project():
    def __init__(self, xmlPath, datPath='', jsonPath=None):
        if xmlPath[-3:] != "xml": #change the last character to xml if it was not and checks that it exists
            if os.path.isfile(xmlPath[:-3]+"xml"):
                xmlPath = xmlPath[:-3]+"xml"
            else:
                raise ValueError("the path "+xmlPath+" doesn't match a .xml file")
        self.xml = xmlPath
        self.baseName = xmlPath[:-4]
        if datPath == '':
            self.dat = self.baseName + '.dat'
        else:
            self.dat = datPath
        findFolder = lambda path: path if path[-1]=='/' or len(path)==0 else findFolder(path[:-1])
        self.folder = findFolder(self.dat)
        self.fil = self.dat[:-4] + '.fil'
        if jsonPath == None:
            self.json = self.baseName + '.json'
            self.graph = self.folder + 'graph/decoder'
            self.graphMeta = self.folder + 'graph/decoder.meta'
        else:
            print('using file:',jsonPath)
            self.json = jsonPath
            self.thresholds, self.graph = self.getThresholdsAndGraph()
            self.graphMeta = self.graph + '.meta'

        self.tfrec = {
            "train": self.folder + 'dataset/trainingDataset.tfrec',
            "test": self.folder + 'dataset/testingDataset.tfrec'}

        #TO change at every experiment:
        self.resultsPath = self.folder + 'resultsBayesUMazeDataset'
        self.resultsNpz = self.resultsPath + '/inferring.npz'
        self.resultsMat = self.resultsPath + '/inferring.mat'

        if not os.path.isdir(self.folder + 'dataset'):
            os.makedirs(self.folder + 'dataset')
        if not os.path.isdir(self.folder + 'graph'):
            os.makedirs(self.folder + 'graph')
        if not os.path.isdir(self.resultsPath):
            os.makedirs(self.resultsPath )
        if not os.path.isdir(os.path.join(self.resultsPath, "resultInference")):
            os.makedirs(os.path.join(self.resultsPath, "resultInference"))

    def clu(self, g):
        return self.baseName + ".clu." + str(g+1)

    def res(self, g):
        return self.baseName + ".res." + str(g+1)

    def spk(self, g):
        return self.baseName + ".spk." + str(g+1)

    def pos(self, g):
        return self.folder + "dataset/pos." + str(g+1) + ".npz"

    def getThresholdsAndGraph(self):
        import json
        with open(self.json, 'r') as f:
            info = json.loads(f.read())
        return [[abs(info[d][f]) for f in ['threshold'+str(c) for c in range(info[d]['nChannels'])]] \
                for d in ['group'+str(g) for g in range(info['nGroups'])]], \
            info['encodingPrefix']

def main():
    # xml_path = "/home/mobs/Documents/PierreCode/dataTest/RatCataneseCluster/rat122-20090731.xml"
    xml_path = "/home/mobs/Documents/PierreCode/dataTest/Mouse-K168/M1168_20210122_UMaze.xml"
    projectPath = Project(xml_path)

    cluster_save_path = os.path.join(projectPath.resultsPath, 'ClusterData')
    if os.path.isfile(os.path.join(cluster_save_path, 'Spike_labels0.csv')):
        cluster_data = {"Spike_labels":[],"Spike_times":[],"Spike_positions":[],"Spike_speed":[]}
        print("Reading saved cluster csv file")
        for l in range(4):
            df = pd.read_csv(os.path.join(cluster_save_path, "Spike_labels"+str(l)+".csv"))
            cluster_data["Spike_labels"].append(df.values[:,1:])
            df = pd.read_csv(os.path.join(cluster_save_path, "spike_time"+str(l)+".csv"))
            cluster_data["Spike_times"].append(df.values[:,1:])
            df = pd.read_csv(os.path.join(cluster_save_path, "spike_positions"+str(l)+".csv"))
            cluster_data["Spike_positions"].append(df.values[:,1:])
            df = pd.read_csv(os.path.join(cluster_save_path, "spike_speed"+str(l)+".csv"))
            cluster_data["Spike_speed"].append(df.values[:,1:])
        print("finished reading")
    else:
        behavior_data = ImportClusters.getBehavior(projectPath.folder)
        cluster_data = ImportClusters.getSpikesfromClu(projectPath, behavior_data)

    behavior_data = ImportClusters.getBehavior(projectPath.folder)
    print('Number of clusters:')
    n_clusters = np.sum(
        [np.shape(cluster_data['Spike_labels'][tetrode])[1] for tetrode in range(len(cluster_data['Spike_labels']))])

    print(n_clusters)

    trainer = decodebayes.Trainer(projectPath)
    bayesMatrices = trainer.train(behavior_data,cluster_data)

    outputs = trainer.test(bayesMatrices,behavior_data)

    posProbaPred = outputs["inferring"]
    posTrue = outputs["pos"]


    path_to_code = os.path.join(projectPath.folder,"../../neuroEncoders/transformData")
    predProjPos = transformData.linearizer.doubleArmMazeLinearization(posProbaPred[:,0:2],scale=False,path_to_folder=path_to_code)
    trueProjPos = transformData.linearizer.doubleArmMazeLinearization(posTrue[:, 0:2], scale=False,
                                                                      path_to_folder=path_to_code)

    fig,ax = plt.subplots(2,1)
    ax[0].plot(posProbaPred[:,1],c="orange",label="pred")
    ax[0].plot(posTrue[:,1],c="black",label="true")
    ax[1].plot(posProbaPred[:,2])
    # ax[1].plot(trueProjPos[:,0])
    fig.legend()
    fig.show()

    fig,ax = plt.subplots(2,1)
    ax[0].scatter(posTrue[:,0],posTrue[:,1])
    ax[0].scatter(posProbaPred[:,0],posProbaPred[:,1])
    ax[1].plot(predProjPos[:,0])
    ax[1].plot(trueProjPos[:,0])
    fig.show()


    df = pd.DataFrame(posProbaPred[:,0:2])
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featurePred.csv"))
    df = pd.DataFrame(posTrue)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "featureTrue.csv"))
    df = pd.DataFrame(outputs["times"])
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "timeStepsPred.csv"))
    df = pd.DataFrame(predProjPos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "preojectionPred.csv"))
    df = pd.DataFrame(trueProjPos)
    df.to_csv(os.path.join(projectPath.resultsPath, "resultInference", "preojectionTrue.csv"))


if __name__=="__main__":
    # In this architecture we use a 2.0 tensorflow backend, predicting solely the position.
    # I.E without using the simpler feature strategy based on stratified spaces.

    main()
