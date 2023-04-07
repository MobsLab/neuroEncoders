# Load libs
import os
from time import sleep
import numpy as np
import sys
import math
from functools import reduce
from tqdm import tqdm
# Pykeops
from pykeops.numpy import LazyTensor as LazyTensor_np
import pykeops as pykeops
# Load custom code
from simpleBayes import butils
from importData import import_clusters
from importData.epochs_management import inEpochs
from transformData.linearizer import UMazeLinearizer

# !!!! TODO: all train-test in one function, too much repetition
# TODO: option to remove zero cluster from training and testing

# trainer class
class Trainer():
    def __init__(self, projectPath, bandwidth=None, kernel='gaussian', maskingFactor=20): # 'epanechnikov' - TODO?
        self.projectPath = projectPath
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.maskingFactor = maskingFactor
        self.clusterData  = import_clusters.load_spike_sorting(self.projectPath)
        # Folders
        self.folderResult = os.path.join(self.projectPath.resultsPath, "results")
        if not os.path.isdir(self.folderResult):
            os.makedirs(self.folderResult)

    def train_order_by_pos(self, behaviorData, linearization_function, onTheFlyCorrection=False): # bandwidth=0.05
        # Gather all spikes in large array and sort it in time
        nbSpikes = [a.shape[0] for a in self.clusterData["Spike_labels"]]
        nbNeurons = [a.shape[1] for a in self.clusterData["Spike_labels"]]
        spikeMatLabels = np.zeros([np.sum(nbSpikes), np.sum(nbNeurons)])
        spikeMatTimes = np.zeros([np.sum(nbSpikes), 1])
        cnbSpikes = np.cumsum(nbSpikes)
        cnbNeurons = np.cumsum(nbNeurons)
        for id in range(len(nbSpikes)):
            if id > 0:
                spikeMatLabels[cnbSpikes[id - 1]:cnbSpikes[id], cnbNeurons[id - 1]:cnbNeurons[id]] = \
                    self.clusterData["Spike_labels"][id]
                spikeMatTimes[cnbSpikes[id - 1]:cnbSpikes[id], :] = self.clusterData["Spike_times"][id]
            else:
                spikeMatLabels[0:cnbSpikes[id], 0:cnbNeurons[id]] = self.clusterData["Spike_labels"][id]
                spikeMatTimes[0:cnbSpikes[id], :] = self.clusterData["Spike_times"][id]

        spikeorder = np.argsort(spikeMatTimes[:, 0])
        self.spikeMatLabels = spikeMatLabels[spikeorder, :]
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
                        posX = np.unravel_index(np.argmax(trRateGroup), shape=trRateGroup.shape)
                        preferredPos = preferredPos + [
                            [bayesMatrices["bins"][0][posX[1]], bayesMatrices["bins"][1][posX[0]]]]
        preferredPos = np.array(preferredPos)
        _, linearPreferredPos = linearization_function(preferredPos)
        self.linearPosArgSort = np.argsort(linearPreferredPos)
        self.linearPreferredPos = linearPreferredPos[self.linearPosArgSort]
        self.spikeMatLabels = self.spikeMatLabels[:, self.linearPosArgSort]
        bs = [np.stack(b) for b in bayesMatrices["rateFunctions"]]
        placefields = np.concatenate(bs)
        self.orderdePlaceFields = placefields[self.linearPosArgSort,:]

        return bayesMatrices

    def train(self, behaviorData, onTheFlyCorrection=False):    
        marginalRateFunctions = []
        rateFunctions = []
        spikePositions = []
        mutualInfo = []
        nTetrodes = len(self.clusterData['Spike_labels'])
        maxPos = np.max(
                behaviorData["Positions"][np.logical_not(np.isnan(np.sum(behaviorData["Positions"], axis=1)))])
        
        ### Align the positions time with the spike_times so we can speed filter each spike time (long step)
        posTimes = pykeops.numpy.Vj(behaviorData['positionTime'][:,0][:,None])
        tSpeedFilt = []
        print('Aligning speed-filter with spike times')
        for tetrode in tqdm(range(nTetrodes)):
            spike_times = pykeops.numpy.LazyTensor(self.clusterData['Spike_times'][tetrode][:,0][:,None],axis=0)
            matching_pos_time = (posTimes - spike_times).abs().argmin_reduction(axis=1)
            # matching_pos_time = np.asarray(matching_pos_time)
            speed_mask = behaviorData['Times']['speedFilter'][matching_pos_time]
            tSpeedFilt += [speed_mask]
        # Check for bandwidth
        if self.bandwidth == None:
            self.bandwidth = behaviorData['Bandwidth']
        # Work with position coordinates		
        selPositions = behaviorData['Positions'][reduce(np.intersect1d,
            (np.where(behaviorData['Times']['speedFilter']),
            inEpochs(behaviorData['positionTime'][:,0], behaviorData['Times']['trainEpochs']),
            ))] # Get speed-filtered coordinates from train epoch
        if onTheFlyCorrection: # setting the position to be between 0 and 1 if necessary
            selPositions = selPositions/maxPos
        selPositions = selPositions[np.logical_not(np.isnan(np.sum(selPositions, axis=1))),:] # Remove NaN positions
        ### Build global occupation map
        gridFeature, occupation = butils.kdenD(selPositions, self.bandwidth, kernel=self.kernel) #0.07s
        occupation[occupation==0] = np.min(occupation[occupation!=0])  # We want to avoid having zeros
        mask = occupation > (np.max(occupation)/self.maskingFactor) # Trick to highlight the differences in occupation map
        occupationInverse = 1/occupation
        occupationInverse[occupationInverse==np.inf] = 0
        occupationInverse = np.multiply(occupationInverse, mask)
        finalOccupation = occupationInverse
        # occupationInverse[occupationInverse==0] = 10/np.min(occupation)
        # finalOccupation = 1/occupationInverse
        ### Build marginal rate functions
        print('Building marginal rate and local rate functions')
        for tetrode in tqdm(range(nTetrodes)):
            tetrodewisePos = self.clusterData['Spike_positions'][tetrode][reduce(np.intersect1d,
                (np.where(tSpeedFilt[tetrode]),
                inEpochs(self.clusterData['Spike_times'][tetrode][:,0], behaviorData['Times']['trainEpochs'])))]
            if onTheFlyCorrection: # setting the position to be between 0 and 1 if necessary
                tetrodewisePos = tetrodewisePos/maxPos
            tetrodewisePos = tetrodewisePos[np.logical_not(np.isnan(np.sum(tetrodewisePos, axis=1))), :] 
            gridFeature, MRF = butils.kdenD(tetrodewisePos, self.bandwidth, edges=gridFeature, kernel=self.kernel)
            MRF[MRF==0] = np.min(MRF[MRF!=0])
            MRF         = MRF/np.sum(MRF)
            MRF         = np.shape(tetrodewisePos)[0]*np.multiply(MRF, finalOccupation)/behaviorData['Times']['learning']
            marginalRateFunctions.append(MRF)
            # Allocate for local rate functions
            localRateFunctions = []
            localSpikePositions = []
            localMutualInfo = []
        ### Build local rate functions (one per cluster)
            for label in range(np.shape(self.clusterData['Spike_labels'][tetrode])[1]):
                clusterwisePos = self.clusterData['Spike_positions'][tetrode][reduce(np.intersect1d,
                    (np.where(tSpeedFilt[tetrode]),
                    np.where(self.clusterData['Spike_labels'][tetrode][:,label] == 1),
                    inEpochs(self.clusterData['Spike_times'][tetrode][:,0], behaviorData['Times']['trainEpochs'])))]
                if onTheFlyCorrection:
                    clusterwisePos = clusterwisePos / maxPos
                clusterwisePos = clusterwisePos[np.logical_not(np.isnan(np.sum(clusterwisePos, axis=1))), :]
                if np.shape(clusterwisePos)[0]!=0:
                    gridFeature, LRF = butils.kdenD(clusterwisePos, self.bandwidth,
                                                    edges=gridFeature, kernel=self.kernel)
                    LRF[LRF==0] = np.min(LRF[LRF!=0])
                    LRF         = LRF/np.sum(LRF)
                    LRF         = np.shape(clusterwisePos)[0]*np.multiply(LRF, finalOccupation)/behaviorData['Times']['learning']
                    localRateFunctions.append(LRF)
                else:
                    localRateFunctions.append(np.ones(np.shape(occupation)))
                localSpikePositions.append(clusterwisePos)
                #Let us compute the mutual information with the positions:
                LRF = localRateFunctions[-1]
                mi = np.sum(occupation[LRF>0]*LRF[LRF>0]/(np.mean(LRF))*np.log(LRF[LRF>0]/(np.mean(LRF)))/np.log(2))
                localMutualInfo.append(mi)
            rateFunctions.append(localRateFunctions)
            spikePositions.append(localSpikePositions)
            mutualInfo.append(localMutualInfo)

        bayesMatrices = {'occupation': occupation, 'marginalRateFunctions': marginalRateFunctions, 'rateFunctions': rateFunctions,
                'bins':[np.unique(gridFeature[i]) for i in range(len(gridFeature))],'spikePositions': spikePositions,
                          'mutualInfo': mutualInfo}
        return bayesMatrices

    def test_as_NN(self, behaviorData, bayesMatrices, timeStepPred, windowSizeMS=36, useTrain=False, sleepEpochs=[]):
        windowSize = windowSizeMS/1000
     
        if useTrain:
            clustersTime = [self.clusterData['Spike_times'][tetrode][
                                         inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                  behaviorData['Times']['trainEpochs'])]
                                     for tetrode in range(len(self.clusterData['Spike_times']))]
            clusters = [self.clusterData['Spike_labels'][tetrode][
                                    inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                             behaviorData['Times']['trainEpochs'])]
                                for tetrode in range(len(self.clusterData['Spike_times']))]
        else:
            if len(sleepEpochs)>0:
                clustersTime = [self.clusterData['Spike_times'][tetrode][
                                             inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                      sleepEpochs)]
                                         for tetrode in range(len(self.clusterData['Spike_times']))]
                clusters = [self.clusterData['Spike_labels'][tetrode][
                                        inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                 sleepEpochs)]
                                    for tetrode in range(len(self.clusterData['Spike_times']))]
            else:
                clustersTime = [self.clusterData['Spike_times'][tetrode][
                                             inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                      behaviorData['Times']['testEpochs'])]
                                         for tetrode in range(len(self.clusterData['Spike_times']))]
                clusters = [self.clusterData['Spike_labels'][tetrode][
                                        inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                 behaviorData['Times']['testEpochs'])]
                                    for tetrode in range(len(self.clusterData['Spike_times']))]
    
        print('\nBUILDING POSITION PROBAS')
        occupation, marginalRateFunctions, rateFunctions = [bayesMatrices[key] for key in
                                                        ['occupation', 'marginalRateFunctions',
                                                                'rateFunctions']]
        mask = occupation > (np.max(occupation) / self.maskingFactor)
        ### Build Poisson term
        allPoisson = [np.exp((-windowSize) * marginalRateFunctions[tetrode]) for tetrode in
                            range(len(clusters))]
        allPoisson = reduce(np.multiply, allPoisson)
        ### Log of rate functions
        logRF = []
        for tetrode in range(np.shape(rateFunctions)[0]):
            temp = []
            for cluster in range(np.shape(rateFunctions[tetrode])[0]):
                temp.append(np.log(rateFunctions[tetrode][cluster] + np.min(
                    rateFunctions[tetrode][cluster][rateFunctions[tetrode][cluster] != 0])))
            logRF.append(temp)

        ### Decoding loop
        print("Parallel pykeops bayesian test")
        outputPOps = parallel_pred_as_NN(timeStepPred, windowSize, allPoisson, clusters,
                                                clustersTime, logRF, occupation)
        print("finished bayesian guess")

        idPos = np.unravel_index(outputPOps[1], shape=allPoisson.shape)
        inferredPos = np.array(
            [bayesMatrices['bins'][i][idPos[i][:, 0]] for i in range(len(bayesMatrices['bins']))])
        inferredProba = outputPOps[0]
        inferResults = np.concatenate([np.transpose(inferredPos), inferredProba], axis=-1)

        # NOTE: A few values of probability predictions present NaN in pykeops....
        print("Resolving nan issue from pykeops over a few bins")
        badBins = np.where(np.isnan(inferResults[:, 2]))[0]
        for bin in badBins:
            binStartTime = timeStepPred[bin]
            binStopTime = binStartTime + windowSize
            tetrodesContributions = []
            tetrodesContributions.append(allPoisson)
            for tetrode in range(len(clusters)):
                binProbas = clusters[tetrode][np.intersect1d(
                    np.where(clustersTime[tetrode][:, 0] > binStartTime),
                    np.where(clustersTime[tetrode][:, 0] < binStopTime))]
                # Note:  we would lose some spikes if we used the cluster_data[Spike_pos_index]
                # because some spike might be closest to one position further away than windowSize, 
                # yet themselves be close to the spike time
                binClusters = np.sum(binProbas, 0)
                # Terms that come from spike information
                if np.sum(binClusters) > 0.5:
                    spikePattern = reduce(np.multiply,
                                           [np.exp(logRF[tetrode][cluster] * binClusters[cluster])
                                            for cluster in range(np.shape(binClusters)[0])])
                else:
                    spikePattern = np.multiply(np.ones(np.shape(occupation)), mask)
                tetrodesContributions.append(spikePattern)
            # Guessed probability map
            positionProba = reduce(np.multiply, tetrodesContributions)
            positionProba = np.multiply(positionProba, occupation) #prior: Occupation deduced from training!!
            positionProba = positionProba / np.sum(positionProba)
            inferResults[bin, 2] = np.max(positionProba)
            inferResults[np.isnan(inferResults[:, 2]), 2] = 0 # to correct for overflow

        # Get the true position
        idTestEpoch = inEpochs(behaviorData["positionTime"][:,0],
                               behaviorData['Times']['testEpochs'])
        realPos = behaviorData["Positions"][idTestEpoch]
        realTimes = behaviorData["positionTime"][idTestEpoch]
        idsNN = []
        for timeStamp in timeStepPred:
            idsNN.append(np.abs(realTimes - timeStamp).argmin())
        idsNN = np.array(idsNN)
        featTrue = realPos[idsNN]

        outputResults = {"featurePred": inferResults[:, :2], 'proba': inferResults[:, 2], 'times': timeStepPred, 'featureTrue': featTrue,
                                           'speed_mask': behaviorData["Times"]["speedFilter"]}  # , "probaMaps": position_proba
        
        return outputResults

    def sleep_decoding(self, behaviorData, bayesMatrices, windowSizeMS=36):
        windowSize = windowSizeMS/1000

        clustersTime = {}
        clusters = {}
        inferResultsDic = {}
        for id,sleepName in enumerate(behaviorData["Times"]["sleepNames"]):
            clustersTime[sleepName] = [self.clusterData['Spike_times'][tetrode][
                                           inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                    behaviorData['Times']['sleepEpochs'][2*id:2*id+2])]
                                       for tetrode in range(len(self.clusterData['Spike_times']))]
            clusters[sleepName] = [self.clusterData['Spike_labels'][tetrode][
                                       inEpochs(self.clusterData['Spike_times'][tetrode][:, 0],
                                                behaviorData['Times']['sleepEpochs'][2*id:2*id+2])]
                                   for tetrode in range(len(self.clusterData['Spike_times']))]

            print('\nBUILDING POSITION PROBAS')
            occupation, marginalRateFunctions, rateFunctions = [bayesMatrices[key] for key in
                                                                ['occupation', 'marginalRateFunctions',
                                                                 'rateFunctions']]
            mask = occupation > (np.max(occupation) / self.maskingFactor)

            ### Build Poisson term
            allPoisson = [np.exp((-windowSize) * marginalRateFunctions[tetrode]) for tetrode in
                          range(len(clusters[sleepName]))]
            allPoisson = reduce(np.multiply, allPoisson)

            ### Log of rate functions
            logRF = []
            for tetrode in range(np.shape(rateFunctions)[0]):
                temp = []
                for cluster in range(np.shape(rateFunctions[tetrode])[0]):
                    temp.append(np.log(rateFunctions[tetrode][cluster] + np.min(
                        rateFunctions[tetrode][cluster][rateFunctions[tetrode][cluster] != 0])))
                logRF.append(temp)

            timeStepPred = behaviorData["positionTime"][
                inEpochs(behaviorData["positionTime"][:, 0], behaviorData['Times']['sleepEpochs'][2*id:2*id+2])]
            # For prediction time step we use the time step of measured speed and feature of the animal.
            # A prediction is made over all time steps in the test set
            # and the prediction results can be latter on filter out.

            print("Parallel pykeops bayesian test")
            outputPOps = parallel_pred_as_NN(timeStepPred, windowSize, allPoisson, clusters[sleepName],
                                             clustersTime[sleepName], logRF, occupation)
            print("finished bayesian guess")

            idPos = np.unravel_index(outputPOps[1], shape=allPoisson.shape)
            inferredPos = np.array([bayesMatrices['bins'][i][idPos[i][:, 0]] for i in
                                    range(len(bayesMatrices['bins']))])
            inferredProba = outputPOps[0]
            inferResults = np.concatenate([np.transpose(inferredPos), inferredProba], axis=-1)

            # NOTE: A few values of probability predictions present NaN in pykeops....
            print("Resolving nan issue from pykeops over a few bins")
            badBins = np.where(np.isnan(inferResults[:, 2]))[0]
            for bin in badBins:
                binStartTime = timeStepPred[bin] - windowSize / 2
                binStopTime = binStartTime + windowSize / 2
                tetrodesContributions = []
                tetrodesContributions.append(allPoisson)
                for tetrode in range(len(clusters[sleepName])):
                    binProbas = clusters[sleepName][tetrode][np.intersect1d(
                        np.where(clustersTime[sleepName][tetrode][:, 0] > binStartTime),
                        np.where(clustersTime[sleepName][tetrode][:, 0] < binStopTime))]
                    # Note:  we would lose some spikes if we used the cluster_data[Spike_pos_index]
                    # because some spike might be closest to one position further away than windowSize,
                    # yet themselves be close to the spike time
                    binClusters = np.sum(binProbas, 0)
                    # Terms that come from spike information
                    if np.sum(binClusters) > 0.5:
                        spikePattern = reduce(np.multiply,
                                              [np.exp(logRF[tetrode][cluster] * binClusters[cluster])
                                               for cluster in range(np.shape(binClusters)[0])])
                    else:
                        spikePattern = np.multiply(np.ones(np.shape(occupation)), mask)
                    tetrodesContributions.append(spikePattern)
                # Guessed probability map
                positionProba = reduce(np.multiply, tetrodesContributions)
                positionProba = np.multiply(positionProba, occupation)  # prior: Occupation deduced from training!!
                positionProba = positionProba / np.sum(positionProba)
                inferResults[bin, 2] = np.max(positionProba)

            inferResultsDic[sleepName] = {"featurePred": inferResults[:, :2], 'proba': inferResults[:, 2],
                                          'times': timeStepPred, 'speed_mask': behaviorData["Times"]["speedFilter"]}

        return inferResultsDic

    def calculate_linear_tuning_curve(self, linearization_function, behaviorData):
        linearPlaceFields = []
        # Create one large epoch that comprises both train and test dataset
        minTime = np.min(np.concatenate((behaviorData['Times']['trainEpochs'],
                                         behaviorData['Times']['testEpochs'])))
        maxTime = np.max(np.concatenate((behaviorData['Times']['trainEpochs'],
                                         behaviorData['Times']['testEpochs'])))
        epochForField = np.array([minTime, maxTime])
        linearTraj = linearization_function(behaviorData['Positions'])[1]
        timesMask = inEpochs(np.squeeze(behaviorData['positionTime']), epochForField)[0]
        timeLinear = np.squeeze(behaviorData['positionTime'][timesMask, :])
        linearTraj = linearTraj[timesMask]
        linSpace = np.arange(min(linearTraj), max(linearTraj), step=0.01)
        histPos, binEdges = np.histogram(linearTraj, bins=linSpace)

        for tetrode in range(len(self.clusterData['Spike_times'])):
            spikeTimesTetrode = np.squeeze(self.clusterData['Spike_times'][tetrode])
            for icell in range(self.clusterData['Spike_labels'][tetrode].shape[1]):
                cellMask = self.clusterData['Spike_labels'][tetrode][:, icell] == 1
                spikeTimes = spikeTimesTetrode[cellMask]
                spikeTimes = spikeTimes[inEpochs(spikeTimes, epochForField)]

                # Find position of the animal at the time of each spike
                if spikeTimes.any():
                    spikeTimesLazy = pykeops.numpy.Vj(spikeTimes[:,None].astype(dtype=np.float64))
                    timeLinearLazy = pykeops.numpy.Vi(timeLinear[:,None].astype(dtype=np.float64))
                    idPosInSpikes = ((spikeTimesLazy - timeLinearLazy).abs().argmin(axis=0))[:, 0]
                    spikePos = linearTraj[idPosInSpikes]
                    # Create histogram of spike position (find P(spikes|position))
                    histSpikes, binEdges = np.histogram(spikePos, bins=linSpace)
                    # Find tuning curve
                    histTuning = histSpikes / histPos
                    linearPlaceFields.append(histTuning) # save
                else:
                    linearPlaceFields.append(np.zeros(len(linSpace) - 1))

        return linearPlaceFields, binEdges


############## Utils ##############
def find_next_bin(times,clusters,start,stop,start_time,stop_time):
    # times: array of times
    # start: last start index
    # stop: last stop index
    # start_time: bin start time
    # stop_time: bin stop time
    newStartID = stop
    while times[newStartID]<start_time:
        newStartID += 1
    newStopId = newStartID+1
    while times[newStopId]<stop_time and newStopId<=len(times)-1:
        newStopId +=1
    newStopId = newStopId-1
    return newStartID,newStopId,clusters[newStartID:newStopId+1]

def parallel_pred_as_NN(firstSpikeNNtime, windowSize, allPoisson, clusters, clustersTime, logRF, occupancy):
    # Use pykeops library to perform an efficient computation of the predicted position, in parallel over all bins.
    # Note: here achieved on the CPU, could also be ported to the GPU by using torch tensor....

    binStartTime = firstSpikeNNtime
    binStopTime = binStartTime + windowSize

    # we will progressively add each tetrode contribution
    tetrodeContribs = 1
    for tetrode in range(len(clusters)):

        gctLazy = LazyTensor_np(clustersTime[tetrode][:,None])
        binStartTimeLazy = LazyTensor_np(binStartTime[None,:])
        binStopTimeLazy = LazyTensor_np(binStopTime[None,:])
        goodStart = (gctLazy - binStartTimeLazy).relu().sign() # similar to gct_lazy > bin_start_times.lazy
        goodStop = (binStopTimeLazy - gctLazy).relu().sign()
        # size: (Number of signal time step,Number of prediction bin,1), indicate for each bin the time step in the bin.
        goodBins = goodStart * goodStop
        gcLazy = LazyTensor_np(clusters[tetrode][:,None,:])
        # For each bin, we gather for each cluster in the tetrode the number of spike detected in signal measurements inside this bin.
        # gathering can be effectively implemented by a element wise matrix multiplication with the mask good_bins
        binClusters = (gcLazy * goodBins).sum(axis=0)
        # # transform into an array of size (Nb bin,Nb cluster in tetrode)

        # Prepare for pykeops operations:
        logRF_r = np.transpose(np.array(logRF[tetrode]), axes=[1, 2, 0])
        logRF_r = np.reshape(logRF_r,newshape=[np.prod(logRF_r.shape[0:-1]),logRF_r.shape[-1]])
        logRFLazy = LazyTensor_np(logRF_r[None,:,:])
        binClustersLazy = LazyTensor_np(binClusters[:,None,:])

        # the Log firing rate of each cluster is multiplied by the number of bin cluster, and the sum is performed over the
        # number of cluster in the tetrode
        res = (logRFLazy * binClustersLazy).sum(dim=-1).exp()
        tetrodeContribs = tetrodeContribs*res

    #Finally we need to add the Poisson terms common to all tetrode finalS
    # position posterior estimation:
    poisson_r = np.reshape(allPoisson,newshape=[np.prod(allPoisson.shape)])[:,None]
    poissonContribVj = pykeops.numpy.Vj(poisson_r)
    tetrodeContribs = tetrodeContribs*poissonContribVj

    # The probability need to be weighted by the position probabilities:
    occupancy_r = np.reshape(occupancy,newshape=[np.prod(occupancy.shape)])[:,None]
    occupancyContrib = pykeops.numpy.Vj(occupancy_r)
    tetrodeContribs = tetrodeContribs * occupancyContrib

    # If we had only one electrode:
    # ... but we need to sum over the different electrodes.
    outputPos = tetrodeContribs.max_argmax_reduction(axis=1)
    # We also need to normalize the probability:
    sumProba = tetrodeContribs.sum_reduction(axis=1)
    outputPos = (outputPos[0]/sumProba,outputPos[1])

    return outputPos
############## Utils ##############



############## Legacy trainer ##############
class LegacyTrainer():
    def __init__(self, projectPath, bandwidth=None, kernel='gaussian', masking_factor=20): # 'epanechnikov' - TODO?
        self.projectPath = projectPath
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.masking_factor = masking_factor
        self.cluster_data  = import_clusters.load_spike_sorting(self.projectPath)

    def train(self, behavior_data, onTheFlyCorrection=False):
        
        Marginal_rate_functions = []
        Rate_functions = []
        Spike_positions = []
        Mutual_info = []
        n_tetrodes = len(self.cluster_data['Spike_labels'])
        maxPos = np.max(
                behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
        ### Align the positions time with the spike_times so we can speed filter each spike time (long step)
        pos_times = pykeops.numpy.Vj(behavior_data['positionTime'][:,0][:,None])
        tetrode_speed_filter_spiketimes = []
        print('Aligning speed-filter with spike times')
        for tetrode in tqdm(range(n_tetrodes)):
            spike_times = pykeops.numpy.LazyTensor(self.cluster_data['Spike_times'][tetrode][:,0][:,None],axis=0)
            matching_pos_time = (pos_times - spike_times).abs().argmin_reduction(axis=1)
            speed_mask = behavior_data['Times']['speedFilter'][matching_pos_time]
            tetrode_speed_filter_spiketimes += [speed_mask]
  
        # Check for bandwidth
        if self.bandwidth == None:
            self.bandwidth = behavior_data['Bandwidth']
        # Work with position coordinates		
        selected_positions = behavior_data['Positions'][reduce(np.intersect1d,
            (np.where(behavior_data['Times']['speedFilter']),
            inEpochs(behavior_data['positionTime'][:,0], behavior_data['Times']['trainEpochs']),
            ))] # Get speed-filtered coordinates from train epoch
        if onTheFlyCorrection: # setting the position to be between 0 and 1 if necessary
            selected_positions = selected_positions/maxPos
        selected_positions = selected_positions[np.logical_not(np.isnan(np.sum(selected_positions,axis=1))),:] # Remove NaN positions

          ### Build global occupation map
        # xEdges, yEdges, Occupation = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, kernel=self.kernel)
        gridFeature, Occupation = butils.kdenD(selected_positions, self.bandwidth,kernel=self.kernel) #0.07s
        Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros
        mask = Occupation > (np.max(Occupation)/self.masking_factor) # Trick to highlight the differences in occupation map
        Occupation_inverse = 1/Occupation
        Occupation_inverse[Occupation_inverse==np.inf] = 0
        Occupation_inverse = np.multiply(Occupation_inverse, mask)

        ### Build marginal rate functions
        print('Building marginal rate and local rate functions')
        for tetrode in tqdm(range(n_tetrodes)):
            tetrodewisePos = self.cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
                (np.where(tetrode_speed_filter_spiketimes[tetrode]),
                inEpochs(self.cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
            if onTheFlyCorrection: # setting the position to be between 0 and 1 if necessary
                tetrodewisePos = tetrodewisePos/maxPos
            tetrodewisePos = tetrodewisePos[np.logical_not(np.isnan(np.sum(tetrodewisePos, axis=1))), :] # Remove NaN: i:e problem with feature recording
            gridFeature, MRF = butils.kdenD(tetrodewisePos, self.bandwidth, edges=gridFeature, kernel=self.kernel)
            MRF[MRF==0] = np.min(MRF[MRF!=0])
            MRF         = MRF/np.sum(MRF)
            MRF         = np.shape(tetrodewisePos)[0]*np.multiply(MRF, Occupation_inverse)/behavior_data['Times']['learning']
            Marginal_rate_functions.append(MRF)
            # Allocate for local rate functions
            Local_rate_functions = []
            Local_Spike_positions = []
            LocalMutualInfo = []

        ### Build local rate functions (one per cluster)
            for label in range(np.shape(self.cluster_data['Spike_labels'][tetrode])[1]):
                clusterwisePos = self.cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
                    (np.where(tetrode_speed_filter_spiketimes[tetrode]),
                    np.where(self.cluster_data['Spike_labels'][tetrode][:,label] == 1),
                    inEpochs(self.cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
                if onTheFlyCorrection:
                    clusterwisePos = clusterwisePos / maxPos
                clusterwisePos = clusterwisePos[np.logical_not(np.isnan(np.sum(clusterwisePos, axis=1))), :]
                if np.shape(clusterwisePos)[0]!=0:
                    gridFeature, LRF = butils.kdenD(clusterwisePos, self.bandwidth,
                                                    edges=gridFeature, kernel=self.kernel)
                    LRF[LRF==0] = np.min(LRF[LRF!=0])
                    LRF         = LRF/np.sum(LRF)
                    LRF         = np.shape(clusterwisePos)[0]*np.multiply(LRF, Occupation_inverse)/behavior_data['Times']['learning']
                    Local_rate_functions.append(LRF)
                else:
                    Local_rate_functions.append(np.ones(np.shape(Occupation)))
                Local_Spike_positions.append(clusterwisePos)
                #Let us compute the mutual information with the positions:
                LRF = Local_rate_functions[-1]
                mutualInfo = np.sum(Occupation[LRF>0]*LRF[LRF>0]/(np.mean(LRF))*np.log(LRF[LRF>0]/(np.mean(LRF)))/np.log(2))
                LocalMutualInfo.append(mutualInfo)

            Rate_functions.append(Local_rate_functions)
            Spike_positions.append(Local_Spike_positions)
            Mutual_info.append(LocalMutualInfo)

        bayes_matrices = {'Occupation': Occupation, 'Marginal rate functions': Marginal_rate_functions, 'Rate functions': Rate_functions,
                'Bins':[np.unique(gridFeature[i]) for i in range(len(gridFeature))],'Spike_positions':Spike_positions,
                          'Mutual_info': Mutual_info}
        return bayes_matrices

    def test(self, bayes_matrices, behavior_data, windowSize=36):
        windowSize = windowSize/1000

        print('\nBUILDING POSITION PROBAS');
        guessed_clusters_time = [self.cluster_data['Spike_times'][tetrode][
                                         inEpochs(self.cluster_data['Spike_times'][tetrode][:, 0],
                                                  behavior_data['Times']['testEpochs'])]
                                 for tetrode in range(len(self.cluster_data['Spike_times']))]
        guessed_clusters= [self.cluster_data['Spike_labels'][tetrode][
                                                          inEpochs(self.cluster_data['Spike_times'][tetrode][:, 0],
                                                                   behavior_data['Times']['testEpochs'])]
                            for tetrode in range(len(self.cluster_data['Spike_times']))]

        Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in ['Occupation','Marginal rate functions','Rate functions']]
        mask = Occupation > (np.max(Occupation)/self.masking_factor)
  
        ### Build Poisson term
        # first we bin the time
        testEpochs = behavior_data['Times']['testEpochs']
        Ttest = np.sum([testEpochs[2 * i + 1] - testEpochs[2 * i] for i in range(len(testEpochs) // 2)])
        n_bins = math.floor(Ttest/windowSize)
        # for each bin we will need to now the test epoch it belongs to, so that we can then
        # set the time correctly to select the corresponding spikes
        timeEachTestEpoch = [testEpochs[2 * i + 1] - testEpochs[2 * i] for i in range(len(testEpochs) // 2)]
        cumTimeEachTestEpoch = np.cumsum(timeEachTestEpoch)
        cumTimeEachTestEpoch = np.concatenate([[0],cumTimeEachTestEpoch])
        # a function that given the bin indicates the bin index:
        binToEpoch = lambda x : np.where(((x*windowSize - cumTimeEachTestEpoch[0:-1])>=0)*((x*windowSize - cumTimeEachTestEpoch[1:])<0))[0][0]
        binToEpochArray = [binToEpoch(bins) for bins in range(n_bins)]
        firstBinEpoch = [np.min(np.where(np.equal(binToEpochArray,epochId))[0]) for epochId in range(len(timeEachTestEpoch))]
        All_Poisson_term = [np.exp((-windowSize)*Marginal_rate_functions[tetrode]) for tetrode in range(len(guessed_clusters))]
        All_Poisson_term = reduce(np.multiply, All_Poisson_term)

        ### Log of rate functions
        log_RF = []
        for tetrode in range(np.shape(Rate_functions)[0]):
            temp = []
            for cluster in range(np.shape(Rate_functions[tetrode])[0]):
                temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster]!=0])))
            log_RF.append(temp)

        ### Decoding loop
        position_proba = [np.ones(np.shape(Occupation))] * n_bins
        position_true = [np.ones(2)] * n_bins
        nSpikes = []
        times = []
        for bin in range(n_bins):

            #Trouble: the test Epochs is discretized in continuous bin
            # whereas we forbid the use of some time steps b filtering them according to speed.
            bin_start_time = testEpochs[2*binToEpoch(bin)] + (bin-firstBinEpoch[binToEpoch(bin)])*windowSize
            bin_stop_time = bin_start_time + windowSize
            times.append(bin_start_time)

            binSpikes = 0
            tetrodes_contributions = []
            tetrodes_contributions.append(All_Poisson_term)

            for tetrode in range(len(guessed_clusters)):
                # Clusters inside our window
                bin_probas = guessed_clusters[tetrode][np.intersect1d(
                    np.where(guessed_clusters_time[tetrode][:,0] > bin_start_time),
                    np.where(guessed_clusters_time[tetrode][:,0] < bin_stop_time))]
                bin_clusters = np.sum(bin_probas, 0)
                binSpikes = binSpikes + np.sum(bin_clusters)


                # Terms that come from spike information
                if np.sum(bin_clusters) > 0.5:
                    spike_pattern = reduce(np.multiply,
                        [np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
                        for cluster in range(np.shape(bin_clusters)[0])])
                else:
                    spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

                tetrodes_contributions.append(spike_pattern)

            nSpikes.append(binSpikes)

            # Guessed probability map
            position_proba[bin] = reduce(np.multiply, tetrodes_contributions)
            position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])
            # True position
            position_true_mean = np.nanmean(behavior_data['Positions'][reduce(np.intersect1d,
                (np.where(behavior_data['positionTime'][:,0] > bin_start_time),
                np.where(behavior_data['positionTime'][:,0] < bin_stop_time)))], axis=0 )
            position_true[bin] = position_true[bin-1] if np.isnan(position_true_mean).any() else position_true_mean

            if bin % 50 == 0:
                sys.stdout.write('[%-30s] : %.3f %%' % ('='*(bin*30//n_bins),bin*100/n_bins))
                sys.stdout.write('\r')
                sys.stdout.flush()
        sys.stdout.write('[%-30s] : %.3f %%' % ('='*((bin+1)*30//n_bins),(bin+1)*100/n_bins))
        sys.stdout.write('\r')
        sys.stdout.flush()

        position_true[0] = position_true[1]
        print('\nDecoding finished')

        # Guessed X and Y
        allProba = [np.unravel_index(np.argmax(position_proba[bin]),position_proba[bin].shape) for bin in range(len(nSpikes))]
        bestProba = [np.max(position_proba[bin]) for bin in range(len(nSpikes))]
        position_guessed = [ [bayes_matrices['Bins'][i][allProba[bin][i]] for i in range(len(bayes_matrices['Bins']))]
                             for bin in range(len(nSpikes)) ]
        inferResults = np.concatenate([np.array(position_guessed),np.array(bestProba).reshape([-1,1])],axis=-1)

        outputResults = {"inferring":inferResults, "pos": np.array(position_true), "probaMaps": position_proba, "times":np.array(times),
                   'nSpikes': np.array(nSpikes)}
        return outputResults

    def full_proba_decoding(self, behavior_data, bayes_matrices, timeStepPred, windowSize=36, useTrain=True):
        windowSize = windowSize/1000

        if useTrain:
            guessed_clusters_time = [self.cluster_data['Spike_times'][tetrode][
                                         inEpochs(self.cluster_data['Spike_times'][tetrode][:, 0],
                                                  behavior_data['Times']['trainEpochs'])]
                                     for tetrode in range(len(self.cluster_data['Spike_times']))]
            guessed_clusters = [self.cluster_data['Spike_labels'][tetrode][
                                    inEpochs(self.cluster_data['Spike_times'][tetrode][:, 0],
                                             behavior_data['Times']['trainEpochs'])]
                                for tetrode in range(len(self.cluster_data['Spike_times']))]
        else:
            guessed_clusters_time = [self.cluster_data['Spike_times'][tetrode][
                                         inEpochs(self.cluster_data['Spike_times'][tetrode][:, 0],
                                                  behavior_data['Times']['testEpochs'])]
                                     for tetrode in range(len(self.cluster_data['Spike_times']))]
            guessed_clusters = [self.cluster_data['Spike_labels'][tetrode][
                                    inEpochs(self.cluster_data['Spike_times'][tetrode][:, 0],
                                             behavior_data['Times']['testEpochs'])]
                                for tetrode in range(len(self.cluster_data['Spike_times']))]
        # print('\nBUILDING POSITION PROBAS')
        Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in
                                                               ['Occupation', 'Marginal rate functions',
                                                                'Rate functions']]
        mask = Occupation > (np.max(Occupation) / self.masking_factor)

        ### Build Poisson term
        All_Poisson_term = [np.exp((-windowSize) * Marginal_rate_functions[tetrode]) for tetrode in
                            range(len(guessed_clusters))]
        All_Poisson_term = reduce(np.multiply, All_Poisson_term)

        ### Log of rate functions
        log_RF = []
        for tetrode in range(np.shape(Rate_functions)[0]):
            temp = []
            for cluster in range(np.shape(Rate_functions[tetrode])[0]):
                temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(
                    Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster] != 0])))
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
                bin_probas = guessed_clusters[tetrode][np.intersect1d(
                    np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
                    np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time))]
                # Note:  we would lose some spikes if we used the cluster_data[Spike_pos_index]
                # because some spike might be closest to one position further away than windowSize, yet themselves be close to the spike time
                bin_clusters = np.sum(bin_probas, 0)
                # Terms that come from spike information
                if np.sum(bin_clusters) > 0.5:
                    spike_pattern = reduce(np.multiply,
                                           [np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
                                            for cluster in range(np.shape(bin_clusters)[0])])
                else:
                    spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)
                tetrodes_contributions.append(spike_pattern)
            # Guessed probability map
            position_proba = reduce(np.multiply, tetrodes_contributions)
            position_proba = np.multiply(position_proba, Occupation)
            position_proba = position_proba / np.sum(position_proba)
            position_probas.append(position_proba)
        return position_probas
