
# Load libs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import  LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from importData.rawdata_parser import get_params
from importData.epochs_management import inEpochsMask

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#ffffff'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

class PaperFigures():
    def __init__(self, projectPath, behaviorData, trainerBayes, l_function, bayesMatrices={}, timeWindows=[36]): # I am here
        self.projectPath = projectPath
        self.trainerBayes = trainerBayes
        self.behaviorData = behaviorData
        self.l_function = l_function
        self.bayesMatrices = bayesMatrices
        self.timeWindows = timeWindows
        _, self.samplingRate, _ = get_params(self.projectPath.xml)
        
        self.binsLinearPosHist = np.arange(0, stop=1, step=0.01) # discretisation of the linear variable to help in some plots
        self.cm = plt.get_cmap("tab20b")
        # Manage folders
        self.folderFigures = os.path.join(self.projectPath.resultsPath, 'figures')
        if not os.path.exists(self.folderFigures):
            os.mkdir(self.folderFigures)
        self.folderAligned = os.path.join(self.projectPath.dataPath, 'aligned')

    def load_data(self):
        # CSV files helping to align the pop vector from spike used in spike sorting
        # with predictions from spike used by the NN are also provided.
        sspikesTrain = []
        swspikesTrain = []
        sspikesTest = []
        swspikesTest = []

        for ws in self.timeWindows:
            sspikesTrain.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned, str(ws), "train", "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32))
            swspikesTrain.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned, str(ws), "train", "startTimeWindow.csv")).values[:,1:],dtype=np.float32))
            sspikesTest.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned ,str(ws), "test", "spikeMat_window_popVector.csv")).values[:,1:],dtype=np.float32))
            swspikesTest.append(np.array(
                pd.read_csv(os.path.join(self.folderAligned ,str(ws), "test", "startTimeWindow.csv")).values[:,1:],dtype=np.float32))


        ### Load the NN prediction without using noise:
        lPredPos = []
        fPredPos = []
        truePos = []
        time = []
        lossPred = []
        speedMask = []
        for ws in self.timeWindows:
            lPredPos.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results", str(ws), "linearPred.csv")).values[:,1:],dtype=np.float32)))
            fPredPos.append(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "featurePred.csv")).values[:, 1:],dtype=np.float32))
            truePos.append(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "featureTrue.csv")).values[:, 1:],dtype=np.float32))
            time.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "timeStepsPred.csv")).values[:, 1:],dtype=np.float32)))
            lossPred.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "lossPred.csv")).values[:, 1:],dtype=np.float32)))
            speedMask.append(np.squeeze(np.array(pd.read_csv(
                os.path.join(self.projectPath.resultsPath, "results",str(ws), "speedMask.csv")).values[:, 1:],dtype=np.float32)))

        speedMask = [ws.astype(np.bool) for ws in speedMask]
        lTruePos = [self.l_function(f)[1] for f in truePos]
        
        # Output
        self.spikesAligned = {
            'spikesTrain': sspikesTrain, 'windowsTrain': swspikesTrain, 'spikesTest': sspikesTest, 'windowsTest': swspikesTest
        }
        self.resultsNN = {
            'time': time, 'speedMask': speedMask, 'linPred': lPredPos, 'fullPred': fPredPos, 'truePos': truePos, 'linTruePos': lTruePos, 'predLoss': lossPred
        }
        
        return self.spikesAligned, self.resultsNN

    def test_bayes(self):

        if not hasattr(self.trainerBayes, 'linearPreferredPos') and not 'Occupation' in self.bayesMatrices.keys():
            self.bayesMatrices = self.trainerBayes.train_order_by_pos(self.behaviorData, self.l_function)

        #quickly obtain bayesian decoding:
        lPredPosBayes = []
        probaBayes = []
        fPredBayes = []
        for i,ws in enumerate(self.timeWindows):
            outputsBayes = self.trainerBayes.test_as_NN(self.behaviorData, self.bayesMatrices, # weird time - check
                self.spikesAligned['windowsTest'][i][:self.resultsNN['time'][i].shape[0]].astype(dtype=np.float64) / self.samplingRate, windowSizeMS=ws)  
            infPos = outputsBayes["featurePred"][:, 0:2]
            _, linearBayesPos = self.l_function(infPos)
            
            lPredPosBayes.append(linearBayesPos)
            fPredBayes.append(infPos)
            probaBayes.append(outputsBayes["proba"]) 
        self.resultsBayes = {
            'linPred': lPredPosBayes, 'fullPred': fPredBayes, 'probaBayes': probaBayes, 'time': self.resultsNN['time']
        }
            
        return self.resultsBayes
           
    def fig_example_linear(self):
        ## Figure 1: on habituation set, speed filtered, we plot an example of bayesian and neural network decoding
        # ANN results
        fig,ax = plt.subplots(len(self.timeWindows), 2 ,sharex=True, figsize=(15,10),sharey=True)
        if len(self.timeWindows) == 1:
            ax[0].plot(self.resultsNN['time'][0],self.resultsNN['linTruePos'][0],c="black",alpha=0.3)
            ax[0].scatter(self.resultsNN['time'][0],self.resultsNN['linPred'][0],c=self.cm(12+0),alpha=0.9,label=(str(self.timeWindows[0])+ ' ms'),s=1)
            ax[0].set_title('Neural network decoder \n ' + str(self.timeWindows[0]) + ' window',fontsize="xx-large")
            ax[0].set_ylabel('linear position',fontsize="xx-large")
            ax[0].set_yticks([0,0.4,0.8])
        else:
            [a.plot(self.resultsNN['time'][i],self.resultsNN['linTruePos'][i],c="black",alpha=0.3) for i,a in enumerate(ax[:,0])]
            for i in range(len(self.timeWindows)):
                ax[i,0].scatter(self.resultsNN['time'][i],self.resultsNN['linPred'][i],c=self.cm(12+i),alpha=0.9,label=(str(self.timeWindows[i])+ ' ms'),s=1)
            if i == 0:
                ax[i,0].set_title('Neural network decoder \n ' + str(self.timeWindows[i]) + ' window',fontsize="xx-large")
            else:
                ax[i,0].set_title(str(self.timeWindows[i]) + ' window',fontsize="xx-large")

        # Bayes
        if len(self.timeWindows) == 1:
            ax[1].plot(self.resultsNN['time'][0], self.resultsNN['linTruePos'][0], c="black", alpha=0.3)
            ax[1].scatter(self.resultsNN['time'][0], self.resultsBayes['linPred'][0], c=self.cm(0), alpha=0.9, label=(str(self.timeWindows[0])+ ' ms'), s=1)
            ax[1].set_title('Bayesian decoder \n' + str(self.timeWindows[0])  + ' window',fontsize="xx-large")
            ax[1].set_xlabel("time (s)",fontsize="xx-large")
        else:
            [a.plot(self.resultsNN['time'][i], self.resultsNN['linTruePos'][i], c="black", alpha=0.3) for i, a in enumerate(ax[:, 1])]
            for i in range(len(self.timeWindows)):
                ax[i,1].scatter(self.resultsNN['time'][i], self.resultsBayes['linPred'][i], c=self.cm(i), alpha=0.9, label=(str(self.timeWindows[i])+ ' ms'), s=1)
                if i == 0:
                    ax[i,1].set_title('Bayesian decoder \n' + str(self.timeWindows[i])  + ' window',fontsize="xx-large")
                else:
                    ax[i,1].set_title(str(self.timeWindows[i]) + ' window',fontsize="xx-large")
            ax[len(self.timeWindows)-1,0].set_xlabel("time (s)",fontsize="xx-large")
            ax[len(self.timeWindows)-1,1].set_xlabel("time (s)",fontsize="xx-large")
            [a.set_ylabel("linear position",fontsize="xx-large") for a in ax[:,0]]
            [ax[i,0].set_yticks([0,0.4,0.8]) for i in range(len(self.timeWindows))]
        # Save figure
        fig.tight_layout()
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, 'example_nn_bayes.png'))
        plt.savefig(os.path.join(self.folderFigures, 'example_nn_bayes.svg'))

    def hist_linerrors(self, speed='all'):
        ### Prepare the data
        # Masks
        habMask = [inEpochsMask(self.resultsNN['time'][i], self.behaviorData["Times"]["testEpochs"])
                         for i in range(len(self.timeWindows))]
        habMaskFast = [(habMask[i]) * (self.resultsNN['speedMask'][i]) for i in range(len(self.timeWindows))]
        habMaskSlow = [(habMask[i]) * np.logical_not(self.resultsNN['speedMask'][i][i]) for i in range(len(self.timeWindows))]
        # Data
        lErrorNN = [np.abs(self.resultsNN['linTruePos'][i]-self.resultsNN['linPred'][i]) for i in range(len(self.timeWindows))]
        lErrorBayes = [np.abs(self.resultsNN['linTruePos'][i]-self.resultsBayes['linPred'][i]) for i in range(len(self.timeWindows))]
        if speed == 'all':
            lErrorNN = [lErrorNN[i][habMask[i]] for i in range(len(self.timeWindows))]
            lErrorBayes = [lErrorBayes[i][habMask[i]] for i in range(len(self.timeWindows))]
        elif speed == 'fast':
            lErrorNN = [lErrorNN[i][habMaskFast[i]] for i in range(len(self.timeWindows))]
            lErrorBayes = [lErrorBayes[i][habMaskFast[i]] for i in range(len(self.timeWindows))]
        elif speed == 'slow':
            lErrorNN = [lErrorNN[i][habMaskSlow[i]] for i in range(len(self.timeWindows))]
            lErrorBayes = [lErrorBayes[i][habMaskSlow[i]] for i in range(len(self.timeWindows))]
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')
        
        ## Figure 2: we plot the histograms of errors
        fig,axes = plt.subplots(2, 2, sharex=True, figsize=(8,6.5), constrained_layout = True)
        ax = axes.flatten()
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0.025, hspace=0.0001)
        for iw in range(len(self.timeWindows)):
            if iw==0:
                ax[iw].hist(lErrorNN[iw], color=self.cm(iw+12), bins=self.binsLinearPosHist, histtype="step",density=True,cumulative=True) # NN hist
                ax[iw].vlines(np.mean(lErrorNN[iw]), 0, 1, color=self.cm(iw + 12), label="NN") # NN mean
                ax[iw].hist(lErrorBayes[iw], color=self.cm(iw), bins=self.binsLinearPosHist,histtype="step",density=True,cumulative=True) # Bayes hist
                ax[iw].vlines(np.mean(lErrorBayes[iw]), 0, 1, color=self.cm(iw), label="Bayesian") # Bayes mean
            else:
                ax[iw].hist(lErrorNN[iw], color=self.cm(iw + 12), bins=self.binsLinearPosHist, histtype="step", density=True, cumulative=True) # NN hist
                ax[iw].vlines(np.mean(lErrorNN[iw]), 0, 1 , color=self.cm(iw+12)) # NN mean
                ax[iw].hist(lErrorBayes[iw], color=self.cm(iw), bins=self.binsLinearPosHist, histtype="step", density=True, cumulative=True) # Bayes hist
                ax[iw].vlines(np.mean(lErrorBayes[iw]),0,1, color=self.cm(iw)) # Bayes mean
            ax[iw].set_ylim(0,1)
            ax[iw].set_title(str(self.timeWindows[iw]) + ' window',fontsize="x-large")
        # Tune graph
        [a.set_aspect('auto') for a in ax]
        [a.set_xticks([0,0.4,0.8]) for a in ax]
        [a.set_xlim(0,0.99) for a in ax]
        [a.set_yticks([0.25,0.5,0.75,1]) for a in ax]
        fig.legend(loc=(0.85,0.57))
        ax[0].set_ylabel("cumulative \n histogram",fontsize="x-large")
        ax[2].set_ylabel("cumulative \n histogram", fontsize="x-large")
        ax[2].set_xlabel("absolute linear error", fontsize="x-large")
        ax[3].set_xlabel("absolute linear error", fontsize="x-large")
        fig.tight_layout()
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, ('cumulativeHist_' + str(speed) + '.png')))
        plt.savefig(os.path.join(self.folderFigures, ('cumulativeHist_' + str(speed) + '.svg')))

    def mean_linerrors(self, speed='all'):
        ### Prepare the data
        # Masks
        habMask = [inEpochsMask(self.resultsNN['time'][i], self.behaviorData["Times"]["testEpochs"])
                         for i in range(len(self.timeWindows))]
        habMaskFast = [(habMask[i]) * (self.resultsNN['speedMask'][i]) for i in range(len(self.timeWindows))]
        habMaskSlow = [(habMask[i]) * np.logical_not(self.resultsNN['speedMask'][i][i]) for i in range(len(self.timeWindows))]
        # Data
        lErrorNN = [np.abs(self.resultsNN['linTruePos'][i]-self.resultsNN['linPred'][i]) for i in range(len(self.timeWindows))]
        lErrorBayes = [np.abs(self.resultsNN['linTruePos'][i]-self.resultsBayes['linPred'][i]) for i in range(len(self.timeWindows))]
        if speed == 'all':
            lErrorNN_mean = np.array([np.mean(lErrorNN[i][habMask[i]]) for i in range(len(self.timeWindows))])
            lErrorNN_std = np.array([np.std(lErrorNN[i][habMask[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_mean = np.array([np.mean(lErrorBayes[i][habMask[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_std = np.array([np.std(lErrorBayes[i][habMask[i]]) for i in range(len(self.timeWindows))])
        elif speed == 'fast':
            lErrorNN_mean = np.array([np.mean(lErrorNN[i][habMaskFast[i]]) for i in range(len(self.timeWindows))])
            lErrorNN_std = np.array([np.std(lErrorNN[i][habMaskFast[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_mean = np.array([np.mean(lErrorBayes[i][habMaskFast[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_std = np.array([np.std(lErrorBayes[i][habMaskFast[i]]) for i in range(len(self.timeWindows))])
        elif speed == 'slow':
            lErrorNN_mean = np.array([np.mean(lErrorNN[i][habMaskSlow[i]]) for i in range(len(self.timeWindows))])
            lErrorNN_std = np.array([np.std(lErrorNN[i][habMaskFast[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_mean = np.array([np.mean(lErrorBayes[i][habMaskSlow[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_std = np.array([np.std(lErrorBayes[i][habMaskFast[i]]) for i in range(len(self.timeWindows))])
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')        
        
        # Fig mean error from window size - total
        fig,ax = plt.subplots(figsize=(10,10))
        ax.plot(self.timeWindows, lErrorNN_mean, c="red", label="neural network")
        ax.fill_between(self.timeWindows, lErrorNN_mean-lErrorNN_std, lErrorNN_mean+lErrorNN_std,color="red", alpha=0.5)
        ax.plot(self.timeWindows, lErrorBayes_mean, c="blue", label="bayesian")
        ax.fill_between(self.timeWindows, lErrorBayes_mean-lErrorBayes_std, lErrorBayes_mean+lErrorBayes_std,color="blue", alpha=0.5)
        ax.set_xlabel("window size (ms)",fontsize="xx-large")
        ax.set_xticks(self.timeWindows)
        ax.set_xticklabels(self.timeWindows,fontsize="xx-large")
        ax.set_yticks(np.unique(np.concatenate([np.round(lErrorNN_mean,2),np.round(lErrorBayes_mean,2)])))
        ax.set_yticklabels(np.unique(np.concatenate([np.round(lErrorNN_mean,2),np.round(lErrorBayes_mean,2)])),fontsize="xx-large")
        ax.set_ylabel("mean linear error",fontsize="xx-large")
        fig.legend(loc=(0.6,0.7),fontsize="xx-large")
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, ('meanError_' + str(speed) + '.png')))
        plt.savefig(os.path.join(self.folderFigures, ('meanError_' + str(speed) + '.svg')))
        
    def nnVSbayes(self, speed='all'):
        # Masks
        habMask = [inEpochsMask(self.resultsNN['time'][i], self.behaviorData["Times"]["testEpochs"])
                         for i in range(len(self.timeWindows))]
        habMaskFast = [(habMask[i]) * (self.resultsNN['speedMask'][i]) for i in range(len(self.timeWindows))]
        habMaskSlow = [(habMask[i]) * np.logical_not(self.resultsNN['speedMask'][i][i]) for i in range(len(self.timeWindows))]
        if speed == 'all':
            masks = habMask
        elif speed == 'fast':
            masks = habMaskFast
        elif speed == 'slow':
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')
        
        #Figure 4:
        cols = plt.get_cmap("terrain")
        fig,ax = plt.subplots(1, len(self.timeWindows), figsize=(10,6))
        if len(self.timeWindows) == 1:
            ax = [ax] # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].scatter(self.resultsBayes['linPred'][iw][masks[iw]],
                        self.resultsNN['linPred'][iw][masks[iw]],s=1,c="grey")
            ax[iw].hist2d(self.resultsBayes['linPred'][iw][masks[iw]],
                        self.resultsNN['linPred'][iw][masks[iw]],(45,45),cmap=white_viridis,
                        alpha=0.8,density=True)
            ax[iw].set_yticks([])
            if iw < len(self.timeWindows):
                ax[iw].set_xticks([])
        # Tune ticks 
        [a.set_xlabel(((str(self.timeWindows[iw]) + ' ms')),fontsize="x-large") for iw, a in enumerate(ax)]
        ax[len(self.timeWindows)-1].set_xlabel((str(self.timeWindows[len(self.timeWindows)-1]) + ' ms \n Bayesian decoding'),fontsize="x-large")
        [a.set_ylabel("NN decoding" ,fontsize="x-large") for a in ax]
        [a.set_aspect("auto") for a in ax]
        [plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,1),cmap=white_viridis), ax=a,label="density") for a in ax]
        plt.suptitle(('Position decoded during \n' +str(speed) + ' speed periods'),fontsize="xx-large")
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, ('NNvsBayesian_' + str(speed) + '.png')))
        plt.savefig(os.path.join(self.folderFigures, ('NNvsBayesian_' + str(speed) + '.svg')))

    def predLoss_vs_trueLoss(self, speed='all', mode='2d'):
        # Calculate error
        if mode == '2d':
            errors = [np.log(np.sqrt(np.sum(np.square(self.resultsNN['truePos'][iw]-self.resultsNN['fullPred'][iw]),axis=1)))
                    for iw in range(len(self.timeWindows))]
        elif mode == '1d':
            errors = [np.abs(self.resultsNN['linTruePos'][iw]-self.resultsNN['linPred'][iw]) for iw in range(len(self.timeWindows))]
        else:
            raise ValueError('mode argument could be only "2d" or "1d"')
        
        # Masks
        habMask = [inEpochsMask(self.resultsNN['time'][i], self.behaviorData["Times"]["testEpochs"])
                         for i in range(len(self.timeWindows))]
        habMaskFast = [(habMask[i]) * (self.resultsNN['speedMask'][i]) for i in range(len(self.timeWindows))]
        habMaskSlow = [(habMask[i]) * np.logical_not(self.resultsNN['speedMask'][i][i]) for i in range(len(self.timeWindows))]
        if speed == 'all':
            masks = habMask
        elif speed == 'fast':
            masks = habMaskFast
        elif speed == 'slow':
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')
        
         # Figure
        fig,ax = plt.subplots(1 ,len(self.timeWindows), figsize=(10,4))
        if len(self.timeWindows) == 1:
            ax = [ax] # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].scatter(self.resultsNN['predLoss'][iw][masks[iw]], errors[iw][masks[iw]],c="grey",s=1)
            ax[iw].hist2d(self.resultsNN['predLoss'][iw][masks[iw]], errors[iw][masks[iw]],(30,30),
                          cmap=white_viridis, alpha=0.4, density=True) #,c="red",alpha=0.4
            ax[iw].set_xlabel("Predicted loss (log)")
            if mode == '2d':
                ax[iw].set_ylabel("True error (log)")
            elif mode == '1d':
                ax[iw].set_ylabel("Linear error")
            ax[iw].set_title(((str(self.timeWindows[iw]) + ' ms')),fontsize="x-large")
            ax[iw].set_ylim(-4, 0)

        fig.tight_layout()
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, ('predLoss_vs_trueLoss' + str(speed) + '.png')))
        plt.savefig(os.path.join(self.folderFigures, ('predLoss_vs_trueLoss' + str(speed) + '.svg')))

    def fig_example_2d(self, speed='all'):
        
        mazeBorder = np.array([[0,0,1,1,0.63,0.63,0.35,0.35,0],[0,1,1,0,0,0.75,0.75,0,0]])
        ts = [self.resultsNN['time'][iw] for iw in range(len(self.timeWindows))]
        # Trajectory figure
        cm = plt.get_cmap("turbo")
        fig,ax = plt.subplots(1 ,len(self.timeWindows), figsize=(10,4))
        if len(self.timeWindows) == 1:
            ax = [ax] # compatibility move
        for iw in range(len(self.timeWindows)):
            ax[iw].plot(self.resultsNN['truePos'][iw][:,0], self.resultsNN['truePos'][iw][:,1], color='black', label="true traj")
            ax[iw].scatter(self.resultsNN['fullPred'][iw][:,0], self.resultsNN['truePos'][iw][:,1],
                       c='red', s=3, label="predicted traj")
            # plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=np.min(ts),vmax=np.max(ts)),cmap=cm),label="prediction time (s)")
            ax[iw].set_xlabel("X")
            ax[iw].set_ylabel("Y")
            ax[iw].plot(mazeBorder.transpose()[:,0],mazeBorder.transpose()[:,1],c="black")
        fig.legend()
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, ('decoded_trajectories_' + str(speed) + '.png')))
        plt.savefig(os.path.join(self.folderFigures, ('decoded_trajectories_' + str(speed) + '.svg')))

    def predLoss_linError(self, speed='all', step=0.1):
        # Masks
        habMask = [inEpochsMask(self.resultsNN['time'][i], self.behaviorData["Times"]["testEpochs"])
                         for i in range(len(self.timeWindows))]
        habMaskFast = [(habMask[i]) * (self.resultsNN['speedMask'][i]) for i in range(len(self.timeWindows))]
        habMaskSlow = [(habMask[i]) * np.logical_not(self.resultsNN['speedMask'][i][i]) for i in range(len(self.timeWindows))]
        if speed == 'all':
            masks = habMask
        elif speed == 'fast':
            masks = habMaskFast
        elif speed == 'slow':
            masks = habMaskSlow
        else:
            raise ValueError('speed argument could be only "full", "fast" or "slow"')
        
        ## Calculate errors at each level of predLoss
        errors = [np.abs(self.resultsNN['linTruePos'][iw]-self.resultsNN['linPred'][iw]) for iw in range(len(self.timeWindows))]
        predLoss_ticks = [np.arange(np.min(self.resultsNN['predLoss'][iw]),np.max(self.resultsNN['predLoss'][iw]), step) for iw in range(len(self.timeWindows))]
        errors_filtered = []
        for iw in range(len(self.timeWindows)):
            errors_filtered.append([np.mean(errors[iw][np.less_equal(self.resultsNN['predLoss'][iw],pfilt)])
                            for pfilt in predLoss_ticks[iw]])

        ## Figure 6: decrease of the mean absolute linear error as a function of the filtering value
        labelNames = [(str(self.timeWindows[iw]) + ' ms') for iw in range(len(self.timeWindows))]
        fig,ax = plt.subplots(figsize=(10,5.3), constrained_layout = True)
        [ax.plot(predLoss_ticks[iw],errors_filtered[iw],c=self.cm(12+iw),label=labelNames[iw]) for iw in range(len(self.timeWindows))]
        ax.set_xlabel("Neural network \n prediction filtering value", fontsize="x-large")
        ax.set_ylabel("mean absolute linear err4or", fontsize="x-large")
        ax.set_title((speed + ' speed'), fontsize="x-large")
        fig.legend(loc=(0.87,0.17), fontsize=12)
        fig.show()
        
        plt.savefig(os.path.join(self.folderFigures, 'predLoss_vs_error.png'))
        plt.savefig(os.path.join(self.folderFigures, 'predLoss_vs_error.svg'))

    def fig_example_linear_filtered(self, fprop=0.25):
        # Calculate filtering values
        fvalues = [(np.min(self.resultsNN['predLoss'][iw]) + np.ptp(self.resultsNN['predLoss'][iw])*fprop) for iw in range(len(self.timeWindows))]
        filters_lpred = [np.ones(self.resultsNN['time'][iw].shape).astype(np.bool)*
                                    np.less_equal(self.resultsNN['predLoss'][iw],fvalues[iw]) for iw in range(len(self.timeWindows))]
        
        
        fig,ax = plt.subplots(len(self.timeWindows), 2 ,sharex=True,figsize=(15,10),sharey=True)
        # All points
        if len(self.timeWindows) == 1:
            ax[0].plot(self.resultsNN['time'][0],self.resultsNN['linTruePos'][0],c="black",alpha=0.3)
            ax[0].scatter(self.resultsNN['time'][0],self.resultsNN['linPred'][0],c=self.cm(12+0),alpha=0.9,label=(str(self.timeWindows[0])+ ' ms'),s=1)
            ax[0].set_title('Neural network decoder \n ' + str(self.timeWindows[0]) + ' window',fontsize="xx-large")
            ax[0].set_ylabel('linear position',fontsize="xx-large")
            ax[0].set_yticks([0,0.4,0.8])
        else:
            [a.plot(self.resultsNN['time'][i],self.resultsNN['linTruePos'][i],c="black",alpha=0.3) for i,a in enumerate(ax[:,0])]
            for i in range(len(self.timeWindows)):
                ax[i,0].scatter(self.resultsNN['time'][i],self.resultsNN['linPred'][i],c=self.cm(12+i),alpha=0.9,label=(str(self.timeWindows[i])+ ' ms'),s=1)
            if i == 0:
                ax[i,0].set_title('Neural network decoder \n ' + str(self.timeWindows[i]) + ' window',fontsize="xx-large")
            else:
                ax[i,0].set_title(str(self.timeWindows[i]) + ' window',fontsize="xx-large")

        # Filtered data
        if len(self.timeWindows) == 1:
            ax[1].plot(self.resultsNN['time'][0], self.resultsNN['linTruePos'][0], c="black", alpha=0.3)
            ax[1].scatter(self.resultsNN['time'][0][filters_lpred[0]], self.resultsBayes['linPred'][0][filters_lpred[0]],
                          c=self.cm(12+0), alpha=0.9, label=(str(self.timeWindows[0])+ ' ms'), s=1)
            ax[1].set_title('Best ' + str(fprop*100) + '% of predicitons \n' + str(self.timeWindows[0])  + ' ms window',fontsize="xx-large")
            ax[1].set_xlabel("time (s)",fontsize="xx-large")
        else:
            [a.plot(self.resultsNN['time'][i], self.resultsNN['linTruePos'][i], c="black", alpha=0.3) for i, a in enumerate(ax[:, 1])]
            for i in range(len(self.timeWindows)):
                ax[i,1].scatter(self.resultsNN['time'][i][filters_lpred[0]], self.resultsBayes['linPred'][i][filters_lpred[0]],
                                c=self.cm(12+i), alpha=0.9, label=(str(self.timeWindows[i])+ ' ms'), s=1)
                if i == 0:
                    ax[i, 1].set_title('Best ' + str(fprop*100) + '% of predicitons \n' + str(self.timeWindows[0])  + ' ms window',fontsize="xx-large")
                else:
                    ax[i,1].set_title(str(self.timeWindows[i]) + ' window',fontsize="xx-large")
            ax[len(self.timeWindows)-1,0].set_xlabel("time (s)",fontsize="xx-large")
            ax[len(self.timeWindows)-1,1].set_xlabel("time (s)",fontsize="xx-large")
            [a.set_ylabel("linear position",fontsize="xx-large") for a in ax[:,0]]
            [ax[i,0].set_yticks([0,0.4,0.8]) for i in range(len(self.timeWindows))]
        # Save figure
        fig.tight_layout()
        fig.show()
        plt.savefig(os.path.join(self.folderFigures, ('example_nn_bayes_filtered_' + str(fprop*100) + '%.png')))
        plt.savefig(os.path.join(self.folderFigures, ('example_nn_bayes_filtered_' + str(fprop*100) + '%.svg')))



# ------------------------------------------------------------------------------------------------------------------------------

    ### Figure 4: we take an example place cell,
    # and we scatter plot a link between its firing rate and the decoding.

   # def calc_tuning_curve(self, spikes_aligned, resultsNN):

    #     # Prepare arrays for tuning curve calculation
    #     spikes_train = spikes_aligned['spikes_train'][0]
    #     spikes_test = spikes_aligned['spikes_test'][0]
        
    #     spikeMat_popVector_hab = np.zeros([timePreds_train.shape[0]+timePreds.shape[0],
    #                                     spikeMat_window_popVector.shape[1]])
    #     spikeMat_popVector_hab[:timePreds_train.shape[0],:] = spikeMat_window_popVector_train[:timePreds_train.shape[0]]
    #     spikeMat_popVector_hab[timePreds_train.shape[0]:, :] = spikeMat_window_popVector[:timePreds.shape[0]]
    #     timePreds_hab = np.concatenate([timePreds_train,timePreds])
    #     trueLinearPos_hab = np.concatenate([trueLinearPos_train,trueLinearPos])
    #     #
    #     #
    # #
    #     placeFieldSort = trainerBayes.linearPosArgSort
    #     prefPos = trainerBayes.linearPreferredPos
        
    #     for target in [5,16,49,46,68,71]:
    #         pcId = np.where(np.equal(placeFieldSort,target))[0][0] # newID_PlaceToStudy[0]
    #         prefPosPC = prefPos[pcId]
    #         #let us compute the tuning curve of the neuron for the linear variable:
    #         binsLinearPos = np.arange(0,1,step=0.01)
    #         pcFiring = spikeMat_popVector_hab[:, pcId+1]
    #         # firing = np.array([np.sum(pcFiring[np.greater_equal(trueLinearPos_hab,binsLinearPos[id])*
    #         #                                    np.less(trueLinearPos_hab,binsLinearPos[id+1])])/np.sum(np.greater_equal(trueLinearPos_hab,binsLinearPos[id])*
    #         #                                    np.less(trueLinearPos_hab,binsLinearPos[id+1]))
    #         #     for id in range(len(binsLinearPos)-1)])
    #         firing = np.array([np.sum(pcFiring[np.greater_equal(trueLinearPos_hab,binsLinearPos[id])*
    #                                         np.less(trueLinearPos_hab,binsLinearPos[id+1])])
    #             for id in range(len(binsLinearPos)-1)])
    #         tc_pc = firing/(0.032)

    #         pcFiring_test = spikeMat_window_popVector[:timePreds.shape[0], pcId + 1]
    #         #When making a prediction around the pcFiring, the proba should be reflecting the firing rate of the cell...
    #         predAroundPrefPos =np.greater(pcFiring_test,0)

    #         error = np.abs(linearNoNoisePos_varyingWind[0]-trueLinearPos)
    #         #using predicted Error:
    #         normalize01 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    #         cm = plt.get_cmap("gray")
    #         fig,ax = plt.subplots()
    #         ax.scatter(linearNoNoisePos_varyingWind[0][predAroundPrefPos],
    #                 (pcFiring_test/np.sum(spikeMat_window_popVector[:timePreds.shape[0],:],axis=1))[predAroundPrefPos],s=12,
    #                 c=cm(normalize01(lossPredNoNoise_varyingWind[0][predAroundPrefPos])),edgecolors="black",linewidths=0.2)
    #         ax.scatter(linearNoNoisePos_varyingWind[0][predAroundPrefPos*np.greater(error,0.5)],
    #                 (pcFiring_test/np.sum(spikeMat_window_popVector[:timePreds.shape[0],:],axis=1))[predAroundPrefPos*np.greater(error,0.5)],s=24,
    #                 c="white",alpha=0.1,edgecolors="red",linewidths=0.4)

    #         plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(lossPredNoNoise_varyingWind[0][predAroundPrefPos]),np.max(lossPredNoNoise_varyingWind[0][predAroundPrefPos])),cmap=cm)
    #                     ,label="NN log error \n estimate")
    #         ax.set_xlabel("predicted position")
    #         ax.set_ylabel("Number of spike \n relative to total number of spike \n in 36ms window")
    #         at = ax.twinx()
    #         at.plot(binsLinearPos[:-1],tc_pc,c="navy",alpha=0.5)
    #         # ax[0].set_xlabel("linear position")
    #         at.set_ylabel("firing rate",color="navy")
    #         fig.tight_layout()
    #         fig.show()
    
    # def fft_pc():
    #     #Compute Fourier transform of predicted positions:
    #     from scipy.fft import fft, fftfreq
    #     from scipy.interpolate import  interp1d
    #     # First interpolate in time the signal so that we sample them well:
    #     filters = [np.less(timePredsNoNoise_varyingWind[i][:,0],1645)*np.greater(timePredsNoNoise_varyingWind[i][:,0],1627) for i in range(4)]
    #     itps = [interp1d(timePredsNoNoise_varyingWind[i][filters[i],0],linearNoNoisePos_varyingWind[i][filters[i],0]) for i in range(4)]
    #     itpLast = np.min([np.max(timePredsNoNoise_varyingWind[i][filters[i],0]) for i in range(4)])
    #     itpFirst = np.max([np.min(timePredsNoNoise_varyingWind[i][filters[i],0]) for i in range(4)])
    #     x = np.arange(itpFirst,itpLast,0.003)
    #     discrete_linearPos = [itp(x) for itp in itps]
    #     fig,ax = plt.subplots()
    #     ax.scatter(x,discrete_linearPos[3])
    #     fig.show()
    #     spectrums = [fft(dlp) for dlp in discrete_linearPos]
    #     xf = fftfreq(x.shape[0], 0.003)
    #     fig,ax =plt.subplots()
    #     [ax.plot(xf[:1000],2.0/(x.shape[0]) * np.abs(spectrums[i][0:1000]),c=cm(i+4)) for i in [3]]
    #     ax.set_xlabel("frequency, Hz")
    #     ax.set_ylabel("Fourrier Power")
    #     fig.show()

    # ### Let us pursue on comparing NN and Bayesian:

    # import tqdm
    # ## We will compare the NN with bayesian, random and shuffled bayesian
    # errors = []
    # errorsRandomMean = []
    # errorsRandomStd = []
    # errorsShuffleMean = []
    # errorsShuffleStd = []
    # for lossVal in tqdm.tqdm(np.arange(np.min(lossPredNoNoise_varyingWind[0][:,0]), np.max(lossPredNoNoise_varyingWind[0][:,0]), step=0.1)):
    #     bayesPred = linearpos_bayes_varying_window[0][
    #         np.less_equal(lossPredNoNoise_varyingWind[0][:,0], lossVal)]
    #     NNpred = linearNoNoisePos_varyingWind[0][:,0][np.less_equal(lossPredNoNoise_varyingWind[0][:,0], lossVal)]
    #     if (NNpred.shape[0] > 0):
    #         randomPred = np.random.uniform(0, 1, [NNpred.shape[0], 100])
    #         errors += [np.mean(np.abs(bayesPred - NNpred))]
    #         errRand = np.mean(np.abs(NNpred[:, None] - randomPred), axis=0)
    #         errorsRandomMean += [np.mean(errRand)]
    #         errorsRandomStd += [np.std(errRand)]
    #     shuffles = []
    #     for id in range(100):
    #         b = np.copy(bayesPred)
    #         np.random.shuffle(b)
    #         shuffles += [np.mean(np.abs(NNpred - b))]
    #     errorsShuffleMean += [np.mean(shuffles)]
    #     errorsShuffleStd += [np.std(shuffles)]
    # errorsRandomMean = np.array(errorsRandomMean)
    # errorsRandomStd = np.array(errorsRandomStd)
    # errorsShuffleMean = np.array(errorsShuffleMean)
    # errorsShuffleStd = np.array(errorsShuffleStd)
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(np.min(lossPredNoNoise_varyingWind[0][:,0]), np.max(lossPredNoNoise_varyingWind[0][:,0]), step=0.1), errors, label="bayesian")
    # ax.plot(np.arange(np.min(lossPredNoNoise_varyingWind[0][:,0]), np.max(lossPredNoNoise_varyingWind[0][:,0]), step=0.1), errorsRandomMean, color="red",
    #         label="random Prediction")
    # ax.fill_between(np.arange(np.min(lossPredNoNoise_varyingWind[0][:,0]), np.max(lossPredNoNoise_varyingWind[0][:,0]), step=0.1),
    #                 errorsRandomMean + errorsRandomStd, errorsRandomMean - errorsRandomStd, color="orange")
    # ax.plot(np.arange(np.min(lossPredNoNoise_varyingWind[0][:,0]), np.max(lossPredNoNoise_varyingWind[0][:,0]), step=0.1), errorsShuffleMean, color="purple",
    #         label="shuffle bayesian")
    # ax.fill_between(np.arange(np.min(lossPredNoNoise_varyingWind[0][:,0]), np.max(lossPredNoNoise_varyingWind[0][:,0]), step=0.1),
    #                 errorsShuffleMean + errorsShuffleStd, errorsShuffleMean - errorsShuffleStd, color="violet")
    # ax.set_ylabel("linear distance from NN predictions to Bayesian \n or random predictions")
    # ax.set_xlabel("probability filtering value")
    # ax.set_title("Wake")
    # fig.legend(loc=[0.2, 0.2])
    # fig.show()
    # plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure_lossPred", "fig_lineardiffBayesNN_wake.png"))
    # plt.savefig(os.path.join(projectPath.resultsPath, "paperFigure_lossPred", "fig_lineardiffBayesNN_wake.svg"))

    # fig,ax = plt.subplots()
    # ax.scatter(linearNoNoisePos_varyingWind[0][habEpochMaskandSpeeds[0]],lossPredNoNoise_varyingWind[0][habEpochMaskandSpeeds[0]],s=1,c="grey")
    # ax.hist2d(linearNoNoisePos_varyingWind[0][habEpochMaskandSpeeds[0],0],lossPredNoNoise_varyingWind[0][habEpochMaskandSpeeds[0],0],(30,30),cmap=white_viridis,alpha=0.4)
    # ax.set_xlabel("predicted linear position")
    # ax.set_ylabel("NN predicted loss")
    # fig.show()
    # # fig,ax = plt.subplots()
    # # ax.scatter(linearNoNoisePos_varyingWind[0][habEpochMask*np.logical_not(windowmask_speed)],lossPredNoNoise_varyingWind[0][habEpochMask*np.logical_not(windowmask_speed)],s=1,c="grey")
    # # ax.hist2d(linearNoNoisePos_varyingWind[0][habEpochMask*np.logical_not(windowmask_speed),0],lossPredNoNoise_varyingWind[0][habEpochMask*np.logical_not(windowmask_speed),0],(30,30),cmap=white_viridis,alpha=0.4)
    # # ax.set_xlabel("predicted linear position")
    # # ax.set_ylabel("NN predicted loss")
    # # fig.show()

    # #
    # # mask_right_arm_pred_argmax = np.greater(linearNoNoisePos_varyingWind[0],0.7)
    # #
    # # error_rightarm = np.abs(linearNoNoisePos_varyingWind[0]-trueLinearPos)[mask_right_arm_pred_argmax*habEpochMaskandSpeed]
    # # error_OtherArm = np.abs(linearNoNoisePos_varyingWind[0] - trueLinearPos)[
    # #     np.logical_not(mask_right_arm_pred_argmax) * habEpochMaskandSpeed]
    # #
    # # fig,ax = plt.subplots()
    # # ax.scatter(linearNoNoisePos_varyingWind[0][mask_right_arm_pred_argmax*habEpochMaskandSpeed],error_rightarm,s=1)
    # # fig.show()
    # #
    # # mask_middle_arm_pred_argmax = np.greater(linearNoNoisePos_varyingWind[0], 0.3)*np.less(linearNoNoisePos_varyingWind[0], 0.7)
    # # error_MiddleArm = np.abs(linearNoNoisePos_varyingWind[0] - trueLinearPos)[
    # #     mask_middle_arm_pred_argmax * habEpochMaskandSpeed]
    # # fig,ax = plt.subplots()
    # # ax.hist(error_rightarm,color="blue",histtype="step",density=True,bins=50)
    # # # ax.vlines(np.mean(error_rightarm),ymin=0,ymax=16,color="blue")
    # # ax.vlines(np.median(error_rightarm), ymin=0, ymax=16, color="blue")
    # # ax.hist(error_MiddleArm,color="red",histtype="step",density=True,bins=50)
    # # # ax.vlines(np.mean(error_MiddleArm),ymin=0,ymax=16,color="red")
    # # ax.vlines(np.median(error_MiddleArm), ymin=0, ymax=16, color="red")
    # # fig.show()