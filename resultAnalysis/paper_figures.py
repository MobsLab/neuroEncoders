
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
    def __init__(self, projectPath, behaviorData, trainerBayes, l_function,
                 bayesMatrices={}, timeWindows=[36]):
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
        self.resultsNN = {
            'time': time, 'speedMask': speedMask, 'linPred': lPredPos, 'fullPred': fPredPos, 'truePos': truePos, 'linTruePos': lTruePos, 'predLoss': lossPred
        }

    def test_bayes(self):

        if not hasattr(self.trainerBayes, 'linearPreferredPos') and not 'Occupation' in self.bayesMatrices.keys():
            self.bayesMatrices = self.trainerBayes.train_order_by_pos(self.behaviorData, self.l_function)

        #quickly obtain bayesian decoding:
        lPredPosBayes = []
        probaBayes = []
        fPredBayes = []
        for i,ws in enumerate(self.timeWindows):
            timesToPredict = self.resultsNN['time'][i][:, np.newaxis].astype(np.float64)
            outputsBayes = self.trainerBayes.test_as_NN(self.behaviorData, self.bayesMatrices,
                                                        timesToPredict, windowSizeMS=ws)
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
        # TODO: why is it speed filtered?
        fig,ax = plt.subplots(len(self.timeWindows), 2, sharex=True, figsize=(15,10),sharey=True)
        if len(self.timeWindows) == 1:
            ax[0].plot(self.resultsNN['time'][0],self.resultsNN['linTruePos'][0],c="black",alpha=0.3)
            ax[0].scatter(self.resultsNN['time'][0],self.resultsNN['linPred'][0],c=self.cm(12+0),alpha=0.9,label=(str(self.timeWindows[0])+ ' ms'),s=1)
            ax[0].set_title('Neural network decoder \n ' + str(self.timeWindows[0]) + ' window',fontsize="xx-large")
            ax[0].set_ylabel('linear position',fontsize="xx-large")
            ax[0].set_yticks([0, 0.4, 0.8])
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
        fig.savefig(os.path.join(self.folderFigures, 'example_nn_bayes.png'))
        fig.savefig(os.path.join(self.folderFigures, 'example_nn_bayes.svg'))

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
        fig.savefig(os.path.join(self.folderFigures, ('cumulativeHist_' + str(speed) + '.png')))
        fig.savefig(os.path.join(self.folderFigures, ('cumulativeHist_' + str(speed) + '.svg')))

    def mean_linerrors(self, speed='all', filtProp=None):
        ### Prepare the data
        # Masks
        habMask = [inEpochsMask(self.resultsNN['time'][i], self.behaviorData["Times"]["testEpochs"])
                         for i in range(len(self.timeWindows))]
        habMaskFast = [(habMask[i]) * (self.resultsNN['speedMask'][i]) for i in range(len(self.timeWindows))]
        habMaskSlow = [(habMask[i]) * np.logical_not(self.resultsNN['speedMask'][i][i]) for i in range(len(self.timeWindows))]
        if filtProp is not None:
            # Calculate filtering values
            sortedLPred = [np.argsort(self.resultsNN['predLoss'][iw]) for iw in range(len(self.timeWindows))]
            thresh = [np.squeeze(self.resultsNN['predLoss'][iw][sortedLPred[iw][int(len(sortedLPred[iw])*filtProp)]])
                      for iw in range(len(self.timeWindows))]
            filters_lpred = [np.ones(self.resultsNN['time'][iw].shape).astype(np.bool)*
                             np.less_equal(self.resultsNN['predLoss'][iw],thresh[iw]) for iw in range(len(self.timeWindows))]
        else:
            filters_lpred = [np.ones(habMask[i].shape).astype(np.bool) for i in range(len(self.timeWindows))]
        finalMasks = [habMask[i] * filters_lpred[i] for i in range(len(self.timeWindows))]
        finalMasksFast = [habMaskFast[i] * filters_lpred[i] for i in range(len(self.timeWindows))]
        finalMasksSlow = [habMaskSlow[i] * filters_lpred[i] for i in range(len(self.timeWindows))]

        # Data
        lErrorNN = [np.abs(self.resultsNN['linTruePos'][i]-self.resultsNN['linPred'][i]) for i in range(len(self.timeWindows))]
        lErrorBayes = [np.abs(self.resultsNN['linTruePos'][i]-self.resultsBayes['linPred'][i]) for i in range(len(self.timeWindows))]
        if speed == 'all':
            lErrorNN_mean = np.array([np.mean(lErrorNN[i][finalMasks[i]]) for i in range(len(self.timeWindows))])
            lErrorNN_std = np.array([np.std(lErrorNN[i][finalMasks[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_mean = np.array([np.mean(lErrorBayes[i][finalMasks[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_std = np.array([np.std(lErrorBayes[i][finalMasks[i]]) for i in range(len(self.timeWindows))])
        elif speed == 'fast':
            lErrorNN_mean = np.array([np.mean(lErrorNN[i][finalMasksFast[i]]) for i in range(len(self.timeWindows))])
            lErrorNN_std = np.array([np.std(lErrorNN[i][finalMasksFast[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_mean = np.array([np.mean(lErrorBayes[i][finalMasksFast[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_std = np.array([np.std(lErrorBayes[i][finalMasksFast[i]]) for i in range(len(self.timeWindows))])
        elif speed == 'slow':
            lErrorNN_mean = np.array([np.mean(lErrorNN[i][finalMasksSlow[i]]) for i in range(len(self.timeWindows))])
            lErrorNN_std = np.array([np.std(lErrorNN[i][finalMasksSlow[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_mean = np.array([np.mean(lErrorBayes[i][finalMasksSlow[i]]) for i in range(len(self.timeWindows))])
            lErrorBayes_std = np.array([np.std(lErrorBayes[i][finalMasksSlow[i]]) for i in range(len(self.timeWindows))])
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
        if filtProp is None:
            fig.savefig(os.path.join(self.folderFigures, ('meanError_' + str(speed) + '.png')))
            fig.savefig(os.path.join(self.folderFigures, ('meanError_' + str(speed) + '.svg')))
        else:
            fig.savefig(os.path.join(self.folderFigures, ('meanError_' + str(speed) + '_filt.png')))
            fig.savefig(os.path.join(self.folderFigures, ('meanError_' + str(speed) + '_filt.svg')))

        return lErrorNN_mean, lErrorNN_std, lErrorBayes_mean, lErrorBayes_std
        
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
        fig.savefig(os.path.join(self.folderFigures, ('NNvsBayesian_' + str(speed) + '.png')))
        fig.savefig(os.path.join(self.folderFigures, ('NNvsBayesian_' + str(speed) + '.svg')))

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
        fig.savefig(os.path.join(self.folderFigures, ('predLoss_vs_trueLoss' + str(speed) + '.png')))
        fig.savefig(os.path.join(self.folderFigures, ('predLoss_vs_trueLoss' + str(speed) + '.svg')))

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
        fig.savefig(os.path.join(self.folderFigures, ('decoded_trajectories_' + str(speed) + '.png')))
        fig.savefig(os.path.join(self.folderFigures, ('decoded_trajectories_' + str(speed) + '.svg')))

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
        
        fig.savefig(os.path.join(self.folderFigures, 'predLoss_vs_error.png'))
        fig.savefig(os.path.join(self.folderFigures, 'predLoss_vs_error.svg'))

    def fig_example_linear_filtered(self, fprop=0.3):
        # Calculate filtering values
        sortedLPred = [np.argsort(self.resultsNN['predLoss'][iw]) for iw in range(len(self.timeWindows))]
        thresh = [np.squeeze(self.resultsNN['predLoss'][iw][sortedLPred[iw][int(len(sortedLPred[iw])*fprop)]])
                    for iw in range(len(self.timeWindows))]
        filters_lpred = [np.ones(self.resultsNN['time'][iw].shape).astype(np.bool)*
                                    np.less_equal(self.resultsNN['predLoss'][iw],thresh[iw]) for iw in range(len(self.timeWindows))]
        
        
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
            ax[1].scatter(self.resultsNN['time'][0][filters_lpred[0]], self.resultsNN['linPred'][0][filters_lpred[0]],
                          c=self.cm(12+0), alpha=0.9, label=(str(self.timeWindows[0])+ ' ms'), s=1)
            ax[1].set_title('Best ' + str(fprop*100) + '% of predicitons \n' + str(self.timeWindows[0])  + ' ms window',fontsize="xx-large")
            ax[1].set_xlabel("time (s)",fontsize="xx-large")
        else:
            [a.plot(self.resultsNN['time'][i], self.resultsNN['linTruePos'][i], c="black", alpha=0.3) for i, a in enumerate(ax[:, 1])]
            for i in range(len(self.timeWindows)):
                ax[i,1].scatter(self.resultsNN['time'][i][filters_lpred[i]], self.resultsNN['linPred'][i][filters_lpred[i]],
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
        fig.savefig(os.path.join(self.folderFigures, ('example_nn_bayes_filtered_' + str(fprop*100) + '%.png')))
        fig.savefig(os.path.join(self.folderFigures, ('example_nn_bayes_filtered_' + str(fprop*100) + '%.svg')))

    # ------------------------------------------------------------------------------------------------------------------------------
    ## Figure 4: we take an example place cell,
    # and we scatter plot a link between its firing rate and the decoding.

    def plot_pc_tuning_curve_and_predictions(self, ws=36):
        dirSave = os.path.join(self.folderFigures, 'tuningCurves')
        if not os.path.isdir(dirSave):
            os.mkdir(dirSave)

        iwindow = self.timeWindows.index(ws)
        # Calculate the tuning curve of all place cells
        linearTuningCurves, binEdges = self.trainerBayes.calculate_linear_tuning_curve(self.l_function,
                                                                                       self.behaviorData)
        placeFieldSort = self.trainerBayes.linearPosArgSort
        loadName = os.path.join(self.projectPath.dataPath, 'aligned', str(ws),
                                'test', 'spikeMat_window_popVector.csv')
        spikePopAligned = np.array(pd.read_csv(loadName).values[:,1:], dtype=np.float32)
        spikePopAligned = spikePopAligned[:len(self.resultsNN['linTruePos'][iwindow]), :]
        predLoss = self.resultsNN['predLoss'][iwindow]
        normalize = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))

        for icell, tuningCurve in enumerate(linearTuningCurves):
            pcId = np.where(np.equal(placeFieldSort, icell))[0][0]
            spikeHist = spikePopAligned[:, pcId + 1][:len(self.resultsNN['linTruePos'][iwindow])]
            spikeMask =np.greater(spikeHist, 0)

            if spikeMask.any(): # some neurons do not spike here
                cm = plt.get_cmap("gray")
                fig,ax = plt.subplots(figsize=(14,9))
                ax.scatter(self.resultsNN['linTruePos'][iwindow][spikeMask],
                          (spikeHist/np.sum(spikePopAligned,axis=1))[spikeMask], s=12,
                          c=cm(normalize(predLoss[spikeMask])), edgecolors="black", linewidths=0.2)
                plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(np.min(predLoss[spikeMask]),
                                                                 np.max(predLoss[spikeMask])),
                                                   cmap=cm),
                             label="Predicted loss")
                ax.set_xlabel("predicted linear position")
                ax.set_ylabel(f"Number of spike \n relative to total number of spike \n in {ws}ms window")
                at = ax.twinx()
                at.plot(binEdges[1:], tuningCurve, c="navy",alpha=0.5)
                at.set_ylabel("firing rate",color="navy")
                fig.tight_layout()
                fig.show()

                fig.savefig(os.path.join(dirSave, (f'{ws}_tc_pred_cluster{pcId}.png')))
                # fig.savefig(os.path.join(dirSave, (f'{ws}_tc_pred_cluster{pcId}.svg')))
                plt.close()


    
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