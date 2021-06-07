# Figures to analyze the model performances
import numpy as np
import matplotlib.pyplot as plt
import os
from  SimpleBayes import decodebayes
from importData.rawDataParser import inEpochsMask

def linear_performance(outputs,foldersave,filter = 1,behave_data=None):
    # - Scatter plot of performances
    # - Confusion matrix obtained by projecting the predictions on a linear variable
    # - Error as function of predLoss
    # - Error as function of speed --> to add!
    if not os.path.isdir(foldersave):
        os.makedirs(foldersave)

    predLoss = outputs["predofLoss"]
    predPos = outputs["featurePred"][:, 0:2]
    truePos = outputs["featureTrue"][:,:]
    speeds =  behave_data["Speed"][np.ravel(outputs["pos_index"])]

    # linearPred = outputs["linearPred"]
    # linearTrue = outputs["linearTrue"]
    # fig,ax = plt.subplots()
    # ax.scatter(np.log(np.square(linearPred-linearTrue)+10**(-8)),predLoss[:,0])
    # fig.show()

    cm = plt.get_cmap("plasma")
    fig,ax = plt.subplots(5,1,figsize=(5,10))
    sc1 = ax[0].scatter(np.log(np.sum(np.square(predPos-truePos)+10**(-8),axis=-1)),predLoss,c=cm(speeds[:,0]/np.max(speeds)),s=1,alpha=0.2)
    # cbar = plt.colorbar(mappable=sc1,label="speed",cax=ax[0])
    ax[0].set_xlabel("decoding error (log scale)")
    ax[0].set_ylabel("predicted log-error")
    ax[1].scatter(np.log(np.sum(np.square(predPos-truePos)+10**(-8),axis=-1)),speeds,c=cm(speeds[:,0]/np.max(speeds)),s=1,alpha=0.2)
    ax[1].set_ylabel("speed")
    ax[1].set_xlabel("decoding error (log scale)")
    ax[2].scatter(np.sum(np.square(predPos - truePos), axis=-1), speeds, alpha=0.2,s=1)
    ax[2].set_ylabel("speed")
    ax[2].set_xlabel("decoding error")# fig.show()

    ax[3].scatter(predLoss, speeds, alpha=0.2,s=1)
    ax[3].set_ylabel("speed")
    ax[3].set_xlabel("predicted decoding error")# fig.show()
    ax[4].scatter(np.exp(predLoss), speeds, alpha=0.2,s=1)
    ax[4].set_ylabel("speed")
    ax[4].set_xlabel("predicted decoding error")# fig.show()
    fig.tight_layout()
    fig.savefig(os.path.join(foldersave,"errorsScatter.png"))

    #Let us scatter plot the predicted position with in color the loss prediction:
    fig, ax = plt.subplots(2,1)
    cm2 = plt.get_cmap("turbo")
    maxLossPred = np.max(np.abs(predLoss[:,0]))
    minLossPred = np.min(np.abs(predLoss[:,0]))
    ax[0].scatter(predPos[:,0], predPos[:,1], s=1,
               c=cm2((np.abs(predLoss[:,0]) - minLossPred) / (maxLossPred - minLossPred)))
    ax[1].scatter(truePos[:,0], truePos[:,1], s=1,
               c=cm2((np.abs(predLoss[:,0]) - minLossPred) / (maxLossPred - minLossPred)))
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[0].set_xlabel("true X")
    ax[0].set_ylabel("true Y")
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=minLossPred, vmax=maxLossPred), cmap=cm2),
                 label="Log Loss Pred (absolute value)", ax=ax)
    fig.show()
    fig.savefig(os.path.join(foldersave, "2dLossPred.png"))


    # from SimpleBayes import  butils
    # xEdges, yEdges, MRF = butils.kde2D(np.log(np.sum(np.square(predPos-truePos)+10**(-8),axis=-1)), speeds[:,0],xbins=15j,ybins=15j,bandwidth=1.0)
    #
    # fig,ax = plt.subplots()
    # ax.matshow(np.log(np.transpose(MRF)))
    # ax.set_xticks(range(15))
    # ax.set_yticks(range(15))
    # ax.set_xticklabels(np.round(xEdges[:,0],2))
    # ax.set_yticklabels(np.round(yEdges[0,:],2))
    # fig.show()

    # at filter = 1, no filtering is applied
    bestPred = (predLoss <= np.quantile(predLoss, filter))[:, 0]

    predPos = outputs["featurePred"][bestPred, 0:2]
    truePos = outputs["featureTrue"][bestPred,:]
    predLoss = predLoss[bestPred,:]
    projPred = outputs["projPred"][bestPred,:]
    projTruePos = outputs["projTruePos"][bestPred,:]
    linearPred = outputs["linearPred"][bestPred]
    linearTrue = outputs["linearTrue"][bestPred]
    timeStepsPred = outputs["times"][bestPred]


    fig,ax = plt.subplots(1+predPos.shape[1],1)
    ax[0].set_xlabel("true")
    ax[0].set_ylabel("prediction")
    for id in range(predPos.shape[1]):
        ax[0].scatter(truePos[:,id],predPos[:,id],alpha=0.3)
        ax[id+1].scatter(timeStepsPred,predPos[:,id],c="red",s=0.3)
        ax[id+1].scatter(timeStepsPred,truePos[:,id],c="black",s=0.3)
        ax[id+1].set_ylabel("feature"+str(id))
        ax[id + 1].set_xlabel("time")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(foldersave,"scatterPlots.png"))


    projBin = np.arange(0,stop=1.2,step=0.2)
    binIndex = [ np.where((linearTrue>=projBin[id])*(linearTrue<projBin[id+1]))[0]  for id in range(len(projBin)-1)]

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    cs = ["red","green","blue","black","orange"]
    for id in range(len(binIndex)):
        ax[0].scatter(np.zeros(len(binIndex[id]))+projBin[id],linearPred[binIndex[id]],alpha=0.5,c=[cs[id]])
        vp = ax[0].violinplot(linearPred[binIndex[id]],positions=[projBin[id]+0.1],widths=[0.2],
                                showmeans = False, showmedians = False,showextrema = False)
        for pc in vp['bodies']:
            pc.set_facecolor(cs[id])
            pc.set_edgecolor(cs[id])
    ax[0].set_xlabel("linearized, binned, true")
    ax[0].set_ylabel("linearized_pred")
    # A confusion matrix:
    confMat = np.zeros([len(binIndex),len(binIndex)])
    for id in range(len(binIndex)):
        confMat[id,:] = [ np.sum((linearPred[binIndex[id]]>=projBin[id2])*(linearPred[binIndex[id]]<projBin[id2+1]))
                          for id2 in range(len(projBin)-1)]
    labels=["bottom left","left-arm","middle","right arm","bottom right"]
    im = ax[1].matshow(confMat)
    # We want to show all ticks...
    ax[1].set_xticks(np.arange(len(binIndex)))
    ax[1].set_yticks(np.arange(len(binIndex)))
    # ... and label them with the respective list entries
    ax[1].set_xticklabels(labels)
    ax[1].set_yticklabels(labels)
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(binIndex)):
        for j in range(len(binIndex)):
            text = ax[1].text(j, i, confMat[i,j],
                           ha="center", va="center", color="w")
    ax[1].set_ylabel("true bin")
    ax[1].set_xlabel("predicted bin")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(foldersave, "confusionPlots.png"))
    fig.savefig(os.path.join(foldersave, "confusionPlots.svg"))


    ##Let us compare to a chance decoder that would simply pick randomly an output of the  dataset:
    #TODO: simply pick in LinearTrue...



    # fig,ax = plt.subplots(2,1)
    # ax[0].scatter(predLoss,np.sum(np.square(predPos-truePos),axis=-1),c="black",alpha=0.2)
    # # ax[0].set_yscale("log")
    # ax[0].set_xlabel("predicted error")
    # ax[0].set_ylabel("true error")
    # ax[1].scatter(predLoss,np.abs(linearPred-linearTrue),c="black",alpha=0.2)
    # ax[1].set_xlabel("predicted error")
    # ax[1].set_ylabel("true linear error")
    # fig.tight_layout()
    # # fig.show()
    # fig.savefig(os.path.join(foldersave, "errorsScatter.png"))



def linear_performance_bayes(outputs,linearizationFunction,foldersave,behavior_data,probaLim = 1):
    # - Scatter plot of performances for the bayesian network algorithm
    # - Confusion matrix obtained by projecting the predictions on a linear variable
    # - Error as function of predLoss
    # - Error as function of speed --> to add!
    if not os.path.isdir(foldersave):
        os.makedirs(foldersave)

    timeStepsPred = behavior_data["Position_time"][
        decodebayes.inEpochs(behavior_data["Position_time"][:, 0], behavior_data['Times']['testEpochs'])[0]][:,0]
    truePos = behavior_data["Positions"][
        decodebayes.inEpochs(behavior_data["Position_time"][:, 0], behavior_data['Times']['testEpochs'])[0],:]
    speeds = behavior_data["Speed"][
        decodebayes.inEpochs(behavior_data["Position_time"][:, 0], behavior_data['Times']['testEpochs'])[0],:]

    predLoss = outputs["inferring"][:,2]

    quantileThresh = np.quantile(predLoss,probaLim)

    # at filter = 1, no filtering is applied
    bestPred = (predLoss > quantileThresh)
    goodTimeStep = np.logical_not(np.isnan(np.sum(truePos, axis=1)))
    bestPred = bestPred * goodTimeStep

    predPos = outputs["inferring"][bestPred, 0:2]
    predLoss = predLoss[bestPred]
    truePos = truePos[bestPred,:]
    timeStepsPred = timeStepsPred[bestPred]
    speeds = speeds[bestPred]

    projPredPos, linearPred = linearizationFunction(predPos)
    projTruePos, linearTrue = linearizationFunction(truePos)

    fig,ax = plt.subplots(1+predPos.shape[1],1)
    ax[0].set_xlabel("true")
    ax[0].set_ylabel("prediction")
    for id in range(predPos.shape[1]):
        ax[0].scatter(truePos[:,id],predPos[:,id],alpha=0.3)
        ax[id+1].scatter(timeStepsPred,predPos[:,id],c="red",s=0.3)
        ax[id+1].scatter(timeStepsPred,truePos[:,id],c="black",s=0.3)
        ax[id+1].set_ylabel("feature"+str(id))
        ax[id + 1].set_xlabel("time")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(foldersave,"scatterPlots.png"))


    projBin = np.arange(0,stop=1.2,step=0.2)
    binIndex = [ np.where((linearTrue>=projBin[id])*(linearTrue<projBin[id+1]))[0]  for id in range(len(projBin)-1)]

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    cs = ["red","green","blue","black","orange"]
    for id in range(len(binIndex)):
        ax[0].scatter(np.zeros(len(binIndex[id]))+projBin[id],linearPred[binIndex[id]],alpha=0.5,c=[cs[id]])
        vp = ax[0].violinplot(linearPred[binIndex[id]],positions=[projBin[id]+0.1],widths=[0.2],
                                showmeans = False, showmedians = False,showextrema = False)
        for pc in vp['bodies']:
            pc.set_facecolor(cs[id])
            pc.set_edgecolor(cs[id])
    ax[0].set_xlabel("linearized, binned, true")
    ax[0].set_ylabel("linearized_pred")
    # A confusion matrix:
    confMat = np.zeros([len(binIndex),len(binIndex)])
    for id in range(len(binIndex)):
        confMat[id,:] = [ np.sum((linearPred[binIndex[id]]>=projBin[id2])*(linearPred[binIndex[id]]<projBin[id2+1]))
                          for id2 in range(len(projBin)-1)]
    labels=["bottom left","left-arm","middle","right arm","bottom right"]
    im = ax[1].matshow(confMat)
    # We want to show all ticks...
    ax[1].set_xticks(np.arange(len(binIndex)))
    ax[1].set_yticks(np.arange(len(binIndex)))
    # ... and label them with the respective list entries
    ax[1].set_xticklabels(labels)
    ax[1].set_yticklabels(labels)
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(binIndex)):
        for j in range(len(binIndex)):
            text = ax[1].text(j, i, confMat[i,j],
                           ha="center", va="center", color="w")
    ax[1].set_ylabel("true bin")
    ax[1].set_xlabel("predicted bin")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(foldersave, "confusionPlots.png"))
    fig.savefig(os.path.join(foldersave, "confusionPlots.svg"))


    fig,ax = plt.subplots(2,1)
    ax[0].scatter(predLoss,np.sum(np.square(predPos-truePos),axis=-1),c="black",alpha=0.2)
    # ax[0].set_yscale("log")
    ax[0].set_xlabel("bayesian confidence")
    ax[0].set_ylabel("true error")
    ax[1].scatter(predLoss,np.abs(linearPred-linearTrue),c="black",alpha=0.2)
    ax[1].set_xlabel("bayesian confidence")
    ax[1].set_ylabel("true linear error")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(foldersave, "errorsScatter.png"))

    cm = plt.get_cmap("plasma")
    fig,ax = plt.subplots(2,1)
    sc1 = ax[0].scatter(np.log(np.sum(np.square(predPos-truePos)+10**(-8),axis=-1)),predLoss,c=cm(speeds[:,0]/np.max(speeds)),alpha=0.2)
    cbar = plt.colorbar(sc1,label="speed")
    ax[0].set_xlabel("decoding error (log scale)")
    ax[0].set_ylabel("predicted log-error")
    ax[1].scatter(np.log(np.sum(np.square(predPos-truePos)+10**(-8),axis=-1)),speeds,c=cm(speeds[:,0]/np.max(speeds)),alpha=0.2)
    ax[1].set_ylabel("speed")
    ax[0].set_xlabel("decoding error (log scale)")
    # fig.show()
    fig.tight_layout()
    fig.savefig(os.path.join(foldersave,"errorsScatter_speed.png"))

    fig, ax = plt.subplots(4, 1)
    ax[0].scatter(predPos[:, 0], predLoss, s=1)
    ax[0].set_xlabel(" X pred")
    ax[0].set_ylabel(" proba pred")
    ax[1].scatter(predPos[:, 1], predLoss, s=1)
    ax[1].set_xlabel(" Y pred")
    ax[1].set_ylabel(" proba pred")
    ax[2].scatter(truePos[:, 0], predLoss, s=1)
    ax[2].set_xlabel(" X true")
    ax[2].set_ylabel(" proba pred")
    ax[3].scatter(truePos[:, 1], predLoss, s=1)
    ax[3].set_xlabel(" Y true")
    ax[3].set_ylabel(" proba pred")
    fig.show()
    fig.savefig(os.path.join(foldersave, "posAndErrors.png"))




def compare_bayes_network(outputs,outputsBayes,behavior_data):
    timeStepsPredBayes = behavior_data["Position_time"][
        decodebayes.inEpochs(behavior_data["Position_time"][:, 0], behavior_data['Times']['testEpochs'])[0]][:,0]
    truePosBayes = behavior_data["Positions"][
        decodebayes.inEpochs(behavior_data["Position_time"][:, 0], behavior_data['Times']['testEpochs'])[0],:]
    predLossBayes = outputsBayes["inferring"][:,2]
    predPosBayes = outputsBayes["inferring"][:, 0:2]

    # Does the NN and the Bayesian network make similar prediction?
    predLoss = outputs["predofLoss"]

    bestPred = np.where(predLoss<np.quantile(predLoss,0.3))[0]

    predPos = outputs["featurePred"][bestPred, 0:2]
    truePos = outputs["featureTrue"][bestPred,:]
    speeds =  behavior_data["Speed"][np.ravel(outputs["pos_index"])[bestPred]]
    PosTimePreds = behavior_data["Position_time"][np.ravel(outputs["pos_index"])[bestPred]]
    predLoss = predLoss[bestPred]

    #issue: the neural network dataset is aligned on window of 32 ms with the position corresponding to the last position of the animal in
    # bins of this window
    # The bayesian network on the other side is aligned exactly to the behavior.
    # so we align them
    I2 = decodebayes.inEpochs(behavior_data["Position_time"][:, 0], behavior_data['Times']['testEpochs'])[0]
    indicesBayesToNN = [ np.where(np.equal(I2,pi))[0][0]  for pi in np.ravel(outputs["pos_index"])[bestPred]]

    predLossBayes = predLossBayes[indicesBayesToNN]
    truePosBayes = truePosBayes[indicesBayesToNN] / np.max(predPosBayes)
    predPosBayes = predPosBayes[indicesBayesToNN]/np.max(predPosBayes)
    timeStepsPredBayes = timeStepsPredBayes[indicesBayesToNN]

    ## Next: we restrict to the best prediction according to the Bayesian network:
    indicesBestBayes = np.where(predLossBayes > np.quantile(predLossBayes,0.5))[0]
    predLossBayes = predLossBayes[indicesBestBayes]
    predPosBayes = predPosBayes[indicesBestBayes]
    truePosBayes = truePosBayes[indicesBestBayes]
    timeStepsPredBayes = timeStepsPredBayes[indicesBestBayes]
    predPos = predPos[indicesBestBayes, 0:2]/np.max(truePos)
    truePos = truePos[indicesBestBayes,:]/np.max(truePos)
    speeds =  speeds[indicesBestBayes]
    PosTimePreds = PosTimePreds[indicesBestBayes]
    predLoss = predLoss[indicesBestBayes]


    fig,ax = plt.subplots()
    ax.scatter(predPos[:,0],predPosBayes[:,0],c="blue",alpha=0.2)
    ax.scatter(predPos[:,1],predPosBayes[:,1],c="orange",alpha=0.2)
    ax.set_xlabel("Neural network decoding")
    ax.set_ylabel("Bayesian decoding")
    fig.show()

    fig,ax = plt.subplots(2,1)
    ax[0].scatter(PosTimePreds,predPos[:,0],c="red",s=1)
    ax[0].scatter(PosTimePreds,predPosBayes[:,0],c="purple",s=1)
    ax[0].scatter(PosTimePreds,truePosBayes[:,0],c="black",s=1)
    ax[0].scatter(PosTimePreds, truePos[:, 0], c="grey", s=1)
    ax[1].plot(PosTimePreds,predPos[:,0],c="red")
    ax[1].plot(PosTimePreds,predPosBayes[:,0],c="purple")
    ax[1].plot(PosTimePreds,truePosBayes[:,0],c="black")
    ax[1].plot(PosTimePreds, truePos[:, 0], c="grey")
    fig.show()

    fig, ax = plt.subplots()
    ax.scatter(np.log(np.sum(np.square(predPos - truePos), axis=-1)),np.log(np.sum(np.square(predPosBayes - truePosBayes), axis=-1)),alpha=0.3)
    ax.set_xlabel("Neural Network decoding error")
    ax.set_ylabel("Bayesian decoding error")
    ax.set_title("full test set (habituation)")
    fig.show()

    # from SimpleBayes import  butils
    # xEdges, yEdges, MRF = butils.kde2D(np.sum(np.square(predPos - truePos), axis=-1),
    #                                    np.sum(np.square(predPosBayes - truePosBayes), axis=-1),
    #                                    xbins=45j,ybins=45j,bandwidth=np.max(np.sum(np.square(predPosBayes - truePosBayes), axis=-1))/20)
    #
    # fig,ax = plt.subplots()
    # ax.matshow(np.log(np.transpose(MRF)))
    # ax.set_xticks(range(45))
    # ax.set_yticks(range(45))
    # ax.set_xticklabels(np.round(xEdges[:,0],2))
    # ax.set_yticklabels(np.round(yEdges[0,:],2))
    # fig.show()

    fig, ax = plt.subplots()
    ax.scatter(np.sum(np.square(predPos - truePos), axis=-1),np.sum(np.square(predPosBayes - truePosBayes), axis=-1),alpha=0.3)
    ax.set_xlabel("Neural Network decoding error")
    ax.set_ylabel("Bayesian decoding error")
    ax.set_title("full test set (habituation)")
    fig.show()

    fig,ax = plt.subplots()
    ax.scatter(predLoss,predLossBayes,alpha=0.2)
    ax.set_xlabel("Predicted loss, Neural network")
    ax.set_ylabel("probability (Bayesian network)")
    fig.show()


def compare_sleep_predictions(outputs1,outputs2,o1name="neural network",o2name="bayesian"):
    # Outputs should be dictionaries. Keys correspond to sleep epoch names
    # each value is a list.
    # 1st element: predicted feature
    # 2nd element: predicted loss or probability, which we rename "trust"
    # 3rd element: time step corresponding to the prediction.
    for sleepName in outputs1.keys():
        assert sleepName in outputs2.keys()

        #Let us plot the predicted position as well as the trust in colors
        fig,ax = plt.subplots(2,1)
        cm = plt.get_cmap("turbo")
        ax[0].scatter(outputs1[sleepName][2], outputs1[sleepName][0][:,0],c="black",s=1)
        ax[0].scatter(outputs2[sleepName][2], outputs2[sleepName][0][:,0],c="red",s=1)
        ax[0].set_ylabel("feature 1")
        ax[0].set_title(o1name)
        ax[1].scatter(outputs1[sleepName][2], outputs1[sleepName][0][:,1],c="black",label=o1name,s=1)
        ax[1].scatter(outputs2[sleepName][2], outputs2[sleepName][0][:,1],c="red",label=o2name,s=1)
        ax[1].set_ylabel("feature 2")
        fig.legend()
        fig.show()

        #Let us try a moving average strategy:
        # L = 100
        # newsPos = np.array([outputs1[sleepName][0][id:-L+id] for id in range(L)])
        # movingAverage = np.sum(newsPos,axis=0)
        #
        # # from scipy.fft import  rfft
        # # b = rfft(outputs1[sleepName][1])
        # # spectrum = np.sqrt(np.power(b.real,2)+np.power(b.imag,2))
        # # fig,ax = plt.subplots()
        # # ax.scatter(spectrum,s=1)
        # # fig.show()
        # # # we apply a filter that remove frequencies with small power:
        # # fout = np.copy(b)
        # # fout[(spectrum>15)[:,0],:] = b[(spectrum>15)[:,0],:]
        # # fout[(spectrum <= 15)[:, 0], :] = 0
        # # from scipy.fft import  irfft
        # # res = irfft(fout[:,0],n=len(outputs1[sleepName][1]))
        # # fig,ax = plt.subplots(2,1)
        # # ax[0].plot(fout)
        # # ax[1].plot(res)
        # # fig.show()

        # when plotting all prediction; the results is not very informative.
        # Instead, let us filter all Bayesian and all neural networks predictions:
        bestBayes = np.where(outputs2[sleepName][1] > np.quantile(outputs2[sleepName][1],0.9))[0]
        bestNN =  np.where(outputs1[sleepName][1] > np.quantile(outputs1[sleepName][1],0.9))[0]
        # bestNN =  np.where(outputs1[sleepName][1] > np.quantile(outputs1[sleepName][1],0.95))[0]
        fig,ax = plt.subplots(4,1)
        cm = plt.get_cmap("turbo")
        ax[0].scatter(outputs1[sleepName][2][bestNN], outputs1[sleepName][0][bestNN,0],c="black",s=1,alpha=0.5)
        ax[0].scatter(outputs2[sleepName][2][bestBayes], outputs2[sleepName][0][bestBayes,0],c="red",s=1)
        ax[0].set_ylabel("feature 1")
        ax[1].scatter(outputs1[sleepName][2][bestNN], outputs1[sleepName][0][bestNN,1],c="black",label=o1name,s=1)
        ax[1].scatter(outputs2[sleepName][2][bestBayes], outputs2[sleepName][0][bestBayes,1],c="red",label=o2name,s=1)
        ax[1].set_ylabel("feature 2")
        ax[2].scatter(outputs1[sleepName][0][bestNN,0],outputs1[sleepName][0][bestNN,1],c="black",s=1,alpha=0.1)
        ax[3].scatter(outputs2[sleepName][0][bestBayes,0], outputs2[sleepName][0][bestBayes, 1], c="red",s=1,alpha=0.1)
        fig.legend()
        fig.tight_layout()
        fig.show()

        # fig,ax = plt.subplots()
        # cm = plt.get_cmap("turbo")
        # max = outputs1[sleepName][1][:].max()
        # min = outputs1[sleepName][1][:].min()
        # ax.scatter(outputs1[sleepName][0][:,1], outputs1[sleepName][0][:,0],c=cm((outputs1[sleepName][1][:]-min)/(max-min)),s=1)
        # fig.legend()
        # fig.show()

        #let us align the two sets of predictions:
        from pykeops.numpy import LazyTensor as LazyTensor_np
        import pykeops as pykeops
        # bestBayes = np.where(outputs2[sleepName][1] > np.quantile(outputs2[sleepName][1],0.9))[0]
        # bestNN =  np.where(outputs1[sleepName][1] < np.quantile(outputs1[sleepName][1],0.1))[0]

        timeNN = pykeops.numpy.Vi(outputs1[sleepName][2][bestNN].astype(dtype=np.float64)[:,None])
        timeBayes = pykeops.numpy.Vj(outputs2[sleepName][2][:,None])
        nnIdBayes = (timeBayes-timeNN).abs().argmin(axis=0)
        #[bestNN,:]
        predNNasBayes = outputs1[sleepName][0][bestNN][nnIdBayes[:,0],:]

        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(predNNasBayes[bestBayes,0], outputs2[sleepName][0][bestBayes,0], c="blue", s=1)
        ax[1].scatter(predNNasBayes[bestBayes,1], outputs2[sleepName][0][bestBayes,1], c="orange", s=1)
        ax[0].set_xlabel(o1name+"prediction,repeated to be \n closest in time with"+o2name+ "prediction")
        ax[0].set_ylabel(o2name+"prediction")
        ax[1].set_xlabel(o1name+"prediction,repeated to be \n closest in time with"+o2name+ "prediction")
        ax[1].set_ylabel(o2name+"prediction")
        fig.tight_layout()
        fig.show()

        ## Bayesian proba should be very correlated with space:
        fig,ax = plt.subplots(4,1)
        ax[0].scatter(outputs2[sleepName][0][:,0],outputs2[sleepName][1],s=1)
        ax[0].set_xlabel(o2name+" X pred")
        ax[0].set_ylabel(o2name + " \n proba pred")
        ax[1].scatter(outputs2[sleepName][0][:,1], outputs2[sleepName][1], s=1)
        ax[1].set_xlabel(o2name+" Y pred")
        ax[1].set_ylabel(o2name + " \n proba pred")
        ax[2].scatter(outputs1[sleepName][0][:,0],outputs1[sleepName][1],s=1)
        ax[2].set_xlabel(o1name+" X pred")
        ax[2].set_ylabel(o1name + " \n proba pred")
        ax[3].scatter(outputs1[sleepName][0][:,1], outputs1[sleepName][1], s=1)
        ax[3].set_xlabel(o1name+" Y pred")
        ax[3].set_ylabel(o1name + " \n  proba pred")
        fig.tight_layout()
        fig.show()



    return



def compare_linear_sleep_predictions(outputs1,outputs2,o1name="neural network",o2name="bayesian"):
    # Outputs should be dictionaries. Keys correspond to sleep epoch names
    # each value is a list.
    # 1st element: predicted feature
    # 2nd element: predicted loss or probability, which we rename "trust"
    # 3rd element: time step corresponding to the prediction.
    for sleepName in outputs1.keys():
        assert sleepName in outputs2.keys()
        #Let us plot the predicted position as well as the trust in colors
        fig,ax = plt.subplots(2,1)
        cm = plt.get_cmap("turbo")
        ax[0].scatter(outputs1[sleepName][2], outputs1[sleepName][0],c="black",s=1)
        ax[0].scatter(outputs2[sleepName][2], outputs2[sleepName][0],c="red",s=1)
        ax[0].set_ylabel("linear position")
        fig.legend()
        fig.show()

        #Let us try a moving average strategy:
        # L = 100
        # newsPos = np.array([outputs1[sleepName][0][id:-L+id] for id in range(L)])
        # movingAverage = np.sum(newsPos,axis=0)
        #
        # # from scipy.fft import  rfft
        # # b = rfft(outputs1[sleepName][1])
        # # spectrum = np.sqrt(np.power(b.real,2)+np.power(b.imag,2))
        # # fig,ax = plt.subplots()
        # # ax.scatter(spectrum,s=1)
        # # fig.show()
        # # # we apply a filter that remove frequencies with small power:
        # # fout = np.copy(b)
        # # fout[(spectrum>15)[:,0],:] = b[(spectrum>15)[:,0],:]
        # # fout[(spectrum <= 15)[:, 0], :] = 0
        # # from scipy.fft import  irfft
        # # res = irfft(fout[:,0],n=len(outputs1[sleepName][1]))
        # # fig,ax = plt.subplots(2,1)
        # # ax[0].plot(fout)
        # # ax[1].plot(res)
        # # fig.show()

        # when plotting all prediction; the results is not very informative.
        # Instead, let us filter all Bayesian and all neural networks predictions:
        bestBayes = np.where(outputs2[sleepName][1] > np.quantile(outputs2[sleepName][1],0.9))[0]
        bestNN =  np.where(outputs1[sleepName][1] > np.quantile(outputs1[sleepName][1],0.9))[0]
        # bestNN =  np.where(outputs1[sleepName][1] > np.quantile(outputs1[sleepName][1],0.95))[0]
        fig,ax = plt.subplots(3,1)
        cm = plt.get_cmap("turbo")
        ax[0].scatter(outputs1[sleepName][2][bestNN], outputs1[sleepName][0][bestNN,0],c="black",s=1,alpha=0.5)
        ax[0].scatter(outputs2[sleepName][2][bestBayes], outputs2[sleepName][0][bestBayes,0],c="red",s=1)
        ax[0].set_ylabel("linear position")
        ax[1].scatter(outputs1[sleepName][0][bestNN,0],outputs1[sleepName][0][bestNN,1],c="black",s=1,alpha=0.1)
        ax[2].scatter(outputs2[sleepName][0][bestBayes,0], outputs2[sleepName][0][bestBayes, 1], c="red",s=1,alpha=0.1)
        fig.legend()
        fig.tight_layout()
        fig.show()

        # fig,ax = plt.subplots()
        # cm = plt.get_cmap("turbo")
        # max = outputs1[sleepName][1][:].max()
        # min = outputs1[sleepName][1][:].min()
        # ax.scatter(outputs1[sleepName][0][:,1], outputs1[sleepName][0][:,0],c=cm((outputs1[sleepName][1][:]-min)/(max-min)),s=1)
        # fig.legend()
        # fig.show()



        #let us align the two sets of predictions:
        from pykeops.numpy import LazyTensor as LazyTensor_np
        import pykeops as pykeops
        #
        # bestBayes = np.where(outputs2[sleepName][1] > np.quantile(outputs2[sleepName][1],0.9))[0]
        # bestNN =  np.where(outputs1[sleepName][1] < np.quantile(outputs1[sleepName][1],0.1))[0]

        timeNN = pykeops.numpy.Vi(outputs1[sleepName][2][bestNN].astype(dtype=np.float64)[:,None])
        timeBayes = pykeops.numpy.Vj(outputs2[sleepName][2][:,None])
        nnIdBayes = (timeBayes-timeNN).abs().argmin(axis=0)
        #[bestNN,:]
        predNNasBayes = outputs1[sleepName][0][bestNN][nnIdBayes[:,0],:]

        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(predNNasBayes[bestBayes,0], outputs2[sleepName][0][bestBayes,0], c="blue", s=1)
        ax[1].scatter(predNNasBayes[bestBayes,1], outputs2[sleepName][0][bestBayes,1], c="orange", s=1)
        ax[0].set_xlabel(o1name+"prediction,repeated to be \n closest in time with"+o2name+ "prediction")
        ax[0].set_ylabel(o2name+"prediction")
        ax[1].set_xlabel(o1name+"prediction,repeated to be \n closest in time with"+o2name+ "prediction")
        ax[1].set_ylabel(o2name+"prediction")
        fig.tight_layout()
        fig.show()

        ## Bayesian proba should be very correlated with space:
        fig,ax = plt.subplots(2,1)
        ax[0].scatter(outputs2[sleepName][0][:,0],outputs2[sleepName][1],s=1)
        ax[0].set_xlabel(o2name+" linear pred")
        ax[0].set_ylabel(o2name + " \n proba pred")
        ax[1].scatter(outputs1[sleepName][0][:,0],outputs1[sleepName][1],s=1)
        ax[1].set_xlabel(o1name+" linear pred")
        ax[1].set_ylabel(o1name + " \n proba pred")
        fig.tight_layout()
        fig.show()



    return

