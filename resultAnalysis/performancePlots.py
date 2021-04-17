# Figures to analyze the model performances
import numpy as np
import matplotlib.pyplot as plt
import os
from transformData.linearizer import  uMazeLinearization2,doubleArmMazeLinearization



def linear_performance(outputs,foldersave,filter = 1):
    # - Scatter plot of performances
    # - Confusion matrix obtained by projecting the predictions on a linear variable
    # - Error as function of predLoss
    # - Error as function of speed --> to add!
    if not os.path.isdir(foldersave):
        os.makedirs(foldersave)

    predLoss = outputs["predofLoss"]
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


    fig,ax = plt.subplots(2,1)
    ax[0].scatter(predLoss,np.sum(np.square(predPos-truePos),axis=-1),c="black",alpha=0.2)
    # ax[0].set_yscale("log")
    ax[0].set_xlabel("predicted error")
    ax[0].set_ylabel("true error")
    ax[1].scatter(predLoss,np.abs(linearPred-linearTrue),c="black",alpha=0.2)
    ax[1].set_xlabel("predicted error")
    ax[1].set_ylabel("true linear error")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(foldersave, "errorsScatter.png"))
