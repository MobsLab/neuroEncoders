import numpy as np
import seaborn as sns

def barplot_linError(lErrorNN_mean, lErrorBayes_mean):
    datToPlot = np.vstack((lErrorNN_mean, lErrorBayes_mean)).T
    sns.boxplot(data=datToPlot, x=['NN', 'Bayes'], y='absolute linear error')