import numpy as np
import seaborn as sns

def barplot_linError(lErrorNN_mean, lErrorBayes_mean):
    # TODO: trnsform data to pandas dataframe or other beautiful format for seaborn
    datToPlot = np.vstack((lErrorNN_mean, lErrorBayes_mean)).T
    sns.boxplot(data=datToPlot, x=['NN', 'Bayes'], y='absolute linear error')