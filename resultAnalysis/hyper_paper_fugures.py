import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.getcwd(), 'resultAnalysis', 'ne.mplstyle'))
import seaborn as sns

def boxplot_linError(lErrorNN_mean, lErrorBayes_mean, timeWindows=[36, 108, 252, 504],
                     dirSave=None, suffix=''):
    #TODO: add significance test?
    data = np.round(np.vstack((lErrorNN_mean, lErrorBayes_mean)).flatten(), 2)
    timeWindowsForDF = timeWindows * lErrorNN_mean.shape[0] * 2
    decoderForPD = ['ANN'] * lErrorNN_mean.shape[0] * len(timeWindows) + ['Bayes'] * lErrorNN_mean.shape[0] * len(timeWindows)
    datToPlot = pd.DataFrame({'linearError': data,
                              'timeWindow': timeWindowsForDF,
                              'decoder': decoderForPD})
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.boxplot(data=datToPlot, x="linearError", y="timeWindow", hue="decoder", orient='h', ax=ax)

    if dirSave is not None:
        fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlot{suffix}.png'))
        fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlot{suffix}.svg'))

def barplot_linError_mouse_by_mouse(lErrorNN_mean, lErrorBayes_mean,
                                    timeWindows=[36, 108, 252, 504],
                                    mouseNames=['994', '1199_1', '1199_2', '1223'],
                                    dirSave=None, suffix=''):
    dataByWindow = []
    for iWindow in range(len(timeWindows)):
        data = np.round(np.vstack((lErrorNN_mean[:, iWindow], lErrorBayes_mean[:, iWindow])).flatten(), 2)
        mouseNamesForDF = mouseNames * 2
        decoderForPD = ['ANN'] * lErrorNN_mean.shape[0] + ['Bayes'] * lErrorBayes_mean.shape[0]
        dataByWindow.append(pd.DataFrame({'linearError': data,
                                         'mouseNames': mouseNamesForDF,
                                         'decoder': decoderForPD}))

        fig, ax = plt.subplots(figsize=(9, 9))
        sns.barplot(data=dataByWindow[iWindow], x="mouseNames", y="linearError",
                    hue="decoder", orient='v', ax=ax)

        if dirSave is not None:
            fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlotMBM{suffix}_{timeWindows[iWindow]}.png'))
            fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlotMBM{suffix}_{timeWindows[iWindow]}.svg'))