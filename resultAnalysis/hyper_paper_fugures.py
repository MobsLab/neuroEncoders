import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.getcwd(), 'resultAnalysis', 'ne.mplstyle'))
import seaborn as sns
from scipy.stats import sem

cm = plt.get_cmap("tab20b")
colorsForSNS = [cm(14), cm(2)]
colorsForSNSF = [cm(12), cm(16)]

def boxplot_linError(lErrorNN_mean, lErrorBayes_mean, timeWindows=[36, 108, 252, 504],
                     dirSave=None, suffix=''):
    #TODO: add significance test?
    data = np.round(np.vstack((lErrorNN_mean, lErrorBayes_mean)).flatten(), 2)
    timeWindowsForDF = timeWindows * lErrorNN_mean.shape[0] * 2
    decoderForPD = ['ANN'] * lErrorNN_mean.shape[0] * len(timeWindows) + ['Bayes'] * lErrorNN_mean.shape[0] * len(timeWindows)
    datToPlot = pd.DataFrame({'linearError': data,
                              'timeWindow (ms)': timeWindowsForDF,
                              'decoder': decoderForPD})
    fig, ax = plt.subplots(figsize=(9, 9))
    myPalette = {"ANN": colorsForSNS[0], "Bayes": colorsForSNS[1]}
    sns.boxplot(data=datToPlot, x="timeWindow (ms)", y="linearError",
                hue="decoder", orient='v', ax=ax, palette=myPalette)

    if dirSave is not None:
        fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlot{suffix}.png'))
        fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlot{suffix}.svg'))

def boxplot_euclError(errorNN_mean, errorBayes_mean, timeWindows=[36, 108, 252, 504],
                     dirSave=None, suffix=''):
    #TODO: add significance test?
    data = np.round(np.vstack((errorNN_mean, errorBayes_mean)).flatten(), 2)
    timeWindowsForDF = timeWindows * errorNN_mean.shape[0] * 2
    decoderForPD = ['ANN'] * errorNN_mean.shape[0] * len(timeWindows) + ['Bayes'] * errorNN_mean.shape[0] * len(timeWindows)
    datToPlot = pd.DataFrame({'eucl. error (cm)': data,
                              'timeWindow (ms)': timeWindowsForDF,
                              'decoder': decoderForPD})
    fig, ax = plt.subplots(figsize=(9, 9))
    myPalette = {"ANN": colorsForSNS[0], "Bayes": colorsForSNS[1]}
    sns.boxplot(data=datToPlot, x="timeWindow (ms)", y="eucl. error (cm)", hue="decoder",
                orient='v', ax=ax, palette=myPalette)

    if dirSave is not None:
        fig.savefig(os.path.join(dirSave, f'errorBoxPlot{suffix}.png'))
        fig.savefig(os.path.join(dirSave, f'errorBoxPlot{suffix}.svg'))

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
        myPalette = {"ANN": colorsForSNS[0], "Bayes": colorsForSNS[1]}
        sns.barplot(data=dataByWindow[iWindow], x="mouseNames", y="linearError",
                    hue="decoder", orient='v', ax=ax, palette=myPalette)

        if dirSave is not None:
            fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlotMBM{suffix}_{timeWindows[iWindow]}.png'))
            fig.savefig(os.path.join(dirSave, f'linearErrorBoxPlotMBM{suffix}_{timeWindows[iWindow]}.svg'))

def barplot_euclError_mouse_by_mouse(errorNN_mean, errorBayes_mean,
                                    timeWindows=[36, 108, 252, 504],
                                    mouseNames=['994', '1199_1', '1199_2', '1223'],
                                    dirSave=None, suffix=''):
    dataByWindow = []
    for iWindow in range(len(timeWindows)):
        data = np.round(np.vstack((errorNN_mean[:, iWindow], errorBayes_mean[:, iWindow])).flatten(), 2)
        mouseNamesForDF = mouseNames * 2
        decoderForPD = ['ANN'] * errorNN_mean.shape[0] + ['Bayes'] * errorBayes_mean.shape[0]
        dataByWindow.append(pd.DataFrame({'eucl. error (cm)': data,
                                          'mouseNames': mouseNamesForDF,
                                          'decoder': decoderForPD}))

        fig, ax = plt.subplots(figsize=(9, 9))
        myPalette = {"ANN": colorsForSNS[0], "Bayes": colorsForSNS[1]}
        sns.barplot(data=dataByWindow[iWindow], x="mouseNames", y="eucl. error (cm)",
                    hue="decoder", orient='v', ax=ax, palette=myPalette)

        if dirSave is not None:
            fig.savefig(os.path.join(dirSave, f'errorBoxPlotMBM{suffix}_{timeWindows[iWindow]}.png'))
            fig.savefig(os.path.join(dirSave, f'errorBoxPlotMBM{suffix}_{timeWindows[iWindow]}.svg'))

def plot_euclError_mouse_by_mouse(errorNN_mean, errorBayes_mean,
                                  errorNN_err, errorBayes_err,
                                  timeWindows=[36, 108, 252, 504],
                                  mouseNames=['994', '1199_1', '1199_2', '1223'],
                                  dirSave=None, suffix=''):
    for iWindow in range(len(timeWindows)):

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.errorbar(mouseNames, errorNN_mean[:, iWindow], yerr=errorNN_err[:, iWindow],
                    fmt='-o', color=colorsForSNS[0], label='ANN')
        ax.errorbar(mouseNames, errorBayes_mean[:, iWindow], yerr=errorBayes_err[:, iWindow],
                    fmt='-o', color=colorsForSNS[1], label='Bayes')
        ax.set_xlabel('Mouse ID')
        ax.set_ylabel('Euclidean error (cm)')
        ax.set_title(f'Euclidean error for {timeWindows[iWindow]} ms time window')
        ax.legend()

        if dirSave is not None:
            fig.savefig(os.path.join(dirSave, f'errorPlotMBM{suffix}_{timeWindows[iWindow]}.png'))
            fig.savefig(os.path.join(dirSave, f'errorPlotMBM{suffix}_{timeWindows[iWindow]}.svg'))


def boxplot_euclDist_decoders(distMean, timeWindows=[36, 108, 252, 504],
                              dirSave=None, suffix=''):
    #TODO: add significance test?
    data = distMean.flatten()
    timeWindowsForDF = timeWindows * distMean.shape[0]
    datToPlot = pd.DataFrame({'eucl. distance (cm)': data,
                              'time window (ms)': timeWindowsForDF})
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.boxplot(data=datToPlot, x="time window (ms)", y="eucl. distance (cm)",
                orient='v', ax=ax, palette='flare')

    if dirSave is not None:
        fig.savefig(os.path.join(dirSave, f'euclDistDecPlot{suffix}.png'))
        fig.savefig(os.path.join(dirSave, f'euclDistDecPlot{suffix}.svg'))


def fig_eucl_error_filtered(errorNN_mean, FerrorNN_mean, timeWindows=[36, 108, 252, 504],
                            dirSave=None, suffix=''):
    #TODO: add significance test?
    data = np.round(np.vstack((errorNN_mean, FerrorNN_mean)).flatten(), 2)
    timeWindowsForDF = timeWindows * errorNN_mean.shape[0] * 2
    decoderForPD = ['Full'] * errorNN_mean.shape[0] * len(timeWindows) + ['30% best'] * errorNN_mean.shape[0] * len(timeWindows)
    datToPlot = pd.DataFrame({'eucl. error (cm)': data,
                              'timeWindow (ms)': timeWindowsForDF,
                              'filtered': decoderForPD})
    fig, ax = plt.subplots(figsize=(9, 9))
    myPalette = {"ANN": colorsForSNS[0], "Bayes": colorsForSNS[1]}
    sns.boxplot(data=datToPlot, x="timeWindow (ms)", y="eucl. error (cm)", hue="filtered",
                orient='v', ax=ax, palette=colorsForSNSF)

    if dirSave is not None:
        fig.savefig(os.path.join(dirSave, f'errorBoxPlot_filtered{suffix}.png'))
        fig.savefig(os.path.join(dirSave, f'errorBoxPlot_filtered{suffix}.svg'))

def fig_average_predLoss_vs_euclError(predLossTicks, euclError, timeWindows=[36, 108, 252, 504],
                                      dirSave=None, suffix=''):

    # Stack euclError into a single array for each time window
    euclErrorAv = []
    euclErrorSEM = []
    for iw in range(len(timeWindows)):
        euclErrorAv.append(np.nanmean(euclError[:, iw, :], axis=0))
        euclErrorSEM.append(sem(euclError[:, iw, :], axis=0, nan_policy='omit'))

    fig, ax = plt.subplots(figsize=(15, 9))
    for iw in range(len(timeWindows)):
        ax.plot(predLossTicks, euclErrorAv[iw], c=cm(12+iw), label=f'{timeWindows[iw]} ms')
        ax.fill_between(predLossTicks, euclErrorAv[iw]-euclErrorSEM[iw],
                         euclErrorAv[iw]+euclErrorSEM[iw], color=cm(12+iw), alpha=0.2)
    ax.set_xlabel('Normalized predicted loss')
    ax.set_ylabel('eucl. error (cm)')
    ax.set_title('Average euclidean error vs. predicted loss')
    ax.legend()

    if dirSave is not None:
        fig.savefig(os.path.join(dirSave, f'euclError_vs_predLoss{suffix}.png'))
        fig.savefig(os.path.join(dirSave, f'euclError_vs_predLoss{suffix}.svg'))
