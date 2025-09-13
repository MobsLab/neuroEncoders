# %%
import os
import matplotlib.pyplot as plt

# Load standard libs
import sys
import pandas as pd

# Load custom code
from neuroencoders.utils.global_classes import Params, Project, save_project_to_pickle
from neuroencoders.utils.global_classes import DataHelper as DataHelperClass
from neuroencoders.importData import rawdata_parser
from neuroencoders.resultAnalysis import print_results
from neuroencoders.transformData.linearizer import UMazeLinearizer
from neuroencoders.utils import management, MOBS_Functions
from neuroencoders.fullEncoder import an_network as Training
from neuroencoders.importData.juliaData.julia_data_parser import julia_spike_filter
from neuroencoders.openEphysExport.generate_json import generate_json
import numpy as np
from importlib import reload
from neuroencoders.importData import epochs_management as ep

# %%
nameExp = "new_4d_GaussianHeatMap_LinearLoss"
nameExp_Transformer = "new_4d_GaussianHeatMap_LinearLoss_Transformer"
nameExp_LSTM = "current_LogLoss_Transformer_Dense_LSTM"

# %%
jsonPath = None
windowSizeMS = [36, 108]
mode = "ann"
target = "pos"
phase = "pre"
nEpochs = 200
mouse = "1199"
manipe = "PAG"

# %%
from neuroencoders.utils.MOBS_Functions import path_for_experiments_df


Dir = path_for_experiments_df("Sub", nameExp)
Dir_Transformer = path_for_experiments_df("Sub", nameExp_Transformer)
Dir_LSTM = path_for_experiments_df("Sub", nameExp_LSTM)

# %% [markdown]
# ## mobs_function

# %%
mice_nb = [
    "M1199_PAG",
    "M994_PAG",
    "M1239_MFB",
    "M1230_Novel",
    "M1230_Known",
    "M1162_MFB",
]
mice_names = ["1199", "994", "1239", "1230", "1230", "1162"]
mice_manipes = ["PAG", "PAG", "MFB", "Novel", "Known", "MFB"]

# %%
Dir_LSTM

# %%
windowSizeMS

# %%
from neuroencoders.utils.MOBS_Functions import Results_Loader, Mouse_Results

# %%
mice_manipes

# %%
loader_Transformer = Results_Loader(
    dir=Dir_Transformer,
    mice_nb=mice_names,
    mice_manipes=mice_manipes,
    target="pos",
    timeWindows=windowSizeMS,
    phases=["pre", "cond", "post"],
    template="pre",
    verbose=False,
    which="both",
    load_bayes=True,
    load_bayesMatrices = False,
    load_pickle = False,
    deviceName="cpu",
    extract_spikes_count = False,
)

# %%
loader_Transformer.convert_to_df(redo = True)

# %%
from pathlib import Path)

# %%
loader_all = loader_Transformer

# %%
# check if nameExp contains "LSTM" or "Transformer" to set the ann_mode row by row
loader_all.results_df["ann_mode"] = "Transformer"
loader_all.results_df.loc[
    loader_all.results_df["nameExp"].str.contains("LSTM"), "ann_mode"
] = "LSTM"

# %%
loader_all.results_df.sort_values(by=["ann_mode", "mouse", "phase"], inplace=True)

# %%
loader_all.results_df

# %%
loader_all.apply_analysis()

# %%
loader_all.results_df.head()

# %%
loader_all.save(
    path=os.path.join(
        Path.home(),
        "Documents",
        "Theotime",
        "DimaERC2",
        "neuroencoders_1021",
        "_work",
        "4d_NOPREDLOSS_results_all.pickle",
    )
)

# %%
loader_all.results_df

# %%
main_dir = os.path.realpath(
    "/home/mickey/Dropbox/Mobs_member/Theotime_De_Charrin/Figures/"
)

# %%
loader_Transformer.results_df

# %%
loader_all.results_dict[nameExp_Transformer]["M1199PAG"][
    "cond"
].error_matrix_linerrors_by_speed()

# %% [markdown]
# ## save dataframe as mat struct

# %% [markdown]
# from scipy.io import savemat
#
# # Save the DataFrame as a .mat file
#
# mdict = dict()
# df = loader_all.results_df.copy()
# for col in df.drop(columns=["results"]).columns:
#     mdict[col] = df[col].to_numpy()
#
# # Save the dictionary as a .mat file
# savemat(os.path.join(main_dir, "..", "resultsNN_withColumns.mat"), mdict)

# %% [markdown]
# ## save dataframe as csv file

# %% [markdown]
# loader_all.results_df.drop(columns=["results"]).to_csv(
#     os.path.join(main_dir, "..", "dataFrame.csv"), index=False
# )

# %% [markdown]
# ## analysis

# %%
import seaborn as sns

# %%
from matplotlib.colors import Normalize

corr = (
    loader_all.results_df.query("phase == 'pre'")
    .drop(columns="real_asymmetry_ratio")
    .corr(numeric_only=True)
)
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    cmap="coolwarm",
    norm=Normalize(vmin=-1, vmax=1),
)

for winMS, Mouse_Results in zip(loader_all.results_df["winMS"].values), loader_all.results_df["results"].values):
    try:
        # %%

        # %%

        # %%
        Mouse_Results.resultsNN.keys()

        # %%
        inferring = Mouse_Results.resultsNN["fullPred"][0]
        linferring = Mouse_Results.resultsNN["linPred"][0]
        pos = Mouse_Results.resultsNN["truePos"][0]
        lpos = Mouse_Results.resultsNN["linTruePos"][0]

        error = np.array(
            [np.linalg.norm(inferring[i, :] - pos[i, :]) for i in range(inferring.shape[0])]
        )  # eucledian distance

        # %%
        error_mask = error > 0.6

        # %%
        linferring.min()

        # %%
        plt.hist2d(
            lpos,
            linferring,
            cmap="viridis",
            bins=[np.linspace(0, 1, 50), np.linspace(0, 1, 50)],
        )

        # %%
        plt.hist2d(inferring[error_mask, 0], inferring[error_mask, 1], cmap="viridis")
        plt.colorbar()

        # %%
        np.where(
            np.linalg.norm(
                [Mouse_Results.resultsNN["fullPred"][0], Mouse_Results.resultsNN["truePos"][0]]
            )
            > 0
        )[0]


        # %%
        from neuroencoders.importData.epochs_management import inEpochsMask

        trainMask = inEpochsMask(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][:, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Times"]["trainEpochs"],
        )
        testMask = inEpochsMask(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][:, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Times"]["testEpochs"],
        )
        speedMask = Mouse_Results.data_helper["108"].fullBehavior["Times"]["speedFilter"]

        mask = (trainMask | testMask) * speedMask

        # %%
        plt.plot(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][:, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Positions"][:, 0],
            "o",
            markersize=1,
            alpha=0.5,
            c="r",
        )
        plt.plot(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][trainMask, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Positions"][trainMask, 0],
            "o",
            markersize=1,
            alpha=1,
        )
        plt.plot(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][trainMask, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Positions"][trainMask, 1],
            "o",
            markersize=1,
            alpha=1,
        )

        # %%
        training_data = Mouse_Results.data_helper["108"].fullBehavior["Positions"][mask]
        training_data.shape

        # %%
        plt.plot(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][:, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Positions"][:, 0],
            "o",
            markersize=1,
            alpha=0.3,
            c="r",
        )
        plt.plot(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][mask, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Positions"][mask, 0],
            "o",
            markersize=1,
            alpha=1,
        )
        plt.plot(
            Mouse_Results.data_helper["108"].fullBehavior["positionTime"][mask, 0],
            Mouse_Results.data_helper["108"].fullBehavior["Positions"][mask, 1],
            "o",
            markersize=1,
            alpha=1,
        )

        # %%
        Mouse_Results.fig_example_XY(int(winMS))

        # %%
        Mouse_Results.fig_example_linear_filtered(fprop=0.1)

        # %%
        Mouse_Results.compare_nn_bayes(int(winMS), isShow=True)

        # %%
        Mouse_Results.mean_euclerrors()

        # %%
        Mouse_Results.predLoss_vs_trueLoss()

        # %%
        Mouse_Results.nnVSbayes()

        # %%
        Mouse_Results.mean_linerrors()

        # %%
        Mouse_Results.predLoss_linError(speed="slow", step=1e-6)

        # %%
        Mouse_Results.predLoss_euclError(step=1e-6, scaled=False)

        # %%
        Mouse_Results.fig_example_2d(speed="fast")

        # %%
        Mouse_Results.hist_linerrors(speed="fast")

        # %%
        Mouse_Results.Params.phase

        # %%
        Mouse_Results.run_spike_alignment(useTrain=False)

        # %%
        Mouse_Results.plot_pc_tuning_curve_and_predictions(ws=int(winMS))
    except:
        print(f"Error with {Mouse_Results.mouse}{Mouse_Results.manip}")

# %%
Mouse_Results.DataHelper.fullBehavior["positionTime"]

# %%
Mouse_Results.load_results(force=True, phase="pre")

# %%
from importData.epochs_management import inEpochs


fig, ax = plt.subplots()

ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)

trainMask = inEpochs(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"],
    Mouse_Results.data_helper["108"].fullBehavior["Times"]["trainEpochs"],
)[0]
testMask = inEpochs(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"],
    Mouse_Results.data_helper["108"].fullBehavior["Times"]["testEpochs"],
)[0]

ax1.plot(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"][trainMask],
    Mouse_Results.data_helper["108"].fullBehavior["Positions"][trainMask, 0],
    "--.",
    color="black",
    label="training",
    markersize=6,
)
ax1.plot(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"][testMask],
    Mouse_Results.data_helper["108"].fullBehavior["Positions"][testMask, 0],
    "--.",
    color="red",
    label="testing",
    markersize=6,
)

ax2.plot(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"][trainMask],
    Mouse_Results.data_helper["108"].fullBehavior["Positions"][trainMask, 1],
    "--.",
    color="black",
    label="training",
    markersize=6,
)
ax2.plot(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"][testMask],
    Mouse_Results.data_helper["108"].fullBehavior["Positions"][testMask, 1],
    "--.",
    color="red",
    label="testing",
    markersize=6,
)

# %%
print_results.print_results(
    Mouse_Results.folderResult, show=True, windowSizeMS=108, phase=phase, target=target
)

# %%
Mouse_Results.data_helper["108"].fullBehavior["Times"]["SessionEpochs"]["pre"]

# %%
plt.plot(np.arange(0, 1000, 1), np.random.rand(1000), label="test")

# %%
plt.plot(testEpochs[0])
plt.show()

# %%
from importData.epochs_management import inEpochs

testEpochs = inEpochs(
    Mouse_Results.data_helper["108"].fullBehavior["positionTime"].flatten(),
    Mouse_Results.data_helper["108"].fullBehavior["Times"]["testEpochs"],
)[0]
plt.plot(
    Mouse_Results.data_helper["108"].fullBehavior["Positions"][testEpochs, 0],
    Mouse_Results.data_helper["108"].fullBehavior["Positions"][testEpochs, 1],
)
plt.show()

# %%
Mouse_Results.projects["108"]

# %%
print(Mouse_Results.projects["108"])

# %%
Mouse_Results.load_trainers()

# %%
Mouse_Results.load_results(force=True)


# %%
NNTrainer = Training.LSTMandSpikeNetwork(
    ProjectPath,
    Parameters[winMS],
    deviceName=deviceName,
    debug=False,
    phase=phase,
)

# %%
DataHelper.resultsPath

# %%
featurePred = pd.read_csv(
    os.path.join(DataHelper.resultsPath, f"featurePred_{phase}.csv")
).to_numpy()
featureTrue = pd.read_csv(
    os.path.join(DataHelper.resultsPath, f"featureTrue_{phase}.csv")
).to_numpy()
lossPred = pd.read_csv(
    os.path.join(DataHelper.resultsPath, f"lossPred_{phase}.csv")
).to_numpy()
speedMask = pd.read_csv(
    os.path.join(DataHelper.resultsPath, f"speedMask_{phase}.csv")
).to_numpy()

linearPred = pd.read_csv(
    os.path.join(DataHelper.resultsPath, f"linearPred_{phase}.csv")
).to_numpy()
linearTrue = pd.read_csv(
    os.path.join(DataHelper.resultsPath, f"linearTrue_{phase}.csv")
).to_numpy()

# %%
featurePred = featurePred[:, -2]
featureTrue = featureTrue[:, -2]
lossPred = lossPred[:, -1]
speedMask = speedMask[:, -1]

# %%
DataHelper.fullBehavior["Times"]["testEpochs"]

# %%
from neuroencoders.importData.epochs_management import inEpochs, inEpochsMask

timeStepPred = DataHelper.fullBehavior["positionTime"][
    inEpochs(
        DataHelper.fullBehavior["positionTime"][:, 0],
        DataHelper.fullBehavior["Times"]["testEpochs"],
    )
]

# %%
timeStepPred

# %%
plt.plot(featureTrue)

# %%
qControltmp = lossPred.copy()
temp = lossPred.argsort(axis=0)

# %%
thresh = np.squeeze(qControltmp[temp[int(len(temp) * 0.5)]])

# %%
np.quantile(lossPred, 0.5)

# %%
selection = np.squeeze(qControltmp < thresh)

# %%
inferring = featurePred
pos = featureTrue

# %%
selection

# %%
temp


# %%
lossPred

# %%
plt.hist(featureTrue, label="True Feature")
plt.hist(featurePred, label="Predicted Feature")
plt.legend()

# %%
for quantile in [0.1, 0.2, 0.3, 0.4, 0.5]:
    qControl = np.quantile(lossPred, quantile)
    plt.figure()
    plt.title(f"Quantile {quantile}")
    plt.plot(
        featurePred[lossPred <= qControl],
        np.abs(featureTrue[lossPred <= qControl] - featurePred[lossPred <= qControl]),
        "k.",
    )
    mean_error = np.mean(
        np.abs(featureTrue[lossPred <= qControl] - featurePred[lossPred <= qControl])
    )
    plt.axhline(
        mean_error, color="r", linestyle="--", label=f"Mean Error: {mean_error:.2f}"
    )
    plt.xlabel("Predicted Feature")

# %%
DataHelper.globalResultsPath

# %%
reload(print_results)
print_results.print_results(
    NNTrainer.folderResult,
    windowSizeMS=108,
    lossSelection=0.5,
    target="pos",
    phase=phase,
)

# %% [markdown]
# ## waveform comparator

# %%
from importData.compareSpikeFiltering import WaveFormComparator

# %%
Mouse_Results.run_spike_alignment(force=True)

# %%
waveform = WaveFormComparator(
    Mouse_Results.Project,
    Mouse_Results.Params,
    Mouse_Results.DataHelper.fullBehavior,
    windowSizeMS=Mouse_Results.windowSizeMS,
    useTrain=True,
)

# %%
waveform.save_alignment_tools(
    Mouse_Results.bayes,
    Mouse_Results.l_function,
    windowSizeMS=Mouse_Results.windowSizeMS,
)

# %% [markdown]
# ## temp

# %%
projectPath = Project(
    os.path.join(
        Dir[
            (Dir["name"].str.contains(mouse)) & (Dir["manipe"].str.contains(manipe))
        ].path.values[0],
        "amplifier.xml",
    ),
    nameExp=nameExp,
    windowSize=0.108,
)

# %%
projectPath.experimentPath

# %%
windowSizeMS

# %%
DataHelper = DataHelperClass(
    projectPath.xml,
    windowSize=0.108,
    mode=mode,
    target=target,
    phase=phase,
    nameExp=nameExp,
)

# %%
DataHelper.folderResult

# %%
Linearizer = UMazeLinearizer(projectPath.folder, phase=phase)


# %%
Linearizer.verify_linearization(
    DataHelper.positions / DataHelper.maxPos(), projectPath.folder, overwrite=False
)

# %%
l_function = Linearizer.pykeops_linearization

# %%
DataHelper.get_true_target(l_function, in_place=True, show=True)

# %%
windowSizeMS

# %%
Parameters = Params(
    helper=DataHelper,
    windowSize=0.108,
    nEpochs=nEpochs,
    phase=phase,
    batchSize=256,
    save_json=True,
)

# %%
Parameters.folderResult

# %%
Parameters.resultsPath

# %%
save_project_to_pickle(projectPath)

# %%
save_project_to_pickle(
    Parameters, output=os.path.join(Parameters.resultsPath, "Parameters.pkl")
)
