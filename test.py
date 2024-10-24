#!/usr/bin/env python
# neuroEncoder by the Memory, Oscillations and Brain States (MOBS) team
# 2017-2024
# by Théotime de Charrin, Thibault Balenbois, Pierre Orhan and Dmitri Bryzgalov
# theotime.decharrin@gmail.com; t.balenbois@gmail.com; brygalovdm@gmail.com

import os
import subprocess

from importData import rawdata_parser

# Load custom code
from utils.global_classes import Params, Project

# Load standard libs


# FILENAME = "/media/mickey/DimaERC2/TEST1_Basile/M1199_20210408_UMaze.xml"
# FILENAME = "/media/mickey/DimaERC2/TEST3_Basile_M1239/M1239_20211105_StimMFBWake.xml"
FILENAME = os.path.expanduser("~/Documents/Theotime/M1199/M1199_20210408_UMaze.xml")
nameExp = "TESTTHEOTIME"
ProjectPath = Project(os.path.expanduser(FILENAME), nameExp=nameExp)
folderCode = str(os.getcwd())

# Creating the file nnBehavior.mat
#####################################################################################################################

if not (
    os.path.isfile(
        os.path.expanduser(os.path.split(FILENAME)[0] + os.path.sep + "nnBehavior.mat")
    )
):
    subprocess.run(
        [
            "./getTsdFeature.sh",
            os.path.expanduser(FILENAME.strip("'")),
            "'" + "pos" + "'",
        ]
    )


##############################################################################################################################
# rawdata_parser.speed_filter(ProjectPath.folder, overWrite = True)
# rawdata_parser.select_epochs(ProjectPath.folder, overWrite = True)

# Creating the datasets
#################################################################################################################################

BUFFERSIZE = 72000


def julia_spike_filter(projectPath, folderCode, windowSize=0.200, singleSpike=False):
    # Launch an extraction of the spikes in Julia:
    if singleSpike:
        test1 = os.path.isfile(
            (os.path.join(projectPath.folder, "dataset", "dataset_singleSpike.tfrec"))
        )
    else:
        test1 = os.path.isfile(
            (
                os.path.join(
                    projectPath.folder,
                    "dataset",
                    "dataset_stride" + str(round(windowSize * 1000)) + ".tfrec",
                )
            )
        )
    if not test1:
        if not os.path.exists(os.path.join(projectPath.folder, "nnBehavior.mat")):
            raise ValueError(
                "the behavior file does not exist :"
                + os.path.join(projectPath.folder, "nnBehavior.mat")
            )
        if not os.path.exists(projectPath.dat):
            raise ValueError("the dat file does not exist :" + projectPath.dat)
        codepath = os.path.join(folderCode, "importData/juliaData/")
        if singleSpike:
            subprocess.run(
                [
                    codepath + "executeFilter_singleSpike.sh",
                    codepath,
                    projectPath.xml,
                    projectPath.dat,
                    os.path.join(projectPath.folder, "nnBehavior.mat"),
                    os.path.join(projectPath.folder, "spikeData_fromJulia.csv"),
                    os.path.join(
                        projectPath.folder, "dataset", "dataset_singleSpike.tfrec"
                    ),
                    os.path.join(
                        projectPath.folder, "dataset", "datasetSleep_singleSpike.tfrec"
                    ),
                    str(BUFFERSIZE),
                    str(windowSize),
                ]
            )
        else:
            subprocess.run(
                [
                    os.path.join(codepath, "executeFilter_stride.sh"),
                    codepath,
                    projectPath.xml,
                    projectPath.dat,
                    os.path.join(projectPath.folder, "nnBehavior.mat"),
                    os.path.join(projectPath.folder, "spikeData_fromJulia.csv"),
                    os.path.join(
                        projectPath.folder,
                        "dataset",
                        "dataset_stride" + str(round(windowSize * 1000)) + ".tfrec",
                    ),
                    os.path.join(
                        projectPath.folder,
                        "dataset",
                        "datasetSleep_stride"
                        + str(round(windowSize * 1000))
                        + ".tfrec",
                    ),
                    str(BUFFERSIZE),
                    str(windowSize),
                    str(0.200),
                ]
            )  # the striding is 36ms based...


julia_spike_filter(ProjectPath, folderCode)

############################################################################################################################

rawdata_parser.speed_filter(ProjectPath.folder, overWrite=True)
rawdata_parser.select_epochs(ProjectPath.folder, overWrite=True)

# Using the model
#################################################################################################################################

rawdata_parser.DataHelper(ProjectPath, "ann")

import os

from openEphysExport.generate_json import generate_json
from resultAnalysis import print_results
from transformData.linearizer import UMazeLinearizer

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Create parameters

DataHelper = rawdata_parser.DataHelper(ProjectPath, "ann")

Parameters = Params(DataHelper, 0.200)
# Create linearization function
Linearizer = UMazeLinearizer(ProjectPath.folder)
Linearizer.verify_linearization(
    DataHelper.positions / DataHelper.maxPos(), ProjectPath.folder, overwrite=False
)
l_function = Linearizer.pykeops_linearization

isPL = False

# Network : train and test

from fullEncoder import an_network as Training

NNTrainer = Training.LSTMandSpikeNetwork(
    ProjectPath, Parameters, deviceName="/device:GPU:0"
)
NNTrainer.fix_linearizer(Linearizer.mazePoints, Linearizer.tsProj)
NNTrainer.train(DataHelper.fullBehavior, windowsizeMS=200, isPredLoss=isPL)
NNTrainer.test(
    DataHelper.fullBehavior, l_function=l_function, windowsizeMS=200, isPredLoss=isPL
)
print_results.print_results(NNTrainer.folderResult, windowSizeMS=200)
# Create json

modelPath = os.path.join(NNTrainer.folderModels, str(0.200), "savedModels", "fullModel")
generate_json(ProjectPath, modelPath, DataHelper.list_channels)

# Network : test with an existing model (turn speed filter off before running this section)
"""
from fullEncoder import an_network as Tester

NNTester = Tester.LSTMandSpikeNetwork(ProjectPath, Parameters, deviceName='/device:GPU:0')
NNTester.test(DataHelper.fullBehavior, l_function=l_function, windowsizeMS=200, isPredLoss=isPL)
print_results.print_results(NNTester.folderResult, windowSizeMS=200)
"""
# Decoding during sleep (turn epoch selection off before running this section)
"""
from fullEncoder import an_network as Training

NNTrainer = Training.LSTMandSpikeNetwork(ProjectPath, Parameters,
                                        deviceName='/device:GPU:0')
NNTrainer.fix_linearizer(Linearizer.mazePoints, Linearizer.tsProj)
NNTrainer.testSleep(DataHelper.fullBehavior, l_function=l_function,
                                    windowsizeMS=36, isPredLoss=isPL)
print_results.print_results(NNTrainer.folderResult, windowSizeMS=36)

# Create json

modelPath = os.path.join(NNTrainer.folderModels, str(0.036),
                                    'savedModels', 'fullModel')
generate_json(ProjectPath, modelPath, DataHelper.list_channels)
"""
# Bayesian (does not work)
"""
#Getting the TimeStepsPred in order to test_as_NN
import pandas as pd

TSP = pd.read_csv(os.path.expanduser(os.path.join(ProjectPath.resultsPath, "results", str(windowSizeMS), 'timeStepsPred.csv'))).values[:,1:]

from simpleBayes import decode_bayes as Training

TrainerBayes = Training.Trainer(ProjectPath)
bayesMatrices = TrainerBayes.train_order_by_pos(DataHelper.fullBehavior,
                                                  				l_function)
outputs = TrainerBayes.test_as_NN(DataHelper.fullBehavior, bayesMatrices, TSP,
                                       			windowSizeMS)
print_results.print_results(TrainerBayes.folderResult, typeDec='bayes',
                            			results = outputs, windowSizeMS=504)

"""