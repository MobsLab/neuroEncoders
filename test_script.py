#!env/bin/python3
# test file....

# Load standard libs
import sys
import os.path
import subprocess
import numpy as np
import tensorflow as tf
# Load custum code
from utils.global_classes import Project, Params
# Get codes into path (jsut in case)
folder_code = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
sys.path.insert(0,folder_code)
# Custom codes
from utils import management
from importData import rawdata_parser
from transformData.linearizer import UMazeLinearizer
from resultAnalysis import print_results
from importData.juliaData.julia_data_parser import julia_spike_filter
from fullEncoder import an_network
from simpleBayes import decode_bayes
from openEphysExport.generate_json import generate_json
from decoder import decode
from importData.compareSpikeFiltering import WaveFormComparator
from resultAnalysis import paper_figures

def main():
	typeDec = 'ann'
	window = 0.036
	windowMS = int(window*1000)
	# Define project path
	# xmlPath = "/home/mobsrick/Dima/neuroencoders_1021/test/amplifier.xml"
	xmlPath = "/media/mobsrick/DimaERC2/neuroencoders_1021/test/amplifier.xml" # For creating datasets
	# xmlPath = "/media/mobsrick/DimaERC2/neuroencoders_1021/M1199_reversal/M1199_20210416_Reversal.xml" # For creating datasets
	# xmlPath = "/media/mobsrick/DimaERC2/neuroencoders_1021/todecode_reversal/amplifier.xml" # For creating datasets
	# xmlPath = "/media/mobsrick/DimaERC2/neuroencoders_1021/test_small_file/amplifier.xml" # For creating datasets
	# json = '/media/mobsrick/DimaERC2/neuroencoders_1021/test/amplifier.json'
	ProjectPath = Project(os.path.expanduser(xmlPath), jsonPath=None)
	
	# Import data to be decoded if it's not present
	subprocess.run(["./getTsdFeature.sh", os.path.expanduser(xmlPath.strip('\'')), "'pos'"])
	rawdata_parser.speed_filter(ProjectPath.folder)
	rawdata_parser.select_epochs(ProjectPath.folder)
	Helper = rawdata_parser.DataHelper(ProjectPath, typeDec)
	Parameters = Params(Helper, window)

	# Create linearization function
	Linearizer = UMazeLinearizer(ProjectPath.folder)
	Linearizer.verify_linearization(Helper.position/Helper.maxPos(), ProjectPath.folder, overwrite=False)
	L_function = Linearizer.pykeops_linearization

	# Data
	# for ws in [0.036, 3*0.036, 7*0.036, 14*0.036]:
	for ws in [0.036]:
		julia_spike_filter(ProjectPath, folder_code, windowSize=window, singleSpike=False)
	devicename = management.manage_devices('GPU')
 
	# -------------------------------------------------------------------------
	# If decode
	# Decoder = decode.Decoder(ProjectPath, Parameters, device_name=devicename)
	# outputs = Decoder.test(Helper.fullBehavior, linearizationFunction=L_function, windowsizeMS=windowMS)
	# print_results.printResults(Decoder.folderResult, windowsize=windowMS)
	# -------------------------------------------------------------------------
	
	windowSizesMS =  [36 ] #, 3*36, 7*36, 14*36] #36,
	for windowSizeMS in windowSizesMS:
		NNTrainer = an_network.LSTMandSpikeNetwork(ProjectPath, Parameters,
												   deviceName=devicename)
		NNTrainer.fix_linearizer(Linearizer.mazePoints, Linearizer.tsProj)
		NNTrainer.train(Helper.fullBehavior, windowsizeMS=windowSizeMS)
		NNTrainer.test(Helper.fullBehavior, l_function=L_function,
														 windowsizeMS=windowSizeMS)
		print_results.print_results(NNTrainer.folderResult, windowSizeMS=windowSizeMS)
		# outputs_sleep = NNTrainer.testSleep(Helper.fullBehavior, windowsizeMS=windowSizeMS)
	# -------------------------------------------------------------------------
 
	print("training Bayesian decoder from spike sorting ")
	TrainerBayes = decode_bayes.Trainer(ProjectPath)
	bayes_matrices = TrainerBayes.train_orderByPos(Helper.fullBehavior, L_function)
	outputs = TrainerBayes.test_parallel(Helper.fullBehavior, bayes_matrices, windowSize=windowSizesMS)
	print_results.printResults(TrainerBayes.folderResult, typeDec='bayes', results = outputs, windowsize=windowMS)
 
	# -------------------------------------------------------------------------
	# we can compare the waveforms resulting in the offline (before spike sorting) filtering
	# and the filtering for the online strategy:

	for ws in [36]: #, 3*36, 7*36, 14*36] #36,
		WFCTrain = WaveFormComparator(ProjectPath, Parameters, Helper.fullBehavior, useTrain=True,windowsizeMS=ws)
		WFCTrain.save_alignment_tools(TrainerBayes, L_function, windowsizeMS=ws)
		WFCTest = WaveFormComparator(ProjectPath, Parameters, Helper.fullBehavior, useTrain=False,windowsizeMS=ws)
		WFCTest.save_alignment_tools(TrainerBayes, L_function, windowsizeMS=ws)
		WFCPreSleep = WaveFormComparator(ProjectPath, Parameters, Helper.fullBehavior, useTrain=False,windowsizeMS=ws, sleepName='PreSleep')
		WFCPreSleep.save_alignment_tools(TrainerBayes, L_function, windowsizeMS=ws)
		WFCPostSleep = WaveFormComparator(ProjectPath, Parameters, Helper.fullBehavior, useTrain=False,windowsizeMS=ws, sleepName='PostSleep')
		WFCPostSleep.save_alignment_tools(TrainerBayes, L_function, windowsizeMS=ws)

	# -------------------------------------------------------------------------
	figures = paper_figures.PaperFigures(ProjectPath, Helper.fullBehavior, TrainerBayes, L_function, bayesMatrices=bayes_matrices)
	figures.load_data()
	figures.test_bayes()
	figures.fig_example_linear()
	figures.hist_linerrors(speed='fast')
	figures.nnVSbayes()
	figures.predLoss_vs_trueLoss()
	figures.fig_example_2d()
	figures.predLoss_linError()
	figures.predLoss_linError()
	figures.fig_example_linear_filtered()
	# paperFigure_sleep.paperFigure_sleep(projectPath, params, linearizationFunction, behavior_data, "PreSleep",saveFolder="resultSleep")



if __name__=="__main__":
	# In this architecture we use a 2.4 tensorflow backend, predicting solely the position.
	main()
