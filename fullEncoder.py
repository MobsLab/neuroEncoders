import sys
import os.path
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange
import matplotlib as mpl
import matplotlib.pyplot as plt
print(flush=True)
if len(sys.argv)>1 and sys.argv[1]=="gpu":
	device_name = "/gpu:0"
	print('MOBS FULL FLOW ENCODER: DEVICE GPU', flush=True)
elif len(sys.argv)==1 or sys.argv[1]=="cpu":
	device_name = "/cpu:0"
	print('MOBS FULL FLOW ENCODER: DEVICE CPU', flush=True)
else:
	raise ValueError('didn\'t understand arguments calling scripts '+sys.argv[0])






class Project():
	def __init__(self, xmlPath):
		self.xml = xmlPath
		findFolder = lambda path: path if path[-1]=='/' or len(path)==0 else findFolder(path[:-1]) 
		self.folder = findFolder(self.xml)
		self.dat = xmlPath[:-3] + 'dat'
		self.fil = xmlPath[:-3] + 'fil'
		self.json = xmlPath[:-3] + 'json'

		self.tfrec = {
			"train": self.folder + 'dataset/trainingDataset.tfrec', 
			"test": self.folder + 'dataset/testingDataset.tfrec'}

		self.graph = self.folder + 'graph/decoder'
		self.graphMeta = self.folder + 'graph/decoder.meta'

		self.resultsNpz = self.folder + 'results/inferring.npz'
		self.resultsMat = self.folder + 'results/inferring.mat'

		if not os.path.isdir(self.folder + 'dataset'):
			os.makedirs(self.folder + 'dataset')
		if not os.path.isdir(self.folder + 'graph'):
			os.makedirs(self.folder + 'graph')
		if not os.path.isdir(self.folder + 'results'):
			os.makedirs(self.folder + 'results')

### Params
class Params:
	def __init__(self, detector):
		self.nGroups = detector.nGroups()
		self.dim_output = detector.dim_output()
		self.nChannels = detector.numChannelsPerGroup()
		self.length = 0

		self.nSteps = int(10000 * 0.036 / float(sys.argv[4]))
		self.nFeatures = 128
		self.lstmLayers = 3
		self.lstmSize = 128
		self.lstmDropout = 0.3

		self.windowLength = float(sys.argv[4]) # in seconds, as all things should be
		self.batch_size = 52
		self.timeMajor = True

		self.learningRates = [0.0003, 0.00003, 0.00001]
		self.lossLearningRate = 0.00003
		self.lossActivation = None


from importData import rawDataParser
from fullEncoder import datasetMaker, nnUtils, nnTraining


trainLosses = []
projectPath = Project(os.path.expanduser(sys.argv[2]))



### Data
filterType = sys.argv[3]
if filterType=='external':
    useOpenEphysFilter=True
else:
    useOpenEphysFilter=False
print('using external filter:', useOpenEphysFilter)

spikeDetector = rawDataParser.SpikeDetector(projectPath, useOpenEphysFilter)
params = Params(spikeDetector)

if (not os.path.isfile(projectPath.tfrec["train"])) or \
	(not os.path.isfile(projectPath.tfrec["test"])):

	spikeGen = nnUtils.spikeGenerator(projectPath, spikeDetector, maxPos=spikeDetector.maxPos())
	spikeSequenceGen = nnUtils.getSpikeSequences(params, spikeGen())
	with tf.python_io.TFRecordWriter(projectPath.tfrec["train"]) as writerTrain, tf.python_io.TFRecordWriter(projectPath.tfrec["test"]) as writerTest:
		for example in spikeSequenceGen:
			train = example[0]
			if train == None:
				continue
			else:
				if train:
					w = writerTrain
				else:
					w = writerTest
				w.write(nnUtils.serializeSpikeSequence(params, *tuple(example[1:])))



# Training, testing, and preparing network for online setup
trainer = nnTraining.Trainer(projectPath, params, device_name=device_name)
trainLosses = trainer.train()
trainer.convert()
outputs = trainer.test()


# Saving files
np.savez(projectPath.resultsNpz, trainLosses=trainLosses, **outputs)

import scipy.io
scipy.io.savemat(projectPath.resultsMat, np.load(projectPath.resultsNpz))

import json
outjsonStr = {};
outjsonStr['encodingPrefix'] = projectPath.graph
outjsonStr['mousePort'] = 0

outjsonStr['nGroups'] = int(params.nGroups)
idx=0
for group in range(len(spikeDetector.list_channels)):
    if os.path.isfile(projectPath.xml[:len(projectPath.xml)-3] + 'clu.' + str(group+1)):
        outjsonStr['group'+str(group-idx)]={}
        outjsonStr['group'+str(group-idx)]['nChannels'] = len(spikeDetector.list_channels[group])
        for chnl in range(len(spikeDetector.list_channels[group])):
            outjsonStr['group'+str(group-idx)]['channel'+str(chnl)]=int(spikeDetector.list_channels[group][chnl])
            outjsonStr['group'+str(group-idx)]['threshold'+str(chnl)]=int(spikeDetector.getThresholds()[group][chnl])
    else:
        idx+=1

outjsonStr['nStimConditions'] = 1
outjsonStr['stimCondition0'] = {}
outjsonStr['stimCondition0']['stimPin'] = 14
outjsonStr['stimCondition0']['lowerX'] = 0.0
outjsonStr['stimCondition0']['higherX'] = 0.0
outjsonStr['stimCondition0']['lowerY'] = 0.0
outjsonStr['stimCondition0']['higherY'] = 0.0
outjsonStr['stimCondition0']['lowerDev'] = 0.0
outjsonStr['stimCondition0']['higherDev'] = 0.0

# print(outjsonStr)

outjson = json.dumps(outjsonStr, indent=4)
with open(projectPath.json,"w") as json_file:
    json_file.write(outjson)
