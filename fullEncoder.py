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






class xmlPath():
	def __init__(self, path):
		self.xml = path
		findFolder = lambda path: path if path[-1]=='/' or len(path)==0 else findFolder(path[:-1]) 
		self.folder = findFolder(self.xml)
		self.dat = path[:-3] + 'dat'
		self.fil = path[:-3] + 'fil'
		self.json = path[:-3] + 'json'
		self.tfrec = self.folder + 'trainingDataset.tfrec'
		self.testTfrec = self.folder + 'testingDataset.tfrec'

### Params
class Params:
	def __init__(self, detector, dim_output):
		self.nGroups = detector.nGroups()
		self.dim_output = dim_output
		self.nChannels = detector.numChannelsPerGroup()

		self.nSteps = 10000
		self.nFeatures = 128
		self.lstmLayers = 3
		self.lstmSize = 128
		self.lstmDropout = 0.3

		self.windowLength = 0.036 # in seconds, as all things should be
		self.batch_size = 52
		self.timeMajor = True

		self.learningRates = [0.0003, 0.00003, 0.00001]
		self.lossLearningRate = 0.00003
		self.lossActivation = None


from importData import rawDataParser
from fullEncoder import datasetMaker, nnUtils, nnTraining


trainLosses = []
projectPath = xmlPath(os.path.expanduser(sys.argv[2]))



### Data
filterType = sys.argv[3]
if filterType=='external':
    useOpenEphysFilter=True
else:
    useOpenEphysFilter=False
print('using external filter:', useOpenEphysFilter)
spikeDetector = rawDataParser.SpikeDetector(projectPath, useOpenEphysFilter)
params = Params(spikeDetector, 2)
if (not os.path.isfile(projectPath.tfrec)) or (not os.path.isfile(projectPath.testTfrec)):
	if not os.path.isfile(projectPath.folder+'_rawSpikesForRnn.npz'):

		allGroups = []
		allSpTime = []
		allSpikes = []
		allSpkPos = []
		allSpkSpd = []

		for spikes in spikeDetector.getSpikes():
			if len(spikes['time'])==0:
				continue
			for grp,time,spk,pos,spd in sorted(zip(spikes['group'],spikes['time'],spikes['spike'],spikes['position'],spikes['speed']), key=lambda x:x[1]):
				allGroups.append(grp)
				allSpTime.append(time)
				allSpikes.append(spk)
				allSpkPos.append(pos)
				allSpkSpd.append(spd)
			
		GRP_data = np.array(allGroups)
		SPT_data = np.array(allSpTime)
		SPK_data = np.array(allSpikes)
		POS_data = np.array(allSpkPos)
		SPD_data = np.array(allSpkSpd)
		print('data parsed.')


		SPT_train, SPT_test, GRP_train, GRP_test, SPK_train, SPK_test, POS_train, POS_test, SPD_train, SPD_test = train_test_split(
			SPT_data, GRP_data, SPK_data, POS_data, SPD_data, test_size=0.1, shuffle=False, random_state=42)
		np.savez(projectPath.folder + '_rawSpikesForRnn', 
			SPT_train, SPT_test, GRP_train, GRP_test, SPK_train, SPK_test, POS_train, POS_test, SPD_train, SPD_test)
	else:
		try:
			print(loaded)
		except NameError:
			print('loading data')
			Results = np.load(projectPath.folder + '_rawSpikesForRnn.npz', allow_pickle=True)
			SPT_train = Results['arr_0']
			SPT_test = Results['arr_1']
			GRP_train = Results['arr_2']
			GRP_test = Results['arr_3']
			SPK_train = Results['arr_4']
			SPK_test = Results['arr_5']
			POS_train = Results['arr_6']
			POS_test = Results['arr_7']
			SPD_train = Results['arr_8']
			SPD_test = Results['arr_9']
			loaded='data loaded'
			print(loaded)




	if not os.path.isfile(projectPath.tfrec):
		gen = nnUtils.getTrainingSpikes(params, SPT_train, POS_train, GRP_train, SPK_train, maxPos = spikeDetector.maxPos())
		print('building training dataset')
		with tf.python_io.TFRecordWriter(projectPath.tfrec) as writer:
			totalLength = SPT_train[-1] - SPT_train[0]
			nBins = int(totalLength // params.windowLength) - 1
			for _ in tqdm(range(nBins)):
				example = next(gen)
				writer.write(nnUtils.serialize(params, *tuple(example)))

	if not os.path.isfile(projectPath.testTfrec):
		gen = nnUtils.getTrainingSpikes(params, SPT_test, POS_test, GRP_test, SPK_test, maxPos = spikeDetector.maxPos())
		print('building testing dataset')
		with tf.python_io.TFRecordWriter(projectPath.testTfrec) as writer:
			totalLength = SPT_test[-1] - SPT_test[0]
			nBins = int(totalLength // params.windowLength) - 1

			for _ in tqdm(range(nBins)):
				example = next(gen)
				writer.write(nnUtils.serialize(params, *tuple(example)))



# Training, testing, and preparing network for online setup
trainer = nnTraining.Trainer(projectPath, params, device_name=device_name)
trainLosses = trainer.train()
trainer.convert()
outputs = trainer.test()


# Saving files
fileName = projectPath.folder + '_resultsForRnn_temp'
np.savez(os.path.expanduser(fileName), trainLosses=trainLosses, **outputs)

import scipy.io
scipy.io.savemat(os.path.expanduser(projectPath.folder + 'inferring.mat'), np.load(os.path.expanduser(fileName+'.npz')))

import json
outjsonStr = {};
outjsonStr['encodingPrefix'] = projectPath.folder + '_graphDecoder'
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
