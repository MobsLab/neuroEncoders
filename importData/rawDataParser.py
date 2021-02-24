import sys
import os

import tables
import struct
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing as ml

from tqdm import tqdm


BUFFERSIZE=10000




def get_params(pathToXml):
	list_channels = []
	try:
		tree = ET.parse(pathToXml)
	except:
		print("impossible to open xml file:", pathToXml)
		sys.exit(1)
	root = tree.getroot()
	for br1Elem in root:
		if br1Elem.tag != 'spikeDetection':
			continue
		for br2Elem in br1Elem:
			if br2Elem.tag != 'channelGroups':
				continue
			for br3Elem in br2Elem:
				if br3Elem.tag != 'group':
					continue
				group=[];
				for br4Elem in br3Elem:
					if br4Elem.tag != 'channels':
						continue
					for br5Elem in br4Elem:
						if br5Elem.tag != 'channel':
							continue
						group.append(int(br5Elem.text))
				list_channels.append(group)
	for br1Elem in root:
		if br1Elem.tag != 'acquisitionSystem':
			continue
		for br2Elem in br1Elem:
			if br2Elem.tag == 'samplingRate':
				samplingRate  = float(br2Elem.text)
			if br2Elem.tag == 'nChannels':
				nChannels = int(br2Elem.text)

	return list_channels, samplingRate, nChannels


def get_position(folder):

	if not os.path.exists(folder + 'nnBehavior.mat'):
		raise ValueError('this file does not exist :'+folder+'nnBehavior.mat')
	with tables.open_file(folder + 'nnBehavior.mat') as f:
		positions = f.root.behavior.positions
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:,:],1,0)
		position_time = np.swapaxes(position_time[:,:],1,0)

		# Problem as this was not in the file:
		#trainEpochs = np.array(f.root.behavior.trainEpochs).flatten()
		#testEpochs = np.array(f.root.behavior.testEpochs).flatten()
		trainEpochs = np.array([1143,4100])
		testEpochs = np.array([4100,8000])


	return positions, position_time, list(trainEpochs), list(testEpochs)









class openEphysFilter:
	'''reads the open-ephys filtered data'''
	def __init__(self, path, list_channels, nChannels):
		if not os.path.exists(path):
			raise ValueError('this file does not exist: '+ path)
		self.path = path
		self.nChannels = nChannels
		self.channelList = [item for sublist in list_channels for item in sublist]
		self.dataFile = open(self.path, 'rb')
		self.dataFile.__enter__()
		self.dataReader = struct.iter_unpack(str(self.nChannels)+'h', self.dataFile.read())

	def filter(self, sample):
		if sample.ndim==1:
			return np.array(next(self.dataReader))[self.channelList]*0.195
		else:
			temp = []
			for i in range(sample.shape[0]):
				temp.append(np.array(next(self.dataReader))[self.channelList]*0.195)
			return np.array(temp)

	def __del__(self):
		try:
			self.dataFile.__exit__(*sys.exc_info())
		except AttributeError:
			pass


class INTANFilter:
	'''A copy of the INTAN filter. No excuse to not recognize waveforms online'''
	def __init__(self, cutOffFreq, sampleFreq, channelList):
		self.channelList = [item for sublist in channelList for item in sublist]
		self.nChannels   = len(channelList)
		self.state	   = np.zeros(len(self.channelList))
		self.cutOffFreq  = cutOffFreq
		self.sampleFreq  = sampleFreq

		self.a = np.exp(-2*np.pi*self.cutOffFreq / self.sampleFreq)
		self.b = 1 - self.a

	def filter(self, sample):
		temp = np.array(sample)[self.channelList] - self.state
		self.state = self.a * self.state + self.b * np.array(sample)[self.channelList]
		return temp












def isTrainingExample(epochs, time):
	for e in range(len(epochs['train'])//2):
		if time >= epochs['train'][2*e] and time < epochs['train'][2*e+1]:
			return True
	for e in range(len(epochs['test'])//2):
		if time >= epochs['test'][2*e] and time < epochs['test'][2*e+1]:
			return False
	return None

def emptyData():
	return {'group':[], 'time':[], 'spike':[], 'position':[], 'train':[]}

class SpikeDetector:
	'''A processor class to go through raw data to filter and extract spikes. Synchronizes with position.'''
	def __init__(self, path, useOpenEphysFilter, mode, jsonPath=None):

		self.path = path
		self.mode = mode

		if not os.path.exists(self.path.xml):
			raise ValueError('this file does not exist: '+ self.path.xml)
		# if not os.path.exists(self.path.dat):
		#	 raise ValueError('this file does not exist: '+ self.path.dat)

		if self.mode == "decode":
			self.list_channels, self.samplingRate, self.nChannels = get_params(self.path.xml)
			if os.path.isfile(self.path.folder + "timestamps.npy"):
				self.nChannels = int( os.path.getsize(self.path.dat) \
					/ 2 \
					/ np.load(self.path.folder + "timestamps.npy").shape[0] )
			self.position = np.array([0,0], dtype=float).reshape([1,2])
			self.position_time = np.array([0], dtype=float)
			self.startTime = 0
			self.stopTime = float("inf")
			self.epochs = {"train": [], "test": [0, float("inf")]}
		else:
			self.list_channels, self.samplingRate, self.nChannels = get_params(self.path.xml)
			self.position, self.position_time, *epochs = get_position(self.path.folder)
			self.epochs = {
				"train": epochs[0],
				"test":  epochs[1]
			}
			self.startTime = min(self.epochs["train"] + self.epochs["test"])
			self.stopTime  = max(self.epochs["train"] + self.epochs["test"])


		if useOpenEphysFilter:
			self.filter = openEphysFilter(self.path.fil, self.list_channels, self.nChannels)
		else:
			self.filter = INTANFilter(350., self.samplingRate, self.list_channels)

		self.thresholdFactor   = 3 # in units of standard deviation
		self.filteredSignal	= np.zeros([BUFFERSIZE, len(self.filter.channelList)])
		self.thresholds		= np.zeros([len(self.filter.channelList)])

		self.endOfLastBuffer = []
		self.lateSpikes = emptyData()
		self.lastBuffer = False
		self.firstBuffer = True

	def nGroups(self):
		return len(self.list_channels)
	def numChannelsPerGroup(self):
		return [len(self.list_channels[n]) for n in range(self.nGroups())]
	def maxPos(self):
		return np.max(self.position) if self.mode != 'decode' else 1
	def learningTime(self):
		return sum([self.epochs["train"][2*n+1]-self.epochs["train"][2*n] for n in range(len(self.epochs["train"])//2)])
	def trainingPositions(self):
		selected = [False for _ in range(len(self.position_time))]
		for idx in range(len(self.position_time)):
			for e in range(len(self.epochs['train'])//2):
				if self.position_time[idx] >= self.epochs['train'][2*e] and self.position_time[idx] < self.epochs['train'][2*e+1]:
					selected[idx] = True
		return self.position[np.where(selected)]/self.maxPos()


	def dim_output(self):
		return self.position.shape[1]



	def __del__(self):
		try:
			self.dataFile.__exit__(*sys.exc_info())
		except AttributeError:
			pass

	def getThresholds(self):
		idx = 0
		nestedThresholds = []
		for group in range(len(self.list_channels)):
			temp = []
			for channel in range(len(self.list_channels[group])):
				temp.append(self.thresholds[idx])
				idx += 1
			nestedThresholds.append(temp)
		return nestedThresholds
	def setThresholds(self, thresholds):
		assert([len(d) for d in thresholds] == [len(s) for s in self.list_channels])
		self.thresholds = [i for d in thresholds for i in d]



	def getFilteredBuffer(self, bufferSize):
		temp = []
		for item in range(bufferSize):
			try:
				temp.append(self.filter.filter(np.array(next(self.dataReader))*0.195))
			except StopIteration:
				self.lastBuffer = True
				break
		temp = np.array(temp)
		return temp



	def getLateSpikes(self):
		temp = self.lateSpikes.copy()
		self.lateSpikes = emptyData()

		for spk in range(len(temp['group'])):
			channelsBefore = self.previousChannels[temp['group'][spk]]
			temp['spike'][spk] = np.concatenate([
				temp['spike'][spk], 
				self.filteredSignal
					[:32-temp['spike'][spk].shape[0], 
					channelsBefore:channelsBefore+len(self.list_channels[temp['group'][spk]])]], axis=0).transpose()

		return temp


	def __iter__(self):
		self.dataFile = open(self.path.dat, 'rb')
		self.dataFile.__enter__()
		self.dataReader = struct.iter_unpack(str(self.nChannels)+'h', self.dataFile.read())
		print('extracting spikes.')

		self.inputQueue = ml.Queue()
		self.outputQueue = ml.Queue()
		self.proc = [ml.Process(target=findSpikesInGroupParallel, args=(self.inputQueue, self.outputQueue, self.samplingRate, self.epochs, self.thresholdFactor, self.startTime, self.position, self.position_time)) for _ in range(len(self.list_channels))]
		for p in self.proc:
			p.deamon = True
			p.start()

		datFileLengthInByte = os.stat(self.path.dat).st_size
		numBuffer = datFileLengthInByte // (2 * self.nChannels * BUFFERSIZE) + 1
		self.pbar = tqdm(total=numBuffer)

		n=0
		self.previousChannels = []
		for group in self.list_channels:
			self.previousChannels.append(n)
			n += len(group)

		return self

	def __next__(self):

		if self.lastBuffer:
			for p in self.proc:
				p.terminate()
				p.join()
			self.inputQueue.close()
			self.outputQueue.close()
			self.pbar.close()
			raise StopIteration

		### filter signal, compute new thresholds and get late spikes from previous buffer
		self.filteredSignal = self.getFilteredBuffer(BUFFERSIZE)
		spikesFound = self.getLateSpikes()

		### copy end of previous buffer
		if self.endOfLastBuffer != []:
			self.firstBuffer = False
			self.filteredSignal = np.concatenate([self.endOfLastBuffer, self.filteredSignal], axis=0)
		self.endOfLastBuffer = self.filteredSignal[-15:,:]

		for group in range(len(self.list_channels)):
			filteredBuffer = self.filteredSignal[:,self.previousChannels[group]:self.previousChannels[group]+len(self.list_channels[group])]
			thresholds	 = self.thresholds	  [self.previousChannels[group]:self.previousChannels[group]+len(self.list_channels[group])]
			self.inputQueue.put([self.pbar.n, group, thresholds, filteredBuffer])


		for _ in range(len(self.list_channels)):
			res = self.outputQueue.get(block=True)
			for key in emptyData().keys():
				spikesFound[key] += res[0][key]
				self.lateSpikes[key] += res[1][key]


		self.pbar.update(1)
		if spikesFound['time'] != [] and max(spikesFound["time"]) > self.stopTime:
			self.lastBuffer = True
		return spikesFound





def findSpikesInGroupParallel(inputQueue, outputQueue, samplingRate, epochs, thresholdFactor, startTime, position, position_time):
	
	while True:
		N, group, thresholds, filteredBuffer = inputQueue.get(block=True)

		if np.all(thresholds == np.zeros([filteredBuffer.shape[1]])):
			thresholds = thresholdFactor * filteredBuffer.std(axis=0)

		triggered = False

		spikesFound = emptyData()
		lateSpikes = emptyData()

		spl = 15
		while spl < filteredBuffer.shape[0]:
			for chnl in range(filteredBuffer.shape[1]):

				# check if cross negative trigger
				if filteredBuffer[spl,chnl] < -thresholds[chnl] and filteredBuffer[spl-1, chnl] > -thresholds[chnl]:
					triggered = True
					positiveTrigger = False

				if triggered:

					# Get peak or trough
					if positiveTrigger:
						spl = spl + np.argmax(filteredBuffer[spl:spl+15, chnl])
					else:
						spl = spl + np.argmin(filteredBuffer[spl:spl+15, chnl])

					if N==0:
						time = spl / samplingRate
					else:
						time = (N * BUFFERSIZE + spl - 15) / samplingRate
					train = isTrainingExample(epochs, time)

					# Do nothing unless after behaviour data has started
					if time > startTime:

						# That's a late spike, we'll have to wait till next buffer
						if spl > BUFFERSIZE - 18 + 15:
							lateSpikes['group'].   append(group)
							lateSpikes['time'].    append(time)
							lateSpikes['spike'].   append(filteredBuffer[spl-15:, :])
							lateSpikes['position'].append( position[np.argmin(np.abs(position_time-time))] )
							lateSpikes['train'].   append(train)

						else:
							spike = filteredBuffer[spl-15:spl+17, :].copy()
							if np.shape(spike)[0]==32:
								spikesFound['group'].   append(group)
								spikesFound['time'].	append(time)
								spikesFound['spike'].   append( np.array(spike).reshape([32,filteredBuffer.shape[1]]).transpose() )
								spikesFound['position'].append( position[np.argmin(np.abs(position_time-time))] )
								spikesFound['train'].   append(train)


					spl += 15
					triggered = False
					break
			spl += 1

		outputQueue.put([spikesFound, lateSpikes])

