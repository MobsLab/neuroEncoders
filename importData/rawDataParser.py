import sys
import os

import tables
import struct
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing as ml

import SimpleBayes.butils as butils
import matplotlib.pyplot as plt

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


def inEpochs(t,epochs):
	# for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
	# test if t is among at least one of these epochs
	# Epochs are treated as closed interval [,]
	mask =  np.sum([(t>=epochs[2*i]) * (t<=epochs[2*i+1]) for i in range(len(epochs)//2)],axis=0)
	return np.where(mask >= 1)
def inEpochsMask(t,epochs):
	# for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
	# test if t is among at least one of these epochs
	# Epochs are treated as closed interval [,]
	mask =  np.sum([(t>=epochs[2*i]) * (t<=epochs[2*i+1]) for i in range(len(epochs)//2)],axis=0)
	return mask >= 1


def get_position(folder):

	if not os.path.exists(folder + 'nnBehavior.mat'):
		raise ValueError('this file does not exist :'+folder+'nnBehavior.mat')
	with tables.open_file(folder + 'nnBehavior.mat') as f:
		positions = f.root.behavior.positions
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:,:],1,0)
		position_time = np.swapaxes(position_time[:,:],1,0)

		# trainEpochs = np.array(f.root.behavior.trainEpochs).flatten()
		# testEpochs = np.array(f.root.behavior.testEpochs).flatten()

	return positions, position_time  #, list(trainEpochs), list(testEpochs)


def speed_filter(folder,overWrite=True):
	## A simple tool to set up a threshold on the speed value
	# The speed threshold is then implemented through a speed_mask:
	# a boolean array indicating for each index (i.e measured feature time step)
	# if it is above threshold or not.

	with tables.open_file(folder + 'nnBehavior.mat',"a") as f:

		children = [c.name for c in f.list_nodes("/behavior")]
		if "speedMask" in children:
			print("speedMask already created")
			if overWrite:
				f.remove_node("/behavior","speedMask")
			else:
				return

		positions = f.root.behavior.positions
		speed = f.root.behavior.speed
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:, :], 1, 0)
		speed = np.swapaxes(speed[:, :], 1, 0)
		posTime = np.swapaxes(position_time[:, :], 1, 0)
		if speed.shape[0]==posTime.shape[0]-1 :
			speed = np.append(speed,speed[-1])
		speed = np.reshape(speed,[speed.shape[0],1])
		tmin = 0
		tmax = posTime[-1]
		myposTime = posTime[((posTime >= tmin) * (posTime <= tmax))[:, 0]]
		mybehave = positions[((posTime >= tmin) * (posTime <= tmax))[:, 0]]
		myspeed = speed[((posTime >= tmin) * (posTime <= tmax))[:, 0]]

		window_len = 40
		s = np.r_[myspeed[window_len - 1:0:-1], myspeed, myspeed[-2:-window_len - 1:-1]]
		w = eval('np.' + "hamming" + '(window_len)')
		myspeed2 = np.convolve(w / w.sum(), s[:, 0], mode='valid')[(window_len // 2 - 1):-(window_len // 2)]

		speedThreshold = np.mean(np.log(myspeed2+10**(-8)))
		speedFilter = myspeed2 > np.exp(speedThreshold)


		fig, ax = plt.subplots(4, 1)
		fig.suptitle("Speed threshold selection")
		l1, = ax[0].plot(myposTime[speedFilter], mybehave[speedFilter, 0],c="red")
		ax[0].set_ylabel("environmental \n variable")
		ax[1].set_ylabel("speed")
		l2, = ax[0].plot(myposTime[speedFilter], mybehave[speedFilter, 1],c="orange")
		l4, = ax[1].plot(myposTime[speedFilter], myspeed[speedFilter],c="blue")
		l3, = ax[1].plot(myposTime[speedFilter], myspeed2[speedFilter],c="purple")
		l5 = ax[0].scatter(myposTime[speedFilter], np.zeros(myposTime[speedFilter].shape[0]) - 4, c="black", s=0.2)
		l6 = ax[1].scatter(myposTime[speedFilter], np.zeros(myposTime[speedFilter].shape[0]) - 4, c="black", s=0.2)
		ax[2].hist(np.log(myspeed2+10**(-8)), bins=200)
		ax[2].set_ylabel("speed histogram")
		l8 = ax[2].axvline(speedThreshold, color="black")
		slider = plt.Slider(ax[3], 'speed Threshold',np.min(np.log(myspeed2+10**(-8))),np.max(np.log(myspeed2+10**(-8))),valinit=speedThreshold,valstep=0.01)

		def update(val):
			speedThreshold = val
			speedFilter = myspeed2 > np.exp(speedThreshold)
			l1.set_ydata(mybehave[speedFilter, 0])
			l2.set_ydata(mybehave[speedFilter, 1])
			l1.set_xdata(myposTime[speedFilter])
			l2.set_xdata(myposTime[speedFilter])
			l5.set_offsets(np.transpose(np.stack([myposTime[speedFilter][:,0],np.zeros(myposTime[speedFilter].shape[0]) - 4])))
			l6.set_offsets(np.transpose(np.stack([myposTime[speedFilter][:,0],np.zeros(myposTime[speedFilter].shape[0]) - 4])))
			l3.set_ydata(myspeed2[speedFilter])
			l4.set_ydata(myspeed[speedFilter])
			l3.set_xdata(myposTime[speedFilter])
			l4.set_xdata(myposTime[speedFilter])
			l8.set_xdata(val)
			fig.canvas.draw_idle()


		slider.on_changed(update)
		plt.show()

		speedFilter = myspeed2 > np.exp(slider.val)
		fig, ax = plt.subplots(3, 1)
		l1, = ax[0].plot(myposTime[speedFilter], mybehave[speedFilter, 0], c="red")
		l2, = ax[0].plot(myposTime[speedFilter], mybehave[speedFilter, 1], c="orange")
		l4, = ax[1].plot(myposTime[speedFilter], myspeed[speedFilter], c="blue")
		l3, = ax[1].plot(myposTime[speedFilter], myspeed2[speedFilter], c="purple")
		l5 = ax[0].scatter(myposTime[speedFilter], np.zeros(myposTime[speedFilter].shape[0]) - 4, c="black", s=0.2)
		l6 = ax[1].scatter(myposTime[speedFilter], np.zeros(myposTime[speedFilter].shape[0]) - 4, c="black", s=0.2)
		ax[2].hist(np.log(myspeed2+10**(-8)), bins=200)
		ax[2].axvline(slider.val, color="black")
		plt.show()

		f.create_array("/behavior","speedMask",speedFilter)
		f.flush()
		f.close()



def modify_feature_forBestTestSet(folder,plimits=[]):
	# Find test set with most uniform covering of speed and environment variable.
	# provides then a little manual tool to change the size of the window
	# and its position.
	# plimits can be used to constraint the range of search for the test sets
	# to a certain time intervals. We added it because in some dataset
	# the nnbehavior contained more measured timesteps than in the .dat files.

	if not os.path.exists(folder + 'nnBehavior.mat'):
		raise ValueError('this file does not exist :'+folder+'nnBehavior.mat')
	with tables.open_file(folder + 'nnBehavior.mat',"a") as f:

		speedMask = f.root.behavior.speedMask[:]

		positions = f.root.behavior.positions
		positions = np.swapaxes(positions[:, :], 1, 0)
		speeds = f.root.behavior.speed
		position_time = f.root.behavior.position_time
		position_time = np.swapaxes(position_time[:,:],1,0)
		speeds = np.swapaxes(speeds[:, :], 1, 0)
		if speeds.shape[0]==position_time.shape[0]-1 :
			speeds = np.append(speeds,speeds[-1]).reshape(position_time.shape[0],speeds.shape[1])


		#if pmin and pmax are not None, we do not use some positions from the behaviour file
		if plimits==[] :
			pmin = 0
			pmax = position_time[-1][0]
		else:
			pmin = position_time[plimits[0]][0]
			pmax = position_time[plimits[1]][0]
			positions = positions[plimits[0]:plimits[1]+1,:]
			position_time = position_time[plimits[0]:plimits[1]+1,:]
			speeds = speeds[plimits[0]:plimits[1]+1,:]
			speedMask = speedMask[plimits[0]:plimits[1]+1]

		positions = positions[speedMask, :]
		speeds = speeds[speedMask,:]
		position_time = position_time[speedMask,:]

		sizeTest = position_time.shape[0]//10

		print("Evaluating the entropy of each possible test set")
		entropiesPositions = []
		entropiesSpeeds = []
		for id in tqdm(np.arange(0,stop=positions.shape[0]-sizeTest,step=sizeTest)):
			# The environmental variable are discretized by equally space bins
			# such that there is 45*...*45 bins per dimension
			# we then fit over the test set a kernel estimation of the probability distribution
			# and evaluate it over the bins
			_ , probaFeatures = butils.kdenD(positions[id:id+sizeTest,:],bandwidth=1.0)
			# We then compute the entropy of the obtained distribution:
			epsilon = 10**(-9)
			entropiesPositions += [-np.sum(probaFeatures*np.log(probaFeatures+epsilon))]

			_ , probaFeatures = butils.kdenD(speeds[id:id+sizeTest,:],bandwidth=1.0)
			# We then compute the entropy of the obtained distribution:
			epsilon = 10**(-9)
			entropiesSpeeds += [-np.sum(probaFeatures*np.log(probaFeatures+epsilon))]
		totEntropy = np.array(entropiesSpeeds) + np.array(entropiesPositions)
		bestTestSet = np.argmax(totEntropy)

		testSetId = bestTestSet*sizeTest
		# Next we provide a small tool to manually change the bestTest set position
		# as well as its size:
		fig,ax = plt.subplots(positions.shape[1]+2,1)
		trainEpoch = np.array([pmin,position_time[bestTestSet*sizeTest][0],position_time[bestTestSet*sizeTest+sizeTest][0],pmax])
		testEpochs = np.array(
			[position_time[bestTestSet * sizeTest][0], position_time[bestTestSet * sizeTest + sizeTest][0]])
		ls = []
		for id in range(positions.shape[1]):
			l1 = ax[id].scatter(position_time[inEpochs(position_time,trainEpoch)[0],0],positions[inEpochs(position_time,trainEpoch)[0],id],c="red",s=0.5)
			l2 = ax[id].scatter(position_time[inEpochs(position_time, testEpochs)[0],0], positions[inEpochs(position_time, testEpochs)[0],id],c="black",s=0.5)
			ls.append([l1,l2])
		ax[0].set_ylabel("first feature")
		#TODO add histograms here...
		slider = plt.Slider(ax[-2], 'test starting index', 0, positions.shape[0]-sizeTest,
							valinit=testSetId, valstep=1)
		sliderSize = plt.Slider(ax[-1], 'test size', 0, positions.shape[0],
							valinit= sizeTest, valstep=1)
		def update(val):
			testSetId = slider.val
			sizeTest = sliderSize.val
			trainEpoch = np.array(
				[pmin, position_time[testSetId ][0], position_time[testSetId + sizeTest][0],
				 pmax])
			testEpochs = np.array(
				[position_time[testSetId][0], position_time[testSetId + sizeTest][0]])
			for id in range(len(ls)):
				l1,l2=ls[id]
				l1.set_offsets(np.transpose(np.stack([position_time[inEpochs(position_time, trainEpoch)[0],0],
										 positions[inEpochs(position_time, trainEpoch)[0],id]])))
				l2.set_offsets(np.transpose(np.stack([position_time[inEpochs(position_time, testEpochs)[0],0],
										 positions[inEpochs(position_time, testEpochs)[0],id]])))
			fig.canvas.draw_idle()

		slider.on_changed(update)
		sliderSize.on_changed(update)
		plt.show()

		testSetId = slider.val
		sizeTest = sliderSize.val
		children = [c.name for c in f.list_nodes("/behavior")]
		if "testEpochs" in children:
			f.remove_node("/behavior", "testEpochs")
		f.create_array("/behavior","testEpochs",np.array([position_time[testSetId][0],position_time[testSetId+sizeTest][0]]))
		if "trainEpochs" in children:
			f.remove_node("/behavior", "trainEpochs")
		f.create_array("/behavior", "trainEpochs", np.array([pmin,position_time[testSetId][0],position_time[testSetId+sizeTest][0],pmax]))
		f.flush() #effectively write down the modification we just made

		# # just a display of the first env variable and the histograms:
		# fig,ax = plt.subplots(2,2)
		# fig.suptitle("Test Set")
		# ax[0,0].plot(position_time[bestTestSet*sizeTest:bestTestSet*sizeTest+sizeTest],positions[bestTestSet*sizeTest:bestTestSet*sizeTest+sizeTest,0])
		# ax[0,1].hist(positions[bestTestSet*sizeTest:bestTestSet*sizeTest+sizeTest,0],bins=100)
		# ax[0,0].set_ylabel("position")
		# ax[1,0].plot(position_time[bestTestSet*sizeTest:bestTestSet*sizeTest+sizeTest],speeds[bestTestSet*sizeTest:bestTestSet*sizeTest+sizeTest,0])
		# ax[1,1].hist(speeds[bestTestSet*sizeTest:bestTestSet*sizeTest+sizeTest,0],bins=100)
		# ax[1,0].set_ylabel("speed")
		# fig.show()







class openEphysFilter:
	'''reads the open-ephys filtered data'''
	"""
		Overall just a parser from the C file which is structued as:
			Chan1Measure1 Chan2Measure2 ... ChanNMeasure1 Chan1Measure2 .....
		with data in short format (ie read as int16 in python).
		
	"""
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
		"""
			Take the next element of the dataReader, according to the shape of sample:
				if sample is one dimensional, extract 1 element
				if sample is @ dim, extract 1 element for each row.
			The read data is multiplied by 0.197 to be set in the correct unit (??)
			Note: dataReader is initialized with  "struct.iter_unpack(str(self.nChannels)+'h', self.dataFile.read())"

		  	the 'h' letter indicates a conversion from the C-type short to the python type Int16
		  	adding an interget in front (str(self.nChannels)) indicate that the packet contains this number of elements of the type h
		"""
		if sample.ndim==1:
			return np.array(next(self.dataReader))[self.channelList]*0.195
			# Pierre: Very suboptimal, because the .dat files and .fil array are both read here
			# and the data in the .dat files is not used...
			# to filter out uninteresting channels
			# Here the channelList simply gather all channels (over different groups) that were acquired
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

#
# def isTrainingExample(epochs, time):
# 	for e in range(len(epochs['train'])//2):
# 		if time >= epochs['train'][2*e] and time < epochs['train'][2*e+1]:
# 			return True
# 	for e in range(len(epochs['test'])//2):
# 		if time >= epochs['test'][2*e] and time < epochs['test'][2*e+1]:
# 			return False
# 	return None

def emptyData():
	return {'group':[], 'time':[], 'spike':[], 'position':[], 'position_index':[]} #'train':[]

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
			# TODO: change this to allow a varying number of features:
			self.position = np.array([0,0], dtype=float).reshape([1,2])
			self.position_time = np.array([0], dtype=float)
			self.startTime = 0
			self.stopTime = float("inf")
			# self.epochs = {"train": [], "test": [0, float("inf")]}
		else:
			self.list_channels, self.samplingRate, self.nChannels = get_params(self.path.xml)
			self.position, self.position_time = get_position(self.path.folder)
			# self.epochs = {
			# 	"train": epochs[0],
			# 	"test":  epochs[1]
			# }
			# self.startTime = min(self.epochs["train"] + self.epochs["test"])
			self.stopTime  = self.position_time[-1] # max(self.epochs["train"] + self.epochs["test"])
			self.startTime = self.position_time[0]


		if useOpenEphysFilter:
			self.filter = openEphysFilter(self.path.fil, self.list_channels, self.nChannels)
		else:
			self.filter = INTANFilter(350., self.samplingRate, self.list_channels)

		self.thresholdFactor   = 3 # in units of standard deviation
		# this is the threshold that is really used here
		# It multiplies the std of each channel in each group, taken over a complete buffer (where the late spike are added)

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
	# def learningTime(self):
	# 	return sum([self.epochs["train"][2*n+1]-self.epochs["train"][2*n] for n in range(len(self.epochs["train"])//2)])
	# def trainingPositions(self):
	# 	selected = [False for _ in range(len(self.position_time))]
	# 	for idx in range(len(self.position_time)):
	# 		for e in range(len(self.epochs['train'])//2):
	# 			if self.position_time[idx] >= self.epochs['train'][2*e] and self.position_time[idx] < self.epochs['train'][2*e+1]:
	# 				selected[idx] = True
	# 	return self.position[np.where(selected)]/self.maxPos()


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
		"""
			Given bufferSize, read bufferSize elements from the filter
			each elements will have the shape of the next(self.dataReader)) element which is
			of size nChannels too and of the type short converted in Int16

		"""
		temp = []
		for item in range(bufferSize):
			try:
				temp.append(self.filter.filter(np.array(next(self.dataReader))*0.195))
				# the value of np.array(next(self.dataReader))*0.195) is not used in the Open Ephys filter
				# so why multiply by 0.195 ?? (Pierre)
				# The key is that it is used by the IntanFilter.
				# In the case of the IntanFilter, the elements of self.dataReader is modified on read time
				# while for the open ephys it had been modified before!
			except StopIteration:
				self.lastBuffer = True
				break
		temp = np.array(temp)
		return temp

	def getLateSpikes(self):
		# store the late spikes
		# by concatenating it with the beginning of the next buffer of the filtered signal.
		# This has to be done over the different channel groups
		# Here this is not achieved in parallel, effectively creating a small bottlneck
		# but as the proces all join before the next buffer this bottleneck exist before this function.
		# Maybe we could design things so that everything is done in parallel over different groups...
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
		#let us try with a mmap:
		# number_timeSteps = os.stat(self.path.dat).st_size//(2*self.nChannels)
		# memmapData = np.memmap(self.path.dat,dtype=np.int16,mode='r',shape=(number_timeSteps,self.nChannels))
		# self.dataReader = map(lambda x:x,memmapData)
		print('extracting spikes.')

		self.inputQueue = ml.Queue()
		self.outputQueue = ml.Queue()
		self.proc = [ml.Process(target=findSpikesInGroupParallel, args=(self.inputQueue, self.outputQueue, self.samplingRate , self.thresholdFactor, self.startTime, self.position, self.position_time)) for _ in range(len(self.list_channels))]
		for p in self.proc:
			p.deamon = True
			p.start()

		datFileLengthInByte = os.stat(self.path.dat).st_size
		numBuffer = datFileLengthInByte // (2 * self.nChannels * BUFFERSIZE) + 1
		self.pbar = tqdm(total=numBuffer)

		n=0
		self.previousChannels = [] #store the number of channel in the previous group...
		# if channel are 1 by 1 incremented and the group association is constant by parts
		# then it also store the index of the first channel of the group
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

		### filter signal (or simply read it in the .fil file), compute new thresholds and get late spikes from previous buffer
		# The filtering is done over BUFFERSIZE elements
		self.filteredSignal = self.getFilteredBuffer(BUFFERSIZE)
		spikesFound = self.getLateSpikes()

		### copy end of previous buffer at the beginning of the filteredSignal
		#As we will detect spike starting from time 0 in this new buffer, we need to know at least 15 ms of the
		# signal just before so that we save all these spikes.
		if self.endOfLastBuffer != []:
			self.firstBuffer = False
			self.filteredSignal = np.concatenate([self.endOfLastBuffer, self.filteredSignal], axis=0)
		self.endOfLastBuffer = self.filteredSignal[-15:,:]

		#The filtered signal is in shape [n_channels]
		# For each group, a set of channels listed in elements of self.list_channels
		# the signal is extracted as well as the corresponding thresholds.
		# Then sent on the inputQueue to be process by the function findSpikesInGroupParallel ran in parallel
		for group in range(len(self.list_channels)):
			filteredBuffer = self.filteredSignal[:,self.previousChannels[group]:self.previousChannels[group]+len(self.list_channels[group])]
			thresholds	 = self.thresholds[self.previousChannels[group]:self.previousChannels[group]+len(self.list_channels[group])]
			self.inputQueue.put([self.pbar.n, group, thresholds, filteredBuffer])


		for _ in range(len(self.list_channels)):
			res = self.outputQueue.get(block=True)
			for key in emptyData().keys():
				spikesFound[key] += res[0][key]
				self.lateSpikes[key] += res[1][key]
		# As the block argument is used, we make sure to wait for len(self.list_channels)==nb groups elements to be available
		# So really working in a bottleneck manner.

		self.pbar.update(1)
		if spikesFound['time'] != [] and max(spikesFound["time"]) > self.stopTime:
			self.lastBuffer = True
		return spikesFound





def findSpikesInGroupParallel(inputQueue, outputQueue, samplingRate, thresholdFactor, startTime, position, position_time):
	"""
		How it detect spikes:
			moves with steps of size 15, ie dt= 15*1/sampling_rate
			A spike has shape 32, -15 from the spike time, 1 at spike time, 16 after.
			Therefore if it is detected at 15 steps before the last time,
		The inputQueue contains:
			N --> the number of buffer of size BUFFERSIZE we have been through, allows to compute the time
			group -->
			thesholds -->
			filteredBuffer --> an array of size BUFFERSIZE plus (if not the first buffer) 15 elements from the end of previous buffer

	"""
	while True:
		N, group, thresholds, filteredBuffer = inputQueue.get(block=True)

		if np.all(thresholds == np.zeros([filteredBuffer.shape[1]])):
			thresholds = thresholdFactor * filteredBuffer.std(axis=0)

		triggered = False

		spikesFound = emptyData()
		lateSpikes = emptyData()

		spl = 15
		#the spike looking time starts at 15, therefore a spike initiating at time t=0, i.e for the first buffer
		# will not be considered !
		while spl < filteredBuffer.shape[0]:
			for chnl in range(filteredBuffer.shape[1]):

				# check if cross negative trigger
				# Note Pierre: this test will work because we force spl to start at 15
				# so no boundary problem....
				# But there is a potential problem at the buffer limit if a late spike was detected on another channel
				# and the voltage goes below the threshold (so with a slight delay) after the buffer limit in the current channel
				# therefore we change the second  test to np.all(filteredBuffer[spl-1,:] > -thresholds)
				# rather than  filteredBuffer[spl-1, chnl] > -thresholds[chnl]
				if filteredBuffer[spl,chnl] < -thresholds[chnl] and np.all(filteredBuffer[spl-1,:] > -thresholds):
					triggered = True
					positiveTrigger = False

				if triggered:

					# Get peak or trough
					if positiveTrigger:
						spl = spl + np.argmax(filteredBuffer[spl:spl+15, chnl])
					else:
						spl = spl + np.argmin(filteredBuffer[spl:spl+15, chnl])

					# note: as the code is now, we are always in the case of negativeTrigger,
					# therefore testing if the filteredBuffer is below the thresholds value...

					if N==0:
						time = spl / samplingRate
						maxSplTimeToUseInBuffer = BUFFERSIZE-17
					else:
						time = (N * BUFFERSIZE + spl - 15) / samplingRate
						maxSplTimeToUseInBuffer = BUFFERSIZE - 17 + 15
					# train = isTrainingExample(epochs, time)
					# assert maxSplTimeToUseInBuffer == filteredBuffer.shape[0]-17

					# Do nothing unless after behaviour data has started
					if time > startTime:

						#OLD PROBLEM: modified by Puere on 23/03/21
						# That's a late spike, we'll have to wait till next buffer
						# why -18 here??
						# if spl = BUFFERSIZE - 17 + 15, spl+17 =  BUFFERSIZE + 15
						# which can be use as the last index, except for the first buffer
						# if we are in the case of the first buffer,
						# and spl = BUFFERSIZE-18+15 for example
						# filteredBuffer[spl-15:spl+17, :]
						# will not raise an error, but give an array of a smaller shape than 32
						# therefore the test... if np.shape(spike)[0]==32: fails
						# and the spike is not stored....
						# Fixed my using maxSplTimeToUseInBuffer varying if N==0 or another buffer

						if spl > maxSplTimeToUseInBuffer:
							lateSpikes['group'].   append(group)
							lateSpikes['time'].    append(time)
							#Note: no transpose of the filteredBuffer because it will then be concatenated with the
							# beginning of the next buffer to create a nice spike window.
							lateSpikes['spike'].   append(filteredBuffer[spl-15:, :])
							lateSpikes['position_index'].append(np.argmin(np.abs(position_time - time)))
							lateSpikes['position'].append( position[lateSpikes['position_index'][-1]] )
							# lateSpikes['train'].   append(train)


						else:
							spike = filteredBuffer[spl-15:spl+17, :].copy()

							if np.shape(spike)[0]==32:
								spikesFound['group'].   append(group)
								spikesFound['time'].	append(time)
								#note: here the reshape does not bring anything...
								spikesFound['spike'].   append( np.array(spike).reshape([32,filteredBuffer.shape[1]]).transpose() )
								spikesFound['position_index'].append(np.argmin(np.abs(position_time - time)))
								spikesFound['position'].append(position[spikesFound['position_index'][-1]])
								# spikesFound['train'].   append(train)

					spl += 15
					triggered = False
					break #leaves the for loop --> so the spike is stored as soon as it is detected over a channel.
			spl += 1

		outputQueue.put([spikesFound, lateSpikes])



def vectorize_spike_filter(filteredBuffer, N, group, thresholds):
	## We vectorize the spike filtering to improve its speed:

	possibleSpike = (filteredBuffer[15:, :] < -thresholds)*np.all(filteredBuffer[14:-1, :] > -thresholds)


	return None








