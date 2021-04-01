import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd



class spikeNet:
	def __init__(self, nChannels=4, device="/cpu:0", nFeatures=128):
		self.nFeatures = nFeatures
		self.nChannels = nChannels
		self.device = device
		with tf.device(self.device):
			# modified by Pierre 16/02/2021: migration to 2.0 of RNN
			self.convLayer1 = tf.keras.layers.Conv2D(8, [2,3], padding='SAME')
			# conv2D: 8 := filters , dimensionality of output space: number of output filters
			#		  [2,3]: kernel_size: height, width of 2D convolution
			#         padding="same" (case-insensitive): padding evenly to the left or up/down so that output
			# 			has same size as input....

			self.convLayer2 = tf.keras.layers.Conv2D(16, [2,3], padding='SAME')
			self.convLayer3 = tf.keras.layers.Conv2D(32, [2,3], padding='SAME') #change from 32 to 16 and [2;3] to [2;2]

			self.maxPoolLayer1 = tf.keras.layers.MaxPool2D([1,2], [1,2], padding='SAME')
			self.maxPoolLayer2 = tf.keras.layers.MaxPool2D([1,2], [1,2], padding='SAME')
			self.maxPoolLayer3 = tf.keras.layers.MaxPool2D([1,2], [1,2], padding='SAME')

			self.dropoutLayer = tf.keras.layers.Dropout(0.5)
			self.denseLayer1 = tf.keras.layers.Dense(self.nFeatures, activation='relu')
			self.denseLayer2 = tf.keras.layers.Dense(self.nFeatures, activation='relu')
			self.denseLayer3 = tf.keras.layers.Dense(self.nFeatures, activation='relu')

	def __call__(self, input):
		return self.apply(input)


	def apply(self, input):
		with tf.device(self.device):
			x = tf.expand_dims(input, axis=3)
			x = self.convLayer1(x)
			x = self.maxPoolLayer1(x)
			x = self.convLayer2(x)
			x = self.maxPoolLayer2(x)
			x = self.convLayer3(x)
			x = self.maxPoolLayer3(x)

			x = tf.reshape(x, [-1, self.nChannels*8*16]) #change from 32 to 16 and 4 to 8
			#by pooling we moved from 32 bins to 4. By convolution we generated 32 channels
			x = self.denseLayer1(x)
			x = self.dropoutLayer(x)
			x = self.denseLayer2(x)
			x = self.denseLayer3(x)
		return x

	def variables(self):
		return self.convLayer1.variables + self.convLayer2.variables + self.convLayer3.variables + \
			self.maxPoolLayer1.variables + self.maxPoolLayer2.variables + self.maxPoolLayer3.variables + \
			self.denseLayer1.variables + self.denseLayer2.variables + self.denseLayer3.variables




def getSpikeSequences(params, generator):
	# Used in the main function to  get the Spike sequence from the spike generator
	# and cast it into an "example" format that will then be decoded by tensorflow inputs system tf.io
	# as the key word yield is used, this function effectively returns a generator
	windowStart = None

	length = 0
	times = []
	groups = []
	allSpikes = [[] for _ in range(params.nGroups)] # nGroups of array each containing the spike of a group
	for train, grp, time, spike, pos in generator:
		if windowStart == None:
			windowStart = time # at the first pass: initialize the windowStart on "time"

		if time > windowStart + params.windowLength:
			# if we got over the window-length
			allSpikes = [np.zeros([0, params.nChannels[g], 32]) \
				if allSpikes[g]==[] else np.stack(allSpikes[g], axis=0) \
				for g in range(params.nGroups)] # stacks each list of array in allSpikes
			# allSpikes then is composed of nGroups array of stacked "spike"
			res = {"train": train, "pos": pos, "groups": groups, "length": length, "times": times}
			res.update({"spikes"+str(g): allSpikes[g] for g in range(params.nGroups)})
			yield res
			# increase the windowStart by one window length
			length = 0
			groups = []
			times = []
			allSpikes = [[] for _ in range(params.nGroups)] # The all Spikes is reset so that we stop gathering the spikes in this window
			windowStart += params.windowLength
			#Pierre: Then we increment the windowStart until it is above the last seen spike time
			while time > windowStart + params.windowLength:
				# res = {"train": train, "pos": pos, "groups": [], "length": 0, "times": []}
				# res.update({"spikes"+str(g): np.zeros([0, params.nChannels[g], 32]) for g in range(params.nGroups)})
				# yield res
				windowStart += params.windowLength
		# Pierre: While we have not entered a new window, we start to gather spikes, time and group
		# of each input.
		times.append(time)
		groups.append(grp)
		# Pierre: so here we understand that groups indicate for each spikes array
		# obtained from the generator the groups from which they belong to !
		# But the spike array are well mapped separately to different groups:
		allSpikes[grp].append(spike)
		length += 1
		# --> so length correspond to the number of spike sequence obtained from the generator for each window considered





def serializeSpikeSequence(params, pos, groups, length, times, *spikes):
	# Moves from the info obtained via the SpikeDetector -> spikeGenerator -> getSpikeSequences pipeline toward the
	# tensorflow storing file. This take a specific format, which is here declared through the dic+tf.train.Feature
	# organisation. We see that groups now correspond to the "spikes" we had before....

	feat = {
		"pos": tf.train.Feature(float_list = tf.train.FloatList(value=pos)), 
		"length": tf.train.Feature(int64_list =  tf.train.Int64List(value=[length])),
		"groups": tf.train.Feature(int64_list = tf.train.Int64List(value=groups)),
		"time": tf.train.Feature(float_list = tf.train.FloatList(value=[np.mean(times)]))
	}
	# Pierre: convert the spikes dic into a tf.train.Feature, used for the tensorflow protocol.
	# their is no reason to change the key name but still done here.
	for g in range(params.nGroups):
		feat.update({"group"+str(g): tf.train.Feature(float_list = tf.train.FloatList(value=spikes[g].ravel()))})

	example_proto = tf.train.Example(features = tf.train.Features(feature = feat))
	return example_proto.SerializeToString() #to string

def serializeSingleSpike(params, clu, spike):
	feat = {
		"clu": tf.train.Feature(int64_list = tf.train.Int64List(value=[clu])),
		"spike": tf.train.Feature(float_list = tf.train.FloatList(value=spike.ravel()))
	}
	example_proto = tf.train.Example(features = tf.train.Features(feature = feat))
	return example_proto.SerializeToString()



@tf.function
def parseSerializedSequence(params, feat_desc, ex_proto, batched=False):
	if batched:
		tensors = tf.io.parse_example(serialized=ex_proto, features=feat_desc)
	else:
		tensors = tf.io.parse_single_example(serialized=ex_proto, features=feat_desc)

	tensors["groups"] = tf.sparse.to_dense(tensors["groups"], default_value=-1)
	# Pierre 13/02/2021: Why use sparse.to_dense, and not directly a FixedLenFeature?
	# Probably because he wanted a variable length <> inputs sequences
	tensors["groups"] = tf.reshape(tensors["groups"], [-1])
	# with this reshape; batch and variable length of time window are merged.... empty values are assigned -1 !
	for g in range(params.nGroups):
		#here 32 correspond to the number of discretized time bin for a spike
		zeros = tf.constant(np.zeros([params.nChannels[g], 32]), tf.float32)
		tensors["group"+str(g)] = tf.sparse.reshape(tensors["group"+str(g)], [-1])
		tensors["group"+str(g)] = tf.sparse.to_dense(tensors["group"+str(g)])
		tensors["group"+str(g)] = tf.reshape(tensors["group"+str(g)], [-1])
		if batched:
			tensors["group"+str(g)] = tf.reshape(tensors["group"+str(g)], [params.batch_size, -1, params.nChannels[g], 32])
		# even if batched: gather all together
		tensors["group"+str(g)] = tf.reshape(tensors["group"+str(g)], [-1, params.nChannels[g], 32])
		# Pierre 12/03/2021: the batch_size and timesteps are gathered together
		nonZeros  = tf.logical_not(tf.equal(tf.reduce_sum(input_tensor=tf.cast(tf.equal(
			tensors["group"+str(g)], zeros), tf.int32), axis=[1,2]), 32*params.nChannels[g]))
		# nonZeros: control that the voltage measured is not 0, at all channels and time bin inside the detected spike
		tensors["group"+str(g)] = tf.gather(tensors["group"+str(g)], tf.where(nonZeros))[:,0,:,:]
		#I don't understand why it can then call [:,0,:,:] as the output tensor of gather should have the same
		# shape as tensors["group"+str(g)"], [-1,params.nChannels[g],32] ...

	return tensors

def parseSerializedSpike(params, feat_desc, ex_proto, batched=False):
	if batched:
		tensors = tf.io.parse_example(serialized=ex_proto, features=feat_desc)
	else:
		tensors = tf.io.parse_single_example(serialized=ex_proto, features=feat_desc)
	return tensors



def spikeGenerator(projectPath, spikeDetector, maxPos=1):
	# Pierre: spikeGenerator:
		# Uses either the _rawSpikesForRnn.npz file if it is present in the folder
		# Or the spikeDetector
	# 	maxPos:
	# The spike generator is used to build the frec dataset which are then used by tensorflow more efficiently....
	if os.path.isfile(projectPath.folder+'_rawSpikesForRnn.npz'):
		# corrected by Pierre ,13/02/2021: the try/except did not make sense
		print('loading data')
		Results = np.load(projectPath.folder + '_rawSpikesForRnn.npz', allow_pickle=True)
		SPT_train = Results['arr_0']
		SPT_test = Results['arr_1']
		GRP_train = Results['arr_2']
		GRP_test = Results['arr_3']
		SPK_train = Results['arr_4']
		SPK_test = Results['arr_5']
		SPK_train = Results['arr_6']
		SPK_test = Results['arr_7']

		print('data loaded')
		def genFromOld():
			pbar = tqdm(total=len(GRP_train)+len(GRP_test))
			for args in zip(GRP_train, SPT_train, SPK_train, POS_train/maxPos):
				yield (True,)+args
				pbar.update(1)
			for args in zip(GRP_test, SPT_test, SPK_test, POS_test/maxPos):
				yield (False,)+args
				pbar.update(1)
			pbar.close()
		return genFromOld
	else:
		# Generator function: from the spikeDetector to a format that will be used by the sequence generator
		def genFromDet():
			for spikes in spikeDetector:
				if len(spikes['time'])==0:
					continue
				# sort by the spike group, and provide a zip object giving tuple of size 5....
				# We observe that the position is normalized here...
				for args in sorted(zip(spikes["train"],spikes['group'],spikes['time'],spikes['spike'],[p/maxPos for p in spikes['position']]), key=lambda x:x[2]):
					yield args
		return genFromDet



