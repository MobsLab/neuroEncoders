import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def last_relevant(output, length, timeMajor=False):
	''' Used to select the right output of 
		tf.rnn.dynamic_rnn for sequences of variable sizes '''
	if timeMajor:
		output = tf.transpose(a=output, perm=[1,0,2])
	batch_size = tf.shape(input=output)[0]
	max_length = tf.shape(input=output)[1]
	out_size = int(output.get_shape()[2])
	index = tf.nn.relu(tf.range(0, batch_size) * max_length + tf.cast(length - 1, tf.int32))
	flat = tf.reshape(output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return relevant




def layerLSTM(lstmSize, dropout=0.0):
	# modified by Pierre 16/02/2021: migration to 2.0 of RNN
	cell = tf.keras.layers.LSTM(lstmSize,recurrent_dropout=dropout)
	# lstmSize: dim of output space
	# recurrent_dropout: faction of units to drop for the linear transfo of the recurrent state
	return cell

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
			self.convLayer3 = tf.keras.layers.Conv2D(32, [2,3], padding='SAME')

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

			x = tf.reshape(x, [-1, self.nChannels*4*32])
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
	windowStart = None

	length = 0
	times = []
	groups = []
	allSpikes = [[] for _ in range(params.nGroups)]
	for train, grp, time, spike, pos in generator:
		if windowStart == None:
			windowStart = time

		if time > windowStart + params.windowLength:
			allSpikes = [np.zeros([0, params.nChannels[g], 32]) \
				if allSpikes[g]==[] else np.stack(allSpikes[g], axis=0) \
				for g in range(params.nGroups)]
			res = {"train": train, "pos": pos, "groups": groups, "length": length, "times": times}
			res.update({"spikes"+str(g): allSpikes[g] for g in range(params.nGroups)})
			yield res
			length = 0
			groups = []
			times = []
			allSpikes = [[] for _ in range(params.nGroups)]
			windowStart += params.windowLength
			while time > windowStart + params.windowLength:
				res = {"train": train, "pos": pos, "groups": [], "length": 0, "times": []}
				res.update({"spikes"+str(g): np.zeros([0, params.nChannels[g], 32]) for g in range(params.nGroups)})
				yield res
				windowStart += params.windowLength

		times.append(time)
		groups.append(grp)
		allSpikes[grp].append(spike)
		length += 1





def serializeSpikeSequence(params, pos, groups, length, times, *spikes):
	feat = {
		"pos": tf.train.Feature(float_list = tf.train.FloatList(value=pos)), 
		"length": tf.train.Feature(int64_list =  tf.train.Int64List(value=[length])),
		"groups": tf.train.Feature(int64_list = tf.train.Int64List(value=groups)),
		"time": tf.train.Feature(float_list = tf.train.FloatList(value=[np.mean(times)]))
	}
	for g in range(params.nGroups):
		feat.update({"group"+str(g): tf.train.Feature(float_list = tf.train.FloatList(value=spikes[g].ravel()))})

	example_proto = tf.train.Example(features = tf.train.Features(feature = feat))
	return example_proto.SerializeToString()

def serializeSingleSpike(params, clu, spike):
	feat = {
		"clu": tf.train.Feature(int64_list = tf.train.Int64List(value=[clu])),
		"spike": tf.train.Feature(float_list = tf.train.FloatList(value=spike.ravel()))
	}
	example_proto = tf.train.Example(features = tf.train.Features(feature = feat))
	return example_proto.SerializeToString()




def parseSerializedSequence(params, feat_desc, ex_proto, batched=False):
	if batched:
		tensors = tf.io.parse_example(serialized=ex_proto, features=feat_desc)
	else:
		tensors = tf.io.parse_single_example(serialized=ex_proto, features=feat_desc)

	tensors["groups"] = tf.sparse.to_dense(tensors["groups"], default_value=-1)
	tensors["groups"] = tf.reshape(tensors["groups"], [-1])
	for g in range(params.nGroups):
		zeros = tf.constant(np.zeros([params.nChannels[g], 32]), tf.float32)
		tensors["group"+str(g)] = tf.sparse.reshape(tensors["group"+str(g)], [-1])
		tensors["group"+str(g)] = tf.sparse.to_dense(tensors["group"+str(g)])
		tensors["group"+str(g)] = tf.reshape(tensors["group"+str(g)], [-1])
		if batched:
			tensors["group"+str(g)] = tf.reshape(tensors["group"+str(g)], [params.batch_size, -1, params.nChannels[g], 32])
		tensors["group"+str(g)] = tf.reshape(tensors["group"+str(g)], [-1, params.nChannels[g], 32])
		nonZeros  = tf.logical_not(tf.equal(tf.reduce_sum(input_tensor=tf.cast(tf.equal(
			tensors["group"+str(g)], zeros), tf.int32), axis=[1,2]), 32*params.nChannels[g]))
		tensors["group"+str(g)] = tf.gather(tensors["group"+str(g)], tf.compat.v1.where(nonZeros))[:,0,:,:]
	return tensors

def parseSerializedSpike(params, feat_desc, ex_proto, batched=False):
	if batched:
		tensors = tf.io.parse_example(serialized=ex_proto, features=feat_desc)
	else:
		tensors = tf.io.parse_single_example(serialized=ex_proto, features=feat_desc)
	return tensors



def spikeGenerator(projectPath, spikeDetector, maxPos=1):
	if os.path.isfile(projectPath.folder+'_rawSpikesForRnn.npz'):
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
			loaded='data loaded'
			print(loaded)
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
		def genFromDet():
			for spikes in spikeDetector:
				if len(spikes['time'])==0:
					continue
				for args in sorted(zip(spikes["train"],spikes['group'],spikes['time'],spikes['spike'],[p/maxPos for p in spikes['position']]), key=lambda x:x[2]):
					yield args
		return genFromDet



