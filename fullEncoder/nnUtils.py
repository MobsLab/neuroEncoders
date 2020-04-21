import numpy as np
import tensorflow as tf

def last_relevant(output, length, timeMajor=False):
	''' Used to select the right output of 
		tf.rnn.dynamic_rnn for sequences of variable sizes '''
	if timeMajor:
		output = tf.transpose(output, [1,0,2])
	batch_size = tf.shape(output)[0]
	max_length = tf.shape(output)[1]
	out_size = int(output.get_shape()[2])
	index = tf.nn.relu(tf.range(0, batch_size) * max_length + tf.cast(length - 1, tf.int32))
	flat = tf.reshape(output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return relevant




def layerLSTM(lstmSize, dropout=0.0):
	cell = tf.contrib.rnn.LSTMBlockCell(lstmSize)
	return tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1-dropout)

class spikeNet:
	def __init__(self, nChannels=4, device="/cpu:0", nFeatures=128):
		self.nFeatures = nFeatures
		self.nChannels = nChannels
		self.device = device
		with tf.device(self.device):
			self.convLayer1 = tf.layers.Conv2D(8, [2,3], padding='SAME')
			self.convLayer2 = tf.layers.Conv2D(16, [2,3], padding='SAME')
			self.convLayer3 = tf.layers.Conv2D(32, [2,3], padding='SAME')

			self.maxPoolLayer1 = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')
			self.maxPoolLayer2 = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')
			self.maxPoolLayer3 = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')

			self.dropoutLayer = tf.layers.Dropout(0.5)
			self.denseLayer1 = tf.layers.Dense(self.nFeatures, activation='relu')
			self.denseLayer2 = tf.layers.Dense(self.nFeatures, activation='relu')
			self.denseLayer3 = tf.layers.Dense(self.nFeatures, activation='relu')

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



def getTrainingSpikes(params, spkTime, spkPos, spkGroups, spkSpikes, maxPos=None):
	totalLength = spkTime[-1] - spkTime[0]
	nBins = int(totalLength // params.windowLength) - 1
	binStartTime = [spkTime[0] + (i*params.windowLength) for i in range(nBins)]
	if maxPos == None:
		maxPos = np.max(spkPos)

	selection = np.zeros([len(binStartTime), 2], dtype=int)
	selection[0,0] = np.where(spkTime>binStartTime[0])[0][0]
	selection[0,1] = np.logical_and(spkTime>binStartTime[0], spkTime<binStartTime[0]+params.windowLength).sum()
	for idx in range(1, len(binStartTime)-1):
		binStart = binStartTime[idx]
		selection[idx,0] = selection[idx-1,0] + selection[idx-1,1]
		n=0
		while spkTime[selection[idx,0]+n] < binStart+params.windowLength:
			n += 1
		selection[idx,1] = n


	for idx in range(selection.shape[0]):
		allSpikes=[]
		pos = spkPos[selection[idx, 0], :] / maxPos
		length = selection[idx,1]
		groups = spkGroups[selection[idx,0]: selection[idx,0]+length]
		spikes = spkSpikes[selection[idx,0]: selection[idx,0]+length]
		for group in range(params.nGroups):
			s = list(spikes[np.where(groups==group)])
			if s == []:
				allSpikes.append(np.zeros([0, params.nChannels[group], 32]))
			else:
				allSpikes.append(np.stack(spikes[np.where(groups==group)], axis=0))
		yield (pos, groups, length) + tuple(allSpikes)




def serialize(params, pos, groups, length, *spikes):
	feat = {
		"pos": tf.train.Feature(float_list = tf.train.FloatList(value=pos)), 
		"length": tf.train.Feature(int64_list =  tf.train.Int64List(value=[length])),
		"groups": tf.train.Feature(int64_list = tf.train.Int64List(value=groups))}
	for g in range(params.nGroups):
		feat.update({"group"+str(g): tf.train.Feature(float_list = tf.train.FloatList(value=spikes[g].ravel()))})

	example_proto = tf.train.Example(features = tf.train.Features(feature = feat))
	return example_proto.SerializeToString()





def parse_serialized_example(params, feat_desc, ex_proto, batched=False):
	if batched:
		tensors = tf.io.parse_example(ex_proto, feat_desc)
	else:
		tensors = tf.io.parse_single_example(ex_proto, feat_desc)

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
		nonZeros  = tf.logical_not(tf.equal(tf.reduce_sum(tf.cast(tf.equal(tensors["group"+str(g)], zeros), tf.int32), axis=[1,2]), 32*params.nChannels[g]))
		tensors["group"+str(g)] = tf.gather(tensors["group"+str(g)], tf.where(nonZeros))[:,0,:,:]
	return tensors