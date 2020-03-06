import tensorflow as tf
import numpy as np
from tqdm import tqdm




def selectGroup(spk,grp,n, params, training):
	spk = tf.cast(spk, tf.float32)
	goodIndex = tf.cast(tf.equal(grp,n), tf.int32)
	idx = tf.where(goodIndex)
	res = tf.gather(spk, idx)
	res = res[:,0,:,:]

	if training:
		idMatrix = tf.eye(params.batch_size*params.maxLength)
	else:
		idMatrix = tf.eye(int(spk.get_shape()[0]))
	completionTensor = tf.gather(idMatrix, idx)

	res = (res, tf.transpose(completionTensor[:,0,:], [1,0]))
	return res

def parse_data_batch(params, training, *vals):
	# print(vals[0]) # length, int64, shape=(batch_size)
	# print(vals[1]) # group, int64, shape=(batch_size, maxLength), padded with -1
	# print(vals[2]) # spike, float64, shape=(batch_size, maxLength,4,32), padded with 0
	# print(vals[3]) # position, float64, shape=(batch_size,2)

	if training and params.timeMajor:
		grp = tf.reshape(tf.transpose(vals[1], [1,0]),     [params.batch_size*params.maxLength])
		spk = tf.reshape(tf.transpose(vals[2], [1,0,2,3]), [params.batch_size*params.maxLength, 4, 32])
	else:
		grp = vals[1]
		spk = vals[2]

	res = sum((selectGroup(spk, grp, n, params, training) for n in range(params.nGroups)), tuple([]))

	return  (vals[3], vals[0]) + res


def sequenceGenerator(maxLength, windowLength, training, groups, times, spikes, positions):

	totalLength = times[-1] - times[0]
	nBins = int(totalLength // windowLength) - 1
	binStartTime = [times[0] + (i*windowLength) for i in range(nBins)]
	if training:
		np.random.shuffle(binStartTime)

	lengthSelection = []
	groupSelection = []
	spikeSelection = []
	positionSelection = []
	print('preparing data parser.')
	for bin_start in tqdm(binStartTime):
		selection = np.where(np.logical_and(
			times >= bin_start,
			times < bin_start + windowLength))
		lengthSelection.append(len(selection[0]))
		groupSelection.append(groups[selection])
		spikeSelection.append(spikes[selection])

		# We need a position even when no spikes were seen in the bin
		if lengthSelection[-1]==0:
			selection2 = selection
			temp=bin_start + windowLength
			while len(selection2[0])==0:
				selection2 = np.where(np.logical_and( 
					times >= temp,
					times < temp + windowLength))
				temp += windowLength
			positionSelection.append(positions[selection2])
		else:
			positionSelection.append(positions[selection])

	while True:
		for bin in range(len(lengthSelection)):
			yield (
				lengthSelection[bin],
				np.pad(groupSelection[bin], (0, maxLength-lengthSelection[bin]), 'constant', constant_values=-1), 
				np.pad(spikeSelection[bin], ((0, maxLength-lengthSelection[bin]),(0,0),(0,0)), 'constant', constant_values=0), 
				np.mean(positionSelection[bin], axis=0))




def makeDataset(params, training=False):
	''' timeMajor is a bool to say if time dimension or batch dimension is first '''

	groupsPH = tf.placeholder(tf.int64,   shape=[None])
	timePH   = tf.placeholder(tf.float64, shape=[None])
	spikesPH = tf.placeholder(tf.float64, shape=[None,4,32])
	posPH    = tf.placeholder(tf.float64, shape=[None,2])

	temp = tf.data.Dataset
	temp = temp.from_generator(sequenceGenerator, 
		output_types = (
			tf.int64, # length
			tf.int64,  # group
			tf.float64,  # spike
			tf.float64),  # position
		output_shapes = (
			tf.TensorShape([]),
			tf.TensorShape([params.maxLength]),
			tf.TensorShape([params.maxLength, 4, 32]),
			tf.TensorShape([2])),
		args = (params.maxLength, params.windowLength, training, groupsPH, timePH, spikesPH, posPH))
	
	if training:
		temp = temp.batch(params.batch_size, drop_remainder=True)
	temp = temp.map(lambda *args: parse_data_batch(params, training, *args))
	it   = temp.make_initializable_iterator()
	return it, it.get_next(), {'groups':groupsPH, 'timeStamps':timePH, 'spikes':spikesPH, 'positions':posPH}
