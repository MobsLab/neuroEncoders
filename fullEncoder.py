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
		self.tfrec = self.folder + 'dataset.tfrec'

from importData import rawDataParser
from fullEncoder import datasetMaker



projectPath = xmlPath(os.path.expanduser(sys.argv[2]))



### Data
filterType = sys.argv[3]
if filterType=='external':
    useOpenEphysFilter=True
else:
    useOpenEphysFilter=False
print('using external filter:', useOpenEphysFilter)
spikeDetector = rawDataParser.SpikeDetector(projectPath, useOpenEphysFilter)
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




### Params
class Params:
	def __init__(self):
		self.nGroups = np.max(GRP_train) + 1
		self.dim_output = POS_train.shape[1]
		self.nChannels = spikeDetector.numChannelsPerGroup()
		self.length = len(SPK_train)

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


		# Figure out what will be the maximum length of a spike sequence for our windowLength
		self.maxLength = max(self.getMaxLength(SPT_train), self.getMaxLength(SPT_test))

	def getMaxLength(self, data):
		maxLength = 0
		n = 0
		bin_stop = data[0]
		while True:
			bin_stop = bin_stop + self.windowLength
			if bin_stop > data[-1]:
				maxLength = max(maxLength, len(data)-idx)
				break
			idx=n
			while data[n] < bin_stop:
				n += 1
			maxLength = max(maxLength, n-idx)
		return maxLength

params = Params()











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





















# xTraining = []
# for g in range(params.nGroups):
# 	xTraining.append(SPK_train.copy())
# 	for spk in range(len(SPK_train)):
# 		if GRP_train[spk]!=g:
# 			xTraining[g][spk] = np.zeros([params.nChannels[g], 32])
# 	xTraining[g] = np.stack(xTraining[g], axis=0)

def dataGenerator(pos, *spikes):
	totalLength = SPT_train[-1] - SPT_train[0]
	nBins = int(totalLength // params.windowLength) - 1
	binStartTime = [SPT_train[0] + (i*params.windowLength) for i in range(nBins)]

	while True:
		np.random.shuffle(binStartTime)
		for binStart in binStartTime:
			allSpikes=[]
			sel = np.logical_and(SPT_train>binStart, SPT_train<binStart+params.windowLength)
			length = np.sum(sel)
			for group in range(params.nGroups):
				spk = spikes[group][sel]
				spk = np.pad(spk, ((0, params.maxLength-spk.shape[0]),(0,0),(0,0)), 'constant', constant_values=0)
				allSpikes.append(spk)
			pos = POS_train[SPT_train>binStart][0,:]
			yield tuple(allSpikes) + (pos/np.max(POS_train), length)


def getTrainingSpikesWithBlanks():
	totalLength = SPT_train[-1] - SPT_train[0]
	nBins = int(totalLength // params.windowLength) - 1
	binStartTime = [SPT_train[0] + (i*params.windowLength) for i in range(nBins)]

	while True:
		np.random.shuffle(binStartTime)
		for binStart in binStartTime:
			allSpikes=[]
			sel = np.logical_and(SPT_train>binStart, SPT_train<binStart+params.windowLength)
			length = np.sum(sel)
			for group in range(params.nGroups):
				spikes = SPK_train[sel]
				groups = GRP_train[sel]==group
				for spk in range(spikes.shape[0]):
					if not groups[spk]:
						spikes[spk] = np.zeros([params.nChannels[group], 32])

				spikes = np.pad(np.stack(spikes, axis=0), ((0, params.maxLength-spikes.shape[0]),(0,0),(0,0)), 'constant', constant_values=0)
				allSpikes.append(spikes)
			pos = POS_train[SPT_train>binStart][0,:]
			yield (pos/np.max(POS_train), length) + tuple(allSpikes)


def serialize(pos, length, *spikes):
	feat = {
		"pos": tf.train.Feature(float_list = tf.train.FloatList(value=pos)), 
		"length": tf.train.Feature(int64_list =  tf.train.Int64List(value=[length]))}
	for g in range(params.nGroups):
		feat.update({"group"+str(g): tf.train.Feature(float_list = tf.train.FloatList(value=spikes[g].ravel()))})

	example_proto = tf.train.Example(features = tf.train.Features(feature = feat))
	return example_proto.SerializeToString()

if not os.path.isfile(projectPath.tfrec):
	dataset = tf.data.Dataset.from_generator(getTrainingSpikesWithBlanks,
		output_types = (tf.float64, tf.int64) + tuple(tf.float32 for _ in range(params.nGroups)),
		output_shapes = (tf.TensorShape([2]), tf.TensorShape([])) + tuple(tf.TensorShape([params.maxLength, params.nChannels[g], 32]) for g in range(params.nGroups)))
	iter = dataset.make_initializable_iterator()
	el = iter.get_next()
	with tf.Session() as sess:
		sess.run(iter.initializer)
		print('saving new dataset')
		with tf.python_io.TFRecordWriter(projectPath.tfrec) as writer:
			totalLength = SPT_train[-1] - SPT_train[0]
			nBins = int(totalLength // params.windowLength) - 1

			for _ in tqdm(range(params.nSteps)):
				example = sess.run(el)
				writer.write(serialize(*tuple(example)))

feat_desc = {"pos": tf.io.FixedLenFeature([2], tf.float32), "length": tf.io.FixedLenFeature([], tf.int64)}
for g in range(params.nGroups):
	feat_desc.update({"group"+str(g): tf.io.FixedLenFeature([params.maxLength, params.nChannels[g], 32], tf.float32)})

def parse_serialized_example(ex_proto):
	return tf.io.parse_example(ex_proto, feat_desc)

### Training model
with tf.Graph().as_default():

	print()
	print('TRAINING')

	dataset = tf.data.TFRecordDataset(projectPath.tfrec)
	dataset = dataset.batch(params.batch_size)
	dataset = dataset.map(parse_serialized_example)
	iter = dataset.make_initializable_iterator()
	iterators = iter.get_next()


	with tf.device(device_name):
		spkParserNet = []
		allFeatures = []

		# CNN plus dense on every group indepedently
		for group in range(params.nGroups):
			with tf.variable_scope("group"+str(group)+"-encoder"):
				x = tf.transpose(iterators["group"+str(group)], [1,0,2,3])
				x = tf.reshape(x, [-1, params.nChannels[group], 32])
				realSpikes = tf.math.logical_not(tf.equal(tf.reduce_sum(x, [1,2]), tf.constant(0.)))
				nSpikesTot = tf.shape(x)[0]; idMatrix = tf.eye(nSpikesTot)
				completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(realSpikes))[:,0,:], [1,0], name="completion")
				x = tf.boolean_mask(x, realSpikes)

			newSpikeNet = spikeNet(nChannels=params.nChannels[group], device="/cpu:0", nFeatures=params.nFeatures)
			x = newSpikeNet.apply(x)
			x = tf.matmul(completionTensor, x)
			x = tf.reshape(x, [params.maxLength, params.batch_size, params.nFeatures])
			allFeatures.append(x)
		allFeatures = tf.tuple(allFeatures)
		allFeatures = tf.concat(allFeatures, axis=2)

		# LSTM on the concatenated outputs of previous graphs
		if device_name=="/gpu:0":
			lstm = tf.contrib.cudnn_rnn.CudnnLSTM(params.lstmLayers, params.lstmSize, dropout=params.lstmDropout)
			outputs, finalState = lstm(allFeatures, training=True)
		else:
			lstm = [layerLSTM(params.lstmSize, dropout=params.lstmDropout) for _ in range(params.lstmLayers)]
			lstm = tf.nn.rnn_cell.MultiRNNCell(lstm)
			outputs, finalState = tf.nn.dynamic_rnn(
				lstm, 
				allFeatures, 
				dtype=tf.float32, 
				time_major=params.timeMajor,
				sequence_length=iterators["length"])

	# dense to extract regression on output and loss
	denseOutput = tf.layers.Dense(params.dim_output, activation = None, name="pos")
	denseLoss1  = tf.layers.Dense(params.lstmSize, activation = tf.nn.relu, name="loss1")
	denseLoss2  = tf.layers.Dense(1, activation = params.lossActivation, name="loss2")

	output = last_relevant(outputs, iterators["length"], timeMajor=params.timeMajor)
	outputLoss = denseLoss2(denseLoss1(output))[:,0]
	outputPos = denseOutput(output)

	lossPos =  tf.losses.mean_squared_error(outputPos, iterators["pos"], reduction=tf.losses.Reduction.NONE)
	lossPos =  tf.reduce_mean(lossPos, axis=1)
	lossLoss = tf.losses.mean_squared_error(outputLoss, lossPos)
	lossPos  = tf.reduce_mean(lossPos)

	optimizers = []
	for lr in range(len(params.learningRates)):
		optimizers.append(tf.train.RMSPropOptimizer(params.learningRates[lr]).minimize(lossPos + lossLoss))
	saver = tf.train.Saver()



	### Training and testing
	trainLosses = []
	with tf.Session() as sess:

		# initialize variables and input framework
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
		# fd = {spikesPH[g]: xTraining[g] for g in range(params.nGroups)}
		# fd.update({posPH: POS_train})
		# sess.run(iter.initializer, feed_dict=fd)
		sess.run(iter.initializer)

		### training
		epoch_loss = 0
		epoch_loss2 = 0
		loopSize = 50
		t = trange(params.nSteps, desc='Bar desc', leave=True)
		for i in t:

			for lr in range(len(params.learningRates)):
				if (i < (lr+1) * params.nSteps / len(params.learningRates)) and (i >= lr * params.nSteps / len(params.learningRates)):
					_, c, c2 = sess.run([optimizers[lr], lossPos, lossLoss])
					break
				if lr==len(params.learningRates)-1:
					print('not run:',i)
			t.set_description("loss: %f" % c)
			t.refresh()
			epoch_loss += c
			epoch_loss2 += c2
			
			if i%loopSize==0 and (i != 0):
				trainLosses.append(np.array([epoch_loss/loopSize, epoch_loss2/loopSize]))
				epoch_loss=0
				epoch_loss2=0

		saver.save(sess, projectPath.folder + '_graphForRnn')












# ### Inferring model
# variables=[]
# with tf.Graph().as_default():
# 	print()
# 	print()
# 	print('INFERRING')
# 	inferSpkNet = []
# 	dataIterator, testingTensors, dataPlaceholders = datasetMaker.makeDataset(params, training=False)
	
# 	with tf.device("/cpu:0"):
# 		inferringFeedDict = []
# 		embeddings = []

# 		# CNN and dense for each group
# 		for group in range(params.nGroups):
# 			inferSpkNet.append(spikeNet(nChannels=params.nChannels, device="/cpu:0", nFeatures=params.nFeatures))

# 			out = inferSpkNet[group].apply(testingTensors[2*group+2])
# 			embeddings.append(tf.matmul(testingTensors[2*group+3], out))
# 			variables += inferSpkNet[group].variables()
# 		embeddings = tf.tuple(embeddings)
# 		fullEmbedding = tf.concat(embeddings, axis=1)
		
# 		# LSTM on concatenated outputs
# 		with tf.variable_scope("cudnn_lstm"):
# 			lstm = tf.nn.rnn_cell.MultiRNNCell(
# 				[tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(params.lstmSize) for _ in range(params.lstmLayers)])
# 			outputs, finalState = tf.nn.dynamic_rnn(
# 				lstm, 
# 				tf.expand_dims(fullEmbedding, axis=1), 
# 				dtype=tf.float32, 
# 				time_major=params.timeMajor, 
# 				sequence_length=tf.expand_dims(testingTensors[1],0))
# 			variables += lstm.variables
		
# 		output = last_relevant(outputs, testingTensors[1], timeMajor=params.timeMajor)
# 		denseOutput = tf.layers.Dense(params.dim_output, activation = None, name="pos")
# 		denseLoss1  = tf.layers.Dense(params.lstmSize, activation = tf.nn.relu, name="loss1")
# 		denseLoss2  = tf.layers.Dense(1, activation = params.lossActivation, name="loss2")

# 		position = tf.reshape(denseOutput(tf.reshape(output, [-1,params.lstmSize])), [2])
# 		loss     = tf.reshape(denseLoss2(denseLoss1(tf.reshape(output, [-1,params.lstmSize]))), [1])
# 		variables += denseOutput.variables
# 		variables += denseLoss1.variables
# 		variables += denseLoss2.variables
		
# 		subGraphToRestore = tf.train.Saver({v.op.name: v for v in variables})

		


# 	### Inferring
# 	with tf.Session() as sess:

# 		# Restoring variables and initialize input framework
# 		subGraphToRestore.restore(sess, projectPath.folder + '_graphForRnn')
# 		sess.run(dataIterator.initializer, 
# 			feed_dict={dataPlaceholders['groups']:GRP_test, 
# 			dataPlaceholders['timeStamps']:SPT_test, 
# 			dataPlaceholders['spikes']:SPK_test, 
# 			dataPlaceholders['positions']:POS_test / np.max(POS_train)})

# 		# inferring
# 		testOutput = []
# 		for bin in trange(int((SPT_test[-1]-SPT_test[0])//params.windowLength)-1):
# 			testOutput.append(np.concatenate(sess.run([position, loss]), axis=0))
	
# 	testOutput = np.array(testOutput)

# 	pos = []
# 	spd = []
# 	n = 0
# 	bin_stop = SPT_test[0]
# 	while True:
# 		bin_stop = bin_stop + params.windowLength
# 		if bin_stop > SPT_test[-1]:
# 			break
# 		idx=n
# 		while SPT_test[n] < bin_stop:
# 			n += 1
# 		pos.append(np.mean(POS_test[idx:n,:], axis=0))
# 		spd.append(np.mean(SPD_test[idx:n,:]))
# 	pos.pop()
# 	spd.pop()
# 	pos = np.array(pos) / np.max(POS_train)
# 	spd = np.array(spd)
	

# 	fileName = projectPath.folder + '_resultsForRnn_temp'
# 	np.savez(os.path.expanduser(fileName), pos=pos, spd=spd, testOutput=testOutput, trainLosses=[])
# 	# np.savez(os.path.expanduser(fileName), pos=pos, spd=spd, testOutput=testOutput, trainLosses=trainLosses)











### Back compatibility converting before inferring
variables = []
with tf.Graph().as_default(), tf.device("/cpu:0"):


	# one CNN network per group of electrode
	embeddings = []
	for group in range(params.nGroups):
		with tf.variable_scope("group"+str(group)+"-encoder"):
			x = tf.placeholder(tf.float32, shape=[None, params.nChannels[group], 32], name="x")
			realSpikes = tf.math.logical_not(tf.equal(tf.reduce_sum(x, [1,2]), tf.constant(0.)))
			nSpikesTot = tf.shape(x)[0]; idMatrix = tf.eye(nSpikesTot)
			completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(realSpikes))[:,0,:], [1,0], name="completion")
			x = tf.boolean_mask(x, realSpikes)
		newSpikeNet = spikeNet(nChannels=params.nChannels[group], device="/cpu:0", nFeatures=params.nFeatures)
		x = newSpikeNet.apply(x)
		x = tf.matmul(completionTensor, x)
		# x = tf.pad(x, [[0,0],[group*params.nFeatures, (params.nGroups-group-1)*params.nFeatures]], "CONSTANT")

		embeddings.append(x)
		variables += newSpikeNet.variables()
	fullEmbedding = tf.concat(embeddings, axis=1)
	# fullEmbedding = tf.concat(embeddings, axis=0)

	
	# LSTM on concatenated outputs
	if device_name=="/gpu:0":
		with tf.variable_scope("cudnn_lstm"):
			lstm = tf.nn.rnn_cell.MultiRNNCell(
				[tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(params.lstmSize) for _ in range(params.lstmLayers)])
	else:
		lstm = [layerLSTM(params.lstmSize, dropout=params.lstmDropout) for _ in range(params.lstmLayers)]
		lstm = tf.nn.rnn_cell.MultiRNNCell(lstm)
		outputs, finalState = tf.nn.dynamic_rnn(
			lstm, 
			tf.expand_dims(fullEmbedding, axis=1), 
			dtype=tf.float32, 
			time_major=params.timeMajor)
		variables += lstm.variables

	# Final position decoder
	output = tf.cond(tf.shape(outputs)[0]>0, lambda: outputs[-1,:,:], lambda: outputs)
	denseOutput = tf.layers.Dense(params.dim_output, activation = None, name="pos")
	denseLoss1  = tf.layers.Dense(params.lstmSize, activation = tf.nn.relu, name="loss1")
	denseLoss2  = tf.layers.Dense(1, activation = params.lossActivation, name="loss2")

	x = denseOutput(tf.reshape(output, [-1,params.lstmSize]))
	y = denseLoss2(denseLoss1(tf.reshape(output, [-1,params.lstmSize])))
	variables += denseOutput.variables
	variables += denseLoss1.variables
	variables += denseLoss2.variables

	with tf.variable_scope("bayesianDecoder"):
		position = tf.identity(tf.cond(tf.shape(outputs)[0]>0, lambda: tf.reshape(x, [2]), lambda: tf.constant([0,0], dtype=tf.float32)), name="positionGuessed")
		loss     = tf.identity(tf.cond(tf.shape(outputs)[0]>0, lambda: tf.reshape(y, [1]), lambda: tf.constant([0], dtype=tf.float32)), name="standardDeviation")
		fakeProba= tf.constant(np.zeros([45,45]), dtype=tf.float32, name="positionProba")		
	
	subGraphToRestore = tf.train.Saver({v.op.name: v for v in variables})

	### Converting
	graphToSave = tf.train.Saver()
	with tf.Session() as sess:
		subGraphToRestore.restore(sess, projectPath.folder + '_graphForRnn')
		graphToSave.save(sess, projectPath.folder + '_graphDecoder')



### Loading and inferring
tf.contrib.rnn
with tf.Graph().as_default(), tf.device("/cpu:0"):
	saver = tf.train.import_meta_graph(projectPath.folder + '_graphDecoder.meta')

	def getSpikes(bin, group):
		binStart = SPT_test[0] + bin*params.windowLength
		return SPK_test[
			np.logical_and(
				np.logical_and(SPT_test>binStart, SPT_test<binStart+params.windowLength),
				GRP_test==group)]
	def getSpikesWithBlanks(bin, group):
		binStart = SPT_test[0] + bin*params.windowLength
		spikes = SPK_test[np.logical_and(SPT_test>binStart, SPT_test<binStart+params.windowLength)]
		groups = GRP_test[np.logical_and(SPT_test>binStart, SPT_test<binStart+params.windowLength)]==group
		for spk in range(spikes.shape[0]):
			spikes[spk,:,:] *= groups[spk]
		return spikes

	with tf.Session() as sess:
		saver.restore(sess, projectPath.folder + '_graphDecoder')

		testOutput = []
		for bin in trange(int((SPT_test[-1]-SPT_test[0])//params.windowLength)-1):
			testOutput.append(np.concatenate(
				sess.run(
					[tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionGuessed:0"), 
					 tf.get_default_graph().get_tensor_by_name("bayesianDecoder/standardDeviation:0")], 
					{tf.get_default_graph().get_tensor_by_name("group"+str(group)+"-encoder/x:0"):getSpikesWithBlanks(bin, group)
						for group in range(params.nGroups)}), 
				axis=0))

	testOutput = np.array(testOutput)

	pos = []
	spd = []
	n = 0
	bin_stop = SPT_test[0]
	while True:
		bin_stop = bin_stop + params.windowLength
		if bin_stop > SPT_test[-1]:
			break
		idx=n
		while SPT_test[n] < bin_stop:
			n += 1
		pos.append(np.mean(POS_test[idx:n,:], axis=0))
		spd.append(np.mean(SPD_test[idx:n,:]))
	pos.pop()
	spd.pop()
	pos = np.array(pos) / np.max(POS_train)
	spd = np.array(spd)

	fileName = projectPath.folder + '_resultsForRnn_temp'
	np.savez(os.path.expanduser(fileName), pos=pos, spd=spd, testOutput=testOutput, trainLosses=[])
	# np.savez(os.path.expanduser(fileName), pos=pos, spd=spd, testOutput=testOutput, trainLosses=trainLosses)







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

print(outjsonStr)

outjson = json.dumps(outjsonStr, indent=4)
with open(projectPath.json,"w") as json_file:
    json_file.write(outjson)
