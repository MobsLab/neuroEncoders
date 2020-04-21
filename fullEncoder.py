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
from fullEncoder import datasetMaker, nnUtils


trainLosses = None
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

















	

feat_desc = {"pos": tf.io.FixedLenFeature([params.dim_output], tf.float32), "length": tf.io.FixedLenFeature([], tf.int64), "groups": tf.io.VarLenFeature(tf.int64)}
for g in range(params.nGroups):
	feat_desc.update({"group"+str(g): tf.io.VarLenFeature(tf.float32)})

### Training model
with tf.Graph().as_default():

	print()
	print('TRAINING')

	dataset = tf.data.TFRecordDataset(projectPath.tfrec).shuffle(params.nSteps).repeat()
	dataset = dataset.batch(params.batch_size)
	dataset = dataset.map(lambda *vals: nnUtils.parse_serialized_example(params, feat_desc, *vals, batched=True))
	iter = dataset.make_initializable_iterator()
	iterators = iter.get_next()


	with tf.device(device_name):
		spkParserNet = []
		allFeatures = []

		# CNN plus dense on every group indepedently
		for group in range(params.nGroups):
			with tf.variable_scope("group"+str(group)+"-encoder"):
				x = iterators["group"+str(group)]
				idMatrix = tf.eye(tf.shape(iterators["groups"])[0])
				completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(tf.equal(iterators["groups"], group)))[:,0,:], [1,0], name="completion")

			newSpikeNet = nnUtils.spikeNet(nChannels=params.nChannels[group], device="/cpu:0", nFeatures=params.nFeatures)
			x = newSpikeNet.apply(x)
			x = tf.matmul(completionTensor, x)
			x = tf.reshape(x, [params.batch_size, -1, params.nFeatures])
			if params.timeMajor:
				x = tf.transpose(x, [1,0,2])
			allFeatures.append(x)
		allFeatures = tf.tuple(allFeatures)
		allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

		# LSTM on the concatenated outputs of previous graphs
		if device_name=="/gpu:0":
			lstm = tf.contrib.cudnn_rnn.CudnnLSTM(params.lstmLayers, params.lstmSize, dropout=params.lstmDropout)
			outputs, finalState = lstm(allFeatures, training=True)
		else:
			lstm = [nnUtils.layerLSTM(params.lstmSize, dropout=params.lstmDropout) for _ in range(params.lstmLayers)]
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

	output = nnUtils.last_relevant(outputs, iterators["length"], timeMajor=params.timeMajor)
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

		saver.save(sess, projectPath.folder + '_graphDecoder')









### Back compatibility converting before inferring
variables = []
print("Cleaning graph.")
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
		newSpikeNet = nnUtils.spikeNet(nChannels=params.nChannels[group], device="/cpu:0", nFeatures=params.nFeatures)
		x = newSpikeNet.apply(x)
		x = tf.matmul(completionTensor, x)

		embeddings.append(x)
		variables += newSpikeNet.variables()
	fullEmbedding = tf.concat(embeddings, axis=1, name="concat2")

	
	# LSTM on concatenated outputs
	if device_name=="/gpu:0":
		with tf.variable_scope("cudnn_lstm"):
			lstm = tf.nn.rnn_cell.MultiRNNCell(
				[tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(params.lstmSize) for _ in range(params.lstmLayers)])
	else:
		lstm = [nnUtils.layerLSTM(params.lstmSize, dropout=params.lstmDropout) for _ in range(params.lstmLayers)]
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
		subGraphToRestore.restore(sess, projectPath.folder + '_graphDecoder')
		graphToSave.save(sess, projectPath.folder + '_graphDecoder')

















### Loading and inferring
print()
print("INFERRING")

tf.contrib.rnn
with tf.Graph().as_default(), tf.device("/cpu:0"):

	dataset = tf.data.TFRecordDataset(projectPath.testTfrec)
	cnt     = dataset.batch(1).repeat(1).reduce(np.int64(0), lambda x, _: x + 1)
	dataset = dataset.map(lambda *vals: nnUtils.parse_serialized_example(params, feat_desc, *vals))
	iter    = dataset.make_initializable_iterator()
	spikes  = iter.get_next()
	for group in range(params.nGroups):
		idMatrix = tf.eye(tf.shape(spikes["groups"])[0])
		completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(tf.equal(spikes["groups"], group)))[:,0,:], [1,0], name="completion")
		spikes["group"+str(group)] = tf.tensordot(completionTensor, spikes["group"+str(group)], axes=[[1],[0]])


	saver = tf.train.import_meta_graph(projectPath.folder + '_graphDecoder.meta')


	with tf.Session() as sess:
		saver.restore(sess, projectPath.folder + '_graphDecoder')

		pos = []
		spd = []
		testOutput = []
		sess.run(iter.initializer)
		for b in trange(cnt.eval()):
			tmp = sess.run(spikes)
			pos.append(tmp["pos"])
			testOutput.append(np.concatenate(
				sess.run(
					[tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionGuessed:0"), 
					 tf.get_default_graph().get_tensor_by_name("bayesianDecoder/standardDeviation:0")], 
					{tf.get_default_graph().get_tensor_by_name("group"+str(group)+"-encoder/x:0"):tmp["group"+str(group)]
						for group in range(params.nGroups)}), 
				axis=0))
		pos = np.array(pos)

	testOutput = np.array(testOutput)

	fileName = projectPath.folder + '_resultsForRnn_temp'
	if trainLosses==None:
		trainLosses = []
	np.savez(os.path.expanduser(fileName), pos=pos, spd=spd, inferring=testOutput, trainLosses=trainLosses)

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
