import numpy as np
import tensorflow as tf
from fullEncoder import nnUtils
from tqdm import trange



class Trainer():
	def __init__(self, projectPath, params, spikeDetector, device_name="/cpu:0"):
		self.projectPath = projectPath
		self.params = params
		self.device_name = device_name
		self.feat_desc = {
		"pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), 
		"length": tf.io.FixedLenFeature([], tf.int64), 
		"groups": tf.io.VarLenFeature(tf.int64),
		"time": tf.io.FixedLenFeature([], tf.float32)}
		for g in range(self.params.nGroups):
			self.feat_desc.update({"group"+str(g): tf.io.VarLenFeature(tf.float32)})

	def train(self):
		### Training model
		with tf.Graph().as_default():

			print()
			print('TRAINING')

			dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"]).shuffle(self.params.nSteps).repeat()
			dataset = dataset.batch(self.params.batch_size)
			dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals, batched=True))


			#Remove by Pierre 16/02/2021: conversion to 2.0: not needed anymore
			#iter = tf.compat.v1.data.make_initializable_iterator(dataset)
			my_iter = iter(dataset)
			iterators = my_iter.get_next()

			with tf.device(self.device_name):
				spkParserNet = []
				allFeatures = []

				# CNN plus dense on every group indepedently
				for group in range(self.params.nGroups):
					# For each group, Thibault would create a variable scope to keep track
					# of the spike net used for encoding the spike of the specific group.
					with tf.compat.v1.variable_scope("group"+str(group)+"-encoder"):
						x = iterators["group"+str(group)]
						#--> [nbKeptSpikeSequence(time steps where we have more than one spiking),nbChannels,32] tensors
						idMatrix = tf.eye(tf.shape(input=iterators["groups"])[0])
						# What is the role of the completionTensor?
						# The id matrix dimension is the number of different spike sequence encoded inside the spike window
						# Indeed iterators["groups"] is the list of group that were emitted during each spike sequence merged into the spike window

						completionTensor = tf.transpose(a=tf.gather(idMatrix, tf.where(tf.equal(iterators["groups"], group)))[:,0,:], perm=[1,0], name="completion")
						# The completion Matrix, gather the row of the idMatrix, ie the spike sequence corresponding to the group: group

					#Pierre: switched device from cpu to self.device_name
					newSpikeNet = nnUtils.spikeNet(nChannels=self.params.nChannels[group], device=self.device_name, nFeatures=self.params.nFeatures)
					x = newSpikeNet.apply(x) # outputs a [nbTimeWindow,nFeatures=self.params.nFeatures(default 128)] tensor.
					x = tf.matmul(completionTensor, x)
					# Pierre: Multiplying by completionTensor allows to remove the windows where no spikes was observed from this group.
					# But I thought that for iterators["group"+str(group)] this was already the case.


					x = tf.reshape(x, [self.params.batch_size, -1, self.params.nFeatures]) # Reshaping the result of the spike net as batch_size:nbTimeSteps:nFeatures
					if self.params.timeMajor:
						x = tf.transpose(a=x, perm=[1,0,2]) # if timeMajor (the case by default): exchange batch-size and nbTimeSteps
					allFeatures.append(x)
				allFeatures = tf.tuple(tensors=allFeatures) #synchronizes the computation of all features (like a join)
				# The concatenation is made over axis 2, which is the Feature axis
				allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

				# LSTM on the concatenated outputs of previous graphs
				# Modif by Pierre 12/02/2021: migration to 2.0
				lstms = [nnUtils.layerLSTM(self.params.lstmSize, dropout=self.params.lstmDropout) for _ in
						 range(self.params.lstmLayers)]
				stacked_lstm = tf.keras.layers.StackedRNNCells(lstms)
				lstm = tf.keras.layers.RNN(stacked_lstm, time_major=self.params.timeMajor, return_state= True)
				outputs, finalState = lstm(allFeatures)


			# dense to extract regression on output and loss
			denseOutput = tf.keras.layers.Dense(self.params.dim_output, activation = None)
			denseLoss1  = tf.keras.layers.Dense(self.params.lstmSize, activation = tf.nn.relu)
			denseLoss2  = tf.keras.layers.Dense(1, activation = self.params.lossActivation)
			#Modif by Pierre 12/02/2021: removed name

			# Pierre: the question is: do we have to use last_relevant?
			output = nnUtils.last_relevant(outputs, iterators["length"], timeMajor=self.params.timeMajor)
			outputLoss = denseLoss2(denseLoss1(output))[:,0]
			outputPos = denseOutput(output)

			lossPos =  tf.compat.v1.losses.mean_squared_error(outputPos, iterators["pos"], reduction=tf.compat.v1.losses.Reduction.NONE)
			lossPos =  tf.reduce_mean(input_tensor=lossPos, axis=1)
			lossLoss = tf.compat.v1.losses.mean_squared_error(outputLoss, lossPos)
			lossPos  = tf.reduce_mean(input_tensor=lossPos)

			optimizers = []
			for lr in range(len(self.params.learningRates)):
				optimizers.append(tf.compat.v1.train.RMSPropOptimizer(self.params.learningRates[lr]).minimize(lossPos + lossLoss))
			saver = tf.compat.v1.train.Saver()



			### Training and testing
			trainLosses = []
			with tf.compat.v1.Session() as sess:

				# initialize variables and input framework
				sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))
				sess.run(iter.initializer)

				### training
				epoch_loss = 0
				epoch_loss2 = 0
				loopSize = 50
				t = trange(self.params.nSteps, desc='Bar desc', leave=True)
				for i in t:

					for lr in range(len(self.params.learningRates)):
						if (i < (lr+1) * self.params.nSteps / len(self.params.learningRates)) and (i >= lr * self.params.nSteps / len(self.params.learningRates)):
							_, c, c2 = sess.run([optimizers[lr], lossPos, lossLoss])
							break
						if lr==len(self.params.learningRates)-1:
							print('not run:',i)
					t.set_description("loss: %f" % c)
					t.refresh()
					epoch_loss += c
					epoch_loss2 += c2
					
					if i%loopSize==0 and (i != 0):
						trainLosses.append(np.array([epoch_loss/loopSize, epoch_loss2/loopSize]))
						epoch_loss=0
						epoch_loss2=0

				saver.save(sess, self.projectPath.graph)
		self.convert()
		return np.array(trainLosses)






	def convert(self):
		### Back compatibility converting before inferring
		variables = []
		print("Cleaning graph.")
		with tf.Graph().as_default(), tf.device("/cpu:0"):


			# one CNN network per group of electrode
			embeddings = []
			for group in range(self.params.nGroups):
				with tf.compat.v1.variable_scope("group"+str(group)+"-encoder"):
					x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.params.nChannels[group], 32], name="x")
					realSpikes = tf.math.logical_not(tf.equal(tf.reduce_sum(input_tensor=x, axis=[1,2]), tf.constant(0.)))
					nSpikesTot = tf.shape(input=x)[0]; idMatrix = tf.eye(nSpikesTot)
					completionTensor = tf.transpose(a=tf.gather(idMatrix, tf.compat.v1.where(realSpikes))[:,0,:], perm=[1,0], name="completion")
					x = tf.boolean_mask(tensor=x, mask=realSpikes)
				newSpikeNet = nnUtils.spikeNet(nChannels=self.params.nChannels[group], device="/cpu:0", nFeatures=self.params.nFeatures)
				x = newSpikeNet.apply(x)
				x = tf.matmul(completionTensor, x)

				embeddings.append(x)
				variables += newSpikeNet.variables()
			fullEmbedding = tf.concat(embeddings, axis=1, name="concat2")

			
			# LSTM on concatenated outputs
			if self.device_name=="/gpu:0":
				with tf.compat.v1.variable_scope("cudnn_lstm"):
					lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
						[tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.params.lstmSize) for _ in range(self.params.lstmLayers)])
			else:
				lstm = [nnUtils.layerLSTM(self.params.lstmSize, dropout=self.params.lstmDropout) for _ in range(self.params.lstmLayers)]
				lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm)
				outputs, finalState = tf.compat.v1.nn.dynamic_rnn(
					lstm, 
					tf.expand_dims(fullEmbedding, axis=1), 
					dtype=tf.float32, 
					time_major=self.params.timeMajor)
				variables += lstm.variables

			# Final position decoder
			output = tf.cond(pred=tf.shape(input=outputs)[0]>0, true_fn=lambda: outputs[-1,:,:], false_fn=lambda: outputs)
			denseOutput = tf.compat.v1.layers.Dense(self.params.dim_output, activation = None, name="pos")
			denseLoss1  = tf.compat.v1.layers.Dense(self.params.lstmSize, activation = tf.nn.relu, name="loss1")
			denseLoss2  = tf.compat.v1.layers.Dense(1, activation = self.params.lossActivation, name="loss2")

			x = denseOutput(tf.reshape(output, [-1,self.params.lstmSize]))
			y = denseLoss2(denseLoss1(tf.reshape(output, [-1,self.params.lstmSize])))
			variables += denseOutput.variables
			variables += denseLoss1.variables
			variables += denseLoss2.variables

			with tf.compat.v1.variable_scope("bayesianDecoder"):
				position = tf.identity(
					tf.cond(
						pred=tf.shape(input=outputs)[0]>0, 
						true_fn=lambda: tf.reshape(x, [self.params.dim_output]), 
						false_fn=lambda: tf.constant(np.zeros([self.params.dim_output]), dtype=tf.float32)), name="positionGuessed")
				loss     = tf.identity(
					tf.cond(
						pred=tf.shape(input=outputs)[0]>0, 
						true_fn=lambda: tf.reshape(y, [1]), 
						false_fn=lambda: tf.constant([0], dtype=tf.float32)), name="standardDeviation")
				fakeProba= tf.constant(np.ones([50,50]), dtype=tf.float32, name="positionProba")		
			
			subGraphToRestore = tf.compat.v1.train.Saver({v.op.name: v for v in variables})

			### Converting
			graphToSave = tf.compat.v1.train.Saver()
			with tf.compat.v1.Session() as sess:
				subGraphToRestore.restore(sess, self.projectPath.graph)
				graphToSave.save(sess, self.projectPath.graph)










	def test(self):
		### Loading and inferring
		print()
		print("INFERRING")

		tf.contrib.rnn
		with tf.Graph().as_default(), tf.device("/cpu:0"):

			dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["test"])
			cnt     = dataset.batch(1).repeat(1).reduce(np.int64(0), lambda x, _: x + 1)
			dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals))
			iter    = tf.compat.v1.data.make_initializable_iterator(dataset)
			spikes  = iter.get_next()
			for group in range(self.params.nGroups):
				idMatrix = tf.eye(tf.shape(input=spikes["groups"])[0])
				completionTensor = tf.transpose(a=tf.gather(idMatrix, tf.compat.v1.where(tf.equal(spikes["groups"], group)))[:,0,:], perm=[1,0], name="completion")
				spikes["group"+str(group)] = tf.tensordot(completionTensor, spikes["group"+str(group)], axes=[[1],[0]])


			saver = tf.compat.v1.train.import_meta_graph(self.projectPath.graphMeta)


			with tf.compat.v1.Session() as sess:
				saver.restore(sess, self.projectPath.graph)

				pos = []
				inferring = []
				probaMaps = []
				times = []
				sess.run(iter.initializer)
				for b in trange(cnt.eval()):
					tmp = sess.run(spikes)
					pos.append(tmp["pos"])
					times.append(tmp["time"])
					temp = sess.run(
							[tf.compat.v1.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0"), 
							 tf.compat.v1.get_default_graph().get_tensor_by_name("bayesianDecoder/positionGuessed:0"), 
							 tf.compat.v1.get_default_graph().get_tensor_by_name("bayesianDecoder/standardDeviation:0")], 
							{tf.compat.v1.get_default_graph().get_tensor_by_name("group"+str(group)+"-encoder/x:0"):tmp["group"+str(group)]
								for group in range(self.params.nGroups)}) 
					inferring.append(np.concatenate([temp[1],temp[2]], axis=0))
					probaMaps.append(temp[0])

				pos = np.array(pos)

		return {"inferring":np.array(inferring), "pos":pos, "probaMaps":np.array(probaMaps), "times":times}
