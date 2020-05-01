import numpy as np
import tensorflow as tf
from fullEncoder import nnUtils
from tqdm import trange



class Trainer():
	def __init__(self, projectPath, params, device_name="/cpu:0"):
		self.projectPath = projectPath
		self.params = params
		self.device_name = device_name
		self.feat_desc = {"pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), "length": tf.io.FixedLenFeature([], tf.int64), "groups": tf.io.VarLenFeature(tf.int64)}
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
			iter = dataset.make_initializable_iterator()
			iterators = iter.get_next()

			with tf.device(self.device_name):
				spkParserNet = []
				allFeatures = []

				# CNN plus dense on every group indepedently
				for group in range(self.params.nGroups):
					with tf.variable_scope("group"+str(group)+"-encoder"):
						x = iterators["group"+str(group)]
						idMatrix = tf.eye(tf.shape(iterators["groups"])[0])
						completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(tf.equal(iterators["groups"], group)))[:,0,:], [1,0], name="completion")

					newSpikeNet = nnUtils.spikeNet(nChannels=self.params.nChannels[group], device="/cpu:0", nFeatures=self.params.nFeatures)
					x = newSpikeNet.apply(x)
					x = tf.matmul(completionTensor, x)
					x = tf.reshape(x, [self.params.batch_size, -1, self.params.nFeatures])
					if self.params.timeMajor:
						x = tf.transpose(x, [1,0,2])
					allFeatures.append(x)
				allFeatures = tf.tuple(allFeatures)
				allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

				# LSTM on the concatenated outputs of previous graphs
				if self.device_name=="/gpu:0":
					lstm = tf.contrib.cudnn_rnn.CudnnLSTM(self.params.lstmLayers, self.params.lstmSize, dropout=self.params.lstmDropout)
					outputs, finalState = lstm(allFeatures, training=True)
				else:
					lstm = [nnUtils.layerLSTM(self.params.lstmSize, dropout=self.params.lstmDropout) for _ in range(self.params.lstmLayers)]
					lstm = tf.nn.rnn_cell.MultiRNNCell(lstm)
					outputs, finalState = tf.nn.dynamic_rnn(
						lstm, 
						allFeatures, 
						dtype=tf.float32, 
						time_major=self.params.timeMajor,
						sequence_length=iterators["length"])

			# dense to extract regression on output and loss
			denseOutput = tf.layers.Dense(self.params.dim_output, activation = None, name="pos")
			denseLoss1  = tf.layers.Dense(self.params.lstmSize, activation = tf.nn.relu, name="loss1")
			denseLoss2  = tf.layers.Dense(1, activation = self.params.lossActivation, name="loss2")

			output = nnUtils.last_relevant(outputs, iterators["length"], timeMajor=self.params.timeMajor)
			outputLoss = denseLoss2(denseLoss1(output))[:,0]
			outputPos = denseOutput(output)

			lossPos =  tf.losses.mean_squared_error(outputPos, iterators["pos"], reduction=tf.losses.Reduction.NONE)
			lossPos =  tf.reduce_mean(lossPos, axis=1)
			lossLoss = tf.losses.mean_squared_error(outputLoss, lossPos)
			lossPos  = tf.reduce_mean(lossPos)

			optimizers = []
			for lr in range(len(self.params.learningRates)):
				optimizers.append(tf.train.RMSPropOptimizer(self.params.learningRates[lr]).minimize(lossPos + lossLoss))
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
		return np.array(trainLosses)






	def convert(self):
		### Back compatibility converting before inferring
		variables = []
		print("Cleaning graph.")
		with tf.Graph().as_default(), tf.device("/cpu:0"):


			# one CNN network per group of electrode
			embeddings = []
			for group in range(self.params.nGroups):
				with tf.variable_scope("group"+str(group)+"-encoder"):
					x = tf.placeholder(tf.float32, shape=[None, self.params.nChannels[group], 32], name="x")
					realSpikes = tf.math.logical_not(tf.equal(tf.reduce_sum(x, [1,2]), tf.constant(0.)))
					nSpikesTot = tf.shape(x)[0]; idMatrix = tf.eye(nSpikesTot)
					completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(realSpikes))[:,0,:], [1,0], name="completion")
					x = tf.boolean_mask(x, realSpikes)
				newSpikeNet = nnUtils.spikeNet(nChannels=self.params.nChannels[group], device="/cpu:0", nFeatures=self.params.nFeatures)
				x = newSpikeNet.apply(x)
				x = tf.matmul(completionTensor, x)

				embeddings.append(x)
				variables += newSpikeNet.variables()
			fullEmbedding = tf.concat(embeddings, axis=1, name="concat2")

			
			# LSTM on concatenated outputs
			if self.device_name=="/gpu:0":
				with tf.variable_scope("cudnn_lstm"):
					lstm = tf.nn.rnn_cell.MultiRNNCell(
						[tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.params.lstmSize) for _ in range(self.params.lstmLayers)])
			else:
				lstm = [nnUtils.layerLSTM(self.params.lstmSize, dropout=self.params.lstmDropout) for _ in range(self.params.lstmLayers)]
				lstm = tf.nn.rnn_cell.MultiRNNCell(lstm)
				outputs, finalState = tf.nn.dynamic_rnn(
					lstm, 
					tf.expand_dims(fullEmbedding, axis=1), 
					dtype=tf.float32, 
					time_major=self.params.timeMajor)
				variables += lstm.variables

			# Final position decoder
			output = tf.cond(tf.shape(outputs)[0]>0, lambda: outputs[-1,:,:], lambda: outputs)
			denseOutput = tf.layers.Dense(self.params.dim_output, activation = None, name="pos")
			denseLoss1  = tf.layers.Dense(self.params.lstmSize, activation = tf.nn.relu, name="loss1")
			denseLoss2  = tf.layers.Dense(1, activation = self.params.lossActivation, name="loss2")

			x = denseOutput(tf.reshape(output, [-1,self.params.lstmSize]))
			y = denseLoss2(denseLoss1(tf.reshape(output, [-1,self.params.lstmSize])))
			variables += denseOutput.variables
			variables += denseLoss1.variables
			variables += denseLoss2.variables

			with tf.variable_scope("bayesianDecoder"):
				position = tf.identity(
					tf.cond(
						tf.shape(outputs)[0]>0, 
						lambda: tf.reshape(x, [self.params.dim_output]), 
						lambda: tf.constant(np.zeros([self.params.dim_output]), dtype=tf.float32)), name="positionGuessed")
				loss     = tf.identity(
					tf.cond(
						tf.shape(outputs)[0]>0, 
						lambda: tf.reshape(y, [1]), 
						lambda: tf.constant([0], dtype=tf.float32)), name="standardDeviation")
				fakeProba= tf.constant(np.zeros([45,45]), dtype=tf.float32, name="positionProba")		
			
			subGraphToRestore = tf.train.Saver({v.op.name: v for v in variables})

			### Converting
			graphToSave = tf.train.Saver()
			with tf.Session() as sess:
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
			iter    = dataset.make_initializable_iterator()
			spikes  = iter.get_next()
			for group in range(self.params.nGroups):
				idMatrix = tf.eye(tf.shape(spikes["groups"])[0])
				completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(tf.equal(spikes["groups"], group)))[:,0,:], [1,0], name="completion")
				spikes["group"+str(group)] = tf.tensordot(completionTensor, spikes["group"+str(group)], axes=[[1],[0]])


			saver = tf.train.import_meta_graph(self.projectPath.graphMeta)


			with tf.Session() as sess:
				saver.restore(sess, self.projectPath.graph)

				pos = []
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
								for group in range(self.params.nGroups)}), 
						axis=0))
				pos = np.array(pos)

		return {"inferring":np.array(testOutput), "pos":pos}