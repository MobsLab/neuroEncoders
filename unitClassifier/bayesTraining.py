import sys
from tqdm import trange
from queue import Empty

import numpy as np
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

from fullEncoder import nnUtils
from unitClassifier import bayesUtils



def encoder(input, nClusters, nChannels, **kwargs):
	size = kwargs.get('size', 512)

	convLayer1 = tf.layers.Conv2D(8, [2,3], padding='SAME')
	convLayer2 = tf.layers.Conv2D(16, [2,3], padding='SAME')
	convLayer3 = tf.layers.Conv2D(32, [2,3], padding='SAME')

	x = tf.expand_dims(input, axis=3)
	x = convLayer1(x)
	x = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')(x)
	x = convLayer2(x)
	x = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')(x)
	x = convLayer3(x)
	x = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')(x)

	x = tf.reshape(x, [-1, nChannels*4*32])
	x = tf.layers.Dense(size, activation=tf.nn.relu)(x)
	x = tf.layers.Dropout(0.5)(x)
	x = tf.layers.Dense(size, activation=tf.nn.relu)(x)
	x = tf.layers.Dense(nClusters, activation=None)(x)
	result = x

	return result, [convLayer1, convLayer2, convLayer3]


def computeKernelDensityEstimator(inputQueue, outputQueue, OccInv, params, edges):
	while True:
		try:
			idx, pos = inputQueue.get(block=True, timeout=0.1)
			nItems = pos.shape[0]
			if nItems==0:
				outputQueue.put((idx, np.ones([50,50])))
				continue

			if len(idx)==1:
				name = "marginal rate function "+str(idx[0])
			else:
				name = "rate function "+str(idx[0])+"-"+str(idx[1])
			sys.stdout.write("computing "+name+" with "+str(nItems)+" items         ")
			sys.stdout.write('\r')
			sys.stdout.flush()

			_, _, F = bayesUtils.kde2D(
				pos[:,0], pos[:,1], 
				params.bandwidth, kernel=params.kernel, edges=edges)
			F[F==0] = np.min(F[F!=0])
			F       = F/np.sum(F)
			F       = nItems*np.multiply(F, OccInv)/params.learningTime
			outputQueue.put((idx, F))
		except Empty:
			return

def computingDone(MRF, RF):
	for m in MRF:
		if m==[]:
			return False
	for t in RF:
		for r in t:
			if r==[]:
				return False
	return True


class Trainer():
	def __init__(self, projectPath, params, spikeDetector, device_name="/cpu:0"):
		self.projectPath = projectPath
		self.params = params
		self.device_name = device_name
		self.clusterPositions = [np.load(projectPath.pos(g)) for g in range(self.params.nGroups)]
		self.params.nClusters = [len(list(self.clusterPositions[g].keys())) for g in range(params.nGroups)]
		self.trainingPositions = spikeDetector.trainingPositions()
		self.feat_desc = {"pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), "length": tf.io.FixedLenFeature([], tf.int64), "groups": tf.io.VarLenFeature(tf.int64)}
		for g in range(self.params.nGroups):
			self.feat_desc.update({"group"+str(g): tf.io.VarLenFeature(tf.float32)})

	def computeSpikeDensities(self):
		print("Computing kernel densty estimators for bayes model")
		xEdges, yEdges, Occupation = bayesUtils.kde2D(
			self.trainingPositions[:,0], self.trainingPositions[:,1], 
			self.params.bandwidth, kernel=self.params.kernel)
		Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

		mask = Occupation > (np.max(Occupation)/self.params.masking)
		Occupation_inverse = 1/Occupation
		Occupation_inverse[Occupation_inverse==np.inf] = 0
		Occupation_inverse = np.multiply(Occupation_inverse, mask)

		self.MRF = [[] for g in range(self.params.nGroups)]
		self.RF = [[[] for c in range(self.params.nClusters[g])] for g in range(self.params.nGroups)]
		inputQueue = mp.Queue()
		outputQueue = mp.Queue()
		for g in range(self.params.nGroups):
			inputQueue.put((
				[g], 
				np.concatenate([self.clusterPositions[g]["clu"+str(c)].reshape([-1,2]) for c in range(self.params.nClusters[g])]) ))
			for c in range(self.params.nClusters[g]):
				inputQueue.put((
					[g,c],
					self.clusterPositions[g]["clu"+str(c)].reshape([-1,2]) ))

		processes = []
		for _ in range(mp.cpu_count()):
			p = mp.Process(target=computeKernelDensityEstimator, args=(inputQueue, outputQueue, Occupation_inverse, self.params, [xEdges, yEdges]))
			p.daemon = True
			p.start()
			processes.append(p)

		while not computingDone(self.MRF, self.RF):
			idx, F = outputQueue.get(block=True)
			if len(idx) == 1:
				self.MRF[idx[0]] = F
			else:
				self.RF[idx[0]][idx[1]] = F
			
		[p.join() for p in processes]
		inputQueue.close()
		outputQueue.close()

		sys.stdout.write("densities computed"+" "*50)
		sys.stdout.write('\r')
		sys.stdout.flush()
		print()
		return xEdges, yEdges, mask








	def train(self):
		print()
		print('TRAINING')
		xEdges, yEdges, mask = self.computeSpikeDensities()

		# Buid and train deep learning network
		efficiencies = []
		convolutions = []
		cnt          = []

		sumConstantTerms = np.sum(self.MRF, axis=0)
		allRateMaps = [np.log(self.RF[g][c] + np.min(self.RF[g][c][self.RF[g][c]!=0])) 
						for g in range(self.params.nGroups)
						for c in range(self.params.nClusters[g])]
		allRateMaps = np.array(allRateMaps)


		# # yTensors = []
		# # probasTensors = []
		# for group in range(self.params.nGroups):
		# 	MOBSgraph = tf.Graph()
		# 	with MOBSgraph.as_default():
		# 		with tf.variable_scope("group"+str(group)+"-encoder"):
		# 			feat_desc = {
		# 				"clu": tf.io.FixedLenFeature([], tf.int64),
		# 				"spike": tf.io.FixedLenFeature([self.params.nChannels[group], 32], tf.float32)
		# 			}
		# 			dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"]+"."+str(group))
		# 			cnt.append( dataset.batch(1).repeat(1).reduce(np.int64(0), lambda x, _: x + 1) )
		# 			dataset = dataset.shuffle(10000).batch(self.params.batch_size).repeat()
		# 			dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.params, feat_desc, *vals, batched=True))
		# 			iter    = dataset.make_one_shot_iterator()
		# 			spikes  = iter.get_next()

		# 			# x                   = tf.placeholder(tf.float32, shape=[None, self.params.nChannels[group], 32],      name='x')
		# 			# y                   = tf.placeholder(tf.float32, shape=[None, self.params.nClusters[group]],          name='y')
		# 			# ySparse             = tf.placeholder(tf.int32,   shape=[None],                                   name='ySparse')
		# 			x                   = tf.identity(spikes["spike"], name='x')
		# 			ySparse             = tf.identity(spikes["clu"],   name="ySparse")
		# 			# realSpikes          = tf.math.logical_not(tf.equal(tf.reduce_sum(x, [1,2]), tf.constant(0.)))
		# 			# x                   = tf.identity(tf.boolean_mask(x, realSpikes), name='onlySpikes')

		# 		spikeEncoder, ops = encoder(x, self.params.nClusters[group], self.params.nChannels[group], size=200)
		# 		convolutions.append(ops)

		# 		with tf.variable_scope("group"+str(group)+"-evaluator"):

		# 			probas              = tf.nn.softmax(spikeEncoder, name='probas')
		# 			sumProbas           = tf.reduce_sum(probas, axis=0, name='sumProbas')
		# 			# yTensors.append(tf.reduce_sum(y, axis=0))

		# 			guesses             = tf.argmax(spikeEncoder,1, name='guesses')
		# 			good_guesses        = tf.equal(ySparse, guesses)
		# 			accuracy            = tf.reduce_mean(tf.cast(good_guesses, tf.float32), name='accuracy')
		# 			# confusion_matrix    = tf.confusion_matrix(tf.argmax(y,1), guesses, name='confusion')

		# 			### Classic cross entropy training
		# 			cross_entropy       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ySparse, logits=spikeEncoder))
		# 			crossTrain          = tf.train.AdamOptimizer(0.00004).minimize(cross_entropy, name='trainer')

		# 		saver = tf.train.Saver()
		# 		with tf.Session() as sess:
		# 			print('Learning clusters of group '+str(group+1))
		# 			acc = 0
		# 			n = cnt[group].eval()
		# 			t = trange(self.params.nEpochs*(n//self.params.batch_size))
		# 			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
		# 			for i in t:
		# 				_, a = sess.run([crossTrain, accuracy])
		# 				acc += a
		# 				if i%50==0:
		# 					t.set_description("acc: %f" % (acc/50))
		# 					t.refresh()
		# 					acc = 0
		# 			saver.save(sess, self.projectPath.graph+"."+str(group))
		
		with tf.Graph().as_default():
			savers = []
			for g in range(self.params.nGroups):
				with tf.variable_scope("group"+str(g)+"-encoder"):
					savers.append( 
						tf.train.import_meta_graph(
							self.projectPath.graph+"."+str(g)+".meta",
							input_map={"group"+str(g)+"-encoder/x:0": tf.placeholder(tf.float32, shape=[None, self.params.nChannels[g], 32], name="x")}) )
			probasTensors = [tf.get_default_graph().get_tensor_by_name("group"+str(g)+"-encoder/group"+str(g)+"-evaluator/sumProbas:0") \
				for g in range(self.params.nGroups)]
			with tf.variable_scope("bayesianDecoder"):

				# binTime                     = tf.placeholder(tf.float32, shape=[1], name='binTime')
				binTime                     = tf.constant([self.params.windowLength], dtype=tf.float32, shape=[1])
				# allProbas                   = tf.reshape(tf.concat(yTensors, 0), [1, Data['nClusters']], name='allProbas');
				allProbas                   = tf.reshape(tf.concat(probasTensors, 0), [1, sum(self.params.nClusters)], name='allProbas');

				# Place map stats
				occMask                     = tf.constant(mask,                      dtype=tf.float64, shape=[50,50])
				constantTerm                = tf.constant(sumConstantTerms,          dtype=tf.float32, shape=[50,50])
				occMask_flat                = tf.reshape(occMask, [50*50], name='maskFlat')
				constantTerm_flat           = tf.reshape(constantTerm, [50*50])

				rateMaps                    = tf.constant(allRateMaps,               dtype=tf.float32, shape=[sum(self.params.nClusters), 50,50], name='rateMaps')
				rateMaps_flat               = tf.reshape(rateMaps, [sum(self.params.nClusters), 50*50])
				spikesWeight                = tf.matmul(allProbas, rateMaps_flat, name='spikesWeight')

				allWeights                  = tf.cast( spikesWeight - binTime * constantTerm_flat, tf.float64, name='allWeights')
				allWeights                  = tf.multiply(allWeights, occMask_flat)
				allWeights_reduced          = tf.identity(allWeights - tf.reduce_mean(allWeights), name='allWeightsReduced')

				termPoisson                 = tf.exp(allWeights_reduced, name='termPoisson')
				positionProba_flat          = tf.multiply( termPoisson, occMask_flat, name='positionProbaFlat')
				positionProba               = tf.reshape(positionProba_flat / tf.reduce_sum(positionProba_flat), [50,50], name='positionProba')

				xBins                       = tf.constant(np.array(xEdges[:,0]), shape=[50], name='xBins')
				yBins                       = tf.constant(np.array(yEdges[0,:]), shape=[50], name='yBins')
				xProba                      = tf.reduce_sum(positionProba, axis=1, name='xProba')
				yProba                      = tf.reduce_sum(positionProba, axis=0, name='yProba')
				xGuessed                    = tf.reduce_sum(tf.multiply(xProba, xBins)) / tf.reduce_sum(xProba)
				yGuessed                    = tf.reduce_sum(tf.multiply(yProba, yBins)) / tf.reduce_sum(yProba)
				xStd                        = tf.sqrt(tf.reduce_sum(xProba*tf.square(xBins-xGuessed)))
				yStd                        = tf.sqrt(tf.reduce_sum(yProba*tf.square(yBins-yGuessed)))

				positionGuessed             = tf.stack([xGuessed, yGuessed], name='positionGuessed')
				# standardDeviation           = tf.stack([xStd, yStd], name='standardDeviation')
				standardDeviation           = tf.sqrt([xStd*xStd + yStd*yStd], name='standardDeviation')
			

			print('Tensorflow graph has been built and is ready to train.')

			### Train
			saver = tf.train.Saver()
			with tf.Session() as sess:
				
				for g in range(self.params.nGroups):
					savers[g].restore(sess, self.projectPath.graph+'.'+str(g))
				saver.save(sess, self.projectPath.graph)
		return 0




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
				inferring = []
				probaMaps = []
				sess.run(iter.initializer)
				for b in trange(cnt.eval()):
					tmp = sess.run(spikes)
					pos.append(tmp["pos"])
					temp = sess.run(
							[tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0"), 
							 tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionGuessed:0"), 
							 tf.get_default_graph().get_tensor_by_name("bayesianDecoder/standardDeviation:0")], 
							{tf.get_default_graph().get_tensor_by_name("group"+str(group)+"-encoder/x:0"):tmp["group"+str(group)]
								for group in range(self.params.nGroups)}) 
					inferring.append(np.concatenate([temp[1],temp[2]], axis=0))
					probaMaps.append(temp[0])
					# plt.imshow(temp[0])
					# plt.show()
						
				pos = np.array(pos)

		return {"inferring":np.array(inferring), "pos":pos, "probaMaps":np.array(probaMaps)}