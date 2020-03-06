import sys
import datetime
import math
import random
import numpy as np
import tensorflow as tf
from datetime import datetime







def next_batch(num, data, labels):
	""" Generates a random batch of matching data and labels """
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)
def shuffle(data, labels):
	return next_batch(len(data), data, labels)











def encoder(input, nClusters, nChannels, **kwargs):
	size = kwargs.get('size', 512)

	# towards tensorflow 2.x interface
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


	# Should return a tensor
	return result, [convLayer1, convLayer2, convLayer3]




def build_position_decoder(Data, results_dir, nSteps):
	"""Trains one artificial neural network to guess position proba from spikes"""

	print('\nENCODING GRAPH\n')


	efficiencies = []
	convolutions = []
	n_tetrodes = Data['nGroups']

	sumConstantTerms = np.sum(Data['Marginal_rate_functions'], axis=0)
	allRateMaps = [np.log(Data['Rate_functions'][group][clu] + np.min(Data['Rate_functions'][group][clu][Data['Rate_functions'][group][clu]!=0])) 
					for group in range(n_tetrodes)
					for clu in range(Data['clustersPerGroup'][group])]
	allRateMaps = np.array(allRateMaps)


	##### BUILDING THE MODEL
	MOBSgraph = tf.Graph()
	with MOBSgraph.as_default():

		yTensors = []
		probasTensors = []
		for tetrode in range(n_tetrodes):
			placeMapsStd = []

			### standard deviation and weights to compute weighted loss
			for label in range(Data['clustersPerGroup'][tetrode]):
				temp = Data['Rate_functions'][tetrode][label] / Data['Rate_functions'][tetrode][label].sum()
				placeMapsStd.append( np.sqrt(np.power(temp.sum(axis=0).std(), 2) + np.power(temp.sum(axis=1).std(), 2)) )
			weights = np.array(placeMapsStd)
			weights -= weights.min() ; weights /= weights.max()
			weights = 1 - weights
			weights += 1/len(weights) ; weights /= weights.sum()

			with tf.variable_scope("group"+str(tetrode)+"-encoder"):

				x                   = tf.placeholder(tf.float32, shape=[None, Data['channelsPerGroup'][tetrode], 32],      name='x')
				y                   = tf.placeholder(tf.float32, shape=[None, Data['clustersPerGroup'][tetrode]],          name='y')
				ySparse             = tf.placeholder(tf.int32,   shape=[None],                                             name='ySparse')
				realSpikes          = tf.math.logical_not(tf.equal(tf.reduce_sum(x, [1,2]), tf.constant(0.)))
				x                   = tf.identity(tf.boolean_mask(x, realSpikes), name='onlySpikes')

			spikeEncoder, ops = encoder(x,Data['clustersPerGroup'][tetrode], Data['channelsPerGroup'][tetrode], size=200)
			convolutions.append(ops)

			with tf.variable_scope("group"+str(tetrode)+"-evaluator"):

				probas              = tf.nn.softmax(spikeEncoder, name='probas')
				probasTensors.append( tf.reduce_sum(probas, axis=0, name='sumProbas'))
				yTensors.append(tf.reduce_sum(y, axis=0))

				guesses             = tf.argmax(spikeEncoder,1, name='guesses')
				good_guesses        = tf.equal(tf.argmax(y,1), guesses)
				accuracy            = tf.reduce_mean(tf.cast(good_guesses, tf.float32), name='accuracy')
				confusion_matrix    = tf.confusion_matrix(tf.argmax(y,1), guesses, name='confusion')
				
				### MSFE loss
				# numLabels           = tf.reduce_sum(y, axis=0)
				# squaredError        = ((probas - y)**2) / 2
				# squaredError        = tf.reduce_sum(squaredError, axis=1)
				# coeffs              = tf.gather(numLabels, tf.argmax(y,1))
				# falseError          = squaredError / coeffs
				# loss                = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y,1),0),     tf.float32) * falseError) ** 2
				# for label in range(1,Data['clustersPerGroup'][tetrode]):
				# 	loss           += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y,1),label), tf.float32) * falseError) ** 2
				# crossTrain          = tf.train.AdamOptimizer(0.00004).minimize(loss, name='trainer')

				### weighted MSE loss
				# loss                = tf.losses.mean_squared_error(y, probas, reduction=tf.losses.Reduction.NONE)
				# loss                = tf.reduce_mean(loss, axis=0)
				# weighted_loss       = tf.losses.compute_weighted_loss(loss, weights=weights)
				# crossTrain          = tf.train.AdamOptimizer(0.00004).minimize(weighted_loss, name='trainer')

				### Classic cross entropy training
				cross_entropy       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ySparse, logits=spikeEncoder))
				crossTrain          = tf.train.AdamOptimizer(0.00004).minimize(cross_entropy, name='trainer')

		

		with tf.variable_scope("bayesianDecoder"):

			binTime                     = tf.placeholder(tf.float32, shape=[1], name='binTime')
			# allProbas                   = tf.reshape(tf.concat(yTensors, 0), [1, Data['nClusters']], name='allProbas');
			allProbas                   = tf.reshape(tf.concat(probasTensors, 0), [1, Data['nClusters']], name='allProbas');

			# Place map stats
			occMask                     = tf.constant(Data['Mask'],              dtype=tf.float64, shape=[45,45])
			constantTerm                = tf.constant(sumConstantTerms,          dtype=tf.float32, shape=[45,45])
			occMask_flat                = tf.reshape(occMask, [45*45])
			constantTerm_flat           = tf.reshape(constantTerm, [45*45])

			rateMaps                    = tf.constant(allRateMaps,               dtype=tf.float32, shape=[Data['nClusters'], 45,45], name='rateMaps')
			rateMaps_flat               = tf.reshape(rateMaps, [Data['nClusters'], 45*45])
			spikesWeight                = tf.matmul(allProbas, rateMaps_flat)

			allWeights                  = tf.cast( spikesWeight - binTime * constantTerm_flat, tf.float64 )
			allWeights_reduced          = allWeights - tf.reduce_mean(allWeights)

			positionProba_flat          = tf.multiply( tf.exp(allWeights_reduced), occMask_flat )
			positionProba               = tf.reshape(positionProba_flat / tf.reduce_sum(positionProba_flat), [45,45], name='positionProba')

			xBins                       = tf.constant(np.array(Data['Bins'][0]), shape=[45], name='xBins')
			yBins                       = tf.constant(np.array(Data['Bins'][1]), shape=[45], name='yBins')
			xProba                      = tf.reduce_sum(positionProba, axis=1, name='xProba')
			yProba                      = tf.reduce_sum(positionProba, axis=0, name='yProba')
			xGuessed                    = tf.reduce_sum(tf.multiply(xProba, xBins)) / tf.reduce_sum(xProba)
			yGuessed                    = tf.reduce_sum(tf.multiply(yProba, yBins)) / tf.reduce_sum(yProba)
			xStd                        = tf.sqrt(tf.reduce_sum(xProba*tf.square(xBins-xGuessed)))
			yStd                        = tf.sqrt(tf.reduce_sum(yProba*tf.square(yBins-yGuessed)))

			positionGuessed             = tf.stack([xGuessed, yGuessed], name='positionGuessed')
			standardDeviation           = tf.stack([xStd, yStd], name='standardDeviation')

		print('Tensorflow graph has been built and is ready to train.')



		### Train
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

			for tetrode in range(n_tetrodes):
				print('Learning clusters of group '+str(tetrode+1))

				# start convolutions from weights learned in previous group
				if tetrode > 0:
					for op in range(len(convolutions[tetrode])):
						convolutions[tetrode][op].set_weights(convolutions[tetrode-1][op].get_weights())

				i=0
				for i in range(nSteps+1):
					batch = next_batch(80, Data['spikes_train'][tetrode], Data['labels_train'][tetrode])
					if i%50 == 0:
						curr_eval = sess.run([MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-evaluator/accuracy:0')], 
												{MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/x:0'): batch[0], 
												MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/y:0'): batch[1]})
						sys.stdout.write('[%-30s] step : %d/%d, efficiency : %g' % ('='*(i*30//nSteps),i,nSteps,curr_eval[0]))
						sys.stdout.write('\r')
						sys.stdout.flush()

					# training step
					MOBSgraph.get_operation_by_name('group'+str(tetrode)+'-evaluator/trainer').run(
									{MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/x:0'): batch[0], 
									# MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/y:0'): batch[1]}) 
									MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/ySparse:0'): np.argmax(batch[1],axis=1)}) 

				final_eval, confusion = sess.run([MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-evaluator/accuracy:0'), 
												  MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-evaluator/confusion/SparseTensorDenseAdd:0')], 
												{MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/x:0'): Data['spikes_test'][tetrode], 
												MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/y:0'): Data['labels_test'][tetrode]}) 
				efficiencies.append(final_eval)
				print('\nglobal efficiency : ', efficiencies[-1])
				print('confusion : ')
				print(confusion)
				print()
			

			saver.save(sess, results_dir + 'mobsGraph')
		

	return efficiencies

















def decode_position(Data, results_dir, start_time, stop_time, bin_time):

	print('\nDECODING\n')

	n_tetrodes = Data['nGroups']

	decodedPositions = []
	truePositions = [] ; truePositions.append([0.,0.])
	nSpikes = []

	feedDictData = []
	feedDictTensors = []


	### Load the required tensors
	print('Restoring tensorflow graph.')
	tf.reset_default_graph()
	saver =                           tf.train.import_meta_graph(results_dir + 'mobsGraph.meta')

	feedDictTensors.append(           tf.get_default_graph().get_tensor_by_name("bayesianDecoder/binTime:0") )

	for tetrode in range(n_tetrodes):
		feedDictTensors.append(       tf.get_default_graph().get_tensor_by_name("group"+str(tetrode)+"-encoder/x:0") )

	positionProba =                   tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0")
	outputShape = positionProba.get_shape().as_list()
	neutralOutput = np.ones(outputShape) / np.sum(outputShape)
	


	### Cut the data up
	nBins = math.floor((stop_time - start_time)/bin_time)
	print('Preparing data.')
	for bin in range(nBins):
		bin_start_time = start_time + bin*bin_time
		bin_stop_time = bin_start_time + bin_time

		feedDictDataBin = []
		feedDictDataBin.append([bin_time])

		spikes = []
		times = []
		groups = []
		for tetrode in range(n_tetrodes):
			temp = Data['spikes_time'][tetrode][np.where(np.logical_and(
								Data['spikes_time'][tetrode][:] >= bin_start_time,
								Data['spikes_time'][tetrode][:] < bin_stop_time))[0]]
			spk = Data['spikes_all'][tetrode][np.where(np.logical_and(
								Data['spikes_time'][tetrode][:] >= bin_start_time,
								Data['spikes_time'][tetrode][:] < bin_stop_time))[0]]
			spikes += [spk]
			times += [temp]
			groups += [tetrode]*len(temp)
		groups = np.array(groups)
		spikes = np.concatenate(spikes, axis=0)
		times = np.concatenate(times, axis=0)
		nSpikes.append(len(spikes))
		feedDictDataBin += [(spikes[:]*(groups==tet)[:,None,None])[np.argsort(times, axis=0)][:,:,:] for tet in range(n_tetrodes)]
		# feedDictDataBin += [(spikes[:]*(groups==tet)[:,None,None])[np.argsort(times, axis=0)][:,0,:,:] for tet in range(n_tetrodes)]

		feedDictData.append(feedDictDataBin)
		

		position_idx = np.argmin(np.abs(bin_stop_time-Data['position_time']))
		position_bin = Data['positions'][position_idx,:]
		truePositions.append( truePositions[-1] if np.isnan(position_bin).any() else position_bin )

		if bin%10==0:
			sys.stdout.write('[%-30s] step : %d/%d' % ('='*(bin*30//nBins),bin,nBins))
			sys.stdout.write('\r')
			sys.stdout.flush()

	truePositions.pop(0)
	print("Data is prepared. We're sending it through the tensorflow graph.")


	# Send the spiking data through the tensorflow graph
	emptyBins = 0
	times = [datetime.now()]
	with tf.Session() as sess:
		saver.restore(sess, results_dir + 'mobsGraph')

		for bin in range(nBins):
			try:
				decodedPositions.append(positionProba.eval({i:j for i,j in zip(feedDictTensors, feedDictData[bin])}))
				if np.isnan(np.sum(decodedPositions[-1])):
					decodedPositions.pop()
					nSpikes.pop(len(decodedPositions))
					truePositions.pop(len(decodedPositions))
			except tf.errors.InvalidArgumentError:
				decodedPositions.append(neutralOutput)
				emptyBins +=1
			if bin%10==0:
				sys.stdout.write('[%-30s] step : %d/%d' % ('='*(bin*30//nBins),bin,nBins))
				sys.stdout.write('\r')
				sys.stdout.flush()
			sys.stdout.write('[%-30s] step : %d/%d' % ('='*((bin+1)*30//nBins),bin+1,nBins))
			sys.stdout.write('\r')
			sys.stdout.flush()
			times.append(datetime.now())

	if emptyBins!=0:
		print('Some bins have not been decoded because of issues with a flattening tensor : %d/%d' % (emptyBins, nBins))
	print('\nfinished.')
	return decodedPositions, truePositions, nSpikes, times



if __name__ == '__main__':
	print(0)
	sys.exit(0)


    