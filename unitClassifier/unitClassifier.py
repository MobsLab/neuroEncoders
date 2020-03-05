import os
import sys
import datetime
import tables
import math
import random
import struct
import numpy as np
import tensorflow as tf
import multiprocessing
from sklearn.neighbors import KernelDensity
from functools import reduce
from datetime import datetime


def swap(array, x, y):
    array[[x, y]] = array[[y,x]]

def shuffle_spike_time(Data, maxDisplacement=0.05):
    '''maxDisplacement in seconds'''
    from tqdm import tqdm
    tot=0
    for group in range(Data['nGroups']):
        tot += len(Data['spikes_time'][group]) - 1

    pbar = tqdm(total=tot)
    for group in range(Data['nGroups']):
        pbar.set_description('group '+str(group))
        pbar.refresh()

        while len(Data['spikes_all'][group]) > len(Data['spikes_time'][group]):
            Data['spikes_all'][group] = Data['spikes_all'][group][:-1]
        while len(Data['labels_all'][group]) > len(Data['spikes_time'][group]):
        	Data['labels_all'][group] = Data['labels_all'][group][:-1]

        for spk in range(len(Data['spikes_time'][group])):
            Data['spikes_time'][group][spk] += 2*maxDisplacement*np.random.random() - maxDisplacement

        for spk in range(1, len(Data['spikes_time'][group])):
            idx = spk

            while Data['spikes_time'][group][idx] < Data['spikes_time'][group][idx-1]:
                swap(Data['spikes_time'][group], idx, idx-1)
                swap(Data['spikes_all'][group], idx, idx-1)
                swap(Data['labels_all'][group], idx, idx-1)
                idx -= 1
                if idx <= 0:
                    break
            pbar.update(1)
    pbar.close()


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

def makeGaussian(size, fwhm = 3, center=None):
	""" Make a square gaussian kernel.
	size is the length of a side of the square
	fwhm is full-width-half-maximum, which
	can be thought of as an effective radius.
	"""
	
	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]
	
	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]
	
	unnormalized = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
	return unnormalized / np.sum(unnormalized)

def kde2D(x, y, bandwidth, xbins=45j, ybins=45j, **kwargs):
	"""Build 2D kernel density estimate (KDE)."""

	kernel       = kwargs.get('kernel',       'epanechnikov')
	if ('edges' in kwargs):
		xx = kwargs['edges'][0]
		yy = kwargs['edges'][1]
	else:
		# create grid of sample locations (default: 45x45)
		xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]


	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T

	kde_skl = KernelDensity(kernel=kernel, bandwidth=bandwidth)
	kde_skl.fit(xy_train)

	# score_samples() returns the log-likelihood of the samples
	z = np.exp(kde_skl.score_samples(xy_sample))
	zz = np.reshape(z, xx.shape)
	return xx, yy, zz/np.sum(zz)


class groupProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class groupContext(type(multiprocessing.get_context())):
    Process = groupProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class groupsPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = groupContext()
        super(groupsPool, self).__init__(*args, **kwargs)


def rateFunctions(clu_path, group, nChannels, 
	start_time, stop_time, end_time, 
	positions, position_time, speed, speed_cut, 
	Occupation_inverse, edges, bandwidth, kernel, samplingRate):
	learning_time = stop_time - start_time
	displ = 2.5 # displacing spikes

	print('Starting data from group '+ str(group+1))
	if os.path.isfile(clu_path + 'clu.' +str(group+1)):
		with open(
					clu_path + 'clu.' + str(group+1), 'r') as fClu, open(
					clu_path + 'res.' + str(group+1), 'r') as fRes, open(
					clu_path + 'spk.' + str(group+1), 'rb') as fSpk:
			clu_str = fClu.readlines()
			res_str = fRes.readlines()
			clusters = np.array([int(clu_str[n]) for n in range(len(clu_str))])
			nClusters = int(clu_str[0]) + (1 - int(bool(np.sum(clusters[1:]==1))))
			print('number of clusters found in .clu.',group+1,':',nClusters)

			spike_time      = np.array([[float(res_str[n])/samplingRate] for n in range(len(res_str))])
			dataSelection   = np.where(np.logical_and(
								spike_time[:,0] > start_time,
								spike_time[:,0] < end_time))[0]

			labels          = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(nClusters)] for n in dataSelection])
			# spike_positions = np.array([positions[np.argmin(np.abs((spike_time[n]+np.random.random()*2*displ - displ)-position_time)),:] for n in dataSelection])
			spike_positions = np.array([positions[np.argmin(np.abs(spike_time[n]-position_time)),:] for n in dataSelection])
			spike_speed     = np.array([speed[np.min((np.argmin(np.abs(spike_time[n]-position_time)), len(speed)-1)),:] for n in dataSelection])
			
			n=0
			spikes = []
			fSpk.seek(dataSelection[0]*2*32*nChannels) #skip the first spikes, that are not going to be read
			spikeReader = struct.iter_unpack(str(32*nChannels)+'h', fSpk.read())
			for it in spikeReader:
				if n > len(dataSelection):
					break
				spike = np.reshape(np.array(it), [32,nChannels])
				if np.sum(spike)!=0:
					spikes.append(np.transpose(spike))
				else:
					labels = np.delete(labels, n, axis=0)
					spike_positions = np.delete(spike_positions, n, axis=0)
					spike_speed = np.delete(spike_speed, n, axis=0)
					dataSelection = np.delete(dataSelection, n, axis=0)
					n -= 1 
				n = n+1
			spikes = np.array(spikes, dtype=float) * 0.195 # microvolts
			spike_time = spike_time[dataSelection]
	else:
		print('File ' + clu_path + 'clu.' + str(group+1) + ' not found.')
		return []
		


	trainingTimeSelection = np.where(np.logical_and(
		spike_time[:,0] > start_time, 
		spike_time[:,0] < stop_time))
	spikes_temp     =          spikes[trainingTimeSelection]
	labels_temp     =          labels[trainingTimeSelection]
	spike_speed     =     spike_speed[trainingTimeSelection]
	spike_positions = spike_positions[trainingTimeSelection]

	spikes = {'all':  spikes,
			  'train':spikes_temp[0:len(spikes_temp)*9//10,:,:],
			  'test': spikes_temp[len(spikes_temp)*9//10:len(spikes_temp),:,:]}
	labels = {'all':  labels,
			  'train':labels_temp[0:len(labels_temp)*9//10,:],
			  'test': labels_temp[len(labels_temp)*9//10:len(labels_temp),:]}
	

	### MARGINAL RATE FUNCTION
	selected_positions = spike_positions[np.where(spike_speed[:,0] > speed_cut)]
	_, _, MRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=edges, kernel=kernel)
	MRF[MRF==0] = np.min(MRF[MRF!=0])
	MRF         = MRF/np.sum(MRF)
	MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/learning_time

	### RATE FUNCTION

	with multiprocessing.Pool(np.shape(labels_temp)[1]) as p:
		Local_rate_functions = p.starmap( 
			rateFunction, 
			((label, labels_temp, 
			spike_positions, spike_speed, speed_cut, 
			bandwidth, edges, kernel, Occupation_inverse, learning_time) for label in range(np.shape(labels_temp)[1])))


	print('Finished data from group '+ str(group+1))
	return [nClusters, MRF, Local_rate_functions, spikes, spike_time, labels]


def rateFunctionsFromSpikes(clu_path, group, nChannels, rawSpikes,
	start_time, stop_time, end_time, 
	positions, position_time, speed, speed_cut, 
	Occupation_inverse, edges, bandwidth, kernel, samplingRate):
	learning_time = stop_time - start_time
	
	print('Starting data from group '+ str(group+1))
	if os.path.isfile(clu_path + 'clu.' +str(group+1)):
		with open(
					clu_path + 'clu.' + str(group+1), 'r') as fClu, open(
					clu_path + 'res.' + str(group+1), 'r') as fRes:
			clu_str = fClu.readlines()
			res_str = fRes.readlines()
			clusters = np.array([int(clu_str[n]) for n in range(len(clu_str))])
			nClusters = int(clu_str[0]) + (1 - int(bool(np.sum(clusters[1:]==1))))
			print('number of clusters found in .clu.',group+1,':',nClusters)

			
			rawSpikes['labels'] = []
			labelCursor = 0
			spikeCursor = 0
			while spikeCursor<len(rawSpikes['times']) and labelCursor<len(res_str):
				if np.abs(float(res_str[labelCursor])/samplingRate - rawSpikes['times'][spikeCursor]) < 0.0003:
					rawSpikes['labels'].append([1. if int(clu_str[labelCursor+1])==l else 0. for l in range(nClusters)])
					spikeCursor += 1
					labelCursor += 1
				elif (float(res_str[labelCursor])/samplingRate < rawSpikes['times'][spikeCursor]):
					labelCursor += 1
				else:
					rawSpikes['labels'].append([1.] + [0. for l in range(nClusters-1)])
					spikeCursor += 1

			while spikeCursor<len(rawSpikes['times']):
				rawSpikes['labels'].append([1.] + [0. for l in range(nClusters-1)])
				spikeCursor += 1

	else:
		print('File ' + clu_path + 'clu.' + str(group+1) + ' not found.')
		return []
		

	rawSpikes['times'] = np.array(rawSpikes['times'])
	trainingTimeSelection = np.where(np.logical_and(
		rawSpikes['times'][:] > start_time, 
		rawSpikes['times'][:] < stop_time))[0]

	spikes_temp     =          np.array(rawSpikes['spikes'])[trainingTimeSelection]
	labels_temp     =          np.array(rawSpikes['labels'])[trainingTimeSelection]
	spike_speed     =          np.array(rawSpikes['speeds'])[trainingTimeSelection]
	spike_positions =       np.array(rawSpikes['positions'])[trainingTimeSelection]


	spikes = {'all':  np.array(rawSpikes['spikes']),
			  'train':spikes_temp[0:len(spikes_temp)*9//10,:,:],
			  'test': spikes_temp[len(spikes_temp)*9//10:len(spikes_temp),:,:]}
	labels = {'all':  np.array(rawSpikes['labels']),
			  'train':labels_temp[0:len(labels_temp)*9//10,:],
			  'test': labels_temp[len(labels_temp)*9//10:len(labels_temp),:]}


	### MARGINAL RATE FUNCTION
	selected_positions = spike_positions[np.where(spike_speed[:,0] > speed_cut)]
	_, _, MRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=edges, kernel=kernel)
	MRF[MRF==0] = np.min(MRF[MRF!=0])
	MRF         = MRF/np.sum(MRF)
	MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/learning_time

	### RATE FUNCTION

	with multiprocessing.Pool(np.shape(labels_temp)[1]) as p:
		Local_rate_functions = p.starmap( 
			rateFunction, 
			((label, labels_temp, 
			spike_positions, spike_speed, speed_cut, 
			bandwidth, edges, kernel, Occupation_inverse, learning_time) for label in range(np.shape(labels_temp)[1])))


	print('Finished data from group '+ str(group+1))
	return [nClusters, MRF, Local_rate_functions, spikes, rawSpikes['times'], labels]


def rateFunction(label, labels, spike_positions, spike_speed, speed_cut, bandwidth, edges, kernel, Occupation_inverse, learning_time):
	selected_positions = spike_positions[np.where(np.logical_and( 
		spike_speed[:,0] > speed_cut, 
		labels[:,label] == 1))]
	if np.shape(selected_positions)[0]!=0:
		_, _, LRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=edges, kernel=kernel)
		LRF[LRF==0] = np.min(LRF[LRF!=0])
		LRF         = LRF/np.sum(LRF)
		return np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/learning_time
	else:
		return np.ones([45,45])

def selectGroupInDict(dictionnary, group):
	return {key:dictionnary[key][group] for key in dictionnary.keys()}

def build_maps(clu_path, list_channels, rawSpikes,
    start_time, stop_time, end_time,
    speed_cut, samplingRate,
    masking_factor, kernel, bandwidth):
	
	with tables.open_file(os.path.dirname(clu_path) + '/nnBehavior.mat') as f:
		positions = f.root.behavior.positions
		speed = f.root.behavior.speed
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:,:],1,0)
		speed = np.swapaxes(speed[:,:],1,0)
		position_time = np.swapaxes(position_time[:,:],1,0)
	if stop_time == None:
		stop_time = position_time[-1]
	if bandwidth == None:
		bandwidth = (np.max(positions) - np.min(positions))/20


	### GLOBAL OCCUPATION
	if np.shape(speed)[0] != np.shape(positions)[0]:
		if np.shape(speed)[0] == np.shape(positions)[0] - 1:
			speed.append(speed[-1])
		elif np.shape(speed)[0] == np.shape(positions)[0] + 1:
			speed = speed[:-1]
		else:
			sys.exit(5)
	selected_positions = positions[np.where(np.logical_and.reduce( 
		[speed[:,0] > speed_cut, 
		position_time[:,0] > start_time, 
		position_time[:,0] < stop_time]))]
	xEdges, yEdges, Occupation = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, kernel=kernel)
	Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

	mask = Occupation > (np.max(Occupation)/masking_factor)
	Occupation_inverse = 1/Occupation
	Occupation_inverse[Occupation_inverse==np.inf] = 0
	Occupation_inverse = np.multiply(Occupation_inverse, mask)

	print('Behavior data extracted')

	totNClusters = 0
	nGroups = len(list_channels)
	channelsPerGroup = []
	clustersPerGroup = []

	spikes_all = []
	spikes_time = []
	spikes_train = []
	spikes_test = []
	labels_all = []
	labels_train = []
	labels_test = []

	Marginal_rate_functions = []
	Rate_functions = []

	### Extract
	undone_tetrodes = 0
	nCoresAvailable = multiprocessing.cpu_count() // 2 # We're mercifully leaving half of cpus for other processes.
	processingPools = [
		[pool*nCoresAvailable + group 
		for group in range(min(nCoresAvailable, nGroups-pool*nCoresAvailable))]
		for pool in range(nGroups//nCoresAvailable+1)]

	for pool in processingPools:
		if pool == []:
			continue
		with groupsPool(len(pool)) as p:
			Results = p.starmap(rateFunctionsFromSpikes, 
				((clu_path, group, len(list_channels[group]), selectGroupInDict(rawSpikes, group),
				start_time, stop_time, end_time,
				positions, position_time, speed, speed_cut,
				Occupation_inverse, [xEdges, yEdges], bandwidth, kernel,
				samplingRate) for group in pool))

		for group in range(len(pool)):

			if Results[group] == []:
				undone_tetrodes += 1
				continue

			totNClusters += Results[group][0]
			clustersPerGroup.append(Results[group][0])
			channelsPerGroup.append(len(list_channels[group]))

			spikes_time.append(Results[group][4])
			spikes_all.append(Results[group][3]['all'])
			spikes_train.append(Results[group][3]['train'])
			spikes_test.append(Results[group][3]['test'])
			labels_all.append(Results[group][5]['all'])
			labels_train.append(Results[group][5]['train'])
			labels_test.append(Results[group][5]['test'])

			Marginal_rate_functions.append(Results[group][1])
			Rate_functions.append(Results[group][2])
	
	return {'nGroups':nGroups - undone_tetrodes, 'nClusters':totNClusters, 'clustersPerGroup':clustersPerGroup, 'channelsPerGroup':channelsPerGroup, 
				'positions':positions, 'position_time':position_time, 'speed':speed,  
				'spikes_all':spikes_all, 'spikes_time':spikes_time, 'labels_all':labels_all,
				'spikes_train':spikes_train, 'spikes_test':spikes_test, 'labels_train':labels_train, 'labels_test':labels_test, 
				'Occupation':Occupation, 'Mask':mask, 
				'Marginal_rate_functions':Marginal_rate_functions, 'Rate_functions':Rate_functions, 'Bins':[xEdges[:,0],yEdges[0,:]]}





def extract_data(clu_path, list_channels, start_time, stop_time, end_time,
					speed_cut, samplingRate, 
					masking_factor, kernel, bandwidth):
	
	print('Extracting data.\n')

	with tables.open_file(os.path.dirname(clu_path) + '/nnBehavior.mat') as f:
		positions = f.root.behavior.positions
		speed = f.root.behavior.speed
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:,:],1,0)
		speed = np.swapaxes(speed[:,:],1,0)
		position_time = np.swapaxes(position_time[:,:],1,0)
	if stop_time == None:
		stop_time = position_time[-1]
	if bandwidth == None:
		bandwidth = (np.max(positions) - np.min(positions))/20


	### GLOBAL OCCUPATION
	if np.shape(speed)[0] != np.shape(positions)[0]:
		if np.shape(speed)[0] == np.shape(positions)[0] - 1:
			speed.append(speed[-1])
		elif np.shape(speed)[0] == np.shape(positions)[0] + 1:
			speed = speed[:-1]
		else:
			sys.exit(5)
	selected_positions = positions[np.where(np.logical_and.reduce( 
		[speed[:,0] > speed_cut, 
		position_time[:,0] > start_time, 
		position_time[:,0] < stop_time]))]
	xEdges, yEdges, Occupation = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, kernel=kernel)
	Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

	mask = Occupation > (np.max(Occupation)/masking_factor)
	Occupation_inverse = 1/Occupation
	Occupation_inverse[Occupation_inverse==np.inf] = 0
	Occupation_inverse = np.multiply(Occupation_inverse, mask)

	print('Behavior data extracted')



	totNClusters = 0
	nGroups = len(list_channels)
	channelsPerGroup = []
	clustersPerGroup = []

	spikes_all = []
	spikes_time = []
	spikes_train = []
	spikes_test = []
	labels_all = []
	labels_train = []
	labels_test = []

	Marginal_rate_functions = []
	Rate_functions = []

	### Extract
	undone_tetrodes = 0
	nCoresAvailable = multiprocessing.cpu_count() // 2 # We're mercifully leaving half of cpus for other processes.
	processingPools = [
		[pool*nCoresAvailable + group 
		for group in range(min(nCoresAvailable, nGroups-pool*nCoresAvailable))]
		for pool in range(nGroups//nCoresAvailable+1)]

	for pool in processingPools:
		if pool == []:
			continue
		with groupsPool(len(pool)) as p:
			Results = p.starmap(rateFunctions, 
				((clu_path, group, len(list_channels[group]), start_time, stop_time, end_time,
				positions, position_time, speed, speed_cut,
				Occupation_inverse, [xEdges, yEdges], bandwidth, kernel,
				samplingRate) for group in pool))

		for group in range(len(pool)):

			if Results[group] == []:
				undone_tetrodes += 1
				continue

			totNClusters += Results[group][0]
			clustersPerGroup.append(Results[group][0])
			channelsPerGroup.append(len(list_channels[group]))

			spikes_time.append(Results[group][4])
			spikes_all.append(Results[group][3]['all'])
			spikes_train.append(Results[group][3]['train'])
			spikes_test.append(Results[group][3]['test'])
			labels_all.append(Results[group][5]['all'])
			labels_train.append(Results[group][5]['train'])
			labels_test.append(Results[group][5]['test'])

			Marginal_rate_functions.append(Results[group][1])
			Rate_functions.append(Results[group][2])



	return {'nGroups':nGroups - undone_tetrodes, 'nClusters':totNClusters, 'clustersPerGroup':clustersPerGroup, 'channelsPerGroup':channelsPerGroup, 
				'positions':positions, 'position_time':position_time, 'speed':speed,  
				'spikes_all':spikes_all, 'spikes_time':spikes_time, 'labels_all':labels_all,
				'spikes_train':spikes_train, 'spikes_test':spikes_test, 'labels_train':labels_train, 'labels_test':labels_test, 
				'Occupation':Occupation, 'Mask':mask, 
				'Marginal_rate_functions':Marginal_rate_functions, 'Rate_functions':Rate_functions, 'Bins':[xEdges[:,0],yEdges[0,:]]}














































def build_position_decoder(Data, results_dir, nSteps):
	"""Trains one artificial neural network to guess position proba from spikes"""
	from unitClassifier import mobs_networkbuilding as nb

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

			spikeEncoder, ops = nb.layeredEncoder(x,Data['clustersPerGroup'][tetrode], Data['channelsPerGroup'][tetrode], size=200)
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


    