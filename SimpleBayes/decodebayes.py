# Load libs
from SimpleBayes import butils
from importData import ImportClusters

import numpy as np
import os
import sys
import math
import tables
import struct
import os.path
import subprocess
from scipy import signal
from functools import reduce
from sklearn.neighbors import KernelDensity

def inEpochs(t,epochs):
	# for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
	# test if t is among at least one of these epochs
	# Epochs are treated as closed interval [,]
	mask =  np.sum([(t>=epochs[2*i]) * (t<=epochs[2*i+1]) for i in range(len(epochs)//2)],axis=0)
	return np.where(mask >= 1)



# trainer class
class Trainer():
	def __init__(self, projectPath, bandwidth=None, kernel='epanechnikov'):
		self.projectPath = projectPath
		self.bandwidth = bandwidth
		self.kernel = kernel

	def train(self, behavior_data, cluster_data, speed_cut=0, masking_factor=20, rand_param=0):
		# Check for bandwidth
		if self.bandwidth == None:
			self.bandwidth = behavior_data['Bandwidth']

		### GLOBAL OCCUPATION
		selected_positions = behavior_data['Positions'][reduce(np.intersect1d,
			(np.where(behavior_data['Speed'][:,0] > speed_cut),
			inEpochs(behavior_data['Position_time'][:,0], behavior_data['Times']['trainEpochs']),
			))]
		# xEdges, yEdges, Occupation = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, kernel=self.kernel)
		gridFeature, Occupation = butils.kdenD(selected_positions, self.bandwidth,kernel=self.kernel)
		Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

		mask = Occupation > (np.max(Occupation)/masking_factor)
		Occupation_inverse = 1/Occupation
		Occupation_inverse[Occupation_inverse==np.inf] = 0
		Occupation_inverse = np.multiply(Occupation_inverse, mask)

		### Built spike-related kernels
		guessed_clusters_time = []
		guessed_clusters = []
		Marginal_rate_functions = []
		Rate_functions = []

		n_tetrodes = len(cluster_data['Spike_labels'])
		for tetrode in range(n_tetrodes):
			# Get cluster for inferring
			guessed_clusters_time.append(cluster_data['Spike_times'][tetrode][inEpochs(cluster_data['Spike_times'][tetrode][:,0],
																					   behavior_data['Times']['testEpochs'])])
			guessed_clusters.append(butils.shuffle_labels(cluster_data['Spike_labels'][tetrode][inEpochs(cluster_data['Spike_times'][tetrode][:,0],
																										  behavior_data['Times']['testEpochs'])], rand_param))

			### MARGINAL RATE FUNCTION
			selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
				(np.where(cluster_data['Spike_speed'][tetrode][:,0] > speed_cut),
				inEpochs(cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
			# xEdges, yEdges, MRF = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
			gridFeature, MRF = butils.kdenD(selected_positions, self.bandwidth,
											   edges=gridFeature, kernel=self.kernel)
			MRF[MRF==0] = np.min(MRF[MRF!=0])
			MRF         = MRF/np.sum(MRF)
			MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/behavior_data['Times']['learning']
			Marginal_rate_functions.append(MRF)

			Local_rate_functions = []
		### LOCAL RATE FUNCTION FOR EACH CLUSTER
			for label in range(np.shape(cluster_data['Spike_labels'][tetrode])[1]):
				selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
					(np.where(cluster_data['Spike_speed'][tetrode][:,0] > speed_cut),
					np.where(cluster_data['Spike_labels'][tetrode][:,label] == 1),
					inEpochs(cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
				if np.shape(selected_positions)[0]!=0:
					# xEdges, yEdges, LRF =  butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
					gridFeature, LRF = butils.kdenD(selected_positions, self.bandwidth,
													edges=gridFeature, kernel=self.kernel)
					LRF[LRF==0] = np.min(LRF[LRF!=0])
					LRF         = LRF/np.sum(LRF)
					LRF         = np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/behavior_data['Times']['learning']
					Local_rate_functions.append(LRF)
				else:
					Local_rate_functions.append(np.ones(np.shape(Occupation)))

			Rate_functions.append(Local_rate_functions)

		#given a n-dim grid, let us find the binning over each edges
		bayes_matrices = {'Occupation': Occupation, 'Marginal rate functions': Marginal_rate_functions, 'Rate functions': Rate_functions,
				'Bins':[np.unique(gridFeature[i]) for i in range(len(gridFeature))],
				'guessed_clusters_info': {'clusters':guessed_clusters, 'time':guessed_clusters_time},
				}
		#'time_limits': [behavior_data['Times']['start_train'], behavior_data['Times']['stop_train'],
				                # behavior_data['Times']['start_test'], behavior_data['Times']['stop_test']]
		return bayes_matrices



	def test(self, bayes_matrices, behavior_data, windowSize=0.036, masking_factor=20):

		print('\nBUILDING POSITION PROBAS')

		### Format matrices
		guessed_clusters_info = bayes_matrices['guessed_clusters_info']
		Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in ['Occupation','Marginal rate functions','Rate functions']]
		guessed_clusters, guessed_clusters_time = [guessed_clusters_info[key] for key in ['clusters','time']]
		mask = Occupation > (np.max(Occupation)/masking_factor)
  
		### Build Poisson term
		# first we bin the time
		testEpochs = behavior_data['Times']['testEpochs']
		Ttest = np.sum([testEpochs[2 * i + 1] - testEpochs[2 * i] for i in range(len(testEpochs) // 2)])
		n_bins = math.floor(Ttest/windowSize)
		# for each bin we will need to now the test epoch it belongs to, so that we can then
		# set the time correctly to select the corresponding spikes
		timeEachTestEpoch = [testEpochs[2 * i + 1] - testEpochs[2 * i] for i in range(len(testEpochs) // 2)]
		cumTimeEachTestEpoch = np.cumsum(timeEachTestEpoch)
		cumTimeEachTestEpoch = np.concatenate([[0],cumTimeEachTestEpoch])
		# a function that given the bin indicates the bin index:
		binToEpoch = lambda x : np.where(((x*windowSize - cumTimeEachTestEpoch[0:-1])>=0)*((x*windowSize - cumTimeEachTestEpoch[1:])<0))[0][0]
		binToEpochArray = [binToEpoch(bins) for bins in range(n_bins)]
		firstBinEpoch = [np.min(np.where(np.equal(binToEpochArray,epochId))[0]) for epochId in range(len(timeEachTestEpoch))]
		All_Poisson_term = [np.exp((-windowSize)*Marginal_rate_functions[tetrode]) for tetrode in range(len(guessed_clusters))]
		All_Poisson_term = reduce(np.multiply, All_Poisson_term)

		### Log of rate functions
		log_RF = []
		for tetrode in range(np.shape(Rate_functions)[0]):
			temp = []
			for cluster in range(np.shape(Rate_functions[tetrode])[0]):
				temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster]!=0])))
			log_RF.append(temp)

		### Decoding loop
		position_proba = [np.ones(np.shape(Occupation))] * n_bins
		position_true = [np.ones(2)] * n_bins
		nSpikes = []
		times = []
		for bin in range(n_bins):

			bin_start_time = testEpochs[2*binToEpoch(bin)] + (bin-firstBinEpoch[binToEpoch(bin)])*windowSize
			bin_stop_time = bin_start_time + windowSize
			times.append(bin_start_time)

			binSpikes = 0
			tetrodes_contributions = []
			tetrodes_contributions.append(All_Poisson_term)

			for tetrode in range(len(guessed_clusters)):
				# Clusters inside our window
				bin_probas = guessed_clusters[tetrode][np.intersect1d(
					np.where(guessed_clusters_time[tetrode][:,0] > bin_start_time),
					np.where(guessed_clusters_time[tetrode][:,0] < bin_stop_time))]
				bin_clusters = np.sum(bin_probas, 0)
				binSpikes = binSpikes + np.sum(bin_clusters)


				# Terms that come from spike information
				if np.sum(bin_clusters) > 0.5:
					spike_pattern = reduce(np.multiply,
						[np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
						for cluster in range(np.shape(bin_clusters)[0])])
				else:
					spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

				tetrodes_contributions.append(spike_pattern)

			nSpikes.append(binSpikes)

			# Guessed probability map
			position_proba[bin] = reduce(np.multiply, tetrodes_contributions)
			position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])
			# True position
			position_true_mean = np.nanmean( behavior_data['Positions'][reduce(np.intersect1d,
				(np.where(behavior_data['Position_time'][:,0] > bin_start_time),
				np.where(behavior_data['Position_time'][:,0] < bin_stop_time)))], axis=0 )
			position_true[bin] = position_true[bin-1] if np.isnan(position_true_mean).any() else position_true_mean

			if bin % 50 == 0:
				sys.stdout.write('[%-30s] : %.3f %%' % ('='*(bin*30//n_bins),bin*100/n_bins))
				sys.stdout.write('\r')
				sys.stdout.flush()
		sys.stdout.write('[%-30s] : %.3f %%' % ('='*((bin+1)*30//n_bins),(bin+1)*100/n_bins))
		sys.stdout.write('\r')
		sys.stdout.flush()

		position_true[0] = position_true[1]
		print('\nDecoding finished')

		# Guessed X and Y

		#Pierre: correction, here we should not marginalize over x and y to find the best predicted proba
		allProba = [np.unravel_index(np.argmax(position_proba[bin]),position_proba[bin].shape) for bin in range(len(nSpikes))]
		bestProba = [np.max(position_proba[bin]) for bin in range(len(nSpikes))]
		position_guessed = [ [bayes_matrices['Bins'][i][allProba[bin][i]] for i in range(len(bayes_matrices['Bins']))]
							 for bin in range(len(nSpikes)) ]
		inferResults = np.concatenate([np.array(position_guessed),np.array(bestProba).reshape([-1,1])],axis=-1)

		outputResults = {"inferring":inferResults, "pos": np.array(position_true), "probaMaps": position_proba, "times":np.array(times), 'nSpikes': np.array(nSpikes)}
		return outputResults



