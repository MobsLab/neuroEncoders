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
			np.where(behavior_data['Position_time'][:,0] > behavior_data['Times']['start']),
			np.where(behavior_data['Position_time'][:,0] < behavior_data['Times']['stop'])))]
		xEdges, yEdges, Occupation = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, kernel=self.kernel)
		Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

		mask = Occupation > (np.max(Occupation)/masking_factor)
		Occupation_inverse = 1/Occupation
		Occupation_inverse[Occupation_inverse==np.inf] = 0
		Occupation_inverse = np.multiply(Occupation_inverse, mask)

		### Built spike-related kernels
		fake_labels_time = []
		fake_labels = []
		Marginal_rate_functions = []
		Local_rate_functions = []
		Rate_functions = []

		n_tetrodes = len(cluster_data['Spike_labels'])
		for tetrode in range(n_tetrodes):
			fake_labels_time.append(cluster_data['Spike_times'][tetrode][reduce(np.intersect1d,
				(np.where(cluster_data['Spike_times'][tetrode][:,0] > behavior_data['Times']['stop']),
				np.where(cluster_data['Spike_times'][tetrode][:,0] < behavior_data['Times']['end']),
				np.where(cluster_data['Spike_times'][tetrode][:,0] > speed_cut)))])
			fake_labels.append(butils.shuffle_labels(cluster_data['Spike_labels'][tetrode][reduce(np.intersect1d,
				(np.where(cluster_data['Spike_times'][tetrode][:,0] > behavior_data['Times']['stop']),
				np.where(cluster_data['Spike_times'][tetrode][:,0] < behavior_data['Times']['end']),
				np.where(cluster_data['Spike_times'][tetrode][:,0] > speed_cut)))], rand_param))

			### MARGINAL RATE FUNCTION
			selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
				(np.where(cluster_data['Spike_speed'][tetrode][:,0] > speed_cut),
				np.where(cluster_data['Spike_times'][tetrode][:,0] > behavior_data['Times']['start']),
				np.where(cluster_data['Spike_times'][tetrode][:,0] < behavior_data['Times']['stop'])))]
			xEdges, yEdges, MRF = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
			MRF[MRF==0] = np.min(MRF[MRF!=0])
			MRF         = MRF/np.sum(MRF)
			MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/behavior_data['Times']['learning']
			Marginal_rate_functions.append(MRF)

		### LOCAL RATE FUNCTION FOR EACH CLUSTER
			for label in range(np.shape(cluster_data['Spike_labels'][tetrode])[1]):
				selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
					(np.where(cluster_data['Spike_speed'][tetrode][:,0] > speed_cut),
					np.where(cluster_data['Spike_labels'][tetrode][:,label] == 1),
					np.where(cluster_data['Spike_times'][tetrode][:,0] > behavior_data['Times']['start']),
					np.where(cluster_data['Spike_times'][tetrode][:,0] < behavior_data['Times']['stop'])))]
				if np.shape(selected_positions)[0]!=0:
					xEdges, yEdges, LRF =  butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
					LRF[LRF==0] = np.min(LRF[LRF!=0])
					LRF         = LRF/np.sum(LRF)
					LRF         = np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/behavior_data['Times']['learning']
					Local_rate_functions.append(LRF)
				else:
					Local_rate_functions.append(np.ones(np.shape(Occupation)))

			Rate_functions.append(Local_rate_functions)

		bayes_matrices = {'Occupation': Occupation, 'Marginal rate functions': Marginal_rate_functions, 'Rate functions': Rate_functions,
				'Bins':[xEdges[:,0],yEdges[0,:]],'fake_labels_info': {'clusters':fake_labels, 'time':fake_labels_time},
				'time_limits': [behavior_data['Times']['start'], behavior_data['Times']['stop']]}

		return bayes_matrices



	def test(self, project_path, bayes_matrices, behavior_data, windowSize=0.036, speed_cut=0, masking_factor=20):

		print('\nBUILDING POSITION PROBAS')

		### Format matrices
		guessed_clusters_info = bayes_matrices['fake_labels_info']
		space_bins = [np.array((bayes_matrices['Bins'][0][b],bayes_matrices['Bins'][1][b])) for b in range(np.shape(bayes_matrices['Bins'][0])[0])]
		Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in ['Occupation','Marginal rate functions','Rate functions']]
		guessed_clusters, guessed_clusters_time = [guessed_clusters_info[key] for key in ['clusters','time']]
		mask = Occupation > (np.max(Occupation)/masking_factor)

		# for tetrode in range(np.shape(guessed_clusters)[0]):
		# 	for spike in range(np.shape(guessed_clusters[tetrode])[0]):
		# 		if guessed_clusters[tetrode][spike, 0] > 1/np.shape(guessed_clusters[tetrode])[1]:
		# 			guessed_clusters[tetrode][spike, :] = np.zeros([1,np.shape(guessed_clusters[tetrode])[1]])
		# 		else:
		# 			guessed_clusters[tetrode][spike, 0] = 0
		# 			guessed_clusters[tetrode][spike, :] = guessed_clusters[tetrode][spike, :]/np.sum(guessed_clusters[tetrode][spike, :], 0)

		### Build Poisson term
		n_bins = math.floor((bayes_matrices['time_limits'][1] - bayes_matrices['time_limits'][0])/windowSize)
		All_Poisson_term = [np.exp( (-windowSize)*Marginal_rate_functions[tetrode] ) for tetrode in range(len(guessed_clusters))]
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
		for bin in range(n_bins):

			bin_start_time = behavior_data['Times']['start'] + bin*windowSize
			bin_stop_time = bin_start_time + windowSize

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


				# Terms that come from spike information (with normalization)
				if np.sum(bin_clusters) > 0.5:
					place_maps = reduce(np.multiply,
						[butils.exp256( (log_RF[tetrode][cluster] * bin_clusters[cluster]) )
						for cluster in range(np.shape(bin_clusters)[0])])


					spike_pattern = math.pow(windowSize, np.sum(bin_clusters)) * place_maps
				else:
					spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

				tetrodes_contributions.append( spike_pattern )#/np.sum(spike_pattern) )

			nSpikes.append(binSpikes)

			position_proba[bin] = reduce( np.multiply, tetrodes_contributions )
			position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])

			position_true_mean = np.nanmean( behavior_data['Positions'][reduce(np.intersect1d,
				(np.where(behavior_data['Position_time'][:,0] > bin_start_time),
				np.where(behavior_data['Position_time'][:,0] < bin_stop_time),
				np.where(behavior_data['Speed'][:,0] > speed_cut)))], axis=0 )
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
		return position_proba, position_true, nSpikes



