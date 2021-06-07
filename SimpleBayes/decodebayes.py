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

from tqdm import tqdm

from importData.rawDataParser import inEpochs

from pykeops.numpy import LazyTensor as LazyTensor_np
import pykeops as pykeops


# trainer class
class Trainer():
	def __init__(self, projectPath, bandwidth=None, kernel='gaussian'): # 'epanechnikov'
		self.projectPath = projectPath
		self.bandwidth = bandwidth
		self.kernel = kernel

	def train(self, behavior_data, cluster_data, speed_cut=0, masking_factor=20, rand_param=0):
		# Check for bandwidth
		if self.bandwidth == None:
			self.bandwidth = behavior_data['Bandwidth']

		# Way to determine a good estimate of the number of bins needed:

		# meanMI = []
		# for nbin in tqdm(np.arange(10,stop=200,step=10)): #np.arange(0.01,stop=0.1,step=0.01)
		# 	meanMI.append(self.bandwidth_cv(behavior_data,cluster_data,self.bandwidth,nbins=[nbin,nbin]))
		# import matplotlib.pyplot as plt
		# fig,ax = plt.subplots()
		# ax.plot(np.arange(10,stop=200,step=10),meanMI)
		# ax.set_xlabel("number of bins")
		# ax.set_ylabel("Mutual Information")
		# fig.show()
		nbins = [80,80]

		#Note: we initially thought to do the same with the bandwith, but the MI estimate is decreasing as the bandwith grow larger...

		### GLOBAL OCCUPATION
		selected_positions = behavior_data['Positions'][reduce(np.intersect1d,
			(np.where(behavior_data['Times']['speedFilter']),
			inEpochs(behavior_data['Position_time'][:,0], behavior_data['Times']['trainEpochs']),
			))]
		# Remove NaN: i:e problem with feature recording
		selected_positions = selected_positions[np.logical_not(np.isnan(np.sum(selected_positions,axis=1))),:]
		# xEdges, yEdges, Occupation = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, kernel=self.kernel)
		gridFeature, Occupation = butils.kdenD(selected_positions, 0.07,kernel=self.kernel,nbins=nbins) #0.07
		# Occupation = butils.hist2D(selected_positions)
		Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

		mask = Occupation > (np.max(Occupation)/masking_factor)
		Occupation_inverse = 1/Occupation
		Occupation_inverse[Occupation_inverse==np.inf] = 0
		Occupation_inverse = np.multiply(Occupation_inverse, mask)

		### Built spike-related kernels
		guessed_clusters_time = []
		guessed_clusters = []

		sleep_clusters_dic = {}
		sleep_clusters_time_dic = {}

		Marginal_rate_functions = []
		Rate_functions = []
		Spike_positions = []
		Mutual_info = []

		n_tetrodes = len(cluster_data['Spike_labels'])
		for tetrode in tqdm(range(n_tetrodes)):
			### MARGINAL RATE FUNCTION
			selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
				(np.where(behavior_data['Times']['speedFilter']),
				inEpochs(cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
			# Remove NaN: i:e problem with feature recording
			selected_positions = selected_positions[np.logical_not(np.isnan(np.sum(selected_positions, axis=1))), :]
			# xEdges, yEdges, MRF = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
			gridFeature, MRF = butils.kdenD(selected_positions, self.bandwidth,
											   edges=gridFeature, kernel=self.kernel,nbins=nbins)
			MRF[MRF==0] = np.min(MRF[MRF!=0])
			MRF         = MRF/np.sum(MRF)
			MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/behavior_data['Times']['learning']
			Marginal_rate_functions.append(MRF)

			Local_rate_functions = []
			Local_Spike_positions = []
			LocalMutualInfo = []
		### LOCAL RATE FUNCTION FOR EACH CLUSTER
			for label in range(np.shape(cluster_data['Spike_labels'][tetrode])[1]):
				selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
					(np.where(behavior_data['Times']['speedFilter']),
					np.where(cluster_data['Spike_labels'][tetrode][:,label] == 1),
					inEpochs(cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
				if np.shape(selected_positions)[0]!=0:
					# xEdges, yEdges, LRF =  butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
					gridFeature, LRF = butils.kdenD(selected_positions, self.bandwidth,
													edges=gridFeature, kernel=self.kernel,nbins=nbins)
					LRF[LRF==0] = np.min(LRF[LRF!=0])
					LRF         = LRF/np.sum(LRF)
					LRF         = np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/behavior_data['Times']['learning']
					Local_rate_functions.append(LRF)
				else:
					Local_rate_functions.append(np.ones(np.shape(Occupation)))
				Local_Spike_positions.append(selected_positions)
				#Let us compute the mutual information with the positions:
				LRF = Local_rate_functions[-1]
				mutualInfo = np.sum(Occupation[LRF>0]*LRF[LRF>0]/(np.mean(LRF))*np.log(LRF[LRF>0]/(np.mean(LRF)))/np.log(2))
				LocalMutualInfo.append(mutualInfo)

			Rate_functions.append(Local_rate_functions)
			Spike_positions.append(Local_Spike_positions)
			Mutual_info.append(LocalMutualInfo)

		#given a n-dim grid, let us find the binning over each edges
		bayes_matrices = {'Occupation': Occupation, 'Marginal rate functions': Marginal_rate_functions, 'Rate functions': Rate_functions,
				'Bins':[np.unique(gridFeature[i]) for i in range(len(gridFeature))],
				'guessed_clusters_info': {'clusters':guessed_clusters, 'time':guessed_clusters_time},'Spike_positions':Spike_positions,
						  'Mutual_info': Mutual_info
				}
		#'time_limits': [behavior_data['Times']['start_train'], behavior_data['Times']['stop_train'],
				                # behavior_data['Times']['start_test'], behavior_data['Times']['stop_test']]
		return bayes_matrices



	def bandwidth_cv(self,behavior_data,cluster_data,bandwidth,masking_factor=20,nbins=[45,45]):
		# this function evaluates the mean mutual information of rate field estimated given the bandwith parameter.

		selected_positions = behavior_data['Positions'][reduce(np.intersect1d,
			(np.where(behavior_data['Times']['speedFilter']),
			inEpochs(behavior_data['Position_time'][:,0], behavior_data['Times']['trainEpochs']),
			))]
		# Remove NaN: i:e problem with feature recording
		selected_positions = selected_positions[np.logical_not(np.isnan(np.sum(selected_positions,axis=1))),:]
		# xEdges, yEdges, Occupation = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, kernel=self.kernel)
		#Remark: todo optimize for the bandwith
		gridFeature, Occupation = butils.kdenD(selected_positions, 0.07,kernel=self.kernel,nbins=nbins)
		# Occupation = butils.hist2D(selected_positions)
		# Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros , why????

		mask = Occupation > (np.max(Occupation)/masking_factor)
		Occupation_inverse = 1/Occupation
		Occupation_inverse[Occupation_inverse==np.inf] = 0
		Occupation_inverse = np.multiply(Occupation_inverse, mask)

		### Built spike-related kernels
		Marginal_rate_functions = []
		Mutual_info = []

		n_tetrodes = len(cluster_data['Spike_labels'])
		for tetrode in range(n_tetrodes):
			LocalMutualInfo = []
		### LOCAL RATE FUNCTION FOR EACH CLUSTER
			for label in range(np.shape(cluster_data['Spike_labels'][tetrode])[1]):
				selected_positions = cluster_data['Spike_positions'][tetrode][reduce(np.intersect1d,
					(np.where(behavior_data['Times']['speedFilter']),
					np.where(cluster_data['Spike_labels'][tetrode][:,label] == 1),
					inEpochs(cluster_data['Spike_times'][tetrode][:,0], behavior_data['Times']['trainEpochs'])))]
				if np.shape(selected_positions)[0]!=0:
					# xEdges, yEdges, LRF =  butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
					gridFeature, LRF = butils.kdenD(selected_positions, bandwidth,
													edges=gridFeature, kernel=self.kernel,nbins=nbins)
					LRF[LRF==0] = np.min(LRF[LRF!=0])
					LRF         = LRF/np.sum(LRF)
					LRF         = np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/behavior_data['Times']['learning']
					mutualInfo = np.sum(Occupation[LRF > 0] * LRF[LRF > 0] / (np.mean(LRF)) * np.log(
						LRF[LRF > 0] / (np.mean(LRF))) / np.log(2))
				else:
					LRF = np.ones(np.shape(Occupation))
					mutualInfo = np.sum(Occupation[LRF > 0] * LRF[LRF > 0] / (np.mean(LRF)) * np.log(
						LRF[LRF > 0] / (np.mean(LRF))) / np.log(2))
				LocalMutualInfo.append(mutualInfo)
			Mutual_info.append(np.mean(LocalMutualInfo))
		return np.mean(Mutual_info)

	def test(self, bayes_matrices, behavior_data, windowSize=0.036, masking_factor=20):

		print('\nBUILDING POSITION PROBAS')

		### Format matrices
		guessed_clusters_info = bayes_matrices['guessed' \
											   '_clusters_info']
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

			#Trouble: the test Epochs is discretized in continuous bin
			# whereas we forbid the use of some time steps b filtering them according to speed.
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

	def test_Pierre(self, bayes_matrices, behavior_data,cluster_data, windowSize=0.036, masking_factor=20):


		guessed_clusters_time = [cluster_data['Spike_times'][tetrode][
										 inEpochs(cluster_data['Spike_times'][tetrode][:, 0],
												  behavior_data['Times']['testEpochs'])]
								 for tetrode in range(len(cluster_data['Spike_times']))]
		guessed_clusters= [ cluster_data['Spike_labels'][tetrode][
														  inEpochs(cluster_data['Spike_times'][tetrode][:, 0],
																   behavior_data['Times']['testEpochs'])]
							for tetrode in range(len(cluster_data['Spike_times']))]

		print('\nBUILDING POSITION PROBAS')

		Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in
															   ['Occupation', 'Marginal rate functions',
																'Rate functions']]
		mask = Occupation > (np.max(Occupation) / masking_factor)

		### Build Poisson term
		All_Poisson_term = [np.exp((-windowSize) * Marginal_rate_functions[tetrode]) for tetrode in
							range(len(guessed_clusters))]
		All_Poisson_term = reduce(np.multiply, All_Poisson_term)

		### Log of rate functions
		log_RF = []
		for tetrode in range(np.shape(Rate_functions)[0]):
			temp = []
			for cluster in range(np.shape(Rate_functions[tetrode])[0]):
				temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(
					Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster] != 0])))
			log_RF.append(temp)

		timeStepPred = behavior_data["Position_time"][inEpochs(behavior_data["Position_time"][:,0],behavior_data['Times']['testEpochs'])]
		# For prediction time step we use the time step of measured speed and feature of the animal.
		# A prediction is made over all time steps in the test set
		# and the prediction results can be latter on filter out.
		n_bins = timeStepPred.shape[0]

		### Decoding loop
		position_proba = [np.ones(np.shape(Occupation))] * n_bins
		nSpikes = []

		# quick speed improvement: provide a start and stop position index, for each tetrode, used to quickly find the guessed_clusters_time
		# which are in the considered bins
		startArray = np.zeros(len(guessed_clusters),dtype=np.int64)
		stopArray = np.zeros(len(guessed_clusters),dtype=np.int64)

		# For now the pykeops speed increase is not working, trouble with the install....
		print("Parallel pykeops bayesian test")
		outputPos_pykeops = parallelize_prediction( timeStepPred, windowSize, All_Poisson_term, guessed_clusters,
							   guessed_clusters_time, log_RF)
		print("finished bayesian guess")

		# for bin in range(n_bins):
		#
		# 	# Trouble: the test Epochs is discretized in continuous bin
		# 	# whereas we forbid the use of some time steps b filtering them according to speed.
		# 	bin_start_time = timeStepPred[bin] - windowSize/2
		# 	bin_stop_time = bin_start_time + windowSize/2
		# 	#Question: use window around or solely after ?
		#
		# 	binSpikes = 0
		# 	tetrodes_contributions = []
		# 	tetrodes_contributions.append(All_Poisson_term)
		# 	for tetrode in range(len(guessed_clusters)):
		# 		# Find clusters inside our window
		# 		# startArray[tetrode],stopArray[tetrode],bin_probas = find_next_bin(guessed_clusters_time[tetrode],guessed_clusters[tetrode],startArray[tetrode],stopArray[tetrode],bin_start_time,bin_stop_time)
		#
		# 		bin_probas = guessed_clusters[tetrode][np.intersect1d(
		# 			np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
		# 			np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time))]
		# 		#Note:  we would lose some spikes if we used the cluster_data[Spike_pos_index]
		# 		# because some spike might be closest to one position further away than windowSize, yet themselves be close to the spike time
		#
		# 		bin_clusters = np.sum(bin_probas, 0)
		# 		binSpikes = binSpikes + np.sum(bin_clusters)
		#
		# 		# Terms that come from spike information
		# 		if np.sum(bin_clusters) > 0.5:
		# 			spike_pattern = reduce(np.multiply,
		# 								   [np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
		# 									for cluster in range(np.shape(bin_clusters)[0])])
		# 		else:
		# 			spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)
		#
		# 		tetrodes_contributions.append(spike_pattern)
		#
		# 	nSpikes.append(binSpikes)
		#
		# 	# Guessed probability map
		# 	position_proba[bin] = reduce(np.multiply, tetrodes_contributions)
		# 	position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])
		#
		# 	if bin % 50 == 0:
		# 		sys.stdout.write('[%-30s] : %.3f %%' % ('=' * (bin * 30 // n_bins), bin * 100 / n_bins))
		# 		sys.stdout.write('\r')
		# 		sys.stdout.flush()
		# sys.stdout.write('[%-30s] : %.3f %%' % ('=' * ((bin + 1) * 30 // n_bins), (bin + 1) * 100 / n_bins))
		# sys.stdout.write('\r')
		# sys.stdout.flush()
		#
		# print('\nDecoding finished')
		# # Guessed X and Y

		# # Pierre: correction, here we should not marginalize over x and y to find the best predicted proba
		# allProba = [np.unravel_index(np.argmax(position_proba[bin]), position_proba[bin].shape) for bin in
		# 			range(len(nSpikes))]
		# bestProba = [np.max(position_proba[bin]) for bin in range(len(nSpikes))]
		# position_guessed = [[bayes_matrices['Bins'][i][allProba[bin][i]] for i in range(len(bayes_matrices['Bins']))]
		# 					for bin in range(len(nSpikes))]
		# inferResults = np.concatenate([np.array(position_guessed), np.array(bestProba).reshape([-1, 1])], axis=-1)
		#
		# # let us compare pykeops and iterative positions predictions:
		# pos_guess = [np.argmax(position_proba[bin]) for bin in
		# 			range(len(nSpikes))]

		tpl_position_guessed_pykeops = np.unravel_index(outputPos_pykeops[1],shape=All_Poisson_term.shape)
		pos_guess_pykeops = np.array([bayes_matrices['Bins'][i][tpl_position_guessed_pykeops[i][:,0]] for i in range(len(bayes_matrices['Bins']))])
		proba_guess_pykeops = outputPos_pykeops[0]
		inferResults_pykeops  = np.concatenate([np.transpose(pos_guess_pykeops),proba_guess_pykeops], axis=-1)

		# import matplotlib.pyplot as plt
		# fig,ax = plt.subplots(3)
		# for i in range(3):
		# 	ax[i].plot(inferResults[:,i],c="red")
		# 	ax[i].plot(inferResults_pykeops[:,i],c="black")
		# 	ax[i].set_yscale("log")
		# fig.show()

		#NOTE: A few values of probability predictions present NaN in pykeops....
		print("Resolving nan issue from pykeops over a few bins")
		badBins = np.where(np.isnan(inferResults_pykeops[:,2]))[0]
		for bin in badBins:
			bin_start_time = timeStepPred[bin] - windowSize / 2
			bin_stop_time = bin_start_time + windowSize / 2
			binSpikes = 0
			tetrodes_contributions = []
			tetrodes_contributions.append(All_Poisson_term)
			for tetrode in range(len(guessed_clusters)):
				bin_probas = guessed_clusters[tetrode][np.intersect1d(
					np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
					np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time))]
				# Note:  we would lose some spikes if we used the cluster_data[Spike_pos_index]
				# because some spike might be closest to one position further away than windowSize, yet themselves be close to the spike time
				bin_clusters = np.sum(bin_probas, 0)
				# Terms that come from spike information
				if np.sum(bin_clusters) > 0.5:
					spike_pattern = reduce(np.multiply,
										   [np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
											for cluster in range(np.shape(bin_clusters)[0])])
				else:
					spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)
				tetrodes_contributions.append(spike_pattern)
			# Guessed probability map
			position_proba = reduce(np.multiply, tetrodes_contributions)
			position_proba= position_proba / np.sum(position_proba)
			inferResults_pykeops[bin,2] = np.max(position_proba)

		# fig,ax = plt.subplots(3)
		# for i in range(3):
		# 	ax[i].plot(inferResults[:,i],c="red")
		# 	ax[i].plot(inferResults_pykeops[:,i],c="black")
		# 	ax[i].set_yscale("log")
		# fig.show()
		outputResults = {"inferring": inferResults_pykeops, 'nSpikes': np.array(nSpikes)} #, "probaMaps": position_proba
		return outputResults

	def sleep_decoding(self, bayes_matrices, behavior_data, cluster_data, windowSize=0.036, masking_factor=20):
		guessed_clusters_time = {}
		guessed_clusters = {}
		inferResults_dic = {}
		for id,sleepName in enumerate(behavior_data["Times"]["sleepNames"]):
			guessed_clusters_time = [cluster_data['Spike_times'][tetrode][
										 inEpochs(cluster_data['Spike_times'][tetrode][:, 0],
												  behavior_data['Times']['sleepEpochs'][2*id:2*id+2])]
									 for tetrode in range(len(cluster_data['Spike_times']))]
			guessed_clusters = [cluster_data['Spike_labels'][tetrode][
									inEpochs(cluster_data['Spike_times'][tetrode][:, 0],
											 behavior_data['Times']['sleepEpochs'][2*id:2*id+2])]
								for tetrode in range(len(cluster_data['Spike_times']))]

			print('\nBUILDING POSITION PROBAS')

			Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in
																   ['Occupation', 'Marginal rate functions',
																	'Rate functions']]
			mask = Occupation > (np.max(Occupation) / masking_factor)

			### Build Poisson term
			All_Poisson_term = [np.exp((-windowSize) * Marginal_rate_functions[tetrode]) for tetrode in
								range(len(guessed_clusters))]
			All_Poisson_term = reduce(np.multiply, All_Poisson_term)

			### Log of rate functions
			log_RF = []
			for tetrode in range(np.shape(Rate_functions)[0]):
				temp = []
				for cluster in range(np.shape(Rate_functions[tetrode])[0]):
					temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(
						Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster] != 0])))
				log_RF.append(temp)

			timeStepPred = behavior_data["Position_time"][
				inEpochs(behavior_data["Position_time"][:, 0],behavior_data['Times']['sleepEpochs'][2*id:2*id+2])]
			# For prediction time step we use the time step of measured speed and feature of the animal.
			# A prediction is made over all time steps in the test set
			# and the prediction results can be latter on filter out.

			print("Parallel pykeops bayesian test")
			outputPos_pykeops = parallelize_prediction(timeStepPred, windowSize, All_Poisson_term, guessed_clusters,
													   guessed_clusters_time, log_RF)
			print("finished bayesian guess")

			tpl_position_guessed_pykeops = np.unravel_index(outputPos_pykeops[1], shape=All_Poisson_term.shape)
			pos_guess_pykeops = np.array([bayes_matrices['Bins'][i][tpl_position_guessed_pykeops[i][:, 0]] for i in
										  range(len(bayes_matrices['Bins']))])
			proba_guess_pykeops = outputPos_pykeops[0]
			inferResults_pykeops = np.concatenate([np.transpose(pos_guess_pykeops), proba_guess_pykeops], axis=-1)

			# NOTE: A few values of probability predictions present NaN in pykeops....
			print("Resolving nan issue from pykeops over a few bins")
			badBins = np.where(np.isnan(inferResults_pykeops[:, 2]))[0]
			for bin in badBins:
				bin_start_time = timeStepPred[bin] - windowSize / 2
				bin_stop_time = bin_start_time + windowSize / 2
				binSpikes = 0
				tetrodes_contributions = []
				tetrodes_contributions.append(All_Poisson_term)
				for tetrode in range(len(guessed_clusters)):
					bin_probas = guessed_clusters[tetrode][np.intersect1d(
						np.where(guessed_clusters_time[tetrode][:, 0] > bin_start_time),
						np.where(guessed_clusters_time[tetrode][:, 0] < bin_stop_time))]
					# Note:  we would lose some spikes if we used the cluster_data[Spike_pos_index]
					# because some spike might be closest to one position further away than windowSize, yet themselves be close to the spike time
					bin_clusters = np.sum(bin_probas, 0)
					# Terms that come from spike information
					if np.sum(bin_clusters) > 0.5:
						spike_pattern = reduce(np.multiply,
											   [np.exp(log_RF[tetrode][cluster] * bin_clusters[cluster])
												for cluster in range(np.shape(bin_clusters)[0])])
					else:
						spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)
					tetrodes_contributions.append(spike_pattern)
				# Guessed probability map
				position_proba = reduce(np.multiply, tetrodes_contributions)
				position_proba = position_proba / np.sum(position_proba)
				inferResults_pykeops[bin, 2] = np.max(position_proba)
			inferResults_dic[sleepName] = [inferResults_pykeops[:,0:2],inferResults_pykeops[:,2],timeStepPred[:,0]]

		return inferResults_dic

def find_next_bin(times,clusters,start,stop,start_time,stop_time):
	# times: array of times
	# start: last start index
	# stop: last stop index
	# start_time: bin start time
	# stop_time: bin stop time
	newStartID = stop
	while times[newStartID]<start_time:
		newStartID += 1
	newStopId = newStartID+1
	while times[newStopId]<stop_time and newStopId<=len(times)-1:
		newStopId +=1
	newStopId = newStopId-1
	return newStartID,newStopId,clusters[newStartID:newStopId+1]

def parallelize_prediction(timeStepPred,windowSize,All_Poisson_term,guessed_clusters,guessed_clusters_time,log_RF):
	# Use pykeops library to perform an efficient computation of the predicted position, in parallel over all bins.
	# Note: here achieved on the CPU, could also be ported to the GPU by using torch tensor....

	bin_start_times = timeStepPred - windowSize/2
	bin_stop_times = bin_start_times + windowSize/2

	# we will progressively add each tetrode contribution
	tetrode_contribs = 1
	for tetrode in range(len(guessed_clusters)):

		gct_lazy = LazyTensor_np(guessed_clusters_time[tetrode][:,None])
		bin_start_times_lazy = LazyTensor_np(bin_start_times[None,:])
		bin_stop_times_lazy = LazyTensor_np(bin_stop_times[None,:])
		good_start = (gct_lazy - bin_start_times_lazy).relu().sign() # similar to gct_lazy > bin_start_times.lazy
		good_stop = (bin_stop_times_lazy - gct_lazy).relu().sign()
		# size: (Number of signal time step,Number of prediction bin,1), indicate for each bin the time step in the bin.
		good_bins = good_start*good_stop
		gc_lazy = LazyTensor_np(guessed_clusters[tetrode][:,None,:])
		# For each bin, we gather for each cluster in the tetrode the number of spike detected in signal measurements inside this bin.
		# gathering can be effectively implemented by a element wise matrix multiplication with the mask good_bins
		bin_clusters = (gc_lazy*good_bins).sum(axis=0)
		# # transform into an array of size (Nb bin,Nb cluster in tetrode)

		# Prepare for pykeops operations:
		log_RF_reshape = np.transpose(np.array(log_RF[tetrode]), axes=[1, 2, 0])
		log_RF_reshape = np.reshape(log_RF_reshape,newshape=[np.prod(log_RF_reshape.shape[0:-1]),log_RF_reshape.shape[-1]])
		log_RF_lazy = LazyTensor_np(log_RF_reshape[None,:,:])
		bin_clusters_lazy = LazyTensor_np(bin_clusters[:,None,:])

		# the Log firing rate of each cluster is multiplied by the number of bin cluster, and the sum is performed over the
		# number of cluster in the tetrode
		res = (log_RF_lazy*bin_clusters_lazy).sum(dim=-1).exp()
		tetrode_contribs = tetrode_contribs*res

	#Finally we need to add the Poisson terms common to all tetrode finalS
	# position posterior estimation:
	poisson_reshape = np.reshape(All_Poisson_term,newshape=[np.prod(All_Poisson_term.shape)])[:,None]
	poisson_contrib_Vj = pykeops.numpy.Vj(poisson_reshape)
	tetrode_contribs = tetrode_contribs*poisson_contrib_Vj

	# If we had only one electrode:
	# ... but we need to sum over the different electrodes.
	outputPos = tetrode_contribs.max_argmax_reduction(axis=1)
	# We also need to normalize the probability:
	sumProba = tetrode_contribs.sum_reduction(axis=1)
	outputPos = (outputPos[0]/sumProba,outputPos[1])

	return outputPos