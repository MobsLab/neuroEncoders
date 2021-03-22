# Load libs
from SimpleBayes import butils
from importData import ImportClusters

import numpy as np
from tqdm import trange
import os
import sys
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

		self.positions = []
		# etc etc etc
  
	def train(self, behavior_data,  tetrode_names, spike_time, spike_speed, speed_cut, start_time, stop_time, end_time, masking_factor, labels, rand_param):
		### GLOBAL OCCUPATION

		selected_positions = behavior_data['Positions'][reduce(np.intersect1d,
			(np.where(behavior_data['Speed'][:,0] > speed_cut),
			np.where(behavior_data['Position_time'][:,0] > start_time),
			np.where(behavior_data['Position_time'][:,0] < stop_time)))]
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

		for tetrode in tetrode_names:
			fake_labels_time.append(spike_time[reduce(np.intersect1d,
				(np.where(spike_time[:,0] > stop_time),
				np.where(spike_time[:,0] < end_time),
				np.where(spike_speed[:,0] > speed_cut)))])
			fake_labels.append(butils.shuffle_labels(labels[reduce(np.intersect1d,
				(np.where(spike_time[:,0] > stop_time),
				np.where(spike_time[:,0] < end_time),
				np.where(spike_speed[:,0] > speed_cut)))], rand_param))
   
		### MARGINAL RATE FUNCTION
		selected_positions = spike_positions[reduce(np.intersect1d, 
			(np.where(spike_speed[:,0] > speed_cut),
			np.where(spike_time[:,0] > start_time),
			np.where(spike_time[:,0] < stop_time)))]
		xEdges, yEdges, MRF = butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
		MRF[MRF==0] = np.min(MRF[MRF!=0])
		MRF         = MRF/np.sum(MRF)
		MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/learning_time
		Marginal_rate_functions.append(MRF)
  
		### LOCAL RATE FUNCTION FOR EACH CLUSTER
		for label in range(np.shape(labels)[1]):
			selected_positions = spike_positions[reduce(np.intersect1d,
				(np.where(spike_speed[:,0] > speed_cut),
				np.where(labels[:,label] == 1),
				np.where(spike_time[:,0] > start_time),
				np.where(spike_time[:,0] < stop_time)))]
			if np.shape(selected_positions)[0]!=0:
				xEdges, yEdges, LRF =  butils.kde2D(selected_positions[:,0], selected_positions[:,1], self.bandwidth, edges=[xEdges,yEdges], kernel=self.kernel)
				LRF[LRF==0] = np.min(LRF[LRF!=0])
				LRF         = LRF/np.sum(LRF)
				LRF         = np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/learning_time
				Local_rate_functions.append(LRF)
			else:
				Local_rate_functions.append(np.ones(np.shape(Occupation)))

		Rate_functions.append(Local_rate_functions)
  
		return {'Ocupation': Occupation, 'Marginal rate functions': Marginal_rate_functions, 'Rate functions': Rate_functions, 'Bins':[xEdges[:,0],yEdges[0,:]], 
				'fake_labels_info': {'clusters':fake_labels, 'time':fake_labels_time}, 'time_limits': [start_time, stop_time]}

  


