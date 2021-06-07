# Load libs
import sys
import os
import tables
import struct
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from functools import reduce
from SimpleBayes import butils
from importData import rawDataParser
import tqdm as tqdm
import pandas as pd


def getBehavior(folder, bandwidth=None,getfilterSpeed = True):
	# Extract behavior
	f = tables.open_file(folder + 'nnBehavior.mat')
	positions = f.root.behavior.positions
	speed = f.root.behavior.speed
	position_time = f.root.behavior.position_time
	positions = np.swapaxes(positions[:,:],1,0)
	speed = np.swapaxes(speed[:,:],1,0)
	position_time = np.swapaxes(position_time[:,:],1,0)

	sleepPeriods = f.root.behavior.sleepPeriods[:]
	sleepNames = ["".join([chr(c) for c in l]) for l in f.root.behavior.sessionSleepNames[:]]

	#  Pierre 04/04/2021: train Epochs should be a continuous array, and test epochs too.
	# odd indices indicate end of training,
	if len(f.root.behavior.trainEpochs.shape) == 2:
		trainEpochs = np.concatenate(f.root.behavior.trainEpochs)
		testEpochs = np.concatenate(f.root.behavior.testEpochs)
		lossPredSetEpochs = np.concatenate(f.root.behavior.lossPredSetEpochs)
	elif len(f.root.behavior.trainEpochs.shape) == 1:
		trainEpochs = f.root.behavior.trainEpochs[:]
		testEpochs= f.root.behavior.testEpochs[:]
		lossPredSetEpochs = f.root.behavior.lossPredSetEpochs[:]
	else:
		raise Exception("bad train and test epochs format in mat file")
	if bandwidth == None:
		goodRecordingTimeStep = np.logical_not(np.isnan(np.sum(positions,axis=1)))
		bandwidth = (np.max(positions[goodRecordingTimeStep,:]) - np.min(positions[goodRecordingTimeStep,:]))/15


	if getfilterSpeed:
		speedFilter = f.root.behavior.speedMask[:]
		samplingWindowPosition = (position_time[1:] - position_time[0:-1])[:,0]

		#NaN positions will not be used in learning; therefore we need to remove them from
		# the computation of the learning time:
		samplingWindowPosition[np.isnan(np.sum(positions[0:-1],axis=1))] = 0

		#find index of at which epochs begin:
		learning_epoch_index = [[np.argmin(np.abs(position_time-trainEpochs[2 * i + 1])),
								 np.argmin(np.abs(position_time-trainEpochs[2 * i]))] for i in range(len(trainEpochs) // 2)]
		learning_time = [np.sum(np.multiply(speedFilter[learning_epoch_index[i][1]:learning_epoch_index[i][0]],
								samplingWindowPosition[learning_epoch_index[i][1]:learning_epoch_index[i][0]])) for i in range(len(learning_epoch_index))]
		learning_time = np.sum(learning_time)
		behavior_data = {'Positions': positions, 'Position_time': position_time, 'Speed': speed, 'Bandwidth': bandwidth,
			'Times': {'trainEpochs': trainEpochs, 'testEpochs': testEpochs, 'learning': learning_time, "speedFilter":speedFilter,
					  'lossPredSetEpochs': lossPredSetEpochs, 'sleepEpochs':sleepPeriods, 'sleepNames':sleepNames}}
	else:
		learning_time = np.sum([trainEpochs[2 * i + 1] - trainEpochs[2 * i] for i in range(len(trainEpochs) // 2)])

		#NaN positions will not be used in learning; therefore we need to remove them from
		# the computation of the learning time:
		samplingWindowPosition = (position_time[1:] - position_time[0:-1])[:, 0]
		learning_time = learning_time - np.sum(samplingWindowPosition[np.isnan(np.sum(positions[0:-1], axis=1))])

		behavior_data = {'Positions': positions, 'Position_time': position_time, 'Speed': speed, 'Bandwidth': bandwidth,
						 'Times': {'trainEpochs': trainEpochs, 'testEpochs': testEpochs, 'learning': learning_time,
								   'lossPredSetEpochs': lossPredSetEpochs,'sleepEpochs':sleepPeriods, 'sleepNames':sleepNames}}
	f.close()
	return behavior_data


def findTime(position_time, lastBestTime, time):
	for i in range(len(position_time) - lastBestTime - 1):
		if np.abs(position_time[lastBestTime + i] - time) < np.abs(position_time[lastBestTime + i + 1] - time):
			return lastBestTime + i
	return len(position_time) - 1

def getSpikesfromClu(projectPath, behavior_data, cluster_modifier=1, savedata=True):
	# Get parameters
	list_channels, samplingRate, _ = rawDataParser.get_params(projectPath.xml)

	# Allocate
	labels = []
	spike_time = []
	spike_positions = []
	spike_speed = []
	spike_pos_index = []

	n_tetrodes = len(list_channels)
	for tetrode in tqdm.tqdm(range(n_tetrodes)):
		if os.path.isfile(projectPath.clu(tetrode)):
			with open(
					projectPath.clu(tetrode), 'r') as fClu, open(
					projectPath.res(tetrode), 'r') as fRes : #open(projectPath.spk(tetrode), 'rb') as fSpk

				clu_str = fClu.readlines()
				res_str = fRes.readlines()
				n_clu = int(clu_str[0])-1

				# Clusters only with labels >= 1
				labels_temp = butils.modify_labels(np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(1, n_clu+1)] for n in range(len(clu_str)-1)]), cluster_modifier)
				st = (np.array([[float(res_str[n])/samplingRate] for n in tqdm.tqdm(range(len(clu_str)-1))]))

				# Efficient wqy to get the closest position_time to each spike time:
				lastBestId = 0
				posID = []
				for n in tqdm.tqdm(range(len(st))):
					lastBestId = findTime(behavior_data['Position_time'], lastBestId, st[n])
					posID += [lastBestId]

				sp = behavior_data['Positions'][posID]
				newposId = np.array(posID)
				newposId[np.where(np.array(posID)>len(behavior_data['Speed'])-1)[0]] = len(behavior_data['Speed'])-1
				ss = behavior_data['Speed'][newposId,:]

				spike_time.append(st)
				spike_positions.append(sp)
				spike_pos_index.append(np.array(posID))
				spike_speed.append(ss)
				labels.append(labels_temp)
		else:
			print("File "+ projectPath.clu(tetrode) +" not found.")
			continue
		sys.stdout.write('File from tetrode '+ ' has been successfully opened. ')
		sys.stdout.write('Processing ...')
		sys.stdout.write('\r')
		sys.stdout.flush()

		sys.stdout.write('We have finished building rates for group ' + str(tetrode+1) + ', loading next                           ')
		sys.stdout.write('\r')
		sys.stdout.flush()
	sys.stdout.write('We have imported clusters.                                                           ')
	sys.stdout.write('\r')
	sys.stdout.flush()

	cluster_data = {'Spike_labels': labels, 'Spike_times': spike_time, 'Spike_positions': spike_positions, 'Spike_speed': spike_speed}
	if savedata:
		cluster_save_path = os.path.join(projectPath.folder,"dataset","clusterData")
		if not os.path.isdir(cluster_save_path):
			os.makedirs(cluster_save_path)
		for l in range(len(labels)):
			df = pd.DataFrame(labels[l])
			df.to_csv(os.path.join(cluster_save_path,"Spike_labels"+str(l)+".csv"))
			df = pd.DataFrame(spike_time[l])
			df.to_csv(os.path.join(cluster_save_path,"spike_time"+str(l)+".csv"))
			df = pd.DataFrame(spike_positions[l])
			df.to_csv(os.path.join(cluster_save_path,"spike_positions"+str(l)+".csv"))
			df = pd.DataFrame(spike_pos_index[l])
			df.to_csv(os.path.join(cluster_save_path, "spike_pos_index" + str(l) + ".csv"))
			df = pd.DataFrame(spike_speed[l])
			df.to_csv(os.path.join(cluster_save_path,"spike_speed"+str(l)+".csv"))
		# np.save(projectPath.folder + 'ClusterData.npy', cluster_data)
	return cluster_data

def load_spike_sorting(projectPath):
	cluster_save_path = os.path.join(projectPath.folder,"dataset","clusterData")
	if os.path.isfile(os.path.join(cluster_save_path, 'Spike_labels0.csv')):
		cluster_data = {"Spike_labels": [], "Spike_times": [], "Spike_positions": [], "Spike_speed": [],
						"Spike_pos_index": []}
		print("Reading saved cluster csv file")
		for l in tqdm.tqdm(range(4)):
			df = pd.read_csv(os.path.join(cluster_save_path, "Spike_labels" + str(l) + ".csv"))
			cluster_data["Spike_labels"].append(df.values[:, 1:])
			df = pd.read_csv(os.path.join(cluster_save_path, "spike_time" + str(l) + ".csv"))
			cluster_data["Spike_times"].append(df.values[:, 1:])
			df = pd.read_csv(os.path.join(cluster_save_path, "spike_positions" + str(l) + ".csv"))
			cluster_data["Spike_positions"].append(df.values[:, 1:])
			df = pd.read_csv(os.path.join(cluster_save_path, "spike_pos_index" + str(l) + ".csv"))
			cluster_data["Spike_pos_index"].append(df.values[:, 1:])
			df = pd.read_csv(os.path.join(cluster_save_path, "spike_speed" + str(l) + ".csv"))
			cluster_data["Spike_speed"].append(df.values[:, 1:])

		print("finished reading")
	else:
		behavior_data = getBehavior(projectPath.folder, getfilterSpeed=False)
		cluster_data = getSpikesfromClu(projectPath, behavior_data)
	return cluster_data