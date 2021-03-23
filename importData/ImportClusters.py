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


def getBehavior(folder, bandwidth=None):
	# Extract behavior
	f = tables.open_file(folder + 'nnBehavior.mat')
	positions = f.root.behavior.positions
	speed = f.root.behavior.speed
	position_time = f.root.behavior.position_time
	positions = np.swapaxes(positions[:,:],1,0)
	speed = np.swapaxes(speed[:,:],1,0)
	position_time = np.swapaxes(position_time[:,:],1,0)

	start_time = f.root.behavior.trainEpochs[:,0]
	stop_time = f.root.behavior.trainEpochs[:,1]
	end_time = f.root.behavior.testEpochs[:,1]
	if bandwidth == None:
		bandwidth = (np.max(positions) - np.min(positions))/20
	learning_time = stop_time - start_time

	behavior_data = {'Positions': positions, 'Position_time': position_time, 'Speed': speed, 'Bandwidth': bandwidth,
		'Times': {'start': start_time, 'stop': stop_time, 'end': end_time, 'learning': learning_time}}

	return behavior_data


def getSpikesfromClu(projectPath, behavior_data, cluster_modifier=1, savedata=True):
	# Get parameters
	list_channels, samplingRate, _ = rawDataParser.get_params(projectPath.xml)

	# Allocate
	labels = []
	spike_time = []
	spike_positions = []
	spike_speed = []

	n_tetrodes = len(list_channels)
	for tetrode in range(n_tetrodes):
		if os.path.isfile(projectPath.clu(tetrode)):
			with open(
					projectPath.clu(tetrode), 'r') as fClu, open(
					projectPath.res(tetrode), 'r') as fRes, open(
					projectPath.spk(tetrode), 'rb') as fSpk:
				clu_str = fClu.readlines()
				res_str = fRes.readlines()
				n_clu = int(clu_str[0])-1
				# n_channels = len(list_channels[tetrode])
				# spikeReader = struct.iter_unpack(str(32*n_channels)+'h', fSpk.read())

				# labels = np.array([[1. if int(clu_str[n+1])-1==l else 0. for l in range(n_clu)] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
				# spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
				labels_temp = butils.modify_labels(np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(n_clu+1)] for n in range(len(clu_str)-1)]), cluster_modifier)
				st = (np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1)]))
				sp = (np.array([behavior_data['Positions'][np.argmin(np.abs(st[n]-behavior_data['Position_time'])),:] for n in range(len(st))]))
				ss = (np.array([behavior_data['Speed'][np.min((np.argmin(np.abs(st[n]-behavior_data['Position_time'])),
					len(behavior_data['Speed'])-1)),:] for n in range(len(st))]))

				spike_time.append(st)
				spike_positions.append(sp)
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
	sys.stdout.write('We have finished building rates.                                                           ')
	sys.stdout.write('\r')
	sys.stdout.flush()

	cluster_data = {'Spike_labels': labels, 'Spike_times': spike_time, 'Spike_positions': spike_positions, 'Spike_speed': spike_speed}
	if savedata:
		np.save(projectPath.folder + 'ClusterData.npy', cluster_data)

	return cluster_data