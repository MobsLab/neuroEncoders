# Load libs
import sys
import os
import tables
import struct
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
	
	return {'Positions': positions, 'Position_time': position_time, 'Speed': speed, 'Times': {'start': start_time, 'stop': stop_time, 'end': end_time, 'learning': learning_time}}


def getSpikesfromClu(projectPath, behavior_data, cluster_modifier=1):
	# Get parameters
	list_channels, samplingRate, _ = rawDataParser.get_params(projectPath.xml)
	
	# Allocate
	labels = []
	spike_time = []
	spike_positions = []
	spike_speed = []

	n_tetrodes = len(list_channels)
	for tetrode in range(n_tetrodes+1):
		if os.path.isfile(projectPath.clu(tetrode)):
			with open(
					projectPath.clu(tetrode), 'r') as fClu, open(
					projectPath.basename + 'res.' + str(tetrode), 'r') as fRes, open(
					projectPath.basename + 'spk.' + str(tetrode), 'rb') as fSpk:
				clu_str = fClu.readlines()
				res_str = fRes.readlines()
				n_clu = int(clu_str[0])-1
				n_channels = len(list_channels[tetrode])
				spikeReader = struct.iter_unpack(str(32*n_channels)+'h', fSpk.read())

				# labels = np.array([[1. if int(clu_str[n+1])-1==l else 0. for l in range(n_clu)] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
				# spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
				labels = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(n_clu+1)] for n in range(len(clu_str)-1)])
				spike_time.append(np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1)]))
				spike_positions.append(np.array([behavior_data['Positions'][np.argmin(np.abs(spike_time[n]-behavior_data['Position_time'])),:] for n in range(len(spike_time))]))
				spike_speed.append(np.array([behavior_data['Speed'][np.min((np.argmin(np.abs(spike_time[n]-behavior_data['Position_time'])),
					len(behavior_data['Speed'])-1)),:] for n in range(len(spike_time))]))
		else:
			print("File "+ projectPath.clu(tetrode) + 'clu.' + str(tetrode) +" not found.")
		continue
		sys.stdout.write('File from tetrode ' + str(tetrode) + ' has been succesfully opened. ')
		sys.stdout.write('Processing ...')
		sys.stdout.write('\r')
		sys.stdout.flush()
		
		labels.append(butils.modify_labels(labels, cluster_modifier))
		
		sys.stdout.write('We have finished building rates for group ' + str(tetrode) + ', loading next                           ')
		sys.stdout.write('\r')
		sys.stdout.flush()
	sys.stdout.write('We have finished building rates.                                                           ')
	sys.stdout.write('\r')
	sys.stdout.flush()
 
	return labels, spike_time, spike_positions, spike_time