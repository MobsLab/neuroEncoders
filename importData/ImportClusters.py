# Load libs
import sys
import os
import datetime
import tables
import struct
import math
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing as ml
from tqdm import tqdm
import subprocess
from scipy import signal
from functools import reduce
from sklearn.neighbors import KernelDensity
from bayesian_utils import butils


def get_params(pathToXml):
	list_channels = []
	try:
		tree = ET.parse(pathToXml)
	except:
		print("impossible to open xml file:", pathToXml)
		sys.exit(1)
	root = tree.getroot()
	for br1Elem in root:
		if br1Elem.tag != 'spikeDetection':
			continue
		for br2Elem in br1Elem:
			if br2Elem.tag != 'channelGroups':
				continue
			for br3Elem in br2Elem:
				if br3Elem.tag != 'group':
					continue
				group=[];
				for br4Elem in br3Elem:
					if br4Elem.tag != 'channels':
						continue
					for br5Elem in br4Elem:
						if br5Elem.tag != 'channel':
							continue
			if br2Elem.tag == 'samplingRate':
				samplingRate  = float(br2Elem.text)
			if br2Elem.tag == 'nChannels':
				nChannels = int(br2Elem.text)

	return list_channels, samplingRate, nChannels


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


def getSpikesfromClu(behavior_data, samplingRate, cluster_modifier=1):
	# Allocate
	labels = []
	spike_time = []
	spike_positions = []
	spike_speed = []
		
	tetrode_names = [keys for keys in list_channels.keys()]
	for tetrode in tetrode_names:
		if os.path.isfile(clu_path + 'clu.' + str(tetrode)):
			with open(
					clu_path + 'clu.' + str(tetrode), 'r') as fClu, open(
					clu_path + 'res.' + str(tetrode), 'r') as fRes, open(
					clu_path + 'spk.' + str(tetrode), 'rb') as fSpk:
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
			print("File "+ clu_path + 'clu.' + str(tetrode) +" not found.")
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
 
