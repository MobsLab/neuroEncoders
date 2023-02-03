# Load libs
import sys
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from functools import reduce
from simpleBayes import butils
from importData import rawdata_parser
import tqdm as tqdm
import pandas as pd


def getSpikesfromClu(projectPath, behavior_data, cluster_modifier=1, savedata=True):
	# Get parameters
	listChannels, samplingRate, _ = rawdata_parser.get_params(projectPath.xml)
	# Allocate
	labels = []
	spikeTime = []
	spikePositions = []
	spikeSpeed = []
	spikePosIndex = []

	nTetrodes = len(listChannels)
	for tetrode in tqdm.tqdm(range(nTetrodes)):
		print(f'Importing sorted spikes from Neuroscope files from electrodes group #{tetrode}')
		if os.path.isfile(projectPath.clu(tetrode)):
			with open(
					projectPath.clu(tetrode), 'r') as fClu, open(
					projectPath.res(tetrode), 'r') as fRes : #open(projectPath.spk(tetrode), 'rb') as fSpk

				cluStr = fClu.readlines()
				resStr = fRes.readlines()
				nClu = int(cluStr[0])-1
				# Clusters only with labels >= 1
				# otherwise all labels are set to 0 (represent cluster 0 - NOISE)
				labels_temp = butils.modify_labels(np.array([[1. if int(cluStr[n+1])==l 
                                                  else 0. for l in range(1, nClu+1)] 
                                                 for n in range(len(cluStr)-1)]), cluster_modifier)
				st = (np.array([[float(resStr[n])/samplingRate] 
                    for n in tqdm.tqdm(range(len(cluStr)-1))]))

				# Efficient way to get the closest position_time to each spike time:
				lastBestId = 0
				posID = []
				for n in tqdm.tqdm(range(len(st))):
					lastBestId = rawdata_parser.findTime(behavior_data['positionTime'], 
                                          lastBestId, st[n])
					posID += [lastBestId]

				sp = behavior_data['Positions'][posID]
				newposId = np.array(posID)
				newposId[np.where(np.array(posID)>len(behavior_data['Speed'])-1)[0]] = len(behavior_data['Speed'])-1
				ss = behavior_data['Speed'][newposId,:]

				spikeTime.append(st)
				spikePositions.append(sp)
				spikePosIndex.append(np.array(posID))
				spikeSpeed.append(ss)
				labels.append(labels_temp)
		else:
			print("File "+ projectPath.clu(tetrode) +" not found.")
			continue
		sys.stdout.write('File from tetrode '+ ' has been successfully opened. ')
		sys.stdout.write('Processing ...')
		sys.stdout.write('\r')
		sys.stdout.flush()

		sys.stdout.write('We have finished building rates for group ' + str(tetrode+1) + 
                   		', loading next                           ')
		sys.stdout.write('\r')
		sys.stdout.flush()
	sys.stdout.write('We have imported clusters.                                                           ')
	sys.stdout.write('\r')
	sys.stdout.flush()

	cluster_data = {'Spike_labels': labels, 'Spike_times': spikeTime, 
                 	'Spike_positions': spikePositions, 'Spike_speed': spikeSpeed}
	if savedata:
		cluster_save_path = os.path.join(projectPath.folder,"dataset","clusterData")
		if not os.path.isdir(cluster_save_path):
			os.makedirs(cluster_save_path)
		for l in range(len(labels)):
			df = pd.DataFrame(labels[l])
			df.to_csv(os.path.join(cluster_save_path,"Spike_labels"+str(l)+".csv"))
			df = pd.DataFrame(spikeTime[l])
			df.to_csv(os.path.join(cluster_save_path,"spike_time"+str(l)+".csv"))
			df = pd.DataFrame(spikePositions[l])
			df.to_csv(os.path.join(cluster_save_path,"spike_positions"+str(l)+".csv"))
			df = pd.DataFrame(spikePosIndex[l])
			df.to_csv(os.path.join(cluster_save_path, "spike_pos_index" + str(l) + ".csv"))
			df = pd.DataFrame(spikeSpeed[l])
			df.to_csv(os.path.join(cluster_save_path,"spike_speed"+str(l)+".csv"))
	return cluster_data

def load_spike_sorting(projectPath):
	cluster_save_path = os.path.join(projectPath.folder,"dataset","clusterData")
	if os.path.isfile(os.path.join(cluster_save_path, 'Spike_labels0.csv')):
		lfiles = glob.glob(os.path.join(cluster_save_path, 'Spike_labels*.csv'))
		num_files = len(lfiles)
		cluster_data = {"Spike_labels": [], "Spike_times": [], "Spike_positions": [], "Spike_speed": [],
						"Spike_pos_index": []}
		print("Reading saved cluster csv file")
		for l in tqdm.tqdm(range(num_files)):
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
		behavior_data = rawdata_parser.get_behavior(projectPath.folder, getfilterSpeed=False)
		cluster_data = getSpikesfromClu(projectPath, behavior_data)
	return cluster_data






