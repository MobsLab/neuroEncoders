import sys
import os
import re

import tables
import struct
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mplt

from interval import interval

import csv


########### Management of epochs ############

# a few tool to help do difference of intervals as sets:
def obtainCloseComplementary(epochs,boundInterval):
	# we obtain the close complementary (intervals share their bounds )
	# Note: to obtain the open complementary, one would need to add some dt to end and start of intervals....
	p1 = interval()
	for i in range(len(epochs)//2):
		p1 = p1 | interval([epochs[2*i],epochs[2*i+1]])
	assert isinstance(p1,interval)
	assert isinstance(boundInterval,interval)
	assert len(boundInterval)==1
	compInterval = interval([boundInterval[0][0],p1[0][0]])
	for i in range(len(p1)-1):
		compInterval = compInterval | interval([p1[i][1],p1[i+1][0]])
	compInterval = compInterval |  interval([p1[-1][1],boundInterval[0][1]])
	return compInterval

def intersect_with_session(epochs,keptSession,starts,stops):
	# we go through the different removed session epoch, and if a train epoch or a test epoch intersect with it we remove it
	# from the train and test epochs
	EpochInterval = interval()
	for i in range(len(epochs) // 2):
		EpochInterval = EpochInterval | interval([epochs[2 * i], epochs[2 * i + 1]])
	includeInterval = interval()
	for id, keptS in enumerate(keptSession):
		if keptS:
			includeInterval = includeInterval | interval([starts[id], stops[id]])
	EpochInterval = EpochInterval & includeInterval
	Epoch = np.ravel(np.array([[p[0], p[1]] for p in EpochInterval]))
	return Epoch

# Auxilliary function
def get_epochs(postime, SetData, keptSession, starts=np.empty(0), stops=np.empty(0)):
			# given the slider values, as well as the selected session, we extract the different sets
			# if starts and stops (of epochs) are present, it means we work with multi-recording

		pmin = postime[0]
		pmax = postime[-1]
  
		testEpochs = np.array(
			[postime[SetData['testSetId']],
				postime[min(SetData['testSetId'] + SetData['sizeTestSet'], postime.shape[0] - 1)]])

		if SetData['useLossPredTrainSet']:
			lossPredSetEpochs = np.array([postime[SetData['lossPredSetId']],
											postime[
												min(SetData['lossPredSetId'] + SetData['sizeLossPredSet'], postime.shape[0] - 1)]])
			lossPredsetinterval = interval(lossPredSetEpochs)
			lossPredsetinterval = lossPredsetinterval & obtainCloseComplementary(testEpochs, interval([pmin, pmax]))
			lossPredSetEpochs = np.ravel(np.array([[p[0], p[1]] for p in lossPredsetinterval]))

			trainInterval = obtainCloseComplementary(testEpochs, interval([pmin, pmax])) & obtainCloseComplementary(lossPredSetEpochs, interval([pmin, pmax]))
		else:
			trainInterval = obtainCloseComplementary(testEpochs, interval([pmin, pmax]))

		trainEpoch = np.ravel(np.array([[p[0], p[1]] for p in trainInterval]))

		if starts.size > 0:
			trainEpoch = intersect_with_session(trainEpoch, keptSession, starts, stops)
			testEpochs = intersect_with_session(testEpochs, keptSession, starts, stops)
			if SetData['useLossPredTrainSet']:
	   
				lossPredSetEpochs = intersect_with_session(lossPredSetEpochs, keptSession, starts, stops)
				return trainEpoch, testEpochs, lossPredSetEpochs
			else:
				return trainEpoch,testEpochs,None
		else:
			if SetData['useLossPredTrainSet']:
				return trainEpoch, testEpochs, lossPredSetEpochs
			else:
				return trainEpoch,testEpochs,None


def inEpochs(t,epochs):
	# for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
	# test if t is among at least one of these epochs
	# Epochs are treated as closed interval [,]
	# returns the index where it is the case
	mask =  np.sum([(t>=epochs[2*i]) * (t<=epochs[2*i+1]) for i in range(len(epochs)//2)],axis=0)
	return np.where(mask >= 1)
def inEpochsMask(t,epochs):
	# for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
	# test if t is among at least one of these epochs
	# Epochs are treated as closed interval [,]
	# return the mask
	mask =  np.sum([(t>=epochs[2*i]) * (t<=epochs[2*i+1]) for i in range(len(epochs)//2)],axis=0)
	return mask >= 1
