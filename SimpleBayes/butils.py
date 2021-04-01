import numpy as np
import random
from scipy import signal
from functools import reduce
from sklearn.neighbors import KernelDensity


def exp256(x):

	"""This function is supposedly 330 times faster than a classic exponential.
	This is an approximation, theoretically very good if x<5.
	It was also tested to be working normally here."""

	temp = 1 + x/1024
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	return temp
	# return np.exp(x)



def epanech_kernel_1d(size_kernel):
	values = np.ones(2*size_kernel)
	for n in range(-size_kernel, size_kernel):
		values[n+size_kernel] = 3/4/size_kernel*(1-(pow(n+1,3) - pow(n,3))/3/size_kernel**2)
	return values


def epanech_kernel_2d(size_kernel):
	kernel_1d = epanech_kernel_1d(size_kernel)
	return np.outer(kernel_1d, kernel_1d)


def kde2D(x, y, bandwidth, xbins=45j, ybins=45j, **kwargs):
	"""Build 2D kernel density estimate (KDE)."""

	kernel       = kwargs.get('kernel',       'epanechnikov')
	if ('edges' in kwargs):
		xx = kwargs['edges'][0]
		yy = kwargs['edges'][1]
	else:
		# create grid of sample locations (default: 150x150)
		xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]


	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T

	kde_skl = KernelDensity(kernel=kernel, bandwidth=bandwidth)
	kde_skl.fit(xy_train)

	# score_samples() returns the log-likelihood of the samples
	z = np.exp(kde_skl.score_samples(xy_sample))
	zz = np.reshape(z, xx.shape)
	return xx, yy, zz/np.sum(zz)


def be_sure_about(bin_probas):
	"""Turns an array of probabilities into an array of zeros and ones"""

	truth = np.zeros(np.shape(bin_probas))
	for event in range(np.shape(bin_probas)[0]):
		truth[event, np.argmax(bin_probas[event,:])] = 1
	return truth


# ########## Data processing

def modify_labels(labels,clu_modifier=1):

	if clu_modifier==1 :
		return labels

	elif clu_modifier==2 :
		n = len(labels)
		idx = np.concatenate(([0]*(n//2), [1]*(n-n//2)))
		random.shuffle(idx)
		cluA = np.array([labels[label,:]*   idx[label]  for label in range(len(idx))])
		cluB = np.array([labels[label,:]*(1-idx[label]) for label in range(len(idx))])
		return np.concatenate((cluA,cluB),1)

	elif clu_modifier==0.5 :
		n = len(labels)
		n_clu = np.shape(labels)[1]
		idx = list(range(n_clu))
		labels2 = np.ndarray([n, (n_clu+1)//2])
		labels2 = [[(labels[x,idx[2*n]] + labels[x,idx[2*n+1]]
			if (2*n+1<=len(idx)-1)
			else labels[x,idx[2*n]])
			for n in range(np.shape(labels2)[1])]
			for x in range(n)]
		return np.array(labels2)


def shuffle_labels(labels, perc=0.5):
	n = int(len(labels)*perc)
	data = list(labels)
	idx = list(range(len(data)))
	random.shuffle(idx)
	idx = idx[:n]
	mapping = dict((idx[i], idx[i-1]) for i in range(n))
	return np.array([data[mapping.get(x,x)] for x in range(len(data))])