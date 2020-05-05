import numpy as np
from sklearn.neighbors import KernelDensity

class ClusterReader():
	def __init__(self, cluReader, resReader, samplingRate):
		self.cluReader = cluReader
		self.resReader = resReader
		self.samplingRate = samplingRate

	def getNext(self):
		self.clu = int(self.cluReader.readline())
		self.res = float(self.resReader.readline())/self.samplingRate


def kde2D(x, y, bandwidth, xbins=50j, ybins=50j, **kwargs):
	"""Build 2D kernel density estimate (KDE)."""

	kernel       = kwargs.get('kernel',       'epanechnikov')
	if ('edges' in kwargs):
		xx = kwargs['edges'][0]
		yy = kwargs['edges'][1]
	else:
		# create grid of sample locations (default: 45x45)
		xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]


	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T

	kde_skl = KernelDensity(kernel=kernel, bandwidth=bandwidth)
	kde_skl.fit(xy_train)

	# score_samples() returns the log-likelihood of the samples
	z = np.exp(kde_skl.score_samples(xy_sample))
	zz = np.reshape(z, xx.shape)
	return xx, yy, zz/np.sum(zz)

