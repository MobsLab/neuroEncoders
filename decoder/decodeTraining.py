import numpy as np
import tensorflow as tf
from tqdm import trange
from fullEncoder import nnUtils

class Trainer():
	def __init__(self, projectPath, params, spikeDetector, device_name="/cpu:0"):
		self.projectPath = projectPath
		self.params = params
		self.device_name = device_name
		self.feat_desc = {"pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), "length": tf.io.FixedLenFeature([], tf.int64), "groups": tf.io.VarLenFeature(tf.int64)}
		for g in range(self.params.nGroups):
			self.feat_desc.update({"group"+str(g): tf.io.VarLenFeature(tf.float32)})

	def train(self):
		return []
	def test(self):
		### Loading and inferring
		print()
		print("INFERRING")

		tf.contrib.rnn
		with tf.Graph().as_default(), tf.device("/cpu:0"):

			dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["test"])
			cnt     = dataset.batch(1).repeat(1).reduce(np.int64(0), lambda x, _: x + 1)
			dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals))
			iter    = dataset.make_initializable_iterator()
			spikes  = iter.get_next()
			for group in range(self.params.nGroups):
				idMatrix = tf.eye(tf.shape(spikes["groups"])[0])
				completionTensor = tf.transpose(tf.gather(idMatrix, tf.where(tf.equal(spikes["groups"], group)))[:,0,:], [1,0], name="completion")
				spikes["group"+str(group)] = tf.tensordot(completionTensor, spikes["group"+str(group)], axes=[[1],[0]])


			saver = tf.train.import_meta_graph(self.projectPath.graphMeta)


			with tf.Session() as sess:
				saver.restore(sess, self.projectPath.graph)

				pos = []
				inferring = []
				probaMaps = []
				sess.run(iter.initializer)
				for b in trange(cnt.eval()):
					tmp = sess.run(spikes)
					pos.append(tmp["pos"])
					temp = sess.run(
							[tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0"),
							tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionGuessed:0"),
							tf.get_default_graph().get_tensor_by_name("bayesianDecoder/standardDeviation:0")],
							{tf.get_default_graph().get_tensor_by_name("group"+str(group)+"-encoder/x:0"):tmp["group"+str(group)]
								for group in range(self.params.nGroups)}) 
					inferring.append(np.concatenate([temp[1],temp[2]], axis=0))
					probaMaps.append(temp[0])

				pos = np.array(pos)

		return {"inferring":np.array(inferring), "pos":pos, "probaMaps": np.array(probaMaps)}
