## Pierre 26/03

import tensorflow as tf
import numpy as np

# OutputEmbeding:
# --> Constructs a place-cell like language: maps an environment variable (here position) to a high-dimensional space
# where each axis is a cell firing rate.
# Each cell is preferentially tuned to a particular point in the output space (here position)
# Axes in output space are assumed to be orthogonal such that the multi-dim tuning curve is an uncorrelated Gaussian
# Further refinement of this work may use more complex tuning curves as output.
# A good pipeline is the following:  spike sort to discover the tuning curves shape
# Then fit with simpler model and use these target tuning curve model as output decoding target for the transformer.

# To pave uniformly the input space we make sure that the model dimension (i.e the number of place cells)
#  n_output_variable (for example 2 for a 2d positional encoding)-root is an integer.
class OutputLanguageModel:
    def __init__(self, d_model,n_output_variable,batchsize=1):
        assert d_model**(1/n_output_variable)//1 == d_model**(1/n_output_variable)
        bin_per_feature = int(d_model**(1/n_output_variable)//1)
        coordinate_arrays = [np.linspace(0,stop=1,num=bin_per_feature)  for _ in range(n_output_variable)]

        prefered_feature = np.zeros((d_model,n_output_variable))
        for i in range(n_output_variable):
            for n in range(d_model):
                    prefered_feature[n,i] = coordinate_arrays[i][(n%(bin_per_feature**(i+1)))//(bin_per_feature**(i))]
        # Proof:
        # n = i + j*bin + h*bin^2 + .... <-- can be seen as recursive euclidean division, or decomposition in base "bin"
        # -> i = n%bin
        #  j = (n%bin^2)//bin
        # h = (n%bin^3)//(bin^2)

        self.prefered_feature_tensor =  tf.constant(prefered_feature,shape=[1,d_model,n_output_variable],dtype=tf.float32)
        # activity of the cell
        # self.A = tf.constant(np.random.lognormal(0,1,[d_model]),dtype=tf.float32)
        self.A = tf.constant(1,shape=[d_model],dtype=tf.float32)
        self.sigma = tf.constant(0.05,shape=[d_model],dtype=tf.float32)

        self.prefered_feature_tensor = tf.stack([self.prefered_feature_tensor for _ in range(batchsize)])
        self.sigma = tf.stack([self.sigma for _ in range(batchsize)])
        self.A = tf.stack([self.A for _ in range(batchsize)])
        self.sigma = tf.expand_dims(self.sigma,axis=1)
        self.A = tf.expand_dims(self.A,axis=1)

    def __call__(self,output):
        # Need to deal with the case of output in form of a sequence:
        # [batch_size,seq_len,n_output_variable]
        output = tf.expand_dims(output,axis=2)
        activity = tf.multiply(self.A,tf.exp(-tf.reduce_sum(tf.square(output - self.prefered_feature_tensor),axis=-1)/self.sigma))
        return activity

    def get_word(self,output):
        # Simply an argmax of the activity to find the most active cells
        # we need to scale them by their activity....
        activity = self(output)
        word = tf.argmax(activity/self.A,axis=-1)
        return word
    #
    # def get_word_weight(self,output):
    #     # We would like to weight the loss
    #     # by the distance of the predicted place cells
    #     # to the real electrode.
    #     #   Right now the prediction is solely based on false negative.
    #
    #     # But probably the first thing to try is to use a cosine similarity...
    #
    #     return word_weight
