import numpy as np
import tensorflow as tf




def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def deconv2d(x, W, output_shape):
	return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,1,2,1], padding='SAME')
def max_pool_1x2(x):
	return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
def lrelu(x,alpha=0.1):
	return tf.maximum(alpha*x,x)




def dropout_layer(x, keep_proba):

	Dropout                             = {}
	Dropout['Input']                    = x
	Dropout['keep_proba']               = keep_proba

	Dropout['Output']                   = tf.nn.dropout(Dropout['Input'], Dropout['keep_proba'])

	return Dropout


def convolution_layer(x, shape):

	Convolution                         = {}
	Convolution['Input']                = x
	Convolution['W_conv']               = weight_variable(shape)
	Convolution['b_conv']               = bias_variable([shape[-1]])

	# Convolution['h_conv']               = tf.nn.leaky_relu(conv2d(Convolution['Input'], Convolution['W_conv']) + Convolution['b_conv'])
	Convolution['h_conv']               = tf.nn.relu(conv2d(Convolution['Input'], Convolution['W_conv']) + Convolution['b_conv'])
	Convolution['Output']               = max_pool_1x2(Convolution['h_conv'])

	return Convolution


def deconvolution_layer(x, output_shape, filter_shape):

	deConvolution                         = {}
	deConvolution['Input']                = x
	deConvolution['W_conv']               = weight_variable(filter_shape)
	deConvolution['b_conv']               = bias_variable([filter_shape[-2]])

	deConvolution['Output']               = tf.nn.relu(deconv2d(deConvolution['Input'], deConvolution['W_conv'], output_shape) + deConvolution['b_conv'])

	return deConvolution


def connected_layer(x, size):
	input_size  = size[0]
	output_size = size[1]

	ConnectedLayer                      = {}
	ConnectedLayer['Input']             = x
	ConnectedLayer['W_fc']              = weight_variable([tf.cast(input_size, tf.int32), tf.cast(output_size, tf.int32)])
	ConnectedLayer['b_fc']              = bias_variable([output_size])

	ConnectedLayer['PreActivation']     = tf.matmul(ConnectedLayer['Input'], ConnectedLayer['W_fc']) + ConnectedLayer['b_fc']
	# ConnectedLayer['Output']            = tf.nn.leaky_relu(ConnectedLayer['PreActivation'])
	ConnectedLayer['Output']            = tf.nn.relu(ConnectedLayer['PreActivation'])

	return ConnectedLayer









def three_convolution(x):

	Graph                              = {}


	Graph['Convolution1']              = convolution_layer(       x,                                  [2,3,1,8])
	Graph['Convolution2']              = convolution_layer(       Graph['Convolution1']['Output'],    [2,3,8,16])
	Graph['Convolution3']              = convolution_layer(       Graph['Convolution2']['Output'],    [2,3,16,32])

	return Graph


def three_deconvolution(x):

	batch_size                           = tf.shape(x)[0]
	Graph                                = {}


	Graph['deConvolution1']              = deconvolution_layer(       x,                                    tf.stack([batch_size,4,8,16]),          [1,2,16,32])
	Graph['deConvolution2']              = deconvolution_layer(       Graph['deConvolution1']['Output'],    tf.stack([batch_size,4,16,8]),          [1,2,8,16])
	Graph['deConvolution3']              = deconvolution_layer(       Graph['deConvolution2']['Output'],    tf.stack([batch_size,4,32,1]),          [1,2,1,8])

	return Graph


def three_fullconnected(x,nClusters,keep_proba,size=512):

	Graph                              = {}

	Input                              = tf.reshape(                        x,                                  [-1,16*32])
	Graph['ConnectedLayer1']           = connected_layer(                   Input,                              [np.shape(Input)[1], size])
	Graph['Dropout']                   = dropout_layer(                     Graph['ConnectedLayer1']['Output'], keep_proba)
	Graph['ConnectedLayer2']           = connected_layer(                   Graph['Dropout']['Output'],         [size, size])
	Graph['ConnectedLayer3']           = connected_layer(                   Graph['ConnectedLayer2']['Output'], [size, nClusters])

	return Graph











def encoder(x,nClusters,keep_proba, **kwargs):

	size                               = kwargs.get('size',          512)

	Graph                              = {}
	Graph['Input']                     = tf.expand_dims(x,           axis=3)

	Graph.update(                        three_convolution(          Graph['Input']))
	Graph.update(                        three_fullconnected(        Graph['Convolution3']['Output'], nClusters, keep_proba, size=size))

	Graph['Output']                    = Graph['ConnectedLayer3']['PreActivation']


	return Graph



def layeredEncoder(input, nClusters, nChannels, **kwargs):
	size = kwargs.get('size', 512)

	# # keeping tensorflow 1.x interface
	# input = tf.expand_dims(x, axis=3)
	# conv1 = tf.layers.conv2d(input, 8, [2,3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv1')
	# mp1 = tf.layers.max_pooling2d(conv1, [1,2], [1,2], padding='SAME')
	# conv2 = tf.layers.conv2d(mp1, 16, [2,3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv2')
	# mp2 = tf.layers.max_pooling2d(conv2, [1,2], [1,2], padding='SAME')
	# conv3 = tf.layers.conv2d(mp2, 32, [2,3], padding='SAME', reuse=tf.AUTO_REUSE, name='conv3')
	# mp3 = tf.layers.max_pooling2d(conv3, [1,2], [1,2], padding='SAME')

	# dense0 = tf.reshape(mp3, [-1, nChannels*4*32])
	# dense1 = tf.layers.dense(dense0, size, activation=tf.nn.relu)
	# dense1 = tf.layers.dropout(dense1, rate=0.5)
	# dense2 = tf.layers.dense(dense1, size, activation=tf.nn.relu)
	# result = tf.layers.dense(dense2, nClusters, activation=None)


	# towards tensorflow 2.x interface
	convLayer1 = tf.layers.Conv2D(8, [2,3], padding='SAME')
	convLayer2 = tf.layers.Conv2D(16, [2,3], padding='SAME')
	convLayer3 = tf.layers.Conv2D(32, [2,3], padding='SAME')

	x = tf.expand_dims(input, axis=3)
	x = convLayer1(x)
	x = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')(x)
	x = convLayer2(x)
	x = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')(x)
	x = convLayer3(x)
	x = tf.layers.MaxPooling2D([1,2], [1,2], padding='SAME')(x)

	x = tf.reshape(x, [-1, nChannels*4*32])
	x = tf.layers.Dense(size, activation=tf.nn.relu)(x)
	x = tf.layers.Dropout(0.5)(x)
	x = tf.layers.Dense(size, activation=tf.nn.relu)(x)
	x = tf.layers.Dense(nClusters, activation=None)(x)
	result = x


	# Should return a tensor
	return result, [convLayer1, convLayer2, convLayer3]



