# Created by Pierre, 26/03/2021

import tensorflow as tf
import numpy as np


class waveFormConvNet(tf.keras.layers.Layer):
    def __init__(self, nChannels=4, nFeatures=128):
        super(waveFormConvNet, self).__init__()
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        # modified by Pierre 16/02/2021: migration to 2.0 of RNN
        self.convLayer1 = tf.keras.layers.Conv2D(8, [2, 3], padding='SAME')
        # conv2D: 8 := filters , dimensionality of output space: number of output filters
        #		  [2,3  ]: kernel_size: height, width of 2D convolution
        #         padding="same" (case-insensitive): padding evenly to the left or up/down so that output
        # 			has same size as input....

        self.convLayer2 = tf.keras.layers.Conv2D(16, [2, 3], padding='SAME')
        self.convLayer3 = tf.keras.layers.Conv2D(32, [2,3], padding='SAME') #change from 32 to 16 and [2;3] to [2;2]

        self.maxPoolLayer1 = tf.keras.layers.MaxPool2D([1, 2], [1, 2], padding='SAME')
        self.maxPoolLayer2 = tf.keras.layers.MaxPool2D([1, 2], [1, 2], padding='SAME')
        self.maxPoolLayer3 = tf.keras.layers.MaxPool2D([1,2], [1,2], padding='SAME')

        self.dropoutLayer = tf.keras.layers.Dropout(0.5)
        self.denseLayer1 = tf.keras.layers.Dense(self.nFeatures, activation='relu')
        self.denseLayer2 = tf.keras.layers.Dense(self.nFeatures, activation='relu')
        self.denseLayer3 = tf.keras.layers.Dense(self.nFeatures, activation='relu')

    def __call__(self, input, training=True):
        x = tf.expand_dims(input, axis=3)
        x = self.convLayer1(x)
        x = self.maxPoolLayer1(x)
        x = self.convLayer2(x)
        x = self.maxPoolLayer2(x)
        x = self.convLayer3(x)
        x = self.maxPoolLayer3(x)

        x = tf.reshape(x, [-1, self.nChannels * 8 * 16])  # change from 32 to 16 and 4 to 8
        # by pooling we moved from 32 bins to 4. By convolution we generated 32 channels
        x = self.denseLayer1(x)
        x = self.dropoutLayer(x)
        x = self.denseLayer2(x)
        x = self.denseLayer3(x)
        return x
