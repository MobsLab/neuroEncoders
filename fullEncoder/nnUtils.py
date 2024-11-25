# Load libs
import numpy as np
import tensorflow as tf


########### CONVOLUTIONAL NETWORK CLASS #####################
class spikeNet:
    """
    This class is a convolutional network that takes as input a spike sequence and returns a feature vector of size nFeatures.

    args:
    nChannels: number of channels in the input
    device: the device on which the network is run
    nFeatures: the size of the output feature vector
    number: a number to identify the network

    Details of the default network:
    The network is composed of 3 convolutional layers followed by 3 max pooling layers. The convolutional layers have 8, 16 and 32 filters of size 2x3. The max pooling layers have a pool size of 1x2.
    The convolutional layers are followed by 3 dense layers with a ReLU activation function. The dense layers have a size of nFeatures and the last dense layer has a size of nFeatures and is named "outputCNN{number}".
    """

    def __init__(
        self,
        nChannels=4,
        device: str = "/cpu:0",
        nFeatures=128,
        number="",
    ):
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        self.device = device
        with tf.device(self.device):
            self.convLayer1 = tf.keras.layers.Conv2D(8, [2, 3], padding="SAME")
            self.convLayer2 = tf.keras.layers.Conv2D(16, [2, 3], padding="SAME")
            self.convLayer3 = tf.keras.layers.Conv2D(32, [2, 3], padding="SAME")

            self.maxPoolLayer1 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="SAME"
            )
            self.maxPoolLayer2 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="SAME"
            )
            self.maxPoolLayer3 = tf.keras.layers.MaxPool2D(
                [1, 2], [1, 2], padding="SAME"
            )

            self.dropoutLayer = tf.keras.layers.Dropout(0.5)
            self.denseLayer1 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer2 = tf.keras.layers.Dense(self.nFeatures, activation="relu")
            self.denseLayer3 = tf.keras.layers.Dense(
                self.nFeatures, activation="relu", name=f"outputCNN{number}"
            )

    def __call__(self, input):
        return self.apply(input)

    def apply(self, input):
        with tf.device(self.device):
            x = tf.expand_dims(input, axis=3)
            x = self.convLayer1(x)
            x = self.maxPoolLayer1(x)
            x = self.convLayer2(x)
            x = self.maxPoolLayer2(x)
            x = self.convLayer3(x)
            x = self.maxPoolLayer3(x)

            x = tf.reshape(
                x, [-1, self.nChannels * 8 * 16]
            )  # change from 32 to 16 and 4 to 8
            # by pooling we moved from 32 bins to 4. By convolution we generated 32 channels
            x = self.denseLayer1(x)
            x = self.dropoutLayer(x)
            x = self.denseLayer2(x)
            x = self.denseLayer3(x)
        return x

    def variables(self):
        return (
            self.convLayer1.variables
            + self.convLayer2.variables
            + self.convLayer3.variables
            + self.maxPoolLayer1.variables
            + self.maxPoolLayer2.variables
            + self.maxPoolLayer3.variables
            + self.denseLayer1.variables
            + self.denseLayer2.variables
            + self.denseLayer3.variables
        )


########### CONVOLUTIONAL NETWORK CLASS #####################


########### SPIKE STORAGE AND PARCING FUNCTIONS #####################
def get_spike_sequences(params, generator):
    # WARNING: This function is actually not used in the code, it might be a helper function to understand the pipeline of the spike sequence??
    """
    Warning: This function is not used in the code.
    Could be used in the main neuroEncoder function to get the Spike sequence from the spike generator
    and cast it into an "example" format that will then be decoded by tensorflow inputs system tf.io as the key word yield is used, this function effectively returns a generator.

    The goal of the function is to bin the set of spikes with respect to times, gather spikes in time windows of fix length.

    args:
    params: the parameters of the network
    generator: the generator that yields the spikes
    """

    windowStart = None

    length = 0
    times = []
    groups = []
    allSpikes = [
        [] for _ in range(params.nGroups)
    ]  # nGroups of array each containing the spike of a group
    for pos_index, grp, time, spike, pos in generator:
        if windowStart is None:
            windowStart = (
                time  # at the first pass: initialize the windowStart on "time"
            )

        if time > windowStart + params.windowLength:
            # if we got over the window-length
            allSpikes = [
                np.zeros([0, params.nChannelsPerGroup[g], 32])
                if allSpikes[g] == []
                else np.stack(allSpikes[g], axis=0)
                for g in range(params.nGroups)
            ]  # stacks each list of array in allSpikes
            # allSpikes then is composed of nGroups array of stacked "spike"
            res = {
                "pos_index": pos_index,
                "pos": pos,
                "groups": groups,
                "length": length,
                "times": times,
            }
            res.update({"spikes" + str(g): allSpikes[g] for g in range(params.nGroups)})
            yield res
            # increase the windowStart by one window length
            length = 0
            groups = []
            times = []
            allSpikes = [
                [] for _ in range(params.nGroups)
            ]  # The all Spikes is reset so that we stop gathering the spikes in this window
            windowStart += params.windowLength
            # Pierre: Then we increment the windowStart until it is above the last seen spike time
            while time > windowStart + params.windowLength:
                # res = {"train": train, "pos": pos, "groups": [], "length": 0, "times": []}
                # res.update({"spikes"+str(g): np.zeros([0, params.nChannels[g], 32]) for g in range(params.nGroups)})
                # yield res
                windowStart += params.windowLength
        # Pierre: While we have not entered a new window, we start to gather spikes, time and group
        # of each input.
        times.append(time)
        groups.append(grp)
        # Pierre: so here we understand that groups indicate for each spikes array
        # obtained from the generator the groups from which they belong to !
        # But the spike array are well mapped separately to different groups:
        allSpikes[grp].append(spike)
        length += 1
        # --> so length correspond to the number of spike sequence obtained from the generator for each window considered


def serialize_spike_sequence(params, pos_index, pos, groups, length, times, *spikes):
    """
    Moves from the info obtained via the SpikeDetector -> spikeGenerator -> getSpikeSequences pipeline toward the tensorflow storing file.
    This take a specific format, which is here declared through the dict+tf.train.Feature organisation. We see that groups now correspond to the "spikes" we had before....
    """

    feat = {
        "pos_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[pos_index])),
        "pos": tf.train.Feature(float_list=tf.train.FloatList(value=pos)),
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[length])),
        "groups": tf.train.Feature(int64_list=tf.train.Int64List(value=groups)),
        "time": tf.train.Feature(float_list=tf.train.FloatList(value=[np.mean(times)])),
    }
    # Pierre: convert the spikes dict into a tf.train.Feature, used for the tensorflow protocol.
    # their is no reason to change the key name but still done here.
    for g in range(params.nGroups):
        feat.update(
            {
                "group"
                + str(g): tf.train.Feature(
                    float_list=tf.train.FloatList(value=spikes[g].ravel())
                )
            }
        )

    example_proto = tf.train.Example(features=tf.train.Features(feature=feat))
    return example_proto.SerializeToString()  # to string


def serialize_single_spike(clu, spike):
    feat = {
        "clu": tf.train.Feature(int64_list=tf.train.Int64List(value=[clu])),
        "spike": tf.train.Feature(float_list=tf.train.FloatList(value=spike.ravel())),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feat))
    return example_proto.SerializeToString()


# @tf.function
def parse_serialized_sequence(params, tensors, batched=False):  # featDesc, ex_proto,
    tensors["groups"] = tf.sparse.to_dense(tensors["groups"], default_value=-1)
    # Pierre 13/02/2021: Why use sparse.to_dense, and not directly a FixedLenFeature?
    # Probably because he wanted a variable length <> inputs sequences
    tensors["groups"] = tf.reshape(tensors["groups"], [-1])

    tensors["indexInDat"] = tf.sparse.to_dense(tensors["indexInDat"], default_value=-1)
    tensors["indexInDat"] = tf.reshape(tensors["indexInDat"], [-1])

    for g in range(params.nGroups):
        # here 32 correspond to the number of discretized time bin for a spike
        zeros = tf.constant(np.zeros([params.nChannelsPerGroup[g], 32]), tf.float32)
        tensors["group" + str(g)] = tf.sparse.reshape(tensors["group" + str(g)], [-1])
        tensors["group" + str(g)] = tf.sparse.to_dense(tensors["group" + str(g)])
        tensors["group" + str(g)] = tf.reshape(tensors["group" + str(g)], [-1])
        if batched:
            tensors["group" + str(g)] = tf.reshape(
                tensors["group" + str(g)],
                [params.batchSize, -1, params.nChannelsPerGroup[g], 32],
            )
        # even if batched: gather all together
        tensors["group" + str(g)] = tf.reshape(
            tensors["group" + str(g)], [-1, params.nChannelsPerGroup[g], 32]
        )
        # Pierre 12/03/2021: the batchSize and timesteps are gathered together
        nonZeros = tf.logical_not(
            tf.equal(
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        tf.equal(tensors["group" + str(g)], zeros), tf.int32
                    ),
                    axis=[1, 2],
                ),
                32 * params.nChannelsPerGroup[g],
            )
        )
        # nonZeros: control that the voltage measured is not 0, at all channels and time bin inside the detected spike
        tensors["group" + str(g)] = tf.gather(
            tensors["group" + str(g)], tf.where(nonZeros)
        )[:, 0, :, :]
        # I don't understand why it can then call [:,0,:,:] as the output tensor of gather should have the same
        # shape as tensors["group"+str(g)"], [-1,params.nChannels[g],32] ...

    return tensors


def parse_serialized_spike(featDesc, ex_proto, batched=False):
    if batched:
        tensors = tf.io.parse_example(serialized=ex_proto, features=featDesc)
    else:
        tensors = tf.io.parse_single_example(serialized=ex_proto, features=featDesc)
    return tensors


########### SPIKE STORAGE AND PARCING FUNCTIONS #####################


def import_true_pos(feature):
    def change_feature(vals):
        vals["pos"] = tf.gather(feature, vals["pos_index"])
        return vals

    return change_feature
