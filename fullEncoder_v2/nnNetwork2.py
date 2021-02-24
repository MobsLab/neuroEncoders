import numpy as np
import tensorflow as tf
from fullEncoder_v2 import nnUtils
from tqdm import trange
import os

# Pierre 14/02/01:
# Reorganization of the code:
    # One class for the network
    # One function for the training
# We save the model every epoch during the training

# We generate a model with the functional Model interface in tensorflow

class LSTMandSpikeNetwork():

    def __init__(self, projectPath, params, device_name="/device:gpu:0"):
        super(LSTMandSpikeNetwork, self).__init__()
        self.projectPath = projectPath
        self.params = params
        self.device_name = device_name
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), #target position: current value of the environmental correlate
            "length": tf.io.FixedLenFeature([], tf.int64), #number of spike sequence gathered in the window
            "groups": tf.io.VarLenFeature(tf.int64), # the index of the groups having spike sequences in the window
            "time": tf.io.FixedLenFeature([], tf.float32)}
        # the exact time-steps of each spike measured in the various groups. Question: should the time not be a VarLenFeature??
        for g in range(self.params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})
            # the voltage values (discretized over 32 time bins) of each channel (4 most of the time)
            # of each spike of a given group in the window

        #Loss obtained during training
        self.trainLosses = []

        # TODO: initialization of the networks
        with tf.device(self.device_name):
            self.iteratorLengthInput = tf.keras.layers.Input(shape=(),name="length")
            self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(None,None),name="group"+str(group)) for group in range(self.params.nGroups)]
            self.inputGroups = tf.keras.layers.Input(shape=(),name="groups")

            # The spike nets acts on each group separately; to reorganize all these computations we use
            # an identity matrix which shape is the total number of spike measured (over all groups)
            self.completionMatrix = tf.keras.layers.Input(shape=(None,), name="completionMatrix")
            # What is the role of the completionTensor?
            # The id matrix dimension is the total number of spikes encoded inside the spike window
            # Indeed iterators["groups"] is the list of group that were emitted during each spike sequence merged into the spike window
            self.completionTensors = [tf.transpose(
                a=tf.gather(self.completionMatrix, tf.where(tf.equal(self.inputGroups, g)))[:, 0, :], perm=[1, 0],
                name="completion") for g in range(self.params.nGroups)]
            # The completion Matrix, gather the row of the idMatrix, ie the spike corresponding to the group: group

            # Declare spike nets for the different groups:
            self.spikeNets = [nnUtils.spikeNet(nChannels=self.params.nChannels[group], device=self.device_name,
                                               nFeatures=self.params.nFeatures) for group in range(self.params.nGroups)]
            lstmsNets = [tf.keras.layers.LSTMCell(self.params.lstmSize,recurrent_dropout=self.params.lstmDropout, dtype=tf.float32) for _ in
                         range(self.params.lstmLayers)]
            # LSTM on the concatenated outputs of previous graphs
            stacked_lstm = tf.keras.layers.StackedRNNCells(lstmsNets)
            self.myRNN = tf.keras.layers.RNN(stacked_lstm, time_major=self.params.timeMajor, return_sequences=True)
            self.denseOutput = tf.keras.layers.Dense(self.params.dim_output, activation=None,name="outputPos")
            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Potential issue: the backprop will go to both student (loss pred network) and teacher (pos pred network...)
            self.truePos = tf.keras.layers.Input(shape=(2), name="truePos")
            self.denseLoss1 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation)

            outputPos, lossFromOutputLoss = self.get_Model()

            self.model = self.mybuild(outputPos, lossFromOutputLoss)

    def get_Model(self):
        # generate and compile the model, lr is the chosen learning rate
        # CNN plus dense on every group independently
        allFeatures = [] # store the result of the computation for each group
        for group in range(self.params.nGroups):
            x = self.inputsToSpikeNets[group]
            # --> [NbKeptSpike,nbChannels,32] tensors
            x = self.spikeNets[group].apply(x)  # outputs a [NbSpikeOfTheGroup,nFeatures=self.params.nFeatures(default 128)] tensor.
            x = tf.matmul(self.completionTensors[group], x)
            # Pierre: Multiplying by completionTensor allows to put the spike of the group at its corresponding position
            # in the identity matrix.
            #   The index of spike detected then become similar to a time value...
            x = tf.reshape(x, [self.params.batch_size, -1,self.params.nFeatures])
            # Reshaping the result of the spike net as batch_size:NbTotSpikeDetected:nFeatures
            # this allow to separate spikes from the same window or from the same batch.
            if self.params.timeMajor:
                x = tf.transpose(a=x, perm=[1, 0,2])  # if timeMajor (the case by default): exchange batch-size and nbTimeSteps
            allFeatures.append(x)
        allFeatures = tf.tuple(tensors=allFeatures)  # synchronizes the computation of all features (like a join)
        print(allFeatures)
        # The concatenation is made over axis 2, which is the Feature axis
        allFeatures = tf.concat(allFeatures, axis=2, name="concat1")
        print(allFeatures)
        print( self.myRNN(allFeatures))
        outputs = self.myRNN(allFeatures)

        # Last_relevant select in each window of each batch the last indexes which effectively had spike.
        # Indeed, for each batch (time window corresponding to a particular time window) there is a varying number of spike
        # detected groups-wide
        # But to have dense matrix; we allocate the same number of spike in each window (so shared by window
        # of a given batch), the inputs are therefore filled with additional 0 value.
        #  Therefore we select the output of the RNN that correspond to the last spike seen in each window
        # This is the role of the last_relevant function/
        output = nnUtils.last_relevant(outputs, self.iteratorLengthInput, timeMajor=self.params.timeMajor)
        print(output)
        outputPos = self.denseOutput(output)
        outputLoss = self.denseLoss2(self.denseLoss1(output))[:, 0]

        # Idea to bypass the fact that we need the loss of the Pos network.
        # We already compute in the main loops the loss of the Pos network by feeding the targetPos to the network
        # to use it in part of the network that predicts the loss.
        mylossPos = tf.losses.mean_squared_error(outputPos, self.truePos)
        lossFromOutputLoss = tf.identity(tf.losses.mean_squared_error(outputLoss, mylossPos),name="lossOfLossPredictor")

        return outputPos, lossFromOutputLoss

    def mybuild(self, outputPos, lossFromOutputLoss):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+[self.iteratorLengthInput,self.truePos,self.inputGroups,self.completionMatrix],
                               outputs=[outputPos]) #, lossFromOutputLoss

        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
            loss={
                "outputPos" : tf.keras.losses.mean_squared_error,
                #"tf_op_layer_lossOfLossPredictor" : tf.keras.losses.mean_squared_error,
            },
            loss_weights=[1.0] #
        )
        return model

    def lr_schedule(self,epoch):
        # look for the learning rate for the given epoch.
        for lr in self.params.learningRates:
            if (epoch < (lr + 1) * self.params.nEpochs / len(self.params.learningRates)) and (
                    epoch >= lr * self.params.nEpochs / len(self.params.learningRates)):
                return lr
        return self.params.learningRates[0]

    def train(self):
        ### Training models
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"]).shuffle(self.params.nSteps).repeat()
        dataset = dataset.batch(self.params.batch_size)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals, batched=True))
        # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
        # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
        dataset = dataset.map(lambda vals: (vals,{"outputPos":vals["pos"]}))
        #,"tf_op_layer_lossOfLossPredictor": tf.zeros_like(vals["pos"])

        def add_CompletionMatrix(vals,valsout):
            vals.update({"completionMatrix" : tf.eye(tf.shape(vals["pos"])[0])})
            vals.update({"truePos": vals["pos"]})
            return vals,valsout
        dataset = dataset.map(add_CompletionMatrix)

        datasetlengths = dataset.take(10).map(lambda x, y: x["length"])
        lengths = list(datasetlengths.as_numpy_iterator())
        datasetg1 = dataset.take(10).map(lambda x, y: x["group1"])
        g1 = list(datasetg1.as_numpy_iterator())
        datasetg2 = dataset.take(10).map(lambda x, y: x["group2"])
        g2 = list(datasetg2.as_numpy_iterator())
        datasetgroups = dataset.take(10).map(lambda x, y: x["groups"])
        groups1 = list(datasetgroups.as_numpy_iterator())



        # The callbacks called during the training:
        callbackLR = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.folder+"results",'training.log'))
        checkpoint_path = os.path.join(self.projectPath.folder+"results","training_1/cp.ckpt")
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[callbackLR, csv_logger, cp_callback],
                              steps_per_epoch = int(self.params.nSteps / self.params.nEpochs))

        #TODO Look at the goal of convert: self.convert()
        return hist.history["loss"]


    def test(self):

        self.model.load_weights(os.path.join(self.projectPath.folder+"results","training_1/cp.ckpt"))

        ### Loading and inferring
        print("INFERRING")
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["test"])
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals,batched=True))
        dataset = dataset.map(lambda vals: ( vals, {"outputPos": vals["pos"], "tf_op_layer_lossOfLossPredictor": tf.zeros_like(vals["pos"])}))
        def add_CompletionMatrix(vals, valsout):
            vals.update({"completionMatrix": tf.eye(tf.shape(vals["pos"])[0])})
            vals.update({"truePos": vals["pos"]})
            return vals, valsout
        dataset = dataset.map(add_CompletionMatrix)

        datasetPos = dataset.map(lambda x, y: x["pos"])
        pos = list(datasetPos.as_numpy_iterator())
        datasetTimes = dataset.map(lambda x, y: x["time"])
        times = list(datasetTimes.as_numpy_iterator())

        datasetlengths = dataset.map(lambda x, y: x["length"])
        lengths = list(datasetlengths.as_numpy_iterator())
        datasetg1 = dataset.map(lambda x, y: x["group1"])
        g1 = list(datasetg1.as_numpy_iterator())

        outputPos_test = self.model.predict(dataset,steps=len(pos)) #, outputLoss_test

        return {"inferring": np.array(outputPos_test), "pos": np.array(pos), "times": times}