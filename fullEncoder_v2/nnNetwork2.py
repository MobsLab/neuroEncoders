import numpy as np
import tensorflow as tf
from fullEncoder import nnUtils
from tqdm import trange
import os

# Pierre 14/02/01:
# Reorganization of the code:
    # One class for the network
    # One function for the training
# We save the model every epoch during the training

# We generate a model with the functional Model interface in tensorflow
class LSTMandSpikeNetwork():
    def __init__(self, projectPath, params, device_name="/cpu:0"):
        super(LSTMandSpikeNetwork, self).__init__()
        self.projectPath = projectPath
        self.params = params
        self.device_name = device_name
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos": tf.io.FixedLenFeature([self.params.dim_output], tf.float32), #target positions
            "length": tf.io.FixedLenFeature([], tf.int64), #number of spike sequence gathered in the window
            "groups": tf.io.VarLenFeature(tf.int64), # the index of the groups having spike sequences in the window
            "time": tf.io.FixedLenFeature([], tf.float32)} # the time-steps
        for g in range(self.params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)}) # the spike sequence of each groups in the window

        #Loss obtained during training
        self.trainLosses = []

        # TODO: initialization of the networks
        with tf.device(self.device_name):
            self.iteratorLengthInput = tf.keras.layers.Input(shape=(),name="length")
            self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(),name="group"+str(group)) for group in range(self.params.nGroups)]
            # Declare spike nets for the different groups:
            self.spikeNets = [nnUtils.spikeNet(nChannels=self.params.nChannels[group], device=self.device_name,
                                               nFeatures=self.params.nFeatures) for group in range(self.params.nGroups)]
            lstmsNets = [[nnUtils.layerLSTM(self.params.lstmSize, dropout=self.params.lstmDropout) for _ in
                         range(self.params.lstmLayers)]]
            # LSTM on the concatenated outputs of previous graphs
            stacked_lstm = tf.keras.layers.StackedRNNCells(lstmsNets)
            self.myRNN = tf.keras.layers.RNN(stacked_lstm, time_major=self.params.timeMajor, return_state=True)
            self.denseOutput = tf.keras.layers.Dense(self.params.dim_output, activation=None)
            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Potential issue: the backprop will go to both student (loss pred network) and teacher (pos pred network...)
            self.truePos = tf.keras.layers.Input(shape=(),name="truePos")
            self.denseLoss1 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation)

        self.model = self.get_Model()

    def get_Model(self):
        # generate and compile the model, lr is the chosen learning rate

        with tf.device(self.device_name):
            # CNN plus dense on every group indepedently
            allFeatures = [] # store the result of the computation for each group
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                # --> [nbKeptSpikeSequence(time steps where we have more than one spiking),nbChannels,32] tensors
                idMatrix = tf.eye(tf.shape(x)[0])
                # What is the role of the completionTensor?
                # The id matrix dimension is the number of different spike sequence encoded inside the spike window
                # Indeed iterators["groups"] is the list of group that were emitted during each spike sequence merged into the spike window
                completionTensor = tf.transpose(
                    a=tf.gather(idMatrix, tf.where(tf.equal(x, group)))[:, 0, :], perm=[1, 0],
                    name="completion")
                # The completion Matrix, gather the row of the idMatrix, ie the spike sequence corresponding to the group: group
                x = self.spikeNets[group].apply(x)  # outputs a [nbTimeWindow,nFeatures=self.params.nFeatures(default 128)] tensor.
                x = tf.matmul(completionTensor, x)
                # Pierre: Multiplying by completionTensor allows to remove the windows where no spikes was observed from this group.
                # But I thought that for iterators["group"+str(group)] this was already the case.

                x = tf.reshape(x, [self.params.batch_size, -1,self.params.nFeatures])
                # Reshaping the result of the spike net as batch_size:nbTimeSteps:nFeatures
                if self.params.timeMajor:
                    x = tf.transpose(a=x, perm=[1, 0,2])  # if timeMajor (the case by default): exchange batch-size and nbTimeSteps
                allFeatures.append(x)
            allFeatures = tf.tuple(tensors=allFeatures)  # synchronizes the computation of all features (like a join)
            # The concatenation is made over axis 2, which is the Feature axis
            allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

            outputs, finalState = self.myRNN(allFeatures)
            # Question: why output the finalStae if we never use it

            # TODO Pierre: the question is: do we have to use last_relevant?
            output = nnUtils.last_relevant(outputs, self.iteratorLengthInput, timeMajor=self.params.timeMajor)
            outputPos = self.denseOutput(output,name="outputPos")
            outputLoss = self.denseLoss2(self.denseLoss1(output),name="outputLoss")[:, 0]

            # Idea to bypass the fact that we need the
            # true position to compute the loss: we already compute in the main loops the loss of the Pos network
            # to use it in part of the network that predicts the loss.
            mylossPos = tf.losses.mean_squared_error(outputPos,self.truePos)
            lossFromOutputLoss = tf.losses.mean_squared_error(outputLoss, mylossPos, name="lossOfLossPredictor")

            model = tf.keras.Model(inputs=self.inputsToSpikeNets+[self.iteratorLengthInput]+[self.truePos],
                           outputs=[output, outputPos, outputLoss, lossFromOutputLoss])

            # Initially compile using the minimal learnign rate.
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]),
                loss={
                    "outputPos" : tf.keras.losses.mean_squared_error(),
                    "lossOfLossPredictor" : tf.keras.losses.mean_squared_error(),
                },
                loss_weights=[1.0, 1.0],
            )
            return model

    def lr_schedule(self,epoch,lastlr):
        # look for the learning rate for the given epoch.
        for lr in range(len(self.params.nEpochs)):
            if (epoch < (lr + 1) * self.params.nEpochs / len(self.params.learningRates)) and (
                    epoch >= lr * self.params.nEpochs / len(self.params.learningRates)):
                return lr
        return self.params.learningRates[0]

    def train(self):
        ### Training models
        with tf.Graph().as_default():
            print()
            print('TRAINING')
            dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"]).shuffle(self.params.nSteps).repeat()
            dataset = dataset.batch(self.params.batch_size)
            dataset = dataset.map(
                lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals, batched=True))
            # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
            # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
            dataset = dataset.map(lambda vals:  (vals, vals))

            # The callbacks called during the training:
            callbackLR = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
            csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath,'training.log'))
            checkpoint_path = os.path.join(self.projectPath,"training_1/cp.ckpt")
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

            hist = self.model.fit(dataset,
                      epochs=self.params.nEpochs,
                      callbacks=[callbackLR, csv_logger, cp_callback])

        #TODO Look at the goal of convert: self.convert()
        return hist.history["Loss"]


    def test(self):
        ### Loading and inferring
        print()
        print("INFERRING")

        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["test"])
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals))

        output_test, outputPos_test, outputLoss_test, lossFromOutputLoss_test = self.model.predict(dataset)
        datasetPos = dataset.map(lambda x: x["pos"])
        pos = list(datasetPos.as_numpy_iterator())
        datasetTimes = dataset.map(lambda x: x["times"])
        times = list(datasetTimes.as_numpy_iterator())

        return {"inferring": np.array(outputPos_test), "pos": np.array(pos), "times": times}