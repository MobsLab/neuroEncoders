import numpy as np
import tensorflow as tf

from fullEncoder_v2 import nnUtils
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
            if self.params.usingMixedPrecision:
                self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(None,None),name="group"+str(group),dtype=tf.float16) for group in range(self.params.nGroups)]
            else:
                self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(None,None),name="group"+str(group)) for group in range(self.params.nGroups)]

            self.inputGroups = tf.keras.layers.Input(shape=(),name="groups")
            # The spike nets acts on each group separately; to reorganize all these computations we use
            # an identity matrix which shape is the total number of spike measured (over all groups)
            #self.completionMatrix = tf.keras.layers.Input(shape=(None,), name="completionMatrix")

            self.indices = [tf.keras.layers.Input(shape=(),name="indices"+str(group),dtype=tf.int32) for group in range(self.params.nGroups)]
            if self.params.usingMixedPrecision:
                self.zeroForGather = tf.zeros([1,self.params.nFeatures],dtype=tf.float16)
            else:
                self.zeroForGather = tf.zeros([1, self.params.nFeatures])
            # What is the role of the completionTensor?
            # The id matrix dimension is the total number of spikes encoded inside the spike window
            # Indeed iterators["groups"] is the list of group that were emitted during each spike sequence merged into the spike window
            #self.completionTensors = [tf.transpose(
            #    a=tf.gather(self.completionMatrix, tf.where(tf.equal(self.inputGroups, g)))[:, 0, :], perm=[1, 0],
            #    name="completion") for g in range(self.params.nGroups)]
            # The completion Matrix, gather the row of the idMatrix, ie the spike corresponding to the group: group

            # Declare spike nets for the different groups:
            self.spikeNets = [nnUtils.spikeNet(nChannels=self.params.nChannels[group], device=self.device_name,
                                               nFeatures=self.params.nFeatures) for group in range(self.params.nGroups)]
            lstmsNets = [tf.keras.layers.LSTMCell(self.params.lstmSize) for _ in
                         range(self.params.lstmLayers)] #,recurrent_dropout=self.params.lstmDropout
            # LSTM on the concatenated outputs of previous graphs
            stacked_lstm = tf.keras.layers.StackedRNNCells(lstmsNets)
            self.myRNN = tf.keras.layers.RNN(stacked_lstm, time_major=self.params.timeMajor) # , return_sequences=True
            self.denseOutput = tf.keras.layers.Dense(self.params.dim_output, activation=None, name="outputPos", dtype=tf.float32)
            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Potential issue: the backprop will go to both student (loss pred network) and teacher (pos pred network...)
            self.truePos = tf.keras.layers.Input(shape=(2), name="pos")
            self.denseLoss1 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation)

            outputPos, lossFromOutputLoss = self.get_Model()

            self.model = self.mybuild(outputPos, lossFromOutputLoss)

    def get_Model(self):
        # generate and compile the model, lr is the chosen learning rate
        # CNN plus dense on every group independently
        with tf.device(self.device_name):
            # Optimization: could we not perform this foor loop more efficiently?
            # Key: try to use a tf.map as well as a parallel access.

            allFeatures = [] # store the result of the computation for each group
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                # --> [NbKeptSpike,nbChannels,32] tensors
                x = self.spikeNets[group].apply(x)
                # outputs a [NbSpikeOfTheGroup,nFeatures=self.params.nFeatures(default 128)] tensor.

                # The gather strategy:

                #extract the final position of the spikes
                #Note: inputGroups is already filled with -1 at position that correspond to filling
                # for batch issues
                # The i-th spike of the group should be positioned at spikePosition[i] in the final tensor
                # We therefore need to set indices[spikePosition[i]] to i  so that it is effectively gathere
                # We then gather either a value of
                filledFeatureTrain = tf.gather(tf.concat([self.zeroForGather ,x],axis=0),self.indices[group],axis=0)
                # At this point; filledFeatureTrain is a tensor of size (NbBatch*max(nbSpikeInBatch),self.params.nFeatures)
                # where we have filled lines corresponding to spike time of the group
                # with the feature computed by the spike net; and let other time with a value of 0:

                #x = tf.matmul(self.completionTensors[group], x)
                # Pierre: Multiplying by completionTensor allows to put the spike of the group at its corresponding position
                # in the identity matrix.
                #   The index of spike detected then become similar to a time value...

                filledFeatureTrain = tf.reshape(filledFeatureTrain, [self.params.batch_size, -1,self.params.nFeatures])
                # Reshaping the result of the spike net as batch_size:NbTotSpikeDetected:nFeatures
                # this allow to separate spikes from the same window or from the same batch.
                if self.params.timeMajor:
                    filledFeatureTrain = tf.transpose(a=filledFeatureTrain, perm=[1, 0,2])  # if timeMajor (the case by default): exchange batch-size and nbTimeSteps
                allFeatures.append(filledFeatureTrain)


            allFeatures = tf.tuple(tensors=allFeatures)  # synchronizes the computation of all features (like a join)

            # The concatenation is made over axis 2, which is the Feature axis
            # So we reserve columns to each output of the spiking networks...
            allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

            #allFeatures = tf.add_n(allFeatures, name="sum1")
            #Other possibility we sum the list of 2D output of each batch elementwise

            #We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups,[self.params.batch_size,-1])
            if self.params.timeMajor:
                batchedInputGroups = tf.transpose(batchedInputGroups)
            mymask = tf.not_equal(batchedInputGroups,-1)
            output = self.myRNN(allFeatures,mask=mymask) #

            # Last_relevant select in each window of each batch the last indexes which effectively had spike.
            # Indeed, for each batch (time window corresponding to a particular time window) there is a varying number of spike
            # detected groups-wide
            # But to have dense matrix; we allocate the same number of spike in each window (so shared by window
            # of a given batch), the inputs are therefore filled with additional 0 value.
            #  Therefore we select the output of the RNN that correspond to the last spike seen in each window
            # This is the role of the last_relevant function/
            #output = nnUtils.last_relevant(output, self.iteratorLengthInput , timeMajor=self.params.timeMajor)


            myoutputPos = self.denseOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss1(output))[:, 0]

            # Idea to bypass the fact that we need the loss of the Pos network.
            # We already compute in the main loops the loss of the Pos network by feeding the targetPos to the network
            # to use it in part of the network that predicts the loss.
            mylossPos = tf.losses.mean_squared_error(myoutputPos, self.truePos)
            lossFromOutputLoss = tf.identity(tf.losses.mean_squared_error(outputLoss, mylossPos),name="lossOfLossPredictor")
        return myoutputPos, lossFromOutputLoss

    def mybuild(self, outputPos, lossFromOutputLoss):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.iteratorLengthInput,self.truePos,self.inputGroups],
                               outputs=[outputPos]) #, lossFromOutputLoss

        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
            loss={
                "outputPos" : tf.keras.losses.mean_squared_error,
                #"tf_op_layer_lossOfLossPredictor" : tf.keras.losses.mean_squared_error,
            }
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
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"]) #.shuffle(self.params.nSteps) #.repeat()
        dataset = dataset.batch(self.params.batch_size,drop_remainder=True)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals, batched=True))
        # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
        # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
        dataset = dataset.map(lambda vals: (vals,{"outputPos":vals["pos"]}),num_parallel_calls=4)

        #,"tf_op_layer_lossOfLossPredictor": tf.zeros_like(vals["pos"])

        def add_CompletionMatrix(vals,valsout):
            #vals.update({"completionMatrix" : tf.eye(tf.shape(vals["groups"])[0])})
            for group in range(self.params.nGroups):
                spikePosition = tf.where(tf.equal(vals["groups"], group))
                # Note: inputGroups is already filled with -1 at position that correspond to filling
                # for batch issues
                # The i-th spike of the group should be positioned at spikePosition[i] in the final tensor
                # We therefore need to set indices[spikePosition[i]] to i  so that it is effectively gathered
                # We need to wrap the use of sparse tensor (tensorflow error otherwise)
                # The sparse tensor allows us to get the list of indices for the gather quite easily
                rangeIndices = tf.range(tf.shape(vals["group" + str(group)])[0]) + 1
                indices = tf.sparse.SparseTensor(spikePosition, rangeIndices, [tf.shape(vals["groups"])[0]])
                indices = tf.cast(tf.sparse.to_dense(indices), dtype=tf.int32)
                vals.update({"indices" + str(group): indices})

                #changing the dtype to allow faster computations
                if self.params.usingMixedPrecision:
                    vals.update({"group"+str(group) : tf.cast(vals["group"+str(group)],dtype=tf.float16)})

            if self.params.usingMixedPrecision:
                vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})
            return vals,valsout
        dataset = dataset.map(add_CompletionMatrix,num_parallel_calls=4)

        dataset = dataset.cache().repeat()
        dataset = dataset.prefetch(self.params.batch_size * 10)


        # The callbacks called during the training:
        callbackLR = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.folder+"results",'training.log'))
        checkpoint_path = os.path.join(self.projectPath.folder+"results","training_1/cp.ckpt")
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.projectPath.folder+"results","profiling"),
                                                     profile_batch = '200,210')

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[callbackLR,tb_callback], # , csv_logger, cp_callback
                              steps_per_epoch = int(self.params.nSteps / self.params.nEpochs))

        #TODO Look at the goal of convert: self.convert()
        return hist.history["loss"]


    def test(self):

        self.model.load_weights(os.path.join(self.projectPath.folder+"results","training_1/cp.ckpt"))

        ### Loading and inferring
        print("INFERRING")
        # TODO: switch back to test set....
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["test"])
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals,batched=True))
        dataset = dataset.map(lambda vals: ( vals, {"outputPos": vals["pos"], "tf_op_layer_lossOfLossPredictor": tf.zeros_like(vals["pos"])}))
        def add_CompletionMatrix(vals, valsout):
            #vals.update({"completionMatrix": tf.eye(tf.shape(vals["pos"])[0])})
            for group in range(self.params.nGroups):
                spikePosition = tf.where(tf.equal(vals["groups"], group))
                # Note: inputGroups is already filled with -1 at position that correspond to filling
                # for batch issues
                # The i-th spike of the group should be positioned at spikePosition[i] in the final tensor
                # We therefore need to set indices[spikePosition[i]] to i  so that it is effectively gathered
                # We need to wrap the use of sparse tensor (tensorflow error otherwise)
                # The sparse tensor allows us to get the list of indices for the gather quite easily
                rangeIndices = tf.range(tf.shape(vals["group" + str(group)])[0]) + 1
                indices = tf.sparse.SparseTensor(spikePosition, rangeIndices, [tf.shape(vals["groups"])[0]])
                indices = tf.cast(tf.sparse.to_dense(indices), dtype=tf.int32)
                vals.update({"indices" + str(group): indices})
                # changing the dtype to allow faster computations
                if self.params.usingMixedPrecision:
                    vals.update({"group" + str(group): tf.cast(vals["group" + str(group)], dtype=tf.float16)})
            if self.params.usingMixedPrecision:
                vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})
            return vals, valsout
        dataset = dataset.map(add_CompletionMatrix)

        datasetPos = dataset.map(lambda x, y: x["pos"])
        pos = list(datasetPos.as_numpy_iterator())
        datasetTimes = dataset.map(lambda x, y: x["time"])
        times = list(datasetTimes.as_numpy_iterator())

        outputPos_test = self.model.predict(dataset) #, outputLoss_test

        return {"inferring": np.array(outputPos_test), "pos": np.array(pos), "times": times}