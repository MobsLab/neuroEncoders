import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
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

            # 4 LSTM cell, we can't use the RNN interface
            # and stacked RNN cells if we want to use the LSTM improvements.
            self.lstmsNets = [tf.keras.layers.LSTM(self.params.lstmSize,return_sequences=True),
                         tf.keras.layers.LSTM(self.params.lstmSize,return_sequences=True),
                         tf.keras.layers.LSTM(self.params.lstmSize,return_sequences=True),
                         tf.keras.layers.LSTM(self.params.lstmSize)] #,,recurrent_dropout=self.params.lstmDropout
            self.denseFeatureOutput = tf.keras.layers.Dense(self.params.dim_output, activation=tf.keras.activations.hard_sigmoid, dtype=tf.float32)
            #tf.keras.activations.hard_sigmoid


            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Potential issue: the backprop will go to both student (loss pred network) and teacher (pos pred network...)
            self.truePos = tf.keras.layers.Input(shape=(self.params.dim_output), name="pos")
            self.denseLoss1 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation)

            outputs = self.get_Model()

            self.model = self.mybuild(outputs)

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
                # We therefore need to set indices[spikePosition[i]] to i  so that it is effectively gather
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
                allFeatures.append(filledFeatureTrain)


            allFeatures = tf.tuple(tensors=allFeatures)  # synchronizes the computation of all features (like a join)

            # The concatenation is made over axis 2, which is the Feature axis
            # So we reserve columns to each output of the spiking networks...
            allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

            #We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups,[self.params.batch_size,-1])
            mymask = tf.not_equal(batchedInputGroups,-1)

            # Next we try a simple strategy to reduce training time:
            # We reduce in duration the sequence fed to the LSTM by summing convnets ouput
            # over bins of 10 successive spikes.
            if self.params.fasterRNN:
                segment_ids = tf.range(tf.shape(allFeatures)[1])
                segment_ids = tf.math.floordiv(segment_ids, 10)
                allFeatures_transposed = tf.transpose(allFeatures,perm=[1,0,2])
                allFeatures_transposed = tf.math.segment_sum(allFeatures_transposed,segment_ids)
                allFeatures = tf.transpose(allFeatures_transposed,perm=[1,0,2])
                tf.ensure_shape(allFeatures,[self.params.batch_size,None,allFeatures.shape[2]])
                # true+true remains true, therefore as soon as a time step should have been ran
                # we can use our mask.
                mymaskFloat = tf.transpose(tf.math.segment_sum(tf.transpose(tf.cast(mymask,dtype=tf.float32)),segment_ids))
                mymask = tf.math.greater_equal(mymaskFloat,1.0)
                tf.ensure_shape(mymask, [self.params.batch_size, None])

            output_seq = self.lstmsNets[0](allFeatures,mask=mymask)
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output = self.lstmsNets[3](output_seq, mask=mymask)

            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss1(tf.stop_gradient(output)))

            # Idea to bypass the fact that we need the loss of the Pos network.
            # We already compute in the main loops the loss of the Pos network by feeding the targetPos to the network
            # to use it in part of the network that predicts the loss.

            posLoss = tf.losses.mean_squared_error(myoutputPos,self.truePos)[:,tf.newaxis]

            idmanifoldloss = tf.identity(tf.math.reduce_mean(posLoss),name="lossOfManifold")
            lossFromOutputLoss = tf.identity(tf.math.reduce_mean(tf.losses.mean_squared_error(outputLoss, posLoss)),name="lossOfLossPredictor")
        return  myoutputPos, outputLoss, idmanifoldloss , lossFromOutputLoss

    def mybuild(self, outputs):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.iteratorLengthInput,self.truePos,self.inputGroups],
                               outputs=outputs)
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True
        )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
            loss={
                "tf_op_layer_lossOfManifold" : lambda x,y:y,
                "tf_op_layer_lossOfLossPredictor" : lambda x,y:y,
            },
        )
        return model

    def lr_schedule(self,epoch):
        # look for the learning rate for the given epoch.
        for lr in self.params.learningRates:
            if (epoch < (lr + 1) * self.params.nEpochs / len(self.params.learningRates)) and (
                    epoch >= lr * self.params.nEpochs / len(self.params.learningRates)):
                return lr
        return self.params.learningRates[0]

    def createIndices(self, vals, valsout):
        # vals.update({"completionMatrix" : tf.eye(tf.shape(vals["groups"])[0])})
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




    def train(self):
        ### Training models
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"]) #.shuffle(self.params.nSteps) #.repeat()
        dataset = dataset.batch(self.params.batch_size,drop_remainder=True)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals, batched=True))
        # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
        # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
        dataset = dataset.map(lambda vals: (vals,{"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                  "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),num_parallel_calls=4)
        dataset = dataset.map(self.createIndices, num_parallel_calls=4)
        dataset = dataset.shuffle(self.params.nSteps,reshuffle_each_iteration=True).cache() #.repeat() #
        dataset = dataset.prefetch(self.params.batch_size* 10) #

        # The callbacks called during the training:
        callbackLR = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.resultsPath,'training.log'))
        checkpoint_path = os.path.join(self.projectPath.resultsPath,"training_1/cp.ckpt")
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.projectPath.folder+"results","profiling"),
        #                                             profile_batch = '200,210')

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[callbackLR, csv_logger, cp_callback], # , tb_callback,cp_callback
                              ) #steps_per_epoch = int(self.params.nSteps / self.params.nEpochs)

        return np.transpose(np.stack([hist.history["tf_op_layer_lossOfManifold_loss"],
                                      hist.history["tf_op_layer_lossOfLossPredictor_loss"]]))


    def test(self):

        self.model.load_weights(os.path.join(self.projectPath.resultsPath,"training_1/cp.ckpt"))

        ### Loading and inferring
        print("INFERRING")
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["test"])
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals,batched=True))
        dataset = dataset.map(lambda vals: ( vals, {"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                    "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),num_parallel_calls=4)
        dataset = dataset.map(self.createIndices,num_parallel_calls=4)

        datasetLength = dataset.map(lambda x, y: x["length"])
        fullLength = list(datasetLength.as_numpy_iterator())
        fullLength = np.array(fullLength)

        datasetgroups = dataset.map(lambda x, y: x["groups"])
        fullgroups = list(datasetgroups.as_numpy_iterator())
        fullgroups = np.array(fullgroups)

        datasetgroups1 = dataset.map(lambda x, y: x["group1"])
        fullgroups1 = list(datasetgroups1.as_numpy_iterator())
        fullgroups1 = np.array(fullgroups1)

        datasetgroups2 = dataset.map(lambda x, y: x["group2"])
        fullgroups2 = list(datasetgroups2.as_numpy_iterator())
        fullgroups2 = np.array(fullgroups2)


        datasetPos = dataset.map(lambda x, y: x["pos"])
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)

        datasetTimes = dataset.map(lambda x, y: x["time"])
        times = list(datasetTimes.as_numpy_iterator())

        output_test = self.model.predict(dataset) #

        predofLoss = np.expand_dims(output_test[1],axis=1)
        outLoss = np.expand_dims(output_test[2], axis=1)
        featureTrue = np.reshape(fullFeatureTrue, [output_test[0].shape[0], output_test[0].shape[-1]])

        times = np.reshape(times, [output_test[0].shape[0]])
        return {"featurePred": output_test[0], "featureTrue": featureTrue,
                "times": times, "predofLoss" : predofLoss,
                "lossFromOutputLoss" : outLoss}
