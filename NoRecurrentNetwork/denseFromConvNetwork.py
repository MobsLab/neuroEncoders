import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd

from importData.ImportClusters import getBehavior
from importData.rawDataParser import  inEpochsMask

# Pierre 11/04/2021:
# A network that runs the convnets over all spikes
# Then sum the outputs of each convnet (so of a group) )over the time window
# The results are concatenated over the different convnet (i.e groups)
# to produce an array that is then densely projected 2 times to produce the position
# prediction....


class DenseFromConvNetwork():

    def __init__(self, projectPath, params, device_name="/device:gpu:0"):
        super(DenseFromConvNetwork, self).__init__()
        self.projectPath = projectPath
        self.params = params
        self.device_name = device_name
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos_index" : tf.io.FixedLenFeature([], tf.int64),
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
            self.denseDecoder1 = tf.keras.layers.Dense(self.params.lstmSize*2,activation = tf.keras.activations.tanh)
            self.denseDecoder3 = tf.keras.layers.Dense(self.params.lstmSize*2,activation = tf.keras.activations.tanh)
            self.denseDecoder2 = tf.keras.layers.Dense(self.params.lstmSize*2,activation = tf.keras.activations.tanh)
            self.denseDecoder4 = tf.keras.layers.Dense(self.params.lstmSize*2,activation = tf.keras.activations.tanh)

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

            if self.params.shuffle_spike_order:
                #Note: to activate this shuffling we will need to recompile the model again...
                indices_shuffle = tf.range(start=0, limit=tf.shape(allFeatures)[1], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices_shuffle)
                allFeatures_transposed = tf.transpose(allFeatures, perm=[1, 0, 2])
                allFeatures = tf.transpose(tf.gather(allFeatures_transposed, shuffled_indices),perm=[1,0,2])
                tf.ensure_shape(allFeatures, [self.params.batch_size, None, allFeatures.shape[2]])
            if self.params.shuffle_convnets_outputs:
                indices_shuffle = tf.range(start=0, limit=tf.shape(allFeatures)[2], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices_shuffle)
                allFeatures = tf.transpose(tf.gather(tf.transpose(allFeatures,perm=[2,1,0]), shuffled_indices),perm=[2,1,0])
                tf.ensure_shape(allFeatures, [self.params.batch_size, None, allFeatures.shape[2]])


            # We simply sum the population vectors which was output by the
            # convnets.
            # Note that shuffling the spike order should therefore not change the result
            allFeatures = tf.reduce_sum(allFeatures,axis=1)

            output = self.denseDecoder1(allFeatures)
            output = self.denseDecoder2(output)
            output = self.denseDecoder3(output)
            output = self.denseDecoder4(output)

            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss1(tf.stop_gradient(output)))

            # Idea to bypass the fact that we need the loss of the Pos network.
            # We already compute in the main loops the loss of the Pos network by feeding the targetPos to the network
            # to use it in part of the network that predicts the loss.

            posLoss = tf.losses.mean_squared_error(myoutputPos,self.truePos)[:,tf.newaxis]

            idmanifoldloss = tf.identity(tf.math.reduce_mean(posLoss),name="lossOfManifold")
            lossFromOutputLoss = tf.identity(tf.math.reduce_mean(tf.losses.mean_squared_error(outputLoss, posLoss)),name="lossOfLossPredictor")
        return  myoutputPos, outputLoss, idmanifoldloss , lossFromOutputLoss

    def mybuild(self, outputs,modelName="model.png"):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.iteratorLengthInput,self.truePos,self.inputGroups],
                               outputs=outputs)
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
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
        ## read behavior matrix:
        behavior_data = getBehavior(self.projectPath.folder,getfilterSpeed = True)
        speed_mask = behavior_data["Times"]["speedFilter"]

        ### Training models
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        dataset = dataset.map(lambda *vals:nnUtils.parseSerializedSpike(self.feat_desc,*vals))

        epochMask  = inEpochsMask(behavior_data['Position_time'][:,0], behavior_data['Times']['trainEpochs'])
        tot_mask = speed_mask * epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)),dtype=tf.int64),
                                                tf.constant(tot_mask,dtype=tf.float64)),default_value=0)
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]),1.0))

        dataset = dataset.batch(self.params.batch_size,drop_remainder=True)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True)) #self.feat_desc, *
        # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
        # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
        dataset = dataset.map(lambda vals: (vals,{"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                  "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),num_parallel_calls=4)
        dataset = dataset.map(self.createIndices, num_parallel_calls=4)
        dataset = dataset.shuffle(self.params.nSteps,reshuffle_each_iteration=True).cache() #.repeat() #
        dataset = dataset.prefetch(self.params.batch_size* 10) #
        #
        # data = list(dataset.take(1).map(lambda x,y:x["group1"]).as_numpy_iterator())
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

        trainLosses = np.transpose(np.stack([hist.history["tf_op_layer_lossOfManifold_loss"],
                                      hist.history["tf_op_layer_lossOfLossPredictor_loss"]]))

        df = pd.DataFrame(trainLosses)
        df.to_csv(os.path.join(self.projectPath.resultsPath, "resultInference", "lossTraining.csv"))
        fig,ax = plt.subplots()
        ax.plot(trainLosses[:,0])
        plt.show()
        fig.savefig(os.path.join(self.projectPath.resultsPath, "lossTraining.png"))

        return trainLosses


    def test(self,linearizationFunction,saveFolder="resultInference"):

        self.model.load_weights(os.path.join(self.projectPath.resultsPath,"training_1/cp.ckpt"))

        ### Loading and inferring
        print("INFERRING")
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)
        speed_mask = behavior_data["Times"]["speedFilter"]

        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals))
        epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['testEpochs'])
        tot_mask = speed_mask * epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals,batched=True))
        dataset = dataset.map(lambda vals: ( vals, {"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                    "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),num_parallel_calls=4)
        dataset = dataset.map(self.createIndices,num_parallel_calls=4)

        datasetPos = dataset.map(lambda x, y: x["pos"])
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)

        datasetTimes = dataset.map(lambda x, y: x["time"])
        times = list(datasetTimes.as_numpy_iterator())

        output_test = self.model.predict(dataset) #

        outLoss = np.expand_dims(output_test[2], axis=1)
        featureTrue = np.reshape(fullFeatureTrue, [output_test[0].shape[0], output_test[0].shape[-1]])
        times = np.reshape(times, [output_test[0].shape[0]])

        projPredPos = linearizationFunction(output_test[0][:,:2])
        projTruePos = linearizationFunction(featureTrue)

        if not os.path.isdir(os.path.join(self.projectPath.resultsPath,saveFolder)):
            os.makedirs(os.path.join(self.projectPath.resultsPath,saveFolder))

        # Saving files
        df = pd.DataFrame(output_test[0][:,:2])
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "featurePred.csv"))
        df = pd.DataFrame(featureTrue)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "featureTrue.csv"))
        df = pd.DataFrame(projPredPos)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "projPredFeature.csv"))
        df = pd.DataFrame(projTruePos)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "projTrueFeature.csv"))
        df = pd.DataFrame(output_test[1])
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "lossPred.csv"))
        df = pd.DataFrame(times)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "timeStepsPred.csv"))


        fig, ax = plt.subplots(2, 1)
        ax[1].scatter(times, featureTrue[:, 1], c="black", label="true Position")
        ax[1].scatter(times, output_test[0][:, 1], c="red", label="predicted Position")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("Y")
        ax[1].set_title("prediction with TF2.0's architecture")
        ax[0].scatter(output_test[0][:, 1], featureTrue[:, 1], alpha=0.1)
        fig.legend()
        plt.show()

        return {"featurePred": output_test[0], "featureTrue": featureTrue,
                "times": times, "predofLoss" : output_test[1],
                "lossFromOutputLoss" : outLoss}
