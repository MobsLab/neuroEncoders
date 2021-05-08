import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd

from importData.ImportClusters import getBehavior
from importData.rawDataParser import  inEpochsMask

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
            # self.iteratorLengthInput = tf.keras.layers.Input(shape=(),name="length")
            if self.params.usingMixedPrecision:
                self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(self.params.nChannels[group],32),name="group"+str(group),dtype=tf.float16) for group in range(self.params.nGroups)]
            else:
                self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(self.params.nChannels[group],32),name="group"+str(group)) for group in range(self.params.nGroups)]

            self.inputGroups = tf.keras.layers.Input(shape=(),name="groups")
            # The spike nets acts on each group separately; to reorganize all these computations we use
            # an identity matrix which shape is the total number of spike measured (over all groups)
            #self.completionMatrix = tf.keras.layers.Input(shape=(None,), name="completionMatrix")

            self.indices = [tf.keras.layers.Input(shape=(),name="indices"+str(group),dtype=tf.int32) for group in range(self.params.nGroups)]
            if self.params.usingMixedPrecision:
                zeroForGather = tf.constant(tf.zeros([1, self.params.nFeatures], dtype=tf.float16))
            else:
                zeroForGather = tf.constant(tf.zeros([1, self.params.nFeatures]))
            self.zeroForGather = tf.keras.layers.Input(tensor=zeroForGather,name="zeroForGather")
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

            self.dropoutLayer = tf.keras.layers.Dropout(0.2)
            # 4 LSTM cell, we can't use the RNN interface
            # and stacked RNN cells if we want to use the LSTM improvements.
            self.lstmsNets = [tf.keras.layers.LSTM(self.params.lstmSize,return_sequences=True),
                         tf.keras.layers.LSTM(self.params.lstmSize,return_sequences=True),
                         tf.keras.layers.LSTM(self.params.lstmSize,return_sequences=True),
                         tf.keras.layers.LSTM(self.params.lstmSize)] #,,recurrent_dropout=self.params.lstmDropout
            self.denseFeatureOutput = tf.keras.layers.Dense(self.params.dim_output, activation=tf.keras.activations.hard_sigmoid, dtype=tf.float32,name="feature_output")
            #tf.keras.activations.hard_sigmoid


            # Used as inputs to already compute the loss in the forward pass and feed it to the loss network.
            # Potential issue: the backprop will go to both student (loss pred network) and teacher (pos pred network...)
            self.truePos = tf.keras.layers.Input(shape=(self.params.dim_output), name="pos")
            self.denseLoss1 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation,name="predicted_loss")

            outputs = self.get_Model()

            self.model = self.mybuild(outputs)

    def get_Model(self):
        # generate and compile the model
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
            allFeatures = tf.concat(allFeatures, axis=2) #, name="concat1"

            #We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups,[self.params.batch_size,-1])
            mymask = tf.not_equal(batchedInputGroups,-1)

            if self.params.shuffle_spike_order:
                #Note: to activate this shuffling we will need to recompile the model again...
                indices_shuffle = tf.range(start=0, limit=tf.shape(allFeatures)[1], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices_shuffle)
                allFeatures_transposed = tf.transpose(allFeatures, perm=[1, 0, 2])
                allFeatures = tf.transpose(tf.gather(allFeatures_transposed, shuffled_indices),perm=[1,0,2])
                tf.ensure_shape(allFeatures, [self.params.batch_size, None, allFeatures.shape[2]])
                mymask = tf.transpose(tf.gather(tf.transpose(mymask,perm=[1,0]), shuffled_indices),perm=[1,0])
                tf.ensure_shape(mymask, [self.params.batch_size, None])
            if self.params.shuffle_convnets_outputs:
                indices_shuffle = tf.range(start=0, limit=tf.shape(allFeatures)[2], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices_shuffle)
                allFeatures = tf.transpose(tf.gather(tf.transpose(allFeatures,perm=[2,1,0]), shuffled_indices),perm=[2,1,0])
                tf.ensure_shape(allFeatures, [self.params.batch_size, None, allFeatures.shape[2]])

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

            # allFeatures = self.dropoutLayer(allFeatures)

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
            #remark: we need to also stop the gradient to progagate from posLoss to the network at the stage of
            # the computations for the loss of the loss predictor
            lossFromOutputLoss = tf.identity(tf.math.reduce_mean(tf.losses.mean_squared_error(outputLoss,tf.stop_gradient(posLoss))),name="lossOfLossPredictor")
        return  myoutputPos, outputLoss, idmanifoldloss , lossFromOutputLoss

    def mybuild(self, outputs,modelName="model.png"):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.truePos,self.inputGroups,self.zeroForGather],
                               outputs=outputs)
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
            loss={
                "tf.identity" : lambda x,y:y, #tf_op_layer_lossOfManifold
                "tf.identity_1" : lambda x,y:y, #tf_op_layer_lossOfLossPredictor
            },
        )
        return model


    def get_model_for_onlineInference(self,modelName="onlineDecodingModel.png",batch=False):
        # the online inference model, it works as the training model but with a batch_size of 1 or params.batch_size
        batch_size = 1
        if batch:
            batch_size = self.params.batch_size

        with tf.device(self.device_name):
            allFeatures = []  # store the result of the computation for each group
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                x = self.spikeNets[group].apply(x)
                filledFeatureTrain = tf.gather(tf.concat([self.zeroForGather, x], axis=0), self.indices[group],
                                               axis=0)
                filledFeatureTrain = tf.reshape(filledFeatureTrain,
                                                [batch_size, -1, self.params.nFeatures])
                allFeatures.append(filledFeatureTrain)
            allFeatures = tf.tuple(
                tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2)

            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups, [batch_size, -1])
            mymask = tf.not_equal(batchedInputGroups, -1)

            output_seq = self.lstmsNets[0](allFeatures, mask=mymask)
            tf.ensure_shape(output_seq, [batch_size, None, self.params.lstmSize])
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output = self.lstmsNets[3](output_seq, mask=mymask)
            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss1(tf.stop_gradient(output)))

        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.inputGroups,self.zeroForGather],
                               outputs=[myoutputPos,outputLoss])
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile()
        return model

    def lr_schedule(self,epoch):
        # look for the learning rate for the given epoch.
        for lr in self.params.learningRates:
            if (epoch < (lr + 1) * self.params.nEpochs / len(self.params.learningRates)) and (
                    epoch >= lr * self.params.nEpochs / len(self.params.learningRates)):
                return lr
        return self.params.learningRates[0]

    def createIndices(self, vals):
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

            if self.params.usingMixedPrecision:
                zeroForGather = tf.zeros([1,self.params.nFeatures],dtype=tf.float16)
            else:
                zeroForGather = tf.zeros([1, self.params.nFeatures])
            vals.update({"zeroForGather":zeroForGather})

            # changing the dtype to allow faster computations
            if self.params.usingMixedPrecision:
                vals.update({"group" + str(group): tf.cast(vals["group" + str(group)], dtype=tf.float16)})

        if self.params.usingMixedPrecision:
            vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})
        return vals




    def train(self):
        ## read behavior matrix:
        behavior_data = getBehavior(self.projectPath.folder,getfilterSpeed = True)
        speed_mask = behavior_data["Times"]["speedFilter"]

        ### Training models
        ndataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        ndataset = ndataset.map(lambda *vals:nnUtils.parseSerializedSpike(self.feat_desc,*vals),
                              num_parallel_calls=tf.data.AUTOTUNE)
        epochMask  = inEpochsMask(behavior_data['Position_time'][:,0], behavior_data['Times']['trainEpochs'])
        tot_mask = speed_mask * epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)),dtype=tf.int64),
                                                tf.constant(tot_mask,dtype=tf.float64)),default_value=0)
        dataset = ndataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]),1.0))

        # the train epochs selection is made on top of the data where we have position measurement
        # we might additionnaly want to remove particular recording sessions, for example when the animal was
        # in a particular position of the arena.
        # next we filter by spike time inside the chosen data epochs
        #TODO
        # potentially save the dataset from here on as a smaller filtered dataset
        # if not os.path.exists(os.path.join(self.projectPath.resultsPath,"dataset_experiment","trainedData")):
        #     os.mkdir(os.path.join(self.projectPath.resultsPath,"dataset_experiment","trainedData"))
        # tf.data.experimental.save(ndataset,os.path.join(self.projectPath.resultsPath,"dataset_experiment","trainedData"))
        # dataset= tf.data.experimental.load(os.path.join(self.projectPath.resultsPath,"dataset_experiment","trainedData"),ndataset.element_spec)

        dataset = dataset.batch(self.params.batch_size,drop_remainder=True)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True),
            num_parallel_calls=tf.data.AUTOTUNE) #self.feat_desc, *
        # We then reorganize the dataset so that it provides (inputsDict,outputsDict) tuple
        # for now we provide all inputs as potential outputs targets... but this can be changed in the future...
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda vals: (vals,{"tf.identity": tf.zeros(self.params.batch_size),
                                                  "tf.identity_1": tf.zeros(self.params.batch_size)}),
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.shuffle(self.params.nSteps,reshuffle_each_iteration=True).cache() #.repeat() #
        dataset = dataset.prefetch(tf.data.AUTOTUNE) #

        callbackLR = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.resultsPath,'training.log'))
        checkpoint_path = os.path.join(self.projectPath.resultsPath,"training_1/cp.ckpt")
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.projectPath.folder,"profiling"),
        #                                             profile_batch = '100,110')

        #We can quickly have an idea of weither there is a bottlneck issue in the data pipeline using this:
        # dataset = dataset.take(1).cache().repeat(len(dataset))

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[callbackLR, csv_logger, cp_callback], # , tb_callback,cp_callback
                              ) #steps_per_epoch = int(self.params.nSteps / self.params.nEpochs)

        trainLosses = np.transpose(np.stack([hist.history["tf.identity"], #tf_op_layer_lossOfManifold
                                      hist.history["tf.identity_1"]]))  #tf_op_layer_lossOfLossPredictor_loss

        df = pd.DataFrame(trainLosses)
        df.to_csv(os.path.join(self.projectPath.resultsPath, "resultInference", "lossTraining.csv"))
        fig,ax = plt.subplots()
        ax.plot(trainLosses[:,0])
        plt.show()
        fig.savefig(os.path.join(self.projectPath.resultsPath, "lossTraining.png"))

        # print("saving model in savedmodel format, for c++")
        # if not os.path.isdir(os.path.join(self.projectPath.resultsPath,"training_1","savedir")):
        #     os.makedirs(os.path.join(self.projectPath.resultsPath,"training_1","savedir"))
        # tf.saved_model.save(self.onlineDecodingModel, os.path.join(self.projectPath.resultsPath,"training_1","savedir"))

        return trainLosses


    def test(self,linearizationFunction,saveFolder="resultInference",useTrain=False,useSpeedFilter=True):
        self.model.load_weights(os.path.join(self.projectPath.resultsPath,"training_1/cp.ckpt"))
        ### Loading and inferring
        print("INFERRING")
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)

        speed_mask = behavior_data["Times"]["speedFilter"]
        if not useSpeedFilter:
            speed_mask = np.zeros_like(speed_mask) + 1

        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                              num_parallel_calls=tf.data.AUTOTUNE)

        if useTrain:
            epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['trainEpochs'])
        else:
            epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['testEpochs'])
        tot_mask = speed_mask * epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals,batched=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda vals: ( vals, {"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                    "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),
                              num_parallel_calls=tf.data.AUTOTUNE)
        output_test = self.model.predict(dataset,verbose=1) #

        print("gathering true feature")
        datasetPos = dataset.map(lambda x, y: x["pos"],num_parallel_calls=tf.data.AUTOTUNE)
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        print("gathering exact time of spikes")
        datasetTimes = dataset.map(lambda x, y: x["time"],num_parallel_calls=tf.data.AUTOTUNE)
        times = list(datasetTimes.as_numpy_iterator())

        outLoss = np.expand_dims(output_test[2], axis=1)
        featureTrue = np.reshape(fullFeatureTrue, [output_test[0].shape[0], output_test[0].shape[-1]])
        times = np.reshape(times, [output_test[0].shape[0]])

        projPredPos,linearPred = linearizationFunction(output_test[0][:,:2])
        projTruePos,linearTrue = linearizationFunction(featureTrue)

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
        df = pd.DataFrame(projPredPos)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "linearPred.csv"))
        df = pd.DataFrame(projTruePos)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "linearTrue.csv"))
        df = pd.DataFrame(output_test[1])
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "lossPred.csv"))
        df = pd.DataFrame(times)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "timeStepsPred.csv"))

        return {"featurePred": output_test[0], "featureTrue": featureTrue,
                "times": times, "predofLoss" : output_test[1],
                "lossFromOutputLoss" : outLoss, "projPred":projPredPos, "projTruePos":projTruePos,
                "linearPred":linearPred,"linearTrue":linearTrue}


    def sleep_decoding(self,linearizationFunction,areaSimulation,saveFolder="resultSleepDecoding",batch=False):
        # linearizationFunction: function to get the linear variable
        # areaSimulation: a sham simulation is declared of the decoding is made in the areaSimulation
        self.onlineDecodingModel = self.get_model_for_onlineInference(batch=batch)
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                              num_parallel_calls=tf.data.AUTOTUNE)
        if batch:
            dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=batch),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset.cache()
        dataset.prefetch(tf.data.AUTOTUNE)
        # we filter out spike of length 0
        # todo: find why there is such spike in the dataset
        # dataset = dataset.filter(lambda val:val["length"]>0)
        output_test = self.onlineDecodingModel.predict(dataset,verbose=1)

        fig,ax = plt.subplots()
        eps = 10**(-6)
        ax.hist(np.log(output_test[1]+eps),bins=100)
        # ax.set_xscale("log")
        fig.show()
        fig,ax = plt.subplots()
        ax.scatter(output_test[0][:,0],output_test[0][:,1])
        fig.show()


