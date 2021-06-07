import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp

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
            self.denseLoss3 = tf.keras.layers.Dense(self.params.lstmSize,activation=tf.nn.relu)
            self.denseLoss4 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss5 = tf.keras.layers.Dense(self.params.lstmSize, activation=tf.nn.relu)
            self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation,name="predicted_loss")
            self.epsilon = tf.constant(10 ** (-8))

            self.predAbsoluteLinearErrorLayer = tf.keras.layers.Dense(self.params.nb_eval_dropout)

            outputs = self.get_Model()

            self.model = self.mybuild(outputs)
            self.modelSecondTraining = self.build_second_training(outputs)

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

            sumFeatures = tf.math.reduce_sum(allFeatures,axis=1)

            allFeatures = self.dropoutLayer(allFeatures)

            output_seq = self.lstmsNets[0](allFeatures,mask=mymask)
            output_seq = self.dropoutLayer(output_seq)
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.dropoutLayer(output_seq)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output_seq = self.dropoutLayer(output_seq)
            output = self.lstmsNets[3](output_seq, mask=mymask)

            myoutputPos = self.denseFeatureOutput(output)

            outputLoss = self.denseLoss2(self.denseLoss3(self.denseLoss4(self.denseLoss5
                                                (self.denseLoss1(tf.stop_gradient(tf.concat([output,sumFeatures],axis=1)))))))

            # Idea to bypass the fact that we need the loss of the Pos network.
            # We already compute in the main loops the loss of the Pos network by feeding the targetPos to the network
            # to use it in part of the network that predicts the loss.

            posLoss = tf.losses.mean_squared_error(myoutputPos,self.truePos)[:,tf.newaxis]

            idmanifoldloss = tf.identity(tf.math.reduce_mean(posLoss),name="lossOfManifold")
            #remark: we need to also stop the gradient to progagate from posLoss to the network at the stage of
            # the computations for the loss of the loss predictor

            logposLoss = tf.math.log(tf.add(posLoss,self.epsilon))
            lossFromOutputLoss = tf.identity(tf.math.reduce_mean(tf.losses.mean_squared_error(outputLoss,tf.stop_gradient(logposLoss))),name="lossOfLossPredictor")
        return  myoutputPos, outputLoss, idmanifoldloss , lossFromOutputLoss

    def mybuild(self, outputs,modelName="model.png"):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.truePos,self.inputGroups,self.zeroForGather],
                               outputs=outputs)
        # tf.keras.utils.plot_model(
        #     model, to_file=modelName, show_shapes=True
        # )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]), # Initially compile with first lr.
            loss={
                "tf.identity" : lambda x,y:y, #tf_op_layer_lossOfManifold
                "tf.identity_1" : lambda x,y:y, #tf_op_layer_lossOfLossPredictor
            },
        )
        return model


    def build_second_training(self,outputs,modelName="modelSecondTraining.png"):
        # The training is divided in two step: on training on the training set
        # and an additional training on a second training set, where the network prediction error
        # should be similar to prediction error on the test set.
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.truePos,self.inputGroups,self.zeroForGather],
                               outputs=[outputs[3]])
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.params.learningRates[0]),  # Initially compile with first lr.
            loss={
                "tf.identity_1": lambda x, y: y,  # tf_op_layer_lossOfLossPredictor
            },
        )
        return model

    def fix_linearizer(self,mazePoints,tsProj):
        ## For the linearization we define two fixed inputs:
        self.mazePoints = tf.keras.layers.Input(tensor=mazePoints, name="zeroForGather")
        self.tsProj = tf.keras.layers.Input(tensor=tsProj, name="zeroForGather")


    def get_model_for_uncertainty_estimate(self,modelName="uncertaintyModel.png",batch=False,batch_size = -1):
        # For each inputs we will run them 100 times through the model while keeping the dropout On
        if batch and batch_size==-1:
            batch_size = self.params.batch_size

        with tf.device(self.device_name):
            allFeatures = []
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                x = self.spikeNets[group].apply(x)
                filledFeatureTrain = tf.gather(tf.concat([self.zeroForGather ,x],axis=0),self.indices[group],axis=0)
                filledFeatureTrain = tf.reshape(filledFeatureTrain, [batch_size, -1,self.params.nFeatures])
                allFeatures.append(filledFeatureTrain)
            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2)

            # We repeat each inputs:
            allFeatures = tf.repeat(allFeatures,repeats=[self.params.nb_eval_dropout for _ in range(batch_size)],axis=0)

            #We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups,[batch_size,-1])
            mymask = tf.not_equal(batchedInputGroups,-1)

            mymask = tf.repeat(mymask,repeats=[self.params.nb_eval_dropout for _ in range(batch_size)],axis=0)
            allFeatures = tf.ensure_shape(allFeatures,[batch_size*self.params.nb_eval_dropout,None,self.params.nFeatures*self.params.nGroups])

            sumFeatures = tf.math.reduce_sum(allFeatures,axis=1)
            allFeatures = self.dropoutLayer(allFeatures,training=True)
            output_seq = self.lstmsNets[0](allFeatures,mask=mymask)
            output_seq = self.dropoutLayer(output_seq,training=True)
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.dropoutLayer(output_seq,training=True)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output_seq = self.dropoutLayer(output_seq,training=True)
            output = self.lstmsNets[3](output_seq, mask=mymask)

            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss3(self.denseLoss4(self.denseLoss5
                                                (self.denseLoss1(tf.stop_gradient(tf.concat([output,sumFeatures],axis=1)))))))

            # Now we reduce to get the mean and variance of each estimate over the 100 droupouts:
            myoutputPos = tf.reshape(myoutputPos,[1,self.params.nb_eval_dropout,batch_size,-1])
            outputLoss = tf.reshape(outputLoss, [1,self.params.nb_eval_dropout, batch_size, -1])

            # mean_outputPos = tf.math.reduce_mean(myoutputPos,axis=0)
            # std_outputPos = tf.math.reduce_std(myoutputPos,axis=0)
            # mean_outputLoss = tf.math.reduce_mean(outputLoss,axis=0)
            # std_outputLoss = tf.math.reduce_mean(outputLoss,axis=0)

        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.inputGroups,self.zeroForGather],
                               outputs=[myoutputPos,outputLoss]) #[mean_outputPos,std_outputPos,mean_outputLoss,std_outputLoss]
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile()
        return model

    # todo: :
    # initialize a final layer with the result of the Ridge regression to project more efficiently during sleep
    # ---> Problem: we will also need to run pykeops .
    def get_model_for_uncertainty_estimate_insleep(self,modelName="uncertaintyModel_sleep.png",batch=False,batch_size = -1):
        # For each inputs we will run them 100 times through the model while keeping the dropout On
        if batch and batch_size==-1:
            batch_size = self.params.batch_size

        with tf.device(self.device_name):
            allFeatures = []
            for group in range(self.params.nGroups):
                x = self.inputsToSpikeNets[group]
                x = self.spikeNets[group].apply(x)
                filledFeatureTrain = tf.gather(tf.concat([self.zeroForGather ,x],axis=0),self.indices[group],axis=0)
                filledFeatureTrain = tf.reshape(filledFeatureTrain, [batch_size, -1,self.params.nFeatures])
                allFeatures.append(filledFeatureTrain)
            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2)

            # We repeat each inputs:
            allFeatures = tf.repeat(allFeatures,repeats=[self.params.nb_eval_dropout for _ in range(batch_size)],axis=0)

            #We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups,[batch_size,-1])
            mymask = tf.not_equal(batchedInputGroups,-1)

            mymask = tf.repeat(mymask,repeats=[self.params.nb_eval_dropout for _ in range(batch_size)],axis=0)
            allFeatures = tf.ensure_shape(allFeatures,[batch_size*self.params.nb_eval_dropout,None,self.params.nFeatures*self.params.nGroups])

            sumFeatures = tf.math.reduce_sum(allFeatures,axis=1)
            allFeatures = self.dropoutLayer(allFeatures,training=True)
            output_seq = self.lstmsNets[0](allFeatures,mask=mymask)
            output_seq = self.dropoutLayer(output_seq,training=True)
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.dropoutLayer(output_seq,training=True)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output_seq = self.dropoutLayer(output_seq,training=True)
            output = self.lstmsNets[3](output_seq, mask=mymask)

            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss3(self.denseLoss4(self.denseLoss5
                                                (self.denseLoss1(tf.stop_gradient(tf.concat([output,sumFeatures],axis=1)))))))

            # Now we reduce to get the mean and variance of each estimate over the 100 droupouts:
            myoutputPos = tf.reshape(myoutputPos,[1,self.params.nb_eval_dropout,batch_size,-1])
            outputLoss = tf.reshape(outputLoss, [1,self.params.nb_eval_dropout, batch_size, -1])

            ## Next we find the linearization corresponding:
            bestLinearPos = tf.argmin(tf.reduce_sum(tf.square(myoutputPos[1,None,:,:,:] - self.mazePoints[1,:,None,:,:]),axis=-1),axis=1)
            LinearPos = tf.gather(self.tsProj,bestLinearPos)
            med_linearPos = tfp.stats.percentile(LinearPos,50,axis=2) # compute the median!
            pred_absoluteLinearError = self.predAbsoluteLinearErrorLayer(med_linearPos)


            # mean_outputPos = tf.math.reduce_mean(myoutputPos,axis=0)
            # std_outputPos = tf.math.reduce_std(myoutputPos,axis=0)
            # mean_outputLoss = tf.math.reduce_mean(outputLoss,axis=0)
            # std_outputLoss = tf.math.reduce_mean(outputLoss,axis=0)

        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.inputGroups,self.zeroForGather],
                               outputs=[myoutputPos,outputLoss]) #[mean_outputPos,std_outputPos,mean_outputLoss,std_outputLoss]
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile()
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
            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2)

            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups, [batch_size, -1])
            mymask = tf.not_equal(batchedInputGroups, -1)

            sumFeatures = tf.math.reduce_sum(allFeatures, axis=1)

            output_seq = self.lstmsNets[0](allFeatures, mask=mymask)
            tf.ensure_shape(output_seq, [batch_size, None, self.params.lstmSize])
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output = self.lstmsNets[3](output_seq, mask=mymask)
            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss3(self.denseLoss4(self.denseLoss5
                                                (self.denseLoss1(tf.stop_gradient(tf.concat([output,sumFeatures],axis=1)))))))


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


    def train(self,onTheFlyCorrection=False):
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
        if onTheFlyCorrection:
            maxPos = np.max(behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"],axis=1)))])
            dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"]/maxPos))
        dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

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
        # d = list(dataset.take(1).map(lambda val, valout: val["groups"]).as_numpy_iterator())
        # d = list(dataset.take(1).map(lambda val, valout: val["length"]).as_numpy_iterator())
        # d = list(dataset.take(1).map(lambda val, valout: val["group1"]).as_numpy_iterator())
        # d1 = list(dataset.take(1).map(lambda val, valout: [val["group1"],val["time"]]).as_numpy_iterator())
        # fig,ax = plt.subplots(5)
        # [[ax[id10].plot(d1[0][0][50+id10,id,:]) for id in range(8)] for id10 in range(5)]
        # fig.show()
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

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[callbackLR, csv_logger, cp_callback], # , tb_callback,cp_callback
                              ) #steps_per_epoch = int(self.params.nSteps / self.params.nEpochs)

        trainLosses = np.transpose(np.stack([hist.history["tf.identity_loss"], #tf_op_layer_lossOfManifold
                                      hist.history["tf.identity_1_loss"]]))  #tf_op_layer_lossOfLossPredictor_loss

        df = pd.DataFrame(trainLosses)
        df.to_csv(os.path.join(self.projectPath.resultsPath, "resultInference", "lossTraining.csv"))
        fig,ax = plt.subplots(2,1)
        ax[0].plot(trainLosses[:,0])
        ax[0].set_title("position loss")
        ax[1].plot(trainLosses[:,1])
        ax[1].set_title("log loss prediction loss")
        plt.show()
        fig.savefig(os.path.join(self.projectPath.resultsPath, "lossTraining.png"))

        ## Stage 2: we perform a second training if there is some defined lossPredSetEpochs:
        if len(behavior_data["Times"]["lossPredSetEpochs"])>0:
            ndataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
            ndataset = ndataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                                    num_parallel_calls=tf.data.AUTOTUNE)
            epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['lossPredSetEpochs'])
            # Warning: we use all datapoints, even at low speed here ! TODO: test the influence of this decision
            #TODO:now changed, consider the result of this change
            speed_mask = behavior_data["Times"]["speedFilter"]
            tot_mask = epochMask*speed_mask
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                    tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
            dataset = ndataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
            if onTheFlyCorrection:
                maxPos = np.max(
                    behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
                dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
            dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))
            dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
            dataset = dataset.map(
                lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True),
                num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(lambda vals: (vals, {"tf.identity_1": tf.zeros(self.params.batch_size)}),
                                  num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(self.params.nSteps, reshuffle_each_iteration=True).cache()
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            callbackLR = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
            csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.resultsPath, 'training.log'))
            checkpoint_path = os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt")
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            self.modelSecondTraining.load_weights(os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt"))

            hist = self.modelSecondTraining.fit(dataset,
                      epochs=self.params.nEpochs,
                      callbacks=[callbackLR, csv_logger, cp_callback])
            trainLosses_2 = np.transpose(np.stack([hist.history["loss"]]))  #tf_op_layer_lossOfLossPredictor_loss

            df = pd.DataFrame(trainLosses_2)
            df.to_csv(os.path.join(self.projectPath.resultsPath, "resultInference", "lossTraining2.csv"))
            fig,ax = plt.subplots()
            ax.plot(trainLosses_2[:,0])
            plt.show()
            fig.savefig(os.path.join(self.projectPath.resultsPath, "lossTraining2.png"))

            # print("saving model in savedmodel format, for c++")
            # if not os.path.isdir(os.path.join(self.projectPath.resultsPath,"training_1","savedir")):
            #     os.makedirs(os.path.join(self.projectPath.resultsPath,"training_1","savedir"))
            # self.onlineDecodingModel.load_weights(os.path.join(self.projectPath.resultsPath,"training_2/cp.ckpt"))
            # tf.saved_model.save(self.onlineDecodingModel, os.path.join(self.projectPath.resultsPath,"training_1","savedir"))

            return trainLosses, trainLosses_2
        else:
            # print("saving model in savedmodel format, for c++")
            # if not os.path.isdir(os.path.join(self.projectPath.resultsPath,"training_1","savedir")):
            #     os.makedirs(os.path.join(self.projectPath.resultsPath,"training_1","savedir"))
            # self.onlineDecodingModel.load_weights(os.path.join(self.projectPath.resultsPath,"training_1/cp.ckpt"))
            # tf.saved_model.save(self.onlineDecodingModel, os.path.join(self.projectPath.resultsPath,"training_1","savedir"))
            return trainLosses


    def test(self,linearizationFunction,saveFolder="resultInference",useTrain=False,useSpeedFilter=True,onTheFlyCorrection=False,forceFirstTrainingWeight=False):
        ### Loading and inferring
        print("INFERRING")
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)

        if len(behavior_data["Times"]["lossPredSetEpochs"])>0 and not forceFirstTrainingWeight:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt"))
        else:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt"))


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

        if onTheFlyCorrection:
            maxPos = np.max(behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
        dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

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

        datasetPos_index = dataset.map(lambda x, y: x["pos_index"],num_parallel_calls=tf.data.AUTOTUNE)
        pos_index = list(datasetPos_index.as_numpy_iterator())

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
        df = pd.DataFrame(pos_index)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "pos_index.csv"))

        return {"featurePred": output_test[0], "featureTrue": featureTrue,
                "times": times, "predofLoss" : output_test[1],
                "lossFromOutputLoss" : outLoss, "projPred":projPredPos, "projTruePos":projTruePos,
                "linearPred":linearPred,"linearTrue":linearTrue,"pos_index":pos_index}


    def sleep_decoding(self,linearizationFunction,areaSimulation,behavior_data,
                       saveFolder="resultSleepDecoding",batch=False,forceFirstTrainingWeight=False,
                       batch_size=-1):
        if batch and batch_size==-1:
            batch_size= self.params.batch_size
        # linearizationFunction: function to get the linear variable
        # areaSimulation: a sham simulation is declared of the decoding is made in the areaSimulation

        #first: initialize the layer with the right parameters learned from training
        # Note: it seems that trying to load weights into the online_decoding architecture fails

        if len(behavior_data["Times"]["lossPredSetEpochs"])>0 and not forceFirstTrainingWeight:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt"))
        else:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt"))

        # Build the online decoding model with the layer already initialized:
        # self.onlineDecodingModel = self.get_model_for_onlineInference(batch=batch)

        self.onlineDecodingModel = self.get_model_for_uncertainty_estimate(batch=batch,batch_size=batch_size)

        #Remark: other solution: at training time, build and save the onlineDecodingModel
        # but now we can save these weights, so that they are used in c++....

        print("decoding sleep epochs")
        outputDic = {}
        for idsleep,sleepName in enumerate(behavior_data["Times"]["sleepNames"]):
            timeSleepStart = behavior_data["Times"]["sleepEpochs"][2*idsleep]
            timeSleepStop = behavior_data["Times"]["sleepEpochs"][2*idsleep+1]
            dataset = tf.data.TFRecordDataset(self.projectPath.tfrecSleep)
            dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                                  num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.filter(lambda x: tf.math.logical_and(tf.math.less_equal(x["time"],timeSleepStop),
                                                                   tf.math.greater_equal(x["time"],timeSleepStart)))
            if batch:
                dataset = dataset.batch(batch_size, drop_remainder=True)
            bs_store = self.params.batch_size
            self.params.batch_size = batch_size
            dataset = dataset.map(
                lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=batch),
                num_parallel_calls=tf.data.AUTOTUNE)
            self.params.batch_size = bs_store
            dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset.cache()
            dataset.prefetch(tf.data.AUTOTUNE)
            # we filter out spike of length 0
            # todo: find why there is such spike in the dataset --> Note: seems to have been solved....
            # dataset = dataset.filter(lambda val:val["length"]>0)
            output_test = self.onlineDecodingModel.predict(dataset,verbose=1)

            dtime = dataset.map(lambda vals: vals["time"])
            timePred = list(dtime.as_numpy_iterator())
            timePreds = np.ravel(timePred)
            # output_test += [timePreds]

            medianLinearPos,confidence = self.sleep_uncertainty_estimate(output_test,linearizationFunction)

            outputDic[sleepName] = [medianLinearPos,confidence,timePreds]

        #
        # fig,ax = plt.subplots(len(outputDic.keys()),2)
        # for id,k in enumerate(outputDic.keys()):
        #     ax[id,0].hist(outputDic[k][1][:],bins=1000)
        #     ax[id,0].set_title(k)
        #     ax[id,0].set_xlabel("decoded loss")
        #     ax[id,0].set_ylabel("histogram")
        #     ax[id,1].hist(outputDic[k][1][:],bins=1000)
        #     ax[id,1].set_title(k)
        #     ax[id,1].set_xlabel("decoded loss")
        #     ax[id,1].set_ylabel("histogram")
        #     ax[id,1].set_yscale("log")
        # fig.tight_layout()
        # fig.show()
        #
        # fig,ax = plt.subplots(len(outputDic.keys()),2,figsize=(5,9))
        # for id, k in enumerate(outputDic.keys()):
        #     ax[id,0].scatter(outputDic[k][0][:,0],outputDic[k][0][:,1],alpha=0.1,s=0.1)
        #     errorPred = outputDic[k][1][:,0]
        #     thresh = np.quantile(errorPred,0.1)
        #     ax[id,1].scatter(outputDic[k][0][errorPred<thresh,0],outputDic[k][0][errorPred<thresh,1],alpha=1,s=0.1)
        #     ax[id,0].set_xlabel("predicted X")
        #     ax[id,0].set_ylabel("predicted Y")
        #     ax[id,1].set_xlabel("predicted X")
        #     ax[id,1].set_ylabel("predicted Y")
        #     ax[id,0].set_title(k+ " ;all predictions" )
        #     ax[id,1].set_title(k + " ;filtered prediction \n by predicted loss")
        #     ax[id,0].set_aspect(1)
        #     ax[id,1].set_aspect(1)
        # fig.tight_layout()
        # fig.show()
        #
        # # let us plot the prediction in time...

        # cm = plt.get_cmap("turbo")
        # fig, ax = plt.subplots(len(outputDic.keys()), 3, figsize=(30,20))
        # for id, k in enumerate(outputDic.keys()):
        #     delta = 10
        #     maxLossPred = np.max(outputDic[k][1])
        #     minLossPred = np.min(outputDic[k][1])
        #     ax[id,0].scatter(outputDic[k][2][1:-1:delta],outputDic[k][0][1:-1:delta,0],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
        #     ax[id,1].scatter(outputDic[k][2][1:-1:delta],outputDic[k][0][1:-1:delta,1],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
        #     ax[id,2].scatter(outputDic[k][2][1:-1:delta],outputDic[k][1][1:-1:delta,0],s=1,c=cm((outputDic[k][1][1:-1:delta,0]-minLossPred)/(maxLossPred-minLossPred)))
        #     ax[id,1].set_xlabel("time")
        #     ax[id,1].set_ylabel("predicted Y")
        #     ax[id,0].set_ylabel("predicted X")
        #     ax[id,2].set_ylabel("predicted loss")
        # fig.show()
        #
        # fig, ax = plt.subplots(len(outputDic.keys()), figsize=(5, 9))
        # for id, k in enumerate(outputDic.keys()):
        #     delta = 10
        #     myfilter = (outputDic[k][1] < np.quantile(outputDic[k][1], 1))[:, 0]
        #     maxLossPred = np.max(np.clip(outputDic[k][1][myfilter,0],-10,1))
        #     minLossPred = np.min(np.clip(outputDic[k][1][myfilter,0],-10,1))
        #     normedLogLoss = (np.clip(outputDic[k][1][myfilter,0][1:-1:delta],-10,1)-minLossPred)/(maxLossPred-minLossPred)
        #     ax[id].scatter(outputDic[k][0][myfilter,0][1:-1:delta],outputDic[k][0][myfilter,1][1:-1:delta],alpha=0.5,s=1,c=cm(normedLogLoss))
        #     ax[id].set_xlabel("predicted X")
        #     ax[id].set_ylabel("predicted Y")
        #     ax[id].set_title(k)
        #     fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=minLossPred,vmax=maxLossPred),cmap=cm), label="Log Loss Pred; clipped" ,ax=ax[id])
        # fig.tight_layout()
        # fig.show()

        print("Ended sleep analysis")

        return outputDic


    def get_model_for_CNN_study(self, modelName="onlineDecodingModel.png", batch=False,batch_size=1):
        # the online inference model, it works as the training model but with a batch_size of 1 or params.batch_size
        if batch:
            batch_size = batch_size # self.params.batch_size

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
            allFeatures = tf.tuple(tensors=allFeatures)
            allFeatures = tf.concat(allFeatures, axis=2)

            # We would like to mask timesteps that were added for batching purpose, before running the RNN
            batchedInputGroups = tf.reshape(self.inputGroups, [batch_size, -1])
            mymask = tf.not_equal(batchedInputGroups, -1)

            sumFeatures = tf.math.reduce_sum(allFeatures, axis=1)

            output_seq = self.lstmsNets[0](allFeatures, mask=mymask)
            tf.ensure_shape(output_seq, [batch_size, None, self.params.lstmSize])
            output_seq = self.lstmsNets[1](output_seq, mask=mymask)
            output_seq = self.lstmsNets[2](output_seq, mask=mymask)
            output = self.lstmsNets[3](output_seq, mask=mymask)
            myoutputPos = self.denseFeatureOutput(output)
            outputLoss = self.denseLoss2(self.denseLoss3(self.denseLoss4(self.denseLoss5
                                                                         (self.denseLoss1(tf.stop_gradient(
                                                                             tf.concat([output, sumFeatures],
                                                                                       axis=1)))))))
        model = tf.keras.Model(
            inputs=self.inputsToSpikeNets + self.indices + [self.inputGroups, self.zeroForGather],
            outputs=[myoutputPos, outputLoss,allFeatures])
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile()
        return model


    def study_CNN_outputs(self,batch=False,forceFirstTrainingWeight=False,
                          useSpeedFilter=False,useTrain=False,onTheFlyCorrection=True):
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)
        if len(behavior_data["Times"]["lossPredSetEpochs"]) > 0 and not forceFirstTrainingWeight:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt"))
        else:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt"))

        # Build the online decoding model with the layer already initialized:
        batch_size = 1000
        self.onlineDecodingModel = self.get_model_for_CNN_study(batch=batch,batch_size=batch_size)

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

        if onTheFlyCorrection:
            maxPos = np.max(
                behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
        dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

        dataset = dataset.batch(batch_size, drop_remainder=True)
        # drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda vals: (vals, {"tf_op_layer_lossOfManifold": tf.zeros(batch_size),
                                                   "tf_op_layer_lossOfLossPredictor": tf.zeros(batch_size)}),
                              num_parallel_calls=tf.data.AUTOTUNE)

        datasetOneBatch = dataset.take(1)
        output_test = self.onlineDecodingModel.predict(datasetOneBatch, verbose=1)  #

        import sklearn.decomposition
        from sklearn.manifold import TSNE
        groupFeatures = []
        svalues = []
        explainedVariances = []
        transformedFeatures = []
        tsneFeatures = []
        for idg in range(self.params.nGroups):
            groupFeatures.append(np.reshape(output_test[2][:,:,idg*128:(idg+1)*128],[np.prod(output_test[2].shape[0:2]),128]))
            res = sklearn.decomposition.PCA()
            res.fit(groupFeatures[idg][np.sum(np.abs(groupFeatures[idg]),axis=1)>0,:])
            svalues.append(res.singular_values_)
            explainedVariances.append(res.explained_variance_ratio_)
            transformedFeatures.append(res.transform(groupFeatures[idg][np.sum(np.abs(groupFeatures[idg]),axis=1)>0,:]))

            tsne = TSNE(n_components=2)
            tsneFeatures.append(tsne.fit_transform(transformedFeatures[idg]))

        from matplotlib import  pyplot as plt
        fig,ax =plt.subplots()
        ax.hist(np.argmax(transformedFeatures[0],axis=1),bins=100) #[np.sum(np.abs(groupFeatures[3]),axis=1)>0,:]
        # ax.set_aspect(transformedFeatures[0].shape[1]/transformedFeatures[0].shape[0])
        fig.show()

        fig,ax = plt.subplots(len(tsneFeatures),figsize=(4,20))
        cm = plt.get_cmap("tab20")
        for idg in range(self.params.nGroups):
            bestsvd = np.argmax(transformedFeatures[idg][:,0:20],axis=1)
            ax[idg].scatter(tsneFeatures[idg][:,0],tsneFeatures[idg][:,1],s=1,c=cm(bestsvd))
            ax[idg].set_aspect(1)
        fig.tight_layout()
        fig.show()

        fig,ax = plt.subplots(3,2)
        for idg in range(self.params.nGroups):
            ax[0,0].plot(range(128),svalues[idg])
            ax[1,0].plot(range(128), explainedVariances[idg])
            ax[2,0].plot(range(128), explainedVariances[idg])
            ax[0,1].plot(range(12),svalues[idg][:12])
            ax[1,1].plot(range(12), explainedVariances[idg][:12])
            ax[2,1].plot(range(12), explainedVariances[idg][:12])
        ax[0,0].set_yscale("log")
        ax[0,1].set_yscale("log")
        ax[2,0].set_yscale("log")
        ax[2,1].set_yscale("log")
        fig.show()

        fig,ax = plt.subplots()
        gf = np.ravel(groupFeatures[0])
        ax.hist(gf[np.not_equal(gf,0)],bins=100)
        fig.show()

        norm = lambda x : (x-np.min(x,axis=1))/(np.max(x,axis=1)-np.min(x,axis=1))

        gf  = groupFeatures[0][np.sum(np.abs(groupFeatures[0]),axis=1)>0,:]

        #let us reorganize all the spikes feature by their PCA argmax:
        #TODO


        corrMat = np.matmul(norm(gf),np.transpose(norm(gf)))
        fig,ax = plt.subplots()
        # cm = plt.get_cmap("Reds")
        ax.imshow(corrMat)
        ax.set_aspect(corrMat.shape[1]/corrMat.shape[0])
        fig.show()

        fig,ax = plt.subplots()
        cm = plt.get_cmap("Reds")
        ax.matshow(transformedFeatures[0][:,0:12],cmap=cm)
        ax.set_aspect(transformedFeatures[0][:,0:12].shape[1]/transformedFeatures[0].shape[0])
        fig.show()

        #More interestingly: let us compare the cluster_data results with the spike being considered
        #todo: confront the two filtering by making sure the time step difference is less than 15/sampling_rate.
        from importData import ImportClusters
        cluster_data = ImportClusters.load_spike_sorting(self.projectPath)
        spike_times = cluster_data["Spike_times"]
        maskTime = inEpochsMask(spike_times[0][:,0], behavior_data['Times']['testEpochs'])
        spike_labels = cluster_data["Spike_labels"]

        sl = spike_labels[0][maskTime]
        st = spike_times[0][maskTime]
        #some spikes are not assigned to any clusters (noisy spikes). we give them the label 0
        cluster_labels = np.zeros(spike_labels[0].shape[0])
        cluster_labels[np.sum(sl,axis=1)>0] = np.argmax(sl[np.sum(sl,axis=1)>0,:],axis=1)+1
        fig,ax = plt.subplots()
        ax.hist(cluster_labels,width=0.5,bins=range(spike_labels[0].shape[1]+1))
        ax.set_yscale("log")
        fig.show()

        #next we extract spike time from what is feeded to tensorflow:
        datasetTimes = datasetOneBatch.map(lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE)
        times = list(datasetTimes.as_numpy_iterator())
        times = np.array(times)[0,:]





        print("gathering true feature")
        datasetPos = dataset.map(lambda x, y: x["pos"], num_parallel_calls=tf.data.AUTOTUNE)
        fullFeatureTrue = list(datasetPos.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)
        print("gathering exact time of spikes")
        datasetTimes = dataset.map(lambda x, y: x["time"], num_parallel_calls=tf.data.AUTOTUNE)
        times = list(datasetTimes.as_numpy_iterator())
        times = np.array(times)
        datasetPos_index = dataset.map(lambda x, y: x["pos_index"], num_parallel_calls=tf.data.AUTOTUNE)
        pos_index = list(datasetPos_index.as_numpy_iterator())

        outLoss = np.expand_dims(output_test[1], axis=1)
        featureTrue = np.reshape(fullFeatureTrue, [output_test[0].shape[0], output_test[0].shape[-1]])
        times = np.reshape(times, [output_test[0].shape[0]])



        convNetFeature = np.stack(output_test[2])

        pca(output_test[2])

        import sklearn.manifold.t_sne as tsne



        tsne()

        # projPredPos, linearPred = linearizationFunction(output_test[0][:, :2])
        # projTruePos, linearTrue = linearizationFunction(featureTrue)

        #What we would like to do:
        # Step0: PCA?
        # Step1: try a TSNE to see if clear clusters
            #Note: stack over all spikes....
        # Step2: k-nn cluster with k=the number of cluster from PCA

    def fit_uncertainty_estimate(self,linearizationFunction,batch=False,forceFirstTrainingWeight=False,
                          useSpeedFilter=False,useTrain=False,onTheFlyCorrection=True):
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)
        if len(behavior_data["Times"]["lossPredSetEpochs"]) > 0 and not forceFirstTrainingWeight:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt"))
        else:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt"))

        # Build the online decoding model with the layer already initialized:
        self.uncertainty_estimate_model = self.get_model_for_uncertainty_estimate(batch=batch)

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

        if onTheFlyCorrection:
            maxPos = np.max(
                behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
        dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        # drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda vals: (vals, {"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                   "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),
                              num_parallel_calls=tf.data.AUTOTUNE)

        output_test = self.uncertainty_estimate_model.predict(dataset, verbose=1)
        euclidData = np.reshape(output_test[0],[np.prod(output_test[0].shape[0:3]),2])
        projectedPos,linearPos = linearizationFunction(euclidData.astype(np.float64))

        linearPos = np.reshape(linearPos,output_test[0].shape[0:3])
        medianLinearPos = np.median(linearPos,axis=1)
        medianLinearPos = np.reshape(medianLinearPos, [np.prod(medianLinearPos.shape[0:2])])

        d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
        truePosFed = np.array(d0)
        truePosFed = truePosFed.reshape([truePosFed.shape[0]*truePosFed.shape[1],2])
        trueProjPos,trueLinearPos = linearizationFunction(truePosFed)

        linearTranspose = np.transpose(linearPos,axes=[0,2,1])
        linearTranspose = linearTranspose.reshape([linearTranspose.shape[0]*linearTranspose.shape[1],linearTranspose.shape[2]])
        histPosPred = np.stack([np.histogram(np.abs(linearTranspose[id,:]-np.median(linearTranspose[id,:])),bins=np.arange(0,stop=1,step=0.01))[0] for id in range(linearTranspose.shape[0])])

        # we first sort by error:
        sortPerm = np.argsort(np.abs(medianLinearPos-trueLinearPos))
        reorderedHist = histPosPred[sortPerm]
        fig,ax = plt.subplots()
        ax.imshow(reorderedHist)
        ax.set_aspect(reorderedHist.shape[1]/reorderedHist.shape[0])
        ax.set_xlabel("histogram of absolute distance to median")
        axy = ax.twiny()
        axy.plot(np.abs(medianLinearPos-trueLinearPos)[sortPerm],range(sortPerm.shape[0]),c="red",alpha=0.5)
        axy.set_xlabel("absolute decoding linear error")
        ax.set_ylabel("time step - \n reordered by decoding error")
        # ax[1].set_aspect(reorderedHist.shape[1]/(np.abs(output_test[0]-trueLinearPos).max()))
        fig.show()

        AbsErrorToMedian = np.abs(linearTranspose - np.median(linearTranspose,axis=1)[:,None])
        linearEnsembleError = np.abs(medianLinearPos-trueLinearPos)

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        clf = Ridge(alpha=1000)
        X_train,X_test,Y_train,Y_test = train_test_split(AbsErrorToMedian,linearEnsembleError,train_size=0.3)
        clf.fit(X_train,Y_train)
        self.LinearNetworkConfidence = clf
        self.predAbsoluteLinearErrorLayer.set_weights(self.LinearNetworkConfidence.coef_)

        clf.predict(X_test)
        predfromTrain2 = clf.predict(X_test)
        fig,ax = plt.subplots(2,1)
        ax[0].scatter(predfromTrain2,Y_test,alpha=0.5,s=1,c="black")
        ax[1].hist(predfromTrain2,bins=50,color="blue",alpha=0.5,density=True)
        # ax[1].hist(Y_test, bins=100, color="orange", alpha=0.5,density=True)
        ax[0].set_xlabel("prediction of linear error, \n regularized linear prediction \n from absolute error to median")
        ax[0].set_ylabel("true linear error")
        fig.tight_layout()
        fig.show()


        wakeConfidence = clf.predict(AbsErrorToMedian)
        fig,ax = plt.subplots()
        ax.hist(wakeConfidence,bins=100)
        ax.set_title("wake set predicted confidence \n (== infered absolute linear error from absolute error to median)")
        fig.show()

        fig,ax = plt.subplots()
        ax.hist(medianLinearPos,bins=50)
        ax.set_title("wake set predicted linear pos")
        fig.show()

    def sleep_uncertainty_estimate(self,output_test,linearizationFunction):
        #output_test: euclid_data,lossPred (not used anymore),time steps
        euclidData = np.reshape(output_test[0],[np.prod(output_test[0].shape[0:3]),2])
        projectedPos,linearPos = linearizationFunction(euclidData.astype(np.float64))
        linearPos = np.reshape(linearPos,output_test[0].shape[0:3])
        medianLinearPos = np.median(linearPos,axis=1)

        # next we estimate the error made, by using the Linear projection of the distance to the median:
        linearTranspose = np.transpose(linearPos,axes=[0,2,1])
        linearTranspose = linearTranspose.reshape([linearTranspose.shape[0]*linearTranspose.shape[1],linearTranspose.shape[2]])
        AbsErrorToMedian = np.abs(linearTranspose - np.median(linearTranspose,axis=1)[:,None])
        predictedConfidence = self.LinearNetworkConfidence.predict(AbsErrorToMedian)

        return medianLinearPos,predictedConfidence

    def study_sleep_uncertainty_estimate(self,output_test,linearizationFunction):
        #output_test: euclid_data,lossPred (not used anymore),time steps
        euclidData = np.reshape(output_test[0],[np.prod(output_test[0].shape[0:3]),2])
        projectedPos,linearPos = linearizationFunction(euclidData.astype(np.float64))
        linearPos = np.reshape(linearPos,output_test[0].shape[0:3])
        medianLinearPos = np.median(linearPos,axis=1)

        fig,ax = plt.subplots()
        [ax.scatter(output_test[-1][1:1000:1],np.ravel(linearPos[:,id,:])[1:1000:1],c="orange",s=1,alpha=0.2) for id in range(linearPos.shape[1])]
        ax.plot(output_test[-1][1:1000:1],np.ravel(medianLinearPos)[1:1000:1],c="red")
        ax.set_xlabel("time")
        ax.set_ylabel("decoded linear position")
        ax.set_title("beginning of sleep")
        fig.show()

        # next we estimate the error made, by using the Linear projection of the distance to the median:
        linearTranspose = np.transpose(linearPos,axes=[0,2,1])
        linearTranspose = linearTranspose.reshape([linearTranspose.shape[0]*linearTranspose.shape[1],linearTranspose.shape[2]])
        AbsErrorToMedian = np.abs(linearTranspose - np.median(linearTranspose,axis=1)[:,None])
        predictedConfidence = self.LinearNetworkConfidence.predict(AbsErrorToMedian)

        cm = plt.get_cmap("turbo")
        fig,ax = plt.subplots()
        ax.plot(output_test[-1][1:1000:1],np.ravel(medianLinearPos)[1:1000:1],c="grey",alpha=0.3)
        ax.scatter(output_test[-1][1:1000:1],np.ravel(medianLinearPos)[1:1000:1],c=cm(predictedConfidence[1:1000:1]/np.max(predictedConfidence)),s=3)
        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,np.max(predictedConfidence)),cmap=cm),label="predicted confidence")
        ax.set_xlabel("time")
        ax.set_ylabel("decoded linear position")
        fig.show()


        cm = plt.get_cmap("turbo")
        fig,ax = plt.subplots()
        ax.plot(output_test[-1][predictedConfidence<0.1],np.ravel(medianLinearPos)[predictedConfidence<0.1],c="grey",alpha=0.3)
        ax.scatter(output_test[-1][predictedConfidence<0.1],np.ravel(medianLinearPos)[predictedConfidence<0.1],c=cm(predictedConfidence[predictedConfidence<0.1]/np.max(predictedConfidence)),s=3)
        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,np.max(predictedConfidence)),cmap=cm),label="predicted confidence")
        ax.set_xlabel("time")
        ax.set_ylabel("decoded linear position")
        fig.show()


        #let us look at the confidence distributions:
        fig,ax = plt.subplots()
        ax.hist(predictedConfidence,bins=100)
        ax.set_title("confidence in sleep")
        ax.set_xlabel("confidence")
        ax.set_ylabel("histogram")
        fig.show()
        #todo: compare sleep and wake confidences

        #let us look at the distribution of linear position jump
        posjump = np.ravel(medianLinearPos)[1:]-np.ravel(medianLinearPos)[:-1]
        fig,ax = plt.subplots()
        ax.hist(np.ravel(medianLinearPos),bins=50,color="red")
        ax.set_xlabel("linear position")
        fig.show()
        fig,ax = plt.subplots()
        ax.hist(posjump,bins=1000,color="red",alpha=0.5)
        ax.set_yscale("log")
        fig.show()
        fig,ax = plt.subplots()
        ax.hist(np.abs(posjump),bins=100,color="red",alpha=0.5)
        ax.set_yscale("log")
        fig.show()

        fig,ax = plt.subplots()
        ax.scatter(output_test[-1][:-1][predictedConfidence[:-1]<0.08],posjump[predictedConfidence[:-1]<0.08],s=1,alpha=0.4)
        fig.show()

        # let us compute the transition probability matrix from one position to another:
        medianLinearPos = np.ravel(medianLinearPos)
        _,binMed = np.histogram(medianLinearPos,bins=10)
        filterForSize = lambda x : x[x<(len(medianLinearPos)-1)]
        findHist= lambda x,j : np.sum((medianLinearPos[x+1]>=binMed[j])*(medianLinearPos[x+1]<binMed[j+1]))
        transMat = [[findHist(filterForSize(np.where((medianLinearPos>=binMed[i]) * (medianLinearPos<binMed[i+1]))[0]),j) for j in range(len(binMed)-1)] for i in range(len(binMed)-1)]
        transMat = np.array(transMat)
        fig,ax = plt.subplots()
        ax.matshow(transMat)
        for i in range(len(binMed)-1):
            for j in range(len(binMed)-1):
                text = ax.text(j, i, transMat[i, j],
                                  ha="center", va="center", color="w")
        ax.set_xticks(range(len(binMed[:-1])))
        ax.set_yticks(range(len(binMed[:-1])))
        ax.set_xticklabels(np.round(binMed[:-1],2))
        ax.set_yticklabels(np.round(binMed[:-1],2))
        fig.show()

        # Looking at data: is seem that a series of small jump is followed by a large jump.
        # let us therefore look at the jump transition matrix:
        absPosJump = np.abs(posjump)
        histAbsPosJump,binJump = np.histogram(absPosJump,bins=100)
        filterForSize = lambda x : x[x<(len(absPosJump)-1)]
        findHist= lambda x,j : np.sum((absPosJump[x+1]>=binJump[j])*(absPosJump[x+1]<binJump[j+1]))
        transMatJump = [[findHist(filterForSize(np.where((absPosJump>=binJump[i]) * (absPosJump<binJump[i+1]))[0]),j) for j in range(len(binJump)-1)] for i in range(len(binJump)-1)]
        transMatJump = np.array(transMatJump)
        # so row corresponds to the state at t
        # so columns corresponds to the state at t+1
        # fig,ax = plt.subplots()
        # ax.scatter(output_test[-1][:-1],np.abs(posjump),s=1)
        # fig.show()

        fig,ax = plt.subplots()
        ax.imshow(transMatJump/histAbsPosJump[:,None]) #effectively normalize each row!
        ax.set_ylabel("jump at t")
        ax.set_xlabel("jump at t+1")
        fig.show()
        #--> jumps are most of the time followed by small jumps....

        # we separate large and small jumps arbitrarily:
        pospeed = posjump/(output_test[-1][1:]-output_test[-1][:-1])
        fig,ax = plt.subplots(2,1)
        ax[0].scatter(output_test[-1][:-1],np.abs(pospeed),s=1,alpha=0.1)
        ax[1].hist(np.log(pospeed[pospeed!=0]),bins=np.arange(-10,10,step=0.1))
        fig.show()




        # Continuity driven by predicted confidence:
        fig,ax = plt.subplots()
        ax.scatter(posjump,predictedConfidence[:-1],s=0.1,alpha=0.3)
        ax.set_xlabel("position jump between two time step")
        ax.set_ylabel("predicted confidence")
        fig.show()
        _,bins = np.histogram(predictedConfidence,bins=100)
        posjump_knowing_confidence = [np.abs(posjump[(predictedConfidence[:-1]>=bins[i])*(predictedConfidence[:-1]<bins[i+1])]) for i in range(len(bins)-1)]
        mposjump_knowing_confidence = [np.mean(p) for p in posjump_knowing_confidence]
        stdposjump_knowing_confidence = [np.std(p) for p in posjump_knowing_confidence]
        fig,ax = plt.subplots()
        ax.plot(bins[:-1],mposjump_knowing_confidence)
        ax.fill_between(bins[:-1], mposjump_knowing_confidence,np.array(mposjump_knowing_confidence)+np.array(stdposjump_knowing_confidence))
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("mean absolute jump")
        fig.show()


        print("ended sleep uncertainty estimate")


    def study_uncertainty_estimate(self,linearizationFunction,batch=False,forceFirstTrainingWeight=False,
                          useSpeedFilter=False,useTrain=False,onTheFlyCorrection=True):
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)
        if len(behavior_data["Times"]["lossPredSetEpochs"]) > 0 and not forceFirstTrainingWeight:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt"))
        else:
            self.model.load_weights(os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt"))

        # Build the online decoding model with the layer already initialized:
        self.uncertainty_estimate_model = self.get_model_for_uncertainty_estimate(batch=batch)

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

        if onTheFlyCorrection:
            maxPos = np.max(
                behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
        dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        # drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda vals: (vals, {"tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                                                   "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),
                              num_parallel_calls=tf.data.AUTOTUNE)

        output_test = self.uncertainty_estimate_model.predict(dataset, verbose=1)
        euclidData = np.reshape(output_test[0],[np.prod(output_test[0].shape[0:3]),2])
        projectedPos,linearPos = linearizationFunction(euclidData.astype(np.float64))
        projectedPos = np.reshape(projectedPos,output_test[0].shape)

        # we can also compute the distance to the projected Pos:
        predPos = np.reshape(euclidData,projectedPos.shape)
        vecToProjPos = predPos-projectedPos
        distToProjPos = np.sqrt(np.sum(np.square(vecToProjPos),axis=-1))
        middlePoint= np.array([0.5,0.5])
        # the second variable sign can be obtained by the sign of the projection on the vector to this middle point from the linearized point
        # of the pred to linear vector
        signOfProj  = np.sign(np.sum((predPos-middlePoint[None,None,None,:])*vecToProjPos,axis=-1))
        signDistToProjPos = distToProjPos*signOfProj

        linearPos = np.reshape(linearPos,output_test[0].shape[0:3])
        from scipy.stats import iqr
        output_test = [np.median(linearPos,axis=1),np.std(linearPos,axis=1),np.mean(linearPos,axis=1),iqr(linearPos,axis=1)]
        output_test = [np.reshape(o,[np.prod(o.shape[0:2])]) for o in output_test]
        output_test_no_dropout = self.model.predict(dataset, verbose=1)

        print(len(output_test))
        speed_data = behavior_data['Speed'][np.where(tot_mask)]

        d1 = list(dataset.map(lambda vals,valsout : vals["pos_index"]).as_numpy_iterator())
        dres = np.ravel(np.array(d1))
        speeds = behavior_data['Speed'][dres]
        truePos = behavior_data['Positions'][dres]

        d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
        truePosFed = np.array(d0)
        truePosFed = truePosFed.reshape([190*52,2])
        trueProjPos,trueLinearPos = linearizationFunction(truePosFed)
        # compute
        trueVecToProjPos = truePosFed-trueProjPos
        trueDistToProjPos = np.sqrt(np.sum(np.square(trueVecToProjPos),axis=-1))
        trueSignOfProj  = np.sign(np.sum((truePosFed-middlePoint[None,:])*trueVecToProjPos,axis=-1))
        trueSignDistToProjPos = trueDistToProjPos*trueSignOfProj

        times = behavior_data['Position_time'][dres]
        fig,ax = plt.subplots(2,1,figsize=(5,10))
        [ax[0].scatter(times,np.ravel(signDistToProjPos[:,id,:]),c="orange",s=1,alpha=0.2) for id in range(signDistToProjPos.shape[1])]
        ax[0].scatter(times, np.ravel(np.median(signDistToProjPos, axis=1)), c="red", s=1)
        ax[0].scatter(times,trueSignDistToProjPos,s=1,c="black")
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("signed distance to linearization line")
        ax[1].scatter(trueSignDistToProjPos,np.ravel(np.median(signDistToProjPos, axis=1)),c="black",s=1)
        ax[1].set_xlabel("true signed distance \n to linearizartion line")
        ax[1].set_ylabel("predicted signed distance \n to linearizartion line")
        fig.show()




        g2 = euclidData.reshape([linearPos.shape[0],linearPos.shape[1],linearPos.shape[2],2])
        g2med = np.reshape(np.median(g2,axis=1),[g2.shape[0]*g2.shape[2],2])
        fig,ax  =plt.subplots(2,1,sharex=True)
        ax[0].plot(times,g2med[:,0],c="red",label="decoded by median of ensemble")
        ax[0].plot(times,truePosFed[:,0],c="black",label="true pos")
        ax[1].plot(times,g2med[:,1],c="red")
        ax[1].plot(times,truePosFed[:,1],c="black")
        ax[0].set_ylabel("X")
        ax[1].set_ylabel("Y")
        ax[1].set_xlabel("time")
        fig.legend()
        fig.show()
        g2res = g2.reshape([190*52,100,2])
        fig,ax = plt.subplots()
        ax.scatter(truePosFed[:, 0], truePosFed[:, 1], c="black", s=1, alpha=0.2)
        ax.scatter(g2med[:,0],g2med[:,1],c="red",s=1)
        [ax.scatter(g2res[:,id,0],g2res[:,id,1],c="orange",s=1,alpha=0.01) for id in range(100)]
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.show()

        # We could do a Maximum Likelihood estimate on the predicted variable:
        from SimpleBayes import  butils
        bw = 0.2
        edges,_ = butils.kdenD(truePosFed,bw,nbins=[20,20])
        def get_mle_estimate(X):
            _,p = butils.kdenD(X,bw,nbins=[20,20],edges=edges)
            xedge = edges[0][:,0]
            yedge = edges[-1][0,:]
            posMLE = np.unravel_index(np.argmax(p),p.shape)
            return [xedge[posMLE[0]],yedge[posMLE[1]]]
        mleDecodedPos = np.array([ get_mle_estimate(g2res[id,:,:]) for id in range(g2res.shape[0])])
        fig,ax = plt.subplots()
        ax.plot(times,mleDecodedPos[:,0],c="red")
        ax.plot(times,truePosFed[:, 0], c="black")
        # ax.scatter(truePosFed[:,1],mleDecodedPos[:,1],s=1,alpha=0.2,c="black")
        fig.show()
        # No good results.



        # fig,ax = plt.subplots()
        # ax.plot(times,output_test[0][:,0],c="red",label="prediction X")
        # ax.plot(times,truePos[:, 0],c="black",label="true X")
        # ax.fill_between(times[:,0],output_test[0][:,0]+output_test[1][:,0],output_test[0][:,0]-output_test[1][:,0],color="orange",label="confidence")
        # ax.set_xlabel("time")
        # ax.set_ylabel("X")
        # fig.legend()
        # fig.show()
        fig,ax = plt.subplots()
        ax.plot(times,output_test[0],c="red",label="median prediction linear")
        # ax[0].plot(times, output_test[2], c="violet", label="mean prediction X")
        ax.plot(times,trueLinearPos,c="black",label="true linear")
        # ax[0].plot(times, output_test[3], c="green", label="iqr prediction X",alpha=0.5)
        fig.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("linear position")
        fig.show()

        fig,ax = plt.subplots()
        ax.plot(times,output_test[0],c="red",label="median prediction X")
        ax.plot(times, output_test[2], c="violet", label="mean prediction X")
        ax.plot(times,trueLinearPos,c="black",label="true X")
        # ax.fill_between(times[:,0],output_test[0]+output_test[1],output_test[0]-output_test[1],color="orange",label="confidence")
        for i in range(100):
            ax.scatter(times[:,0],np.reshape(linearPos[:,i,:],[linearPos.shape[0]*linearPos.shape[2]]),s=1,alpha=0.05,c="orange")
        ax.set_xlabel("time")
        ax.set_ylabel("X")
        fig.legend()
        fig.show()

        #Question: given the distribution of predicted position
        # Are there some particular pattern?
        # to see that: we can look at the distributions distance to the median.

        linearTranspose = np.transpose(linearPos,axes=[0,2,1])
        linearTranspose = linearTranspose.reshape([linearTranspose.shape[0]*linearTranspose.shape[1],linearTranspose.shape[2]])
        histPosPred = np.stack([np.histogram(np.abs(linearTranspose[id,:]-np.median(linearTranspose[id,:])),bins=np.arange(0,stop=1,step=0.01))[0] for id in range(linearTranspose.shape[0])])
        fig,ax = plt.subplots()
        ax.matshow(np.transpose(histPosPred))
        ax.set_aspect(9880/99)
        fig.show()

        fig,ax = plt.subplots()
        cm = plt.get_cmap("turbo")
        colors = cm(np.abs(output_test[0]-trueLinearPos)/np.max(np.abs(output_test[0]-trueLinearPos)))
        for i in range(histPosPred.shape[0]):
            ax.plot(histPosPred[i],alpha=0.4,c=colors[i])
        fig.show()

        # we first sort by error:
        sortPerm = np.argsort(np.abs(output_test[0]-trueLinearPos))
        reorderedHist = histPosPred[sortPerm]

        fig,ax = plt.subplots()
        ax.imshow(reorderedHist)
        ax.set_aspect(reorderedHist.shape[1]/reorderedHist.shape[0])
        ax.set_xlabel("histogram of absolute distance to median")
        axy = ax.twiny()
        axy.plot(np.abs(output_test[0]-trueLinearPos)[sortPerm],range(sortPerm.shape[0]),c="red",alpha=0.5)
        axy.set_xlabel("absolute decoding linear error")
        ax.set_ylabel("time step - \n reordered by decoding error")
        # ax[1].set_aspect(reorderedHist.shape[1]/(np.abs(output_test[0]-trueLinearPos).max()))
        fig.show()

        AbsErrorToMedian = np.abs(linearTranspose - np.median(linearTranspose,axis=1)[:,None])
        meanAbsErrorToMedian = np.mean(AbsErrorToMedian,axis=1)
        fig,ax = plt.subplots()
        ax.scatter(meanAbsErrorToMedian,np.abs(output_test[0]-trueLinearPos)[sortPerm])
        fig.show()

        linearEnsembleError = np.abs(output_test[0]-trueLinearPos)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        clf = Ridge(alpha=1000)
        X_train,X_test,Y_train,Y_test = train_test_split(AbsErrorToMedian,linearEnsembleError,train_size=0.3)
        clf.fit(X_train,Y_train)
        self.LinearNetworkConfidence = clf
        predfromTrain2 = clf.predict(X_test)
        fig,ax = plt.subplots(2,1)
        ax[0].scatter(predfromTrain2,Y_test,alpha=0.5,s=1,c="black")
        ax[1].hist(predfromTrain2,bins=50,color="blue",alpha=0.5,density=True)
        # ax[1].hist(Y_test, bins=100, color="orange", alpha=0.5,density=True)
        ax[0].set_xlabel("prediction of linear error, \n regularized linear prediction \n from absolute error to median")
        ax[0].set_ylabel("true linear error")
        fig.tight_layout()
        fig.show()

        fig,ax = plt.subplots()
        predError = clf.predict(AbsErrorToMedian)
        ax.plot(times, output_test[0], c="grey",alpha=0.2)
        ax.scatter(times,output_test[0],c=cm(predError/np.max(predError)),s=1)
        plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,np.max(predError)),cmap=cm),label="predicted error")
        ax.plot(times,trueLinearPos,c="black")
        ax.set_xlabel("time")
        ax.set_ylabel("linear position")
        fig.show()

        # let us filter by pred Error
        fig,ax = plt.subplots()
        ax.plot(times[np.where(predError<0.08)],output_test[0][predError<0.08],c="red")
        ax.plot(times[np.where(predError < 0.08)], trueLinearPos[predError < 0.08],c="black")
        fig.show()




        import matplotlib.patches as patches
        fig,ax = plt.subplots()
        for i in range(output_test[0].shape[0]):
            circle = patches.Circle(tuple(output_test[0][i,:]),radius = np.mean(output_test[1][i])/2,edgecolor="orange",fill=False,alpha=0.1,zorder=0)
            ax.add_patch(circle)
        ax.scatter(output_test[0][:, 0], output_test[0][:, 1], s=2, c="red")
        ax.scatter(truePos[:,0],truePos[:,1],s=2,c="black",alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect(1)
        fig.show()

        speeds = behavior_data['Speed'][dres]
        window_len = 10
        s = np.r_[speeds[window_len - 1:0:-1], speeds, speeds[-2:-window_len - 1:-1]]
        w = eval('np.' + "hamming" + '(window_len)')
        speeds = np.convolve(w / w.sum(), s[:,0], mode='valid')[(window_len // 2 - 1):-(window_len // 2)]
        fig,ax = plt.subplots()
        ax.scatter(speeds,output_test[1],s=2,alpha=0.5)
        ax.set_xlabel("speed")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        cm = plt.get_cmap("turbo")
        fig,ax = plt.subplots(2,1)
        ax[0].scatter(times,trueLinearPos,c=cm(speeds/np.max(speeds)),s=1)
        ax[1].scatter(times,speeds,s=1)
        ax[0].plot(times,output_test[3],c="orange")
        fig.show()

        fig,ax = plt.subplots()
        ax.scatter(speeds,np.mean(output_test[1],axis=1),s=2,alpha=0.5)
        ax.set_xlabel("speed")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        fig,ax = plt.subplots()
        ax.scatter(np.abs(output_test[0]-trueLinearPos),speeds,s=2,alpha=0.5)
        ax.set_xlabel("prediction error of linear variable")
        ax.set_ylabel("speeds")
        fig.show()

        fig,ax = plt.subplots()
        ax.scatter(output_test[3],output_test[1],s=1,c=cm(np.abs(output_test[0]-trueLinearPos)/np.max(np.abs(output_test[0]-trueLinearPos))))
        fig.show()
        fig,ax = plt.subplots()
        ax.scatter( np.abs(output_test[0]-trueLinearPos),output_test[3],s=2,alpha=0.5,c="green")
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()
        fig,ax = plt.subplots()
        ax.scatter( np.sqrt(np.sum(np.square(output_test_no_dropout[0]-truePos),axis=1)),output_test[1],s=2,alpha=0.5)
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()


        predPos =output_test[0]
        fig,ax = plt.subplots()
        ax.scatter( np.sqrt(np.sum(np.square(predPos-truePos),axis=1))  ,np.mean(output_test[1],axis=1),s=2,alpha=0.5)
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        predPos_dropoutfree =output_test_no_dropout[0]
        fig,ax = plt.subplots()
        ax.scatter(np.sqrt(np.sum(np.square(predPos - truePos), axis=1)), np.mean(output_test[1], axis=1), s=2,
                   alpha=0.5, c="red",label="dropout prediction")
        ax.scatter( np.sqrt(np.sum(np.square(predPos_dropoutfree-truePos),axis=1))  ,np.mean(output_test[1],axis=1),s=2,alpha=0.5,c="violet",label="no dropout prediction")
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.legend()
        fig.show()


