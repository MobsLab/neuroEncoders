import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fullEncoder_v1 import nnUtils
import os
import pandas as pd
import tensorflow_probability as tfp
from tqdm import tqdm

from importData.ImportClusters import getBehavior
from importData.rawDataParser import  inEpochsMask

import pykeops

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
            "time": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64)}
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

            # Declare spike nets for the different groups:
            self.spikeNets = [nnUtils.spikeNet(nChannels=self.params.nChannels[group], device=self.device_name,
                                               nFeatures=self.params.nFeatures) for group in range(self.params.nGroups)]

            self.dropoutLayer = tf.keras.layers.Dropout(0.5) #todo: changed for test....
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

            self.predAbsoluteLinearErrorLayer = tf.keras.layers.Dense(1)

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
                # The index of spike detected then become similar to a time value...

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
        self.mazePoints_tensor = tf.convert_to_tensor(mazePoints[None,:],dtype=tf.float32)
        self.mazePoints = tf.keras.layers.Input(tensor=self.mazePoints_tensor, name="mazePoints")
        self.tsProj_tensor = tf.convert_to_tensor(tsProj[None,:],dtype=tf.float32)
        self.tsProj = tf.keras.layers.Input(tensor=self.tsProj_tensor, name="tsProj")

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

            myoutputPos = tf.reshape(myoutputPos,[1,batch_size,self.params.nb_eval_dropout,-1])
            outputLoss = tf.reshape(outputLoss, [1,batch_size,self.params.nb_eval_dropout, -1])
            myoutputPos = tf.transpose(myoutputPos,perm=[0,2,1,3])
            outputLoss = tf.transpose(outputLoss,perm=[0,2,1,3])
            # We had a reshaping mistake here, which was breaking the predictions by mixing adjacent windows...
            # [1,self.params.nb_eval_dropout,batch_size,-1] was bad...

        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.inputGroups,self.zeroForGather],
                               outputs=[myoutputPos,outputLoss])
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile()
        return model

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
            myoutputPos = tf.reshape(myoutputPos,[1,batch_size,self.params.nb_eval_dropout,-1])
            outputLoss = tf.reshape(outputLoss, [1,batch_size,self.params.nb_eval_dropout, -1])
            myoutputPos = tf.transpose(myoutputPos,perm=[0,2,1,3])
            outputLoss = tf.transpose(outputLoss,perm=[0,2,1,3])

            ## Next we find the linearization corresponding:
            bestLinearPos = tf.argmin(tf.reduce_sum(tf.square(myoutputPos - self.mazePoints[0,:,None,None,:]),axis=-1),axis=0)
            LinearPos = tf.gather(self.tsProj[0,:],bestLinearPos)
            med_linearPos = tfp.stats.percentile(LinearPos,50,axis=0)# compute the median!

            LinearPos = tf.transpose(LinearPos,perm=[1,0])
            outputLoss = tf.transpose(outputLoss[0,:,:,0],perm=[1,0])

            # TODO: we need to find a way to compute the entropy here.
            # from keras import backend as K
            # constants = np.arange(0,stop=1,step=0.01).astype(np.float32)
            # k_constants = K.variable(constants)
            # fixed_input = tf.keras.layers.Input(tensor=tf.constant(np.arange(0,stop=1,step=0.01).astype(np.float32)))
            #
            # hist = tfp.stats.histogram(LinearPos,edges=fixed_input ,axis=0)
            # proba = (hist/tf.math.reduce_sum(hist,axis=-1))
            # entropy = tf.math.reduce_mean(tf.math.xlogy(proba,proba))
            # # TODO: correct the entropy given the position predicted...
            #pred_absoluteLinearError = self.predAbsoluteLinearErrorLayer(tf.transpose(tf.abs(LinearPos-med_linearPos[None,:])))
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+[self.inputGroups,self.zeroForGather,self.mazePoints,self.tsProj],
                               outputs=[med_linearPos, LinearPos,outputLoss])
        tf.keras.utils.plot_model(
            model, to_file=modelName, show_shapes=True
        )
        model.compile()
        return model

    # used in the data pipepline
    def createIndices(self, vals,addLinearizationTensor=False):
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

            if addLinearizationTensor:
                vals.update({"mazePoints":self.mazePoints_tensor})
                vals.update({"tsProj":self.tsProj_tensor})

        if self.params.usingMixedPrecision:
            vals.update({"pos": tf.cast(vals["pos"], dtype=tf.float16)})
        return vals

    def train(self,onTheFlyCorrection=False,windowsizeMS=36):
        ## read behavior matrix:
        behavior_data = getBehavior(self.projectPath.folder,getfilterSpeed = True)
        speed_mask = behavior_data["Times"]["speedFilter"]

        if not os.path.isdir(os.path.join(self.projectPath.resultsPath, "resultInference",str(windowsizeMS))):
            os.makedirs(os.path.join(self.projectPath.resultsPath, "resultInference",str(windowsizeMS)))

        ### Training models
        ndataset = tf.data.TFRecordDataset(self.projectPath.tfdata+"_stride"+str(windowsizeMS)+".tfrec")
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

        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.resultsPath,"training"+str(windowsizeMS)+".log"))
        checkpoint_path = os.path.join(self.projectPath.resultsPath,"training_1_"+str(windowsizeMS)+"/cp.ckpt")
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.projectPath.folder,"profiling"),
        #                                             profile_batch = '100,110')


        speed_mask = behavior_data["Times"]["speedFilter"]
        DatasetOverfit = tf.data.TFRecordDataset(self.projectPath.tfdata+"_stride"+str(windowsizeMS)+".tfrec")
        DatasetOverfit = DatasetOverfit.map(lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
                              num_parallel_calls=tf.data.AUTOTUNE)
        epochMaskOverfit = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['testEpochs'])
        tot_mask = speed_mask * epochMaskOverfit
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
        DatasetOverfit = DatasetOverfit.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
        if onTheFlyCorrection:
            maxPos = np.max(behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            DatasetOverfit = DatasetOverfit.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
        DatasetOverfit = DatasetOverfit.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

        DatasetOverfit = DatasetOverfit.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        DatasetOverfit = DatasetOverfit.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals,batched=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        DatasetOverfit = DatasetOverfit.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        DatasetOverfit = DatasetOverfit.map(lambda vals: (vals,{"tf.identity": tf.zeros(self.params.batch_size),
                                                  "tf.identity_1": tf.zeros(self.params.batch_size)}),
                              num_parallel_calls=tf.data.AUTOTUNE)

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[csv_logger, cp_callback], # , tb_callback,cp_callback
                  validation_data=DatasetOverfit) #steps_per_epoch = int(self.params.nSteps / self.params.nEpochs)

        trainLosses = np.transpose(np.stack([hist.history["tf.identity_loss"], #tf_op_layer_lossOfManifold
                                      hist.history["tf.identity_1_loss"]])) #tf_op_layer_lossOfLossPredictor_loss
        valLosses = np.transpose(np.stack([hist.history["val_tf.identity_loss"], #tf_op_layer_lossOfManifold
                                      hist.history["val_tf.identity_1_loss"]]))


        df = pd.DataFrame(trainLosses)
        df.to_csv(os.path.join(self.projectPath.resultsPath, "resultInference",str(windowsizeMS), "lossTraining.csv"))
        fig,ax = plt.subplots(2,1)
        ax[0].plot(trainLosses[:,0])
        ax[0].set_title("position loss")
        ax[0].plot(valLosses[:,0],label="validation position loss",c="orange")
        ax[1].plot(trainLosses[:,1])
        ax[1].set_title("log loss prediction loss")
        ax[1].plot(valLosses[:,1],label="validation log loss prediction loss")
        fig.legend()
        plt.show()
        fig.savefig(os.path.join(self.projectPath.resultsPath,str(windowsizeMS), "lossTraining.png"))

        ## Stage 2: we perform a second training if there is some defined lossPredSetEpochs:
        if len(behavior_data["Times"]["lossPredSetEpochs"])>0:
            ndataset = tf.data.TFRecordDataset(self.projectPath.tfdata+"_stride"+str(windowsizeMS)+".tfrec")
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
            checkpoint_path = os.path.join(self.projectPath.resultsPath,"training_2_"+str(windowsizeMS)+"/cp.ckpt")
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            self.modelSecondTraining.load_weights(os.path.join(self.projectPath.resultsPath,"training_1_"+str(windowsizeMS)+"/cp.ckpt"))

            hist = self.modelSecondTraining.fit(dataset,
                      epochs=self.params.nEpochs,
                      callbacks=[callbackLR, csv_logger, cp_callback])
            trainLosses_2 = np.transpose(np.stack([hist.history["loss"]]))  #tf_op_layer_lossOfLossPredictor_loss

            df = pd.DataFrame(trainLosses_2)
            df.to_csv(os.path.join(self.projectPath.resultsPath, "resultInference",str(windowsizeMS),"lossTraining2.csv"))
            fig,ax = plt.subplots()
            ax.plot(trainLosses_2[:,0])
            plt.show()
            fig.savefig(os.path.join(self.projectPath.resultsPath,str(windowsizeMS), "lossTraining2.png"))

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
            epochMask = inEpochsMask(behavior_data['Position_time'][:, 0], behavior_data['Times']['testEpochs'][0:4])
        tot_mask = speed_mask * epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                                                tf.constant(tot_mask, dtype=tf.float64)), default_value=0)
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))

        if onTheFlyCorrection:
            maxPos = np.max(behavior_data["Positions"][np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))])
            dataset = dataset.map(nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos))
        dataset = dataset.filter(lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"]))))

        # d = dataset.take(1)
        # d = d.map(lambda val: (val["group0"]))
        # waveform = list(d.as_numpy_iterator())

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
        df = pd.DataFrame(linearPred)
        df.to_csv(os.path.join(self.projectPath.resultsPath, saveFolder, "linearPred.csv"))
        df = pd.DataFrame(linearTrue)
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

        self.onlineDecodingModel = self.get_model_for_uncertainty_estimate_insleep(batch=batch,batch_size=batch_size)

        #Remark: other solution: at training time, build and save the onlineDecodingModel
        # but now we can save these weights, so that they are used in c++....

        print("decoding sleep epochs")
        outputDic = {}
        for idsleep,sleepName in enumerate(behavior_data["Times"]["sleepNames"]):
            timeSleepStart = behavior_data["Times"]["sleepEpochs"][2*idsleep]
            timeSleepStop = behavior_data["Times"]["sleepEpochs"][2*idsleep+1]

            if not os.path.exists(os.path.join(self.projectPath.resultsPath,sleepName+"_all_loss_Preds.csv")):
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
                dataset = dataset.map(lambda x:self.createIndices(x,True), num_parallel_calls=tf.data.AUTOTUNE)
                dataset.cache()
                dataset.prefetch(tf.data.AUTOTUNE)
                output_test = self.onlineDecodingModel.predict(dataset,verbose=1)

                dtime = dataset.map(lambda vals: vals["time"])
                timePred = list(dtime.as_numpy_iterator())
                timePreds = np.ravel(timePred)

                df = pd.DataFrame(output_test[1])
                df.to_csv(os.path.join(self.projectPath.resultsPath,sleepName+"_allPreds.csv"))
                df = pd.DataFrame(timePreds)
                df.to_csv(os.path.join(self.projectPath.resultsPath,sleepName+"_timePreds.csv"))
                df = pd.DataFrame(output_test[2])
                df.to_csv(os.path.join(self.projectPath.resultsPath,sleepName+"_all_loss_Preds.csv"))
                #
                # outputDic[sleepName] = [output_test[0],output_test[1][:,0],timePreds] #median linear pos, entropy, time Preds
                #
                # df = pd.DataFrame(np.array([output_test[0],output_test[1][:,0],timePreds]))
                # df.to_csv(os.path.join(self.projectPath.resultsPath,sleepName))

            startTimes = pd.read_csv(os.path.join(self.projectPath.resultsPath,"dataset","alignment","sleep","startTimeWindow.csv"))

            # todo: save in CSV format for each sleep epochs...
            # and run another analysis to stop loosing time!

            from importData.rawDataParser import get_params
            _, samplingRate, _ = get_params(self.projectPath.xml)

            if not os.path.exists(os.path.join(self.projectPath.resultsPath,sleepName + "_proba_bayes1.csv")):
                from SimpleBayes import decodebayes
                trainerBayes = decodebayes.Trainer(self.projectPath)
                trainerBayes.bandwidth = 0.05
                from importData import ImportClusters
                cluster_data = ImportClusters.load_spike_sorting(self.projectPath)
                bayesMatrices = trainerBayes.train(behavior_data, cluster_data)
                ##Let us run Bayesian decoding in sleep
                linearpos_bayes_varying_window = []
                proba_bayes_varying_window = []
                for window_size in [1, 3, 7, 14]:
                    outputsBayes = trainerBayes.test_as_NN(
                        np.array(startTimes.values[:,1:],dtype=np.float32).astype(dtype=np.float64)/samplingRate,
                        bayesMatrices, behavior_data, cluster_data, windowSize=window_size * 0.036, masking_factor=1000,
                        useTrain=False,sleepEpochs=behavior_data["Times"]["sleepEpochs"][0:2])  # using window size of 36 ms!
                    pos = outputsBayes["inferring"][:, 0:2]
                    _, linearBayesPos = linearizationFunction(pos)
                    linearpos_bayes_varying_window += [linearBayesPos]
                    proba = outputsBayes["inferring"][:, 2]
                    proba_bayes_varying_window += [proba]
                    for id,window_size in enumerate([1,3,7,14]):
                        df = pd.DataFrame(proba_bayes_varying_window[id])
                        df.to_csv(os.path.join(self.projectPath.resultsPath,sleepName + "_proba_bayes"+str(window_size)+".csv"))
                        df = pd.DataFrame(linearpos_bayes_varying_window[id])
                        df.to_csv(os.path.join(self.projectPath.resultsPath,sleepName + "_linear_bayes"+str(window_size)+".csv"))


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

