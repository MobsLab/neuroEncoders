import numpy as np
import tensorflow as tf

import nnUtils
import waveFormConvNet
import transformerNetwork
from outputLanguageModel import OutputLanguageModel
import os

# Pierre 14/02/01:
# Reorganization of the code:
    # One class for the network
    # One function for the training
# We save the model every epoch during the training

# We generate a model with the functional Model interface in tensorflow
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000.0):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(tf.cast(step,dtype=tf.float32))
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(tf.cast(self.d_model,dtype=tf.float32)) * tf.math.minimum(arg1, arg2)



class WaveformTranslator():

    def __init__(self, projectPath, params, device_name="/device:gpu:0"):
        super(WaveformTranslator, self).__init__()
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
            # of each spike of a given group in the window. A group merges different recording site.

        #Loss obtained during training
        self.trainLosses = []

        # TODO: check initialization of the networks
        self.iteratorLengthInput = tf.keras.layers.Input(shape=(),name="length")
        if self.params.usingMixedPrecision:
            self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(None,None),name="group"+str(group),dtype=tf.float16) for group in range(self.params.nGroups)]
        else:
            self.inputsToSpikeNets = [tf.keras.layers.Input(shape=(None,None),name="group"+str(group)) for group in range(self.params.nGroups)]

        self.inputGroups = tf.keras.layers.Input(shape=(),name="groups")

        self.indices = [tf.keras.layers.Input(shape=(),name="indices"+str(group),dtype=tf.int32) for group in range(self.params.nGroups)]
        if self.params.usingMixedPrecision:
            self.zeroForGather = tf.zeros([1,self.params.nFeatures],dtype=tf.float16)
        else:
            self.zeroForGather = tf.zeros([1, self.params.nFeatures])

        # Declare spike nets for the different groups:
        self.spikeNets = [waveFormConvNet.waveFormConvNet(nChannels=self.params.nChannels[group],
                                           nFeatures=self.params.nFeatures) for group in range(self.params.nGroups)]

        # The convnet outputs spans over large real values
        # We normalize them  for each 36 ms window bin
        self.convNetNormalization = tf.keras.layers.LayerNormalization(axis=-1,center=False,scale=False, epsilon=1e-6)

        # Transformer Network:
        self.transformerEncoder = transformerNetwork.Encoder(self.params.num_layers_encoder_transformer,
                                                      self.params.d_model,self.params.num_heads_transformer,
                                                      self.params.dff,self.params.max_nb_spike_in_window,
                                                      self.params.transformerDropout)
        self.transformerDecoder = transformerNetwork.Decoder(self.params.num_layers_decoder_transformer,
                                                             self.params.d_model,self.params.num_heads_transformer,
                                                             self.params.dff,self.params.max_nb_spike_in_window,
                                                             self.params.transformerDropout)



        # In our implementation we use an output dictionnary of size d_model
        # d_model corresponds to the number of place cells.
        # Remarkably, we only use the positive values for the embedding
        # the negative number are therefore left to encode the "BOS" elements.
        # For example we set all cells to -1. for the "BOS" element.
        # The "EOS" element is accessed to by a last layer which expands into a dimension d_model+1 the output.
        # But here it is always given as label, and therefore we do not need to define its embedding.
        # Note:
        # The use of "EOS" or "BOS" elements allow us to predict sequence of decoded position of varying
        # length.

        self.lastLayer = tf.keras.layers.Dense(self.params.d_model+1,dtype=tf.float32)
        self.lastLayerSparseCat = tf.keras.layers.Dense(self.params.placeCellVocSize,dtype=tf.float32)

        self.outputEmbeding = OutputLanguageModel(self.params.d_model,self.params.dim_output,batchsize=self.params.batch_size)
        self.placeCellVocEmbeding = OutputLanguageModel(self.params.placeCellVocSize, self.params.dim_output,batchsize=self.params.batch_size)

        # trueFeature will correspond to the pop vector in the place cell space + the "BOS" keyword pop vector
        # at the beginning
        # it is what is fed to the decoder
        self.trueFeature = tf.keras.layers.Input(shape=(None, self.params.d_model), name="pos")
        # here word refer to the name of "position", it can be obtained with the outputEmbeding
        # it is the target of the decoder, and is simply a series of integer (token id) from 1 to d_model (place cells id)
        # the 'EOS' token id is d_model +1.
        self.trueWord = tf.keras.layers.Input(shape=(None,1), name="wordIndex",dtype=tf.int64)
        # target_feature correspond to the population vector in the place cell space to which
        # we added an additional dimension, a dimension reserved for the 'EOS' token
        # Additionally, this token's pop vector is put at the beginning of target_feature.
        self.target_feature = tf.keras.layers.Input(shape=(None, self.params.d_model+1), name="target_feature")

        #For the loss we have two choice:
        # either the use of a sparseCategoricalCrossentropy,
        # which does not take into account the proximity of different place cells
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.loss_layer_sparseCatCrossEntropy = tf.keras.layers.Lambda(lambda xs : self.loss_object(xs[0],xs[1]))

        # or the use of cosine similarity, which will be slower to compute,
        # but takes into account the predicted proximity of place cells
        # The transformer originally used sparse categorical crossentropy
        # but here I am not sure it can work without telling the network it is okay
        # to predict close to the current place cells.
        self.loss_cosinesim = tf.keras.losses.CosineSimilarity(axis=-1,reduction='none')
        self.loss_layer_CosineSim = tf.keras.layers.Lambda(lambda xs: self.loss_cosinesim(xs[0], xs[1]))


        self.denseLoss1 = tf.keras.layers.Dense(self.params.d_model+1, activation=tf.nn.relu)
        self.denseLoss2 = tf.keras.layers.Dense(1, activation=self.params.lossActivation)


        outputs = self.get_Model()
        self.model = self.mybuild(outputs)

    def get_Model(self):
        # generate and compile the model, lr is the chosen learning rate
        # CNN plus dense on every group independently

        # Optimization: could we not perform this foor loop more efficiently?
        # Key: try to use a tf.map as well as a parallel access.

        allFeatures = [] # store the result of the computation for each group
        for group in range(self.params.nGroups):
            x = self.inputsToSpikeNets[group]
            # --> [NbKeptSpike,nbChannels,32] tensors
            x = self.spikeNets[group](x)
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

            filledFeatureTrain = tf.reshape(filledFeatureTrain, [self.params.batch_size, -1,self.params.nFeatures])
            # Reshaping the result of the spike net as batch_size:NbTotSpikeDetected:nFeatures
            # this allow to separate spikes from the same window or from the same batch.
            allFeatures.append(filledFeatureTrain)

        allFeatures = tf.tuple(tensors=allFeatures)  # synchronizes the computation of all features (like a join)
        # The concatenation is made over axis 2, which is the Feature axis
        # So we reserve columns to each output of the spiking networks...
        allFeatures = tf.concat(allFeatures, axis=2, name="concat1")

        #We would like to mask timesteps that were added for batching purpose:
        batchedInputGroups = tf.reshape(self.inputGroups,[self.params.batch_size,-1])
        encoderMask = tf.cast(tf.equal(batchedInputGroups,-1),dtype=tf.float32)
        # this mask can be used for the multi-head attention of the inputs in the  encoder
        # and in the key value passed from the encoder to the decoder

        # self.trueFeature shape: [Batchsize,target_seq_len,d_model]
        look_ahead_mask = transformerNetwork.create_look_ahead_mask(tf.shape(self.trueFeature)[1])

        #self.trueWord : [batchsize,target_seq_len,1]
        currenttrueWord = tf.squeeze(self.trueWord,axis=-1)

        # TODO: Add a way to padd output sequences...
        # for now we assume that the target sequence provide the right length
        # output sequences should be padded to -1 to be skipped (to implement in the input pipeline)
        MaskOutputSeq = tf.cast(tf.math.equal(currenttrueWord, -1), tf.float32)
        dec_target_padding_mask = tf.cast(MaskOutputSeq[:, tf.newaxis, tf.newaxis, :],dtype=tf.float32)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        #Mask to prevent the transformer to look ahead of its output answer.

        allFeatures = self.convNetNormalization(allFeatures)
        encoding = self.transformerEncoder(allFeatures,mask=encoderMask[:, tf.newaxis, tf.newaxis, :])
        decoding = self.transformerDecoder([self.trueFeature,encoding, encoderMask[:, tf.newaxis, tf.newaxis, :], combined_mask])

        # the decoding (batch_size,target_seq_len,d_model) is then projected toward
        # the space of words (place cells + "EOS"): (batch_size,target_seq_len,d_model+1)
        # a softmax could even be used here....
        decoding = self.lastLayer(decoding)

        # computes loss, a mask is used to manage padded sequence
        # Remember: sequences are padded to facilitate batching
        # TODO: weight the loss by the place field distance for the sparse categorical cross entropy loss
        loss_ = self.loss_layer_CosineSim([self.target_feature,decoding])

        # mask = tf.cast(tf.math.not_equal(currenttrueWord, -1), dtype=loss_.dtype)
        # loss_ *= mask
        manifoldLoss = tf.reduce_sum(loss_) #/ tf.reduce_sum(mask)


        # compute accuracies
        # accuracies = tf.equal(currenttrueWord, tf.argmax(decoding, axis=2))
        # mask = tf.math.logical_not(tf.math.equal(currenttrueWord, 0))
        # accuracies = tf.math.logical_and(mask, accuracies)
        # accuracies = tf.cast(accuracies, dtype=tf.float32)
        # mask = tf.cast(mask, dtype=tf.float32)
        # accuracies = tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
        # accuracies = manifoldLoss # TODO  problem with accuracies computations

        # We treat the loss network on the side and stop the gradient from propagating in the main
        # prediction network. Intuitively, if we do not do that, part of the gradient will
        # tend to make the decoder prediction as predictable as possible, therefore
        # the decoder might fall into a local minima which is to predict the same thing all the time.
        # Another solution is to use a much smaller learning rate for the decoder network, which is similar
        # to scaling it by a small constant.
        outputLoss = tf.squeeze(self.denseLoss2(self.denseLoss1(tf.stop_gradient(decoding))),axis=-1)

        idmanifoldloss = tf.identity(decoding[:,0:-1,:],name="lossOfManifold")

        decodingclass = self.lastLayerSparseCat(decoding)
        idsparseCatLoss = tf.identity(decodingclass[:,0:-1,:],name="sparseCrossEntropy")


        lossFromOutputLoss = tf.identity(tf.losses.mean_squared_error(outputLoss, manifoldLoss),name="lossOfLossPredictor")
        return  idmanifoldloss, idsparseCatLoss, encoding[:,0,:], allFeatures[:,0,:], currenttrueWord, manifoldLoss  , lossFromOutputLoss,

    def mybuild(self, outputs):
        model = tf.keras.Model(inputs=self.inputsToSpikeNets+self.indices+
                                      [self.iteratorLengthInput, self.trueFeature, self.trueWord, self.inputGroups,self.target_feature],
                               outputs=outputs) #,

        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.params.learningRates[0], beta_1=0.9, beta_2=0.98,
                                               epsilon=1e-9), # Initially compile with first lr.
            loss={
                #"tf_op_layer_lossOfManifold" : tf.keras.losses.cosine_similarity, #
                "tf_op_layer_sparseCrossEntropy" : lambda x,y : self.loss_object(x,y),
                "tf_op_layer_lossOfLossPredictor" : tf.keras.losses.mean_absolute_error,
            },
        )
        #loss_weight removed here to try to get better probability and loss of loss learning
        return model

    def tokenizeAndMask(self,vals, valsout):
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

        # the data could come as either an array of size batch_size (one position predicted)
        # or as an array of size [batch_size,max_len], or as batch_size*max_len
        word_val = self.outputEmbeding.get_word(
            tf.reshape(vals["pos"], [self.params.batch_size, -1, self.params.dim_output]))
        word_val = tf.reshape(word_val, [self.params.batch_size, -1, 1])
        # We create a end of sequence word for each sequence.
        # We add it to the end of each padded sequence, but I am not sure this is the right way to do
        # should it not be positioned at the end of the non-padded sequence instead?
        end_of_seq_word = tf.zeros([self.params.batch_size, 1, 1], dtype=tf.int64) + self.params.d_model
        vals.update({"wordIndex": word_val})
        # vals.update({"wordIndex": tf.concat([word_val, end_of_seq_word], axis=1)})

        # We use another objective for the sparse cross entropy loss:
        word_val = self.placeCellVocEmbeding.get_word(
            tf.reshape(vals["pos"], [self.params.batch_size, -1, self.params.dim_output]))
        word_val = tf.reshape(word_val, [self.params.batch_size, -1, 1])
        valsout.update({"tf_op_layer_sparseCrossEntropy": word_val})

        # convert the position into feature
        # We need to use d_model variable, therefore we encode the beginning vector as -1
        # we could also choose the 0 vector...
        feature_val = self.outputEmbeding(tf.reshape(vals["pos"], [self.params.batch_size, -1, self.params.dim_output]))
        begining_of_seq_feature = np.zeros([self.params.batch_size, 1, self.params.d_model])
        begining_of_seq_feature[:, :, :] = -1.0
        vals.update({"pos": tf.concat([begining_of_seq_feature, feature_val], axis=1)})

        # In the case of the similarity loss we decide to allocate a new dimension for the end of seq feature
        # this time it is used by the output
        end_of_seq_feature = np.zeros([self.params.batch_size, 1, self.params.d_model + 1])
        end_of_seq_feature[:, :, -1] = 1
        new_feature_val_axis = tf.zeros([self.params.batch_size, tf.shape(feature_val)[1], 1])
        feature_val = tf.concat([feature_val, new_feature_val_axis], axis=-1)
        vals.update({"target_feature": feature_val})

        valsout.update({"tf_op_layer_lossOfManifold": feature_val})
        # vals.update({"target_feature": tf.concat([feature_val, end_of_seq_feature], axis=1)})
        # valsout.update({"tf_op_layer_lossOfManifold": tf.concat([feature_val, end_of_seq_feature], axis=1)})

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
        dataset = dataset.map(lambda vals: (vals,{
                                                  "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}),num_parallel_calls=4)
        dataset = dataset.map(self.tokenizeAndMask,num_parallel_calls=4)

        # The callbacks called during the training:
        callbackLR = tf.keras.callbacks.LearningRateScheduler(CustomSchedule(self.params.d_model))
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.projectPath.folder+"results",'training.log'))
        checkpoint_path = os.path.join(self.projectPath.folder+"results","training_1/cp.ckpt")
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.projectPath.folder+"results","profiling"),
        #                                             profile_batch = '200,210')

        hist = self.model.fit(dataset,
                  epochs=self.params.nEpochs,
                  callbacks=[ callbackLR, csv_logger, cp_callback], # , tb_callback,callbackLR
                              ) #steps_per_epoch = int(self.params.nSteps / self.params.nEpochs)

        return np.transpose(np.stack([#hist.history["tf_op_layer_lossOfManifold_loss"],
                                      hist.history["tf_op_layer_sparseCrossEntropy_loss"],
                                      hist.history["tf_op_layer_lossOfLossPredictor_loss"]
                                      ])) #


    def test(self):

        self.model.load_weights(os.path.join(self.projectPath.folder+"results","training_1/cp.ckpt"))

        ### Loading and inferring
        print("INFERRING")
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec["train"])
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        #drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(lambda *vals: nnUtils.parseSerializedSequence(self.params, self.feat_desc, *vals,batched=True))
        dataset = dataset.map(lambda vals: (vals, {"tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size)}), num_parallel_calls=4)
        dataset = dataset.map(self.tokenizeAndMask, num_parallel_calls=4)


        datasetTargetFeature = dataset.map(lambda x, y: x["target_feature"])
        fullFeatureTrue = list(datasetTargetFeature.as_numpy_iterator())
        fullFeatureTrue = np.array(fullFeatureTrue)

        datasetTargetWord = dataset.map(lambda x, y: x["wordIndex"])
        targetWord  = list(datasetTargetWord.as_numpy_iterator())
        targetWord  = np.array(targetWord)

        datasetInputFeature = dataset.map(lambda x, y: x["pos"])
        inputFeature= list(datasetInputFeature.as_numpy_iterator())
        inputFeature = np.array(inputFeature)

        datasetTimes = dataset.map(lambda x, y: x["time"])
        times = list(datasetTimes.as_numpy_iterator())

        datasetGroups = dataset.map(lambda x, y: x["groups"])
        groups = list(datasetGroups.as_numpy_iterator())

        datasetTargetCellWord = dataset.map(lambda x, y: y["tf_op_layer_sparseCrossEntropy"])
        targetCellWord = list(datasetTargetCellWord.as_numpy_iterator())
        targetCellWord = np.array(targetCellWord)
        targetCellWord = np.reshape(targetCellWord,[targetCellWord.shape[0]*self.params.batch_size,1])

        output_test = self.model.predict(dataset)

        encoding_gen = output_test[2]

        # outLoss = np.expand_dims(output_test[2],axis=1)
        # outLoss2 = np.expand_dims(output_test[3], axis=1)
        #
        # times = np.reshape(times, [output_test[0].shape[0]])


        pred_pos = tf.argmax(output_test[0][:,0,:-1],axis=-1)
        featureTrue = tf.reshape(fullFeatureTrue,[output_test[0].shape[0], 1, -1])

        tf.keras.losses.cosine_similarity(featureTrue[:,0,:],output_test[0][:,0,:])

        similarityThroughout = tf.keras.losses.cosine_similarity(featureTrue[0,0, 0:400],featureTrue[:,0, 0:400])

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.plot(similarityThroughout)
        plt.show()
        fig,ax = plt.subplots(2,1)
        ax[1].imshow(output_test[0][1000:1200,0,0:400])
        ax[0].imshow(featureTrue[1000:1200,0, 0:400])
        plt.show()
        losses = tf.keras.losses.sparse_categorical_crossentropy(targetCellWord, output_test[1][:, 0, :],from_logits=True)
        fig,ax = plt.subplots(2,1)
        ax[0].plot(targetCellWord[:,0],c="red")
        ax[0].plot(tf.argmax(output_test[1],axis=-1)[:,0])
        ax[1].plot(losses)
        plt.show()



        similarityThroughout = tf.keras.losses.cosine_similarity(output_test[0][0, 0, 0:400], output_test[0][:, 0, 0:400])
        fig,ax = plt.subplots()
        ax.plot(similarityThroughout)
        plt.show()

        # fig,ax = plt.subplots(3,1)
        # ax[0].imshow(self.transformerEncoder.pos_encoding[0,:,:])
        # ax[1].imshow(inputFeature[:,0,0,:])
        # ax[2].imshow(inputFeature[:, 0, 0, :] + self.transformerEncoder.pos_encoding[0,0,:] )
        # plt.show()
        #
        # allFeatures = output_test[3]
        # fig,ax = plt.subplots(3,1)
        # ax[0].imshow(encoding_gen[0:120, :])
        # ax[1].imshow(allFeatures[0:120,:])
        # ax[2].imshow(self.transformerEncoder.dropout(allFeatures[0:120,:]))
        # plt.show()
        #
        # # let's go deeply through what the encoder is doing
        # testInput = tf.stack([allFeatures[0:35,:] for _ in range(52)],axis=0)
        # btachedInputGroups = groups[0].reshape([52,-1])
        # mask  = tf.cast(tf.equal(btachedInputGroups,-1),dtype=tf.float32)
        # encoder_mask = mask[:,tf.newaxis,tf.newaxis,:]
        #
        # x = testInput
        # seq_len = tf.shape(x)[1]
        # x *= tf.math.sqrt(tf.cast(self.transformerEncoder.d_model, tf.float32))
        # x += self.transformerEncoder.pos_encoding[:, :seq_len, :]
        # x = self.transformerEncoder.dropout(x)
        # layernormConvNet = tf.keras.layers.LayerNormalization(axis=[1,2],epsilon=1e-6)
        # x = layernormConvNet(x)
        # fig,ax = plt.subplots(6,1)
        # # ax[0].imshow(x[0,:,:])
        # ax[0].imshow((self.transformerEncoder.enc_layers[0]).mha.wq(x)[0,:,:])
        # ax[1].imshow((self.transformerEncoder.enc_layers[0]).mha.wk(x)[0,:,:])
        # y,att_weight0 = (self.transformerEncoder.enc_layers[0]).mha(x,x,x,encoder_mask)
        # ax[1].imshow(att_weight0[0,0, :, :])
        # ax[2].imshow(y[0, :, :])
        # y = self.transformerEncoder.enc_layers[0].dropout1(y)
        # y = self.transformerEncoder.enc_layers[0].layernorm1(y+x)
        # ybis = self.transformerEncoder.enc_layers[0].ffn(y)
        # y = self.transformerEncoder.enc_layers[0].dropout2(ybis)
        # y = self.transformerEncoder.enc_layers[0].layernorm2(y + ybis)
        # ax[3].imshow(y[0, :, :])
        # # y2, att_weight = (self.transformerEncoder.enc_layers[1]).mha(y, y, y, encoder_mask)
        # # y2 = self.transformerEncoder.enc_layers[1].dropout1(y2)
        # # y2 = self.transformerEncoder.enc_layers[1].layernorm1(y + y2)
        # # y3, att_weight = (self.transformerEncoder.enc_layers[2]).mha(y2, y2, y2, encoder_mask)
        # # y3 = self.transformerEncoder.enc_layers[2].dropout1(y3)
        # # y3 = self.transformerEncoder.enc_layers[2].layernorm1(y2 + y3)
        # # y4, att_weight = (self.transformerEncoder.enc_layers[3]).mha(y3, y3, y3, encoder_mask)
        # # y4 = self.transformerEncoder.enc_layers[3].dropout1(y4)
        # # y4 = self.transformerEncoder.enc_layers[3].layernorm1(y4 + y3)
        # # ax[4].imshow(att_weight[0,0,:,:])
        # # ax[5].imshow(y4[0, :, :])
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # ax.imshow(att_weight0[0,0,:,:])
        # plt.show()
        #
        # fig,ax = plt.subplots()
        # opp = tf.matmul(testInput[0,:,:], testInput[0,:,:], transpose_b=True)
        # ax.imshow(tf.nn.softmax(opp,axis=-1))
        # plt.show()
        #


        return {"featurePred": output_test[0][:,0,:], "featureTrue": featureTrue,
                "times": times, "lossesFeaturePred" : outLoss,
                "lossFromOutputLoss" : outLoss2}
