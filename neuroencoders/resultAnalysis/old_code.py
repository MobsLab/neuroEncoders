def fit_uncertainty_estimate(
    self,
    linearizationFunction,
    batch=False,
    forceFirstTrainingWeight=False,
    useSpeedFilter=False,
    useTrain=False,
    onTheFlyCorrection=True,
):
    # spike sorted: spike times , with neurons index sorted by linear estimated max position of place field
    # in detail: [spike,one hot neurons index],[spiketime,

    # todo: use the validation set here by default.
    behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)
    if (
        len(behavior_data["Times"]["lossPredSetEpochs"]) > 0
        and not forceFirstTrainingWeight
    ):
        self.model.load_weights(
            os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt")
        )
    else:
        self.model.load_weights(
            os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt")
        )

    # Build the online decoding model with the layer already initialized:
    self.uncertainty_estimate_model = self.get_model_for_uncertainty_estimate(
        batch=batch
    )

    speed_mask = behavior_data["Times"]["speedFilter"]
    if not useSpeedFilter:
        speed_mask = np.zeros_like(speed_mask) + 1
    dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
    dataset = dataset.map(
        lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if useTrain:
        epochMask = inEpochsMask(
            behavior_data["Position_time"][:, 0], behavior_data["Times"]["trainEpochs"]
        )
    else:
        epochMask = inEpochsMask(
            behavior_data["Position_time"][:, 0], behavior_data["Times"]["testEpochs"]
        )
    tot_mask = speed_mask * epochMask
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
            tf.constant(tot_mask, dtype=tf.float64),
        ),
        default_value=0,
    )
    dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))

    if onTheFlyCorrection:
        maxPos = np.max(
            behavior_data["Positions"][
                np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))
            ]
        )
        dataset = dataset.map(
            nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos)
        )
    dataset = dataset.filter(
        lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
    )

    dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
    # drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
    dataset = dataset.map(
        lambda *vals: nnUtils.parseSerializedSequence(self.params, *vals, batched=True),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda vals: (
            vals,
            {
                "tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size),
            },
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if useTrain and not os.path.exists(
        os.path.join(
            self.projectPath.resultsPath,
            "uncertainty_network_fit",
            "networkPosPred.csv",
        )
    ):
        dtime = dataset.map(lambda vals, valsout: vals["time"])
        timePred = list(dtime.as_numpy_iterator())
        timePreds = np.ravel(timePred)
        output_test = self.uncertainty_estimate_model.predict(dataset, verbose=1)

        euclidData = np.reshape(output_test[0], [np.prod(output_test[0].shape[0:3]), 2])
        # save the result of the uncertainty prediction:
        if not os.path.exists(
            os.path.join(self.projectPath.resultsPath, "uncertainty_network_fit")
        ):
            os.makedirs(
                os.path.join(self.projectPath.resultsPath, "uncertainty_network_fit")
            )
        df = pd.DataFrame(euclidData)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_fit",
                "networkPosPred.csv",
            )
        )

        df = pd.DataFrame(timePreds)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath, "uncertainty_network_fit", "timePreds.csv"
            )
        )

        d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
        truePosFed = np.array(d0)
        truePosFed = truePosFed.reshape([truePosFed.shape[0] * truePosFed.shape[1], 2])
        df = pd.DataFrame(truePosFed)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_fit",
                "truePosFed.csv",
            )
        )
    elif (not useTrain) and not os.path.exists(
        os.path.join(
            self.projectPath.resultsPath,
            "uncertainty_network_test",
            "networkPosPred.csv",
        )
    ):
        dtime = dataset.map(lambda vals, valsout: vals["time"])
        timePred = list(dtime.as_numpy_iterator())
        timePreds = np.ravel(timePred)
        output_test = self.uncertainty_estimate_model.predict(dataset, verbose=1)

        euclidData = np.reshape(output_test[0], [np.prod(output_test[0].shape[0:3]), 2])
        # save the result of the uncertainty prediction:
        if not os.path.exists(
            os.path.join(self.projectPath.resultsPath, "uncertainty_network_test")
        ):
            os.makedirs(
                os.path.join(self.projectPath.resultsPath, "uncertainty_network_test")
            )
        df = pd.DataFrame(euclidData)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_test",
                "networkPosPred.csv",
            )
        )

        df = pd.DataFrame(timePreds)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_test",
                "timePreds.csv",
            )
        )

        d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
        truePosFed = np.array(d0)
        truePosFed = truePosFed.reshape([truePosFed.shape[0] * truePosFed.shape[1], 2])
        df = pd.DataFrame(truePosFed)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_test",
                "truePosFed.csv",
            )
        )
    if useTrain:
        euclidData = np.array(
            pd.read_csv(
                os.path.join(
                    self.projectPath.resultsPath,
                    "uncertainty_network_fit",
                    "networkPosPred.csv",
                )
            ).values[:, 1:],
            dtype=np.float32,
        )
        timePreds = np.array(
            pd.read_csv(
                os.path.join(
                    self.projectPath.resultsPath,
                    "uncertainty_network_fit",
                    "timePreds.csv",
                )
            ).values[:, 1],
            dtype=np.float32,
        )
        truePosFed = np.array(
            pd.read_csv(
                os.path.join(
                    self.projectPath.resultsPath,
                    "uncertainty_network_fit",
                    "truePosFed.csv",
                )
            ).values[:, 1:],
            dtype=np.float32,
        )
    else:
        euclidData = np.array(
            pd.read_csv(
                os.path.join(
                    self.projectPath.resultsPath,
                    "uncertainty_network_test",
                    "networkPosPred.csv",
                )
            ).values[:, 1:],
            dtype=np.float32,
        )
        timePreds = np.array(
            pd.read_csv(
                os.path.join(
                    self.projectPath.resultsPath,
                    "uncertainty_network_test",
                    "timePreds.csv",
                )
            ).values[:, 1],
            dtype=np.float32,
        )
        truePosFed = np.array(
            pd.read_csv(
                os.path.join(
                    self.projectPath.resultsPath,
                    "uncertainty_network_test",
                    "truePosFed.csv",
                )
            ).values[:, 1:],
            dtype=np.float32,
        )

    output_test = [
        np.reshape(
            euclidData,
            [
                -1,
                self.params.nb_eval_dropout,
                self.params.batch_size,
                self.params.dim_output,
            ],
        )
    ]

    projectedPos, linearPos = linearizationFunction(euclidData.astype(np.float64))

    linearPos = np.reshape(linearPos, output_test[0].shape[0:3])
    medianLinearPos = np.median(linearPos, axis=1)
    medianLinearPos = np.reshape(medianLinearPos, [np.prod(medianLinearPos.shape[0:2])])

    d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
    truePosFed = np.array(d0)
    truePosFed = truePosFed.reshape([truePosFed.shape[0] * truePosFed.shape[1], 2])
    trueProjPos, trueLinearPos = linearizationFunction(truePosFed)

    linearTranspose = np.transpose(linearPos, axes=[0, 2, 1])
    linearTranspose = linearTranspose.reshape(
        [linearTranspose.shape[0] * linearTranspose.shape[1], linearTranspose.shape[2]]
    )
    histPosPred = np.stack(
        [
            np.histogram(
                np.abs(linearTranspose[id, :] - np.median(linearTranspose[id, :])),
                bins=np.arange(0, stop=1, step=0.01),
                density=True,
            )[0]
            for id in range(linearTranspose.shape[0])
        ]
    )

    # let us get the window speedmask:
    dposIndex = dataset.map(lambda vals, valsout: vals["pos_index"])
    dposIndex = list(dposIndex.as_numpy_iterator())
    pos_index = np.ravel(np.array(dposIndex))
    speed_mask = behavior_data["Times"]["speedFilter"]
    windowmask_speed = speed_mask[pos_index]
    if useTrain:
        df = pd.DataFrame(windowmask_speed)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_fit",
                "windowmask_speed.csv",
            )
        )
    else:
        df = pd.DataFrame(windowmask_speed)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "uncertainty_network_test",
                "windowmask_speed.csv",
            )
        )

    histlinearPosPred = np.stack(
        [
            np.histogram(
                linearTranspose[id, :],
                bins=np.arange(0, stop=1, step=0.01),
                density=True,
            )[0]
            for id in range(linearTranspose.shape[0])
        ]
    )
    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(
        trueLinearPos[windowmask_speed],
        np.mean(linearTranspose, axis=1)[windowmask_speed],
        s=1,
        alpha=0.2,
    )
    ax[1].scatter(
        trueLinearPos[windowmask_speed],
        np.median(linearTranspose, axis=1)[windowmask_speed],
        s=1,
        alpha=0.2,
    )
    ax[2].scatter(
        trueLinearPos[windowmask_speed],
        np.argmax(histlinearPosPred, axis=1)[windowmask_speed],
        s=1,
        alpha=0.2,
    )
    fig.show()

    histlinearPosPred_density = (
        histlinearPosPred / (np.sum(histlinearPosPred, axis=1)[:, None])
    )

    fig, ax = plt.subplots()
    ax.matshow(
        np.transpose(histlinearPosPred_density),
        cmap=plt.get_cmap("Reds"),
        aspect="auto",
    )
    ax.plot(
        range(trueLinearPos.shape[0]),
        trueLinearPos * histlinearPosPred_density.shape[1],
        c="black",
        alpha=0.1,
    )
    testSet = inEpochsMask(timePreds, behavior_data["Times"]["testEpochs"])
    trainSet = inEpochsMask(timePreds, behavior_data["Times"]["trainEpochs"])
    ax.plot(
        range(trueLinearPos.shape[0]),
        testSet + histlinearPosPred_density.shape[1],
        c="black",
    )
    ax.plot(
        range(trueLinearPos.shape[0]),
        trainSet + histlinearPosPred_density.shape[1] + 2,
        c="blue",
    )
    ax.plot(
        range(trueLinearPos.shape[0]),
        windowmask_speed + histlinearPosPred_density.shape[1] + 4,
        c="red",
    )
    ax.plot(
        range(trueLinearPos.shape[0]),
        windowmask_speed * testSet + histlinearPosPred_density.shape[1] + 6,
        c="purple",
    )

    cmt = plt.get_cmap("tab20")
    cms = plt.get_cmap("Set3")
    for i in range(int(len(behavior_data["Times"]["testEpochs"]) / 2)):
        epoch = behavior_data["Times"]["testEpochs"][2 * i : 2 * i + 2]
        maskEpoch = inEpochsMask(timePreds, epoch)
        if i < 20:
            ax.plot(
                range(trueLinearPos.shape[0]),
                maskEpoch + histlinearPosPred_density.shape[1] + 8 + 2 * i,
                c=cmt(i),
            )
        else:
            ax.plot(
                range(trueLinearPos.shape[0]),
                maskEpoch + histlinearPosPred_density.shape[1] + 8 + 2 * i,
                c=cms(i),
            )
    # ax.set_aspect(histlinearPosPred.shape[0]/histlinearPosPred.shape[1])
    ax.plot(
        range(trueLinearPos.shape[0]),
        np.isnan(timePreds) + histlinearPosPred_density.shape[1] - 1,
        c="black",
    )
    fig.show()

    def xlogx(x):
        y = np.zeros_like(x)
        y[np.greater(x, 0)] = np.log(x[np.greater(x, 0)]) * (x[np.greater(x, 0)])
        return y

    # let us compute the absolute error over each test Epochs:
    absError_epochs = []
    absError_epochs_mean = []
    names = []
    entropies_epochs_mean = []
    entropies_epochs = []
    keptSession = behavior_data["Times"]["keptSession"]
    sessNames = behavior_data["Times"]["sessionNames"].copy()
    for idk, k in enumerate(keptSession.astype(np.bool)):
        if not k:
            sessNames.remove(behavior_data["Times"]["sessionNames"][idk])
    testEpochs = behavior_data["Times"]["testEpochs"].copy()
    for x in behavior_data["Times"]["sleepNames"]:
        for id2, x2 in enumerate(sessNames):
            if x == x2:
                sessNames.remove(x2)
                testEpochs[id2 * 2] = -1
                testEpochs[id2 * 2 + 1] = -1
    testEpochs = testEpochs[np.logical_not(np.equal(testEpochs, -1))]
    for i in range(int(len(testEpochs) / 2)):
        epoch = testEpochs[2 * i : 2 * i + 2]
        maskEpoch = inEpochsMask(timePreds, epoch)
        maskTot = maskEpoch * windowmask_speed
        if np.sum(maskTot) > 0:
            absError_epochs_mean += [
                np.mean(
                    np.abs(
                        trueLinearPos[maskTot]
                        - np.mean(linearTranspose[maskTot], axis=1)
                    )
                )
            ]
            absError_epochs += [
                np.abs(
                    trueLinearPos[maskTot] - np.mean(linearTranspose[maskTot], axis=1)
                )
            ]
            names += [sessNames[i]]
            entropies_epochs_mean += [
                np.mean(np.sum(-xlogx(histlinearPosPred_density[maskTot, :]), axis=1))
            ]
            entropies_epochs += [
                np.sum(-xlogx(histlinearPosPred_density[maskTot, :]), axis=1)
            ]
        else:
            print(sessNames[i])
    fig, ax = plt.subplots()
    ax.scatter(range(len(absError_epochs)), absError_epochs_mean)
    ax.plot(absError_epochs_mean)
    ax.set_ylabel("absolute decoding error")
    ax.set_xticks(range(len(absError_epochs)))
    ax.set_xticklabels(names)
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots()
    ax.scatter(range(len(absError_epochs)), entropies_epochs_mean)
    ax.plot(entropies_epochs_mean)
    ax.violinplot(entropies_epochs, positions=range(len(absError_epochs)))
    ax.set_ylabel("mean entropies")
    ax.set_xticks(range(len(absError_epochs)))
    ax.set_xticklabels(names)
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(7, 3, figsize=(5, 5))
    for i in range(7):
        for j in range(3):
            ax[i, j].scatter(
                entropies_epochs[3 * i + j],
                absError_epochs[3 * i + j],
                s=0.5,
                alpha=0.2,
            )
            ax[i, j].set_ylabel(sessNames[3 * i + j])
            # ax[i,j].set_xlabel("entropies")
            # ax[i, j].set_aspect(1)
    fig.tight_layout()
    fig.show()

    # Let us correct the predicted entropies by the distribution of predicted entropy given the predicted
    # position
    # TODO TODO

    #
    # cm = plt.get_cmap("Reds")
    # toDisplay = np.arange(135000, stop=138000)
    # fig,ax = plt.subplots()
    # linearvariable = np.arange(0, stop=1, step=0.01)
    # for i in range(histlinearPosPred.shape[1]):
    #     ax.scatter(timePreds[toDisplay], linearvariable[i] + np.zeros_like(timePreds[toDisplay]),
    #                c=cm(histlinearPosPred_density[toDisplay, i]), s=1)
    # ax.plot(timePreds[toDisplay],trueLinearPos[toDisplay],c="black",alpha=0.3)
    # fig.show()

    # we first sort by error:
    sortPerm = np.argsort(
        np.abs(medianLinearPos[windowmask_speed] - trueLinearPos[windowmask_speed])
    )
    reorderedHist = histPosPred[windowmask_speed][sortPerm]
    fig, ax = plt.subplots()
    ax.matshow(reorderedHist)
    ax.set_aspect(reorderedHist.shape[1] / reorderedHist.shape[0])
    ax.set_xlabel("histogram of absolute distance to median")
    axy = ax.twiny()
    axy.plot(
        np.abs(medianLinearPos[windowmask_speed] - trueLinearPos[windowmask_speed])[
            sortPerm
        ],
        range(sortPerm.shape[0]),
        c="red",
        alpha=0.5,
    )
    axy.set_xlabel("absolute decoding linear error")
    ax.set_ylabel("time step - \n reordered by decoding error")
    # ax[1].set_aspect(reorderedHist.shape[1]/(np.abs(output_test[0]-trueLinearPos).max()))
    fig.show()

    speeds = behavior_data["Speed"][pos_index]
    entropyPosPred = -np.sum(xlogx(histPosPred / 100), axis=-1)
    linearEnsembleError = np.abs(medianLinearPos - trueLinearPos)
    fig, ax = plt.subplots()
    cm = plt.get_cmap("Reds")
    ax.scatter(
        entropyPosPred[windowmask_speed],
        linearEnsembleError[windowmask_speed],
        s=2,
        c=cm(speeds[windowmask_speed] / np.max(speeds[windowmask_speed])),
        alpha=0.6,
    )
    ax.set_xlabel("entropy of the prediction")
    ax.set_ylabel("linear error")
    fig.show()

    fig, ax = plt.subplots()
    cm = plt.get_cmap("Reds")
    ax.scatter(
        entropyPosPred[windowmask_speed],
        linearEnsembleError[windowmask_speed],
        s=1 / (speeds[windowmask_speed] / np.max(speeds[windowmask_speed])),
        c="orange",
        alpha=0.6,
    )
    ax.set_xlabel("entropy of the prediction")
    ax.set_ylabel("linear error")
    fig.show()

    fig, ax = plt.subplots()
    cm = plt.get_cmap("Reds")
    ax.scatter(
        entropyPosPred[np.logical_not(windowmask_speed)],
        linearEnsembleError[np.logical_not(windowmask_speed)],
        s=10
        * speeds[np.logical_not(windowmask_speed)]
        / np.max(speeds[np.logical_not(windowmask_speed)]),
        c="orange",
        alpha=0.6,
    )
    ax.set_xlabel("entropy of the prediction")
    ax.set_ylabel("linear error")
    fig.show()

    fig, ax = plt.subplots()
    ax.scatter(
        linearEnsembleError[np.logical_not(windowmask_speed)],
        speeds[np.logical_not(windowmask_speed)],
        s=1,
        alpha=0.5,
    )
    ax.scatter(
        linearEnsembleError[windowmask_speed],
        speeds[windowmask_speed],
        s=1,
        c="red",
        alpha=0.5,
    )
    fig.show()

    fig, ax = plt.subplots()
    cm = plt.get_cmap("turbo")
    ax.scatter(
        timePreds, medianLinearPos, c=cm(entropyPosPred / np.max(entropyPosPred)), s=3
    )
    ax.plot(timePreds, trueLinearPos, c="grey")
    plt.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(0, np.max(entropyPosPred)), cmap=cm),
        label="entropy of predictions",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("linear position")
    fig.show()

    # build hist of entropy as a function of the predicted position
    posHistMapper = [
        np.where(
            np.less_equal(medianLinearPos, bin) * np.greater(medianLinearPos, bin - 0.1)
        )[0]
        for bin in np.arange(0.1, stop=1, step=0.1)
    ]
    entropy = [entropyPosPred[phm] for phm in posHistMapper]
    fig, ax = plt.subplots()
    ax.violinplot(entropy, positions=np.arange(0.1, stop=1, step=0.1), widths=0.1)
    ax.scatter(medianLinearPos, entropyPosPred, s=1, c="black", alpha=0.5)
    ax.set_xlabel("linear position")
    ax.set_ylabel("entropy")
    ax.set_title("all movement")
    fig.show()

    posHistMapper = [
        np.where(
            np.less_equal(medianLinearPos[windowmask_speed], bin)
            * np.greater(medianLinearPos[windowmask_speed], bin - 0.1)
        )[0]
        for bin in np.arange(0.1, stop=1, step=0.1)
    ]
    entropy = [entropyPosPred[windowmask_speed][phm] for phm in posHistMapper]
    fig, ax = plt.subplots()
    ax.violinplot(entropy, positions=np.arange(0.1, stop=1, step=0.1), widths=0.1)
    ax.scatter(
        medianLinearPos[windowmask_speed],
        entropyPosPred[windowmask_speed],
        s=1,
        c="black",
        alpha=0.5,
    )
    ax.set_xlabel("linear position")
    ax.set_ylabel("entropy")
    ax.set_title("speed filtered")
    fig.show()

    windowmask_speed_slow = np.logical_not(windowmask_speed)
    posHistMapper = [
        np.where(
            np.less_equal(medianLinearPos[windowmask_speed_slow], bin)
            * np.greater(medianLinearPos[windowmask_speed_slow], bin - 0.1)
        )[0]
        for bin in np.arange(0.1, stop=1, step=0.1)
    ]
    entropy = [entropyPosPred[windowmask_speed_slow][phm] for phm in posHistMapper]
    fig, ax = plt.subplots()
    ax.violinplot(entropy, positions=np.arange(0.1, stop=1, step=0.1), widths=0.1)
    ax.scatter(
        medianLinearPos[windowmask_speed_slow],
        entropyPosPred[windowmask_speed_slow],
        s=1,
        c="black",
        alpha=0.5,
    )
    ax.set_xlabel("linear position")
    ax.set_ylabel("entropy")
    ax.set_title("speed filtered (keeping slow speed)")
    fig.show()

    AbsErrorToMedian = np.abs(
        linearTranspose[windowmask_speed]
        - np.median(linearTranspose[windowmask_speed], axis=1)[:, None]
    )
    linearEnsembleError = np.abs(
        medianLinearPos[windowmask_speed] - trueLinearPos[windowmask_speed]
    )

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    clf = Ridge(alpha=0.01)
    X_train, X_test, Y_train, Y_test = train_test_split(
        AbsErrorToMedian, linearEnsembleError, train_size=0.3
    )
    clf.fit(X_train, Y_train)
    self.LinearNetworkConfidence = clf
    # to set the weights of the Linear Error Layer we need to call it first...
    self.predAbsoluteLinearErrorLayer(tf.convert_to_tensor(X_train[0:2, :]))
    self.predAbsoluteLinearErrorLayer.set_weights(
        [
            self.LinearNetworkConfidence.coef_[:, None],
            np.array([self.LinearNetworkConfidence.intercept_]),
        ]
    )

    # test to check that the NN and the scikit learn give same result.
    predFromNN = self.predAbsoluteLinearErrorLayer(tf.convert_to_tensor(X_test[:, :]))
    predfromTrain2 = clf.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(predFromNN, predfromTrain2)
    fig.show()

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(predfromTrain2, Y_test, alpha=0.5, s=1, c="black")
    ax[1].hist(predfromTrain2, bins=50, color="blue", alpha=0.5, density=True)
    # ax[1].hist(Y_test, bins=100, color="orange", alpha=0.5,density=True)
    ax[0].set_xlabel(
        "prediction of linear error, \n regularized linear prediction \n from absolute error to median"
    )
    ax[0].set_ylabel("true linear error")
    fig.tight_layout()
    fig.show()

    wakeConfidence = clf.predict(AbsErrorToMedian)
    fig, ax = plt.subplots()
    ax.hist(wakeConfidence, bins=100)
    ax.set_title(
        "wake set predicted confidence \n (== infered absolute linear error from absolute error to median)"
    )
    fig.show()

    fig, ax = plt.subplots()
    ax.hist(medianLinearPos, bins=50)
    ax.set_title("wake set predicted linear pos")
    fig.show()

    import subprocess

    import tables

    if not os.path.exists(os.path.join(self.projectPath.folder, "nnSWR.mat")):
        subprocess.run(["./getRipple.sh", self.projectPath.folder])
    with tables.open_file(self.projectPath.folder + "nnSWR.mat", "a") as f:
        ripples = f.root.ripple[:, :].transpose()

        predConfidence = clf.predict(AbsErrorToMedian)
        cm = plt.get_cmap("turbo")
        fig, ax = plt.subplots()
        # ax.plot(timePreds,medianLinearPos,c="red",alpha=0.3)
        ax.plot(timePreds, trueLinearPos, c="grey", alpha=0.3)
        ax.scatter(
            timePreds,
            medianLinearPos,
            s=1,
            c=cm(predConfidence / np.max(predConfidence)),
        )
        ax.vlines(
            ripples[ripples[:, 2] <= np.max(timePreds), 2],
            ymin=0,
            ymax=1,
            color="grey",
            linewidths=1,
        )
        fig.show()

        ##
        # Step 2: after having trained a mapping from error histogram to linear decoding error on active wake
        # we study its effect over the full wake
        ##

        # no speed masking this time
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if useTrain:
            epochMask = inEpochsMask(
                behavior_data["Position_time"][:, 0],
                behavior_data["Times"]["trainEpochs"],
            )
        else:
            epochMask = inEpochsMask(
                behavior_data["Position_time"][:, 0],
                behavior_data["Times"]["testEpochs"],
            )
        tot_mask = epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                tf.constant(tot_mask, dtype=tf.float64),
            ),
            default_value=0,
        )
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))
        if onTheFlyCorrection:
            maxPos = np.max(
                behavior_data["Positions"][
                    np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))
                ]
            )
            dataset = dataset.map(
                nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos)
            )
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        # drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                    "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        output_test = self.uncertainty_estimate_model.predict(dataset, verbose=1)
        euclidData = np.reshape(output_test[0], [np.prod(output_test[0].shape[0:3]), 2])
        projectedPos, linearPos = linearizationFunction(euclidData.astype(np.float64))

        linearPos = np.reshape(linearPos, output_test[0].shape[0:3])
        medianLinearPos = np.median(linearPos, axis=1)
        medianLinearPos = np.reshape(
            medianLinearPos, [np.prod(medianLinearPos.shape[0:2])]
        )

        d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
        truePosFed = np.array(d0)
        truePosFed = truePosFed.reshape([truePosFed.shape[0] * truePosFed.shape[1], 2])
        trueProjPos, trueLinearPos = linearizationFunction(truePosFed)

        linearTranspose = np.transpose(linearPos, axes=[0, 2, 1])
        linearTranspose = linearTranspose.reshape(
            [
                linearTranspose.shape[0] * linearTranspose.shape[1],
                linearTranspose.shape[2],
            ]
        )
        histPosPred = np.stack(
            [
                np.histogram(
                    np.abs(linearTranspose[id, :] - np.median(linearTranspose[id, :])),
                    bins=np.arange(0, stop=1, step=0.05),
                )[0]
                for id in range(linearTranspose.shape[0])
            ]
        )

        # we first sort by error:
        sortPerm = np.argsort(np.abs(medianLinearPos - trueLinearPos))
        reorderedHist = histPosPred[sortPerm]
        fig, ax = plt.subplots()
        ax.matshow(reorderedHist)
        ax.set_aspect(reorderedHist.shape[1] / reorderedHist.shape[0])
        ax.set_xlabel("histogram of absolute distance to median")
        axy = ax.twiny()
        axy.plot(
            np.abs(medianLinearPos - trueLinearPos)[sortPerm],
            range(sortPerm.shape[0]),
            c="red",
            alpha=0.5,
        )
        axy.set_xlabel("absolute decoding linear error")
        ax.set_ylabel("time step - \n reordered by decoding error")
        # ax[1].set_aspect(reorderedHist.shape[1]/(np.abs(output_test[0]-trueLinearPos).max()))
        fig.show()

        AbsErrorToMedian = np.abs(
            linearTranspose - np.median(linearTranspose, axis=1)[:, None]
        )
        linearEnsembleError = np.abs(medianLinearPos - trueLinearPos)

        predConfidence = clf.predict(AbsErrorToMedian)

        dtime = dataset.map(lambda vals, valsout: vals["time"])
        timePred = list(dtime.as_numpy_iterator())
        timePreds = np.ravel(timePred)

        cm = plt.get_cmap("turbo")
        fig, ax = plt.subplots()
        # ax.plot(timePreds,medianLinearPos,c="red",alpha=0.3)
        ax.plot(timePreds, trueLinearPos, c="grey", alpha=0.3)
        ax.scatter(
            timePreds,
            medianLinearPos,
            s=1,
            c=cm(predConfidence / np.max(predConfidence)),
        )
        ax.vlines(
            ripples[ripples[:, 2] <= np.max(timePreds), 2],
            ymin=0,
            ymax=1,
            color="grey",
            linewidths=1,
        )
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(predConfidence, linearEnsembleError, s=1)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("absolute linear error")
        fig.show()

        import sklearn.decomposition

        pcaDecomp = sklearn.decomposition.PCA()
        pcaDecomp.fit(reorderedHist)
        svalues = pcaDecomp.singular_values_
        explainedVariances = pcaDecomp.explained_variance_ratio_
        transformedFeature = pcaDecomp.transform(reorderedHist)
        fig, ax = plt.subplots()
        ax.plot(svalues, c="red")
        ax.twinx().plot(explainedVariances)
        fig.show()

        fig, ax = plt.subplots()
        ax.imshow(transformedFeature[:, 0:6])
        ax.set_aspect(6 / transformedFeature.shape[0])
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(
            transformedFeature[:, 0],
            transformedFeature[:, 1],
            s=1,
            c=cm(linearEnsembleError / np.max(linearEnsembleError)),
        )
        fig.show()
        permLinearEnsembleError = linearEnsembleError[sortPerm]

        # let us save the histrogramm as well as the linearEnsembleError and predConfidence in csv files so that
        # we can analyze them in Julia
        if not os.path.exists(
            os.path.join(self.projectPath.resultsPath, "unsupervisedConfidence")
        ):
            os.makedirs(
                os.path.join(self.projectPath.resultsPath, "unsupervisedConfidence")
            )
        df = pd.DataFrame(histPosPred)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "unsupervisedConfidence",
                "histDistToMed.csv",
            )
        )
        df = pd.DataFrame(linearEnsembleError)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "unsupervisedConfidence",
                "linearEnsembleError.csv",
            )
        )
        df = pd.DataFrame(predConfidence)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "unsupervisedConfidence",
                "predConfidence.csv",
            )
        )

        df = pd.DataFrame(linearTranspose)
        df.to_csv(
            os.path.join(
                self.projectPath.resultsPath,
                "unsupervisedConfidence",
                "lineaPredictions.csv",
            )
        )

        # let us sort the linear transpose by their median:
        sortPerm = np.argsort(np.median(linearTranspose, axis=1))
        histLinearPosPred = np.stack(
            [
                np.histogram(
                    linearTranspose[id, :], bins=np.arange(0, stop=1, step=0.05)
                )[0]
                for id in range(linearTranspose.shape[0])
            ]
        )
        reorderedHist = histLinearPosPred[sortPerm]
        fig, ax = plt.subplots()
        ax.matshow(reorderedHist)
        ax.set_aspect(reorderedHist.shape[1] / reorderedHist.shape[0])
        axy = ax
        axy.plot(
            medianLinearPos[sortPerm] * reorderedHist.shape[1],
            range(medianLinearPos.shape[0]),
            c="orange",
            label="decoded position",
        )
        axy.scatter(
            trueLinearPos[sortPerm] * reorderedHist.shape[1],
            range(medianLinearPos.shape[0]),
            c="red",
            label="true position",
            alpha=0.6,
            s=1,
        )
        # axy.set_xlabel("decoded position",color="orange")
        ax.set_xlabel("linear bin")
        ax.set_ylabel("time step id (sorted by predicted position)")
        fig.legend()
        fig.show()

        histDiffPosPred = np.stack(
            [
                np.histogram(
                    (linearTranspose[id, :] - np.median(linearTranspose[id, :])),
                    bins=np.arange(-1, stop=1, step=0.05),
                )[0]
                for id in range(linearTranspose.shape[0])
            ]
        )
        reorderedHist = histDiffPosPred[sortPerm]
        fig, ax = plt.subplots()
        ax.matshow(reorderedHist)
        ax.set_aspect(reorderedHist.shape[1] / reorderedHist.shape[0])
        axy = ax
        axy.plot(
            medianLinearPos[sortPerm] * reorderedHist.shape[1],
            range(medianLinearPos.shape[0]),
            c="orange",
            label="decoded position",
        )
        # axy.set_xlabel("decoded position",color="orange")
        ax.set_xlabel("linear bin")
        ax.set_ylabel("time step id (sorted by predicted position)")
        fig.legend()
        fig.show()

        # filtering by error > 0.2:
        # reorganising by

        from sklearn.manifold import Isomap

        embedding = Isomap(n_components=2)
        X_transformed = embedding.fit_transform(histPosPred[0:20000, :])
        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            s=1,
            c=cm(linearEnsembleError[0:20000] / np.max(linearEnsembleError)),
        )
        ax[1].scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            s=1,
            c=cm(predConfidence[0:20000] / np.max(predConfidence)),
        )
        fig.show()

    def sleep_uncertainty_estimate(self, output_test, linearizationFunction):
        # output_test: euclid_data,lossPred (not used anymore),time steps
        euclidData = np.reshape(output_test[0], [np.prod(output_test[0].shape[0:3]), 2])
        projectedPos, linearPos = linearizationFunction(euclidData.astype(np.float64))
        linearPos = np.reshape(linearPos, output_test[0].shape[0:3])
        medianLinearPos = np.median(linearPos, axis=1)

        # next we estimate the error made, by using the Linear projection of the distance to the median:
        linearTranspose = np.transpose(linearPos, axes=[0, 2, 1])
        linearTranspose = linearTranspose.reshape(
            [
                linearTranspose.shape[0] * linearTranspose.shape[1],
                linearTranspose.shape[2],
            ]
        )
        AbsErrorToMedian = np.abs(
            linearTranspose - np.median(linearTranspose, axis=1)[:, None]
        )
        predictedConfidence = self.LinearNetworkConfidence.predict(AbsErrorToMedian)

        return medianLinearPos, predictedConfidence

    def study_sleep_uncertainty_estimate(self, output_test, linearizationFunction):
        # output_test: euclid_data,lossPred (not used anymore),time steps
        # euclidData = np.reshape(output_test[0],[np.prod(output_test[0].shape[0:3]),2])
        # projectedPos,linearPos = linearizationFunction(euclidData.astype(np.float64))
        # linearPos = np.reshape(linearPos,output_test[0].shape[0:3])
        # medianLinearPos = np.median(linearPos,axis=1)
        medianLinearPos, predictedConfidence, timePreds = output_test

        fig, ax = plt.subplots()
        [
            ax.scatter(
                output_test[-1][1:1000:1],
                np.ravel(linearPos[:, id, :])[1:1000:1],
                c="orange",
                s=1,
                alpha=0.2,
            )
            for id in range(linearPos.shape[1])
        ]
        ax.plot(output_test[-1][1:1000:1], np.ravel(medianLinearPos)[1:1000:1], c="red")
        ax.set_xlabel("time")
        ax.set_ylabel("decoded linear position")
        ax.set_title("beginning of sleep")
        fig.show()

        # # next we estimate the error made, by using the Linear projection of the distance to the median:
        # linearTranspose = np.transpose(linearPos,axes=[0,2,1])
        # linearTranspose = linearTranspose.reshape([linearTranspose.shape[0]*linearTranspose.shape[1],linearTranspose.shape[2]])
        # AbsErrorToMedian = np.abs(linearTranspose - np.median(linearTranspose,axis=1)[:,None])
        # predictedConfidence = self.LinearNetworkConfidence.predict(AbsErrorToMedian)

        cm = plt.get_cmap("turbo")
        fig, ax = plt.subplots()
        ax.plot(
            output_test[-1][1:1000:1],
            np.ravel(medianLinearPos)[1:1000:1],
            c="grey",
            alpha=0.3,
        )
        ax.scatter(
            output_test[-1][1:1000:1],
            np.ravel(medianLinearPos)[1:1000:1],
            c=cm(predictedConfidence[1:1000:1] / np.max(predictedConfidence)),
            s=3,
        )
        plt.colorbar(
            plt.cm.ScalarMappable(
                plt.Normalize(0, np.max(predictedConfidence)), cmap=cm
            ),
            label="predicted confidence",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("decoded linear position")
        fig.show()

        cm = plt.get_cmap("turbo")
        fig, ax = plt.subplots()
        ax.plot(
            output_test[-1][predictedConfidence < 0.1],
            np.ravel(medianLinearPos)[predictedConfidence < 0.1],
            c="grey",
            alpha=0.3,
        )
        ax.scatter(
            output_test[-1][predictedConfidence < 0.1],
            np.ravel(medianLinearPos)[predictedConfidence < 0.1],
            c=cm(
                predictedConfidence[predictedConfidence < 0.1]
                / np.max(predictedConfidence)
            ),
            s=3,
        )
        plt.colorbar(
            plt.cm.ScalarMappable(
                plt.Normalize(0, np.max(predictedConfidence)), cmap=cm
            ),
            label="predicted confidence",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("decoded linear position")
        fig.show()

        # let us look at the confidence distributions:
        fig, ax = plt.subplots()
        ax.hist(predictedConfidence, bins=100)
        ax.set_title("confidence in sleep")
        ax.set_xlabel("confidence")
        ax.set_ylabel("histogram")
        fig.show()
        # todo: compare sleep and wake confidences

        # let us look at the distribution of linear position jump
        posjump = np.ravel(medianLinearPos)[1:] - np.ravel(medianLinearPos)[:-1]
        fig, ax = plt.subplots()
        ax.hist(np.ravel(medianLinearPos), bins=50, color="red")
        ax.set_xlabel("linear position")
        fig.show()
        fig, ax = plt.subplots()
        ax.hist(posjump, bins=1000, color="red", alpha=0.5)
        ax.set_yscale("log")
        fig.show()
        fig, ax = plt.subplots()
        ax.hist(np.abs(posjump), bins=100, color="red", alpha=0.5)
        ax.set_yscale("log")
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(
            output_test[-1][:-1][predictedConfidence[:-1] < 0.08],
            posjump[predictedConfidence[:-1] < 0.08],
            s=1,
            alpha=0.4,
        )
        fig.show()

        # let us compute the transition probability matrix from one position to another:
        medianLinearPos = np.ravel(medianLinearPos)
        _, binMed = np.histogram(medianLinearPos, bins=10)
        filterForSize = lambda x: x[x < (len(medianLinearPos) - 1)]
        findHist = lambda x, j: np.sum(
            (medianLinearPos[x + 1] >= binMed[j])
            * (medianLinearPos[x + 1] < binMed[j + 1])
        )
        transMat = [
            [
                findHist(
                    filterForSize(
                        np.where(
                            (medianLinearPos >= binMed[i])
                            * (medianLinearPos < binMed[i + 1])
                        )[0]
                    ),
                    j,
                )
                for j in range(len(binMed) - 1)
            ]
            for i in range(len(binMed) - 1)
        ]
        transMat = np.array(transMat)
        fig, ax = plt.subplots()
        ax.matshow(transMat)
        for i in range(len(binMed) - 1):
            for j in range(len(binMed) - 1):
                text = ax.text(
                    j, i, transMat[i, j], ha="center", va="center", color="w"
                )
        ax.set_xticks(range(len(binMed[:-1])))
        ax.set_yticks(range(len(binMed[:-1])))
        ax.set_xticklabels(np.round(binMed[:-1], 2))
        ax.set_yticklabels(np.round(binMed[:-1], 2))
        fig.show()

        # Looking at data: is seem that a series of small jump is followed by a large jump.
        # let us therefore look at the jump transition matrix:
        absPosJump = np.abs(posjump)
        histAbsPosJump, binJump = np.histogram(absPosJump, bins=100)
        filterForSize = lambda x: x[x < (len(absPosJump) - 1)]
        findHist = lambda x, j: np.sum(
            (absPosJump[x + 1] >= binJump[j]) * (absPosJump[x + 1] < binJump[j + 1])
        )
        transMatJump = [
            [
                findHist(
                    filterForSize(
                        np.where(
                            (absPosJump >= binJump[i]) * (absPosJump < binJump[i + 1])
                        )[0]
                    ),
                    j,
                )
                for j in range(len(binJump) - 1)
            ]
            for i in range(len(binJump) - 1)
        ]
        transMatJump = np.array(transMatJump)
        # so row corresponds to the state at t
        # so columns corresponds to the state at t+1
        # fig,ax = plt.subplots()
        # ax.scatter(output_test[-1][:-1],np.abs(posjump),s=1)
        # fig.show()

        fig, ax = plt.subplots()
        ax.imshow(
            transMatJump / histAbsPosJump[:, None]
        )  # effectively normalize each row!
        ax.set_ylabel("jump at t")
        ax.set_xlabel("jump at t+1")
        fig.show()
        # --> jumps are most of the time followed by small jumps....

        # we separate large and small jumps arbitrarily:
        pospeed = posjump / (output_test[-1][1:] - output_test[-1][:-1])
        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(output_test[-1][:-1], np.abs(pospeed), s=1, alpha=0.1)
        ax[1].hist(np.log(pospeed[pospeed != 0]), bins=np.arange(-10, 10, step=0.1))
        fig.show()

        # Continuity driven by predicted confidence:
        fig, ax = plt.subplots()
        ax.scatter(posjump, predictedConfidence[:-1], s=0.1, alpha=0.3)
        ax.set_xlabel("position jump between two time step")
        ax.set_ylabel("predicted confidence")
        fig.show()
        _, bins = np.histogram(predictedConfidence, bins=100)
        posjump_knowing_confidence = [
            np.abs(
                posjump[
                    (predictedConfidence[:-1] >= bins[i])
                    * (predictedConfidence[:-1] < bins[i + 1])
                ]
            )
            for i in range(len(bins) - 1)
        ]
        mposjump_knowing_confidence = [np.mean(p) for p in posjump_knowing_confidence]
        stdposjump_knowing_confidence = [np.std(p) for p in posjump_knowing_confidence]
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], mposjump_knowing_confidence)
        ax.fill_between(
            bins[:-1],
            mposjump_knowing_confidence,
            np.array(mposjump_knowing_confidence)
            + np.array(stdposjump_knowing_confidence),
        )
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("mean absolute jump")
        fig.show()

        print("ended sleep uncertainty estimate")

    def study_uncertainty_estimate(
        self,
        linearizationFunction,
        batch=False,
        forceFirstTrainingWeight=False,
        useSpeedFilter=False,
        useTrain=False,
        onTheFlyCorrection=True,
    ):
        behavior_data = getBehavior(self.projectPath.folder, getfilterSpeed=True)
        if (
            len(behavior_data["Times"]["lossPredSetEpochs"]) > 0
            and not forceFirstTrainingWeight
        ):
            self.model.load_weights(
                os.path.join(self.projectPath.resultsPath, "training_2/cp.ckpt")
            )
        else:
            self.model.load_weights(
                os.path.join(self.projectPath.resultsPath, "training_1/cp.ckpt")
            )

        # Build the online decoding model with the layer already initialized:
        self.uncertainty_estimate_model = self.get_model_for_uncertainty_estimate(
            batch=batch
        )

        speed_mask = behavior_data["Times"]["speedFilter"]
        if not useSpeedFilter:
            speed_mask = np.zeros_like(speed_mask) + 1
        dataset = tf.data.TFRecordDataset(self.projectPath.tfrec)
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSpike(self.feat_desc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if useTrain:
            epochMask = inEpochsMask(
                behavior_data["Position_time"][:, 0],
                behavior_data["Times"]["trainEpochs"],
            )
        else:
            epochMask = inEpochsMask(
                behavior_data["Position_time"][:, 0],
                behavior_data["Times"]["testEpochs"],
            )
        tot_mask = speed_mask * epochMask
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(np.arange(len(tot_mask)), dtype=tf.int64),
                tf.constant(tot_mask, dtype=tf.float64),
            ),
            default_value=0,
        )
        dataset = dataset.filter(lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0))

        if onTheFlyCorrection:
            maxPos = np.max(
                behavior_data["Positions"][
                    np.logical_not(np.isnan(np.sum(behavior_data["Positions"], axis=1)))
                ]
            )
            dataset = dataset.map(
                nnUtils.onthefly_feature_correction(behavior_data["Positions"] / maxPos)
            )
        dataset = dataset.filter(
            lambda x: tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(x["pos"])))
        )

        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        # drop_remainder allows us to remove the last batch if it does not contain enough elements to form a batch.
        dataset = dataset.map(
            lambda *vals: nnUtils.parseSerializedSequence(
                self.params, *vals, batched=True
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.createIndices, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda vals: (
                vals,
                {
                    "tf_op_layer_lossOfManifold": tf.zeros(self.params.batch_size),
                    "tf_op_layer_lossOfLossPredictor": tf.zeros(self.params.batch_size),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        output_test = self.uncertainty_estimate_model.predict(dataset, verbose=1)
        euclidData = np.reshape(output_test[0], [np.prod(output_test[0].shape[0:3]), 2])
        projectedPos, linearPos = linearizationFunction(euclidData.astype(np.float64))
        projectedPos = np.reshape(projectedPos, output_test[0].shape)

        # we can also compute the distance to the projected Pos:
        predPos = np.reshape(euclidData, projectedPos.shape)
        vecToProjPos = predPos - projectedPos
        distToProjPos = np.sqrt(np.sum(np.square(vecToProjPos), axis=-1))
        middlePoint = np.array([0.5, 0.5])
        # the second variable sign can be obtained by the sign of the projection on the vector to this middle point from the linearized point
        # of the pred to linear vector
        signOfProj = np.sign(
            np.sum((predPos - middlePoint[None, None, None, :]) * vecToProjPos, axis=-1)
        )
        signDistToProjPos = distToProjPos * signOfProj

        linearPos = np.reshape(linearPos, output_test[0].shape[0:3])
        from scipy.stats import iqr

        output_test = [
            np.median(linearPos, axis=1),
            np.std(linearPos, axis=1),
            np.mean(linearPos, axis=1),
            iqr(linearPos, axis=1),
        ]
        output_test = [np.reshape(o, [np.prod(o.shape[0:2])]) for o in output_test]
        output_test_no_dropout = self.model.predict(dataset, verbose=1)

        print(len(output_test))
        speed_data = behavior_data["Speed"][np.where(tot_mask)]

        d1 = list(
            dataset.map(lambda vals, valsout: vals["pos_index"]).as_numpy_iterator()
        )
        dres = np.ravel(np.array(d1))
        speeds = behavior_data["Speed"][dres]
        truePos = behavior_data["Positions"][dres]

        d0 = list(dataset.map(lambda vals, valsout: vals["pos"]).as_numpy_iterator())
        truePosFed = np.array(d0)
        truePosFed = truePosFed.reshape([190 * 52, 2])
        trueProjPos, trueLinearPos = linearizationFunction(truePosFed)
        # compute
        trueVecToProjPos = truePosFed - trueProjPos
        trueDistToProjPos = np.sqrt(np.sum(np.square(trueVecToProjPos), axis=-1))
        trueSignOfProj = np.sign(
            np.sum((truePosFed - middlePoint[None, :]) * trueVecToProjPos, axis=-1)
        )
        trueSignDistToProjPos = trueDistToProjPos * trueSignOfProj

        times = behavior_data["Position_time"][dres]
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        [
            ax[0].scatter(
                times, np.ravel(signDistToProjPos[:, id, :]), c="orange", s=1, alpha=0.2
            )
            for id in range(signDistToProjPos.shape[1])
        ]
        ax[0].scatter(
            times, np.ravel(np.median(signDistToProjPos, axis=1)), c="red", s=1
        )
        ax[0].scatter(times, trueSignDistToProjPos, s=1, c="black")
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("signed distance to linearization line")
        ax[1].scatter(
            trueSignDistToProjPos,
            np.ravel(np.median(signDistToProjPos, axis=1)),
            c="black",
            s=1,
        )
        ax[1].set_xlabel("true signed distance \n to linearizartion line")
        ax[1].set_ylabel("predicted signed distance \n to linearizartion line")
        fig.show()

        g2 = euclidData.reshape(
            [linearPos.shape[0], linearPos.shape[1], linearPos.shape[2], 2]
        )
        g2med = np.reshape(np.median(g2, axis=1), [g2.shape[0] * g2.shape[2], 2])
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(times, g2med[:, 0], c="red", label="decoded by median of ensemble")
        ax[0].plot(times, truePosFed[:, 0], c="black", label="true pos")
        ax[1].plot(times, g2med[:, 1], c="red")
        ax[1].plot(times, truePosFed[:, 1], c="black")
        ax[0].set_ylabel("X")
        ax[1].set_ylabel("Y")
        ax[1].set_xlabel("time")
        fig.legend()
        fig.show()
        g2res = g2.reshape([190 * 52, 100, 2])
        fig, ax = plt.subplots()
        ax.scatter(truePosFed[:, 0], truePosFed[:, 1], c="black", s=1, alpha=0.2)
        ax.scatter(g2med[:, 0], g2med[:, 1], c="red", s=1)
        [
            ax.scatter(g2res[:, id, 0], g2res[:, id, 1], c="orange", s=1, alpha=0.01)
            for id in range(100)
        ]
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.show()

        # We could do a Maximum Likelihood estimate on the predicted variable:
        from SimpleBayes import butils

        bw = 0.2
        edges, _ = butils.kdenD(truePosFed, bw, nbins=[20, 20])

        def get_mle_estimate(X):
            _, p = butils.kdenD(X, bw, nbins=[20, 20], edges=edges)
            xedge = edges[0][:, 0]
            yedge = edges[-1][0, :]
            posMLE = np.unravel_index(np.argmax(p), p.shape)
            return [xedge[posMLE[0]], yedge[posMLE[1]]]

        mleDecodedPos = np.array(
            [get_mle_estimate(g2res[id, :, :]) for id in range(g2res.shape[0])]
        )
        fig, ax = plt.subplots()
        ax.plot(times, mleDecodedPos[:, 0], c="red")
        ax.plot(times, truePosFed[:, 0], c="black")
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
        fig, ax = plt.subplots()
        ax.plot(times, output_test[0], c="red", label="median prediction linear")
        # ax[0].plot(times, output_test[2], c="violet", label="mean prediction X")
        ax.plot(times, trueLinearPos, c="black", label="true linear")
        # ax[0].plot(times, output_test[3], c="green", label="iqr prediction X",alpha=0.5)
        fig.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("linear position")
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(times, output_test[0], c="red", label="median prediction X")
        ax.plot(times, output_test[2], c="violet", label="mean prediction X")
        ax.plot(times, trueLinearPos, c="black", label="true X")
        # ax.fill_between(times[:,0],output_test[0]+output_test[1],output_test[0]-output_test[1],color="orange",label="confidence")
        for i in range(100):
            ax.scatter(
                times[:, 0],
                np.reshape(
                    linearPos[:, i, :], [linearPos.shape[0] * linearPos.shape[2]]
                ),
                s=1,
                alpha=0.05,
                c="orange",
            )
        ax.set_xlabel("time")
        ax.set_ylabel("X")
        fig.legend()
        fig.show()

        # Question: given the distribution of predicted position
        # Are there some particular pattern?
        # to see that: we can look at the distributions distance to the median.

        linearTranspose = np.transpose(linearPos, axes=[0, 2, 1])
        linearTranspose = linearTranspose.reshape(
            [
                linearTranspose.shape[0] * linearTranspose.shape[1],
                linearTranspose.shape[2],
            ]
        )
        histPosPred = np.stack(
            [
                np.histogram(
                    np.abs(linearTranspose[id, :] - np.median(linearTranspose[id, :])),
                    bins=np.arange(0, stop=1, step=0.01),
                )[0]
                for id in range(linearTranspose.shape[0])
            ]
        )
        fig, ax = plt.subplots()
        ax.matshow(np.transpose(histPosPred))
        ax.set_aspect(9880 / 99)
        fig.show()

        fig, ax = plt.subplots()
        cm = plt.get_cmap("turbo")
        colors = cm(
            np.abs(output_test[0] - trueLinearPos)
            / np.max(np.abs(output_test[0] - trueLinearPos))
        )
        for i in range(histPosPred.shape[0]):
            ax.plot(histPosPred[i], alpha=0.4, c=colors[i])
        fig.show()

        # we first sort by error:
        sortPerm = np.argsort(np.abs(output_test[0] - trueLinearPos))
        reorderedHist = histPosPred[sortPerm]

        fig, ax = plt.subplots()
        ax.imshow(reorderedHist)
        ax.set_aspect(reorderedHist.shape[1] / reorderedHist.shape[0])
        ax.set_xlabel("histogram of absolute distance to median")
        axy = ax.twiny()
        axy.plot(
            np.abs(output_test[0] - trueLinearPos)[sortPerm],
            range(sortPerm.shape[0]),
            c="red",
            alpha=0.5,
        )
        axy.set_xlabel("absolute decoding linear error")
        ax.set_ylabel("time step - \n reordered by decoding error")
        # ax[1].set_aspect(reorderedHist.shape[1]/(np.abs(output_test[0]-trueLinearPos).max()))
        fig.show()

        AbsErrorToMedian = np.abs(
            linearTranspose - np.median(linearTranspose, axis=1)[:, None]
        )
        meanAbsErrorToMedian = np.mean(AbsErrorToMedian, axis=1)
        fig, ax = plt.subplots()
        ax.scatter(
            meanAbsErrorToMedian, np.abs(output_test[0] - trueLinearPos)[sortPerm]
        )
        fig.show()

        linearEnsembleError = np.abs(output_test[0] - trueLinearPos)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split

        clf = Ridge(alpha=1000)
        X_train, X_test, Y_train, Y_test = train_test_split(
            AbsErrorToMedian, linearEnsembleError, train_size=0.3
        )
        clf.fit(X_train, Y_train)
        self.LinearNetworkConfidence = clf
        predfromTrain2 = clf.predict(X_test)
        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(predfromTrain2, Y_test, alpha=0.5, s=1, c="black")
        ax[1].hist(predfromTrain2, bins=50, color="blue", alpha=0.5, density=True)
        # ax[1].hist(Y_test, bins=100, color="orange", alpha=0.5,density=True)
        ax[0].set_xlabel(
            "prediction of linear error, \n regularized linear prediction \n from absolute error to median"
        )
        ax[0].set_ylabel("true linear error")
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots()
        predError = clf.predict(AbsErrorToMedian)
        ax.plot(times, output_test[0], c="grey", alpha=0.2)
        ax.scatter(times, output_test[0], c=cm(predError / np.max(predError)), s=1)
        plt.colorbar(
            plt.cm.ScalarMappable(plt.Normalize(0, np.max(predError)), cmap=cm),
            label="predicted error",
        )
        ax.plot(times, trueLinearPos, c="black")
        ax.set_xlabel("time")
        ax.set_ylabel("linear position")
        fig.show()

        # let us filter by pred Error
        fig, ax = plt.subplots()
        ax.plot(
            times[np.where(predError < 0.08)], output_test[0][predError < 0.08], c="red"
        )
        ax.plot(
            times[np.where(predError < 0.08)],
            trueLinearPos[predError < 0.08],
            c="black",
        )
        fig.show()

        import matplotlib.patches as patches

        fig, ax = plt.subplots()
        for i in range(output_test[0].shape[0]):
            circle = patches.Circle(
                tuple(output_test[0][i, :]),
                radius=np.mean(output_test[1][i]) / 2,
                edgecolor="orange",
                fill=False,
                alpha=0.1,
                zorder=0,
            )
            ax.add_patch(circle)
        ax.scatter(output_test[0][:, 0], output_test[0][:, 1], s=2, c="red")
        ax.scatter(truePos[:, 0], truePos[:, 1], s=2, c="black", alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect(1)
        fig.show()

        speeds = behavior_data["Speed"][dres]
        window_len = 10
        s = np.r_[
            speeds[window_len - 1 : 0 : -1], speeds, speeds[-2 : -window_len - 1 : -1]
        ]
        w = eval("np." + "hamming" + "(window_len)")
        speeds = np.convolve(w / w.sum(), s[:, 0], mode="valid")[
            (window_len // 2 - 1) : -(window_len // 2)
        ]
        fig, ax = plt.subplots()
        ax.scatter(speeds, output_test[1], s=2, alpha=0.5)
        ax.set_xlabel("speed")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        cm = plt.get_cmap("turbo")
        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(times, trueLinearPos, c=cm(speeds / np.max(speeds)), s=1)
        ax[1].scatter(times, speeds, s=1)
        ax[0].plot(times, output_test[3], c="orange")
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(speeds, np.mean(output_test[1], axis=1), s=2, alpha=0.5)
        ax.set_xlabel("speed")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(np.abs(output_test[0] - trueLinearPos), speeds, s=2, alpha=0.5)
        ax.set_xlabel("prediction error of linear variable")
        ax.set_ylabel("speeds")
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(
            output_test[3],
            output_test[1],
            s=1,
            c=cm(
                np.abs(output_test[0] - trueLinearPos)
                / np.max(np.abs(output_test[0] - trueLinearPos))
            ),
        )
        fig.show()
        fig, ax = plt.subplots()
        ax.scatter(
            np.abs(output_test[0] - trueLinearPos),
            output_test[3],
            s=2,
            alpha=0.5,
            c="green",
        )
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()
        fig, ax = plt.subplots()
        ax.scatter(
            np.sqrt(np.sum(np.square(output_test_no_dropout[0] - truePos), axis=1)),
            output_test[1],
            s=2,
            alpha=0.5,
        )
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        predPos = output_test[0]
        fig, ax = plt.subplots()
        ax.scatter(
            np.sqrt(np.sum(np.square(predPos - truePos), axis=1)),
            np.mean(output_test[1], axis=1),
            s=2,
            alpha=0.5,
        )
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.show()

        predPos_dropoutfree = output_test_no_dropout[0]
        fig, ax = plt.subplots()
        ax.scatter(
            np.sqrt(np.sum(np.square(predPos - truePos), axis=1)),
            np.mean(output_test[1], axis=1),
            s=2,
            alpha=0.5,
            c="red",
            label="dropout prediction",
        )
        ax.scatter(
            np.sqrt(np.sum(np.square(predPos_dropoutfree - truePos), axis=1)),
            np.mean(output_test[1], axis=1),
            s=2,
            alpha=0.5,
            c="violet",
            label="no dropout prediction",
        )
        ax.set_xlabel("prediction error")
        ax.set_ylabel("100 droupout pass variance")
        fig.legend()
        fig.show()
