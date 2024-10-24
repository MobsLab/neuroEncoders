# Load libs
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables

mplt.use("TkAgg")
from tkinter import Button, Entry, Label, Tk, Toplevel

sys.path.append("./importData")
import os
import re

# from importData import epochs_management as ep
import sys

import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
import tables
from interval import interval

# Custom codes
from importData import epochs_management as ep

########### Management of epochs ############


# a few tool to help do difference of intervals as sets:
def obtainCloseComplementary(epochs, boundInterval):
    # we obtain the close complementary ( intervals share their bounds )
    # Note: to obtain the open complementary, one would need to add some dt to end and start of intervals....
    p1 = interval()
    for i in range(len(epochs) // 2):
        p1 = p1 | interval([epochs[2 * i], epochs[2 * i + 1]])
    assert isinstance(p1, interval)
    assert isinstance(boundInterval, interval)
    assert len(boundInterval) == 1
    compInterval = interval([boundInterval[0][0], p1[0][0]])
    for i in range(len(p1) - 1):
        compInterval = compInterval | interval([p1[i][1], p1[i + 1][0]])
    compInterval = compInterval | interval([p1[-1][1], boundInterval[0][1]])
    return compInterval


def intersect_with_session(epochs, keptSession, starts, stops):
    # we go through the different removed session epoch, and if a train epoch or a test epoch intersect with it we remove it
    # from the train and test epochs
    EpochInterval = interval()
    for i in range(len(epochs) // 2):
        EpochInterval = EpochInterval | interval([epochs[2 * i], epochs[2 * i + 1]])
    includeInterval = interval()
    for id, keptS in enumerate(keptSession):
        if keptS:
            includeInterval = includeInterval | interval([starts[id], stops[id]])
    EpochInterval = EpochInterval & includeInterval
    Epoch = np.ravel(np.array([[p[0], p[1]] for p in EpochInterval]))
    return Epoch


# Auxilliary function
def get_epochs(postime, SetData, keptSession, starts=np.empty(0), stops=np.empty(0)):
    # given the slider values, as well as the selected session, we extract the different sets
    # if starts and stops (of epochs) are present, it means we work with multi-recording

    pmin = postime[0]
    pmax = postime[-1]

    testEpochs = np.array(
        [
            postime[SetData["testSetId"]],
            postime[
                min(SetData["testSetId"] + SetData["sizeTestSet"], postime.shape[0] - 1)
            ],
        ]
    )

    if SetData["useLossPredTrainSet"]:
        lossPredSetEpochs = np.array(
            [
                postime[SetData["lossPredSetId"]],
                postime[
                    min(
                        SetData["lossPredSetId"] + SetData["sizeLossPredSet"],
                        postime.shape[0] - 1,
                    )
                ],
            ]
        )
        lossPredsetinterval = interval(lossPredSetEpochs)
        lossPredsetinterval = lossPredsetinterval & obtainCloseComplementary(
            testEpochs, interval([pmin, pmax])
        )
        lossPredSetEpochs = np.ravel(
            np.array([[p[0], p[1]] for p in lossPredsetinterval])
        )

        trainInterval = obtainCloseComplementary(
            testEpochs, interval([pmin, pmax])
        ) & obtainCloseComplementary(lossPredSetEpochs, interval([pmin, pmax]))
    else:
        trainInterval = obtainCloseComplementary(testEpochs, interval([pmin, pmax]))

    trainEpoch = np.ravel(np.array([[p[0], p[1]] for p in trainInterval]))

    if starts.size > 0:
        trainEpoch = intersect_with_session(trainEpoch, keptSession, starts, stops)
        testEpochs = intersect_with_session(testEpochs, keptSession, starts, stops)
        if SetData["useLossPredTrainSet"]:
            lossPredSetEpochs = intersect_with_session(
                lossPredSetEpochs, keptSession, starts, stops
            )
            return trainEpoch, testEpochs, lossPredSetEpochs
        else:
            return trainEpoch, testEpochs, None
    else:
        if SetData["useLossPredTrainSet"]:
            return trainEpoch, testEpochs, lossPredSetEpochs
        else:
            return trainEpoch, testEpochs, None


def inEpochs(t, epochs):
    # for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
    # test if t is among at least one of these epochs
    # Epochs are treated as closed interval [,]
    # returns the index where it is the case
    mask = np.sum(
        [
            (t >= epochs[2 * i]) * (t <= epochs[2 * i + 1])
            for i in range(len(epochs) // 2)
        ],
        axis=0,
    )
    return np.where(mask >= 1)


def inEpochsMask(t, epochs):
    # for a list of epochs, where each epochs starts is on even index [0,2,... and stops on odd index: [1,3,...
    # test if t is among at least one of these epochs
    # Epochs are treated as closed interval [,]
    # return the mask
    mask = np.sum(
        [
            (t >= epochs[2 * i]) * (t <= epochs[2 * i + 1])
            for i in range(len(epochs) // 2)
        ],
        axis=0,
    )
    return mask >= 1


############################


def get_params(pathToXml):
    listChannels = []
    try:
        tree = ET.parse(pathToXml)
    except:
        print("impossible to open xml file:", pathToXml)
        sys.exit(1)
    root = tree.getroot()
    for br1Elem in root:
        if br1Elem.tag != "spikeDetection":
            continue
        for br2Elem in br1Elem:
            if br2Elem.tag != "channelGroups":
                continue
            for br3Elem in br2Elem:
                if br3Elem.tag != "group":
                    continue
                group = []
                for br4Elem in br3Elem:
                    if br4Elem.tag != "channels":
                        continue
                    for br5Elem in br4Elem:
                        if br5Elem.tag != "channel":
                            continue
                        group.append(int(br5Elem.text))
                listChannels.append(group)
    for br1Elem in root:
        if br1Elem.tag != "acquisitionSystem":
            continue
        for br2Elem in br1Elem:
            if br2Elem.tag == "samplingRate":
                samplingRate = float(br2Elem.text)
            if br2Elem.tag == "nChannels":
                nChannels = int(br2Elem.text)

    return listChannels, samplingRate, nChannels


def findTime(positionTime, lastBestTime, time):
    for i in range(len(positionTime) - lastBestTime - 1):
        if np.abs(positionTime[lastBestTime + i] - time) < np.abs(
            positionTime[lastBestTime + i + 1] - time
        ):
            return lastBestTime + i
    return len(positionTime) - 1


############################


# Load the positions
def get_behavior(folder, bandwidth=None, getfilterSpeed=True, decode=False):
    if not os.path.exists(folder + "nnBehavior.mat"):
        raise ValueError("this file does not exist :" + folder + "nnBehavior.mat")
    # Extract basic behavior
    f = tables.open_file(folder + "nnBehavior.mat")
    positions = f.root.behavior.positions
    positions = np.swapaxes(positions[:, :], 1, 0)
    positionTime = f.root.behavior.position_time
    positionTime = np.swapaxes(positionTime[:, :], 1, 0)
    speed = f.root.behavior.speed
    speed = np.swapaxes(speed[:, :], 1, 0)
    if bandwidth == None:
        goodRecordingTimeStep = np.logical_not(np.isnan(np.sum(positions, axis=1)))
        bandwidth = (
            np.max(positions[goodRecordingTimeStep, :])
            - np.min(positions[goodRecordingTimeStep, :])
        ) / 15
    # Check for sleep sessions
    sleepPeriods = f.root.behavior.sleepPeriods[:]
    if np.sum(sleepPeriods) > 0:  # If sleepPeriods exist
        sleepNames = [
            "".join([chr(c) for c in l[0][:, 0]])
            for l in f.root.behavior.sessionSleepNames[:, 0]
        ]
        sessionNames = [
            "".join([chr(c) for c in l[0][:, 0]])
            for l in f.root.behavior.SessionNames[:, 0]
        ]
    # Train and test epochs
    if not decode:  # Not applicable for decode mode where all dataset is to be decoded
        if len(f.root.behavior.trainEpochs.shape) == 2:
            trainEpochs = np.concatenate(f.root.behavior.trainEpochs)
            testEpochs = np.concatenate(f.root.behavior.testEpochs)
            lossPredSetEpochs = np.concatenate(f.root.behavior.lossPredSetEpochs)
        elif len(f.root.behavior.trainEpochs.shape) == 1:
            trainEpochs = f.root.behavior.trainEpochs[:]
            testEpochs = f.root.behavior.testEpochs[:]
            lossPredSetEpochs = f.root.behavior.lossPredSetEpochs[:]
        else:
            raise Exception("bad train and test epochs format")
    # Get learning time and if needed speedFilter
    samplingWindowPosition = (positionTime[1:] - positionTime[0:-1])[:, 0]
    samplingWindowPosition[np.isnan(np.sum(positions[0:-1], axis=1))] = 0
    if not decode:
        if getfilterSpeed:
            speedFilter = f.root.behavior.speedMask[:]
        else:
            speedFilter = np.ones_like(f.root.behavior.speedMask[:])
        lEpochIndex = [
            [
                np.argmin(np.abs(positionTime - trainEpochs[2 * i + 1])),
                np.argmin(np.abs(positionTime - trainEpochs[2 * i])),
            ]
            for i in range(len(trainEpochs) // 2)
        ]
        learningTime = [
            np.sum(
                np.multiply(
                    speedFilter[lEpochIndex[i][1] : lEpochIndex[i][0]],
                    samplingWindowPosition[lEpochIndex[i][1] : lEpochIndex[i][0]],
                )
            )
            for i in range(len(lEpochIndex))
        ]
        learningTime = np.sum(learningTime)
    else:
        learningTime = np.sum(samplingWindowPosition)

    # Organize output
    behavior_data = {
        "Positions": positions,
        "positionTime": positionTime,
        "Speed": speed,
        "Bandwidth": bandwidth,
        "Times": {"learning": learningTime},
    }
    if not decode:
        behavior_data["Times"]["trainEpochs"] = trainEpochs
        behavior_data["Times"]["testEpochs"] = testEpochs
        behavior_data["Times"]["lossPredSetEpochs"] = lossPredSetEpochs
    if getfilterSpeed:
        behavior_data["Times"]["speedFilter"] = speedFilter
    if np.sum(sleepPeriods) > 0:
        behavior_data["Times"]["sleepEpochs"] = sleepPeriods
        behavior_data["Times"]["sleepNames"] = sleepNames
        behavior_data["Times"]["sessionNames"] = sessionNames
    f.close()

    return behavior_data


def speed_filter(folder, overWrite=True):
    ## A simple tool to set up a threshold on the speed value
    # The speed threshold is then implemented through a speed_mask:
    # a boolean array indicating for each index (i.e measured feature time step)
    # if it is above threshold or not.

    # Parameters
    window_len = 14  # changed following Dima's advice

    with tables.open_file(folder + "nnBehavior.mat", "a") as f:
        children = [c.name for c in f.list_nodes("/behavior")]
        if "speedMask" in children:
            print("speedMask already created")
            if overWrite:
                f.remove_node("/behavior", "speedMask")
            else:
                return

        # Prepare data
        positions = f.root.behavior.positions
        speed = f.root.behavior.speed
        positionTime = f.root.behavior.position_time
        sessionNames = [
            "".join([chr(c) for c in l[0][:, 0]])
            for l in f.root.behavior.SessionNames[:, 0]
        ]
        if sessionNames[0] != "Recording":
            IsMultiSessions = True
            sessionStart = f.root.behavior.SessionStart[:, :][:, 0]
            sessionStop = f.root.behavior.SessionStop[:, :][:, 0]
        else:
            IsMultiSessions = False

        positions = np.swapaxes(positions[:, :], 1, 0)
        speed = np.swapaxes(speed[:, :], 1, 0)
        posTime = np.swapaxes(positionTime[:, :], 1, 0)
        if speed.shape[0] == posTime.shape[0] - 1:
            speed = np.append(speed, speed[-1])
        speed = np.reshape(speed, [speed.shape[0], 1])
        # Make sure all variables stay in the time limits
        tmin = 0
        tmax = posTime[-1]
        myposTime = posTime[((posTime >= tmin) * (posTime <= tmax))[:, 0]]
        myspeed = speed[((posTime >= tmin) * (posTime <= tmax))[:, 0]]

        # Select the representative behavior to show
        epochToShow = []
        if IsMultiSessions:
            hab_list = [
                re.search("hab", sessionNames[x], re.IGNORECASE)
                for x in range(len(sessionNames))
            ]
            id_hab = [x for x in range(len(hab_list)) if hab_list[x] != None]
            sleep_list = [
                re.search("sleep", sessionNames[x], re.IGNORECASE)
                for x in range(len(sessionNames))
            ]
            id_sleep = [x for x in range(len(sleep_list)) if sleep_list[x] != None]
            if id_hab:
                for id in id_hab:
                    epochToShow.extend([sessionStart[id], sessionStop[id]])
            elif id_sleep:
                id_toshow = list(
                    range(id_sleep[0] + 1, id_sleep[1])
                )  # in between two sleeps
                for id in id_toshow:
                    epochToShow.extend([sessionStart[id], sessionStop[id]])
        else:
            epochToShow.extend(
                [myposTime[0, 0], myposTime[-1, 0]]
            )  # if no sleeps, or session names, or hab take everything
        maskToShow = ep.inEpochsMask(myposTime[:, 0], epochToShow)
        behToShow = positions[maskToShow, :]
        timeToShow = myposTime[maskToShow, 0]

        # Smooth speed
        s = np.r_[
            myspeed[window_len - 1 : 0 : -1],
            myspeed,
            myspeed[-2 : -window_len - 1 : -1],
        ]
        w = eval("np." + "hamming" + "(window_len)")
        myspeed2 = np.convolve(w / w.sum(), s[:, 0], mode="valid")[
            (window_len // 2 - 1) : -(window_len // 2)
        ]

        speedToshowSm = myspeed2[maskToShow]
        speedThreshold = np.mean(np.log(speedToshowSm[speedToshowSm >= 0] + 10 ** (-8)))
        speedFilter = speedToshowSm > np.exp(speedThreshold)

        # Figure
        fig = plt.figure(figsize=(7, 15))
        fig.suptitle("Speed threshold selection", fontsize=18, fontweight="bold")
        # Coordinates over time
        ax0 = fig.add_subplot(6, 2, (1, 2))
        (l1,) = ax0.plot(timeToShow[speedFilter], behToShow[speedFilter, 0], c="red")
        (l2,) = ax0.plot(timeToShow[speedFilter], behToShow[speedFilter, 1], c="orange")
        l3 = ax0.scatter(
            timeToShow[speedFilter],
            np.zeros(timeToShow[speedFilter].shape[0]) - 4,
            c="black",
            s=0.2,
        )
        ax0.set_ylabel("environmental \n variable")
        # Speed over time
        ax1 = fig.add_subplot(6, 2, (3, 4), sharex=ax0)
        (l4,) = ax1.plot(
            timeToShow[speedFilter], speedToshowSm[speedFilter], c="purple"
        )  # smoothed
        ax1.set_ylabel("speed")
        ax1.set_xlabel("Time (s)")
        # Speed histogram
        ax2 = fig.add_subplot(6, 2, 7)
        speed_log = np.log(speedToshowSm[np.not_equal(speedToshowSm, 0)] + 10 ** (-8))
        ax2.hist(speed_log, histtype="step", bins=200, color="blue")
        ax2.set_yticks([])
        l5 = ax2.axvline(speedThreshold, color="black")
        ax2.set_xlabel("log speed")
        ax2.set_xlim(
            np.percentile(speed_log[~np.isnan(speed_log)], 0.3),
            np.max(speed_log[~np.isnan(speed_log)]),
        )
        ax3 = fig.add_subplot(6, 2, 8)
        speed_plot = speedToshowSm[np.not_equal(speedToshowSm, 0)]
        ax3.hist(speed_plot, histtype="step", bins=200, color="blue")
        ax3.set_yticks([])
        l6 = ax3.axvline(np.exp(speedThreshold), color="black")
        ax3.set_xlabel("raw speed")
        ax3.set_xlim(0, np.percentile(speed_plot[~np.isnan(speed_plot)], 98))
        ax4 = fig.add_subplot(6, 2, (11, 12))
        slider = plt.Slider(
            ax4,
            " ",
            np.min(np.log(speedToshowSm[speedToshowSm >= 0] + 10 ** (-8))),
            np.max(np.log(speedToshowSm[speedToshowSm >= 0] + 10 ** (-8))),
            valinit=speedThreshold,
            valstep=0.01,
        )
        ax4.set_ylabel("speed Threshold")
        ax = [ax0, ax1, ax2, ax3, ax4]

        def update(val):
            speedThreshold = val
            speedFilter = speedToshowSm > np.exp(speedThreshold)
            l1.set_ydata(behToShow[speedFilter, 0])
            l2.set_ydata(behToShow[speedFilter, 1])
            l1.set_xdata(timeToShow[speedFilter])
            l2.set_xdata(timeToShow[speedFilter])
            l3.set_offsets(
                np.transpose(
                    np.stack(
                        [
                            timeToShow[speedFilter],
                            np.zeros(timeToShow[speedFilter].shape[0]) - 4,
                        ]
                    )
                )
            )
            l4.set_ydata(speedToshowSm[speedFilter])
            l4.set_xdata(timeToShow[speedFilter])
            l5.set_xdata(val)
            l6.set_xdata(np.exp(val))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        # Final value
        speedFilter = myspeed2 > np.exp(
            slider.val
        )  # Apply the value to the whole dataset

        f.create_array("/behavior", "speedMask", speedFilter)
        f.flush()
        f.close()
        # Change the way you save
        df = pd.DataFrame([np.exp(slider.val)])
        df.to_csv(folder + "speedFilterValue.csv")  # save the speed filter value


def select_epochs(folder, overWrite=True):
    # Find test set with most uniform covering of speed and environment variable.
    # provides then a little manual tool to change the size of the window
    # and its position.

    if not os.path.exists(folder + "nnBehavior.mat"):
        raise ValueError("this file does not exist :" + folder + "nnBehavior.mat")
    with tables.open_file(folder + "nnBehavior.mat", "a") as f:
        children = [c.name for c in f.list_nodes("/behavior")]
        if (
            overWrite == False
            and "trainEpochs" in children
            and "testEpochs" in children
        ):
            return

        # Get info from the file
        speedMask = f.root.behavior.speedMask[:]
        positions = f.root.behavior.positions
        positions = np.swapaxes(positions[:, :], 1, 0)
        speeds = f.root.behavior.speed
        positionTime = f.root.behavior.position_time
        positionTime = np.swapaxes(positionTime[:, :], 1, 0)
        speeds = np.swapaxes(speeds[:, :], 1, 0)
        if speeds.shape[0] == positionTime.shape[0] - 1:
            speeds = np.append(speeds, speeds[-1]).reshape(
                positionTime.shape[0], speeds.shape[1]
            )
        # We extract session names:
        sessionNames = [
            "".join([chr(c) for c in l[0][:, 0]])
            for l in f.root.behavior.SessionNames[:, 0]
        ]
        if sessionNames[0] != "Recording":
            IsMultiSessions = True
            sessionNames = [
                "".join([chr(c) for c in l[0][:, 0]])
                for l in f.root.behavior.SessionNames[:, 0]
            ]
            sessionStart = f.root.behavior.SessionStart[:, :][:, 0]
            sessionStop = f.root.behavior.SessionStop[:, :][:, 0]
        else:
            IsMultiSessions = False

        sessionValue = np.zeros(speedMask.shape[0])
        if IsMultiSessions:
            for k in range(len(sessionNames)):
                sessionValue[
                    ep.inEpochs(positionTime[:, 0], [sessionStart[k], sessionStop[k]])
                ] = k

        # Select the representative behavior without sleeps to show
        epochToShow = []
        if IsMultiSessions:
            sleep_list = [
                re.search("sleep", sessionNames[x], re.IGNORECASE)
                for x in range(len(sessionNames))
            ]
            id_sleep = [x for x in range(len(sleep_list)) if sleep_list[x] != None]
            if id_sleep:
                all_id = set(range(len(sessionNames)))
                id_toshow = list(all_id.difference(id_sleep))  # all except sleeps
                for id in id_toshow:
                    epochToShow.extend([sessionStart[id], sessionStop[id]])
        else:
            epochToShow.extend(
                [positionTime[0, 0], positionTime[-1, 0]]
            )  # if no sleeps, or session names, or hab take everything
        maskToShow = ep.inEpochsMask(positionTime[:, 0], epochToShow)
        behToShow = positions[maskToShow, :]
        timeToShow = positionTime[maskToShow, 0]
        sessionValue_toshow = sessionValue[maskToShow]
        if IsMultiSessions:
            SessionNames_toshow = [sessionNames[i] for i in id_toshow]

        ### Get times of show
        if IsMultiSessions:
            if id_sleep[0] == 0:  # if sleep goes first get the end of it
                ids = np.where(sessionValue == id_sleep[0])[0]
                st = positionTime[ids[-1] + 1]
                for i in id_sleep[1:]:
                    ids = np.where(sessionValue == i)[0]
                    st = np.append(st, (positionTime[ids[0]], positionTime[ids[-1]]))
                if st[-1] != positionTime[-1]:
                    st = np.append(st, positionTime[-1])
                else:
                    st = st[:-1]
            else:  # if it starts with maze
                st = positionTime[0]
                for i in id_sleep:
                    ids = np.where(sessionValue == i)[0]
                    st = np.append(st, (positionTime[ids[0]], positionTime[ids[-1]]))
                if st[-1] != positionTime[-1]:
                    st = np.append(st, positionTime[-1])
                else:
                    st = st[:-1]
            assert st.shape[0] % 2 == 0
            showtimes = tuple(zip(st[::2], st[1::2]))

        # Default train and test sets
        sizeTest = timeToShow.shape[0] // 10
        testSetId = timeToShow.shape[0] - timeToShow.shape[0] // 10
        bestTestSet = 0
        useLossPredTrainSet = False
        lossPredSetId = 0
        sizelossPredSet = timeToShow.shape[0] // 10
        SetData = {
            "sizeTestSet": sizeTest,
            "testSetId": testSetId,
            "bestTestSet": bestTestSet,
            "useLossPredTrainSet": useLossPredTrainSet,
            "lossPredSetId": lossPredSetId,
            "sizeLossPredSet": sizelossPredSet,
        }

        #### Next we provide a tool to manually change the bestTest set position
        # as well as its size:

        # Cut the cmap to avoid black colors
        min_val, max_val = 0.3, 1.0
        n = 20
        cmap = plt.get_cmap("nipy_spectral")
        colors = cmap(np.linspace(min_val, max_val, n))
        cmSessValue = mplt.colors.LinearSegmentedColormap.from_list("mycmap", colors)
        colorSess = cmSessValue(np.arange(len(sessionNames)) / (len(sessionNames)))
        keptSession = np.zeros(len(colorSess)) + 1  # a mask for the session
        if IsMultiSessions:
            keptSession[id_sleep] = 0

        fig = plt.figure()
        gs = plt.GridSpec(positions.shape[1] + 5, max(len(colorSess), 2), figure=fig)
        if IsMultiSessions:
            ax = [
                fig.add_subplot(gs[id, :]) for id in range(positions.shape[1])
            ]  # ax for feature display
            ax[0].get_shared_x_axes().join(ax[0], ax[1])
            # ax = [brokenaxes(xlims=showtimes, subplot_spec=gs[id,:]) for id in range(positions.shape[1])] #ax for feature display
        else:
            ax = [
                fig.add_subplot(gs[id, :]) for id in range(positions.shape[1])
            ]  # ax for feature display
            ax[0].get_shared_x_axes().join(ax[0], ax[1])
        ax += [fig.add_subplot(gs[-5, id]) for id in range(len(sessionNames))]
        ax += [
            fig.add_subplot(gs[-4, max(len(colorSess) - 3, 1) : max(len(colorSess), 2)])
        ]
        ax += [
            fig.add_subplot(gs[-4, 0 : max(len(colorSess) - 4, 1)]),
            fig.add_subplot(gs[-3, 0 : max(len(colorSess) - 4, 1)]),
        ]  # loss pred training set slider
        ax += [
            fig.add_subplot(gs[-2, :]),
            fig.add_subplot(gs[-1, : max(len(colorSess) - 4, 1)]),
        ]  # test set.
        ax += [
            fig.add_subplot(gs[-3, max(len(colorSess) - 3, 1) : max(len(colorSess), 2)])
        ]  # buttons for manual range selection
        ax += [
            fig.add_subplot(gs[-1, max(len(colorSess) - 3, 1) : max(len(colorSess), 2)])
        ]  # buttons for manual range selection

        if IsMultiSessions:
            trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                timeToShow, SetData, keptSession, starts=sessionStart, stops=sessionStop
            )
        else:
            trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                timeToShow, SetData, keptSession
            )

        ls = []
        for dim in range(positions.shape[1]):
            l1 = ax[dim].scatter(
                timeToShow[ep.inEpochs(timeToShow, trainEpoch)[0]],
                behToShow[ep.inEpochs(timeToShow, trainEpoch)[0], dim],
                c="black",
                s=0.5,
            )
            l2 = ax[dim].scatter(
                timeToShow[ep.inEpochs(timeToShow, testEpochs)[0]],
                behToShow[ep.inEpochs(timeToShow, testEpochs)[0], dim],
                c="red",
                s=0.5,
            )
            if SetData["useLossPredTrainSet"]:
                l3 = ax[dim].scatter(
                    timeToShow[ep.inEpochs(timeToShow, lossPredSetEpochs)[0]],
                    behToShow[ep.inEpochs(timeToShow, lossPredSetEpochs)[0], dim],
                    c="orange",
                    s=0.5,
                )
            else:
                l3 = ax[dim].scatter(
                    timeToShow[0], behToShow[0, dim], c="orange", s=0.5
                )
            ax[dim].get_yaxis().set_visible(False)
            if dim == 0:
                ax[dim].get_xaxis().set_visible(False)
            ls.append([l1, l2, l3])

            # display the sessions positions at the bottom:
            if IsMultiSessions:
                for idk, k in enumerate(id_toshow):
                    if len(np.where(np.equal(sessionValue_toshow, k))[0]) > 0:
                        ax[id].hlines(
                            np.min(
                                behToShow[
                                    np.logical_not(np.isnan(behToShow[:, dim])), dim
                                ]
                            )
                            - np.std(
                                behToShow[
                                    np.logical_not(np.isnan(behToShow[:, dim])), dim
                                ]
                            ),
                            xmin=timeToShow[
                                np.min(np.where(np.equal(sessionValue_toshow, k)))
                            ],
                            xmax=timeToShow[
                                np.max(np.where(np.equal(sessionValue_toshow, k)))
                            ],
                            color=colorSess[idk],
                            linewidth=3.0,
                        )

        # TODO add histograms here...
        sliderTest = plt.Slider(
            ax[-4],
            "test starting index",
            0,
            behToShow.shape[0] - SetData["sizeTestSet"],
            valinit=SetData["testSetId"],
            valstep=1,
        )
        sliderTestSize = plt.Slider(
            ax[-3],
            "test size",
            0,
            behToShow.shape[0],
            valinit=SetData["sizeTestSet"],
            valstep=1,
        )
        if SetData["useLossPredTrainSet"]:
            buttLossPred = plt.Button(ax[-7], "lossPred", color="orange")
        else:
            buttLossPred = plt.Button(ax[-7], "lossPred", color="white")
        sliderLossPredTrain = plt.Slider(
            ax[-6],
            "loss network training \n set starting index",
            0,
            behToShow.shape[0],
            valinit=0,
            valstep=1,
        )
        sliderLossPredTrainSize = plt.Slider(
            ax[-5],
            "loss network training set size",
            0,
            behToShow.shape[0],
            valinit=SetData["sizeTestSet"],
            valstep=1,
        )
        ButtlPManual = mplt.widgets.Button(
            ax[-2],
            "Choose lossPred set manually",
            color="sandybrown",
            hovercolor="peachpuff",
        )
        ButtTestManual = mplt.widgets.Button(
            ax[-1],
            "Choose test set manually",
            color="lightcoral",
            hovercolor="mistyrose",
        )
        axesInst = fig.add_axes([0.45, 0.89, 0.1, 0.05])
        buttInstructions = mplt.widgets.Button(
            axesInst, "Instructions", color="lightgrey", hovercolor="lightyellow"
        )

        # Next we add buttons to select the sessions we would like to keep:
        butts = [
            plt.Button(ax[len(ax) - k - 8], sessionNames[k], color=colorSess[k])
            for k in range(len(colorSess))
        ]
        if IsMultiSessions:
            for id in id_sleep:
                ax[len(ax) - id - 8].set_axis_off()

        def update(val):
            SetData["testSetId"] = sliderTest.val
            SetData["sizeTestSet"] = sliderTestSize.val
            SetData["lossPredSetId"] = sliderLossPredTrain.val
            SetData["sizeLossPredSet"] = sliderLossPredTrainSize.val

            if IsMultiSessions:
                trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                    timeToShow,
                    SetData,
                    keptSession,
                    starts=sessionStart,
                    stops=sessionStop,
                )
            else:
                trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                    timeToShow, SetData, keptSession
                )

            for dim in range(len(ls)):
                l1, l2, l3 = ls[dim]
                if isinstance(l1, list):
                    for iaxis in range(len(l1)):
                        l1[iaxis].set_offsets(
                            np.transpose(
                                np.stack(
                                    [
                                        timeToShow[
                                            ep.inEpochs(timeToShow, trainEpoch)[0]
                                        ],
                                        behToShow[
                                            ep.inEpochs(timeToShow, trainEpoch)[0], dim
                                        ],
                                    ]
                                )
                            )
                        )
                        l2[iaxis].set_offsets(
                            np.transpose(
                                np.stack(
                                    [
                                        timeToShow[
                                            ep.inEpochs(timeToShow, testEpochs)[0]
                                        ],
                                        behToShow[
                                            ep.inEpochs(timeToShow, testEpochs)[0], dim
                                        ],
                                    ]
                                )
                            )
                        )
                        if SetData["useLossPredTrainSet"]:
                            try:
                                ls[dim][2][iaxis].remove()
                            except:
                                pass
                        else:
                            try:
                                ls[dim][2][iaxis].remove()
                            except:
                                pass
                    if SetData["useLossPredTrainSet"]:
                        ls[dim][2] = ax[dim].scatter(
                            timeToShow[ep.inEpochs(timeToShow, lossPredSetEpochs)[0]],
                            behToShow[
                                ep.inEpochs(timeToShow, lossPredSetEpochs)[0], dim
                            ],
                            c="orange",
                            s=0.5,
                        )
                else:
                    l1.set_offsets(
                        np.transpose(
                            np.stack(
                                [
                                    timeToShow[ep.inEpochs(timeToShow, trainEpoch)[0]],
                                    behToShow[
                                        ep.inEpochs(timeToShow, trainEpoch)[0], dim
                                    ],
                                ]
                            )
                        )
                    )
                    l2.set_offsets(
                        np.transpose(
                            np.stack(
                                [
                                    timeToShow[ep.inEpochs(timeToShow, testEpochs)[0]],
                                    behToShow[
                                        ep.inEpochs(timeToShow, testEpochs)[0], dim
                                    ],
                                ]
                            )
                        )
                    )
                    if SetData["useLossPredTrainSet"]:
                        try:
                            ls[dim][2].remove()
                        except:
                            pass
                        ls[dim][2] = ax[dim].scatter(
                            timeToShow[ep.inEpochs(timeToShow, lossPredSetEpochs)[0]],
                            behToShow[
                                ep.inEpochs(timeToShow, lossPredSetEpochs)[0], dim
                            ],
                            c="orange",
                            s=0.5,
                        )
                    else:
                        try:
                            l3.remove()
                        except:
                            pass
            fig.canvas.draw_idle()

        sliderTest.on_changed(update)
        sliderTestSize.on_changed(update)
        sliderLossPredTrain.on_changed(update)
        sliderLossPredTrainSize.on_changed(update)

        def buttUpdate(id):
            def buttUpdate(val):
                if keptSession[id]:
                    butts[id].color = [0, 0, 0, 0]
                    keptSession[id] = 0
                else:
                    keptSession[id] = 1
                    butts[id].color = colorSess[id]
                update(0)

            return buttUpdate

        [b.on_clicked(buttUpdate(id)) for id, b in enumerate(butts)]

        def buttUpdateLossPred(val):
            if SetData["useLossPredTrainSet"]:
                buttLossPred.color = [0, 0, 0, 0]
                SetData["useLossPredTrainSet"] = False
            else:
                SetData["useLossPredTrainSet"] = True
                buttLossPred.color = "orange"
            update(0)
            return SetData["useLossPredTrainSet"]

        buttLossPred.on_clicked(buttUpdateLossPred)

        class rangeButton:
            nameDict = {"test": "test", "lossPred": "predicted loss"}

            def __init__(self, typeButt="test", relevantSliders=None):
                if typeButt == "test" or typeButt == "lossPred":
                    self.typeButt = typeButt
                if relevantSliders is None:
                    raise ValueError("relevantSliders must be a list of 2 sliders")
                else:
                    self.relevantSliders = relevantSliders

            def __call__(self, val):
                self.win = Toplevel()
                self.win.title(
                    f"Manual setting of the {self.nameDict[self.typeButt]} set"
                )
                self.win.geometry("400x200")

                textLabel = self.construct_label()
                self.rangeLabel = Label(self.win, text=textLabel)
                self.rangeLabel.place(relx=0.5, y=30, anchor="center")

                self.rangeEntry = Entry(self.win, width=18, bd=5)
                defaultValues = self.update_def_values(SetData)
                self.rangeEntry.insert(0, f"{defaultValues[0]}-{defaultValues[1]}")
                self.rangeEntry.place(relx=0.5, y=90, anchor="center")

                self.okButton = Button(self.win, width=5, height=1, text="Ok")
                self.okButton.bind(
                    "<Button-1>", lambda event: self.set_sliders_and_close()
                )
                self.okButton.place(relx=0.5, y=175, anchor="center")

                self.win.mainloop()

            def construct_label(self):
                text = f"Enter the range of the {self.nameDict[self.typeButt]} set in sec (e.g. 0-1000)"
                return text

            def update_def_values(self, SetData):
                nameId = f"{self.typeButt}SetId"
                nameSize = f"size{self.typeButt[0].upper()}{self.typeButt[1:]}Set"
                firstTS = round(timeToShow[SetData[nameId]], 2)
                lastId = round(timeToShow[SetData[nameId] + SetData[nameSize] - 1], 2)
                return [firstTS, lastId]

            def convert_entry_to_id(self):
                strEntry = self.rangeEntry.get()
                if len(strEntry) > 0:
                    try:
                        parsedRange = [float(num) for num in list(strEntry.split("-"))]
                        convertedRange = [
                            self.closestId(timeToShow, num) for num in parsedRange
                        ]
                        startId = convertedRange[0]
                        sizeSetinId = convertedRange[1] - convertedRange[0]

                        return startId, sizeSetinId
                    except ValueError:
                        self.okButton.configure(bg="red")
                        raise ValueError(
                            "Please enter a valid range in the format 'start-end'"
                        )

            def set_sliders_and_close(self):
                valuesForSlider = self.convert_entry_to_id()
                for ivalue, slider in enumerate(self.relevantSliders):
                    slider.set_val(valuesForSlider[ivalue])
                self.win.destroy()

            def closestId(self, arr, valToFind):
                return (np.abs(arr - valToFind)).argmin()

        ButtlPManual.on_clicked(
            rangeButton(
                typeButt="lossPred",
                relevantSliders=[sliderLossPredTrain, sliderLossPredTrainSize],
            )
        )
        ButtTestManual.on_clicked(
            rangeButton(typeButt="test", relevantSliders=[sliderTest, sliderTestSize])
        )

        def buttInstructionsShow(val):
            intructions_str = (
                "Black will become train dataset, red will become test dataset, "
                + "and orange (if lossPred button is pressed) will become a set "
                + "to fine-tune predicted loss.\n \n By pressing button with session names "
                + "you can choose which sessions to include in the analysis. "
                + "Either use sliders to regulate size and position of test and fine-tuning sets.\n \n"
                + "Or click on manual setter for test or fine-tuning sets \n \n"
                + "USE ZOOM TOOL OF THE FIGURE WINDOW TO AVOID GAPS IF YOU HAVE ANY \n \n"
                + "Simply close the window when you are satisfied with your choice."
            )

            win = Toplevel()
            win.title("Instructions")
            instLabel = Label(win, text=intructions_str, font=("Helvetica", 16))
            instLabel.pack()
            win.mainloop()

        buttInstructions.on_clicked(buttInstructionsShow)

        suptitle_str = (
            "Please choose train (black) and test (red) sets. You can add a set (orange) to "
            "fine-tune predicted loss (by pressing lossPred button)"
        )
        plt.suptitle(suptitle_str, fontsize=22)
        plt.text(
            x=0.93, y=0.82, s="X", fontsize=36, ha="center", transform=fig.transFigure
        )
        plt.text(
            x=0.93, y=0.71, s="Y", fontsize=36, ha="center", transform=fig.transFigure
        )

        if mplt.get_backend() == "QtAgg":
            plt.get_current_fig_manager().window.showMaximized()
        elif mplt.get_backend() == "TkAgg":
            plt.get_current_fig_manager().resize(
                *plt.get_current_fig_manager().window.maxsize()
            )
            # plt.get_current_fig_manager().window.state('zoomed') # on windows
        plt.show()

        if IsMultiSessions:
            trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                timeToShow, SetData, keptSession, starts=sessionStart, stops=sessionStop
            )
        else:
            trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                timeToShow, SetData, keptSession
            )

        if "testEpochs" in children:
            f.remove_node("/behavior", "testEpochs")
        f.create_array("/behavior", "testEpochs", testEpochs)
        if "trainEpochs" in children:
            f.remove_node("/behavior", "trainEpochs")
        f.create_array("/behavior", "trainEpochs", trainEpoch)

        if "keptSession" in children:
            f.remove_node("/behavior", "keptSession")
        f.create_array("/behavior", "keptSession", keptSession)

        if "lossPredSetEpochs" in children:
            f.remove_node("/behavior", "lossPredSetEpochs")
        if SetData["useLossPredTrainSet"]:
            f.create_array("/behavior", "lossPredSetEpochs", lossPredSetEpochs)
        else:
            f.create_array("/behavior", "lossPredSetEpochs", [])

        f.flush()  # effectively write down the modification we just made

        fig, ax = plt.subplots()
        trainMask = ep.inEpochsMask(positionTime, trainEpoch)[:, 0]
        testMask = ep.inEpochsMask(positionTime, testEpochs)[:, 0]
        ax.scatter(positionTime[trainMask], positions[trainMask, 0], c="black")
        ax.scatter(positionTime[testMask], positions[testMask, 0], c="red")
        ax.set_title("Linearized coordinate of the animal")
        if SetData["useLossPredTrainSet"]:
            lossPredMask = ep.inEpochsMask(positionTime, lossPredSetEpochs)[:, 0]
            ax.scatter(
                positionTime[lossPredMask], positions[lossPredMask, 0], c="orange"
            )
        fig.show()


class DataHelper:
    """A class to detect and describe the main properties on the signal and behavior"""

    def __init__(self, path, mode, jsonPath=None):
        self.path = path
        self.mode = mode

        if not os.path.exists(self.path.xml):
            raise ValueError("this file does not exist: " + self.path.xml)
        # if not os.path.exists(self.path.dat):
        # raise ValueError('this file does not exist: '+ self.path.dat)

        # if self.mode == "decode":
        #     self.list_channels, self.samplingRate, self.nChannels = get_params(self.path.xml)
        #     if os.path.isfile(self.path.folder + "timestamps.npy"):
        #         self.nChannels = int( os.path.getsize(self.path.dat) \
        #             / 2 \
        #             / np.load(self.path.folder + "timestamps.npy").shape[0] )
        #     # TODO: change this to allow a varying number of features:
        #     self.position = np.array([0,0], dtype=float).reshape([1,2])
        #     self.positionTime = np.array([0], dtype=float)
        #     self.startTime = 0
        #     self.stopTime = float("inf")
        #     # self.epochs = {"train": [], "test": [0, float("inf")]}
        # else:
        self.list_channels, self.samplingRate, self.nChannels = get_params(
            self.path.xml
        )
        if self.mode == "decode":
            self.fullBehavior = get_behavior(
                self.path.folder, getfilterSpeed=False, decode=True
            )
        else:
            self.fullBehavior = get_behavior(self.path.folder)
        self.positions = self.fullBehavior["Positions"]
        self.positionTime = self.fullBehavior["positionTime"]

        self.stopTime = self.positionTime[
            -1
        ]  # max(self.epochs["train"] + self.epochs["test"])
        self.startTime = self.positionTime[0]

    def nGroups(self):
        return len(self.list_channels)

    def numChannelsPerGroup(self):
        return [len(self.list_channels[n]) for n in range(self.nGroups())]

    def maxPos(self):
        maxPos = np.max(
            self.positions[np.logical_not(np.isnan(np.sum(self.positions, axis=1)))]
        )
        return maxPos if self.mode != "decode" else 1

    def dim_output(self):
        return self.positions.shape[1]

    def getThresholds(self):
        idx = 0
        nestedThresholds = []
        for group in range(len(self.list_channels)):
            temp = []
            for channel in range(len(self.list_channels[group])):
                temp.append(self.thresholds[idx])
                idx += 1
            nestedThresholds.append(temp)
        return nestedThresholds

    def setThresholds(self, thresholds):
        assert [len(d) for d in thresholds] == [len(s) for s in self.list_channels]
        self.thresholds = [i for d in thresholds for i in d]
