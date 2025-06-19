# Load libs
import os
import re
import sys
import xml.etree.ElementTree as ET
from tkinter import Button, Entry, Label, Toplevel
from typing import Literal, Optional

# import matplotlib as mplt
# mplt.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables

sys.path.append("./importData")

import matplotlib as mplt

# Custom codes
from importData import epochs_management as ep
from simpleBayes.butils import kdenD


def get_params(pathToXml):
    """
    This function parses the xml file - as neuroscope would.

    Parameters
    ----------
    pathToXml : str, path to the xml file

    Returns
    -------
    listChannels : list, list of channels
    samplingRate : float, sampling rate
    nChannels : int, number of channels
    """
    listChannels = []
    samplingRate = None
    nChannels = None
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
                # by filtering for channels, we only get the `Spike Groups` from neuroscope, i.e. the selected channels from each
                # anatomical group (HPC, PFC, etc.) that seem to have spiking neurons
                for br4Elem in br3Elem:
                    if br4Elem.tag != "channels":
                        continue
                    for br5Elem in br4Elem:
                        if br5Elem.tag != "channel":
                            continue
                        group.append(int(br5Elem.text))
                # each channel is a group of spike channels
                listChannels.append(group)
    for br1Elem in root:
        if br1Elem.tag != "acquisitionSystem":
            continue
        for br2Elem in br1Elem:
            if br2Elem.tag == "samplingRate":
                samplingRate = float(br2Elem.text)
            if br2Elem.tag == "nChannels":
                nChannels = int(br2Elem.text)

    if samplingRate is None or nChannels is None or not listChannels:
        raise ValueError(
            f"""The xml file does not contain the required information
            Please check the xml file: {pathToXml}
            """
        )
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
def get_behavior(
    folder: str,
    bandwidth: Optional[int] = None,
    getfilterSpeed: bool = True,
    decode: bool = False,
        "extinction",
        None,
) -> dict[str, np.ndarray]:
    """
    Load the behavior data from the nnBehavior.mat file

    Parameters
    ----------
    folder : str, where the mat `nnBehavior.mat` file is located
    bandwidth : int, optional (default=None)
    getfilterSpeed : bool, optional, whether to the speed filter on train/test data. Should be True for training (ann|bayes), false otherwise.
    decode : bool, optional, wether to train-test split the data or not.

    Returns
    -------
    dict[str, np.ndarray] with the following
    keys:
    Positions, positionTime, Speed, Bandwidth, Times

    """

    # check if the file exists
    filename = os.path.join(folder, "nnBehavior.mat")
    if not os.path.exists(filename):
        raise ValueError("this file does not exist :" + folder + "nnBehavior.mat")
    if phase is not None:
        filename = os.path.join(folder, "nnBehavior_" + phase + ".mat")
            assert tables.is_hdf5_file(folder + "nnBehavior.mat")
            import shutil

            print("weird to copy that file now")

            shutil.copyfile(
                folder + "nnBehavior.mat",
                folder + "nnBehavior_" + phase + ".mat",
                follow_symlinks=True,
            )
    # Extract basic behavior
    with tables.open_file(filename) as f:
        positions = f.root.behavior.positions
        positions = np.swapaxes(positions[:, :], 1, 0)
        positionTime = np.swapaxes(positionTime[:, :], 1, 0)
        speed = f.root.behavior.speed
        speed = np.swapaxes(speed[:, :], 1, 0)
        if bandwidth is None:
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
            if sessionNames[0] != "Recording":
                sessionStart = f.root.behavior.SessionStart[:, :][:, 0]
                sessionStop = f.root.behavior.SessionStop[:, :][:, 0]
                import re

                SessionEpochs = dict()
                for phase in ["pre", "hab", "cond", "post", "extinction"]:
                    if phase == "pre":
                        # Look for Pre sessions or Hab sessions
                        pattern = "(pre|hab)"
                    elif phase == "hab":
                        pattern = "hab"
                    elif phase == "cond":
                        pattern = "cond"
                    elif phase == "post":
                        pattern = "(post|extinction|extinct|ext)"
                    elif phase == "extinction":
                        pattern = "(extinction|extinct|ext)"

                    list_sessions = [
                        re.search(pattern, name, re.IGNORECASE) for name in sessionNames
                    ]
                    id_sessions = [
                        i
                        if session is not None
                    ]
                    # create epochs from the session start and stop times
                    for id in id_sessions:
                        SessionEpochs[phase].extend([sessionStart[id], sessionStop[id]])

        # Train and test epochs
        if (
            not decode
        ):  # Not applicable for decode mode where all dataset is to be decoded
                trainEpochs = np.concatenate(f.root.behavior.trainEpochs)
                testEpochs = np.concatenate(f.root.behavior.testEpochs)
            elif len(f.root.behavior.trainEpochs.shape) == 1:
                trainEpochs = f.root.behavior.trainEpochs[:]
        if not decode:
                    np.argmin(np.abs(positionTime - trainEpochs[2 * i])),
                ]
                for i in range(len(trainEpochs) // 2)
            ]
            learningTime = [
                np.sum(
                    np.multiply(
                        speedFilter[lEpochIndex[i][1] : lEpochIndex[i][0]],
                )

        if "ref" in f.root.behavior:  # NEW: try to load the images if they exist.
            behavior_data["ref"] = f.root.behavior.ref[:]
        if "aligned_ref" in f.root.behavior:
            behavior_data["aligned_ref"] = f.root.behavior.aligned_ref[:]
        if "xyOutput" in f.root.behavior:
            behavior_data["xyOutput"] = f.root.behavior.xyOutput[:]
        if "ratioIMAonREAL" in f.root.behavior:
            behavior_data["ratioIMAonREAL"] = float(
                f.root.behavior.ratioIMAonREAL.read()
            )
    if getfilterSpeed:
        behavior_data["Times"]["speedFilter"] = speedFilter
    if np.sum(sleepPeriods) > 0:
        behavior_data["Times"]["sleepEpochs"] = sleepPeriods
        if "shock_zone" in f.root.behavior:

    return behavior_data


def speed_filter(
    folder: str, overWrite: bool = True, phase=None, template=None, force: bool = False
) -> None:
    """
    A simple tool to set up a threshold on the speed value
    The speed threshold is then implemented through a speed_mask:
    a boolean array indicating for each index (i.e measured feature time step)
    if it is above threshold or not.

    """
    # TODO: change the order of the epochs AND be able to select the training set in the middle of the dataset
    # Parameters
    window_len = 14  # changed following Dima's advice

    filename = os.path.join(folder, "nnBehavior.mat")
    # as this if the first function, it should create the appropriate nnbehavior. Next functions will check for file existence
    if phase is not None:
        filename = os.path.join(folder, "nnBehavior_" + phase + ".mat")
        if not os.path.exists(filename):
            if template is not None:
                templ_file = f"nnBehavior_{template}.mat"
            else:
                follow_symlinks=True,
            )
    # Extract basic behavior
    with tables.open_file(filename, "a") as f:
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

        if not force:
            # Figure
            fig = plt.figure(figsize=(7, 15))
            fig.suptitle("Speed threshold selection", fontsize=18, fontweight="bold")
            # Coordinates over time
            ax0 = fig.add_subplot(6, 2, (1, 2))
            (l1,) = ax0.plot(
                timeToShow[speedFilter], behToShow[speedFilter, 0], c="red"
            )
            (l2,) = ax0.plot(
                timeToShow[speedFilter], behToShow[speedFilter, 1], c="orange"
            )
            l3 = ax0.scatter(
                timeToShow[speedFilter],
                np.zeros(timeToShow[speedFilter].shape[0]) - 0.5,
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

            # override default matplotlib rc ticks
            plt.setp(ax0.get_xticklabels(), visible=False)
            ax0.tick_params(which="both", labelsize=15, labelbottom=False)
            ax1.tick_params(which="both", labelsize=15)

            # Speed histogram
            ax2 = fig.add_subplot(6, 2, 7)
            speed_log = np.log(
                speedToshowSm[np.not_equal(speedToshowSm, 0)] + 10 ** (-8)
            )
            ax2.hist(speed_log, histtype="step", bins=200, color="blue")
            plt.setp(ax2.get_yticklabels(), visible=False)
            l5 = ax2.axvline(speedThreshold, color="black")
            ax2.set_xlabel("log speed")
            ax2.set_xlim(
                np.percentile(speed_log[~np.isnan(speed_log)], 0.3),
                np.max(speed_log[~np.isnan(speed_log)]),
            )
            ax3 = fig.add_subplot(6, 2, 8)
            speed_plot = speedToshowSm[np.not_equal(speedToshowSm, 0)]
            ax3.hist(speed_plot, histtype="step", bins=200, color="blue")
            plt.setp(ax3.get_yticklabels(), visible=False)
            l6 = ax3.axvline(np.exp(speedThreshold), color="black")
            ax3.set_xlabel(f"raw speed ({np.exp(speedThreshold):.2f} cm/s)")
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
                l2.set_xdata(timeToShow[speedFilter])
                l3.set_offsets(
                    np.transpose(
                        np.stack(
                            [
                                timeToShow[speedFilter],
                                np.zeros(timeToShow[speedFilter].shape[0]) - 0.5,
                            ]
                        )
                    )
                )
                l5.set_xdata(val)
                l6.set_xdata(np.exp(val))
            speedThreshold = slider.val
        # Final value
        speedFilter = myspeed2 > np.exp(
            slider.val
        )  # Apply the value to the whole dataset

        f.create_array("/behavior", "speedMask", speedFilter)
        f.flush()
        f.close()
        # Change the way you save
        df = pd.DataFrame([speedThreshold])
        df.to_csv(folder + "speedFilterValue.csv")  # save the speed filter value


def select_epochs(folder: str, overWrite=True):
    folder: str,
    overWrite: bool = True,
    phase=None,
    force: bool = False,
    find_best_sets: bool = False,
    isPredLoss: bool = False,
):
    """
    Find test set with most uniform covering of speed and environment variable.
    provides then a little manual tool to change the size of the window
    and its position.

    folder: str, path to the folder containing the nnBehavior.mat file
    overWrite: bool, whether to overwrite the existing train and test epochs
    phase: str, whether to pre-select only some specific sessions (pre, hab, cond, or post for now)
    force: bool, whether to force the function to run without figure preview
    find_best_sets: bool, whether to find the best test set based on the entropy of the speed and environment variable
    returns
    -------
    None, but creates a nnBehavior_{phase}.mat file with the trainEpochs, testEpochs, and lossPredSetEpochs
    """

    # create globals variables
    global SetData, IsMultiSessions
    global timeToShow, keptSession, sessionStart, sessionStop, ep
    from importData import epochs_management as ep

    # TODO: add a way to select training set in the middle of the dataset
    filename = os.path.join(folder, "nnBehavior.mat")
    if not os.path.exists(filename):
        raise ValueError("this file does not exist :" + folder + "nnBehavior.mat")
    # As we will be selecting specific epochs for training and testing different phases
    # we will need to copy the behavior data to a new file called nnBehavior_{phase}.mat
    if phase is not None:
        filename = os.path.join(folder, "nnBehavior_" + phase + ".mat")
            print("weird to copy that file now")
            assert tables.is_hdf5_file(folder + "nnBehavior.mat")
            import shutil

            shutil.copyfile(
                folder + "nnBehavior.mat",
                folder + "nnBehavior_" + phase + ".mat",
        children = [c.name for c in f.list_nodes("/behavior")]
        if not overWrite and "trainEpochs" in children and "testEpochs" in children:
            and "trainEpochs" in children
            and "testEpochs"
            and "lossPredSetEpochs" in children
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
            id_sleep = [x for x in range(len(sleep_list)) if sleep_list[x] is not None]
            if id_sleep:
                all_id = set(range(len(sessionNames)))
                id_toshow = list(all_id.difference(id_sleep))  # all except sleeps
                if phase is not None:
                    if phase == "pre":
                        # Look for Pre sessions or Hab sessions
                        pattern = "(pre|hab)"
                    elif phase == "preNoHab":
                        pattern = "pre"
                    elif phase == "hab":
                        pattern = "hab"
                    elif phase == "cond":
                        pattern = "cond"
                    elif phase == "post":
                        pattern = "(post|extinction|extinct|ext)"
                    elif phase == "postNoExtinction":
                        pattern = "post"
                    elif phase == "extinction":
                        pattern = "(extinction|extinct|ext)"
                    else:
                        raise ValueError(
                            "phase must be one of pre, hab, cond, extinction, or post"
                        )

                    # create preselection list
                        for x in range(len(sessionNames))
                    ]
                    id_pre = [
                        x for x in range(len(pre_list)) if pre_list[x] is not None
                    ]
                    id_toselectPRE = list(
                for id in id_toshow:
                    epochToShow.extend([sessionStart[id], sessionStop[id]])
                        epochToSelectPRE.extend(
                            [sessionStart[id], sessionStop[id]]
                        )  # the only epochs we want to select by default
        else:
            epochToShow.extend(
                [positionTime[0, 0], positionTime[-1, 0]]
            )  # if no sleeps, or session names, or hab take everything

        maskToShow = ep.inEpochsMask(positionTime[:, 0], epochToShow)
        behToShow = positions[maskToShow, :]
        timeToShow = positionTime[maskToShow, 0]
        speedsToShow = speeds[maskToShow, :]
        timeToShowPRE = timeToShow
        speedsToShowPRE = speedsToShow
        xmin, xmax = timeToShow[0], timeToShow[-1]
        sessionValue_toshow = sessionValue[maskToShow]

        if phase is not None:
            maskToShowPRE = ep.inEpochsMask(positionTime[:, 0], epochToSelectPRE)
            timeToShowPRE = positionTime[maskToShowPRE, 0]
            speedsToShowPRE = speeds[maskToShowPRE, :]
            xmin, xmax = timeToShowPRE[0], timeToShowPRE[-1]

                Available Sessions are:
                {session_pretty}
                """
            )

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
        sizeTest = (
            timeToShow.shape[0] // 10
            if phase == "all"
            else timeToShowPRE.shape[0] // 10
        )
        testSetId = (
            timeToShow.shape[0] - timeToShow.shape[0] // 10
            if phase == "all"
            else idx_cut + timeToShowPRE.shape[0] - timeToShowPRE.shape[0] // 10
        )

        useLossPredTrainSet = False  # whether to use a loss prediction training set
        lossPredSetId = 0  # the loss prediction set id
        sizelossPredSet = (
            timeToShow.shape[0] // 10
            if phase == "all"
            else timeToShowPRE.shape[0] // 10
        )

        if find_best_sets:
            from tqdm import tqdm

            print("Evaluating the entropy of each possible test set")
            entropiesPositions = []
            entropiesSpeeds = []
            epsilon = 10 ** (-9)
            for idx_testSet in tqdm(
                np.arange(
                    idx_cut,
                    stop=idx_cut + timeToShowPRE.shape[0] - sizeTest,
                    step=sizeTest,
                )
            ):
                # The environmental variable are discretized by equally space bins
                # such that there is 45*...*45 bins per dimension
                # we then fit over the test set a kernel estimation of the probability distribution
                # and evaluate it over the bins
                _, probaFeatures = kdenD(
                    behToShow[idx_testSet : idx_testSet + sizeTest, :], bandwidth=1.0
                )
                # We then compute the entropy of the obtained distribution:
                entropiesPositions += [
                    -np.sum(probaFeatures * np.log(probaFeatures + epsilon))
                ]
                _, probaFeatures = kdenD(
                    speedsToShow[idx_testSet : idx_testSet + sizeTest, :], bandwidth=1.0
                )
                # We then compute the entropy of the obtained distribution:
                entropiesSpeeds += [
                    -np.sum(probaFeatures * np.log(probaFeatures + epsilon))
                ]
            totEntropy = np.array(entropiesSpeeds) + np.array(entropiesPositions)
            bestTestSet = np.argmax(totEntropy)
            testSetId = bestTestSet * sizeTest + idx_cut
            print("Found best test set at index", bestTestSet)
            if isPredLoss:
                useLossPredTrainSet = True
                bestPLSet = np.argsort(entropiesPositions)[-2]
                print("Found best loss pred set at index", bestPLSet)
                lossPredSetId = bestPLSet * sizelossPredSet + idx_cut
            bestPLSet = 0  # the best loss pred set is the one that covers the most of the speed and the environment variable

        # TODO: implement this best test set.
        SetData = {
            "sizeTestSet": sizeTest,
            "testSetId": testSetId,
            "bestTestSet": bestTestSet,
            "useLossPredTrainSet": useLossPredTrainSet,
            "lossPredSetId": lossPredSetId,
            "sizeLossPredSet": sizelossPredSet,
            "bestPLSet": bestPLSet,
        }
        # as well as its size:

        cmap = plt.get_cmap("nipy_spectral")
        keptSession = np.ones(len(sessionNames))  # a mask for the session
        if IsMultiSessions:
            keptSession[id_sleep] = 0
            if phase is not None:
                keptSession = np.zeros(len(sessionNames))
                keptSession[id_toselectPRE] = 1

        if force:
            if IsMultiSessions:
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
        else:
            #### Next we provide a tool to manually change the bestTest set position
            # as well as its size:

            # Cut the cmap to avoid black colors
            min_val, max_val = 0.3, 1.0
            n = 20
            cmap = plt.get_cmap("nipy_spectral")
                "mycmap", colors
            )
            colorSess = cmSessValue(np.arange(len(sessionNames)) / (len(sessionNames)))
            fig = plt.figure()
            gs = plt.GridSpec(
                positions.shape[1] + 5, max(len(colorSess), 2), figure=fig
            )

                    timeToShow[ep.inEpochs(timeToShow, lossPredSetEpochs)[0]],
            if IsMultiSessions:
                ax = [
                    fig.add_subplot(gs[id, :]) for id in range(positions.shape[1])
                ]  # ax for feature display
                ax[0].get_shared_x_axes().join(ax[0], ax[1])
                # ax = [brokenaxes(xlims=showtimes, subplot_spec=gs[id,:]) for id in range(positions.shape[1])] #ax for feature display
            else:
                    fig.add_subplot(gs[id, :]) for id in range(positions.shape[1])
                ]  # ax for feature display
                ax[0].get_shared_x_axes().join(ax[0], ax[1])

            ax += [
                fig.add_subplot(gs[-5, id]) for id in range(len(sessionNames))
            ]  # ax for session names
            ax += [
                fig.add_subplot(
                    gs[-4, max(len(colorSess) - 3, 1) : max(len(colorSess), 2)]
                )
            ]  # loss pred ON/OFF button
            ax += [
                fig.add_subplot(
                    gs[-4, 0 : max(len(colorSess) - 4, 1)]
                ),  # starting index of losspred
                fig.add_subplot(
                    gs[-3, 0 : max(len(colorSess) - 4, 1)]
                ),  # size of losspred
            ]  # loss pred training set slider
            ax += [
                fig.add_subplot(gs[-2, :]),  # test set starting index
                fig.add_subplot(gs[-1, : max(len(colorSess) - 4, 1)]),  # test set size
                fig.add_subplot(
                    gs[-1, max(len(colorSess) - 3, 1) : max(len(colorSess), 2)]
                )
            ]  # buttons for manual range selection (Test set)

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

            ls = []
            for dim in range(positions.shape[1]):
                l1 = ax[dim].scatter(
                    timeToShow[
                        ep.inEpochs(timeToShow, trainEpoch)[0]
                    ],  # inEpochs returns a tuple
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
                ax[dim].set_xlim(xmin, xmax)
                ax[dim].get_yaxis().set_visible(False)
                # change xlabel font size
                ax[dim].tick_params(axis="both", which="both", labelsize=10)
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
                                color=colorSess[id],
                                linewidth=3.0,
                            )

            # TODO: add histograms here...
            sliderTest = plt.Slider(
                ax[-4],
                "test starting index",
                0,
                behToShow.shape[0],
                # - SetData["sizeTestSet"],
                valinit=SetData["testSetId"],
                valstep=1,
            )

            sliderTest.label.set_size(10)
            sliderTest.valtext.set_fontsize(13)
            sliderTest.valtext.set_position((0.5, -0.1))

            sliderTestSize = plt.Slider(
                ax[-3],
                "test size",
                0,
                behToShow.shape[0],
                valinit=SetData["sizeTestSet"],
                valstep=1,
            )
            sliderTestSize.label.set_size(10)
            sliderTestSize.valtext.set_fontsize(13)
            sliderTestSize.valtext.set_position((0.5, -0.1))
            if SetData["useLossPredTrainSet"]:
                buttLossPred = plt.Button(ax[-7], "lossPred", color="orange")
            else:
                buttLossPred = plt.Button(ax[-7], "lossPred", color="white")
            buttLossPred.label.set_size(11)

            sliderLossPredTrain = plt.Slider(
                ax[-6],
                "loss network training\nset starting index",
                0,
                behToShow.shape[0],
                valinit=0,
                valstep=1,
            )
            sliderLossPredTrain.label.set_size(10)
            sliderLossPredTrain.valtext.set_fontsize(13)
            sliderLossPredTrain.valtext.set_position((0.5, -0.1))
            sliderLossPredTrainSize = plt.Slider(
                ax[-5],
                "loss network\ntraining set size",
                0,
                behToShow.shape[0],
                valinit=SetData["sizeTestSet"],
                valstep=1,
            )
            sliderLossPredTrainSize.label.set_size(10)
            sliderLossPredTrainSize.valtext.set_fontsize(13)
            sliderLossPredTrainSize.valtext.set_position((0.5, -0.1))

            ButtlPManual = mplt.widgets.Button(
                ax[-2],
                "Choose lossPred\nset manually",
                color="sandybrown",
                hovercolor="peachpuff",
            )
            ButtlPManual.label.set_size(11)
            ButtTestManual = mplt.widgets.Button(
                ax[-1],
                "Choose test\nset manually",
                color="lightcoral",
                hovercolor="mistyrose",
            )
            ButtTestManual.label.set_size(11)
            axesInst = fig.add_axes([0.45, 0.89, 0.1, 0.05])
            buttInstructions = mplt.widgets.Button(
                axesInst, "Instructions", color="lightgrey", hovercolor="lightyellow"
            )

            # Next we add buttons to select the sessions we would like to keep:
            butts = [
                plt.Button(
                    ax[positions.shape[1] + k],
                    sessionNames[k],
                    color=colorSess[k]
                    if keptSession[k] == 1
                    else [0, 0, 0, 0],  # we color only the pre-selected sessions
                )
                for k in range(len(colorSess))
            ]
            # Modify the text properties of each button
            for butt in butts:
                butt.label.set_fontsize(10)  # Reduce font size (adjust as needed)
                butt.label.set_rotation(-90)  # Rotate 90 degrees
                butt.label.set_verticalalignment("center")  # Center the rotated text
                butt.label.set_horizontalalignment("center")  # Center the rotated text

            if IsMultiSessions:
                for id in id_sleep:
                    ax[positions.shape[1] + id].set_axis_off()

            def get_current_test_train_epochs():
                return globals().get("testEpochs", None), globals().get(
                    "trainEpoch", None
                )

            def can_click_button(id):
                """
                Allow to click on the button only if the session doesn't contain all the test set or all the train set or all the lossPred set.
                """
                # Get fresh testEpochs from current slider state
                try:
                    testEpochs, trainEpoch = get_current_test_train_epochs()
                    if testEpochs is None or trainEpoch is None:
                        return True

                except Exception as e:
                    print(f"Error getting testEpochs: {e}")
                    return True

                episodes = [testEpochs, trainEpoch]
                session_names = ["test", "train"]

                if SetData["useLossPredTrainSet"]:
                    episodes.append(lossPredSetEpochs)
                    session_names.append("lossPred")
                if IsMultiSessions:
                    if np.sum(keptSession) == 1 and keptSession[id] == 1:
                        # If only one session is kept, allow clicking
                        return False
                    for i, epi in enumerate(episodes):
                        # Check if the session contains all the test set or all the train set or all the lossPred set
                        if (
                            (
                                keptSession[id] == 1
                                and ep.intersect_with_session(
                                    epi,
                                    [keptSession[id]],
                                    starts=[sessionStart[id]],
                                    stops=[sessionStop[id]],
                                ).sum()
                                > 0
                                and ep.intersect_with_session(
                                    epi,
                                    [keptSession[id]],
                                    starts=[sessionStart[id]],
                                    stops=[sessionStop[id]],
                                ).sum()
                                == ep.intersect_with_session(
                                    epi,
                                    keptSession,
                                    starts=sessionStart,
                                    stops=sessionStop,
                                ).sum()
                            )  # if the session is kept and intersects with the epochs
                            and (
                                (
                                    sessionStart[id]
                                    <= np.min(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                    and sessionStop[id]
                                    >= np.max(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                    and keptSession[id + 1] == 0
                                    and keptSession[id - 1] == 1
                                )
                                or (
                                    sessionStart[id]
                                    <= np.min(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                    and sessionStop[id]
                                    >= np.max(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                    and keptSession[id + 1] == 0
                                    and keptSession[id - 1] == 0
                                )
                                or (
                                    sessionStart[id]
                                    <= np.min(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                    and sessionStop[id]
                                    >= np.max(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                    and keptSession[id + 1] == 1
                                    and keptSession[id - 1] == 0
                                )
                                or (
                                    np.min(timeToShow[ep.inEpochs(timeToShow, epi)[0]])
                                    <= xmin
                                    and sessionStop[id]
                                    >= np.max(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                )
                                or (
                                    np.max(timeToShow[ep.inEpochs(timeToShow, epi)[0]])
                                    >= xmax
                                    and sessionStart[id]
                                    <= np.min(
                                        timeToShow[ep.inEpochs(timeToShow, epi)[0]]
                                    )
                                )
                            )
                        ):
                            return False

                return True

            def update_button_states():
                """Check all buttons and update their appearance based on current conditions"""
                if not IsMultiSessions:
                    return

                for button_id in range(len(butts)):
                    if button_id < len(keptSession) and keptSession[button_id] == 1:
                        if not can_click_button(button_id):
                            # Make button appear disabled/grayed out
                            butts[button_id].color = [0.5, 0.5, 0.5, 0.5]  # Gray
                        else:
                            # Restore normal appearance
                            butts[button_id].color = colorSess[button_id]
                    else:
                        # If the session is not kept, make it transparent
                        butts[button_id].color = [0, 0, 0, 0]

            def update(val):
                global trainEpoch, testEpochs, lossPredSetEpochs
                global xmin, xmax
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
                                                ep.inEpochs(timeToShow, trainEpoch)[0],
                                                dim,
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
                                                ep.inEpochs(timeToShow, testEpochs)[0],
                                                dim,
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
                                s=0.5,
                            )
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
                                ls[dim][2].remove()
                            except:
                                pass
                                timeToShow[
                                    ep.inEpochs(timeToShow, lossPredSetEpochs)[0]
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
                    # modify the xlim of the axes according to the changed epochs
                    xmin, xmax = (
                        min(
                            np.min(timeToShow[ep.inEpochs(timeToShow, trainEpoch)[0]]),
                            np.min(timeToShow[ep.inEpochs(timeToShow, testEpochs)[0]]),
                        ),
                        max(
                            np.max(timeToShow[ep.inEpochs(timeToShow, trainEpoch)[0]]),
                            np.max(timeToShow[ep.inEpochs(timeToShow, testEpochs)[0]]),
                        ),
                    )
                    ax[dim].set_xlim(xmin, xmax)
                fig.canvas.draw_idle()

                """
                Create a function to update the button state when clicked.
                """

                def buttUpdate_inner(val):
                    if not can_click_button(id):
                        return
                    if keptSession[id]:
                        butts[id].color = [0, 0, 0, 0]
                        keptSession[id] = 0
                    else:
                        keptSession[id] = 1
                        butts[id].color = colorSess[id]

                    update_with_buttons(0)

                return buttUpdate_inner

            def update_with_buttons(val):
                """
                Update the button states and the figure when a button is clicked.
                """
                update(val)
                update_button_states()

            # Finally, Connect the sliders and buttons to the update function

            sliderTest.on_changed(update_with_buttons)
            sliderTestSize.on_changed(update_with_buttons)
            sliderLossPredTrainSize.on_changed(update_with_buttons)

            [b.on_clicked(buttUpdate(id)) for id, b in enumerate(butts)]


                if SetData["useLossPredTrainSet"]:
                    buttLossPred.color = [0, 0, 0, 0]
                else:
                    buttLossPred.color = "orange"
                update(0)
                return SetData["useLossPredTrainSet"]

            buttLossPred.on_clicked(buttUpdateLossPred)

            class rangeButton:
                nameDict = {"test": "test", "lossPred": "predicted loss"}

                SetData["useLossPredTrainSet"] = True
                def __init__(self, typeButt="test", relevantSliders=None):
                    if typeButt == "test" or typeButt == "lossPred":
                        self.typeButt = typeButt
                    if relevantSliders is None:
                    else:
                        self.relevantSliders = relevantSliders

                def __call__(self, val):
                    self.win.title(
                        f"Manual setting of the {self.nameDict[self.typeButt]} set"
                    )
                    self.win.geometry("400x200")

                    textLabel = self.construct_label()
                    self.rangeLabel = Label(self.win, text=textLabel)
                    self.rangeLabel.place(relx=0.5, y=30, anchor="center")

            def __init__(self, typeButt="test", relevantSliders=None):
                    self.rangeEntry = Entry(self.win, width=18, bd=5)
                    defaultValues = self.update_def_values(SetData)
                    self.rangeEntry.insert(0, f"{defaultValues[0]}-{defaultValues[1]}")
                    self.rangeEntry.place(relx=0.5, y=90, anchor="center")

            def __call__(self, val):
                self.win.geometry("400x200")
                    self.okButton = Button(self.win, width=5, height=1, text="Ok")
                    self.okButton.bind(
                        "<Button-1>", lambda event: self.set_sliders_and_close()
                    )
                    self.okButton.place(relx=0.5, y=175, anchor="center")

                self.rangeLabel = Label(self.win, text=textLabel)
                    self.win.mainloop()

                def construct_label(self):
                    text = f"Enter the range of the {self.nameDict[self.typeButt]} set in sec (e.g. 0-1000)"
                    return text

                        raise ValueError(
                def update_def_values(self, SetData):
                    nameId = f"{self.typeButt}SetId"
                    nameSize = f"size{self.typeButt[0].upper()}{self.typeButt[1:]}Set"
                    firstTS = round(timeToShow[SetData[nameId]], 2)
                    lastId = round(
                        timeToShow[SetData[nameId] + SetData[nameSize] - 1], 2
                    )
                    return [firstTS, lastId]

                def convert_entry_to_id(self):
                    strEntry = self.rangeEntry.get()
                    if len(strEntry) > 0:
                        try:
                            parsedRange = [
                                float(num) for num in list(strEntry.split("-"))
                            ]
                            convertedRange = [
                            ]
                            startId = convertedRange[0]
                            sizeSetinId = convertedRange[1] - convertedRange[0]

                            return startId, sizeSetinId
                        except ValueError:
                            self.okButton.configure(bg="red")
                                "Please enter a valid range in the format 'start-end'"
                            )

                def set_sliders_and_close(self):
                    valuesForSlider = self.convert_entry_to_id()
                    for ivalue, slider in enumerate(self.relevantSliders):
                        slider.set_val(valuesForSlider[ivalue])
                    self.win.destroy()

        ButtlPManual.on_clicked(
                def closestId(self, arr, valToFind):
                    return (np.abs(arr - valToFind)).argmin()

                + "Simply close the window when you are satisfied with your choice."
                rangeButton(
                    typeButt="lossPred",
                    relevantSliders=[sliderLossPredTrain, sliderLossPredTrainSize],
                )
            )
            ButtTestManual.on_clicked(
                rangeButton(
                    typeButt="test", relevantSliders=[sliderTest, sliderTestSize]
                )
            )

                intructions_str = (
                    "Black will become train dataset, red will become test dataset, "
                    + "and orange (if lossPred button is pressed) will become a set "
                    + "to fine-tune predicted loss.\n \n By pressing button with session names "
                    + "you can choose which sessions to include in the analysis. "
                    + "Either use sliders to regulate size and position of test and fine-tuning sets.\n \n"
                    + "USE ZOOM TOOL OF THE FIGURE WINDOW TO AVOID GAPS IF YOU HAVE ANY \n \n"
                    + "Simply close the window when you are satisfied with your choice."
                )

                win = Toplevel()
                win.title("Instructions")
                instLabel = Label(win, text=intructions_str, font=("Helvetica", 16))
                instLabel.pack()
                win.mainloop()

            "Please choose train (black) and test (red) sets. You can add a set (orange) to "
        )
            buttInstructions.on_clicked(buttInstructionsShow)

            suptitle_str = (
                "Please choose train (black) and test (red) sets. You can add a set (orange) to "
                "fine-tune predicted loss (by pressing lossPred button)"
            )
        if IsMultiSessions:
            plt.suptitle(suptitle_str, fontsize=22)
            plt.text(
                x=0.93,
                y=0.82,
                s="X",
                fontsize=36,
                ha="center",
                transform=fig.transFigure,
            )
        else:
            plt.text(
                x=0.93,
                y=0.71,
                s="Y",
                fontsize=36,
                ha="center",
                transform=fig.transFigure,
            )

            if mplt.get_backend() == "QtAgg":
                plt.get_current_fig_manager().window.showMaximized()
            elif mplt.get_backend() == "TkAgg":
                plt.get_current_fig_manager().resize(
                    *plt.get_current_fig_manager().window.maxsize()
                )
                # plt.get_current_fig_manager().window.state('zoomed') # on windows
            plt.show(block=not force)

            if IsMultiSessions:
                trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
                trainEpoch, testEpochs, lossPredSetEpochs = ep.get_epochs(
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
        f.close()

                positionTime[lossPredMask], positions[lossPredMask, 0], c="orange"
        if not force:
            fig, ax = plt.subplots()
            trainMask = ep.inEpochsMask(positionTime, trainEpoch)[:, 0]
            testMask = ep.inEpochsMask(positionTime, testEpochs)[:, 0]
            ax.plot(
                positionTime[trainMask],
                positions[trainMask, 0],
                c="black",
                markersize=6,
            )
            ax.plot(
                positionTime[testMask],
                positions[testMask, 0],
                "--.",
                c="red",
                markersize=6,
            )
