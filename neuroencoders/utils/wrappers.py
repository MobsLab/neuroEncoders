import os
import sys
from typing import Optional, Tuple

import mat73
import numpy as np
import pandas as pd
import pynapple as nts
import scipy.io
import scipy.signal

"""
Wrappers should be able to distinguish between raw data or matlab processed data
"""


def loadSpikeData(
    path: str, index: Optional[int] = None, fs: int = 20000, force: bool = False
) -> Tuple[nts.TsGroup | dict, np.ndarray, dict]:
    """
    if the path contains a folder named /Analysis,
    the script will look into it to load either
            - SpikeData.mat saved from matlab
            - SpikeData.h5 saved from this same script
    if not, the res and clu file will be loaded
    and an /Analysis folder will be created to save the data
    Thus, the next loading of spike times will be faster
    Notes :
            If the frequency is not givne, it's assumed 20kH
    Args:
            path : string
            index : int, optional, the shank index to load

    Returns:
            spikes : TsGroup containing the spike times for each neuron, indexed by the shank and neuron number
            shank : array of shank index for each neuron
            spikedata : dict containing the spikedata.mat file if it exists, None otherwise
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()

    if os.path.exists(path):
        files = os.listdir(path)
        if "SpikeData.mat" in files:
            # using mat73 to load the SpikeData.mat file

            try:
                spikedata = mat73.loadmat(
                    os.path.join(path, "SpikeData.mat"), use_attrdict=True
                )
            except TypeError:
                return loadSpikeData_falllback(path, index, fs)

            shanksPairs = spikedata["TT"]
            shank = 0 * np.ones(len(shanksPairs), dtype=int)
            for i, shank_idx in enumerate(shanksPairs):
                shank[i] = shank_idx[0]
            if index is None:
                shankIndex = 0 * np.ones(len(shanksPairs), dtype=int)
                for i, shank_idx in enumerate(shanksPairs):
                    shankIndex[i] = i
            else:
                shankIndex = np.where(shank == index)[0]
            spikes = {}
            for i in shankIndex:
                # go from 1e-4 seconds to us
                spikes[i] = nts.Ts(spikedata["S"]["C"][i]["t"] * 100, time_units="us")

            a = spikes[0].as_units("s").index.values
            if ((a[-1] - a[0]) / 60.0) / 60.0 > 20.0:  # VERY BAD
                spikes = {}
                for i in shankIndex:
                    spikes[i] = nts.Ts(
                        spikedata["S"][0][0][0][i][0][0][0][1][0][0][2] * 0.0001,
                        time_units="s",
                    )
                raise ValueError(
                    "The SpikeData.mat file seems to be in microseconds, you need to convert to seconds."
                )
            return spikes, shank, spikedata
        elif "SpikeData.h5" in files:
            final_path = os.path.join(path, "SpikeData.h5")
            try:
                spikes = pd.read_hdf(final_path, mode="r")
                # Returning a dictionnary | can be changed to return a dataframe
                toreturn = {}
                for i, j in spikes:
                    toreturn[j] = nts.Ts(
                        t=spikes[(i, j)].replace(0, np.nan).dropna().index.values,
                        time_units="s",
                    )
                shank = spikes.columns.get_level_values(0).values[:, np.newaxis]
                return toreturn, shank
            except (OSError, KeyError):
                spikes = pd.HDFStore(final_path, "r")
                shanks = spikes["/shanks"]
                toreturn = {}
                for j in shanks.index:
                    toreturn[j] = nts.Ts(spikes["/spikes/s" + str(j)])
                shank = shanks.values
                spikes.close()
                del spikes
                return toreturn, shank

        else:
            if not force:
                raise FileNotFoundError(
                    "Couldn't find any SpikeData file in " + path + "; Exiting ..."
                )
            print("Couldn't find any SpikeData file in " + path)
            print(
                "If clu and res files are present in "
                + path
                + ", a SpikeData.h5 is going to be created"
            )

    # Creating /Analysis/ Folder here if not already present
    if not os.path.exists(path):
        os.makedirs(path)
    files = os.listdir(path)
    clu_files = np.sort(
        [f for f in files if "clu" in f and f[0] != "." and f[-2] == "."]
    )
    res_files = np.sort(
        [f for f in files if "res" in f and f[0] != "." and f[-2] == "."]
    )
    clu1 = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2 = np.sort([int(f.split(".")[-1]) for f in res_files])
    if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
        print("Not the same number of clu and res files in " + path + "; Exiting ...")
        sys.exit()
    count = 0
    spikes = []
    basename = clu_files[0].split(".")[0]
    for i, s in zip(range(len(clu_files)), clu1):
        clu = np.genfromtxt(
            os.path.join(path, basename + ".clu." + str(s)), dtype=np.int32
        )[1:]
        if np.max(clu) > 1:
            # print(i,s)
            res = np.genfromtxt(os.path.join(path, basename + ".res." + str(s)))
            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp > 1]
            idx_col = np.arange(count, count + len(idx_clu))
            tmp = pd.DataFrame(
                index=np.unique(res) / fs,
                columns=pd.MultiIndex.from_product([[s], idx_col]),
                data=0,
                dtype=np.uint16,
            )
            for j, k in zip(idx_clu, idx_col):
                tmp.loc[res[clu == j] / fs, (s, k)] = np.uint16(k + 1)
            spikes.append(tmp)
            count += len(idx_clu)

            # tmp2 = pd.DataFrame(index=res[clu==j]/fs, data = k+1, ))
            # spikes = pd.concat([spikes, tmp2], axis = 1)

    # Returning a dictionnary
    toreturn = {}
    shank = []
    for s in spikes:
        shank.append(s.columns.get_level_values(0).values)
        np.unique(shank[-1])[0]
        for i, j in s:
            toreturn[j] = nts.Ts(
                t=s[(i, j)].replace(0, np.nan).dropna().index.values, time_units="s"
            )

    del spikes
    shank = np.hstack(shank)

    final_path = os.path.join(path, "SpikeData.h5")
    store = pd.HDFStore(final_path)
    for s in toreturn.keys():
        store.put("spikes/s" + str(s), toreturn[s].as_series())
    store.put("shanks", pd.Series(index=list(toreturn.keys()), data=shank))
    store.close()

    # OLD WAY
    # spikes = pd.concat(spikes, axis = 1)
    # spikes = spikes.fillna(0)
    # spikes = spikes.astype(np.uint16)

    # Saving SpikeData.h5
    # final_path = os.path.join(path, 'SpikeData.h5')
    # spikes.columns.set_names(['shank', 'neuron'], inplace=True)
    # spikes.to_hdf(final_path, key='spikes', mode='w')

    # Returning a dictionnary
    # toreturn = {}
    # for i,j in spikes:
    # 	toreturn[j] = nts.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

    # shank = spikes.columns.get_level_values(0).values[:,np.newaxis].flatten()

    return toreturn, shank


def loadSpikeData_falllback(path: str, index: Optional = None, fs: int = 20000):
    spikedata = scipy.io.loadmat(path + "SpikeData.mat")
    shanksPairs = spikedata["TT"].flatten()
    shanksPairs = np.array([s.flatten() for s in shanksPairs])
    shank = 0 * np.ones(len(shanksPairs), dtype=int)
    for i, shank_idx in enumerate(shanksPairs):
        shank[i] = shank_idx[0]

    if index is None:
        shankIndex = 0 * np.ones(len(shanksPairs), dtype=int)
        for i, shank_idx in enumerate(shanksPairs):
            shankIndex[i] = i
    else:
        shankIndex = np.where(shank == index)[0]

    spikes = {}
    for i in shankIndex:
        to_add = spikedata["S"]["C"][0][0][0][i][0][0][2].flatten()
        if len(to_add) == 1:
            to_add = spikedata["S"]["C"][0][0][0][i][0][0][0][1][0][0][2].flatten()

        spikes[i] = nts.Ts(to_add * 100, time_units="us")
    a = spikes[0].as_units("s").index.values
    if ((a[-1] - a[0]) / 60.0) / 60.0 > 20.0:  # VERY BAD
        spikes = {}
        for i in shankIndex:
            to_add = spikedata["S"]["C"][0][0][0][i][0][0][2].flatten()
            if len(to_add) == 1:
                to_add = spikedata["S"]["C"][0][0][0][i][0][0][0][1][0][0][2].flatten()

            spikes[i] = nts.Ts(
                to_add * 0.0001,
                time_units="s",
            )
            raise ValueError(
                "The SpikeData.mat file seems to be in microseconds, you need to convert to seconds."
            )
    return spikes, shank, spikedata


def loadXML(path):
    """
    path should be the folder session containing the XML file
    Function returns :
            1. the number of channels
            2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
                    eeg file first if both are present or both are absent
            3. the mappings shanks to channels as a dict
    Args:
            path : string

    Returns:
            int, int, dict
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith(".xml")]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    path = os.path.join(path, xmlfiles[0])

    from xml.dom import minidom

    xmldoc = minidom.parse(path)
    nChannels = (
        xmldoc.getElementsByTagName("acquisitionSystem")[0]
        .getElementsByTagName("nChannels")[0]
        .firstChild.data
    )
    fs_dat = (
        xmldoc.getElementsByTagName("acquisitionSystem")[0]
        .getElementsByTagName("samplingRate")[0]
        .firstChild.data
    )
    fs_eeg = (
        xmldoc.getElementsByTagName("fieldPotentials")[0]
        .getElementsByTagName("lfpSamplingRate")[0]
        .firstChild.data
    )
    if os.path.splitext(xmlfiles[0])[0] + ".dat" in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] + ".eeg" in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    groups = (
        xmldoc.getElementsByTagName("anatomicalDescription")[0]
        .getElementsByTagName("channelGroups")[0]
        .getElementsByTagName("group")
    )
    for i in range(len(groups)):
        shank_to_channel[i] = np.sort(
            [
                int(child.firstChild.data)
                for child in groups[i].getElementsByTagName("channel")
            ]
        )
    return int(nChannels), int(fs), shank_to_channel


def downsampleDatFile(path, n_channels, fs):
    """
    downsample .dat file to .eeg 1/16 (20000 -> 1250 Hz)

    Since .dat file can be very big, the strategy is to load one channel at the time,
    downsample it, and free the memory.

    Args:
            path: string
            n_channel: int
            fs: int
    Return:
            none
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    datfile = [f for f in listdir if f.endswith(".dat")]
    if not len(datfile):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    path = os.path.join(path, datfile[0])

    f = open(path, "rb")
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2
    n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
    n_samples / fs
    f.close()

    chunksize = 100000
    eeg = np.zeros((int(n_samples / 16), n_channels))

    for n in range(n_channels):
        # Loading
        rawchannel = np.zeros(n_samples, np.int16)
        count = 0
        while count < n_samples:
            f = open(path, "rb")
            seekstart = count * n_channels * bytes_size
            f.seek(seekstart)
            block = np.fromfile(
                f, np.int16, n_channels * np.minimum(chunksize, n_samples - count)
            )
            f.close()
            block = block.reshape(np.minimum(chunksize, n_samples - count), n_channels)
            rawchannel[count : count + np.minimum(chunksize, n_samples - count)] = (
                np.copy(block[:, n])
            )
            count += chunksize
        # Downsampling
        eeg[:, n] = scipy.signal.resample_poly(rawchannel, 1, 16)
        del rawchannel

    # Saving
    eeg_path = os.path.join(path, os.path.splitext(datfile[0])[0] + ".eeg")
    with open(eeg_path, "wb") as f:
        eeg.astype("int16").tofile(f)

    return


def makeEpochs(path, order, file=None, start=None, end=None, time_units="s"):
    """
    The pre-processing pipeline should spit out a csv file containing all the successive epoch of sleep/wake
    This function will load the csv and write IntervalSet of wake and sleep in /Analysis/BehavEpochs.h5
    If no csv exists, it's still possible to give by hand the start and end of the epochs
    Notes:
            The function assumes no header on the csv file
    Args:
            path: string
            order: list
            file: string
            start: list/array (optional)
            end: list/array (optional)
            time_units: string (optional)
    Return:
            none
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    if file:
        listdir = os.listdir(path)
        if file not in listdir:
            print("The file " + file + " cannot be found in the path " + path)
            sys.exit()
        filepath = os.path.join(path, file)
        epochs = pd.read_csv(filepath, header=None)
    elif file is None and len(start) and len(end):
        epochs = pd.DataFrame(np.vstack((start, end)).T)
    elif file is None and start is None and end is None:
        print(
            "You have to specify either a file or arrays of start and end; Exiting ..."
        )
        sys.exit()

    # Creating /Analysis/ Folder here if not already present
    path = os.path.join(path, "Analysis/")
    if not os.path.exists(path):
        os.makedirs(path)
    # Writing to BehavEpochs.h5
    new_file = os.path.join(path, "BehavEpochs.h5")
    store = pd.HDFStore(new_file, "a")
    epoch = np.unique(order)
    for i, n in enumerate(epoch):
        idx = np.where(np.array(order) == n)[0]
        ep = nts.IntervalSet(
            start=epochs.loc[idx, 0], end=epochs.loc[idx, 1], time_units=time_units
        )
        store[n] = pd.DataFrame(ep)
    store.close()

    return None


def makePositions(
    path,
    file_order,
    episodes,
    n_ttl_channels=1,
    optitrack_ch=None,
    names=["ry", "rx", "rz", "x", "y", "z"],
    update_wake_epoch=True,
):
    """
    Assuming that makeEpochs has been runned and a file BehavEpochs.h5 can be
    found in /Analysis/, this function will look into path  for analogin file
    containing the TTL pulses. The position time for all events will thus be
    updated and saved in Analysis/Position.h5.
    BehavEpochs.h5 will although be updated to match the time between optitrack
    and intan

    Notes:
            The function assumes headers on the csv file of the position in the following order:
                    ['ry', 'rx', 'rz', 'x', 'y', 'z']
    Args:
            path: string
            file_order: list
            names: list
    Return:
            None
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    files = os.listdir(path)
    for file in file_order:
        if not np.any([file + ".csv" in g for g in files]):
            print("Could not find " + file + ".csv; Exiting ...")
            sys.exit()
    path = os.path.join(path, "Analysis/")
    if not os.path.exists(path):
        os.makedirs(path)
    file_epoch = os.path.join(path, "Analysis", "BehavEpochs.h5")
    if os.path.exists(file_epoch):
        wake_ep = loadEpoch(path, "wake")
    else:
        makeEpochs(path, episodes, file="Epoch_TS.csv")
        wake_ep = loadEpoch(path, "wake")
    if len(wake_ep) != len(file_order):
        print("Number of wake episodes doesn't match; Exiting...")
        sys.exit()

    frames = []

    for i, file in enumerate(file_order):
        csv_file = os.path.join(path, "".join(s for s in files if file + ".csv" in s))
        position = pd.read_csv(csv_file, header=[4, 5], index_col=1)
        if 1 in position.columns:
            position = position.drop(labels=1, axis=1)
        position = position[~position.index.duplicated(keep="first")]
        analogin_file = os.path.splitext(csv_file)[0] + "_analogin.dat"
        if os.path.split(analogin_file)[1] not in files:
            print("No analogin.dat file found.")
            print("Please provide it as " + os.path.split(analogin_file)[1])
            print("Exiting ...")
            sys.exit()
        else:
            ttl = loadTTLPulse(analogin_file, n_ttl_channels, optitrack_ch)

        length = np.minimum(len(ttl), len(position))
        ttl = ttl.iloc[0:length]
        position = position.iloc[0:length]
        time_offset = wake_ep.as_units("s").iloc[i, 0] + ttl.index[0]
        position.index += time_offset
        wake_ep.iloc[i, 0] = np.int64(
            np.maximum(wake_ep.as_units("s").iloc[i, 0], position.index[0]) * 1e6
        )
        wake_ep.iloc[i, 1] = np.int64(
            np.minimum(wake_ep.as_units("s").iloc[i, 1], position.index[-1]) * 1e6
        )

        frames.append(position)

    position = pd.concat(frames)
    # position = nts.TsdFrame(t = position.index.values, d = position.values, time_units = 's', columns = names)
    position.columns = names
    position[["ry", "rx", "rz"]] *= np.pi / 180
    position[["ry", "rx", "rz"]] += 2 * np.pi
    position[["ry", "rx", "rz"]] %= 2 * np.pi

    if update_wake_epoch:
        store = pd.HDFStore(file_epoch, "a")
        store["wake"] = pd.DataFrame(wake_ep)
        store.close()

    position_file = os.path.join(path, "Analysis", "Position.h5")
    store = pd.HDFStore(position_file, "w")
    store["position"] = position
    store.close()

    return


def loadEpoch(path, epoch, episodes=None):
    """
    load the epoch contained in path
    If the path contains a folder analysis, the function will load either the BehavEpochs.mat or the BehavEpochs.h5
    Run makeEpochs(data_directory, ['sleep', 'wake', 'sleep', 'wake'], file='Epoch_TS.csv') to create the BehavEpochs.h5

    Args:
            path: string
            epoch: string

    Returns:
            pynapple.IntervalSet
    """
    if not os.path.exists(path):  # Check for path
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    filepath = os.path.join(path, "Analysis")
    if os.path.exists(filepath):  # Check for path/Analysis/
        listdir = os.listdir(filepath)
        file = [f for f in listdir if "BehavEpochs" in f]
    if len(file) == 0:  # Running makeEpochs
        makeEpochs(path, episodes, file="Epoch_TS.csv")
        listdir = os.listdir(filepath)
        file = [f for f in listdir if "BehavEpochs" in f]
    if file[0] == "BehavEpochs.h5":
        new_file = os.path.join(filepath, "BehavEpochs.h5")
        store = pd.HDFStore(new_file, "r")
        if "/" + epoch in store.keys():
            ep = store[epoch]
            store.close()
            return nts.IntervalSet(ep)
        else:
            print(
                "The file BehavEpochs.h5 does not contain the key "
                + epoch
                + "; Exiting ..."
            )
            sys.exit()
    elif file[0] == "BehavEpochs.mat":
        behepochs = scipy.io.loadmat(os.path.join(filepath, file[0]))
        if epoch == "wake":
            wake_ep = np.hstack(
                [behepochs["wakeEp"][0][0][1], behepochs["wakeEp"][0][0][2]]
            )
            return nts.IntervalSet(
                wake_ep[:, 0], wake_ep[:, 1], time_units="s"
            ).drop_short_intervals(0.0)
        elif epoch == "sleep":
            sleep_pre_ep, sleep_post_ep = [], []
            if "sleepPreEp" in behepochs.keys():
                sleep_pre_ep = behepochs["sleepPreEp"][0][0]
                sleep_pre_ep = np.hstack([sleep_pre_ep[1], sleep_pre_ep[2]])
                behepochs["sleepPreEpIx"][0]
            if "sleepPostEp" in behepochs.keys():
                sleep_post_ep = behepochs["sleepPostEp"][0][0]
                sleep_post_ep = np.hstack([sleep_post_ep[1], sleep_post_ep[2]])
                behepochs["sleepPostEpIx"][0]
            if len(sleep_pre_ep) and len(sleep_post_ep):
                sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
            elif len(sleep_pre_ep):
                sleep_ep = sleep_pre_ep
            elif len(sleep_post_ep):
                sleep_ep = sleep_post_ep
            return nts.IntervalSet(sleep_ep[:, 0], sleep_ep[:, 1], time_units="s")
        ###################################
        # WORKS ONLY FOR MATLAB FROM HERE #
        ###################################
        elif epoch == "sws":
            sampling_freq = 1250
            new_listdir = os.listdir(path)
            for file in new_listdir:
                if "sts.SWS" in file:
                    sws = np.genfromtxt(os.path.join(path, file)) / float(sampling_freq)
                    return nts.IntervalSet.drop_short_intervals(
                        nts.IntervalSet(sws[:, 0], sws[:, 1], time_units="s"), 0.0
                    )

                elif "-states.mat" in file:
                    sws = scipy.io.loadmat(os.path.join(path, file))["states"][0]
                    index = np.logical_or(sws == 2, sws == 3) * 1.0
                    index = index[1:] - index[0:-1]
                    start = np.where(index == 1)[0] + 1
                    stop = np.where(index == -1)[0]
                    return nts.IntervalSet.drop_short_intervals(
                        nts.IntervalSet(start, stop, time_units="s", expect_fix=True),
                        0.0,
                    )

        elif epoch == "rem":
            sampling_freq = 1250
            new_listdir = os.listdir(path)
            for file in new_listdir:
                if "sts.REM" in file:
                    rem = np.genfromtxt(os.path.join(path, file)) / float(sampling_freq)
                    return nts.IntervalSet(
                        rem[:, 0], rem[:, 1], time_units="s"
                    ).drop_short_intervals(0.0)

                elif "-states/m" in listdir:
                    rem = scipy.io.loadmat(path + file)["states"][0]
                    index = (rem == 5) * 1.0
                    index = index[1:] - index[0:-1]
                    start = np.where(index == 1)[0] + 1
                    stop = np.where(index == -1)[0]
                    return nts.IntervalSet(
                        start,
                        stop,
                        time_units="s",
                    ).drop_short_intervals(0.0)


def loadPosition(
    path,
    events=None,
    episodes=None,
    n_ttl_channels=1,
    optitrack_ch=None,
    names=["ry", "rx", "rz", "x", "y", "z"],
    update_wake_epoch=True,
):
    """
    load the position contained in /Analysis/Position.h5

    Notes:
            The order of the columns is assumed to be
                    ['ry', 'rx', 'rz', 'x', 'y', 'z']
    Args:
            path: string

    Returns:
            pynapple.TsdFrame
    """
    if not os.path.exists(path):  # Checking for path
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    path = os.path.join(path, "Analysis")
    if not os.path.exists(path):
        os.mkdir(path)
    file = os.path.join(path, "Analysis", "Position.h5")
    if not os.path.exists(file):
        makePositions(
            path,
            events,
            episodes,
            n_ttl_channels,
            optitrack_ch,
            names,
            update_wake_epoch,
        )
    if os.path.exists(file):
        store = pd.HDFStore(file, "r")
        position = store["position"]
        store.close()
        position = nts.TsdFrame(
            t=position.index.values,
            d=position.values,
            columns=position.columns,
            time_units="s",
        )
        return position
    else:
        print("Cannot find " + file + " for loading position")
        sys.exit()


def loadTTLPulse(file, n_ttl_channels=1, optitrack_ch=None, fs=20000):
    """
    load ttl from analogin.dat
    """
    f = open(file, "rb")
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2
    n_samples = int((endoffile - startoffile) / n_ttl_channels / bytes_size)
    f.close()
    with open(file, "rb") as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_ttl_channels))
    if optitrack_ch:
        data = data[:, optitrack_ch].astype(np.int32)
    else:
        data = data.flatten().astype(np.int32)

    peaks, _ = scipy.signal.find_peaks(np.diff(data), height=30000)
    timestep = np.arange(0, len(data)) / fs
    # analogin = pd.Series(index = timestep, data = data)
    peaks += 1
    ttl = pd.Series(index=timestep[peaks], data=data[peaks])
    return ttl


def loadAuxiliary(path, fs=20000):
    """
    Extract the acceleration from the auxiliary.dat for each epochs

    Args:
            path: string
            epochs_ids: list
    Return:
            TsdArray
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    if "Acceleration.h5" in os.listdir(os.path.join(path, "Analysis")):
        accel_file = os.path.join(path, "Analysis", "Acceleration.h5")
        store = pd.HDFStore(accel_file, "r")
        accel = store["acceleration"]
        store.close()
        return accel
    else:
        aux_files = np.sort([f for f in os.listdir(path) if "auxiliary" in f])
        if len(aux_files) == 0:
            print("Could not find any file in" + path + "_auxiliary.dat; Exiting ...")
            sys.exit()
        accel = []
        sample_size = []
        for i, f in enumerate(aux_files):
            path = os.path.join(path, f)
            f = open(path, "rb")
            startoffile = f.seek(0, 0)
            endoffile = f.seek(0, 2)
            bytes_size = 2
            n_samples = int((endoffile - startoffile) / 3 / bytes_size)
            n_samples / fs
            f.close()
            tmp = np.fromfile(open(path, "rb"), np.uint16).reshape(n_samples, 3)
            accel.append(tmp)
            sample_size.append(n_samples)

        accel = np.concatenate(accel)
        factor = 37.4e-6
        # timestep = np.arange(0, len(accel))/fs
        # accel = pd.DataFrame(index = timestep, data= accel*37.4e-6)
        tmp = scipy.signal.resample_poly(accel * factor, 1, 16)
        timestep = np.arange(0, len(tmp)) / (fs / 16)
        tmp = pd.DataFrame(index=timestep, data=tmp)
        accel_file = os.path.join(path, "Analysis", "Acceleration.h5")
        store = pd.HDFStore(accel_file, "w")
        store["acceleration"] = tmp
        store.close()
        return tmp


def clean_mat_structure(element):
    """
    Recursively unpacks nested numpy structured arrays into clean Python dicts/lists.
    """
    # 1. Handle NumPy structured arrays (they have dtype fields)
    if isinstance(element, np.ndarray) and element.dtype.names is not None:
        # If it's an array of structs, we usually just want the first record
        # or a list of dicts if it has multiple entries.
        if element.size == 1:
            record = element[0]
            return {
                name: clean_mat_structure(record[name]) for name in element.dtype.names
            }
        else:
            return [clean_mat_structure(item) for item in element]

    # 2. Handle standard NumPy arrays (unwrap dimensions)
    if isinstance(element, np.ndarray):
        # If it's empty, return None or empty list
        if element.size == 0:
            return None
        # If it's a single element array (e.g., array([[1199]])), unwrap it
        if element.size == 1:
            return clean_mat_structure(element.item())
        # If it's a 1D or multi-D array/list of data (like your folder paths)
        return [clean_mat_structure(x) for x in element.flatten()]

    # 3. Clean up specific data types (bytes, numpy scalars)
    if isinstance(element, (bytes, str)):
        # Decode bytes to string if necessary, strip whitespace
        val = element.decode("utf-8") if isinstance(element, bytes) else element
        return val.strip()

    if isinstance(element, np.generic):
        # Convert numpy types (uint16, uint8, etc.) to standard Python types
        return element.item()

    return element


def loadRespiData(path):
    """
    Extract the respiration data from the respiration.dat for each epochs

    Args:
    path: string

    Returns:
    Respiration times,
    Respiration values
    """
    if not os.path.exists(path):
        if os.path.isdir(
            os.path.join(os.path.dirname(path), "LFPData")
        ) and os.path.isdir(os.path.join(os.path.dirname(path), "ChannelsToAnalyse")):
            try:
                print(
                    "Could not find respiration data at "
                    + path
                    + "; Attempting to reconstruct from LFPData and ChannelsToAnalyse through matlab..."
                )
                compute_spectro_from_matlab(os.path.dirname(path))
            except Exception as e:
                print(
                    "Could not compute spectro from matlab for respiration data. Error: ",
                    e,
                )
        raise FileNotFoundError(f"The path {path} doesn't exist; Exiting ...")

    if not os.path.isfile(path):
        path = os.path.join(os.path.dirname(path), "Bulb_deep_low_Spectrum.mat")

    try:
        from scipy.io import loadmat

        loaded_file = loadmat(path)
        spectro = clean_mat_structure(loaded_file["Spectro"])
    except NotImplementedError:
        from mat73 import loadmat

        loaded_file = loadmat(path)
        spectro = loaded_file["Spectro"]

    full_spectro, time, freqs = spectro

    return full_spectro, time, freqs


def compute_spectro_from_matlab(path):
    """
    Launches MATLAB, navigates to the target folder, loads the specified
    channel file, and runs the LowSpectrum_AD analysis.

    Parameters:
    -----------
    path : str
        The absolute path to the directory where the MATLAB files are located.
    """
    print("Importing MATLAB engine...")
    import matlab.engine

    # Clean and normalize the path for the operating system
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"The folder path '{path}' does not exist.")

    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()

    try:
        eng.addtopath(nargout=0)
        eng.cd(path, nargout=0)

        print(f"Changed MATLAB directory to: {path}")

        # 3. Load the specific .mat file
        # MATLAB equivalent: load('ChannelsToAnalyse/Bulb_deep.mat');
        mat_file_path = os.path.join(path, "ChannelsToAnalyse", "Bulb_deep.mat")
        if not os.path.isfile(mat_file_path):
            mat_file_path = os.path.join(path, "ChannelsToAnalyse", "B.mat")

        if not os.path.isfile(mat_file_path):
            raise FileNotFoundError(
                f"Could not find the .mat file at '{mat_file_path}'."
            )

        eng.load(mat_file_path, nargout=0)
        print("Loaded 'ChannelsToAnalyse/Bulb_deep.mat'.")

        # 4. Fetch the 'channel' variable from the MATLAB workspace
        # (Assuming 'channel' is inside Bulb_deep.mat)
        channel_var = eng.workspace["channel"]

        # 5. Execute the analysis function
        # MATLAB: LowSpectrum_AD([cd filesep], channel, 'Bulb_deep');
        # In python, eng.cd() returns the current path string
        current_dir_with_sep = eng.cd() + os.sep

        print("Running LowSpectrum_AD...")
        eng.LowSpectrum_AD(current_dir_with_sep, channel_var, "Bulb_deep", nargout=0)
        print("Analysis completed successfully!")

    except matlab.engine.MatlabExecutionError as e:
        print(f"MATLAB Error encountered:\n{e}")
    finally:
        # Always close the engine connection to free up memory/licenses
        eng.quit()
        print("MATLAB engine closed.")


##########################################################################################################
# TODO
##########################################################################################################


def loadShankStructure(generalinfo):
    """
    load Shank Structure from dictionnary
    Only useful for matlab now
    Note :
            TODO for raw data.

    Args:
            generalinfo : dict

    Returns: dict
    """
    shankStructure = {}
    for k, i in zip(
        generalinfo["shankStructure"][0][0][0][0],
        range(len(generalinfo["shankStructure"][0][0][0][0])),
    ):
        if len(generalinfo["shankStructure"][0][0][1][0][i]):
            shankStructure[k[0]] = generalinfo["shankStructure"][0][0][1][0][i][0] - 1
        else:
            shankStructure[k[0]] = []

    return shankStructure


def loadShankMapping(path):
    spikedata = scipy.io.loadmat(path)
    shank = spikedata["shank"]
    return shank


def loadHDCellInfo(path, index):
    """
    load the session_id_HDCells.mat file that contains the index of the HD neurons
    Only useful for matlab now
    Note :
            TODO for raw data.

    Args:
            generalinfo : string, array

    Returns:
            array
    """
    # units shoud be the value to convert in s
    import scipy.io

    hd_info = scipy.io.loadmat(path)["hdCellStats"][:, -1]
    return np.where(hd_info[index])[0]


def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision="int16"):
    import pynapple as nts

    if type(channel) is not list:
        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        n_samples / frequency
        1 / frequency
        f.close()
        with open(path, "rb") as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:, channel]
        timestep = np.arange(0, len(data)) / frequency
        return nts.Tsd(timestep, data, time_units="s")
    elif type(channel) is list:
        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2

        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        n_samples / frequency
        f.close()
        with open(path, "rb") as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:, channel]
        timestep = np.arange(0, len(data)) / frequency
        return nts.TsdFrame(timestep, data, time_units="s")


def loadBunch_Of_LFP(
    path, start, stop, n_channels=90, channel=64, frequency=1250.0, precision="int16"
):
    import pynapple as nts

    bytes_size = 2
    start_index = int(start * frequency * n_channels * bytes_size)
    stop_index = int(stop * frequency * n_channels * bytes_size)
    fp = np.memmap(
        path, np.int16, "r", start_index, shape=(stop_index - start_index) // bytes_size
    )
    data = np.array(fp).reshape(len(fp) // n_channels, n_channels)

    if type(channel) is not list:
        timestep = np.arange(0, len(data)) / frequency
        return nts.Tsd(timestep, data[:, channel], time_units="s")
    elif type(channel) is list:
        timestep = np.arange(0, len(data)) / frequency
        return nts.TsdFrame(timestep, data[:, channel], time_units="s")


def compute_matrix_correlation(matrix_A, matrix_B):
    # Center the rows (subtract the mean of each neuron's tuning curve)
    A_centered = matrix_A - matrix_A.mean(axis=1, keepdims=True)
    B_centered = matrix_B - matrix_B.mean(axis=1, keepdims=True)

    # Compute the dot product along the bins, normalized by the norms
    numerator = np.sum(A_centered * B_centered, axis=1)
    denominator = np.linalg.norm(A_centered, axis=1) * np.linalg.norm(
        B_centered, axis=1
    )

    # Avoid division by zero for silent/dead neurons
    with np.errstate(divide="ignore", invalid="ignore"):
        neuron_correlations_fast = numerator / denominator
        neuron_correlations_fast[~np.isfinite(neuron_correlations_fast)] = 0.0

    return neuron_correlations_fast


def get_raw_pv_corr(M1, M2):
    # Center population vectors across columns (axis=0)
    m1_c = M1 - M1.mean(axis=0, keepdims=True)
    m2_c = M2 - M2.mean(axis=0, keepdims=True)
    num = np.sum(m1_c * m2_c, axis=0)
    denom = np.linalg.norm(m1_c, axis=0) * np.linalg.norm(m2_c, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(num / denom, nan=0.0)


def compute_debiased_pv_corr(A_full, B_full, A_odd, A_even, B_odd, B_even):
    """
    Computes true debiased population vector correlation across spatial bins (columns).
    Inputs shape: (n_neurons, n_bins)
    """

    rho_AB = get_raw_pv_corr(A_full, B_full)
    rho_AA = get_raw_pv_corr(A_odd, A_even)
    rho_BB = get_raw_pv_corr(B_odd, B_even)

    reliability_term = np.sqrt(np.clip(rho_AA * rho_BB, 1e-6, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected_pv_corr = rho_AB / reliability_term

    return np.clip(corrected_pv_corr, -1.0, 1.0)
