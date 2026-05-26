# Load libs
import glob
import os
import sys

import numpy as np
import tqdm as tqdm

from neuroencoders.importData import rawdata_parser
from neuroencoders.simpleBayes import butils
from neuroencoders.utils.backend import pd
from neuroencoders.utils.global_classes import Project


def getSpikesfromClu(
    projectPath: Project, behavior_data: dict, cluster_modifier=1, savedata=True
) -> dict[str, list]:
    """
    Get spikes from clu files and save them in a dictionary.

    Parameters
    ----------
    projectPath : Project object
    behavior_data : dictionary w Positions, positionTime, Speed, Bandwidth, Times
    cluster_modifier : int, optional
        The default is 1.
    savedata : bool, optional
        The default is True.

    Returns
    -------
    cluster_data : dictionary with Spike_labels, Spike_times, Spike_positions, Spike_speed
    """
    # Get parameters
    listChannels, samplingRate, _ = rawdata_parser.get_params(projectPath.xml)
    # Allocate
    labels = []
    spikeTime = []
    spikePositions = []
    spikeSpeed = []
    spikePosIndex = []

    nTetrodes = len(listChannels)
    for tetrode in tqdm.tqdm(range(nTetrodes)):
        print(
            f"Importing sorted spikes from Neuroscope files from electrodes group #{tetrode}"
        )
        if os.path.isfile(projectPath.clu(tetrode)):
            with (
                open(projectPath.clu(tetrode), "r") as fClu,
                open(projectPath.res(tetrode), "r") as fRes,
            ):  # open(projectPath.spk(tetrode), 'rb') as fSpk
                cluStr = fClu.readlines()
                resStr = fRes.readlines()
                nClu = int(cluStr[0]) - 1
                # Clusters only with labels >= 1
                # otherwise all labels are set to 0 (represent cluster 0 - NOISE)
                labels_temp = butils.modify_labels(
                    np.array(
                        [
                            [
                                1.0 if int(cluStr[n + 1]) == cluster_idx else 0.0
                                for cluster_idx in range(1, nClu + 1)
                            ]
                            for n in range(len(cluStr) - 1)
                        ]
                    ),
                    cluster_modifier,
                )
                st = np.array(
                    [
                        [float(resStr[n]) / samplingRate]
                        for n in tqdm.tqdm(range(len(cluStr) - 1))
                    ]
                )

                # Efficient way to get the closest position_time to each spike time:
                lastBestId = 0
                posID = []
                for n in tqdm.tqdm(range(len(st))):
                    lastBestId = rawdata_parser.findTime(
                        behavior_data["positionTime"], lastBestId, st[n]
                    )
                    posID += [lastBestId]

                sp = behavior_data["Positions"][posID]
                newposId = np.array(posID)
                newposId[
                    np.where(np.array(posID) > len(behavior_data["Speed"]) - 1)[0]
                ] = len(behavior_data["Speed"]) - 1
                ss = behavior_data["Speed"][newposId, :]

                spikeTime.append(st)
                spikePositions.append(sp)
                spikePosIndex.append(np.array(posID))
                spikeSpeed.append(ss)
                labels.append(labels_temp)
        else:
            print("File " + projectPath.clu(tetrode) + " not found.")
            continue
        sys.stdout.write("File from tetrode " + " has been successfully opened. ")
        sys.stdout.write("Processing ...")
        sys.stdout.write("\r")
        sys.stdout.flush()

        sys.stdout.write(
            "We have finished building rates for group "
            + str(tetrode + 1)
            + ", loading next                           "
        )
        sys.stdout.write("\r")
        sys.stdout.flush()
    sys.stdout.write(
        "We have imported clusters.                                                           "
    )
    sys.stdout.write("\r")
    sys.stdout.flush()

    cluster_data = {
        "Spike_labels": labels,
        "Spike_times": spikeTime,
        "Spike_positions": spikePositions,
        "Spike_speed": spikeSpeed,
    }
    if savedata:
        cluster_save_path = os.path.join(projectPath.folder, "dataset", "clusterData")
        if not os.path.isdir(cluster_save_path):
            os.makedirs(cluster_save_path)
        for shank in range(len(labels)):
            df = pd.DataFrame(labels[shank])
            df.to_csv(
                os.path.join(cluster_save_path, "Spike_labels" + str(shank) + ".csv")
            )
            df = pd.DataFrame(spikeTime[shank])
            df.to_csv(
                os.path.join(cluster_save_path, "spike_time" + str(shank) + ".csv")
            )
            df = pd.DataFrame(spikePositions[shank])
            df.to_csv(
                os.path.join(cluster_save_path, "spike_positions" + str(shank) + ".csv")
            )
            df = pd.DataFrame(spikePosIndex[shank])
            df.to_csv(
                os.path.join(cluster_save_path, "spike_pos_index" + str(shank) + ".csv")
            )
            df = pd.DataFrame(spikeSpeed[shank])
            df.to_csv(
                os.path.join(cluster_save_path, "spike_speed" + str(shank) + ".csv")
            )
    return cluster_data


def _load_linear_spike_sorting_from_clu(projectPath: Project, flatten=True) -> dict:
    """
    Load spike sorting data from the original klustakwik files and linearize them in time.

    Parameters
    ----------
    projectPath : Project object

    Returns
    -------
    cluster_data : dict
    """
    # Get parameters
    listChannels, samplingRate, _ = rawdata_parser.get_params(projectPath.xml)
    # Allocate
    labels = []
    indexInDat = []

    nTetrodes = len(listChannels)
    last_n_clu = 0
    for tetrode in tqdm.tqdm(range(nTetrodes)):
        print(
            f"Importing sorted spikes from Neuroscope files from electrodes group #{tetrode}"
        )
        if os.path.isfile(projectPath.clu(tetrode)):
            with (
                open(projectPath.clu(tetrode), "r") as fClu,
                open(projectPath.res(tetrode), "r") as fRes,
            ):  # open(projectPath.spk(tetrode), 'rb') as fSpk
                cluStr = fClu.readlines()
                resStr = fRes.readlines()
                clu = np.array(
                    [int(cluStr[n + 1]) + last_n_clu for n in range(len(cluStr) - 1)]
                )
                # Clusters only with labels >= 1
                labels_mask = clu >= 1 + last_n_clu
                labels_temp = clu[labels_mask]

                # turn spike times from str to float
                index = np.array([int(x.strip()) for x in resStr])
                index_temp = index[labels_mask]

                indexInDat.append(index_temp)
                labels.append(labels_temp)
                last_n_clu += int(cluStr[0]) - 1

        else:
            print("File " + projectPath.clu(tetrode) + " not found.")
            continue
        sys.stdout.write("File from tetrode " + " has been successfully opened. ")
        sys.stdout.write("Processing ...")
        sys.stdout.write("\r")
        sys.stdout.flush()

    if flatten:
        # Now sort all spikes in time
        all_index = np.concatenate(indexInDat)
        sort_idx = np.argsort(all_index)
        all_labels = np.concatenate(labels)
        labels = all_labels[sort_idx]
        indexInDat = all_index[sort_idx]
        if not os.path.isdir(
            os.path.join(projectPath.folder, "dataset", "clusterData")
        ):
            os.makedirs(os.path.join(projectPath.folder, "dataset", "clusterData"))

    sys.stdout.write(
        "We have imported linear-time clusters.                                                           "
    )
    sys.stdout.write("\r")
    sys.stdout.flush()

    cluster_data = {
        "Spike_labels": labels,
        "Spike_index": indexInDat,
    }
    return cluster_data


def load_spike_sorting(projectPath: Project, phase=None) -> dict:
    """
    Load spike sorting data from the dataset/clusterData folder if the files are present, otherwise
    run the spike sorting algorithm and save the data.

    Parameters
    ----------
    projectPath : Project object

    Returns
    -------
    cluster_data : dict
    """
    cluster_save_path = os.path.join(projectPath.folder, "dataset", "clusterData")
    if os.path.isfile(os.path.join(cluster_save_path, "Spike_labels0.csv")):
        lfiles = glob.glob(os.path.join(cluster_save_path, "Spike_labels*.csv"))
        num_files = len(lfiles)
        cluster_data = {
            "Spike_labels": [],
            "Spike_times": [],
            "Spike_positions": [],
            "Spike_speed": [],
            "Spike_pos_index": [],
        }
        print("Reading saved cluster csv file")
        for shank in tqdm.tqdm(range(num_files)):
            df = pd.read_csv(
                os.path.join(cluster_save_path, "Spike_labels" + str(shank) + ".csv")
            )
            cluster_data["Spike_labels"].append(df.values[:, 1:])
            df = pd.read_csv(
                os.path.join(cluster_save_path, "spike_time" + str(shank) + ".csv")
            )
            cluster_data["Spike_times"].append(df.values[:, 1:])
            df = pd.read_csv(
                os.path.join(cluster_save_path, "spike_positions" + str(shank) + ".csv")
            )
            cluster_data["Spike_positions"].append(df.values[:, 1:])
            df = pd.read_csv(
                os.path.join(cluster_save_path, "spike_pos_index" + str(shank) + ".csv")
            )
            cluster_data["Spike_pos_index"].append(df.values[:, 1:])
            df = pd.read_csv(
                os.path.join(cluster_save_path, "spike_speed" + str(shank) + ".csv")
            )
            cluster_data["Spike_speed"].append(df.values[:, 1:])

        print("finished reading")
    else:
        behavior_data = rawdata_parser.get_behavior(
            projectPath.folder, getfilterSpeed=False, phase=phase
        )
        cluster_data = getSpikesfromClu(projectPath, behavior_data)
    return cluster_data
