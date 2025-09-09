# Load libs
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykeops
import tensorflow as tf
from tqdm import tqdm

from neuroencoders.fullEncoder import nnUtils
from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.utils.global_classes import Params, Project

## Different strategies are used for spike filtering in the case of the NN and of spike sorting.
# To make sure we end up with a fair comparison between the bayesian algorithm
# and the NN, one needs to give the same spike input to either decoding algorithm,
#
# This is done to have clear population-spike trains for the NN as a file:
# we translate the times of spikes found by the manual spike sorting algo
# (including the noise cluster) to a dataset for the NN

# Clarification from Dima: what is not done really is a bayesian decoder without noise

# TODO: another important idea: get detected spikes from the NN and use them to
#  do bayesian decoding. This would be a fair comparison of the two methods.
#  And, inversly, to input spike sorting results to the NN decoder.
#  However, none of this is done here.

pykeops.set_verbose(False)


class WaveFormComparator:
    def __init__(
        self,
        projectPath: Project,
        params: Params,
        behavior_data: dict,
        windowSizeMS: int = 36,
        useTrain: bool = True,
        useAll: bool = False,
        sleepName=[],
        **kwargs,
    ):  # todo allow for speed filtering
        self.projectPath = projectPath
        self.params = params
        self.behavior_data = behavior_data
        self.useTrain = useTrain
        self.useAll = useAll
        self.sleepName = sleepName
        self.windowSizeMS = windowSizeMS
        phase = kwargs.get("phase", None)
        self.phase = phase
        self.suffix = f"_{phase}" if phase is not None else ""
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.feat_desc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            "pos": tf.io.FixedLenFeature(
                [self.params.dimOutput], tf.float32
            ),  # target position: current value of the environmental correlate
            "length": tf.io.FixedLenFeature(
                [], tf.int64
            ),  # number of spike sequence gathered in the window
            "groups": tf.io.VarLenFeature(
                tf.int64
            ),  # the index of the groups having spike sequences in the window
            "time": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64),
        }
        for g in range(self.params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})

        useAll_suffix = "_all" if useAll else ""
        # Manage folder
        self.alignedDataPath = os.path.join(
            self.projectPath.dataPath,
            f"aligned_{phase}{useAll_suffix}",
            str(windowSizeMS),
        )
        if not os.path.isdir(self.alignedDataPath):
            os.makedirs(self.alignedDataPath)

        # Manage epochs
        if self.useTrain:
            epochMask = inEpochsMask(
                behavior_data["positionTime"][:, 0],
                behavior_data["Times"]["trainEpochs"],
            )
            if self.useAll:
                epochMask = np.logical_or(
                    epochMask,
                    inEpochsMask(
                        behavior_data["positionTime"][:, 0],
                        behavior_data["Times"]["testEpochs"],
                    ),
                )
        else:
            if bool(self.sleepName):
                idsleep = behavior_data["Times"]["sleepNames"].index(self.sleepName)
                timeSleepStart = behavior_data["Times"]["sleepEpochs"][2 * idsleep][0]
                timeSleepStop = behavior_data["Times"]["sleepEpochs"][2 * idsleep + 1][
                    0
                ]
            else:
                epochMask = inEpochsMask(
                    behavior_data["positionTime"][:, 0],
                    behavior_data["Times"]["testEpochs"],
                )

        # Load dataset
        if bool(self.sleepName):
            dataset_name = os.path.join(
                self.projectPath.dataPath,
                ("datasetSleep" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )
        else:
            dataset_name = os.path.join(
                self.projectPath.dataPath,
                ("dataset" + "_stride" + str(windowSizeMS) + ".tfrec"),
            )

        # Verify that the dataset is not empty
        if not tf.io.gfile.exists(dataset_name) or not tf.io.gfile.glob(dataset_name):
            raise FileNotFoundError(
                f"The dataset file does not exist: {dataset_name}. "
            )

        dataset = tf.data.TFRecordDataset(dataset_name)
        # Parse dataset
        self.dataset = dataset.map(
            lambda *vals: nnUtils.parse_serialized_spike(self.feat_desc, *vals),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if bool(self.sleepName):
            self.dataset = self.dataset.filter(
                lambda x: tf.math.logical_and(
                    tf.math.less_equal(x["time"], timeSleepStop),
                    tf.math.greater_equal(x["time"], timeSleepStart),
                )
            )
        else:
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(np.arange(len(epochMask)), dtype=tf.int64),
                    tf.constant(epochMask, dtype=tf.float64),
                ),
                default_value=0,
            )
            self.dataset = self.dataset.filter(
                lambda x: tf.equal(table.lookup(x["pos_index"]), 1.0)
            )
            self.dataset = self.dataset.map(
                nnUtils.import_true_pos(behavior_data["Positions"])
            )
            self.dataset = self.dataset.filter(
                lambda x: tf.math.logical_not(
                    tf.math.is_nan(tf.math.reduce_sum(x["pos"]))
                )
            )

        self.dataset = self.dataset.map(
            lambda *vals: nnUtils.parse_serialized_sequence(
                self.params, *vals, batched=False
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    def save_alignment_tools(
        self, trainerBayes, linearizationFunction, windowSizeMS=36, redo=False
    ):
        # Manage folder
        if self.useTrain:
            foldertosave = os.path.join(self.alignedDataPath, "train")
        else:
            if bool(self.sleepName) and not self.useTrain:
                foldertosave = os.path.join(self.alignedDataPath, self.sleepName)
            else:
                foldertosave = os.path.join(self.alignedDataPath, "test")
        if not os.path.isdir(foldertosave):
            os.makedirs(foldertosave)
        if (
            os.path.isfile(
                os.path.join(foldertosave, f"spikeMat_times_window{self.suffix}.csv")
            )
            and not redo
        ):
            return

        # Get data
        self.get_data()
        if not hasattr(trainerBayes, "linearPreferredPos"):
            _ = trainerBayes.train_order_by_pos(
                self.behavior_data, l_function=linearizationFunction
            )
        # gather all windows in the tensorflow dataset
        inputNN = self.get_NNdataset_spikepos()

        ### Mapping spikes from automatic ANN pipeline to windows
        lenInputNN = []  # Number of spikes per window
        meanTimeWindow = []  # Mean time of spikes in the window
        startTimeWindow = []  # Start of windows
        startTimeWindowInSamples = []  # Start of windows in samples
        for _, startTime in tqdm(enumerate(inputNN)):
            if len(startTime) > 0:
                startTimeWindow += [startTime[0] / self.samplingRate]
                startTimeWindowInSamples += [startTime[0]]
            else:
                startTimeWindow += [
                    np.nan
                ]  # we make sure these windows are never selected
            lenInputNN += [len(startTime)]
            timeWindowInSec = [sample / self.samplingRate for sample in startTime]
            meanTimeWindow += [np.mean(timeWindowInSec)]
        lenInputNN = np.array(lenInputNN)
        meanTimeWindow = np.array(meanTimeWindow)
        startTimeWindow = np.array(startTimeWindow)
        startTimeWindowInSamples = np.array(startTimeWindowInSamples)
        # Get rid of empty windows
        goodStartTimeWindowInSamples = startTimeWindowInSamples[
            np.logical_not(np.isnan(startTimeWindowInSamples))
        ]
        stopTimeWindowInSamples = goodStartTimeWindowInSamples + int(
            windowSizeMS / 1000 * self.samplingRate
        )

        ### Mapping spike sorted spike times to windows
        spikeMat_times_window = np.zeros([trainerBayes.spikeMatTimes.shape[0], 2])
        spikeMat_times_window[:, 0] = trainerBayes.spikeMatTimes[:, 0]
        spikeTime_lazy = pykeops.numpy.LazyTensor(
            trainerBayes.spikeMatTimes[:, 0][:, None] * self.samplingRate, axis=0
        )
        startTimeWindow_lazy = pykeops.numpy.Vj(
            goodStartTimeWindowInSamples[:, None].astype(dtype=np.float64)
        )
        stopTimeWindow_lazy = pykeops.numpy.Vj(
            stopTimeWindowInSamples[:, None].astype(dtype=np.float64)
        )
        ans = (spikeTime_lazy - startTimeWindow_lazy).relu().sign() * (
            (stopTimeWindow_lazy - spikeTime_lazy).relu().sign()
        )
        ans2 = ans.max_argmax_reduction(dim=1)
        ans2[1][np.equal(ans2[0], 0)] = -1
        spikeMat_times_window[:, 1] = ans2[1][:, 0]
        # for the pop vector we add one label for the noisy cluster
        spikeMat_window_popVector = np.zeros(
            [len(inputNN), trainerBayes.spikeMatLabels.shape[1] + 1]
        )
        for idSpike, window in tqdm(enumerate(spikeMat_times_window[:, 1])):
            if window != -1:
                cluster = np.where(
                    np.equal(trainerBayes.spikeMatLabels[idSpike, :], 1)
                )[0]
                if len(cluster) > 0:
                    spikeMat_window_popVector[int(window), 1 + cluster[0]] += 1
                else:
                    spikeMat_window_popVector[int(window), 0] += 1  # noisy cluster

        ### Saving
        df = pd.DataFrame(spikeMat_window_popVector)
        df.to_csv(
            os.path.join(foldertosave, f"spikeMat_window_popVector{self.suffix}.csv")
        )
        df = pd.DataFrame(meanTimeWindow)
        df.to_csv(os.path.join(foldertosave, f"meanTimeWindow{self.suffix}.csv"))
        df = pd.DataFrame(spikeMat_times_window)
        df.to_csv(os.path.join(foldertosave, f"spikeMat_times_window{self.suffix}.csv"))
        df = pd.DataFrame(startTimeWindow)
        df.to_csv(os.path.join(foldertosave, f"startTimeWindow{self.suffix}.csv"))
        df = pd.DataFrame(lenInputNN)
        df.to_csv(os.path.join(foldertosave, f"lenInputNN{self.suffix}.csv"))

    def get_NNdataset_spikepos(self):
        resData = self.dataset.map(lambda vals: vals["indexInDat"])
        return list(resData.as_numpy_iterator())

    def get_data(self):
        # Get names
        filPath = self.projectPath.fil
        datPath = self.projectPath.dat
        xmlPath = self.projectPath.xml
        # Map the data
        _, self.samplingRate, nChannels = get_params(xmlPath)
        self.number_timeSteps = os.stat(datPath).st_size // (2 * nChannels)
        self.memmapData = np.memmap(
            datPath, dtype=np.int16, mode="r", shape=(self.number_timeSteps, nChannels)
        )
        self.memmapFil = np.memmap(
            filPath, dtype=np.int16, mode="r", shape=(self.number_timeSteps, nChannels)
        )


def reconstruct_spike_waveforms(vals, params):
    """
    Reconstruct individual spike waveforms from the processed tensors.

    Args:
        vals: Dictionary containing groups, group+str(g), and indices tensors
        params: Parameters object with batchSize, nGroups, nChannelsPerGroup

    Returns:
        reconstructed_spikes: List of [batch, nspikes, nChannels, 32] arrays per group
        spike_positions: List of positions where spikes occurred per group
        batch_assignments: Which batch each spike belongs to
    """
    batch_size = params.batchSize
    reconstructed_spikes = []
    spike_positions = []
    batch_assignments = []

    # Reshape groups to [batch_size, seq_len]
    groups_per_batch = tf.reshape(vals["groups"], [batch_size, -1])
    seq_len_per_batch = tf.cast(tf.shape(groups_per_batch)[1], tf.int64)

    for group in range(params.nGroups):
        # Get spike waveforms for this group: [n_spikes, nChannels, 32]
        group_spikes = vals[f"group{group}"]
        n_channels = params.nChannelsPerGroup[group]

        if tf.shape(group_spikes)[0] == 0:
            # No spikes for this group
            reconstructed_spikes.append(tf.zeros([batch_size, 0, n_channels, 32]))
            spike_positions.append([])
            batch_assignments.append([])
            continue

        # Get indices for this group: [total_positions]
        indices = vals[f"indices{group}"]

        # Find where spikes occur (non-zero indices)
        spike_locations = tf.where(indices > 0)[:, 0]  # Positions with spikes
        spike_indices = (
            tf.gather(indices, spike_locations) - 1
        )  # Convert to 0-based (subtract the +1 from create_indices)

        # Convert positions to batch and sequence indices
        batch_ids = spike_locations // seq_len_per_batch
        seq_positions = spike_locations % seq_len_per_batch

        # Group spikes by batch
        spikes_per_batch = []
        positions_per_batch = []

        for batch_idx in range(batch_size):
            # Find spikes belonging to this batch
            batch_mask = tf.equal(batch_ids, batch_idx)
            batch_spike_indices = tf.boolean_mask(spike_indices, batch_mask)
            batch_positions = tf.boolean_mask(seq_positions, batch_mask)

            # Get actual spike waveforms
            if tf.shape(batch_spike_indices)[0] > 0:
                batch_spikes = tf.gather(group_spikes, batch_spike_indices)
            else:
                batch_spikes = tf.zeros([0, n_channels, 32], dtype=group_spikes.dtype)

            spikes_per_batch.append(batch_spikes)
            positions_per_batch.append(batch_positions)

        # Convert to consistent format
        # Find max spikes per batch for padding
        max_spikes = max(
            [tf.shape(batch_spikes)[0] for batch_spikes in spikes_per_batch]
        )
        if max_spikes == 0:
            max_spikes = 1  # Avoid empty tensor

        padded_spikes = []
        for batch_spikes in spikes_per_batch:
            n_spikes = tf.shape(batch_spikes)[0]
            if n_spikes > 0:
                padding = max_spikes - n_spikes
                if padding > 0:
                    pad_zeros = tf.zeros(
                        [padding, n_channels, 32], dtype=batch_spikes.dtype
                    )
                    padded_batch = tf.concat([batch_spikes, pad_zeros], axis=0)
                else:
                    padded_batch = batch_spikes
            else:
                padded_batch = tf.zeros(
                    [max_spikes, n_channels, 32], dtype=group_spikes.dtype
                )

            padded_spikes.append(padded_batch)

        # Stack to [batch_size, max_spikes, nChannels, 32]
        reconstructed_group = tf.stack(padded_spikes, axis=0)

        reconstructed_spikes.append(reconstructed_group)
        spike_positions.append(positions_per_batch)
        batch_assignments.append(batch_ids)

    return reconstructed_spikes, spike_positions, batch_assignments


def plot_spike_examples(
    reconstructed_spikes,
    spike_positions,
    params,
    batch_idx=0,
    group=0,
    max_spikes=10,
    figsize=(15, 10),
):
    """
    Plot examples of reconstructed spike waveforms.

    Args:
        reconstructed_spikes: Output from reconstruct_spike_waveforms
        spike_positions: Spike positions from reconstruction
        params: Parameters object
        batch_idx: Which batch sample to plot
        group: Which electrode group to plot
        max_spikes: Maximum number of spikes to plot
        figsize: Figure size for plotting
    """
    if group >= len(reconstructed_spikes):
        print(
            f"Group {group} not available. Available groups: 0-{len(reconstructed_spikes) - 1}"
        )
        return

    group_spikes = reconstructed_spikes[group][batch_idx]  # [max_spikes, nChannels, 32]
    n_channels = params.nChannelsPerGroup[group]

    # Convert to numpy for plotting
    spikes_np = group_spikes.numpy() if hasattr(group_spikes, "numpy") else group_spikes

    # Find actual (non-zero) spikes
    spike_norms = np.linalg.norm(spikes_np.reshape(spikes_np.shape[0], -1), axis=1)
    valid_spike_mask = spike_norms > 1e-6  # Threshold for non-zero spikes
    valid_spikes = spikes_np[valid_spike_mask]

    n_valid = valid_spikes.shape[0]
    n_to_plot = min(n_valid, max_spikes)

    if n_to_plot == 0:
        print(f"No valid spikes found in batch {batch_idx}, group {group}")
        return

    print(f"Plotting {n_to_plot} spikes from batch {batch_idx}, group {group}")
    print(f"Total valid spikes in this sample: {n_valid}")

    # Create subplot grid
    rows = min(4, n_to_plot)
    cols = max(1, n_to_plot // rows)
    if n_to_plot % rows != 0:
        cols += 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_to_plot == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    time_axis = np.linspace(0, 32, 32)  # 32 time steps

    for i in range(n_to_plot):
        row = i // cols
        col = i % cols

        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col] if cols > 1 else axes[i]

        spike_waveform = valid_spikes[i]  # [nChannels, 32]

        # Plot each channel
        for ch in range(n_channels):
            ax.plot(
                time_axis,
                spike_waveform[ch, :],
                label=f"Ch {ch}",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_title(f"Spike {i + 1}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Voltage")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide empty subplots
    total_subplots = rows * cols
    for i in range(n_to_plot, total_subplots):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            if cols > 1:
                axes[col].set_visible(False)

    plt.tight_layout()
    plt.suptitle(f"Spike Waveforms - Batch {batch_idx}, Group {group}", y=1.02)
    plt.show()


def analyze_spike_statistics(reconstructed_spikes, params):
    """
    Analyze statistics of reconstructed spikes across batches and groups.
    """
    print("=== SPIKE STATISTICS ===")

    total_spikes_per_batch = []

    for batch_idx in range(params.batchSize):
        batch_total = 0
        for group in range(params.nGroups):
            group_spikes = reconstructed_spikes[group][batch_idx].numpy()

            # Count non-zero spikes
            spike_norms = np.linalg.norm(
                group_spikes.reshape(group_spikes.shape[0], -1), axis=1
            )
            n_valid = np.sum(spike_norms > 1e-6)
            batch_total += n_valid

            if batch_idx < 5:  # Print details for first few batches
                print(f"Batch {batch_idx}, Group {group}: {n_valid} spikes")

        total_spikes_per_batch.append(batch_total)

        if batch_idx < 5:
            print(f"Batch {batch_idx} total: {batch_total} spikes")

    print("\nOverall statistics:")
    print(f"Mean spikes per batch: {np.mean(total_spikes_per_batch):.1f}")
    print(f"Std spikes per batch: {np.std(total_spikes_per_batch):.1f}")
    print(
        f"Min/Max spikes per batch: {np.min(total_spikes_per_batch)}/{np.max(total_spikes_per_batch)}"
    )


# Usage example:
def plot_spike_examples_from_vals(vals, params, **kwargs):
    """
    Complete pipeline: reconstruct and plot spikes from vals dictionary.

    ```python
    # Reconstruct spike waveforms
    print("Reconstructing spike waveforms...")
    reconstructed_spikes, spike_positions, batch_assignments = (
        reconstruct_spike_waveforms(vals, params)
    )

    # Analyze statistics
    analyze_spike_statistics(reconstructed_spikes, params)

    # Plot examples
    plot_spike_examples(reconstructed_spikes, spike_positions, params, **kwargs)

    return reconstructed_spikes, spike_positions, batch_assignments
    ```
    """
    # Reconstruct spike waveforms
    print("Reconstructing spike waveforms...")
    reconstructed_spikes, spike_positions, batch_assignments = (
        reconstruct_spike_waveforms(vals, params)
    )

    # Analyze statistics
    analyze_spike_statistics(reconstructed_spikes, params)

    # Plot examples
    plot_spike_examples(reconstructed_spikes, spike_positions, params, **kwargs)

    return reconstructed_spikes, spike_positions, batch_assignments
