# Load libs
import os
from typing import Callable

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL", "2"
)  # 0=all, 1=no Info, 2=no Warnings, 3=no Errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykeops
import tensorflow as tf
from tqdm import tqdm

from neuroencoders.fullEncoder import nnUtils
from neuroencoders.importData.epochs_management import get_epochs_mask, inEpochsMask
from neuroencoders.importData.rawdata_parser import get_params
from neuroencoders.simpleBayes.decode_bayes import Trainer
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
        useTest: bool = True,
        useAll: bool = False,
        sleepName=[],
        **kwargs,
    ):  # todo allow for speed filtering
        """
        Class to compare spike waveforms from different electrode groups.
        Args:
            projectPath: Project object with paths to data files
            params: Params object with model parameters
            behavior_data: Dictionary with behavioral data and epochs
            windowSizeMS: Size of the time window in milliseconds (default=36)
            useTrain: If True, use training epochs; else use test epochs (default=True)
            useTest: If True, use test epochs (default=True)
            useAll: If True, use both training and test epochs (default=False)
            sleepName: Name of the sleep epoch to filter on (default=[])
            **kwargs: Additional arguments, including:
                - phase: 'train' or 'test' to specify dataset phase
                - strideFactor: Factor for striding the dataset (default=4)

        """
        self.projectPath = projectPath
        self.params = params
        self.behavior_data = behavior_data
        self.useTrain = useTrain
        self.useTest = useTest
        self.useAll = useAll
        if self.useAll:
            self.useTrain = True
            self.useTest = True
        if self.useTrain and self.useTest and not self.useAll:
            raise ValueError(
                "If both useTrain and useTest are True, set useAll to True."
            )
        self.sleepName = sleepName
        self.windowSizeMS = windowSizeMS
        phase = kwargs.get("phase", None)
        self.phase = phase
        self.suffix = f"_{phase}" if phase is not None else ""
        # The feat_desc is used by the tf.io.parse_example to parse what we previously saved
        # as tf.train.Feature in the proto format.
        self.max_nb_spikes = kwargs.get("max_nb_spikes", 400)
        self.max_spikes_per_group = kwargs.get(
            "max_spikes_per_group", int(self.max_nb_spikes / self.params.nGroups)
        )
        self.feat_desc = {
            # index of the position in the position array
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            # target position: current value of the environmental correlate
            # WARNING: if the target is not position, this might change
            # this is very dirty, but we need to hardcode the position dimension to 2 for the TFRecord parsing
            # then it will be modified by the DataHelper.get_true_target method to actually match the target.
            "pos": tf.io.FixedLenFeature([2], tf.float32),
            # number of spike sequence gathered in the window
            "length": tf.io.FixedLenFeature([], tf.int64),
            # the index of the groups having spike sequences in the window
            "groups": tf.io.VarLenFeature(tf.int64),
            # the mean time-steps of each spike measured in the various groups.
            # Question: should the time not be a VarLenFeature??
            "time": tf.io.FixedLenFeature([], tf.float32),
            # the exact time step from behaviorData["Times"]
            "time_behavior": tf.io.FixedLenFeature([], tf.float32),
            # sample of the spike
            "indexInDat": tf.io.VarLenFeature(tf.int64),
        }
        for g in range(self.params.nGroups):
            self.feat_desc.update({"group" + str(g): tf.io.VarLenFeature(tf.float32)})

        strideFactor = kwargs.get("strideFactor", 4)
        useAll_suffix = "_all" if useAll else ""
        strideFactor_suffix = f"_factor{strideFactor}" if strideFactor > 1 else ""
        # Manage folder
        self.alignedDataPath = os.path.join(
            self.projectPath.dataPath,
            f"aligned_{phase}{useAll_suffix}{strideFactor_suffix}",
            str(windowSizeMS),
        )
        if not os.path.isdir(self.alignedDataPath):
            os.makedirs(self.alignedDataPath)

        # Manage epochs and determine mask based on dataset split
        if bool(self.sleepName):
            # Sleep dataset path
            idsleep = behavior_data["Times"]["sleepNames"].index(self.sleepName)
            timeSleepStart = behavior_data["Times"]["sleepEpochs"][2 * idsleep][0]
            timeSleepStop = behavior_data["Times"]["sleepEpochs"][2 * idsleep + 1][0]
            use_sleep_filter = True
            totMask = None
        else:
            use_sleep_filter = False
            if self.useTrain or self.useTest:
                totMask = get_epochs_mask(
                    behaviorData=behavior_data,
                    useTrain=self.useTrain,
                    useTest=self.useTest,
                )
            else:
                totMask = inEpochsMask(
                    behavior_data["positionTime"][:, 0],
                    behavior_data["Times"]["testEpochs"],
                )

        # Determine filename
        if bool(self.sleepName):
            if strideFactor > 1:
                filename = (
                    f"datasetSleep_stride{windowSizeMS}_factor{strideFactor}.tfrec"
                )
            else:
                filename = f"datasetSleep_stride{windowSizeMS}.tfrec"
            dataset_name = os.path.join(self.projectPath.dataPath, filename)
        else:
            if strideFactor > 1:
                filename = f"dataset_stride{windowSizeMS}_factor{strideFactor}.tfrec"
            else:
                filename = f"dataset_stride{windowSizeMS}.tfrec"
            dataset_name = os.path.join(self.projectPath.dataPath, filename)

        # Verify that the dataset exists
        if not tf.io.gfile.exists(dataset_name) or not tf.io.gfile.glob(dataset_name):
            raise FileNotFoundError(
                f"The dataset file does not exist: {dataset_name}. "
            )

        # Pipeline configuration: defaults for "true test dataset"
        # No shuffling, no augmentation, no oversampling
        shuffle = kwargs.get("shuffle", False)

        # Define parse function with @tf.function
        @tf.function
        def _parse_function(*vals):
            return nnUtils.parse_serialized_spike(self.feat_desc, *vals)

        # Define filter functions
        def get_mask_filter(mask):
            mask_tensor = tf.constant(mask, dtype=tf.float32)

            @tf.function
            def filter_by_pos_index(x):
                pos_index = x["pos_index"]
                mask_size = tf.size(mask_tensor, out_type=pos_index.dtype)
                is_non_negative = tf.math.greater_equal(pos_index, 0)
                is_in_range = tf.math.less(pos_index, mask_size)
                valid = tf.math.logical_and(is_non_negative, is_in_range)
                safe_index = tf.where(valid, pos_index, tf.zeros_like(pos_index))
                gathered = tf.gather(mask_tensor, safe_index)
                return tf.math.logical_and(valid, tf.equal(gathered, 1.0))

            return filter_by_pos_index

        @tf.function
        def filter_nan_pos(x):
            pos_data = x["pos"]
            return tf.math.logical_not(tf.math.is_nan(tf.math.reduce_sum(pos_data)))

        # Load dataset with buffer
        ndataset = tf.data.TFRecordDataset(
            dataset_name,
            buffer_size=100 * 1024 * 1024,  # 100MB read buffer
        )

        # Optional shuffling before parsing (at raw record level)
        if shuffle:
            print("Shuffling the dataset (pre-parsing)")
            ndataset = ndataset.shuffle(10000)

        # Parse the records
        ndataset = ndataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ndataset = ndataset.prefetch(tf.data.AUTOTUNE)

        # Apply filtering based on dataset type (sleep or epoch-based)
        if use_sleep_filter:
            self.dataset = ndataset.filter(
                lambda x: tf.math.logical_and(
                    tf.math.less_equal(x["time"], timeSleepStop),
                    tf.math.greater_equal(x["time"], timeSleepStart),
                )
            )
        else:
            filter_op = get_mask_filter(totMask)
            self.dataset = ndataset.filter(filter_op)

        # Import true positions
        self.dataset = self.dataset.map(
            nnUtils.import_true_pos(behavior_data["Positions"]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Filter out NaN positions
        self.dataset = self.dataset.filter(filter_nan_pos).prefetch(tf.data.AUTOTUNE)

        # Parse spike sequences
        def parse_serialized_sequence(vals):
            return nnUtils.parse_serialized_sequence(
                self.params,
                vals,
                max_spikes=self.max_nb_spikes,
                max_spikes_per_group=self.max_spikes_per_group,
            )

        self.dataset = self.dataset.map(
            parse_serialized_sequence, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Create indices for spike gathering
        def create_indices_callable(vals, shuffle=False):
            return self.create_indices(vals, shuffle=shuffle)

        self.dataset = self.dataset.map(
            create_indices_callable, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Final prefetch and options
        options = tf.data.Options()
        options.threading.private_threadpool_size = 4
        self.dataset = self.dataset.with_options(options).prefetch(tf.data.AUTOTUNE)

        print(f"Dataset {dataset_name} loaded.")

    def create_indices(self, vals, shuffle=False):
        """
        Create relative indices for gathering spikes from each group.
        The i-th spike of the group should be positioned at spikePosition[i] in the final tensor.

        Args:
            vals (dict): A dictionary containing the input tensors, including "groups" and "group{n}" for each group.
            addLinearizationTensor (bool): Whether to add linearization tensors to the output.
            shuffle (bool): Whether to shuffle the indices within each group for null hypothesis/control.
        Returns:
            dict: Updated dictionary with indices for each group. The indices are stored under the keys "indices{n}" for each group.
        """
        groups = vals["groups"]
        for group_id in range(self.params.nGroups):
            # Find positions of spikes belonging to this group
            is_in_group = tf.equal(groups, group_id)

            # 2. Use cumsum to generate sequential IDs (1, 2, 3...) for these spikes
            # This replaces the SparseTensor logic entirely.
            # Example: [0, 1, 0, 1] -> [0, 1, 1, 2]
            relative_indices = tf.cast(
                tf.cumsum(tf.cast(is_in_group, tf.int32)), tf.int32
            )

            # Ensure that relative indices do not exceed the available spike slots
            # per group (to avoid out-of-bounds accesses when gathering).
            max_spikes_per_group = getattr(self.params, "max_nb_spikes_per_group", None)
            if max_spikes_per_group is not None:
                relative_indices = tf.clip_by_value(
                    relative_indices,
                    clip_value_min=0,
                    clip_value_max=max_spikes_per_group - 1,
                )

            # 3. Apply the mask so only spikes in this group have a non-zero index
            # Example: [0, 1, 1, 2] -> [0, 1, 0, 2]
            indices_tensor = tf.where(is_in_group, relative_indices, 0)

            vals[f"indices{group_id}"] = indices_tensor

        return vals

    def __repr__(self):
        return f"WaveFormComparator(projectPath={self.projectPath}, params={self.params}, behavior_data=Dict with keys {list(self.behavior_data.keys())}, windowSizeMS={self.windowSizeMS}, useTrain={self.useTrain}, useTest={self.useTest}, useAll={self.useAll}, sleepName={self.sleepName}, phase={self.phase})"

    def save_alignment_tools(
        self,
        bayes: Trainer,
        linearizationFunction: Callable,
        windowSizeMS: int = 36,
        redo: bool = False,
    ):
        # Manage folder
        if self.useTrain:
            foldertosave = os.path.join(self.alignedDataPath, "train")
        else:
            if bool(self.sleepName) and not self.useTrain:
                foldertosave = os.path.join(self.alignedDataPath, self.sleepName)
            else:
                if not self.useTest:
                    raise ValueError(
                        "Either useTrain or useTest must be True to save alignment tools."
                    )
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
        if not hasattr(bayes, "linearPreferredPos"):
            _ = bayes.train_order_by_pos(
                self.behavior_data, l_function=linearizationFunction
            )
        # gather all windows in the tensorflow dataset
        inputNN, posIndexNN = self.get_NNdataset_spikepos()

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
        spikeMat_times_window = np.zeros([bayes.spikeMatTimes.shape[0], 2])
        spikeMat_times_window[:, 0] = bayes.spikeMatTimes[:, 0]
        spikeTime_lazy = pykeops.numpy.LazyTensor(
            bayes.spikeMatTimes[:, 0][:, None] * self.samplingRate, axis=0
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
            [len(inputNN), bayes.spikeMatLabels.shape[1] + 1]
        )
        for idSpike, window in tqdm(enumerate(spikeMat_times_window[:, 1])):
            if window != -1:
                cluster = np.where(np.equal(bayes.spikeMatLabels[idSpike, :], 1))[0]
                if len(cluster) > 0:
                    spikeMat_window_popVector[int(window), 1 + cluster[0]] += 1
                else:
                    spikeMat_window_popVector[int(window), 0] += 1  # noisy cluster

        ### Saving
        df = pd.DataFrame(spikeMat_window_popVector)
        print(
            f"Saving spikeMat_window_popVector with shape {spikeMat_window_popVector.shape} to {os.path.join(foldertosave, f'spikeMat_window_popVector{self.suffix}.csv')}"
        )
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
        df = pd.DataFrame(posIndexNN)
        df.to_csv(os.path.join(foldertosave, f"posIndexNN{self.suffix}.csv"))

    def get_NNdataset_spikepos(self):
        """
        From the dataset, get the spike indices in the datfile and their corresponding position indices.
        This way we get a precise mapping between prediction time and spike times, which allows us to compare the spike waveforms from the NN pipeline to the ones from spike sorting.

        """
        resData = self.dataset.map(lambda vals: vals["indexInDat"])
        posIndexData = self.dataset.map(lambda vals: vals["pos_index"])
        # Filter out padding (-1) introduced by parse_serialized_sequence
        return (
            [
                x[x != -1]
                for x in tqdm(
                    resData.as_numpy_iterator(), desc="Collecting spike indices"
                )
            ],
            np.array(
                [
                    x
                    for x in tqdm(
                        posIndexData.as_numpy_iterator(),
                        desc="Collecting position indices",
                    )
                ]
            ),
        )

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

    def get_batched_dataset(self, batch_size=None, shuffle=True):
        """
        Get a batched dataset iterator, mimicking the network input pipeline.
        Args:
           batch_size: If None, uses self.params.batch_size
        Returns:
           iterator: yielding batched dictionaries
        """
        bs = batch_size if batch_size is not None else self.params.batch_size
        ds = self.dataset
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)

        # We need to make sure we output the same structure as the network expects
        # The network usually expects a tuple (inputs, targets) or just inputs dictionary if custom loop
        # Here we return the dictionary

        ds = ds.batch(bs, drop_remainder=True)
        return ds

    def reconstruct_spike_waveforms(self, vals):
        """
        Get spike waveforms from the processed tensors (NO sparse reconstruction needed anymore).

        Args:
            vals: Dictionary containing batched tensors

        Returns:
            reconstructed_spikes: List of [batch, max_spikes, nChannels, 32] tensors per group
        """
        reconstructed_spikes = []

        for group in range(self.params.nGroups):
            key = f"group{group}"
            if key in vals:
                # The parsing already delivers [batch, max_spikes, nCh, 32]
                reconstructed_spikes.append(vals[key])
            else:
                # Fallback / Placeholder
                print(f"Warning: {key} not found in vals")
                reconstructed_spikes.append(None)

        return reconstructed_spikes

    def analyze_spike_statistics(self, reconstructed_spikes):
        """
        Analyze statistics of spikes across the batch.
        Args:
            reconstructed_spikes: List of 4D tensors per group
        """
        print("=== SPIKE STATISTICS ===")
        if not reconstructed_spikes:
            return

        # Assume batch dim is 0
        batch_size = reconstructed_spikes[0].shape[0]

        for i, group_data in enumerate(reconstructed_spikes):
            if group_data is None:
                continue

            # shape: [Batch, MaxSpikes, Ch, Time]
            # Verify if it's tensor or numpy
            if hasattr(group_data, "numpy"):
                data = group_data.numpy()
            else:
                data = group_data

            # Check for non-zero spikes (energy > threshold)
            # Flatten to [Batch*MaxSpikes, Features]
            flat = data.reshape(-1, np.prod(data.shape[2:]))
            norms = np.linalg.norm(flat, axis=1)
            n_valid = np.sum(norms > 1e-6)

            print(f"Group {i}: {n_valid} valid spikes in batch of {batch_size} samples")
            print(f"  Shape: {data.shape}")
            print(
                f"  Mean Amp: {np.mean(np.abs(data)):.2f}, Max: {np.max(np.abs(data)):.2f}"
            )

    def plot_spike_examples(
        self, reconstructed_spikes, batch_idx=0, group=0, max_spikes=10
    ):
        """
        Plot examples of spikes from a specific batch item and group.
        """
        if group >= len(reconstructed_spikes) or reconstructed_spikes[group] is None:
            print(f"Group {group} not available")
            return

        data = reconstructed_spikes[group]
        if hasattr(data, "numpy"):
            data = data.numpy()

        # Get single batch item: [MaxSpikes, Ch, Time]
        if batch_idx >= data.shape[0]:
            print("Batch index out of bounds")
            return

        spikes = data[batch_idx]
        # Filter empty slots
        norms = np.linalg.norm(spikes.reshape(spikes.shape[0], -1), axis=1)
        valid_indices = np.where(norms > 1e-6)[0]

        if len(valid_indices) == 0:
            print("No valid spikes in this sample.")
            return

        to_plot = valid_indices[:max_spikes]

        n_plot = len(to_plot)
        cols = 5
        rows = (n_plot // cols) + (1 if n_plot % cols > 0 else 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = np.array(axes).flatten()

        parameters = self.params
        n_ch = parameters.nChannelsPerGroup[group]

        for i, idx in enumerate(to_plot):
            ax = axes[i]
            # waveform: [nCh, 32]
            wf = spikes[idx]
            for c in range(n_ch):
                ax.plot(wf[c], label=f"Ch{c}")
            ax.set_title(f"Spike {idx}")

        plt.tight_layout()
        plt.show()

    def check_distribution(self, reconstructed_spikes, group=0):
        """Basic PCA check on the batch"""
        from sklearn.decomposition import PCA

        if group >= len(reconstructed_spikes) or reconstructed_spikes[group] is None:
            return

        data = reconstructed_spikes[group]  # [Batch, MaxSpikes, Ch, Time]
        if hasattr(data, "numpy"):
            data = data.numpy()

        # Flatten all spikes in batch
        # [Batch * MaxSpikes, Ch * Time]
        flat_all = data.reshape(-1, np.prod(data.shape[-2:]))

        # Filter zeros
        norms = np.linalg.norm(flat_all, axis=1)
        valid = flat_all[norms > 1e-6]

        if len(valid) < 3:
            print("Not enough spikes for PCA")
            return

        print(f"Running PCA on {len(valid)} spikes...")
        pca = PCA(n_components=2)
        proj = pca.fit_transform(valid)

        plt.figure(figsize=(8, 6))
        plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5, s=5)
        plt.title(f"PCA of Batch Spikes (Group {group})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()


def reconstruct_spike_waveforms(vals, params: Params):
    """
    Reconstruct individual spike waveforms from the processed tensors.

    Args:
        vals: Dictionary containing groups, group+str(g), and indices tensors
        params: Parameters object with batch_size, nGroups, nChannelsPerGroup

    Returns:
        reconstructed_spikes: List of [batch, nspikes, nChannels, 32] arrays per group
        spike_positions: List of positions where spikes occurred per group
        batch_assignments: Which batch each spike belongs to
    """
    batch_size = params.batch_size
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


def analyze_spike_statistics(reconstructed_spikes, params: Params):
    """
    Analyze statistics of reconstructed spikes across batches and groups.
    """
    print("=== SPIKE STATISTICS ===")

    total_spikes_per_batch = []

    for batch_idx in range(params.batch_size):
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
