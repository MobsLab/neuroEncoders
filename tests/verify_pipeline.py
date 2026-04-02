#!/usr/bin/env python3
"""
Visual debug script using a real LSTMandSpikeNetwork pipeline.

Outputs:
1) Before/after normalization visuals for SpikeNet1D and SpikeNet2D.
2) Masking logic visuals from waveform padding -> per-group masks ->
   sequence mask -> gathered features -> masked sequence.

Run with the local venv from repo root:
  ./.venv/bin/python verify_pipeline.py
"""

import argparse
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from neuroencoders.fullEncoder.an_network import LSTMandSpikeNetwork
from neuroencoders.transformData.linearizer import UMazeLinearizer


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def subset_tfrecord(src: str, dst: str, n_records: int) -> int:
    """Write first n_records of src TFRecord into dst and return max pos_index seen."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)

    max_pos_index = 0
    ds = tf.data.TFRecordDataset(src)
    with tf.io.TFRecordWriter(dst) as writer:
        for raw in ds.take(n_records):
            b = raw.numpy()
            writer.write(b)

            ex = tf.train.Example()
            ex.ParseFromString(b)
            if "pos_index" in ex.features.feature:
                vals = ex.features.feature["pos_index"].int64_list.value
                if len(vals) > 0:
                    max_pos_index = max(max_pos_index, int(max(vals)))

    return max_pos_index


def prepare_behavior_subset(
    full_behavior: dict, max_pos_index: int, pad: int = 128
) -> dict:
    """Create a behavior dict large enough for pos_index references in subset."""
    n_req = int(max_pos_index + pad)
    n_src = full_behavior["Positions"].shape[0]
    n = min(n_req, n_src)

    positions = np.asarray(full_behavior["Positions"][:n]).copy()
    position_time = np.asarray(full_behavior["positionTime"][:n]).copy()

    speed_filter = np.ones(n, dtype=bool)
    train_epochs = [[float(position_time[0, 0]), float(position_time[-1, 0])]]

    behavior = {
        "Positions": positions,
        "positionTime": position_time,
        "Times": {
            "speedFilter": speed_filter,
            "trainEpochs": train_epochs,
            "testEpochs": train_epochs,
            "lossPredSetEpochs": [],
        },
    }
    return behavior


def set_norm_weights_from_stats(
    net: LSTMandSpikeNetwork, means: list, stds: list
) -> None:
    """Set Keras Normalization weights with channel-vector shapes expected by the layer."""
    for g in range(net.params.nGroups):
        m = np.asarray(means[g], dtype=np.float32)
        v = np.asarray(stds[g], dtype=np.float32) ** 2
        count = np.array(1.0, dtype=np.float32)

        # Keras Normalization for axis=1 stores mean/variance as 1D channel vectors.
        print(
            f"Setting normalization weights for SpikeNet{g}: mean shape {m.shape}, var shape {v.shape}"
        )
        print(f"Mean sample values: {m[:5]} ...")
        net.spikeNets[g].input_normalization.set_weights([m, v, count])


def get_first_valid_waveform(group_tensor: np.ndarray) -> np.ndarray:
    """From (B, S, C, T), return first non-zero waveform (C, T)."""
    for b in range(group_tensor.shape[0]):
        for s in range(group_tensor.shape[1]):
            w = group_tensor[b, s]
            if np.any(w != 0.0):
                return w
    return np.zeros((group_tensor.shape[2], group_tensor.shape[3]), dtype=np.float32)


def get_nonzero_waveform_from_dataset(
    dataset: tf.data.Dataset, group_id: int = 0
) -> np.ndarray:
    """Find the first non-zero waveform across several batches."""
    key = f"group{group_id}"
    for inputs, _ in dataset.take(20):
        w = get_first_valid_waveform(inputs[key].numpy())
        if np.any(w != 0.0):
            return w
    raise RuntimeError(
        f"Could not find any non-zero waveform in {key} across sampled batches"
    )


def fallback_channel_stats_from_dataset(
    dataset: tf.data.Dataset,
    n_groups: int,
    n_channels_per_group: list,
    max_batches: int = 50,
) -> tuple:
    """Compute per-channel mean/std from non-zero spikes as fallback normalization stats."""
    sums = [np.zeros(c, dtype=np.float64) for c in n_channels_per_group]
    sums2 = [np.zeros(c, dtype=np.float64) for c in n_channels_per_group]
    counts = [0 for _ in range(n_groups)]

    for i, (inputs, _) in enumerate(dataset):
        if i >= max_batches:
            break
        for g in range(n_groups):
            arr = inputs[f"group{g}"].numpy()  # (B,S,C,T)
            valid = np.any(arr != 0.0, axis=(2, 3))
            if not np.any(valid):
                continue
            spikes = arr[valid]  # (N,C,T)
            flat = np.transpose(spikes, (0, 2, 1)).reshape(-1, n_channels_per_group[g])
            sums[g] += flat.sum(axis=0)
            sums2[g] += (flat**2).sum(axis=0)
            counts[g] += flat.shape[0]

    means = []
    stds = []
    for g in range(n_groups):
        if counts[g] == 0:
            means.append(np.zeros(n_channels_per_group[g], dtype=np.float32))
            stds.append(np.ones(n_channels_per_group[g], dtype=np.float32))
            continue
        m = sums[g] / counts[g]
        v = np.maximum((sums2[g] / counts[g]) - m**2, 1e-8)
        means.append(m.astype(np.float32))
        stds.append(np.sqrt(v).astype(np.float32))
    return means, stds


def build_masks_and_features(net: LSTMandSpikeNetwork, batch_inputs: dict):
    """Reconstruct internal masking stages from SpikeSequenceProcessor for plotting."""
    processor = net.spike_sequence_processor
    n_groups = net.params.nGroups

    groups_list = [batch_inputs[f"group{g}"] for g in range(n_groups)]
    indices_list = [batch_inputs[f"indices{g}"] for g in range(n_groups)]
    input_groups = batch_inputs["groups"]

    all_group_masks = [
        tf.reduce_any(tf.not_equal(g, 0.0), axis=[-1, -2]) for g in groups_list
    ]

    group_latents_raw = processor.spike_encoder(
        groups_list, mask=all_group_masks, training=False
    )

    all_group_latents = []
    for g, latent in enumerate(group_latents_raw):
        all_group_latents.append(processor.add_null_spike_layers[g](latent))

    pool = tf.concat(all_group_latents, axis=1)
    all_features = processor.sequence_reconstructor([pool, indices_list, input_groups])
    seq_mask = processor.safe_mask_creation(input_groups)
    masked_features = processor.masking_layer([seq_mask, all_features])

    return {
        "groups_list": groups_list,
        "indices_list": indices_list,
        "input_groups": input_groups,
        "group_masks": all_group_masks,
        "all_features": all_features,
        "seq_mask": seq_mask,
        "masked_features": masked_features,
    }


def plot_normalization_compare(
    out_path: str, waveform: np.ndarray, after_1d: np.ndarray, after_2d: np.ndarray
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    for ch in range(waveform.shape[0]):
        axs[0, 0].plot(waveform[ch], color="gray", alpha=0.7)
        axs[1, 0].plot(waveform[ch], color="gray", alpha=0.7)
        axs[0, 1].plot(after_1d[ch], color="blue", alpha=0.7)
        axs[1, 1].plot(after_2d[ch], color="green", alpha=0.7)
    axs[0, 0].set_title("Input waveform (C x T)")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Channels")

    axs[0, 1].set_title("After SpikeNet1D normalization")
    axs[0, 1].set_xlabel("Time")

    axs[0, 2].hist(waveform.ravel(), bins=40, alpha=0.55, label="before")
    axs[0, 2].hist(after_1d.ravel(), bins=40, alpha=0.55, label="after 1D")
    axs[0, 2].legend(fontsize=8)
    axs[0, 2].set_title("1D distribution shift")

    axs[1, 0].set_title("Input waveform (same)")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Channels")

    axs[1, 1].set_title("After SpikeNet2D normalization")
    axs[1, 1].set_xlabel("Time")

    axs[1, 2].hist(waveform.ravel(), bins=40, alpha=0.55, label="before")
    axs[1, 2].hist(after_2d.ravel(), bins=40, alpha=0.55, label="after 2D")
    axs[1, 2].legend(fontsize=8)
    axs[1, 2].set_title("2D distribution shift")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_masking_flow(out_path: str, stage: dict) -> None:
    groups0 = stage["groups_list"][0].numpy()
    indices0 = stage["indices_list"][0].numpy()
    input_groups = stage["input_groups"].numpy()
    group_mask0 = stage["group_masks"][0].numpy().astype(np.float32)
    seq_mask = stage["seq_mask"].numpy().astype(np.float32)

    all_features = stage["all_features"].numpy()
    masked_features = stage["masked_features"].numpy()

    # Pick a sample with the highest number of valid timesteps.
    valid_counts = np.sum(seq_mask > 0.5, axis=1)
    sample = int(np.argmax(valid_counts))
    valid_slots = (np.abs(groups0[sample]).sum(axis=(1, 2)) > 0).astype(np.float32)

    pre_norm = np.linalg.norm(all_features[sample], axis=-1)
    post_norm = np.linalg.norm(masked_features[sample], axis=-1)

    print(
        f"Sample {sample} has {valid_counts[sample]} valid timesteps in sequence mask."
    )
    print(f"pre_norm: {pre_norm[:10]}")
    print(f"post_norm: {post_norm[:10]}")

    fig, axs = plt.subplots(4, 2, figsize=(14, 10))

    axs[0, 0].imshow(groups0[sample, 0], aspect="auto", cmap="viridis")
    axs[0, 0].set_title("Single waveform (group0, first slot)")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Channels")

    axs[0, 1].plot(valid_slots, marker="o")
    axs[0, 1].set_ylim(-0.1, 1.1)
    axs[0, 1].set_title("Waveform-based spike-slot validity")
    axs[0, 1].set_xlabel("Spike slot")

    axs[1, 0].imshow(input_groups[sample][None, :], aspect="auto", cmap="tab20")
    axs[1, 0].set_title("input_groups sequence (-1 padded)")
    axs[1, 0].set_xlabel("Sequence timestep")
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(indices0[sample][None, :], aspect="auto", cmap="magma")
    axs[1, 1].set_title("indices0 sequence (0 = null spike)")
    axs[1, 1].set_xlabel("Sequence timestep")
    axs[1, 1].set_yticks([])

    axs[2, 0].plot(group_mask0[sample], lw=1.3, label="group0 mask")
    axs[2, 0].plot(seq_mask[sample], lw=1.8, label="sequence mask")
    axs[2, 0].set_ylim(-0.1, 1.1)
    axs[2, 0].set_title("Mask signals")
    axs[2, 0].set_xlabel("Timestep / Slot")
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].plot(pre_norm + 1, lw=1.8, label="feature norm pre-mask")
    axs[2, 1].plot(post_norm, lw=1.8, label="feature norm post-mask")
    axs[2, 1].set_title("Final sequence feature norms")
    axs[2, 1].set_xlabel("Sequence timestep")
    axs[2, 1].legend(fontsize=8)

    # plot the full sequence of features as successive waveforms, with masked timesteps in gray.
    masked = masked_features[sample].copy()
    masked[seq_mask[sample] < 0.5] = 0.0
    for i, g in enumerate(stage["input_groups"][sample].numpy()):
        pseudo_time = np.arange(32) + i * 32
        index = stage["indices_list"][g][sample][i].numpy()
        if index <= 0 or g < 0:
            axs[3, 0].plot(
                pseudo_time,
                np.zeros_like(pseudo_time),
                marker="x",
                color="red",
                label="invalid index",
            )
            print(
                f"Step {i}: group {g}, index {index}, seq_mask {seq_mask[sample, i]:.2f}"
            )
            continue
        waveform = stage["groups_list"][g][sample][index].numpy()
        for ch in range(waveform.shape[0]):
            axs[3, 0].plot(pseudo_time, waveform[ch], alpha=0.7)
    axs[3, 0].set_title("Original waveforms for sequence (masked timesteps in gray)")
    axs[3, 0].set_xlabel("Time")
    axs[3, 0].set_ylabel("Voltage")

    for t in range(masked.shape[1]):
        if seq_mask[sample, t] < 0.5:
            axs[3, 1].plot(
                masked[sample, t], color="gray", alpha=0.5, label="masked timestep"
            )
        else:
            axs[3, 1].plot(masked[sample, t], alpha=0.7)
    axs[3, 1].set_title("Masked sequence features (gray = masked timesteps)")
    axs[3, 1].set_xlabel("Feature dimension")
    axs[3, 1].set_ylabel("Feature value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize normalization and masking with real LSTMandSpikeNetwork."
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=300,
        help="How many TFRecord examples to keep from tests/dummy_dataset.tfrec",
    )
    parser.add_argument(
        "--out-dir",
        default="test_verification_run/visual_debug_output",
        help="Directory for outputs",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    # Disable external tracking side-effects for this local debug script.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    set_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("Importing neuroencoders modules...", flush=True)
    from neuroencoders.fullEncoder.an_network import LSTMandSpikeNetwork
    from neuroencoders.utils.global_classes import DataHelper, Params, Project

    root = Path(os.getcwd())
    tests_dir = root / "tests"
    run_dir = root / "test_verification_run"
    out_dir = root / args.out_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_tfrec = tests_dir / "dummy_dataset.tfrec"
    subset_tfrec = run_dir / "dataset_stride36.tfrec"

    print(f"Creating subset TFRecord: {src_tfrec} -> {subset_tfrec}")
    max_pos_index = subset_tfrecord(str(src_tfrec), str(subset_tfrec), args.subset_size)
    print(f"Subset done. Max pos_index: {max_pos_index}")

    print("Loading Project / Params / DataHelper test artifacts...")
    project = Project.load(str(tests_dir / "project.pkl"))
    params_base = Params.load(str(tests_dir / "params.pkl"))
    data_helper = DataHelper.load(str(tests_dir / "dummy_datahelper.pkl"))

    Linearizer = UMazeLinearizer(
        project.folder,
        phase=data_helper.phase,
        data_helper=data_helper,
    )
    behavior_data = prepare_behavior_subset(
        data_helper.fullBehavior, max_pos_index=max_pos_index
    )

    # Force this run to use our local subset path and folders.
    project.dataPath = str(run_dir)
    project.experimentPath = str(run_dir / "mock_experiment")
    project.folderResult = str(Path(project.experimentPath) / "results")
    project.folderResultSleep = str(Path(project.experimentPath) / "results_sleep")
    os.makedirs(project.folderResult, exist_ok=True)
    os.makedirs(project.folderResultSleep, exist_ok=True)

    print("Building real LSTMandSpikeNetwork models (1D and 2D)...")
    params_1d = copy.copy(params_base)
    params_2d = copy.copy(params_base)

    params_1d.use_conv2d = False
    params_2d.use_conv2d = True
    params_1d.usingMixedPrecision = False
    params_2d.usingMixedPrecision = False

    net_1d = LSTMandSpikeNetwork(
        project,
        params_1d,
        deviceName="/cpu:0",
        debug=False,
        max_nb_spikes=200,
        phase="pre",
        isTransformer=params_1d.isTransformer,
        verbose=True,
        behaviorData=behavior_data,
        alpha=params_1d.denseweightAlpha,
        linearizer=Linearizer,
        jit_compile=False,
    )

    net_2d = LSTMandSpikeNetwork(
        project,
        params_2d,
        deviceName="/cpu:0",
        debug=False,
        max_nb_spikes=200,
        phase="pre",
        isTransformer=params_2d.isTransformer,
        verbose=True,
        behaviorData=behavior_data,
        alpha=params_2d.denseweightAlpha,
        linearizer=Linearizer,
        jit_compile=False,
    )

    print("Preparing masks and loading one dataset batch...")
    total_n = behavior_data["Positions"].shape[0]
    all_true = np.ones(total_n, dtype=bool)
    tot_mask = {"train": all_true, "test": all_true}

    datasets_1d, _ = net_1d._dataset_loading_pipeline(
        filename="dataset_stride36.tfrec",
        windowSizeMS=36,
        behaviorData=behavior_data,
        totMask=tot_mask,
        augmentation_config=None,
        enable_augmentation=False,
        oversampling_resampling=False,
        shuffle=False,
        is_interleaving_subdataset=False,
        batch_size=8,
    )

    # Use unbatched train parse for normalization stats estimation.
    ds_stats, _ = net_1d._dataset_loading_pipeline(
        filename="dataset_stride36.tfrec",
        windowSizeMS=36,
        behaviorData=behavior_data,
        totMask=tot_mask,
        augmentation_config=None,
        enable_augmentation=False,
        oversampling_resampling=False,
        shuffle=False,
        is_interleaving_subdataset=True,
    )

    means, stds = net_1d.compute_normalization_stats(
        ds_stats["train"], max_samples=1000
    )

    # If stats look like identity for all groups, recompute from explicit non-zero spikes.
    looks_identity = True
    for g in range(len(means)):
        if not (
            np.allclose(means[g], 0.0, atol=1e-3)
            and np.allclose(stds[g], 1.0, atol=1e-2)
        ):
            looks_identity = False
            break
    if looks_identity:
        print(
            "Primary normalization stats are near identity; using fallback non-zero spike stats."
        )
        means, stds = fallback_channel_stats_from_dataset(
            ds_stats["train"],
            n_groups=net_1d.params.nGroups,
            n_channels_per_group=net_1d.params.nChannelsPerGroup,
            max_batches=80,
        )

    print("Final normalization stats summary:")
    for g in range(len(means)):
        print(f"  group{g}: mean[:4]={means[g][:4]}, std[:4]={stds[g][:4]}")

    set_norm_weights_from_stats(net_1d, means, stds)
    set_norm_weights_from_stats(net_2d, means, stds)

    batch_inputs, _ = next(iter(datasets_1d["test"].take(1)))

    wave = get_nonzero_waveform_from_dataset(datasets_1d["test"], group_id=0)
    wave_tf = tf.convert_to_tensor(wave[None, ...], dtype=tf.float32)

    after_1d = net_1d.spikeNets[0].input_normalization(wave_tf)[0].numpy()
    after_2d = net_2d.spikeNets[0].input_normalization(wave_tf)[0].numpy()

    print(
        f"Waveform mean/std before: {np.mean(wave):.4f}/{np.std(wave):.4f} | "
        f"after1d: {np.mean(after_1d):.4f}/{np.std(after_1d):.4f} | "
        f"after2d: {np.mean(after_2d):.4f}/{np.std(after_2d):.4f}"
    )

    norm_fig = out_dir / "01_normalization_spikenet1d_2d.png"
    plot_normalization_compare(str(norm_fig), wave, after_1d, after_2d)

    stage = build_masks_and_features(net_1d, batch_inputs)
    mask_fig = out_dir / "02_masking_flow_from_waveforms_to_sequence.png"
    plot_masking_flow(str(mask_fig), stage)

    print("Done.")
    print(f"Saved: {norm_fig}")
    print(f"Saved: {mask_fig}")


if __name__ == "__main__":
    main()
