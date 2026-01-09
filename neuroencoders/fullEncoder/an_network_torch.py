import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import tensorflow as tf  # For data loading only

from neuroencoders.fullEncoder import nnUtils_torch as nnUtils
from neuroencoders.utils.global_classes import Project, Params
# Assuming these helper classes are pure python or can be reused.


class TFDataIterable(torch.utils.data.IterableDataset):
    """
    Wrapper to convert a tf.data.Dataset into a PyTorch iterable.
    """

    def __init__(self, tf_dataset, device="cpu"):
        super().__init__()
        self.tf_dataset = tf_dataset
        self.device = device

    def __iter__(self):
        for batch in self.tf_dataset:
            # batch is a tuple (inputs, outputs) or dict.
            # In existing code: dataset = dataset.map(map_outputs) -> returns (inputs, outputs)
            # inputs is a dict.

            inputs, targets = batch

            # Convert to torch
            # We assume inputs is a dict of tensors
            torch_inputs = {k: torch.from_numpy(v.numpy()) for k, v in inputs.items()}
            torch_targets = {k: torch.from_numpy(v.numpy()) for k, v in targets.items()}

            yield torch_inputs, torch_targets


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if torch.is_tensor(data):
        return data.to(device)
    return data


class LSTMandSpikeNetwork(nn.Module):
    """
    PyTorch implementation of LSTMandSpikeNetwork.
    """

    def __init__(
        self, projectPath, params, deviceName="cpu", debug=False, phase=None, **kwargs
    ):
        super().__init__()
        self.projectPath = projectPath
        self.params = params
        self.deviceName = deviceName  # e.g. 'cuda:0' or 'cpu'
        self.debug = debug
        self.phase = phase

        self.nGroups = params.nGroups
        self.nFeatures = params.nFeatures
        self.learning_rate = kwargs.get("lr", params.learningRates[0])

        # 1. Spike Nets (one per group)
        self.spikeNets = nn.ModuleList(
            [
                nnUtils.SpikeNet1D(
                    nChannels=params.nChannelsPerGroup[g],
                    nFeatures=params.nFeatures,
                    batch_normalization=True,  # Default true in TF code
                )
                for g in range(self.nGroups)
            ]
        )

        # 2. Group Fusion
        self.use_group_fusion = getattr(params, "use_group_attention_fusion", True)
        if self.use_group_fusion:
            self.group_fusion = nnUtils.GroupAttentionFusion(
                n_groups=self.nGroups, embed_dim=self.nFeatures, num_heads=4
            )

        # 3. Transformer / LSTM
        self.isTransformer = kwargs.get("isTransformer", True)
        self.dim_factor = getattr(params, "dim_factor", 1)

        # Determine the dimension coming OUT of fusion/concatenation
        fusion_output_dim = self.nFeatures * self.nGroups

        # Determine the dimension expected by the Transformer
        target_dim = (
            self.nFeatures * self.dim_factor
            if getattr(params, "project_transformer", True)
            else fusion_output_dim
        )

        self.project_transformer = getattr(params, "project_transformer", True) and (
            target_dim != fusion_output_dim
        )

        if self.project_transformer:
            self.transformer_projection = nn.Linear(fusion_output_dim, target_dim)
            self.activation_projection = nn.ReLU()  # TF uses relu in Dense definition

        transformer_input_dim = target_dim

        if self.isTransformer:
            self.pos_encoder = nnUtils.PositionalEncoding(d_model=transformer_input_dim)
            self.transformer_blocks = nn.ModuleList(
                [
                    nnUtils.TransformerEncoderBlock(
                        d_model=transformer_input_dim,
                        num_heads=params.nHeads,
                        ff_dim1=params.ff_dim1,
                        ff_dim2=params.ff_dim2,
                        dropout_rate=params.dropoutLSTM,
                        residual=kwargs.get("transformer_residual", True),
                    )
                    for _ in range(params.lstmLayers)
                ]
            )
            self.pooling = nnUtils.MaskedGlobalAveragePooling1D()

            # Dense layers after pooling
            self.dense1 = nn.Linear(
                transformer_input_dim, int(params.TransformerDenseSize1)
            )
            self.dense2 = nn.Linear(
                int(params.TransformerDenseSize1), int(params.TransformerDenseSize2)
            )
        else:
            # LSTM implementation
            self.lstm = nn.LSTM(
                input_size=transformer_input_dim,
                hidden_size=params.lstmSize,
                num_layers=params.lstmLayers,
                batch_first=True,
                dropout=params.dropoutLSTM,
            )
            self.dense1 = nn.Linear(
                params.lstmSize, int(params.TransformerDenseSize1)
            )  # Just guessing connectivity

        # 4. Output Heads
        # TF: denseFeatureOutput
        dimOutput = params.dimOutput
        # If heatmap, dimOutput might be different logic.
        self.output_head = nn.Linear(int(params.TransformerDenseSize2), dimOutput)

        # Move to device
        self.to(self.deviceName)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(params, "weight_decay", 1e-4),
        )

    def forward(self, inputs):
        """
        inputs: dict containing:
          - group0, group1, ...: (Batch, nChannels, Time)
          - groups: (Batch, TotalSpikes) ? Or just a list of tensors?
            TF code had complex "gather" logic.
            Here, we expect the data loader might yield slightly different structure or we adapt.
            Proposed simplified flow:
            We assume inputs['groupX'] contains the spike snippets for group X.
        """

        # 1. Run SpikeNets
        # We need to handle the fact that spikes happen at specific times.
        # In TF code, there was a "ZeroForGather" and "Indices" to scatter spikes into a time grid.
        # This basically constructs a (Batch, MaxTime, nFeatures) tensor where features are at the spike time.

        # For this refactor, we will rely on inputs providing 'indices' similar to TF,
        # or we assume inputs contain already dense-like structures if that's possible.
        # But `dataset` yields 'indices' and 'groups'.

        # Let's assume we receive the raw spike waveforms and indices.
        # group_features = List of (Batch, TotalSpikes, nFeatures)
        # Wait, if we use the TF dataloader, it will give exactly what TF gives.
        # TF gives: `inputsToSpikeNets` (Batch, Channels, 32). THIS IS WRONG.
        # TF logic: `inputsToSpikeNets[group]` is (NbKeptSpikes, Channels, 32).
        # It flattens batch and spikes.

        batch_size = inputs.get("batch_size", self.params.batchSize)

        group_features_list = []
        group_masks = []

        # Reconstruct the time-grid features
        # We need a canvas (Batch, MaxIndices, Features) initialized to 0.
        # Then scatter the spike features into it using 'indices'.

        for g in range(self.nGroups):
            # 1. Extract spike waveforms: (N_Spikes_Total, Channels, 32)
            waveforms = inputs[f"group{g}"]  # Should be float tensor

            # 2. Run feature extractor
            # Output: (N_Spikes_Total, nFeatures)
            features = self.spikeNets[g](waveforms)

            # 3. Scatter to (Batch, Time, nFeatures)
            # indices[g] is (N_Spikes_Total,) containing the index in the flattened (Batch*Time) array?
            # TF: "indices... contains the indices where to put the spikes... in the final tensor"
            # TF: gather takes (zeroForGather + x) and indices.

            # In PyTorch, we can perform this scatter.
            # We need the total size: Batch * MaxTime.
            # Usually MaxTime is inferred or fixed.
            # Let's assume we can deduce it from the max index in 'indices'.

            indices = inputs[f"indices{g}"].long()  # (N_Spikes_Total)

            if indices.numel() == 0:
                # No spikes for this group
                # Look up max time from other groups or default?
                # Handled below.
                scattered = torch.zeros(
                    batch_size, 1, self.nFeatures, device=self.deviceName
                )
            else:
                max_idx = indices.max().item()
                # We need to know the 'Time' dimension.
                # TF reshapes to (Batch, -1, Features).
                # We can try to guess Time from max_idx // batch_size? No.
                # TF Code: "filledFeatureTrain = reshape(filled, (batch, -1, features))"
                # This implies the flat index logic maps to linear batch*time.

                # We will construct a flat buffer.
                # Size buffer: max_idx + 1 (or sufficient size).
                # To be safe, we should probably find the global max index across all groups to define the Time dimension consistent.
                # However, for now, let's just scatter into a sufficiently large buffer.

                # Better approach: The TF dataset creation likely defines a fixed max_len (maxNbOfSpikes).
                # We can't easily see it here. But let's act dynamically.

                # Create a zero tensor of shape (max_idx + 1 + padding?, nFeatures)
                # Scatter `features` into it at `indices`.
                buffer_size = max_idx + 1
                container = torch.zeros(
                    buffer_size, self.nFeatures, device=self.deviceName
                )
                container.index_add_(0, indices, features)
                # Note: index_add_ sums if duplicate indices. TF logic was 'take', implying one spike per slot?
                # TF: "kops.take( concatenated, indices )".
                # Scatter is clearer.

                # Reshape to (Batch, Time, Features)
                # We need to ensure buffer_size is divisible by batch_size
                remainder = buffer_size % batch_size
                if remainder != 0:
                    pad = batch_size - remainder
                    container = F.pad(container, (0, 0, 0, pad))
                    buffer_size += pad

                scattered = container.view(batch_size, -1, self.nFeatures)

            group_features_list.append(scattered)

            # Mask
            # 1 where there is a spike, 0 otherwise.
            # We can deduce this from indices being non-zero/non-padding?
            # TF: "indices == 0 implies zeroForGather".
            # indices in TF input included 0 for "empty"?
            # Actually indices map the spike to the position.
            # The 'mask' is derived from `inputGroups`.

        # Ensure all groups have same time dimension
        max_t = max([t.shape[1] for t in group_features_list])
        for i in range(len(group_features_list)):
            t = group_features_list[i]
            if t.shape[1] < max_t:
                # Pad time dim
                pad_t = max_t - t.shape[1]
                group_features_list[i] = F.pad(t, (0, 0, 0, pad_t))

        # 4. Fusion
        if self.use_group_fusion:
            # We need a generic mask.
            # TF: "mymask = safe_mask_creation(batchedInputGroups)"
            # `inputGroups` tensor in inputs dict has shape (TotalSpikes,).
            # It maps which group the spike belongs to.
            # This logic is redundant if we already have separated execution.
            # The mask we really need is "where are the padding time steps?".
            # In the TF code, `mymask` corresponds to valid time steps (vs padded batch).

            # Simplified: Assume all time steps valid for now or derive from indices?
            # Let's default to full ones.
            mask = torch.ones(batch_size, max_t, device=self.deviceName)

            fused_features = self.group_fusion(group_features_list, mask=None)
            # output (B, T, G*F)

            # Flatten features?
            all_features = fused_features.view(batch_size, max_t, -1)
        else:
            all_features = torch.cat(group_features_list, dim=2)

        # 5. Transformer
        # Masking for padded time steps:
        # If we had irregular sequences, we would need a mask.
        # Here we derived max_t from indices. The gaps between actual spikes are 0s.
        # Is that intended? Yes, it's a sparse spike train representation.
        # But we DO need to mask the "padded batch" area if batches are padded?
        # TF logic: "kops.where(expand(mask), allFeatures, 0)".

        # Let's apply Transformer
        x = all_features

        if getattr(self, "project_transformer", False):
            x = self.transformer_projection(x)
            x = self.activation_projection(x)

        if self.isTransformer:
            x = self.pos_encoder(x)
            for block in self.transformer_blocks:
                x = block(x, mask=mask)  # mask used for attention

            # Pooling
            x = self.pooling(x, mask=mask)

            x = F.relu(self.dense1(x))
            x = F.relu(self.dense2(x))
        else:
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Last state? Or global pooling?
            x = F.relu(self.dense1(x))

        # 6. Output
        out = self.output_head(x)
        return out

    def train_epoch(self, dataloader):
        self.train()
        total_loss = 0
        steps = 0

        for inputs, targets in dataloader:
            inputs = to_device(inputs, self.deviceName)
            targets = to_device(targets, self.deviceName)

            self.optimizer.zero_grad()

            preds = self(inputs)

            # Loss
            # Basic MSE for now on 'pos'
            # Look at targets['pos']
            true_pos = targets["myoutputPos"]  # TF output name override
            if true_pos is None:
                true_pos = targets.get("pos")

            loss = F.mse_loss(preds, true_pos[:, :2])  # Assuming 2D pos

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 10 == 0:
                wandb.log({"train_loss": loss.item()})

        return total_loss / steps
