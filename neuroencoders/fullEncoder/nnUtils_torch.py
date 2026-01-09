import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpikeNet(nn.Module):
    """
    Convolutional network for spike sequences.
    Input: (Batch, N_Channels, Time) or something similar, tailored to match TF input.
    TF input was (Batch, Channels, Time) but Conv2D in TF is (Batch, H, W, C).
    Wait, let's re-read the TF code.
    TF `spikeNet`:
       Input is `input`. `expand_dims(input, axis=3)`.
       TF Conv2D(8, [2, 3]) -> kernel size (2,3).
       Input shape validation says (batch, time, channels).
       So expanded is (batch, time, channels, 1).
       Conv2D filter (2,3) usually means (height, width).
       So it convolves over (Time, Channels).

    This is slightly unusual. Standard Conv2D in PyTorch expects (Batch, C, H, W).
    If we map Time->H, Channels->W, Input Channels->1.
    """

    def __init__(
        self,
        nChannels=4,
        nFeatures=128,
        number="",
        reduce_dense=False,
        no_cnn=False,
        batch_normalization=True,
    ):
        super().__init__()
        self.nChannels = nChannels
        self.nFeatures = nFeatures
        self.reduce_dense = reduce_dense
        self.no_cnn = no_cnn
        self.batch_normalization = batch_normalization

        if not self.no_cnn:
            # Conv layers
            # In TF: Conv2D(8, [2, 3], padding="same")
            # Kernel (2, 3) over (Time, Channels).
            # Note: TF "same" padding is tricky to replicate exactly if strides > 1 or odd kernels.
            # Here kernel is (2,3).

            # We will treat input as (Batch, 1, Time, Channels)
            self.conv1 = nn.Conv2d(1, 8, kernel_size=(2, 3), padding="same")
            self.conv2 = nn.Conv2d(8, 16, kernel_size=(2, 3), padding="same")
            self.conv3 = nn.Conv2d(16, 32, kernel_size=(2, 3), padding="same")

            # MaxPool (1,2) -> Pool over Channels dimension only?
            # TF: MaxPool2D([1, 2], [1, 2], padding="same")
            # Pool size (1, 2), Strides (1, 2).
            # This reduces the Channel dimension by factor of 2.
            self.pool1 = nn.MaxPool2d(
                kernel_size=(1, 2), stride=(1, 2), ceil_mode=True
            )  # padding same-ish
            self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), ceil_mode=True)
            self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), ceil_mode=True)

            if self.batch_normalization:
                self.bn1 = nn.BatchNorm2d(8)
                self.bn2 = nn.BatchNorm2d(16)
                self.bn3 = nn.BatchNorm2d(32)

        # Dense layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)

        # We need to calculate the input feature size for the dense layer dynamically or hardcode if standard
        # For now, we will assume it is dynamically calculated or we use LazyLinear
        # But for robustness, let's use LazyLinear or similar if available, else Linear.
        # Since we don't know the exact Time dimension after pooling (Time is not pooled),
        # and Channels is pooled 3 times by factor 2.

        self.dense1 = nn.LazyLinear(nFeatures)
        self.dense2 = nn.Linear(nFeatures, nFeatures)
        # Final output layer
        self.dense3 = nn.Linear(nFeatures, nFeatures)

    def forward(self, x):
        # Input x: (Batch, nChannels, Time) or (Batch, Time, nChannels)?
        # TF Code: "Expected input shape (batch, time, channels)"
        # x is (Batch, Time, Channels)

        if self.no_cnn:
            x = self.flatten(x)
            x = self.dense3(x)
            x = self.dropout(x)
            return x

        # Torch expects (Batch, C_in, H, W). Let H=Time, W=Channels.
        # Add channel dim: (Batch, 1, Time, Channels)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        if self.batch_normalization:
            x = self.bn3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        if not self.reduce_dense:
            x = F.relu(self.dense1(x))
            x = self.dropout(x)
            x = F.relu(self.dense2(x))
            x = F.relu(self.dense3(x))
        else:
            x = F.relu(self.dense3(x))
            x = self.dropout(x)

        return x


class SpikeNet1D(nn.Module):
    """
    Refined Spike Encoder.
    Input shape: (Batch, Channels, Time) -> e.g., (128, 6, 32)
    This version transposes the input so Conv1D operates on the Time axis (32)
    while keeping the 6 channels separate (by processing them in batch).
    """

    def __init__(
        self, nChannels=4, nFeatures=128, dropout_rate=0.2, batch_normalization=True
    ):
        super().__init__()
        self.nChannels = nChannels
        self.nFeatures = nFeatures
        self.batch_normalization = batch_normalization

        # Backbone (Shared Weights)
        # Conv1D in Torch: (Batch, C_in, L_in) -> we want to convolve over Time.
        # TF: Conv1D(16, 3, padding="same").
        # So input channels=1, output=16.
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(64)

        # Global Average Pooling -> mean over the remaining time dimension

        self.dropout = nn.Dropout(dropout_rate)

        # Dense Fusion
        # Input dim = nChannels * 64
        self.dense_fusion = nn.Linear(nChannels * 64, nFeatures * 2)
        self.dense_out = nn.Linear(nFeatures * 2, nFeatures)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        B, C, T = x.shape

        # Reshape to (Batch * Channels, 1, Time) to process each channel independently
        x = x.view(B * C, 1, T)

        # Layer 1
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        if self.batch_normalization:
            x = self.bn3(x)
        x = F.relu(x)

        # Global Average Pool
        # x is (B*C, 64, Time_Reduced)
        x = x.mean(dim=-1)  # (B*C, 64)

        # Concatenate channels back
        # (B, C*64)
        x = x.view(B, C * 64)

        # Dense Fusion
        x = self.dense_fusion(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Out
        x = self.dense_out(x)
        return x


class GroupAttentionFusion(nn.Module):
    """
    Fuses features from multiple spike groups using Self-Attention.
    """

    def __init__(self, n_groups, embed_dim, num_heads=4):
        super().__init__()
        self.n_groups = n_groups
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Learnable positional embedding for each group ID: (1, 1, n_groups, embed_dim)
        # In PyTorch parameter: (1, 1, n_groups, embed_dim)
        self.group_embeddings = nn.Parameter(torch.randn(1, 1, n_groups, embed_dim))

    def forward(self, inputs, mask=None):
        # inputs: List of tensors, each (Batch, Time, Features) -> we want stack -> (Batch, Time, Groups, Features)
        # Note: Time here could be "maxNbSpikes".

        x = torch.stack(inputs, dim=2)  # (B, T, G, F)

        # Add embeddings
        x = x + self.group_embeddings  # Broadcasts

        B, T, G, F = x.shape
        # Flatten B and T for MHA: (B*T, G, F)
        x_reshaped = x.view(B * T, G, F)

        attn_mask = None
        key_padding_mask = None

        if mask is not None:
            # mask shape (Batch, max_nSpikes, nGroups)?
            # TF code: mask is (Batch, nGroups) expanded to match dimensions?
            # Let's check TF code:
            # "mask comes in as shape (Batch, max(nSpikes), n_groups)"
            # "reshape to (Batch*Time, n_groups)"
            # In PyTorch MultiheadAttention, key_padding_mask expected (N, S) where True is ignore.
            # TF is_active (True if valid). So PyTorch padding_mask should be ~is_active (True if invalid).

            mask_reshaped = mask.view(B * T, G)
            # Inverse boolean for PyTorch padding mask
            key_padding_mask = ~mask_reshaped  # (B*T, G)

        attn_out, _ = self.mha(
            x_reshaped, x_reshaped, x_reshaped, key_padding_mask=key_padding_mask
        )

        x_reshaped = self.norm(x_reshaped + self.dropout(attn_out))

        # Reshape back: (B, T, G*F)
        output = x_reshaped.view(B, T, G * F)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, Feature)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_heads=8,
        ff_dim1=256,
        ff_dim2=64,
        dropout_rate=0.5,
        residual=True,
    ):
        super().__init__()
        self.residual = residual
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim1), nn.ReLU(), nn.Linear(ff_dim1, ff_dim2)
        )
        # Note: if ff_dim2 != d_model, the residual connection dimension won't match if we do x + ff_output.
        # But in TF code: "TransformerEncoderBlock... output maintains info but with ff_dim2".
        # Let's check TF code closely.
        # "x = self.norm2(x + ff_output)" -> This implies ff_output has same shape as x.
        # So ff_dim2 MUST equal d_model for the residual to work naturally, OR there is a projection.
        # In TF code: `self.ff_layer2 = Dense(self.ff_dim2)`.
        # And `x = self.norm2(x + ff_output)`.
        # So yes, ff_dim2 MUST be equal to d_model for valid add.
        # Unless the `d_model` passed to constructor IS `d_model`, and `ff_dim2` is intended to be output size?
        # Re-reading TF `build`: "if feature_dim != self.d_model: raise..."
        # So input is d_model.
        # The output of this block is `ff_dim2` size.
        # BUT `x + ff_output` is done. `x` is `d_model` size.
        # Thus `ff_dim2` MUST equal `d_model`.

        self.norm2 = nn.LayerNorm(ff_dim2)

    def forward(self, x, mask=None):
        # mask: (Batch, SeqLen) - True for valid.

        x_norm = self.norm1(x)

        key_padding_mask = None
        if mask is not None:
            # In PyTorch: True means ignore (pad).
            key_padding_mask = ~mask.bool()

        attn_out, _ = self.mha(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        attn_out = self.dropout1(attn_out)

        if self.residual:
            x = x + attn_out

        ff_out = self.ff(x)
        # Final residual
        # Note: if ff_dim2 != d_model, this will fail. Assuming they are same.
        x = self.norm2(x + ff_out)
        return x


class MaskedGlobalAveragePooling1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        # x: (Batch, SeqLen, Features)
        # mask: (Batch, SeqLen) - 1 for valid, 0 for pad

        if mask is None:
            return x.mean(dim=1)

        mask = mask.unsqueeze(-1)  # (B, S, 1)
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)
        count_x = mask.sum(dim=1)

        return sum_x / count_x.clamp(min=1.0)


class LinearizationLayer(nn.Module):
    """
    A simple layer to linearize Euclidean data into a maze-like linear track.
    """

    def __init__(self, maze_points, ts_proj, device="cpu"):
        super().__init__()
        self.device = device
        # Register as buffers to be saved with state_dict but not trained
        self.register_buffer(
            "maze_points", torch.tensor(maze_points, dtype=torch.float32)
        )
        self.register_buffer("ts_proj", torch.tensor(ts_proj, dtype=torch.float32))

    def forward(self, euclidean_data):
        # euclidean_data: (Batch, 2)
        # maze_points: (J, 2)

        # Compute distances: ||x - p||^2
        # (Batch, 1, 2) - (1, J, 2) -> (Batch, J, 2)
        diff = euclidean_data.unsqueeze(1) - self.maze_points.unsqueeze(0)
        dists = torch.sum(diff**2, dim=2)  # (Batch, J)

        # Argmin to find closest point
        min_indices = torch.argmin(dists, dim=1)  # (Batch,)

        # Gather projected points and linear positions
        projected_pos = self.maze_points[min_indices]  # (Batch, 2)
        linear_pos = self.ts_proj[min_indices]  # (Batch,)

        return projected_pos, linear_pos


class GaussianHeatmapLayer(nn.Module):
    """
    Layer to produce a Gaussian heatmap from 2D positions or decode 2D positions from a heatmap.
    """

    def __init__(self, grid_size=(40, 40), std=2.0, device="cpu"):
        super().__init__()
        self.grid_size = grid_size
        self.std = std
        self.device = device

        # Create grid coordinates
        H, W = grid_size
        # Assuming grid covers [0, 1] x [0, 1] ??
        # Or [0, W] x [0, H]?
        # TF code usually implies scaling. Let's assume normalized 0-1 or pixel coordinates.
        # If input pos is 0-1, we need to map to grid indices.
        # Let's assume input 'pos' is normalized [0,1].

        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        self.register_buffer("grid_xx", xx.float())
        self.register_buffer("grid_yy", yy.float())

    def gaussian_heatmap_targets(self, true_pos):
        """
        Generate target heatmaps from true positions.
        true_pos: (Batch, 2) in range [0, 1] ?
        """
        B = true_pos.shape[0]
        H, W = self.grid_size

        # Scale pos to grid coords
        # pos x -> W, pos y -> H
        # assuming pos is (x, y)
        target_x = true_pos[:, 0] * W
        target_y = true_pos[:, 1] * H

        # (Batch, H, W)
        # exp( -((x-mu_x)^2 + (y-mu_y)^2) / (2*std^2) )

        grid_x = self.grid_xx.unsqueeze(0).expand(B, -1, -1)
        grid_y = self.grid_yy.unsqueeze(0).expand(B, -1, -1)

        t_x = target_x.reshape(B, 1, 1)
        t_y = target_y.reshape(B, 1, 1)

        dist_sq = (grid_x - t_x) ** 2 + (grid_y - t_y) ** 2
        heatmap = torch.exp(-dist_sq / (2 * self.std**2))

        # Normalize sum to 1? Or max to 1?
        # Usually target heatmaps max is 1.
        return heatmap

    def forward(self, x, flatten=True):
        # x input is usually the Dense layer output of size H*W
        # reshape to (B, H, W)
        B = x.shape[0]
        H, W = self.grid_size
        heatmap = x.view(B, H, W)

        # Use softmax? Or sigmoid?
        # TF code: "activation=None" in dense output, then maybe applied later?
        # Usually heatmap regression uses linear output trained with MSE/BCE.
        # Let's assume raw logits.

        if flatten:
            return heatmap.view(B, -1)
        return heatmap


class GaussianHeatmapLosses(nn.Module):
    def __init__(self, loss_type="safe_kl"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, inputs):
        # inputs dict: 'logits' (Batch, H, W), 'targets' (Batch, H, W)
        logits = inputs["logits"]
        targets = inputs["targets"]

        if self.loss_type == "mse":
            return F.mse_loss(logits, targets)
        elif self.loss_type == "safe_kl":
            # Softmax logits to get probability distribution
            # KLDivLoss expects log-probabilities

            # Flatten spatial dims
            B = logits.shape[0]
            log_probs = F.log_softmax(logits.view(B, -1), dim=1)
            target_probs = targets.view(B, -1)

            # Normalize target to be proper distribution
            target_probs = target_probs / (target_probs.sum(dim=1, keepdim=True) + 1e-8)

            return F.kl_div(log_probs, target_probs, reduction="batchmean")

        return F.mse_loss(logits, targets)


class ContrastiveLossLayer(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, inputs):
        # inputs: [predicted_pos, linearized_pos]
        # This seems to implement a specific regularizer.
        # "ContrastiveLossLayer... weights the first 2 dimensions by linPos?"
        # Wait, the TF code usage was:
        # projected_pos, linear_pos = l_function(truePos)
        # regression_loss = regression_loss_layer([myoutputPos, projected_pos])

        # It encourages predicted pos to be close to the manifold (projected pos).

        pred = inputs[0]
        target = inputs[1]

        dist = F.pairwise_distance(pred, target)
        loss = torch.mean(dist**2)
        return loss
