import pytest
import torch
import numpy as np
from neuroencoders.fullEncoder.nnUtils_torch import (
    LinearizationLayer,
    GaussianHeatmapLayer,
    GaussianHeatmapLosses,
    ContrastiveLossLayer,
)


def test_linearization_layer():
    # Simple linear track: (0,0) -> (1,0)
    maze_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    ts_proj = np.array([0.0, 0.5, 1.0])

    linearizer = LinearizationLayer(maze_points, ts_proj)

    # Test point close to start
    test_pt = torch.tensor([[0.1, 0.1]])  # Should map to (0,0) -> 0.0
    proj, lin = linearizer(test_pt)

    # Distance to (0,0) is 0.02, to (0.5,0) is 0.4^2 + 0.1^2 = 0.17
    # So should map to index 0
    assert torch.allclose(proj, torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(lin, torch.tensor([0.0]))

    # Test point close to end
    test_pt2 = torch.tensor([[0.9, 0.0]])  # Should map to (1,0) -> 1.0
    proj2, lin2 = linearizer(test_pt2)
    assert torch.allclose(proj2, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(lin2, torch.tensor([1.0]))


def test_gaussian_heatmap_layer():
    grid_size = (10, 10)
    std = 1.0
    layer = GaussianHeatmapLayer(grid_size=grid_size, std=std)

    # Target in middle of grid
    true_pos = torch.tensor([[0.5, 0.5]])  # Corresponds to index (5, 5)

    # Generate heatmap
    heatmap = layer.gaussian_heatmap_targets(true_pos)

    # Shape check
    assert heatmap.shape == (1, 10, 10)

    # Max value should be at (5, 5) or nearby
    # (0.5 * 10 = 5.0). Grid indices are 0..9.
    # Closest indices are 5.

    max_idx = torch.argmax(heatmap.view(-1))
    # 5 * 10 + 5 = 55
    assert max_idx.item() == 55

    # Test forward (reshape)
    flat_input = torch.randn(2, 100)
    out = layer(flat_input, flatten=False)
    assert out.shape == (2, 10, 10)


def test_gaussian_heatmap_losses():
    loss_fn = GaussianHeatmapLosses(loss_type="mse")

    logits = torch.zeros(2, 10, 10)
    targets = torch.zeros(2, 10, 10)

    # MSE of zeros should be 0
    loss = loss_fn({"logits": logits, "targets": targets})
    assert loss.item() == 0.0

    # Test safe_kl
    loss_fn_kl = GaussianHeatmapLosses(loss_type="safe_kl")
    # Perfect match (uniform)
    loss_kl = loss_fn_kl({"logits": logits, "targets": torch.ones(2, 10, 10) / 100})
    # Softmax of 0s is uniform. Target is uniform. KL should be 0.
    assert abs(loss_kl.item()) < 1e-6


def test_contrastive_loss():
    loss_fn = ContrastiveLossLayer()

    # Pred = Target -> Loss 0
    p1 = torch.tensor([[0.5, 0.5]])
    p2 = torch.tensor([[0.5, 0.5]])
    loss = loss_fn([p1, p2])
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Pred != Target
    p3 = torch.tensor([[0.0, 0.0]])
    # Dist = sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707
    # Loss = mean(dist^2) = 0.5
    loss2 = loss_fn([p1, p3])
    assert loss2.item() == pytest.approx(0.5, abs=1e-5)
