import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tables

from neuroencoders.transformData.linearizer import UMazeLinearizer


@pytest.fixture
def mock_mat_file(tmp_path):
    """Creates a mock nnBehavior.mat file."""
    mat_path = tmp_path / "nnBehavior.mat"
    with tables.open_file(str(mat_path), mode="w") as f:
        f.create_group("/", "behavior")
        # Add some mock data if needed, but UMazeLinearizer can generate its own
    return str(tmp_path)


def test_linearizer_initialization_with_file(mock_mat_file):
    """Test that UMazeLinearizer initializes correctly when a file exists."""
    linearizer = UMazeLinearizer(folder=mock_mat_file, nb_bins=50)
    assert linearizer.nb_bins == 50
    assert hasattr(linearizer, "mazePoints")
    assert linearizer.mazePoints.shape == (50, 2)


def test_linearizer_canonical_path_generation():
    """Test the internal path generation logic."""
    # We need to mock the file check to allow initialization
    with (
        patch("os.path.exists", return_value=True),
        patch("tables.open_file") as mock_open,
    ):
        # Mocking the tables structure
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.list_nodes.return_value = []  # Force path generation

        linearizer = UMazeLinearizer(folder="/dummy/path", nb_bins=100)

        assert len(linearizer.nnPoints) > 0
        assert linearizer.nnPoints.shape[1] == 2
        # Check if it follows U-shape basics (starts and ends at bottom)
        assert linearizer.nnPoints[0][1] < 0.1
        assert linearizer.nnPoints[-1][1] < 0.1


def test_apply_linearization():
    """Test projecting 2D points onto the linear path."""
    with (
        patch("os.path.exists", return_value=True),
        patch("tables.open_file") as mock_open,
    ):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.list_nodes.return_value = []

        linearizer = UMazeLinearizer(folder="/dummy/path", nb_bins=100)

        # Test points
        points = np.array(
            [
                [0.15, 0.5],  # On left arm
                [0.5, 0.9],  # On top bridge
                [0.85, 0.5],  # On right arm
            ]
        )

        projected, linear_vals = linearizer.apply_linearization(points, keops=False)

        assert projected.shape == (3, 2)
        assert linear_vals.shape == (3,)
        # Linear values should be roughly increasing along the U-path
        assert linear_vals[0] < linear_vals[1] < linear_vals[2]


@pytest.mark.skipif(
    os.environ.get("SKIP_KEOPS_TESTS") == "1", reason="Skipping KeOps tests"
)
def test_pykeops_linearization():
    """Test KeOps-based linearization if available."""
    try:
        import pykeops

        print(
            f"PyKeOps v{pykeops.__version__} is available, running KeOps linearization test."
        )
    except ImportError:
        pytest.skip("PyKeOps not installed")

    with (
        patch("os.path.exists", return_value=True),
        patch("tables.open_file") as mock_open,
    ):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.list_nodes.return_value = []

        linearizer = UMazeLinearizer(folder="/dummy/path", nb_bins=100)
        points = np.random.rand(10, 2).astype(np.float32)

        proj_np, lin_np = linearizer.apply_linearization(points, keops=False)
        proj_keops, lin_keops = linearizer.pykeops_linearization(points)

        np.testing.assert_allclose(proj_np, proj_keops, atol=1e-5)
        np.testing.assert_allclose(lin_np, lin_keops, atol=1e-5)
