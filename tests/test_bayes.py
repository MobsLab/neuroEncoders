import os
from unittest.mock import MagicMock, patch

from neuroencoders.simpleBayes.decode_bayes import Trainer


@pytest.fixture
def mock_trainer_deps():
    with patch(
        "neuroencoders.importData.import_clusters.load_spike_sorting"
    ) as mock_load:
        mock_load.return_value = {
            "Spike_labels": [np.array([[1], [0]])],
            "Spike_times": [np.array([[0.1], [0.2]])],
            "Spike_positions": [np.array([[0.5, 0.5], [0.6, 0.6]])],
        }
        yield mock_load


def test_trainer_init(mock_trainer_deps, temp_project_dir):
    project_dir, xml_path = temp_project_dir
    project = MagicMock()
    project.experimentPath = project_dir
    project.folderResult = os.path.join(project_dir, "results")
    project.folderResultSleep = os.path.join(project_dir, "results_Sleep")

    trainer = Trainer(projectPath=project, phase="pre")

    assert trainer.phase == "pre"
    assert trainer.projectPath == project
    assert os.path.exists(trainer.folderResult)


def test_build_occupation_map(mock_trainer_deps, temp_project_dir):
    project_dir, xml_path = temp_project_dir
    project = MagicMock()
    project.experimentPath = project_dir

    trainer = Trainer(projectPath=project)

    # Create some mock positions
    positions = np.random.rand(100, 2)

    inv_occ, occ, grid = trainer._build_occupation_map(positions)

    assert occ.shape == (trainer.GRID_H, trainer.GRID_W)
    assert inv_occ.shape == (trainer.GRID_H, trainer.GRID_W)
    # Check that forbidden regions are masked in inverse occupation
    # (assuming default MAZE_COORDS)
    # Gap is 0.35 to 0.65, Y < 0.75.
    # Meshgrid 'xy' indexing: X is columns, Y is rows.
    # Let's check a point in the gap: (0.5, 0.1)

    # grid[0] is Xc, grid[1] is Yc
    # Find index closest to (0.5, 0.1)
    x_idx = np.argmin(np.abs(grid[0][0, :] - 0.5))
    y_idx = np.argmin(np.abs(grid[1][:, 0] - 0.1))

    assert inv_occ[y_idx, x_idx] == 0.0  # Should be masked


def test_compute_rate_function(mock_trainer_deps, temp_project_dir):
    project_dir, xml_path = temp_project_dir
    project = MagicMock()
    project.experimentPath = project_dir

    trainer = Trainer(projectPath=project)
    # Set bandwidth explicitly for testing
    trainer.config.bandwidth = 0.1

    spike_positions = np.random.rand(50, 2)
    grid_feature = [trainer.Xc_np, trainer.Yc_np]
    final_occ = np.ones((trainer.GRID_H, trainer.GRID_W))

    rate_map = trainer._compute_rate_function(
        spike_positions, grid_feature, final_occ, len(spike_positions), 100.0
    )

    assert rate_map.shape == (trainer.GRID_H, trainer.GRID_W)
    assert not np.isnan(rate_map).any()


def test_align_speed_filters(mock_trainer_deps, temp_project_dir):
    project_dir, xml_path = temp_project_dir
    project = MagicMock()
    project.experimentPath = project_dir

    trainer = Trainer(projectPath=project)

    behaviorData = {
        "positionTime": np.array([[0.1], [0.2], [0.3]]),
        "Times": {"speedFilter": np.array([True, False, True])},
    }

    # Trainer clusterData is mocked by fixture: Spike_times: [0.1, 0.2]
    speed_filters = trainer._align_speed_filters(behaviorData)

    assert len(speed_filters) == 1
    # Spike at 0.1 should match pos at 0.1 -> speed True (1)
    # Spike at 0.2 should match pos at 0.2 -> speed False (0)
    assert speed_filters[0][0] == 1
    assert speed_filters[0][1] == 0
