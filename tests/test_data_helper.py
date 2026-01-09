import os
from unittest.mock import MagicMock

import numpy as np

from unittest.mock import MagicMock
from neuroencoders.utils.global_classes import DataHelper, Project, is_in_zone, ZONEDEF


def test_project_init(temp_project_dir):
    project_dir, xml_path = temp_project_dir

    prj = Project(str(xml_path), nameExp="Network")

    assert prj.xml == str(xml_path)
    assert prj.baseName == str(xml_path)[:-4]
    assert prj.experimentPath == os.path.join(project_dir, "Network")


def test_is_in_zone():
    # ZoneDef: [[x_min, x_max], [y_min, y_max]]
    # shock zone: [[0, 0.35], [0, 0.43]]
    zone = ZONEDEF[0]

    # Point inside
    p_in = np.array([[0.1, 0.1]])
    assert is_in_zone(p_in, zone).all()

    # Point outside
    p_out = np.array([[0.5, 0.5]])
    assert not is_in_zone(p_out, zone).any()


def test_dist2wall():
    """Test DataHelper.dist2wall method with mocked instance.

    Creates a minimal DataHelper instance without calling __init__ to test
    the dist2wall functionality. The test sets up positions in different maze
    regions (lower, upper, and other) to verify that the method correctly
    computes distances to maze boundaries. The maze is defined with specific
    coordinates forming a polygon with a hole in the middle, and get_maze_limits
    is mocked to avoid auto-detection logic affecting the test outcome.
    """
    dh = DataHelper.__new__(DataHelper)
    dh.positions = np.array(
        [
            [0.1, 0.1],  # Lower region
            [0.9, 0.1],  # Upper region
            [0.5, 0.8],  # Other
        ]
    )
    dh.old_positions = dh.positions

    dh.lower_x = 0.35
    dh.upper_x = 0.65
    dh.ylim = 0.75

    dh.get_maze_limits = MagicMock(return_value=([0.35, 0.65], 0.75))

    dh.maze_coords = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [dh.upper_x, 0],
        [dh.upper_x, dh.ylim],
        [dh.lower_x, dh.ylim],
        [dh.lower_x, 0],
        [0, 0],
    ]

    pos = np.array([[0.5, 0.8]])
    dist = dh.dist2wall(pos, show=False)
    
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (1,)
    assert dist[0] >= 0


def test_helper_linearization_target():
    # Mock l_function
    def mock_l_func(pos):
        # map x coordinate to linear pos
        return None, pos[:, 0]

    dh = DataHelper.__new__(DataHelper)
    dh.positions = np.array([[0.2, 0.2]])
    dh.target = "lin"

    # Mock get_maze_limits to avoid it failing
    dh.get_maze_limits = MagicMock(return_value=[0, 1])

    res = dh.get_true_target(l_function=mock_l_func, in_place=False)

    assert res.shape == (1,)
    assert res[0] == 0.2


def test_data_helper_persistence(temp_project_dir):
    project_dir, _ = temp_project_dir
    save_path = os.path.join(project_dir, "dh.pkl")

    dh = DataHelper.__new__(DataHelper)
    dh.custom_attr = "hello"
    dh.positions = np.array([[1, 2]])

    # Save
    import dill as pickle

    with open(save_path, "wb") as f:
        pickle.dump(dh, f)

    # Load
    dh_loaded = DataHelper.load(save_path)

    assert dh_loaded.custom_attr == "hello"
    assert np.array_equal(dh_loaded.positions, dh.positions)
    assert getattr(dh_loaded, "_loaded_from_pickle", True)


def test_project_paths(temp_project_dir):
    project_dir, xml_path = temp_project_dir

    # Test normalization of paths
    prj = Project(xmlPath=xml_path, nameExp="TestExp")

    assert prj.experimentPath == os.path.join(project_dir, "TestExp")
    # Verify subfolders are set
    assert hasattr(prj, "experimentPath")
