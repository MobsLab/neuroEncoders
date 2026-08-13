import os
from unittest.mock import MagicMock

import numpy as np
import scipy.io
from pynapple import Tsd

from neuroencoders.importData.compareSpikeFiltering import WaveFormComparator
from neuroencoders.utils.global_classes import ZONEDEF, DataHelper, Project, is_in_zone
from neuroencoders.utils.wrappers import loadLFPData


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


def test_load_lfp_data_lazy(tmp_path):
    session_dir = tmp_path / "session"
    lfp_dir = session_dir / "LFPData"
    channels_dir = session_dir / "ChannelsToAnalyse"
    lfp_dir.mkdir(parents=True)
    channels_dir.mkdir(parents=True)

    scipy.io.savemat(channels_dir / "Bulb_deep.mat", {"channel": np.array([1])})

    time = np.linspace(0.0, 4.0, 5, endpoint=False)
    data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    scipy.io.savemat(lfp_dir / "LFP1.mat", {"LFP": {"t": time * 1e4, "data": data}})

    loaded, channels = loadLFPData(str(session_dir), lazy=True)

    assert channels["Bulb_deep"] == "LFP1.mat"
    signal = loaded["Bulb_deep"]
    if hasattr(signal, "as_tsd"):
        signal = signal.as_tsd()

    assert isinstance(signal, Tsd)
    assert np.allclose(signal.index.values[:3], time[:3])
    assert np.allclose(signal.values[:3], data[:3])


def test_waveform_comparator_lazy_memmap(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    xml_path = session_dir / "test.xml"
    xml_path.write_text(
        """
        <root>
          <spikeDetection>
            <channelGroups>
              <group>
                <channels>
                  <channel>0</channel>
                  <channel>1</channel>
                </channels>
              </group>
            </channelGroups>
          </spikeDetection>
          <acquisitionSystem>
            <samplingRate>20000</samplingRate>
            <nChannels>2</nChannels>
          </acquisitionSystem>
        </root>
        """
    )
    dat_path = session_dir / "test.dat"
    fil_path = session_dir / "test.fil"
    dat_values = np.arange(20, dtype=np.int16).reshape(10, 2)
    dat_values.tofile(dat_path)
    fil_values = dat_values + 1
    fil_values.tofile(fil_path)

    project = Project(str(xml_path), datPath=str(dat_path), nameExp="Network")
    comparator = WaveFormComparator.__new__(WaveFormComparator)
    comparator.projectPath = project
    comparator.samplingRate = 20000.0
    comparator.number_timeSteps = dat_values.shape[0]
    comparator.memmapData = None
    comparator.memmapFil = None

    comparator.load_memmap()
    first_window = comparator.read_window(0.0, 0.0005, channels=[0, 1])
    assert first_window.shape[0] > 0
    assert first_window.shape[1] == 2
    assert first_window[0, 0] == 0
    assert np.allclose(first_window[:, 0], dat_values[: first_window.shape[0], 0])

    filtered_window = comparator.read_window(
        0.0, 0.0005, channels=[0], use_filtered=True
    )
    assert filtered_window.shape[1] == 1
    assert filtered_window[0, 0] == 1
