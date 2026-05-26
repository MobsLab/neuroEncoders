from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pynapple import IntervalSet, Tsd, TsdFrame

from neuroencoders.utils.MOBS_Functions import (
    Load_Behav,
    dict_to_dataframe,
    restrict_path_for_experiment,
)
from neuroencoders.utils.backend import pd


@pytest.fixture
def mock_behav_data():
    return {
        "tsdX": {
            "data": np.array([[1, 2, 3]]),
            "t": np.array([[0.1, 0.2, 0.3]]),
        },
        "Pos": np.array([[0.1, 1, 2, 0], [0.2, 3, 4, 1]]),
        "EpochSession": {
            "dtype": MagicMock(fields={"keys": lambda: ["pre", "post"]}),
            "pre": np.array([[0, 1]]),
            "post": np.array([[1, 2]]),
        },
        "TTLInfo": {
            "StartSession": np.array([[0]]),
            "StopSession": np.array([[1]]),
        },
        "__header__": b"MATLAB 5.0",
        "__version__": "1.0",
        "__globals__": [],
    }


@patch("scipy.io.loadmat")
def test_load_behav(mock_loadmat, mock_behav_data):
    mock_loadmat.return_value = mock_behav_data

    # We need to mock Make_Epoch because it uses complex numpy dtype fields
    with patch("neuroencoders.utils.MOBS_Functions.Make_Epoch") as mock_make_epoch:

        def fake_make_epoch(struc, dic, key, **kwargs):
            if key == "Session":
                dic[key] = {
                    "pre": IntervalSet(0, 1e6),
                    "post": IntervalSet(1e6, 2e6),
                    "TestPre": IntervalSet(2e6, 3e6),
                }
            else:
                dic[key] = IntervalSet(0, 1e6)

        mock_make_epoch.side_effect = fake_make_epoch

        behav = Load_Behav("dummy_path")

        assert "Tracking" in behav
        assert "Epoch" in behav
        assert "Other" in behav

        assert "X" in behav["Tracking"]
        assert isinstance(behav["Tracking"]["X"], Tsd)
        assert "Pos" in behav["Tracking"]
        assert isinstance(behav["Tracking"]["Pos"], TsdFrame)


def test_restrict_path_for_experiment():
    df = pd.DataFrame(
        {
            "name": ["Mouse001", "Mouse002", "Mouse003"],
            "group": ["A", "B", "A"],
            "Session": ["S1", "S2", "S1"],
            "Treatment": ["T1", "T2", "T1"],
        }
    )

    # Test Group filter
    res = restrict_path_for_experiment(df, "Group", "A")
    assert len(res) == 2
    assert all(res["group"] == "A")

    # Test nMice filter
    res = restrict_path_for_experiment(df, "nMice", 1)
    assert len(res) == 1
    assert res.iloc[0]["name"] == "Mouse001"

    # Test Session filter
    res = restrict_path_for_experiment(df, "Session", "S2")
    assert len(res) == 1
    assert res.iloc[0]["Session"] == "S2"

    # Test Treatment filter
    res = restrict_path_for_experiment(df, "Treatment", "T2")
    assert len(res) == 1
    assert res.iloc[0]["Treatment"] == "T2"

    # Test 'all' filter
    res = restrict_path_for_experiment(df, "all", None)
    assert len(res) == 3


def test_dict_to_dataframe():
    data_dict = {
        "path": ["/path1", "/path2"],
        "name": ["N1", "N2"],
        "manipe": ["M1", "M2"],
        "group": ["G1", "G2"],
    }
    df = dict_to_dataframe(data_dict)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns
    assert df.iloc[0]["name"] == "N1"
