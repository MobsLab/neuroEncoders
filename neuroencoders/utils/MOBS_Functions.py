#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:28:52 2020

@author: quarantine-charenton
"""

import copy
import gc
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Union
from warnings import warn

import dill as pickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from pynapple import (
    IntervalSet,
    Ts,
    TsGroup,
    Tsd,
    TsdFrame,
    TsdTensor,
    compute_perievent,
)
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import FastICA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from statannotations.Annotator import Annotator
from tqdm import tqdm

from neuroencoders.importData.epochs_management import get_epochs_mask, inEpochsMask
from neuroencoders.importData.gui_elements import connect_points
from neuroencoders.importData.rawdata_parser import get_behavior
from neuroencoders.resultAnalysis import print_results
from neuroencoders.resultAnalysis.paper_figures import PaperFigures
from neuroencoders.transformData.linearizer import UMazeLinearizer
from neuroencoders.utils.PathForExperiments import path_for_experiments
from neuroencoders.utils.func_wrappers import timing
from neuroencoders.utils.global_classes import (
    ZONEDEF,
    ZONELABELS,
    ZONE_COLORS,
    Params,
    Project,
    SpatialConstraintsMixin,
    TuningCurvesPlotter,
    _compute_tuning_curves_for_result,
    gaussian_filter_nan,
    get_max_nb_spikes,
)
from neuroencoders.utils.global_classes import DataHelper as DataHelperClass
from neuroencoders.utils.viz_params import (
    ALL_STIMS_COLOR,
    GROUPS_PALETTE,
    RIPPLES_COLOR,
)
from neuroencoders.utils.wrappers import clean_mat_structure

EXPORT_COLS = [
    "mouse",
    "manipe",
    "mouse_name",
    "phase",
    "winMS",
    "time",
    "x",
    "y",
    "head_dir",
    "thigmo",
    "head_dir_hat",
    "thigmo_hat",
    "x_hat",
    "y_hat",
    "linear",
    "linear_hat",
    "speed",
    "is_ripples",
    "is_freezing",
    "is_stim",
    "is_fast",
    "certainty",
    "breathing_rate",
]
[EXPORT_COLS.append(f"{zone}_epoch") for zone in ZONELABELS]

PHASE_MAPPING = {
    "training": 0,
    "pre": 1,
    "full_pre": 2,
    "cond": 3,
    "post": 4,
    "extinct": 5,
}
EPOCH_MAPPING = {f"{k}_epoch": i for i, k in enumerate(ZONELABELS)}

plt.style.use("neuroencoders.mobs")


def _normalize_phase_name(phase: Optional[str]) -> str:
    """Normalize phase aliases used across reactivation analyses.

    This disambiguates active pre/post tests from pre/post sleep and keeps the
    original phase names intact when they already match the canonical names.
    """
    if phase is None:
        return ""
    if not isinstance(phase, str):
        return str(phase)

    normalized = re.sub(r"[^a-z0-9]+", "_", phase.strip().lower()).strip("_")
    aliases = {
        "pretest": "pre_test",
        "testpre": "pre_test",
        "test_pre": "pre_test",
        "pre_test": "pre_test",
        "posttest": "post_test",
        "testpost": "post_test",
        "test_post": "post_test",
        "post_test": "post_test",
        "presleep": "pre_sleep",
        "sleeppre": "pre_sleep",
        "pre_sleep": "pre_sleep",
        "postsleep": "post_sleep",
        "sleeppost": "post_sleep",
        "post_sleep": "post_sleep",
        "presleep_sws": "pre_sleep_sws",
        "postsleep_sws": "post_sleep_sws",
        "presleepsw": "pre_sleep_sws",
        "postsleepsw": "post_sleep_sws",
        "pre_sws": "pre_sleep_sws",
        "post_sws": "post_sleep_sws",
        "pre_sleep_sws": "pre_sleep_sws",
        "post_sleep_sws": "post_sleep_sws",
    }
    return aliases.get(normalized, normalized)


def _parse_session_key(session_key: str) -> tuple[str, str]:
    """Split a session key into mouse name and manipulation label."""
    if not isinstance(session_key, str) or "_" not in session_key:
        return session_key, ""
    mouse_name, manipe = session_key.rsplit("_", 1)
    return mouse_name, manipe


def _summarize_reactivation_strengths(
    rs_tsd, epochs: Dict[str, Any], event_intervals=None
):
    """Create a compact summary of reactivation strength across behavioral epochs."""
    summary: Dict[str, float] = {}

    def _safe_mean(interval):
        if interval is None:
            return np.nan
        try:
            values = np.asarray(rs_tsd.restrict(interval).values, dtype=float)
        except Exception:
            return np.nan
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    for key in (
        "pre_test",
        "pre",
        "pre_sleep",
        "cond",
        "post_test",
        "post",
        "post_sleep",
        "hab",
        "training",
        "testing",
        "sleep",
    ):
        if key in epochs and epochs[key] is not None:
            summary[key] = _safe_mean(epochs[key])

    if "pre_test" in summary and "cond" in summary:
        summary["cond_minus_pre_test"] = summary["cond"] - summary["pre_test"]
    if "post_test" in summary and "cond" in summary:
        summary["cond_minus_post_test"] = summary["cond"] - summary["post_test"]
    if "pre_sleep" in summary and "post_sleep" in summary:
        summary["sleep_delta"] = summary["post_sleep"] - summary["pre_sleep"]
    if "cond" in summary and "pre_sleep" in summary:
        summary["cond_minus_pre_sleep"] = summary["cond"] - summary["pre_sleep"]
    if "cond" in summary and "post_sleep" in summary:
        summary["cond_minus_post_sleep"] = summary["cond"] - summary["post_sleep"]

    if event_intervals:
        for name, interval in event_intervals.items():
            if interval is not None:
                summary[name] = _safe_mean(interval)

    return summary


# %% Info_LFP -> load the InfoLFP.mat file in a DataFrame with the LFPs' path


def Info_LFP(LFP_directory, Info_name="InfoLFP"):
    from os.path import join

    import pandas as pd

    # Loading .mat file

    try:
        Info_path = join(LFP_directory, Info_name + ".mat")
        Info = loadmat(Info_path, squeeze_me=True)
    except FileNotFoundError:
        from os.path import join

        LFP_directory = join(LFP_directory, "LFPData")
        Info_path = join(LFP_directory, Info_name + ".mat")
        Info = loadmat(Info_path, squeeze_me=True)
    Info = Info["InfoLFP"]

    # Getting the features

    Features = list(Info.dtype.names)

    if "channel" in Features:
        channel = Info["channel"].tolist()
        Features.remove("channel")
    else:
        channel = np.arange(0, len(Info[Features[0]].tolist()))

    LFP_Path = []

    for c in channel:
        LFP_Path.append(join(LFP_directory, "LFP" + str(c) + ".mat"))

    LFP_Path = np.transpose(LFP_Path)
    Info_LFP = np.vstack((Info[Features].tolist(), LFP_Path))
    Info_LFP = pd.DataFrame(Info_LFP, index=Features + ["path"], columns=channel)

    return Info_LFP.transpose()


# %% Load_LFP -> load LFP.mat as Tsd or TsdFrame object


def Load_LFP(LFP_path, time_unit="us", frequency=1250.0):
    if isinstance(LFP_path, str):
        try:
            LFP = loadmat(LFP_path, squeeze_me=True)
        except FileNotFoundError:
            from os.path import join

            LFP_path = join(LFP_path, "LFPData", "LFP1.mat")
            LFP = loadmat(LFP_path, squeeze_me=True)
        LFP = LFP["LFP"]
        t = LFP["t"].tolist()
        unit = (t[1] - t[0]) * frequency / 100
        t = unit * t
        data = LFP["data"].tolist()
        return Tsd(t, data, time_units=time_unit)

    else:
        channels = (LFP_path.index).tolist()
        data = []

        for n in channels:
            LFP = loadmat(LFP_path[n], squeeze_me=True)
            LFP = LFP["LFP"]
            dat = LFP["data"].tolist()
            data.append(dat)
        t = LFP["t"].tolist()
        unit = (t[1] - t[0]) * frequency / 100
        t = unit * t
        return TsdFrame(t, np.transpose(data), time_units=time_unit, columns=channels)


# %% Help function for Load_Behav


def Make_Epoch(struc, dic, key, time_unit="us", word="start"):
    try:
        if word in list(struc.dtype.fields.keys()):
            if time_unit == "us":
                struc = struc.tolist()
                # handle tuple to list conversion
                if isinstance(struc, tuple):
                    struc = list(struc)
                struc[1] *= 100  # convert to us
                struc[2] *= 100
            else:
                raise ValueError("Unsupported time unit. Use 'us' for microseconds.")
            dic[key] = IntervalSet(struc[1], struc[2], time_units=time_unit)
        else:
            dic[key] = {}
            for k in list(struc.dtype.fields.keys()):
                Make_Epoch(struc[k], dic[key], k, time_unit=time_unit, word=word)

    except AttributeError:
        Make_Epoch(struc.tolist(), dic, key)


def _parse_tracking_data(Behav_data, keys, time_unit):
    Tracking = {}

    tsd_keys = [key for key in keys if "tsd" in key]
    for key in keys:
        if "LinearDist" in key:
            tsd_keys.append(key)

    for key in tsd_keys:
        tsd_temp = Behav_data[key]

        # Robustly handle MATLAB nested structure
        dat_arr = np.atleast_1d(np.array(tsd_temp["data"]).squeeze())
        t_arr = np.atleast_1d(np.array(tsd_temp["t"]).squeeze())

        if t_arr.dtype == object and t_arr.size > 0:
            t_arr = t_arr.item() if t_arr.ndim == 0 else t_arr[0]
        if dat_arr.dtype == object and dat_arr.size > 0:
            dat_arr = dat_arr.item() if dat_arr.ndim == 0 else dat_arr[0]

        t = np.asarray(t_arr, dtype=np.float64).ravel()
        dat = np.asarray(dat_arr, dtype=np.float64)

        # Correct temporal scaling (seconds to microseconds)
        t = t * 10**6

        new_key = key.replace("tsd", "")
        Tracking[new_key] = Tsd(t, dat.T, time_units=time_unit)

    Pos_keys = [key for key in keys if "Pos" in key]
    for key in Pos_keys:
        if key in keys:
            keys.remove(key)
        Pos_temp = Behav_data[key]

        # Robustly handle matrix array wraps
        pos_arr = np.atleast_1d(np.array(Pos_temp).squeeze())
        if pos_arr.dtype == object and pos_arr.size > 0:
            pos_arr = pos_arr.item() if pos_arr.ndim == 0 else pos_arr[0]
        pos_arr = np.asarray(pos_arr, dtype=np.float64)

        t = pos_arr[:, 0] * 10**6
        d = pos_arr[:, 1:4]

        t = np.asarray(t, dtype=np.float64).ravel()
        Tracking[key] = TsdFrame(t, d, columns=["x", "y", "stim"], time_units=time_unit)

    Im = ["im_diff", "im_diffInit"]
    for key in Im:
        if key in keys:
            keys.remove(key)
            Im_temp = Behav_data[key]

            # Robustly unpack imagery data metrics
            im_arr = np.atleast_1d(np.array(Im_temp).squeeze())
            if im_arr.dtype == object and im_arr.size > 0:
                im_arr = im_arr.item() if im_arr.ndim == 0 else im_arr[0]

            Tracking[key] = pd.DataFrame(
                im_arr, columns=["times", "average change", "pixel range"]
            )

    if "MouseTemp" in keys:
        keys.remove("MouseTemp")
        Temp_temp = Behav_data["MouseTemp"]

        temp_arr = np.atleast_1d(np.array(Temp_temp).squeeze())
        if temp_arr.dtype == object and temp_arr.size > 0:
            temp_arr = temp_arr.item() if temp_arr.ndim == 0 else temp_arr[0]
        temp_arr = np.asarray(temp_arr, dtype=np.float64)

        t = temp_arr[:, 0] * 10**6
        d = temp_arr[:, 1]

        t = np.asarray(t, dtype=np.float64).ravel()
        Tracking["MouseTemp"] = Tsd(t, d, time_units=time_unit)

    return Tracking


def _parse_epoch_data(Behav_data, keys, time_unit):
    Epoch = {}
    Epoch_keys = [key for key in keys if "Epoch" in key]

    for key in Epoch_keys:
        if key in keys:
            keys.remove(key)
        Epoch_temp = Behav_data[key]
        new_key = key.replace("Epoch", "")
        Make_Epoch(struc=Epoch_temp, dic=Epoch, key=new_key, time_unit=time_unit)

    if Epoch:
        epoch_keys = list(Epoch["Session"].keys())
        print(f"Available epochs: {epoch_keys}")

        patterns = {
            "TestPre": r".*[Tt]est[Pp]re\d*.*",
            "TestPost": r".*[Tt]est[Pp]ost\d*.*",
            "Hab": r".*[Hh]ab\d*.*",
            "Cond": r".*[Cc]ond\d*.*",
            "Sleep": r".*[Ss]leep.*",
        }

        for name, pattern in patterns.items():
            matching_keys = [k for k in epoch_keys if re.match(pattern, k)]
            if matching_keys:
                Epoch["Session"][name] = Epoch["Session"][matching_keys[0]]
                for key in matching_keys[1:]:
                    Epoch["Session"][name] = Epoch["Session"][name].union(
                        Epoch["Session"][key]
                    )

        awake_keys = [k for k in epoch_keys if not re.match(r".*[Ss]leep.*", k)]
        if awake_keys:
            Epoch["Session"]["Awake"] = Epoch["Session"][awake_keys[0]]
            for key in awake_keys[1:]:
                Epoch["Session"]["Awake"] = Epoch["Session"]["Awake"].union(
                    Epoch["Session"][key]
                )

    return Epoch


def _parse_other_data(Behav_data, keys, time_unit, Tracking, Epoch):
    Other = {}

    if "tpsCatEvt" in keys and "nameCatEvt" in keys:
        keys.remove("tpsCatEvt")
        keys.remove("nameCatEvt")
        t = Behav_data["tpsCatEvt"]
        name = Behav_data["nameCatEvt"]

        # Robustly handle text / timestamps references from MATLAB
        t_arr = np.atleast_1d(np.array(t).squeeze())
        if t_arr.dtype == object and t_arr.size > 0:
            t_arr = t_arr.item() if t_arr.ndim == 0 else t_arr[0]

        name_arr = np.atleast_1d(np.array(name).squeeze())
        if name_arr.dtype == object and name_arr.size > 0:
            name_arr = name_arr.item() if name_arr.ndim == 0 else name_arr[0]

        t_final = np.atleast_1d(t_arr).ravel()
        name_final = np.atleast_1d(name_arr).ravel()
        Other["CatEvt"] = pd.DataFrame({"t": t_final, "name": name_final})

    if "TTLInfo" in keys:
        keys.remove("TTLInfo")
        TTL = Behav_data["TTLInfo"]

        # Safe extraction for TTL data structs
        start_arr = np.atleast_1d(np.array(TTL["StartSession"]).squeeze())
        stop_arr = np.atleast_1d(np.array(TTL["StopSession"]).squeeze())

        if start_arr.dtype == object and start_arr.size > 0:
            start_arr = start_arr.item() if start_arr.ndim == 0 else start_arr[0]
        if stop_arr.dtype == object and stop_arr.size > 0:
            stop_arr = stop_arr.item() if stop_arr.ndim == 0 else stop_arr[0]

        start = np.asarray(start_arr, dtype=np.float64) * 100
        stop = np.asarray(stop_arr, dtype=np.float64) * 100
        Other["TTLInfo"] = IntervalSet(start, stop, time_units=time_unit)

    if "ThousandFrames" in keys:
        keys.remove("ThousandFrames")
        data = Behav_data["ThousandFrames"]
        TF = {}
        Nb_session = len(data)
        Session_name = list(Epoch["Session"].keys())

        for n in range(Nb_session):
            # Safe parsing for array profiles nested deep inside a cell list
            session_data = data[n]
            if isinstance(session_data, np.ndarray) and session_data.dtype == object:
                session_data = (
                    session_data.item() if session_data.ndim == 0 else session_data[0]
                )

            data_temp = session_data["tsd"]
            if isinstance(data_temp, np.ndarray) and data_temp.dtype == object:
                data_temp = data_temp.item() if data_temp.ndim == 0 else data_temp[0]

            t_raw = data_temp["t"]
            t_arr = np.atleast_1d(np.array(t_raw).squeeze())
            if t_arr.dtype == object and t_arr.size > 0:
                t_arr = t_arr.item() if t_arr.ndim == 0 else t_arr[0]

            t = np.asarray(t_arr, dtype=np.float64) * 100

            if n < len(Session_name):
                lbl = Session_name[n]
            else:
                lbl = f"Session_{n + 1}"

            TF[lbl] = Ts(t, time_units=time_unit)
        Other["ThousandFrames"] = TF

    if "GotFrame" in keys:
        keys.remove("GotFrame")

        gf_arr = np.atleast_1d(np.array(Behav_data["GotFrame"]).squeeze())
        if gf_arr.dtype == object and gf_arr.size > 0:
            gf_arr = gf_arr.item() if gf_arr.ndim == 0 else gf_arr[0]

        GF = np.transpose(gf_arr.astype(bool))
        t = None
        for k in ["X", "x", "Xpos", "pos"]:
            if k in Tracking:
                t = Tracking[k].times()
                break
        if t is not None:
            Other["GotFrame"] = Tsd(t, GF, time_units=time_unit)

    ZI_keys = [key for key in keys if "ZoneIndices" in key]
    for key in ZI_keys:
        keys.remove(key)
        Z_temp = Behav_data[key]

        # Handle cases where ZoneIndices array is inside an object array block
        if isinstance(Z_temp, np.ndarray) and Z_temp.dtype == object:
            Z_temp = Z_temp.item() if Z_temp.ndim == 0 else Z_temp[0]

        Z = {}
        names = list(Z_temp.dtype.fields.keys())
        for n in names:
            val_arr = np.atleast_1d(np.array(Z_temp[n]).squeeze())
            if val_arr.dtype == object and val_arr.size > 0:
                val_arr = val_arr.item() if val_arr.ndim == 0 else val_arr[0]
            Z[n] = val_arr.tolist()
        Other[key] = Z

    for key in list(keys):
        Other[key] = Behav_data[key]
        keys.remove(key)

    return Other


# %% BehavResources loading


def Load_Behav(Behav_path: str, time_unit="us"):
    try:
        Behav_data = loadmat(Behav_path, squeeze_me=True)
    except FileNotFoundError:
        from os.path import join

        Behav_path = join(Behav_path, "behavResources.mat")
        Behav_data = loadmat(Behav_path, squeeze_me=True)

    # Initial keys cleanup
    keys = list(Behav_data.keys())
    for internal_key in ["__header__", "__version__", "__globals__"]:
        if internal_key in keys:
            keys.remove(internal_key)

    BehavRessources = {}

    # Sequential parsing of different data types
    BehavRessources["Tracking"] = _parse_tracking_data(Behav_data, keys, time_unit)
    BehavRessources["Epoch"] = _parse_epoch_data(Behav_data, keys, time_unit)
    BehavRessources["Other"] = _parse_other_data(
        Behav_data,
        keys,
        time_unit,
        BehavRessources["Tracking"],
        BehavRessources["Epoch"],
    )

    return BehavRessources


def _ensure_list(value):
    """Ensure value is a list for consistent processing."""
    if value is None:
        return []
    elif isinstance(value, (str, int, float)):
        return [value]
    elif isinstance(value, list):
        return value
    else:
        return [value]


def _restrict_by_group(df, filter_value):
    """Filter DataFrame by group."""
    filter_values = _ensure_list(filter_value)
    group_str = " + ".join(map(str, filter_values))
    print(f"Getting groups {group_str} from Dir")

    if "group" in df.columns:
        return df[df["group"].isin(filter_values)]

    group_columns = [
        col
        for col in df.columns
        if any(
            g in str(col).lower()
            for g in ["lfp", "neurons", "ecg", "ob_resp", "ob_gamma", "pfc"]
        )
    ]
    if group_columns:
        mask = pd.Series([False] * len(df))
        for group_col in group_columns:
            for filter_val in filter_values:
                col_mask = df[group_col].apply(
                    lambda x: _check_element_match(x, filter_val)
                )
                mask |= col_mask
        return df[mask]

    print("No group columns found")
    return pd.DataFrame()


def _check_element_match(cell_value, filter_val):
    """Helper to check if a filter value matches or exists inside a cell's object."""
    if hasattr(cell_value, "values") and not isinstance(
        cell_value, (pd.Series, pd.DataFrame)
    ):
        arr = cell_value.values
        return np.any(arr == filter_val)

    # 2. Handle standard lists, tuples, or numpy arrays stored in the cell
    elif isinstance(cell_value, (list, tuple, np.ndarray)):
        return np.any(np.array(cell_value) == filter_val)

    # 3. Handle standard scalar fallback
    try:
        return cell_value == filter_val
    except Exception:
        return False


def _restrict_by_nmice(df, filter_value):
    """Filter DataFrame by mouse numbers."""
    filter_values = _ensure_list(filter_value)
    mice_str = ", ".join(map(str, filter_values))
    print(f"Getting Mice {mice_str} from Dir")

    mouse_names = [f"Mouse{str(num).zfill(3)}" for num in filter_values]

    if "name" in df.columns:
        mask = df["name"].isin(mouse_names)
        filtered_df = df[mask]
        found_mice = filtered_df["name"].unique()
        for name in mouse_names:
            if name not in found_mice:
                print(f"No {name} in Dir")
        return filtered_df

    print("No 'name' column found")
    return pd.DataFrame()


def _restrict_by_session(df, filter_value):
    """Filter DataFrame by session name."""
    filter_values = _ensure_list(filter_value)
    session_str = " + ".join(filter_values)
    print(f"Getting Session {session_str} from Dir")

    if "Session" in df.columns:
        mask = pd.Series([False] * len(df))
        for session_name in filter_values:
            mask |= df["Session"].astype(str).str.contains(session_name)
        filtered_df = df[mask]
        if filtered_df.empty:
            for session_name in filter_values:
                print(f"Session {session_name} is empty")
        return filtered_df

    print("No 'Session' column found")
    return pd.DataFrame()


def _restrict_by_treatment(df, filter_value):
    """Filter DataFrame by treatment."""
    filter_values = _ensure_list(filter_value)
    treatment_str = " + ".join(filter_values)
    print(f"Getting Treatments {treatment_str} from Dir")

    if "Treatment" in df.columns:
        filtered_df = df[df["Treatment"].isin(filter_values)]
        found_treatments = filtered_df["Treatment"].unique()
        for missing in [t for t in filter_values if t not in found_treatments]:
            print(f"Treatment {missing} is empty")
        return filtered_df

    print("No 'Treatment' column found")
    return pd.DataFrame()


def restrict_path_for_experiment(
    Dir: Union[Dict[str, Any], pd.DataFrame],
    filter_type: str,
    filter_value: Union[str, List, int],
) -> pd.DataFrame:
    """
    Python equivalent of RestrictPathForExperiment MATLAB function.
    """
    # Convert input to DataFrame if it's a dictionary
    df = dict_to_dataframe(Dir) if isinstance(Dir, dict) else Dir.copy()

    # Handle 'all' cases
    if filter_type == "all" or filter_value == "all" or filter_value is None:
        return df

    # Dispatch to appropriate filter helper
    filter_map = {
        "Group": _restrict_by_group,
        "nMice": _restrict_by_nmice,
        "Session": _restrict_by_session,
        "Treatment": _restrict_by_treatment,
    }

    if filter_type not in filter_map:
        raise ValueError(
            f"filter_type must be one of {list(filter_map.keys())} or 'all'"
        )

    filtered_df = filter_map[filter_type](df, filter_value)

    return filtered_df.reset_index(drop=True)


def dict_to_dataframe(Dir: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert dictionary structure from path_for_experiments_erc to pandas DataFrame.

    Args:
        Dir: Dictionary containing experiment information

    Returns:
        pandas DataFrame with experiments as rows and attributes as columns
    """
    # Handle empty dictionary
    if not Dir or "path" not in Dir:
        return pd.DataFrame()

    # Get the number of experiments
    n_experiments = len(Dir["path"]) if Dir["path"] else 0

    if n_experiments == 0:
        return pd.DataFrame()

    # Initialize DataFrame dictionary
    df_dict = {}

    # Handle basic fields
    basic_fields = ["path", "name", "manipe"]
    for field in basic_fields:
        if field in Dir and Dir[field]:
            df_dict[field] = Dir[field][:n_experiments]
        else:
            df_dict[field] = [None] * n_experiments

    # Handle optional fields
    optional_fields = [
        "CorrecAmpli",
        "Session",
        "delay",
        "date",
        "Treatment",
        "expe_info",
        "results",
        "network_path",
    ]
    for field in optional_fields:
        if field in Dir and Dir[field]:
            # Ensure the field has the right length
            field_data = Dir[field]
            if len(field_data) >= n_experiments:
                df_dict[field] = field_data[:n_experiments]
            else:
                # Pad with None if shorter
                df_dict[field] = field_data + [None] * (n_experiments - len(field_data))
        else:
            df_dict[field] = [None] * n_experiments

    # Handle group field (can be dictionary or list)
    if "group" in Dir and Dir["group"]:
        if isinstance(Dir["group"], dict):
            # Group is a dictionary with keys like 'LFP', 'Neurons', etc.
            for group_key, group_values in Dir["group"].items():
                if (
                    isinstance(group_values, list)
                    and len(group_values) >= n_experiments
                ):
                    df_dict[f"group_{group_key}"] = group_values[:n_experiments]
                else:
                    df_dict[f"group_{group_key}"] = [None] * n_experiments
        else:
            # Group is a simple list
            if len(Dir["group"]) >= n_experiments:
                df_dict["group"] = Dir["group"][:n_experiments]
            else:
                df_dict["group"] = Dir["group"] + [None] * (
                    n_experiments - len(Dir["group"])
                )
    else:
        df_dict["group"] = [None] * n_experiments

    # Create DataFrame
    df = pd.DataFrame(df_dict)

    return df


def dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert pandas DataFrame back to dictionary structure for compatibility.

    Args:
        df: pandas DataFrame with experiment data

    Returns:
        Dictionary structure compatible with original MATLAB format
    """
    if df.empty:
        return {"path": [], "name": [], "manipe": []}

    result = {}

    # Handle group columns
    group_columns = [col for col in df.columns if col.startswith("group_")]
    if group_columns:
        result["group"] = {}
        for col in group_columns:
            group_key = col.replace("group_", "")
            result["group"][group_key] = df[col].tolist()
    elif "group" in df.columns:
        result["group"] = df["group"].tolist()

    # Handle other columns
    for col in df.columns:
        if not col.startswith("group_"):
            result[col] = df[col].tolist()

    return result


def merge_path_for_experiment(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple experiment DataFrames into one.

    Args:
        *dfs: Variable number of DataFrames to merge

    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()

    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on 'name' and 'path' if they exist
    if "name" in merged_df.columns and "path" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["name", "path"])
    elif "name" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["name"])

    return merged_df.reset_index(drop=True)


def intersect_path_for_experiment(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Find intersection of two experiment DataFrames based on mouse names.

    Args:
        df1: First DataFrame
        df2: Second DataFrame

    Returns:
        DataFrame containing only common experiments
    """
    if df1.empty or df2.empty or "name" not in df1.columns or "name" not in df2.columns:
        return pd.DataFrame()

    # Find common mouse names
    common_names = set(df1["name"]) & set(df2["name"])

    if not common_names:
        return pd.DataFrame()

    # Filter df1 to keep only common experiments
    result_df = df1[df1["name"].isin(common_names)].reset_index(drop=True)

    return result_df


# Updated path_for_experiments_erc to return DataFrame
def path_for_experiments_df(experiment_name: str, training_name: str) -> pd.DataFrame:
    """
    Modified version of path_for_experiments_erc that returns a DataFrame directly.

    Args:
        experiment_name: Name of the experiment type
        training_name: Name of the training session if it occurred

    Returns:
        pandas DataFrame containing experiment information
    """
    # This would use the original function and convert to DataFrame
    # For now, assuming the original function exists
    try:
        Dir = path_for_experiments(
            experiment_name=experiment_name, training_name=training_name
        )
        return dict_to_dataframe(Dir)
    except ImportError:
        print("Original path_for_experiments_erc function not available")
        return pd.DataFrame()


class Mouse_Results(Params, PaperFigures, SpatialConstraintsMixin):
    """
    Class to handle results for a specific mouse in an experiment.
    It will load the directory structure and parse all available windows.

    args:
    -------
        Dir: pd.DataFrame containing the directory structure of n_experiment
        mouse_name: str, name of the mouse (e.g., 'Mouse245')
        manipe: str, manipulation type (e.g., 'SubMFB', 'SubPAG')
        nameExp: str, name of the experiment (e.g., 'current', 'final_results', 'LossAndDirection...')
        full_path: str, full path to the experiment directory (optional, if not provided it will be found automatically)

    Returns:
        None

    This class is used to store and manage results related to a specific mouse.
    """

    # Bypass Params __new__ to avoid unwanted initialization
    def __new__(cls, *args, **kwargs):
        # Completely bypass Params.__new__
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        """
        Initialize the Mouse_Results class.
        """
        self._parse_init_args(args, kwargs)

        # find all window directories in the results path
        self.find_window_size(**kwargs)
        self.parameters: Dict[str, Params] = dict()
        self.projects: Dict[str, Project] = dict()

        for i, winMS in enumerate(self.windows):
            self._initialize_window(winMS, i, **kwargs)

        # Initialize PaperFigures and load trainers if requested
        if kwargs.get("load_trainers_at_init", True):
            self.load_trainers(**kwargs)

        add_training = kwargs.get("add_training", True)
        add_full_pre = kwargs.get("add_full_pre", False)
        PaperFigures.__init__(
            self,
            projectPath=self.Project,
            behaviorData=self.DataHelper.fullBehavior,
            bayes=self.bayes if hasattr(self, "bayes") else None,
            bayesMatrices=self.bayesMatrices
            if hasattr(self, "bayes_matrices")
            else None,
            l_function=self.l_function,
            timeWindows=self.windows_values,
            phase=self.phase,
            verbose=self.verbose,
            add_training=add_training,
            add_full_pre=add_full_pre,
            grid_size=self.Params.GaussianGridSize,
            maze_params=self.Linearizer.maze_params,
            sleep=kwargs.get("sleep", False),
        )

    def _parse_init_args(self, args, kwargs):
        """Extracts and validates core attributes from args and kwargs."""
        for key, value in kwargs.items():
            if hasattr(self, key) or callable(getattr(self, key, None)):
                continue
            setattr(self, key, value)

        args_list = list(args)
        Dir = args_list.pop(0) if args_list else kwargs.pop("Dir", None)
        mouse_name = args_list.pop(0) if args_list else kwargs.get("mouse_name", None)
        manipe = args_list.pop(0) if args_list else kwargs.get("manipe", None)

        exp_index = kwargs.get("exp_index", None)
        full_path = kwargs.get("full_path", "")
        phase = kwargs.get("phase", "pre")
        nameExp = kwargs.get("nameExp", "Network")
        target = kwargs.get("target", "pos")
        self.verbose = kwargs.get("verbose", True)

        if kwargs.get("deviceName") is not None:
            self.deviceName = kwargs["deviceName"]

        if any(v is None for v in [Dir, mouse_name, manipe, target, nameExp]):
            raise ValueError(
                "Dir, mouse_name, manipe, target, and nameExp are required"
            )

        self.Dir = Dir
        self.mouse_name = mouse_name
        self.manipe = manipe
        self.nameExp = nameExp
        self.target = target
        self.phase = phase
        self.which = kwargs.get("which", "all")
        self.exp_index = exp_index

        if full_path == "":
            self.find_path()
        else:
            self.path = full_path

        self.find_xml()
        self.folderResult = os.path.join(self.path, self.nameExp, "results")
        self.results = pd.DataFrame()

    def _initialize_window(self, winMS, i, **kwargs):
        """Loads or creates Project, Params and DataHelper for a given window."""
        self.projects[winMS] = Project(
            self.xml,
            windowSize=int(winMS) / 1000,
            **kwargs,
        )
        if i == 0:
            self.data_helper = DataHelperClass(
                self.xml,
                mode="compare",
                **kwargs,
            )
            self._setup_main_window(winMS, **kwargs)
        else:
            self.parameters[winMS] = self._load_params_fallback(winMS, **kwargs)

    def _load_params_fallback(self, winMS, **kwargs):
        """Fallback to load Params from json or create new one."""
        params_path = os.path.join(self.folderResult, winMS, "params.json")
        if os.path.exists(params_path):
            print(f"Loading saved params from {params_path}")
            with open(params_path, "r") as f:
                saved_params = json.load(f)
            # Update kwargs with saved params if not already set
            for k, v in saved_params.items():
                if k not in kwargs and k != "windowSize":
                    kwargs[k] = v

        return Params(
            helper=self.data_helper,
            windowSize=int(winMS) / 1000,
            save_json=True,
            **kwargs,
        )

    def _setup_main_window(self, winMS, **kwargs):
        """Initializes linearizer and sets main window references."""
        self.linearizer = UMazeLinearizer(
            self.projects[winMS].folder,
            data_helper=self.data_helper,
            **kwargs,
        )
        self.linearizer.verify_linearization(
            self.data_helper.positions[:, :2] / self.data_helper.maxPos(),
            self.projects[winMS].folder,
        )

        self.l_function = (
            self.linearizer.pykeops_linearization
            if kwargs.get("keops_linearization", False)
            else self.cpu_linearization
        )

        self.data_helper.get_true_target(
            windowSizeMS=int(winMS),
            l_function=self.l_function,
            in_place=True,
            show=kwargs.get("show", False),
        )

        if winMS not in self.parameters:
            self.parameters[winMS] = self._load_params_fallback(winMS, **kwargs)

        # Set main references to the first processed window
        self.DataHelper = self.data_helper
        self.Params = self.parameters[winMS]
        self.Project = self.projects[winMS]
        self.Linearizer = self.linearizer

        # Initialize base Params class
        Params.__init__(
            self,
            helper=self.DataHelper,
            windowSize=int(winMS) / 1000,
            **kwargs,
        )
        print(self)

    def cpu_linearization(self, x):
        self.windows[0]
        return self.linearizer.apply_linearization(x, keops=False)

    def __getstate__(self):
        """
        Custom getstate method to avoid pickling issues with certain attributes.
        This is necessary for compatibility with multiprocessing and other serialization methods.
        """
        state = self.__dict__.copy()
        # Remove attributes that cannot be pickled or are not needed for serialization (too big)
        state.pop("ann", None)
        state.pop("bayes", None)
        state.pop("bayes_matrices", None)
        return state

    def __setstate__(self, state):
        """
        Custom setstate method to restore the object state.
        This is necessary for compatibility with multiprocessing and other serialization methods.
        """
        self.__dict__.update(state)

    def to_pickle(cls, path: str):
        """
        Save Mouse_Results object to a pickle file.

        Args:
            obj: Mouse_Results object to save

        """

        with open(path, "wb") as f:
            pickle.dump(cls, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Mouse_Results object saved to {path}")

    @classmethod
    def from_pickle(cls, path: str, load_trainers: bool = True):
        """
        Load Mouse_Results object from a pickle file.

        Args:
            path: Path to the pickle file
            load_trainers: Whether to load trainers after loading the object

        Returns:
            Mouse_Results object
        """
        import dill as pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if load_trainers:
            cls._load_trainers_after_load(obj)

        print(f"Mouse_Results object loaded from {path}")
        return obj

    def _load_trainers_after_load(self):
        """
        Static method to load trainers after loading the Mouse_Results selfect.
        This is necessary because the trainers are not pickled.
        """
        state = self.__getstate__()
        # If the selfect has a load_trainers method, call it
        if hasattr(self, "load_trainers"):
            self.which = state.pop("which", "both")
            keys_to_pop = [
                "deviceName",
                "phase",
                "isTransformer",
                "linearizer",
                "behaviorData",
                "alpha",
                "transform_w_log",
                "denseweight",
                "projectPath",
            ]
            for key in keys_to_pop:
                # Remove keys that may not exist in the state
                state.pop(key, None)
            # Reinitialize attributes that were removed in getstate
            self.load_trainers(which=self.which, **state)

            print("Trainers loaded after pickle load.")

    def find_path(self):
        conditions = (
            self.Dir.name.str.lower().str.contains(self.mouse_name.lower())
        ) & (self.Dir.manipe.str.lower().str.contains(self.manipe.lower()))

        if not conditions.any():
            raise ValueError(
                f"No path found for mouse {self.mouse_name} with manipulation {self.manipe}."
            )

        if conditions.sum() > 1:
            if self.exp_index is None or self.exp_index == 0:
                manipe_in_path = self.Dir[conditions].path.str.contains(
                    self.manipe, case=False
                )
                if manipe_in_path.sum() == 1:
                    conditions = conditions & manipe_in_path
                else:
                    raise ValueError(
                        f"Multiple paths found for mouse {self.mouse_name} with manipulation {self.manipe}. Please specify exp_index to disambiguate and choose one of the following paths:\n{self.Dir[conditions][['path']].to_string()}"
                    )
            else:
                # add as a condition that os.path.basename of path contains exp_index
                suppl_conditions = self.Dir.path.str.contains(f"exp{self.exp_index}")
                conditions = conditions & suppl_conditions

                if conditions.sum() == 0:
                    raise ValueError(
                        f"No path found for mouse {self.mouse_name} with manipulation {self.manipe} and exp_index {self.exp_index}."
                    )
                elif conditions.sum() > 1:
                    raise ValueError(
                        f"Multiple paths found for mouse {self.mouse_name} with manipulation {self.manipe} and exp_index {self.exp_index}. Please check the exp_index value and choose one of the following paths:\n{self.Dir[conditions][['path']].to_string()}"
                    )

        if hasattr(self.Dir, "to_pandas"):
            Dir = self.Dir.to_pandas()
            conditions = conditions.to_pandas()
        else:
            Dir = self.Dir
        self.path = Dir[conditions].iloc[0].path
        self.network_path = Dir[conditions].iloc[0].network_path
        self.subDir = Dir[conditions]
        print(f"Path for {self.mouse_name} found: {self.path}")

    def find_xml(self):
        """
        Find the XML file for the mouse in the experiment directory.
        This is used to load the DataHelper object.
        """
        import fnmatch

        xml_file = None
        for pattern in [
            "*SpikeRef*.xml",
            f"*{os.path.basename(self.path)[:4]}*.xml",
            f"*{self.mouse_name}*.xml",
            "*amplifier*.xml",
            "*.xml",
        ]:
            xml_file = next(
                (
                    os.path.join(self.path, f)
                    for f in os.listdir(self.path)
                    if f.endswith(".xml")
                    and not (f.endswith("_fil.xml") or "filtered" in f)
                    and fnmatch.fnmatch(f, pattern)
                ),
                None,
            )
            if xml_file:
                self.xml = xml_file
                return xml_file

    def find_window_size(self, **kwargs):
        if not os.path.isdir(self.folderResult):
            raise FileNotFoundError(f"Results path {self.folderResult} does not exist.")
        windows = kwargs.get("windows", None)
        # convert to strings if windows is a list of integers
        if isinstance(windows, list):
            windows = [str(window) for window in windows]

        if windows is None:
            self.windows = [
                str(d)
                for d in os.listdir(self.folderResult)
                if os.path.isdir(os.path.join(self.folderResult, d))
            ]
            if not self.windows:
                raise ValueError(
                    f"No windows found in {self.folderResult} for {self.mouse_name}."
                )
        else:
            self.windows = windows
            if not isinstance(self.windows, list):
                self.windows = [str(self.windows)]

        # to be in dir you need to have a folder named + at least one csv file inside
        in_dir = [
            os.path.isdir(os.path.join(self.folderResult, d))
            and os.path.isfile(
                os.path.join(self.folderResult, d, "posIndex_training.csv")
            )
            for d in self.windows
        ]
        if not all(in_dir) and not kwargs.get("force_windows", False):
            warn(
                f"Some specified windows not found in {self.folderResult} for {self.mouse_name}:{[w for w, exists in zip(self.windows, in_dir) if not exists]}. Fixing..."
            )
            self.windows = [w for w, exists in zip(self.windows, in_dir) if exists]
        else:
            self.windows = [
                w for w in self.windows if w in os.listdir(self.folderResult)
            ]

        # order windows by their name (assuming they are named only with a number)
        self.windows.sort(key=lambda x: int(x))
        # convert windows str to int
        self.windows_values = [int(window) for window in self.windows]
        print(f"Windows found for {self.mouse_name}: {self.windows}")

    def __repr__(self):
        return f"Mouse_Results(mouse_name={self.mouse_name}, manipe={self.manipe}, name_exp={self.nameExp}, target={self.target}, phase={self.phase}, path={self.path}, windows={self.windows})"

    def __str__(self):
        return (
            f"{'M' + self.mouse_name:=^50}\n"
            f"Mouse_Results for {self.mouse_name} ({self.manipe})\n"
            f"Experiment: {self.nameExp}\n"
            f"Target: {self.target}\n"
            f"Phase: {self.phase}\n"
            f"Path: {self.path}\n"
            f"Windows: {', '.join(self.windows)}"
            f"\n{'=' * 50}"
        )

    def load_trainers(self, which="both", **kwargs) -> Dict[int, Any]:
        """
        Load trainers for each window size.

        Parameters:
            which (str): Type of trainer to load ('ann', 'bayes', or 'both').
            **kwargs: Additional keyword arguments for trainer initialization such as:
                deviceName (str): Device to use for training ('gpu' or 'cpu').
                debug (bool): Whether to run in debug mode.

                Regarding the bayes trainer and DecoderConfig kwargs:
                    bandwidth (int): Bandwidth for the bayes trainer.
                    kernel (str): Kernel type for the bayes trainer.
                    maskingFactor (float): Masking factor for the bayes trainer.


        """
        from neuroencoders.fullEncoder.an_network import (
            LSTMandSpikeNetwork as NNTrainer,
        )
        from neuroencoders.simpleBayes.decode_bayes import DecoderConfig
        from neuroencoders.simpleBayes.decode_bayes import Trainer as BayesTrainer

        if hasattr(self, "deviceName"):
            deviceName = kwargs.pop("deviceName", self.deviceName)
        else:
            deviceName = kwargs.pop("deviceName", "cpu")

        if deviceName.lower() == "gpu" or deviceName.lower() == "cpu":
            from neuroencoders.utils.management import manage_devices

            self.deviceName = manage_devices(
                deviceName.upper(),
                set_memory_growth=kwargs.get("set_memory_growth", True),
            )
        else:
            self.deviceName = deviceName

        phase = kwargs.pop("phase", self.phase)
        isTransformer = kwargs.pop("isTransformer", self.Params.isTransformer)
        transform_w_log = kwargs.pop("transform_w_log", self.Params.transform_w_log)
        denseweight = kwargs.pop("denseweight", self.Params.denseweight)

        for i, winMS in enumerate(self.windows):
            if i == 0 and which.lower() in ["ann", "both"]:
                if not hasattr(self, "ann") or kwargs.get("redo", False):
                    max_nb_spikes = kwargs.pop(
                        "max_nb_spikes", get_max_nb_spikes(winMS)
                    )
                    max_spikes_per_group = kwargs.pop("max_spikes_per_group", None)
                    self.ann = NNTrainer(
                        self.projects[winMS],
                        self.parameters[winMS],
                        deviceName=self.deviceName,
                        phase=phase,
                        isTransformer=isTransformer,
                        linearizer=self.linearizer,
                        behaviorData=self.data_helper.fullBehavior,
                        alpha=self.parameters[winMS].denseweightAlpha,
                        # we dont really care about the dynamic loss, but this way we load the training data in memory, with speedMask,
                        transform_w_log=transform_w_log,
                        denseweight=denseweight,
                        max_nb_spikes=max_nb_spikes,
                        max_spikes_per_group=max_spikes_per_group,
                        **kwargs,
                    )
            if i == 0 and which.lower() in ["bayes", "both"]:
                if (
                    not hasattr(self, "bayes")
                    or self.bayes is None
                    or kwargs.get("redo", False)
                ):
                    self.bayes_config = DecoderConfig(**kwargs)
                    if kwargs.get("bayes_project_path", None) is not None:
                        self.bayes_config.extra_kwargs["project_path"] = kwargs.get(
                            "bayes_project_path", None
                        )
                        print(
                            f"loading custom bayes project path from {self.bayes_config.extra_kwargs['project_path']}"
                        )
                        try:
                            project = Project.load(
                                os.path.join(
                                    self.bayes_config.extra_kwargs["project_path"],
                                    f"Project_{winMS}.pkl",
                                )
                            )
                        except (FileNotFoundError, AttributeError):
                            project = Project.load(
                                os.path.join(
                                    self.path,
                                    self.bayes_config.extra_kwargs["project_path"],
                                    f"Project_{winMS}.pkl",
                                )
                            )
                    else:
                        project = self.projects[winMS]
                    self.bayes = BayesTrainer(
                        project,
                        config=self.bayes_config,
                        phase=self.phase,
                        maze_params=self.data_helper.maze_coords,
                        **kwargs,
                    )
                    if kwargs.get("load_bayesMatrices", False):
                        try:
                            # allows to initialize bayes matrices if the pickle exists
                            self.bayesMatrices = self.bayes.train_order_by_pos(
                                self.data_helper.fullBehavior,
                                l_function=self.l_function,
                                **kwargs,
                            )
                        except (FileNotFoundError, AttributeError):
                            warn(
                                "You asked for bayes trainer, but no bayes matrices pickle was found."
                            )

    def load_results(
        self,
        winMS=None,
        redo=False,
        force=False,
        phase=None,
        which="both",
        show=False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load results for the specified window size.

        Args:
            winMS (int): Window size in milliseconds. If None, loads results for all windows.
            redo (bool): If True, forces reloading results even if they already exist.
            force (bool): If True, it will train the model if it wasnt trained before.
            which (str): Type of trainer to use ('ann', 'bayes', or 'both').
        kwargs: Additional keyword arguments for result loading.
            such as:
                show (bool): Whether to print results.
                lossSelection (str): Loss selection value
                euclidean (bool): Whether to use Euclidean distance.
                deviceName (str): Device to use for training ('gpu' or 'cpu').

        Returns:
            pd.DataFrame: append to the DataFrame containing the results.
        """

        if phase is None:
            phase = self.phase

        if which.lower() in ["bayes", "both"]:
            if not hasattr(self, "bayes_matrices"):
                try:
                    with open(
                        os.path.join(
                            self.bayes.folderResult,
                            "bayesMatrices.pkl",
                        ),
                        "rb",
                    ) as f:
                        self.bayesMatrices = pickle.load(f)
                except (FileNotFoundError, AttributeError):
                    if not force:
                        raise ValueError(
                            "Bayes matrices not found, please run the bayes trainer first or force the training with `force = True`."
                        )
                    else:
                        self.load_trainers(which="bayes", **kwargs)
                        self.retrain(which="bayes", **kwargs)
                        with open(
                            os.path.join(
                                self.bayes.folderResult,
                                "bayesMatrices.pkl",
                            ),
                            "rb",
                        ) as f:
                            self.bayesMatrices = pickle.load(f)

        windows, winValues = self._select_window(winMS)
        # Load results for all windows
        for win, win_value in zip(windows, winValues):
            if which.lower() in ["ann", "both"]:
                if not redo:
                    try:
                        suffix = f"_{phase}" if phase is not None else ""
                        pd.read_csv(
                            os.path.expanduser(
                                os.path.join(
                                    self.folderResult,
                                    win,
                                    f"featureTrue{suffix}.csv",
                                )
                            )
                        ).values[:, 1:]
                    except FileNotFoundError:
                        self.load_trainers(which="ann", **kwargs)
                        self.ann.test(
                            self.data_helper.fullBehavior,
                            windowSizeMS=win_value,
                            phase=phase,
                            l_function=self.l_function,
                            **kwargs,
                        )
                else:
                    print(f"Force loading ann results for window {win}.")
                    self.load_trainers(which="ann", **kwargs)
                    try:
                        self.ann.test(
                            self.data_helper.fullBehavior,
                            windowSizeMS=win_value,
                            phase=phase,
                            l_function=self.l_function,
                            **kwargs,
                        )
                    except Exception:
                        if not force:
                            raise ValueError(
                                f"Results for window {win} not found. Please run the ANN trainer first or force the training with `force = True`."
                            )
                        else:
                            print(
                                f"Results for window {win} not found, forcing training."
                            )
                            self.retrain(which="ann", window=win, phase=phase, **kwargs)

                (mean_ann, select_ann, mean_lin_ann, select_lin_ann) = (
                    print_results.print_results(
                        self.folderResult,
                        windowSizeMS=win_value,
                        target=self.target,
                        phase=phase,
                        typeDec="NN",
                        training_data=self.ann.training_data,
                        l_function=self.l_function,
                        show=show,
                        **kwargs,
                    )
                )

            if which.lower() in ["bayes", "both"]:
                outputs = None
                if not redo:
                    try:
                        suffix = f"_{phase}" if phase is not None else ""
                        with open(
                            os.path.expanduser(
                                os.path.join(
                                    self.bayes.folderResult,
                                    win,
                                    f"bayes_decoding_results{suffix}.pkl",
                                )
                            ),
                            "rb",
                        ) as f:
                            outputs = pickle.load(f)
                    except FileNotFoundError:
                        self.load_trainers(which="bayes", **kwargs)
                        epochMask = get_epochs_mask(
                            behaviorData=self.data_helper.fullBehavior,
                            useTrain=phase != self.phase,
                            useTest=phase != "training",
                        )
                        timeStepPred = self.data_helper.fullBehavior["positionTime"][
                            epochMask
                        ]
                        outputs = self.bayes.test_as_NN(
                            self.data_helper.fullBehavior,
                            self.bayesMatrices,
                            timeStepPred,
                            windowSizeMS=win_value,
                            l_function=self.l_function,
                            useTrain=phase != self.phase,
                            useTest=phase != "training",
                            **kwargs,
                        )
                else:
                    print(f"Force loading bayesian results for window {win}.")
                    self.load_trainers(which="bayes", **kwargs)
                    epochMask = get_epochs_mask(
                        behaviorData=self.data_helper.fullBehavior,
                        useTrain=phase != self.phase,
                        useTest=phase != "training",
                    )
                    timeStepPred = self.data_helper.fullBehavior["positionTime"][
                        epochMask
                    ]
                    outputs = self.bayes.test_as_NN(
                        self.data_helper.fullBehavior,
                        self.bayesMatrices,
                        timeStepPred,
                        windowSizeMS=win_value,
                        l_function=self.l_function,
                        useTrain=phase != self.phase,
                        useTest=phase != "training",
                        **kwargs,
                    )

                (
                    mean_eucl_bayes,
                    select_lin_bayes,
                    mean_lin_bayes,
                    select_lin_bayes,
                ) = print_results.print_results(
                    self.bayes.folderResult,
                    typeDec="bayes",
                    results=outputs,
                    windowSizeMS=win_value,
                    target=self.target,
                    phase=phase,
                    show=show,
                    **kwargs,
                )

            # append those results to the results DataFrame
            results_dict = {"phase": [phase], "windowSizeMS": [win_value]}
            if which.lower() in ["ann", "both"]:
                results_dict.update(
                    {
                        "mean_ann": [mean_ann],
                        "select_ann": [select_ann],
                        "mean_lin_ann": [mean_lin_ann],
                        "select_lin_ann": [select_lin_ann],
                    }
                )

            if which.lower() in ["bayes", "both"]:
                results_dict.update(
                    {
                        "mean_eucl_bayes": [mean_eucl_bayes],
                        "select_lin_bayes": [select_lin_bayes],
                        "mean_lin_bayes": [mean_lin_bayes],
                    }
                )
            if self.results.empty:
                self.results = pd.DataFrame(results_dict)

            else:
                self.results = pd.concat(
                    [
                        self.results,
                        pd.DataFrame(results_dict),
                    ],
                    ignore_index=True,
                )

        return self.results

    def show_results(self, winMS=None, phase=None, **kwargs):
        if winMS is None:
            self.windows[-1]
            winMS = self.windows_values[-1]
        else:
            idx = self.windows_values.index(winMS)
            self.windows[idx]

        if phase is None:
            phase = self.phase

        print_results.print_results(
            self.folderResult,
            windowSizeMS=winMS,
            target=kwargs.pop("target", self.target),
            phase=phase,
            training_data=self.ann.training_data,
            l_function=self.l_function,
            **kwargs,
        )

    def init_plotter(self, winMS=None, **kwargs):
        """
        Initialize the plotter for the specified window size.
        """
        which = kwargs.get("which", "ann")
        if winMS is None:
            self.windows[-1]
            winMS = self.windows_values[-1]

        idWindow = self.timeWindows.index(int(winMS))
        self.windows[idWindow]

        phase = kwargs.get("phase", self.phase)
        phase = (
            "_" + phase if phase is not None and not phase.startswith("_") else phase
        )

        from neuroencoders.importData.gui_elements import AnimatedPositionPlotter

        data_helper = kwargs.pop("data_helper", None)
        if data_helper is None:
            data_helper = self.data_helper

        positions_from_NN = kwargs.pop("positions_from_NN", None)
        if positions_from_NN is None:
            if which.lower() == "bayes":
                positions_from_NN = self.resultsBayes_phase[phase]["featureTrue"][
                    idWindow
                ]
            else:
                positions_from_NN = self.resultsNN_phase[phase]["featureTrue"][idWindow]
            if positions_from_NN is None:
                raise ValueError(
                    f"True positions not found in resultsNN_phase[{phase}]. Please run load_results first."
                )

        predicted = kwargs.pop("predicted", None)
        if predicted is None:
            if which.lower() == "bayes":
                predicted = self.resultsBayes_phase[phase]["featurePred"][idWindow]
            else:
                predicted = self.resultsNN_phase[phase]["featurePred"][idWindow]

        speedMaskArray = kwargs.pop("speedMaskArray", None)
        if speedMaskArray is None and kwargs.get("useSpeedMask", False):
            speedMaskArray = self.resultsNN_phase[phase]["speedMask"][idWindow]
        speedMaskArray_for_dim = self.resultsNN_phase[phase]["speedMask"][idWindow]

        prediction_time = kwargs.pop("prediction_time", None)
        if prediction_time is None:
            if which.lower() == "bayes":
                prediction_time = self.resultsBayes_phase[phase]["times"][idWindow]
            else:
                prediction_time = self.resultsNN_phase[phase]["times"][idWindow]

        posIndex = kwargs.pop("posIndex", None)
        if posIndex is None:
            posIndex = self.resultsNN_phase[phase]["posIndex"][idWindow]

        blit = kwargs.pop("blit", True)
        predicted_probs = None
        if kwargs.get("plot_heatmap", False):
            if which.lower() == "ann":
                self.load_trainers(which="ann", **kwargs)
                if (
                    getattr(self.ann.params, "GaussianHeatmap", False)
                    and kwargs.get("predicted_heatmap", None) is None
                    and kwargs.get("plot_heatmap", False)
                ):
                    try:
                        predicted_logits = self.resultsNN_phase_pkl[phase]["logits_hw"][
                            idWindow
                        ]
                    except (AttributeError, KeyError, TypeError):
                        if phase not in self.resultsNN_phase_pkl:
                            self.resultsNN_phase_pkl[phase] = {}
                        try:
                            with open(
                                os.path.join(
                                    self.projectPath.experimentPath,
                                    "results",
                                    str(winMS),
                                    f"decoding_results{phase}.pkl",
                                ),
                                "rb",
                            ) as f:
                                results = pickle.load(f)
                                for key in results.keys():
                                    if (
                                        not isinstance(
                                            self.resultsNN_phase_pkl[phase][key], list
                                        )
                                        or key not in self.resultsNN_phase_pkl[phase]
                                    ):
                                        self.resultsNN_phase_pkl[phase][key] = []
                                    if idWindow == len(
                                        self.resultsNN_phase_pkl[phase][key]
                                    ):
                                        self.resultsNN_phase_pkl[phase][key].append(
                                            results[key]
                                        )
                                    if (
                                        self.resultsNN_phase_pkl[phase][key][idWindow]
                                        is None
                                    ):
                                        self.resultsNN_phase_pkl[phase][key][
                                            idWindow
                                        ] = results[key]

                            predicted_logits = self.resultsNN_phase_pkl[phase][
                                "logits_hw"
                            ][idWindow]
                        except FileNotFoundError:
                            print(
                                f"No decoding_results{phase}.pkl found for window {winMS}."
                            )
                            self.ann.params.GaussianHeatmap = False
                            kwargs["predicted_heatmap"] = None
                            kwargs["plot_heatmap"] = False
                            predicted_probs = None
                    if predicted_probs is not None:
                        try:
                            predicted_probs = (
                                self.ann.GaussianHeatmap.decode_and_uncertainty(
                                    predicted_logits, return_probs=True
                                )[-1].numpy()
                            )
                        except Exception:
                            self.load_trainers(which="ann")
                            predicted_probs = (
                                self.ann.GaussianHeatmap.decode_and_uncertainty(
                                    predicted_logits, return_probs=True
                                )[-1].numpy()
                            )
            else:
                try:
                    predicted_map = self.resultsBayes_phase_pkl[phase]["probaMaps"][
                        idWindow
                    ]
                    predicted_heatmap = np.array(predicted_map)
                    kwargs["predicted_heatmap"] = predicted_heatmap
                except (AttributeError, KeyError, TypeError):
                    if phase not in self.resultsBayes_phase_pkl:
                        self.resultsBayes_phase_pkl[phase] = {}
                    try:
                        with open(
                            os.path.join(
                                self.projectPath.experimentPath,
                                "results",
                                str(winMS),
                                f"bayes_decoding_results{phase}.pkl",
                            ),
                            "rb",
                        ) as f:
                            results = pickle.load(f)

                            for key in results.keys():
                                if (
                                    not isinstance(
                                        self.resultsBayes_phase_pkl[phase][key], list
                                    )
                                    or key not in self.resultsBayes_phase_pkl[phase]
                                ):
                                    self.resultsBayes_phase_pkl[phase][key] = []
                                if idWindow == len(
                                    self.resultsBayes_phase_pkl[phase][key]
                                ):
                                    self.resultsBayes_phase_pkl[phase][key].append(
                                        results[key]
                                    )
                                if (
                                    self.resultsBayes_phase_pkl[phase][key][idWindow]
                                    is None
                                ):
                                    self.resultsBayes_phase_pkl[phase][key][
                                        idWindow
                                    ] = results[key]
                        predicted_map = self.resultsBayes_phase_pkl[phase]["probaMaps"][
                            idWindow
                        ]
                        predicted_heatmap = np.array(predicted_map)
                    except FileNotFoundError:
                        print(
                            f"No bayes_decoding_results{phase}.pkl found for window {winMS}."
                        )
                        kwargs["predicted_heatmap"] = None
                        kwargs["plot_heatmap"] = False
                        predicted_probs = None

        predicted_heatmap = kwargs.pop("predicted_heatmap", None)
        if not kwargs.get("plot_heatmap", False):
            predicted_heatmap = None

        if which.lower() == "bayes":
            data_helper.target = "pos"  # for now we did not try anything else

        plotter = AnimatedPositionPlotter(
            data_helper=data_helper,
            positions_from_NN=positions_from_NN,
            predicted=predicted,
            speedMaskArray=speedMaskArray,
            prediction_time=prediction_time,
            posIndex=posIndex,
            predicted_heatmap=predicted_heatmap,
            optional_predicted_dim=speedMaskArray_for_dim,
            blit=blit,
            l_function=kwargs.pop("l_function", self.l_function),
            **kwargs,
        )
        return plotter

    def show_movie(self, winMS=None, **kwargs):
        """
        Show the animated position plotter for the specified window size.
        Available kwargs are for figsaving, and FuncAnimation parameters such as:
            colormap: Colormap for direction coding (default: 'hsv')
            alpha_trail_line: Transparency for trail lines (default: 0.6)
            alpha_trail_points: Transparency for trail points (default: 0.95)
            alpha_delta_line: Transparency for delta line (default: 0.6)
            pair_points: Whether to pair predicted and true points (default: False)
            binary_colors: Use binary coloring (auto-detected if None)
            shock_color: Color for shock zone direction (1 values, default: 'hotpink')
            safe_color: Color for safe zone direction (0 values, default: 'cornflowerblue')
            hlines: List of y-values for horizontal lines (default: None)
            vlines: List of x-values for vertical lines (default: None)
            line_colors: Color(s) for reference lines (default: 'black')
            line_styles: Style(s) for reference lines (default: '--')
            line_widths: Width(s) for reference lines (default: 1.0)
            line_alpha: Transparency for reference lines (default: 0.7)
            custom_lines: List of line segments as [(x1,y1), (x2,y2), ...] or numpy array (default: None)
            custom_line_colors: Color(s) for custom lines (default: 'black')
            custom_line_styles: Style(s) for custom lines (default: '-')
            custom_line_widths: Width(s) for custom lines (default: 2.0)
            custom_line_alpha: Transparency for custom lines (default: 0.8)
            with_ref_bg: Whether to use a reference background image (default: True)
        """
        block = kwargs.pop("block", True)
        plotter = self.init_plotter(winMS, **kwargs)
        plotter.show(
            block=block,
            show=True,
            **kwargs,
        )

    def render_frame_static(self, frame: int, winMS=None, **kwargs):
        """
        Render a single frame for the animated position plotter.

        Args:
            frame_idx (int): Index of the frame to render.
            **kwargs: Additional keyword arguments for rendering.

        Returns:
            None
        """
        setup_plot = kwargs.pop("setup_plot", True)
        # as we never call the show method, we need to setup the plot here with the correct kwargs
        plotter = self.init_plotter(winMS, setup_plot=setup_plot, **kwargs)
        # we need to initialize one plotter per frame to avoid issues with joblib/multiprocessing in the future.
        plotter.animate_frame(frame=frame, **kwargs)

    @timing
    def save_video_frame_linearly(self, winMS=None, output_dir=None, **kwargs):
        """
        Save video frames for the specified window size using a simple loop.
        """

        from tqdm import tqdm

        if winMS is None:
            winMS = self.windows_values[-1]

        if output_dir is None:
            output_dir = os.path.join(self.folderResult, str(winMS), "video_frames")

        os.makedirs(output_dir, exist_ok=True)

        phase = kwargs.get("phase", self.phase)
        kwargs["output_dir"] = output_dir
        kwargs["setup_plot"] = (
            True  # Ensure setup_plot is True for worker initialization
        )
        kwargs["init_animation"] = True  # Ensure animation is initialized
        force = kwargs.get("force", False)

        init_plotter = self.init_plotter(winMS, **kwargs)
        total_frames = init_plotter.total_frames

        i = 5
        save_path = os.path.join(init_plotter.output_dir, f"frame_{i:09d}.png")
        if not os.path.exists(save_path):
            try:
                print("🚀 Using linear loop for rendering")

                for i in tqdm(
                    range(total_frames), desc="Rendering frames", unit="frame"
                ):
                    if i > 10 and kwargs.get("debug", False):
                        break  # DEBUG LIMIT TO 100 FRAMES
                    save_path = os.path.join(
                        init_plotter.output_dir, f"frame_{i:09d}.png"
                    )
                    init_plotter.animate_frame(i, **kwargs, save_path=save_path)
            except Exception as e:
                print("❌ Error during frame rendering:", e)
                if not force:
                    raise e
                else:
                    print("⚠️ Continuing despite the error due to force=True.")

        if kwargs.get("auto_encode", True):
            print("🎬 Encoding video with ffmpeg...")

            input_pattern = os.path.join(output_dir, "frame_%09d.png")
            video_name = kwargs.get(
                "video_name",
                f"mouse_{self.mouse_name}_win_{winMS}_phase_{phase}.mp4",
            )
            ffmpeg_path = kwargs.get("ffmpeg_path", "ffmpeg")  # Default to 'ffmpeg'
            output_video_path = (
                os.path.join(output_dir, video_name)
                if kwargs.get("video_path", None) is None
                else kwargs.get("video_path")
            )

            ffmpeg_cmd = f'{ffmpeg_path} -y -framerate 60 -i "{input_pattern}" -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p -g 40 -keyint_min 40 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" "{output_video_path}"'

            import subprocess

            try:
                subprocess.run(ffmpeg_cmd, shell=True, check=True)
                print(f"✅ Video saved to {output_video_path}")
            except subprocess.CalledProcessError as e:
                print("❌ ffmpeg encoding failed:", e)

            if kwargs.get("remove_frames", True):
                print("🗑️ Removing temporary frame files...")
                for i in range(total_frames):
                    frame_path = os.path.join(output_dir, f"frame_{i:09d}.png")
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                print("✅ Temporary frames removed.")

    @timing
    def save_video_frame_with_pool(self, winMS=None, output_dir=None, **kwargs):
        """
        Save video frames for the specified window size using multiprocessing.Pool for parallel processing.
        """

        from multiprocessing import Pool

        from tqdm import tqdm

        if winMS is None:
            winMS = self.windows_values[-1]

        if output_dir is None:
            output_dir = os.path.join(self.folderResult, winMS, "video_frames")

        os.makedirs(output_dir, exist_ok=True)

        if not kwargs.get("skip_frame_rendering", True):
            # We prepare a dummy to get frame count
            init_plotter = self.init_plotter(winMS, output_dir=output_dir, **kwargs)
            total_frames = init_plotter.total_frames

            kwargs["output_dir"] = output_dir
            kwargs["setup_plot"] = (
                True  # Ensure setup_plot is True for worker initialization
            )
            kwargs["init_animation"] = True  # Ensure animation is initialized

            print("🚀 Using multiprocessing.Pool for rendering")

            # with get_context("spawn").Pool(
            with Pool(
                initializer=_init_worker_plotter, initargs=(self, winMS, kwargs)
            ) as pool:
                list(
                    tqdm(
                        pool.imap(_render_frame_worker, range(total_frames)),
                        total=total_frames,
                        desc="Rendering frames",
                    )
                )

        if kwargs.get("auto_encode", False):
            print("🎬 Encoding video with ffmpeg...")

            input_pattern = os.path.join(output_dir, "frame_%04d.png")
            video_name = kwargs.get(
                "video_name",
                f"mouse_{self.mouse_name}_win_{winMS}_phase_{self.phase}.mp4",
            )
            ffmpeg_path = kwargs.get("ffmpeg_path", "ffmpeg")  # Default to 'ffmpeg'
            output_video_path = (
                os.path.join(output_dir, video_name)
                if kwargs.get("video_path", None) is None
                else kwargs.get("video_path")
            )

            ffmpeg_cmd = f'{ffmpeg_path} -y -framerate 20 -i "{input_pattern}" -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -g 40 -keyint_min 40 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" "{output_video_path}"'

            import subprocess

            try:
                subprocess.run(ffmpeg_cmd, shell=True, check=True)
                print(f"✅ Video saved to {output_video_path}")
            except subprocess.CalledProcessError as e:
                print("❌ ffmpeg encoding failed:", e)

            if kwargs.get("remove_frames", True):
                print("🗑️ Removing temporary frame files...")
                for i in range(total_frames):
                    frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                print("✅ Temporary frames removed.")

    def retrain(self, window=None, which="both", **kwargs):
        """
        Retrain the model for the specified window size.

        Args:
            window (int or str): Window size in milliseconds. If None, retrains for all windows.
            which (str): Type of trainer to retrain ('ann', 'bayes', or 'both').

        kwargs: Additional keyword arguments for training such as:
            isPredLoss : bool, whether to predict loss.
            earlyStopping : bool, whether to use early stopping.
            scheduler : str, decay or fixed.

        Returns:
            None
        """

        if which.lower() in ["bayes", "both"]:
            self.bayes.train_order_by_pos(
                self.DataHelper.fullBehavior,
                l_function=self.l_function,
                **kwargs,
            )

        windows, winValues = self._select_window(window)
        for win, win_val in zip(windows, winValues):
            if which.lower() in ["ann", "both"]:
                self.ann.train(
                    self.data_helper.fullBehavior,
                    windowSizeMS=win_val,
                    l_function=self.l_function,
                    **kwargs,
                )

    def _select_window(self, window):
        """
        Helper function to select the appropriate window size based on the input.

        Args:
            window (int, str, or None): Window size to select. If None, selects all available windows. WARNING: Input must be in MS.

        Returns:
            list: List of window sizes as strings.
            list: List of window sizes as integers if available.
        """
        if window is None:
            windows = self.windows
            windows_values = self.windows_values
        elif isinstance(window, int):
            if window not in self.windows_values:
                raise ValueError(
                    f"Window size {window} not found in available windows: {self.windows_values}"
                )
            windows = [str(window)]
            windows_values = [window]
        elif isinstance(window, str):
            if window not in self.windows:
                raise ValueError(
                    f"Window size {window} not found in available windows: {self.windows}"
                )
            windows = [window]
            windows_values = [int(window)]
        else:
            raise TypeError(f"window must be an int or str, got {type(window)}")
        return windows, windows_values

    def get_epoch_interval(self, phase):
        phase_name = _normalize_phase_name(phase)
        return_dict = {
            "training": (
                getattr(self, "training", None),
                getattr(self, "trainMask", None),
            ),
            "testing": (
                getattr(self, "testing", None),
                getattr(self, "testMask", None),
            ),
            # pre is both Habituation and PreTests merged together
            "pre": (getattr(self, "pre", None), getattr(self, "preMask", None)),
            "pre_test": (getattr(self, "pre", None), getattr(self, "preMask", None)),
            "hab": (getattr(self, "hab", None), getattr(self, "habMask", None)),
            "cond": (getattr(self, "cond", None), getattr(self, "condMask", None)),
            "post": (getattr(self, "post", None), getattr(self, "postMask", None)),
            "post_test": (getattr(self, "post", None), getattr(self, "postMask", None)),
            "sleep": (getattr(self, "sleep", None), getattr(self, "sleepMask", None)),
            "presleep": (
                getattr(self, "presleep", None),
                getattr(self, "presleepMask", None),
            ),
            "postsleep": (
                getattr(self, "postsleep", None),
                getattr(self, "postsleepMask", None),
            ),
            "pre_sleep": (
                getattr(self, "presleep", None),
                getattr(self, "presleepMask", None),
            ),
            "post_sleep": (
                getattr(self, "postsleep", None),
                getattr(self, "postsleepMask", None),
            ),
        }
        if hasattr(self, "extinct") and hasattr(self, "extinctMask"):
            return_dict["extinction"] = (self.extinct, self.extinctMask)

        if phase_name not in return_dict:
            raise ValueError(
                f"Phase '{phase}' not recognized. Available phases: {list(return_dict.keys())}"
            )

        return return_dict[phase_name]

    def run_spike_alignment(self, **kwargs):
        """
        Run spike alignment for the mouse results.
        This method will align spikes based on the linearized positions and save the results.

        Args:
            **kwargs: Additional keyword arguments for spike alignment such as:
                force (bool): Whether to force re-alignment.
                useTrain (bool): Whether to use training data for alignment.
                useTest (bool): Whether to use testing data for alignment.
                sleepName (List[str]): List of sleep names to consider for alignment.
                phase (str): phase to use to compute the tuning curves and spike alignment.
        """
        from neuroencoders.importData.compareSpikeFiltering import WaveFormComparator

        force = kwargs.get("force", False)
        useTrain = kwargs.pop("useTrain", False)
        useTest = kwargs.pop("useTest", not useTrain)
        useAll = kwargs.pop("useAll", useTrain and useTest)
        if useAll:
            useTrain = True
            useTest = True
        redo = kwargs.pop("redo", False)
        phase = kwargs.pop("phase", self.phase)
        if phase != self.phase:
            warn(
                "Phase specified in kwargs is different from the current phase. This may lead to unexpected results."
            )
        fullBehavior = self.get_fullBehavior_from_phase(phase)
        positions = self.DataHelper.get_true_target(
            windowSizeMS=self.windows_values[-1],
            l_function=self.l_function,
            in_place=False,
        )
        fullBehavior["Positions"] = positions

        if not hasattr(self, "waveform_comparators") or force:
            if not hasattr(self, "bayes") or self.bayes is None:
                self.load_trainers(which="both", **kwargs)

            self.waveform_comparators: Dict[str, WaveFormComparator] = dict()
            for win, winValue in zip(self.windows, self.windows_values):
                self.waveform_comparators[win] = WaveFormComparator(
                    self.projects[win],
                    self.parameters[win],
                    fullBehavior,
                    winValue,
                    phase=phase,
                    useTrain=useTrain,
                    useTest=useTest,
                    useAll=useTrain and useTest,
                    **kwargs,
                )
                self.waveform_comparators[win].save_alignment_tools(
                    self.bayes, self.l_function, winValue, redo=redo
                )

    def get_fullBehavior_from_phase(
        self,
        phase: Literal[
            "training",
            "all",
            "pre",
            "preNoHab",
            "hab",
            "cond",
            "post",
            "postNoExtinction",
            "extinction",
        ],
    ):
        """
        Starting from base fullBehavior, simply return a fullBehavior with adapted train/test Epochs.
        """
        if phase == self.phase or phase == "training":
            return self.data_helper.fullBehavior

        if "_" in phase:
            phase = phase.strip("_")

        fullbehav_phase = get_behavior(self.data_helper.folder, phase=phase)

        return fullbehav_phase

    def convert_to_df(self, redo=False, disable=False):
        if (
            hasattr(self, "results_df")
            and not redo
            and (
                isinstance(self.results_df, pd.DataFrame)
                or isinstance(self.results_df, pd.DataFrame)
            )
        ):
            print("Results DataFrame already exists. Use redo=True to recreate it.")
            return self.results_df

        # Pre-check to avoid repeated hasattr calls
        has_resultsNN = hasattr(self, "resultsNN_phase")
        has_resultsNN_obj = hasattr(self, "resultsNN")
        has_bayes = hasattr(self, "resultsBayes") and "featurePred" in self.resultsBayes

        if not has_resultsNN:
            raise ValueError("resultsNN_phase not found in results")

        data = []
        total_iterations = len(self.suffixes) * len(self.windows_values)

        with tqdm(
            total=total_iterations,
            desc=f"Converting {self.mouse_name}",
            disable=disable,
        ) as pbar:
            for suffix in self.suffixes:
                # Pre-strip suffix once
                phase_name = suffix.strip("_") if suffix else "all"

                for id, win in enumerate(self.windows_values):
                    data_helper_win = self.data_helper
                    resultsNN_suffix = self.resultsNN_phase[suffix]
                    if (
                        resultsNN_suffix is None
                        or resultsNN_suffix["posIndex"][id] is None
                    ):
                        print(
                            f"Results for mouse {self.mouse_name} and suffix '{suffix}' not found in resultsNN_phase. Skipping this suffix."
                        )
                        pbar.update(1)
                        continue

                    # Extract posIndex once
                    posIndex = resultsNN_suffix["posIndex"][id].flatten()

                    # Get frequently used behavior data
                    fullBehavior = data_helper_win.fullBehavior
                    full_truePos_from_behavior = fullBehavior["Positions"]
                    speed = fullBehavior["Speed"].flatten()

                    # Compute linearized positions once
                    full_trueLinPos_from_behavior = self.l_function(
                        full_truePos_from_behavior[:, :2]
                    )[1]

                    # Compute direction once
                    direction_from_behavior = data_helper_win._get_traveling_direction(
                        full_trueLinPos_from_behavior
                    )[posIndex]

                    linTruePos = resultsNN_suffix["linearTrue"][id].flatten()
                    direction_fromNN = data_helper_win._get_traveling_direction(
                        linTruePos
                    )
                    if posIndex.max() == len(speed):
                        posIndex = posIndex - 1

                    # Build row dictionary
                    row = {
                        "nameExp": self.nameExp,
                        "mouse": self.mouse_name,
                        "manipe": self.manipe,
                        "phase": phase_name,
                        "winMS": win,
                        "asymmetry_index": data_helper_win.get_training_imbalance()
                        * np.ones_like(linTruePos),
                        "alignedTruePos_fromBehavior": full_truePos_from_behavior[
                            posIndex
                        ],
                        "alignedTrueLinPos_from_behavior": full_trueLinPos_from_behavior[
                            posIndex
                        ],
                        "alignedTimeBehavior": fullBehavior["positionTime"][posIndex],
                        "timeNN": resultsNN_suffix["times"][id].flatten(),
                        "alignedSpeed": speed[posIndex],
                        "posIndex_NN": posIndex,
                        "speedMask": resultsNN_suffix["speedMask"][id].flatten(),
                        "linearPred": resultsNN_suffix["linearPred"][id].flatten(),
                        "featurePred": resultsNN_suffix["featurePred"][id],
                        "featureTrue": resultsNN_suffix["featureTrue"][id],
                        "linearTrue": linTruePos,
                        "predLoss": resultsNN_suffix["predLoss"][id].flatten(),
                        "resultsNN": self.resultsNN if has_resultsNN_obj else None,
                        "direction_fromBehavior": direction_from_behavior,
                        "direction_fromNN": direction_fromNN,
                    }

                    # Add Bayesian results if available
                    if has_bayes:
                        resultsBayes_suffix = self.resultsBayes_phase[suffix]
                        row["bayesPred"] = resultsBayes_suffix["featurePred"][id]
                        row["bayesLinPred"] = resultsBayes_suffix["linearPred"][
                            id
                        ].flatten()
                        row["bayesProba"] = resultsBayes_suffix["predLoss"][
                            id
                        ].flatten()

                    data.append(row)
                    pbar.update(1)

        self.results_df = pd.DataFrame(data)
        self.results_df.set_index(
            ["nameExp", "mouse", "manipe", "phase", "winMS"], inplace=True
        )
        return self.results_df

    def get_tuning_curves(
        self,
        suffix: Optional[str] = None,
        feature_name: str = "linearTrue",
        idWindow: int = 0,
        use_speed_filter: bool = True,
        count_thresh: Optional[int] = None,
        **kwargs,
    ):
        """
        Computes the tuning curves for all mice on one suffix.

        Parameters:
        - suffix: The suffix to use for accessing the results. If None, it will be determined as training.
        - feature_name: The name of the feature to compute tuning curves for (default is "linearTrue").
        - idWindow: The index of the window to use for accessing the feature and speed mask (default is 0).
        - use_speed_filter: Whether to apply a speed filter to the epochs used for computing tuning curves (default is True).
        - count_thresh: If provided, neurons with total counts below this threshold will be excluded from the tuning curves.
        - kwargs: Additional keyword arguments for plotting the tuning curves. If 'plot' is True (default), the tuning curves will be plotted. You can also provide 'sort_map' and 'list_neurons' for sorting the tuning curves.

        Returns:
        - final: A concatenated array of tuning curves for all mice.
        - sort_map: A mapping of neuron IDs to their sorted positions, if sorting was performed.
        """

        bin_size = kwargs.pop("bin_size", 0.036)
        mode = kwargs.pop("mode", "closest")

        final, id_neurons, spike_data, phase = _compute_tuning_curves_for_result(
            self,
            suffix=suffix,
            feature_name=feature_name,
            idWindow=idWindow,
            use_speed_filter=use_speed_filter,
            count_thresh=count_thresh,
            bin_size=bin_size,
            mode=mode,
            **kwargs,
        )
        self.spikeData = spike_data

        if kwargs.pop("plot", True):
            ordered, sort_map = self.compute_linear_tuning_curves_order(
                lin_place_fields=final.values,
                bin_edges=np.linspace(0, 1, final.values.shape[1] + 1),
                sort_map=kwargs.pop("sort_map", None),
                list_neurons=kwargs.pop("list_neurons", None),
            )
            title = kwargs.pop(
                "title",
                f"LT Curves on {feature_name} ({phase} - speed {use_speed_filter})",
            )
            kwargs["title"] = title
            self.plot_linear_tuning_curves(ordered, **kwargs)
            return final, sort_map, id_neurons

        return final, np.arange(final.shape[0]), id_neurons

    def plot_tuning_curves_in_order(self, d=2, n=5, **kwargs):
        if d == 1:
            return self.plot_linear_tuning_curves_in_order(n=n, **kwargs)
        elif d == 2:
            return self.plot_2d_tuning_curves_in_order(n=n, **kwargs)

    def plot_linear_tuning_curves_in_order(self, n=5, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure(figsize=(20, 8))
        else:
            fig = ax.figure

        final1d, id_neurons1d, _, _ = _compute_tuning_curves_for_result(self, **kwargs)
        bin_edges = np.linspace(0, 1, final1d.values.shape[1] + 1)
        ordered1d, sort_map = self.compute_linear_tuning_curves_order(
            lin_place_fields=final1d.values,
            bin_edges=bin_edges,
            sort_map=kwargs.get("sort_map", None),
            list_neurons=kwargs.get("list_neurons", None),
        )

        positions = Tsd(
            t=self.DataHelper.fullBehavior["positionTime"].flatten(),
            d=self.l_function(self.DataHelper.fullBehavior["Positions"][:, :2])[
                1
            ].flatten(),
        )

        time_epoch = self.get_epoch_interval(kwargs.get("suffix", "_training"))[0]
        not_nan_epoch = np.isnan(positions).threshold(0.5, "below").time_support
        ep = time_epoch.intersect(not_nan_epoch)

        if kwargs.get("use_speed_filter", True):
            speed_ep = self.DataHelper.get_mov_epochs()
            ep = ep.intersect(speed_ep)

        positions = positions.restrict(ep)
        linpos = np.linspace(0, 1, final1d.values.shape[1])

        for i in range(n):
            ax_top = fig.add_subplot(2, n, i + 1)
            tc_1d = ordered1d[i]
            tc_1d = (tc_1d - np.nanmin(tc_1d)) / (
                np.nanmax(tc_1d) - np.nanmin(tc_1d) + 1e-8
            )
            ax_top.plot(linpos, tc_1d)
            ax_top.set_xlabel("Linear Position")
            ax_top.set_ylabel("Firing Rate (normalized)")
            ax_top.set_title(f"Neu. {id_neurons1d[sort_map][i]} - Top {i + 1}")

            ax_bot = fig.add_subplot(2, n, n + i + 1)
            tc_1d = ordered1d[-(i + 1)]
            tc_1d = (tc_1d - np.nanmin(tc_1d)) / (
                np.nanmax(tc_1d) - np.nanmin(tc_1d) + 1e-8
            )
            ax_bot.plot(linpos, tc_1d)
            ax_bot.set_xlabel("Linear Position")
            ax_bot.set_ylabel("Firing Rate")
            ax_bot.set_title(
                f"Neu. {id_neurons1d[sort_map][-(i + 1)]} - Bottom {i + 1}"
            )

        plt.tight_layout()
        plt.show()

    def plot_2d_tuning_curves_in_order(self, n=5, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure(figsize=(20, 8))
        else:
            fig = ax.figure

        kwargs["feature_name"] = "linearTrue"
        title = kwargs.get("title", "2D Tuning Curves in Order")

        sigma = kwargs.pop("sigma", None)
        final1d, id_neurons1d, _, _ = _compute_tuning_curves_for_result(self, **kwargs)
        bin_edges = np.linspace(0, 1, final1d.values.shape[1] + 1)
        _, sort_map_1d = self.compute_linear_tuning_curves_order(
            lin_place_fields=final1d.values,
            bin_edges=bin_edges,
            sort_map=kwargs.get("sort_map", None),
            list_neurons=kwargs.get("list_neurons", None),
        )
        path = kwargs.get("path", None)

        kwargs["count_thresh"] = None
        kwargs["sigma"] = sigma
        kwargs["feature_name"] = "featureTrue"

        final2d, id_neurons2d, _, _ = _compute_tuning_curves_for_result(
            self, n_dims=2, **kwargs
        )
        bin_edges = np.linspace(0, 1, final2d.values.shape[1] + 1)

        ordered2d, _ = self.compute_linear_tuning_curves_order(
            lin_place_fields=final2d.values,
            bin_edges=bin_edges,
            sort_map=sort_map_1d,
            list_neurons=id_neurons1d,
        )

        ordered2d[
            :,
            ~self.get_allowed_mask_for_bin_size(
                ordered2d.shape[1], ordered2d.shape[2]
            ).T,
        ] = np.nan

        positions = TsdFrame(
            t=self.DataHelper.fullBehavior["positionTime"].flatten(),
            d=self.DataHelper.fullBehavior["Positions"][:, :2],
        )

        time_epoch = self.get_epoch_interval(kwargs.get("suffix", "_training"))[0]
        not_nan_epoch = np.isnan(positions).any(1).threshold(0.5, "below").time_support
        ep = time_epoch.intersect(not_nan_epoch)

        if kwargs.get("use_speed_filter", True):
            speed_ep = self.DataHelper.get_mov_epochs()
            ep = ep.intersect(speed_ep)

        positions = positions.restrict(ep)
        extent = (0, 1, 0, 1)

        self.spikeData = self.DataHelper.get_spike_data()

        for i in range(n):
            ax_top = fig.add_subplot(2, n, i + 1)
            tc_2d = ordered2d[i].T
            tc_2d = (tc_2d - np.nanmin(tc_2d)) / (
                np.nanmax(tc_2d) - np.nanmin(tc_2d) + 1e-8
            )
            ax_top.imshow(tc_2d, aspect="auto", origin="lower", extent=extent)
            spike_pos = (
                self.spikeData[id_neurons1d[sort_map_1d][i]]
                .restrict(ep)
                .value_from(positions)
            )
            ax_top.scatter(
                spike_pos[:, 0], spike_pos[:, 1], s=2, alpha=0.2, marker="+", c="red"
            )
            ax_top.set_xlabel("X")
            ax_top.set_ylabel("Y")
            ax_top.set_title(f"Neu. {id_neurons1d[sort_map_1d][i]} - Top {i + 1}")

            ax_bot = fig.add_subplot(2, n, n + i + 1)
            tc_2d = ordered2d[-(i + 1)].T
            tc_2d = (tc_2d - np.nanmin(tc_2d)) / (
                np.nanmax(tc_2d) - np.nanmin(tc_2d) + 1e-8
            )
            ax_bot.imshow(tc_2d, aspect="auto", origin="lower", extent=extent)

            spike_pos = (
                self.spikeData[id_neurons1d[sort_map_1d][-(i + 1)]]
                .restrict(ep)
                .value_from(positions)
            )
            ax_bot.scatter(
                spike_pos[:, 0], spike_pos[:, 1], s=2, alpha=0.2, marker="+", c="red"
            )
            ax_bot.set_xlabel("X")
            ax_bot.set_ylabel("Y")
            ax_bot.set_title(
                f"Neu. {id_neurons1d[sort_map_1d][-(i + 1)]} - Bottom {i + 1}"
            )

        if title is not None:
            plt.suptitle(title, fontsize=16)

        plt.tight_layout()

        if path is not None:
            plt.savefig(path)

        plt.show()

    def plot_respi_spectro_during_immobility(
        self, which: str = "freezing", path: Optional[str] = None, **kwargs
    ):
        try:
            respi = self.DataHelper.get_respi_data()
        except FileNotFoundError:
            respi = self.DataHelper.get_respi_data(self.network_path)

        max_respi_freq = kwargs.get("max_respi_freq", None)
        smooth_fact = kwargs.get("smooth_fact", 5)
        bin_size = kwargs.get("bin_size", 1)
        phase = kwargs.get("phase", "cond")
        interval_set = getattr(self, phase)

        if which == "freezing":
            interval = self.DataHelper.get_freeze_epochs()
        elif which == "ripples":
            interval = self.DataHelper.get_ripples_epochs()
        elif which == "stims":
            interval = self.DataHelper.get_stim_epochs()
        else:
            raise ValueError(
                "Invalid 'which' parameter. Choose from 'freezing', 'ripples', or 'stims'."
            )

        spectr, _ = self.DataHelper.compute_breathing_rate(
            spectro_tsd=respi, bin_size=bin_size, smooth_fact=smooth_fact
        )

        if max_respi_freq is not None:
            respi = respi.loc[[i for i in respi.columns if i <= max_respi_freq]]

        to_plot = respi.restrict(interval)

        to_plot_clean = to_plot.restrict(interval_set)

        results = self.resultsNN_phase[f"_{phase}"]

        pred_pos = results["linearPred"][0]
        true_pos = results["linearTrue"][0]
        times = results["times"][0]

        spectr = spectr.restrict(self.DataHelper.freeze_epochs).restrict(interval_set)
        pred_tsd = (
            Tsd(t=times, d=pred_pos)
            .restrict(interval_set)
            .restrict(self.DataHelper.freeze_epochs)
        )
        true_tsd = (
            Tsd(t=times, d=true_pos)
            .restrict(interval_set)
            .restrict(self.DataHelper.freeze_epochs)
        )
        stim_tsd = (
            self.DataHelper.get_stim_epochs()
            .intersect(interval_set)
            .intersect(self.DataHelper.freeze_epochs)
        )
        ripples_tsd = (
            self.DataHelper.get_ripples_epochs()
            .intersect(interval_set)
            .intersect(self.DataHelper.freeze_epochs)
        )

        # -------------------------------------------------------------------------
        # NEW: Create a continuous virtual time index to defeat non-linear spacing
        # -------------------------------------------------------------------------
        timestamps = to_plot_clean.as_units("s").index.values
        n_samples = len(timestamps)
        virtual_time = np.arange(n_samples)  # Simply 0, 1, 2, ..., N

        # Extent for imshow now uses indices instead of raw absolute time
        extent = [0, n_samples - 1, to_plot_clean.columns[0], to_plot_clean.columns[-1]]

        # Helper function to map non-linear timestamps to continuous virtual indices
        def time_to_index(t_array, reference_timestamps=timestamps):
            # Searches where the raw timestamps fit into our sliced/concatenated array
            return np.searchsorted(reference_timestamps, t_array)

        # 2. Map zones to their designated colors
        zone_color_dict = dict(zip(ZONELABELS, ZONE_COLORS))

        # 3. Create a 3-panel stacked layout
        fig, (ax_hypno, ax_matrix, ax_bias) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(14, 11),
            sharex=True,  # This still works flawlessly using indices!
            gridspec_kw={"height_ratios": [1, 3, 1.5]},
        )

        # -------------------------------------------------------------------------
        # PANEL 1: Hypnogram (Zone Epochs mapped to index space)
        # -------------------------------------------------------------------------
        zone_labels = self.DataHelper.ZoneLabels
        y_ticks = np.arange(len(zone_labels))

        zone_mapping = {
            "Safe": 1,
            "SafeCenter": 2,
            "Center": 3,
            "ShockCenter": 4,
            "Shock": 5,
        }
        zone_labels = sorted(zone_labels, key=lambda t: zone_mapping[t])

        for y_idx, zone_name in enumerate(zone_labels):
            zone_epochs = getattr(self.DataHelper, f"{zone_name}_EpochAligned")
            zone_epochs_clean = zone_epochs.intersect(
                to_plot_clean.time_support
            )  # Match the exact matrix timeframe

            for start, end in zip(zone_epochs_clean.start, zone_epochs_clean.end):
                # Convert absolute epochs boundaries into their corresponding indexed positions
                start_idx = time_to_index(start)
                end_idx = time_to_index(end)

                ax_hypno.hlines(
                    y=y_idx,
                    xmin=start_idx,
                    xmax=end_idx,
                    color=zone_color_dict[zone_name],
                    linewidth=6,
                )

        ax_hypno.set_yticks(y_ticks)
        ax_hypno.set_yticklabels(zone_labels)
        ax_hypno.set_ylim(-0.5, len(zone_labels) - 0.5)
        ax_hypno.invert_yaxis()
        ax_hypno.set_ylabel("Zones")
        ax_hypno.grid(axis="x", alpha=0.3, linestyle="--")
        ax_hypno.title.set_text(
            "Behavioral Zone, OB spectro and linear predictions (Concatenated Freeze Epochs)"
        )

        # -------------------------------------------------------------------------
        # PANEL 2: Main Continuous Matrix
        # -------------------------------------------------------------------------
        ax_matrix.imshow(
            to_plot_clean.values.T,
            aspect="auto",
            extent=extent,
            cmap="cmc.batlow",
            origin="lower",
        )
        ax_matrix.plot(
            virtual_time,
            to_plot_clean.value_from(spectr).values,
            color="grey",
            label="Detected breathing rate",
            linewidth=1.5,
        )
        ax_matrix.set_ylabel("Frequency (Hz)")

        # -------------------------------------------------------------------------
        # PANEL 3: Evolution of Shock Zone Bias (Decoded vs True)
        # -------------------------------------------------------------------------
        true_tsd = to_plot_clean.value_from(
            true_tsd.restrict(to_plot_clean.time_support)
        )
        pred_tsd = to_plot_clean.value_from(
            pred_tsd.restrict(to_plot_clean.time_support)
        )

        try:
            # Smooth in value space so non-linear gaps don't corrupt the smoothing logic
            true_vals = true_tsd.smooth(3).values
            pred_vals = pred_tsd.smooth(3).values
        except ValueError:
            true_vals = true_tsd.values
            pred_vals = pred_tsd.values

        # We plot directly against our linear virtual_time step spacing
        ax_bias.plot(
            virtual_time,
            true_vals,
            color="crimson",
            alpha=0.8,
            label="True Position",
            linewidth=1.5,
        )
        ax_bias.plot(
            virtual_time,
            pred_vals,
            color="navy",
            alpha=0.8,
            label="Predicted Position",
            linewidth=1.5,
        )

        ax_bias.set_ylabel("Shock Zone Bias\n(1=Safe, 0=Shock)")
        ax_bias.set_xlabel("Cumulative Sliced Time (Samples)")
        ax_bias.grid(True, alpha=0.3, linestyle="--")
        ax_bias.set_xlim(0, n_samples - 1)

        tick_locs = ax_bias.get_xticks()
        ax_bias.set_xticklabels([f"{int(loc)}" for loc in tick_locs])

        for end_time in stim_tsd.intersect(to_plot_clean.time_support).end:
            stim_idx = time_to_index(end_time)
            ax_matrix.axvline(stim_idx, linestyle="--", c=ALL_STIMS_COLOR, label="Stim")
            ax_bias.plot(stim_idx, 1.1, "*", c=ALL_STIMS_COLOR, label="Stim")

        for end_time in ripples_tsd.intersect(to_plot_clean.time_support).end:
            ripples_idx = time_to_index(end_time)
            ax_matrix.plot(
                ripples_idx,
                max(to_plot_clean.columns.values) - 1,
                "*",
                c=RIPPLES_COLOR,
                zorder=4,
                label="Ripple",
            )
            ax_bias.plot(ripples_idx, 1.1, "*", c=RIPPLES_COLOR, label="Ripple")

        handles, labels = ax_bias.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        handles_mat, labels_mat = ax_matrix.get_legend_handles_labels()
        by_label.update(dict(zip(labels_mat, handles_mat)))
        plt.legend(by_label.values(), by_label.keys(), loc="best")
        plt.tight_layout()

        if path is not None:
            import pathlib

            if not pathlib.Path(path).suffix:
                plt.savefig(
                    os.path.join(path, f"OB_spectro_during_{which}.png"), dpi=300
                )
                plt.savefig(os.path.join(path, f"OB_spectro_during_{which}.svg"))
            else:
                suffix = pathlib.Path(path).suffix
                # If the provided path has an extension, save directly to that path + svg
                plt.savefig(path, dpi=300)
                if suffix != ".svg":
                    plt.savefig(path.replace(suffix, ".svg"))

        plt.show()

    def compute_freeze_onoff_counts(
        self, wi=2.0, bin_size=0.05, smooth_sigma=0.05, count_thresh=200
    ) -> Optional[Dict[str, Any]]:
        """
        Identifies ON, OFF, Uninfluenced, and NAN (below threshold) freezing neurons
        using raw count/firing rates.
        """
        print("Computing freeze ON/OFF counts...")
        try:
            spikes = self.DataHelper.get_spike_data()
        except FileNotFoundError:
            if os.path.isfile(os.path.join(self.network_path, "SpikeData.mat")):
                import shutil

                shutil.copyfile(
                    os.path.join(self.network_path, "SpikeData.mat"),
                    os.path.join(self.DataHelper.folder, "SpikeData.mat"),
                )
                spikes = self.DataHelper.get_spike_data()
            else:
                raise

        cond_epoch = IntervalSet(self.cond)
        freeze_epochs = self.DataHelper.get_freeze_epochs().intersect(cond_epoch)
        if len(freeze_epochs) == 0:
            return None

        # Get raw spike counts per bin restricted to the target condition epoch
        spikes_cond = spikes.restrict(cond_epoch)
        counts = spikes_cond.count(bin_size)

        # Calculate the total integrated spike count for each neuron across the entire epoch
        total_counts_per_neuron = np.array(counts.restrict(freeze_epochs).sum(axis=0))

        # 1. IDENTIFY NAN NEURONS (Below count threshold)
        valid_mask = total_counts_per_neuron >= count_thresh
        nan_neurons_mask = ~valid_mask
        n_neurons_raw = len(valid_mask)

        print(
            f"{valid_mask.sum()} neurons passed the {count_thresh} thresh out of {n_neurons_raw}"
        )

        if valid_mask.sum() == 0:
            return None

        # Convert counts to firing rates (Hz) and smooth for robust PETH calculations
        fr = counts / bin_size
        smoothed_fr = fr.smooth(smooth_sigma)

        onsets = Ts(freeze_epochs.start)
        offsets = Ts(freeze_epochs.end)

        peth_on = compute_perievent(
            smoothed_fr, onsets, window=(-wi, wi), epochs=cond_epoch
        )
        peth_off = compute_perievent(
            smoothed_fr, offsets, window=(-wi, wi), epochs=cond_epoch
        )

        # Extract trial-averaged timecourses
        mean_on = np.nanmean(peth_on.values, axis=1).astype(float)
        mean_off = np.nanmean(peth_off.values, axis=1).astype(float)

        mean_on[np.isinf(mean_on)] = np.nan
        mean_off[np.isinf(mean_off)] = np.nan

        t_on = peth_on.times()
        t_off = peth_off.times()

        # Define baseline vs active freezing windows
        baseline_mask = t_on <= -0.5
        response_mask_on = (t_on >= 0.0) & (t_on <= wi)
        response_mask_off = t_off <= 0
        response_mask = np.concatenate([response_mask_on, response_mask_off])
        mean_full = np.vstack([mean_on, mean_off])

        # Compute raw average firing rates within windows
        base_line_avg = np.nanmean(mean_on[baseline_mask, :], axis=0)
        response_avg = np.nanmean(mean_full[response_mask, :], axis=0)

        # Absolute difference in firing rate (Hz)
        modulation = response_avg - base_line_avg

        # 2. CLASSIFY ON & OFF NEURONS (Must be valid)
        on_neurons_mask = (modulation > 0.5) & valid_mask
        off_neurons_mask = (modulation < -1) & valid_mask

        # 3. IDENTIFY UNINFLUENCED NEURONS
        # (Valid, but modulation doesn't cross either threshold boundary)
        uninfluenced_neurons_mask = (
            valid_mask & (~on_neurons_mask) & (~off_neurons_mask)
        )

        # 4. EXTRACT INTEGER EXPERIMENTAL IDs/INDICES
        all_indices = np.arange(n_neurons_raw)
        on_ids = all_indices[on_neurons_mask]
        off_ids = all_indices[off_neurons_mask]
        uninfluenced_ids = all_indices[uninfluenced_neurons_mask]
        nan_ids = all_indices[nan_neurons_mask]

        # Sort map layout calculation based on mean activation trajectory
        sort_idx = np.argsort(np.nanmean(mean_on, axis=0))

        return {
            "peth_on": peth_on,
            "peth_off": peth_off,
            "mean_on_raw": mean_on,
            "mean_off_raw": mean_off,
            "sort_idx": sort_idx,
            "n_neurons_raw": n_neurons_raw,
            "valid_mask": valid_mask,
            # Boolean logical masks (same length as the original spike object)
            "on_neurons_mask": on_neurons_mask,
            "off_neurons_mask": off_neurons_mask,
            "uninfluenced_neurons_mask": uninfluenced_neurons_mask,
            "nan_neurons_mask": nan_neurons_mask,
            # Integer indices of specific subpopulations
            "on_ids": on_ids,
            "off_ids": off_ids,
            "uninfluenced_ids": uninfluenced_ids,
            "nan_ids": nan_ids,
        }

    def compute_event_onoff_counts(
        self,
        wi=2.0,
        bin_size=0.05,
        smooth_sigma=0.05,
        count_thresh=200,
        around="ripples",
        focus_on=0.5,
    ) -> Optional[Dict[str, Any]]:
        """
        Identifies ON, OFF, Neutral, and NAN modulated neurons around single-point events
        (ripples/stims). If around="stims", extracts a 5th 'Delayed ON' population.
        """
        try:
            spikes = self.DataHelper.get_spike_data()
        except FileNotFoundError:
            if os.path.isfile(os.path.join(self.network_path, "SpikeData.mat")):
                import shutil

                shutil.copyfile(
                    os.path.join(self.network_path, "SpikeData.mat"),
                    os.path.join(self.DataHelper.folder, "SpikeData.mat"),
                )
                spikes = self.DataHelper.get_spike_data()
            else:
                raise

        cond_epoch = IntervalSet(self.cond)

        # --- SINGLE POINT EVENT EXTRACTION ---
        if around == "ripples":
            event_ts = Ts(self.DataHelper.get_ripples_epochs().start).restrict(
                cond_epoch
            )
        elif around == "stims":
            event_ts = Ts(self.DataHelper.get_stim_epochs().start).restrict(cond_epoch)
        else:
            raise ValueError(
                f"Undefined value {around}: choose between either 'ripples' or 'stims'"
            )

        if len(event_ts) == 0:
            return None

        # Calculate raw spike counts per bin over the condition epoch
        spikes_cond = spikes.restrict(cond_epoch)
        counts = spikes_cond.count(bin_size)

        # 1. IDENTIFY NAN NEURONS (Below count threshold)
        total_counts_per_neuron = np.array(counts.sum(axis=0))
        valid_mask = total_counts_per_neuron >= count_thresh
        nan_neurons_mask = ~valid_mask
        n_neurons_raw = len(valid_mask)

        print(
            f"{valid_mask.sum()} neurons passed the {count_thresh} thresh out of {n_neurons_raw}"
        )

        if valid_mask.sum() == 0:
            return None

        # Convert counts to firing rates (Hz) and apply smoothing
        fr = counts / bin_size
        smoothed_fr = fr.smooth(smooth_sigma)

        # --- SINGLE PETH COMPUTATION ---
        peth_event = compute_perievent(
            smoothed_fr, event_ts, window=(-wi, wi), epochs=cond_epoch
        )

        # Extract trial-averaged timecourse along axis=1 -> Shape: [time_bins, neurons]
        mean_event = np.nanmean(peth_event.values, axis=1).astype(float)
        mean_event[np.isinf(mean_event)] = np.nan

        t_event = peth_event.times()

        # Define baseline window
        baseline_mask = t_event <= -0.1
        base_line_avg = np.nanmean(mean_event[baseline_mask, :], axis=0)

        # Initialize empty placeholder masks for conditional delayed logic
        delayed_on_neurons_mask = np.zeros(n_neurons_raw, dtype=bool)

        # =========================================================
        # TARGET POPULATION CLASSIFICATION
        # =========================================================
        if around == "stims":
            # Window A: Immediate response window (0 to 250ms)
            early_mask = (t_event >= 0.0) & (t_event <= 0.25)
            early_avg = np.nanmean(mean_event[early_mask, :], axis=0)
            early_mod = early_avg - base_line_avg

            # Window B: Delayed response window (250ms to focus_on)
            # Safe boundary catch: if focus_on is <= 250ms, fallback to look up to wi (2.0s)
            late_end = focus_on if focus_on > 0.25 else wi
            late_mask = (t_event > 0.25) & (t_event <= late_end)
            late_avg = np.nanmean(mean_event[late_mask, :], axis=0)
            late_mod = late_avg - base_line_avg

            # Define immediate early responses
            on_neurons_mask = (early_mod > 1.0) & valid_mask
            off_neurons_mask = (early_mod < -1.0) & valid_mask

            # Define delayed on: neutral early, but active late
            was_early_neutral = (early_mod <= 1.0) & (early_mod >= -1.0)
            delayed_on_neurons_mask = was_early_neutral & (late_mod > 1.0) & valid_mask

        else:
            # Standard Ripples Logic: Single unified focus_on window
            response_mask = (t_event >= 0.0) & (t_event <= focus_on)
            response_avg = np.nanmean(mean_event[response_mask, :], axis=0)
            modulation = response_avg - base_line_avg

            on_neurons_mask = (modulation > 1.0) & valid_mask
            off_neurons_mask = (modulation < -1.0) & valid_mask

        # Identify Uninfluenced/Neutral Neurons (Valid, but missed all active criteria)
        uninfluenced_neurons_mask = (
            valid_mask
            & (~on_neurons_mask)
            & (~off_neurons_mask)
            & (~delayed_on_neurons_mask)
        )

        # --- EXTRACT ID INTEGER LISTS ---
        all_indices = np.arange(n_neurons_raw)
        on_ids = all_indices[on_neurons_mask]
        off_ids = all_indices[off_neurons_mask]
        delayed_on_ids = all_indices[delayed_on_neurons_mask]
        uninfluenced_ids = all_indices[uninfluenced_neurons_mask]
        nan_ids = all_indices[nan_neurons_mask]

        # Sort map layout calculation based on mean activation trajectory
        sort_idx = np.argsort(np.nanmean(mean_event, axis=0))

        return {
            "peth_event": peth_event,
            "mean_event_raw": mean_event,
            "sort_idx": sort_idx,
            "n_neurons_raw": n_neurons_raw,
            "valid_mask": valid_mask,
            # Boolean logical masks (Full length of neuron arrays)
            "on_neurons_mask": on_neurons_mask,
            "off_neurons_mask": off_neurons_mask,
            "delayed_on_neurons_mask": delayed_on_neurons_mask,
            "uninfluenced_neurons_mask": uninfluenced_neurons_mask,
            "nan_neurons_mask": nan_neurons_mask,
            # Integer experimental identity IDs
            "on_ids": on_ids,
            "off_ids": off_ids,
            "delayed_on_ids": delayed_on_ids,
            "uninfluenced_ids": uninfluenced_ids,
            "nan_ids": nan_ids,
        }

    def add_sleep_scoring(self, force: bool = False) -> Dict[str, Any]:
        if (
            hasattr(self.DataHelper, "sleep_scoring")
            and isinstance(self.DataHelper.sleep_scoring, dict)
            and not force
        ):
            return self.DataHelper.sleep_scoring

        from neuroencoders.utils.wrappers import loadSleepScoring

        try:
            if (
                not os.path.exists(
                    os.path.join(self.DataHelper.folder, "nnSleepScoring.mat")
                )
                and not os.path.exists(
                    os.path.join(self.DataHelper.folder, "SleepScoring_OBGamma.mat")
                )
                and not os.path.exists(
                    os.path.join(self.DataHelper.folder, "SleepScoring_Accelero.mat")
                )
            ):
                raise FileNotFoundError(
                    "Sleep scoring files not found in DataHelper folder. Attempting to load from network path."
                )
            self.DataHelper.sleep_scoring = loadSleepScoring(self.DataHelper.folder)
        except FileNotFoundError:
            self.DataHelper.sleep_scoring = loadSleepScoring(self.network_path)

        return self.DataHelper.sleep_scoring


class Results_Loader(TuningCurvesPlotter):
    """
    Class to load results from several Mouse_Results object.
    Will create a dict and a pandas DataFrame with the results.
    """

    @classmethod
    def from_pickle(cls, path: str) -> "Results_Loader":
        """
        Load Results_Loader object from a pickle file.

        Args:
            path: Path to the pickle file
        Returns:
            Results_Loader object
        """
        import dill as pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"Results_Loader object loaded from {path}")
        return obj

    def __init__(
        self,
        dir: pd.DataFrame,
        mice_nb: Optional[List[str]] = None,
        mice_manipes: Optional[List[str]] = None,
        timeWindows: Optional[List[int]] = None,
        phases=None,
        exp_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize Results_Loader with a DataFrame containing mouse results paths.

        Args:
            dir (pd.DataFrame): PathForExperiments DataFrame with columns for folder Results, mouse names, manipes, network paths, etc.
            mice_nb (List[str]): List of mouse numbers to filter results.
            mice_manipes (List[str]): List of manipes to filter results.
            timeWindows (List[int]): List of time windows in milliseconds to filter results. If None, uses all available windows.
            phase (str or List[str]): Phase of the experiment to filter results. If None, uses 'all' as default.

        keyword Args for Mouse_Results and ANN init:
            dict (dict): Dictionary to store results, default is empty.
            df (pd.DataFrame): DataFrame to store results, default is empty.
            If both of these are provided, the dict will be used to initialize the Mouse_Results objects.
            target (str): Target for the results, default is 'pos'. This can be 'pos', 'LinAndDirection', or any other target you want to analyze.
            load_trainers_at_init (bool): Whether to load trainers at initialization. Default is True.
            which (str): Type of trainer to load ('ann', 'bayes', or 'both'). Default is 'both'.
            deviceName (str): Device to use for training ('gpu' or 'cpu'). Default is 'gpu'.
            nEpochs (int): Number of epochs to consider for the ANN.
            isTransformer (bool): Whether to use a transformer model for the ANN. Default is False.
            batch_size (int): Batch size for training the ANN. Default is 64.
            transform_w_log (bool): Whether to apply a logarithmic transformation to the ann loss. Default is False.


        """
        super().__init__()
        self.all_spikes = None
        if mice_nb is None:
            mice_nb = dir.name.str.extract(r"(\d+)").astype(int)
        else:
            mice_nb = [int(m) for m in mice_nb]
        if mice_manipes is None:
            mice_manipes = dir.manipe.str.extract(r"(\w+)").astype(str)
        else:
            mice_manipes = [str(m) for m in mice_manipes]
        if timeWindows is None:
            warn("No timeWindows provided, using all windowSizeMS available in Dir.")
            self.timeWindows = "all"
        else:
            self.timeWindows = timeWindows
        if phases is None:
            warn("No phase provided, using 'all' as default.")
            self.phases = None
        else:
            self.phases = phases

        if exp_indices is None:
            exp_indices = np.zeros(len(dir), dtype=bool).tolist()
        if not isinstance(self.phases, List):
            self.phases = [self.phases]
        if not isinstance(self.timeWindows, List):
            if isinstance(self.timeWindows, int):
                self.timeWindows = [self.timeWindows]
            elif self.timeWindows == "all":
                self.timeWindows = ["all"]
            else:
                raise TypeError(
                    f"timeWindows must be a list of integers or an integer, got {type(self.timeWindows)}"
                )
        self.suffixes = [f"_{p}" for p in self.phases] if self.phases else [""]

        self.Dir = dir
        self.mice_nb = mice_nb
        self.mice_manipes = mice_manipes
        self.mice_names = [
            f"M{nb}{manipe}" for nb, manipe in zip(mice_nb, mice_manipes)
        ]
        self.exp_indices = exp_indices
        if kwargs.get("dict", None) is None:
            self.results_dict = {}
        else:
            self.results_dict = kwargs["dict"]
        if kwargs.get("df", None) is None:
            self.results_df = pd.DataFrame()
        else:
            self.results_df = kwargs["df"]

        if kwargs.get("nameExp", None) is not None:
            self.nameExp = kwargs["nameExp"]

        isTransformer = kwargs.pop("isTransformer", None)
        transform_w_log = kwargs.pop("transform_w_log", None)
        denseweight = kwargs.pop("denseweight", True)
        found_training = False

        if kwargs.get("dict", None) is None:
            for mouse_nb, manipe, mouse_full_name, exp_index in zip(
                self.mice_nb, self.mice_manipes, self.mice_names, self.exp_indices
            ):
                mouse_nb = str(mouse_nb)
                if exp_index is not None and exp_index != 0:
                    exp_index = int(exp_index)
                    mouse_full_name = f"{mouse_full_name}_exp{exp_index}"
                conditions = (
                    self.Dir.name.str.lower().str.contains(mouse_nb.lower())
                ) & (self.Dir.manipe.str.lower().str.contains(manipe.lower()))
                print(
                    f"Loading results for mouse {mouse_full_name} with conditions: {conditions.sum()} matching entries."
                )
                if not conditions.any():
                    raise ValueError(
                        f"Mouse {mouse_nb} with manipe {manipe} not found in the directory."
                    )
                window_tmp = []
                if conditions.sum() > 1:
                    if exp_index is None or exp_index == 0:
                        manipe_in_path = self.Dir[conditions].path.str.contains(
                            manipe, case=False
                        )
                        if manipe_in_path.sum() == 1:
                            conditions = conditions & manipe_in_path
                        else:
                            raise ValueError(
                                f"Multiple entries found for mouse {mouse_nb} with manipe {manipe}. Please provide exp_index to disambiguate."
                            )
                    else:
                        suppl_conditions = self.Dir.path.str.contains(f"exp{exp_index}")
                        conditions = conditions & suppl_conditions
                        if not conditions.any():
                            raise ValueError(
                                f"Mouse {mouse_nb} with manipe {manipe} and exp_index {exp_index} not found in the directory."
                            )
                        elif conditions.sum() > 1:
                            raise ValueError(
                                f"Multiple entries found for mouse {mouse_nb} with manipe {manipe} and exp_index {exp_index}. Please check the directory."
                            )

                path = self.Dir[conditions].iloc[0].path
                nameExp = os.path.basename(
                    self.Dir[
                        (self.Dir.name.str.lower().str.contains(mouse_nb.lower()))
                        & (self.Dir.manipe.str.lower().str.contains(manipe.lower()))
                    ]
                    .iloc[0]
                    .results
                )
                if nameExp not in self.results_dict:
                    self.results_dict[nameExp] = {}
                if not hasattr(self, "nameExp"):
                    self.nameExp = [nameExp]
                if nameExp not in self.nameExp:
                    self.nameExp.append(nameExp)
                if mouse_full_name not in self.results_dict[nameExp]:
                    self.results_dict[nameExp][mouse_full_name] = {}

                folderResult = os.path.join(path, nameExp, "results")
                if not os.path.exists(folderResult):
                    print(
                        f"Folder {folderResult} does not exist. Skipping mouse {mouse_nb} with manipulation {manipe}."
                    )
                    continue
                if self.timeWindows == "all":
                    windowSizeMS = [
                        int(d) for d in os.listdir(folderResult) if d.isdigit()
                    ]
                else:
                    windowSizeMS = self.timeWindows

                for win in windowSizeMS:
                    if os.path.exists(
                        os.path.join(
                            folderResult,
                            str(win),
                            f"errorFig_2d_NN{self.suffixes[0]}_pos.png",
                        )
                    ):
                        window_tmp.append(win)

                if window_tmp != windowSizeMS:
                    warn(
                        f"Warning: Not all windows found for mouse {mouse_nb} with manipulation {manipe}. Found: {window_tmp}, expected: {windowSizeMS}"
                    )

                for suffix, phase in zip(self.suffixes, self.phases):
                    add_training = phase == kwargs.get("template", "pre")
                    add_full_pre = phase == kwargs.get("template", "pre")
                    self.results_dict[nameExp][mouse_full_name][phase] = Mouse_Results(
                        dir,
                        mouse_name=mouse_nb,
                        manipe=manipe,
                        nameExp=nameExp,
                        phase=suffix.strip("_"),
                        exp_index=exp_index,
                        isTransformer=isTransformer
                        if isTransformer is not None
                        else "transformer" in nameExp.lower(),
                        windows=window_tmp,
                        transform_w_log=transform_w_log
                        if transform_w_log is not None
                        else "log" in nameExp.lower(),
                        denseweight=denseweight,
                        add_training=add_training,
                        add_full_pre=add_full_pre,
                        **kwargs,
                    )

                    try:
                        self.results_dict[nameExp][mouse_full_name][phase].load_data(
                            suffixes=[suffix],
                            add_training=phase == kwargs.get("template", "pre"),
                            add_full_pre=phase == kwargs.get("template", "pre"),
                            load_pickle=kwargs.get("load_pickle", False),
                        )
                        found_training = True
                        if kwargs.get("load_bayes", False) or kwargs.get(
                            "which", "ann"
                        ) in ["both", "bayes"]:
                            self.results_dict[nameExp][mouse_full_name][
                                phase
                            ].load_bayes(
                                suffixes=[suffix],
                                add_training=phase == kwargs.get("template", "pre"),
                                add_full_pre=phase == kwargs.get("template", "pre"),
                                **kwargs,
                            )
                    except FileNotFoundError:
                        self.results_dict[nameExp][mouse_full_name][phase].load_data(
                            suffixes=[suffix],
                            add_training=False,
                            add_full_pre=False,
                            load_pickle=kwargs.get("load_pickle", False),
                        )
                        if kwargs.get("load_bayes", False) or kwargs.get(
                            "which", "ann"
                        ) in ["both", "bayes"]:
                            self.results_dict[nameExp][mouse_full_name][
                                phase
                            ].load_bayes(
                                suffixes=[suffix],
                                add_training=False,
                                add_full_pre=False,
                                **kwargs,
                            )

        if found_training and "training" not in self.phases:
            self.phases.append("training")
            self.suffixes.append("_training")

            if "pre" in self.phases:
                self.phases.append("full_pre")
                self.suffixes.append("_full_pre")

        if kwargs.get("df", None) is None:
            try:
                self.convert_to_df()
            except Exception as e:
                print(f"Issues converting to DataFrame: {e}")

    def convert_to_df(self, redo=False, redo_sub=False):
        """
        Convert the results_dict to a pandas DataFrame.
        Collects row dictionaries from all windows, mice, and experiments
        and packages them cleanly into a single MultiIndexed DataFrame.
        """
        if (
            hasattr(self, "results_df")
            and not redo
            and isinstance(self.results_df, pd.DataFrame)
            and self.results_df.shape[0] > 0
        ):
            print(
                "Results DataFrame for Results_Loader already exists. Use redo=True to recreate it."
            )
            return self.results_df

        # Calculate total iterations for progress bar
        total_iterations = sum(
            len([p for p in phases.keys() if p != "training" and p != "full_pre"])
            for mice in self.results_dict.values()
            for phases in mice.values()
        )

        # Pre-allocate a single master list for ALL row dictionaries
        df_list = []

        # Create progress bar
        with tqdm(total=total_iterations, desc="Processing results") as pbar:
            for nameExp, mice in self.results_dict.items():
                for mouse_name, phases in mice.items():
                    for phase, results in phases.items():
                        if phase == "training" or phase == "full_pre":
                            continue

                        sub_df = copy.deepcopy(
                            results.convert_to_df(redo=redo_sub, disable=True)
                        )
                        if (
                            results is None
                            or not hasattr(results, "results_df")
                            or results.results_df is None
                        ):
                            print(
                                f"Warning: No results DataFrame for {nameExp} - {mouse_name} - {phase}. Skipping."
                            )
                            continue
                        if not isinstance(sub_df.index, pd.RangeIndex):
                            sub_df = sub_df.reset_index()

                        # 2. Inject the contextual loop variables as columns
                        sub_df["nameExp"] = nameExp
                        sub_df["mouse_name"] = mouse_name

                        # Add a direct reference back to the original results object row-by-row
                        sub_df["results"] = results

                        # Accumulate the DataFrame slice
                        df_list.append(sub_df)

                        pbar.update(1)

        if df_list:
            data = pd.concat(df_list, ignore_index=True)
            sort_keys = [
                k for k in ["mouse", "mouse_name", "phase"] if k in data.columns
            ]
            if sort_keys:
                data = data.sort_values(by=sort_keys)

            index_keys = ["nameExp", "mouse_name", "manipe", "phase", "winMS"]
            actual_index_keys = [k for k in index_keys if k in data.columns]
            data.set_index(actual_index_keys, inplace=True)
            self.results_df = data

            return self.results_df

    def __getitem__(self, key):
        """
        Get the results for a specific mouse name and phase.

        Args:
            key (str): Mouse name and phase in the format 'mouse_name_phase'.

        Returns:
            Mouse_Results: The Mouse_Results object for the specified mouse and phase.
        """
        try:
            mouse_name, phase = key.split("_")
        except ValueError:
            # simply extract M + Number as mouse_name and the rest as phase
            import re

            mouse_name_match = re.match(r"(M\d+)(.*)", key)
            if mouse_name_match:
                mouse_name = mouse_name_match.group(1)
                phase = mouse_name_match.group(2).lstrip("_")
            else:
                raise ValueError(
                    f"Key '{key}' is not in the expected format 'mouse_name_phase'."
                )
        if mouse_name in self.results_dict and phase in self.results_dict[mouse_name]:
            return self.results_dict[mouse_name][phase]
        else:
            raise KeyError(f"Results for {key} not found.")

    def __repr__(self):
        """
        String representation of the Results_Loader object.
        Returns a table summary of the object, including the nameExp, mice names, phases, time windows, and a preview of the results DataFrame.
        """
        # Create the header
        result = f"\n{self.__class__.__name__} Object\n"
        result += "=" * 50 + "\n\n"

        # Create table headers
        headers = ["NameExp", "Names", "Phases", "TimeWindows"]

        # Calculate column widths based on content
        col_widths = []
        data_columns = [self.nameExp, self.mice_names, self.phases, self.timeWindows]

        for i, (header, column) in enumerate(zip(headers, data_columns)):
            # Convert all items to strings to calculate max width
            str_items = [str(item) for item in column] + [header]
            col_widths.append(max(len(item) for item in str_items))

        # Create format string for rows
        row_format = " | ".join([f"{{:<{width}}}" for width in col_widths])

        # Add table header
        result += row_format.format(*headers) + "\n"
        result += "-" * (sum(col_widths) + 3 * (len(headers) - 1)) + "\n"

        # Add data rows
        max_rows = max(len(col) for col in data_columns)
        for i in range(max_rows):
            row_data = []
            for column in data_columns:
                if i < len(column):
                    row_data.append(str(column[i]))
                else:
                    row_data.append("")  # Empty cell if column is shorter
            result += row_format.format(*row_data) + "\n"

        # Add dataframe section
        result += "\n" + "=" * 50 + "\n"
        result += "DataFrame Head:\n"
        result += "-" * 20 + "\n"

        if hasattr(self, "results_df") and self.results_df is not None:
            # Convert dataframe head to string with nice formatting
            df_str = str(self.results_df.head())
            result += df_str
        else:
            result += "No dataframe available"

        return result

    def __str__(self):
        """
        String representation of the Results_Loader object.
        """
        return str(self.results_df.head())

    def save(self, path: Optional[str] = None):
        """
        Save the Results_Loader object to a pickle file.

        Args:
            path (str): Path to save the pickle file.
        """
        import dill as pickle

        if path is None:
            path = "results_loader.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __add__(self, other):
        """
        Add two Results_Loader objects together.
        This will concatenate the results DataFrames of both objects, as well as their results_dict.

        Args:
            other (Results_Loader): Another Results_Loader object to add.

        Returns:
            Results_Loader: A new Results_Loader object with combined results.
        """

        combined_results_dict = self.results_dict.copy()
        for nameExp, mice in other.results_dict.items():
            if nameExp not in combined_results_dict:
                combined_results_dict[nameExp] = {}
            for mouse_name, phases in mice.items():
                if mouse_name not in combined_results_dict[nameExp]:
                    combined_results_dict[nameExp][mouse_name] = {}
                for phase, results in phases.items():
                    combined_results_dict[nameExp][mouse_name][phase] = results

        combined_results_df = pd.concat(
            [self.results_df, other.results_df], ignore_index=True
        )
        nameExp = list(combined_results_dict.keys())
        timeWindows = (
            self.timeWindows.copy() + other.timeWindows.copy()
            if self.timeWindows != "all"
            else "all"
        )
        phases = self.phases + other.phases if self.phases is not None else other.phases
        # get only unique timeWindows
        if isinstance(timeWindows, list):
            timeWindows = list(set(timeWindows))
        if isinstance(phases, list):
            phases = list(set(phases))

        return Results_Loader.from_dict_and_df(
            dir=self.Dir,
            mice_nb=self.mice_nb + other.mice_nb,
            mice_manipes=self.mice_manipes + other.mice_manipes,
            dict=combined_results_dict,
            df=combined_results_df,
            nameExp=nameExp,
            timeWindows=timeWindows,
            phases=phases,
        )

    def __iadd__(self, other):
        """
        In-place addition of two Results_Loader objects.
        This will concatenate the results DataFrames of both objects, as well as their results_dict.

        Args:
            other (Results_Loader): Another Results_Loader object to add.

        Returns:
            Results_Loader: The current Results_Loader object with combined results.
        """
        self.results_dict.update(other.results_dict)
        self.results_df = pd.concat(
            [self.results_df, other.results_df], ignore_index=True
        )
        nameExp = list(self.results_dict.keys())
        timeWindows = (
            self.timeWindows.copy() + other.timeWindows.copy()
            if self.timeWindows != "all"
            else "all"
        )
        phases = self.phases + other.phases if self.phases is not None else other.phases
        # get only unique timeWindows
        if isinstance(timeWindows, list):
            timeWindows = list(set(timeWindows))
        if isinstance(phases, list):
            phases = list(set(phases))

        return Results_Loader.from_dict_and_df(
            dir=self.Dir,
            mice_nb=self.mice_nb + other.mice_nb,
            mice_manipes=self.mice_manipes + other.mice_manipes,
            dict=self.results_dict,
            df=self.results_df,
            nameExp=nameExp,
            timeWindows=timeWindows,
            phases=phases,
        )

    def apply_analysis(self, redo=False):
        """
        Apply common analysis metrics to the results DataFrame.
        Optimized for global MultiIndex datasets using parallel core execution.
        """
        current_index_names = list(self.results_df.index.names)
        if "mean_error" in self.results_df.columns and not redo:
            print("Analysis already applied to the DataFrame.")
            return self.results_df

        flat_df = self.results_df.reset_index()
        # if we redo, drop all columns we will create
        columns_to_drop = [
            "error",
            "mean_error",
            "lin_error",
            "mean_lin_error",
            "predLossThresholderror_selected",
            "mean_error_selected",
            "lin_error_selected",
            "mean_lin_error_selected",
            "asymmetry_index_on_predictedtrue_binary_direction",
            "predicted_binary_direction",
            "training_scalar_index",
            "training_asymmetry_index",
            "real_asymmetry_ratiopredicted_asymmetry_ratio",
            "predicted_asymmetry_ratio_on_selected",
            "predicted_asymmetry_ratio_normalized",
            "selected_predicted_asymmetry_ratio_normalized",
        ]
        if redo:
            flat_df = flat_df.drop(columns=columns_to_drop, errors="ignore")

        def process_row(row):
            res = {}

            # Base validation flags
            has_pred = row["featurePred"] is not None and row["featureTrue"] is not None
            has_lin = row["linearPred"] is not None and row["linearTrue"] is not None
            has_loss = row["predLoss"] is not None

            # 1. Base Errors and Speed
            if has_pred:
                errors = np.linalg.norm(
                    row["featurePred"] - row["featureTrue"], axis=1
                ).astype(np.float32)
                res["error"] = errors
                res["mean_error"] = (
                    np.nanmean(errors, dtype=np.float32) * np.ones_like(errors)
                ).astype(np.float32)

            if has_lin:
                lin_errors = np.abs(row["linearPred"] - row["linearTrue"]).astype(
                    np.float32
                )
                res["lin_error"] = lin_errors
                res["mean_lin_error"] = (
                    np.nanmean(lin_errors, dtype=np.float32) * np.ones_like(lin_errors)
                ).astype(np.float32)

            # 2. Selected metrics (lowest 20% loss window evaluation)
            if has_loss:
                threshold = np.quantile(row["predLoss"], 0.2).astype(np.float32)
                res["predLossThreshold"] = (
                    threshold * np.ones_like(row["predLoss"])
                ).astype(np.float32)
                mask = (row["predLoss"] <= threshold).astype(bool)

                if has_pred:
                    errors_selected = copy.deepcopy(errors)
                    errors_selected[~mask] = np.nan
                    res["error_selected"] = errors_selected
                    res["mean_error_selected"] = (
                        np.nanmean(errors_selected, dtype=np.float32)
                        * np.ones_like(errors_selected)
                    ).astype(np.float32)

                    # Fetch custom object pointer mapping safely from row values
                    res["asymmetry_index_on_selected_predicted"] = (
                        np.array(
                            row["results"].get_training_imbalance(
                                positions=row["featurePred"][mask]
                            ),
                            dtype=np.float32,
                        )
                        * np.ones_like(errors_selected)
                    ).astype(np.float32)

                if has_lin:
                    lin_errors_select = copy.deepcopy(lin_errors)
                    lin_errors_select[~mask] = np.nan
                    res["lin_error_selected"] = lin_errors_select
                    res["mean_lin_error_selected"] = (
                        np.nanmean(lin_errors_select, dtype=np.float32)
                        * np.ones_like(lin_errors_select)
                    ).astype(np.float32)

            # 3. Indices and Directions
            if has_pred:
                res["asymmetry_index_on_predicted"] = (
                    np.array(
                        row["results"].get_training_imbalance(
                            positions=row["featurePred"]
                        ),
                        dtype=np.float32,
                    )
                    * np.ones(row["featurePred"].shape[0], dtype=np.float32)
                ).astype(np.float32)

            if has_lin:
                res["true_binary_direction"] = (
                    np.array(
                        row["results"].data_helper._get_traveling_direction(
                            row["linearTrue"]
                        ),
                        dtype=np.float32,
                    )
                    * np.ones_like(row["linearTrue"])
                ).astype(np.float32)

                res["predicted_binary_direction"] = (
                    np.array(
                        row["results"].data_helper._get_traveling_direction(
                            row["linearPred"]
                        ),
                        dtype=np.float32,
                    )
                    * np.ones_like(row["linearPred"])
                ).astype(np.float32)

            return pd.Series(res)

        analysis_columns = flat_df.apply(process_row, axis=1)

        # Join the evaluated metrics back into our working flat DataFrame
        for col in analysis_columns.columns:
            flat_df[col] = analysis_columns[col]
        # --- 4. FIXED VECTORIZED RATIO CALCULATIONS ---
        group_keys = ["nameExp", "mouse_name", "manipe", "winMS"]
        group_keys = [k if k in flat_df.columns else "mouse" for k in group_keys]

        # Isolate training rows cleanly
        training_df = flat_df[flat_df["phase"] == "training"].copy()

        if not training_df.empty:
            # Extract the true scalar value out of the array cell
            # Since it's an array of repeated values, we grab the very first element [0]
            training_df["training_scalar_index"] = training_df["asymmetry_index"].apply(
                lambda x: (
                    x.flatten()[0] if isinstance(x, np.ndarray) and x.size > 0 else x
                )
            )

            # Group by and extract the scalar baseline profiles
            training_values = (
                training_df.groupby(group_keys)["training_scalar_index"]
                .first()
                .reset_index()
                .rename(columns={"training_scalar_index": "training_asymmetry_index"})
            )

            # Merge the single numeric baseline value back into your master flat DataFrame
            flat_df = flat_df.merge(training_values, on=group_keys, how="left")
        else:
            # Fallback if no training records exist in this tracking block
            flat_df["training_asymmetry_index"] = np.nan

        # Now train_idx becomes a simple scalar column (or NaN), which broadcasts
        # beautifully across your target data columns containing arrays!
        train_idx = flat_df["training_asymmetry_index"].replace(0, np.nan)

        # Convert train_idx into a pandas Series alignment value for seamless row broadcasting
        # This allows array / scalar division to work natively element-by-element across every row!
        flat_df["real_asymmetry_ratio"] = flat_df["asymmetry_index"] / train_idx
        flat_df["predicted_asymmetry_ratio"] = (
            flat_df["asymmetry_index_on_predicted"] / train_idx
        )
        flat_df["predicted_asymmetry_ratio_on_selected"] = (
            flat_df["asymmetry_index_on_selected_predicted"] / train_idx
        )

        # To handle dividing an array column by an array column, use element-wise custom operations
        def divide_array_columns(row, num_col, denom_col):
            num = row[num_col]
            denom = row[denom_col]
            if isinstance(num, np.ndarray) and isinstance(denom, np.ndarray):
                # Safe division: mask out denominators that are equal to 0
                safe_denom = np.where(denom == 0, np.nan, denom)
                return num / safe_denom
            return np.nan

        print("Normalizing ratio profiles across nested array channels...")
        flat_df["predicted_asymmetry_ratio_normalized"] = flat_df.apply(
            divide_array_columns,
            axis=1,
            args=("asymmetry_index_on_predicted", "real_asymmetry_ratio"),
        )
        flat_df["selected_predicted_asymmetry_ratio_normalized"] = flat_df.apply(
            divide_array_columns,
            axis=1,
            args=("asymmetry_index_on_selected_predicted", "real_asymmetry_ratio"),
        )

        # go back to full dim for training_asymmetry_index
        flat_df["training_asymmetry_index"] = flat_df.apply(
            lambda row: (
                row["training_asymmetry_index"] * np.ones_like(row["timeNN"])
                if isinstance(row["training_asymmetry_index"], (int, float))
                else row["training_asymmetry_index"]
            ),
            axis=1,
        )

        flat_df = flat_df.loc[:, ~flat_df.columns.duplicated()].copy()

        if current_index_names and not all(x is None for x in current_index_names):
            flat_df.set_index(current_index_names, inplace=True)
        else:
            # Fallback default array index configuration matching your description
            default_keys = [
                "nameExp",
                "mouse_name",
                "manipe",
                "phase",
                "winMS",
                "mouse",
            ]
            actual_keys = [k for k in default_keys if k in flat_df.columns]
            flat_df.set_index(actual_keys, inplace=True)

        self.results_df = flat_df
        return self.results_df

    def add_breathing(self, redo=False):
        """
        Add breathing-related metrics to the results DataFrame.
        This method computes breathing rate and amplitude from the raw breathing signal,
        and adds these as new columns to the results DataFrame. It also handles any necessary data cleaning and normalization steps.

        If found, it will also add the heart rate as a new column, extracted from the same raw signal if available.
        """
        if self.results_df is None:
            raise ValueError("Please run evaluate() before adding breathing data.")
        if "breathing_rate" in self.results_df.columns and not redo:
            print("Breathing column already exists. Set redo=True to overwrite.")
            return self.results_df

        # Ensure we have the necessary columns to compute breathing
        required_cols = ["timeNN", "results"]

        for col in required_cols:
            if col not in self.results_df.columns:
                raise ValueError(
                    f"Missing required column '{col}' to compute breathing."
                )
        for _, df in self.results_df.groupby(level="mouse_name"):
            res: Mouse_Results = df.iloc[0].results
            try:
                respi = res.DataHelper.get_respi_data()
            except FileNotFoundError:
                respi = res.DataHelper.get_respi_data(folder=res.network_path)

            spectro, power = res.DataHelper.compute_breathing_rate(spectro_tsd=respi)
            res.find_path()
            found_ekg = False

            if os.path.exists(
                os.path.join(res.DataHelper.folder, "HeartBeatInfo.mat")
            ) or os.path.exists(os.path.join(res.network_path, "HeartBeatInfo.mat")):
                folder = (
                    res.DataHelper.folder
                    if os.path.exists(
                        os.path.join(res.DataHelper.folder, "HeartBeatInfo.mat")
                    )
                    else res.network_path
                )
                heart_file = loadmat(os.path.join(folder, "HeartBeatInfo.mat"))
                if "EKG" not in heart_file:
                    raise KeyError(
                        f"EKG data not found in HeartBeatInfo.mat for {res.mouse_name}. Skipping EKG analysis."
                    )
                if (
                    "HBRate" not in heart_file["EKG"].dtype.names
                    or "LFP" not in heart_file["EKG"].dtype.names
                ):
                    raise KeyError(
                        f"EKG data missing 'HBRate' or 'LFP' in HeartBeatInfo.mat for {res.mouse_name}. Skipping EKG analysis."
                    )

                hb_rate = clean_mat_structure(heart_file["EKG"]["HBRate"])
                hb_lfp = clean_mat_structure(heart_file["EKG"]["LFP"])

                hb_rate_tsd = Tsd(
                    t=np.array(hb_rate["t"]) / 1e4, d=np.array(hb_rate["data"])
                )
                hb_lfp_tsd = Tsd(
                    t=np.array(hb_lfp["t"]) / 1e4, d=np.array(hb_lfp["data"])
                )

                found_ekg = True

            for phase, sub in df.groupby("phase"):
                if phase == "training" or phase == "full_pre":
                    continue
                subres = sub.iloc[0].results
                subres.find_session_epochs()

                updated_subsubs = []

                for subphase, subsub in subres.results_df.groupby("phase"):
                    time_mask = getattr(subres, subphase)
                    subsub["lfp_bulb"] = respi.restrict(time_mask)
                    subsub["breathing_rate"] = spectro.restrict(time_mask)
                    subsub["breathing_power"] = power.restrict(time_mask)

                    if found_ekg:
                        subsub["lfp_ekg"] = hb_lfp_tsd.restrict(time_mask)
                        subsub["heart_rate"] = hb_rate_tsd.restrict(time_mask)

                    updated_subsubs.append(subsub)

                if updated_subsubs:
                    subres.results_df = pd.concat(updated_subsubs)

                self.results_df.loc[sub.index, "results"] = subres

        # now that each subres has the new columns, we need to extract them back to the main results_df
        base_df = self.results_df.copy()

        concat_results_df = self.convert_to_df(True)
        cols_to_add = base_df.columns.difference(concat_results_df.columns)
        self.results_df = pd.concat([concat_results_df, base_df[cols_to_add]], axis=1)
        return self.results_df

    def add_zone_epoch(self, redo=False):
        if self.results_df is None:
            raise ValueError("Please run evaluate() before adding zone epoch data.")
        if (
            any(f"{zone}_epoch" in self.results_df.columns for zone in ZONELABELS)
            and not redo
        ):
            print("zone_epoch column already exists. Set redo=True to overwrite.")
            return self.results_df

        # Ensure we have the necessary columns to compute zone_epoch
        required_cols = ["results", "timeNN"]

        for col in required_cols:
            if col not in self.results_df.columns:
                raise ValueError(
                    f"Missing required column '{col}' to compute zone_epoch."
                )
        for _, row in self.results_df.iterrows():
            row["results"].DataHelper._compute_zone_epochs()

        for zone in ZONELABELS:
            self.results_df[f"{zone}_epoch"] = self.results_df[
                ["timeNN", "results"]
            ].apply(
                lambda row: (
                    Ts(row["timeNN"])
                    .value_from(getattr(row["results"].DataHelper, f"{zone}_tsd"))
                    .values
                ),
                axis=1,
            )

    def save_to_mat(self, df, path):
        from scipy.io import savemat

        mdict = dict()
        for col in df.drop(columns=["results"], errors="ignore").columns:
            mdict[col] = df[col].to_numpy()
        savemat(os.path.realpath(path), mdict)

    def save_mat_for_each_mouse(self, output_dir: str):
        """
        Saves the processed results DataFrame into .mat files for each mouse.

        Parameters:
        - output_dir: Directory where the .mat files will be saved.
        """
        if self.results_df is None:
            raise ValueError(
                "Results DataFrame is not available. Please run process_results() first."
            )

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        for mouse_name, group_df in self.results_df.groupby("mouse_name"):
            clean_df = self.get_clean_df(group_df)
            output_path = os.path.join(output_dir, f"{mouse_name}_results.mat")
            self.save_to_mat(clean_df, output_path)

    def _compute_epoch_mask(
        self, df: pd.DataFrame, col_name: str, fetch_method: str, attr_name: str
    ) -> pd.Series:
        """Helper method to dynamically compute an epoch mask column via .apply()"""
        if "results" not in df.columns:
            raise ValueError(
                f"Requested '{col_name}' column but 'results' is not found in the DataFrame."
            )

        def _get_mask(row):
            # 1. Dynamically find and call the fetch method (e.g., get_ripples_epochs)
            fetch_func = getattr(row.results.DataHelper, fetch_method)
            fetch_func()

            # 2. Dynamically pull the populated attribute (e.g., ripples_epochs)
            epochs = getattr(row.results.DataHelper, attr_name)

            # 3. Compute and return the mask matching timeNN
            return inEpochsMask(row["timeNN"], epochs).flatten()

        return df.apply(_get_mask, axis=1)

    def get_clean_df(
        self, df: Optional[pd.DataFrame] = None, cols: Optional[list] = None
    ) -> pd.DataFrame:
        """Flattens nested variable-length time-series data dynamically based on the

        specific columns requested, ensuring uniform timepoint alignment via

        pynapple.
        """
        if df is None:
            warn(
                "No DataFrame provided to get_clean_df. Using self.results_df. Ensure this is what you intend."
            )
            df = self.results_df.copy()
        else:
            # Avoid mutating the original dataframe passed to the function
            df = df.copy()

        if cols is None:
            cols = EXPORT_COLS

        name_time_column = "timeNN"

        # Standardize names up front
        col_name_mapping = {
            "speed": "alignedSpeed",
            "time": name_time_column,
            "certainty": "predLoss",
            "linear": "linearTrue",
            "linear_hat": "linearPred",
            "is_fast": "speedMask",
        }
        cols = [col_name_mapping.get(col, col) for col in cols]
        name_time_column = col_name_mapping.get(name_time_column, name_time_column)

        assert name_time_column in df.columns, (
            f"Time column '{name_time_column}' is not present in the DataFrame."
        )

        restrict_targets = [
            "breathing_rate",
            "breathing_power",
            "lfp_bulb",
            "heart_rate",
        ]
        restrict_cols_present = [c for c in cols if c in restrict_targets]
        need_to_restrict = len(restrict_cols_present) > 0

        which_restrictor = next((c for c in cols if c in restrict_targets), None)

        # 1. Configuration Mappings
        if df.iloc[0].featureTrue.shape[1] == 4:
            true_mapping = {"x": 0, "y": 1, "head_dir": 2, "thigmo": 3}
            pred_mapping = {"x_hat": 0, "y_hat": 1, "head_dir_hat": 2, "thigmo_hat": 3}
        elif df.iloc[0].featureTrue.shape[1] == 3:
            try:
                cols.remove("thigmo")
                cols.remove("thigmo_hat")
            except ValueError:
                pass
            true_mapping = {"x": 0, "y": 1, "head_dir": 2}
            pred_mapping = {"x_hat": 0, "y_hat": 1, "head_dir_hat": 2}

        epoch_configs = {
            "is_ripples": {
                "fetch_method": "get_ripples_epochs",
                "attr_name": "ripples_epochs",
            },
            "is_freezing": {
                "fetch_method": "get_freeze_epochs",
                "attr_name": "freeze_epochs",
            },
            "is_stim": {
                "fetch_method": "get_stim_epochs",
                "attr_name": "stim_epochs",
            },
        }

        # 2. Preprocess Metadata, Index levels, and Custom Mask Arrays
        for col in cols:
            if col in true_mapping or col in pred_mapping:
                continue
            if col in df.index.names and col not in df.columns:
                df[col] = df.index.get_level_values(col)
            if col in epoch_configs and col not in df.columns:
                cfg = epoch_configs[col]
                df[col] = self._compute_epoch_mask(
                    df, col, cfg["fetch_method"], cfg["attr_name"]
                )
            if col not in df.columns:
                raise ValueError(f"Requested column '{col}' not found in DataFrame.")
            if col == "phase":
                df[col] = df[col].map(PHASE_MAPPING)
            if col == "mouse":
                df[col] = df[col].astype(int)

        if need_to_restrict:
            # Automatically detect any column that contains array data per row
            # This ensures features, speeds, and custom masks are ALL processed
            array_cols = [
                c
                for c in df.columns
                if isinstance(df[c].iloc[0], (np.ndarray, list)) and c in cols
            ]

            array_cols.append(
                name_time_column
            )  # Ensure timeNN is included for remapping
            array_cols.append(
                which_restrictor
            )  # Ensure the restrictor column is included
            if any(c in true_mapping for c in cols):
                array_cols.append("featureTrue")
            if any(c in pred_mapping for c in cols):
                array_cols.append("featurePred")

            array_cols = list(set(array_cols))  # Remove duplicates if any

            def align_row_time_series(row):
                times = row[name_time_column]
                time_ts = Ts(t=times)

                target_tsd = row[which_restrictor].restrict(time_ts.time_support)
                expected_len = len(target_tsd.values)

                # --- CASE 1: EMPTY TIME SUPPORT FOR THIS PHASE ---
                if expected_len == 0:
                    fallback_len = len(times)
                    for c in array_cols:
                        if c in cols or c == which_restrictor:
                            val = row[c]
                            if isinstance(val, (int, float, str)) or (
                                hasattr(val, "__len__")
                                and len(val) == 1
                                and not isinstance(val, (np.ndarray, list))
                            ):
                                continue

                            if isinstance(val, np.ndarray) and len(val.shape) > 1:
                                row[c] = [np.full((fallback_len, val.shape[1]), np.nan)]
                            else:
                                row[c] = [np.full(fallback_len, np.nan)]
                    return row

                for c in array_cols:
                    if c == which_restrictor:
                        row[c] = [target_tsd.values]
                        continue

                    val = np.array(row[c])

                    if len(val.shape) == 1 or val.shape[1] <= 1:
                        source_tsd = Tsd(t=times, d=val)
                        # Map to the new target baseline timestamps
                        row[c] = [target_tsd.value_from(source_tsd).values]
                    else:
                        source_tsd = TsdFrame(t=times, d=val)
                        row[c] = [target_tsd.value_from(source_tsd).values]

                return row

            df[array_cols] = df[array_cols].apply(align_row_time_series, axis=1)

            # UNWRAP: Restore the inner numpy arrays back to direct cell values
            for c in array_cols:
                df[c] = df[c].apply(lambda x: x[0] if isinstance(x, list) else x)

            print("Time alignment complete.")

        # 4. Establish accurate tracking for final array lengths
        if "featureTrue" in df.columns:
            lengths = [len(x) for x in df["featureTrue"]]
        elif "featurePred" in df.columns:
            lengths = [len(x) for x in df["featurePred"]]
        elif need_to_restrict:
            lengths = [len(x) for x in df[which_restrictor]]
        else:
            raise ValueError("Cannot determine time-series lengths securely.")

        # 5. Extract and Construct long-form Matrix arrays
        true_matrix = None
        pred_matrix = None
        extracted_data = {}

        for col in cols:
            if col in true_mapping:
                if true_matrix is None:
                    true_matrix = np.concatenate(df["featureTrue"].values, axis=0)
                idx = true_mapping[col]
                extracted_data[col] = true_matrix[:, idx]

            elif col in pred_mapping:
                if pred_matrix is None:
                    pred_matrix = np.concatenate(df["featurePred"].values, axis=0)
                idx = pred_mapping[col]
                extracted_data[col] = pred_matrix[:, idx]

            elif col in df.columns:
                first_val = df[col].iloc[0]
                if isinstance(first_val, (np.ndarray, list)):
                    extracted_data[col] = np.concatenate(df[col].values)
                else:
                    extracted_data[col] = np.repeat(df[col].values, lengths)
            else:
                raise KeyError(f"Column '{col}' not found in configuration layouts.")

        # 6. Revert standardized names back to their requested external aliases
        inverted_mapping = {v: k for k, v in col_name_mapping.items()}
        clean_df = pd.DataFrame(extracted_data)
        clean_df.rename(columns=inverted_mapping, inplace=True)

        for col in clean_df.columns:
            if col.startswith("is") and len(np.unique(clean_df[col].values)) == 2:
                clean_df[col] = clean_df[col].astype(bool)

        if all([col in clean_df.columns for col in EPOCH_MAPPING.keys()]):
            epoch_cols = list(EPOCH_MAPPING.keys())
            clean_df["ZoneEpoch"] = (
                clean_df.reset_index(drop=True)[epoch_cols]
                .astype(bool)
                .idxmax(axis=1)
                .map(EPOCH_MAPPING)
            )

        return clean_df

    def plot_1d_tuning_curves_during_events_from_clean(
        self,
        clean_df: Optional[pd.DataFrame] = None,
        feature: str = "linear",
        phase="cond",
        which: Union[str, List[str]] = "ripples",
        plot_bias: bool = False,
        path: Optional[str] = None,
        title_suffix: Optional[str] = None,
    ):
        if clean_df is None:
            clean_df = self.get_clean_df()
        else:
            clean_df = clean_df.copy()

        # we assume otherwise the dataframe is already flattened in the correct format, and subsetted for interesting manipe etc
        clean_df.reset_index(drop=False, inplace=True)

        if isinstance(which, str):
            which = [which]

        event_mask = np.zeros(len(clean_df), dtype=bool)
        if "ripples" in which:
            event_mask += clean_df["is_ripples"].astype(bool)

        if "freezing" in which:
            event_mask += clean_df["is_freezing"].astype(bool)

        if "fast" in which:
            event_mask += clean_df["is_fast"].astype(bool)

        if "stim" in which:
            event_mask += clean_df["is_stim"].astype(bool)

        event_mask = event_mask.astype(bool)
        phase_mask = clean_df["phase"] == PHASE_MAPPING[phase]

        true_feature = clean_df[feature][event_mask & phase_mask].values
        pred_feature = clean_df[f"{feature}_hat"][event_mask & phase_mask].values

        true_density, centers = get_1d_tuning_curve(
            true_feature, np.ones_like(true_feature, dtype=bool)
        )
        pred_density, _ = get_1d_tuning_curve(
            pred_feature, np.ones_like(pred_feature, dtype=bool)
        )

        if plot_bias:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1, ax2 = axs
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

        ax1.plot(
            centers, true_density, label=f"True Position {which}", color="black", lw=2
        )
        ax1.plot(
            centers,
            pred_density,
            label="Predicted Position",
            color="red",
            linestyle="--",
        )
        ax1.fill_between(
            centers,
            pred_density,
            alpha=0.3,
            color="red",
        )

        title = f"Spatial Representation During {which} in {phase} (n_evts = {clean_df[phase_mask & event_mask].shape[0]})."

        if title_suffix is None:
            title_suffix = f"Gathered from {clean_df[phase_mask]['mouse_name'].unique().shape[0]} PAG mice."
        title += f"\n{title_suffix}"

        ax1.set_title(title)
        ax1.set_ylabel("Normalized Density")
        ax1.legend()

        if plot_bias:
            bias = pred_density - true_density
            ax2.bar(
                centers,
                bias,
                width=(centers[1] - centers[0]),
                color="purple",
                alpha=0.7,
            )
            ax2.axhline(0, color="grey", lw=1)
            ax2.set_title(
                "Prediction Bias (Pred - True) during Freezing in Conditioning"
            )
            ax2.set_xlabel("Linearized Position (0-1)")
            ax2.set_ylabel("Bias Delta")
            ax2.legend()

        plt.tight_layout()
        if path is not None:
            plt.savefig(path)

        plt.show()

    @classmethod
    def from_dict_and_df(
        cls,
        dir: pd.DataFrame,
        mice_nb: List[int],
        mice_manipes: List[str],
        dict: dict,
        df: pd.DataFrame,
        nameExp: List[str] = None,
        timeWindows: List[int] = None,
        phases: List[str] = None,
        **kwargs,
    ):
        """
        Create a Results_Loader object from a dictionary and a DataFrame.

        Args:
            dir (pd.DataFrame): PathForExperiments DataFrame with columns for folder Results, mouse names, manipes, network paths, etc.
            mice_nb (List[int]): List of mouse numbers to filter results.
            dict (dict): Dictionary containing the results.
            df (pd.DataFrame): DataFrame containing the results.
            timeWindows (List[int]): List of time windows in milliseconds to filter results. If None, uses all available windows.

        Returns:
            Results_Loader: A new Results_Loader object.
        """
        return cls(
            dir=dir,
            mice_nb=mice_nb,
            mice_manipes=mice_manipes,
            dict=dict,
            df=df,
            timeWindows=timeWindows,
            nameExp=nameExp,
            phases=phases,
            **kwargs,
        )

    def mean_error_matrix_linerrors_by_speed_new(
        self,
        nbins=40,
        save=True,
        folder=None,
        show=False,
        nameExp_list=None,
        phase_list=None,
        winMS_list=None,
        removeMice_list=None,
    ):
        import cmcrameri.cm as cmc

        folder = folder or getattr(self, "folderFigures", None)
        used_df = self.results_df.copy()

        if removeMice_list is not None:
            if not isinstance(removeMice_list, list):
                removeMice_list = [removeMice_list]
            used_df = used_df.query("mouse_name not in @removeMice_list")

        grouped = used_df.groupby(["nameExp", "phase", "winMS"])

        # --- [Filtering Logic] ---
        if nameExp_list is not None:
            if not isinstance(nameExp_list, list):
                nameExp_list = [nameExp_list]
            grouped = grouped.filter(lambda x: x.name[0] in nameExp_list).groupby(
                ["nameExp", "phase", "winMS"]
            )
        if phase_list is not None:
            if not isinstance(phase_list, list):
                phase_list = [phase_list]
            grouped = grouped.filter(lambda x: x.name[1] in phase_list).groupby(
                ["nameExp", "phase", "winMS"]
            )
        if winMS_list is not None:
            if not isinstance(winMS_list, list):
                winMS_list = [winMS_list]
            winMS_list = [int(w) for w in winMS_list]
            grouped = grouped.filter(lambda x: int(x.name[2]) in winMS_list).groupby(
                ["nameExp", "phase", "winMS"]
            )

        # ---------------------------------------------------------------------
        # PASS 1: Extract data and determine the GLOBAL minimum sample size
        # ---------------------------------------------------------------------
        global_min_samples = float("inf")
        processed_groups = {}

        for (nameExp, phase, winMS), df in grouped:
            df = df.reset_index(drop=False)

            linPred_fast_list, linTrue_fast_list = [], []
            linPred_slow_list, linTrue_slow_list = [], []

            for _, row in df.iterrows():
                speed_mask = row["speedMask"]
                epochMask = np.ones_like(row["timeNN"], dtype=bool)

                # Fast
                mask_fast = speed_mask & epochMask
                linPred_fast_list.append(row["linearPred"][mask_fast])
                linTrue_fast_list.append(row["linearTrue"][mask_fast])

                # Slow
                mask_slow = ~speed_mask & epochMask
                linPred_slow_list.append(row["linearPred"][mask_slow])
                linTrue_slow_list.append(row["linearTrue"][mask_slow])

            # Flatten row-wise outputs into continuous arrays
            fast_pred = (
                np.concatenate(linPred_fast_list).reshape(-1)
                if linPred_fast_list
                else np.array([])
            )
            fast_true = (
                np.concatenate(linTrue_fast_list).reshape(-1)
                if linTrue_fast_list
                else np.array([])
            )

            slow_pred = (
                np.concatenate(linPred_slow_list).reshape(-1)
                if linPred_slow_list
                else np.array([])
            )
            slow_true = (
                np.concatenate(linTrue_slow_list).reshape(-1)
                if linTrue_slow_list
                else np.array([])
            )

            n_fast = len(fast_pred)
            n_slow = len(slow_pred)

            # Update our global minimum threshold with non-empty groups
            if n_fast > 0:
                global_min_samples = min(global_min_samples, n_fast)
            if n_slow > 0:
                global_min_samples = min(global_min_samples, n_slow)

            # Cache raw data subsets in memory to avoid repetitive parsing
            processed_groups[(nameExp, phase, winMS)] = {
                "fast_pred": fast_pred,
                "fast_true": fast_true,
                "slow_pred": slow_pred,
                "slow_true": slow_true,
            }

        # If no valid data points were found anywhere, terminate early
        if global_min_samples == float("inf") or global_min_samples == 0:
            print("Warning: No valid data found to compute matrices.")
            return

        print(
            f"Global matching enabled. Undersampling all conditions to: {global_min_samples} points."
        )

        # ---------------------------------------------------------------------
        # PASS 2: Apply global undersampling, build matrices, and calculate global vmax
        # ---------------------------------------------------------------------
        all_matrices = []
        global_vmax = 0.0
        rng = np.random.default_rng(
            seed=42
        )  # Structured random state for reproducibility

        for (nameExp, phase, winMS), data in processed_groups.items():
            fast_pred, fast_true = data["fast_pred"], data["fast_true"]
            slow_pred, slow_true = data["slow_pred"], data["slow_true"]

            # Undersample Fast
            if len(fast_pred) >= global_min_samples:
                idx_fast = rng.choice(
                    len(fast_pred), size=global_min_samples, replace=False
                )
                fast_pred = fast_pred[idx_fast]
                fast_true = fast_true[idx_fast]

            # Undersample Slow
            if len(slow_pred) >= global_min_samples:
                idx_slow = rng.choice(
                    len(slow_pred), size=global_min_samples, replace=False
                )
                slow_pred = slow_pred[idx_slow]
                slow_true = slow_true[idx_slow]

            # Compute Histograms
            H_fast, H_slow, xedges, yedges = None, None, None, None

            if len(fast_pred) > 0:
                H_fast, xedges, yedges = np.histogram2d(
                    fast_pred,
                    fast_true,
                    bins=(nbins, nbins),
                    range=[[0, 1], [0, 1]],
                )
                with np.errstate(invalid="ignore"):
                    row_sums = H_fast.sum(axis=1, keepdims=True)
                    H_fast = np.where(row_sums > 0, H_fast / row_sums, 0)
                global_vmax = max(global_vmax, H_fast.max())

            if len(slow_pred) > 0:
                H_slow, xedges, yedges = np.histogram2d(
                    slow_pred,
                    slow_true,
                    bins=(nbins, nbins),
                    range=[[0, 1], [0, 1]],
                )
                with np.errstate(invalid="ignore"):
                    row_sums = H_slow.sum(axis=1, keepdims=True)
                    H_slow = np.where(row_sums > 0, H_slow / row_sums, 0)
                global_vmax = max(global_vmax, H_slow.max())

            if H_fast is not None or H_slow is not None:
                extent = (
                    [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    if xedges is not None
                    else [0, 1, 0, 1]
                )
                all_matrices.append(
                    {
                        "metadata": (nameExp, phase, winMS),
                        "H_fast": H_fast,
                        "H_slow": H_slow,
                        "extent": extent,
                    }
                )

        # Fallback value if all computed arrays turned out empty
        if global_vmax == 0:
            global_vmax = 1.0

        # ---------------------------------------------------------------------
        # PASS 3: Plot everything with a shared dynamic global scale
        # ---------------------------------------------------------------------
        for item in all_matrices:
            nameExp, phase, winMS = item["metadata"]
            extent = item["extent"]

            fig, axes = plt.subplots(
                ncols=2, nrows=1, figsize=(11, 5), sharex=True, sharey=True
            )

            # Plot Fast Speeds
            ax_fast = axes[0]
            ax_fast.set_xlim(0, 1)
            ax_fast.set_ylim(0, 1)
            if item["H_fast"] is not None:
                im_fast = ax_fast.imshow(
                    item["H_fast"].T,
                    extent=extent,
                    cmap=cmc.batlow,
                    vmin=0,
                    vmax=global_vmax,  # Scale matches globally across all plots
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                )
                fig.colorbar(im_fast, ax=ax_fast)
            ax_fast.set_title("Fast speeds only")

            # Plot Slow Speeds
            ax_slow = axes[1]
            ax_slow.set_xlim(0, 1)
            ax_slow.set_ylim(0, 1)
            if item["H_slow"] is not None:
                im_slow = ax_slow.imshow(
                    item["H_slow"].T,
                    extent=extent,
                    cmap=cmc.batlow,
                    vmin=0,
                    vmax=global_vmax,  # Scale matches globally across all plots
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                )
                fig.colorbar(im_slow, ax=ax_slow)
            ax_slow.set_title("Slow speeds only")

            fig.suptitle(
                f"{nameExp} | Phase: {phase} | winMS: {winMS} (N={global_min_samples})"
            )
            fig.text(0.5, 0.04, "Predicted linPos", ha="center")
            fig.text(0.04, 0.5, "True linPos", va="center", rotation="vertical")
            fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])

            if save and folder is not None:
                fname = (
                    f"errorMatrix_{nameExp}_phase{phase}_win{winMS}_globalMatch_batlow"
                )
                fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                fig.savefig(os.path.join(folder, fname + ".svg"))

            if show:
                plt.show()
            plt.close(fig)

    def mean_error_matrix_linerrors_by_speed(
        self,
        nbins=40,
        save=True,
        folder=None,
        show=False,
        nameExp_list=None,
        phase_list=None,
        winMS_list=None,
        removeMice_list=None,
    ):
        import cmcrameri.cm as cmc

        folder = folder or getattr(self, "folderFigures", None)
        used_df = self.results_df.copy()

        if removeMice_list is not None:
            if not isinstance(removeMice_list, list):
                removeMice_list = [removeMice_list]
            used_df = used_df.query("mouse_name not in @removeMice_list")

        grouped = used_df.groupby(["nameExp", "phase", "winMS"])

        # --- [Filtering Logic remains the same as your original code] ---
        if nameExp_list is not None:
            if not isinstance(nameExp_list, list):
                nameExp_list = [nameExp_list]
            grouped = grouped.filter(lambda x: x.name[0] in nameExp_list).groupby(
                ["nameExp", "phase", "winMS"]
            )
        if phase_list is not None:
            if not isinstance(phase_list, list):
                phase_list = [phase_list]
            grouped = grouped.filter(lambda x: x.name[1] in phase_list).groupby(
                ["nameExp", "phase", "winMS"]
            )
        if winMS_list is not None:
            if not isinstance(winMS_list, list):
                winMS_list = [winMS_list]
            winMS_list = [int(w) for w in winMS_list]
            grouped = grouped.filter(lambda x: int(x.name[2]) in winMS_list).groupby(
                ["nameExp", "phase", "winMS"]
            )

        # ---------------------------------------------------------
        # PASS 1: Gather data and normalize rows locally [0, 1]
        # ---------------------------------------------------------
        all_matrices = []

        for (nameExp, phase, winMS), df in grouped:
            df = df.reset_index(drop=False)

            linPred_fast, linTrue_fast = [], []
            linPred_slow, linTrue_slow = [], []

            for _, row in df.iterrows():
                row["mouse_name"]
                speed_mask = row["speedMask"]
                epochMask = np.ones_like(row["timeNN"], dtype=bool)

                # Fast
                mask_fast = speed_mask & epochMask
                linPred_fast.append(row["linearPred"][mask_fast])
                linTrue_fast.append(row["linearTrue"][mask_fast])

                # Slow
                mask_slow = ~speed_mask & epochMask
                linPred_slow.append(row["linearPred"][mask_slow])
                linTrue_slow.append(row["linearTrue"][mask_slow])

            # Compute Histograms
            H_fast, H_slow, xedges, yedges = None, None, None, None

            if linPred_fast:
                # Note: We use density=False here because we will manually normalize rows by counts
                H_fast, xedges, yedges = np.histogram2d(
                    np.concatenate(linPred_fast).reshape(-1),
                    np.concatenate(linTrue_fast).reshape(-1),
                    bins=(nbins, nbins),
                    range=[[0, 1], [0, 1]],
                )
                # Local Row-wise Normalization (Maximum of each true position row becomes 1.0)
                # with np.errstate(invalid="ignore"):
                #     row_maxs = H_fast.max(axis=1, keepdims=True)
                #     H_fast = np.where(row_maxs > 0, H_fast / row_maxs, 0)
                # Normalize by row SUM instead of row MAX
                with np.errstate(invalid="ignore"):
                    row_sums = H_fast.sum(axis=1, keepdims=True)
                    H_fast = np.where(row_sums > 0, H_fast / row_sums, 0)

            if linPred_slow:
                H_slow, xedges, yedges = np.histogram2d(
                    np.concatenate(linPred_slow).reshape(-1),
                    np.concatenate(linTrue_slow).reshape(-1),
                    bins=(nbins, nbins),
                    range=[[0, 1], [0, 1]],
                )
                # # Local Row-wise Normalization
                # with np.errstate(invalid="ignore"):
                #     row_maxs = H_slow.max(axis=1, keepdims=True)
                #     H_slow = np.where(row_maxs > 0, H_slow / row_maxs, 0)
                #
                with np.errstate(invalid="ignore"):
                    row_sums = H_slow.sum(axis=1, keepdims=True)
                    H_slow = np.where(row_sums > 0, H_slow / row_sums, 0)

            if H_fast is not None or H_slow is not None:
                extent = (
                    [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    if xedges is not None
                    else [0, 1, 0, 1]
                )
                all_matrices.append(
                    {
                        "metadata": (nameExp, phase, winMS),
                        "H_fast": H_fast,
                        "H_slow": H_slow,
                        "extent": extent,
                    }
                )

        # ---------------------------------------------------------
        # PASS 2: Plot everything with a forced absolute limit of [0, 1]
        # ---------------------------------------------------------
        for item in all_matrices:
            nameExp, phase, winMS = item["metadata"]
            extent = item["extent"]

            fig, axes = plt.subplots(
                ncols=2, nrows=1, figsize=(11, 5), sharex=True, sharey=True
            )

            # Plot Fast Speeds
            ax_fast = axes[0]
            ax_fast.set_xlim(0, 1)
            ax_fast.set_ylim(0, 1)
            if item["H_fast"] is not None:
                im_fast = ax_fast.imshow(
                    item["H_fast"].T,
                    extent=extent,
                    cmap=cmc.batlow,
                    vmin=0,
                    vmax=1.0,  # Scale is now strictly 0 to 1 across ALL phases
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                )
                fig.colorbar(im_fast, ax=ax_fast)
            ax_fast.set_title("Fast speeds only")

            # Plot Slow Speeds
            ax_slow = axes[1]
            ax_slow.set_xlim(0, 1)
            ax_slow.set_ylim(0, 1)
            if item["H_slow"] is not None:
                im_slow = ax_slow.imshow(
                    item["H_slow"].T,
                    extent=extent,
                    cmap=cmc.batlow,
                    vmin=0,
                    vmax=1.0,  # Scale is now strictly 0 to 1 across ALL phases
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                )
                fig.colorbar(im_slow, ax=ax_slow)
            ax_slow.set_title("Slow speeds only")

            fig.suptitle(f"{nameExp} | Phase: {phase} | winMS: {winMS}")
            fig.text(0.5, 0.04, "Predicted linPos", ha="center")
            fig.text(0.04, 0.5, "True linPos", va="center", rotation="vertical")
            fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])

            if save and folder is not None:
                fname = f"errorMatrix_{nameExp}_phase{phase}_win{winMS}_rowNorm_batlow"
                fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                fig.savefig(os.path.join(folder, fname + ".svg"))

            if show:
                plt.show()
            plt.close(fig)

    def correlation_entropy_maxp_vs_KL(
        self, suffixes=None, save=True, folder=None, show=False
    ):
        """
        For each (nameExp, mouse, phase, winMS), load decoding_results pkl from Mouse_Results,
        compute KL loss, and plot correlations vs entropy and maxp.

        Args:
            suffixes (list[str], optional): Suffixes to load. If None, uses self.suffixes if present.
            save (bool): Whether to save figures.
            folder (str): Folder to save into. Defaults to self.folderFigures.
        """

        folder = folder or getattr(self, "folderFigures", None)
        suffixes = suffixes or getattr(self, "suffixes", [""])
        # Try to get ANN loss layer
        from neuroencoders.fullEncoder.nnUtils import GaussianHeatmapLosses

        try:
            loss_layer = GaussianHeatmapLosses(
                **self.results_df["results"][0].ann.gaussian_layer_loss_config
            )
            logits_layer = self.results_df["results"][0].ann.GaussianHeatmap
        except Exception:
            print("Trying to load ANN trainers...")
            try:
                self.results_df["results"][0].load_trainers(which="ann")
                loss_layer = GaussianHeatmapLosses(
                    **self.results_df["results"][0].ann.gaussian_layer_loss_config
                )
                logits_layer = self.results_df["results"][0].ann.GaussianHeatmap
            except Exception as e2:
                print(f"Could not get ANN loss layer: {e2}")
                raise

        for _, row in self.results_df.iterrows():
            nameExp = row["nameExp"]
            phase = row["phase"]
            winMS = row["winMS"]
            mouse_name = row["mouse_name"]
            mouse_results = row["results"]  # <-- the Mouse_Results object

            for suffix in suffixes:
                ws = str(winMS)
                pkl_path = os.path.join(
                    mouse_results.projectPath.experimentPath,
                    "results",
                    ws,
                    f"decoding_results{suffix}.pkl",
                )
                if not os.path.exists(pkl_path):
                    print(f"Missing {pkl_path}, skipping")
                    continue

                # --- Load only this file ---
                try:
                    with open(pkl_path, "rb") as f:
                        decoding_results = pickle.load(f)
                except Exception as e:
                    print(f"Failed to load {pkl_path}: {e}")
                    continue

                # --- Compute KL loss ---
                logits_hw = decoding_results["logits_hw"]
                target_hw = decoding_results["featureTrue"][:, :2]
                target_hw = logits_layer.gaussian_heatmap_targets(target_hw).numpy()
                inputs = {"logits": logits_hw, "targets": target_hw}
                kl_loss = (
                    loss_layer(inputs["targets"], inputs["logits"]).numpy().flatten()
                )

                entropy = decoding_results["Hn"].flatten()
                max_proba = decoding_results["maxp"].flatten()
                times = decoding_results["times"].flatten()

                # --- Plot correlations ---
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))

                # KL vs Entropy
                sc = axs[0, 0].scatter(entropy, kl_loss, c=times, cmap="viridis", s=5)
                axs[0, 0].set_xlabel("Entropy")
                axs[0, 0].set_ylabel("KL Loss")
                axs[0, 0].set_title("KL Loss vs Entropy")
                plt.colorbar(sc, ax=axs[0, 0], label="Time")
                if len(entropy) > 2:
                    p = np.polyfit(entropy, kl_loss, 2)
                    x_fit = np.linspace(entropy.min(), entropy.max(), 100)
                    axs[0, 0].plot(x_fit, np.polyval(p, x_fit), "r-", label="Poly2 fit")
                    axs[0, 0].legend()

                # KL vs Max Proba
                sc = axs[0, 1].scatter(max_proba, kl_loss, c=times, cmap="viridis", s=5)
                axs[0, 1].set_xlabel("Max Proba")
                axs[0, 1].set_ylabel("KL Loss")
                axs[0, 1].set_title("KL Loss vs Max Proba")
                plt.colorbar(sc, ax=axs[0, 1], label="Time")
                if len(max_proba) > 2:
                    p = np.polyfit(max_proba, kl_loss, 2)
                    x_fit = np.linspace(max_proba.min(), max_proba.max(), 100)
                    axs[0, 1].plot(x_fit, np.polyval(p, x_fit), "r-", label="Poly2 fit")
                    axs[0, 1].legend()

                # KL vs Time
                axs[1, 0].plot(times, kl_loss, "k.", markersize=3, alpha=0.5)
                axs[1, 0].set_xlabel("Time")
                axs[1, 0].set_ylabel("KL Loss")
                axs[1, 0].set_title("KL Loss over Time")

                # KL vs Entropy/MaxP ratio
                ratio = entropy / (max_proba + 1e-9)
                sc = axs[1, 1].scatter(ratio, kl_loss, c=times, cmap="viridis", s=5)
                axs[1, 1].set_xlabel("Entropy / MaxP")
                axs[1, 1].set_ylabel("KL Loss")
                axs[1, 1].set_title("KL Loss vs Entropy/MaxP")
                plt.colorbar(sc, ax=axs[1, 1], label="Time")

                fig.suptitle(
                    f"{nameExp} | Mouse: {mouse_name} | Phase: {phase} | winMS: {winMS} | {suffix}",
                    y=1.02,
                )
                fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])

                if save and folder is not None:
                    fname = (
                        f"klCorr_{nameExp}_{mouse_name}_phase{phase}_win{winMS}{suffix}"
                    )
                    fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                    fig.savefig(os.path.join(folder, fname + ".svg"))
                if show:
                    plt.show()
                plt.close(fig)

                # cleanup
                del decoding_results

    def pooled_correlation_entropy_maxp_vs_KL(
        self,
        suffixes=None,
        against="entropy",
        z_var_cmap="times",
        save=True,
        folder=None,
        show=False,
    ):
        """
        For each suffix:
          - Select rows with phase == f"_{suffix}"
          - Group by (nameExp, phase, winMS)
          - Load decoding_results.pkl for each row
          - Compute pooled correlations vs entropy/maxp
          - Plot + save one figure per suffix
        """

        folder = folder or getattr(self, "folderFigures", None)
        suffixes = suffixes or getattr(self, "suffixes", [""])

        # --- Try to get ANN loss layer once ---
        from neuroencoders.fullEncoder.nnUtils import GaussianHeatmapLosses

        try:
            loss_layer = GaussianHeatmapLosses(
                **self.results_df["results"][0].ann.gaussian_layer_loss_config
            )
            logits_layer = self.results_df["results"][0].ann.GaussianHeatmap
        except Exception:
            print("Trying to load ANN trainers...")
            self.results_df["results"][0].load_trainers(which="ann")
            loss_layer = GaussianHeatmapLosses(
                **self.results_df["results"][0].ann.gaussian_layer_loss_config
            )
            logits_layer = self.results_df["results"][0].ann.GaussianHeatmap

        # --- loop over suffixes ---
        for suffix in suffixes:
            suffix_tag = suffix.strip("_")
            print(f"\nProcessing suffix: {suffix} (tag: {suffix_tag})")
            grouped = self.results_df.query("phase == @suffix_tag").groupby(
                ["nameExp", "phase", "winMS"]
            )
            for (nameExp, phase, winMS), df in grouped:
                all_entropy, all_maxp, all_times, all_kl = [], [], [], []
                fast_entropy, fast_maxp, fast_times, fast_kl = [], [], [], []
                z_var, fast_z_var = [], []
                if phase != suffix_tag:
                    continue  # only keep rows for this suffix

                for _, row in df.iterrows():
                    mouse_results = row["results"]

                    ws = str(winMS)
                    pkl_path = os.path.join(
                        mouse_results.projectPath.experimentPath,
                        "results",
                        ws,
                        f"decoding_results{suffix}.pkl",
                    )
                    if not os.path.exists(pkl_path):
                        continue

                    try:
                        with open(pkl_path, "rb") as f:
                            decoding_results = pickle.load(f)
                    except Exception as e:
                        print(f"Failed to load {pkl_path}: {e}")
                        continue

                    # --- compute KL loss ---
                    logits_hw = decoding_results["logits_hw"]
                    target_hw = decoding_results["featureTrue"][:, :2]
                    target_hw = logits_layer.gaussian_heatmap_targets(target_hw).numpy()
                    inputs = {"logits": logits_hw, "targets": target_hw}
                    kl_loss = (
                        loss_layer(inputs["targets"], inputs["logits"])
                        .numpy()
                        .flatten()
                    )

                    entropy = decoding_results["Hn"].flatten()
                    max_proba = decoding_results["maxp"].flatten()
                    times = decoding_results["times"].flatten()

                    # append pooled
                    all_entropy.append(entropy)
                    all_maxp.append(max_proba)
                    all_times.append(times)
                    all_kl.append(kl_loss)
                    mask = row["speedMask"]
                    fast_entropy.append(entropy[mask])
                    fast_maxp.append(max_proba[mask])
                    fast_times.append(times[mask])
                    fast_kl.append(kl_loss[mask])
                    if z_var_cmap == "mouse":
                        z_var.append(
                            np.repeat(
                                f"{row['nameExp']}_{row['mouse_name']}",
                                len(entropy),
                            )
                        )
                        fast_z_var.append(
                            np.repeat(
                                f"{row['nameExp']}_{row['mouse_name']}",
                                np.sum(mask),
                            )
                        )
                    else:
                        pass  # z_var = all_times

                    # cleanup
                    del decoding_results

                # --- skip if no data ---
                if not all_kl:
                    print(
                        f"No valid data for suffix {suffix}, nameExp {nameExp}, phase {phase}, winMS {winMS}"
                    )
                    continue

                # --- concatenate pooled data ---
                all_entropy = np.concatenate(all_entropy).reshape(-1)
                all_maxp = np.concatenate(all_maxp).reshape(-1)
                all_times = np.concatenate(all_times).reshape(-1)
                all_kl = np.concatenate(all_kl).reshape(-1)
                fast_entropy = np.concatenate(fast_entropy).reshape(-1)
                fast_maxp = np.concatenate(fast_maxp).reshape(-1)
                fast_times = np.concatenate(fast_times).reshape(-1)
                fast_kl = np.concatenate(fast_kl).reshape(-1)

                length_entropy = len(all_entropy)
                to_plot = min(5000, length_entropy)
                ratio_to_plot = max(1, length_entropy // to_plot)
                print(f"Plotting {to_plot} points (1 every {ratio_to_plot})")

                length_entropy_fast = len(fast_entropy)
                to_plot = min(5000, length_entropy_fast)
                ratio_to_plot_fast = max(1, length_entropy_fast // to_plot)
                if z_var_cmap == "mouse":
                    z_var = np.concatenate(z_var).reshape(-1)
                    # create a color map for each mouse (unique value in z_var)
                    unique_mice = np.unique(z_var)
                    colors = plt.cm.get_cmap("tab20", len(unique_mice))
                    color_dict = {m: colors(i) for i, m in enumerate(unique_mice)}
                    z_var = np.array([color_dict[m] for m in z_var])
                    scatter_kwargs = {"c": z_var[::ratio_to_plot], "s": 5}
                    # same for fast
                    fast_z_var = np.concatenate(fast_z_var).reshape(-1)
                    fast_z_var = np.array([color_dict[m] for m in fast_z_var])
                    fast_scatter_kwargs = {
                        "c": fast_z_var[::ratio_to_plot_fast],
                        "s": 5,
                    }
                else:
                    z_var = all_times
                    fast_z_var = fast_times
                    scatter_kwargs = {
                        "c": z_var[::ratio_to_plot],
                        "cmap": "viridis",
                        "s": 5,
                    }
                    fast_scatter_kwargs = {
                        "c": fast_z_var[::ratio_to_plot_fast],
                        "cmap": "viridis",
                        "s": 5,
                    }

                ratio = all_entropy / (all_maxp + 1e-9)

                # --- correlations ---
                print(f"\n===== Correlations for suffix {suffix} =====")
                for name, x in {
                    "Entropy": all_entropy,
                    "Max Proba": all_maxp,
                    "Entropy/MaxP": ratio,
                }.items():
                    pear_r, pear_p = pearsonr(x, all_kl)
                    spear_r, spear_p = spearmanr(x, all_kl)
                    print(
                        f"{name:12s} vs KL Loss : "
                        f"Pearson r={pear_r:.3f} (p={pear_p:.1e}), "
                        f"Spearman r={spear_r:.3f} (p={spear_p:.1e})"
                    )

                # --- make figure ---
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))

                # KL vs Entropy
                if against == "entropy":
                    var_to_show = all_entropy
                    var_to_show_fast = fast_entropy
                elif against == "maxp":
                    var_to_show = all_maxp
                    var_to_show_fast = fast_maxp
                else:
                    raise ValueError("against must be 'entropy' or 'maxp'")

                sc = axs[0, 0].scatter(
                    var_to_show[::ratio_to_plot],
                    all_kl[::ratio_to_plot],
                    **scatter_kwargs,
                )

                axs[0, 0].set_xlabel(against.capitalize())
                axs[0, 0].set_ylabel("KL Loss")
                axs[0, 0].set_title(f"KL Loss vs {against.capitalize()}")
                plt.colorbar(sc, ax=axs[0, 0], label=z_var_cmap.capitalize())

                if len(var_to_show) > 2:
                    p = np.polyfit(var_to_show, all_kl, 2)
                    x_fit = np.linspace(var_to_show.min(), var_to_show.max(), 200)
                    axs[0, 0].plot(x_fit, np.polyval(p, x_fit), "r-", label="Poly2 fit")
                    axs[0, 0].legend()

                # Same but for fast
                sc = axs[0, 1].scatter(
                    var_to_show_fast[::ratio_to_plot_fast],
                    fast_kl[::ratio_to_plot_fast],
                    **fast_scatter_kwargs,
                )
                axs[0, 1].set_xlabel(f"Fast {against.capitalize()}")
                axs[0, 1].set_ylabel("KL Loss")
                axs[0, 1].set_title(f"Fast Epochs - KL Loss vs {against.capitalize()}")
                plt.colorbar(sc, ax=axs[0, 1], label=z_var_cmap.capitalize())

                if len(var_to_show_fast) > 2:
                    p = np.polyfit(var_to_show_fast, fast_kl, 2)
                    x_fit = np.linspace(
                        var_to_show_fast.min(), var_to_show_fast.max(), 200
                    )
                    axs[0, 1].plot(x_fit, np.polyval(p, x_fit), "r-", label="Poly2 fit")
                    axs[0, 1].legend()

                # KL vs Time
                axs[1, 0].plot(
                    all_times[::ratio_to_plot],
                    all_kl[::ratio_to_plot],
                    "k.",
                    markersize=2,
                    alpha=0.5,
                )
                axs[1, 0].set_xlabel("Time")
                axs[1, 0].set_ylabel("KL Loss")
                axs[1, 0].set_title("KL Loss over Time")

                # KL vs Entropy/MaxP ratio
                sc = axs[1, 1].scatter(ratio, all_kl, c=all_times, cmap="viridis", s=5)
                axs[1, 1].set_xlabel("Entropy / MaxP")
                axs[1, 1].set_ylabel("KL Loss")
                axs[1, 1].set_title("KL Loss vs Entropy/MaxP")
                plt.colorbar(sc, ax=axs[1, 1], label="Time")

                fig.suptitle(f"{nameExp} | Phase: {phase} | winMS: {winMS}")
                fig.tight_layout()

                if save and folder is not None:
                    fname = f"klCorr_pooled_{nameExp}_phase{phase}_{winMS}_against_{against}_by_{z_var_cmap}"
                    fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                    fig.savefig(os.path.join(folder, fname + ".svg"))
                if show:
                    plt.show()
                plt.close(fig)

    def plot_ann_vs_bayes_linerror(
        self,
        bayes_nameExp="new_4d_GaussianHeatMap_LinearLoss_Transformer",
        error_type="selected",  # "selected" or "full"
        speed="all",  # "all", "fast", "slow"
        save=True,
        folder=None,
        show=False,
    ):
        """
        Plot ANN vs Bayesian linear errors across phases, grouped by (nameExp, winMS).
        One figure is generated per (nameExp, phase).
        Bayes results are always shown from the reference bayes_nameExp.
        Outliers are labeled with mouse IDs.

        Args:
            bayes_nameExp (str): The nameExp where Bayes results are stored.
            error_type (str): "selected" (default) or "full" to pick error metric.
            speed (str): "all" (default), "fast" (speedMask == True), "slow" (speedMask == False).
            save (bool): Save figures.
            folder (str): Output folder.
            show (bool): Show figures interactively.
        """

        df = self.results_df.copy()
        folder = folder or getattr(self, "folderFigures", None)

        # --- compute per-row mean errors from full arrays ---
        ann_errors, bayes_errors = [], []
        for _, row in df.iterrows():
            # pick which arrays to use
            ann_arr = row["lin_error"]
            bayes_arr = row["lin_error_bayes"]

            # apply speed filter if needed
            if speed != "all" and "speedMask" in row and row["speedMask"] is not None:
                mask = row["speedMask"].astype(bool)
                if speed == "fast":
                    speed_mask = mask
                elif speed == "slow":
                    speed_mask = ~mask
            else:
                speed_mask = np.ones_like(ann_arr, dtype=bool)

            if error_type == "selected":
                thresh_mask_ann = row["predLoss"] <= row["predLossThreshold"]
                thresh_mask_bayes = row["bayesProba"] >= row["bayesProbThreshold"]
            else:
                thresh_mask_ann = np.ones_like(ann_arr, dtype=bool)
                thresh_mask_bayes = np.ones_like(bayes_arr, dtype=bool)

            ann_arr = (
                ann_arr[thresh_mask_ann & speed_mask] if ann_arr is not None else None
            )
            bayes_arr = (
                bayes_arr[thresh_mask_bayes & speed_mask]
                if bayes_arr is not None
                else None
            )

            # compute means
            ann_mean = (
                np.nan if ann_arr is None or len(ann_arr) == 0 else np.nanmean(ann_arr)
            )
            bayes_mean = (
                np.nan
                if bayes_arr is None or len(bayes_arr) == 0
                else np.nanmean(bayes_arr)
            )

            ann_errors.append(ann_mean)
            bayes_errors.append(bayes_mean)

        df[f"{error_type}_error_ann"] = ann_errors
        df[f"{error_type}_error_bayes"] = bayes_errors

        # --- ensure Bayes results come only from bayes_nameExp ---
        df["use_bayes"] = df["nameExp"] == bayes_nameExp
        df.loc[~df["use_bayes"], f"{error_type}_error_bayes"] = np.nan
        # --- and then apply bayes results back to all nameExps --
        # --- extract Bayes gold standard ---
        bayes_df = df[df["nameExp"] == bayes_nameExp].copy()
        bayes_df = bayes_df[["mouse", "phase", "winMS", f"{error_type}_error_bayes"]]

        # rename to something neutral
        bayes_df = bayes_df.rename(columns={f"{error_type}_error_bayes": "bayes_gold"})

        # --- broadcast back to all rows ---
        df = df.merge(bayes_df, on=["mouse", "phase", "winMS"], how="left")

        # overwrite with gold everywhere
        df[f"{error_type}_error_bayes"] = df["bayes_gold"]
        df = df.drop(columns=["bayes_gold"])

        # --- loop over nameExp ---
        for nameExp in df["nameExp"].unique():
            df_exp = df[df["nameExp"] == nameExp].copy()

            for phase in df_exp["phase"].unique():
                df_phase = df_exp[df_exp["phase"] == phase].copy()
                if df_phase.empty:
                    continue

                order = sorted(df_phase["winMS"].unique(), key=lambda x: float(x))

                fig, ax = plt.subplots(figsize=(8, 10))

                # Melt for combined plotting
                long_df = pd.melt(
                    df_phase,
                    id_vars=["mouse", "winMS", "nameExp", "use_bayes"],
                    value_vars=[f"{error_type}_error_ann", f"{error_type}_error_bayes"],
                    var_name="Decoder",
                    value_name="Error",
                )
                long_df = long_df.dropna(subset=["Error"])  # Drop Bayes NaNs

                palette = {
                    f"{error_type}_error_ann": "#427590",
                    f"{error_type}_error_bayes": "#cccccc",
                }

                # Boxplots
                sns.boxplot(
                    data=long_df,
                    x="winMS",
                    y="Error",
                    hue="Decoder",
                    ax=ax,
                    order=order,
                    palette=palette,
                    showfliers=False,
                )
                # Stripplots
                sns.stripplot(
                    data=long_df,
                    x="winMS",
                    y="Error",
                    hue="Decoder",
                    dodge=True,
                    ax=ax,
                    order=order,
                    palette=palette,
                    size=7,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Remove duplicate legends
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:2], ["ANN", "Bayes"], fontsize=12, loc="best")

                # --- Statistical annotation ANN vs Bayes ---
                pairs = [
                    (
                        (win, f"{error_type}_error_ann"),
                        (win, f"{error_type}_error_bayes"),
                    )
                    for win in order
                    if (long_df["winMS"] == win).any()
                ]
                annotator = Annotator(
                    ax,
                    pairs,
                    data=long_df,
                    x="winMS",
                    y="Error",
                    hue="Decoder",
                    order=order,
                )
                annotator.configure(
                    test="t-test_paired", text_format="star", loc="inside"
                )
                annotator.apply_and_annotate()

                # --- Outlier labeling ---
                for decoder_type, metric in zip(
                    ["ANN", "Bayes"],
                    [f"{error_type}_error_ann", f"{error_type}_error_bayes"],
                ):
                    color = "#427590" if decoder_type == "ANN" else "#cccccc"
                    df_metric = df_phase[["winMS", "mouse", metric]].dropna()
                    for winMS in order:
                        vals = df_metric[df_metric["winMS"] == winMS][metric].dropna()
                        if vals.empty:
                            continue
                        fliers = [
                            y for stat in boxplot_stats(vals) for y in stat["fliers"]
                        ]
                        for outlier in fliers:
                            outlier_rows = df_metric[
                                (df_metric["winMS"] == winMS)
                                & (df_metric[metric] == outlier)
                            ]
                            for _, row in outlier_rows.iterrows():
                                x = order.index(winMS)
                                sign = +1 if decoder_type == "ANN" else -1
                                ax.annotate(
                                    row["mouse"],
                                    xy=(x, row[metric]),
                                    xytext=(2 * sign, 2 * sign),
                                    textcoords="offset points",
                                    fontsize=10,
                                    color=color,
                                )

                # --- Chance levels ---
                if hasattr(self, "chance_level"):
                    for i, winMS in enumerate(order):
                        if str(winMS) in self.chance_level:
                            ax.plot(
                                [i - 0.2, i + 0.2],
                                [self.chance_level[str(winMS)]] * 2,
                                color="black",
                                linestyle="--",
                                linewidth=3,
                                label="Chance level" if i == 0 else "",
                            )

                # --- Formatting ---
                ax.set_title(
                    f"ANN vs Bayes ({error_type} error, {speed} speed) | "
                    f"Phase: {phase} | nameExp: {nameExp} | Bayes: {bayes_nameExp}"
                )
                ax.set_xlabel("Window size (ms)", fontsize=16)
                ax.set_ylabel("Linear Error (u.a.)", fontsize=16)
                ax.tick_params(axis="both", labelsize=14)

                fig.tight_layout()

                if save and folder is not None:
                    fname = f"ann_vs_bayes_linerror_{nameExp}_phase{phase}_{error_type}_{speed}"
                    fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                    fig.savefig(os.path.join(folder, fname + ".svg"))
                if show:
                    plt.show()
                plt.close(fig)

    def around_ripples_METAverage(
        self,
        suffixes=None,
        nameExp=None,
        against="entropy",
        around=0.5,  # seconds before and after ripple
        dt=0.01,  # time bin resolution
        smooth_window=5,  # moving average window in bins
        save=True,
        folder=None,
        show=False,
    ):
        """
        Compute METAverage around ripples, then z-score and optionally smooth the mean and std.

        Z-scoring is done **after** averaging across mice and ripples.
        """

        folder = folder or getattr(self, "folderFigures", None)
        suffixes = suffixes or getattr(self, "suffixes", [""])
        nameExps = nameExp or self.results_df["nameExp"].unique()

        for suffix in suffixes:
            suffix_tag = suffix.strip("_")
            print(f"\nProcessing suffix: {suffix} (tag: {suffix_tag})")
            grouped = self.results_df.query("phase == @suffix_tag").groupby(
                ["nameExp", "phase", "winMS"]
            )

            for (nameExp, phase, winMS), df in grouped:
                if nameExp not in nameExps:
                    print(f"Skipping nameExp {nameExp}")
                    continue
                if df.empty:
                    continue

                time_vec = np.arange(-around, around + dt, dt)
                all_peri_values = []

                # --- collect all peri-ripple traces ---
                for _, row in df.iterrows():
                    mouse_results = row["results"]

                    ws = str(winMS)
                    pkl_path = os.path.join(
                        mouse_results.projectPath.experimentPath,
                        "results",
                        ws,
                        f"decoding_results{suffix}.pkl",
                    )
                    if not os.path.exists(pkl_path):
                        continue

                    try:
                        with open(pkl_path, "rb") as f:
                            decoding_results = pickle.load(f)
                    except Exception as e:
                        print(f"Failed to load {pkl_path}: {e}")
                        continue

                    if against == "entropy":
                        values = decoding_results["Hn"].flatten()
                    elif against == "maxp":
                        values = decoding_results["maxp"].flatten()
                    else:
                        raise ValueError("against must be 'entropy' or 'maxp'")

                    times = decoding_results["times"].flatten()
                    tRipples = mouse_results.data_helper.fullBehavior["Times"].get(
                        "tRipples", None
                    )
                    if tRipples is None or len(tRipples) == 0:
                        continue

                    for tr in tRipples:
                        mask = (times >= tr - around) & (times <= tr + around)
                        if mask.any():
                            peri_times = times[mask] - tr
                            interp_values = np.interp(
                                time_vec, peri_times, values[mask]
                            )
                            all_peri_values.append(interp_values)

                    del decoding_results

                if len(all_peri_values) == 0:
                    print(
                        f"No valid ripple data for {nameExp}, phase {phase}, winMS {winMS}"
                    )
                    continue

                all_peri_values = np.vstack(
                    all_peri_values
                )  # shape: ripples × time bins

                # --- METAverage across ripples ---
                mean_trace = np.mean(all_peri_values, axis=0)
                std_trace = np.std(all_peri_values, axis=0)

                # --- z-score **after averaging** ---
                mean_trace_z = (mean_trace - np.mean(mean_trace)) / np.std(mean_trace)
                std_trace_z = std_trace / np.std(mean_trace)  # normalized std

                # --- optional smoothing ---
                if smooth_window > 1:
                    from scipy.ndimage import uniform_filter1d

                    mean_trace_z = uniform_filter1d(mean_trace_z, size=smooth_window)
                    std_trace_z = uniform_filter1d(std_trace_z, size=smooth_window)

                # --- plot ---
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(time_vec, mean_trace_z, color="blue", lw=2)
                ax.fill_between(
                    time_vec,
                    mean_trace_z - std_trace_z,
                    mean_trace_z + std_trace_z,
                    color="blue",
                    alpha=0.3,
                )
                ax.axvline(0, color="black", linestyle="--", lw=1)
                ax.set_xlabel("Time around ripple (s)")
                ax.set_ylabel(f"Z-scored {against} (METAverage)")
                ax.set_title(f"{nameExp} | phase: {phase} | winMS: {winMS}")
                fig.tight_layout()

                if save and folder is not None:
                    fname = f"real_METAverage_{suffix_tag}_{nameExp}_win{winMS}_{against}_zscored"
                    fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                    fig.savefig(os.path.join(folder, fname + ".svg"))
                if show:
                    plt.show()
                plt.close(fig)

    def correlation_global_predictions(
        self,
        against="entropy",
        error_type="lin",
        mode="full",
        speed="all",
        suffixes=None,
        nameExps=None,
        winMS_list=None,
        save=True,
        folder=None,
        show=False,
        zscore=False,
        max_points=50000,
    ):
        """
        Global point-by-point correlation.
        If data exceeds max_points, it subsamples to keep plotting fast and meaningful.
        """
        from scipy.stats import linregress

        folder = folder or getattr(self, "folderFigures", None)
        df = self.results_df.copy()

        # --- apply filters ---
        if suffixes:
            df = df[df["phase"].isin([s.strip("_") for s in suffixes])]
        if nameExps:
            df = df[df["nameExp"].isin(nameExps)]
        if winMS_list:
            df = df[df["winMS"].isin(winMS_list)]

        col_error = "lin_error" if error_type == "lin" else "error"
        all_data: List[pd.DataFrame] = []

        for _, row in df.iterrows():
            # 1. Load Decoding Data
            ws = str(row["winMS"])
            pkl_path = os.path.join(
                row["results"].projectPath.experimentPath,
                "results",
                ws,
                f"decoding_results_{row['phase']}.pkl",
            )
            if not os.path.exists(pkl_path):
                continue
            try:
                with open(pkl_path, "rb") as f:
                    res = pickle.load(f)
            except FileNotFoundError:
                continue

            # 2. Extract arrays
            x_full = (
                res["Hn"].flatten() if against == "entropy" else res["maxp"].flatten()
            )
            y_full = row[col_error]

            # 3. Apply Masking
            mask = np.ones_like(y_full, dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask &= row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]

            x_masked, y_masked = x_full[mask], y_full[mask]

            if zscore and len(x_masked) > 0:
                x_masked = (x_masked - np.nanmean(x_masked)) / (
                    np.nanstd(x_masked) + 1e-9
                )

            if len(x_masked) > 0:
                all_data.append(
                    pd.DataFrame(
                        {
                            "x": x_masked,
                            "y": y_masked,
                            "phase": row["phase"],
                            "mouse": row["mouse_name"],
                        }
                    )
                )

        if not all_data:
            print("No data found.")
            return

        full_df = pd.concat(all_data, ignore_index=True).dropna(subset=["x", "y"])

        total_count = full_df.shape[0]

        # --- Smart Subsampling ---
        if total_count > max_points:
            print(
                f"Downsampling for visualization: {total_count} -> {max_points} points."
            )
            plot_df = full_df.sample(n=max_points, random_state=42)
        else:
            plot_df = full_df

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue="phase",
            alpha=0.2,
            s=5,
            edgecolor=None,
            ax=ax,
            rasterized=True,
        )

        # --- Global Stats (always on the FULL dataset, not just the subset) ---
        slope, intercept, r_val, p_val, _ = linregress(full_df["x"], full_df["y"])
        line_x = np.array([full_df["x"].min(), full_df["x"].max()])
        ax.plot(
            line_x,
            intercept + slope * line_x,
            color="black",
            lw=2.5,
            ls="--",
            label=f"Global R={r_val:.3f}\np={p_val:.2e}\nN={total_count}",
        )

        ax.set_title(
            f"Global Correlation: {against} vs {error_type}\n({mode} mode, {speed} speed)"
        )
        ax.set_xlabel(against)
        ax.set_ylabel(f"Error ({error_type})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        if save and folder:
            fig.savefig(
                os.path.join(folder, f"global_pointwise_{against}.png"), dpi=200
            )
        if show:
            plt.show()

        plt.close(fig)

    def correlation_global_spikes(
        self,
        against="entropy",  # "entropy", "maxp", or "error"
        error_type="lin",  # "lin" for linear error
        mode="all",  # "selected" or "full"
        speed="all",  # "fast", "slow", "all"
        suffixes=None,  # list of suffixes to select (phase)
        nameExps=None,  # list of nameExp to include
        winMS_list=None,  # list of winMS to include
        save=True,
        folder=None,
        show=False,
        zscore=False,
        max_points=50000,  # Max points to plot (meaningful subsampling)
    ):
        """
        Correlate decoder values (entropy/maxp/error) with spike counts globally.
        Every point is one single prediction time-bin across all sessions.
        """
        from scipy.stats import linregress

        folder = folder or getattr(self, "folderFigures", None)
        df = self.results_df.copy()

        # --- apply filters ---
        if suffixes is not None:
            suffix_tags = [s.strip("_") for s in suffixes]
            df = df[df["phase"].isin(suffix_tags)]
        if nameExps is not None:
            df = df[df["nameExp"].isin(nameExps)]
        if winMS_list is not None:
            df = df[df["winMS"].isin(winMS_list)]

        col_error = "lin_error" if error_type == "lin" else "error"
        all_sessions_data: List[pd.DataFrame] = []

        for _, row in df.iterrows():
            mouse_results = row["results"]
            ws = str(row["winMS"])
            suffix = f"_{row['phase']}"
            pkl_path = os.path.join(
                mouse_results.projectPath.experimentPath,
                "results",
                ws,
                f"decoding_results{suffix}.pkl",
            )
            if not os.path.exists(pkl_path):
                continue

            try:
                with open(pkl_path, "rb") as f:
                    decoding_results = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {pkl_path}: {e}")
                continue

            # --- apply masks ---
            mask = np.ones_like(row[col_error], dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask &= row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]

            # 1. Extract Spikes
            clusters_time_file = os.path.join(
                mouse_results.folderResult, "clusters_time_pre_wTrain_False.pkl"
            )
            try:
                try:
                    with open(clusters_time_file, "rb") as f:
                        clusters_time = pickle.load(f)
                except FileNotFoundError:
                    clusters_time_file = os.path.abspath(
                        os.path.join(
                            mouse_results.folderResult,
                            "..",
                            "..",
                            "last_bayes",
                            "results",
                            f"clusters_time_pre_wTrain_{'True' if row['phase'] == 'training' else 'False'}.pkl",
                        )
                    )
                    with open(clusters_time_file, "rb") as f:
                        clusters_time = pickle.load(f)
            except Exception as e:
                print(f"Failed to load spikes for {row['mouse_name']}: {e}")
                continue

            times = decoding_results["times"].flatten()
            spikes_count = np.zeros_like(times, dtype=float)
            for cl_time in clusters_time:
                spikes_count += np.histogram(
                    cl_time,
                    bins=np.append(times, times[-1] + np.median(np.diff(times))),
                )[0]

            # 2. Extract 'Against' variable
            if against in ["entropy", "maxp"]:
                val_array = (
                    decoding_results["Hn"].flatten()
                    if against == "entropy"
                    else decoding_results["maxp"].flatten()
                )
            elif against == "error":
                val_array = row[col_error]

            # 3. Apply mask and collect raw points
            x_raw = val_array[mask]
            y_raw = spikes_count[mask]

            if zscore and len(x_raw) > 0:
                x_raw = (x_raw - np.nanmean(x_raw)) / (np.nanstd(x_raw) + 1e-9)
                y_raw = (y_raw - np.nanmean(y_raw)) / (np.nanstd(y_raw) + 1e-9)

            if len(x_raw) > 0:
                all_sessions_data.append(
                    pd.DataFrame(
                        {
                            "x": x_raw,
                            "y": y_raw,
                            "phase": row["phase"],
                            "nameExp": row["nameExp"],
                        }
                    )
                )

            del decoding_results

        if not all_sessions_data:
            print("No data available for this selection.")
            return

        full_df = pd.concat(all_sessions_data, ignore_index=True).dropna()

        total_points = len(full_df)

        # --- Smart Subsampling for Visualization ---
        if total_points > max_points:
            print(f"Plotting {max_points} / {total_points} points for clarity.")
            plot_df = full_df.sample(n=max_points, random_state=42)
        else:
            plot_df = full_df

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 7))

        # rasterized=True keeps the SVG file size small by rendering points as a bitmap
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue="phase",
            style="nameExp",
            s=10,
            alpha=0.3,
            ax=ax,
            palette="tab10",
            rasterized=True,
            edgecolor=None,
        )

        # --- Global Regression (computed on ALL data, not just sampled) ---
        slope, intercept, r_val, p_val, _ = linregress(full_df["x"], full_df["y"])
        x_range = np.array([full_df["x"].min(), full_df["x"].max()])
        ax.plot(x_range, intercept + slope * x_range, color="black", lw=2, ls="--")

        ax.set_xlabel(f"{against}{' (z-scored)' if zscore else ''}")
        ax.set_ylabel("Spike Count (per bin)")
        ax.set_title(
            f"Global Correlation: {against} vs Spikes\n(N={total_points} bins, R={r_val:.3f}, p={p_val:.2e})"
        )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        if save and folder is not None:
            fname = f"global_corr_{against}_vs_spikes_{mode}_{speed}"
            fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
            fig.savefig(os.path.join(folder, fname + ".svg"))

        if show:
            plt.show()
        plt.close(fig)

    def barplot_correlation_spikes(
        self,
        against="entropy",  # "entropy", "maxp", or "error"
        error_type="lin",  # "lin" for linear error
        mode="full",  # "selected" or "full"
        speed="all",  # "fast", "slow", "all"
        suffixes=None,  # list of suffixes/phases
        nameExps=None,  # list of nameExps
        winMS_list=None,  # list of winMS
        hue="winMS",
        save=True,
        folder=None,
        show=False,
        zscore=False,
    ):
        """
        Compute a global correlation between decoder variables and spikes.
        Instead of averaging R-values per mouse, it pools all time-bins for each
        category (Phase/WinMS) to get a true point-by-point global correlation.
        """
        from scipy.stats import spearmanr

        folder = folder or getattr(self, "folderFigures", None)
        df = self.results_df.copy()

        # --- apply filters ---
        if suffixes is not None:
            suffix_tags = [s.strip("_") for s in suffixes]
            df = df[df["phase"].isin(suffix_tags)]
        if nameExps is not None:
            df = df[df["nameExp"].isin(nameExps)]
        if winMS_list is not None:
            df = df[df["winMS"].isin(winMS_list)]

        col_error = "lin_error" if error_type == "lin" else "error"

        # Dictionary to pool raw data points: key is (phase, hue_val)
        pooled_data = {}

        for _, row in df.iterrows():
            mouse_results = row["results"]
            ws = str(row["winMS"])
            suffix = f"_{row['phase']}"
            pkl_path = os.path.join(
                mouse_results.projectPath.experimentPath,
                "results",
                ws,
                f"decoding_results{suffix}.pkl",
            )
            if not os.path.exists(pkl_path):
                continue

            try:
                with open(pkl_path, "rb") as f:
                    decoding_results = pickle.load(f)
            except Exception:
                continue

            # --- apply masks ---
            mask = np.ones_like(row[col_error], dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask &= row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]

            # --- extract variables ---
            if against in ["entropy", "maxp"]:
                val_array = (
                    decoding_results["Hn"].flatten()
                    if against == "entropy"
                    else decoding_results["maxp"].flatten()
                )
            elif against == "error":
                val_array = row[col_error]

            val_array = val_array[mask]

            # --- load spikes ---
            clusters_time_file = os.path.join(
                mouse_results.folderResult, "clusters_time_pre_wTrain_False.pkl"
            )
            if not os.path.exists(clusters_time_file):
                # Fallback path logic
                clusters_time_file = os.path.abspath(
                    os.path.join(
                        mouse_results.folderResult,
                        "..",
                        "..",
                        "last_bayes",
                        "results",
                        f"clusters_time_pre_wTrain_{'True' if row['phase'] == 'training' else 'False'}.pkl",
                    )
                )

            try:
                with open(clusters_time_file, "rb") as f:
                    clusters_time = pickle.load(f)
            except Exception:
                continue

            times = decoding_results["times"].flatten()
            spikes_count = np.zeros_like(times, dtype=float)
            for cl_time in clusters_time:
                spikes_count += np.histogram(
                    cl_time,
                    bins=np.append(times, times[-1] + np.median(np.diff(times))),
                )[0]
            spikes_count = spikes_count[mask]

            if len(val_array) == 0 or len(spikes_count) == 0:
                continue

            if zscore:
                val_array = (val_array - np.nanmean(val_array)) / (
                    np.nanstd(val_array) + 1e-9
                )
                spikes_count = (spikes_count - np.nanmean(spikes_count)) / (
                    np.nanstd(spikes_count) + 1e-9
                )

            # --- Pooling ---
            group_key = (row["phase"], row[hue])
            if group_key not in pooled_data:
                pooled_data[group_key] = {"x": [], "y": []}

            pooled_data[group_key]["x"].extend(val_array)
            pooled_data[group_key]["y"].extend(spikes_count)

        # --- Compute Correlation on Pooled Data ---
        final_corrs = []
        for (phase, h_val), data in pooled_data.items():
            r, p = spearmanr(data["x"], data["y"])
            final_corrs.append(
                {
                    "phase": phase,
                    hue: h_val,
                    "correlation": r,
                    "p_value": p,
                    "n_points": len(data["x"]),
                }
            )

        if not final_corrs:
            print("No correlations computed.")
            return

        corr_df = pd.DataFrame(final_corrs)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=corr_df,
            x="phase",
            y="correlation",
            hue=hue,
            ax=ax,
            palette="tab10",
            order=sorted(corr_df["phase"].unique()),
        )

        ax.set_ylabel(f"Global Spearman R ({against} vs Spikes)")
        ax.set_xlabel("Phase")
        ax.set_title(
            f"Global Point-by-Point Correlation\n(Total bins pooled per {hue})"
        )

        # Add N labels on top of bars
        for i, p in enumerate(ax.patches):
            if p.get_height() != 0:
                ax.annotate(
                    f"n={corr_df.iloc[i]['n_points']:.0e}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="baseline",
                    fontsize=8,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

        fig.tight_layout()

        if save and folder:
            fname = f"global_barplot_{against}_vs_spikes"
            fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def correlation_ann_vs_bayes(
        self,
        ann_var="maxp",  # "maxp", "entropy", "lin_error", etc.
        bayes_var="bayesPred",  # "bayesPred" or "bayesProba"
        mode="selected",  # "selected" or "full"
        bayes_nameExp="new_4d_GaussianHeatMap_LinearLoss_Transformer",
        speed="all",  # "fast", "slow", "all"
        suffixes=None,  # list of phases to include
        nameExps=None,  # list of nameExps
        winMS_list=None,  # list of winMS
        save=True,
        folder=None,
        show=False,
        zscore=False,  # whether to z-score values before correlation
    ):
        """
        Compute correlation between ANN metric (maxp/entropy/error) and Bayesian predictions/probabilities.
        One point per (mouse, nameExp, phase, winMS).

        Args:
            ann_var (str): ANN variable to correlate ("maxp", "entropy", "lin_error", etc.).
            bayes_var (str): Bayesian variable ("bayesPred" or "bayesProba").
            mode (str): "selected" or "full" for error columns.
            speed (str): "fast", "slow", "all" for filtering speed.
            suffixes (list): which phases/suffixes to include.
            nameExps (list): which nameExps to include.
            winMS_list (list): which winMS to include.
            save (bool): save figure.
            folder (str): folder to save figure.
            show (bool): show figure interactively.
            zscore (bool): z-score values before correlation.
        """
        from scipy.stats import spearmanr

        folder = folder or getattr(self, "folderFigures", None)
        df = self.results_df.copy()

        # --- ensure Bayes results come only from bayes_nameExp ---
        df["use_bayes"] = df["nameExp"] == bayes_nameExp
        df.loc[~df["use_bayes"], bayes_var] = np.nan
        # --- and then apply bayes results back to all nameExps --
        # --- extract Bayes gold standard ---
        bayes_df = df[df["nameExp"] == bayes_nameExp].copy()
        bayes_df = bayes_df[["mouse", "phase", "winMS", bayes_var]]

        # rename to something neutral
        bayes_df = bayes_df.rename(columns={bayes_var: "bayes_gold"})

        # --- broadcast back to all rows ---
        df = df.merge(bayes_df, on=["mouse", "phase", "winMS"], how="left")

        # overwrite with gold everywhere
        df[bayes_var] = df["bayes_gold"]
        df = df.drop(columns=["bayes_gold"])

        # --- apply filters ---
        if suffixes is not None:
            suffix_tags = [s.strip("_") for s in suffixes]
            df = df[df["phase"].isin(suffix_tags)]
        if nameExps is not None:
            df = df[df["nameExp"].isin(nameExps)]
        if winMS_list is not None:
            df = df[df["winMS"].isin(winMS_list)]

        correlations = []

        for _, row in df.iterrows():
            mouse_results = row["results"]
            ws = str(row["winMS"])
            suffix = f"_{row['phase']}"

            # --- load ANN decoding results ---
            pkl_path = os.path.join(
                mouse_results.projectPath.experimentPath,
                "results",
                ws,
                f"decoding_results{suffix}.pkl",
            )
            if not os.path.exists(pkl_path):
                continue
            try:
                with open(pkl_path, "rb") as f:
                    decoding_results = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {pkl_path}: {e}")
                continue

            # --- extract ANN variable ---
            if ann_var == "entropy":
                ann_vals = decoding_results["Hn"].flatten()
            elif ann_var == "maxp":
                ann_vals = decoding_results["maxp"].flatten()
            elif ann_var in ["lin_error", "predLoss", "linearPred"]:
                col = ann_var

                if isinstance(row[col], np.ndarray):
                    ann_vals = row[col]
                else:
                    ann_vals = np.array([row[col]])
            else:
                raise ValueError("Unknown ann_var")

            # --- apply speed mask ---
            mask = np.ones_like(ann_vals, dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]
            ann_vals = ann_vals[mask]

            # --- extract Bayesian variable ---
            if bayes_var not in row or row[bayes_var] is None:
                continue
            bayes_vals = row[bayes_var]

            # --- apply speed mask ---
            mask = np.ones_like(bayes_vals, dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["bayesProba"] >= row["bayesProbaThreshold"]
            bayes_vals = bayes_vals[mask]

            if isinstance(bayes_vals, np.ndarray):
                bayes_vals = bayes_vals[mask]
            else:
                bayes_vals = np.array([bayes_vals])

            if np.isnan(bayes_vals).all():
                print(
                    f"All NaN bayes_vals for {row['mouse_name']} {row['nameExp']} {row['phase']} {row['winMS']}"
                )
                continue

            # --- apply speed mask ---
            mask = np.ones_like(bayes_vals, dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["bayesProba"] >= row["bayesProbaThreshold"]
            bayes_vals = bayes_vals[mask]

            # --- optional z-score ---
            if zscore:
                ann_vals = (ann_vals - np.nanmean(ann_vals)) / np.nanstd(ann_vals)
                bayes_vals = (bayes_vals - np.nanmean(bayes_vals)) / np.nanstd(
                    bayes_vals
                )

            # --- compute correlation per mouse × nameExp × winMS × phase ---
            if len(ann_vals) == 0 or len(bayes_vals) == 0:
                continue
            r, _ = spearmanr(ann_vals, bayes_vals)
            correlations.append(
                {
                    "mouse": row["mouse_name"],
                    "phase": row["phase"],
                    "winMS": row["winMS"],
                    "nameExp": row["nameExp"],
                    "correlation": r,
                }
            )

            del decoding_results

        if len(correlations) == 0:
            print("No correlations computed.")
            return

        corr_df = pd.DataFrame(correlations)

        # --- barplot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        phase_order = sorted(corr_df["phase"].unique())
        sns.barplot(
            data=corr_df,
            x="phase",
            y="correlation",
            hue="nameExp",
            ci="sd",
            ax=ax,
            palette="tab10",
            order=phase_order,
        )
        sns.stripplot(
            data=corr_df,
            x="phase",
            y="correlation",
            hue="nameExp",
            dodge=True,
            ax=ax,
            palette="tab10",
            size=7,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.set_ylabel(f"Spearman correlation ({ann_var} vs {bayes_var})")
        ax.set_xlabel("Phase")
        ax.set_title("ANN vs Bayesian correlation per mouse")
        ax.legend(loc="best")
        fig.tight_layout()

        if save and folder is not None:
            fname = f"correlation_{ann_var}_vs_{bayes_var}"
            fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
            fig.savefig(os.path.join(folder, fname + ".svg"))
        if show:
            plt.show()
        plt.close(fig)

    def hist2d_linpred_vs_bayes(
        self,
        ann_var="linearPred",  # "maxp", "entropy", "lin_error", etc.
        bayes_var="bayesLinPred",  # "bayesPred" or "bayesProba"
        mode="full",  # "selected" or "full"
        speed="fast",  # "fast", "slow", "all"
        suffixes=None,  # list of phases to include
        nameExps=None,  # list of nameExps
        winMS_list=None,  # list of winMS
        bins=50,  # number of bins for hist2d
        save=True,
        folder=None,
        show=False,
        normed=True,  # whether to normalize the 2D histogram
    ):
        """
        Plot a mean 2D histogram (heatmap) between ANN linear predictions and Bayesian predictions.
        Aggregated across (mouse, nameExp, phase, winMS).

        Args:
            bayes_var (str): Bayesian variable ("bayesPred" or "bayesProba").
            mode (str): "selected" or "full" for error filtering.
            speed (str): "fast", "slow", "all".
            suffixes (list): phases to include.
            nameExps (list): which experiments to include.
            winMS_list (list): which window sizes to include.
            bins (int): number of bins for hist2d.
            save (bool): save the figure.
            folder (str): folder to save figures.
            show (bool): show interactively.
            normed (bool): normalize histogram to probability density.
        """

        folder = folder or getattr(self, "folderFigures", None)
        df = self.results_df.copy()

        # --- apply filters ---
        if suffixes is not None:
            suffix_tags = [s.strip("_") for s in suffixes]
            df = df[df["phase"].isin(suffix_tags)]
        if nameExps is not None:
            df = df[df["nameExp"].isin(nameExps)]
        if winMS_list is not None:
            df = df[df["winMS"].isin(winMS_list)]

        all_ann, all_bayes = [], []

        for _, row in df.iterrows():
            mouse_results = row["results"]
            ws = str(row["winMS"])
            suffix = f"_{row['phase']}"

            # --- load decoding results (ANN) ---
            pkl_path = os.path.join(
                mouse_results.projectPath.experimentPath,
                "results",
                ws,
                f"decoding_results{suffix}.pkl",
            )
            if not os.path.exists(pkl_path):
                continue
            try:
                with open(pkl_path, "rb") as f:
                    decoding_results = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {pkl_path}: {e}")
                continue

            # --- ANN linpred ---
            if not (ann_var in decoding_results or ann_var in row):
                continue
            try:
                ann_vals = decoding_results[ann_var].flatten()
            except KeyError:
                ann_vals = row[ann_var]

            # --- apply speed mask if needed ---
            mask = np.ones_like(ann_vals, dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]

            ann_vals = ann_vals[mask]

            # --- Bayesian variable ---
            if bayes_var not in row or row[bayes_var] is None:
                continue
            bayes_vals = row[bayes_var]
            mask = np.ones_like(bayes_vals, dtype=bool)
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["bayesProba"] >= row["bayesProbaThreshold"]

            if isinstance(bayes_vals, np.ndarray):
                bayes_vals = bayes_vals[mask]
            else:
                bayes_vals = np.array([bayes_vals])

            if len(ann_vals) == 0 or len(bayes_vals) == 0:
                continue

            all_ann.append(ann_vals)
            all_bayes.append(bayes_vals)

            del decoding_results

        if not all_ann:
            print("No data to plot hist2d.")
            return

        # --- concatenate all mice/conditions ---
        all_ann = np.concatenate(all_ann)
        all_bayes = np.concatenate(all_bayes)

        # --- 2D histogram ---
        H, xedges, yedges = np.histogram2d(
            all_bayes, all_ann, bins=bins, density=normed
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax, label="Density" if normed else "Counts")
        ax.set_xlabel(f"{bayes_var}")
        ax.set_ylabel("ANN linPred")
        ax.set_title("Mean 2D correlation: ANN linpred vs Bayesian predictions")

        fig.tight_layout()
        if save and folder is not None:
            fname = f"hist2d_linpred_vs_{bayes_var}"
            fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
            fig.savefig(os.path.join(folder, fname + ".svg"))
        if show:
            plt.show()
        plt.close(fig)

    def plot_ann_pred_by_stride_and_phase(
        self,
        phase_list=None,
        stride_list=None,
        winMS_list=None,
        folder=None,
        show=False,
        reduce_fn="median",  # function to reduce errors within each group
    ):
        # --- Filter relevant rows first ---
        df = self.results_df.copy()
        if phase_list is not None:
            phase_list = phase_list if isinstance(phase_list, list) else [phase_list]
            df = df[df["phase"].isin(phase_list)]
        if stride_list is not None:
            stride_list = (
                stride_list if isinstance(stride_list, list) else [stride_list]
            )
            df = df[df["stride"].isin(stride_list)]
        if winMS_list is not None:
            winMS_list = winMS_list if isinstance(winMS_list, list) else [winMS_list]
            winMS_list = [int(w) for w in winMS_list]
            df = df[df["winMS"].astype(int).isin(winMS_list)]

        # --- helper to get speed mask from training phase ---
        def get_speed_mask(row, df):
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )["results"]
            if len(res) == 0:
                return None
            return (
                res.iloc[0]
                .data_helper.fullBehavior["Times"]["speedFilter"]
                .flatten()[row.posIndex_NN]
            )

        # --- helper to get true training mask ---
        def get_true_train_mask(row, df):
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )["results"]
            if len(res) == 0:
                return None
            train_mask = res.iloc[0].data_helper.fullBehavior["Times"]["trainEpochs"]
            return inEpochsMask(row.timeNN, train_mask)

        # --- compute mean errors ---
        errors = []
        for (mouse, phase, winMS, stride), group in df.groupby(
            ["mouse_manipe", "phase", "winMS", "stride"]
        ):
            # Build mask (vector of booleans, same length as group)
            speed_mask = group.apply(lambda r: get_speed_mask(r, df), axis=1)
            mask = np.array(speed_mask.tolist(), dtype=bool)

            if phase == "training":
                train_mask = group.apply(lambda r: get_true_train_mask(r, df), axis=1)
                mask = mask & np.array(train_mask.tolist(), dtype=bool)

            lin_true = np.array(group["linearTrue"].tolist())[mask]
            lin_pred = np.array(group["linearPred"].tolist())[mask]

            if len(lin_true) > 0:
                if reduce_fn == "mean":
                    mean_err = np.mean(np.abs(lin_true - lin_pred))
                elif reduce_fn == "median":
                    mean_err = np.median(np.abs(lin_true - lin_pred))
                else:
                    raise ValueError("reduce_fn must be 'mean' or 'median'")
                errors.append([mouse, phase, winMS, stride, mean_err])

        err_df = pd.DataFrame(
            errors,
            columns=[
                "mouse_manipe",
                "phase",
                "winMS",
                "stride",
                "mean_error" if reduce_fn == "mean" else "median_error",
            ],
        )

        fig, ax = plt.subplots()
        # Draw boxplot
        sns.boxplot(
            data=err_df,
            x="stride",
            y="mean_error" if reduce_fn == "mean" else "median_error",
            hue="phase",
            showcaps=False,
            showfliers=False,
            order=stride_list if stride_list is not None else ["1", "2", "4"],
            hue_order=phase_list if phase_list is not None else ["training", "pre"],
            ax=ax,
        )

        # Draw scatter
        strip = sns.stripplot(
            data=err_df,
            x="stride",
            y="mean_error" if reduce_fn == "mean" else "median_error",
            hue="phase",
            dodge=True,
            marker="o",
            linewidth=1,
            edgecolor="k",
            alpha=0.7,
            order=stride_list if stride_list is not None else ["1", "2", "4"],
            hue_order=phase_list if phase_list is not None else ["training", "pre"],
            ax=ax,
        )

        # Now connect corresponding dots across phases
        # Extract positions from the scatter artists
        paths = strip.collections  # one PathCollection per hue per x

        # Build a lookup: (stride, phase) -> list of (x, y) coords
        coords = {}
        x_ticks = stride_list if stride_list is not None else ["1", "2", "4"]
        phases = phase_list if phase_list is not None else ["training", "pre"]
        len(phases)

        for i, (stride, phase) in enumerate([(s, p) for s in x_ticks for p in phases]):
            coll = paths[i]
            offsets = coll.get_offsets()
            coords[(stride, phase)] = offsets
            # --- connect dots ---
        if len(phase_list) > 1:
            # case 1: connect across phases
            for (mouse, winMS, stride), sub in err_df.groupby(
                ["mouse_manipe", "winMS", "stride"]
            ):
                if set(sub["phase"]) >= set(phase_list):  # both phases present
                    pts = []
                    for _, row in sub.iterrows():
                        stride_val = str(row["stride"])
                        phase_val = row["phase"]
                        arr = coords[(stride_val, phase_val)]
                        idx = np.argmin(
                            np.abs(
                                arr[:, 1]
                                - row[
                                    (
                                        "mean_error"
                                        if reduce_fn == "mean"
                                        else "median_error"
                                    )
                                ]
                            )
                        )
                        pts.append(arr[idx])
                    if len(pts) == len(phase_list):
                        ax.plot(
                            [p[0] for p in pts],
                            [p[1] for p in pts],
                            color="gray",
                            alpha=0.6,
                            linewidth=1,
                        )

        else:
            # case 2: connect across strides (same mouse+winMS, one phase only)
            phase = phase_list[0]
            for (mouse, winMS), sub in err_df.query("phase == @phase").groupby(
                ["mouse_manipe", "winMS"]
            ):
                pts = []
                for _, row in sub.iterrows():
                    stride_val = str(row["stride"])
                    arr = coords[(stride_val, phase)]
                    idx = np.argmin(
                        np.abs(
                            arr[:, 1]
                            - row[
                                (
                                    "mean_error"
                                    if reduce_fn == "mean"
                                    else "median_error"
                                )
                            ]
                        )
                    )
                    pts.append(arr[idx])
                if len(pts) > 1:
                    pts = sorted(pts, key=lambda x: x[0])  # sort by x-position (stride)
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        color="gray",
                        alpha=0.6,
                        linewidth=1,
                    )
        # Now loop over each mouse/winMS/stride and connect
        for (mouse, winMS, stride), sub in err_df.groupby(
            ["mouse_manipe", "winMS", "stride"]
        ):
            if len(sub) == 2:  # both phases present
                pts = []
                for _, row in sub.iterrows():
                    stride_val = str(row["stride"])
                    phase_val = row["phase"]
                    # find closest point (match y)
                    arr = coords[(stride_val, phase_val)]
                    idx = np.argmin(
                        np.abs(
                            arr[:, 1]
                            - row[
                                (
                                    "mean_error"
                                    if reduce_fn == "mean"
                                    else "median_error"
                                )
                            ]
                        )
                    )
                    pts.append(arr[idx])
                if len(pts) == 2:
                    ax.plot(
                        [pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        color="gray",
                        alpha=0.6,
                        linewidth=1,
                    )

        # --- outlier labeling ---
        df_phase = err_df.copy()
        df_phase = df_phase.rename(columns={"mouse_manipe": "mouse"})
        df_metric = df_phase[
            ["stride", "mouse", "mean_error" if reduce_fn == "mean" else "median_error"]
        ].dropna()

        for stride in stride_list:
            vals = df_metric[df_metric["stride"] == stride][
                "mean_error" if reduce_fn == "mean" else "median_error"
            ].dropna()
            if vals.empty:
                continue
            fliers = [y for stat in boxplot_stats(vals) for y in stat["fliers"]]
            for outlier in fliers:
                outlier_rows = df_metric[
                    (df_metric["stride"] == stride)
                    & (
                        df_metric[
                            "mean_error" if reduce_fn == "mean" else "median_error"
                        ]
                        == outlier
                    )
                ]
                for _, row in outlier_rows.iterrows():
                    x = stride_list.index(stride)
                    ax.annotate(
                        row["mouse"],
                        xy=(
                            x,
                            row[
                                "mean_error" if reduce_fn == "mean" else "median_error"
                            ],
                        ),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=10,
                        color="red",
                    )

        plt.xlabel("Stride")
        plt.ylabel(
            "Mean Linear Error" if reduce_fn == "mean" else "Median Linear Error"
        )
        plt.title(
            "Mean LinError (Pred vs True) filtered by speed_mask"
            if reduce_fn == "mean"
            else "Median LinError (Pred vs True) filtered by speed_mask"
        )
        plt.legend(title="Dataset")
        plt.tight_layout()
        if folder is not None:
            plt.savefig(
                os.path.join(folder, f"{reduce_fn}_linError_by_stride_and_phase.png"),
                dpi=150,
            )
            plt.savefig(
                os.path.join(folder, f"{reduce_fn}_linError_by_stride_and_phase.svg")
            )
        if show:
            plt.show()

    def plot_ann_pred_by_stride_and_winMS(
        self,
        phase_list=None,
        stride_list=None,
        winMS_list=None,
        folder=None,
        show=False,
        reduce_fn="median",  # function to reduce errors within each group
    ):
        # --- Filter relevant rows first ---
        df = self.results_df.copy()
        if phase_list is not None:
            phase_list = phase_list if isinstance(phase_list, list) else [phase_list]
            df = df[df["phase"].isin(phase_list)]
        else:
            phase_list = sorted(df["phase"].unique().tolist())
        if stride_list is not None:
            stride_list = (
                stride_list if isinstance(stride_list, list) else [stride_list]
            )
            df = df[df["stride"].isin(stride_list)]
        else:
            stride_list = sorted(df["stride"].unique().tolist())
        if winMS_list is not None:
            winMS_list = winMS_list if isinstance(winMS_list, list) else [winMS_list]
            winMS_list = [int(w) for w in winMS_list]
            df = df[df["winMS"].astype(int).isin(winMS_list)]
        else:
            winMS_list = sorted(df["winMS"].astype(int).unique().tolist())

        # --- helper to get speed mask from training phase ---
        def get_speed_mask(row, df):
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )["results"]
            if len(res) == 0:
                return None
            return (
                res.iloc[0]
                .data_helper.fullBehavior["Times"]["speedFilter"]
                .flatten()[row.posIndex_NN]
            )

        # --- helper to get true training mask ---
        def get_true_train_mask(row, df):
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )["results"]
            if len(res) == 0:
                return None
            train_mask = res.iloc[0].data_helper.fullBehavior["Times"]["trainEpochs"]
            return inEpochsMask(row.timeNN, train_mask)

        # --- compute errors using reduce_fn ---
        errors = []
        for (mouse, phase, winMS, stride), group in df.groupby(
            ["mouse_manipe", "phase", "winMS", "stride"]
        ):
            speed_mask = group.apply(lambda r: get_speed_mask(r, df), axis=1)
            mask = np.array(speed_mask.tolist(), dtype=bool)

            if phase == "training":
                train_mask = group.apply(lambda r: get_true_train_mask(r, df), axis=1)
                mask = mask & np.array(train_mask.tolist(), dtype=bool)

            lin_true = np.array(group["linearTrue"].tolist())[mask]
            lin_pred = np.array(group["linearPred"].tolist())[mask]

            if len(lin_true) > 0:
                if reduce_fn == "mean":
                    err_val = np.mean(np.abs(lin_true - lin_pred))
                elif reduce_fn == "median":
                    err_val = np.median(np.abs(lin_true - lin_pred))
                else:
                    raise ValueError("reduce_fn must be 'mean' or 'median'")
                errors.append([mouse, phase, winMS, stride, err_val])

        err_df = pd.DataFrame(
            errors,
            columns=[
                "mouse_manipe",
                "phase",
                "winMS",
                "stride",
                "mean_error" if reduce_fn == "mean" else "median_error",
            ],
        )

        fig, ax = plt.subplots()
        # Draw boxplot
        sns.boxplot(
            data=err_df,
            x="stride",
            y="mean_error" if reduce_fn == "mean" else "median_error",
            hue="winMS",
            showcaps=False,
            showfliers=False,
            order=stride_list if stride_list is not None else ["1", "2", "4"],
            hue_order=winMS_list if winMS_list is not None else ["36", "108", "252"],
            ax=ax,
        )

        # Draw scatter
        strip = sns.stripplot(
            data=err_df,
            x="stride",
            y="mean_error" if reduce_fn == "mean" else "median_error",
            hue="winMS",
            dodge=True,
            marker="o",
            linewidth=1,
            edgecolor="k",
            alpha=0.7,
            order=stride_list if stride_list is not None else ["1", "2", "4"],
            hue_order=winMS_list if winMS_list is not None else ["36", "108", "252"],
            ax=ax,
        )

        # Now connect corresponding dots across winMSs
        paths = strip.collections  # one PathCollection per hue per x

        # Build a lookup: (stride, winMS) -> list of (x, y) coords
        coords = {}
        x_ticks = stride_list if stride_list is not None else ["1", "2", "4"]
        winMSs = winMS_list if winMS_list is not None else ["36", "108", "252"]
        len(winMSs)

        for i, (stride, winMS) in enumerate([(s, p) for s in x_ticks for p in winMSs]):
            coll = paths[i]
            offsets = coll.get_offsets()
            coords[(stride, winMS)] = offsets

        if len(winMS_list) > 1:
            # case 1: connect across winMSs
            for (mouse, phase, stride), sub in err_df.groupby(
                ["mouse_manipe", "phase", "stride"]
            ):
                if set(sub["winMS"]) >= set(winMS_list):
                    pts = []
                    for _, row in sub.iterrows():
                        stride_val = str(row["stride"])
                        winMS_val = row["winMS"]
                        arr = coords[(stride_val, winMS_val)]
                        idx = np.argmin(
                            np.abs(
                                arr[:, 1]
                                - row[
                                    "mean_error"
                                    if reduce_fn == "mean"
                                    else "median_error"
                                ]
                            )
                        )
                        pts.append(arr[idx])
                    if len(pts) == len(winMS_list):
                        ax.plot(
                            [p[0] for p in pts],
                            [p[1] for p in pts],
                            color="gray",
                            alpha=0.6,
                            linewidth=1,
                        )

        else:
            # case 2: connect across strides (same mouse+winMS, one winMS only)
            winMS = winMS_list[0]
            for (mouse, phase), sub in err_df.query("winMS == @winMS").groupby(
                ["mouse_manipe", "phase"]
            ):
                pts = []
                for _, row in sub.iterrows():
                    stride_val = str(row["stride"])
                    arr = coords[(stride_val, winMS)]
                    idx = np.argmin(
                        np.abs(
                            arr[:, 1]
                            - row[
                                "mean_error" if reduce_fn == "mean" else "median_error"
                            ]
                        )
                    )
                    pts.append(arr[idx])
                if len(pts) > 1:
                    pts = sorted(pts, key=lambda x: x[0])  # sort by x-position (stride)
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        color="gray",
                        alpha=0.6,
                        linewidth=1,
                    )
        # Now loop over each mouse/phase/stride and connect
        for (mouse, phase, stride), sub in err_df.groupby(
            ["mouse_manipe", "phase", "stride"]
        ):
            if len(sub) == 2:  # both winMSs present
                pts = []
                for _, row in sub.iterrows():
                    stride_val = str(row["stride"])
                    winMS_val = row["winMS"]
                    arr = coords[(stride_val, winMS_val)]
                    idx = np.argmin(
                        np.abs(
                            arr[:, 1]
                            - row[
                                "mean_error" if reduce_fn == "mean" else "median_error"
                            ]
                        )
                    )
                    pts.append(arr[idx])
                if len(pts) == 2:
                    ax.plot(
                        [pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        color="gray",
                        alpha=0.6,
                        linewidth=1,
                    )

        # --- outlier labeling ---
        df_winMS = err_df.copy()
        df_winMS = df_winMS.rename(columns={"mouse_manipe": "mouse"})
        # Filter the dataframe to relevant columns
        df_winMS = df_winMS[
            ["stride", "mouse", "mean_error" if reduce_fn == "mean" else "median_error"]
        ].dropna()

        # Uncomment to annotate outliers
        # for stride in stride_list:
        #     vals = df_metric[df_metric["stride"] == stride][
        #         "mean_error" if reduce_fn == "mean" else "median_error"
        #     ].dropna()
        #     if vals.empty:
        #         continue
        #     fliers = [y for stat in boxplot_stats(vals) for y in stat["fliers"]]
        #     for outlier in fliers:
        #         outlier_rows = df_metric[
        #             (df_metric["stride"] == stride)
        #             & (
        #                 df_metric[
        #                     "mean_error" if reduce_fn == "mean" else "median_error"
        #                 ]
        #                 == outlier
        #             )
        #         ]
        #         for _, row in outlier_rows.iterrows():
        #             x = stride_list.index(stride)
        #             ax.annotate(
        #                 row["mouse"],
        #                 xy=(x, row["mean_error" if reduce_fn == "mean" else "median_error"]),
        #                 xytext=(6, 6),
        #                 textcoords="offset points",
        #                 fontsize=10,
        #                 color="red",
        #             )

        plt.xlabel("Stride")
        plt.ylabel(
            "Mean Linear Error" if reduce_fn == "mean" else "Median Linear Error"
        )
        plt.title(
            "Mean LinError (Pred vs True) filtered by speed_mask"
            if reduce_fn == "mean"
            else "Median LinError (Pred vs True) filtered by speed_mask"
        )
        plt.legend(title="Window Size (ms)")
        plt.tight_layout()
        if folder is not None:
            plt.savefig(
                os.path.join(folder, f"{reduce_fn}_linError_by_stride_and_winMS.png"),
                dpi=150,
            )
            plt.savefig(
                os.path.join(folder, f"{reduce_fn}_linError_by_stride_and_winMS.svg")
            )
        if show:
            plt.show()

    def plot_ann_pred_by_phase_and_winMS(
        self,
        phase_list=None,
        stride_list=None,
        winMS_list=None,
        folder=None,
        show=False,
        add_bayes=False,
        bayes_nameExp="new_4d_GaussianHeatMap_LinearLoss_Transformer",
        ax=None,
        entropy_thresh_pct=None,
        chance_level=None,
        palette="Set1",
        alpha=1,
        by="entropy",
        reduce_fn="median",  # function to reduce errors within each group
    ):
        # --- Filter relevant rows first ---
        df = self.results_df.copy().reset_index(drop=False)
        if phase_list is not None:
            tmp_phase_list = (
                phase_list if isinstance(phase_list, list) else [phase_list]
            )
            if "training" not in tmp_phase_list:
                tmp_phase_list = ["training"] + tmp_phase_list
            df = df[df["phase"].isin(tmp_phase_list)]
        else:
            phase_list = sorted(df["phase"].unique().tolist())
        if stride_list is not None:
            stride_list = (
                stride_list if isinstance(stride_list, list) else [stride_list]
            )
            df = df[df["stride"].isin(stride_list)]
        else:
            stride_list = sorted(df["stride"].unique().tolist())

        if len(stride_list) > 1:
            raise ValueError(
                "Warning: Multiple strides found. Consider filtering by a single stride for clarity."
            )

        if winMS_list is not None:
            winMS_list = winMS_list if isinstance(winMS_list, list) else [winMS_list]
            winMS_list = [int(w) for w in winMS_list]
            df = df[df["winMS"].astype(int).isin(winMS_list)]
        else:
            winMS_list = sorted(df["winMS"].astype(int).unique().tolist())

        if add_bayes:
            # Filter for bayes_nameExp
            bayes_df = self.results_df[
                self.results_df["nameExp"] == bayes_nameExp
            ].copy()

        # --- helper to get speed mask from training phase ---
        def get_speed_mask(row, df):
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )["results"]
            if len(res) == 0:
                return None
            return (
                res.iloc[0]
                .data_helper.fullBehavior["Times"]["speedFilter"]
                .flatten()
                .reshape(-1)[row.posIndex_NN]
                .flatten()
            )

        def get_entropy_mask(row, df, thresh_pct):
            good_row = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )
            res = good_row["results"]
            if len(res) == 0:
                return None
            speed_mask = (
                res.iloc[0]
                .data_helper.fullBehavior["Times"]["speedFilter"]
                .flatten()
                .reshape(-1)[good_row["posIndex_NN"].iloc[0]]
                .flatten()
            )
            if by == "entropy":
                thresh = np.percentile(
                    good_row["predLoss"].iloc[0][speed_mask], thresh_pct
                )
                return (row["predLoss"] <= thresh).flatten()
            elif by == "maxp":
                with open(
                    os.path.join(
                        row["results"].projectPath.experimentPath,
                        "..",
                        row["nameExp"],
                        "results",
                        str(row["winMS"]),
                        "decoding_results_training.pkl",
                    ),
                    "rb",
                ) as f:
                    decoding_results = pickle.load(f)
                    thresh = np.percentile(decoding_results["maxp"], 100 - thresh_pct)
                with open(
                    os.path.join(
                        row["results"].projectPath.experimentPath,
                        "..",
                        row["nameExp"],
                        "results",
                        str(row["winMS"]),
                        f"decoding_results_{row['phase']}.pkl",
                    ),
                    "rb",
                ) as f:
                    decoding_results = pickle.load(f)
                    maxp = decoding_results["maxp"]
                    return (maxp >= thresh).flatten()

        def get_speed_mask_bayes(row, df):
            with open(
                os.path.join(
                    row["results"].projectPath.experimentPath,
                    "..",
                    bayes_nameExp,
                    "results",
                    str(row["winMS"]),
                    f"bayes_decoding_results_{row['phase']}.pkl",
                ),
                "rb",
            ) as f:
                decoding_results = pickle.load(f)
            speed_mask = decoding_results["speed_mask"].flatten()
            row["phase"]
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == @phase_value "
                "and winMS == @row.winMS and stride == @row.stride"
            )["posIndex_NN"]
            del decoding_results
            return speed_mask[res.iloc[0]].flatten()

        def get_true_train_mask_bayes(row, df):
            with open(
                os.path.join(
                    row["results"].projectPath.experimentPath,
                    "..",
                    bayes_nameExp,
                    "results",
                    str(row["winMS"]),
                    "bayes_decoding_results_training.pkl",
                ),
                "rb",
            ) as f:
                decoding_results = pickle.load(f)
            times = decoding_results["times"].reshape(-1)
            trainEpochs = row["results"].data_helper.fullBehavior["Times"][
                "trainEpochs"
            ]
            del decoding_results
            return inEpochsMask(times, trainEpochs).flatten()

        # --- helper to get true training mask ---
        def get_true_train_mask(row, df):
            res = df.query(
                "mouse_manipe == @row.mouse_manipe and phase == 'training' "
                "and winMS == @row.winMS and stride == @row.stride"
            )["results"]
            if len(res) == 0:
                return None
            train_mask = res.iloc[0].data_helper.fullBehavior["Times"]["trainEpochs"]
            return inEpochsMask(row.timeNN, train_mask).flatten()

        # --- compute median errors ---
        errors = []
        errors_filtered = []
        bayes_errors = []

        # Collect errors
        for (mouse, phase, winMS, stride), group in df.groupby(
            ["mouse_manipe", "phase", "winMS", "stride"]
        ):
            if phase not in phase_list or int(winMS) not in winMS_list:
                continue
            # Build mask
            speed_mask = group.apply(lambda r: get_speed_mask(r, df), axis=1)
            mask = np.array(speed_mask.tolist(), dtype=bool)

            if phase == "training":
                train_mask = group.apply(lambda r: get_true_train_mask(r, df), axis=1)
                mask = mask & np.array(train_mask.tolist(), dtype=bool)

            lin_true = np.array(group["linearTrue"].tolist())[mask]
            lin_pred = np.array(group["linearPred"].tolist())[mask]

            if len(lin_true) > 0:
                if reduce_fn == "mean":
                    median_err = np.mean(np.abs(lin_true - lin_pred))
                elif reduce_fn == "median":
                    median_err = np.median(np.abs(lin_true - lin_pred))
                errors.append([mouse, phase, winMS, stride, median_err])

            if entropy_thresh_pct is not None:
                entropy_mask = group.apply(
                    lambda r: get_entropy_mask(r, df, entropy_thresh_pct), axis=1
                )
                mask = mask & np.array(entropy_mask.tolist(), dtype=bool)
                lin_true_filtered = np.array(group["linearTrue"].tolist())[mask]
                lin_pred_filtered = np.array(group["linearPred"].tolist())[mask]

                if len(lin_true_filtered) > 0:
                    if reduce_fn == "mean":
                        median_err_filtered = np.mean(
                            np.abs(lin_true_filtered - lin_pred_filtered)
                        )
                    elif reduce_fn == "median":
                        median_err_filtered = np.median(
                            np.abs(lin_true_filtered - lin_pred_filtered)
                        )
                    errors_filtered.append(
                        [mouse, phase, winMS, stride, median_err_filtered]
                    )

            if add_bayes:
                # Build mask
                bayes_df["stride"] = np.unique(df["stride"].values)[0]
                speed_mask = group.apply(
                    lambda r: get_speed_mask_bayes(r, bayes_df), axis=1
                )
                mask = np.array(speed_mask.tolist(), dtype=bool)

                if phase == "training":
                    train_mask = group.apply(
                        lambda r: get_true_train_mask_bayes(r, bayes_df), axis=1
                    )
                    mask = mask & np.array(train_mask.tolist(), dtype=bool)
                bayes_pred = np.array(
                    bayes_df[
                        (bayes_df["mouse_manipe"] == mouse)
                        & (bayes_df["phase"] == phase)
                        & (bayes_df["winMS"] == winMS)
                    ]["linearPred"].tolist()
                )[mask]
                bayes_true = np.array(
                    bayes_df[
                        (bayes_df["mouse_manipe"] == mouse)
                        & (bayes_df["phase"] == phase)
                        & (bayes_df["winMS"] == winMS)
                    ]["linearTrue"].tolist()
                )[mask]
                if len(lin_true) > 0:
                    if reduce_fn == "mean":
                        bayes_median_err = np.mean(np.abs(bayes_true - bayes_pred))
                    elif reduce_fn == "median":
                        bayes_median_err = np.median(np.abs(bayes_true - bayes_pred))
                    bayes_errors.append([mouse, phase, winMS, stride, bayes_median_err])

        # Create DataFrames
        err_df = pd.DataFrame(
            errors,
            columns=[
                "mouse_manipe",
                "phase",
                "winMS",
                "stride",
                "median_error" if reduce_fn == "median" else "mean_error",
            ],
        )

        if add_bayes:
            bayes_err_df = pd.DataFrame(
                bayes_errors,
                columns=[
                    "mouse_manipe",
                    "phase",
                    "winMS",
                    "stride",
                    "median_error" if reduce_fn == "median" else "mean_error",
                ],
            )
        if entropy_thresh_pct is not None:
            err_df_filtered = pd.DataFrame(
                errors_filtered,
                columns=[
                    "mouse_manipe",
                    "phase",
                    "winMS",
                    "stride",
                    "median_error_filtered"
                    if reduce_fn == "median"
                    else "mean_error_filtered",
                ],
            )
            err_df = err_df.merge(
                err_df_filtered[
                    [
                        "mouse_manipe",
                        "phase",
                        "winMS",
                        "stride",
                        "median_error_filtered"
                        if reduce_fn == "median"
                        else "mean_error_filtered",
                    ]
                ],
                on=["mouse_manipe", "phase", "winMS", "stride"],
                how="left",
            )

        if ax is None:
            fig, ax = plt.subplots()

        # --- ANN plots ---
        sns.boxplot(
            data=err_df,
            x="phase",
            y="median_error" if reduce_fn == "median" else "mean_error",
            hue="winMS",
            showcaps=False,
            showfliers=False,
            order=phase_list
            if phase_list is not None
            else ["training", "pre", "cond", "post"],
            hue_order=winMS_list if winMS_list is not None else ["36", "108", "252"],
            ax=ax,
            palette=palette,
            boxprops=dict(alpha=alpha),
        )

        ann_strip = sns.stripplot(
            data=err_df,
            x="phase",
            y="median_error" if reduce_fn == "median" else "mean_error",
            hue="winMS",
            dodge=True,
            marker="o",
            linewidth=1,
            edgecolor="k",
            order=phase_list
            if phase_list is not None
            else ["training", "pre", "cond", "post"],
            hue_order=winMS_list if winMS_list is not None else ["36", "108", "252"],
            ax=ax,
            palette=palette,
            alpha=0.7 * alpha,
        )

        # --- Bayesian plots ---
        if add_bayes:
            sns.boxplot(
                data=bayes_err_df,
                x="phase",
                y="median_error" if reduce_fn == "median" else "mean_error",
                hue="winMS",
                showcaps=False,
                showfliers=False,
                order=phase_list
                if phase_list is not None
                else ["training", "pre", "cond", "post"],
                hue_order=winMS_list
                if winMS_list is not None
                else ["36", "108", "252"],
                ax=ax,
                palette="Set1",
                boxprops=dict(alpha=0.3),
            )

            bayes_strip = sns.stripplot(
                data=bayes_err_df,
                x="phase",
                y="median_error" if reduce_fn == "median" else "mean_error",
                hue="winMS",
                dodge=True,
                marker="D",
                linewidth=1,
                edgecolor="k",
                alpha=0.7,
                order=phase_list
                if phase_list is not None
                else ["training", "pre", "cond", "post"],
                hue_order=winMS_list
                if winMS_list is not None
                else ["36", "108", "252"],
                ax=ax,
                palette="Set1",
            )

        # --- Connect points for ANN and Bayes ---
        def connect_points(strip, data_df, color="gray"):
            paths = strip.collections
            x_ticks = (
                phase_list
                if phase_list is not None
                else ["training", "pre", "cond", "post"]
            )
            winMSs = winMS_list if winMS_list is not None else ["36", "108", "252"]

            coords = {}
            for i, (phase, winMS) in enumerate(
                [(s, p) for s in x_ticks for p in winMSs]
            ):
                coll = paths[i]
                offsets = coll.get_offsets()
                coords[(phase, winMS)] = offsets

            # Connect across phases or winMSs
            if len(winMSs) > 1:
                for (mouse, phase, _), sub in data_df.groupby(
                    ["mouse_manipe", "phase", "phase"]
                ):
                    if set(sub["winMS"]) >= set(winMSs):
                        pts = []
                        for _, row in sub.iterrows():
                            phase_val = str(row["phase"])
                            winMS_val = row["winMS"]
                            arr = coords[(phase_val, winMS_val)]
                            idx = np.argmin(
                                np.abs(
                                    arr[:, 1]
                                    - row[
                                        "median_error"
                                        if reduce_fn == "median"
                                        else "mean_error"
                                    ]
                                )
                            )
                            pts.append(arr[idx])
                        if len(pts) == len(winMSs):
                            ax.plot(
                                [p[0] for p in pts],
                                [p[1] for p in pts],
                                color=color,
                                alpha=0.6,
                                linewidth=1,
                            )
            else:
                winMS = winMSs[0]
                for (mouse, phase), sub in data_df.query("winMS == @winMS").groupby(
                    ["mouse_manipe", "phase"]
                ):
                    pts = []
                    for _, row in sub.iterrows():
                        phase_val = str(row["phase"])
                        arr = coords[(phase_val, winMS)]
                        idx = np.argmin(
                            np.abs(
                                arr[:, 1]
                                - row[
                                    "median_error"
                                    if reduce_fn == "median"
                                    else "mean_error"
                                ]
                            )
                        )
                        pts.append(arr[idx])
                    if len(pts) > 1:
                        pts = sorted(pts, key=lambda x: x[0])
                        ax.plot(
                            [p[0] for p in pts],
                            [p[1] for p in pts],
                            color=color,
                            alpha=0.6,
                            linewidth=1,
                        )

        connect_points(ann_strip, err_df, color="gray")
        if add_bayes:
            connect_points(bayes_strip, bayes_err_df, color="blue")

        # --- Outlier labeling (ANN only) ---
        df_winMS = err_df.copy().rename(columns={"mouse_manipe": "mouse"})
        df_metric = df_winMS[
            [
                "phase",
                "mouse",
                "median_error" if reduce_fn == "median" else "mean_error",
            ]
        ].dropna()

        for phase in phase_list:
            vals = df_metric[df_metric["phase"] == phase][
                "median_error" if reduce_fn == "median" else "mean_error"
            ].dropna()
            if vals.empty:
                continue
            fliers = [y for stat in boxplot_stats(vals) for y in stat["fliers"]]
            for outlier in fliers:
                outlier_rows = df_metric[
                    (df_metric["phase"] == phase)
                    & (df_metric["median_error"] == outlier)
                ]
                for _, row in outlier_rows.iterrows():
                    x = phase_list.index(phase)
                    ax.annotate(
                        row["mouse"],
                        xy=(
                            x,
                            row["median_error"]
                            if reduce_fn == "median"
                            else row["mean_error"],
                        ),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=10,
                        color="red",
                    )

        if chance_level is not None:
            ax.axhline(
                y=chance_level,
                color="black",
                linestyle="--",
                label="Chance Level",
                linewidth=2.5,
            )
        # change ylim to 0.5 at least
        if ax.get_ylim()[1] < 0.5:
            ax.set_ylim(0, max(0.5, 1.15 * ax.get_ylim()[1]))

        ax.set_xlabel("Phase")
        ax.set_ylabel(
            "Median Linear Error" if reduce_fn == "median" else "Mean Linear Error"
        )
        if reduce_fn == "median":
            title = "Median LinError"
        elif reduce_fn == "mean":
            title = "Mean LinError"
        if entropy_thresh_pct is not None:
            title += f" (Filtered by {entropy_thresh_pct}th Percentile {by})"
        if add_bayes:
            title += " (ANN vs Bayes)"
        else:
            title += " (ANN)"
        plt.title(title)
        plt.legend(title="Window Size (ms)")
        plt.tight_layout()
        if folder is not None:
            plt.savefig(
                os.path.join(folder, f"{reduce_fn}_linError_by_phase_and_winMS.png"),
                dpi=150,
            )
            plt.savefig(
                os.path.join(folder, f"{reduce_fn}_linError_by_phase_and_winMS.svg")
            )
        if show:
            plt.show()
        plt.close()

        return err_df

    def get_concatenated_tuning_curves(
        self,
        suffix: str = "_training",
        feature_name: str = "linearTrue",
        idWindow: int = 0,
        use_speed_filter: bool = True,
        count_thresh: Optional[int] = None,
        **kwargs,
    ):
        """
        Computes the tuning curves for all mice on one suffix.

        Parameters:
        - suffix: The suffix to use for accessing the results. If None, it will be determined as training.
        - feature_name: The name of the feature to compute tuning curves for (default is "linearTrue").
        - idWindow: The index of the window to use for accessing the feature and speed mask (default is 0).
        - use_speed_filter: Whether to apply a speed filter to the epochs used for computing tuning curves (default is True).
        - count_thresh: If provided, neurons with total counts below this threshold will be excluded from the tuning curves.
        - kwargs: Additional keyword arguments for plotting the tuning curves. If 'plot' is True (default), the tuning curves will be plotted. You can also provide 'sort_map' and 'list_neurons' for sorting the tuning curves.

        Returns:
        - concat: A concatenated array of tuning curves for all mice.
        - sort_map: A mapping of neuron IDs to their sorted positions, if sorting was performed.
        """

        keep_mice = kwargs.pop("keep_mice", None)
        remove_mice = kwargs.pop("remove_mice", None)
        bin_size = kwargs.pop("bin_size", 0.05)
        mode = kwargs.pop("mode", "closest")

        if keep_mice is not None and remove_mice is not None:
            raise ValueError("Cannot specify both keep_mice and remove_mice.")

        if keep_mice is not None:
            if not isinstance(keep_mice, list):
                keep_mice = [keep_mice]
            keep_mice = set([str(mouse) for mouse in keep_mice])

        if remove_mice is not None:
            if not isinstance(remove_mice, list):
                remove_mice = [remove_mice]
            remove_mice = set([str(mouse) for mouse in remove_mice])

        # Standardize phase extraction text string
        phase = suffix.strip("_") if "_" in suffix else suffix

        # --- MULTIINDEX FIX 1: Extract phase cross-section safely ---
        # Instead of flat column querying, pull the explicit index slice
        if "phase" in self.results_df.index.names:
            phase_df = self.results_df.xs(phase, level="phase")
        else:
            # Fallback if index was completely flattened beforehand
            phase_df = self.results_df[self.results_df["phase"] == phase]

        if phase_df.empty:
            print(f"Warning: No data found matching phase '{phase}' in results_df.")
            return np.array([]), np.array([]), np.array([])

        manipe = kwargs.pop("manipe", None)
        if manipe is not None:
            if "manipe" in phase_df.index.names:
                phase_df = phase_df.xs(manipe, level="manipe", drop_level=False)
            else:
                phase_df = phase_df[phase_df["manipe"] == manipe]

            if phase_df.empty:
                print(
                    f"Warning: No data found matching manipe '{manipe}' in phase '{phase}' of results_df."
                )

        spike_datas_list = []
        tuning_curves_list = []

        # Dynamic detection of your exact indexing names to support either variant
        mouse_col = "mouse_name" if "mouse_name" in phase_df.index.names else "mouse"
        groupby_levels = [mouse_col, "manipe"]

        # --- MULTIINDEX FIX 2: Group by MultiIndex Levels cleanly ---
        for (mouse, manipe), group_df in phase_df.groupby(level=groupby_levels):
            # Keep/Remove identifier filtering flags
            if keep_mice is not None and str(mouse) not in keep_mice:
                print(f"Skipping mouse {mouse} as it is not in the keep_mice list.")
                continue
            if remove_mice is not None and str(mouse) in remove_mice:
                print(f"Skipping mouse {mouse} as it is in the remove_mice list.")
                continue

            # --- MULTIINDEX FIX 3: Target a single window safely ---
            # Since group_df contains multiple windows, locate the specified idWindow index row
            if "winMS" in group_df.index.names:
                try:
                    # Find the row matching the explicit window index value
                    # (Assuming windows are indexed by actual ID numbers or ordinal locations)
                    unique_wins = group_df.index.get_level_values("winMS").unique()
                    target_win = unique_wins[idWindow]
                    row_slice = group_df.xs(target_win, level="winMS").iloc[0]
                except IndexError:
                    print(
                        f"Warning: Window offset idWindow={idWindow} out of range for mouse {mouse}. Defaulting to first row."
                    )
                    row_slice = group_df.iloc[0]
            else:
                row_slice = group_df.iloc[0]

            # Extract object data container out of your row metrics
            mouse_results = row_slice["results"]
            mouse_label = f"mouse {mouse} | {manipe}"

            mouse_tuning_curves, _, mouse_spike_data, _ = (
                _compute_tuning_curves_for_result(
                    mouse_results,
                    suffix=suffix,
                    feature_name=feature_name,
                    idWindow=idWindow,
                    use_speed_filter=use_speed_filter,
                    count_thresh=None,
                    bin_size=bin_size,
                    mode=mode,
                    epoch=kwargs.get("epoch", None),
                    half=kwargs.get("half", None),
                )
            )

            if (
                mouse_tuning_curves is None
                or np.nansum(mouse_tuning_curves.values) == 0
            ):
                warn(
                    f"Warning: Tuning curves or spike data could not be computed for {mouse_label}. Adding NaN only."
                )

            mouse_spike_data.set_info(
                metadata={
                    "phase": [phase] * len(mouse_spike_data),
                    "mouse": [mouse_label] * len(mouse_spike_data),
                }
            )

            spike_datas_list.append(mouse_spike_data)
            tuning_curves_list.append(mouse_tuning_curves)

        if not tuning_curves_list:
            print("No valid data processed for concatenated tuning curves.")
            return np.array([]), np.array([]), np.array([])

        self.all_spikes = TsGroup.merge_group(
            *spike_datas_list, reset_index=True, reset_time_support=True
        )

        id_neurons = np.arange(0, len(self.all_spikes))
        if count_thresh is not None:
            concat = []
            kept = []
            for tc in tuning_curves_list:
                under_thresh = np.sum(tc.counts, axis=1) < count_thresh
                concat.append(tc[~under_thresh])
                kept.append(~under_thresh)
            concat = np.concatenate(concat, axis=0)
            kept = np.concatenate(kept, axis=0)
            id_neurons = id_neurons[kept]
            print(
                f"Applied count threshold of {count_thresh}, keeping {len(id_neurons)} neurons out of {kept.shape[0]}."
            )
        else:
            concat = np.concatenate(tuning_curves_list, axis=0)

        if kwargs.pop("plot", True):
            ordered, sort_map = self.compute_linear_tuning_curves_order(
                lin_place_fields=concat,
                bin_edges=np.linspace(0, 1, concat.shape[1] + 1),
                sort_map=kwargs.pop("sort_map", None),
                list_neurons=kwargs.pop("list_neurons", None),
            )
            title = kwargs.pop(
                "title",
                f"""LT Curves on {feature_name}
                ({phase} - speed {use_speed_filter})""",
            )
            kwargs["title"] = title
            kwargs["normalize"] = True
            self.plot_linear_tuning_curves(ordered, **kwargs)
            return concat, sort_map, id_neurons

        return concat, np.arange(concat.shape[0]), id_neurons

    def run_comprehensive_onoff_analysis(
        self,
        feature_name: str = "linearTrue",
        around: str = "freezing",
        use_speed_filter: bool = True,
        remove_mice: Optional[List[str]] = None,
        count_thresh: int = 200,
        min_count_thresh: int = 200,
        manipe: Optional[str] = None,
        focus_on: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extracts PETH count properties, defines ON/OFF/Neutral/NAN populations,
        and cross-references them to Multi-Phase Spatial Tuning Curves including
        first/second session halves for split-half baseline control stability metrics
        across ALL neurons and specific functional subpopulations.
        """
        phase_build = "_training"
        phases = ["cond", "post"]
        all_phases = [phase_build] + phases

        # 1. Base reference extraction to establish master arrays and id allocations
        print("Extracting baseline tuning curve structures for freezing...")
        concat_base, sort_map_base, id_neurons_base = (
            self.get_concatenated_tuning_curves(
                suffix=phase_build,
                feature_name=feature_name,
                add_colorbar=False,
                count_thresh=count_thresh,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
                manipe=manipe,
            )
        )
        concat_base, sort_map_base = self.compute_linear_tuning_curves_order(
            concat_base, bin_edges=np.linspace(0, 1, concat_base.shape[1] + 1)
        )

        mouse_peth_registry = {}

        phase = phase_build.strip("_") if "_" in phase_build else phase_build
        phase_df = (
            self.results_df.xs(phase, level="phase")
            if "phase" in self.results_df.index.names
            else self.results_df
        )
        if manipe is not None:
            phase_df = phase_df.xs(manipe, level="manipe", drop_level=False)

        mouse_col = "mouse_name" if "mouse_name" in phase_df.index.names else "mouse"
        groupby_levels = [mouse_col, "manipe"]

        # 2. Iterate through rows to find raw modulated freeze properties
        print("Processing freeze modulation metrics on raw count data...")
        global_neuron_offset = 0
        global_on_indices = []
        global_off_indices = []
        global_uninfluenced_indices = []
        global_nan_indices = []
        global_delayed_on_indices = []

        for (mouse, manipe), group_df in phase_df.groupby(level=groupby_levels):
            if remove_mice is not None and str(mouse) in remove_mice:
                continue

            row_slice = (
                group_df.xs(
                    group_df.index.get_level_values("winMS").unique()[0], level="winMS"
                ).iloc[0]
                if "winMS" in group_df.index.names
                else group_df.iloc[0]
            )
            mouse_results: Mouse_Results = row_slice["results"]

            if around == "freezing":
                peth_res = mouse_results.compute_freeze_onoff_counts(
                    count_thresh=min_count_thresh
                )
            else:
                peth_res = mouse_results.compute_event_onoff_counts(
                    around=around, count_thresh=min_count_thresh, focus_on=focus_on
                )

            if peth_res is not None:
                mouse_peth_registry[str(mouse)] = peth_res

                # Map local boolean masks to the cumulative global vector layout
                local_on = (
                    np.where(peth_res["on_neurons_mask"])[0] + global_neuron_offset
                )
                local_off = (
                    np.where(peth_res["off_neurons_mask"])[0] + global_neuron_offset
                )
                local_uninfluenced = (
                    np.where(peth_res["uninfluenced_neurons_mask"])[0]
                    + global_neuron_offset
                )
                local_nan = (
                    np.where(peth_res["nan_neurons_mask"])[0] + global_neuron_offset
                )
                # Safely handle delayed_on if it exists (returns an empty list if around="ripples")
                local_delayed = []
                if "delayed_on_neurons_mask" in peth_res:
                    local_delayed = (
                        np.where(peth_res["delayed_on_neurons_mask"])[0]
                        + global_neuron_offset
                    )

                global_on_indices.extend(local_on)
                global_off_indices.extend(local_off)
                global_uninfluenced_indices.extend(local_uninfluenced)
                global_nan_indices.extend(local_nan)
                global_delayed_on_indices.extend(local_delayed)
                global_neuron_offset += peth_res["n_neurons_raw"]
            else:
                try:
                    n_raw = len(mouse_results.DataHelper.get_spike_data())
                    global_nan_indices.extend(np.arange(n_raw) + global_neuron_offset)
                    global_neuron_offset += n_raw
                except Exception:
                    pass

        global_on_indices = np.array(global_on_indices)
        global_off_indices = np.array(global_off_indices)
        global_uninfluenced_indices = np.array(global_uninfluenced_indices)
        global_nan_indices = np.array(global_nan_indices)
        global_delayed_on_indices = np.array(global_delayed_on_indices)

        # 3. Dynamic multi-phase tracking loops for spatial tuning maps (Full, 1st half, 2nd half)
        tuning_curves_by_phase = {}
        tuning_curves_by_phase_1st = {}
        tuning_curves_by_phase_2nd = {}

        print(f"Extracting spatial patterns for feature target: {feature_name}...")
        for suff in all_phases:
            # Full phase block matrix extraction
            tc_matrix, _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                feature_name=feature_name,
                add_colorbar=False,
                count_thresh=None,
                sort_map=sort_map_base,
                list_neurons=id_neurons_base,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            tuning_curves_by_phase[suff] = tc_matrix

            # First half-block matrix extraction
            tc_matrix_1st, _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                feature_name=feature_name,
                half="first",
                add_colorbar=False,
                count_thresh=None,
                sort_map=sort_map_base,
                list_neurons=id_neurons_base,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            tuning_curves_by_phase_1st[suff] = tc_matrix_1st

            # Second half-block matrix extraction
            tc_matrix_2nd, _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                feature_name=feature_name,
                half="second",
                add_colorbar=False,
                count_thresh=None,
                sort_map=sort_map_base,
                list_neurons=id_neurons_base,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            tuning_curves_by_phase_2nd[suff] = tc_matrix_2nd

        # Cross-reference tracking definitions against baseline template neurons
        on_neurons_aligned = np.intersect1d(id_neurons_base, global_on_indices)
        off_neurons_aligned = np.intersect1d(id_neurons_base, global_off_indices)
        uninfluenced_neurons_aligned = np.intersect1d(
            id_neurons_base, global_uninfluenced_indices
        )
        delayed_on_neurons_aligned = np.intersect1d(
            id_neurons_base, global_delayed_on_indices
        )
        nan_neurons_aligned = np.intersect1d(id_neurons_base, global_nan_indices)

        on_locs_in_base = np.searchsorted(id_neurons_base, on_neurons_aligned)
        off_locs_in_base = np.searchsorted(id_neurons_base, off_neurons_aligned)
        uninfluenced_locs_in_base = np.searchsorted(
            id_neurons_base, uninfluenced_neurons_aligned
        )
        delayed_on_locs_in_base = np.searchsorted(
            id_neurons_base, delayed_on_neurons_aligned
        )
        nan_locs_in_base = np.searchsorted(id_neurons_base, nan_neurons_aligned)

        # Initialize multi-cohort containers
        tuning_curves_on_population = {}
        tuning_curves_off_population = {}
        tuning_curves_uninfluenced_population = {}
        tuning_curves_delayed_population = {}
        tuning_curves_nan_population = {}

        tuning_curves_on_population_1st = {}
        tuning_curves_off_population_1st = {}
        tuning_curves_delayed_population_1st = {}
        tuning_curves_uninfluenced_population_1st = {}

        tuning_curves_on_population_2nd = {}
        tuning_curves_off_population_2nd = {}
        tuning_curves_delayed_population_2nd = {}
        tuning_curves_uninfluenced_population_2nd = {}

        for suff in all_phases:
            # Slicing full session matrices
            tuning_curves_on_population[suff] = tuning_curves_by_phase[suff][
                on_locs_in_base
            ]
            tuning_curves_off_population[suff] = tuning_curves_by_phase[suff][
                off_locs_in_base
            ]
            tuning_curves_uninfluenced_population[suff] = tuning_curves_by_phase[suff][
                uninfluenced_locs_in_base
            ]
            tuning_curves_delayed_population[suff] = tuning_curves_by_phase[suff][
                delayed_on_locs_in_base
            ]

            tuning_curves_nan_population[suff] = tuning_curves_by_phase[suff][
                nan_locs_in_base
            ]

            # Slicing first-half session matrices
            tuning_curves_on_population_1st[suff] = tuning_curves_by_phase_1st[suff][
                on_locs_in_base
            ]
            tuning_curves_off_population_1st[suff] = tuning_curves_by_phase_1st[suff][
                off_locs_in_base
            ]
            tuning_curves_uninfluenced_population_1st[suff] = (
                tuning_curves_by_phase_1st[suff][uninfluenced_locs_in_base]
            )
            tuning_curves_delayed_population_1st[suff] = tuning_curves_by_phase_1st[
                suff
            ][delayed_on_locs_in_base]

            # Slicing second-half session matrices
            tuning_curves_on_population_2nd[suff] = tuning_curves_by_phase_2nd[suff][
                on_locs_in_base
            ]
            tuning_curves_off_population_2nd[suff] = tuning_curves_by_phase_2nd[suff][
                off_locs_in_base
            ]
            tuning_curves_uninfluenced_population_2nd[suff] = (
                tuning_curves_by_phase_2nd[suff][uninfluenced_locs_in_base]
            )
            tuning_curves_delayed_population_2nd[suff] = tuning_curves_by_phase_2nd[
                suff
            ][delayed_on_locs_in_base]

        print("Pipeline run successfully completed.")
        return {
            "mouse_peths": mouse_peth_registry,
            "id_neurons_true_baseline": id_neurons_base,
            "sort_map_true_baseline": sort_map_base,
            # Aligned Population ID lists
            "id_neurons_on": on_neurons_aligned,
            "id_neurons_off": off_neurons_aligned,
            "id_neurons_uninfluenced": uninfluenced_neurons_aligned,
            "id_neurons_nan": nan_neurons_aligned,
            "id_neurons_delayed_on": delayed_on_neurons_aligned,
            # Master Continuous Tuning Curve Block Lists
            "tuning_curves_all_phases": tuning_curves_by_phase,
            "tuning_curves_all_phases_first_half": tuning_curves_by_phase_1st,
            "tuning_curves_all_phases_second_half": tuning_curves_by_phase_2nd,
            # Full Phase Subpopulations
            "tuning_curves_on_subset": tuning_curves_on_population,
            "tuning_curves_off_subset": tuning_curves_off_population,
            "tuning_curves_uninfluenced_subset": tuning_curves_uninfluenced_population,
            "tuning_curves_nan_subset": tuning_curves_nan_population,
            "tuning_curves_delayed_on_subset": tuning_curves_delayed_population,
            # First Half Subpopulations (Split Control)
            "tuning_curves_on_subset_first_half": tuning_curves_on_population_1st,
            "tuning_curves_off_subset_first_half": tuning_curves_off_population_1st,
            "tuning_curves_uninfluenced_subset_first_half": tuning_curves_uninfluenced_population_1st,
            "tuning_curves_delayed_on_subset_first_half": tuning_curves_delayed_population_1st,
            # Second Half Subpopulations (Split Control)
            "tuning_curves_on_subset_second_half": tuning_curves_on_population_2nd,
            "tuning_curves_off_subset_second_half": tuning_curves_off_population_2nd,
            "tuning_curves_uninfluenced_subset_second_half": tuning_curves_uninfluenced_population_2nd,
            "tuning_curves_delayed_on_subset_second_half": tuning_curves_delayed_population_2nd,
        }

    def run_tuning_curve_analysis(
        self,
        feature_name: str = "linearPred",
        use_speed_filter: bool = True,
        remove_mice: Optional[List[str]] = None,
        count_thresh: int = 200,
        path: Optional[str] = None,
    ):
        phase_build = "_training"
        phases = ["cond", "post"]
        all_phases = [phase_build] + phases

        # remove all mouse_name that dont have each of all_phases in the results_df
        if remove_mice is None:
            mice_with_all_phases = set(
                self.results_df.index.get_level_values("mouse_name")
            )
            for phase in all_phases:
                if "_" in phase:
                    phase = phase.split("_")[1]
                mice_with_phase = set(
                    self.results_df.xs(phase, level="phase").index.get_level_values(
                        "mouse_name"
                    )
                )
                mice_with_all_phases.intersection_update(mice_with_phase)
            remove_mice = list(
                set(self.results_df.index.get_level_values("mouse_name"))
                - mice_with_all_phases
            )
            if remove_mice:
                print(
                    f"Removing mice that do not have all phases {all_phases}: {remove_mice}"
                )

        fig, axs = plt.subplots(
            2, len(phases) + 1, figsize=(14, 10), sharex=True, sharey=True
        )

        # Initializing containers for continuous blocks instead of odd/even
        raw_true, raw_true_1st, raw_true_2nd = dict(), dict(), dict()
        raw_pred, raw_pred_1st, raw_pred_2nd = dict(), dict(), dict()

        # 1. Primary initialization to establish fixed mapping criteria
        unordered_true_training, sort_map, id_neurons = (
            self.get_concatenated_tuning_curves(
                suffix=phase_build,
                add_colorbar=False,
                count_thresh=count_thresh,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
        )
        _, sort_map = self.compute_linear_tuning_curves_order(
            lin_place_fields=unordered_true_training,
            bin_edges=np.linspace(0, 1, unordered_true_training.shape[1] + 1),
        )

        # 2. Extract structured block components across experimental conditions
        for suff in all_phases:
            print(f"Extracting Chronological Blocks for phase: {suff}...")

            # Ground Truth Blocks
            raw_true[suff], _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                add_colorbar=False,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            raw_true_1st[suff], _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                half="first",
                add_colorbar=False,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            raw_true_2nd[suff], _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                half="second",
                add_colorbar=False,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )

            # Model Predictions Blocks
            raw_pred[suff], _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                feature_name=feature_name,
                add_colorbar=False,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            raw_pred_1st[suff], _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                feature_name=feature_name,
                half="first",
                add_colorbar=False,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )
            raw_pred_2nd[suff], _, _ = self.get_concatenated_tuning_curves(
                suffix=suff,
                feature_name=feature_name,
                half="second",
                add_colorbar=False,
                plot=False,
                remove_mice=remove_mice,
                use_speed_filter=use_speed_filter,
            )

        # 3. Clean, coordinate alignment mapping across blocks
        concat_true, concat_true_1st, concat_true_2nd = dict(), dict(), dict()
        concat_pred, concat_pred_1st, concat_pred_2nd = dict(), dict(), dict()

        for i, suff in enumerate(all_phases):
            concat_true[suff] = raw_true[suff][id_neurons][sort_map]
            concat_true_1st[suff] = raw_true_1st[suff][id_neurons][sort_map]
            concat_true_2nd[suff] = raw_true_2nd[suff][id_neurons][sort_map]

            concat_pred[suff] = raw_pred[suff][id_neurons][sort_map]
            concat_pred_1st[suff] = raw_pred_1st[suff][id_neurons][sort_map]
            concat_pred_2nd[suff] = raw_pred_2nd[suff][id_neurons][sort_map]

            axs[0, i].imshow(
                self.normalize_tuning_curves(concat_true[suff]),
                aspect="auto",
                cmap="cmc.batlow",
            )
            axs[1, i].imshow(
                self.normalize_tuning_curves(concat_pred[suff]),
                aspect="auto",
                cmap="cmc.batlow",
            )

            for ax in [axs[0, i], axs[1, i]]:
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )
                ax.set_xlabel("Linear Position")

        # 4. Packaging the fully structured data dictionary for statistics
        analysis_data = {
            "true": concat_true,
            "true_first_half": concat_true_1st,
            "true_second_half": concat_true_2nd,
            "pred": concat_pred,
            "pred_first_half": concat_pred_1st,
            "pred_second_half": concat_pred_2nd,
        }

        fig.suptitle(f"Tuning Curves Half-Split Control Maps ({feature_name})")
        fig.tight_layout()
        if path is not None:
            fig.savefig(os.path.join(path, f"tc_{feature_name}_halfsplit.png"))
            fig.savefig(os.path.join(path, f"tc_{feature_name}_halfsplit.svg"))
        plt.show()

        return analysis_data

    def compute_all_chance_levels(
        self, use_speed_filter: bool = True, num_shuffles: int = 5, redo: bool = False
    ):
        """
        Computes 1D and 2D chance levels for EVERY row/mouse simultaneously.
        Returns a clean DataFrame with the calculated chance values to join back.
        """

        if (
            all(
                col in self.results_df.columns
                for col in [
                    "chance_mean_1d",
                    "chance_median_1d",
                    "chance_mean_2d",
                    "chance_median_2d",
                ]
            )
            and not redo
        ):
            print("Chance levels already computed. Skipping recalculation.")
            return self.results_df
        flat_df = self.results_df.reset_index()

        # 1. Define grouping keys to map rows back to their training baseline
        group_keys = ["nameExp", "mouse_name", "manipe", "winMS"]
        group_keys = [k if k in flat_df.columns else "mouse" for k in group_keys]

        # 2. Build a high-speed lookup dictionary for the training coordinates
        training_rows = flat_df[flat_df["phase"] == "training"]
        train_pool_lookup = {}

        for _, t_row in training_rows.iterrows():
            m_key = tuple(t_row[k] for k in group_keys)
            t_pos = t_row["featureTrue"]

            if t_pos is not None and isinstance(t_pos, np.ndarray):
                t_pos_2d = t_pos[:, :2]
                # Apply speed mask if it exists
                if (
                    use_speed_filter
                    and "speedMask" in t_row
                    and t_row["speedMask"] is not None
                ):
                    s_mask = np.array(t_row["speedMask"]).astype(bool)
                    t_pos_2d = t_pos_2d[s_mask]

                # Drop NaNs so they don't corrupt the random pool
                t_pos_2d = t_pos_2d[~np.isnan(t_pos_2d).any(axis=1)]
                train_pool_lookup[m_key] = t_pos_2d

        # 3. Storage for our fast calculation
        chance_results = []
        res_obj: Mouse_Results = self.results_df["results"].iloc[
            0
        ]  # Grab reference for l_function

        # 4. Process every row using vectorized NumPy arrays
        for idx, row in flat_df.iterrows():
            m_key = tuple(row[k] for k in group_keys)

            # Default fallbacks if data is missing
            row_res = {
                "chance_mean_1d": np.nan,
                "chance_median_1d": np.nan,
                "chance_mean_2d": np.nan,
                "chance_median_2d": np.nan,
            }

            if row["featureTrue"] is not None and m_key in train_pool_lookup:
                training_pool = train_pool_lookup[m_key]
                pool_size = training_pool.shape[0]
                full_len = row["featureTrue"].shape[0]

                if pool_size > 0 and full_len > 0:
                    # --- Vectorized Sampling Grid ---
                    # Generate a 2D matrix of random indices (40 shuffles x timeline length)
                    rand_idx = np.random.randint(
                        0, pool_size, size=(num_shuffles, full_len)
                    )
                    random_positions = training_pool[
                        rand_idx
                    ]  # Shape: (40, full_len, 2)

                    # --- 2D Euclidean Error Calculation ---
                    true_pos_2d = row["featureTrue"][:, :2]
                    # Subtract and compute norm across all 40 shuffles at the exact same time
                    errors_2d_shuffled = np.linalg.norm(
                        true_pos_2d - random_positions, axis=2
                    )  # Shape: (40, full_len)

                    row_res["chance_mean_2d"] = np.nanmean(
                        np.nanmean(errors_2d_shuffled, axis=1)
                    )
                    row_res["chance_median_2d"] = np.nanmean(
                        np.nanmedian(errors_2d_shuffled, axis=1)
                    )

                    # --- 1D Linear Error Calculation ---
                    if row["linearTrue"] is not None:
                        # Flatten the 3D random matrix to 2D to run the spatial function efficiently
                        flat_rand = random_positions.reshape(-1, 2)
                        flat_lin_rand = res_obj.l_function(flat_rand)[1]

                        # Unpack back to (40, full_len)
                        lin_random = flat_lin_rand.reshape(num_shuffles, full_len)
                        errors_lin_shuffled = np.abs(row["linearTrue"] - lin_random)

                        row_res["chance_mean_1d"] = np.nanmean(
                            np.nanmean(errors_lin_shuffled, axis=1)
                        )
                        row_res["chance_median_1d"] = np.nanmean(
                            np.nanmedian(errors_lin_shuffled, axis=1)
                        )

            chance_results.append(row_res)

        # 5. Convert results to columns and assign them directly back to the DataFrame
        chance_df = pd.DataFrame(chance_results, index=self.results_df.index)

        for col in chance_df.columns:
            self.results_df[col] = chance_df[col]

        print("Mice-individualized chance levels successfully attached!")
        return self.results_df

    def compute_chance_level(
        self, phase: str, use_speed_filter: bool = True, dim: int = 1
    ):
        res_obj: Mouse_Results = self.results_df["results"].iloc[0]

        # 1. Get the target phase timeline interval mask
        # get_epoch_interval returns a tuple: (epoch_data, phase_mask)
        _, phase_mask = res_obj.get_epoch_interval(phase)
        phase_mask = phase_mask.astype(bool)

        # Slice the ground truth position to match just this phase's total duration
        true_position_unfiltered = res_obj.DataHelper.positions[:, :2][phase_mask]
        true_position_unfiltered = true_position_unfiltered[
            ~np.isnan(true_position_unfiltered).any(axis=1)
        ]
        full_len = true_position_unfiltered.shape[0]

        # 2. Isolate your universal "Training Distribution" pool (from the training mask)
        train_mask = res_obj.trainMask.astype(bool)
        if use_speed_filter:
            speed_mask = res_obj.fullBehavior["Times"]["speedFilter"]
            combined_train_mask = speed_mask & train_mask
        else:
            combined_train_mask = train_mask

        training_pool = res_obj.DataHelper.positions[:, :2][combined_train_mask]
        training_pool = training_pool[~np.isnan(training_pool).any(axis=1)]

        # 3. Handle base true dimensions for this phase
        if dim == 1:
            lin_true = res_obj.l_function(true_position_unfiltered)[1]
        elif dim != 2:
            raise ValueError("dim must be 1 or 2")

        chance_level_mean = []
        chance_level_median = []

        # 4. Monte Carlo sampling from the training pool to match this phase's length
        for _ in range(10):
            random_indices = np.random.choice(
                training_pool.shape[0], size=full_len, replace=True
            )
            random_position = training_pool[random_indices]

            if dim == 1:
                lin_random = res_obj.l_function(random_position)[1]
                error_lin = np.abs(lin_true - lin_random)
                chance_level_mean.append(np.nanmean(error_lin))
                chance_level_median.append(np.nanmedian(error_lin))

            elif dim == 2:
                error_2d = np.linalg.norm(
                    true_position_unfiltered - random_position, axis=1
                )
                chance_level_mean.append(np.nanmean(error_2d))
                chance_level_median.append(np.nanmedian(error_2d))

        # 5. Calculate means and standard deviations across shuffles for plotting spans
        return (
            np.mean(chance_level_mean),
            np.mean(chance_level_median),
            np.std(chance_level_mean),
            np.std(chance_level_median),
        )

    def plot_error_barplot(
        self,
        winMS: int,
        dim: int = 1,
        use_speed_filter: bool = True,
        path: Optional[str] = None,
        reduce="mean",
        thresh=0.65,
        split_by: Optional[str] = None,
        redo: bool = False,
    ):
        from statannotations.Annotator import Annotator

        phase_order = ["training", "pre", "cond", "post"]

        reduce_func = getattr(np, f"nan{reduce}")

        def compute_row_metrics(row):
            # 1. Compute Model Prediction Errors
            p_loss = np.array(row["predLoss"])
            if dim == 1:
                l_pred = np.array(row["linearPred"])
                l_true = np.array(row["linearTrue"])
                error = np.abs(l_pred - l_true)
            elif dim == 2:
                p_pred = np.array(row["featurePred"][:, :2])
                p_true = np.array(row["featureTrue"][:, :2])
                error = np.linalg.norm(p_pred - p_true, axis=1)
            else:
                warn("dim should be 1 or 2, will try with more")
                p_pred = np.array(row["featurePred"][:, :dim])
                p_true = np.array(row["featureTrue"][:, :dim])
                error = np.linalg.norm(p_pred - p_true, axis=1)

            loss_mask = p_loss < thresh
            if use_speed_filter:
                s_mask = np.array(row["speedMask"]).astype(bool)
                err_unfiltered = reduce_func(error[s_mask])
                err_filtered = reduce_func(error[s_mask & loss_mask])
            else:
                err_unfiltered = reduce_func(error)
                err_filtered = reduce_func(error[loss_mask])

            # # 2. Compute Chance level for this specific row/mouse
            # ch_mean, ch_med = self.compute_chance_level_for_row(
            #     row, use_speed_filter=use_speed_filter, dim=dim
            # )
            # row_chance = ch_med if reduce == "median" else ch_mean
            #
            return err_unfiltered, err_filtered

        # Apply the function and split results into the dynamic columns
        result_cols = [
            f"computed_{reduce}_error_{thresh}_{dim}d",
            f"computed_{reduce}_error_filtered_{thresh}_{dim}d",
        ]

        if any(res not in self.results_df.columns for res in result_cols) or redo:
            self.results_df[result_cols] = self.results_df.apply(
                compute_row_metrics, axis=1, result_type="expand"
            )

        result_cols.append(
            f"chance_{reduce}_{dim}d",
        )

        if reduce == "mean":
            prefix = "Mean"
        else:
            prefix = "Median"

        if dim == 1:
            value_name = f"{prefix} Linear Error"
        else:
            value_name = f"{prefix} Euclidean Error"

        # Keep split_by column in dataframe if it exists
        id_vars = ["mouse_name", "phase", "winMS"]
        if (
            split_by
            and split_by in self.results_df.columns
            or split_by in self.results_df.index.names
        ):
            id_vars.append(split_by)

        to_plot = pd.melt(
            self.results_df.xs(winMS, level="winMS", drop_level=False)
            .query("phase != 'full_pre'")
            .reset_index(),
            id_vars=id_vars,
            value_vars=result_cols,
            var_name=f"Error Type {dim}d",
            value_name=value_name,
        ).copy()

        # Determine plotting layout based on split_by
        if split_by and split_by in to_plot.columns:
            unique_splits = sorted(to_plot[split_by].dropna().unique())
            n_splits = len(unique_splits)
            fig, axes = plt.subplots(
                1, n_splits, figsize=(6 * n_splits, 9), sharey=True
            )
            if n_splits == 1:
                axes = [axes]
        else:
            unique_splits = [None]
            n_splits = 1
            fig, ax = plt.subplots(figsize=(16, 9))
            axes = [ax]

        # Loop through each subplot group
        for idx, split_val in enumerate(unique_splits):
            ax = axes[idx]

            if split_val is not None:
                sub_data = to_plot[to_plot[split_by] == split_val].copy()
                ax.set_title(
                    f"{split_by}: {split_val} (n={sub_data[sub_data['phase'] == phase_order[0]].shape[0] / len(result_cols):.0f} mice)",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                sub_data = to_plot.copy()

            if sub_data.empty:
                continue

            # Plot main Barplot
            sns.barplot(
                data=sub_data,
                x="phase",
                y=value_name,
                hue=f"Error Type {dim}d",
                palette="Set2",
                order=phase_order,
                ax=ax,
            )

            # Superimpose Stripplot
            sns.stripplot(
                data=sub_data,
                x="phase",
                y=value_name,
                hue=f"Error Type {dim}d",
                palette="Set2",
                order=phase_order,
                dodge=True,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
                ax=ax,
                marker="o",
                size=10,
            )

            # Statistical Annotations per subplot
            pairs = [
                ((phase, result_cols[0]), (phase, result_cols[1]))
                for phase in phase_order
            ]

            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=sub_data,
                    x="phase",
                    y=value_name,
                    hue=f"Error Type {dim}d",
                    order=phase_order,
                )
                annotator.configure(
                    test="Wilcoxon", text_format="star", loc="inside", verbose=False
                )
                annotator.apply_and_annotate()
            except Exception:
                # Fallback if a specific subset doesn't have matching pairs for Wilcoxon
                print(
                    f"Skipping stats for {split_val} due to insufficient paired data."
                )

            # Custom line connections per mouse
            connect_points(
                ax=ax,
                df=sub_data,
                x_col="phase",
                y_col=value_name,
                hue_col=f"Error Type {dim}d",
                id_col="mouse_name",
                x_order=phase_order,
            )

            # Handle axis labels cleanly across shared-Y subplots
            if idx > 0:
                ax.set_ylabel("")

            # Remove individual legends to avoid clutter; we will create one global legend
            if ax.get_legend():
                ax.get_legend().remove()

        # Build clean global legend from the last active axis
        all_handles, all_labels = axes[-1].get_legend_handles_labels()
        unique_labels, unique_handles = [], []
        for handle, label in zip(all_handles, all_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        fig.legend(
            handles=unique_handles,
            labels=unique_labels,
            loc="upper left",
            bbox_to_anchor=(0.98, 0.95),
        )

        fig.tight_layout()

        # Handle adaptive saving format
        if path is not None:
            split_suffix = f"_split_by_{split_by}" if split_by else ""
            filename_base = f"boxplot_{reduce}_lin_error_{thresh}_speed_{use_speed_filter}{split_suffix}_{winMS}_{dim}d"

            fig.savefig(
                os.path.join(path, f"{filename_base}.png"), dpi=300, bbox_inches="tight"
            )
            fig.savefig(os.path.join(path, f"{filename_base}.svg"), bbox_inches="tight")

        plt.show()
        return to_plot

    def compute_zone_classification_metrics(
        self, winMS: int, shock_threshold: float = 0.15, use_speed_filter: bool = True
    ):
        """Computes true vs. predicted zone classification metrics across all mice

        and phases, now including Balanced Accuracy and F1-Score.
        """
        # Filter for the specific window size
        df_subset = self.results_df.xs(winMS, level="winMS", drop_level=False).copy()

        classification_results = []

        for idx, row in df_subset.iterrows():
            mouse_name = idx[1]
            phase = idx[3]

            # Extract time-series arrays
            y_true_lin = np.array(row["linearTrue"])
            y_pred_lin = np.array(row["linearPred"])

            # Drop nan values if any exist in the alignment
            valid_mask = ~np.isnan(y_true_lin) & ~np.isnan(y_pred_lin)

            if use_speed_filter:
                speedMask = np.array(row["speedMask"])
                valid_mask &= speedMask

            if not np.any(valid_mask):
                continue

            y_true_lin = y_true_lin[valid_mask]
            y_pred_lin = y_pred_lin[valid_mask]

            # Determine binary states (In Shock Zone vs Not In Shock Zone)
            is_in_shock_true = y_true_lin <= shock_threshold
            is_in_shock_pred = y_pred_lin <= shock_threshold

            # --- Standard Metrics ---
            acc = accuracy_score(is_in_shock_true, is_in_shock_pred)
            class_error = 1.0 - acc

            # --- New Robust Metrics ---
            # Balanced Accuracy handles class imbalance by averaging recall on both classes
            balanced_acc = balanced_accuracy_score(is_in_shock_true, is_in_shock_pred)

            # F1-score is the harmonic mean of precision and recall (specifically for the Shock Zone class)
            f1 = f1_score(
                is_in_shock_true, is_in_shock_pred, pos_label=True, zero_division=0
            )

            # Confusion matrix elements for rates
            cm = confusion_matrix(
                is_in_shock_true, is_in_shock_pred, labels=[False, True]
            )
            tn, fp, fn, tp = cm.ravel()

            false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else np.nan
            miss_rate = fn / (tp + fn) if (tp + fn) > 0 else np.nan

            classification_results.append(
                {
                    "mouse_name": mouse_name,
                    "phase": phase,
                    "winMS": winMS,
                    "classification_error": class_error,
                    "balanced_accuracy": balanced_acc,
                    "f1_score": f1,
                    "false_alarm_rate": false_alarm_rate,
                    "miss_rate": miss_rate,
                    "total_timepoints": len(y_true_lin),
                }
            )

        return pd.DataFrame(classification_results)

    def plot_classification_error_barplot(self, winMS: int, path: Optional[str] = None):
        """Plots the overall classification error alongside false alarm and miss rates

        per phase per mouse.
        """
        from neuroencoders.importData.gui_elements import connect_points

        phase_order = ["training", "pre", "cond", "post"]

        # 1. Compute the metrics dataframe using the previously defined method
        metrics_df = self.compute_zone_classification_metrics(winMS=winMS)

        if metrics_df.empty:
            print(f"No valid data found for winMS={winMS}")
            return

        # 2. Melt the dataframe to make it seaborn-friendly
        # We want to compare overall error, false alarms, and missed detections side-by-side
        # 2. Melt the dataframe to include the new metrics
        value_vars = ["classification_error", "balanced_accuracy", "f1_score"]
        to_plot = pd.melt(
            metrics_df.query("phase != 'full_pre'"),
            id_vars=["mouse_name", "phase", "winMS"],
            value_vars=value_vars,
            var_name="Metric Type",
            value_name="Rate (0-1)",
        )

        # Clean up metric names for the legend
        metric_labels = {
            "classification_error": "Total Classification Error",
            "balanced_accuracy": "Balanced Accuracy",
            "f1_score": "F1-Score (Shock Zone)",
        }
        to_plot["Metric Type"] = to_plot["Metric Type"].map(metric_labels)

        # 3. Setup the plotting canvas
        fig, ax = plt.subplots(figsize=(16, 9))

        # Base barplot showing the mean performance per phase
        sns.barplot(
            data=to_plot,
            x="phase",
            y="Rate (0-1)",
            hue="Metric Type",
            palette="Set2",
            order=phase_order,
            ax=ax,
            edgecolor="black",
            linewidth=1,
        )

        # Overlay individual mouse points (stripplot) to see variance
        sns.stripplot(
            data=to_plot,
            x="phase",
            y="Rate (0-1)",
            hue="Metric Type",
            palette="Set2",
            order=phase_order,
            dodge=True,
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
            ax=ax,
            marker="o",
            size=10,
            legend=False,  # Avoid duplicating legend items from stripplot
        )

        # Connect individual mice paths across phases for each distinct metric type
        connect_points(
            ax=ax,
            df=to_plot,
            x_col="phase",
            y_col="Rate (0-1)",
            hue_col="Metric Type",
            id_col="mouse_name",
            x_order=phase_order,
        )

        # 4. Refine Aesthetics and Labels
        ax.set_title(
            f"Zone Classification Performance (Window: {winMS}ms)",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel("Rate (Proportion of Frames)", fontsize=14)
        ax.set_xlabel("Experimental Phase", fontsize=14)
        ax.set_ylim(-0.05, 1.05)  # Error rates are strictly bounded between 0 and 1

        # De-duplicate and position the legend cleanly outside the plot frame
        all_handles, all_labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(all_handles, all_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        # fig.legend(
        #     handles=unique_handles,
        #     labels=unique_labels,
        #     loc="upper left",
        #     bbox_to_anchor=(0.95, 0.95),
        #     title="Performance Metrics",
        # )

        fig.tight_layout()

        # 5. Save functionality
        if path is not None:
            os.makedirs(path, exist_ok=True)
            base_filename = f"classification_error_summary_{winMS}"
            fig.savefig(
                os.path.join(path, f"{base_filename}.png"),
                dpi=300,
            )
            fig.savefig(os.path.join(path, f"{base_filename}.svg"))

        plt.show()

    def compute_error_vs_distance_to_boundary(
        self,
        winMS: int,
        shock_threshold: float = 0.15,
        n_bins: int = 10,
        use_speed_filter: bool = True,
        manipe: Optional[str] = None,
    ):
        """
        Computes binary classification error binned by the true distance to the shock boundary.
        """
        df_subset = self.results_df.xs(winMS, level="winMS", drop_level=False).copy()
        if manipe is not None:
            df_subset = df_subset[df_subset.index.get_level_values("manipe") == manipe]

        # Track raw frame statistics across all mice/phases
        all_frames = []

        for idx, row in df_subset.iterrows():
            mouse_name = idx[1]
            phase = idx[3]

            y_true = np.array(row["linearTrue"])
            y_pred = np.array(row["linearPred"])

            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if use_speed_filter:
                valid_mask &= np.array(row["speedMask"])

            if not np.any(valid_mask):
                continue

            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

            # Calculate absolute distance to the decision boundary
            distance_to_boundary = y_true

            # Binary classifications
            is_in_shock_true = y_true <= shock_threshold
            is_in_shock_pred = y_pred <= shock_threshold

            # Was it misclassified? (Binary Error: True/False)
            is_misclassified = is_in_shock_true != is_in_shock_pred

            # Store every frame data point
            df_mouse_frames = pd.DataFrame(
                {
                    "mouse_name": mouse_name,
                    "phase": phase,
                    "distance_to_boundary": distance_to_boundary,
                    "is_misclassified": is_misclassified.astype(int),
                }
            )
            all_frames.append(df_mouse_frames)

        if not all_frames:
            return pd.DataFrame()

        total_frame_df = pd.concat(all_frames, ignore_index=True)

        # Define bins for the distance to boundary (max distance possible is max |x - threshold|)
        max_dist = total_frame_df["distance_to_boundary"].max()
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        bin_labels = [
            f"{np.round((bin_edges[i] + bin_edges[i + 1]) / 2, 2)}"
            for i in range(n_bins)
        ]

        # Assign each frame to a distance bin
        total_frame_df["Distance Bin Center"] = pd.cut(
            total_frame_df["distance_to_boundary"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )

        # Aggregate: Calculate mean classification error per bin, per phase, per mouse
        binned_results = (
            total_frame_df.groupby(
                ["mouse_name", "phase", "Distance Bin Center"], observed=False
            )["is_misclassified"]
            .mean()
            .reset_index()
        )

        binned_results.rename(
            columns={"is_misclassified": "Classification Error Rate"}, inplace=True
        )
        return binned_results

    def plot_error_vs_distance(
        self,
        winMS: int,
        shock_threshold: float = 0.15,
        n_bins: int = 25,
        use_speed_filter: bool = True,
        path: Optional[str] = None,
        manipe: Optional[str] = None,
    ):
        """
        Plots a line plot tracking how binary classification error rates drop
        as the animal moves further away from the shock decision boundary.
        """
        phase_order = ["training", "pre", "cond", "post"]

        # 1. Gather the binned data
        plot_df = self.compute_error_vs_distance_to_boundary(
            winMS=winMS,
            shock_threshold=shock_threshold,
            n_bins=n_bins,
            use_speed_filter=use_speed_filter,
            manipe=manipe,
        )

        if plot_df.empty:
            print("No valid tracking data found to compute distance relationships.")
            return

        # Filter out any auxiliary phases you don't track
        plot_df = plot_df[plot_df["phase"].isin(phase_order)].copy()

        # Ensure categorical order for distance bins on the X-axis
        plot_df["Distance Bin Center"] = pd.to_numeric(plot_df["Distance Bin Center"])

        fig, ax = plt.subplots(figsize=(12, 7))

        # 2. Draw lines with error bands across experimental phases
        # Using lineplot will aggregate across mice automatically and show confidence intervals
        sns.lineplot(
            data=plot_df,
            x="Distance Bin Center",
            y="Classification Error Rate",
            hue="phase",
            hue_order=phase_order,
            palette="Set2",
            marker="o",
            markersize=8,
            linewidth=2.5,
            ax=ax,
        )

        # 3. Aesthetics
        ax.set_title(
            f"Classification Error Rate vs. Distance to Shock Boundary (Window: {winMS}ms {'manipe: ' + manipe if manipe else ''})",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(
            "Absolute Distance to Shock Boundary (|True Position - Threshold|)",
            fontsize=12,
        )
        ax.set_ylabel("Classification Error Rate (Proportion)", fontsize=12)

        ax.set_ylim(
            -0.02, 0.55
        )  # Error rate maxes mathematically at 0.5 (pure chance) at the exact boundary
        ax.axhline(0.5, linestyle=":", color="red", alpha=0.5, label="Chance level")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.legend(
            title="Experimental Phase",
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )
        fig.tight_layout()

        # 4. Save Options
        if path is not None:
            os.makedirs(path, exist_ok=True)
            fig.savefig(
                os.path.join(path, f"error_vs_boundary_distance_{winMS}.png"), dpi=300
            )
            fig.savefig(os.path.join(path, f"error_vs_boundary_distance_{winMS}.svg"))

        plt.show()

    def _compute_spatial_overprediction(
        self, winMS, during, compute_type, input_type="2d"
    ):
        # Data collection list
        rows = []
        bins = 30

        if input_type.lower() == "2d":
            results_attr = "resultsNN_phase"
        elif input_type.lower() == "logits":
            results_attr = "resultsNN_phase_pkl"
        else:
            raise ValueError("Invalid input_type. Use '2d' or 'logits'.")

        # Loop through your existing dataframe structure
        for (mouse_manipe, manipe), df in self.results_df.xs(
            (winMS, "cond"), level=("winMS", "phase")
        ).groupby(by=["mouse_name", "manipe"]):
            results = df.iloc[0].results
            idWindow = results.timeWindows.index(winMS)

            time = getattr(results, results_attr)["_cond"]["times"][idWindow].flatten()
            if during == "ripples":
                events = results.DataHelper.get_ripples_epochs()
            elif during == "freezing":
                events = results.DataHelper.get_freeze_epochs()
            elif during == "stims":
                events = results.DataHelper.get_stim_epochs()
            else:
                raise ValueError(
                    "Invalid 'during' parameter. Use 'ripples', 'freezing', or 'stims'."
                )

            if compute_type == "percentage":
                true_pos2d = TsdFrame(
                    t=time,
                    d=getattr(results, results_attr)["_cond"]["featureTrue"][idWindow][
                        :, :2
                    ],
                    columns=["x", "y"],
                )
            elif compute_type == "overprediction":
                true_pos2d = TsdFrame(
                    t=time,
                    d=getattr(results, results_attr)["_cond"]["featureTrue"][idWindow][
                        :, :2
                    ],
                    columns=["x", "y"],
                ).restrict(events)

            if input_type.lower() == "2d":
                pred_pos2d = TsdFrame(
                    t=time,
                    d=getattr(results, results_attr)["_cond"]["featurePred"][idWindow][
                        :, :2
                    ],
                    columns=["x", "y"],
                ).restrict(events)
            elif input_type.lower() == "logits":
                pred_pos2d = TsdTensor(
                    t=time,
                    d=getattr(results, results_attr)["_cond"]["logits_hw"][idWindow],
                ).restrict(events)

            H_true, _, _ = np.histogram2d(
                true_pos2d.values[:, 0],
                true_pos2d.values[:, 1],
                bins=bins,
                range=[[0, 1], [0, 1]],
                density=True,
            )
            if input_type.lower() == "2d":
                H_pred, _, _ = np.histogram2d(
                    pred_pos2d.values[:, 0],
                    pred_pos2d.values[:, 1],
                    bins=bins,
                    range=[[0, 1], [0, 1]],
                    density=True,
                )
            elif input_type.lower() == "logits":
                # we already have a 2D matrix of logits, so we can just use it directly its mean across the time dimension
                H_pred = np.nanmean(pred_pos2d.values, axis=0)
                total_sum = np.nansum(H_pred)
                if total_sum > 0:
                    H_pred = (
                        H_pred / total_sum
                    )  # Normalize to make it a probability distribution

            smooth_true = gaussian_filter_nan(H_true, (1.5, 2.5))
            smooth_pred = gaussian_filter_nan(H_pred, (1.5, 2.5))
            baseline = smooth_true.copy()
            allowed = results.get_allowed_mask_for_bin_size(
                smooth_true.shape[0], smooth_true.shape[1]
            ).T
            smooth_true[~allowed] = np.nan
            smooth_pred[~allowed] = np.nan
            baseline[~allowed] = np.nan

            baseline = baseline / np.nansum(baseline)
            pred = smooth_pred / np.nansum(smooth_pred)
            eps = 1e-12

            if compute_type == "percentage":
                final = (pred - baseline) / (pred + baseline + eps)
            elif compute_type == "overprediction":
                final = pred - baseline
            else:
                raise ValueError(
                    "Invalid type specified. Use 'percentage' or 'overprediction'."
                )

            # --- NEW: Extract Group and Zone Data ---
            # Adjust this string check to match how your PAG vs Control mice are named
            group_label = manipe

            for i, label in enumerate(ZONELABELS):
                x_lim, y_lim = ZONEDEF[i]

                # Map continuous spatial coordinates [0, 1] to matrix bin indices
                x_start, x_end = int(x_lim[0] * bins), int(x_lim[1] * bins)
                y_start, y_end = int(y_lim[0] * bins), int(y_lim[1] * bins)

                # Calculate mean overprediction/reactivation metric for this zone
                mean_overpred = np.nanmean(final[x_start:x_end, y_start:y_end])

                rows.append(
                    {
                        "Mouse": mouse_manipe,
                        "Group": group_label,
                        "Zone": label,
                        "Overprediction": mean_overpred,
                    }
                )

        # Create the master long-form DataFrame
        df_zones = pd.DataFrame(rows)

        return df_zones

    def barplot_zones_prediction(
        self,
        winMS: int,
        during="ripples",
        compute_type="percentage",
        text_format="star",
        path: Optional[str] = None,
        interactive: bool = False,
    ):
        df_zones = self._compute_spatial_overprediction(
            winMS=winMS, compute_type=compute_type, during=during
        )

        group_order = sorted(df_zones["Group"].unique())
        zones = df_zones["Zone"].unique()

        # ----------------------------------------------------------------------
        # PATH A: INTERACTIVE PLOTLY PIPELINE
        # ----------------------------------------------------------------------
        if interactive:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create a 1-row, N-column layout dynamically matching your zones count
            fig = make_subplots(
                rows=1,
                cols=len(zones),
                subplot_titles=[f"<b>{z} Zone</b>" for z in zones],
                shared_yaxes=True,
                horizontal_spacing=0.03,
            )

            # Detect mouse identifier column safely
            mouse_col = "Mouse" if "Mouse" in df_zones.columns else df_zones.columns[0]

            for i, zone in enumerate(zones):
                col_idx = i + 1
                zone_data = df_zones[df_zones["Zone"] == zone]

                # Plot traces for each experimental group
                for group in group_order:
                    g_data = zone_data[zone_data["Group"] == group]
                    if g_data.empty:
                        continue

                    raw_color = GROUPS_PALETTE.get(group, "xkcd:gray")
                    g_color = mcolors.to_hex(raw_color)

                    # 1. Overlay Box plot underlying architecture
                    fig.add_trace(
                        go.Box(
                            y=g_data["Overprediction"],
                            name=group,
                            marker_color=g_color,
                            boxpoints=False,  # We use explicit jitter/stripplot points below
                            line=dict(width=2.5),
                            fillcolor=g_color,
                            opacity=0.5,
                            showlegend=(i == 0),  # Avoid legendary duplicate pollution
                        ),
                        row=1,
                        col=col_idx,
                    )

                    # 2. Add Jittered Scatter markers for Interactive Mouse auditing
                    hover_text = [
                        f"Mouse: {row[mouse_col]}<br>Group: {group}<br>Value: {row['Overprediction']:.3f}"
                        for _, row in g_data.iterrows()
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=[group]
                            * len(g_data),  # Aligns horizontally with the Box trace
                            y=g_data["Overprediction"],
                            mode="markers",
                            name=group,
                            text=hover_text,
                            hoverinfo="text",
                            marker=dict(
                                color=g_color,
                                size=8,
                                opacity=0.85,
                                line=dict(width=1, color="black"),
                            ),
                            showlegend=False,
                        ),
                        row=1,
                        col=col_idx,
                    )

                # Add baseline horizontal threshold trace marker at y=0
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(group_order) - 0.5,
                    y0=0,
                    y1=0,
                    line=dict(color="gray", width=1.5, dash="dash"),
                    row=1,
                    col=col_idx,
                )

            # Global Layout Customization
            fig.update_layout(
                title_text=f"Spatial Overprediction Bias during Ripples ({winMS} ms)",
                title_x=0.5,
                title_font=dict(size=16),
                template="plotly_white",
                height=600,
                width=300 * len(zones),
                yaxis_title="Δ of Overprediction (Δ=Pred - True)",
                showlegend=True,
                legend_title_text="Group",
            )

            # Clean up interactive tick angles across subplots
            for col_idx in range(1, len(zones) + 1):
                fig.update_xaxes(tickangle=45, row=1, col=col_idx)

            fig.show()

        # ----------------------------------------------------------------------
        # PATH B: STATIC MATPLOTLIB + SEABORN PIPELINE (ORIGINAL)
        # ----------------------------------------------------------------------
        else:
            from statannotations.Annotator import Annotator

            # Create a row or column of subplots (1 row, 5 columns)
            fig, axs = plt.subplots(1, len(zones), figsize=(20, 8), sharey=True)
            if len(zones) == 1:
                axs = [axs]

            for i, zone in enumerate(zones):
                ax = axs[i]
                zone_data = df_zones[df_zones["Zone"] == zone]

                # 1. Boxplot per zone
                sns.boxplot(
                    data=zone_data,
                    x="Group",
                    y="Overprediction",
                    order=group_order,
                    palette=GROUPS_PALETTE,
                    width=0.6,
                    fliersize=0,
                    boxprops=dict(alpha=0.6),
                    ax=ax,
                    linewidth=3,
                )

                # 2. Stripplot per zone
                sns.stripplot(
                    data=zone_data,
                    x="Group",
                    y="Overprediction",
                    order=group_order,
                    palette=GROUPS_PALETTE,
                    size=8,
                    jitter=0.2,
                    linewidth=1.5,
                    edgecolor="black",
                    ax=ax,
                    alpha=0.7,
                )

                # --- STATS PER ZONE ---
                other_groups = [g for g in group_order if g != "PAG"]
                box_pairs = [("PAG", other) for other in other_groups] + [
                    ("MFB", "Known")
                ]

                if len(zone_data) > 0:
                    try:
                        annotator = Annotator(
                            ax,
                            box_pairs,
                            data=zone_data,
                            x="Group",
                            y="Overprediction",
                            order=group_order,
                        )
                        annotator.configure(
                            test="t-test_ind",
                            text_format=text_format,
                            loc="inside",
                            comparisons_correction="Bonferroni",
                            hide_non_significant=True,
                        )
                        annotator.apply_and_annotate()
                    except Exception:
                        pass  # Handle edge cases gracefully if specific pairs are missing

                # Styling tweaks
                ax.set_title(f"{zone} Zone", fontsize=12, fontweight="bold")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                if i == 0:
                    ax.set_ylabel(r"$\Delta$ of Overprediction ($\Delta$=Pred - True)")
                else:
                    ax.set_ylabel("")

            plt.suptitle(
                f"Spatial Overprediction Bias during Ripples ({winMS} ms)",
                fontsize=16,
                y=1.05,
            )
            sns.despine()
            plt.tight_layout()

            if path is not None:
                os.makedirs(path, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        path,
                        f"bias_during_{during}_by_manipe_{text_format}_{compute_type}_{winMS}ms.png",
                    ),
                    bbox_inches="tight",
                )
            plt.show()

    def _compute_spatial_proportion(
        self, winMS: int, during: str, input_type: str = "2d"
    ):
        """Computes the proportion of discrete events (e.g., individual ripples)

        whose average neural predictions fall within each defined spatial zone.
        """

        rows = []

        if input_type.lower() == "2d":
            results_attr = "resultsNN_phase"
        elif input_type.lower() == "logits":
            results_attr = "resultsNN_phase_pkl"
        else:
            raise ValueError("Invalid input_type. Use '2d' or 'logits'.")

        # Loop through your existing dataframe structure grouped by mouse and manipulation
        for (mouse_manipe, manipe), df in self.results_df.xs(
            (winMS, "cond"), level=("winMS", "phase")
        ).groupby(by=["mouse_name", "manipe"]):
            results = df.iloc[0].results
            idWindow = results.timeWindows.index(winMS)

            time = getattr(results, results_attr)["_cond"]["times"][idWindow].flatten()

            # Isolate the targeted event epochs
            if during == "ripples":
                events = results.DataHelper.get_ripples_epochs()
            elif during == "freezing":
                events = results.DataHelper.get_freeze_epochs()
            elif during == "stims":
                events = results.DataHelper.get_stim_epochs()
            else:
                raise ValueError(
                    "Invalid 'during' parameter. Use 'ripples', 'freezing', or 'stims'."
                )

            # Extract full 2D predicted positions array
            pred_pos2d = TsdFrame(
                t=time,
                d=getattr(results, results_attr)["_cond"]["featurePred"][idWindow][
                    :, :2
                ],
                columns=["x", "y"],
            )

            events = events.intersect(pred_pos2d.time_support)
            total_events = len(events)

            # Initialize counts for each zone to 0 for this session
            zone_counts = {label: 0 for label in ZONELABELS}

            if total_events > 0:
                # Loop through each discrete interval (e.g., one specific ripple)
                for event in events:
                    # Extract positions decoded solely during this event
                    event_pred = pred_pos2d.restrict(event)

                    if len(event_pred) == 0:
                        continue

                    # Calculate the center of mass (mean trajectory) for this event
                    mean_x = np.nanmean(event_pred.values[:, 0])
                    mean_y = np.nanmean(event_pred.values[:, 1])

                    # Check which defined bounding box contains this mean prediction
                    for i, label in enumerate(ZONELABELS):
                        x_lim, y_lim = ZONEDEF[i]

                        if (x_lim[0] <= mean_x <= x_lim[1]) and (
                            y_lim[0] <= mean_y <= y_lim[1]
                        ):
                            zone_counts[label] += 1
                            break  # Assumes zones are mutually exclusive; remove if they overlap

            # Package statistics for long-form DataFrame conversion
            group_label = manipe
            for label in ZONELABELS:
                proportion = (
                    (zone_counts[label] / total_events) if total_events > 0 else 0.0
                )

                rows.append(
                    {
                        "Mouse": mouse_manipe,
                        "Group": group_label,
                        "Zone": label,
                        "Proportion": proportion,  # e.g., 0.50 means 50% of events landed here
                        "Total_Events": total_events,
                    }
                )

        # Build and return the master long-form DataFrame
        df_zones = pd.DataFrame(rows)
        return df_zones

    def barplot_zones_proportion_count(
        self,
        winMS: int,
        during: str = "ripples",
        text_format: str = "star",
        path: Optional[str] = None,
    ):
        """Generates a multi-panel boxplot comparing the event prediction proportions

        across zones and animal groups.
        """

        # Fetch data processed by the updated event-counting logic
        df_zones = self._compute_spatial_proportion(winMS=winMS, during=during)

        # Sort the manipulation groups consistently
        group_order = sorted(df_zones["Group"].unique())
        zones = df_zones["Zone"].unique()

        # Dynamically fit the subplot column amount based on unique zones
        fig, axs = plt.subplots(1, len(zones), figsize=(20, 8), sharey=True)

        # Catch instances where only 1 zone exists to avoid indexing errors
        if len(zones) == 1:
            axs = [axs]

        for i, zone in enumerate(zones):
            ax = axs[i]
            zone_data = df_zones[df_zones["Zone"] == zone]

            # 1. Boxplot distribution per group within this zone
            sns.boxplot(
                data=zone_data,
                x="Group",
                y="Proportion",
                order=group_order,
                palette=GROUPS_PALETTE,
                width=0.6,
                fliersize=0,
                boxprops=dict(alpha=0.6),
                ax=ax,
                linewidth=3,
            )

            # 2. Stripplot overlay to visualize individual mouse points
            sns.stripplot(
                data=zone_data,
                x="Group",
                y="Proportion",
                order=group_order,
                palette=GROUPS_PALETTE,
                size=8,
                jitter=0.2,
                linewidth=1.5,
                edgecolor="black",
                ax=ax,
                alpha=0.7,
            )

            # --- Statistical Testing Setup ---
            # Automatically parse groups to compare against your targeted "PAG" category
            other_groups = [g for g in group_order if g != "PAG"]
            box_pairs = [("PAG", other) for other in other_groups]

            # Adding your specific hardcoded comparison baseline if both items are present
            if "MFB" in group_order and "Known" in group_order:
                box_pairs.append(("MFB", "Known"))

            if len(zone_data) > 0 and len(group_order) > 1:
                annotator = Annotator(
                    ax,
                    box_pairs,
                    data=zone_data,
                    x="Group",
                    y="Proportion",
                    order=group_order,
                )
                annotator.configure(
                    test="t-test_ind",
                    text_format=text_format,
                    loc="inside",
                    comparisons_correction="Bonferroni",  # Corrects across your comparisons
                    hide_non_significant=True,
                )
                annotator.apply_and_annotate()

            # Individual subplot formatting tweaks
            ax.set_title(f"{zone} Zone", fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=11
            )

            # Clean up shared y labels and force scale to absolute bounds [0, 1]
            if i == 0:
                ax.set_ylabel("Proportion of Events", fontsize=12)
            else:
                ax.set_ylabel("")

        plt.suptitle(
            f"Proportion of Decoded Predictions per Spatial Zone during {during.capitalize()}",
            fontsize=16,
            y=1.02,
            fontweight="bold",
        )
        sns.despine()
        plt.tight_layout()

        # Handle exporting plots securely
        if path is not None:
            os.makedirs(path, exist_ok=True)
            filename = (
                f"event_proportion_during_{during}_by_manipe_{text_format}_{winMS}ms"
            )
            plt.savefig(
                os.path.join(path, filename + ".png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(path, filename + ".svg"),
            )

        plt.show()

    def compute_kudrimoti_variance(
        self,
        winMS=100,
        task_phase: str = "cond",
        subtask: Optional[str] = None,
        subsleep: Optional[str] = None,
    ):
        """Computes Explained Variance (EV) and Reverse Explained Variance (REV)

        based on the Kudrimoti et al. 1999 pairwise correlation design.
        """
        rows = []
        bin_size_sec = winMS / 1000.0

        # Grouping sessions via your dataframe loop architecture
        for (mouse_name, manipe), df in self.results_df.groupby(
            by=["mouse_name", "manipe"]
        ):
            results: Mouse_Results = df.iloc[0].results

            # 1. Fetch Spike trains (TsGroup)
            spike_group = results.DataHelper.get_spike_data()
            if len(spike_group) < 4:  # Minimum cells required for reliable matrix math
                continue

            try:
                pre_test, _ = results.get_epoch_interval("pre_test")
                pre_sleep, _ = results.get_epoch_interval("pre_sleep")
                task_epoch, _ = results.get_epoch_interval(task_phase)
                post_test, _ = results.get_epoch_interval("post_test")
                post_sleep, _ = results.get_epoch_interval("post_sleep")
            except Exception as e:
                # Fallback wrapper if explicitly designated within your helper setup
                print(
                    f"Could not retrieve epochs for {mouse_name} ({manipe}): {e}. Skipping session."
                )
                continue

            if subtask is not None:
                subtask = subtask.lower()
                try:
                    if "ripple" in subtask:
                        sub_intervals = results.DataHelper.get_ripples_epochs()
                    elif "freeze" in subtask:
                        sub_intervals = results.DataHelper.get_freeze_epochs()
                    elif "mov" in subtask:
                        sub_intervals = results.DataHelper.get_mov_epochs()
                    elif "sws" in subtask or "nrem" in subtask:
                        try:
                            sub_intervals = results.DataHelper.get_sws_epochs()
                        except FileNotFoundError:
                            sub_intervals = results.DataHelper.get_sws_epochs(
                                folder=results.network_path
                            )
                    elif "rem" in subtask:
                        try:
                            sub_intervals = results.DataHelper.get_rem_epochs()
                        except FileNotFoundError:
                            sub_intervals = results.DataHelper.get_rem_epochs(
                                folder=results.network_path
                            )
                    else:
                        raise ValueError(f"Unknown subtask: {subtask}")

                    task_epoch = task_epoch.intersect(sub_intervals)
                except AttributeError as e:
                    print(
                        f"Warning: DataHelper missing sub-epoch generator for '{subtask}': {e}. Using raw phase."
                    )

            if subsleep is not None:
                subsleep = subsleep.lower()
                try:
                    if "ripple" in subsleep:
                        subsleep_intervals = results.DataHelper.get_ripples_epochs()
                    elif "freeze" in subsleep:
                        subsleep_intervals = results.DataHelper.get_freeze_epochs()
                    elif "mov" in subsleep:
                        subsleep_intervals = results.DataHelper.get_mov_epochs()
                    elif "sws" in subsleep or "nrem" in subsleep:
                        try:
                            subsleep_intervals = results.DataHelper.get_sws_epochs()
                        except FileNotFoundError:
                            subsleep_intervals = results.DataHelper.get_sws_epochs(
                                folder=results.network_path
                            )
                    elif "rem" in subsleep:
                        try:
                            subsleep_intervals = results.DataHelper.get_rem_epochs()
                        except FileNotFoundError:
                            subsleep_intervals = results.DataHelper.get_rem_epochs(
                                folder=results.network_path
                            )
                    else:
                        raise ValueError(f"Unknown subsleep: {subsleep}")

                    pre_sleep = pre_sleep.intersect(subsleep_intervals)
                    post_sleep = post_sleep.intersect(subsleep_intervals)
                except AttributeError as e:
                    print(
                        f"Warning: DataHelper missing sub-epoch generator for '{subsleep}': {e}. Using raw phase."
                    )

            # 3. Bin spike data across the entire session to ensure shared structural alignments
            # Quantize neural spike times into uniform temporal bin counts (Q-Matrix)
            q_matrix = spike_group.count(bin_size_sec)

            # 4. Restrict Q-matrices to their respective behavioral phases
            q_pre_test = q_matrix.restrict(pre_test).values
            q_pre_sleep = q_matrix.restrict(pre_sleep).values
            q_task = q_matrix.restrict(task_epoch).values
            q_post_test = q_matrix.restrict(post_test).values
            q_post_sleep = q_matrix.restrict(post_sleep).values

            # Filter out completely silent cells within these segments to avoid NaN covariances
            active_cells = (
                (np.std(q_pre_test, axis=0) > 0)
                & (np.std(q_pre_sleep, axis=0) > 0)
                & (np.std(q_task, axis=0) > 0)
                & (np.std(q_post_test, axis=0) > 0)
                & (np.std(q_post_sleep, axis=0) > 0)
            )

            if np.sum(active_cells) < 4:
                continue

            q_pre_test = q_pre_test[:, active_cells]
            q_pre_sleep = q_pre_sleep[:, active_cells]
            q_task = q_task[:, active_cells]
            q_post_test = q_post_test[:, active_cells]
            q_post_sleep = q_post_sleep[:, active_cells]

            # 5. Compute Cell-by-Cell Pearson Correlation Matrices
            corr_pre = np.corrcoef(q_pre_sleep, rowvar=False)
            corr_task = np.corrcoef(q_task, rowvar=False)
            corr_post = np.corrcoef(q_post_sleep, rowvar=False)

            # 6. Extract Upper Triangular Indices (excluding the identity self-correlation diagonal)
            iu = np.triu_indices(corr_pre.shape[0], k=1)
            v_pre = corr_pre[iu]
            v_task = corr_task[iu]
            v_post = corr_post[iu]

            # Replace any residual interior NaNs with 0
            v_pre = np.nan_to_num(v_pre)
            v_task = np.nan_to_num(v_task)
            v_post = np.nan_to_num(v_post)

            # 7. Compute Inter-epoch Similarity Coefficients (R-values)
            r_task_pre = np.corrcoef(v_task, v_pre)[0, 1]
            r_task_post = np.corrcoef(v_task, v_post)[0, 1]
            r_post_pre = np.corrcoef(v_post, v_pre)[0, 1]

            # Handle potential correlation boundary edge issues
            eps = 1e-6
            denom_ev = np.sqrt((1 - r_task_pre**2) * (1 - r_post_pre**2))
            denom_rev = np.sqrt((1 - r_task_post**2) * (1 - r_post_pre**2))

            # 8. Compute final Partial Variances
            if denom_ev > eps and denom_rev > eps:
                ev_val = ((r_task_post - (r_task_pre * r_post_pre)) / denom_ev) ** 2
                rev_val = ((r_task_pre - (r_task_post * r_post_pre)) / denom_rev) ** 2
            else:
                ev_val, rev_val = np.nan, np.nan

            rows.append(
                {
                    "Mouse": mouse_name,
                    "Group": manipe,
                    "EV": ev_val * 100,
                    "REV": rev_val * 100,
                    "R_Task_PreTest": r_task_pre,
                    "R_Task_PostTest": r_task_post,
                    "R_Post_Pre": r_post_pre,
                    "MeanRate_PreTest": float(np.mean(q_pre_test))
                    if q_pre_test.size
                    else np.nan,
                    "MeanRate_Task": float(np.mean(q_task)) if q_task.size else np.nan,
                    "MeanRate_PostTest": float(np.mean(q_post_test))
                    if q_post_test.size
                    else np.nan,
                    "MeanRate_PreSleep": float(np.mean(q_pre_sleep))
                    if q_pre_sleep.size
                    else np.nan,
                    "MeanRate_PostSleep": float(np.mean(q_post_sleep))
                    if q_post_sleep.size
                    else np.nan,
                    "N_ActiveCells": int(np.sum(active_cells)),
                }
            )

        return pd.DataFrame(rows)

    def plot_kudrimoti_results(
        self,
        winMS=100,
        task_phase="cond",
        subtask: Optional[str] = None,
        subsleep: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        # Run the core computation
        df_results = self.compute_kudrimoti_variance(
            winMS=winMS, task_phase=task_phase, subtask=subtask, subsleep=subsleep
        )

        # Convert to long-form for seaborn grouping by Metric type (EV vs REV)
        df_melted = df_results.melt(
            id_vars=["Mouse", "Group"],
            value_vars=["EV", "REV"],
            var_name="Metric",
            value_name="Percentage",
        )

        unique_groups = df_melted["Group"].unique()
        fig, axs = plt.subplots(
            1, len(unique_groups), figsize=(5 * len(unique_groups), 6), sharey=True
        )

        if len(unique_groups) == 1:
            axs = [axs]

        for idx, group in enumerate(unique_groups):
            ax = axs[idx]
            group_data = df_melted[df_melted["Group"] == group]

            sns.boxplot(
                data=group_data,
                x="Metric",
                y="Percentage",
                palette="Set2",
                width=0.5,
                ax=ax,
                boxprops=dict(alpha=0.6),
            )
            sns.stripplot(
                data=group_data,
                x="Metric",
                y="Percentage",
                color="black",
                size=6,
                jitter=0.15,
                ax=ax,
            )

            # Paired Comparison test: EV vs REV within group
            box_pairs = [("EV", "REV")]
            annotator = Annotator(
                ax, box_pairs, data=group_data, x="Metric", y="Percentage"
            )
            annotator.configure(
                test="Wilcoxon", text_format="star", loc="inside"
            )  # Matching your MATLAB Wilcoxon choice
            annotator.apply_and_annotate()

            ax.set_title(f"Group: {group}")
            ax.set_xlabel("")
            if idx > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("% Variance Explained")

        plt.suptitle(
            f"Pairwise Coupling Reactivation ({task_phase.upper()} {subtask or ''} {subsleep or ''})\nWindow Size: {winMS} ms",
            fontsize=14,
            y=1.02,
        )
        sns.despine()
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"kudrimoti_reactivation_{task_phase}_{subtask or 'all'}_{subsleep or 'all'}_{winMS}ms"
            plt.savefig(
                os.path.join(save_path, filename + ".png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.savefig(
                os.path.join(save_path, filename + ".svg"),
            )
        plt.show()

    def compute_ev_behavior_correlation(
        self,
        winMS=100,
        task_phase="cond",
        subtask: Optional[str] = None,
        subsleep: Optional[str] = None,
        zone: str = "Shock",
        compute_type: str = "relative",
    ):
        """Extracts Kudrimoti EV metrics and correlates them with differences

        in Shock Zone (SZ) occupancy and first entry latency between Pre and Post tests.
        """

        rows = []

        # 1. Reuse our previously defined Kudrimoti function to collect raw EV metrics per session
        df_ev = self.compute_kudrimoti_variance(
            winMS=winMS, task_phase=task_phase, subtask=subtask, subsleep=subsleep
        )

        # 2. Gather behavioral parameters alongside neural metrics per session
        for idx, row in df_ev.iterrows():
            mouse_name = row["Mouse"]
            group_label = row["Group"]
            ev_val = row["EV"]

            if np.isnan(ev_val):
                continue

            # Isolate target dataset dataframe structures for specific session info
            df_session = self.results_df.xs(
                (mouse_name, group_label), level=("mouse_name", "manipe")
            )
            if df_session.empty:
                continue

            results: Mouse_Results = df_session.iloc[0].results

            try:
                occup_pre = results.DataHelper.get_zone_occupancy(
                    session_type="TestPre", zone=zone
                )
                occup_post = results.DataHelper.get_zone_occupancy(
                    session_type="TestPost", zone=zone
                )

                latency_pre = results.DataHelper.get_first_entry_latency(
                    session_type="TestPre",
                    zone=zone,
                    max_duration=500,
                )
                latency_post = results.DataHelper.get_first_entry_latency(
                    session_type="TestPost", zone=zone, max_duration=500
                )

                pos_xy = results.DataHelper.old_positions[:, :2]
                pos_pre_test = pos_xy[results.full_preMask, :2]
                pos_cond_test = pos_xy[results.condMask, :2]
                pos_post_test = pos_xy[results.postMask, :2]

                thigmo_pre = results.DataHelper.compute_distance_weighted_thigmo(
                    pos_pre_test
                )
                thigmo_cond = results.DataHelper.compute_distance_weighted_thigmo(
                    pos_cond_test
                )
                thigmo_post = results.DataHelper.compute_distance_weighted_thigmo(
                    pos_post_test
                )

                diff_latency = latency_post - latency_pre

                if compute_type == "relative":
                    diff_occup = ((occup_post - occup_pre) / occup_pre) * 100.0
                    diff_thigmo_post = (
                        (thigmo_post - thigmo_cond) / thigmo_cond
                    ) * 100.0
                    diff_thigmo_cond = ((thigmo_cond - thigmo_pre) / thigmo_pre) * 100.0
                elif compute_type == "absolute":
                    diff_occup = occup_post - occup_pre
                    diff_thigmo_post = thigmo_post - thigmo_cond
                    diff_thigmo_cond = thigmo_cond - thigmo_pre
                else:
                    raise ValueError(
                        "Invalid type specified. Use 'relative' or 'absolute'."
                    )

                rows.append(
                    {
                        "Mouse": mouse_name,
                        "Group": group_label,
                        "EV": ev_val,
                        "Delta_Occupancy": diff_occup,
                        "Delta_Latency": diff_latency,
                        "Delta_Thigmotaxis_Cond": diff_thigmo_cond,
                        "Delta_Thigmotaxis_Post": diff_thigmo_post,
                    }
                )
            except ValueError as e:
                # Fallback if your helper uses direct properties or raw tracking frames
                print(
                    f"Could not compute behavioral deltas for {mouse_name} ({group_label}): {e}. Skipping session."
                )
                continue

        return pd.DataFrame(rows)

    def inspect_mouse_trajectories(
        self, zone: str = "Shock", max_duration=2000, save_path: Optional[str] = None
    ):
        """Generates trajectory plots for each mouse across TestPre and TestPost sessions.
        Includes a localized sequential colormap showing ±10 frames around the first entry
        to distinguish real behavioral entries from tracking artifacts.
        """
        import matplotlib.patches as patches

        try:
            zone_idx = ZONELABELS.index(zone)
            x_lim, y_lim = ZONEDEF[zone_idx]
        except NameError:
            print(
                "Error: ZONELABELS or ZONEDEF parameters are not accessible within scope."
            )
            return

        for (mouse_name, group_label), df_session in self.results_df.groupby(
            level=["mouse_name", "manipe"]
        ):
            if df_session.empty:
                continue

            results: Mouse_Results = df_session.iloc[0].results
            data_helper = results.DataHelper

            # Setup subplots for side-by-side session inspection (Pre vs Post)
            fig, axs = plt.subplots(1, 3, figsize=(15, 6.5), sharex=True, sharey=True)
            sessions = ["TestPre", "Cond", "TestPost"]

            print(
                f"Generating diagnostic trajectory verification for mouse: {mouse_name}..."
            )

            # Keep track of session-specific metrics to compute overall Delta at the end
            session_metrics = {
                "TestPre": {"occ": np.nan, "lat": np.nan, "thigmo": np.nan},
                "Cond": {"occ": np.nan, "lat": np.nan, "thigmo": np.nan},
                "TestPost": {"occ": np.nan, "lat": np.nan, "thigmo": np.nan},
            }

            for col_idx, session_type in enumerate(sessions):
                ax = axs[col_idx]

                # --- Exact replication of underlying slicing logic ---
                session_names = data_helper.fullBehavior["Times"].get(
                    "sessionNames", []
                )
                starts = data_helper.fullBehavior["Times"].get("sessionStart", [])
                stops = data_helper.fullBehavior["Times"].get("sessionStop", [])

                session_id = [
                    idx
                    for idx, name in enumerate(session_names)
                    if session_type.lower() in name.lower()
                ]
                if not session_id:
                    ax.text(
                        0.5,
                        0.5,
                        "Session Not Found",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                full_latency = []
                full_occupancy = []
                full_thigmotaxis = []
                for i, sess_id in enumerate(session_id):
                    if i >= 2:
                        continue

                    session_interval = IntervalSet(
                        start=starts[sess_id], end=stops[sess_id]
                    )
                    pos_tsd = TsdFrame(
                        t=data_helper.fullBehavior["positionTime"].flatten(),
                        d=data_helper.fullBehavior["Positions"],
                    ).restrict(session_interval)

                    my_data = pos_tsd.values
                    my_data = my_data[~np.isnan(my_data).any(axis=1)]  # Remove NaNs
                    ax.scatter(
                        my_data[0, 0],
                        my_data[0, 1],
                        color="green",
                        s=50,
                    )

                    if len(pos_tsd) == 0:
                        ax.text(
                            0.5,
                            0.5,
                            "Empty Position Data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        continue

                    x_coords = pos_tsd.values[:, 0]
                    y_coords = pos_tsd.values[:, 1]
                    t_coords = pos_tsd.times()

                    # Calculate mask vectors for global visualization
                    in_zone_mask = (
                        (x_coords >= x_lim[0])
                        & (x_coords <= x_lim[1])
                        & (y_coords >= y_lim[0])
                        & (y_coords <= y_lim[1])
                    )

                    # Fetch computed scalars using your helper functions
                    try:
                        computed_occup = (
                            data_helper.get_zone_occupancy(
                                session_type=session_type, zone=zone
                            )
                            * 100.0
                        )
                        computed_latency = data_helper.get_first_entry_latency(
                            session_type=session_type,
                            zone=zone,
                            max_duration=max_duration,
                        )
                        computed_thigmo = data_helper.compute_distance_weighted_thigmo(
                            pos_tsd.values[:, :2]
                        )
                        full_occupancy.append(computed_occup)
                        full_latency.append(computed_latency)
                        full_thigmotaxis.append(computed_thigmo)
                    except Exception:
                        computed_occup, computed_latency = np.nan, np.nan

                    # 1. Plot entire session path background (lightgray)
                    ax.plot(
                        x_coords,
                        y_coords,
                        color="lightgray",
                        alpha=0.5,
                        linewidth=1,
                        # label="Full Path",
                    )

                    # 2. Highlight all points detected inside the zone natively (crimson dots)
                    ax.scatter(
                        x_coords[in_zone_mask],
                        y_coords[in_zone_mask],
                        color="crimson",
                        s=3,
                        alpha=0.2,
                        #     label="In-Zone",
                    )

                    # 3. --- TIME-WINDOW CMAP AROUND FIRST ENTRY ---
                    # Replicate the drop_short_intervals(1.0) logic to find the exact true entry frame
                    in_zone_intervalset = (
                        Tsd(t=t_coords, d=in_zone_mask.astype(int))
                        .threshold(0.5, "above")
                        .time_support
                    )
                    in_zone_intervalset = in_zone_intervalset.drop_short_intervals(1.0)

                    if in_zone_intervalset.tot_length() > 1:
                        first_entry_time = in_zone_intervalset.as_units("s").start[0]
                        # Find closest index matching this specific timestamps
                        entry_idx = np.argmin(
                            np.abs(pos_tsd.times("s") - first_entry_time)
                        )

                        # Window slice bounds: safely bounded by array limits
                        start_w = max(0, entry_idx - 10)
                        end_w = min(len(pos_tsd), entry_idx + 11)

                        x_window = x_coords[start_w:end_w]
                        y_window = y_coords[start_w:end_w]
                        time_steps = np.arange(
                            start_w - entry_idx, end_w - entry_idx
                        )  # Relative to entry (0)

                        # Plot window path as a connected dark line to track chronological vector direction
                        ax.plot(
                            x_window,
                            y_window,
                            color="dimgray",
                            linewidth=1.5,
                            linestyle="-",
                            alpha=0.8,
                        )

                        # Scatter plot colored by relative step index (-10 to +10)
                        sc = ax.scatter(
                            x_window,
                            y_window,
                            c=time_steps,
                            cmap="viridis",
                            s=50,
                            edgecolor="black",
                            linewidth=0.6,
                            zorder=5,
                            # label="Entry Window",
                        )

                        if sess_id == session_id[-1]:
                            cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
                            cbar.set_label("Relative Frame (0=Entry)", fontsize=9)
                            cbar.ax.tick_params(labelsize=8)

                computed_occup = (
                    np.nanmedian(full_occupancy) if full_occupancy else np.nan
                )
                session_metrics[session_type]["occ"] = computed_occup
                computed_latency = (
                    np.nanmedian(full_latency) if full_latency else np.nan
                )
                session_metrics[session_type]["lat"] = computed_latency
                computed_thigmo = (
                    np.nanmedian(full_thigmotaxis) if full_thigmotaxis else np.nan
                )
                session_metrics[session_type]["thigmo"] = computed_thigmo
                # 4. Draw explicit ZONEDEF bounding box patch
                zone_width = x_lim[1] - x_lim[0]
                zone_height = y_lim[1] - y_lim[0]
                rect = patches.Rectangle(
                    (x_lim[0], y_lim[0]),
                    zone_width,
                    zone_height,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="blue",
                    alpha=0.08,
                    linestyle="--",
                )
                ax.add_patch(rect)

                # Axis layout titles
                ax.set_title(
                    f"{session_type}\nOccupancy: {computed_occup:.2f}%\nLatency: {f'{computed_latency:.1f}s' if not np.isnan(computed_latency) else 'None'}\nThigmotaxis: {computed_thigmo:.3f}",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel("X Tracking Coordinates")
                if col_idx == 0:
                    ax.set_ylabel("Y Tracking Coordinates")
                    ax.legend(loc="lower left", fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.5)

            # Compute delta evaluations for output verification banner
            d_occ = (
                100
                * (
                    session_metrics["TestPost"]["occ"]
                    - session_metrics["TestPre"]["occ"]
                )
                / session_metrics["TestPre"]["occ"]
            )
            d_lat = (
                session_metrics["TestPost"]["lat"] - session_metrics["TestPre"]["lat"]
            )

            plt.suptitle(
                f"Spatial Trajectory Verification & Entry Entry Audit\n"
                f"Mouse: {mouse_name} | Group: {group_label}\n"
                f"Delta Occupancy: {d_occ:+.2f}% | Delta Latency: {d_lat:+.1f}s",
                fontsize=13,
                fontweight="bold",
                y=1.03,
            )
            plt.tight_layout()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                filename = f"trajectory_audit_{mouse_name}_{group_label}_{zone}_{max_duration}s"
                plt.savefig(
                    os.path.join(save_path, filename + ".png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()

    def plot_ev_behavior_correlation(
        self,
        winMS=100,
        task_phase="cond",
        zone: str = "Shock",
        subtask: Optional[str] = None,
        subsleep: Optional[str] = None,
        compute_type: str = "relative",
        model_name: str = "TheilSen",  # or Huber
        save_path=None,
    ):
        """Plots individual correlation lines per Group/manipe to observe differential

        impacts of manipulations on the EV vs Behavior relationship.
        """
        from scipy.stats import spearmanr

        df_corr = self.compute_ev_behavior_correlation(
            winMS=winMS,
            task_phase=task_phase,
            zone=zone,
            subtask=subtask,
            subsleep=subsleep,
            compute_type=compute_type,
        )

        if df_corr.empty or len(df_corr) < 3:
            print("Insufficient paired data found to calculate trends.")
            return

        # Initialize subplots
        fig, axs = plt.subplots(1, 4, figsize=(21, 7))

        # Get unique experimental manipulations (groups)
        unique_groups = df_corr["Group"].unique()

        # Define metrics to plot on the Y axes
        y_metrics = [
            "Delta_Occupancy",
            "Delta_Latency",
            "Delta_Thigmotaxis_Cond",
            "Delta_Thigmotaxis_Post",
        ]
        y_labels = [
            rf"{compute_type.capitalize()} $\Delta$ {zone.capitalize()} Zone Occupancy {'(Post - Pre %)' if compute_type == 'absolute' else '((Post - Pre) / Pre * 100%)'}",
            rf"$\Delta$ {zone.capitalize()} Entry Latency (Post - Pre sec)",
            r"$\Delta$ Thigmotaxis Cond (Cond - Pre)",
            r"$\Delta$ Thigmotaxis Post (Post - Cond)",
        ]

        # Loop over both behavioral metric axes
        for col_idx, metric in enumerate(y_metrics):
            ax = axs[col_idx]

            # 1. Base scatter plot colored cleanly by group using your palette
            sns.scatterplot(
                data=df_corr,
                x="EV",
                y=metric,
                hue="Group",
                palette=GROUPS_PALETTE,
                s=140,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.85,
                ax=ax,
                legend=(col_idx == 0),  # Only draw legend on first plot
            )

            mouse_col = "Mouse" if "Mouse" in df_corr.columns else df_corr.columns[0]

            y_range = df_corr[metric].max() - df_corr[metric].min()
            y_offset = y_range * 0.02 if y_range > 0 else 0.1

            for _, row in df_corr.iterrows():
                ax.text(
                    x=row["EV"],
                    y=row[metric] + y_offset,
                    s=str(row[mouse_col]),
                    fontsize=8,
                    color="black",
                    alpha=0.75,
                    ha="center",
                    va="bottom",
                )
            # --------------------------------------

            text_box_lines = []

            from sklearn.linear_model import HuberRegressor, TheilSenRegressor

            for group in unique_groups:
                group_df = df_corr[df_corr["Group"] == group]

                # Check if group has enough sample variance to calculate a correlation line
                if len(group_df) < 3:
                    continue

                x_g = group_df["EV"].values
                y_g = group_df[metric].values

                rho, p_val = spearmanr(x_g, y_g)
                g_color = GROUPS_PALETTE.get(group, "black")

                X_fit = x_g.reshape(-1, 1)
                if model_name == "Huber":
                    model = HuberRegressor()
                elif model_name == "TheilSen":
                    model = TheilSenRegressor()
                else:
                    raise ValueError("Invalid model_name. Use 'Huber' or 'TheilSen'.")

                model.fit(X_fit, y_g)
                x_vals = np.linspace(x_g.min(), x_g.max(), 100).reshape(-1, 1)
                y_pred = model.predict(x_vals)

                ax.plot(
                    x_vals.flatten(),
                    y_pred,
                    color=g_color,
                    linewidth=2.5,
                )

                # Construct clean string for the statistics display box
                text_box_lines.append(f"{group}: $\\rho$ = {rho:.3f} (p = {p_val:.3f})")

            # Formatting labels and display markers
            ax.set_xlabel("Explained Variance (EV %)", fontsize=13, fontweight="bold")
            ax.set_ylabel(y_labels[col_idx], fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=11)

            # Add descriptive statistical box in the corner
            ax.text(
                0.05,
                0.95,
                "\n".join(text_box_lines),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                ),
            )

        # Put legend safely outside or standard internal positioning
        axs[0].legend(
            title="Experimental Group", title_fontsize=11, fontsize=10, loc="lower left"
        )

        plt.suptitle(
            f"Relationship between neural reactivations and {zone} behaviour (Phase: {task_phase.upper()} {subtask if subtask else ''} {subsleep if subsleep else ''})",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        sns.despine()
        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"EV_behavior_group_corr_{task_phase}{'_' + subtask if subtask else ''}{'_' + subsleep if subsleep else ''}"
            fig.savefig(
                os.path.join(save_path, filename + ".png"),
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(save_path, filename + ".svg"),
            )
        plt.show()

    def compute_pca_reactivation(
        self, winMS=100, template_period="cond", num_templates=2
    ):
        """Computes population reactivation strength (Peyrache et al. 2010 style)

        using PCA template projection across behavioral blocks.
        """
        bin_size_sec = winMS / 1000.0
        all_session_data = {}

        for (mouse_name, manipe), df in self.results_df.groupby(
            by=["mouse_name", "manipe"]
        ):
            results: Mouse_Results = df.iloc[0].results

            # 1. Fetch Spike Trains
            spike_group = results.DataHelper.get_spike_data()
            if len(spike_group) < 5:
                continue

            # 2. Extract Macro Phase Intervals
            pre_epoch, _ = results.get_epoch_interval("pre_test")
            pre_sleep, _ = results.get_epoch_interval("pre_sleep")
            cond_epoch, _ = results.get_epoch_interval("cond")
            post_epoch, _ = results.get_epoch_interval("post_test")
            post_sleep, _ = results.get_epoch_interval("post_sleep")
            hab_epoch, _ = results.get_epoch_interval("hab")

            # Secondary physiological sub-epoch layers
            sws_epochs = results.DataHelper.get_sws_epochs()
            rem_epochs = results.DataHelper.get_rem_epochs()
            ripples_epochs = results.DataHelper.get_ripples_epochs()
            mov_epochs = results.DataHelper.get_mov_epochs()

            try:
                freeze_epochs = results.DataHelper.get_freeze_epochs()
            except AttributeError:
                freeze_epochs = IntervalSet(start=[], end=[])

            try:
                stim_epochs = results.DataHelper.get_stim_epochs(before=0.1, after=0.1)
            except AttributeError:
                stim_epochs = IntervalSet(start=[], end=[])

            # 3. Bin full session spike activity (Q-Matrix)
            q_tsd = spike_group.count(bin_size_sec)

            # 4. Map the requested Template Interval Configuration
            if template_period == "wake":
                template_interval = (
                    pre_epoch.union(hab_epoch).union(cond_epoch).union(post_epoch)
                )  # Full task / conditioning wake block
            elif template_period == "cond":
                template_interval = cond_epoch
            elif template_period == "condMov":
                template_interval = cond_epoch.intersect(mov_epochs)
            elif template_period == "condFree":
                template_interval = cond_epoch.intersect(freeze_epochs)
            elif template_period == "postRip":
                template_interval = post_epoch.intersect(sws_epochs).intersect(
                    ripples_epochs
                )
            elif template_period == "condRip":
                template_interval = cond_epoch.intersect(ripples_epochs)
            else:
                raise ValueError(f"Unknown template period paradigm: {template_period}")

            # 5. Build PCA Template via Covariance Matrix
            q_template = q_tsd.restrict(template_interval).values
            if q_template.shape[0] < 10:
                continue

            # Compute Pearson Correlation Matrix of the population template
            corr_matrix = np.corrcoef(q_template, rowvar=False)
            corr_matrix = np.nan_to_num(corr_matrix)
            np.fill_diagonal(corr_matrix, 0)  # Clear out self-correlation diagonals

            # Singular Value Decomposition / Eigendecomposition wrapper
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

            # Sort in descending order to match MATLAB's pcacov outputs
            idx_sorted = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx_sorted]
            eigenvalues = eigenvalues[idx_sorted]

            # 6. Project templates onto full session frames to calculate Reactivation Strength (RS)
            q_full_normalized = (q_tsd.values - np.mean(q_tsd.values, axis=0)) / (
                np.std(q_tsd.values, axis=0) + 1e-12
            )
            q_full_normalized = np.nan_to_num(q_full_normalized)

            pc_scores = {}
            rs_templates = {}
            n_comp = min(num_templates, eigenvectors.shape[1])

            for idx_t in range(n_comp):
                v_i = eigenvectors[:, idx_t]

                # Peyrache Formulation: Score^2 minus single neuron contributions
                score_t = np.dot(q_full_normalized, v_i)
                pc_scores[idx_t] = Tsd(t=q_tsd.index, d=score_t)

                single_neuron_contribution = np.dot(q_full_normalized**2, v_i**2)
                rs_vector = (score_t**2) - single_neuron_contribution
                rs_templates[idx_t] = Tsd(t=q_tsd.index, d=rs_vector)

            # 7. Collect structural metadata dictionary package for this session slice
            event_intervals = {
                "cond_ripples": cond_epoch.intersect(ripples_epochs),
                "cond_no_ripples": cond_epoch.set_diff(ripples_epochs),
                "cond_freeze": cond_epoch.intersect(freeze_epochs),
                "cond_move": cond_epoch.intersect(mov_epochs),
                "cond_stim": cond_epoch.intersect(stim_epochs),
                "pre_sleep_sws": pre_sleep.intersect(sws_epochs),
                "post_sleep_sws": post_sleep.intersect(sws_epochs),
                "pre_sleep_rem": pre_sleep.intersect(rem_epochs),
                "post_sleep_rem": post_sleep.intersect(rem_epochs),
            }
            template_summaries = {}
            for t_idx, rs_tsd in rs_templates.items():
                template_summaries[t_idx] = _summarize_reactivation_strengths(
                    rs_tsd,
                    {
                        "pre_test": pre_epoch,
                        "pre_sleep": pre_sleep,
                        "cond": cond_epoch,
                        "post_test": post_epoch,
                        "post_sleep": post_sleep,
                        "hab": hab_epoch,
                    },
                    event_intervals,
                )

            all_session_data[f"{mouse_name}_{manipe}"] = {
                "rs": rs_templates,
                "pc_scores": pc_scores,
                "eigenvectors": eigenvectors[:, :n_comp],
                "eigenvalues": eigenvalues[:n_comp],
                "cell_ids": list(spike_group.keys()),
                "q_tsd": q_tsd,
                "summaries": template_summaries,
                "epochs": {
                    "pre_test": pre_epoch,
                    "pre_sleep": pre_sleep,
                    "pre_sleep_sws": pre_sleep.intersect(sws_epochs),
                    "pre": pre_epoch,
                    "hab": hab_epoch,
                    "cond": cond_epoch,
                    "cond_freeze": cond_epoch.intersect(freeze_epochs),
                    "cond_ripples": cond_epoch.intersect(ripples_epochs),
                    "cond_no_ripples": cond_epoch.set_diff(ripples_epochs),
                    "cond_move": cond_epoch.intersect(mov_epochs),
                    "cond_stim": cond_epoch.intersect(stim_epochs),
                    "post_test": post_epoch,
                    "post": post_epoch,
                    "post_sleep": post_sleep,
                    "post_sleep_sws": post_sleep.intersect(sws_epochs),
                    "ripples": ripples_epochs,
                    "sws": sws_epochs,
                    "rem": rem_epochs,
                    "mov": mov_epochs,
                    "freeze": freeze_epochs,
                    "pre_sleep_rem": pre_sleep.intersect(rem_epochs),
                    "post_sleep_rem": post_sleep.intersect(rem_epochs),
                },
                "positions": {
                    "x": results.DataHelper.fullBehavior["Positions"][:, 0],
                    "y": results.DataHelper.fullBehavior["Positions"][:, 1],
                    "time": results.DataHelper.fullBehavior["positionTime"].flatten(),
                    "linear": results.l_function(
                        results.DataHelper.fullBehavior["Positions"][:, :2],
                    )[1].flatten(),
                },
                "results": results,
            }

        return all_session_data

    def map_reactivation_space(self, session_dict, template_idx=0, bins=10):
        """Bins continuous 2D position space to map mean reactivation distribution profiles."""
        x_pos = session_dict["positions"]["x"]
        y_pos = session_dict["positions"]["y"]
        t_pos = session_dict["positions"]["time"]
        rs_tsd = session_dict["rs"][template_idx]

        pos_tsd = TsdFrame(t=t_pos, d=np.vstack([x_pos, y_pos]).T, columns=["x", "y"])

        # Grid allocation steps
        edges = np.linspace(0, 1, bins + 1)
        hab_map = np.zeros((bins, bins))
        cond_map = np.zeros((bins, bins))
        pre_epoch = session_dict["epochs"].get(
            "pre_test", session_dict["epochs"].get("pre")
        )

        for i in range(bins):
            for j in range(bins):
                # Spatial masking
                spatial_mask = (
                    (pos_tsd.values[:, 0] >= edges[i])
                    & (pos_tsd.values[:, 0] < edges[i + 1])
                    & (pos_tsd.values[:, 1] >= edges[j])
                    & (pos_tsd.values[:, 1] < edges[j + 1])
                )
                if not np.any(spatial_mask):
                    continue

                intervals_xy = (
                    Tsd(t=t_pos, d=spatial_mask.astype(bool))
                    .threshold(0.5, "above")
                    .time_support
                )

                # Extract mean RS values targeting this pixel coordinates frame
                hab_map[i, j] = np.nanmean(
                    rs_tsd.restrict(pre_epoch.intersect(intervals_xy)).values
                )
                cond_map[i, j] = np.nanmean(
                    rs_tsd.restrict(
                        session_dict["epochs"]["cond"].intersect(intervals_xy)
                    ).values
                )

        # Apply standard Gaussian smoothing matching MATLAB's smooth2a logic
        hab_map = gaussian_filter(np.nan_to_num(hab_map), sigma=0.8)
        cond_map = gaussian_filter(np.nan_to_num(cond_map), sigma=0.8)

        return hab_map, cond_map

    def plot_pca_master_results(
        self,
        winMS=100,
        template_period="cond",
        num_templates=1,
        save_path=None,
        session_data: Optional[dict] = None,
        spike_data: bool = True,
    ):
        """Executes computation loops and draws macro bar plots alongside 2D tracking matrices."""

        if session_data is None:
            if spike_data:
                session_data = self.compute_pca_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=num_templates,
                )
            else:
                session_data = self.compute_latent_pca_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=num_templates,
                )

        if not session_data:
            print("No valid session matrices computed.")
            return

        unique_manipes = self.results_df.index.get_level_values("manipe").unique()

        for manipe in unique_manipes:
            bar_rows = []
            spatial_hab_list, spatial_cond_list = [], []
            print(f"looking at {manipe}")
            # Filter data for the current manipe
            # Loop to parse scalar summary statistics for category comparisons
            for s_key, data in session_data.items():
                if manipe.lower() not in s_key.lower():
                    continue

                print(f"adding {s_key} to summary")

                for t_idx in range(num_templates):
                    rs_tsd = data["rs"][t_idx]
                    epochs = data["epochs"]
                    summary = data.get("summaries", {}).get(t_idx, {})

                    m_pre = np.nanmean(rs_tsd.restrict(epochs["pre_sleep_sws"]).values)
                    m_hab = np.nanmean(rs_tsd.restrict(epochs["pre_test"]).values)
                    m_cond = np.nanmean(rs_tsd.restrict(epochs["cond"]).values)
                    m_post = np.nanmean(
                        rs_tsd.restrict(epochs["post_sleep_sws"]).values
                    )

                    bar_rows.append(
                        {
                            "Session": s_key,
                            "Epoch": "PreSleep",
                            "Score": m_pre,
                            "Template": t_idx,
                        }
                    )
                    bar_rows.append(
                        {
                            "Session": s_key,
                            "Epoch": "FreeExplo",
                            "Score": m_hab,
                            "Template": t_idx,
                        }
                    )
                    bar_rows.append(
                        {
                            "Session": s_key,
                            "Epoch": "Learning",
                            "Score": m_cond,
                            "Template": t_idx,
                        }
                    )
                    bar_rows.append(
                        {
                            "Session": s_key,
                            "Epoch": "PostSleep",
                            "Score": m_post,
                            "Template": t_idx,
                        }
                    )
                    if summary:
                        for row in bar_rows[-4:]:
                            row.update(
                                {
                                    k: v
                                    for k, v in summary.items()
                                    if k
                                    in {
                                        "cond_minus_pre_test",
                                        "cond_minus_post_test",
                                        "sleep_delta",
                                        "cond_ripples",
                                        "cond_freeze",
                                        "cond_stim",
                                        "cond_move",
                                        "cond_no_ripples",
                                    }
                                }
                            )

                    # Map spatial elements
                    h_map, c_map = self.map_reactivation_space(
                        data, template_idx=t_idx, bins=20
                    )
                    spatial_hab_list.append(h_map)
                    spatial_cond_list.append(c_map)

            df_bars = pd.DataFrame(bar_rows)

            # Initialize multi-panel visualization framework
            fig = plt.figure(figsize=(18, 5))
            gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1])

            # Panel 1: Main Population Trend Bars
            ax0 = fig.add_subplot(gs[0])
            sns.barplot(
                data=df_bars,
                x="Epoch",
                y="Score",
                ax=ax0,
                palette=["#7F7F7F", "#C9E62F", "#E60000", "#000000"],
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
                errorbar="se",
            )
            sns.stripplot(
                data=df_bars,
                x="Epoch",
                y="Score",
                ax=ax0,
                color="black",
                alpha=0.6,
                jitter=0.15,
                size=5,
            )
            ax0.set_ylabel("PC Reactivation Score", fontweight="bold")
            ax0.set_xlabel("")
            ax0.set_title("Global Assembly Reactivation Profile")
            pairs = [
                ("PreSleep", "FreeExplo"),
                ("FreeExplo", "Learning"),
                ("Learning", "PostSleep"),
                ("PreSleep", "PostSleep"),
            ]
            annotator = Annotator(ax0, pairs, data=df_bars, x="Epoch", y="Score")
            annotator.configure(test="t-test_paired", text_format="star", loc="inside")
            annotator.apply_and_annotate()

            # Compute averaged matrices for 2D spatial layouts
            mean_hab_spatial = np.nanmean(np.array(spatial_hab_list), axis=0)
            mean_cond_spatial = np.nanmean(np.array(spatial_cond_list), axis=0)

            # Panel 2: 2D Hab Map
            ax1 = fig.add_subplot(gs[1])
            im1 = ax1.imshow(
                mean_hab_spatial.T, origin="lower", cmap="hot", extent=[0, 1, 0, 1]
            )
            ax1.set_title("Mean Score: Hab")
            plt.colorbar(im1, ax=ax1, shrink=0.7)

            # Panel 3: 2D Cond Map
            ax2 = fig.add_subplot(gs[2])
            im2 = ax2.imshow(
                mean_cond_spatial.T, origin="lower", cmap="hot", extent=[0, 1, 0, 1]
            )
            ax2.set_title("Mean Score: Cond")
            plt.colorbar(im2, ax=ax2, shrink=0.7)

            # Panel 4: 2D Differential Map (Cond - Hab Topology)
            ax3 = fig.add_subplot(gs[3])
            im3 = ax3.imshow(
                (mean_cond_spatial - mean_hab_spatial).T,
                origin="lower",
                cmap="jet",
                extent=[0, 1, 0, 1],
            )
            ax3.set_title(r"$\Delta$ Topology (Cond - Hab)")
            fig.suptitle(
                f"PCA reactivations for {manipe} (n = {len(spatial_hab_list) / num_templates:.0f} sessions, {num_templates} templates)",
            )
            plt.colorbar(im3, ax=ax3, shrink=0.7)

            sns.despine()
            plt.tight_layout()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                filename = f"pca_reactivation_summary_{template_period}_{winMS}ms_{'wLSpikeData' if spike_data else 'wLatentData'}_{manipe}"
                plt.savefig(
                    os.path.join(save_path, filename + ".png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    os.path.join(save_path, filename + ".svg"),
                )
            plt.show()

    def summarize_reactivation_by_condition(
        self,
        winMS=100,
        template_period="cond",
        num_templates=2,
        spike_data=True,
        session_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Return a compact dataframe comparing reactivation across groups and event windows."""
        if session_data is None:
            if spike_data:
                session_data = self.compute_pca_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=num_templates,
                )
            else:
                session_data = self.compute_latent_pca_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=num_templates,
                )

        rows = []
        for session_key, data in session_data.items():
            mouse_name, manipe = _parse_session_key(session_key)
            for t_idx in range(num_templates):
                summary = data.get("summaries", {}).get(t_idx, {})
                if not summary:
                    continue
                row = {
                    "Session": session_key,
                    "Mouse": mouse_name,
                    "Group": manipe,
                    "Template": t_idx,
                }
                row.update(summary)
                rows.append(row)

        return pd.DataFrame(rows)

    def plot_reactivation_condition_comparison(
        self,
        summary_df: Optional[pd.DataFrame] = None,
        winMS=100,
        template_period="cond",
        num_templates=2,
        spike_data=True,
        save_path: Optional[str] = None,
    ):
        """Plot a group-wise comparison of reactivation summaries across conditions."""
        if summary_df is None:
            summary_df = self.summarize_reactivation_by_condition(
                winMS=winMS,
                template_period=template_period,
                num_templates=num_templates,
                spike_data=spike_data,
            )

        if summary_df.empty:
            print("No reactivation summary data available.")
            return

        metric_candidates = [
            "cond_minus_pre_test",
            "cond_minus_post_test",
            "sleep_delta",
            "cond_ripples",
            "cond_freeze",
            "cond_stim",
            "cond_move",
            "cond_no_ripples",
        ]
        available_metrics = [
            col for col in metric_candidates if col in summary_df.columns
        ]
        if not available_metrics:
            print("No comparison metrics were found in the summary dataframe.")
            return

        fig, axes = plt.subplots(
            1, len(available_metrics), figsize=(5 * len(available_metrics), 6)
        )
        if len(available_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available_metrics):
            sns.boxplot(
                data=summary_df,
                x="Group",
                y=metric,
                hue="Group",
                palette={
                    k: GROUPS_PALETTE.get(k, "#7F7F7F")
                    for k in summary_df["Group"].unique()
                },
                ax=ax,
                showfliers=False,
            )
            sns.stripplot(
                data=summary_df,
                x="Group",
                y=metric,
                hue="Group",
                palette={
                    k: GROUPS_PALETTE.get(k, "#7F7F7F")
                    for k in summary_df["Group"].unique()
                },
                ax=ax,
                dodge=False,
                size=5,
                alpha=0.7,
                legend=False,
            )
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Manipulation")
            ax.set_ylabel("Reactivation difference")
            ax.tick_params(axis="x", rotation=45)
            ax.legend().remove()

        fig.suptitle(
            f"Group-wise reactivation summaries ({template_period}, {winMS} ms)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(
                    save_path,
                    f"reactivation_condition_comparison_{template_period}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def plot_pc_cell_weights_and_spatial_fields(
        self, session_dict, pc_idx=0, phase="cond"
    ):
        """
        Reproduces Peyrache Fig 1a/c for the U-Maze:
        1. Sorts cells by their loading (weight) in PC `pc_idx`.
        2. Plots a sorted Peri-Shock PETH / Spatial Rate Map for all neurons.
        3. Plots the signed PC score across the linearized U-Maze track.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.ndimage import gaussian_filter1d

        weights = session_dict["eigenvectors"][:, pc_idx]  # [N_cells]
        q_tsd = session_dict["q_tsd"].restrict(session_dict["epochs"][phase])

        # 1. Sort cells by PC Weight (Descending: large positive to large negative)
        sort_idx = np.argsort(weights)[::-1]
        sorted_weights = weights[sort_idx]
        sorted_q = q_tsd.values[:, sort_idx]

        # 2. Compute Spatial Tuning / Linearized Rate Maps for Sorted Cells
        lin_pos = session_dict["positions"]["linear"]
        pos_time = session_dict["positions"]["time"]

        # Bin positions (0 = Shock Zone, 1 = Safe Zone)
        bins = np.linspace(0, 1, 50)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Interpolate linearized position onto Q-matrix timestamps
        interp_lin_pos = np.interp(q_tsd.index, pos_time, lin_pos)
        lin_tsd = Tsd(t=q_tsd.index, d=interp_lin_pos).restrict(
            session_dict["epochs"][phase]
        )

        # Spatial rate map array: [N_cells, N_spatial_bins]
        spatial_rate_maps = np.zeros((len(sort_idx), len(bins) - 1))

        for i, cell_idx in enumerate(sort_idx):
            spk_counts = sorted_q[:, i]
            # Calculate mean firing rate per spatial bin
            bin_idx = np.digitize(lin_tsd.values, bins) - 1
            for b in range(len(bins) - 1):
                mask = bin_idx == b
                if np.any(mask):
                    spatial_rate_maps[i, b] = np.mean(spk_counts[mask])

            # Smooth spatial tuning curve
            spatial_rate_maps[i, :] = gaussian_filter1d(
                spatial_rate_maps[i, :], sigma=1.5
            )

        # --- PLOTTING ---
        fig, axs = plt.subplots(
            1, 3, figsize=(18, 8), gridspec_kw={"width_ratios": [0.5, 2, 2]}
        )

        # Panel A: PC Weights
        axs[0].barh(range(len(sorted_weights)), sorted_weights, color="black")
        axs[0].axvline(0, color="red", linestyle="--")
        axs[0].set_ylabel("Cells (Sorted by Weight)")
        axs[0].set_xlabel(f"Weight in PC {pc_idx + 1}")
        axs[0].invert_yaxis()

        # Panel B: Sorted Spatial Rate Maps (Peyrache Fig 1a analog)
        im = axs[1].imshow(
            spatial_rate_maps,
            aspect="auto",
            cmap="viridis",
            extent=[0, 1, len(sort_idx), 0],
        )
        axs[1].axvline(0.2, color="red", linestyle="--", label="Shock Zone Boundary")
        axs[1].set_xlabel("Linearized Position (0=Shock, 1=Safe)")
        axs[1].set_ylabel("Cells (Sorted by Weight)")
        axs[1].set_title(f"Cell Spatial Firing Sorted by PC {pc_idx + 1} Weight")
        plt.colorbar(im, ax=axs[1], label="Normalized Firing Rate")

        # Panel C: Signed PC Score vs Spatial Position (Peyrache Fig 1c analog)
        pc_score_tsd = session_dict["pc_scores"][pc_idx].restrict(
            session_dict["epochs"][phase]
        )

        # Average PC score across spatial bins
        mean_pc_score = np.zeros(len(bins) - 1)
        bin_idx = np.digitize(lin_tsd.values, bins) - 1
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if np.any(mask):
                mean_pc_score[b] = np.mean(pc_score_tsd.values[mask])

        axs[2].plot(bin_centers, mean_pc_score, color="purple", linewidth=2.5)
        axs[2].axhline(0, color="gray", linestyle="--")
        axs[2].axvline(0.2, color="red", linestyle="--", label="Shock Zone")
        axs[2].set_xlabel("Linearized Position (0=Shock, 1=Safe)")
        axs[2].set_ylabel(f"Mean Signed PC {pc_idx + 1} Score")
        axs[2].set_title("Assembly Reactivation Profile Across Track")

        sns.despine()
        plt.tight_layout()
        plt.show()

    def compute_assembly_zone_migration(
        self, winMS=100, template_period="cond", num_templates=1
    ):
        """Calculates the specific mean reactivation strength within each ZONEDEF arena

        sub-slice across Hab, Cond, and SWR periods to look for spatial re-tuning.
        """

        # 1. Compute the raw PCA continuous traces using our previous method
        session_data = self.compute_pca_reactivation(
            winMS=winMS, template_period=template_period, num_templates=num_templates
        )

        migration_rows = []

        for s_key, data in session_data.items():
            # Deconstruct session keys back to metadata parameters
            try:
                mouse_name, group_label = s_key.split("_")
            except ValueError:
                mouse_name, expindex, group_label = s_key.split("_")
                mouse_name = f"{mouse_name}_{expindex}"

            for t_idx in range(num_templates):
                rs_tsd = data["rs"][t_idx]
                epochs = data["epochs"]

                # Align continuous RS values with actual coordinate positions
                pos_tsd = TsdFrame(
                    t=data["positions"]["time"],
                    d=np.vstack([data["positions"]["x"], data["positions"]["y"]]).T,
                )

                # 2. Iterate through your physical U-Maze zones
                for z_idx, zone_name in enumerate(ZONELABELS):
                    x_lim, y_lim = ZONEDEF[z_idx]

                    # Create a spatial interval mask where the mouse is physically inside this zone
                    in_zone_mask = (
                        (pos_tsd.values[:, 0] >= x_lim[0])
                        & (pos_tsd.values[:, 0] <= x_lim[1])
                        & (pos_tsd.values[:, 1] >= y_lim[0])
                        & (pos_tsd.values[:, 1] <= y_lim[1])
                    )

                    if not np.any(in_zone_mask):
                        continue

                    zone_intervals = (
                        Tsd(t=data["positions"]["time"], d=in_zone_mask.astype(int))
                        .threshold(0.5, "above")
                        .time_support
                    )

                    # 3. Calculate mean reactivation strength under different combined conditions
                    # Active exploration in this physical zone during Habituation
                    hab_zone_rs = np.nanmean(
                        rs_tsd.restrict(epochs["pre"].intersect(zone_intervals)).values
                    )

                    # Active exploration in this physical zone during Conditioning
                    cond_zone_rs = np.nanmean(
                        rs_tsd.restrict(epochs["cond"].intersect(zone_intervals)).values
                    )

                    # Micro-state intersection: SWRs that occur while the animal is awake on the maze inside this zone
                    awake_ripple_zone_rs = np.nanmean(
                        rs_tsd.restrict(
                            epochs["cond_ripples"].intersect(zone_intervals)
                        ).values
                    )

                    migration_rows.append(
                        {
                            "Mouse": mouse_name,
                            "Group": group_label,
                            "Template": t_idx,
                            "Zone": zone_name,
                            "RS_Hab": hab_zone_rs,
                            "RS_Cond": cond_zone_rs,
                            "RS_Awake_SWR": awake_ripple_zone_rs,
                        }
                    )

        return pd.DataFrame(migration_rows)

    def plot_assembly_migration(self, winMS=100, template_period="cond"):
        """Plots a comparison of assembly expression during Habituation vs Conditioning

        broken down by your physical ZONEDEF boundaries.
        """

        df_mig = self.compute_assembly_zone_migration(
            winMS=winMS, template_period=template_period
        )
        if df_mig.empty:
            print("No migration data compiled.")
            return

        # Reshape to long-form for clean comparison plotting
        df_melted = df_mig.melt(
            id_vars=["Mouse", "Group", "Zone"],
            value_vars=["RS_Hab", "RS_Cond"],
            var_name="Phase",
            value_name="Reactivation_Strength",
        )
        df_melted["Phase"] = df_melted["Phase"].map(
            {"RS_Hab": "Habituation", "RS_Cond": "Conditioning"}
        )

        # Plot the data
        g = sns.catplot(
            data=df_melted,
            x="Phase",
            y="Reactivation_Strength",
            hue="Group",
            col="Zone",
            kind="point",
            palette={
                k: GROUPS_PALETTE.get(k, "#7F7F7F") for k in df_melted["Group"].unique()
            },
            dodge=0.25,
            capsize=0.1,
            markers=list(["o", "s", "D", "X", "*"] * 3)[
                : len(df_melted["Group"].unique())
            ],
            linestyles=list(["-", "--", ":", "-.", ":"] * 3)[
                : len(df_melted["Group"].unique())
            ],
            errorbar="se",
            height=5,
            aspect=0.8,
        )

        g.set_axis_labels("", "Mean Reactivation Strength")
        g.set_titles("{col_name} Zone", weight="bold")
        plt.suptitle(
            f"Spatial Migration of PCA Assemblies ({template_period.upper()} Template)",
            y=1.05,
            fontsize=14,
            weight="bold",
        )
        plt.show()

    def plot_comprehensive_summary_matrix(self, winMS=100, save_path=None):
        """Generates a multi-panel production figure matching the complete
        3x4 macro dashboard of group-specific boxplots and robust behavior regressions.
        """
        from scipy.stats import spearmanr, wilcoxon
        from sklearn.linear_model import TheilSenRegressor

        # Initialize a 3 rows by 4 columns figure layout
        fig, axs = plt.subplots(3, 4, figsize=(24, 15), sharex=False, sharey=False)

        # Grid parameter dictionary setup
        row_configs = [
            {
                "phase": "pre",
                "subtask": "mov",
                "label": "Free exploration\nbefore learning",
            },
            {
                "phase": "cond",
                "subtask": "mov",
                "label": "Moving periods\nduring conditioning",
            },
            {
                "phase": "cond",
                "subtask": "ripples",
                "label": "Ripples\nduring conditioning",
            },
        ]

        for row_idx, cfg in enumerate(row_configs):
            phase = cfg["phase"]
            subtask = cfg["subtask"]

            # ==========================================
            # COLUMNS 1 & 2: EV vs REV BOXPLOTS PER GROUP
            # ==========================================
            df_ev = self.compute_kudrimoti_variance(
                winMS=winMS, task_phase=phase, subtask=subtask
            )

            if not df_ev.empty:
                df_melt = df_ev.melt(
                    id_vars=["Mouse", "Group"],
                    value_vars=["EV", "REV"],
                    var_name="Metric",
                    value_name="Percentage",
                )

                # Plot MFB/Control on Column 0, PAG/Aversive on Column 1
                for g_col_idx, group_name in enumerate(["MFB", "PAG"]):
                    ax_box = axs[row_idx, g_col_idx]
                    g_data = df_melt[df_melt["Group"] == group_name]

                    if not g_data.empty:
                        box_color = GROUPS_PALETTE.get(group_name, "#7F7F7F")

                        sns.boxplot(
                            data=g_data,
                            x="Metric",
                            y="Percentage",
                            ax=ax_box,
                            color=box_color,
                            width=0.4,
                            fliersize=0,
                            boxprops=dict(alpha=0.6),
                        )
                        sns.stripplot(
                            data=g_data,
                            x="Metric",
                            y="Percentage",
                            ax=ax_box,
                            color="black",
                            size=6,
                            jitter=0.15,
                            edgecolor="black",
                            linewidth=1,
                        )

                        # Calculate Wilcoxon Signed-Rank Test between EV and REV pairs
                        ev_vals = g_data[g_data["Metric"] == "EV"]["Percentage"].values
                        rev_vals = g_data[g_data["Metric"] == "REV"][
                            "Percentage"
                        ].values

                        if len(ev_vals) >= 3 and not np.all(ev_vals == rev_vals):
                            stat, p_w = wilcoxon(ev_vals, rev_vals)

                            # Standard alpha threshold string generation
                            sig_label = (
                                "***"
                                if p_w < 0.001
                                else "**"
                                if p_w < 0.01
                                else "*"
                                if p_w < 0.05
                                else "n.s."
                            )

                            # Annotate stats inside the boxplot window
                            ax_box.text(
                                0.5,
                                0.90,
                                f"Wilcoxon: {sig_label}\np = {p_w:.3f}",
                                transform=ax_box.transAxes,
                                fontsize=9,
                                ha="center",
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    alpha=0.7,
                                ),
                            )

                    ax_box.set_title(
                        f"{cfg['label']}\n({group_name})", fontsize=10, weight="bold"
                    )
                    ax_box.set_ylabel("% explained" if g_col_idx == 0 else "")
                    ax_box.set_xlabel("")
                    ax_box.set_ylim(-5, 45)

            # ==========================================
            # COLUMNS 3 & 4: GROUP-SPECIFIC CORRELATIONS
            # ==========================================
            df_corr = self.compute_ev_behavior_correlation(
                winMS=winMS, task_phase=phase, subtask=subtask
            )

            if not df_corr.empty and len(df_corr) >= 3:
                metrics = ["Delta_Latency", "Delta_Occupancy"]
                y_labels = [
                    "Latency delta (Post-Pre sec)",
                    "Occupancy delta (Post-Pre %)",
                ]
                unique_groups = df_corr["Group"].unique()

                for m_idx, metric_name in enumerate(metrics):
                    ax_scat = axs[row_idx, 2 + m_idx]

                    # Base scatter colored by Group matching our earlier functions
                    sns.scatterplot(
                        data=df_corr,
                        x="EV",
                        y=metric_name,
                        hue="Group",
                        palette={
                            k: GROUPS_PALETTE.get(k, "#7F7F7F")
                            for k in df_corr["Group"].unique()
                        },
                        s=100,
                        edgecolor="black",
                        alpha=0.85,
                        ax=ax_scat,
                        legend=(row_idx == 0 and m_idx == 0),
                    )

                    text_box_lines = []

                    # Fit separate lines and calculate individual correlations per group
                    for group in unique_groups:
                        group_df = df_corr[df_corr["Group"] == group]

                        if len(group_df) < 3:
                            continue

                        x_g = group_df["EV"].values
                        y_g = group_df[metric_name].values

                        # Spearman stats per isolated configuration
                        rho, p_val = spearmanr(x_g, y_g)
                        g_color = GROUPS_PALETTE.get(group, "black")

                        # Apply Theil-Sen Robust Regression
                        try:
                            reg = TheilSenRegressor(random_state=42).fit(
                                x_g.reshape(-1, 1), y_g
                            )
                            x_line = np.linspace(x_g.min(), x_g.max(), 100).reshape(
                                -1, 1
                            )
                            y_line = reg.predict(x_line)
                            ax_scat.plot(
                                x_line.flatten(), y_line, color=g_color, linewidth=2.5
                            )
                        except Exception:
                            slope, intercept = np.polyfit(x_g, y_g, 1)
                            x_line = np.linspace(x_g.min(), x_g.max(), 100)
                            ax_scat.plot(
                                x_line,
                                slope * x_line + intercept,
                                color=g_color,
                                linewidth=1.5,
                                linestyle="--",
                            )

                        text_box_lines.append(
                            f"{group}: $\\rho$={rho:.2f} (p={p_val:.2f})"
                        )

                    # Annotate robust line metrics
                    ax_scat.text(
                        0.05,
                        0.95,
                        "\n".join(text_box_lines),
                        transform=ax_scat.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
                    )

                    ax_scat.set_ylabel(y_labels[m_idx], fontweight="bold")
                    ax_scat.set_xlabel(f"EV ({subtask} epoch %)")

        # Legend styling adjustments
        if axs[0, 2].get_legend() is not None:
            axs[0, 2].legend(loc="upper right", fontsize=8, title="Groups")

        sns.despine()
        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                os.path.join(save_path, "comprehensive_summary_matrix.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def plot_spatial_similarity_summary(
        self,
        winMS=100,
        template_period="cond",
        num_templates=1,
        save_path: Optional[str] = None,
        session_data: Optional[dict] = None,
        spike_data: bool = True,
    ):
        """Generates a comprehensive multi-panel spatial topology dashboard matching
        the U-Maze activation maps, differential matrices, and cross-zone bar plots,
        filtered for MFB and PAG groups with group-specific occupancy tracking.
        """
        from scipy.ndimage import gaussian_filter

        # 1. Compute the raw PCA continuous traces using your existing method
        if session_data is None:
            if spike_data:
                session_data = self.compute_pca_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=num_templates,
                )
            else:
                session_data = self.compute_latent_pca_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=num_templates,
                )

        if not session_data:
            print("No valid session matrices computed.")
            return

        # Target only specified groups
        target_groups = ["MFB", "PAG"]

        # Initialize group-specific storage dictionaries
        group_spatial_hab = {g: [] for g in target_groups}
        group_spatial_cond = {g: [] for g in target_groups}
        group_occupancy_hab = {g: [] for g in target_groups}
        group_occupancy_cond = {g: [] for g in target_groups}

        zone_rows = []

        # 2. Iterate through sessions and group maps by manipulation type
        for s_key, data in session_data.items():
            try:
                mouse_name, group_label = s_key.split("_")
            except ValueError:
                mouse_name, expindex, group_label = s_key.split("_")
                mouse_name = f"{mouse_name}_{expindex}"

            # Filter out non-target groups (e.g., Controls or other manipulations)
            if group_label not in target_groups:
                continue

            for t_idx in range(num_templates):
                rs_tsd = data["rs"][t_idx]
                epochs = data["epochs"]

                x_pos = data["positions"]["x"]
                y_pos = data["positions"]["y"]
                t_pos = data["positions"]["time"]
                pos_tsd = TsdFrame(
                    t=t_pos, d=np.vstack([x_pos, y_pos]).T, columns=["x", "y"]
                )

                # --- 2D Matrix Computations ---
                h_map, c_map = self.map_reactivation_space(
                    data, template_idx=t_idx, bins=20
                )
                group_spatial_hab[group_label].append(h_map)
                group_spatial_cond[group_label].append(c_map)

                # Compute behavioral occupancy density grids
                edges = np.linspace(0, 1, 21)
                h_occ, _, _ = np.histogram2d(
                    pos_tsd.restrict(epochs["pre"]).values[:, 0],
                    pos_tsd.restrict(epochs["pre"]).values[:, 1],
                    bins=edges,
                )
                c_occ, _, _ = np.histogram2d(
                    pos_tsd.restrict(epochs["cond"]).values[:, 0],
                    pos_tsd.restrict(epochs["cond"]).values[:, 1],
                    bins=edges,
                )

                group_occupancy_hab[group_label].append(
                    gaussian_filter(h_occ / (np.sum(h_occ) + 1e-12), sigma=0.8)
                )
                group_occupancy_cond[group_label].append(
                    gaussian_filter(c_occ / (np.sum(c_occ) + 1e-12), sigma=0.8)
                )

                # --- Discrete Zone Value Extractions ---
                for z_idx, zone_name in enumerate(ZONELABELS):
                    x_lim, y_lim = ZONEDEF[z_idx]

                    in_zone_mask = (
                        (pos_tsd.values[:, 0] >= x_lim[0])
                        & (pos_tsd.values[:, 0] <= x_lim[1])
                        & (pos_tsd.values[:, 1] >= y_lim[0])
                        & (pos_tsd.values[:, 1] <= y_lim[1])
                    )

                    if np.any(in_zone_mask):
                        zone_intervals = (
                            Tsd(t=data["positions"]["time"], d=in_zone_mask.astype(int))
                            .threshold(0.5, "above")
                            .time_support
                        )

                        hab_val = np.nanmean(
                            rs_tsd.restrict(
                                epochs["pre"].intersect(zone_intervals)
                            ).values
                        )
                        cond_val = np.nanmean(
                            rs_tsd.restrict(
                                epochs["cond"].intersect(zone_intervals)
                            ).values
                        )

                        zone_rows.append(
                            {
                                "Mouse": mouse_name,
                                "Group": group_label,
                                "Zone": zone_name,
                                "Delta_Score": cond_val - hab_val,
                            }
                        )

        df_zones = pd.DataFrame(zone_rows)
        if df_zones.empty:
            print("No valid target group data matched.")
            return

        # 3. Calculate Group-Specific Averages
        mean_maps = {}
        for g in target_groups:
            if len(group_spatial_hab[g]) > 0:
                mean_maps[g] = {
                    "spatial_diff": np.nanmean(np.array(group_spatial_cond[g]), axis=0)
                    - np.nanmean(np.array(group_spatial_hab[g]), axis=0),
                    "occupancy_diff": np.nanmean(
                        np.array(group_occupancy_cond[g]), axis=0
                    )
                    - np.nanmean(np.array(group_occupancy_hab[g]), axis=0),
                }

        # 4. Construct Multi-Panel Grid Layout (3 Rows x 3 Columns)
        # Column 0: MFB Neural/Behav Maps, Column 1: PAG Neural/Behav Maps, Column 2: Occupancy Deltas
        fig = plt.figure(figsize=(20, 16), facecolor="white")
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

        heatmap_axes = []

        # --- ROW 0: Neural Assembly Reactivation Difference (Cond - Hab) ---
        for g_idx, g in enumerate(target_groups):
            ax = fig.add_subplot(gs[0, g_idx])
            heatmap_axes.append(ax)
            if g in mean_maps:
                im = ax.imshow(
                    mean_maps[g]["spatial_diff"].T,
                    origin="lower",
                    cmap="bwr",
                    extent=[0, 1, 0, 1],
                    vmin=-1.5,
                    vmax=1.5,
                )
                plt.colorbar(im, ax=ax, shrink=0.7, label=r"$\Delta$ Similarity")
            ax.set_title(
                rf"Neural Assembly $\Delta$ Matrix\n({g} Group)",
                weight="bold",
                fontsize=11,
            )

        # --- ROW 1: Behavioral Occupancy Difference (Cond - Hab) ---
        for g_idx, g in enumerate(target_groups):
            ax = fig.add_subplot(gs[1, g_idx])
            heatmap_axes.append(ax)
            if g in mean_maps:
                # Using 'bwr' or 'coolwarm' to capture areas of avoidance (blue) vs preference (red)
                im = ax.imshow(
                    mean_maps[g]["occupancy_diff"].T,
                    origin="lower",
                    cmap="bwr",
                    extent=[0, 1, 0, 1],
                )
                plt.colorbar(im, ax=ax, shrink=0.7, label=r"$\Delta$ Density")
            ax.set_title(
                rf"Behavioral Occupancy $\Delta$ Map\n({g} Group)",
                weight="bold",
                fontsize=11,
            )

        # --- ROW 2: Combined Downstream Quantitative Analysis Panels ---
        # Panel E: Consolidated Bar Plot across regions
        ax_e = fig.add_subplot(gs[2, 0:2])  # Spans across columns 0 and 1
        sns.barplot(
            data=df_zones,
            x="Zone",
            y="Delta_Score",
            hue="Group",
            order=ZONELABELS,
            palette={
                k: GROUPS_PALETTE.get(k, "#7F7F7F") for k in df_zones["Group"].unique()
            },
            edgecolor="black",
            linewidth=1.5,
            errorbar="se",
            alpha=0.85,
            ax=ax_e,
        )
        sns.stripplot(
            data=df_zones,
            x="Zone",
            y="Delta_Score",
            hue="Group",
            order=ZONELABELS,
            palette={
                k: GROUPS_PALETTE.get(k, "#7F7F7F") for k in df_zones["Group"].unique()
            },
            size=5,
            jitter=0.15,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
            dodge=True,
            ax=ax_e,
            legend=False,
        )
        ax_e.axhline(0, color="gray", linestyle="--", alpha=0.6)
        ax_e.set_title(
            r"E. Learning Effects: $\Delta$ Similarity Score across Zones",
            weight="bold",
            fontsize=12,
        )
        ax_e.set_ylabel(r"$\Delta$ Similarity Score (Cond - Hab)")
        ax_e.set_xlabel("")
        ax_e.set_xticklabels(ZONELABELS, rotation=15, ha="right")
        ax_e.legend(title="Group", loc="upper right")

        # Hide empty remaining grids or place customized schematic info
        for col in [2]:
            ax_hide = fig.add_subplot(gs[2, col])
            ax_hide.axis("off")

        # Formatting maps
        for ax in heatmap_axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

        sns.despine(left=False, bottom=False)
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"spatial_similarity_summary_{template_period}_{winMS}ms_{'wSpikeData' if spike_data else 'wLatentData'}"
            plt.savefig(
                os.path.join(save_path, filename + ".png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(save_path, filename + ".svg"),
            )

        plt.show()

    def compute_latent_pca_reactivation(
        self, winMS=100, template_period="cond", num_templates=2
    ):
        """Smarter full-session latent manager. Dynamically stitches together latent spaces
        from all behavioral phases (pre, hab, cond, post) to ensure continuous monitoring
        across sleep and active exploration.
        """
        all_session_data = {}

        for (mouse_name, manipe), df in self.results_df.groupby(
            by=["mouse_name", "manipe"]
        ):
            results: Mouse_Results = df.iloc[0].results
            idWindow = results.timeWindows.index(winMS)
            base_results_path = os.path.join(
                results.projectPath.experimentPath, "results"
            )

            pre_sleep, _ = results.get_epoch_interval("pre_sleep")
            pre_epoch, _ = results.get_epoch_interval("pre")
            cond_epoch, _ = results.get_epoch_interval("cond")
            post_epoch, _ = results.get_epoch_interval("post")
            post_sleep, _ = results.get_epoch_interval("post_sleep")
            hab_epoch, _ = results.get_epoch_interval("hab")

            sws_epochs = results.DataHelper.get_sws_epochs()
            ripples_epochs = results.DataHelper.get_ripples_epochs()

            try:
                freeze_epochs = results.DataHelper.get_freeze_epochs()
            except AttributeError:
                freeze_epochs = IntervalSet(start=[], end=[])

            # 2. STITCHING LAYER: Load and combine latents across all available phases
            # This ensures we have coverage for Pre-Sleep, Hab, Cond, and Post-Sleep
            phases_to_load = [
                "_pre",
                "_cond",
                "_post",
                "_training",
            ]

            stitched_times = []
            stitched_latents = []

            for p_suffix in phases_to_load:
                # Check if it's already in memory first
                if (
                    hasattr(results, "resultsNN_phase_pkl")
                    and p_suffix in results.resultsNN_phase_pkl
                ):
                    pkl_data = results.resultsNN_phase_pkl[p_suffix]
                    if (
                        "latent_output" in pkl_data
                        and len(pkl_data["latent_output"]) >= idWindow + 1
                        and pkl_data["latent_output"][idWindow] is not None
                        and "times" in results.resultsNN_phase[p_suffix]
                    ):
                        stitched_times.append(
                            results.resultsNN_phase[p_suffix]["times"][
                                idWindow
                            ].flatten()
                        )
                        stitched_latents.append(pkl_data["latent_output"][idWindow])
                        continue

                # Fallback: Read file dynamically from disk folder
                pkl_path = os.path.join(
                    base_results_path, str(winMS), f"decoding_results{p_suffix}.pkl"
                )
                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, "rb") as f:
                            temp_pkl = pickle.load(f)
                            if "times" in temp_pkl:
                                t_steps = temp_pkl["times"]
                            else:
                                raise ValueError(f"Missing 'times' key in {pkl_path}")

                            l_mat = temp_pkl.get("latent_output", None)
                            if isinstance(l_mat, list):
                                l_mat = np.array(l_mat)

                            stitched_times.append(t_steps.flatten())
                            stitched_latents.append(l_mat)

                            del temp_pkl
                            gc.collect()
                    except Exception as e:
                        print(f"Failed loading phase {p_suffix} for {mouse_name}: {e}")
                else:
                    print(
                        f"Phase file {pkl_path} not found for {mouse_name}. Skipping this phase."
                    )

            base_results_path_sleep = os.path.join(
                results.projectPath.experimentPath, "results_Sleep"
            )
            for sleep_name in results.DataHelper.fullBehavior["Times"].get(
                "sleepNames", []
            ):
                pkl_path = os.path.join(
                    base_results_path_sleep,
                    str(winMS),
                    sleep_name,
                    "decoding_results.pkl",
                )

                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, "rb") as f:
                            temp_pkl = pickle.load(f)
                            if "times" in temp_pkl:
                                t_steps = temp_pkl["times"]
                            else:
                                raise ValueError(
                                    f"Sleep file {pkl_path} missing 'times' key."
                                )

                            l_mat = temp_pkl.get("latent_output", None)
                            if isinstance(l_mat, list):
                                l_mat = np.array(l_mat)

                            stitched_times.append(t_steps.flatten())
                            stitched_latents.append(l_mat)

                            del temp_pkl
                            phases_to_load.append(sleep_name)
                            gc.collect()
                    except Exception as e:
                        print(
                            f"Failed loading sleep {sleep_name} for {mouse_name}: {e}"
                        )
                else:
                    print(
                        f"Sleep file {pkl_path} not found for {mouse_name}. Skipping this sleep phase."
                    )

            if not stitched_latents:
                print(
                    f"Skipping {mouse_name}: No latent data could be collected across phases."
                )
                continue

            flat_times = np.concatenate(stitched_times)
            flat_latents = np.concatenate(stitched_latents, axis=0)

            full_latent_tsd = TsdFrame(t=flat_times, d=flat_latents)

            # 5. Define template interval mapping windows
            if template_period == "wake":
                template_interval = cond_epoch
            elif template_period == "cond":
                template_interval = cond_epoch
            elif template_period == "condFree":
                template_interval = cond_epoch.intersect(freeze_epochs)
            elif template_period == "postRip":
                template_interval = post_sleep.intersect(sws_epochs).intersect(
                    ripples_epochs
                )
            elif template_period == "condRip":
                template_interval = cond_epoch.intersect(ripples_epochs)
            else:
                raise ValueError(f"Unknown template period paradigm: {template_period}")

            # 6. Build Eigenvectors from the Conditioning/Target template window
            lat_template = full_latent_tsd.restrict(template_interval).values
            if lat_template.shape[0] < 10:
                print(
                    f"Skipping {mouse_name}: Template window has insufficient data frames."
                )
                continue

            cov_matrix = np.cov(lat_template, rowvar=False)
            cov_matrix = np.nan_to_num(cov_matrix)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            idx_sorted = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx_sorted]

            # 7. Project full continuous session timeline onto the learning manifold
            lat_full_norm = (
                full_latent_tsd.values - np.mean(full_latent_tsd.values, axis=0)
            ) / (np.std(full_latent_tsd.values, axis=0) + 1e-12)
            lat_full_norm = np.nan_to_num(lat_full_norm)

            rs_templates = {}
            num_templates = min(num_templates, eigenvectors.shape[1])
            for idx_t in range(num_templates):
                v_i = eigenvectors[:, idx_t]
                score_t = np.dot(lat_full_norm, v_i)
                rs_templates[idx_t] = Tsd(t=full_latent_tsd.index, d=score_t**2)

            # 8. Return structured metadata package
            event_intervals = {
                "cond_ripples": cond_epoch.intersect(ripples_epochs),
                "cond_freeze": cond_epoch.intersect(freeze_epochs),
                "cond_no_ripples": cond_epoch.set_diff(ripples_epochs),
                "pre_sleep_sws": pre_sleep.intersect(sws_epochs),
                "post_sleep_sws": post_sleep.intersect(sws_epochs),
            }
            template_summaries = {}
            for idx_t, rs_tsd in rs_templates.items():
                template_summaries[idx_t] = _summarize_reactivation_strengths(
                    rs_tsd,
                    {
                        "pre_test": pre_epoch,
                        "pre_sleep": pre_sleep,
                        "cond": cond_epoch,
                        "post_test": post_epoch,
                        "post_sleep": post_sleep,
                        "hab": hab_epoch,
                    },
                    event_intervals,
                )

            all_session_data[f"{mouse_name}_{manipe}"] = {
                "rs": rs_templates,
                "summaries": template_summaries,
                "epochs": {
                    "pre_test": pre_epoch,
                    "pre_sleep": pre_sleep,
                    "pre_sleep_sws": pre_sleep.intersect(sws_epochs),
                    "pre": pre_epoch,
                    "hab": hab_epoch,
                    "cond": cond_epoch,
                    "cond_freeze": cond_epoch.intersect(freeze_epochs),
                    "cond_ripples": cond_epoch.intersect(ripples_epochs),
                    "cond_no_ripples": cond_epoch.set_diff(ripples_epochs),
                    "post_test": post_epoch,
                    "post": post_epoch,
                    "post_sleep": post_sleep,
                    "post_sleep_sws": post_sleep.intersect(sws_epochs),
                    "ripples": ripples_epochs,
                },
                "positions": {
                    "x": results.DataHelper.fullBehavior["Positions"][:, 0],
                    "y": results.DataHelper.fullBehavior["Positions"][:, 1],
                    "time": results.DataHelper.fullBehavior["positionTime"].flatten(),
                    "linear": results.l_function(
                        results.DataHelper.fullBehavior["Positions"][:, :2],
                    )[1].flatten(),
                },
            }

        return all_session_data

    def compute_assembly_reactivation_advanced(
        self,
        winMS=100,
        template_period="cond",
        num_templates=3,
        calc_type="PCA",  # "PCA" or "ICA"
        keep_mua=False,
        keep_interneurons=True,
        force=False,
        fig=False,
        max_templates=5,
        save_fig_path=None,
    ):
        """Advanced assembly reactivation tracker replicating react_pca_ica_AG.py.

        Supports PCA/ICA forks, neuron layer sub-filtering, shuffling controls,
        Marčenko-Pastur limits, and automated figure generation matching MATLAB output.
        """
        bin_size_sec = winMS / 1000.0
        all_session_data = {}

        for (mouse_name, manipe), df in self.results_df.groupby(
            by=["mouse_name", "manipe"]
        ):
            results: Mouse_Results = df.iloc[0].results
            print(
                f"Processing {mouse_name} ({manipe}) for {calc_type} reactivation analysis..."
            )

            # 1. Fetch Spike trains and corresponding cell layer classifications
            spike_group = results.DataHelper.get_spike_data()
            if len(spike_group) < 5:
                continue

            try:
                try:
                    neuron_types = results.DataHelper.get_neuron_classifications(
                        force=force
                    )
                except FileNotFoundError:
                    neuron_types = results.DataHelper.get_neuron_classifications(
                        folder=results.network_path, force=force
                    )
            except AttributeError:
                warn(
                    f"Neuron classification data missing for {mouse_name}. Defaulting to 'Pyramidal'."
                )
                neuron_types = np.array(["Pyramidal"] * len(spike_group))

            # 2. Filter neuron vectors based on multiU and interN parameters
            valid_indices = []
            for idx, n_type in enumerate(neuron_types):
                if "mua" in str(n_type).lower() and not keep_mua:
                    continue
                if "interneuron" in str(n_type).lower() and not keep_interneurons:
                    continue
                valid_indices.append(idx)

            if len(valid_indices) < 4:
                continue

            valid_indices = np.array(valid_indices)
            filtered_spikes = TsGroup({i: spike_group[i] for i in valid_indices})
            filtered_types = np.array(neuron_types)[valid_indices]
            cell_labels = [f"Cell_{i}" for i in valid_indices]

            # 3. Extract behavioral intervals
            pre_sleep, _ = results.get_epoch_interval("pre_sleep")
            pre_epoch, _ = results.get_epoch_interval("pre_test")
            cond_epoch, _ = results.get_epoch_interval("cond")
            post_epoch, _ = results.get_epoch_interval("post_test")
            post_sleep, _ = results.get_epoch_interval("post_sleep")
            hab_epoch, _ = results.get_epoch_interval("hab")

            sws_epochs = results.DataHelper.get_sws_epochs()
            ripples_epochs = results.DataHelper.get_ripples_epochs()

            try:
                freeze_epochs = results.DataHelper.get_freeze_epochs()
            except AttributeError:
                freeze_epochs = IntervalSet(start=[], end=[])

            # 4. Generate the main binned matrix (Q-Matrix)
            q_tsd = filtered_spikes.count(bin_size_sec)

            if template_period == "wake":
                template_interval = pre_epoch.union(cond_epoch).union(post_epoch)
            elif template_period == "cond":
                template_interval = cond_epoch
            elif template_period == "condFree":
                template_interval = cond_epoch.intersect(freeze_epochs)
            elif template_period == "postRip":
                template_interval = post_epoch.intersect(sws_epochs).intersect(
                    ripples_epochs
                )
            elif template_period == "condRip":
                template_interval = cond_epoch.intersect(ripples_epochs)
            else:
                raise ValueError(f"Unknown template window: {template_period}")

            # 5. Extract template frames and compute Z-scores
            q_template_raw = q_tsd.restrict(template_interval).values
            if q_template_raw.shape[0] < 10:
                continue

            mean_t = np.mean(q_template_raw, axis=0)
            std_t = np.std(q_template_raw, axis=0) + 1e-12
            q_template = (q_template_raw - mean_t) / std_t

            num_bins, num_neurons = q_template.shape

            # PCA vs ICA Fork
            if calc_type.upper() == "PCA":
                corr_matrix = np.corrcoef(q_template, rowvar=False)
                corr_matrix = np.nan_to_num(corr_matrix)
                np.fill_diagonal(corr_matrix, 0)

                eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
                sort_idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sort_idx]
                assemblies = eigenvectors[:, sort_idx]

                lambda_max = (1 + np.sqrt(num_neurons / num_bins)) ** 2

                shuffled_q = q_template.copy()
                for col in range(num_neurons):
                    shuffled_q[:, col] = np.random.permutation(shuffled_q[:, col])

                shuff_corr = np.corrcoef(shuffled_q, rowvar=False)
                np.fill_diagonal(shuff_corr, 0)
                shuff_values, _ = np.linalg.eigh(np.nan_to_num(shuff_corr))
                percentile_shuff = np.percentile(shuff_values, 100)

            elif calc_type.upper() == "ICA":
                ica = FastICA(n_components=num_templates, random_state=42, max_iter=500)
                ica.fit(q_template)
                assemblies = ica.components_.T
                eigenvalues = np.ones(num_templates)
                lambda_max, percentile_shuff = np.nan, np.nan
            else:
                raise ValueError("calc_type must be either 'PCA' or 'ICA'")

            # 6. Projection onto session timeline
            q_full_normalized = (q_tsd.values - np.mean(q_tsd.values, axis=0)) / (
                np.std(q_tsd.values, axis=0) + 1e-12
            )
            q_full_normalized = np.nan_to_num(q_full_normalized)

            rs_templates = {}
            num_templates = min(num_templates, assemblies.shape[1])
            for idx_t in range(num_templates):
                v_i = assemblies[:, idx_t]
                outer_product = np.outer(v_i, v_i)
                np.fill_diagonal(outer_product, 0)

                react_matrix = np.dot(q_full_normalized, outer_product)
                rs_vector = np.sum(react_matrix * q_full_normalized, axis=1)
                rs_tsd = Tsd(t=q_tsd.times(), d=rs_vector)

                mean_pre = np.mean(
                    rs_tsd.restrict(
                        IntervalSet(q_tsd.time_support).intersect(pre_epoch)
                    ).values
                )
                mean_post = np.mean(
                    rs_tsd.restrict(
                        IntervalSet(q_tsd.time_support).intersect(post_epoch)
                    ).values
                )

                if (
                    max(abs(mean_pre), abs(mean_post)) != max(mean_pre, mean_post)
                    and calc_type.upper() == "PCA"
                ):
                    rs_tsd = rs_tsd * -1.0

                rs_templates[idx_t] = rs_tsd

            # Position tracking data
            pos_x = results.DataHelper.fullBehavior["Positions"][:, 0]
            pos_y = results.DataHelper.fullBehavior["Positions"][:, 1]
            pos_t = results.DataHelper.fullBehavior["positionTime"].flatten()

            # 7. Automated Figure Generation
            if fig:
                session_id = f"{mouse_name}_{manipe}"

                # FIG 1: Stem Plot for Assembly Weights
                for idx_t in range(num_templates):
                    if lambda_max is not None and eigenvalues[idx_t] < 0.9 * lambda_max:
                        warn(
                            f"Skipping template {idx_t + 1} for {mouse_name}: Eigenvalue below Marčenko-Pastur threshold."
                        )
                        break
                    if idx_t >= max_templates:
                        warn(
                            f"Skipping template {idx_t + 1} for {mouse_name}: Exceeds max_templates limit of {max_templates}."
                        )
                        break
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    w = assemblies[:, idx_t]
                    x_axis = np.arange(len(w))
                    mu_w, std_w = np.mean(w), np.std(w)

                    colors = {
                        "SUA_pyramidal": "#77AC30",
                        "SUA_interneuron": "#0072BD",
                        "MUA": "#D95319",
                        "unclassified": "#7F7F7F",
                    }
                    for n_type in np.unique(filtered_types):
                        m_idx = np.where(filtered_types == n_type)[0]
                        c = colors.get(n_type, None)
                        if c is None:
                            for key in colors.keys():
                                if n_type in key:
                                    c = colors[key]
                                    break

                        marker, stem, base = ax1.stem(
                            m_idx,
                            w[m_idx],
                            linefmt=c,
                            markerfmt="o",
                            label=n_type,
                        )
                        plt.setp(
                            marker, markerfacecolor=c, markeredgecolor=c, markersize=5
                        )
                        plt.setp(stem, color=c)

                    ax1.axhline(
                        mu_w + 2 * std_w, color="r", linestyle="--", label="±2 SD"
                    )
                    ax1.axhline(mu_w - 2 * std_w, color="r", linestyle="--")
                    ax1.set_xticks(x_axis)
                    ax1.set_xticklabels(cell_labels, rotation=90, fontsize=6)
                    ax1.set_ylabel("Weight")
                    ax1.set_title(
                        f"{session_id} | {calc_type} Template #{idx_t + 1} ({template_period})"
                    )
                    ax1.set_ylim([-0.55, 0.55])
                    ax1.legend(loc="upper right")
                    plt.tight_layout()
                    if save_fig_path:
                        fig1.savefig(
                            f"{save_fig_path}/{session_id}_weight_PC{idx_t + 1}.png"
                        )

                # FIG 2 & 3: Epoch Comparison & Ripple-Triggered PETH
                print(f"Generating epoch comparison plots for {session_id}...")
                pre_sws_rs = [
                    np.mean(rs.restrict(pre_sleep.intersect(sws_epochs)).values)
                    for rs in rs_templates.values()
                ]
                hab_rs = [
                    np.mean(rs.restrict(hab_epoch.union(pre_epoch)).values)
                    for rs in rs_templates.values()
                ]
                cond_rs = [
                    np.mean(rs.restrict(cond_epoch).values)
                    for rs in rs_templates.values()
                ]
                post_sws_rs = [
                    np.mean(rs.restrict(post_sleep.intersect(sws_epochs)).values)
                    for rs in rs_templates.values()
                ]
                normalized_eigenvalues = eigenvalues / np.max(eigenvalues)

                fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))

                # Scatter Pre vs Post
                scatter = axes2[0].scatter(
                    pre_sws_rs,
                    post_sws_rs,
                    c=normalized_eigenvalues[:num_templates],
                    alpha=0.8,
                )
                if len(pre_sws_rs) > 1:
                    r_val, p_val = stats.pearsonr(pre_sws_rs, post_sws_rs)
                    axes2[0].set_title(
                        f"Pre vs Post SWS (r={r_val:.2f}, p={p_val:.3f})"
                    )
                axes2[0].plot(
                    [
                        min(pre_sws_rs + post_sws_rs),
                        max(pre_sws_rs + post_sws_rs),
                    ],
                    [
                        min(pre_sws_rs + post_sws_rs),
                        max(pre_sws_rs + post_sws_rs),
                    ],
                    "k:",
                )
                axes2[0].set_xlabel("Reactivation (Pre-Sleep SWS)")
                axes2[0].set_ylabel("Reactivation (Post-Sleep SWS)")
                plt.colorbar(scatter, ax=axes2[0], label="Normalized Eigenvalue")

                # Bar chart over epochs
                epoch_names = ["PreSleep", "FreeExplo", "Learning", "PostSleep"]
                means = [
                    np.mean(pre_sws_rs),
                    np.mean(hab_rs),
                    np.mean(cond_rs),
                    np.mean(post_sws_rs),
                ]
                sems = [
                    stats.sem(pre_sws_rs) if len(pre_sws_rs) > 1 else 0,
                    stats.sem(hab_rs) if len(hab_rs) > 1 else 0,
                    stats.sem(cond_rs) if len(cond_rs) > 1 else 0,
                    stats.sem(post_sws_rs) if len(post_sws_rs) > 1 else 0,
                ]
                axes2[1].bar(
                    epoch_names,
                    means,
                    yerr=sems,
                    color=["white", "#CAE62F", "#E60000", "black"],
                    edgecolor="k",
                    alpha=0.7,
                    capsize=4,
                )
                axes2[1].set_ylabel("Reactivation Score")
                axes2[1].set_title("Mean Reactivation Across Epochs")
                plt.tight_layout()

                if save_fig_path:
                    fig2.savefig(f"{save_fig_path}/{session_id}_epoch_comparison.png")

                # FIG 4: Spatial Maps (10x10 Maze Spatial Binning)
                grid_hab = _compute_2d_spatial_reactivation(
                    rs_templates[0], pos_x, pos_y, pos_t, hab_epoch.union(pre_epoch)
                )
                grid_cond = _compute_2d_spatial_reactivation(
                    rs_templates[0], pos_x, pos_y, pos_t, cond_epoch
                )

                fig3, axes3 = plt.subplots(1, 3, figsize=(12, 3.8))
                vmax = max(np.nanmax(grid_hab), np.nanmax(grid_cond))
                vmin = min(np.nanmin(grid_hab), np.nanmin(grid_cond))

                im0 = axes3[0].imshow(
                    grid_hab.T, origin="lower", cmap="hot", vmin=vmin, vmax=vmax
                )
                axes3[0].set_title("Habituation Map")
                fig3.colorbar(im0, ax=axes3[0])

                im1 = axes3[1].imshow(
                    grid_cond.T, origin="lower", cmap="hot", vmin=vmin, vmax=vmax
                )
                axes3[1].set_title("Conditioning Map")
                fig3.colorbar(im1, ax=axes3[1])

                diff_map = grid_cond - grid_hab
                im2 = axes3[2].imshow(diff_map.T, origin="lower", cmap="bwr")
                axes3[2].set_title("Cond - Hab Difference")
                fig3.colorbar(im2, ax=axes3[2])

                for ax in axes3:
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.tight_layout()
                if save_fig_path:
                    fig3.savefig(
                        f"{save_fig_path}/{session_id}_spatial_reactivation.png"
                    )

            # 8. Store output dictionary
            event_intervals = {
                "cond_ripples": cond_epoch.intersect(ripples_epochs),
                "cond_freeze": cond_epoch.intersect(freeze_epochs),
                "cond_no_ripples": cond_epoch.set_diff(ripples_epochs),
                "pre_sleep_sws": pre_sleep.intersect(sws_epochs),
                "post_sleep_sws": post_sleep.intersect(sws_epochs),
            }
            template_summaries = {}
            for idx_t, rs_tsd in rs_templates.items():
                template_summaries[idx_t] = _summarize_reactivation_strengths(
                    rs_tsd,
                    {
                        "pre_test": pre_epoch,
                        "pre_sleep": pre_sleep,
                        "cond": cond_epoch,
                        "post_test": post_epoch,
                        "post_sleep": post_sleep,
                        "hab": hab_epoch,
                    },
                    event_intervals,
                )

            all_session_data[f"{mouse_name}_{manipe}"] = {
                "rs": rs_templates,
                "weights": assemblies[:, :num_templates],
                "neuron_labels": cell_labels,
                "neuron_types": filtered_types,
                "spikes": filtered_spikes,
                "lfp": results.DataHelper.get_lfp_data(
                    channel_type="ripple", network_path=results.network_path
                ),
                "summaries": template_summaries,
                "stats": {
                    "eigenvalues": eigenvalues[:num_templates],
                    "marcenko_pastur": lambda_max,
                    "shuffle_max": percentile_shuff,
                },
                "epochs": {
                    "pre_test": pre_epoch,
                    "pre_sleep": pre_sleep,
                    "pre_sleep_sws": pre_sleep.intersect(sws_epochs),
                    "pre": pre_epoch,
                    "hab": hab_epoch,
                    "cond": cond_epoch,
                    "cond_freeze": cond_epoch.intersect(freeze_epochs),
                    "cond_ripples": cond_epoch.intersect(ripples_epochs),
                    "post_test": post_epoch,
                    "post": post_epoch,
                    "post_sleep": post_sleep,
                    "post_sleep_sws": post_sleep.intersect(sws_epochs),
                    "ripples": ripples_epochs,
                },
                "positions": {
                    "x": pos_x,
                    "y": pos_y,
                    "time": pos_t,
                },
            }

        return all_session_data

    def plot_advanced_diagnostics(self, all_session_data, session_key, calc_type="PCA"):
        """Plots the eigenvalues against Marčenko-Pastur lines and renders stem

        diagrams mapping unit contribution weights across single assemblies.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if session_key not in all_session_data:
            print(f"Session '{session_key}' not found.")
            return

        data = all_session_data[session_key]
        weights = data["weights"]
        n_labels = data["neuron_labels"]
        n_types = data["neuron_types"]
        stats = data["stats"]
        num_templates = weights.shape[1]

        # Plot 1: Scree Plot / Marčenko-Pastur law checking (Only applicable for PCA mode)
        if calc_type.upper() == "PCA":
            plt.figure(figsize=(6, 4))
            plt.plot(stats["eigenvalues"], "o-", color="red", label="Eigenvalues")
            plt.axhline(
                stats["marcenko_pastur"],
                color="orange",
                linestyle="--",
                label="Marcenko-Pastur Limit",
            )
            plt.axhline(
                stats["shuffle_max"],
                color="blue",
                linestyle=":",
                label="Shuffle Max Limit",
            )
            plt.xlabel("Component Index")
            plt.ylabel("Eigenvalue Magnitude")
            plt.title(f"Manifold Dimensionality check ({session_key})")
            plt.legend()
            sns.despine()
            plt.show()

        # Plot 2: Neuron Component Contribution Weights Diagrams
        color_map = {
            "Pyramidal": "#76A92F",
            "interNeuron": "#0072BA",
            "MultiUnit": "#EDB119",
        }

        for t_idx in range(num_templates):
            _, ax = plt.subplots(figsize=(12, 4))
            w_vector = weights[:, t_idx]
            x_indices = np.arange(len(w_vector))

            # Compute threshold bounds (+/- 2 Standard Deviations)
            th_upper = np.mean(w_vector) + 2 * np.std(w_vector)
            th_lower = np.mean(w_vector) - 2 * np.std(w_vector)

            # Draw stem lines color-coded by cell subcategory
            for idx, cell_type in enumerate(n_types):
                if "pyramidal" in cell_type.lower():
                    color = color_map["Pyramidal"]
                elif "interneuron" in cell_type.lower():
                    color = color_map["interNeuron"]
                elif "multiunit" in cell_type.lower() or "mua" in cell_type.lower():
                    color = color_map["MultiUnit"]
                else:
                    color = "gray"  # Default color for unclassified types

                markerline, _, _ = ax.stem(
                    [x_indices[idx]],
                    [w_vector[idx]],
                    linefmt=color,
                    basefmt="gray",
                )
                plt.setp(markerline, color=color, markersize=5)

            # Draw upper/lower outlier limits
            ax.axhline(
                th_upper,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=r"2$\sigma$ Threshold",
            )
            ax.axhline(th_lower, color="red", linestyle="--", alpha=0.7)

            ax.set_xticks(x_indices)
            ax.set_xticklabels(n_labels, rotation=90, fontsize=7)

            # Highlight cells crossing the significance boundaries in red
            for idx, val in enumerate(w_vector):
                if val >= th_upper or val <= th_lower:
                    ax.get_xticklabels()[idx].set_color("red")
                    ax.get_xticklabels()[idx].set_weight("bold")

            ax.set_ylabel("Assembly Loading Weight")
            ax.set_title(
                f"{calc_type} Component Assembly Matrix Element #{t_idx + 1} ({session_key})"
            )
            ax.set_ylim(-0.6, 0.6)
            sns.despine()
            plt.tight_layout()
            plt.show()

def _compute_2d_spatial_reactivation(
    rs_tsd, pos_x, pos_y, pos_t, epoch_interval, bins=10
):
    """Helper to bin reactivation strength onto a 2D spatial maze grid."""
    rs_restricted = rs_tsd.restrict(epoch_interval)
    if len(rs_restricted) == 0:
        return np.zeros((bins, bins))

    rs_times = rs_restricted.times()
    rs_vals = rs_restricted.values

    # Interpolate x, y positions to match reactivation time bins
    interp_x = np.interp(rs_times, pos_t, pos_x)
    interp_y = np.interp(rs_times, pos_t, pos_y)

    x_edges = np.linspace(0, 1, bins + 1)
    y_edges = np.linspace(0, 1, bins + 1)

    grid = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            mask = (
                (interp_x >= x_edges[i])
                & (interp_x < x_edges[i + 1])
                & (interp_y >= y_edges[j])
                & (interp_y < y_edges[j + 1])
            )
            if np.any(mask):
                grid[i, j] = np.nanmean(rs_vals[mask])
            else:
                grid[i, j] = np.nan
    return grid

def _init_worker_plotter(cls_ref, winMS, kwargs_dict):
    """
    Initialize the worker plotter for rendering frames.
    This is used to set up the plotter in a multiprocessing context.
    """
    import matplotlib

    matplotlib.use("Agg")  # Use a non-interactive backend for rendering
    global _plotter_instance
    _plotter_instance = cls_ref.init_plotter(winMS, **kwargs_dict)
    return _plotter_instance


def _render_frame_worker(i, **kwargs):
    """
    Worker function to render a single frame in parallel.
    This is used by joblib to render frames in parallel.
    """
    global _plotter_instance
    save_path = os.path.join(_plotter_instance.output_dir, f"frame_{i:04d}.png")
    _plotter_instance.animate_frame(i, save_path=save_path, **kwargs)


def get_1d_tuning_curve(positions, mask_indices, bins=50, sigma=1.5):
    """
    Generates a smoothed 1D density curve.
    Returns: (density_values, bin_centers)
    """
    # Filter data by mask (e.g., is_freeze)
    masked_data = positions[mask_indices]

    # Calculate histogram
    counts, bin_edges = np.histogram(masked_data, bins=bins, range=(0, 1))

    # Smooth the curve
    density = gaussian_filter1d(counts.astype(float), sigma=sigma)

    # Optional: Normalize to unit area or max (unit area is better for probability)
    if np.sum(density) > 0:
        density /= np.sum(density)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return density, bin_centers


# Example usage:
if __name__ == "__main__":
    # Example DataFrame structure
    example_data = {
        "path": ["/path1/", "/path2/", "/path3/", "/path4/"],
        "name": ["Mouse245", "Mouse246", "Mouse247", "Mouse245"],
        "manipe": ["SubMFB", "SubMFB", "SubPAG", "SubMFB"],
        "group": ["LFP", "Neurons", "LFP", "ECG"],
        "Treatment": ["CNO1", "CNO2", "CNO1", "Saline"],
        "Session": ["EXT-24h", "baseline", "EXT-24h", "training"],
    }

    df = pd.DataFrame(example_data)
    print("Original DataFrame:")
    print(df)
    print("\n")

    # Test different filtering options
    try:
        # Filter by mice numbers
        result1 = restrict_path_for_experiment(df, "nMice", [245, 246])
        print("Filtered by mice:")
        print(result1[["name", "manipe"]])
        print("\n")

        # Filter by group
        result2 = restrict_path_for_experiment(df, "Group", "LFP")
        print("Filtered by group:")
        print(result2[["name", "group"]])
        print("\n")

        # Filter by treatment
        result3 = restrict_path_for_experiment(df, "Treatment", "CNO1")
        print("Filtered by treatment:")
        print(result3[["name", "Treatment"]])
        print("\n")

        # Filter by session
        result4 = restrict_path_for_experiment(df, "Session", "EXT")
        print("Filtered by session (contains 'EXT'):")
        print(result4[["name", "Session"]])
        print("\n")

        # Test merging DataFrames
        merged = merge_path_for_experiment(result1, result2)
        print("Merged DataFrames:")
        print(merged[["name", "manipe", "group"]])

    except Exception as e:
        print(f"Error: {e}")
# %% End of MOBS_Functions.py
