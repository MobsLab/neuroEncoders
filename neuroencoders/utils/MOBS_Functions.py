#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:28:52 2020

@author: quarantine-charenton
"""

import os
from typing import Any, Dict, List, Union
from warnings import warn

import dill as pickle
import numpy as np
import pandas as pd

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.resultAnalysis import print_results
from neuroencoders.resultAnalysis.paper_figures import PaperFigures
from neuroencoders.transformData.linearizer import UMazeLinearizer
from neuroencoders.utils.func_wrappers import timing
from neuroencoders.utils.global_classes import DataHelper as DataHelperClass
from neuroencoders.utils.global_classes import Params, Project
from neuroencoders.utils.PathForExperiments import path_for_experiments

# %% Info_LFP -> load the InfoLFP.mat file in a DataFrame with the LFPs' path


def Info_LFP(LFP_directory, Info_name="InfoLFP"):
    from os.path import join

    import pandas as pd
    from scipy.io import loadmat

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
    from pynapple import Tsd, TsdFrame
    from scipy.io import loadmat

    if type(LFP_path) == str:
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
    from pynapple import IntervalSet

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


# %% BehavResources loading


def Load_Behav(Behav_path: str, time_unit="us"):
    import pandas as pd
    from pynapple import IntervalSet, Ts, Tsd, TsdFrame
    from scipy.io import loadmat

    try:
        Behav_data = loadmat(Behav_path, squeeze_me=True)
    except FileNotFoundError:
        from os.path import join

        Behav_path = join(Behav_path, "behavResources.mat")
        Behav_data = loadmat(Behav_path, squeeze_me=True)
    keys = list(Behav_data.keys())
    keys.remove("__header__")
    keys.remove("__version__")
    keys.remove("__globals__")
    ################ Tracking information

    BehavRessources = {}

    Tracking = {}

    tsd_keys = [key for key in keys if "tsd" in key]

    for key in keys:
        if "LinearDist" in key:
            tsd_keys.append(key)

    for key in tsd_keys:
        tsd_temp = Behav_data[key]
        dat = tsd_temp["data"].tolist()
        t = tsd_temp["t"].tolist() * 100  # convert to us
        new_key = key.replace("tsd", "")
        Tracking[new_key] = Tsd(t, np.transpose(dat), time_units=time_unit)
        keys.remove(key)

    Pos_keys = [key for key in keys if "Pos" in key]

    for key in Pos_keys:
        keys.remove(key)
        Pos_temp = Behav_data[key]
        t = Pos_temp[:, 0] * 10**6
        d = Pos_temp[:, 1:4]
        Tsd_temp = TsdFrame(t, d, columns=["x", "y", "stim"], time_units=time_unit)
        # Tsd_temp["stim"] = Tsd_temp["stim"].astype(bool)
        Tracking[key] = Tsd_temp

    Im = ["im_diff", "im_diffInit"]
    for key in Im:
        if key in keys:
            keys.remove(key)
            Im_temp = Behav_data[key]
            Tracking[key] = pd.DataFrame(
                Im_temp, columns=["time", "average change", "pixel range"]
            )

    if "MouseTemp" in keys:
        keys.remove("MouseTemp")
        Temp_temp = Behav_data["MouseTemp"]
        t = Temp_temp[:, 0] * 10**6
        d = Temp_temp[:, 1]
        Tracking["MouseTemp"] = Tsd(t, d, time_units=time_unit)

    BehavRessources["Tracking"] = Tracking
    ################     Epoch information

    Epoch = {}

    Epoch_keys = [key for key in keys if "Epoch" in key]

    for key in Epoch_keys:
        keys.remove(key)
        Epoch_temp = Behav_data[key]
        new_key = key.replace("Epoch", "")
        Make_Epoch(struc=Epoch_temp, dic=Epoch, key=new_key, time_unit=time_unit)

    if Epoch:
        import re

        # create pre, hab, cond, post, sleep keys by merging Epochs
        # first for 'pre'
        epoch_keys = list(Epoch["Session"].keys())
        print(f"Available epochs: {epoch_keys}")

        # Group TestPre epochs (TestPre1, TestPre2, etc.)
        testpre_keys = [k for k in epoch_keys if re.match(r".*[Tt]est[Pp]re\d*.*", k)]
        if testpre_keys:
            Epoch["Session"]["TestPre"] = Epoch["Session"][testpre_keys[0]]
            for key in testpre_keys[1:]:
                Epoch["Session"]["TestPre"] = Epoch["Session"]["TestPre"].union(
                    Epoch["Session"][key]
                )

        # Group TestPost epochs (TestPost1, TestPost2, etc.)
        testpost_keys = [k for k in epoch_keys if re.match(r".*[Tt]est[Pp]ost\d*.*", k)]
        if testpost_keys:
            Epoch["Session"]["TestPost"] = Epoch["Session"][testpost_keys[0]]
            for key in testpost_keys[1:]:
                Epoch["Session"]["TestPost"] = Epoch["Session"]["TestPost"].union(
                    Epoch["Session"][key]
                )

        # Group Hab epochs (Hab1, Hab2, etc.)
        hab_keys = [k for k in epoch_keys if re.match(r".*[Hh]ab\d*.*", k)]
        if hab_keys:
            Epoch["Session"]["Hab"] = Epoch["Session"][hab_keys[0]]
            for key in hab_keys[1:]:
                Epoch["Session"]["Hab"] = Epoch["Session"]["Hab"].union(
                    Epoch["Session"][key]
                )

        # Group Cond epochs (Cond1, Cond2, etc.)
        cond_keys = [k for k in epoch_keys if re.match(r".*[Cc]ond\d*.*", k)]
        if cond_keys:
            Epoch["Session"]["Cond"] = Epoch["Session"][cond_keys[0]]
            for key in cond_keys[1:]:
                Epoch["Session"]["Cond"] = Epoch["Session"]["Cond"].union(
                    Epoch["Session"][key]
                )

        # Group Sleep epochs (PreSleep, PostSleep, etc.)
        sleep_keys = [k for k in epoch_keys if re.match(r".*[Ss]leep.*", k)]
        if sleep_keys:
            Epoch["Session"]["Sleep"] = Epoch["Session"][sleep_keys[0]]
            for key in sleep_keys[1:]:
                Epoch["Session"]["Sleep"] = Epoch["Session"]["Sleep"].union(
                    Epoch["Session"][key]
                )

        awake_keys = [k for k in epoch_keys if not re.match(r".*[Ss]leep.*", k)]
        if awake_keys:
            Epoch["Session"]["Awake"] = Epoch["Session"][awake_keys[0]]
            for key in awake_keys[1:]:
                Epoch["Session"]["Awake"] = Epoch["Session"]["Awake"].union(
                    Epoch["Session"][key]
                )

    BehavRessources["Epoch"] = Epoch

    ################    Other information

    Other = {}

    if "tpsCatEvt" and "nameCatEvt" in keys:
        keys.remove("tpsCatEvt")
        keys.remove("nameCatEvt")
        t = Behav_data["tpsCatEvt"]
        name = Behav_data["nameCatEvt"]
        Other["CatEvt"] = pd.DataFrame(np.transpose([t, name]), columns=["t", "name"])

    if "TTLInfo" in keys:
        keys.remove("TTLInfo")
        TTL = Behav_data["TTLInfo"]
        start = TTL["StartSession"].tolist() * 100  # convert to us
        stop = TTL["StopSession"].tolist() * 100  # convert to us
        Other["TTLInfo"] = IntervalSet(start, stop, time_units=time_unit)

    if "ThousandFrames" in keys:
        keys.remove("ThousandFrames")
        data = Behav_data["ThousandFrames"]
        TF = {}
        Nb_session = len(data)
        Session_name = list(Epoch["Session"].keys())
        for n in range(Nb_session):
            data_temp = data[n]["tsd"].tolist()
            t = data_temp["t"].tolist() * 100
            TF[Session_name[n]] = Ts(t, time_units=time_unit)
        Other["ThousandFrames"] = TF

    if "GotFrame" in keys:
        keys.remove("GotFrame")
        GF = np.transpose(Behav_data["GotFrame"].astype(bool))
        t = Tracking["X"].times()
        Other["GotFrame"] = Tsd(t, GF, time_units=time_unit)

    if ("CleanZoneIndices" or "ZoneIndices") in keys:
        ZI_keys = [key for key in keys if "ZoneIndices" in key]
        for key in ZI_keys:
            keys.remove(key)
            Z_temp = Behav_data[key]
            Z = {}
            names = list(Z_temp.dtype.fields.keys())
            for n in names:
                Z[n] = Z_temp[n].tolist()
            Other[key] = Z

    for key in keys:
        Other[key] = Behav_data[key]

    BehavRessources["Other"] = Other

    return BehavRessources


def restrict_path_for_experiment(
    Dir: Union[Dict[str, Any], pd.DataFrame],
    filter_type: str,
    filter_value: Union[str, List, int],
) -> pd.DataFrame:
    """
    Python equivalent of RestrictPathForExperiment MATLAB function.

    Restricts/filters experiment directories based on various criteria.

    Args:
        Dir: Dictionary or DataFrame containing experiment information (from path_for_experiments_erc)
        filter_type: Type of filter ('Group', 'nMice', 'Session', 'Treatment', 'all')
        filter_value: Value(s) to filter by (string, list of strings, or list of numbers)

    Returns:
        pandas DataFrame containing filtered experiment information

    Examples:
        df = restrict_path_for_experiment(Dir, 'nMice', [245, 246])
        df = restrict_path_for_experiment(Dir, 'Group', ['OBX', 'hemiOBX'])
        df = restrict_path_for_experiment(Dir, 'Session', 'EXT-24h')
        df = restrict_path_for_experiment(Dir, 'Treatment', 'CNO1')
        df = restrict_path_for_experiment(Dir, 'all', None)
    """

    # Convert input to DataFrame if it's a dictionary
    if isinstance(Dir, dict):
        df = dict_to_dataframe(Dir)
    else:
        df = Dir.copy()

    # Handle 'all' case - return original DataFrame
    if filter_type == "all" or filter_value == "all":
        return df

    # Validate inputs
    valid_filters = ["Group", "nMice", "Session", "Treatment", "all"]
    if filter_type not in valid_filters:
        raise ValueError(f"filter_type must be one of {valid_filters}")

    # Ensure filter_value is a list for consistent processing
    def ensure_list(value):
        if value is None:
            return []
        elif isinstance(value, (str, int, float)):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            return [value]

    # Process filter parameters and filter DataFrame
    if filter_type == "Group":
        filter_values = ensure_list(filter_value)
        group_str = " + ".join(map(str, filter_values))
        print(f"Getting groups {group_str} from Dir")

        # Handle different group column structures
        if "group" in df.columns:
            # Simple group column
            mask = df["group"].isin(filter_values)
            filtered_df = df[mask]
        else:
            # Check for group-specific columns (LFP, Neurons, etc.)
            group_columns = [
                col
                for col in df.columns
                if any(
                    group in str(col).lower()
                    for group in ["lfp", "neurons", "ecg", "ob_resp", "ob_gamma", "pfc"]
                )
            ]
            if group_columns:
                mask = pd.Series([False] * len(df))
                for group_col in group_columns:
                    for filter_val in filter_values:
                        mask |= df[group_col] == filter_val
                filtered_df = df[mask]
            else:
                print("No group columns found")
                filtered_df = pd.DataFrame()

    elif filter_type == "nMice":
        filter_values = ensure_list(filter_value)
        mice_str = ", ".join(map(str, filter_values))
        print(f"Getting Mice {mice_str} from Dir")

        # Format mouse names (pad with zeros to 3 digits)
        mouse_names = [f"Mouse{str(num).zfill(3)}" for num in filter_values]

        if "name" in df.columns:
            mask = df["name"].isin(mouse_names)
            filtered_df = df[mask]

            # Check for missing mice
            found_mice = df[mask]["name"].unique()
            missing_mice = [name for name in mouse_names if name not in found_mice]
            for missing in missing_mice:
                print(f"No {missing} in Dir")
        else:
            print("No 'name' column found")
            filtered_df = pd.DataFrame()

    elif filter_type == "Session":
        filter_values = ensure_list(filter_value)
        session_str = " + ".join(filter_values)
        print(f"Getting Session {session_str} from Dir")

        if "Session" in df.columns:
            mask = pd.Series([False] * len(df))
            for session_name in filter_values:
                # Use string containment for flexible matching
                session_mask = (
                    df["Session"].astype(str).str.contains(session_name, na=False)
                )
                mask |= session_mask

            filtered_df = df[mask]

            if filtered_df.empty:
                for session_name in filter_values:
                    print(f"Session {session_name} is empty")
        else:
            print("No 'Session' column found")
            filtered_df = pd.DataFrame()

    elif filter_type == "Treatment":
        filter_values = ensure_list(filter_value)
        treatment_str = " + ".join(filter_values)
        print(f"Getting Treatments {treatment_str} from Dir")

        if "Treatment" in df.columns:
            mask = df["Treatment"].isin(filter_values)
            filtered_df = df[mask]

            # Check for missing treatments
            found_treatments = df[mask]["Treatment"].unique()
            missing_treatments = [
                treat for treat in filter_values if treat not in found_treatments
            ]
            for missing in missing_treatments:
                print(f"Treatment {missing} is empty")
        else:
            print("No 'Treatment' column found")
            filtered_df = pd.DataFrame()

    # Reset index for clean output
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


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


class Mouse_Results(Params, PaperFigures):
    # (Project, Params, DataHelper, UMazeLinearizer):
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

    def __init__(self, *args, **kwargs):
        """
        Initialize the Mouse_Results class.

        Args:
        ----------
        *args: Positional arguments
            - Dir: pd.DataFrame containing the directory structure of n_experiment
            - mouse_name: str, name of the mouse (e.g., 'Mouse245')
            - manipe: str, manipulation type (e.g., 'SubMFB', 'SubPAG')
        **kwargs: Keyword arguments
            - target: str, target type (e.g., 'LFP', 'Neurons')
            - nameExp: str, name of the experiment (e.g., 'current', 'final_results', 'LossAndDirection...')
            - phase: str, phase of the experiment (default is "pre")
            - full_path: str, full path to the experiment directory (optional, if not provided it will be found automatically)
        """
        # Extract Mouse_Results specific parameters

        # FOR DEV ONLY: update inner dict with kwargs
        # skip args and kwargs that already exist in the class or are methods of the class
        for key, value in kwargs.items():
            if hasattr(self, key) or callable(getattr(self, key, None)):
                continue
            setattr(self, key, value)

        # Start parsing args
        if len(args) >= 1:
            Dir = args[0]
            args = args[1:]
        else:
            Dir = kwargs.pop("Dir", None)

        if len(args) >= 1:
            mouse_name = args[0]
            args = args[1:]
        else:
            mouse_name = kwargs.get("mouse_name", None)

        if len(args) >= 1:
            manipe = args[0]
            args = args[1:]
        else:
            manipe = kwargs.get("manipe", None)

        # Extract optional parameters
        full_path = kwargs.get("full_path", "")
        phase = kwargs.get("phase", "pre")
        nameExp = kwargs.get("nameExp", "Network")
        target = kwargs.get("target", "pos")
        if kwargs.get("deviceName", None) is not None:
            self.deviceName = kwargs["deviceName"]

        # Validate required parameters
        if (
            Dir is None
            or mouse_name is None
            or manipe is None
            or target is None
            or nameExp is None
        ):
            raise ValueError(
                "Dir, mouse_name, manipe, target, and nameExp are required"
            )

        # Initialize parameters
        self.Dir = Dir
        self.mouse_name = mouse_name
        self.manipe = manipe
        self.nameExp = nameExp
        self.target = target
        self.phase = phase
        self.which = kwargs.get("which", "all")
        if full_path == "":
            self.find_path()
        else:
            self.path = full_path

        self.find_xml()
        self.folderResult = os.path.join(self.path, self.nameExp, "results")

        # Initialize empty results DataFrame
        self.results = pd.DataFrame()

        # find all window directories in the results path
        self.find_window_size(**kwargs)
        self.parameters = dict()
        self.projects = dict()
        self.data_helper = dict()
        self.linearizer = dict()

        for i, winMS in enumerate(self.windows):
            try:
                self.projects[winMS] = Project.load(
                    os.path.join(self.path, self.nameExp, f"Project_{winMS}.pkl")
                )
                print(self.projects[winMS].folderResult)
                self.parameters[winMS] = Params.load(
                    os.path.join(self.folderResult, winMS)
                )
                # otherwise will be loaded by super init
                self.data_helper[winMS] = DataHelperClass(
                    self.projects[winMS].xml,
                    mode="compare",
                    windowSize=int(winMS) / 1000,
                    **kwargs,
                )
                assert os.path.realpath(self.projects[winMS].xml) == os.path.realpath(
                    self.xml
                )
            except (FileNotFoundError, AttributeError) as e:
                warn(
                    f"Failed to load project for window {winMS} with error: {e}. "
                    "Creating new Project and DataHelper."
                )
                # if something went wrong, create a new Project and DataHelper
                self.projects[winMS] = Project(
                    self.xml,
                    windowSize=int(winMS) / 1000,
                    **kwargs,
                )
                self.data_helper[winMS] = DataHelperClass(
                    self.xml,
                    mode="compare",
                    windowSize=int(winMS) / 1000,
                    **kwargs,
                )
                self.parameters[winMS] = Params(
                    self.data_helper[winMS],
                    windowSize=int(winMS) / 1000,
                    save_json=True,
                    **kwargs,
                )

            self.linearizer[winMS] = UMazeLinearizer(
                self.projects[winMS].folder,
                data_helper=self.data_helper[winMS],
                **kwargs,
            )
            self.linearizer[winMS].verify_linearization(
                self.data_helper[winMS].positions / self.data_helper[winMS].maxPos(),
                self.projects[winMS].folder,
            )

            if kwargs.get("keops_linearization", False):
                self.l_function = self.linearizer[winMS].pykeops_linearization
            else:

                def cpu_linearization(x):
                    return self.linearizer[winMS].apply_linearization(x, keops=False)

                self.l_function = cpu_linearization

            self.data_helper[winMS].get_true_target(
                self.l_function, in_place=True, show=kwargs.get("show", False)
            )
            if i == 0:
                # Initialize the first window as the main one
                self.DataHelper = self.data_helper[winMS]
                self.Params = self.parameters[winMS]
                self.Project = self.projects[winMS]
                self.Linearizer = self.linearizer[winMS]

                # construct from Params
                Params.__init__(
                    self,
                    helper=self.DataHelper,
                    windowSize=self.Params.windowSize,
                    **kwargs,
                )
                self.find_session_epochs()

        # Initialize PaperFigures
        if kwargs.get("load_trainers_at_init", True):
            self.load_trainers(**kwargs)
        PaperFigures.__init__(
            self,
            self.Project,
            self.DataHelper.fullBehavior,
            self.bayes if hasattr(self, "bayes") else None,
            self.l_function,
            timeWindows=self.windows_values,
            phase=self.phase,
        )
        print(self)

    def __getstate__(self):
        """
        Custom getstate method to avoid pickling issues with certain attributes.
        This is necessary for compatibility with multiprocessing and other serialization methods.
        """
        state = self.__dict__.copy()
        # Remove attributes that cannot be pickled
        state.pop("ann", None)
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
        import dill as pickle

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
            self.which = state.pop("which", "all")
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
        self.path = (
            self.Dir[
                (self.Dir.name.str.contains(self.mouse_name))
                & (self.Dir.manipe.str.contains(self.manipe))
            ]
            .iloc[0]
            .path
        )
        self.network_path = (
            self.Dir[
                (self.Dir.name.str.contains(self.mouse_name))
                & (self.Dir.manipe.str.contains(self.manipe))
            ]
            .iloc[0]
            .network_path
        )
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
            "*amplifier*.xml",
            "*.xml",
        ]:
            xml_file = next(
                (
                    os.path.join(self.path, f)
                    for f in os.listdir(self.path)
                    if f.endswith(".xml") and fnmatch.fnmatch(f, pattern)
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
                self.windows = [self.windows]
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
                bandwidth (int): Bandwidth for the bayes trainer.
                kernel (str): Kernel type for the bayes trainer.
                maskingFactor (float): Masking factor for the bayes trainer.


        """
        from neuroencoders.fullEncoder.an_network import (
            LSTMandSpikeNetwork as NNTrainer,
        )
        from neuroencoders.simpleBayes.decode_bayes import Trainer as BayesTrainer

        if hasattr(self, "deviceName"):
            deviceName = kwargs.pop("deviceName", self.deviceName)
        else:
            deviceName = kwargs.pop("deviceName", "gpu")

        if deviceName.lower() == "gpu" or deviceName.lower() == "cpu":
            from neuroencoders.utils.management import manage_devices

            self.deviceName = manage_devices(
                deviceName.upper(),
                set_memory_growth=kwargs.get("set_memory_growth", True),
            )
        else:
            self.deviceName = deviceName

        phase = kwargs.pop("phase", self.phase)
        if not hasattr(self, "ann"):
            self.ann = {}
        isTransformer = kwargs.pop("isTransformer", self.Params.isTransformer)
        transform_w_log = kwargs.pop("transform_w_log", self.Params.transform_w_log)
        denseweight = kwargs.pop("denseweight", self.Params.denseweight)

        for i, winMS in enumerate(self.windows):
            if which.lower() in ["ann", "both"]:
                if not self.ann.get(winMS):
                    self.ann[winMS] = NNTrainer(
                        self.projects[winMS],
                        self.parameters[winMS],
                        deviceName=deviceName,
                        phase=phase,
                        isTransformer=isTransformer,
                        linearizer=self.linearizer[winMS],
                        behaviorData=self.data_helper[winMS].fullBehavior,
                        alpha=self.parameters[winMS].denseweightAlpha,
                        # we dont really care about the dynamic loss, but this way we load the training data in memory, with speedMask,
                        transform_w_log=transform_w_log,
                        denseweight=denseweight,
                        **kwargs,
                    )
            if i == 0 and which.lower() in ["bayes", "both"]:
                if not hasattr(self, "bayes"):
                    self.bayes = BayesTrainer(
                        self.projects[winMS], phase=self.phase, **kwargs
                    )

    def load_results(
        self, winMS=None, redo=False, force=False, phase=None, which="both", **kwargs
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
                            self.folderResult,
                            "bayesMatrices.pkl",
                        ),
                        "rb",
                    ) as f:
                        self.bayes_matrices = pickle.load(f)
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
                                self.folderResult,
                                "bayesMatrices.pkl",
                            ),
                            "rb",
                        ) as f:
                            self.bayes_matrices = pickle.load(f)

        windows, winValues = self._select_window(winMS)
        # Load results for all windows
        for win, win_value in zip(windows, winValues):
            if which.lower() in ["ann", "both"]:
                if not redo:
                    try:
                        suffix = f"_{phase}" if phase is not None else ""
                        pos = pd.read_csv(
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
                        self.ann[win].test(
                            self.data_helper[win].fullBehavior,
                            windowSizeMS=win_value,
                            phase=phase,
                            l_function=self.l_function,
                            **kwargs,
                        )
                else:
                    print(f"Force loading results for window {win}.")
                    self.load_trainers(which="ann", **kwargs)
                    try:
                        self.ann[win].test(
                            self.data_helper[win].fullBehavior,
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
                        training_data=self.ann[win].training_data,
                        l_function=self.l_function,
                        **kwargs,
                    )
                )

            if which.lower() in ["bayes", "both"]:
                self.load_trainers(which="bayes", **kwargs)
                mask = inEpochsMask(
                    self.data_helper[win].fullBehavior["positionTime"][:, 0],
                    self.data_helper[win].fullBehavior["Times"]["testEpochs"],
                )
                if phase == "training":
                    mask += inEpochsMask(
                        self.data_helper[win].fullBehavior["positionTime"][:, 0],
                        self.data_helper[win].fullBehavior["Times"]["trainEpochs"],
                    )
                timeStepPred = self.data_helper[win].fullBehavior["positionTime"][mask]
                outputs = self.bayes.test_as_NN(
                    self.data_helper[win].fullBehavior,
                    self.bayes_matrices,
                    timeStepPred,
                    windowSizeMS=win_value,
                    l_function=self.l_function,
                    useTrain=phase == "training",
                    **kwargs,
                )
                (
                    mean_eucl_bayes,
                    select_lin_bayes,
                    mean_lin_bayes,
                    select_lin_bayes,
                ) = print_results.print_results(
                    self.folderResult,
                    typeDec="bayes",
                    results=outputs,
                    windowSizeMS=win_value,
                    target=self.target,
                    phase=phase,
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
            win = self.windows[-1]
            winMS = self.windows_values[-1]
        else:
            idx = self.windows_values.index(winMS)
            win = self.windows[idx]

        print_results.print_results(
            self.folderResult,
            windowSizeMS=winMS,
            target=self.target,
            phase=phase,
            training_data=self.ann[win].training_data,
            l_function=self.l_function,
            **kwargs,
        )

    def init_plotter(self, winMS=None, **kwargs):
        """
        Initialize the plotter for the specified window size.
        """
        if winMS is None:
            win = self.windows[-1]
            winMS = self.windows_values[-1]

        idWindow = self.timeWindows.index(winMS)
        win = self.windows[idWindow]

        phase = kwargs.get("phase", self.phase)
        phase = (
            "_" + phase if phase is not None and not phase.startswith("_") else phase
        )

        from neuroencoders.importData.gui_elements import AnimatedPositionPlotter

        data_helper = kwargs.pop("data_helper", None)
        if data_helper is None:
            data_helper = self.data_helper[win]

        positions_from_NN = kwargs.pop("positions_from_NN", None)
        if positions_from_NN is None:
            positions_from_NN = self.resultsNN_phase[phase]["truePos"][idWindow]

        predicted = kwargs.pop("predicted", None)
        if predicted is None:
            predicted = self.resultsNN_phase[phase]["fullPred"][idWindow]

        speedMaskArray = kwargs.pop("speedMaskArray", None)
        if speedMaskArray is None and kwargs.get("useSpeedMask", False):
            speedMaskArray = self.resultsNN_phase[phase]["speedMask"][idWindow]

        prediction_time = kwargs.pop("prediction_time", None)
        if prediction_time is None:
            prediction_time = self.resultsNN_phase[phase]["time"][idWindow]

        posIndex = kwargs.pop("posIndex", None)
        if posIndex is None:
            posIndex = self.resultsNN_phase[phase]["posIndex"][idWindow]

        blit = kwargs.pop("blit", True)

        plotter = AnimatedPositionPlotter(
            data_helper=data_helper,
            positions_from_NN=positions_from_NN,
            predicted=predicted,
            speedMaskArray=speedMaskArray,
            prediction_time=prediction_time,
            posIndex=posIndex,
            blit=blit,
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
        anim = plotter.show(
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
            output_dir = os.path.join(self.folderResult, winMS, "video_frames")

        os.makedirs(output_dir, exist_ok=True)

        kwargs["output_dir"] = output_dir
        kwargs["setup_plot"] = (
            True  # Ensure setup_plot is True for worker initialization
        )
        kwargs["init_animation"] = True  # Ensure animation is initialized

        init_plotter = self.init_plotter(winMS, **kwargs)
        total_frames = init_plotter.total_frames

        print("🚀 Using linear loop for rendering")

        for i in tqdm(range(total_frames), desc="Rendering frames", unit="frame"):
            save_path = os.path.join(init_plotter.output_dir, f"frame_{i:04d}.png")
            init_plotter.animate_frame(i, **kwargs, save_path=save_path)

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
                self.ann[win].train(
                    self.data_helper[win].fullBehavior,
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

    def find_session_epochs(self):
        """
        Find session epochs from the fullBehavior data.
        This method extracts the pre, hab, cond, post, and extinct epochs from the fullBehavior data.
        """
        try:
            self.pre = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["pre"]
        except KeyError:
            warn(
                "Pre epoch not found in fullBehavior. Is your Data MultiSession ? If so, there was an issue."
            )
        try:
            self.hab = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["hab"]
        except KeyError:
            pass
        try:
            self.cond = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["cond"]
        except KeyError:
            pass
        try:
            self.post = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["post"]
        except KeyError:
            pass
        try:
            self.extinct = self.DataHelper.fullBehavior["Times"]["SessionEpochs"][
                "extinct"
            ]
        except KeyError:
            pass

    def run_spike_alignment(self, **kwargs):
        """
        Run spike alignment for the mouse results.
        This method will align spikes based on the linearized positions and save the results.

        Args:
            **kwargs: Additional keyword arguments for spike alignment such as:
                force (bool): Whether to force re-alignment.
                useTraining (bool): Whether to use training data for alignment.
                sleepName (List[str]): List of sleep names to consider for alignment.
        """
        from neuroencoders.importData.compareSpikeFiltering import WaveFormComparator

        force = kwargs.get("force", False)
        useTrain = kwargs.pop("useTrain", False)

        if not hasattr(self, "waveform_comparators") or force:
            self.waveform_comparators = dict()
            for win, winValue in zip(self.windows, self.windows_values):
                self.waveform_comparators[win] = WaveFormComparator(
                    self.projects[win],
                    self.parameters[win],
                    self.data_helper[win].fullBehavior,
                    winValue,
                    phase=self.phase,
                    useTrain=useTrain,
                    **kwargs,
                )
                self.waveform_comparators[win].save_alignment_tools(
                    self.bayes, self.l_function, winValue
                )


class Results_Loader:
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
        mice_nb: List[int] = None,
        mice_manipes: List[str] = None,
        timeWindows: List[int] = None,
        phases=None,
        **kwargs,
    ):
        """
        Initialize Results_Loader with a DataFrame containing mouse results paths.

        Args:
            dir (pd.DataFrame): PathForExperiments DataFrame with columns for folder Results, mouse names, manipes, network paths, etc.
            mice_nb (List[int]): List of mouse numbers to filter results.
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
            batchSize (int): Batch size for training the ANN. Default is 64.
            transform_w_log (bool): Whether to apply a logarithmic transformation to the ann loss. Default is False.


        """
        if mice_nb is None:
            mice_nb = dir.name.str.extract(r"(\d+)").astype(int)
        if mice_manipes is None:
            mice_manipes = dir.manipe.str.extract(r"(\w+)").astype(str)
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
            for mouse_nb, manipe, mouse_full_name in zip(
                self.mice_nb, self.mice_manipes, self.mice_names
            ):
                if not any(
                    (self.Dir.name.str.contains(mouse_nb))
                    & (self.Dir.manipe.str.contains(manipe))
                ):
                    raise ValueError(
                        f"Mouse {mouse_nb} with manipe {manipe} not found in the directory."
                    )
                window_tmp = []
                path = (
                    dir[
                        dir.name.str.contains(mouse_nb)
                        & dir.manipe.str.contains(manipe)
                    ]
                    .iloc[0]
                    .path
                )
                nameExp = os.path.basename(
                    dir[
                        dir.name.str.contains(mouse_nb)
                        & dir.manipe.str.contains(manipe)
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
                    self.results_dict[nameExp][mouse_full_name][phase] = Mouse_Results(
                        dir,
                        mouse_name=mouse_nb,
                        manipe=manipe,
                        nameExp=nameExp,
                        phase=suffix.strip("_"),
                        isTransformer=isTransformer
                        if isTransformer is not None
                        else "transformer" in nameExp.lower(),
                        windows=window_tmp,
                        transform_w_log=transform_w_log
                        if transform_w_log is not None
                        else "log" in nameExp.lower(),
                        denseweight=denseweight,
                        **kwargs,
                    )
                    if phase == kwargs.get("template", "pre"):
                        self.results_dict[nameExp][mouse_full_name]["training"] = (
                            self.results_dict[nameExp][mouse_full_name][phase]
                        )
                        self.results_dict[nameExp][mouse_full_name][
                            "training"
                        ].load_data(
                            suffixes=["_training", "_" + kwargs.get("template", "pre")]
                        )

                    try:
                        self.results_dict[nameExp][mouse_full_name][phase].load_data(
                            suffixes=["_training", suffix]
                        )
                        found_training = True
                    except FileNotFoundError:
                        self.results_dict[nameExp][mouse_full_name][phase].load_data(
                            suffixes=[suffix]
                        )

        if found_training:
            self.phases.append("training")
            self.suffixes.append("_training")

        if kwargs.get("df", None) is None:
            self.convert_to_df()

    def convert_to_df(self):
        """
        Convert the results_dict to a pandas DataFrame.
        This method will create a DataFrame with the mouse names, manipes, phases, and results.
        """
        data = []
        for nameExp, mice in self.results_dict.items():
            for mouse_name, phases in mice.items():
                for phase, results in phases.items():
                    if isinstance(results, Mouse_Results):
                        for id, win in enumerate(results.windows_values):
                            prefix = "_" + phase

                            id = results.windows_values.index(win)
                            posIndex = (
                                results.resultsNN_phase[prefix]["posIndex"][
                                    id
                                ].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None
                            )
                            if posIndex is None:
                                raise ValueError(
                                    f"posIndex not found in resultss for {mouse_manipe} in {ann_mode} for phase {phase} and window {win}."
                                )
                            time_behavior = results.data_helper[str(win)].fullBehavior[
                                "positionTime"
                            ][posIndex]

                            speed = (
                                results.data_helper[str(win)]
                                .fullBehavior["Speed"]
                                .flatten()
                            )
                            aligned_speed = (
                                speed[posIndex] if posIndex is not None else None
                            )

                            asymmetry_index = results.data_helper[
                                str(win)
                            ].get_training_imbalance()

                            full_truePos_from_behavior = results.data_helper[
                                str(win)
                            ].fullBehavior["Positions"]
                            aligned_truePos_from_behavior = (
                                full_truePos_from_behavior[posIndex]
                                if posIndex is not None
                                else None
                            )
                            full_trueLinPos_from_behavior = results.l_function(
                                results.data_helper[str(win)].fullBehavior["Positions"]
                            )[1]
                            aligned_trueLinPos_from_behavior = (
                                full_trueLinPos_from_behavior[posIndex]
                                if posIndex is not None
                                else None
                            )
                            direction_from_behavior = (
                                results.data_helper[str(win)]._get_traveling_direction(
                                    full_trueLinPos_from_behavior
                                )[posIndex]
                                if posIndex is not None
                                else None
                            )
                            direction_fromNN = results.data_helper[
                                str(win)
                            ]._get_traveling_direction(
                                results.resultsNN_phase[prefix]["linTruePos"][id]
                            )
                            row = {
                                "nameExp": nameExp,
                                "mouse": mouse_name,  # if you need to split again
                                "manipe": results.manipe,
                                "phase": phase,
                                "results": results,
                                "winMS": win,
                                "asymmetry_index": asymmetry_index,
                                "fullTruePos_fromBehavior": full_truePos_from_behavior,
                                "alignedTruePos_fromBehavior": aligned_truePos_from_behavior,
                                "fullTrueLinPos_from_behavior": full_trueLinPos_from_behavior,
                                "alignedTrueLinPos_from_behavior": aligned_trueLinPos_from_behavior,
                                "fullTimeBehavior": results.data_helper[str(win)]
                                .fullBehavior["positionTime"]
                                .flatten(),
                                "alignedTimeBehavior": time_behavior,
                                "timeNN": results.resultsNN_phase[prefix]["time"][
                                    id
                                ].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "fullSpeed": speed,
                                "alignedSpeed": aligned_speed,
                                "posIndex_NN": results.resultsNN_phase[prefix][
                                    "posIndex"
                                ][id].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "speedMask": results.resultsNN_phase[prefix][
                                    "speedMask"
                                ][id].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "linPred": results.resultsNN_phase[prefix]["linPred"][
                                    id
                                ].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "fullPred": results.resultsNN_phase[prefix]["fullPred"][
                                    id
                                ]
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "truePos": results.resultsNN_phase[prefix]["truePos"][
                                    id
                                ]
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "linTruePos": results.resultsNN_phase[prefix][
                                    "linTruePos"
                                ][id].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "predLoss": results.resultsNN_phase[prefix]["predLoss"][
                                    id
                                ].flatten()
                                if hasattr(results, "resultsNN_phase")
                                else None,
                                "resultsNN": results.resultsNN
                                if hasattr(results, "resultsNN")
                                else None,
                                "direction_fromBehavior": direction_from_behavior,
                                "direction_fromNN": direction_fromNN,
                            }
                            data.append(row)
                    else:
                        raise TypeError(
                            f"Expected Mouse_Results object, got {type(results)} for mouse {mouse_name} and phase {phase}."
                        )
        self.results_df = (
            pd.DataFrame(data).sort_values(by=["mouse", "phase"]).reset_index(drop=True)
        )

        return self.results_df

    def __getitem__(self, key):
        """
        Get the results for a specific mouse name and phase.

        Args:
            key (str): Mouse name and phase in the format 'mouse_name_phase'.

        Returns:
            Mouse_Results: The Mouse_Results object for the specified mouse and phase.
        """
        mouse_name, phase = key.split("_")
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

    def save(self, path: str = None):
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
        nameExp = combined_results_dict.keys()
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
        nameExp = self.results_dict.keys()
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

    def apply_analysis(self):
        """
        Apply some usual ML operations on the results df.
        """
        self.results_df["mean_speed"] = self.results_df.apply(
            lambda row: np.nanmean(row["alignedSpeed"])
            if row["alignedSpeed"] is not None
            else np.nan,
            axis=1,
        )
        self.results_df["mean_error"] = self.results_df.apply(
            lambda row: np.nanmean(
                np.linalg.norm(row["fullPred"] - row["truePos"], axis=1)
            )
            if row["fullPred"] is not None and row["truePos"] is not None
            else None,
            axis=1,
        )
        self.results_df["lin_error"] = self.results_df.apply(
            lambda row: np.nanmean(np.abs(row["linPred"] - row["linTruePos"]))
            if row["linPred"] is not None and row["linTruePos"] is not None
            else None,
            axis=1,
        )

        # add the selected mean error and lin error to the dataframe
        # we defined the selected prediction as the prediction with the predLoss being amongs the lowest 20% for this row.
        self.results_df["predLossThreshold"] = self.results_df.apply(
            lambda row: np.quantile(row["predLoss"], 0.2)
            if row["predLoss"] is not None
            else None,
            axis=1,
        )
        # we use the predLossThreshold to select the mean_error and lin_error
        self.results_df["mean_error_selected"] = self.results_df.apply(
            lambda row: np.nanmean(
                np.linalg.norm(
                    row["fullPred"][row["predLoss"] <= row["predLossThreshold"]]
                    - row["truePos"][row["predLoss"] <= row["predLossThreshold"]],
                    axis=1,
                )
            )
            if row["fullPred"] is not None and row["truePos"] is not None
            else None,
            axis=1,
        )

        self.results_df["lin_error_selected"] = self.results_df.apply(
            lambda row: np.nanmean(
                np.abs(
                    row["linPred"][row["predLoss"] <= row["predLossThreshold"]]
                    - row["linTruePos"][row["predLoss"] <= row["predLossThreshold"]]
                )
            )
            if row["linPred"] is not None and row["linTruePos"] is not None
            else None,
            axis=1,
        )

        self.results_df["asymmetry_index_on_predicted"] = self.results_df.apply(
            lambda row: row["results"].get_training_imbalance(positions=row["fullPred"])
            if row["fullPred"] is not None
            else None,
            axis=1,
        )

        self.results_df["asymmetry_index_on_selected_predicted"] = (
            self.results_df.apply(
                lambda row: row["results"].get_training_imbalance(
                    positions=row["fullPred"][
                        row["predLoss"] <= row["predLossThreshold"]
                    ]
                )
                if row["fullPred"] is not None
                else None,
                axis=1,
            )
        )

        training_values = (
            self.results_df[self.results_df["phase"] == "training"].groupby(
                ["nameExp", "mouse", "winMS"]
            )[
                "asymmetry_index"
            ]  # or .mean(), depending on what you want if multiple rows exist
        ).first()

        self.results_df["training_asymmetry_index"] = self.results_df.set_index(
            ["nameExp", "mouse", "winMS"]
        ).index.map(training_values)

        self.results_df["real_asymmetry_ratio"] = (
            self.results_df["asymmetry_index"]
            / self.results_df["training_asymmetry_index"]
        )
        self.results_df["predicted_asymmetry_ratio"] = self.results_df.apply(
            lambda row: row["asymmetry_index_on_predicted"]
            / row["training_asymmetry_index"]
            if row["training_asymmetry_index"] != 0
            else None,
            axis=1,
        )
        self.results_df["predicted_asymmetry_ratio_on_selected"] = (
            self.results_df.apply(
                lambda row: row["asymmetry_index_on_selected_predicted"]
                / row["training_asymmetry_index"]
                if row["training_asymmetry_index"] != 0
                else None,
                axis=1,
            )
        )
        self.results_df["predicted_asymmetry_ratio_normalized"] = (
            self.results_df["asymmetry_index_on_predicted"]
            / self.results_df["real_asymmetry_ratio"]
        )
        self.results_df["selected_predicted_asymmetry_ratio_normalized"] = (
            self.results_df["asymmetry_index_on_selected_predicted"]
            / self.results_df["real_asymmetry_ratio"]
        )

        self.results_df["true_binary_direction"] = self.results_df.apply(
            lambda row: row["results"]
            .data_helper[str(row["winMS"])]
            ._get_traveling_direction(row["linTruePos"])
            if row["linTruePos"] is not None
            else None,
            axis=1,
        )
        self.results_df["predicted_binary_direction"] = self.results_df.apply(
            lambda row: row["results"]
            .data_helper[str(row["winMS"])]
            ._get_traveling_direction(row["linPred"])
            if row["linPred"] is not None
            else None,
            axis=1,
        )

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
