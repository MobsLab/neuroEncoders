#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:28:52 2020

@author: quarantine-charenton
"""

import os
import sys
from typing import Any, Dict, List, Union
from warnings import warn

import dill as pickle
import numpy as np
import pandas as pd

from resultAnalysis import print_results
from resultAnalysis.paper_figures import PaperFigures
from importData.epochs_management import inEpochs, inEpochsMask

sys.path.append(".")

from transformData.linearizer import UMazeLinearizer
from utils.PathForExperiments import path_for_experiments
from utils.global_classes import DataHelper as DataHelperClass
from utils.global_classes import Params, Project

# %% Info_LFP -> load the InfoLFP.mat file in a DataFrame with the LFPs' path


def Info_LFP(LFP_directory, Info_name="InfoLFP"):
    from os.path import join

    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
        # Dir: pd.DataFrame,
        # mouse_name: str,
        # manipe: str,
        # target: str,
        # nameExp: str,
        # phase: str = "pre",
        # full_path: str = "",
        # Extract Mouse_Results specific parameters
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
            except (FileNotFoundError, AttributeError):
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
                self.projects[winMS].folder, **kwargs
            )
            self.linearizer[winMS].verify_linearization(
                self.data_helper[winMS].positions / self.data_helper[winMS].maxPos(),
                self.projects[winMS].folder,
            )
            self.l_function = self.linearizer[winMS].pykeops_linearization
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

    def load_trainers(
        self, deviceName: str = "gpu", which="both", **kwargs
    ) -> Dict[int, Any]:
        """
        Load trainers for each window size.

        Parameters:
            deviceName (str): Device to use for training ('gpu' or 'cpu').
            **kwargs: Additional keyword arguments for trainer initialization such as:
                debug (bool): Whether to run in debug mode.
                bandwidth (int): Bandwidth for the bayes trainer.
                kernel (str): Kernel type for the bayes trainer.
                maskingFactor (float): Masking factor for the bayes trainer.


        """
        from fullEncoder.an_network import LSTMandSpikeNetwork as NNTrainer
        from simpleBayes.decode_bayes import Trainer as BayesTrainer

        if deviceName.lower() == "gpu" or deviceName.lower() == "cpu":
            from utils.management import manage_devices

            self.deviceName = manage_devices(deviceName.upper())
        else:
            self.deviceName = deviceName

        phase = kwargs.pop("phase", self.phase)
        deviceName = kwargs.pop("deviceName", self.deviceName)
        if not hasattr(self, "ann"):
            self.ann = {}
        for i, winMS in enumerate(self.windows):
            if which.lower() in ["ann", "both"]:
                if not self.ann.get(winMS):
                    self.ann[winMS] = NNTrainer(
                        self.projects[winMS],
                        self.parameters[winMS],
                        deviceName=deviceName,
                        phase=phase,
                        isTransformer=kwargs.pop(
                            "isTransformer", self.parameters[winMS].isTransformer
                        ),
                        linearizer=self.linearizer[winMS],
                        behaviorData=self.data_helper[winMS].fullBehavior,
                        alpha=self.parameters[winMS].denseweightAlpha,
                        # we dont really care about the dynamic loss, but this way we load the training data in memory, with speedMask
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

        print_results.print_results(
            self.folderResult,
            windowSizeMS=winMS,
            target=self.target,
            phase=phase,
            typeDec="NN",
            training_data=self.ann[win].training_data,
            l_function=self.l_function,
            **kwargs,
        )

    def show_movie(self, winMS=None, **kwargs):
        if winMS is None:
            win = self.windows[-1]
            winMS = self.windows_values[-1]

        idWindow = self.timeWindows.index(winMS)

        from importData.gui_elements import AnimatedPositionPlotter

        if kwargs.get("data_helper", None) is None:
            data_helper = self.data_helper[win]

        if kwargs.get("truepos", None) is None:
            truepos = self.resultsNN["truePos"][idWindow]

        if kwargs.get("predicted", None) is None:
            predicted = self.resultsNN["fullPred"][idWindow]

        if kwargs.get("speedMaskArray", None) is None:
            speedMaskArray = self.resultsNN["speedMask"][idWindow]

        plotter = AnimatedPositionPlotter(
            data_helper=data_helper,
            positions=truepos,
            predicted=predicted,
            speedMaskArray=speedMaskArray,
            **kwargs,
        )
        interval = kwargs.pop("interval", 10)
        repeat = kwargs.pop("repeat", True)
        block = kwargs.pop("block", True)
        with_ref_bg = kwargs.pop("with_ref_bg", True)
        anim = plotter.show(
            interval=interval,
            repeat=repeat,
            block=block,
            with_ref_bg=with_ref_bg,
            show=True,
            **kwargs,
        )

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
        from importData.compareSpikeFiltering import WaveFormComparator

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
