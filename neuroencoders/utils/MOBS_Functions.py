#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:28:52 2020

@author: quarantine-charenton
"""

import os
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from scipy.stats import pearsonr, spearmanr
from statannotations.Annotator import Annotator
from tqdm import tqdm

from neuroencoders.importData.epochs_management import inEpochsMask
from neuroencoders.resultAnalysis import print_results
from neuroencoders.resultAnalysis.paper_figures import PaperFigures
from neuroencoders.transformData.linearizer import UMazeLinearizer
from neuroencoders.utils.PathForExperiments import path_for_experiments
from neuroencoders.utils.func_wrappers import timing
from neuroencoders.utils.global_classes import DataHelper as DataHelperClass
from neuroencoders.utils.global_classes import Params, Project

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
                try:
                    self.data_helper[winMS] = DataHelperClass.load(
                        self.projects[winMS].experimentPath
                    )
                except FileNotFoundError as e:
                    print("did not manage to load DataHelper:", e)
                    self.data_helper[winMS] = DataHelperClass(
                        self.projects[winMS].xml,
                        mode="compare",
                        windowSize=int(winMS) / 1000,
                        **kwargs,
                    )
                assert os.path.realpath(self.projects[winMS].xml) == os.path.realpath(
                    self.xml
                )
            except (FileNotFoundError, AttributeError, ModuleNotFoundError) as e:
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
                # we need that before loading params to have the right target
                self.data_helper[winMS].get_true_target(in_place=True, **kwargs)
                if os.path.exists(
                    os.path.join(self.folderResult, winMS, "params.json")
                ):
                    import json

                    print(
                        f"Loading saved params from {os.path.join(self.folderResult, winMS, 'params.json')}"
                    )
                    # load the json dict and update kwargs
                    with open(
                        os.path.join(self.folderResult, winMS, "params.json"), "r"
                    ) as f:
                        saved_params = json.load(f)
                    # update kwargs with saved params, but do not overwrite existing keys in kwargs
                    params_dict = dict()
                    for key, value in saved_params.items():
                        if key not in kwargs:
                            params_dict[key] = value
                    del params_dict["windowSize"]
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
                self.l_function = self.cpu_linearization

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
            projectPath=self.Project,
            behaviorData=self.DataHelper.fullBehavior,
            trainerBayes=self.bayes if hasattr(self, "bayes") else None,
            bayesMatrices=self.bayes_matrices
            if hasattr(self, "bayes_matrices")
            else None,
            l_function=self.l_function,
            timeWindows=self.windows_values,
            phase=self.phase,
        )
        print(self)

    def cpu_linearization(self, x):
        winMS = self.windows[0]
        return self.linearizer[winMS].apply_linearization(x, keops=False)

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
        self.path = (
            self.Dir[
                (self.Dir.name.str.lower().str.contains(self.mouse_name.lower()))
                & (self.Dir.manipe.str.lower().str.contains(self.manipe.lower()))
            ]
            .iloc[0]
            .path
        )
        self.network_path = (
            self.Dir[
                (self.Dir.name.str.lower().str.contains(self.mouse_name.lower()))
                & (self.Dir.manipe.str.lower().str.contains(self.manipe.lower()))
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
                self.windows = [self.windows]

        in_dir = [
            os.path.isdir(os.path.join(self.folderResult, d)) for d in self.windows
        ]
        if not all(in_dir):
            warn(
                f"Some specified windows not found in {self.folderResult} for {self.mouse_name}:{[w for w, exists in zip(self.windows, in_dir) if not exists]}. Fixing..."
            )
            self.windows = [w for w, exists in zip(self.windows, in_dir) if exists]

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
                        **kwargs,
                    )
                    try:
                        with open(
                            os.path.join(
                                self.bayes.folderResult,
                                "bayesMatrices.pkl",
                            ),
                            "rb",
                        ) as f:
                            bayes_matrices = pickle.load(f)
                        self.bayes_matrices = bayes_matrices
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
                                self.bayes.folderResult,
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
                    print(f"Force loading ann results for window {win}.")
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
                        mask = inEpochsMask(
                            self.data_helper[win].fullBehavior["positionTime"][:, 0],
                            self.data_helper[win].fullBehavior["Times"]["testEpochs"],
                        )
                        if phase == "training":
                            mask += inEpochsMask(
                                self.data_helper[win].fullBehavior["positionTime"][
                                    :, 0
                                ],
                                self.data_helper[win].fullBehavior["Times"][
                                    "trainEpochs"
                                ],
                            )
                        timeStepPred = self.data_helper[win].fullBehavior[
                            "positionTime"
                        ][mask]
                        outputs = self.bayes.test_as_NN(
                            self.data_helper[win].fullBehavior,
                            self.bayes_matrices,
                            timeStepPred,
                            windowSizeMS=win_value,
                            l_function=self.l_function,
                            useTrain=phase == "training",
                            **kwargs,
                        )
                else:
                    print(f"Force loading bayesian results for window {win}.")
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
                    timeStepPred = self.data_helper[win].fullBehavior["positionTime"][
                        mask
                    ]
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

        idWindow = self.timeWindows.index(int(winMS))
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
        speedMaskArray_for_dim = self.resultsNN_phase[phase]["speedMask"][idWindow]

        prediction_time = kwargs.pop("prediction_time", None)
        if prediction_time is None:
            prediction_time = self.resultsNN_phase[phase]["time"][idWindow]

        posIndex = kwargs.pop("posIndex", None)
        if posIndex is None:
            posIndex = self.resultsNN_phase[phase]["posIndex"][idWindow]

        blit = kwargs.pop("blit", True)
        predicted_probs = None
        if kwargs.get("plot_heatmap", False):
            self.load_trainers(which="ann", **kwargs)
            if (
                getattr(self.ann[str(win)].params, "GaussianHeatmap", False)
                and kwargs.get("predicted_heatmap", None) is None
                and kwargs.get("plot_heatmap", False)
            ):
                try:
                    predicted_logits = self.resultsNN_phase_pkl[phase][idWindow][
                        "logits_hw"
                    ]
                except (AttributeError, KeyError, TypeError):
                    # create an empty list of size windows_values
                    self.resultsNN_phase_pkl[phase] = [None] * len(self.windows_values)
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
                        self.resultsNN_phase_pkl[phase][idWindow] = results
                        predicted_logits = self.resultsNN_phase_pkl[phase][idWindow][
                            "logits_hw"
                        ]
                    except FileNotFoundError:
                        print(
                            f"No decoding_results{phase}.pkl found for window {winMS}."
                        )
                        self.ann[str(win)].params.GaussianHeatmap = False
                        kwargs["predicted_heatmap"] = None
                        kwargs["plot_heatmap"] = False
                        predicted_probs = None
                if predicted_probs is not None:
                    try:
                        predicted_probs = (
                            self.ann[str(win)]
                            .GaussianHeatmap.decode_and_uncertainty(
                                predicted_logits, return_probs=True
                            )[-1]
                            .numpy()
                        )
                    except Exception as e:
                        self.load_trainers(which="ann")
                        predicted_probs = (
                            self.ann[str(win)]
                            .GaussianHeatmap.decode_and_uncertainty(
                                predicted_logits, return_probs=True
                            )[-1]
                            .numpy()
                        )

        predicted_heatmap = kwargs.pop("predicted_heatmap", None)
        if not kwargs.get("plot_heatmap", False):
            predicted_heatmap = None

        plotter = AnimatedPositionPlotter(
            data_helper=data_helper,
            positions_from_NN=positions_from_NN,
            predicted=predicted,
            speedMaskArray=speedMaskArray,
            prediction_time=prediction_time,
            posIndex=posIndex,
            predicted_heatmap=predicted_heatmap,
            predicted_dim_please=speedMaskArray_for_dim,
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
        anim = plotter.show(
            block=block,
            show=True,
            **kwargs,
        )

    def plot_and_save_analysis(self, winMS=None, **kwargs):
        # we want to plot the behavior during training and test epochs, to be sure we have the right behavior, and compare witht the decoding
        # this way we get a nice identity card of the quality of the data + decoding

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        from scipy.stats import pearsonr
        from pathlib import Path

        from neuroencoders.importData.epochs_management import inEpochs
        from neuroencoders.utils.viz_params import (
            TRUE_LINE_COLOR,
            PREDICTED_COLOR,
            PREDICTED_LINE_COLOR,
            DELTA_COLOR_FORWARD,
            DELTA_COLOR_REVERSE,
        )

        # ----------------------------------------------------------------------
        # INPUTS (replace these with your own arrays)
        # ----------------------------------------------------------------------
        # x_true, y_true, x_pred, y_pred, time, train_mask = ...
        # sampling_rate = 30  # Hz (or infer from np.diff(time))

        # training + test together
        if winMS is None:
            winMS = self.windows_values[-1]
        idx = self.windows_values.index(winMS)

        x_true = np.concatenate(
            [
                self.resultsNN_phase["_training"]["truePos"][idx][:, 0],
                self.resultsNN_phase["_pre"]["truePos"][idx][:, 0],
            ]
        )
        y_true = np.concatenate(
            [
                self.resultsNN_phase["_training"]["truePos"][idx][:, 1],
                self.resultsNN_phase["_pre"]["truePos"][idx][:, 1],
            ]
        )
        x_pred = np.concatenate(
            [
                self.resultsNN_phase["_training"]["fullPred"][idx][:, 0],
                self.resultsNN_phase["_pre"]["fullPred"][idx][:, 0],
            ]
        )
        y_pred = np.concatenate(
            [
                self.resultsNN_phase["_training"]["fullPred"][idx][:, 1],
                self.resultsNN_phase["_pre"]["fullPred"][idx][:, 1],
            ]
        )
        lin_true = np.concatenate(
            [
                self.resultsNN_phase["_training"]["linTruePos"][idx],
                self.resultsNN_phase["_pre"]["linTruePos"][idx],
            ]
        )
        lin_pred = np.concatenate(
            [
                self.resultsNN_phase["_training"]["linPred"][idx],
                self.resultsNN_phase["_pre"]["linPred"][idx],
            ]
        )
        time = np.concatenate(
            [
                self.resultsNN_phase["_training"]["time"][idx],
                self.resultsNN_phase["_pre"]["time"][idx],
            ]
        )
        posIndex = [
            self.resultsNN_phase["_training"]["posIndex"][idx],
            self.resultsNN_phase["_pre"]["posIndex"][idx],
        ]
        # train mask not defined based on datahelper (can change if epochs were modified after loading data)
        test_mask = (time >= min(self.resultsNN_phase["_pre"]["time"][idx])) & (
            time <= max(self.resultsNN_phase["_pre"]["time"][idx])
        )
        train_mask = ~test_mask
        speed_mask = np.concatenate(
            [
                self.resultsNN_phase["_training"]["speedMask"][idx],
                self.resultsNN_phase["_pre"]["speedMask"][idx],
            ]
        ).astype(bool)
        sampling_rate = 1 / max(0.036, int(winMS) / 4000)

        # add chance level
        random_pred = np.random.permutation(lin_true)
        chance_mae = np.mean(np.abs(random_pred - lin_true))
        chance_mae_speed = np.mean(
            np.abs(
                random_pred[train_mask & speed_mask] - lin_true[train_mask & speed_mask]
            )
        )

        # ----------------------------------------------------------------------
        # 1. BEHAVIORAL MEASURES
        # ----------------------------------------------------------------------

        # --- Speed ---
        speed_train = self.DataHelper.fullBehavior["Speed"][posIndex[0]].flatten()
        speed_test = self.DataHelper.fullBehavior["Speed"][posIndex[1]].flatten()
        speed = np.concatenate([speed_train, speed_test])

        # --- Freezing ---
        freeze_train = inEpochsMask(
            self.DataHelper.fullBehavior["positionTime"][posIndex[0]],
            self.DataHelper.fullBehavior["Times"]["FreezeEpoch"],
        )
        freeze_fraction_train = np.mean(freeze_train)
        freeze_duration_train = np.sum(
            self.DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 1]
            - self.DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 0]
        )  # total freeze duration in seconds
        freeze_duration_train /= sampling_rate  # convert to seconds

        freeze_test = inEpochsMask(
            self.DataHelper.fullBehavior["positionTime"][posIndex[1]],
            self.DataHelper.fullBehavior["Times"]["FreezeEpoch"],
        )
        freeze_fraction_test = np.mean(freeze_test)
        freeze_duration_test = np.sum(
            self.DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 1]
            - self.DataHelper.fullBehavior["Times"]["FreezeEpoch"][:, 0]
        )  # total freeze duration in seconds
        freeze_duration_test /= sampling_rate  # convert to seconds
        freeze_duration_total = freeze_duration_train + freeze_duration_test

        # ----------------------------------------------------------------------
        # 1b. Reordering in time and removing "false" train epochs (ie test epochs)
        # ----------------------------------------------------------------------
        time_order = np.argsort(np.concatenate([posIndex[0], posIndex[1]]).flatten())
        x_true = x_true[time_order]
        y_true = y_true[time_order]
        x_pred = x_pred[time_order]
        y_pred = y_pred[time_order]
        lin_true = lin_true[time_order]
        lin_pred = lin_pred[time_order]
        time = time[time_order]
        train_mask = train_mask[time_order]
        test_mask = test_mask[time_order]
        speed_mask = speed_mask[time_order]
        speed = speed[time_order]

        # --- Occupancy heatmap (true behavior) ---
        # look only for test set to avoid train bias
        bins = 40
        H_true, xedges, yedges = np.histogram2d(
            x_true[test_mask & speed_mask], y_true[test_mask & speed_mask], bins=bins
        )
        H_true_smooth = gaussian_filter(H_true, sigma=1)
        H_true_norm = H_true_smooth / np.sum(H_true_smooth)

        # ----------------------------------------------------------------------
        # 2. DECODING METRICS
        # ----------------------------------------------------------------------

        # --- Decoding error ---
        # either euclidean
        error = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)
        # or linearized

        error = np.abs(lin_pred - lin_true)
        mae_train = np.mean(error[speed_mask & train_mask])
        mae_train_raw = np.mean(error[train_mask])
        mae_test = np.mean(error[speed_mask & test_mask])
        mae_test_raw = np.mean(error[test_mask])

        # Rolling MAE (5 s window)
        win = int(5 * sampling_rate)
        rolling_mae = np.convolve(error, np.ones(win) / win, mode="same")

        # --- Spatial error heatmap ---
        H_err_sum = np.zeros_like(H_true, dtype=float)
        H_count = np.zeros_like(H_true, dtype=float)

        # assign each sample to a bin
        x_bin = np.digitize(x_true[test_mask & speed_mask], xedges) - 1
        y_bin = np.digitize(y_true[test_mask & speed_mask], yedges) - 1

        valid = (x_bin >= 0) & (x_bin < bins) & (y_bin >= 0) & (y_bin < bins)
        for xb, yb, e in zip(
            x_bin[valid], y_bin[valid], error[test_mask & speed_mask][valid]
        ):
            H_err_sum[xb, yb] += e
            H_count[xb, yb] += 1

        H_err_mean = np.divide(H_err_sum, H_count, where=H_count > 0)
        H_err_smooth = gaussian_filter(H_err_mean, sigma=1)

        # --- Occupancy map from decoded position ---
        H_decoded, _, _ = np.histogram2d(
            x_pred[test_mask & speed_mask],
            y_pred[test_mask & speed_mask],
            bins=[xedges, yedges],
        )
        H_decoded_smooth = gaussian_filter(H_decoded, sigma=1)
        H_decoded_norm = H_decoded_smooth / np.sum(H_decoded_smooth)

        # --- Spatial correlation between true and decoded occupancy ---
        mask_nonzero = (H_true_norm > 0) | (H_decoded_norm > 0)
        r_occ, _ = pearsonr(
            H_true_norm[mask_nonzero].flatten(), H_decoded_norm[mask_nonzero].flatten()
        )

        # ----------------------------------------------------------------------
        # 3. PLOTTING
        # ----------------------------------------------------------------------

        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        ax = axes.ravel()

        # (A) Occupancy heatmap (true)
        im0 = ax[0].imshow(
            H_true_norm.T,
            origin="lower",
            cmap="hot",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax[0].set_title("Occupancy (True)")
        fig.colorbar(im0, ax=ax[0], label="Occupancy probability")

        # (B) Speed distribution
        ax[1].hist(
            speed[train_mask],
            bins=30,
            label="Train",
            alpha=0.7,
            density=True,
            color="green",
        )
        ax[1].hist(
            speed[test_mask],
            bins=30,
            label="Test",
            alpha=0.7,
            density=True,
            color="orange",
        )
        ax[1].set_title("Speed Distribution")
        ax[1].set_xlabel("Speed (cm/s)")
        ax[1].set_ylabel("Density")
        ax[1].grid(True, alpha=0.3)
        # removed per-axis legend here

        # (C) Freezing fraction
        ax[2].bar(
            ["Train", "Test"],
            [freeze_fraction_train, freeze_fraction_test],
            color=["green", "orange"],
        )
        ax[2].set_title("Freezing Fraction")
        ax[2].set_ylabel("Proportion of time frozen")
        ax[2].set_ylim(0, 1)

        # (D) Error over time with speed mask - + speed on a twin axis
        ax[3].plot(
            time[speed_mask],
            error[speed_mask],
            lw=0.5,
            alpha=0.6,
            label="Instantaneous",
        )
        ax[3].plot(
            time[speed_mask], rolling_mae[speed_mask], lw=2, label="Rolling MAE (5s)"
        )
        ax[3].axvspan(
            min(time[test_mask]),
            max(time[test_mask]),
            color="orange",
            alpha=0.5,
            label="Test",
        )
        ax[3].set_title("Decoding Error Over Time (With Speed Mask)")
        ax[3].set_ylabel("Error (lin. dist.)")
        # removed per-axis legend here
        # add chance level as a horizontal line
        ax[3].axhline(
            chance_mae_speed,
            color="red",
            linestyle="--",
            lw=1,
            label="Chance level on movement",
        )

        # (E) Error over time without speed mask
        ax[4].plot(time, error, lw=0.5, alpha=0.6, label="Instantaneous")
        ax[4].plot(time, rolling_mae, lw=2, label="Rolling MAE (5s)")
        ax[4].axvspan(
            min(time[test_mask]),
            max(time[test_mask]),
            color="orange",
            alpha=0.5,
            label="Test",
        )
        ax[4].set_title("Decoding Error Over Time (No Speed Mask)")
        ax[4].set_ylabel("Error (lin. dist.)")
        # removed per-axis legend here
        # add chance level as a horizontal line
        ax[4].axhline(
            chance_mae, color="red", linestyle="--", lw=1, label="Chance level"
        )

        # (F) Spatial decoding error heatmap
        im2 = ax[5].imshow(
            H_err_smooth.T,
            origin="lower",
            cmap="coolwarm",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax[5].set_title("Mean Spatial Decoding Error (lin. dist.)")
        fig.colorbar(im2, ax=ax[5], label="Error (lin. dist.)")

        # (G) Occupancy map (decoded)
        im3 = ax[6].imshow(
            H_decoded_norm.T,
            origin="lower",
            cmap="hot",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax[6].set_title("Occupancy (Decoded)")
        fig.colorbar(im3, ax=ax[6], label="Occupancy probability")

        # (H) True vs Decoded Occupancy comparison
        ax[7].scatter(H_true_norm.flatten(), H_decoded_norm.flatten(), s=8, alpha=0.5)
        ax[7].plot(
            [0, max(H_true_norm.max(), H_decoded_norm.max())],
            [0, max(H_true_norm.max(), H_decoded_norm.max())],
            "k--",
            lw=1,
        )
        ax[7].set_xlabel("True occupancy")
        ax[7].set_ylabel("Decoded occupancy")
        ax[7].set_title(f"Occupancy correlation r = {r_occ:.2f}")

        # (I) Summary metrics + legend consolidation
        text = (
            f"Train MAE: {mae_train:.2f} lin. dist. (~{mae_train * 100:.1f} cm) vs {mae_train_raw:.2f} with immobility.\n"
            f"Test MAE: {mae_test:.2f} lin. dist. (~{mae_test * 100:.1f} cm) vs {mae_test_raw:.2f} with immobility.\n"
            f"Freeze duration: {freeze_duration_total:.1f}s\n"
            f"Occupancy corr (r): {r_occ:.2f}\n"
            f"Train freeze: {freeze_fraction_train * 100:.1f}%\n"
            f"Test freeze: {freeze_fraction_test * 100:.1f}%"
        )
        ax[8].text(0.02, 0.5, text, fontsize=12, va="center", wrap=True)

        ax[8].axis("off")
        ax[8].set_title("Summary Metrics")

        # the last subplot line is an ax that spans the whole row for better layout
        # it just shows an example of true and decoded trajectory in the linear space
        # first remove the original axis
        ax[9].remove()
        ax[10].remove()
        ax[11].remove()
        ax[9] = fig.add_subplot(4, 1, 4)
        ax[9].plot(
            time[test_mask], lin_true[test_mask], alpha=0.7, color=TRUE_LINE_COLOR
        )
        ax[9].plot(
            time[test_mask],
            lin_pred[test_mask],
            "--",
            alpha=0.5,
            color=PREDICTED_LINE_COLOR,
        )
        ax[9].scatter(
            time[test_mask & speed_mask],
            lin_pred[test_mask & speed_mask],
            label="Decoded Position on movement",
            s=20,
            alpha=0.7,
            color=DELTA_COLOR_FORWARD,
            marker="o",
        )
        ax[9].scatter(
            time[test_mask & ~speed_mask],
            lin_pred[test_mask & ~speed_mask],
            label="Decoded Position while immobile",
            s=6,
            alpha=0.7,
            color=DELTA_COLOR_REVERSE,
            marker="o",
        )
        ax[9].set_xlabel("Time (s)")
        ax[9].set_ylabel("Linearized Position")
        ax[9].set_title("Performance on test set")

        # Collect handles & labels from axes that used legends and put a single legend box into ax[8]
        source_axes = [ax[1], ax[3], ax[4], ax[9]]
        handles_map = {}
        for a in source_axes:
            h, l = a.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in handles_map:  # keep first handle for each unique label
                    handles_map[ll] = hh
        if len(handles_map) > 0:
            unique_labels = list(handles_map.keys())
            unique_handles = [handles_map[k] for k in unique_labels]
            # place legend on the right side of ax[8] so it doesn't overlap the text
            # + move it even more to the right
            legend = ax[8].legend(
                unique_handles,
                unique_labels,
                loc="lower left",
                bbox_to_anchor=(0.7, -0.5),
            )
            ax[8].add_artist(legend)
        fig.suptitle(
            f"Decoding Analysis for Mouse {self.mouse_name} - Experiment: {self.nameExp}"
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                Path.home(),
                "download",
                "tmp_results",
                f"decoding_analysis_mouse_{self.mouse_name}_{self.nameExp}_{winMS}.png",
            ),
            dpi=300,
        )
        if kwargs.get("show", True):
            plt.show()
        plt.close(fig)

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
                f"mouse_{self.mouse_name}_win_{winMS}_phase_{self.phase}.mp4",
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
            self.preMask = inEpochsMask(
                self.DataHelper.fullBehavior["positionTime"][:, 0], self.pre
            )
        except KeyError:
            warn(
                "Pre epoch not found in fullBehavior. Is your Data MultiSession ? If so, there was an issue."
            )
        try:
            self.hab = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["hab"]
            self.habMask = inEpochsMask(
                self.DataHelper.fullBehavior["positionTime"][:, 0], self.hab
            )
        except KeyError:
            pass
        try:
            self.cond = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["cond"]
            self.condMask = inEpochsMask(
                self.DataHelper.fullBehavior["positionTime"][:, 0], self.cond
            )
        except KeyError:
            pass
        try:
            self.post = self.DataHelper.fullBehavior["Times"]["SessionEpochs"]["post"]
            self.postMask = inEpochsMask(
                self.DataHelper.fullBehavior["positionTime"][:, 0], self.post
            )
        except KeyError:
            pass
        try:
            self.extinct = self.DataHelper.fullBehavior["Times"]["SessionEpochs"][
                "extinct"
            ]
            self.extinctMask = inEpochsMask(
                self.DataHelper.fullBehavior["positionTime"][:, 0], self.extinct
            )
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

    def convert_to_df(self, redo=False):
        if (
            hasattr(self, "results_df")
            and not redo
            and isinstance(self.results_df, pd.DataFrame)
        ):
            print("Results DataFrame already exists. Use redo=True to recreate it.")
            return self.results_df

        # Pre-check to avoid repeated hasattr calls
        has_resultsNN = hasattr(self, "resultsNN_phase")
        has_resultsNN_obj = hasattr(self, "resultsNN")
        has_bayes = hasattr(self, "resultsBayes") and "fullPred" in self.resultsBayes

        if not has_resultsNN:
            raise ValueError("resultsNN_phase not found in results")

        data = []
        total_iterations = len(self.suffixes) * len(self.windows_values)

        with tqdm(total=total_iterations, desc=f"Converting {self.mouse_name}") as pbar:
            for suffix in self.suffixes:
                # Pre-strip suffix once
                phase_name = suffix.strip("_") if suffix else "all"

                for id, win in enumerate(self.windows_values):
                    win_str = str(win)
                    data_helper_win = self.data_helper[win_str]
                    resultsNN_suffix = self.resultsNN_phase[suffix]

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

                    linTruePos = resultsNN_suffix["linTruePos"][id].flatten()
                    direction_fromNN = data_helper_win._get_traveling_direction(
                        linTruePos
                    )

                    # Build row dictionary
                    row = {
                        "nameExp": self.nameExp,
                        "mouse": self.mouse_name,
                        "manipe": self.manipe,
                        "phase": phase_name,
                        "winMS": win,
                        "asymmetry_index": data_helper_win.get_training_imbalance(),
                        "fullTruePos_fromBehavior": full_truePos_from_behavior,
                        "alignedTruePos_fromBehavior": full_truePos_from_behavior[
                            posIndex
                        ],
                        "fullTrueLinPos_from_behavior": full_trueLinPos_from_behavior,
                        "alignedTrueLinPos_from_behavior": full_trueLinPos_from_behavior[
                            posIndex
                        ],
                        "fullTimeBehavior": fullBehavior["positionTime"].flatten(),
                        "alignedTimeBehavior": fullBehavior["positionTime"][posIndex],
                        "timeNN": resultsNN_suffix["time"][id].flatten(),
                        "fullSpeed": speed,
                        "alignedSpeed": speed[posIndex - 1],  # -1 for shift
                        "posIndex_NN": posIndex,
                        "speedMask": resultsNN_suffix["speedMask"][id].flatten(),
                        "linPred": resultsNN_suffix["linPred"][id].flatten(),
                        "fullPred": resultsNN_suffix["fullPred"][id],
                        "truePos": resultsNN_suffix["truePos"][id],
                        "linTruePos": linTruePos,
                        "predLoss": resultsNN_suffix["predLoss"][id].flatten(),
                        "resultsNN": self.resultsNN if has_resultsNN_obj else None,
                        "direction_fromBehavior": direction_from_behavior,
                        "direction_fromNN": direction_fromNN,
                    }

                    # Add Bayesian results if available
                    if has_bayes:
                        resultsBayes_suffix = self.resultsBayes_phase[suffix]
                        row["bayesPred"] = resultsBayes_suffix["fullPred"][id]
                        row["bayesLinPred"] = resultsBayes_suffix["linPred"][
                            id
                        ].flatten()
                        row["bayesProba"] = resultsBayes_suffix["predLoss"][
                            id
                        ].flatten()

                    data.append(row)
                    pbar.update(1)

        self.results_df = pd.DataFrame(data)
        return self.results_df


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
        mice_nb: Optional[List[int]] = None,
        mice_manipes: Optional[List[str]] = None,
        timeWindows: Optional[List[int]] = None,
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
                mouse_nb = str(mouse_nb)
                if not any(
                    (self.Dir.name.str.lower().str.contains(mouse_nb.lower()))
                    & (self.Dir.manipe.str.lower().str.contains(manipe.lower()))
                ):
                    raise ValueError(
                        f"Mouse {mouse_nb} with manipe {manipe} not found in the directory."
                    )
                window_tmp = []
                path = (
                    self.Dir[
                        (self.Dir.name.str.lower().str.contains(mouse_nb.lower()))
                        & (self.Dir.manipe.str.lower().str.contains(manipe.lower()))
                    ]
                    .iloc[0]
                    .path
                )
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

                    try:
                        self.results_dict[nameExp][mouse_full_name][phase].load_data(
                            suffixes=["_training", suffix]
                            if phase == kwargs.get("template", "pre")
                            else [suffix]
                        )
                        found_training = True
                        if kwargs.get("load_bayes", False) or kwargs.get(
                            "which", "ann"
                        ) in ["both", "bayes"]:
                            self.results_dict[nameExp][mouse_full_name][
                                phase
                            ].load_bayes(
                                suffixes=["_training", suffix]
                                if phase == kwargs.get("template", "pre")
                                else [suffix],
                                **kwargs,
                            )
                    except FileNotFoundError:
                        self.results_dict[nameExp][mouse_full_name][phase].load_data(
                            suffixes=[suffix]
                        )
                        if kwargs.get("load_bayes", False) or kwargs.get(
                            "which", "ann"
                        ) in ["both", "bayes"]:
                            self.results_dict[nameExp][mouse_full_name][
                                phase
                            ].load_bayes(suffixes=[suffix], **kwargs)

        if found_training:
            self.phases.append("training")
            self.suffixes.append("_training")

        if kwargs.get("df", None) is None:
            try:
                self.convert_to_df()
            except Exception as e:
                print(f"Issues converting to DataFrame: {e}")

    def convert_to_df(self, redo=False):
        """
        Convert the results_dict to a pandas DataFrame.
        This method will create a DataFrame with the mouse names, manipes, phases, and results.
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
            len([p for p in phases.keys() if p != "training"])
            for mice in self.results_dict.values()
            for phases in mice.values()
        )

        # Pre-allocate list for better performance
        data_list = []

        # Create progress bar
        with tqdm(total=total_iterations, desc="Processing results") as pbar:
            for nameExp, mice in self.results_dict.items():
                for mouse_name, phases in mice.items():
                    for phase, results in phases.items():
                        if phase == "training":
                            continue

                        results_df = results.convert_to_df()
                        # Add metadata columns
                        results_df["results"] = results
                        results_df["nameExp"] = nameExp
                        results_df["mouse_name"] = mouse_name

                        # Append to list instead of concatenating
                        data_list.append(results_df)
                        pbar.update(1)

        # Single concatenation at the end (much faster)
        data = pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()
        self.results_df = data.sort_values(by=["mouse", "phase"]).reset_index(drop=True)

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
        Apply some usual ML operations on the results df.
        """
        if "mean_speed" in self.results_df.columns:
            print("Analysis already applied to the DataFrame.")
            if not redo:
                return self.results_df

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
        self.results_df["error"] = self.results_df.apply(
            lambda row: np.linalg.norm(row["fullPred"] - row["truePos"], axis=1)
            if row["fullPred"] is not None and row["truePos"] is not None
            else None,
            axis=1,
        )
        self.results_df["mean_lin_error"] = self.results_df.apply(
            lambda row: np.nanmean(np.abs(row["linPred"] - row["linTruePos"]))
            if row["linPred"] is not None and row["linTruePos"] is not None
            else None,
            axis=1,
        )
        self.results_df["lin_error"] = self.results_df.apply(
            lambda row: np.abs(row["linPred"] - row["linTruePos"])
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
            lambda row: np.abs(
                row["linPred"][row["predLoss"] <= row["predLossThreshold"]]
                - row["linTruePos"][row["predLoss"] <= row["predLossThreshold"]]
            )
            if row["linPred"] is not None and row["linTruePos"] is not None
            else None,
            axis=1,
        )
        self.results_df["mean_lin_error_selected"] = self.results_df.apply(
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

    def mean_error_matrix_linerrors_by_speed(
        self,
        nbins=40,
        normalized=True,
        save=True,
        folder=None,
        show=False,
        nameExp_list=None,
        phase_list=None,
        winMS_list=None,
    ):
        """
        Plot error matrices (2D histograms) of predicted vs. true linear position,
        split by speed, across experiments, phases, and time windows.

        Args:
            nbins (int): Number of bins for the 2D histogram.
            normalized (bool): Whether to normalize rows of the histogram to [0,1].
            save (bool): If True, saves figures to disk.
            folder (str): Folder to save figures in (defaults to self.folderFigures).
        """

        folder = folder or getattr(self, "folderFigures", None)
        grouped = self.results_df.groupby(["nameExp", "phase", "winMS"])
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
            # assert winMS_list is only int
            winMS_list = [int(w) for w in winMS_list]

            grouped = grouped.filter(lambda x: int(x.name[2]) in winMS_list).groupby(
                ["nameExp", "phase", "winMS"]
            )

        for (nameExp, phase, winMS), df in grouped:
            fig, axes = plt.subplots(
                ncols=2, nrows=1, figsize=(10, 5), sharex=True, sharey=True
            )
            # -------- Fast speeds --------
            # Assumes "speedMask" is stored per row in the df
            linPred_fast = []
            linTrue_fast = []
            for _, row in df.iterrows():
                # get speed_mask from training Mouse_Results object
                mouse_val = row["mouse"]
                mouse_manipe = row["manipe"]
                speed_mask = (
                    self.results_df.query(
                        "nameExp == @nameExp and phase == 'training' and winMS == @winMS and mouse == @mouse_val and manipe == @mouse_manipe"
                    )["results"]
                    .values[0]
                    .data_helper[str(winMS)]
                    .fullBehavior["Times"]["speedFilter"]
                    .flatten()[row["posIndex_NN"]]
                )

                if phase == "training":
                    # remove the last bit that is actually not training
                    real_train = (
                        self.results_df.query(
                            "nameExp == @nameExp and phase == 'training' and winMS == @winMS and mouse == @mouse_val and manipe == @mouse_manipe"
                        )["results"]
                        .values[0]
                        .data_helper[str(winMS)]
                        .fullBehavior["Times"]["trainEpochs"]
                    )

                    epochMask = inEpochsMask(row["timeNN"], real_train)
                else:
                    epochMask = np.ones_like(row["timeNN"], dtype=bool)

                mask = speed_mask & epochMask

                linPred_fast.append(row["linPred"][mask])
                linTrue_fast.append(row["linTruePos"][mask])

            if linPred_fast:  # check non-empty
                H, xedges, yedges = np.histogram2d(
                    np.concatenate(linPred_fast).reshape(-1),
                    np.concatenate(linTrue_fast).reshape(-1),
                    bins=(nbins, nbins),
                    density=True,
                )
                if normalized:
                    with np.errstate(invalid="ignore"):
                        H = H / H.max(axis=1, keepdims=True)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                ax_fast = axes[0]
                ax_fast.set_xlim(0, 1)
                ax_fast.set_ylim(0, 1)
                im = ax_fast.imshow(
                    H.T,
                    extent=extent,
                    cmap="viridis",
                    interpolation="none",
                    origin="lower",
                    aspect="auto",
                )
                fig.colorbar(im, ax=ax_fast)
                ax_fast.set_title("Fast speeds only")

            # -------- Slow speeds --------
            # Assumes "speedMask" is stored per row in the df
            linPred = []
            linTrue = []
            for _, row in df.iterrows():
                # get speed_mask from training Mouse_Results object
                mouse_val = row["mouse"]
                mouse_manipe = row["manipe"]
                speed_mask = (
                    self.results_df.query(
                        "nameExp == @nameExp and phase == 'training' and winMS == @winMS and mouse == @mouse_val and manipe == @mouse_manipe"
                    )["results"]
                    .values[0]
                    .data_helper[str(winMS)]
                    .fullBehavior["Times"]["speedFilter"]
                    .flatten()[row["posIndex_NN"]]
                )

                if phase == "training":
                    # remove the last bit that is actually not training
                    real_train = (
                        self.results_df.query(
                            "nameExp == @nameExp and phase == 'training' and winMS == @winMS and mouse == @mouse_val and manipe == @mouse_manipe"
                        )["results"]
                        .values[0]
                        .data_helper[str(winMS)]
                        .fullBehavior["Times"]["trainEpochs"]
                    )

                    epochMask = inEpochsMask(row["timeNN"], real_train)
                else:
                    epochMask = np.ones_like(row["timeNN"], dtype=bool)

                mask = ~speed_mask & epochMask

                linPred.append(row["linPred"][mask])
                linTrue.append(row["linTruePos"][mask])

            H, xedges, yedges = np.histogram2d(
                np.concatenate(linPred).reshape(-1),
                np.concatenate(linTrue).reshape(-1),
                bins=(nbins, nbins),
                density=True,
            )
            if normalized:
                with np.errstate(invalid="ignore"):
                    H = H / H.max(axis=1, keepdims=True)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            ax_all = axes[1]
            ax_all.set_xlim(0, 1)
            ax_all.set_ylim(0, 1)
            im = ax_all.imshow(
                H.T,
                extent=extent,
                cmap="viridis",
                interpolation="none",
                origin="lower",
                aspect="auto",
            )
            fig.colorbar(im, ax=ax_all)
            ax_all.set_title("Slow speeds only")

            # -------- Labels and layout --------
            fig.suptitle(f"{nameExp} | Phase: {phase} | winMS: {winMS}")
            fig.text(0.5, 0.04, "Predicted linPos", ha="center")
            fig.text(0.04, 0.5, "True linPos", va="center", rotation="vertical")
            fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])

            if save and folder is not None:
                fname = f"errorMatrix_{nameExp}_phase{phase}_win{winMS}"
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
        try:
            loss_layer = (
                self.results_df["results"][0]
                .ann[str(self.timeWindows[0])]
                .GaussianLoss_layer
            )
            logits_layer = (
                self.results_df["results"][0]
                .ann[str(self.timeWindows[0])]
                .GaussianHeatmap
            )
        except Exception:
            print("Trying to load ANN trainers...")
            try:
                self.results_df["results"][0].load_trainers(which="ann")
                loss_layer = (
                    self.results_df["results"][0]
                    .ann[str(self.timeWindows[0])]
                    .GaussianLoss_layer
                )
                logits_layer = (
                    self.results_df["results"][0]
                    .ann[str(self.timeWindows[0])]
                    .GaussianHeatmap
                )
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
                kl_loss = loss_layer(inputs, return_batch=True).numpy().flatten()

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
        z_var_cmap="time",
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
        try:
            loss_layer = (
                self.results_df["results"][0]
                .ann[str(self.timeWindows[0])]
                .GaussianLoss_layer
            )
            logits_layer = (
                self.results_df["results"][0]
                .ann[str(self.timeWindows[0])]
                .GaussianHeatmap
            )
        except Exception:
            print("Trying to load ANN trainers...")
            self.results_df["results"][0].load_trainers(which="ann")
            loss_layer = (
                self.results_df["results"][0]
                .ann[str(self.timeWindows[0])]
                .GaussianLoss_layer
            )
            logits_layer = (
                self.results_df["results"][0]
                .ann[str(self.timeWindows[0])]
                .GaussianHeatmap
            )

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
                    kl_loss = loss_layer(inputs, return_batch=True).numpy().flatten()

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

    def around_ripples(
        self,
        suffixes=None,
        against="entropy",
        around=0.5,  # seconds before and after ripple
        dt=0.01,  # time bin resolution in seconds
        save=True,
        folder=None,
        show=False,
        zscore=True,  # z-score per mouse relative to full session
    ):
        """
        Compute and plot METAverage around ripples.

        For each suffix:
            - Select rows with phase == f"_{suffix}"
            - Group by (nameExp, phase, winMS)
            - Load decoding_results.pkl for each row
            - Load ripple times from Mouse_Results
            - Compute z-score evolution of entropy/maxp around ripples
            - Aggregate across mice: mean ± std
        """

        folder = folder or getattr(self, "folderFigures", None)
        suffixes = suffixes or getattr(self, "suffixes", [""])

        for suffix in suffixes:
            suffix_tag = suffix.strip("_")
            print(f"\nProcessing suffix: {suffix} (tag: {suffix_tag})")
            grouped = self.results_df.query("phase == @suffix_tag").groupby(
                ["nameExp", "phase", "winMS"]
            )

            for (nameExp, phase, winMS), df in grouped:
                if df.empty:
                    continue

                # Define common time vector around ripple
                time_vec = np.arange(-around, around + dt, dt)
                values_matrix = []  # rows: mice × columns: time bins

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

                    # --- extract values ---
                    if against == "entropy":
                        values = decoding_results["Hn"].flatten()
                    elif against == "maxp":
                        values = decoding_results["maxp"].flatten()
                    else:
                        raise ValueError("against must be 'entropy' or 'maxp'")

                    times = decoding_results["times"].flatten()

                    tRipples = (
                        mouse_results.data_helper[str(winMS)]
                        .fullBehavior["Times"]
                        .get("tRipples", None)
                    )
                    if tRipples is None or len(tRipples) == 0:
                        print(
                            f"No ripple times for mouse {mouse_results.mouse_name}, winMS {winMS}"
                        )
                        continue

                    # --- extract peri-ripple windows ---
                    peri_values_all_ripples = []
                    for tr in tRipples:
                        mask = (times >= tr - around) & (times <= tr + around)
                        if mask.any():
                            peri_times = times[mask] - tr
                            # interpolate values onto common time vector
                            interp_values = np.interp(
                                time_vec, peri_times, values[mask]
                            )
                            peri_values_all_ripples.append(interp_values)

                    if len(peri_values_all_ripples) == 0:
                        continue

                    # average across ripples for this mouse
                    mouse_mean = np.mean(peri_values_all_ripples, axis=0)
                    if zscore:
                        # z-score relative to full session
                        mouse_mean = (mouse_mean - np.mean(values)) / np.std(values)
                        values_matrix.append(mouse_mean)

                    del decoding_results

                if len(values_matrix) == 0:
                    print(
                        f"No valid ripple data for {nameExp}, phase {phase}, winMS {winMS}"
                    )
                    continue

                # --- METAverage across mice ---
                values_matrix = np.vstack(values_matrix)
                mean_trace = np.mean(values_matrix, axis=0)
                std_trace = np.std(values_matrix, axis=0)

                # --- plot ---
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(time_vec, mean_trace, color="blue", lw=2)
                ax.fill_between(
                    time_vec,
                    mean_trace - std_trace,
                    mean_trace + std_trace,
                    color="blue",
                    alpha=0.3,
                )
                ax.axvline(0, color="black", linestyle="--", lw=1)
                ax.set_xlabel("Time around ripple (s)")
                ax.set_ylabel(f"Z-scored {against}")
                ax.set_title(f"{nameExp} | phase: {phase} | winMS: {winMS}")
                fig.tight_layout()

                if save and folder is not None:
                    fname = (
                        f"bof_METAverage_{suffix_tag}_{nameExp}_win{winMS}_{against}"
                    )
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
                    tRipples = (
                        mouse_results.data_helper[str(winMS)]
                        .fullBehavior["Times"]
                        .get("tRipples", None)
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
                    fname = f"METAverage_{suffix_tag}_{nameExp}_win{winMS}_{against}_zscored"
                    fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
                    fig.savefig(os.path.join(folder, fname + ".svg"))
                if show:
                    plt.show()
                plt.close(fig)

    def correlation_per_mouse(
        self,
        against="entropy",  # "entropy" or "maxp"
        error_type="lin",  # "lin" for linear error
        mode="full",  # "selected" or "full"
        speed="all",  # "fast", "slow", "all"
        suffixes=None,  # list of suffixes to select (phase)
        nameExps=None,  # list of nameExp to include
        winMS_list=None,  # list of winMS to include
        save=True,
        folder=None,
        show=False,
        zscore=False,  # whether to z-score values before correlation
    ):
        """
        Plot correlation of averaged per-mouse values between decoder confidence (entropy/maxp)
        and linear error, optionally filtered by suffix, nameExp, and winMS.

        Each mouse contributes one point per (phase/suffix) averaged across selected winMS.

        Args:
            against (str): "entropy" or "maxp".
            error_type (str): "lin" or other error type.
            mode (str): "selected" or "full".
            speed (str): "fast", "slow", "all".
            suffixes (list[str]): Which suffixes/phases to include.
            nameExps (list[str]): Which nameExps to include.
            winMS_list (list[float]): Which winMS to include.
            save (bool): Save figure.
            folder (str): Folder to save figures.
            show (bool): Show figure interactively.
            zscore (bool): Whether to z-score values before correlation.
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

        if error_type == "lin":
            col_error = "lin_error"
        else:
            col_error = "error"

        # --- collect per-mouse averages ---
        mouse_data = []
        for _, row in df.iterrows():
            mask = np.ones_like(row[col_error], dtype=bool)  # default

            # --- apply speed mask if needed ---
            if speed in ["fast", "slow"] and "speedMask" in row:
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                thresh_mask = row["predLoss"] <= row["predLossThreshold"]
                mask = mask & thresh_mask

            # select error values
            if isinstance(row[col_error], np.ndarray):
                print("applying mask to error array")
                error_vals = row[col_error][mask]
            else:
                error_vals = np.array([row[col_error]])

            # select against variable from decoding results
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

            if against == "entropy":
                val_array = decoding_results["Hn"].flatten()
            elif against == "maxp":
                val_array = decoding_results["maxp"].flatten()
            else:
                raise ValueError("against must be 'entropy' or 'maxp'")

            # apply same mask
            if zscore:
                val_array = (val_array - np.mean(val_array)) / np.std(val_array)
            val_array = val_array[mask]
            if len(val_array) == 0 or len(error_vals) == 0:
                continue

            mouse_data.append(
                {
                    "mouse": row["mouse_name"],
                    "nameExp": row["nameExp"],
                    "phase": row["phase"],
                    "winMS": row["winMS"],
                    "x": np.nanmean(val_array),
                    "y": np.nanmean(error_vals),
                }
            )

            del decoding_results

        if len(mouse_data) == 0:
            print("No data available for this selection.")
            return

        plot_df = pd.DataFrame(mouse_data)

        # --- plot correlations ---
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue="phase",
            style="nameExp",
            s=80,
            ax=ax,
            palette="tab10",
        )

        # --- regression line ---
        slope, intercept, r_value, p_value, std_err = linregress(
            plot_df["x"], plot_df["y"]
        )
        x_vals = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
        y_vals = intercept + slope * x_vals
        ax.plot(
            x_vals,
            y_vals,
            color="black",
            lw=2,
            label=f"R={r_value:.2f}, p={p_value:.3f}",
        )

        ax.set_xlabel(against)
        ax.set_ylabel(f"{error_type}_error_{mode} ({speed})")
        ax.set_title(f"Correlation per mouse ({against} vs error)")
        ax.legend()
        fig.tight_layout()

        if save and folder is not None:
            fname = f"correlation_{against}_vs_{error_type}_{mode}_{speed}{'_zscored' if zscore else ''}"
            fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
            fig.savefig(os.path.join(folder, fname + ".svg"))
        if show:
            plt.show()
        plt.close(fig)

    def correlation_per_mouse_spikes(
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
    ):
        """
        Correlate decoder values (entropy/maxp/error) with spike counts per mouse.
        Each point is per mouse, averaged across winMS/phase as specified.
        """

        import os
        import pickle

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
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

        if error_type == "lin":
            col_error = "lin_error"
        else:
            col_error = "error"

        mouse_data = []

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
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]

            if isinstance(row[col_error], np.ndarray):
                print("applying mask to error array")
                error_vals = row[col_error][mask]
            else:
                error_vals = np.array([row[col_error]])

            # --- extract against variable ---
            if against in ["entropy", "maxp"]:
                val_array = (
                    decoding_results["Hn"].flatten()
                    if against == "entropy"
                    else decoding_results["maxp"].flatten()
                )
                val_array = val_array[mask]
            elif against == "error":
                val_array = error_vals
            else:
                raise ValueError("against must be 'entropy', 'maxp' or 'error'")

            # --- load spikes ---
            clusters_file = os.path.join(
                mouse_results.folderResult, "clusters_pre_wTrain_False.pkl"
            )
            clusters_time_file = os.path.join(
                mouse_results.folderResult, "clusters_time_pre_wTrain_False.pkl"
            )
            try:
                try:
                    # with open(clusters_file, "rb") as f:
                    #     clusters = pickle.load(f)
                    with open(clusters_time_file, "rb") as f:
                        clusters_time = pickle.load(f)
                except:
                    clusters_file = os.path.abspath(
                        os.path.join(
                            mouse_results.folderResult,
                            "..",
                            "..",
                            "last_bayes",
                            "results",
                            f"clusters_pre_wTrain_{'True' if row['phase'] == 'training' else 'False'}.pkl",
                        )
                    )
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
                    # with open(clusters_file, "rb") as f:
                    #     clusters = pickle.load(f)
                    with open(clusters_time_file, "rb") as f:
                        clusters_time = pickle.load(f)
            except Exception as e:
                print(f"Failed to load spike clusters for {row['mouse_name']}: {e}")
                continue

            # --- count spikes per decoding time bin ---
            times = decoding_results["times"].flatten()
            spikes_count = np.zeros_like(times, dtype=float)
            for cl_time in clusters_time:
                spikes_count += np.histogram(
                    cl_time,
                    bins=np.append(times, times[-1] + np.median(np.diff(times))),
                )[0]
            spikes_count = spikes_count[mask]

            # --- average per mouse ---
            if len(val_array) == 0 or len(spikes_count) == 0:
                continue

            x_val = np.nanmean(val_array)
            y_val = np.nanmean(spikes_count)

            if zscore:
                x_val = (x_val - np.nanmean(val_array)) / np.nanstd(val_array)
                y_val = (y_val - np.nanmean(spikes_count)) / np.nanstd(spikes_count)

            mouse_data.append(
                {
                    "mouse": row["mouse_name"],
                    "phase": row["phase"],
                    "winMS": row["winMS"],
                    "nameExp": row["nameExp"],
                    "x": x_val,
                    "y": y_val,
                }
            )

            del decoding_results

        if len(mouse_data) == 0:
            print("No data available for this selection.")
            return

        plot_df = pd.DataFrame(mouse_data)

        # --- plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue="phase",
            style="nameExp",
            s=80,
            ax=ax,
            palette="tab10",
        )

        # --- regression ---
        slope, intercept, r_value, p_value, std_err = linregress(
            plot_df["x"], plot_df["y"]
        )
        x_vals = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
        y_vals = intercept + slope * x_vals
        ax.plot(
            x_vals,
            y_vals,
            color="black",
            lw=2,
            label=f"R={r_value:.2f}, p={p_value:.3f}",
        )

        ax.set_xlabel(against)
        ax.set_ylabel("Spikes count")
        ax.set_title(f"Correlation per mouse ({against} vs spikes)")
        ax.legend()
        fig.tight_layout()

        if save and folder is not None:
            fname = f"correlation_{against}_vs_spikes"
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
        Compute correlation between decoder variable (entropy/maxp/error) and spike counts
        for each (mouse, nameExp, winMS, phase) and plot a barplot of the mean correlations.
        """

        import os
        import pickle

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
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

        if error_type == "lin":
            col_error = "lin_error"
        else:
            col_error = "error"

        correlations = []

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
                mask = row["speedMask"] if speed == "fast" else ~row["speedMask"]
            if mode == "selected" and "predLoss" in row and "predLossThreshold" in row:
                mask &= row["predLoss"] <= row["predLossThreshold"]

            if isinstance(row[col_error], np.ndarray):
                error_vals = row[col_error][mask]
            else:
                error_vals = np.array([row[col_error]])

            # --- extract against variable ---
            if against in ["entropy", "maxp"]:
                val_array = (
                    decoding_results["Hn"].flatten()
                    if against == "entropy"
                    else decoding_results["maxp"].flatten()
                )
                val_array = val_array[mask]
            elif against == "error":
                val_array = error_vals
            else:
                raise ValueError("against must be 'entropy', 'maxp' or 'error'")

            # --- load spikes ---
            clusters_file = os.path.join(
                mouse_results.folderResult, "clusters_pre_wTrain_False.pkl"
            )
            clusters_time_file = os.path.join(
                mouse_results.folderResult, "clusters_time_pre_wTrain_False.pkl"
            )
            try:
                try:
                    # with open(clusters_file, "rb") as f:
                    #     clusters = pickle.load(f)
                    with open(clusters_time_file, "rb") as f:
                        clusters_time = pickle.load(f)
                except:
                    clusters_file = os.path.abspath(
                        os.path.join(
                            mouse_results.folderResult,
                            "..",
                            "..",
                            "last_bayes",
                            "results",
                            f"clusters_pre_wTrain_{'True' if row['phase'] == 'training' else 'False'}.pkl",
                        )
                    )
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
                    # with open(clusters_file, "rb") as f:
                    #     clusters = pickle.load(f)
                    with open(clusters_time_file, "rb") as f:
                        clusters_time = pickle.load(f)
            except Exception as e:
                print(f"Failed to load spike clusters for {row['mouse_name']}: {e}")
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

            # optional z-score
            if zscore:
                val_array = (val_array - np.nanmean(val_array)) / np.nanstd(val_array)
                spikes_count = (spikes_count - np.nanmean(spikes_count)) / np.nanstd(
                    spikes_count
                )

            # --- compute correlation per mouse × winMS × phase × nameExp ---
            r, _ = spearmanr(val_array, spikes_count)
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
        sns.barplot(
            data=corr_df,
            x="phase",
            y="correlation",
            hue=hue,
            errorbar="sd",
            ax=ax,
            palette="tab10",
            order=sorted(corr_df["phase"].unique()),
        )
        # overlay points
        sns.stripplot(
            data=corr_df,
            x="phase",
            y="correlation",
            hue=hue,
            dodge=True,
            ax=ax,
            palette="tab10",
            size=7,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.set_ylabel(f"Spearman correlation ({against} vs spikes)")
        ax.set_xlabel("Phase")
        ax.set_title("Correlation per mouse/phase/winMS/nameExp")
        ax.legend(loc="best")
        fig.tight_layout()

        if save and folder is not None:
            fname = f"barplot_correlation_{against}_vs_spikes"
            fig.savefig(os.path.join(folder, fname + ".png"), dpi=150)
            fig.savefig(os.path.join(folder, fname + ".svg"))
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
        import os
        import pickle

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
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
            elif ann_var in ["lin_error", "predLoss", "linPred"]:
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
        ann_var="linPred",  # "maxp", "entropy", "lin_error", etc.
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
        import os
        import pickle

        import matplotlib.pyplot as plt
        import numpy as np

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
            except:
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
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["speedFilter"]
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
            train_mask = (
                res.iloc[0]
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["trainEpochs"]
            )
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

            lin_true = np.array(group["linTruePos"].tolist())[mask]
            lin_pred = np.array(group["linPred"].tolist())[mask]

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
        n_hues = len(phases)

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
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["speedFilter"]
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
            train_mask = (
                res.iloc[0]
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["trainEpochs"]
            )
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

            lin_true = np.array(group["linTruePos"].tolist())[mask]
            lin_pred = np.array(group["linPred"].tolist())[mask]

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
        n_hues = len(winMSs)

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
        df_metric = df_winMS[
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
        df = self.results_df.copy()
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
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["speedFilter"]
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
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["speedFilter"]
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
            phase_value = row["phase"]
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
            trainEpochs = (
                row["results"]
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["trainEpochs"]
            )
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
            train_mask = (
                res.iloc[0]
                .data_helper[str(row.winMS)]
                .fullBehavior["Times"]["trainEpochs"]
            )
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

            lin_true = np.array(group["linTruePos"].tolist())[mask]
            lin_pred = np.array(group["linPred"].tolist())[mask]

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
                lin_true_filtered = np.array(group["linTruePos"].tolist())[mask]
                lin_pred_filtered = np.array(group["linPred"].tolist())[mask]

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
                    ]["linPred"].tolist()
                )[mask]
                bayes_true = np.array(
                    bayes_df[
                        (bayes_df["mouse_manipe"] == mouse)
                        & (bayes_df["phase"] == phase)
                        & (bayes_df["winMS"] == winMS)
                    ]["linTruePos"].tolist()
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
