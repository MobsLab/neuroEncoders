#!/usr/bin/env python3

import os
import warnings
from typing import Any, Dict, List, Optional

import scipy.io


def path_for_experiments(
    experiment_name: str, training_name: str = "Final_Results_v3"
) -> Dict[str, Any]:
    """
    Python equivalent of PathForExperimentsERC MATLAB function.

    Returns experiment directory information and metadata for neuroscience experiments

    Args:
        experiment_name: Name of the experiment type (e.g., 'SubMFB', 'UMazePAG', etc.)
        training_name: Name of the training session (default is 'Final_Results_v3')

    Returns:
        Dictionary containing:
        - 'path': List of experiment paths (strings)
        - 'additional_paths': List of additional paths for experiments with multiple sessions
        - 'results': List of results paths (for Sub experiments)
        - 'expe_info': List of experiment info dictionaries
        - 'manipe': List of experiment types
        - 'name': List of mouse names
        - 'group': List of group classifications (for UMazePAG)

     Currently used mice 28/05/2025
     -----------------------------------------------------------------------------------------------
     |      Known       |        MFB         |        UMazePAG        |    Reversal     |   Novel   |
     -----------------------------------------------------------------------------------------------
     |  M1336_known     |  M1336_MFB         |  M1186                 |  M1199_reversal | M1230_Novel |
     |  M1230_Known     |  M1117             |  M1199_PAG             |                 |             |
     |                  |  M1281_MFB         |  M1182                 |                 |             |
     |                  |  M1168MFB          |  M994_PAG              |                 |             |
     |                  |  M1239MFB          |  M1239_PAG             |                 |             |
     |                  |  M1162_MFB         |  M1162_PAG             |                 |             |
     |                  |  M1199_MFB         |                        |                 |             |
     -----------------------------------------------------------------------------------------------

    Exemple:
        >>> path_for_experiments("SubMFB")
        {
            'path': ['/path/to/experiment1', '/path/to/experiment2'],
            'additional_paths': [['/path/to/additional1'], []],
            'results': ['/path/to/results1', '/path/to/results2'],
            'expe_info': [{'SessionType': 'UMaze'}, {'SessionType': 'MFB'}],
            'manipe': ['SubUMazePAG', 'SubMFB'],
            'name': ['Mouse123', 'Mouse456'],
            'group': {'LFP': [None, None], ...}
        }

    """

    # Groups for PAG experiments
    LFP = list(range(6, 22))  # 6:21 in MATLAB
    Neurons = [6, 7, 8, 10] + list(range(12, 22))  # [6 7 8 10 12:21]
    ECG = [6, 9, 10, 14, 15]  # [6 9 10 14 15]
    OB_resp = list(range(6, 11)) + [13, 14] + list(range(16, 22))  # [6:10 13 14 16:21]
    OB_gamma = (
        list(range(6, 9)) + [10, 11, 13, 14] + list(range(16, 21))
    )  # [6:8 10 11 13 14 16:20]
    PFC = list(range(6, 15)) + list(range(16, 19)) + [21]  # [6:14 16:18 21]

    # All groups
    LFP_All = list(range(1, 22))  # 1:21
    Neurons_All = [1] + [6, 7, 8, 10] + list(range(12, 22))  # [1 6 7 8 10 12:21]
    ECG_All = [1, 3, 5, 6, 9, 10, 14, 15]  # [1 3 5 6 9 10 14 15]

    # Define experiment categories
    MFB_keys = [
        "m1336_mfb",
        "m1117",
        "m1281_mfb",
        "m1168MFB",
        "m1239MFB",
        "m1162_mfb",
        "m1199_mfb",
    ]
    UMazePAG_keys = [
        "m1186",
        "m1199_pag",
        "m1182",
        "m994_PAG",
        "m1239_PAG",
        "m1162_PAG",
    ]
    Reversal_keys = ["m1199_reversal"]
    Known_keys = ["m1336_known", "m1230_Known"]
    Novel_keys = ["m1230_Novel"]

    # Base path directory
    pathdir = "/media/mickey/DataTheotime210/DimaERC2"

    # Define the first dictionary (python_dict equivalent)
    python_dict = {
        "m1168MFB": f"neuroencoders_1021/_work/M1168/{training_name}",
        "m1186": f"neuroencoders_1021/_work/M1186/{training_name}",
        "m1336_mfb": f"neuroencoders_1021/_work/M1336_MFB/{training_name}",
        "m1336_known": f"neuroencoders_1021/_work/M1336_known/{training_name}",
        "m1199_pag": f"neuroencoders_1021/_work/M1199_PAG/{training_name}",
        "m1199_reversal": f"neuroencoders_1021/_work/M1199_reversal/{training_name}",
        "m1117": f"neuroencoders_1021/_work/M1117_MFB/{training_name}",
        "m1281_mfb": f"neuroencoders_1021/_work/M1281_MFB/{training_name}",
        "m1182": f"neuroencoders_1021/_work/M1182_PAG/{training_name}",
        "m994_PAG": f"neuroencoders_1021/_work/M994_PAG/{training_name}",
        "m1239MFB": f"neuroencoders_1021/_work/M1239_MFB/{training_name}",
        "m1162_mfb": f"neuroencoders_1021/_work/M1162_MFB/{training_name}",
        "m1199_mfb": f"neuroencoders_1021/_work/M1199_MFB/exp1/{training_name}",
        "m1239_PAG": f"neuroencoders_1021/_work/M1239_PAG/{training_name}",
        "m1162_PAG": f"neuroencoders_1021/_work/M1162_PAG/{training_name}",
        "m1230_Known": f"neuroencoders_1021/_work/M1230_Known/{training_name}",
        "m1230_Novel": f"neuroencoders_1021/_work/M1230_Novel/{training_name}",
    }

    # Define the second dictionary (subpython_REAL equivalent)
    subpython_REAL = {
        "m1336_mfb": "/media/nas7/ProjetERC1/StimMFBWake/M1336/",
        "m1336_known": "/media/nas7/ProjetERC1/Known/M1336/",
        "m1186": "/media/nas6/ProjetERC2/Mouse-K186/20210409/_Concatenated/",
        "m1199_pag": "/media/nas6/ProjetERC2/Mouse-K199/20210408/_Concatenated/",
        "m1199_reversal": "/media/nas6/ProjetERC3/M1199/Reversal/",
        "m1117": "/media/nas5/ProjetERC1/StimMFBWake/M1117/",
        "m1281_mfb": "/media/nas7/ProjetERC1/StimMFBWake/M1281/",
        "m1168MFB": "/media/nas5/ProjetERC1/StimMFBWake/M1168/",
        "m1182": "/media/nas6/ProjetERC2/Mouse-K182/20200301/_Concatenated/",
        "m994_PAG": "/media/nas5/ProjetERC2/Mouse-994/20191013/PagExp/_Concatenated/",
        "m1239MFB": "/media/nas6/ProjetERC1/StimMFBWake/M1239/Exp2/",
        "m1162_mfb": "/media/nas5/ProjetERC1/StimMFBWake/M1162/",
        "m1199_mfb": "/media/nas6/ProjetERC1/StimMFBWake/M1199/exp1/",
        "m1239_PAG": "/media/nas7/ProjetERC2/Mouse-K239/2021110/_Concatenated/",
        "m1162_PAG": "/media/nas5/ProjetERC2/Mouse-K162/20210121/_Concatenated/",
        "m1230_Known": "/media/nas6/ProjetERC1/Known/M1230/",
        "m1230_Novel": "/media/nas6/ProjetERC1/Novel/M1230/",
    }

    # Select appropriate keys based on experiment name
    if experiment_name == "SubMFB":
        selected_keys = MFB_keys
    elif experiment_name == "SubMFBReal":
        selected_keys = MFB_keys
    elif experiment_name == "SubPAG":
        selected_keys = UMazePAG_keys
    elif experiment_name == "SubPAGReal":
        selected_keys = UMazePAG_keys
    elif experiment_name == "SubReversal":
        selected_keys = Reversal_keys
    elif experiment_name == "SubReversalReal":
        selected_keys = Reversal_keys
    elif experiment_name == "SubKnown":
        selected_keys = Known_keys
    elif experiment_name == "SubKnownReal":
        selected_keys = Known_keys
    elif experiment_name == "SubNovel":
        selected_keys = Novel_keys
    elif experiment_name == "SubNovelReal":
        selected_keys = Novel_keys
    elif experiment_name == "Sub":
        selected_keys = (
            MFB_keys + UMazePAG_keys + Reversal_keys + Known_keys + Novel_keys
        )
    elif experiment_name == "SubReal":
        selected_keys = (
            MFB_keys + UMazePAG_keys + Reversal_keys + Known_keys + Novel_keys
        )
    else:
        selected_keys = []
        warnings.warn(
            "Not a subtype. Will choose from general MFB, UMazePAG, or Reversal (no ann)."
        )

    # Initialize result dictionary
    Dir = {
        "path": [],
        "additional_paths": [],  # For experiments with multiple sessions
        "results": [],
        "expe_info": [],
        "manipe": [],
        "name": [],
        "group": [],
        "network_path": [],
    }

    def load_expe_info(path: str) -> Optional[Dict]:
        """Helper function to load ExpeInfo.mat file"""
        expe_info_file = os.path.join(path, "ExpeInfo.mat")
        if os.path.isfile(expe_info_file):
            try:
                mat_data = scipy.io.loadmat(expe_info_file)
                return mat_data.get("ExpeInfo", None)
            except Exception as e:
                warnings.warn(f"Could not load ExpeInfo.mat from {expe_info_file}: {e}")
                return None
        else:
            warnings.warn(f"ExpeInfo.mat not found for {expe_info_file}")
            return None

    def add_experiment(
        main_path: str,
        additional_paths: List[str] = None,
        results_path: str = None,
        network_path: Optional[str] = None,
    ):
        """Helper function to add an experiment to the Dir structure"""
        Dir["path"].append(main_path)
        Dir["additional_paths"].append(additional_paths or [])
        Dir["results"].append(results_path or "")
        Dir["network_path"].append(network_path or "")

        # Load ExpeInfo from main path
        expe_info = load_expe_info(main_path)
        Dir["expe_info"].append(expe_info)

    # Handle Sub experiments
    if "Sub" in experiment_name:
        use_real = "Real" in experiment_name
        current_dict = subpython_REAL if use_real else python_dict
        real_dict = subpython_REAL

        for key in current_dict.keys():
            if key in selected_keys:
                results_path = current_dict[key]

                try:
                    if use_real:
                        current_path = results_path
                        add_experiment(current_path, results_path=results_path)
                    else:
                        current_path = os.path.dirname(results_path)
                        full_current_path = os.path.join(pathdir, current_path)
                        full_results_path = os.path.join(pathdir, results_path)
                        full_network_path = real_dict.get(key, None)
                        add_experiment(
                            full_current_path,
                            results_path=full_results_path,
                            network_path=full_network_path,
                        )

                except Exception as e:
                    warnings.warn(f"Error processing path for {key}: {e}")

    # Handle UMazePAG experiment
    elif experiment_name.lower() == "umazepag":
        umaze_paths = [
            "/media/DataMOBsRAIDN/ProjetERC2/Mouse-711/17032018/_Concatenated/",
            "/media/DataMOBsRAIDN/ProjetERC2/Mouse-712/12042018/_Concatenated/",
            "/media/DataMOBsRAIDN/ProjetERC2/Mouse-714/27022018/_Concatenated/",
            "/media/DataMOBsRAIDN/ProjetERC2/Mouse-742/31052018/_Concatenated/",
            "/media/DataMOBsRAIDN/ProjetERC2/Mouse-753/17072018/_Concatenated/",
            "/media/DataMOBsRAIDN/ProjetERC2/Mouse-797/11112018/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-798/12112018/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-828/20190305/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-861/20190313/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-882/20190409/PAGexp/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-905/20190404/PAGExp/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-906/20190418/PAGExp/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-911/20190508/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-912/20190515/PAGexp/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-977/20190812/PAGexp/Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-994/20191013/PagExp/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-K117/20201109/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-K124/20201120/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-K161/20201224/_Concatenated/",
            "/media/nas5/ProjetERC2/Mouse-K162/20210121/_Concatenated/",
            "/media/nas6/ProjetERC2/Mouse-K168/20210122/_Concatenated/",
            "/media/nas6/ProjetERC2/Mouse-K182/20200301/_Concatenated/",
            "/media/nas6/ProjetERC2/Mouse-K186/20210409/_Concatenated/",
            "/media/nas6/ProjetERC2/Mouse-K199/20210408/_Concatenated/",
            "/media/nas7/ProjetERC2/Mouse-K230/20210927/_Concatenated/",
            "/media/nas7/ProjetERC2/Mouse-K239/2021110/_Concatenated/",
        ]

        # Special case for Mouse1230 with two paths
        for i, path in enumerate(umaze_paths):
            if i == 24:  # Mouse1230
                additional_path = (
                    "/media/nas7/ProjetERC2/Mouse-K230/20211004/_Concatenated/"
                )
                add_experiment(path, additional_paths=[additional_path])
            else:
                add_experiment(path)

    # Handle StimMFBWake experiment
    elif experiment_name.lower() == "stimmfbwake":
        mfb_paths = [
            "/media/nas5/ProjetERC1/StimMFBWake/M0882/",
            "/media/nas5/ProjetERC1/StimMFBWake/M0936/",
            "/media/nas5/ProjetERC1/StimMFBWake/M0941/",
            "/media/nas5/ProjetERC1/StimMFBWake/M0934/",
            "/media/nas5/ProjetERC1/StimMFBWake/M0935/",
            "/media/nas5/ProjetERC1/StimMFBWake/M0863/",
            "/media/nas5/ProjetERC1/StimMFBWake/M0913/",
            "/media/nas6/ProjetERC1/StimMFBWake/M1081/",
            "/media/nas5/ProjetERC1/StimMFBWake/M1117/",
            "/media/nas5/ProjetERC1/StimMFBWake/M1161/",
            "/media/nas5/ProjetERC1/StimMFBWake/M1162/",
            "/media/nas5/ProjetERC1/StimMFBWake/M1168/",
            "/media/nas6/ProjetERC1/StimMFBWake/M1182/",
            "/media/nas6/ProjetERC1/StimMFBWake/M1199/exp1/",
            "/media/nas6/ProjetERC1/StimMFBWake/M1223/",
            "/media/nas6/ProjetERC1/StimMFBWake/M1228/take2/",
            "/media/nas6/ProjetERC1/StimMFBWake/M1239/Exp1/",
            "/media/nas7/ProjetERC1/StimMFBWake/M1257/",
            "/media/nas7/ProjetERC1/StimMFBWake/M1281/",
            "/media/nas7/ProjetERC1/StimMFBWake/M1317/1/",
            "/media/nas7/ProjetERC1/StimMFBWake/M1334/",
            "/media/nas7/ProjetERC1/StimMFBWake/M1336/",
        ]

        # Handle special cases with multiple paths
        special_cases = {
            12: ["/media/nas7/ProjetERC1/StimMFBWake/M1182/2/"],  # M1182
            13: ["/media/nas6/ProjetERC1/StimMFBWake/M1199/exp2/"],  # M1199
            16: ["/media/nas6/ProjetERC1/StimMFBWake/M1239/Exp2/"],  # M1239
            19: ["/media/nas7/ProjetERC1/StimMFBWake/M1317/2/"],  # M1317
        }

        for i, path in enumerate(mfb_paths):
            additional_paths = special_cases.get(i, [])
            add_experiment(path, additional_paths=additional_paths)

    # Handle Reversal experiment
    elif experiment_name.lower() == "reversal":
        reversal_paths = [
            "/media/nas5/ProjetERC3/M994/Reversal/",
            "/media/nas5/ProjetERC2/Mouse-K081/20200925/_Concatenated/",
            "/media/nas6/ProjetERC3/M1199/Reversal/",
        ]

        for path in reversal_paths:
            add_experiment(path)

    # Handle UMazePAGPCdriven experiment
    elif experiment_name.lower() == "umazepagpcdriven":
        add_experiment("/media/nas5/ProjetERC2/Mouse-K115/20201006/_Concatenated/")

    # Handle Novel experiment
    elif experiment_name.lower() == "novel":
        novel_paths = [
            "/media/nas6/ProjetERC1/Novel/M1016/",
            "/media/nas6/ProjetERC1/Novel/M1081/",
            "/media/nas6/ProjetERC1/Novel/M1083/",
            "/media/nas6/ProjetERC1/Novel/M1183/",
            "/media/nas6/ProjetERC1/Novel/M1116/",
            "/media/nas6/ProjetERC1/Novel/M1117/",
            "/media/nas6/ProjetERC1/Novel/M1161/",
            "/media/nas6/ProjetERC1/Novel/M1162/",
            "/media/nas6/ProjetERC1/Novel/M1182/",
            "/media/nas6/ProjetERC1/Novel/M1185/",
            "/media/nas6/ProjetERC1/Novel/M1223/",
            "/media/nas6/ProjetERC1/Novel/M1228/",
            "/media/nas6/ProjetERC1/Novel/M1230/",
            "/media/nas7/ProjetERC1/Novel/M1281/",
            "/media/nas7/ProjetERC1/Novel/M1317/",
            "/media/nas7/ProjetERC1/Novel/M1336/",
            "/media/nas6/ProjetERC1/Novel/M1168/",
            "/media/nas6/ProjetERC1/Novel/M1239/",
        ]

        for path in novel_paths:
            add_experiment(path)

    # Handle BaselineSleep experiment
    elif experiment_name.lower() == "baselinesleep":
        baseline_data = [
            {
                "main": "/media/hobbes/DataMOBs155/M1162/BaselineSleep/1/",
                "additional": ["/media/hobbes/DataMOBs155/M1162/BaselineSleep/2/"],
            },
            {
                "main": "/media/hobbes/DataMOBs155/M1168/BaselineSleep/1/",
                "additional": [
                    "/media/hobbes/DataMOBs155/M1168/BaselineSleep/2/",
                    "/media/hobbes/DataMOBs155/M1168/BaselineSleep/3/",
                ],
            },
            {
                "main": "/media/nas6/ProjetERC1/BaselineSleep/M1185/20210412/",
                "additional": [],
            },
            {"main": "/media/nas6/ProjetERC1/BaselineSleep/M1199/", "additional": []},
            {"main": "/media/nas6/ProjetERC1/BaselineSleep/M1230/", "additional": []},
        ]

        for entry in baseline_data:
            add_experiment(entry["main"], additional_paths=entry["additional"])

    # Handle Known experiment
    elif experiment_name.lower() == "known":
        known_data = [
            {"main": "/media/nas6/ProjetERC1/Known/M1230/", "additional": []},
            {"main": "/media/nas7/ProjetERC1/Known/M1281/", "additional": []},
            {"main": "/media/nas7/ProjetERC1/Known/M1317/", "additional": []},
            {
                "main": "/media/nas7/ProjetERC1/Known/M1334/1/",
                "additional": ["/media/nas7/ProjetERC1/Known/M1334/2/"],
            },
            {"main": "/media/nas7/ProjetERC1/Known/M1336/", "additional": []},
        ]

        for entry in known_data:
            add_experiment(entry["main"], additional_paths=entry["additional"])

    # Get mouse names
    for i, path in enumerate(Dir["path"]):
        try:
            Dir["manipe"].append(Dir["expe_info"][i]["SessionType"].item(0)[0])
        except:
            Dir["manipe"].append(experiment_name)

        # Adjust manipe for sub experiments
        if Dir["expe_info"][i] is not None:
            if "sub" in experiment_name.lower():
                prefix = "Sub"
            else:
                prefix = ""
            session_type = str(Dir["manipe"][i]).lower()
            if "mfb" in session_type:
                Dir["manipe"][i] = f"{prefix}MFB"
            elif "reversal" in session_type:
                Dir["manipe"][i] = f"{prefix}Reversal"
            elif "known" in session_type:
                Dir["manipe"][i] = f"{prefix}Known"
            elif "novel" in session_type:
                Dir["manipe"][i] = f"{prefix}Novel"
            elif "umaze" in session_type or "pag" in session_type:
                Dir["manipe"][i] = f"{prefix}PAG"
            else:
                Dir["manipe"][i] = f"{prefix}{session_type.capitalize()}"

        # Extract mouse name from path
        if "Mouse-" in path:
            start_idx = path.find("Mouse-") + 6
            if path[start_idx] == "K":
                mouse_name = f"Mouse1{path[start_idx + 1 : start_idx + 4]}"
            else:
                mouse_name = f"Mouse{path[start_idx : start_idx + 3]}"
        elif "/M" in path:
            start_idx = path.find("/M") + 2
            if path[start_idx] == "1":
                mouse_name = f"Mouse{path[start_idx : start_idx + 4]}"
            elif path[start_idx] == "0":
                mouse_name = f"Mouse{path[start_idx + 1 : start_idx + 4]}"
            else:
                mouse_name = f"Mouse{path[start_idx : start_idx + 3]}"
        else:
            mouse_name = "UnknownMouse"

        Dir["name"].append(mouse_name)

    # Get mouse groups for UMazePAG
    if experiment_name.lower() == "umazepag":
        # Initialize group lists
        group_dict = {
            "LFP": [None] * len(Dir["path"]),
            "Neurons": [None] * len(Dir["path"]),
            "ECG": [None] * len(Dir["path"]),
            "OB_resp": [None] * len(Dir["path"]),
            "OB_gamma": [None] * len(Dir["path"]),
            "PFC": [None] * len(Dir["path"]),
        }

        # Assign groups (using 0-based indexing, converting from 1-based MATLAB)
        for j in LFP:
            if j - 1 < len(Dir["path"]):  # Convert to 0-based
                group_dict["LFP"][j - 1] = "LFP"

        for j in Neurons:
            if j - 1 < len(Dir["path"]):
                group_dict["Neurons"][j - 1] = "Neurons"

        for j in ECG:
            if j - 1 < len(Dir["path"]):
                group_dict["ECG"][j - 1] = "ECG"

        for j in OB_resp:
            if j - 1 < len(Dir["path"]):
                group_dict["OB_resp"][j - 1] = "OB_resp"

        for j in OB_gamma:
            if j - 1 < len(Dir["path"]):
                group_dict["OB_gamma"][j - 1] = "OB_gamma"

        for j in PFC:
            if j - 1 < len(Dir["path"]):
                group_dict["PFC"][j - 1] = "PFC"

        Dir["group"] = group_dict

    return Dir


# Example usage and helper functions
def get_all_paths(Dir: Dict[str, Any], index: int) -> List[str]:
    """
    Get all paths (main + additional) for a given experiment index.

    Args:
        Dir: Directory dictionary from path_for_experiments
        index: Index of the experiment

    Returns:
        List of all paths for the experiment
    """
    if index >= len(Dir["path"]):
        return []

    all_paths = [Dir["path"][index]]
    if Dir["additional_paths"][index]:
        all_paths.extend(Dir["additional_paths"][index])

    return all_paths


def print_experiment_summary(Dir: Dict[str, Any]):
    """Print a summary of the experiments in the directory."""
    print(f"Found {len(Dir['path'])} experiments:")
    print("-" * 60)

    for i in range(len(Dir["path"])):
        name = Dir["name"][i] if i < len(Dir["name"]) else "Unknown"
        manipe = Dir["manipe"][i] if i < len(Dir["manipe"]) else "Unknown"
        main_path = Dir["path"][i]
        additional_count = (
            len(Dir["additional_paths"][i]) if Dir["additional_paths"][i] else 0
        )

        print(f"{i + 1:2d}. {name:<12} | {manipe:<15} | {main_path}")
        if additional_count > 0:
            print(f"    + {additional_count} additional path(s)")


if __name__ == "__main__":
    # Test the function
    try:
        result = path_for_experiments("Sub")
        print_experiment_summary(result)

        # Example of getting all paths for first experiment
        if result["path"]:
            all_paths = get_all_paths(result, 0)
            print(f"\nAll paths for first experiment: {all_paths}")

    except Exception as e:
        print(f"Error: {e}")
