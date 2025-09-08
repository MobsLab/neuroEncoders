#!/usr/bin/env python
import fnmatch
import gc
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import psutil

from neuroencoders.utils.MOBS_Functions import path_for_experiments_df

win_values = [2.16]
win_values = [0.036, 0.108, 0.18, 0.252, 0.504, 1.08, 2.16]
win_values = [0.18, 2.16]
win_values = [0.036, 0.108]
win_values = [0.108, 0.18, 0.252, 0.504]
win_values = [0.504]  # only kept for new dataset
win_values = [0.108, 0.252]
win_values = [0.108]
win_values = [0.108, 0.252]  # only kept for new dataset
# Mice name
mice_nb = []
mice_nb = [
    "M1199_PAG",
    "M994_PAG",
    "M1239_MFB",
    "M1230_Novel",
    "M1230_Known",
    "M1162_MFB",
]
# mice_nb = ["M1199_PAG"]
nameExp = "NO_PREDLOSS_GaussianHeatMap_LinearLoss"
nbEpochs = str(200)
run_ann = True
target = "pos"
phase = "pre"


def check_memory_usage():
    """Monitor memory usage and print warnings if getting high"""
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        print(f"WARNING: Memory usage at {memory.percent:.1f}%")
        return True
    return False


def cleanup_memory():
    """Force garbage collection and print memory stats"""
    gc.collect()
    memory = psutil.virtual_memory()
    print(f"Memory usage after cleanup: {memory.percent:.1f}%")


def process_directory(dir, win, force, redo, lstmAndTransfo=False):
    xml_file = None
    for pattern in [
        "*SpikeRef*.xml",
        f"*{os.path.basename(dir)[:4]}*.xml",
        f"*{os.path.basename(os.path.dirname(dir))[:4]}*.xml",
        "*amplifier*.xml",
        "*.xml",
    ]:
        xml_file = next(
            (
                os.path.join(dir, f)
                for f in os.listdir(dir)
                if f.endswith(".xml") and fnmatch.fnmatch(f, pattern)
            ),
            None,
        )
        if xml_file:
            break

    if (
        os.path.exists(
            os.path.join(
                dir, nameExp, "results", str(int(win * 1000)), "featurePred.csv"
            )
        )
        or os.path.exists(
            os.path.join(
                dir,
                nameExp,
                "results",
                str(int(win * 1000)),
                f"featurePred_{phase}.csv",
            )
        )
        or os.path.exists(
            os.path.join(
                dir,
                nameExp,
                "results",
                str(int(win * 1000)),
                "featurePred_training.csv",
            )
        )
        or (
            os.path.exists(
                os.path.join(
                    dir,
                    nameExp + "_LSTM",
                    "results",
                    str(int(win * 1000)),
                    "featurePred.csv",
                )
            )
            and lstmAndTransfo
        )
        or (
            os.path.exists(
                os.path.join(
                    dir,
                    nameExp + "_LSTM",
                    "results",
                    str(int(win * 1000)),
                    f"featurePred_{phase}.csv",
                )
            )
            and lstmAndTransfo
        )
        or (
            os.path.exists(
                os.path.join(
                    dir,
                    nameExp + "_LSTM",
                    "results",
                    str(int(win * 1000)),
                    "featurePred_training.csv",
                )
            )
            and lstmAndTransfo
        )
        or (
            os.path.exists(
                os.path.join(
                    dir,
                    nameExp + "_Transformer",
                    "results",
                    str(int(win * 1000)),
                    "featurePred_training.csv",
                )
            )
            and not lstmAndTransfo
        )
        or (
            os.path.exists(
                os.path.join(
                    dir,
                    nameExp + "_Transformer",
                    "results",
                    str(int(win * 1000)),
                    "featurePred.csv",
                )
            )
            and not lstmAndTransfo
        )
        or (
            os.path.exists(
                os.path.join(
                    dir,
                    nameExp + "_Transformer",
                    "results",
                    str(int(win * 1000)),
                    f"featurePred_{phase}.csv",
                )
            )
            and not lstmAndTransfo
        )
    ) and (not force):
        print(f"featurePred+-{phase}.csv already exists in {dir}. Skipping...")
        return None, None

    if xml_file:
        cmd_ann = [
            "/usr/bin/env",
            "/home/mickey/Documents/Theotime/neuroEncoders/.venv/bin/python",
            "/home/mickey/Documents/Theotime/neuroEncoders/neuroEncoder",
            "ann",
            xml_file,
            "--window",
            str(win),
            "--striding",
            str(win),
            "-e",
            nbEpochs,
            "--gpu",
            "--target",
            target,
            # "--predicted_loss",
            "--early_stop",
            # "--transform_w_log",
            "--no-dense",
        ]
        if lstmAndTransfo:
            cmd_ann += ["--lstm", "--name", nameExp + "_LSTM"]
        else:
            cmd_ann += ["--name", nameExp + "_Transformer"]
        if sleep:
            cmd_ann += ["--test_sleep"]

        cmd_bayes = [
            "/usr/bin/env",
            "/home/mickey/Documents/Theotime/neuroEncoders/.venv/bin/python",
            "/home/mickey/Documents/Theotime/neuroEncoders/neuroEncoder",
            "bayes",
            xml_file,
            "--window",
            str(win),
            "-e",
            nbEpochs,
            "--target",
            target,
        ]
        if lstmAndTransfo:
            cmd_bayes += ["--name", nameExp + "_LSTM"]
        else:
            cmd_bayes += ["--name", nameExp + "_Transformer"]
        if sleep:
            cmd_bayes += ["--test_sleep"]

        if "_test" not in xml_file:
            cmd_bayes += ["--phase", phase]
        if redo:
            cmd_ann += ["--redo"]
        if "_test" not in xml_file:
            cmd_ann += ["--phase", phase]
        if run_bayes and not lstmAndTransfo:
            return cmd_ann, cmd_bayes
        else:
            return cmd_ann, None
    else:
        print(f"No .xml file found in {dir}")
        return None, None


def run_commands_for_mouse(commands):
    """Run all commands for a single mouse sequentially."""
    for cmd in commands:  # Check memory before running command
        if check_memory_usage():
            print("High memory usage detected, forcing cleanup...")
            cleanup_memory()
            time.sleep(2)
        subprocess.run(cmd)


def run_commands_sequentially(mouse_commands):
    """Run all commands sequentially for all mice."""
    for dir, commands in mouse_commands.items():
        print(f"Processing mouse directory: {dir}")
        for cmd in commands:
            print(f"Running command: {cmd}")
            if check_memory_usage():
                print("High memory usage detected, forcing cleanup...")
                cleanup_memory()
                time.sleep(2)
            subprocess.run(cmd)
            cleanup_memory()
            print(f"Finished command: {cmd}")


def run_commands_parallel(mouse_commands):
    """Run all commands in parallel for all mice."""
    with ProcessPoolExecutor() as executor:
        for dir, commands in mouse_commands.items():
            print(f"Processing mouse directory in parallel: {dir}")
            executor.map(subprocess.run, commands)


if __name__ == "__main__":
    mode = "sequential"
    force = False
    lstm = False
    redo = "--redo" in sys.argv
    rsync = "--rsync" in sys.argv
    sleep = "--sleep" in sys.argv
    force = "--force" in sys.argv
    lstm = "--lstm" in sys.argv
    run_bayes = "--bayes" in sys.argv

    if len(sys.argv) < 2:
        print("Usage: python runAllMice.py <mode> [force]")
        print(f"Mode: {mode}")
        print(f"Force: {force}")
        mode = "sequential"

    if len(sys.argv) > 1 and sys.argv[1].lower() == "sequential":
        mode = "sequential"
        print("Running in sequential mode")
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "parallel":
        mode = "parallel"
        print("Running in parallel mode")
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "force":
        force = True
        print("Running with force option")
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "lstm":
        lstm = True
        print("Running with LSTM option")
    elif len(sys.argv) > 2 and sys.argv[2].lower() == "force":
        force = True
    elif len(sys.argv) > 2 and sys.argv[2].lower() == "lstm":
        lstm = True
        print("Running with LSTM and Transformer")

    if len(sys.argv) > 3 and sys.argv[3].lower() == "lstm":
        lstm = True
        print("Running with LSTM and Transformer")

    dirs = [
        os.path.abspath(os.path.expanduser(d))
        for d in os.listdir(".")
        if os.path.isdir(d)
    ]

    PathForExperiments = path_for_experiments_df("Sub", training_name=nameExp)

    # print(f"Found directories: {dirs}")
    mouse_commands = {}
    for dir in dirs:
        if any(mouse in dir for mouse in mice_nb) or not mice_nb:
            if "M1199_MFB" not in dir:
                mouse_commands[dir] = []
                for win in win_values:
                    if lstm:
                        for lstmAndTransfo in [False, True]:
                            # print(
                            #     f"Processing {dir} with window {win} and lstmAndTransfo {lstmAndTransfo}"
                            # )
                            cmd_ann, cmd_bayes = process_directory(
                                dir=dir,
                                win=win,
                                force=force,
                                redo=redo,
                                lstmAndTransfo=lstmAndTransfo,
                            )
                            if cmd_ann:
                                mouse_commands[dir].append(cmd_ann)
                            if cmd_bayes:
                                mouse_commands[dir].append(cmd_bayes)
                    else:
                        cmd_ann, cmd_bayes = process_directory(dir, win, force, redo)
                        if cmd_ann:
                            mouse_commands[dir].append(cmd_ann)
                        if cmd_bayes:
                            mouse_commands[dir].append(cmd_bayes)
            else:
                print(f"Processing M1199MFB mouse in directory: {dir}")
                for dirmfb in ["exp1", "exp2"]:
                    mouse_commands[os.path.join(dir, dirmfb)] = []
                    for win in win_values:
                        cmd_ann, cmd_bayes = process_directory(
                            os.path.join(dir, dirmfb), win, force, redo
                        )
                        if cmd_ann:
                            mouse_commands[os.path.join(dir, dirmfb)].append(cmd_ann)
                        if cmd_bayes:
                            mouse_commands[os.path.join(dir, dirmfb)].append(cmd_bayes)

            if rsync:
                PathForExperiments["realPath"] = PathForExperiments["path"].apply(
                    lambda x: os.path.realpath(x)
                )
                try:
                    Mouse = PathForExperiments[
                        PathForExperiments["realPath"] == os.path.realpath(dir)
                    ].iloc[0]["name"]
                    print(f"Mouse: {Mouse} from PathForExperiments")
                    SOURCE = os.path.realpath(dir)
                    DESTINATION = PathForExperiments[
                        PathForExperiments["name"] == Mouse
                    ].iloc[0]["network_path"]

                    runNasCMD = [
                        "/usr/bin/env",
                        "/home/mickey/Documents/Theotime/neuroEncoders/.venv/bin/python",
                        "/home/mickey/Documents/Theotime/neuroEncoders/utils/NAS.py",
                        SOURCE,
                        DESTINATION,
                        "--force",
                    ]
                    if "M1199_MFB" not in dir:
                        mouse_commands[dir].append(runNasCMD)
                    else:
                        for dirmfb in ["exp1", "exp2"]:
                            mouse_commands[os.path.join(dir, dirmfb)].append(runNasCMD)
                except:
                    pass

    if mode == "sequential":
        run_commands_sequentially(mouse_commands)
    elif mode == "parallel":
        run_commands_parallel(mouse_commands)
