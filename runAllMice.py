#!/usr/bin/env python
import fnmatch
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

win_values = [2.16]
win_values = [0.036, 0.108, 0.18, 0.252, 0.504, 1.08, 2.16]
win_values = [0.18, 2.16]
win_values = [0.036, 0.108]
win_values = [0.108, 0.18, 0.252, 0.504]
win_values = [0.108, 0.252]  # only kept for new dataset
# Mice name
mice_nb = ["M1199_PAG"]
mice_nb = []
nameExpBayes = "Pos_pre_Retracked"
nameExp = "LinAndThigmo"
nameExp = "LinAndDirection_SpecificLoss"
nameExp = "pos_transformer"
nbEpochs = str(100)
targetBayes = "pos"
target = "pos"
phase = "pre"


def process_directory(dir, win, force):
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
    ) and (not force):
        print(f"featurePred+-{phase}.csv already exists in {dir}. Skipping...")
        return None, None

    if xml_file:
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
            "--name",
            nameExp,
            "--target",
            target,
        ]
        if "_test" not in xml_file:
            cmd_bayes += ["--phase", phase]
        cmd_ann = [
            "/usr/bin/env",
            "/home/mickey/Documents/Theotime/neuroEncoders/.venv/bin/python",
            "/home/mickey/Documents/Theotime/neuroEncoders/neuroEncoder",
            "ann",
            xml_file,
            "--window",
            str(win),
            "-e",
            nbEpochs,
            "--gpu",
            "--name",
            nameExp,
            "--target",
            target,
            "--predicted_loss",
            "--early_stop",
        ]
        if force:
            cmd_ann += ["--redo"]
        if "_test" not in xml_file:
            cmd_ann += ["--phase", phase]
        if win == 0.108:
            return cmd_ann, cmd_bayes
        else:
            return cmd_ann, None
    else:
        print(f"No .xml file found in {dir}")
        return None, None


def run_commands_for_mouse(commands):
    """Run all commands for a single mouse sequentially."""
    for cmd in commands:
        subprocess.run(cmd)


def run_commands_sequentially(mouse_commands):
    """Run all commands sequentially for all mice."""
    for dir, commands in mouse_commands.items():
        print(f"Processing mouse directory: {dir}")
        for cmd in commands:
            print(f"Running command: {cmd}")
            subprocess.run(cmd)
            print(f"Finished command: {cmd}")


def run_commands_parallel(mouse_commands):
    """Run all commands in parallel for all mice."""
    with ProcessPoolExecutor() as executor:
        for dir, commands in mouse_commands.items():
            print(f"Processing mouse directory in parallel: {dir}")
            executor.map(subprocess.run, commands)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runAllMice.py <mode> [force]")
        print("Modes: parallel")
        print("       !!!sequential!!!")
        print("Force: force")
        mode = "sequential"
    else:
        mode = sys.argv[1].lower()
    if mode not in ["sequential", "parallel"]:
        print("Invalid mode. Use 'sequential' or 'parallel'.")
        sys.exit(1)

    force = False
    if len(sys.argv) > 2 and sys.argv[2].lower() == "force":
        force = True

    dirs = [
        os.path.abspath(os.path.expanduser(d))
        for d in os.listdir(".")
        if os.path.isdir(d)
    ]
    print(f"Found directories: {dirs}")
    mouse_commands = {}
    for dir in dirs:
        if any(mouse in dir for mouse in mice_nb) or not mice_nb:
            if "M1199_MFB" not in dir:
                mouse_commands[dir] = []
                for win in win_values:
                    cmd_ann, cmd_bayes = process_directory(dir, win, force)
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
                            os.path.join(dir, dirmfb), win, force
                        )
                        if cmd_ann:
                            mouse_commands[os.path.join(dir, dirmfb)].append(cmd_ann)
                        if cmd_bayes:
                            mouse_commands[os.path.join(dir, dirmfb)].append(cmd_bayes)

    if mode == "sequential":
        run_commands_sequentially(mouse_commands)
    elif mode == "parallel":
        run_commands_parallel(mouse_commands)
