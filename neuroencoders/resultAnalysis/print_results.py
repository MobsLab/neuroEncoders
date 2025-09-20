# Get libs
import os
import stat
import sys
import warnings
from typing import Tuple

import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from neuroencoders.importData.gui_elements import (
    circular_mean_error,
    create_polar_colorbar,
    plot_circular_comparison,
)

EC = np.array([45, 39])  # range of x and y in cm
EC2 = np.array([44, 38])  # range of x and y in cm
EC3 = np.array([45, 35])  # range of x and y in cm


def print_results(
    dir: str,
    **kwargs,
):
    """
    This function is used to print the results of the decoding and save some associated figures.

    args:
    ----
    - dir: the directory where the results are stored (also known as folderResult)
    - show: if True, the figures will be shown
    - typeDec: the type of decoder used (NN or bayes)
    - euclidean: if True, the euclidean distance will be used
    - results: the results of the Decoding (mantadory to provide if typeDec is bayes)
    - lossSelection: the percentage of the best windows to selected
    - avoid_zero: if True, the zero values will be avoided and we will plot
     only the values with confidence above zero.
    - windowSizeMS: the size of the window in ms
    - force: if True, the figures will be re-computed even if they already exist
    - target: the target of the decoding (pos or lin or linAndThigmo #TODO)
    - phase: the phase of the experiment (e.g. "pre", "cond", "post")
    - with_hist_distribution: if True, the histogram distribution of the error will be plotted
    - save: if True, the figures will be saved (default: True)
    - training_data: the training data used for the decoding (optional for histogram)
    - useSpeedMask: if True, the speed mask will be used to select the windows

    return:
    ---
    - a tuple containing the mean euclidean error, the mean euclidean error of the selected windows, the mean linear error and the mean linear error of the selected windows if relevant

    It will automatically close the figures at the end.
    """

    # Manage kwargs arguments
    show = kwargs.pop("show", True)
    typeDec = kwargs.get("typeDec", "NN")
    euclidean = kwargs.get("euclidean", False)
    results = kwargs.get("results", [])
    lossSelection = kwargs.get("lossSelection", 0.3)
    avoid_zero = kwargs.get("avoid_zero", False)
    windowSizeMS = kwargs.get("windowSizeMS", 36)
    target = kwargs.get("target", "pos")
    phase = kwargs.get("phase", None)
    skip_linear_plots = kwargs.get("skip_linear_plots", None)
    l_function = kwargs.get("l_function", None)
    if skip_linear_plots is None:
        skip_linear_plots = False

    # Check if the directory exists
    if not os.path.isdir(dir):
        raise ValueError(f"The directory {dir} does not exist")

    outdir = os.path.join(dir, str(windowSizeMS))
    suffix = f"_{phase}" if phase is not None else ""

    # Manage arguments
    if typeDec == "bayes" and not results:
        try:
            import dill as pickle

            with open(
                os.path.join(outdir, f"bayes_decoding_results{suffix}.pkl"), "rb"
            ) as f:
                results = pickle.load(f)
        except Exception as e:
            raise ValueError(
                f"You should provide results from BayesTrainer.test() function: {e}"
            )

    block = show
    maxPos = 1
    # Get data
    if typeDec == "NN":
        predLossName = "entropy"
        pos = pd.read_csv(
            os.path.expanduser(
                os.path.join(dir, str(windowSizeMS), f"featureTrue{suffix}.csv")
            )
        ).values[:, 1:]
        inferring = pd.read_csv(
            os.path.expanduser(
                os.path.join(dir, str(windowSizeMS), f"featurePred{suffix}.csv")
            )
        ).values[:, 1:]
        speedMask = (
            pd.read_csv(
                os.path.expanduser(
                    os.path.join(dir, str(windowSizeMS), f"speedMask{suffix}.csv")
                )
            )
            .values[:, 1:]
            .flatten()
            .astype(bool)
        )
        posIndex = (
            pd.read_csv(
                os.path.expanduser(
                    os.path.join(dir, str(windowSizeMS), f"posIndex{suffix}.csv")
                )
            )
            .values[:, 1]
            .astype(int)
        )
        timeStepsPred = (
            pd.read_csv(
                os.path.expanduser(
                    os.path.join(dir, str(windowSizeMS), f"timeStepsPred{suffix}.csv")
                )
            )
            .values[:, 1]
            .astype(int)
        )
        try:
            qControl = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(dir, str(windowSizeMS), f"Hn{suffix}.csv")
                    )
                ).values[:, 1:]
            )
        except FileNotFoundError:
            qControl = np.random.uniform(0, 1, size=pos.shape[0])
        if os.path.isfile(
            os.path.expanduser(
                os.path.join(dir, str(windowSizeMS), f"linearPred{suffix}.csv")
            )
        ):
            linear = True
            lpos = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(dir, str(windowSizeMS), f"linearTrue{suffix}.csv")
                    )
                ).values[:, 1:]
            )
            linferring = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(dir, str(windowSizeMS), f"linearPred{suffix}.csv")
                    )
                ).values[:, 1:]
            )
        else:
            linear = False
    elif typeDec == "bayes":
        try:
            predLossName = "proba"
            pos = results["featureTrue"]
            inferring = results["featurePred"]
            qControl = np.squeeze(results[predLossName])
            if "linearPred" in results.keys():
                linear = True
                lpos = results["linearTrue"]
                linferring = results["linearPred"]
            else:
                linear = False
            # create a fake speedMask for all timepoints
            speedMask = np.ones(
                shape=pos.shape[0], dtype=bool
            )  # TODO:dont know how to use speedMask in bayes, it returns the full speedFilter mask which is not what we want.
            posIndex = np.arange(pos.shape[0])
            timeStepsPred = results["times"]
        except KeyError as e:
            print(
                "Error: results should contain the keys 'featureTrue', 'featurePred', 'proba' and 'times'.",
                e,
                "Will try to load the data from the folder.",
            )
            predLossName = "proba"
            pos = pd.read_csv(
                os.path.expanduser(
                    os.path.join(
                        dir, str(windowSizeMS), f"bayes_featureTrue{suffix}.csv"
                    )
                )
            ).values[:, 1:]
            inferring = pd.read_csv(
                os.path.expanduser(
                    os.path.join(
                        dir, str(windowSizeMS), f"bayes_featurePred{suffix}.csv"
                    )
                )
            ).values[:, 1:]
            qControl = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(
                            dir, str(windowSizeMS), f"bayes_{predLossName}{suffix}.csv"
                        )
                    )
                ).values[:, 1:]
            )
            timeStepsPred = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(
                            dir, str(windowSizeMS), f"bayes_timeStepsPred{suffix}.csv"
                        )
                    )
                ).values[:, 1:]
            )
            speedMask = np.ones(
                shape=pos.shape[0], dtype=bool
            )  # TODO:dont know how to use speedMask in bayes, it returns the full speedFilter mask which is not what we want.
            posIndex = np.arange(pos.shape[0])

            if os.path.isfile(
                os.path.expanduser(
                    os.path.join(
                        dir, str(windowSizeMS), f"bayes_linearPred{suffix}.csv"
                    )
                )
            ):
                linear = True
                lpos = np.squeeze(
                    pd.read_csv(
                        os.path.expanduser(
                            os.path.join(
                                dir, str(windowSizeMS), f"bayes_linearTrue{suffix}.csv"
                            )
                        )
                    ).values[:, 1:]
                )
                linferring = np.squeeze(
                    pd.read_csv(
                        os.path.expanduser(
                            os.path.join(
                                dir, str(windowSizeMS), f"bayes_linearPred{suffix}.csv"
                            )
                        )
                    ).values[:, 1:]
                )
            else:
                linear = False
    else:
        raise ValueError('typeDec should be either "NN" or bayes"')
    dimOutput = pos.shape[1]
    assert pos.shape[1] == inferring.shape[1]
    if euclidean:
        pos = pos * EC
        inferring = inferring * EC
        if typeDec == "NN":
            qControl = qControl * np.linalg.norm(EC)

    # Save the executable
    if typeDec == "NN":
        with open(os.path.join(outdir, "reDrawFigures"), "w") as f:
            f.write(sys.executable + " " + os.path.abspath(__file__) + " " + dir)
        st = os.stat(os.path.join(outdir, "reDrawFigures"))
        os.chmod(os.path.join(outdir, "reDrawFigures"), st.st_mode | stat.S_IEXEC)

    # Get the best <loss_selection*100>% of data
    qControltmp = qControl.copy()
    if avoid_zero:
        qControltmp = (
            qControltmp
            + np.abs(np.nanmin(qControltmp))
            + 0.1
            + np.random.normal(0, 0.01, qControltmp.shape)
        )
    temp = qControltmp.argsort(axis=0)
    if typeDec == "NN":
        thresh = np.squeeze(qControltmp[temp[int(len(temp) * lossSelection)]])
        selection = np.squeeze(qControl <= thresh)  # even with entropy, lower is better
    elif typeDec == "bayes":
        thresh = qControltmp[temp[int(len(temp) * (1 - lossSelection))]]
        selection = np.squeeze(qControl >= thresh)

    # move training data to 1D if needed
    training_data = kwargs.pop("training_data", None)
    useSpeedMask = kwargs.get("useSpeedMask", False)
    if training_data is not None and dimOutput == 1 and training_data.shape[1] >= 2:
        if l_function is None:
            twoDtraining_data = training_data.copy()
            _, training_data = l_function(twoDtraining_data[:, :2])
        else:
            raise ValueError(
                "l_function is None, if you want 1D plot with training_data, you should provide a l_function to convert the 2D data to 1D"
            )

    if training_data is not None and useSpeedMask:
        warnings.warn(
            "Warning: useSpeedMask is True, and training_data is provided, so will be using speedMask on data."
        )
        selection = np.logical_and(selection, speedMask)

    frames = np.where(selection)[0]
    print(
        "total windows:",
        len(temp),
        "| selected windows:",
        len(frames),
        "(thresh",
        thresh,
        " (",
        lossSelection * 100,
        "%)",
        ")",
    )
    # Calculate 1d and 2d errors
    num_cols = min(inferring.shape[1], 2)
    error = np.array(
        [
            np.linalg.norm(inferring[i, :num_cols] - pos[i, :num_cols])
            for i in range(inferring.shape[0])
        ]
    )  # eucledian distance
    if target.lower() != "pos":
        warnings.warn(
            f"You are using a target different from pos, the error will be calculated as the euclidean distance between inferred and true for all dimensions {target=}. You should not expect anything from this error."
        )
    print(
        "mean eucl. error:",
        np.nanmean(error) * maxPos,
        "| selected error:",
        np.nanmean(error[frames]) * maxPos,
    )
    if (
        np.isinf(np.nanmean(error) * maxPos) or np.isinf(np.nanmean(error[frames]))
    ) and target.lower() == "linandthigmo":
        warnings.warn(
            "Warning: Infinite values in the error - check your data. Will skip fig_interror"
        )
        skip_fig_interror = True
    else:
        skip_fig_interror = False

    if linear:
        if "pos" in target.lower():
            lError = np.array(
                [np.abs((linferring[n] - lpos[n])) for n in range(linferring.shape[0])]
            )
            print(
                "mean linear error:",
                np.nanmean(lError) * maxPos,
                "| selected error:",
                np.nanmean(lError[frames]) * maxPos,
            )
        elif "lin" in target.lower():
            lError = np.array(
                [np.abs(inferring[i, 0] - pos[i, 0]) for i in range(inferring.shape[0])]
            )
            linferring = inferring
            # bypass Linear as it's not really relevant
            skip_linear_plots = True
            print(
                "mean linear error:",
                np.nanmean(lError),
                "| selected error:",
                np.nanmean(lError[frames]) * maxPos,
            )
        else:
            warnings.warn("is this a new kind of target? You should check it.")
            lError = np.array(
                [np.abs(inferring[i, 0] - pos[i, 0]) for i in range(inferring.shape[0])]
            )
            linferring = inferring
            # bypass Linear as it's not really relevant
            skip_linear_plots = True
            print(
                "mean linear error:",
                np.nanmean(lError),
                "| selected error:",
                np.nanmean(lError[frames]) * maxPos,
            )

    if skip_linear_plots is None:
        skip_linear_plots = False

    sys.stdout.write("threshold value: " + str(thresh) + "\r")
    sys.stdout.flush()

    # Plot things
    if (np.nanmax(qControl) != 0) | (
        np.nanmax(qControl) == 0 and avoid_zero
    ) and not skip_fig_interror:
        qControl = qControltmp
        thresh, selection = fig_interror(
            pos=pos,
            Error=error,
            q_control=qControl,
            selection=selection,
            thresh=thresh,
            outfolder=outdir,
            dimOutput=dimOutput,
            show=block,
            name="pos",
            plotMaze=target == "pos",
            speedMask=speedMask,
            posIndex=posIndex,
            timeStepsPred=timeStepsPred,
            **kwargs,
        )
        thresh, selection = fig_interror(
            pos=inferring,
            Error=error,
            q_control=qControl,
            selection=selection,
            thresh=thresh,
            outfolder=outdir,
            dimOutput=dimOutput,
            show=block,
            name="inferring",
            speedMask=speedMask,
            plotMaze=target == "pos",
            posIndex=posIndex,
            timeStepsPred=timeStepsPred,
            **kwargs,
        )
    else:
        warnings.warn("qControl is all zeros, not plotting the interactive figure")
    dimOutputs = [dimOutput]
    if target.lower() == "posandheaddirectionandthigmo" and dimOutput == 4:
        dimOutputs = [dimOutput, 2]
    for dimOut in dimOutputs:
        overview_fig(
            pos=pos,
            inferring=inferring,
            selection=selection,
            outfolder=outdir,
            dimOutput=dimOut,
            show=block,
            speedMask=speedMask,
            training_data=training_data,
            posIndex=posIndex,
            timeStepsPred=timeStepsPred,
            **kwargs,
        )

    if not skip_linear_plots:
        if training_data is not None and training_data.shape[1] == 2:
            if l_function is not None:
                twoDtraining_data = training_data.copy()
                _, training_data = l_function(twoDtraining_data)
            else:
                raise ValueError(
                    "l_function is None, if you want 1D plot with training_data, you should provide a l_function to convert the 2D data to 1D"
                )
        thresh, selection = fig_interror(
            pos=lpos,
            Error=lError,
            q_control=qControl,
            selection=selection,
            thresh=thresh,
            outfolder=outdir,
            dimOutput=1,
            speedMask=speedMask,
            show=block,
            name="lpos",
            plotMaze=target == "pos",
            posIndex=posIndex,
            timeStepsPred=timeStepsPred,
            **kwargs,
        )
        thresh, selection = fig_interror(
            pos=linferring,
            Error=lError,
            q_control=qControl,
            selection=selection,
            thresh=thresh,
            outfolder=outdir,
            dimOutput=1,
            show=block,
            speedMask=speedMask,
            name="linferring",
            plotMaze=target == "pos",
            posIndex=posIndex,
            timeStepsPred=timeStepsPred,
            **kwargs,
        )
        overview_fig(
            pos=lpos,
            inferring=linferring,
            selection=selection,
            outfolder=outdir,
            dimOutput=1,
            show=block,
            speedMask=speedMask,
            training_data=training_data,
            posIndex=posIndex,
            timeStepsPred=timeStepsPred,
            **kwargs,
        )

    return (
        np.nanmean(error) * maxPos,
        np.nanmean(error[frames]) * maxPos,
        np.nanmean(lError) if linear else None,
        np.nanmean(lError[frames]) * maxPos if linear else None,
    )

    ### Figures
    # Overview


def overview_fig(
    pos,
    inferring,
    selection,
    outfolder,
    dimOutput=2,
    show=False,
    typeDec="NN",
    force=True,
    with_hist_distribution: bool = True,
    speedMask=None,
    **kwargs,
):
    """
    This function is used to plot the overview figure of the decoding.

    args:
    ----
    - pos: the true position of the mouse
    - inferring: the inferred position of the mouse
    - selection: the selected windows
    - outfolder: the folder where to save the figure
    - dimOutput: the dimension of the output (1, 2, or 4 for PosAndHeadDirectionAndThigmo)
    - show: if True, the figure will be shown (default: True)
    - typeDec: the type of decoder used (NN or bayes)
    - force: if True, the figure will be re-computed even if it already exists
    - with_hist_distribution: if True, the histogram distribution of the error will be plotted
    - speedMask : a mask to select the speed of the mouse (optional, if not provided, all data will be used)
    - **kwargs: additional arguments that can be passed to the function, such as:
        - target: the target of the decoding (pos or lin or linAndThigmo or PosAndHeadDirectionAndThigmo)
        - phase: the phase of the experiment (e.g. "pre", "cond", "post")
        - training_data: the data used for training the decoder (optional for histogram)
        - posIndex: the index of the position in the original data (optional, used for plotting)
        - timeStepsPred: the time steps of the predictions (optional, used for plotting)

    return:
    ----
    - None
    """
    phase = kwargs.pop("phase", None)
    target = kwargs.pop("target", "pos")
    training_data = kwargs.pop("training_data", None)
    posIndex = kwargs.pop("posIndex", None)
    timeStepsPred = kwargs.pop("timeStepsPred", None)
    if timeStepsPred is None:
        raise ValueError(
            "timeStepsPred should be provided in kwargs, it is mandatory for plotting the overview figure. Otherwise you need to implement the range(pos.shape[0]) as timeStepsPred."
        )
    useSpeedMask = kwargs.get("useSpeedMask", False)

    suffix = f"_{phase}" if phase is not None else ""
    if (
        os.path.isfile(
            os.path.expanduser(
                os.path.join(
                    outfolder, f"overviewFig_{dimOutput}d_{typeDec}{suffix}.png"
                )
            )
        )
        and not force
    ):
        return

    # Define dimension names based on target
    if target.lower() == "pos":
        dim_names = ["X", "Y", "Linear Position"]
    elif target.lower() == "lin":
        dim_names = ["Linear Position", "Thigmo", "Linear Position"]
    elif target.lower() == "linandthigmo":
        dim_names = ["Linear Position", "Thigmo", "Linear Position"]
    elif target.lower() == "linanddirection":
        dim_names = ["Linear Position", "Direction", "Linear Position"]
    elif target.lower() == "direction":
        dim_names = ["Direction"]
    elif target.lower() == "posandheaddirectionandthigmo":
        if dimOutput != 4:
            dim_names = ["X", "Y", "Linear Position", "Distance to Wall"]
        else:
            dim_names = ["X", "Y", "Head Direction", "Distance to Wall"]

    else:
        dim_names = ["position 0", "position 1", "Linear Position"]

    pos_to_show_on_histograms = training_data if training_data is not None else pos

    if training_data is not None:
        speedMask_on_histograms = np.ones(
            shape=pos_to_show_on_histograms.shape[0], dtype=bool
        )
    elif speedMask is not None:
        speedMask_on_histograms = speedMask
    else:
        speedMask_on_histograms = np.ones(
            shape=pos_to_show_on_histograms.shape[0], dtype=bool
        )

    speedMask = (
        speedMask if speedMask is not None else np.ones(shape=pos.shape[0], dtype=bool)
    )

    shown = "speedMask on test" if training_data is None else "speedMask on fullTrain"

    if useSpeedMask:
        selection = np.logical_and(selection, speedMask)
    else:
        shown = "all test data, no speedMask"

    # Main plotting logic
    if target.lower() == "posandheaddirectionandthigmo" and dimOutput == 4:
        # Create comprehensive figure with multiple subplots
        fig = plt.figure()

        # Main (x,y) scatter plot with head direction coloring
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)

        # Use head direction for coloring
        # rescale head direction to good radians in [-pi, pi]
        inferring[:, 2] = ((inferring[:, 2] + np.pi) % (2 * np.pi)) - np.pi
        scatter = ax_main.scatter(
            inferring[selection, 0],
            inferring[selection, 1],
            c=inferring[selection, 2],
            cmap="hsv",
            s=30,
            alpha=0.7,
            label="Predicted positions",
        )

        # Add colorbar for head direction
        cbar = create_polar_colorbar(fig, scatter, ax_main.get_position())
        # cbar = plt.colorbar(scatter, ax=ax_main)
        # cbar.set_label("Head Direction (radians)", rotation=270, labelpad=15)

        ax_main.set_xlabel("X Position")
        ax_main.set_ylabel("Y Position")
        ax_main.set_title("Position Predictions Colored by Head Direction")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # Circular plot for head direction
        ax_circular = plt.subplot2grid((3, 4), (0, 2))
        plot_circular_comparison(
            ax_circular,
            inferring[selection, 2],
            pos[selection, 2],
            "Head Direction Comparison",
        )

        # Distance to wall time series
        ax_dist = plt.subplot2grid((3, 4), (0, 3))
        ax_dist.plot(
            timeStepsPred[selection],
            inferring[selection, 3],
            "r-",
            alpha=0.7,
            label="Predicted distance",
        )
        ax_dist.plot(
            timeStepsPred[selection],
            pos[selection, 3],
            "b-",
            alpha=0.7,
            label="True distance",
        )
        ax_dist.set_xlabel("Time")
        ax_dist.set_ylabel("Distance to Wall")
        ax_dist.set_title("Distance to Wall Over Time")
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)

        # Time series for X and Y positions
        ax_x_time = plt.subplot2grid((3, 4), (1, 2))
        ax_x_time.plot(
            timeStepsPred[selection],
            inferring[selection, 0],
            "r-",
            alpha=0.7,
            label="Predicted X",
        )
        ax_x_time.plot(
            timeStepsPred[selection], pos[selection, 0], "b-", alpha=0.7, label="True X"
        )
        ax_x_time.set_xlabel("Time")
        ax_x_time.set_ylabel("X Position")
        ax_x_time.set_title("X Position Over Time")
        ax_x_time.legend()
        ax_x_time.grid(True, alpha=0.3)

        ax_y_time = plt.subplot2grid((3, 4), (1, 3))
        ax_y_time.plot(
            timeStepsPred[selection],
            inferring[selection, 1],
            "r-",
            alpha=0.7,
            label="Predicted Y",
        )
        ax_y_time.plot(
            timeStepsPred[selection], pos[selection, 1], "b-", alpha=0.7, label="True Y"
        )
        ax_y_time.set_xlabel("Time")
        ax_y_time.set_ylabel("Y Position")
        ax_y_time.set_title("Y Position Over Time")
        ax_y_time.legend()
        ax_y_time.grid(True, alpha=0.3)

        # Error analysis plots
        ax_pos_error = plt.subplot2grid((3, 4), (2, 0))
        pos_error = np.sqrt(
            (inferring[selection, 0] - pos[selection, 0]) ** 2
            + (inferring[selection, 1] - pos[selection, 1]) ** 2
        )
        ax_pos_error.hist(pos_error, bins=30, alpha=0.7, color="purple")
        ax_pos_error.set_xlabel("Position Error")
        ax_pos_error.set_ylabel("Frequency")
        ax_pos_error.set_title("Position Error Distribution")
        ax_pos_error.grid(True, alpha=0.3)

        ax_dir_error = plt.subplot2grid((3, 4), (2, 1))
        dir_error = np.abs(
            circular_mean_error(inferring[selection, 2], pos[selection, 2])
        )
        ax_dir_error.hist(dir_error, bins=30, alpha=0.7, color="orange")
        ax_dir_error.set_xlabel("Head Direction Error (radians)")
        ax_dir_error.set_ylabel("Frequency")
        ax_dir_error.set_title("Head Direction Error Distribution")
        ax_dir_error.grid(True, alpha=0.3)

        ax_dist_error = plt.subplot2grid((3, 4), (2, 2))
        dist_error = np.abs(inferring[selection, 3] - pos[selection, 3])
        ax_dist_error.hist(dist_error, bins=30, alpha=0.7, color="green")
        ax_dist_error.set_xlabel("Distance Error")
        ax_dist_error.set_ylabel("Frequency")
        ax_dist_error.set_title("Distance to Wall Error Distribution")
        ax_dist_error.grid(True, alpha=0.3)

        # Summary statistics
        ax_stats = plt.subplot2grid((3, 4), (2, 3))
        ax_stats.axis("off")
        stats_text = f"""Summary Statistics:

Position RMSE: {np.sqrt(np.mean(pos_error**2)):.3f}
Direction MAE: {np.mean(dir_error):.3f} rad
Distance MAE: {np.mean(dist_error):.3f}

N samples: {selection.sum()}
        """
        ax_stats.text(
            0.1,
            0.9,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        fig.savefig(
            os.path.join(
                outfolder, f"4d_overviewFig_{dimOutput}d_{typeDec}{suffix}.png"
            ),
            dpi=150,
        )
        if show:
            plt.show(block=True)
        plt.close(fig)

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            inferring[selection, 0],
            inferring[selection, 1],
            c=inferring[selection, 3],
            label="Predicted positions",
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Distance to Wall", rotation=270, labelpad=15)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Position Predictions Colored by Distance to Wall")
        ax.legend()
        if show:
            plt.show(block=True)
        fig.savefig(
            os.path.join(
                outfolder, f"overviewFig_{dimOutput}d_{typeDec}_DistWall{suffix}.png"
            ),
            dpi=150,
        )
        plt.close(fig)
        return

    elif dimOutput == 2:
        fig, ax = plt.subplots()
        if target.lower() != "linanddirection":
            for dim in range(dimOutput):
                if dim > 0:
                    ax1 = plt.subplot2grid(
                        (dimOutput, 4 if not with_hist_distribution else 5),
                        (dim, 0),
                        sharex=ax1,
                        colspan=4,
                    )
                else:
                    ax1 = plt.subplot2grid(
                        (dimOutput, 4 if not with_hist_distribution else 5),
                        (dim, 0),
                        colspan=4,
                    )
                    plt.setp(ax1.get_xticklabels(), visible=False)
                ax1.scatter(
                    timeStepsPred[selection],
                    inferring[selection, dim],
                    c=inferring[selection, 2]
                    if (
                        target.lower() == "posandheaddirectionandthigmo"
                        and inferring.shape[1] > 2
                    )
                    else None,
                    cmap="hsv"
                    if (
                        target.lower() == "posandheaddirectionandthigmo"
                        and inferring.shape[1] > 2
                    )
                    else None,
                    label=f"guessed {dim_names[dim]} selection",
                    s=20,
                )
                ax1.scatter(
                    timeStepsPred if not useSpeedMask else timeStepsPred[selection],
                    pos[:, dim] if not useSpeedMask else pos[selection, dim],
                    s=24,
                    alpha=0.6,
                    label=f"true {dim_names[dim]}" + str(dim),
                    color="xkcd:dark pink",
                )
                if with_hist_distribution and selection.sum() > 0:
                    ax2 = plt.subplot2grid((dimOutput, 5), (dim, 4), sharey=ax1)
                    isfinit = np.isfinite(inferring[:, dim])
                    ax2.hist(
                        inferring[isfinit & selection, dim],
                        bins=50,
                        alpha=0.5,
                        orientation="horizontal",
                        label=f"guessed {dim_names[dim]} distribution",
                        density=True,
                    )
                    ax2.hist(
                        pos[selection, dim],
                        bins=50,
                        color="xkcd:pink",
                        alpha=0.5,
                        orientation="horizontal",
                        label=f"true {dim_names[dim]} distribution on selection (test set)",
                        density=True,
                    )
                    ax2.hist(
                        pos_to_show_on_histograms[speedMask_on_histograms, dim],
                        bins=50,
                        color="xkcd:neon purple",
                        alpha=0.5,
                        orientation="horizontal",
                        label=f"true {dim_names[dim]} distribution ({shown})",
                        density=True,
                    )
                    plt.setp(ax2.get_yticklabels(), visible=False)
                    ax2.set_xlabel(f"{dim_names[dim]} distribution")
                if dim > 0:
                    handles_1, labels_1 = ax1.get_legend_handles_labels()
                    handles_2, labels_2 = ax2.get_legend_handles_labels()
                    all_handles = handles_1 + handles_2
                    all_labels = labels_1 + labels_2
                    fig.legend(
                        all_handles,
                        all_labels,
                        loc="lower center",
                        fontsize="x-small",
                        bbox_to_anchor=(0.5, 0),
                    )
                    ax1.set_xlabel("time")
                ax1.set_ylabel(f"{dim_names[dim]}")
        else:
            # Keep existing linanddirection logic
            from neuroencoders.importData.gui_elements import ModelPerformanceVisualizer

            ax1 = plt.subplot2grid(
                (1, 4 if not with_hist_distribution else 5), (0, 0), colspan=4
            )

            from matplotlib.colors import ListedColormap

            ax1.plot(
                timeStepsPred[selection],
                inferring[selection, 0],
                "--.",
                zorder=2,
                linewidth=0.7,
                alpha=0.5,
            )
            ax1.scatter(
                timeStepsPred[selection],
                inferring[selection, 0],
                s=36,
                c=inferring[selection, 1],
                marker="o",
                alpha=1,
                cmap=ListedColormap(["hotpink", "cornflowerblue"]),
                zorder=3,
            )
            ax1.plot(
                timeStepsPred if not useSpeedMask else timeStepsPred[selection],
                pos[:, 0] if not useSpeedMask else pos[selection, 0],
                ".-" if not useSpeedMask else "-",
                markersize=6,
                alpha=0.6,
                label=f"true {dim_names[0]}",
                color="xkcd:dark pink",
                zorder=1,
            )
            handles, labels = plt.gca().get_legend_handles_labels()
            shock_patch = Line2D(
                [0],
                [0],
                marker="o",
                color="hotpink",
                label="Guessed Linear Position Towards Shock",
            )
            safe_patch = Line2D(
                [0],
                [0],
                marker="o",
                color="cornflowerblue",
                label="Guessed LinearPos Away from Shock",
            )
            all_handles = handles + [shock_patch, safe_patch]
            all_labels = labels + [
                "Guessed Linear Position Towards Shock",
                "Guessed LinearPos Away from Shock",
            ]
            plt.legend(handles=all_handles, labels=all_labels, loc="upper left")

            ax1.set_title(f"{dim_names[0]} + {dim_names[1]}")
            visualizer = ModelPerformanceVisualizer(
                predictions=inferring[selection, 1],
                ground_truth=pos[selection, 1],
                timestamps=np.where(selection)[0],
            )
            ax1.set_xlabel("time")
            ax1.set_ylabel("Distance to Shock (lin. pos)")
            if with_hist_distribution and selection.sum() > 0:
                ax2 = plt.subplot2grid((1, 5), (0, 4), sharey=ax1)
                isfinit = np.isfinite(inferring[:, 0])
                ax2.hist(
                    inferring[isfinit & selection, 0],
                    bins=50,
                    alpha=0.5,
                    orientation="horizontal",
                    label=f"guessed {dim_names[0]} distribution",
                    density=True,
                )
                ax2.hist(
                    pos[selection, 0],
                    bins=50,
                    color="xkcd:pink",
                    alpha=0.5,
                    orientation="horizontal",
                    label=f"true {dim_names[0]} distribution on selection (test set)",
                    density=True,
                )
                ax2.hist(
                    pos_to_show_on_histograms[speedMask_on_histograms, 0],
                    bins=50,
                    color="xkcd:neon purple",
                    alpha=0.5,
                    orientation="horizontal",
                    label=f"true {dim_names[0]} distribution ({shown})",
                    density=True,
                )
                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_xlabel(f"{dim_names[0]} distribution")
                ax2.legend(fontsize="xx-small")
            else:
                if selection.sum() > 0:
                    ax3 = plt.subplot2grid((1, 5), (0, 4))
                    visualizer._plot_error_distribution(ax=ax3)

    elif dimOutput == 1:
        if target.lower() != "direction":
            fig, ax = plt.subplots()
            ax2 = plt.subplot2grid(
                (1, 4 if not with_hist_distribution else 5), (0, 0), colspan=4
            )
            ax2.scatter(
                timeStepsPred[selection],
                inferring[selection],
                s=20,
                label=f"guessed {dim_names[2]} selection",
            )
            ax2.scatter(
                timeStepsPred if not useSpeedMask else timeStepsPred[selection],
                pos if not useSpeedMask else pos[selection],
                s=24,
                alpha=0.6,
                label=f"true {dim_names[2]}",
                color="xkcd:dark pink",
            )
            ax2.legend(
                loc="lower center",
                fontsize="x-small",
                bbox_to_anchor=(0.5, -0.1),
            )
            ax2.set_title(f"{dim_names[2]}")
            if with_hist_distribution and selection.sum() > 0:
                ax3 = plt.subplot2grid((1, 5), (0, 4), sharey=ax2)
                isfinit = np.isfinite(inferring)
                ax3.hist(
                    inferring[isfinit & selection],
                    bins=50,
                    alpha=0.5,
                    orientation="horizontal",
                    label=f"guessed {dim_names[2]} distribution",
                    density=True,
                )
                ax3.hist(
                    pos[selection],
                    bins=50,
                    color="xkcd:pink",
                    alpha=0.5,
                    orientation="horizontal",
                    label=f"true {dim_names[2]} distribution on selection (test set)",
                    density=True,
                )
                ax3.hist(
                    pos_to_show_on_histograms[speedMask_on_histograms],
                    bins=50,
                    color="xkcd:neon purple",
                    alpha=0.5,
                    orientation="horizontal",
                    label=f"true {dim_names[2]} distribution ({shown})",
                    density=True,
                )
                plt.setp(ax3.get_yticklabels(), visible=False)
                ax3.set_xlabel(f"{dim_names[2]} distribution")
                ax3.legend(
                    loc="lower center",
                    fontsize="xx-small",
                    bbox_to_anchor=(0.5, -0.1),
                )
        else:
            from neuroencoders.importData.gui_elements import ModelPerformanceVisualizer

            fig, ax1 = plt.subplots()
            visualizer = ModelPerformanceVisualizer(
                predictions=inferring[selection, 1],
                ground_truth=pos[selection, 1],
                timestamps=np.where(selection)[0],
            )
            ax1 = plt.subplot2grid((dimOutput, 2), (1, 0))
            visualizer._plot_main_timeline(ax=ax1)
            if selection.sum() > 0:
                ax2 = plt.subplot2grid((dimOutput, 2), (1, 1))
                visualizer._plot_error_distribution(ax=ax2)

    plt.subplots_adjust(bottom=0.15)
    if kwargs.get("save", True):
        plt.savefig(
            os.path.expanduser(
                os.path.join(
                    outfolder, f"overviewFig_{dimOutput}d_{typeDec}{suffix}.png"
                )
            ),
            bbox_inches="tight",
        )
    if show:
        if mplt.get_backend() == "QtAgg":
            plt.get_current_fig_manager().window.showMaximized()
        elif mplt.get_backend() == "TkAgg":
            plt.get_current_fig_manager().resize(
                *plt.get_current_fig_manager().window.maxsize()
            )
        plt.show(block=True)

    plt.close("all")
    print()

    # Interactive figure 2D


def fig_interror(
    pos: np.ndarray,
    Error: np.ndarray,
    q_control: np.ndarray,
    selection: np.ndarray,
    thresh: float,
    outfolder: str,
    dimOutput=2,
    show: bool = False,
    force: bool = True,
    phase=None,
    name=None,
    plotMaze: bool = False,
    **kwargs,
) -> Tuple[float, np.ndarray]:
    """
    This function is used to plot the interactive figure of the decoding and returns the new threshold value set by the slider.

        args:
        ----
        - pos: the true position of the mouse
        - Error: the error of the decoding
        - q_control: the control of the decoding
        - selection: the selected windows
        - thresh: the threshold of the selection
        - outfolder: the folder where to save the figure
        - dimOutput: the dimension of the output (1 or 2)
        - show: if True, the figure will be shown
        - typeDec: the type of decoder used (NN or bayes)
        - force: if True, the figure will be re-computed even if it already exists
        - save: if True, the figure will be saved

        returns:
        ----
        - new_thresh: the new threshold value set by the slider
        - selection: the selection of the windows based on the threshold
    """
    typeDec = kwargs.pop("typeDec", "NN")
    new_thresh = thresh
    timeStepsPred = kwargs.pop("timeStepsPred", None)
    speedMask = kwargs.pop("speedMask", None)
    if speedMask is None:
        print("Warning: speedMask is None, using all data")
        speedMask = np.ones(shape=pos.shape[0], dtype=bool)
    useSpeedMask = kwargs.get("useSpeedMask", False)
    if useSpeedMask:
        warnings.warn(
            "useSpeedMask is True, so will be using speedMask to select the windows in fig_interror."
        )

    if dimOutput > 2:
        # WARN: this is very hacky and specific
        warnings.warn(
            f"dimOutput should be 1 or 2 but is {dimOutput}, setting to 2 and plotting first 2 dimensions only."
        )
        dimOutput = 2
        pos = pos[:, :2]

    suffix = f"_{phase}" if phase is not None else ""
    suffix = f"{suffix}_{name}" if name is not None else suffix
    if (
        os.path.isfile(
            os.path.expanduser(
                os.path.join(outfolder, f"errorFig_{dimOutput}d_{typeDec}{suffix}.png")
            )
        )
        and not force
    ):
        return
    fig, ax = plt.subplots()
    from matplotlib.widgets import Slider
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig.tight_layout(pad=3.0)
    ax = plt.subplot2grid((1, 2), (0, 0))
    if dimOutput == 2:
        s = plt.scatter(
            pos[selection, 0], pos[selection, 1], c=Error[selection], s=10
        )  # X and Y
        plt.axis("scaled") if not plotMaze else plt.axis("equal")

        if plotMaze:
            maze_coords = np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.65, 0.0],
                    [0.65, 0.75],
                    [0.35, 0.75],
                    [0.35, 0.0],
                    [0.0, 0.0],
                ]
            )
            ax.plot(
                maze_coords[:, 0],
                maze_coords[:, 1],
                color="k",
                linewidth=3,
            )
    elif dimOutput == 1:
        s = plt.scatter(
            timeStepsPred[selection], pos[selection], c=Error[selection], s=10
        )
    rangey = ax.get_ylim()[1] - ax.get_ylim()[0]
    rangex = ax.get_xlim()[1] - ax.get_xlim()[0]
    if dimOutput == 2:
        crange = max(rangex, rangey)
    elif dimOutput == 1:
        crange = rangey
    plt.clim(0, crange)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label("decoding error", rotation=270)
    ax.set_title(f"decoding error, {suffix=}, {useSpeedMask=}")
    fig.suptitle(
        f"decoding error, {suffix=}, threshold: {thresh:.2f} (selected error: {np.nanmean(Error):.2f})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(val):
        nonlocal new_thresh, selection
        new_thresh = val
        thresh_line.set_xdata([val, val])  # Changed from set_ydata to set_xdata
        if typeDec == "NN":
            selection = np.squeeze(q_control <= val)
            if useSpeedMask:
                selection = np.logical_and(selection, speedMask)
        elif typeDec == "bayes":
            selection = np.squeeze(q_control >= val)
            if useSpeedMask:
                selection = np.logical_and(selection, speedMask)

        if dimOutput == 2:
            s.set_offsets(pos[selection, :])
            s.set_array(Error[selection])
            sel2 = selection[selNoNans]
            scatt.set_array(np.array([1 if sel2[n] else 0.2 for n in range(len(sel2))]))
        elif dimOutput == 1:
            s.set_offsets(np.c_[np.where(selection)[0], pos[selection]])
            s.set_array(Error[selection])
            sel2 = selection[selNoNans]
            scatt.set_array(np.array([1 if sel2[n] else 0.2 for n in range(len(sel2))]))
        error = np.mean(Error[selection])
        sys.stdout.write("threshold value: " + str(val) + "\r")
        sys.stdout.flush()
        fig.suptitle(
            f"decoding error, {suffix=}, threshold: {val:.2f} (selected error: {error:.2f})"
        )
        fig.canvas.draw_idle()

    axcolor = "lightgoldenrodyellow"
    axfreq = plt.axes([0.1, 0.15, 0.25, 0.03], facecolor=axcolor)
    slider = Slider(
        axfreq,
        "selection",
        np.nanmin(q_control[np.isfinite(q_control)]),
        np.nanmax(q_control[np.isfinite(q_control)]),
        valinit=thresh,
        valstep=max(
            np.nanmax(q_control[np.isfinite(q_control)])
            - np.nanmin(q_control[np.isfinite(q_control)]),
            0.1,
        )
        / 100,
    )
    slider.on_changed(update)

    # ERROR & STD - AXES SWAPPED
    ax_error = plt.subplot2grid((1, 2), (0, 1))
    selNoNans = np.squeeze(~np.isnan(Error))
    if useSpeedMask:
        selection = np.logical_and(selection, speedMask)

    sel2 = selection[selNoNans]
    z = [1 if sel2[n] else 0.2 for n in range(len(sel2))]
    # Swapped: q_control on x-axis, Error on y-axis
    scatt = ax_error.scatter(
        q_control[selNoNans],
        Error[selNoNans],
        c=z,
        s=10,
        cmap=plt.cm.get_cmap("Greys"),
        vmin=0,
        vmax=1,
    )

    # plot kde overlays of error and q control based on speedMask if useSpeedMask is True
    if useSpeedMask:
        fast_speed_color = "xkcd:mustard yellow"
        slow_speed_color = "xkcd:violet"
        sns.kdeplot(
            x=q_control[selNoNans & speedMask],
            y=Error[selNoNans & speedMask],
            ax=ax_error,
            color=fast_speed_color,
            alpha=0.6,
            levels=5,
            label="fast speed",
        )
        sns.kdeplot(
            x=q_control[selNoNans & ~speedMask],
            y=Error[selNoNans & ~speedMask],
            ax=ax_error,
            color=slow_speed_color,
            alpha=0.6,
            levels=5,
            label="slow speed",
        )
        # FIXED: X-axis marginal distribution (top)
        ax_x_dist = inset_axes(
            ax_error,
            width="100%",
            height="30%",
            loc="upper center",
            bbox_to_anchor=(0, 0.7, 1, 0.3),  # Moved up slightly
            bbox_transform=ax_error.transAxes,
        )
        ax_x_dist.set_facecolor("none")

        ax_x_dist.hist(
            q_control[selNoNans & speedMask],
            bins=100,
            alpha=0.7,
            color=fast_speed_color,
            density=True,
            label="fast speed",
        )
        ax_x_dist.hist(
            q_control[selNoNans & ~speedMask],
            bins=100,
            alpha=0.7,
            color=slow_speed_color,
            density=True,
            label="slow speed",
        )

        # CRITICAL FIXES:
        ax_x_dist.set_xlim(ax_error.get_xlim())  # Match main plot x-limits
        ax_x_dist.tick_params(labelbottom=False)  # Remove x-axis labels
        ax_x_dist.set_ylabel("Density", fontsize=10)

        # Add Y-axis marginal distribution (right)
        ax_y_dist = inset_axes(
            ax_error,
            width="30%",
            height="100%",
            loc="center right",
            bbox_to_anchor=(0.7, 0, 0.3, 1),  # Moved right slightly
            bbox_transform=ax_error.transAxes,
        )

        ax_y_dist.set_facecolor("none")
        ax_y_dist.hist(
            Error[selNoNans & speedMask],
            bins=100,
            alpha=0.7,
            color=fast_speed_color,
            density=True,
            orientation="horizontal",
        )
        ax_y_dist.hist(
            Error[selNoNans & ~speedMask],
            bins=100,
            alpha=0.7,
            color=slow_speed_color,
            density=True,
            orientation="horizontal",
        )

        # CRITICAL FIXES:
        ax_y_dist.set_ylim(ax_error.get_ylim())  # Match main plot y-limits
        ax_y_dist.tick_params(labelleft=False)  # Remove y-axis labels
        ax_y_dist.set_xlabel("Density", fontsize=10)

        # Main plot settings
        ax_error.set_xlabel("q_control")
        ax_error.set_ylabel("Error")

        handles = [
            Line2D([0], [0], color=color)
            for color in [fast_speed_color, slow_speed_color]
        ]
        labels = ["fast speed", "slow speed"]
        ax_error.legend(handles, labels, loc="lower left")

    # plt.scatter(Error[selNoNans], inferring[selNoNans,dim_output], c=z, s=10)
    nBins = 30
    _, edges = np.histogram(q_control[selNoNans], nBins)
    histIdx = []
    for bin in range(nBins):
        temp = []
        for n in range(len(Error)):
            if not selNoNans[n]:
                continue
            if q_control[n] <= edges[bin + 1] and q_control[n] > edges[bin]:
                temp.append(n)
        histIdx.append(temp)

    # TODO: plot mean error
    # plot the mean error based on filtering values
    mean_error = []
    for bin in range(nBins):
        selection_tmp = np.squeeze(q_control <= edges[bin + 1])
        if useSpeedMask:
            selection_tmp = np.logical_and(selection_tmp, speedMask)
        error_values = np.nanmean(Error[selection_tmp])
        mean_error.append(np.mean(error_values))
    mean_error = np.array(mean_error)
    ax_error.plot(
        [edges[n + 1] for n in range(nBins)],
        0.8 + mean_error,
        alpha=0.5,
        c="xkcd:blue",
        label="mean error depending on filtering thresh",
        linewidth=3,
    )

    err = np.array(
        [
            [
                np.median(Error[histIdx[n]]) - np.percentile(Error[histIdx[n]], 30)
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ],
            [
                np.percentile(Error[histIdx[n]], 70) - np.median(Error[histIdx[n]])
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ],
        ]
    )
    # Swapped: q_control median on x-axis, bin centers on y-axis
    ax_error.errorbar(
        [(edges[n + 1] + edges[n]) / 2 for n in range(nBins) if len(histIdx[n]) > 10],
        [np.median(Error[histIdx[n]]) for n in range(nBins) if len(histIdx[n]) > 10],
        c="xkcd:cherry red",
        yerr=err,
        label=r"$median \pm 20 percentile$",
        linewidth=3,
    )
    # Create new range based on q_control values instead of Error
    x_new = np.linspace(
        np.min(
            [
                (edges[n + 1] + edges[n]) / 2
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ]
        ),
        np.max(
            [
                (edges[n + 1] + edges[n]) / 2
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ]
        ),
        num=len(Error),
    )
    # Fit polynomial with swapped axes: q_control vs Error (bin centers)
    coefs = poly.polyfit(
        [(edges[n + 1] + edges[n]) / 2 for n in range(nBins) if len(histIdx[n]) > 10],
        [np.median(Error[histIdx[n]]) for n in range(nBins) if len(histIdx[n]) > 10],
        2,
    )
    # coefs = poly.polyfit(Error[selNoNans], inferring[selNoNans,dim_output], 2)
    ffit = poly.polyval(x_new, coefs)
    ax_error.plot(x_new, ffit, "k", linewidth=3)
    # # add a linear fit to the plot = y =x
    # plt.plot(x_new, x_new, "c--", linewidth=1)

    thresh_line = ax_error.axvline(thresh, c="k")  # Changed from axhline to axvline
    ax_error.set_xlabel("evaluated loss")  # Swapped labels
    if typeDec == "bayes":
        ax_error.set_xlabel("bayes probability")
    ax_error.set_ylabel("decoding error")  # Swapped labels
    ax_error.set_title(f"evaluated loss vs. decoding error, {phase=}")  # Updated title

    if kwargs.get("save", True):
        plt.savefig(
            os.path.expanduser(
                os.path.join(outfolder, f"errorFig_{dimOutput}d_{typeDec}{suffix}.png")
            ),
            bbox_inches="tight",
        )
    if show:
        if mplt.get_backend() == "QtAgg":
            plt.get_current_fig_manager().window.showMaximized()
        elif mplt.get_backend() == "TkAgg":
            plt.get_current_fig_manager().resize(
                *plt.get_current_fig_manager().window.maxsize()
            )
        plt.show(block=True)
    plt.close("all")
    print()
    return new_thresh, selection


# Re-do (or do the figures)
if __name__ == "__main__":
    print_results(sys.argv[1], show=True)

    # # Movie
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax1 = plt.subplot2grid((1,1),(0,0))
    # im2, = ax1.plot([pos[0,1]*maxPos],[pos[0,0]*maxPos],marker='o', markersize=15, color="red")
    # im2b, = ax1.plot([inferring[0,1]*maxPos],[inferring[0,0]*maxPos],marker='P', markersize=15, color="green")

    # im3 = ax1.plot([125,170,170,215,215,210,60,45,45,90,90], [35,70,110,210,225,250,250,225,210,110,35], color="red")
    # im4 = ax1.plot([125,125,115,90,90,115,125], [100,215,225,220,185,100,100], color="red")
    # n = 135; nn=2*n
    # im4 = ax1.plot([nn-125,nn-125,nn-115,nn-90,nn-90,nn-115,nn-125], [100,215,225,220,185,100,100], color="red")
    # ax1.set_title('Decoding using full stack decoder', size=25)
    # ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)

    # def updatefig(frame, *args):
    #     reduced_frame = frame % len(frames)
    #     selected_frame = frames[reduced_frame]
    #     im2.set_data([pos[selected_frame,1]*maxPos],[pos[selected_frame,0]*maxPos])
    #     im2b.set_data([inferring[selected_frame,1]*maxPos],[inferring[selected_frame,0]*maxPos])
    #     return im2,im2b

    # save_len = len(frames)
    # ani = animation.FuncAnimation(fig,updatefig,interval=250, save_count=save_len)
    # ani.save(os.path.expanduser(folder+'/_tempMovie.mp4'))
    # fig.show()

    # np.savez(os.path.expanduser(fileName), pos, speed, inferring, trainLosses)
    # print('Results saved at:', fileName)
