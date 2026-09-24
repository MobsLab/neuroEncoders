"""
Shared ripple analysis utilities for paper figures.

This module provides reusable methods for analyzing neural network predictions
relative to ripple events, used by both PaperFigures and PaperFiguresSleep classes.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

from neuroencoders.utils.viz_params import white_viridis


def plot_ripple_losspredict_distribution(
    predloss_dict: Dict[str, List],
    ripple_indices_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
    filename_prefix: str = "distr_lossPred",
) -> None:
    """
    Plot histogram of predicted loss during ripples vs all times.

    Args:
        predloss_dict: Dictionary keyed by epoch with lists of predLoss arrays per window
        ripple_indices_dict: Dictionary keyed by epoch with lists of ripple indices per window
        time_windows: List of time windows in milliseconds
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
        filename_prefix: Prefix for saved figure filenames
    """
    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(10, 10),
        sharex="row",
        sharey="col",
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            lossPredInQ = predloss_dict[epoch_label][i]
            ripple_mask = ripple_indices_dict[epoch_label][i]

            ax[i, iepoch].hist(
                lossPredInQ[ripple_mask],
                bins=50,
                color="green",
                alpha=0.2,
                density=True,
                label=f"lossPred during ripples {epoch_label}",
            )
            ax[i, iepoch].axvline(
                np.nanmean(lossPredInQ[ripple_mask]),
                color="green",
            )
            ax[i, iepoch].hist(
                lossPredInQ,
                bins=50,
                color="red",
                alpha=0.2,
                density=True,
                label=f"lossPred during {epoch_label} (all time)",
            )
            ax[i, iepoch].axvline(
                np.nanmean(lossPredInQ),
                color="red",
            )
            if i == len(time_windows) - 1:
                ax[i, iepoch].set_xlabel("predicted loss")
            if i == 0:
                ax[i, iepoch].set_title(f"{epoch_label} {time_windows} ms")
            if epoch_label == epoch_labels[-1] and i == len(time_windows) - 1:
                ax[i, iepoch].legend()

    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_figures, f"{filename_prefix}.png"))
    fig.savefig(os.path.join(folder_figures, f"{filename_prefix}.svg"))


def plot_ripple_linear_pred_distribution(
    linpred_dict: Dict[str, List],
    ripple_indices_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
    during_ripples: bool = True,
) -> None:
    """
    Plot histogram of linear predicted position during/all ripple times.

    Args:
        linpred_dict: Dictionary keyed by epoch with lists of linPred arrays per window
        ripple_indices_dict: Dictionary keyed by epoch with lists of ripple indices per window
        time_windows: List of time windows
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
        during_ripples: Whether to filter for ripple times only
    """
    filename = (
        "distr_linearPred_during_ripples" if during_ripples else "distr_linearPred"
    )

    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(10, 10),
        sharex=True,
        sharey=True,
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            if during_ripples:
                mask = ripple_indices_dict[epoch_label][i]
            else:
                mask = np.arange(0, len(linpred_dict[epoch_label][i]))

            ax[i, iepoch].hist(linpred_dict[epoch_label][i][mask], bins=100)
            if i == 0:
                ax[i, iepoch].set_title(f"{epoch_label} {time_windows} ms")
            if i == len(time_windows) - 1:
                ax[i, iepoch].set_xlabel("predicted linear position")
            ax[i, iepoch].set_ylabel("count")

    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_figures, f"{filename}.png"))
    fig.savefig(os.path.join(folder_figures, f"{filename}.svg"))


def plot_ripple_position_vs_losspredict(
    linpred_dict: Dict[str, List],
    predloss_dict: Dict[str, List],
    ripple_indices_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
) -> None:
    """
    Scatter plot of linear predicted position vs predicted loss during ripples.

    Args:
        linpred_dict: Dictionary keyed by epoch with lists of linPred arrays per window
        predloss_dict: Dictionary keyed by epoch with lists of predLoss arrays per window
        ripple_indices_dict: Dictionary keyed by epoch with lists of ripple indices per window
        time_windows: List of time windows
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
    """
    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(10, 10),
        sharex=True,
        sharey=True,
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            mask = ripple_indices_dict[epoch_label][i]
            ax[i, iepoch].scatter(
                linpred_dict[epoch_label][i][mask],
                predloss_dict[epoch_label][i][mask],
                s=1,
            )
            if i == 0:
                ax[i, iepoch].set_title(f"{epoch_label} {time_windows[i]} ms")
            if i == len(time_windows) - 1:
                ax[i, iepoch].set_xlabel("predicted linear position")
            if iepoch == 0:
                ax[i, iepoch].set_ylabel("predicted loss")

    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_figures, "predLoss_position_during_ripples.png"))
    fig.savefig(os.path.join(folder_figures, "predLoss_position_during_ripples.svg"))


def plot_ripple_time_vs_losspredict(
    time_dict: Dict[str, List],
    predloss_dict: Dict[str, List],
    ripple_indices_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
) -> None:
    """
    Scatter plot of time vs predicted loss during ripples.

    Args:
        time_dict: Dictionary keyed by epoch with lists of time arrays per window
        predloss_dict: Dictionary keyed by epoch with lists of predLoss arrays per window
        ripple_indices_dict: Dictionary keyed by epoch with lists of ripple indices per window
        time_windows: List of time windows
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
    """
    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(10, 10),
        sharex=True,
        sharey=True,
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            mask = ripple_indices_dict[epoch_label][i]
            ax[i, iepoch].scatter(
                time_dict[epoch_label][i][mask],
                predloss_dict[epoch_label][i][mask],
                s=1,
            )
            if i == 0:
                ax[i, iepoch].set_title(f"{epoch_label} {time_windows[i]} ms")
            if i == len(time_windows) - 1:
                ax[i, iepoch].set_xlabel("time (s)")
            if iepoch == 0:
                ax[i, iepoch].set_ylabel("predicted loss")

    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_figures, "predLoss_time_during_ripples.png"))
    fig.savefig(os.path.join(folder_figures, "predLoss_time_during_ripples.svg"))


def plot_ripple_time_distance_vs_losspredict(
    time_distance_dict: Dict[str, List],
    predloss_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
    window: float = 0.4,
    filename_prefix: str = "lossRipples",
) -> None:
    """
    Plot predicted loss as function of time-to-ripple with mean/std overlay.

    Args:
        time_distance_dict: Dictionary keyed by epoch with lists of time-to-ripple arrays per window
        predloss_dict: Dictionary keyed by epoch with lists of predLoss arrays per window
        time_windows: List of time windows
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
        window: Time window cutoff in seconds
        filename_prefix: Prefix for saved figure filenames
    """
    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(14, 10),
        sharex=True,
        sharey="row",
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            timeDist = time_distance_dict[epoch_label][i]
            predLoss = predloss_dict[epoch_label][i]

            valid_mask = np.less(timeDist, window)
            hist, xedges, yedges = np.histogram2d(
                timeDist[valid_mask],
                predLoss[valid_mask],
                (1000, 100),
            )

            ax[i, iepoch].imshow(
                hist.T,
                origin="lower",
                aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap=white_viridis,
            )

            row_sum = np.sum(hist, axis=1)

            means = np.divide(
                np.sum(hist * yedges[:-1][None, :], axis=1),
                row_sum,
                out=np.full_like(row_sum, np.nan, dtype=np.float64),
                where=row_sum > 0,
            )
            stds = np.sqrt(
                np.divide(
                    np.sum(
                        hist * np.power(yedges[:-1][None, :] - means[:, None], 2),
                        axis=1,
                    ),
                    row_sum,
                    out=np.full_like(row_sum, np.nan, dtype=np.float64),
                    where=row_sum > 0,
                )
            )

            ax[i, iepoch].plot(
                xedges[:-1],
                means,
                c="red",
                label="mean predicted loss \n given time to ripple",
                alpha=0.3,
            )
            ax[i, iepoch].fill_between(
                xedges[:-1], means - stds, means + stds, color="orange", alpha=0.3
            )

            if i == 0:
                ax[i, iepoch].set_title(f"{epoch_label} {time_windows} ms")
            if i == len(time_windows) - 1:
                ax[i, iepoch].set_xlabel("Time to ripple (s)")
                ax[i, iepoch].tick_params(axis="x")
                if epoch_label == epoch_labels[-1]:
                    ax[i, iepoch].legend(loc=(0.65, 0.13))
            if iepoch == 0:
                ax[i, iepoch].set_ylabel("predicted loss")
                ax[i, iepoch].tick_params(axis="y")

    fig.tight_layout()
    fig.show()
    plt.savefig(os.path.join(folder_figures, f"{filename_prefix}.png"))
    plt.savefig(os.path.join(folder_figures, f"{filename_prefix}.svg"))


def plot_ripple_density_vs_confidence(
    predloss_dict: Dict[str, List],
    ripple_indicators_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
    gaussian_sigma: float = 30,
) -> None:
    """
    Analyze correlation between predicted confidence and ripple density using linear regression.

    Args:
        predloss_dict: Dictionary keyed by epoch with lists of predLoss arrays per window
        ripple_indicators_dict: Dictionary keyed by epoch with lists of binary ripple indicators per window
        time_windows: List of time windows
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
        gaussian_sigma: Sigma for gaussian filtering of ripple density
    """
    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(12, 4 * len(time_windows)),
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            predloss = predloss_dict[epoch_label][i]
            predConfidence = (
                np.mean(predloss, axis=1) if predloss.ndim > 1 else predloss
            )

            isRipple = ripple_indicators_dict[epoch_label][i]
            gaussRippleDensity = gaussian_filter1d(
                isRipple.astype(np.float32), gaussian_sigma
            )

            valid_mask = np.greater_equal(gaussRippleDensity, 0)
            reg = LinearRegression().fit(
                predConfidence[valid_mask][:, None],
                gaussRippleDensity[valid_mask][:, None],
            )
            r2_score = reg.score(
                predConfidence[valid_mask][:, None],
                gaussRippleDensity[valid_mask][:, None],
            )

            ax[i, iepoch].scatter(
                predConfidence[valid_mask],
                gaussRippleDensity[valid_mask],
                c="grey",
                s=1,
                alpha=0.5,
            )
            ax[i, iepoch].hist2d(
                predConfidence[valid_mask],
                gaussRippleDensity[valid_mask],
                (500, 500),
                cmap=white_viridis,
                alpha=0.4,
            )

            x_range = np.arange(
                np.min(predConfidence), np.max(predConfidence), step=0.1
            )
            y_pred = reg.coef_[0, 0] * x_range + reg.intercept_[0]
            ax[i, iepoch].plot(x_range, y_pred, c="black", linewidth=2)

            ax[i, iepoch].set_xlabel("Predicted confidence")
            ax[i, iepoch].set_ylabel("Ripple density (gaussian filtered)")
            ax[i, iepoch].set_title(
                f"{epoch_label} - {time_windows[i]} ms (R²={r2_score:.3f})"
            )

    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_figures, "ripple_density_vs_confidence.png"))
    fig.savefig(os.path.join(folder_figures, "ripple_density_vs_confidence.svg"))


def plot_ripple_density_log_vs_confidence(
    predloss_dict: Dict[str, List],
    ripple_indicators_dict: Dict[str, List],
    time_windows: List,
    epoch_labels: List,
    folder_figures: str,
    gaussian_sigma: float = 30,
) -> None:
    """
    Analyze correlation between predicted confidence and log ripple density.

    Args:
        predloss_dict: Dictionary keyed by epoch with lists of predLoss arrays per window
        ripple_indicators_dict: Dictionary keyed by epoch with lists of binary ripple indicators per window
        time_windows: List of time windows
        epoch_labels: List of epoch/sleep/suffix labels
        folder_figures: Path to save figures
        gaussian_sigma: Sigma for gaussian filtering of ripple density
    """
    fig, ax = plt.subplots(
        len(time_windows),
        len(epoch_labels),
        figsize=(12, 4 * len(time_windows)),
    )
    if len(time_windows) == 1 and len(epoch_labels) == 1:
        ax = np.array([[ax]])
    elif len(time_windows) == 1:
        ax = ax[np.newaxis, :]
    elif len(epoch_labels) == 1:
        ax = ax[:, np.newaxis]

    for iepoch, epoch_label in enumerate(epoch_labels):
        for i in range(len(time_windows)):
            predloss = predloss_dict[epoch_label][i]
            predConfidence = (
                np.mean(predloss, axis=1) if predloss.ndim > 1 else predloss
            )

            isRipple = ripple_indicators_dict[epoch_label][i]
            gaussRippleDensity = gaussian_filter1d(
                isRipple.astype(np.float32), gaussian_sigma
            )

            valid_mask = np.greater(gaussRippleDensity, 0)
            log_ripple_density = np.log(gaussRippleDensity[valid_mask])

            reg = LinearRegression().fit(
                predConfidence[valid_mask][:, None],
                log_ripple_density[:, None],
            )
            r2_score = reg.score(
                predConfidence[valid_mask][:, None],
                log_ripple_density[:, None],
            )

            ax[i, iepoch].scatter(
                predConfidence[valid_mask],
                log_ripple_density,
                c="grey",
                s=1,
                alpha=0.5,
            )
            r = ax[i, iepoch].hist2d(
                predConfidence[valid_mask],
                log_ripple_density,
                (100, 100),
                cmap=white_viridis,
                alpha=0.4,
            )

            x_range = np.arange(
                np.min(predConfidence), np.max(predConfidence), step=0.1
            )
            y_pred = reg.coef_[0, 0] * x_range + reg.intercept_[0]
            ax[i, iepoch].plot(
                x_range, y_pred, c="red", linewidth=2, label="Linear fit"
            )

            mean_log_density = np.array(
                [
                    np.mean(
                        log_ripple_density[
                            (predConfidence[valid_mask] >= r[1][e])
                            & (predConfidence[valid_mask] < r[1][e + 1])
                        ]
                    )
                    for e in range(len(r[1]) - 1)
                ]
            )
            std_log_density = np.array(
                [
                    np.std(
                        log_ripple_density[
                            (predConfidence[valid_mask] >= r[1][e])
                            & (predConfidence[valid_mask] < r[1][e + 1])
                        ]
                    )
                    for e in range(len(r[1]) - 1)
                ]
            )

            valid_mean = np.logical_not(np.isnan(mean_log_density))
            if np.any(valid_mean):
                ax[i, iepoch].plot(
                    r[1][:-1][valid_mean],
                    mean_log_density[valid_mean],
                    c="red",
                    linewidth=2,
                )
                ax[i, iepoch].fill_between(
                    r[1][:-1][valid_mean],
                    mean_log_density[valid_mean] - std_log_density[valid_mean],
                    mean_log_density[valid_mean] + std_log_density[valid_mean],
                    color="orange",
                    alpha=0.3,
                )

            ax[i, iepoch].set_xlabel("Predicted confidence")
            ax[i, iepoch].set_ylabel("Log ripple density")
            ax[i, iepoch].set_title(
                f"{epoch_label} - {time_windows[i]} ms (R²={r2_score:.3f})"
            )
            ax[i, iepoch].legend()

    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_figures, "ripple_density_log_vs_confidence.png"))
    fig.savefig(os.path.join(folder_figures, "ripple_density_log_vs_confidence.svg"))
