#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected PlaceField analysis matching MATLAB PlaceField function
Based on the original MATLAB PlaceField_DB function

@author: corrected version
"""

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import label

try:
    import pynapple as nap  # More common neuroseries library

    nts = nap
    HAS_NEUROSERIES = True
except ImportError:
    try:
        import neuroseries as nts

        HAS_NEUROSERIES = True
    except ImportError:
        HAS_NEUROSERIES = False
        print("Warning: No neuroseries library found. Using numpy arrays.")


def smooth_dec(data: np.ndarray, smooth: Union[List, Tuple, np.ndarray]) -> np.ndarray:
    """
    Improved smoothing function matching MATLAB SmoothDec behavior

    Args:
        data: Input data (vector or 2D matrix)
        smooth: Smoothing parameters [smooth_x, smooth_y] or single value

    Returns:
        Smoothed data array
    """

    # Input validation
    if data.size == 0:
        return data

    # Handle data dimensions
    vector = data.ndim == 1 or min(data.shape) == 1
    matrix = data.ndim == 2 and not vector

    if not vector and not matrix:
        raise ValueError("Data must be a vector or a 2D matrix")

    # Ensure smooth is array-like with 2 elements
    if np.isscalar(smooth):
        smooth = [smooth, smooth]
    elif len(smooth) == 1:
        smooth = [smooth[0], smooth[0]]

    smooth = np.array(smooth)

    # No smoothing case
    if smooth[0] == 0 and smooth[1] == 0:
        return data

    if vector:
        # Handle vector smoothing
        if data.ndim == 2:
            data = data.flatten()

        v_size = min(len(data), 501)  # Reasonable kernel size limit
        if v_size < 3:
            return data

        # Create Gaussian kernel
        sigma = smooth[0]
        if sigma <= 0:
            return data

        kernel_size = min(int(6 * sigma), v_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)

        # Apply convolution
        smoothed = signal.convolve(data, kernel, mode="same")
        return smoothed

    else:  # Matrix case
        smoothed = data.copy().astype(float)

        # Vertical smoothing (along columns, first dimension)
        if smooth[0] > 0:
            sigma = smooth[0]
            kernel_size = min(int(6 * sigma), smoothed.shape[0])
            if kernel_size >= 3:
                if kernel_size % 2 == 0:
                    kernel_size += 1
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-(x**2) / (2 * sigma**2))
                kernel = kernel / np.sum(kernel)

                # Apply column-wise smoothing
                for i in range(smoothed.shape[1]):
                    smoothed[:, i] = signal.convolve(
                        smoothed[:, i], kernel, mode="same"
                    )

        # Horizontal smoothing (along rows, second dimension)
        if smooth[1] > 0:
            sigma = smooth[1]
            kernel_size = min(int(6 * sigma), smoothed.shape[1])
            if kernel_size >= 3:
                if kernel_size % 2 == 0:
                    kernel_size += 1
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-(x**2) / (2 * sigma**2))
                kernel = kernel / np.sum(kernel)

                # Apply row-wise smoothing
                for i in range(smoothed.shape[0]):
                    smoothed[i, :] = signal.convolve(
                        smoothed[i, :], kernel, mode="same"
                    )

        return smoothed


def find_peak_location(rate_map: np.ndarray) -> Tuple[int, int]:
    """Find coordinates of peak firing rate"""
    peak = np.max(rate_map)
    peak_coords = np.where(rate_map == peak)
    # Return first occurrence if multiple peaks (x, y as col, row)
    return peak_coords[1][0], peak_coords[0][0]


def find_firing_field(
    center_x: int,
    center_y: int,
    n_bins_x: int,
    n_bins_y: int,
    threshold: float,
    rate_map: np.ndarray,
) -> np.ndarray:
    """
    Find connected firing field around center point above threshold

    Args:
        center_x, center_y: Center coordinates
        n_bins_x, n_bins_y: Map dimensions
        threshold: Minimum firing rate threshold
        rate_map: Firing rate map

    Returns:
        Binary mask of firing field
    """
    # Create binary mask above threshold
    above_threshold = rate_map >= threshold

    # Find connected components
    labeled_array, num_features = label(above_threshold)

    # Find which component contains the center point
    if 0 <= center_y < rate_map.shape[0] and 0 <= center_x < rate_map.shape[1]:
        center_label = labeled_array[center_y, center_x]

        if center_label > 0:
            # Return the connected component containing center
            return labeled_array == center_label

    # If center not in any component, return empty field
    return np.zeros_like(rate_map, dtype=bool)


def clean_map(
    rate_map: np.ndarray,
    threshold: float,
    min_field_size: int = 5,
    max_iterations: int = 50,
) -> np.ndarray:
    """
    Remove small isolated firing regions from rate map

    Args:
        rate_map: Input firing rate map
        threshold: Threshold for field detection (as fraction of peak)
        min_field_size: Minimum field size in bins
        max_iterations: Maximum cleaning iterations

    Returns:
        Cleaned rate map
    """
    cleaned_map = rate_map.copy()

    for iteration in range(max_iterations):
        peak = np.max(cleaned_map)
        if peak == 0:
            break

        peak_x, peak_y = find_peak_location(cleaned_map)
        field = find_firing_field(
            peak_x,
            peak_y,
            cleaned_map.shape[1],
            cleaned_map.shape[0],
            threshold * peak,
            cleaned_map,
        )
        field_size = np.sum(field)

        if field_size > 0 and field_size < min_field_size:
            # Remove small field
            cleaned_map[field] = 0
        else:
            # Field is acceptable size, stop cleaning
            break

    return cleaned_map


def poisson_kb(lambda_rate: float, t_max: float) -> np.ndarray:
    """
    Simulate a Poisson process (equivalent to MATLAB poissonKB)

    Args:
        lambda_rate: Arrival rate (Hz)
        t_max: Maximum time (seconds)

    Returns:
        Array of spike times
    """
    if lambda_rate <= 0:
        return np.array([])

    spike_times = []
    current_time = np.random.exponential(1 / lambda_rate)

    while current_time < t_max:
        spike_times.append(current_time)
        current_time += np.random.exponential(1 / lambda_rate)

    return np.array(spike_times)


def get_time_range(tsd_obj):
    """
    Get time range from neuroseries object with fallback methods
    """
    if hasattr(tsd_obj, "time_support"):
        try:
            support = tsd_obj.time_support
            if hasattr(support, "start") and hasattr(support, "end"):
                return support.start[0], support.end[0]
            elif hasattr(support, "values"):
                return support.values[0, 0], support.values[0, 1]
            else:
                return np.min(support), np.max(support)
        except:
            pass

    # Fallback: use time range from timestamps
    if hasattr(tsd_obj, "times"):
        times = tsd_obj.times()
        return np.min(times), np.max(times)
    elif hasattr(tsd_obj, "index"):
        times = tsd_obj.index
        return np.min(times), np.max(times)
    elif hasattr(tsd_obj, "t"):
        times = tsd_obj.t
        return np.min(times), np.max(times)
    else:
        raise ValueError("Cannot determine time range from neuroseries object")


def realign_spikes(pos_tsd, spike_tsd, method="closest"):
    """
    Realign spike times to position times with fallback methods
    """
    if hasattr(pos_tsd, "realign"):
        return pos_tsd.realign(spike_tsd, align=method)
    elif hasattr(spike_tsd, "restrict"):
        # Try restrict method
        try:
            return spike_tsd.restrict(pos_tsd, align=method)
        except:
            pass

    # Manual interpolation fallback
    if hasattr(pos_tsd, "times") and hasattr(pos_tsd, "values"):
        pos_times = pos_tsd.times()
        pos_values = pos_tsd.values
    else:
        pos_times = pos_tsd.index
        pos_values = pos_tsd.values

    if hasattr(spike_tsd, "times"):
        spike_times = spike_tsd.times()
    else:
        spike_times = spike_tsd.index

    # Interpolate position at spike times
    spike_pos = np.interp(spike_times, pos_times, pos_values)

    # Create new TSD-like object or just return values
    if HAS_NEUROSERIES:
        try:
            return nts.Tsd(spike_times, spike_pos)
        except:
            pass

    # Return as simple object with times and values
    class SimpleTsd:
        def __init__(self, times, values):
            self.times_array = times
            self.values = values
            self.size = len(values)

        def times(self):
            return self.times_array

    return SimpleTsd(spike_times, spike_pos)


def PlaceField_DB(
    spike_times,
    pos_x,
    pos_y,
    epoch=None,
    smoothing: Union[int, List] = 3,
    freq_video: float = 30.0,  # Default to 30 Hz like MATLAB version
    threshold: float = 0.7,  # Default to 0.7 like MATLAB version
    size_map: int = 50,
    limit_maze=None,
    large_matrix: bool = True,
    plot_results: bool = True,
    plot_poisson: bool = False,
) -> Union[Dict, Tuple]:
    """
    Comprehensive place field analysis matching MATLAB PlaceField function

    Args:
        spike_times: Spike timestamps (neuroseries Tsd or array-like)
        pos_x, pos_y: Position coordinates (neuroseries Tsd or array-like)
        epoch: Optional epoch restriction
        smoothing: Spatial smoothing factor(s)
        freq_video: Video sampling rate (Hz) - default 30 Hz
        threshold: Threshold for field detection (fraction of peak) - default 0.7
        size_map: Map size in bins
        limit_maze: Optional spatial limits [x_min, x_max, y_min, y_max]
        large_matrix: Add padding to maps
        plot_results: Generate plots
        plot_poisson: Compare with Poisson control

    Returns:
        If plot_poisson=False: Dictionary containing maps, statistics, and coordinates
        If plot_poisson=True: Tuple matching MATLAB signature
    """

    # Ensure freq_video is always defined
    if freq_video is None:
        freq_video = 30.0
        print("Warning: freq_video was None, using default value of 30.0 Hz")

    # Run the core analysis
    results = _run_place_field_analysis(
        spike_times,
        pos_x,
        pos_y,
        epoch,
        smoothing,
        freq_video,
        threshold,
        size_map,
        limit_maze,
        large_matrix,
    )

    if not plot_poisson:
        # Single analysis mode
        if plot_results:
            plot_place_field_results(results, pos_x, pos_y, spike_times, epoch)
        return results

    else:
        # Poisson control mode

        # Generate Poisson spike train
        try:
            t_start, t_end = get_time_range(pos_x)
            total_time = t_end - t_start
        except:
            # Fallback time calculation
            if hasattr(pos_x, "times"):
                times = pos_x.times()
            else:
                times = pos_x.index
            total_time = np.max(times) - np.min(times)
            t_start = np.min(times)

        # Use original firing rate for Poisson process
        original_firing_rate = results["firing_rate"]

        # Generate Poisson spike times
        poisson_times = poisson_kb(original_firing_rate, total_time)
        if len(poisson_times) > 0:
            # Convert to absolute times
            poisson_times_abs = poisson_times + t_start

            # Create neuroseries object if possible
            if HAS_NEUROSERIES:
                try:
                    poisson_spike_times = nts.Tsd(poisson_times_abs, poisson_times_abs)
                except:
                    # Fallback: simple object
                    class SimpleTsd:
                        def __init__(self, times):
                            self.times_array = times
                            self.size = len(times)

                        def times(self):
                            return self.times_array

                        def __len__(self):
                            return self.size

                    poisson_spike_times = SimpleTsd(poisson_times_abs)
            else:
                # Simple object
                class SimpleTsd:
                    def __init__(self, times):
                        self.times_array = times
                        self.size = len(times)

                    def times(self):
                        return self.times_array

                    def __len__(self):
                        return self.size

                poisson_spike_times = SimpleTsd(poisson_times_abs)
        else:
            # Empty spike train
            if HAS_NEUROSERIES:
                try:
                    poisson_spike_times = nts.Tsd(np.array([]), np.array([]))
                except:

                    class SimpleTsd:
                        def __init__(self):
                            self.times_array = np.array([])
                            self.size = 0

                        def times(self):
                            return self.times_array

                        def __len__(self):
                            return self.size

                    poisson_spike_times = SimpleTsd()
            else:

                class SimpleTsd:
                    def __init__(self):
                        self.times_array = np.array([])
                        self.size = 0

                    def times(self):
                        return self.times_array

                    def __len__(self):
                        return self.size

                poisson_spike_times = SimpleTsd()

        # Run analysis on Poisson data
        poisson_results = _run_place_field_analysis(
            poisson_spike_times,
            pos_x,
            pos_y,
            epoch,
            smoothing,
            freq_video,
            threshold,
            size_map,
            limit_maze,
            large_matrix,
        )

        # Plotting for Poisson comparison
        if plot_results:
            plot_poisson_comparison(
                results,
                poisson_results,
                pos_x,
                pos_y,
                spike_times,
                poisson_spike_times,
                epoch,
            )

        # Return tuple matching MATLAB signature
        return (
            # Original results (10 elements)
            results["map"],  # map
            results["map_ns"],  # mapS (non-smoothed)
            results["stats"],  # stats
            results["spike_coords"]["x"],  # px
            results["spike_coords"]["y"],  # py
            results["firing_rate"],  # FR
            len(results["bin_centers"]["x"]),  # sizeFinal
            results.get("pr_field", None),  # PrField
            results.get("center", None),  # C
            results.get("sc_field", None),  # ScField
            # Poisson results (10 elements)
            poisson_results["map"],  # map2
            poisson_results["map_ns"],  # mapS2
            poisson_results["stats"],  # stats2
            poisson_results["spike_coords"]["x"],  # px2
            poisson_results["spike_coords"]["y"],  # py2
            poisson_results["firing_rate"],  # FR2
            len(poisson_results["bin_centers"]["x"]),  # sizeFinal2
            poisson_results.get("pr_field", None),  # PrField2
            poisson_results.get("center", None),  # C2
            poisson_results.get("sc_field", None),  # ScField2
            # Poisson spike times
            poisson_spike_times,  # Ts
        )


def _run_place_field_analysis(
    spike_times,
    pos_x,
    pos_y,
    epoch,
    smoothing: Union[int, List],
    freq_video: float,
    threshold: float,
    size_map: int,
    limit_maze,
    large_matrix: bool,
) -> Dict:
    """
    Core place field analysis function

    Returns:
        Dictionary containing analysis results
    """

    # Initialize outputs
    results = {
        "map": {"rate": None, "time": None, "count": None},
        "map_ns": {"rate": None, "time": None, "count": None},
        "stats": {},
        "spike_coords": {"x": None, "y": None},
        "firing_rate": None,
        "bin_centers": {"x": None, "y": None},
        "epoch_length": None,
        "epoch_restricted": epoch is not None,
    }

    # Restrict to epoch if provided
    if epoch is not None:
        if HAS_NEUROSERIES and hasattr(spike_times, "restrict"):
            spike_times = spike_times.restrict(epoch)
            pos_x = pos_x.restrict(epoch)
            pos_y = pos_y.restrict(epoch)
        else:
            print(
                "Warning: Epoch restriction not supported with current neuroseries setup"
            )

    # Extract position data
    if hasattr(pos_x, "values"):
        x_data = pos_x.values
        y_data = pos_y.values
    else:
        x_data = np.array(pos_x)
        y_data = np.array(pos_y)

    # Remove NaN values
    valid_pos = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_pos]
    y_data = y_data[valid_pos]

    if len(x_data) == 0:
        print("No valid position data found.")
        return results

    # Handle spatial limits
    if limit_maze is not None:
        x_min, x_max, y_min, y_max = limit_maze
    else:
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

    # Create bin edges
    x_edges = np.linspace(x_min, x_max, size_map + 1)
    y_edges = np.linspace(y_min, y_max, size_map + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create 2D histogram of occupancy
    occ_hist, _, _ = np.histogram2d(x_data, y_data, bins=[x_edges, y_edges])

    # Get spike positions using realignment
    spike_pos_x = realign_spikes(pos_x, spike_times)
    spike_pos_y = realign_spikes(pos_y, spike_times)

    # Extract spike position data
    if hasattr(spike_pos_x, "values"):
        spike_x_data = spike_pos_x.values
        spike_y_data = spike_pos_y.values
    else:
        spike_x_data = np.array(spike_pos_x)
        spike_y_data = np.array(spike_pos_y)

    # Remove NaN values from spike positions
    valid_spikes = np.isfinite(spike_x_data) & np.isfinite(spike_y_data)
    spike_x_data = spike_x_data[valid_spikes]
    spike_y_data = spike_y_data[valid_spikes]

    # Create 2D histogram of spike counts
    if len(spike_x_data) > 0:
        spike_hist, _, _ = np.histogram2d(
            spike_x_data, spike_y_data, bins=[x_edges, y_edges]
        )
    else:
        spike_hist = np.zeros_like(occ_hist)

    # Create firing rate map
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_map = freq_video * spike_hist / occ_hist
        rate_map[~np.isfinite(rate_map)] = 0

    # Handle large matrix (add padding)
    if large_matrix:
        pad_size = int(size_map / 8)

        # Pad rate map
        rate_map_padded = np.pad(rate_map, pad_size, mode="constant", constant_values=0)
        occ_hist_padded = np.pad(occ_hist, pad_size, mode="constant", constant_values=0)
        spike_hist_padded = np.pad(
            spike_hist, pad_size, mode="constant", constant_values=0
        )

        # Extend bin centers
        x_step = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1
        y_step = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 1

        x_left = np.linspace(
            x_centers[0] - pad_size * x_step, x_centers[0] - x_step, pad_size
        )
        x_right = np.linspace(
            x_centers[-1] + x_step, x_centers[-1] + pad_size * x_step, pad_size
        )
        x_centers_padded = np.concatenate([x_left, x_centers, x_right])

        y_left = np.linspace(
            y_centers[0] - pad_size * y_step, y_centers[0] - y_step, pad_size
        )
        y_right = np.linspace(
            y_centers[-1] + y_step, y_centers[-1] + pad_size * y_step, pad_size
        )
        y_centers_padded = np.concatenate([y_left, y_centers, y_right])

        # Update arrays
        rate_map = rate_map_padded
        occ_hist = occ_hist_padded
        spike_hist = spike_hist_padded
        x_centers = x_centers_padded
        y_centers = y_centers_padded

    # Clean rate map from outliers (like MATLAB version)
    rate_flat = rate_map[np.isfinite(rate_map)]
    if len(rate_flat) > 5:
        # Remove top 5 values like in MATLAB
        sorted_vals = np.sort(rate_flat)
        if len(sorted_vals) > 5:
            threshold_val = sorted_vals[-6]  # 6th largest value
            rate_map[rate_map > threshold_val] = 0

    # Store non-smoothed maps
    results["map_ns"]["rate"] = rate_map.T  # Transpose for consistency with MATLAB
    results["map_ns"]["time"] = occ_hist.T / freq_video
    results["map_ns"]["count"] = spike_hist.T

    # Apply smoothing
    if np.isscalar(smoothing):
        smooth_params = [smoothing, smoothing]
    else:
        smooth_params = smoothing

    rate_map_smooth = smooth_dec(rate_map, smooth_params)
    occ_hist_smooth = smooth_dec(occ_hist, smooth_params)
    spike_hist_smooth = smooth_dec(spike_hist, smooth_params)

    # Store smoothed maps (transposed for consistency)
    results["map"]["rate"] = rate_map_smooth.T
    results["map"]["time"] = occ_hist_smooth.T / freq_video
    results["map"]["count"] = spike_hist_smooth.T

    # Calculate statistics
    results["stats"] = calculate_place_field_stats(results["map"], threshold)

    # Calculate firing rate and epoch length
    if epoch is not None and HAS_NEUROSERIES:
        try:
            if hasattr(epoch, "tot_length"):
                epoch_length = epoch.tot_length()
            else:
                epoch_length = np.sum(epoch[:, 1] - epoch[:, 0])
        except:
            epoch_length = get_time_range(pos_x)[1] - get_time_range(pos_x)[0]
    else:
        try:
            t_start, t_end = get_time_range(pos_x)
            epoch_length = t_end - t_start
        except:
            epoch_length = 1.0  # Fallback

    results["firing_rate"] = len(spike_times) / epoch_length if epoch_length > 0 else 0
    results["epoch_length"] = epoch_length

    # Store coordinates
    results["spike_coords"]["x"] = spike_x_data
    results["spike_coords"]["y"] = spike_y_data
    results["bin_centers"]["x"] = x_centers
    results["bin_centers"]["y"] = y_centers

    return results


def calculate_place_field_stats(maps: Dict, threshold: float) -> Dict:
    """
    Calculate comprehensive place field statistics matching MATLAB version

    Args:
        maps: Dictionary containing rate, time, and count maps
        threshold: Threshold for field detection (fraction of peak)

    Returns:
        Dictionary of statistics
    """
    stats = {
        "x": [],
        "y": [],
        "field": [],
        "size": [],
        "peak": [],
        "mean": [],
        "field_x": [],
        "field_y": [],
        "spatial_info": None,
        "sparsity": None,
        "specificity": None,
    }

    rate_map = maps["rate"]
    time_map = maps["time"]
    count_map = maps["count"]

    if np.max(rate_map) == 0:
        stats["field"] = np.zeros_like(rate_map, dtype=bool)
        stats["specificity"] = 0
        return stats

    # Clean the map from small firing regions (matching MATLAB CleanMap)
    cleaned_rate = clean_map(rate_map, threshold, min_field_size=5)

    # Find up to 2 firing fields (matching MATLAB logic)
    rate_temp = cleaned_rate.copy()
    peaks = []
    fields = []
    field_sizes = []
    centers = []

    for i in range(2):
        peak_rate = np.max(rate_temp)
        if peak_rate == 0:
            break

        # Find peak location
        peak_x, peak_y = find_peak_location(rate_temp)

        # Find firing field
        field = find_firing_field(
            peak_x,
            peak_y,
            rate_temp.shape[1],
            rate_temp.shape[0],
            threshold * peak_rate,
            rate_temp,
        )
        field_size = np.sum(field)

        if field_size > 0:
            peaks.append(peak_rate)
            fields.append(field)
            field_sizes.append(field_size)
            centers.append((peak_x, peak_y))

            # Remove this field for next iteration
            rate_temp[field] = 0
        else:
            break

    # Choose between fields (matching MATLAB logic)
    if len(fields) == 0:
        stats["field"] = np.zeros_like(rate_map, dtype=bool)
        winner = None
    elif len(fields) == 1:
        winner = 0
    else:
        # Compare field sizes like in MATLAB
        if field_sizes[1] == 0:
            winner = 0
        else:
            size_relative_diff = (field_sizes[1] - field_sizes[0]) / (
                (field_sizes[0] + field_sizes[1]) / 2
            )
            if size_relative_diff > 0.2:
                winner = 1  # Choose larger field
            else:
                winner = 0  # Choose first field

    # Set statistics
    if winner is not None and len(fields) > winner:
        stats["x"] = centers[winner][0]
        stats["y"] = centers[winner][1]
        stats["field"] = fields[winner]
        stats["size"] = field_sizes[winner]
        stats["peak"] = peaks[winner]
        stats["mean"] = (
            np.mean(rate_map[fields[winner]]) if np.any(fields[winner]) else 0
        )

        # Field boundaries
        field_coords = np.where(fields[winner])
        if len(field_coords[1]) > 0:  # x coordinates
            stats["field_x"] = [np.min(field_coords[1]), np.max(field_coords[1])]
        if len(field_coords[0]) > 0:  # y coordinates
            stats["field_y"] = [np.min(field_coords[0]), np.max(field_coords[0])]

    # Calculate spatial information measures (matching MATLAB)
    total_time = np.sum(time_map)
    if total_time > 0:
        occupancy = time_map / (total_time + np.finfo(float).eps)
        total_spikes = np.sum(count_map)
        mean_firing_rate = total_spikes / (total_time + np.finfo(float).eps)

        if mean_firing_rate > 0:
            # Spatial specificity (Skaggs et al., 1993) - matching MATLAB exactly
            log_arg = count_map / mean_firing_rate
            log_arg[log_arg <= 1] = 1
            stats["specificity"] = (
                np.sum(count_map * np.log2(log_arg) * occupancy) / mean_firing_rate
            )

            # Spatial information
            valid_bins = rate_map > 0
            if np.any(valid_bins):
                rate_occupancy = rate_map * occupancy
                info_per_bin = rate_occupancy[valid_bins] * np.log2(
                    rate_map[valid_bins] / mean_firing_rate
                )
                stats["spatial_info"] = np.sum(info_per_bin)
            else:
                stats["spatial_info"] = 0

            # Sparsity
            mean_rate_squared = np.sum(rate_map**2 * occupancy)
            stats["sparsity"] = (
                mean_firing_rate**2 / mean_rate_squared if mean_rate_squared > 0 else 0
            )
        else:
            stats["specificity"] = 0
            stats["spatial_info"] = 0
            stats["sparsity"] = 0
    else:
        stats["specificity"] = 0
        stats["spatial_info"] = 0
        stats["sparsity"] = 0

    return stats


def plot_place_field_results(results: Dict, pos_x, pos_y, spike_times, epoch) -> None:
    """Plot place field analysis results - epoch-restricted data only"""

    # Determine if epoch was used
    epoch_restricted = results.get("epoch_restricted", False)

    # Restrict to epoch if provided
    if epoch is not None:
        if HAS_NEUROSERIES and hasattr(spike_times, "restrict"):
            spike_times = spike_times.restrict(epoch)
            pos_x = pos_x.restrict(epoch)
            pos_y = pos_y.restrict(epoch)
        else:
            print(
                "Warning: Epoch restriction not supported with current neuroseries setup"
            )

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    title = "Place Field Analysis"
    if epoch_restricted:
        title += " (Epoch Only)"
    fig.suptitle(title, fontsize=16)

    # Occupancy map
    im1 = axes[0, 0].imshow(results["map"]["time"], origin="lower", aspect="auto")
    axes[0, 0].set_title("Occupancy Map (s)")
    plt.colorbar(im1, ax=axes[0, 0])

    # Spike count map
    im2 = axes[0, 1].imshow(results["map"]["count"], origin="lower", aspect="auto")
    axes[0, 1].set_title("Spike Count Map")
    plt.colorbar(im2, ax=axes[0, 1])

    # Firing rate map
    im3 = axes[1, 0].imshow(results["map"]["rate"], origin="lower", aspect="auto")
    axes[1, 0].set_title("Firing Rate Map (Hz)")
    plt.colorbar(im3, ax=axes[1, 0])

    # Trajectory with spikes - ONLY epoch-restricted data
    if hasattr(pos_x, "values"):
        x_vals = pos_x.values
        y_vals = pos_y.values
    else:
        x_vals = np.array(pos_x)
        y_vals = np.array(pos_y)

    # Plot trajectory (epoch-restricted only)
    axes[1, 1].plot(x_vals, y_vals, "lightgray", alpha=0.8, linewidth=1.0)

    # Plot spikes (epoch-restricted only)
    if len(results["spike_coords"]["x"]) > 0:
        axes[1, 1].plot(
            results["spike_coords"]["x"],
            results["spike_coords"]["y"],
            "r.",
            markersize=2,
        )

    title_str = f"Trajectory + Spikes\nFR: {results['firing_rate']:.2f} Hz"
    if epoch_restricted:
        title_str += f"\nDuration: {results['epoch_length']:.1f}s"
    axes[1, 1].set_title(title_str)
    axes[1, 1].set_xlabel("X Position")
    axes[1, 1].set_ylabel("Y Position")

    # Position over time - ONLY epoch-restricted data
    if hasattr(pos_x, "times"):
        x_times = pos_x.times()
        y_times = pos_y.times()
    else:
        # Fallback: create time array for epoch data only
        x_times = np.arange(len(x_vals))
        y_times = np.arange(len(y_vals))

    # Plot position over time (epoch-restricted only)
    axes[2, 0].plot(x_times, x_vals, "b-", alpha=0.8, linewidth=1, label="X position")
    axes[2, 1].plot(y_times, y_vals, "g-", alpha=0.8, linewidth=1, label="Y position")

    # Add spike times (epoch-restricted only)
    if hasattr(spike_times, "times"):
        spike_time_vals = spike_times.times()
    elif hasattr(spike_times, "__len__") and len(spike_times) > 0:
        spike_time_vals = np.array(spike_times)
    else:
        spike_time_vals = []

    if len(spike_time_vals) > 0:
        # Determine y-positions for spike markers
        y_max_x = np.max(x_vals) if len(x_vals) > 0 else 1
        y_max_y = np.max(y_vals) if len(y_vals) > 0 else 1

        spike_y_vals_x = np.full(len(spike_time_vals), y_max_x * 1.1)
        spike_y_vals_y = np.full(len(spike_time_vals), y_max_y * 1.1)

        axes[2, 0].plot(
            spike_time_vals,
            spike_y_vals_x,
            "r.",
            markersize=2,
            alpha=0.8,
            label="Spikes",
        )
        axes[2, 1].plot(
            spike_time_vals,
            spike_y_vals_y,
            "r.",
            markersize=2,
            alpha=0.8,
            label="Spikes",
        )

    axes[2, 0].set_title("X Position Over Time (Epoch Only)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].legend(fontsize=10)
    axes[2, 1].set_title("Y Position Over Time (Epoch Only)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n=== Place Field Statistics ===")
    if epoch_restricted:
        print(f"Analysis on EPOCH ONLY (duration: {results['epoch_length']:.1f}s)")
    else:
        print(f"Analysis on FULL SESSION (duration: {results['epoch_length']:.1f}s)")
    print(f"Firing Rate: {results['firing_rate']:.2f} Hz")
    print(f"Spatial Information: {results['stats']['spatial_info']:.3f} bits/spike")
    print(f"Sparsity: {results['stats']['sparsity']:.3f}")
    print(f"Specificity: {results['stats']['specificity']:.3f}")
    if results["stats"]["peak"]:
        peak_val = results["stats"]["peak"]
        print(f"Peak Firing Rate: {peak_val:.2f} Hz")


def plot_poisson_comparison(
    results: Dict,
    poisson_results: Dict,
    pos_x,
    pos_y,
    spike_times,
    poisson_spike_times,
    epoch,
) -> None:
    """Plot comparison between original and Poisson control analysis"""

    # Check if epoch restricted
    epoch_restricted = results.get("epoch_restricted", False)

    if epoch_restricted and epoch is not None:
        if HAS_NEUROSERIES and hasattr(spike_times, "restrict"):
            spike_times = spike_times.restrict(epoch)
            pos_x = pos_x.restrict(epoch)
            pos_y = pos_y.restrict(epoch)
        else:
            print(
                "Warning: Epoch restriction not supported with current neuroseries setup"
            )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = "Place Field vs Poisson Control Comparison"
    if epoch_restricted:
        title += " (Epoch Restricted)"
    fig.suptitle(title, fontsize=16)

    # Original firing rate map
    im1 = axes[0, 0].imshow(results["map"]["rate"], origin="lower", aspect="auto")
    axes[0, 0].set_title(
        f"Original Firing Map\nSpatial Info: {results['stats']['spatial_info']:.3f}"
    )
    plt.colorbar(im1, ax=axes[0, 0])

    # Original trajectory with spikes - only show epoch data
    if hasattr(pos_x, "values"):
        x_vals = pos_x.values
        y_vals = pos_y.values
    else:
        x_vals = np.array(pos_x)
        y_vals = np.array(pos_y)

    axes[0, 1].plot(x_vals, y_vals, "lightgray", alpha=0.7, linewidth=0.8)

    if len(results["spike_coords"]["x"]) > 0:
        axes[0, 1].plot(
            results["spike_coords"]["x"],
            results["spike_coords"]["y"],
            "r.",
            markersize=2,
        )

    title_str = f"Original Trajectory + Spikes\nFR: {results['firing_rate']:.2f} Hz"
    if epoch_restricted:
        title_str += f"\n(Duration: {results['epoch_length']:.1f}s)"
    axes[0, 1].set_title(title_str)
    axes[0, 1].set_xlabel("X Position")
    axes[0, 1].set_ylabel("Y Position")

    # Poisson firing rate map
    im2 = axes[1, 0].imshow(
        poisson_results["map"]["rate"], origin="lower", aspect="auto"
    )
    axes[1, 0].set_title(
        f"Poisson Control Map\nSpatial Info: {poisson_results['stats']['spatial_info']:.3f}"
    )
    plt.colorbar(im2, ax=axes[1, 0])

    # Poisson trajectory with spikes - same trajectory as original
    axes[1, 1].plot(x_vals, y_vals, "lightgray", alpha=0.7, linewidth=0.8)

    if len(poisson_results["spike_coords"]["x"]) > 0:
        axes[1, 1].plot(
            poisson_results["spike_coords"]["x"],
            poisson_results["spike_coords"]["y"],
            "r.",
            markersize=2,
        )

    title_str = (
        f"Poisson Trajectory + Spikes\nFR: {poisson_results['firing_rate']:.2f} Hz"
    )
    if epoch_restricted:
        title_str += f"\n(Duration: {poisson_results['epoch_length']:.1f}s)"
    axes[1, 1].set_title(title_str)
    axes[1, 1].set_xlabel("X Position")
    axes[1, 1].set_ylabel("Y Position")

    plt.tight_layout()
    plt.show()

    # Print comparison statistics
    print("\n=== Place Field vs Poisson Control Comparison ===")
    if epoch_restricted:
        print(
            f"Analysis performed on epoch-restricted data (duration: {results['epoch_length']:.1f}s)"
        )
    print(
        f"Original - Spatial Info: {results['stats']['spatial_info']:.3f}, Specificity: {results['stats']['specificity']:.3f}"
    )
    print(
        f"Poisson  - Spatial Info: {poisson_results['stats']['spatial_info']:.3f}, Specificity: {poisson_results['stats']['specificity']:.3f}"
    )
    print(
        f"Difference - Spatial Info: {results['stats']['spatial_info'] - poisson_results['stats']['spatial_info']:.3f}"
    )


def analyze_multiple_epochs(
    spike_times,
    pos_x,
    pos_y,
    epochs_dict,
    epoch_names=None,
    plot_results=True,
    **kwargs,
):
    """
    Analyze place fields across multiple epochs

    Args:
        spike_times, pos_x, pos_y: Neural and position data
        epochs_dict: Dictionary containing epoch definitions (e.g., Epochs['Session'])
        epoch_names: List of epoch names to analyze (if None, analyzes all)
        plot_results: Whether to plot results for each epoch
        **kwargs: Additional arguments passed to PlaceField_DB

    Returns:
        Dictionary with results for each epoch
    """
    if epoch_names is None:
        epoch_names = list(epochs_dict.keys())

    results = {}

    for epoch_name in epoch_names:
        if epoch_name not in epochs_dict:
            print(f"Warning: Epoch '{epoch_name}' not found, skipping...")
            continue

        print(f"\n=== Analyzing epoch: {epoch_name} ===")

        try:
            result = PlaceField_DB(
                spike_times,
                pos_x,
                pos_y,
                epoch=epochs_dict[epoch_name],
                plot_results=plot_results,
                **kwargs,
            )
            results[epoch_name] = result

            # Print summary stats
            if isinstance(result, dict):
                stats = result.get("stats", {})
                print(f"  Spatial Info: {stats.get('spatial_info', 0):.3f}")
                print(f"  Firing Rate: {result.get('firing_rate', 0):.2f} Hz")
                print(f"  Peak Rate: {stats.get('peak', 0):.2f} Hz")

        except Exception as e:
            print(f"Error analyzing epoch '{epoch_name}': {e}")
            results[epoch_name] = None

    return results


# Example usage
if __name__ == "__main__":
    print("Corrected PlaceField analysis functions loaded successfully!")
    print("Use PlaceField_DB() with your spike times and position data.")
    print(
        "This version handles missing time_support methods and ensures freq_video is always defined."
    )
    print("\nExample usage:")
    print("# Basic analysis")
    print("results = PlaceField_DB(spike_times, pos_x, pos_y)")
    print("\n# With epoch restriction")
    print(
        "results = PlaceField_DB(spike_times, pos_x, pos_y, epoch=Epochs['Session']['TestPre'])"
    )
    print("\n# With grouped epochs (after grouping script)")
    print(
        "results = PlaceField_DB(spike_times, pos_x, pos_y, epoch=Epochs['Session']['Cond'])"
    )
    print("\n# Analyze multiple epochs")
    print(
        "results_all = analyze_multiple_epochs(spike_times, pos_x, pos_y, Epochs['Session'], ['TestPre', 'Cond', 'TestPost'])"
    )
    print("\n# With Poisson control and epoch")
    print(
        "results_tuple = PlaceField_DB(spike_times, pos_x, pos_y, epoch=Epochs['Session']['Hab'], plot_poisson=True)"
    )
    print("\nPlotting will show:")
    print("- Only the epoch-restricted data (if epoch provided)")
    print("- Analysis period trajectory and spikes")
    print("- No background/context from full session")
    print("- Clean focused view on your specific epoch")
