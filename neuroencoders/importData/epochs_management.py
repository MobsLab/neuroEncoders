"""
All functions to deal with Epochs and time data. Mainly inspired from Tsd, and should at some point be integrated with pynapple package.
"""

from typing import Dict, Optional

import numpy as np

########### Management of epochs ############


# a few tool to help do difference of intervals as sets:
def obtainCloseComplementary(epochs, bound_interval):
    """
    New version using numpy only.
    Obtain the close complementary of intervals (intervals share their bounds).
    Note: To obtain the open complementary, one would need to add some dt to the end and start of intervals.

    :param epochs: List of epochs as [start1, end1, start2, end2, ...]
    :param bound_interval: The bounding interval as [start, end]

    :return: Complementary intervals as a list of [start, end, ...]
    """
    epochs = np.array(epochs).reshape(-1, 2)
    bound_start, bound_end = bound_interval

    complementary = []
    if epochs[0, 0] > bound_start:
        complementary.append([bound_start, epochs[0, 0]])

    for i in range(len(epochs) - 1):
        complementary.append([epochs[i, 1], epochs[i + 1, 0]])

    if epochs[-1, 1] <= bound_end:
        complementary.append([epochs[-1, 1], bound_end])

    return np.array(complementary).reshape(-1, 2)


def intersect_with_session(epochs, kept_session, starts, stops):
    """
    New version using numpy only.
    We go through the different removed session epoch, and if a train epoch or a
    test epoch intersect with it we remove it from the train and test epochs.

    :param epochs: the epochs to be intersected with the kept session
    :param keptSession: the session kept
    :param starts: the start of the session
    :param stops: the stop of the session

    :return: the epochs that are not intersecting with the removed session
    """
    epochs = np.array(epochs).reshape(-1, 2)
    session_intervals = np.array(
        [[starts[i], stops[i]] for i, keep in enumerate(kept_session) if keep]
    )

    intersected = []
    for epoch in epochs:
        for session in session_intervals:
            start = max(epoch[0], session[0])
            end = min(epoch[1], session[1])
            if start < end:
                intersected.append([start, end])

    return np.array(intersected).reshape(-1, 2)


def interval_union(comp_interval, p1, bound_interval):
    """
    Perform the union of intervals using NumPy.

    :param comp_interval: Existing intervals as a 2D NumPy array.
    :param p1: A 2D NumPy array of intervals.
    :param bound_interval: A 1D NumPy array representing the bounding interval [start, end].
    :return: Updated comp_interval with the new interval added.
    """
    # Create the new interval to add
    comp_interval = np.array(comp_interval).reshape(-1, 2)
    p1 = np.array(p1).reshape(-1, 2)
    new_interval = np.array([[p1[-1, 1], bound_interval[1]]])

    # Combine the existing intervals with the new interval
    all_intervals = np.vstack([comp_interval, new_interval])

    # Merge overlapping or adjacent intervals
    return merge_intervals(all_intervals)


def merge_intervals(intervals):
    """
    Merge overlapping or adjacent intervals.

    :param intervals: A 2D NumPy array where each row is [start, end].
    :return: A 2D NumPy array with merged intervals.
    """
    if len(intervals) == 0:
        return intervals

    # Sort intervals by start time
    intervals = intervals[np.argsort(intervals[:, 0])]

    # Initialize merged intervals
    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]
        # Check if intervals overlap or are adjacent
        if current[0] <= previous[1]:
            # Merge intervals
            merged[-1] = [previous[0], max(previous[1], current[1])]
        else:
            # Add non-overlapping interval
            merged.append(current)

    return np.array(merged).reshape(-1, 2)


# Auxilliary function
def get_epochs(postime, SetData, keptSession, starts=np.empty(0), stops=np.empty(0)):
    # given the slider values, as well as the selected session, we extract the different sets
    # if starts and stops (of epochs) are present, it means we work with multi-recording

    pmin = postime[0]
    pmax = postime[-1]

    testEpochs = np.array(
        [
            postime[SetData["testSetId"]],
            postime[
                min(SetData["testSetId"] + SetData["sizeTestSet"], postime.shape[0] - 1)
            ],
        ]
    )

    if SetData["useLossPredTrainSet"]:
        lossPredSetEpochs = np.array(
            [
                postime[SetData["lossPredSetId"]],
                postime[
                    min(
                        SetData["lossPredSetId"] + SetData["sizeLossPredSet"],
                        postime.shape[0] - 1,
                    )
                ],
            ]
        )
        lossPredsetinterval = obtainCloseComplementary(testEpochs, [pmin, pmax])
        lossPredsetinterval = np.ravel(
            interval_union(lossPredsetinterval, lossPredSetEpochs, [pmin, pmax])
        )

        trainInterval = interval_union(
            obtainCloseComplementary(testEpochs, [pmin, pmax]),
            obtainCloseComplementary(lossPredSetEpochs, [pmin, pmax]),
            [pmin, pmax],
        )
    else:
        trainInterval = obtainCloseComplementary(testEpochs, [pmin, pmax])

    trainEpoch = np.ravel(np.array([[p[0], p[1]] for p in trainInterval]))

    if starts.size > 0:
        trainEpoch = intersect_with_session(trainEpoch, keptSession, starts, stops)
        testEpochs = intersect_with_session(testEpochs, keptSession, starts, stops)
        if SetData["useLossPredTrainSet"]:
            lossPredSetEpochs = intersect_with_session(
                lossPredSetEpochs, keptSession, starts, stops
            )
            return trainEpoch, testEpochs, lossPredSetEpochs
        else:
            return trainEpoch, testEpochs, None
    else:
        if SetData["useLossPredTrainSet"]:
            return trainEpoch, testEpochs, lossPredSetEpochs
        else:
            return trainEpoch, testEpochs, None


def inEpochs(t, epochs):
    """
    For a list of epochs, where each epochs starts is on even index [0,2,...
    and stops on odd index: [1,3,...
    Test if t is among at least one of these epochs
    Epochs are treated as closed interval [,]

    returns the index where it is the case
    """
    epochs = np.array(epochs).reshape(-1)
    mask = np.sum(
        [
            (t >= epochs[2 * i]) * (t <= epochs[2 * i + 1])
            for i in range(len(epochs) // 2)
        ],
        axis=0,
    )
    return np.where(mask >= 1)


def inEpochsMask(t, epochs):
    """
    For a list of epochs, where each epochs starts is on even index [0,2,...
    and stops on odd index: [1,3,...
    Test if t is among at least one of these epochs
    Epochs are treated as closed interval [,]
    returns the mask

    :param t: list of time points
    :param epochs: list of epochs

    :return: mask
    """

    epochs = np.array(epochs).reshape(-1)
    mask = np.sum(
        [
            (t >= epochs[2 * i]) * (t <= epochs[2 * i + 1])
            for i in range(len(epochs) // 2)
        ],
        axis=0,
    )
    return mask >= 1


def get_epochs_mask(
    times: Optional[np.ndarray] = None,
    epochs: Optional[np.ndarray] = None,
    behaviorData: Optional[Dict] = None,
    useTrain: bool = False,
    useTest: bool = True,
    usePredLoss: bool = False,
    sleepEpochs: Optional = None,
):
    """
    Get the epochs mask for training or testing.

    parameters:
    ______________________________________________________
    behaviorData : dict
        dictionary containing the behavioral data. In particular, it needs to contain the following keys:
        - Times : dict with trainEpochs and testEpochs keys
    - positionTime : np.ndarray with time points
    Otherwise, times and epochs can be provided directly.
    times : np.ndarray (default=None)
        time points to consider. If None, will use behaviorData["positionTime"][:, 0]
    epochs : dict (default=None)
        dictionary containing the epochs. If None, will use behaviorData["Times"]

    sleepEpochs : list or np.ndarray (default=None)
        If provided, will return the mask for these epochs only.

    useTrain : bool (default=False)
    useTest : bool (default=True)
    usePredLoss : bool (default=False)

    returns:
    ______________________________________________________
    epochMask : np.ndarray
        boolean mask for the epochs to be used.
    """
    if times is None and behaviorData is None:
        raise ValueError("Either behaviorData or times must be provided.")
    if epochs is None and behaviorData is None:
        raise ValueError("Either behaviorData or epochs must be provided.")

    if times is None:
        times = behaviorData["positionTime"][:, 0]
    if epochs is None:
        epochs = behaviorData["Times"]
    epochMask = np.zeros_like(times, dtype=bool)

    if sleepEpochs is not None and len(sleepEpochs) > 0:
        print("returning sleep epochs for testing")
        return inEpochsMask(times, sleepEpochs)

    if useTrain:
        print("Adding train epochs for testing")
        epochMask += inEpochsMask(times, epochs["trainEpochs"])
    if useTest:
        print("Adding test epochs for testing")
        epochMask += inEpochsMask(times, epochs["testEpochs"])
    if usePredLoss:
        print("Adding predLossSet epochs for testing")
        epochMask += inEpochsMask(
            times,
            epochs["lossPredSetEpochs"],
        )

    return epochMask


def align_timestamps(A, B, tolerance=None):
    """
    Aligns each timestamp in A to the closest timestamp in B.

    Parameters:
        A (array-like): Reference timestamps.
        B (array-like): Timestamps to be aligned to A.
        tolerance (float, optional): Maximum allowed difference for matching (in the same unit as A/B).

    Returns:
        aligned_indices: Indices in B corresponding to closest timestamps to A.
        aligned_B: Aligned timestamps from B (same shape as A).
        diffs: Differences between A and matched B.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # Ensure B is sorted (required for searchsorted)
    B_sorted = np.sort(B)

    # Find index in B where each A[i] would be inserted to keep order
    idx = np.searchsorted(B_sorted, A)

    # Clip indices to stay in bounds
    idx = np.clip(idx, 1, len(B_sorted) - 1)

    # Compare which neighbor (left or right) is closer
    left = B_sorted[idx - 1]
    right = B_sorted[idx]
    idx_closest = idx - (np.abs(A - left) < np.abs(A - right))

    aligned_B = B_sorted[idx_closest]
    diffs = np.abs(A - aligned_B)

    if tolerance is not None:
        idx_closest[diffs > tolerance] = -1  # mark unmatched

    return idx_closest, aligned_B, diffs


def find_closest_index(arr, value, tolerance=None):
    """
    Find the index of the closest value in an array.

    :param arr: 1D numpy array of values.
    :param value: The value to find the closest index for.
    :param tolerance: Optional tolerance value. If provided, checks if the closest value is within this tolerance.
                      If True, uses half the average sampling rate as tolerance.
                      If False or None, no tolerance check is performed.
    :return: Index of the closest value in arr.
    """
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()

    if tolerance is not None:
        # check if tolerance is a number:
        if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
            if np.abs(arr[idx] - value) > tolerance:
                raise ValueError(
                    f"No value in the array is within the tolerance of {tolerance} for the value {value}."
                )
        elif isinstance(tolerance, bool) and tolerance:
            from warnings import warn

            warn(
                "Tolerance is set to True, will compare against apparent sampling rate."
            )
            tolerance = 2 * np.nanmean(
                np.diff(arr)
            )  # should be half the average sampling rate
            if np.abs(arr[idx] - value) > tolerance:
                warn(
                    f"No value in the array is within the tolerance of {tolerance} for the value {value}: found {arr[idx]} which gives a difference of {np.abs(arr[idx] - value)}."
                )
                return -1
        else:
            raise ValueError("Tolerance must be a number or boolean.")
    return idx
