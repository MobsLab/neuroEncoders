"""
This module contains the Project, DataHelper and Params classes, which are used to manage the data and behavior of the experiment.
The Project class is used to store the paths of the project, while the DataHelper class is used to detect and describe the main properties of the signal and behavior. The Params class is used to store the parameters of the experiment, such as the number of channels, the sampling rate, and the list of channels.
It also provides methods to compute the true target of interest, the distance to the wall, and the reference and xy coordinates.
"""

# Load libs
import json

# Load custom code
import os
import os.path
from datetime import date
from typing import Dict, Tuple
from warnings import warn

import dill as pickle

# import matplotlib as mplt
# mplt.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
import tensorflow as tf
from shapely import MultiPoint, Polygon

from neuroencoders.importData import epochs_management as ep
from neuroencoders.importData.rawdata_parser import get_behavior, get_params

MAZE_COORDS = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0.65, 0],
        [0.65, 0.75],
        [0.35, 0.75],
        [0.35, 0],
        [0, 0],
    ]
)
ZONEDEF = np.array(
    [
        [[0, 0.35], [0, 0.43]],  # shock
        [[0.65, 1], [0, 0.43]],  # safe
        [[0.35, 0.65], [0.75, 1]],  # center
        [[0, 0.35], [0.43, 1]],  # shock center
        [[0.65, 1], [0.43, 1]],  # safe center
    ]
)
ZONELABELS = ["Shock", "Safe", "Center", "ShockCenter", "SafeCenter"]

ZONE_COLORS = ["r", "b", "k", "m", "c"]


class Project:
    """
    Class to store the paths of the project.
    xmlPath: path to the xml find
    datPath: path to the dat file
    jsonPath: path to the json file with the Julia thresholds and the graph
    nameExp: name of the experiment, defaults to "Network"
    windowSize: size of the window in seconds, defaults to 0.036
    """

    def __init__(self, xmlPath, *args, **kwargs):
        # datPath="", jsonPath=None, nameExp="Network", windowSize=0.036
        datPath = kwargs.get("datPath", "")
        jsonPath = kwargs.get("jsonPath", None)
        nameExp = kwargs.get("nameExp", "Network")

        # check whether we received windowSize or windowSizeMS
        if "windowSizeMS" in kwargs:
            warn(
                "Using windowSizeMS argument is deprecated, please use windowSize instead."
            )
            windowSize = kwargs["windowSizeMS"] / 1000.0
        else:
            # if not, we use the windowSize argument
            windowSize = kwargs.get("windowSize", 0.036)

        # Basic names
        if xmlPath[-3:] != "xml":
            if os.path.isfile(xmlPath[:-3] + "xml"):
                xmlPath = xmlPath[:-3] + "xml"
            else:
                raise ValueError(f"the path {xmlPath} doesn't match a .xml file")
        self.xml = xmlPath
        self.baseName = xmlPath[:-4]
        if datPath == "":
            self.dat = self.baseName + ".dat"
        else:
            self.dat = datPath
        self.fil = self.dat[:-4] + ".fil"
        # Folders
        findFolder = (
            lambda path: path
            if path[-1] == "/" or len(path) == 0
            else findFolder(path[:-1])
        )
        self.folder = findFolder(self.dat)
        self.dataPath = os.path.join(self.folder, "dataset")
        # Allows change at every experiment
        self.experimentPath = os.path.join(
            self.folder, nameExp
        )  # replaces self.resultsPath
        self.folderResult = os.path.join(self.experimentPath, "results")
        self.folderResultSleep = os.path.join(self.experimentPath, "results_Sleep")
        self.windowSize = windowSize
        self.windowSizeMS = int(windowSize * 1000)  # in ms
        # Create dirs if don't exist
        if not os.path.isdir(self.dataPath):
            os.makedirs(self.dataPath)
        if not os.path.isdir(self.experimentPath):
            os.makedirs(self.experimentPath)
        if not os.path.isdir(self.folderResult):
            os.makedirs(self.folderResult)
        # Json
        if jsonPath is None:
            self.json = self.baseName + ".json"
            self.graph = os.path.join(self.folder, nameExp, "models")
        else:
            print("using file:", jsonPath)
            self.json = jsonPath
            self.thresholds, self.graph = self.getThresholdsAndGraph()
            self.graphChkP = self.graph + "/cp.ckpt"

    # Functions
    def clu(self, g):
        return self.baseName + ".clu." + str(g + 1)

    def res(self, g):
        return self.baseName + ".res." + str(g + 1)

    def spk(self, g):
        return self.baseName + ".spk." + str(g + 1)

    def pos(self, g):
        return self.folder + "dataset/pos." + str(g + 1) + ".npz"

    def getThresholdsAndGraph(self):
        import json

        with open(self.json, "r") as f:
            info = json.loads(f.read())
        return [
            [
                abs(info[d][f])
                for f in ["threshold" + str(c) for c in range(info[d]["nChannels"])]
            ]
            for d in ["group" + str(g) for g in range(info["nGroups"])]
        ], info["encodingPrefix"]

    def __str__(self):
        return (
            f"{'M' + os.path.basename(os.path.dirname(self.baseName)):=^50}\n"
            f"windowSize={self.windowSize}\n"
            f"xmlPath={self.xml}\n"
            f"datPath={os.path.basename(self.dat)}\n"
            f"jsonPath={os.path.basename(self.json)}\n"
            f"nameExp={os.path.basename(self.experimentPath)}\n"
            f"folderResults={os.path.basename(self.folderResult)}\n"
            f"\n{'=' * 50}"
        )

    def __repr__(self):
        return (
            f"{'M' + os.path.basename(os.path.dirname(self.baseName)):=^50}\n"
            f"windowSize={self.windowSize}\n"
            f"Project(xmlPath={self.xml}, datPath={os.path.basename(self.dat)}, jsonPath={os.path.basename(self.json)}, "
            f"nameExp={os.path.basename(self.experimentPath)}, folderResult={os.path.basename(self.folderResult)})\n"
            f"\n{'=' * 50}"
        )

    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (IsADirectoryError, FileNotFoundError, EOFError):
            with open(os.path.join(path, "Project_108.pkl"), "rb") as f:
                return pickle.load(f)


class DataHelper(Project):
    """
        A class to detect and describe the main properties on the signal and behavior
        args:
        - xmlPath: the xmlPath that will inherit an Object containing the path to the data (xml & dat)
        - mode: the argument of the neuroEncoder command
        - target: the target of the experiment, can be "pos", "lin", "linear", "LinAndThigmo", "linAndThigmo", "direction", or "Direction"
    - nameExp: the name of the experiment, defaults to "Network"

    """

    def __init__(
        self,
        xmlPath=None,
        mode=None,
        target=None,
        *args,
        **kwargs,
    ):
        """
        Initializes the DataHelper object.

        Args:
        --------

        - xmlPath: should be a xmlPath to instantiate a Project object.
        - mode: the mode of the experiment, can be "ann", "bayes", "compare", or "decode".
        - target: the target of the experiment, can be "pos", "lin", "linear", "LinAndThigmo", "linAndThigmo", "direction", or "Direction".
        - *args: additional positional arguments to pass to the Project constructor.
        - **kwargs: additional keyword arguments to pass to the Project constructor.
            - phase: the phase of the experiment, can be "pre", "preNoHab", "hab", "cond", "post", "extinction"... or None.
            - force_ref: whether to force the computation of the reference and xy coordinates, even if they are already saved.
        """
        # Handle positional arguments
        if xmlPath is None and len(args) >= 3:
            xmlPath, mode, target = args[0], args[1], args[2]
            args = args[3:]
        elif xmlPath is not None and mode is None and len(args) >= 2:
            mode, target = args[0], args[1]
            args = args[2:]
        elif (
            xmlPath is not None
            and mode is not None
            and target is None
            and len(args) >= 1
        ):
            target = args[0]
            args = args[1:]

        self.mode = mode
        self.target = target
        self.phase = kwargs.get("phase", None)  # remove from the kwargs
        self.force_ref = kwargs.get("force_ref", False)
        self.isPredLoss = kwargs.get("isPredLoss", False)

        # LEGACY: old called directly called Project object in the init
        if isinstance(xmlPath, Project):
            xmlPath = xmlPath.xml
            warn(
                "You are using a legacy version of DataHelper, please use the new version with xmlPath as a string."
            )

        super().__init__(xmlPath, *args, **kwargs)

        self.resultsPath = os.path.join(
            self.experimentPath,
            "results",
            str(int(self.windowSizeMS)),
        )
        self.suffix = f"_{self.phase}" if self.phase is not None else ""

        self.list_channels, self.samplingRate, self.nChannels = get_params(self.xml)
        if self.mode == "decode":
            self.fullBehavior = get_behavior(
                self.folder, getfilterSpeed=False, decode=True, phase=self.phase
            )
        else:
            self.fullBehavior = get_behavior(self.folder, phase=self.phase)
        self.positions = self.fullBehavior["Positions"]
        # we compute the true target on-the-fly
        self.positionTime = self.fullBehavior["positionTime"]

        self.stopTime = self.positionTime[
            -1
        ]  # max(self.epochs["train"] + self.epochs["test"])
        self.startTime = self.positionTime[0]

        if (
            not self.isPredLoss
            and len(self.fullBehavior["Times"]["lossPredSetEpochs"]) > 0
        ):
            self.fullBehavior["Times"]["old_trainEpochs"] = self.fullBehavior["Times"][
                "trainEpochs"
            ].copy()
            self.fullBehavior["Times"]["trainEpochs"] = np.concatenate(
                (
                    self.fullBehavior["Times"]["trainEpochs"],
                    self.fullBehavior["Times"]["lossPredSetEpochs"],
                )
            )
            print(
                "extending trainEpochs with lossPredSetEpochs as you're not using predLoss"
            )

        self.lower_x = 0.35
        self.upper_x = 0.65
        self.ylim = 0.75
        self._define_maze_zones()
        self._get_ref_and_xy(phase=self.phase, force=self.force_ref)

    def nGroups(self):
        """
        Returns the number of **spike** groups (visually, neurons are spiking) of channels by looking
        at the list of channels in the .xml file.
        """
        return len(self.list_channels)

    def numChannelsPerGroup(self):
        """
        Returns the number of channels per "spiking" group by looking at the list of channels in the .xml file.
        """
        return [len(self.list_channels[n]) for n in range(self.nGroups())]

    def maxPos(self):
        """
        Returns the maximum position (in the position dimensions) of the animal by looking at the positions array.
        """

        maxPos = np.max(
            self.positions[np.logical_not(np.isnan(np.sum(self.positions, axis=1)))],
            axis=0,
        )
        return maxPos if self.mode != "decode" else 1

    def dim_output(self):
        """
        Returns the number of output features by looking at the number of columns in the positions array.
        """
        return self.positions.shape[1]

    def getThresholds(self):
        idx = 0
        nestedThresholds = []
        for group in range(len(self.list_channels)):
            temp = []
            for channel in range(len(self.list_channels[group])):
                temp.append(self.thresholds[idx])
                idx += 1
            nestedThresholds.append(temp)
        return nestedThresholds

    def setThresholds(self, thresholds):
        assert [len(d) for d in thresholds] == [len(s) for s in self.list_channels]
        self.thresholds = [i for d in thresholds for i in d]

    def get_true_target(self, l_function=None, in_place=False, show=False, **kwargs):
        """
        Returns the true target of interest by looking and modifying the positions array.

        Args:
        - l_function: a function that takes a position and returns the linearized position
        - in_place: whether to modify the positions array in place
        - show: whether to show the distance to the wall
        - **kwargs: additional keyword arguments to pass to the l_function
            - speedMask : whether to use a speed mask to filter the positions based on speed (for AnimatedPositionPlotter only), defaults to False

        Returns:
        - the modified positions array
        - if in_place is True,the modified positions array will be found in the positions attribute of the object
        """

        if hasattr(self, "old_positions"):
            warn(
                "old_positions already exist,meaning you already ran the true target ! beware."
            )
            self.positions = self.old_positions

        if not hasattr(self, "l_function") and l_function is None:
            self.l_function = l_function

        self.get_maze_limits(show=False)

        if self.target == "pos":
            positions = self.positions

        elif self.target == "lin" or self.target == "linear":
            assert l_function is not None, (
                "l_function must be provided for linearization"
            )
            assert self.positions.shape[1] == 2, "positions must have 2 dimensions"
            _, positions = l_function(self.positions)
            positions = positions.reshape(-1)
            self.linearized = positions

        elif self.target.lower() == "linandthigmo":
            assert l_function is not None, (
                "l_function must be provided for linearization"
            )
            assert self.positions.shape[1] == 2, "positions must have 2 dimensions"
            _, positions = l_function(self.positions)
            thigmo = self.dist2wall(self.positions, show=show)
            positions = np.concatenate(
                (positions.reshape(-1, 1), thigmo.reshape(-1, 1)), axis=1
            )
        elif self.target.lower() == "linanddirection":
            assert l_function is not None, (
                "l_function must be provided for linearization"
            )
            assert self.positions.shape[1] == 2, "positions must have 2 dimensions"
            _, positions = l_function(self.positions)
            positions = positions.reshape(-1)
            self.direction = self._get_traveling_direction(positions)
            positions = np.concatenate(
                (positions.reshape(-1, 1), self.direction.reshape(-1, 1)), axis=1
            )

        elif self.target.lower() == "direction":
            _, positions = l_function(self.positions)
            positions = positions.reshape(-1)
            self.direction = self._get_traveling_direction(positions)
            positions = self.direction.reshape(-1, 1)
        elif self.target.lower() == "linandheaddirection":
            _, positions = l_function(self.positions)
            positions = positions.reshape(-1)
            self.head_direction = self._get_head_direction(self.positions)
            positions = np.concatenate(
                (positions.reshape(-1, 1), self.head_direction.reshape(-1, 1)),
                axis=1,
            )
        elif self.target.lower() == "linandspeed":
            _, positions = l_function(self.positions)
            positions = positions.reshape(-1)
            self.speed = self._get_speed(
                self.positions, interval=1 / (15 // self.windowSizeMS)
            )
            positions = np.concatenate(
                (positions.reshape(-1, 1), self.speed.reshape(-1, 1)), axis=1
            )
        elif self.target.lower() == "posanddirection":
            positions = self.positions
            self.direction = self._get_traveling_direction(self.positions)
            positions = np.concatenate(
                (positions.reshape(-1, 2), self.direction.reshape(-1, 1)), axis=1
            )
        elif self.target.lower() == "posandheaddirection":
            positions = self.positions
            self.head_direction = self._get_head_direction(self.positions)
            positions = np.concatenate(
                (positions.reshape(-1, 2), self.head_direction.reshape(-1, 1)), axis=1
            )
        elif self.target.lower() == "posanddirectionandthigmo":
            positions = self.positions
            self.direction = self._get_traveling_direction(self.positions)
            thigmo = self.dist2wall(self.positions, show=show)
            positions = np.concatenate(
                (
                    positions.reshape(-1, 2),
                    self.direction.reshape(-1, 1),
                    thigmo.reshape(-1, 1),
                ),
                axis=1,
            )
        elif self.target.lower() == "posandheaddirectionandthigmo":
            positions = self.positions
            self.head_direction = self._get_head_direction(self.positions)
            thigmo = self.dist2wall(self.positions, show=show)
            positions = np.concatenate(
                (
                    positions.reshape(-1, 2),
                    self.head_direction.reshape(-1, 1),
                    thigmo.reshape(-1, 1),
                ),
                axis=1,
            )
        elif self.target.lower() == "posandspeed":
            positions = self.positions
            self.speed = self._get_speed(self.positions, interval=1 / (15))
            positions = np.concatenate(
                (positions.reshape(-1, 2), self.speed.reshape(-1, 1)), axis=1
            )
        elif self.target.lower() == "posandheaddirectionandspeed":
            positions = self.positions
            self.head_direction = self._get_head_direction(self.positions)
            self.speed = self._get_speed(self.positions, interval=1 / (15))
            positions = np.concatenate(
                (
                    positions.reshape(-1, 2),
                    self.head_direction.reshape(-1, 1),
                    self.speed.reshape(-1, 1),
                ),
                axis=1,
            )
        else:
            raise ValueError(
                f"target {self.target} not recognized. Please use 'pos', 'lin', 'linear', 'LinAndThigmo' or 'linAndThigmo'"
            )

        if show:
            from neuroencoders.importData.gui_elements import AnimatedPositionPlotter

            plotter = AnimatedPositionPlotter(
                data_helper=self,
                trail_length=40,
                l_function=l_function,
                linear_position_mode=True,
                **kwargs,
            )
            anim = plotter.show(interval=1, repeat=True, block=True)

        if in_place:
            if not hasattr(self, "old_positions"):
                self.old_positions = self.positions
            self.positions = positions
            self.fullBehavior["old_positions"] = self.old_positions
            self.fullBehavior["Positions"] = positions

        return positions

    def dist2wall(self, positions, show=False):
        """
        Calculate the distance to the wall for each position.

        Parameters
        ----------
        positions : np.ndarray
            Array of positions with shape (n_samples, 2).
        show : bool, optional

        Returns
        -------
        dist_to_wall : np.ndarray
            Array of distances to the wall with shape (n_samples,).
        """
        assert positions.shape[1] == 2, "positions must have 2 dimensions"

        self.get_maze_limits(show=show)
        self.create_polygon()
        boundary = self.polygon.boundary
        self.shapePoints = MultiPoint(positions)
        dist_to_wall = np.array(
            [point.distance(boundary) for point in list(self.shapePoints.geoms)]
        )
        self.thigmo = dist_to_wall
        return dist_to_wall

    def get_maze_limits(self, show=False):
        """
        Returns the limits of the "allowed" maze.
        We assume the maze is a unit-square with a hole in the middle.

        """
        # First we make sure the maze is unit square
        # assert np.allclose(np.nanmin(self.positions), 0, atol=0.05)
        # assert np.allclose(np.nanmax(self.positions), 1, atol=0.05)
        try:
            positions = self.old_positions
        except AttributeError:
            positions = self.positions

        assert positions.shape[1] == 2, "positions must have 2 dimensions"
        lower_mask = np.where((positions[:, 1] < 0.75) & (positions[:, 0] < 0.5))
        upper_mask = np.where((positions[:, 1] < 0.75) & (positions[:, 0] > 0.5))
        self.lower_x = positions[lower_mask, 0].max()
        self.upper_x = positions[upper_mask, 0].min()
        self.xlims = [self.lower_x, self.upper_x]

        y_mask = np.where(
            (positions[:, 0] > self.lower_x) & (positions[:, 0] < self.upper_x)
        )
        self.ylim = positions[y_mask, 1].min()
        self._define_maze_zones()
        if show:
            plt.plot(positions[:, 0], positions[:, 1], "--.")
            plt.axvline(self.lower_x, color="r")
            plt.axvline(self.upper_x, color="r")
            plt.axhline(self.ylim, color="black")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.title(f"Maze limits for {self.folder}")
            plt.show()
        return (
            self.lower_x,
            self.upper_x,
        ), (self.ylim)

    def _define_maze_zones(self):
        """
        Defines the maze zones.
        """
        self.xlims = [self.lower_x, self.upper_x]
        self.ylim = self.ylim
        self.zone_height = 0.43

        self.maze_coords = [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [self.upper_x, 0],
            [self.upper_x, self.ylim],
            [self.lower_x, self.ylim],
            [self.lower_x, 0],
            [0, 0],
        ]
        self.shock_zone = [
            [0, 0],
            [0, 0.43],
            [self.lower_x, self.zone_height],
            [self.lower_x, 0],
            [0, 0],
        ]

        self.safe_zone = [
            [self.upper_x, 0],
            [self.upper_x, self.zone_height],
            [1, self.zone_height],
            [1, 0],
            [0, 0],
        ]

        self.ZoneDef = [
            [[0, self.lower_x], [0, self.zone_height]],  # shock
            [[self.upper_x, 1], [0, self.zone_height]],  # safe
            [[self.lower_x, self.upper_x], [self.ylim, 1]],  # center
            [[0, self.lower_x], [self.zone_height, 1]],  # shock center
            [[self.upper_x, 1], [self.zone_height, 1]],  # safe center
        ]
        self.ZoneLabels = ["Shock", "Safe", "Center", "ShockCenter", "SafeCenter"]

        self.zone_colors = ["r", "b", "k", "m", "c"]
        self.shock_zone = np.array(self.shock_zone)
        self.safe_zone = np.array(self.safe_zone)
        self.maze_coords = np.array(self.maze_coords)
        self.create_polygon()

    def create_polygon(self):
        """
        Creates the polygon of the maze.
        """
        self.polygon = Polygon(self.maze_coords)

    def _get_ref_and_xy(
        self, phase=None, save=True, plot=False, force: bool = False, positions=None
    ):
        """
        Returns the reference and xy coordinates for the given SessionName.

        args:
        -----------
        - phase: the phase of the experiment. if it does not exist we will find for the closest match/first test.
        - save: whether to save the reference and xy coordinates in the nnBehavior.mat  and the fullBehavior.
        - plot: whether to plot the coordinates and the reference.
        - force: whether to force the computation of the reference and xy coordinates, even if they are already saved.
        - positions: the positions to use for the computation of the reference and xy coordinates. If None, it will use the positions from the fullBehavior.

        returns:
        -----------

        """
        if phase is None:
            phase = self.phase
        if "aligned_ref" in self.fullBehavior and not force:
            self.ref = self.fullBehavior["ref"]
            self.aligned_ref = self.fullBehavior["aligned_ref"]
            self.xyOutput = self.fullBehavior["xyOutput"]
            self.ratioIMAonREAL = self.fullBehavior["ratioIMAonREAL"]
            self.shock_zone_mask = self.fullBehavior["shock_zone_mask"]
            if plot:
                self._plot_coordinates_and_ref(normalized_positions=positions)
        elif (
            "ref" in self.fullBehavior and "aligned_ref" not in self.fullBehavior
        ) and not force:  # should not happen
            print("ref found but not aligned_ref")
            self.ref = self.fullBehavior["ref"]
            self.xyOutput = self.fullBehavior["xyOutput"]
            self.shock_zone_mask = self.fullBehavior["shock_zone_mask"]
            self.ratioIMAonREAL = self.fullBehavior["ratioIMAonREAL"]
            self.normalized_positions, self.aligned_ref = (
                self._transform_coordinates_and_image(positions=None)
            )
            if plot:
                self._plot_coordinates_and_ref(normalized_positions=positions)
        else:
            self._get_ref_and_xy_from_behavResources(phase=phase, save=save, plot=plot)
        # try to recover for plotting videos later
        if "M" in self.fullBehavior:
            self.TransformMatrix = self.fullBehavior["M"]
        if "outputSize" in self.fullBehavior:
            self.output_size = self.fullBehavior["outputSize"]

    def _get_ref_and_xy_from_behavResources(self, phase="Cond1", save=True, plot=False):
        """
        Returns the reference and xy coordinates for the given SessionName,
        starting from the behavResources.

        args:
        -----------
        - phase: the phase of the experiment  - if it does not exist we will find for the closest match/first test.
        - save: whether to save the reference and xy coordinates in the
        fullBehavior dictionary and the f"nnBehavior_{phase}"

        returns:
        -----------

        """
        filename = os.path.join(self.folder, "behavResources.mat")
        from scipy.io import loadmat

        f = loadmat(filename)
        if (
            "behavResources" in f
        ):  # handle single session case - eg small files or non concatenatedlab
            session_names = [
                sess[0] for sess in list(f["behavResources"]["SessionName"][0])
            ]
            if phase in session_names:
                idx = session_names.index(phase)
            else:
                if phase == "all":
                    pattern = "Cond"
                elif phase == "pre":
                    # Look for Pre sessions or Hab sessions
                    pattern = "pre"
                elif phase == "preNoHab":
                    pattern = "pre"
                elif phase == "hab":
                    pattern = "hab"
                elif phase == "cond":
                    pattern = "cond"
                elif phase == "post":
                    pattern = "post"
                elif phase == "postNoExtinction":
                    pattern = "post"
                else:
                    raise ValueError("phase must be one of pre, cond, post or all")
                # find the session names that contain the phase
                idx = [
                    i
                    for i, name in enumerate(session_names)
                    if pattern.lower() in name.lower() and "sleep" not in name.lower()
                ]
                print(
                    f"Using the first session that contains {phase}: {session_names[idx[0]]}"
                )
                idx = idx[0]
                root = f["behavResources"]
        else:
            idx = 0
            root = f

        try:
            self.ref = root["ref"][0][idx]
            idx_shock = list(root["ZoneLabels"][0][idx][0]).index("Shock")
            self.shock_zone_mask = root["Zone"][0][idx][0][idx_shock]
            Xdata = root["Xtsd"][0][idx][0][0][-2].flatten()
            Ydata = root["Ytsd"][0][idx][0][0][-2].flatten()
        except ValueError:
            idx_shock = list(root["ZoneLabels"][idx]).index("Shock")
            self.shock_zone_mask = root["Zone"][idx][idx_shock]
            self.ref = root["ref"]
            Xdata = root["Xtsd"][0][idx][-2].flatten()
            Ydata = root["Ytsd"][0][idx][-2].flatten()

        assert self.ref.shape == self.shock_zone_mask.shape
        self.ratioIMAonREAL = root["Ratio_IMAonREAL"][0][idx].flatten()[
            0
        ]  # only a digit

        positions = np.array([Xdata, Ydata]).T

        self.xyOutput = self._get_XYOutput_morph_maze(
            positions, self.shock_zone_mask, self.ref, self.ratioIMAonREAL
        )

        self.normalized_positions, self.aligned_ref = (
            self._transform_coordinates_and_image(positions)
        )

        if plot:
            self._plot_coordinates_and_ref()

        if save:
            self.fullBehavior["ref"] = self.ref
            self.fullBehavior["xyOutput"] = self.xyOutput
            self.fullBehavior["ratioIMAonREAL"] = self.ratioIMAonREAL
            self.fullBehavior["shock_zone_mask"] = self.shock_zone_mask
            self.fullBehavior["aligned_ref"] = self.aligned_ref
            self.fullBehavior["M"] = self.TransformMatrix
            self.fullBehavior["outputSize"] = self.output_size

            # save theses infos in the nnBehavior file.
            file = os.path.join(self.folder, f"nnBehavior{self.suffix}.mat")
            with tables.open_file(file, "a") as nnBehavior:
                children = [c.name for c in nnBehavior.list_nodes("/behavior")]
                if "ref" in children:
                    nnBehavior.remove_node("/behavior", "ref")
                nnBehavior.create_array("/behavior", "ref", self.ref)
                if "xyOutput" in children:
                    nnBehavior.remove_node("/behavior", "xyOutput")
                nnBehavior.create_array("/behavior", "xyOutput", self.xyOutput)
                if "shock_zone" in children:
                    nnBehavior.remove_node("/behavior", "shock_zone")
                if "shock_zone_mask" in children:
                    nnBehavior.remove_node("/behavior", "shock_zone_mask")
                nnBehavior.create_array(
                    "/behavior", "shock_zone_mask", self.shock_zone_mask
                )
                if "aligned_ref" in children:
                    nnBehavior.remove_node("/behavior", "aligned_ref")
                nnBehavior.create_array("/behavior", "aligned_ref", self.aligned_ref)
                if "ratioIMAonREAL" in children:
                    nnBehavior.remove_node("/behavior", "ratioIMAonREAL")
                nnBehavior.create_array(
                    "/behavior", "ratioIMAonREAL", self.ratioIMAonREAL
                )
                if "M" in children:
                    nnBehavior.remove_node("/behavior", "M")
                nnBehavior.create_array("/behavior", "M", self.TransformMatrix)
                if "outputSize" in children:
                    nnBehavior.remove_node("/behavior", "outputSize")
                nnBehavior.create_array("/behavior", "outputSize", self.output_size)
                nnBehavior.flush()
                nnBehavior.close()

        return self.ref, self.xyOutput

    def _get_XYOutput_morph_maze(
        self, positions, shock_zone_mask, ref, Ratio_IMAonREAL
    ):
        """
        Morphs UMaze coordinates into a system with (0,0) as the bottom corner of the shock zone
        and the rest of the maze going from 0 to 1.

        Parameters:
            positions [Xtsd, Ytsd]: numpy array of positions
            shock_zone_mask: binary mask of the shock zone
            Ref: reference image of the maze
            Ratio_IMAonREAL: scaling factor between image and real-world coordinates

        Returns:
            XYOutput: user-defined or pre-defined coordinates
        """
        from cmcrameri import cm
        from matplotlib.widgets import Button, Slider
        from skimage.measure import regionprops

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        positions = positions
        img = ax.imshow(ref, cmap=cm.batlow)
        # , extent = [positions[:,1].min(), positions[:,1].max(), positions[:,0].min(), positions[:,0].max()])
        scatter = ax.plot(
            positions[:, 1] * Ratio_IMAonREAL,
            positions[:, 0] * Ratio_IMAonREAL,
            color=[0.8, 0.8, 0.8],
        )[0]
        props = regionprops(shock_zone_mask.astype(int))
        centroid = props[0].centroid
        ax.plot(centroid[1], centroid[0], "r.", markersize=30)
        ax.plot(centroid[1], centroid[0], "w*", markersize=10)

        ax.set_title("Shock ext. corner - Safe ext. corner - Shock side far wall")

        # Add slider for colormap intensity
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(
            ax_slider, "Intensity", np.min(ref), np.max(ref), valinit=np.max(ref)
        )

        # Add reset button
        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(
            ax_reset, "Reset", color="lightgoldenrodyellow", hovercolor="0.975"
        )

        # Coordinates storage
        coords = []
        markers = []

        def update(val):
            img.set_clim(0, slider.val)
            fig.canvas.draw_idle()

        def reset(event):
            # Clear all markers
            for marker in markers:
                marker.remove()
            markers.clear()

            # Clear coordinates
            coords.clear()

            # Redraw canvas
            fig.canvas.draw_idle()

        slider.on_changed(update)
        reset_button.on_clicked(reset)

        # Restrict ginput to the plot area and add red stars
        def on_click(event):
            if event.inaxes == ax:  # Only register clicks inside the plot area
                coords.append((event.xdata, event.ydata))
                marker = ax.plot(event.xdata, event.ydata, "b+", markersize=10)[0]
                markers.append(marker)
                ax.plot(event.xdata, event.ydata, "b+", markersize=10)  # Add red star
                fig.canvas.draw_idle()
                if len(coords) == 3:  # Stop after 3 points
                    fig.canvas.mpl_disconnect(cid)
                    plt.close(fig)

        cid = fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show(block=True)

        if len(coords) == 3:
            x, y = zip(*coords)
            XYOutput = np.array([y, x])
            plt.close(fig)
            return XYOutput

    def _transform_coordinates_and_image(
        self,
        positions,
        reference_image=None,
        reference_points=None,
        ratio_ima_on_real=None,
    ):
        """
        Transform both positions and reference image to normalized [0,1]² space

        Args:
            positions: Array of shape (n, 2) containing (x, y) positions to transform
            reference_image: Numpy array of shape (320, 240) with values 0-1 representing the grayscale image
            reference_points: Array of shape (3, 2) containing reference points in camera space
                            in order: [0,0], [1,0], [0,1]
            ratio_ima_on_real: Scaling factor (optional)

        Returns:
            transformed_positions: Array of shape (n, 2) containing normalized positions
            transformed_image: The reference image transformed to match the normalized space
        """
        import cv2

        if reference_image is None:
            reference_image = self.ref.T
        if reference_points is None:
            reference_points = self.xyOutput.T
        if ratio_ima_on_real is None:
            ratio_ima_on_real = self.ratioIMAonREAL

        # Make sure reference image is in the right format
        if reference_image.max() <= 1.0:
            # Scale to 0-255 for better OpenCV handling
            reference_image_for_cv = (reference_image * 255).astype(np.uint8)
        else:
            reference_image_for_cv = reference_image.astype(np.uint8)

        self.reference_image_for_cv = reference_image_for_cv

        # Get source image dimensions
        h, w = reference_image.shape[:2]

        # Define source points (from camera tracking)
        src_pts = np.float32(
            [
                reference_points[0],  # [0,0] point
                reference_points[1],  # [1,0] point
                reference_points[2],  # [0,1] point
                [
                    reference_points[1][0]
                    + reference_points[2][0]
                    - reference_points[0][0],
                    reference_points[1][1]
                    + reference_points[2][1]
                    - reference_points[0][1],
                ],  # [1,1] point (computed)
            ]
        )

        # Create destination image size that preserves aspect ratio
        # Calculate width and height of the source quadrilateral
        width1 = np.linalg.norm(src_pts[1] - src_pts[0])
        width2 = np.linalg.norm(src_pts[3] - src_pts[2])
        height1 = np.linalg.norm(src_pts[2] - src_pts[0])
        height2 = np.linalg.norm(src_pts[3] - src_pts[1])

        # Average width and height
        avg_width = int((width1 + width2) / 2)
        avg_height = int((height1 + height2) / 2)

        # Define output size with a margin
        output_size = (avg_width, avg_height)
        self.output_size = output_size

        # Define destination points scaled to output size
        dst_pts = np.float32(
            [
                [0, 0],  # [0,0]
                [output_size[0] - 1, 0],  # [1,0]
                [0, output_size[1] - 1],  # [0,1]
                [output_size[0] - 1, output_size[1] - 1],  # [1,1]
            ]
        )

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.TransformMatrix = M

        # Apply perspective transformation to the image with appropriate flags
        transformed_image_cv = cv2.warpPerspective(
            reference_image_for_cv,
            M,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Convert back to 0-1 range if original was in that range
        if reference_image.max() <= 1.0:
            transformed_image = transformed_image_cv.astype(np.float32) / 255.0
        else:
            transformed_image = transformed_image_cv

        # Create transformation matrix for normalizing to [0,1]² space
        # For positions, we need to map to normalized space
        # Using flipped coordinates to match the image orientation
        norm_dst_pts = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])

        norm_M = cv2.getPerspectiveTransform(src_pts, norm_dst_pts)

        # Transform the positions
        # Scale the input positions if needed
        if positions is not None:
            scaled_positions = positions * ratio_ima_on_real

            # Add homogeneous coordinate
            homogeneous_positions = np.hstack(
                (scaled_positions, np.ones((len(scaled_positions), 1)))
            )

            # Apply transformation
            transformed = np.dot(norm_M, homogeneous_positions.T).T

            # Convert back from homogeneous coordinates
            normalized_positions = transformed[:, :2] / transformed[:, 2:3]
            valid_idx = ~np.isnan(normalized_positions[:, 0]) & ~np.isnan(
                normalized_positions[:, 1]
            )
            normalized_positions = normalized_positions[valid_idx]

        self.aligned_ref = transformed_image
        self.normalized_positions = (
            normalized_positions if normalized_positions is not None else None
        )

        return normalized_positions, transformed_image

    def _plot_coordinates_and_ref(
        self, normalized_positions=None, reference_image=None
    ):
        """
        Plots the transformed coordinates and reference image.
        Args:
            normalized_positions: Array of shape (n, 2) containing normalized positions
            reference_image: Numpy array of shape (320, 240) with values 0-1 representing the grayscale image

        """

        if normalized_positions is None:
            try:
                normalized_positions = self.normalized_positions
            except AttributeError:
                mask = ep.inEpochsMask(
                    self.positionTime, self.fullBehavior["Times"]["trainEpochs"]
                )[:, 0]
                normalized_positions = self.positions[mask]
        if reference_image is None:
            try:
                reference_image = self.aligned_ref
            except AttributeError:
                raise ValueError(
                    "No reference image found. Please provide a reference image."
                )

        AlignedXtsd = (
            normalized_positions[:, 0] if normalized_positions is not None else None
        )
        AlignedYtsd = (
            normalized_positions[:, 1] if normalized_positions is not None else None
        )

        # Define zones
        self.ZoneEpochAligned = []

        fig, ax = plt.subplots()

        ax.imshow(
            reference_image,
            cmap="gray",
            extent=[0, 1, 0, 1],
            vmin=reference_image.min(),
            vmax=reference_image.max(),
            origin="lower",
        )

        if normalized_positions is not None:
            for z, zone in enumerate(self.ZoneDef):
                x_mask = (AlignedXtsd >= zone[0][0]) & (AlignedXtsd <= zone[0][1])
                y_mask = (AlignedYtsd >= zone[1][0]) & (AlignedYtsd <= zone[1][1])
                zone_mask = x_mask & y_mask
                self.ZoneEpochAligned.append(zone_mask)
                ax.plot(
                    AlignedXtsd[zone_mask],
                    AlignedYtsd[zone_mask],
                    "--.",
                    color=self.zone_colors[z],
                )

        ax.axhline(0, color="k")
        ax.axhline(1, color="k")
        ax.axvline(0, color="k")
        ax.axvline(1, color="k")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(["SHK", "SF", "CNT", "SHKCNT", "SFCNT"])
        ax.axis("off")  # Hide axes
        ax.set_title(
            f"Transformed coordinates and reference image - {self.phase} session"
        )
        plt.tight_layout()
        plt.show()

        return fig

    def _get_traveling_direction(
        self,
        linearized_positions: np.ndarray,
        window_size: int = 4,
        method: str = "difference",
    ) -> np.ndarray:
        """
        Compute direction using rolling window approach based on the linearized position.

        Args:
            linearized_positions: 1D array of linearized positions
            window_size: Size of rolling window (default: 4)
            method: 'difference' or 'regression' (default: 'difference')

        Returns:
            Binary direction array (1 = forward, 0 = backward) of shape (n_points,).
        """
        if linearized_positions.ndim != 1:
            raise ValueError(
                f"Input must be a 1D array. Received shape: {linearized_positions.shape}"
            )

        n_points = len(linearized_positions)
        direction = np.zeros(n_points)

        if method == "difference":
            # Method 1: Compare start vs end of rolling window
            for i in range(n_points):
                # Define window bounds
                start_idx = max(0, i - window_size // 2)
                end_idx = min(n_points - 1, i + window_size // 2)

                if end_idx > start_idx:
                    # Compare average of first half vs second half of window
                    mid_point = (start_idx + end_idx) // 2
                    first_half = np.mean(
                        linearized_positions[start_idx : mid_point + 1]
                    )
                    second_half = np.mean(
                        linearized_positions[mid_point + 1 : end_idx + 1]
                    )
                    direction[i] = 1 if second_half > first_half else 0
                else:
                    # Fallback for edge cases
                    direction[i] = direction[i - 1] if i > 0 else 0

        elif method == "regression":
            # Method 2: Linear regression slope within window
            for i in range(n_points):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(n_points, i + window_size // 2 + 1)

                if end_idx - start_idx >= 3:  # Need at least 3 points for regression
                    x = np.arange(start_idx, end_idx)
                    y = linearized_positions[start_idx:end_idx]

                    # Simple linear regression
                    slope = np.polyfit(x, y, 1)[0]
                    direction[i] = 1 if slope > 0 else 0
                else:
                    direction[i] = direction[i - 1] if i > 0 else 0

        return direction.astype(int)

    def _get_head_direction(self, positions, smoothing=True, return_as_deg=False):
        """
        Calculates heading direction based on X and Y coordinates. With one measurement we can only calculate heading direction

        Parameters
        ----------
        positions : (N, 2) array_like
            N samples of observations, containing X and Y coordinates
        smoothing : bool or int, optional
            If speeds should be smoothed, by default False/0
        return_as_deg : bool
            Return heading in radians or degree

        Returns
        -------
        heading_direction : (N, 1) array_like
            Heading direction of the animal
        """
        assert positions.shape[1] == 2, "positions must have 2 dimensions"
        X, Y = positions[:, 0], positions[:, 1]
        # Smooth diffs instead of speeds directly
        Xdiff = np.diff(X)
        Ydiff = np.diff(Y)
        if smoothing:
            Xdiff = smooth_signal(Xdiff, smoothing)
            Ydiff = smooth_signal(Ydiff, smoothing)
        # Calculate heading direction
        heading_direction = np.arctan2(Ydiff, Xdiff)
        heading_direction = np.append(heading_direction, heading_direction[-1])
        if return_as_deg:
            heading_direction = heading_direction * (180 / np.pi)

        return heading_direction

    def _get_speed(self, positions, interval, smoothing=True):
        """
        Calculate speed from X,Y coordinates

        Parameters
        ----------
        positions : (N, 2) array_like
            N samples of observations, containing X and Y coordinates
        interval : int
            Duration between observations (in s, equal to 1 / sr)
        smoothing : bool or int, optional
            If speeds should be smoothed, by default False/0

        Returns
        -------
        speed : (N, 1) array_like
            Instantenous speed of the animal
        """
        assert positions.shape[1] == 2, "positions must have 2 dimensions"
        X, Y = positions[:, 0], positions[:, 1]
        # Smooth diffs instead of speeds directly
        Xdiff = np.diff(X)
        Ydiff = np.diff(Y)
        if smoothing:
            Xdiff = smooth_signal(Xdiff, smoothing)
            Ydiff = smooth_signal(Ydiff, smoothing)
        speed = np.sqrt(Xdiff**2 + Ydiff**2) / interval
        speed = np.append(speed, speed[-1])

        return speed

    def get_training_imbalance(self, by_arm=True, positions=None):
        """
        Returns a simple ratio of left arm/shock zone vs right arm/safe zone training samples

        parameters
        ----------
        by_arm : bool, optional
            If True, returns the ratio of left arm to right arm training samples, else returns the ratio by zones. by default True

        positions : np.ndarray, optional

        returns
        -------
        ratio : float
            Ratio of left arm to right arm training samples or shock zone to safe zone training samples
        """
        if positions is None:
            training_mask = ep.inEpochsMask(
                self.fullBehavior["positionTime"],
                self.fullBehavior["Times"]["trainEpochs"],
            ).flatten()

            in_left_mask = (
                self.fullBehavior["Positions"][training_mask, 0] <= self.lower_x
            )
            in_right_mask = (
                self.fullBehavior["Positions"][training_mask, 0] >= self.upper_x
            )
            if not by_arm:
                in_left_mask = in_left_mask & (
                    self.fullBehavior["Positions"][training_mask, 1] <= self.zone_height
                )
                in_right_mask = in_right_mask & (
                    self.fullBehavior["Positions"][training_mask, 1] <= self.zone_height
                )
        else:
            in_left_mask = positions[:, 0] <= self.lower_x
            in_right_mask = positions[:, 0] >= self.upper_x
            if not by_arm:
                in_left_mask = in_left_mask & (positions[:, 1] <= self.zone_height)
                in_right_mask = in_right_mask & (positions[:, 1] <= self.zone_height)

        return np.sum(in_left_mask) / np.sum(in_right_mask)


class Params:
    """
    Class to store the parameters of the project.

        helper: instance of the DataHelper class
        windowSize: size of the window in seconds (see neuroencoder for arg default)
        nEpochs: number of epochs to train the network
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Params object with required and optional parameters.

        Args:
            *args: positional arguments, expected to contain:
                - helper (DataHelper instance)
                - windowSize (float, in seconds)
            **kwargs: optional keyword arguments, can include:
                - nEpochs (int, default 100)
                - phase (str, optional)
                - batchSize (int, default 256)
                - save_json (bool, default False)
                - isTransformer (bool, optional, default True)
                - transform_w_log (bool, optional, default False)
        """
        # Extract required parameters
        if len(args) >= 1:
            helper = args[0]
            args = args[1:]
        else:
            helper = kwargs.pop("helper", None)

        if len(args) >= 1:
            windowSize = args[0]
            args = args[1:]
        else:
            windowSize = kwargs.pop("windowSize", None)

        if helper is None:
            raise ValueError("helper (DataHelper instance) is required")
        if windowSize is None:
            raise ValueError("windowSize is required")

        # Extract optional parameters
        nEpochs = kwargs.pop("nEpochs", 100)
        phase = kwargs.pop("phase", None)
        batchSize = kwargs.pop("batchSize", 256)
        save_json = kwargs.pop("save_json", False)

        # Initialize attributes
        self.date = date.today().strftime("%Y-%m-%d")

        # Store parameters
        self.nEpochs = nEpochs
        self.phase = phase
        self.batchSize = batchSize
        if not hasattr(self, "windowSize"):
            self.windowSize = windowSize  # in seconds
            self.windowSizeMS = int(windowSize * 1000)  # in milliseconds

        # add the helper object
        self.helper = helper
        # Initialize all other parameters...
        self._initialize_params_attributes(helper, **kwargs)
        # save those to json file
        if save_json:
            self.save_params_to_json()
        if not os.path.isfile(os.path.join(self.resultsPath, "Parameters.pkl")):
            # save the parameters to a pickle file
            save_project_to_pickle(
                self, output=os.path.join(self.resultsPath, "Parameters.pkl")
            )
        self._copy_from_helper(helper)
        # re initialize for method vs attribute
        self._initialize_params_attributes(helper, **kwargs)

    def _initialize_params_attributes(self, helper, **kwargs):
        """
        Initialize all parameters from the helper object and set default values.
        This is where you want to modify or add any additional parameters.
        """
        self.nGroups = helper.nGroups()  # number of anatomical spiking groups
        self.dimOutput = helper.dim_output()  # dimension of what needs to be predicted
        self.originalDimOutput = (
            helper.old_positions.shape[1]
            if hasattr(helper, "old_positions")
            else helper.dim_output()
        )  # original dimension
        self.nChannelsPerGroup = (
            helper.numChannelsPerGroup()
        )  # number of channels per "spiking" anatomical group
        self.length = 0
        self.target = (
            helper.target
        )  # target to predict, e.g. "pos", "lin", "direction", etc.
        self.resultsPath = helper.resultsPath  # path to save results

        # regarding data augmentation
        self.dataAugmentation = kwargs.pop("dataAugmentation", True)

        # TODO: check if this is still relevant
        # WARNING: maybe striding is actually 0.036 ms based ???
        self.nSteps = int(10000 * 0.036 / self.windowSize)  # used in the encoder

        ### from units encoder params
        # MOVED TO DECODE CONFIG
        self.validCluWindow = 0.0005
        self.kernel = "epanechnikov"  # is not connected
        self.bandwidth = 0.1
        self.masking = 20

        ### full encoder params
        self.nFeatures = kwargs.pop("nFeatures", 64)
        self.isTransformer = (
            True
            if kwargs.get("isTransformer", None) is None
            else kwargs.get("isTransformer", None)
        )  # use transformer instead of LSTM
        self.nHeads = kwargs.pop(
            "nHeads", 8
        )  # number of attention heads in the transformer if used
        self.project_transformer = kwargs.pop("project_transformer", False)

        default_lstm_layers = 2 if not self.isTransformer else 4
        self.lstmLayers = kwargs.pop("lstmLayers", default_lstm_layers)
        self.dropoutCNN = kwargs.pop("dropoutCNN", 0.2)
        self.lstmSize = kwargs.pop("lstmSize", 64)
        default_dropout_lstm = 0.3 if not self.isTransformer else 0.5
        self.dropoutLSTM = kwargs.pop("dropoutLSTM", default_dropout_lstm)
        self.ff_dim1 = kwargs.pop(
            "ff_dim1",
            self.nFeatures * 2
            if not self.project_transformer
            else self.nFeatures * self.nGroups * 2,
        )  # first fully connected layer in Transformer arch dimension
        self.ff_dim2 = (
            self.nFeatures
            if not self.project_transformer
            else self.nFeatures * self.nGroups
        )  # second fully connected layer in Transformer arch dimension

        # if using transformer, we need to set 2 other dense layers output (after the multihead attention blocks)
        self.TransformerDenseSize1 = kwargs.pop(
            "TransformerDense1",
            self.nFeatures * 8
            if not self.project_transformer
            else self.nFeatures * self.nGroups * 4,
        )
        self.TransformerDenseSize2 = kwargs.pop(
            "TransformerDense2",
            self.nFeatures * 4
            if not self.project_transformer
            else self.nFeatures * self.nGroups * 2,
        )

        self.nDenseLayers = kwargs.pop(
            "nDenseLayers", 2
        )  # number of dense layers in the network

        # TODO: check if this is still relevant
        # we might want to introduce some Adam or stuff like that - update : RMSProp quite good
        self.learningRates = kwargs.pop(
            "learningRates", [0.0004]
        )  #  [0.00003, 0.00003, 0.00001]

        self.optimizer = kwargs.pop("optimizer", "adam")  # TODO: not implemented yet

        self.GaussianHeatmap = kwargs.pop("GaussianHeatmap", True)
        self.GaussianGridSize = kwargs.pop("GaussianGridSize", (45, 45))
        self.GaussianSigma = kwargs.pop(
            "GaussianSigma", 0.025
        )  # 1/44 ~= 0.023, so it should cover ~3 bins
        self.GaussianEps = kwargs.pop("GaussianEps", 1e-6)
        self.GaussianNeg = -50  # value for forbidden zones in the heatmap

        self.OversamplingResampling = kwargs.pop("OversamplingResampling", True)

        self.lossActivation = None  # activation function for the loss layer
        self.featureActivation = kwargs.pop("featureActivation", None)

        # TODO: put it in a function
        self.loss = kwargs.pop(
            "loss", "huber"
        )  # "mse" or "huber" or "msle" or "logcosh" or "mse_plus_msle"
        if self.target.lower() == "direction":
            self.loss = "binary_crossentropy"

        self.column_losses = {
            "0": "cyclic_mae"
            if "head" in self.target.lower()
            else "binary_crossentropy",
            "1": self.loss,
        }
        self.heatmap_weight = kwargs.pop("heatmap_weight", 1.0)
        self.others_weight = kwargs.pop("other_weight", 0.5)

        self.column_weights = (
            {"0": 0.6, "1": 0.4} if "direction" in self.target.lower() else {}
        )
        self.merge_columns = []
        self.merge_losses = []
        self.merge_weights = []

        if self.target.lower() == "posandheaddirectionandspeed":
            self.column_losses = {
                "0": "mse",
                "1": "mse",
                "2": "cyclic_mae",
                "3": "mae",
            }
            self.column_weights = {
                "0": 0.6,
                "1": 0.6,
                "2": 0.3,
                "3": 0.3,
            }
        self.denseweight = kwargs.pop(
            "denseweight", True
        )  # dense weight loss for dataset imbalance
        self.denseweightAlpha = 0.8

        self.mutual_info_method = kwargs.pop(
            "mutual_info_method", "shannon"
        )  # "skaggs" or "I_spike" or shannon or I_sec

        # self.transform = "log"  # "log" or "sqrt" or None
        self.transform_w_log = kwargs.pop(
            "transform_w_log", False
        )  # "log" or "sqrt" or None
        self.delta = (
            0.01  # for the huber loss - roughly the random prediction threshold
        )
        self.alpha = 5  # for combined loss mse + msle

        self.reduce_lr_on_plateau = kwargs.pop("reduce_lr_on_plateau", True)

        self.usingMixedPrecision = False
        # enforcing float16 computations whenever possible
        # According to tf tutorials, we can allow that in most layer
        # except the output for unclear reasons linked to gradient computations

    def save_params_to_json(self):
        """
        Save the experimentation parameters to a json file in the results folder.
        Is called in the main func.
        Should not be mistaken with the generate_json function for the ann, which is
        called to save the julia thresholds in the datPath.
        """
        if not os.path.isdir(self.resultsPath):
            os.makedirs(self.resultsPath)
        dict_params = vars(self).copy()
        # datahelper is not serializable, so we remove it
        dict_params.pop("helper")
        # remove the helper methods from the dict
        dict_params = {
            k: v
            for k, v in dict_params.items()
            if not callable(v) and not k.startswith("_")
        }
        with open(os.path.join(self.resultsPath, "params.json"), "w") as f:
            json.dump(dict_params, f, cls=NumpyEncoder)

    def _copy_from_helper(self, helper):
        """Copy all attributes from a DataHelper instance"""
        # Copy all attributes from the helper
        for attr_name in dir(helper):
            if attr_name.startswith("_") or attr_name in ["load", "save"]:
                continue
            try:
                setattr(self, attr_name, getattr(helper, attr_name))
            except AttributeError:
                pass  # Skip methods that can't be copied

    def __str__(self):
        """
        String representation of the Params object.
        """
        return (
            f"Params(\n"
            f"  nEpochs={self.nEpochs},\n"
            f"  phase={self.phase},\n"
            f"  batchSize={self.batchSize},\n"
            f"  windowSize={self.windowSize},\n"
            f"  nGroups={self.nGroups},\n"
            f"  dimOutput={self.dimOutput},\n"
            f"  originalDimOutput={self.originalDimOutput},\n"
            f"  nChannelsPerGroup={self.nChannelsPerGroup},\n"
            f"  target='{self.target}',\n"
            f"  resultsPath='{self.resultsPath}'\n"
            f")"
        )

    def __repr__(self):
        """
        Representation of the Params object.
        """
        return self.__str__()

    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (IsADirectoryError, FileNotFoundError, EOFError):
            with open(os.path.join(path, "Parameters.pkl"), "rb") as f:
                return pickle.load(f)


def save_project_to_pickle(project, output=None):
    if output is None:
        output = os.path.join(
            project.experimentPath, f"Project_{int(project.windowSize * 1000)}.pkl"
        )
    with open(os.path.join(os.path.expanduser(output)), "wb") as f:
        pickle.dump(
            project,
            f,
            pickle.HIGHEST_PROTOCOL,
        )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


class SpatialConstraintsMixin:
    """
    Mixin class to provide unified spatial constraints for both Bayesian and ANN decoders
    """

    def __init__(self, grid_size=(45, 45), maze_params=None, **kwargs):
        self.grid_size = grid_size
        self.GRID_H, self.GRID_W = grid_size

        # Create coordinate grids (both numpy and tensorflow versions)
        self._setup_coordinate_grids()

        # Setup spatial constraints
        self.maze_params_dict = self._extract_maze_boundaries(maze_params)
        self.forbid_mask_np, self.forbid_mask_tf = self._create_spatial_masks()

    def _setup_coordinate_grids(self):
        """Create coordinate grids for both numpy and tensorflow"""
        # Numpy version (for Bayesian decoder)
        x_cent_np = np.linspace(0.5 / self.GRID_W, 1 - 0.5 / self.GRID_W, self.GRID_W)
        y_cent_np = np.linspace(0.5 / self.GRID_H, 1 - 0.5 / self.GRID_H, self.GRID_H)
        self.Xc_np, self.Yc_np = np.meshgrid(x_cent_np, y_cent_np, indexing="xy")

        # TensorFlow version (for ANN decoder)
        x_cent_tf = tf.linspace(0.5 / self.GRID_W, 1 - 0.5 / self.GRID_W, self.GRID_W)
        y_cent_tf = tf.linspace(0.5 / self.GRID_H, 1 - 0.5 / self.GRID_H, self.GRID_H)
        self.Xc_tf, self.Yc_tf = tf.meshgrid(x_cent_tf, y_cent_tf, indexing="xy")

    def _extract_maze_boundaries(self, maze_params=None) -> Dict[str, float]:
        """
        Extract maze boundaries from provided parameters or default coordinates.
        Args:
            maze_params (dict or array-like): If dict, should contain keys:
                'x_min', 'x_max', 'y_min', 'y_max', 'gap_x_min', 'gap_x_max', 'gap_y_min'.
                If array-like, should be an array of shape (N, 2) with (x, y) coordinates.
                If None, defaults to predefined MAZE_COORDS.
        Returns:
            dict: Extracted maze boundaries.
        """

        if maze_params is None or not isinstance(maze_params, dict):
            if maze_params is not None:
                maze_coords = np.array(maze_params)
            else:
                maze_coords = MAZE_COORDS
            maze_params = {
                "x_min": maze_coords[:, 0].min(),
                "x_max": maze_coords[:, 0].max(),
                "y_min": maze_coords[:, 1].min(),
                "y_max": maze_coords[:, 1].max(),
                "gap_x_min": maze_coords[-2, 0],
                "gap_x_max": maze_coords[-4, 0],
                "gap_y_min": maze_coords[-3, 1],
            }
        return maze_params

    def _create_spatial_masks(self) -> Tuple[np.ndarray, tf.Tensor]:
        """Create spatial constraint masks for both numpy and tensorflow"""
        # Create forbidden region mask
        # Note: Using your original logic where FORBID=1 means forbidden

        # Numpy version
        forbid_np = (
            (self.Xc_np > self.maze_params_dict["gap_x_min"])
            & (self.Xc_np < self.maze_params_dict["gap_x_max"])
            & (self.Yc_np <= self.maze_params_dict["gap_y_min"])
        ).astype(np.float32)

        # TensorFlow version
        forbid_tf = tf.cast(
            (self.Xc_tf > self.maze_params_dict["gap_x_min"])
            & (self.Xc_tf < self.maze_params_dict["gap_x_max"])
            & (self.Yc_tf <= self.maze_params_dict["gap_y_min"]),
            tf.float32,
        )

        return forbid_np, forbid_tf

    def get_allowed_mask(self, use_tensorflow=False):
        """Get allowed mask"""
        if use_tensorflow:
            return 1.0 - self.forbid_mask_tf
        else:
            return 1.0 - self.forbid_mask_np

    def remove_isolated_zeros(self, forbid_mask, occ):
        # Apply same zero-count bin removal logic as bayes occupation map
        # Identify zero-count bins in allowed regions
        allowed_mask = ~forbid_mask
        zero_threshold = (
            np.mean(occ[allowed_mask & (occ > 0)]) * 0.1
            if np.any(allowed_mask & (occ > 0))
            else 0
        )

        # identiy problematic zones
        print(
            f"Occupation map: {np.sum(occ == 0)} zero-occupation bins, {np.sum(occ[allowed_mask.astype(bool)] < zero_threshold)} low-density bins (below {zero_threshold:.4e}) in allowed zones."
        )
        low_density = (occ < zero_threshold) & allowed_mask

        # Expand forbidden zones by 1 pixel
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        expanded_forbidden = ndimage.binary_dilation(
            forbid_mask, structure=structure, iterations=5
        )

        # Remove low-density bins adjacent to forbidden zones
        problematic_bins = low_density & expanded_forbidden
        forbid_mask = forbid_mask | problematic_bins  # Update forbidden mask
        occ[problematic_bins] = 0.0

        print(
            f"Weight map: Removed {np.sum(problematic_bins)} low-density bins adjacent to forbidden zones"
        )

        self.update_allowed_mask(forbid_mask)
        return forbid_mask, occ

    def update_allowed_mask(self, forbid_mask):
        self.forbid_mask_np = forbid_mask.astype(np.float32)  # dynamic update for ANN
        self.forbid_mask_tf = tf.cast(forbid_mask, tf.float32)  # dynamic update for ANN


def smooth_signal(signal, N):
    """
    Simple smoothing by convolving a filter with 1/N.

    Parameters
    ----------
    signal : array_like
        Signal to be smoothed
    N : int
        smoothing_factor

    Returns
    -------
    signal : array_like
            Smoothed signal
    """
    if N is True:
        N = 10
    # Preprocess edges
    signal = np.concatenate([signal[0:N], signal, signal[-N:]])
    # Convolve
    signal = np.convolve(signal, np.ones((N,)) / N, mode="same")
    # Postprocess edges
    signal = signal[N:-N]

    return signal
