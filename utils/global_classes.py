# Load libs
import json
import os.path
import sys
from datetime import date

import dill as pickle

# Load custom code
sys.path.append("../importData")
import os
import sys
from warnings import warn

# import matplotlib as mplt
# mplt.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from shapely import MultiPoint, Polygon

from importData.rawdata_parser import get_behavior, get_params

sys.path.append("./importData")


# Custom codes
from importData import epochs_management as ep


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
        *args,
        **kwargs,
    ):
        """
        Initializes the DataHelper object.

        Args:
        --------

        - xmlPath: should be a xmlPath to instantiate a Project object.
        - *args: additional positional arguments to pass to the Project constructor.
        - **kwargs: additional keyword arguments to pass to the Project constructor.
            - force_ref: whether to force the computation of the reference and xy coordinates, even if they are already saved.
        """
        # Handle positional arguments
        if xmlPath is None and len(args) >= 3:
            xmlPath, mode, target = args[0], args[1], args[2]
            args = args[3:]
        self.mode = mode
        self.force_ref = kwargs.get("force_ref", False)
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
        batchSize = kwargs.pop("batchSize", 256)
        save_json = kwargs.pop("save_json", False)

        # Initialize attributes
        self.date = date.today().strftime("%Y-%m-%d")

        # Store parameters
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        if not hasattr(self, "windowSize"):
            self.windowSize = windowSize  # in seconds
            self.windowSizeMS = int(windowSize * 1000)  # in milliseconds

        # add the helper object
        self.helper = helper
        # Initialize all other parameters...
        self._initialize_params_attributes(helper)
        # save those to json file
        if save_json:
            self.save_params_to_json()
        save_project_to_pickle(
            self, output=os.path.join(self.resultsPath, "Parameters.pkl")
        )
        self._copy_from_helper(helper)
        # re initialize for method vs attribute
        self._initialize_params_attributes(helper)

    def _initialize_params_attributes(self, helper):
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

        # TODO: check if this is still relevant
        # WARNING: maybe striding is actually 0.036 ms based ???
        self.nSteps = int(10000 * 0.036 / self.windowSize)  # used in the encoder

        ### from units encoder params
        self.validCluWindow = 0.0005
        self.kernel = "epanechnikov"  # is not connected
        self.bandwidth = 0.1
        self.masking = 20

        ### full encoder params
        self.nFeatures = 64
        self.lstmLayers = 2
        self.dropoutCNN = 0.2
        self.lstmSize = 128
        self.lstmDropout = 0.3  # is not implemented(code uses self.dropout_CNN)
        self.batchSize = 256  # Change that if your GPU (or CPU) is not powerful enough

        # TODO: check if this is still relevant
        # we might want to introduce some Adam or stuff like that - update : RMSProp quite good
        self.learningRates = [0.0003]  #  [0.00003, 0.00003, 0.00001]
        self.lossActivation = None  # tf.nn.relu

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
