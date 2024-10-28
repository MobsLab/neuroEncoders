# Load libs
import os.path

# Load custom code
from ..importData.rawdata_parser import DataHelper


class Project:
    """
    Class to store the paths of the project.
    xmlPath: path to the xml find
    datPath: path to the dat file
    jsonPath: path to the json file
    nameExp: name of the experiment, defaults to "Network"
    """

    def __init__(self, xmlPath, datPath="", jsonPath=None, nameExp="Network"):
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
        self.resultsPath = os.path.join(
            self.folder, nameExp
        )  # Allows change at every experiment
        if not os.path.isdir(self.dataPath):
            os.makedirs(self.dataPath)
        if not os.path.isdir(self.resultsPath):
            os.makedirs(self.resultsPath)
        if not os.path.isdir(os.path.join(self.resultsPath, "results")):
            os.makedirs(os.path.join(self.resultsPath, "results"))
        # Json
        if jsonPath == None:
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


class Params:
    """
    Class to store the parameters of the project.

        helper: instance of the DataHelper class
        windowSize: size of the window in seconds (see neuroencoder for arg default)
        nEpochs: number of epochs to train the network
    """

    def __init__(self, helper: DataHelper, windowSize: float, nEpochs=100):
        self.nGroups = helper.nGroups()
        self.dimOutput = helper.dim_output()
        self.nChannelsPerGroup = helper.numChannelsPerGroup()
        self.length = 0

        self.nSteps = int(10000 * 0.036 / windowSize)  # useless
        self.nEpochs = nEpochs
        self.windowLength = windowSize  # in seconds

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
        self.batchSize = 128  # Change that if your GPU (or CPU) is not powerful enough

        # TODO: check if this is still relevant
        # we might want to introduce some Adam or stuff like that
        self.learningRates = [0.0003]  #  [0.00003, 0.00003, 0.00001]
        self.lossActivation = None  # tf.nn.relu

        self.usingMixedPrecision = False
        # enforcing float16 computations whenever possible
        # According to tf tutorials, we can allow that in most layer
        # except the output for unclear reasons linked to gradient computations
