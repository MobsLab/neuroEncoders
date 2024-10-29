import os
import subprocess

# Load custom code
from utils.global_classes import Project

BUFFERSIZE = 72000


def julia_spike_filter(
    projectPath: Project, folderCode, windowSize=0.200, singleSpike=False
):
    """
    Launch an extraction of the spikes in Julia:
    This function is used to extract the spikes from the nnBehavior.mat file using Julia.
    The spikes are then saved in a csv file and then converted to a tfrec file with the correspondig striding.

    args:
    projectPath: Project object, containing the paths to the xml and dat files
    folderCode: str, path to the folder containing the neuroEncoder code

    """
    if singleSpike:
        test1 = os.path.isfile(
            (os.path.join(projectPath.folder, "dataset", "dataset_singleSpike.tfrec"))
        )
    else:
        test1 = os.path.isfile(
            (
                os.path.join(
                    projectPath.folder,
                    "dataset",
                    "dataset_stride" + str(round(windowSize * 1000)) + ".tfrec",
                )
            )
        )
    if not test1:
        if not os.path.exists(os.path.join(projectPath.folder, "nnBehavior.mat")):
            raise ValueError(
                "the behavior file does not exist :"
                + os.path.join(projectPath.folder, "nnBehavior.mat")
            )
        if not os.path.exists(projectPath.dat):
            raise ValueError("the dat file does not exist :" + projectPath.dat)
        codepath = os.path.join(folderCode, "importData/juliaData/")
        if singleSpike:
            subprocess.run(
                [
                    codepath + "executeFilter_singleSpike.sh",
                    codepath,
                    projectPath.xml,
                    projectPath.dat,
                    os.path.join(projectPath.folder, "nnBehavior.mat"),
                    os.path.join(projectPath.folder, "spikeData_fromJulia.csv"),
                    os.path.join(
                        projectPath.folder, "dataset", "dataset_singleSpike.tfrec"
                    ),
                    os.path.join(
                        projectPath.folder, "dataset", "datasetSleep_singleSpike.tfrec"
                    ),
                    str(BUFFERSIZE),
                    str(windowSize),
                ]
            )
        else:
            subprocess.run(
                [
                    os.path.join(codepath, "executeFilter_stride.sh"),
                    codepath,
                    projectPath.xml,
                    projectPath.dat,
                    os.path.join(projectPath.folder, "nnBehavior.mat"),
                    os.path.join(projectPath.folder, "spikeData_fromJulia.csv"),
                    os.path.join(
                        projectPath.folder,
                        "dataset",
                        "dataset_stride" + str(round(windowSize * 1000)) + ".tfrec",
                    ),
                    os.path.join(
                        projectPath.folder,
                        "dataset",
                        "datasetSleep_stride"
                        + str(round(windowSize * 1000))
                        + ".tfrec",
                    ),
                    str(BUFFERSIZE),
                    str(windowSize),
                    str(0.200),
                ]
            )  # the striding is 36ms based...
