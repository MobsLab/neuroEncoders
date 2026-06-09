import os
import subprocess

from neuroencoders.utils.func_wrappers import timing

# Load custom code
from neuroencoders.utils.global_classes import Project


@timing
def julia_spike_filter(
    projectPath: Project,
    folderCode,
    windowSize=0.036,
    windowStride=0.036,
    strideFactor=4,
    singleSpike=False,
    BUFFERSIZE=72000,
    redo=False,
):
    """
    Launch an extraction of the spikes in Julia:
    This function is used to extract the spikes from the nnBehavior.mat file using Julia.
    The spikes are then saved in a csv file and then converted to a tfrec file with the corresponding striding.

    args:
    projectPath: Project object, containing the paths to the xml and dat files
    folderCode: str, path to the folder containing the neuroEncoder code
    windowSize: float, size of the window in seconds (default = 0.036)
    windowStride: float, size of the extract-stride in seconds (default = 0.036)
    singleSpike: bool, if True, the spikes are extracted without any striding (default = False)
    BUFFERSIZE: int, size of the buffer for the tfrec file (default = 72000)
    redo : bool, if True, the function will redo the extraction even if the tfrec file already exists (default = False)

    """
    if singleSpike:
        test1 = os.path.isfile(
            (os.path.join(projectPath.folder, "dataset", "dataset_singleSpike.tfrec"))
        )
    else:
        if strideFactor == 1:
            filename = os.path.join(
                projectPath.folder,
                "dataset",
                f"dataset_stride{str(round(windowSize * 1000))}.tfrec",
            )
            sleepFilename = os.path.join(
                projectPath.folder,
                "dataset",
                f"datasetSleep_stride{str(round(windowSize * 1000))}.tfrec",
            )
        else:
            filename = os.path.join(
                projectPath.folder,
                "dataset",
                f"dataset_stride{str(round(windowSize * 1000))}_factor{str(strideFactor)}.tfrec",
            )
            sleepFilename = os.path.join(
                projectPath.folder,
                "dataset",
                f"datasetSleep_stride{str(round(windowSize * 1000))}_factor{str(strideFactor)}.tfrec",
            )
        test1 = os.path.isfile(filename) and os.path.isfile(sleepFilename)
    if redo:
        test1 = False

    if not test1:
        if not os.path.exists(os.path.join(projectPath.folder, "nnBehavior.mat")):
            raise ValueError(
                "the behavior file does not exist :"
                + os.path.join(projectPath.folder, "nnBehavior.mat")
                + " Please run the behavior extraction first using the extractTsd.m function - should be handled by neuroEncoder main script as well."
            )
        if not os.path.exists(projectPath.dat):
            raise ValueError("the dat file does not exist :" + projectPath.dat)
        if "juliaData" not in folderCode:
            codepath = os.path.join(folderCode, "importData/juliaData/")
        else:
            codepath = folderCode
        # TODO: Update to have the correct code path
        if singleSpike:
            subprocess.run(
                [
                    os.path.join(codepath, "executeFilter_singleSpike.sh"),
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
            print(
                f"Extracting spikes with window size {windowSize} and stride {windowStride}. This will create the files {filename} and {sleepFilename} and may take a while..."
            )
            subprocess.run(
                [
                    os.path.join(codepath, "executeFilter_stride.sh"),
                    codepath,
                    projectPath.xml,
                    projectPath.dat,
                    os.path.join(projectPath.folder, "nnBehavior.mat"),
                    os.path.join(projectPath.folder, "spikeData_fromJulia.csv"),
                    filename,
                    sleepFilename,
                    str(BUFFERSIZE),
                    str(windowSize),
                    str(windowStride),
                ]
            )
