import os
import subprocess

BUFFERSIZE = 72000
def julia_spike_filter(projectPath, folderCode, windowSize = 0.036, singleSpike=False):
    # Launch an extraction of the spikes in Julia:
    if singleSpike:
        test1 = os.path.isfile((os.path.join(projectPath,"dataset","dataset_singleSpike.tfrec")))
    else:
        test1 = os.path.isfile((os.path.join(projectPath, "dataset", "dataset_stride"+str(round(windowSize*1000))+".tfrec")))
    if not test1 :
        if not os.path.exists(os.path.join(projectPath,'behavResources.mat')):
            raise ValueError('the behavior file does not exist :' + os.path.join(projectPath,'behavResources.mat'))
        if not os.path.exists(projectPath):
            raise ValueError('the dat file does not exist :' + projectPath)
        codepath = os.path.join(folderCode,"importData/juliaData/")
        if singleSpike:
            subprocess.run([codepath + "executeFilter_singleSpike.sh",
                            codepath,
                            projectPath,
                            projectPath,
                            os.path.join(projectPath,"behavResources.mat"),
                            os.path.join(projectPath,"spikeData_fromJulia.csv"),
                            os.path.join(projectPath,"dataset","dataset_singleSpike.tfrec"),
                            os.path.join(projectPath, "dataset", "datasetSleep_singleSpike.tfrec"),
                            str(BUFFERSIZE),
                            str(windowSize)])
        else:
            subprocess.run([os.path.join(codepath, "executeFilter_stride.sh"),
                            codepath,
                            projectPath,
                            projectPath,
                            os.path.join(projectPath,"behavResources.mat"),
                            os.path.join(projectPath,"spikeData_fromJulia.csv"),
                            os.path.join(projectPath,"dataset","dataset_stride"+str(round(windowSize*1000))+".tfrec"),
                            os.path.join(projectPath, "dataset", "datasetSleep_stride"+str(round(windowSize*1000))+".tfrec"),
                            str(BUFFERSIZE),
                            str(windowSize),
                            str(0.036)]) #the striding is 36ms based...

