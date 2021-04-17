import os
import subprocess

BUFFERSIZE = 10000
def julia_spike_filter(projectPath,window_length = 0.036,eraseSpike=False):
    # Launch an extraction of the spikes in Julia:
    if not os.path.exists(projectPath.folder + 'nnBehavior.mat'):
        raise ValueError('the behavior file does not exist :' + projectPath.folder + 'nnBehavior.mat')
    if not os.path.exists(projectPath.dat):
        raise ValueError('the dat file does not exist :' + projectPath.dat)
    test1 = os.path.isfile((os.path.join(projectPath.folder,"dataset","spikeData_fromJulia.csv")))
    test2 = os.path.isfile((os.path.join(projectPath.folder,"dataset","dataset.tfrec")))
    if not test1 and not test2 :
        subprocess.run(["./../importData/JuliaData/executeFilter.sh",
                        "/home/mobs/Documents/PierreCode/neuroEncoders/importData/JuliaData/",
                        projectPath.xml,
                        projectPath.dat,
                        os.path.join(projectPath.folder,"nnBehavior.mat"),
                        os.path.join(projectPath.folder,"spikeData_fromJulia.csv"),
                        os.path.join(projectPath.folder,"dataset","dataset.tfrec"),
                        str(BUFFERSIZE)])

