import os
import subprocess

BUFFERSIZE = 72000
def julia_spike_filter(projectPath,window_length = 0.036,singleSpike=True):
    # Launch an extraction of the spikes in Julia:
    test1 = os.path.isfile((os.path.join(projectPath.folder,"dataset","spikeData_fromJulia.csv")))
    if singleSpike:
        test2 = os.path.isfile((os.path.join(projectPath.folder,"dataset","dataset_singleSpike.tfrec")))
    else:
        # test2 = os.path.isfile((os.path.join(projectPath.folder, "dataset", "dataset.tfrec")))
        test2 = os.path.isfile((os.path.join(projectPath.folder, "dataset", "dataset_stride"+str(round(window_length*1000))+".tfrec")))
    if not test2 :
        if not os.path.exists(projectPath.folder + 'nnBehavior.mat'):
            raise ValueError('the behavior file does not exist :' + projectPath.folder + 'nnBehavior.mat')
        if not os.path.exists(projectPath.dat):
            raise ValueError('the dat file does not exist :' + projectPath.dat)
        if singleSpike:
            subprocess.run(["./../importData/JuliaData/executeFilter_singleSpike.sh",
                            "/home/mobs/Documents/PierreCode/neuroEncoders/importData/JuliaData/", #todo Correct here
                            projectPath.xml,
                            projectPath.dat,
                            os.path.join(projectPath.folder,"nnBehavior.mat"),
                            os.path.join(projectPath.folder,"spikeData_fromJulia.csv"),
                            os.path.join(projectPath.folder,"dataset","dataset_singleSpike.tfrec"),
                            os.path.join(projectPath.folder, "dataset", "datasetSleep_singleSpike.tfrec"),
                            str(BUFFERSIZE),
                            str(window_length)])
        else:
            subprocess.run(["./../importData/JuliaData/executeFilter_stride.sh",
                            "/home/mobs/Documents/PierreCode/neuroEncoders/importData/JuliaData/", #todo Correct here
                            projectPath.xml,
                            projectPath.dat,
                            os.path.join(projectPath.folder,"nnBehavior.mat"),
                            os.path.join(projectPath.folder,"spikeData_fromJulia.csv"),
                            os.path.join(projectPath.folder,"dataset","dataset_stride"+str(round(window_length*1000))+".tfrec"),
                            os.path.join(projectPath.folder, "dataset", "datasetSleep_stride"+str(round(window_length*1000))+".tfrec"),
                            str(BUFFERSIZE),
                            str(window_length),
                            str(0.036)]) #the striding is 36ms based...

            # Let us obtain all spike dataset in parallel:
            # import threading
            # class worker:
            #     def run(self,args):
            #         subprocess.run(args)
            # threads = [threading.Thread(target=worker(), args=["./../importData/JuliaData/executeFilter_stride.sh",
            #                             "/home/mobs/Documents/PierreCode/neuroEncoders/importData/JuliaData/", #todo Correct here
            #                             projectPath.xml,
            #                             projectPath.dat,
            #                             os.path.join(projectPath.folder,"nnBehavior.mat"),
            #                             os.path.join(projectPath.folder,"spikeData_fromJulia.csv"),
            #                             os.path.join(projectPath.folder,"dataset","dataset_stride"+str(wl)+".tfrec"),
            #                             os.path.join(projectPath.folder, "dataset", "datasetSleep_stride"+str(wl)+".tfrec"),
            #                             str(BUFFERSIZE),
            #                             str(wl*0.036),
            #                             str(0.036)]) for wl in [1,3,7,14]]
            # for thread in threads:
            #     thread.start()
            # # wait for all thread to end:
            # for thread in threads:
            #     thread.join()



