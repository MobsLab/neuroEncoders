import json
import os
import subprocess

import pandas as pd

from importData.rawdata_parser import get_params


def generate_json(
    projectPath, modelPath, listChannels, useOpenEphysFilter=False, offline=False
):
    # Start creating json
    outjsonStr = {}
    outjsonStr["encodingPrefix"] = modelPath
    outjsonStr["defaultFilter"] = not useOpenEphysFilter
    outjsonStr["mousePort"] = 0
    outjsonStr["nGroups"] = int(len(listChannels))
    # Get thresholds
    df = pd.read_csv(os.path.join(projectPath.folder, "thresholds_Julia.csv"))
    thresholdsJulia = df.values[-len(listChannels) :, :]
    thresholdsParsed = [
        [float(s) for s in t.split("Float32[")[1].split("]")[0].split(" ")]
        for t in thresholdsJulia[:, 1]
    ]
    thresholds = [[] for _ in range(len(listChannels))]
    for i, j in enumerate(thresholdsJulia[:, 0]):
        if offline:
            thresholds[j - 1] = [
                t / 0.195 for t in thresholdsParsed[i]
            ]  ## in offline mode the data
        else:
            thresholds[j - 1] = thresholdsParsed[i]
    # Write thresholds
    idx = 0
    for group in range(len(listChannels)):
        outjsonStr["group" + str(group - idx)] = {}
        outjsonStr["group" + str(group - idx)]["nChannels"] = len(listChannels[group])
        for chnl in range(len(listChannels[group])):
            outjsonStr["group" + str(group - idx)]["channel" + str(chnl)] = int(
                listChannels[group][chnl]
            )
            # I don't think this is ever used, so for mow I remove it.
            # Note: it is called in the createOpenEphysTemplate... so we might have to remove it from there too.
            # (Pierre)
            outjsonStr["group" + str(group - idx)]["threshold" + str(chnl)] = int(
                thresholds[group][chnl]
            )

    # Write stimulation conditions
    outjsonStr["nStimConditions"] = 1
    outjsonStr["stimCondition0"] = {}
    outjsonStr["stimCondition0"]["stimPin"] = 14
    outjsonStr["stimCondition0"]["lowerX"] = 0.0
    outjsonStr["stimCondition0"]["higherX"] = 0.0
    outjsonStr["stimCondition0"]["lowerY"] = 0.0
    outjsonStr["stimCondition0"]["higherY"] = 0.0
    outjsonStr["stimCondition0"]["lowerDev"] = 0.0
    outjsonStr["stimCondition0"]["higherDev"] = 0.0

    outjson = json.dumps(outjsonStr, indent=4)
    with open(projectPath.json, "w") as jsonFile:
        jsonFile.write(outjson)

    # Create json
    if not offline:
        subprocess.run(
            [
                os.path.join(os.path.dirname(__file__), "create_json.sh"),
                projectPath.json,
            ]
        )

    else:
        # create a structure.oebin file:
        file = open(os.path.join(projectPath.folder, "structure.oebin"), mode="r")
        outjson = json.load(file)
        listChannels, _, nChannels = get_params(projectPath.xml)
        exampleChannel = outjson["continuous"][0]["channels"][0].copy()
        exampleChannelAux = outjson["continuous"][0]["channels"][-1].copy()
        # For compatibility issues: here we will assume that auxiliary channels are the last 4 channels:
        # one with 'digital' input (last), three with other kind of signals...
        # we just set them to be auxiliary channels as they won't be use in the online decoding analysis...
        assert os.path.exists(os.path.join(projectPath.folder, "continuous"))
        assert os.path.exists(os.path.join(projectPath.folder, "continuous", "hab"))
        newDict = outjson.copy()
        newDict["continuous"][0]["folder_name"] = "hab"
        newDict["continuous"][0]["source_processor_name"] = "offline input"
        newDict["continuous"][0]["recorded_processor"] = "offline input"
        newDict["continuous"][0]["num_channels"] = nChannels
        newDict["continuous"][0]["channels"] = []
        # nbTrueChannel = sum([len(l) for l in listChannels])
        for e in range(nChannels):
            newExampleChannel = exampleChannel.copy()
            newExampleChannel["channel_name"] = "CH" + str(e + 1)
            newExampleChannel["source_processor_index"] = e
            newExampleChannel["recorded_processor_index"] = e
            newDict["continuous"][0]["channels"] = newDict["continuous"][0][
                "channels"
            ] + [newExampleChannel]
        for i in range(4):
            newExampleChannel = exampleChannelAux.copy()
            newExampleChannel["channel_name"] = "AUX" + str(i + 1)
            newExampleChannel["source_processor_index"] = i + nChannels
            newExampleChannel["recorded_processor_index"] = i + nChannels
            newDict["continuous"][0]["channels"] = newDict["continuous"][0][
                "channels"
            ] + [newExampleChannel]
        file.close()

        filer = open(os.path.join(projectPath.folder, "structure2.oebin"), mode="w")
        json.dump(newDict, filer, indent=0)
        filer.close()

        subprocess.run(["./openEphysExport/create_json_offline.sh", projectPath.json])
