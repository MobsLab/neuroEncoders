import os
import subprocess
import json

def generate_json_fromProject(projectPath,list_channels):
    outjsonStr = {};
    outjsonStr['encodingPrefix'] = projectPath.graph
    outjsonStr['defaultFilter'] = not useOpenEphysFilter
    outjsonStr['mousePort'] = 0

    outjsonStr['nGroups'] = int(len(list_channels))
    idx = 0
    for group in range(len(list_channels)):
        outjsonStr['group' + str(group - idx)] = {}
        outjsonStr['group' + str(group - idx)]['nChannels'] = len(list_channels[group])
        for chnl in range(len(list_channels[group])):
            outjsonStr['group' + str(group - idx)]['channel' + str(chnl)] = int(list_channels[group][chnl])
            #I don't think this is ever used, so for mow I remove it.
            # Note: is is called in the createOpenEphysTemplate... so we might have to remove it from there too.
            # (Pierre)
            # outjsonStr['group' + str(group - idx)]['threshold' + str(chnl)] = int(
            #     spikeDetector.getThresholds()[group][chnl])



    outjsonStr['nStimConditions'] = 1
    outjsonStr['stimCondition0'] = {}
    outjsonStr['stimCondition0']['stimPin'] = 14
    outjsonStr['stimCondition0']['lowerX'] = 0.0
    outjsonStr['stimCondition0']['higherX'] = 0.0
    outjsonStr['stimCondition0']['lowerY'] = 0.0
    outjsonStr['stimCondition0']['higherY'] = 0.0
    outjsonStr['stimCondition0']['lowerDev'] = 0.0
    outjsonStr['stimCondition0']['higherDev'] = 0.0

    outjson = json.dumps(outjsonStr, indent=4)
    with open(projectPath.json, "w") as json_file:
        json_file.write(outjson)

    subprocess.run(["./createOpenEphysTemplateFromJson.sh", projectPath.json])