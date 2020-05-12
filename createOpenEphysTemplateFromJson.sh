#!/bin/bash

# sudo apt-get install jq
# sudo apt-get install xmlstarlet

if [[ $# < 1 ]]
then
	echo usage : createOpenEphysTemplateFromJson /path/to/file.json
	exit 1
elif [[ $# > 1 ]]
then
	echo usage : createOpenEphysTemplateFromJson /path/to/file.json
	exit 1
else
	pathToJson=$1
fi

nElectrodes=0
nGroups=`jq '.nGroups' $pathToJson`
for i in `seq 1 $nGroups`
do
	groupChannels=`jq ".group$(( i-1 )).nChannels" $pathToJson`
	nElectrodes=$(( $nElectrodes+$groupChannels ))
done

destPath=`dirname $pathToJson`
pathToXml=$destPath/openEphysLoadFile.xml
cp $HOME/Dropbox/Kteam/intanMobsSource/open-ephys-install/openEphysLoadExample.xml $pathToXml


xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN -t elem -n PROCESSOR -v "" $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n name -v Filters/Spike\ Sorter $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n insertionPoint -v 1 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n pluginName -v Spike\ Sorter $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n pluginType -v 1 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n pluginIndex -v 4 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n libraryName -v Spike\ Sorter $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n libraryVersion -v 1 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n isSource -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n isSink -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t attr -n NodeId -v 102 $pathToXml

xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t elem -n SpikeSorter -v "" $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n numElectrodes -v $nElectrodes $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n numPreSamples -v 14 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n numPostSamples -v 18 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n autoDACassignement -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n syncThreshold -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n uniqueID -v 2 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t attr -n filpSignal -v 0 $pathToXml

xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t elem -n ELECTRODE_COUNTER $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE_COUNTER -t attr -n numElectrodeTypes -v 1 $pathToXml
xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE_COUNTER -t elem -n ELECTRODE_TYPE $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE_COUNTER/ELECTRODE_TYPE -t attr -n type -v Single\ Electrode  $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE_COUNTER/ELECTRODE_TYPE -t attr -n count -v $(( nElectrodes+1 ))  $pathToXml

electrodeIdx=1
for group in `seq 1 $nGroups`
do
	groupChannels=`jq ".group$(( group-1 )).nChannels" $pathToJson`

	for el in `seq 1 $groupChannels`
	do
		electrodeName=Group\ $group\ [$el/$groupChannels]
		xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter -t elem -n ELECTRODE $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n name -v "$electrodeName" $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n numChannels -v 1 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n prePeakSamples -v 14 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n postPeakSamples -v 18 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n advancerID -v -1 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n depthOffsetMM -v 0 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t attr -n electrodeID -v $electrodeIdx $pathToXml

		channelNumber=`jq ".group$(( group-1 )).channel$(( el-1 ))" $pathToJson`
		threshold=`jq ".group$(( group-1 )).threshold$(( el-1 ))" $pathToJson`
		xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t elem -n SUBCHANNEL $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SUBCHANNEL -t attr -n ch -v $channelNumber $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SUBCHANNEL -t attr -n thresh -v -$threshold $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SUBCHANNEL -t attr -n isActive -v 1 $pathToXml

		xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx] -t elem -n SPIKESORTING $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SPIKESORTING -t attr -n numBoxUnits -v 0 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SPIKESORTING -t attr -n numPCAUnits -v 0 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SPIKESORTING -t attr -n selectedUnits -v -1 $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SpikeSorter/ELECTRODE[$electrodeIdx]/SPIKESORTING -t attr -n selectedBox -v -1 $pathToXml

		spikeChannelName=group$(( group-1 )).$(( $el-1 ))
		xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[3] -t elem -n SPIKECHANNEL $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SPIKECHANNEL[$electrodeIdx] -t attr -n name -v $spikeChannelName $pathToXml
		xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[3]/SPIKECHANNEL[$electrodeIdx] -t attr -n number -v $(( electrodeIdx-1 )) $pathToXml
		electrodeIdx=$(( electrodeIdx+1 ))
	done

done

xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN -t elem -n PROCESSOR -v "" $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n name -v Filters/Position\ Decoder $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n insertionPoint -v 1 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n pluginName -v Position\ Decoder $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n pluginType -v 1 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n pluginIndex -v 13 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n libraryName -v Online\ Decoding $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n libraryVersion -v 1 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n isSource -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n isSink -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t attr -n NodeId -v 103 $pathToXml

xmlstarlet ed -L -s /SETTINGS/SIGNALCHAIN/PROCESSOR[4] -t elem -n PositionDecoderProcessor -v "" $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4]/PositionDecoderProcessor -t attr -n numParameters -v 0 $pathToXml
xmlstarlet ed -L -i /SETTINGS/SIGNALCHAIN/PROCESSOR[4]/PositionDecoderProcessor -t attr -n path -v $pathToJson $pathToXml