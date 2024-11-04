#!/bin/bash
# This script is used to execute the spike extraction using the julia language
# The script receives the following parameters:
# $1: The path to this code directory
# $2: The path of the xml file that links to the data structure
# $3: The path of the dat file that contains the data
# $4: The path of the nnBehavior.mat file that contains the behavior data
# $5: The path of the output csv file that will be created by the julia code
# $6: The path of the dataset_stride*.tfrec that will be created by tensorflow
# $7: The path of the datasetSleep_stride*.tfrec that will be created by tensorflow
# $8: The buffer size
# $9: The window size in seconds
# $10: The window stride size in fractions of window size

echo "Starting Spike extraction using the julia language"
cd $1
pwd
julia -t auto spikeFilter_withStride.jl --projectDir=$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}
