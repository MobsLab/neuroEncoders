#!/bin/bash

# If some files are missing, run some matlab code to generate them
path=$(dirname "$1")
path="$path/"
log=mlog.out
if [ ! -f "$path""SleepScoring_OBGamma.mat" ] || [ ! -f "$path""$log" ] || [ ! -f "$path""SleepScoring_Accelero.mat" ]
then

        rm -f $path$log
        echo
        echo MATLAB is now exporting behavior data, please see $log for more infos.
cat <<EOF | matlab -nodesktop -nosplash -nodisplay /> $path$log
        extractSleepState $1;
        exit
EOF

        matlabExitCatch=$?
        if test $matlabExitCatch -ne 0; then
                echo
                echo Matlab was unable to extract data from "$path""SWR.mat", does this file exist ?
                exit 1
        fi
fi
if [ ! -f "$path""SleepScoring_OBGamma.mat" ] || [ ! -f "$path""$log" ]  || [ ! -f "$path""SleepScoring_Accelero.mat" ]; then
        echo A problem happened while extracting matlab tsd.
        exit 1
else
        echo Behavior data exported.
fi
