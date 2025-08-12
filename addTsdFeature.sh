#!/bin/bash

# If some files are missing, run some matlab code to generate them

# $0: The path to this code directory
# $1: The path of the dat file that contains the data
cd "$(dirname "$0")"
path=$(dirname "$1")
path="${path}/"
log="mlog_optional.out"
if [ ! -f "$path""optional_nnBehavior.mat" ] || [ ! -f "$path""$log" ]
then

        rm -f $path$log
        echo
        echo MATLAB is now exporting behavior data, please see $path$log for more info
cat <<EOF | matlab -nodesktop -nosplash -nodisplay /> $path$log
        addTsd $path $2;
        exit
EOF

        matlabExitCatch=$?
        if test $matlabExitCatch -ne 0; then
                echo
                echo Matlab was unable to extract data from "$path""behavResources.mat", does this file exist ?
                exit 1
        fi
fi
if [ ! -f "$path""optional_nnBehavior.mat" ] || [ ! -f "$path""$log" ]; then
        echo A problem happened while extracting matlab tsd.
        exit 1
else
        echo Behavior data exported.
fi
