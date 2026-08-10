#!/bin/bash

cd "$(dirname "$0")"
# if path is a file we take the directory of the file, if it is a directory we take the directory itself
if [ -f "$1" ]; then
    path=$(dirname "$1")
else
    path="$1"
fi


path="$path/"
log=mlog_swr.out

if [ -f "$path""SWR.mat" ] || [ -f "$path""Ripples.mat" ] || [ -f "$path""ripples.mat" ]
then

        rm -f $path$log
        echo
        echo MATLAB is now exporting behavior data, please see "$path""$log" for more infos.
cat <<EOF | matlab -nodesktop -nosplash -nodisplay /> $path$log
        extractSWR $1;
        exit
EOF

        matlabExitCatch=$?
        if test $matlabExitCatch -ne 0; then
                echo
                echo Matlab was unable to extract data from "$path""SWR.mat", does this file exist ?
                exit 1
        fi
else 
    echo Did not run matlab because none of "$path""SWR.mat", "ripples.mat" or "Ripples.mat" exists . Please check the log file for more information.
fi
if [ ! -f "$path""nnSWR.mat" ] || [ ! -f "$path""$log" ]; then
        echo A problem happened while extracting matlab tsd.
        exit 1
else
        echo Ripples data exported.
fi
