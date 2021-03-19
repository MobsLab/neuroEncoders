#!/bin/bash

# If some files are missing, run some matlab code to generate them
path=$(dirname "$1")
path="$path/"
log=mlog.out
prjpath="$HOME/Dropbox/Kteam/PrgMatlab/OnlinePlaceDecoding"
if [ ! -f "$path""nnBehavior.mat" ] || [ ! -f "$path""$log" ]
then

        rm -f $path$log
        echo
        echo MATLAB is now exporting behavior data, please see $log for more infos.
cat <<EOF | matlab -nodesktop -nosplash -nodisplay /> $path$log
        cd $prjpath
        LoadPATHencoder $USER;
        cd $path;
        export_spikes($2, $3, $4);
        exit
EOF

        matlabExitCatch=$?
        if test $matlabExitCatch -ne 0; then
                echo
                echo Matlab was unable to extract data from "$path""behavResources.mat", does this file exist ?
                exit 1
        fi
fi
if [ ! -f "$path""nnBehavior.mat" ] || [ ! -f "$path""$log" ]; then
        echo A problem happened while extracting matlab tsd.
        exit 1
else
        echo Behavior data exported.
fi
