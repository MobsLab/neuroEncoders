#!/bin/bash

cd "$(dirname "$0")"

if [ -f "$1" ]; then
    path=$(dirname "$1")
else
    path="$1"
fi

# Ensure trailing slash isn't doubled up later
path="${path%/}/"
log="mlog_sleepscoring.out"

if [ ! -f "${path}nnSleepScoring.mat" ]; then

    rm -f "${path}${log}"
    echo
    echo "MATLAB is now exporting behavior data, please see ${path}${log} for more info."
    
    # Corrected the heredoc piping syntax and wrapped $1 in single quotes
    matlab -nodesktop -nosplash -nodisplay > "${path}${log}" <<EOF
        extractSleepState '$1';
        exit
EOF

    matlabExitCatch=$?
    if [ $matlabExitCatch -ne 0 ]; then
        echo
        echo "Matlab was unable to extract data from ${path}SWR.mat, does this file exist?"
        exit 1
    fi
fi

# Final validation check
if [ ! -f "${path}nnSleepScoring.mat" ]; then
    echo "A problem happened while extracting matlab tsd."
    exit 1
else
    echo "Behavior data exported successfully."
fi
