sudo apt-get install python3-dev python3-pip python3-venv python3-tk jq xmlstarlet
if { conda env list | grep 'base'; } >/dev/null 2>&1; then
    echo "Conda base environment exists, creating new environment for neuroEncoder"
    conda env create -f environment.yml
else
    echo "Conda base environment does not exist"
    echo "Please install conda"
    exit 1
fi
