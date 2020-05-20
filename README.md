# neuroEncoders
Using ANN to compute spike trains and position from electrophysiology.
Usage :
```
./neuroEncoder <full | fromUnits>
```
"full" mode extracts spikes from a .dat or a .fil file, and creates a neural network graph from the spikes to the position. "fromUnits" mode loads units from .clu and .res files and creates a neural net to classify them correctly, using then a bayesian decoder to get to the position.

The option ```-p <path>``` allows you to specify the path to the .xml of your dataset (if not present, the script will ask it from you). The script assumes that the .dat is in the same folder as this .xml (and with the same name !). The position is taken from the variables Xtsd and Ytsd of the behavResources.mat file (in the same folder as the other files), unless the option -t or --target is used.

If you wish to decode something else than position, you can use the option ```--target <nameOfTsd>``` to specify the name of tsd object inside the behavResources.mat file. If specified, the data from this object will be used as target for the decoding.

The option ```--window <windowLength>``` allows you to specify the window size (in seconds). Defaults to 0.036.

By default, the script also uses data from a .fil file to get the spikes. If you don't have one of those, you can add the option -f or --filter so the script filters data itself.

If tensorflow is installed with gpu version, you can add the option -g or --gpu so the script runs in gpu mode. The data isn't straightforward to load, so the gpu acceleration is not always great on a regular workstation.

# Installation
CPU not supported for installation for now
```
sudo ./install.sh
```
