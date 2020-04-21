# neuroEncoders
Using ANN to compute spike trains and position from electrophysiology.
Usage :
```
./neuroEncoder <full | fromUnits>
```
"full" mode extracts spikes from a .dat or a .fil file, and creates a neural network graph from the spikes to the position. "fromUnits" mode loads units from .clu and .res files and creates a neural net to classify them correctly, using then a bayesian decoder to get to the position.

The option ```-p <path>``` allows you to specify the path to the .xml of your dataset (if not present, the script will ask it from you). The script assumes that the .dat is in the same folder as this .xml (and with the same name !). The position and speed are taken from the variables Xtsd, Ytsd, Vtsd of the behavResources.mat file (in the same folder as the other files).

By default, the script also uses data from a .fil file to get the spikes. If you don't have one of those, you can add the option -f or --filter so the script filters data itself.

If tensorflow is installed with gpu version, you can add the option -g or --gpu so the script runs in gpu mode. The data is pretty complicated to load, so the gpu acceleration is not that great on a regular workstation.

# Installation
You need to have python3 installed and all the packages in "requirements.txt". To do this easily on an ubuntu machine:
```
sudo apt-get install python3 python3-dev python3-pip python3-venv python3-tk
pip3 install -r requirements.txt
```
It is advised to use a virtual environment, you can learn how to do that on google (it's not mandatory). The version of tensorflow in requirements.txt is not the gpu version. If you feel confortable enough to install the gpu version, you can use : https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070 but you have to NOT install the cpu version first (which would happen if you use the pip3 command shown previously).
