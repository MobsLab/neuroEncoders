# neuroEncoders

Using ANN to compute spike trains and position from electrophysiology.
Usage :

```
./neuroEncoder <ann | bayes> <path> ...
```

"ann" mode extracts spikes from a .dat file, and creates a neural network graph from the spikes to the position. "bayes" mode loads units from .clu and .res files and use a bayesian decoder to get to the position. "decode" mode takes already existing graph from `.json` file to decode position of the animal from the new file. "compare" mode will plot the figures comparing results of bayesian decoder and neuroEncoders on the same file

The argument `<path>` specifies the path to the .xml of your dataset. The script assumes that the .dat is in the same folder as this .xml (and with the same name !). The position is taken from the variables AlignedXtsd and AlignedYtsd (in their absence - Xtsd and Ytsd) of the behavResources.mat file (in the same folder as the other files), unless the option -t or --target is used.

If you wish to decode something else than position, you can use the option `--target <nameOfTsd>` to specify the name of tsd object inside the behavResources.mat file. If specified, the data from this object will be used as target for the decoding. We get decoded features from Matlab files, which requires both Matlab installation and presense of Tsd package. Tsd package (originally written by [Francesco Battaglia](https://www.ru.nl/english/people/battaglia-f/)) will be distributed along with the main code because it is hard to find. One place to find it is [here](https://github.com/PeyracheLab/TStoolbox)

The option `--window <windowLength>` allows you to specify the window size (in seconds). Default is 0.036 (s).

In "compare" mode, the script also uses data from a .fil file to get the spikes. If you don't have one of those, please create it by applying hipass filter on you .dat file (`ndm_hipass` command in Neuroscope suite)

If tensorflow is installed with gpu version, you can use the option `--gpu` so the script runs in gpu mode. The data isn't straightforward to load, so the gpu acceleration may not be great on a regular workstation.

# Installation - Updated 2024-10-15 (theotime branch)

The application is written in four languages: the main one is Python, the three auxiliary ones are Matlab, Julia and shell.

- Matlab was tested on versions from 2016 to 2019
- Shell part was only tested on Ubuntu computers <= 18.04 LTS, and Ubuntu 24.04 LTS
- Julia requires separate installation. It was _only_ tested on versin `1.5.3` (make sure Julia interpreter is contained in the PATH).
- neuroEncoders was tested on 3.6 <= Python <= 3.10.15

Running the following command will install the CPU version of tensorflow (tested on Python >= 3.8). Please make sure conda is installed on your computer.

```
sudo ./install.sh
```

To install GPU version, please install NVIDIA drivers, CUDA and CuDNN compatible with your version of operating system, graphics card and tensorflow 2.10.0.
The configuration of the GPU might be complex, especially with the integration of CUDA and CUDNN to tensorflow. Here, the pip version of tensorflow 2.10.0 was used with Nvidia driver 560.35.03, CUDA 11.2, CuDNN 8.1.0 and TensorRT 7.1.3.4. Manual installation was preferred to `.deb` packages when possible. CUDA versioning, PATH and LD_LIBRARY_PATH variables issues were mainly dealt thanks to [this post](https://stackoverflow.com/questions/78894063/how-can-i-resolve-tensorflow-warnings-cudnn-cufft-cublas-and-numa).

More information [here](https://www.tensorflow.org/install/pip#linux_1)
To test if installation was successful, please run :

```
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

To install FreeImage which might be needed to setup CuDNN, run :

```
sudo apt-get install libfreeimage3 libfreeimage-dev
```
