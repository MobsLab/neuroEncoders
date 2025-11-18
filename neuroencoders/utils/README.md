# For those willing to have an alternative to Neuroscope's default viewer, here is a simple guide to get you started with using a custom ephyviewer based on our neuroscope usage.

## Prerequisites

1. Make sure you have Python installed on your system. You will also need to install the following packages in a virtual environment specific to this viewer and only this viewer (be it Conda, venv, or uv):

```bash
pip install ephyviewer spikeinterface numpy
```

2. Either clone the [neuroEncoders](https://github.com/MobsLab/neuroEncoders/) repository (our repo to compute spike trains and positions from CA1 ephy), or simply download the relevant python script `all_with_spike_interface.py` from the `neuroEncoders/utils/` folder.

## Installation and usage

1. Go the `all_with_spike_interface.py` script you just downloaded or cloned. Change the very first line to point to your local python interpreter inside your virtual environment (the one where you installed ephyviewer and spikeinterface):

```python
#!/path/to/your/virtualenv/bin/python
```

Mine is located in a .venv folder in my neuroEncoders directory, so I would write:

```python
#!/home/mickey/Documents/Theotime/neuroEncoders/.venv/bin/python
```

2. Make sure that the python interpreter and the Prerequisites packages are correctly installed in your virtual environment by running INSIDE THE `all_with_spike_interface.py` FOLDER:

```bash
chmod +x all_with_spike_interface.py
./all_with_spike_interface.py
```

3. You're all set! You can now run the script with your data. The script takes as input the path to a `.dat` or `.fil` file containing the electrophysiological recording (any Neuroscope-like format should be supported here by SpikeInterface). If the same folder contains KlustaKwik spike sorting results (i.e. a `.clu.n` and `.res.n` file), those will be automatically loaded as well. They need to have the same base name as the recording file.

Example usage:

```bash
./all_with_spike_interface.py /path/to/your/recording.dat
```

4. Bonus: If you want to directly click on a `.dat` or `.fil` file to open it with this viewer, you can simply link the `.py` script to the executable of your system. On Linux, you can create a symbolic link in `/usr/local/bin`:

```bash
sudo ln -s /path/to/all_with_spike_interface.py /usr/local/bin/ephyviewer_neuroscope
```

Then, depending on your distro, you might be able to right-click on a `.dat` or `.fil` file, choose "Open With" or "Properties/Open With", and select `ephyviewer_neuroscope` as the default application.
For GNOME distributions, you might need to create a `.desktop` file in `~/.local/share/applications/` to register the application properly:
I called mine MOBSViewer.desktop with the following content:

```ini
[Desktop Entry]
Version=1.0
Type=Application
Name=MOBSViewer
Comment=Quick ephyviewer that looks like neuroscope
Exec=/usr/bin/ephyviewer_neuroscope %U
Terminal=false
StartupNotify=false
Categories=Utility
MimeType=application/x-dat;application/x-fil;
```

It's this application that you will then be able to select as the default app for `.dat` or `.fil` files.
