#!/usr/bin/env python

import argparse

import ephyviewer
import numpy as np
import spikeinterface.extractors as se
from ephyviewer import (
    MainViewer,
    TimeFreqViewer,
    TraceViewer,
    mkQApp,
)

argparser = argparse.ArgumentParser(description="View Neuroscope data")
argparser.add_argument("--filename", "-f", type=str, help="Path to Neuroscope data")
args = argparser.parse_args()
# you must first create a main Qt application (for event loop)
app = mkQApp()
# Create the main window that can contain several viewers
win = MainViewer(debug=False, show_auto_scale=True)

recording = se.extractor_classes.NeuroScopeRecordingExtractor(file_path=args.filename)
recording._set_neuroscope_groups()

sig_source = ephyviewer.SpikeInterfaceRecordingSource(recording=recording)

# create a viewer for signal with TraceViewer
view1 = TraceViewer(source=sig_source, name="signals")
view1.params["scale_mode"] = "same_for_all"
view1.params["xsize"] = 5.0
view1.auto_scale()

colors = recording.get_property("colors")
discarded = recording.get_property("discarded_channels")
channel_groups = recording.get_property("neuroscope_group")
num_channels = recording.get_num_channels()

# Set a group-based offset: the first group will start at n_channels, and then each channel will get -1, until the last channel from the last group, which will be at 0.
# This way, the first group will be at the top of the plot, and the last group will be at the bottom.:

# 1. Get the number of channels in each group
group_id, group_sizes = np.unique(channel_groups, return_counts=True)
# 2. Calculate the offsets for each group
group_offsets = [sum(group_sizes[:i]) for i in range(len(group_sizes))]
# 3. Create a list of offsets for each channel by starting at n_channels and then subtracting the index of the channel in the group
# prefill the offsets with zeros
offsets = np.zeros(num_channels, dtype=int)
for group_id in channel_groups:
    group_offset = group_offsets[group_id]
    group_channels = np.where(channel_groups == group_id)[0]
    # Assign the offset for each channel in the group
    for i, channel in enumerate(group_channels):
        offsets[channel] = num_channels - 1 - group_offset - i
# 5. Set the colors and visibility for each channel
for i, channel in enumerate(channel_groups):
    view1.by_channel_params[f"ch{i}", "color"] = colors[i]
    if discarded[i]:
        view1.by_channel_params[f"ch{i}", "visible"] = False

view1.auto_scale()

current_offsets = np.array(
    [view1.by_channel_params["ch{}".format(i), "offset"] for i in range(num_channels)]
)

# get indices where the offsets are different by more than 20%
indices = np.where(np.abs(current_offsets - offsets) > 0.2 * current_offsets)[0]

# 4. Set the offsets for each channel in the view1
for idx in indices:
    view1.by_channel_params[f"ch{idx}", "offset"] = offsets[idx]

# create a time freq viewer connected to the same source
view2 = TimeFreqViewer(source=sig_source, name="tfr")

view2.params["xsize"] = 5.0
view2.params["show_axis"] = True
view2.params["timefreq", "deltafreq"] = 1
view2.params["timefreq", "f_start"] = 90
view2.params["timefreq", "f_stop"] = 250
view2.params["timefreq", "f0"] = 1
view2.params["timefreq", "normalisation"] = 0
view2.auto_scale()

# add them to mainwindow
win.add_view(view1)
win.add_view(view2)
view1.auto_scale()
view2.auto_scale()


# show main window and run Qapp
win.show()
app.exec()
