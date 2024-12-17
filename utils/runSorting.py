#!/usr/bin/env python
# coding: utf-8
import glob
import os
import sys
from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

## import glob
from probeinterface import generate_linear_probe
from spikeinterface.sorters import run_sorter_jobs

file_to_load = Path(sys.argv[1])

required_extensions = [
    "random_spikes",
    "waveforms",
    "templates",
    "noise_levels",
    "unit_locations",
    "template_similarity",
    "spike_amplitudes",
    "correlograms",
]

sorter_names = ["mountainsort5", "kilosort4", "spykingcircus2"]

dat = Path(sys.argv[1])

try:
    print(dat)
except:
    print("No file provided")
    if os.path.exists(dat):
        dat = os.path.abspath(dat)
        print(dat)


recording = se.NeuroScopeRecordingExtractor(dat)

print(f"Recording loaded: {recording}")

num_elec = recording.get_num_channels()
probe = generate_linear_probe(
    num_elec=num_elec,
    ypitch=20,
    contact_shapes="circle",
    contact_shape_params={"radius": 6},
)
probe.set_device_channel_indices(np.arange(num_elec))
recording = recording.set_probe(probe)
recording = spre.depth_order(recording)

print(f"Probe generated: {probe}")

# run sorter (if not already done)

# we need to respect this structure
# job_list = [
#   {'sorter_name': 'tridesclous', 'recording': recording, 'output_folder': 'folder1','detect_threshold': 5.},
#   {'sorter_name': 'tridesclous', 'recording': another_recording, 'output_folder': 'folder2', 'detect_threshold': 5.},
#   {'sorter_name': 'herdingspikes', 'recording': recording, 'output_folder': 'folder3', 'clustering_bandwidth': 8., 'docker_image': True},
#   {'sorter_name': 'herdingspikes', 'recording': another_recording, 'output_folder': 'folder4', 'clustering_bandwidth': 8., 'docker_image': True},
# ]

job_list = []
print(f"Checking sorters for {dat}")
for sorter_name in sorter_names:
    output_folder = Path(dat.parent) / f"sorters/sorter_output_{sorter_name}"
    if output_folder.exists():
        print(f"Sorter {sorter_name} already run")
    else:
        print(f"Sorter {sorter_name} not run, adding to job list.")
        job_list.append(
            {
                "sorter_name": sorter_name,
                "recording": recording,
                "output_folder": output_folder,
            }
        )


# run all sorters in job_list
print(f"Running sorters for {dat}")
sortings = run_sorter_jobs(job_list=job_list, engine="joblib")
