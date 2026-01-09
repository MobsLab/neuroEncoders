#!/usr/bin/env python
# coding: utf-8

# ## Load results

# In[4]:


import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

datadir = os.path.join(os.path.expanduser("~/Documents/Theotime"), "DimaERC2")
assert os.path.isdir(datadir)


# In[7]:


# What Basile did
BasileMiceNumber = [
    "1239vBasile",
    "1281vBasile",
    "1199",
    "1336",
    "1168MFB",
    "905",
    "1161w1199",
    "1161",
    "1124",
    "1186",
    "1182",
    "1168UMaze",
    "1117",
    "994",
    "1336v3",
    "1336v2",
    "1281v2",
    "1239v3",
]

# What Dima did according to Baptiste
DimaMiceNumber = [
    "905",
    "906",
    "911",
    "994",
    "1161",
    "1162",
    "1168",
    "1186",
    "1230",
    "1239",
]

# Files wrt to datadir
path_list = [
    "M1239TEST3_Basile_M1239/TEST",
    "M1281TEST3_Basile_1281MFB/TEST",
    "M1199TEST1_Basile/TEST",
    "M1336_Known/TEST/",
    # "DataERC2/M994/20191013/TEST/",
    # "DataERC2/M906/TEST/",
    "DataERC2/M1168/TEST/",
    "DataERC2/M905/TEST/",
    "DataERC2/M1161/TEST_with_1199_model/",
    "DataERC2/M1161/TEST initial/",
    "DataERC2/M1124/TEST/",
    "DataERC2/M1186/TEST/",
    "DataERC2/M1182/TEST/",
    "DataERC1/M1168/TEST/",
    "DataERC1/M1117/TEST/",
    "neuroencoders_1021/_work/M994_PAG/Final_results_v3",
    "neuroencoders_1021/_work/M1336_MFB/Final_results_v3",
    "neuroencoders_1021/_work/M1336_known/Final_results_v2",
    "neuroencoders_1021/_work/M1281_MFB/Final_results_v2",
    "neuroencoders_1021/_work/M1239_MFB/Final_results_v3",
]
assert len(BasileMiceNumber) == len(path_list)
len(BasileMiceNumber)
path_dict = dict(zip(BasileMiceNumber, path_list))


# In[9]:


# 1168MFB does not have any results
path_dict.pop("1168MFB")


# In[10]:


path_dict


# In[11]:


def get_size(file_path, unit="bytes"):
    file_size = os.path.getsize(file_path)
    exponents_map = {"bytes": 0, "kb": 1, "mb": 2, "gb": 3}
    if unit not in exponents_map:
        raise ValueError(
            "Must select from \
        ['bytes', 'kb', 'mb', 'gb']"
        )
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)


# In[12]:


keys_to_include = set()
size_dat = dict()
for mouse, path in path_dict.items():
    path = os.path.join(datadir, path, "../")
    if len(glob.glob(path + "*.dat")) >= 1:
        keys_to_include.add(mouse)
        size_dat[mouse] = get_size(glob.glob(path + "*.dat")[0], unit="gb")

dath_dict = {k: path_dict[k] for k in keys_to_include}


# In[13]:


size_dat


# In[14]:


conditions = {
    "MFB": ["1281vBasile", "1281v31239vBasile", "1239v3", "1336v3", "1336v2"],
    "Known": ["1336", "1336v3"],
    "PAG": ["1186", "1161", "1161w1199", "1124", "1186", "1117", "1199", "994"],
    "Umaze": ["1199", "906", "1168", "905", "1182"],
    # WARNING: 994 has non-aligned nnbehavior.positions; hence the results should not be trusted
}

list_windows = [36, 108, 200, 252, 504]


# In[15]:


from resultAnalysis.print_results import print_results

# In[16]:


list_windows


# In[17]:


# bypass to avoid heavy comput and fill the memory for nothing
force = False

todo = dict()
dirmouse = dict()
mouse_id = []
windowMS = []
mean_eucl = []
select_eucl = []
mean_lin = []
select_lin = []
has_dat = []
sizes = []
for mouse, path in path_dict.items():
    todo[mouse] = []
    returned = False
    dirmouse[mouse] = os.path.join(datadir, path, "results")
    assert os.path.isdir(dirmouse[mouse])
    for win in list_windows:
        try:
            mean, select, linmean, linselect = print_results(
                dirmouse[mouse], show=False, windowSizeMS=win, force=False
            )
            mean_eucl.append(mean)
            select_eucl.append(select)
            mean_lin.append(linmean)
            select_lin.append(linselect)
            mouse_id.append(mouse)
            windowMS.append(win)
            has_dat.append(mouse in dath_dict)
            if mouse in dath_dict:
                sizes.append(size_dat[mouse])
            else:
                sizes.append(0)
            returned = True
        except Exception as e:
            print(e)
            todo[mouse].append(win)
            print(f"Available windows: {os.listdir(dirmouse[mouse])}")
    if not returned:
        print(f"nothing at all for {mouse}, {os.listdir(dirmouse[mouse])}")


results_df = pd.DataFrame(
    data={
        "mouse_id": mouse_id,
        "windowMS": windowMS,
        "mean_eucl": mean_eucl,
        "select_eucl": select_eucl,
        "mean_lin": mean_lin,
        "select_lin": select_lin,
        "has_dat": has_dat,
        "size_dat": sizes,
    }
)


# In[18]:


todo


# In[19]:


for cdt in conditions:
    for mouse in conditions[cdt]:
        try:
            results_df.loc[results_df.mouse_id == mouse, "condition"] = cdt
        except Exception as e:
            print(e)

results_df = results_df.sort_values(
    by=["condition", "mouse_id", "windowMS"]
).reset_index(drop=True)


# In[20]:


# In[21]:


# ## Let's focus on M994 and 12390 (good results, PAG & MFB, several windowMS)

# In[22]:


selected_mice = ["994", "1239v3"]


# In[23]:


subresults_df = results_df[results_df["mouse_id"].isin(selected_mice)]


# In[24]:


# In[25]:


import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.qualitymetrics

# In[26]:


dirmouse


# In[27]:


sorting_folder = dict()

for mouse in selected_mice:
    sorting_folder[mouse] = os.path.join(Path(dirmouse[mouse]).parents[1])


# In[28]:


sorting_folder


# In[29]:


si.set_global_job_kwargs(n_jobs=-1, progress_bar=True)


# In[63]:


## import glob
from probeinterface import generate_linear_probe
from spikeinterface.sorters import read_sorter_folder, run_sorter

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

recording = dict()
sorting = dict()
v2sorting = dict()
analyzer = dict()
for mouse in selected_mice:
    dat = glob.glob(os.path.join(sorting_folder[mouse], "*.dat"))[0]
    print(dat)
    recording[mouse] = se.NeuroScopeRecordingExtractor(dat)
    print(recording[mouse])
    sorting[mouse] = se.NeuroScopeSortingExtractor(os.path.join(sorting_folder[mouse]))

    num_elec = recording[mouse].get_num_channels()
    probe = generate_linear_probe(
        num_elec=num_elec,
        ypitch=20,
        contact_shapes="circle",
        contact_shape_params={"radius": 6},
    )
    probe.set_device_channel_indices(np.arange(num_elec))
    recording[mouse] = recording[mouse].set_probe(probe)
    recording[mouse] = spre.depth_order(recording[mouse])
    # run sorter (if not already done)
    sortings = {}
    for sorter_name in sorter_names:
        # output_folder = Path(sorting_folder[mouse]) / f'sorter_output_{sorter_name}'
        output_folder = (
            Path("/home/mickey/Documents/Theotime/speed/SpikeSorting/")
            / f"tmp_spikesorting/{mouse}/sorter_output_{sorter_name}"
        )
        print(sorter_name, output_folder)
        if output_folder.exists():
            sortings[sorter_name] = read_sorter_folder(output_folder)
        else:
            sortings[sorter_name] = run_sorter(
                sorter_name, recording[mouse], output_folder, verbose=True
            )

    sortings["init"] = sorting[mouse]
    v2sorting[mouse] = sortings
