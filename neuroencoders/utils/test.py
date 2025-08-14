#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:57:05 2020

@author: quarantine-charenton
"""

# %%
import matplotlib.pyplot as plt
from MOBS_Functions import Info_LFP, Load_LFP

# %% Info LFP

LFP_directory = "/home/mickey/Documents/Theotime/DimaERC2/neuroencoders_1021/_work_reloaded/M1199_MFB/exp1/LFPData/"

LFP_info = Info_LFP(LFP_directory)

# LFP_Info is a DataFrame where indexes are channels' number and the columns are the parameters including
# the path to load the LFP

# You can accesses the paramaters' name by using .columns with optionnal .tolist() :

Feat_LFP = LFP_info.columns.tolist()

######################## example of use : find the LFP coming from dHPC

LFP_dHPC_index = LFP_info.index[
    LFP_info["structure"] == "dHPC"
].tolist()  # to find indexes

# then to get their path or their depth

LFP_dHPC_path = LFP_info.loc[LFP_dHPC_index, "path"]
LFP_dHPC_depth = LFP_info.loc[LFP_dHPC_index, "depth"]

# Then you can load them

# %% Load LFP

# To load one LFP

n = 41  # Number of the LFP

LFP_path = LFP_info.loc[n, "path"]  # Get the path

LFP = Load_LFP(LFP_path, time_unit="us")  # Load it (time_unit 'us', 'ms' or 's')

# To load multiple LFP

N = [12, 34, 53]
LFP_paths = LFP_info.loc[N, "path"]
LFPs = Load_LFP(LFP_paths, "s")

LFP.plot()
plt.show(block=True)

# If you want to load LFP with structure 'Accelero'

LFP_Accelero_index = LFP_info.index[LFP_info["structure"] == "Accelero"].tolist()
LFP_Accelero_path = LFP_info.loc[LFP_Accelero_index, "path"]  # DataFrame with paths

LFP_Accelero = Load_LFP(LFP_Accelero_path, time_unit="us")
