#!/usr/bin/env python3

"""
This module defines visualization parameters for plotting functions.
"""

import numpy as np
from cmcrameri import cm as cmc
from matplotlib.colors import LinearSegmentedColormap

white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0, "#ffffff"),
        (1e-20, "#ffffff"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)
EC = np.array([45, 39])  # range of x and y in cm

TRUE_COLOR = "xkcd:royal blue"
CURRENT_POINT_COLOR = "red"
TRUE_LINE_COLOR = "xkcd:royal blue"
PREDICTED_COLOR = "xkcd:peach"
CURRENT_PREDICTED_POINT_COLOR = "xkcd:peach"
PREDICTED_LINE_COLOR = "xkcd:bluey grey"

SHOCK_COLOR = "xkcd:hot pink"
SAFE_COLOR = "cornflowerblue"
SHOCK_COLOR_PREDICTED = "xkcd:carnation"
SAFE_COLOR_PREDICTED = "xkcd:bright sky blue"
MIDDLE_COLOR = "xkcd:lavender"

FREEZING_POINTS_COLOR = "xkcd:light blue"
FREEZING_LINE_COLOR = "xkcd:light blue"
RIPPLES_COLOR = "xkcd:rusty orange"
ALL_STIMS_COLOR = "xkcd:pale pink"

COLORMAP = cmc.buda
PREDICTED_CMAP = cmc.imola

ALPHA_TRAIL_LINE = 0.6
ALPHA_TRAIL_POINTS = 1
ALPHA_DELTA_LINE = 0.6
BINARY_COLORS = None
HLINES = None
VLINES = None
REMOVE_TICKS = True
WITH_REF_BG = True
DELTA_COLOR = "xkcd:vivid purple"
DELTA_COLOR_CONCORDANT = "xkcd:electric green"
DELTA_COLOR_DISCORDANT = "xkcd:red orange"
MAX_NUM_STARS = 5
