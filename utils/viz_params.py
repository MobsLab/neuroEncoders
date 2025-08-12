#!/usr/bin/env python3

"""
This module defines visualization parameters for plotting functions.
"""

from cmcrameri import cm as cmc

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
