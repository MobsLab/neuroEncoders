"""
ImportData submodule: Functions to load and parse neural data from various sources.

Exposes:
        - inEpochsMask, get_epochs, merge_intervals, etc. (from epochs_management)
"""

from . import juliaData as juliaData
from . import rawdata_parser as rawdata_parser
from .epochs_management import get_epochs as get_epochs
from .epochs_management import inEpochsMask as inEpochsMask
from .epochs_management import merge_intervals as merge_intervals
