"""
ImportData submodule: Functions to load and parse neural data from various sources.

Exposes:
        - inEpochsMask, get_epochs, merge_intervals, etc. (from epochs_management)
"""

from . import juliaData, rawdata_parser
from .epochs_management import get_epochs, inEpochs, inEpochsMask, merge_intervals
