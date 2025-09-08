"""
FullEncoder submodule: Encoding models for neural data analysis.

Exposes:
        - LSTMandSpikeNetwork (from an_network)
        - an_network (module containing neural network architectures)
"""

from . import an_network
from .an_network import LSTMandSpikeNetwork
