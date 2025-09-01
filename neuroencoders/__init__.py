"""
Neuroencoders: Python package for neural data analysis.

Submodules:
        - decoder: Decoding models and utilities
        - importData: Data import and parsing
        - fullEncoder: Encoding models
        - openEphysExport: OpenEphys data export tools
        - resultAnalysis: Analysis and visualization of results
        - simpleBayes: Bayesian decoding tools
        - transformData: Data transformation utilities
        - utils: General utilities
"""

from importlib.metadata import PackageNotFoundError

try:
    from importlib.metadata import version

    __version__ = version("neuroencoders")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback if not installed

from . import (
    decoder,
    fullEncoder,
    importData,
    openEphysExport,
    resultAnalysis,
    simpleBayes,
    transformData,
    utils,
)
