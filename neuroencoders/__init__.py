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
    decoder as decoder,
)
from . import (
    fullEncoder as fullEncoder,
)
from . import (
    importData as importData,
)
from . import (
    openEphysExport as openEphysExport,
)
from . import (
    resultAnalysis as resultAnalysis,
)
from . import (
    simpleBayes as simpleBayes,
)
from . import (
    transformData as transformData,
)
from . import (
    utils as utils,
)
