"""
Unified hardware acceleration backend for neuroencoders.
Dynamically routes computations to GPU (NVIDIA RAPIDS) or CPU (Pandas/NumPy/SciPy/Skimage).
"""

import logging

logger = logging.getLogger("neuroencoders.backend")

# 1. Dataframes: cuDF (GPU) vs Pandas (CPU)
try:
    import cudf as pd  # noqa: F401
    import dask_cudf  # noqa: F401

    HAS_GPU_DF = True
except ImportError:
    dask_cudf = None
    HAS_GPU_DF = False

# 2. Machine Learning: cuML (GPU) vs Scikit-Learn (CPU)
try:
    import cuml as ml  # noqa: F401

    HAS_GPU_ML = True
except ImportError:
    HAS_GPU_ML = False

# 3. Image/Signal Processing: cuCIM (GPU) vs Scikit-Image (CPU)
try:
    import cucim.skimage as skimage  # noqa: F401

    HAS_GPU_IMG = True
except ImportError:
    HAS_GPU_IMG = False

# 4. Graph Network Analysis: cuGraph (GPU) vs NetworkX (CPU)
try:
    import cugraph as graph  # noqa: F401
    import networkx as nx  # noqa: F401

    HAS_GPU_GRAPH = True
except ImportError:
    graph = None
    HAS_GPU_GRAPH = False


def get_backend_status():
    """Returns a dictionary indicating which components are GPU accelerated."""
    status = {
        "dataframes": "GPU (cuDF)" if HAS_GPU_DF else "CPU (Pandas)",
        "machine_learning": "GPU (cuML)" if HAS_GPU_ML else "CPU (Scikit-Learn)",
        "image_processing": "GPU (cuCIM)" if HAS_GPU_IMG else "CPU (Scikit-Image)",
        "graph_networks": "GPU (cuGraph)" if HAS_GPU_GRAPH else "CPU (NetworkX)",
    }
    return status


def log_backend_info():
    """Logs the current hardware acceleration status."""
    logger.info("--- Neuroencoders Backend Routing ---")
    for component, backend in get_backend_status().items():
        logger.info(f"{component.replace('_', ' ').title()}: {backend}")
