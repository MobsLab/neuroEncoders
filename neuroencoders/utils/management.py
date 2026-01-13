# Get libs
from pathlib import Path


def manage_devices(usedDevice: str = "GPU", set_memory_growth=True) -> str:
    """
    Manage the devices used by TensorFlow.

    Parameters
    ----------
    usedDevice : str
        The device to be used, e.g., "CPU", "GPU", "MULTI-GPU", "GPU:0", "GPU:1".

    Returns
    -------
    device : str or tf.distribute.Strategy
        The name of the device or a distribution strategy.
    """
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import config

    # if gpu set memory growth
    if "GPU" in usedDevice.upper():
        device_phys = config.list_physical_devices("GPU")
        if set_memory_growth:
            if not device_phys:
                raise ValueError("No GPU devices found.")
            # set memory growth to True
            for device in device_phys:
                try:
                    config.experimental.set_memory_growth(device, True)
                    print(f"Memory growth set for device: {device}")
                except Exception as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(f"Warning: Could not set memory growth for {device}: {e}")

    if usedDevice.upper() == "MULTI-GPU":
        device_phys = config.list_physical_devices("GPU")
        if len(device_phys) > 1:
            print(
                f"Initializing Multi-GPU strategy (MirroredStrategy) with {len(device_phys)} GPUs"
            )
            strategy = tf.distribute.MirroredStrategy()
            return strategy
        elif len(device_phys) == 1:
            import warnings

            warnings.warn(
                "MULTI-GPU requested but only one GPU found. Using single GPU mode."
            )
            usedDevice = "GPU"  # Fallback to single GPU logic below
        else:
            raise ValueError("MULTI-GPU requested but no GPU devices found.")

    if ":" in usedDevice:
        # specific device like GPU:0 or CPU:0
        dev_type, dev_idx = usedDevice.split(":")
        logical_devices = config.list_logical_devices(dev_type.upper())
        if int(dev_idx) < len(logical_devices):
            return logical_devices[int(dev_idx)].name
        else:
            raise ValueError(
                f"Requested device {usedDevice} but only {len(logical_devices)} {dev_type} devices found."
            )

    # get the name of the device
    device = config.list_logical_devices(usedDevice.upper())
    if device:
        devicename = device[0].name
    else:
        # fallback to first available logical device
        all_logical = config.list_logical_devices()
        if all_logical:
            devicename = all_logical[0].name
        else:
            devicename = "/device:CPU:0"

    return devicename


def convert_normalized_to_euclidian(normalized, maxPos):
    return normalized * maxPos


def get_git_info():
    import subprocess

    repo_dir = Path(__file__).resolve().parent

    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
            .decode("utf-8")
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir
            )
            .decode("utf-8")
            .strip()
        )
        try:
            tag = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--abbrev=0"], cwd=repo_dir
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            tag = None  # No tags found
        return {"commit": commit_hash, "branch": branch, "tag": tag}
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "branch": "unknown", "tag": None}
