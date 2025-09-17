# Get libs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import config


def manage_devices(usedDevice: str = "GPU", set_memory_growth=True) -> str:
    """
    Manage the devices used by TensorFlow.

    Parameters
    ----------
    usedDevice : str
        The device to be used, either "CPU" or "GPU".

    Returns
    -------
    devicename : str
        The name of the device being used (in TF nomenclature).
    """
    # if gpu set memory growth
    if usedDevice == "GPU":
        device_phys = config.list_physical_devices(usedDevice)
        if set_memory_growth:
            if not device_phys:
                raise ValueError("No GPU devices found.")
            # set memory growth to True
            for device in device_phys:
                config.experimental.set_memory_growth(device, True)
                print(f"Memory growth set for device: {device}")
    # get the name of the device
    device = config.list_logical_devices(usedDevice)
    if device:
        devicename = device[0].name
    else:
        devicename = device = config.list_logical_devices()[0].name

    return devicename


def convert_normalized_to_euclidian(normalized, maxPos):
    return normalized * maxPos
