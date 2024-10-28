# Get libs
import tensorflow as tf


def manage_devices(usedDevice):
    # TODO: let the user chose 1 GPU or all GPUs
    # if gpu set memory growth
    if usedDevice == "GPU":
        device_phys = tf.config.list_physical_devices(usedDevice)
        tf.config.experimental.set_memory_growth(device_phys[0], True)
    # get the name of the device
    device = tf.config.list_logical_devices(usedDevice)
    if device:
        devicename = [dv.name for dv in device]
        if len(devicename) == 1:
            devicename = devicename[0]
    else:
        devicename = device = tf.config.list_logical_devices()[0].name

    return devicename


def convert_normalized_to_euclidian(normalized, maxPos):
    return normalized * maxPos
