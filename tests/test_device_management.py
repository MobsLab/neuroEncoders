import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from neuroencoders.utils.management import manage_devices


class TestDeviceManagement(unittest.TestCase):
    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.config.list_logical_devices")
    @patch("tensorflow.config.experimental.set_memory_growth")
    def test_manage_devices_cpu(self, mock_growth, mock_logical, mock_physical):
        mock_logical.return_value = [MagicMock(name="/device:CPU:0")]
        mock_logical.return_value[0].name = "/device:CPU:0"

        device = manage_devices("CPU")
        self.assertEqual(device, "/device:CPU:0")
        mock_growth.assert_not_called()

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.config.list_logical_devices")
    @patch("tensorflow.config.experimental.set_memory_growth")
    def test_manage_devices_gpu_single(self, mock_growth, mock_logical, mock_physical):
        mock_physical.return_value = [MagicMock()]
        mock_logical.return_value = [MagicMock()]
        mock_logical.return_value[0].name = "/device:GPU:0"

        device = manage_devices("GPU")
        self.assertEqual(device, "/device:GPU:0")
        mock_growth.assert_called_once()

    @patch("tensorflow.distribute.MirroredStrategy")
    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.config.experimental.set_memory_growth")
    def test_manage_devices_multi_gpu(
        self, mock_growth, mock_physical, mock_strategy_class
    ):
        mock_physical.return_value = [MagicMock(), MagicMock()]
        mock_strategy = MagicMock(spec=tf.distribute.Strategy)
        mock_strategy_class.return_value = mock_strategy

        strategy = manage_devices("MULTI-GPU")
        self.assertEqual(strategy, mock_strategy)

    def test_manage_devices_specific_gpu(self):
        with patch("tensorflow.config.list_logical_devices") as mock_logical:
            mock_logical.return_value = [
                MagicMock(name="/device:GPU:0"),
                MagicMock(name="/device:GPU:1"),
            ]
            mock_logical.return_value[0].name = "/device:GPU:0"
            mock_logical.return_value[1].name = "/device:GPU:1"

            device = manage_devices("GPU:1")
            self.assertEqual(device, "/device:GPU:1")

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.config.list_logical_devices")
    @patch("tensorflow.config.experimental.set_memory_growth")
    def test_manage_devices_multi_gpu_fallback(
        self, mock_growth, mock_logical, mock_physical
    ):
        # Simulate only one physical GPU
        mock_physical.return_value = [MagicMock()]
        # Simulate one logical GPU
        mock_logical.return_value = [MagicMock()]
        mock_logical.return_value[0].name = "/device:GPU:0"

        with self.assertWarns(UserWarning):
            device = manage_devices("MULTI-GPU")
            self.assertEqual(device, "/device:GPU:0")


if __name__ == "__main__":
    unittest.main()
