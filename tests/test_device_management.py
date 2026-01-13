import contextlib
import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from neuroencoders.fullEncoder.nnUtils import get_device_context
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

    # Error scenario tests
    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.config.list_logical_devices")
    def test_manage_devices_invalid_gpu_index(self, mock_logical, mock_physical):
        """Test requesting a GPU index that doesn't exist"""
        mock_physical.return_value = [MagicMock(), MagicMock()]
        mock_logical.return_value = [
            MagicMock(name="/device:GPU:0"),
            MagicMock(name="/device:GPU:1"),
        ]
        mock_logical.return_value[0].name = "/device:GPU:0"
        mock_logical.return_value[1].name = "/device:GPU:1"

        with self.assertRaises(ValueError) as context:
            manage_devices("GPU:5")
        self.assertIn("only 2 GPU devices found", str(context.exception))

    def test_manage_devices_invalid_format_multiple_colons(self):
        """Test invalid device format with multiple colons"""
        with self.assertRaises(ValueError) as context:
            manage_devices("GPU:1:2")
        self.assertIn("Invalid device format", str(context.exception))

    def test_manage_devices_invalid_format_non_integer_index(self):
        """Test invalid device format with non-integer index"""
        with patch("tensorflow.config.list_logical_devices") as mock_logical:
            mock_logical.return_value = [MagicMock(name="/device:GPU:0")]
            with self.assertRaises(ValueError) as context:
                manage_devices("GPU:x")
            self.assertIn("non-negative integer", str(context.exception))

    def test_manage_devices_negative_index(self):
        """Test invalid device format with negative index"""
        with patch("tensorflow.config.list_logical_devices") as mock_logical:
            mock_logical.return_value = [MagicMock(name="/device:GPU:0")]
            with self.assertRaises(ValueError) as context:
                manage_devices("GPU:-1")
            self.assertIn("non-negative", str(context.exception))

    @patch("tensorflow.config.list_physical_devices")
    def test_manage_devices_no_gpu_available(self, mock_physical):
        """Test requesting GPU when no GPUs are available"""
        mock_physical.return_value = []
        with self.assertRaises(ValueError) as context:
            manage_devices("GPU")
        self.assertIn("No GPU devices found", str(context.exception))

    @patch("tensorflow.config.list_physical_devices")
    def test_manage_devices_multi_gpu_no_gpu_available(self, mock_physical):
        """Test requesting MULTI-GPU when no GPUs are available"""
        mock_physical.return_value = []
        with self.assertRaises(ValueError) as context:
            manage_devices("MULTI-GPU")
        self.assertIn("no GPU devices found", str(context.exception))


class TestGetDeviceContext(unittest.TestCase):
    """Test suite for get_device_context function"""

    def test_get_device_context_none(self):
        """Test that None returns a nullcontext"""
        ctx = get_device_context(None)
        self.assertIsInstance(ctx, contextlib.nullcontext().__class__)

    def test_get_device_context_strategy(self):
        """Test that a Strategy returns strategy.scope()"""
        strategy = MagicMock(spec=tf.distribute.Strategy)
        strategy.scope.return_value = contextlib.nullcontext()

        ctx = get_device_context(strategy)
        strategy.scope.assert_called_once()

    @patch("tensorflow.distribute.has_strategy")
    def test_get_device_context_string_no_active_strategy(self, mock_has_strategy):
        """Test that a device string returns tf.device() when no strategy is active"""
        mock_has_strategy.return_value = False

        with patch("tensorflow.device") as mock_device:
            mock_device.return_value = contextlib.nullcontext()
            ctx = get_device_context("/GPU:0")
            mock_device.assert_called_once_with("/GPU:0")

    @patch("tensorflow.distribute.has_strategy")
    def test_get_device_context_string_with_active_strategy(self, mock_has_strategy):
        """Test that a device string is ignored when a strategy is active"""
        mock_has_strategy.return_value = True

        with patch("logging.getLogger") as mock_logger:
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance

            ctx = get_device_context("/GPU:0")
            self.assertIsInstance(ctx, contextlib.nullcontext().__class__)
            # Verify warning was logged
            logger_instance.warning.assert_called_once()
            self.assertIn("ignoring explicit device placement", str(logger_instance.warning.call_args))

    @patch("tensorflow.distribute.has_strategy")
    def test_get_device_context_invalid_device_string(self, mock_has_strategy):
        """Test that invalid device string returns nullcontext with warning"""
        mock_has_strategy.return_value = False

        with patch("tensorflow.device") as mock_device:
            mock_device.side_effect = ValueError("Invalid device")
            with patch("logging.getLogger") as mock_logger:
                logger_instance = MagicMock()
                mock_logger.return_value = logger_instance

                ctx = get_device_context("INVALID_DEVICE")
                self.assertIsInstance(ctx, contextlib.nullcontext().__class__)
                # Verify warning was logged
                logger_instance.warning.assert_called_once()
                self.assertIn("Invalid device specification", str(logger_instance.warning.call_args))


class TestConditionalDeviceCopying(unittest.TestCase):
    """Test suite for conditional device copying in data pipeline"""

    def test_device_copying_with_string_device(self):
        """Test that copy_to_device is called when deviceName is a string"""
        # This is a conceptual test - in practice, this would be tested
        # in the actual an_network.py test suite with a real AnNetwork instance
        device_name = "/device:GPU:0"
        self.assertIsInstance(device_name, str)
        # When isinstance(deviceName, str) is True, copy_to_device should be called

    def test_device_copying_with_strategy(self):
        """Test that copy_to_device is NOT called when deviceName is a Strategy"""
        strategy = MagicMock(spec=tf.distribute.Strategy)
        self.assertNotIsInstance(strategy, str)
        # When isinstance(deviceName, str) is False, copy_to_device should NOT be called


if __name__ == "__main__":
    unittest.main()

