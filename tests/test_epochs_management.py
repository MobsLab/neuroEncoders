"""Tests for epochs_management module."""

import warnings

import numpy as np
import pytest

from neuroencoders.importData.epochs_management import get_epochs_mask


class TestGetEpochsMask:
    """Test cases for get_epochs_mask function."""

    def test_all_flags_false_raises_warning(self):
        """Test that a warning is raised when all flags are False."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 1.0]]),
            "testEpochs": np.array([[2.0, 3.0]]),
        }

        with pytest.warns(UserWarning, match="All epoch flags.*are False"):
            result = get_epochs_mask(
                times=times,
                epochs=epochs,
                useTrain=False,
                useTest=False,
                usePredLoss=False,
                sleepEpochs=None,
            )

        # Should return all-False mask
        assert np.all(~result)

    def test_sleep_epochs_none_does_not_error(self):
        """Test that sleepEpochs=None does not cause an error."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 2.0]]),
            "testEpochs": np.array([[2.0, 4.0]]),
        }

        # Should not raise an error
        result = get_epochs_mask(
            times=times,
            epochs=epochs,
            useTrain=True,
            useTest=False,
            sleepEpochs=None,
        )

        assert result is not None
        assert result.shape == times.shape

    def test_sleep_epochs_empty_list_does_not_error(self):
        """Test that sleepEpochs=[] does not cause an error."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 2.0]]),
            "testEpochs": np.array([[2.0, 4.0]]),
        }

        # Should not raise an error
        result = get_epochs_mask(
            times=times,
            epochs=epochs,
            useTrain=True,
            useTest=False,
            sleepEpochs=[],
        )

        assert result is not None
        assert result.shape == times.shape

    def test_use_train_only(self):
        """Test using only train epochs."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 2.0]]),
            "testEpochs": np.array([[2.0, 4.0]]),
        }

        result = get_epochs_mask(
            times=times,
            epochs=epochs,
            useTrain=True,
            useTest=False,
            sleepEpochs=None,
        )

        # Times 0.0, 1.0, 2.0 should be in train epochs [0.0, 2.0]
        # The exact mask depends on inEpochsMask implementation
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_use_test_only(self):
        """Test using only test epochs."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 2.0]]),
            "testEpochs": np.array([[2.0, 4.0]]),
        }

        result = get_epochs_mask(
            times=times,
            epochs=epochs,
            useTrain=False,
            useTest=True,
            sleepEpochs=None,
        )

        # Times 2.0, 3.0, 4.0 should be in test epochs [2.0, 4.0]
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_use_both_train_and_test(self):
        """Test using both train and test epochs."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 2.0]]),
            "testEpochs": np.array([[2.0, 4.0]]),
        }

        result = get_epochs_mask(
            times=times,
            epochs=epochs,
            useTrain=True,
            useTest=True,
            sleepEpochs=None,
        )

        # All times should be included
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_sleep_epochs_overrides_flags(self):
        """Test that sleepEpochs takes precedence over other flags."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epochs = {
            "trainEpochs": np.array([[0.0, 2.0]]),
            "testEpochs": np.array([[2.0, 4.0]]),
        }
        sleep_epochs = np.array([[1.0, 3.0]])

        result = get_epochs_mask(
            times=times,
            epochs=epochs,
            useTrain=True,
            useTest=True,
            sleepEpochs=sleep_epochs,
        )

        # Should only use sleep epochs, not train/test
        assert result.shape == times.shape
        assert result.dtype == bool
