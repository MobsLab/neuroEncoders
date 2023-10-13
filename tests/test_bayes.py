import pytest
import pathlib, sys
import numpy as np

rootFolder = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(rootFolder))
import simpleBayes.decode_bayes as plg


class TestPyKeops:

    def test_align_denser_with_sparser(self):
        sparser = np.random.rand(100)
        denser = np.random.rand(5000)
        ids = plg.Trainer.align_denser_with_sparser(denser, sparser)

        u, c = np.unique(ids, return_counts=True)
        dup = u[c > 1]
        # Check dimensionality
        assert ids.shape == denser.shape
        # Check that values repeat
        assert len(dup) > 0


class TestEpochManipulations:

    def test_get_speed_filtered_mask_in_epoch(self):
        behaviorData = dict()
        behaviorData['Times'] = dict()
        # Create a speed vector
        behaviorData['Times']['speedFilter'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        behaviorData['positionTime'] = np.arange(1, 21)[:, np.newaxis]
        # Create an epoch
        behaviorData['Times']['trainEpochs'] = np.array([15, 20])
        # Get the mask
        ids = plg.Trainer.get_speed_filtered_mask_in_epoch(behaviorData, 'trainEpochs')
        # Check that the mask is correct
        assert np.all(ids == np.array([14, 15, 16, 17, 18, 19]))