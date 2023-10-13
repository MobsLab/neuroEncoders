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