### Fixtures pytest https://docs.pytest.org/en/6.2.x/fixture.html

import pytest
import pathlib, sys, shutil
import numpy as np
from sklearn.preprocessing import OneHotEncoder

rootFolder = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(rootFolder))
import simpleBayes.decode_bayes as plg
from utils.global_classes import Project


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

    def test_find_speed_filtered_spikes_in_epoch_tetrodewise(self):
        # Create a speed vector
        speedMask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        times = np.arange(1, 21)
        # Create an epoch
        epoch = np.array([15, 20])
        # Get the mask
        ids = plg.Trainer.find_speed_filtered_spikes_in_epoch(times, epoch, speedMask)
        # Check that the mask is correct
        assert np.all(ids == np.array([14, 15, 16, 17, 18, 19]))

    def test_find_speed_filtered_spikes_in_epoch_clusterwise(self):
        # Create a speed vector
        speedMask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        times = np.arange(1, 21)
        # Create an epoch
        epoch = np.array([15, 20])
        # Create spike labels
        spikeLabels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3,
                                3, 3, 4, 5, 5, 6, 7, 7, 7, 7])
        spikeLabels = OneHotEncoder().fit_transform(spikeLabels.reshape(-1, 1)).toarray()
        numCluster = 7
        # Get the mask
        ids = plg.Trainer.find_speed_filtered_spikes_in_epoch(times, epoch, speedMask,
                                                              spikeLabels=spikeLabels,
                                                              numCluster=numCluster)
        # Check that the mask is correct
        assert np.all(ids == np.array([16, 17, 18, 19]))

    def test_find_speed_filtered_spikes_in_epoch_gives_value_error(self):
        # Create a speed vector
        speedMask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        times = np.arange(1, 21)
        # Create an epoch
        epoch = np.array([15, 20])
        # Create spike labels
        spikeLabels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3,
                                3, 3, 4, 5, 5, 6, 7, 7, 7, 7])
        spikeLabels = OneHotEncoder().fit_transform(spikeLabels.reshape(-1, 1)).toarray()
        with pytest.raises(ValueError):
            plg.Trainer.find_speed_filtered_spikes_in_epoch(times, epoch, speedMask,
                                                              spikeLabels=spikeLabels)


class TestKernelMaps:
    # TODO: Implement tests for kernel maps
    pass


# This does not work. I should learn how to do teardown in pytest
@pytest.fixture(scope='class')
def setup_trainer(request):
    path = Project(str(pathlib.Path(__file__).parent.absolute().joinpath('filesForTest', 'test.xml')))
    request.cls.trainer = plg.Trainer(path) # TODO: Do not forget to remove path
    request.cls.trainer.clusterData = dict()
    request.cls.trainer.clusterData['trainEpochs'] = np.array([[20, 30, 40, 75]]).T
    request.cls.trainer.clusterData['Spike_positions'] = []
    request.cls.trainer.clusterData['Spike_times'] = []
    for i in range(3):
        request.cls.trainer.clusterData['Spike_positions'].append(np.random.random((100, 2)))
        spikeTimes = np.arange(100) + np.random.random(1)
        spikeTimes = spikeTimes[:, np.newaxis]
        request.cls.trainer.clusterData['Spike_times'].append(spikeTimes)
    yield path
    shutil.rmtree(path.dataPath)
    shutil.rmtree(path.resultsPath)

@pytest.mark.usefixtures('setup_trainer')
class TestFullBayes:  
    # TODO: Implement tests for epochs

    def test_get_spike_pos_for_use_tetrodewise(self):
        ### Parameters
        # Speed mask
        speedMask = np.zeros(100)
        speedMask[50:] += 1
        # Epoch
        epoch = self.trainer.clusterData['trainEpochs']
        #numSpikeGroup
        numSpikeGroup = 1

        spikesInEpoch = self.trainer.get_spike_pos_for_use(epoch, numSpikeGroup,
                                                                    speedMask)

        assert len(spikesInEpoch) == 25 # From 50 to 75
        assert (self.trainer.clusterData['Spike_positions'][numSpikeGroup][50:75] == \
                spikesInEpoch).all()

    #TODO:test_get_spike_pos_for_use_clusterwise

    
