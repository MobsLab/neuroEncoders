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
    request.cls.trainer.clusterData['Spike_labels'] = []
    num_cells_in_tetrode = 2
    request.cls.samples = [100, 80, 120] # number of samples per tetrode
    samples = request.cls.samples
    for i in range(3):
        # Mock spike positions
        request.cls.trainer.clusterData['Spike_positions'].append(np.random.random((samples[i], 2)))
        # Mock spike times
        spikeTimes = np.arange(samples[i]) + np.random.random(1)
        spikeTimes = spikeTimes[:, np.newaxis]
        request.cls.trainer.clusterData['Spike_times'].append(spikeTimes)
        # Mock spike labels
        spikeLabels = np.zeros((samples[i], num_cells_in_tetrode))
        spikeLabels[:samples[i]//2, 0] += 1
        spikeLabels[samples[i]//2:, 1] += 1
        request.cls.trainer.clusterData['Spike_labels'].append(spikeLabels)
    # Remove created folder
    yield path
    shutil.rmtree(path.dataPath)
    shutil.rmtree(path.resultsPath)

@pytest.mark.usefixtures('setup_trainer')
class TestFullBayes:  
    # TODO: Implement tests for epochs

    def test_get_spike_pos_for_use_tetrodewise(self):
        """
        This test assumes that in the first tetrode there are two cells that
        form a 100 samples long spike train. There is also a train epoch that
        lasts from 20 to 30 and from 40 to 75 samples. And speed mask that take
        all samples from 50 to the end is given.

        One needs to find the spike positions from the first tetrode group that
        belong to the train epoch and satisfy the speed mask. The result should
        be the 25 samples from the start of the speed mask (20 to 30 samples are
        out because they are not in the speed mask) to 75th samples because that
        is where train epoch ends.
        """
        # Epoch
        epoch = self.trainer.clusterData['trainEpochs']
        #numSpikeGroup
        numSpikeGroup = 0
        # Speed mask
        speedMask = np.zeros(self.samples[numSpikeGroup])
        speedMask[50:] += 1

        spikesInEpoch = self.trainer.get_spike_pos_for_use(epoch, numSpikeGroup,
                                                                    speedMask)

        assert len(spikesInEpoch) == 25 # From 50 to 75
        assert (self.trainer.clusterData['Spike_positions'][numSpikeGroup][50:75] == \
                spikesInEpoch).all()
        
    def test_get_spike_pos_for_use_clusterwise(self):
        """
        This test assumes that in the third tetrode there are two cells that
        form a 120 samples long spike train. The first 60 samples belong to
        the first cell and the last 60 samples belong to the second cell.
        There is also a train epoch that lasts from 20 to 30 and from 40 to 75
        samples. And speed mask that take all samples from 50 to the end is given.

        One needs to find the spike positions from the second cluster of the third
        tetrode group that belong to the train epoch and satisfy the speed mask.
        The result should be the 15 samples from the start of the spike label
        mask (20 to 30 samples are out because they are not in the speed mask,
        and 50 to 60 is out because they don't belong to the second cluster) to
        75th samples because that is where train epoch ends.
        """
        # Epoch
        epoch = self.trainer.clusterData['trainEpochs']
        #numSpikeGroup
        numSpikeGroup = 2
        numCluster = 1

        # Speed mask
        speedMask = np.zeros(self.samples[numSpikeGroup])
        speedMask[50:] += 1

        spikesInEpoch = self.trainer.get_spike_pos_for_use(epoch, numSpikeGroup,
                                                           speedMask,
                                                           numCluster=numCluster)

        assert len(spikesInEpoch) == 15 # From 60 to 75
        assert (self.trainer.clusterData['Spike_positions'][numSpikeGroup][60:75] == \
                spikesInEpoch).all()

    
