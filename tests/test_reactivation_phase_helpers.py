from neuroencoders.utils.MOBS_Functions import Mouse_Results, _normalize_phase_name


class DummyResults:
    def __init__(self):
        self.pre = "pre_test"
        self.preMask = "pre_mask"
        self.hab = "hab"
        self.habMask = "hab_mask"
        self.cond = "cond"
        self.condMask = "cond_mask"
        self.post = "post_test"
        self.postMask = "post_mask"
        self.training = "training"
        self.trainMask = "train_mask"
        self.sleep = "sleep"
        self.sleepMask = "sleep_mask"
        self.presleep = "pre_sleep"
        self.presleepMask = "pre_sleep_mask"
        self.postsleep = "post_sleep"
        self.postsleepMask = "post_sleep_mask"


def test_normalize_phase_name_supports_test_and_sleep_aliases():
    assert _normalize_phase_name("pre_test") == "pre_test"
    assert _normalize_phase_name("test_pre") == "pre_test"
    assert _normalize_phase_name("pre-test") == "pre_test"
    assert _normalize_phase_name("post_test") == "post_test"
    assert _normalize_phase_name("test_post") == "post_test"
    assert _normalize_phase_name("postsleep") == "post_sleep"
    assert _normalize_phase_name("pre-sleep") == "pre_sleep"


def test_get_epoch_interval_resolves_test_and_sleep_aliases():
    dummy = DummyResults()
    assert Mouse_Results.get_epoch_interval(dummy, "pre_test")[0] == "pre_test"
    assert Mouse_Results.get_epoch_interval(dummy, "test_pre")[0] == "pre_test"
    assert Mouse_Results.get_epoch_interval(dummy, "post_test")[0] == "post_test"
    assert Mouse_Results.get_epoch_interval(dummy, "test_post")[0] == "post_test"
    assert Mouse_Results.get_epoch_interval(dummy, "pre_sleep")[0] == "pre_sleep"
    assert Mouse_Results.get_epoch_interval(dummy, "postsleep")[0] == "post_sleep"
