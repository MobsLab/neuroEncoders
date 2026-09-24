import numpy as np
import pandas as pd
from pynapple import IntervalSet, Tsd

from neuroencoders.resultAnalysis.ephys_sleep_analysis import (
    SleepAnalysisConfig,
    SleepEphysAnalyser,
)


class DummyDataHelper:
    def __init__(self):
        self.sleep_scoring = {
            "rem_epochs": IntervalSet(start=[30.0], end=[40.0], time_units="s"),
            "sws_epochs": IntervalSet(
                start=[10.0, 50.0], end=[20.0, 60.0], time_units="s"
            ),
            "wake_epochs": IntervalSet(
                start=[0.0, 20.0], end=[10.0, 30.0], time_units="s"
            ),
        }
        self.fullBehavior = {
            "Times": {
                "tRipples": np.array([2.0, 6.0, 12.0, 16.0, 35.0, 52.0, 58.0]),
                "sleepNames": [],
            }
        }

    def get_mov_epochs(self):
        return IntervalSet(start=[0.0], end=[30.0], time_units="s")


class DummyProject:
    def __init__(self):
        self.experimentPath = "/tmp/non_existing_sleep_project"


class DummyMouseResults:
    def __init__(self):
        self.DataHelper = DummyDataHelper()
        self.timeWindows = [100]
        self.projectPath = DummyProject()

        # Minimal in-memory model outputs across phases
        phase_times = np.linspace(0.0, 60.0, 61)
        self.resultsNN_phase = {
            "_pre": {
                "times": [phase_times],
                "Hn": [np.linspace(1.0, 2.0, phase_times.size)],
                "maxp": [np.linspace(0.9, 0.6, phase_times.size)],
                "predLoss": [np.linspace(1.0, 2.0, phase_times.size)],
            }
        }
        # latent_output in pkl-like storage to exercise derived latent metrics
        latent = np.vstack(
            [
                np.linspace(0.0, 1.0, phase_times.size),
                np.linspace(1.0, 0.0, phase_times.size),
            ]
        ).T
        self.resultsNN_phase_pkl = {
            "_pre": {
                "latent_output_pooled": [latent],
                "maxp": [np.linspace(0.9, 0.6, phase_times.size)],
                "Hn": [np.linspace(1.0, 2.0, phase_times.size)],
            }
        }

    def add_sleep_scoring(self, force=False):
        return self.DataHelper.sleep_scoring

    def get_epoch_interval(self, phase):
        if phase == "pre_sleep":
            return IntervalSet(start=[0.0], end=[30.0], time_units="s"), None
        if phase == "post_sleep":
            return IntervalSet(start=[30.0], end=[60.0], time_units="s"), None
        return IntervalSet(start=[0.0], end=[60.0], time_units="s"), None


def test_analyze_ripples_by_state_summary_has_expected_columns():
    analyzer = SleepEphysAnalyser(
        SleepAnalysisConfig(transition_window_sec=20.0, bin_size_sec=5.0)
    )
    mouse = DummyMouseResults()

    states = analyzer.get_sleep_state_intervals(mouse)
    out = analyzer.analyze_ripples_by_state(mouse, states)

    assert isinstance(out["summary"], pd.DataFrame)
    assert set(["state", "n_ripples", "duration_sec", "ripple_rate_hz"]).issubset(
        out["summary"].columns
    )
    assert set(out["summary"]["state"].values) == {"rem", "nrem", "wake"}


def test_extract_model_metric_supports_derived_metrics():
    analyzer = SleepEphysAnalyser()
    mouse = DummyMouseResults()

    tsds = analyzer._extract_model_metric_tsd(
        mouse_results=mouse,
        winMS=100,
        metric_key=["Hn", "maxp", "certainty", "surprisal", "latent_l2", "latent_var"],
    )

    assert tsds["Hn"] is not None
    assert tsds["maxp"] is not None
    assert tsds["certainty"] is not None
    assert tsds["surprisal"] is not None
    assert tsds["latent_l2"] is not None
    assert tsds["latent_var"] is not None


def test_analyze_mouse_returns_multi_model_outputs_and_drowsiness():
    analyzer = SleepEphysAnalyser(
        SleepAnalysisConfig(drowsiness_window_sec=30.0, bin_size_sec=5.0)
    )
    mouse = DummyMouseResults()

    # simple reactivation trace
    t = np.linspace(0.0, 60.0, 121)
    r = np.sin(t / 10.0)
    reactivation_tsd = Tsd(t=t, d=r)

    out = analyzer.analyze_mouse(
        mouse_results=mouse,
        reactivation_tsd=reactivation_tsd,
        winMS=100,
        model_metric_key=["Hn", "latent_l2"],
    )

    assert "model" in out
    assert "Hn" in out["model"]
    assert "latent_l2" in out["model"]
    assert "drowsiness" in out
    assert "slopes" in out["drowsiness"]


def test_plotting_helpers_run_without_error(tmp_path):
    analyzer = SleepEphysAnalyser()
    summary_df = pd.DataFrame(
        {
            "state": ["wake", "nrem", "rem", "wake", "nrem", "rem"],
            "mean": [0.2, 0.4, 0.3, 0.25, 0.45, 0.35],
        }
    )

    fig1, _ = analyzer.plot_state_summary(
        summary_df=summary_df,
        value_col="mean",
        title="State summary",
        save_path=str(tmp_path / "state_summary.png"),
    )
    assert fig1 is not None

    transitions = {
        "wake": {
            "time_sec": np.array([-5, 0, 5]),
            "mean": np.array([0.1, 0.2, 0.1]),
            "sem": np.array([0.01, 0.01, 0.01]),
        },
        "nrem": {
            "time_sec": np.array([-5, 0, 5]),
            "mean": np.array([0.2, 0.3, 0.25]),
            "sem": np.array([0.02, 0.02, 0.02]),
        },
        "rem": {
            "time_sec": np.array([-5, 0, 5]),
            "mean": np.array([0.15, 0.18, 0.16]),
            "sem": np.array([0.015, 0.015, 0.015]),
        },
    }

    fig2, _ = analyzer.plot_transition_profiles(
        transition_profiles=transitions,
        value_key="mean",
        title="Transitions",
        ylabel="Signal",
        save_path=str(tmp_path / "transitions.png"),
    )
    assert fig2 is not None

    drowsiness = {
        "bin_centers_sec": np.array([-20, -10, 0]),
        "ripple_rate_curve_hz": np.array([0.1, 0.2, 0.3]),
        "reactivation_curve": np.array([0.4, 0.5, 0.6]),
        "model_curve": np.array([0.7, 0.8, 0.9]),
    }
    fig3, _ = analyzer.plot_drowsiness_curves(
        drowsiness=drowsiness,
        save_path=str(tmp_path / "drowsiness.png"),
    )
    assert fig3 is not None
