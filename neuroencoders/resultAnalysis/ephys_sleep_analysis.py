"""Sleep-focused ephys/model analysis utilities.

This module adds a reusable analyzer that can be used from both Any
and Any to quantify:
- Ripple dynamics across REM/NREM/Wake
- Reactivation strength across REM/NREM/Wake
- Transition dynamics around sleep-state onsets
- Drowsiness trends before NREM onset

The analyzer is intentionally robust to partially missing signals and returns
structured dictionaries + DataFrames for downstream plotting/statistics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynapple import IntervalSet, TsGroup, Tsd, TsdFrame
from scipy.signal import hilbert
from sklearn.decomposition import PCA


@dataclass
class SleepAnalysisConfig:
    transition_window_sec: float = 120.0
    bin_size_sec: float = 5.0
    psth_bin_sec: float = 0.02  # 20 ms bins for SWR PSTHs
    psth_window_sec: float = 1.0  # [-1s, +1s] around ripples
    drowsiness_window_sec: float = 180.0
    min_samples_for_stats: int = 5


class SleepEphysAnalyzer:
    """Sleep dynamics analyzer for ephys cell types and 128D model latent manifolds."""

    def __init__(self, config: Optional[SleepAnalysisConfig] = None):
        self.config = config or SleepAnalysisConfig()

    # =========================================================================
    # 1. CELL TYPE PARSING & SPIKE PROCESSING
    # =========================================================================

    @staticmethod
    def parse_cell_types(
        neuron_classifications: Iterable[Any],
    ) -> Dict[str, np.ndarray]:
        """Categorize neuron classifications array into boolean masks."""
        raw_types = np.asarray([str(x).lower() for x in neuron_classifications])

        is_pyr = np.array(
            ["pyramidal" in t and "sua" in t for t in raw_types], dtype=bool
        )
        is_int = np.array(
            ["interneuron" in t and "sua" in t for t in raw_types], dtype=bool
        )
        is_mua = np.array(["mua" in t for t in raw_types], dtype=bool)
        is_unclass = np.array(
            ["unclassified" in t and "sua" in t for t in raw_types], dtype=bool
        )

        # Fallback if no specific tags matched
        if not np.any(is_pyr) and not np.any(is_int):
            is_pyr = np.ones(raw_types.shape[0], dtype=bool)

        return {
            "pyr": is_pyr,
            "int": is_int,
            "mua": is_mua,
            "unclassified": is_unclass,
            "all_sua": is_pyr | is_int | is_unclass,
        }

    def get_cell_type_spike_rates(
        self,
        spike_group: TsGroup,
        neuron_classifications: Iterable[Any],
        bin_size_sec: float = 0.1,
    ) -> Dict[str, Optional[TsdFrame]]:
        """Bin Pynapple TsGroup spikes into TsdFrame rate time-series split by cell type."""
        if spike_group is None or len(spike_group) == 0:
            return {"pyr": None, "int": None, "mua": None, "all": None}

        cell_masks = self.parse_cell_types(neuron_classifications)
        binned = spike_group.count(bin_size_sec)
        # Convert counts to Hz
        rate_frame = TsdFrame(t=binned.index, d=binned.values / bin_size_sec)

        out: Dict[str, Optional[TsdFrame]] = {}
        for ctype, mask in cell_masks.items():
            if np.any(mask) and mask.shape[0] == rate_frame.shape[1]:
                out[ctype] = TsdFrame(t=rate_frame.index, d=rate_frame.values[:, mask])
            else:
                out[ctype] = None

        out["all"] = rate_frame
        return out

    # =========================================================================
    # 2. SWR PSTH ANALYSIS BY CELL TYPE
    # =========================================================================

    def compute_swr_psth_by_celltype(
        self,
        mouse_results: Union[Any, Any],
        state_intervals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute Peri-SWR PSTHs for Pyramidal, Interneuron, and MUA populations."""
        ripple_times = mouse_results.DataHelper.fullBehavior["Times"].get(
            "tRipples", None
        )
        spike_group = getattr(
            mouse_results.DataHelper, "get_spike_data", lambda: None
        )()

        if ripple_times is None or spike_group is None:
            return {"psth_time": np.array([]), "profiles": {}}

        try:
            nclass = mouse_results.DataHelper.get_neuron_classifications()
        except Exception:
            nclass = np.array(["SUA_pyramidal"] * len(spike_group))

        ripple_times = np.asarray(ripple_times, dtype=float).flatten()
        ripple_times = ripple_times[np.isfinite(ripple_times)]

        # Restrict SWRs to NREM epoch if available
        nrem_ep = state_intervals.get("nrem", None)
        if nrem_ep is not None and len(ripple_times) > 0:
            mask = self._mask_times(ripple_times, nrem_ep)
            ripple_times = ripple_times[mask]

        bins = np.arange(
            -self.config.psth_window_sec,
            self.config.psth_window_sec + self.config.psth_bin_sec,
            self.config.psth_bin_sec,
        )
        centers = 0.5 * (bins[:-1] + bins[1:])

        cell_masks = self.parse_cell_types(nclass)
        profiles: Dict[str, Dict[str, np.ndarray]] = {}

        for ctype, mask in cell_masks.items():
            if not np.any(mask) or len(ripple_times) == 0:
                continue

            sub_group = TsGroup({i: spike_group[i] for i in np.where(mask)[0]})
            # Pynapple SWR-triggered average firing rate matrix
            psth_per_unit = []
            for unit_id in sub_group.keys():
                st = sub_group[unit_id].index
                unit_traces = []
                for rip in ripple_times:
                    dt = st - rip
                    keep = (dt >= -self.config.psth_window_sec) & (
                        dt <= self.config.psth_window_sec
                    )
                    counts, _ = np.histogram(dt[keep], bins=bins)
                    unit_traces.append(counts / self.config.psth_bin_sec)
                if unit_traces:
                    psth_per_unit.append(np.mean(unit_traces, axis=0))

            if psth_per_unit:
                arr = np.array(psth_per_unit)  # (n_units, n_bins)
                profiles[ctype] = {
                    "mean": np.mean(arr, axis=0),
                    "sem": np.std(arr, axis=0) / np.sqrt(max(1, arr.shape[0])),
                    "n_units": arr.shape[0],
                }

        return {"psth_time": centers, "profiles": profiles}

    # =========================================================================
    # 3. MANIFOLD PROJECTIONS & LATENT DYNAMICS
    # =========================================================================

    def analyze_latent_manifold_projections(
        self,
        mouse_results: Union[Any, Any],
        winMS: int = 100,
        n_components: int = 3,
    ) -> Dict[str, Any]:
        """Project 128D latent_output onto global PCA axes and compute manifold metrics."""
        # Extract full latent series across phases
        model_data = self._extract_model_metric_tsd(
            mouse_results=mouse_results,
            winMS=winMS,
            metric_key=["latent_output", "Hn"],
        )

        print(model_data)
        latent_tsd = model_data.get("latent_output", None)
        if latent_tsd is None or latent_tsd.values.ndim < 2:
            return {"pca_model": None, "projected_states": {}, "velocity_tsd": None}

        states = self.get_sleep_state_intervals(mouse_results=mouse_results)
        z_all = np.asarray(latent_tsd.values, dtype=float)
        ts_all = np.asarray(latent_tsd.index, dtype=float)

        # Fit global PCA on Wake + NREM + REM valid latent vectors
        pca = PCA(n_components=min(n_components, z_all.shape[1]))
        pca.fit(z_all)

        projected_states = {}
        for sname in ("wake", "nrem", "rem"):
            print(f"Projecting latent manifold for state: {sname}")
            sep = states.get(sname, None)
            print(sep)
            if sep is not None:
                try:
                    z_state = latent_tsd.restrict(sep)
                    if len(z_state) > 0:
                        projected_states[sname] = {
                            "times": np.asarray(z_state.index, dtype=float),
                            "proj": pca.transform(
                                np.asarray(z_state.values, dtype=float)
                            ),
                        }
                except Exception:
                    pass

        # Compute trajectory velocity dz/dt in 128D space
        dt = np.diff(ts_all)
        dt[dt <= 0] = np.nan
        dz = np.linalg.norm(np.diff(z_all, axis=0), axis=1) / dt
        velocity_tsd = self._to_tsd(ts_all[1:], dz)

        return {
            "pca_model": pca,
            "projected_states": projected_states,
            "velocity_tsd": velocity_tsd,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

    # =========================================================================
    # 4. PUBLICATION-READY PLOTTING FUNCTIONS
    # =========================================================================

    def plot_celltype_swr_psths(
        self,
        swr_psth_data: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Plot Peri-SWR PSTH curves split by cell class (Pyramidal, Interneuron, MUA)."""
        centers = swr_psth_data.get("psth_time", np.array([]))
        profiles = swr_psth_data.get("profiles", {})
        if centers.size == 0 or not profiles:
            return None, None

        colors = {
            "pyr": "#2C7FB8",
            "int": "#D7191C",
            "mua": "#7F7F7F",
            "unclassified": "#998EC3",
        }
        labels = {
            "pyr": "Pyramidal (SUA)",
            "int": "Interneuron (SUA)",
            "mua": "MUA",
            "unclassified": "Unclassified",
        }

        fig, ax = plt.subplots(figsize=(7, 4))
        for ctype, prof in profiles.items():
            mean = prof["mean"]
            sem = prof["sem"]
            c = colors.get(ctype, "black")
            lab = f"{labels.get(ctype, ctype)} (n={prof['n_units']})"

            ax.plot(centers, mean, color=c, linewidth=2.0, label=lab)
            ax.fill_between(centers, mean - sem, mean + sem, color=c, alpha=0.2)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title("Peri-SWR Population PSTH by Cell Type")
        ax.set_xlabel("Time from Ripple Peak (s)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_latent_manifold_3d(
        self,
        manifold_data: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Plot 3D PCA projection of the 128D latent_output colored by sleep state."""
        proj_states = manifold_data.get("projected_states", {})
        if not proj_states:
            return None, None

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        palette = {"wake": "#7F7F7F", "nrem": "#2C7FB8", "rem": "#D7191C"}

        for sname in ["wake", "nrem", "rem"]:
            if sname not in proj_states:
                continue
            P = proj_states[sname]["proj"]
            if P.shape[1] < 3:
                continue
            # Downsample for clear rendering
            idx = np.random.choice(
                P.shape[0], size=min(1500, P.shape[0]), replace=False
            )
            ax.scatter(
                P[idx, 0],
                P[idx, 1],
                P[idx, 2],
                c=palette[sname],
                label=sname.upper(),
                alpha=0.4,
                s=8,
                edgecolor="none",
            )

        var_ratio = manifold_data.get("explained_variance_ratio", [0, 0, 0])
        ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")
        ax.set_zlabel(f"PC3 ({var_ratio[2] * 100:.1f}%)")
        ax.set_title("128D Model Latent Manifold Topography")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================

    @staticmethod
    def _safe_interval_df(interval: Any) -> pd.DataFrame:
        if interval is None:
            return pd.DataFrame(columns=["start", "end"])
        try:
            df = interval.as_units("s")
            if isinstance(df, pd.DataFrame):
                return df
            if hasattr(df, "values") and len(df) > 0:
                arr = np.asarray(df.values, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return pd.DataFrame({"start": arr[:, 0], "end": arr[:, 1]})
            if hasattr(df, "iloc"):
                starts = np.asarray(df.iloc[:, 0], dtype=float)
                ends = np.asarray(df.iloc[:, 1], dtype=float)
                return pd.DataFrame({"start": starts, "end": ends})
        except Exception:
            pass

        try:
            arr = np.asarray(interval, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return pd.DataFrame({"start": arr[:, 0], "end": arr[:, 1]})
        except Exception:
            pass

        return pd.DataFrame(columns=["start", "end"])

    @classmethod
    def _interval_duration_sec(cls, interval: Any) -> float:
        df = cls._safe_interval_df(interval)
        if df.empty:
            return 0.0
        dur = np.asarray(df["end"], dtype=float) - np.asarray(df["start"], dtype=float)
        return float(np.nansum(np.clip(dur, a_min=0.0, a_max=None)))

    @classmethod
    def _interval_starts(cls, interval: Any) -> np.ndarray:
        df = cls._safe_interval_df(interval)
        if df.empty:
            return np.array([], dtype=float)
        return np.asarray(df["start"], dtype=float)

    @staticmethod
    def _interval_intersection(a: Any, b: Any) -> Optional[IntervalSet]:
        if a is None or b is None:
            return None
        try:
            return a.intersect(b)
        except Exception:
            return None

    @staticmethod
    def _interval_union(a: Any, b: Any) -> Any:
        if a is None:
            return b
        if b is None:
            return a
        try:
            return a.union(b)
        except Exception:
            return a

    @staticmethod
    def _mask_times(times: np.ndarray, interval: Any) -> np.ndarray:
        df = SleepEphysAnalyzer._safe_interval_df(interval)
        if df.empty or times.size == 0:
            return np.zeros(times.shape[0], dtype=bool)

        mask = np.zeros(times.shape[0], dtype=bool)
        for start, end in zip(df["start"].to_numpy(), df["end"].to_numpy()):
            mask |= (times >= start) & (times <= end)
        return mask

    @staticmethod
    def _to_tsd(times: np.ndarray, values: np.ndarray) -> Optional[Tsd]:
        if times is None or values is None:
            return None
        times = np.asarray(times).flatten()
        values = np.asarray(values).flatten()
        keep = np.isfinite(times) & np.isfinite(values)
        if np.sum(keep) < 2:
            return None

        t = times[keep]
        d = values[keep]
        order = np.argsort(t)
        return Tsd(t=t[order], d=d[order])

    @staticmethod
    def extract_accelerometer_tsd(
        mouse_results: Union[Any, Any],
    ) -> Optional[Tsd]:
        """Extract smoothed accelerometer movement acceleration as a Pynapple Tsd."""
        try:
            times = np.asarray(
                mouse_results.DataHelper.fullBehavior["MovTimes"]
            ).flatten()
            acc = np.asarray(mouse_results.DataHelper.fullBehavior["MovAcc"]).flatten()
            keep = np.isfinite(times) & np.isfinite(acc)
            if np.sum(keep) < 2:
                return None
            order = np.argsort(times[keep])
            return Tsd(t=times[keep][order], d=acc[keep][order])
        except (KeyError, AttributeError):
            return None

    @staticmethod
    def extract_lfp_signals(
        mouse_results: Union[Any, Any],
    ) -> Dict[str, Optional[Tsd]]:
        """Extract key canonical LFP channels (Theta, Delta, Ripple) using DataHelper aliases."""
        data_helper = getattr(mouse_results, "DataHelper", None)
        if data_helper is None:
            return {"theta": None, "delta": None, "ripple_lfp": None}

        out = {}
        for key in ["theta", "delta", "ripple"]:
            try:
                lfp = data_helper.get_lfp_data(
                    channel_type=key,
                    network_path=getattr(mouse_results, "network_path", None),
                )
                if isinstance(lfp, dict):
                    lfp = list(lfp.values())[0] if lfp else None
                out[key if key != "ripple" else "ripple_lfp"] = lfp
            except Exception:
                out[key if key != "ripple" else "ripple_lfp"] = None

        return out

    @staticmethod
    def _ensure_metric_keys(metric_key: Union[str, Iterable[str]]) -> list[str]:
        """Ensure metric_key input is formatted as a list of strings."""
        if isinstance(metric_key, str):
            return [metric_key]
        return [str(k) for k in metric_key]

    def get_sleep_state_intervals(
        self,
        mouse_results: Union[Any, Any],
        use_sleep_union: bool = True,
    ) -> Dict[str, Any]:
        """Build canonical REM/NREM/Wake intervals for one mouse."""
        mouse_results.add_sleep_scoring()

        scoring = getattr(mouse_results.DataHelper, "sleep_scoring", {})
        rem = scoring.get("rem_epochs", None)
        nrem = scoring.get("sws_epochs", None)
        wake = (
            mouse_results.get_epoch_interval("hab")[0]
            .union(mouse_results.get_epoch_interval("pre_test")[0])
            .union(mouse_results.get_epoch_interval("cond")[0])
            .union(mouse_results.get_epoch_interval("post_test")[0])
        )
        print(wake)

        if wake is None:
            try:
                wake = mouse_results.DataHelper.get_mov_epochs()
            except Exception:
                wake = None

        pre_sleep, _ = mouse_results.get_epoch_interval("pre_sleep")
        post_sleep, _ = mouse_results.get_epoch_interval("post_sleep")
        sleep_union = self._interval_union(pre_sleep, post_sleep)

        if use_sleep_union and sleep_union is not None:
            rem = self._interval_intersection(rem, sleep_union)
            nrem = self._interval_intersection(nrem, sleep_union)
            if wake is not None:
                wake = self._interval_intersection(wake, sleep_union)

        return {"rem": rem, "nrem": nrem, "wake": wake, "sleep_union": sleep_union}

    def _extract_model_metric_tsd(
        self,
        mouse_results: Union[Any, Any],
        winMS: int,
        metric_key: Union[str, Iterable[str]] = "Hn",
    ) -> Dict[str, Optional[Union[Tsd, TsdFrame]]]:
        """Extract continuous model metrics (scalars or 128D vectors) across phases."""
        metric_keys = [metric_key] if isinstance(metric_key, str) else list(metric_key)
        times_all: Dict[str, List[np.ndarray]] = {k: [] for k in metric_keys}
        values_all: Dict[str, List[np.ndarray]] = {k: [] for k in metric_keys}

        id_window = None
        if hasattr(mouse_results, "timeWindows") and winMS in mouse_results.timeWindows:
            id_window = mouse_results.timeWindows.index(winMS)
        elif (
            hasattr(mouse_results, "windows_values")
            and winMS in mouse_results.windows_values
        ):
            id_window = mouse_results.windows_values.index(winMS)

        if id_window is None:
            return {k: None for k in metric_keys}

        for suffix in ["_pre", "_training", "_cond", "_post"]:
            phase_dict = getattr(mouse_results, "resultsNN_phase", {}).get(suffix, None)
            pkl_dict = getattr(mouse_results, "resultsNN_phase_pkl", {}).get(
                suffix, None
            )
            if phase_dict is None:
                og_suffixes = mouse_results.suffixes
                mouse_results.load_data(suffixes=[suffix], load_pickle=True, redo=True)
                phase_dict = getattr(mouse_results, "resultsNN_phase", {}).get(
                    suffix, None
                )
                pkl_dict = getattr(mouse_results, "resultsNN_phase_pkl", {}).get(
                    suffix, None
                )
                mouse_results.suffixes = og_suffixes
                if hasattr(mouse_results, "unload"):
                    mouse_results.unload()

            try:
                times = np.asarray(phase_dict["times"][id_window]).flatten()
            except Exception:
                continue

            arrays: Dict[str, Any] = {}
            if isinstance(phase_dict, dict):
                for k in ["Hn", "maxp", "predLoss"]:
                    if k in phase_dict and len(phase_dict[k]) > id_window:
                        arrays[k] = phase_dict[k][id_window]
            if isinstance(pkl_dict, dict):
                for k in ["Hn", "maxp", "predLoss", "latent_output"]:
                    if k in pkl_dict and len(pkl_dict[k]) > id_window:
                        arrays[k] = pkl_dict[k][id_window]

            for key in metric_keys:
                vals = arrays.get(key, None)
                if vals is None:
                    continue
                v_arr = np.asarray(vals, dtype=float)
                min_n = min(times.shape[0], v_arr.shape[0])
                if min_n < 2:
                    continue
                times_all[key].append(times[:min_n])
                values_all[key].append(v_arr[:min_n])

        out: Dict[str, Optional[Union[Tsd, TsdFrame]]] = {}
        for key in metric_keys:
            if not times_all[key]:
                out[key] = None
                continue
            t = np.concatenate(times_all[key])
            v = np.concatenate(values_all[key], axis=0)
            if v.ndim == 1:
                out[key] = self._to_tsd(t, v)
            else:
                order = np.argsort(t)
                out[key] = TsdFrame(t=t[order], d=v[order])

        return out

    def compute_lfp_and_motion_metrics(
        self,
        mouse_results: Union[Any, Any],
        state_intervals: Dict[str, Any],
        immobility_threshold: float = 1.7e7,
    ) -> Dict[str, Any]:
        """Compute continuous LFP power ratios, sleep pressure decay, and quiet vs. active wake."""
        acc_tsd = self.extract_accelerometer_tsd(mouse_results)
        lfp_dict = self.extract_lfp_signals(mouse_results)

        theta_lfp = lfp_dict.get("theta", None)
        delta_lfp = lfp_dict.get("delta", None)

        # 1. Continuous Theta / Delta Power Ratio
        theta_delta_ratio = None
        if theta_lfp is not None and delta_lfp is not None:
            try:
                # Re-align LFP timestamps if slightly offset
                t_common = theta_lfp.index
                d_val = np.interp(t_common, delta_lfp.index, delta_lfp.values)

                # Compute Hilbert envelope power for theta and delta bands
                p_theta = np.abs(hilbert(theta_lfp.values)) ** 2
                p_delta = np.abs(hilbert(d_val)) ** 2

                ratio = p_theta / (p_delta + 1e-12)
                theta_delta_ratio = Tsd(t=t_common, d=ratio)
            except Exception:
                pass

        # 2. Quiet Wake (QW) vs. Active Wake (AW) Partitioning
        wake_ep = state_intervals.get("wake", None)
        qw_epochs, aw_epochs = None, None

        if wake_ep is not None and acc_tsd is not None:
            try:
                acc_wake = acc_tsd.restrict(wake_ep)
                is_immobile = acc_wake.values < immobility_threshold

                # Convert boolean mask to Pynapple IntervalSets
                qw_epochs = acc_wake[is_immobile].time_support
                aw_epochs = acc_wake[~is_immobile].time_support
            except Exception:
                pass

        # 3. NREM Sleep Pressure (Delta Power Envelope during Slow-Wave Sleep)
        delta_power_nrem = None
        nrem_ep = state_intervals.get("nrem", None)
        if delta_lfp is not None and nrem_ep is not None:
            try:
                delta_nrem = delta_lfp.restrict(nrem_ep)
                delta_env = np.abs(hilbert(delta_nrem.values)) ** 2
                delta_power_nrem = Tsd(t=delta_nrem.index, d=delta_env)
            except Exception:
                pass

        return {
            "acc_tsd": acc_tsd,
            "theta_delta_ratio": theta_delta_ratio,
            "delta_power_nrem": delta_power_nrem,
            "qw_epochs": qw_epochs,
            "aw_epochs": aw_epochs,
        }

    def analyze_drowsiness_multi_signal(
        self,
        ripple_times: np.ndarray,
        reactivation_tsd: Optional[Tsd],
        state_intervals: Dict[str, Any],
        acc_tsd: Optional[Tsd] = None,
        model_tsd: Optional[Tsd] = None,
        theta_delta_tsd: Optional[Tsd] = None,
    ) -> Dict[str, Any]:
        """Quantify multi-signal trends (Motion, LFP, Ripples, Reactivation, Model) prior to NREM onset."""
        nrem_onsets = self._interval_starts(state_intervals.get("nrem", None))
        if nrem_onsets.size == 0:
            return {"n_nrem_onsets": 0, "bin_centers_sec": np.array([])}

        bins = np.arange(
            -self.config.drowsiness_window_sec,
            self.config.bin_size_sec,
            self.config.bin_size_sec,
        )
        centers = 0.5 * (bins[:-1] + bins[1:])

        def _interp_curve(tsd_signal: Optional[Tsd]) -> np.ndarray:
            if tsd_signal is None:
                return np.full_like(centers, np.nan, dtype=float)
            ts = np.asarray(tsd_signal.index, dtype=float)
            vs = np.asarray(tsd_signal.values, dtype=float)
            order = np.argsort(ts)
            ts, vs = ts[order], vs[order]

            traces = []
            for onset in nrem_onsets:
                sample_times = onset + centers
                traces.append(
                    np.interp(sample_times, ts, vs, left=np.nan, right=np.nan)
                )
            traces_arr = np.array(traces, dtype=float)
            valid_n = np.sum(np.isfinite(traces_arr), axis=0)
            return np.divide(
                np.nansum(traces_arr, axis=0),
                np.maximum(valid_n, 1),
                out=np.full_like(centers, np.nan, dtype=float),
                where=valid_n > 0,
            )

        # Pre-NREM Ripple Histogram
        ripple_hists = []
        for onset in nrem_onsets:
            dt = ripple_times - onset
            keep = (dt >= -self.config.drowsiness_window_sec) & (dt <= 0)
            hist, _ = np.histogram(dt[keep], bins=bins)
            ripple_hists.append(hist.astype(float) / self.config.bin_size_sec)

        ripple_curve = (
            np.nanmean(np.array(ripple_hists, dtype=float), axis=0)
            if ripple_hists
            else np.full_like(centers, np.nan)
        )

        return {
            "n_nrem_onsets": int(nrem_onsets.size),
            "bin_centers_sec": centers,
            "motion_curve": _interp_curve(acc_tsd),
            "ripple_rate_curve_hz": ripple_curve,
            "reactivation_curve": _interp_curve(reactivation_tsd),
            "model_curve": _interp_curve(model_tsd),
            "theta_delta_curve": _interp_curve(theta_delta_tsd),
        }

    def plot_drowsiness_multi_panel(
        self,
        drowsiness: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Plot pre-NREM transition curves for Motion, Theta/Delta Ratio, Ripples, Reactivation, and Model."""
        x = np.asarray(drowsiness.get("bin_centers_sec", []), dtype=float)
        if x.size == 0:
            return None, None

        curves = [
            (
                drowsiness.get("motion_curve", np.array([])),
                "Accelerometer Acc",
                "#7F7F7F",
            ),
            (
                drowsiness.get("theta_delta_curve", np.array([])),
                "Theta/Delta Ratio",
                "#998EC3",
            ),
            (
                drowsiness.get("ripple_rate_curve_hz", np.array([])),
                "Ripple Rate (Hz)",
                "#2C7FB8",
            ),
            (
                drowsiness.get("reactivation_curve", np.array([])),
                "Reactivation Score",
                "#D7191C",
            ),
            (
                drowsiness.get("model_curve", np.array([])),
                "Model Metric (Hn)",
                "#1A9641",
            ),
        ]

        fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
        for ax, (y, lab, color) in zip(axes, curves):
            if y.size and np.any(np.isfinite(y)):
                ax.plot(x, y, color=color, linewidth=2.0)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Time before NREM onset (s)")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def analyze_ripples_by_state(
        self,
        mouse_results: Union[Any, Any],
        state_intervals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute ripple rates by state and peri-state-onset ripple profiles."""
        ripple_times = mouse_results.DataHelper.fullBehavior["Times"].get(
            "tRipples", None
        )
        if ripple_times is None:
            return {
                "summary": pd.DataFrame(),
                "transition_profiles": {},
                "ripple_times": np.array([], dtype=float),
            }

        ripple_times = np.asarray(ripple_times).astype(float).flatten()
        ripple_times = ripple_times[np.isfinite(ripple_times)]

        rows = []
        for state_name in ("rem", "nrem", "wake"):
            state_ep = state_intervals.get(state_name, None)
            duration = self._interval_duration_sec(state_ep)
            mask = self._mask_times(ripple_times, state_ep)
            count = int(np.sum(mask))
            rate_hz = count / max(duration, 1e-9)
            rows.append(
                {
                    "state": state_name,
                    "n_ripples": count,
                    "duration_sec": duration,
                    "ripple_rate_hz": rate_hz,
                }
            )

        bins = np.arange(
            -self.config.transition_window_sec,
            self.config.transition_window_sec + self.config.bin_size_sec,
            self.config.bin_size_sec,
        )
        centers = 0.5 * (bins[:-1] + bins[1:])

        transition_profiles: Dict[str, Dict[str, np.ndarray]] = {}
        for state_name in ("rem", "nrem", "wake"):
            onsets = self._interval_starts(state_intervals.get(state_name, None))
            if onsets.size == 0:
                transition_profiles[state_name] = {
                    "time_sec": centers,
                    "ripple_rate_hz": np.full_like(centers, np.nan, dtype=float),
                    "n_onsets": 0,
                }
                continue

            dt_all = []
            for onset in onsets:
                dt = ripple_times - onset
                keep = np.abs(dt) <= self.config.transition_window_sec
                if np.any(keep):
                    dt_all.append(dt[keep])

            if not dt_all:
                hist = np.zeros(centers.shape[0], dtype=float)
            else:
                dt_concat = np.concatenate(dt_all)
                hist, _ = np.histogram(dt_concat, bins=bins)

            hist_rate = hist.astype(float) / (
                max(onsets.size, 1) * self.config.bin_size_sec
            )
            transition_profiles[state_name] = {
                "time_sec": centers,
                "ripple_rate_hz": hist_rate,
                "n_onsets": int(onsets.size),
            }

        return {
            "summary": pd.DataFrame(rows),
            "transition_profiles": transition_profiles,
            "ripple_times": ripple_times,
        }

    def _analyze_tsd_by_state(
        self,
        tsd_signal: Optional[Union[Tsd, TsdFrame]],
        state_intervals: Dict[str, Any],
        value_name: str,
    ) -> Dict[str, Any]:
        """Quantify summary statistics and state-onset transition profiles for a Tsd/TsdFrame."""
        if tsd_signal is None:
            return {
                "summary": pd.DataFrame(),
                "transition_profiles": {},
            }

        rows = []
        is_frame = isinstance(tsd_signal, TsdFrame) or (
            hasattr(tsd_signal, "values") and tsd_signal.values.ndim > 1
        )

        for state_name in ("rem", "nrem", "wake"):
            state_ep = state_intervals.get(state_name, None)
            try:
                state_vals = np.asarray(
                    tsd_signal.restrict(state_ep).values, dtype=float
                )
            except Exception:
                state_vals = np.array([], dtype=float)

            finite = state_vals[np.isfinite(state_vals)]

            # If multi-column TsdFrame, compute mean across features first or aggregate
            mean_val = np.nanmean(finite) if finite.size else np.nan
            median_val = np.nanmedian(finite) if finite.size else np.nan
            std_val = np.nanstd(finite) if finite.size else np.nan
            p95_val = np.nanpercentile(finite, 95) if finite.size else np.nan

            rows.append(
                {
                    "state": state_name,
                    "metric": value_name,
                    "n_samples": int(finite.size),
                    "mean": float(mean_val) if np.isfinite(mean_val) else np.nan,
                    "median": float(median_val) if np.isfinite(median_val) else np.nan,
                    "std": float(std_val) if np.isfinite(std_val) else np.nan,
                    "p95": float(p95_val) if np.isfinite(p95_val) else np.nan,
                }
            )

        bins = np.arange(
            -self.config.transition_window_sec,
            self.config.transition_window_sec + self.config.bin_size_sec,
            self.config.bin_size_sec,
        )
        centers = 0.5 * (bins[:-1] + bins[1:])

        ts = np.asarray(tsd_signal.index, dtype=float)
        vs = np.asarray(tsd_signal.values, dtype=float)
        if vs.ndim > 1:
            vs = np.nanmean(vs, axis=1)

        order = np.argsort(ts)
        ts, vs = ts[order], vs[order]

        transition_profiles: Dict[str, Dict[str, np.ndarray]] = {}
        for state_name in ("rem", "nrem", "wake"):
            onsets = self._interval_starts(state_intervals.get(state_name, None))
            if onsets.size == 0:
                transition_profiles[state_name] = {
                    "time_sec": centers,
                    "mean": np.full_like(centers, np.nan, dtype=float),
                    "sem": np.full_like(centers, np.nan, dtype=float),
                    "n_onsets": 0,
                }
                continue

            all_traces = []
            for onset in onsets:
                sample_times = onset + centers
                interp = np.interp(sample_times, ts, vs, left=np.nan, right=np.nan)
                all_traces.append(interp)

            traces = np.array(all_traces, dtype=float)
            valid_n = np.sum(np.isfinite(traces), axis=0)
            sum_vals = np.nansum(traces, axis=0)
            mean = np.divide(
                sum_vals,
                np.maximum(valid_n, 1),
                out=np.full_like(sum_vals, np.nan, dtype=float),
                where=valid_n > 0,
            )
            sem = np.nanstd(traces, axis=0) / np.sqrt(np.maximum(valid_n, 1))

            transition_profiles[state_name] = {
                "time_sec": centers,
                "mean": mean,
                "sem": sem,
                "n_onsets": int(onsets.size),
            }

        return {
            "summary": pd.DataFrame(rows),
            "transition_profiles": transition_profiles,
        }

    def analyze_celltype_firing_rates_by_state(
        self,
        mouse_results: Union[Any, Any],
        state_intervals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute firing rates across REM, NREM, and Wake split by cell type (Pyr, Int, MUA)."""
        spike_group = getattr(
            mouse_results.DataHelper, "get_spike_data", lambda: None
        )()
        if spike_group is None or len(spike_group) == 0:
            return {"summary": pd.DataFrame()}

        try:
            nclass = mouse_results.DataHelper.get_neuron_classifications()
        except Exception:
            nclass = np.array(["SUA_pyramidal"] * len(spike_group))

        rate_tsds = self.get_cell_type_spike_rates(
            spike_group=spike_group,
            neuron_classifications=nclass,
            bin_size_sec=1.0,
        )

        rows = []
        for ctype, frame in rate_tsds.items():
            if frame is None or ctype == "all":
                continue

            # Analyze mean rates per cell type across states
            for state_name in ("rem", "nrem", "wake"):
                state_ep = state_intervals.get(state_name, None)
                if state_ep is None:
                    continue
                try:
                    restricted_vals = frame.restrict(state_ep).values  # (time, units)
                    if restricted_vals.size > 0:
                        unit_means = np.nanmean(
                            restricted_vals, axis=0
                        )  # Mean FR per unit
                        for u_idx, fr in enumerate(unit_means):
                            rows.append(
                                {
                                    "cell_type": ctype,
                                    "state": state_name,
                                    "unit_id": u_idx,
                                    "firing_rate_hz": float(fr),
                                }
                            )
                except Exception:
                    pass

        df_fr = pd.DataFrame(rows)
        return {"summary": df_fr}

    def plot_firing_rates_by_state(
        self,
        fr_data: Dict[str, Any],
        title: str = "Cell-Type Firing Rates Across Sleep States",
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Bar plot with overlaid unit scatter points for Pyramidal, Interneuron, and MUA firing rates."""
        df_fr = fr_data.get("summary", pd.DataFrame())
        if df_fr is None or df_fr.empty:
            return None, None

        state_order = ["wake", "nrem", "rem"]
        cell_types = [
            c for c in ["pyr", "int", "mua"] if c in df_fr["cell_type"].unique()
        ]

        palette = {
            "pyr": "#2C7FB8",
            "int": "#D7191C",
            "mua": "#7F7F7F",
        }
        type_labels = {
            "pyr": "Pyramidal",
            "int": "Interneuron",
            "mua": "MUA",
        }

        fig, axes = plt.subplots(
            1, len(cell_types), figsize=(4 * len(cell_types), 4), sharey=False
        )
        if len(cell_types) == 1:
            axes = [axes]

        for ax, ctype in zip(axes, cell_types):
            sub_df = df_fr[df_fr["cell_type"] == ctype]

            means, sems, grouped_vals = [], [], []
            for st in state_order:
                vals = sub_df[sub_df["state"] == st]["firing_rate_hz"].to_numpy()
                vals = vals[np.isfinite(vals)]
                grouped_vals.append(vals)
                means.append(np.nanmean(vals) if vals.size else np.nan)
                sems.append(
                    np.nanstd(vals) / np.sqrt(max(vals.size, 1))
                    if vals.size
                    else np.nan
                )

            positions = np.arange(len(state_order))
            color = palette.get(ctype, "#333333")

            ax.bar(
                positions,
                means,
                yerr=sems,
                color=color,
                alpha=0.75,
                edgecolor="black",
                linewidth=1.0,
                capsize=4,
            )

            # Overlay unit scatter
            for i, vals in enumerate(grouped_vals):
                if vals.size:
                    jitter = np.random.uniform(-0.1, 0.1, size=vals.size)
                    ax.scatter(
                        positions[i] + jitter,
                        vals,
                        color="black",
                        alpha=0.5,
                        s=15,
                        zorder=3,
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels([s.upper() for s in state_order])
            ax.set_ylabel("Firing Rate (Hz)")
            ax.set_title(f"{type_labels.get(ctype, ctype)}")
            ax.grid(alpha=0.2, axis="y")

        fig.suptitle(title, y=1.02, fontsize=12, fontweight="bold")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axes

    # =========================================================================
    # SINGLE MOUSE ANALYSIS
    # =========================================================================

    def analyze_mouse(
        self,
        mouse_results: Union[Any, Any],
        reactivation_tsd: Optional[Tsd] = None,
        winMS: int = 100,
        model_metric_key: Union[str, Iterable[str]] = "Hn",
    ) -> Dict[str, Any]:
        """Run complete multi-signal sleep analysis for a single Union[Any, Any] instance."""
        # 1. Macro Sleep-State Epochs (REM, NREM, Wake)
        states = self.get_sleep_state_intervals(mouse_results=mouse_results)

        # 2. Ripple Dynamics & Cell-Type PSTHs
        ripple_out = self.analyze_ripples_by_state(
            mouse_results=mouse_results, state_intervals=states
        )
        swr_psth_out = self.compute_swr_psth_by_celltype(
            mouse_results=mouse_results, state_intervals=states
        )

        # 3. Cell-Type Firing Rate Dynamics Across States
        fr_out = self.analyze_celltype_firing_rates_by_state(
            mouse_results=mouse_results, state_intervals=states
        )

        # 4. Assembly Reactivation Strength Analysis
        react_out = self._analyze_tsd_by_state(
            tsd_signal=reactivation_tsd,
            state_intervals=states,
            value_name="reactivation_strength",
        )

        # 5. Extract Continuous Scalar Model Metrics
        model_metrics = self._ensure_metric_keys(model_metric_key)
        model_tsds = self._extract_model_metric_tsd(
            mouse_results=mouse_results,
            winMS=winMS,
            metric_key=model_metrics,
        )
        model_out = {}
        for mk in model_metrics:
            model_out[mk] = self._analyze_tsd_by_state(
                tsd_signal=model_tsds.get(mk, None),
                state_intervals=states,
                value_name=f"model_{mk}",
            )

        primary_model_key = model_metrics[0]
        primary_model_tsd = model_tsds.get(primary_model_key, None)

        # 6. Extract LFP Ratios, Accelerometer Motion & Quiet/Active Wake
        lfp_motion_data = self.compute_lfp_and_motion_metrics(
            mouse_results=mouse_results, state_intervals=states
        )

        # 7. 128D Latent Manifold Geometry & Trajectory Velocity
        manifold_out = self.analyze_latent_manifold_projections(
            mouse_results=mouse_results, winMS=winMS, n_components=3
        )

        # 8. Multi-Signal Pre-NREM Drowsiness Trend Alignment
        drowsiness = self.analyze_drowsiness_multi_signal(
            ripple_times=ripple_out["ripple_times"],
            reactivation_tsd=reactivation_tsd,
            state_intervals=states,
            acc_tsd=lfp_motion_data.get("acc_tsd", None),
            model_tsd=primary_model_tsd,
            theta_delta_tsd=lfp_motion_data.get("theta_delta_ratio", None),
        )

        return {
            "states": states,
            "ripple": ripple_out,
            "swr_psths": swr_psth_out,
            "firing_rates": fr_out,
            "reactivation": react_out,
            "model": model_out,
            "model_tsds": model_tsds,
            "lfp_motion": lfp_motion_data,
            "manifold": manifold_out,
            "drowsiness": drowsiness,
            "primary_model_metric": primary_model_key,
        }

    def analyze_loader(
        self,
        results_loader: Any,
        winMS: int = 100,
        rs_source: str = "spikes",
        template_period: str = "cond",
        session_data: Optional[Dict[str, Any]] = None,
        num_templates: int = 1,
        template_idx: int = 0,
        model_metric_key: Union[str, Iterable[str]] = "Hn",
        method: str = "pca_ica",
    ) -> Dict[str, Any]:
        """Run cohort-level sleep-state analysis across loaded mice.

        Parameters
        ----------
        results_loader : Any
            Host container with session DataFrames.
        winMS : int
            Binning window width in ms.
        rs_source : str
            Source for reactivation: "spikes" (ephys assembly pipeline) or "latent".
        template_period : str
            Template epoch identifier ('cond', 'wake', 'condFree', etc.).
        num_templates : int
            Number of assembly templates computed upstream.
        template_idx : int
            Target assembly index selected for downstream statistics.
        model_metric_key : str or list of str
            Model scalar metric key(s) (e.g. 'Hn', 'maxp').
        method : str
            Assembly extraction paradigm ('pca', 'pca_ica', or 'ica').
        """
        rs_source = str(rs_source).lower()
        if rs_source not in {"spikes", "latent"}:
            raise ValueError("rs_source must be 'spikes' or 'latent'.")

        # 1. Compute Assembly Reactivation Data
        if session_data is None:
            if rs_source == "spikes":
                session_data = results_loader.compute_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=max(1, num_templates),
                    method=method,
                )
            else:
                session_data = results_loader.compute_latent_reactivation(
                    winMS=winMS,
                    template_period=template_period,
                    num_templates=max(1, num_templates),
                )

        per_session: Dict[str, Dict[str, Any]] = {}
        ripple_rows = []
        react_rows = []
        model_metrics = self._ensure_metric_keys(model_metric_key)
        model_rows: Dict[str, List[pd.DataFrame]] = {k: [] for k in model_metrics}
        drowsy_rows = []

        # 2. Iterate Across Cohort Sessions
        for session_key, sdict in session_data.items():
            if session_key in {"winMS", "template", "method"}:
                continue

            results_obj: Union[Any, Any] = sdict.get("results", None)
            if results_obj is None:
                mouse_name, manipe = _parse_session_key(session_key)
                try:
                    subset = results_loader.results_df.reset_index()
                    subset = subset[
                        (subset["mouse_name"] == mouse_name)
                        & (subset["manipe"] == manipe)
                    ]
                    if subset.empty:
                        continue
                    results_obj = subset.iloc[0]["results"]
                except Exception:
                    continue

            # Extract target reactivation Tsd
            rs_tsd = None
            rs_dict = sdict.get("rs", {})
            if template_idx in rs_dict:
                rs_tsd = rs_dict[template_idx]
            elif rs_dict:
                rs_tsd = rs_dict[list(rs_dict.keys())[0]]

            # Analyze individual session
            out = self.analyze_mouse(
                mouse_results=results_obj,
                reactivation_tsd=rs_tsd,
                winMS=winMS,
                model_metric_key=model_metric_key,
            )
            per_session[session_key] = out

            # Pool summary DataFrames
            if not out["ripple"]["summary"].empty:
                tmp = out["ripple"]["summary"].copy()
                tmp["session"] = session_key
                ripple_rows.append(tmp)

            if not out["reactivation"]["summary"].empty:
                tmp = out["reactivation"]["summary"].copy()
                tmp["session"] = session_key
                react_rows.append(tmp)

            for mk, mk_out in out["model"].items():
                if not mk_out["summary"].empty:
                    tmp = mk_out["summary"].copy()
                    tmp["session"] = session_key
                    tmp["model_metric"] = mk
                    model_rows[mk].append(tmp)

            d = out["drowsiness"]
            drowsy_rows.append(
                {
                    "session": session_key,
                    "n_nrem_onsets": d.get("n_nrem_onsets", 0),
                    "ripple_rate_slope": self._poly_slope(
                        d.get("bin_centers_sec"), d.get("ripple_rate_curve_hz")
                    ),
                    "reactivation_slope": self._poly_slope(
                        d.get("bin_centers_sec"), d.get("reactivation_curve")
                    ),
                    "model_slope": self._poly_slope(
                        d.get("bin_centers_sec"), d.get("model_curve")
                    ),
                    "motion_slope": self._poly_slope(
                        d.get("bin_centers_sec"), d.get("motion_curve")
                    ),
                    "theta_delta_slope": self._poly_slope(
                        d.get("bin_centers_sec"), d.get("theta_delta_curve")
                    ),
                }
            )

        return {
            "config": self.config,
            "rs_source": rs_source,
            "per_session": per_session,
            "pooled": {
                "ripple_summary": pd.concat(ripple_rows, ignore_index=True)
                if ripple_rows
                else pd.DataFrame(),
                "reactivation_summary": pd.concat(react_rows, ignore_index=True)
                if react_rows
                else pd.DataFrame(),
                "model_summary": pd.concat(
                    [df for lst in model_rows.values() for df in lst],
                    ignore_index=True,
                )
                if any(model_rows.values())
                else pd.DataFrame(),
                "model_summary_by_metric": {
                    mk: (pd.concat(lst, ignore_index=True) if lst else pd.DataFrame())
                    for mk, lst in model_rows.items()
                },
                "drowsiness_summary": pd.DataFrame(drowsy_rows),
            },
        }

    # =========================================================================
    # AUXILIARY UTILITIES
    # =========================================================================

    def _poly_slope(self, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> float:
        """Compute degree-1 polynomial slope safely."""
        if x is None or y is None or x.size == 0 or y.size == 0:
            return np.nan
        keep = np.isfinite(x) & np.isfinite(y)
        if np.sum(keep) < self.config.min_samples_for_stats:
            return np.nan
        return float(np.polyfit(x[keep], y[keep], deg=1)[0])


def _parse_session_key(session_key: str) -> tuple[str, str]:
    """Split a session key into mouse name and manipulation label."""
    if not isinstance(session_key, str) or "_" not in session_key:
        return session_key, ""
    mouse_name, manipe = session_key.rsplit("_", 1)
    return mouse_name.capitalize(), manipe[:1].upper() + manipe[1:]
