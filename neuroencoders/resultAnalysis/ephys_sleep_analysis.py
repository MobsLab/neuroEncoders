"""Sleep-focused ephys/model analysis utilities.

This module provides a reusable analyser (SleepEphysAnalyser) to quantify:
- Ripple dynamics and cell-type specific PSTHs across sleep states.
- 128D Neural Network (NN) latent manifold trajectories and velocity.
- Multi-signal continuous correlations (NN metrics vs LFP vs Motion).
- Memory Reactivation (e.g., Peyrache 2010) vs NN predictions (maxp, pos2d).
- Pre-Sleep vs Post-Sleep comparative statistics and temporal emergence.

Data Sources:
- `mouse_results`: Contains discrete epochs, LFP, behavior, and spike data
  (generated via the standard lab data extraction pipeline (see `utils.MOBS_Functions`)).
- `reactivation_tsd`: Classical discrete cell-assembly reactivation strengths
  computed upstream (e.g., PCA/ICA Peyrache 2010 method).
- `model_tsds`: Continuous outputs from the Neural Network inference
  (e.g., `pos2d`, `maxp`, `Hn`, `latent_output`), extracted via `_extract_model_metric_tsd`.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from pynapple import IntervalSet, TsGroup, Tsd, TsdFrame
from scipy.signal import hilbert


@dataclass
class SleepAnalysisConfig:
    transition_window_sec: float = 120.0
    bin_size_sec: float = 5.0
    psth_bin_sec: float = 0.02  # 20 ms bins for SWR PSTHs
    psth_window_sec: float = 1.0  # [-1s, +1s] around ripples
    drowsiness_window_sec: float = 180.0
    min_samples_for_stats: int = 5


class SleepEphysAnalyser:
    """Sleep dynamics analyser for ephys cell types and continuous model metrics."""

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
        mouse_results: Any,
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

    def analyse_latent_manifold_projections(
        self,
        mouse_results: Any,
        winMS: int = 108,
        n_components: int = 3,
    ) -> Dict[str, Any]:
        """Project 128D latent_output onto global PCA/UMAP axes and compute manifold metrics."""
        model_data = self._extract_model_metric_tsd(
            mouse_results=mouse_results,
            winMS=winMS,
            metric_key=["latent_output_pooled", "Hn", "maxp", "latent_output"],
        )

        latent_tsd = model_data.get(
            "latent_output_pooled", model_data.get("latent_output", None)
        )

        if latent_tsd is None or latent_tsd.values.ndim < 2:
            return {"proj_model": None, "projected_states": {}, "velocity_tsd": None}

        states = self.get_sleep_state_intervals(mouse_results=mouse_results)
        z_all = np.asarray(latent_tsd.values, dtype=float)
        ts_all = np.asarray(latent_tsd.index, dtype=float)

        try:
            try:
                from cuml.manifold import UMAP

                proj_model = UMAP(n_components=min(n_components, z_all.shape[1]))
                proj_model.fit(z_all[:3])  # test
            except Exception:
                from umap import UMAP

                proj_model = UMAP(n_components=min(n_components, z_all.shape[1]))
        except Exception:
            from sklearn.decomposition import PCA

            proj_model = PCA(n_components=min(n_components, z_all.shape[1]))

        proj_model.fit(z_all)
        projected_states = {}
        for sname in ("wake", "microwake", "nrem", "rem"):
            sep = states.get(sname, None)
            if sep is not None:
                try:
                    z_state = latent_tsd.restrict(sep)
                    if len(z_state) > 0:
                        projected_states[sname] = {
                            "times": np.asarray(z_state.index, dtype=float),
                            "proj": proj_model.transform(
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
            "proj_model": proj_model,
            "projected_states": projected_states,
            "velocity_tsd": velocity_tsd,
            "explained_variance_ratio": getattr(
                proj_model, "explained_variance_ratio_", None
            ),
        }

    # =========================================================================
    # 4. PUBLICATION-READY PLOTTING FUNCTIONS
    # =========================================================================

    def plot_celltype_swr_psths(
        self,
        swr_psth_data: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Plot Peri-SWR PSTH curves split by cell class."""
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
            mean, sem = prof["mean"], prof["sem"]
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

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "swr_psth.png"), dpi=300, bbox_inches="tight"
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_latent_manifold_2d(
        self,
        manifold_data: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Plot 2D PCA/UMAP projection colored by sleep state."""
        proj_states = manifold_data.get("projected_states", {})
        if not proj_states:
            return None, None

        fig, ax = plt.subplots(figsize=(7, 6))
        palette = {
            "wake": "#7F7F7F",
            "nrem": "#2C7FB8",
            "rem": "#D7191C",
            "microwake": "#FF7F00",
        }

        for sname in ["wake", "microwake", "nrem", "rem"]:
            if sname not in proj_states:
                continue
            P = proj_states[sname]["proj"]
            if P.shape[1] < 2:
                continue

            idx = np.random.choice(
                P.shape[0], size=min(2500, P.shape[0]), replace=False
            )
            ax.scatter(
                P[idx, 0],
                P[idx, 1],
                c=palette[sname],
                label=sname.upper(),
                alpha=0.4,
                s=10,
                edgecolor="none",
                rasterized=True,
            )

        var_ratio = manifold_data.get("explained_variance_ratio", [0, 0])
        var_ratio = (
            var_ratio if var_ratio is not None and len(var_ratio) >= 2 else [0, 0]
        )

        ax.set_xlabel(f"Dim 1 ({var_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"Dim 2 ({var_ratio[1] * 100:.1f}%)")
        ax.set_title("Model Latent Manifold Topography (2D)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right", markerscale=2)
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "manifold_2d.png"), dpi=300, bbox_inches="tight"
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_latent_manifold_3d(
        self,
        manifold_data: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        proj_states = manifold_data.get("projected_states", {})
        if not proj_states:
            return None, None

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        palette = {
            "wake": "#7F7F7F",
            "nrem": "#2C7FB8",
            "rem": "#D7191C",
            "microwake": "#FF7F00",
        }

        for sname in ["wake", "microwake", "nrem", "rem"]:
            if sname not in proj_states:
                continue
            P = proj_states[sname]["proj"]
            if P.shape[1] < 3:
                continue
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

        ax.set_title("Model Latent Manifold Topography (3D)")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "manifold_3d.png"), dpi=300, bbox_inches="tight"
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    # =========================================================================
    # 7. PRE- VS POST-SLEEP COMPARATIVE ANALYSIS
    # =========================================================================

    def compute_pre_post_statistics(
        self,
        pre_data: Dict[str, Any],
        post_data: Dict[str, Any],
    ) -> pd.DataFrame:
        pre_binned = pre_data.get("binned_evolution", pd.DataFrame())
        post_binned = post_data.get("binned_evolution", pd.DataFrame())
        if pre_binned.empty or post_binned.empty:
            return pd.DataFrame()

        pre_means = pre_binned.mean(numeric_only=True)
        post_means = post_binned.mean(numeric_only=True)

        for col in ["time_from_start_min", "time_sec"]:
            pre_means.drop(col, errors="ignore", inplace=True)
            post_means.drop(col, errors="ignore", inplace=True)

        summary = pd.DataFrame({"Pre_Mean": pre_means, "Post_Mean": post_means})
        summary["Delta (Post - Pre)"] = summary["Post_Mean"] - summary["Pre_Mean"]
        summary["% Change"] = (
            summary["Delta (Post - Pre)"] / summary["Pre_Mean"].replace(0, np.nan)
        ) * 100
        return summary

    def plot_pre_post_comparison(
        self,
        pre_data: Dict[str, Any],
        post_data: Dict[str, Any],
        normalize: bool = True,
        smoothing_window_bins: int = 3,
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        pre_binned = pre_data.get("binned_evolution", pd.DataFrame())
        post_binned = post_data.get("binned_evolution", pd.DataFrame())
        if pre_binned.empty or post_binned.empty:
            return None, None

        def _process(s_pre: pd.Series, s_post: pd.Series):
            sp1, sp2 = s_pre.copy(), s_post.copy()
            if smoothing_window_bins > 1:
                sp1 = sp1.rolling(
                    window=smoothing_window_bins, center=True, min_periods=1
                ).mean()
                sp2 = sp2.rolling(
                    window=smoothing_window_bins, center=True, min_periods=1
                ).mean()

            if not normalize:
                return sp1, sp2
            g_min, g_max = min(sp1.min(), sp2.min()), max(sp1.max(), sp2.max())
            if pd.isna(g_min) or g_max == g_min:
                return sp1, sp2
            return (sp1 - g_min) / (g_max - g_min), (sp2 - g_min) / (g_max - g_min)

        metrics = [
            c
            for c in pre_binned.columns
            if c not in ["time_from_start_min", "time_sec"]
            and c in post_binned.columns
            and "rate_per_min" not in c
        ]
        fig = plt.figure(figsize=(12, 4 + 3 * len(metrics)))
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(
            len(metrics) + 1, 2, figure=fig, height_ratios=[1.5] + [1] * len(metrics)
        )

        ax_bar1, ax_bar2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        cont_mets = [m for m in metrics if m.startswith("continuous_")]
        rip_mets = [
            m for m in metrics if m.startswith("ripple_") and "rate_per_min" not in m
        ]

        pre_means, post_means = (
            pre_binned.mean(skipna=True, numeric_only=True),
            post_binned.mean(skipna=True, numeric_only=True),
        )
        pre_normed = pre_binned[metrics].div(pre_means, axis=1)
        post_normed = post_binned[metrics].div(pre_means, axis=1)

        # Now pre_normed has a mean of 1.0, and post_normed is scaled relative to 1.0
        # Compute SEM or SD across time bins:
        pre_sem = pre_normed.sem(skipna=True)
        post_sem = post_normed.sem(skipna=True)

        def _draw_grouped_bars(ax, metric_list, title):
            if not metric_list:
                ax.axis("off")
                return

            x = np.arange(len(metric_list))
            width = 0.35

            # Values & errors
            y_pre = [1.0] * len(metric_list)
            y_post = (post_means[metric_list] / pre_means[metric_list]).values

            err_pre = pre_sem[metric_list].values
            err_post = post_sem[metric_list].values

            # Bars
            ax.bar(
                x - width / 2,
                y_pre,
                width,
                yerr=err_pre,
                capsize=4,
                label="PRE",
                color="#4a7bb0",
                alpha=0.85,
            )
            ax.bar(
                x + width / 2,
                y_post,
                width,
                yerr=err_post,
                capsize=4,
                label="POST",
                color="#e15759",
                alpha=0.85,
            )

            # Visual cues: baseline at 1.0 and formatted labels
            ax.axhline(
                1.0,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                zorder=0,
                label="Baseline",
            )
            ax.set_xticks(x)
            clean_labels = [
                m.replace("continuous_", "").replace("ripple_", "") for m in metric_list
            ]
            ax.set_xticklabels(clean_labels, rotation=25, ha="right")
            ax.set_ylabel("Fold Change (Rel. to PRE)")
            ax.set_title(title)
            ax.legend(frameon=True)
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        if cont_mets:
            _draw_grouped_bars(ax_bar1, cont_mets, "Continuous Metrics (Relative)")

        if rip_mets:
            _draw_grouped_bars(ax_bar2, rip_mets, "Ripple Metrics (Relative)")

        axes_evo = []
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[i + 1, :])
            axes_evo.append(ax)
            p1, p2 = _process(pre_binned[metric], post_binned[metric])

            ax.plot(
                pre_binned["time_from_start_min"],
                p1,
                color="#2C7FB8",
                lw=2.5,
                alpha=0.85,
                label="Pre-Sleep",
            )
            ax.plot(
                post_binned["time_from_start_min"],
                p2,
                color="#D7191C",
                lw=2.5,
                alpha=0.85,
                label="Post-Sleep",
            )

            clean_name = (
                metric.replace("continuous_", "Continuous: ")
                .replace("ripple_quality_", "Ripple Quality: ")
                .replace("ripple_rate_per_min", "Ripple Rate (SWRs/min)")
            )
            ax.set_ylabel("Norm. Score" if normalize else "Raw Value")
            ax.set_title(f"{clean_name} (Smoothed: {smoothing_window_bins} bins)")
            ax.grid(alpha=0.2)

            if i == 0:
                ax.legend(frameon=False, loc="upper right")
            if i == len(metrics) - 1:
                ax.set_xlabel("Time from Epoch Start (minutes)")

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "pre_post_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, (ax_bar1, ax_bar2, axes_evo)

    # =========================================================================
    # 8. REPLAY CONTENT & EMERGENCE ANALYSIS (PRE VS POST)
    # =========================================================================

    def analyse_replay_content_and_emergence(
        self,
        mouse_results: Any,
        pos2d_tsd: Union[TsdFrame, Tsd, Any],
        reactivation_tsd: Tsd,
        maxp_tsd: Optional[Tsd] = None,
        spatial_bins: int = 15,
        rs_threshold_z: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Analyzes the spatial content (predicted pos2d) of strong reactivations
        and tracks their emergence over time to compare Pre-Sleep and Post-Sleep.

        Uses Pynapple's `value_from()` for perfect time alignment across disparate
        sampling rates. For multi-token sweeps, it maps ALL tokens into the spatial
        histogram avoiding the "forbidden zone" averaging trap.
        """
        import scipy.ndimage as ndimage

        pos_vals = pos2d_tsd.values
        spread_tsd = None
        is_3d, n_tokens = False, 1

        if pos_vals.ndim == 3:
            is_3d = True
            n_tokens = pos_vals.shape[1]
            diffs = np.diff(pos_vals, axis=1)  # (Time, Tokens-1, 2)
            path_lengths = np.sum(np.linalg.norm(diffs, axis=2), axis=1)
            spread_tsd = Tsd(t=pos2d_tsd.index, d=path_lengths)
            pos_vals_flat = pos_vals.reshape(pos_vals.shape[0], -1)
            pos2d_clean = TsdFrame(t=pos2d_tsd.index, d=pos_vals_flat)

        elif pos_vals.ndim == 2:
            spread_tsd = Tsd(t=pos2d_tsd.index, d=np.zeros(pos_vals.shape[0]))
            pos2d_clean = TsdFrame(t=pos2d_tsd.index, d=pos_vals, columns=["x", "y"])

        elif pos_vals.ndim == 1:
            warn("pos2d_tsd appears 1D. Defaulting y to 0.")
            pos_vals = np.column_stack((pos_vals, np.zeros_like(pos_vals)))
            spread_tsd = Tsd(t=pos2d_tsd.index, d=np.zeros(pos_vals.shape[0]))
            pos2d_clean = TsdFrame(t=pos2d_tsd.index, d=pos_vals, columns=["x", "y"])

        # Pynapple strict nearest-neighbor time synchronization
        pos2d_clean = reactivation_tsd.value_from(pos2d_clean)
        if spread_tsd is not None:
            spread_tsd = reactivation_tsd.value_from(spread_tsd)
        if maxp_tsd is not None:
            maxp_tsd = reactivation_tsd.value_from(maxp_tsd)

        if is_3d:
            restored_shape = pos2d_clean.values.reshape(-1, n_tokens, 2)
            x_min, x_max = np.nanpercentile(restored_shape[:, :, 0], [1, 99])
            y_min, y_max = np.nanpercentile(restored_shape[:, :, 1], [1, 99])
        else:
            x_min, x_max = np.nanpercentile(pos2d_clean.values[:, 0], [1, 99])
            y_min, y_max = np.nanpercentile(pos2d_clean.values[:, 1], [1, 99])

        pos_range = [[x_min, x_max], [y_min, y_max]]

        rs_vals = reactivation_tsd.values
        valid_rs = rs_vals[np.isfinite(rs_vals)]
        if len(valid_rs) == 0:
            return {"error": "Reactivation TSD contains only NaNs."}

        rs_mean, rs_std = np.mean(valid_rs), np.std(valid_rs)
        rs_z = (rs_vals - rs_mean) / (rs_std + 1e-9)
        rs_z_tsd = Tsd(t=reactivation_tsd.index, d=rs_z)
        strong_intervals = rs_z_tsd.threshold(rs_threshold_z, "above").time_support

        results = {"pos_range": pos_range}

        for ep_name in ["pre_sleep", "post_sleep"]:
            ep, _ = mouse_results.get_epoch_interval(ep_name)
            df_ep = self._safe_interval_df(ep)
            if df_ep.empty:
                results[ep_name] = None
                continue

            ep_start, ep_end = float(df_ep["start"].min()), float(df_ep["end"].max())
            ep_strong = self._interval_intersection(strong_intervals, ep)

            if ep_strong is None or len(ep_strong) == 0:
                results[ep_name] = None
                continue

            pos_strong = pos2d_clean.restrict(ep_strong)
            rs_strong = reactivation_tsd.restrict(ep_strong)

            if is_3d:
                pos_strong_3d = pos_strong.values.reshape(-1, n_tokens, 2)
                x = pos_strong_3d[:, :, 0].flatten()
                y = pos_strong_3d[:, :, 1].flatten()
                weights = np.repeat(rs_strong.values, n_tokens)
            else:
                x = pos_strong.values[:, 0]
                y = pos_strong.values[:, 1]
                weights = rs_strong.values

            sum_map, _, _ = np.histogram2d(
                x, y, bins=spatial_bins, range=pos_range, weights=weights
            )
            count_map, _, _ = np.histogram2d(x, y, bins=spatial_bins, range=pos_range)

            mean_map = np.divide(
                sum_map, count_map, out=np.zeros_like(sum_map), where=count_map > 0
            )
            mean_map_smoothed = ndimage.gaussian_filter(
                np.nan_to_num(mean_map), sigma=0.8
            )

            bins = np.arange(ep_start, ep_end + 60.0, 60.0)
            centers = bins[:-1] + 30.0

            strong_times = self._interval_starts(ep_strong)
            emergence_counts, _ = np.histogram(strong_times, bins=bins)
            emergence_cumulative = np.cumsum(emergence_counts)

            maxp_evo, spread_evo = [], []
            for i in range(len(bins) - 1):
                b_ep = IntervalSet(start=[bins[i]], end=[bins[i + 1]])
                b_strong = self._interval_intersection(b_ep, ep_strong)

                if b_strong is not None and len(b_strong) > 0:
                    maxp_evo.append(
                        np.nanmean(maxp_tsd.restrict(b_strong).values)
                        if maxp_tsd is not None
                        else np.nan
                    )
                    spread_evo.append(
                        np.nanmean(spread_tsd.restrict(b_strong).values)
                        if spread_tsd is not None
                        else np.nan
                    )
                else:
                    maxp_evo.append(np.nan)
                    spread_evo.append(np.nan)

            if maxp_tsd is None:
                maxp_evo = np.full(len(centers), np.nan)
            if spread_tsd is None:
                spread_evo = np.full(len(centers), np.nan)

            results[ep_name] = {
                "mean_map": mean_map_smoothed,
                "time_mins": (centers - ep_start) / 60.0,
                "emergence_rate": emergence_counts,
                "emergence_cumulative": emergence_cumulative,
                "robustness_maxp": np.array(maxp_evo),
                "continuity_spread": np.array(spread_evo),
                "n_strong_events": len(strong_times),
            }

        return results

    def plot_replay_content_and_emergence(
        self,
        content_data: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        if "error" in content_data:
            return None, None

        has_maxp, has_spread = False, False
        for ep in ["pre_sleep", "post_sleep"]:
            if content_data.get(ep) is not None:
                if not np.isnan(content_data[ep]["robustness_maxp"]).all():
                    has_maxp = True
                if not np.isnan(content_data[ep]["continuity_spread"]).all():
                    has_spread = True

        n_time_rows = 1 + int(has_maxp) + int(has_spread)
        fig = plt.figure(figsize=(10, 4 + 2.5 * n_time_rows))
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(
            1 + n_time_rows, 2, figure=fig, height_ratios=[1.5] + [1] * n_time_rows
        )

        ax_map_pre, ax_map_post = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        extent = None
        if "pos_range" in content_data:
            pr = content_data["pos_range"]
            extent = [pr[0][0], pr[0][1], pr[1][0], pr[1][1]]

        global_vmax = 0
        for ep in ["pre_sleep", "post_sleep"]:
            if content_data.get(ep) is not None:
                global_vmax = max(global_vmax, np.max(content_data[ep]["mean_map"]))

        for ax, ep, title in zip(
            [ax_map_pre, ax_map_post],
            ["pre_sleep", "post_sleep"],
            ["Pre-Sleep", "Post-Sleep"],
        ):
            if content_data.get(ep) is not None:
                n_ev = content_data[ep]["n_strong_events"]
                im = ax.imshow(
                    content_data[ep]["mean_map"].T,
                    origin="lower",
                    extent=extent,
                    cmap="inferno",
                    vmin=0,
                    vmax=global_vmax,
                )
                ax.set_title(f"{title} Spatial Replay\n(n={n_ev} strong events)")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Strong Events",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{title} Spatial Replay")

            ax.set_xlabel("Predicted X")
            ax.set_ylabel("Predicted Y")

        if global_vmax > 0:
            fig.colorbar(
                im,
                ax=[ax_map_pre, ax_map_post],
                label="Mean Reactivation Strength",
                shrink=0.8,
            )

        ax_em = fig.add_subplot(gs[1, :])
        time_axes = [ax_em]
        colors, labels = (
            {"pre_sleep": "#2C7FB8", "post_sleep": "#D7191C"},
            {"pre_sleep": "Pre-Sleep", "post_sleep": "Post-Sleep"},
        )

        for ep in ["pre_sleep", "post_sleep"]:
            if content_data.get(ep) is not None:
                ax_em.plot(
                    content_data[ep]["time_mins"],
                    content_data[ep]["emergence_cumulative"],
                    color=colors[ep],
                    lw=2.5,
                    label=labels[ep],
                )
        ax_em.set_ylabel("Cumulative Count")
        ax_em.set_title("Emergence of Strong Reactivations Over Time")
        ax_em.grid(alpha=0.3)
        ax_em.legend(frameon=False)

        current_row = 2
        if has_maxp:
            ax_rob = fig.add_subplot(gs[current_row, :], sharex=ax_em)
            time_axes.append(ax_rob)
            for ep in ["pre_sleep", "post_sleep"]:
                if content_data.get(ep) is not None:
                    ax_rob.plot(
                        content_data[ep]["time_mins"],
                        content_data[ep]["robustness_maxp"],
                        color=colors[ep],
                        lw=2,
                        alpha=0.8,
                    )
            ax_rob.set_ylabel("Mean MaxP")
            ax_rob.set_title("Prediction Robustness During Strong Replay")
            ax_rob.grid(alpha=0.3)
            current_row += 1

        if has_spread:
            ax_cont = fig.add_subplot(gs[current_row, :], sharex=ax_em)
            time_axes.append(ax_cont)
            for ep in ["pre_sleep", "post_sleep"]:
                if content_data.get(ep) is not None:
                    ax_cont.plot(
                        content_data[ep]["time_mins"],
                        content_data[ep]["continuity_spread"],
                        color=colors[ep],
                        lw=2,
                        alpha=0.8,
                    )
            ax_cont.set_ylabel("Path Length")
            ax_cont.set_title(
                "Prediction Continuity / Sweep Length (Higher = Longer Continuous Trajectory)"
            )
            ax_cont.grid(alpha=0.3)

        time_axes[-1].set_xlabel("Time from Sleep Epoch Start (minutes)")

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "replay_content.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, (ax_map_pre, ax_map_post, time_axes)

    # =========================================================================
    # INTERNAL HELPER METHODS (Extraction, Bins, Math)
    # =========================================================================

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

        wake = scoring.get("wake_epochs", None)

        microwake = scoring.get("microwake_epochs", None)

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

            microwake = self._interval_intersection(microwake, sleep_union)

        return {
            "rem": rem,
            "nrem": nrem,
            "wake": wake,
            "microwake": microwake,
            "sleep_union": sleep_union,
        }

    def _extract_model_metric_tsd(
        self,
        mouse_results: Any,
        winMS: int,
        metric_key: Union[str, Iterable[str]] = ["Hn", "maxp"],
    ) -> Dict[str, Any]:
        """
        Extract continuous model metrics (scalars, 2D vectors, or 3D multi-token sequences) across phases.
        Automatically pools 3D sequence tensors into 2D via masked average pooling UNLESS the metric
        key ends with '_k' (which denotes a multi-token output you want to preserve).
        """
        if not hasattr(mouse_results, "resultsNN_phase") and not hasattr(
            mouse_results, "resultsNN_phase_pkl"
        ):
            warn(
                "Mouse results do not contain 'resultsNN_phase' or 'resultsNN_phase_pkl'; returning None."
            )
            return {k: None for k in self._ensure_metric_keys(metric_key)}

        if not hasattr(mouse_results, "sleepFigures"):
            warn(
                "Please Load Mouse_Results with sleep = True to ensure access to predictions during sleep."
            )
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
            warn(
                f"Window {winMS} ms not found in mouse_results; returning None for all metrics."
            )
            return {k: None for k in metric_keys}

        # --- WAKE PHASES ---
        if (
            getattr(mouse_results, "resultsNN_phase", None) is None
            or len(mouse_results.resultsNN_phase.keys()) == 0
        ):
            mouse_results.load_data()

        for suffix in ["_pre", "_training", "_cond", "_post"]:
            must_unload = False
            phase_dict = getattr(mouse_results, "resultsNN_phase", {}).get(suffix, None)
            pkl_dict = getattr(mouse_results, "resultsNN_phase_pkl", {}).get(
                suffix, None
            )

            if phase_dict is None or (
                any(k not in phase_dict for k in metric_keys)
                and (pkl_dict is None or any(k not in pkl_dict for k in metric_keys))
            ):
                must_unload = True
                og_suffixes = mouse_results.suffixes
                mouse_results.load_data(
                    suffixes=[suffix],
                    load_pickle=True,
                    redo=True,
                    keys_to_load=metric_keys,
                )
                phase_dict = getattr(mouse_results, "resultsNN_phase", {}).get(
                    suffix, None
                )
                pkl_dict = getattr(mouse_results, "resultsNN_phase_pkl", {}).get(
                    suffix, None
                )
                mouse_results.suffixes = og_suffixes

            try:
                times = np.array(phase_dict["times"][id_window]).flatten()
            except Exception:
                if must_unload and hasattr(mouse_results, "unload"):
                    mouse_results.unload()
                continue

            arrays: Dict[str, Any] = {}
            if isinstance(phase_dict, dict):
                for k in set(
                    self._ensure_metric_keys(metric_key) + ["maxp", "predLoss", "Hn"]
                ):
                    if k in phase_dict and len(phase_dict[k]) > id_window:
                        arrays[k] = np.array(phase_dict[k][id_window])

            if isinstance(pkl_dict, dict) and any(k not in arrays for k in metric_keys):
                for k in set(
                    self._ensure_metric_keys(metric_key)
                    + [
                        "maxp",
                        "predLoss",
                        "Hn",
                        "latent_output",
                        "latent_output_pooled",
                    ]
                ):
                    if k in pkl_dict and len(pkl_dict[k]) > id_window:
                        arrays[k] = np.array(pkl_dict[k][id_window])

            for key in metric_keys:
                vals = arrays.get(key, None)
                if vals is None:
                    continue
                v_arr = np.asarray(vals, dtype=float)

                if v_arr.ndim == 3 and not key.endswith("_k"):
                    mask = np.any(v_arr != 0, axis=-1, keepdims=True)
                    sum_inputs = np.sum(v_arr * mask, axis=1)
                    count_inputs = np.maximum(np.sum(mask, axis=1), 1.0)
                    v_arr = sum_inputs / count_inputs

                min_n = min(times.shape[0], v_arr.shape[0])
                if min_n < 2:
                    continue
                times_all[key].append(times[:min_n])
                values_all[key].append(v_arr[:min_n])

            del phase_dict, pkl_dict, arrays
            gc.collect()
            if must_unload and hasattr(mouse_results, "unload"):
                mouse_results.unload()

        # --- SLEEP PHASES ---
        if (
            getattr(mouse_results.sleepFigures, "resultsNN_phase", None) is None
            or len(mouse_results.sleepFigures.resultsNN_phase.keys()) == 0
        ):
            mouse_results.sleepFigures.load_data()

        sleep_names = list(mouse_results.sleepFigures.resultsNN_phase.keys())

        for sleep_name in sleep_names:
            phase_dict = mouse_results.sleepFigures.resultsNN_phase.get(
                sleep_name, None
            )
            pkl_dict = getattr(
                mouse_results.sleepFigures, "resultsNN_phase_pkl", {}
            ).get(sleep_name, None)

            if phase_dict is None or (
                any(k not in phase_dict for k in metric_keys)
                and (pkl_dict is None or any(k not in pkl_dict for k in metric_keys))
            ):
                mouse_results.sleepFigures.load_data(
                    sleepNames=[sleep_name], load_pickle=True, keys_to_load=metric_keys
                )
                phase_dict = mouse_results.sleepFigures.resultsNN_phase.get(
                    sleep_name, None
                )
                pkl_dict = getattr(
                    mouse_results.sleepFigures, "resultsNN_phase_pkl", {}
                ).get(sleep_name, None)

            if phase_dict is None:
                warn(
                    f"Sleep phase '{sleep_name}' not found in resultsNN_phase; skipping."
                )
                continue

            try:
                times = np.array(phase_dict["times"][id_window]).flatten()
            except Exception:
                continue

            arrays: Dict[str, Any] = {}
            if isinstance(phase_dict, dict):
                for k in set(
                    self._ensure_metric_keys(metric_key) + ["maxp", "predLoss", "Hn"]
                ):
                    if k in phase_dict and len(phase_dict[k]) > id_window:
                        arrays[k] = np.array(phase_dict[k][id_window])

            if isinstance(pkl_dict, dict):
                for k in set(
                    self._ensure_metric_keys(metric_key)
                    + [
                        "maxp",
                        "predLoss",
                        "Hn",
                        "latent_output",
                        "latent_output_pooled",
                    ]
                ):
                    if k in pkl_dict and len(pkl_dict[k]) > id_window:
                        arrays[k] = np.array(pkl_dict[k][id_window])

            for key in metric_keys:
                vals = arrays.get(key, None)
                if vals is None:
                    continue
                v_arr = np.asarray(vals, dtype=float)

                if v_arr.ndim == 3 and not key.endswith("_k"):
                    mask = np.any(v_arr != 0, axis=-1, keepdims=True)
                    sum_inputs = np.sum(v_arr * mask, axis=1)
                    count_inputs = np.maximum(np.sum(mask, axis=1), 1.0)
                    v_arr = sum_inputs / count_inputs

                min_n = min(times.shape[0], v_arr.shape[0])
                if min_n < 2:
                    continue
                times_all[key].append(times[:min_n])
                values_all[key].append(v_arr[:min_n])

            del phase_dict, pkl_dict, arrays
            gc.collect()

        # --- AGGREGATION ---
        out: Dict[str, Any] = {}
        for key in metric_keys:
            if not times_all[key]:
                out[key] = None
                continue

            t = np.concatenate(times_all[key])
            v = np.concatenate(values_all[key], axis=0)

            order = np.argsort(t)
            t_ord, v_ord = t[order], v[order]

            if v_ord.ndim == 1:
                out[key] = self._to_tsd(t_ord, v_ord)
            elif v_ord.ndim == 2:
                out[key] = TsdFrame(t=t_ord, d=v_ord)
            else:
                try:
                    from pynapple import TsdTensor

                    out[key] = TsdTensor(t=t_ord, d=v_ord)
                except ImportError:
                    warn(
                        "Pynapple TsdTensor not found. Using custom 3D Fallback object."
                    )

                    class Tsd3DFallback:
                        def __init__(self, index, values):
                            self.index = index
                            self.values = values

                    out[key] = Tsd3DFallback(t_ord, v_ord)

        return out

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
                return pd.DataFrame(
                    {
                        "start": np.asarray(df.iloc[:, 0], dtype=float),
                        "end": np.asarray(df.iloc[:, 1], dtype=float),
                    }
                )
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
        df = SleepEphysAnalyser._safe_interval_df(interval)
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
        times, values = np.asarray(times).flatten(), np.asarray(values).flatten()
        keep = np.isfinite(times) & np.isfinite(values)
        if np.sum(keep) < 2:
            return None

        t, d = times[keep], values[keep]
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

    @staticmethod
    def _safe_value_from(
        target_times: np.ndarray, source_data: Union[Tsd, TsdFrame, Any]
    ) -> np.ndarray:
        """
        Safely extracts nearest values using Pynapple's value_from, padding with NaNs
        for out-of-bounds queries. This strictly guarantees the output array length
        exactly matches len(target_times), avoiding inhomogeneous shape crashes.
        """
        if source_data is None or len(source_data) == 0:
            if hasattr(source_data, "columns") or (
                hasattr(source_data, "values") and source_data.values.ndim > 1
            ):
                return np.full((len(target_times), source_data.values.shape[1]), np.nan)
            return np.full(len(target_times), np.nan)

        import pynapple as nap

        target_ts = nap.Ts(t=target_times, time_units="s")
        aligned = target_ts.value_from(source_data)

        # Ensure output shape matches 1D (Tsd) or 2D (TsdFrame) exactly
        if hasattr(source_data, "columns") or source_data.values.ndim > 1:
            out_array = np.full(
                (len(target_times), source_data.values.shape[1]), np.nan
            )
        else:
            out_array = np.full(len(target_times), np.nan)

        if len(aligned) > 0:
            # aligned.index contains the exact target_times that were successfully matched
            idx = np.searchsorted(target_times, aligned.index.values)
            valid = (idx >= 0) & (idx < len(target_times))
            out_array[idx[valid]] = aligned.values[valid]

        return out_array

    def _poly_slope(self, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> float:
        """Compute degree-1 polynomial slope safely."""
        if x is None or y is None or x.size == 0 or y.size == 0:
            return np.nan
        keep = np.isfinite(x) & np.isfinite(y)
        if np.sum(keep) < self.config.min_samples_for_stats:
            return np.nan
        return float(np.polyfit(x[keep], y[keep], deg=1)[0])

    def analyse_celltype_firing_rates_by_state(
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

            # Analyse mean rates per cell type across states
            for state_name in ("rem", "nrem", "wake", "microwake"):
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

    def analyse_ripples_by_state(
        self,
        mouse_results: Any,
        state_intervals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute ripple rates by state and peri-state-onset ripple profiles.

        Data Sources:
        - `ripple_times`: Extracted from `mouse_results.DataHelper.fullBehavior["Times"]["tRipples"]`.
          Because these are discrete timestamp events (point process) rather than a continuous
          signal, we do not use `value_from()`. Instead, we cast them to a Pynapple `Ts` object
          and use native `.restrict()` to elegantly filter events within specific sleep states.
        - `state_intervals`: Dictionary of sleep state IntervalSets.
        """
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

        # Convert discrete events into a native Pynapple Point Process (Ts)
        import pynapple as nap

        rip_ts = nap.Ts(t=ripple_times, time_units="s")

        rows = []
        for state_name in ("rem", "nrem", "wake", "microwake"):
            state_ep = state_intervals.get(state_name, None)
            duration = self._interval_duration_sec(state_ep)

            # Use Pynapple's native restrict() to count events falling inside the epochs
            if state_ep is not None and not self._safe_interval_df(state_ep).empty:
                count = len(rip_ts.restrict(state_ep))
            else:
                count = 0

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
        for state_name in ("rem", "nrem", "wake", "microwake"):
            onsets = self._interval_starts(state_intervals.get(state_name, None))
            if onsets.size == 0:
                transition_profiles[state_name] = {
                    "time_sec": centers,
                    "ripple_rate_hz": np.full_like(centers, np.nan, dtype=float),
                    "n_onsets": 0,
                }
                continue

            # Compute Peri-Event Time Histogram (PETH) around State Onsets
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

    # =========================================================================
    # SINGLE MOUSE ANALYSIS
    # =========================================================================

    def analyse_mouse(
        self,
        mouse_results: Union[Any, Any],
        reactivation_tsd: Optional[Tsd] = None,
        winMS: int = 108,
        model_metric_key: Union[str, Iterable[str]] = ["Hn", "maxp"],
    ) -> Dict[str, Any]:
        """Run complete multi-signal sleep analysis for a single Union[Any, Any] instance."""
        # 1. Macro Sleep-State Epochs (REM, NREM, Wake)
        states = self.get_sleep_state_intervals(mouse_results=mouse_results)

        # 2. Ripple Dynamics & Cell-Type PSTHs
        ripple_out = self.analyse_ripples_by_state(
            mouse_results=mouse_results, state_intervals=states
        )
        swr_psth_out = self.compute_swr_psth_by_celltype(
            mouse_results=mouse_results, state_intervals=states
        )

        # 3. Cell-Type Firing Rate Dynamics Across States
        fr_out = self.analyse_celltype_firing_rates_by_state(
            mouse_results=mouse_results, state_intervals=states
        )

        # 4. Assembly Reactivation Strength Analysis
        react_out = self._analyse_tsd_by_state(
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
            model_out[mk] = self._analyse_tsd_by_state(
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
        manifold_out = self.analyse_latent_manifold_projections(
            mouse_results=mouse_results, winMS=winMS, n_components=3
        )

        # 8. Multi-Signal Pre-NREM Drowsiness Trend Alignment
        drowsiness = self.analyse_drowsiness_multi_signal(
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

    def analyse_loader(
        self,
        results_loader: Any,
        winMS: int = 108,
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

            results_obj = sdict.get("results", None)
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

            # Analyse individual session
            out = self.analyse_mouse(
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

    def plot_drowsiness_multi_panel(
        self,
        drowsiness: Dict[str, Any],
        save_dir: Optional[str] = None,
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
            (drowsiness.get("model_curve", np.array([])), "Model Metric", "#1A9641"),
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

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "drowsiness_multi_panel.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    # =========================================================================
    # MULTI-SIGNAL ALIGNMENT (LFP, Drowsiness, State-Transitions)
    # =========================================================================

    def compute_lfp_and_motion_metrics(
        self,
        mouse_results: Union[Any, Any],
        state_intervals: Dict[str, Any],
        immobility_threshold: float = 1.7e7,
    ) -> Dict[str, Any]:
        """
        Compute continuous LFP power ratios, sleep pressure decay, and quiet vs. active wake.

        Data Sources:
        - `mouse_results`: LFP and motion arrays extracted from DataHelper.
        - `state_intervals`: Sleep epochs from get_sleep_state_intervals.
        """
        acc_tsd = self.extract_accelerometer_tsd(mouse_results)
        lfp_dict = self.extract_lfp_signals(mouse_results)

        theta_lfp = lfp_dict.get("theta", None)
        delta_lfp = lfp_dict.get("delta", None)

        theta_delta_ratio = None
        if theta_lfp is not None and delta_lfp is not None:
            try:
                # Safely align delta onto theta's time grid using safe nearest-neighbor
                d_val = self._safe_value_from(theta_lfp.index.values, delta_lfp)

                p_theta = np.abs(hilbert(theta_lfp.values)) ** 2
                p_delta = np.abs(hilbert(d_val)) ** 2

                ratio = p_theta / (p_delta + 1e-12)
                theta_delta_ratio = Tsd(t=theta_lfp.index, d=ratio)
            except Exception:
                pass

        wake_ep = state_intervals.get("wake", None)
        qw_epochs, aw_epochs = None, None

        if wake_ep is not None and acc_tsd is not None:
            try:
                acc_wake = acc_tsd.restrict(wake_ep)
                is_immobile = acc_wake.values < immobility_threshold

                qw_epochs = acc_wake[is_immobile].time_support
                aw_epochs = acc_wake[~is_immobile].time_support
            except Exception:
                pass

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

    def analyse_drowsiness_multi_signal(
        self,
        ripple_times: np.ndarray,
        reactivation_tsd: Optional[Tsd],
        state_intervals: Dict[str, Any],
        acc_tsd: Optional[Tsd] = None,
        model_tsd: Optional[Tsd] = None,
        theta_delta_tsd: Optional[Tsd] = None,
    ) -> Dict[str, Any]:
        """
        Quantify multi-signal trends prior to NREM onset.

        Data Sources:
        - `ripple_times`: tRipples from behavior.
        - `reactivation_tsd`: Assembly strengths (Peyrache 2010).
        - `model_tsd`: Extracted NN metric (e.g., maxp, Hn).
        """
        nrem_onsets = self._interval_starts(state_intervals.get("nrem", None))
        if nrem_onsets.size == 0:
            return {"n_nrem_onsets": 0, "bin_centers_sec": np.array([])}

        bins = np.arange(
            -self.config.drowsiness_window_sec,
            self.config.bin_size_sec,
            self.config.bin_size_sec,
        )
        centers = 0.5 * (bins[:-1] + bins[1:])

        def _align_curve(tsd_signal: Optional[Tsd]) -> np.ndarray:
            if tsd_signal is None:
                return np.full_like(centers, np.nan, dtype=float)

            traces = []
            for onset in nrem_onsets:
                sample_times = onset + centers
                aligned_vals = self._safe_value_from(sample_times, tsd_signal)
                traces.append(aligned_vals)

            traces_arr = np.array(traces, dtype=float)
            valid_n = np.sum(np.isfinite(traces_arr), axis=0)
            return np.divide(
                np.nansum(traces_arr, axis=0),
                np.maximum(valid_n, 1),
                out=np.full_like(centers, np.nan, dtype=float),
                where=valid_n > 0,
            )

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
            "motion_curve": _align_curve(acc_tsd),
            "ripple_rate_curve_hz": ripple_curve,
            "reactivation_curve": _align_curve(reactivation_tsd),
            "model_curve": _align_curve(model_tsd),
            "theta_delta_curve": _align_curve(theta_delta_tsd),
        }

    def _analyse_tsd_by_state(
        self,
        tsd_signal: Optional[Union[Tsd, TsdFrame]],
        state_intervals: Dict[str, Any],
        value_name: str,
    ) -> Dict[str, Any]:
        """Quantify summary statistics and state-onset transition profiles."""
        if tsd_signal is None:
            return {"summary": pd.DataFrame(), "transition_profiles": {}}

        rows = []
        for state_name in ("rem", "nrem", "wake", "microwake"):
            state_ep = state_intervals.get(state_name, None)
            try:
                state_vals = np.asarray(
                    tsd_signal.restrict(state_ep).values, dtype=float
                )
            except Exception:
                state_vals = np.array([], dtype=float)

            finite = state_vals[np.isfinite(state_vals)]
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

        # Flatten multi-dimensional frames to 1D for state-transition analysis
        if isinstance(tsd_signal, TsdFrame) or (
            hasattr(tsd_signal, "values") and tsd_signal.values.ndim > 1
        ):
            ts = np.asarray(tsd_signal.index, dtype=float)
            vs = np.nanmean(np.asarray(tsd_signal.values, dtype=float), axis=1)
            order = np.argsort(ts)
            target_tsd = Tsd(t=ts[order], d=vs[order])
        else:
            target_tsd = tsd_signal

        transition_profiles: Dict[str, Dict[str, np.ndarray]] = {}
        for state_name in ("rem", "nrem", "wake", "microwake"):
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
                aligned_vals = self._safe_value_from(sample_times, target_tsd)
                all_traces.append(aligned_vals)

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

    # =========================================================================
    # SLEEP EPOCH EVOLUTION (e.g. Post-Sleep Drowsiness)
    # =========================================================================

    def analyse_sleep_dynamics(
        self,
        mouse_results: Union[Any, Any],
        reactivation_tsd: Optional[Tsd] = None,
        model_tsds: Optional[Dict[str, Tsd]] = None,
        bin_size_sec: float = 60.0,
        smoothing_std_sec: float = 10.0,
        session="postsleep",
    ) -> Dict[str, Any]:
        """
        Quantify temporal evolution of drowsiness and sleep depth across the sleep period.

        Data Sources:
        - `reactivation_tsd`: Assembly strengths (Peyrache 2010).
        - `model_tsds`: Continuous metric outputs from Neural Network inference.
        """
        model_tsds = model_tsds or {}

        def _safe_mean(vals: np.ndarray) -> float:
            vals = np.asarray(vals)
            if vals.size == 0 or np.sum(np.isfinite(vals)) == 0:
                return np.nan
            return float(np.nanmean(vals))

        if "sleep" not in session.lower():
            warn(f"Session '{session}' does not appear to be a sleep epoch.")

        sleep_epoch = (
            getattr(mouse_results, "sleep", None)
            if session == "all"
            else getattr(mouse_results, session.lower(), None)
        )
        if sleep_epoch is None:
            sleep_epoch, _ = mouse_results.get_epoch_interval(session)

        df_ps = self._safe_interval_df(sleep_epoch)
        if df_ps.empty:
            return {"error": "No sleep interval found."}

        ps_start = float(df_ps["start"].min())
        ps_end = float(df_ps["end"].max())

        all_states = self.get_sleep_state_intervals(mouse_results)
        ps_states = {
            st_name: self._interval_intersection(st_ep, sleep_epoch) if st_ep else None
            for st_name, st_ep in all_states.items()
        }

        data_helper = getattr(mouse_results, "DataHelper", None)
        rip_epochs = getattr(data_helper, "ripples_epochs", None)
        ps_rip_epochs = self._interval_intersection(rip_epochs, sleep_epoch)
        df_rip = self._safe_interval_df(ps_rip_epochs)

        rip_centers = (
            df_rip["start"].values + (df_rip["end"].values - df_rip["start"].values) / 2
            if not df_rip.empty
            else np.array([])
        )
        rip_metrics = {"time_sec": rip_centers}

        if not df_rip.empty:
            if reactivation_tsd is not None:
                rs_vals = []
                for _, r in df_rip.iterrows():
                    ep = IntervalSet(start=[r["start"]], end=[r["end"]])
                    rs_vals.append(_safe_mean(reactivation_tsd.restrict(ep).values))
                rip_metrics["reactivation"] = np.array(rs_vals)

            for m_name, m_tsd in model_tsds.items():
                if m_tsd is not None:
                    m_vals = []
                    for _, r in df_rip.iterrows():
                        ep = IntervalSet(start=[r["start"]], end=[r["end"]])
                        m_vals.append(_safe_mean(m_tsd.restrict(ep).values))
                    rip_metrics[m_name] = np.array(m_vals)

        bins = np.arange(ps_start, ps_end + bin_size_sec, bin_size_sec)
        bin_centers = bins[:-1] + bin_size_sec / 2

        binned_results = pd.DataFrame(
            {
                "time_from_start_min": (bin_centers - ps_start) / 60.0,
                "time_sec": bin_centers,
            }
        )

        if not df_rip.empty:
            counts, _ = np.histogram(rip_centers, bins=bins)
            binned_results["ripple_rate_per_min"] = counts / (bin_size_sec / 60.0)
        else:
            binned_results["ripple_rate_per_min"] = 0.0

        if not df_rip.empty:
            for k, v in rip_metrics.items():
                if k == "time_sec":
                    continue
                bin_means = []
                for i in range(len(bins) - 1):
                    mask = (rip_centers >= bins[i]) & (rip_centers < bins[i + 1])
                    bin_means.append(_safe_mean(v[mask]) if np.any(mask) else np.nan)
                binned_results[f"ripple_quality_{k}"] = bin_means

        def _smooth_and_bin_tsd(tsd: Tsd) -> np.ndarray:
            if tsd is None:
                return np.full(len(bin_centers), np.nan)
            if smoothing_std_sec > 0:
                tsd = tsd.smooth(std=smoothing_std_sec, time_units="s")

            res = []
            for i in range(len(bins) - 1):
                ep = IntervalSet(start=[bins[i]], end=[bins[i + 1]])
                res.append(_safe_mean(tsd.restrict(ep).values))
            return np.array(res)

        if reactivation_tsd is not None:
            binned_results["continuous_reactivation"] = _smooth_and_bin_tsd(
                reactivation_tsd
            )

        for m_name, m_tsd in model_tsds.items():
            if m_tsd is not None:
                binned_results[f"continuous_model_{m_name}"] = _smooth_and_bin_tsd(
                    m_tsd
                )

        return {
            "ripple_metrics": pd.DataFrame(rip_metrics)
            if not df_rip.empty
            else pd.DataFrame(),
            "binned_evolution": binned_results,
            "bin_size_sec": bin_size_sec,
            "ps_start": ps_start,
            "states_ps": ps_states,
            "session": session,
        }

    # =========================================================================
    # CORRELATIONS: 0-LAG MATRICES & CCF LAG ARRAYS
    # =========================================================================

    def analyse_metric_correlations_by_state(
        self,
        tsds_to_correlate: Dict[str, Optional[Tsd]],
        state_intervals: Dict[str, Any],
        smoothing_std_sec: float = 0.0,
        bin_sec: float = 0.05,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute Pearson correlation matrices (0-Lag Heatmap) between continuous metrics.

        Data Sources:
        - `tsds_to_correlate`: The NN metrics or LFPs you wish to compare.
        """
        valid_tsds = {k: v for k, v in tsds_to_correlate.items() if v is not None}
        if len(valid_tsds) < 2:
            return {}

        if smoothing_std_sec > 0:
            valid_tsds = {
                k: v.smooth(std=smoothing_std_sec, time_units="s")
                for k, v in valid_tsds.items()
            }

        ref_key = list(valid_tsds.keys())[0]
        ref_tsd = valid_tsds[ref_key]
        df_combined = pd.DataFrame(index=ref_tsd.index)

        for k, tsd in valid_tsds.items():
            # Align everything onto the reference timebase
            df_combined[k] = self._safe_value_from(ref_tsd.index.values, tsd)

        combined_frame = TsdFrame(
            t=df_combined.index.values,
            d=df_combined.values,
            columns=df_combined.columns,
        )

        correlations_by_state = {}
        for state_name in ("wake", "nrem", "rem", "microwake"):
            ep = state_intervals.get(state_name, None)
            if ep is not None:
                try:
                    state_frame = combined_frame.restrict(ep)
                    if len(state_frame) > 2:
                        correlations_by_state[state_name] = pd.DataFrame(
                            state_frame.values, columns=state_frame.columns
                        ).corr()
                    else:
                        correlations_by_state[state_name] = pd.DataFrame(
                            np.nan,
                            index=df_combined.columns,
                            columns=df_combined.columns,
                        )
                except Exception:
                    correlations_by_state[state_name] = pd.DataFrame(
                        np.nan, index=df_combined.columns, columns=df_combined.columns
                    )
            else:
                correlations_by_state[state_name] = pd.DataFrame(
                    np.nan, index=df_combined.columns, columns=df_combined.columns
                )

        return correlations_by_state

    def analyse_cross_correlations_by_state(
        self,
        tsds_to_correlate: Dict[str, Optional[Tsd]],
        state_intervals: Dict[str, Any],
        max_lag_sec: float = 2.0,
        bin_sec: float = 0.05,
        smoothing_std_sec: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes the Cross-Correlation Function (CCF Lag Arrays) to determine lead/lag relationships.

        Data Sources:
        - `tsds_to_correlate`: The NN metrics or LFPs you wish to check temporal offset for.
        """
        import itertools

        valid_tsds = {k: v for k, v in tsds_to_correlate.items() if v is not None}
        if len(valid_tsds) < 2:
            return {}

        if smoothing_std_sec > 0:
            valid_tsds = {
                k: v.smooth(std=smoothing_std_sec, time_units="s")
                for k, v in valid_tsds.items()
            }

        pairs = list(itertools.combinations(list(valid_tsds.keys()), 2))
        lags = np.arange(-max_lag_sec, max_lag_sec + bin_sec / 2, bin_sec)
        results = {}

        for k1, k2 in pairs:
            pair_key = f"{k1}_vs_{k2}"
            results[pair_key] = {"lags_sec": lags}
            tsd1, tsd2 = valid_tsds[k1], valid_tsds[k2]

            for state_name, ep in state_intervals.items():
                if ep is None:
                    continue
                df_ep = self._safe_interval_df(ep)
                if df_ep.empty:
                    continue

                t_start, t_end = df_ep["start"].min(), df_ep["end"].max()
                t_grid = np.arange(t_start, t_end + bin_sec, bin_sec)

                is_valid = np.zeros_like(t_grid, dtype=bool)
                for _, r in df_ep.iterrows():
                    is_valid |= (t_grid >= r["start"]) & (t_grid <= r["end"])

                v1_grid = self._safe_value_from(t_grid, tsd1)
                v2_grid = self._safe_value_from(t_grid, tsd2)

                v1_grid[~is_valid] = np.nan
                v2_grid[~is_valid] = np.nan

                corrs = []
                for lag in lags:
                    shift_bins = int(np.round(lag / bin_sec))

                    if shift_bins == 0:
                        v1, v2 = v1_grid, v2_grid
                    elif shift_bins > 0:
                        v1, v2 = v1_grid[:-shift_bins], v2_grid[shift_bins:]
                    else:
                        v1, v2 = v1_grid[-shift_bins:], v2_grid[:shift_bins]

                    mask = np.isfinite(v1) & np.isfinite(v2)
                    r = (
                        np.corrcoef(v1[mask], v2[mask])[0, 1]
                        if np.sum(mask) > 10
                        else np.nan
                    )
                    corrs.append(r)

                results[pair_key][state_name] = np.array(corrs)

        return results

    # =========================================================================
    # ALL PLOTTING METHODS
    # =========================================================================

    def plot_firing_rates_by_state(
        self,
        fr_data: Dict[str, Any],
        title: str = "Cell-Type Firing Rates Across Sleep States",
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Bar plot with overlaid unit scatter points for cell firing rates."""
        df_fr = fr_data.get("summary", pd.DataFrame())
        if df_fr is None or df_fr.empty:
            return None, None

        state_order = ["wake", "nrem", "rem", "microwake"]
        cell_types = [
            c for c in ["pyr", "int", "mua"] if c in df_fr["cell_type"].unique()
        ]
        palette = {"pyr": "#2C7FB8", "int": "#D7191C", "mua": "#7F7F7F"}
        type_labels = {"pyr": "Pyramidal", "int": "Interneuron", "mua": "MUA"}

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

            pos = np.arange(len(state_order))
            c = palette.get(ctype, "#333333")

            ax.bar(
                pos,
                means,
                yerr=sems,
                color=c,
                alpha=0.75,
                edgecolor="black",
                linewidth=1.0,
                capsize=4,
            )
            for i, vals in enumerate(grouped_vals):
                if vals.size:
                    ax.scatter(
                        pos[i] + np.random.uniform(-0.1, 0.1, size=vals.size),
                        vals,
                        color="black",
                        alpha=0.5,
                        s=15,
                        zorder=3,
                    )

            ax.set_xticks(pos)
            ax.set_xticklabels([s.upper() for s in state_order])
            ax.set_ylabel("Firing Rate (Hz)")
            ax.set_title(f"{type_labels.get(ctype, ctype)}")
            ax.grid(alpha=0.2, axis="y")

        fig.suptitle(title, y=1.02, fontsize=12, fontweight="bold")
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "fr_by_state.png"), dpi=300, bbox_inches="tight"
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_hypnogram_with_features(
        self,
        sleep_results: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        states = sleep_results.get("states", {})
        ripple_times = sleep_results.get("ripple", {}).get("ripple_times", np.array([]))
        primary_metric = sleep_results.get("primary_model_metric", None)
        model_tsd = sleep_results.get("model_tsds", {}).get(primary_metric, None)

        fig, (ax_hyp, ax_met) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )

        state_colors = {
            "wake": "#7F7F7F",
            "nrem": "#2C7FB8",
            "rem": "#D7191C",
            "microwake": "#FF7F00",
        }
        state_y = {"wake": 3, "rem": 2, "nrem": 1, "microwake": 2.5}

        for state_name, intervals in states.items():
            if state_name not in state_colors or intervals is None:
                continue
            df = self._safe_interval_df(intervals)
            for _, row in df.iterrows():
                ax_hyp.plot(
                    [row["start"], row["end"]],
                    [state_y[state_name]] * 2,
                    color=state_colors[state_name],
                    linewidth=8,
                    solid_capstyle="butt",
                )

        ax_hyp.set_yticks(list(state_y.values()))
        ax_hyp.set_yticklabels([k.upper() for k in state_y.keys()])
        ax_hyp.set_ylabel("State")
        ax_hyp.grid(axis="y", alpha=0.3)
        ax_hyp.set_title("Sleep Architecture & Manifold Dynamics")

        if model_tsd is not None:
            ts = np.asarray(model_tsd.index, dtype=float)
            vs = np.asarray(model_tsd.values, dtype=float)
            if vs.ndim > 1:
                vs = np.nanmean(vs, axis=1)
            ax_met.plot(
                ts, vs, color="#1A9641", linewidth=1.5, label=f"Model: {primary_metric}"
            )

        if ripple_times.size > 0:
            ax_met.vlines(
                ripple_times,
                ymin=ax_met.get_ylim()[0],
                ymax=ax_met.get_ylim()[1] * 0.1,
                color="black",
                alpha=0.5,
                linewidth=0.5,
                label="SWRs",
            )

        ax_met.set_ylabel(primary_metric or "Metric Value")
        ax_met.set_xlabel("Time (s)")
        ax_met.legend(loc="upper right", frameon=False)
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "hypnogram_w_features.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, (ax_hyp, ax_met)

    def plot_state_transition_profiles(
        self,
        sleep_results: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        states_to_plot = ["nrem", "rem", "wake"]
        metrics = [
            (
                "Ripple Rate (Hz)",
                sleep_results.get("ripple", {}).get("transition_profiles", {}),
            ),
            (
                "Reactivation",
                sleep_results.get("reactivation", {}).get("transition_profiles", {}),
            ),
        ]

        primary_metric = sleep_results.get("primary_model_metric")
        if primary_metric:
            mod_profiles = (
                sleep_results.get("model", {})
                .get(primary_metric, {})
                .get("transition_profiles", {})
            )
            metrics.append((f"Model ({primary_metric})", mod_profiles))

        fig, axes = plt.subplots(
            len(metrics),
            len(states_to_plot),
            figsize=(3.5 * len(states_to_plot), 3 * len(metrics)),
            sharex=True,
            sharey="row",
        )
        state_colors = {"wake": "#7F7F7F", "nrem": "#2C7FB8", "rem": "#D7191C"}

        for row_idx, (y_label, profiles) in enumerate(metrics):
            for col_idx, state_name in enumerate(states_to_plot):
                ax = axes[row_idx, col_idx] if len(metrics) > 1 else axes[col_idx]
                prof = profiles.get(state_name, {})
                centers = prof.get("time_sec", np.array([]))
                if len(centers) == 0:
                    continue

                if "ripple_rate_hz" in prof:
                    mean = prof["ripple_rate_hz"]
                    ax.plot(centers, mean, color=state_colors[state_name], lw=2)
                else:
                    mean = prof.get("mean", np.zeros_like(centers))
                    sem = prof.get("sem", np.zeros_like(centers))
                    ax.plot(centers, mean, color=state_colors[state_name], lw=2)
                    ax.fill_between(
                        centers,
                        mean - sem,
                        mean + sem,
                        color=state_colors[state_name],
                        alpha=0.2,
                    )

                ax.axvline(0, color="black", linestyle="--", alpha=0.5)
                if row_idx == len(metrics) - 1:
                    ax.set_xlabel(f"Time from {state_name.upper()} onset (s)")
                if col_idx == 0:
                    ax.set_ylabel(y_label)
                if row_idx == 0:
                    ax.set_title(
                        f"-> {state_name.upper()} (n={prof.get('n_onsets', 0)})"
                    )
                ax.grid(alpha=0.2)

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "state_transition_profiles.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_latent_velocity_by_state(
        self,
        sleep_results: Dict[str, Any],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        velocity_tsd = sleep_results.get("manifold", {}).get("velocity_tsd", None)
        states = sleep_results.get("states", {})
        if velocity_tsd is None:
            return None, None

        state_order = ["wake", "nrem", "rem"]
        palette = {"wake": "#7F7F7F", "nrem": "#2C7FB8", "rem": "#D7191C"}

        plot_data, labels, colors = [], [], []
        for st in state_order:
            ep = states.get(st, None)
            if ep is not None:
                try:
                    vals = velocity_tsd.restrict(ep).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        plot_data.append(vals)
                        labels.append(st.upper())
                        colors.append(palette[st])
                except Exception:
                    pass

        if not plot_data:
            return None, None

        fig, ax = plt.subplots(figsize=(6, 4))
        parts = ax.violinplot(plot_data, showmeans=True, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Trajectory Velocity (dz/dt)")
        ax.set_title("128D Manifold Velocity by Sleep State")
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "latent_velocity_by_state.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_sleep_dynamics(
        self,
        sleep_data: Dict[str, Any],
        normalize: bool = True,
        rate_smoothing_bins: int = 3,
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        if "error" in sleep_data:
            return None, None
        session = sleep_data.get("session", "postsleep")
        binned = sleep_data.get("binned_evolution", pd.DataFrame())
        if binned.empty:
            return None, None

        time_x = binned["time_from_start_min"]
        ps_start = sleep_data.get("ps_start", 0.0)
        states_ps = sleep_data.get("states_ps", {})

        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

        def _norm(series: pd.Series) -> pd.Series:
            if not normalize:
                return series
            s_min, s_max = series.min(), series.max()
            if pd.isna(s_min) or s_max == s_min:
                return series
            return (series - s_min) / (s_max - s_min)

        import matplotlib.patches as mpatches

        state_styles = {
            "wake": {"color": "#7F7F7F", "alpha": 0.15, "label": "Wake"},
            "nrem": {"color": "#2C7FB8", "alpha": 0.35, "label": "SWS (NREM)"},
            "rem": {"color": "#D7191C", "alpha": 0.15, "label": "REM"},
            "microwake": {"color": "#FF7F00", "alpha": 0.2, "label": "Microwake"},
        }

        def add_state_backgrounds(ax):
            for st_name, style in state_styles.items():
                st_ep = states_ps.get(st_name, None)
                if st_ep is None:
                    continue
                df_st = self._safe_interval_df(st_ep)
                for _, row in df_st.iterrows():
                    start_min = (row["start"] - ps_start) / 60.0
                    end_min = (row["end"] - ps_start) / 60.0
                    ax.axvspan(
                        start_min,
                        end_min,
                        color=style["color"],
                        alpha=style["alpha"],
                        lw=0,
                    )

        # 1. Ripple Quantity (Rate)
        rate_s = binned["ripple_rate_per_min"].copy()
        if rate_smoothing_bins > 1:
            rate_s = rate_s.rolling(
                window=rate_smoothing_bins, center=True, min_periods=1
            ).mean()

        axes[0].plot(time_x, rate_s, color="#2C7FB8", lw=2)
        axes[0].set_ylabel("SWRs per minute")
        axes[0].set_title(f"{session.capitalize()} Ripple Quantity Over Time")

        # 2. Ripple Quality
        qual_cols = [c for c in binned.columns if c.startswith("ripple_quality_")]
        for c in qual_cols:
            label = c.replace("ripple_quality_", "")
            axes[1].plot(
                time_x, _norm(binned[c]), marker="o", markersize=4, label=label
            )

        axes[1].set_ylabel(
            "Normalized Score" if normalize else "Mean Value inside SWRs"
        )
        axes[1].set_title("Ripple Quality / Content Evolution")
        if qual_cols:
            axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

        # 3. Continuous Signals
        cont_cols = [c for c in binned.columns if c.startswith("continuous_")]
        for c in cont_cols:
            label = c.replace("continuous_", "")
            axes[2].plot(time_x, _norm(binned[c]), lw=2, label=label)

        axes[2].set_xlabel(f"Time from {session.capitalize()} Start (minutes)")
        axes[2].set_ylabel("Normalized Score" if normalize else "Mean Metric Value")
        axes[2].set_title("Continuous Signals / Drowsiness")
        if cont_cols:
            axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

        legend_patches = [
            mpatches.Patch(color=v["color"], alpha=v["alpha"], label=v["label"])
            for v in state_styles.values()
        ]
        for ax in axes:
            add_state_backgrounds(ax)
            ax.grid(alpha=0.2)

        axes[0].legend(
            handles=legend_patches,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Sleep States",
        )

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "sleep_dynamics.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_metric_correlations_by_state(
        self,
        correlations_by_state: Dict[str, pd.DataFrame],
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Plot correlation matrices as heatmaps for each sleep state."""
        valid_states = {
            k: v
            for k, v in correlations_by_state.items()
            if not isinstance(v, dict) and not v.isna().all().all()
        }
        if not valid_states:
            return None, None

        n_states = len(valid_states)
        fig, axes = plt.subplots(1, n_states, figsize=(4 * n_states, 4))
        if n_states == 1:
            axes = [axes]

        for ax, (state_name, corr_matrix) in zip(axes, valid_states.items()):
            cax = ax.imshow(
                corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal"
            )
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(corr_matrix.index, fontsize=9)

            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    val = corr_matrix.values[i, j]
                    if pd.notna(val):
                        color = "white" if abs(val) > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=10,
                        )

            ax.set_title(state_name.upper())
            fig.colorbar(cax, ax=ax, shrink=0.8)

        fig.suptitle(
            "Cross-Metric Correlations by Sleep State (0-Lag)",
            y=1.05,
            fontweight="bold",
        )
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "metrics_corr_matrices.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_cross_correlations(
        self,
        ccf_data: Dict[str, Dict[str, Any]],
        states_to_plot: Iterable[str] = ("nrem", "wake", "rem"),
        save_dir: Optional[str] = None,
        show: bool = False,
    ):
        """Plots the lag cross-correlation functions (CCF)."""
        if not ccf_data:
            return None, None

        pairs = list(ccf_data.keys())
        fig, axes = plt.subplots(
            len(pairs),
            len(states_to_plot),
            figsize=(4 * len(states_to_plot), 3 * len(pairs)),
            squeeze=False,
        )
        state_colors = {
            "wake": "#7F7F7F",
            "nrem": "#2C7FB8",
            "rem": "#D7191C",
            "microwake": "#FF7F00",
        }

        for row_idx, pair_key in enumerate(pairs):
            k1, k2 = pair_key.split("_vs_")
            lags = ccf_data[pair_key]["lags_sec"]

            for col_idx, state_name in enumerate(states_to_plot):
                ax = axes[row_idx, col_idx]
                corrs = ccf_data[pair_key].get(state_name, None)

                if corrs is None or np.isnan(corrs).all():
                    ax.text(
                        0.5,
                        0.5,
                        "No Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                else:
                    color = state_colors.get(state_name, "black")
                    ax.plot(lags, corrs, color=color, lw=2)

                    max_idx = np.nanargmax(np.abs(corrs))
                    peak_lag, peak_corr = lags[max_idx], corrs[max_idx]
                    ax.plot(peak_lag, peak_corr, marker="o", color="black")
                    ax.vlines(
                        peak_lag, 0, peak_corr, color="black", linestyle=":", alpha=0.7
                    )

                    if abs(peak_lag) > 0.05:
                        leader = k1 if peak_lag > 0 else k2
                        ax.text(
                            0.05,
                            0.95,
                            f"{leader} leads by {abs(peak_lag):.2f}s\n(r={peak_corr:.2f})",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                        )
                    else:
                        ax.text(
                            0.05,
                            0.95,
                            f"Synchronous\n(r={peak_corr:.2f})",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                        )

                ax.axvline(0, color="gray", linestyle="--", alpha=0.8)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
                ax.grid(alpha=0.2)

                if row_idx == 0:
                    ax.set_title(state_name.upper(), fontweight="bold")
                if row_idx == len(pairs) - 1:
                    ax.set_xlabel("Lag (seconds)")
                if col_idx == 0:
                    ax.set_ylabel(f"Corr: {k1} vs {k2}\nPearson r")

        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "cross_correlations_lag.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes


def _parse_session_key(session_key: str) -> tuple[str, str]:
    """Split a session key into mouse name and manipulation label."""
    if not isinstance(session_key, str) or "_" not in session_key:
        return session_key, ""
    mouse_name, manipe = session_key.rsplit("_", 1)
    return mouse_name, manipe[:1].upper() + manipe[1:]


def _extract_number_and_manipe(mouse_name):
    """Extract the first number and the first experiment (manipe) from mouse string formatting."""
    if mouse_name.startswith("M"):
        mouse_name = mouse_name[1:]

    if "_" in mouse_name:
        parts = mouse_name.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Mouse name '{mouse_name}' does not contain enough parts to extract number and manipulator."
            )
        elif len(parts) == 2:
            if not parts[0].isdigit():
                name = "".join(filter(str.isdigit, parts[0]))
                manipe = parts[0][len(name) :]
                pseudo_manipe = parts[1]
                if manipe != pseudo_manipe:
                    warn(
                        f"Extracted manipulator '{manipe}' does not match the second part '{pseudo_manipe}'. Using '{manipe}'."
                    )
            else:
                name, manipe = parts[0], parts[1]
        elif len(parts) == 1:
            name = "".join(filter(str.isdigit, parts[0]))
            manipe = parts[0][len(name) :]
            pseudo_manipe = parts[1]
            if manipe != pseudo_manipe:
                warn(
                    f"Extracted manipulator '{manipe}' does not match the second part '{pseudo_manipe}'. Using '{manipe}'."
                )
        else:
            raise NotImplementedError(
                f"Mouse name '{mouse_name}' has more than two parts. Cannot extract number and manipulator."
            )
    return name, manipe
