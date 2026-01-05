#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
mice_nb = [
    "M1199_PAG",
    "M994_PAG",
    "M1239_MFB",
    "M1230_Novel",
    "M1230_Known",
    "M1162_MFB",
    "M1117_MFB",
    "M1162_PAG",
    "M1168_MFB",
    "M1182_PAG",
    "M1239_PAG",
    "M1199_reversal",
    "M905",
    "M1199_MFB",
]

win_values = [0.108]
baseNameExp = "fixedChannel_concat_2Transformer_small_contrastive_groupFusion"
useStridingFactor = True
stridingFactor = 4
HPO_PROJECT_NAME = "transformer_wasserstein_tuning"
OBJECTIVE_KEY = "val_rawPosLoss_mean"

# Default N to use for final output (can be overriden by analysis)
DEFAULT_N = 5

if useStridingFactor:
    nameExp = f"STRIDE_{stridingFactor}_{baseNameExp}"
else:
    nameExp = baseNameExp
nameExpFolder = nameExp + "_Transformer"


# --- HELPERS ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_mouse_directories(base_dir="."):
    dirs = [
        os.path.abspath(os.path.join(base_dir, d))
        for d in os.listdir(base_dir)
        if os.path.isdir(d)
    ]
    valid = []
    for d in dirs:
        if any(m in d for m in mice_nb) or not mice_nb:
            if "M1199_MFB" in d:
                if os.path.exists(os.path.join(d, "exp1")):
                    valid.append(os.path.join(d, "exp1"))
                if os.path.exists(os.path.join(d, "exp2")):
                    valid.append(os.path.join(d, "exp2"))
            else:
                valid.append(d)
    return valid


def load_all_trials(tuner_dir):
    """Loads ALL trials (raw) from a directory."""
    trials = []
    if not os.path.exists(tuner_dir):
        return []

    for item in os.listdir(tuner_dir):
        path = os.path.join(tuner_dir, item)
        if os.path.isdir(path) and item.startswith("trial_"):
            json_file = os.path.join(path, "trial.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    score = data.get("score")
                    if score is None:
                        try:
                            score = data["metrics"]["metrics"][OBJECTIVE_KEY][
                                "observations"
                            ][0]["value"][0]
                        except:
                            pass

                    if score is None:
                        continue

                    trials.append(
                        {
                            "trial_id": data.get("trial_id", item),
                            "score": score,
                            **data.get("hyperparameters", {}).get("values", {}),
                        }
                    )
                except:
                    continue
    return trials


def analyze_consensus(df, top_n, silent=False):
    """Calculates consensus for a specific Top N."""
    if df.empty:
        return {}
    consensus = {}
    ignore = ["trial_id", "score", "mouse", "window"]
    hp_cols = [c for c in df.columns if c not in ignore]

    if not silent:
        print(f"\n--- Consensus for Top {top_n} Trials ---")

    for col in hp_cols:
        # Conditional Logic
        if col == "contrastive_lambda" and "contrastive_loss" in df.columns:
            subset = df[df["contrastive_loss"] == True]
        else:
            subset = df

        if subset.empty:
            continue

        counts = subset[col].value_counts(normalize=True)
        if counts.empty:
            continue

        winner = counts.index[0]
        consensus[col] = winner

        if not silent:
            win_str = f"{winner:.5f}" if isinstance(winner, float) else str(winner)
            print(f"{col:<20} : {win_str:<10} ({counts.iloc[0] * 100:.0f}%)")

    return consensus


def plot_n_best_decay(all_trials_df, win_ms):
    """
    Visualizes how the quality of trials degrades as we increase N.
    Helps choose the perfect N_BEST_TRIALS.
    """
    plt.figure(figsize=(10, 6))

    # We want to see the curve for EACH mouse to spot outliers
    mice = all_trials_df["mouse"].unique()

    max_n = 20
    summary_data = []

    for mouse in mice:
        mouse_data = all_trials_df[all_trials_df["mouse"] == mouse].sort_values("score")
        if len(mouse_data) < 2:
            continue

        # Normalize score relative to the best trial (Best = 1.0)
        # or use raw loss if comparable. Raw loss is usually fine here.
        scores = mouse_data["score"].values[:max_n]
        x_axis = range(1, len(scores) + 1)

        plt.plot(
            x_axis,
            scores,
            marker="o",
            markersize=4,
            alpha=0.4,
            label=mouse if len(mice) < 5 else "",
        )

        # Save for average curve
        for i, s in enumerate(scores):
            summary_data.append({"n": i + 1, "score": s})

    # Plot the Average Curve (The "Global Decay")
    if summary_data:
        sum_df = pd.DataFrame(summary_data)
        avg_curve = sum_df.groupby("n")["score"].mean()
        plt.plot(
            avg_curve.index,
            avg_curve.values,
            color="red",
            linewidth=3,
            label="Global Average",
        )

        # Add "Elbow" annotations
        plt.annotate(
            "Score increases rapidly here?",
            xy=(5, avg_curve.get(5, avg_curve.iloc[-1])),
            xytext=(5, avg_curve.get(5, avg_curve.iloc[-1]) + 0.5),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    plt.title(
        f"Trial Quality Decay (Win: {win_ms}ms)\nHow deep should we dig?", fontsize=14
    )
    plt.xlabel("N-th Best Trial", fontsize=12)
    plt.ylabel("Validation Loss (Lower is Better)", fontsize=12)
    plt.xticks(range(1, max_n + 1))
    plt.grid(True, alpha=0.3)
    if len(mice) < 10:
        plt.legend()

    out_file = f"Analysis_N_Best_Decay_{win_ms}.png"
    plt.savefig(out_file)
    print(f"[PLOT] Generated N-Best Decay analysis: {out_file}")
    plt.close()


def plot_deep_dive(df, win_ms):
    """Generates the Scatter and Box plots."""
    sns.set_theme(style="whitegrid")

    # 1. LR vs Loss
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="learning_rate",
        y="score",
        hue="n_transformers",
        palette="viridis",
        s=80,
        alpha=0.7,
    )
    plt.xscale("log")
    plt.title(f"Learning Rate Landscape ({win_ms}ms)", fontsize=14)
    plt.savefig(f"Analysis_LR_vs_Loss_{win_ms}.png")
    plt.close()

    # 2. Transformer Stability
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="n_transformers", y="score", palette="Blues")
    plt.title(f"Transformer Depth Stability ({win_ms}ms)", fontsize=14)
    plt.savefig(f"Analysis_Transfo_Box_{win_ms}.png")
    plt.close()


def main():
    all_mice_dirs = get_mouse_directories()

    for win in win_values:
        win_ms = str(int(win * 1000))
        print(f"\n{'#' * 40}\nProcessing Window: {win_ms}ms\n{'#' * 40}")

        all_trials_raw = []

        # 1. Load EVERYTHING first
        for mouse_dir in all_mice_dirs:
            # Check paths
            path_nested = os.path.join(
                mouse_dir,
                nameExpFolder,
                "results",
                win_ms,
                "hpo_neuro_results",
                HPO_PROJECT_NAME,
            )
            path_direct = os.path.join(
                mouse_dir,
                nameExpFolder,
                "results",
                "hpo_neuro_results",
                HPO_PROJECT_NAME,
            )
            tuner_path = path_nested if os.path.exists(path_nested) else path_direct

            trials = load_all_trials(tuner_path)
            for t in trials:
                t["mouse"] = os.path.basename(mouse_dir)
            all_trials_raw.extend(trials)

        if not all_trials_raw:
            print(f"No data for {win_ms}")
            continue

        full_df = pd.DataFrame(all_trials_raw)

        # 2. Run Visualization (Decay + Deep Dive)
        # Filter outliers for plotting clarity (98th percentile)
        plot_df = full_df[full_df["score"] <= full_df["score"].quantile(0.98)].copy()

        #  -> Helps user see where the curve breaks
        plot_n_best_decay(plot_df, win_ms)
        plot_deep_dive(plot_df, win_ms)

        # 3. Compare Consensus Stability (N=1 vs N=5 vs N=10)
        print("\nChecking Consensus Stability across N:")
        print("-" * 30)

        candidates = [1, 3, 5, 10]
        consensus_results = {}

        for n in candidates:
            # For every mouse, take top N
            top_n_frames = []
            for mouse in full_df["mouse"].unique():
                mouse_data = full_df[full_df["mouse"] == mouse].sort_values("score")
                top_n_frames.append(mouse_data.head(n))

            combined_top_n = pd.concat(top_n_frames)
            res = analyze_consensus(combined_top_n, top_n=n, silent=True)
            consensus_results[n] = res

        # Print Comparison Table
        # We grab the keys from the N=5 result as reference
        keys = consensus_results[DEFAULT_N].keys()
        row_fmt = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10}"
        print(row_fmt.format("PARAM", "N=1", "N=3", "N=5", "N=10"))
        print("-" * 70)

        for k in keys:
            vals = []
            for n in candidates:
                val = consensus_results[n].get(k, "-")
                if isinstance(val, float):
                    val = f"{val:.5f}"
                vals.append(str(val))
            print(row_fmt.format(k, *vals))

        # 4. Final Output using DEFAULT_N (or you can change logic to pick best N)
        print(f"\nGenerating Final Config using N={DEFAULT_N}...")
        final_hps = consensus_results[DEFAULT_N]

        out_name = f"consensus_HPs_{win_ms}ms.json"
        with open(out_name, "w") as f:
            json.dump(final_hps, f, indent=4, cls=NumpyEncoder)
        print(f"Saved: {out_name}")


if __name__ == "__main__":
    main()
