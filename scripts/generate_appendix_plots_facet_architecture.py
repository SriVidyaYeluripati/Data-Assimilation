#!/usr/bin/env python3
"""
Generate redesigned appendix plots using Option A (Facet by Architecture).

This script creates clean academic-quality figures for the thesis appendix by:
- Loading metrics from all_metrics_summary.csv
- Creating faceted plots (3 vertical subplots) for each mode × architecture combination
- Showing only 3 clean lines per subplot (Baseline, FixedMean, Resample)
- Eliminating spaghetti-like plots

Outputs: 9 PNG files (3 modes × 3 architectures) saved to Report/figs/redesigned_facet_arch/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
CSV_PATH = "Report/figs/all_metrics_summary.csv"
OUTPUT_DIR = "Report/figs/redesigned_facet_arch"

ARCHS = ["mlp", "gru", "lstm"]
MODES = ["x", "xy", "x2"]

# Regime mapping (CSV has lowercase, but we want proper case for display)
REGIME_MAP = {
    "baseline": "Baseline",
    "fixedmean": "FixedMean", 
    "resample": "Resample"
}

REGIME_COLORS = {
    "Baseline": "dimgray",
    "FixedMean": "#E69F00",
    "Resample": "#0072B2"
}

# Metrics to plot (CSV column name, Display label)
METRICS = [
    ("rmse_a_mean", "Analysis RMSE"),
    ("improvement_mean", r"Improvement $(\Delta RMSE / RMSE_b)$"),
    ("hausdorff_mean", "Normalized Hausdorff Distance")
]

SIGMA_ORDER = [0.05, 0.1, 0.5, 1.0]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Add display regime names
df['regime_display'] = df['regime'].map(REGIME_MAP)

print(f"Loaded {len(df)} rows")
print(f"Regimes: {sorted(df['regime'].unique())}")
print(f"Modes: {sorted(df['mode'].unique())}")
print(f"Architectures: {sorted(df['arch'].unique())}")


def plot_mode_arch(mode, arch):
    """
    Create a faceted plot for a specific mode and architecture.
    
    Args:
        mode: observation mode (x, xy, or x2)
        arch: architecture (mlp, gru, or lstm)
    """
    # Filter data for this mode and architecture
    # Note: baseline architecture doesn't have separate mlp/gru/lstm variants
    mode_df = df[df["mode"] == mode]
    
    # For non-baseline architectures, filter by arch
    # For baseline regime, we use the baseline arch data
    arch_data = []
    for regime in ["baseline", "fixedmean", "resample"]:
        if regime == "baseline":
            # Baseline regime uses arch='baseline'
            regime_data = mode_df[(mode_df["regime"] == regime) & (mode_df["arch"] == "baseline")]
        else:
            # Other regimes use the specified architecture
            regime_data = mode_df[(mode_df["regime"] == regime) & (mode_df["arch"] == arch)]
        arch_data.append(regime_data)
    
    sub = pd.concat(arch_data, ignore_index=True)
    
    if len(sub) == 0:
        print(f"  Warning: No data found for {mode}/{arch}")
        return
    
    # Create figure with 3 vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    fig.suptitle(f"{mode.upper()} Mode — {arch.upper()} Architecture", fontsize=16, fontweight='bold')
    
    # Plot each metric
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for regime_csv in ["baseline", "fixedmean", "resample"]:
            regime_display = REGIME_MAP[regime_csv]
            
            # Get data for this regime
            tmp = sub[sub["regime"] == regime_csv]
            
            if len(tmp) == 0:
                continue
            
            # Sort by sigma to ensure proper line plotting
            tmp_sorted = tmp.sort_values("sigma")
            
            # Filter to only the sigma values we want
            tmp_sorted = tmp_sorted[tmp_sorted["sigma"].isin(SIGMA_ORDER)]
            
            if len(tmp_sorted) == 0:
                continue
            
            # Plot the line
            ax.plot(
                tmp_sorted["sigma"],
                tmp_sorted[metric],
                marker="o",
                label=regime_display,
                color=REGIME_COLORS[regime_display],
                linewidth=2,
                markersize=8
            )
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only show legend in the top subplot
        if ax == axes[0]:
            ax.legend(loc='best', frameon=True, fontsize=10)
    
    # Set x-axis label and tick marks
    axes[-1].set_xlabel(r"Observation Noise $\sigma$", fontsize=12)
    axes[-1].set_xticks(SIGMA_ORDER)
    axes[-1].set_xticklabels([f"{s:.2f}" for s in SIGMA_ORDER])
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"{mode}_{arch}_facet.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    """Generate all faceted plots."""
    print("\n" + "="*70)
    print("Generating Redesigned Appendix Plots (Facet by Architecture)")
    print("="*70 + "\n")
    
    total_plots = len(MODES) * len(ARCHS)
    plot_count = 0
    
    for mode in MODES:
        print(f"\nProcessing mode: {mode.upper()}")
        for arch in ARCHS:
            plot_count += 1
            print(f"  [{plot_count}/{total_plots}] Creating {mode}_{arch}_facet.png...")
            plot_mode_arch(mode, arch)
    
    print("\n" + "="*70)
    print(f"Successfully generated {total_plots} faceted plots!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
