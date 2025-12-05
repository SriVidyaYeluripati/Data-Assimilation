#!/usr/bin/env python3
"""
Regenerate Figure 4_6a: Background Sampling Stability

Hans's comments addressed:
- Add clear regime labels (Resample vs FixedMean)
- Show noise level progression clearly
- Emphasize stability differences

This script regenerates the ORIGINAL figure with corrections.
Original: Report/4_6a_background_sampling_stability.png
Output: Report/figures_new/4_6a_background_sampling_stability_corrected.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")

def load_resample_data():
    """Load RMSE data from resample regime."""
    csv_path = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics/notebook_eval_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load_fixedmean_data():
    """Load RMSE data from fixedmean regime."""
    csv_path = os.path.join(RESULTS_DIR, "fixedmean /run_20251008_133752 /metrics/notebook_eval_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def generate_figure():
    """Generate stability comparison plot."""
    df_resample = load_resample_data()
    df_fixedmean = load_fixedmean_data()
    
    if df_resample is None or df_fixedmean is None:
        print("Error: Could not load data")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Background Sampling Stability: RMSE vs Noise Level\n'
                 'Resample (solid) vs FixedMean (dashed) — GRU Architecture', fontsize=12)
    
    modes = ['x', 'xy', 'x2']
    mode_labels = ['$h(x) = x_1$', '$h(x) = (x_1, x_2)$', '$h(x) = x_1^2$']
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    
    for idx, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        
        # Resample data (GRU)
        resample_gru = df_resample[(df_resample['mode'] == mode) & (df_resample['model'] == 'gru')]
        resample_rmse = [resample_gru[resample_gru['sigma'] == s]['rmse_a'].values[0] 
                        for s in noise_levels if len(resample_gru[resample_gru['sigma'] == s]) > 0]
        
        # FixedMean data (GRU) - cap extreme values for visualization
        fixedmean_gru = df_fixedmean[(df_fixedmean['mode'] == mode) & (df_fixedmean['model'] == 'gru')]
        fixedmean_rmse = []
        for s in noise_levels:
            fm_data = fixedmean_gru[fixedmean_gru['sigma'] == s]['rmse_a'].values
            if len(fm_data) > 0:
                # Cap at 100 for visualization (FixedMean can have extreme failures)
                fixedmean_rmse.append(min(fm_data[0], 100))
        
        # Plot with clear labels
        if resample_rmse:
            ax.plot(noise_levels[:len(resample_rmse)], resample_rmse, 
                   'o-', color='#2ecc71', linewidth=2, markersize=8,
                   label='Resample (stable)')
        
        if fixedmean_rmse:
            ax.plot(noise_levels[:len(fixedmean_rmse)], fixedmean_rmse, 
                   's--', color='#e74c3c', linewidth=2, markersize=8,
                   label='FixedMean (unstable)')
        
        ax.set_xlabel('Noise Level $\\sigma$', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for FixedMean instability
        if mode == 'x2' and fixedmean_rmse:
            ax.annotate('Catastrophic\nfailures', 
                       xy=(0.5, 50), fontsize=8, color='#e74c3c',
                       ha='center')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_6a_background_sampling_stability_corrected.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
