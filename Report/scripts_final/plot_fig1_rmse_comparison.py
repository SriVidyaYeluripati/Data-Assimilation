#!/usr/bin/env python3
"""
Figure 1: Post-Assimilation RMSE Comparison
Compares Resample, FixedMean, and Baseline regimes across observation modes.
Addresses Hans ID 105 (log scale) and ID 111 (consolidation).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'font.family': 'serif'
})

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.dirname(REPORT_DIR)

RESAMPLE_CSV = os.path.join(RESULTS_DIR, 'results', 'resample ', 'run_20251008_134240 ', 'metrics', 'notebook_eval_results.csv')
FIXEDMEAN_CSV = os.path.join(RESULTS_DIR, 'results', 'fixedmean ', 'run_20251008_133752 ', 'metrics', 'notebook_eval_results.csv')
BASELINE_CSV = os.path.join(RESULTS_DIR, 'results', 'baseline', 'metrics', 'baseline_metrics.csv')
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new_final')

def load_data():
    """Load all experimental data."""
    resample = pd.read_csv(RESAMPLE_CSV)
    fixedmean = pd.read_csv(FIXEDMEAN_CSV)
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline.rename(columns={'mean_rmse_a': 'rmse_a', 'mean_rmse_b': 'rmse_b'})
    return resample, fixedmean, baseline

def plot_rmse_comparison():
    """Create consolidated RMSE comparison figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    resample, fixedmean, baseline = load_data()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = [r'$h(\mathbf{x}) = x_1$', r'$h(\mathbf{x}) = (x_1, x_2)$', r'$h(\mathbf{x}) = x_1^2$']
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    
    colors = {'Resample': '#3498db', 'Baseline': '#2ecc71'}
    
    for idx, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        
        # Resample data
        res_mode = resample[resample['mode'] == mode]
        res_by_noise = res_mode.groupby('sigma')['rmse_a'].mean()
        
        # Baseline data
        bl_mode = baseline[baseline['mode'] == mode]
        bl_by_noise = bl_mode.groupby('sigma')['rmse_a'].mean()
        
        x = np.arange(len(noise_levels))
        width = 0.35
        
        # Plot bars
        res_vals = [res_by_noise.get(n, np.nan) for n in noise_levels]
        bl_vals = [bl_by_noise.get(n, np.nan) for n in noise_levels]
        
        ax.bar(x - width/2, res_vals, width, label='Resample', color=colors['Resample'], alpha=0.8)
        ax.bar(x + width/2, bl_vals, width, label='Baseline', color=colors['Baseline'], alpha=0.8)
        
        ax.set_xlabel(r'Noise level $\sigma$')
        if idx == 0:
            ax.set_ylabel('Post-assimilation RMSE')
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in noise_levels])
        ax.set_ylim(0, 20)
        
        if idx == 2:
            ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig1_rmse_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_rmse_comparison()
