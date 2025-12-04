#!/usr/bin/env python3
"""
Figure 3: FixedMean Divergence Analysis (Log Scale)
Shows the divergence behavior when using fixed climatological background.
Addresses Hans ID 105 (log scale request) and ID 107 (something wrong).
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.dirname(REPORT_DIR)

RESAMPLE_CSV = os.path.join(RESULTS_DIR, 'results', 'resample ', 'run_20251008_134240 ', 'metrics', 'notebook_eval_results.csv')
FIXEDMEAN_CSV = os.path.join(RESULTS_DIR, 'results', 'fixedmean ', 'run_20251008_133752 ', 'metrics', 'notebook_eval_results.csv')
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new_final')

def plot_fixedmean_divergence():
    """Create divergence analysis figure with log scale."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    resample = pd.read_csv(RESAMPLE_CSV)
    fixedmean = pd.read_csv(FIXEDMEAN_CSV)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = [r'$h(\mathbf{x}) = x_1$', r'$h(\mathbf{x}) = (x_1, x_2)$', r'$h(\mathbf{x}) = x_1^2$']
    
    for idx, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        
        # Resample
        res_mode = resample[resample['mode'] == mode].groupby('sigma')['rmse_a'].mean()
        # FixedMean
        fm_mode = fixedmean[fixedmean['mode'] == mode].groupby('sigma')['rmse_a'].mean()
        
        noise_levels = sorted(res_mode.index)
        
        ax.semilogy(noise_levels, [res_mode.get(n, np.nan) for n in noise_levels], 
                   'o-', color='#3498db', label='Resample', linewidth=1.5, markersize=6)
        ax.semilogy(noise_levels, [fm_mode.get(n, np.nan) for n in noise_levels], 
                   's-', color='#e74c3c', label='FixedMean', linewidth=1.5, markersize=6)
        
        ax.set_xlabel(r'Noise level $\sigma$')
        if idx == 0:
            ax.set_ylabel('Post-assimilation RMSE (log scale)')
        ax.set_title(label)
        ax.set_xscale('log')
        
        if idx == 2:
            ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig3_fixedmean_divergence_log.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_fixedmean_divergence()
