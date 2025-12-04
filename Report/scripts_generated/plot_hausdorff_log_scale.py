#!/usr/bin/env python3
"""
Plot Hausdorff distance metrics with LOG SCALE y-axis.
Addresses Hans's comment ID 105: "can you make it a log plot?"

Original figure: attractor_metrics_resample_fixedmean.png
New figure: figures_new/hausdorff_log_scale_resample_fixedmean.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

# Paths
REPORT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(REPORT_DIR, 'geometry_metrics_resample_fixedmean.csv')
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new')

# Color scheme
REGIME_COLORS = {
    'Resample': '#3498db',   # Blue
    'FixedMean': '#e74c3c',  # Red
    'Baseline': '#2ecc71'    # Green
}

NOISE_MARKERS = {0.05: 'o', 0.1: 's', 0.5: '^', 1.0: 'D'}

def load_data():
    """Load geometry metrics data."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} rows from {DATA_FILE}")
        return df
    else:
        # Try global file
        global_file = os.path.join(REPORT_DIR, 'geometry_global_metrics_all.csv')
        if os.path.exists(global_file):
            df = pd.read_csv(global_file)
            # Filter to Resample and FixedMean
            df = df[df['regime'].isin(['Resample', 'FixedMean'])]
            print(f"Loaded {len(df)} rows from {global_file}")
            return df
    raise FileNotFoundError(f"Could not find geometry metrics data")

def plot_hausdorff_log_scale():
    """Create Hausdorff distance plot with LOG SCALE y-axis."""
    df = load_data()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    modes = ['x', 'xy', 'x2']
    mode_labels = ['h(x) = x₁', 'h(x) = (x₁, x₂)', 'h(x) = x₁²']
    
    for idx, (mode, mode_label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        
        for regime in ['Resample', 'FixedMean']:
            regime_data = df[(df['regime'] == regime) & (df['mode'] == mode)]
            
            if regime_data.empty:
                continue
            
            # Group by noise level and average across models
            grouped = regime_data.groupby('sigma')['H_norm_global'].agg(['mean', 'std']).reset_index()
            
            noise_levels = grouped['sigma'].values
            means = grouped['mean'].values
            stds = grouped['std'].values
            
            # Use LOG SCALE - ensure positive values
            means = np.maximum(means, 1e-6)  # Avoid log(0)
            
            ax.errorbar(noise_levels, means, yerr=stds, 
                       label=regime, color=REGIME_COLORS[regime],
                       marker='o', markersize=8, linewidth=2, capsize=4, capthick=1.5)
        
        # LOG SCALE Y-AXIS - Hans's request
        ax.set_yscale('log')
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Normalized Hausdorff Distance' if idx == 0 else '')
        ax.set_title(f'Mode: {mode_label}')
        ax.set_xticks([0.05, 0.1, 0.5, 1.0])
        ax.set_xticklabels(['0.05', '0.1', '0.5', '1.0'])
        
        if idx == 0:
            ax.legend(loc='upper left')
    
    fig.suptitle('Attractor Geometry: Normalized Hausdorff Distance (Log Scale)\n'
                 'Resample vs FixedMean Regimes | All Architectures Aggregated',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'hausdorff_log_scale_resample_fixedmean.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_hausdorff_log_scale()
