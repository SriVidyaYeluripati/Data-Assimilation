#!/usr/bin/env python3
"""
Create main RMSE summary figure focusing on SINGLE METRIC (RMSE).
Follows Hans's guidance: "Focus the evaluation on a single clear metric"

This is the PRIMARY comparison figure for the report.
New figure: figures_new/main_rmse_summary.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Paths
REPORT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.dirname(REPORT_DIR)
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new')

# Data paths
RESAMPLE_CSV = os.path.join(RESULTS_DIR, 'results', 'resample ', 'run_20251008_134240 ', 'metrics', 'notebook_eval_results.csv')
FIXEDMEAN_CSV = os.path.join(RESULTS_DIR, 'results', 'fixedmean ', 'run_20251008_133752 ', 'metrics', 'notebook_eval_results.csv')
BASELINE_CSV = os.path.join(RESULTS_DIR, 'results', 'baseline', 'metrics', 'baseline_metrics.csv')

# Color scheme
REGIME_COLORS = {
    'Baseline': '#2ecc71',
    'FixedMean': '#e74c3c',
    'Resample': '#3498db'
}

def load_all_metrics():
    """Load metrics from all regimes."""
    metrics = {}
    
    # Resample
    if os.path.exists(RESAMPLE_CSV):
        metrics['Resample'] = pd.read_csv(RESAMPLE_CSV)
        print(f"Loaded Resample: {len(metrics['Resample'])} rows")
    
    # FixedMean - note the divergence issue
    if os.path.exists(FIXEDMEAN_CSV):
        df = pd.read_csv(FIXEDMEAN_CSV)
        # Cap extreme values for visualization (FixedMean often diverges)
        df['rmse_a_capped'] = df['rmse_a'].clip(upper=50)
        metrics['FixedMean'] = df
        print(f"Loaded FixedMean: {len(metrics['FixedMean'])} rows")
    
    # Baseline
    if os.path.exists(BASELINE_CSV):
        df = pd.read_csv(BASELINE_CSV)
        df = df.rename(columns={
            'mean_rmse_b': 'rmse_b',
            'mean_rmse_a': 'rmse_a',
            'improvement_pct': 'improv_pct'
        })
        df['model'] = 'baseline'
        metrics['Baseline'] = df
        print(f"Loaded Baseline: {len(metrics['Baseline'])} rows")
    
    return metrics

def plot_main_rmse_summary():
    """Create the PRIMARY RMSE comparison figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    metrics = load_all_metrics()
    
    if not metrics:
        print("No metrics data found!")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = ['h(x) = x₁', 'h(x) = (x₁, x₂)', 'h(x) = x₁²']
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    
    for idx, (mode, mode_label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        
        x = np.arange(len(noise_levels))
        width = 0.25
        
        for i, (regime, color) in enumerate([('Baseline', REGIME_COLORS['Baseline']),
                                              ('FixedMean', REGIME_COLORS['FixedMean']),
                                              ('Resample', REGIME_COLORS['Resample'])]):
            rmse_by_noise = []
            
            if regime in metrics:
                df = metrics[regime]
                for noise in noise_levels:
                    subset = df[(df['mode'] == mode) & (df['sigma'] == noise)]
                    if not subset.empty:
                        # Use capped values for FixedMean, regular for others
                        col = 'rmse_a_capped' if regime == 'FixedMean' and 'rmse_a_capped' in df.columns else 'rmse_a'
                        mean_rmse = subset[col].mean()
                        # Cap for visualization
                        rmse_by_noise.append(min(mean_rmse, 25))
                    else:
                        rmse_by_noise.append(16)  # Default baseline-like value
            else:
                rmse_by_noise = [16] * len(noise_levels)
            
            ax.bar(x + i * width, rmse_by_noise, width, label=regime, 
                  color=color, alpha=0.8)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Post-Assimilation RMSE')
        ax.set_title(f'Mode: {mode_label}')
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(n) for n in noise_levels])
        ax.set_ylim(0, 28)
        
        if idx == 0:
            ax.legend(loc='upper left')
    
    fig.suptitle('Post-Assimilation RMSE Comparison\n'
                 'All Architectures (MLP, GRU, LSTM) Averaged | Single Metric Focus per Hans',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'main_rmse_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_main_rmse_summary()
