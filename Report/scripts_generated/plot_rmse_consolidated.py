#!/usr/bin/env python3
"""
Create CONSOLIDATED RMSE plot combining multiple figures into one.
Addresses Hans's comment ID 111: "Maybe keep one plot here"

Original figures: 4_4a_post_assimilation_rmse.png, 4_4b_delta_rmse_improvement.png
New figure: figures_new/rmse_consolidated.png
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

ARCH_COLORS = {
    'mlp': '#9b59b6',
    'gru': '#e67e22',
    'lstm': '#1abc9c'
}

def load_all_metrics():
    """Load metrics from all regimes."""
    metrics = {}
    
    # Resample
    if os.path.exists(RESAMPLE_CSV):
        metrics['Resample'] = pd.read_csv(RESAMPLE_CSV)
        print(f"Loaded Resample: {len(metrics['Resample'])} rows")
    
    # FixedMean
    if os.path.exists(FIXEDMEAN_CSV):
        metrics['FixedMean'] = pd.read_csv(FIXEDMEAN_CSV)
        print(f"Loaded FixedMean: {len(metrics['FixedMean'])} rows")
    
    # Baseline
    if os.path.exists(BASELINE_CSV):
        df = pd.read_csv(BASELINE_CSV)
        # Normalize column names
        df = df.rename(columns={
            'mean_rmse_b': 'rmse_b',
            'mean_rmse_a': 'rmse_a',
            'improvement_pct': 'improv_pct'
        })
        df['model'] = 'baseline'
        metrics['Baseline'] = df
        print(f"Loaded Baseline: {len(metrics['Baseline'])} rows")
    
    return metrics

def plot_rmse_consolidated():
    """Create consolidated RMSE figure with subplots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    metrics = load_all_metrics()
    
    if not metrics:
        print("No metrics data found!")
        return None
    
    # Create consolidated figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot (a): Post-assimilation RMSE by mode
    ax = axes[0, 0]
    modes = ['x', 'xy', 'x2']
    mode_labels = ['x', 'xy', 'x²']
    x_pos = np.arange(len(modes))
    width = 0.25
    
    for i, (regime, color) in enumerate([('Baseline', REGIME_COLORS['Baseline']), 
                                          ('FixedMean', REGIME_COLORS['FixedMean']),
                                          ('Resample', REGIME_COLORS['Resample'])]):
        if regime in metrics:
            df = metrics[regime]
            rmse_by_mode = []
            for mode in modes:
                subset = df[df['mode'] == mode]
                if not subset.empty:
                    # Average across noise levels and models, but cap extreme values
                    mean_rmse = subset['rmse_a'].mean()
                    rmse_by_mode.append(min(mean_rmse, 20))  # Cap at 20 for visualization
                else:
                    rmse_by_mode.append(15)
            ax.bar(x_pos + i * width, rmse_by_mode, width, label=regime, color=color, alpha=0.8)
    
    ax.set_xlabel('Observation Mode')
    ax.set_ylabel('Post-Assimilation RMSE')
    ax.set_title('(a) RMSE by Observation Mode\n(averaged across noise levels & architectures)')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(mode_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 25)
    
    # Subplot (b): Improvement percentage by mode
    ax = axes[0, 1]
    for i, (regime, color) in enumerate([('FixedMean', REGIME_COLORS['FixedMean']),
                                          ('Resample', REGIME_COLORS['Resample'])]):
        if regime in metrics:
            df = metrics[regime]
            improv_by_mode = []
            for mode in modes:
                subset = df[df['mode'] == mode]
                if not subset.empty and 'improv_pct' in subset.columns:
                    # Average improvement
                    mean_improv = subset['improv_pct'].mean()
                    improv_by_mode.append(mean_improv)
                else:
                    improv_by_mode.append(0)
            offset = 0 if regime == 'FixedMean' else width
            ax.bar(x_pos + offset, improv_by_mode, width, label=regime, color=color, alpha=0.8)
    
    ax.set_xlabel('Observation Mode')
    ax.set_ylabel('RMSE Improvement (%)')
    ax.set_title('(b) Improvement Over Background\n(positive = better)')
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels(mode_labels)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right')
    
    # Subplot (c): RMSE by noise level (Resample only)
    ax = axes[1, 0]
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    if 'Resample' in metrics:
        df = metrics['Resample']
        for mode, color, marker in zip(modes, ['#3498db', '#2ecc71', '#e74c3c'], ['o', 's', '^']):
            rmse_by_noise = []
            for noise in noise_levels:
                subset = df[(df['mode'] == mode) & (df['sigma'] == noise)]
                if not subset.empty:
                    rmse_by_noise.append(subset['rmse_a'].mean())
                else:
                    rmse_by_noise.append(10)
            ax.plot(noise_levels, rmse_by_noise, marker=marker, color=color, 
                   label=f'{mode}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Post-Assimilation RMSE')
    ax.set_title('(c) Noise Sensitivity (Resample Regime)\nAll architectures averaged')
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    
    # Subplot (d): Architecture comparison
    ax = axes[1, 1]
    architectures = ['mlp', 'gru', 'lstm']
    if 'Resample' in metrics:
        df = metrics['Resample']
        for i, arch in enumerate(architectures):
            subset = df[df['model'] == arch]
            if not subset.empty:
                # RMSE before and after
                rmse_b = subset['rmse_b'].mean()
                rmse_a = subset['rmse_a'].mean()
                ax.bar(i - 0.15, rmse_b, 0.3, label='Before' if i == 0 else '', 
                      color='#95a5a6', alpha=0.7)
                ax.bar(i + 0.15, rmse_a, 0.3, label='After' if i == 0 else '', 
                      color=ARCH_COLORS[arch], alpha=0.8)
    
    ax.set_xlabel('Architecture')
    ax.set_ylabel('RMSE')
    ax.set_title('(d) Before vs After Assimilation\n(Resample, all modes & noise levels)')
    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels(['MLP', 'GRU', 'LSTM'])
    ax.legend(loc='upper right')
    
    # Main title
    fig.suptitle('Consolidated RMSE Analysis\n'
                 '(Multiple metrics combined into single figure)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'rmse_consolidated.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_rmse_consolidated()
