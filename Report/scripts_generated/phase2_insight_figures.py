#!/usr/bin/env python3
"""
Phase 2: Generate robust insight figures

This script generates figures that correctly represent the experimental results
using both RMSE and RMdSE metrics as appropriate.

Key insights from Hans's meeting:
1. RMSE is susceptible to outliers - use RMdSE (Root Median SE) for robustness
2. FixedMean has 70-80% trajectory dropout (actually 94% in our data!)
3. The expected ordering of observation modes should be verified
4. Results can be inconclusive - that's okay

Figures generated:
1. RMSE comparison with proper sample sizes shown (no misleading aggregation)
2. Regime stability comparison (Resample vs FixedMean)
3. Observation mode difficulty ranking
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")

def load_data():
    """Load all experimental data."""
    resample_path = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics/notebook_eval_results.csv")
    fixedmean_path = os.path.join(RESULTS_DIR, "fixedmean /run_20251008_133752 /metrics/notebook_eval_results.csv")
    
    df_resample = pd.read_csv(resample_path)
    df_fixedmean = pd.read_csv(fixedmean_path)
    
    return df_resample, df_fixedmean

def compute_rmdse(values):
    """Compute Root Median Squared Error (robust to outliers)."""
    return np.sqrt(np.median(values**2))

def compute_rmse(values):
    """Compute Root Mean Squared Error (standard)."""
    return np.sqrt(np.mean(values**2))

def generate_figure_1_resample_rmse_by_mode():
    """
    Figure 1: RMSE by observation mode for Resample regime
    
    Shows the ordering x < xy < x² with proper error bars.
    Uses LOG SCALE as Hans requested.
    """
    df_resample, _ = load_data()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = ['$h(x) = x_1$\n(Linear Partial)', 
                   '$h(x) = (x_1, x_2)$\n(Bilinear Coupled)', 
                   '$h(x) = x_1^2$\n(Nonlinear)']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    positions = np.arange(len(modes))
    width = 0.25
    
    for i, model in enumerate(['mlp', 'gru', 'lstm']):
        means = []
        stds = []
        for mode in modes:
            data = df_resample[(df_resample['mode'] == mode) & (df_resample['model'] == model)]['rmse_a']
            means.append(data.mean())
            stds.append(data.std())
        
        offset = (i - 1) * width
        bars = ax.bar(positions + offset, means, width, 
                     yerr=stds, capsize=3,
                     label=model.upper(), alpha=0.8)
    
    ax.set_ylabel('RMSE (log scale)', fontsize=11)
    ax.set_xlabel('Observation Operator', fontsize=11)
    ax.set_title('Post-Assimilation RMSE by Observation Mode\n(Resample Regime, All Noise Levels)', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(mode_labels)
    ax.set_yscale('log')
    ax.legend(title='Architecture')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add annotation about ordering
    ax.annotate('Note: x < xy < x² ordering reflects learning difficulty,\nnot information content',
               xy=(0.5, 0.02), xycoords='axes fraction',
               fontsize=9, style='italic', ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "fig_resample_rmse_by_mode.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def generate_figure_2_regime_comparison():
    """
    Figure 2: Resample vs FixedMean regime comparison
    
    Shows stability difference - FixedMean has 94% failures!
    Uses RMdSE for FixedMean to handle outliers.
    """
    df_resample, df_fixedmean = load_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Success/Failure rates
    ax1 = axes[0]
    
    # Count failures (RMSE > 100)
    fm_failures = len(df_fixedmean[df_fixedmean['rmse_a'] > 100])
    fm_success = len(df_fixedmean) - fm_failures
    rs_failures = len(df_resample[df_resample['rmse_a'] > 100])
    rs_success = len(df_resample) - rs_failures
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [rs_success, fm_success], width, label='Success', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, [rs_failures, fm_failures], width, label='Failure (RMSE>100)', color='#e74c3c')
    
    ax1.set_ylabel('Number of Configurations', fontsize=11)
    ax1.set_xlabel('Training Regime', fontsize=11)
    ax1.set_title('Regime Stability: Success vs Failure Rates\n(out of 36 configurations each)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Resample', 'FixedMean'])
    ax1.legend()
    
    # Add percentage labels
    for bar, val, total in zip(bars1, [rs_success, fm_success], [36, 36]):
        ax1.annotate(f'{100*val/total:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    for bar, val, total in zip(bars2, [rs_failures, fm_failures], [36, 36]):
        if val > 0:
            ax1.annotate(f'{100*val/total:.0f}%', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    # Right plot: RMSE comparison (only successful runs)
    ax2 = axes[1]
    
    # Filter successful runs for FixedMean
    df_fm_success = df_fixedmean[df_fixedmean['rmse_a'] < 100]
    
    modes = ['x', 'xy', 'x2']
    positions = np.arange(len(modes))
    width = 0.35
    
    rs_means = [df_resample[df_resample['mode'] == m]['rmse_a'].mean() for m in modes]
    fm_means = [df_fm_success[df_fm_success['mode'] == m]['rmse_a'].mean() if len(df_fm_success[df_fm_success['mode'] == m]) > 0 else np.nan for m in modes]
    
    bars1 = ax2.bar(positions - width/2, rs_means, width, label='Resample (n=36)', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(positions + width/2, fm_means, width, label=f'FixedMean (n={len(df_fm_success)})', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('RMSE (successful runs only)', fontsize=11)
    ax2.set_xlabel('Observation Mode', fontsize=11)
    ax2.set_title('RMSE Comparison\n(FixedMean shows only non-diverged runs)', fontsize=12)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['$x_1$', '$(x_1,x_2)$', '$x_1^2$'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add warning annotation
    ax2.annotate('⚠ FixedMean comparison misleading:\n94% of runs diverged',
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "fig_regime_stability_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def generate_figure_3_noise_sensitivity():
    """
    Figure 3: RMSE vs Noise Level for Resample regime
    
    Shows how performance degrades with noise.
    """
    df_resample, _ = load_data()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    models = ['mlp', 'gru', 'lstm']
    model_colors = {'mlp': '#3498db', 'gru': '#2ecc71', 'lstm': '#e74c3c'}
    modes = ['x', 'xy', 'x2']
    mode_titles = ['$h(x) = x_1$', '$h(x) = (x_1, x_2)$', '$h(x) = x_1^2$']
    
    for idx, (mode, title) in enumerate(zip(modes, mode_titles)):
        ax = axes[idx]
        
        for model in models:
            rmses = []
            for sigma in noise_levels:
                data = df_resample[(df_resample['mode'] == mode) & 
                                   (df_resample['model'] == model) & 
                                   (df_resample['sigma'] == sigma)]['rmse_a']
                rmses.append(data.values[0] if len(data) > 0 else np.nan)
            
            ax.plot(noise_levels, rmses, 'o-', color=model_colors[model], 
                   label=model.upper(), linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level $\\sigma$', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Post-Assimilation RMSE vs Noise Level (Resample Regime)', fontsize=12, y=1.02)
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "fig_noise_sensitivity.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def generate_figure_4_boxplot_log_scale():
    """
    Figure 4: Box plots with LOG SCALE as Hans requested
    
    Shows RMSE distribution for each observation mode.
    """
    df_resample, _ = load_data()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    modes = ['x', 'xy', 'x2']
    mode_titles = ['$h(x) = x_1$', '$h(x) = (x_1, x_2)$', '$h(x) = x_1^2$']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for idx, (mode, title, color) in enumerate(zip(modes, mode_titles, colors)):
        ax = axes[idx]
        
        # Collect data for each architecture
        data_by_arch = []
        labels = []
        for model in ['mlp', 'gru', 'lstm']:
            data = df_resample[(df_resample['mode'] == mode) & 
                              (df_resample['model'] == model)]['rmse_a'].values
            data_by_arch.append(data)
            labels.append(model.upper())
        
        bp = ax.boxplot(data_by_arch, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_yscale('log')
        ax.set_ylabel('RMSE (log scale)', fontsize=10)
        ax.set_xlabel('Architecture', fontsize=10)
        ax.set_title(f'{title}\n(n=4 per architecture)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('RMSE Distribution by Architecture (Resample Regime, Log Scale)', fontsize=12, y=1.02)
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "fig_boxplot_log_scale.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def generate_figure_5_rmse_vs_rmdse():
    """
    Figure 5: RMSE vs RMdSE comparison
    
    Shows when each metric is appropriate:
    - RMSE for Resample (no outliers)
    - RMdSE for FixedMean (many outliers)
    """
    df_resample, df_fixedmean = load_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = ['$x_1$', '$(x_1,x_2)$', '$x_1^2$']
    
    # Left: Resample (RMSE ≈ RMdSE because no outliers)
    ax1 = axes[0]
    rmse_vals = []
    rmdse_vals = []
    for mode in modes:
        data = df_resample[df_resample['mode'] == mode]['rmse_a'].values
        rmse_vals.append(compute_rmse(data))
        rmdse_vals.append(compute_rmdse(data))
    
    x = np.arange(len(modes))
    width = 0.35
    ax1.bar(x - width/2, rmse_vals, width, label='RMSE', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, rmdse_vals, width, label='RMdSE', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Error Metric Value', fontsize=10)
    ax1.set_xlabel('Observation Mode', fontsize=10)
    ax1.set_title('Resample Regime\n(RMSE ≈ RMdSE, no outliers)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: FixedMean (RMSE >> RMdSE because many outliers)
    ax2 = axes[1]
    rmse_vals = []
    rmdse_vals = []
    for mode in modes:
        data = df_fixedmean[df_fixedmean['mode'] == mode]['rmse_a'].values
        rmse_vals.append(compute_rmse(data))
        rmdse_vals.append(compute_rmdse(data))
    
    # Use log scale for FixedMean due to huge values
    ax2.bar(x - width/2, rmse_vals, width, label='RMSE', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, rmdse_vals, width, label='RMdSE', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Error Metric Value (log scale)', fontsize=10)
    ax2.set_xlabel('Observation Mode', fontsize=10)
    ax2.set_title('FixedMean Regime\n(Both metrics ~same due to 94% failures)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_labels)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('RMSE vs RMdSE: When to Use Each Metric', fontsize=12, y=1.02)
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "fig_rmse_vs_rmdse.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    """Generate all Phase 2 insight figures."""
    print("=" * 60)
    print("PHASE 2: GENERATING ROBUST INSIGHT FIGURES")
    print("=" * 60)
    
    generate_figure_1_resample_rmse_by_mode()
    generate_figure_2_regime_comparison()
    generate_figure_3_noise_sensitivity()
    generate_figure_4_boxplot_log_scale()
    generate_figure_5_rmse_vs_rmdse()
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE: 5 insight figures generated")
    print("=" * 60)

if __name__ == "__main__":
    main()
