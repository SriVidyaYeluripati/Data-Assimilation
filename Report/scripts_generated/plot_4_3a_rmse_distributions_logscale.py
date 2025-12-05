#!/usr/bin/env python3
"""
Regenerate Figure 4_3a: Post-assimilation RMSE (Resample regime)

Hans's comments addressed:
- #105: "can you make it a log plot?" - Using log scale
- Missing model labels: Clear MLP, GRU, LSTM labels  
- Missing noise levels: σ labels on x-axis
- Observation mode clarity: Shows h(x) notation

Academic best practices:
- Grouped bar chart instead of boxplots (single values per config)
- Clear axis labels with units
- Consistent color scheme
- Legend with observation mode notation
- Error metrics clearly labeled

Output: Report/figures_new/4_3a_resample_rmse_distributions_logscale.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")
RESAMPLE_DIR = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics")

def load_resample_data():
    """Load RMSE data from resample regime."""
    csv_path = os.path.join(RESAMPLE_DIR, "notebook_eval_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def generate_figure():
    """Generate clear grouped bar chart for RMSE comparison."""
    df = load_resample_data()
    if df is None:
        print("Error: Could not load resample data")
        return
    
    # Set up the figure with publication-quality settings
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle('Post-Assimilation RMSE by Architecture and Noise Level\n'
                 '(Resample Regime — Log Scale)', fontsize=14, fontweight='bold')
    
    models = ['mlp', 'gru', 'lstm']
    model_labels = ['MLP', 'GRU', 'LSTM']
    modes = ['x', 'xy', 'x2']
    mode_labels = [r'$h(x) = x_1$', r'$h(x) = (x_1, x_2)$', r'$h(x) = x_1^2$']
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    
    # Colors for observation modes
    colors = {'x': '#3498db', 'xy': '#2ecc71', 'x2': '#e74c3c'}
    
    bar_width = 0.25
    x = np.arange(len(noise_levels))
    
    for idx, (model, label) in enumerate(zip(models, model_labels)):
        ax = axes[idx]
        
        for i, (mode, mode_label) in enumerate(zip(modes, mode_labels)):
            # Get RMSE values for this model and mode across noise levels
            rmse_values = []
            for sigma in noise_levels:
                row = df[(df['model'] == model) & (df['mode'] == mode) & (df['sigma'] == sigma)]
                if len(row) > 0:
                    rmse_values.append(row['rmse_a'].values[0])
                else:
                    rmse_values.append(np.nan)
            
            # Plot bars with offset for grouping
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, rmse_values, bar_width, 
                         label=mode_label, color=colors[mode], alpha=0.8,
                         edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, rmse_values):
                if not np.isnan(val):
                    ax.annotate(f'{val:.1f}', 
                               xy=(bar.get_x() + bar.get_width()/2, val),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=7, rotation=90)
        
        # Set log scale (Hans comment #105)
        ax.set_yscale('log')
        ax.set_ylim(1, 20)
        
        ax.set_xlabel(r'Noise Level $\sigma_{obs}$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('RMSE (log scale)', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}' for s in noise_levels])
        ax.grid(True, alpha=0.3, axis='y', which='both')
        ax.set_axisbelow(True)
        
        if idx == 2:
            ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_3a_resample_rmse_distributions_logscale.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
