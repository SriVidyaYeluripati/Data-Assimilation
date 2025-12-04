#!/usr/bin/env python3
"""
Figure 4: Improvement Percentage Analysis
Shows the relative improvement (or degradation) from assimilation.
Addresses negative improvement findings and Hans's emphasis on RMSE.
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
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new_final')

def plot_improvement_analysis():
    """Create improvement percentage analysis figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(RESAMPLE_CSV)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = [r'$x_1$', r'$(x_1, x_2)$', r'$x_1^2$']
    architectures = ['gru', 'lstm', 'mlp']
    
    x_positions = []
    colors = []
    labels = []
    improvements = []
    
    pos = 0
    for mode, mode_label in zip(modes, mode_labels):
        mode_data = df[df['mode'] == mode]
        for arch in architectures:
            arch_data = mode_data[mode_data['model'] == arch]
            mean_improv = arch_data['improv_pct'].mean()
            
            x_positions.append(pos)
            improvements.append(mean_improv)
            labels.append(f'{mode_label}\n{arch.upper()}')
            colors.append('#2ecc71' if mean_improv > 0 else '#e74c3c')
            pos += 1
        pos += 0.5  # Gap between modes
    
    bars = ax.bar(x_positions, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Mean improvement (%)')
    ax.set_xlabel('Observation mode and architecture')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('Improvement in RMSE from Assimilation (Resample Regime)')
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height > 0 else -10),
                   textcoords="offset points",
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=7)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig4_improvement_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_improvement_analysis()
