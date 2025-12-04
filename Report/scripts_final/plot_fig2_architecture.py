#!/usr/bin/env python3
"""
Figure 2: Architecture Comparison (GRU, LSTM, MLP)
Shows RMSE by architecture across modes and noise levels.
Addresses Hans comments about model specification (ID 114).
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

def plot_architecture_comparison():
    """Create architecture comparison figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(RESAMPLE_CSV)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    modes = ['x', 'xy', 'x2']
    mode_labels = [r'$h(\mathbf{x}) = x_1$', r'$h(\mathbf{x}) = (x_1, x_2)$', r'$h(\mathbf{x}) = x_1^2$']
    architectures = ['gru', 'lstm', 'mlp']
    arch_labels = ['GRU', 'LSTM', 'MLP']
    colors = {'gru': '#e74c3c', 'lstm': '#3498db', 'mlp': '#2ecc71'}
    markers = {'gru': 'o', 'lstm': 's', 'mlp': '^'}
    
    for idx, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        mode_data = df[df['mode'] == mode]
        
        for arch, arch_label in zip(architectures, arch_labels):
            arch_data = mode_data[mode_data['model'] == arch]
            arch_data = arch_data.sort_values('sigma')
            
            ax.plot(arch_data['sigma'], arch_data['rmse_a'], 
                   marker=markers[arch], color=colors[arch], 
                   label=arch_label, linewidth=1.5, markersize=6)
        
        ax.set_xlabel(r'Noise level $\sigma$')
        if idx == 0:
            ax.set_ylabel('Post-assimilation RMSE')
        ax.set_title(label)
        ax.set_xscale('log')
        
        if idx == 2:
            ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig2_architecture_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_architecture_comparison()
