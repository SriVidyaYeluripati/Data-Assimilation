#!/usr/bin/env python3
"""
Regenerate Figure 4_4a: Post-assimilation RMSE by Observation Mode

Hans's comments addressed:
- Log scale for y-axis
- Clear h(x) notation for observation operators
- Model comparison across modes

Academic best practices:
- Grouped bar chart for clear comparison
- Mathematical notation for observation operators
- Consistent color scheme per architecture
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")
RESAMPLE_DIR = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics")

def load_resample_data():
    csv_path = os.path.join(RESAMPLE_DIR, "notebook_eval_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def generate_figure():
    df = load_resample_data()
    if df is None:
        print("Error: Could not load resample data")
        return
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle('Post-Assimilation RMSE by Observation Mode\n'
                 '(Resample Regime — Log Scale)', fontsize=14, fontweight='bold')
    
    models = ['mlp', 'gru', 'lstm']
    model_labels = ['MLP', 'GRU', 'LSTM']
    modes = ['x', 'xy', 'x2']
    mode_labels = [r'$h(x)=x_1$', r'$h(x)=(x_1,x_2)$', r'$h(x)=x_1^2$']
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    
    colors = {'mlp': '#e74c3c', 'gru': '#3498db', 'lstm': '#2ecc71'}
    bar_width = 0.25
    x = np.arange(len(modes))
    
    for idx, sigma in enumerate(noise_levels):
        ax = axes[idx]
        
        for i, (model, label) in enumerate(zip(models, model_labels)):
            rmse_values = []
            for mode in modes:
                row = df[(df['model'] == model) & (df['mode'] == mode) & (df['sigma'] == sigma)]
                if len(row) > 0:
                    rmse_values.append(row['rmse_a'].values[0])
                else:
                    rmse_values.append(np.nan)
            
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, rmse_values, bar_width, 
                         label=label, color=colors[model], alpha=0.8,
                         edgecolor='black', linewidth=0.5)
        
        ax.set_yscale('log')
        ax.set_ylim(1, 20)
        ax.set_xlabel('Observation Operator', fontsize=11)
        if idx == 0:
            ax.set_ylabel('RMSE (log scale)', fontsize=11)
        ax.set_title(f'$\\sigma_{{obs}} = {sigma}$', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.grid(True, alpha=0.3, axis='y', which='both')
        ax.set_axisbelow(True)
        
        if idx == 3:
            ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_4a_post_assimilation_rmse_logscale.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
