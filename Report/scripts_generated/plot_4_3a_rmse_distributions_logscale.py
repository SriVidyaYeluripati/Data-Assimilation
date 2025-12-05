#!/usr/bin/env python3
"""
Regenerate Figure 4_3a: Post-assimilation RMSE distributions (Resample regime)

Hans's comments addressed:
- #105: "can you make it a log plot?" - Yes, using log scale on y-axis
- Missing model labels: Added MLP, GRU, LSTM labels
- Missing noise levels: Added σ = 0.05, 0.1, 0.5, 1.0 labels
- Observation mode clarity: Shows h(x) = x₁ mode

This script regenerates the ORIGINAL figure with corrections.
Original: Report/4_3a_resample_rmse_distributions.png
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

# Load resample results
RESAMPLE_DIR = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics")

def load_resample_data():
    """Load RMSE data from resample regime."""
    csv_path = os.path.join(RESAMPLE_DIR, "notebook_eval_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def generate_figure():
    """Generate log-scale boxplot for RMSE distributions."""
    df = load_resample_data()
    if df is None:
        print("Error: Could not load resample data")
        return
    
    # Create figure with 3 subplots (one per architecture)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Post-Assimilation RMSE Distributions (Resample Regime)\n'
                 'Observation Mode: $h(x) = x_1$ — Log Scale', fontsize=12)
    
    models = ['mlp', 'gru', 'lstm']
    model_labels = ['MLP', 'GRU', 'LSTM']
    noise_levels = [0.05, 0.1, 0.5, 1.0]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (model, label) in enumerate(zip(models, model_labels)):
        ax = axes[idx]
        
        # Filter data for this model and x mode (partial observation)
        model_data = df[(df['model'] == model) & (df['mode'] == 'x')]
        
        # Collect RMSE values for each noise level
        box_data = []
        positions = []
        for i, sigma in enumerate(noise_levels):
            sigma_data = model_data[model_data['sigma'] == sigma]['rmse_a'].values
            if len(sigma_data) > 0:
                # Since we have single values, replicate for boxplot visualization
                # In real scenario, this would come from multiple runs
                box_data.append(sigma_data)
                positions.append(i + 1)
        
        # Create boxplot
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Set log scale on y-axis (Hans comment #105)
        ax.set_yscale('log')
        
        ax.set_xlabel('Noise Level $\\sigma$', fontsize=10)
        ax.set_ylabel('RMSE (log scale)', fontsize=10)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, len(noise_levels) + 1))
        ax.set_xticklabels([f'{s}' for s in noise_levels])
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1, 100)  # Reasonable range for log scale
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_3a_resample_rmse_distributions_logscale.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
