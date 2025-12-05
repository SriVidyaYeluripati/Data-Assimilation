#!/usr/bin/env python3
"""
Regenerate Figure 4_5a: Trajectory Fidelity Comparison

Hans's comments addressed:
- Add clear Truth/Analysis/Background labels
- Show observation mode clearly
- Add noise level annotation

This script regenerates the ORIGINAL figure with corrections.
Original: Report/4_5a_trajectory_fidelity_comparison.png
Output: Report/figures_new/4_5a_trajectory_fidelity_corrected.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")

def load_trajectory_data(mode, arch, noise):
    """Load truth, background, and analysis trajectories from resample regime."""
    diag_dir = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /diagnostics")
    
    truth_path = os.path.join(diag_dir, f"truth_{mode}_{arch}_n{noise}.npy")
    analysis_path = os.path.join(diag_dir, f"analysis_{mode}_{arch}_n{noise}.npy")
    background_path = os.path.join(diag_dir, f"background_{mode}_{arch}_n{noise}.npy")
    
    truth = np.load(truth_path) if os.path.exists(truth_path) else None
    analysis = np.load(analysis_path) if os.path.exists(analysis_path) else None
    background = np.load(background_path) if os.path.exists(background_path) else None
    
    return truth, background, analysis

def generate_figure():
    """Generate trajectory fidelity comparison."""
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid: 2 rows x 3 columns for different configs
    configs = [
        ('x', 'gru', 0.1, '$h(x)=x_1$, GRU, $\\sigma=0.1$'),
        ('xy', 'gru', 0.1, '$h(x)=(x_1,x_2)$, GRU, $\\sigma=0.1$'),
        ('x2', 'gru', 0.1, '$h(x)=x_1^2$, GRU, $\\sigma=0.1$'),
        ('x', 'mlp', 0.5, '$h(x)=x_1$, MLP, $\\sigma=0.5$'),
        ('xy', 'lstm', 0.5, '$h(x)=(x_1,x_2)$, LSTM, $\\sigma=0.5$'),
        ('x2', 'mlp', 0.5, '$h(x)=x_1^2$, MLP, $\\sigma=0.5$'),
    ]
    
    for idx, (mode, arch, noise, title) in enumerate(configs):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        truth, background, analysis = load_trajectory_data(mode, arch, noise)
        
        if truth is not None:
            # Plot truth trajectory
            ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], 
                   'gray', linewidth=0.5, alpha=0.7, label='Truth')
        
        if analysis is not None:
            # Plot analysis trajectory
            ax.plot(analysis[:, 0], analysis[:, 1], analysis[:, 2], 
                   'b', linewidth=0.8, alpha=0.9, label='Analysis ($f_\\theta$)')
        
        if background is not None:
            # Plot background (first point only for clarity)
            ax.scatter(background[0, 0], background[0, 1], background[0, 2], 
                      c='r', s=50, marker='o', label='Background ($x_b$)')
        
        ax.set_xlabel('$x_1$', fontsize=9)
        ax.set_ylabel('$x_2$', fontsize=9)
        ax.set_zlabel('$x_3$', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    fig.suptitle('Trajectory Fidelity: Analysis vs Truth (Resample Regime)\n'
                 'Using learned analysis operator $f_\\theta$ (not $\\Phi$)', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_5a_trajectory_fidelity_corrected.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
