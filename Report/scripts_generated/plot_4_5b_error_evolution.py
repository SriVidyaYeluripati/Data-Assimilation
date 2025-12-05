#!/usr/bin/env python3
"""
Regenerate Figure 4_5b: Error Evolution Profiles

Hans's comments addressed:
- Add clear model labels
- Show regime comparison (Resample vs FixedMean)
- Add noise level annotations

This script regenerates the ORIGINAL figure with corrections.
Original: Report/4_5b_error_evolution_profiles.png
Output: Report/figures_new/4_5b_error_evolution_profiles_corrected.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")

def load_trajectory_data(regime, mode, arch, noise):
    """Load truth, background, and analysis trajectories."""
    if regime == 'resample':
        diag_dir = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /diagnostics")
    elif regime == 'fixedmean':
        diag_dir = os.path.join(RESULTS_DIR, "fixedmean /run_20251008_133752 /diagnostics")
    else:
        return None, None, None
    
    truth_path = os.path.join(diag_dir, f"truth_{mode}_{arch}_n{noise}.npy")
    analysis_path = os.path.join(diag_dir, f"analysis_{mode}_{arch}_n{noise}.npy")
    background_path = os.path.join(diag_dir, f"background_{mode}_{arch}_n{noise}.npy")
    
    truth = np.load(truth_path) if os.path.exists(truth_path) else None
    analysis = np.load(analysis_path) if os.path.exists(analysis_path) else None
    background = np.load(background_path) if os.path.exists(background_path) else None
    
    return truth, background, analysis

def compute_error_evolution(truth, estimate):
    """Compute Euclidean error at each time step."""
    if truth is None or estimate is None:
        return None
    return np.sqrt(np.sum((truth - estimate)**2, axis=1))

def generate_figure():
    """Generate error evolution profiles for Resample vs FixedMean."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Temporal Error Evolution: Resample vs FixedMean\n'
                 'GRU Architecture, $h(x) = (x_1, x_2)$ Mode', fontsize=12)
    
    noise_levels = [0.1, 0.5]
    regimes = ['resample', 'fixedmean']
    regime_labels = ['Resample', 'FixedMean']
    colors = {'resample': '#2ecc71', 'fixedmean': '#e74c3c'}
    
    for row, sigma in enumerate(noise_levels):
        for col, (regime, regime_label) in enumerate(zip(regimes, regime_labels)):
            ax = axes[row, col]
            
            truth, background, analysis = load_trajectory_data(regime, 'xy', 'gru', sigma)
            
            if truth is not None and analysis is not None:
                error = compute_error_evolution(truth, analysis)
                time_steps = np.arange(len(error))
                
                ax.plot(time_steps, error, color=colors[regime], linewidth=1.5, 
                       label=f'{regime_label} Analysis')
                
                if background is not None:
                    bg_error = compute_error_evolution(truth, background)
                    ax.plot(time_steps, bg_error, color='gray', linewidth=1, 
                           linestyle='--', alpha=0.7, label='Background')
                
                ax.set_xlabel('Time Step', fontsize=10)
                ax.set_ylabel('Euclidean Error', fontsize=10)
                ax.set_title(f'{regime_label}, $\\sigma = {sigma}$', fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{regime_label}, $\\sigma = {sigma}$', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_5b_error_evolution_profiles_corrected.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
