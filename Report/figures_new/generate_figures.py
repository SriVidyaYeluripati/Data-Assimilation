#!/usr/bin/env python3
"""
Figure Generation Scripts for AI-Based Data Assimilation Report

This module contains scripts to regenerate key figures from the stored .npy results.
Run individual functions to create updated versions of specific figures.

Usage:
    python generate_figures.py [figure_name]
    
Available figures:
    - rmse_comparison
    - convergence_envelopes
    - attractor_geometry
    - noise_sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_NEW_DIR = SCRIPT_DIR  # Output to same directory as script

# Ensure output directory exists
FIGURES_NEW_DIR.mkdir(exist_ok=True)

# Observation modes and noise levels
MODES = ['x', 'xy', 'x2']
NOISE_LEVELS = [0.05, 0.1, 0.5, 1.0]
ARCHITECTURES = ['MLP', 'GRU', 'LSTM']


def load_baseline_results(mode, noise):
    """
    Load baseline regime results from .npy files.
    
    Parameters
    ----------
    mode : str
        Observation mode ('x', 'xy', or 'x2')
    noise : float
        Noise level (0.05, 0.1, 0.5, or 1.0)
    
    Returns
    -------
    tuple
        (truth, analysis, background) arrays
    
    Raises
    ------
    FileNotFoundError
        If any of the required .npy files are not found. Callers should handle
        this exception and provide appropriate fallback behavior.
    """
    diag_dir = RESULTS_DIR / "baseline" / "diagnostics"
    
    truth_path = diag_dir / f"truth_{mode}_baseline_n{noise}.npy"
    analysis_path = diag_dir / f"analysis_{mode}_baseline_n{noise}.npy"
    background_path = diag_dir / f"background_{mode}_baseline_n{noise}.npy"
    
    # Check all files exist before loading
    for path in [truth_path, analysis_path, background_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    
    truth = np.load(truth_path)
    analysis = np.load(analysis_path)
    background = np.load(background_path)
    
    return truth, analysis, background


def compute_rmse(pred, truth):
    """Compute RMSE between prediction and truth."""
    return np.sqrt(np.mean((pred - truth) ** 2))


def generate_rmse_comparison():
    """
    Generate RMSE comparison across noise levels and observation modes.
    
    Output: figures_new/rmse_comparison_new.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    
    for idx, mode in enumerate(MODES):
        rmse_values = []
        for noise in NOISE_LEVELS:
            try:
                truth, analysis, background = load_baseline_results(mode, noise)
                rmse = compute_rmse(analysis, truth)
                rmse_values.append(rmse)
            except FileNotFoundError:
                rmse_values.append(np.nan)
        
        axes[idx].plot(NOISE_LEVELS, rmse_values, 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel(r'Observation noise $\sigma_{\mathrm{obs}}$')
        axes[idx].set_title(f'Mode: {mode}')
        axes[idx].set_xscale('log')
        axes[idx].grid(True, alpha=0.3)
    
    axes[0].set_ylabel('RMSE')
    fig.suptitle('Post-Assimilation RMSE vs Noise Level (Baseline Regime)', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "rmse_comparison_new.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_trajectory_sample():
    """
    Generate sample trajectory comparison.
    
    Output: figures_new/trajectory_sample_new.png
    """
    mode = 'xy'
    noise = 0.1
    
    try:
        truth, analysis, background = load_baseline_results(mode, noise)
    except FileNotFoundError:
        print(f"Results not found for mode={mode}, noise={noise}")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Plot first 100 time steps of first trajectory
    T = min(100, truth.shape[1] if len(truth.shape) > 1 else truth.shape[0])
    
    # Reshape if needed
    if len(truth.shape) == 2:
        truth_plot = truth[:T, :]
        analysis_plot = analysis[:T, :]
        background_plot = background[:T, :]
    else:
        truth_plot = truth.reshape(-1, 3)[:T, :]
        analysis_plot = analysis.reshape(-1, 3)[:T, :]
        background_plot = background.reshape(-1, 3)[:T, :]
    
    labels = [r'$x_1$', r'$x_2$', r'$x_3$']
    
    for i in range(3):
        axes[i].plot(truth_plot[:, i], 'k-', label='Truth', linewidth=1.5)
        axes[i].plot(analysis_plot[:, i], 'b--', label='Analysis', linewidth=1.2)
        axes[i].plot(background_plot[:, i], 'r:', label='Background', linewidth=1.0, alpha=0.7)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper right', ncol=3)
    
    axes[-1].set_xlabel('Time step')
    fig.suptitle(f'Trajectory Reconstruction (mode={mode}, $\\sigma_{{\\mathrm{{obs}}}}$={noise})', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "trajectory_sample_new.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_attractor_projection():
    """
    Generate attractor phase-space projections.
    
    Output: figures_new/attractor_projection_new.png
    """
    mode = 'xy'
    noise = 0.1
    
    try:
        truth, analysis, background = load_baseline_results(mode, noise)
    except FileNotFoundError:
        print(f"Results not found for mode={mode}, noise={noise}")
        return
    
    # Reshape if needed
    if len(truth.shape) == 3:
        truth_flat = truth.reshape(-1, 3)
        analysis_flat = analysis.reshape(-1, 3)
    elif len(truth.shape) == 2 and truth.shape[1] == 3:
        truth_flat = truth
        analysis_flat = analysis
    else:
        print(f"Unexpected array shape: truth.shape={truth.shape}")
        print("Expected either:")
        print("  - 3D array with shape (n_trajectories, n_timesteps, 3)")
        print("  - 2D array with shape (n_samples, 3)")
        print("Check that the .npy files contain properly formatted state vectors.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # X-Y projection
    axes[0].plot(truth_flat[::10, 0], truth_flat[::10, 1], 'k.', alpha=0.3, markersize=1, label='Truth')
    axes[0].plot(analysis_flat[::10, 0], analysis_flat[::10, 1], 'b.', alpha=0.3, markersize=1, label='Analysis')
    axes[0].set_xlabel(r'$x_1$')
    axes[0].set_ylabel(r'$x_2$')
    axes[0].set_title('X-Y Projection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # X-Z projection
    axes[1].plot(truth_flat[::10, 0], truth_flat[::10, 2], 'k.', alpha=0.3, markersize=1, label='Truth')
    axes[1].plot(analysis_flat[::10, 0], analysis_flat[::10, 2], 'b.', alpha=0.3, markersize=1, label='Analysis')
    axes[1].set_xlabel(r'$x_1$')
    axes[1].set_ylabel(r'$x_3$')
    axes[1].set_title('X-Z Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Attractor Geometry (mode={mode}, $\\sigma_{{\\mathrm{{obs}}}}$={noise})', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "attractor_projection_new.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all figures or specific figure if name provided."""
    figures = {
        'rmse_comparison': generate_rmse_comparison,
        'trajectory_sample': generate_trajectory_sample,
        'attractor_projection': generate_attractor_projection,
    }
    
    if len(sys.argv) > 1:
        fig_name = sys.argv[1]
        if fig_name in figures:
            figures[fig_name]()
        else:
            print(f"Unknown figure: {fig_name}")
            print(f"Available: {list(figures.keys())}")
    else:
        print("Generating all figures...")
        for name, func in figures.items():
            print(f"\n--- {name} ---")
            try:
                func()
            except Exception as e:
                print(f"Error generating {name}: {e}")


if __name__ == "__main__":
    main()
