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

# Resample regime results directory (note: directory names have trailing spaces)
RESAMPLE_DIR = RESULTS_DIR / "resample " / "run_20251008_134240 " / "diagnostics"

# Raw data directory (note: directory name has trailing space)
RAW_DATA_DIR = REPO_ROOT / "data " / "raw"

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


def compute_rmdse(pred, truth):
    """
    Compute Root Median Squared Error (RMdSE) between prediction and truth.
    
    This is a robust alternative to RMSE that is less sensitive to outliers.
    RMdSE uses the median instead of the mean, making it more appropriate
    when distributions may have heavy tails or occasional catastrophic failures.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values
    truth : np.ndarray
        True values
        
    Returns
    -------
    float
        Root Median Squared Error
    """
    return np.sqrt(np.median((pred - truth) ** 2))


def compute_sample_wise_rmse(truth, analysis):
    """
    Compute per-sample RMSE values for boxplot distributions.
    
    Handles different array shapes:
    - 3D arrays: (n_trajectories, n_timesteps, state_dim)
    - 2D arrays: (n_samples, state_dim) or (n_timesteps, state_dim)
    - 1D arrays: flattened data
    
    Parameters
    ----------
    truth : np.ndarray
        True state values
    analysis : np.ndarray
        Analysis (predicted) state values
        
    Returns
    -------
    list
        List of RMSE values, one per sample/trajectory
    """
    if len(truth.shape) >= 2:
        n_samples = truth.shape[0]
        rmse_samples = []
        for i in range(n_samples):
            if len(truth.shape) == 3:
                sample_rmse = compute_rmse(analysis[i], truth[i])
            else:
                sample_rmse = compute_rmse(analysis[i:i+1], truth[i:i+1])
            rmse_samples.append(sample_rmse)
        return rmse_samples
    else:
        return [compute_rmse(analysis, truth)]


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
    Generate sample trajectory comparison showing truth vs analysis reconstruction.
    
    Uses resample regime data for better visualization of assimilation quality.
    
    Output: figures_new/trajectory_sample_new.png
    """
    mode = 'xy'
    noise = 0.1
    arch = 'gru'  # Best performing architecture
    
    try:
        truth_path = RESAMPLE_DIR / f"truth_{mode}_{arch}_n{noise}.npy"
        analysis_path = RESAMPLE_DIR / f"analysis_{mode}_{arch}_n{noise}.npy"
        background_path = RESAMPLE_DIR / f"background_{mode}_{arch}_n{noise}.npy"
        
        if truth_path.exists():
            truth = np.load(truth_path)
            analysis = np.load(analysis_path)
            background = np.load(background_path)
        else:
            raise FileNotFoundError("Resample data not found")
    except FileNotFoundError:
        # Fall back to baseline
        try:
            truth, analysis, background = load_baseline_results(mode, noise)
        except FileNotFoundError:
            print(f"No data found for mode={mode}, noise={noise}")
            return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Determine number of time steps to plot
    if len(truth.shape) == 2:
        T = min(100, truth.shape[0])
        truth_plot = truth[:T, :]
        analysis_plot = analysis[:T, :]
        background_plot = background[:T, :]
    else:
        T = min(100, truth.shape[0])
        truth_plot = truth.reshape(-1, 3)[:T, :]
        analysis_plot = analysis.reshape(-1, 3)[:T, :]
        background_plot = background.reshape(-1, 3)[:T, :]
    
    labels = [r'$x_1$', r'$x_2$', r'$x_3$']
    colors = {'Truth': 'black', 'Analysis': 'blue', 'Background': 'red'}
    
    for i in range(3):
        axes[i].plot(truth_plot[:, i], 'k-', label='Truth', linewidth=2.0)
        axes[i].plot(analysis_plot[:, i], 'b--', label='Analysis', linewidth=1.5)
        axes[i].plot(background_plot[:, i], 'r:', label='Background', linewidth=1.2, alpha=0.7)
        axes[i].set_ylabel(labels[i], fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=10)
        if i == 0:
            axes[i].legend(loc='upper right', ncol=3, fontsize=10)
    
    axes[-1].set_xlabel('Time step', fontsize=12)
    fig.suptitle(f'State Reconstruction: Truth vs Analysis (mode={mode}, $\\sigma_{{obs}}$={noise}, {arch.upper()})', 
                 y=1.01, fontsize=13)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "trajectory_sample_new.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_attractor_projection():
    """
    Generate attractor phase-space projections using full trajectory data.
    
    This function loads the raw trajectory data to generate a proper visualization
    of the Lorenz attractor with sufficient data points for a clear butterfly pattern.
    
    Output: figures_new/attractor_projection_new.png
    """
    # Use raw trajectory data for better visualization
    raw_data_path = RAW_DATA_DIR / "test_traj.npy"
    
    try:
        # Load full trajectory data (500 trajectories x 200 timesteps x 3 dims)
        trajectories = np.load(raw_data_path)
        
        # Flatten to get all points
        truth_flat = trajectories.reshape(-1, 3)
        
    except FileNotFoundError:
        print(f"Raw trajectory data not found at {raw_data_path}")
        # Fall back to baseline results
        mode = 'xy'
        noise = 0.1
        try:
            truth, analysis, background = load_baseline_results(mode, noise)
            if len(truth.shape) == 3:
                truth_flat = truth.reshape(-1, 3)
            else:
                truth_flat = truth
        except FileNotFoundError:
            print("No data available for attractor projection")
            return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use every 2nd point for better density visualization
    step = max(1, len(truth_flat) // 50000)  # Limit to ~50k points for clarity
    
    # X-Y projection (classic Lorenz butterfly view)
    axes[0].scatter(truth_flat[::step, 0], truth_flat[::step, 1], 
                   c='darkblue', s=0.5, alpha=0.4, label='Trajectory')
    axes[0].set_xlabel(r'$x_1$', fontsize=12)
    axes[0].set_ylabel(r'$x_2$', fontsize=12)
    axes[0].set_title('X-Y Projection (Top View)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='datalim')
    
    # X-Z projection (classic butterfly side view)
    axes[1].scatter(truth_flat[::step, 0], truth_flat[::step, 2], 
                   c='darkgreen', s=0.5, alpha=0.4, label='Trajectory')
    axes[1].set_xlabel(r'$x_1$', fontsize=12)
    axes[1].set_ylabel(r'$x_3$', fontsize=12)
    axes[1].set_title('X-Z Projection (Butterfly View)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Lorenz-63 Attractor Geometry (Test Trajectories)', y=1.02, fontsize=13)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "attractor_projection_new.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_rmse_boxplot_logscale():
    """
    Generate RMSE boxplot with logarithmic y-scale.
    
    This addresses Hans's comment (#105) requesting log-scale for better 
    visualization when there are outliers from catastrophic failures.
    
    Output: figures_new/rmse_boxplot_logscale.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for idx, mode in enumerate(MODES):
        all_rmse = []
        positions = []
        labels = []
        
        for noise_idx, noise in enumerate(NOISE_LEVELS):
            try:
                truth, analysis, background = load_baseline_results(mode, noise)
                
                # Compute per-sample RMSE (not aggregated)
                rmse_samples = compute_sample_wise_rmse(truth, analysis)
                all_rmse.append(rmse_samples)
                    
                positions.append(noise_idx + 1)
                labels.append(f'{noise}')
            except FileNotFoundError:
                continue
        
        if all_rmse:
            axes[idx].boxplot(all_rmse, positions=positions, widths=0.6)
            axes[idx].set_yscale('log')
            axes[idx].set_xlabel(r'Observation noise $\sigma_{\mathrm{obs}}$')
            axes[idx].set_title(f'Mode: {mode}')
            axes[idx].set_xticks(positions)
            axes[idx].set_xticklabels(labels)
            axes[idx].grid(True, alpha=0.3, which='both')
    
    axes[0].set_ylabel('RMSE (log scale)')
    fig.suptitle('Post-Assimilation RMSE Distribution (Log Scale)', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "rmse_boxplot_logscale.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_robust_metrics_comparison():
    """
    Generate comparison of RMSE vs RMdSE (Root Median Squared Error).
    
    RMdSE is more robust to outliers and provides a better characterization
    when there are catastrophic failures like attractor escape.
    
    Output: figures_new/rmse_vs_rmdse_comparison.png
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for idx, mode in enumerate(MODES):
        rmse_values = []
        rmdse_values = []
        
        for noise in NOISE_LEVELS:
            try:
                truth, analysis, background = load_baseline_results(mode, noise)
                rmse = compute_rmse(analysis, truth)
                rmdse = compute_rmdse(analysis, truth)
                rmse_values.append(rmse)
                rmdse_values.append(rmdse)
            except FileNotFoundError:
                rmse_values.append(np.nan)
                rmdse_values.append(np.nan)
        
        # Top row: RMSE
        axes[0, idx].plot(NOISE_LEVELS, rmse_values, 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, idx].set_xlabel(r'$\sigma_{\mathrm{obs}}$')
        axes[0, idx].set_title(f'Mode: {mode}')
        axes[0, idx].set_xscale('log')
        axes[0, idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[0, idx].set_ylabel('RMSE')
        
        # Bottom row: RMdSE
        axes[1, idx].plot(NOISE_LEVELS, rmdse_values, 's-', linewidth=2, markersize=8, color='green')
        axes[1, idx].set_xlabel(r'$\sigma_{\mathrm{obs}}$')
        axes[1, idx].set_xscale('log')
        axes[1, idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[1, idx].set_ylabel('RMdSE (robust)')
    
    fig.suptitle('RMSE vs RMdSE: Sensitivity to Outliers', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "rmse_vs_rmdse_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all figures or specific figure if name provided."""
    figures = {
        'rmse_comparison': generate_rmse_comparison,
        'trajectory_sample': generate_trajectory_sample,
        'attractor_projection': generate_attractor_projection,
        'rmse_boxplot_logscale': generate_rmse_boxplot_logscale,
        'robust_metrics': generate_robust_metrics_comparison,
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
