#!/usr/bin/env python3
"""
Generate Section 4 metrics and plots directly from trained model weights.

This script:
1. Loads test data from data/raw, data/obs, data/splits
2. Loads all model .pth files from results directories
3. Computes metrics from scratch (RMSE, improvement, Hausdorff, divergence)
4. Aggregates statistics across regime × arch × mode × sigma
5. Saves all_metrics_summary.csv
6. Generates 6 PNG figures
"""

import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import mean_squared_error

# Constants
EPSILON = 1e-8  # Small value to avoid division by zero
DIVERGENCE_THRESHOLD = 10.0  # RMSE threshold for trajectory divergence
LORENZ_ATTRACTOR_RANGE = 50.0  # Typical coordinate range for normalization
RANDOM_SEED = 42  # For reproducibility

# Set up project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import MLPModel, GRUModel, LSTMModel, BaselineMLP
from src.utils.lorenz import lorenz63_step

# Global configurations
SEQ_LEN = 5
DT = 0.01
STEPS = 200
N_TEST = 500  # Use all test trajectories

# Directory paths (handle spaces in directory names)
DATA_DIR = os.path.join(PROJECT_ROOT, "data ")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OBS_DIR = os.path.join(DATA_DIR, "obs")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
REPORT_DIR = os.path.join(PROJECT_ROOT, "Report")
FIGS_DIR = os.path.join(REPORT_DIR, "figs")

# Results directories (handle spaces)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESAMPLE_DIR = os.path.join(RESULTS_DIR, "resample ", "run_20251008_134240 ", "models")
FIXEDMEAN_DIR = os.path.join(RESULTS_DIR, "fixedmean ", "run_20251008_133752 ", "models")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline", "metrics")

# Ensure output directory exists
os.makedirs(FIGS_DIR, exist_ok=True)


# ==================== Observation Operators ====================
def obs_operator(x, mode="x"):
    """
    Observation operators as defined in observation_operators.py
    Args:
        x: state vector [3]
        mode: 'x', 'xy', or 'x2'
    Returns:
        observation vector
    """
    if mode == "x":
        return np.array([x[0]])
    elif mode == "xy":
        return np.array([x[0], x[1]])
    elif mode == "x2":
        return np.array([x[0]**2])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_noisy_observations(traj, mode, sigma):
    """
    Generate noisy observations from a trajectory.
    Args:
        traj: [steps, 3]
        mode: observation mode
        sigma: noise level
    Returns:
        obs: [steps, obs_dim]
    """
    np.random.seed(RANDOM_SEED)  # For reproducibility
    obs = np.array([obs_operator(x, mode) for x in traj])
    obs_noisy = obs + np.random.normal(0, sigma, obs.shape)
    return obs_noisy


# ==================== Model Loading ====================
def get_obs_dim(mode):
    """Get observation dimension for a given mode."""
    if mode == "x":
        return 1
    elif mode == "xy":
        return 2
    elif mode == "x2":
        return 1
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_model(path, arch, obs_dim, device="cpu"):
    """
    Load a model from a .pth file.
    Args:
        path: path to .pth file
        arch: architecture name ('mlp', 'gru', 'lstm', 'baseline')
        obs_dim: observation dimension
        device: torch device
    Returns:
        model
    """
    if arch == "mlp":
        model = MLPModel(x_dim=3, y_dim=obs_dim, hidden=64)
    elif arch == "gru":
        model = GRUModel(x_dim=3, y_dim=obs_dim, hidden=64)
    elif arch == "lstm":
        model = LSTMModel(x_dim=3, y_dim=obs_dim, hidden=64)
    elif arch == "baseline":
        model = BaselineMLP(x_dim=3, y_dim=obs_dim, hidden=32)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model


# ==================== Trajectory Reconstruction ====================
def reconstruct_trajectory(model, arch, test_traj_idx, obs_seq, B_mean, 
                          steps=50, device="cpu"):
    """
    Reconstruct a single trajectory using the model.
    Args:
        model: trained model
        arch: architecture name
        test_traj_idx: ground truth trajectory [steps, 3]
        obs_seq: observation sequence [steps, obs_dim]
        B_mean: background mean [3]
        steps: number of steps to reconstruct
        device: torch device
    Returns:
        pred_b: background predictions [steps, 3]
        pred_a: analysis predictions [steps, 3]
    """
    # Start with background mean
    x_b = torch.tensor(B_mean, dtype=torch.float32).unsqueeze(0).to(device)
    
    pred_b = []
    pred_a = []
    
    for t in range(SEQ_LEN - 1, steps):
        # Get observation sequence
        y_seq = torch.tensor(
            obs_seq[t - SEQ_LEN + 1:t + 1], 
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        # Perform analysis
        if arch == "baseline":
            # Baseline doesn't use background
            x_a = model(y_seq)
        else:
            # MLP/GRU/LSTM use background
            x_a = model(x_b, y_seq)
        
        # Store predictions
        pred_b.append(x_b.cpu().numpy().squeeze())
        pred_a.append(x_a.detach().cpu().numpy().squeeze())
        
        # Propagate forward using Lorenz-63
        x_next = lorenz63_step(x_a.detach().cpu().numpy().squeeze(), dt=DT)
        x_b = torch.tensor(x_next, dtype=torch.float32).unsqueeze(0).to(device)
    
    return np.array(pred_b), np.array(pred_a)


# ==================== Metrics Computation ====================
def compute_rmse(truth, pred):
    """Compute RMSE between truth and prediction."""
    try:
        return np.sqrt(mean_squared_error(truth, pred))
    except (ValueError, Exception) as e:
        return np.nan


def compute_hausdorff(truth, pred):
    """
    Compute symmetric Hausdorff distance (normalized by max coordinate range).
    Args:
        truth: [steps, 3]
        pred: [steps, 3]
    Returns:
        normalized Hausdorff distance
    """
    try:
        d1 = directed_hausdorff(truth, pred)[0]
        d2 = directed_hausdorff(pred, truth)[0]
        hausdorff = max(d1, d2)
        
        # Normalize by typical Lorenz attractor range
        return hausdorff / LORENZ_ATTRACTOR_RANGE
    except (ValueError, Exception) as e:
        return np.nan


def compute_divergence_rate(errors, threshold=None):
    """
    Compute divergence rate: step at which RMSE exceeds threshold.
    Args:
        errors: array of per-step errors
        threshold: divergence threshold (uses DIVERGENCE_THRESHOLD if None)
    Returns:
        divergence step (or -1 if never diverges)
    """
    if threshold is None:
        threshold = DIVERGENCE_THRESHOLD
    diverged_steps = np.where(errors > threshold)[0]
    if len(diverged_steps) > 0:
        return diverged_steps[0]
    else:
        return -1


def evaluate_model(model, arch, mode, sigma, test_traj, B_mean, 
                  n_test=50, steps=50, device="cpu"):
    """
    Evaluate a model on test trajectories.
    Args:
        model: trained model
        arch: architecture name
        mode: observation mode
        sigma: noise level
        test_traj: test trajectories [n_traj, steps, 3]
        B_mean: background mean [3]
        n_test: number of test trajectories to use
        steps: number of steps to evaluate
        device: torch device
    Returns:
        dict with evaluation metrics
    """
    rmse_b_list = []
    rmse_a_list = []
    improvement_list = []
    hausdorff_list = []
    divergence_list = []
    
    for i in range(min(n_test, len(test_traj))):
        traj = test_traj[i]
        
        # Generate noisy observations for this trajectory
        obs_seq = make_noisy_observations(traj, mode, sigma)
        
        # Reconstruct trajectory
        pred_b, pred_a = reconstruct_trajectory(
            model, arch, traj, obs_seq, B_mean, steps=steps, device=device
        )
        
        # Ground truth for comparison (skip first SEQ_LEN-1 steps)
        truth = traj[SEQ_LEN - 1:steps]
        
        # Compute RMSE
        rmse_b = compute_rmse(truth, pred_b)
        rmse_a = compute_rmse(truth, pred_a)
        
        # Compute improvement
        improvement = (rmse_b - rmse_a) / (rmse_b + EPSILON)
        
        # Compute Hausdorff distance
        hausdorff = compute_hausdorff(truth, pred_a)
        
        # Compute divergence rate (step-by-step errors)
        errors = np.sqrt(np.mean((truth - pred_a)**2, axis=1))
        divergence = compute_divergence_rate(errors, threshold=10.0)
        
        rmse_b_list.append(rmse_b)
        rmse_a_list.append(rmse_a)
        improvement_list.append(improvement)
        hausdorff_list.append(hausdorff)
        divergence_list.append(divergence)
    
    # Aggregate statistics
    # Handle divergence_rate separately to avoid warnings with empty lists
    diverged_values = [d for d in divergence_list if d >= 0]
    divergence_rate_mean = np.mean(diverged_values) if len(diverged_values) > 0 else np.nan
    
    results = {
        'rmse_b_mean': np.nanmean(rmse_b_list),
        'rmse_b_std': np.nanstd(rmse_b_list),
        'rmse_a_mean': np.nanmean(rmse_a_list),
        'rmse_a_std': np.nanstd(rmse_a_list),
        'rmse_a_median': np.nanmedian(rmse_a_list),
        'rmse_a_q25': np.nanpercentile(rmse_a_list, 25),
        'rmse_a_q75': np.nanpercentile(rmse_a_list, 75),
        'improvement_mean': np.nanmean(improvement_list),
        'improvement_std': np.nanstd(improvement_list),
        'improvement_median': np.nanmedian(improvement_list),
        'hausdorff_mean': np.nanmean(hausdorff_list),
        'hausdorff_std': np.nanstd(hausdorff_list),
        'hausdorff_median': np.nanmedian(hausdorff_list),
        'divergence_rate': divergence_rate_mean,
    }
    
    return results


# ==================== Model File Discovery ====================
def find_all_models():
    """
    Find all model files in the results directories.
    Returns:
        list of dicts with model metadata
    """
    models = []
    
    # Resample models
    if os.path.exists(RESAMPLE_DIR):
        for fpath in glob.glob(os.path.join(RESAMPLE_DIR, "*.pth")):
            fname = os.path.basename(fpath)
            # Pattern: {mode}_{arch}_n{sigma}_R{R}.pth
            m = re.match(r"([a-z0-9]+)_([a-z]+)_n([0-9.]+)_R([0-9.]+)\.pth", fname)
            if m:
                mode, arch, sigma_str, R_str = m.groups()
                models.append({
                    'path': fpath,
                    'regime': 'resample',
                    'mode': mode,
                    'arch': arch,
                    'sigma': float(sigma_str),
                    'R': float(R_str)
                })
    
    # Fixedmean models
    if os.path.exists(FIXEDMEAN_DIR):
        for fpath in glob.glob(os.path.join(FIXEDMEAN_DIR, "*.pth")):
            fname = os.path.basename(fpath)
            # Pattern: {mode}_{arch}_n{sigma}_R{R}.pth
            m = re.match(r"([a-z0-9]+)_([a-z]+)_n([0-9.]+)_R([0-9.]+)\.pth", fname)
            if m:
                mode, arch, sigma_str, R_str = m.groups()
                models.append({
                    'path': fpath,
                    'regime': 'fixedmean',
                    'mode': mode,
                    'arch': arch,
                    'sigma': float(sigma_str),
                    'R': float(R_str)
                })
    
    # Baseline models
    if os.path.exists(BASELINE_DIR):
        for fpath in glob.glob(os.path.join(BASELINE_DIR, "*.pth")):
            fname = os.path.basename(fpath)
            # Pattern: {mode}_n{sigma}.pth
            m = re.match(r"([a-z0-9]+)_n([0-9.]+)\.pth", fname)
            if m:
                mode, sigma_str = m.groups()
                models.append({
                    'path': fpath,
                    'regime': 'baseline',
                    'mode': mode,
                    'arch': 'baseline',
                    'sigma': float(sigma_str),
                    'R': None
                })
    
    return models


# ==================== Plotting Functions ====================
def plot_core_rmse_by_mode(df, output_path):
    """Plot RMSE (after) by mode for all regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['x', 'xy', 'x2']
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        df_mode = df[df['mode'] == mode]
        
        for regime in ['baseline', 'fixedmean', 'resample']:
            df_regime = df_mode[df_mode['regime'] == regime]
            if len(df_regime) > 0:
                # Group by sigma and compute mean RMSE
                grouped = df_regime.groupby('sigma')['rmse_a_mean'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=regime)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('RMSE (Analysis)')
        ax.set_title(f'Mode: {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_core_improvement_by_mode(df, output_path):
    """Plot improvement metric by mode for all regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['x', 'xy', 'x2']
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        df_mode = df[df['mode'] == mode]
        
        for regime in ['baseline', 'fixedmean', 'resample']:
            df_regime = df_mode[df_mode['regime'] == regime]
            if len(df_regime) > 0:
                grouped = df_regime.groupby('sigma')['improvement_mean'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=regime)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Improvement (normalized)')
        ax.set_title(f'Mode: {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_core_hausdorff_by_mode(df, output_path):
    """Plot Hausdorff distance by mode for all regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['x', 'xy', 'x2']
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        df_mode = df[df['mode'] == mode]
        
        for regime in ['baseline', 'fixedmean', 'resample']:
            df_regime = df_mode[df_mode['regime'] == regime]
            if len(df_regime) > 0:
                grouped = df_regime.groupby('sigma')['hausdorff_mean'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=regime)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Hausdorff Distance (normalized)')
        ax.set_title(f'Mode: {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mode_summary(df, mode, output_path):
    """
    Plot comprehensive summary for a specific mode.
    Shows RMSE, improvement, and Hausdorff across all architectures and noise levels.
    """
    df_mode = df[df['mode'] == mode]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Mode: {mode} - Comprehensive Summary', fontsize=14, fontweight='bold')
    
    # Plot 1: RMSE by architecture
    ax = axes[0, 0]
    for regime in ['baseline', 'fixedmean', 'resample']:
        for arch in df_mode['arch'].unique():
            df_subset = df_mode[(df_mode['regime'] == regime) & (df_mode['arch'] == arch)]
            if len(df_subset) > 0:
                label = f"{regime}_{arch}" if regime != 'baseline' else 'baseline'
                ax.plot(df_subset['sigma'], df_subset['rmse_a_mean'], 
                       marker='o', label=label, alpha=0.7)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('RMSE (Analysis)')
    ax.set_title('RMSE vs Noise Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Improvement by architecture
    ax = axes[0, 1]
    for regime in ['baseline', 'fixedmean', 'resample']:
        for arch in df_mode['arch'].unique():
            df_subset = df_mode[(df_mode['regime'] == regime) & (df_mode['arch'] == arch)]
            if len(df_subset) > 0:
                label = f"{regime}_{arch}" if regime != 'baseline' else 'baseline'
                ax.plot(df_subset['sigma'], df_subset['improvement_mean'], 
                       marker='o', label=label, alpha=0.7)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Improvement')
    ax.set_title('Improvement vs Noise Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Hausdorff distance
    ax = axes[1, 0]
    for regime in ['baseline', 'fixedmean', 'resample']:
        for arch in df_mode['arch'].unique():
            df_subset = df_mode[(df_mode['regime'] == regime) & (df_mode['arch'] == arch)]
            if len(df_subset) > 0:
                label = f"{regime}_{arch}" if regime != 'baseline' else 'baseline'
                ax.plot(df_subset['sigma'], df_subset['hausdorff_mean'], 
                       marker='o', label=label, alpha=0.7)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Hausdorff Distance')
    ax.set_title('Hausdorff Distance vs Noise Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Divergence rate
    ax = axes[1, 1]
    for regime in ['baseline', 'fixedmean', 'resample']:
        for arch in df_mode['arch'].unique():
            df_subset = df_mode[(df_mode['regime'] == regime) & (df_mode['arch'] == arch)]
            if len(df_subset) > 0 and not df_subset['divergence_rate'].isna().all():
                label = f"{regime}_{arch}" if regime != 'baseline' else 'baseline'
                ax.plot(df_subset['sigma'], df_subset['divergence_rate'], 
                       marker='o', label=label, alpha=0.7)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Divergence Step')
    ax.set_title('Divergence Rate vs Noise Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ==================== Main Function ====================
def main():
    """Main execution function."""
    print("=" * 70)
    print("Section 4 Metrics and Plots Generation")
    print("=" * 70)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading test data...")
    test_traj = np.load(os.path.join(RAW_DIR, "test_traj.npy"))
    B_mean = np.load(os.path.join(RAW_DIR, "B_mean.npy"))
    print(f"Test trajectories shape: {test_traj.shape}")
    print(f"B_mean: {B_mean}")
    
    # Find all models
    print("\nDiscovering models...")
    models = find_all_models()
    print(f"Found {len(models)} models to evaluate")
    
    # Print summary
    regimes = set(m['regime'] for m in models)
    print(f"Regimes: {sorted(regimes)}")
    for regime in sorted(regimes):
        regime_models = [m for m in models if m['regime'] == regime]
        print(f"  {regime}: {len(regime_models)} models")
    
    # Evaluate all models
    print("\nEvaluating models...")
    results = []
    
    for i, model_info in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Evaluating: {model_info['regime']}/{model_info['mode']}/"
              f"{model_info['arch']}/σ={model_info['sigma']}")
        
        try:
            # Load model
            obs_dim = get_obs_dim(model_info['mode'])
            model = load_model(
                model_info['path'], 
                model_info['arch'], 
                obs_dim, 
                device=device
            )
            
            # Evaluate on test set
            metrics = evaluate_model(
                model, 
                model_info['arch'],
                model_info['mode'],
                model_info['sigma'],
                test_traj,
                B_mean,
                n_test=50,  # Use 50 test trajectories for speed
                steps=50,   # Evaluate first 50 steps
                device=device
            )
            
            # Combine metadata and metrics
            result = {**model_info, **metrics}
            results.append(result)
            
            print(f"  RMSE_b: {metrics['rmse_b_mean']:.4f}, "
                  f"RMSE_a: {metrics['rmse_a_mean']:.4f}, "
                  f"Improvement: {metrics['improvement_mean']:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary DataFrame
    print("\nCreating summary DataFrame...")
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_csv = os.path.join(FIGS_DIR, "all_metrics_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved metrics to: {output_csv}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Core plots
    plot_core_rmse_by_mode(df, os.path.join(FIGS_DIR, "core_rmse_by_mode.png"))
    plot_core_improvement_by_mode(df, os.path.join(FIGS_DIR, "core_improvement_by_mode.png"))
    plot_core_hausdorff_by_mode(df, os.path.join(FIGS_DIR, "core_hausdorff_by_mode.png"))
    
    # Mode-specific summaries
    plot_mode_summary(df, 'x', os.path.join(FIGS_DIR, "mode_x_summary.png"))
    plot_mode_summary(df, 'xy', os.path.join(FIGS_DIR, "mode_xy_summary.png"))
    plot_mode_summary(df, 'x2', os.path.join(FIGS_DIR, "mode_x2_summary.png"))
    
    print("\n" + "=" * 70)
    print("Metrics and plots generation complete!")
    print(f"Output directory: {FIGS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
