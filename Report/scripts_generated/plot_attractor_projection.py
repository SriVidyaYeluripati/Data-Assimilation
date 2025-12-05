#!/usr/bin/env python3
"""
plot_attractor_projection.py

Generates Lorenz-63 attractor phase-space projections using full trajectory data.
Shows the classic butterfly pattern with sufficient data points for clear visualization.

Output: figures_new/attractor_projection_new.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Repository paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_NEW_DIR = REPO_ROOT / "Report" / "figures_new"
RAW_DATA_DIR = REPO_ROOT / "data " / "raw"

FIGURES_NEW_DIR.mkdir(exist_ok=True)


def load_trajectory_data():
    """Load raw trajectory data for attractor visualization."""
    test_traj_path = RAW_DATA_DIR / "test_traj.npy"
    train_traj_path = RAW_DATA_DIR / "train_traj.npy"
    
    if test_traj_path.exists():
        data = np.load(test_traj_path)
        print(f"Loaded test trajectory data: shape={data.shape}")
        return data.reshape(-1, 3)
    elif train_traj_path.exists():
        data = np.load(train_traj_path)
        print(f"Loaded train trajectory data: shape={data.shape}")
        return data.reshape(-1, 3)
    else:
        print("No trajectory data found, generating synthetic Lorenz data")
        return generate_synthetic_lorenz()


def generate_synthetic_lorenz(n_steps=100000, dt=0.01):
    """Generate synthetic Lorenz-63 trajectory."""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    x = np.zeros((n_steps, 3))
    x[0] = [1.0, 1.0, 1.0]
    
    for i in range(1, n_steps):
        dx = sigma * (x[i-1, 1] - x[i-1, 0])
        dy = x[i-1, 0] * (rho - x[i-1, 2]) - x[i-1, 1]
        dz = x[i-1, 0] * x[i-1, 1] - beta * x[i-1, 2]
        x[i] = x[i-1] + dt * np.array([dx, dy, dz])
    
    return x


def generate_figure():
    """Generate the attractor projection figure."""
    trajectory_data = load_trajectory_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subsample for clarity (max ~50k points)
    step = max(1, len(trajectory_data) // 50000)
    data = trajectory_data[::step]
    
    # X-Y projection (top view)
    axes[0].scatter(data[:, 0], data[:, 1], c='darkblue', s=0.5, alpha=0.4)
    axes[0].set_xlabel(r'$x_1$', fontsize=12)
    axes[0].set_ylabel(r'$x_2$', fontsize=12)
    axes[0].set_title('X-Y Projection (Top View)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='datalim')
    
    # X-Z projection (butterfly view)
    axes[1].scatter(data[:, 0], data[:, 2], c='darkgreen', s=0.5, alpha=0.4)
    axes[1].set_xlabel(r'$x_1$', fontsize=12)
    axes[1].set_ylabel(r'$x_3$', fontsize=12)
    axes[1].set_title('X-Z Projection (Butterfly View)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Lorenz-63 Attractor Geometry ({len(data):,} points)', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_NEW_DIR / "attractor_projection_new.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_figure()
