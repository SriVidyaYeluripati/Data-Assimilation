#!/usr/bin/env python3
"""
Generate training + validation loss convergence grid.

Produces a 3×4 grid:
- Rows: observation modes (x, xy, x²)
- Columns: noise levels (σ = 0.05, 0.1, 0.5, 1.0)

For each cell, plots MLP, GRU, LSTM training losses (solid lines)
and validation losses (dashed lines).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration
MODES = ['x', 'xy', 'x2']
MODE_LABELS = {
    'x': r'$h(x) = x_1$',
    'xy': r'$h(x) = (x_1, x_2)$',
    'x2': r'$h(x) = x_1^2$'
}
NOISE_LEVELS = [0.05, 0.1, 0.5, 1.0]
ARCHS = ['mlp', 'gru', 'lstm']
ARCH_COLORS = {'mlp': '#1f77b4', 'gru': '#ff7f0e', 'lstm': '#2ca02c'}
ARCH_LABELS = {'mlp': 'MLP', 'gru': 'GRU', 'lstm': 'LSTM'}

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESAMPLE_DIR = REPO_ROOT / "results" / "resample " / "run_20251008_134240 " / "metrics"
FIGURES_DIR = SCRIPT_DIR.parent / "figures_new"
FIGURES_DIR.mkdir(exist_ok=True)


def load_loss_data(mode, arch, noise):
    """Load training and validation loss for a configuration."""
    # Try multiple path formats
    noise_str = f"n{noise}_R{noise}"
    filename = f"loss_{mode}_{arch}_{noise_str}.json"
    filepath = RESAMPLE_DIR / filename
    
    if not filepath.exists():
        # Try without space in path
        alt_dir = REPO_ROOT / "results" / "resample" / "run_20251008_134240" / "metrics"
        filepath = alt_dir / filename
    
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        return data.get('train_loss', []), data.get('val_loss', [])
    
    return None, None


def generate_loss_grid():
    """Generate the 3×4 loss convergence grid."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    
    for row, mode in enumerate(MODES):
        for col, noise in enumerate(NOISE_LEVELS):
            ax = axes[row, col]
            
            has_data = False
            for arch in ARCHS:
                train_loss, val_loss = load_loss_data(mode, arch, noise)
                
                if train_loss:
                    epochs = np.arange(1, len(train_loss) + 1)
                    
                    # Plot training loss (solid)
                    ax.plot(epochs, train_loss, 
                           color=ARCH_COLORS[arch], 
                           linestyle='-',
                           linewidth=1.5,
                           label=f'{ARCH_LABELS[arch]} Train')
                    
                    # Plot validation loss (dashed)
                    if val_loss:
                        ax.plot(epochs, val_loss,
                               color=ARCH_COLORS[arch],
                               linestyle='--',
                               linewidth=1.5,
                               alpha=0.7,
                               label=f'{ARCH_LABELS[arch]} Val')
                    has_data = True
            
            # Configure axes
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Row labels (modes)
            if col == 0:
                ax.set_ylabel(f'{MODE_LABELS[mode]}\nLoss', fontsize=10)
            
            # Column labels (noise)
            if row == 0:
                ax.set_title(f'$\\sigma_{{\\text{{obs}}}} = {noise}$', fontsize=11)
            
            # X-axis label
            if row == 2:
                ax.set_xlabel('Epoch', fontsize=10)
            
            if not has_data:
                ax.text(0.5, 0.5, 'No data', 
                       ha='center', va='center', 
                       transform=ax.transAxes,
                       fontsize=10, color='gray')
    
    # Add legend to first subplot
    handles, labels = [], []
    for arch in ARCHS:
        handles.append(plt.Line2D([0], [0], color=ARCH_COLORS[arch], linestyle='-', linewidth=2))
        labels.append(f'{ARCH_LABELS[arch]} Train')
        handles.append(plt.Line2D([0], [0], color=ARCH_COLORS[arch], linestyle='--', linewidth=2, alpha=0.7))
        labels.append(f'{ARCH_LABELS[arch]} Val')
    
    fig.legend(handles, labels, 
               loc='upper center', 
               ncol=6, 
               bbox_to_anchor=(0.5, 1.02),
               fontsize=9)
    
    fig.suptitle('Training and Validation Loss Convergence Grid (Resample Regime)',
                 fontsize=14, fontweight='bold', y=1.06)
    
    plt.tight_layout()
    
    # Save figure
    output_path = FIGURES_DIR / "training_validation_loss_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    return output_path


if __name__ == "__main__":
    generate_loss_grid()
