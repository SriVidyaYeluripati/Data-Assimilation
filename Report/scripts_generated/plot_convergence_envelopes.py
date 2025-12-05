#!/usr/bin/env python3
"""
Regenerate Figure 7: Mean convergence envelopes across training regimes.

This figure shows training loss convergence for the three regimes:
- Resample (blue): Fast, stable convergence
- FixedMean (orange): Fast initial descent but unstable at high noise
- Baseline (green): Slow convergence, retains high error

Hans's comments addressed:
- Clear regime labels
- Proper normalization (fraction of initial loss)
- Epoch axis clearly labeled
- Standard deviation envelopes shown

Output: Report/figures_new/4_2k_mean_convergence_envelopes_corrected.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")

# Regime directories
RESAMPLE_DIR = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics")
FIXEDMEAN_DIR = os.path.join(RESULTS_DIR, "fixedmean /run_20251008_133752 /metrics")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline/metrics")

def load_loss_curves(directory, pattern="loss_*.json"):
    """Load all loss curves from a directory."""
    loss_curves = []
    files = glob(os.path.join(directory, pattern))
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                # Handle different key names
                if 'train_loss' in data:
                    curve = np.array(data['train_loss'])
                elif 'train' in data:
                    curve = np.array(data['train'])
                else:
                    continue
                    
                if len(curve) > 0 and curve[0] > 0:
                    # Normalize to fraction of initial loss
                    normalized = curve / curve[0]
                    loss_curves.append(normalized)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return loss_curves

def compute_envelope(curves, max_epochs=30):
    """Compute mean and std envelopes from multiple loss curves."""
    # Pad/truncate curves to same length
    padded = []
    for curve in curves:
        if len(curve) >= max_epochs:
            padded.append(curve[:max_epochs])
        else:
            # Pad with last value
            pad = np.full(max_epochs, curve[-1])
            pad[:len(curve)] = curve
            padded.append(pad)
    
    if len(padded) == 0:
        return None, None, None
    
    stacked = np.stack(padded, axis=0)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    epochs = np.arange(1, max_epochs + 1)
    
    return epochs, mean, std

def generate_figure():
    """Generate the corrected convergence envelopes figure."""
    
    # Load loss curves for each regime
    print("Loading Resample loss curves...")
    resample_curves = load_loss_curves(RESAMPLE_DIR)
    print(f"  Found {len(resample_curves)} curves")
    
    print("Loading FixedMean loss curves...")
    fixedmean_curves = load_loss_curves(FIXEDMEAN_DIR)
    print(f"  Found {len(fixedmean_curves)} curves")
    
    print("Loading Baseline loss curves...")
    baseline_curves = load_loss_curves(BASELINE_DIR)
    print(f"  Found {len(baseline_curves)} curves")
    
    # Compute envelopes
    max_epochs = 30
    resample_epochs, resample_mean, resample_std = compute_envelope(resample_curves, max_epochs)
    fixedmean_epochs, fixedmean_mean, fixedmean_std = compute_envelope(fixedmean_curves, max_epochs)
    baseline_epochs, baseline_mean, baseline_std = compute_envelope(baseline_curves, max_epochs)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot Resample (blue)
    if resample_mean is not None:
        plt.plot(resample_epochs, resample_mean, 'b-', linewidth=2, label='Resample')
        plt.fill_between(resample_epochs, 
                        resample_mean - resample_std, 
                        resample_mean + resample_std, 
                        alpha=0.3, color='blue')
    
    # Plot FixedMean (orange)
    if fixedmean_mean is not None:
        plt.plot(fixedmean_epochs, fixedmean_mean, color='orange', linewidth=2, label='FixedMean')
        plt.fill_between(fixedmean_epochs, 
                        fixedmean_mean - fixedmean_std, 
                        fixedmean_mean + fixedmean_std, 
                        alpha=0.3, color='orange')
    
    # Plot Baseline (green)
    if baseline_mean is not None:
        plt.plot(baseline_epochs, baseline_mean, 'g-', linewidth=2, label='Baseline')
        plt.fill_between(baseline_epochs, 
                        baseline_mean - baseline_std, 
                        baseline_mean + baseline_std, 
                        alpha=0.3, color='green')
    
    # Formatting
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Normalized Loss (fraction of initial)', fontsize=12)
    plt.title('Mean Convergence Envelopes Across Training Regimes', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max_epochs)
    plt.ylim(0, 1.1)
    
    # Add annotations
    if resample_mean is not None:
        final_resample = resample_mean[-1]
        plt.annotate(f'Resample: {final_resample:.1%}', 
                    xy=(max_epochs, final_resample),
                    xytext=(max_epochs-5, final_resample+0.1),
                    fontsize=10, color='blue')
    
    if fixedmean_mean is not None:
        final_fixedmean = fixedmean_mean[-1]
        plt.annotate(f'FixedMean: {final_fixedmean:.1%}', 
                    xy=(max_epochs, final_fixedmean),
                    xytext=(max_epochs-5, final_fixedmean+0.1),
                    fontsize=10, color='orange')
    
    if baseline_mean is not None:
        final_baseline = baseline_mean[-1]
        plt.annotate(f'Baseline: {final_baseline:.1%}', 
                    xy=(max_epochs, final_baseline),
                    xytext=(max_epochs-5, final_baseline-0.1),
                    fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_2k_mean_convergence_envelopes_corrected.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    if resample_mean is not None:
        print(f"  Resample: Final loss = {resample_mean[-1]:.1%} of initial")
    if fixedmean_mean is not None:
        print(f"  FixedMean: Final loss = {fixedmean_mean[-1]:.1%} of initial")
    if baseline_mean is not None:
        print(f"  Baseline: Final loss = {baseline_mean[-1]:.1%} of initial")

if __name__ == "__main__":
    generate_figure()
