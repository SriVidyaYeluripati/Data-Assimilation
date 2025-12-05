#!/usr/bin/env python3
"""
Master script to regenerate all figures Hans commented on.

This script runs all individual plotting scripts to regenerate the
ORIGINAL figures with Hans's requested corrections:

1. 4_3a_resample_rmse_distributions.png → Log scale, model labels
2. 4_4a_post_assimilation_rmse.png → Log scale, observation mode notation
3. 4_5a_trajectory_fidelity_comparison.png → f_θ notation, clear labels
4. 4_5b_error_evolution_profiles.png → Regime comparison
5. 4_6a_background_sampling_stability.png → Stability labels

Output directory: Report/figures_new/
"""

import subprocess
import os
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    """Run a plotting script."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if os.path.exists(script_path):
        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print('='*60)
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        return result.returncode == 0
    else:
        print(f"Script not found: {script_path}")
        return False

def main():
    """Run all plotting scripts."""
    scripts = [
        "plot_4_3a_rmse_distributions_logscale.py",
        "plot_4_4a_obs_mode_rmse_logscale.py",
        "plot_4_5a_trajectory_fidelity.py",
        "plot_4_5b_error_evolution.py",
        "plot_4_6a_stability.py",
    ]
    
    print("="*60)
    print("REGENERATING FIGURES WITH HANS'S CORRECTIONS")
    print("="*60)
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(scripts)} figures generated successfully")
    print("Output directory: Report/figures_new/")
    print("="*60)

if __name__ == "__main__":
    main()
