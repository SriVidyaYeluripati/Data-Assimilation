#!/usr/bin/env python3
"""
Master script to regenerate ALL corrected figures.

This script runs all individual plotting scripts to regenerate
the figures with Hans's requested corrections.

Usage:
    python3 Report/scripts_generated/generate_all_corrected_figures.py
"""

import subprocess
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# List of all plotting scripts to run
SCRIPTS = [
    "plot_4_3a_rmse_distributions_logscale.py",
    "plot_4_4a_obs_mode_rmse_logscale.py",
    "plot_4_5a_trajectory_fidelity.py",
    "plot_4_5b_error_evolution.py",
    "plot_4_6a_stability.py",
]

def main():
    """Run all plotting scripts."""
    print("=" * 60)
    print("Generating all corrected figures...")
    print("=" * 60)
    
    success_count = 0
    failed_scripts = []
    
    for script in SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script)
        print(f"\n[Running] {script}")
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(SCRIPT_DIR)
            )
            
            if result.returncode == 0:
                print(f"  [OK] {result.stdout.strip()}")
                success_count += 1
            else:
                print(f"  [ERROR] {result.stderr.strip()}")
                failed_scripts.append(script)
                
        except Exception as e:
            print(f"  [EXCEPTION] {str(e)}")
            failed_scripts.append(script)
    
    print("\n" + "=" * 60)
    print(f"Summary: {success_count}/{len(SCRIPTS)} scripts succeeded")
    
    if failed_scripts:
        print("Failed scripts:")
        for s in failed_scripts:
            print(f"  - {s}")
    
    print("=" * 60)
    
    return 0 if not failed_scripts else 1

if __name__ == "__main__":
    sys.exit(main())
