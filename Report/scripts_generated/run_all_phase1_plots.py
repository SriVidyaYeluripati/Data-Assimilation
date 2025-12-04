#!/usr/bin/env python3
"""
Master script to run ALL Phase 1 figure generation scripts.
Creates new figures in Report/figures_new/ based on Hans's comments.

Usage: python run_all_phase1_plots.py
"""

import os
import sys
import subprocess

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new')

# List of scripts to run
SCRIPTS = [
    'plot_hausdorff_log_scale.py',       # ID 105: Log-scale request
    'plot_error_evolution_with_mode.py', # ID 112: Mode specification
    'plot_lobe_occupancy_detailed.py',   # ID 113, 114: Noise level and model spec
    'plot_rmse_consolidated.py',         # ID 111: Consolidate plots
    'plot_main_rmse_summary.py',         # Single metric focus (Hans's main guidance)
    'plot_classical_da_annotated.py',    # ID 57: Source and Phi label
]

def main():
    """Run all plotting scripts."""
    print("=" * 60)
    print("PHASE 1: Regenerating Figures Based on Hans's Comments")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Install required packages
    print("Installing required packages...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                   'numpy', 'pandas', 'matplotlib'], check=False)
    print()
    
    # Run each script
    success_count = 0
    failed_scripts = []
    
    for script in SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script)
        print(f"Running: {script}")
        print("-" * 40)
        
        try:
            result = subprocess.run([sys.executable, script_path], 
                                   capture_output=True, text=True, 
                                   cwd=REPORT_DIR)
            if result.returncode == 0:
                print(result.stdout)
                success_count += 1
            else:
                print(f"Error: {result.stderr}")
                failed_scripts.append(script)
        except Exception as e:
            print(f"Exception: {e}")
            failed_scripts.append(script)
        
        print()
    
    # Summary
    print("=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"Scripts run: {len(SCRIPTS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print(f"\nFailed scripts: {', '.join(failed_scripts)}")
    
    # List generated figures
    print(f"\nGenerated figures in {OUTPUT_DIR}:")
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                filepath = os.path.join(OUTPUT_DIR, f)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  - {f} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
