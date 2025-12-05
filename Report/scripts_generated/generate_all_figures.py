#!/usr/bin/env python3
"""
generate_all_figures.py

Master script that generates all figures addressing Hans's comments from:
1. PDF annotations (PR_AIVar_Report_24_11_2025_some_feedback.pdf)
2. Meeting discussions

This script:
- Runs all individual plotting scripts
- Documents which figure addresses which comment
- Outputs to Report/figures_new/ without touching original figures

Usage:
    python Report/scripts_generated/generate_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FIGURES_NEW_DIR = SCRIPT_DIR.parent / "figures_new"

# Mapping of Hans's comments to figure scripts
FIGURE_MAPPING = {
    "plot_rmse_boxplot_logscale.py": {
        "comment": "#105: 'can you make it a log plot?'",
        "output": "rmse_boxplot_logscale.png",
        "description": "RMSE boxplot with logarithmic y-axis to better visualize outlier distribution"
    },
    "plot_rmse_vs_rmdse.py": {
        "comment": "Meeting: Robust metrics when RMSE affected by outliers",
        "output": "rmse_vs_rmdse_comparison.png", 
        "description": "Comparison of RMSE vs RMdSE (Root Median Squared Error) for robustness"
    },
    "plot_attractor_projection.py": {
        "comment": "Visualization quality - attractor geometry",
        "output": "attractor_projection_new.png",
        "description": "Lorenz-63 attractor projections with sufficient data points"
    },
    "plot_trajectory_reconstruction.py": {
        "comment": "Data clarity - truth vs analysis comparison",
        "output": "trajectory_sample_new.png",
        "description": "Trajectory reconstruction showing truth, analysis, and background"
    },
    "plot_loss_curves.py": {
        "comment": "Training convergence analysis",
        "output": "loss_curves_comparison.png",
        "description": "Training loss curves across architectures and noise levels"
    },
    "plot_rmse_comparison.py": {
        "comment": "Performance evaluation across configurations",
        "output": "rmse_comparison_new.png",
        "description": "RMSE comparison across modes, noise levels, and architectures"
    },
}


def run_script(script_name):
    """Run a single plotting script and capture output."""
    script_path = SCRIPT_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {script_name}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ FAILED: {script_name}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {script_name}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {script_name}: {e}")
        return False


def main():
    """Run all figure generation scripts."""
    print("=" * 70)
    print("FIGURE GENERATION FOR HANS COMMENT RESOLUTION")
    print("=" * 70)
    print(f"\nOutput directory: {FIGURES_NEW_DIR}")
    print(f"Number of scripts: {len(FIGURE_MAPPING)}")
    
    # Ensure output directory exists
    FIGURES_NEW_DIR.mkdir(exist_ok=True)
    
    results = {}
    for script_name, info in FIGURE_MAPPING.items():
        print(f"\n--- {info['comment']} ---")
        success = run_script(script_name)
        results[script_name] = success
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nSuccessful: {successful}/{total}")
    print(f"\nGenerated figures:")
    
    for script_name, info in FIGURE_MAPPING.items():
        status = "✓" if results[script_name] else "✗"
        output_file = FIGURES_NEW_DIR / info['output']
        exists = "EXISTS" if output_file.exists() else "MISSING"
        print(f"  {status} {info['output']} [{exists}]")
        print(f"      → {info['description']}")
    
    print("\n" + "=" * 70)
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
