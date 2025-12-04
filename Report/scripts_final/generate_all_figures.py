#!/usr/bin/env python3
"""
Master script to generate all figures for the revised report.
Run this to regenerate all figures from experimental data.
"""

import os
import sys

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_fig1_rmse_comparison import plot_rmse_comparison
from plot_fig2_architecture import plot_architecture_comparison
from plot_fig3_divergence import plot_fixedmean_divergence
from plot_fig4_improvement import plot_improvement_analysis

def main():
    print("="*60)
    print("GENERATING ALL FIGURES FOR REVISED REPORT")
    print("="*60)
    
    figures = []
    
    print("\n[1/4] RMSE Comparison Figure...")
    figures.append(plot_rmse_comparison())
    
    print("\n[2/4] Architecture Comparison Figure...")
    figures.append(plot_architecture_comparison())
    
    print("\n[3/4] FixedMean Divergence Figure (Log Scale)...")
    figures.append(plot_fixedmean_divergence())
    
    print("\n[4/4] Improvement Analysis Figure...")
    figures.append(plot_improvement_analysis())
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
    
    for fig in figures:
        print(f"  - {fig}")
    
    return figures

if __name__ == '__main__':
    main()
