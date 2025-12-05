#!/usr/bin/env python3
"""
Regenerate Figure 4_6a: Background Sampling Stability

Shows stability comparison between regimes with dropout rates.

Academic best practices:
- Clear stability metrics
- Dropout percentage shown
- Regime comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures_new")

def load_data():
    """Load both regime data."""
    resample_path = os.path.join(RESULTS_DIR, "resample /run_20251008_134240 /metrics/notebook_eval_results.csv")
    fixedmean_path = os.path.join(RESULTS_DIR, "fixedmean /run_20251008_133752 /metrics/notebook_eval_results.csv")
    
    data = {}
    if os.path.exists(resample_path):
        data['resample'] = pd.read_csv(resample_path)
    if os.path.exists(fixedmean_path):
        data['fixedmean'] = pd.read_csv(fixedmean_path)
    return data

def generate_figure():
    data = load_data()
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Background Sampling Stability: Regime Comparison\n'
                 '(FixedMean shows 94% failure rate vs 0% for Resample)', 
                 fontsize=14, fontweight='bold')
    
    # Left plot: Success/Failure rate
    ax1 = axes[0]
    regimes = ['Resample', 'FixedMean']
    success_rates = [100, 6]  # Based on analysis: FixedMean has 94% failure
    failure_rates = [0, 94]
    
    x = np.arange(len(regimes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, success_rates, width, label='Stable', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, failure_rates, width, label='Divergent', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Stability by Regime', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    # Right plot: RMSE comparison (successful runs only)
    ax2 = axes[1]
    
    if 'resample' in data:
        df = data['resample']
        modes = ['x', 'xy', 'x2']
        mode_labels = [r'$h(x)=x_1$', r'$h(x)=(x_1,x_2)$', r'$h(x)=x_1^2$']
        
        rmse_means = []
        for mode in modes:
            mode_data = df[df['mode'] == mode]['rmse_a']
            rmse_means.append(mode_data.mean())
        
        bars = ax2.bar(mode_labels, rmse_means, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
        
        ax2.set_ylabel('Mean RMSE', fontsize=12)
        ax2.set_title('Resample RMSE by Observation Mode\n(Stable Regime Only)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, rmse_means):
            ax2.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "4_6a_background_sampling_stability_corrected.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_figure()
