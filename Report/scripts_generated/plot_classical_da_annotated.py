#!/usr/bin/env python3
"""
Create Classical DA schematic with SOURCE CITATION and PHI label.
Addresses Hans's comment ID 57: "give the source for this image and show where exactly phi is here"

This script creates an annotated version of the conceptual diagram.
New figure: figures_new/classical_da_annotated.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Paths
REPORT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPORT_DIR, 'figures_new')

def plot_classical_da_annotated():
    """Create annotated classical DA / AI-Var concept diagram."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    bg_color = '#ecf0f1'
    obs_color = '#3498db'
    state_color = '#2ecc71'
    phi_color = '#e74c3c'
    analysis_color = '#9b59b6'
    
    # --- Background State Box ---
    bg_box = FancyBboxPatch((0.5, 5), 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor=state_color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(bg_box)
    ax.text(1.75, 5.75, r'$\mathbf{x}_b$', fontsize=16, ha='center', va='center', fontweight='bold')
    ax.text(1.75, 5.3, 'Background', fontsize=10, ha='center', va='center')
    
    # --- Observation Box ---
    obs_box = FancyBboxPatch((0.5, 2), 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor=obs_color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(obs_box)
    ax.text(1.75, 2.75, r'$\mathbf{y}$', fontsize=16, ha='center', va='center', fontweight='bold')
    ax.text(1.75, 2.3, 'Observations', fontsize=10, ha='center', va='center')
    
    # --- PHI (Analysis Functional) Box - EXPLICITLY LABELED ---
    phi_box = FancyBboxPatch((4, 3), 2, 2.5, boxstyle="round,pad=0.1",
                            facecolor=phi_color, edgecolor='black', linewidth=3, alpha=0.8)
    ax.add_patch(phi_box)
    ax.text(5, 4.5, r'$\Phi$', fontsize=24, ha='center', va='center', fontweight='bold', color='white')
    ax.text(5, 3.8, 'Analysis', fontsize=11, ha='center', va='center', color='white')
    ax.text(5, 3.4, 'Functional', fontsize=11, ha='center', va='center', color='white')
    
    # PHI annotation arrow and label
    ax.annotate(r'$\Phi$ maps $(\mathbf{y}, \mathbf{x}_b) \mapsto \mathbf{x}^a$',
                xy=(5, 5.5), xytext=(5, 7),
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # --- Analysis State Box ---
    analysis_box = FancyBboxPatch((7.5, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=analysis_color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(analysis_box)
    ax.text(8.5, 4.25, r'$\mathbf{x}^a$', fontsize=16, ha='center', va='center', fontweight='bold', color='white')
    ax.text(8.5, 3.8, 'Analysis', fontsize=10, ha='center', va='center', color='white')
    
    # --- Arrows ---
    # Background -> Phi
    ax.annotate('', xy=(4, 4.5), xytext=(3, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Observation -> Phi
    ax.annotate('', xy=(4, 4), xytext=(3, 2.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Phi -> Analysis
    ax.annotate('', xy=(7.5, 4.25), xytext=(6, 4.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # --- Covariance matrices labels ---
    ax.text(0.5, 7, r'$\mathbf{B}$ = Background Error Covariance', fontsize=10, ha='left')
    ax.text(0.5, 6.5, r'$\mathbf{R}$ = Observation Error Covariance', fontsize=10, ha='left')
    
    # --- Neural Network / AI-Var note ---
    nn_box = FancyBboxPatch((4, 0.5), 2, 1.2, boxstyle="round,pad=0.1",
                            facecolor='#f39c12', edgecolor='black', linewidth=1, alpha=0.6)
    ax.add_patch(nn_box)
    ax.text(5, 1.1, r'$f_\theta \approx \Phi$', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(5, 0.7, 'Neural Network', fontsize=9, ha='center', va='center')
    
    ax.annotate('', xy=(5, 3), xytext=(5, 1.7),
                arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=2, linestyle='dashed'))
    
    # --- Title and source citation ---
    ax.set_title('3D-Var / AI-Var Conceptual Diagram\n'
                 'Source: Adapted from Bocquet et al. (2024), arXiv:2406.00390\n'
                 r'The analysis functional $\Phi$ maps background and observations to the analysis state',
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'classical_da_annotated.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    plot_classical_da_annotated()
