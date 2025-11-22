#!/usr/bin/env python3
"""
Generate 6 publication-quality comparison plots for Lorenz-63 data assimilation experiments.

This script loads all experiment metrics, aggregates them using formulas defined in the
scientific report, and generates six publication-quality PNG figures for Section 4.

Author: GitHub Copilot
Date: 2025-11-22
"""

import os
import sys
import glob
import json
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set random seed for deterministic behavior
np.random.seed(42)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 7
})

# STRICT COLOR & STYLE CONVENTIONS
# Architectures (consistent across all 6 plots)
ARCH_COLORS = {
    'MLP': 'blue',      # Blue
    'GRU': 'green',     # Green
    'LSTM': 'red'       # Red
}

# Regime linestyles
REGIME_STYLES = {
    'baseline': ':',     # Dotted
    'fixedmean': '--',   # Dashed
    'resample': '-'      # Solid
}

# Small epsilon for numerical stability
EPSILON = 1e-8


def extract_info_from_path(filepath):
    """
    Extract regime, architecture, mode, and sigma from file path.
    
    Args:
        filepath: Path to the metric file
        
    Returns:
        dict with regime info or None if not identifiable
    """
    path = str(filepath)
    
    # Extract regime from path
    regime = None
    if '/baseline/' in path or '\\baseline\\' in path:
        regime = 'baseline'
    elif '/fixedmean' in path or '\\fixedmean' in path:
        regime = 'fixedmean'
    elif '/resample' in path or '\\resample' in path:
        regime = 'resample'
    
    return regime


def parse_csv_metrics(filepath):
    """
    Parse CSV metric files and return list of normalized records.
    
    Each record contains:
        regime, architecture, mode, sigma_obs, seed,
        rmse_b, rmse_a, improvement_bg, hausdorff, diverged
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dictionaries, each representing one metric record
    """
    try:
        df = pd.read_csv(filepath)
        records = []
        
        regime = extract_info_from_path(filepath)
        if regime is None:
            return records
        
        # Handle different CSV formats
        if 'mode' in df.columns and 'model' in df.columns and 'sigma' in df.columns:
            for _, row in df.iterrows():
                mode_raw = str(row['mode']).lower().strip()
                model_raw = str(row['model']).lower().strip()
                
                # Normalize mode
                if mode_raw in ['x', 'x_only']:
                    mode = 'x'
                elif mode_raw in ['xy', 'x_y', 'xy_only']:
                    mode = 'xy'
                elif mode_raw in ['x2', 'x_squared', 'xsquared']:
                    mode = 'x2'
                else:
                    mode = mode_raw
                
                # Normalize architecture
                if 'mlp' in model_raw or model_raw == 'baseline_no_mean':
                    architecture = 'MLP'
                elif 'gru' in model_raw:
                    architecture = 'GRU'
                elif 'lstm' in model_raw:
                    architecture = 'LSTM'
                else:
                    continue  # Skip unknown architectures
                
                # Get sigma
                try:
                    sigma = float(row['sigma'])
                except:
                    continue
                
                # Get RMSE values
                rmse_b = row.get('rmse_b', row.get('mean_rmse_b', np.nan))
                rmse_a = row.get('rmse_a', row.get('mean_rmse_a', np.nan))
                
                # Calculate improvement_bg using EXACT formula from report
                # Improvement_bg = (RMSE_before - RMSE_after) / (RMSE_before + 1e-8)
                if not np.isnan(rmse_b) and not np.isnan(rmse_a):
                    improvement_bg = (rmse_b - rmse_a) / (rmse_b + EPSILON)
                else:
                    improvement_bg = np.nan
                
                # Get hausdorff if available (normalized)
                hausdorff = row.get('hausdorff', row.get('hausdorff_norm', np.nan))
                
                # Check for divergence
                # Divergence rule: flagged if rmse_a is very high or improvement is very negative
                diverged = 0
                if not np.isnan(rmse_a):
                    if rmse_a > 1e6 or (not np.isnan(improvement_bg) and improvement_bg < -1.0):
                        diverged = 1
                
                # Seed (not available in current data, use 0 as placeholder)
                seed = row.get('seed', 0)
                
                record = {
                    'regime': regime,
                    'architecture': architecture,
                    'mode': mode,
                    'sigma_obs': sigma,
                    'seed': seed,
                    'rmse_b': float(rmse_b) if not np.isnan(rmse_b) else np.nan,
                    'rmse_a': float(rmse_a) if not np.isnan(rmse_a) else np.nan,
                    'improvement_bg': float(improvement_bg) if not np.isnan(improvement_bg) else np.nan,
                    'hausdorff': float(hausdorff) if not np.isnan(hausdorff) else np.nan,
                    'diverged': diverged
                }
                records.append(record)
        
        return records
    except Exception as e:
        warnings.warn(f"Failed to parse CSV {filepath}: {e}")
        return []


def collect_all_metrics(results_dirs):
    """
    Recursively collect all metrics from specified directories.
    
    Args:
        results_dirs: List of directories to search
        
    Returns:
        pandas DataFrame with all metrics
    """
    all_records = []
    file_count = 0
    
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            warnings.warn(f"Directory {results_dir} does not exist, skipping")
            continue
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(results_dir, '**/metrics/*.csv'), recursive=True)
        for csv_file in csv_files:
            records = parse_csv_metrics(csv_file)
            all_records.extend(records)
            if records:
                file_count += 1
    
    print(f"  Parsed {file_count} metric files")
    return pd.DataFrame(all_records)


def aggregate_metrics(df):
    """
    Aggregate metrics by (regime, architecture, mode, sigma_obs).
    
    Computes:
        - mean, std of rmse_a
        - median, IQR of rmse_a
        - mean, std of improvement_bg
        - median, IQR of improvement_bg
        - mean, std of hausdorff
        - median, IQR of hausdorff
        - divergence_rate = mean(diverged)
    
    Args:
        df: DataFrame with individual metric records
        
    Returns:
        DataFrame with aggregated statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Remove diverged cases for aggregation statistics (but count them for divergence rate)
    df_valid = df[df['diverged'] == 0].copy()
    
    # Group by key dimensions
    group_cols = ['regime', 'architecture', 'mode', 'sigma_obs']
    
    agg_results = []
    for name, group in df.groupby(group_cols):
        regime, architecture, mode, sigma_obs = name
        
        # Get all cases (including diverged) for divergence rate
        all_cases = df[(df['regime'] == regime) & 
                      (df['architecture'] == architecture) & 
                      (df['mode'] == mode) & 
                      (df['sigma_obs'] == sigma_obs)]
        
        # Calculate statistics
        stats = {
            'regime': regime,
            'architecture': architecture,
            'mode': mode,
            'sigma_obs': sigma_obs,
            'n_samples': len(group),
            'n_diverged': int(all_cases['diverged'].sum()),
        }
        
        # Add statistics for each metric
        for metric in ['rmse_a', 'improvement_bg', 'hausdorff']:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_mean'] = values.mean()
                    stats[f'{metric}_std'] = values.std()
                    stats[f'{metric}_median'] = values.median()
                    stats[f'{metric}_q25'] = values.quantile(0.25)
                    stats[f'{metric}_q75'] = values.quantile(0.75)
                else:
                    stats[f'{metric}_mean'] = np.nan
                    stats[f'{metric}_std'] = np.nan
                    stats[f'{metric}_median'] = np.nan
                    stats[f'{metric}_q25'] = np.nan
                    stats[f'{metric}_q75'] = np.nan
        
        # Divergence rate
        total_in_group = len(all_cases)
        stats['divergence_rate'] = stats['n_diverged'] / total_in_group if total_in_group > 0 else 0
        
        agg_results.append(stats)
    
    # Create DataFrame and sort by mode, architecture, regime, sigma
    agg_df = pd.DataFrame(agg_results)
    if not agg_df.empty:
        # Sort: mode (X, XY, X²), then architecture, then regime, then sigma
        mode_order = {'x': 0, 'xy': 1, 'x2': 2}
        agg_df['mode_sort'] = agg_df['mode'].map(mode_order)
        agg_df = agg_df.sort_values(['mode_sort', 'architecture', 'regime', 'sigma_obs'])
        agg_df = agg_df.drop('mode_sort', axis=1)
    
    return agg_df


def plot_core_figures(agg_df, output_dir):
    """
    Generate 3 global comparison figures (Section 4.2).
    
    Each figure is a 3-panel horizontal plot (X, XY, X²).
    
    Args:
        agg_df: Aggregated metrics DataFrame
        output_dir: Output directory for PNG files
    """
    
    modes = ['x', 'xy', 'x2']
    mode_labels = {'x': 'X', 'xy': 'XY', 'x2': 'X²'}
    
    # Figure 1: RMSE by mode
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Post-Assimilation RMSE by Observation Mode', fontsize=14, y=1.00)
    
    # Figure 2: Improvement by mode
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Improvement by Observation Mode', fontsize=14, y=1.00)
    
    # Figure 3: Hausdorff by mode
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle('Hausdorff Distance by Observation Mode', fontsize=14, y=1.00)
    
    # Determine y-limits across all panels for consistency
    rmse_min, rmse_max = np.inf, -np.inf
    improv_min, improv_max = np.inf, -np.inf
    haus_min, haus_max = np.inf, -np.inf
    
    for mode in modes:
        mode_data = agg_df[agg_df['mode'] == mode]
        for _, row in mode_data.iterrows():
            if not np.isnan(row.get('rmse_a_mean', np.nan)):
                rmse_min = min(rmse_min, row['rmse_a_mean'] - row.get('rmse_a_std', 0))
                rmse_max = max(rmse_max, row['rmse_a_mean'] + row.get('rmse_a_std', 0))
            if not np.isnan(row.get('improvement_bg_mean', np.nan)):
                improv_min = min(improv_min, row['improvement_bg_mean'] - row.get('improvement_bg_std', 0))
                improv_max = max(improv_max, row['improvement_bg_mean'] + row.get('improvement_bg_std', 0))
            if not np.isnan(row.get('hausdorff_mean', np.nan)):
                haus_min = min(haus_min, row['hausdorff_mean'] - row.get('hausdorff_std', 0))
                haus_max = max(haus_max, row['hausdorff_mean'] + row.get('hausdorff_std', 0))
    
    # Add padding to y-limits
    if rmse_max > rmse_min:
        rmse_range = rmse_max - rmse_min
        rmse_min -= 0.1 * rmse_range
        rmse_max += 0.1 * rmse_range
    if improv_max > improv_min:
        improv_range = improv_max - improv_min
        improv_min -= 0.1 * improv_range
        improv_max += 0.1 * improv_range
    if haus_max > haus_min:
        haus_range = haus_max - haus_min
        haus_min -= 0.1 * haus_range
        haus_max += 0.1 * haus_range
    
    # Plot each mode
    for mode_idx, mode in enumerate(modes):
        ax1 = axes1[mode_idx]
        ax2 = axes2[mode_idx]
        ax3 = axes3[mode_idx]
        
        mode_data = agg_df[agg_df['mode'] == mode]
        
        # Plot for each architecture × regime combination
        for arch in ['MLP', 'GRU', 'LSTM']:
            for regime in ['baseline', 'fixedmean', 'resample']:
                subset = mode_data[(mode_data['architecture'] == arch) & 
                                  (mode_data['regime'] == regime)]
                
                if len(subset) == 0:
                    continue
                
                subset = subset.sort_values('sigma_obs')
                sigmas = subset['sigma_obs'].values
                
                label = f'{arch}-{regime.capitalize()}'
                color = ARCH_COLORS[arch]
                style = REGIME_STYLES[regime]
                
                # RMSE plot
                if 'rmse_a_mean' in subset.columns:
                    rmse_mean = subset['rmse_a_mean'].values
                    rmse_std = subset['rmse_a_std'].values
                    
                    ax1.plot(sigmas, rmse_mean, color=color, linestyle=style, 
                            marker='o', label=label, alpha=0.85, linewidth=2.5)
                    ax1.fill_between(sigmas, rmse_mean - rmse_std, rmse_mean + rmse_std,
                                    color=color, alpha=0.15)
                
                # Improvement plot
                if 'improvement_bg_mean' in subset.columns:
                    improv_mean = subset['improvement_bg_mean'].values * 100  # Convert to percentage
                    improv_std = subset['improvement_bg_std'].values * 100
                    
                    ax2.plot(sigmas, improv_mean, color=color, linestyle=style,
                            marker='o', label=label, alpha=0.85, linewidth=2.5)
                    ax2.fill_between(sigmas, improv_mean - improv_std, improv_mean + improv_std,
                                    color=color, alpha=0.15)
                
                # Hausdorff plot
                if 'hausdorff_mean' in subset.columns:
                    haus_mean = subset['hausdorff_mean'].values
                    haus_std = subset['hausdorff_std'].values
                    
                    # Only plot if we have non-NaN data
                    if not np.all(np.isnan(haus_mean)):
                        ax3.plot(sigmas, haus_mean, color=color, linestyle=style,
                                marker='o', label=label, alpha=0.85, linewidth=2.5)
                        ax3.fill_between(sigmas, haus_mean - haus_std, haus_mean + haus_std,
                                        color=color, alpha=0.15)
        
        # Configure axes
        ax1.set_xlabel('σ_obs', fontsize=12)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_title(f'Mode: {mode_labels[mode]}', fontsize=13)
        if rmse_max > rmse_min:
            ax1.set_ylim(rmse_min, rmse_max)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('σ_obs', fontsize=12)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title(f'Mode: {mode_labels[mode]}', fontsize=13)
        if improv_max > improv_min:
            ax2.set_ylim(improv_min * 100, improv_max * 100)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('σ_obs', fontsize=12)
        ax3.set_ylabel('Hausdorff Distance', fontsize=12)
        ax3.set_title(f'Mode: {mode_labels[mode]}', fontsize=13)
        ax3.grid(True, alpha=0.3)
        
        # Check if we have any Hausdorff data
        has_hausdorff_data = False
        if 'hausdorff_mean' in mode_data.columns:
            has_hausdorff_data = mode_data['hausdorff_mean'].notna().any()
        if not has_hausdorff_data:
            ax3.text(0.5, 0.5, 'Data not yet available\n(Placeholder for future metric)', 
                    ha='center', va='center', transform=ax3.transAxes, 
                    fontsize=11, alpha=0.5, style='italic')
    
    # Add legend outside panels (right side)
    handles, labels = axes1[0].get_legend_handles_labels()
    if handles:
        fig1.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   ncol=1, frameon=True, fancybox=True)
        fig2.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   ncol=1, frameon=True, fancybox=True)
        handles3, labels3 = axes3[0].get_legend_handles_labels()
        if handles3:
            fig3.legend(handles3, labels3, loc='center left', bbox_to_anchor=(1.0, 0.5),
                       ncol=1, frameon=True, fancybox=True)
    
    # Save figures
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, 'core_rmse_by_mode.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, 'core_improvement_by_mode.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    plt.figure(fig3.number)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, 'core_hausdorff_by_mode.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"  ✓ Generated core_rmse_by_mode.png")
    print(f"  ✓ Generated core_improvement_by_mode.png")
    print(f"  ✓ Generated core_hausdorff_by_mode.png")


def plot_mode_specific_figures(agg_df, output_dir):
    """
    Generate 3 mode-specific zoom figures (Section 4.3).
    
    Each figure is a 3×1 vertical panel for a single observation mode.
    
    Args:
        agg_df: Aggregated metrics DataFrame
        output_dir: Output directory for PNG files
    """
    
    modes = ['x', 'xy', 'x2']
    mode_labels = {'x': 'X', 'xy': 'XY', 'x2': 'X²'}
    
    for mode in modes:
        fig, axes = plt.subplots(3, 1, figsize=(9, 12))
        fig.suptitle(f'Mode {mode_labels[mode]} Summary', fontsize=14, y=0.995)
        
        mode_data = agg_df[agg_df['mode'] == mode]
        
        # Panel 1: RMSE vs σ_obs
        ax0 = axes[0]
        # Panel 2: Improvement vs σ_obs
        ax1 = axes[1]
        # Panel 3: Hausdorff vs σ_obs
        ax2 = axes[2]
        
        for arch in ['MLP', 'GRU', 'LSTM']:
            for regime in ['baseline', 'fixedmean', 'resample']:
                subset = mode_data[(mode_data['architecture'] == arch) & 
                                  (mode_data['regime'] == regime)]
                
                if len(subset) == 0:
                    continue
                
                subset = subset.sort_values('sigma_obs')
                sigmas = subset['sigma_obs'].values
                
                label = f'{arch}-{regime.capitalize()}'
                color = ARCH_COLORS[arch]
                style = REGIME_STYLES[regime]
                
                # RMSE panel
                if 'rmse_a_mean' in subset.columns:
                    rmse_mean = subset['rmse_a_mean'].values
                    rmse_std = subset['rmse_a_std'].values
                    
                    ax0.plot(sigmas, rmse_mean, color=color, linestyle=style,
                            marker='o', label=label, alpha=0.85, linewidth=2.5)
                    ax0.fill_between(sigmas, rmse_mean - rmse_std, rmse_mean + rmse_std,
                                    color=color, alpha=0.15)
                
                # Improvement panel
                if 'improvement_bg_mean' in subset.columns:
                    improv_mean = subset['improvement_bg_mean'].values * 100
                    improv_std = subset['improvement_bg_std'].values * 100
                    
                    ax1.plot(sigmas, improv_mean, color=color, linestyle=style,
                            marker='o', label=label, alpha=0.85, linewidth=2.5)
                    ax1.fill_between(sigmas, improv_mean - improv_std, improv_mean + improv_std,
                                    color=color, alpha=0.15)
                
                # Hausdorff panel
                if 'hausdorff_mean' in subset.columns:
                    haus_mean = subset['hausdorff_mean'].values
                    haus_std = subset['hausdorff_std'].values
                    
                    # Only plot if we have non-NaN data
                    if not np.all(np.isnan(haus_mean)):
                        ax2.plot(sigmas, haus_mean, color=color, linestyle=style,
                                marker='o', label=label, alpha=0.85, linewidth=2.5)
                        ax2.fill_between(sigmas, haus_mean - haus_std, haus_mean + haus_std,
                                        color=color, alpha=0.15)
        
        # Configure axes
        ax0.set_xlabel('σ_obs', fontsize=12)
        ax0.set_ylabel('RMSE', fontsize=12)
        ax0.set_title('Post-Assimilation RMSE', fontsize=13)
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc='best', fontsize=9)
        
        ax1.set_xlabel('σ_obs', fontsize=12)
        ax1.set_ylabel('Improvement (%)', fontsize=12)
        ax1.set_title('Improvement', fontsize=13)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        ax2.set_xlabel('σ_obs', fontsize=12)
        ax2.set_ylabel('Hausdorff Distance', fontsize=12)
        ax2.set_title('Hausdorff Distance', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Check if we have any Hausdorff data for this mode
        has_hausdorff_data = False
        if 'hausdorff_mean' in mode_data.columns:
            has_hausdorff_data = mode_data['hausdorff_mean'].notna().any()
        if has_hausdorff_data:
            ax2.legend(loc='best', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Data not yet available\n(Placeholder for future metric)', 
                    ha='center', va='center', transform=ax2.transAxes, 
                    fontsize=11, alpha=0.5, style='italic')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'mode_{mode}_summary.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Generated mode_{mode}_summary.png")


def main():
    """Main function to orchestrate the entire process."""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    comprehensive_dir = project_root / 'comprehensive_diagnostics'
    output_dir = project_root / 'Report' / 'figs'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Lorenz-63 Data Assimilation: Comparison Figure Generator")
    print("="*70)
    print()
    
    # Step 1: Collect all metrics
    print("Step 1: Collecting metrics from all regime directories...")
    results_dirs = [results_dir, comprehensive_dir]
    df = collect_all_metrics(results_dirs)
    
    if df.empty:
        print("ERROR: No metrics found. Please check that metric files exist.")
        print(f"Searched in: {results_dir}")
        return 1
    
    print(f"  Found {len(df)} metric entries")
    print(f"  Regimes: {sorted(df['regime'].unique().tolist())}")
    print(f"  Architectures: {sorted(df['architecture'].unique().tolist())}")
    print(f"  Modes: {sorted(df['mode'].unique().tolist())}")
    print()
    
    # Step 2: Aggregate metrics
    print("Step 2: Aggregating metrics by (regime, architecture, mode, sigma_obs)...")
    agg_df = aggregate_metrics(df)
    
    if agg_df.empty:
        print("ERROR: No valid metrics for aggregation.")
        return 1
    
    print(f"  Generated {len(agg_df)} aggregated entries")
    print()
    
    # Step 3: Save aggregated table
    print("Step 3: Saving aggregated metrics table...")
    output_csv = output_dir / 'aggregated_metrics_summary.csv'
    agg_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved to {output_csv}")
    print()
    
    # Step 4: Generate global comparison figures (Section 4.2)
    print("Step 4: Generating global comparison figures (Section 4.2)...")
    plot_core_figures(agg_df, output_dir)
    print()
    
    # Step 5: Generate mode-specific figures (Section 4.3)
    print("Step 5: Generating mode-specific zoom figures (Section 4.3)...")
    plot_mode_specific_figures(agg_df, output_dir)
    print()
    
    print("="*70)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print()
    print(f"Output location: {output_dir}")
    print()
    print("Generated files:")
    print("  1. aggregated_metrics_summary.csv")
    print("  2. core_rmse_by_mode.png")
    print("  3. core_improvement_by_mode.png")
    print("  4. core_hausdorff_by_mode.png")
    print("  5. mode_x_summary.png")
    print("  6. mode_xy_summary.png")
    print("  7. mode_x2_summary.png")
    print()
    print("Summary:")
    print(f"  • Metric files parsed: {len(df)}")
    print(f"  • Aggregated metric rows: {len(agg_df)}")
    print(f"  • Color scheme: MLP=blue, GRU=green, LSTM=red")
    print(f"  • Line styles: baseline=dotted, fixedmean=dashed, resample=solid")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
