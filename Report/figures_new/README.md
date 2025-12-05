# figures_new Directory

This directory contains regenerated figures and the scripts to produce them from the stored `.npy` result files.

## Contents

- `generate_figures.py` - Python script to regenerate key figures
- `*.png` - Generated figure files

## Usage

To regenerate all figures:
```bash
python3 generate_figures.py
```

To regenerate a specific figure:
```bash
python3 generate_figures.py rmse_comparison
python3 generate_figures.py trajectory_sample
python3 generate_figures.py attractor_projection
python3 generate_figures.py rmse_boxplot_logscale
python3 generate_figures.py robust_metrics
```

## Available Figures

| Figure Name | Description |
|-------------|-------------|
| `rmse_comparison_new.png` | RMSE comparison across noise levels and observation modes |
| `trajectory_sample_new.png` | Sample trajectory reconstruction showing truth vs analysis |
| `attractor_projection_new.png` | Phase-space projections of the attractor |
| `rmse_boxplot_logscale.png` | RMSE boxplot with logarithmic y-scale (addresses Hans comment #105) |
| `rmse_vs_rmdse_comparison.png` | Comparison of RMSE vs RMdSE (Root Median Squared Error) for robust outlier handling |

## Data Sources

The figures are generated from `.npy` files stored in:
- `results/baseline/diagnostics/` - Baseline regime results

Result files follow the naming convention:
- `truth_{mode}_baseline_n{noise}.npy`
- `analysis_{mode}_baseline_n{noise}.npy`
- `background_{mode}_baseline_n{noise}.npy`

Where:
- `mode` ∈ {x, xy, x2}
- `noise` ∈ {0.05, 0.1, 0.5, 1.0}

## Robust Metrics

The script includes computation of both RMSE and RMdSE (Root Median Squared Error):
- **RMSE**: Standard metric, sensitive to outliers from catastrophic failures
- **RMdSE**: Robust alternative using median instead of mean, better characterizes typical performance when a small fraction of runs diverge

## Note

The existing figures in `Report/` are retained. These new figures are additional/updated versions that can be used to replace specific figures if desired. The original figures are not overwritten as per the requirements.
