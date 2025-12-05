# figures_new Directory

This directory contains regenerated figures and the scripts to produce them from the stored `.npy` result files.

## Contents

- `generate_figures.py` - Original Python script to regenerate key figures
- `*.png` - Generated figure files

See also: `Report/scripts_generated/` for individual plotting scripts

## Quick Usage

To regenerate all figures using the master script:
```bash
python3 Report/scripts_generated/generate_all_figures.py
```

Or run individual scripts:
```bash
python3 Report/scripts_generated/plot_rmse_boxplot_logscale.py
python3 Report/scripts_generated/plot_attractor_projection.py
python3 Report/scripts_generated/plot_rmse_vs_rmdse.py
```

## Available Figures

| Figure Name | Description | Hans Comment Addressed |
|-------------|-------------|------------------------|
| `attractor_projection_new.png` | Lorenz-63 attractor X-Y and X-Z projections (50,000 points) | Visualization quality |
| `trajectory_sample_new.png` | State reconstruction: Truth vs Analysis vs Background | Data clarity |
| `rmse_comparison_new.png` | RMSE comparison across modes, noise, architectures | Performance evaluation |
| `rmse_boxplot_logscale.png` | RMSE boxplot with logarithmic y-scale | Comment #105: "can you make it a log plot?" |
| `rmse_vs_rmdse_comparison.png` | RMSE vs RMdSE comparison for robust outlier handling | Meeting: robust metrics discussion |
| `loss_curves_comparison.png` | Training loss convergence across configurations | Training analysis |

## Data Sources

The figures are generated from `.npy` files stored in:
- `results/baseline/diagnostics/` - Baseline regime results
- `results/resample /run_20251008_134240 /diagnostics/` - Resample regime results
- `data /raw/` - Raw trajectory data

Result files follow the naming convention:
- `truth_{mode}_{arch}_n{noise}.npy`
- `analysis_{mode}_{arch}_n{noise}.npy`
- `background_{mode}_{arch}_n{noise}.npy`

Where:
- `mode` ∈ {x, xy, x2}
- `arch` ∈ {mlp, gru, lstm, baseline}
- `noise` ∈ {0.05, 0.1, 0.5, 1.0}

## Robust Metrics

The script includes computation of both RMSE and RMdSE (Root Median Squared Error):
- **RMSE**: Standard metric, sensitive to outliers from catastrophic failures
- **RMdSE**: Robust alternative using median instead of mean, better characterizes typical performance when a small fraction of runs diverge

## Note

The existing figures in `Report/` are retained. These new figures are additional/updated versions that can be used to replace specific figures if desired. The original figures are not overwritten as per the requirements.
