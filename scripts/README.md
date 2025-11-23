# Section 4 Metrics and Plots Generator

This script computes all Section 4 metrics directly from trained model weights (.pth files), without relying on precomputed diagnostics or CSVs.

## Overview

The script:
1. Loads test data from `data/raw`, `data/obs`, and `data/splits`
2. Discovers all model files from:
   - `results/resample/run_20251008_134240/models`
   - `results/fixedmean/run_20251008_133752/models`
   - `results/baseline/models`
3. Evaluates each model on test trajectories
4. Computes comprehensive metrics
5. Generates summary CSV and visualization plots

## Usage

```bash
python scripts/generate_section4_metrics_and_plots.py
```

The script will:
- Discover and evaluate all 84 models (36 resample + 36 fixedmean + 12 baseline)
- Generate metrics for all combinations of:
  - Regimes: baseline, fixedmean, resample
  - Modes: x, xy, x2
  - Architectures: mlp, gru, lstm, baseline
  - Noise levels (σ): 0.05, 0.1, 0.5, 1.0

## Outputs

### CSV File
- `Report/figs/all_metrics_summary.csv` - Comprehensive metrics table with columns:
  - `regime`, `mode`, `arch`, `sigma`, `R` - Model metadata
  - `rmse_b_mean`, `rmse_b_std` - Background RMSE statistics
  - `rmse_a_mean`, `rmse_a_std`, `rmse_a_median`, `rmse_a_q25`, `rmse_a_q75` - Analysis RMSE statistics
  - `improvement_mean`, `improvement_std`, `improvement_median` - Improvement metrics
  - `hausdorff_mean`, `hausdorff_std`, `hausdorff_median` - Hausdorff distance
  - `divergence_rate` - Average divergence step

### PNG Figures
1. `core_rmse_by_mode.png` - RMSE (analysis) by observation mode
2. `core_improvement_by_mode.png` - Improvement metric by observation mode
3. `core_hausdorff_by_mode.png` - Hausdorff distance by observation mode
4. `mode_x_summary.png` - Comprehensive summary for mode 'x'
5. `mode_xy_summary.png` - Comprehensive summary for mode 'xy'
6. `mode_x2_summary.png` - Comprehensive summary for mode 'x2'

## Metrics Computed

### RMSE (Root Mean Square Error)
- **Background RMSE** (`rmse_b`): Error using climatological mean
- **Analysis RMSE** (`rmse_a`): Error after data assimilation

### Improvement Metric
```
improvement = (rmse_b - rmse_a) / (rmse_b + 1e-8)
```
Positive values indicate assimilation improves over background.

### Hausdorff Distance
Symmetric Hausdorff distance between truth and analysis trajectories, normalized by typical Lorenz attractor range (50).

### Divergence Rate
The time step at which RMSE exceeds a threshold (10.0), indicating trajectory divergence.

## Model Filename Conventions

- **Resample/Fixedmean**: `{mode}_{arch}_n{sigma}_R{R}.pth`
  - Example: `x_mlp_n0.1_R0.1.pth`
- **Baseline**: `{mode}_n{sigma}.pth`
  - Example: `xy_n0.5.pth`

## Architecture Mapping

| Architecture Name | Model Class    |
|------------------|----------------|
| mlp              | MLPModel       |
| gru              | GRUModel       |
| lstm             | LSTMModel      |
| baseline         | BaselineMLP    |

## Observation Operators

Implemented exactly as in `observation_operators.py`:

- **x**: Observe first component only → `y = [x[0]]`
- **xy**: Observe first two components → `y = [x[0], x[1]]`
- **x2**: Nonlinear, observe squared first component → `y = [x[0]²]`

## Dependencies

Required Python packages:
- numpy
- pandas
- torch
- matplotlib
- scipy
- scikit-learn

Install with:
```bash
pip install numpy pandas torch matplotlib scipy scikit-learn
```

## Notes

- The script evaluates 50 test trajectories per model for efficiency
- Evaluates the first 50 time steps of each trajectory
- Uses CPU by default; will use CUDA if available
- Runtime: ~3-5 minutes on CPU
