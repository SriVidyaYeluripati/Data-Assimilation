# Scripts Directory

This directory contains utility scripts for the Data Assimilation project.

## generate_comparison_figures.py

Generates publication-quality comparison plots for Lorenz-63 data assimilation experiments.

### Usage

```bash
python scripts/generate_comparison_figures.py
```

### What it does

1. **Collects metrics** from all regime directories:
   - `results/baseline/metrics/*.csv`
   - `results/fixedmean/run_*/metrics/*.csv`
   - `results/resample/run_*/metrics/*.csv`

2. **Normalizes and aggregates** metrics by:
   - Regime (baseline, fixedmean, resample)
   - Architecture (MLP, GRU, LSTM)
   - Observation mode (x, xy, x²)
   - Observation noise (σ_obs: 0.05, 0.10, 0.50, 1.00)

3. **Saves aggregated table** to:
   - `Report/figs/aggregated_metrics_summary.csv`

4. **Generates 5 PNG figures**:

   **Global comparison figures (Section 4.2):**
   - `core_rmse_by_mode.png` - RMSE vs σ_obs for each mode (3 panels)
   - `core_improvement_by_mode.png` - Improvement vs σ_obs for each mode (3 panels)

   **Mode-specific figures (Section 4.3):**
   - `mode_x_summary.png` - Summary for observation mode X (2 stacked panels)
   - `mode_xy_summary.png` - Summary for observation mode XY (2 stacked panels)
   - `mode_x2_summary.png` - Summary for observation mode X² (2 stacked panels)

### Output

All figures are saved to `Report/figs/` with:
- High resolution (300 DPI)
- Consistent styling (white background, clean grids)
- Mean ± standard deviation ribbons
- Clear legends and axis labels

**Note:** Hausdorff distance plots are not generated as this metric is not available in the source data files.

### Requirements

- Python 3.8+
- matplotlib
- pandas
- numpy

Install with:
```bash
pip install matplotlib pandas numpy
```
