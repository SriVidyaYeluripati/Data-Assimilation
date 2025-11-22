# Scripts Directory

This directory contains utility scripts for the Data Assimilation project.

## generate_comparison_figures.py

Generates 6 publication-quality comparison plots for Lorenz-63 data assimilation experiments.

### Usage

```bash
python scripts/generate_comparison_figures.py
```

### What it does

1. **Collects metrics** from all regime directories:
   - `results/baseline/metrics/*.csv`
   - `results/fixedmean/run_*/metrics/*.csv`
   - `results/resample/run_*/metrics/*.csv`
   - `comprehensive_diagnostics/**/metrics/*.{json,csv}` (if present)

2. **Normalizes and aggregates** metrics by:
   - Regime (baseline, fixedmean, resample)
   - Architecture (MLP, GRU, LSTM)
   - Observation mode (x, xy, x²)
   - Observation noise (σ_obs: 0.05, 0.10, 0.50, 1.00)

3. **Computes aggregated statistics**:
   - mean, std, median, IQR for: rmse_a, improvement_bg, hausdorff
   - divergence_rate (fraction of diverged runs)

4. **Saves aggregated table** to:
   - `Report/figs/aggregated_metrics_summary.csv`

5. **Generates 6 PNG figures**:

   **Global comparison figures (Section 4.2):**
   - `core_rmse_by_mode.png` - RMSE vs σ_obs for each mode (3 panels)
   - `core_improvement_by_mode.png` - Improvement vs σ_obs for each mode (3 panels)
   - `core_hausdorff_by_mode.png` - Hausdorff distance vs σ_obs (3 panels)

   **Mode-specific figures (Section 4.3):**
   - `mode_x_summary.png` - Summary for observation mode X (3 stacked panels)
   - `mode_xy_summary.png` - Summary for observation mode XY (3 stacked panels)
   - `mode_x2_summary.png` - Summary for observation mode X² (3 stacked panels)

### Color & Style Conventions

**Architectures (consistent across all 6 plots):**
- MLP: blue
- GRU: green
- LSTM: red

**Regime linestyles:**
- baseline: dotted (`:`)
- fixedmean: dashed (`--`)
- resample: solid (`-`)

### Formulas

**Improvement_bg:**
```
improvement_bg = (rmse_b - rmse_a) / (rmse_b + 1e-8)
```

**Divergence Rule:**
- Flagged if rmse_a > 1e6 or improvement_bg < -1.0

### Output

All figures are saved to `Report/figs/` with:
- High resolution (300 DPI)
- Consistent styling (white background, clear grids)
- Mean ± standard deviation ribbons
- Clear legends positioned outside panels (right side)
- Figure sizes: ~18×6 for horizontal multi-panel, 9×12 for vertical

**Note:** Hausdorff distance is currently a placeholder (data not yet available in source files). The panels display placeholder text when no data is present.

### Requirements

- Python 3.8+
- matplotlib
- pandas
- numpy

Install with:
```bash
pip install matplotlib pandas numpy
```
