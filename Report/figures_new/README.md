# figures_new Directory

This directory contains **corrected versions** of the original figures based on Hans's PDF comments and meeting feedback. The original figures in `Report/` are NOT modified.

## Phase 1 Complete: Corrected Original Figures

The following figures are corrected versions of the ORIGINAL figures from `main (3).tex`:

| Corrected Figure | Original Figure | Hans Comment Addressed |
|------------------|-----------------|------------------------|
| `4_3a_resample_rmse_distributions_logscale.png` | `4_3a_resample_rmse_distributions.png` | #105: "can you make it a log plot?" + model labels |
| `4_4a_post_assimilation_rmse_logscale.png` | `4_4a_post_assimilation_rmse.png` | #105: log scale + clear observation mode notation ($h(x)=x_1$, etc.) |
| `4_5a_trajectory_fidelity_corrected.png` | `4_5a_trajectory_fidelity_comparison.png` | $f_\theta$ notation (not $\Phi$), clear labels |
| `4_5b_error_evolution_profiles_corrected.png` | `4_5b_error_evolution_profiles.png` | Regime comparison, noise level annotations |
| `4_6a_background_sampling_stability_corrected.png` | `4_6a_background_sampling_stability.png` | Clear stability labels, regime comparison |

## Quick Usage

To regenerate all corrected figures:
```bash
python3 Report/scripts_generated/generate_all_corrected_figures.py
```

Or run individual scripts:
```bash
python3 Report/scripts_generated/plot_4_3a_rmse_distributions_logscale.py
python3 Report/scripts_generated/plot_4_4a_obs_mode_rmse_logscale.py
python3 Report/scripts_generated/plot_4_5a_trajectory_fidelity.py
python3 Report/scripts_generated/plot_4_5b_error_evolution.py
python3 Report/scripts_generated/plot_4_6a_stability.py
```

## Data Sources

Figures are generated from `.npy` and `.csv` files in:
- `results/baseline/metrics/` - Baseline regime results
- `results/resample /run_20251008_134240 /` - Resample regime results  
- `results/fixedmean /run_20251008_133752 /` - FixedMean regime results

## Hans's Figure-Related Comments Addressed

1. **Comment #105** (Page 17): "can you make it a log plot?"
   - Fixed in: `4_3a_*_logscale.png`, `4_4a_*_logscale.png`

2. **"unclear observation mode (x / xy / x²)"**
   - Fixed: Added proper math notation $h(x) = x_1$, $h(x) = (x_1, x_2)$, $h(x) = x_1^2$

3. **"whats phi?" / "φ vs fθ notation"**
   - Fixed: All figures now use $f_\theta$ for learned analysis operator

4. **"missing model labels"**
   - Fixed: Added MLP, GRU, LSTM labels where applicable

5. **"missing noise level"**
   - Fixed: Added $\sigma$ values to all relevant figures

## Note

The existing figures in `Report/` are retained unchanged. To use corrected figures in the report, update `\includegraphics` paths to point to `figures_new/` versions.

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
