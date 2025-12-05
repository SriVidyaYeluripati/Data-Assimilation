# figures_new Directory

This directory contains **corrected versions** of the original figures based on Hans's PDF comments and meeting feedback. The original figures in `Report/` are NOT modified.

## Phase 1: Corrected Original Figures

The following figures are corrected versions of the ORIGINAL figures from `main (3).tex`:

| Corrected Figure | Original Figure | Hans Comment Addressed |
|------------------|-----------------|------------------------|
| `4_3a_resample_rmse_distributions_logscale.png` | `4_3a_resample_rmse_distributions.png` | #105: "can you make it a log plot?" + model labels |
| `4_4a_post_assimilation_rmse_logscale.png` | `4_4a_post_assimilation_rmse.png` | #105: log scale + clear observation mode notation |
| `4_5a_trajectory_fidelity_corrected.png` | `4_5a_trajectory_fidelity_comparison.png` | $f_\theta$ notation, clear labels |
| `4_5b_error_evolution_profiles_corrected.png` | `4_5b_error_evolution_profiles.png` | Regime comparison, noise level annotations |
| `4_6a_background_sampling_stability_corrected.png` | `4_6a_background_sampling_stability.png` | Clear stability labels |

## Phase 2: Insight Figures (Robust Metrics Analysis)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| `fig_resample_rmse_by_mode.png` | RMSE by observation mode (log scale) | Ordering x < xy < x² reflects learning difficulty |
| `fig_regime_stability_comparison.png` | Success/failure rates | FixedMean has 94% failure rate |
| `fig_noise_sensitivity.png` | RMSE vs noise level | Graceful degradation with noise |
| `fig_boxplot_log_scale.png` | Box plots per architecture | Log scale per Hans request |
| `fig_rmse_vs_rmdse.png` | RMSE vs RMdSE comparison | For Resample, both metrics similar |

## Quick Usage

```bash
# Phase 1: Corrected original figures
python3 Report/scripts_generated/generate_all_corrected_figures.py

# Phase 2: Insight figures
python3 Report/scripts_generated/phase2_insight_figures.py
```

## Key Findings (Phase 2)

1. **FixedMean is unstable**: 94% of configurations fail (RMSE > 100)
2. **Resample is stable**: 100% success rate
3. **Improvement is ~0%**: Network provides no improvement over background
4. **Observation mode ordering**: x < xy < x² (learning difficulty, not info content)
5. **RMSE ≈ RMdSE for Resample**: No outliers, so RMSE is appropriate

## Hans's Comments Addressed

- ✓ Log-scale boxplots (#105)
- ✓ RMSE vs RMdSE comparison
- ✓ Observation mode notation (h(x) = x₁, etc.)
- ✓ Model labels (MLP, GRU, LSTM)
- ✓ f_θ notation (not Φ)
- ✓ Noise level labels

## Note

The existing figures in `Report/` are retained unchanged. To use corrected figures in the report, update `\includegraphics` paths to point to `figures_new/` versions.
