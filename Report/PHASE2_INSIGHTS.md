# Phase 2: Insights Analysis Summary

## Key Findings from Robust Metrics Analysis

### 1. Regime Stability (Critical Finding)

| Regime | Success Rate | Failure Rate | Total Configs |
|--------|-------------|--------------|---------------|
| **Resample** | 100% (36/36) | 0% | 36 |
| **FixedMean** | 5.6% (2/36) | **94.4%** (34/36) | 36 |

**Implication**: FixedMean regime is fundamentally unstable. Any comparison using FixedMean data is misleading because 94% of trajectories diverge (RMSE > 100).

### 2. Observation Mode Ordering

**Expected** (by information content): xy < x < x²
- xy: Two components, most information
- x: One component, linear
- x²: Nonlinear, loses sign information

**Observed** (by RMSE, Resample regime): x < xy < x²
- x: Mean RMSE = 7.01
- xy: Mean RMSE = 7.89
- x²: Mean RMSE = 11.73

**Interpretation**: The ordering reflects *learning difficulty*, not information content:
- x mode: Simple linear observation, easiest to learn
- xy mode: More information but also more noise dimensions
- x² mode: Nonlinear mapping, sign ambiguity makes learning hardest

This is **expected behavior** for the ML model—the network finds linear observations easier to map.

### 3. RMSE vs RMdSE (When to Use Each)

| Regime | RMSE | RMdSE | Difference | Recommendation |
|--------|------|-------|------------|----------------|
| Resample | 8.68 | 8.46 | 2.5% | Either metric OK |
| FixedMean | ~800,000 | ~800,000 | <1% | Neither useful (94% failures) |

**Conclusion**: 
- For Resample (no outliers): RMSE and RMdSE are nearly identical
- For FixedMean: Both metrics dominated by catastrophic failures
- **Recommendation**: Use RMSE for Resample regime (standard, comparable)

### 4. Architecture Comparison (Resample Regime)

| Mode | MLP | GRU | LSTM | Best |
|------|-----|-----|------|------|
| x | 6.97 | 7.05 | 7.01 | MLP |
| xy | 7.95 | 7.79 | 7.93 | GRU |
| x² | 13.31 | 10.83 | 11.06 | GRU |

**Finding**: Results are **inconclusive** for architecture selection:
- Differences are small (within 10-20%)
- No clear winner across all modes
- GRU slightly better for nonlinear (x²) mode

### 5. Noise Sensitivity

RMSE increases modestly with noise level σ:
- σ = 0.05 to σ = 1.0: ~5-15% increase in RMSE
- Performance degradation is graceful (no catastrophic failures)
- Nonlinear mode (x²) most sensitive to noise

### 6. Assimilation Improvement

| Mode | RMSE Before | RMSE After | Improvement |
|------|-------------|------------|-------------|
| x | 7.01 | 7.01 | -0.07% |
| xy | 7.83 | 7.89 | -0.61% |
| x² | 11.65 | 11.73 | -0.72% |

**Critical Finding**: The learned network provides **no improvement** over the background!
- Improvement is essentially zero or slightly negative
- This is an **inconclusive result** as Hans mentioned is acceptable
- Suggests the AI-Var approach may need further refinement for this problem

## Figures Generated

1. **fig_resample_rmse_by_mode.png**: RMSE by observation mode with log scale
2. **fig_regime_stability_comparison.png**: Success/failure rates and RMSE comparison
3. **fig_noise_sensitivity.png**: RMSE vs noise level for each mode
4. **fig_boxplot_log_scale.png**: Box plots with log scale per Hans's request
5. **fig_rmse_vs_rmdse.png**: When to use each metric

## Recommendations for Report

1. **Focus on Resample regime** - FixedMean is unstable and comparisons are misleading
2. **Use RMSE as primary metric** - No outliers in Resample regime
3. **Acknowledge inconclusive results**:
   - Architecture differences are small
   - Improvement over background is negligible
   - This is okay for a pilot study
4. **Explain observation mode ordering** - It reflects learning difficulty, not information content
5. **Emphasize stability finding** - Resample regime prevents catastrophic failures

## Hans's Key Points Addressed

✓ "RMSE is susceptible to outliers" → Verified: For Resample, RMSE ≈ RMdSE (no outliers)
✓ "Use log plots" → All box plots use log scale
✓ "94% failure rate in FixedMean" → Confirmed and visualized
✓ "Results can be inconclusive" → Improvement is ~0%, acknowledged
✓ "Focus on one experiment" → Focused on Resample regime
✓ "Don't force conclusions" → Results honestly presented as inconclusive
