# Hans Comments Resolution Mapping

This document maps each of Hans's feedback comments to the specific paragraphs/sections that were rewritten to address them in the academic revision.

## Line-by-Line Mapping to Original main (3).tex

The following table maps Hans's PDF annotations (page 4 comments) to the exact lines in the original `main (3).tex`:

| Hans Comment | Original Line(s) | Original Text | Resolution in main_rewritten.tex |
|-------------|------------------|---------------|----------------------------------|
| "Swap, ENKF has gaussian approx, 3D Var has iterative optimization" | Line 74 | "depend on linear-Gaussian assumptions and costly iterative optimization" | Line 76: "3D-Var require iterative minimization...EnKF employ Gaussian approximations" |
| "whats phi?" | Line 74 | "learns $\Phi$" without definition | Lines 76, 111-130: Φ defined as "analysis functional—the mapping from observations and background to MAP estimate" + Notation table |
| "Make this crucial, the method does not use re-analysis" | Line 74 | "(no analysis labels)" - not emphasized | Line 76: "Crucially, no ground truth or re-analysis data are used during training" |
| "Do you refer to AI-Var from the paper?" | Line 74, 93 | "AI-Based Data Assimilation (AI-DA)" | Lines 76, 105-106: "the AI-Var framework introduced by Fablet et al." with citation |
| "new paragraph" | Line 77 | Paragraph structure | Restructured with proper paragraph breaks |
| "You dont quite replicate the AI-Var paper" | Line 77 | "replicates and evaluates" | Line 78: Changed to "pilot study investigates adaptations" |
| "investigating state estimation crucial for forecasting" | Line 85 | "State estimation...is a fundamental challenge" | Line 92-93: Added "crucial for accurate forecasting" |
| "in the type of problem investigated here" | Line 93 | Context unclear | Line 102-103: "in this problem setting" clarified |
| "the employed background mean" | Line 80 | "background priors" | Line 80: Changed to "employed background mean" |
| "Performance is evaluated via the test RMSE..." | Line 80 | RMSE evaluation sentence | Lines 80-81: Restructured evaluation description |

## High Priority Comments

| ID | Comment | Resolution Location | Description |
|----|---------|---------------------|-------------|
| 4 | "Make this crucial, the method does not use re-analysis" | Abstract, Section 1 | Explicitly stated: "Crucially, no ground truth or re-analysis data are used during training; the true state is employed only for offline evaluation." |
| 19 | "no ground truth or re-analysis data" | Abstract, Section 1, Section 2.3 | Made prominent throughout that training is self-supervised without analysis labels |
| 30 | "define them in the next sentence precisely" (B and R) | Section 1.1 (Notation), Section 2.1 | Added explicit definitions: B = background error covariance, R = observation error covariance in notation table and mathematical formulation |
| 71 | "why switch between phi and f?" | Section 1.1 (Notation table) | Added explicit explanation: "Φ is the analysis functional (theoretical optimal mapping); f_θ is its neural network approximation" |
| 93 | "make very clear how phi and f_theta relate" | Section 1.1 | Created dedicated notation subsection with table explaining the distinction |
| 94 | "much earlier" (B and R definitions) | Section 1.1 | Moved B and R definitions to notation section at start of paper |
| 105 | "can you make it a log plot?" | Section 3.5 (Metrics), figures_new/ | Created log-scale boxplot (`rmse_boxplot_logscale.png`) with logarithmic y-axis for better visualization of outlier distribution |
| - | Meeting discussion: robust metrics | Section 3.5 (Metrics), figures_new/ | Added RMdSE (Root Median Squared Error) as robust alternative to RMSE for handling outliers from catastrophic failures; generated comparison figure (`rmse_vs_rmdse_comparison.png`) |

## Medium Priority Comments

| ID | Comment | Resolution Location | Description |
|----|---------|---------------------|-------------|
| 2 | "Swap, EnKF has gaussian approx, 3D Var has iterative optimization" | Abstract, Section 1 | Fixed: "3D-Var require iterative minimization... while EnKF employ Gaussian approximations" |
| 3 | "whats phi?" | Abstract, Section 1.1 | Added definition: "the analysis functional Φ—the mapping from observations and background to the MAP state estimate" |
| 5 | "Do you refer to AI-Var from the paper?" | Abstract, Section 1 | Added explicit reference: "the AI-Var framework introduced by Fablet et al." at first mention |
| 7 | "You dont quite replicate the AI-Var paper" | Abstract, Title | Changed to "pilot study" and "investigates adaptations of the AI-Var scheme" |
| 13 | "These results suggest that the AI Var scheme can be improved..." | Abstract | Reworded conclusion to match suggested phrasing |
| 16 | "knowledge of the physical model" | Section 1 | Added: "Classical data assimilation methods rely on knowledge of the physical model" |
| 18 | "Machine learning based data assimilation such as AI-Var" | Section 1 | Restructured to use AI-Var terminology with citation |
| 25 | "the maximum a-posteriori (MAP) state estimate" | Abstract, Section 1.1, Section 2.2 | Added MAP definition to analysis functional description |
| 26 | "observation operators h(x)" | Section 2.5 (Table) | Changed to lowercase h(x) notation consistently |
| 37 | "bar{x}_B should be the static average over all training runs" | Section 3.2 | Clarified: "computed from this ensemble" for climatological statistics |
| 46 | "distinct partial and one non-linear observation operator" | Section 2.5, Abstract | Clarified: "two partial linear operators... and one nonlinear operator" |
| 58 | "stay consistent with notation" | Section 1.1 | Added dedicated Notation subsection |
| 60 | "(via ground truth simulation or re-analysis)" | Section 1 | Added: "available here due to the nature of the simulation study" |
| 62 | "while training with re-analysis data is a frequent option..." | Section 1 | Added exact phrasing: "While training with re-analysis data is a common option, this study instead investigates self-supervised training" |
| 67 | "shouldnt it predict the analysis x^*?" | Section 3.3 | Clarified: network outputs x̂^a which approximates optimal analysis |
| 75 | "90% improvement refers to RMSE_a is 1/10 of RMSE_b" | Section 3.5 | Added: "note that 90% improvement corresponds to RMSE_a being one-tenth of RMSE_b" |
| 92 | "doesnt every observation need noise?" | Section 3.1 | Clarified: "All observations are corrupted by additive Gaussian noise" |
| 97 | "x^2 provides the least amount info" | Section 2.5 | Added: "The operators are ordered by information content: xy provides the most... x^2 provides the least" |

## Meeting Discussion Items (Added)

| Item | Resolution Location | Description |
|------|---------------------|-------------|
| Log-scale boxplots | Section 3.5, figures_new/generate_figures.py | Added `generate_rmse_boxplot_logscale()` function to create boxplots with logarithmic y-axis for better visualization of outliers. Output: `rmse_boxplot_logscale.png` |
| RMdSE (robust metric) | Section 3.5 (Metrics) | Added Root Median Squared Error definition and discussion as robust alternative to RMSE when distributions have heavy tails due to catastrophic failures. Output: `rmse_vs_rmdse_comparison.png` |
| Paragraph formatting | Throughout document | Added `\noindent` formatting for proper paragraph alignment |
| Attractor visualization | figures_new/ | Regenerated `attractor_projection_new.png` with full trajectory data (100,000 points) showing proper Lorenz butterfly pattern |

## Low Priority Comments (Editorial)

| ID | Comment | Resolution | 
|----|---------|------------|
| 1, 6 | Paragraph structure | Reorganized into flowing academic prose |
| 8 | "investigating state estimation crucial for forecasting" | Added: "crucial for accurate forecasting" |
| 10 | "the employed background mean" | Changed throughout |
| 14 | "frequent" | Word choice updated |
| 15 | "in different disciplines in science" | Changed to "many disciplines in science and engineering" |
| 17 | "reducing" vs "limiting" | Changed to "reducing" throughout |
| 40, 41, 70 | "AI-Var" consistency | Used AI-Var consistently |
| 50 | "provides first insights for future exploration" | Used in Contributions section |
| 51 | "many" vs "several" | Updated word choice |
| 74 | Capital vs lowercase letters | Added note in notation section |
| 89 | "fine here as you have now introduced what the mean" | Positive feedback - maintained |
| 90, 91 | "lots of doubling" | Reduced redundancy throughout |
| 102, 103, 104 | Table references and placement | Tables properly referenced in text |
| 115, 117 | "this is exactly how it should be", "this is very useful" | Positive feedback - maintained |

## Comments Requiring Human Decision (NEEDS HUMAN)

| ID | Comment | Status |
|----|---------|--------|
| 31 | "i dont understand this citation" | Citation context reviewed; specific citation unclear - please clarify |
| 38 | "how do the references come into play here?" | Reference usage reviewed; please specify which reference |
| 42, 43 | "reference?" | Please specify which claims need references |
| 76 | "unclear" (page 16) | Please specify which passage |
| 77, 83, 84 | Citation placement questions | Please clarify which references seem misplaced |
| 78, 79, 80 | Unclear content markers | Please specify what needs clarification |
| 81 | "reference?" (page 21) | Please specify which claim needs reference |
| 85 | "This needs closer specification" | Please specify what needs more detail |
| 88 | "I dont quite understand this" (page 23) | Complex passage rewritten for clarity |
| 99 | "why is that?" | Context provided where possible |
| 107 | "something is wrong here" | Please specify the error |
| 109 | "we need to discuss this" | Flagged for discussion |
| 111, 112, 113, 114 | Figure caption details | Mode/noise/model specifications added where applicable |
| 118 | "no control, maybe test smaller steps" | Noted for future work |

## Style Improvements Summary

1. **Removed all bullet lists** except for algorithmic summaries and table contents
2. **Converted to flowing academic prose** throughout all main sections
3. **Replaced ML-style overloaded words**: "robust" → "stable", "rigorous" → "systematic", etc.
4. **Added notation section** (Section 1.1) defining all symbols before use
5. **Introduced all key concepts** (Φ, f_θ, B, R, h) before first technical use
6. **Reduced repetition** while preserving substance
7. **Honest treatment of inconclusive results** - emphasized in Discussion section
8. **Pilot study framing** - not overselling contributions
9. **Clear transitions** between sections and paragraphs
10. **Consistent mathematical notation** throughout

## New Figures Generated

| Figure | Description | Addresses |
|--------|-------------|-----------|
| `attractor_projection_new.png` | Full Lorenz attractor with 100,000 data points showing butterfly pattern | Visualization quality |
| `trajectory_sample_new.png` | State reconstruction showing Truth vs Analysis vs Background | Data clarity |
| `rmse_comparison_new.png` | RMSE across noise levels | Standard metric |
| `rmse_boxplot_logscale.png` | Log-scale boxplot for outlier visualization | Hans comment #105 |
| `rmse_vs_rmdse_comparison.png` | RMSE vs RMdSE comparison showing how RMdSE (robust) is more stable than RMSE when outliers exist | Meeting discussion: handling outliers in performance metrics |

## Appendix Status

The appendix (Sections A.1-A.7) was **kept unchanged** as per instructions.
