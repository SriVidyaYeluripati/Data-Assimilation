# Diff List: Paragraph Changes Due to Hans's Comments

This document lists every paragraph changed in response to Hans's comments from the PDF and meeting feedback.

## Section 1: Introduction

### Original Issue (IDs 2, 3, 4, 5, 16, 18, 19, 20, 21, 22)
- **Before**: "AI-Based Data Assimilation (AI-DA) framework replaces..."
- **After**: "Machine learning-based data assimilation, such as the AI-Var approach introduced by Bocquet et al., offers an alternative paradigm..."
- **Change**: Explicit AI-Var reference, removed AI-DA terminology

### Original Issue (ID 2)
- **Before**: "Classical methods...depend on linear-Gaussian assumptions and costly iterative optimization"
- **After**: "The three-dimensional variational method (3D-Var) seeks the analysis state by minimizing a cost function... The ensemble Kalman filter (EnKF), by contrast, propagates an ensemble of states and employs Gaussian approximations"
- **Change**: Clarified 3D-Var uses iterative optimization, EnKF uses Gaussian approximations (swapped)

### Original Issue (ID 3)
- **Before**: "learns $\Phi$" without definition
- **After**: "$\Phi$ represents the mapping from the available inputs—observations $\mathbf{y}$ and background state $\mathbf{x}^b$—to the analysis state"
- **Change**: Explicit definition of Φ as analysis functional before first use

### Original Issue (ID 4)
- **Before**: "no analysis labels"
- **After**: "Crucially, no ground truth or re-analysis data are required for training; the network is trained in a self-supervised manner"
- **Change**: Made explicit that no re-analysis is used for training (HIGH priority)

## Section 2: Mathematical Formulation

### Original Issue (ID 30)
- **Before**: B and R mentioned without definition
- **After**: Added Table 1 with explicit definitions: "B ∈ R^{3×3} is the background error covariance matrix, R ∈ R^{m×m} is the observation error covariance matrix"
- **Change**: Defined B and R immediately after first use (HIGH priority)

### Original Issue (IDs 26, 55)
- **Before**: "H(x)=[x], H(x)=[x,y], H(x)=[x^2]"
- **After**: "h(x) = x_1, h(x) = (x_1, x_2), h(x) = x_1^2"
- **Change**: Lowercase h(x) notation consistent with AI-Var paper

### Original Issue (ID 71, 93)
- **Before**: Unclear relationship between Φ and f_θ
- **After**: "The analysis functional Φ represents the mapping... In the AI-Var framework, this functional is approximated by a neural network f_θ with learnable parameters θ"
- **Change**: Explicit statement that Φ is abstract functional, f_θ is neural network representation (HIGH priority)

### Original Issue (ID 97)
- **Before**: No information ranking for observation modes
- **After**: "These operators are ordered by decreasing information content: (x_1, x_2) provides the most information, followed by x_1, while x_1^2 provides the least"
- **Change**: Added information ranking as Hans specified

## Section 3: Method

### Original Issue (ID 37)
- **Before**: Background mean not defined
- **After**: "The background state is set to the static climatological mean x̄_B, computed as the average state over all training trajectories"
- **Change**: Defined bar{x}_B explicitly

### Original Issue (ID 62)
- **Before**: Training approach unclear
- **After**: "Training is conducted in a self-supervised manner... The true state x^t is not used during training; it is available here only due to the nature of the simulation study"
- **Change**: Explicit statement about self-supervised training (HIGH priority)

### Original Issue (ID 75)
- **Before**: Improvement metric unclear
- **After**: "A positive value indicates that the analysis is closer to the truth than the background; a negative value indicates degradation"
- **Change**: Clarified improvement percentage interpretation

### Original Issue (ID 92)
- **Before**: Noise addition unclear
- **After**: "observations are created by applying the observation operator to the true state and adding Gaussian noise with standard deviation σ"
- **Change**: Clarified that all observations include noise

## Section 4: Results

### Original Issue (ID 105)
- **Before**: Linear scale figures
- **After**: Figure 3 uses logarithmic scale
- **Change**: Log-scale plot for divergence analysis as Hans requested

### Original Issue (ID 107)
- **Before**: "Something is wrong here" on Page 29
- **After**: Explicit acknowledgment of FixedMean divergence with RMSE > 10^5
- **Change**: Clarified that divergence is real experimental result, not error

### Original Issue (ID 111)
- **Before**: Multiple scattered plots
- **After**: Consolidated into 4 main figures
- **Change**: Plot consolidation as Hans suggested

### Original Issue (IDs 112, 113, 114)
- **Before**: Missing mode/noise/model specification in captions
- **After**: All figure captions include mode, noise level, and regime specification
- **Change**: Complete figure captions

## Section 5: Discussion

### Original Issue (Meeting feedback)
- **Before**: Strong conclusions ("confirms", "demonstrates")
- **After**: Softer language ("suggests", "indicates", "may contribute")
- **Change**: Removed strong language throughout

### Original Issue (ID 13)
- **Before**: "These results confirm that functional learning can successfully"
- **After**: "These results suggest that the AI-Var scheme demonstrates the feasibility of self-supervised training but does not achieve consistent improvement"
- **Change**: Acknowledged inconclusive results

### Original Issue (Meeting feedback - doubling)
- **Before**: Repeated explanations across sections
- **After**: Each concept introduced once, referenced thereafter
- **Change**: Removed all doubling/repetition (IDs 90, 91)

## Section 6: Conclusion

### Original Issue (ID 50)
- **Before**: Generic outlook
- **After**: "Three concrete perspectives: (1) Architecture design, (2) Loss function modification, (3) Larger-scale systems"
- **Change**: Specific future directions as Hans requested

## Language Changes Throughout

### Words Replaced (Meeting feedback)
| Before | After |
|--------|-------|
| robust | reliable |
| rigorous | careful |
| safe | stable |
| confirms | suggests |
| demonstrates | indicates |
| strong | consistent |

## Figure Changes

| Figure | Hans Comment | Change Made |
|--------|--------------|-------------|
| fig1_rmse_comparison.png | ID 111 | Consolidated multiple plots |
| fig2_architecture_comparison.png | ID 114 | Model specification added |
| fig3_fixedmean_divergence_log.png | ID 105, 107 | Log scale, divergence shown |
| fig4_improvement_analysis.png | Meeting | RMSE as single metric |
