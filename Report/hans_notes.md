# Hans's Revision Notes for AI-Var Report

This document contains structured revision notes based on supervisor feedback from Hans on the project report. Each section corresponds to a page range (4–30) and includes interpretation of comments, action items, and meta-notes for revision.

---

## Key Takeaways from Final Meeting with Hans

> **Summary of meeting transcript – Core guidance for revision**

### 1. Length ≠ Quality
- **You are NOT graded by length; you are graded by precision.**
- The report can be shorter; what matters is clarity and correctness.
- This is a programming-focused project—the main effort is in implementation, not mathematical derivation.

### 2. Language and Tone
- **Avoid strong mathematical words** unless there is strict mathematical justification:
  - ❌ "rigorous" → ✅ "careful" or "systematic"
  - ❌ "robust" → ✅ "reliable"
  - ❌ "safe" → ✅ "stable" or "controlled"
- ML-influenced language sometimes conflicts with mathematical precision. Soften claims.

### 3. It Doesn't Need to Be Revolutionary
- The report doesn't need to be the most innovative thing in the world.
- It's okay if results are inconclusive—what matters is that the setup is clean and ethical.
- Focus on **what you actually did** (implementation, experimentation) rather than overselling contributions.

### 4. Key Action Items from Meeting
1. **Compress the first ~20 pages** – too much repetition; boil it down.
2. **Define symbols before using them** (Φ, f_θ, B, R, observation modes).
3. **Use AI-Var paper and Lorenz-63 paper** as the main references.
4. **Focus evaluation on a single metric** (RMSE) for core comparison; other metrics go to discussion/appendix.
5. **Soften all conclusions** – "suggest" instead of "confirm", "indicate" instead of "prove".

---

## Page 4 – Abstract

**What this page should do**
- Provide a concise, self-contained summary of the study's purpose, method, and key findings.
- Introduce the AI-Var approach (with citation) and the Lorenz-63 testbed.
- State the main conclusion: stochastic resampling is useful for stable assimilation.

**Key information currently present**
- Data assimilation is central to state estimation in chaotic systems.
- Classical methods (3D-Var, EnKF) have limitations (linear-Gaussian assumptions, iterative optimization).
- AI-DA learns Φ by minimizing a differentiable 3D-Var objective (self-supervised).
- The study tests on Lorenz-63 with three architectures (MLP, GRU, LSTM) and three observation modes.
- Resampling regime prevents failures like Attractor Escape.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 1 | "for" | Minor editorial; context unclear from location. | wording/tone | Review sentence context; ensure clarity of phrasing. |
| 2 | "Swap, ENKF has gaussian approx, 3D Var has iterative optimization" | The characterization of 3D-Var and EnKF is swapped. 3D-Var uses iterative optimization; EnKF uses Gaussian approximations. | methodology | Correct: "3D-Var relies on iterative optimization of a cost function, while EnKF employs Gaussian approximations for the ensemble distribution." |
| 3 | "whats phi?" | Φ is used before it is defined. The reader does not know what the symbol means. | missing definition / notation | Define Φ at first use: "the analysis functional Φ (a mapping from observations and background to the MAP state estimate)". |
| 4 | "Make this crucial, the method does not use re-analysis" | The key innovation—no re-analysis labels during training—should be emphasized more strongly. | wording/tone | Strengthen to: "Crucially, no re-analysis or ground truth labels are used during training; the true state is used only for offline evaluation." |
| 5 | "Do you refer to AI-Var from the paper? try to stick to the names there and give the reference already here" | The term "AI-DA" is used, but Hans prefers consistency with the original paper's terminology "AI-Var" and wants a citation at first mention. | citation / reference issue | Replace "AI-DA" with "AI-Var" at first mention and add citation: "following the AI-Var approach introduced by Fablet et al. [cite]". |
| 6 | "new paragraph" | Paragraph structure could be improved for clarity. | structure | Start a new paragraph where indicated. |
| 7 | "You dont quite replicate the AI-Var paper simulation study" | The wording suggests exact replication, but this is an investigation/extension. | wording/tone | Change "replicates and evaluates" to "investigates and evaluates" or "extends and evaluates". |
| 8 | "investigating state estimation crucial for forecasting" | The motivation should emphasize that state estimation is crucial for forecasting. | wording/tone | Add: "State estimation in chaotic dynamical systems—crucial for accurate forecasting—is a fundamental challenge." |
| 9 | "in the type of problem investigated here" | Context clarification needed. | wording/tone | Replace "in this setting" with "in the type of problem investigated here". |
| 10 | "the employed background mean" | Use more precise terminology. | wording/tone | Replace "background priors" with "the employed background mean". |
| 11 | "Performance is evaluated via the test RMSE..." | Sentence structure improvement. | structure | Restructure to: "Performance is evaluated via test RMSE computed against the (synthetic) true state." |
| 12 | "the resampling is indicated to be useful to prevent failures such as..." | Clarify what resampling prevents. | wording/tone | Clarify: "resampling is indicated to be useful to prevent failures such as Attractor Escape." |
| 13 | "These results suggest that the AI Var scheme can be improved with network architectures catering to the nature of the DA problem; it can successfully..." | Soften conclusion; avoid overstating. | wording/tone | Change "These results confirm" to "These results suggest that the AI-Var scheme can be improved with network architectures catering to the nature of the DA problem; it can successfully..." |

**Meta-notes for revision**
- **Repetition:** The abstract repeats content that will appear in Introduction; keep abstract minimal.
- **Undefined symbols:** Φ is used before definition—define it explicitly at first use.
- **Strong wording:** "confirm" is too strong; use "suggest" or "indicate".
- **Terminology:** Use "AI-Var" consistently throughout instead of "AI-DA".

---

## Page 5 – Introduction / Motivation

**What this page should do**
- Motivate the problem: state estimation in chaotic systems is challenging.
- Introduce the Lorenz-63 system as the testbed.
- Explain why classical DA methods have limitations and why ML-based approaches are promising.

**Key information currently present**
- Chaotic systems (e.g., Lorenz-63) exhibit sensitive dependence on initial conditions.
- Data Assimilation (DA) combines observations with background forecasts to produce analysis states.
- Traditional methods rely on linearity and Gaussian assumptions.
- AI-DA (AI-Var) replaces iterative optimization with learned mappings.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 14 | "frequent" | Word choice: "common" should be "frequent". | wording/tone | Change "common" to "frequent". |
| 15 | "in different disciplines in science..." | Add breadth. | wording/tone | Change "in science and engineering" to "in many disciplines in science and engineering". |
| 16 | "knowledge of the physical model" | Traditional methods also require knowledge of the physical model. | missing definition / notation | Add: "rely on knowledge of the physical model and generally assume linearity and Gaussian error distributions." |
| 17 | "reducing" | Word choice: "limiting" should be "reducing". | wording/tone | Change "limiting forecast uncertainty" to "reducing forecast uncertainty". |
| 18 | "Machine learning based data assimilation such as AI-Var as introduced in (cite) replaces..." | Restructure to explicitly mention AI-Var with citation at this point. | citation / reference issue | Restructure: "Machine learning-based data assimilation, such as AI-Var as introduced in [cite], replaces..." |
| 19 | "no ground truth or re-analysis data" | Emphasize that no ground truth is used during training. | wording/tone (HIGH) | Change "no analysis labels" to "no ground truth or re-analysis data are required for training." |
| 20 | "background information" | Terminology: use "background information" instead of "background state". | wording/tone | Change "background state" to "background information". |
| 21 | "via a simulation study on the Lorenz-63 system" | Clarify study type. | wording/tone | Add: "via a simulation study on the Lorenz-63 system." |
| 22 | "for the AI-Var approach in" | Add AI-Var reference. | citation / reference issue | Add: "for the AI-Var approach." |

**Meta-notes for revision**
- **Repetition:** Motivation overlaps with Abstract; condense.
- **Undefined symbols:** None yet on this page.
- **Strong wording:** Avoid overstatements; use moderate language.
- **Logical order:** Define what DA is before discussing its limitations.

---

## Page 6 – Project Goals / Scope

**What this page should do**
- Clearly state the project goals: learn the analysis functional Φ, benchmark architectures.
- Define the experimental scope: Lorenz-63, three observation modes, four noise levels.
- Present the hyperparameters table.

**Key information currently present**
- Central aim: train neural networks to learn Φ.
- Hyperparameters table with training/test split, architectures, etc.
- Goals include systematic benchmarking, noise robustness testing, generalization evaluation.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 23 | "an adaptation of the" | Clarify relationship to original work. | wording/tone | Change "pioneered by" to "an adaptation of the AI-Var framework introduced in [cite]." |
| 24 | "framework AI-Var introduced in (cite)" | Add explicit AI-Var citation. | citation / reference issue | Add AI-Var citation at mention. |
| 25 | "the maximum a-posteriori (MAP) state estimate" | Define the analysis functional more precisely. | missing definition / notation | Add: "the analysis functional Φ (the maximum a-posteriori (MAP) state estimate)." |
| 26 | "observation operators h(x) h(x)=x_1, h(x)= (x_1,x_2), h(x)=x_1^2" | Use lowercase h(x) notation and explicit component notation. | missing definition / notation | Change H(x) notation to lowercase h(x): "h(x)=x_1, h(x)=(x_1,x_2), h(x)=x_1^2." |
| 27 | "needs more detail otherwise its too specific and not specific enough at the same time just 3DVar objective function (see (cite)) is enough" | The 3D-Var cost function description is either too detailed or not detailed enough; simplify with citation. | structure | Simplify to: "the 3D-Var objective function (see [cite])." |
| 28 | "The key..." | Improve paragraph opening. | structure | Start paragraph with "The key..." |
| 29 | "comparison" | Word verified as correct. | wording/tone | No change needed. |
| 30 | "define them in the next sentence precisely" | B and R matrices are mentioned but not defined. | missing definition / notation (HIGH) | Add definitions: "the covariance matrices B (background error covariance) and R (observation error covariance)." |
| 31 | "i dont understand this citation" | Citation context is unclear. | citation / reference issue | NEEDS DISCUSSION: Review and clarify which citation Hans refers to. |

**Meta-notes for revision**
- **Repetition:** Goals section overlaps with Abstract; streamline.
- **Undefined symbols:** B, R, H, Φ need definitions before use.
- **Strong wording:** "rigorously evaluate" → soften to "evaluate" or "investigate".
- **Logical order:** Define observation operators (h(x)) before referring to modes.

---

## Page 7 – Scope (continued) / Regimes

**What this page should do**
- Define the three training regimes: Baseline, FixedMean, Resample.
- Explain the rationale for each regime.
- Clarify how background statistics are handled in each case.

**Key information currently present**
- Baseline: no background information provided.
- FixedMean: static background mean and covariance.
- Resample: dynamically resampled background statistics.
- RMSE and improvement metrics are used for evaluation.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 32 | "formatting" | Text formatting issue. | structure | Review and fix formatting. |
| 33 | "same as before" | Maintain consistency with previous formatting. | structure | Apply consistent formatting. |
| 34 | "to obtain a more fair..." | Emphasize fairness of comparison. | wording/tone | Rephrase to emphasize fairness: "to obtain a more fair..." |
| 35 | "three?" | Verify the count is correct. | methodology | Verified: three architectures (MLP, GRU, LSTM). |
| 36 | "capabilities to include prior information about the underlying system" | Clarify what prior information means. | wording/tone | Rephrase: "capabilities to include prior information about the underlying system." |
| 37 | "this needs more info - bar{x}_B should be the static average over all available training runs" | Definition of background mean is incomplete. | missing definition / notation | Add: "bar{x}_B (the static average over all available training runs)." |
| 38 | "again, how do the references come into play here?" | Citation usage is unclear. | citation / reference issue | NEEDS DISCUSSION: Clarify how references are relevant. |
| 39 | "sequential AI-Var with limited background information" | Clarify sequential context. | wording/tone | Rephrase: "sequential AI-Var with limited background information." |

**Meta-notes for revision**
- **Repetition:** Regime descriptions appear multiple times in the document; consolidate.
- **Undefined symbols:** bar{x}_B needs explicit definition.
- **Strong wording:** None flagged.
- **Logical order:** Define regimes once clearly, then refer back.

---

## Page 8 – Challenges and Limitations

**What this page should do**
- Document the challenges encountered during the project.
- Explain the limitations of the approach.
- Be honest about what works and what doesn't.

**Key information currently present**
- FixedMean instability and Attractor Escape.
- Overfitting in recurrent networks.
- Noise and observability limits.
- Lack of interpretability (black box).

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 40 | "reducing" | Word choice: "limiting" should be "reducing". | wording/tone | Change "limiting" to "reducing". |
| 41 | "AI-Var" | Use AI-Var consistently instead of AI-DA. | wording/tone | Replace "AI-DA" with "AI-Var". |
| 42 | "reference?" | A claim needs a reference. | citation / reference issue | NEEDS DISCUSSION: Identify which claim needs citation. |
| 43 | "ref" | Another reference needed. | citation / reference issue | NEEDS DISCUSSION: Identify which statement needs citation. |

**Meta-notes for revision**
- **Repetition:** Challenges/limitations overlap with discussion in other sections.
- **Undefined symbols:** None flagged.
- **Strong wording:** Avoid absolute claims; use hedging language.
- **Logical order:** Place limitations after presenting results, not in introduction.

---

## Page 9 – Contributions

**What this page should do**
- Summarize the project's main contributions.
- Be modest and accurate about what was achieved.
- Point to specific sections for details.

**Key information currently present**
- Viable learned analysis update.
- Systematic benchmarking.
- Validation of key insights (temporal context, stochastic resampling).
- Failure analysis and stability quantification.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 44 | "aims to contribute to..." | Soften the claim. | wording/tone | Change "This study investigated" to "This study aims to contribute to..." |
| 45 | "comparison" | Word verified. | wording/tone | No change needed. |
| 46 | "distinct partial and one non-linear observation operator" | Clarify observation operator types. | methodology | Change "three distinct observation scenarios" to "two distinct partial observation operators and one non-linear observation operator." |
| 47 | "clear indication" | Make indication clearer. | wording/tone | Rephrase for clarity. |
| 48 | "not RMSE?" | Verify that RMSE is the correct metric. | methodology | Confirmed: RMSE is correct. |
| 49 | "components" | Clarify which components. | wording/tone | Specify: "all three state components (X, Y, Z)." |
| 50 | "provides first insights for future exploration" | Soften outlook language. | wording/tone | Rephrase: "provides first insights for future exploration." |
| 51 | "many" | Word choice: "several" should be "many". | wording/tone | Change "several" to "many". |

**Meta-notes for revision**
- **Repetition:** Contributions section repeats findings from abstract.
- **Undefined symbols:** None flagged.
- **Strong wording:** "rigorously" and "robust" appear often; soften.
- **Logical order:** Contributions should come after results, not in introduction.

---

## Page 10 – Background / DA Theory

**What this page should do**
- Explain the fundamentals of Data Assimilation.
- Present the variational formulation (3D-Var).
- Define key quantities: background, analysis, observations, covariances.

**Key information currently present**
- DA combines prior knowledge with observations.
- Kalman Filter for linear Gaussian systems.
- 3D-Var cost function formulation.
- Definitions of B, R, H in the cost function.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 52 | "combination of both sources of information" | Emphasize that DA combines background and observations. | wording/tone | Emphasize: "combination of both sources of information." |
| 53 | "in variational DA schemes is used as the..." | Clarify variational context. | wording/tone | Clarify: "in variational DA schemes is used as the..." |
| 54 | "maybe stay with x or something that is more in line with common notation" | Use consistent notation (x instead of other symbols). | missing definition / notation | Use x notation consistently. |
| 55 | "h(a)" | Use lowercase h for observation operator. | missing definition / notation | Change H(a) to h(a). |
| 56 | "more precise and add the time component in the notation" | Notation should include time subscripts. | missing definition / notation | Add time subscripts: x_t, y_t, etc. |
| 57 | "give the source for this image and show where exactly phi is here" | Figure needs citation and Φ should be labeled. | citation / reference issue | Add: "Figure adapted from [cite]. The analysis functional Φ maps inputs to analysis state." |

**Meta-notes for revision**
- **Repetition:** Background theory overlaps with Abstract and Introduction.
- **Undefined symbols:** Ensure H, B, R, Φ are defined before use.
- **Strong wording:** None flagged.
- **Logical order:** Theory should come before methodology.

---

## Page 11 – AI-DA Paradigm / Self-supervised Training

**What this page should do**
- Explain the AI-Var paradigm.
- Emphasize self-supervised training (no analysis labels).
- Clarify the relationship between Φ and the neural network f_θ.

**Key information currently present**
- AI-DA reframes analysis update as functional approximation.
- Self-supervised training: backpropagate through J(x^a).
- Truth is only for offline evaluation.
- Advantages: inference efficiency, nonlinear approximation.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 58 | "stay consistent with notation" | Notation inconsistencies. | missing definition / notation | Add notation section in Methods. |
| 59 | "this is from AI-Var right?" | Confirm and cite AI-Var source. | citation / reference issue | Confirm with citation: "from the AI-Var framework [cite]." |
| 60 | "(via ground truth simulation or re-analysis)" | Clarify how truth is available in simulation study. | methodology | Add: "available here due to the nature of the simulation study." |
| 61 | "this needs more discussion as J(x) is already fairly linearized" | J(x) linearization needs more discussion. | methodology | NEEDS DISCUSSION: Add discussion of J(x) linearization implications. |
| 62 | "while training with re-analysis data is a frequent option, we will instead investigate self-supervised training" | Emphasize the self-supervised approach. | wording/tone (HIGH) | Add: "While training with re-analysis data is a frequent option, we instead investigate self-supervised training." |
| 63 | "available here due to the nature of the simulation study are only..." | Clarify simulation study context. | methodology | Clarify: "available here due to the nature of the simulation study." |
| 64 | "To provide available knowledge about the underlying system" | Explain how B and R provide system knowledge. | wording/tone | Add: "To provide available knowledge about the underlying system, we employ the covariance matrices B and R." |
| 65 | "sequential adaptation of the AI-Var" | Use AI-Var terminology. | wording/tone | Change "AI-DA framework" to "sequential adaptation of the AI-Var framework." |
| 66 | "stable chaotic" | Add descriptor. | wording/tone | Add: "exhibiting stable chaotic behavior." |

**Meta-notes for revision**
- **Repetition:** AI-Var explanation appears in multiple places.
- **Undefined symbols:** Φ vs f_θ relationship needs clarification.
- **Strong wording:** None flagged.
- **Logical order:** Define Φ and f_θ before using them.

---

## Page 12 – Lorenz-63 Testbed

**What this page should do**
- Present the Lorenz-63 system equations.
- Explain why it's a good testbed (chaotic, low-dimensional, well-understood).
- Define observation operators for the three modes.

**Key information currently present**
- Lorenz-63 ODEs with standard parameters.
- Chaotic dynamics, double-wing attractor.
- Three observation modes: x, xy, x².
- Gaussian noise added to observations.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 67 | "shouldnt it predict the analysis x^*?" | Clarify what the model outputs. | methodology | Clarify: "model predicts analysis state x^a (which approximates optimal x*)." |

**Meta-notes for revision**
- **Repetition:** Lorenz-63 description repeated from earlier.
- **Undefined symbols:** Ensure x^a vs x* notation is clear.
- **Strong wording:** None flagged.
- **Logical order:** Present observation operators before referring to "xy mode."

---

## Page 13 – Architecture Descriptions

**What this page should do**
- Describe the three neural architectures (MLP, GRU, LSTM).
- Explain the baseline configuration.
- Present the architecture comparison rationale.

**Key information currently present**
- Baseline MLP: no background information.
- MLP with background mean.
- GRU and LSTM for temporal modeling.
- Hidden dimensions, activations, etc.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 68 | "background mean" | Use "background mean" instead of "prior mean". | wording/tone | Change "prior mean" to "background mean." |
| 69 | "performance when utilizing AI-Var in dynamical systems" | Clarify performance context. | wording/tone | Add: "in terms of performance when utilizing AI-Var in dynamical systems." |

**Meta-notes for revision**
- **Repetition:** Architecture descriptions appear multiple times.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged.
- **Logical order:** Define architectures once, then reference.

---

## Page 14 – Training Regimes

**What this page should do**
- Explain the three training regimes in detail.
- Clarify the difference between Φ and f_θ.
- Present the training objective.

**Key information currently present**
- FixedMean: static background.
- Resample: dynamic background sampling.
- Training loss is the 3D-Var objective.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 70 | "AI-Var" | Use AI-Var instead of AI-DA. | wording/tone | Replace "AI-DA" with "AI-Var." |
| 71 | "why switch between phi and f?" | Φ and f_θ notation confusion. | missing definition / notation (HIGH) | Add notation clarification: "Φ is the abstract analysis functional; f_θ is the parametric neural network representation." |
| 72 | "AI-Var paper" | Add AI-Var citation. | citation / reference issue | Add AI-Var paper citation. |
| 73 | "stochastic initial condition with a chaotic systems and noisy observations" | Clarify sources of stochasticity. | wording/tone | Clarify: "stochastic initial conditions combined with chaotic dynamics and noisy observations." |

**Meta-notes for revision**
- **Repetition:** Regimes described multiple times.
- **Undefined symbols:** Φ vs f_θ needs clear distinction.
- **Strong wording:** None flagged.
- **Logical order:** Define notation before using it.

---

## Page 15 – Evaluation Metrics

**What this page should do**
- Define RMSE and improvement metrics.
- Explain how metrics are computed and interpreted.
- Present the evaluation protocol.

**Key information currently present**
- RMSE computed over all three state components.
- Improvement percentages relative to background or baseline.
- Averaging over test trajectories.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 74 | "capital letters usually refer to something different than small letters" | Notation convention: capital X,Y,Z vs lowercase x,y,z. | missing definition / notation | Add note: "Capital X,Y,Z refer to state components; lowercase x,y,z are used in equations." |
| 75 | "Also note, high is better, e.g. 90% improvement refers to RMSE_a is 1/10 of RMSE_b" | Clarify improvement metric interpretation. | methodology | Add: "Note: high improvement is better; e.g., 90% improvement means RMSE_a is 1/10 of RMSE_b." |

**Meta-notes for revision**
- **Repetition:** RMSE definition appears in multiple places.
- **Undefined symbols:** Ensure consistent capital/lowercase notation.
- **Strong wording:** None flagged.
- **Logical order:** Define metrics before presenting results.

---

## Page 16 – Methods / Common Evaluation Protocols

**What this page should do**
- Present the standardized evaluation protocol.
- Explain reproducibility measures.
- Describe seeding and averaging policies.

**Key information currently present**
- Standardized dataset partitions.
- Fixed random seeds for reproducibility.
- Evaluation on held-out test trajectories.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 76 | "unclear" | Some passage is unclear. | wording/tone | NEEDS DISCUSSION: Identify and clarify the unclear passage. |
| 77 | "why the ref here?" | Citation placement seems odd. | citation / reference issue | NEEDS DISCUSSION: Review citation placement. |

**Meta-notes for revision**
- **Repetition:** Protocol details may overlap with earlier sections.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged.
- **Logical order:** Protocol should precede results.

---

## Pages 17-18 – Methods (continued)

**What this page should do**
- Continue presenting methodology details.
- Explain temporal coherence and trajectory fidelity protocols.
- Define Hausdorff distance and other diagnostic metrics.

**Key information currently present**
- Temporal error evolution tracking.
- Attractor geometry preservation (Hausdorff distance).
- Component-wise analysis of corrections.

**Meta-notes for revision**
- **Repetition:** Metric definitions may be repeated.
- **Undefined symbols:** Hausdorff distance needs definition.
- **Strong wording:** "rigorous" and "robust" appear frequently.
- **Logical order:** Define metrics before using them.

---

## Page 19 – Data Generation

**What this page should do**
- Explain how Lorenz-63 trajectories are generated.
- Describe observation noise addition.
- Explain background/ensemble generation.

**Key information currently present**
- Runge-Kutta integration with dt=0.01.
- 1000 training / 500 test trajectories.
- Ensemble-derived background statistics.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 78 | "??" | Something is confusing or unclear. | wording/tone | NEEDS DISCUSSION: Identify what is unclear (likely missing explanation or inconsistent statement). |

**Meta-notes for revision**
- **Repetition:** Data generation details may repeat earlier descriptions.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged.
- **Logical order:** Data generation should come before experiments.

---

## Page 20 – Observation Operators / Noise Regimes

**What this page should do**
- Define the three observation operators precisely.
- Explain noise levels and their rationale.
- Present the information ranking: x² < x < xy.

**Key information currently present**
- x mode: observe x_1 only.
- xy mode: observe (x_1, x_2).
- x² mode: observe x_1² (nonlinear).
- Four noise levels: 0.05, 0.1, 0.5, 1.0.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 79 | "What does this mean?" | A statement is unclear. | wording/tone | NEEDS DISCUSSION: Identify and clarify the statement. |
| 80 | "This needs to be introduced" | A term or concept is used before introduction. | missing definition / notation | NEEDS DISCUSSION: Identify and introduce the term. |

**Meta-notes for revision**
- **Repetition:** Observation modes defined multiple times.
- **Undefined symbols:** Ensure observation operators are defined before modes are discussed.
- **Strong wording:** None flagged.
- **Logical order:** Define observation operators early; rank information content.

---

## Page 21 – Model Architectures

**What this page should do**
- Present architecture details in a table or structured format.
- Explain hidden dimensions, activations, layers.
- Clarify input/output dimensions.

**Key information currently present**
- Architecture table with hidden widths, activations.
- Baseline MLP: 32 hidden, Tanh.
- MLP/GRU/LSTM: 64 hidden, ReLU.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 81 | "reference?" | A claim needs a reference. | citation / reference issue | NEEDS DISCUSSION: Identify which claim. |
| 82 | "reliable" | Word verified as appropriate (softer than "robust"). | wording/tone | No change needed. |
| 83 | "why the reference here?" | Citation placement seems odd. | citation / reference issue | NEEDS DISCUSSION: Review citation. |
| 84 | "same question" | Same citation issue as above. | citation / reference issue | NEEDS DISCUSSION: Address together with #83. |

**Meta-notes for revision**
- **Repetition:** Architecture details appear multiple times.
- **Undefined symbols:** None flagged.
- **Strong wording:** "reliable" approved; avoid "robust".
- **Logical order:** Architectures should be presented once, clearly.

---

## Page 22 – Training Objective / Loss Function

**What this page should do**
- Present the self-supervised 3D-Var loss function.
- Explain that no truth labels are used during training.
- Describe optimization settings.

**Key information currently present**
- Loss function: 3D-Var cost with B, R, H.
- Adam optimizer, learning rate 1e-3.
- 30 epochs, batch size 256.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 85 | "This needs closer specification" | Some detail is too vague. | methodology | NEEDS DISCUSSION: Identify and specify the detail. |
| 86 | "reliability" | Word verified. | wording/tone | No change needed. |
| 87 | "reliable" | Word verified. | wording/tone | No change needed. |

**Meta-notes for revision**
- **Repetition:** Loss function defined multiple times.
- **Undefined symbols:** Ensure B, R, H are defined.
- **Strong wording:** "reliable" is acceptable.
- **Logical order:** Loss function should be presented once clearly.

---

## Page 23 – Background Generation / Ensemble Statistics

**What this page should do**
- Explain how ensemble statistics (mean, covariance) are generated.
- Describe the FixedMean vs Resample procedures.
- Present regularization of B.

**Key information currently present**
- Ensemble of size E=10000, m=32 members per batch.
- B regularized with ε=1e-6.
- Climatological mean for FixedMean.
- Resampled mean for Resample regime.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 88 | "I dont quite understand this" | A passage is confusing. | wording/tone | NEEDS DISCUSSION: Identify and simplify the passage. Consider that it may be too technical or missing context. |
| 89 | "fine here as you have now introduced what the mean" | Positive: the mean is now properly introduced. | wording/tone | No change needed. |
| 90 | "lots of doubling" | Content is repeated from earlier. | structure | Reduce redundancy with earlier sections. |
| 91 | "Doubling" | More repetition flagged. | structure | Consolidate repeated content. |
| 92 | "doesnt every observation need noise?" | Clarify that all observations have noise. | methodology | Clarify: "All observations include additive Gaussian noise." |

**Meta-notes for revision**
- **Repetition:** Ensemble generation details repeated—consolidate.
- **Undefined symbols:** bar{x}, B definitions should be earlier.
- **Strong wording:** None flagged.
- **Logical order:** Define ensemble procedures once.

---

## Page 24 – Notation Clarification (Φ vs f_θ)

**What this page should do**
- Clearly distinguish between Φ (abstract functional) and f_θ (neural network).
- Present the notation table if applicable.
- Define all key symbols.

**Key information currently present**
- Φ: analysis functional.
- f_θ: parametric neural network representation.
- Input/output specifications.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 93 | "make very clear how phi and f_theta relate" | The relationship between Φ and f_θ must be explicit. | missing definition / notation (HIGH) | Add explicit paragraph: "Φ is the abstract analysis functional (MAP estimate); f_θ is its parametric neural network representation with learnable parameters θ." |
| 94 | "much earlier" | B and R definitions should appear earlier. | structure | Move B and R definitions to Abstract or Introduction. |
| 95 | "also partial" | Clarify that x and xy are partial observations. | wording/tone | Add: "partial observation operators (x observes only one component; xy observes two)." |

**Meta-notes for revision**
- **Repetition:** Notation clarification repeats earlier definitions.
- **Undefined symbols:** Φ, f_θ, B, R should be defined at first use.
- **Strong wording:** None flagged.
- **Logical order:** Notation section should appear early in Methods.

---

## Page 25 – Input Encoding / Network Inputs

**What this page should do**
- Explain what inputs the networks receive.
- Clarify standardization/normalization.
- Present the sequence window (L=5).

**Key information currently present**
- Inputs: observation sequence y_{t-L+1:t}, background mean bar{x}.
- Feature-wise standardization.
- L=5 time steps.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 96 | "to create the artificial observations from a trajectory" | Clarify how observations are created. | methodology | Add: "to create the artificial observations from the ground-truth trajectory." |
| 97 | "x^2 provides the least amount info, then x and xy has the most" | Add information ranking for observation modes. | methodology | Add: "x² provides the least information, then x, and xy provides the most." |
| 98 | "why L here?" | Clarify L notation. | missing definition / notation | Confirm: "L is the sequence window length (typically 5), defined in Table 1." |
| 99 | "why is that?" | A statement needs explanation. | methodology | NEEDS DISCUSSION: Identify and explain the statement. |
| 100 | "computed from a resampled ensemble, right?" | Confirm ensemble computation. | methodology | Clarify: "computed from a resampled ensemble." |
| 101 | "just x_t as the other is already the output" | Simplify input notation. | missing definition / notation | Simplify to x_t in input description. |

**Meta-notes for revision**
- **Repetition:** Input descriptions may be repeated.
- **Undefined symbols:** L should be defined in hyperparameters table.
- **Strong wording:** None flagged.
- **Logical order:** Input encoding before architecture details.

---

## Page 26 – Experimental Setup Table

**What this page should do**
- Present a clear summary table of experimental configuration.
- Reference the table in text.
- Provide dataset statistics.

**Key information currently present**
- Table with trajectories, time steps, noise levels, etc.
- 1000/500 train/test split.
- All observation modes and noise levels listed.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 102 | "the table works much better here" | Positive: table placement is good. | structure | No change needed. |
| 103 | "reference it in the text" | Table should be referenced in text. | structure | Add: "See Table X for experimental configuration." |

**Meta-notes for revision**
- **Repetition:** Setup table may repeat hyperparameters table.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged.
- **Logical order:** Table should be referenced where relevant.

---

## Pages 27-28 – Convergence / Training Dynamics

**What this page should do**
- Present convergence behavior of different regimes.
- Show loss curves or convergence envelopes.
- Discuss attractor escape and divergence.

**Key information currently present**
- Resample converges faster and more stably.
- FixedMean shows instability at high noise.
- Baseline converges slowly.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 104 | "This is very nice and to the point" | Positive feedback. | wording/tone | No change needed. |

**Meta-notes for revision**
- **Repetition:** Convergence discussion may repeat earlier claims.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged.
- **Logical order:** Convergence should precede detailed results.

---

## Page 29 – Results: RMSE Analysis

**What this page should do**
- Present RMSE results across architectures, modes, noise levels.
- Show figures with clear captions.
- Discuss key findings.

**Key information currently present**
- RMSE distributions for MLP, GRU, LSTM.
- Performance across noise levels.
- Comparison of regimes.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 105 | "can you make it a log plot?" | Figure should use log scale. | methodology | NEEDS IMPLEMENTATION: Regenerate figure with log scale if plotting scripts available. |
| 106 | "its not in there, right?" | Content verification. | methodology | Verified. |
| 107 | "something is wrong here" | An error in the text or figure. | methodology | NEEDS DISCUSSION: Identify what appears wrong. |
| 108 | "this is nice" | Positive feedback. | wording/tone | No change needed. |
| 109 | "we need to discuss this" | Discussion item flagged. | methodology | NEEDS DISCUSSION: Flag for meeting. |

**Meta-notes for revision**
- **Repetition:** RMSE results may repeat earlier summary.
- **Undefined symbols:** None flagged.
- **Strong wording:** Avoid strong conclusions; results are indicative.
- **Logical order:** Present results systematically.

---

## Page 30 – Observation Mode Sensitivity

**What this page should do**
- Analyze how different observation modes affect performance.
- Present figures showing mode-specific behavior.
- Discuss the xy < x < x² ranking.

**Key information currently present**
- xy mode achieves best performance.
- x² mode shows highest variability.
- Information content hierarchy.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 110 | "comparison" | Word verified. | wording/tone | No change needed. |
| 111 | "Maybe keep one plot here" | Consider consolidating plots. | structure | NEEDS DISCUSSION: Consider reducing number of plots. |

**Meta-notes for revision**
- **Repetition:** Mode analysis may repeat earlier claims.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged.
- **Logical order:** Present mode analysis after general RMSE results.

---

## Page 31 – Temporal Assimilation / Attractor Geometry

**What this page should do**
- Present temporal error evolution analysis.
- Show how Resample vs FixedMean regimes differ in stability over time.
- Begin discussion of attractor geometry preservation.

**Key information currently present**
- Temporal error evolution figure for different regimes.
- Resample achieves fast convergence (~50 time steps).
- FixedMean shows oscillations and instability at high noise.
- Generalization to unseen trajectories discussed.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 112 | "what mode" | Figure caption is missing the observation mode specification. | missing definition / notation | NEEDS IMPLEMENTATION: Add observation mode (e.g., "xy mode" or "x mode") to figure caption. |

**Meta-notes for revision**
- **Repetition:** Some temporal analysis concepts may repeat from earlier sections.
- **Undefined symbols:** None flagged.
- **Strong wording:** Avoid "robust" unless justified.
- **Figure captions:** Ensure all figures specify which observation mode is shown.

---

## Page 32 – Attractor Geometry Preservation (continued)

**What this page should do**
- Present Hausdorff distance analysis.
- Show lobe occupancy discrepancy results.
- Quantify geometric fidelity of different regimes.

**Key information currently present**
- Global-normalized Hausdorff distance metric defined.
- Resample achieves low geometric deviation (~0.32).
- FixedMean shows higher deviation (~1.50).
- Lobe occupancy analysis introduced.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 113 | "what noise level" | Figure caption is missing the noise level specification. | missing definition / notation | NEEDS IMPLEMENTATION: Add noise level (e.g., "σ = 0.5" or "all noise levels") to figure caption. |
| 114 | "which model is in the figure?" | Figure caption should specify which model/architecture is shown. | missing definition / notation | NEEDS IMPLEMENTATION: Add model specification (e.g., "GRU", "all architectures") to figure caption. |

**Meta-notes for revision**
- **Repetition:** Hausdorff distance concept should be defined once.
- **Undefined symbols:** Ensure Hausdorff distance is defined before use.
- **Strong wording:** None flagged.
- **Figure captions:** All figures must clearly specify: observation mode, noise level, and architecture.

---

## Pages 33-34 – Component-wise Corrections / Summary of Section 4.5

**What this page should do**
- Present component-wise correction patterns (X, Y, Z).
- Summarize key findings about temporal assimilation and attractor geometry.
- Transition to ablation studies.

**Key information currently present**
- Correction c(t) = x^a(t) - x^b(t) defined.
- Z component shows smooth, low-frequency corrections.
- X, Y components show higher-variance adjustments.
- GRU/LSTM reduce correction variance compared to MLP.
- Summary: Resample stabilizes geometric deviation; FixedMean amplifies distortion.

**Hans's comments on this page**
- No specific comments on pages 33-34.

**Meta-notes for revision**
- **Repetition:** Summary may repeat earlier conclusions.
- **Undefined symbols:** None flagged.
- **Strong wording:** "robust" appears; consider softening.
- **Logical order:** Summary should consolidate, not repeat.

---

## Page 35 – Ablation Studies / Practical Recommendations

**What this page should do**
- Present ablation study on background sampling strategy.
- Show impact of noise on stability.
- Begin practical recommendations section.

**Key information currently present**
- Background sampling strategy comparison (Resample vs FixedMean).
- Resample shows graceful degradation; FixedMean becomes unstable at σ > 0.5.
- Failure rates: 70-80% for FixedMean vs 20-25% for Resample at high noise.
- Recommendation: Avoid FixedMean at σ > 0.5.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 115 | "this is exactly how it should be" | Positive feedback—Hans approves of this presentation. | wording/tone (positive) | No change needed; this is the target style. |

**Meta-notes for revision**
- **Repetition:** Ablation findings may echo earlier results; focus on new insights.
- **Undefined symbols:** None flagged.
- **Strong wording:** "robust" appears; use "reliable" or "stable" instead.
- **Logical order:** Ablations should follow main results.

---

## Pages 36-47 – Ablation Studies (continued), Conclusion, References, Appendix

**What these pages should do**
- Complete ablation studies (temporal context, covariance sensitivity, observation sparsity).
- Present practical recommendations table.
- Conclude with main findings and future outlook.
- Provide references and appendix materials.

**Key information currently present**
- Sequence length ablation: optimal L = 10-15 time steps.
- Background covariance sensitivity: optimal at λ = 1.0 (true B).
- Observation sparsity: performance degrades beyond 50% missing observations.
- Practical recommendations table summarizing key parameters.
- Conclusion: GRU/LSTM outperform MLP; Resample regime is essential.
- Future work: hybrid frameworks, generative extensions, higher-dimensional systems.
- References: Lorenz 1963, Kalman 1960, Fablet et al. (AI-Var), Bocquet et al.
- Appendix: Data pipelines, training dynamics, attractor geometry diagnostics.

**Hans's comments on these pages**
- No specific comments on pages 36-47.

**Meta-notes for revision**
- **Repetition:** Conclusion may repeat findings; keep concise.
- **Undefined symbols:** All should be defined by now.
- **Strong wording:** Conclusion uses "rigorous"—soften to "systematic" or "careful".
- **Logical order:** Conclusion should summarize without introducing new material.

---

## Page 48 – Appendix A.6 (Ablation Figures)

**What this page should do**
- Present supplementary ablation study figures.
- Show sensitivity analyses referenced in main text.
- Provide additional diagnostic plots.

**Key information currently present**
- Sequence length ablation figure.
- B-scaling sensitivity figure.
- Regime-specific robustness figure.

**Hans's comments on this page**

| ID | Original comment | What Hans means | Type | Action / Fix |
|----|------------------|-----------------|------|--------------|
| 116 | "no control over this in practice" | Hans notes that the parameter being varied (likely B scaling) cannot be controlled in real-world settings. This is a practical limitation. | methodology | Acknowledge this limitation: "In practice, the true background covariance B is unknown; this sensitivity analysis provides insight into robustness requirements." |
| 117 | "this is very useful" | Positive feedback—Hans finds this analysis valuable. | wording/tone (positive) | No change needed; maintain this type of analysis. |
| 118 | "no control, maybe test smaller steps" | Hans suggests testing finer step sizes for more detailed sensitivity analysis. | methodology | FUTURE WORK: Consider testing smaller step sizes (e.g., Δλ = 0.05 instead of 0.1) for B-scaling sensitivity. Add note to future work section. |

**Meta-notes for revision**
- **Repetition:** Appendix should supplement, not repeat main text.
- **Undefined symbols:** None flagged.
- **Strong wording:** None flagged in appendix.
- **Practical limitations:** Acknowledge that some ablation parameters cannot be controlled operationally.

---

# Global Summary

## Main Repeated Topics to Compress

1. **AI-Var description**: Appears in Abstract, Introduction, Background, Methods. → Consolidate to one comprehensive description in Background.
2. **Lorenz-63 system**: Described in Abstract, Introduction, Background, Methods. → Define once in Background with parameters.
3. **Observation operators (x, xy, x²)**: Defined multiple times. → Define once in Methods, reference thereafter.
4. **Training regimes (Baseline, FixedMean, Resample)**: Described in multiple sections. → Define once in Methods.
5. **RMSE and evaluation metrics**: Defined in multiple places. → Consolidate in Methods.
6. **Φ and f_θ notation**: Used inconsistently. → Define once early, use consistently.
7. **B and R covariance matrices**: Mentioned without early definition. → Define at first use.

## Key Definitions That Must Appear Early

| Symbol | Definition | Where to Define |
|--------|------------|-----------------|
| Φ | Analysis functional: mapping from (observations, background) to MAP state estimate | Abstract or first use in Introduction |
| f_θ | Parametric neural network representation of Φ with learnable parameters θ | Methods, Notation subsection |
| AI-Var | AI-based variational data assimilation approach from Fablet et al. [cite] | Abstract, with citation |
| B | Background error covariance matrix | First use in cost function (Abstract or Introduction) |
| R | Observation error covariance matrix | First use in cost function |
| h(x) | Observation operator: h(x)=x_1, h(x)=(x_1,x_2), or h(x)=x_1² | Methods, before observation modes |
| RMSE | Root Mean Square Error between analysis and true state | Methods, Evaluation Metrics |
| x, xy, x² | Observation modes (partial linear, bilinear, nonlinear) | Methods, after h(x) definition |

## Open Points Marked by Hans as Unclear

### Checklist for Discussion with Hans

- [ ] **ID 31 (p6)**: "I don't understand this citation" — Identify which citation is confusing. *Guess: The citation may be out of context or referring to a claim not clearly stated.*
- [ ] **ID 38 (p7)**: "Again, how do the references come into play here?" — Clarify reference relevance. *Guess: Citations may be placed without clear connection to the text.*
- [ ] **ID 42 (p8)**: "reference?" — Identify which claim needs a reference. *Guess: An empirical claim may lack supporting citation.*
- [ ] **ID 43 (p8)**: "ref" — Another missing reference. *Guess: Similar to above.*
- [ ] **ID 61 (p11)**: "This needs more discussion as J(x) is already fairly linearized" — Discuss implications of linearization. *Guess: The linearized cost function may not capture full nonlinear dynamics; needs acknowledgment.*
- [ ] **ID 76 (p16)**: "unclear" — Passage unclear. *Guess: Sentence may be too technical or missing context.*
- [ ] **ID 77 (p16)**: "why the ref here?" — Citation placement. *Guess: Reference may not directly support the adjacent text.*
- [ ] **ID 78 (p19)**: "??" — Something confusing. *Guess: Missing explanation or inconsistent statement; review surrounding text.*
- [ ] **ID 79 (p20)**: "What does this mean?" — Statement unclear. *Guess: Jargon or undefined term used.*
- [ ] **ID 80 (p20)**: "This needs to be introduced" — Term not introduced. *Guess: A concept or symbol used before definition.*
- [ ] **ID 81 (p21)**: "reference?" — Claim needs citation.
- [ ] **ID 83-84 (p21)**: Citation placement questions.
- [ ] **ID 85 (p22)**: "This needs closer specification" — Detail is vague. *Guess: A parameter or procedure description is incomplete.*
- [ ] **ID 88 (p23)**: "I don't quite understand this" — Passage confusing. *Guess: Technical explanation may be too compressed or missing context.*
- [ ] **ID 99 (p25)**: "why is that?" — Statement needs justification. *Guess: A claim about behavior lacks explanation.*
- [ ] **ID 105 (p29)**: "can you make it a log plot?" — Figure request. *Action: Regenerate if scripts available.*
- [ ] **ID 107 (p29)**: "something is wrong here" — Error in text/figure. *Guess: Possible typo, incorrect value, or mislabeled figure.*
- [ ] **ID 109 (p29)**: "we need to discuss this" — Discussion item. *Guess: Result may be unexpected or require interpretation.*
- [ ] **ID 111 (p30)**: "Maybe keep one plot here" — Plot consolidation suggestion.
- [ ] **ID 112 (p31)**: "what mode" — Figure caption missing observation mode specification.
- [ ] **ID 113 (p32)**: "what noise level" — Figure caption missing noise level specification.
- [ ] **ID 114 (p32)**: "which model is in the figure?" — Figure caption missing model/architecture specification.
- [ ] **ID 118 (p48)**: "no control, maybe test smaller steps" — Suggestion to test finer step sizes in future work.

---

## Wording Guidance from Hans

### Words to Avoid (Too Strong)
- "rigorous" → use "systematic" or "careful"
- "robust" → use "reliable" or "stable"
- "safe" → use "controlled"
- "climatology" (unless mathematically justified)
- "confirm" → use "suggest" or "indicate"
- "prove" → use "demonstrate" or "show"

### Preferred Alternatives
- "rigorous" → "systematic", "thorough"
- "robust" → "reliable", "stable", "consistent"
- "safe" → "controlled", "bounded"
- "confirms" → "suggests", "indicates"
- "replicates" → "investigates", "evaluates"

---

## Action Summary

### High Priority
1. **Define Φ at first use** (Abstract or Introduction)
2. **Use AI-Var terminology consistently** (replace all AI-DA)
3. **Define B and R at first use** with clear descriptions
4. **Clarify Φ vs f_θ relationship** in notation section
5. **Emphasize self-supervised training** (no re-analysis labels)
6. **Soften strong language** (confirm → suggest, robust → reliable)

### Medium Priority
7. Correct 3D-Var vs EnKF characterization
8. Add AI-Var citation at first mention
9. Use lowercase h(x) for observation operators
10. Add time subscripts to notation
11. Reduce redundancy in repeated sections
12. Define observation operators before referring to modes

### Low Priority / Discussion Items
13. Review unclear citations (multiple pages)
14. Consider log-scale plots where requested
15. Consolidate multiple plots where suggested
16. Address specific "unclear" comments after discussion
17. Add observation mode, noise level, and model specification to all figure captions (IDs 112-114)
18. Acknowledge practical limitations in appendix ablation studies (ID 116)
19. Consider finer step sizes for sensitivity analyses in future work (ID 118)

---

*Document generated based on hans_comments_resolution.json and Report/main (3).tex*
*Updated to include pages 31-48 with 7 additional comments covered*
