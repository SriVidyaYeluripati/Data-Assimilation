# 🎓 Part 1 — Introduction and Problem Setup (Slides 1 – 12)

---

## **Slide 1 – Title Slide**

**Best Subset Selection via Mixed Integer Optimization**

“Good [morning/afternoon]. Today I’ll present *Best Subset Selection via Mixed Integer Optimization*, a work by Bertsimas, King and Mazumder.  
This talk revisits a classical statistical problem — best subset selection — through a modern optimization lens.  
The central message is that what was once computationally impossible can, with advances in optimization algorithms and hardware, be solved exactly and efficiently for moderate-sized problems.”

“We’ll move from the classical formulation to its modern MIO version, analyze theoretical equivalence, and end with computational and statistical insights.”  
*(pause → transition)* “Let me begin with the precise problem statement.”

---

## **Slide 2 – Outline**

“Here is the roadmap.  
We start with the problem formulation, then develop the mixed integer optimization approach.  
We’ll follow with theoretical and statistical analysis, the algorithmic implementation, and finally discuss empirical results and implications.”  
*(transition)* “So let’s begin by formalizing what best subset selection actually means.”

---

## **Slide 3 – Best Subset Selection Problem**

Given data matrix $X \in \mathbb{R}^{n\times p}$ and response vector $y \in \mathbb{R}^n$:

$$
\min_{\beta \in \mathbb{R}^p}\;\tfrac{1}{2}\|y - X\beta\|_2^2
\quad\text{s.t.}\quad
\|\beta\|_0 \le k
$$

where $\|\beta\|_0 = |\{j : \beta_j \ne 0\}|$ counts non-zero coefficients.

**Intuition** — “This seeks the linear model that minimizes residual sum of squares using ≤ k predictors.  
It yields exactly sparse, interpretable models and is theoretically optimal under oracle assumptions.”

*(side note)* If asked ‘Why ½?’ → for derivative convenience; doesn’t change minimizer.  
*(transition)* “However, this elegance hides a severe computational difficulty.”

---

## **Slide 4 – Understanding the Objective Function**

$$
\tfrac{1}{2}\|y − X\beta\|_2^2 = \tfrac{1}{2}\sum_{i=1}^n (y_i − \hat y_i)^2
$$

“The factor ½ simplifies derivatives.  
The constraint $\|\beta\|_0 \le k$ forces most coefficients to zero — selecting ≤ k variables.”

**Geometric intuition** — “Searching for the least-squares hyperplane restricted to coordinate subspaces spanned by k variables.  
This discrete structure gives interpretability but makes the search combinatorial.”  
*(transition)* “That combinatorial explosion is what historically made this problem ‘impossible.’”

---

## **Slide 5 – The Computational Nightmare**

“To find the best subset, one would test every combination of k predictors among p: $\binom{p}{k}$ possibilities.  
For $p = 1000$ and $k = 10$, $\binom{1000}{10} \approx 2.6\times10^{23}$ — utterly intractable.”

**Historical workarounds**
- Convex relaxations (LASSO: replace $\|\beta\|_0$ with $\|\beta\|_1$)  
- Greedy algorithms (forward/backward stepwise)  
- Heuristics (genetic algorithms, simulated annealing)

“All sacrifice exact optimality for speed.”  
*(transition)* “So can we recover exact optimality yet stay computationally reasonable?  
The next slide introduces the key idea — a mixed-integer reformulation.”

---

## **Slide 6 – MIO Reformulation — The Key Insight**

Introduce binary variables $z_i \in \{0,1\}$ indicating whether feature $i$ is active:

$$
\min_{\beta,z}\tfrac{1}{2}\|y − X\beta\|_2^2
$$

subject to

$$
−M z_i \le \beta_i \le M z_i,\quad i=1,\ldots,p,
\qquad
\sum_{i=1}^p z_i \le k,\quad z_i \in \{0,1\}.
$$

Here $M$ is a large constant bounding coefficients.

**Intuition** — “If $z_i=0$ ⇒ $\beta_i=0$; if $z_i=1$ ⇒ $\beta_i\in[−M,M]$.  
$\sum z_i \le k$ imposes sparsity logically rather than by counting.”  
*(transition)* “Let’s see precisely how these Big-M constraints work.”

---

## **Slide 7 – How Big-M Constraints Work**

For each $i$:

$$
−M z_i \le \beta_i \le M z_i
$$

- If $z_i=0$ ⇒ $\beta_i=0$  
- If $z_i=1$ ⇒ $\beta_i\in[−M,M]$

“M must be large enough so all optimal β’s lie inside [−M,M]; too large → numerical issues.  
Later we’ll see theoretical bounds for M.”

*(transition)* “This reformulation was known decades ago, but only recently became computable thanks to hardware and algorithmic advances.”

---

## **Slide 8 – Why MIO Works Now — Computational Revolution (I)**

“From 1990s → 2010s:  
CPU ≈ 10³× faster, memory ≈ 10⁴× larger, multi-core parallelism standard.  
Algorithmic breakthroughs (cutting planes, branching heuristics, presolve) made MIP vastly faster.”

*(Gesture to plot)* “Overall solver capability improved ≈ 10¹¹–10¹²× since early 1990s.”  
*(transition)* “Equally important were software advances.”

---

## **Slide 9 – Why MIO Works Now — Computational Revolution (II)**

“State-of-the-art solvers (Gurobi, CPLEX, SCIP) implement branch-and-bound + cutting planes + presolve + heuristics that find good solutions early and tighten bounds.”

**Key point** — “Net effect: ≈ 200-billion-fold improvement in practical solvability.  
Problems once hopeless are now feasible on modern hardware.”  
*(transition)* “But we must prove that the MIO formulation really solves the original statistical problem.”

---

## **Slide 10 – Why Theoretical Analysis Is Crucial**

“The authors ask:  
1️⃣ Equivalence — does MIO yield same solution as original?  
2️⃣ Parameter bounds — how large must M be?  
3️⃣ Statistical properties — does it retain optimal behavior?  
4️⃣ Certificates — can we know when optimum is reached?”

“Without this analysis we lack guarantees that computation matches statistical truth.”  
*(transition)* “Let’s set up some mathematical preliminaries before proving equivalence.”

---

## **Slide 11 – Mathematical Preliminaries**

Definitions:  
- Support of β: $\text{supp}(\beta)=\{i:\beta_i\ne0\}$  
- ℓ₀ norm: $\|\beta\|_0=|\text{supp}(\beta)|$  
- ℓ₁ norm: $\|\beta\|_1=\sum_i|\beta_i|$  
- ℓ∞ norm: $\|\beta\|_\infty=\max_i|\beta_i|$

Matrix quantities:  
- Restricted eigenvalue  

$$
\eta_k = \min_{|S|\le k}\min_{\substack{\text{supp}(\beta)\subseteq S\\\|\beta\|_2=1}} \frac{\|X\beta\|_2^2}{n}
$$

- Coherence  

$$
\mu = \max_{i\ne j}\frac{|X_i^T X_j|}{\|X_i\|_2 \|X_j\|_2}
$$

*(side note)* $\eta_k$ → conditioning of submatrices; $\mu$ → pairwise correlation.  
*(transition)* “With these we can compare three formulations next.”

---

## **Slide 12 – Problem Hierarchy — Mathematical Structure**

Three problems:

**P₀ (Original Best Subset):**

$$
\min_{\beta}\tfrac{1}{2}\|y − X\beta\|_2^2
\quad\text{s.t.}\quad
\|\beta\|_0\le k
$$

**P₁ (MIO Reformulation):**

$$
\min_{\beta,z}\tfrac{1}{2}\|y − X\beta\|_2^2
\quad\text{s.t.}\quad
−M z_i\le \beta_i\le M z_i,\;
\sum_i z_i\le k,\;
z_i\in\{0,1\}
$$

**P₂ (Relaxation):** Convex approximation adding ℓ₁ and ℓ∞ bounds.

**Interpretation**
- P₀: pure combinatorial  
- P₁: explicit sparsity with binary variables  
- P₂: convex relaxation for analysis  

*(transition)* “Next we see how these relate through the Hierarchy Theorem — showing MIO ≡ original best subset.”

| Concept | What it measures | Good / Bad | Consequence |
|:--|:--|:--|:--|
| $\mu$ (coherence) | Pairwise correlation | $\mu\approx0$ good | High $\mu$ → hard to distinguish variables |
| $\eta_k$ (restricted eigenvalue) | Conditioning of k-subsets | Large $\eta_k$ good | Small $\eta_k$ → instability, large coeffs |

# 🎓 Part 2 — Theoretical Equivalence and Statistical Analysis (Slides 13 – 24)

---

## **Slide 13 – Problem Hierarchy: Three Related Formulations**

“Now that we’ve seen how the MIO formulation is constructed, let’s look at how it fits into a hierarchy of related problems.  
The authors define three formulations — $(P_0)$, $(P_1)$, and $(P_2)$.  
Understanding these helps us see where the exact version and the relaxations live.”

---

### **1️⃣ $P_0$ — Original Best Subset Problem**

$$
\min_{\beta}\frac{1}{2}\|y - X\beta\|_2^2 \quad \text{s.t.} \quad \|\beta\|_0 \le k
$$

“This is the classical version we started with: find the regression coefficients $\beta$ that minimize total squared prediction error using at most $k$ non-zero coefficients.  
It’s purely combinatorial because we must decide which variables are non-zero.”

---

### **2️⃣ $P_1$ — MIO Formulation**

$$
\min_{\beta,z}\frac{1}{2}\|y - X\beta\|_2^2
$$

subject to

$$
-Mz_i \le \beta_i \le Mz_i,\quad \sum_{i=1}^p z_i \le k,\quad z_i\in\{0,1\}
$$

“Here we introduce binary variables $z_i$ to indicate if a feature is active (1) or inactive (0).  
This turns the implicit sparsity constraint into explicit logical rules.”

---

### **3️⃣ $P_2$ — Relaxed Version**

$$
\min_{\beta}\frac{1}{2}\|y - X\beta\|_2^2
\quad\text{s.t.}\quad
\|\beta\|_1\le R_1,\;
\|\beta\|_\infty\le R_\infty,\;
\|\beta\|_0\le k
$$

“$P_2$ adds softer bounds — one on total magnitude ($\ell_1$) and one on the largest coefficient ($\ell_\infty$).  
This makes it closer to convex optimization and easier to analyze mathematically.”

---

**Intuitive picture:**  
“Think of $(P_2)$ as a relaxed outer shell, $(P_1)$ as the exact computational engine, and $(P_0)$ as the ideal goal.  
Together, they form a sandwich of tractability and exactness.”  
*(transition)* “Next we’ll see how these problems relate through an important equivalence theorem.”

---

## **Slide 14 – Hierarchy Theorem (Equivalence and Bounds)**

“The theorem links their optimal values $(Z_0, Z_1, Z_2)$:”

$$
Z_2 \le Z_1 = Z_0
$$

- $(Z_1 = Z_0)$: MIO gives the same optimum as classical best subset  
- $(Z_2 \le Z_1)$: Relaxation provides a lower bound  
- If $M \ge \|\hat\beta_{\text{OLS}}\|_\infty$, the equivalence holds exactly

“In plain words: as long as $M$ isn’t too small, solving the MIO problem is equivalent to solving best subset selection.”  
*(transition)* “Let’s go a step deeper and see why this equivalence is mathematically true.”

---

## **Slides 15–16 – Proof of Equivalence (The MIO–BSS Theorem)**

“The proof proceeds in two directions.”

---

### **Step 1 – From $P_0$ to $P_1$**

Suppose $\beta^*$ solves $(P_0)$.  
Define

$$
z_i^* =
\begin{cases}
1, & \text{if } \beta_i^* \ne 0 \\
0, & \text{otherwise}
\end{cases}
$$

- If $\beta_i^* \ne 0$, constraint $−Mz_i \le \beta_i \le Mz_i$ holds with $z_i=1$  
- If $\beta_i^* = 0$, it holds with $z_i=0$

Thus $(\beta^*, z^*)$ is feasible for $(P_1)$ and gives the same objective.

---

### **Step 2 – From $P_1$ to $P_0$**

If $(\beta^*, z^*)$ solves $(P_1)$:

- $z_i=0 \Rightarrow \beta_i=0$  
- So the number of non-zero $\beta$ equals $\sum z_i \le k$  

Therefore $\beta^*$ is feasible for $(P_0)$ and has the same objective.

**Conclusion:** $(P_0)$ and $(P_1)$ are equivalent — proving that the MIO reformulation exactly reproduces best subset selection.

*(transition)* “But to analyze performance and stability, we need to understand the geometry of $X$ — that’s where coherence and restricted eigenvalues come in.”

---

## **Slide 17 – Matrix Coherence ($\mu$)**

$$
\mu[k-1] = \max_{|T|\le k-1}\max_{i\notin T,j\in T}
\frac{|X_i^T X_j|}{\|X_i\|_2 \|X_j\|_2}
$$

“You can think of this as the worst correlation between any variable outside a chosen subset and those inside it.  
Each $X_i^T X_j$ measures the cosine of the angle between two columns of $X$.”

| μ value | Meaning |
|:--|:--|
| 0 | Orthogonal (independent) |
| 0.5 | Moderate correlation |
| 0.9 | Strong correlation |
| 1 | Identical columns (redundant) |

“High coherence means predictors overlap in information — making variable selection harder.”  
*(transition)* “Next, the restricted eigenvalue provides a broader version of this idea.”

---

# 🎓 **Slides 18–19 (with Proof: Coefficient Bounds Integrated)**

---

## **Slide 18 – Restricted Eigenvalue ($\eta_k$)**

$$
\eta_k = \min_{|S|\le k}\lambda_{\min}\!\left(\frac{1}{n}X_S^T X_S\right)
$$

> “Here we introduce the *restricted eigenvalue*, denoted $\eta_k$.  
> For every subset $S$ of up to $k$ features, we look at the smallest eigenvalue of the corresponding matrix $(X_S^T X_S / n)$.  
> This matrix describes how strongly those features interact.”

---

**Intuitive explanation**

> “Imagine each feature as a direction in space.  
> When some directions almost overlap, it becomes hard to tell them apart — this is what happens when $\eta_k$ is small.  
> A large $\eta_k$ means that even small subsets of features remain independent enough for stable estimation.”

| $\eta_k$ value | Interpretation |
| -------- | ------------------------------------------------------------------ |
| Large | Subsets well-conditioned — coefficients stable |
| Small | Some subset nearly redundant — coefficients can blow up |
| Zero | Perfect collinearity — model cannot distinguish features |

---

**Analogy**

> “Think of a tripod:  
> If its legs are far apart, it stands firm (large $\eta_k$).  
> If the legs are nearly parallel, it topples easily (small $\eta_k$).  
> That’s the geometric intuition here.”

---

**Transition to Proof**

> “Now, using both $\mu$ — the coherence — and $\eta_k$ — the restricted eigenvalue — we can rigorously *prove* the coefficient bounds that we saw earlier.  
> The next slide outlines that proof step by step.”

---

## **(Slide 18.1) Proof: Coefficient Bounds**

> “This slide formally shows how the $\ell_1$ and $\ell_\infty$ bounds are derived using basic linear algebra tools — matrix inversion, perturbation, and inequalities.”

---

### **Part I – $\ell_1$ Bound (Total coefficient magnitude)**

**Step 1: Start from the optimality condition**

$$
X_S^T(y - X_S \hat{\beta}_S) = 0
$$

> “This is the standard least-squares condition: at the optimum, residuals are orthogonal to the selected features.”

---

**Step 2: Rearrange**

$$
X_S^T X_S \hat{\beta}_S = X_S^T y
$$

> “We can express $\hat\beta_S$ as a linear transformation of $y$.”

---

**Step 3: Write $(X_S^T X_S)$ as $(I + G)$**

> “If predictors were perfectly uncorrelated, $(X_S^T X_S)$ would equal the identity $I$.  
> But correlations cause deviations, captured by $G$, where $\|G\|_{1,1} \le \mu[k−1]$.”

*(Say aloud)* “$G$ measures how correlated the chosen features are — when $\mu$ is small, $G$ is tiny.”

---

**Step 4: Bound the matrix inverse**

$$
(X_S^T X_S)^{-1} = (I + G)^{-1}, \quad
\|(I + G)^{-1}\|_{1,1} \le \frac{1}{1 - \|G\|_{1,1}}
$$

> “This comes from matrix perturbation theory.  
> As long as $G$ isn’t too large (predictors not too correlated), the inverse is stable.”

---

**Step 5: Substitute back**

$$
\|\hat{\beta}_S\|_1 \le \frac{1}{1 - \mu[k-1]}\|X_S^T y\|_1
$$

> “In words: the total coefficient magnitude grows as predictors become more correlated.”

---

### **Part II – $\ell_\infty$ Bound (Maximum coefficient)**

**Step 1: Start from normal equations**

$$
\hat{\beta}_S = (X_S^T X_S)^{-1} X_S^T y
$$

**Step 2: Apply Cauchy–Schwarz**

$$
\|\hat{\beta}_S\|_\infty \le \|(X_S^T X_S)^{-1}\|_2 \, \|X_S^T y\|_2
$$

> “This bounds how large any single coefficient can be, depending on matrix conditioning.”

---

**Step 3: Use restricted eigenvalue**

$$
\|(X_S^T X_S)^{-1}\|_2 = 1 / \lambda_{\min}(X_S^T X_S) \le 1 / \eta_k
$$

---

**Step 4: Bound the right-hand term**

$$
\|X_S^T y\|_2 \le \sqrt{k}\|y\|_2
$$

---

**Step 5: Combine**

$$
\|\hat{\beta}_S\|_\infty \le \frac{1}{\sqrt{\eta_k}}\|y\|_2
$$

> “The largest coefficient is bounded by $\|y\|_2$ scaled by $1/\sqrt{\eta_k}$ — poor conditioning ($\eta_k$ small) makes coefficients explode.”

---

**Slide 18.1 Summary (to say aloud)**

> “So the $\ell_1$ bound depends on $\mu$ (pairwise correlation),  
> and the $\ell_\infty$ bound depends on $\eta_k$ (global stability).  
> Together they tell us when coefficients stay small — and guide how to set the Big-M constant in the MIO problem.”

---

**Transition to Slide 19**

> “Now that we’ve proved these bounds, let’s interpret them statistically — how to pick $M$ and ensure stability.”

---

## **Slide 19 – Parameter Bounds and Big-M Choice**

> “The coefficient bounds we just proved provide explicit guidance for choosing the Big-M constant and understanding numerical stability.”

$$
\|\hat\beta\|_1
\le
\frac{1}{1-\mu[k-1]}
\sum_{j=1}^{k}|\langle X_{(j)},y\rangle|,
\quad
\|\hat\beta\|_\infty
\le
\frac{1}{\sqrt{\eta_k}}\|y\|_2
$$

---

**Interpretation**

> “The $\ell_1$ bound controls total coefficient size (depends on $\mu[k−1]$),  
> while the $\ell_\infty$ bound limits the largest coefficient (depends on $\eta_k$).  
> High $\mu$ or small $\eta_k$ means instability and larger coefficients.”

---

**Practical takeaway**

> “Pick $M$ large enough to exceed these bounds, ensuring feasibility but not too large to avoid solver instability:  
> typically, $M \approx \frac{1}{\sqrt{\eta_k}}\|y\|_2$.”

*(transition)* “With bounded coefficients and proper $M$, the MIO is mathematically correct and numerically stable.  
Next, we link these properties to statistical prediction risk.”

---

## **Slide 21 – Statistical Risk — Measuring Prediction Error**

Prediction risk:

$$
E[\|X(\hat\beta - \beta^*)\|_2^2]
$$

- $(\hat\beta - \beta^*)$: estimation error  
- $X(\hat\beta - \beta^*)$: prediction error  
- Expectation averages over data noise

“This measures the average squared error on new data.”  
*(transition)* “Next, we introduce the notation used in the theorem.”

---

## **Slide 22 – Statistical Optimality Notation Guide**

| Symbol | Description | Meaning |
|:--|:--|:--|
| $E[\cdot]$ | Expectation | Average over data randomness |
| $\hat\beta_{\text{BSS}}$ | Best-subset estimator | From MIO |
| $\hat\beta_{\text{LASSO}}$ | LASSO estimator | From $\ell_1$-penalized regression |
| $\beta^*$ | True coefficients | Ground truth |
| $\sigma^2$ | Noise variance | Unexplained variation |
| $s$ | True sparsity | Non-zeros in $\beta^*$ |
| $(n,p,k)$ | Sample size, features, subset size | Dimensions |
| $\phi_{\min}^2(k)$ | Restricted eigenvalue (for LASSO) | Correlation penalty |

*(transition)* “Now we can state the main statistical theorem.”

---

## **Slide 23 – Theorem on Prediction Risk Bounds**

$$
E[\|\hat{\beta}_{\text{BSS}} - \beta^{*}\|_2^2]
\le
C_1\,\frac{\sigma^2 s \log p}{n}
$$

$$
E[\|\hat{\beta}_{\text{LASSO}} - \beta^{*}\|_2^2]
\le
C_2\,\frac{\sigma^2 s \log p}{\phi_{\min}^{2}(k)\, n}
$$



**Interpretation:**
- Both share the optimal rate $\frac{\sigma^2 s \log p}{n}$.  
- LASSO worsens when $\phi_{\min}(k)$ decreases (high correlation).  
- Best-subset keeps the oracle rate regardless of correlations.

*(transition)* “Let’s see why this happens mathematically.”

---

## **Slide 24 – Proof Sketch for Statistical Optimality**

1️⃣ **Oracle case:**  
If we knew the true support $S^*$, OLS on those columns gives risk ≈ $\sigma^2 s/n$ (oracle rate).  

2️⃣ **Search penalty:**  
Since we must identify $S^*$ among $p$ candidates, a $\log p$ factor appears → $\sigma^2 s \log p / n$.  

3️⃣ **Correlation impact:**  
LASSO’s $\ell_1$ shrinkage introduces dependence on $\phi_{\min}^2(k)$; small $\phi_{\min}$ inflates risk.  
Best-subset avoids this issue.

**Summary:**  
- Best-subset achieves oracle rate without extra assumptions.  
- LASSO degrades under strong correlation but remains cheaper computationally.

*(transition)* “Next, we’ll move to the algorithmic implementation and empirical evidence.”  


# 🎓 Part 3 — Algorithm, Experiments, and Implications (Slides 25 – 37)

---

## **Slide 25 – When Best Subset Wins**

“So far, we’ve seen that best-subset selection is statistically optimal in theory.  
But when does this actually matter in practice — when does it beat alternatives like LASSO?”

### **When LASSO tends to fail**
“LASSO struggles when predictors are highly correlated.  
Because it shrinks coefficients via an $\ell_1$ penalty, correlated variables tend to share or split coefficients unpredictably.  
It also fails when the design matrix is ill-conditioned — some columns nearly linearly dependent.”

### **When Best Subset Selection stays strong**
“Best subset selection performs well when:
- The signal-to-noise ratio is decent (true features stand out),  
- The true sparsity level $s$ is known or approximated,  
- The correct support can be recovered.”

| Method | Statistical Quality | Computational Cost |
| :-- | :-- | :-- |
| Best Subset (MIO) | Oracle-level accuracy | Heavy optimization (~minutes–hours) |
| LASSO | Slightly biased under correlation | Very fast (ms–s) |

“So we gain exactness at the price of computation.”  
*(transition)* “Let’s explore how MIO keeps that computation feasible — through branch-and-bound.”

---

## **Slide 26 – Branch-and-Bound Example**

“Mixed-integer optimization solvers use **branch-and-bound** — a smart search over possible subsets.”

**Example:**  
“Suppose we have four features and want to pick two.”

1️⃣ Solve a relaxed version with fractional $z_i \in [0,1]$ → maybe $z^* = (0.7, 0.6, 0.4, 0.3)$ with objective 100.  
2️⃣ Branch on $z_1=0$ or $z_1=1$, creating two subproblems.  
3️⃣ Compare relaxed objectives (say 105 and 110) with the current best integer solution.  
4️⃣ If a branch can’t improve the best solution, it’s pruned.

“In essence, it searches the combinatorial space but skips huge chunks proven suboptimal.”  
*(transition)* “To decide when to stop, solvers rely on optimality certificates.”

---

## **Slide 27 – Optimality Certificates**

“An optimality certificate quantifies how close our current solution is to the global optimum.”

$$
\frac{Z_{\text{true opt}} - Z_{\text{current best}}}{Z_{\text{current best}}} \le \varepsilon
$$

“If this relative gap ≤ ε, the solution is within ε of the true optimum.”

**Example:**  
“A 2% gap means the model’s objective is at most 2% worse than the absolute best possible — a mathematical guarantee.”

“LASSO can’t provide this; MIO can.”  
*(transition)* “However, starting MIO from scratch can be slow.  
The authors fix this with a two-stage algorithm.”

---

## **Slide 28 – Two-Stage Algorithm: Motivation**

“They identify a **cold-start problem** — solvers waste time before finding any good integer solution.”

### **Stage 1 — Warm-start with a fast heuristic**
“Run a discrete first-order method (≈30–60s) that quickly finds a good subset and feasible $(\beta,z)$.”

### **Stage 2 — MIO refinement**
“Start MIO from that warm-start, not from scratch.  
Because the solver already knows a good upper bound, it prunes aggressively — converging in 1–10 minutes instead of hours.”

“This hybrid design merges speed from heuristics and optimality from MIO.”  
*(transition)* “Let’s see the exact steps.”

---

## **Slide 29 – Complete Algorithm I**

### **Stage 1 — Discrete first-order method**

1. Initialize $\beta^{(0)} = 0$  
2. For $t = 1,2,\dots,T_{\max}$:
   - Compute gradient $g^{(t)} = X^T(X\beta^{(t-1)} - y)$  
   - Select support $S^{(t)}$ of $k$ indices with smallest $|g_i|$  
   - Solve restricted least squares on $S^{(t)}$ to update $\beta$

*(Comment)* “This is a greedy yet efficient search that focuses on the most informative features.”

---

### **Stage 2 — MIO with warm start**

- Set $M = 2\|\beta^{(T_{\max})}\|_\infty$  
- Initialize $(\beta, z) = (\beta^{(T_{\max})}, z^{(T_{\max})})$  
- Run solver until optimality gap $< \varepsilon$ or time limit

**Output:** $\beta^*$ — optimal coefficients + certificate of global optimality.  
*(transition)* “Now let’s see how this performs in experiments.”

---

## **Slide 30 – Key Experimental Findings**

**Effect of correlation:**

| Correlation $\rho$ | Observation |
| :-- | :-- |
| 0.5 (low) | MIO ≈ LASSO — similar performance |
| 0.8 (moderate) | MIO starts outperforming |
| 0.9 (high) | MIO ≈ 30% lower prediction error |

“As $\rho$ increases, MIO’s advantage grows — it chooses exact subsets rather than shared coefficients.”

---

**Computation times**

| Method | Typical runtime |
| :-- | :-- |
| LASSO | ms–s |
| Warm-start heuristic | 30–60 s |
| MIO (warm start) | 1–10 min |
| MIO (cold start) | Often fails to finish |

“MIO yields exact sparsity and unbiased coefficients; LASSO shrinks coefficients toward zero.”  
*(transition)* “The next slide visualizes this difference.”

---

## **Slide 31 – Key Experimental Figures**

“The figure plots prediction error vs. correlation level.”

- For $\rho=0.5$, both curves overlap.  
- As $\rho \to 0.9$, LASSO’s error rises steeply, MIO’s stays nearly flat.

*(Gesture)* “Notice LASSO’s curve climbing — MIO remains robust.”  
*(transition)* “Still, the experiments have some limitations.”

---

## **Slide 32 – Experimental Design Flaws and Missing Answers**

“The experiments are convincing but limited in scope.”

- Mostly synthetic data: $n=500$, $p=100$ → relatively easy  
- Real-world high-dimensional cases are much larger

| Domain | Typical $n$ | Typical $p$ |
| :-- | :-- | :-- |
| Genomics | 50 | 20 000 |
| NLP (Text) | $10^6$ | $10^6$ |

“In such regimes, MIO becomes impractical — results don’t scale directly.”  
*(transition)* “The next slide continues this critique.”

---

## **Slide 33 – Critical Experimental Limitations**

“Key missing tests:”

- Outliers, missing data, or heteroscedastic noise  
- Non-sparse true models (model misspecification)  
- Comparison to modern penalized methods (Elastic Net, SCAD, MCP)

**Parameter-selection gap:**  
“In real use, $k$ is unknown — choosing it via cross-validation multiplies runtime by ≈10×.”

“So, MIO works beautifully under ideal conditions but not yet for large, messy data.”  
*(transition)* “Let’s consider what this means for high-dimensional statistics.”

---

## **Slide 34 – Implications for High-Dimensional Statistics I**

“Scalability is the main issue.”

“When $p \gg n$ (e.g., $p=10^6$, $n=10^2$):  
$\binom{p}{k}$ explodes, making even smart branch-and-bound infeasible.  
Statistically, reliable recovery also needs $n \gg s\log p$, which rarely holds.”

*(transition)* “The next slide summarizes this paradox.”

---

## **Slide 35 – The High-Dimensional Paradox**

| We want BSS because… | But it fails because… |
| :-- | :-- |
| Handles correlation & gives interpretability | Search complexity explodes as $p$ grows |
| Yields unbiased coefficients | Computationally infeasible beyond a few thousand features |

“Thus, MIO-based subset selection is powerful for moderate $p$, but not a replacement for convex relaxations in large-scale ML.”  
*(transition)* “The authors close with a balanced summary.”

---

## **Slide 36 – Summary and Assessment**

### **Core Contributions**

1. **Computational feasibility:**  
   Proved that exact best-subset selection is solvable for moderate $p$ with modern solvers.

2. **Rigorous theory:**  
   Proved mathematical equivalence and explicit parameter bounds.

3. **Statistical optimality:**  
   Showed best-subset achieves the oracle risk rate.

4. **Algorithmic innovation:**  
   Proposed a two-stage solver with optimality certificates in minutes.

---

**Critical assessment:**  
“A landmark methodological step, but not a silver bullet.  
It re-establishes best-subset selection as viable, yet convex methods like LASSO and Elastic Net remain essential for high-dimensional problems.”  
*(transition)* “The final slide captures its broader impact.”

---

## **Slide 37 – Conclusion and Broader Impact**

“To conclude, *Best Subset Selection via Mixed Integer Optimization* bridges statistics and optimization.  
It shows that with modern solvers, we can achieve exact sparsity and theoretical optimality once thought impossible.”

### **Broader significance**

“Beyond regression, this approach redefines how we view estimation — as discrete optimization solvable with rigorous tools.  
It’s a paradigm shift: not replacing convex methods, but unifying computational and statistical perspectives.”

---

**Closing line:**  
“In summary:

- Conceptually, it unites classical statistics with modern optimization.  
- Practically, it offers a powerful tool for medium-scale problems.  
- Philosophically, it reminds us that tractability evolves — what was impossible twenty years ago may be routine tomorrow.  

**Thank you.**”
