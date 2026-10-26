# Uncertainty Design Document — LitAgent

## 1. Overview
LitAgent introduces a runtime uncertainty estimator producing a composite score in $[0.0, 1.0]$, where **higher values indicate higher uncertainty**. Low-confidence answers are tagged with explicit warning notices or abstained in strict mode.

---

## 2. Signals

### S1: Retrieval Margin
* **Formula:** $U_{\text{S1}} = 1.0 - \min(1.0, 2 \times (\text{top}_1 - \text{mean}(\text{rest})))$
* **Meaning:** A large gap between the top-ranked candidate and the remaining retrieved candidates indicates high distinctiveness and low uncertainty.
* **Applies to:** `vector`, `both`.

### S2: Evidence Grader Sufficiency
* **Formula:** $U_{\text{S2}} = 1.0 - \text{grade}$
* **Meaning:** The LLM evidence judge rates evidence sufficiency on $[0.0, 1.0]$. Lower grades directly produce higher uncertainty.
* **Applies to:** All routes.

### S5: Graph Execution Outcome
* **Formula:**
  - Status `ok`: $0.0$
  - Status `retried`: $0.3$
  - Status `empty`: $0.8$
  - Status `error`: $1.0$
* **Meaning:** Failed, empty, or retried Cypher queries indicate structural uncertainty.
* **Applies to:** `graph`, `both`.

### S3: Cross-Source Agreement *(Optional for `both` route)*
* **Formula:** $1.0 - \text{cosine}(\text{embed}(\text{graph\_fact}), \text{embed}(\text{vector\_fact}))$.

### S4: Self-Consistency *(Configurable switch)*
* **Formula:** $1.0 - \text{mean\_pairwise\_similarity}$ across $k=3$ sample generations.

---

## 3. Weighting and Calibration
* **Default Weights:**
  - $w_{\text{S1}} = 0.35$
  - $w_{\text{S2}} = 0.35$
  - $w_{\text{S5}} = 0.30$
* **Calibration Protocol:** In Phase 6, logistic regression is fitted on the development split (30%) against binary correctness labels, and evaluated on the test split (70%) reporting AUROC and Expected Calibration Error (ECE).
