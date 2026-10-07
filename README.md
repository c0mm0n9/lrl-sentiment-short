# Rationale of Latent-Space Oversampling: SMOTE for Low-Resource Emotion Classification

### *High-Dimensional Interpolation and Generalization Under Severe Class Imbalance*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/HuggingFace-UAReviews-FFD21E.svg)](https://huggingface.co/datasets/KSE-RESEARCH-Group/UAReviews)
[![Encoder](https://img.shields.io/badge/Encoder-Qwen3--Embedding--0.6B-green.svg)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
[![Artifacts](https://img.shields.io/badge/Artifacts-Google%20Drive-4285F4.svg?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/195TJc0iXsjVrNvXVaJjSjJKEzOE7wQlT?usp=drive_link)

---

## 📖 Table of Contents

- [Summary & Key Takeaways](#summary-key-takeaways)
  - [Claim Status](#claim-status)
  - [Key Takeaways](#key-takeaways)
- [Limitations & Methodological Caveats](#limitations-methodological-caveats)
- [Repository Structure, Pipeline & Reproducibility](#repository-structure-pipeline-reproducibility)
  - [Reproducibility Note](#reproducibility-note)
- [1. The Real-World Dilemma](#1-the-real-world-dilemma)
- [2. Existing Works](#2-existing-works)
- [3. What is Added](#3-what-is-added)
- [4. Data & Latent-Space Analysis](#4-data-latent-space-analysis)
  - [4.1 Data](#41-data)
  - [4.2 Latent-Space Representation](#42-latent-space-representation)
  - [4.3 Similarity and Class Purity](#43-similarity-and-class-purity)
  - [4.4 Tail Class Margin Distribution](#44-tail-class-margin-distribution)
- [5. Experiment Setup](#5-experiment-setup)
- [6. Empirical Results](#6-empirical-results)
  - [6.1 Master Benchmark Performance Table](#61-master-benchmark-performance-table)
  - [6.2 In-Distribution Results](#62-in-distribution-results)
  - [6.3 Out-of-Distribution Challenge Results](#63-out-of-distribution-challenge-results)
  - [6.4 Comparison (Macro F1 Gain Heatmaps)](#64-comparison-macro-f1-gain-heatmaps)
  - [6.5 Comparison (Tail-Class Gain Matrices)](#65-comparison-tail-class-gain-matrices)
  - [6.6 Statistical Significance Analysis (McNemar's Test)](#66-statistical-significance-analysis-mcnemars-test)
  - [6.7 Stratified Bootstrap Statistical Significance](#67-stratified-bootstrap-statistical-significance)
- [7. Conclusion](#7-conclusion)
- [8. References](#8-references)
- [9. Quick Start & Reproduction Instructions](#9-quick-start-reproduction-instructions)
- [License](#license)

---

## Summary & Key Takeaways

This repository benchmarks latent-space oversampling on frozen 1024-d transformer embeddings under severe natural class imbalance (139.2:1 head-to-tail ratio).

Using Ukrainian consumer reviews from [`KSE-RESEARCH-Group/UAReviews`](https://huggingface.co/datasets/KSE-RESEARCH-Group/UAReviews), we evaluate four resampling strategies (**Exact Random Duplication**, **Classic SMOTE**, **smote_renorm**, and **EmbSMOTE**) across a 5-point oversampling sweep $\rho \in \{0.10, 0.25, 0.50, 0.75, 1.00\}$ on two classifier architectures: a linear hyperplane (`LinearSVC`) and a neural network (`TorchMLPClassifier`).

### Claim Status

| Claim | Status | Supporting Evidence |
| :--- | :---: | :--- |
| Mild oversampling improves Macro F1 over an unweighted baseline | **Supported** (LinearSVC) / **Exploratory** (TorchMLP) | LinearSVC gains are positive across all $\rho$ on both splits (Test $\Delta = +0.0294$, Challenge $\Delta = +0.0598$); TorchMLP gains are subject to run-to-run neural variance ([§6.1](#61-master-benchmark-performance-table), [§6.7](#67-stratified-bootstrap-statistical-significance)). |
| $\rho \approx 0.10 - 0.25$ represents an optimal operating range over full balance ($\rho = 1.00$) | **Exploratory** | Peak observed at $\rho=0.25$ (SVC) and $\rho=0.10$ (MLP) within the evaluated 5-point grid, but variations across $\rho$ (~1–2 pp) are small relative to bootstrap sampling uncertainty (~2–3 pp half-width), and LinearSVC gains remain positive at $\rho=1.00$ ([§6.1](#61-master-benchmark-performance-table)). |
| Duplication leads in-distribution; SMOTE generalizes better out-of-distribution | **Exploratory / Indistinguishable** | LinearSVC duplication descriptively leads SMOTE on Test by 0.48 pp, but also leads on Challenge at $\rho \in \{0.75, 1.00\}$; observed gaps (~0.5–0.7 pp) lack confidence intervals and are within sampling noise ([§6.1](#61-master-benchmark-performance-table), [§6.4](#64-comparison-macro-f1-gain-heatmaps)). |
| Latent oversampling recovers dead tail classes (Disgust, Sadness, Surprise, Fear) | **Not confirmed on Test** / **Exploratory on Challenge** | On Test, Disgust $\Delta\text{F1} = -0.043$ and Sadness $\Delta\text{F1} = +0.005$ (all 95% CIs include 0). On Challenge, Disgust gain ($+0.163$) is significant only in an ad-hoc family of 4 ($p_{\text{Holm}}=0.042^*$) and attenuates to $p_{\text{Holm}}=0.063$ (ns) across all 7 classes ([§6.5](#65-comparison-tail-class-gain-matrices), [§6.7](#67-stratified-bootstrap-statistical-significance)). |
| TorchMLP statistically outperforms LinearSVC under SMOTE | **Supported for Accuracy** / **No detectable difference for Macro F1** | Statistically significant improvement in overall sample accuracy via McNemar test ($\chi^2 = 11.52, p < 0.001$, [§6.6](#66-statistical-significance-analysis-mcnemars-test)), but paired Macro-F1 bootstrap difference is non-significant ($\Delta = -0.0069$, 95% BCa CI $[-0.059, +0.032]$, $p_{\text{Holm}} = 0.7520$, [§6.7](#67-stratified-bootstrap-statistical-significance)). |

### Key Takeaways

1. **Best observed operating range occurs at mild oversampling ($\rho \approx 0.10 - 0.25$)**:
   Across the 5-point grid, the highest Macro F1 scores are observed at $\rho=0.10$ for `TorchMLP` (Challenge F1 = 0.4605 in sweep) and $\rho=0.25$ for `LinearSVC` (Challenge F1 = 0.4401). Full rebalancing ($\rho \to 1.00$) inflates dataset size $4.57\times$ (from 8,106 to 37,030 samples) and increases LinearSVC training latency up to $21.6\times$ (from 6.16s to 133.10s) without improving F1 over mild oversampling, though $\rho$ effect sizes (~1–2 pp) remain small relative to sampling error (details in [§6.1](#61-master-benchmark-performance-table)).

2. **Resampling strategies are statistically indistinguishable in aggregate**:
   While exact duplication descriptively achieves the highest single In-Distribution score on `LinearSVC` (+4.56 pp vs. +4.08 pp for SMOTE at $\rho=0.10$), and Classic SMOTE achieves the highest observed score on Challenge (+5.98 pp at $\rho=0.25$), duplication also outperforms SMOTE at higher ratios on Challenge ($\rho=0.75, 1.00$). The ~0.5–0.7 pp differences between methods lack paired confidence intervals and should be interpreted as descriptive rankings rather than structural advantages (details in [§6.1](#61-master-benchmark-performance-table), [§6.4](#64-comparison-macro-f1-gain-heatmaps)).

3. **Classifier architecture influences sensitivity to resampling**:
   `LinearSVC` exhibits steady, low-variance metric progressions across oversampling conditions ($\sigma_{\text{F1}} = 0.009$, calculated as the sample standard deviation of Macro F1 across all 20 evaluated oversampled conditions). In contrast, `TorchMLP` displays substantial metric variability ($\sigma_{\text{F1}} = 0.025$). The hypothesis that neural decision boundaries overfit to localized synthetic clusters remains an unverified mechanism that is confounded with neural initialization and optimization stochasticity (details in [§6.1](#61-master-benchmark-performance-table), [§Limitations](#limitations--methodological-caveats)).

4. **EmbSMOTE in closed training sets functions as degree-weighted duplication**:
   Without an external retrieval pool, 1-NN nearest-neighbor projection snaps synthetic chord vectors back to one of their two parent endpoints 100% of the time. EmbSMOTE effectively operates as exact duplication weighted by local training neighborhood density (details in [§5](#5-experiment-setup)).

---

## Limitations & Methodological Caveats

To ensure research integrity, the findings in this repository must be interpreted alongside five methodological boundaries:

1. **Hyperparameter Selection Leakage (Winner's Curse)**:
   The champion oversampling ratios ($\rho=0.25$ for LinearSVC, $\rho=0.10$ for TorchMLP) were selected post-hoc by identifying peak Macro F1 on the Challenge set. Evaluating significance on the same partition introduces optimistic selection bias. On Challenge, Disgust exhibits $\Delta\text{F1} = +0.1633$ and Sadness $\Delta\text{F1} = +0.1108$. When evaluated on the unselected In-Distribution Test set, Disgust reverses to $\Delta\text{F1} = -0.0431$ and Sadness drops to $\Delta\text{F1} = +0.0054$. The In-Distribution Test set partition acts as the unselected confirmation benchmark.

2. **Run-to-Run Variance & Neural Training Divergence**:
   Due to GPU compute constraints, initial experimental conditions were evaluated with a single training run per configuration ($N_{\text{runs}} = 1$, SEED=42). A discrepancy historically existed between experimental artifacts for TorchMLP at $\rho=0.10$:
   - The full grid sweep checkpoint ([`phase4_sweep_summary.csv`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase4/phase4_sweep_summary.csv)) records Baseline Macro F1 = 0.4012 and SMOTE Macro F1 = 0.4605 on Challenge (+5.92 pp gain; Test: 0.4592 $\to$ 0.4904).
   - The dedicated prediction export checkpoint ([`predictions_mlp_rho01.json`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase4/predictions_mlp_rho01.json)) used for Figure 8 and Figure 9 reflects Baseline Macro F1 = 0.4071 and SMOTE Macro F1 = 0.4332 on Challenge (+2.61 pp gain; Test: 0.4208 $\to$ 0.4490).
   To enforce complete programmatic reproducibility, the training and inference pipeline with fixed random seed (`SEED = 42`) has been appended directly to [`phase_4.ipynb`](./phase_4.ipynb) (Section 7, Cell 13) and integrated into [`phase_4_analysis.ipynb`](./phase_4_analysis.ipynb) (Section 11a, Cell 29). Running this fixed-seed pipeline guarantees deterministic prediction serialization directly to disk.

3. **Sparse Tail Evaluation Support**:
   In both Test and Challenge evaluation splits ($N = 1,737$ each), the extreme tail categories possess very few positive instances:
   - `Surprise`: $N=9$ (Test), $N=8$ (Challenge)
   - `Fear`: $N=9$ (Test), $N=8$ (Challenge)
   - `Disgust`: $N=16$ (Test), $N=16$ (Challenge)
   - `Sadness`: $N=64$ (Test), $N=63$ (Challenge)
   With $N=8$, correctly classifying a single additional review shifts per-class F1 by $\approx 0.10$. Confidence intervals for these categories are correspondingly wide and frequently include zero.

4. **Unweighted Baseline Objective**:
   The baseline LinearSVC models were fit using standard unweighted loss ($C=1.0$, `class_weight=None`). Consequently, the tail-class recall collapse observed on raw data is partly driven by the unweighted optimization objective rather than inherent representation failure. Cost-sensitive loss weighting (`class_weight='balanced'`) was not evaluated as a competing baseline.

5. **Characterization of Evaluation Splits**:
   Class distributions across the Train, Test, and Challenge splits are statistically identical (Happiness $\approx 65.2\%$, Fear $\approx 0.5\%$). Text overlap analysis reveals comparable near-duplicate rates with Train (1.04% for Challenge vs. 1.21% for Test). Rather than an adversarially shifted stress test, the Challenge split constitutes a distinct held-out crawl partition from the `UAReviews` benchmark.

---

## Repository Structure, Pipeline & Reproducibility

> **Precomputed Artifacts & Checkpoints**: Pre-encoded embeddings, split indices, and sweep checkpoints are archived on [Google Drive](https://drive.google.com/drive/folders/195TJc0iXsjVrNvXVaJjSjJKEzOE7wQlT?usp=drive_link).

| Notebook | Purpose | Key Artifacts |
| :--- | :--- | :--- |
| **[`pre_encode.ipynb`](./pre_encode.ipynb)** | Downloads `UAReviews` from Hugging Face, cleans text, extracts stratified splits (`train`: 8,106, `test`: 1,737, `challenge`: 1,737), and generates normalized 1024-d embeddings using `Qwen3-Embedding-0.6B`. | `lrl_cache/cache/ua_reviews_clean.parquet`<br>`lrl_cache/cache/embeddings/qwen3_embeddings_*.npy`<br>`lrl_cache/cache/split_indices.json`<br>`lrl_cache/cache/label_encoder.json` |
| **[`phase_1_3.ipynb`](./phase_1_3.ipynb)** | **Phases 1–3**: Exploratory data analysis and latent geometry visualization. Computes UMAP, t-SNE, and PCA 2D projections, class centroid cosine geometries, 5-NN neighborhood purity, 5-fold Stratified **Out-of-Fold (OOF) Linear SVM** margin distributions, and baseline confusion matrices. | `figures/class_imbalance_distribution.png`<br>`figures/embedding_natural_clusters_umap_tsne.png`<br>`figures/manifold_purity_centroids.png`<br>`figures/oof_margin_distributions.png`<br>`figures/baseline_confusion_matrices.png` |
| **[`phase_4.ipynb`](./phase_4.ipynb)** | **Phase 4 (GPU Sweep)**: Controlled class imbalance experiment with 4 oversampling strategies across target ratios $\rho \in \{0.10, 0.25, 0.50, 0.75, 1.00\}$. Evaluates `LinearSVC` and CUDA `TorchMLPClassifier` on both test and challenge sets. | `phase4/phase4_sweep_summary.csv`<br>`phase4/strat_*.json` (42 conditions) |
| **[`phase_4_analysis.ipynb`](./phase_4_analysis.ipynb)** | **Phase 4 (Plot-Building & Empirical Analysis)**: Computes absolute and relative gains, generates publication figures (Macro F1 curves, gain heatmaps, class gain matrices), McNemar tests (Figure 8), and Stratified Bootstrap intervals (Figure 9). | `figures/fig1_macro_f1_vs_rho.png`<br>`figures/fig2d_test_class_gain_heatmap.png`<br>`figures/fig2d_chal_class_gain_heatmap.png`<br>`figures/fig8_mcnemar_statistical_test.png`<br>`figures/fig9_stratified_bootstrap_analysis.png` |

### Reproducibility Note

- **Random Seeds**: Resampling pipelines and baseline splits use `SEED = 42`.
- **Runs per Condition**: Single run per experimental condition ($N_{\text{runs}} = 1$).
- **Figure Sources**:
  - Figures 1–7 are generated directly from [`phase4/phase4_sweep_summary.csv`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase4/phase4_sweep_summary.csv) by [`phase_4_analysis.ipynb`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase_4_analysis.ipynb) (Cells 1–25).
  - Prediction checkpoints (`predictions_linearsvc_rho025.json` and `predictions_mlp_rho01.json`) can be generated via [`phase_4.ipynb`](./phase_4.ipynb) (Section 7, Cell 13) or directly within [`phase_4_analysis.ipynb`](./phase_4_analysis.ipynb) (Section 11a, Cell 29).
  - Figure 8 (McNemar) is generated from these prediction JSON files by [`phase_4_analysis.ipynb`](./phase_4_analysis.ipynb) Cell 31.
  - Figure 9 (Stratified Bootstrap) is generated from the same prediction JSON files by [`phase_4_analysis.ipynb`](./phase_4_analysis.ipynb) Cell 34.

---

## 1. The Real-World Dilemma

- Real-world text classification frequently deals with severe class imbalance (e.g., 100+:1).
- Generative data augmentation with large language models introduces substantial compute overhead and can be prone to hallucination or dialectal drift in low-resource language (LRL) settings such as Ukrainian.
- Latent-space oversampling (SMOTE on frozen transformer embeddings) runs efficiently on CPU/GPU, but its empirical behavior in high-dimensional embedding spaces ($d=1024$) under severe natural imbalance remains less extensively evaluated than tabular benchmarks.

---

## 2. Existing Works

- **Chawla et al. (2002)**: Foundational SMOTE algorithm on low-dimensional tabular data.
- **Blagus & Lusa (2013)**: Showed through simulation and empirical analysis that SMOTE often fails to alleviate majority bias in high-dimensional $d \gg N$ settings due to distance concentration.
- **Chen et al. (2014, WEMOTE)**: Applied interpolation to static Word2Vec representations; observed out-of-vocabulary vector blending.
- **Taşkıran et al. (2025)**: Evaluated 31 SMOTE variants on English MiniLMv2 embeddings without evaluating oversampling ratio sweeps ($\rho$) or out-of-distribution generalization.
- **Inoshita (2026, EmbSMOTE)**: Benchmarked retrieval-anchored oversampling on GoEmotions-28 to mitigate generative augmentation drift.

---

## 3. What is Added

- **Target Language**: Ukrainian (morphologically rich) on real-world customer reviews (`UAReviews`).
- **Imbalance Severity**: Extreme natural 139.2:1 ratio (`Happiness`: 5,290 vs. `Fear`: 38).
- **Sweep Range**: Controlled 5-point sweep $\rho \in \{0.10, 0.25, 0.50, 0.75, 1.00\}$.
- **Evaluation**: Dual in-distribution (Test) and held-out Challenge split evaluation.

| Dimension | Existing Literature | This Work | Notes |
| :--- | :--- | :--- | :--- |
| **Language & Resources** | High-resource English (TREC, GoEmotions). | **Ukrainian Customer Reviews** (`UAReviews`). | Evaluates behavior on morphologically rich text with moderate pretraining exposure. |
| **Imbalance Severity** | Moderate or synthetic ratios ($10:1$ to $30:1$). | **Natural Extreme Imbalance (139.2:1)** (`Happiness`: 5,290 vs. `Fear`: 38). | Tests the high-dimensional setting where $N_{\text{tail}} = 38 \ll d = 1024$. |
| **Sweep Dynamics** | Fixed full rebalancing ($\rho = 1.00$) or ad-hoc ratios. | **Controlled Sweep**: $\rho \in \{0.10, 0.25, 0.50, 0.75, 1.00\}$. | Evaluates metric progression across $\rho$; identifies small performance variations relative to sampling noise. |
| **Evaluation Splits** | Single in-distribution (IID) splits. | **Dual In-Distribution (Test) vs. Held-out Challenge Split**. | Evaluates generalization stability; observes that resampling variants are descriptively ranked but statistically indistinguishable. |
| **Norm Regularization** | Standard Euclidean SMOTE on normalized vectors. | **Explicit Hyperspherical Renormalization** (`smote_renorm`) on $\mathbb{S}^{1023}$. | Tests the effect of projecting interpolated chords back onto the unit hypersphere. |
| **Classifier Geometries** | Heterogeneous ensembles without geometric isolation. | **Linear Hyperplanes (`LinearSVC`) vs. Neural Classifiers (`TorchMLP`)**. | Compares rigid linear margins against flexible continuous neural boundaries. |
| **EmbSMOTE Mechanism** | Described as generative augmentation. | **Analysis of Closed-Set Graph-Density Duplication**. | Demonstrates that in closed training sets, EmbSMOTE functions as degree-weighted sample duplication. |

---

## 4. Data & Latent-Space Analysis

### 4.1 Data

11,580 Ukrainian consumer reviews across 7 emotion categories ([`KSE-RESEARCH-Group/UAReviews`](https://huggingface.co/datasets/KSE-RESEARCH-Group/UAReviews)):

- **Train Split**: 8,106 samples (70.0%)
- **In-Distribution Test Split**: 1,737 samples (15.0%)
- **Out-of-Distribution Challenge Split**: 1,737 samples (15.0%)
- **Imbalance**: Majority (`Happiness`: 5,290) to tail minority (`Fear`: 38) is **139.2 : 1**.

Class support across splits:
- `Happiness`: Train 5,290 (65.26%), Test 1,133 (65.23%), Challenge 1,134 (65.28%)
- `Anger`: Train 1,585 (19.55%), Test 339 (19.52%), Challenge 340 (19.57%)
- `Neutral`: Train 782 (9.65%), Test 167 (9.61%), Challenge 168 (9.67%)
- `Sadness`: Train 297 (3.66%), Test 64 (3.68%), Challenge 63 (3.63%)
- `Disgust`: Train 74 (0.91%), Test 16 (0.92%), Challenge 16 (0.92%)
- `Surprise`: Train 40 (0.49%), Test 9 (0.52%), Challenge 8 (0.46%)
- `Fear`: Train 38 (0.47%), Test 9 (0.52%), Challenge 8 (0.46%)

![Dataset Splits & Class Imbalance](figures/class_imbalance_distribution.png)

### 4.2 Latent-Space Representation

Embeddings generated with `Qwen3-Embedding-0.6B` ($d = 1024$, $\ell_2$-normalized).

![Latent-Space Representation - UMAP & t-SNE](figures/embedding_natural_clusters_umap_tsne.png)

- 2D UMAP and t-SNE projections show `Happiness` forming a relatively distinct cluster, while negative and neutral emotions overlap heavily.
- Tail classes (`Fear` and `Surprise`) do not form isolated clusters; they overlap heavily with the boundaries of `Anger` and `Happiness`.

### 4.3 Similarity and Class Purity

![Centroid Cosine Similarity & 5-NN Local Neighborhood Purity](figures/manifold_purity_centroids.png)

- **Centroid Cosine Similarities (1024-d)**: High pairwise cosine similarities across classes ($\cos(\text{Anger}, \text{Sadness}) = 0.944$, $\cos(\text{Sadness}, \text{Disgust}) = 0.954$, $\cos(\text{Neutral}, \text{Sadness}) = 0.949$). *Caveat*: Cosine similarities in the 0.80–0.95 range across all pairs reflect the well-documented anisotropy (cone effect) of transformer embedding spaces rather than pure emotional equivalence.
- **5-NN Local Neighborhood Purity & Class Prior Lift**:
  - `Happiness`: **93.6%** (prior: 65.3%, lift: $1.4\times$)
  - `Anger`: **63.8%** (prior: 19.6%, lift: $3.3\times$)
  - `Neutral`: **35.3%** (prior: 9.65%, lift: $3.7\times$)
  - `Sadness`: **18.8%** (prior: 3.66%, lift: $5.1\times$)
  - `Fear`: **10.5%** (prior: 0.47%, lift: **$22.3\times$**)
  - `Disgust`: **7.0%** (prior: 0.91%, lift: **$7.7\times$**)
  - `Surprise`: **0.5%** (prior: 0.49%, lift: **$1.0\times$**)

Consistent with Blagus & Lusa (2013), when $d = 1024$ and $N_{\text{tail}} \le 40$, local Euclidean and cosine neighborhoods are heavily populated by majority samples, though Fear and Disgust maintain local neighborhood lift well above their baseline empirical priors.

### 4.4 Tail Class Margin Distribution

![Linear SVM Signed Margin Gap Distribution](figures/oof_margin_distributions.png)

5-fold Stratified **Out-of-Fold (OOF) Linear SVM** signed margin gaps $\Delta(x) = s_y - \max_{c \neq y} s_c$:
- Evaluated strictly within the **training set** using 5-fold cross-validation with holdout validation folds to inspect margins without test set leakage.
- Majority classes maintain median margins $> +2.0$.
- For `Fear` and `Surprise`, the margin distribution falls below 0 (median $\Delta < -1.0$), resulting in zero diagonal recall.
- *Limitation*: Because the baseline SVM uses standard unweighted loss ($C=1.0$, `class_weight=None`), this margin collapse is partly an expected consequence of unweighted empirical risk minimization under extreme imbalance.

![Baseline LinearSVC and TorchMLP Confusion Matrices](figures/baseline_confusion_matrices.png)

- **Holdout Test & Challenge Evaluation**: Evaluates baseline `LinearSVC` and `TorchMLP` trained on the full training set ($N=8,106$) on separate holdout splits ($N=1,737$ each).
- On both holdout splits, baseline models yield **F1 = 0.0000** for `Fear` and `Surprise` due to complete decision-boundary suppression.

---

## 5. Experiment Setup

- **Embeddings**: Frozen 1024-d unit-normalized vectors from `Qwen3-Embedding-0.6B`.
- **Classifiers**:
  - `LinearSVC`: Scikit-learn fixed defaults ($C=1.0$, `class_weight=None`, `max_iter=5000`, `random_state=42`).
  - `TorchMLPClassifier`: PyTorch MLP architecture ($1024 \to 256 \to 64 \to 7$), AdamW optimizer ($\text{lr}=10^{-3}$, weight decay $= 10^{-4}$), batch size 128, early stopping on 10% stratified validation split carved from train.
- **Oversampling Strategies**:
  - `duplication`: Exact random sampling with replacement from minority class instances.
  - `smote`: Classic Euclidean interpolation between $k=5$ nearest cosine neighbors.
  - `smote_renorm`: SMOTE interpolation followed by $\ell_2$-normalization back onto $\mathbb{S}^{1023}$.
  - `embsmote`: 1-NN cosine projection of synthetic vectors back to existing training instances.
- **Target Ratios**: $\rho \in \{0.10, 0.25, 0.50, 0.75, 1.00\}$, defining target count $N_c = \max(N_c^{\text{orig}}, \text{round}(\rho \cdot N_{\text{majority}}))$.

---

## 6. Empirical Results

### 6.1 Master Benchmark Performance Table

*All 17 rows verified against [`phase4/phase4_sweep_summary.csv`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase4/phase4_sweep_summary.csv). Note on metric progression variability: Across all 20 evaluated oversampled conditions per classifier, LinearSVC exhibits highly stable performance with sample standard deviation $\sigma_{\text{F1}} = 0.0094 \approx 0.009$ on Test Macro F1 ($\sigma = 0.0093$ on Challenge), whereas TorchMLP displays higher sensitivity to oversampling and training dynamics with $\sigma_{\text{F1}} = 0.0245 \approx 0.025$ on Test ($\sigma = 0.0192$ on Challenge).*

| Strategy | $\rho$ | Classifier | Train Size | Test Macro F1 | Test Acc | Chal Macro F1 | Fear F1 | Surprise F1 | Disgust F1 | Sadness F1 | Train Time (s) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **none (Baseline)** | 1.00 | LinearSVC | 8,106 | 0.4465 | 0.8780 | 0.3803 | 0.0000 | 0.0000 | 0.3158 | 0.3596 | 6.16 |
| **none (Baseline)** | 1.00 | TorchMLP | 8,106 | 0.4592 | 0.8751 | 0.4012 | 0.0000 | 0.0000 | 0.3333 | 0.4174 | 8.63 |
| **embsmote (Peak Test)** | 0.10 | TorchMLP | 9,773 | **0.4968** | 0.8739 | 0.4377 | 0.0000 | **0.2000** | 0.4167 | 0.4248 | 5.20 |
| **duplication** | 0.10 | TorchMLP | 9,773 | 0.4924 | **0.8785** | 0.4555 | 0.1667 | 0.0000 | 0.4167 | 0.3697 | 3.22 |
| **duplication** | 0.10 | LinearSVC | 9,773 | 0.4921 | 0.8653 | 0.4264 | 0.1905 | 0.1111 | 0.3500 | 0.3762 | 9.79 |
| **smote (Peak Chal)** | 0.10 | TorchMLP | 9,773 | 0.4904 | 0.8687 | **0.4605** | 0.0000 | 0.1818 | 0.4444 | 0.3838 | 5.58 |
| **smote** | 0.10 | LinearSVC | 9,773 | 0.4873 | 0.8664 | 0.4294 | 0.2105 | 0.1053 | 0.2703 | 0.3962 | 8.75 |
| **smote_renorm** | 0.10 | LinearSVC | 9,773 | 0.4864 | 0.8682 | 0.4029 | 0.2353 | 0.1250 | 0.2424 | 0.3762 | 8.80 |
| **smote_renorm** | 0.25 | LinearSVC | 13,485 | 0.4874 | 0.8601 | 0.4376 | 0.2353 | 0.1176 | 0.2564 | 0.3906 | 16.33 |
| **smote (Peak Chal SVC)** | 0.25 | LinearSVC | 13,485 | 0.4759 | 0.8515 | **0.4401** | 0.2000 | 0.1000 | 0.2727 | 0.3650 | 15.28 |
| **embsmote** | 0.25 | LinearSVC | 13,485 | 0.4692 | 0.8486 | 0.4378 | 0.1818 | 0.1000 | 0.2727 | 0.3529 | 19.12 |
| **duplication** | 0.50 | LinearSVC | 21,160 | 0.4879 | 0.8359 | 0.4162 | 0.2857 | 0.1053 | 0.3043 | 0.3841 | 44.27 |
| **embsmote** | 0.50 | TorchMLP | 21,160 | 0.4883 | 0.8705 | 0.4003 | 0.1538 | 0.1818 | 0.2500 | 0.4068 | 9.30 |
| **duplication** | 0.75 | TorchMLP | 29,098 | 0.4890 | 0.8659 | 0.4046 | **0.3333** | 0.0000 | 0.3200 | 0.3571 | 18.42 |
| **duplication** | 1.00 | LinearSVC | 37,030 | 0.4823 | 0.8227 | 0.4332 | 0.2727 | 0.0909 | 0.3333 | 0.3515 | 133.10 |
| **smote** | 1.00 | LinearSVC | 37,030 | 0.4719 | 0.8261 | 0.4228 | 0.2000 | 0.0952 | 0.3256 | 0.3684 | 53.25 |
| **embsmote** | 1.00 | LinearSVC | 37,030 | 0.4596 | 0.8261 | 0.4176 | 0.1000 | 0.1053 | 0.3256 | 0.3660 | 189.36 |

---

### 6.2 In-Distribution Results

![LinearSVC & TorchMLP In-Distribution Test Macro F1 vs Oversampling Ratio](figures/fig1_macro_f1_vs_rho.png)

- Test set Macro F1 peaks at mild oversampling ($\rho = 0.10$, descriptive mean F1 = 0.4792 across methods).
- On `LinearSVC`, exact duplication descriptively achieves the highest point estimate (+4.56 pp gain vs. +4.08 pp for SMOTE at $\rho=0.10$), though variants remain statistically indistinguishable.
- Full rebalancing ($\rho \to 1.00$) decreases classification accuracy from $87.8\%$ to $82.3\%$ ($r = -0.923$).

---

### 6.3 Out-of-Distribution Challenge Results

![Macro F1 Out-of-Distribution Challenge Curves](figures/fig3_test_vs_challenge_robustness.png)

- On the Challenge split, Classic SMOTE achieves the highest observed point estimates (+5.98 pp on `LinearSVC` at $\rho=0.25$; +5.92 pp on `TorchMLP` at $\rho=0.10$ in the sweep run).
- At higher ratios ($\rho \in \{0.75, 1.00\}$), duplication matches or exceeds SMOTE on LinearSVC (0.4332 vs 0.4228 at $\rho=1.00$).
- Because these differences (~0.5–1.0 pp) are smaller than the bootstrap sampling uncertainty (~2–3 pp), claims of structural superiority between duplication and SMOTE are not statistically supported.

---

### 6.4 Comparison (Macro F1 Gain Heatmaps)

![Macro F1 Gain Heatmap Across All Conditions](figures/fig6_gain_heatmap.png)

- Evaluates gain progression relative to unweighted baseline across all 40 oversampled conditions.
- Demonstrates that gains on `LinearSVC` remain positive across all $\rho \in [0.10, 1.00]$ on both splits, with peak gains concentrated at $\rho \in [0.10, 0.50]$.

---

### 6.5 Comparison (Tail-Class Gain Matrices)

#### In-Distribution Test Set Class Gain Matrix
![In-Distribution Test Set Tail-Class Gain Matrix](figures/fig2d_test_class_gain_heatmap.png)

#### Out-of-Distribution Challenge Set Class Gain Matrix
![Out-of-Distribution Challenge Set Tail-Class Gain Matrix](figures/fig2d_chal_class_gain_heatmap.png)

- **Class Support Context**: In both evaluation splits ($N=1,737$), tail evaluations reflect small sample counts: `Fear` ($N=8$ Chal, $N=9$ Test), `Surprise` ($N=8$ Chal, $N=9$ Test), `Disgust` ($N=16$ Chal, $N=16$ Test), `Sadness` ($N=63$ Chal, $N=64$ Test).
- **Disgust Baseline Qualification**: On the Challenge split, baseline models achieve 0.0000 F1 for Disgust. On the In-Distribution Test set, however, baseline models achieve substantial non-zero performance (0.3158 for LinearSVC, 0.3333 for TorchMLP).
- **Observed Per-Class Shifts**: On Challenge, LinearSVC SMOTE ($\rho=0.25$) records positive point gains on all four tail emotions (`Disgust` $+0.163$, `Sadness` $+0.111$, `Fear` $+0.105$, `Surprise` $+0.083$). On Test, however, Disgust decreases by $-0.043$ and Sadness changes by $+0.005$.

---

### 6.6 Statistical Significance Analysis (McNemar's Test)

![Figure 8: McNemar Statistical Test](figures/fig8_mcnemar_statistical_test.png)

To evaluate paired sample-level classification discordance, we compute McNemar's test across the primary deployed configurations using both Edwards continuity-corrected $\chi^2$ and two-sided exact binomial tests:

#### McNemar Contingency & Hypothesis Test Summary Table

*Recomputed from [`phase4/predictions_linearsvc_rho025.json`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase4/predictions_linearsvc_rho025.json) and [`phase4/predictions_mlp_rho01.json`](file:///c:/Users/admin/OneDrive/Documents/COMP-2501-Project/lrl-sentiment-short/phase4/predictions_mlp_rho01.json):*

| Comparison | Evaluation Split | Baseline Acc | SMOTE Acc | $b$ (M1 $\times$, M2 $\checkmark$) | $c$ (M1 $\checkmark$, M2 $\times$) | Net ($b - c$) | Odds Ratio (smoothed) | Raw $b/c$ | Edwards $\chi^2$ | Edwards $p$ | Exact Binomial $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearSVC: Base vs SMOTE ($\rho=0.25$)** | Challenge (OOD) | 86.47% | 83.59% | 33 | 83 | $-50$ | 0.401 | 0.398 | 20.70 | $5.38 \times 10^{-6}$ | $3.87 \times 10^{-6}$ |
| **LinearSVC: Base vs SMOTE ($\rho=0.25$)** | Test (In-Dist) | 87.80% | 85.15% | 31 | 77 | $-46$ | 0.406 | 0.403 | 18.75 | $1.49 \times 10^{-5}$ | $1.12 \times 10^{-5}$ |
| **TorchMLP: Base vs SMOTE ($\rho=0.10$)** | Challenge (OOD) | 85.72% | 85.90% | 57 | 54 | $+3$ | 1.055 | 1.056 | 0.036 | 0.849 | 0.850 |
| **TorchMLP: Base vs SMOTE ($\rho=0.10$)\*** | Test (In-Dist) | 86.47% | 86.93% | 54 | 46 | $+8$ | 1.172 | 1.174 | 0.490 | 0.484 | 0.484 |
| **H2H: LinearSVC vs TorchMLP (SMOTE)** | Challenge (OOD) | 83.59% | 85.90% | 86 | 46 | $+40$ | 1.860 | 1.870 | 11.52 | $6.88 \times 10^{-4}$ | $6.31 \times 10^{-4}$ |
| **H2H: LinearSVC vs TorchMLP (SMOTE)** | Test (In-Dist) | 85.15% | 86.93% | 69 | 38 | $+31$ | 1.805 | 1.816 | 8.41 | 0.0037 | 0.0035 |

*\*Note: Baseline Test accuracy is 86.47% in the cached inference run, compared to 87.51% in the full grid sweep checkpoint ([§Limitations](#limitations--methodological-caveats)).*

#### Key Statistical Takeaways
1. **Decision Boundary Tilt in LinearSVC**: For `LinearSVC`, oversampling at $\rho=0.25$ causes a statistically significant drop in overall sample accuracy ($86.47\% \to 83.59\%$, Edwards $\chi^2 = 20.70, p = 5.38 \times 10^{-6}$; exact $p = 3.87 \times 10^{-6}$). The linear hyperplane tilts to capture tail instances at the expense of 50 net misclassifications in majority classes.
2. **No Detectable Change in Overall Accuracy for TorchMLP**: For `TorchMLP`, SMOTE at $\rho=0.10$ yields no statistically detectable shift in overall accuracy on Challenge ($b=57, c=54$, net $+3$, Edwards $\chi^2 = 0.036, p = 0.849$).
3. **Statistically Significant Accuracy Advantage for TorchMLP**: In the head-to-head comparison, `TorchMLP (ρ=0.10 SMOTE)` achieves significantly higher overall sample accuracy than `LinearSVC (ρ=0.25 SMOTE)` on Challenge (Edwards $\chi^2 = 11.52, p = 6.88 \times 10^{-4}$; exact $p = 6.31 \times 10^{-4}$) and Test ($\chi^2 = 8.41, p = 0.0037$). *Important distinction*: This advantage is strictly for overall sample accuracy; as shown below in [§6.7](#67-stratified-bootstrap-statistical-significance), paired Macro-F1 differences between the two models are non-significant.

---

### 6.7 Stratified Bootstrap Statistical Significance

While McNemar's test evaluates paired overall sample misclassifications, Macro F1 is an unweighted harmonic mean across classes. To quantify evaluation-set sampling uncertainty, we run a **paired, class-stratified bootstrap** ($B = 2,000$ replicates, fixed empirical class priors, seed 42) on both Test and Challenge sets ($N = 1,737$ each). We report observed differences, percentile and **BCa 95% intervals** with stratum-scaled acceleration ($a$), two-sided $p$-values, and Holm-adjusted $p$-values split by evaluation domain ($m=3$ Challenge, $m=3$ Test).

![Figure 9: Stratified Bootstrap Non-Parametric Significance & Confidence Intervals](figures/fig9_stratified_bootstrap_analysis.png)

#### Paired Macro-F1 Gains ($B=2,000$)

| Model & Resampling Condition | Evaluation Split | Baseline Macro F1 | SMOTE Macro F1 | Observed $\Delta$ | 95% BCa CI | Raw $p$ (BCa) | Holm $p$ (Domain $m=3$) | Significance (Holm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearSVC + SMOTE ($\rho=0.25$)** | Challenge (OOD) | 0.380 | 0.440 | **+0.0598** | **[+0.023, +0.108]** | 0.0003 | 0.0008 | *** |
| **TorchMLP + SMOTE ($\rho=0.10$)** | Challenge (OOD) | 0.407 | 0.433 | **+0.0261** | **[+0.003, +0.063]** | 0.0189 | 0.0379 | * |
| **Head-to-Head (MLP vs SVC)** | Challenge (OOD) | 0.440 (SVC) | 0.433 (MLP) | -0.0069 | [-0.059, +0.032] | 0.7520 | 0.7520 | ns |
| **LinearSVC + SMOTE ($\rho=0.25$)** | Test (In-Dist) | 0.447 | 0.476 | +0.0294 | [-0.018, +0.094] | 0.2780 | 0.7647 | ns |
| **TorchMLP + SMOTE ($\rho=0.10$)** | Test (In-Dist) | 0.421 | 0.449 | +0.0282 | [-0.018, +0.101] | 0.2549 | 0.7647 | ns |
| **Head-to-Head (MLP vs SVC)** | Test (In-Dist) | 0.476 (SVC) | 0.449 (MLP) | -0.0269 | [-0.086, +0.025] | 0.3238 | 0.7647 | ns |

#### LinearSVC Tail Emotion Recovery on Challenge Set ($B=2,000$)

*Comparing ad-hoc family of 4 tail classes vs. complete family of 7 classes and unselected Test set:*

| Emotion Category | Challenge $\Delta$ F1 | 95% Percentile CI | Raw $p$ (2-sided) | Holm $p$ (Family $m=4$) | Holm $p$ (All 7 Classes) | Test $\Delta$ F1 [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sadness** | +0.111 | [+0.010, +0.222] | 0.0305 | 0.0915 (ns) | 0.1524 (ns) | +0.005 [-0.097, +0.117] |
| **Disgust** | +0.163 | [+0.040, +0.304] | 0.0105 | **0.0420 (*)** | 0.0630 (ns) | -0.043 [-0.244, +0.194] |
| **Surprise** ($N=8$) | +0.083 | [+0.000, +0.240] | 0.3233 | 0.6467 (ns) | 1.0000 (ns) | +0.100 [0.000, +0.300] |
| **Fear** ($N=8$) | +0.105 | [+0.000, +0.300] | 0.3578 | 0.6467 (ns) | 1.0000 (ns) | +0.200 [0.000, +0.435] |

*(Omitted head class context on Challenge: Anger $\Delta\text{F1} = -0.0547$ [$-0.082, -0.027$], Raw $p=0.0010$, Holm $p=0.0070^{**}$; Happiness $\Delta\text{F1} = -0.0039$, Holm $p=1.0000$).*

#### Key Empirical Insights from Bootstrap Resampling
1. **Unselected In-Distribution Test Gains Are Not Statistically Significant**: On the unselected Test set, neither `LinearSVC` ($\Delta = +0.0294$, 95% BCa $[-0.018, +0.094]$, $p_{\text{Holm}} = 0.7647$) nor `TorchMLP` ($\Delta = +0.0282$, 95% BCa $[-0.018, +0.101]$, $p_{\text{Holm}} = 0.7647$) excludes zero from their 95% confidence intervals. On the Challenge set, observed gains ($\Delta = +0.0598$ for SVC, $\Delta = +0.0261$ for MLP) are statistically significant within the domain family of 3, but remain exploratory due to post-hoc ratio selection ([§Limitations](#limitations--methodological-caveats)).
2. **Disgust Tail Gain Does Not Survive Multiplicity Control Across All Classes**: In an ad-hoc family of 4 tail classes, Disgust achieves $p_{\text{Holm}} = 0.0420^*$. However, when controlling across all 7 evaluated classes, the Holm adjusted $p$-value rises to $p_{\text{Holm}} = 0.0630$ (ns), and **zero tail classes achieve statistical significance**. Furthermore, on the unselected Test set, Disgust shows a negative point gain ($\Delta\text{F1} = -0.043$).
3. **No Detectable Macro-F1 Difference Between LinearSVC and TorchMLP**: Under paired bootstrap evaluation, the Macro F1 gap between TorchMLP and LinearSVC on Challenge is non-significant ($\Delta = -0.0069$, 95% BCa CI $[-0.059, +0.032]$, $p_{\text{Holm}} = 0.7520$). While the 90% percentile interval ($[-0.047, +0.029]$) falls inside an illustrative $\pm 0.05$ margin, the 95% interval exceeds it; we conclude there is no detectable difference in Macro F1 rather than asserting statistical equivalence.

#### Stratum-Scaled Acceleration Factor Derivation
For multi-sample (stratified) bootstrap, the acceleration factor uses stratum-scaled influence values (Davison & Hinkley, 1997, §5.3.2):
$$a = \frac{\sum_c n_c^{-3} \sum_i U_{ic}^3}{6 \left( \sum_c n_c^{-2} \sum_i U_{ic}^2 \right)^{3/2}}$$
where $U_{ic} = (n_c - 1)(\bar{\theta}_{c\cdot} - \hat{\theta}_{(ic)})$. The stratum weights $n_c^{-3}$ and $n_c^{-2}$ are necessary for imbalanced corpora ($n_c \in [8, 1134]$): omitting them causes majority strata to disproportionately dominate the cubic sum, which distorts the acceleration constant $a$ and biases intervals. Stratum-scaling restores equitable contributions across imbalanced categories.

---

## 7. Conclusion

1. **Mild oversampling provides positive observed Macro F1 gains on linear models**: Across both evaluation splits, `LinearSVC` demonstrates consistent positive Macro F1 gains under mild oversampling ($\rho \in [0.10, 0.25]$), while full rebalancing ($\rho \to 1.00$) inflates dataset size $4.57\times$ and latency up to $21.6\times$ without improving performance ([§6.1](#61-master-benchmark-performance-table)).
2. **Resampling variants are statistically indistinguishable**: Duplication and SMOTE variants perform within ~0.5–0.7 pp of each other, well within evaluation sampling uncertainty ([§6.1](#61-master-benchmark-performance-table), [§6.4](#64-comparison-macro-f1-gain-heatmaps)).
3. **Accuracy gains do not imply Macro F1 gains**: `TorchMLP` significantly outperforms `LinearSVC` in overall sample classification accuracy under McNemar's test ([§6.6](#66-statistical-significance-analysis-mcnemars-test)), but exhibits no detectable Macro F1 advantage under paired bootstrap evaluation ([§6.7](#67-stratified-bootstrap-statistical-significance)).
4. **Tail-class recovery claims require cautious calibration**: High tail gains observed on the Challenge split reflect selection bias and small evaluation support ($N \le 16$), and do not replicate on the unselected Test set ([§6.5](#65-comparison-tail-class-gain-matrices), [§6.7](#67-stratified-bootstrap-statistical-significance)).

---

## 8. References

- **Blagus, R., & Lusa, L. (2013).** SMOTE for high-dimensional class-imbalanced data. *BMC Bioinformatics*, 14, Article 106. [DOI: 10.1186/1471-2105-14-106](https://doi.org/10.1186/1471-2105-14-106)
- **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. [DOI: 10.1613/jair.953](https://doi.org/10.1613/jair.953)
- **Chen, T., Xu, R., Liu, B., Lu, Q., & Xu, J. (2014).** WEMOTE: Word embedding based minority oversampling technique for imbalanced emotion and sentiment classification. In *Proceedings of the 4th International Workshop on Web Intelligence & Mining (WISDOM '14)* (pp. 1–8). Association for Computing Machinery. [URL](https://sentic.net/wisdom2014chen.pdf)
- **Davison, A. C., & Hinkley, D. V. (1997).** *Bootstrap Methods and Their Application*. Cambridge University Press.
- **KSE Research Group. (2024).** UAReviews: Ukrainian customer reviews and emotion classification benchmark [Data set]. Hugging Face. [URL](https://huggingface.co/datasets/KSE-RESEARCH-Group/UAReviews)
- **Taşkıran, S. F., Türkoğlu, B., Kaya, E., & Aşuroğlu, T. (2025).** A comprehensive evaluation of oversampling techniques for enhancing text classification performance. *Scientific Reports*, 15(1), Article 5791. [DOI: 10.1038/s41598-025-05791-7](https://doi.org/10.1038/s41598-025-05791-7)
- **Qwen Team. (2025).** Qwen3-Embedding-0.6B [Large language and embedding model]. Hugging Face. [URL](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Inoshita, K. (2026).** Class-structure preservation beats diversity: A comprehensive benchmark of text augmentation methods for imbalanced text classification (arXiv:2608.12340). arXiv. [DOI: 10.48550/arXiv.2608.12340](https://doi.org/10.48550/arXiv.2608.12340)

---

## 9. Quick Start & Reproduction Instructions

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support recommended (e.g., Google Colab T4/L4/A100 or local GPU)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/c0mm0n9/lrl-sentiment-short.git
cd lrl-sentiment-short
pip install -r requirements.txt
```

### 2. Download Precomputed Artifacts (Optional)
To run analysis or generate figures without re-encoding text or re-running the GPU sweep:
- Download cached embeddings and sweep checkpoints from [Google Drive Artifacts](https://drive.google.com/drive/folders/195TJc0iXsjVrNvXVaJjSjJKEzOE7wQlT?usp=drive_link).
- The repository expects data in the `lrl_cache/cache/` directory (with embeddings stored in `lrl_cache/cache/embeddings/` as `qwen3_embeddings_train.npy`, `qwen3_embeddings_test.npy`, and `qwen3_embeddings_challenge.npy`).

### 3. Execution Pipeline
1. **Pre-encoding**: Run [`pre_encode.ipynb`](./pre_encode.ipynb) to download `UAReviews` and cache 1024-d embeddings.
2. **Exploratory Analysis**: Run [`phase_1_3.ipynb`](./phase_1_3.ipynb) to inspect UMAP/t-SNE clusters, cosine purity, and baseline OOF margins.
3. **Resampling Sweep**: Run [`phase_4.ipynb`](./phase_4.ipynb) to execute the resampling matrix across $\rho \in [0.10, 1.00]$.
4. **Figure Generation**: Run [`phase_4_analysis.ipynb`](./phase_4_analysis.ipynb) to generate all publication plots, McNemar tests, and bootstrap intervals.

---

## License

This project is prepared as an academic deliverable for HKU COMP2501 (Introduction to Data Science and Engineering).

- **Research Artifacts, Documentation & Presentation Deck**: Licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Code & Pipeline Implementation**: Licensed under the [MIT License](LICENSE).
