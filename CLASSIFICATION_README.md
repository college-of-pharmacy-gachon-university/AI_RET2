# Classification Models Documentation

Detailed technical specifications for the two classification pipelines in AI-RET.

---

## Table of Contents

- [Overview](#overview)
- [Multi-Kinase Selectivity Classification](#multi-kinase-selectivity-classification)
- [NSCLC Phenotypic Response Classification](#nsclc-phenotypic-response-classification)
- [Model Evaluation Metrics](#model-evaluation-metrics)

---

## Overview

Two classification tasks are implemented:

1. **Multi-Kinase Selectivity Classification** — multi-task activity prediction across 23 kinase targets from ChEMBL
2. **NSCLC Phenotypic Response Classification** — drug sensitivity prediction across 108 NSCLC cell lines from GDSC1/GDSC2

Both tasks use the same four molecular fingerprints (Avalon, ECFP, ECFP_COUNT, MACCS), seven classifiers, and four train/test split strategies, producing a comprehensive benchmark across all combinations.

---

## Multi-Kinase Selectivity Classification

**Code:** `MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/multitask_classifier_with_default_parameters.py`

### Specifications

| Item | Details |
|------|---------|
| **Input** | 94,213 unique compounds × 23 kinase targets |
| **Activity threshold** | Per-target Youden's J applied to pIC50 values |
| **Fingerprints** | Avalon (2048 bits), ECFP (r=2, 2048 bits), ECFP_COUNT (r=2, 2048 bits), MACCS (167 bits) |
| **Classifiers** | RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, SVM, KNeighbors, GaussianNB |
| **Split strategies** | Random, Scaffold (Murcko), Butina, UMAP-cluster (80/20) |
| **Feature scaling** | MaxAbsScaler (preserves sparse matrix structure) |
| **Probability threshold** | Fixed 0.5 for binary predictions |

### Pipeline

```
Input SMILES
  → Fingerprint generation (Avalon / ECFP / ECFP_COUNT / MACCS)
  → MaxAbsScaler
  → 7 classifiers × 4 split strategies
  → Per-target AUC, F1, MCC, AUPRC reported
```

### Run

```bash
conda activate ai_ret

# Data transformation (pIC50 matrix + per-target Youden thresholds):
python MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/transform_kinase_multitask.py \
    MULTI_KINASE_CLASSIFICATION_CODES_DATA/raw_input/KINASE_Dataset_With_IC50_188217.csv \
    --output_prefix kinase_multitask_14May_V15_188217_Original \
    --dedup_mode smiles

# Training (all FP × classifier × split combinations):
python MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/multitask_classifier_with_default_parameters.py --max_workers 32

# Youden threshold analysis:
python MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/run_youden_analysis.py
```

---

## NSCLC Phenotypic Response Classification

**Code:** `PHENOTYPES_CLASSIFICATION_CODES_DATA/code/`

Three scripts share the same base class (`multitask_classifier_zscore.py`) and differ only in class-imbalance handling:

| Script | Imbalance strategy |
|--------|--------------------|
| `multitask_classifier_unweighted.py` | None (baseline) |
| `multitask_classifier_weighted.py` | SAURON-RF sample weights: w = N_res / N_sens for sensitive compounds |

### Specifications

| Item | Details |
|------|---------|
| **Input** | 429 NSCLC drugs × 108 cell lines (41,352 drug–cell-line pairs) |
| **Sensitivity label** | Z-score threshold (Z=0, −0.5, −0.75, −1.0); compound is sensitive if z ≤ threshold |
| **Fingerprints** | Avalon, ECFP, ECFP_COUNT, MACCS |
| **Classifiers** | RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, SVM, KNeighbors, GaussianNB |
| **Split strategies** | Random, Scaffold (Murcko), Butina, UMAP-cluster (80/20) |
| **Feature scaling** | StandardScaler |
| **Probability threshold** | Youden's J (per cell line) |

### Pipeline

```
GDSC1/GDSC2 combined Z-scores
  → Z-score threshold → binary sensitivity labels
  → Fingerprint generation (Avalon / ECFP / ECFP_COUNT / MACCS)
  → StandardScaler
  → 7 classifiers × 4 split strategies (per cell line)
  → Youden's J threshold per cell line
  → AUC, F1, MCC, AUPRC reported per cell line
```

### Run (Z=0, primary analysis)

```bash
conda activate ai_ret

# Data transformation:
python PHENOTYPES_CLASSIFICATION_CODES_DATA/code/transform_for_multitask_zscore.py \
    GDSC1_GDSC2_Combined_Dataset_With_SMILES.csv \
    -o gdsc_multitask_threshold_zeo --threshold 0.0

# Unweighted (baseline):
python PHENOTYPES_CLASSIFICATION_CODES_DATA/code/multitask_classifier_unweighted.py \
    --data_folder gdsc_multitask_threshold_zeo_output \
    --output_prefix gdsc_multitask_threshold_zeo \
    --z_threshold 0.0 --no_save_models \
    --results_path PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted

# Weighted (SAURON-RF):
python PHENOTYPES_CLASSIFICATION_CODES_DATA/code/multitask_classifier_weighted.py \
    --data_folder gdsc_multitask_threshold_zeo_output \
    --output_prefix gdsc_multitask_threshold_zeo \
    --z_threshold 0.0 --imbalance_strategy sample_weight \
    --no_save_models --max_workers 32 \
    --results_path PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_weighted
```

---

## Model Evaluation Metrics

All combinations report the following per split:

| Metric | Description |
|--------|-------------|
| AUC-ROC | Area under ROC curve; primary selection metric |
| AUPRC | Area under precision-recall curve |
| F1 | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient (−1 to 1; 0 = random) |
| Accuracy | Fraction of correct predictions |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |

### Threshold selection

- **Multi-kinase**: per-target Youden's J on pIC50 to set the activity/inactivity boundary during data preparation; probability cutoff fixed at 0.5 during evaluation.
- **NSCLC phenotype**: Youden's J (J = TPR − FPR) applied post-hoc per cell line to select the optimal probability cutoff.

---

## References

1. Sushko I, et al. Applicability Domains for Classification Problems: Benchmarking of Distance to Models for Ames Mutagenicity Set. *J. Chem. Inf. Model.* 2010;50:2094–2111. DOI: [10.1021/ci100253r](https://doi.org/10.1021/ci100253r)
2. Lenhof K, et al. Simultaneous regression and classification for drug sensitivity prediction using an advanced random forest method. *Sci. Rep.* 2022;12:13458. DOI: [10.1038/s41598-022-17609-x](https://doi.org/10.1038/s41598-022-17609-x)
3. Youden WJ. Index for rating diagnostic tests. *Cancer.* 1950;3(1):32–35.
4. Rogers D, Hahn M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* 2010;50:742–754. DOI: [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
5. Bemis GW, Murcko MA. The Properties of Known Drugs. 1. Molecular Frameworks. *J. Med. Chem.* 1996;39(15):2887–2893. DOI: [10.1021/jm9602928](https://doi.org/10.1021/jm9602928)
6. Butina D. Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets *J. Chem. Inf. Comput. Sci.* 1999;39:747–750. DOI: [10.1021/ci9803381](https://doi.org/10.1021/ci9803381)
7. McInnes L, et al. UMAP. *arXiv:1802.03426.* 2018.
8. Garnett MJ, et al. Systematic identification of genomic markers of drug sensitivity in cancer cells. *Nature.* 2012;483:570–575. DOI: [10.1038/nature11005](https://doi.org/10.1038/nature11005)
9. Iorio F, et al. A Landscape of Pharmacogenomic Interactions in Cancer. *Cell.* 2016;166:740–754. DOI: [10.1016/j.cell.2016.06.017](https://doi.org/10.1016/j.cell.2016.06.017)
10. Pedregosa F, et al. Scikit-learn: Machine Learning in Python. *JMLR.* 2011;12:2825–2830.
