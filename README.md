# AI-RET: Reinforcement Learning-Based RET-Specific Molecular Design

This repository hosts the code, data, and documentation for the project.

## Authors

**Surendra Kumar, Vinay Pogaku, Mi-Hyun Kim***

**Affiliation:**  
Gachon Institute of Pharmaceutical Science and Department of Pharmacy, College of Pharmacy, Gachon University, 191 Hambakmoe-ro, Yeonsu-gu, Incheon, Republic of Korea

**Corresponding author:** kmh0515@gachon.ac.kr

---

## Table of Contents

- [Overview](#overview)
- [Requirements and Installation](#requirements-and-installation)
- [Repository Structure](#repository-structure)
- [Dataset Collection](#dataset-collection)
- [Machine Learning Models](#machine-learning-models)
- [Applicability Domain and Hypothesis Analysis](#applicability-domain-and-hypothesis-analysis)
- [Molecular Generation using Reinforcement Learning](#molecular-generation-using-reinforcement-learning)
- [Post-Processing and Analysis](#post-processing-and-analysis)
- [Contact Information](#contact-information)

---

## Overview

This project implements a comprehensive workflow for RET-specific molecular design using:

1. **Machine Learning Models**: Classification and regression models for predicting kinase selectivity, phenotypic responses, and RET mutant activity
2. **Applicability Domain Analysis**: CLASS-LAG and Consensus AD methods for NSCLC phenotype models, and root-cause diagnostics (H1–H4) explaining model limitations
3. **Reinforcement Learning**: REINVENT v3.2-based molecular generation with single and multi-objective optimization
4. **Cheminformatics Pipeline**: KNIME workflows for data curation and post-processing

The workflow integrates predictive models with generative AI to design novel RET inhibitors with improved selectivity and potency profiles.

---

## Requirements and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI_RET2
```

### 2. Create the Conda Environment

An `environment.yml` file is provided at the root of the repository. It installs all Python dependencies needed to run the machine learning pipelines and AD analyses.

```bash
conda env create -f environment.yml
conda activate ai_ret
```

This installs: Python 3.9, RDKit, scikit-learn, numpy, pandas, scipy, matplotlib, seaborn, joblib, tqdm, umap-learn, xgboost.

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

### 3. Required External Software

1. **REINVENT v3.2**
   - Reinforcement learning framework for molecular generation
   - Installation: Follow instructions at [REINVENT GitHub](https://github.com/MolecularAI/Reinvent)

2. **KNIME Analytics Platform**
   - For data curation and post-processing workflows
   - Download from [KNIME website](https://www.knime.com/)

3. **Schrödinger Software Suite (2020-4 or later)**
   - Requires a valid license
   - Used for molecular modeling and docking studies

4. **OpenEye Scientific Software**
   - Requires a valid license
   - Used for molecular descriptor generation

### 4. Verify Installation

```bash
conda activate ai_ret
python -c "from rdkit import Chem; import sklearn, umap; print('All dependencies OK')"
```

---

## Repository Structure

```
AI_RET2/
├── README.md                                      # This file
├── CLASSIFICATION_README.md                       # Classification models documentation
├── REGRESSION_README.md                           # Regression models documentation
│
├── MULTI_KINASE_CLASSIFICATION_CODES_DATA/                           # 23-target kinase multi-task classification
│   ├── raw_input/
│   │   └── KINASE_Dataset_With_IC50_188217.csv
│   ├── input/                                     # Cleaned input matrices (94,213 compounds × 23 targets)
│   ├── results/                                   # Authoritative classification results
│   └── code/                                      # Transformation and training scripts
│
├── PHENOTYPES_CLASSIFICATION_CODES_DATA/                            # NSCLC drug response phenotype modelling (GDSC)
│   ├── input/
│   │   ├── z0_threshold/                          # Z=0 (primary analysis; 37.6% sensitive)
│   │   ├── z0.5_threshold/                        # Z=−0.5 (19.6% sensitive)
│   │   ├── z0.75_threshold/                       # Z=−0.75 (12.9% sensitive)
│   │   └── z1.0_threshold/                        # Z=−1.0 (~7.8% sensitive)
│   ├── results/
│   │   ├── z0_unweighted/                         # Primary NSCLC result (mean AUC=0.550)
│   │   ├── z0_weighted/
│   │   ├── z0.5_weighted/
│   │   ├── z0.75_weighted/
│   │   └── z1.0_weighted/
│   └── code/                                      # Transformation and training scripts
│
├── PHENOTYPE_APPLICABILITY_DOMAIN/                # AD analysis + H1–H4 root-cause diagnostics
│   ├── 03_APPLICABILITY_DOMAIN/                   # CLASS-LAG and Consensus AD
│   │   ├── code/
│   │   └── results/
│   └── 04_HYPOTHESIS_ANALYSIS_H1_H4/             # H1–H4 NSCLC failure analysis
│       ├── code/
│       └── results/
│
├── RET_MUT_REGRESSION_CODES_DATA/                 # RET G810R mutant regression
│   ├── train_model_randomized_MUT.py
│   ├── RET_Mutant_Selected_Dataset_With_Additional_Information.csv
│   └── RET_MUTANT_final_model_Hyper.pkl
│
├── SINGLE_OBJECTIVE_REINVENT_JSON/                # Single-objective RL configs (JOB01–JOB05)
├── MULTI_OBJECTIVE_REINVENT_JSON/                 # Multi-objective RL configs (JOB01–JOB05)
└── KNIME_WORKFLOW/                                # KNIME post-processing workflows
```

---

## Dataset Collection

To facilitate RET-MUT Model development, we provide a dataset:

**File:** `RET_MUT_REGRESSION_CODES_DATA/RET_Mutant_Selected_Dataset_With_Additional_Information.csv`

---

## Machine Learning Models

This project includes three types of machine learning models:

### 1. Multi-Kinase Selectivity Classification (`MULTI_KINASE_CLASSIFICATION_CODES_DATA/`)

Predicts selectivity across 23 kinase targets using multi-task learning.

- **Input:** 94,213 unique compounds (188,217 raw ChEMBL bioactivity rows); 23 targets
- **Algorithms:** RandomForest, GradientBoosting, LogisticRegression, SVM, KNN, NaiveBayes, MLP
- **Features:** ECFP, ECFP_COUNT, MACCS, Avalon fingerprints
- **Split types:** Random, Scaffold, Butina, UMAP-cluster
- **Key result:** random split mean AUC = 0.8927

**Run:**

```bash
conda activate ai_ret

# Data transformation:
python MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/transform_kinase_multitask.py \
    MULTI_KINASE_CLASSIFICATION_CODES_DATA/raw_input/KINASE_Dataset_With_IC50_188217.csv \
    --output_prefix kinase_multitask_14May_V15_188217_Original \
    --dedup_mode smiles

# Multi-task classifier training:
python MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/multitask_classifier_with_default_parameters.py --max_workers 32

# Youden threshold analysis:
python MULTI_KINASE_CLASSIFICATION_CODES_DATA/code/run_youden_analysis.py
```

📖 **See [CLASSIFICATION_README.md](CLASSIFICATION_README.md) for detailed documentation**

---

### 2. NSCLC Phenotypic Response Classification (`PHENOTYPES_CLASSIFICATION_CODES_DATA/`)

Predicts drug sensitivity across 108 NSCLC cell lines from GDSC1/GDSC2 datasets.

- **Input:** 429 unique NSCLC drugs × 108 cell lines (41,352 drug–cell-line pairs)
- **Algorithms:** RandomForest, SVM, ExtraTrees, LogisticRegression, KNeighbors, GradientBoosting, GaussianNB
- **Features:** ECFP, ECFP_COUNT, MACCS, Avalon fingerprints
- **Split types:** Random, Scaffold, Butina, UMAP-cluster
- **Key result:** Primary (Z=0, unweighted) mean AUC = 0.550 — near-random; see `PHENOTYPE_APPLICABILITY_DOMAIN/` for root-cause analysis

**Run (Z=0, primary analysis):**

```bash
conda activate ai_ret

# Data transformation:
python PHENOTYPES_CLASSIFICATION_CODES_DATA/code/transform_for_multitask_zscore.py \
    GDSC1_GDSC2_Combined_Dataset_With_SMILES.csv \
    -o gdsc_multitask_threshold_zeo --threshold 0.0

# Unweighted training (primary result):
python PHENOTYPES_CLASSIFICATION_CODES_DATA/code/multitask_classifier_unweighted.py \
    --data_folder gdsc_multitask_threshold_zeo_output \
    --output_prefix gdsc_multitask_threshold_zeo \
    --z_threshold 0.0 --no_save_models \
    --results_path PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted

# Weighted training:
python PHENOTYPES_CLASSIFICATION_CODES_DATA/code/multitask_classifier_weighted.py \
    --data_folder gdsc_multitask_threshold_zeo_output \
    --output_prefix gdsc_multitask_threshold_zeo \
    --z_threshold 0.0 --imbalance_strategy sample_weight \
    --no_save_models --max_workers 32 \
    --results_path PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_weighted
```

📖 **See [CLASSIFICATION_README.md](CLASSIFICATION_README.md) for detailed documentation**

---

### 3. RET G810R Mutant Activity Regression (`RET_MUT_REGRESSION_CODES_DATA/`)

Predicts pIC50 values for the RET G810R mutant using comprehensive model evaluation.

- **Algorithms:** RandomForest, ExtraTrees, ElasticNet
- **Features:** ECFP, ECFP_Counts, Avalon, MACCS
- **Split types:** Random (multiple seeds), Scaffold, Butina, UMAP_Clustering

📖 **See [REGRESSION_README.md](REGRESSION_README.md) for detailed documentation**

---

## Applicability Domain and Hypothesis Analysis

This section contains two related analyses that characterise the reliability of the NSCLC phenotype models and diagnose their near-random performance.

### 03_APPLICABILITY_DOMAIN — CLASS-LAG and Consensus AD

Evaluates whether NSCLC test compounds fall within the applicability domain of the trained models using two complementary methods:

- **CLASS-LAG AD**: Lazy-learning AD — compares a test compound to its nearest training neighbours in chemical space and checks label consistency.
- **Consensus AD**: Combines Tanimoto structural similarity with response concordance to define the AD boundary.

Analyses are performed across all four Z-score thresholds (Z=0, −0.5, −0.75, −1.0) and both weighting schemes (weighted, unweighted).

**Run:**

```bash
conda activate ai_ret
python PHENOTYPE_APPLICABILITY_DOMAIN/03_APPLICABILITY_DOMAIN/code/run_all_AD.py
```

### 04_HYPOTHESIS_ANALYSIS_H1_H4 — Root-Cause Analysis

Tests four hypotheses to explain the near-random NSCLC AUC (≈ 0.550):

| Hypothesis | Conclusion |
|---|---|
| H1 — Wrong algorithm | All algorithms fail similarly; not the cause. |
| H2 — Data quality / class imbalance | Imbalance is moderate; not the primary cause. |
| H3 — Threshold / label noise | Poor performance across all Z thresholds; not the cause. |
| H4 — Feature insufficiency | 2D fingerprints cannot encode cell-line-specific context; **primary cause.** |

**Run:**

```bash
conda activate ai_ret
python PHENOTYPE_APPLICABILITY_DOMAIN/04_HYPOTHESIS_ANALYSIS_H1_H4/code/analyze_nsclc_failure.py --tag z0
```

📖 **See [PHENOTYPE_APPLICABILITY_DOMAIN/README.md](PHENOTYPE_APPLICABILITY_DOMAIN/README.md) for detailed documentation**

---

## Molecular Generation using Reinforcement Learning

This repository includes multiple JSON configuration files for executing reinforcement learning jobs using **REINVENT v3.2**.

### Single-Objective Optimization

Located in `SINGLE_OBJECTIVE_REINVENT_JSON/`

- **JOB01–JOB05**: Different optimization strategies
- Each job focuses on a single objective (e.g., RET potency, selectivity)

### Multi-Objective Optimization

Located in `MULTI_OBJECTIVE_REINVENT_JSON/`

- **JOB01–JOB05**: Simultaneous optimization of multiple objectives
- Balances potency, selectivity, and drug-like properties

### Running REINVENT Jobs

1. Install REINVENT v3.2 following the [official instructions](https://github.com/MolecularAI/Reinvent)

2. Navigate to a job directory:
   ```bash
   cd SINGLE_OBJECTIVE_REINVENT_JSON/JOB01
   ```

3. Run the job:
   ```bash
   python input.py RL_config.json
   ```

> **Important:** Before running any job, update the file paths in the JSON configuration files for reading input and writing output.

### Inception Files

Inception files containing known RET-specific ligands are included to guide the reinforcement learning model during training. The SMILES of these RET-specific ligands are already used in the REINVENT configuration files.

---

## Post-Processing and Analysis

Two comprehensive KNIME workflows are provided for post-processing generated molecules under Single-Objective and Multi-Objective optimization:

**Files:**
- `KNIME_WORKFLOW/SINGLE_OBJECTIVE_MULTI_STAGE_FILTER.knwf`
- `KNIME_WORKFLOW/MULTI_OBJECTIVE_MULTI_STAGE_FILTER.knwf`

These workflows implement:
- Molecular filtering and quality checks
- Property calculations
- Diversity analysis
- Visualization of results

**Note:** Due to confidentiality, the SMILES structures of all de novo generated molecules are not publicly disclosed or included in this repository. The workflows demonstrate the exact procedure used in the study.

---

## Contact Information

For any queries, please contact:

- **Mi-Hyun Kim** (Principal Investigator): kmh0515@gachon.ac.kr
- **Surendra Kumar**: surendramph@gmail.com

---

## Acknowledgments

This work was supported by [funding information to be added].
