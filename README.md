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
- [Molecular Generation using Reinforcement Learning](#molecular-generation-using-reinforcement-learning)
- [Post-Processing and Analysis](#post-processing-and-analysis)
- [Contact Information](#contact-information)

---

## Overview

This project implements a comprehensive workflow for RET-specific molecular design using:

1. **Machine Learning Models**: Classification and regression models for predicting kinase selectivity, phenotypic responses, and RET mutant activity
2. **Reinforcement Learning**: REINVENT v3.2-based molecular generation with single and multi-objective optimization
3. **Cheminformatics Pipeline**: KNIME workflows for data curation and post-processing

The workflow integrates predictive models with generative AI to design novel RET inhibitors with improved selectivity and potency profiles.

---

## Requirements and Installation

This project requires several external tools and software modules. Please ensure the following are installed on your system:

### Required Software

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

### Python Dependencies

The machine learning models require:
- Python 3.7+
- RDKit
- scikit-learn
- pandas, numpy
- scipy
- UMAP (optional, for clustering-based splits)

See individual model directories for specific requirements.

---

## Repository Structure

```
AI_RET2
├── README.md                                                         # This file
├── CLASSIFICATION_README.md                                          # Classification models documentation
├── REGRESSION_README.md                                              # Regression models documentation
│
├── MULTI_KINASE_CLASSIFICATION_CODES_DATA/                           # Multi-kinase selectivity classification
│   ├── multitask_classifier_with_default_parameters_SINGLE_MODEL.py  # code for training/validation/testing
│   ├── kinase_ml_output/                                             # Multi-Kinase data for training/validation/testing
│   └── saved_model_19Nov.zip/                                        # Trained models
│
├── PHENOTYPES_CLASSIFICATION_CODES_DATA/                             # Phenotypic classification
│   ├── multitask_classifier_zscore_INDIVIDUAL_FILES.py               # code for training/validation/testing
│   ├── GDSC1_GDSC2_Combined_Dataset_With_SMILES.csv                  # Phenotype data for training/validation/testing
│   └── results_zscore_individual.zip/                                # Model outputs
│
├── RET_MUT_REGRESSION_CODES_DATA/                                    # RET G810R mutant regression
│   ├── train_model_randomized_MUT.py                                 # code for training/validation/testing
│   ├── RET_Mutant_Selected_Dataset_With_Additional_Information.csv   # RET_Mut data for training/validation/testing
│   └── RET_MUTANT_final_model_Hyper.pkl                              # Best trained model
│
├── SINGLE_OBJECTIVE_REINVENT_JSON/                                   # Single-objective RL configs
│   ├── JOB01/ through JOB05/
│   └── INCEPTIONS_USED_IN_SINGLE_OBJECTIVE.xlsx
│
├── MULTI_OBJECTIVE_REINVENT_JSON/                                   # Multi-objective RL configs
│   ├── JOB01/ through JOB05/
│   └── INCEPTIONS_USED_IN_SINGLE_OBJECTIVE.xlsx
│
├── KNIME_WORKFLOW/                                                  # KNIME Workflow
    └──  README.md
```

---

## Dataset

To facilitate RET-MUT Model development, we provide a dataset:

**File:** `RET_MUT_REGRESSION_CODES_DATA/RET_Mutant_Selected_Dataset_With_Additional_Information.csv`

---

## Machine Learning Models

This project includes three types of machine learning models:

### 1. Multi-Kinase Selectivity Classification

Predicts selectivity across multiple kinase targets using multi-task learning.

- **Algorithms:** RandomForest, SVM, KNeighbors, ExtraTrees, GradientBoosting, LogisticRegression, GaussianNB
- **Features:** ECFP_Counts (Extended Connectivity Fingerprints with count vectors), ECFP, MACCS, Avalon fingerprints
- **Split Types:** Random, cross_validation, scaffold, butina, umap_clustering split

📖 **See [CLASSIFICATION_README.md](CLASSIFICATION_README.md) for detailed documentation**

### 2. Phenotypic Response Classification

Predicts drug sensitivity across multiple cell lines from GDSC1/GDSC2 datasets.

- **Algorithms:** RandomForest, SVM, LogisticRegression, XGBoost
- **Features:** ECFP, MACCS, Avalon fingerprints
- **Split Types:** Random, scaffold, butina, umap_clustering split

📖 **See [CLASSIFICATION_README.md](CLASSIFICATION_README.md) for detailed documentation**

### 3. RET G810R Mutant Activity Regression

Predicts pIC50 values for the RET G810R mutant using comprehensive model evaluation.

- **Algorithms:** RandomForest, ExtraTrees, ElasticNet
- **Features:** ECFP, ECFP_Counts, Avalon, MACCS
- **Split Types:** Random (multiple seeds), Scaffold, Butina, UMAP_Clustering

📖 **See [REGRESSION_README.md](REGRESSION_README.md) for detailed documentation**

---

## Molecular Generation using Reinforcement Learning

This repository includes multiple JSON configuration files for executing reinforcement learning jobs using **REINVENT v3.2**.

### Single-Objective Optimization

Located in `SINGLE_OBJECTIVE_REINVENT_JSON/`

- **JOB01-JOB05**: Different optimization strategies
- Each job focuses on a single objective (e.g., RET potency, selectivity)

### Multi-Objective Optimization

Located in `MULTI_OBJECTIVE_REINVENT_JSON/`

- **JOB01-JOB05**: Simultaneous optimization of multiple objectives
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

⚠️ **Important:** Before running any job, update the file paths in the JSON configuration files for reading input and writing output.

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
---

## Acknowledgments

This work was supported by [funding information to be added].
