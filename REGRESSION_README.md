# Regression Models Documentation

This document provides detailed technical specifications for the regression models used in the AI-RET project.

---

## Table of Contents

- [Overview](#overview)
- [RET G810R Mutant pIC50 Regression](#ret-g810r-mutant-pic50-regression)
- [Algorithms](#algorithms)
- [Molecular Features](#molecular-features)
- [Data Splitting Strategies](#data-splitting-strategies)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Model Evaluation](#model-evaluation)
- [Usage Instructions](#usage-instructions)

---

## Overview

This project includes a comprehensive regression modeling pipeline for predicting pIC50 values of compounds against the **RET G810R mutant**. The pipeline evaluates multiple algorithms, molecular fingerprints, and data splitting strategies to identify the optimal model configuration.

**Key Goal**: Build a robust QSAR (Quantitative Structure-Activity Relationship) model compatible with REINVENT for scoring generated molecules during reinforcement learning.

---

## RET G810R Mutant pIC50 Regression

### Purpose

Predict the biological activity (pIC50) of compounds against the RET G810R mutant, a clinically relevant resistance mutation in RET-driven cancers.

### Dataset

- **File**: `RET_Mutant_Selected_Dataset_With_Additional_Information.csv`
- **Target Variable**: pIC50 (negative logarithm of IC50)
- **Compounds**: Curated dataset of RET inhibitors with experimental activity data
- **Data Quality**: Standardized SMILES, validated structures

### Comprehensive Evaluation Strategy

The pipeline systematically evaluates:
- **3 Algorithms** × **4 Fingerprint Types** × **4 Split Strategies** = **48 model configurations**

This comprehensive approach ensures:
1. Robust model selection
2. Understanding of model generalization
3. Identification of optimal feature representation
4. Assessment of splitting strategy impact

---

## Algorithms

Three complementary machine learning algorithms are evaluated:

### 1. RandomForest Regressor

**Type**: Ensemble tree-based method

**Characteristics**:
- Non-linear modeling capability
- Handles high-dimensional data well
- Robust to outliers
- Provides feature importance
- No scaling required

**Hyperparameter Search Space**:
```python
{
    'n_estimators': randint(200, 401),      # Number of trees
    'max_depth': randint(10, 21),           # Maximum tree depth
    'min_samples_split': randint(5, 16),    # Min samples to split node
    'min_samples_leaf': randint(2, 7)       # Min samples in leaf node
}
```

**Best For**: Capturing complex non-linear relationships in molecular data

---

### 2. ExtraTrees Regressor

**Type**: Extremely randomized trees ensemble

**Characteristics**:
- Similar to RandomForest but with more randomization
- Faster training than RandomForest
- Often better generalization
- Reduces overfitting through increased randomness
- No scaling required

**Hyperparameter Search Space**:
```python
{
    'n_estimators': randint(200, 401),      # Number of trees
    'max_depth': randint(10, 21),           # Maximum tree depth
    'min_samples_split': randint(5, 16),    # Min samples to split node
    'min_samples_leaf': randint(2, 7)       # Min samples in leaf node
}
```

**Best For**: Improved generalization through extreme randomization

---

### 3. ElasticNet Regressor

**Type**: Linear model with L1 and L2 regularization

**Characteristics**:
- Linear modeling approach
- Feature selection via L1 penalty (Lasso)
- Coefficient shrinkage via L2 penalty (Ridge)
- Requires feature scaling
- Interpretable coefficients

**Hyperparameter Search Space**:
```python
{
    'alpha': uniform(0.001, 0.999),         # Regularization strength
    'l1_ratio': uniform(0.1, 0.8)           # Balance between L1 and L2
}
```

**Scaling**: StandardScaler applied before training

**Best For**: Linear relationships and feature selection

---

## Molecular Features

Four types of molecular fingerprints are evaluated to represent chemical structures:

### 1. ECFP (Extended Connectivity Fingerprints)

**Type**: Binary circular fingerprints

**Parameters**:
- Radius: 3 (equivalent to ECFP6)
- Size: 2048 bits
- Use counts: False (binary)

---

### 2. ECFP_Counts

**Type**: Count-based circular fingerprints

**Parameters**:
- Radius: 3 (equivalent to ECFP6)
- Size: 2048 bits
- Use counts: True

---

### 3. Avalon Fingerprints

**Type**: Structural key fingerprints

**Parameters**:
- Size: 2048 bits

---

### 4. MACCS Keys

**Type**: Predefined structural keys

**Parameters**:
- Size: 167 bits (fixed)

---

## Data Splitting Strategies

Eight different splitting strategies are evaluated to assess model generalization:

### 1. Random Split

**Seeds**: 1234

**Method**: Random 80/20 train/test split

**Purpose**:
- Baseline performance assessment
- Evaluate model stability across random splits
- Standard approach for QSAR modeling

---

### 2. Scaffold Split

**Method**: Murcko scaffold-based splitting

**Algorithm**:
1. Extract Murcko scaffold for each molecule
2. Group molecules by scaffold
3. Split scaffolds into train/test (80/20)
4. Ensures no scaffold appears in both sets

---

### 3. Butina Clustering Split

**Method**: Butina clustering-based splitting

**Algorithm**:
1. Generate ECFP fingerprints (radius=3, 2048 bits)
2. Calculate Tanimoto distance matrix
3. Perform Butina clustering (distance threshold=0.35)
4. Split clusters into train/test sets

---

### 4. UMAP Clustering Split

**Method**: UMAP dimensionality reduction + K-means clustering

**Algorithm**:
1. Generate ECFP fingerprints
2. Apply UMAP dimensionality reduction (n_components=2)
3. Perform K-means clustering
4. Split clusters into train/test sets

---

## Hyperparameter Optimization

### RandomizedSearchCV

**Method**: Randomized search over hyperparameter distributions

**Configuration**:
```python
RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,              # 20 random combinations
    cv=5,                   # 5-fold cross-validation
    scoring='r2',           # R² as optimization metric
    random_state=56789,     # Reproducibility
    n_jobs=-1               # Parallel processing
)
```

**Process**:
1. Define hyperparameter distributions
2. Sample 20 random combinations
3. Evaluate each with 5-fold CV
4. Select best performing combination
5. Retrain on full training set

---

## Model Evaluation

### Evaluation Metrics

Comprehensive metrics are calculated for each model:

| Metric | Description | Formula | Interpretation |
|--------|-------------|---------|----------------|
| **R²** | Coefficient of determination | 1 - (SS_res / SS_tot) | Closer to 1 is better; can be negative |
| **RMSE** | Root mean squared error | √(Σ(y_true - y_pred)² / n) | Lower is better; same units as target |
| **MAE** | Mean absolute error | Σ\|y_true - y_pred\| / n | Lower is better; robust to outliers |
| **Pearson r** | Linear correlation | Covariance / (σ_x × σ_y) | -1 to 1; measures linear relationship |

### Cross-Validation

**10-fold cross-validation** on training set:
- Provides robust performance estimates
- Detects overfitting
- Assesses model stability

**Metrics Reported**:
- CV R² (mean ± std)
- CV RMSE (mean ± std)
- CV MAE (mean ± std)

### Train/Test Evaluation

**Training Set**:
- R², RMSE, MAE, Pearson correlation
- Assesses model fit

**Test Set**:
- R², RMSE, MAE, Pearson correlation, Spearman correlation
- Assesses generalization capability

---

## Usage Instructions

### Training Models

#### Basic Usage

```bash
python train_model_randomized_MUT.py \
    --input RET_Mutant_Selected_Dataset_With_Additional_Information.csv \
    --models_dir ./trained_models \
    --output qsar_results.csv
```

#### Command-Line Arguments

```
--input, -i      Path to input CSV file (required)
                 Must contain 'RDKIT_SMILES' and 'pIC50' columns

--models_dir     Directory for saving trained models (default: ./models)
                 Creates directory if it doesn't exist

--output         Path for results CSV file (default: qsar_results.csv)
                 Contains comprehensive evaluation metrics
```

### Output Files

#### 1. Results CSV

Comprehensive results for all model configurations:

```csv
Split_Type,Fingerprint,Algorithm,Evaluation_Type,RandomSeed,Test_R2,Test_RMSE,Test_MAE,CV_R2_Mean,CV_R2_Std,...
Random_Seed_1234,ECFP_Counts,RandomForest,comparison,1234,0.85,0.42,0.31,0.82,0.05,...
Scaffold,Avalon,ExtraTrees,comparison,None,0.78,0.51,0.38,0.80,0.06,...
...
```

### Making Predictions

```python
import pickle
import numpy as np
from train_model_randomized_MUT import REINVENTDescriptors

# Load saved model
with open('Random_Seed_1234_ECFP_Counts_RandomForest_comparison.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
fp_type = model_data['fingerprint_type']
fp_params = model_data['fingerprint_params']

# Generate fingerprints for new molecules
descriptor_gen = REINVENTDescriptors()
smiles_list = ['CCO', 'c1ccccc1', ...]

if fp_type == 'ECFP_Counts':
    X = descriptor_gen.get_ecfp_fingerprints(smiles_list, **fp_params)
elif fp_type == 'Avalon':
    X = descriptor_gen.get_avalon_fingerprints(smiles_list, **fp_params)
# ... other fingerprint types

# Apply scaling if needed
if scaler is not None:
    X = scaler.transform(X)

# Predict pIC50
predictions = model.predict(X)
print(f"Predicted pIC50 values: {predictions}")
```

---

### Best Model

The final selected model for REINVENT integration:

**File**: `RET_MUTANT_final_model_Hyper.pkl`

This model represents the best-performing configuration based on:
- Test set R²
- Cross-validation performance
- Generalization capability
- Robustness across splits

---
