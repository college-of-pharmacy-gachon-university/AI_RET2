# Classification Models Documentation

This document provides detailed technical specifications for the classification models used in the AI-RET project.

---

## Table of Contents

- [Overview](#overview)
- [Multi-Kinase Selectivity Classification](#multi-kinase-selectivity-classification)
- [Phenotypic Response Classification](#phenotypic-response-classification)
- [Usage Instructions](#usage-instructions)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Saved Models](#saved-models)

---

## Overview

This project includes two classification tasks:

1. **Multi-Kinase Selectivity Classification**: Multi-task learning for predicting activity across multiple kinase targets
2. **Phenotypic Response Classification**: Predicting drug sensitivity across cancer cell lines from GDSC datasets

Both models are designed to be compatible with REINVENT for use as scoring functions during molecular generation.

---

## Multi-Kinase Selectivity Classification

### Purpose

Predict whether a compound is active (based on pIC50 threshold) across multiple kinase targets simultaneously using a multi-task learning approach.

### Technical Specifications

| Specification | Details |
|--------------|---------|
| **Algorithm** | RandomForest Classifier |
| **Hyperparameters** | Default parameters (random_state=42) |
| **Molecular Features** | ECFP_Counts (Extended Connectivity Fingerprints with count vectors) |
| **Fingerprint Parameters** | Radius: 3, Size: 2048 bits, Use counts: True |
| **Data Representation** | Sparse matrix (for memory efficiency) |
| **Split Type** | Random split (80% train / 20% test) |
| **Learning Paradigm** | Multi-task classification (single model for all targets) |
| **Threshold Method** | Fixed 0.5 probability threshold |

### Key Features

- **ECFP_Counts Fingerprints**: Unlike binary ECFP, count fingerprints capture the frequency of substructural features, providing richer molecular representation
- **Multi-task Learning**: A single RandomForest model predicts activity across all kinase targets simultaneously, learning shared representations
- **Sparse Matrix Optimization**: Uses scipy sparse matrices to handle high-dimensional fingerprints efficiently
- **Memory-Efficient Scaling**: MaxAbsScaler for sparse data preservation

### Model Architecture

```
Input: SMILES strings
  ↓
ECFP_Counts Generation (radius=3, 2048 bits)
  ↓
Sparse Matrix (n_samples × 2048)
  ↓
MaxAbsScaler (preserves sparsity)
  ↓
RandomForest Multi-task Classifier
  ↓
Output: Binary predictions for each kinase target
```

### Evaluation Metrics

The model is evaluated using:
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **Specificity**: True negative rate
- **MCC**: Matthews Correlation Coefficient
- **Balanced Accuracy**: Average of recall obtained on each class

### File Location

**Code:** `MULTI_KINASE_CLASSIFICATION_CODES_DATA/multitask_classifier_with_default_parameters_SINGLE_MODEL.py`

**Saved Models:** `MULTI_KINASE_CLASSIFICATION_CODES_DATA/saved_model_19Nov/`

---

## Phenotypic Response Classification

### Purpose

Predict drug sensitivity (z-score based binary classification) across multiple cancer cell lines using data from GDSC1 and GDSC2 (Genomics of Drug Sensitivity in Cancer) datasets.

### Technical Specifications

| Specification | Details |
|--------------|---------|
| **Algorithm** | RandomForest Classifier |
| **Hyperparameters** | Optimized via RandomizedSearchCV |
| **Molecular Features** | Avalon Fingerprints |
| **Fingerprint Parameters** | Size: 2048 bits |
| **Split Type** | Random split (80% train / 20% test) |
| **Learning Paradigm** | Individual models per cell line (task) |
| **Missing Value Handling** | Task-specific models handle missing data naturally |

### Hyperparameter Optimization

RandomizedSearchCV is used to find optimal hyperparameters:

```python
{
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}
```

- **Search Strategy**: RandomizedSearchCV with cross-validation
- **CV Folds**: 5-fold cross-validation
- **Scoring Metric**: ROC-AUC

### Key Features

- **Avalon Fingerprints**: Structural fingerprints that capture molecular features relevant to biological activity
- **Individual Task Models**: Each cell line has its own RandomForest model, allowing task-specific feature importance
- **Z-score Normalization**: Drug sensitivity values are z-score normalized for binary classification
- **Robust to Missing Data**: Individual models naturally handle missing values for specific cell lines

### Model Architecture

```
Input: SMILES strings
  ↓
Avalon Fingerprint Generation (2048 bits)
  ↓
Feature Matrix (n_samples × 2048)
  ↓
For each cell line:
  ↓
  RandomForest Classifier (hyperparameter optimized)
  ↓
  Binary prediction (sensitive/resistant)
```

### Data Source

- **GDSC1 & GDSC2**: Combined dataset with drug sensitivity measurements
- **Format**: Multi-task format with z-score normalized IC50 values
- **File**: `GDSC1_GDSC2_Combined_Dataset_With_SMILES_multitask_zscore_format.csv`

### Evaluation Metrics

The model is evaluated using:
- **AUC-ROC**: Primary metric for model selection
- **Average Precision**: Area under precision-recall curve
- **F1 Score**: Balance between precision and recall
- **Accuracy**: Overall classification accuracy
- **MCC**: Matthews Correlation Coefficient
- **Confusion Matrix**: True/False positives and negatives

### File Location

**Code:** `PHENOTYPES_CLASSIFICATION_CODES_DATA/multitask_classifier_zscore_INDIVIDUAL_FILES.py`

**Data:** `PHENOTYPES_CLASSIFICATION_CODES_DATA/GDSC1_GDSC2_Combined_Dataset_With_SMILES.csv`

**Results:** `PHENOTYPES_CLASSIFICATION_CODES_DATA/results_zscore_individual/`

---

## Usage Instructions

### Multi-Kinase Classification

#### Training a New Model

```python
from multitask_classifier_with_default_parameters_SINGLE_MODEL import MultiTaskDrugClassifier

# Initialize classifier
classifier = MultiTaskDrugClassifier(
    data_path='path/to/kinase_data.csv',
    results_path='./results',
    models_path='./models'
)

# Load data and generate fingerprints
classifier.load_data()
classifier.generate_fingerprints()

# Train and evaluate
classifier.run_complete_pipeline()
```

#### Making Predictions with Saved Model

```python
# Load a saved model
model, scaler, metadata = classifier.load_model('model_id')

# Predict on new SMILES
smiles_list = ['CCO', 'c1ccccc1', ...]
predictions = classifier.predict_with_saved_model(
    model_id='model_id',
    smiles_list=smiles_list
)
```

### Phenotypic Classification

#### Training a New Model

```python
from multitask_classifier_zscore_INDIVIDUAL_FILES import MultiTaskDrugClassifier

# Initialize classifier
classifier = MultiTaskDrugClassifier(
    data_path='GDSC1_GDSC2_Combined_Dataset_With_SMILES_multitask_zscore_format.csv',
    results_path='./results'
)

# Run complete pipeline with hyperparameter tuning
classifier.run_complete_pipeline(use_hyperparameter_tuning=True)
```

#### Loading Individual Models

```python
# Load models for a specific configuration
model_data = classifier.load_individual_models('avalon_RandomForest_random')

# Make predictions
predictions, probabilities = classifier.predict_with_model(X, model_data)
```

---

## Model Evaluation Metrics

### Classification Metrics Used

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **AUC-ROC** | Area under ROC curve | 0-1 | Higher is better; 0.5 = random |
| **F1 Score** | Harmonic mean of precision/recall | 0-1 | Higher is better; balances precision/recall |
| **Accuracy** | Correct predictions / Total predictions | 0-1 | Higher is better |
| **Precision** | True positives / (True + False positives) | 0-1 | Higher is better |
| **Recall** | True positives / (True + False negatives) | 0-1 | Higher is better (sensitivity) |
| **Specificity** | True negatives / (True + False positives) | 0-1 | Higher is better |
| **MCC** | Matthews Correlation Coefficient | -1 to 1 | 1 = perfect, 0 = random, -1 = inverse |
| **Balanced Accuracy** | Average of recall per class | 0-1 | Better for imbalanced datasets |

### Threshold Optimization

#### Multi-Kinase Model

Uses a **fixed 0.5 probability threshold** for binary classification:
- Simple and interpretable
- Standard threshold for balanced classification
- Applied uniformly across all kinase targets

#### Phenotypic Model

Uses **Youden's J statistic** to determine optimal classification thresholds per cell line:

```
J = Sensitivity + Specificity - 1 = TPR - FPR
```

This maximizes the difference between true positive rate and false positive rate, providing balanced classification performance tailored to each cell line's data distribution.

---

## Saved Models

### Model File Structure

Both classification models save trained models as pickle files (`.pkl`) for compatibility with REINVENT and other tools.

### Model Metadata

Each saved model includes:

```python
{
    'model': trained_classifier,           # The trained RandomForest model
    'scaler': fitted_scaler,               # Fitted scaler (if used)
    'fingerprint_type': 'ecfp_counts',     # Type of molecular fingerprint
    'fingerprint_params': {...},           # Fingerprint generation parameters
    'target_names': [...],                 # List of prediction targets
    'threshold': 0.5,                      # Classification threshold
    'performance_metrics': {...},          # Evaluation metrics
    'training_date': '2024-XX-XX',        # Model training date
    'n_features': 2048                     # Number of input features
}
```

### Loading Saved Models

Models can be loaded using standard pickle:

```python
import pickle

with open('model_file.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
```

### REINVENT Integration

These models are designed to be used as scoring components in REINVENT:

1. Load the saved model
2. Generate fingerprints for candidate molecules
3. Apply the scaler (if present)
4. Predict activity/sensitivity
5. Use predictions as scoring function

---
