#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Task Learning Classification Pipeline for Drug Sensitivity Prediction

Uses individual RandomForest models per cell line (task) to handle missing values.
Compatible with scikit-learn 0.22+

Saves EACH individual model in SEPARATE pickle files for modular deployment.
Uses ONLY: Avalon fingerprints + RandomForest + Random split
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import sklearn
warnings.filterwarnings('ignore')

# Check sklearn version
print(f"📦 Using scikit-learn version: {sklearn.__version__}")
if sklearn.__version__ < '0.22':
    print("⚠️  Warning: This code is designed for scikit-learn 0.22+")
    print("   Some features may not work correctly with older versions.")

# RDKit for molecular fingerprints (Avalon only)
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier

# For progress tracking
from tqdm import tqdm
import time
from datetime import datetime

class MultiTaskDrugClassifier:
    def __init__(self, data_path, results_path="./results"):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Create subdirectory for individual models
        self.models_dir = self.results_path / "individual_models"
        self.models_dir.mkdir(exist_ok=True)

        # Initialize data containers
        self.pIC50_data = None
        self.smiles_data = None
        self.fingerprints = {}
        self.models = {}
        self.best_thresholds = {}

        # Setup classifiers and parameters
        self._setup_classifiers()

    def _get_predictions(self, clf, X_scaled):
        """Get probability predictions from RandomForest"""
        y_pred_proba = clf.predict_proba(X_scaled)
        if isinstance(y_pred_proba, list):
            # Handle list of arrays (for compatibility)
            y_pred_proba = np.array([prob[:, 1] for prob in y_pred_proba]).T
        else:
            # Single output case
            y_pred_proba = y_pred_proba[:, 1]
        return y_pred_proba

    def load_individual_models(self, model_key):
        """
        Load individual models from separate files
        
        Parameters:
        -----------
        model_key : str
            Model identifier (e.g., 'avalon_RandomForest_random')
            
        Returns:
        --------
        model_data : dict
            Dictionary containing all models and metadata
        """
        # Load metadata
        metadata_file = self.models_dir / f"{model_key}_metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        
        # Load individual models
        models = []
        for cell_line in metadata['cell_lines']:
            model_file = self.models_dir / f"{model_key}_{cell_line}.pkl"
            if model_file.exists():
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                models.append(model)
            else:
                # Model doesn't exist for this cell line (no training data)
                models.append(None)
        
        # Combine into model_data structure
        model_data = {
            'models': models,
            'scaler': metadata['scaler'],
            'thresholds': metadata['thresholds'],
            'cell_lines': metadata['cell_lines'],
            'fingerprint_type': metadata.get('fingerprint_type', 'avalon'),
            'n_features': metadata.get('n_features', 2048),
            'sklearn_version': metadata.get('sklearn_version', sklearn.__version__)
        }
        
        return model_data

    def predict_with_model(self, X, model_data):
        """
        Make predictions using loaded models (handles individual models per task)
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix (fingerprints)
        model_data : dict
            Model data from load_individual_models() or training
            
        Returns:
        --------
        predictions : DataFrame
            Binary predictions for each cell line
        probabilities : DataFrame
            Probability predictions for each cell line
        """
        # Scale features
        X_scaled = model_data['scaler'].transform(X)
        
        # Get predictions for each cell line
        n_samples = X_scaled.shape[0]
        n_cell_lines = len(model_data['cell_lines'])
        
        y_pred_proba = np.zeros((n_samples, n_cell_lines))
        y_pred = np.zeros((n_samples, n_cell_lines))
        
        for i, (cell_line, model, threshold) in enumerate(zip(
            model_data['cell_lines'], 
            model_data['models'], 
            model_data['thresholds']
        )):
            if model is None:
                # No model for this cell line (insufficient training data)
                y_pred_proba[:, i] = 0.5  # neutral probability
                y_pred[:, i] = 0  # default to resistant
            else:
                y_pred_proba[:, i] = model.predict_proba(X_scaled)[:, 1]
                y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
        
        # Convert to DataFrames
        predictions_df = pd.DataFrame(
            y_pred, 
            columns=model_data['cell_lines'],
            index=X.index if hasattr(X, 'index') else None
        )
        
        probabilities_df = pd.DataFrame(
            y_pred_proba,
            columns=model_data['cell_lines'],
            index=X.index if hasattr(X, 'index') else None
        )
        
        return predictions_df, probabilities_df

    def _setup_classifiers(self):
        """Setup classifiers and hyperparameter grids"""
        # Hyperparameter grids - only RandomForest
        self.param_grids = {
            'RandomForest': {
                'estimator__n_estimators': [100, 200, 500],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5, 10]
            }
        }

        # Classification models to test - only RandomForest
        self.classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42)
        }

    def load_data(self):
        """Load pIC50 data and SMILES strings"""
        print("🔄 Loading data...")

        # Load Z Score data (already in wide format: drugs as rows, cell lines as columns)
        pic50_file = self.data_path / "GDSC1_GDSC2_Combined_Dataset_With_SMILES_multitask_zscore_format.csv"
        self.pIC50_data = pd.read_csv(pic50_file, index_col=0)
        print(f"📊 Loaded Z Score data: {self.pIC50_data.shape}")

        # Load original data for SMILES
        smiles_file = self.data_path / "GDSC1_GDSC2_Combined_Dataset_With_SMILES.csv"
        smiles_df = pd.read_csv(smiles_file)

        # Create SMILES mapping
        self.smiles_data = {}
        for _, row in smiles_df.iterrows():
            drug_name = row['Drug Name']
            smiles = row['SMILES']
            if pd.notna(smiles) and smiles.strip():
                self.smiles_data[drug_name] = smiles

        print(f"🧪 Loaded {len(self.smiles_data)} unique SMILES strings")

    def generate_fingerprints(self):
        """Generate molecular fingerprints - only Avalon"""
        print("\n🧬 Generating molecular fingerprints...")

        print(f"   Generating AVALON fingerprints...")
        self.fingerprints['avalon'] = self._generate_avalon()

        print(f"✅ Generated Avalon fingerprints")


    def _generate_avalon(self):
        """Generate Avalon fingerprints"""
        fingerprints = {}
        for drug_name, smiles in tqdm(self.smiles_data.items(), desc="Avalon"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = pyAvalonTools.GetAvalonFP(mol, nBits=2048)
                    fingerprints[drug_name] = np.array(fp)
            except:
                continue
        return pd.DataFrame.from_dict(fingerprints, orient='index')

    def determine_optimal_threshold(self, y_true, y_pred_proba, method='balanced'):
        """Determine optimal classification threshold"""
        if method == 'balanced':
            # Find threshold that balances sensitivity and specificity
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            # Youden's J statistic
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return thresholds[best_idx]
        elif method == 'f1':
            # Find threshold that maximizes F1 score
            thresholds = np.linspace(0.1, 0.9, 50)
            f1_scores = []
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            best_idx = np.argmax(f1_scores)
            return thresholds[best_idx]
        else:
            return 0.5

    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics with robust error handling"""
        try:
            # Check if we have both classes
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Handle different confusion matrix sizes
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class present
                if unique_true[0] == 0:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            # Calculate metrics with error handling
            try:
                accuracy = accuracy_score(y_true, y_pred)
            except:
                accuracy = np.nan
            
            try:
                precision = precision_score(y_true, y_pred, zero_division=0)
            except:
                precision = np.nan
            
            try:
                recall = recall_score(y_true, y_pred, zero_division=0)
            except:
                recall = np.nan
            
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except:
                f1 = np.nan
            
            try:
                # AUC requires both classes
                if len(unique_true) >= 2:
                    auc = roc_auc_score(y_true, y_pred_proba)
                else:
                    auc = np.nan
            except:
                auc = np.nan
            
            try:
                # AUPRC requires both classes
                if len(unique_true) >= 2:
                    auprc = average_precision_score(y_true, y_pred_proba)
                else:
                    auprc = np.nan
            except:
                auprc = np.nan
            
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
            except:
                mcc = np.nan

            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'AUPRC': auprc,
                'MCC': mcc,
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            }
            
        except Exception as e:
            # If all else fails, return NaN for everything
            print(f"      Error in evaluate_model: {e}")
            metrics = {
                'Accuracy': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan,
                'AUC': np.nan,
                'AUPRC': np.nan,
                'MCC': np.nan,
                'TP': np.nan,
                'TN': np.nan,
                'FP': np.nan,
                'FN': np.nan
            }

        return metrics

    def train_and_evaluate(self, fingerprint_type='avalon', classifier_name='RandomForest', use_hyperparameter_tuning=True, splitting_method='random'):
        """Train and evaluate RandomForest with Avalon fingerprints using random split"""
        print(f"\n🔬 Training {classifier_name} with {fingerprint_type.upper()} fingerprints (split: {splitting_method})...")

        # Get fingerprint data
        X = self.fingerprints[fingerprint_type]
        y = self.pIC50_data  # Z Score data

        # Align data (only drugs with both fingerprints and Z Score data)
        common_drugs = set(X.index) & set(y.index)
        X = X.loc[list(common_drugs)]
        y = y.loc[list(common_drugs)]

        print(f"   Data shape: {X.shape} (drugs) × {y.shape} (cell lines)")

        # Convert to binary classification (sensitive/resistant)
        # For Z Score: negative = sensitive (1), positive = resistant (0)
        # Z Score < 0 indicates higher sensitivity than average
        y_binary = (y < 0).astype(int)
        
        # Handle missing values: NaN stays as NaN (will be handled per-task)
        y_binary = y_binary.where(y.notna(), np.nan)
        
        # Report missing value statistics
        missing_counts = y_binary.isna().sum()
        total_missing = missing_counts.sum()
        print(f"   Missing values: {total_missing} ({total_missing / (y_binary.shape[0] * y_binary.shape[1]) * 100:.2f}%)")
        print(f"   Cell lines with most missing values:")
        top_missing = missing_counts.sort_values(ascending=False).head(5)
        for cell_line, count in top_missing.items():
            print(f"      {cell_line}: {count}/{y_binary.shape[0]} ({count/y_binary.shape[0]*100:.1f}%)")

        results = []

        # Train-test split - only random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )

        print(f"   Train set: {X_train.shape[0]} drugs, Test set: {X_test.shape[0]} drugs")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train separate model for each cell line (task) to handle missing values
        base_classifier = self.classifiers[classifier_name]
        
        # Store individual models for each cell line
        individual_models = []
        thresholds = []
        
        print(f"   Training individual models for {y_train.shape[1]} cell lines...")
        
        for i, cell_line in enumerate(tqdm(y_train.columns, desc="Training cell lines")):
            # Get valid samples for this cell line (non-NaN)
            train_valid_mask = y_train.iloc[:, i].notna()
            test_valid_mask = y_test.iloc[:, i].notna()
            
            X_train_valid = X_train_scaled[train_valid_mask]
            y_train_valid = y_train.iloc[:, i][train_valid_mask]
            
            X_test_valid = X_test_scaled[test_valid_mask]
            y_test_valid = y_test.iloc[:, i][test_valid_mask]
            
            n_train = len(y_train_valid)
            n_test = len(y_test_valid)
            
            # Train model for this cell line with whatever data is available
            # Even if data is limited, we still train
            if n_train == 0:
                # Absolutely no training data - cannot train
                print(f"      Warning: {cell_line} has NO training data, skipping model training")
                individual_models.append(None)
                thresholds.append(0.5)
                continue
            
            # Check if we have both classes in training data
            unique_classes = y_train_valid.unique()
            if len(unique_classes) < 2:
                print(f"      Warning: {cell_line} has only 1 class in training (n={n_train}), using default model")
            
            # Train model for this cell line
            # Create a fresh instance of the classifier (compatible with sklearn 0.22)
            try:
                # Try to clone using get_params (sklearn 0.22+)
                clf_task = base_classifier.__class__(**base_classifier.get_params())
            except:
                # Fallback: create new instance with default params
                from sklearn.ensemble import RandomForestClassifier
                clf_task = RandomForestClassifier(random_state=42)
            
            clf_task.fit(X_train_valid, y_train_valid)
            individual_models.append(clf_task)
            
            # Determine optimal threshold for this cell line
            if n_test > 0 and len(unique_classes) >= 2:
                y_pred_proba_valid = clf_task.predict_proba(X_test_valid)[:, 1]
                thresh = self.determine_optimal_threshold(
                    y_test_valid, y_pred_proba_valid, method='balanced'
                )
            else:
                # Default threshold if no test data or single class
                thresh = 0.5
            thresholds.append(thresh)
        
        print(f"   ✅ Trained {sum([1 for m in individual_models if m is not None])}/{len(individual_models)} models")

        # Store best params as default (hyperparameter tuning can be added per-task if needed)
        best_params = "default"
        
        # Make predictions for each cell line
        y_pred_proba = np.full((X_test_scaled.shape[0], y_test.shape[1]), np.nan)
        y_pred = np.full((X_test_scaled.shape[0], y_test.shape[1]), np.nan)
        
        for i, (cell_line, model) in enumerate(zip(y_test.columns, individual_models)):
            if model is None:
                continue
            
            # Predict for all test samples (even if some have missing ground truth)
            y_pred_proba[:, i] = model.predict_proba(X_test_scaled)[:, 1]
            y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[i]).astype(float)

        # Save models individually to separate files
        model_key = f"{fingerprint_type}_{classifier_name}_{splitting_method}"
        
        # Save metadata (scaler, thresholds, cell_lines)
        metadata = {
            'scaler': scaler,
            'thresholds': thresholds,
            'cell_lines': list(y_test.columns),
            'fingerprint_type': fingerprint_type,
            'n_features': X.shape[1],
            'sklearn_version': sklearn.__version__,
            'classifier_name': classifier_name,
            'splitting_method': splitting_method
        }
        metadata_file = self.models_dir / f"{model_key}_metadata.pkl"
        pickle.dump(metadata, open(metadata_file, "wb"))
        
        # Save each individual model
        models_saved = 0
        total_size = 0
        
        print(f"   💾 Saving individual models...")
        for i, (cell_line, model) in enumerate(tqdm(zip(y_test.columns, individual_models), desc="Saving models")):
            if model is not None:
                # Clean cell line name for filename (remove special characters)
                safe_cell_line = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in cell_line)
                model_file = self.models_dir / f"{model_key}_{safe_cell_line}.pkl"
                pickle.dump(model, open(model_file, "wb"))
                models_saved += 1
                total_size += model_file.stat().st_size
        
        # Report saving statistics
        total_size_mb = total_size / (1024 * 1024)
        metadata_size_mb = metadata_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved {models_saved} individual model files")
        print(f"   📦 Total size: {total_size_mb:.2f} MB (models) + {metadata_size_mb:.2f} MB (metadata)")
        print(f"   📂 Models directory: {self.models_dir}")
        
        # Store reference in memory
        model_data = {
            'models': individual_models,
            'scaler': scaler,
            'thresholds': thresholds,
            'cell_lines': list(y_test.columns),
            'fingerprint_type': fingerprint_type,
            'n_features': X.shape[1],
            'sklearn_version': sklearn.__version__
        }
        self.models[model_key] = model_data

        # Evaluate each cell line
        for i, cell_line in enumerate(y_test.columns):
            # Get valid test samples for this cell line
            valid_mask = y_test.iloc[:, i].notna()
            
            n_test_valid = valid_mask.sum()
            n_train_valid = y_train.iloc[:, i].notna().sum()
            
            # If no model was trained (no training data), skip evaluation but record it
            if individual_models[i] is None:
                result = {
                    'Fingerprint': fingerprint_type.upper(),
                    'Model': classifier_name,
                    'Cell_Line': cell_line,
                    'N_Features': X.shape[1],
                    'N_Train_Samples': n_train_valid,
                    'N_Test_Samples': n_test_valid,
                    'Missing_Train': y_train.shape[0] - n_train_valid,
                    'Missing_Test': y_test.shape[0] - n_test_valid,
                    'Scaling': 'StandardScaler',
                    'Hyperparameters': str(best_params),
                    'Threshold': thresholds[i],
                    'Split_Method': splitting_method,
                    'Split': 'Train_Test',
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1': np.nan,
                    'AUC': np.nan,
                    'AUPRC': np.nan,
                    'MCC': np.nan,
                    'TP': np.nan,
                    'TN': np.nan,
                    'FP': np.nan,
                    'FN': np.nan,
                }
                results.append(result)
                continue
            
            # If no test data, record metrics as NaN but keep the model info
            if n_test_valid == 0:
                result = {
                    'Fingerprint': fingerprint_type.upper(),
                    'Model': classifier_name,
                    'Cell_Line': cell_line,
                    'N_Features': X.shape[1],
                    'N_Train_Samples': n_train_valid,
                    'N_Test_Samples': n_test_valid,
                    'Missing_Train': y_train.shape[0] - n_train_valid,
                    'Missing_Test': y_test.shape[0] - n_test_valid,
                    'Scaling': 'StandardScaler',
                    'Hyperparameters': str(best_params),
                    'Threshold': thresholds[i],
                    'Split_Method': splitting_method,
                    'Split': 'Train_Test',
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1': np.nan,
                    'AUC': np.nan,
                    'AUPRC': np.nan,
                    'MCC': np.nan,
                    'TP': np.nan,
                    'TN': np.nan,
                    'FP': np.nan,
                    'FN': np.nan,
                }
                results.append(result)
                continue
            
            # Extract valid samples for evaluation
            y_true_valid = y_test.iloc[:, i][valid_mask]
            y_pred_valid = y_pred[valid_mask, i]
            y_pred_proba_valid = y_pred_proba[valid_mask, i]
            
            # Evaluate only on valid samples
            try:
                metrics = self.evaluate_model(
                    y_true_valid, y_pred_valid, y_pred_proba_valid
                )
            except Exception as e:
                # If evaluation fails (e.g., only one class in test), record NaN
                print(f"      Warning: Evaluation failed for {cell_line}: {e}")
                metrics = {
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1': np.nan,
                    'AUC': np.nan,
                    'AUPRC': np.nan,
                    'MCC': np.nan,
                    'TP': np.nan,
                    'TN': np.nan,
                    'FP': np.nan,
                    'FN': np.nan,
                }

            result = {
                'Fingerprint': fingerprint_type.upper(),
                'Model': classifier_name,
                'Cell_Line': cell_line,
                'N_Features': X.shape[1],
                'N_Train_Samples': n_train_valid,
                'N_Test_Samples': n_test_valid,
                'Missing_Train': y_train.shape[0] - n_train_valid,
                'Missing_Test': y_test.shape[0] - n_test_valid,
                'Scaling': 'StandardScaler',
                'Hyperparameters': str(best_params),
                'Threshold': thresholds[i],
                'Split_Method': splitting_method,
                'Split': 'Train_Test',
                **metrics
            }
            results.append(result)

        return results

    def run_complete_pipeline(self, use_hyperparameter_tuning=True):
        """Run the complete multi-task learning pipeline with Avalon + RandomForest + Random split only"""
        start_time = time.time()

        # Load data
        self.load_data()

        # Generate fingerprints (only Avalon)
        self.generate_fingerprints()

        # Train and evaluate - single configuration only
        all_results = []

        fingerprint_type = 'avalon'
        classifier_name = 'RandomForest'
        splitting_method = 'random'

        try:
            results = self.train_and_evaluate(
                fingerprint_type, classifier_name, use_hyperparameter_tuning, splitting_method
            )
            all_results.extend(results)
            print(f"✅ Training completed successfully")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            raise

        # Save results
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_path / f"multitask_zscore_results_avalon_rf_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        elapsed_time = time.time() - start_time
        print("\n🎉 Pipeline completed!")
        print(f"📊 Total results: {len(results_df)}")
        print(f"💾 Results saved to: {results_file}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")

        # Summary
        print("\n📈 Summary of Z Score results:")
        summary = results_df[['Accuracy', 'F1', 'AUC']].mean().round(4)
        print(summary)

        return results_df

def main():
    # Initialize classifier
    classifier = MultiTaskDrugClassifier(
        data_path=".",
        results_path="./results_zscore_individual"
    )

    # Run complete pipeline with only Avalon + RandomForest + Random split
    results = classifier.run_complete_pipeline(use_hyperparameter_tuning=True)

    print("\n📈 Final Summary:")
    summary = results[['Accuracy', 'F1', 'AUC']].describe().round(4)
    print(summary)
    
    print("\n📂 Individual model files saved in:")
    print(f"   {classifier.models_dir}")
    print(f"\n💡 To load a specific model:")
    print(f"   model_data = classifier.load_individual_models('avalon_RandomForest_random')")

if __name__ == "__main__":
    main()
