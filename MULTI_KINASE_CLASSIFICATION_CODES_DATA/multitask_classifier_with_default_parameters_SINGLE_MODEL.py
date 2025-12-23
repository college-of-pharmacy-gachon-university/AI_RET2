#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Task Learning Classification Pipeline for Drug Sensitivity Prediction
Using molecular fingerprints (Avalon, ECFP, ECFP_Count, MACCS) with pIC50 targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# RDKit for molecular fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem

# Machine Learning
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# For sparse matrices and memory efficiency
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler  # Better for sparse data

# For progress tracking
from tqdm import tqdm
import time
from datetime import datetime

class MultiTaskDrugClassifier:
    def __init__(self, data_path, results_path="./results", models_path="./models"):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.models_path = Path(models_path)
        self.results_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)

        # Initialize data containers
        self.pIC50_data = None
        self.smiles_data = None
        self.fingerprints = {}
        self.models = {}
        self.best_thresholds = {}

        # Results file for progressive saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.results_path / f"multitask_classification_results_default_parameters_{timestamp}.csv"
        self.results_buffer = []  # Buffer to store results before saving

        # Setup classifiers (default parameters only) - ONLY RandomForest
        self.classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42)
        }
    
    def save_results_progressively(self, results, force_save=False):
        """
        Save results to file progressively to avoid memory issues
        
        Args:
            results: List of result dictionaries to save
            force_save: If True, save immediately regardless of buffer size
        """
        self.results_buffer.extend(results)
        
        # Save every 1000 results or when forced
        if len(self.results_buffer) >= 1000 or force_save:
            results_df = pd.DataFrame(self.results_buffer)
            
            # If file exists, append; otherwise create new
            if self.results_file.exists():
                results_df.to_csv(self.results_file, mode='a', header=False, index=False)
            else:
                results_df.to_csv(self.results_file, index=False)
            
            print(f"   💾 Saved {len(self.results_buffer)} results to {self.results_file}")
            self.results_buffer = []  # Clear buffer
    
    def _get_predictions(self, clf, X_scaled):
        """Get probability predictions, handling different classifier types"""
        try:
            # Try predict_proba first
            if hasattr(clf, 'predict_proba'):
                y_pred_proba = clf.predict_proba(X_scaled)
                if isinstance(y_pred_proba, list):
                    # MultiOutputClassifier returns list of arrays
                    y_pred_proba = np.array([prob[:, 1] for prob in y_pred_proba]).T
                else:
                    # Single output case
                    y_pred_proba = y_pred_proba[:, 1]
                return y_pred_proba
            else:
                raise AttributeError("No predict_proba method")
        except (AttributeError, IndexError):
            # Fallback to decision_function for SVM
            if hasattr(clf, 'decision_function'):
                decision_scores = clf.decision_function(X_scaled)
                if isinstance(decision_scores, list):
                    # MultiOutputClassifier with decision_function
                    decision_scores = np.array(decision_scores).T
                # Convert decision scores to probabilities using sigmoid
                y_pred_proba = 1 / (1 + np.exp(-decision_scores))
                return y_pred_proba
            else:
                # Last resort: use predict and assume binary classification
                y_pred = clf.predict(X_scaled)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                # Create dummy probabilities (0.5 for predicted class)
                y_pred_proba = np.zeros_like(y_pred, dtype=float)
                y_pred_proba[y_pred == 1] = 0.5
                y_pred_proba[y_pred == 0] = 0.5
                return y_pred_proba

    def _evaluate_target_model(self, clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target, 
                              fingerprint_type, classifier_name, target, pic50_threshold, split_method):
        """Helper method to evaluate a single target model and return results"""
        # Get predictions
        y_pred_proba_train = self._get_predictions_single(clf, X_train_scaled)
        y_pred_proba_test = self._get_predictions_single(clf, X_test_scaled)
        
        # Use standard probability threshold for classification (0.5)
        probability_threshold = 0.5
        
        # Apply probability threshold to model predictions
        y_pred_train = (y_pred_proba_train >= probability_threshold).astype(int)
        y_pred_test = (y_pred_proba_test >= probability_threshold).astype(int)
        
        # Evaluate on test set
        metrics = self.evaluate_model(y_test_target, y_pred_test, y_pred_proba_test)
        
        result = {
            'Fingerprint': fingerprint_type.upper(),
            'Model': classifier_name,
            'Target': target,
            'N_Features': X_train_scaled.shape[1],
            'Scaling': 'MaxAbsScaler',
            'Hyperparameters': 'default',
            'Threshold': pic50_threshold,
            'Split_Method': split_method,
            'Split': 'Train_Test',
            **metrics
        }
        
        # Also evaluate on training set for completeness
        metrics_train = self.evaluate_model(y_train_target, y_pred_train, y_pred_proba_train)
        
        result_train = {
            'Fingerprint': fingerprint_type.upper(),
            'Model': classifier_name,
            'Target': target,
            'N_Features': X_train_scaled.shape[1],
            'Scaling': 'MaxAbsScaler',
            'Hyperparameters': 'default',
            'Threshold': pic50_threshold,
            'Split_Method': split_method,
            'Split': 'Train_Train',
            **metrics_train
        }
        
        return [result, result_train]

    def _get_predictions_single(self, clf, X_scaled):
        """Get probability predictions for single-output classifier"""
        try:
            # Try predict_proba first
            if hasattr(clf, 'predict_proba'):
                y_pred_proba = clf.predict_proba(X_scaled)
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # Get positive class probabilities
                return y_pred_proba
            else:
                raise AttributeError("No predict_proba method")
        except (AttributeError, IndexError):
            # Fallback to decision_function for SVM
            if hasattr(clf, 'decision_function'):
                decision_scores = clf.decision_function(X_scaled)
                # Convert decision scores to probabilities using sigmoid
                y_pred_proba = 1 / (1 + np.exp(-decision_scores))
                return y_pred_proba
            else:
                # Last resort: use predict and assume binary classification
                y_pred = clf.predict(X_scaled)
                # Create dummy probabilities (0.5 for predicted class)
                y_pred_proba = np.where(y_pred == 1, 0.5, 0.5)
                return y_pred_proba

    def save_model(self, model, scaler, metadata, fingerprint_type, classifier_name, split_method, target_name=None):
        """
        Save trained model with metadata for future predictions
        Uses standard pickle format (.pkl) for compatibility with RL tools
        
        Args:
            model: Trained classifier
            scaler: Fitted StandardScaler
            metadata: Dict with model information (thresholds, features, etc.)
            fingerprint_type: Type of fingerprint used
            classifier_name: Name of the classifier
            split_method: Splitting method used
            target_name: Specific target name (optional, for single-target models)
        """
        # Sanitize target name for filesystem (remove invalid characters)
        def sanitize_filename(name):
            """Replace invalid filesystem characters with underscores"""
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            sanitized = name
            for char in invalid_chars:
                sanitized = sanitized.replace(char, '_')
            return sanitized
        
        # Create model ID
        model_id = f"{fingerprint_type}_{classifier_name}_{split_method}"
        if target_name:
            sanitized_target = sanitize_filename(target_name)
            model_id += f"_{sanitized_target}"
            
        # Create subdirectory for this model
        model_dir = self.models_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using standard pickle (for RL compatibility)
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler using standard pickle
        scaler_file = model_dir / "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"   💾 Saved model: {model_id} (pickle format for RL compatibility)")
        return model_id
    
    def load_model(self, model_id):
        """
        Load a saved model with its scaler and metadata
        Now uses standard pickle format (.pkl) for RL compatibility
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            tuple: (model, scaler, metadata)
        """
        model_dir = self.models_path / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id} not found in {self.models_path}")
            
        # Load model (try .pkl first for new format, fallback to .joblib for old format)
        model_file_pkl = model_dir / "model.pkl"
        model_file_joblib = model_dir / "model.joblib"
        
        if model_file_pkl.exists():
            with open(model_file_pkl, 'rb') as f:
                model = pickle.load(f)
        elif model_file_joblib.exists():
            print(f"   ⚠️  Loading old joblib format, consider re-training for RL compatibility")
            model = joblib.load(model_file_joblib)
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
        
        # Load scaler (try .pkl first, fallback to .joblib)
        scaler_file_pkl = model_dir / "scaler.pkl"
        scaler_file_joblib = model_dir / "scaler.joblib"
        
        if scaler_file_pkl.exists():
            with open(scaler_file_pkl, 'rb') as f:
                scaler = pickle.load(f)
        elif scaler_file_joblib.exists():
            scaler = joblib.load(scaler_file_joblib)
        else:
            raise FileNotFoundError(f"No scaler file found in {model_dir}")
        
        # Load metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        return model, scaler, metadata
    
    def list_saved_models(self):
        """List all saved models with their metadata"""
        models_info = []
        
        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir():
                try:
                    metadata_file = model_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        models_info.append({
                            'model_id': model_dir.name,
                            'fingerprint': metadata.get('fingerprint_type', 'unknown'),
                            'classifier': metadata.get('classifier_name', 'unknown'),
                            'split_method': metadata.get('split_method', 'unknown'),
                            'n_targets': metadata.get('n_targets', 'unknown'),
                            'n_features': metadata.get('n_features', 'unknown'),
                            'avg_auc': metadata.get('avg_auc', 'unknown'),
                            'avg_f1': metadata.get('avg_f1', 'unknown'),
                            'timestamp': metadata.get('timestamp', 'unknown')
                        })
                except Exception as e:
                    print(f"Warning: Could not read metadata for {model_dir.name}: {e}")
                    
        return pd.DataFrame(models_info)
    
    def predict_with_saved_model(self, model_id, smiles_list, target_names=None):
        """
        Make predictions using a saved model
        
        Args:
            model_id: ID of the saved model
            smiles_list: List of SMILES strings to predict
            target_names: List of target names (if None, will predict all targets the model was trained on)
            
        Returns:
            DataFrame with predictions
        """
        # Load model
        model, scaler, metadata = self.load_model(model_id)
        
        print(f"🔮 Making predictions with model: {model_id}")
        print(f"   Fingerprint: {metadata['fingerprint_type']}")
        print(f"   Classifier: {metadata['classifier_name']}")
        print(f"   Trained targets: {len(metadata.get('target_names', []))}")
        
        # Generate fingerprints for input SMILES
        fingerprint_type = metadata['fingerprint_type'].lower()
        
        # Create temporary fingerprint data
        temp_smiles_data = {f"compound_{i}": smiles for i, smiles in enumerate(smiles_list)}
        
        # Only ECFP_COUNT is supported
        if fingerprint_type == 'ecfp_counts':
            fingerprints_df = self._generate_ecfp_count_from_smiles(temp_smiles_data)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}. Only 'ecfp_counts' is supported.")
        
        # Scale features
        X_scaled = scaler.transform(fingerprints_df)
        
        # Make predictions
        y_pred_proba = self._get_predictions(model, X_scaled)
        
        # Apply thresholds
        thresholds = metadata.get('thresholds', [0.5] * y_pred_proba.shape[1])
        y_pred_binary = (y_pred_proba >= np.array(thresholds)).astype(int)
        
        # Create results DataFrame
        target_names_used = metadata.get('target_names', [f"Target_{i}" for i in range(y_pred_proba.shape[1])])
        
        results = pd.DataFrame(index=[f"compound_{i}" for i in range(len(smiles_list))])
        results['SMILES'] = smiles_list
        
        # Add probability predictions
        for i, target in enumerate(target_names_used):
            results[f"{target}_probability"] = y_pred_proba[:, i]
            results[f"{target}_prediction"] = y_pred_binary[:, i]
            results[f"{target}_threshold"] = thresholds[i]
        
        return results
    
    def _generate_ecfp_count_from_smiles(self, smiles_dict):
        """Helper method to generate ECFP Count fingerprints from SMILES as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in smiles_dict.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
                else:
                    fingerprints.append([0] * 2048)
                    valid_drugs.append(drug_name)
            except:
                fingerprints.append([0] * 2048)
                valid_drugs.append(drug_name)
        
        # Convert to sparse matrix
        dense_array = np.array(fingerprints, dtype=np.int8)
        sparse_matrix = sparse.csr_matrix(dense_array)
        return sparse_matrix, valid_drugs

    def find_and_save_best_models(self, results_df, top_k=5, metric='AUC'):
        """
        Identify and save the best performing models based on specified metric
        
        Args:
            results_df: DataFrame with model results
            top_k: Number of top models to save
            metric: Metric to use for ranking ('AUC', 'F1', 'Accuracy', etc.)
            
        Returns:
            DataFrame with information about saved best models
        """
        print(f"\n🏆 Finding top {top_k} models based on {metric}...")
        
        # Filter to only train-test results (not cross-validation folds)
        train_test_results = results_df[results_df['Split'] == 'Train_Test'].copy()
        
        if train_test_results.empty:
            print("❌ No train-test results found")
            return pd.DataFrame()
        
        # Calculate average metric per model configuration
        model_avg_performance = train_test_results.groupby([
            'Fingerprint', 'Model', 'Split_Method'
        ])[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Sort by mean performance descending
        model_avg_performance = model_avg_performance.sort_values('mean', ascending=False)
        
        print("📊 Top model configurations:")
        print(model_avg_performance.head(top_k))
        
        # Create summary of best models with their IDs
        best_models_info = []
        for idx, row in model_avg_performance.head(top_k).iterrows():
            fingerprint = row['Fingerprint']
            classifier = row['Model'] 
            split_method = row['Split_Method']
            avg_metric = row['mean']
            std_metric = row['std']
            
            model_id = f"{fingerprint}_{classifier}_{split_method}"
            
            best_models_info.append({
                'rank': len(best_models_info) + 1,
                'model_id': model_id,
                'fingerprint': fingerprint,
                'classifier': classifier,
                'split_method': split_method,
                f'avg_{metric.lower()}': avg_metric,
                f'std_{metric.lower()}': std_metric,
                'n_targets': row['count']
            })
        
        best_models_df = pd.DataFrame(best_models_info)
        
        # Save best models summary
        best_models_file = self.results_path / "best_models_summary.csv"
        best_models_df.to_csv(best_models_file, index=False)
        print(f"💾 Saved best models summary to: {best_models_file}")
        
        return best_models_df

    def load_data(self):
        """Load transformed kinase data"""
        print("🔄 Loading transformed kinase dataset...")

        # Load the transformed pIC50 matrix
        pic50_file = self.data_path / "kinase_ml_output" / "kinase_ml_pIC50_matrix.csv"
        if not pic50_file.exists():
            raise FileNotFoundError(f"Transformed pIC50 data not found at {pic50_file}. Please run transform_kinase_multitask.py first.")
        
        self.pIC50_data = pd.read_csv(pic50_file, index_col=0)
        
        # Load SMILES mapping
        smiles_file = self.data_path / "kinase_ml_output" / "kinase_ml_smiles_mapping.csv"
        if smiles_file.exists():
            smiles_df = pd.read_csv(smiles_file)
            self.smiles_data = dict(zip(smiles_df['Compound_ID'], smiles_df['SMILES']))
        else:
            print("   Warning: SMILES mapping not found. Some splitting methods may not work.")
            self.smiles_data = {}

        # Load full dataset - no sampling
        # self.pIC50_data = self.pIC50_data.sample(n=min(500, len(self.pIC50_data)), random_state=42)
        self.smiles_data = {k: v for k, v in self.smiles_data.items() if k in self.pIC50_data.index}

        print(f"📊 Loaded pIC50 data: {self.pIC50_data.shape}")
        print(f"💊 SMILES data: {len(self.smiles_data)} compounds")
        print(f"🎯 Targets: {list(self.pIC50_data.columns[:5])}{'...' if len(self.pIC50_data.columns) > 5 else ''}")
        
        # Display sample statistics
        print(f"📈 pIC50 range: {self.pIC50_data.min().min():.3f} to {self.pIC50_data.max().max():.3f}")
        print(f"📊 Data completeness: {(1 - self.pIC50_data.isna().sum().sum() / (self.pIC50_data.shape[0] * self.pIC50_data.shape[1])) * 100:.1f}%")

    def generate_fingerprints(self):
        """Generate molecular fingerprints - ONLY ECFP_COUNT"""
        print("\n🧬 Generating molecular fingerprints...")

        # Only generate ECFP_COUNT fingerprints
        print(f"   Generating ECFP_COUNT fingerprints...")
        sparse_matrix, valid_drugs = self._generate_ecfp_count()
        print(f"   Generated ECFP_COUNT fingerprints for {len(valid_drugs)} molecules")
        self.fingerprints['ecfp_counts'] = (sparse_matrix, valid_drugs)

        print(f"✅ Generated ECFP_COUNT fingerprints")
        print(f"   ECFP_COUNT: {sparse_matrix.shape} (sparse), {len(valid_drugs)} molecules")

    def _generate_ecfp_count(self):
        """Generate ECFP Count fingerprints as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in tqdm(self.smiles_data.items(), desc="ECFP Count"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
            except:
                continue
        
        # Convert to sparse matrix
        dense_array = np.array(fingerprints, dtype=np.int8)  # Use int8 to save memory
        sparse_matrix = sparse.csr_matrix(dense_array)
        print(f"   ECFP Count sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz} non-zero elements ({sparse_matrix.nnz/sparse_matrix.size*100:.1f}% density)")
        return sparse_matrix, valid_drugs

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
        """Calculate all evaluation metrics"""
        try:
            # Convert to numpy arrays if they're pandas series
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            if hasattr(y_pred_proba, 'values'):
                y_pred_proba = y_pred_proba.values
                
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (1, 1):
                # Only one class present in y_true
                if y_true[0] == 0:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                tn, fp, fn, tp = cm.ravel()
        except ValueError:
            # Fallback for edge cases
            tn, fp, fn, tp = 0, 0, 0, len(y_true)

        # Try with zero_division parameter, fall back if not supported
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except TypeError:
            # Older sklearn version doesn't support zero_division
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            # Replace nan with 0
            precision = 0.0 if np.isnan(precision) else precision
            recall = 0.0 if np.isnan(recall) else recall
            f1 = 0.0 if np.isnan(f1) else f1

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': roc_auc_score(y_true, y_pred_proba) if len(set(y_true)) > 1 else 0.5,
            'AUPRC': average_precision_score(y_true, y_pred_proba) if len(set(y_true)) > 1 else 0.5,
            'MCC': matthews_corrcoef(y_true, y_pred),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }

        return metrics

    def train_and_evaluate(self, fingerprint_type, classifier_name, split_method='random'):
        """Train and evaluate a single model configuration with specified splitting method using only default parameters"""
        print(f"\n🔬 Training {classifier_name} with {fingerprint_type.upper()} fingerprints using {split_method} splitting...")

        try:
            # Get fingerprint data (sparse matrix and valid drugs)
            sparse_matrix, valid_drugs = self.fingerprints[fingerprint_type]
            y = self.pIC50_data

            print(f"   Fingerprint sparse matrix shape: {sparse_matrix.shape}")
            print(f"   Valid drugs: {len(valid_drugs)}")
            print(f"   pIC50 data shape: {y.shape}")

            # Align data (only drugs with both fingerprints and pIC50 data)
            common_drugs = set(valid_drugs) & set(y.index)
            print(f"   Common drugs: {len(common_drugs)}")

            if len(common_drugs) == 0:
                print("   ❌ No common drugs found between fingerprints and pIC50 data")
                return []

            # Get indices of common drugs in the sparse matrix
            common_indices = [valid_drugs.index(drug) for drug in common_drugs if drug in valid_drugs]
            X_sparse = sparse_matrix[common_indices]  # Subset sparse matrix
            y_aligned = y.loc[list(common_drugs)]

            print(f"   Aligned data shape: {X_sparse.shape[0]} (drugs) × {X_sparse.shape[1]} (features)")

            # Check for NaN values
            y_na_count = y_aligned.isna().sum().sum()
            print(f"   NaN values in pIC50 data: {y_na_count}")

            # For multi-task learning, we should NOT fill missing values with medians
            # Instead, we'll handle missing values properly by training on available data only
            if y_na_count > 0:
                print("   ⚠️  Multi-task learning: Keeping NaN values as-is (will be handled per-target)")
                print("   📊 Missing value distribution:")
                missing_per_target = y_aligned.isna().sum()
                for target, missing_count in missing_per_target.items():
                    total_samples = len(y_aligned)
                    missing_pct = (missing_count / total_samples) * 100
                    print(f"      {target}: {missing_count}/{total_samples} missing ({missing_pct:.1f}%)")

            # Convert to binary classification using target-specific thresholds
            # We'll do this PER TARGET, only using available data for each target
            print("   🎯 Converting pIC50 values to binary classification labels (per target)...")
            y_binary = pd.DataFrame(index=y_aligned.index, columns=y_aligned.columns)
            target_thresholds = {}
            target_available_data = {}  # Track which compounds have data for each target
            
            for target in y_aligned.columns:
                target_values = y_aligned[target].dropna()
                if len(target_values) < 10:  # Skip targets with insufficient data
                    print(f"      {target}: Insufficient data ({len(target_values)} samples), skipping")
                    continue
                
                # Use target-specific threshold based on data distribution
                # Options: median, 75th percentile, or domain-specific
                median_threshold = target_values.median()
                percentile_75_threshold = target_values.quantile(0.75)
                percentile_60_threshold = target_values.quantile(0.60)  # More balanced
                
                # Choose threshold that creates reasonable class balance (30-70% split)
                # Test different thresholds and pick the one closest to 40-60% positive class
                candidate_thresholds = [
                    median_threshold,
                    percentile_60_threshold, 
                    percentile_75_threshold,
                    6.0,  # Domain-specific fallback
                    7.0   # Stricter domain-specific
                ]
                
                best_threshold = median_threshold
                best_balance_score = float('inf')
                
                for thresh in candidate_thresholds:
                    if thresh < target_values.min() or thresh > target_values.max():
                        continue
                    positive_ratio = (target_values > thresh).mean()
                    # Prefer ratios between 0.2 and 0.8 (20-80% positive)
                    if 0.1 <= positive_ratio <= 0.9:
                        balance_score = abs(positive_ratio - 0.5)  # Closer to 50-50 is better
                        if balance_score < best_balance_score:
                            best_balance_score = balance_score
                            best_threshold = thresh
                
                target_thresholds[target] = best_threshold
                target_available_data[target] = target_values.index.tolist()
                
                # Count positive/negative classes for this threshold
                positive_count = (target_values > best_threshold).sum()
                negative_count = (target_values <= best_threshold).sum()
                positive_pct = (positive_count / len(target_values)) * 100
                
                print(f"      {target}: threshold={best_threshold:.3f}, {positive_count} active ({positive_pct:.1f}%), {negative_count} inactive, {len(target_values)} total")
                
                # Apply threshold to create binary labels ONLY for compounds that have data
                y_binary[target] = np.nan  # Initialize with NaN
                available_mask = y_aligned[target].notna()
                y_binary.loc[available_mask, target] = (y_aligned.loc[available_mask, target] > target_thresholds[target]).astype(int)
            
            # Remove targets that couldn't be processed
            valid_targets = [target for target in y_binary.columns if target in target_thresholds]
            y_binary = y_binary[valid_targets].copy()
            
            # For multi-task learning with missing labels, we'll train separate models for each target
            # This ensures we only use compounds that were actually tested for each target
            print(f"   🤖 Training separate models for each of {len(valid_targets)} targets...")
            
            target_models = {}
            target_scalers = {}
            target_thresholds_final = {}
            results = []

            for target in valid_targets:
                print(f"      Training model for {target}...")
                
                # Get compounds that have data for this target
                available_compounds = target_available_data[target]
                target_indices = [list(y_binary.index).index(compound) for compound in available_compounds if compound in y_binary.index]
                
                if len(target_indices) == 0:
                    print(f"         ❌ No valid compounds for {target}, skipping")
                    continue
                
                # Subset data for this target
                X_target = X_sparse[target_indices]
                y_target = y_binary.loc[available_compounds, target].dropna().astype(int)
                
                if len(y_target) < 10:
                    print(f"         ❌ Insufficient data for {target} ({len(y_target)} samples), skipping")
                    continue
                
                print(f"         📊 {target}: {X_target.shape[0]} compounds, {X_target.shape[1]} features")
                
                # Get the pIC50 threshold for this target for reporting
                pic50_threshold = target_thresholds[target]
                
                # Split data for this target using random split only
                X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
                    X_target, y_target, test_size=0.2, random_state=42
                )
                
                # Scale features for this target
                scaler = MaxAbsScaler()
                X_train_scaled = scaler.fit_transform(X_train_target)
                X_test_scaled = scaler.transform(X_test_target)
                
                # Train classifier for this target
                clf = self.classifiers[classifier_name]
                clf.fit(X_train_scaled, y_train_target)
                
                # Store model and scaler
                target_models[target] = clf
                target_scalers[target] = scaler
                
                # Save the trained model
                model_metadata = {
                    'target': target,
                    'pic50_threshold': pic50_threshold,
                    'n_train_samples': len(y_train_target),
                    'n_test_samples': len(y_test_target),
                    'n_features': X_train_scaled.shape[1],
                    'split_method': split_method,
                    'timestamp': datetime.now().isoformat()
                }
                self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                
                # Evaluate model and get results
                target_results = self._evaluate_target_model(
                    clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                    fingerprint_type, classifier_name, target, pic50_threshold, split_method
                )
                results.extend(target_results)
                
                print(f"         ✅ {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
            
            print(f"   ✅ Generated {len(results)} results for {len(valid_targets)} targets")
            
            # Save results progressively
            self.save_results_progressively(results)
            
            return results



        except Exception as e:
            print(f"   ❌ Error in train_and_evaluate: {e}")
            import traceback
            traceback.print_exc()
            return []


    def run_complete_pipeline(self):
        """Run the complete multi-task learning pipeline - ONLY ECFP_COUNT + RandomForest + Random split"""
        start_time = time.time()

        # Load data
        self.load_data()

        # Generate fingerprints (only ECFP_COUNT)
        self.generate_fingerprints()

        # Train and evaluate ONLY the specified combination
        all_results = []

        # Only one combination: ECFP_COUNT + RandomForest + Random split
        fingerprint_types = ['ecfp_counts']
        classifier_names = ['RandomForest']
        split_methods = ['random']

        total_combinations = 1
        completed = 0

        for fp_type in fingerprint_types:
            for clf_name in classifier_names:
                for split_method in split_methods:
                    try:
                        results = self.train_and_evaluate(
                            fp_type, clf_name, split_method
                        )
                        all_results.extend(results)
                        completed += 1
                        print(f"✅ Completed {completed}/{total_combinations} combinations")
                    except Exception as e:
                        print(f"❌ Error with {fp_type} + {clf_name} + {split_method}: {e}")
                        continue

        # Save any remaining results in buffer
        if self.results_buffer:
            self.save_results_progressively([], force_save=True)

        # Load final results for analysis
        if self.results_file.exists():
            results_df = pd.read_csv(self.results_file)
            print(f"\n📊 Final results loaded: {results_df.shape}")
        else:
            results_df = pd.DataFrame(all_results)
        
        # Find and summarize best models
        if len(results_df) > 0:
            best_models = self.find_and_save_best_models(results_df, top_k=5, metric='AUC')
            print(f"\n🏆 Best models saved and summarized!")

        elapsed_time = time.time() - start_time
        print("\n🎉 Pipeline completed!")
        print(f"📊 Total results: {len(results_df)}")
        print(f"💾 Results saved to: {self.results_file}")
        print(f"📁 Models saved to: {self.models_path}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        return results_df

def main():
    # Initialize classifier
    classifier = MultiTaskDrugClassifier(
        data_path=".",
        results_path="./results_default_19Nov",
        models_path="./saved_model_19Nov"
    )

    # Run complete pipeline with default parameters only
    results = classifier.run_complete_pipeline()

    print("\n📈 Summary of results:")

    if results is None or len(results) == 0:
        print("❌ No results generated. Check the logs above for errors.")
        return

    # Convert to DataFrame and check columns
    results_df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results columns: {list(results_df.columns)}")

    # Check if required columns exist
    required_cols = ['Fingerprint', 'Model', 'Split_Method', 'Accuracy', 'F1', 'AUC']
    missing_cols = [col for col in required_cols if col not in results_df.columns]

    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print("Available columns:", list(results_df.columns))
        return

    try:
        # Separate results by split type
        train_results = results_df[results_df['Split'] == 'Train_Train']
        test_results = results_df[results_df['Split'] == 'Train_Test'] 
        
        print("\n📊 Training Set Performance (Default Parameters):")
        if not train_results.empty:
            train_summary = train_results.groupby(['Fingerprint', 'Model', 'Split_Method'])[['Accuracy', 'F1', 'AUC']].mean().round(4)
            print(train_summary)
        
        print("\n📊 Test Set Performance (Default Parameters):")
        if not test_results.empty:
            test_summary = test_results.groupby(['Fingerprint', 'Model', 'Split_Method'])[['Accuracy', 'F1', 'AUC']].mean().round(4)
            print(test_summary)
        
        # Show saved models
        print(f"\n📁 Saved Models:")
        saved_models = classifier.list_saved_models()
        if len(saved_models) > 0:
            print(saved_models.to_string(index=False))
        else:
            print("No models were saved.")
            
    except Exception as e:
        print(f"❌ Error creating summary: {e}")
        print("First few rows of results:")
        print(results_df.head())

if __name__ == "__main__":
    main()
