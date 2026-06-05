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
import argparse
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors

from sklearn.cluster import KMeans
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. UMAP clustering will use K-means instead.")

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

from tqdm import tqdm
import time
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

class MultiTaskDrugClassifier:
    def __init__(self, data_path, results_path="./results", models_path="./models",
                 fingerprints_cache_dir="./fingerprints_cache", n_jobs_clf=1):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.models_path = Path(models_path)
        self.fingerprints_cache_dir = Path(fingerprints_cache_dir)
        self.n_jobs_clf = n_jobs_clf
        self.results_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        self.fingerprints_cache_dir.mkdir(exist_ok=True)

        self.pIC50_data = None
        self.smiles_data = None
        self.fingerprints = {}
        self.models = {}
        self.best_thresholds = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.results_path / f"multitask_classification_results_default_parameters_{timestamp}.csv"
        self.results_buffer = []

        self.classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=self.n_jobs_clf),
            'SVM': SVC(probability=True, random_state=42, max_iter=10000),
            'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=self.n_jobs_clf),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=self.n_jobs_clf),
            'KNeighbors': KNeighborsClassifier(n_jobs=self.n_jobs_clf),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'GaussianNB': GaussianNB()
        }

        self.requires_dense = {'GaussianNB'}
    
    def _convert_to_dense_if_needed(self, X, classifier_name):
        """Convert sparse matrix to dense if classifier requires it"""
        if classifier_name in self.requires_dense:
            if sparse.issparse(X):
                return X.toarray()
        return X
    
    # Column order for the results CSV — must match keys in _evaluate_target_model result dicts
    RESULTS_COLUMNS = [
        'Fingerprint', 'Model', 'Target', 'N_Features', 'Scaling',
        'Hyperparameters', 'Threshold', 'Split_Method', 'Split',
        'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AUPRC', 'MCC',
        'TP', 'TN', 'FP', 'FN'
    ]

    def _initialize_results_file(self):
        """Create the results CSV with headers immediately so progress is visible from the start."""
        if not self.results_file.exists():
            pd.DataFrame(columns=self.RESULTS_COLUMNS).to_csv(self.results_file, index=False)
            print(f"   Results file created: {self.results_file}")

    def save_results_progressively(self, results, force_save=False):
        """
        Append results to the CSV file.  Called with force_save=True after each
        target completes so every row is on disk before the next target starts.
        The buffer / 1000-threshold path is kept as a fallback for the final flush.
        """
        if not results:
            return

        self.results_buffer.extend(results)

        if len(self.results_buffer) >= 1000 or force_save:
            results_df = pd.DataFrame(self.results_buffer)

            # Reorder columns to canonical order (extra cols appended at end)
            ordered = [c for c in self.RESULTS_COLUMNS if c in results_df.columns]
            extra = [c for c in results_df.columns if c not in self.RESULTS_COLUMNS]
            results_df = results_df[ordered + extra]

            # Header already written by _initialize_results_file; always append
            results_df.to_csv(self.results_file, mode='a', header=False, index=False)

            print(f"   Saved {len(self.results_buffer)} results to {self.results_file}")
            self.results_buffer = []
    
    def calculate_youdens_j_threshold(self, y_true, y_scores):
        """Optimal binary threshold by Youden's J = TPR - FPR."""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            return thresholds[best_threshold_idx], j_scores[best_threshold_idx]
        except Exception as e:
            print(f"   Warning: Failed to calculate Youden's J threshold: {e}")
            return np.median(y_scores), 0.0

    def _get_predictions(self, clf, X_scaled):
        """Get probability predictions, handling different classifier types."""
        try:
            if hasattr(clf, 'predict_proba'):
                y_pred_proba = clf.predict_proba(X_scaled)
                if isinstance(y_pred_proba, list):
                    y_pred_proba = np.array([prob[:, 1] for prob in y_pred_proba]).T
                else:
                    y_pred_proba = y_pred_proba[:, 1]
                return y_pred_proba
            else:
                raise AttributeError("No predict_proba method")
        except (AttributeError, IndexError):
            if hasattr(clf, 'decision_function'):
                decision_scores = clf.decision_function(X_scaled)
                if isinstance(decision_scores, list):
                    decision_scores = np.array(decision_scores).T
                return 1 / (1 + np.exp(-decision_scores))
            else:
                y_pred = clf.predict(X_scaled)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                y_pred_proba = np.zeros_like(y_pred, dtype=float)
                y_pred_proba[y_pred == 1] = 0.5
                y_pred_proba[y_pred == 0] = 0.5
                return y_pred_proba

    def _evaluate_target_model(self, clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                              fingerprint_type, classifier_name, target, pic50_threshold, split_method):
        """Evaluate a single target model; returns [test_result, train_result]."""
        y_pred_proba_train = self._get_predictions_single(clf, X_train_scaled)
        y_pred_proba_test  = self._get_predictions_single(clf, X_test_scaled)

        probability_threshold = 0.5
        y_pred_train = (y_pred_proba_train >= probability_threshold).astype(int)
        y_pred_test  = (y_pred_proba_test  >= probability_threshold).astype(int)

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

    def _evaluate_target_model_cv(self, X_target, y_target, fingerprint_type, classifier_name, target, pic50_threshold, n_folds=5):
        """5-fold cross-validation for a single target model."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_target)):
            X_train_fold = X_target[train_idx]
            X_test_fold  = X_target[test_idx]
            y_train_fold = y_target.iloc[train_idx]
            y_test_fold  = y_target.iloc[test_idx]

            scaler = MaxAbsScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled  = scaler.transform(X_test_fold)

            X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
            X_test_scaled  = self._convert_to_dense_if_needed(X_test_scaled,  classifier_name)

            clf = self.classifiers[classifier_name]
            clf.fit(X_train_scaled, y_train_fold)

            y_pred_proba_test = self._get_predictions_single(clf, X_test_scaled)
            y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

            metrics = self.evaluate_model(y_test_fold, y_pred_test, y_pred_proba_test)
            
            result = {
                'Fingerprint': fingerprint_type.upper(),
                'Model': classifier_name,
                'Target': target,
                'N_Features': X_train_scaled.shape[1],
                'Scaling': 'MaxAbsScaler',
                'Hyperparameters': 'default',
                'Threshold': pic50_threshold,
                'Split_Method': 'cross_validation',
                'Split': f'Fold_{fold+1}',
                **metrics
            }
            cv_results.append(result)
        
        return cv_results

    def _get_predictions_single(self, clf, X_scaled):
        """Positive-class probability for single-output classifiers."""
        try:
            if hasattr(clf, 'predict_proba'):
                y_pred_proba = clf.predict_proba(X_scaled)
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                return y_pred_proba
            else:
                raise AttributeError("No predict_proba method")
        except (AttributeError, IndexError):
            if hasattr(clf, 'decision_function'):
                return 1 / (1 + np.exp(-clf.decision_function(X_scaled)))
            else:
                return np.where(clf.predict(X_scaled) == 1, 0.5, 0.5)


    def save_model(self, model, scaler, metadata, fingerprint_type, classifier_name, split_method, target_name=None):
        """Save trained model, scaler, and metadata to a subdirectory under models_path."""
        def sanitize_filename(name):
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            sanitized = name
            for char in invalid_chars:
                sanitized = sanitized.replace(char, '_')
            return sanitized
        
        model_id = f"{fingerprint_type}_{classifier_name}_{split_method}"
        if target_name:
            model_id += f"_{sanitize_filename(target_name)}"

        model_dir = self.models_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model,  model_dir / "model.joblib")
        joblib.dump(scaler, model_dir / "scaler.joblib")

        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"   Saved model: {model_id}")
        return model_id
    
    def load_model(self, model_id):
        """Load model, scaler, and metadata from a saved model directory."""
        model_dir = self.models_path / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id} not found in {self.models_path}")

        model = joblib.load(model_dir / "model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        with open(model_dir / "metadata.json", 'r') as f:
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
    
    def predict_with_saved_model(self, model_id, smiles_list):
        """Predict activity for a list of SMILES using a saved single-target model."""
        model, scaler, metadata = self.load_model(model_id)

        target         = metadata.get('target', model_id)
        fp_type        = metadata.get('fingerprint_type', '').lower()
        clf_name       = metadata.get('classifier_name', '')
        pic50_thresh   = metadata.get('pic50_threshold', float('nan'))
        prob_thresh    = 0.5   # probability cut-off for binary label

        print(f"Predicting with model: {model_id}")
        print(f"   Target     : {target}")
        print(f"   Fingerprint: {fp_type.upper()}")
        print(f"   Classifier : {clf_name}")
        print(f"   pIC50 threshold used during training: {pic50_thresh:.4f}")

        temp_smiles = {f"cpd_{i}": smi for i, smi in enumerate(smiles_list)}

        fp_generators = {
            'avalon':      self._generate_avalon_from_smiles,
            'ecfp':        self._generate_ecfp_from_smiles,
            'ecfp_count':  self._generate_ecfp_count_from_smiles,
            'maccs':       self._generate_maccs_from_smiles,
        }
        if fp_type not in fp_generators:
            raise ValueError(f"Unknown fingerprint type '{fp_type}' in model metadata")

        X_sparse, valid_names = fp_generators[fp_type](temp_smiles)
        X_scaled = scaler.transform(X_sparse)
        X_scaled = self._convert_to_dense_if_needed(X_scaled, clf_name)  # e.g. GaussianNB
        y_proba = self._get_predictions_single(model, X_scaled)
        y_pred  = (y_proba >= prob_thresh).astype(int)

        results = pd.DataFrame({
            'compound_id':      valid_names,
            'SMILES':           [smiles_list[int(n.split('_')[1])] for n in valid_names],
            'probability':      y_proba,
            'prediction':       y_pred,
            'pic50_threshold':  pic50_thresh,
            'prob_threshold':   prob_thresh,
            'target':           target,
        })
        return results
    
    def _generate_avalon_from_smiles(self, smiles_dict):
        """Helper method to generate Avalon fingerprints from SMILES as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in smiles_dict.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = pyAvalonTools.GetAvalonFP(mol, nBits=2048)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
                else:
                    fingerprints.append([0] * 2048)
                    valid_drugs.append(drug_name)
            except:
                fingerprints.append([0] * 2048)
                valid_drugs.append(drug_name)

        sparse_matrix = sparse.csr_matrix(np.array(fingerprints, dtype=np.int8))
        return sparse_matrix, valid_drugs

    def _generate_ecfp_from_smiles(self, smiles_dict):
        """Helper method to generate ECFP fingerprints from SMILES as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in smiles_dict.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
                else:
                    fingerprints.append([0] * 2048)
                    valid_drugs.append(drug_name)
            except:
                fingerprints.append([0] * 2048)
                valid_drugs.append(drug_name)

        sparse_matrix = sparse.csr_matrix(np.array(fingerprints, dtype=np.int8))
        return sparse_matrix, valid_drugs

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

        sparse_matrix = sparse.csr_matrix(np.array(fingerprints, dtype=np.int8))
        return sparse_matrix, valid_drugs

    def _generate_maccs_from_smiles(self, smiles_dict):
        """Helper method to generate MACCS fingerprints from SMILES as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in smiles_dict.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
                else:
                    fingerprints.append([0] * 167)  # MACCS keys have 167 bits
                    valid_drugs.append(drug_name)
            except:
                fingerprints.append([0] * 167)  # MACCS keys have 167 bits
                valid_drugs.append(drug_name)

        sparse_matrix = sparse.csr_matrix(np.array(fingerprints, dtype=np.int8))
        return sparse_matrix, valid_drugs

    def find_and_save_best_models(self, results_df, top_k=5, metric='AUC'):
        """Identify and save top_k best model configurations ranked by mean metric."""
        print(f"\nFinding top {top_k} models based on {metric}...")
        
        train_test_results = results_df[results_df['Split'] == 'Train_Test'].copy()

        if train_test_results.empty:
            print("No train-test results found")
            return pd.DataFrame()

        model_avg_performance = train_test_results.groupby([
            'Fingerprint', 'Model', 'Split_Method'
        ])[metric].agg(['mean', 'std', 'count']).reset_index()

        model_avg_performance = model_avg_performance.sort_values('mean', ascending=False)

        print("Top model configurations:")
        print(model_avg_performance.head(top_k))

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

        best_models_file = self.results_path / "best_models_summary.csv"
        best_models_df.to_csv(best_models_file, index=False)
        print(f"Saved best models summary to: {best_models_file}")
        
        return best_models_df

    def load_data(self):
        """Load transformed kinase data."""
        print("Loading transformed kinase dataset...")

        pic50_file = self.data_path / "kinase_multitask_14May_V15_188217_Original_output" / "kinase_multitask_14May_V15_188217_Original_pIC50_matrix.csv"
        if not pic50_file.exists():
            raise FileNotFoundError(f"Transformed pIC50 data not found at {pic50_file}. Please run transform_kinase_multitask.py first.")
        
        self.pIC50_data = pd.read_csv(pic50_file, index_col=0)

        # Drop metadata columns injected by the transform script — they are not kinase targets
        metadata_cols = [c for c in self.pIC50_data.columns
                         if c in ('InChIKey_Block1', 'Binding_Monomer_ID')]
        if metadata_cols:
            self.pIC50_data = self.pIC50_data.drop(columns=metadata_cols)
            print(f"   Dropped metadata columns: {metadata_cols}")

        smiles_file = self.data_path / "kinase_multitask_14May_V15_188217_Original_output" / "kinase_multitask_14May_V15_188217_Original_smiles_mapping.csv"
        if smiles_file.exists():
            smiles_df = pd.read_csv(smiles_file)
            self.smiles_data = dict(zip(smiles_df['Compound_ID'], smiles_df['SMILES']))
        else:
            print("   Warning: SMILES mapping not found. Some splitting methods may not work.")
            self.smiles_data = {}

        self.smiles_data = {k: v for k, v in self.smiles_data.items() if k in self.pIC50_data.index}

        print(f"Loaded pIC50 data: {self.pIC50_data.shape}")
        print(f"SMILES data: {len(self.smiles_data)} compounds")
        print(f"Targets: {list(self.pIC50_data.columns[:5])}{'...' if len(self.pIC50_data.columns) > 5 else ''}")
        print(f"pIC50 range: {self.pIC50_data.min().min():.3f} to {self.pIC50_data.max().max():.3f}")
        print(f"Data completeness: {(1 - self.pIC50_data.isna().sum().sum() / (self.pIC50_data.shape[0] * self.pIC50_data.shape[1])) * 100:.1f}%")

    def _fingerprint_cache_path(self, fp_name):
        """Return the cache file path for a given fingerprint type."""
        return self.fingerprints_cache_dir / f"{fp_name}_fingerprints.joblib"

    def _load_fingerprint_cache(self, fp_name):
        """
        Load fingerprints from cache if the file exists and matches the
        current compound set (validated by compound count + sorted ID list).

        Returns (sparse_matrix, valid_drugs) on success, None on miss/mismatch.
        """
        cache_path = self._fingerprint_cache_path(fp_name)
        if not cache_path.exists():
            return None

        try:
            cache = joblib.load(cache_path)
            cached_ids   = set(cache['compound_ids'])
            current_ids  = set(self.smiles_data.keys())

            if cached_ids != current_ids:
                print(f"   {fp_name.upper()} cache compound set mismatch — regenerating")
                return None

            print(f"   Loaded {fp_name.upper()} fingerprints from cache: {cache_path}")
            return cache['matrix'], cache['valid_drugs']

        except Exception as e:
            print(f"   Could not read {fp_name.upper()} cache ({e}) — regenerating")
            return None

    def _save_fingerprint_cache(self, fp_name, sparse_matrix, valid_drugs):
        """Save fingerprints to cache for future reuse."""
        cache_path = self._fingerprint_cache_path(fp_name)
        cache = {
            'matrix':       sparse_matrix,
            'valid_drugs':  valid_drugs,
            'compound_ids': list(self.smiles_data.keys()),
            'n_compounds':  len(valid_drugs),
            'fp_name':      fp_name,
            'timestamp':    datetime.now().isoformat(),
        }
        joblib.dump(cache, cache_path, compress=3)
        size_mb = cache_path.stat().st_size / 1_048_576
        print(f"   Saved {fp_name.upper()} fingerprint cache: {cache_path} ({size_mb:.1f} MB)")

    def generate_fingerprints(self, force_regenerate=False):
        """Generate (or load from cache) molecular fingerprints for all compounds."""
        print("\nGenerating/loading molecular fingerprints...")
        if force_regenerate:
            print("   force_regenerate=True — ignoring existing cache")

        fingerprint_functions = {
            'avalon':     self._generate_avalon,
            'ecfp':       self._generate_ecfp,
            'ecfp_count': self._generate_ecfp_count,
            'maccs':      self._generate_maccs,
        }

        for fp_name, fp_func in fingerprint_functions.items():
            if not force_regenerate:
                cached = self._load_fingerprint_cache(fp_name)
                if cached is not None:
                    self.fingerprints[fp_name] = cached
                    sparse_matrix, valid_drugs = cached
                    print(f"   {fp_name.upper()}: {sparse_matrix.shape} (from cache), {len(valid_drugs)} molecules")
                    continue

            print(f"   Generating {fp_name.upper()} fingerprints...")
            sparse_matrix, valid_drugs = fp_func()
            self._save_fingerprint_cache(fp_name, sparse_matrix, valid_drugs)
            self.fingerprints[fp_name] = (sparse_matrix, valid_drugs)
            print(f"   {fp_name.upper()}: {sparse_matrix.shape} (freshly generated), {len(valid_drugs)} molecules")

        print(f"Fingerprints ready for {len(self.fingerprints)} types")

    def get_scaffold_splits(self, smiles_list, test_size=0.2, random_state=42):
        """Generate scaffold-based train/test splits"""
        scaffolds = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds.append(Chem.MolToSmiles(scaffold))
                else:
                    scaffolds.append("invalid")
            except:
                scaffolds.append("invalid")

        scaffold_groups = {}
        for i, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_groups:
                scaffold_groups[scaffold] = []
            scaffold_groups[scaffold].append(i)

        sorted_scaffolds = sorted(scaffold_groups.items(),
                                key=lambda x: len(x[1]), reverse=True)

        test_indices = []
        train_indices = []
        target_test_size = int(len(smiles_list) * test_size)

        for scaffold, indices in sorted_scaffolds:
            if len(test_indices) < target_test_size:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)

        # If we don't have enough test samples, take from training
        if len(test_indices) < target_test_size:
            remaining = target_test_size - len(test_indices)
            additional_indices = train_indices[:remaining]
            test_indices.extend(additional_indices)
            train_indices = train_indices[remaining:]

        return train_indices, test_indices

    def get_butina_splits(self, smiles_list, test_size=0.2, random_state=42):
        """Generate Butina clustering-based train/test splits using memory-efficient approach"""
        fingerprints = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints.append(fp)
                    valid_indices.append(i)
            except:
                continue

        if len(fingerprints) < 10:
            # Fallback to random split if too few valid molecules
            np.random.seed(random_state)
            indices = np.random.permutation(len(smiles_list))
            split_point = int(len(smiles_list) * (1 - test_size))
            return indices[:split_point], indices[split_point:]

        n = len(fingerprints)
        print(f"   Butina clustering: {n} valid molecules")

        # For large datasets, use memory-efficient clustering
        if n > 5000:
            print(f"   Large dataset ({n} molecules), using K-means clustering instead of Butina")
            from sklearn.cluster import KMeans
            X = np.array([list(fp) for fp in fingerprints])
            n_clusters = max(2, min(20, n // 100))
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(X)
        else:
            # For smaller datasets, use proper Butina clustering with memory-efficient implementation
            try:
                from rdkit import DataStructs
                from rdkit.ML.Cluster import Butina

                distances = []
                for i in range(n):
                    for j in range(i+1, n):
                        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                        distances.append(1 - similarity)

                clusters = Butina.ClusterData(distances, n, 0.7, isDistData=True)
                cluster_array = np.zeros(n, dtype=int)
                for cluster_id, cluster_indices in enumerate(clusters):
                    for idx in cluster_indices:
                        cluster_array[idx] = cluster_id

                clusters = cluster_array

            except ImportError:
                print("   Warning: RDKit Butina not available, using K-means fallback")
                # Fallback to K-means
                from sklearn.cluster import KMeans
                X = np.array([list(fp) for fp in fingerprints])
                n_clusters = max(2, min(10, n // 50))
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                clusters = kmeans.fit_predict(X)

        cluster_counts = {}
        for cluster_id in clusters:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

        test_indices = []
        train_indices = []
        target_test_size = int(len(valid_indices) * test_size)

        for cluster_id, _ in sorted_clusters:
            cluster_indices = [valid_indices[i] for i in range(len(clusters))
                             if clusters[i] == cluster_id]

            if len(test_indices) < target_test_size:
                test_indices.extend(cluster_indices)
            else:
                train_indices.extend(cluster_indices)

        if len(test_indices) < target_test_size:
            remaining = target_test_size - len(test_indices)
            additional_indices = train_indices[:remaining]
            test_indices.extend(additional_indices)
            train_indices = train_indices[remaining:]

        return train_indices, test_indices

    def get_umap_clustering_splits(self, X, test_size=0.2, random_state=42):
        """Generate UMAP clustering-based train/test splits"""
        if UMAP_AVAILABLE and X.shape[0] > 10:
            umap_reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=15)
            X_umap = umap_reducer.fit_transform(X)
        else:
            # Fallback to PCA if UMAP not available or too few samples
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            X_umap = pca.fit_transform(X)

        n_clusters = max(2, min(10, X.shape[0] // 20))  # Adaptive cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X_umap)

        cluster_counts = {}
        for cluster_id in clusters:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

        test_indices = []
        train_indices = []
        target_test_size = int(X.shape[0] * test_size)

        for cluster_id, _ in sorted_clusters:
            cluster_indices = [i for i in range(len(clusters)) if clusters[i] == cluster_id]

            if len(test_indices) < target_test_size:
                test_indices.extend(cluster_indices)
            else:
                train_indices.extend(cluster_indices)

        if len(test_indices) < target_test_size:
            remaining = target_test_size - len(test_indices)
            additional_indices = train_indices[:remaining]
            test_indices.extend(additional_indices)
            train_indices = train_indices[remaining:]

        return train_indices, test_indices

    def _generate_avalon(self):
        """Generate Avalon fingerprints as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in tqdm(self.smiles_data.items(), desc="Avalon"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = pyAvalonTools.GetAvalonFP(mol, nBits=2048)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
            except:
                continue

        dense_array = np.array(fingerprints, dtype=np.int8)  # Use int8 to save memory
        sparse_matrix = sparse.csr_matrix(dense_array)
        print(f"   Avalon sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz} non-zero elements ({sparse_matrix.nnz/sparse_matrix.size*100:.1f}% density)")
        return sparse_matrix, valid_drugs

    def _generate_ecfp(self):
        """Generate ECFP (Extended Connectivity Fingerprints) as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        valid_count = 0
        invalid_count = 0

        for drug_name, smiles in tqdm(self.smiles_data.items(), desc="ECFP"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
                    valid_count += 1
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
                continue

        print(f"   ECFP: {valid_count} valid, {invalid_count} invalid molecules")

        dense_array = np.array(fingerprints, dtype=np.int8)  # Use int8 to save memory
        sparse_matrix = sparse.csr_matrix(dense_array)
        print(f"   ECFP sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz} non-zero elements ({sparse_matrix.nnz/sparse_matrix.size*100:.1f}% density)")
        return sparse_matrix, valid_drugs

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

        dense_array = np.array(fingerprints, dtype=np.int8)  # Use int8 to save memory
        sparse_matrix = sparse.csr_matrix(dense_array)
        print(f"   ECFP Count sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz} non-zero elements ({sparse_matrix.nnz/sparse_matrix.size*100:.1f}% density)")
        return sparse_matrix, valid_drugs

    def _generate_maccs(self):
        """Generate MACCS keys as sparse matrix"""
        fingerprints = []
        valid_drugs = []
        for drug_name, smiles in tqdm(self.smiles_data.items(), desc="MACCS"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    fingerprints.append(list(fp))
                    valid_drugs.append(drug_name)
            except:
                continue

        dense_array = np.array(fingerprints, dtype=np.int8)  # Use int8 to save memory
        sparse_matrix = sparse.csr_matrix(dense_array)
        print(f"   MACCS sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz} non-zero elements ({sparse_matrix.nnz/sparse_matrix.size*100:.1f}% density)")
        return sparse_matrix, valid_drugs

    def determine_optimal_threshold(self, y_true, y_pred_proba, method='balanced'):
        """Determine optimal classification threshold"""
        if method == 'balanced':
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return thresholds[best_idx]
        elif method == 'f1':
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
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            if hasattr(y_pred_proba, 'values'):
                y_pred_proba = y_pred_proba.values

            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (1, 1):
                if y_true[0] == 0:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                tn, fp, fn, tp = cm.ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, len(y_true)

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
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
        print(f"\nTraining {classifier_name} with {fingerprint_type.upper()} fingerprints, {split_method} split...")

        try:
            sparse_matrix, valid_drugs = self.fingerprints[fingerprint_type]
            y = self.pIC50_data

            print(f"   Fingerprint sparse matrix shape: {sparse_matrix.shape}")
            print(f"   Valid drugs: {len(valid_drugs)}")
            print(f"   pIC50 data shape: {y.shape}")

            common_drugs = set(valid_drugs) & set(y.index)
            print(f"   Common drugs: {len(common_drugs)}")

            if len(common_drugs) == 0:
                print("   No common drugs found between fingerprints and pIC50 data")
                return []

            common_indices = [valid_drugs.index(drug) for drug in common_drugs if drug in valid_drugs]
            X_sparse = sparse_matrix[common_indices]
            y_aligned = y.loc[list(common_drugs)]

            print(f"   Aligned data shape: {X_sparse.shape[0]} (drugs) × {X_sparse.shape[1]} (features)")

            y_na_count = y_aligned.isna().sum().sum()
            print(f"   NaN values in pIC50 data: {y_na_count}")

            if y_na_count > 0:
                print("   Multi-task learning: keeping NaN values as-is (handled per-target)")
                print("   Missing value distribution:")
                missing_per_target = y_aligned.isna().sum()
                for target, missing_count in missing_per_target.items():
                    total_samples = len(y_aligned)
                    missing_pct = (missing_count / total_samples) * 100
                    print(f"      {target}: {missing_count}/{total_samples} missing ({missing_pct:.1f}%)")

            print("   Converting pIC50 values to binary classification labels (per target)...")
            y_binary = pd.DataFrame(index=y_aligned.index, columns=y_aligned.columns)
            target_thresholds = {}
            target_available_data = {}  # Track which compounds have data for each target
            
            for target in y_aligned.columns:
                target_values = y_aligned[target].dropna()
                if len(target_values) < 10:  # Skip targets with insufficient data
                    print(f"      {target}: Insufficient data ({len(target_values)} samples), skipping")
                    continue
                
                median_threshold = target_values.median()
                percentile_75_threshold = target_values.quantile(0.75)
                percentile_60_threshold = target_values.quantile(0.60)  # More balanced
                
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
                        balance_score = abs(positive_ratio - 0.5)
                        if balance_score < best_balance_score:
                            best_balance_score = balance_score
                            best_threshold = thresh
                
                target_thresholds[target] = best_threshold
                target_available_data[target] = target_values.index.tolist()
                
                positive_count = (target_values > best_threshold).sum()
                negative_count = (target_values <= best_threshold).sum()
                positive_pct = (positive_count / len(target_values)) * 100
                
                print(f"      {target}: threshold={best_threshold:.3f}, {positive_count} active ({positive_pct:.1f}%), {negative_count} inactive, {len(target_values)} total")
                
                y_binary[target] = np.nan
                available_mask = y_aligned[target].notna()
                y_binary.loc[available_mask, target] = (y_aligned.loc[available_mask, target] > target_thresholds[target]).astype(int)
            
            valid_targets = [target for target in y_binary.columns if target in target_thresholds]
            y_binary = y_binary[valid_targets].copy()
            
            print(f"   Training separate models for each of {len(valid_targets)} targets...")
            
            target_models = {}
            target_scalers = {}
            target_thresholds_final = {}
            results = []
            self._initialize_results_file()

            for target in valid_targets:
                print(f"      Training model for {target}...")
                
                available_compounds = target_available_data[target]
                target_indices = [list(y_binary.index).index(compound) for compound in available_compounds if compound in y_binary.index]
                
                if len(target_indices) == 0:
                    print(f"         No valid compounds for {target}, skipping")
                    continue
                
                X_target = X_sparse[target_indices]
                y_target = y_binary.loc[available_compounds, target].dropna().astype(int)
                
                if len(y_target) < 10:
                    print(f"         Insufficient data for {target} ({len(y_target)} samples), skipping")
                    continue
                
                print(f"         {target}: {X_target.shape[0]} compounds, {X_target.shape[1]} features")
                
                pic50_threshold = target_thresholds[target]

                if split_method == 'random':
                    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
                        X_target, y_target, test_size=0.2, random_state=42
                    )
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)

                    # 5-fold cross-validation on the full target dataset (all compounds)
                    cv_fold_results = self._evaluate_target_model_cv(
                        X_target, y_target, fingerprint_type, classifier_name, target, pic50_threshold
                    )
                    results.extend(cv_fold_results)
                    self.save_results_progressively(cv_fold_results, force_save=True)

                    cv_aucs = [r['AUC'] for r in cv_fold_results]
                    cv_mean = sum(cv_aucs) / len(cv_aucs)
                    cv_std  = (sum((v - cv_mean) ** 2 for v in cv_aucs) / (len(cv_aucs) - 1)) ** 0.5
                    print(f"         {target}: Test AUC={target_results[0]['AUC']:.3f} | "
                          f"CV AUC={cv_mean:.3f} ± {cv_std:.3f} (5-fold)")
                    
                elif split_method == 'scaffold':
                    target_smiles = [self.smiles_data.get(compound, "") for compound in available_compounds]
                    train_indices, test_indices = self.get_scaffold_splits(target_smiles, test_size=0.2)
                    X_train_target, X_test_target = X_target[train_indices], X_target[test_indices]
                    y_train_target, y_test_target = y_target.iloc[train_indices], y_target.iloc[test_indices]
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                elif split_method == 'butina':
                    target_smiles = [self.smiles_data.get(compound, "") for compound in available_compounds]
                    train_indices, test_indices = self.get_butina_splits(target_smiles, test_size=0.2)
                    X_train_target, X_test_target = X_target[train_indices], X_target[test_indices]
                    y_train_target, y_test_target = y_target.iloc[train_indices], y_target.iloc[test_indices]
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                elif split_method == 'umap_clustering':
                    X_dense = X_target.toarray()
                    train_indices, test_indices = self.get_umap_clustering_splits(X_dense, test_size=0.2)
                    X_train_target, X_test_target = X_target[train_indices], X_target[test_indices]
                    y_train_target, y_test_target = y_target.iloc[train_indices], y_target.iloc[test_indices]
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                else:
                    print(f"         Unknown split method: {split_method}")
                    continue
            
            print(f"   Generated {len(results)} results for {len(valid_targets)} targets")
            return results



        except Exception as e:
            print(f"   Error in train_and_evaluate: {e}")
            import traceback
            traceback.print_exc()
            return []


    def train_and_evaluate_specific(self, fingerprint_type, classifier_name, split_method, target_name=None):
        """Train and evaluate a specific model configuration for specific target(s)"""
        print(f"\nTraining {classifier_name} with {fingerprint_type.upper()} fingerprints, {split_method} split...")

        try:
            sparse_matrix, valid_drugs = self.fingerprints[fingerprint_type]
            y = self.pIC50_data

            print(f"   Fingerprint sparse matrix shape: {sparse_matrix.shape}")
            print(f"   Valid drugs: {len(valid_drugs)}")
            print(f"   pIC50 data shape: {y.shape}")

            common_drugs = set(valid_drugs) & set(y.index)
            print(f"   Common drugs: {len(common_drugs)}")

            if len(common_drugs) == 0:
                print("   No common drugs found between fingerprints and pIC50 data")
                return []

            common_indices = [valid_drugs.index(drug) for drug in common_drugs if drug in valid_drugs]
            X_sparse = sparse_matrix[common_indices]
            y_aligned = y.loc[list(common_drugs)]

            print(f"   Aligned data shape: {X_sparse.shape[0]} (drugs) × {X_sparse.shape[1]} (features)")

            y_na_count = y_aligned.isna().sum().sum()
            print(f"   NaN values in pIC50 data: {y_na_count}")

            if y_na_count > 0:
                print("   Multi-task learning: keeping NaN values as-is (handled per-target)")
                print("   Missing value distribution:")
                missing_per_target = y_aligned.isna().sum()
                for target, missing_count in missing_per_target.items():
                    total_samples = len(y_aligned)
                    missing_pct = (missing_count / total_samples) * 100
                    print(f"      {target}: {missing_count}/{total_samples} missing ({missing_pct:.1f}%)")

            print("   Converting pIC50 values to binary classification labels (per target)...")
            y_binary = pd.DataFrame(index=y_aligned.index, columns=y_aligned.columns)
            target_thresholds = {}
            target_available_data = {}  # Track which compounds have data for each target
            
            for target in y_aligned.columns:
                target_values = y_aligned[target].dropna()
                if len(target_values) < 10:  # Skip targets with insufficient data
                    print(f"      {target}: Insufficient data ({len(target_values)} samples), skipping")
                    continue
                
                median_threshold = target_values.median()
                percentile_75_threshold = target_values.quantile(0.75)
                percentile_60_threshold = target_values.quantile(0.60)  # More balanced
                
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
                        balance_score = abs(positive_ratio - 0.5)
                        if balance_score < best_balance_score:
                            best_balance_score = balance_score
                            best_threshold = thresh
                
                target_thresholds[target] = best_threshold
                target_available_data[target] = target_values.index.tolist()
                
                positive_count = (target_values > best_threshold).sum()
                negative_count = (target_values <= best_threshold).sum()
                positive_pct = (positive_count / len(target_values)) * 100
                
                print(f"      {target}: threshold={best_threshold:.3f}, {positive_count} active ({positive_pct:.1f}%), {negative_count} inactive, {len(target_values)} total")
                
                y_binary[target] = np.nan
                available_mask = y_aligned[target].notna()
                y_binary.loc[available_mask, target] = (y_aligned.loc[available_mask, target] > target_thresholds[target]).astype(int)
            
            valid_targets = [target for target in y_binary.columns if target in target_thresholds]
            y_binary = y_binary[valid_targets].copy()
            
            if target_name:
                if target_name in valid_targets:
                    valid_targets = [target_name]
                    print(f"   Training only for target: {target_name}")
                else:
                    print(f"   Target '{target_name}' not found in valid targets")
                    return []
            
            print(f"   Training separate models for each of {len(valid_targets)} targets...")
            
            target_models = {}
            target_scalers = {}
            target_thresholds_final = {}
            results = []
            self._initialize_results_file()

            for target in valid_targets:
                print(f"      Training model for {target}...")
                
                available_compounds = target_available_data[target]
                target_indices = [list(y_binary.index).index(compound) for compound in available_compounds if compound in y_binary.index]
                
                if len(target_indices) == 0:
                    print(f"         No valid compounds for {target}, skipping")
                    continue
                
                X_target = X_sparse[target_indices]
                y_target = y_binary.loc[available_compounds, target].dropna().astype(int)
                
                if len(y_target) < 10:
                    print(f"         Insufficient data for {target} ({len(y_target)} samples), skipping")
                    continue
                
                print(f"         {target}: {X_target.shape[0]} compounds, {X_target.shape[1]} features")
                
                pic50_threshold = target_thresholds[target]

                if split_method == 'random':
                    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
                        X_target, y_target, test_size=0.2, random_state=42
                    )
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                elif split_method == 'scaffold':
                    target_smiles = [self.smiles_data.get(compound, "") for compound in available_compounds]
                    train_indices, test_indices = self.get_scaffold_splits(target_smiles, test_size=0.2)
                    X_train_target, X_test_target = X_target[train_indices], X_target[test_indices]
                    y_train_target, y_test_target = y_target.iloc[train_indices], y_target.iloc[test_indices]
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                elif split_method == 'butina':
                    target_smiles = [self.smiles_data.get(compound, "") for compound in available_compounds]
                    train_indices, test_indices = self.get_butina_splits(target_smiles, test_size=0.2)
                    X_train_target, X_test_target = X_target[train_indices], X_target[test_indices]
                    y_train_target, y_test_target = y_target.iloc[train_indices], y_target.iloc[test_indices]
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                elif split_method == 'umap_clustering':
                    X_dense = X_target.toarray()
                    train_indices, test_indices = self.get_umap_clustering_splits(X_dense, test_size=0.2)
                    X_train_target, X_test_target = X_target[train_indices], X_target[test_indices]
                    y_train_target, y_test_target = y_target.iloc[train_indices], y_target.iloc[test_indices]
                    
                    scaler = MaxAbsScaler()
                    X_train_scaled = scaler.fit_transform(X_train_target)
                    X_test_scaled = scaler.transform(X_test_target)

                    X_train_scaled = self._convert_to_dense_if_needed(X_train_scaled, classifier_name)
                    X_test_scaled = self._convert_to_dense_if_needed(X_test_scaled, classifier_name)

                    clf = self.classifiers[classifier_name]
                    clf.fit(X_train_scaled, y_train_target)

                    target_models[target] = clf
                    target_scalers[target] = scaler

                    model_metadata = {
                        'fingerprint_type': fingerprint_type,
                        'classifier_name': classifier_name,
                        'target': target,
                        'pic50_threshold': pic50_threshold,
                        'n_train_samples': len(y_train_target),
                        'n_test_samples': len(y_test_target),
                        'n_features': X_train_scaled.shape[1],
                        'split_method': split_method,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_model(clf, scaler, model_metadata, fingerprint_type, classifier_name, split_method, target)
                    
                    target_results = self._evaluate_target_model(
                        clf, X_train_scaled, X_test_scaled, y_train_target, y_test_target,
                        fingerprint_type, classifier_name, target, pic50_threshold, split_method
                    )
                    results.extend(target_results)
                    self.save_results_progressively(target_results, force_save=True)
                    
                    print(f"         {target}: Test {target_results[0]['Accuracy']:.3f}, Train {target_results[1]['Accuracy']:.3f}")
                    
                else:
                    print(f"         Unknown split method: {split_method}")
                    continue
            
            print(f"   Generated {len(results)} results for {len(valid_targets)} targets")
            return results



        except Exception as e:
            print(f"   Error in train_and_evaluate_specific: {e}")
            import traceback
            traceback.print_exc()
            return []


    def run_complete_pipeline(self, max_workers=None):
        """Run all FP × classifier × split combinations. max_workers=-1 uses cpu_count-2."""
        start_time = time.time()

        # Load data and pre-generate ALL fingerprint caches in the main process.
        # Workers will simply load from cache — no contention, no recompute.
        self.load_data()
        self.generate_fingerprints()

        fingerprint_types = ['avalon', 'ecfp', 'ecfp_count', 'maccs']
        classifier_names  = list(self.classifiers.keys())
        split_methods     = ['random', 'scaffold', 'butina', 'umap_clustering']

        total_combinations = len(fingerprint_types) * len(classifier_names) * len(split_methods)

        if max_workers == -1:
            max_workers = max(1, (os.cpu_count() or 2) - 2)

        if max_workers is not None and max_workers > 1:
            print(f"\nParallel mode: {max_workers} workers, {total_combinations} combinations")

            tmp_dir = self.results_path / "_tmp_parallel"
            tmp_dir.mkdir(exist_ok=True)

            combo_args = [
                (
                    str(self.data_path),
                    str(self.results_path),
                    str(self.models_path),
                    str(self.fingerprints_cache_dir),
                    fp_type, clf_name, split_method,
                    str(tmp_dir / f"{fp_type}_{clf_name}_{split_method}.csv"),
                )
                for fp_type   in fingerprint_types
                for clf_name  in classifier_names
                for split_method in split_methods
            ]

            completed = 0
            errors = []
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_worker_run_combination, args): args
                           for args in combo_args}
                for future in tqdm(as_completed(futures), total=total_combinations,
                                   desc="Combinations", unit="combo"):
                    fp_type, clf_name, split_method, tb = future.result()
                    completed += 1
                    if tb:
                        errors.append((fp_type, clf_name, split_method))
                        print(f"\nFAILED {fp_type}+{clf_name}+{split_method}:\n{tb}")
                    else:
                        print(f"  OK [{completed}/{total_combinations}] {fp_type}+{clf_name}+{split_method}")

            tmp_csvs = sorted(tmp_dir.glob("*.csv"))
            non_empty = [f for f in tmp_csvs if f.stat().st_size > 0]
            if non_empty:
                results_df = pd.concat(
                    [pd.read_csv(f) for f in non_empty], ignore_index=True
                )
                results_df.to_csv(self.results_file, index=False)
                print(f"\nMerged {len(non_empty)} result files → {self.results_file}")
                for f in tmp_csvs:
                    f.unlink()
                tmp_dir.rmdir()
            else:
                print("No result files were produced — check worker errors above.")
                results_df = pd.DataFrame()

            if errors:
                print(f"\n{len(errors)} combination(s) failed: {errors}")

        else:
            print(f"\nSequential mode: {total_combinations} combinations")
            self._initialize_results_file()
            all_results = []
            completed = 0
            for fp_type in fingerprint_types:
                for clf_name in classifier_names:
                    for split_method in split_methods:
                        try:
                            results = self.train_and_evaluate(fp_type, clf_name, split_method)
                            all_results.extend(results)
                            completed += 1
                            print(f"Completed {completed}/{total_combinations} combinations")
                        except Exception as e:
                            print(f"Error with {fp_type}+{clf_name}+{split_method}: {e}")

            if self.results_file.exists():
                results_df = pd.read_csv(self.results_file)
            else:
                results_df = pd.DataFrame(all_results)

        if len(results_df) > 0:
            self.find_and_save_best_models(results_df, top_k=5, metric='AUC')
            print("\nBest models saved and summarized.")

        elapsed_time = time.time() - start_time
        print("\nPipeline completed.")
        print(f"Total results: {len(results_df)}")
        print(f"Results saved to: {self.results_file}")
        print(f"Models saved to:  {self.models_path}")
        print(f"Total time: {elapsed_time / 60:.1f} min")
        return results_df

def _worker_run_combination(packed_args):
    """Worker subprocess: trains all targets for one FP × classifier × split combo."""
    (data_path, results_path, models_path, fps_cache_dir,
     fp_type, clf_name, split_method, tmp_csv_path) = packed_args
    try:
        clf = MultiTaskDrugClassifier(
            data_path=data_path,
            results_path=results_path,
            models_path=models_path,
            fingerprints_cache_dir=fps_cache_dir,
            n_jobs_clf=1,   # parallelism comes from multiple workers, not internal threads
        )
        clf.load_data()
        clf.generate_fingerprints()          # loads from cache — no recompute
        clf.results_file = Path(tmp_csv_path)
        clf._initialize_results_file()
        clf.train_and_evaluate(fp_type, clf_name, split_method)
        return (fp_type, clf_name, split_method, None)
    except Exception:
        import traceback
        return (fp_type, clf_name, split_method, traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description='Train kinase inhibitor prediction models')
    parser.add_argument('--fingerprint', type=str, choices=['ecfp', 'ecfp_count', 'avalon', 'maccs'],
                       help='Specific fingerprint type to use')
    parser.add_argument('--model', type=str,
                       choices=['RandomForest', 'SVM', 'ExtraTrees', 'LogisticRegression', 'KNeighbors', 'GradientBoosting', 'GaussianNB'],
                       help='Specific model to use')
    parser.add_argument('--split_method', type=str,
                       choices=['random', 'scaffold', 'butina', 'umap_clustering'],
                       help='Specific split method to use')
    parser.add_argument('--target', type=str,
                       help='Specific target to train (quoted if contains spaces)')
    parser.add_argument('--train_only', action='store_true',
                       help='Only train models, skip evaluation and results saving')
    parser.add_argument('--fingerprints_cache_dir', type=str, default='./fingerprints_cache',
                       help='Directory to store/load fingerprint cache files (default: ./fingerprints_cache)')
    parser.add_argument('--force_regenerate_fps', action='store_true',
                       help='Ignore existing fingerprint cache and regenerate from scratch')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Run full pipeline in parallel with N worker processes. '
                            'Use -1 for auto (cpu_count - 2). '
                            'Omit to train a specific combination with --fingerprint/--model/--split_method.')

    args = parser.parse_args()

    # When running in parallel mode each worker uses n_jobs=1 (parallelism is across workers).
    # When running a single combination, use all cores inside the classifiers.
    running_parallel = args.max_workers is not None
    n_jobs_clf = 1 if running_parallel else -1

    classifier = MultiTaskDrugClassifier(
        data_path=".",
        results_path="./results_default_14MAY",
        models_path="./models_default_14MAY",
        fingerprints_cache_dir=args.fingerprints_cache_dir,
        n_jobs_clf=n_jobs_clf,
    )

    classifier.load_data()
    classifier.generate_fingerprints(force_regenerate=args.force_regenerate_fps)


    if args.max_workers is not None:
        results_df = classifier.run_complete_pipeline(max_workers=args.max_workers)
        return


    if args.fingerprint and args.model and args.split_method:
        print(f"Training: {args.fingerprint} + {args.model} + {args.split_method}")
        if args.target:
            print(f"   Target: {args.target}")

        results = classifier.train_and_evaluate_specific(
            args.fingerprint, args.model, args.split_method, args.target
        )

        if results:
            print("Training completed successfully")
        else:
            print("Training failed")
            return

    else:
        # COMMENTED OUT: Complete pipeline disabled to prevent time-consuming training of all combinations
        # Use specific arguments (--fingerprint, --model, --split_method) to train individual combinations
        print("No specific training parameters provided.")
        print("   Please specify --fingerprint, --model, and --split_method to train a specific combination.")
        print("   Example: python multitask_classifier_with_default_parameters.py --fingerprint ecfp_count --model RandomForest --split_method random")
        return

    if not args.train_only:
        print("\nSummary of results:")

        if results is None or len(results) == 0:
            print("No results generated. Check logs for errors.")
            return

        results_df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
        print(f"Results DataFrame shape: {results_df.shape}")
        print(f"Results columns: {list(results_df.columns)}")

        required_cols = ['Fingerprint', 'Model', 'Split_Method', 'Accuracy', 'F1', 'AUC']
        missing_cols = [col for col in required_cols if col not in results_df.columns]

        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print("Available columns:", list(results_df.columns))
            return

        try:
            train_results = results_df[results_df['Split'] == 'Train_Train']
            test_results = results_df[results_df['Split'] == 'Train_Test']
            cv_results = results_df[results_df['Split'].str.startswith('Fold_')]

            print("\nTraining Set Performance:")
            if not train_results.empty:
                train_summary = train_results.groupby(['Fingerprint', 'Model', 'Split_Method'])[['Accuracy', 'F1', 'AUC']].mean().round(4)
                print(train_summary)

            print("\nTest Set Performance:")
            if not test_results.empty:
                test_summary = test_results.groupby(['Fingerprint', 'Model', 'Split_Method'])[['Accuracy', 'F1', 'AUC']].mean().round(4)
                print(test_summary)

            print("\nCross-Validation Performance (5-fold, mean +/- SD across targets and folds):")
            if not cv_results.empty:
                cv_per_target = (
                    cv_results.groupby(['Fingerprint', 'Model', 'Split_Method', 'Target'])[['Accuracy', 'F1', 'AUC']]
                    .mean()
                    .reset_index()
                )
                cv_summary = (
                    cv_per_target.groupby(['Fingerprint', 'Model', 'Split_Method'])[['Accuracy', 'F1', 'AUC']]
                    .agg(['mean', 'std'])
                    .round(4)
                )
                cv_summary.columns = [f'{m}_{s}' for m, s in cv_summary.columns]
                print(cv_summary[['AUC_mean', 'AUC_std', 'F1_mean', 'F1_std', 'Accuracy_mean', 'Accuracy_std']])

                cv_summary_file = str(self.results_file).replace('.csv', '_cv_summary.csv')
                cv_summary.reset_index().to_csv(cv_summary_file, index=False)
                print(f"\nCV summary saved to: {cv_summary_file}")

            print("\nSaved Models:")
            saved_models = classifier.list_saved_models()
            if len(saved_models) > 0:
                print(saved_models.to_string(index=False))
            else:
                print("No models were saved.")

        except Exception as e:
            print(f"Error creating summary: {e}")
            print("First few rows of results:")
            print(results_df.head())

if __name__ == "__main__":
    main()
