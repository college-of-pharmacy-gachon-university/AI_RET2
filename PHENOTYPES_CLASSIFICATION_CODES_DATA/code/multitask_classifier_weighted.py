#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Task Drug Sensitivity Classification Pipeline (GDSC) — Weighted

Extends multitask_classifier_zscore.py with SAURON-RF-inspired class
imbalance correction strategies applied during model training.

Imbalance strategies (--imbalance_strategy):
  none          Original behaviour — no correction during training.
  sample_weight SAURON-RF Eq.1: w_i = N_res/N_sens for sensitive samples,
                w_i = 1 for resistant.  Applied per cell line in fit().
                Classifiers that do not support sample_weight (KNeighbors)
                fall back to unweighted fit.
  upsample      Oversample sensitive drugs (minority class) by drawing with
                replacement until N_sens == N_res before fitting.
  class_weight  sklearn class_weight='balanced' baked into classifier init.

All strategies keep Youden's J post-hoc threshold (complementary correction).

Reference: Lenhof et al. (2022) Sci. Rep. 12:13458
           https://doi.org/10.1038/s41598-022-17609-x
"""

import argparse
import os
import pickle
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


VALID_STRATEGIES = ('none', 'sample_weight', 'upsample', 'class_weight')
_SUPPORTS_SAMPLE_WEIGHT = frozenset({
    'RandomForest', 'ExtraTrees', 'LogisticRegression',
    'SVM', 'GradientBoosting', 'GaussianNB',
})
_SUPPORTS_CLASS_WEIGHT = frozenset({
    'RandomForest', 'ExtraTrees', 'LogisticRegression', 'SVM',
})


def _fp_avalon(name_smi):
    name, smi = name_smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return name, np.array(pyAvalonTools.GetAvalonFP(mol, nBits=2048))
    except Exception:
        pass
    return name, None


def _fp_ecfp(name_smi):
    name, smi = name_smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return name, np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except Exception:
        pass
    return name, None


def _fp_ecfp_count(name_smi):
    name, smi = name_smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp  = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return name, arr
    except Exception:
        pass
    return name, None


def _fp_maccs(name_smi):
    name, smi = name_smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return name, np.array(MACCSkeys.GenMACCSKeys(mol))
    except Exception:
        pass
    return name, None


_FP_FUNCS = {
    'avalon':     _fp_avalon,
    'ecfp':       _fp_ecfp,
    'ecfp_count': _fp_ecfp_count,
    'maccs':      _fp_maccs,
}


def _worker_combination(args):
    warnings.filterwarnings('ignore')
    (data_folder, output_prefix, z_threshold,
     fp_cache_dir, fp_type, clf_name, split_method,
     results_path, models_path, run_cv, n_jobs_clf,
     imbalance_strategy, save_models) = args
    try:
        clf_obj = MultiTaskDrugClassifier(
            results_path=results_path,
            models_path=models_path,
            n_jobs=n_jobs_clf,
            imbalance_strategy=imbalance_strategy,
            save_models=save_models,
        )
        clf_obj.load_data(data_folder, output_prefix, z_threshold=z_threshold,
                          verbose=False)
        cache_path = Path(fp_cache_dir) / f"{fp_type}.joblib"
        clf_obj.fingerprints = {fp_type: joblib.load(cache_path)}
        rows = clf_obj.train_and_evaluate(fp_type, clf_name, split_method,
                                          run_cv=run_cv)
        return fp_type, clf_name, split_method, rows, None
    except Exception:
        return fp_type, clf_name, split_method, [], traceback.format_exc()


class MultiTaskDrugClassifier:

    def __init__(self, results_path="./results_gdsc_weighted",
                 models_path="./models_gdsc_weighted",
                 n_jobs=1,
                 imbalance_strategy='sample_weight',
                 save_models=True):

        if imbalance_strategy not in VALID_STRATEGIES:
            raise ValueError(f"imbalance_strategy must be one of {VALID_STRATEGIES}")

        self.results_path       = Path(results_path)
        self.models_path        = Path(models_path)
        self.imbalance_strategy = imbalance_strategy
        self.save_models        = save_models

        self.results_path.mkdir(parents=True, exist_ok=True)
        if save_models:
            self.models_path.mkdir(parents=True, exist_ok=True)

        self.binary_data = None
        self.smiles_data = {}
        self.fingerprints = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = (self.results_path /
                             f"gdsc_weighted_results_{timestamp}.csv")
        self._results_header_written = False

        self.n_jobs      = n_jobs
        self.classifiers = self._build_classifiers(n_jobs, imbalance_strategy)

    def _build_classifiers(self, n_jobs, strategy):
        cw = 'balanced' if strategy == 'class_weight' else None
        return {
            'RandomForest':
                RandomForestClassifier(random_state=42, n_jobs=n_jobs,
                                       **({'class_weight': cw} if cw else {})),
            'SVM':
                SVC(probability=True, random_state=42, max_iter=10000,
                    **({'class_weight': cw} if cw else {})),
            'ExtraTrees':
                ExtraTreesClassifier(random_state=42, n_jobs=n_jobs,
                                     **({'class_weight': cw} if cw else {})),
            'LogisticRegression':
                LogisticRegression(random_state=42, max_iter=1000, n_jobs=n_jobs,
                                   **({'class_weight': cw} if cw else {})),
            'KNeighbors':       KNeighborsClassifier(n_jobs=n_jobs),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'GaussianNB':       GaussianNB(var_smoothing=1e-2),
        }

    @staticmethod
    def _compute_sample_weights(y_tr):
        n_sens = int((y_tr == 1).sum())
        n_res  = int((y_tr == 0).sum())
        if n_sens == 0 or n_res == 0:
            return np.ones(len(y_tr), dtype=float)
        ratio = n_res / n_sens
        return np.where(y_tr.values == 1, ratio, 1.0).astype(float)

    @staticmethod
    def _upsample(X_sub, y_tr, seed=42):
        idx_res  = np.where(y_tr.values == 0)[0]
        idx_sens = np.where(y_tr.values == 1)[0]
        if len(idx_sens) == 0 or len(idx_sens) >= len(idx_res):
            return X_sub, y_tr.values
        n_needed = len(idx_res) - len(idx_sens)
        rng      = np.random.RandomState(seed)
        extra    = rng.choice(idx_sens, size=n_needed, replace=True)
        all_idx  = np.concatenate([np.arange(len(y_tr)), extra])
        return X_sub[all_idx], y_tr.values[all_idx]

    def load_data(self, data_folder, output_prefix, z_threshold=None, verbose=True):
        data_folder = Path(data_folder)
        smiles_file = data_folder / f"{output_prefix}_smiles_mapping.csv"

        if z_threshold is not None:
            zscore_file = data_folder / f"{output_prefix}_zscore_matrix.csv"
            if not zscore_file.exists():
                raise FileNotFoundError(f"Z-score matrix not found: {zscore_file}")
            df = pd.read_csv(zscore_file, index_col='Drug Name')
            if 'SMILES' in df.columns:
                self.smiles_data = df['SMILES'].dropna().to_dict()
                df = df.drop(columns=['SMILES'])
            binary = df.copy().astype(float)
            for col in binary.columns:
                tested = binary[col].notna()
                binary.loc[tested, col] = (df.loc[tested, col] <= z_threshold).astype(float)
            self.binary_data = binary
        else:
            binary_file = data_folder / f"{output_prefix}_binary_matrix.csv"
            if not binary_file.exists():
                raise FileNotFoundError(f"Binary matrix not found: {binary_file}")
            df = pd.read_csv(binary_file, index_col='Drug Name')
            if 'SMILES' in df.columns:
                self.smiles_data = df['SMILES'].dropna().to_dict()
                df = df.drop(columns=['SMILES'])
            self.binary_data = df.astype(float)

        if not self.smiles_data and smiles_file.exists():
            smiles_df = pd.read_csv(smiles_file)
            id_col = 'Drug_Name' if 'Drug_Name' in smiles_df.columns else smiles_df.columns[0]
            self.smiles_data = {k: v for k, v in
                                zip(smiles_df[id_col], smiles_df['SMILES'])
                                if pd.notna(v) and str(v).strip()}

        if verbose:
            n_drugs, n_targets = self.binary_data.shape
            total_cells = n_drugs * n_targets
            non_nan   = self.binary_data.notna().sum().sum()
            sensitive  = (self.binary_data == 1).sum().sum()
            sens_pct   = 100.0 * sensitive / non_nan if non_nan > 0 else 0.0
            thr_label  = f"Z<={z_threshold}" if z_threshold is not None else "precomputed"
            print(f"  Drugs              : {n_drugs}")
            print(f"  Targets            : {n_targets}")
            print(f"  SMILES             : {len(self.smiles_data)}")
            print(f"  Non-NaN            : {int(non_nan)} / {total_cells} "
                  f"(sparsity {100*(1-non_nan/total_cells):.1f}%)")
            print(f"  Sensitive          : {int(sensitive)} ({sens_pct:.1f}%)  [{thr_label}]")
            print(f"  Imbalance strategy : {self.imbalance_strategy}")

    def generate_fingerprints(self, fp_types=None, n_jobs=-1):
        if fp_types is None:
            fp_types = ['avalon', 'ecfp', 'ecfp_count', 'maccs']
        items = list(self.smiles_data.items())
        for fp in fp_types:
            fn = _FP_FUNCS[fp]
            print(f"  Generating {fp.upper()} ({len(items)} compounds, "
                  f"n_jobs={n_jobs})...", flush=True)
            results = joblib.Parallel(n_jobs=n_jobs, prefer='processes')(
                joblib.delayed(fn)(item) for item in items
            )
            fps = {name: arr for name, arr in results if arr is not None}
            self.fingerprints[fp] = pd.DataFrame.from_dict(fps, orient='index')
            print(f"    {len(self.fingerprints[fp])} compounds OK")

    def _cache_fingerprints(self, cache_dir):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for fp_type, df in self.fingerprints.items():
            out = cache_dir / f"{fp_type}.joblib"
            joblib.dump(df, out, compress=1)
            mb = out.stat().st_size / 1_048_576
            print(f"  Cached {fp_type.upper()} -> {out.name}  ({mb:.1f} MB)")

    def _split_random(self, drugs, test_size=0.2, seed=42):
        return train_test_split(drugs, test_size=test_size, random_state=seed)

    def _split_scaffold(self, drugs, test_size=0.2):
        scaffold_map = {}
        for d in drugs:
            smi = self.smiles_data.get(d, '')
            try:
                mol = Chem.MolFromSmiles(smi)
                sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else 'NONE'
            except Exception:
                sc = 'NONE'
            scaffold_map.setdefault(sc, []).append(d)
        groups = sorted(scaffold_map.values(), key=len, reverse=True)
        target = int(len(drugs) * test_size)
        train_d, test_d = [], []
        for grp in groups:
            (test_d if len(test_d) < target else train_d).extend(grp)
        return train_d, test_d

    def _split_butina(self, drugs, test_size=0.2):
        fps, valid = [], []
        for d in drugs:
            smi = self.smiles_data.get(d, '')
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                valid.append(d)
        if not valid:
            return self._split_random(drugs, test_size)
        n = len(fps)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dist[i, j] = dist[j, i] = 1.0 - s
        n_clusters = max(2, int(n * (1 - test_size)))
        labels = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage='average',
                                         metric='precomputed').fit_predict(dist)
        clusters = {}
        for drug, lbl in zip(valid, labels):
            clusters.setdefault(lbl, []).append(drug)
        groups = sorted(clusters.values(), key=len, reverse=True)
        target = int(len(valid) * test_size)
        train_d, test_d = [], []
        for grp in groups:
            (test_d if len(test_d) < target else train_d).extend(grp)
        train_d.extend(set(drugs) - set(valid))
        return train_d, test_d

    def _split_umap_cluster(self, drugs, X_fp, test_size=0.2, seed=42):
        drug_list = list(drugs)
        Xd = X_fp.loc[X_fp.index.isin(drug_list)].reindex(drug_list).values
        if UMAP_AVAILABLE:
            embedding = UMAP(n_components=2, random_state=seed,
                             n_neighbors=15).fit_transform(Xd)
        else:
            from sklearn.decomposition import PCA
            embedding = PCA(n_components=2, random_state=seed).fit_transform(Xd)
        n_clusters = max(2, int(len(drug_list) * (1 - test_size)))
        labels = KMeans(n_clusters=n_clusters, random_state=seed,
                        n_init='auto').fit_predict(embedding)
        clusters = {}
        for drug, lbl in zip(drug_list, labels):
            clusters.setdefault(int(lbl), []).append(drug)
        groups = sorted(clusters.values(), key=len, reverse=True)
        target = int(len(drug_list) * test_size)
        train_d, test_d = [], []
        for grp in groups:
            (test_d if len(test_d) < target else train_d).extend(grp)
        return train_d, test_d

    @staticmethod
    def _proba_positive(clf, X):
        p = clf.predict_proba(X)
        if p.shape[1] < 2:
            return np.zeros(len(X), dtype=float)
        return np.clip(p[:, 1], 1e-6, 1.0 - 1e-6)

    def _youden_threshold(self, y_true, y_prob):
        try:
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            j    = tpr[1:] - fpr[1:]
            best = float(thr[1 + np.argmax(j)])
            return float(np.clip(best, 0.0, 1.0))
        except Exception:
            return 0.5

    def _evaluate(self, y_true, y_pred, y_prob):
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                tn, fp, fn, tp = ((cm[0,0],0,0,0) if y_true.iloc[0]==0
                                  else (0,0,0,cm[0,0]))
            else:
                tn = fp = fn = tp = 0
        except Exception:
            tn = fp = fn = tp = 0

        def safe(fn_, *a, **kw):
            try:
                return fn_(*a, **kw)
            except Exception:
                return np.nan

        unique = np.unique(y_true)
        return {
            'Accuracy':  safe(accuracy_score, y_true, y_pred),
            'Precision': safe(precision_score, y_true, y_pred, zero_division=0),
            'Recall':    safe(recall_score, y_true, y_pred, zero_division=0),
            'F1':        safe(f1_score, y_true, y_pred, zero_division=0),
            'AUC':   safe(roc_auc_score, y_true, y_prob) if len(unique) >= 2 else np.nan,
            'AUPRC': safe(average_precision_score, y_true, y_prob) if len(unique) >= 2 else np.nan,
            'MCC':       safe(matthews_corrcoef, y_true, y_pred),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        }

    def _train_per_cell_line(self, clf_proto, clf_name,
                              X_train, X_test, y_train_df, y_test_df):
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)

        individual_models = []
        thresholds        = []
        per_cell_results  = []
        strategy          = self.imbalance_strategy

        for cell_line in y_train_df.columns:
            tr_mask = y_train_df[cell_line].notna()
            te_mask = y_test_df[cell_line].notna()
            y_tr    = y_train_df[cell_line][tr_mask].astype(int)
            y_te    = y_test_df[cell_line][te_mask].astype(int)

            if len(y_tr) == 0 or len(y_tr.unique()) < 2:
                individual_models.append(None)
                thresholds.append(0.5)
                per_cell_results.append((cell_line, None, None, None, 0.5))
                continue

            X_tr_cell = X_tr_sc[tr_mask]
            clf = clf_proto.__class__(**clf_proto.get_params())

            if strategy == 'sample_weight' and clf_name in _SUPPORTS_SAMPLE_WEIGHT:
                sw = self._compute_sample_weights(y_tr)
                clf.fit(X_tr_cell, y_tr.values, sample_weight=sw)
            elif strategy == 'upsample':
                X_up, y_up = self._upsample(X_tr_cell, y_tr)
                clf.fit(X_up, y_up)
            else:
                clf.fit(X_tr_cell, y_tr.values)

            individual_models.append(clf)

            if len(y_te) > 0 and len(y_te.unique()) >= 2:
                thr = self._youden_threshold(
                    y_te, self._proba_positive(clf, X_te_sc[te_mask]))
            else:
                thr = 0.5
            thresholds.append(thr)
            per_cell_results.append((cell_line, clf, X_te_sc[te_mask], y_te, thr))

        return individual_models, scaler, thresholds, per_cell_results

    def train_and_evaluate(self, fp_type, clf_name, split_method, run_cv=True):
        X_all = self.fingerprints[fp_type]
        y_all = self.binary_data

        common = sorted(set(X_all.index) & set(y_all.index))
        if not common:
            return []

        X = X_all.loc[common]
        y = y_all.loc[common]

        if split_method == 'random':
            train_d, test_d = self._split_random(common)
        elif split_method == 'scaffold':
            train_d, test_d = self._split_scaffold(common)
        elif split_method == 'butina':
            train_d, test_d = self._split_butina(common)
        elif split_method == 'umap_cluster':
            train_d, test_d = self._split_umap_cluster(common, X)
        else:
            raise ValueError(f"Unknown split method: {split_method}")

        train_d = [d for d in train_d if d in X.index]
        test_d  = [d for d in test_d  if d in X.index]
        if not train_d or not test_d:
            return []

        X_train = X.loc[train_d].values
        X_test  = X.loc[test_d].values
        y_train = y.loc[train_d]
        y_test  = y.loc[test_d]

        clf_proto = self.classifiers[clf_name]
        individual_models, scaler, thresholds, per_cell = \
            self._train_per_cell_line(clf_proto, clf_name,
                                      X_train, X_test, y_train, y_test)

        if self.save_models:
            model_key  = f"{fp_type}_{clf_name}_{split_method}"
            model_data = {
                'models':              individual_models,
                'scaler':              scaler,
                'thresholds':          thresholds,
                'cell_lines':          list(y_test.columns),
                'fingerprint_type':    fp_type,
                'n_features':          X_train.shape[1],
                'sklearn_version':     sklearn.__version__,
                'classifier':          clf_name,
                'split_method':        split_method,
                'imbalance_strategy':  self.imbalance_strategy,
                'timestamp':           datetime.now().isoformat(),
            }
            model_file = self.models_path / f"{model_key}.pkl"
            with open(model_file, 'wb') as fh:
                pickle.dump(model_data, fh)

        def _make_base(cell_line, n_tr, n_te, thr, split_label):
            return {
                'Fingerprint':        fp_type.upper(),
                'Model':              clf_name,
                'Target':             cell_line,
                'N_Features':         X_train.shape[1],
                'N_Train':            n_tr,
                'N_Test':             n_te,
                'Scaling':            'StandardScaler',
                'Threshold':          round(thr, 4),
                'Split_Method':       split_method,
                'Split':              split_label,
                'Imbalance_Strategy': self.imbalance_strategy,
            }

        results = []
        for cell_line, clf, X_te_sub, y_te_sub, thr in per_cell:
            n_tr = int(y_train[cell_line].notna().sum())
            n_te = len(y_te_sub) if y_te_sub is not None else 0
            base = _make_base(cell_line, n_tr, n_te, thr, 'Train_Test')
            if clf is None or y_te_sub is None or len(y_te_sub) == 0:
                metrics = {k: np.nan for k in
                           ['Accuracy','Precision','Recall','F1','AUC','AUPRC',
                            'MCC','TP','TN','FP','FN']}
            else:
                prob = self._proba_positive(clf, X_te_sub)
                metrics = self._evaluate(y_te_sub, (prob >= thr).astype(int), prob)
            results.append({**base, **metrics})

        if run_cv:
            common_arr = np.array(common)
            for fold, (tr_idx, te_idx) in enumerate(
                    KFold(n_splits=5, shuffle=True, random_state=42).split(common)):
                tr_d = common_arr[tr_idx].tolist()
                te_d = common_arr[te_idx].tolist()
                _, _, _, per_cell_cv = self._train_per_cell_line(
                    clf_proto, clf_name,
                    X.loc[tr_d].values, X.loc[te_d].values,
                    y.loc[tr_d],        y.loc[te_d],
                )
                for cell_line, clf_cv, X_te_sub, y_te_sub, thr in per_cell_cv:
                    n_tr = int(y.loc[tr_d, cell_line].notna().sum())
                    n_te = len(y_te_sub) if y_te_sub is not None else 0
                    base = _make_base(cell_line, n_tr, n_te, thr, f'Fold_{fold+1}')
                    if clf_cv is None or y_te_sub is None or len(y_te_sub) == 0:
                        metrics = {k: np.nan for k in
                                   ['Accuracy','Precision','Recall','F1','AUC','AUPRC',
                                    'MCC','TP','TN','FP','FN']}
                    else:
                        prob = self._proba_positive(clf_cv, X_te_sub)
                        metrics = self._evaluate(y_te_sub, (prob >= thr).astype(int), prob)
                    results.append({**base, **metrics})

        return results

    def _save_results(self, rows):
        if not rows:
            return
        pd.DataFrame(rows).to_csv(
            self.results_file, mode='a',
            header=not self._results_header_written, index=False)
        self._results_header_written = True

    def run_pipeline(self, data_folder, output_prefix,
                     fp_types=None, clf_names=None, split_methods=None,
                     run_cv=True, z_threshold=None, max_workers=None,
                     n_jobs_clf=None, fp_gen_jobs=-1):

        t0 = time.time()
        fp_types      = fp_types      or ['avalon', 'ecfp', 'ecfp_count', 'maccs']
        clf_names     = clf_names     or list(self.classifiers.keys())
        split_methods = split_methods or ['random', 'scaffold', 'butina', 'umap_cluster']

        n_cpu = os.cpu_count() or 1
        if max_workers == -1:
            max_workers = n_cpu
        parallel = max_workers is not None and max_workers > 1
        if n_jobs_clf is None:
            n_jobs_clf = 1 if parallel else -1

        total = len(fp_types) * len(clf_names) * len(split_methods)

        print(f"\n{'='*60}")
        print(f"GDSC Multi-Task Classification Pipeline (Weighted)")
        print(f"  CPU cores          : {n_cpu}")
        print(f"  Max workers        : {max_workers if parallel else 1}")
        print(f"  Combinations       : {total}  "
              f"({len(fp_types)} FP x {len(clf_names)} clf x {len(split_methods)} splits)")
        print(f"  CV                 : {'yes (5-fold)' if run_cv else 'no'}")
        print(f"  Imbalance strategy : {self.imbalance_strategy}")
        print(f"  Save models        : {self.save_models}")
        print(f"  Results file       : {self.results_file}")
        print(f"{'='*60}\n")

        print("Loading data...")
        self.load_data(data_folder, output_prefix, z_threshold=z_threshold)
        print("\nGenerating fingerprints...")
        self.generate_fingerprints(fp_types, n_jobs=fp_gen_jobs)

        if parallel:
            fp_cache_dir = self.results_path / "_fp_cache"
            print("\nCaching fingerprints for workers...")
            self._cache_fingerprints(fp_cache_dir)

            combo_args = [
                (str(data_folder), output_prefix, z_threshold,
                 str(fp_cache_dir), fp, clf, split,
                 str(self.results_path), str(self.models_path),
                 run_cv, n_jobs_clf,
                 self.imbalance_strategy, self.save_models)
                for split in split_methods
                for fp    in fp_types
                for clf   in clf_names
            ]

            done = 0
            print(f"\nRunning {total} combinations with {max_workers} workers...\n")
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_worker_combination, args): args
                           for args in combo_args}
                pbar = tqdm(as_completed(futures), total=total,
                            desc="Combinations", unit="combo")
                for future in pbar:
                    fp_, clf_, split_, rows, err = future.result()
                    if err:
                        tqdm.write(f"  ERROR [{fp_}|{clf_}|{split_}]: {err[:200]}")
                    else:
                        self._save_results(rows)
                        done += 1
                    pbar.set_postfix(ok=done, err=total - done -
                                     sum(1 for f in futures if not f.done()))
        else:
            done = 0
            for split in split_methods:
                for fp in fp_types:
                    for clf in clf_names:
                        try:
                            rows = self.train_and_evaluate(fp, clf, split, run_cv=run_cv)
                            self._save_results(rows)
                            done += 1
                            print(f"  [{done}/{total}]  {fp.upper()} | {clf} | {split}  "
                                  f"({len(rows)} rows)")
                        except Exception as exc:
                            print(f"  ERROR [{fp}|{clf}|{split}]: {exc}")
                            traceback.print_exc()

        elapsed = time.time() - t0
        print(f"\nPipeline complete in {elapsed/60:.1f} min.")
        print(f"Results: {self.results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task GDSC classifier with SAURON-RF imbalance correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_folder',         required=True)
    parser.add_argument('--output_prefix',        required=True)
    parser.add_argument('--results_path',         default='./results_gdsc_weighted')
    parser.add_argument('--models_path',          default='./models_gdsc_weighted')
    parser.add_argument('--fingerprints',  nargs='+',
                        default=['avalon', 'ecfp', 'ecfp_count', 'maccs'],
                        choices=['avalon', 'ecfp', 'ecfp_count', 'maccs'])
    parser.add_argument('--classifiers',   nargs='+',
                        default=['RandomForest', 'SVM', 'ExtraTrees',
                                 'LogisticRegression', 'KNeighbors',
                                 'GradientBoosting', 'GaussianNB'],
                        choices=['RandomForest', 'SVM', 'ExtraTrees',
                                 'LogisticRegression', 'KNeighbors',
                                 'GradientBoosting', 'GaussianNB'])
    parser.add_argument('--splits',        nargs='+',
                        default=['random', 'scaffold', 'butina', 'umap_cluster'],
                        choices=['random', 'scaffold', 'butina', 'umap_cluster'])
    parser.add_argument('--z_threshold',   type=float, default=None)
    parser.add_argument('--imbalance_strategy', default='sample_weight',
                        choices=list(VALID_STRATEGIES))
    parser.add_argument('--no_save_models', action='store_true',
                        help="Skip saving model pickles (saves disk space).")
    parser.add_argument('--max_workers',   type=int, default=-1)
    parser.add_argument('--n_jobs_clf',    type=int, default=None)
    parser.add_argument('--fp_gen_jobs',   type=int, default=-1)
    parser.add_argument('--no_cv',         action='store_true')
    args = parser.parse_args()

    classifier = MultiTaskDrugClassifier(
        results_path=args.results_path,
        models_path=args.models_path,
        imbalance_strategy=args.imbalance_strategy,
        save_models=not args.no_save_models,
    )
    classifier.run_pipeline(
        data_folder=args.data_folder,
        output_prefix=args.output_prefix,
        fp_types=args.fingerprints,
        clf_names=args.classifiers,
        split_methods=args.splits,
        run_cv=not args.no_cv,
        z_threshold=args.z_threshold,
        max_workers=args.max_workers,
        n_jobs_clf=args.n_jobs_clf,
        fp_gen_jobs=args.fp_gen_jobs,
    )


if __name__ == '__main__':
    main()
