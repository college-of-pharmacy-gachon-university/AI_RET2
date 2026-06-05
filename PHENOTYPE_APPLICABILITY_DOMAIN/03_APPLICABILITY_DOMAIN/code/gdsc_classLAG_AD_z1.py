"""
CLASS-LAG Applicability Domain — z<=-1.0 threshold (sample_weight run)
------------------------------------------------------------------------
Source data : gdsc_weighted_results_z1_20260521.csv  (z<=-1.0, sample_weight)
Z threshold : -1.0  →  sensitive if z-score > -1.0  (~8% sensitive)
AD method   : CLASS-LAG = min(p, 1-p)   (Sushko et al. 2010)

Changes vs. original gdsc_classLAG_applicability_domain.py:
  - Fixed Z_THRESHOLD = -1.0  (replaces per-cell Youden threshold from old CSV)
  - sample_weight='balanced' added to RandomForest.fit()
  - Cell line list taken from z1 results CSV
  - Output files prefixed gdsc_classLAG_AD_z1_*

Reference:
  Sushko et al. (2010) J. Chem. Inf. Model. 50:2094-2111
  DOI: 10.1021/ci100253r
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


Z_THRESHOLD  = -1.0         # sensitive if z-score > -1.0
TAG          = 'z1'
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
COVERAGES    = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

ROOT = Path(__file__).resolve().parents[3]
RESULTS_CSV  = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA' / 'results' / 'z1.0_weighted' / 'gdsc_weighted_results_20260521_111123.csv'

# Training-folder files (deduplicated, 429 drugs, 108 NSCLC cell lines)
TRAIN_FOLDER = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA' / 'input' / 'z1.0_threshold'
SMILES_CSV   = TRAIN_FOLDER / 'gdsc_nsclc_20May_smiles_mapping.csv'
ZSCORE_CSV   = TRAIN_FOLDER / 'gdsc_nsclc_20May_zscore_matrix.csv'

print(f"[{TAG}] Loading data  (Z_THRESHOLD = {Z_THRESHOLD}) ...")

# Cleaned SMILES mapping (no duplicates — raw CSV has 16 drug-name pairs sharing same SMILES)
smiles_map  = pd.read_csv(SMILES_CSV).set_index('Drug_Name')['SMILES']

# Z-score matrix from training folder: rows=drugs, first col=SMILES, rest=108 NSCLC cell lines
z_raw  = pd.read_csv(ZSCORE_CSV, index_col=0)
zscore = z_raw.drop(columns=['SMILES'])   # rows=drugs, cols=cell lines

cell_lines = list(zscore.columns)
print(f"  Drugs: {len(smiles_map)}  |  Cell lines: {len(cell_lines)}")
print(f"  Source: {TRAIN_FOLDER}")

print("Generating Avalon 2048-bit fingerprints...")
drug_smiles = smiles_map   # already clean
fp_dict = {}
for drug, smi in drug_smiles.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        arr = np.zeros(2048, dtype=np.uint8)
        ConvertToNumpyArray(GetAvalonFP(mol, nBits=2048), arr)
        fp_dict[drug] = arr
print(f"  Valid fingerprints: {len(fp_dict)} / {len(drug_smiles)} drugs")

print(f"\nRunning CLASS-LAG AD ({TAG}) for {len(cell_lines)} cell lines...")

rows         = []
all_classLAG = []

for i, cell in enumerate(cell_lines):
    if cell not in zscore.columns:
        continue

    vals        = zscore[cell].dropna()
    valid_drugs = [d for d in vals.index if d in fp_dict]
    if len(valid_drugs) < 20:
        continue

    X = np.array([fp_dict[d] for d in valid_drugs])
    y = (vals[valid_drugs].values <= Z_THRESHOLD).astype(int)  # sensitive = z <= threshold

    if len(np.unique(y)) < 2:
        continue

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_drugs)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_test)) < 2:
        continue

    sw = compute_sample_weight('balanced', y_train)

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train, sample_weight=sw)

    proba       = rf.predict_proba(X_test)[:, 1]
    classLAG    = np.minimum(proba, 1 - proba)
    overall_auc = roc_auc_score(y_test, proba)

    all_classLAG.extend(classLAG.tolist())

    sort_order = np.argsort(classLAG)
    auc_at_cov = {}
    for cov in COVERAGES:
        n_keep   = max(2, int(np.ceil(cov * len(test_idx))))
        idx_keep = sort_order[:n_keep]
        y_sub    = y_test[idx_keep]
        p_sub    = proba[idx_keep]
        key      = f'AUC_cov{int(cov*100):03d}'
        auc_at_cov[key] = (round(roc_auc_score(y_sub, p_sub), 4)
                           if len(np.unique(y_sub)) >= 2 else float('nan'))

    pct_sens = round(100 * y.mean(), 2)
    row = {
        'Cell_Line'           : cell,
        'N_train'             : len(train_idx),
        'N_test'              : len(test_idx),
        'Pct_sensitive'       : pct_sens,
        'Overall_AUC'         : round(overall_auc, 4),
        'Mean_CLASS_LAG'      : round(float(np.mean(classLAG)), 4),
        'Median_CLASS_LAG'    : round(float(np.median(classLAG)), 4),
        'Pct_confident_20pct' : round(100 * float((classLAG < 0.2).mean()), 2),
        'Pct_confident_30pct' : round(100 * float((classLAG < 0.3).mean()), 2),
    }
    row.update(auc_at_cov)
    rows.append(row)

    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1} / {len(cell_lines)} cell lines...")

df = pd.DataFrame(rows)

out_per  = str(Path(__file__).resolve().parents[1] / 'results' / f'gdsc_classLAG_AD_{TAG}_per_cellline.csv')
out_summ = str(Path(__file__).resolve().parents[1] / 'results' / f'gdsc_classLAG_AD_{TAG}_summary.csv')

df.to_csv(out_per, index=False)
print(f"\nSaved: {out_per}  ({len(df)} cell lines)")

all_classLAG = np.array(all_classLAG)
cov_cols     = [f'AUC_cov{int(c*100):03d}' for c in COVERAGES]
best_col     = df[cov_cols].mean().idxmax()
best_auc     = df[cov_cols].mean().max()
baseline     = df['AUC_cov100'].mean()

print("\n" + "=" * 65)
print(f"CLASS-LAG AD SUMMARY  ({TAG.upper()}: z <= {Z_THRESHOLD})")
print("=" * 65)
print(f"  Cell lines       : {len(df)}")
print(f"  Z_THRESHOLD      : {Z_THRESHOLD}  (sensitive if z-score > {Z_THRESHOLD})")
print(f"  Mean % sensitive : {df['Pct_sensitive'].mean():.1f}%")
print(f"  Fingerprint      : Avalon 2048-bit")
print(f"  Classifier       : RandomForest (sample_weight=balanced)")
print(f"  Split            : random 80/20, seed={RANDOM_STATE}")
print()
print(f"  Overall mean AUC : {df['Overall_AUC'].mean():.4f}")
print(f"  Mean CLASS-LAG   : {all_classLAG.mean():.4f}")
print(f"  % CLASS-LAG < 0.2: {100*(all_classLAG<0.2).mean():.1f}%")
print(f"  % CLASS-LAG < 0.3: {100*(all_classLAG<0.3).mean():.1f}%")
print()
print(f"  {'Coverage':>9}  {'Mean AUC':>10}")
print(f"  {'-'*9}  {'-'*10}")
for cov, col in zip(COVERAGES, cov_cols):
    note = ' ← most confident' if cov <= 0.20 else (' ← all compounds' if cov == 1.0 else '')
    print(f"  {int(cov*100):>8}%  {df[col].mean():>10.4f}{note}")
print()
print(f"  Best AUC (any coverage) : {best_auc:.4f}  ({best_col})")
print(f"  Baseline (100%)         : {baseline:.4f}")
print(f"  Max AUC gain            : {best_auc - baseline:+.4f}")

summary_rows = [
    {'Metric': 'Source script',     'Value': f'gdsc_classLAG_AD_{TAG}.py'},
    {'Metric': 'Source results',    'Value': RESULTS_CSV},
    {'Metric': 'Z_THRESHOLD',       'Value': Z_THRESHOLD},
    {'Metric': 'AD method',         'Value': 'CLASS-LAG = min(p, 1-p)'},
    {'Metric': 'Fingerprint',       'Value': 'Avalon 2048-bit'},
    {'Metric': 'Classifier',        'Value': f'RandomForest n={N_ESTIMATORS} seed={RANDOM_STATE} sample_weight=balanced'},
    {'Metric': 'Cell lines',        'Value': len(df)},
    {'Metric': 'Mean % sensitive',  'Value': round(df['Pct_sensitive'].mean(), 1)},
    {'Metric': 'Overall AUC',       'Value': round(df['Overall_AUC'].mean(), 4)},
    {'Metric': 'Mean CLASS-LAG',    'Value': round(float(all_classLAG.mean()), 4)},
    {'Metric': '% CLASS-LAG < 0.2', 'Value': round(float(100*(all_classLAG<0.2).mean()), 1)},
    {'Metric': '% CLASS-LAG < 0.3', 'Value': round(float(100*(all_classLAG<0.3).mean()), 1)},
    {'Metric': 'Best AUC',          'Value': round(best_auc, 4)},
    {'Metric': 'Best coverage col', 'Value': best_col},
    {'Metric': 'Baseline AUC',      'Value': round(baseline, 4)},
    {'Metric': 'Max AUC gain',      'Value': round(best_auc - baseline, 4)},
]
for cov, col in zip(COVERAGES, cov_cols):
    summary_rows.append({'Metric': f'Mean AUC {int(cov*100)}% coverage', 'Value': round(df[col].mean(), 4)})

pd.DataFrame(summary_rows).to_csv(out_summ, index=False)
print(f"Saved: {out_summ}")
print("=" * 65)
