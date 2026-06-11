# CLASS-LAG applicability domain — z <= 0.0, unweighted RF
# Sushko et al. (2010) J. Chem. Inf. Model. 50:2094-2111

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

Z_THRESHOLD  = 0.0
TAG          = 'z0_unwt'
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
COVERAGES    = list(np.arange(0.01, 0.10, 0.01)) + [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

ROOT         = Path(__file__).resolve().parents[3]
RESULTS_CSV  = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA' / 'results' / 'z0_unweighted' / 'gdsc_unweighted_results_20260522_154422.csv'
TRAIN_FOLDER = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA' / 'input' / 'z0_threshold'
SMILES_CSV   = TRAIN_FOLDER / 'gdsc_multitask_threshold_zeo_smiles_mapping.csv'
ZSCORE_CSV   = TRAIN_FOLDER / 'gdsc_multitask_threshold_zeo_zscore_matrix.csv'

print(f"[{TAG}] loading data ...")

smiles_map  = pd.read_csv(SMILES_CSV).set_index('Drug_Name')['SMILES']
z_raw       = pd.read_csv(ZSCORE_CSV, index_col=0)
zscore      = z_raw.drop(columns=['SMILES'])
cell_lines  = list(zscore.columns)
print(f"  {len(smiles_map)} drugs, {len(cell_lines)} cell lines")

fp_dict = {}
for drug, smi in smiles_map.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        arr = np.zeros(2048, dtype=np.uint8)
        ConvertToNumpyArray(GetAvalonFP(mol, nBits=2048), arr)
        fp_dict[drug] = arr
print(f"  {len(fp_dict)} valid fingerprints")

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
    y = (vals[valid_drugs].values <= Z_THRESHOLD).astype(int)

    if len(np.unique(y)) < 2:
        continue

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_drugs)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_test)) < 2:
        continue

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train)

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

    row = {
        'Cell_Line'           : cell,
        'N_train'             : len(train_idx),
        'N_test'              : len(test_idx),
        'Pct_sensitive'       : round(100 * y.mean(), 2),
        'Overall_AUC'         : round(overall_auc, 4),
        'Mean_CLASS_LAG'      : round(float(np.mean(classLAG)), 4),
        'Median_CLASS_LAG'    : round(float(np.median(classLAG)), 4),
        'Pct_confident_20pct' : round(100 * float((classLAG < 0.2).mean()), 2),
        'Pct_confident_30pct' : round(100 * float((classLAG < 0.3).mean()), 2),
    }
    row.update(auc_at_cov)
    rows.append(row)

    if (i + 1) % 20 == 0:
        print(f"  {i+1} / {len(cell_lines)} done...")

df = pd.DataFrame(rows)

OUT_DIR  = Path(__file__).resolve().parents[1] / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_per  = str(OUT_DIR / f'gdsc_classLAG_AD_{TAG}_per_cellline.csv')
out_summ = str(OUT_DIR / f'gdsc_classLAG_AD_{TAG}_summary.csv')

df.to_csv(out_per, index=False)
print(f"saved: {out_per}  ({len(df)} cell lines)")

all_classLAG = np.array(all_classLAG)
cov_cols     = [f'AUC_cov{int(c*100):03d}' for c in COVERAGES]
best_col     = df[cov_cols].mean().idxmax()
best_auc     = df[cov_cols].mean().max()
baseline     = df['AUC_cov100'].mean()

print(f"\nResults ({TAG}):")
print(f"  cell lines: {len(df)}  |  mean AUC: {df['Overall_AUC'].mean():.4f}")
print(f"  mean CLASS-LAG: {all_classLAG.mean():.4f}  |  <0.2: {100*(all_classLAG<0.2).mean():.1f}%  |  <0.3: {100*(all_classLAG<0.3).mean():.1f}%")
print(f"\n  {'coverage':>9}  {'mean_AUC':>10}")
for cov, col in zip(COVERAGES, cov_cols):
    print(f"  {int(cov*100):>8}%  {df[col].mean():.4f}")
print(f"\n  best: {best_auc:.4f} at {best_col}  (baseline {baseline:.4f}, gain {best_auc - baseline:+.4f})")

summary_rows = [
    {'Metric': 'Source script',     'Value': f'gdsc_classLAG_AD_{TAG}.py'},
    {'Metric': 'Source results',    'Value': RESULTS_CSV},
    {'Metric': 'Z_THRESHOLD',       'Value': Z_THRESHOLD},
    {'Metric': 'AD method',         'Value': 'CLASS-LAG = min(p, 1-p)'},
    {'Metric': 'Fingerprint',       'Value': 'Avalon 2048-bit'},
    {'Metric': 'Classifier',        'Value': f'RandomForest n={N_ESTIMATORS} seed={RANDOM_STATE} unweighted'},
    {'Metric': 'sample_weight',     'Value': 'none (unweighted)'},
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
print(f"saved: {out_summ}")
