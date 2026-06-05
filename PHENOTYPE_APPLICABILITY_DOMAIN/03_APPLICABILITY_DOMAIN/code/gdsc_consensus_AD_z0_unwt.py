"""
CONS-STD-PROB Applicability Domain — z<=0.0 threshold (unweighted run)
-----------------------------------------------------------------------
Source data : gdsc_unweighted_results_20260522_154422.csv  (z<=0.0, unweighted, deduplicated)
Z threshold : 0.0  →  sensitive if z-score <= 0.0  (~38% sensitive)

Changes vs. gdsc_consensus_AD_z0.py (weighted):
  - sample_weight removed — plain clf.fit(X, y) with no class correction
  - ROOT = Path(__file__).resolve().parents[3]
RESULTS_CSV updated to unweighted deduplicated result file
  - TAG = 'z0_unwt'
  - Output files prefixed gdsc_consensus_AD_z0_unwt_*

Ensemble: 3 FP (AVALON, ECFP, MACCS) x 4 algorithms (RF, LR, SVM, XGB) = 12 models
AD methods: CLASS-LAG, CONS-STD, CONS-STD-PROB (Sushko et al. 2010)

Reference:
  Sushko et al. (2010) J. Chem. Inf. Model. 50:2094-2111
  DOI: 10.1021/ci100253r
"""

import os
import warnings
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import norm
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


Z_THRESHOLD  = 0.0
TAG          = 'z0_unwt'
TEST_SIZE    = 0.2
RANDOM_STATE = 42
COVERAGES    = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

N_CORES      = multiprocessing.cpu_count()
N_JOBS_INNER = 3
N_JOBS_OUTER = max(1, N_CORES // N_JOBS_INNER)
MIN_MODELS   = 8

RESULTS_CSV  = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA' / 'results' / 'z0_unweighted' / 'gdsc_unweighted_results_20260522_154422.csv'

# Training-folder files (deduplicated, 429 drugs, 108 NSCLC cell lines)
TRAIN_FOLDER = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA' / 'input' / 'z0_threshold'
SMILES_CSV   = TRAIN_FOLDER / 'gdsc_multitask_threshold_zeo_smiles_mapping.csv'
ZSCORE_CSV   = TRAIN_FOLDER / 'gdsc_multitask_threshold_zeo_zscore_matrix.csv'

ALGO_NAMES = ['RF', 'LR', 'SVM', 'XGB']
FP_NAMES   = ['AVALON', 'ECFP', 'MACCS']
MODEL_KEYS = [f'{fp}_{alg}' for fp in FP_NAMES for alg in ALGO_NAMES]


def avalon_fp(mol, n=2048):
    arr = np.zeros(n, dtype=np.uint8)
    ConvertToNumpyArray(GetAvalonFP(mol, nBits=n), arr)
    return arr

def ecfp4_fp(mol, n=2048):
    arr = np.zeros(n, dtype=np.uint8)
    ConvertToNumpyArray(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n), arr)
    return arr

def maccs_fp(mol):
    arr = np.zeros(167, dtype=np.uint8)
    ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
    return arr

FP_FUNCS = {'AVALON': avalon_fp, 'ECFP': ecfp4_fp, 'MACCS': maccs_fp}


def _cons_std_prob(p_mean, p_std):
    EPS = 1e-9
    return np.where(
        p_std < EPS,
        np.minimum(p_mean, 1.0 - p_mean),
        np.minimum(
            norm.cdf((0.5 - p_mean) / p_std),
            1.0 - norm.cdf((0.5 - p_mean) / p_std)
        )
    )


def process_cell(cell, zscore_col, z_threshold, fp_bank, n_jobs_inner):
    vals        = zscore_col.dropna()
    valid_drugs = [d for d in vals.index
                   if all(d in fp_bank[fp] for fp in FP_NAMES)]
    if len(valid_drugs) < 20:
        return None

    y = (vals[valid_drugs].values <= z_threshold).astype(int)
    if len(np.unique(y)) < 2:
        return None

    train_idx, test_idx = train_test_split(
        np.arange(len(valid_drugs)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_test)) < 2:
        return None

    # Per-fingerprint arrays
    fp_arrays = {}
    fp_scaled = {}
    for fp in FP_NAMES:
        X = np.array([fp_bank[fp][d] for d in valid_drugs], dtype=np.float32)
        Xtr, Xte = X[train_idx], X[test_idx]
        fp_arrays[fp] = (Xtr, Xte)
        sc = StandardScaler()
        fp_scaled[fp] = (sc.fit_transform(Xtr), sc.transform(Xte))

    def build_clf(algo):
        if   algo == 'RF':
            return RandomForestClassifier(
                n_estimators=100, max_depth=20, min_samples_split=2,
                random_state=RANDOM_STATE, n_jobs=n_jobs_inner)
        elif algo == 'LR':
            return LogisticRegression(
                max_iter=1000, C=1.0, random_state=RANDOM_STATE, n_jobs=n_jobs_inner)
        elif algo == 'SVM':
            return SVC(kernel='rbf', C=1.0, gamma='scale',
                       probability=True, random_state=RANDOM_STATE)
        elif algo == 'XGB':
            return XGBClassifier(
                n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                eval_metric='logloss', verbosity=0, n_jobs=n_jobs_inner)

    probas = {}
    aucs   = {}
    for fp in FP_NAMES:
        Xtr_raw, Xte_raw = fp_arrays[fp]
        Xtr_sc,  Xte_sc  = fp_scaled[fp]
        for algo in ALGO_NAMES:
            key  = f'{fp}_{algo}'
            Xtr  = Xtr_sc if algo in ('LR', 'SVM') else Xtr_raw
            Xte  = Xte_sc if algo in ('LR', 'SVM') else Xte_raw
            try:
                clf = build_clf(algo)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    clf.fit(Xtr, y_train)
                p = clf.predict_proba(Xte)[:, 1]
                probas[key] = p
                aucs[key]   = roc_auc_score(y_test, p)
            except Exception:
                pass

    if len(probas) < MIN_MODELS:
        return None

    proba_mat = np.stack(list(probas.values()), axis=0)
    p_mean    = proba_mat.mean(axis=0)
    p_std     = proba_mat.std(axis=0, ddof=1)

    classLAG  = np.minimum(p_mean, 1.0 - p_mean)
    cons_std  = p_std
    cstd_prob = _cons_std_prob(p_mean, p_std)

    overall_auc = roc_auc_score(y_test, p_mean)

    def auc_at_coverage(dm, prefix):
        order = np.argsort(dm)
        out   = {}
        for cov in COVERAGES:
            n_keep   = max(2, int(np.ceil(cov * len(test_idx))))
            idx_keep = order[:n_keep]
            y_sub    = y_test[idx_keep]
            p_sub    = p_mean[idx_keep]
            key      = f'{prefix}_cov{int(cov*100):03d}'
            out[key] = (round(roc_auc_score(y_sub, p_sub), 4)
                        if len(np.unique(y_sub)) >= 2 else float('nan'))
        return out

    row = {
        'Cell_Line'          : cell,
        'N_train'            : len(train_idx),
        'N_test'             : len(test_idx),
        'N_models'           : len(probas),
        'Pct_sensitive'      : round(100 * y.mean(), 2),
        'Consensus_AUC'      : round(overall_auc, 4),
        **{f'AUC_{k}': round(v, 4) for k, v in aucs.items()},
        'Mean_CLASS_LAG'     : round(float(classLAG.mean()), 4),
        'Mean_CONS_STD'      : round(float(cons_std.mean()), 4),
        'Mean_CONS_STD_PROB' : round(float(cstd_prob.mean()), 4),
        'Pct_AD_STD_lt010'   : round(100 * float((cons_std < 0.10).mean()), 2),
        'Pct_AD_STD_lt020'   : round(100 * float((cons_std < 0.20).mean()), 2),
        '_classLAG'          : classLAG.tolist(),
        '_cons_std'          : cons_std.tolist(),
        '_cstd_prob'         : cstd_prob.tolist(),
    }
    row.update(auc_at_coverage(classLAG,  'CL'))
    row.update(auc_at_coverage(cons_std,  'CS'))
    row.update(auc_at_coverage(cstd_prob, 'CSP'))
    return row


print(f"[{TAG}] Loading data  (Z_THRESHOLD = {Z_THRESHOLD}) ...")

smiles_map = pd.read_csv(SMILES_CSV).set_index('Drug_Name')['SMILES']
z_raw      = pd.read_csv(ZSCORE_CSV, index_col=0)
zscore     = z_raw.drop(columns=['SMILES'])
cell_lines = list(zscore.columns)
print(f"  Drugs: {len(smiles_map)}  |  Cell lines: {len(cell_lines)}")
print(f"  Source: {TRAIN_FOLDER}")

print("Generating AVALON (2048-bit), ECFP4 (2048-bit), MACCS (167-bit) fingerprints...")
fp_bank = {fp: {} for fp in FP_NAMES}
n_fail  = 0
for drug, smi in smiles_map.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp_bank['AVALON'][drug] = avalon_fp(mol)
        fp_bank['ECFP'][drug]   = ecfp4_fp(mol)
        fp_bank['MACCS'][drug]  = maccs_fp(mol)
    else:
        n_fail += 1
print(f"  Valid: {len(fp_bank['AVALON'])} / {len(smiles_map)} drugs  (failed: {n_fail})")

print(f"\nRunning consensus AD ({TAG}) on {len(cell_lines)} cell lines")
print(f"  Models/cell: {len(FP_NAMES)} FP × {len(ALGO_NAMES)} algo = {len(MODEL_KEYS)}")
print(f"  CPU cores  : {N_CORES}  ({N_JOBS_OUTER} outer × {N_JOBS_INNER} inner)")

results_list = Parallel(n_jobs=N_JOBS_OUTER, verbose=5)(
    delayed(process_cell)(cell, zscore[cell], Z_THRESHOLD, fp_bank, N_JOBS_INNER)
    for cell in cell_lines
)

rows = [r for r in results_list if r is not None]
print(f"\n  Completed: {len(rows)} / {len(cell_lines)} cell lines")

all_clag, all_constd, all_cstdprob = [], [], []
for r in rows:
    all_clag.extend(r.pop('_classLAG'))
    all_constd.extend(r.pop('_cons_std'))
    all_cstdprob.extend(r.pop('_cstd_prob'))

all_clag     = np.array(all_clag)
all_constd   = np.array(all_constd)
all_cstdprob = np.array(all_cstdprob)

df = pd.DataFrame(rows)

out_per  = str(Path(__file__).resolve().parents[1] / 'results' / f'gdsc_consensus_AD_{TAG}_per_cellline.csv')
out_comp = str(Path(__file__).resolve().parents[1] / 'results' / f'gdsc_consensus_AD_{TAG}_comparison.csv')
out_summ = str(Path(__file__).resolve().parents[1] / 'results' / f'gdsc_consensus_AD_{TAG}_summary.csv')

df.to_csv(out_per, index=False)
print(f"Saved: {out_per}  ({len(df)} cell lines)")

cl_cols  = [f'CL_cov{int(c*100):03d}'  for c in COVERAGES]
cs_cols  = [f'CS_cov{int(c*100):03d}'  for c in COVERAGES]
csp_cols = [f'CSP_cov{int(c*100):03d}' for c in COVERAGES]

comp = pd.DataFrame([
    {
        'Coverage_pct'           : int(cov * 100),
        'Mean_AUC_CLASS_LAG'     : round(df[cl_c].mean(), 4),
        'Mean_AUC_CONS_STD'      : round(df[cs_c].mean(), 4),
        'Mean_AUC_CONS_STD_PROB' : round(df[csp_c].mean(), 4),
        'N_valid_celllines'      : df[csp_c].notna().sum(),
    }
    for cov, cl_c, cs_c, csp_c in zip(COVERAGES, cl_cols, cs_cols, csp_cols)
])
comp.to_csv(out_comp, index=False)
print(f"Saved: {out_comp}")

baseline_auc = df['CSP_cov100'].mean()
best_csp     = df[csp_cols].mean().max()
best_csp_col = df[csp_cols].mean().idxmax()
best_cs      = df[cs_cols].mean().max()
best_cl      = df[cl_cols].mean().max()

print("\n" + "=" * 72)
print(f"CONSENSUS AD SUMMARY  ({TAG.upper()}: z <= {Z_THRESHOLD})")
print("=" * 72)
print(f"\n  Source results   : {RESULTS_CSV}")
print(f"  Z_THRESHOLD      : {Z_THRESHOLD}  (sensitive if z-score <= {Z_THRESHOLD})")
print(f"  Mean % sensitive : {df['Pct_sensitive'].mean():.1f}%")
print(f"  Cell lines       : {len(df)}")
print(f"  Fingerprints     : {', '.join(FP_NAMES)}")
print(f"  Algorithms       : {', '.join(ALGO_NAMES)}")
print(f"  Models/cell line : {len(MODEL_KEYS)}")
print(f"  sample_weight    : none (unweighted)")
print()
print(f"  Consensus AUC (all)    : {df['Consensus_AUC'].mean():.4f}")
print(f"  Mean CONS-STD          : {all_constd.mean():.4f}")
print(f"  % CONS-STD < 0.10      : {100*(all_constd<0.10).mean():.1f}%")
print(f"  % CONS-STD < 0.20      : {100*(all_constd<0.20).mean():.1f}%")
print()
print(f"  {'Coverage':>9}  {'CLASS-LAG':>11}  {'CONS-STD':>10}  {'CONS-STD-PROB':>14}")
print(f"  {'-'*9}  {'-'*11}  {'-'*10}  {'-'*14}")
for cov, cl_c, cs_c, csp_c in zip(COVERAGES, cl_cols, cs_cols, csp_cols):
    note = ' ← most confident' if cov <= 0.20 else (' ← all compounds' if cov == 1.00 else '')
    print(f"  {int(cov*100):>8}%  {df[cl_c].mean():>11.4f}  "
          f"{df[cs_c].mean():>10.4f}  {df[csp_c].mean():>14.4f}{note}")

print()
print(f"  Best CLASS-LAG         : {best_cl:.4f}")
print(f"  Best CONS-STD          : {best_cs:.4f}")
print(f"  Best CONS-STD-PROB     : {best_csp:.4f}  ({best_csp_col})")
print(f"  Baseline (100%)        : {baseline_auc:.4f}")
print(f"  Max AUC gain           : {best_csp - baseline_auc:+.4f}")

summary_rows = [
    {'Metric': 'Source script',           'Value': f'gdsc_consensus_AD_{TAG}.py'},
    {'Metric': 'Source results',          'Value': RESULTS_CSV},
    {'Metric': 'Z_THRESHOLD',             'Value': Z_THRESHOLD},
    {'Metric': 'AD method',               'Value': 'CONS-STD-PROB (Sushko 2010)'},
    {'Metric': 'Fingerprints',            'Value': ', '.join(FP_NAMES)},
    {'Metric': 'Algorithms',              'Value': ', '.join(ALGO_NAMES)},
    {'Metric': 'Ensemble size',           'Value': f'{len(MODEL_KEYS)} ({len(FP_NAMES)} FP x {len(ALGO_NAMES)} algo)'},
    {'Metric': 'sample_weight',           'Value': 'none (unweighted)'},
    {'Metric': 'Cell lines',              'Value': len(df)},
    {'Metric': 'Mean % sensitive',        'Value': round(df['Pct_sensitive'].mean(), 1)},
    {'Metric': 'Consensus AUC (100%)',    'Value': round(df['Consensus_AUC'].mean(), 4)},
    {'Metric': 'Mean CONS-STD',           'Value': round(float(all_constd.mean()), 4)},
    {'Metric': '% CONS-STD < 0.10',       'Value': round(float(100*(all_constd<0.10).mean()), 1)},
    {'Metric': '% CONS-STD < 0.20',       'Value': round(float(100*(all_constd<0.20).mean()), 1)},
    {'Metric': 'Best CLASS-LAG AUC',      'Value': round(best_cl, 4)},
    {'Metric': 'Best CONS-STD AUC',       'Value': round(best_cs, 4)},
    {'Metric': f'Best CONS-STD-PROB AUC ({best_csp_col})', 'Value': round(best_csp, 4)},
    {'Metric': 'Baseline AUC (100%)',     'Value': round(baseline_auc, 4)},
    {'Metric': 'Max AUC gain',            'Value': round(best_csp - baseline_auc, 4)},
]
for cov, cl_c, cs_c, csp_c in zip(COVERAGES, cl_cols, cs_cols, csp_cols):
    pct = int(cov * 100)
    summary_rows += [
        {'Metric': f'Mean AUC {pct}% — CLASS-LAG',     'Value': round(df[cl_c].mean(), 4)},
        {'Metric': f'Mean AUC {pct}% — CONS-STD',      'Value': round(df[cs_c].mean(), 4)},
        {'Metric': f'Mean AUC {pct}% — CONS-STD-PROB', 'Value': round(df[csp_c].mean(), 4)},
    ]

pd.DataFrame(summary_rows).to_csv(out_summ, index=False)
print(f"Saved: {out_summ}")
print("=" * 72)
