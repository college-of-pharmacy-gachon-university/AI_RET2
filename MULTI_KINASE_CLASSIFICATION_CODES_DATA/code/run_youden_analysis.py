"""
Kinase Off-Target Classifier: Youden's J Threshold Analysis
Re-run with authoritative input file: multitask_classification_results_default_parameters_20260519_161617.csv
"""
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

warnings.filterwarnings('ignore')
print('All imports OK')

# ROOT = AI-RET_MS_REVISED_FILES_FOR_UPLOAD/ (two levels above this script)
ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'MULTI_KINASE_CLASSIFICATION_CODES_DATA' / 'input'
RESULTS_FILE = ROOT / 'MULTI_KINASE_CLASSIFICATION_CODES_DATA' / 'results' / \
    'multitask_classification_results_default_parameters_20260519_161617.csv'

PIC50_MATRIX = DATA_DIR / 'kinase_multitask_14May_V15_188217_Original_pIC50_matrix.csv'
SMILES_MAP   = DATA_DIR / 'kinase_multitask_14May_V15_188217_Original_smiles_mapping.csv'
YOUDEN_THR   = DATA_DIR / 'kinase_multitask_14May_V15_188217_Original_youden_thresholds.csv'
OUT_CSV      = ROOT / 'MULTI_KINASE_CLASSIFICATION_CODES_DATA' / 'results' / 'kinase_threshold_youden_analysis.csv'

THRESHOLD_P = 0.3
RANDOM_SEED = 42
TEST_SIZE   = 0.2
N_TREES     = 100
RADIUS      = 2
N_BITS      = 2048

print('\nFile inventory:')
for label, path in [
    ('Results CSV (20260519)', RESULTS_FILE),
    ('pIC50 matrix',           PIC50_MATRIX),
    ('SMILES map',             SMILES_MAP),
    ('Youden thresholds',      YOUDEN_THR),
]:
    size_mb = os.path.getsize(path) / 1e6
    print(f'  {label:28s}: {os.path.basename(path)}  ({size_mb:.1f} MB)')

print('\n=== Section 3: Model Selection ===')
results_raw = pd.read_csv(RESULTS_FILE)
print(f'Total rows: {len(results_raw):,}')

test_df = results_raw[results_raw['Split'] == 'Train_Test'].copy()
print(f'Train_Test rows: {len(test_df):,}')

rank_df = (
    test_df.groupby(['Fingerprint', 'Model', 'Split_Method'])['AUC']
    .agg(Mean_AUC='mean', Std_AUC='std', N_targets='count')
    .reset_index()
    .sort_values('Mean_AUC', ascending=False)
    .reset_index(drop=True)
)
print('\nTop 10 combinations:')
print(rank_df.head(10).to_string(index=True))

best_fp    = 'ECFP_COUNT'
best_model = 'RandomForest'
best_split = 'random'

best_df = test_df[
    (test_df['Fingerprint']  == best_fp) &
    (test_df['Model']        == best_model) &
    (test_df['Split_Method'] == best_split)
].copy()

BEST_AUC = best_df['AUC'].mean()
print(f'\nBest combination: {best_fp} + {best_model} (split: {best_split})')
print(f'Mean AUC (from authoritative 20260519 CSV): {BEST_AUC:.6f}')
print(f'AUC range: {best_df["AUC"].min():.4f} – {best_df["AUC"].max():.4f}')

print('\n=== Section 4: Loading V15 Data ===')
smiles_df = pd.read_csv(SMILES_MAP)
print(f'SMILES mapping: {len(smiles_df):,} compounds')
id2smiles = dict(zip(smiles_df['Compound_ID'].astype(str), smiles_df['SMILES']))

thr_df  = pd.read_csv(YOUDEN_THR)
thr_map = dict(zip(thr_df['Target'], thr_df['Threshold']))
print(f'Targets with pIC50 threshold: {len(thr_df)}')

print('Loading pIC50 matrix (~94k rows)...')
pic50_df = pd.read_csv(PIC50_MATRIX)
meta_cols   = ['Compound_ID', 'InChIKey_Block1', 'Binding_Monomer_ID']
target_cols = [c for c in pic50_df.columns if c not in meta_cols]
print(f'Shape: {pic50_df.shape[0]:,} compounds × {len(target_cols)} targets')

def ecfp_count(smiles, radius=RADIUS, n_bits=N_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

print('\n=== Section 6: Per-Target Youden Analysis ===')
print(f'Model: {best_fp} + {best_model} (pre-computed mean AUC = {BEST_AUC:.4f})')
print(f'Split: random 80/20, seed={RANDOM_SEED}')
print('-' * 100)

youden_results = []

for i, target in enumerate(target_cols, 1):
    if target not in thr_map:
        print(f'  [{i:2d}] SKIP {target}: no pIC50 threshold')
        continue

    pic50_thr = thr_map[target]
    subset = pic50_df[['Compound_ID', target]].dropna(subset=[target]).copy()
    subset['Compound_ID'] = subset['Compound_ID'].astype(str)

    fps, labels = [], []
    for _, row in subset.iterrows():
        smi = id2smiles.get(row['Compound_ID'])
        if smi is None:
            continue
        fp_arr = ecfp_count(smi)
        if fp_arr is None:
            continue
        fps.append(fp_arr)
        labels.append(1 if row[target] >= pic50_thr else 0)

    X = np.array(fps)
    y = np.array(labels)

    if len(y) < 50:
        print(f'  [{i:2d}] SKIP {target}: only {len(y)} compounds')
        continue

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_te, y_prob)
    auc_val  = roc_auc_score(y_te, y_prob)
    j_scores = tpr - fpr
    opt_idx  = np.argmax(j_scores)
    opt_thr  = float(thresholds[opt_idx])
    opt_sens = float(tpr[opt_idx])
    opt_spec = float(1.0 - fpr[opt_idx])
    opt_j    = float(j_scores[opt_idx])

    true_active   = (y_te == 1)
    true_inactive = (y_te == 0)
    retained      = (y_prob <= THRESHOLD_P)

    spec_03 = float(retained[true_inactive].mean())   if true_inactive.sum() > 0 else float('nan')
    npv_03  = float(true_inactive[retained].mean())   if retained.sum() > 0      else float('nan')
    sens_03 = float((~retained[true_active]).mean())  if true_active.sum() > 0   else float('nan')
    fer_03  = float((~retained[true_inactive]).mean()) if true_inactive.sum() > 0 else float('nan')
    conservative = THRESHOLD_P <= opt_thr

    print(f'  [{i:2d}] {target[:50]:<50} | AUC={auc_val:.3f} | Youden={opt_thr:.3f} | '
          f'NPV@0.3={npv_03:.3f} Sens@0.3={sens_03:.3f} Spec@0.3={spec_03:.3f} | Conservative={conservative}')

    youden_results.append({
        'Target':                      target,
        'N_compounds':                 len(y),
        'N_train':                     len(y_tr),
        'N_test':                      len(y_te),
        'N_active_test':               int(true_active.sum()),
        'Active_fraction_test':        round(float(true_active.mean()), 3),
        'AUC':                         round(auc_val, 4),
        'Youden_Optimal_Threshold':    round(opt_thr, 4),
        'Youden_J':                    round(opt_j, 4),
        'Sensitivity_at_Youden':       round(opt_sens, 4),
        'Specificity_at_Youden':       round(opt_spec, 4),
        'Specificity_at_0p3':          round(spec_03, 4),
        'False_Exclusion_Rate_at_0p3': round(fer_03, 4),
        'NPV_at_0p3':                  round(npv_03, 4),
        'Sensitivity_at_0p3':          round(sens_03, 4),
        'Is_0p3_below_Youden':         int(conservative),
    })

df = pd.DataFrame(youden_results)
print('-' * 100)
print(f'Done. {len(df)}/23 targets analysed.')

n       = len(df)
n_below = int(df['Is_0p3_below_Youden'].sum())

SEP = '=' * 70
print(f'\n{SEP}')
print('SUMMARY — JUSTIFICATION FOR OFF-TARGET THRESHOLD = 0.3')
print(f'  Dataset  : V15 (94,213 unique compounds, stereo-deduped)')
print(f'  Results  : multitask_classification_results_default_parameters_20260519_161617.csv')
print(f'  Model    : {best_fp} + {best_model} (rank #1 in pre-computed evaluation)')
print(f'  Split    : random 80/20, seed={RANDOM_SEED}')
print(SEP)
print(f'  Classifiers analysed : {n} off-target kinase targets')
print()
print('  Discriminative ability (AUC):')
rep_auc = df['AUC'].mean()
print(f'    Replication AUC : {rep_auc:.4f} ± {df["AUC"].std():.4f}  (range {df["AUC"].min():.4f}–{df["AUC"].max():.4f})')
print(f'    Pre-computed AUC: {BEST_AUC:.4f} (from 20260519 authoritative CSV)')
print()
mean_y = df['Youden_Optimal_Threshold'].mean()
std_y  = df['Youden_Optimal_Threshold'].std()
min_y  = df['Youden_Optimal_Threshold'].min()
max_y  = df['Youden_Optimal_Threshold'].max()
print("  Youden's J optimal PROBABILITY threshold:")
print(f'    {mean_y:.4f} ± {std_y:.4f}  (range {min_y:.4f}–{max_y:.4f})')
print(f'    0.3 is conservative (below Youden) for: {n_below}/{n} classifiers')
print()
mean_sens = df['Sensitivity_at_0p3'].mean()
mean_spec = df['Specificity_at_0p3'].mean()
mean_npv  = df['NPV_at_0p3'].mean()
print(f'  Performance at threshold = {THRESHOLD_P}:')
print(f'    Sensitivity : {mean_sens*100:.1f}%  (expected: 94.5%)')
print(f'    Specificity : {mean_spec*100:.1f}%  (expected: 76.7%)')
print(f'    NPV         : {mean_npv*100:.1f}%  (expected: 93.5%)')
print()
print(f'  Youden mean: {mean_y:.3f} ± {std_y:.3f}  (expected: 0.503 ± 0.075)')
print(f'  Youden range: {min_y:.3f}–{max_y:.3f}  (expected: 0.360–0.690)')
print(SEP)

df.to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}')
print(f'Rows : {len(df)} targets')

print('\n=== MANUSCRIPT VALUE VERIFICATION ===')
print(f'MS claims: mean Youden = 0.503 ± 0.075, range 0.360–0.690')
print(f'Computed : mean Youden = {mean_y:.3f} ± {std_y:.3f}, range {min_y:.3f}–{max_y:.3f}')
print(f'MS claims: all 23/23 classifiers have Youden > 0.3')
print(f'Computed : {n_below}/{n} classifiers have Youden > 0.3')
print(f'MS claims: sensitivity 94.5%, NPV 93.5%, specificity 76.7%')
print(f'Computed : sensitivity {mean_sens*100:.1f}%, NPV {mean_npv*100:.1f}%, specificity {mean_spec*100:.1f}%')
