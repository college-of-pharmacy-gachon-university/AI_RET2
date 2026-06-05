"""
H2 — Poor data quality / class imbalance
-----------------------------------------
Tests whether near-random AUC is explained by insufficient training data
or class imbalance, rather than a fundamental feature problem.

Tests
-----
H2a  Pearson r: training set size vs mean per-cell-line AUC
H2b  Pearson r: active fraction (class imbalance) vs mean AUC
H2c  Mann-Whitney U: AUC in balanced cell lines vs imbalanced cell lines

Null hypotheses (all):  no association between data quantity/quality and AUC
Alternative: two-sided

Inputs
------
  PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted/gdsc_unweighted_results_20260522_154422.csv
  PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_target_summary.csv

Output: printed summary
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # AI-RET_MS_REVISED_FILES_FOR_UPLOAD/
BASE = ROOT  # retained for compatibility
RESULTS_ALL_FILE = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted/gdsc_unweighted_results_20260522_154422.csv'
CL_SUMMARY_FILE = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_target_summary.csv'
TAG = 'z0_unwt'

results_all = pd.read_csv(RESULTS_ALL_FILE)
cl_summary = pd.read_csv(CL_SUMMARY_FILE)

MODELS_FIG = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
FPS_FIG = ['AVALON', 'ECFP', 'ECFP_COUNT', 'MACCS']

# Match phenotype_figures_analysis.ipynb:
# Z=0 unweighted, Train_Test rows, 4 models x 4 fingerprints x 4 split methods.
tt = results_all[
    (results_all['Split'] == 'Train_Test') &
    (results_all['Model'].isin(MODELS_FIG)) &
    (results_all['Fingerprint'].isin(FPS_FIG))
]
cl_auc = tt.groupby('Target')['AUC'].mean()

cl_n          = cl_summary.set_index('Cell_Line')['N_total']
cl_active_pct = cl_summary.set_index('Cell_Line')['Pct_sensitive']

print("=" * 65)
print(f"H2 — DATA QUALITY / CLASS IMBALANCE ({TAG}, z<=0, unweighted)")
print("=" * 65)
print(f"  Results file : {RESULTS_ALL_FILE}")
print(f"  Summary file : {CL_SUMMARY_FILE}")
print("  Selection    : Z=0 unweighted; Split=Train_Test;")
print("                 Models=RandomForest, GradientBoosting, LogisticRegression, SVM;")
print("                 Fingerprints=AVALON, ECFP, ECFP_COUNT, MACCS; all split methods")
print(f"  Rows used    : {len(tt)}")
print(f"\n  Cell lines with AUC data: {len(cl_auc)}")

common = cl_auc.index.intersection(cl_n.index)
r_size, p_size = stats.pearsonr(cl_n[common], cl_auc[common])
print("\n  H2a — Pearson r: training set size vs mean AUC")
print(f"    n = {len(common)} cell lines")
print(f"    Sample size range: {int(cl_n[common].min())} – {int(cl_n[common].max())}"
      f"  (mean {cl_n[common].mean():.0f})")
print(f"    r = {r_size:.4f},  p = {p_size:.4f}")
if p_size >= 0.05:
    print("    → NOT significant: training set size does NOT explain AUC variation")
else:
    print("    → Significant: training set size IS associated with AUC variation")

common2 = cl_auc.index.intersection(cl_active_pct.index)
r_bal, p_bal = stats.pearsonr(cl_active_pct[common2], cl_auc[common2])
print("\n  H2b — Pearson r: active fraction vs mean AUC")
print(f"    n = {len(common2)} cell lines")
print(f"    Active fraction range: {cl_active_pct[common2].min():.1f}% – "
      f"{cl_active_pct[common2].max():.1f}%  (mean {cl_active_pct[common2].mean():.1f}%)")
print(f"    r = {r_bal:.4f},  p = {p_bal:.4f}")
if p_bal >= 0.05:
    print("    → NOT significant: class imbalance does NOT explain AUC variation")
else:
    print("    → Significant: class imbalance IS associated with AUC variation")

balanced_cl   = cl_active_pct[(cl_active_pct >= 40) & (cl_active_pct <= 60)].index
imbalanced_cl = cl_active_pct[(cl_active_pct < 40)  | (cl_active_pct > 60)].index
bal_auc   = cl_auc[cl_auc.index.isin(balanced_cl)]
imbal_auc = cl_auc[cl_auc.index.isin(imbalanced_cl)]
mw_stat, mw_p = stats.mannwhitneyu(bal_auc, imbal_auc, alternative='two-sided')
print("\n  H2c — Mann-Whitney U: balanced vs imbalanced cell lines")
print(f"    Balanced (40–60% active, n={len(bal_auc)}): mean AUC = {bal_auc.mean():.4f}")
print(f"    Imbalanced            (n={len(imbal_auc)}): mean AUC = {imbal_auc.mean():.4f}")
print(f"    U = {mw_stat:.1f},  p = {mw_p:.4f}")
if mw_p >= 0.05:
    print("    → NOT significant: class balance does NOT explain AUC variation")
else:
    print("    → Significant: class balance IS associated with AUC variation")

print("\n" + "=" * 65)
print("SUMMARY:")
print("  Training-set size shows a weak but statistically significant")
print("  association with mean AUC in this z0 unweighted figure-level")
print("  selection. Percent sensitive is not significantly correlated")
print("  with AUC, and balanced vs imbalanced cell lines do not differ")
print("  significantly by Mann-Whitney testing.")
print("=" * 65)
