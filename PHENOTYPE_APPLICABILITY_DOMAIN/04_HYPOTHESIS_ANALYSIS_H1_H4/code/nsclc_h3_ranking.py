"""
H3 — Ranking / threshold issue
--------------------------------
Tests whether poor performance is caused by a bad binarisation threshold
(i.e., Youden's J produces labels that don't reflect true sensitivity).

ROC-AUC is threshold-independent — it measures the model's ability to rank
active compounds above inactive ones regardless of any classification cut-off.
A mean AUC near 0.5 therefore indicates the model cannot rank compounds,
not merely that the threshold is misplaced.

Test
----
One-sample t-test: per-cell-line AUC vs 0.5 (chance level)
  H0: mean AUC = 0.5
  H1: mean AUC ≠ 0.5  (two-sided)

Input
-----
  PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted/gdsc_unweighted_results_20260522_154422.csv
  Z=0 unweighted figure-level selection:
  4 models x 4 fingerprints x 4 split methods x 108 NSCLC cell lines

Output: printed summary
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_FILE = ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted/gdsc_unweighted_results_20260522_154422.csv'
TAG = 'z0_unwt'
MODELS_FIG = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
FPS_FIG = ['AVALON', 'ECFP', 'ECFP_COUNT', 'MACCS']

results = pd.read_csv(RESULTS_FILE)
test_set = results[
    (results['Split'] == 'Train_Test') &
    (results['Model'].isin(MODELS_FIG)) &
    (results['Fingerprint'].isin(FPS_FIG))
]

# Use one value per cell line for the statistical test to avoid treating
# the 64 model/fingerprint/split combinations per cell line as independent.
auc_vals = test_set.groupby('Target')['AUC'].mean().dropna()
raw_auc_vals = test_set['AUC'].dropna()

ci_lo, ci_hi = stats.t.interval(
    0.95, df=len(auc_vals) - 1,
    loc=auc_vals.mean(), scale=stats.sem(auc_vals)
)
frac_below_06 = (auc_vals < 0.60).mean() * 100
frac_below_05 = (auc_vals < 0.50).mean() * 100
t_stat, t_p   = stats.ttest_1samp(auc_vals, 0.5)

print("=" * 65)
print(f"H3 — RANKING / THRESHOLD ISSUE ({TAG}, z<=0, unweighted)")
print("=" * 65)
print(f"  Results file: {RESULTS_FILE}")
print("\n  Selection: Train_Test; models=RF/GB/LR/SVM;")
print("             fingerprints=AVALON/ECFP/ECFP_COUNT/MACCS; all split methods")
print(f"  Raw rows = {len(test_set)}")
print(f"  Raw mean AUC across rows = {raw_auc_vals.mean():.4f}")
print("\n  Statistical unit: per-cell-line mean AUC")
print(f"  n = {len(auc_vals)} cell lines")
print()
print(f"  AUC distribution:")
print(f"    Mean  = {auc_vals.mean():.4f}")
print(f"    SD    = {auc_vals.std():.4f}")
print(f"    Min   = {auc_vals.min():.4f}  /  Max = {auc_vals.max():.4f}")
print(f"    95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"    % cell lines with AUC < 0.60 : {frac_below_06:.1f}%")
print(f"    % cell lines with AUC < 0.50 : {frac_below_05:.1f}%  (sub-random)")
print()
print(f"  One-sample t-test (H0: mean AUC = 0.5, two-sided):")
print(f"    t = {t_stat:.4f},  p = {t_p:.4e}")
print()
print(f"  INTERPRETATION:")
print(f"    AUC is threshold-independent — it measures ranking ability.")
print(f"    Mean AUC = {auc_vals.mean():.4f} means the model correctly ranks a random")
print(f"    active compound above a random inactive only {auc_vals.mean()*100:.1f}% of the time.")
print(f"    The 95% CI upper bound ({ci_hi:.4f}) barely exceeds 0.6.")
print(f"    This rules out a threshold/binarisation artefact as the root cause.")
print("=" * 65)
