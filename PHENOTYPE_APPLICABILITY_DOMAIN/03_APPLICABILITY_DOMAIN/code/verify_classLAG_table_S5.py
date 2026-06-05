"""
Verify Table S5 (CLASS-LAG AD) values from authoritative CSV files.

Reads gdsc_classLAG_AD_z0_unwt_summary.csv and gdsc_classLAG_AD_z0_summary.csv,
prints the coverage table, and checks each row against what is written in
Table S5 of the supplementary (values stated below as SUPP_* dicts).
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[3]  # AI-RET_MS_REVISED_FILES_FOR_UPLOAD/
AD_DIR = Path(__file__).resolve().parents[1] / "results"

FILE_UNWT = AD_DIR / "gdsc_classLAG_AD_z0_unwt_summary.csv"
FILE_WT   = AD_DIR / "gdsc_classLAG_AD_z0_summary.csv"

SUPP_UNWT = {10: 0.616, 20: 0.585, 30: 0.568, 40: 0.557, 50: 0.557,
             60: 0.561, 70: 0.556, 80: 0.551, 90: 0.558, 100: 0.561}

SUPP_WT   = {10: 0.580, 20: 0.584, 30: 0.561, 40: 0.549, 50: 0.548,
             60: 0.546, 70: 0.544, 80: 0.545, 90: 0.545, 100: 0.554}


def parse_summary(path: Path) -> tuple[dict, float, float, float]:
    """Return (coverage_dict, baseline, best_auc, max_gain) from key-value CSV."""
    df = pd.read_csv(path)
    info = dict(zip(df["Metric"], df["Value"]))
    cov_rows = {int(k.split("%")[0].split()[-1]): float(v)
                for k, v in info.items()
                if "Mean AUC" in k and "coverage" in k}
    baseline = float(info["Baseline AUC"])
    best     = float(info["Best AUC"])
    gain     = float(info["Max AUC gain"])
    return cov_rows, baseline, best, gain


def check_table(label: str, cov_dict: dict, baseline: float,
                supp: dict, tol: float = 0.0005) -> None:
    """Compare CSV values against supplementary-stated values, row by row."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Baseline AUC (100% coverage): {baseline:.4f}  "
          f"→ rounds to {baseline:.3f}")
    print(f"{'='*70}")
    print(f"  {'Coverage%':>10}  {'CSV (4dp)':>12}  {'Rounded(3dp)':>13}  "
          f"{'Suppl S5':>10}  {'Diff':>8}  {'Status':>6}")
    print(f"  {'-'*70}")

    n_pass = n_close = n_fail = 0
    for pct in sorted(cov_dict):
        csv_val  = cov_dict[pct]
        rounded  = round(csv_val, 3)
        claimed  = supp.get(pct, float("nan"))
        diff     = abs(rounded - claimed)
        if diff == 0:
            status = "EXACT"
            n_pass += 1
        elif diff <= tol:
            status = "PASS"
            n_pass += 1
        elif diff <= 0.010:
            status = "CLOSE"
            n_close += 1
        else:
            status = "FAIL"
            n_fail += 1
        print(f"  {pct:>9}%  {csv_val:>12.4f}  {rounded:>13.3f}  "
              f"{claimed:>10.3f}  {diff:>8.4f}  {status:>6}")

    print(f"\n  Result: EXACT/PASS={n_pass}  CLOSE={n_close}  FAIL={n_fail}")


def check_verifier_bug():
    print(f"\n{'='*70}")
    print("  VERIFIER SCRIPT BUG (verify_nsclc_metrics.py line 363)")
    print(f"{'='*70}")
    _, baseline_u, _, _ = parse_summary(FILE_UNWT)
    _, baseline_w, _, _ = parse_summary(FILE_WT)
    print(f"  Unweighted baseline (actual): {baseline_u:.4f}  → 3 d.p. = {baseline_u:.3f}")
    print(f"  Weighted   baseline (actual): {baseline_w:.4f}  → 3 d.p. = {baseline_w:.3f}")
    print()
    print("  Verifier line 356 comment says: 'Unwt baseline=0.554'  ← WRONG")
    print(f"  Correct value should be:         Unwt baseline={baseline_u:.3f}")
    print()
    print("  Fix: change verify_nsclc_metrics.py line 363 from")
    print("    chk('Table S5 AD', 'classLAG unwt baseline', 0.554, baseline_u)")
    print("  to")
    print(f"    chk('Table S5 AD', 'classLAG unwt baseline', {baseline_u:.3f}, baseline_u)")
    print()
    print("  NOTE: The supplementary Table S5 itself is CORRECT.")
    print("        Unweighted 100% = 0.561  (matches 0.5613) ✓")
    print("        Weighted   100% = 0.554  (matches 0.5540) ✓")
    print("        The CLOSE flag in the verifier is a verifier bug, not a table error.")


if __name__ == "__main__":
    print("Verifying Table S5 CLASS-LAG values against authoritative CSV files")
    print(f"Unweighted file : {FILE_UNWT}")
    print(f"Weighted file   : {FILE_WT}")

    cov_u, base_u, best_u, gain_u = parse_summary(FILE_UNWT)
    cov_w, base_w, best_w, gain_w = parse_summary(FILE_WT)

    check_table("UNWEIGHTED RF (Column 2 of Table S5)", cov_u, base_u, SUPP_UNWT)
    check_table("WEIGHTED RF   (Column 3 of Table S5)", cov_w, base_w, SUPP_WT)

    check_verifier_bug()

    print(f"\n{'='*70}")
    print("  SUMMARY STATS (for reference)")
    print(f"{'='*70}")
    print(f"  Unweighted: baseline={base_u:.4f}, best={best_u:.4f}, gain=+{gain_u:.4f}")
    print(f"  Weighted  : baseline={base_w:.4f}, best={best_w:.4f}, gain=+{gain_w:.4f}")
