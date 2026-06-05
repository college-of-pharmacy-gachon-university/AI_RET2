"""
run_all_AD.py
=============
Orchestrator: verify inputs, run all 10 AD scripts with the ai_ret conda
environment, and collect every output CSV into Applicability_Domain/.

Scripts run (in order)
──────────────────────
  classLAG (fast, ~10-15 min each):
    gdsc_classLAG_AD_z0.py        Z=0   weighted
    gdsc_classLAG_AD_z0_unwt.py   Z=0   unweighted
    gdsc_classLAG_AD_z05.py       Z=-0.5 weighted
    gdsc_classLAG_AD_z075.py      Z=-0.75 weighted
    gdsc_classLAG_AD_z1.py        Z=-1.0 weighted

  Consensus-ensemble (slower, ~30-60 min each):
    gdsc_consensus_AD_z0.py
    gdsc_consensus_AD_z0_unwt.py
    gdsc_consensus_AD_z05.py
    gdsc_consensus_AD_z075.py
    gdsc_consensus_AD_z1.py

Output
──────
  All CSVs → Applicability_Domain/
  Logs     → Applicability_Domain/logs/
  A run-summary table → Applicability_Domain/AD_run_summary.csv

Usage
─────
  conda activate ai_ret
  python run_all_AD.py [--skip-done] [--dry-run]
"""

import os
import sys
import glob
import shutil
import subprocess
import argparse
import time
from datetime import datetime, timedelta

from pathlib import Path
ROOT    = Path(__file__).resolve().parents[3]   # AI-RET_MS_REVISED_FILES_FOR_UPLOAD/
CODE_DIR = Path(__file__).resolve().parents[0]
OUT_DIR  = str(Path(__file__).resolve().parents[1] / 'results')
LOG_DIR  = os.path.join(OUT_DIR, 'logs')
PYTHON   = sys.executable           # same interpreter that launched this script

#  Each entry: (script_filename, TAG, expected_output_files)
SCRIPTS = [
    (
        'gdsc_classLAG_AD_z0.py', 'z0',
        [
            'gdsc_classLAG_AD_z0_per_cellline.csv',
            'gdsc_classLAG_AD_z0_summary.csv',
        ],
    ),
    (
        'gdsc_classLAG_AD_z0_unwt.py', 'z0_unwt',
        [
            'gdsc_classLAG_AD_z0_unwt_per_cellline.csv',
            'gdsc_classLAG_AD_z0_unwt_summary.csv',
        ],
    ),
    (
        'gdsc_classLAG_AD_z05.py', 'z05',
        [
            'gdsc_classLAG_AD_z05_per_cellline.csv',
            'gdsc_classLAG_AD_z05_summary.csv',
        ],
    ),
    (
        'gdsc_classLAG_AD_z075.py', 'z075',
        [
            'gdsc_classLAG_AD_z075_per_cellline.csv',
            'gdsc_classLAG_AD_z075_summary.csv',
        ],
    ),
    (
        'gdsc_classLAG_AD_z1.py', 'z1',
        [
            'gdsc_classLAG_AD_z1_per_cellline.csv',
            'gdsc_classLAG_AD_z1_summary.csv',
        ],
    ),
    (
        'gdsc_consensus_AD_z0.py', 'z0',
        [
            'gdsc_consensus_AD_z0_per_cellline.csv',
            'gdsc_consensus_AD_z0_comparison.csv',
            'gdsc_consensus_AD_z0_summary.csv',
        ],
    ),
    (
        'gdsc_consensus_AD_z0_unwt.py', 'z0_unwt',
        [
            'gdsc_consensus_AD_z0_unwt_per_cellline.csv',
            'gdsc_consensus_AD_z0_unwt_comparison.csv',
            'gdsc_consensus_AD_z0_unwt_summary.csv',
        ],
    ),
    (
        'gdsc_consensus_AD_z05.py', 'z05',
        [
            'gdsc_consensus_AD_z05_per_cellline.csv',
            'gdsc_consensus_AD_z05_comparison.csv',
            'gdsc_consensus_AD_z05_summary.csv',
        ],
    ),
    (
        'gdsc_consensus_AD_z075.py', 'z075',
        [
            'gdsc_consensus_AD_z075_per_cellline.csv',
            'gdsc_consensus_AD_z075_comparison.csv',
            'gdsc_consensus_AD_z075_summary.csv',
        ],
    ),
    (
        'gdsc_consensus_AD_z1.py', 'z1',
        [
            'gdsc_consensus_AD_z1_per_cellline.csv',
            'gdsc_consensus_AD_z1_comparison.csv',
            'gdsc_consensus_AD_z1_summary.csv',
        ],
    ),
]

#  All paths relative to ROOT (AI-RET_MS_REVISED_FILES_FOR_UPLOAD/)
INPUT_FILES = {
    'zeo SMILES'     : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_smiles_mapping.csv',
    'zeo zscore'     : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_zscore_matrix.csv',
    '0.5 SMILES'     : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.5_threshold/gdsc_multitask_threshold_0.5_smiles_mapping.csv',
    '0.5 zscore'     : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.5_threshold/gdsc_multitask_threshold_0.5_zscore_matrix.csv',
    '0.75 SMILES'    : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.75_threshold/gdsc_multitask_threshold_0.75_smiles_mapping.csv',
    '0.75 zscore'    : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.75_threshold/gdsc_multitask_threshold_0.75_zscore_matrix.csv',
    '20May SMILES'   : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z1.0_threshold/gdsc_nsclc_20May_smiles_mapping.csv',
    '20May zscore'   : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z1.0_threshold/gdsc_nsclc_20May_zscore_matrix.csv',
    'results z0 wt'  : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_weighted/gdsc_weighted_results_20260521_154749.csv',
    'results z0 unwt': ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted/gdsc_unweighted_results_20260522_154422.csv',
    'results z05'    : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0.5_weighted/gdsc_weighted_results_20260522_172746.csv',
    'results z075'   : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0.75_weighted/gdsc_weighted_results_20260522_140030.csv',
    'results z1'     : ROOT / 'PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z1.0_weighted/gdsc_weighted_results_20260521_111123.csv',
}


def hms(seconds):
    return str(timedelta(seconds=int(seconds)))


def verify_inputs():
    print("\n" + "=" * 65)
    print("  INPUT FILE VERIFICATION")
    print("=" * 65)
    missing = []
    for label, path in INPUT_FILES.items():
        full = Path(path)
        ok   = full.is_file()
        status = "OK  " if ok else "MISSING"
        print(f"  {status}  {label:15s}  {os.path.basename(full)}")
        if not ok:
            missing.append((label, full))
    if missing:
        print(f"\n  MISSING: {len(missing)} file(s) — aborting.")
        for lbl, p in missing:
            print(f"    {lbl}: {p}")
        sys.exit(1)
    print(f"\n  All {len(INPUT_FILES)} input files verified.\n")


def collect_outputs(script_name, expected_files, dry_run=False):
    """Check expected output CSVs in OUT_DIR (scripts write directly there)."""
    collected, missing = [], []
    for fname in expected_files:
        dst = os.path.join(OUT_DIR, fname)
        if os.path.isfile(dst):
            collected.append(fname)
        else:
            missing.append(fname)
    return collected, missing


def run_script(script_name, expected_files, skip_done=False, dry_run=False):
    script_path = str(CODE_DIR / script_name)
    log_path    = os.path.join(LOG_DIR, script_name.replace('.py', '.log'))

    # Skip if all outputs already exist in OUT_DIR
    if skip_done:
        already = all(
            os.path.isfile(os.path.join(OUT_DIR, f)) for f in expected_files
        )
        if already:
            print(f"  SKIP  {script_name}  (all outputs already in Applicability_Domain/)")
            return 'skipped', 0, []

    print(f"\n{'─'*65}")
    print(f"  RUN   {script_name}")
    print(f"  Log → {log_path}")
    print(f"{'─'*65}")

    if dry_run:
        print("  [DRY RUN — skipping execution]")
        return 'dry-run', 0, expected_files

    t0 = time.time()
    with open(log_path, 'w') as flog:
        flog.write(f"Script : {script_name}\n")
        flog.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        flog.write("=" * 65 + "\n\n")
        flog.flush()

        proc = subprocess.run(
            [PYTHON, script_path],
            cwd=str(CODE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        flog.write(proc.stdout)
        flog.write(f"\n\n{'='*65}\n")
        flog.write(f"Return code : {proc.returncode}\n")
        flog.write(f"Elapsed     : {hms(time.time() - t0)}\n")

    elapsed = time.time() - t0

    # Print last 20 lines of output for visibility
    lines = proc.stdout.strip().splitlines()
    for line in lines[-20:]:
        print(f"    {line}")

    if proc.returncode != 0:
        print(f"\n  FAILED  (exit {proc.returncode})  elapsed={hms(elapsed)}")
        print(f"    Full log: {log_path}")
        return 'failed', elapsed, []

    # Collect outputs
    collected, missing_out = collect_outputs(script_name, expected_files)
    print(f"\n  OK  elapsed={hms(elapsed)}")
    print(f"    Collected {len(collected)} file(s) → Applicability_Domain/")
    if missing_out:
        print(f"    WARNING: {len(missing_out)} expected output(s) not found: {missing_out}")

    return 'ok', elapsed, collected


def main():
    parser = argparse.ArgumentParser(description='Run all AD scripts and collect outputs.')
    parser.add_argument('--skip-done', action='store_true',
                        help='Skip scripts whose output files already exist in Applicability_Domain/')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify inputs and print plan without running scripts')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    verify_inputs()

    print("=" * 65)
    print("  RUN PLAN")
    print("=" * 65)
    for script, tag, outputs in SCRIPTS:
        status = ''
        if args.skip_done and all(
            os.path.isfile(os.path.join(OUT_DIR, f)) for f in outputs
        ):
            status = '  [will skip — done]'
        print(f"  {script:<35s}  TAG={tag}{status}")
        for f in outputs:
            print(f"    → {f}")
    print()

    if args.dry_run:
        print("  [DRY RUN — no scripts will be executed]")
        return

    summary_rows = []
    t_total_start = time.time()

    for script, tag, expected_outputs in SCRIPTS:
        status, elapsed, collected = run_script(
            script, expected_outputs,
            skip_done=args.skip_done,
            dry_run=args.dry_run,
        )
        summary_rows.append({
            'Script'        : script,
            'TAG'           : tag,
            'Status'        : status,
            'Elapsed'       : hms(elapsed) if elapsed > 0 else '-',
            'Files_collected': len(collected),
            'Output_files'  : '; '.join(collected),
        })

    total_elapsed = time.time() - t_total_start

    import csv
    summary_path = os.path.join(OUT_DIR, 'AD_run_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    ok      = sum(1 for r in summary_rows if r['Status'] == 'ok')
    skipped = sum(1 for r in summary_rows if r['Status'] == 'skipped')
    failed  = sum(1 for r in summary_rows if r['Status'] == 'failed')
    n_files = sum(r['Files_collected'] for r in summary_rows)

    print("\n" + "=" * 65)
    print("  COMPLETED")
    print("=" * 65)
    print(f"  Scripts run    : {ok}")
    print(f"  Skipped        : {skipped}")
    print(f"  Failed         : {failed}")
    print(f"  Files collected: {n_files}")
    print(f"  Total elapsed  : {hms(total_elapsed)}")
    print(f"  Output folder  : {OUT_DIR}/")
    print(f"  Log folder     : {LOG_DIR}/")
    print(f"  Run summary    : {summary_path}")

    print("\n  Files in Applicability_Domain/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.csv'):
            size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
            print(f"    {f:<55s}  {size_kb:>5} KB")
    print("=" * 65)


if __name__ == '__main__':
    main()
