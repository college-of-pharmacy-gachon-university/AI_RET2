#!/usr/bin/env python3
"""GDSC dataset transformation: long format to wide multi-task Z-score / binary matrices."""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available. InChIKey-based deduplication will be skipped.")

warnings.filterwarnings('ignore')


class GDSCDatasetTransformer:
    """Transform GDSC long-format data into wide multi-task format with full output suite."""

    def __init__(self, input_file, output_prefix="gdsc_multitask", sensitivity_threshold=-2.0):
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.sensitivity_threshold = sensitivity_threshold
        self.raw_data = None
        self.zscore_matrix = None
        self.smiles_map = {}
        self.processed_stats = {}
        self.stage_stats = {}

    # SMILES helpers

    def _smiles_to_inchikey(self, smiles):
        """Return full InChIKey for a SMILES string, or None if invalid."""
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            inchi = MolToInchi(mol)
            if inchi is None:
                return None
            return InchiToInchiKey(inchi)
        except Exception:
            return None

    def _smiles_to_canonical(self, smiles):
        """Return RDKit canonical SMILES, or original string if conversion fails."""
        if not isinstance(smiles, str) or not smiles.strip():
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol)
        except Exception:
            return smiles

    # Stage tracking

    def _record_stage(self, df, stage_name):
        """Record per-cell-line unique drug counts and global unique drug count."""
        counts = df.groupby('Cell Line Name')['Drug Name'].nunique().to_dict()
        self.stage_stats[stage_name] = counts
        self.stage_stats[f'{stage_name}_global'] = int(df['Drug Name'].nunique())

    def load_and_clean_data(self):
        """Load CSV and run InChIKey-based deduplication pipeline."""
        print("Loading GDSC dataset...")
        df = pd.read_csv(self.input_file)

        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        required_cols = ['Cell Line Name', 'Drug Name', 'Z score', 'SMILES', 'Dataset Version']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"\n   Unique cell lines : {df['Cell Line Name'].nunique()}")
        print(f"   Unique drugs      : {df['Drug Name'].nunique()}")
        print(f"   Total rows        : {len(df)}")

        print("\nZ Score statistics (raw):")
        print(f"   Range  : {df['Z score'].min():.3f} to {df['Z score'].max():.3f}")
        print(f"   Mean   : {df['Z score'].mean():.3f}  "
              f"Std: {df['Z score'].std():.3f}  "
              f"Median: {df['Z score'].median():.3f}")

        self._record_stage(df, 'stage_0_raw')
        self.processed_stats['raw_entries'] = len(df)
        self.processed_stats['raw_drugs'] = int(df['Drug Name'].nunique())
        self.processed_stats['raw_cell_lines'] = int(df['Cell Line Name'].nunique())

        if RDKIT_AVAILABLE:
            df = self._deduplicate_by_inchikey(df)
        else:
            print("Skipping InChIKey deduplication (RDKit not available)")
            df = df.copy()
            df['Canonical_SMILES'] = df['SMILES']
            self.smiles_map = df.groupby('Drug Name')['SMILES'].first().to_dict()

        self._record_stage(df, 'stage_1_inchikey_dedup')
        self.raw_data = df
        return df

    def _deduplicate_by_inchikey(self, df):
        """Deduplicate drugs with the same InChIKey Block1; GDSC2 preferred over GDSC1."""
        print("\nComputing InChIKeys (first block, stereo-insensitive)...")

        unique_smiles = df['SMILES'].dropna().unique()
        smiles_to_key = {s: self._smiles_to_inchikey(s) for s in unique_smiles}
        smiles_to_can = {s: self._smiles_to_canonical(s) for s in unique_smiles}

        df = df.copy()
        df['InChIKey'] = df['SMILES'].map(smiles_to_key)
        df['Canonical_SMILES'] = df['SMILES'].map(smiles_to_can)

        invalid_count = df['InChIKey'].isna().sum()
        if invalid_count > 0:
            print(f"   {invalid_count} rows with invalid/missing SMILES excluded")
            df = df.dropna(subset=['InChIKey'])
        self.processed_stats['removed_invalid_smiles'] = int(invalid_count)

        # First block only — stereo-insensitive
        df['InChIKey_Block1'] = df['InChIKey'].str.split('-').str[0]

        print(f"   Unique InChIKeys (full) before dedup        : {df['InChIKey'].nunique()}")
        print(f"   Unique InChIKeys (first block) before dedup : {df['InChIKey_Block1'].nunique()}")
        print(f"   Unique Drug Names before dedup              : {df['Drug Name'].nunique()}")

        resolved_parts = []
        n_gdsc2_merges = 0
        n_same_merges = 0

        for inchikey, group in df.groupby('InChIKey_Block1'):
            drug_names = group['Drug Name'].unique()

            if len(drug_names) == 1:
                resolved_parts.append(group)
                continue

            gdsc2_drugs = group[group['Dataset Version'] == 'GDSC2']['Drug Name'].unique()

            if len(gdsc2_drugs) > 0:
                # Cross-dataset: prefer GDSC2 drug with most cell line coverage
                gdsc2_coverage = {
                    d: group[group['Drug Name'] == d]['Cell Line Name'].nunique()
                    for d in gdsc2_drugs
                }
                preferred = max(gdsc2_coverage, key=gdsc2_coverage.get)
                primary_rows = group[group['Drug Name'] == preferred].copy()
                covered = set(primary_rows['Cell Line Name'].unique())

                # Fill cell lines absent from GDSC2 using GDSC1 rows
                gdsc1_fill = group[
                    (group['Dataset Version'] == 'GDSC1') &
                    (~group['Cell Line Name'].isin(covered))
                ].drop_duplicates(subset=['Cell Line Name']).copy()
                gdsc1_fill['Drug Name'] = preferred

                merged = pd.concat([primary_rows, gdsc1_fill])
                print(f"   GDSC2-preferred : {list(drug_names)} → '{preferred}' "
                      f"({primary_rows['Cell Line Name'].nunique()} GDSC2 "
                      f"+ {len(gdsc1_fill)} GDSC1 fill = "
                      f"{merged['Cell Line Name'].nunique()} total)")
                resolved_parts.append(merged)
                n_gdsc2_merges += 1

            else:
                # Same-dataset duplicate: keep drug with most cell line coverage
                coverage = {
                    d: group[group['Drug Name'] == d]['Cell Line Name'].nunique()
                    for d in drug_names
                }
                preferred = max(coverage, key=coverage.get)
                kept = group[group['Drug Name'] == preferred]
                print(f"   Same-dataset    : {list(drug_names)} → kept '{preferred}' "
                      f"({coverage[preferred]} cell lines)")
                resolved_parts.append(kept)
                n_same_merges += 1

        result = pd.concat(resolved_parts).reset_index(drop=True)
        self.smiles_map = result.groupby('Drug Name')['Canonical_SMILES'].first().to_dict()

        print(f"\n   Unique Drug Names after dedup               : {result['Drug Name'].nunique()}")

        self.processed_stats.update({
            'gdsc2_cross_dataset_merges': n_gdsc2_merges,
            'same_dataset_merges': n_same_merges,
        })
        return result

    def create_multitask_format(self):
        """Pivot deduplicated data to wide format: rows=drugs, columns=cell lines."""
        print("\nPivoting to wide format (drugs x cell lines)...")

        pivot_df = self.raw_data.pivot_table(
            index='Cell Line Name',
            columns='Drug Name',
            values='Z score',
            aggfunc='first'
        ).T  # rows=drugs, columns=cell lines

        total_cells = len(pivot_df) * len(pivot_df.columns)
        filled_cells = int(pivot_df.notna().sum().sum())
        sparsity = (1 - filled_cells / total_cells) * 100

        print(f"Shape: {pivot_df.shape}  "
              f"({len(pivot_df)} drugs x {len(pivot_df.columns)} cell lines)")
        print(f"   Non-null data points : {filled_cells}")
        print(f"   Data sparsity        : {sparsity:.1f}%")

        self.zscore_matrix = pivot_df
        z_vals = pivot_df.stack()

        self.processed_stats.update({
            'drugs_after_dedup': len(pivot_df),
            'cell_lines': len(pivot_df.columns),
            'non_null_datapoints': filled_cells,
            'sparsity_percent': round(sparsity, 4),
            'zscore_mean': round(float(z_vals.mean()), 6),
            'zscore_std': round(float(z_vals.std()), 6),
            'zscore_min': round(float(z_vals.min()), 6),
            'zscore_max': round(float(z_vals.max()), 6),
            'sensitivity_threshold': self.sensitivity_threshold,
        })
        return pivot_df

    def determine_binary_labels(self):
        """Assign binary sensitivity labels: 1=sensitive (Z<=threshold), 0=resistant, NaN=untested."""
        print(f"\nComputing binary labels "
              f"(sensitive: Z score <= {self.sensitivity_threshold})...")

        binary_data = pd.DataFrame(index=self.zscore_matrix.index)
        thresholds = {}

        for cell_line in self.zscore_matrix.columns:
            col = self.zscore_matrix[cell_line]
            tested = col.notna()
            labels = col.copy().astype(float)
            labels[tested] = (col[tested] <= self.sensitivity_threshold).astype(float)
            binary_data[cell_line] = labels
            thresholds[cell_line] = self.sensitivity_threshold

        n_sensitive = int((binary_data == 1).sum().sum())
        n_resistant = int((binary_data == 0).sum().sum())
        total_tested = n_sensitive + n_resistant

        print(f"   Sensitive (Z ≤ {self.sensitivity_threshold}) : "
              f"{n_sensitive} ({100*n_sensitive/total_tested:.1f}%)")
        print(f"   Resistant (Z > {self.sensitivity_threshold}) : "
              f"{n_resistant} ({100*n_resistant/total_tested:.1f}%)")

        self.processed_stats.update({
            'n_sensitive': n_sensitive,
            'n_resistant': n_resistant,
            'sensitivity_pct': round(100 * n_sensitive / total_tested, 2),
        })
        return binary_data, thresholds

    def generate_target_summary(self, binary_data, thresholds):
        """Per-cell-line sensitive / resistant counts."""
        rows = []
        for cell_line in binary_data.columns:
            col = binary_data[cell_line].dropna()
            n_sens = int((col == 1).sum())
            n_res = int((col == 0).sum())
            rows.append({
                'Cell_Line': cell_line,
                'Z_Score_Threshold': thresholds.get(cell_line, self.sensitivity_threshold),
                'N_total': n_sens + n_res,
                'N_sensitive': n_sens,
                'N_resistant': n_res,
                'Pct_sensitive': round(100 * n_sens / (n_sens + n_res), 2)
                    if (n_sens + n_res) > 0 else 0,
            })
        return pd.DataFrame(rows).sort_values('Cell_Line').reset_index(drop=True)

    def generate_cleaning_report(self):
        """Per-cell-line drug counts at each cleaning stage with % removed."""
        stage_order = ['stage_0_raw', 'stage_1_inchikey_dedup']
        stage_labels = {
            'stage_0_raw':           'Raw drugs',
            'stage_1_inchikey_dedup': 'After InChIKey Block1 dedup (final)',
        }

        all_cell_lines = sorted(
            set(cl for s in self.stage_stats.values()
                if isinstance(s, dict) for cl in s)
        )

        rows = []
        for cl in all_cell_lines:
            row = {'Cell_Line': cl}
            prev = None
            for stage in stage_order:
                count = self.stage_stats.get(stage, {}).get(cl, 0)
                label = stage_labels[stage]
                row[label] = count
                if prev is not None and prev > 0:
                    row[f'% removed → {label}'] = round(100 * (prev - count) / prev, 2)
                prev = count
            rows.append(row)

        total_row = {'Cell_Line': 'TOTAL (global unique drugs)'}
        prev = None
        for stage in stage_order:
            label = stage_labels[stage]
            total = self.stage_stats.get(f'{stage}_global', 0)
            total_row[label] = total
            if prev is not None and prev > 0:
                total_row[f'% removed → {label}'] = round(100 * (prev - total) / prev, 2)
            prev = total
        rows.append(total_row)

        return pd.DataFrame(rows)

    def save_transformed_data(self):
        """Save all output files into {output_prefix}_output/ folder."""
        output_dir = Path(f"{self.output_prefix}_output")
        output_dir.mkdir(exist_ok=True)

        def _with_smiles(matrix: pd.DataFrame) -> pd.DataFrame:
            out = matrix.copy()
            out.insert(0, 'SMILES', out.index.map(self.smiles_map))
            return out

        zscore_file = output_dir / f"{self.output_prefix}_zscore_matrix.csv"
        _with_smiles(self.zscore_matrix).to_csv(zscore_file)
        print(f"Saved: {zscore_file}")

        binary_data, thresholds = self.determine_binary_labels()
        binary_file = output_dir / f"{self.output_prefix}_binary_matrix.csv"
        _with_smiles(binary_data).to_csv(binary_file)
        print(f"Saved: {binary_file}")

        thr_df = pd.DataFrame(
            list(thresholds.items()), columns=['Cell_Line', 'Threshold']
        )
        thr_file = output_dir / f"{self.output_prefix}_zscore_thresholds.csv"
        thr_df.to_csv(thr_file, index=False)
        print(f"Saved: {thr_file}")

        smiles_df = pd.DataFrame(
            [{'Drug_Name': k, 'SMILES': v} for k, v in self.smiles_map.items()]
        )
        smiles_file = output_dir / f"{self.output_prefix}_smiles_mapping.csv"
        smiles_df.to_csv(smiles_file, index=False)
        print(f"Saved: {smiles_file}")

        summary_df = self.generate_target_summary(binary_data, thresholds)
        summary_file = output_dir / f"{self.output_prefix}_target_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved: {summary_file}")

        cleaning_df = self.generate_cleaning_report()
        cleaning_file = output_dir / f"{self.output_prefix}_cleaning_stages.csv"
        cleaning_df.to_csv(cleaning_file, index=False)
        print(f"Saved: {cleaning_file}")

        stats_df = pd.DataFrame(
            list(self.processed_stats.items()), columns=['Metric', 'Value']
        )
        stats_file = output_dir / f"{self.output_prefix}_processing_stats.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"Saved: {stats_file}")

        return {
            'zscore_matrix':    zscore_file,
            'binary_matrix':    binary_file,
            'thresholds':       thr_file,
            'smiles_mapping':   smiles_file,
            'target_summary':   summary_file,
            'cleaning_stages':  cleaning_file,
            'processing_stats': stats_file,
        }

    def run_complete_transformation(self):
        """Run the full pipeline: load, clean, pivot, save."""
        self.load_and_clean_data()
        self.create_multitask_format()
        output_files = self.save_transformed_data()
        print("Done.")
        return output_files

def main():
    parser = argparse.ArgumentParser(
        description='Transform GDSC dataset for multi-task learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files (saved to {prefix}_output/)
-----------------------------------------
  {prefix}_zscore_matrix.csv      : drugs × cell lines Z score matrix (+ SMILES)
  {prefix}_binary_matrix.csv      : binary sensitivity labels (1=sensitive, 0=resistant)
  {prefix}_zscore_thresholds.csv  : Z score threshold per cell line
  {prefix}_smiles_mapping.csv     : Drug Name → canonical SMILES
  {prefix}_target_summary.csv     : per-cell-line sensitive/resistant counts
  {prefix}_cleaning_stages.csv    : drug counts at each cleaning stage
  {prefix}_processing_stats.csv   : overall pipeline statistics

Examples
--------
  python transform_for_multitask_zscore.py data.csv -o gdsc_nsclc
  python transform_for_multitask_zscore.py data.csv -o gdsc_nsclc --threshold -2.0
"""
    )
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output_prefix', '-o', default='gdsc_multitask',
                        help='Output prefix and folder name (default: gdsc_multitask)')
    parser.add_argument('--threshold', '-t', type=float, default=-2.0,
                        help='Z score sensitivity threshold (default: -2.0)')

    args = parser.parse_args()

    try:
        transformer = GDSCDatasetTransformer(
            args.input_file,
            output_prefix=args.output_prefix,
            sensitivity_threshold=args.threshold,
        )
        transformer.run_complete_transformation()
    except Exception as e:
        print(f"Error during transformation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
