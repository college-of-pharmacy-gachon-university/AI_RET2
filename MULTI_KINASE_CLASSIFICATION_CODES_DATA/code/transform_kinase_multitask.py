#!/usr/bin/env python3
"""Kinase IC50 dataset transformation for multi-task classification."""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
from sklearn.metrics import roc_curve

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, inchi as rdinchi
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings('ignore')

class KinaseDatasetTransformer:
    """Transform kinase IC50 dataset into multi-task pIC50 / binary matrices."""

    def __init__(self, input_file, output_prefix="kinase_multitask", dedup_mode="entry"):
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.dedup_mode = dedup_mode
        self.raw_data = None
        self.multitask_data = None
        self.smiles_data = {}
        self.binding_monomer_data = {}
        self.inchikey_data = {}
        self.processed_stats = {}
        self.stage_stats = {}   # per-target counts at each cleaning stage
        
    def load_and_clean_data(self):
        """Load CSV and run staged deduplication + SMILES cleaning pipeline."""
        print("Loading kinase dataset...")
        df = pd.read_csv(self.input_file)

        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        required_cols = ['Target Name', 'IC50 (nM)', 'Ligand SMILES']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"Unique targets: {df['Target Name'].nunique()}")
        print(f"Unique SMILES: {df['Ligand SMILES'].nunique()}")
        print(f"Total entries: {len(df)}")
        print(f"Deduplication mode: '{self.dedup_mode}'")

        # Stage 0 — baseline counts
        self._record_stage(df, 'stage_0_raw')

        if self.dedup_mode == 'entry':
            # Round-1: remove rows identical on (Target, IC50, SMILES)
            print("\n--- MODE: entry (Round-1 exact dedup + stereo dedup) ---")
            df = self.exact_deduplicate(df)
            self._record_stage(df, 'stage_1_exact_dedup')

        elif self.dedup_mode == 'smiles':
            # Collapse to unique SMILES first (one row per unique SMILES string,
            # keeping all target/IC50 information intact via the full df)
            print("\n--- MODE: smiles (unique SMILES baseline + stereo dedup) ---")
            df = self.smiles_deduplicate(df)
            self._record_stage(df, 'stage_1_smiles_dedup')

        else:
            raise ValueError(f"Unknown dedup_mode '{self.dedup_mode}'. Use 'entry' or 'smiles'.")

        # Stage 2 — SMILES validation (both modes)
        df = self.validate_smiles(df)
        self._record_stage(df, 'stage_2_clean_smiles')

        # Stage 3 — InChIKey-based stereo deduplication (both modes)
        df = self.deduplicate_by_inchikey(df)
        self._record_stage(df, 'stage_3_stereo_dedup')

        binding_col = next(
            (c for c in df.columns if c.lower().replace(' ', '') == 'bindingdbmonomerid'),
            None
        )
        if binding_col and self.binding_monomer_data:
            print(f"Binding Monomer IDs captured from column '{binding_col}': {len(self.binding_monomer_data)}")
        elif not binding_col:
            print("BindingDB MonomerID column not found — skipping")

        self.raw_data = df
        return df

    def _record_stage(self, df, stage_name):
        """Record per-target counts and global unique compound total at each cleaning stage."""
        if self.dedup_mode == 'smiles':
            if 'InChIKey_Block1' in df.columns:
                counts = df.groupby('Target Name')['InChIKey_Block1'].nunique().to_dict()
                global_count = int(df['InChIKey_Block1'].nunique())
            else:
                counts = df.groupby('Target Name')['Ligand SMILES'].nunique().to_dict()
                global_count = int(df['Ligand SMILES'].nunique())
        else:
            counts = df.groupby('Target Name').size().to_dict()
            global_count = int(len(df))

        self.stage_stats[stage_name] = counts
        self.stage_stats[f'{stage_name}_global'] = global_count

    def exact_deduplicate(self, df):
        """Drop rows identical on (Target Name, IC50 (nM), Ligand SMILES) — true BindingDB duplicates."""
        print("Round-1 deduplication (exact match: Target + IC50 + SMILES)...")
        n_before = len(df)
        df_deduped = df.drop_duplicates(
            subset=['Target Name', 'IC50 (nM)', 'Ligand SMILES'],
            keep='first'
        ).copy()
        n_removed = n_before - len(df_deduped)

        # Per-target breakdown
        before_counts = df.groupby('Target Name').size()
        after_counts  = df_deduped.groupby('Target Name').size()
        removed_per_target = (before_counts - after_counts).fillna(0).astype(int)

        print(f"   Exact duplicates removed: {n_removed} / {n_before} entries")
        print(f"   Retained: {len(df_deduped)} entries")
        print(f"   Per-target removals:")
        for tgt, cnt in removed_per_target[removed_per_target > 0].items():
            pct = 100 * cnt / before_counts[tgt]
            print(f"      {tgt}: -{cnt} ({pct:.1f}%)")

        self.processed_stats['exact_duplicates_removed'] = int(n_removed)
        return df_deduped

    def smiles_deduplicate(self, df):
        """Establish unique-SMILES baseline without removing rows (MODE 'smiles', Stage 1)."""
        print("SMILES-mode: establishing unique-compound baseline...")
        n_unique_smiles = df['Ligand SMILES'].nunique()
        n_entries = len(df)
        avg = n_entries / n_unique_smiles if n_unique_smiles else 0

        print(f"   Unique SMILES (compound identities): {n_unique_smiles}")
        print(f"   Total entries (kept for pivot averaging): {n_entries}")
        print(f"   Avg entries per unique SMILES: {avg:.2f}")

        per_target = df.groupby('Target Name')['Ligand SMILES'].nunique()
        print(f"   Unique SMILES per target:")
        for tgt, cnt in per_target.items():
            print(f"      {tgt}: {cnt}")

        self.processed_stats['unique_smiles_baseline'] = int(n_unique_smiles)
        self.processed_stats['total_entries_at_baseline'] = int(n_entries)
        return df.copy()

    def generate_per_stage_report(self):
        """
        Build a per-target, per-stage entry-count table and compute
        the % removed between consecutive stages.
        """
        # Build stage order from whichever stages were actually recorded
        # (differs between 'entry' and 'smiles' modes)
        if 'stage_1_exact_dedup' in self.stage_stats:
            stage_order = [
                'stage_0_raw',
                'stage_1_exact_dedup',
                'stage_2_clean_smiles',
                'stage_3_stereo_dedup',
            ]
        else:
            stage_order = [
                'stage_0_raw',
                'stage_1_smiles_dedup',
                'stage_2_clean_smiles',
                'stage_3_stereo_dedup',
            ]
        if self.dedup_mode == 'smiles':
            stage_labels = {
                'stage_0_raw':          'Raw unique SMILES',
                'stage_1_smiles_dedup': 'Unique SMILES baseline',
                'stage_2_clean_smiles': 'After * / invalid removal',
                'stage_3_stereo_dedup': 'After stereo dedup (final)',
            }
        else:
            stage_labels = {
                'stage_0_raw':          'Raw entries',
                'stage_1_exact_dedup':  'After exact dedup',
                'stage_2_clean_smiles': 'After * / invalid removal',
                'stage_3_stereo_dedup': 'After stereo dedup (final)',
            }

        all_targets = sorted(
            set(t for s in self.stage_stats.values() if isinstance(s, dict) for t in s)
        )

        rows = []
        for target in all_targets:
            row = {'Target': target}
            prev_count = None
            for stage in stage_order:
                count = self.stage_stats.get(stage, {}).get(target, 0)
                label = stage_labels[stage]
                row[label] = count
                if prev_count is not None and prev_count > 0:
                    pct_removed = 100 * (prev_count - count) / prev_count
                    row[f'% removed → {label}'] = round(pct_removed, 2)
                prev_count = count
            rows.append(row)

        # Add totals row — use global unique counts (not sum of per-target counts,
        # which would double-count compounds tested against multiple targets)
        total_row = {'Target': 'TOTAL (global unique)'}
        prev_total = None
        for stage in stage_order:
            label = stage_labels[stage]
            global_key = f'{stage}_global'
            total = self.stage_stats.get(global_key, sum(self.stage_stats.get(stage, {}).values()))
            total_row[label] = total
            if prev_total is not None and prev_total > 0:
                pct = 100 * (prev_total - total) / prev_total
                total_row[f'% removed → {label}'] = round(pct, 2)
            prev_total = total
        rows.append(total_row)

        return pd.DataFrame(rows)
    
    def validate_smiles(self, df):
        """Remove wildcard ('*') and RDKit-invalid SMILES entries."""
        print("Validating SMILES entries...")
        n_before = len(df)

        # Flag entries that contain '*' (wildcard atoms — corrupt BindingDB records)
        wildcard_mask = df['Ligand SMILES'].astype(str).str.contains(r'\*', regex=True)
        n_wildcard = wildcard_mask.sum()

        # Flag entries where RDKit cannot parse the SMILES
        def _rdkit_invalid(smi):
            try:
                return Chem.MolFromSmiles(str(smi)) is None
            except Exception:
                return True

        rdkit_invalid_mask = df['Ligand SMILES'].apply(_rdkit_invalid)
        n_rdkit_invalid = (rdkit_invalid_mask & ~wildcard_mask).sum()

        df_valid = df[~wildcard_mask & ~rdkit_invalid_mask].copy()
        n_after = len(df_valid)

        print(f"   Removed {n_wildcard} entries with wildcard '*' SMILES")
        print(f"   Removed {n_rdkit_invalid} entries with RDKit-unparseable SMILES")
        print(f"   Valid entries retained: {n_after} / {n_before}")

        self.processed_stats['removed_wildcard_smiles'] = int(n_wildcard)
        self.processed_stats['removed_invalid_smiles'] = int(n_rdkit_invalid)

        return df_valid

    def generate_inchikey_first_block(self, smiles):
        """Return the connectivity-only first block of InChIKey (stereo-insensitive, 14 chars)."""
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        try:
            ik = rdinchi.MolToInchiKey(mol)   # full InChIKey (all 3 blocks)
            return ik.split('-')[0] if ik else None   # first block only
        except Exception:
            return None

    def deduplicate_by_inchikey(self, df):
        """Deduplicate stereoisomers using the connectivity-only first block of InChIKey."""
        print("Generating InChIKey first-block identifiers (stereo-insensitive)...")
        df = df.copy()
        df['InChIKey_Block1'] = df['Ligand SMILES'].apply(self.generate_inchikey_first_block)

        n_before = len(df)
        n_null_ik = df['InChIKey_Block1'].isna().sum()
        if n_null_ik:
            print(f"   {n_null_ik} entries could not get an InChIKey — dropped")
            df = df.dropna(subset=['InChIKey_Block1'])

        n_unique_ik = df['InChIKey_Block1'].nunique()
        n_stereo_dupes = n_before - n_null_ik - n_unique_ik
        # n_stereo_dupes can be negative if multiple rows share the same key
        # (multiple target measurements per compound); count properly:
        ik_counts = df.groupby('InChIKey_Block1')['Ligand SMILES'].nunique()
        n_stereo_dupes = int((ik_counts > 1).sum())

        print(f"   Unique InChIKey blocks (stereo-insensitive): {n_unique_ik}")
        print(f"   SMILES groups with >1 stereoisomer variant: {n_stereo_dupes}")

        self.processed_stats['stereo_duplicate_groups'] = n_stereo_dupes
        self.processed_stats['unique_inchikey_blocks'] = n_unique_ik

        # Rebuild Compound_ID on the InChIKey block so stereoisomers share one ID
        df['Compound_ID'] = df['InChIKey_Block1'].astype('category').cat.codes

        # Rebuild SMILES mapping: one canonical SMILES per Compound_ID
        # (take the most frequent SMILES within each group as canonical)
        canonical = (
            df.groupby('Compound_ID')['Ligand SMILES']
            .agg(lambda x: x.value_counts().index[0])
            .to_dict()
        )
        self.smiles_data = canonical

        self.inchikey_data = (
            df.groupby('Compound_ID')['InChIKey_Block1']
            .first()
            .to_dict()
        )

        binding_col = next(
            (c for c in df.columns if c.lower().replace(' ', '') == 'bindingdbmonomerid'),
            None
        )
        if binding_col:
            self.binding_monomer_data = (
                df.groupby('Compound_ID')[binding_col]
                .apply(lambda x: ';'.join(x.dropna().astype(str).unique()))
                .to_dict()
            )

        return df

    def generate_summary_table(self, binary_data, thresholds):
        """
        Build per-target summary: total compounds tested, active count, inactive count.
        Active = binary label 1 (pIC50 above Youden threshold).
        """
        rows = []
        for target in binary_data.columns:
            col = binary_data[target].dropna()
            n_active = int((col == 1).sum())
            n_inactive = int((col == 0).sum())
            rows.append({
                'Target': target,
                'Threshold_pIC50': round(thresholds.get(target, float('nan')), 4),
                'N_total': n_active + n_inactive,
                'N_active': n_active,
                'N_inactive': n_inactive,
                'Ratio_active_pct': round(100 * n_active / (n_active + n_inactive), 2)
                    if (n_active + n_inactive) > 0 else 0
            })
        return pd.DataFrame(rows).sort_values('Target').reset_index(drop=True)

    def handle_inequality_values(self, ic50_series):
        """Parse IC50 strings with < and > prefixes into numeric values and inequality flags."""
        print("Processing IC50 inequality values...")
        
        processed_values = []
        inequality_flags = []
        
        greater_count = 0
        less_count = 0
        normal_count = 0
        
        for val in ic50_series:
            val_str = str(val).strip()
            
            if val_str.startswith('>'):
                numeric_val = float(val_str[1:])
                processed_values.append(numeric_val)
                inequality_flags.append('>')
                greater_count += 1
            elif val_str.startswith('<'):
                numeric_val = float(val_str[1:])
                processed_values.append(numeric_val)
                inequality_flags.append('<')
                less_count += 1
            else:
                try:
                    numeric_val = float(val_str)
                    processed_values.append(numeric_val)
                    inequality_flags.append('=')
                    normal_count += 1
                except ValueError:
                    processed_values.append(np.nan)
                    inequality_flags.append('invalid')
        
        print(f"   Inequality distribution: normal={normal_count}, >={greater_count}, <={less_count}, "
              f"invalid={len(processed_values) - normal_count - greater_count - less_count}")
        
        return processed_values, inequality_flags
    
    def convert_ic50_to_pic50(self, ic50_values):
        """Convert IC50 (nM) to pIC50 using -log10(IC50_nM / 1e9)."""
        print("Converting IC50 (nM) to pIC50 (M)...")

        ic50_array = np.array(ic50_values, dtype=float)

        valid_ic50 = ic50_array[~np.isnan(ic50_array)]
        print(f"   IC50 (nM): {len(valid_ic50)} valid, "
              f"range {valid_ic50.min():.2f}–{valid_ic50.max():.2f}, "
              f"mean {valid_ic50.mean():.2f}, median {np.median(valid_ic50):.2f}")
        
        ic50_m = ic50_array / 1e9
        min_positive_ic50 = ic50_m[ic50_m > 0].min() if len(ic50_m[ic50_m > 0]) > 0 else 1e-15
        ic50_m_cleaned = np.where(ic50_m <= 0, min_positive_ic50, ic50_m)
        pic50_values = -np.log10(ic50_m_cleaned)
        
        valid_pic50 = pic50_values[~np.isnan(pic50_values)]
        print(f"   pIC50: range {valid_pic50.min():.3f}–{valid_pic50.max():.3f}, "
              f"mean {valid_pic50.mean():.3f}, median {np.median(valid_pic50):.3f}")
        
        return pic50_values
    
    def calculate_youdens_j_threshold(self, y_true, y_scores):
        """Optimal binary threshold by Youden's J = TPR - FPR."""
        sorted_indices = np.argsort(y_scores)
        fpr, tpr, thresholds = roc_curve(
            y_true[sorted_indices], y_scores[sorted_indices])
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        return thresholds[best_threshold_idx], j_scores[best_threshold_idx]
    
    def create_multitask_format(self):
        """Pivot cleaned data to compound × target pIC50 matrix."""
        print("Creating multi-task format...")
        
        df = self.raw_data.copy()

        processed_ic50, inequality_flags = self.handle_inequality_values(df['IC50 (nM)'])
        df['Processed_IC50_nM'] = processed_ic50
        df['Inequality_Flag'] = inequality_flags
        df['pIC50'] = self.convert_ic50_to_pic50(processed_ic50)
        df = df.dropna(subset=['pIC50'])

        print(f"After IC50 cleaning: {len(df)} data points, "
              f"{df['Target Name'].nunique()} targets, "
              f"{df['Compound_ID'].nunique()} compounds (stereo-deduped)")

        pivot_df = df.pivot_table(
            index='Compound_ID',
            columns='Target Name',
            values='pIC50',
            aggfunc='mean'
        )

        print(f"Multi-task matrix: {pivot_df.shape} ({len(pivot_df)} compounds x {len(pivot_df.columns)} targets)")

        total_cells = len(pivot_df) * len(pivot_df.columns)
        filled_cells = pivot_df.notna().sum().sum()
        sparsity = (1 - filled_cells / total_cells) * 100
        print(f"  Data sparsity: {sparsity:.1f}%")

        self.multitask_data = pivot_df
        self.processed_stats.update({
            'original_entries': len(self.raw_data),
            'processed_entries': len(df),
            'compounds': len(pivot_df),
            'targets': len(pivot_df.columns),
            'sparsity_percent': sparsity,
            'pic50_mean': df['pIC50'].mean(),
            'pic50_std': df['pIC50'].std(),
            'pic50_min': df['pIC50'].min(),
            'pic50_max': df['pIC50'].max()
        })
        
        return pivot_df
    
    def determine_binary_thresholds(self, method='youden'):
        """Compute binary classification threshold for each target using Youden's J Statistic."""
        print(f"Determining binary thresholds using {method} method...")
        
        thresholds = {}
        binary_data = pd.DataFrame(index=self.multitask_data.index)
        
        for target in self.multitask_data.columns:
            target_values = self.multitask_data[target].dropna()
            
            if len(target_values) < 10:  # Skip targets with too few data points
                continue
            
            if method == 'youden':
                # For Youden's J, we need binary labels to calculate ROC
                # Use median as initial split to create pseudo-binary labels
                median_threshold = target_values.median()
                pseudo_labels = (target_values > median_threshold).astype(int)
                
                try:
                    optimal_threshold, j_score = self.calculate_youdens_j_threshold(
                        pseudo_labels.values, target_values.values
                    )
                    thresholds[target] = optimal_threshold
                except:
                    # Fallback to median if Youden's J calculation fails
                    thresholds[target] = median_threshold
            
            elif method == 'median':
                thresholds[target] = target_values.median()
            
            # Create binary labels — preserve NaN for untested compounds
            # (NaN > threshold evaluates to False, so .astype(int) would wrongly
            #  count untested compounds as inactive; we mask them back to NaN)
            col = self.multitask_data[target]
            tested = col.notna()
            labels = col.copy().astype(float)          # NaN stays NaN
            labels[tested] = (col[tested] > thresholds[target]).astype(float)
            binary_data[target] = labels
        
        threshold_values = list(thresholds.values())
        print(f"   Thresholds for {len(thresholds)} targets: "
              f"mean={np.mean(threshold_values):.3f}, "
              f"median={np.median(threshold_values):.3f}, "
              f"range {np.min(threshold_values):.3f}–{np.max(threshold_values):.3f}")
        
        return binary_data, thresholds
    
    def save_transformed_data(self):
        """Save transformed data in formats compatible with multitask_classifier.py"""
        output_dir = Path(f"{self.output_prefix}_output")
        output_dir.mkdir(exist_ok=True)
        
        def _inject_metadata(matrix: pd.DataFrame) -> pd.DataFrame:
            """Prepend InChIKey_Block1 and Binding_Monomer_ID columns (when available)."""
            out = matrix.copy()
            # Insert rightmost first so column order is: InChIKey | Binding_Monomer | targets
            if self.binding_monomer_data:
                out.insert(0, 'Binding_Monomer_ID', out.index.map(self.binding_monomer_data))
            if hasattr(self, 'inchikey_data') and self.inchikey_data:
                out.insert(0, 'InChIKey_Block1', out.index.map(self.inchikey_data))
            return out

        pic50_file = output_dir / f"{self.output_prefix}_pIC50_matrix.csv"
        _inject_metadata(self.multitask_data).to_csv(pic50_file)
        print(f"Saved: {pic50_file}")

        binary_data, thresholds = self.determine_binary_thresholds(method='youden')
        binary_file = output_dir / f"{self.output_prefix}_binary_matrix.csv"
        _inject_metadata(binary_data).to_csv(binary_file)
        print(f"Saved: {binary_file}")

        threshold_df = pd.DataFrame(list(thresholds.items()), columns=['Target', 'Threshold'])
        threshold_file = output_dir / f"{self.output_prefix}_youden_thresholds.csv"
        threshold_df.to_csv(threshold_file, index=False)
        print(f"Saved: {threshold_file}")

        smiles_df = pd.DataFrame(list(self.smiles_data.items()), columns=['Compound_ID', 'SMILES'])
        if hasattr(self, 'inchikey_data') and self.inchikey_data:
            smiles_df['InChIKey_Block1'] = smiles_df['Compound_ID'].map(self.inchikey_data)
        if self.binding_monomer_data:
            smiles_df['Binding_Monomer_ID'] = smiles_df['Compound_ID'].map(self.binding_monomer_data)
        smiles_file = output_dir / f"{self.output_prefix}_smiles_mapping.csv"
        smiles_df.to_csv(smiles_file, index=False)
        print(f"Saved: {smiles_file}")

        summary_df = self.generate_summary_table(binary_data, thresholds)
        summary_file = output_dir / f"{self.output_prefix}_target_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved: {summary_file}")

        stage_report_df = self.generate_per_stage_report()
        stage_report_file = output_dir / f"{self.output_prefix}_cleaning_stages.csv"
        stage_report_df.to_csv(stage_report_file, index=False)
        print(f"Saved: {stage_report_file}")

        stats_df = pd.DataFrame(list(self.processed_stats.items()), columns=['Metric', 'Value'])
        stats_file = output_dir / f"{self.output_prefix}_processing_stats.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"Saved: {stats_file}")

        return {
            'pic50_matrix': pic50_file,
            'binary_matrix': binary_file,
            'thresholds': threshold_file,
            'smiles_mapping': smiles_file,
            'target_summary': summary_file,
            'cleaning_stages': stage_report_file,
            'processing_stats': stats_file
        }
    
    def run_complete_transformation(self):
        """Run the complete transformation pipeline."""
        self.load_and_clean_data()
        self.create_multitask_format()
        output_files = self.save_transformed_data()
        print("Done.")
        return output_files

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='Transform kinase dataset for multi-task learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deduplication modes
-------------------
  entry  (default)
      Round-1: remove exact database duplicates (same Target + IC50 + SMILES).
      Round-2: merge stereoisomers via InChIKey block-1.
      Use this for the corrected, publication-ready dataset.

  smiles
      Skip Round-1.  Start from the 104,856 unique-SMILES baseline (mirrors
      the original pipeline), then apply SMILES cleaning and stereo dedup.
      Use this to see what % of the original compounds survive cleaning.

Examples
--------
  python transform_kinase_multitask.py data.csv -o run_v1
  python transform_kinase_multitask.py data.csv -o run_v1_smiles --dedup_mode smiles
"""
    )
    parser.add_argument('input_file', help='Path to input CSV file (kinase dataset)')
    parser.add_argument('--output_prefix', '-o', default='kinase_multitask',
                       help='Output file prefix (default: kinase_multitask)')
    parser.add_argument('--dedup_mode', '-d', default='entry',
                       choices=['entry', 'smiles'],
                       help="Deduplication mode: 'entry' (default) or 'smiles' (see epilog)")

    args = parser.parse_args()

    try:
        transformer = KinaseDatasetTransformer(args.input_file, args.output_prefix,
                                               dedup_mode=args.dedup_mode)
        transformer.run_complete_transformation()
    except Exception as e:
        print(f"Error during transformation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
