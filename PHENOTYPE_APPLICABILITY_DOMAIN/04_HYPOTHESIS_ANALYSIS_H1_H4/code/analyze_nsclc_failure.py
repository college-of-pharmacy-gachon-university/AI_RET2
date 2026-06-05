"""
NSCLC diagnostic analysis, configurable by Z-score threshold and weighting.

The path presets mirror GDSC_DATASET/run_all_AD.py. By default this script runs
the weighted z=0 analysis (tag: z0). Use --tag z0_unwt for the unweighted z=0
run, or another preset listed in CONFIGS.

Example
-------
  python analyze_nsclc_failure.py --tag z0
  python analyze_nsclc_failure.py --tag z0_unwt
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.warning")


ROOT    = Path(__file__).resolve().parents[3]  # AI-RET_MS_REVISED_FILES_FOR_UPLOAD/
BASE    = ROOT  # retained for CONFIGS below
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(exist_ok=True)

MODELS_FIG = ["RandomForest", "GradientBoosting", "LogisticRegression", "SVM"]
FPS_FIG = ["AVALON", "ECFP", "ECFP_COUNT", "MACCS"]

CONFIGS = {
    # Weighted z=0. This is the default and matches run_all_AD.py tag "z0".
    "z0": {
        "z_threshold": 0.0,
        "weighting": "weighted",
        "results": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_weighted/gdsc_weighted_results_20260521_154749.csv",
        "smiles": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_smiles_mapping.csv",
        "zscore": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_zscore_matrix.csv",
        "classlag": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_classLAG_AD_z0_per_cellline.csv",
        "consensus": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_consensus_AD_z0_per_cellline.csv",
    },
    "z0_unwt": {
        "z_threshold": 0.0,
        "weighting": "unweighted",
        "results": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0_unweighted/gdsc_unweighted_results_20260522_154422.csv",
        "smiles": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_smiles_mapping.csv",
        "zscore": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0_threshold/gdsc_multitask_threshold_zeo_zscore_matrix.csv",
        "classlag": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_classLAG_AD_z0_unwt_per_cellline.csv",
        "consensus": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_consensus_AD_z0_unwt_per_cellline.csv",
    },
    "z05": {
        "z_threshold": -0.5,
        "weighting": "weighted",
        "results": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0.5_weighted/gdsc_weighted_results_20260522_172746.csv",
        "smiles": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.5_threshold/gdsc_multitask_threshold_0.5_smiles_mapping.csv",
        "zscore": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.5_threshold/gdsc_multitask_threshold_0.5_zscore_matrix.csv",
        "classlag": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_classLAG_AD_z05_per_cellline.csv",
        "consensus": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_consensus_AD_z05_per_cellline.csv",
    },
    "z075": {
        "z_threshold": -0.75,
        "weighting": "weighted",
        "results": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z0.75_weighted/gdsc_weighted_results_20260522_140030.csv",
        "smiles": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.75_threshold/gdsc_multitask_threshold_0.75_smiles_mapping.csv",
        "zscore": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z0.75_threshold/gdsc_multitask_threshold_0.75_zscore_matrix.csv",
        "classlag": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_classLAG_AD_z075_per_cellline.csv",
        "consensus": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_consensus_AD_z075_per_cellline.csv",
    },
    "z1": {
        "z_threshold": -1.0,
        "weighting": "weighted",
        "results": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/results/z1.0_weighted/gdsc_weighted_results_20260521_111123.csv",
        "smiles": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z1.0_threshold/gdsc_nsclc_20May_smiles_mapping.csv",
        "zscore": ROOT / "PHENOTYPES_CLASSIFICATION_CODES_DATA/input/z1.0_threshold/gdsc_nsclc_20May_zscore_matrix.csv",
        "classlag": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_classLAG_AD_z1_per_cellline.csv",
        "consensus": ROOT / "03_APPLICABILITY_DOMAIN/results/gdsc_consensus_AD_z1_per_cellline.csv",
    },
}


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required input file(s) are missing:\n  " + "\n  ".join(missing)
        )


def load_zscore_matrix(path):
    z_raw = pd.read_csv(path, index_col=0)
    if "SMILES" in z_raw.columns:
        z_raw = z_raw.drop(columns=["SMILES"])
    return z_raw.apply(pd.to_numeric, errors="coerce")


def bitvect_from_array(arr):
    bv = DataStructs.ExplicitBitVect(len(arr))
    for bit_pos in np.where(arr)[0]:
        bv.SetBit(int(bit_pos))
    return bv


def build_fingerprints(smiles_map):
    fp_dicts = {"AVALON": {}, "ECFP4": {}, "MACCS": {}}

    for drug, smi in smiles_map.items():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        arr_av = np.zeros(2048, dtype=np.uint8)
        ConvertToNumpyArray(GetAvalonFP(mol, nBits=2048), arr_av)
        fp_dicts["AVALON"][drug] = bitvect_from_array(arr_av)

        arr_ec = np.zeros(2048, dtype=np.uint8)
        ConvertToNumpyArray(
            rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            ),
            arr_ec,
        )
        fp_dicts["ECFP4"][drug] = bitvect_from_array(arr_ec)

        arr_ma = np.zeros(167, dtype=np.uint8)
        ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr_ma)
        fp_dicts["MACCS"][drug] = bitvect_from_array(arr_ma)

    return fp_dicts


def safe_kruskal(groups):
    groups = [pd.Series(g).dropna().values for g in groups if len(pd.Series(g).dropna())]
    if len(groups) < 2:
        return np.nan, np.nan
    return stats.kruskal(*groups)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NSCLC diagnostic analysis for a threshold/weight preset."
    )
    parser.add_argument(
        "--tag",
        choices=sorted(CONFIGS),
        default="z0",
        help="Preset copied from run_all_AD.py. Default: z0 (weighted).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CONFIGS[args.tag]

    tag = args.tag
    z_threshold = cfg["z_threshold"]
    weighting = cfg["weighting"]
    results_all_file = cfg["results"]
    smiles_file = cfg["smiles"]
    zscore_file = cfg["zscore"]
    classlag_file = cfg["classlag"]
    consensus_ad_file = cfg["consensus"]

    out_stats = OUT_DIR / f"nsclc_diagnostic_statistics_{tag}.csv"
    out_struct_resp = OUT_DIR / f"nsclc_structural_vs_response_{tag}.csv"
    out_struct_fp = OUT_DIR / f"nsclc_structural_vs_response_by_fp_{tag}.csv"

    require_files(
        [
            results_all_file,
            smiles_file,
            zscore_file,
            classlag_file,
            consensus_ad_file,
        ]
    )

    print("=" * 72)
    print(f"NSCLC diagnostic analysis: {tag} ({weighting}, z <= {z_threshold})")
    print("=" * 72)
    print(f"Results file    : {results_all_file}")
    print(f"Z-score matrix  : {zscore_file}")
    print(f"SMILES mapping  : {smiles_file}")
    print(f"CLASS-LAG AD    : {classlag_file}")
    print(f"Consensus AD    : {consensus_ad_file}")

    results_all = pd.read_csv(results_all_file)
    zscore = load_zscore_matrix(zscore_file)
    smiles_map = pd.read_csv(smiles_file).dropna(subset=["SMILES"])
    smiles_map = smiles_map.drop_duplicates(subset=["Drug_Name"]).set_index("Drug_Name")[
        "SMILES"
    ]
    classlag_per_cl = pd.read_csv(classlag_file)
    consensus_per_cl = pd.read_csv(consensus_ad_file)

    rows = []

    tt = results_all[
        (results_all["Split"] == "Train_Test")
        & (results_all["Model"].isin(MODELS_FIG))
        & (results_all["Fingerprint"].isin(FPS_FIG))
    ].copy()

    print(f"\nLoaded {len(tt):,} figure-level Train_Test rows")
    print(f"Selection: models={MODELS_FIG}; fingerprints={FPS_FIG}; all split methods")
    print(f"Z-score matrix: {zscore.shape[0]} drugs x {zscore.shape[1]} cell lines")

    # H1: wrong algorithm.
    algo_auc = tt.groupby("Model")["AUC"].apply(list)
    algo_means = tt.groupby("Model")["AUC"].mean().sort_values(ascending=False)
    kw_stat, kw_p = safe_kruskal([algo_auc[a] for a in algo_auc.index])
    auc_range = float(algo_means.max() - algo_means.min())
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H1 - Wrong algorithm",
            "Test": (
                "Kruskal-Wallis AUC across algorithms "
                f"({', '.join(algo_means.index)})"
            ),
            "Statistic": f"H={kw_stat:.4f}; AUC range={auc_range:.4f}",
            "p_value": f"{kw_p:.4e}",
            "Interpretation": (
                "Algorithms differ statistically, but the practical AUC range is "
                f"{auc_range:.4f} and the best mean AUC is {algo_means.max():.4f}."
            ),
        }
    )

    # H1: AD restriction as a practical overfitting/extrapolation check.
    ad_pair = classlag_per_cl[["Cell_Line", "AUC_cov010", "AUC_cov100"]].dropna()
    if len(ad_pair) > 0:
        w_stat, w_p = stats.wilcoxon(ad_pair["AUC_cov010"], ad_pair["AUC_cov100"])
        ad_gain = float((ad_pair["AUC_cov010"] - ad_pair["AUC_cov100"]).mean())
        rows.append(
            {
                "Configuration": tag,
                "Hypothesis": "H1 - Overfitting / AD test",
                "Test": (
                    "Paired Wilcoxon: CLASS-LAG AUC@10% vs AUC@100% "
                    f"({len(ad_pair)} valid cell lines)"
                ),
                "Statistic": f"W={w_stat:.1f}; paired mean gain={ad_gain:.4f}",
                "p_value": f"{w_p:.4f}",
                "Interpretation": (
                    "If AD filtering rescued the model, the top-confidence subset "
                    "would show a large AUC gain."
                ),
            }
        )

    # H2: sample size and class balance. Use per-cell-line mean AUC across the
    # same 64 combinations used in phenotype_figures_analysis.ipynb.
    cl_auc = tt.groupby("Target")["AUC"].mean().rename("Mean_AUC")
    cl_n = zscore.notna().sum(axis=0).rename("N_total")
    cl_active_pct = (
        (zscore.le(z_threshold) & zscore.notna()).sum(axis=0) / cl_n * 100.0
    ).rename("Pct_sensitive")

    common_size = cl_auc.index.intersection(cl_n.index)
    r_size, p_size = stats.pearsonr(cl_n.loc[common_size], cl_auc.loc[common_size])
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H2 - Insufficient data",
            "Test": f"Pearson r: cell-line sample size vs mean AUC ({len(common_size)} cell lines)",
            "Statistic": f"r={r_size:.4f}",
            "p_value": f"{p_size:.4f}",
            "Interpretation": "Tests whether larger cell-line datasets have higher AUC.",
        }
    )

    common_bal = cl_auc.index.intersection(cl_active_pct.index)
    r_bal, p_bal = stats.pearsonr(
        cl_active_pct.loc[common_bal], cl_auc.loc[common_bal]
    )
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H2 - Class imbalance",
            "Test": f"Pearson r: percent sensitive vs mean AUC ({len(common_bal)} cell lines)",
            "Statistic": f"r={r_bal:.4f}",
            "p_value": f"{p_bal:.4f}",
            "Interpretation": "Tests whether class balance explains AUC variation.",
        }
    )

    balanced_cl = cl_active_pct[
        (cl_active_pct >= 40.0) & (cl_active_pct <= 60.0)
    ].index
    imbalanced_cl = cl_active_pct[
        (cl_active_pct < 40.0) | (cl_active_pct > 60.0)
    ].index
    bal_auc = cl_auc[cl_auc.index.isin(balanced_cl)].dropna()
    imbal_auc = cl_auc[cl_auc.index.isin(imbalanced_cl)].dropna()
    mw_stat, mw_p = stats.mannwhitneyu(bal_auc, imbal_auc, alternative="two-sided")
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H2 - Class imbalance (balanced vs not)",
            "Test": f"Mann-Whitney: balanced(n={len(bal_auc)}) vs imbalanced(n={len(imbal_auc)})",
            "Statistic": (
                f"U={mw_stat:.1f}; balanced={bal_auc.mean():.4f}; "
                f"imbalanced={imbal_auc.mean():.4f}"
            ),
            "p_value": f"{mw_p:.4f}",
            "Interpretation": "Compares AUC in balanced and imbalanced cell lines.",
        }
    )

    # H3: AUC is threshold-independent, so weak AUC is weak ranking.
    # Use one value per cell line for inference; raw row mean is reported in
    # the figures but the 64 combinations per cell line are not independent.
    auc_vals = tt.groupby("Target")["AUC"].mean().dropna()
    ci_lo, ci_hi = stats.t.interval(
        0.95, df=len(auc_vals) - 1, loc=auc_vals.mean(), scale=stats.sem(auc_vals)
    )
    t_stat, t_p = stats.ttest_1samp(auc_vals, 0.5)
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H3 - Ranking / threshold issue",
            "Test": "One-sample t-test: per-cell-line mean AUC vs 0.5",
            "Statistic": (
                f"t={t_stat:.4f}; mean AUC={auc_vals.mean():.4f}; "
                f"95% CI=[{ci_lo:.4f},{ci_hi:.4f}]"
            ),
            "p_value": f"{t_p:.4e}",
            "Interpretation": (
                "AUC is threshold-independent; a low AUC indicates weak ranking, "
                "not only poor calibration."
            ),
        }
    )

    # Consensus AD: add a compact row for the selected unweighted z=0 run.
    cons_cols = [c for c in consensus_per_cl.columns if c.startswith("CS_cov")]
    if cons_cols:
        cons_means = consensus_per_cl[cons_cols].mean(numeric_only=True)
        best_cons_col = cons_means.idxmax()
        cons_base = float(cons_means.get("CS_cov100", np.nan))
        cons_best = float(cons_means.max())
        rows.append(
            {
                "Configuration": tag,
                "Hypothesis": "AD - Consensus model agreement",
                "Test": "Best CONS-STD AUC coverage vs 100% coverage",
                "Statistic": (
                    f"best={cons_best:.4f} ({best_cons_col}); "
                    f"baseline={cons_base:.4f}; gain={cons_best - cons_base:+.4f}"
                ),
                "p_value": "n/a",
                "Interpretation": (
                    "Checks whether selecting compounds with stronger consensus "
                    "agreement improves AUC."
                ),
            }
        )

    # H4a: variance decomposition.
    z = zscore.copy()
    drug_means = z.mean(axis=1)
    cellline_means = z.mean(axis=0)
    grand_mean = z.stack().mean()

    var_drug = float(drug_means.var())
    var_cellline = float(cellline_means.var())
    var_residual = float(
        z.subtract(drug_means, axis=0)
        .subtract(cellline_means, axis=1)
        .add(grand_mean)
        .stack()
        .var()
    )
    var_total = var_drug + var_cellline + var_residual
    pct_drug = 100 * var_drug / var_total
    pct_cellline = 100 * var_cellline / var_total
    pct_residual = 100 * var_residual / var_total
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H4 - Feature insufficiency (variance decomposition)",
            "Test": "Additive variance decomposition of Z-score matrix",
            "Statistic": (
                f"Drug={pct_drug:.1f}%; Cell-line={pct_cellline:.1f}%; "
                f"Residual={pct_residual:.1f}%"
            ),
            "p_value": "n/a",
            "Interpretation": (
                f"Cell-line identity explains {pct_cellline / pct_drug:.1f}x "
                "more variance than drug identity."
            ),
        }
    )

    # H4b: structural similarity vs response similarity.
    fp_dicts = build_fingerprints(smiles_map)
    common_drugs = [
        drug
        for drug in zscore.index
        if all(drug in fp_dicts[fp_name] for fp_name in fp_dicts)
    ]
    z_sub = zscore.loc[common_drugs].values
    n_drugs = len(common_drugs)
    n_pairs_target = min(5000, n_drugs * (n_drugs - 1) // 2)

    rng = np.random.default_rng(42)
    idx_i_pool = rng.integers(0, n_drugs, size=n_pairs_target * 3)
    idx_j_pool = rng.integers(0, n_drugs, size=n_pairs_target * 3)
    keep = idx_i_pool != idx_j_pool
    idx_i_cand = idx_i_pool[keep][:n_pairs_target]
    idx_j_cand = idx_j_pool[keep][:n_pairs_target]

    response_corr = []
    valid_flags = []
    for i, j in zip(idx_i_cand, idx_j_cand):
        zi, zj = z_sub[i], z_sub[j]
        ok = ~(np.isnan(zi) | np.isnan(zj))
        if ok.sum() >= 10:
            r_s, _ = stats.spearmanr(zi[ok], zj[ok])
            response_corr.append(r_s)
            valid_flags.append(True)
        else:
            valid_flags.append(False)

    valid_flags = np.array(valid_flags)
    idx_i = idx_i_cand[valid_flags]
    idx_j = idx_j_cand[valid_flags]
    response_corr = np.array(response_corr)
    fp_results = {}

    for fp_name, fp_dict in fp_dicts.items():
        fp_all = [fp_dict[d] for d in common_drugs]
        tan_vals = np.array(
            [
                DataStructs.TanimotoSimilarity(fp_all[ii], fp_all[jj])
                for ii, jj in zip(idx_i, idx_j)
            ]
        )
        r_st, p_st = stats.spearmanr(tan_vals, response_corr)
        fp_results[fp_name] = {"tanimoto": tan_vals, "r_s": r_st, "p": p_st}
        rows.append(
            {
                "Configuration": tag,
                "Hypothesis": f"H4 - Feature insufficiency (structural vs response, {fp_name})",
                "Test": (
                    f"Spearman r: pairwise Tanimoto ({fp_name}) vs pairwise "
                    f"response correlation (n={len(response_corr)} drug pairs)"
                ),
                "Statistic": f"r_s={r_st:.4f}",
                "p_value": f"{p_st:.2e}",
                "Interpretation": (
                    f"r^2={100 * r_st**2:.2f}%: {fp_name} Tanimoto explains "
                    "little response-profile similarity."
                ),
            }
        )

    pd.DataFrame(
        {
            "Tanimoto": fp_results["AVALON"]["tanimoto"],
            "Response_Spearman": response_corr,
        }
    ).to_csv(out_struct_resp, index=False)

    pd.DataFrame(
        [
            {
                "Configuration": tag,
                "Fingerprint": fp_name,
                "Spearman_r_s": fp_results[fp_name]["r_s"],
                "p_value": fp_results[fp_name]["p"],
                "r_squared_pct": 100 * fp_results[fp_name]["r_s"] ** 2,
                "n_pairs": len(response_corr),
            }
            for fp_name in fp_results
        ]
    ).to_csv(out_struct_fp, index=False)

    # H4c: within-drug response variance.
    drug_var = zscore.var(axis=1).dropna()
    cellline_var = zscore.var(axis=0).dropna()
    rows.append(
        {
            "Configuration": tag,
            "Hypothesis": "H4 - Feature insufficiency (within-drug response variance)",
            "Test": f"Mean variance of drug response across {zscore.shape[1]} cell lines",
            "Statistic": (
                f"Drug-across-cell-lines var={drug_var.mean():.4f}; "
                f"Cell-line-across-drugs var={cellline_var.mean():.4f}"
            ),
            "p_value": "n/a",
            "Interpretation": (
                "High within-drug variance means the same drug behaves differently "
                "across cell-line contexts."
            ),
        }
    )

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(out_stats, index=False)

    print("\nSaved outputs:")
    print(f"  {out_stats}")
    print(f"  {out_struct_resp}")
    print(f"  {out_struct_fp}")
    print("\nSummary:")
    print(df_stats[["Hypothesis", "Statistic", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
