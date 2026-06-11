# PHENOTYPE_APPLICABILITY_DOMAIN

This directory combines two related analyses that characterise and explain the applicability domain (AD) of the NSCLC phenotype models:

1. **03_APPLICABILITY_DOMAIN** — CLASS-LAG and Consensus AD analyses across all Z-score thresholds
2. **04_HYPOTHESIS_ANALYSIS_H1_H4** — Root-cause diagnostics (H1–H4) explaining near-random NSCLC model performance

---

## Directory Structure

```
PHENOTYPE_APPLICABILITY_DOMAIN/
│
├── 03_APPLICABILITY_DOMAIN/
│   ├── code/
│   │   ├── run_all_AD.py
│   │   ├── gdsc_classLAG_AD_z0.py           # original run (10–100% coverage)
│   │   ├── gdsc_classLAG_AD_z0_test.py      # extended run (1–100% coverage) [NEW]
│   │   ├── gdsc_classLAG_AD_z0_unwt.py      # original run (10–100% coverage)
│   │   ├── gdsc_classLAG_AD_z0_unwt_test.py # extended run (1–100% coverage) [NEW]
│   │   ├── gdsc_classLAG_AD_z05.py
│   │   ├── gdsc_classLAG_AD_z075.py
│   │   ├── gdsc_classLAG_AD_z1.py
│   │   ├── gdsc_consensus_AD_z0.py           # original run (10–100% coverage)
│   │   ├── gdsc_consensus_AD_z0_test.py      # extended run (1–100% coverage) [NEW]
│   │   ├── gdsc_consensus_AD_z0_unwt.py      # original run (10–100% coverage)
│   │   ├── gdsc_consensus_AD_z0_unwt_test.py # extended run (1–100% coverage) [NEW]
│   │   ├── gdsc_consensus_AD_z05.py
│   │   ├── gdsc_consensus_AD_z075.py
│   │   ├── gdsc_consensus_AD_z1.py
│   │   └── verify_classLAG_table_S5.py
│   └── results/
│       ├── AD_run_summary.csv
│       ├── gdsc_classLAG_AD_z0_per_cellline.csv / _summary.csv   # updated: 1–100% coverage [REV]
│       ├── gdsc_classLAG_AD_z0_unwt_per_cellline.csv / _summary.csv  # updated: 1–100% [REV]
│       ├── gdsc_classLAG_AD_z05_per_cellline.csv / _summary.csv
│       ├── gdsc_classLAG_AD_z075_per_cellline.csv / _summary.csv
│       ├── gdsc_classLAG_AD_z1_per_cellline.csv / _summary.csv
│       ├── gdsc_consensus_AD_z0_per_cellline.csv / _summary.csv / _comparison.csv  # updated: 1–100% [REV]
│       ├── gdsc_consensus_AD_z0_unwt_per_cellline.csv / _summary.csv / _comparison.csv  # updated: 1–100% [REV]
│       ├── gdsc_consensus_AD_z05_per_cellline.csv / _summary.csv / _comparison.csv
│       ├── gdsc_consensus_AD_z075_per_cellline.csv / _summary.csv / _comparison.csv
│       ├── gdsc_consensus_AD_z1_per_cellline.csv / _summary.csv / _comparison.csv
│       ├── nsclc_diagnostic_statistics_z0.csv
│       ├── nsclc_diagnostic_statistics_z0_unwt.csv
│       ├── nsclc_structural_vs_response_z0.csv
│       ├── nsclc_structural_vs_response_z0_unwt.csv
│       ├── nsclc_structural_vs_response_by_fp_z0.csv
│       └── nsclc_structural_vs_response_by_fp_z0_unwt.csv
│
└── 04_HYPOTHESIS_ANALYSIS_H1_H4/
    ├── code/
    │   ├── analyze_nsclc_failure.py
    │   ├── nsclc_h2_data_quality.py
    │   ├── nsclc_h3_ranking.py
    │   └── nsclc_h4_feature_insufficiency.py
    └── results/
        ├── nsclc_diagnostic_statistics_z0.csv
        ├── nsclc_diagnostic_statistics_z0_unwt.csv
        ├── nsclc_structural_vs_response_z0.csv
        ├── nsclc_structural_vs_response_z0_unwt.csv
        ├── nsclc_structural_vs_response_by_fp_z0.csv
        └── nsclc_structural_vs_response_by_fp_z0_unwt.csv
```

---

## 03_APPLICABILITY_DOMAIN — CLASS-LAG and Consensus AD

Evaluates whether test compounds fall within the applicability domain of the NSCLC phenotype models using two complementary AD methods.

### CLASS-LAG AD

Lazy-learning AD: the CLASS-LAG score = min(p, 1−p) from a per-cell-line Random Forest (Sushko et al. 2010, Eq. 2). Test compounds are ranked by ascending CLASS-LAG (most confident first) and mean ROC-AUC is evaluated at each coverage level.

**Coverage range:** the `_test.py` scripts (revised second submission) evaluate coverage from **1% to 100%** in fine steps (1, 2, 3, 4, 5, 6, 8, 9, 10, 20, …, 100%). The original `_z0.py` / `_z0_unwt.py` scripts evaluated 10–100% only. The result CSVs in this repository now contain 1–100% coverage (updated for second revision).

| File pattern | Description |
|---|---|
| `gdsc_classLAG_AD_z0_per_cellline.csv` | Per-cell-line mean AUC at each coverage level (Z=0, weighted RF). Source for Table S7. |
| `gdsc_classLAG_AD_z0_summary.csv` | Summary across 108 cell lines (Z=0, weighted). Best AUC = 0.710 at 3% coverage. |
| `gdsc_classLAG_AD_z0_unwt_*.csv` | Same for Z=0 unweighted. Best AUC = 0.674 at 1–2% coverage. |
| `gdsc_classLAG_AD_z05_*.csv` | Same for Z=−0.5 (10–100% coverage only). |
| `gdsc_classLAG_AD_z075_*.csv` | Same for Z=−0.75 (10–100% coverage only). |
| `gdsc_classLAG_AD_z1_*.csv` | Same for Z=−1.0 (10–100% coverage only). |

### Consensus AD

12-model ensemble (AVALON × ECFP × MACCS fingerprints, RF × LR × SVM × XGBoost algorithms). CONS-STD = standard deviation of the 12 model probabilities (Sushko Eq. 3); CONS-STD-PROB = Gaussian tail probability (Eqs. 4–5); CLASS-LAG = min(p̄, 1−p̄) from the 12-model mean probability.

**Coverage range:** the `_test.py` scripts evaluate 1–100% coverage. Result CSVs for z0 and z0_unwt are updated to 1–100% for the second revision.

| File pattern | Description |
|---|---|
| `gdsc_consensus_AD_z0_per_cellline.csv` | Per-cell-line mean AUC at each coverage level (Z=0, weighted). Source for Table S8. |
| `gdsc_consensus_AD_z0_summary.csv` | Summary across 108 cell lines (Z=0, weighted). Consensus CLASS-LAG best AUC = **0.769** at 1–2% coverage. |
| `gdsc_consensus_AD_z0_comparison.csv` | Coverage vs mean AUC table for CLASS-LAG, CONS-STD, and CONS-STD-PROB (Z=0, weighted). Used for Figure S17 Panel A. |
| `gdsc_consensus_AD_z0_unwt_*.csv` | Same for Z=0 unweighted. CONS-STD-PROB best AUC = 0.692 at 1–2%. |
| `gdsc_consensus_AD_z05_*.csv` | Same for Z=−0.5 (10–100% coverage only). |
| `gdsc_consensus_AD_z075_*.csv` | Same for Z=−0.75 (10–100% coverage only). |
| `gdsc_consensus_AD_z1_*.csv` | Same for Z=−1.0 (10–100% coverage only). |

### Master summary

| File | Description |
|---|---|
| `AD_run_summary.csv` | Master summary of all AD runs across all thresholds and weighting schemes. |

### Running the AD scripts

```bash
conda activate drugai

# --- Extended coverage (1–100%): use the _test.py scripts ---
# CLASS-LAG, Z=0, weighted RF:
python 03_APPLICABILITY_DOMAIN/code/gdsc_classLAG_AD_z0_test.py

# CLASS-LAG, Z=0, unweighted RF:
python 03_APPLICABILITY_DOMAIN/code/gdsc_classLAG_AD_z0_unwt_test.py

# Consensus 12-model, Z=0, weighted:
python 03_APPLICABILITY_DOMAIN/code/gdsc_consensus_AD_z0_test.py

# Consensus 12-model, Z=0, unweighted:
python 03_APPLICABILITY_DOMAIN/code/gdsc_consensus_AD_z0_unwt_test.py

# --- Original coverage (10–100%): use the non-_test scripts ---
python 03_APPLICABILITY_DOMAIN/code/run_all_AD.py

# Or skip scripts whose output already exists:
python 03_APPLICABILITY_DOMAIN/code/run_all_AD.py --skip-done
```

All outputs are written to `03_APPLICABILITY_DOMAIN/results/`.

> **Note:** The `_test.py` scripts (z0 and z0_unwt) are the canonical scripts for the second revision. They extend coverage evaluation to 1% and produce the results reported in Tables S7–S8 and Figure S17 of the revised manuscript.

---

## 04_HYPOTHESIS_ANALYSIS_H1_H4 — Root-Cause Analysis of NSCLC Model Performance

Tests four competing hypotheses to explain why NSCLC phenotype models achieve near-random performance (mean AUC ≈ 0.550).

| Hypothesis | Tested | Conclusion |
|---|---|---|
| H1 — Wrong algorithm | Does model architecture (RF, GB, LR, SVM) explain poor performance? | All architectures fail similarly; not the cause. |
| H2 — Data quality / class imbalance | Does the 37.6% sensitive rate or data quality cause poor performance? | Class imbalance is moderate; not the primary cause. |
| H3 — Threshold / ranking issue | Does Z-score binarisation threshold introduce label noise? | Performance is poor across all thresholds (Z=0 to Z=−1.0); not the cause. |
| H4 — Feature insufficiency | Do 2D fingerprints lack the information to predict cell-line-specific drug response? | Structural similarity poorly predicts response concordance; **feature insufficiency is the primary cause.** |

### Code

| File | Description |
|---|---|
| `analyze_nsclc_failure.py` | Combined H1–H4 diagnostic runner. Run as: `python analyze_nsclc_failure.py --tag z0` |
| `nsclc_h2_data_quality.py` | Standalone H2 data quality and class imbalance analysis. |
| `nsclc_h3_ranking.py` | Standalone H3 Z-score threshold sensitivity analysis. |
| `nsclc_h4_feature_insufficiency.py` | Standalone H4 feature insufficiency analysis (pairwise Tanimoto vs response concordance). |

### Results

| File | Description |
|---|---|
| `nsclc_diagnostic_statistics_z0.csv` | H1–H4 diagnostic statistics for Z=0 weighted: AUC inside/outside AD, Tanimoto distributions, class imbalance metrics. |
| `nsclc_diagnostic_statistics_z0_unwt.csv` | Same for Z=0 unweighted. |
| `nsclc_structural_vs_response_z0.csv` | Pairwise structural similarity vs response concordance, Z=0 weighted. Key evidence for H4. |
| `nsclc_structural_vs_response_z0_unwt.csv` | Same for Z=0 unweighted. |
| `nsclc_structural_vs_response_by_fp_z0.csv` | Same broken down by fingerprint type (ECFP, MACCS, Avalon, ECFP_COUNT), Z=0 weighted. |
| `nsclc_structural_vs_response_by_fp_z0_unwt.csv` | Same for Z=0 unweighted. |

### Running the hypothesis scripts

```bash
conda activate ai_ret

# Combined H1-H4 runner (Z=0 weighted):
python 04_HYPOTHESIS_ANALYSIS_H1_H4/code/analyze_nsclc_failure.py --tag z0

# Combined H1-H4 runner (Z=0 unweighted):
python 04_HYPOTHESIS_ANALYSIS_H1_H4/code/analyze_nsclc_failure.py --tag z0_unwt

# Individual analyses:
python 04_HYPOTHESIS_ANALYSIS_H1_H4/code/nsclc_h2_data_quality.py
python 04_HYPOTHESIS_ANALYSIS_H1_H4/code/nsclc_h3_ranking.py
python 04_HYPOTHESIS_ANALYSIS_H1_H4/code/nsclc_h4_feature_insufficiency.py
```

---

## References

### AD Methods (cited in all `gdsc_classLAG_AD_*.py` and `gdsc_consensus_AD_*.py` scripts)

1. Sushko I, Novotarskyi S, Körner R, Pandey AK, Rupp M, Teetz W, Brandmaier S, Abdelaziz A, Prokopenko VV, Tanchuk VY, Todeschini R, Varnek A, Marcou G, Ertl P, Potemkin V, Grishina M, Gasteiger J, Schwab CH, Saller H, Kovalishyn V, Tetko IV.
   **Applicability Domains for Classification Problems: Benchmarking of Distance to Models for Ames Mutagenicity Set.**
   *J. Chem. Inf. Model.* 2010;50(12):2094–2111.
   DOI: [10.1021/ci100253r](https://doi.org/10.1021/ci100253r)
   *(CLASS-LAG: AD score = min(p, 1−p); Consensus AD: CONS-STD, CONS-STD-PROB)*

### Class-Imbalance Correction (cited in `multitask_classifier_weighted.py`)

2. Lenhof K, Eckhart L, Gerstner N, et al.
   **Simultaneous regression and classification for drug sensitivity prediction using an advanced random forest method.**
   *Sci. Rep.* 2022;12:13458.
   DOI: [10.1038/s41598-022-17609-x](https://doi.org/10.1038/s41598-022-17609-x)
   *(Sample-weight correction: w_i = N_res/N_sens for sensitive samples)*

### Threshold Selection

3. Youden WJ.
   **Index for rating diagnostic tests.**
   *Cancer.* 1950;3(1):32–35.
   DOI: [10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3](https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3)
   *(Youden's J = TPR − FPR; used for binary classification threshold selection throughout)*

### Dataset

4. Garnett MJ, Edelman EJ, Heidorn SJ, et al.
   **Systematic identification of genomic markers of drug sensitivity in cancer cells.**
   *Nature.* 2012;483(7391):570–575.
   DOI: [10.1038/nature11005](https://doi.org/10.1038/nature11005)

5. Iorio F, Knijnenburg TA, Vis DJ, et al.
   **A Landscape of Pharmacogenomic Interactions in Cancer.**
   *Cell.* 2016;166(3):740–754.
   DOI: [10.1016/j.cell.2016.06.017](https://doi.org/10.1016/j.cell.2016.06.017)
   *(GDSC1 and GDSC2 drug sensitivity Z-scores used as phenotype labels)*

### Molecular Fingerprints

6. Rogers D, Hahn M.
   **Extended-Connectivity Fingerprints.**
   *J. Chem. Inf. Model.* 2010;50(5):742–754.
   DOI: [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
   *(ECFP and ECFP_COUNT fingerprints)*

### Scaffold and Clustering-Based Splits

7. Bemis GW, Murcko MA.
   **The Properties of Known Drugs. 1. Molecular Frameworks.**
   *J. Med. Chem.* 1996;39(15):2887–2893.
   DOI: [10.1021/jm9602928](https://doi.org/10.1021/jm9602928)

   *(Murcko scaffold-based train/test split)*

8. Butina D.
   **Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets.**
   *J. Chem. Inf. Comput. Sci.* 1999;39(4):747–750.
   DOI: [10.1021/ci9803381](https://doi.org/10.1021/ci9803381)
   *(Butina clustering-based train/test split)*

9. McInnes L, Healy J, Melville J.
   **UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.**
   *arXiv:1802.03426.* 2018.
   URL: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
   *(UMAP-clustering-based train/test split)*

### Software

10. RDKit: Open-source cheminformatics.
    URL: [https://www.rdkit.org](https://www.rdkit.org)
    *(Molecular fingerprint computation, scaffold perception, Tanimoto similarity)*

11. Pedregosa F, Varoquaux G, Gramfort A, et al.
    **Scikit-learn: Machine Learning in Python.**
    *J. Mach. Learn. Res.* 2011;12:2825–2830.
    URL: [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)
    *(All classification algorithms: RandomForest, SVM, ExtraTrees, LogisticRegression, KNeighbors, GradientBoosting, GaussianNB)*
