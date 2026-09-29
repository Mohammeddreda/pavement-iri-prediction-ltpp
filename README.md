# Pavement IRI Prediction Using Machine Learning on LTPP Data

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Companion repository for:
> Mohammed R. Salah, "Dual-Purpose Pavement IRI Prediction with Group-Aware Validation: A Machine Learning Approach Using LTPP Data", submitted to ASCE Journal of Transportation Engineering, Part B: Pavements (under review).

---

## Overview

This repository provides the data and code for a machine-learning study that predicts the International Roughness Index (IRI) of asphalt pavements using the FHWA Long-Term Pavement Performance (LTPP) database.

The key scientific contribution is the demonstration of data leakage in published IRI prediction models. Prior work used random train/test splits on the LTPP panel dataset, where the same road sections (SHRP_ID) appear in both training and testing. We show this inflates R² by 0.32–0.40 points. Applying Group K-Fold cross-validation (grouped by section ID) yields honest estimates, and adding the previous IRI measurement as a feature closes most of the performance gap.

---

## Repository Structure

```
pavement-iri-prediction-ltpp/
│
├── data/                          # Processed scenario datasets (CSV)
│   ├── S1_A.csv                   # Network Planning — base features only
│   ├── S1_B.csv                   # Maintenance Planning — base features only
│   ├── S2_A.csv                   # Network Planning + Bulk Specific Gravity
│   ├── S2_B.csv                   # Maintenance Planning + Bulk Specific Gravity
│   ├── S3_A.csv                   # Network Planning + Mean Asphalt Content
│   ├── S3_B.csv                   # Maintenance Planning + Mean Asphalt Content
│   ├── S4_A.csv                   # Network Planning + BSG + MAC
│   ├── S4_B.csv                   # Maintenance Planning + BSG + MAC
│   ├── X_for_shap.csv             # Feature matrix used for SHAP analysis
│   ├── shap_values.csv            # Computed SHAP values (best model)
│   ├── all_results.csv            # Full cross-validation results (all scenarios × models)
│   ├── tuning_results.csv         # Hyperparameter tuning results (B-scenarios)
│   ├── forward_chaining_results.csv         # Temporal (leave-last-visit-out) validation
│   ├── forward_chaining_visit_counts.csv    # Visit counts per section
│   ├── subset_confound_results.csv          # Section-subset confound check
│   └── residual_diagnostics_summary.csv     # Residual analysis vs IRI, age, ESAL
│
└── pavement_iri_pipeline.py       # Full reproducible ML pipeline
```

---

## Scenario Naming Convention

| Scenario | Use Case | IRI_prev | Additional Material Features |
|----------|----------|----------|------------------------------|
| S1_A | Network Planning | No | None |
| S1_B | Maintenance Planning | Yes | None |
| S2_A | Network Planning | No | Bulk Specific Gravity |
| S2_B | Maintenance Planning | Yes | Bulk Specific Gravity |
| S3_A | Network Planning | No | Mean Asphalt Content |
| S3_B | Maintenance Planning | Yes | Mean Asphalt Content |
| S4_A | Network Planning | No | Both |
| S4_B | Maintenance Planning | Yes | Both |

- Network Planning (_A): Predicts IRI for sections not in the training set — simulates deploying to unseen roads.
- Maintenance Planning (_B): Predicts the next IRI visit for sections already monitored — includes previous IRI as a feature.

---

## Data Source

All data originate from the LTPP database (FHWA), specifically Dry-No-Freeze pavement sections. These sections were selected because their climatic conditions (low precipitation, no freeze–thaw cycles) are representative of arid climates including Egypt, which motivated this study.

| LTPP Source Table | Variable(s) Extracted |
|---|---|
| `PROJECT_HIST_AGE_EXP` | Original construction date |
| `EXPERIMENT_SECTION` | Construction and maintenance assignment dates |
| `TRF_TREND` | Annual ESAL trend |
| `TST_L05B` | AC and granular base layer thickness |
| `TST_AC04` | Mean asphalt content (%) |
| `TST_AC02` | Bulk specific gravity |
| `MON_HSS_PROFILE_SECTION` | IRI (m/km) |
| `MON_DIS_AC_CRACK_INDEX` | Fatigue and wheel path cracking (%) |
| `VW_MERRA_PRECIP_YEAR / VW_MERRA_TEMP_YEAR` | Annual precipitation and mean temperature |

The raw LTPP database is available through the [LTPP InfoPave portal](https://infopave.fhwa.dot.gov/).

---

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
openpyxl
scipy
```

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn openpyxl scipy
```

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Mohammeddreda/pavement-iri-prediction-ltpp.git
   cd pavement-iri-prediction-ltpp
   ```

2. To reproduce the ML results directly, the processed scenario CSVs in `data/` are already provided — the pipeline will use them from Stage 3 onward without the raw Excel file.

   To re-run the full pipeline from scratch (including EDA and scenario building), place the raw LTPP Excel file (`Final Data - R01.xlsx`) in the root directory. This file is derived from the [LTPP InfoPave portal](https://infopave.fhwa.dot.gov/) and is available from the authors upon request.

3. Run the full pipeline:
   ```bash
   python pavement_iri_pipeline.py
   ```

   The pipeline executes 7 sequential stages:
   - Stage 0 — Exploratory Data Analysis (7 figures)
   - Stage 1 — Data preparation and IRI_prev feature engineering
   - Stage 2 — Scenario building (S1–S4, _A and _B variants)
   - Stage 3 — Training and evaluation (Group K-Fold + Random K-Fold)
   - Stage 4 — Hyperparameter tuning (RandomizedSearchCV, 50 iterations)
   - Stage 5 — Out-of-fold predictions for the best model
   - Stage 6 — SHAP feature importance analysis
   - Stage 7 — Publication-ready figures

   All outputs are saved under `Results/` (figures, tables, SHAP outputs).

---

## Key Results

| Use Case | Best Model | R² (Group K-Fold) | RMSE (m/km) | Top Feature |
|---|---|---|---|---|
| Maintenance Planning | Tuned XGBoost (S3_B) | 0.9125 | 0.1682 | Previous IRI (44.7%) |
| Network Planning | Random Forest (S2_A) | 0.5285 | 0.3863 | Wheel Path Cracking (23.7%) |

Data leakage inflated prior-work R² by 0.32–0.40 (from ~0.39 honest to ~0.72 inflated on the same data).

---

## Citation

If you use this data or code, please cite:

```
Mohammed R. Salah (2026). Dual-Purpose Pavement IRI Prediction with Group-Aware Validation:
A Machine Learning Approach Using LTPP Data.
ASCE Journal of Transportation Engineering, Part B: Pavements (under review).
Repository: https://github.com/Mohammeddreda/pavement-iri-prediction-ltpp
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The underlying LTPP data is a U.S. government database; users should also consult [FHWA's LTPP data access terms](https://infopave.fhwa.dot.gov/).
