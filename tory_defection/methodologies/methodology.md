# Tory MP Defection Risk Model: Methodology

## Overview

This model scores Conservative MPs for potential defection risk to Reform UK. The final pipeline uses:
- Speech analysis features derived from Hansard data.
- Career and tenure features derived from Parliament/IFG sources.
- Faction membership indicators from desk research.
- A set of supervised ML models plus an optimized composite score for interpretability.

The implementation lives in `marketing/tory_defection/analysis/final_model` and uses curated source data in `marketing/tory_defection/source_data`.

---

## Pipeline (Current)

1. **Build speech features**
   - `training_speech_analysis.py` and `test_speech_analysis.py` parse Hansard data and create TF-IDF alignment, immigration focus, and radicalization trend features.
   - Outputs: `training_speech_features.csv`, `test_speech_features.csv`.

2. **Assemble training data**
   - `create_training_data.py` merges 2019 GE MPs with career data, speech features, and faction membership.
   - Output: `training_data.csv` with engineered features and the `defected` target.

3. **Assemble test data**
   - `create_test_data.py` merges 2024 sitting MPs with career data, speech features, and faction membership.
   - Output: `test_data.csv` with the same engineered feature set.

4. **Fit models and optimize composite weights**
   - `fit_model_to_training_data.py` trains multiple ML models and optimizes composite score weights using cross-validated AUC.
   - Outputs: `model_comparison.csv`, `variable_importance.csv`, `composite_score_weights.csv`.

5. **Apply models to sitting MPs**
   - `apply_model_to_test_data.py` fits ML models on `training_data.csv`, scores `test_data.csv`, then computes the composite score using `composite_score_weights.csv`.
   - Outputs in `test_results/` (one CSV per model + composite).

---

## Folder Structure (Reviewer Map)

### `marketing/tory_defection/analysis/final_model/`
- `training_speech_analysis.py` / `test_speech_analysis.py`: build speech-derived features
- `create_training_data.py` / `create_test_data.py`: build modeling tables
- `fit_model_to_training_data.py`: train models + optimize composite weights
- `apply_model_to_test_data.py`: score current MPs
- `faction_membership_data.py`: desk-research faction flags + weights
- Outputs:
  - `training_speech_features.csv`, `test_speech_features.csv`
  - `training_data.csv`, `test_data.csv`
  - `composite_score_weights.csv`, `model_comparison.csv`, `variable_importance.csv`
  - `test_results/` (model outputs)

### `marketing/tory_defection/source_data/`
- **`hansard/`**: raw Hansard debate XML used for speech analysis.
- **`elections/`**: 2019/2024 election results used to define training/test MP cohorts.
- **`mp_careers/`**: career/tenure inputs and fetch scripts (see below).
- **`ifg_ministers/`**: Institute for Government ministerial data exports.
- **`defection_tracker/`**: curated list of confirmed defections used as training labels.
- **`mp_demographics.csv`**: age and biographical data used in career features.

### Fetching scripts that build source inputs
These scripts live in `marketing/tory_defection/source_data/mp_careers/` and populate the tenure/ministerial CSVs that feed `create_training_data.py` and `create_test_data.py`:
- `fetch_mp_tenure.py` (historical MPs)
- `fetch_mp_tenure_sitting_mps.py` (current MPs)
- `fetch_ministerial_tenure.py` (historical MPs)
- `fetch_ministerial_tenure_sitting_mps.py` (current MPs)

Other `source_data` inputs (elections, IFG downloads, defection tracker, Hansard dumps) are imported as static files rather than created by scripts in `final_model`.

---

## Data Inputs and Feature Engineering

### Speech features (Hansard)
- TF-IDF alignment with Reform-linked rhetoric.
- Immigration focus ratio.
- Radicalization trend (slope-derived), stored as `radicalizing`.

### Career/tenure features
- Total MP tenure, ministerial tenure, rank, and backbench years.
- Interaction features capturing career stagnation and sidelining.

### Faction membership
- Binary indicators for major Conservative factions.
- Aggregated rightwing/moderate faction scores.

---

## Model Training and Scoring

- Multiple supervised models are trained for comparison: logistic regression (none/L1/L2), random forest, gradient boosting, and PCA + logistic.
- The composite score is a weighted sum of engineered features; weights are optimized on training data to maximize cross-validated AUC, then saved to `composite_score_weights.csv`.
- `apply_model_to_test_data.py` uses those optimized weights (no defaults) to generate the composite output.

---

## Outputs

- **Training outputs**: `model_comparison.csv`, `variable_importance.csv`, `composite_score_weights.csv`.
- **Scoring outputs**: `test_results/sitting_mp_defection_risk_<model>.csv` (one per model + composite).

---

## Assumptions and Limits (Short)

- Defection labels are based on confirmed public cases in the tracker.
- Speech features capture ideological alignment but not private intent.
- The composite score provides interpretability, while ML models provide benchmark performance.

**Last updated**: 19 January 2026
