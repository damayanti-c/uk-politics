# Model Specification: Basic Demographics Model

**Model Version:** v0.1 (Initial Baseline)
**Date Created:** January 15, 2026
**Status:** Superseded by subsequent iterations

---

## Theory of Change

This initial baseline model hypothesizes that Conservative MP defection to Reform UK can be predicted using **basic parliamentary career demographics alone**, without analyzing speech content or voting patterns.

### Core Hypothesis
MPs are more likely to defect if they have:
1. **Career stagnation** - Long time in Parliament with limited ministerial advancement
2. **Red Wall vulnerability** - Represent constituencies won in 2019 that are vulnerable to Reform UK
3. **Parliamentary rebellion history** - Pattern of voting against party whip
4. **Limited ministerial experience** - Never held significant government positions
5. **Marginal seat pressure** - Small majority, vulnerable to Reform UK challenge

### Assumptions
- Defection is primarily driven by **career frustration** and **constituency pressure**
- Speech content and ideology are secondary to structural career factors
- Public data from Parliament API and election results is sufficient

---

## Variables Used

### Demographic Variables (7 features)

1. **years_in_parliament** - Total years serving as MP
   - Source: Parliament API (members endpoint)
   - Range: 0-50+ years
   - Hypothesis: Long tenure without advancement → higher risk

2. **ministerial_positions_held** - Count of ministerial roles
   - Source: Parliament API (government posts)
   - Range: 0-10+
   - Hypothesis: Fewer positions → higher risk

3. **years_in_ministerial_roles** - Total time in government
   - Source: Calculated from government posts
   - Range: 0-20+ years
   - Hypothesis: Less ministerial time → higher risk

4. **red_wall_seat** - Binary indicator
   - Source: 2019 election results
   - Values: 0 (safe seat) or 1 (Red Wall)
   - Hypothesis: Red Wall MPs face more Reform UK pressure

5. **majority_percentage** - 2024 election victory margin
   - Source: Election results data
   - Range: 0-50%
   - Hypothesis: Smaller majority → higher risk

6. **rebellion_count** - Votes against party whip (2019-2024)
   - Source: Public Whip / They Work For You
   - Range: 0-100+
   - Hypothesis: More rebellions → higher risk

7. **time_since_last_ministerial_role** - Years since last government position
   - Source: Calculated from government posts
   - Range: 0-20+
   - Hypothesis: Long gap → career stagnation → higher risk

---

## Techniques Used

### Data Collection
- **Parliament API** - Official UK Parliament REST API
- **Election results** - House of Commons Library data
- **Public Whip** - Voting record database

### Feature Engineering
- Simple arithmetic calculations (years, counts)
- Binary indicators (Red Wall vs safe seat)
- No normalization or scaling (initial baseline)

### Modeling Approach
- **Logistic Regression** - Primary model for interpretability
- **Random Forest** - Comparison model for feature importance
- **Gradient Boosting** - Advanced ensemble for comparison

### Evaluation
- **Cross-validation** - 5-fold stratified
- **No ground truth data** - This is purely predictive/exploratory
- **Feature importance** - Ranked by logistic regression coefficients

---

## Folder Structure

```
basic_demogs_model/
│
├── MODEL_SPEC.md                          (this file)
│
├── tory_defection_analysis.py             Main analysis script
│   └── Contains:
│       - Data source configuration
│       - Parliament API data fetching
│       - Feature engineering
│       - Model training (Logistic Regression, Random Forest, GBM)
│       - Risk score generation
│       - Results output
│
├── run_analysis.py                        Simple runner script
│   └── Executes tory_defection_analysis.py
│   └── Minimal configuration
│
├── mp_defection_risk_scores.csv           Model output (650 MPs)
│   └── Columns:
│       - name: MP name
│       - defection_risk_score: 0-100 risk score
│       - years_in_parliament
│       - ministerial_positions_held
│       - majority_percentage
│       - red_wall_seat
│       - rebellion_count
│
└── defection_risk_report.txt              Summary report
    └── Top 20 highest-risk MPs
    └── Feature importance rankings
    └── Model performance metrics
```

---

## Results Summary

### Top Risk Factors (Feature Importance)
1. **years_in_parliament** - Longest-serving MPs at higher risk
2. **time_since_last_ministerial_role** - Career stagnation indicator
3. **red_wall_seat** - Constituency pressure
4. **rebellion_count** - Anti-party voting pattern
5. **ministerial_positions_held** - (Negative) More positions = lower risk

### Model Performance
- **No validation data available** - Cannot compute accuracy metrics
- Purely exploratory/predictive
- Risk scores range from 0-100

### Top Predicted Defection Risks
(See defection_risk_report.txt for full list)

---

## Limitations

### Critical Gaps
1. **No speech content analysis** - Ignores ideological alignment with Reform UK
2. **No ground truth** - Cannot validate predictions against actual defections
3. **No temporal dynamics** - Static snapshot, no trend analysis
4. **Limited features** - Only 7 variables, missing key indicators
5. **Crude proxies** - "Red Wall seat" is a blunt instrument

### Data Quality Issues
1. Parliament API incomplete for historical data
2. Rebellion counts may be outdated
3. Ministerial role classifications inconsistent

### Methodological Issues
1. **Class imbalance unaddressed** - Very few actual defections
2. **No regularization** - Risk of overfitting
3. **No feature scaling** - Variables on different scales
4. **Circular reasoning** - Model predicts rare events without seeing any

---

## Why This Model Was Superseded

This baseline model was replaced by subsequent iterations that added:

1. **Speech content analysis** (basic_demogs_keywords_model)
   - Keyword counting for Reform UK rhetoric
   - Immigration/sovereignty focus detection

2. **Semantic vectorization** (basic_demogs_vectorisation_model)
   - TF-IDF vectorization of full speeches
   - Cosine similarity to Reform UK rhetoric

3. **Training on ground truth** (training_tf_idf_model)
   - Best for Britain defection tracker (23 confirmed defections)
   - Proper train/test methodology
   - Volume-normalized speech metrics

The basic demographics model provided useful baseline insights but lacked the predictive power of speech analysis combined with demographics.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn requests

# Run analysis
cd marketing/tory_defection/analysis/basic_demogs_model
python run_analysis.py

# Output files created:
# - mp_defection_risk_scores.csv
# - defection_risk_report.txt
```

---

**Next Model:** See [../basic_demogs_keywords_model/MODEL_SPEC.md](../basic_demogs_keywords_model/MODEL_SPEC.md)
