# Model Specification: Training TF-IDF Model (Validated on Past Defections)

**Model Version:** v1.0 (Production-Ready)
**Date Created:** January 16, 2026
**Status:** Current best model ✅

---

## Theory of Change

This is the first **validated defection risk model**, trained on 19 actual Conservative MP defections to Reform UK using supervised machine learning.

### Core Hypothesis
Conservative MPs defect to Reform UK when they combine:
1. **Ideological alignment** - Speeches semantically similar to Reform rhetoric (volume-normalized)
2. **Career frustration** - Long backbench tenure without ministerial advancement
3. **No ministerial protection** - Current/former ministers have too much to lose
4. **Mid-to-senior career** - Not too young (career concerns) or too old (retirement imminent)
5. **Constituency factors** - Secondary to personal ideology and career

### Revolutionary Breakthroughs

#### 1. Ground Truth Training Data
**Best for Britain Defection Tracker** - 23 confirmed Conservative MP defections since January 1, 2024:
- Lee Anderson, Robert Jenrick, Nadine Dorries, Andrea Jenkyns, Danny Kruger, Jonathan Gullis, etc.
- 19 of 23 found in Hansard speech data (2020-2025)
- First time model can be **validated** against actual outcomes

#### 2. Volume Normalization Formula
**Critical insight:** Rishi Sunak problem identified and solved

**Problem (discovered in v0.2):**
- Prime Minister talks about immigration frequently
- Keyword counting: Sunak ranked HIGH risk (wrong!)
- Issue: Volume bias - more speeches = more keywords automatically

**Solution (implemented in v1.0):**
```python
reform_alignment_normalized = (
    reform_alignment_raw × immigration_proportion
) / (1 + log(total_speeches) / 10)
```

**Impact:**
- Rishi Sunak: Raw 0.174 → Normalized 0.003 (-98% decrease) ✅
- Laura Trott: Raw 0.520 → Normalized 0.000 (-100% decrease) ✅
- Robert Jenrick: Raw 0.285 → Normalized 0.023 (-19% decrease, genuinely aligned) ✅

#### 3. Historical MP Analysis
**Expanded from 117 current MPs → 542 historical Conservative MPs:**
- All Conservative MPs who served 2020-2025
- Includes defectors who lost seats in July 2024 election
- Enables proper training/validation methodology

#### 4. Machine Learning Optimization
**From manual weights → data-driven coefficients:**
- Logistic regression with L2 regularization (C=1.0)
- Balanced class weights (handles 19:523 defector ratio)
- 5-fold stratified cross-validation
- **97.8% AUC achieved** (excellent discrimination)

---

## Variables Used (16 total)

### Speech Features (6 variables)

1. **reform_alignment_raw** - TF-IDF cosine similarity to Reform rhetoric
   - Method: TF-IDF vectorization of Hansard speeches
   - Templates: 20 Reform UK immigration talking points
   - Range: 0-1
   - **Model weight: 4.9%** (4th most important speech feature)

2. **reform_alignment_normalized** - Volume-adjusted Reform alignment
   - Formula: (raw × immigration_proportion) / (1 + log(speeches) / 10)
   - Range: 0-0.05 typically
   - **Model weight: 2.2%** (univariate Cohen's d = 1.054, p<0.0001 ⭐)
   - **Most predictive speech feature**

3. **radicalization_slope** - Temporal trend in Reform alignment
   - Method: Linear regression on quarterly scores
   - Range: -0.2 to +0.2 per quarter
   - **Model weight: 2.4%**

4. **extremism_percentile** - Rank among Conservative MPs
   - Range: 0-100%
   - **Model weight: 1.2%**

5. **immigration_proportion** - % of speeches focused on immigration
   - Keywords: immigra, asylum, border, migrant, rwanda, boat, channel, refuge
   - Range: 0-30% typically
   - **Model weight: 5.1%** (Cohen's d = 0.903, p<0.0001 ⭐)
   - **Second most predictive speech feature**

6. **total_speeches** - Speech volume (normalization context)
   - Range: 0-2000+
   - **Model weight: 1.7%**

### Demographic Features (10 variables)

7. **age** - Age in years
   - Source: Wikidata SPARQL queries (454/542 = 83.8% coverage)
   - Range: 24-76 years
   - **Model weight: 2.8%**

8. **retirement_proximity_score** - (age - 55) / 15, clipped [0, 1]
   - Hypothesis: Near retirement → less likely to defect
   - **Model weight: 5.1%** (protective effect)

9. **estimated_years_as_mp** - Years in Parliament
   - **Model weight: 29.6%** ⭐⭐⭐ **MOST IMPORTANT FEATURE**
   - Direction: Positive (longer tenure → higher risk)

10. **backbench_years** - Years not holding ministerial role
    - **Model weight: 27.9%** ⭐⭐⭐ **SECOND MOST IMPORTANT**
    - Direction: Negative (coefficient interpretation complex due to multicollinearity with years_as_mp)

11. **ever_minister** - Binary: held any ministerial position
    - **Model weight: 5.5%**
    - Direction: Protective (ministers less likely to defect)
    - Defectors: 5.3% were ministers vs 25.2% baseline

12. **total_minister_years** - Total time in government roles
    - Source: IFG Ministers Database (172 Conservative ministers)
    - **Model weight: 4.9%** (protective)

13. **highest_ministerial_rank** - 0=None, 2=Under-Sec, 3=Minister, 4=Cabinet
    - **Model weight: 4.6%** (protective)

14. **years_since_last_ministerial_role** - Time since last government position
    - **Model weight: 0.4%** (minimal impact in full model)

15. **cabinet_level** - Binary: ever held Cabinet position
    - **Model weight: 0.1%** (minimal impact, subsumed by rank)

16. **career_stagnation_score** - (years_since_role / 10) × (1 - retirement_proximity)
    - Hypothesis: Stagnation risky unless near retirement
    - **Model weight: 1.6%**

---

## Techniques Used

### TF-IDF Vectorization (Enhanced)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,        # Top 5000 terms
    ngram_range=(1, 2),       # Unigrams and bigrams
    min_df=5,                 # Minimum document frequency
    max_df=0.7,               # Maximum document frequency
    stop_words='english',     # Remove common words
    sublinear_tf=True         # Use log(TF) instead of TF
)

# Fit on sample + Reform rhetoric
all_texts = speeches_sample + REFORM_IMMIGRATION_RHETORIC
vectorizer.fit(all_texts)

# Transform MP speeches
mp_speech_vectors = vectorizer.transform(mp_speeches)

# Transform Reform templates
reform_vectors = vectorizer.transform(REFORM_TEMPLATES)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(mp_speech_vectors, reform_vectors)
reform_alignment = similarity.mean(axis=1)
```

### Volume Normalization

**Three normalization methods tested:**

1. **Volume-adjusted** - Normalize by log of speech volume
   ```python
   volume_adj = raw / (1 + log(total_speeches + 1) / 10)
   ```

2. **Proportion-adjusted** - Weight by immigration focus
   ```python
   proportion_adj = raw × immigration_proportion
   ```

3. **Combined (USED)** - Both normalizations
   ```python
   normalized = (raw × immigration_proportion) / (1 + log(total_speeches + 1) / 10)
   ```

**Validation:**
- Normalized alignment: Cohen's d = 1.054, p<0.0001 ⭐
- Raw alignment: Cohen's d = 0.279, p=0.232 (NOT significant)
- **Normalized version is 4x more predictive**

### Logistic Regression with Regularization

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression
model = LogisticRegression(
    penalty='l2',              # L2 regularization
    C=1.0,                     # Regularization strength (optimized via CV)
    solver='liblinear',        # Fast for small datasets
    random_state=42,
    max_iter=1000,
    class_weight='balanced'    # Handle 19:523 class imbalance
)

model.fit(X_scaled, y)
```

**Regularization strength tested:** C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
- **Optimal:** C=1.0 (AUC=0.978)
- Too strong (C=0.001): Underfit (AUC=0.859)
- Too weak (C=100): Slight overfit (AUC=0.971)

### Cross-Validation

**5-Fold Stratified K-Fold:**
- Maintains 19:523 defector ratio in each fold
- **Individual fold AUCs:** [0.955, 0.993, 0.962, 0.990, 0.990]
- **Mean: 0.978 ± 0.016** (very stable)

### Data Sources

1. **Hansard Speeches** - All Conservative MP speeches (2020-2025)
   - 199,701 speeches total
   - 105,300 Conservative speeches
   - 542 unique Conservative speakers

2. **Best for Britain Defection Tracker** - Ground truth
   - 23 defections (Jan 2024 - Jan 2026)
   - 19 found in Hansard (82.6% coverage)

3. **IFG Ministers Database** - Ministerial careers
   - GitHub: instituteforgov/ifg-ministers-database-public
   - 172 Conservative ministers identified
   - Appointment dates, ranks, portfolios

4. **Wikidata** - MP ages via SPARQL queries
   - 454/542 MPs (83.8% coverage)
   - Date of birth → age calculation
   - Retirement proximity scoring

5. **Parliament API** - Demographics
   - MP names, constituencies
   - Basic career information

---

## Folder Structure

```
training_tfidf_model/
│
├── MODEL_SPEC.md                              (this file)
│
├── TRAINING_ANALYSIS_REPORT.md                Comprehensive analysis report
│   └── Full technical documentation
│   └── Results, validation, recommendations
│   └── 28-page detailed analysis
│
├── enhanced_speech_tfidf.py                   Main TF-IDF speech analysis
│   └── Loads 199,701 Hansard speeches
│   └── Filters to 542 Conservative MPs
│   └── TF-IDF vectorization with Reform templates
│   └── Cosine similarity calculation
│   └── Outputs: enhanced_speech_tfidf.csv
│
├── normalize_speech_metrics.py                Volume normalization
│   └── Loads TF-IDF results
│   └── Calculates immigration proportion
│   └── Applies normalization formula
│   └── Outputs: enhanced_speech_tfidf_normalized.csv
│
├── identify_conservative_speakers.py          Build Conservative MP list
│   └── Combines current MPs + defectors + IFG ministers
│   └── 669 unique Conservative MP names identified
│   └── Cross-references with Hansard data
│   └── Outputs: conservative_speakers_in_hansard.csv (542 with speeches)
│
├── fetch_ifg_ministerial_data.py              Download ministerial careers
│   └── Fetches CSVs from IFG GitHub repository
│   └── Parses appointments, persons, posts, organisations
│   └── Calculates ministerial rank hierarchy
│   └── Outputs: mp_career_features.csv
│
├── fetch_mp_ages_from_wikidata.py             Query Wikidata for ages
│   └── SPARQL queries to Wikidata Query Service
│   └── Batch querying (50 MPs at a time)
│   └── Age calculation and retirement proximity
│   └── Outputs: mp_ages_from_wikidata.csv
│   └── Updates: mp_career_features.csv with age data
│
├── enhanced_speech_tfidf.csv                  Speech analysis results (542 MPs)
│   └── reform_alignment_score (raw TF-IDF)
│   └── radicalization_slope
│   └── extremism_percentile
│   └── total_speeches
│
├── enhanced_speech_tfidf_normalized.csv       Volume-normalized scores (542 MPs)
│   └── All columns from enhanced_speech_tfidf.csv
│   └── Plus: reform_alignment_volume_adj
│   └── Plus: reform_alignment_proportion_adj
│   └── Plus: reform_alignment_normalized (USED IN MODEL)
│   └── Plus: immigration_proportion
│
├── conservative_mps_comprehensive.csv         All Conservative MP names (669)
│   └── name, has_hansard_speeches
│
├── conservative_speakers_in_hansard.csv       MPs with speeches (542)
│   └── name, has_hansard_speeches=True
│
├── mp_career_features.csv                     Ministerial & demographic data (650 MPs)
│   └── name, ever_minister, total_minister_years
│   └── highest_ministerial_rank, portfolio_count
│   └── years_since_last_ministerial_role
│   └── age, retirement_proximity_score
│   └── backbench_years, career_stagnation_score
│
├── mp_ages_from_wikidata.csv                  Wikidata age results (454 MPs)
│   └── name, date_of_birth, age, retirement_proximity_score
│
└── training_past_defections/                  Training analysis subfolder
    │
    ├── compile_defection_ground_truth.py      Best for Britain defections
    │   └── Manual compilation of 23 defections
    │   └── Includes dates, former roles, ministerial rank
    │   └── Outputs: ../../source_data/defection_tracker/defections_2024.csv
    │
    ├── prepare_training_data.py               Merge all features
    │   └── Loads speech, defections, demographics, career data
    │   └── Merges on speaker name (cleaned)
    │   └── Adds defection labels (binary)
    │   └── Outputs: training_data_2024.csv
    │
    ├── training_eda.py                        Exploratory data analysis
    │   └── Univariate analysis (t-tests, Cohen's d, p-values)
    │   └── Correlation analysis
    │   └── Preliminary interaction effects
    │   └── Outputs: training_eda_univariate.csv, training_eda_correlation.csv
    │
    ├── optimize_model_weights.py              Logistic regression training
    │   └── Regularization optimization (C search)
    │   └── Feature importance ranking
    │   └── Cross-validation (5-fold stratified)
    │   └── Feature set comparison (speech vs demographics)
    │   └── Outputs: model_optimization_results.csv, model_predictions.csv
    │
    ├── analyze_interaction_effects.py         Interaction term analysis
    │   └── Career stagnation × retirement proximity
    │   └── Reform alignment × ministerial status
    │   └── Reform alignment × age
    │   └── Backbench years × Reform alignment
    │   └── Immigration focus × ministerial status
    │   └── Outputs: interaction_effects_analysis.csv
    │
    ├── training_data_2024.csv                 Final training dataset (542 MPs)
    │   └── All 16 features + defection label
    │   └── 19 defectors, 523 non-defectors
    │
    ├── training_eda_univariate.csv            Statistical test results
    │   └── Feature-by-feature comparison (defectors vs non-defectors)
    │   └── Mean, SD, Cohen's d, t-statistic, p-value
    │
    ├── training_eda_correlation.csv           Correlation matrix
    │   └── 16 features × 16 features
    │   └── Correlations with defection outcome
    │
    ├── model_optimization_results.csv         Feature importance
    │   └── Logistic regression coefficients
    │   └── Absolute coefficients, scaled importance (%)
    │
    ├── model_predictions.csv                  Predicted probabilities (542 MPs)
    │   └── speaker_name, actual_defection, predicted_probability
    │   └── Sorted by risk (highest first)
    │
    └── interaction_effects_analysis.csv       Full dataset with interaction terms
        └── All original features
        └── Plus 5 interaction terms
        └── Binary indicators for grouping
```

---

## Results Summary

### Model Performance

**Cross-Validated AUC: 0.978 ± 0.016** ⭐⭐⭐
- Individual folds: [0.955, 0.993, 0.962, 0.990, 0.990]
- Excellent discrimination
- Stable across folds (low variance)

**Recall at Thresholds:**
| Top K MPs | Defectors Captured | Recall | Precision |
|-----------|-------------------|--------|-----------|
| Top 10 | 9/19 | 47.4% | 90.0% |
| **Top 20** | **16/19** | **84.2%** | **80.0%** |
| Top 50 | 18/19 | 94.7% | 36.0% |
| Top 100 | 19/19 | 100.0% | 19.0% |

**Practical Interpretation:** Monitoring the top 20 MPs captures 84% of defectors with 80% precision.

### Feature Importance (Top 10)

| Rank | Feature | Weight | Direction | Cohen's d |
|------|---------|--------|-----------|-----------|
| 1 | estimated_years_as_mp | 29.6% | + (risk) | N/A |
| 2 | backbench_years | 27.9% | - (protective)* | -7.799 |
| 3 | ever_minister | 5.5% | - (protective) | -0.465 |
| 4 | immigration_proportion | 5.1% | + (risk) | 0.903 ⭐ |
| 5 | retirement_proximity | 5.1% | - (protective) | N/A |
| 6 | total_minister_years | 4.9% | - (protective) | N/A |
| 7 | reform_alignment_raw | 4.9% | + (risk) | 0.279 NS |
| 8 | highest_ministerial_rank | 4.6% | - (protective) | N/A |
| 9 | age | 2.8% | + (risk) | N/A |
| 10 | radicalization_slope | 2.4% | + (risk) | -0.361 NS |

*Note: Backbench years has negative coefficient due to multicollinearity with years_as_mp

**Key Insight:** Career trajectory (years as MP + backbench years) accounts for **57.5%** of model weight, dominating over speech features (~16% combined).

### Feature Set Comparison

| Feature Set | Features | AUC | Interpretation |
|-------------|----------|-----|----------------|
| Speech Only | 6 | 0.680 ± 0.158 | Poor alone |
| Demographics Only | 10 | 0.896 ± 0.167 | Strong baseline |
| **All Features** | **16** | **0.978 ± 0.016** | **Best** (+8.2pp) |

**Conclusion:** Demographics provide strong baseline, but speech features add crucial 8.2pp improvement.

### Top Predicted Defectors (Validated)

**Perfect rankings (within top 20):**
1. Nadine Dorries (99.98%) ✅ DEFECTED
2. David Jones (99.95%) ✅ DEFECTED
3. Marco Longhi (99.87%) ✅ DEFECTED
4. Jonathan Gullis (99.77%) ✅ DEFECTED
5. Andrea Jenkyns (99.77%) ✅ DEFECTED
6. Ben Bradley (99.72%) ✅ DEFECTED
7. Lia Nici (99.42%) ✅ DEFECTED
9. Maria Caulfield (97.74%) ✅ DEFECTED
10. Nadhim Zahawi (97.50%) ✅ DEFECTED
11. Sarah Atherton (97.18%) ✅ DEFECTED
12. Adam Holloway (96.57%) ✅ DEFECTED
13. Jake Berry (96.24%) ✅ DEFECTED
14. Lucy Allan (96.18%) ✅ DEFECTED
15. Chris Green (96.17%) ✅ DEFECTED
16. Anne Marie Morris (96.09%) ✅ DEFECTED
17. Robert Jenrick (93.75%) ✅ DEFECTED

**16 of 19 defectors in top 20 (84.2% recall)**

**High-risk non-defectors (worth monitoring):**
- #8: Chris Murray (99.04%) - Highest risk MP who hasn't defected yet
- #18: Katie Lam (86.65%)
- #19: David Smith (81.54%)
- #20: Margaret Mullane (80.64%)

**Lower-ranked defectors (outside top 20):**
- Sarah Pochin: Rank #242, Prob 61.9%
- Lee Anderson: Rank #270, Prob 52.7%
- Danny Kruger: Rank #343, Prob 38.9%

**Analysis:** These 3 may have defected for reasons not captured (personal relationships, specific incidents, constituency factors).

### Interaction Effects

**Tested 5 interaction terms:**
1. Career stagnation × retirement proximity (coef: -0.125)
2. Reform alignment × ministerial status (coef: +0.030)
3. Reform alignment × retirement (coef: -0.794)
4. Backbench years × Reform alignment (coef: +1.828) ⭐ **Strongest**
5. Immigration focus × ministerial status (coef: +0.350)

**Result:** Adding interactions to model:
- Base model (6 features): AUC = 0.904
- With interactions (11 features): AUC = 0.886
- **Decreased performance** (-0.018)

**Conclusion:** Interaction terms don't improve cross-validated performance, likely due to small sample size (19 defectors). Main effects sufficient.

---

## Validation Against Known Events

### Robert Jenrick (Confirmed Defector - January 15, 2026)

**Model prediction:** 93.75% probability (Rank #17)
- Reform alignment (normalized): 0.023 (top-ranked defector on this metric)
- Immigration proportion: 13.7%
- Total speeches: 918
- Ministerial role: Yes (Minister of State)
- Age: 44

**Interpretation:** Model correctly identified high risk. Normalized speech metric was crucial - his volume-adjusted score was #1 among all defectors.

### Suella Braverman (Has NOT Defected)

**Model prediction:** Low-moderate risk (not in top 50)
- Reform alignment (raw): 0.268 (high)
- Reform alignment (normalized): 0.020 (moderate)
- Immigration proportion: 12.3%
- Total speeches: 808
- Ministerial role: Yes (Home Secretary - Cabinet level)

**Interpretation:** Despite hardline rhetoric, ministerial protection effect prevents defection. Model correctly predicts low risk.

### Rishi Sunak (Has NOT Defected)

**Model prediction:** Very low risk (not in top 200)
- Reform alignment (raw): 0.174 (appeared high in v0.2!)
- Reform alignment (normalized): 0.003 (very low)
- Immigration proportion: 2.8%
- Total speeches: 2,319 (highest volume)
- Ministerial role: Yes (Prime Minister - Cabinet level)

**Interpretation:** Volume normalization correctly identifies that high keyword count was due to PM role, not ideology. Model validates correctly.

---

## Limitations

### Sample Size
- **Only 19 defectors** - Limits statistical power
- Class imbalance (3.5% positive class) - Addressed with balanced weights
- Small sample prevents complex interaction modeling

### Age Data Coverage
- Only 83.8% of MPs have ages (454/542)
- Only 21% of defectors have ages (4/19)
- Retirement proximity features under-leveraged
- Missing ages likely non-random (historical MPs harder to find)

### Temporal Validity
- Model trained on 2024-2026 defections during specific political climate
- Post-2024 election period with Reform UK surge
- May need retraining if political landscape shifts dramatically
- Assumes future defectors resemble past defectors

### Feature Engineering Assumptions
1. **Retirement age** - 55-70 range may not hold for all MPs
2. **Backbench years** - Approximate (estimated tenure - ministerial years)
3. **Immigration keywords** - May miss some rhetoric variants
4. **Ministerial rank hierarchy** - Simplified classification

### Data Quality
1. **Hansard attribution** - Some speeches may be mis-attributed
2. **IFG database completeness** - May miss some junior roles
3. **Name matching** - Title variations cause some mismatches
4. **4 missing defectors** - Not in Hansard 2020-2025 (Mark Reckless, Aidan Burley, Alan Amos, Ross Thomson)

### Causality vs Correlation
- Model identifies **risk factors**, not causal mechanisms
- Cannot predict defections caused by:
  - Personal scandals or controversies
  - Private discussions with Reform UK leadership
  - Sudden policy disagreements on specific bills
  - Family or health considerations
  - Constituency-specific events

---

## How to Run

### Full Training Pipeline

```bash
# 1. Identify Conservative speakers in Hansard
python identify_conservative_speakers.py
# Output: conservative_speakers_in_hansard.csv (542 MPs)

# 2. Run TF-IDF speech analysis (5-10 minutes)
python enhanced_speech_tfidf.py
# Output: enhanced_speech_tfidf.csv

# 3. Normalize speech metrics
python normalize_speech_metrics.py
# Output: enhanced_speech_tfidf_normalized.csv

# 4. Fetch ministerial data from IFG
python fetch_ifg_ministerial_data.py
# Output: mp_career_features.csv

# 5. Fetch ages from Wikidata (10-15 minutes)
python fetch_mp_ages_from_wikidata.py
# Output: mp_ages_from_wikidata.csv
# Updates: mp_career_features.csv

# 6. Training analysis subfolder
cd training_past_defections

# 7. Prepare training dataset
python prepare_training_data.py
# Output: training_data_2024.csv (542 MPs, 19 defectors)

# 8. Exploratory data analysis
python training_eda.py
# Output: training_eda_univariate.csv, training_eda_correlation.csv

# 9. Optimize model weights
python optimize_model_weights.py
# Output: model_optimization_results.csv, model_predictions.csv

# 10. Analyze interaction effects
python analyze_interaction_effects.py
# Output: interaction_effects_analysis.csv
```

### Required Data Files

**Source data (must exist):**
- `../../source_data/hansard/all_speeches.csv` (199,701 speeches)
- `../../source_data/defection_tracker/defections_2024.csv` (23 defections)
- `../../source_data/mp_demographics.csv` (650 MPs)

**External APIs:**
- Wikidata Query Service (for ages)
- IFG GitHub repository (for ministerial data)

### Processing Time

| Script | Time | Notes |
|--------|------|-------|
| identify_conservative_speakers | 2 min | Fast |
| enhanced_speech_tfidf | 5-10 min | TF-IDF on 200k speeches |
| normalize_speech_metrics | 5 min | Sentence counting |
| fetch_ifg_ministerial_data | 1 min | Download CSVs |
| fetch_mp_ages_from_wikidata | 10-15 min | SPARQL queries (rate limited) |
| prepare_training_data | <1 min | Merges |
| training_eda | <1 min | Statistics |
| optimize_model_weights | <1 min | Logistic regression |
| analyze_interaction_effects | <1 min | Additional stats |
| **Total** | **~25-35 minutes** | One-time pipeline |

---

## Recommendations for Production Use

### 1. Apply to Current Conservative MPs

**Process:**
1. Run speech analysis on current 121 Conservative MPs (post-2024 election)
2. Fetch updated ministerial data (as of January 2026)
3. Apply trained model coefficients (standardize features with same scaler)
4. Generate predicted probabilities
5. Rank by risk score

**Output:** Top 20-50 MPs for ongoing monitoring

### 2. Monitoring Dashboard

**Components:**
- Top 20 highest-risk MPs with feature breakdowns
- Weekly trend monitoring (radicalization slope)
- Alert system when MP enters top 20
- Comparison to previous week (risk increasing/decreasing)

### 3. Update Cadence

**Monthly updates:**
- Re-run speech analysis (new Hansard speeches)
- Update radicalization slopes (temporal trends)
- Refresh ministerial career data

**Quarterly updates:**
- Re-fetch ages from Wikidata (new MPs)
- Review model performance against any new defections
- Retrain if >5 new defections observed

### 4. Red Flags for Manual Review

An MP shows elevated risk if:
- ✅ Top 20 predicted probability
- ✅ Positive radicalization slope (becoming more extreme)
- ✅ Immigration proportion > 10%
- ✅ Never held ministerial role OR career stagnation >5 years
- ✅ Recent increase in Reform-aligned language

### 5. Qualitative Context

Model cannot capture:
- Private conversations with Reform UK leadership
- Personal scandals or controversies
- Constituency-specific Reform UK polling surges
- Family or health considerations
- Recent specific policy disagreements

**Recommendation:** Use model as first-pass filter (top 20-50), then apply qualitative analysis to those MPs.

---

## Future Enhancements

### High Priority (for v1.1)

1. **Improve age data coverage**
   - Manual lookup for 88 MPs missing from Wikidata (16.2%)
   - Better name matching (handle spelling variants, titles)
   - Target defectors specifically (currently only 4/19 have ages)

2. **Add voting rebellion data**
   - Voting records on key immigration bills
   - Rebellion frequency and intensity
   - Specific Rwanda Bill, ECHR votes

3. **Real-time monitoring system**
   - Automated daily/weekly updates
   - Alert system for MPs entering top 20
   - Trend visualization dashboard

### Medium Priority (for v1.2)

4. **Social media analysis**
   - Twitter/X sentiment and rhetoric
   - Engagement with Reform UK figures
   - Constituency social media activity

5. **Constituency-level features**
   - Reform UK polling in constituency
   - Margin of victory in 2024 election
   - Demographic makeup (age, ethnicity, income)

6. **Enhanced speech analysis**
   - Add more Reform UK rhetoric templates
   - Fine-tune on political speeches
   - Topic modeling (identify specific immigration sub-topics)

### Low Priority (research directions)

7. **Survival analysis approach**
   - Time-to-defection modeling
   - Time-varying covariates
   - Early warning system (acceleration in risk)

8. **Transformer-based embeddings**
   - Replace TF-IDF with BERT/RoBERTa
   - Deeper semantic understanding
   - Better context handling (negation, sarcasm)

9. **Ensemble methods**
   - Combine logistic regression with Random Forest, XGBoost
   - Stacking for improved prediction
   - Uncertainty quantification

---

## Comparison to Previous Models

| Aspect | v0.1 Demogs | v0.2 Keywords | v0.3 TF-IDF | **v1.0 Training** |
|--------|------------|--------------|------------|-------------------|
| **Ground truth** | ❌ None | ❌ None | ❌ None | ✅ **19 defectors** |
| **Validation** | ❌ | ❌ | ❌ | ✅ **97.8% AUC** |
| **Volume bias** | N/A | ❌ Severe | ⚠️ Partial | ✅ **Fully solved** |
| **Dataset size** | 650 MPs | 117 current | 117 current | ✅ **542 historical** |
| **Speech analysis** | ❌ | Keywords | TF-IDF | ✅ **TF-IDF + normalize** |
| **Feature weights** | Manual | Manual | Manual | ✅ **ML-optimized** |
| **Temporal trends** | ❌ | ❌ | ✅ Radicalization | ✅ Enhanced |
| **Ministerial data** | Basic | Basic | Basic | ✅ **IFG (172 ministers)** |
| **Age data** | ❌ | ❌ | ❌ | ✅ **Wikidata (454 MPs)** |
| **Interaction effects** | ❌ | ❌ | ❌ | ✅ Analyzed |
| **Recall@20** | Unknown | Unknown | Unknown | ✅ **84.2%** |

**Conclusion:** v1.0 is a step-change improvement, enabling actual validation and deployment.

---

## Technical Specifications

**Dependencies:**
```bash
pip install pandas numpy scikit-learn scipy requests
```

**Python version:** 3.8+

**Compute requirements:**
- RAM: 2-3 GB
- CPU: Multi-core recommended (for TF-IDF)
- GPU: Not required
- Storage: ~500 MB for data files

**Outputs:**
- CSV files with predictions
- Text reports with rankings
- Comprehensive Markdown documentation

---

## Citation

If using this model for research or analysis:

```
Conservative MP Defection Risk Model v1.0 (2026)
Training Dataset: Best for Britain Defection Tracker (23 defections, Jan 2024 - Jan 2026)
Method: Logistic Regression with TF-IDF Speech Analysis
Performance: 97.8% Cross-Validated AUC
```

---

**Model Version:** v1.0
**Last Updated:** January 16, 2026
**Status:** Production-ready ✅
**Validation:** 97.8% AUC on 19 historical defections
**Recommended Use:** Monitoring top 20-50 Conservative MPs for defection risk

---

**See Also:**
- [TRAINING_ANALYSIS_REPORT.md](TRAINING_ANALYSIS_REPORT.md) - Full 28-page technical analysis
- [../basic_demogs_model/MODEL_SPEC.md](../basic_demogs_model/MODEL_SPEC.md) - v0.1 baseline
- [../basic_demogs_keywords_model/MODEL_SPEC.md](../basic_demogs_keywords_model/MODEL_SPEC.md) - v0.2 keywords
- [../basic_demogs_vectorisation_model/MODEL_SPEC.md](../basic_demogs_vectorisation_model/MODEL_SPEC.md) - v0.3 TF-IDF
