# Model Specification: Basic Demographics + TF-IDF Vectorization Model

**Model Version:** v0.3 (Semantic Similarity Analysis)
**Date Created:** January 15, 2026
**Status:** Superseded by training-based model (v1.0)

---

## Theory of Change

This model addresses the **volume bias problem** discovered in v0.2 by replacing keyword counting with **semantic similarity analysis** using TF-IDF vectorization.

### Core Hypothesis
MPs are more likely to defect if they:
1. **Sound like Reform UK** - Speeches semantically similar to Reform rhetoric (not just keyword matches)
2. **Are radicalizing** - Becoming MORE extreme over time (temporal trajectory)
3. **Use hardline language** - Employ emotionally charged anti-immigration rhetoric
4. **Face constituency pressure** - Represent seats with high Reform UK vote share
5. **Have career frustration** - Parliamentary rebellion + career stagnation

### Key Innovation Over v0.2
**From keyword counting → semantic similarity:**
- TF-IDF vectorization of full speech content
- Cosine similarity to Reform UK rhetoric templates
- Context-aware (distinguishes pro vs anti-immigration stances)
- Temporal radicalization tracking

### Critical Problem Solved: Volume Bias

**v0.2 Problem:**
- Rishi Sunak ranked high due to 2,319 speeches as PM
- Keyword count = 847 mentions of immigration
- But: He discussed immigration **in opposition** to Reform, not support

**v0.3 Solution:**
- TF-IDF measures **semantic similarity**, not raw counts
- Rishi Sunak speeches show **low cosine similarity** to Reform rhetoric
- Correctly identifies he's not ideologically aligned despite high volume

---

## Variables Used

### Speech Features (40% weight) - 6 variables

1. **reform_alignment_score** - TF-IDF cosine similarity to Reform UK rhetoric
   - Method: TF-IDF vectorization + cosine similarity
   - Templates: 15 Reform UK immigration talking points
   - Range: 0-1 (0=no alignment, 1=perfect alignment)
   - Hypothesis: Higher alignment → ideological similarity

2. **radicalization_slope** - Temporal trend in Reform alignment
   - Method: Linear regression on quarterly alignment scores
   - Range: -0.5 to +0.5 per quarter
   - Hypothesis: Positive slope → becoming more extreme → defection risk

3. **extremism_percentile** - Rank among all Conservative MPs
   - Method: Percentile of reform_alignment_score
   - Range: 0-100%
   - Hypothesis: Top 10% (90-100th percentile) = highest risk

4. **hardline_compassionate_ratio** - Balance of rhetoric type
   - Method: Semantic similarity to hardline vs compassionate templates
   - Range: -1 (very compassionate) to +1 (very hardline)
   - Hypothesis: Hardline language → Reform alignment

5. **speech_volume** - Total speeches (normalization factor)
   - Source: Hansard database
   - Range: 0-2000+
   - Use: Context for interpreting other metrics

6. **enhanced_speech_score** - Composite metric
   - Formula: `(alignment × 0.35) + (radicalization × 0.25) + (hardline_ratio × 0.20) + (extremism_pct × 0.15) + (volume × 0.05)`
   - Range: 0-1
   - Purpose: Single speech-based risk indicator

### Voting Features (30% weight) - Same as v0.2

7. **overall_rebellion_rate** - % votes against party whip
8. **rwanda_bill_rebellion** - Binary: voted against Rwanda Bill
9. **recent_rebellion_trend** - Change in rebellion rate

### Constituency Features (20% weight) - Reduced from 25%

10. **reform_vote_share_2024** - Reform UK % in 2024 election
11. **majority_percentage** - MP's victory margin
12. **seat_vulnerability_score** - reform_share / majority

### Demographic Features (10% weight) - Same as v0.2

13. **years_in_parliament** - Parliamentary tenure
14. **ministerial_experience** - Binary: ever held role
15. **backbench_duration** - Years since last ministerial role

---

## Techniques Used

### TF-IDF Vectorization

**What is TF-IDF?**
- **TF** (Term Frequency): How often a word appears in a document
- **IDF** (Inverse Document Frequency): How unique a word is across all documents
- **TF-IDF = TF × IDF**: High score for words that are frequent in one document but rare overall

**Why TF-IDF?**
- Captures importance of words, not just counts
- Down-weights common words like "the", "and"
- Up-weights distinctive political vocabulary
- Fast and interpretable (compared to neural networks)

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    stop_words='english'
)

# Fit on all speeches
tfidf_matrix = vectorizer.fit_transform(all_speeches)

# Transform Reform UK templates
reform_vectors = vectorizer.transform(REFORM_TEMPLATES)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
alignment_scores = cosine_similarity(mp_vectors, reform_vectors).mean(axis=1)
```

### Reform UK Rhetoric Templates

15 exemplar statements capturing Reform immigration positions:
1. "Stop the boats and take back control of our borders"
2. "Mass immigration is destroying British sovereignty and culture"
3. "The Rwanda scheme is necessary to deter illegal channel crossings"
4. "We must leave the ECHR to protect our borders"
5. "Economic migrants are abusing the asylum system"
6. "Uncontrolled immigration threatens our economy and security"
7. "Britain is full - we cannot take unlimited migrants"
8. "Illegal immigration is an invasion that must be stopped"
9. "The immigration crisis is out of control and broken"
10. "We need an Australian-style points system for real control"
... (5 more)

### Temporal Radicalization Analysis

**Method:** Linear regression on time-series alignment scores

```python
import numpy as np
from scipy.stats import linregress

# Group speeches by quarter
quarters = ['2023-Q1', '2023-Q2', '2023-Q3', ...]
quarterly_scores = [0.3, 0.35, 0.42, 0.48, ...]  # Example MP

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(
    range(len(quarters)),
    quarterly_scores
)

# slope > 0 → radicalizing
# slope < 0 → moderating
```

**Interpretation:**
- **Slope = +0.08/quarter:** MP is rapidly radicalizing (2 years → +0.64 alignment)
- **Slope = 0:** Stable rhetoric
- **Slope = -0.05/quarter:** Moderating over time

### Cosine Similarity

Measures angle between vectors (0-1):
- **1.0** = Identical meaning (vectors point same direction)
- **0.5** = Somewhat similar (45° angle)
- **0.0** = Completely different (90° angle)

**Example:**
- Speech 1: "We must stop illegal immigration now"
- Speech 2: "Halting unlawful migration is essential"
- Cosine similarity: ~0.85 (very similar meaning, different words)

---

## Folder Structure

```
basic_demogs_vectorisation_model/
│
├── MODEL_SPEC.md                            (this file)
│
├── enhanced_speech_vectorization.py         TF-IDF speech analysis
│   └── Contains:
│       - TF-IDF vectorizer setup
│       - Reform UK template encoding
│       - Cosine similarity calculation
│       - Temporal radicalization analysis
│       - Extremism percentile ranking
│       - Output: enhanced_speech_vectorization.csv
│
├── run_vectorized_analysis.py               Main risk model
│   └── Contains:
│       - Load speech vectorization results
│       - Combine with voting/constituency/demographics
│       - Weighted composite risk scoring (40/30/20/10)
│       - Rankings and output generation
│       - Output: mp_vectorized_risk_scores.csv
│
├── README_VECTORIZATION.md                  User guide
│   └── Quick start instructions
│   └── Explanation of techniques
│   └── Comparison to keyword model
│   └── Troubleshooting
│
├── mp_vectorized_risk_scores.csv            Final model output (117 current MPs)
│   └── Columns:
│       - name, constituency
│       - defection_risk_score (0-100 weighted composite)
│       - reform_alignment_score (0-1 cosine similarity)
│       - radicalization_slope (per quarter)
│       - extremism_percentile (0-100)
│       - hardline_compassionate_ratio (-1 to +1)
│       - enhanced_speech_score (0-1 composite)
│       - rebellion_rate, rwanda_rebellion
│       - reform_vote_share_2024, majority_pct
│       - years_in_parliament, ministerial_experience
│
└── vectorized_defection_risk_report.txt     Summary report
    └── Top 25 highest-risk MPs
    └── Comparison to keyword model
    └── Validation against known alignments
    └── Temporal radicalization patterns
```

---

## Results Summary

### Top Risk Factors (by category weight)
1. **Speech Analysis (40%)** - TF-IDF semantic similarity
   - Reform alignment: Most important single variable
   - Radicalization slope: Strong predictor of future behavior
2. **Voting Rebellion (30%)** - Anti-party voting pattern
3. **Constituency Risk (20%)** - Reform vote share pressure
4. **Demographics (10%)** - Career stagnation

### Volume Bias Correction - Examples

| MP | v0.2 Keyword Score | v0.3 TF-IDF Score | Explanation |
|----|-------------------|-------------------|-------------|
| **Rishi Sunak** | 0.82 (HIGH) | 0.15 (LOW) | ✅ Fixed: PM talks a lot but NOT aligned |
| **Suella Braverman** | 0.48 (MED) | 0.75 (HIGH) | ✅ Correct: Genuinely hardline |
| **Robert Jenrick** | 0.61 (HIGH) | 0.68 (HIGH) | ✅ Stable: Truly Reform-aligned |
| **Laura Trott** | 0.23 (LOW) | 0.57 (HIGH) | ⚠️ New: Dense Reform rhetoric in few speeches |

### Radicalization Detection

**MPs with positive slopes (becoming more extreme):**
- Suella Braverman: +0.08 per quarter (rapid radicalization)
- Robert Jenrick: +0.06 per quarter (steady increase)
- Andrea Jenkyns: +0.12 per quarter (very rapid)

**MPs with stable rhetoric:**
- Most moderate Tories: slope ≈ 0
- Consistent messaging over time

### Top Predicted Defection Risks
(See vectorized_defection_risk_report.txt for full rankings)

**Notable changes from v0.2:**
- Frontbenchers who talk a lot → DOWN in rankings (volume bias corrected)
- Backbenchers with concentrated Reform rhetoric → UP in rankings
- Radicalization trajectory → New predictive signal

---

## Limitations

### TF-IDF Weaknesses
1. **Bag-of-words model** - Ignores word order
   - "Immigration is not a crisis" vs "Immigration is a crisis"
   - TF-IDF may struggle with negation
2. **Fixed vocabulary** - Can't handle new phrases
   - If Reform UK adopts new slogan, model won't detect it immediately
3. **No semantic depth** - Captures lexical similarity, not deep meaning
   - Better than keywords but not as good as transformer models

### Methodological Issues
1. **Still no ground truth** - Cannot validate accuracy
2. **Reform templates may be incomplete** - 15 statements might not capture full Reform ideology
3. **Temporal analysis requires data** - MPs with <10 speeches can't be analyzed for trends
4. **Arbitrary weight scheme** - 40/30/20/10 not empirically derived

### Data Quality
1. **Hansard attribution** - Some speeches mis-attributed in source data
2. **Speech selection bias** - Not all speeches captured in dataset
3. **Context missing** - Don't know if MP is speaking as minister vs backbencher

---

## Key Insights Leading to v1.0 (Training Model)

### Critical Gap: No Validation

**Problem:** All v0.1-v0.3 models are **exploratory/predictive only**. We have:
- ❌ No ground truth defections to train on
- ❌ No way to validate predictions
- ❌ No way to optimize weights empirically
- ❌ No way to measure accuracy

**Solution Required:** Find actual defections to use as training data

### Breakthrough: Best for Britain Defection Tracker

**Discovery (January 2026):** Best for Britain maintains list of 23 Conservative MPs who defected to Reform UK since January 1, 2024:
- Lee Anderson (March 2024)
- Robert Jenrick (January 2026)
- Nadine Dorries, Andrea Jenkyns, Danny Kruger, etc.

**This enables:**
- ✅ Supervised machine learning with labeled data
- ✅ Cross-validation and accuracy metrics
- ✅ Empirically optimized feature weights
- ✅ Model comparison (speech vs demographics)
- ✅ Interaction effect analysis

### Additional Improvements for v1.0

1. **Expand dataset** - Analyze 542 historical Conservative MPs (not just current 117)
2. **Volume normalization formula** - More sophisticated than TF-IDF alone
3. **Ministerial career data** - Fetch from IFG database (172 ministers)
4. **Age data** - Query Wikidata for retirement proximity
5. **Logistic regression with regularization** - Replace manual weighting

This led to **v1.0: training_tf_idf_model** with 97.8% cross-validated AUC

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn scipy pandas numpy

# Step 1: Run TF-IDF speech analysis (5-10 minutes)
cd marketing/tory_defection/analysis/basic_demogs_vectorisation_model
python enhanced_speech_vectorization.py

# Step 2: Generate risk scores
python run_vectorized_analysis.py

# Output files created:
# - mp_vectorized_risk_scores.csv (final scores)
# - vectorized_defection_risk_report.txt (summary)
```

**Requirements:**
- Hansard speeches CSV (199,701 speeches)
- Election results (2024)
- Voting rebellion data
- ~5-10 minutes processing time

---

## Comparison Across Models

| Aspect | v0.1 Demogs | v0.2 Keywords | v0.3 TF-IDF | v1.0 Training |
|--------|------------|--------------|------------|--------------|
| Speech analysis | ❌ | ✅ Keywords | ✅ TF-IDF | ✅ TF-IDF + normalize |
| Volume bias | N/A | ❌ Severe | ⚠️ Partially fixed | ✅ Fully fixed |
| Temporal trends | ❌ | ❌ | ✅ Radicalization | ✅ Enhanced |
| Ground truth | ❌ | ❌ | ❌ | ✅ 19 defectors |
| Validation | ❌ | ❌ | ❌ | ✅ 97.8% AUC |
| Feature weights | Manual | Manual | Manual | ✅ ML-optimized |
| Dataset size | 650 MPs | 117 current | 117 current | ✅ 542 historical |

---

## Performance Notes

**Processing Time:**
- Load speeches: 30 seconds
- TF-IDF vectorization: 2-3 minutes
- Similarity calculations: 1-2 minutes
- Temporal analysis: 2-3 minutes
- **Total: ~5-10 minutes**

**Memory Usage:**
- Speech data: ~500 MB
- TF-IDF matrix: ~200 MB
- **Peak: ~1 GB RAM**

**Accuracy:**
- Cannot be measured (no validation data)
- Face validity: Top-ranked MPs match known Reform-aligned figures

---

**Previous Model:** See [../basic_demogs_keywords_model/MODEL_SPEC.md](../basic_demogs_keywords_model/MODEL_SPEC.md)

**Next Model:** See [../training_tf_idf_model/MODEL_SPEC.md](../training_tf_idf_model/MODEL_SPEC.md) ← **Current best model**
