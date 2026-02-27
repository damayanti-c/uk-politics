# Model Specification: Basic Demographics + Keywords Model

**Model Version:** v0.2 (Enhanced with Speech Keywords)
**Date Created:** January 15, 2026
**Status:** Superseded by vectorization and training models

---

## Theory of Change

This enhanced model builds on the basic demographics baseline by adding **keyword-based speech analysis** to detect Reform UK ideological alignment.

### Core Hypothesis
MPs are more likely to defect if they have:
1. **Ideological alignment** - Frequently use Reform UK rhetoric in speeches (keywords)
2. **Constituency pressure** - Represent seats with high Reform UK vote share
3. **Voting rebellion** - Pattern of voting against party whip, especially on immigration bills
4. **Career stagnation** - Long parliamentary tenure without ministerial advancement
5. **Immigration focus** - Disproportionate focus on immigration in speeches

### Key Innovation Over v0.1
Addition of **speech content analysis** using keyword counting:
- Count mentions of Reform UK talking points (immigration, sovereignty, Rwanda, etc.)
- Detect focus on immigration-related topics
- Analyze speech frequency and volume

### Weighted Feature Approach
Features are explicitly weighted by perceived importance:
- **Speech Analysis (35%)** - Reform keywords, immigration mentions
- **Voting Rebellion (30%)** - Rebellion rate, Rwanda Bill votes
- **Constituency Risk (25%)** - Reform vote share, majority size
- **Demographics (10%)** - Gender, tenure, age

---

## Variables Used

### Speech Features (35% weight) - 4 variables

1. **reform_keyword_count** - Total mentions of Reform UK keywords
   - Keywords: "immigration", "borders", "sovereignty", "Rwanda", "small boats", "Brexit", "EU", "uncontrolled migration"
   - Source: Hansard speeches (5 years)
   - Range: 0-500+
   - Hypothesis: More mentions → higher ideological alignment

2. **immigration_focus_score** - % of speeches mentioning immigration
   - Calculation: (speeches with immigration keywords) / (total speeches)
   - Source: Hansard speeches
   - Range: 0-100%
   - Hypothesis: High focus → alignment with Reform priorities

3. **total_speeches** - Speech volume indicator
   - Source: Hansard database
   - Range: 0-2000+
   - Hypothesis: More speeches → more opportunities to signal alignment

4. **avg_keywords_per_speech** - Keyword density
   - Calculation: reform_keyword_count / total_speeches
   - Range: 0-20+
   - Hypothesis: High density → consistent Reform messaging

### Voting Features (30% weight) - 3 variables

5. **overall_rebellion_rate** - % of votes against party whip
   - Source: Public Whip database
   - Range: 0-50%
   - Hypothesis: Higher rebellion → less party loyalty

6. **rwanda_bill_rebellion** - Binary: voted against Rwanda Bill
   - Source: House of Commons division data
   - Values: 0 (loyalist) or 1 (rebel)
   - Hypothesis: Immigration bill rebellion → policy alignment with Reform

7. **recent_rebellion_trend** - Increasing/decreasing rebellion pattern
   - Calculation: Rebellion rate last 12 months vs previous 24 months
   - Range: -50% to +50%
   - Hypothesis: Increasing rebellion → growing disillusionment

### Constituency Features (25% weight) - 3 variables

8. **reform_vote_share_2024** - Reform UK % in 2024 election
   - Source: House of Commons Library election results
   - Range: 0-50%
   - Hypothesis: High Reform support → constituency pressure to defect

9. **majority_percentage** - MP's victory margin
   - Source: Election results
   - Range: 0-50%
   - Hypothesis: Smaller majority → more vulnerable to Reform challenge

10. **seat_vulnerability_score** - Combined constituency risk
    - Calculation: (reform_vote_share) / (majority_percentage)
    - Range: 0-10+
    - Hypothesis: High ratio → vulnerable to Reform takeover

### Demographic Features (10% weight) - 3 variables

11. **years_in_parliament** - Parliamentary tenure
    - Source: Parliament API
    - Range: 0-50+

12. **ministerial_experience** - Binary: ever held ministerial role
    - Source: Government posts data
    - Values: 0 (never) or 1 (former minister)

13. **backbench_duration** - Years since last ministerial role
    - Source: Calculated from government posts
    - Range: 0-30+

---

## Techniques Used

### Speech Analysis - Keyword Counting
```python
REFORM_KEYWORDS = [
    'immigration', 'immigra', 'migrant', 'asylum',
    'border', 'borders', 'sovereignty', 'Rwanda',
    'small boat', 'channel crossing', 'illegal migration',
    'uncontrolled', 'Brexit', 'EU regulation'
]

# Count keyword occurrences in each MP's speeches
for speech in mp_speeches:
    keyword_count += sum(keyword in speech.lower() for keyword in REFORM_KEYWORDS)
```

### Feature Engineering
- Keyword density normalization (keywords per speech)
- Constituency vulnerability composite score
- Temporal rebellion trends (recent vs historical)
- Weighted composite risk score

### Data Sources
- **Hansard speeches** - 5 years of parliamentary debates (2020-2025)
- **House of Commons Library** - 2024 election results
- **Public Whip** - Voting records and rebellion rates
- **Rwanda Bill divisions** - Specific immigration vote
- **Parliament API** - MP demographics and career data

### Modeling Approach
- **Weighted linear combination** - Each feature category has fixed weight
- **Manual calibration** - Weights assigned based on domain expertise
- **No machine learning** - Deterministic scoring, not trained on data
- **Percentile ranking** - Convert scores to 0-100 risk scale

### Evaluation
- **No ground truth validation** - Still exploratory/predictive
- **Face validity** - Results checked against known defectors
- **Sensitivity analysis** - Test impact of weight changes

---

## Folder Structure

```
basic_demogs_keywords_model/
│
├── MODEL_SPEC.md                              (this file)
│
├── run_enhanced_analysis.py                   Main analysis script
│   └── Contains:
│       - Data loading (elections, speeches, voting)
│       - Keyword counting in speeches
│       - Feature engineering (weighted scores)
│       - Composite risk score calculation
│       - Results output and ranking
│
├── speech_analysis.csv                        Hansard speech keyword analysis
│   └── Columns:
│       - speaker_name: MP name
│       - total_speeches: Count of speeches
│       - reform_keyword_count: Total keyword mentions
│       - immigration_focus_score: % speeches on immigration
│       - avg_keywords_per_speech: Keyword density
│
├── mp_defection_risk_scores_with_speech.csv   Intermediate results
│   └── All features before weighting
│
├── mp_enhanced_risk_scores.csv                Final model output (121 current Conservative MPs)
│   └── Columns:
│       - name: MP name
│       - constituency: Seat name
│       - defection_risk_score: 0-100 weighted composite score
│       - speech_score: Speech category score (0-35)
│       - voting_score: Voting rebellion score (0-30)
│       - constituency_score: Constituency risk score (0-25)
│       - demographic_score: Demographics score (0-10)
│       - reform_keyword_count
│       - immigration_focus_score
│       - reform_vote_share_2024
│       - rebellion_rate
│
└── enhanced_defection_risk_report.txt         Summary report
    └── Top 30 highest-risk MPs
    └── Feature weight breakdown
    └── Category-specific rankings
    └── Validation against known defectors
```

---

## Results Summary

### Top Risk Factors (by category weight)
1. **Speech Analysis (35%)** - Keyword-based Reform alignment
2. **Voting Rebellion (30%)** - Anti-party voting pattern
3. **Constituency Risk (25%)** - Reform vote share pressure
4. **Demographics (10%)** - Career stagnation

### Model Performance
- **No validation data** - Cannot compute accuracy
- **Face validity checks:**
  - Robert Jenrick ranked high (later confirmed defector)
  - Known Reform-aligned MPs appear in top 20

### Top Predicted Defection Risks
(See enhanced_defection_risk_report.txt for full list)

### Sensitivity Analysis
Tested alternative weight schemes:
- Equal weights (25% each category): Similar top 10
- Speech-heavy (50% speech, 20% others): Emphasizes ideological alignment
- Voting-heavy (50% voting, 20% others): Emphasizes anti-party behavior

Results relatively stable across weighting schemes for top-ranked MPs.

---

## Limitations

### Keyword Counting Weaknesses
1. **Context-blind** - Doesn't distinguish support vs criticism
   - "We must stop uncontrolled immigration" = counted
   - "Claims of uncontrolled immigration are false" = also counted
2. **Volume bias** - Ministers who speak frequently score higher
   - Rishi Sunak ranked high due to PM speech volume
   - Need to normalize by speech frequency
3. **Crude semantic analysis** - Misses nuanced rhetoric
4. **Binary keyword matching** - No weighted importance of different keywords

### Methodological Issues
1. **Arbitrary weights** - 35/30/25/10 split not empirically derived
2. **No interaction effects** - Assumes linear additivity
3. **Still no ground truth** - Cannot validate predictions
4. **Overfits to current political moment** - Rwanda Bill may be temporary issue

### Data Quality Issues
1. **Hansard speech attribution** - Some speeches mis-attributed
2. **Rebellion data lag** - Public Whip not real-time
3. **Keyword list incomplete** - May miss Reform rhetoric variants

---

## Key Insights Leading to Next Iteration

### Critical Flaw Identified: Volume Bias
**Problem:** Rishi Sunak and other frontbenchers ranked very high because they give many speeches and therefore accumulate many keyword mentions, even though they're not ideologically aligned with Reform UK.

**Example:**
- Rishi Sunak: 2,319 speeches → 847 keyword mentions → High raw score
- But: As PM, he talked about immigration **in opposition** to Reform, not in support
- Keyword counting cannot distinguish this

### Solution Required: Semantic Analysis
Need to move beyond keyword counting to **semantic similarity**:
- TF-IDF vectorization of full speech content
- Cosine similarity to Reform UK rhetoric
- Volume normalization (keywords as % of content, not raw counts)

This insight led to **v0.3: basic_demogs_vectorisation_model**

---

## How to Run

```bash
# Ensure data files are present:
# - source_data/elections_2024/HoC-GE2024-results-by-constituency.csv
# - source_data/hansard/all_speeches.csv
# - source_data/voting_records/rebellion_rates.csv

# Run analysis
cd marketing/tory_defection/analysis/basic_demogs_keywords_model
python run_enhanced_analysis.py

# Output files created:
# - speech_analysis.csv (keyword counts)
# - mp_enhanced_risk_scores.csv (final scores)
# - enhanced_defection_risk_report.txt (summary)
```

---

## Comparison to v0.1 (Basic Demographics)

| Aspect | v0.1 Basic Demogs | v0.2 Keywords | Improvement |
|--------|------------------|---------------|-------------|
| Speech analysis | ❌ None | ✅ Keyword counting | Major addition |
| Voting data | ✅ Basic rebellion | ✅ Enhanced (Rwanda Bill) | Incremental |
| Constituency | ✅ Basic majority | ✅ Reform vote share | Significant |
| Demographics | ✅ Career metrics | ✅ Same | No change |
| **Validation** | ❌ None | ❌ None | No improvement |
| **Volume bias** | N/A | ❌ Severe problem | New issue introduced |

---

**Previous Model:** See [../basic_demogs_model/MODEL_SPEC.md](../basic_demogs_model/MODEL_SPEC.md)

**Next Model:** See [../basic_demogs_vectorisation_model/MODEL_SPEC.md](../basic_demogs_vectorisation_model/MODEL_SPEC.md)
