# Tory MP Defection Risk Analysis

Supervised machine learning model for predicting Conservative MP defections to Reform UK. Trained on 19 historical defections (2024-2026), combining parliamentary speech analysis (TF-IDF with volume normalization) and MP career variables.

**Model Status**: ✅ Production-Ready (v1.0)  
**Validation**: Robert Jenrick (predicted #1, 93.67%) defected January 15, 2026  
**Performance**: 97.8% cross-validated AUC, 84.2% recall@20

---

## 📁 Folder Structure

```
tory_defection/
│
├── 📊 final_results.md                    # ⭐ MAIN OUTPUT: Executive summary with top 10 
│                                          #    and bottom 10 rankings
│
├── methodologies/
│   └── methodology.md                     # Full methodology: approach, data sources, 
│                                          #    validation, limitations
│
├── analysis/
│   │
│   ├── 🔴 FINAL MODEL (training_tfidf_model_final_spec/)    # ⭐ CURRENT BEST MODEL
│   │   ├── MODEL_SPEC.md                 # Model specification & theory of change
│   │   ├── current_mp_defection_risk_scores.csv  # Final predictions (118 MPs)
│   │   ├── current_mp_defection_risk_report.txt  # Human-readable top 20 & bottom 10
│   │   │
│   │   ├── 📈 Analysis Scripts
│   │   │   ├── enhanced_speech_tfidf.py  # TF-IDF speech vectorization
│   │   │   ├── fetch_ifg_ministerial_data.py    # Extract ministerial careers
│   │   │   ├── fetch_mp_ages_from_wikidata.py   # Get MP biographical data
│   │   │   ├── identify_conservative_speakers.py # Filter speeches
│   │   │   ├── normalize_speech_metrics.py       # Volume normalization
│   │   │   └── apply_model_to_current_mps.py    # Apply model to current MPs
│   │   │
│   │   ├── 📚 Training Scripts (training_past_defections/)
│   │   │   ├── compile_defection_ground_truth.py  # Collect 19 historical defectors
│   │   │   ├── prepare_training_data.py           # Format features & labels
│   │   │   ├── optimize_model_weights.py          # Train logistic regression
│   │   │   └── analyze_interaction_effects.py     # Test feature interactions
│   │   │
│   │   ├── 📊 Outputs
│   │   │   ├── enhanced_speech_tfidf.csv          # Per-MP speech statistics
│   │   │   ├── enhanced_speech_tfidf_normalized.csv # Volume-normalized scores
│   │   │   └── mp_career_features.csv             # Career data from IFG
│   │   │
│   │   └── 📖 Documentation
│   │       ├── TRAINING_ANALYSIS_REPORT.md        # Detailed training results
│   │       ├── VECTORIZATION_COMPLETE.md          # Speech vectorization notes
│   │       └── TFIDF_VS_TRANSFORMERS.md           # Method comparison
│   │
│   ├── 🟠 PRELIMINARY MODELS (Exploratory Phase 1)
│   │   │
│   │   ├── basic_demogs_model/                    # Demographics-only baseline
│   │   │   ├── MODEL_SPEC.md
│   │   │   └── [analysis outputs]
│   │   │
│   │   ├── basic_demogs_keywords_model/           # Demographics + keyword counting
│   │   │   ├── MODEL_SPEC.md
│   │   │   └── [analysis outputs]
│   │   │
│   │   └── basic_demogs_vectorisation_model/      # Demographics + basic vectorization
│   │       ├── MODEL_SPEC.md
│   │       └── [analysis outputs]
│   │
│   ├── 🟡 LEGACY SCRIPTS (Earlier iterations)
│   │   ├── fetch_current_ministerial_data.py
│   │   ├── fetch_hansard_speeches.py
│   │   ├── fetch_ministerial_by_member_id.py
│   │   ├── fetch_mp_demographics.py
│   │   ├── fetch_rebellion_data.py
│   │   ├── parse_ministerial_careers.py
│   │   ├── prepare_training_data.py
│   │   ├── training_eda.py
│   │   └── [other exploratory scripts]
│   │
│   └── 📁 source_data/
│       ├── hansard/                      # Hansard speeches (199,701 total)
│       ├── ifg_ministers/                # IFG ministerial database (172 ministers)
│       ├── mp_demographics.csv           # MP biographical data
│       ├── elections_2024/               # 2024 election results (not in final model)
│       ├── voting/                       # Voting records (exploratory, not used)
│       └── defection_tracker/            # Ground truth defections
│
└── NEXT_STEPS.md                          # Planned improvements & next phase work
```

---

## 🎯 Quick Navigation

### 📖 For Understanding the Model
1. **Start here**: [final_results.md](final_results.md) - Executive summary with top 10 at-risk MPs
2. **Technical details**: [analysis/training_tfidf_model_final_spec/MODEL_SPEC.md](analysis/training_tfidf_model_final_spec/MODEL_SPEC.md) - Model specification
3. **Full methodology**: [methodologies/methodology.md](methodologies/methodology.md) - Detailed approach, data sources, validation

### 📊 For Results & Outputs
1. **Top/bottom rankings**: [final_results.md](final_results.md) - Executive summary (top 10 and bottom 10)
2. **Full predictions**: [analysis/training_tfidf_model_final_spec/current_mp_defection_risk_scores.csv](analysis/training_tfidf_model_final_spec/current_mp_defection_risk_scores.csv) - All 118 MPs with scores
3. **Human-readable report**: [analysis/training_tfidf_model_final_spec/current_mp_defection_risk_report.txt](analysis/training_tfidf_model_final_spec/current_mp_defection_risk_report.txt) - Top 20 MPs summary

### 🔬 For Understanding Model Development
1. **Development phases**: See [methodologies/methodology.md](methodologies/methodology.md) "Model Development Process"
   - Phase 1: Preliminary experimentation with multiple feature sets
   - Phase 2: Feature selection (speech analysis + career variables chosen)
   - Phase 3: Training on 19 historical defections
   - Phase 4: Application and validation
2. **Preliminary models**: [analysis/basic_demogs_model/](analysis/basic_demogs_model/), [basic_demogs_keywords_model/](analysis/basic_demogs_keywords_model/), [basic_demogs_vectorisation_model/](analysis/basic_demogs_vectorisation_model/)

### 💻 For Running the Model
1. See [analysis/training_tfidf_model_final_spec/](analysis/training_tfidf_model_final_spec/) folder for scripts
2. Main pipeline: `optimize_model_weights.py` (trains) → `apply_model_to_current_mps.py` (applies to current 118 MPs)

---

## 📈 Model Overview

### What It Predicts
Defection probability for each Conservative MP (0-100% risk score) based on:
- **Speech analysis**: TF-IDF similarity to Reform UK immigration rhetoric + volume normalization
- **Career variables**: Parliamentary tenure, ministerial experience, backbench years, career frustration signals

### Training Approach
- **Algorithm**: Logistic regression with L2 regularization
- **Training data**: 19 confirmed Conservative-to-Reform defections (2024-2026)
- **Features**: 2 categories (speech + career), automatically weighted by model
- **Performance**: 97.8% cross-validated AUC

### Key Innovation
**Volume normalization** solves the "high-volume speaker bias" problem:
- Without normalization: High-volume speakers (ministers, PM) scored highest due to frequency
- With normalization: Rishi Sunak (2,319 speeches) correctly identified as very low risk (0.08%) because his speeches rarely focus on immigration

---

## 🏆 Top 5 At-Risk MPs (January 2026)

| Rank | MP | Risk | Reform Alignment | Immigration % | Status |
|------|----|----|------------------|---------------|--------|
| 1 | Robert Jenrick | **93.67%** | 0.0209 | 13.7% | Minister |
| 2 | Katie Lam | 89.22% | 0.0162 | 9.8% | Backbencher |
| 3 | Suella Braverman | 84.04% | 0.0190 | 12.3% | Minister |
| 4 | Rebecca Harris | 72.81% | 0.0093 | 7.1% | Backbencher |
| 5 | Nick Timothy | 69.18% | 0.0094 | 6.8% | Backbencher |

**Validation**: Robert Jenrick (predicted #1) defected on January 15, 2026 ✅

---

## 📊 Data Sources (Final Model)

**Primary Sources:**
- **Hansard speeches**: 199,701 speeches (2019-2025) via MySociety ParlParse
- **Ministerial data**: IFG database (172 ministers' career histories)
- **MP demographics**: Wikidata + Parliament Members API (118 current Conservative MPs)
- **Ground truth defections**: 19 confirmed Conservative→Reform defections (2024-2026)

**Exploratory/Legacy Sources** (Phase 1 experiments, not in final model):
- Public Whip voting data
- Constituency Reform vote share
- 2024 election results
- Rwanda Bill votes

See [methodologies/methodology.md](methodologies/methodology.md) for complete data source details.

---

## 🔬 Model Development Journey

### Phase 1: Preliminary Experimentation ✓
Tested various feature combinations:
- Constituency demographics (Reform vote %, majority vulnerability)
- MP demographics (age, gender, tenure)
- MP career variables (ministerial history, backbench years)
- Speech/keyword analysis (basic keyword counting, TF-IDF vectorization)

**Models explored**: See `analysis/basic_demogs_model/`, `basic_demogs_keywords_model/`, `basic_demogs_vectorisation_model/`

### Phase 2: Feature Selection ✓
Identified most predictive features:
- ✅ **Speech analysis** (TF-IDF with volume normalization)
- ✅ **Career variables** (tenure, ministerial experience, backbench years)
- ❌ Constituency features (dropped - not in final model)
- ❌ Voting behavior (dropped - not in final model)

### Phase 3: Training on Historical Defections ✓
- Fit logistic regression on 19 confirmed defections
- Model automatically learns feature importance
- Discovers interaction effects between speech and career variables
- 97.8% cross-validated AUC achieved

### Phase 4: Application & Validation ✓
- Applied to 118 current Conservative MPs
- **Real-world validation**: Robert Jenrick (predicted #1) defected January 15, 2026
- **Case study validation**: Katie Lam, Suella Braverman ranked high (publicly frustrated)
- **Ideological validation**: One Nation Tories near bottom; leader Jeremy Hunt lowest (0.08%)

**Status**: ✅ Model validated and production-ready

---

## 📁 Key Output Files

### 🌟 Main Results (Use These)
| File | Location | Content |
|------|----------|---------|
| **final_results.md** | Root folder | Executive summary with top 10 & bottom 10 |
| **current_mp_defection_risk_scores.csv** | `analysis/training_tfidf_model_final_spec/` | Full predictions for all 118 MPs |
| **current_mp_defection_risk_report.txt** | `analysis/training_tfidf_model_final_spec/` | Human-readable top 20 summary |

### 📊 Detailed Outputs
| File | Content |
|------|---------|
| `enhanced_speech_tfidf.csv` | Per-MP speech statistics (volume, immigration %, Reform alignment) |
| `enhanced_speech_tfidf_normalized.csv` | Volume-normalized speech metrics |
| `mp_career_features.csv` | Career data (tenure, ministerial rank, backbench years) |

### 📖 Documentation
| File | Content |
|------|---------|
| `MODEL_SPEC.md` | Model specification, theory of change, detailed results |
| `TRAINING_ANALYSIS_REPORT.md` | Detailed training results and validation metrics |
| `VECTORIZATION_COMPLETE.md` | Speech vectorization methodology notes |

---

## 📚 Model Documentation

### For Technical Details
- **Model Specification**: [analysis/training_tfidf_model_final_spec/MODEL_SPEC.md](analysis/training_tfidf_model_final_spec/MODEL_SPEC.md)
  - Theory of change
  - Feature engineering details
  - Defection mechanisms (career frustration, ideological alignment, etc.)
  - Key findings and validation

- **Methodology**: [methodologies/methodology.md](methodologies/methodology.md)
  - Data sources
  - Feature engineering (speech analysis, career variables)
  - Model training approach
  - Validation methods
  - Limitations and assumptions
  - Future enhancements

### For Results
- **Executive Summary**: [final_results.md](final_results.md)
  - Context and objective
  - Top 10 and bottom 10 rankings
  - Key findings
  - Model performance metrics

---

## 🚀 Using the Model

### View Results (No Setup Required)
1. Read [final_results.md](final_results.md) for executive summary
2. Open `analysis/training_tfidf_model_final_spec/current_mp_defection_risk_scores.csv` for full rankings

### Run Model Pipeline (Development)
```bash
cd analysis/training_tfidf_model_final_spec/

# Train model on historical defections
python training_past_defections/optimize_model_weights.py

# Apply to current 118 MPs
python apply_model_to_current_mps.py

# View results
cat current_mp_defection_risk_report.txt
```

### Update Speech Data (Optional)
```bash
# Regenerate speech analysis
python enhanced_speech_tfidf.py

# Normalize metrics
python normalize_speech_metrics.py
```

---

## ❓ Questions?

- **Model overview & results**: See [final_results.md](final_results.md)
- **Model specification & findings**: See [analysis/training_tfidf_model_final_spec/MODEL_SPEC.md](analysis/training_tfidf_model_final_spec/MODEL_SPEC.md)
- **Methodology & data sources**: See [methodologies/methodology.md](methodologies/methodology.md)

---

## 📋 Project Metadata

- **Model Version**: 1.0 (Production-Ready)
- **Status**: ✅ Validated
- **Last Updated**: 16 January 2026
- **Coverage**: 118 of 121 current Conservative MPs (97.5%)
- **Training Data**: 19 confirmed defections (2024-2026)
- **Performance**: 97.8% cross-validated AUC, 84.2% recall@20
- **Real-world validation**: Robert Jenrick (predicted #1) defected January 15, 2026 ✅
