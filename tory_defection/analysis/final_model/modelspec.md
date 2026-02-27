# Model Spec: Tory Defectors model

## Overview
We use guided machine learning to build an optimal model for predicting upcoming MP defections from the Conservative Party to Reform UK. 

We build a training dataset of 2019-2024 Conservative MPs and a test dataset of current sitting Conservative MPs. The datasets merge parliamentary tenure, ministerial experience, age data, speech-derived features, and desk research on faction membership. We use TF-IDF vectorisation to analyse 10 years' of Hansard data on the speeches MPs have made to quantify their attitudes and alignment with Reform.

We also build additional features from these base indicators to measure MPs' attitudes towards their career prospects and party, allowing flexibility in how our models represent complex choices / defection risk, We also specify plausible interaction effects as potential features for our models. 

Our theory of change is that MPs who defect to Reform are those on the right of the Conservative Party who are aligned with Reform's policy positions - particularly on immigration - but also those experiencing career stagnation or disenfranchisement with their track in the party. We use machine learning to weight a composite score model of these variables and fit this model on the test dataset of sitting Conservative MPs to predict the lieklihood of their defecting. 

## Data Sources and Features
Base inputs are pulled from:
- MP tenure and ministerial records (Parliament Members API)
- Age data (Wikidata)
- Speech analysis features (Hansard)
- Desk research faction memberships (sources linked in scripts)

Engineered features include:
- Intensity of right-wing and anti-immigration views, alignment with Reform, and how views have changed over time
- Scores measuring ex-ministers facing career ceilings, rebels without ministerial experience, bachbench frustration, and establishment loyalty
- Socres representing membership of right-wing / moderate factions, and anti-defection penalties for party leaders

## Modelling Approach
We fit multiple supervised learning models on the training data to evaluate predictive performance and variable importance:
- Logistic regression (no regularization)
- Logistic regression (L1 and L2)
- Random forest
- Gradient boosting
- PCA + logistic regression

While these models favour some features more than others, they don't converge strongly on e.g., a top 3-5 variable set and see an AUC of 0.50 to 0.79. With only 23 defectors, there is (a) a concern from statistics that we are overfitting while also using too small a dataset that does not have enough degrees of freedom to allow for us to build a meaningful model and (b) a conceptual concern that Conservative defectors of the future will not look like Conservative defectors of the past; as more MPs defect, it becomes a real consideration for a wider group of MPs.

As a result, we also create a composite score model (ie we prescribe that the model must use all the features we have defined previously as being important or relevant to our theory of change) but use ML to detemrmine the relative importance those features have in the model. This gives us more control over the determinants of defection to keep them in line with our theory of change while also allowing us to use ML to fit a good model. One further advantage is that this approach gives us an interpretable index that can be compared against black-box models.

## Model Selection and Outputs
We compare models using cross-validated AUC on the training data. Given a range of AUCs and the data set size concerns outlined above, we appliy each trained model to the sitting MP test dataset and validate results partly using known context about likely defectors. The composite score model provides the best approach from this perspective.

## Future Improvements
- Refresh dataset as number of defectors grow
- Combine rankings across model outputs into a single ensemble ranking.
- Test alternative composite functional forms or constrained weight regimes.

## How to navigate this modelling
- `marketing/tory_defection/source_data/`
  - Demographics fetch scripts (in `mp_careers/`):
    - `fetch_mp_dob.py`: MP date-of-birth and age data.
    - `fetch_mp_tenure.py`: historical MP tenure data.
    - `fetch_mp_tenure_sitting_mps.py`: current MP tenure data.
    - `fetch_ministerial_tenure.py`: historical ministerial tenure data.
    - `fetch_ministerial_tenure_sitting_mps.py`: current ministerial tenure data.
  - `elections/`: 2019 and 2024 election results used to define training/test MP cohorts.
  - `hansard/`: raw Hansard debate XML used for speech feature extraction.
  - `mp_careers/`: fetch scripts and data inputs for demographics/tenure/ministerial data.
- `marketing/tory_defection/analysis/final_model/`
  - `faction_membership_data.py`: desk-research faction flags and weights.
  - `training_speech_analysis.py` / `test_speech_analysis.py`: build speech-derived features.
  - `create_training_data.py`: compiles the training dataset.
  - `create_test_data.py`: compiles the test dataset.
  - `fit_model_to_training_data.py`: trains ML models and optimizes composite weights.
  - `apply_model_to_test_data.py`: applies trained models and composite scoring to sitting MPs.

  - Outputs:
    - `training_data.csv`, `test_data.csv`
    - `training_speech_features.csv`, `test_speech_features.csv`
    - `model_comparison.csv`, `variable_importance.csv`, `composite_score_weights.csv`
    - `test_results/` (model output CSVs)


