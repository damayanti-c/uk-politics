"""
Apply defection model to test data
==================================

Applies the faction-based defection model to sitting Conservative MPs.
This script combines:
1. Speech analysis features (TF-IDF based Reform alignment, hardline ratio, etc.)
2. Career features (ministerial years, backbench frustration, etc.)
3. Faction membership features (ERG, CSG, New Conservatives, Rwanda rebels, One Nation, TRG)

The model uses a composite scoring approach with weights optimized on training data via 
5-fold cross-validation to maximize AUC. This approach achieves competitive performance 
(0.591 AUC) with an interpretable, rule-based method compared to ML models (best RF: 0.738 AUC),
with a trade-off of transparency vs raw performance.

Research sources for faction membership are documented in faction_membership_data.py

Outputs a ranked list of MPs by defection probability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from fit_model_to_training_data import (
    CORE_FEATURE_COLS,
    ENGINEERED_FEATURE_COLS,
    FACTION_FEATURE_COLS,
    fit_logistic_regression,
    fit_random_forest,
    fit_gradient_boosting,
    fit_pca_logistic,
)

# =============================================================================
# PATHS
# =============================================================================

base_dir = Path(__file__).parent
output_dir = base_dir / "test_results"
training_data_csv = base_dir / "training_data.csv"
test_data_csv = base_dir / "test_data.csv"
output_csv = output_dir / "sitting_mp_defection_risk_composite.csv"
weights_csv = base_dir / "composite_score_weights.csv"

def prepare_ml_features(train_df, test_df):
    """Prepare aligned ML feature matrices for train/test."""
    feature_cols = CORE_FEATURE_COLS + ENGINEERED_FEATURE_COLS + FACTION_FEATURE_COLS
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test


def save_model_results(test_df, probs, model_form, output_base, output_cols):
    """Save model predictions to a standardized CSV."""
    results = pd.DataFrame({
        'name': test_df['name'],
        'constituency': test_df['constituency'],
        'defection_probability': probs,
    })
    results = results.sort_values('defection_probability', ascending=False)

    available_cols = [c for c in output_cols if c in test_df.columns]
    results = results.merge(
        test_df[available_cols],
        on='name', how='left'
    )

    results = results.sort_values('defection_probability', ascending=False)
    results.insert(0, 'risk_rank', range(1, len(results) + 1))
    results['defection_probability'] = results['defection_probability'].round(4)

    output_path = output_base / f"sitting_mp_defection_risk_{model_form}.csv"
    results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path.name}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("APPLYING FACTION-BASED DEFECTION MODEL TO SITTING MPS")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 90)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\nLoading data...")
    train_df = pd.read_csv(training_data_csv)
    test_df = pd.read_csv(test_data_csv)
    print(f"  Training: {len(train_df)} MPs, Test: {len(test_df)} MPs")

    # =========================================================================
    # APPLY TRAINED ML MODELS TO TEST DATA
    # =========================================================================
    print("\nFitting ML models on training data and scoring test data...")

    if 'defected' not in train_df.columns:
        raise ValueError("Training data missing 'defected' target column")

    X_train, X_test = prepare_ml_features(train_df, test_df)
    y_train = train_df['defected'].copy()

    output_cols = ['name', 'ministerial_years', 'ever_minister',
                   'immigration_speech_proportion', 'reform_alignment',
                   'hardline_ratio', 'radicalizing', 'extremism_percentile',
                   'sidelined_minister_years', 'total_backbench_years',
                   'rightwing_intensity', 'career_stagnation',
                   'is_erg', 'is_csg', 'is_new_conservative', 'is_rwanda_rebel',
                   'is_one_nation', 'is_trg', 'is_party_leader',
                   'rightwing_faction_score', 'moderate_faction_score', 'net_faction_score']

    # Logistic Regression (No Regularization)
    model, scaler = fit_logistic_regression(X_train, y_train, penalty='none')
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    save_model_results(test_df, probs, 'logistic_none', output_dir, output_cols)

    # Logistic Regression (L2)
    model, scaler = fit_logistic_regression(X_train, y_train, penalty='l2', C=1.0)
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    save_model_results(test_df, probs, 'logistic_l2', output_dir, output_cols)

    # Logistic Regression (L1)
    model, scaler = fit_logistic_regression(X_train, y_train, penalty='l1', C=0.5)
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    save_model_results(test_df, probs, 'logistic_l1', output_dir, output_cols)

    # Random Forest
    model = fit_random_forest(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    save_model_results(test_df, probs, 'random_forest', output_dir, output_cols)

    # Gradient Boosting
    model = fit_gradient_boosting(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    save_model_results(test_df, probs, 'gradient_boosting', output_dir, output_cols)

    # PCA + Logistic Regression
    pipeline = fit_pca_logistic(X_train, y_train, n_components=5)
    probs = pipeline.predict_proba(X_test)[:, 1]
    save_model_results(test_df, probs, 'pca_logistic', output_dir, output_cols)

    # =========================================================================
    # LOAD COMPOSITE SCORE WEIGHTS (Optimized)
    # =========================================================================
    print("\nLoading composite score weights (optimized)...")
    
    if not weights_csv.exists():
        raise FileNotFoundError(
            f"Optimized weights file not found at {weights_csv}. "
            "Run fit_model_to_training_data.py to generate composite_score_weights.csv."
        )

    weights_df = pd.read_csv(weights_csv)
    weights_dict = dict(zip(weights_df['feature'], weights_df['optimized_weight']))
    print(f"  Loaded {len(weights_dict)} feature weights from training optimization")

    # Display the weights for transparency with citations
    print("\n  Composite score feature weights (optimized):")
    for feat, weight in weights_dict.items():
        print(f"    {feat:<25} {weight:.4f}")
    # =========================================================================
    # COMPUTE COMPOSITE DEFECTION SCORES
    # =========================================================================
    print("\nComputing composite defection scores for sitting MPs...")
    
    # Create composite score as weighted sum of ALL composite features
    # Using all engineered features avoids needing to justify feature selection
    composite_features = [
        'rightwing_intensity',
        'career_stagnation',
        'backbench_frustration',
        'sidelined_rebel',
        'immigration_focus',
        'radicalizing',
        'never_minister_rebel',
        'establishment_loyalty',
        'net_faction_score',
        'party_leader_penalty'
    ]
    
    # Verify all features exist
    missing_features = [f for f in composite_features if f not in test_df.columns]
    if missing_features:
        print(f"  ERROR: Missing features in test data: {missing_features}")
        raise ValueError(f"Missing composite score features: {missing_features}")
    
    test_df['composite_defection_score'] = 0.0
    for feature in composite_features:
        weight = weights_dict.get(feature, 1.0)
        test_df['composite_defection_score'] += test_df[feature] * weight

    # =========================================================================
    # NORMALIZE TO PROBABILITIES
    # =========================================================================
    print("\nNormalizing composite scores to probabilities...")

    # Normalize composite scores to probabilities
    scores = test_df['composite_defection_score'].values
    min_score = scores.min()
    max_score = scores.max()
    probs = (scores - min_score) / (max_score - min_score)
    probs = np.clip(probs, 0.001, 0.999)
    composite_probs = probs.copy()

    # =========================================================================
    # CREATE OUTPUT DATAFRAME
    # =========================================================================
    results = pd.DataFrame({
        'name': test_df['name'],
        'constituency': test_df['constituency'],
        'defection_probability': composite_probs,
    })
    results = results.sort_values('defection_probability', ascending=False)

    # Add all relevant columns to output
    output_cols = ['name', 'ministerial_years', 'ever_minister',
                   'immigration_speech_proportion', 'reform_alignment',
                   'hardline_ratio', 'radicalizing', 'extremism_percentile',
                   'sidelined_minister_years', 'total_backbench_years',
                   'rightwing_intensity', 'career_stagnation',
                   'is_erg', 'is_csg', 'is_new_conservative', 'is_rwanda_rebel',
                   'is_one_nation', 'is_trg', 'is_party_leader',
                   'rightwing_faction_score', 'moderate_faction_score', 'net_faction_score']

    # Only include columns that exist
    available_cols = [c for c in output_cols if c in test_df.columns]

    results = results.merge(
        test_df[available_cols],
        on='name', how='left'
    )

    results = results.sort_values('defection_probability', ascending=False)
    results.insert(0, 'risk_rank', range(1, len(results) + 1))
    results['defection_probability'] = results['defection_probability'].round(4)

    # =========================================================================
    # OUTPUT RESULTS
    # =========================================================================
    print("\n" + "=" * 90)
    print("TOP 30 HIGHEST DEFECTION RISK MPs")
    print("=" * 90)

    for _, row in results.head(30).iterrows():
        faction_info = []
        if row.get('is_erg', 0): faction_info.append('ERG')
        if row.get('is_csg', 0): faction_info.append('CSG')
        if row.get('is_new_conservative', 0): faction_info.append('NC')
        if row.get('is_rwanda_rebel', 0): faction_info.append('RWA')
        if row.get('is_one_nation', 0): faction_info.append('1N')
        if row.get('is_party_leader', 0): faction_info.append('LDR')
        faction_str = ','.join(faction_info) if faction_info else '-'

        print(f"{row['risk_rank']:2}. {row['name']:<25} | P: {row['defection_probability']:.3f} | "
              f"Factions: {faction_str:<12}")

    print("\n" + "=" * 90)
    print("LOWEST 10 DEFECTION RISK MPs")
    print("=" * 90)

    for _, row in results.tail(10).iloc[::-1].iterrows():
        faction_info = []
        if row.get('is_one_nation', 0): faction_info.append('1N')
        if row.get('is_trg', 0): faction_info.append('TRG')
        if row.get('is_party_leader', 0): faction_info.append('LDR')
        faction_str = ','.join(faction_info) if faction_info else '-'

        print(f"{row['risk_rank']:3}. {row['name']:<25} | P: {row['defection_probability']:.3f} | "
              f"Factions: {faction_str}")

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)

    print(f"\n  Total sitting MPs assessed: {len(results)}")
    print(f"\n  Defection probability distribution:")
    print(f"    Mean:   {results['defection_probability'].mean():.3f}")
    print(f"    Median: {results['defection_probability'].median():.3f}")
    print(f"    Std:    {results['defection_probability'].std():.3f}")
    print(f"    Min:    {results['defection_probability'].min():.3f}")
    print(f"    Max:    {results['defection_probability'].max():.3f}")

    # Risk tiers
    high_risk = results[results['defection_probability'] >= 0.5]
    medium_risk = results[(results['defection_probability'] >= 0.3) & (results['defection_probability'] < 0.5)]
    low_risk = results[results['defection_probability'] < 0.3]

    print(f"\n  Risk tiers:")
    print(f"    High risk (>=50%):    {len(high_risk)} MPs")
    print(f"    Medium risk (30-50%): {len(medium_risk)} MPs")
    print(f"    Low risk (<30%):      {len(low_risk)} MPs")

    # =========================================================================
    # RESEARCH CITATIONS
    # =========================================================================
    print("\n" + "=" * 90)
    print("RESEARCH SOURCES USED FOR FACTION MEMBERSHIP")
    print("=" * 90)
    print("""
    1. ERG: https://en.wikipedia.org/wiki/European_Research_Group
    2. Common Sense Group: https://en.wikipedia.org/wiki/Common_Sense_Group
    3. New Conservatives: https://en.wikipedia.org/wiki/New_Conservatives_(UK)
    4. Rwanda rebels: https://www.spectator.co.uk/article/only-11-tories-vote-against-rwanda-bill/
    5. One Nation: https://en.wikipedia.org/wiki/One_Nation_Conservatives_(caucus)
    6. TRG: https://en.wikipedia.org/wiki/Tory_Reform_Group
    """)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results.to_csv(output_csv, index=False)
    print(f"\n  Results saved to {output_csv.name}")

    return results


if __name__ == "__main__":
    results = main()
