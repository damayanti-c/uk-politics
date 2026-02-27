"""
Optimize Model Weights via Logistic Regression
===============================================

Uses logistic regression with L1/L2 regularization to determine optimal feature
weights for predicting defection risk.

Analyzes:
1. Feature importance from logistic regression coefficients
2. Optimal balance between speech features and demographic features
3. Model performance with different feature sets
4. Cross-validation to assess generalization

Note: With only 19 defectors, we use stratified K-fold CV and focus on
interpretability over complex models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent  # training_past_defections/
BASE_DIR = SCRIPT_DIR.parent  # training_tfidf_model_final_spec/

TRAINING_DATA = SCRIPT_DIR / "training_data_2024.csv"

# =============================================================================
# LOAD DATA
# =============================================================================

def load_and_prepare_data():
    """Load training data and prepare for modeling."""

    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATA)
    print(f"  Loaded {len(df)} MPs ({df['defected'].sum()} defectors)")

    # Define feature groups
    speech_features = [
        'reform_alignment_raw',
        'reform_alignment_normalized',
        'radicalization_slope',
        'extremism_percentile',
        'immigration_proportion',
        'total_speeches'
    ]

    # Career trajectory features (no age - too much missing data)
    career_features = [
        'backbench_years',
        'ever_minister',
        'total_minister_years',
        'highest_ministerial_rank',
        'years_since_last_ministerial_role',
        'cabinet_level',
        'career_stagnation_score'
    ]

    # Remove features with all zeros or NaNs
    available_features = []
    for feature in speech_features + career_features:
        if feature in df.columns:
            if df[feature].std() > 0:  # Has variation
                available_features.append(feature)

    print(f"\n  Available features with variation: {len(available_features)}")

    # Separate into feature groups
    available_speech = [f for f in speech_features if f in available_features]
    available_demographic = [f for f in career_features if f in available_features]

    print(f"    Speech features: {len(available_speech)}")
    print(f"    Demographic features: {len(available_demographic)}")

    # Prepare X and y
    X = df[available_features].fillna(0).values
    y = df['defected'].values

    feature_names = available_features
    speaker_names = df['speaker_name'].values

    return X, y, feature_names, speaker_names, available_speech, available_demographic


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_logistic_regression(X, y, feature_names, penalty='l2', C=1.0):
    """Train logistic regression model."""

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_scaled, y)

    # Get feature importances (coefficients)
    coefficients = model.coef_[0]

    # Create results dataframe
    results = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'scaled_importance': np.abs(coefficients) / np.abs(coefficients).sum()
    })

    results = results.sort_values('abs_coefficient', ascending=False)

    return model, scaler, results


def cross_validate_model(X, y, penalty='l2', C=1.0, n_splits=5):
    """Perform stratified K-fold cross-validation."""

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create model
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )

    # Stratified K-fold (ensures each fold has similar defector ratio)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validate
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc')

    return cv_scores


def find_optimal_regularization(X, y):
    """Find optimal regularization strength via cross-validation."""

    print("\n" + "=" * 80)
    print("FINDING OPTIMAL REGULARIZATION")
    print("=" * 80)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different C values (inverse regularization strength)
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    results = []

    for C in C_values:
        cv_scores = cross_validate_model(X, y, penalty='l2', C=C, n_splits=5)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        results.append({
            'C': C,
            'mean_auc': mean_score,
            'std_auc': std_score
        })

        print(f"  C={C:>7.3f} | Mean AUC: {mean_score:.4f} +/- {std_score:.4f}")

    results_df = pd.DataFrame(results)
    best_C = results_df.loc[results_df['mean_auc'].idxmax(), 'C']

    print(f"\n  Best C: {best_C} (AUC: {results_df['mean_auc'].max():.4f})")

    return best_C, results_df


# =============================================================================
# FEATURE SET COMPARISON
# =============================================================================

def compare_feature_sets(X, y, feature_names, speech_features, career_features):
    """Compare models with different feature sets."""

    print("\n" + "=" * 80)
    print("COMPARING FEATURE SETS")
    print("=" * 80)

    # Get indices for different feature sets
    speech_indices = [i for i, f in enumerate(feature_names) if f in speech_features]
    demographic_indices = [i for i, f in enumerate(feature_names) if f in career_features]

    feature_sets = {
        'Speech Only': speech_indices,
        'Demographics Only': demographic_indices,
        'All Features': list(range(len(feature_names)))
    }

    results = []

    for set_name, indices in feature_sets.items():
        if len(indices) == 0:
            continue

        X_subset = X[:, indices]
        cv_scores = cross_validate_model(X_subset, y, C=1.0, n_splits=5)

        results.append({
            'feature_set': set_name,
            'n_features': len(indices),
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'features': [feature_names[i] for i in indices]
        })

        print(f"\n{set_name}:")
        print(f"  Features: {len(indices)}")
        print(f"  Cross-validated AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    results_df = pd.DataFrame(results)

    return results_df


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("MODEL WEIGHT OPTIMIZATION")
    print("=" * 80)
    print()

    # Load data
    X, y, feature_names, speaker_names, speech_features, career_features = load_and_prepare_data()

    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)
    print(f"\n  Defectors: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  Non-defectors: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
    print(f"\n  Note: Severe class imbalance - using balanced class weights")

    # Find optimal regularization
    best_C, regularization_results = find_optimal_regularization(X, y)

    # Train final model with optimal C
    print("\n" + "=" * 80)
    print("FINAL MODEL WITH OPTIMAL REGULARIZATION")
    print("=" * 80)

    model, scaler, feature_importance = train_logistic_regression(
        X, y, feature_names, penalty='l2', C=best_C
    )

    print("\nFeature Importance (sorted by absolute coefficient):")
    print("-" * 80)
    for _, row in feature_importance.iterrows():
        direction = "+" if row['coefficient'] > 0 else "-"
        print(f"  {row['feature']:<40} | {direction} | Coef: {row['coefficient']:>8.4f} | Weight: {row['scaled_importance']:.2%}")

    # Cross-validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION PERFORMANCE")
    print("=" * 80)

    cv_scores = cross_validate_model(X, y, penalty='l2', C=best_C, n_splits=5)
    print(f"\n  5-Fold Cross-Validated AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Compare feature sets
    feature_set_comparison = compare_feature_sets(
        X, y, feature_names, speech_features, career_features
    )

    # Predict probabilities on full dataset
    X_scaled = scaler.transform(X)
    predicted_probs = model.predict_proba(X_scaled)[:, 1]

    # Create results dataframe
    predictions_df = pd.DataFrame({
        'speaker_name': speaker_names,
        'actual_defection': y,
        'predicted_probability': predicted_probs
    })

    predictions_df = predictions_df.sort_values('predicted_probability', ascending=False)

    # Save results
    output_path = SCRIPT_DIR / "model_optimization_results.csv"
    feature_importance.to_csv(output_path, index=False)
    print(f"\n\nSaved feature importance to: {output_path}")

    predictions_path = SCRIPT_DIR / "model_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("\nTop 5 Most Important Features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['scaled_importance']:.1%} weight")

    print("\n\nFeature Set Performance:")
    for _, row in feature_set_comparison.iterrows():
        print(f"  {row['feature_set']:<25} AUC: {row['mean_auc']:.4f}")

    # Show top predicted defectors
    print("\n" + "=" * 80)
    print("TOP 20 PREDICTED DEFECTION RISKS")
    print("=" * 80)

    for _, row in predictions_df.head(20).iterrows():
        actual = "DEFECTED" if row['actual_defection'] == 1 else ""
        print(f"  {row['speaker_name']:<35} | Prob: {row['predicted_probability']:.4f} | {actual}")

    # Show actual defectors and their rankings
    print("\n" + "=" * 80)
    print("ACTUAL DEFECTORS - MODEL RANKINGS")
    print("=" * 80)

    defectors = predictions_df[predictions_df['actual_defection'] == 1].copy()
    defectors['rank'] = range(1, len(defectors) + 1)

    for _, row in defectors.iterrows():
        print(f"  Rank #{row['rank']:<4} | {row['speaker_name']:<30} | Prob: {row['predicted_probability']:.4f}")

    # Calculate recall at different thresholds
    print("\n" + "=" * 80)
    print("MODEL RECALL AT DIFFERENT THRESHOLDS")
    print("=" * 80)

    total_defectors = y.sum()
    for top_k in [10, 20, 50, 100]:
        if top_k > len(predictions_df):
            continue
        defectors_in_top_k = predictions_df.head(top_k)['actual_defection'].sum()
        recall = defectors_in_top_k / total_defectors
        print(f"  Top {top_k:>3} MPs: Captured {defectors_in_top_k}/{total_defectors} defectors ({recall:.1%} recall)")

    return model, scaler, feature_importance, predictions_df


if __name__ == "__main__":
    results = main()
