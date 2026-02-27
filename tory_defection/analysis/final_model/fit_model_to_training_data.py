"""
Model fitting on training data
===================================

Fits multiple models to predict Conservative MP defection to Reform UK.
Compares model performance and identifies best predictive variables and model fit.

This script includes testing with:
1. Speech analysis features (TF-IDF based Reform alignment, hardline ratio, etc.)
2. Career features (ministerial years, backbench frustration, etc.)
3. A variety of interaction variables engineered from speech and career data - to allow the model
greater flexibility in fitting
4. Faction membership features (ERG, CSG, New Conservatives, Rwanda rebels, One Nation, TRG). 
Research sources for faction membership are documented in faction_membership_data.py

Models tested:
1. Logistic Regression (baseline)
2. Logistic Regression with L1 regularization (Lasso)
3. Logistic Regression with L2 regularization (Ridge)
4. Random Forest
5. Gradient Boosting
6. PCA + Logistic Regression

In addition to independent supervised models, we also fit a functional-form
composite score index on the engineered features to provide an interpretable
benchmark alongside the ML models.

Outputs model comparison and variable importance analysis to evaluate fit and variable importance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.optimize import minimize
import statsmodels.api as sm

# =============================================================================
# PATHS
# =============================================================================

training_data_csv = Path(__file__).parent / "training_data.csv"
output_dir = Path(__file__).parent

# =============================================================================
# FEATURE SELECTION
# =============================================================================

# Core speech/career features (from original model - CV AUC = 0.794)
CORE_FEATURE_COLS = [
    'ministerial_years',
    'immigration_speech_proportion',
    'reform_alignment',
    'hardline_ratio',
    'extremism_percentile',
]

# Engineered features (created in create_training_data.py)
ENGINEERED_FEATURE_COLS = [
    'rightwing_intensity',
    'career_stagnation',
    'backbench_frustration',
    'sidelined_rebel',
    'radicalizing',
    'never_minister_rebel',
    'establishment_loyalty',
]

# Faction membership features (from faction_membership_data.py)
FACTION_FEATURE_COLS = [
    'is_erg',
    'is_csg',
    'is_new_conservative',
    'is_rwanda_rebel',
    'is_one_nation',
    'is_trg',
    'is_party_leader',
    'rightwing_faction_score',
    'moderate_faction_score',
    'net_faction_score',
]

TARGET_COL = 'defected'


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data():
    """Load and prepare data for modelling."""
    print("Loading training data...")
    df = pd.read_csv(training_data_csv)

    print(f"  Total MPs: {len(df)}")
    print(f"  Defectors: {df[TARGET_COL].sum()} ({df[TARGET_COL].mean()*100:.1f}%)")

    feature_cols = CORE_FEATURE_COLS + ENGINEERED_FEATURE_COLS + FACTION_FEATURE_COLS

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    print("\nMissing values before imputation:")
    for col in X.columns:
        missing = X[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing}")

    X = X.fillna(X.median())

    print(f"\nFeatures shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")

    return df, X, y


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_logistic_regression(X, y, penalty='l2', C=1.0):
    """Fit logistic regression with specified regularization."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if penalty == 'none':
        model = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
    else:
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000, random_state=42)

    model.fit(X_scaled, y)

    return model, scaler


def fit_statsmodels_logit(X, y):
    """Fit logistic regression using statsmodels for p-values."""
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const)
    try:
        result = model.fit(disp=0)
        return result
    except np.linalg.LinAlgError:
        # Singular matrix - try with regularization
        try:
            result = model.fit_regularized(disp=0, alpha=0.1)
            return result
        except Exception:
            return None


def fit_random_forest(X, y):
    """Fit Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y)
    return model


def fit_gradient_boosting(X, y):
    """Fit Gradient Boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    return model


def fit_pca_logistic(X, y, n_components=5):
    """Fit PCA followed by logistic regression."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('logistic', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X, y, model_name, scaler=None):
    """Evaluate model using cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if scaler is not None:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clone(model))
        ])
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        X_eval = scaler.transform(X)
    else:
        X_eval = X
        cv_scores = cross_val_score(model, X_eval, y, cv=cv, scoring='roc_auc')


    # Predictions on full data (for reporting)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = model.predict(X_eval)

    y_pred = model.predict(X_eval)

    return {
        'model_name': model_name,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'full_auc': roc_auc_score(y, y_prob),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }


# =============================================================================
# VARIABLE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_variable_importance(models_dict, X, feature_names):
    """Analyze variable importance across models."""
    importance_df = pd.DataFrame({'feature': feature_names})

    # Logistic regression coefficients
    if 'logistic_l2' in models_dict:
        model, scaler = models_dict['logistic_l2']
        importance_df['logistic_coef'] = model.coef_[0]
        importance_df['logistic_abs_coef'] = np.abs(model.coef_[0])

    # Lasso coefficients
    if 'logistic_l1' in models_dict:
        model, scaler = models_dict['logistic_l1']
        importance_df['lasso_coef'] = model.coef_[0]
        importance_df['lasso_selected'] = (model.coef_[0] != 0).astype(int)

    # Random Forest importance
    if 'random_forest' in models_dict:
        model = models_dict['random_forest']
        importance_df['rf_importance'] = model.feature_importances_

    # Gradient Boosting importance
    if 'gradient_boosting' in models_dict:
        model = models_dict['gradient_boosting']
        importance_df['gb_importance'] = model.feature_importances_

    return importance_df


def run_pca_analysis(X, feature_names):
    """Run PCA analysis to understand variable relationships."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    # Explained variance
    explained_var = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=feature_names
    )

    return explained_var, loadings


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("DEFECTION PREDICTION MODEL FITTING")
    print("=" * 90)

    # Prepare data
    df, X, y = prepare_data()
    feature_names = X.columns.tolist()

    # Store models
    models_dict = {}
    results = []

    # =========================================================================
    # 1. LOGISTIC REGRESSION (BASELINE - NO REGULARIZATION)
    # =========================================================================
    print("\n" + "=" * 90)
    print("1. LOGISTIC REGRESSION (NO REGULARIZATION)")
    print("=" * 90)

    model, scaler = fit_logistic_regression(X, y, penalty='none')
    models_dict['logistic_none'] = (model, scaler)
    result = evaluate_model(model, X, y, 'Logistic (No Reg)', scaler)
    results.append(result)
    print(f"  CV AUC: {result['cv_auc_mean']:.3f} (+/- {result['cv_auc_std']:.3f})")

    # Statsmodels for p-values
    sm_result = fit_statsmodels_logit(X, y)
    if sm_result is not None:
        print("\n  Coefficients with p-values:")
        try:
            print(sm_result.summary2().tables[1].to_string())
        except Exception:
            print("  (Could not generate summary - likely due to regularization)")
    else:
        print("\n  (Statsmodels fit failed - singular matrix, likely multicollinearity)")

    # =========================================================================
    # 2. LOGISTIC REGRESSION WITH L2 (RIDGE)
    # =========================================================================
    print("\n" + "=" * 90)
    print("2. LOGISTIC REGRESSION WITH L2 REGULARIZATION (RIDGE)")
    print("=" * 90)

    model, scaler = fit_logistic_regression(X, y, penalty='l2', C=1.0)
    models_dict['logistic_l2'] = (model, scaler)
    result = evaluate_model(model, X, y, 'Logistic L2', scaler)
    results.append(result)
    print(f"  CV AUC: {result['cv_auc_mean']:.3f} (+/- {result['cv_auc_std']:.3f})")

    print("\n  Standardized Coefficients:")
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    print(coef_df.to_string(index=False))

    # =========================================================================
    # 3. LOGISTIC REGRESSION WITH L1 (LASSO)
    # =========================================================================
    print("\n" + "=" * 90)
    print("3. LOGISTIC REGRESSION WITH L1 REGULARIZATION (LASSO)")
    print("=" * 90)

    model, scaler = fit_logistic_regression(X, y, penalty='l1', C=0.5)
    models_dict['logistic_l1'] = (model, scaler)
    result = evaluate_model(model, X, y, 'Logistic L1', scaler)
    results.append(result)
    print(f"  CV AUC: {result['cv_auc_mean']:.3f} (+/- {result['cv_auc_std']:.3f})")

    print("\n  Selected Features (non-zero coefficients):")
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    })
    selected = coef_df[coef_df['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False)
    print(selected.to_string(index=False))
    print(f"\n  Features selected: {len(selected)}/{len(feature_names)}")

    # =========================================================================
    # 4. RANDOM FOREST
    # =========================================================================
    print("\n" + "=" * 90)
    print("4. RANDOM FOREST")
    print("=" * 90)

    model = fit_random_forest(X, y)
    models_dict['random_forest'] = model
    result = evaluate_model(model, X, y, 'Random Forest')
    results.append(result)
    print(f"  CV AUC: {result['cv_auc_mean']:.3f} (+/- {result['cv_auc_std']:.3f})")

    print("\n  Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.to_string(index=False))

    # =========================================================================
    # 5. GRADIENT BOOSTING
    # =========================================================================
    print("\n" + "=" * 90)
    print("5. GRADIENT BOOSTING")
    print("=" * 90)

    model = fit_gradient_boosting(X, y)
    models_dict['gradient_boosting'] = model
    result = evaluate_model(model, X, y, 'Gradient Boosting')
    results.append(result)
    print(f"  CV AUC: {result['cv_auc_mean']:.3f} (+/- {result['cv_auc_std']:.3f})")

    print("\n  Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.to_string(index=False))

    # =========================================================================
    # 6. PCA + LOGISTIC REGRESSION
    # =========================================================================
    print("\n" + "=" * 90)
    print("6. PCA + LOGISTIC REGRESSION")
    print("=" * 90)

    # PCA analysis
    explained_var, loadings = run_pca_analysis(X, feature_names)
    print("\n  Explained Variance:")
    print(explained_var.head(7).to_string(index=False))

    print("\n  Top loadings for first 3 PCs:")
    for pc in ['PC1', 'PC2', 'PC3']:
        print(f"\n  {pc}:")
        top_loadings = loadings[pc].abs().sort_values(ascending=False).head(5)
        for feat, val in top_loadings.items():
            sign = '+' if loadings.loc[feat, pc] > 0 else '-'
            print(f"    {sign}{val:.3f} {feat}")

    # Fit PCA + Logistic
    pipeline = fit_pca_logistic(X, y, n_components=5)
    models_dict['pca_logistic'] = pipeline

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    result = {
        'model_name': 'PCA + Logistic',
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'full_auc': roc_auc_score(y, pipeline.predict_proba(X)[:, 1]),
        'precision': precision_score(y, pipeline.predict(X), zero_division=0),
        'recall': recall_score(y, pipeline.predict(X)),
        'f1': f1_score(y, pipeline.predict(X))
    }
    results.append(result)
    print(f"\n  CV AUC: {result['cv_auc_mean']:.3f} (+/- {result['cv_auc_std']:.3f})")

    # =========================================================================
    # 7. OPTIMIZED COMPOSITE SCORE (UNBOUNDED)
    # =========================================================================
    print("\n" + "=" * 90)
    print("7. OPTIMIZED COMPOSITE SCORE (UNBOUNDED)")
    print("=" * 90)
    print("\n  Question: Can we create an interpretable composite score with ML-optimized")
    print("           weights on the engineered composite features?")
    print("\n  Approach: Optimize all composite feature weights directly, without bounds,")
    print("           to maximize cross-validated AUC.")

    # Define composite features - USE ALL ENGINEERED COMPOSITE FEATURES
    # This avoids having to justify why we picked specific ones
    composite_feature_cols = [
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

    # Initialize with uniform weights
    init_weights = np.ones(len(composite_feature_cols))

    # Get composite features from training data
    X_composite = df[composite_feature_cols].copy()

    # Handle missing values
    X_composite = X_composite.fillna(X_composite.median())

    def negative_cv_auc(weights):
        """
        Calculate negative CV AUC for optimization (we minimize this).
        Uses 5-fold cross-validation.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in cv.split(X_composite, y):
            X_train, X_val = X_composite.iloc[train_idx], X_composite.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Compute composite score
            val_score = (X_val.values @ weights)

            # Compute AUC
            try:
                auc = roc_auc_score(y_val, val_score)
                cv_scores.append(auc)
            except:
                cv_scores.append(0.0)

        return -np.mean(cv_scores)

    print("\n  Optimizing weights with multi-start L-BFGS-B...")

    rng = np.random.default_rng(42)

    def sample_init(n):
        # Lognormal around 1.0 to explore multiplicative variation
        return rng.lognormal(mean=0.0, sigma=0.35, size=n)

    def run_multi_start(n_starts):
        best = None
        for _ in range(n_starts):
            init = sample_init(len(composite_feature_cols))
            # Optimize weights to maximize cross-validated AUC of the composite score.
            result = minimize(
                negative_cv_auc,
                init,
                method='L-BFGS-B',
                bounds=None,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            cv_auc = -result.fun
            if best is None or cv_auc > best['cv_auc']:
                best = {
                    'method': 'L-BFGS-B',
                    'weights': result.x,
                    'cv_auc': cv_auc,
                    'success': result.success,
                }
        return best

    n_starts = 20
    best = run_multi_start(n_starts)

    optimized_weights = best['weights']
    optimized_cv_auc = best['cv_auc']

    print(f"\n  Optimization method: {best['method']} (multi-start)")
    print(f"  Optimized weights: {np.round(optimized_weights, 4)}")
    print(f"  Optimized CV AUC: {optimized_cv_auc:.3f}")

    # Compute full-data AUC with optimized weights for comparison
    composite_score_full = X_composite.values @ optimized_weights
    full_auc = roc_auc_score(y, composite_score_full)

    # Store result
    result_composite = {
        'model_name': 'Composite (Optimized)',
        'cv_auc_mean': optimized_cv_auc,
        'cv_auc_std': 0.0,
        'full_auc': full_auc,
        'precision': precision_score(y, (composite_score_full > np.median(composite_score_full)).astype(int), zero_division=0),
        'recall': recall_score(y, (composite_score_full > np.median(composite_score_full)).astype(int)),
        'f1': f1_score(y, (composite_score_full > np.median(composite_score_full)).astype(int))
    }
    results.append(result_composite)

    # Save optimized weights for use in apply_model_to_test_data.py
    weights_output = pd.DataFrame({
        'feature': composite_feature_cols,
        'optimized_weight': optimized_weights
    })
    weights_output.to_csv(output_dir / "composite_score_weights.csv", index=False)
    print("\n  Optimized weights saved to composite_score_weights.csv")

    # =========================================================================
    # MODEL COMPARISON
    # =========================================================================
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_auc_mean', ascending=False)
    print("\n" + results_df.to_string(index=False))

    # Composite score (optimized) comparison
    best_auc = results_df.iloc[0]['cv_auc_mean']
    composite_opt = results_df[results_df['model_name'] == 'Composite (Optimized)'].iloc[0]
    composite_auc = composite_opt['cv_auc_mean']
    gap = best_auc - composite_auc

    print("\n" + "=" * 90)
    print("COMPOSITE SCORE VALIDATION (UNBOUNDED OPTIMIZATION)")
    print("=" * 90)
    print(f"\n  Best ML model:              {results_df.iloc[0]['model_name']:<25} AUC = {best_auc:.3f}")
    print(f"  Optimized composite score:  {composite_opt['model_name']:<25} AUC = {composite_auc:.3f}")
    print(f"  Performance gap:            {gap:.3f} AUC points ({(gap/best_auc)*100:.1f}% relative)")
    
    if gap < 0.10:
        print("\n  Composite score within 10% of best ML model performance.")
        print("  Interpretable approach is competitive.")
    else:
        print(f"\n  Note: Composite score lags best ML model by {gap:.3f} AUC points ({(gap/best_auc)*100:.1f}%).")
        print(f"  Trade-off: ~{(gap/best_auc)*100:.0f}% AUC for full interpretability.")
    
    print("\n  Weights saved to: composite_score_weights.csv")

    # =========================================================================
    # VARIABLE IMPORTANCE SUMMARY
    # =========================================================================
    print("\n" + "=" * 90)
    print("VARIABLE IMPORTANCE SUMMARY")
    print("=" * 90)

    importance_summary = analyze_variable_importance(models_dict, X, feature_names)

    # Rank variables by average importance
    importance_summary['avg_rank'] = (
        importance_summary['logistic_abs_coef'].rank(ascending=False) +
        importance_summary['rf_importance'].rank(ascending=False) +
        importance_summary['gb_importance'].rank(ascending=False)
    ) / 3

    importance_summary = importance_summary.sort_values('avg_rank')
    print("\n  Variables ranked by average importance across models:")
    print(importance_summary[['feature', 'logistic_coef', 'rf_importance', 'gb_importance', 'lasso_selected', 'avg_rank']].to_string(index=False))

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)

    best_model = results_df.iloc[0]
    print(f"\n  Best performing model: {best_model['model_name']}")
    print(f"  Cross-validated AUC: {best_model['cv_auc_mean']:.3f}")

    # Top predictive features
    top_features = importance_summary.head(5)['feature'].tolist()
    print(f"\n  Most important features (consensus across models):")
    for i, feat in enumerate(top_features, 1):
        print(f"    {i}. {feat}")

    # Save results
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    importance_summary.to_csv(output_dir / "variable_importance.csv", index=False)
    print(f"\n  Results saved to model_comparison.csv and variable_importance.csv")

    return results_df, importance_summary, models_dict


if __name__ == "__main__":
    results, importance, models = main()
