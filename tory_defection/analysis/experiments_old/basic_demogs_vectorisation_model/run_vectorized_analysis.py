"""
Vectorized Risk Analysis
========================

Combines enhanced speech vectorization with existing risk factors.

Updated model weights:
- Speech Analysis (40%): Enhanced with semantic similarity
- Voting Rebellion (30%): Rebellion rate + Rwanda Bill
- Constituency Risk (20%): Reform vote share, majority
- Demographics (10%): Gender, tenure, age
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SOURCE_DATA = BASE_DIR / "source_data"
ANALYSIS_OUTPUT = BASE_DIR / "analysis"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_all_data():
    """Load all data sources including enhanced speech vectorization."""

    print("Loading data sources...")

    # Enhanced speech analysis (vectorized)
    # Try TF-IDF version first (no PyTorch), then transformer version
    tfidf_path = ANALYSIS_OUTPUT / "enhanced_speech_tfidf.csv"
    transformer_path = ANALYSIS_OUTPUT / "enhanced_speech_vectorization.csv"

    if tfidf_path.exists():
        enhanced_speech = pd.read_csv(tfidf_path)
        print(f"OK Loaded enhanced speech (TF-IDF) ({len(enhanced_speech)} MPs)")
    elif transformer_path.exists():
        enhanced_speech = pd.read_csv(transformer_path)
        print(f"OK Loaded enhanced speech (Transformers) ({len(enhanced_speech)} MPs)")
    else:
        print("X Enhanced speech analysis not found")
        print("  Run: python enhanced_speech_tfidf.py (no admin needed)")
        print("  Or: python enhanced_speech_vectorization.py (requires Visual C++)")
        enhanced_speech = pd.DataFrame()

    # Existing analysis
    existing_scores_path = ANALYSIS_OUTPUT / "mp_enhanced_risk_scores.csv"
    existing_scores = pd.read_csv(existing_scores_path)
    print(f"OK Loaded existing risk scores ({len(existing_scores)} MPs)")

    # Election results
    election_path = SOURCE_DATA / "elections_2024" / "HoC-GE2024-results-by-constituency.csv"
    election = pd.read_csv(election_path)
    print(f"OK Loaded election results")

    # Rebellion data
    rebellion_path = SOURCE_DATA / "voting_records" / "rebellion_rates.csv"
    rebellion = pd.read_csv(rebellion_path)
    print(f"OK Loaded rebellion rates ({len(rebellion)} MPs)")

    # Rwanda votes
    rwanda_path = SOURCE_DATA / "voting_records" / "rwanda_bill_votes.csv"
    rwanda = pd.read_csv(rwanda_path)
    print(f"OK Loaded Rwanda Bill votes ({len(rwanda)} MPs)")

    return {
        'enhanced_speech': enhanced_speech,
        'existing_scores': existing_scores,
        'election': election,
        'rebellion': rebellion,
        'rwanda': rwanda
    }


# =============================================================================
# MERGE AND CALCULATE VECTORIZED RISK SCORES
# =============================================================================

def calculate_vectorized_risk_scores(data):
    """Calculate risk scores with enhanced speech vectorization."""

    print("\nCalculating vectorized risk scores...")

    df = data['existing_scores'].copy()

    # Merge enhanced speech scores
    if not data['enhanced_speech'].empty:
        df = df.merge(
            data['enhanced_speech'][[
                'speaker_name', 'reform_alignment_score', 'enhanced_speech_score',
                'radicalization_slope', 'hardline_compassion_ratio',
                'extremism_percentile'
            ]],
            left_on='name',
            right_on='speaker_name',
            how='left'
        )
        print(f"OK Merged enhanced speech scores for {df['reform_alignment_score'].notna().sum()} MPs")
    else:
        df['reform_alignment_score'] = 0.0
        df['enhanced_speech_score'] = 0.0
        df['radicalization_slope'] = 0.0
        df['hardline_compassion_ratio'] = 0.0
        df['extremism_percentile'] = 50.0

    # Fill missing values
    df['reform_alignment_score'] = df['reform_alignment_score'].fillna(0.0)
    df['enhanced_speech_score'] = df['enhanced_speech_score'].fillna(0.0)
    df['radicalization_slope'] = df['radicalization_slope'].fillna(0.0)
    df['hardline_compassion_ratio'] = df['hardline_compassion_ratio'].fillna(0.0)
    df['extremism_percentile'] = df['extremism_percentile'].fillna(50.0)

    # Calculate NEW weighted risk score
    # Speech: 40% (increased from 35% with better analysis)
    # Voting: 30%
    # Constituency: 20% (decreased from 25%)
    # Demographics: 10%

    df['vectorized_speech_score'] = df['enhanced_speech_score'] * 0.40

    # Normalize rebellion rate to 0-1 (cap at 10%)
    df['rebellion_norm'] = df['rebellion_rate'].fillna(0) / 10
    df['rebellion_norm'] = df['rebellion_norm'].clip(0, 1)

    df['vectorized_voting_score'] = (
        df['rebellion_norm'] * 0.70 +
        df['rwanda_rebellion'].fillna(0) * 0.30
    ) * 0.30

    # Normalize constituency metrics to 0-1
    df['reform_norm'] = df['reform_pct'].fillna(0) / 30  # Cap at 30%
    df['majority_norm'] = (20 - df['majority_pct'].fillna(20)) / 20  # Lower is riskier
    df['con_weakness_norm'] = (40 - df['con_pct'].fillna(40)) / 40  # Lower is riskier

    df['vectorized_constituency_score'] = (
        df['reform_norm'].clip(0, 1) * 0.50 +
        df['majority_norm'].clip(0, 1) * 0.30 +
        df['con_weakness_norm'].clip(0, 1) * 0.20
    ) * 0.20

    # Demographics (use simple normalizations)
    df['gender_norm'] = df['gender'].apply(lambda x: 0.7 if x == 'M' else 0.3)
    df['tenure_norm'] = df['years_as_mp'].fillna(0) / 25  # Cap at 25 years

    df['vectorized_demographics_score'] = (
        df['gender_norm'] * 0.50 +
        df['tenure_norm'].clip(0, 1) * 0.30 +
        0.5 * 0.20  # Age default (not available)
    ) * 0.10

    # Overall vectorized risk score
    df['vectorized_risk_score'] = (
        df['vectorized_speech_score'] +
        df['vectorized_voting_score'] +
        df['vectorized_constituency_score'] +
        df['vectorized_demographics_score']
    )

    # Categorize risk
    df['vectorized_risk_category'] = pd.cut(
        df['vectorized_risk_score'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    # Sort by new risk score
    df = df.sort_values('vectorized_risk_score', ascending=False)

    return df


# =============================================================================
# GENERATE REPORT
# =============================================================================

def generate_vectorized_report(df):
    """Generate report with vectorized analysis."""

    report_path = ANALYSIS_OUTPUT / "vectorized_defection_risk_report.txt"

    lines = [
        "=" * 80,
        "VECTORIZED TORY MP DEFECTION RISK REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 80,
        "",
        "METHODOLOGY - ENHANCED WITH SEMANTIC VECTORIZATION",
        "-" * 80,
        "Risk Score Composition (updated model):",
        "",
        "  1. SPEECH ANALYSIS (40% weight) - ENHANCED WITH VECTORIZATION",
        "     - Reform UK rhetoric alignment via semantic similarity (35% of speech)",
        "     - Radicalization trajectory over time (25% of speech)",
        "     - Hardline vs compassionate language ratio (20% of speech)",
        "     - Extremism percentile ranking (15% of speech)",
        "     - Speech volume/engagement (5% of speech)",
        "     Method: Sentence transformers (all-MiniLM-L6-v2) measure cosine",
        "             similarity to Reform UK immigration talking points",
        "",
        "  2. VOTING REBELLION (30% weight)",
        "     - Overall rebellion rate vs party (70% of voting = 21% total)",
        "     - Rwanda Bill rebellion/abstention (30% of voting = 9% total)",
        "",
        "  3. CONSTITUENCY RISK (20% weight)",
        "     - Reform UK vote share 2024 (50% of constituency = 10% total)",
        "     - Majority size vulnerability (30% of constituency = 6% total)",
        "     - Conservative vote weakness (20% of constituency = 4% total)",
        "",
        "  4. DEMOGRAPHICS (10% weight)",
        "     - Gender (50% of demographics = 5% total)",
        "     - Parliamentary tenure (30% of demographics = 3% total)",
        "     - Age (20% of demographics = 2% total)",
        "",
        "=" * 80,
        "",
        "SUMMARY STATISTICS",
        "-" * 80,
        f"Total Conservative MPs analyzed: {len(df)}",
        ""
    ]

    # Risk distribution
    if 'vectorized_risk_category' in df.columns:
        risk_counts = df['vectorized_risk_category'].value_counts()
        lines.append("Risk distribution:")
        for cat in ['Very High', 'High', 'Medium', 'Low']:
            count = risk_counts.get(cat, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            lines.append(f"  {cat}: {count} MPs ({pct:.1f}%)")

    # Comparison to old model
    lines.extend([
        "",
        "Comparison to keyword-based model:",
        f"  Average score change: {(df['vectorized_risk_score'] - df['risk_score']).mean():+.3f}",
        f"  Largest increase: {(df['vectorized_risk_score'] - df['risk_score']).max():+.3f}",
        f"  MPs with higher risk (+0.1): {sum((df['vectorized_risk_score'] - df['risk_score']) > 0.1)}",
        ""
    ])

    # Top 25
    lines.extend([
        "=" * 80,
        "",
        "TOP 25 MPS AT HIGHEST DEFECTION RISK (VECTORIZED MODEL)",
        "-" * 80,
        ""
    ])

    top_25 = df.head(25)

    for idx, (_, row) in enumerate(top_25.iterrows(), 1):
        name = row.get('name', 'Unknown')
        constituency = row.get('constituency', 'Unknown')
        v_score = row['vectorized_risk_score']
        old_score = row.get('risk_score', 0)
        change = v_score - old_score

        lines.append(f"\n{idx:2}. {name} ({constituency})")
        lines.append(f"    Vectorized Risk Score: {v_score:.3f} ({row.get('vectorized_risk_category', 'N/A')})")
        lines.append(f"    Change from keyword model: {change:+.3f}")
        lines.append(f"    Breakdown:")
        lines.append(f"      - Speech (40%): {row.get('vectorized_speech_score', 0):.3f}")
        lines.append(f"      - Voting (30%): {row.get('vectorized_voting_score', 0):.3f}")
        lines.append(f"      - Constituency (20%): {row.get('vectorized_constituency_score', 0):.3f}")
        lines.append(f"      - Demographics (10%): {row.get('vectorized_demographics_score', 0):.3f}")

        # Enhanced speech details
        reform_alignment = row.get('reform_alignment_score', 0)
        radical_slope = row.get('radicalization_slope', 0)
        extremism_pct = row.get('extremism_percentile', 0)

        lines.append(f"    Speech Analysis Details:")
        lines.append(f"      - Reform rhetoric alignment: {reform_alignment:.3f}")
        if radical_slope != 0:
            lines.append(f"      - Radicalization trajectory: {radical_slope:+.4f} per quarter")
        lines.append(f"      - Extremism percentile: {extremism_pct:.1f}%")

        # Context
        rebellion = row.get('rebellion_rate', 0)
        reform_pct = row.get('reform_pct', 0)
        if pd.notna(rebellion) and rebellion > 0:
            lines.append(f"      - Rebellion rate: {rebellion:.1f}%")
        if pd.notna(reform_pct):
            lines.append(f"      - Constituency Reform vote: {reform_pct:.1f}%")

    lines.extend([
        "",
        "=" * 80,
        "",
        "KEY FINDINGS",
        "-" * 80,
        ""
    ])

    # Top movers
    df_sorted_by_change = df.copy()
    df_sorted_by_change['score_change'] = df_sorted_by_change['vectorized_risk_score'] - df_sorted_by_change['risk_score']
    top_movers = df_sorted_by_change.nlargest(10, 'score_change')

    lines.append("MPs with LARGEST RISK INCREASE from vectorization:")
    for _, row in top_movers.iterrows():
        change = row['score_change']
        lines.append(f"  - {row['name']}: {change:+.3f} (now {row['vectorized_risk_score']:.3f})")

    lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])

    report = "\n".join(lines)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nOK Report saved to: {report_path}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("VECTORIZED DEFECTION RISK ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    data = load_all_data()

    if data['enhanced_speech'].empty:
        print("\nERROR: Enhanced speech vectorization data not found!")
        print("Please run: python enhanced_speech_vectorization.py")
        return None

    # Calculate vectorized risk scores
    df = calculate_vectorized_risk_scores(data)

    # Save results
    output_path = ANALYSIS_OUTPUT / "mp_vectorized_risk_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"\nOK Vectorized risk scores saved to: {output_path}")

    # Generate report
    report = generate_vectorized_report(df)

    # Print top 10
    print("\n" + "=" * 80)
    print("TOP 10 MPS - VECTORIZED RISK MODEL")
    print("=" * 80)
    for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
        name = row['name']
        v_score = row['vectorized_risk_score']
        old_score = row.get('risk_score', 0)
        change = v_score - old_score

        print(f"\n{idx:2}. {name}")
        print(f"    Vectorized Score: {v_score:.3f} | Old Score: {old_score:.3f} | Change: {change:+.3f}")
        print(f"    Reform Alignment: {row['reform_alignment_score']:.3f} | Extremism: {row['extremism_percentile']:.0f}%")

    return df


if __name__ == "__main__":
    results = main()
