"""
Normalize Speech Metrics
=========================

Creates volume-adjusted Reform alignment scores to account for MPs
who talk about immigration frequently due to their role (e.g., PM)
rather than ideological alignment.

Outputs both raw and normalized versions for comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

TORY_DEFECTION_DIR = Path(__file__).parent.parent.parent
ANALYSIS_OUTPUT = Path(__file__).parent

SPEECH_TFIDF = ANALYSIS_OUTPUT / "enhanced_speech_tfidf.csv"
# Use extended dataset (2015-2026) for fuller coverage
HANSARD_CSV = TORY_DEFECTION_DIR / "source_data" / "hansard" / "all_speeches_extended.csv"

# =============================================================================
# LOAD DATA
# =============================================================================

def load_speech_data():
    """Load TF-IDF speech analysis and raw Hansard data."""

    print("Loading speech data...")

    # Load TF-IDF analysis
    tfidf_df = pd.read_csv(SPEECH_TFIDF)
    print(f"Loaded {len(tfidf_df)} MPs with TF-IDF speech analysis")

    # Load raw Hansard for sentence counts
    print("Loading Hansard speeches for sentence counting...")
    hansard_df = pd.read_csv(HANSARD_CSV)
    print(f"Loaded {len(hansard_df)} Hansard speeches")

    return tfidf_df, hansard_df


# =============================================================================
# CALCULATE IMMIGRATION SPEECH PROPORTION
# =============================================================================

def calculate_immigration_proportion(hansard_df):
    """Calculate what proportion of each MP's speeches are about immigration."""

    print("\nCalculating immigration speech proportions...")

    # Immigration keywords
    immigration_keywords = [
        'immigra', 'asylum', 'border', 'migrant', 'rwanda',
        'boat', 'channel', 'refuge'
    ]

    # Group by speaker
    speaker_stats = []

    for speaker_name in hansard_df['speaker_name'].unique():
        if pd.isna(speaker_name):
            continue

        speaker_speeches = hansard_df[hansard_df['speaker_name'] == speaker_name]

        total_speeches = len(speaker_speeches)
        total_sentences = 0
        immigration_sentences = 0

        for _, row in speaker_speeches.iterrows():
            text = row['text']
            if pd.isna(text):
                continue

            # Split into sentences (simple split on periods)
            sentences = [s.strip() for s in str(text).split('.') if len(s.strip()) > 20]
            total_sentences += len(sentences)

            # Count immigration sentences
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in immigration_keywords):
                    immigration_sentences += 1

        if total_sentences > 0:
            immigration_proportion = immigration_sentences / total_sentences
        else:
            immigration_proportion = 0

        speaker_stats.append({
            'speaker_name': speaker_name,
            'total_speeches': total_speeches,
            'total_sentences': total_sentences,
            'immigration_sentences': immigration_sentences,
            'immigration_proportion': immigration_proportion
        })

    proportion_df = pd.DataFrame(speaker_stats)
    print(f"Calculated proportions for {len(proportion_df)} speakers")

    return proportion_df


# =============================================================================
# NORMALIZE REFORM ALIGNMENT SCORES
# =============================================================================

def normalize_speech_scores(tfidf_df, proportion_df):
    """Create normalized Reform alignment scores."""

    print("\nNormalizing Reform alignment scores...")

    # Check if we already have the necessary columns from enhanced_speech_tfidf.py
    has_immigration_proportion = 'immigration_proportion' in tfidf_df.columns
    has_reform_alignment_raw = 'reform_alignment_raw' in tfidf_df.columns

    if has_immigration_proportion and has_reform_alignment_raw:
        # enhanced_speech_tfidf.py already calculated these, use them directly
        print("  Using pre-calculated immigration_proportion from TF-IDF analysis")
        normalized_df = tfidf_df.copy()
    else:
        # Merge proportion data (drop duplicate columns from proportion_df)
        proportion_df_for_merge = proportion_df.drop(columns=['total_speeches'], errors='ignore')

        normalized_df = tfidf_df.merge(
            proportion_df_for_merge,
            on='speaker_name',
            how='left'
        )

        # Keep raw score (column name changed after Brexit analysis was added)
        if 'reform_alignment_score' in normalized_df.columns:
            normalized_df['reform_alignment_raw'] = normalized_df['reform_alignment_score']

    # Method 1: Normalize by log of speech volume
    # This reduces scores for MPs who simply talk A LOT
    normalized_df['speech_volume_factor'] = np.log(normalized_df['total_speeches'] + 1)
    normalized_df['reform_alignment_volume_adj'] = (
        normalized_df['reform_alignment_raw'] / (1 + normalized_df['speech_volume_factor'] / 10)
    )

    # Method 2: Weight by immigration proportion (scaled)
    # This accounts for how FOCUSED they are on immigration
    # Scale up since proportions are very small (mean ~0.07%)
    normalized_df['immigration_proportion'] = normalized_df['immigration_proportion'].fillna(0)
    # Scale proportions to 0-1 range based on max
    max_proportion = normalized_df['immigration_proportion'].max()
    if max_proportion > 0:
        immigration_proportion_scaled = normalized_df['immigration_proportion'] / max_proportion
    else:
        immigration_proportion_scaled = normalized_df['immigration_proportion']

    normalized_df['reform_alignment_proportion_adj'] = (
        normalized_df['reform_alignment_raw'] * immigration_proportion_scaled
    )

    # Method 3: Combined normalization
    # Combines both volume and proportion adjustments
    # Formula: raw_score * scaled_proportion / (1 + log_volume_factor)
    normalized_df['reform_alignment_normalized'] = (
        normalized_df['reform_alignment_raw'] *
        immigration_proportion_scaled /
        (1 + np.log(normalized_df['total_speeches'] + 1) / 10)
    )

    # Fill NaN values
    for col in ['reform_alignment_volume_adj', 'reform_alignment_proportion_adj', 'reform_alignment_normalized']:
        normalized_df[col] = normalized_df[col].fillna(0)

    print(f"\nNormalization complete:")
    print(f"  Mean raw score: {normalized_df['reform_alignment_raw'].mean():.3f}")
    print(f"  Mean volume-adjusted: {normalized_df['reform_alignment_volume_adj'].mean():.3f}")
    print(f"  Mean proportion-adjusted: {normalized_df['reform_alignment_proportion_adj'].mean():.3f}")
    print(f"  Mean combined normalized: {normalized_df['reform_alignment_normalized'].mean():.3f}")

    return normalized_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("NORMALIZING SPEECH METRICS")
    print("=" * 80)
    print()

    # Load data
    tfidf_df, hansard_df = load_speech_data()

    # Calculate immigration proportions
    proportion_df = calculate_immigration_proportion(hansard_df)

    # Normalize scores
    normalized_df = normalize_speech_scores(tfidf_df, proportion_df)

    # Save output
    output_path = ANALYSIS_OUTPUT / "enhanced_speech_tfidf_normalized.csv"
    normalized_df.to_csv(output_path, index=False)
    print(f"\nSaved normalized speech metrics to: {output_path}")

    # Analysis: Compare raw vs normalized for key MPs
    print("\n" + "=" * 80)
    print("COMPARISON: RAW VS NORMALIZED SCORES")
    print("=" * 80)

    # Find MPs with biggest changes
    normalized_df['score_change'] = (
        normalized_df['reform_alignment_normalized'] -
        normalized_df['reform_alignment_raw']
    )

    print("\nTop 10 MPs with DECREASED scores (volume normalization effect):")
    decreased = normalized_df.nlargest(10, 'score_change')[
        ['speaker_name', 'reform_alignment_raw', 'reform_alignment_normalized',
         'score_change', 'total_speeches', 'immigration_proportion']
    ]
    for _, row in decreased.iterrows():
        print(f"  {row['speaker_name']:<30} | Raw: {row['reform_alignment_raw']:.3f} -> Norm: {row['reform_alignment_normalized']:.3f} | "
              f"Change: {row['score_change']:+.3f} | Speeches: {row['total_speeches']:.0f}")

    print("\nTop 10 MPs with INCREASED scores (proportion normalization effect):")
    increased = normalized_df.nsmallest(10, 'score_change')[
        ['speaker_name', 'reform_alignment_raw', 'reform_alignment_normalized',
         'score_change', 'total_speeches', 'immigration_proportion']
    ]
    for _, row in increased.iterrows():
        print(f"  {row['speaker_name']:<30} | Raw: {row['reform_alignment_raw']:.3f} -> Norm: {row['reform_alignment_normalized']:.3f} | "
              f"Change: {row['score_change']:+.3f} | Imm %: {row['immigration_proportion']:.1%}")

    # Spot check known MPs
    print("\n" + "=" * 80)
    print("SPOT CHECKS (Known MPs)")
    print("=" * 80)

    known_mps = ['Rishi Sunak', 'Suella Braverman', 'Laura Trott', 'Robert Jenrick']
    for mp_name in known_mps:
        mp_data = normalized_df[normalized_df['speaker_name'] == mp_name]
        if len(mp_data) > 0:
            row = mp_data.iloc[0]
            print(f"\n{mp_name}:")
            print(f"  Total speeches: {row['total_speeches']:.0f}")
            print(f"  Immigration proportion: {row['immigration_proportion']:.1%}")
            print(f"  Raw Reform alignment: {row['reform_alignment_raw']:.3f}")
            print(f"  Normalized Reform alignment: {row['reform_alignment_normalized']:.3f}")
            print(f"  Change: {row['score_change']:+.3f}")
        else:
            print(f"\n{mp_name}: NOT FOUND")

    return normalized_df


if __name__ == "__main__":
    results = main()
