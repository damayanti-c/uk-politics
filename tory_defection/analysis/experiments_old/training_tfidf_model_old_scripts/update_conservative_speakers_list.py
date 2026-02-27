"""
Update Conservative Speakers List to Include Current MPs
=========================================================

Adds the 121 current Conservative MPs (elected July 2024) to the
conservative_speakers_in_hansard.csv list so they can be analyzed.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
CONSERVATIVE_SPEAKERS = BASE_DIR / "conservative_speakers_in_hansard.csv"
CURRENT_CONS = BASE_DIR / "current_conservative_mps_2024.csv"
HANSARD_CSV = BASE_DIR / "../../source_data/hansard/all_speeches.csv"

# Clean name function
def clean_name(name):
    if pd.isna(name):
        return name
    for title in ['Ms ', 'Mr ', 'Mrs ', 'Dr ', 'Sir ', 'Dame ', 'Lord ', 'Lady ', 'Rt Hon ']:
        name = name.replace(title, '')
    return name.strip()

print("=" * 80)
print("UPDATING CONSERVATIVE SPEAKERS LIST")
print("=" * 80)
print()

# Load existing list
existing = pd.read_csv(CONSERVATIVE_SPEAKERS)
print(f"Existing conservative speakers: {len(existing)}")

# Load current Conservative MPs
current_con = pd.read_csv(CURRENT_CONS)
current_con['name_clean'] = current_con['full_name'].apply(clean_name)
print(f"Current Conservative MPs (elected July 2024): {len(current_con)}")

# Load Hansard to verify they have speeches
hansard = pd.read_csv(HANSARD_CSV)
hansard_speakers = set(hansard['speaker_name'].unique())
print(f"Total unique speakers in Hansard: {len(hansard_speakers)}")

# Find current MPs not in existing list
existing['name_clean'] = existing['name'].apply(clean_name)
existing_names = set(existing['name_clean'])
current_names = set(current_con['name_clean'])

missing_from_list = current_names - existing_names
print(f"\nCurrent Conservative MPs missing from list: {len(missing_from_list)}")

# Check which of these missing MPs have Hansard speeches
new_to_add = []
for name in missing_from_list:
    # Find matching Hansard speaker name
    hansard_matches = [s for s in hansard_speakers if clean_name(s) == name]

    if hansard_matches:
        # Add to list
        hansard_name = hansard_matches[0]  # Use first match
        speech_count = (hansard['speaker_name'] == hansard_name).sum()
        new_to_add.append({
            'name': hansard_name,
            'has_hansard_speeches': True,
            'speech_count': speech_count
        })
        print(f"  Adding: {hansard_name:<40} ({speech_count} speeches)")

print(f"\nTotal new MPs to add: {len(new_to_add)}")

# Add to existing list
new_df = pd.DataFrame(new_to_add)
if len(new_df) > 0:
    # Append to existing
    updated = pd.concat([existing[['name', 'has_hansard_speeches']],
                         new_df[['name', 'has_hansard_speeches']]],
                        ignore_index=True)

    # Remove duplicates
    updated = updated.drop_duplicates(subset=['name'])

    # Save
    updated.to_csv(CONSERVATIVE_SPEAKERS, index=False)
    print(f"\nUpdated conservative_speakers_in_hansard.csv")
    print(f"  Previous: {len(existing)} MPs")
    print(f"  New: {len(updated)} MPs (+{len(updated) - len(existing)})")

    # Also save the new additions separately for reference
    new_additions_path = BASE_DIR / "newly_added_conservative_speakers.csv"
    new_df.to_csv(new_additions_path, index=False)
    print(f"  Saved new additions to: {new_additions_path}")

else:
    print("\nNo new MPs to add (all current MPs already in list)")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Re-run enhanced_speech_tfidf.py to generate speech analysis for new MPs")
print("2. Re-run normalize_speech_metrics.py")
print("3. Re-run apply_model_to_current_mps.py")
