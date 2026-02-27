"""
Compile Defection Ground Truth from Best for Britain
====================================================

Creates a CSV of Conservative MPs who defected to Reform UK since Jan 1, 2024
using data previously fetched from Best for Britain tracker.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SOURCE_DATA = BASE_DIR / "source_data"
DEFECTION_TRACKER_DIR = SOURCE_DATA / "defection_tracker"
DEFECTION_TRACKER_DIR.mkdir(exist_ok=True)

#  =============================================================================
# DEFECTION DATA (from Best for Britain Google Sheets)
# =============================================================================

DEFECTIONS_2024 = [
    {"name": "Lee Anderson", "defection_date": "2024-03-11", "former_role": "MP (Chief Whip)", "notes": "Former Deputy Chairman"},
    {"name": "Lucy Allan", "defection_date": "2024-05-29", "former_role": "Former MP", "notes": "No government role listed"},
    {"name": "Mark Reckless", "defection_date": "2024-11-08", "former_role": "Former MP", "notes": "Previously served as MP"},
    {"name": "Andrea Jenkyns", "defection_date": "2024-11-28", "former_role": "Mayor of Lincolnshire", "notes": "Minister of State for Skills, Apprenticeships and Higher Education"},
    {"name": "Aidan Burley", "defection_date": "2024-12-10", "former_role": "Former MP", "notes": "No recent government position"},
    {"name": "Marco Longhi", "defection_date": "2025-01-03", "former_role": "Former MP", "notes": "No recent government position"},
    {"name": "Sarah Pochin", "defection_date": "2025-03-24", "former_role": "MP", "notes": "Former Councillor background"},
    {"name": "Alan Amos", "defection_date": "2025-04-03", "former_role": "Councillor", "notes": "Previously served as MP"},
    {"name": "Ross Thomson", "defection_date": "2025-06-24", "former_role": "Former MP", "notes": "No government role documented"},
    {"name": "Anne Marie Morris", "defection_date": "2025-07-02", "former_role": "Position Specialist", "notes": "Head of Social Care Policy background"},
    {"name": "David Jones", "defection_date": "2025-07-07", "former_role": "Ministerial Role", "notes": "Minister of State for Exiting the European Union"},
    {"name": "Jake Berry", "defection_date": "2025-07-09", "former_role": "Former Official", "notes": "Former Chairman and Former Minister of State for the Northern Powerhouse"},
    {"name": "Adam Holloway", "defection_date": "2025-07-30", "former_role": "Treasury Position", "notes": "Lord Commissioner of HM Treasury"},
    {"name": "Nadine Dorries", "defection_date": "2025-09-05", "former_role": "Former Secretary", "notes": "Secretary of State for Digital, Culture, Media and Sport"},
    {"name": "Danny Kruger", "defection_date": "2025-09-15", "former_role": "MP", "notes": "Political Secretary to the Prime Minister"},
    {"name": "Maria Caulfield", "defection_date": "2025-09-16", "former_role": "Healthcare Role", "notes": "Minister of State for Health"},
    {"name": "Sarah Atherton", "defection_date": "2025-10-02", "former_role": "Defence Position", "notes": "Parliamentary Under-Secretary of State for Defence"},
    {"name": "Chris Green", "defection_date": "2025-12-01", "former_role": "Former MP", "notes": "No specific role documented"},
    {"name": "Jonathan Gullis", "defection_date": "2025-12-01", "former_role": "Parliamentary Role", "notes": "Parliamentary Under-Secretary of State for School Standards"},
    {"name": "Lia Nici", "defection_date": "2025-12-04", "former_role": "Advisory Position", "notes": "Parliamentary Under-Secretary of State for Levelling Up"},
    {"name": "Ben Bradley", "defection_date": "2025-12-12", "former_role": "Advisor Role", "notes": "Position: To cut council spending"},
    {"name": "Nadhim Zahawi", "defection_date": "2026-01-12", "former_role": "Finance Role", "notes": "Chancellor of the Exchequer"},
    {"name": "Robert Jenrick", "defection_date": "2026-01-15", "former_role": "MP", "notes": "Minister of State for Health"},
]


# =============================================================================
# PARSE MINISTERIAL STATUS FROM ROLES
# =============================================================================

def parse_ministerial_rank(former_role, notes):
    """Determine ministerial rank from role description."""

    text = f"{former_role} {notes}".upper()

    if "SECRETARY OF STATE" in text or "CHANCELLOR" in text:
        return 4  # Cabinet
    if "MINISTER OF STATE" in text or "MINISTER FOR" in text:
        return 3  # Minister of State
    if "UNDER-SECRETARY" in text or "PARLIAMENTARY SECRETARY" in text:
        return 2  # Parliamentary Under-Secretary
    if "PPS" in text or "POLITICAL SECRETARY" in text or "ADVISOR" in text or "COMMISSIONER" in text:
        return 1  # PPS/Special Advisor level

    # Former MP with government role mentioned
    if "MINISTER" in text or "CHAIRMAN" in text or "CHAIR" in text:
        return 3

    return 0  # No ministerial role


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""

    print("=" * 80)
    print("COMPILING DEFECTION GROUND TRUTH DATA")
    print("=" * 80)
    print()

    # Convert to DataFrame
    defections_df = pd.DataFrame(DEFECTIONS_2024)
    defections_df['defected'] = 1
    defections_df['source'] = 'Best for Britain tracker'

    # Parse dates
    defections_df['defection_date'] = pd.to_datetime(defections_df['defection_date'])

    # Extract ministerial rank
    defections_df['ministerial_rank'] = defections_df.apply(
        lambda row: parse_ministerial_rank(row['former_role'], row['notes']),
        axis=1
    )

    defections_df['ever_minister'] = (defections_df['ministerial_rank'] > 0).astype(int)

    print(f"Compiled {len(defections_df)} defections since January 1, 2024")
    print(f"\nDefections by year:")
    defections_df['year'] = defections_df['defection_date'].dt.year
    print(defections_df['year'].value_counts().sort_index())

    print(f"\nDefections by ministerial status:")
    print(f"  Rank 4 (Cabinet): {(defections_df['ministerial_rank'] == 4).sum()}")
    print(f"  Rank 3 (Minister of State): {(defections_df['ministerial_rank'] == 3).sum()}")
    print(f"  Rank 2 (Under-Secretary): {(defections_df['ministerial_rank'] == 2).sum()}")
    print(f"  Rank 1 (PPS/Advisor): {(defections_df['ministerial_rank'] == 1).sum()}")
    print(f"  Rank 0 (No ministerial role): {(defections_df['ministerial_rank'] == 0).sum()}")

    # Save to CSV
    output_path = DEFECTION_TRACKER_DIR / "defections_2024.csv"
    defections_df.to_csv(output_path, index=False)
    print(f"\nSaved defection data to: {output_path}")

    # Print list of defectors
    print("\n" + "=" * 80)
    print("DEFECTOR LIST (Chronological)")
    print("=" * 80)

    for _, row in defections_df.sort_values('defection_date').iterrows():
        rank_label = {0: "Backbencher", 1: "PPS", 2: "Under-Sec", 3: "Minister", 4: "Cabinet"}[row['ministerial_rank']]
        print(f"{row['defection_date'].strftime('%Y-%m-%d')} | {row['name']:<25} | {rank_label:<12} | {row['former_role']}")

    # Notable defectors
    print("\n" + "=" * 80)
    print("NOTABLE CABINET/MINISTERIAL DEFECTORS")
    print("=" * 80)

    cabinet_defectors = defections_df[defections_df['ministerial_rank'] >= 3].sort_values('defection_date')
    for _, row in cabinet_defectors.iterrows():
        print(f"\n{row['name']} ({row['defection_date'].strftime('%Y-%m-%d')})")
        print(f"  Role: {row['former_role']}")
        print(f"  Notes: {row['notes']}")

    return defections_df


if __name__ == "__main__":
    results = main()
