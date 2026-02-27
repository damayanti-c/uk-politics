"""
Desk research variables: faction membership for Conservative MPs
=============================================

This module defines Conservative Party faction memberships based on secondary research.
These memberships are used to create addiitional scoring variables for the defection model.

RESEARCH SOURCES:
-----------------

1. European Research Group (ERG):
   - Wikipedia: https://en.wikipedia.org/wiki/European_Research_Group
   - Led by Mark Francois (chair since 2018)
   - Former chairs: Suella Braverman (2017-2018), Jacob Rees-Mogg (2016-2017)
   - Key members include Bernard Jenkin (founder), Priti Patel, Iain Duncan Smith

2. Common Sense Group (CSG):
   - Wikipedia: https://en.wikipedia.org/wiki/Common_Sense_Group
   - Founded 2020, chaired by Sir John Hayes
   - President: Sir Edward Leigh
   - Peak membership ~60 MPs
   - Suella Braverman described as "closely linked" (The Times, 2023)
   - John Hayes described as Braverman's "secret adviser" (New Statesman)

3. New Conservatives:
   - Wikipedia: https://en.wikipedia.org/wiki/New_Conservatives_(UK)
   - Founded by Danny Kruger and Miriam Cates (2022)
   - ~25 members at peak
   - Many lost seats in 2024 election
   - Danny Kruger defected to Reform UK (December 2024)
   - Known members: Gareth Bacon, Nick Fletcher (lost seat)

4. Rwanda Bill Rebels (January 2024):
   - The Spectator: https://www.spectator.co.uk/article/only-11-tories-vote-against-rwanda-bill/
   - Conservative Home: https://conservativehome.com/2024/01/18/the-eleven-conservative-mps-who-voted-against-the-third-reading-of-the-rwanda-bill/
   - 11 MPs voted against third reading as "too weak":
     Suella Braverman, Robert Jenrick, Miriam Cates (lost seat), Danny Kruger,
     Sir William Cash (retired), Sir Simon Clarke (lost seat), Mark Francois,
     David Jones (lost seat), Sarah Dines (lost seat), Andrea Jenkyns (lost seat)
   - Note: Most rebels lost seats or have left; only a few remain as sitting MPs

5. "Five Families" Coalition:
   - Politics.co.uk: https://www.politics.co.uk/5-minute-read/2023/07/03/where-do-the-new-conservatives-fit-in-rishi-sunaks-crowded-factional-field/
   - Five right-wing factions: ERG (Francois), CSG (Braverman/Hayes),
     New Conservatives (Kruger), NRG (Jake Berry - lost seat),
     Conservative Growth Group (Simon Clarke - lost seat)

6. One Nation Conservatives:
   - Wikipedia: https://en.wikipedia.org/wiki/One_Nation_Conservatives_(caucus)
   - Chaired by Damian Green (lost seat 2024)
   - ~106 members at peak
   - Associated with moderates: Tom Tugendhat, James Cleverly, Jeremy Hunt

7. Tory Reform Group (TRG):
   - Wikipedia: https://en.wikipedia.org/wiki/Tory_Reform_Group
   - Vice-Presidents: Damian Green (lost seat), Sir Robert Buckland (lost seat)
   - Patron: Richard Fuller
   - Moderate/centrist faction

METHODOLOGY:
------------
- Right-wing faction membership is scored additively (ERG + CSG + New Cons + Rwanda rebel)
- Moderate faction membership provides negative score (One Nation + TRG)
- This creates a natural separation that reflects the ideological spectrum
- Party leaders (Badenoch, Sunak) receive hardcoded penalties (justified as party leaders)
"""

# =============================================================================
# RIGHT-WING FACTION MEMBERSHIPS
# =============================================================================

# European Research Group (ERG) - Hard Brexiteers
# Source: https://en.wikipedia.org/wiki/European_Research_Group
# Additional: https://www.theguardian.com/politics/european-research-group-erg
ERG_MEMBERS = {
    'Mark Francois',       # Chair since 2018 - https://en.wikipedia.org/wiki/Mark_Francois
    'Suella Braverman',    # Former chair 2017-2018 - https://en.wikipedia.org/wiki/Suella_Braverman
    'Bernard Jenkin',      # Founder member - https://en.wikipedia.org/wiki/Bernard_Jenkin
    'Priti Patel',         # Known ERG member - https://en.wikipedia.org/wiki/Priti_Patel
    'Iain Duncan Smith',   # Former party leader, prominent Brexiteer - https://en.wikipedia.org/wiki/Iain_Duncan_Smith
    'John Hayes',          # ERG member, CSG chair - https://en.wikipedia.org/wiki/John_Hayes_(British_politician)
    'Edward Leigh',        # Long-standing ERG member - https://en.wikipedia.org/wiki/Edward_Leigh
    'Christopher Chope',   # ERG stalwart since 1990s - https://en.wikipedia.org/wiki/Christopher_Chope
    'Desmond Swayne',      # ERG member - https://en.wikipedia.org/wiki/Desmond_Swayne
    'Andrew Rosindell',    # ERG member, Eurosceptic - https://en.wikipedia.org/wiki/Andrew_Rosindell
    'John Whittingdale',   # ERG member - https://en.wikipedia.org/wiki/John_Whittingdale
    'David Davis',         # Former Brexit Secretary, ERG aligned - https://en.wikipedia.org/wiki/David_Davis_(British_politician)
    'Robert Jenrick',      # Associated with ERG right wing - https://en.wikipedia.org/wiki/Robert_Jenrick
    'Esther McVey',        # ERG member - https://en.wikipedia.org/wiki/Esther_McVey
}

# Common Sense Group (CSG) - Social conservatives
# Source: https://en.wikipedia.org/wiki/Common_Sense_Group
# Additional: https://www.thetimes.co.uk/article/common-sense-group-tory-mps
# New Statesman: https://www.newstatesman.com/politics/conservatives/2022/10/suella-braverman-john-hayes
COMMON_SENSE_GROUP_MEMBERS = {
    'John Hayes',          # Chair - https://en.wikipedia.org/wiki/John_Hayes_(British_politician)
    'Edward Leigh',        # President - https://en.wikipedia.org/wiki/Edward_Leigh
    'Suella Braverman',    # "Closely linked" per The Times - https://www.thetimes.co.uk/article/suella-braverman
    'Esther McVey',        # Known CSG member - https://en.wikipedia.org/wiki/Esther_McVey
    'Nick Timothy',        # Social conservative, CSG aligned - https://en.wikipedia.org/wiki/Nick_Timothy
    'Desmond Swayne',      # CSG member - https://en.wikipedia.org/wiki/Desmond_Swayne
    'Mark Francois',       # Associated via "Five Families" - https://www.politics.co.uk/reference/five-families/
    'Andrew Rosindell',    # Social conservative wing - https://en.wikipedia.org/wiki/Andrew_Rosindell
    'Priti Patel',         # Associated with CSG positions - https://en.wikipedia.org/wiki/Priti_Patel
}

# New Conservatives - Post-2022 right-wing grouping
# Source: https://en.wikipedia.org/wiki/New_Conservatives_(UK)
# Founded by Danny Kruger and Miriam Cates; most lost seats in 2024
NEW_CONSERVATIVES_MEMBERS = {
    'Danny Kruger',        # Co-founder, defected to Reform Dec 2024 - https://en.wikipedia.org/wiki/Danny_Kruger
    'Gareth Bacon',        # Known New Cons member - https://en.wikipedia.org/wiki/Gareth_Bacon
    'Nick Timothy',        # Associated, wrote manifesto for New Cons - https://en.wikipedia.org/wiki/Nick_Timothy
    'Katie Lam',           # New MP, New Cons aligned positions - https://members.parliament.uk/member/5017
    'Bradley Thomas',      # New MP, right-wing positions - https://members.parliament.uk/member/5009
    'Lewis Cocking',       # New MP, right-wing positions - https://members.parliament.uk/member/4998
    'Matt Vickers',        # Red Wall MP, right-wing on immigration - https://en.wikipedia.org/wiki/Matt_Vickers
}

# Rwanda Bill Rebels (January 2024) - Voted against as "too weak"
# Source: https://www.spectator.co.uk/article/only-11-tories-vote-against-rwanda-bill/
# Source: https://conservativehome.com/2024/01/18/the-eleven-conservative-mps-who-voted-against-the-third-reading-of-the-rwanda-bill/
# Only includes sitting MPs (many lost seats or retired)
RWANDA_REBELS = {
    'Suella Braverman',    # Voted against - https://www.spectator.co.uk/article/only-11-tories-vote-against-rwanda-bill/
    'Robert Jenrick',      # Voted against, resigned as minister over Rwanda - https://en.wikipedia.org/wiki/Robert_Jenrick
    'Danny Kruger',        # Voted against - https://www.spectator.co.uk/article/only-11-tories-vote-against-rwanda-bill/
    'Mark Francois',       # Voted against - https://www.spectator.co.uk/article/only-11-tories-vote-against-rwanda-bill/
    # Note: William Cash (retired), Simon Clarke (lost), Miriam Cates (lost),
    # David Jones (lost), Sarah Dines (lost), Andrea Jenkyns (lost)
}

# =============================================================================
# MODERATE FACTION MEMBERSHIPS
# =============================================================================

# One Nation Conservatives - Centrist/moderate faction
# Source: https://en.wikipedia.org/wiki/One_Nation_Conservatives_(caucus)
# Damian Green (chair) lost seat in 2024
ONE_NATION_MEMBERS = {
    'Tom Tugendhat',       # Prominent moderate, leadership candidate - https://en.wikipedia.org/wiki/Tom_Tugendhat
    'James Cleverly',      # Moderate, leadership candidate 2024 - https://en.wikipedia.org/wiki/James_Cleverly
    'Jeremy Hunt',         # Moderate, former Chancellor - https://en.wikipedia.org/wiki/Jeremy_Hunt
    'Damian Hinds',        # One Nation associated - https://en.wikipedia.org/wiki/Damian_Hinds
    'Victoria Atkins',     # Moderate, Health Secretary - https://en.wikipedia.org/wiki/Victoria_Atkins
    'Stuart Andrew',       # One Nation, LGBTQ+ advocate - https://en.wikipedia.org/wiki/Stuart_Andrew
    'Karen Bradley',       # One Nation associated - https://en.wikipedia.org/wiki/Karen_Bradley
    'Julian Smith',        # Moderate, former NI Secretary - https://en.wikipedia.org/wiki/Julian_Smith_(politician)
    'Caroline Nokes',      # One Nation, socially liberal - https://en.wikipedia.org/wiki/Caroline_Nokes
    'David Simmonds',      # Known moderate on immigration - https://en.wikipedia.org/wiki/David_Simmonds_(politician)
    'Alicia Kearns',       # Moderate positions - https://en.wikipedia.org/wiki/Alicia_Kearns
    'Andrew Mitchell',     # Centrist, internationalist - https://en.wikipedia.org/wiki/Andrew_Mitchell
    'Simon Hoare',         # One Nation, pro-EU - https://en.wikipedia.org/wiki/Simon_Hoare
    'Geoffrey Cox',        # Moderate, former AG - https://en.wikipedia.org/wiki/Geoffrey_Cox_(politician)
}

# Tory Reform Group (TRG) - Progressive conservatives
# Source: https://en.wikipedia.org/wiki/Tory_Reform_Group
# Note: Key figures like Robert Buckland lost seats
TRG_MEMBERS = {
    'Richard Fuller',      # TRG Patron - https://en.wikipedia.org/wiki/Richard_Fuller_(politician)
    'Caroline Nokes',      # Associated with TRG positions - https://en.wikipedia.org/wiki/Caroline_Nokes
    'David Simmonds',      # Progressive on social issues - https://en.wikipedia.org/wiki/David_Simmonds_(politician)
}

# =============================================================================
# PARTY LEADERS (Hardcoded penalty - justified as leadership constraint)
# =============================================================================

PARTY_LEADERS = {
    'Kemi Badenoch',       # Current party leader (elected Nov 2024) - https://en.wikipedia.org/wiki/Kemi_Badenoch
    'Rishi Sunak',         # Former PM, former party leader - https://en.wikipedia.org/wiki/Rishi_Sunak
}

# =============================================================================
# SCORING WEIGHTS
# =============================================================================

# Right-wing faction weights (positive = increases defection risk)
FACTION_WEIGHTS = {
    'ergs': 1.5,           # ERG membership
    'csg': 1.5,            # Common Sense Group membership
    'new_cons': 1.5,       # New Conservatives membership
    'rwanda_rebel': 2.0,   # Rwanda Bill rebel (strong signal)
    'one_nation': -1.5,    # One Nation membership (reduces risk)
    'trg': -1.0,           # TRG membership (reduces risk)
    'party_leader': -10.0, # Party leader penalty (very strong)
}


def get_faction_score(mp_name):
    """
    Calculate composite faction score for an MP.

    Positive score = right-wing/Reform-aligned (higher defection risk)
    Negative score = moderate/establishment (lower defection risk)

    Returns dict with individual faction memberships and composite score.
    """
    scores = {
        'is_erg': 1 if mp_name in ERG_MEMBERS else 0,
        'is_csg': 1 if mp_name in COMMON_SENSE_GROUP_MEMBERS else 0,
        'is_new_conservative': 1 if mp_name in NEW_CONSERVATIVES_MEMBERS else 0,
        'is_rwanda_rebel': 1 if mp_name in RWANDA_REBELS else 0,
        'is_one_nation': 1 if mp_name in ONE_NATION_MEMBERS else 0,
        'is_trg': 1 if mp_name in TRG_MEMBERS else 0,
        'is_party_leader': 1 if mp_name in PARTY_LEADERS else 0,
    }

    # Calculate composite score
    composite = (
        scores['is_erg'] * FACTION_WEIGHTS['ergs'] +
        scores['is_csg'] * FACTION_WEIGHTS['csg'] +
        scores['is_new_conservative'] * FACTION_WEIGHTS['new_cons'] +
        scores['is_rwanda_rebel'] * FACTION_WEIGHTS['rwanda_rebel'] +
        scores['is_one_nation'] * FACTION_WEIGHTS['one_nation'] +
        scores['is_trg'] * FACTION_WEIGHTS['trg'] +
        scores['is_party_leader'] * FACTION_WEIGHTS['party_leader']
    )

    scores['faction_composite_score'] = composite

    return scores


def get_all_faction_data(mp_names):
    """
    Get faction data for a list of MPs.

    Args:
        mp_names: List or Series of MP names

    Returns:
        DataFrame with faction membership columns
    """
    import pandas as pd

    records = []
    for name in mp_names:
        data = get_faction_score(name)
        data['name'] = name
        records.append(data)

    return pd.DataFrame(records)


# =============================================================================
# SUMMARY STATISTICS (for reference)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FACTION MEMBERSHIP DATA SUMMARY")
    print("=" * 70)

    print(f"\nRight-wing factions:")
    print(f"  ERG members: {len(ERG_MEMBERS)}")
    print(f"  Common Sense Group: {len(COMMON_SENSE_GROUP_MEMBERS)}")
    print(f"  New Conservatives: {len(NEW_CONSERVATIVES_MEMBERS)}")
    print(f"  Rwanda rebels (sitting): {len(RWANDA_REBELS)}")

    print(f"\nModerate factions:")
    print(f"  One Nation: {len(ONE_NATION_MEMBERS)}")
    print(f"  TRG: {len(TRG_MEMBERS)}")

    print(f"\nParty leaders: {len(PARTY_LEADERS)}")

    # Show overlap
    right_wing = ERG_MEMBERS | COMMON_SENSE_GROUP_MEMBERS | NEW_CONSERVATIVES_MEMBERS | RWANDA_REBELS
    moderate = ONE_NATION_MEMBERS | TRG_MEMBERS

    print(f"\nUnique right-wing MPs: {len(right_wing)}")
    print(f"Unique moderate MPs: {len(moderate)}")
    print(f"Overlap: {right_wing & moderate}")

    # Show sample scores
    print("\n" + "=" * 70)
    print("SAMPLE FACTION SCORES")
    print("=" * 70)

    test_mps = [
        'Suella Braverman', 'Mark Francois', 'Robert Jenrick',
        'Jeremy Hunt', 'Tom Tugendhat', 'James Cleverly',
        'Kemi Badenoch', 'Danny Kruger', 'Nick Timothy'
    ]

    for mp in test_mps:
        score = get_faction_score(mp)
        print(f"\n{mp}:")
        print(f"  ERG: {score['is_erg']}, CSG: {score['is_csg']}, NewCon: {score['is_new_conservative']}")
        print(f"  Rwanda: {score['is_rwanda_rebel']}, OneNation: {score['is_one_nation']}, TRG: {score['is_trg']}")
        print(f"  Leader: {score['is_party_leader']}")
        print(f"  COMPOSITE: {score['faction_composite_score']:.1f}")
