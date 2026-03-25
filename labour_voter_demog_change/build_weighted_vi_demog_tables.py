from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import snowflake.connector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build weighted Labour vs Reform VI tables by survey and demographic group "
            "from SURVEYS_STONEHAVEN.PUBLIC.ALL_SURVEYS_V2."
        )
    )
    parser.add_argument(
        "--start-date",
        default="2024-07-01",
        help="Inclusive start date (YYYY-MM-DD). Default: 2024-07-01",
    )
    parser.add_argument(
        "--end-date",
        default="2026-03-31",
        help="Inclusive end date (YYYY-MM-DD). Default: 2026-03-31",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--env-file",
        default=str(
            Path(__file__).resolve().parents[2]
            / "economics-module"
            / "credentials"
            / "snowflake"
            / ".env"
        ),
        help=(
            "Path to Snowflake .env containing SNOWFLAKE_USER, SNOWFLAKE_ACCOUNT, "
            "SNOWFLAKE_PAT, SNOWFLAKE_ROLE."
        ),
    )
    parser.add_argument(
        "--warehouse",
        default="SNOWFLAKE_LEARNING_WH",
        help="Warehouse to use for query execution.",
    )
    parser.add_argument(
        "--adjust-age-with-ons",
        action="store_true",
        help=(
            "Apply within-age-band post-stratification using ONS single-year age shares "
            "(age_band only; other demographics unchanged)."
        ),
    )
    parser.add_argument(
        "--ons-age-shares-csv",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "ons_age_single_year_shares_2024_uk.csv"
        ),
        help=(
            "CSV with columns age,share for ONS single-year age shares. "
            "Default file in data/ is UK mid-2024 (ONS MYE2 Persons, K02000001)."
        ),
    )
    return parser.parse_args()


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        raise FileNotFoundError(f"Snowflake env file not found: {env_file}")

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        os.environ[key] = value


def require_env(keys: Iterable[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def connect() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        password=os.getenv("SNOWFLAKE_PAT"),
        authenticator="snowflake",
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


def common_cte(start_date: str, end_date: str) -> str:
    # Notes on extraction logic:
    # - We keep UK/GB surveys only, identified by the 3rd token in SURVEY_ID.
    # - We drop InterimData variants and dedupe GB_TEAL iterations to one latest variant.
    # - We select one VI question per survey (largest respondent count) that contains both Labour and Reform responses.
    # - We use question='weight' + question_type='weight' as respondent weights; missing/invalid weight -> 1.0.
    # - Demographics use strict question text patterns to avoid false positives from long prompt text.
    return f"""
WITH raw_base AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        SURVEY_DATE,
        QUESTION,
        QUESTION_TYPE,
        ENCODED_CHOICE,
        RESPONSE_TEXT,
        LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) AS response_text_l
    FROM SURVEYS_STONEHAVEN.PUBLIC.ALL_SURVEYS_V2
    WHERE SURVEY_DATE BETWEEN '{start_date}' AND '{end_date}'
      AND UPPER(SPLIT_PART(SURVEY_ID, '_', 3)) IN ('UK', 'GB')
),
survey_variants AS (
    SELECT
        SURVEY_ID,
        MIN(SURVEY_DATE) AS SURVEY_DATE,
        REGEXP_REPLACE(SURVEY_ID, '(_InterimData|_Final_MaxDiff|_Final)$', '') AS canonical_key,
        CASE
            WHEN LOWER(SURVEY_ID) LIKE '%_final_maxdiff' THEN 4
            WHEN LOWER(SURVEY_ID) LIKE '%_final' THEN 3
            WHEN LOWER(SURVEY_ID) LIKE '%interimdata%' THEN 1
            ELSE 2
        END AS variant_rank
    FROM raw_base
    GROUP BY 1
),
selected_survey_ids AS (
    SELECT SURVEY_ID
    FROM (
        SELECT
            SURVEY_ID,
            canonical_key,
            ROW_NUMBER() OVER (
                PARTITION BY canonical_key
                ORDER BY variant_rank DESC, SURVEY_ID DESC
            ) AS rn
        FROM survey_variants
        WHERE LOWER(SURVEY_ID) NOT LIKE '%interimdata%'
    )
    WHERE
        (canonical_key LIKE '%GB_TEAL%' AND rn = 1)
        OR (canonical_key NOT LIKE '%GB_TEAL%')
),
base AS (
    SELECT rb.*
    FROM raw_base rb
    INNER JOIN selected_survey_ids s
        ON rb.SURVEY_ID = s.SURVEY_ID
),
weights_raw AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        MAX(TRY_TO_DOUBLE(ENCODED_CHOICE)) AS weight_value
    FROM base
    WHERE QUESTION_TYPE = 'weight'
      AND LOWER(TRIM(QUESTION)) = 'weight'
    GROUP BY 1, 2
),
weights AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        COALESCE(NULLIF(weight_value, 0), 1.0) AS weight_value
    FROM weights_raw
),
vi_candidates AS (
    SELECT
        SURVEY_ID,
        SURVEY_DATE,
        RESPONDENT_ID,
        QUESTION,
        RESPONSE_TEXT,
        LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) AS response_text_l
    FROM base
    WHERE QUESTION_TYPE = 'single_choice'
      AND (
          LOWER(QUESTION) LIKE '%if there was a general election tomorrow - who would you vote for%'
          OR LOWER(QUESTION) LIKE '%if a general election were held tomorrow%which party would you vote for%'
          OR LOWER(QUESTION) LIKE '%how would you vote if a general election was called tomorrow%'
          OR LOWER(QUESTION) LIKE '%headline voting intention%'
          OR LOWER(QUESTION) LIKE '%which party would you vote for%'
      )
      AND response_text_l NOT IN ('', 'nan')
),
vi_question_stats AS (
    SELECT
        SURVEY_ID,
        MIN(SURVEY_DATE) AS SURVEY_DATE,
        QUESTION,
        COUNT(DISTINCT RESPONDENT_ID) AS n_resp,
        MAX(IFF(response_text_l LIKE '%labour%', 1, 0)) AS has_labour,
        MAX(IFF(response_text_l LIKE '%reform%', 1, 0)) AS has_reform
    FROM vi_candidates
    GROUP BY 1, 3
),
selected_vi_question AS (
    SELECT
        SURVEY_ID,
        SURVEY_DATE,
        QUESTION,
        n_resp,
        ROW_NUMBER() OVER (
            PARTITION BY SURVEY_ID
            ORDER BY n_resp DESC, QUESTION
        ) AS rn
    FROM vi_question_stats
    WHERE has_labour = 1
      AND has_reform = 1
),
vi_rows AS (
    SELECT
        c.SURVEY_ID,
        c.SURVEY_DATE,
        c.RESPONDENT_ID,
        q.QUESTION AS vi_question,
        c.RESPONSE_TEXT,
        CASE
            WHEN LOWER(c.RESPONSE_TEXT) LIKE '%labour%' THEN 'Labour'
            WHEN LOWER(c.RESPONSE_TEXT) LIKE '%reform%' THEN 'Reform'
            ELSE 'Other'
        END AS party_bucket
    FROM vi_candidates c
    INNER JOIN selected_vi_question q
        ON c.SURVEY_ID = q.SURVEY_ID
       AND c.QUESTION = q.QUESTION
       AND q.rn = 1
),
weighted_vi AS (
    SELECT
        v.SURVEY_ID,
        v.SURVEY_DATE,
        v.RESPONDENT_ID,
        v.vi_question,
        v.RESPONSE_TEXT AS vi_response,
        v.party_bucket,
        COALESCE(w.weight_value, 1.0) AS weight_value
    FROM vi_rows v
    LEFT JOIN weights w
        ON v.SURVEY_ID = w.SURVEY_ID
       AND v.RESPONDENT_ID = w.RESPONDENT_ID
),
age_demog_prepped AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        QUESTION,
        RESPONSE_TEXT,
        LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) AS response_text_l,
        REPLACE(LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))), ' ', '') AS response_text_compact,
        COALESCE(
            TRY_TO_NUMBER(REPLACE(LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))), ' ', '')),
            TRY_TO_NUMBER(SPLIT_PART(REPLACE(LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))), ' ', ''), '-', 1)),
            TRY_TO_NUMBER(REPLACE(REPLACE(LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))), ' ', ''), '+', ''))
        ) AS age_num
    FROM base
    WHERE QUESTION_TYPE IN ('single_choice', 'free_text', 'scale')
      AND LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) NOT IN ('', 'nan')
      AND (
          LOWER(TRIM(QUESTION)) IN ('age', 'age group', 'what is your age?')
          OR LOWER(QUESTION) LIKE 'which age group%'
          OR LOWER(QUESTION) LIKE 'what is your age%'
      )
),
age_demog AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        age_num,
        CASE
            WHEN age_num < 18 THEN 'Under 18'
            WHEN age_num BETWEEN 18 AND 24 THEN '18-24'
            WHEN age_num BETWEEN 25 AND 34 THEN '25-34'
            WHEN age_num BETWEEN 35 AND 44 THEN '35-44'
            WHEN age_num BETWEEN 45 AND 54 THEN '45-54'
            WHEN age_num BETWEEN 55 AND 64 THEN '55-64'
            WHEN age_num >= 65 THEN '65+'
            WHEN response_text_compact LIKE '14-17%' THEN 'Under 18'
            WHEN response_text_compact LIKE '15-17%' THEN 'Under 18'
            WHEN response_text_compact LIKE '16-17%' THEN 'Under 18'
            WHEN response_text_compact LIKE '18-24%' THEN '18-24'
            WHEN response_text_compact LIKE '25-34%' THEN '25-34'
            WHEN response_text_compact LIKE '35-44%' THEN '35-44'
            WHEN response_text_compact LIKE '45-54%' THEN '45-54'
            WHEN response_text_compact LIKE '55-64%' THEN '55-64'
            WHEN response_text_compact LIKE '65+%' THEN '65+'
            WHEN response_text_l LIKE '%65 and over%' THEN '65+'
            WHEN response_text_l LIKE '%65 or over%' THEN '65+'
            WHEN response_text_l LIKE '%75+%' THEN '65+'
            ELSE RESPONSE_TEXT
        END AS demog_value,
        ROW_NUMBER() OVER (
            PARTITION BY SURVEY_ID, RESPONDENT_ID
            ORDER BY
                CASE
                    WHEN LOWER(TRIM(QUESTION)) IN ('age', 'age group', 'what is your age?') THEN 1
                    WHEN LOWER(QUESTION) LIKE 'which age group%' THEN 2
                    WHEN LOWER(QUESTION) LIKE 'what is your age%' THEN 3
                    ELSE 99
                END,
                QUESTION
        ) AS rn
    FROM age_demog_prepped
),
education_demog AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        RESPONSE_TEXT AS demog_value,
        ROW_NUMBER() OVER (
            PARTITION BY SURVEY_ID, RESPONDENT_ID
            ORDER BY
                CASE
                    WHEN LOWER(TRIM(QUESTION)) = 'education' THEN 1
                    WHEN LOWER(QUESTION) LIKE '%highest level of education%' THEN 2
                    WHEN LOWER(QUESTION) LIKE '%what is your highest level of education%' THEN 3
                    ELSE 99
                END,
                QUESTION
        ) AS rn
    FROM base
    WHERE QUESTION_TYPE = 'single_choice'
      AND LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) NOT IN ('', 'nan')
      AND (
          LOWER(TRIM(QUESTION)) = 'education'
          OR LOWER(QUESTION) LIKE '%highest level of education%'
          OR LOWER(QUESTION) LIKE '%what is your highest level of education%'
      )
),
gender_demog AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        RESPONSE_TEXT AS demog_value,
        ROW_NUMBER() OVER (
            PARTITION BY SURVEY_ID, RESPONDENT_ID
            ORDER BY
                CASE
                    WHEN LOWER(TRIM(QUESTION)) IN ('gender', 'sex', 'what is your gender?') THEN 1
                    WHEN LOWER(QUESTION) LIKE 'what is your gender%' THEN 2
                    ELSE 99
                END,
                QUESTION
        ) AS rn
    FROM base
    WHERE QUESTION_TYPE = 'single_choice'
      AND LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) NOT IN ('', 'nan')
      AND (
          LOWER(TRIM(QUESTION)) IN ('gender', 'sex', 'what is your gender?')
          OR LOWER(QUESTION) LIKE 'what is your gender%'
      )
),
ethnicity_demog AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        RESPONSE_TEXT AS demog_value,
        ROW_NUMBER() OVER (
            PARTITION BY SURVEY_ID, RESPONDENT_ID
            ORDER BY
                CASE
                    WHEN LOWER(TRIM(QUESTION)) IN ('ethnicity', 'ethnic group') THEN 1
                    WHEN LOWER(QUESTION) LIKE '%ethnicity%' THEN 2
                    ELSE 99
                END,
                QUESTION
        ) AS rn
    FROM base
    WHERE QUESTION_TYPE = 'single_choice'
      AND LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) NOT IN ('', 'nan')
      AND (
          LOWER(TRIM(QUESTION)) IN ('ethnicity', 'ethnic group')
          OR LOWER(QUESTION) LIKE 'what is your ethnic group%'
          OR LOWER(QUESTION) LIKE '%ethnicity%'
      )
),
past_vote_demog AS (
    SELECT
        SURVEY_ID,
        RESPONDENT_ID,
        RESPONSE_TEXT AS demog_value,
        ROW_NUMBER() OVER (
            PARTITION BY SURVEY_ID, RESPONDENT_ID
            ORDER BY
                CASE
                    WHEN LOWER(QUESTION) LIKE 'how did you vote in the uk 2024 general election%' THEN 1
                    WHEN LOWER(QUESTION) LIKE 'how did you vote in the 2024 general election%' THEN 2
                    WHEN LOWER(QUESTION) LIKE 'how did you vote in the most recent general election in 2024%' THEN 3
                    WHEN LOWER(QUESTION) LIKE 'who did you vote for in the 2024 general election%' THEN 4
                    WHEN LOWER(QUESTION) LIKE 'who did you vote for in the 2019 general election%' THEN 5
                    WHEN LOWER(QUESTION) LIKE 'past vote%' THEN 6
                    ELSE 99
                END,
                QUESTION
        ) AS rn
    FROM base
    WHERE QUESTION_TYPE = 'single_choice'
      AND LOWER(TRIM(COALESCE(RESPONSE_TEXT, ''))) NOT IN ('', 'nan')
      AND (
          LOWER(QUESTION) LIKE 'how did you vote in the uk 2024 general election%'
          OR LOWER(QUESTION) LIKE 'how did you vote in the 2024 general election%'
          OR LOWER(QUESTION) LIKE 'how did you vote in the most recent general election in 2024%'
          OR LOWER(QUESTION) LIKE 'who did you vote for in the 2024 general election%'
          OR LOWER(QUESTION) LIKE 'who did you vote for in the 2019 general election%'
          OR LOWER(QUESTION) LIKE 'past vote%'
      )
),
demogs AS (
    SELECT SURVEY_ID, RESPONDENT_ID, 'age_band' AS demographic_type, demog_value, age_num
    FROM age_demog
    WHERE rn = 1

    UNION ALL

    SELECT SURVEY_ID, RESPONDENT_ID, 'education_level' AS demographic_type, demog_value, NULL AS age_num
    FROM education_demog
    WHERE rn = 1

    UNION ALL

    SELECT SURVEY_ID, RESPONDENT_ID, 'gender' AS demographic_type, demog_value, NULL AS age_num
    FROM gender_demog
    WHERE rn = 1

    UNION ALL

    SELECT SURVEY_ID, RESPONDENT_ID, 'ethnicity' AS demographic_type, demog_value, NULL AS age_num
    FROM ethnicity_demog
    WHERE rn = 1

    UNION ALL

    SELECT SURVEY_ID, RESPONDENT_ID, 'past_vote' AS demographic_type, demog_value, NULL AS age_num
    FROM past_vote_demog
    WHERE rn = 1
)
"""


def fetch_pandas(cur: snowflake.connector.cursor.SnowflakeCursor, sql: str) -> pd.DataFrame:
    cur.execute(sql)
    return cur.fetch_pandas_all()


def age_to_band(age: int) -> str:
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def load_ons_age_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ONS age shares file not found: {path}")
    target = pd.read_csv(path)
    required = {"age", "share"}
    if not required.issubset(set(target.columns)):
        raise ValueError(f"ONS age shares CSV must contain columns: {sorted(required)}")
    target = target.copy()
    target["age"] = pd.to_numeric(target["age"], errors="coerce")
    target["share"] = pd.to_numeric(target["share"], errors="coerce")
    target = target.dropna(subset=["age", "share"])
    target["age"] = target["age"].astype(int)
    target = target[target["age"] >= 0]
    target["age_key"] = np.where(target["age"] >= 90, 90, target["age"])
    target["age_band"] = target["age"].apply(age_to_band)
    target = (
        target.groupby(["age_key", "age_band"], as_index=False)["share"]
        .sum()
        .sort_values(["age_band", "age_key"])
    )
    return target


def normalize_demographic_value(demog_type: str, value: object) -> object:
    if value is None:
        return value
    text = str(value).strip()
    if text == "":
        return text
    low = text.lower()
    low_norm = low.replace("’", "'")

    if demog_type == "past_vote":
        if low_norm in {"conservative", "conservatives", "conservative party"}:
            return "Conservative"
        if low_norm in {"labour", "labour party"}:
            return "Labour"
        if low_norm in {"lib dem", "lib dems", "liberal democrat", "liberal democrats"}:
            return "Liberal Democrats"
        if low_norm in {"green", "green party"}:
            return "Green Party"
        if low_norm in {"reform", "reform uk"}:
            return "Reform UK"
        if low_norm in {"snp", "scottish national party"}:
            return "Scottish National Party (SNP)"
        if low_norm in {"plaid", "plaid cymru"}:
            return "Plaid Cymru"
        if (
            low_norm in {"did not vote", "didn't vote", "did not vote / not eligible", "non-voter"}
            or "didn't vote" in low_norm
            or "did not vote" in low_norm
        ):
            return "Did not vote"
        if low_norm in {"can't remember", "cannot remember", "dont know", "don't know"}:
            return "Can't remember"
        if low_norm.startswith("other"):
            return "Other"

    if demog_type == "ethnicity":
        if "white" in low:
            return "White"
        if "black" in low:
            return "Black"
        if "asian" in low:
            return "Asian"
        if "mixed" in low or "multiple" in low:
            return "Mixed / Multiple ethnic groups"
        if "arab" in low:
            return "Arab"
        if "other" in low:
            return "Other ethnic group"
        if "prefer not to say" in low:
            return "Prefer not to say"

    if demog_type == "gender":
        if low in {"male", "man"}:
            return "Male"
        if low in {"female", "woman"}:
            return "Female"
        if low in {"prefer not to say"}:
            return "Prefer not to say"

    return text


def normalize_demographic_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DEMOGRAPHIC_VALUE"] = [
        normalize_demographic_value(str(t), v)
        for t, v in zip(out["DEMOGRAPHIC_TYPE"].astype(str), out["DEMOGRAPHIC_VALUE"])
    ]
    return out


def apply_ons_age_adjustment(df: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["adjusted_weight"] = out["WEIGHT_VALUE"].astype(float)
    out["age_num_clean"] = pd.to_numeric(out["AGE_NUM"], errors="coerce")
    out["age_key"] = np.where(out["age_num_clean"] >= 90, 90, out["age_num_clean"])
    out["age_key"] = out["age_key"].where(out["age_key"].notna(), np.nan)

    age_mask = out["DEMOGRAPHIC_TYPE"] == "age_band"
    age_df = out[age_mask].copy()
    if age_df.empty:
        return out

    target_by_band = {
        b: g.set_index("age_key")["share"] for b, g in target.groupby("age_band")
    }

    for (survey_id, band), grp in age_df.groupby(["SURVEY_ID", "DEMOGRAPHIC_VALUE"]):
        if band not in target_by_band:
            continue
        valid = grp[grp["age_key"].notna()].copy()
        if valid.empty:
            continue
        sample_w = valid.groupby("age_key")["WEIGHT_VALUE"].sum()
        sample_total = sample_w.sum()
        if sample_total <= 0:
            continue
        sample_share = sample_w / sample_total

        target_share = target_by_band[band].reindex(sample_share.index).fillna(0.0)
        if target_share.sum() <= 0:
            continue
        target_share = target_share / target_share.sum()

        factor = target_share / sample_share
        idx = valid.index
        out.loc[idx, "adjusted_weight"] = (
            out.loc[idx, "WEIGHT_VALUE"].astype(float)
            * out.loc[idx, "age_key"].map(factor).astype(float)
        )

    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(Path(args.env_file).resolve())
    require_env(["SNOWFLAKE_USER", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PAT", "SNOWFLAKE_ROLE"])

    cte = common_cte(args.start_date, args.end_date)

    demog_rows_sql = f"""
{cte}
SELECT
    w.SURVEY_ID,
    TO_VARCHAR(w.SURVEY_DATE, 'YYYY-MM-DD') AS SURVEY_DATE,
    w.vi_question,
    d.demographic_type,
    d.demog_value AS demographic_value,
    d.age_num,
    w.RESPONDENT_ID,
    w.party_bucket,
    w.weight_value
FROM weighted_vi w
INNER JOIN demogs d
    ON w.SURVEY_ID = d.SURVEY_ID
   AND w.RESPONDENT_ID = d.RESPONDENT_ID
ORDER BY SURVEY_DATE, SURVEY_ID, demographic_type, demographic_value, RESPONDENT_ID
"""

    overall_sql = f"""
{cte}
SELECT
    SURVEY_ID,
    TO_VARCHAR(SURVEY_DATE, 'YYYY-MM-DD') AS SURVEY_DATE,
    vi_question,
    COUNT(DISTINCT RESPONDENT_ID) AS respondent_n,
    SUM(weight_value) AS weighted_base,
    SUM(IFF(party_bucket = 'Labour', weight_value, 0)) AS weighted_labour,
    SUM(IFF(party_bucket = 'Reform', weight_value, 0)) AS weighted_reform,
    100.0 * SUM(IFF(party_bucket = 'Labour', weight_value, 0)) / NULLIF(SUM(weight_value), 0) AS labour_vi_pct,
    100.0 * SUM(IFF(party_bucket = 'Reform', weight_value, 0)) / NULLIF(SUM(weight_value), 0) AS reform_vi_pct,
    100.0 * (
        SUM(IFF(party_bucket = 'Labour', weight_value, 0))
        - SUM(IFF(party_bucket = 'Reform', weight_value, 0))
    ) / NULLIF(SUM(weight_value), 0) AS labour_minus_reform_lead_pct
FROM weighted_vi
GROUP BY 1, 2, 3
ORDER BY SURVEY_DATE, SURVEY_ID
"""

    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute(f"USE WAREHOUSE {args.warehouse}")

        demog_rows_df = fetch_pandas(cur, demog_rows_sql)
        overall_df = fetch_pandas(cur, overall_sql)

    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    if args.adjust_age_with_ons:
        ons_targets = load_ons_age_targets(Path(args.ons_age_shares_csv).resolve())
        demog_rows_df = apply_ons_age_adjustment(demog_rows_df, ons_targets)
        demog_rows_df["WEIGHTING_METHOD"] = "survey_weight_with_ons_age_within_band_adjustment"
    else:
        demog_rows_df["adjusted_weight"] = demog_rows_df["WEIGHT_VALUE"].astype(float)
        demog_rows_df["WEIGHTING_METHOD"] = "survey_weight_only"

    demog_rows_df = normalize_demographic_values(demog_rows_df)

    group_cols = ["SURVEY_ID", "SURVEY_DATE", "VI_QUESTION", "DEMOGRAPHIC_TYPE", "DEMOGRAPHIC_VALUE"]
    demog_df = (
        demog_rows_df.groupby(group_cols, as_index=False)
        .agg(
            RESPONDENT_N=("RESPONDENT_ID", "nunique"),
            WEIGHTED_BASE=("adjusted_weight", "sum"),
            WEIGHTED_LABOUR=("adjusted_weight", lambda s: s[demog_rows_df.loc[s.index, "PARTY_BUCKET"] == "Labour"].sum()),
            WEIGHTED_REFORM=("adjusted_weight", lambda s: s[demog_rows_df.loc[s.index, "PARTY_BUCKET"] == "Reform"].sum()),
        )
    )
    demog_df["LABOUR_VI_PCT"] = 100.0 * demog_df["WEIGHTED_LABOUR"] / demog_df["WEIGHTED_BASE"].replace(0, np.nan)
    demog_df["REFORM_VI_PCT"] = 100.0 * demog_df["WEIGHTED_REFORM"] / demog_df["WEIGHTED_BASE"].replace(0, np.nan)
    demog_df["LABOUR_MINUS_REFORM_LEAD_PCT"] = demog_df["LABOUR_VI_PCT"] - demog_df["REFORM_VI_PCT"]
    demog_df["WEIGHTING_METHOD"] = demog_rows_df["WEIGHTING_METHOD"].iloc[0]
    demog_df = demog_df.sort_values(group_cols).reset_index(drop=True)

    demog_out = out_dir / "weighted_vi_by_demographic.csv"
    overall_out = out_dir / "survey_level_vi_summary.csv"

    demog_df.to_csv(demog_out, index=False)
    overall_df.to_csv(overall_out, index=False)

    print("Done.")
    print(f"Demographic table: {demog_out}")
    print(f"Survey summary:    {overall_out}")
    print(f"Rows (demographic): {len(demog_df):,}")
    print(f"Rows (survey):      {len(overall_df):,}")


if __name__ == "__main__":
    main()
