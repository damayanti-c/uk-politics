from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATABASE = "GOOGLE_AI_EIR"
SCHEMA = "SPEECH_ANALYSIS"
TABLE = "HANSARD"


def _source_mode() -> str:
    return os.getenv("HANSARD_SOURCE", "snowflake").strip().lower()


def _local_csv_path(default_path: Path) -> Path:
    override = os.getenv("HANSARD_DATA_PATH", "").strip()
    if override:
        return Path(override)
    return default_path


def load_hansard_dataframe(
    default_local_csv_path: Path,
    speaker_names: set[str] | None = None,
    for_analysis_only: bool = False,
) -> pd.DataFrame:
    """
    Load Hansard speeches from local CSV or Snowflake.

    Env controls:
      - HANSARD_SOURCE=local|snowflake  (default: snowflake)
      - HANSARD_DATA_PATH=/path/to/all_speeches_extended.csv  (optional local override)
    """
    mode = _source_mode()
    if mode == "snowflake":
        return _load_from_snowflake(speaker_names=speaker_names, for_analysis_only=for_analysis_only)
    return _load_from_local(default_local_csv_path, speaker_names=speaker_names, for_analysis_only=for_analysis_only)


def _load_from_local(
    default_local_csv_path: Path,
    speaker_names: set[str] | None,
    for_analysis_only: bool,
) -> pd.DataFrame:
    csv_path = _local_csv_path(default_local_csv_path)
    df = pd.read_csv(csv_path)
    if for_analysis_only and speaker_names:
        df = df[df["speaker_name"].isin(speaker_names)]
    return df


def _load_from_snowflake(
    speaker_names: set[str] | None,
    for_analysis_only: bool,
) -> pd.DataFrame:
    try:
        from snowflake_tools import connect
    except ImportError as exc:
        raise RuntimeError(
            "snowflake_tools not available. Run with the existing `snowflake_env` environment."
        ) from exc

    connection, cursor = connect()
    try:
        cursor.execute(f"USE DATABASE {DATABASE}")
        cursor.execute(f"USE SCHEMA {SCHEMA}")

        base_query = f"""
            SELECT
                PERSON_ID AS person_id,
                SPEAKER_NAME AS speaker_name,
                TEXT AS text,
                SPEECH_DATE AS date,
                SPEECH_ID AS speech_id
            FROM {TABLE}
            WHERE INGEST_TYPE = 'speech_row'
        """

        params: list[str] = []
        if for_analysis_only and speaker_names:
            names = sorted(speaker_names)
            placeholders = ", ".join(["%s"] * len(names))
            base_query += f" AND SPEAKER_NAME IN ({placeholders})"
            params.extend(names)

        cursor.execute(base_query, params if params else None)
        rows = cursor.fetchall()
        columns = [desc[0].lower() for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        return df
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
