"""
Rebuild all_speeches_extended.csv from Snowflake.

Source table:
  GOOGLE_AI_EIR.SPEECH_ANALYSIS.HANSARD

Rows used:
  INGEST_TYPE = 'speech_row'

Run with:
  conda run -n snowflake_env python tory_defection/source_data/hansard/build_all_speeches_extended_from_snowflake.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATABASE = "GOOGLE_AI_EIR"
SCHEMA = "SPEECH_ANALYSIS"
TABLE = "HANSARD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all_speeches_extended.csv from Snowflake speech_row data.")
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parent / "all_speeches_extended.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=50_000,
        help="Rows fetched per batch from Snowflake cursor.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_csv).resolve()

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Re-run with --overwrite.")

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
        cursor.execute(
            f"""
            SELECT
                PERSON_ID,
                SPEAKER_NAME,
                TEXT,
                TO_VARCHAR(SPEECH_DATE, 'YYYY-MM-DD') AS SPEECH_DATE,
                SPEECH_ID
            FROM {TABLE}
            WHERE INGEST_TYPE = 'speech_row'
            ORDER BY SPEECH_DATE, SPEECH_ID
            """
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person_id", "speaker_name", "text", "date", "speech_id"])

            while True:
                rows = cursor.fetchmany(args.fetch_size)
                if not rows:
                    break

                writer.writerows(rows)
                total += len(rows)
                print(f"Wrote {len(rows)} rows (total {total})")

        print("\nBuild complete.")
        print(f"  Output: {out_path}")
        print(f"  Rows:   {total}")
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
