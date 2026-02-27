"""
Upload Hansard speech data to Snowflake.

Target table:
  GOOGLE_AI_EIR.SPEECH_ANALYSIS.HANSARD

Run with:
  conda run -n snowflake_env python tory_defection/source_data/hansard/upload_hansard_to_snowflake.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from snowflake_tools import connect, write_pandas_add_rows
except ImportError as exc:
    print(f"ERROR: Could not import snowflake_tools: {exc}")
    print("Use the existing Snowflake environment, e.g. `conda run -n snowflake_env ...`.")
    raise SystemExit(1)


DATABASE = "GOOGLE_AI_EIR"
SCHEMA = "SPEECH_ANALYSIS"
TABLE = "HANSARD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Hansard XML + all_speeches_extended.csv to Snowflake.")
    parser.add_argument(
        "--tory-defection-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to tory_defection root.",
    )
    parser.add_argument("--csv-chunk-rows", type=int, default=50_000, help="Rows per CSV upload batch.")
    parser.add_argument(
        "--xml-chunk-chars",
        type=int,
        default=8_000_000,
        help="Max chars per XML chunk row to avoid oversized values.",
    )
    parser.add_argument("--xml-batch-rows", type=int, default=500, help="Rows per XML upload batch.")
    parser.add_argument(
        "--truncate-table",
        action="store_true",
        help="Truncate GOOGLE_AI_EIR.SPEECH_ANALYSIS.HANSARD before upload.",
    )
    return parser.parse_args()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _extract_date_from_filename(filename: str) -> pd.Timestamp | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), errors="coerce")
    except Exception:
        return None


def _ensure_schema_and_table(cursor) -> None:
    cursor.execute(f"USE DATABASE {DATABASE}")
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    cursor.execute(f"USE SCHEMA {SCHEMA}")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            INGEST_TYPE VARCHAR,
            SOURCE_FILE VARCHAR,
            SOURCE_PATH VARCHAR,
            PERSON_ID VARCHAR,
            SPEAKER_NAME VARCHAR,
            SPEECH_DATE DATE,
            SPEECH_ID VARCHAR,
            TEXT VARCHAR,
            CHUNK_INDEX NUMBER,
            CHUNK_TOTAL NUMBER,
            XML_RAW VARCHAR,
            LOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
    )


def _upload_df(connection, df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    return bool(
        write_pandas_add_rows(
            data_frame=df,
            table_name=TABLE,
            database=DATABASE,
            schema=SCHEMA,
            connection=connection,
            auto_upper_case_cols=True,
        )
    )


def upload_csv_rows(connection, csv_path: Path, chunk_rows: int) -> int:
    print(f"Uploading CSV rows from: {csv_path}")
    total_rows = 0
    for idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_rows), start=1):
        batch = pd.DataFrame(
            {
                "ingest_type": "speech_row",
                "source_file": csv_path.name,
                "source_path": str(csv_path).replace("\\", "/"),
                "person_id": chunk.get("person_id"),
                "speaker_name": chunk.get("speaker_name"),
                "speech_date": pd.to_datetime(chunk.get("date"), errors="coerce").dt.date,
                "speech_id": chunk.get("speech_id"),
                "text": chunk.get("text"),
                "chunk_index": None,
                "chunk_total": None,
                "xml_raw": None,
            }
        )
        ok = _upload_df(connection, batch)
        if not ok:
            raise RuntimeError(f"CSV batch upload failed for chunk {idx}.")
        total_rows += len(batch)
        print(f"  CSV batch {idx}: uploaded {len(batch)} rows (total {total_rows})")
    return total_rows


def _flush_xml_rows(connection, rows: list[dict], batch_no: int) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    ok = _upload_df(connection, df)
    if not ok:
        raise RuntimeError(f"XML batch upload failed for batch {batch_no}.")
    print(f"  XML batch {batch_no}: uploaded {len(df)} rows")
    return len(df)


def upload_xml_files(
    connection,
    xml_files: Iterable[Path],
    max_chars: int,
    batch_rows: int,
) -> int:
    total_rows = 0
    batch: list[dict] = []
    batch_no = 1

    xml_list = list(xml_files)
    print(f"Uploading XML files: {len(xml_list)} files")

    for file_idx, xml_path in enumerate(xml_list, start=1):
        raw = xml_path.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(raw, max_chars)
        parsed_date = _extract_date_from_filename(xml_path.name)
        speech_date = parsed_date.date() if parsed_date is not None and not pd.isna(parsed_date) else None

        for chunk_idx, chunk in enumerate(chunks, start=1):
            batch.append(
                {
                    "ingest_type": "xml_chunk",
                    "source_file": xml_path.name,
                    "source_path": str(xml_path).replace("\\", "/"),
                    "person_id": None,
                    "speaker_name": None,
                    "speech_date": speech_date,
                    "speech_id": None,
                    "text": None,
                    "chunk_index": chunk_idx,
                    "chunk_total": len(chunks),
                    "xml_raw": chunk,
                }
            )
            if len(batch) >= batch_rows:
                total_rows += _flush_xml_rows(connection, batch, batch_no)
                batch.clear()
                batch_no += 1

        if file_idx % 200 == 0:
            print(f"  Processed {file_idx}/{len(xml_list)} XML files")

    if batch:
        total_rows += _flush_xml_rows(connection, batch, batch_no)
    return total_rows


def main() -> None:
    args = parse_args()
    root = Path(args.tory_defection_root).resolve()
    hansard_dir = root / "source_data" / "hansard"
    csv_path = hansard_dir / "all_speeches_extended.csv"
    xml_files = sorted(hansard_dir.glob("*.xml"))

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under: {hansard_dir}")

    print("Connecting to Snowflake...")
    connection, cursor = connect()
    try:
        _ensure_schema_and_table(cursor)

        if args.truncate_table:
            print(f"Truncating {DATABASE}.{SCHEMA}.{TABLE} ...")
            cursor.execute(f"TRUNCATE TABLE {DATABASE}.{SCHEMA}.{TABLE}")

        csv_rows = upload_csv_rows(connection, csv_path, args.csv_chunk_rows)
        xml_rows = upload_xml_files(connection, xml_files, args.xml_chunk_chars, args.xml_batch_rows)

        print("\nUpload complete.")
        print(f"  CSV speech rows: {csv_rows}")
        print(f"  XML chunk rows:  {xml_rows}")
        print(f"  Target table:    {DATABASE}.{SCHEMA}.{TABLE}")
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
