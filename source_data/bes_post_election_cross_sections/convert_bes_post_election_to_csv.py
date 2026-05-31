from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pyreadstat


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_STEMS = {
    "bes_f2f_2015_v4.0": "bes_post_election_2015",
    "bes_f2f_2017_v1.5": "bes_post_election_2017",
    "bes_rps_2019_1.3.0": "bes_post_election_2019",
    "bes_rps_2024_1.0.1": "bes_post_election_2024",
}


def get_source_dir() -> Path:
    """Prefer raw inputs when present, but support the pre-raw layout as fallback."""
    if RAW_DIR.exists():
        return RAW_DIR
    return BASE_DIR


def extract_zip_archives(source_dir: Path) -> list[Path]:
    """Extract any zipped SPSS files in-place and return extracted .sav paths."""
    extracted_paths: list[Path] = []
    for zip_path in sorted(source_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if not member.filename.lower().endswith(".sav"):
                    continue
                target_path = source_dir / Path(member.filename).name
                if not target_path.exists():
                    with archive.open(member) as source, target_path.open("wb") as target:
                        target.write(source.read())
                extracted_paths.append(target_path)
    return extracted_paths


def build_codebook_frame(meta: pyreadstat._readstat_parser.metadata_container) -> pd.DataFrame:
    """Convert SPSS metadata into a flat, CSV-friendly codebook table."""
    rows: list[dict[str, str]] = []
    value_labels = meta.variable_value_labels or {}
    missing_ranges = meta.missing_ranges or {}
    missing_user_values = meta.missing_user_values or {}
    variable_measure = meta.variable_measure or {}
    original_types = meta.original_variable_types or {}

    for variable_name, variable_label in zip(meta.column_names, meta.column_labels):
        rows.append(
            {
                "variable_name": variable_name,
                "variable_label": variable_label or "",
                "measure": variable_measure.get(variable_name, ""),
                "original_type": original_types.get(variable_name, ""),
                "value_labels_json": json.dumps(
                    value_labels.get(variable_name, {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "missing_ranges_json": json.dumps(
                    missing_ranges.get(variable_name, []),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "missing_user_values_json": json.dumps(
                    missing_user_values.get(variable_name, []),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    return pd.DataFrame(rows)


def get_output_stem(sav_path: Path) -> str:
    """Return the cleaned output stem for a source SPSS filename."""
    return OUTPUT_STEMS.get(sav_path.stem, sav_path.stem)


def convert_sav_to_csv(sav_path: Path, output_dir: Path) -> tuple[int, int]:
    """Write a data CSV and metadata CSV for a single SPSS file."""
    data_frame, meta = pyreadstat.read_sav(
        sav_path,
        apply_value_formats=False,
        user_missing=True,
    )

    output_stem = get_output_stem(sav_path)
    csv_path = output_dir / f"{output_stem}.csv"
    codebook_path = output_dir / f"{output_stem}_codebook.csv"

    data_frame.to_csv(csv_path, index=False)
    build_codebook_frame(meta).to_csv(codebook_path, index=False)

    return len(data_frame), len(data_frame.columns)


def main() -> None:
    source_dir = get_source_dir()
    source_dir.mkdir(exist_ok=True)
    extract_zip_archives(source_dir)

    sav_paths = sorted(source_dir.glob("*.sav"))
    if not sav_paths:
        raise FileNotFoundError(f"No .sav files found in {source_dir}")

    for sav_path in sav_paths:
        rows, cols = convert_sav_to_csv(sav_path, BASE_DIR)
        print(f"Converted {sav_path.name}: {rows} rows x {cols} columns")


if __name__ == "__main__":
    main()
