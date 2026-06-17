from __future__ import annotations

import csv
import io
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyreadstat


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
MAX_STORED_FILE_BYTES = 95 * 1024 * 1024
DEFAULT_SAMPLE_ROWS = 100
TARGET_CELLS_PER_CHUNK = 20_000_000


@dataclass(frozen=True)
class RawArtifact:
    raw_name: str
    source_path: Path
    description: str


@dataclass(frozen=True)
class ArchiveSpec:
    raw_name: str
    output_stem: str
    description: str


@dataclass(frozen=True)
class PartRecord:
    dataset: str
    part_file: str
    row_start: int
    row_end: int
    row_count: int
    file_size_bytes: int


RAW_ARTIFACTS: tuple[RawArtifact, ...] = (
    RawArtifact(
        raw_name="Bes_wave30Documentationv30.1.pdf",
        source_path=Path(r"C:\Users\DamayantiChatterjee\Downloads\Bes_wave30Documentationv30.1.pdf"),
        description="Documentation PDF for the v30.1 combined panel (waves 1-30).",
    ),
    RawArtifact(
        raw_name="BES2024_W30Strings_v30.1.sav.zip",
        source_path=Path(r"C:\Users\DamayantiChatterjee\Downloads\BES2024_W30Strings_v30.1.sav.zip"),
        description="Combined panel strings SPSS archive, v30.1 release (waves 1-30).",
    ),
    RawArtifact(
        raw_name="BES2024_W30_Panel_v30.1.sav.zip",
        source_path=Path(r"C:\Users\DamayantiChatterjee\Downloads\BES2024_W30_Panel_v30.1.sav.zip"),
        description="Combined panel SPSS archive, v30.1 release (waves 1-30, variables suffixed W1..W30).",
    ),
)

ARCHIVES: tuple[ArchiveSpec, ...] = (
    ArchiveSpec(
        raw_name="BES2024_W30_Panel_v30.1.sav.zip",
        output_stem="bes_voter_panel_2024_combined_w1_w30_panel",
        description="Combined BES internet panel (all waves 1-30, variables suffixed W1..W30).",
    ),
    ArchiveSpec(
        raw_name="BES2024_W30Strings_v30.1.sav.zip",
        output_stem="bes_voter_panel_2024_combined_w1_w30_strings",
        description="Companion strings file for the combined panel (all waves 1-30).",
    ),
)


def build_codebook_frame(meta: pyreadstat._readstat_parser.metadata_container) -> pd.DataFrame:
    """Convert SPSS metadata into a flat codebook table."""
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


def artifact_is_staged(raw_name: str) -> bool:
    """Return True when a raw artifact already exists in raw/ as a file or parts."""
    return (RAW_DIR / raw_name).exists() or any(RAW_DIR.glob(f"{raw_name}.part*"))


def split_copy_file(source_path: Path, dest_name: str) -> list[Path]:
    """Copy a large source file into raw/ as numbered parts under the size limit."""
    parts: list[Path] = []
    with source_path.open("rb") as source:
        part_number = 1
        while True:
            chunk = source.read(MAX_STORED_FILE_BYTES)
            if not chunk:
                break
            part_path = RAW_DIR / f"{dest_name}.part{part_number:03d}"
            part_path.write_bytes(chunk)
            parts.append(part_path)
            part_number += 1
    return parts


def stage_raw_artifacts() -> None:
    """Copy source files into raw/, splitting oversized files into smaller parts."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for artifact in RAW_ARTIFACTS:
        if artifact_is_staged(artifact.raw_name):
            continue
        if not artifact.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {artifact.source_path}")
        if artifact.source_path.stat().st_size <= MAX_STORED_FILE_BYTES:
            shutil.copy2(artifact.source_path, RAW_DIR / artifact.raw_name)
        else:
            split_copy_file(artifact.source_path, artifact.raw_name)


def staged_paths_for(raw_name: str) -> list[Path]:
    """Return the stored raw path or split parts for a logical raw artifact."""
    single = RAW_DIR / raw_name
    if single.exists():
        return [single]
    parts = sorted(RAW_DIR.glob(f"{raw_name}.part*"))
    if parts:
        return parts
    raise FileNotFoundError(f"No staged raw artifact found for {raw_name}")


def materialize_staged_artifact(raw_name: str, temp_dir: Path) -> Path:
    """Rebuild a staged artifact into a concrete file path when it is stored in parts."""
    staged_paths = staged_paths_for(raw_name)
    if len(staged_paths) == 1 and staged_paths[0].name == raw_name:
        return staged_paths[0]

    rebuilt_path = temp_dir / raw_name
    with rebuilt_path.open("wb") as target:
        for part_path in staged_paths:
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, target)
    return rebuilt_path


def extract_single_sav(zip_path: Path, temp_dir: Path) -> Path:
    """Extract the single .sav member from a zip archive into a temporary directory."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.infolist() if member.filename.lower().endswith(".sav")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one .sav file in {zip_path.name}, found {len(members)}")
        member = members[0]
        sav_path = temp_dir / Path(member.filename).name
        with archive.open(member) as source, sav_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        return sav_path


def get_read_kwargs(encoding: str | None) -> dict[str, object]:
    """Return common pyreadstat kwargs, including encoding when explicitly selected."""
    kwargs: dict[str, object] = {
        "apply_value_formats": False,
        "user_missing": True,
    }
    if encoding is not None:
        kwargs["encoding"] = encoding
    return kwargs


def choose_working_encoding(sav_path: Path, total_rows: int, preferred_encoding: str | None) -> str | None:
    """Pick an encoding that can read sampled sections of the file without errors."""
    candidates: list[str | None] = []
    if preferred_encoding:
        candidates.append(preferred_encoding)
    candidates.extend([None, "utf-8", "latin1", "cp1252"])

    deduped: list[str | None] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = "" if candidate is None else candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    sample_offsets = sorted({0, max(0, total_rows // 2), max(0, total_rows - DEFAULT_SAMPLE_ROWS)})
    for candidate in deduped:
        try:
            kwargs = get_read_kwargs(candidate)
            for offset in sample_offsets:
                pyreadstat.read_sav(
                    sav_path,
                    row_offset=offset,
                    row_limit=DEFAULT_SAMPLE_ROWS,
                    **kwargs,
                )
            return candidate
        except pyreadstat._readstat_parser.ReadstatError:
            continue

    raise ValueError(f"Unable to find a working encoding for {sav_path.name}")


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a frame to UTF-8 CSV bytes without an index."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def estimate_rows_per_chunk(sav_path: Path, column_count: int, encoding: str | None) -> int:
    """Estimate a conservative chunk row count that stays within memory and file-size limits."""
    kwargs = get_read_kwargs(encoding)
    sample_frame, _ = pyreadstat.read_sav(sav_path, row_limit=DEFAULT_SAMPLE_ROWS, **kwargs)
    if sample_frame.empty:
        return 1

    sample_bytes = len(dataframe_to_csv_bytes(sample_frame))
    bytes_per_row = max(sample_bytes / len(sample_frame), 1)
    rows_by_size = max(1, math.floor((MAX_STORED_FILE_BYTES * 0.75) / bytes_per_row))
    rows_by_cells = max(1, TARGET_CELLS_PER_CHUNK // max(column_count, 1))
    return max(1, min(rows_by_size, rows_by_cells))


def split_frame_to_csv_parts(frame: pd.DataFrame) -> Iterable[tuple[int, bytes]]:
    """Recursively split a frame until each CSV fragment is within the file-size cap."""
    csv_bytes = dataframe_to_csv_bytes(frame)
    if len(csv_bytes) <= MAX_STORED_FILE_BYTES:
        yield len(frame), csv_bytes
        return

    if len(frame) <= 1:
        raise ValueError("A single-row CSV fragment exceeds the 95 MB file limit.")

    midpoint = len(frame) // 2
    yield from split_frame_to_csv_parts(frame.iloc[:midpoint].copy())
    yield from split_frame_to_csv_parts(frame.iloc[midpoint:].copy())


def cleanup_existing_outputs(output_stem: str) -> None:
    """Remove stale output parts and manifests for a dataset before rewriting it."""
    for path in BASE_DIR.glob(f"{output_stem}_part*.csv"):
        path.unlink()
    for suffix in ("_codebook.csv", "_parts_manifest.csv"):
        target = BASE_DIR / f"{output_stem}{suffix}"
        if target.exists():
            target.unlink()


def write_parts_manifest(output_stem: str, records: list[PartRecord]) -> None:
    """Write a manifest describing every CSV part emitted for a dataset."""
    manifest_path = BASE_DIR / f"{output_stem}_parts_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "part_file",
                "row_start",
                "row_end",
                "row_count",
                "file_size_bytes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_raw_manifest() -> None:
    """Record how the raw source artifacts are stored inside raw/."""
    manifest_path = RAW_DIR / "raw_parts_manifest.csv"
    rows: list[dict[str, object]] = []
    for artifact in RAW_ARTIFACTS:
        staged = staged_paths_for(artifact.raw_name)
        for index, path in enumerate(staged, start=1):
            rows.append(
                {
                    "raw_name": artifact.raw_name,
                    "description": artifact.description,
                    "stored_name": path.name,
                    "part_number": index,
                    "part_count": len(staged),
                    "file_size_bytes": path.stat().st_size,
                    "source_path": str(artifact.source_path),
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_name",
                "description",
                "stored_name",
                "part_number",
                "part_count",
                "file_size_bytes",
                "source_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_readme(dataset_summaries: list[dict[str, object]]) -> None:
    """Summarize the staged raw files and cleaned outputs for repository use."""
    readme_path = BASE_DIR / "README.md"
    lines = [
        "# BES voter panel internet study",
        "",
        "This folder stores the BES internet panel (combined file, v30.1 release) inputs "
        "and cleaned CSV exports.",
        "",
        "Note: this is the *combined* longitudinal panel covering all waves 1-30 (2014-2024), "
        "not a single wave-30 cross-section. Each survey item is repeated per wave with a "
        "`W1`..`W30` suffix (e.g. `generalElectionVoteW30`); `wave1`..`wave30` flag which "
        "waves each respondent took. The `W30` in the source filename is the release version, "
        "not the wave coverage.",
        "",
        "- Raw sources are staged in `raw/`.",
        "- Any staged raw artifact larger than 95 MB is stored as `.partNNN` chunks to keep every file under 100 MB.",
        "- Cleaned outputs are UTF-8 CSV files split into `*_partNNN.csv` files, each below 95 MB.",
        "- Metadata is flattened into `*_codebook.csv` and part-level row ranges are tracked in `*_parts_manifest.csv`.",
        "- Refresh command: `python source_data/bes_voter_panel_internet_study/convert_bes_voter_panel_internet_study.py`.",
        "",
        "Datasets:",
    ]

    for summary in dataset_summaries:
        lines.append(
            f"- `{summary['output_stem']}`: {summary['rows']:,} rows, {summary['cols']:,} columns, "
            f"{summary['part_count']} CSV parts, max part size {summary['max_part_size_mb']:.1f} MB, "
            f"encoding `{summary['encoding'] or 'default'}`."
        )

    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_archive(archive: ArchiveSpec) -> dict[str, object]:
    """Convert a staged SPSS archive into chunked CSV parts and a codebook."""
    cleanup_existing_outputs(archive.output_stem)

    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        zip_path = materialize_staged_artifact(archive.raw_name, temp_dir)
        sav_path = extract_single_sav(zip_path, temp_dir)

        _, meta = pyreadstat.read_sav(sav_path, metadataonly=True)
        encoding = choose_working_encoding(sav_path, meta.number_rows, meta.file_encoding)
        rows_per_chunk = estimate_rows_per_chunk(sav_path, len(meta.column_names), encoding)
        build_codebook_frame(meta).to_csv(BASE_DIR / f"{archive.output_stem}_codebook.csv", index=False)

        kwargs = get_read_kwargs(encoding)
        reader = pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav,
            str(sav_path),
            chunksize=rows_per_chunk,
            **kwargs,
        )

        part_records: list[PartRecord] = []
        next_row_start = 0
        part_number = 1

        for frame, _ in reader:
            if frame.empty:
                continue
            local_row_offset = 0
            for row_count, csv_bytes in split_frame_to_csv_parts(frame):
                part_name = f"{archive.output_stem}_part{part_number:03d}.csv"
                part_path = BASE_DIR / part_name
                part_path.write_bytes(csv_bytes)
                row_start = next_row_start + local_row_offset
                row_end = row_start + row_count
                part_records.append(
                    PartRecord(
                        dataset=archive.output_stem,
                        part_file=part_name,
                        row_start=row_start,
                        row_end=row_end,
                        row_count=row_count,
                        file_size_bytes=part_path.stat().st_size,
                    )
                )
                local_row_offset += row_count
                part_number += 1
            next_row_start += len(frame)

        write_parts_manifest(archive.output_stem, part_records)

        max_part_size = max(record.file_size_bytes for record in part_records)
        return {
            "output_stem": archive.output_stem,
            "rows": meta.number_rows,
            "cols": len(meta.column_names),
            "part_count": len(part_records),
            "max_part_size_mb": max_part_size / (1024 * 1024),
            "encoding": encoding,
        }


def main() -> None:
    stage_raw_artifacts()
    write_raw_manifest()
    summaries = [convert_archive(archive) for archive in ARCHIVES]
    write_readme(summaries)

    for summary in summaries:
        print(
            f"Converted {summary['output_stem']}: {summary['rows']:,} rows x {summary['cols']:,} columns "
            f"into {summary['part_count']} parts (max {summary['max_part_size_mb']:.1f} MB)"
        )


if __name__ == "__main__":
    main()
