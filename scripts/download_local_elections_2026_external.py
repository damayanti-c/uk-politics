from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "source_data" / "election_results" / "local_elections" / "2026_external"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

RACES_URL = "https://electionresults.uk/data/races.csv"
CANDIDATES_URL = "https://electionresults.uk/data/candidates.csv"
COUNCIL_CONTROL_URL = "https://opencouncildata.co.uk/history2016-26.csv"

RACES_RAW_PATH = OUTPUT_DIR / "electionresults_uk_races.csv"
CANDIDATES_RAW_PATH = OUTPUT_DIR / "electionresults_uk_candidates.csv"
COUNCIL_CONTROL_RAW_PATH = OUTPUT_DIR / "opencouncildata_history2016_26.csv"

RACES_2026_PATH = OUTPUT_DIR / "electionresults_uk_races_2026.csv"
CANDIDATES_2026_PATH = OUTPUT_DIR / "electionresults_uk_candidates_2026.csv"
COUNCIL_CONTROL_2026_PATH = OUTPUT_DIR / "opencouncildata_council_control_2026.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"


@dataclass(frozen=True)
class OutputRecord:
    source_name: str
    source_url: str
    output_path: str
    scope: str
    notes: str


def download_text(url: str, output_path: Path) -> str:
    """Download a text file and write it to disk."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=300)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    return response.text


def load_remote_csv(url: str, raw_output_path: Path) -> pd.DataFrame:
    """Fetch a CSV source, cache the raw text locally, and parse it with pandas."""
    text = download_text(url, raw_output_path)
    return pd.read_csv(io.StringIO(text))


def build_manifest(records: list[OutputRecord]) -> None:
    """Write a simple manifest for the 2026 external result files."""
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_name", "source_url", "output_path", "scope", "notes"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def main() -> None:
    races = load_remote_csv(RACES_URL, RACES_RAW_PATH)
    candidates = load_remote_csv(CANDIDATES_URL, CANDIDATES_RAW_PATH)
    council_control = load_remote_csv(COUNCIL_CONTROL_URL, COUNCIL_CONTROL_RAW_PATH)

    races_2026 = races.loc[races["year"] == 2026].copy()
    candidates_2026 = candidates.loc[candidates["year"] == 2026].copy()

    council_control.columns = [str(column).strip() for column in council_control.columns]
    council_control_2026 = council_control.loc[council_control["year"] == 2026].copy()
    council_control_2026 = council_control_2026.rename(columns={"Unnamed: 15": "authority_code"})

    races_2026.to_csv(RACES_2026_PATH, index=False)
    candidates_2026.to_csv(CANDIDATES_2026_PATH, index=False)
    council_control_2026.to_csv(COUNCIL_CONTROL_2026_PATH, index=False)

    manifest_records = [
        OutputRecord(
            source_name="electionresults.uk races feed",
            source_url=RACES_URL,
            output_path=str(RACES_RAW_PATH.relative_to(OUTPUT_DIR.parent.parent.parent)),
            scope="All available local-election race rows cached locally.",
            notes="External source used because a Commons Library 2026 local-election release was not located as of 2026-05-31.",
        ),
        OutputRecord(
            source_name="electionresults.uk races feed",
            source_url=RACES_URL,
            output_path=str(RACES_2026_PATH.relative_to(OUTPUT_DIR.parent.parent.parent)),
            scope="2026 rows filtered from the full race feed.",
            notes="One row per ward or division contest.",
        ),
        OutputRecord(
            source_name="electionresults.uk candidates feed",
            source_url=CANDIDATES_URL,
            output_path=str(CANDIDATES_RAW_PATH.relative_to(OUTPUT_DIR.parent.parent.parent)),
            scope="All available local-election candidate rows cached locally.",
            notes="External source used because a Commons Library 2026 local-election release was not located as of 2026-05-31.",
        ),
        OutputRecord(
            source_name="electionresults.uk candidates feed",
            source_url=CANDIDATES_URL,
            output_path=str(CANDIDATES_2026_PATH.relative_to(OUTPUT_DIR.parent.parent.parent)),
            scope="2026 rows filtered from the full candidates feed.",
            notes="One row per candidate in a 2026 local-election contest.",
        ),
        OutputRecord(
            source_name="Open Council Data annual composition archive",
            source_url=COUNCIL_CONTROL_URL,
            output_path=str(COUNCIL_CONTROL_RAW_PATH.relative_to(OUTPUT_DIR.parent.parent.parent)),
            scope="2016-2026 council composition archive cached locally.",
            notes="Used here for 2026 council-control context.",
        ),
        OutputRecord(
            source_name="Open Council Data annual composition archive",
            source_url=COUNCIL_CONTROL_URL,
            output_path=str(COUNCIL_CONTROL_2026_PATH.relative_to(OUTPUT_DIR.parent.parent.parent)),
            scope="2026 rows filtered from the council composition archive.",
            notes="Includes the raw majority label and authority code column.",
        ),
    ]
    build_manifest(manifest_records)

    print(f"Saved {len(races_2026):,} 2026 race rows to {RACES_2026_PATH}")
    print(f"Saved {len(candidates_2026):,} 2026 candidate rows to {CANDIDATES_2026_PATH}")
    print(f"Saved {len(council_control_2026):,} 2026 council-control rows to {COUNCIL_CONTROL_2026_PATH}")
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
