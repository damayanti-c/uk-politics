from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "source_data" / "election_results"
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xls", ".xlsx"}


@dataclass(frozen=True)
class SourcePage:
    category: str
    year: int
    page_type: str
    page_url: str
    target_subdir: str
    notes: str = ""


@dataclass(frozen=True)
class DownloadRecord:
    category: str
    year: int
    page_type: str
    page_title: str
    page_url: str
    source_text: str
    source_url: str
    local_path: str
    notes: str


SOURCE_PAGES: tuple[SourcePage, ...] = (
    SourcePage(
        category="national_general_elections",
        year=2017,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-7979/",
        target_subdir="national_general_elections/2017_results_analysis",
    ),
    SourcePage(
        category="national_general_elections",
        year=2019,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-8749/",
        target_subdir="national_general_elections/2019_results_analysis",
    ),
    SourcePage(
        category="national_general_elections",
        year=2024,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-10009/",
        target_subdir="national_general_elections/2024_results_analysis",
        notes="Includes constituency, candidate, regional, MP and change-log files published on the Commons Library page.",
    ),
    SourcePage(
        category="local_elections",
        year=2016,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-7596/",
        target_subdir="local_elections/2016_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2017,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-7975/",
        target_subdir="local_elections/2017_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2018,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-8306/",
        target_subdir="local_elections/2018_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2019,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-8566/",
        target_subdir="local_elections/2019_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2021,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-9228/",
        target_subdir="local_elections/2021_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2021,
        page_type="handbook_dataset",
        page_url="https://commonslibrary.parliament.uk/data/parliament-elections-data/2021-local-elections-handbook-and-dataset/",
        target_subdir="local_elections/2021_handbook_dataset",
    ),
    SourcePage(
        category="local_elections",
        year=2022,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-9545/",
        target_subdir="local_elections/2022_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2022,
        page_type="handbook_dataset",
        page_url="https://commonslibrary.parliament.uk/data/parliament-elections-data/2022-local-elections-handbook-and-dataset/",
        target_subdir="local_elections/2022_handbook_dataset",
    ),
    SourcePage(
        category="local_elections",
        year=2023,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-9798/",
        target_subdir="local_elections/2023_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2023,
        page_type="handbook_dataset",
        page_url="https://commonslibrary.parliament.uk/2023-local-elections-handbook-and-dataset/",
        target_subdir="local_elections/2023_handbook_dataset",
    ),
    SourcePage(
        category="local_elections",
        year=2024,
        page_type="handbook_dataset",
        page_url="https://commonslibrary.parliament.uk/2024-local-elections-handbook-and-dataset/",
        target_subdir="local_elections/2024_handbook_dataset",
        notes="No separate 2024 Commons Library local-election results-analysis page was located during source collection.",
    ),
    SourcePage(
        category="local_elections",
        year=2025,
        page_type="results_analysis",
        page_url="https://commonslibrary.parliament.uk/research-briefings/cbp-10272/",
        target_subdir="local_elections/2025_results_analysis",
    ),
    SourcePage(
        category="local_elections",
        year=2025,
        page_type="handbook_dataset",
        page_url="https://commonslibrary.parliament.uk/2025-local-elections-handbook-and-dataset/",
        target_subdir="local_elections/2025_handbook_dataset",
    ),
)


def normalize_text(value: str) -> str:
    """Collapse internal whitespace so titles and labels are stable for matching."""
    return " ".join(value.split())


def build_driver(download_dir: Path) -> webdriver.Chrome:
    """Create a headless Chrome driver configured for unattended file downloads."""
    options = Options()
    options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
            "safebrowsing.enabled": True,
        },
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {"behavior": "allow", "downloadPath": str(download_dir)},
    )
    driver.set_page_load_timeout(120)
    return driver


def set_download_dir(driver: webdriver.Chrome, download_dir: Path) -> None:
    """Update Chrome's active download directory for the next file requests."""
    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {"behavior": "allow", "downloadPath": str(download_dir)},
    )


def iter_download_links(driver: webdriver.Chrome) -> Iterable[tuple[str, str]]:
    """Yield unique document links from the current page that point to supported file types."""
    seen: set[str] = set()
    for anchor in driver.find_elements(By.TAG_NAME, "a"):
        href = anchor.get_attribute("href")
        text = normalize_text(anchor.text or "")
        if not href or href in seen:
            continue
        parsed = urlparse(href)
        extension = Path(parsed.path).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        if parsed.netloc not in {
            "researchbriefings.files.parliament.uk",
            "commonslibrary.parliament.uk",
        }:
            continue
        seen.add(href)
        yield text, href


def expected_filename(source_url: str) -> str:
    """Return the file name implied by the source URL path."""
    return unquote(Path(urlparse(source_url).path).name)


def wait_for_file(download_dir: Path, file_name: str, timeout_seconds: int = 120) -> Path:
    """Wait for Chrome to finish downloading a named file into the target directory."""
    target = download_dir / file_name
    partial = download_dir / f"{file_name}.crdownload"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if target.exists() and not partial.exists():
            return target
        time.sleep(1)

    raise TimeoutError(f"Timed out waiting for {file_name} in {download_dir}")


def collect_page_documents(
    driver: webdriver.Chrome, page: SourcePage
) -> tuple[str, list[tuple[str, str]]]:
    """Open a source page and collect its downloadable Commons Library documents."""
    driver.get(page.page_url)
    time.sleep(5)
    page_title = normalize_text(driver.title.replace(" - House of Commons Library", ""))
    documents = sorted(iter_download_links(driver), key=lambda item: item[1].lower())
    return page_title, documents


def download_document(
    driver: webdriver.Chrome, download_dir: Path, source_url: str
) -> Path:
    """Download a single file into the requested directory, skipping files already present."""
    file_name = expected_filename(source_url)
    destination = download_dir / file_name
    if destination.exists():
        return destination

    driver.get(source_url)
    return wait_for_file(download_dir, file_name)


def write_manifest(records: list[DownloadRecord]) -> None:
    """Persist a flat CSV manifest describing every downloaded file and its source."""
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "year",
                "page_type",
                "page_title",
                "page_url",
                "source_text",
                "source_url",
                "local_path",
                "notes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_readme(records: list[DownloadRecord]) -> None:
    """Write a short repository-facing summary of the downloaded election files."""
    readme_path = OUTPUT_ROOT / "README.md"
    page_count = len({record.page_url for record in records})
    lines = [
        "# House of Commons Library election results",
        "",
        "This folder contains House of Commons Library files downloaded for the 2016-2025 calendar years.",
        "",
        "- Scope used here: Westminster general election results pages and local election results/handbook pages published by the Commons Library.",
        "- Storage layout: `national_general_elections/` and `local_elections/`, grouped into page-specific subfolders.",
        "- Manifest: see `manifest.csv` for the source page, download URL and saved path for every file.",
        "- Refresh command: `python scripts/download_hoc_election_results.py`.",
        "",
        f"Downloaded files: {len(records)}",
        f"Source page groups: {page_count}",
        "",
        "Notes:",
        "- A separate 2024 Commons Library local-election results-analysis page was not located during source collection, so only the 2024 handbook/dataset page is included for that year.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Download the curated Commons Library election-result files into the repo."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    driver = build_driver(OUTPUT_ROOT)
    records: list[DownloadRecord] = []

    try:
        for page in SOURCE_PAGES:
            page_title, documents = collect_page_documents(driver, page)
            target_dir = OUTPUT_ROOT / page.target_subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            set_download_dir(driver, target_dir)

            print(f"[page] {page.year} {page.page_type}: {page_title}")
            for source_text, source_url in documents:
                local_path = download_document(driver, target_dir, source_url)
                record = DownloadRecord(
                    category=page.category,
                    year=page.year,
                    page_type=page.page_type,
                    page_title=page_title,
                    page_url=page.page_url,
                    source_text=source_text,
                    source_url=source_url,
                    local_path=local_path.relative_to(OUTPUT_ROOT).as_posix(),
                    notes=page.notes,
                )
                records.append(record)
                print(f"  saved {record.local_path}")
    finally:
        driver.quit()

    records.sort(key=lambda item: (item.category, item.year, item.page_type, item.local_path))
    write_manifest(records)
    write_readme(records)
    print(f"Wrote manifest for {len(records)} files to {OUTPUT_ROOT / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # pragma: no cover - operational script
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
