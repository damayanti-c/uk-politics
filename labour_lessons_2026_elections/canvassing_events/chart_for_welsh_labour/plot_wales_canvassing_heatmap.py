"""Draw a Wales ward heatmap of Labour canvassing intensity in spring 2026.

This script joins the existing ward-level Labour canvassing tally to the ONS
December 2025 UK ward boundaries, computes each Welsh ward's area from the
British National Grid geometry embedded in the GeoJSON, and renders a Wales-only
choropleth.

Outputs
-------
* ``wales_canvassing_heatmap_2026.csv`` - every Welsh ward, including wards with
  zero canvassing events.
* ``wales_canvassing_heatmap_2026.png`` - static choropleth of canvassing event
  density (events per sq km), with zero-event wards shown separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_COUNTS_CSV = BASE_DIR.parent / "ward_canvassing_frequency_2026.csv"
DEFAULT_GEOJSON = BASE_DIR.parent.parent.parent / "source_data" / "geography" / "wards_december_2025_boundaries_uk_bgc.geojson"
DEFAULT_OUTPUT_CSV = BASE_DIR / "wales_canvassing_heatmap_2026.csv"
DEFAULT_OUTPUT_PNG = BASE_DIR / "wales_canvassing_heatmap_2026.png"

BACKGROUND = "#F2F2F2"
TEXT = "#001D3B"
SOURCE_TEXT = "#4A4A4A"
ZERO_FILL = "#C6CCD4"
WARD_EDGE = "#FFFFFF"
LABOUR_RED = "#E4003B"

LABOUR_HEAT_COLORS = [
    "#F1AFC2",
    "#EB7D9B",
    LABOUR_RED,
    "#BF0032",
    "#990028",
    "#73001E",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-csv", default=str(DEFAULT_COUNTS_CSV))
    parser.add_argument("--geojson", default=str(DEFAULT_GEOJSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG))
    parser.add_argument(
        "--date-window",
        default="27 Mar-7 May 2026",
        help="Human-readable label used in the subtitle/footer.",
    )
    return parser.parse_args()


def close_ring(ring: Iterable[Iterable[float]]) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in ring]
    if not points:
        return []
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def ring_area_sq_m(ring: Iterable[Iterable[float]]) -> float:
    points = close_ring(ring)
    if len(points) < 4:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_area_sq_m(coords: list[list[list[float]]]) -> float:
    if not coords:
        return 0.0
    area = ring_area_sq_m(coords[0])
    for hole in coords[1:]:
        area -= ring_area_sq_m(hole)
    return max(area, 0.0)


def geometry_area_sq_km(geometry: dict) -> float:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        return polygon_area_sq_m(coords) / 1_000_000
    if gtype == "MultiPolygon":
        return sum(polygon_area_sq_m(poly) for poly in coords) / 1_000_000
    raise ValueError(f"Unsupported geometry type: {gtype}")


def iter_exterior_rings(geometry: dict) -> Iterable[np.ndarray]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        if coords:
            yield np.asarray(coords[0], dtype=float)
        return
    if gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield np.asarray(poly[0], dtype=float)
        return
    raise ValueError(f"Unsupported geometry type: {gtype}")


def load_wales_counts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"ward_code": str})
    needed = {"ward_code", "nation", "canvassing_events"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    wales = df.loc[df["nation"] == "Wales", ["ward_code", "canvassing_events"]].copy()
    wales["canvassing_events"] = pd.to_numeric(wales["canvassing_events"], errors="coerce").fillna(0).astype(int)
    return wales.groupby("ward_code", as_index=False)["canvassing_events"].sum()


def load_wales_geography(geojson_path: Path) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    rings_by_code: dict[str, list[np.ndarray]] = {}
    for feature in payload["features"]:
        props = feature["properties"]
        code = str(props["WD25CD"])
        if not code.startswith("W05"):
            continue
        geometry = feature["geometry"]
        rows.append(
            {
                "ward_code": code,
                "ward_name": props["WD25NM"],
                "ward_name_welsh": props.get("WD25NMW", ""),
                "council": props["LAD25NM"],
                "area_sq_km": geometry_area_sq_km(geometry),
            }
        )
        rings_by_code[code] = list(iter_exterior_rings(geometry))

    geo = pd.DataFrame(rows)
    if geo.empty:
        raise ValueError(f"No Welsh wards were found in {geojson_path}")
    return geo, rings_by_code


def build_wales_dataset(counts_csv: Path, geojson_path: Path) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    counts = load_wales_counts(counts_csv)
    geo, rings_by_code = load_wales_geography(geojson_path)

    df = geo.merge(counts, on="ward_code", how="left")
    df["canvassing_events"] = df["canvassing_events"].fillna(0).astype(int)
    df["has_canvassing_event"] = df["canvassing_events"] > 0
    df["events_per_sq_km"] = np.where(
        df["area_sq_km"] > 0,
        df["canvassing_events"] / df["area_sq_km"],
        np.nan,
    )
    df["rank_by_events_per_sq_km"] = (
        df["events_per_sq_km"].rank(method="min", ascending=False).astype(int)
    )
    df["rank_by_events"] = df["canvassing_events"].rank(method="min", ascending=False).astype(int)
    return (
        df.sort_values(
            ["events_per_sq_km", "canvassing_events", "ward_name"],
            ascending=[False, False, True],
        ),
        rings_by_code,
    )


def plot_heatmap(df: pd.DataFrame, rings_by_code: dict[str, list[np.ndarray]], output_png: Path, date_window: str) -> None:
    positive = df.loc[df["events_per_sq_km"] > 0, "events_per_sq_km"]
    if positive.empty:
        raise ValueError("No positive Wales canvassing densities were found to plot.")

    display_cap = float(positive.quantile(0.975))
    if display_cap <= 0:
        display_cap = float(positive.max())
    bin_edges = [0.0, 0.15, 0.4, 0.8, 1.5, 3.0, display_cap]
    bin_edges = sorted(set(round(edge, 6) for edge in bin_edges if edge < display_cap))
    if not bin_edges or bin_edges[0] != 0.0:
        bin_edges.insert(0, 0.0)
    bin_edges.append(round(display_cap, 6))
    if len(bin_edges) - 1 > len(LABOUR_HEAT_COLORS):
        bin_edges = [0.0, 0.1, 0.3, 0.7, 1.5, display_cap]
    cmap = ListedColormap(LABOUR_HEAT_COLORS[: len(bin_edges) - 1], name="labour_heat_steps")
    norm = BoundaryNorm(bin_edges, cmap.N, clip=True)

    patches: list[MplPolygon] = []
    facecolors: list[tuple[float, float, float, float]] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for row in df.itertuples(index=False):
        density = float(row.events_per_sq_km) if pd.notna(row.events_per_sq_km) else 0.0
        clipped = min(density, display_cap)
        color = ZERO_FILL if density <= 0 else cmap(norm(clipped))
        for ring in rings_by_code[row.ward_code]:
            if len(ring) < 3:
                continue
            patches.append(MplPolygon(ring, closed=True))
            facecolors.append(color)
            all_x.extend(ring[:, 0].tolist())
            all_y.extend(ring[:, 1].tolist())

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    pad_x = (max_x - min_x) * 0.02
    pad_y = (max_y - min_y) * 0.02

    fig = plt.figure(figsize=(14.492, 8.9), dpi=150, facecolor=BACKGROUND)
    map_ax = fig.add_axes([0.04, 0.18, 0.92, 0.66])
    map_ax.set_facecolor(BACKGROUND)

    collection = PatchCollection(
        patches,
        facecolor=facecolors,
        edgecolor=WARD_EDGE,
        linewidth=0.28,
        antialiased=True,
    )
    map_ax.add_collection(collection)
    map_ax.set_xlim(min_x - pad_x, max_x + pad_x)
    map_ax.set_ylim(min_y - pad_y, max_y + pad_y)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_axis_off()

    total_events = int(df["canvassing_events"].sum())
    zero_wards = int((df["canvassing_events"] == 0).sum())

    fig.text(
        0.04,
        0.93,
        "Labour canvassing density by ward in Wales",
        ha="left",
        va="top",
        fontsize=26,
        color=TEXT,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.885,
        f"Events per sq km, {date_window}. Grey wards had no recorded canvassing event; "
        f"colour scale stepped and capped at the 97.5th percentile.",
        ha="left",
        va="top",
        fontsize=14,
        color=TEXT,
    )

    cax = fig.add_axes([0.63, 0.085, 0.25, 0.018])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", boundaries=bin_edges, spacing="proportional")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors=TEXT, length=0, pad=2)
    cbar.set_ticks(bin_edges[1:])
    cbar.set_ticklabels([f"{tick:g}" for tick in bin_edges[1:]])

    fig.text(0.63, 0.112, "Canvassing events per sq km", ha="left", va="bottom", fontsize=12, color=TEXT)
    fig.text(
        0.63,
        0.035,
        f"Grey = 0 events. Welsh wards: {len(df)}; zero-event wards: {zero_wards}.\n"
        f"Recorded Welsh canvassing events: {total_events}.",
        ha="left",
        va="bottom",
        fontsize=9,
        color=SOURCE_TEXT,
        linespacing=1.35,
    )
    fig.text(
        0.04,
        0.035,
        "Source: Labour public events data (`ward_canvassing_frequency_2026.csv`) joined to\n"
        "ONS Wards (December 2025) boundaries. Density = events divided by ward area in sq km.",
        ha="left",
        va="bottom",
        fontsize=9,
        color=SOURCE_TEXT,
        linespacing=1.35,
    )

    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    counts_csv = Path(args.counts_csv)
    geojson_path = Path(args.geojson)
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)

    df, rings_by_code = build_wales_dataset(counts_csv, geojson_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "ward_code",
        "ward_name",
        "ward_name_welsh",
        "council",
        "area_sq_km",
        "canvassing_events",
        "events_per_sq_km",
        "has_canvassing_event",
        "rank_by_events_per_sq_km",
        "rank_by_events",
    ]
    df.loc[:, export_cols].to_csv(output_csv, index=False)
    plot_heatmap(df, rings_by_code, output_png, args.date_window)

    top = df.head(10).copy()
    print("Top Welsh wards by canvassing event density (events per sq km):")
    for row in top.itertuples(index=False):
        print(
            f"  {row.ward_name:28s} {row.council:20s} "
            f"{row.canvassing_events:>3} events  {row.events_per_sq_km:>6.2f} / sq km"
        )
    print(f"\nWrote {output_csv.name} ({len(df)} wards) and {output_png.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
