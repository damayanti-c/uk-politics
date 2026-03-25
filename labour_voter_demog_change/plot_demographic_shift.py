from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Labour-to-Reform shift over time by demographic subgroup, "
            "using weighted_vi_by_demographic.csv"
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "weighted_vi_by_demographic.csv"
        ),
        help="Path to weighted_vi_by_demographic.csv",
    )
    parser.add_argument(
        "--output-png",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "labour_to_reform_shift_by_demographic.png"
        ),
        help="Path for the output chart PNG",
    )
    parser.add_argument(
        "--output-arrow-png",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "labour_to_reform_shift_arrow_summary.png"
        ),
        help="Path for the arrow-summary chart PNG",
    )
    parser.add_argument(
        "--output-arrow-csv",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "labour_to_reform_shift_arrow_summary_values.csv"
        ),
        help=(
            "Path for CSV containing all subgroup values behind the arrow chart "
            "(start lead, end lead, and change)."
        ),
    )
    parser.add_argument(
        "--max-series-per-panel",
        type=int,
        default=12,
        help="Max subgroup lines shown per demographic panel (largest absolute shift)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of time points required for a subgroup line",
    )
    parser.add_argument(
        "--arrow-max-rows",
        type=int,
        default=35,
        help="Maximum number of subgroup rows shown in the arrow summary chart.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def canonical_demog_order(types: list[str]) -> list[str]:
    preferred = ["age_band", "education_level", "gender", "ethnicity", "past_vote"]
    out = [t for t in preferred if t in types]
    out.extend([t for t in types if t not in out])
    return out


def pretty_demog_type(demog_type: str) -> str:
    return demog_type.replace("_", " ").title()


def pick_series_for_panel(sub: pd.DataFrame, max_series: int, min_points: int) -> list[str]:
    grp = (
        sub.groupby("DEMOGRAPHIC_VALUE", as_index=False)
        .agg(
            n_points=("SURVEY_DATE", "nunique"),
            first_date=("SURVEY_DATE", "min"),
            last_date=("SURVEY_DATE", "max"),
        )
        .sort_values(["n_points", "DEMOGRAPHIC_VALUE"], ascending=[False, True])
    )

    candidates = []
    for value in grp["DEMOGRAPHIC_VALUE"].tolist():
        ts = sub[sub["DEMOGRAPHIC_VALUE"] == value].sort_values("SURVEY_DATE")
        if ts["SURVEY_DATE"].nunique() < min_points:
            continue
        first = ts.iloc[0]["LABOUR_MINUS_REFORM_LEAD_PCT"]
        last = ts.iloc[-1]["LABOUR_MINUS_REFORM_LEAD_PCT"]
        shift = float(last - first)
        candidates.append((value, abs(shift), shift))

    if not candidates:
        return []

    # Prioritize the largest absolute movement in lead (towards or away from Reform)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [v for v, _, _ in candidates[:max_series]]


def plot_panel(ax: plt.Axes, sub: pd.DataFrame, demog_type: str, max_series: int, min_points: int) -> None:
    series_values = pick_series_for_panel(sub, max_series=max_series, min_points=min_points)
    if not series_values:
        ax.set_title(f"{demog_type} (insufficient data)")
        ax.axis("off")
        return

    plot_df = sub[sub["DEMOGRAPHIC_VALUE"].isin(series_values)].copy()
    plot_df = plot_df.sort_values(["DEMOGRAPHIC_VALUE", "SURVEY_DATE"])

    cmap = plt.get_cmap("tab20")
    for i, value in enumerate(series_values):
        ts = plot_df[plot_df["DEMOGRAPHIC_VALUE"] == value]
        ax.plot(
            ts["SURVEY_DATE"],
            ts["LABOUR_MINUS_REFORM_LEAD_PCT"],
            label=str(value),
            color=cmap(i % 20),
            linewidth=1.8,
            alpha=0.9,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_title(
        f"{demog_type} (Labour VI - Reform VI, pp)",
        loc="left",
        fontsize=10,
        pad=6,
    )
    ax.set_ylabel("Lead (pp)")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        frameon=False,
        ncol=1,
    )


def build_shift_table(df: pd.DataFrame, min_points: int) -> pd.DataFrame:
    rows = []
    for (demog_type, demog_value), grp in df.groupby(["DEMOGRAPHIC_TYPE", "DEMOGRAPHIC_VALUE"]):
        ts = grp.sort_values("SURVEY_DATE")
        if ts["SURVEY_DATE"].nunique() < min_points:
            continue
        first = float(ts.iloc[0]["LABOUR_MINUS_REFORM_LEAD_PCT"])
        last = float(ts.iloc[-1]["LABOUR_MINUS_REFORM_LEAD_PCT"])
        rows.append(
            {
                "DEMOGRAPHIC_TYPE": demog_type,
                "DEMOGRAPHIC_VALUE": demog_value,
                "LABEL": f"{pretty_demog_type(str(demog_type))}: {demog_value}",
                "FIRST_LEAD": first,
                "LAST_LEAD": last,
                "DELTA_LEAD": last - first,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("DELTA_LEAD", ascending=True).reset_index(drop=True)
    return out


def plot_arrow_summary(shift_df: pd.DataFrame, out_path: Path, max_rows: int) -> None:
    if shift_df.empty:
        raise ValueError("No rows available for arrow summary chart.")

    plot_df = shift_df.copy()
    if len(plot_df) > max_rows:
        # Keep the most extreme changes by absolute movement.
        plot_df = (
            plot_df.assign(abs_delta=plot_df["DELTA_LEAD"].abs())
            .sort_values("abs_delta", ascending=False)
            .head(max_rows)
            .drop(columns=["abs_delta"])
            .sort_values("DELTA_LEAD", ascending=True)
            .reset_index(drop=True)
        )

    fig_h = max(6, 0.32 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = list(range(len(plot_df)))
    ax.axvline(0, color="black", linewidth=1.2, alpha=0.8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)

    for i, delta in enumerate(plot_df["DELTA_LEAD"].tolist()):
        ax.annotate(
            "",
            xy=(delta, i),
            xytext=(0, i),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black", shrinkA=0, shrinkB=0),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["LABEL"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in Labour Lead Over Reform (pp), end minus start")
    ax.set_title(
        "Overall Shift in Labour Lead Over Reform by Demographic Subgroup\n"
        "(negative = movement toward Reform)",
        loc="left",
        fontsize=12,
    )

    # Add small visual margin so arrowheads are not clipped.
    x_min = min(plot_df["DELTA_LEAD"].min(), 0) - 0.6
    x_max = max(plot_df["DELTA_LEAD"].max(), 0) + 0.6
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv).resolve()
    out_path = Path(args.output_png).resolve()
    out_arrow_path = Path(args.output_arrow_png).resolve()
    out_arrow_csv_path = Path(args.output_arrow_csv).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    df = normalize_columns(df)

    required = {
        "SURVEY_DATE",
        "DEMOGRAPHIC_TYPE",
        "DEMOGRAPHIC_VALUE",
        "LABOUR_MINUS_REFORM_LEAD_PCT",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df["SURVEY_DATE"] = pd.to_datetime(df["SURVEY_DATE"], errors="coerce")
    df = df.dropna(subset=["SURVEY_DATE", "DEMOGRAPHIC_TYPE", "DEMOGRAPHIC_VALUE", "LABOUR_MINUS_REFORM_LEAD_PCT"])

    demog_types = sorted(df["DEMOGRAPHIC_TYPE"].astype(str).unique().tolist())
    demog_types = canonical_demog_order(demog_types)

    if not demog_types:
        raise ValueError("No demographic rows found to plot.")

    n = len(demog_types)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, 3.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, demog_type in zip(axes, demog_types):
        sub = df[df["DEMOGRAPHIC_TYPE"] == demog_type].copy()
        plot_panel(
            ax,
            sub,
            demog_type,
            max_series=args.max_series_per_panel,
            min_points=args.min_points,
        )

    axes[-1].set_xlabel("Survey date")
    fig.suptitle(
        "Shift Away From Labour Toward Reform Over Time by Demographic Subgroup\n"
        "Metric shown: Labour VI - Reform VI (percentage point lead)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    shift_df = build_shift_table(df, min_points=args.min_points)
    plot_arrow_summary(shift_df, out_arrow_path, max_rows=args.arrow_max_rows)

    # Export all subgroup values behind the arrow chart, sorted by lead change descending.
    shift_export = shift_df.sort_values("DELTA_LEAD", ascending=False).reset_index(drop=True)
    out_arrow_csv_path.parent.mkdir(parents=True, exist_ok=True)
    shift_export.to_csv(out_arrow_csv_path, index=False)

    print(f"Saved chart: {out_path}")
    print(f"Saved arrow summary: {out_arrow_path}")
    print(f"Saved arrow values CSV: {out_arrow_csv_path}")


if __name__ == "__main__":
    main()
