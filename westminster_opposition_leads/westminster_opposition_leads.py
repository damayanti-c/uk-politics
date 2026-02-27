from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

POLLBASE_LANDING_PAGE = "https://www.markpack.org.uk/opinion-polls/"
POLITICO_URL = "https://www.politico.eu/wp-json/politico/v1/poll-of-polls/GB-parliament"
NANOS_PER_DAY = 86_400_000_000_000

POLLSTER_RENAMES = {
    "Find Out Now / Electoral Calculus": "Find Out Now",
    "Findoutnow": "Find Out Now",
    "FindOutNow": "Find Out Now",
    "Ipsos": "Ipsos Mori",
    "Ipsos MORI": "Ipsos Mori",
    "Kantar Public": "Kantar",
    "Kantar Public ": "Kantar",
    "Number Cruncher Politics ": "Number Cruncher Politics",
    "Opinium ": "Opinium",
    "Redfield & Wilton ": "Redfield & Wilton",
    "Redfield and Wilton Strategies": "Redfield & Wilton",
    "Savanta ComRes": "Savanta Comres",
    "Savanta ComRes ": "Savanta Comres",
    "SavantaComRes": "Savanta Comres",
    "Survation ": "Survation",
    "Techne UK ": "Techne UK",
    "TechneUK": "Techne UK",
    "TechneUK ": "Techne UK",
    "Techne": "Techne UK",
    "YouGov": "Yougov",
    "YouGov ": "Yougov",
    "Savanta": "Savanta Comres",
    "ComRes": "Comres",
    "Kantar ": "Kantar",
    "Kantar TNS": "Kantar",
    "OnePoll": "Onepoll",
    "Omnisis": "WeThink",
    "Number Cruncher": "Number Cruncher Politics",
    "TNS-BMRB": "TNS",
    "We Think": "WeThink",
}

SHEET_TO_PARLIAMENT = {
    "55-59": "1955-1959",
    "59-64": "1959-1964",
    "64-66": "1964-1966",
    "66-70": "1966-1970",
    "70-74": "1970-1974",
    "74-79": "1974-1979",
    "79-83": "1979-1983",
    "83-87": "1983-1987",
    "87-92": "1987-1992",
    "92-97": "1992-1997",
    "97-01": "1997-2001",
    "01-05": "2001-2005",
    "05-10": "2005-2010",
    "10-15": "2010-2015",
    "15-17": "2015-2017",
    "17-19": "2017-2019",
    "19-24": "2019-2024",
    "24-": "2024-2029",
}

ELECTION_DATES = {
    "1955-1959": date(1959, 10, 8),
    "1959-1964": date(1964, 10, 15),
    "1964-1966": date(1966, 3, 31),
    "1966-1970": date(1970, 6, 18),
    "1970-1974": date(1974, 2, 28),
    "1974-1979": date(1979, 5, 3),
    "1979-1983": date(1983, 6, 9),
    "1983-1987": date(1987, 6, 11),
    "1987-1992": date(1992, 4, 9),
    "1992-1997": date(1997, 5, 1),
    "1997-2001": date(2001, 6, 7),
    "2001-2005": date(2005, 5, 5),
    "2005-2010": date(2010, 5, 6),
    "2010-2015": date(2015, 5, 7),
    "2015-2017": date(2017, 6, 8),
    "2017-2019": date(2019, 12, 12),
    "2019-2024": date(2024, 7, 4),
    "2024-2029": date(2029, 8, 15),
}

GOVERNING_PARTY = {
    "1955-1959": "con",
    "1959-1964": "con",
    "1964-1966": "lab",
    "1966-1970": "lab",
    "1970-1974": "con",
    "1974-1979": "lab",
    "1979-1983": "con",
    "1983-1987": "con",
    "1987-1992": "con",
    "1992-1997": "con",
    "1997-2001": "lab",
    "2001-2005": "lab",
    "2005-2010": "lab",
    "2010-2015": "con",
    "2015-2017": "con",
    "2017-2019": "con",
    "2019-2024": "con",
    "2024-2029": "lab",
}

OPPOSITION_ELECTION_RESULT_LEAD = {
    "1955-1959": -0.056,
    "1959-1964": 0.007,
    "1964-1966": -0.061,
    "1966-1970": 0.033,
    "1970-1974": -0.007,
    "1974-1979": 0.070,
    "1979-1983": -0.148,
    "1983-1987": -0.114,
    "1987-1992": -0.075,
    "1992-1997": 0.125,
    "1997-2001": -0.090,
    "2001-2005": -0.028,
    "2005-2010": 0.071,
    "2010-2015": -0.064,
    "2015-2017": -0.023,
    "2017-2019": -0.115,
    "2019-2024": 0.100,
}

PARLIAMENTS_FOR_HISTORY = [
    "1955-1959",
    "1959-1964",
    "1964-1966",
    "1966-1970",
    "1970-1974",
    "1974-1979",
    "1979-1983",
    "1983-1987",
    "1987-1992",
    "1992-1997",
    "1997-2001",
    "2001-2005",
    "2005-2010",
    "2010-2015",
    "2015-2017",
    "2017-2019",
    "2019-2024",
]


def _date_num(ts: pd.Series) -> np.ndarray:
    return (pd.to_datetime(ts).astype("int64") // NANOS_PER_DAY).to_numpy(dtype=float)


def find_pollbase_file_url(timeout: int = 30) -> str:
    html = requests.get(POLLBASE_LANDING_PAGE, timeout=timeout).text
    match = re.search(r'href="([^"]+\.xlsx)"[^>]*>Download latest edition of PollBase here \(Excel\)', html)
    if not match:
        raise RuntimeError("Could not find PollBase .xlsx URL on landing page.")
    return match.group(1)


def ensure_pollbase_xlsx(path: Path, timeout: int = 60) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    url = find_pollbase_file_url(timeout=timeout)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def parse_pollbase(path: Path) -> pd.DataFrame:
    def _num_col(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    records: list[pd.DataFrame] = []
    xls = pd.ExcelFile(path)
    for sheet, parliament in SHEET_TO_PARLIAMENT.items():
        if sheet not in xls.sheet_names:
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        reform_series = _num_col(df, "Reform")
        reform_series = reform_series.fillna(_num_col(df, "BXP/Reform"))
        reform_series = reform_series.fillna(_num_col(df, "BXP"))
        subset = pd.DataFrame(
            {
                "mid_date": pd.to_datetime(df.get("Unnamed: 3"), errors="coerce").dt.normalize(),
                "pollster": df.get("Polling"),
                "con": pd.to_numeric(df.get("Con"), errors="coerce") / 100.0,
                "lab": pd.to_numeric(df.get("Lab"), errors="coerce") / 100.0,
                "reform": reform_series / 100.0,
                "parliament": parliament,
                "source": "pollbase",
            }
        )
        subset["pollster"] = subset["pollster"].astype(str).str.strip()
        subset = subset[subset["mid_date"].notna()]
        subset = subset[subset[["con", "lab", "reform"]].notna().any(axis=1)]
        records.append(subset)
    if not records:
        raise RuntimeError("No rows parsed from PollBase workbook.")
    return pd.concat(records, ignore_index=True)


def _infer_parliament(dt: pd.Timestamp) -> str | None:
    if pd.isna(dt):
        return None
    d = dt.date()
    if d >= date(2024, 7, 5):
        return "2024-2029"
    if d >= date(2019, 12, 13):
        return "2019-2024"
    if d >= date(2017, 6, 9):
        return "2017-2019"
    if d >= date(2015, 5, 8):
        return "2015-2017"
    if d >= date(2010, 5, 7):
        return "2010-2015"
    return None


def parse_politico() -> pd.DataFrame:
    polls_json = requests.get(POLITICO_URL, timeout=45).json().get("polls", [])
    rows: list[dict[str, object]] = []
    for poll in polls_json:
        p = poll.get("parties", {})
        con = p.get("Con")
        lab = p.get("Lab")
        reform = p.get("BP")
        if reform is None:
            reform = p.get("UKIP_2024")
        if con is None and lab is None and reform is None:
            continue
        dt = pd.to_datetime(poll.get("date"), errors="coerce").normalize()
        parliament = _infer_parliament(dt)
        if parliament is None:
            continue
        rows.append(
            {
                "mid_date": dt,
                "pollster": str(poll.get("firm", "")).strip(),
                "con": (float(con) / 100.0) if con is not None else np.nan,
                "lab": (float(lab) / 100.0) if lab is not None else np.nan,
                "reform": (float(reform) / 100.0) if reform is not None else np.nan,
                "parliament": parliament,
                "source": "politico",
            }
        )
    if not rows:
        raise RuntimeError("No rows parsed from Politico endpoint.")
    return pd.DataFrame(rows)


def combine_polls(pollbase_path: Path) -> pd.DataFrame:
    pollbase = parse_pollbase(pollbase_path)
    politico = parse_politico()

    pollbase = pollbase.copy()
    politico = politico.copy()
    pollbase["pollster"] = pollbase["pollster"].replace(POLLSTER_RENAMES).fillna("Unknown")
    politico["pollster"] = politico["pollster"].replace(POLLSTER_RENAMES).fillna("Unknown")

    # Keep Politico rows only when no PollBase row exists for same date + pollster.
    pb_keys = (
        pollbase.assign(_key=pollbase["mid_date"].astype(str) + "|" + pollbase["pollster"].astype(str))
        ["_key"]
        .drop_duplicates()
    )
    politico = politico.assign(_key=politico["mid_date"].astype(str) + "|" + politico["pollster"].astype(str))
    politico_nondup = politico[~politico["_key"].isin(set(pb_keys))].drop(columns=["_key"])

    combined = pd.concat([pollbase, politico_nondup], ignore_index=True)
    combined = combined.dropna(subset=["mid_date"]).sort_values("mid_date").reset_index(drop=True)
    return combined


def _select_subset_for_date(party_polls: pd.DataFrame, date_n: pd.Timestamp) -> pd.DataFrame:
    work = party_polls.copy()
    work["date_dist"] = (work["mid_date"] - date_n).abs().dt.days.astype(float)
    work["date_dist_rank"] = work["date_dist"].rank(method="average")

    within_month = 0.0
    vals = work.loc[work["date_dist"] < 30, "date_dist_rank"]
    if not vals.empty:
        within_month = float(vals.max())

    if within_month < 51:
        subset = work[work["date_dist_rank"] < 51].copy()
    else:
        subset = work[work["date_dist"] < 31].copy()
    return subset


def _apply_base_weights(subset: pd.DataFrame, date_n: pd.Timestamp) -> pd.DataFrame:
    pollster_counts = subset["pollster"].value_counts()
    pollster_weights = 1.0 / np.log(pollster_counts + 1.0)

    d = (subset["mid_date"] - date_n).abs().dt.days.to_numpy(dtype=float)
    max_d = float(max(np.nanmax(d), 0.0))
    date_weights = (1.0 - np.power(d / (max_d + 1.0), 3)) ** 3

    subset["wt"] = date_weights * subset["pollster"].map(pollster_weights).to_numpy(dtype=float)
    return subset


def _fit_weighted_quadratic_centered(
    x_days: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    return_interval: bool = False,
) -> tuple[float, float | None, float | None]:
    if len(x_days) < 3:
        pred = float(np.average(y, weights=w)) if np.sum(w) > 0 else float(np.mean(y))
        return pred, None, None

    X = np.column_stack([np.ones(len(x_days)), x_days, x_days**2])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    pred = float(beta[0])

    if not return_interval:
        return pred, None, None

    resid = y - X @ beta
    p = X.shape[1]
    dof = max(len(y) - p, 1)
    mse = float(np.sum(w * resid**2) / dof)
    x0 = np.array([1.0, 0.0, 0.0])
    xtwx = X.T @ (w[:, None] * X)
    inv = np.linalg.pinv(xtwx)
    h0 = float(x0 @ inv @ x0)
    se_pred = float(np.sqrt(max(mse * (1.0 + h0), 0.0)))
    lwr = pred - 1.96 * se_pred
    upr = pred + 1.96 * se_pred
    return pred, lwr, upr


def average_func(party: str, polls: pd.DataFrame, parliament: str) -> pd.DataFrame | None:
    if party not in polls.columns:
        return None
    party_polls = polls[(polls[party].notna()) & (polls["parliament"] == parliament)][["mid_date", "pollster", party]].copy()
    if len(party_polls) < 10:
        return None
    party_polls = party_polls.rename(columns={party: "pct"}).sort_values("mid_date").reset_index(drop=True)

    min_date = party_polls["mid_date"].min()
    max_date = party_polls["mid_date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")

    estimates = pd.DataFrame({"mid_date": all_dates})

    est1 = []
    for d in all_dates:
        subset = _select_subset_for_date(party_polls, d)
        subset = _apply_base_weights(subset, d)
        subset = subset[subset["wt"] > 0]
        x = (subset["mid_date"] - d).dt.days.to_numpy(dtype=float)
        y = subset["pct"].to_numpy(dtype=float)
        w = subset["wt"].to_numpy(dtype=float)
        pred, _, _ = _fit_weighted_quadratic_centered(x, y, w, return_interval=False)
        est1.append(pred)
    estimates["est1"] = est1

    est1_map = estimates.set_index("mid_date")["est1"].to_dict()
    est2 = []
    for d in all_dates:
        subset = _select_subset_for_date(party_polls, d)
        subset = _apply_base_weights(subset, d)
        subset["est1"] = subset["mid_date"].map(est1_map)
        resid = np.abs(subset["pct"] - subset["est1"]).to_numpy(dtype=float)
        max_res = np.nanmax(resid) if len(resid) else np.nan
        if np.isnan(max_res) or max_res <= 0:
            distance_weights = np.ones(len(subset), dtype=float)
        else:
            distance_weights = 1.0 - (resid / max_res)
        subset["wt"] = subset["wt"].to_numpy(dtype=float) * distance_weights
        subset = subset[subset["wt"] > 0]
        x = (subset["mid_date"] - d).dt.days.to_numpy(dtype=float)
        y = subset["pct"].to_numpy(dtype=float)
        w = subset["wt"].to_numpy(dtype=float)
        pred, _, _ = _fit_weighted_quadratic_centered(x, y, w, return_interval=False)
        est2.append(pred)
    estimates["est2"] = est2

    party_polls["est2"] = party_polls["mid_date"].map(estimates.set_index("mid_date")["est2"].to_dict())
    party_polls["resid2"] = party_polls["pct"] - party_polls["est2"]

    party_polls["house_effect"] = 0.0
    try:
        md = sm.MixedLM.from_formula("resid2 ~ 1", groups="pollster", data=party_polls)
        fit = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        party_polls["house_effect"] = fit.fittedvalues.to_numpy(dtype=float)
    except Exception:
        party_polls["house_effect"] = 0.0

    est3 = []
    for d in all_dates:
        subset = _select_subset_for_date(party_polls, d)
        subset = _apply_base_weights(subset, d)
        subset = subset[subset["wt"] > 0]
        x = (subset["mid_date"] - d).dt.days.to_numpy(dtype=float)
        y = (subset["pct"] - subset["house_effect"]).to_numpy(dtype=float)
        w = subset["wt"].to_numpy(dtype=float)
        pred, _, _ = _fit_weighted_quadratic_centered(x, y, w, return_interval=False)
        est3.append(pred)
    estimates["est3"] = est3

    est3_map = estimates.set_index("mid_date")["est3"].to_dict()
    est4 = []
    lwr = []
    upr = []
    for d in all_dates:
        subset = _select_subset_for_date(party_polls, d)
        subset = _apply_base_weights(subset, d)
        subset["est3"] = subset["mid_date"].map(est3_map)
        resid = np.abs(subset["pct"] - subset["est3"]).to_numpy(dtype=float)
        max_res = np.nanmax(resid) if len(resid) else np.nan
        if np.isnan(max_res) or max_res <= 0:
            distance_weights = np.ones(len(subset), dtype=float)
        else:
            distance_weights = 1.0 - (resid / max_res)
        subset["wt"] = subset["wt"].to_numpy(dtype=float) * distance_weights
        subset = subset[subset["wt"] > 0]

        x = (subset["mid_date"] - d).dt.days.to_numpy(dtype=float)
        y = (subset["pct"] - subset["house_effect"]).to_numpy(dtype=float)
        w = subset["wt"].to_numpy(dtype=float)
        pred, low, high = _fit_weighted_quadratic_centered(x, y, w, return_interval=True)
        est4.append(pred)
        lwr.append(low if low is not None else pred)
        upr.append(high if high is not None else pred)

    out = pd.DataFrame({"mid_date": all_dates, "avg": est4, "lwr": lwr, "upr": upr})
    out["avg"] = out["avg"].clip(0, 1)
    out["lwr"] = out["lwr"].clip(0, 1)
    out["upr"] = out["upr"].clip(0, 1)
    return out


def build_current_lead_series(
    polls: pd.DataFrame,
    numerator_party: str,
    denominator_party: str = "lab",
    parliament: str = "2024-2029",
) -> pd.DataFrame | None:
    est_num = average_func(numerator_party, polls, parliament=parliament)
    est_den = average_func(denominator_party, polls, parliament=parliament)
    if est_num is None or est_den is None:
        return None

    merged = est_num[["mid_date", "avg"]].rename(columns={"avg": "num_avg"}).merge(
        est_den[["mid_date", "avg"]].rename(columns={"avg": "den_avg"}),
        on="mid_date",
        how="inner",
    )
    if merged.empty:
        return None
    election_date = pd.Timestamp(ELECTION_DATES[parliament])
    merged["time_to_election"] = (election_date - merged["mid_date"]).dt.days.astype(int)
    merged["lead"] = merged["num_avg"] - merged["den_avg"]
    return merged.sort_values("time_to_election").reset_index(drop=True)


def build_combined_estimates(polls: pd.DataFrame) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for parliament in PARLIAMENTS_FOR_HISTORY:
        for party in ("con", "lab"):
            est = average_func(party, polls, parliament=parliament)
            if est is None:
                continue
            est["party"] = party
            est["parliament"] = parliament
            chunks.append(est)
    if not chunks:
        raise RuntimeError("No combined estimates produced.")

    out = pd.concat(chunks, ignore_index=True)
    out["election_date"] = out["parliament"].map(ELECTION_DATES)
    out["time_to_election"] = (
        pd.to_datetime(out["election_date"]) - pd.to_datetime(out["mid_date"])
    ).dt.days.astype(int)

    out["government"] = np.where(
        out["party"] == out["parliament"].map(GOVERNING_PARTY),
        1,
        0,
    )

    out["opposition_won"] = np.where(
        out["parliament"].map(OPPOSITION_ELECTION_RESULT_LEAD) > 0,
        "Opposition won",
        "Opposition lost",
    )
    return out


def build_opposition_lead_series(combined_estimates: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        combined_estimates
        .groupby(["time_to_election", "parliament", "government", "opposition_won"], as_index=False)["avg"]
        .mean()
    )

    wide = grouped.pivot_table(
        index=["time_to_election", "parliament", "opposition_won"],
        columns="government",
        values="avg",
        aggfunc="mean",
    ).reset_index()

    wide = wide.rename(columns={0: "opposition_avg", 1: "government_avg"})
    wide = wide.dropna(subset=["opposition_avg", "government_avg"])
    wide["opposition_lead"] = wide["opposition_avg"] - wide["government_avg"]
    return wide


def _local_linear_smooth(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, frac: float = 0.2) -> np.ndarray:
    n = len(x)
    if n < 10:
        order = np.argsort(x)
        return np.interp(x_grid, x[order], y[order])

    k = max(40, int(np.ceil(frac * n)))
    yhat = np.empty(len(x_grid), dtype=float)
    for i, gx in enumerate(x_grid):
        dist = np.abs(x - gx)
        idx = np.argpartition(dist, min(k, n - 1))[:k]
        xloc = x[idx]
        yloc = y[idx]
        dloc = dist[idx]
        dmax = max(float(dloc.max()), 1.0)
        w = (1.0 - np.power(dloc / dmax, 3)) ** 3
        X = np.column_stack([np.ones(len(xloc)), xloc - gx])
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], yloc * sw, rcond=None)
        yhat[i] = beta[0]
    return yhat


def plot_chart(oppo_gov: pd.DataFrame, output_path: Path, max_days: int) -> None:
    d = oppo_gov[(oppo_gov["time_to_election"] >= 0) & (oppo_gov["time_to_election"] <= max_days)].copy()

    won = d[d["opposition_won"] == "Opposition won"]
    lost = d[d["opposition_won"] == "Opposition lost"]
    if won.empty or lost.empty:
        raise RuntimeError("Need both won and lost opposition groups to draw chart.")

    x_grid = np.arange(0, max_days + 1)
    won_y = _local_linear_smooth(won["time_to_election"].to_numpy(), won["opposition_lead"].to_numpy(), x_grid)
    lost_y = _local_linear_smooth(lost["time_to_election"].to_numpy(), lost["opposition_lead"].to_numpy(), x_grid)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.plot(x_grid, won_y, color="#5DBB46", lw=2.8, label="Opposition won")
    ax.plot(x_grid, lost_y, color="#9B9B9B", lw=2.8, label="Opposition lost")

    ax.axhline(0, color="#A6A6A6", ls="--", lw=1)
    ax.grid(True, color="#DDDDDD", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(max_days, 0)
    ax.set_ylabel("Opposition lead")
    ax.set_xlabel("Days to election")
    ax.set_yticks([-0.2, 0.0, 0.2])
    ax.set_yticklabels(["-20%", "0%", "20%"])
    ax.set_xticks([1500, 1000, 500, 0] if max_days >= 1500 else [max_days, int(max_days * 2 / 3), int(max_days / 3), 0])
    ax.legend(frameon=False)
    ax.set_title("Average Opposition Polling Lead by Outcome")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_chart_with_overlay(
    oppo_gov: pd.DataFrame,
    overlay: pd.DataFrame,
    output_path: Path,
    max_days: int,
    overlay_label: str,
) -> None:
    d = oppo_gov[(oppo_gov["time_to_election"] >= 0) & (oppo_gov["time_to_election"] <= max_days)].copy()
    won = d[d["opposition_won"] == "Opposition won"]
    lost = d[d["opposition_won"] == "Opposition lost"]
    if won.empty or lost.empty:
        raise RuntimeError("Need both won and lost opposition groups to draw chart.")

    x_grid = np.arange(0, max_days + 1)
    won_y = _local_linear_smooth(won["time_to_election"].to_numpy(), won["opposition_lead"].to_numpy(), x_grid)
    lost_y = _local_linear_smooth(lost["time_to_election"].to_numpy(), lost["opposition_lead"].to_numpy(), x_grid)

    ov = overlay[(overlay["time_to_election"] >= 0) & (overlay["time_to_election"] <= max_days)].copy()
    ov = ov.sort_values("time_to_election")

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.plot(x_grid, won_y, color="#5DBB46", lw=2.8, label="Opposition won")
    ax.plot(x_grid, lost_y, color="#9B9B9B", lw=2.8, label="Opposition lost")
    if not ov.empty:
        ax.plot(ov["time_to_election"], ov["lead"], color="#E4003B", lw=2.2, label=overlay_label)

    ax.axhline(0, color="#A6A6A6", ls="--", lw=1)
    ax.grid(True, color="#DDDDDD", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(max_days, 0)
    ax.set_ylabel("Opposition lead")
    ax.set_xlabel("Days to election")
    ax.set_yticks([-0.2, 0.0, 0.2])
    ax.set_yticklabels(["-20%", "0%", "20%"])
    ax.set_xticks([1500, 1000, 500, 0] if max_days >= 1500 else [max_days, int(max_days * 2 / 3), int(max_days / 3), 0])
    ax.legend(frameon=False)
    ax.set_title("Average Opposition Polling Lead by Outcome")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="R-style UK opposition lead chart (won vs lost).")
    parser.add_argument("--pollbase-xlsx", default=str(base_dir / "PollBase-latest.xlsx"))
    parser.add_argument("--output-png", default=str(base_dir / "westminster_opposition_leads.png"))
    parser.add_argument("--output-reform-png", default=str(base_dir / "westminster_reform_opposition_lead.png"))
    parser.add_argument("--output-conservative-png", default=str(base_dir / "westminster_conservative_opposition_lead.png"))
    parser.add_argument("--output-csv", default=str(base_dir / "westminster_opposition_leads_data.csv"))
    parser.add_argument("--max-days", type=int, default=1800)
    parser.add_argument(
        "--recompute-baseline",
        action="store_true",
        help="Recompute historical won/lost baseline from raw polls instead of reusing output-csv if present.",
    )
    args = parser.parse_args()

    pollbase_path = ensure_pollbase_xlsx(Path(args.pollbase_xlsx))
    polls = combine_polls(pollbase_path)

    output_csv_path = Path(args.output_csv)
    if (not args.recompute_baseline) and output_csv_path.exists():
        try:
            cached = pd.read_csv(output_csv_path)
            required_cols = {"time_to_election", "parliament", "opposition_won", "opposition_lead"}
            if required_cols.issubset(set(cached.columns)):
                oppo_gov = cached[list(required_cols)].copy()
            else:
                raise ValueError("missing columns")
        except Exception:
            combined_estimates = build_combined_estimates(polls)
            oppo_gov = build_opposition_lead_series(combined_estimates)
    else:
        combined_estimates = build_combined_estimates(polls)
        oppo_gov = build_opposition_lead_series(combined_estimates)

    plot_chart(oppo_gov, Path(args.output_png), max_days=args.max_days)
    reform_lead = build_current_lead_series(polls, numerator_party="reform", denominator_party="lab", parliament="2024-2029")
    con_lead = build_current_lead_series(polls, numerator_party="con", denominator_party="lab", parliament="2024-2029")
    if reform_lead is not None:
        plot_chart_with_overlay(
            oppo_gov,
            reform_lead,
            Path(args.output_reform_png),
            max_days=args.max_days,
            overlay_label="Reform - Labour",
        )
    if con_lead is not None:
        plot_chart_with_overlay(
            oppo_gov,
            con_lead,
            Path(args.output_conservative_png),
            max_days=args.max_days,
            overlay_label="Conservative - Labour",
        )

    out = oppo_gov.sort_values(["parliament", "time_to_election"]).reset_index(drop=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv_path, index=False)

    print(f"Saved chart: {args.output_png}")
    if reform_lead is not None:
        print(f"Saved chart: {args.output_reform_png}")
    if con_lead is not None:
        print(f"Saved chart: {args.output_conservative_png}")
    print(f"Saved data:  {args.output_csv}")


if __name__ == "__main__":
    main()
