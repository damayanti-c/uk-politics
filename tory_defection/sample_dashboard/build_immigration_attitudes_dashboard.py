#!/usr/bin/env python
"""
Build an immigration-attitudes dashboard for MPs.

This script uses the tory_defection findings (immigration topic filter, hardline
and pro-immigration keyword sets) to score MPs and produce a Flourish-style
interactive HTML dashboard with party filtering.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Findings-derived keyword sets from analysis/final_model/test_speech_analysis.py
HARDLINE_KEYWORDS = [
    "illegal", "crisis", "flood", "flooding", "invasion", "invading", "control our borders",
    "small boats", "small boat", "channel crossing", "channel crossings", "failed", "broken system",
    "stop the boats", "take back control", "sovereignty", "mass immigration", "mass migration",
    "deport", "deportation", "detention", "detain", "remove", "removal", "echr", "rwanda scheme",
    "economic migrants", "economic migrant", "abuse", "abusing", "exploiting", "criminal gangs",
    "criminal gang", "out of control", "lost control", "unsustainable", "overwhelm", "overwhelming",
    "threat", "threatens", "prioritize british", "british first", "british people first",
    "reduce immigration", "reduce numbers", "cut immigration", "lower immigration", "cap on immigration",
    "immigration cap", "limit immigration", "immigration limit", "sustainable levels",
    "sustainable immigration", "controlled immigration", "firm but fair", "managed migration",
    "points-based", "points based", "tens of thousands", "too many", "too high", "excessive immigration",
    "burden on", "strain on", "pressure on services", "pressure on housing", "queue jump", "queue-jump",
    "unfair on", "british workers", "wage compression", "downward pressure on wages", "undercut",
]

PRO_IMMIGRATION_KEYWORDS = [
    "humanitarian", "refuge", "refugee", "compassion", "compassionate", "safe routes", "safe route",
    "international obligations", "persecution", "protection", "protect", "family reunion", "sanctuary",
    "welcome", "diversity", "multicultural", "contribution", "fleeing", "vulnerable", "human rights",
    "asylum seekers right", "legitimate", "war-torn", "displaced", "shelter",
]

IMMIGRATION_FILTER_KEYWORDS = ["immigra", "asylum", "border", "migrant", "rwanda", "boat", "channel", "refugee"]

# Known alias corrections in existing findings scripts.
NAME_ALIASES = {
    "Nus Ghani": "Nusrat Ghani",
    "Greg Stafford": "Gregory Stafford",
    "Caroline Johnson": "Dr Caroline Johnson",
    "Tom Tugendhat": "Thomas Tugendhat",
}


def escape_kw(text: str) -> str:
    return re.escape(text.strip())


def build_regex_from_keywords(keywords: list[str]) -> str:
    return "|".join(escape_kw(k) for k in keywords if k and k.strip())


def percentile_0_1(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.rank(pct=True, method="average").fillna(0)


def load_mps(election_xlsx: Path) -> pd.DataFrame:
    use_cols = ["ons_id", "constituency_name", "party_name", "firstname", "surname"]
    mps = pd.read_excel(election_xlsx, usecols=use_cols)
    mps = mps.dropna(subset=["firstname", "surname", "party_name"]).copy()
    mps["name"] = (mps["firstname"].astype(str).str.strip() + " " + mps["surname"].astype(str).str.strip()).str.replace(r"\s+", " ", regex=True)
    mps["constituency"] = mps["constituency_name"].astype(str).str.strip()
    mps["party"] = mps["party_name"].astype(str).str.strip()
    mps["ons_id"] = mps["ons_id"].astype(str).str.strip()
    mps = mps[["name", "party", "constituency", "ons_id"]].drop_duplicates()
    return mps


def build_name_mapping(mps: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {name: name for name in mps["name"].tolist()}
    for k, v in NAME_ALIASES.items():
        if k in mapping:
            mapping[v] = k
    return mapping


def compute_scores(
    mps: pd.DataFrame,
    speeches_csv: Path,
    chunksize: int,
    max_rows: int | None,
) -> pd.DataFrame:
    immigration_pattern = build_regex_from_keywords(IMMIGRATION_FILTER_KEYWORDS)
    hardline_pattern = build_regex_from_keywords(HARDLINE_KEYWORDS)
    pro_pattern = build_regex_from_keywords(PRO_IMMIGRATION_KEYWORDS)

    name_mapping = build_name_mapping(mps)
    candidate_names = set(name_mapping.keys())

    agg = {
        name: {
            "total_speeches": 0,
            "immigration_speeches": 0,
            "hardline_mentions": 0,
            "pro_mentions": 0,
        }
        for name in mps["name"].tolist()
    }

    processed = 0
    for chunk in pd.read_csv(speeches_csv, usecols=["speaker_name", "text"], chunksize=chunksize):
        if max_rows is not None and processed >= max_rows:
            break

        if max_rows is not None and processed + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - processed]

        processed += len(chunk)

        chunk = chunk.dropna(subset=["speaker_name", "text"])
        chunk = chunk[chunk["speaker_name"].isin(candidate_names)]
        if chunk.empty:
            continue

        chunk = chunk.copy()
        chunk["mp_name"] = chunk["speaker_name"].map(name_mapping)
        chunk = chunk.dropna(subset=["mp_name"])
        if chunk.empty:
            continue

        text_lower = chunk["text"].str.lower()
        immigration_mask = text_lower.str.contains(immigration_pattern, regex=True)

        chunk["is_immigration"] = immigration_mask.astype(int)
        chunk["hardline_mentions"] = text_lower.str.count(hardline_pattern)
        chunk["pro_mentions"] = text_lower.str.count(pro_pattern)

        grouped = chunk.groupby("mp_name", as_index=False).agg(
            total_speeches=("text", "size"),
            immigration_speeches=("is_immigration", "sum"),
            hardline_mentions=("hardline_mentions", "sum"),
            pro_mentions=("pro_mentions", "sum"),
        )

        for _, row in grouped.iterrows():
            mp = row["mp_name"]
            agg[mp]["total_speeches"] += int(row["total_speeches"])
            agg[mp]["immigration_speeches"] += int(row["immigration_speeches"])
            agg[mp]["hardline_mentions"] += int(row["hardline_mentions"])
            agg[mp]["pro_mentions"] += int(row["pro_mentions"])

    scores = pd.DataFrame(
        [{"name": n, **vals} for n, vals in agg.items()]
    )
    scores = mps.merge(scores, on="name", how="left")

    for c in ["total_speeches", "immigration_speeches", "hardline_mentions", "pro_mentions"]:
        scores[c] = scores[c].fillna(0)

    scores["immigration_speech_proportion"] = np.where(
        scores["total_speeches"] > 0,
        scores["immigration_speeches"] / scores["total_speeches"],
        0.0,
    )

    mention_total = scores["hardline_mentions"] + scores["pro_mentions"]
    scores["stance_balance"] = np.where(
        mention_total > 0,
        (scores["hardline_mentions"] - scores["pro_mentions"]) / mention_total,
        0.0,
    )

    salience = percentile_0_1(scores["immigration_speech_proportion"])
    stance_scaled = (scores["stance_balance"] + 1.0) / 2.0

    # Composite attitude score grounded in findings:
    # - harder line rhetoric dominates
    # - immigration focus/salience contributes secondary weight
    scores["immigration_attitude_score"] = (0.7 * stance_scaled + 0.3 * salience) * 100
    scores["immigration_attitude_score"] = scores["immigration_attitude_score"].round(1)

    scores["attitude_band"] = pd.cut(
        scores["immigration_attitude_score"],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=[
            "Strongly liberal",
            "Liberal",
            "Mixed / unclear",
            "Restriction-leaning",
            "Strongly restrictionist",
        ],
    ).astype(str)

    scores = scores.sort_values("immigration_attitude_score", ascending=False).reset_index(drop=True)
    scores.insert(0, "rank", range(1, len(scores) + 1))

    return scores


def score_from_precomputed_features(mps: pd.DataFrame, features_csv: Path) -> pd.DataFrame:
    """
    Build dashboard scores from existing speech feature outputs.

    Expected columns include:
    - name
    - immigration_speech_proportion
    - hardline_ratio (preferred)
    - reform_alignment (fallback stance proxy)
    """
    feats = pd.read_csv(features_csv)
    if "name" not in feats.columns:
        raise ValueError(f"Missing required 'name' column in {features_csv}")

    keep_cols = [c for c in ["name", "immigration_speech_proportion", "hardline_ratio", "reform_alignment"] if c in feats.columns]
    feats = feats[keep_cols].drop_duplicates(subset=["name"])

    scores = mps.merge(feats, on="name", how="left")
    scores["immigration_speech_proportion"] = scores.get("immigration_speech_proportion", 0).fillna(0.0)

    if "hardline_ratio" in scores.columns:
        stance_scaled = ((scores["hardline_ratio"].fillna(0).clip(-1, 1) + 1.0) / 2.0).clip(0, 1)
    elif "reform_alignment" in scores.columns:
        # Rank-based scaling avoids needing model-specific absolute thresholds.
        stance_scaled = percentile_0_1(scores["reform_alignment"].fillna(0))
    else:
        stance_scaled = pd.Series(0.5, index=scores.index)

    salience = percentile_0_1(scores["immigration_speech_proportion"].fillna(0))
    scores["immigration_attitude_score"] = ((0.7 * stance_scaled + 0.3 * salience) * 100).round(1)

    # Fill dashboard-compatible columns expected by the HTML renderer.
    scores["total_speeches"] = np.nan
    scores["immigration_speeches"] = np.nan
    scores["hardline_mentions"] = np.nan
    scores["pro_mentions"] = np.nan
    scores["stance_balance"] = np.nan

    scores["attitude_band"] = pd.cut(
        scores["immigration_attitude_score"],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=[
            "Strongly liberal",
            "Liberal",
            "Mixed / unclear",
            "Restriction-leaning",
            "Strongly restrictionist",
        ],
    ).astype(str)

    scores = scores.sort_values("immigration_attitude_score", ascending=False).reset_index(drop=True)
    scores.insert(0, "rank", range(1, len(scores) + 1))
    return scores


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def render_html(scores: pd.DataFrame, output_html: Path) -> None:
    records = scores[
        [
            "rank",
            "name",
            "party",
            "constituency",
            "ons_id",
            "immigration_attitude_score",
            "attitude_band",
            "total_speeches",
            "immigration_speeches",
            "immigration_speech_proportion",
            "hardline_mentions",
            "pro_mentions",
        ]
    ].to_dict(orient="records")

    data_json = json.dumps(records)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MP Immigration Attitudes Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --left: #2563eb;
      --right: #dc2626;
      --accent: #0f172a;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: radial-gradient(circle at 0 0, #eef2ff 0%, var(--bg) 45%); color: var(--ink); }}
    .wrap {{ max-width: 1200px; margin: 28px auto; padding: 0 16px; }}
    .top {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0.2px; }}
    .sub {{ color: var(--muted); margin: 0 0 16px; font-size: 14px; }}
    .controls {{ display: grid; grid-template-columns: 180px 1fr 140px; gap: 10px; align-items: center; }}
    select, input {{ width: 100%; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; font-size: 14px; background: #fff; }}
    .stats {{ display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }}
    .card {{ background: #f9fafb; border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; min-width: 180px; }}
    .card b {{ display: block; font-size: 18px; }}
    .card span {{ color: var(--muted); font-size: 12px; }}
    .tabs {{ display: flex; gap: 8px; margin-top: 12px; }}
    .tab-btn {{ border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: #fff; cursor: pointer; font-size: 13px; }}
    .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

    .chart {{ margin-top: 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    .panel.hidden {{ display: none; }}
    .axis {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 10px; margin-bottom: 12px; }}
    .axis .left {{ color: var(--left); font-weight: 600; }}
    .axis .mid {{ color: var(--muted); font-size: 12px; }}
    .axis .right {{ color: var(--right); text-align: right; font-weight: 600; }}
    .rows {{ max-height: 70vh; overflow: auto; border-top: 1px solid var(--line); }}
    .row {{ display: grid; grid-template-columns: 280px 1fr 88px; gap: 10px; align-items: center; padding: 9px 0; border-bottom: 1px solid #f1f5f9; }}
    .who {{ font-size: 13px; }}
    .who b {{ display: block; font-size: 14px; }}
    .who small {{ color: var(--muted); }}
    .track {{ position: relative; height: 12px; border-radius: 999px; background: linear-gradient(90deg, var(--left) 0%, #9ca3af 50%, var(--right) 100%); }}
    .dot {{ position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; border: 2px solid #fff; background: var(--accent); transform: translate(-50%, -50%); box-shadow: 0 2px 6px rgba(0,0,0,0.25); }}
    .score {{ text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }}
    #map {{ width: 100%; height: 72vh; min-height: 560px; }}
    .map-note {{ color: var(--muted); font-size: 12px; margin: 0 0 8px; }}

    @media (max-width: 900px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .row {{ grid-template-columns: 1fr; gap: 8px; }}
      .score {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="top">
      <h1>MP Immigration Attitudes</h1>
      <p class="sub">Score built from tory_defection findings: immigration speech focus + hardline vs compassionate rhetoric. Use party filter to compare cohorts.</p>
      <div class="controls">
        <select id="party"></select>
        <input id="search" type="search" placeholder="Search MP or constituency" />
        <select id="limit">
          <option value="50">Top 50</option>
          <option value="100" selected>Top 100</option>
          <option value="250">Top 250</option>
          <option value="10000">All</option>
        </select>
      </div>
      <div class="stats">
        <div class="card"><b id="nMps">0</b><span>MPs shown</span></div>
        <div class="card"><b id="avg">0</b><span>Average score</span></div>
        <div class="card"><b id="topBand">-</b><span>Most common band</span></div>
      </div>
      <div class="tabs">
        <button id="tabRank" class="tab-btn active" type="button">Rankings</button>
        <button id="tabMap" class="tab-btn" type="button">Constituency Map</button>
      </div>
    </section>

    <section id="panelRank" class="chart panel">
      <div class="axis">
        <div class="left">More liberal / pro-immigration</div>
        <div class="mid">Immigration attitude score (0-100)</div>
        <div class="right">More restrictionist / hardline</div>
      </div>
      <div id="rows" class="rows"></div>
    </section>
    <section id="panelMap" class="chart panel hidden">
      <p class="map-note">Constituencies are colored by MP immigration attitude score. Party filter applies to this map.</p>
      <div id="map"></div>
    </section>
  </div>

  <script>
    const DATA = {data_json};
    const UK_PCON24_GEOJSON_URL = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BSC/FeatureServer/0/query?where=1%3D1&outFields=PCON24CD,PCON24NM&outSR=4326&f=geojson";
    const partyEl = document.getElementById('party');
    const searchEl = document.getElementById('search');
    const limitEl = document.getElementById('limit');
    const rowsEl = document.getElementById('rows');
    const mapEl = document.getElementById('map');
    const panelRank = document.getElementById('panelRank');
    const panelMap = document.getElementById('panelMap');
    const tabRank = document.getElementById('tabRank');
    const tabMap = document.getElementById('tabMap');

    const nMpsEl = document.getElementById('nMps');
    const avgEl = document.getElementById('avg');
    const topBandEl = document.getElementById('topBand');
    let geojsonCache = null;

    const parties = ['All parties', ...Array.from(new Set(DATA.map(d => d.party))).sort()];
    partyEl.innerHTML = parties.map(p => `<option value="${{p}}">${{p}}</option>`).join('');

    function mode(arr) {{
      if (!arr.length) return '-';
      const counts = new Map();
      for (const x of arr) counts.set(x, (counts.get(x) || 0) + 1);
      return [...counts.entries()].sort((a,b) => b[1]-a[1])[0][0];
    }}

    function render() {{
      const party = partyEl.value;
      const q = searchEl.value.trim().toLowerCase();
      const limit = Number(limitEl.value || 100);

      let baseRows = DATA.filter(d => party === 'All parties' || d.party === party);
      if (q) {{
        baseRows = baseRows.filter(d =>
          d.name.toLowerCase().includes(q) ||
          d.constituency.toLowerCase().includes(q)
        );
      }}

      const mapRows = baseRows.slice();
      const rows = baseRows
        .sort((a, b) => b.immigration_attitude_score - a.immigration_attitude_score)
        .slice(0, limit);

      nMpsEl.textContent = String(rows.length);
      avgEl.textContent = rows.length ? (rows.reduce((s, r) => s + r.immigration_attitude_score, 0) / rows.length).toFixed(1) : '0';
      topBandEl.textContent = mode(rows.map(r => r.attitude_band));

      rowsEl.innerHTML = rows.map(r => `
        <div class="row">
          <div class="who">
            <b>${{r.name}}</b>
            <small>${{r.party}} | ${{r.constituency}}</small>
          </div>
          <div class="track" title="${{r.attitude_band}}">
            <span class="dot" style="left: ${{Math.max(0, Math.min(100, r.immigration_attitude_score))}}%"></span>
          </div>
          <div class="score">${{r.immigration_attitude_score.toFixed(1)}}</div>
        </div>
      `).join('');

      renderMap(mapRows, party);
    }}

    function setTab(tab) {{
      const isMap = tab === 'map';
      panelMap.classList.toggle('hidden', !isMap);
      panelRank.classList.toggle('hidden', isMap);
      tabMap.classList.toggle('active', isMap);
      tabRank.classList.toggle('active', !isMap);
      if (isMap) render();
    }}

    async function loadGeojson() {{
      if (geojsonCache) return geojsonCache;
      const res = await fetch(UK_PCON24_GEOJSON_URL);
      geojsonCache = await res.json();
      return geojsonCache;
    }}

    async function renderMap(filteredRows, party) {{
      if (panelMap.classList.contains('hidden')) return;
      if (typeof Plotly === 'undefined') {{
        mapEl.innerHTML = '<p style=\"color:#b91c1c\">Map library failed to load. Open via local server and ensure internet access.</p>';
        return;
      }}
      try {{
        const gj = await loadGeojson();
        const zmin = 0, zmax = 100;
        const loc = filteredRows.map(r => r.ons_id).filter(Boolean);
        const z = filteredRows.filter(r => r.ons_id).map(r => Number(r.immigration_attitude_score));
        const text = filteredRows.filter(r => r.ons_id).map(r => `${{r.name}}<br>${{r.constituency}}<br>${{r.party}}<br>Score: ${{Number(r.immigration_attitude_score).toFixed(1)}}`);

        const trace = {{
          type: 'choropleth',
          geojson: gj,
          featureidkey: 'properties.PCON24CD',
          locations: loc,
          z: z,
          text: text,
          hovertemplate: '%{{text}}<extra></extra>',
          zmin: zmin,
          zmax: zmax,
          colorscale: [
            [0.0, '#2563eb'],
            [0.5, '#9ca3af'],
            [1.0, '#dc2626']
          ],
          marker: {{line: {{color: '#ffffff', width: 0.3}}}},
          colorbar: {{title: 'Score'}}
        }};

        const layout = {{
          margin: {{l: 0, r: 0, t: 10, b: 0}},
          geo: {{
            scope: 'europe',
            projection: {{type: 'mercator'}},
            fitbounds: 'locations',
            showland: true,
            landcolor: '#f8fafc',
            bgcolor: '#ffffff',
            showcountries: false,
            showsubunits: false
          }},
          annotations: [{{
            text: party === 'All parties' ? 'All parties' : `Filtered: ${{party}}`,
            showarrow: false,
            x: 0,
            xref: 'paper',
            y: 1.05,
            yref: 'paper',
            font: {{size: 12, color: '#6b7280'}}
          }}]
        }};

        const config = {{displayModeBar: false, responsive: true}};
        Plotly.newPlot(mapEl, [trace], layout, config);
      }} catch (e) {{
        mapEl.innerHTML = `<p style=\"color:#b91c1c\">Failed to load constituency map data: ${{e}}</p>`;
      }}
    }}

    partyEl.addEventListener('change', render);
    searchEl.addEventListener('input', render);
    limitEl.addEventListener('change', render);
    tabRank.addEventListener('click', () => setTab('rank'));
    tabMap.addEventListener('click', () => setTab('map'));
    render();
  </script>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    tory_defection_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Build MP immigration attitudes dashboard")
    parser.add_argument(
        "--election-xlsx",
        type=Path,
        default=tory_defection_root / "source_data" / "elections" / "MPs-elected.xlsx",
        help="Path to election workbook with MP-party mapping",
    )
    parser.add_argument(
        "--speeches-csv",
        type=Path,
        default=tory_defection_root / "source_data" / "hansard" / "all_speeches_extended.csv",
        help="Path to Hansard speech CSV",
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=script_dir / "immigration_attitude_scores.csv",
        help="Cached per-MP score output CSV",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=script_dir / "immigration_attitudes_dashboard.html",
        help="Dashboard HTML output path",
    )
    parser.add_argument("--rebuild", action="store_true", help="Recompute scores from source data")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for speech CSV streaming")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=250_000,
        help="Cap rows processed from speeches CSV fallback (default is lightweight; use --full-scan for all)",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Process full Hansard CSV in fallback mode (can be slow)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mps = load_mps(args.election_xlsx)
    script_dir = Path(__file__).parent
    final_model_dir = script_dir.parent / "analysis" / "final_model"
    analysis_dir = final_model_dir.parent
    precomputed_candidates = [
        final_model_dir / "test_speech_features.csv",
        final_model_dir / "training_speech_features.csv",
        analysis_dir / "experiments_old" / "training_tfidf_model_old_scripts" / "enhanced_speech_tfidf_normalized.csv",
        analysis_dir / "experiments_old" / "training_tfidf_model_old_scripts" / "enhanced_speech_tfidf.csv",
    ]

    if args.scores_csv.exists() and not args.rebuild:
        scores = pd.read_csv(args.scores_csv)
        if "ons_id" not in scores.columns:
            scores = scores.merge(mps[["name", "ons_id"]], on="name", how="left")
        print(f"Loaded cached scores: {args.scores_csv}")
    else:
        precomputed = first_existing(precomputed_candidates)
        if precomputed is not None:
            print(f"Using precomputed speech features: {precomputed}")
            scores = score_from_precomputed_features(mps=mps, features_csv=precomputed)
        else:
            effective_max_rows = None if args.full_scan else args.max_rows
            if effective_max_rows is None:
                print("Computing immigration-attitude scores from full speeches CSV...")
            else:
                print(f"Computing lightweight scores from first {effective_max_rows:,} speech rows...")
            scores = compute_scores(
                mps=mps,
                speeches_csv=args.speeches_csv,
                chunksize=args.chunksize,
                max_rows=effective_max_rows,
            )
        args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.scores_csv, index=False)
        print(f"Saved scores: {args.scores_csv}")

    render_html(scores, args.output_html)
    print(f"Saved dashboard: {args.output_html}")


if __name__ == "__main__":
    main()
