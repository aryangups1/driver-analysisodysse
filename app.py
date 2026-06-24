import os
import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from config import DRIVER_NAMES, BAD_DRIVER_IDS, TOP_DRIVER_IDS
import db
from zones import enrich_zones, parse_dms, assign_zone, CENTER_LAT, CENTER_LON, GEOJSON_DATA, calc_true_rph, estimate_ping

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Odysse Driver Analysis",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#ffffff !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio > label { color:#1e1e2e !important; }
[data-testid="stSidebar"] hr { border-color:#e2e8f0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#64748b !important; }
.metric-card {
    background:#1e1e2e;border-radius:8px;padding:16px 18px;
    border-left:4px solid #f59e0b;margin-bottom:8px;
}
.metric-label{color:#9ca3af;font-size:12px;text-transform:uppercase;letter-spacing:.05em;}
.metric-value{color:#f9fafb;font-size:26px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ── DB connection health check ────────────────────────────────────────────────
# psycopg2 connections cached via @st.cache_resource go stale after server-side
# idle timeout. On every rerun, ping with SELECT 1; if the connection is closed,
# clear the resource cache so get_conn() creates a fresh one on next query.
try:
    _hc_conn = db.get_conn()
    if getattr(_hc_conn, "closed", 0) != 0:
        raise RuntimeError("connection closed")
    with _hc_conn.cursor() as _hc_cur:
        _hc_cur.execute("SELECT 1")
except Exception:
    db.get_conn.clear()

# ── Module-level constants ────────────────────────────────────────────────────
_FLOW_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_data.parquet")
_CAT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver_categories.csv")
_WEST_LON    = -0.12
_FLEET_RPH_DEFAULT    = 23.04
_FLEET_ACCEPT_DEFAULT = 57.2
_FLEET_UTIL_DEFAULT   = 68.2

CAT_COLORS = {
    "A":  "#22c55e",
    "B1": "#4ade80",
    "B2": "#60a5fa",
    "C1": "#f59e0b",
    "C2": "#fb923c",
    "D":  "#ef4444",
    None: "#94a3b8",
    "—":  "#94a3b8",
}
CAT_LABELS = {
    "A": "A — Elite", "B1": "B1 — Strong", "B2": "B2 — Solid",
    "C1": "C1 — Developing", "C2": "C2 — Below avg", "D": "D — Low performer",
    None: "Unclassified", "—": "Unclassified",
}
CAT_ORDER = [None, "D", "C2", "C1", "B2", "B1", "A"]

# ── Acceptance rate period loader (April + May 2026, full fleet) ─────────────

@st.cache_data(ttl=3600)
def _load_acceptance_period():
    return db.query("""
        SELECT dim_driver_id, driver_full_name,
               status, trip_price_in_pound,
               pickup_lat_long,
               COALESCE(pickedup_trip_datetime, trip_booking_datetime) AS event_dt,
               dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE COALESCE(pickedup_trip_datetime, trip_booking_datetime) >= '2026-04-01'
          AND COALESCE(pickedup_trip_datetime, trip_booking_datetime) <  '2026-06-01'
          AND status IN ('completed','Finished','Driver did not respond','Driver rejected')
        ORDER BY dim_driver_id, event_dt
    """, ())

# ── Fleet day loader ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_fleet_day(date_str):
    return db.query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, pickedup_trip_datetime
        FROM rep_fact_trips
        WHERE status IN ('completed','Finished')
          AND DATE(pickedup_trip_datetime) = %s
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
        ORDER BY pickedup_trip_datetime
    """, (date_str,))

# ── Zone analysis data loader ────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_zone_trips(days_back=30):
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    return db.query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               pickedup_trip_datetime, dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE status IN ('completed','Finished')
          AND pickup_lat_long  IS NOT NULL AND pickup_lat_long  != ''
          AND dropoff_latlong  IS NOT NULL AND dropoff_latlong  != ''
          AND distance_in_miles BETWEEN 0.3 AND 40
          AND pickedup_trip_datetime >= %s
        ORDER BY dim_driver_id, pickedup_trip_datetime
    """, (from_date,))

# ── Driver search (broader than db.search_drivers — covers rep_fact_trips) ───

@st.cache_data(ttl=3600)
def _search_drivers(name_fragment):
    return db.query("""
        SELECT DISTINCT dim_driver_id, MAX(driver_full_name) AS driver_full_name
        FROM rep_fact_trips
        WHERE UPPER(driver_full_name) LIKE UPPER(%s)
          AND status IN ('completed','Finished')
          AND driver_full_name IS NOT NULL AND driver_full_name <> ''
        GROUP BY dim_driver_id
        ORDER BY driver_full_name
        LIMIT 30
    """, (f"%{name_fragment}%",))

# ── BYOC helpers ─────────────────────────────────────────────────────────────
_BYOC_RPH_THRESHOLD = 21.0

_BYOC_SEARCH_KEYS = [
    ("Mohammed Saiful Islam",     "Saiful"),
    ("Akeame Plummer",             "Plummer"),
    ("MD Mamunur Rashid Mahmood", "Mamunur"),
    ("Emmanuel Kazi Kakai",       "Kakai"),
    ("Aadil Bouhlaoui",           "Bouhlaoui"),
    ("Ephrem Fkadu",              "Fkadu"),
    ("Amjad Ali Jaffery",         "Jaffery"),
    ("MD Abdul Quayum",           "Quayum"),
    ("MD Parvej Alom Khan",       "Parvej"),
    ("Imran Naeem",               "Naeem"),
]

@st.cache_data(ttl=3600)
def _byoc_daily_perf(driver_ids):
    return db.query("""
        SELECT dim_driver_id,
               MAX(driver_full_name)                                                AS driver_name,
               driver_performance_date,
               SUM(number_of_finished_rides)                                        AS rides,
               ROUND(SUM(online_time_in_hrs)::numeric, 2)                          AS online_hrs,
               ROUND(SUM(revenue)::numeric, 2)                                     AS revenue,
               ROUND((SUM(revenue)/NULLIF(SUM(online_time_in_hrs),0))::numeric,2)  AS rph,
               ROUND(AVG(utilisation_percent)::numeric, 1)                         AS utilisation,
               ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric, 1)     AS acceptance
        FROM rep_fact_driver_performance
        WHERE dim_driver_id = ANY(%s) AND online_time_in_hrs > 0
        GROUP BY dim_driver_id, driver_performance_date
        ORDER BY dim_driver_id, driver_performance_date
    """, (list(driver_ids),))

# ── Behavioral analysis loaders (TOP + BAD driver IDs only) ──────────────────

@st.cache_data(ttl=3600)
def _load_behavior_accepted(days_back=30):
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_ids = list(TOP_DRIVER_IDS + BAD_DRIVER_IDS)
    return db.query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               pickedup_trip_datetime, dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE status IN ('completed','Finished')
          AND dim_driver_id = ANY(%s)
          AND pickedup_trip_datetime >= %s
        ORDER BY dim_driver_id, pickedup_trip_datetime
    """, (all_ids, from_date))

@st.cache_data(ttl=3600)
def _load_behavior_declines(days_back=30):
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_ids = list(TOP_DRIVER_IDS + BAD_DRIVER_IDS)
    return db.query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound,
               COALESCE(trip_booking_datetime, pickedup_trip_datetime) AS event_datetime
        FROM rep_fact_trips
        WHERE status IN ('Driver did not respond','Driver rejected')
          AND dim_driver_id = ANY(%s)
          AND COALESCE(trip_booking_datetime, pickedup_trip_datetime) >= %s
        ORDER BY dim_driver_id, event_datetime
    """, (all_ids, from_date))

# ── Shared helpers ────────────────────────────────────────────────────────────

def _stat_card_html(title, rph, accept, util, color, rph_d=None, accept_d=None, util_d=None):
    def _delta(val, inverse=False):
        if val is None:
            return ""
        c = ("#22c55e" if val > 0 else "#ef4444") if not inverse else ("#ef4444" if val > 0 else "#22c55e")
        sign = "+" if val > 0 else ""
        return f'<span style="color:{c};font-size:11px;margin-left:6px;">{sign}{val:.1f}</span>'
    rows = [
        ("Revenue / hr",    f"£{rph:.2f}",   _delta(rph_d)),
        ("Acceptance rate", f"{accept:.1f}%", _delta(accept_d, inverse=True)),
        ("Utilisation",     f"{util:.1f}%",   _delta(util_d)),
    ]
    trs = "".join(
        f'<tr><td style="padding:5px 0;color:#9ca3af;font-size:12px;">{lbl}</td>'
        f'<td style="text-align:right;font-size:17px;font-weight:700;color:#f9fafb;">{val}</td>'
        f'<td style="text-align:right;white-space:nowrap;">{delta}</td></tr>'
        for lbl, val, delta in rows
    )
    return (
        f'<div style="background:#1e1e2e;border-top:3px solid {color};border-radius:8px;padding:18px 20px;">'
        f'<div style="color:{color};font-size:13px;font-weight:700;margin-bottom:12px;">{title}</div>'
        f'<table style="width:100%;border-collapse:collapse;">{trs}</table>'
        f'</div>'
    )


def _ff_zone_matrix(flow_raw):
    df = enrich_zones(flow_raw[flow_raw["status"].isin(["completed", "Finished"])].copy())
    df = df.dropna(subset=["pickup_zone", "dropoff_zone"])
    df["pickup_zone"]  = df["pickup_zone"].astype(int)
    df["dropoff_zone"] = df["dropoff_zone"].astype(int)
    mat = df.groupby(["pickup_zone", "dropoff_zone"]).size().reset_index(name="trips")
    total = mat["trips"].sum()
    mat["pct"] = (mat["trips"] / total * 100).round(1)
    pivot = mat.pivot(index="pickup_zone", columns="dropoff_zone", values="pct").fillna(0)
    for z in range(1, 7):
        if z not in pivot.index:   pivot.loc[z] = 0
        if z not in pivot.columns: pivot[z]     = 0
    return pivot.sort_index()[sorted(pivot.columns)]


def _safe_cell(mat, r, c):
    try:
        return mat.loc[r, c]
    except Exception:
        return 0.0


def _flow_west_pct(flow_df):
    df = flow_df[flow_df["status"].isin(["completed", "Finished"])].copy()
    if df.empty:
        return 0.0
    coords = df["pickup_lat_long"].apply(parse_dms)
    lons = pd.Series([c[1] for c in coords]).dropna()
    return (lons < _WEST_LON).mean() * 100 if len(lons) else 0.0


def _compute_gaps(df):
    if df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["pickedup_trip_datetime"] = pd.to_datetime(df["pickedup_trip_datetime"])
    df["dropoff_trip_datetime"]  = pd.to_datetime(df["dropoff_trip_datetime"])
    df = df.sort_values(["dim_driver_id", "pickedup_trip_datetime"])
    df["prev_drop"] = df.groupby("dim_driver_id")["dropoff_trip_datetime"].shift(1)
    gaps = (df["pickedup_trip_datetime"] - df["prev_drop"]).dt.total_seconds().div(60).clip(lower=0)
    return gaps.dropna()


def _gap_buckets(s):
    s = s[s > 0]
    if len(s) == 0:
        return {"<25m": 0, "25-75m": 0, ">75m": 0, "median": 0}
    return {
        "<25m":   (s < 25).mean() * 100,
        "25-75m": ((s >= 25) & (s <= 75)).mean() * 100,
        ">75m":   (s > 75).mean() * 100,
        "median": s.median(),
    }


def _ew_parse_and_flag(flow_df):
    df = flow_df[flow_df["status"].isin(["completed", "Finished"])].copy()
    if df.empty:
        return df
    coords = df["pickup_lat_long"].apply(parse_dms)
    df["_plon"] = [c[1] for c in coords]
    df = df.dropna(subset=["_plon"])
    df["is_west"] = df["_plon"] < _WEST_LON
    return df


def _ping_stats(acc_df, dec_df, n_drivers):
    _driver_days = max(n_drivers * 14, 1)
    n_acc   = len(acc_df)
    n_dec   = len(dec_df) if not dec_df.empty else 0
    n_total = n_acc + n_dec

    lons = []
    if not acc_df.empty and "pickup_lat_long" in acc_df.columns:
        c = acc_df["pickup_lat_long"].apply(parse_dms)
        lons += [x[1] for x in c if x[1] is not None]
    if not dec_df.empty and "pickup_lat_long" in dec_df.columns:
        c = dec_df["pickup_lat_long"].apply(parse_dms)
        lons += [x[1] for x in c if x[1] is not None]
    lons_s = pd.Series(lons).dropna()
    west_pct = round((lons_s < _WEST_LON).mean() * 100, 1) if len(lons_s) else 0

    if not dec_df.empty and "trip_price_in_pound" in dec_df.columns:
        fares = dec_df["trip_price_in_pound"].dropna()
        fares = fares[fares > 0]
        dec_avg  = round(fares.mean(), 2) if len(fares) else 0
        dec_sub10 = round((fares < 10).mean() * 100, 1) if len(fares) else 0
        dec_30p   = round((fares >= 30).mean() * 100, 1) if len(fares) else 0
    else:
        dec_avg = dec_sub10 = dec_30p = 0

    return {
        "n_total": n_total, "n_acc": n_acc, "n_dec": n_dec,
        "pings_per_dd": round(n_total / _driver_days, 1),
        "acc_per_dd":   round(n_acc   / _driver_days, 1),
        "west_pings_pct": west_pct,
        "dec_avg_fare": dec_avg, "dec_sub10_pct": dec_sub10, "dec_30p_pct": dec_30p,
        "accept_rate": round(n_acc / max(n_total, 1) * 100, 1),
    }


def _scorecard(col, label, color, rph, acc, fare, sub10, west, med_gap, gap_short):
    rows = [
        ("RPH",             f"£{rph:.2f}/hr"),
        ("Acceptance rate", f"{acc:.0f}%"),
        ("Avg fare",        f"£{fare:.2f}"),
        ("Sub-£10 trips",   f"{sub10:.0f}%"),
        ("West positioning",f"{west:.0f}%"),
        ("Median gap",      f"{med_gap:.0f} min"),
        ("Gaps < 25 min",   f"{gap_short:.0f}%"),
    ]
    body = "".join(
        f'<tr><td style="padding:4px 0;color:#94a3b8;font-size:12px;">{k}</td>'
        f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
        for k, v in rows
    )
    col.markdown(
        f'<div style="background:#1e1e2e;border:2px solid {color};border-radius:8px;padding:16px 18px;">'
        f'<div style="color:{color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:6px;">{label}</div>'
        f'<table style="width:100%;border-collapse:collapse;">{body}</table>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚖 Driver Analysis")
    st.caption("Odysse Fleet Intelligence")
    st.divider()
    page = st.radio("View", [
        "Final Findings",
        "Fleet Map",
        "Driver Day",
        "Gap Analysis",
        "Zone Analysis",
        "Driver Behavior",
        "Acceptance Rate",
        "BYOC",
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Final Findings":
    st.title("Final Findings")
    st.caption("Key conclusions from the Odysse fleet driver performance analysis · Jan–Jun 2026")

    with st.spinner("Loading performance data..."):
        _ff_perf     = db.load_overview()
        _ff_perf["display_name"] = _ff_perf["dim_driver_id"].map(DRIVER_NAMES).fillna(_ff_perf["driver_name"])
        _ff_comp     = db.load_comparison_performance(list(BAD_DRIVER_IDS))
        _ff_baseline = db.load_fleet_baseline_excluding(list(TOP_DRIVER_IDS) + list(BAD_DRIVER_IDS))

    _ff_top10_rph    = _ff_perf["rph"].mean()
    _ff_top10_accept = _ff_perf["avg_acceptance"].mean()
    _ff_top10_util   = _ff_perf["avg_util"].mean()
    _ff_comp_rph     = _ff_comp["rph"].mean()
    _ff_comp_accept  = _ff_comp["acceptance"].mean()
    _ff_comp_util    = _ff_comp["utilisation"].mean()

    if not _ff_baseline.empty and _ff_baseline.iloc[0]["fleet_rph"] is not None:
        _FLEET_RPH    = float(_ff_baseline.iloc[0]["fleet_rph"])
        _FLEET_UTIL   = float(_ff_baseline.iloc[0]["fleet_util"])
        _FLEET_ACCEPT = float(_ff_baseline.iloc[0]["fleet_accept"])
    else:
        _FLEET_RPH, _FLEET_UTIL, _FLEET_ACCEPT = _FLEET_RPH_DEFAULT, _FLEET_UTIL_DEFAULT, _FLEET_ACCEPT_DEFAULT

    _fleet_driver_count = int(_ff_baseline.iloc[0]["driver_count"]) if not _ff_baseline.empty else 0

    # ── SECTION 1: The performance gap ───────────────────────────────────────
    st.subheader("1 — The performance gap")
    st.caption(
        f"Rest of fleet ({_fleet_driver_count} drivers, excl. top 10 + comparison) "
        f"vs top 10 performers vs 5 lowest-performing comparison drivers."
    )

    _hc1, _hc2, _hc3 = st.columns(3)
    with _hc1:
        st.markdown(_stat_card_html("Rest of Fleet", _FLEET_RPH, _FLEET_ACCEPT, _FLEET_UTIL, "#94a3b8"),
                    unsafe_allow_html=True)
    with _hc2:
        st.markdown(_stat_card_html(
            "Top 10 Drivers", _ff_top10_rph, _ff_top10_accept, _ff_top10_util, "#22c55e",
            rph_d=_ff_top10_rph - _FLEET_RPH,
            accept_d=_ff_top10_accept - _FLEET_ACCEPT,
            util_d=_ff_top10_util - _FLEET_UTIL,
        ), unsafe_allow_html=True)
    with _hc3:
        st.markdown(_stat_card_html(
            "Comparison Drivers", _ff_comp_rph, _ff_comp_accept, _ff_comp_util, "#ef4444",
            rph_d=_ff_comp_rph - _FLEET_RPH,
            accept_d=_ff_comp_accept - _FLEET_ACCEPT,
            util_d=_ff_comp_util - _FLEET_UTIL,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _rph_gap   = _ff_top10_rph - _ff_comp_rph
    _daily_gap = _rph_gap * 9
    _weekly_gap = _daily_gap * 5
    st.info(
        f"📌 Top 10 earn **£{_rph_gap:.2f}/hr more** than comparison drivers. "
        f"Over a 9-hour shift that's **£{_daily_gap:.0f}/day** — roughly **£{_weekly_gap:.0f}/week** per driver."
    )

    # ── SECTION 2: Selectivity beats volume ──────────────────────────────────
    st.divider()
    st.subheader("2 — Selectivity beats volume")

    _s2a, _s2b = st.columns([3, 2])
    with _s2a:
        st.markdown(
            '<div style="background:#1e1e2e;border-left:4px solid #6366f1;padding:16px 18px;'
            'border-radius:6px;color:#e2e8f0;">'
            '<strong style="font-size:15px;">The counterintuitive result:</strong><br><br>'
            'Top 10 drivers <strong>accept fewer pings</strong> than the fleet average — yet earn '
            'significantly more per hour. On Bolt, drivers see estimated <strong>fare and destination '
            'before accepting</strong>. The top 10 decline roughly 63% of all pings. They treat the '
            'platform as a curated feed, not first-come-first-served. Comparison drivers do the '
            'opposite: higher acceptance, lower RPH, more trips on the clock — and worse earnings.'
            '</div>',
            unsafe_allow_html=True,
        )
    with _s2b:
        _acc_df = pd.DataFrame({
            "Group":    ["Fleet avg", "Top 10", "Comparison"],
            "Accept %": [_FLEET_ACCEPT, _ff_top10_accept, _ff_comp_accept],
        })
        _fig_acc = px.bar(
            _acc_df, x="Group", y="Accept %",
            color="Group",
            color_discrete_map={"Fleet avg": "#94a3b8", "Top 10": "#22c55e", "Comparison": "#ef4444"},
            text="Accept %",
            title="Acceptance rate comparison",
            height=280,
        )
        _fig_acc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_acc.update_layout(showlegend=False, yaxis_ticksuffix="%", yaxis_range=[0, 80], margin=dict(t=40, b=10))
        st.plotly_chart(_fig_acc, use_container_width=True)

    # ── SECTION 3: West positioning ───────────────────────────────────────────
    st.divider()
    st.subheader("3 — Where you end up matters more than where you start")

    _s3a, _s3b = st.columns([2, 3])
    with _s3a:
        _west_outliers = pd.DataFrame([
            {"Driver": "Plummer (Cat D)", "West %": 92.6, "Group": "Cat D"},
            {"Driver": "Abdi (Cat A)",    "West %": 42.6, "Group": "Cat A"},
            {"Driver": "Mukhtar (Cat A)", "West %": 33.8, "Group": "Cat A"},
            {"Driver": "Yousuf (Cat A)",  "West %": 25.7, "Group": "Cat A"},
        ])
        _fig_wo = px.bar(
            _west_outliers, x="West %", y="Driver", orientation="h",
            color="Group",
            color_discrete_map={"Cat A": "#22c55e", "Cat D": "#ef4444"},
            text="West %",
            title="West pickup % — 4 outlier drivers",
            height=260,
        )
        _fig_wo.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                          annotation_text="50%", annotation_position="top")
        _fig_wo.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        _fig_wo.update_layout(xaxis_range=[0, 110], legend=dict(orientation="h", y=1.15), margin=dict(t=55, b=10))
        st.plotly_chart(_fig_wo, use_container_width=True)
    with _s3b:
        st.markdown(
            '<div style="background:#1e1e2e;border-left:4px solid #facc15;padding:16px 18px;'
            'border-radius:6px;color:#e2e8f0;">'
            '<strong>West positioning is useful — but not the whole story.</strong><br><br>'
            'The west corridor (Mayfair / Kensington / Chelsea / Knightsbridge) generates higher-value '
            'pings and more of them. A driver idle in Hackney will wait longer and earn less than one '
            'in South Kensington — not because of decisions made during the gap, but because of '
            '<em>where the previous trip left them</em>.<br><br>'
            'However, four outlier drivers break the rule: <strong>three Cat A drivers operating '
            'predominantly east, and one Cat D driver operating 93% west</strong>. West positioning '
            'is an advantage, not a guarantee. What converts it is selectivity.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── SECTION 4: Zone 3 trap ────────────────────────────────────────────────
    st.divider()
    st.subheader("4 — Zone 3 daytime is a gravity well")

    _s4a, _s4b = st.columns([5, 4])
    with _s4a:
        _zone_tbl = pd.DataFrame([
            {"Zone": "Zone 1",             "True RPH": "~£21/hr",   "Avg wait": "6.7 min", "Verdict": "✅ Best daytime positioning"},
            {"Zone": "Zone 2",             "True RPH": "~£19/hr",   "Avg wait": "11 min",  "Verdict": "✅ Solid"},
            {"Zone": "Zone 3 (daytime)",   "True RPH": "~£17/hr",   "Avg wait": "32 min",  "Verdict": "❌ Avoid 09:00–17:00"},
            {"Zone": "Zone 3 (night 00–06)","True RPH": "£32–43/hr","Avg wait": "—",       "Verdict": "✅ Valid for night shift"},
            {"Zone": "Zone 6",             "True RPH": "~£23/hr",   "Avg wait": "15 min",  "Verdict": "⚠️ Long drop — factor return cost"},
        ])
        st.dataframe(_zone_tbl, use_container_width=True, hide_index=True)
        st.caption("True RPH = fare ÷ (inter-trip gap + pickup time + ride time). Zone 3 loses to Zone 1 almost entirely because of the 32-min avg wait.")
    with _s4b:
        st.markdown(
            '<div style="background:#1e1e2e;border-left:4px solid #ef4444;padding:16px 18px;'
            'border-radius:6px;color:#e2e8f0;">'
            '<strong>Once in Zone 3, drivers tend to stay.</strong><br><br>'
            'Comparison drivers chain <strong>Z3→Z3 at higher rates</strong> than top 10 drivers. '
            'After a Zone 3 drop-off, the next ping comes from a nearby Zone 3 pickup. There\'s no '
            'natural exit without deliberately declining pings that would keep you there.<br><br>'
            '<strong>Zone 3 at night (00:00–06:00) is different:</strong> longer trips heading into '
            'the city, £32–43/hr, with a 03:00 peak of £43/hr. The early-shift archetype '
            '(Marius, Abdi) exploits this deliberately.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── SECTION 5: What separates the categories ──────────────────────────────
    st.divider()
    st.subheader("5 — What separates the categories")
    st.caption("Cat A through D are not just rankings — they reflect distinct behavioural patterns visible consistently across the fleet.")

    _cat_cards = [
        {
            "cat": "A — Elite", "color": "#22c55e",
            "headline": "High selectivity + premium positioning",
            "bullets": [
                "Lowest acceptance rates in the fleet — but highest RPH",
                "Zone 1 & 2 pickup dominance; minimal unproductive Zone 3 time",
                "West of Charing Cross during peak hours; no blind east stays",
                "Long-haul trip bias: accepting Zone 5/6 dropoffs, filtering £10 shorts",
                "Short gaps — utilisation high because wait time is spent in high-ping areas",
            ],
        },
        {
            "cat": "B1/B2 — Strong/Solid", "color": "#60a5fa",
            "headline": "Consistent fundamentals, room to grow on selectivity",
            "bullets": [
                "Good west positioning (50–65% typically) and solid Zone 1 presence",
                "Acceptance rate near fleet average (~50–60%) — not yet filtering aggressively",
                "Some Zone 3 drift during off-peak hours that a Cat A avoids",
                "RPH above fleet average, but the gap to Cat A lives in per-trip filtering",
                "Shift discipline solid — tend to avoid the lowest-value early morning hours",
            ],
        },
        {
            "cat": "C1/C2 — Developing/Below avg", "color": "#f59e0b",
            "headline": "Mixed positioning, inconsistent selectivity",
            "bullets": [
                "West % variable — well-positioned some days, drifting east on others",
                "Higher Zone 3 pickup share; more sub-£10 local hops accepted",
                "Accept rate elevated — not declining the low-value pings that dilute RPH",
                "Longer median inter-trip gaps — wait time not spent repositioning",
                "Pattern: Zone 3 drop → accept nearby cheap ping → stuck in Zone 3 all shift",
            ],
        },
        {
            "cat": "D — Low performer", "color": "#ef4444",
            "headline": "Position or selectivity (or both) broken",
            "bullets": [
                "Either stuck in Zone 3/4 daytime or poorly positioned in outer east",
                "High acceptance rate does not translate to earnings — accepting the noise",
                "Sub-£10 trips making up 30–50%+ of accepted rides",
                "Zone 3→Zone 3 chaining: one outer drop leads to the next, shift after shift",
                "Key insight: some Cat D drivers are geographically well-placed but selectivity is broken",
            ],
        },
    ]

    for _card in _cat_cards:
        st.markdown(
            f'<div style="background:#1e1e2e;border-left:4px solid {_card["color"]};'
            f'padding:14px 18px;border-radius:6px;color:#e2e8f0;margin-bottom:10px;">'
            f'<div style="color:{_card["color"]};font-size:12px;font-weight:bold;letter-spacing:1px;">{_card["cat"]}</div>'
            f'<div style="font-size:14px;font-weight:600;color:#f8fafc;margin:4px 0 10px 0;">{_card["headline"]}</div>'
            f'{"".join(f"""<div style=\'font-size:12px;line-height:1.9;\'>· {b}</div>""" for b in _card["bullets"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="background:#1e1e2e;border-left:4px solid #6366f1;padding:14px 16px;'
        'border-radius:6px;color:#e2e8f0;">'
        'The core lever is the same across all categories: <strong>lower your acceptance threshold '
        'and reposition before going available after a Zone 3 drop.</strong> Cat A drivers do both '
        'consistently. Cat B does one of the two. Cat C/D do neither reliably.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 6: Three groups, three realities ──────────────────────────────
    st.divider()
    st.subheader("6 — Three groups, three realities")
    st.caption("Top 10 elite drivers vs 5 specific comparison drivers vs the rest of the fleet — zone flow, gaps, positioning, fare quality side by side.")

    with st.spinner("Loading fleet driver IDs..."):
        _all_fleet_ids = db.load_fleet_driver_ids(days_back=30)
        _top10_set = set(TOP_DRIVER_IDS)
        _cmp_set   = set(BAD_DRIVER_IDS)
        _rest_ids  = [i for i in _all_fleet_ids if i not in _top10_set and i not in _cmp_set]

    with st.spinner("Loading performance data for all groups..."):
        _perf_top  = db.load_comparison_performance(list(TOP_DRIVER_IDS))
        _perf_cmp  = db.load_comparison_performance(list(BAD_DRIVER_IDS))
        _perf_rest = db.load_comparison_performance(_rest_ids)

    with st.spinner("Loading zone flow (30 days)..."):
        _flow_top  = db.load_comparison_flow(list(TOP_DRIVER_IDS), days_back=30)
        _flow_cmp  = db.load_comparison_flow(list(BAD_DRIVER_IDS), days_back=30)
        _flow_rest = db.load_comparison_flow(_rest_ids, days_back=30)

    with st.spinner("Enriching zone matrices..."):
        _mat_top  = _ff_zone_matrix(_flow_top)
        _mat_cmp  = _ff_zone_matrix(_flow_cmp)
        _mat_rest = _ff_zone_matrix(_flow_rest)

    with st.spinner("Loading gap data (14 days)..."):
        _gaps_top  = db.load_gap_accepted(list(TOP_DRIVER_IDS), days_back=14)
        _gaps_cmp  = db.load_gap_accepted(list(BAD_DRIVER_IDS), days_back=14)
        _gaps_rest = db.load_gap_accepted(_rest_ids, days_back=14)

    with st.spinner("Loading declined ping data..."):
        _dec_top  = db.load_gap_declined(list(TOP_DRIVER_IDS), days_back=14)
        _dec_cmp  = db.load_gap_declined(list(BAD_DRIVER_IDS), days_back=14)
        _dec_rest = db.load_gap_declined(_rest_ids, days_back=14)

    # Aggregate stats
    _rph_top  = _perf_top["rph"].mean()  if not _perf_top.empty  else 0
    _rph_cmp  = _perf_cmp["rph"].mean()  if not _perf_cmp.empty  else 0
    _rph_rest = _FLEET_RPH

    _acc_top  = _perf_top["acceptance"].mean()  if not _perf_top.empty  else 0
    _acc_cmp  = _perf_cmp["acceptance"].mean()  if not _perf_cmp.empty  else 0
    _acc_rest = _FLEET_ACCEPT

    def _avg_fare(flow):
        f = flow[flow["status"].isin(["completed", "Finished"])]["trip_price_in_pound"] if not flow.empty else pd.Series()
        return f.mean() if len(f) else 0

    def _sub10_pct(flow):
        f = flow[flow["status"].isin(["completed", "Finished"])]["trip_price_in_pound"] if not flow.empty else pd.Series()
        return (f < 10).mean() * 100 if len(f) else 0

    _fare_top  = _avg_fare(_flow_top)
    _fare_cmp  = _avg_fare(_flow_cmp)
    _fare_rest = _avg_fare(_flow_rest)
    _sub10_top  = _sub10_pct(_flow_top)
    _sub10_cmp  = _sub10_pct(_flow_cmp)
    _sub10_rest = _sub10_pct(_flow_rest)
    _west_top  = _flow_west_pct(_flow_top)
    _west_cmp  = _flow_west_pct(_flow_cmp)
    _west_rest = _flow_west_pct(_flow_rest)

    _gb_top  = _gap_buckets(_compute_gaps(_gaps_top))
    _gb_cmp  = _gap_buckets(_compute_gaps(_gaps_cmp))
    _gb_rest = _gap_buckets(_compute_gaps(_gaps_rest))

    # Scorecards
    st.markdown("#### At a glance — six metrics, three groups")
    _sc1, _sc2, _sc3 = st.columns(3)
    _scorecard(_sc1, "TOP 10 ELITE",         "#22c55e", _rph_top,  _acc_top,  _fare_top,  _sub10_top,  _west_top,  _gb_top["median"],  _gb_top["<25m"])
    _scorecard(_sc2, "REST OF FLEET",        "#94a3b8", _rph_rest, _acc_rest, _fare_rest, _sub10_rest, _west_rest, _gb_rest["median"], _gb_rest["<25m"])
    _scorecard(_sc3, "COMPARISON (WORST 5)", "#ef4444", _rph_cmp,  _acc_cmp,  _fare_cmp,  _sub10_cmp,  _west_cmp,  _gb_cmp["median"],  _gb_cmp["<25m"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Zone heatmaps
    st.markdown("#### Zone flow: Top 10 vs Comparison (extreme ends)")
    st.caption("Last 30 days accepted trips — % of trips on each pickup→dropoff route")
    _hm1, _hm2 = st.columns(2)
    with _hm1:
        _fig_hm_top = px.imshow(_mat_top, text_auto=".1f", color_continuous_scale="Blues",
                                labels=dict(x="Dropoff zone", y="Pickup zone", color="% trips"),
                                title="Top 10 — zone flow (%)", aspect="equal")
        _fig_hm_top.update_layout(height=360)
        st.plotly_chart(_fig_hm_top, use_container_width=True)
    with _hm2:
        _fig_hm_cmp = px.imshow(_mat_cmp, text_auto=".1f", color_continuous_scale="Reds",
                                labels=dict(x="Dropoff zone", y="Pickup zone", color="% trips"),
                                title="Comparison — zone flow (%)", aspect="equal")
        _fig_hm_cmp.update_layout(height=360)
        st.plotly_chart(_fig_hm_cmp, use_container_width=True)

    # Z1 vs Z3 bar
    _z1p = lambda m: m.loc[1].sum() if 1 in m.index else 0
    _z3p = lambda m: m.loc[3].sum() if 3 in m.index else 0
    _z3c = lambda m: _safe_cell(m, 3, 3)
    _z1c = lambda m: _safe_cell(m, 1, 1)
    _ff_zone_cmp = pd.DataFrame([
        {"Zone": "Z1 pickup",   "Top 10": _z1p(_mat_top), "Rest of fleet": _z1p(_mat_rest), "Comparison": _z1p(_mat_cmp)},
        {"Zone": "Z1→Z1 chain", "Top 10": _z1c(_mat_top), "Rest of fleet": _z1c(_mat_rest), "Comparison": _z1c(_mat_cmp)},
        {"Zone": "Z3 pickup",   "Top 10": _z3p(_mat_top), "Rest of fleet": _z3p(_mat_rest), "Comparison": _z3p(_mat_cmp)},
        {"Zone": "Z3→Z3 chain", "Top 10": _z3c(_mat_top), "Rest of fleet": _z3c(_mat_rest), "Comparison": _z3c(_mat_cmp)},
    ])
    _fig_zcmp = px.bar(
        _ff_zone_cmp.melt(id_vars="Zone", var_name="Group", value_name="% of trips"),
        x="Zone", y="% of trips", color="Group", barmode="group",
        color_discrete_map={"Top 10": "#22c55e", "Rest of fleet": "#94a3b8", "Comparison": "#ef4444"},
        category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
        text_auto=".1f", title="Zone 1 vs Zone 3 — all three groups", height=360,
    )
    _fig_zcmp.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(_fig_zcmp, use_container_width=True)

    # Gap distribution
    st.markdown("#### Inter-trip gap distribution — all three groups (last 14 days)")
    st.caption("<25 min = productive flow · 25–75 min = stranded · >75 min = break or app off")
    _gap_df = pd.DataFrame([
        {"Group": "Top 10",        "<25m": _gb_top["<25m"],  "25–75m": _gb_top["25–75m"],  ">75m": _gb_top[">75m"]},
        {"Group": "Rest of fleet", "<25m": _gb_rest["<25m"], "25–75m": _gb_rest["25–75m"], ">75m": _gb_rest[">75m"]},
        {"Group": "Comparison",    "<25m": _gb_cmp["<25m"],  "25–75m": _gb_cmp["25–75m"],  ">75m": _gb_cmp[">75m"]},
    ]).melt(id_vars="Group", var_name="Bucket", value_name="% of gaps")
    _fig_gaps = px.bar(
        _gap_df, x="Group", y="% of gaps", color="Bucket", barmode="stack",
        color_discrete_map={"<25m": "#22c55e", "25–75m": "#f59e0b", ">75m": "#ef4444"},
        category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
        title="Gap distribution — Top 10 vs Rest vs Comparison", height=340,
    )
    _fig_gaps.update_layout(yaxis_ticksuffix="%", legend_title="Gap bucket")
    st.plotly_chart(_fig_gaps, use_container_width=True)

    # East/West fare quality
    st.markdown("#### East vs West fare quality — all three groups (last 30 days)")
    with st.spinner("Computing east/west fare split..."):
        _ew_top  = _ew_parse_and_flag(_flow_top)
        _ew_cmp  = _ew_parse_and_flag(_flow_cmp)
        _ew_rest = _ew_parse_and_flag(_flow_rest)

    _ew_rows = []
    for _grp_label, _grp_df in [("Top 10", _ew_top), ("Rest of fleet", _ew_rest), ("Comparison", _ew_cmp)]:
        for _side, _side_label in [(True, "West (<−0.12°)"), (False, "East (≥−0.12°)")]:
            _sg = _grp_df[_grp_df["is_west"] == _side] if not _grp_df.empty else pd.DataFrame()
            if len(_sg) == 0:
                continue
            _ew_rows.append({
                "Group":       _grp_label,
                "Side":        _side_label,
                "Avg fare (£)": round(_sg["trip_price_in_pound"].mean(), 2),
                "Sub-£10 %":   round((_sg["trip_price_in_pound"] < 10).mean() * 100, 1),
            })
    _ew_df = pd.DataFrame(_ew_rows)

    if not _ew_df.empty:
        _ew_c1, _ew_c2 = st.columns(2)
        with _ew_c1:
            _fig_ew1 = px.bar(_ew_df, x="Group", y="Avg fare (£)", color="Side", barmode="group",
                              color_discrete_map={"West (<−0.12°)": "#60a5fa", "East (≥−0.12°)": "#fb923c"},
                              category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
                              text="Avg fare (£)", title="Avg fare — East vs West", height=340)
            _fig_ew1.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
            st.plotly_chart(_fig_ew1, use_container_width=True)
        with _ew_c2:
            _fig_ew2 = px.bar(_ew_df, x="Group", y="Sub-£10 %", color="Side", barmode="group",
                              color_discrete_map={"West (<−0.12°)": "#60a5fa", "East (≥−0.12°)": "#fb923c"},
                              category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
                              text="Sub-£10 %", title="Sub-£10 rate — East vs West", height=340)
            _fig_ew2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            _fig_ew2.update_layout(yaxis_ticksuffix="%")
            st.plotly_chart(_fig_ew2, use_container_width=True)

    # Ping analysis
    st.divider()
    st.markdown("#### Ping volume, location & quality")
    st.caption("Last 14 days · accepted + declined pings · normalised per driver per day")

    _ps_top  = _ping_stats(_gaps_top,  _dec_top,  len(TOP_DRIVER_IDS))
    _ps_cmp  = _ping_stats(_gaps_cmp,  _dec_cmp,  len(BAD_DRIVER_IDS))
    _ps_rest = _ping_stats(_gaps_rest, _dec_rest, len(_rest_ids))

    _pv1, _pv2, _pv3 = st.columns(3)
    for _col, _label, _color, _ps in [
        (_pv1, "TOP 10",        "#22c55e", _ps_top),
        (_pv2, "REST OF FLEET", "#94a3b8", _ps_rest),
        (_pv3, "COMPARISON",    "#ef4444", _ps_cmp),
    ]:
        _rows = [
            ("Pings/driver/day",    str(_ps["pings_per_dd"])),
            ("Accepted/driver/day", str(_ps["acc_per_dd"])),
            ("Acceptance rate",     f"{_ps['accept_rate']:.0f}%"),
            ("West ping source",    f"{_ps['west_pings_pct']:.0f}%"),
            ("Declined avg fare",   f"£{_ps['dec_avg_fare']:.2f}" if _ps["dec_avg_fare"] else "—"),
            ("Declined sub-£10",    f"{_ps['dec_sub10_pct']:.0f}%" if _ps["dec_avg_fare"] else "—"),
            ("Declined £30+",       f"{_ps['dec_30p_pct']:.0f}%" if _ps["dec_avg_fare"] else "—"),
        ]
        _body = "".join(
            f'<tr><td style="padding:4px 0;color:#94a3b8;font-size:12px;">{k}</td>'
            f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
            for k, v in _rows
        )
        _col.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {_color};border-radius:8px;padding:14px 16px;">'
            f'<div style="color:{_color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:6px;">{_label}</div>'
            f'<table style="width:100%;border-collapse:collapse;">{_body}</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _pp1, _pp2 = st.columns(2)
    with _pp1:
        _ping_src = pd.DataFrame([
            {"Group": "Top 10",        "West %": _ps_top["west_pings_pct"],  "East %": 100 - _ps_top["west_pings_pct"]},
            {"Group": "Rest of fleet", "West %": _ps_rest["west_pings_pct"], "East %": 100 - _ps_rest["west_pings_pct"]},
            {"Group": "Comparison",    "West %": _ps_cmp["west_pings_pct"],  "East %": 100 - _ps_cmp["west_pings_pct"]},
        ]).melt(id_vars="Group", var_name="Side", value_name="% of pings")
        _fig_psrc = px.bar(_ping_src, x="Group", y="% of pings", color="Side", barmode="stack",
                           color_discrete_map={"West %": "#60a5fa", "East %": "#fb923c"},
                           category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
                           text_auto=".1f", title="Ping source — East vs West", height=340)
        _fig_psrc.update_layout(yaxis_ticksuffix="%", legend_title="Ping source")
        st.plotly_chart(_fig_psrc, use_container_width=True)

    with _pp2:
        _dec_fare_rows = []
        for _glabel, _dec_df in [("Top 10", _dec_top), ("Rest of fleet", _dec_rest), ("Comparison", _dec_cmp)]:
            if _dec_df.empty:
                continue
            _fares = _dec_df["trip_price_in_pound"].dropna()
            _fares = _fares[_fares > 0]
            if len(_fares) == 0:
                continue
            for _band, _mask in [
                ("Sub-£10",  _fares < 10),
                ("£10–20",  (_fares >= 10) & (_fares < 20)),
                ("£20–30",  (_fares >= 20) & (_fares < 30)),
                ("£30+",     _fares >= 30),
            ]:
                _dec_fare_rows.append({"Group": _glabel, "Fare band": _band,
                                       "% of declined": round(_mask.mean() * 100, 1)})
        if _dec_fare_rows:
            _fig_df = px.bar(pd.DataFrame(_dec_fare_rows), x="Group", y="% of declined",
                             color="Fare band", barmode="stack",
                             color_discrete_map={"Sub-£10": "#ef4444", "£10–20": "#f59e0b",
                                                 "£20–30": "#60a5fa", "£30+": "#22c55e"},
                             category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
                             title="Quality of pings each group is declining", height=340)
            _fig_df.update_layout(yaxis_ticksuffix="%", legend_title="Fare band")
            st.plotly_chart(_fig_df, use_container_width=True)

    _ppd = pd.DataFrame([
        {"Group": "Top 10",        "Metric": "Total pings/driver/day",    "Value": _ps_top["pings_per_dd"]},
        {"Group": "Rest of fleet", "Metric": "Total pings/driver/day",    "Value": _ps_rest["pings_per_dd"]},
        {"Group": "Comparison",    "Metric": "Total pings/driver/day",    "Value": _ps_cmp["pings_per_dd"]},
        {"Group": "Top 10",        "Metric": "Accepted pings/driver/day", "Value": _ps_top["acc_per_dd"]},
        {"Group": "Rest of fleet", "Metric": "Accepted pings/driver/day", "Value": _ps_rest["acc_per_dd"]},
        {"Group": "Comparison",    "Metric": "Accepted pings/driver/day", "Value": _ps_cmp["acc_per_dd"]},
    ])
    _fig_ppd = px.bar(_ppd, x="Metric", y="Value", color="Group", barmode="group",
                      color_discrete_map={"Top 10": "#22c55e", "Rest of fleet": "#94a3b8", "Comparison": "#ef4444"},
                      category_orders={"Group": ["Top 10", "Rest of fleet", "Comparison"]},
                      text_auto=".1f", title="Ping volume — total vs accepted per driver per day", height=340)
    st.plotly_chart(_fig_ppd, use_container_width=True)

    _z3p_top_val = _z3p(_mat_top)
    _z3p_cmp_val = _z3p(_mat_cmp)
    st.markdown(
        f'<div style="background:#1e1e2e;border-left:4px solid #a78bfa;padding:14px 16px;'
        f'border-radius:6px;color:#e2e8f0;">'
        f'<strong>Why comparison drivers see fewer (and worse) pings:</strong><br><br>'
        f'<strong>1. Location.</strong> Comparison get <strong>{_ps_cmp["west_pings_pct"]:.0f}%</strong> of pings '
        f'from west of Charing Cross vs <strong>{_ps_top["west_pings_pct"]:.0f}%</strong> for the top 10.<br><br>'
        f'<strong>2. Zone 3 trap self-reinforces.</strong> Accepting a Z3 drop → stranded → low ping volume → '
        f'accept the next ping out of desperation → probably another Z3 trip.<br><br>'
        f'<strong>3. Quality of what they see.</strong> Even pings comparison drivers decline are '
        f'<strong>{_ps_cmp["dec_sub10_pct"]:.0f}% sub-£10</strong>. Top 10 decline pings worth '
        f'<strong>£{_ps_top["dec_avg_fare"]:.2f} avg</strong> — they\'re filtering a higher-quality offer pool.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 7: Decision gap ───────────────────────────────────────────────
    st.divider()
    st.subheader("7 — It's a decision gap, not a location gap")
    st.caption("Same streets. Same pings on screen. Completely different acceptance thresholds.")

    _same_df = pd.DataFrame([
        {"Category": "Cat A (sample)", "Pings": 230, "Accepted": 29,  "Accept %": 12.6, "Avg fare £": 25.27, "Sub-£10 %": 0,  "£20+ %": 79},
        {"Category": "Cat D (sample)", "Pings": 254, "Accepted": 94,  "Accept %": 37.0, "Avg fare £": 11.80, "Sub-£10 %": 51, "£20+ %": 12},
    ])

    _sl1, _sl2 = st.columns(2)
    with _sl1:
        _fig_sf = px.bar(_same_df, x="Category", y="Avg fare £", color="Category",
                         color_discrete_map={"Cat A (sample)": "#22c55e", "Cat D (sample)": "#ef4444"},
                         text="Avg fare £", title="Avg fare — same City/Inner East streets", height=300)
        _fig_sf.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
        _fig_sf.update_layout(showlegend=False, yaxis_range=[0, 32])
        st.plotly_chart(_fig_sf, use_container_width=True)
    with _sl2:
        _fig_ss = px.bar(_same_df, x="Category", y="Sub-£10 %", color="Category",
                         color_discrete_map={"Cat A (sample)": "#22c55e", "Cat D (sample)": "#ef4444"},
                         text="Sub-£10 %", title="Sub-£10 rate — same location", height=300)
        _fig_ss.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        _fig_ss.update_layout(showlegend=False, yaxis_ticksuffix="%", yaxis_range=[0, 65])
        st.plotly_chart(_fig_ss, use_container_width=True)

    st.dataframe(_same_df.set_index("Category"), use_container_width=True)
    st.caption("City of London / Clerkenwell / Inner East. Full 2026 dataset. Cat A accepted 13%, Cat D accepted 37% — from the same pool of pings.")

    st.markdown(
        '<div style="background:#1e1e2e;border-left:4px solid #6366f1;padding:14px 16px;'
        'border-radius:6px;color:#e2e8f0;margin-top:8px;">'
        'Cat A: <strong>£25.27 avg fare</strong>, 0% sub-£10. '
        'Cat D: <strong>£11.80 avg fare</strong>, 51% sub-£10. '
        'The location is identical. The pings are identical. '
        '<strong>The earnings gap is entirely a decision gap — it happens at the moment of acceptance.</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 8: Fleet map ──────────────────────────────────────────────────
    st.divider()
    st.subheader("8 — Where the fleet actually operates")
    st.caption("Pickup density — every active driver, last 30 days, coloured by performance category.")

    with st.spinner("Loading fleet positioning data..."):
        _ff_cat_df  = pd.read_csv(_CAT_PATH)
        _ff_cat_map = dict(zip(_ff_cat_df["dim_driver_id"], _ff_cat_df["category"]))
        _ff_raw     = db.load_all_driver_coords(sample_per_driver=60, days_back=30)

    if not _ff_raw.empty:
        _ff_raw["category"] = _ff_raw["dim_driver_id"].map(_ff_cat_map)
        _ff_raw["category"] = _ff_raw["category"].where(
            _ff_raw["category"].isin(["A", "B1", "B2", "C1", "C2", "D"]), other=None
        )
        _ff_coords = _ff_raw["pickup_lat_long"].apply(parse_dms)
        _ff_raw["plat"] = [c[0] for c in _ff_coords]
        _ff_raw["plon"] = [c[1] for c in _ff_coords]
        _ff_fleet = _ff_raw.dropna(subset=["plat", "plon"])
        _ff_fleet = _ff_fleet[_ff_fleet["plat"].between(51.3, 51.7) & _ff_fleet["plon"].between(-0.55, 0.3)].copy()
        _ff_fleet["cat_label"] = _ff_fleet["category"].map(CAT_LABELS).fillna("Unclassified")

        _cat_order_labels = ["Unclassified", "D — Low performer", "C2 — Below avg", "C1 — Developing",
                             "B2 — Solid", "B1 — Strong", "A — Elite"]
        _color_map_labels = {CAT_LABELS.get(k, "Unclassified"): v for k, v in CAT_COLORS.items() if k is not None}
        _color_map_labels["Unclassified"] = "#94a3b8"

        _ff_fig_map = px.scatter_mapbox(
            _ff_fleet.sort_values("cat_label", key=lambda s: s.map({v: i for i, v in enumerate(_cat_order_labels)})),
            lat="plat", lon="plon",
            color="cat_label",
            color_discrete_map=_color_map_labels,
            category_orders={"cat_label": _cat_order_labels},
            hover_name="driver_full_name",
            hover_data={"plat": False, "plon": False, "cat_label": True},
            zoom=10, height=520,
            mapbox_style="carto-positron",
            opacity=0.6,
            title="Fleet pickup density — last 30 days",
        )
        _ff_fig_map.update_traces(marker=dict(size=5))
        _ff_fig_map.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(title="Category", x=0.01, y=0.99, bgcolor="rgba(30,30,46,0.85)", bordercolor="#555", borderwidth=1),
        )
        st.plotly_chart(_ff_fig_map, use_container_width=True)

        # West % bar
        _west_rows = []
        for _, _grp in _ff_fleet.groupby("dim_driver_id"):
            _did = _grp["dim_driver_id"].iloc[0]
            _cat = _grp["category"].iloc[0] if pd.notna(_grp["category"].iloc[0]) else None
            _lons = _grp["plon"].dropna()
            _west_rows.append({
                "Driver":   DRIVER_NAMES.get(_did, _grp["driver_full_name"].iloc[0]),
                "Category": _cat if _cat else "—",
                "West %":   round((_lons < _WEST_LON).mean() * 100, 1),
            })
        _west_df = pd.DataFrame(_west_rows)
        _cat_sort = ["A", "B1", "B2", "C1", "C2", "D", "—"]
        _west_df["_s"] = _west_df["Category"].map({c: i for i, c in enumerate(_cat_sort)})
        _west_df = _west_df.sort_values(["_s", "West %"], ascending=[True, False]).drop(columns="_s")

        _cmap2 = {"A": "#22c55e", "B1": "#4ade80", "B2": "#60a5fa",
                  "C1": "#f59e0b", "C2": "#fb923c", "D": "#ef4444", "—": "#94a3b8"}
        _fig_west = px.bar(_west_df, x="West %", y="Driver", orientation="h",
                           color="Category", color_discrete_map=_cmap2, text="West %",
                           title="West London pickup % — all drivers",
                           height=max(480, len(_west_df) * 20 + 80))
        _fig_west.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                            annotation_text="50%", annotation_position="top")
        _fig_west.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        _fig_west.update_layout(xaxis_title="% pickups west of −0.12°", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(_fig_west, use_container_width=True)

        # Summary stats table
        _cat_summ = (
            _west_df.groupby("Category")["West %"]
            .agg(Drivers="count", Mean="mean", Median="median", Min="min", Max="max")
            .round(1).reset_index()
        )
        _cat_summ["_s"] = _cat_summ["Category"].map({c: i for i, c in enumerate(_cat_sort)})
        _cat_summ = _cat_summ.sort_values("_s").drop(columns="_s")
        st.dataframe(_cat_summ, use_container_width=True, hide_index=True)
    else:
        st.warning("Fleet positioning data not available.")

    # ── SECTION 9: Key takeaways ──────────────────────────────────────────────
    st.divider()
    st.subheader("9 — Key takeaways")
    st.markdown(f"""
| # | Finding | Key stat | Implication |
|---|---------|----------|-------------|
| 1 | **Lower acceptance = higher RPH** | Top 10: {_ff_top10_accept:.0f}% accept vs Fleet: {_FLEET_ACCEPT}% | Selectivity is the strategy, not volume |
| 2 | **West positioning generates better pings** | Cat A outliers still beat Cat D despite lower west % | Positioning helps — selectivity converts it |
| 3 | **Zone 3 daytime is a trap** | True RPH ~£17/hr + 32 min avg wait | Leave or decline until clear before 09:00 |
| 4 | **Zone 3 at night is valuable** | £32–43/hr, peaks £43/hr at 03:00 | Night shift: Zone 3 00:00–06:00 is valid |
| 5 | **Cat A/B/C/D reflect distinct behaviours** | Not a ranking — a pattern of decisions | Each category has a clear behavioural signature |
| 6 | **Position helps but selectivity converts** | Cat D driver at 93% west still underperforms | Moving west without filtering just earns cheap trips in a premium area |
| 7 | **It's a decision gap, not a location gap** | Cat A: £25.27 avg · Cat D: £11.80 avg, same streets | The earnings difference happens at the moment of acceptance |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# FLEET MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Fleet Map":
    st.title("Fleet Positioning Map")
    st.caption("Every active driver's pickup locations over the last 30 days, coloured by performance category.")

    _CAT_COLOR = {
        "A": "#22c55e", "B1": "#4ade80", "B2": "#60a5fa",
        "C1": "#f59e0b", "C2": "#fb923c", "D": "#ef4444", None: "#94a3b8",
    }
    _CAT_LABEL = {
        "A": "A — Elite", "B1": "B1 — Strong", "B2": "B2 — Solid",
        "C1": "C1 — Developing", "C2": "C2 — Below avg", "D": "D — Low performer",
        None: "Unclassified",
    }
    _CAT_RENDER_ORDER = [None, "D", "C2", "C1", "B2", "B1", "A"]
    _CAT_OPACITY = {None: 0.25, "D": 0.50, "C2": 0.50, "C1": 0.50, "B2": 0.55, "B1": 0.60, "A": 0.70}
    _CAT_RADIUS  = {None: 2, "D": 3, "C2": 3, "C1": 3, "B2": 3, "B1": 3, "A": 4}

    _cat_df  = pd.read_csv(_CAT_PATH)
    _cat_map = dict(zip(_cat_df["dim_driver_id"], _cat_df["category"]))

    with st.spinner("Loading fleet pickup data..."):
        _raw = db.load_all_driver_coords(sample_per_driver=60, days_back=30)

    if _raw.empty:
        st.warning("No data returned.")
        st.stop()

    _raw["category"] = _raw["dim_driver_id"].map(_cat_map)
    _raw["category"] = _raw["category"].where(_raw["category"].isin(["A","B1","B2","C1","C2","D"]), other=None)

    with st.spinner("Parsing coordinates..."):
        _coords = _raw["pickup_lat_long"].apply(parse_dms)
        _raw["plat"] = [c[0] for c in _coords]
        _raw["plon"] = [c[1] for c in _coords]

    _fleet = _raw.dropna(subset=["plat","plon"])
    _fleet = _fleet[_fleet["plat"].between(51.3,51.7) & _fleet["plon"].between(-0.55,0.3)]

    _n_drivers = _fleet["dim_driver_id"].nunique()
    _n_points  = len(_fleet)

    # Category counts
    _cat_counts = _fleet.groupby("category", dropna=False)["dim_driver_id"].nunique()
    _mc = st.columns(7)
    for i, cat in enumerate(["A","B1","B2","C1","C2","D",None]):
        _mc[i].metric(_CAT_LABEL[cat].split(" — ")[0], _cat_counts.get(cat, 0), help=_CAT_LABEL[cat])

    # Driver filter
    st.divider()
    _all_driver_names = sorted(
        _fleet.drop_duplicates("dim_driver_id")
        .apply(lambda r: DRIVER_NAMES.get(r["dim_driver_id"], r["driver_full_name"]), axis=1)
    )
    _sel_drivers = st.multiselect("Highlight specific drivers (empty = all)", options=_all_driver_names)

    if _sel_drivers:
        _did_set = {
            r["dim_driver_id"]
            for _, r in _fleet.drop_duplicates("dim_driver_id").iterrows()
            if DRIVER_NAMES.get(r["dim_driver_id"], r["driver_full_name"]) in _sel_drivers
        }
        _fleet_map = _fleet[_fleet["dim_driver_id"].isin(_did_set)]
        st.caption(f"Showing {len(_sel_drivers)} selected driver(s) · {len(_fleet_map):,} points")
    else:
        _fleet_map = _fleet
        st.caption(f"Showing all {_n_drivers} drivers · {_n_points:,} pickup points")

    # Folium map
    _m = folium.Map(location=[51.505, -0.13], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(GEOJSON_DATA, style_function=lambda f: {
        "fillColor": "#e2e8f0", "fillOpacity": 0.08, "color": "#94a3b8", "weight": 1
    }).add_to(_m)
    folium.Marker([51.51, _WEST_LON], icon=folium.DivIcon(
        html='<div style="color:#1e40af;font-size:11px;white-space:nowrap;font-weight:bold;">← West | East →</div>',
        icon_size=(110, 20), icon_anchor=(55, 10),
    ), tooltip="High-demand corridor boundary").add_to(_m)

    for cat in _CAT_RENDER_ORDER:
        _sub = _fleet_map[_fleet_map["category"] == cat] if cat is not None else _fleet_map[_fleet_map["category"].isna()]
        for _, row in _sub.iterrows():
            folium.CircleMarker(
                [row["plat"], row["plon"]], radius=_CAT_RADIUS[cat],
                color=_CAT_COLOR[cat], fill=True, fill_color=_CAT_COLOR[cat],
                fill_opacity=_CAT_OPACITY[cat], weight=0,
                tooltip=f"{row['driver_full_name']} · {_CAT_LABEL[cat]}",
            ).add_to(_m)

    st_folium(_m, width="100%", height=580, returned_objects=[])

    # Legend
    _legend = "".join(
        f'<span style="margin-right:20px;"><span style="color:{_CAT_COLOR[c]};font-size:16px;">●</span> {_CAT_LABEL[c]}</span>'
        for c in ["A","B1","B2","C1","C2","D",None]
    )
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;">{_legend}</div>', unsafe_allow_html=True)

    # West % analysis
    st.divider()
    st.subheader("West London pickup % by category")
    st.caption("% of each driver's pickups west of −0.12° (Charing Cross line)")

    _west_rows = []
    for _, _grp in _fleet_map.groupby("dim_driver_id"):
        _did = _grp["dim_driver_id"].iloc[0]
        _cat = _grp["category"].iloc[0] if pd.notna(_grp["category"].iloc[0]) else None
        _lons = _grp["plon"].dropna()
        _west_rows.append({
            "Driver":   DRIVER_NAMES.get(_did, _grp["driver_full_name"].iloc[0]),
            "Category": _cat if _cat else "—",
            "West %":   round((_lons < _WEST_LON).mean() * 100, 1),
            "Pickups":  len(_lons),
        })
    _west_df = pd.DataFrame(_west_rows)
    _csort = ["A","B1","B2","C1","C2","D","—"]
    _west_df["_s"] = _west_df["Category"].map({c:i for i,c in enumerate(_csort)})
    _west_df = _west_df.sort_values(["_s","West %"], ascending=[True,False]).drop(columns="_s")

    # Summary table
    _summ = (_west_df.groupby("Category")["West %"]
             .agg(Drivers="count", Mean="mean", Median="median", Min="min", Max="max")
             .round(1).reset_index())
    _summ["_s"] = _summ["Category"].map({c:i for i,c in enumerate(_csort)})
    _summ = _summ.sort_values("_s").drop(columns="_s")
    st.dataframe(_summ, use_container_width=True, hide_index=True)

    # Bar chart
    _disc = {c: _CAT_COLOR.get(c if c != "—" else None, "#94a3b8") for c in _csort}
    _fig_all = px.bar(_west_df, x="West %", y="Driver", orientation="h",
                      color="Category", color_discrete_map=_disc, text="West %",
                      title="West London pickup % — all drivers",
                      height=max(500, len(_west_df) * 22 + 80))
    _fig_all.add_vline(x=50, line_dash="dash", line_color="#94a3b8", annotation_text="50%", annotation_position="top")
    _fig_all.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _fig_all.update_layout(xaxis_title="% pickups west of −0.12°", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(_fig_all, use_container_width=True)

    # Narrative
    _a_avg = _west_df[_west_df["Category"] == "A"]["West %"].mean() if "A" in _west_df["Category"].values else 0
    _d_avg = _west_df[_west_df["Category"] == "D"]["West %"].mean() if "D" in _west_df["Category"].values else 0
    _all_avg = _west_df["West %"].mean()
    st.markdown(
        f'<div style="background:#1e1e2e;border-left:4px solid #22c55e;padding:14px 16px;'
        f'border-radius:6px;color:#e2e8f0;margin-top:8px;">'
        f'<strong>Fleet positioning by category:</strong><br><br>'
        f'<span style="color:#22c55e;">●</span> <strong>Cat A avg: {_a_avg:.1f}%</strong> west<br>'
        f'<span style="color:#ef4444;">●</span> <strong>Cat D avg: {_d_avg:.1f}%</strong> west '
        f'(<strong>{_a_avg - _d_avg:.1f}pp below Cat A</strong>)<br>'
        f'<span style="color:#94a3b8;">●</span> Fleet overall: <strong>{_all_avg:.1f}%</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER DAY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Driver Day":
    st.title("Driver Day")
    _dd_mode = st.radio("", ["Single driver", "Fleet day view"], horizontal=True, key="dd_mode")
    st.divider()

    # ── FLEET DAY VIEW ────────────────────────────────────────────────────────
    if _dd_mode == "Fleet day view":
        st.caption("All drivers' pickup locations for a single day, coloured by performance category.")

        _yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).date()
        _fday_date = st.date_input("Date", value=_yesterday, max_value=_yesterday, key="fday_date")
        _fday_str  = str(_fday_date)

        with st.spinner(f"Loading all trips for {_fday_str}..."):
            _fday_raw = _load_fleet_day(_fday_str)

        if _fday_raw.empty:
            st.warning(f"No completed trips found for {_fday_str}.")
            st.stop()

        # Parse coords and assign categories
        _fday = _fday_raw.copy()
        _fday["fare"] = pd.to_numeric(_fday["trip_price_in_pound"], errors="coerce").fillna(0)
        _fday["pickedup_trip_datetime"] = pd.to_datetime(_fday["pickedup_trip_datetime"])
        _fday["time_str"] = _fday["pickedup_trip_datetime"].dt.strftime("%H:%M")
        _fpu = _fday["pickup_lat_long"].apply(lambda x: parse_dms(str(x or "")))
        _fday["plat"] = [c[0] for c in _fpu]
        _fday["plon"] = [c[1] for c in _fpu]
        _fday = _fday.dropna(subset=["plat","plon"])
        _fday = _fday[
            (_fday["plat"].between(51.2, 51.8)) &
            (_fday["plon"].between(-0.6, 0.3))
        ].copy()

        # Load categories
        try:
            _fday_cat_df  = pd.read_csv(_CAT_PATH)
            _fday_cat_map = dict(zip(_fday_cat_df["dim_driver_id"], _fday_cat_df["category"]))
        except Exception:
            _fday_cat_map = {}
        _fday["category"]  = _fday["dim_driver_id"].map(_fday_cat_map).fillna("—")
        _fday["cat_label"] = _fday["category"].map(CAT_LABELS).fillna("Unclassified")
        _fday["color"]     = _fday["category"].map(CAT_COLORS).fillna("#94a3b8")

        # Summary row
        _fd_c1, _fd_c2, _fd_c3, _fd_c4 = st.columns(4)
        _fd_c1.metric("Total trips",    f"{len(_fday):,}")
        _fd_c2.metric("Active drivers", f"{_fday['dim_driver_id'].nunique():,}")
        _fd_c3.metric("Avg fare",       f"£{_fday[_fday['fare']>0]['fare'].mean():.2f}" if (_fday['fare']>0).any() else "—")
        _fd_c4.metric("Total revenue",  f"£{_fday['fare'].sum():,.0f}")

        # Category filter
        _cat_options = ["All"] + [c for c in ["A","B1","B2","C1","C2","D"] if c in _fday["category"].values]
        _cat_filter  = st.multiselect("Filter by category", options=_cat_options[1:], default=[], key="fday_cat_filter")
        _fday_map = _fday[_fday["category"].isin(_cat_filter)].copy() if _cat_filter else _fday.copy()

        # Map
        _cat_order_labels = [CAT_LABELS.get(c, c) for c in ["A","B1","B2","C1","C2","D","—"]]
        _color_map_labels = {CAT_LABELS.get(k, k): v for k, v in CAT_COLORS.items()}

        _fig_fday = px.scatter_mapbox(
            _fday_map.sort_values("category"),
            lat="plat", lon="plon",
            color="cat_label",
            color_discrete_map=_color_map_labels,
            category_orders={"cat_label": _cat_order_labels},
            size_max=9,
            opacity=0.75,
            hover_name="driver_full_name",
            hover_data={"fare": ":.2f", "time_str": True, "cat_label": True, "plat": False, "plon": False},
            mapbox_style="carto-positron",
            zoom=10,
            center={"lat": 51.505, "lon": -0.09},
            title=f"Fleet pickup map — {_fday_str}",
            height=620,
            labels={"fare": "Fare £", "time_str": "Time", "cat_label": "Category", "driver_full_name": "Driver"},
        )
        _fig_fday.update_layout(
            legend_title_text="Category",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig_fday, use_container_width=True)

        # Trips per driver for the day
        st.subheader("Trips per driver")
        _fday_drv = (
            _fday_map.groupby(["dim_driver_id","driver_full_name","cat_label"])
            .agg(trips=("fare","count"), total_fare=("fare","sum"), avg_fare=("fare","mean"))
            .reset_index()
            .sort_values("trips", ascending=False)
        )
        _fday_drv["total_fare"] = _fday_drv["total_fare"].round(2)
        _fday_drv["avg_fare"]   = _fday_drv["avg_fare"].round(2)
        st.dataframe(
            _fday_drv[["driver_full_name","cat_label","trips","total_fare","avg_fare"]]
            .rename(columns={"driver_full_name":"Driver","cat_label":"Category","total_fare":"Total £","avg_fare":"Avg fare £"}),
            use_container_width=True, hide_index=True,
        )
        st.stop()

    # ── SINGLE DRIVER VIEW ────────────────────────────────────────────────────
    st.caption("Pick any driver and a date — every trip, every declined ping, earnings through the shift.")

    _col_srch, _col_date = st.columns([2, 1])
    with _col_srch:
        _dd_search = st.text_input("Search driver by name",
                                   placeholder="e.g. Yousuf, Mukhtar, Monier...",
                                   key="dd_search")

    if not _dd_search or len(_dd_search) < 2:
        st.info("Type at least 2 characters to search for a driver.")
        st.stop()

    with st.spinner("Searching..."):
        _dd_results = _search_drivers(_dd_search)

    if _dd_results.empty:
        st.warning(f"No drivers found matching '{_dd_search}'.")
        st.stop()

    _dd_opts = {f"{row.driver_full_name} (ID {row.dim_driver_id})": int(row.dim_driver_id)
                for _, row in _dd_results.iterrows()}

    with _col_srch:
        _dd_label = st.selectbox("Select driver", options=list(_dd_opts.keys()), key="dd_select")
        _driver_id = _dd_opts[_dd_label]

    with st.spinner(f"Loading trips..."):
        _dd_raw = db.load_any_driver_trips(_driver_id)

    _today = pd.Timestamp.now().date()

    if _dd_raw.empty:
        _driver_trips = pd.DataFrame()
        _avail_dates  = []
    else:
        _enriched = enrich_zones(_dd_raw)
        _enriched = calc_true_rph(_enriched)
        _enriched["pickedup_trip_datetime"] = pd.to_datetime(_enriched["pickedup_trip_datetime"])
        _enriched["dropoff_trip_datetime"]  = pd.to_datetime(_enriched["dropoff_trip_datetime"])
        _enriched["trip_date"] = _enriched["pickedup_trip_datetime"].dt.date
        _driver_trips = _enriched.copy()
        _avail_dates  = sorted(_driver_trips["trip_date"].dropna().unique(), reverse=True)

    if _today not in _avail_dates:
        _avail_dates = [_today] + list(_avail_dates)

    _date_strs   = [str(d) for d in _avail_dates]
    _default_idx = 1 if len(_date_strs) > 1 and str(_today) == _date_strs[0] else 0

    with _col_date:
        _sel_date = st.selectbox("Date", options=_date_strs, index=_default_idx)

    if not _driver_trips.empty:
        _day_trips = (
            _driver_trips[_driver_trips["trip_date"] == pd.Timestamp(_sel_date).date()]
            .sort_values("pickedup_trip_datetime")
            .reset_index(drop=True)
        )
        _day_trips["trip_num"] = range(1, len(_day_trips) + 1)
    else:
        _day_trips = pd.DataFrame()

    _has_trips = not _day_trips.empty

    # Declined pings
    with st.spinner("Loading declined pings..."):
        _dec_raw = db.load_driver_declined_day(_driver_id, _sel_date)

    if not _dec_raw.empty:
        _dec_coords = _dec_raw["pickup_lat_long"].apply(parse_dms)
        _dec_raw["plat"] = [c[0] for c in _dec_coords]
        _dec_raw["plon"] = [c[1] for c in _dec_coords]
        _dec_day = _dec_raw.dropna(subset=["plat","plon"])
    else:
        _dec_day = pd.DataFrame()

    _n_dec = len(_dec_day)

    # Stats strip
    _total_fare = _day_trips["trip_price_in_pound"].sum() if _has_trips else 0
    _n_trips    = len(_day_trips)
    _day_rph    = _day_trips["true_rph"].replace([np.inf,-np.inf], np.nan).median() if _has_trips else None
    _accept_pct = round(_n_trips / max(_n_trips + _n_dec, 1) * 100)

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _sc1.metric("Trips completed", _n_trips)
    _sc2.metric("Total earnings",  f"£{_total_fare:.2f}")
    _sc3.metric("Median RPH",      f"£{_day_rph:.2f}" if _day_rph else "—")
    _sc4.metric("Accept rate",     f"{_accept_pct}%", f"{_n_dec} declined")

    st.divider()

    if not _has_trips:
        st.info(f"No completed trips found for {_sel_date}.")
        if not _dec_day.empty:
            st.markdown(f"Found **{_n_dec} declined pings** on this date.")
        st.stop()

    # ── Map ───────────────────────────────────────────────────────────────────
    st.subheader("Trip map")

    _m_day = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(GEOJSON_DATA, style_function=lambda f: {
        "fillColor": "#e2e8f0", "fillOpacity": 0.08, "color": "#94a3b8", "weight": 1,
    }).add_to(_m_day)

    # Plot trips
    for _, row in _day_trips.iterrows():
        _pc = parse_dms(str(row.get("pickup_lat_long", "") or ""))
        _dc = parse_dms(str(row.get("dropoff_latlong", "") or ""))
        if _pc[0] and _dc[0]:
            folium.CircleMarker(
                [_pc[0], _pc[1]], radius=6, color="#22c55e", fill=True, fill_opacity=0.8, weight=0,
                tooltip=f"Trip {row['trip_num']}: pickup · £{row['trip_price_in_pound']:.2f}",
            ).add_to(_m_day)
            folium.CircleMarker(
                [_dc[0], _dc[1]], radius=6, color="#ef4444", fill=True, fill_opacity=0.8, weight=0,
                tooltip=f"Trip {row['trip_num']}: dropoff · Zone {row.get('dropoff_zone','')}",
            ).add_to(_m_day)
            folium.PolyLine([[_pc[0],_pc[1]], [_dc[0],_dc[1]]],
                            color="#6366f1", weight=1.5, opacity=0.5).add_to(_m_day)

    # Declined pings
    for _, row in _dec_day.iterrows():
        folium.CircleMarker(
            [row["plat"], row["plon"]], radius=4, color="#f59e0b", fill=True, fill_opacity=0.5, weight=0,
            tooltip=f"Declined · £{row.get('trip_price_in_pound', 0):.2f}",
        ).add_to(_m_day)

    st_folium(_m_day, width="100%", height=500, returned_objects=[])
    st.caption("🟢 Pickup  🔴 Dropoff  🟡 Declined ping  🟣 Route")

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Trip timeline")

    _tl_df = _day_trips.copy()
    _tl_df["start"] = _tl_df["pickedup_trip_datetime"]
    _tl_df["end"]   = _tl_df["dropoff_trip_datetime"]
    _tl_df["label"] = _tl_df.apply(
        lambda r: f"Trip {r['trip_num']} · £{r['trip_price_in_pound']:.2f} · Z{r.get('pickup_zone','')}→Z{r.get('dropoff_zone','')}",
        axis=1,
    )

    _fig_tl = px.timeline(
        _tl_df, x_start="start", x_end="end", y="label",
        color="trip_price_in_pound",
        color_continuous_scale="RdYlGn",
        title=f"Trip timeline — {_sel_date}",
        labels={"trip_price_in_pound": "Fare £"},
        height=max(300, _n_trips * 40 + 80),
    )
    _fig_tl.update_xaxes(tickformat="%H:%M")
    _fig_tl.update_layout(yaxis_title="", coloraxis_colorbar_title="Fare £", showlegend=False)
    st.plotly_chart(_fig_tl, use_container_width=True)

    # ── Earnings curve ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Earnings through the shift")

    _earn_df = _day_trips.copy()
    _earn_df["cumulative"] = _earn_df["trip_price_in_pound"].cumsum()
    _earn_df["time"]       = _earn_df["dropoff_trip_datetime"]

    _fig_earn = go.Figure()
    _fig_earn.add_trace(go.Scatter(
        x=_earn_df["time"], y=_earn_df["cumulative"],
        mode="lines+markers+text",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=8, color="#22c55e"),
        text=[f"£{v:.0f}" for v in _earn_df["cumulative"]],
        textposition="top center",
        name="Cumulative earnings",
    ))
    _fig_earn.update_layout(
        title=f"Cumulative earnings — {_sel_date}",
        xaxis_title="Time", yaxis_title="£ earned",
        xaxis=dict(tickformat="%H:%M"),
        height=360,
    )
    st.plotly_chart(_fig_earn, use_container_width=True)

    # ── Trip table ────────────────────────────────────────────────────────────
    with st.expander("Full trip table"):
        _tbl = _day_trips[[
            "trip_num", "pickedup_trip_datetime", "dropoff_trip_datetime",
            "trip_price_in_pound", "pickup_zone", "dropoff_zone", "true_rph",
        ]].copy()
        _tbl.columns = ["#", "Pickup time", "Dropoff time", "Fare £", "Pickup zone", "Dropoff zone", "True RPH"]
        _tbl["Pickup time"]  = _tbl["Pickup time"].dt.strftime("%H:%M")
        _tbl["Dropoff time"] = _tbl["Dropoff time"].dt.strftime("%H:%M")
        _tbl["Fare £"]    = _tbl["Fare £"].round(2)
        _tbl["True RPH"]  = _tbl["True RPH"].replace([np.inf,-np.inf], np.nan).round(2)
        st.dataframe(_tbl, use_container_width=True, hide_index=True)



# ═══════════════════════════════════════════════════════════════════════════════
# GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Gap Analysis":
    st.title("Gap Analysis")
    st.caption("Find when drivers get stuck, where they were stranded, and what high-value pings they passed on.")

    _LHR = {"lat": (51.45, 51.49), "lon": (-0.50, -0.42)}
    _LGW = {"lat": (51.13, 51.18), "lon": (-0.22, -0.14)}

    def _is_airport(lat, lon):
        if lat is None or lon is None:
            return None
        if _LHR["lat"][0] <= lat <= _LHR["lat"][1] and _LHR["lon"][0] <= lon <= _LHR["lon"][1]:
            return "Heathrow"
        if _LGW["lat"][0] <= lat <= _LGW["lat"][1] and _LGW["lon"][0] <= lon <= _LGW["lon"][1]:
            return "Gatwick"
        return None

    def _stuck_events(trips_df):
        rows = []
        if trips_df.empty or len(trips_df) < 2:
            return pd.DataFrame(rows)
        df = trips_df.copy()
        df["pickedup_trip_datetime"] = pd.to_datetime(df["pickedup_trip_datetime"])
        df["dropoff_trip_datetime"]  = pd.to_datetime(df["dropoff_trip_datetime"])
        df = df.sort_values("pickedup_trip_datetime").reset_index(drop=True)
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            gap  = (curr["pickedup_trip_datetime"] - prev["dropoff_trip_datetime"]).total_seconds() / 60
            if gap < 0:
                continue
            dc      = parse_dms(str(prev.get("dropoff_latlong", "") or ""))
            airport = _is_airport(dc[0], dc[1])
            west    = (dc[1] is not None and dc[1] < _WEST_LON)
            _npu    = parse_dms(str(curr.get("pickup_lat_long", "") or ""))
            _ndo    = parse_dms(str(curr.get("dropoff_latlong", "") or ""))
            rows.append({
                "gap_mins":    round(gap, 1),
                "stuck_from":  prev["dropoff_trip_datetime"],
                "stuck_until": curr["pickedup_trip_datetime"],
                "stuck_lat":   dc[0],
                "stuck_lon":   dc[1],
                "west":        west,
                "airport":     airport,
                "prev_fare":   float(prev.get("trip_price_in_pound", 0) or 0),
                "next_fare":   float(curr.get("trip_price_in_pound", 0) or 0),
                "next_pz":     assign_zone(_npu[0], _npu[1]),
                "next_dz":     assign_zone(_ndo[0], _ndo[1]),
                "bucket":      ("<25m" if gap < 25 else ("25-75m" if gap <= 75 else ">75m")),
            })
        return pd.DataFrame(rows)

    def _classify_ping(fare, pickup_zone, dropoff_zone, pickup_west, dropoff_west):
        # Primary signal: direction relative to city centre
        # Inbound (lower zone at dropoff) = almost always worth taking
        # Outbound (higher zone at dropoff) = needs progressively higher fare to justify
        if pickup_zone and dropoff_zone:
            pz, dz    = int(pickup_zone), int(dropoff_zone)
            good_dest = dz <= 2 or dropoff_west      # drops in premium area
            bad_dest  = dz >= 5 and not dropoff_west  # strands driver far out
            zones_out = max(0, dz - pz)
            if dz < pz or good_dest:
                min_fare = 7          # inbound or great destination — take it
            elif zones_out == 0:
                min_fare = 11         # local — modest threshold
            else:
                min_fare = 12 + zones_out * 7  # outbound: +£7 each zone further out
            if bad_dest:
                min_fare += 7         # extra penalty for ending up stranded far out
        else:
            # No zone data: fall back to west/east heuristic
            min_fare = 8 if (pickup_west or dropoff_west) else 14
        if fare >= min_fare:
            return "Should accept"
        if fare >= min_fare * 0.65:
            return "Borderline"
        return "Fine to decline"

    # ── PART 1: INDIVIDUAL DRIVER ─────────────────────────────────────────────
    st.subheader("Driver stuck analysis")
    st.caption("Select any driver to see their stuck events for a given week — where stranded, how long, what pings they passed on.")

    _ga_c1, _ga_c2 = st.columns([2, 1])
    with _ga_c1:
        _ga_q = st.text_input("Search driver by name", placeholder="e.g. Yousuf, Marius, Emran...", key="ga_q")

    if not _ga_q or len(_ga_q) < 2:
        st.info("Type at least 2 characters to search for a driver.")
    else:
        with st.spinner("Searching..."):
            _ga_res = _search_drivers(_ga_q)

        if _ga_res.empty:
            st.warning(f"No drivers found matching '{_ga_q}'.")
        else:
            _ga_opts = {f"{r.driver_full_name} (ID {r.dim_driver_id})": int(r.dim_driver_id)
                        for _, r in _ga_res.iterrows()}
            with _ga_c1:
                _ga_lbl = st.selectbox("Select driver", list(_ga_opts.keys()), key="ga_sel")
                _ga_did = _ga_opts[_ga_lbl]

            _today_d = pd.Timestamp.now().date()
            _week_opts = {
                "Last 7 days":    (_today_d - pd.Timedelta(days=7),  _today_d),
                "8-14 days ago":  (_today_d - pd.Timedelta(days=14), _today_d - pd.Timedelta(days=8)),
                "15-21 days ago": (_today_d - pd.Timedelta(days=21), _today_d - pd.Timedelta(days=15)),
                "22-28 days ago": (_today_d - pd.Timedelta(days=28), _today_d - pd.Timedelta(days=22)),
            }
            with _ga_c2:
                _ga_wlbl = st.selectbox("Week", list(_week_opts.keys()), key="ga_week")
            _wstart, _wend = _week_opts[_ga_wlbl]

            with st.spinner("Loading trip data..."):
                _ga_acc = db.load_gap_accepted([_ga_did], days_back=30)
                _ga_dec = db.load_gap_declined([_ga_did], days_back=30)

            if _ga_acc.empty:
                st.warning("No trip data found for this driver in the last 30 days.")
            else:
                _ga_acc["pickedup_trip_datetime"] = pd.to_datetime(_ga_acc["pickedup_trip_datetime"])
                _ga_acc["dropoff_trip_datetime"]  = pd.to_datetime(_ga_acc["dropoff_trip_datetime"])
                _ga_acc = _ga_acc.sort_values("pickedup_trip_datetime").reset_index(drop=True)

                _ga_wk = _ga_acc[
                    (_ga_acc["pickedup_trip_datetime"].dt.date >= _wstart) &
                    (_ga_acc["pickedup_trip_datetime"].dt.date <= _wend)
                ].reset_index(drop=True)

                if _ga_wk.empty:
                    st.info("No trips found in this window.")
                else:
                    _se_df    = _stuck_events(_ga_wk)
                    _se_stuck = _se_df[_se_df["bucket"] == "25-75m"].reset_index(drop=True) if not _se_df.empty else pd.DataFrame()

                    _n_trips   = len(_ga_wk)
                    _n_stuck   = len(_se_stuck)
                    _n_airport = int(_se_stuck["airport"].notna().sum()) if not _se_stuck.empty else 0
                    _avg_stuck = _se_stuck["gap_mins"].mean() if not _se_stuck.empty else 0
                    _total_lost = (_se_stuck["gap_mins"].sum() / 60 * _FLEET_RPH_DEFAULT) if not _se_stuck.empty else 0

                    _sm1, _sm2, _sm3, _sm4, _sm5 = st.columns(5)
                    _sm1.metric("Trips completed",        _n_trips)
                    _sm2.metric("Stuck events (25-75 min)", _n_stuck)
                    _sm3.metric("Avg stuck duration",     f"{_avg_stuck:.0f} min" if _n_stuck else "---")
                    _sm4.metric("Stuck at airport",       _n_airport, help="Heathrow or Gatwick bounding box")
                    _sm5.metric("Est. revenue lost",      f"GBP{_total_lost:.0f}", help=f"Stuck minutes x fleet RPH GBP{_FLEET_RPH_DEFAULT}/hr")

                    # Missed pings during stuck windows — capture full start + end of each ping
                    _missed_rows = []
                    if not _ga_dec.empty and not _se_stuck.empty:
                        _ga_dec2 = _ga_dec.copy()
                        _time_col = "trip_booking_datetime" if "trip_booking_datetime" in _ga_dec2.columns else "pickedup_trip_datetime"
                        _ga_dec2["_pt"] = pd.to_datetime(_ga_dec2[_time_col])
                        for _se_idx, _se in _se_stuck.iterrows():
                            _win = _ga_dec2[
                                (_ga_dec2["_pt"] >= _se["stuck_from"]) &
                                (_ga_dec2["_pt"] <= _se["stuck_until"])
                            ]
                            for _, _p in _win.iterrows():
                                _pf_raw = float(_p.get("trip_price_in_pound", 0) or 0)
                                _pc     = parse_dms(str(_p.get("pickup_lat_long", "") or ""))
                                _dc     = parse_dms(str(_p.get("dropoff_latlong", "") or ""))
                                if _pf_raw <= 0 and _pc[0] is not None and _dc[0] is not None:
                                    _, _pf  = estimate_ping(_pc[0], _pc[1], _dc[0], _dc[1])
                                    _pf     = float(_pf) if _pf else 0.0
                                    _fe     = True
                                else:
                                    _pf, _fe = _pf_raw, False
                                _pw  = _pc[1] is not None and _pc[1] < _WEST_LON
                                _dw  = _dc[1] is not None and _dc[1] < _WEST_LON
                                _pz  = assign_zone(_pc[0], _pc[1])
                                _dz  = assign_zone(_dc[0], _dc[1])
                                _cls = _classify_ping(_pf, _pz, _dz, _pw, _dw)
                                _missed_rows.append({
                                    "ping_time":    _p["_pt"],
                                    "fare":         round(_pf, 2),
                                    "fare_est":     _fe,
                                    "plat":         _pc[0],  "plon": _pc[1],
                                    "dlat":         _dc[0],  "dlon": _dc[1],
                                    "pickup_west":  _pw,
                                    "dropoff_west": _dw,
                                    "pickup_zone":  _pz,
                                    "dropoff_zone": _dz,
                                    "decision":     _cls,
                                    "gap_start":    _se["stuck_from"],
                                    "gap_mins":     _se["gap_mins"],
                                    "se_idx":       _se_idx,
                                })
                    _missed_df = pd.DataFrame(_missed_rows)
                    _n_should  = int((_missed_df["decision"] == "Should accept").sum()) if not _missed_df.empty else 0

                    if _n_stuck == 0:
                        st.success(f"No stuck events this week -- {_n_trips} trips completed cleanly.")
                    else:
                        if _n_should > 0:
                            st.markdown(
                                f'<div style="background:#fef2f2;border-left:4px solid #ef4444;'
                                f'padding:12px 16px;border-radius:6px;color:#1e1e2e;margin-bottom:8px;">'
                                f'<strong>Red flag: {_n_should} high-value ping{"s" if _n_should>1 else ""} declined during stuck windows</strong> '
                                f'(fare >= 20, or fare >= 15 + west of Charing Cross)'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        # MAP
                        st.markdown("#### Stuck locations map")
                        st.caption("Orange/red bubbles = stuck windows (click to see pings)  |  Red = should accept  |  Yellow = borderline  |  Green = fine to decline  |  Filled dot = pickup, hollow = dropoff  |  Toggle layers top-right")

                        _m_s = folium.Map(location=[51.505, -0.13], zoom_start=11, tiles="CartoDB positron")
                        folium.GeoJson(GEOJSON_DATA, style_function=lambda f: {
                            "fillColor": "#e2e8f0", "fillOpacity": 0.04, "color": "#cbd5e1", "weight": 0.6,
                        }, name="Zone overlay").add_to(_m_s)
                        folium.Marker(
                            [51.507, _WEST_LON],
                            icon=folium.DivIcon(
                                html='<div style="color:#3b82f6;font-size:10px;white-space:nowrap;">← W | E →</div>',
                                icon_size=(64, 14), icon_anchor=(32, 7),
                            ),
                        ).add_to(_m_s)

                        _cls_colors = {
                            "Should accept":   "#ef4444",
                            "Borderline":      "#f59e0b",
                            "Fine to decline": "#22c55e",
                        }
                        _cls_labels = {
                            "Should accept": "SHOULD ACCEPT",
                            "Borderline":    "Borderline",
                            "Fine to decline": "OK",
                        }

                        # Separate FeatureGroups so user can toggle each layer
                        _fg_stk = folium.FeatureGroup(name="Stuck windows",          show=True)
                        _fg_sa  = folium.FeatureGroup(name="Should accept pings",    show=True)
                        _fg_bl  = folium.FeatureGroup(name="Borderline pings",       show=True)
                        _fg_ftd = folium.FeatureGroup(name="Fine to decline pings",  show=False)
                        _fg_map = {"Should accept": _fg_sa, "Borderline": _fg_bl, "Fine to decline": _fg_ftd}

                        # Ping lines + dots → routed into their FeatureGroup
                        if not _missed_df.empty:
                            for _, _mp in _missed_df.iterrows():
                                _mc   = _cls_colors[_mp["decision"]]
                                _fg   = _fg_map[_mp["decision"]]
                                _mpz  = _mp.get("pickup_zone")
                                _mdz  = _mp.get("dropoff_zone")
                                _mrte = (f"Zone {int(_mpz)} → Zone {int(_mdz)}"
                                         if _mpz and _mdz else
                                         f"{'West' if _mp['pickup_west'] else 'East'} → {'West' if _mp['dropoff_west'] else 'East'}")
                                _mfl  = f"~£{_mp['fare']:.2f} (est.)" if _mp.get("fare_est") else f"£{_mp['fare']:.2f}"
                                _mlbl = _cls_labels[_mp["decision"]]
                                _popup_p = folium.Popup(
                                    html=(
                                        f'<div style="font-family:sans-serif;min-width:200px;">'
                                        f'<b style="font-size:13px;color:{_mc};">{_mlbl}</b>'
                                        f'<hr style="margin:5px 0;border-color:#ddd;">'
                                        f'<div style="font-size:12px;line-height:1.8;">'
                                        f'<b>Time:</b> {_mp["ping_time"].strftime("%H:%M")}<br>'
                                        f'<b>Fare:</b> {_mfl}<br>'
                                        f'<b>Route:</b> {_mrte}</div></div>'
                                    ),
                                    max_width=250,
                                )
                                _has_pu = _mp["plat"] is not None and not pd.isna(_mp["plat"]) and 51.0 < _mp["plat"] < 52.0
                                _has_do = _mp["dlat"] is not None and not pd.isna(_mp["dlat"]) and 51.0 < _mp["dlat"] < 52.0
                                if _has_pu:
                                    folium.CircleMarker(
                                        [_mp["plat"], _mp["plon"]], radius=6,
                                        color=_mc, fill=True, fill_opacity=0.85, weight=1,
                                        tooltip=f"{_mlbl} | {_mfl} | {_mrte}",
                                        popup=_popup_p,
                                    ).add_to(_fg)
                                if _has_do:
                                    folium.CircleMarker(
                                        [_mp["dlat"], _mp["dlon"]], radius=4,
                                        color=_mc, fill=False, fill_opacity=0, weight=1.5,
                                        tooltip=f"Dropoff | {_mrte}",
                                    ).add_to(_fg)
                                if _has_pu and _has_do:
                                    folium.PolyLine(
                                        [[_mp["plat"], _mp["plon"]], [_mp["dlat"], _mp["dlon"]]],
                                        color=_mc, weight=1.5, opacity=0.45, dash_array="5",
                                    ).add_to(_fg)

                        # Stuck bubbles → stuck FeatureGroup, popup contains pings
                        for _se_idx, _se in _se_stuck.iterrows():
                            if _se["stuck_lat"] is None or pd.isna(_se["stuck_lat"]):
                                continue
                            _se_pings = _missed_df[_missed_df["se_idx"] == _se_idx] if not _missed_df.empty else pd.DataFrame()
                            if not _se_pings.empty:
                                _popup_rows = ""
                                for _, _sp in _se_pings.sort_values("ping_time").iterrows():
                                    _cc   = _cls_colors[_sp["decision"]]
                                    _lbl  = _cls_labels[_sp["decision"]]
                                    _spz  = _sp.get("pickup_zone")
                                    _sdz  = _sp.get("dropoff_zone")
                                    _srte = (f"Z{int(_spz)}→Z{int(_sdz)}"
                                             if _spz and _sdz else
                                             f"{'W' if _sp['pickup_west'] else 'E'}→{'W' if _sp['dropoff_west'] else 'E'}")
                                    _sfl  = f"~£{_sp['fare']:.2f}" if _sp.get("fare_est") else f"£{_sp['fare']:.2f}"
                                    _popup_rows += (
                                        f'<tr style="border-bottom:1px solid #eee;">'
                                        f'<td style="padding:3px 6px;color:#555;font-size:11px;">{_sp["ping_time"].strftime("%H:%M")}</td>'
                                        f'<td style="padding:3px 6px;font-weight:bold;font-size:12px;">{_sfl}</td>'
                                        f'<td style="padding:3px 6px;color:#555;font-size:11px;">{_srte}</td>'
                                        f'<td style="padding:3px 6px;font-weight:bold;color:{_cc};font-size:11px;">{_lbl}</td>'
                                        f'</tr>'
                                    )
                                _ping_table = (
                                    f'<table style="width:100%;border-collapse:collapse;font-family:sans-serif;">'
                                    f'<tr style="background:#f5f5f5;font-size:11px;">'
                                    f'<th style="padding:4px 6px;text-align:left;">Time</th>'
                                    f'<th style="padding:4px 6px;text-align:left;">Fare</th>'
                                    f'<th style="padding:4px 6px;text-align:left;">Route</th>'
                                    f'<th style="padding:4px 6px;text-align:left;">Call</th></tr>'
                                    f'{_popup_rows}</table>'
                                )
                            else:
                                _ping_table = '<p style="color:#999;font-size:12px;font-family:sans-serif;margin:4px 0;">No pings in this window</p>'

                            _airport_badge = (
                                f'<span style="background:#ef4444;color:#fff;font-size:10px;'
                                f'padding:2px 6px;border-radius:3px;margin-left:6px;">STUCK AT {str(_se["airport"]).upper()}</span>'
                                if _se["airport"] else ""
                            )
                            _popup_html = (
                                f'<div style="min-width:300px;font-family:sans-serif;">'
                                f'<b style="font-size:13px;">'
                                f'{pd.Timestamp(_se["stuck_from"]).strftime("%a %d %b %H:%M")}'
                                f' → {pd.Timestamp(_se["stuck_until"]).strftime("%H:%M")}'
                                f' ({_se["gap_mins"]:.0f} min)</b>{_airport_badge}'
                                f'<hr style="margin:6px 0;border-color:#ddd;">'
                                f'<div style="font-size:11px;color:#666;margin-bottom:4px;">PINGS DURING THIS WINDOW</div>'
                                f'{_ping_table}</div>'
                            )
                            _radius = max(8, min(20, _se["gap_mins"] / 4))
                            _sc     = "#dc2626" if _se["airport"] else "#f97316"
                            folium.CircleMarker(
                                [_se["stuck_lat"], _se["stuck_lon"]], radius=_radius,
                                color=_sc, fill=True, fill_opacity=0.55, weight=2.5,
                                tooltip=f"Click for pings | {_se['gap_mins']:.0f} min | {pd.Timestamp(_se['stuck_from']).strftime('%H:%M')}–{pd.Timestamp(_se['stuck_until']).strftime('%H:%M')}",
                                popup=folium.Popup(html=_popup_html, max_width=380),
                            ).add_to(_fg_stk)

                        _fg_stk.add_to(_m_s)
                        _fg_sa.add_to(_m_s)
                        _fg_bl.add_to(_m_s)
                        _fg_ftd.add_to(_m_s)
                        folium.LayerControl(position="topright", collapsed=False).add_to(_m_s)
                        st_folium(_m_s, width="100%", height=560, returned_objects=[])

                        # Gap timeline
                        if not _se_df.empty:
                            _fig_tl = px.bar(
                                _se_df, x="stuck_from", y="gap_mins",
                                color="bucket",
                                color_discrete_map={"<25m": "#22c55e", "25-75m": "#f97316", ">75m": "#ef4444"},
                                category_orders={"bucket": ["<25m", "25-75m", ">75m"]},
                                title="All gaps this week -- coloured by duration bucket",
                                labels={"stuck_from": "Time", "gap_mins": "Gap (min)", "bucket": "Category"},
                                height=280,
                            )
                            _fig_tl.add_hline(y=25, line_dash="dash", line_color="#f97316",
                                              annotation_text="25 min", annotation_position="top right")
                            _fig_tl.add_hline(y=75, line_dash="dash", line_color="#ef4444",
                                              annotation_text="75 min", annotation_position="top right")
                            st.plotly_chart(_fig_tl, use_container_width=True)

                        # Stuck events table
                        st.markdown("#### Stuck events this week")
                        _se_tbl = _se_stuck.copy()
                        _se_tbl["Day & time"]    = pd.to_datetime(_se_tbl["stuck_from"]).dt.strftime("%a %d %b %H:%M")
                        _se_tbl["Until"]         = pd.to_datetime(_se_tbl["stuck_until"]).dt.strftime("%H:%M")
                        _se_tbl["Gap (min)"]     = _se_tbl["gap_mins"]
                        _se_tbl["Prev fare"]     = _se_tbl["prev_fare"].apply(lambda x: f"GBP{x:.2f}")
                        _se_tbl["Stuck at"]      = _se_tbl.apply(
                            lambda r: f"Airport ({r['airport']})" if r["airport"]
                            else ("West" if r["west"] else "East"),
                            axis=1,
                        )
                        _se_tbl["Eventual ride"] = _se_tbl.apply(
                            lambda r: (
                                f"Z{int(r['next_pz'])} → Z{int(r['next_dz'])}  GBP{r['next_fare']:.2f}"
                                if r.get("next_pz") and r.get("next_dz")
                                else f"GBP{r['next_fare']:.2f}"
                            ), axis=1,
                        )
                        st.dataframe(
                            _se_tbl[["Day & time", "Until", "Gap (min)", "Prev fare", "Stuck at", "Eventual ride"]],
                            use_container_width=True, hide_index=True,
                        )

                        # Missed pings table
                        st.markdown("#### Pings missed during stuck windows")
                        if not _missed_df.empty:
                            _mp_tbl = _missed_df.copy()
                            _mp_tbl["Time"]  = _mp_tbl["ping_time"].dt.strftime("%a %H:%M")
                            _mp_tbl["Fare"]  = _mp_tbl.apply(
                                lambda r: f"~GBP{r['fare']:.2f}" if r.get("fare_est") else f"GBP{r['fare']:.2f}",
                                axis=1,
                            )
                            _mp_tbl["Route"] = _mp_tbl.apply(
                                lambda r: (
                                    f"Zone {int(r['pickup_zone'])} → Zone {int(r['dropoff_zone'])}"
                                    if r.get("pickup_zone") and r.get("dropoff_zone") else
                                    f"{'West' if r['pickup_west'] else 'East'} → {'West' if r['dropoff_west'] else '?'}"
                                ), axis=1,
                            )
                            _mp_tbl["Decision"]   = _mp_tbl["decision"].map({
                                "Should accept":   "RED - Should accept",
                                "Borderline":      "YELLOW - Borderline",
                                "Fine to decline": "OK - Fine to decline",
                            })
                            _mp_tbl["During gap"] = _mp_tbl["gap_start"].apply(
                                lambda x: pd.Timestamp(x).strftime("%a %H:%M") if pd.notna(x) else "---"
                            )
                            st.dataframe(
                                _mp_tbl[["Time","Fare","Route","Decision","During gap"]].sort_values("Time"),
                                use_container_width=True, hide_index=True,
                            )

                            _dec_counts = _missed_df["decision"].value_counts().reset_index()
                            _dec_counts.columns = ["Decision", "Count"]
                            _fig_dec = px.bar(
                                _dec_counts, x="Decision", y="Count",
                                color="Decision",
                                color_discrete_map={
                                    "Should accept":   "#ef4444",
                                    "Borderline":      "#f59e0b",
                                    "Fine to decline": "#22c55e",
                                },
                                title="Declined ping classification during stuck windows",
                                height=280,
                            )
                            _fig_dec.update_layout(showlegend=False)
                            st.plotly_chart(_fig_dec, use_container_width=True)
                        else:
                            st.info("No declined pings found during stuck windows.")

    # ── PART 2: FLEET COMPARISON ──────────────────────────────────────────────
    st.divider()
    st.subheader("Fleet comparison -- last 30 days")
    st.caption("Stuck rate, gap quality, airport stranding, and revenue lost for Top 10 vs Comparison vs Rest of fleet.")

    with st.spinner("Loading fleet trip data (30 days)..."):
        _fc_top_acc  = db.load_gap_accepted(list(TOP_DRIVER_IDS), days_back=30)
        _fc_cmp_acc  = db.load_gap_accepted(list(BAD_DRIVER_IDS), days_back=30)
        _fc_all_ids  = db.load_fleet_driver_ids(days_back=30)
        _fc_rest_ids = [i for i in _fc_all_ids if i not in set(TOP_DRIVER_IDS) and i not in set(BAD_DRIVER_IDS)]
        _fc_rest_acc = db.load_gap_accepted(_fc_rest_ids, days_back=30)

    with st.spinner("Loading declined pings for fleet..."):
        _fc_top_dec  = db.load_gap_declined(list(TOP_DRIVER_IDS),  days_back=30)
        _fc_cmp_dec  = db.load_gap_declined(list(BAD_DRIVER_IDS),  days_back=30)
        _fc_rest_dec = db.load_gap_declined(_fc_rest_ids, days_back=30)

    def _fleet_stuck_stats(acc_df):
        if acc_df.empty:
            return {"stuck_rate":0,"lhr_rate":0,"avg_stuck":0,"median_gap":0,
                    "pct_lt25":0,"west_stuck_pct":0,"est_lost_per_driver":0}
        df = acc_df.copy()
        df["pickedup_trip_datetime"] = pd.to_datetime(df["pickedup_trip_datetime"])
        df["dropoff_trip_datetime"]  = pd.to_datetime(df["dropoff_trip_datetime"])
        df = df.sort_values(["dim_driver_id","pickedup_trip_datetime"])
        all_gaps, stuck, lhr_stuck, west_stuck = [], [], [], []
        for _, grp in df.groupby("dim_driver_id"):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                gap = (grp.iloc[i]["pickedup_trip_datetime"] - grp.iloc[i-1]["dropoff_trip_datetime"]).total_seconds()/60
                if gap < 0: continue
                all_gaps.append(gap)
                if 25 <= gap <= 75:
                    dc      = parse_dms(str(grp.iloc[i-1].get("dropoff_latlong","") or ""))
                    airport = _is_airport(dc[0], dc[1])
                    w       = dc[1] is not None and dc[1] < _WEST_LON
                    stuck.append(gap)
                    if airport: lhr_stuck.append(gap)
                    if w:       west_stuck.append(gap)
        n_gaps = len(all_gaps)
        n_stuck = len(stuck)
        n_drv = max(acc_df["dim_driver_id"].nunique(), 1)
        est_lost = (sum(stuck) / 60 * _FLEET_RPH_DEFAULT) / n_drv
        return {
            "stuck_rate":          round(n_stuck / max(n_gaps,1) * 100, 1),
            "lhr_rate":            round(len(lhr_stuck) / max(n_stuck,1) * 100, 1),
            "avg_stuck":           round(np.mean(stuck), 1) if stuck else 0,
            "median_gap":          round(np.median(all_gaps), 1) if all_gaps else 0,
            "pct_lt25":            round(sum(1 for g in all_gaps if g < 25) / max(n_gaps,1) * 100, 1),
            "west_stuck_pct":      round(len(west_stuck) / max(n_stuck,1) * 100, 1),
            "est_lost_per_driver": round(est_lost, 0),
        }

    with st.spinner("Computing fleet stuck stats..."):
        _fs_top  = _fleet_stuck_stats(_fc_top_acc)
        _fs_cmp  = _fleet_stuck_stats(_fc_cmp_acc)
        _fs_rest = _fleet_stuck_stats(_fc_rest_acc)

    # Scorecard row
    st.markdown("#### Stuck-event summary cards -- three groups")
    _fsc1, _fsc2, _fsc3 = st.columns(3)
    for _col, _lbl, _color, _fs in [
        (_fsc1, "TOP 10",        "#22c55e", _fs_top),
        (_fsc2, "REST OF FLEET", "#94a3b8", _fs_rest),
        (_fsc3, "COMPARISON",    "#ef4444", _fs_cmp),
    ]:
        _rows2 = [
            ("Stuck rate (25-75 min)", f"{_fs['stuck_rate']:.1f}%"),
            ("Free-flowing gaps <25m", f"{_fs['pct_lt25']:.1f}%"),
            ("Median gap",             f"{_fs['median_gap']:.0f} min"),
            ("Avg stuck duration",     f"{_fs['avg_stuck']:.0f} min"),
            ("Airport stuck %",        f"{_fs['lhr_rate']:.1f}%"),
            ("West when stuck",        f"{_fs['west_stuck_pct']:.1f}%"),
            ("Est. lost/driver (30d)", f"GBP{_fs['est_lost_per_driver']:.0f}"),
        ]
        _body2 = "".join(
            f'<tr><td style="padding:4px 0;color:#94a3b8;font-size:12px;">{k}</td>'
            f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
            for k, v in _rows2
        )
        _col.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {_color};border-radius:8px;padding:14px 16px;">'
            f'<div style="color:{_color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:6px;">{_lbl}</div>'
            f'<table style="width:100%;border-collapse:collapse;">{_body2}</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gap distribution stacked bar
    _gb_top30  = _gap_buckets(_compute_gaps(_fc_top_acc))
    _gb_cmp30  = _gap_buckets(_compute_gaps(_fc_cmp_acc))
    _gb_rest30 = _gap_buckets(_compute_gaps(_fc_rest_acc))

    _gd30 = pd.DataFrame([
        {"Group": "Top 10",        "<25m": _gb_top30["<25m"],  "25-75m": _gb_top30["25-75m"],  ">75m": _gb_top30[">75m"]},
        {"Group": "Rest of fleet", "<25m": _gb_rest30["<25m"], "25-75m": _gb_rest30["25-75m"], ">75m": _gb_rest30[">75m"]},
        {"Group": "Comparison",    "<25m": _gb_cmp30["<25m"],  "25-75m": _gb_cmp30["25-75m"],  ">75m": _gb_cmp30[">75m"]},
    ]).melt(id_vars="Group", var_name="Bucket", value_name="% of gaps")

    _fig_gd30 = px.bar(
        _gd30, x="Group", y="% of gaps", color="Bucket", barmode="stack",
        color_discrete_map={"<25m": "#22c55e", "25-75m": "#f97316", ">75m": "#ef4444"},
        category_orders={"Group": ["Top 10","Rest of fleet","Comparison"]},
        title="Gap distribution -- 30 days", height=320,
    )
    _fig_gd30.update_layout(yaxis_ticksuffix="%", legend_title="Gap bucket")
    st.plotly_chart(_fig_gd30, use_container_width=True)

    # Stuck rate + free-flowing side by side
    _sr_df = pd.DataFrame([
        {"Group":"Top 10",       "Stuck %": _fs_top["stuck_rate"],  "Free-flowing %": _fs_top["pct_lt25"]},
        {"Group":"Rest of fleet","Stuck %": _fs_rest["stuck_rate"], "Free-flowing %": _fs_rest["pct_lt25"]},
        {"Group":"Comparison",   "Stuck %": _fs_cmp["stuck_rate"],  "Free-flowing %": _fs_cmp["pct_lt25"]},
    ])
    _sr_c1, _sr_c2 = st.columns(2)
    with _sr_c1:
        _fig_sr = px.bar(_sr_df, x="Group", y="Stuck %", color="Group",
                         color_discrete_map={"Top 10":"#22c55e","Rest of fleet":"#94a3b8","Comparison":"#ef4444"},
                         text="Stuck %", title="Stuck rate -- % of gaps 25-75 min", height=300)
        _fig_sr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_sr.update_layout(showlegend=False, yaxis_ticksuffix="%")
        st.plotly_chart(_fig_sr, use_container_width=True)
    with _sr_c2:
        _fig_ff = px.bar(_sr_df, x="Group", y="Free-flowing %", color="Group",
                         color_discrete_map={"Top 10":"#22c55e","Rest of fleet":"#94a3b8","Comparison":"#ef4444"},
                         text="Free-flowing %", title="Free-flowing -- % of gaps under 25 min", height=300)
        _fig_ff.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_ff.update_layout(showlegend=False, yaxis_ticksuffix="%")
        st.plotly_chart(_fig_ff, use_container_width=True)

    # Airport + west-when-stuck
    _ap_df = pd.DataFrame([
        {"Group":"Top 10",       "Airport stuck %": _fs_top["lhr_rate"],  "West when stuck %": _fs_top["west_stuck_pct"]},
        {"Group":"Rest of fleet","Airport stuck %": _fs_rest["lhr_rate"], "West when stuck %": _fs_rest["west_stuck_pct"]},
        {"Group":"Comparison",   "Airport stuck %": _fs_cmp["lhr_rate"],  "West when stuck %": _fs_cmp["west_stuck_pct"]},
    ])
    _ap_c1, _ap_c2 = st.columns(2)
    with _ap_c1:
        _fig_ap = px.bar(_ap_df, x="Group", y="Airport stuck %", color="Group",
                         color_discrete_map={"Top 10":"#22c55e","Rest of fleet":"#94a3b8","Comparison":"#ef4444"},
                         text="Airport stuck %", title="% of stuck events at Heathrow / Gatwick", height=300)
        _fig_ap.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_ap.update_layout(showlegend=False, yaxis_ticksuffix="%")
        st.plotly_chart(_fig_ap, use_container_width=True)
    with _ap_c2:
        _fig_ws = px.bar(_ap_df, x="Group", y="West when stuck %", color="Group",
                         color_discrete_map={"Top 10":"#22c55e","Rest of fleet":"#94a3b8","Comparison":"#ef4444"},
                         text="West when stuck %", title="% stuck in west London (better recovery zone)", height=300)
        _fig_ws.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_ws.update_layout(showlegend=False, yaxis_ticksuffix="%")
        st.plotly_chart(_fig_ws, use_container_width=True)

    # Revenue lost per driver
    _el_df = pd.DataFrame([
        {"Group":"Top 10",       "Est lost / driver GBP": _fs_top["est_lost_per_driver"]},
        {"Group":"Rest of fleet","Est lost / driver GBP": _fs_rest["est_lost_per_driver"]},
        {"Group":"Comparison",   "Est lost / driver GBP": _fs_cmp["est_lost_per_driver"]},
    ])
    _fig_el = px.bar(_el_df, x="Group", y="Est lost / driver GBP", color="Group",
                     color_discrete_map={"Top 10":"#22c55e","Rest of fleet":"#94a3b8","Comparison":"#ef4444"},
                     text="Est lost / driver GBP",
                     title="Estimated revenue lost per driver from stuck events (30 days)", height=300)
    _fig_el.update_traces(texttemplate="GBP%{text:.0f}", textposition="outside")
    _fig_el.update_layout(showlegend=False, yaxis_title="GBP lost")
    st.plotly_chart(_fig_el, use_container_width=True)
    st.caption(f"Estimate: total stuck minutes x fleet avg RPH (GBP{_FLEET_RPH_DEFAULT}/hr) / driver count. Assumes driver was available during gap.")

    # Quality of declined pings
    st.markdown("#### Quality of pings declined -- what is each group passing on?")
    _dec_fare_rows = []
    for _gl, _df_dec in [("Top 10",_fc_top_dec),("Rest of fleet",_fc_rest_dec),("Comparison",_fc_cmp_dec)]:
        if _df_dec.empty: continue
        _f = _df_dec["trip_price_in_pound"].dropna()
        _f = _f[_f > 0]
        if len(_f) == 0: continue
        for _band, _mask in [("Sub-10",_f<10),("10-20",(_f>=10)&(_f<20)),("20-30",(_f>=20)&(_f<30)),("30+",_f>=30)]:
            _dec_fare_rows.append({"Group":_gl, "Fare band":_band, "Pct of declined":round(_mask.mean()*100,1)})
    if _dec_fare_rows:
        _fig_dfq = px.bar(
            pd.DataFrame(_dec_fare_rows), x="Group", y="Pct of declined",
            color="Fare band", barmode="stack",
            color_discrete_map={"Sub-10":"#ef4444","10-20":"#f59e0b","20-30":"#60a5fa","30+":"#22c55e"},
            category_orders={"Group":["Top 10","Rest of fleet","Comparison"]},
            title="Fare bands of pings each group is declining",
            height=320,
        )
        _fig_dfq.update_layout(yaxis_ticksuffix="%", legend_title="Fare band")
        st.plotly_chart(_fig_dfq, use_container_width=True)

    # Per-driver median gap bar
    st.markdown("#### Median gap per driver (last 30 days)")
    _pd_rows = []
    for _gl, _gdf in [("Top 10",_fc_top_acc),("Comparison",_fc_cmp_acc),("Rest of fleet",_fc_rest_acc)]:
        if _gdf.empty: continue
        for _did, _dgrp in _gdf.groupby("dim_driver_id"):
            _dg  = _compute_gaps(_dgrp)
            _dgb = _gap_buckets(_dg)
            _db_name = (
                _dgrp["driver_full_name"].dropna().iloc[0]
                if "driver_full_name" in _dgrp.columns and not _dgrp["driver_full_name"].dropna().empty
                else None
            )
            _dname = DRIVER_NAMES.get(_did) or _db_name or str(_did)
            _pd_rows.append({
                "Driver":     _dname,
                "Group":      _gl,
                "Median gap": _dgb["median"],
                "Stuck %":    _dgb["25-75m"],
            })
    if _pd_rows:
        _pd_df = pd.DataFrame(_pd_rows).sort_values("Median gap")
        _fig_pd = px.bar(
            _pd_df, x="Median gap", y="Driver", orientation="h",
            color="Group",
            color_discrete_map={"Top 10":"#22c55e","Comparison":"#ef4444","Rest of fleet":"#94a3b8"},
            text="Median gap",
            title="Median inter-trip gap per driver -- last 30 days",
            height=max(400, len(_pd_df)*22+80),
        )
        _fig_pd.update_traces(texttemplate="%{text:.0f} min", textposition="outside")
        _fig_pd.update_layout(xaxis_title="Median gap (min)", yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(_fig_pd, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Zone Analysis":
    st.title("Zone Analysis")
    st.caption("Why Zone 3 is a trap, exactly which parts, which drivers keep ending up there, and whether the inner-East theory holds.")

    with st.spinner("Loading 30 days of fleet trips (cached after first load)..."):
        _za_raw = _load_zone_trips(days_back=30)

    if _za_raw.empty:
        st.error("No trip data found.")
        st.stop()

    with st.spinner("Assigning zones and computing gaps..."):
        _za_raw = _za_raw.copy()
        _za_raw["pickedup_trip_datetime"] = pd.to_datetime(_za_raw["pickedup_trip_datetime"])
        _za_raw["dropoff_trip_datetime"]  = pd.to_datetime(_za_raw["dropoff_trip_datetime"])
        _pu = _za_raw["pickup_lat_long"].apply(lambda x: parse_dms(str(x or "")))
        _do = _za_raw["dropoff_latlong"].apply(lambda x: parse_dms(str(x or "")))
        _za_raw["plat"] = [c[0] for c in _pu]
        _za_raw["plon"] = [c[1] for c in _pu]
        _za_raw["dlat"] = [c[0] for c in _do]
        _za_raw["dlon"] = [c[1] for c in _do]
        _za = _za_raw.dropna(subset=["plat","plon","dlat","dlon"]).copy()
        _za = _za[
            (51.2 < _za["dlat"]) & (_za["dlat"] < 51.8) &
            (51.2 < _za["plat"]) & (_za["plat"] < 51.8)
        ].reset_index(drop=True)
        _za["pickup_zone"]  = [assign_zone(lat, lon) for lat, lon in zip(_za["plat"], _za["plon"])]
        _za["dropoff_zone"] = [assign_zone(lat, lon) for lat, lon in zip(_za["dlat"], _za["dlon"])]
        _za["dropoff_west"] = _za["dlon"] < _WEST_LON
        _za["pickup_west"]  = _za["plon"] < _WEST_LON
        _za["fare"]         = pd.to_numeric(_za["trip_price_in_pound"], errors="coerce").fillna(0)

        # Gap to next trip per driver
        _za = _za.sort_values(["dim_driver_id","pickedup_trip_datetime"]).reset_index(drop=True)
        _za["next_pickup"] = _za.groupby("dim_driver_id")["pickedup_trip_datetime"].shift(-1)
        _za["gap_after"]   = (_za["next_pickup"] - _za["dropoff_trip_datetime"]).dt.total_seconds() / 60
        _za["gap_after"]   = _za["gap_after"].clip(0, 90)  # cap at 90 min

        # Handy labels
        _za["dropoff_side"]  = _za["dlon"].apply(lambda x: "West" if x < _WEST_LON else "East")
        _za["zone_side"]     = "Z" + _za["dropoff_zone"].astype(str) + " " + _za["dropoff_side"]

    # ── SECTION 1: Avg gap by zone ────────────────────────────────────────────
    st.subheader("1 — Average wait time after dropping off, by zone")
    st.caption("How long does a driver typically sit idle after each zone? Lower = better.")

    _gap_by_zone = (
        _za.dropna(subset=["gap_after","dropoff_zone"])
        .groupby("dropoff_zone")["gap_after"]
        .agg(avg_gap="mean", median_gap="median", count="count")
        .reset_index()
        .sort_values("dropoff_zone")
    )
    _gap_by_zone["Zone"] = "Zone " + _gap_by_zone["dropoff_zone"].astype(str)
    _color_z = ["#22c55e","#4ade80","#f59e0b","#fb923c","#ef4444","#dc2626"]
    _fig_gbz = px.bar(
        _gap_by_zone, x="Zone", y="avg_gap",
        color="Zone",
        color_discrete_sequence=_color_z,
        text="avg_gap",
        title="Average gap after dropoff — all fleet drivers, 30 days",
        labels={"avg_gap": "Avg gap (min)", "Zone": ""},
        height=320,
    )
    _fig_gbz.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    _fig_gbz.update_layout(showlegend=False, yaxis_title="Minutes waiting")
    st.plotly_chart(_fig_gbz, use_container_width=True)

    # Also show fare by zone
    _fare_by_zone = (
        _za[_za["fare"] > 0]
        .groupby("dropoff_zone")["fare"]
        .agg(avg_fare="mean", median_fare="median")
        .reset_index()
        .sort_values("dropoff_zone")
    )
    _fare_by_zone["Zone"] = "Zone " + _fare_by_zone["dropoff_zone"].astype(str)
    _fz_c1, _fz_c2 = st.columns(2)
    with _fz_c1:
        _fig_fz = px.bar(
            _fare_by_zone, x="Zone", y="avg_fare",
            color="Zone", color_discrete_sequence=_color_z,
            text="avg_fare", title="Avg fare by dropoff zone",
            labels={"avg_fare": "Avg fare (£)"}, height=280,
        )
        _fig_fz.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
        _fig_fz.update_layout(showlegend=False)
        st.plotly_chart(_fig_fz, use_container_width=True)
    with _fz_c2:
        # % of trips by zone
        _zone_counts = _za["dropoff_zone"].value_counts().reset_index()
        _zone_counts.columns = ["dropoff_zone","count"]
        _zone_counts["Zone"] = "Zone " + _zone_counts["dropoff_zone"].astype(str)
        _fig_zc = px.pie(
            _zone_counts.sort_values("dropoff_zone"), values="count", names="Zone",
            color="Zone",
            color_discrete_map={f"Zone {i}": _color_z[i-1] for i in range(1,7)},
            title="Share of all dropoffs by zone",
            height=280,
        )
        _fig_zc.update_traces(textinfo="percent+label")
        _fig_zc.update_layout(showlegend=False)
        st.plotly_chart(_fig_zc, use_container_width=True)

    st.divider()

    # ── SECTION 2: East vs West within each zone ──────────────────────────────
    st.subheader("2 — The inner-East theory: does it hold?")
    st.caption("Avg gap per zone split by East vs West. If inner East is fine but outer East is bad, you'll see it here.")

    _zone_side_gap = (
        _za.dropna(subset=["gap_after","dropoff_zone"])
        .groupby(["dropoff_zone","dropoff_side"])["gap_after"]
        .agg(avg_gap="mean", trips="count")
        .reset_index()
    )
    _zone_side_gap["Zone"] = "Zone " + _zone_side_gap["dropoff_zone"].astype(str)
    _zone_side_gap = _zone_side_gap[_zone_side_gap["trips"] >= 10]  # exclude tiny samples

    _fig_ew = px.bar(
        _zone_side_gap.sort_values(["dropoff_zone","dropoff_side"]),
        x="Zone", y="avg_gap",
        color="dropoff_side",
        barmode="group",
        color_discrete_map={"East": "#ef4444", "West": "#3b82f6"},
        text="avg_gap",
        title="Avg gap after dropoff — East vs West, per zone",
        labels={"avg_gap": "Avg gap (min)", "dropoff_side": "Side"},
        height=360,
    )
    _fig_ew.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    _fig_ew.update_layout(yaxis_title="Minutes waiting")
    st.plotly_chart(_fig_ew, use_container_width=True)

    # Write the conclusion in a callout
    _z1e = _zone_side_gap[(_zone_side_gap["dropoff_zone"]==1) & (_zone_side_gap["dropoff_side"]=="East")]["avg_gap"].values
    _z3e = _zone_side_gap[(_zone_side_gap["dropoff_zone"]==3) & (_zone_side_gap["dropoff_side"]=="East")]["avg_gap"].values
    _z3w = _zone_side_gap[(_zone_side_gap["dropoff_zone"]==3) & (_zone_side_gap["dropoff_side"]=="West")]["avg_gap"].values
    _z1e_v = float(_z1e[0]) if len(_z1e) else None
    _z3e_v = float(_z3e[0]) if len(_z3e) else None
    _z3w_v = float(_z3w[0]) if len(_z3w) else None

    if _z1e_v and _z3e_v:
        _diff = _z3e_v - _z1e_v
        st.markdown(
            f'<div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;'
            f'border-radius:6px;color:#1e1e2e;margin-bottom:8px;">'
            f'<strong>Verdict:</strong> Zone 3 East averages <strong>{_z3e_v:.0f} min</strong> idle vs '
            f'<strong>{_z1e_v:.0f} min</strong> in Zone 1 East — a <strong>+{_diff:.0f} min penalty</strong> '
            f'per drop. '
            + (f'Zone 3 West ({_z3w_v:.0f} min) is meaningfully better than Zone 3 East.'
               if _z3w_v and _z3w_v < _z3e_v else "") +
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── SECTION 3: Zone 3 dead zone map ──────────────────────────────────────
    st.subheader("3 — Where exactly in Zone 3 are drivers getting stuck?")
    st.caption("Each dot is a Zone 3 dropoff. Red = long wait, green = quick next trip. Hover for details.")

    _z3_map = _za[(_za["dropoff_zone"] == 3) & _za["gap_after"].notna()].copy()

    if not _z3_map.empty:
        _z3_map["gap_label"] = _z3_map["gap_after"].apply(lambda x: f"{x:.0f} min gap")
        _z3_map["side_label"] = _z3_map["dropoff_side"]

        _fig_z3m = px.scatter_mapbox(
            _z3_map, lat="dlat", lon="dlon",
            color="gap_after",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            range_color=[0, 50],
            size="gap_after",
            size_max=12,
            hover_name="driver_full_name",
            hover_data={"gap_after": ":.0f", "fare": ":.2f", "dropoff_side": True, "dlat": False, "dlon": False},
            mapbox_style="carto-positron",
            zoom=10.5,
            center={"lat": 51.49, "lon": -0.02},
            title="Zone 3 dropoffs — coloured by wait time after drop (red = stuck)",
            height=540,
            labels={"gap_after": "Gap (min)", "fare": "Fare £", "dropoff_side": "Side", "driver_full_name": "Driver"},
        )
        _fig_z3m.update_layout(
            coloraxis_colorbar=dict(title="Gap (min)", ticksuffix=" min"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig_z3m, use_container_width=True)
    else:
        st.info("No Zone 3 dropoff data with gap information.")

    st.divider()

    # ── SECTION 4: Zone 3 East vs West deep-dive ─────────────────────────────
    st.subheader("4 — Zone 3 East vs Zone 3 West: detailed breakdown")

    _z3e_df = _za[(_za["dropoff_zone"] == 3) & (_za["dropoff_side"] == "East")].copy()
    _z3w_df = _za[(_za["dropoff_zone"] == 3) & (_za["dropoff_side"] == "West")].copy()

    # Recovery rate = % of NEXT trips that pick up in Zone 1 or 2
    def _recovery_rate(df):
        """% of next pickups in Zone 1/2 (high-demand zone)."""
        if df.empty or "next_pickup" not in df.columns:
            return 0
        _nxt = df.dropna(subset=["next_pickup"]).copy()
        if _nxt.empty:
            return 0
        _nxt_pz = [assign_zone(lat, lon) for lat, lon in zip(_nxt["plat"], _nxt["plon"])]
        return round(sum(1 for z in _nxt_pz if z and z <= 2) / max(len(_nxt_pz), 1) * 100, 1)

    def _stuck_pct(df):
        _g = df["gap_after"].dropna()
        return round((((_g >= 25) & (_g <= 75)).sum()) / max(len(_g), 1) * 100, 1)

    _east_stats = {
        "Trips dropped here":    len(_z3e_df),
        "Avg gap after":         f"{_z3e_df['gap_after'].mean():.1f} min" if not _z3e_df.empty else "—",
        "Median gap after":      f"{_z3e_df['gap_after'].median():.1f} min" if not _z3e_df.empty else "—",
        "Stuck rate (25-75 min)":f"{_stuck_pct(_z3e_df):.1f}%",
        "Avg fare for that trip":f"£{_z3e_df['fare'][_z3e_df['fare']>0].mean():.2f}" if (_z3e_df["fare"]>0).any() else "—",
        "Recovery to Z1/Z2":     f"{_recovery_rate(_z3e_df):.1f}%",
    }
    _west_stats = {
        "Trips dropped here":    len(_z3w_df),
        "Avg gap after":         f"{_z3w_df['gap_after'].mean():.1f} min" if not _z3w_df.empty else "—",
        "Median gap after":      f"{_z3w_df['gap_after'].median():.1f} min" if not _z3w_df.empty else "—",
        "Stuck rate (25-75 min)":f"{_stuck_pct(_z3w_df):.1f}%",
        "Avg fare for that trip":f"£{_z3w_df['fare'][_z3w_df['fare']>0].mean():.2f}" if (_z3w_df["fare"]>0).any() else "—",
        "Recovery to Z1/Z2":     f"{_recovery_rate(_z3w_df):.1f}%",
    }

    _s4c1, _s4c2 = st.columns(2)
    for _col, _lbl, _color, _stats in [
        (_s4c1, "ZONE 3 EAST", "#ef4444", _east_stats),
        (_s4c2, "ZONE 3 WEST", "#3b82f6", _west_stats),
    ]:
        _rows3 = "".join(
            f'<tr><td style="padding:5px 0;color:#94a3b8;font-size:12px;">{k}</td>'
            f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
            for k, v in _stats.items()
        )
        _col.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {_color};border-radius:8px;padding:14px 16px;">'
            f'<div style="color:{_color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:8px;">{_lbl}</div>'
            f'<table style="width:100%;border-collapse:collapse;">{_rows3}</table></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gap distribution: Z3 East vs West
    _z3_both = _za[_za["dropoff_zone"] == 3].dropna(subset=["gap_after"]).copy()
    if not _z3_both.empty:
        _fig_z3hist = px.histogram(
            _z3_both, x="gap_after", color="dropoff_side",
            barmode="overlay",
            color_discrete_map={"East": "#ef4444", "West": "#3b82f6"},
            nbins=30,
            opacity=0.7,
            title="Gap distribution — Zone 3 East vs West",
            labels={"gap_after": "Gap after dropoff (min)", "dropoff_side": "Side"},
            height=300,
        )
        _fig_z3hist.add_vline(x=25, line_dash="dash", line_color="#f59e0b", annotation_text="25 min")
        _fig_z3hist.add_vline(x=75, line_dash="dash", line_color="#dc2626", annotation_text="75 min")
        st.plotly_chart(_fig_z3hist, use_container_width=True)

    st.divider()

    # ── SECTION 5: Worst Zone 3 East offenders ───────────────────────────────
    st.subheader("5 — Drivers most frequently ending up in Zone 3 East")
    st.caption("Ranked by what % of their drops land in Zone 3 East. These are the drivers to have a conversation with.")

    _drv_z3e = (
        _za.groupby("dim_driver_id")
        .apply(lambda g: pd.Series({
            "name":       g["driver_full_name"].dropna().iloc[0] if not g["driver_full_name"].dropna().empty else str(g["dim_driver_id"].iloc[0]),
            "total_drops":len(g),
            "z3e_drops":  int(((g["dropoff_zone"] == 3) & (g["dropoff_side"] == "East")).sum()),
            "avg_gap_z3e": g.loc[(g["dropoff_zone"]==3) & (g["dropoff_side"]=="East"), "gap_after"].mean(),
        }))
        .reset_index(drop=True)
    )
    _drv_z3e["z3e_pct"] = (_drv_z3e["z3e_drops"] / _drv_z3e["total_drops"] * 100).round(1)
    _drv_z3e = _drv_z3e[_drv_z3e["z3e_drops"] >= 3].sort_values("z3e_pct", ascending=False).head(20)

    if not _drv_z3e.empty:
        _drv_z3e["avg_gap_z3e"] = _drv_z3e["avg_gap_z3e"].fillna(0).round(1)
        _fig_off = px.bar(
            _drv_z3e, x="z3e_pct", y="name", orientation="h",
            color="avg_gap_z3e",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            text="z3e_pct",
            title="Zone 3 East drop rate per driver (min. 3 trips there)",
            labels={"z3e_pct": "% of drops in Z3 East", "name": "Driver", "avg_gap_z3e": "Avg gap after (min)"},
            height=max(320, len(_drv_z3e) * 26 + 80),
        )
        _fig_off.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_off.update_layout(
            xaxis_title="% of all drops in Zone 3 East",
            yaxis={"categoryorder": "total ascending"},
            coloraxis_colorbar=dict(title="Avg gap (min)"),
        )
        st.plotly_chart(_fig_off, use_container_width=True)

        # Table version
        _off_tbl = _drv_z3e.copy()
        _off_tbl["Z3E drops"]      = _off_tbl["z3e_drops"].astype(int)
        _off_tbl["% of their drops"] = _off_tbl["z3e_pct"].apply(lambda x: f"{x:.1f}%")
        _off_tbl["Avg gap after"]  = _off_tbl["avg_gap_z3e"].apply(lambda x: f"{x:.0f} min" if x else "—")
        st.dataframe(
            _off_tbl[["name","Z3E drops","% of their drops","Avg gap after"]].rename(columns={"name":"Driver"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Not enough Zone 3 East drop data to rank drivers.")

    st.divider()

    # ── SECTION 6: Where do drivers go AFTER Zone 3 East? ────────────────────
    st.subheader("6 — Recovery: where does the next trip start after Zone 3?")
    st.caption("After dropping in Zone 3 East, can drivers get back to Zone 1/2 quickly — or do they drift further out?")

    _z3e_next = _za[((_za["dropoff_zone"] == 3) & (_za["dropoff_side"] == "East"))].copy()
    _z3e_next = _z3e_next.dropna(subset=["next_pickup"])

    if not _z3e_next.empty:
        _z3e_next["next_pz"] = [assign_zone(lat, lon) for lat, lon in zip(_z3e_next["plat"], _z3e_next["plon"])]
        _z3e_next_pz = _z3e_next["next_pz"].dropna().astype(int)
        _next_zone_counts = _z3e_next_pz.value_counts().sort_index().reset_index()
        _next_zone_counts.columns = ["Zone", "Count"]
        _next_zone_counts["Zone label"] = "Zone " + _next_zone_counts["Zone"].astype(str)
        _next_zone_counts["pct"] = (_next_zone_counts["Count"] / _next_zone_counts["Count"].sum() * 100).round(1)

        _fig_nxt = px.bar(
            _next_zone_counts, x="Zone label", y="pct",
            color="Zone label",
            color_discrete_map={f"Zone {i}": _color_z[i-1] for i in range(1,7)},
            text="pct",
            title="Next pickup zone after dropping in Zone 3 East",
            labels={"pct": "% of next trips", "Zone label": ""},
            height=320,
        )
        _fig_nxt.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_nxt.update_layout(showlegend=False, yaxis_title="% of next trips")
        st.plotly_chart(_fig_nxt, use_container_width=True)

        _back_z12 = _next_zone_counts[_next_zone_counts["Zone"] <= 2]["pct"].sum()
        _stay_z3  = _next_zone_counts[_next_zone_counts["Zone"] == 3]["pct"].sum()
        _further  = _next_zone_counts[_next_zone_counts["Zone"] > 3]["pct"].sum()
        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("Recover to Z1/Z2",  f"{_back_z12:.1f}%", help="Good outcome")
        _r2.metric("Stay in Zone 3",    f"{_stay_z3:.1f}%",  help="Neutral")
        _r3.metric("Drift to Zone 4+",  f"{_further:.1f}%",  help="Bad outcome — gets worse")
    else:
        st.info("No Zone 3 East next-trip data found.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # ZONE 1 ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.header("Zone 1 — The premium core")
    st.caption("Zone 1 is where the highest fares originate. The question isn't whether to be here — it's whether you can hold position and cycle back after each drop.")

    st.markdown(
        '<div style="background:#f0fdf4;border-left:4px solid #22c55e;padding:12px 16px;'
        'border-radius:6px;color:#1e1e2e;margin-bottom:16px;">'
        '<strong>Why Zone 1 is different:</strong> Zone 1 pickups carry the highest avg fares and the shortest gaps. '
        'The trap isn\'t that Zone 1 is bad — it\'s that most drivers can\'t stay there. After a Zone 1 drop '
        'the trip pulls you outward. Good drivers dead-head back or hold position before going available. '
        'Most drift to Zone 2 or 3 and never return that shift.'
        '</div>',
        unsafe_allow_html=True,
    )

    def _pct_of_zone_list(zone_list, target, op="eq"):
        if not zone_list:
            return "—"
        n = len(zone_list)
        if op == "eq":
            c = sum(1 for z in zone_list if z and z == target)
        elif op == "ge":
            c = sum(1 for z in zone_list if z and z >= target)
        elif op == "le":
            c = sum(1 for z in zone_list if z and z <= target)
        else:
            return "—"
        return f"{c / n * 100:.1f}%"

    # Zone 1 East vs West subsets
    _z1e_df = _za[(_za["dropoff_zone"] == 1) & (_za["dropoff_side"] == "East")].copy()
    _z1w_df = _za[(_za["dropoff_zone"] == 1) & (_za["dropoff_side"] == "West")].copy()

    _z1e_nxt_df = _z1e_df.dropna(subset=["next_pickup"])
    _z1w_nxt_df = _z1w_df.dropna(subset=["next_pickup"])
    _z1e_nxt_pz = ([assign_zone(lat, lon) for lat, lon in zip(_z1e_nxt_df["plat"], _z1e_nxt_df["plon"])]
                   if not _z1e_nxt_df.empty else [])
    _z1w_nxt_pz = ([assign_zone(lat, lon) for lat, lon in zip(_z1w_nxt_df["plat"], _z1w_nxt_df["plon"])]
                   if not _z1w_nxt_df.empty else [])

    # ── Stats cards ─────────────────────────────────────────────────────────
    st.subheader("1 — Zone 1 East vs Zone 1 West: do they behave differently?")
    st.caption("Zone 1 East = City of London corridor. Zone 1 West = Mayfair / Soho / South Bank.")

    _z1e_stats = {
        "Trips dropped here":      len(_z1e_df),
        "Avg fare for that trip":  f"£{_z1e_df['fare'][_z1e_df['fare']>0].mean():.2f}" if (_z1e_df["fare"]>0).any() else "—",
        "Avg gap after":           f"{_z1e_df['gap_after'].mean():.1f} min" if not _z1e_df.empty else "—",
        "Stuck rate (25-75 min)":  f"{_stuck_pct(_z1e_df):.1f}%",
        "Cycles back to Z1":       _pct_of_zone_list(_z1e_nxt_pz, 1),
        "Drifts to Z3+":           _pct_of_zone_list(_z1e_nxt_pz, 3, "ge"),
    }
    _z1w_stats = {
        "Trips dropped here":      len(_z1w_df),
        "Avg fare for that trip":  f"£{_z1w_df['fare'][_z1w_df['fare']>0].mean():.2f}" if (_z1w_df["fare"]>0).any() else "—",
        "Avg gap after":           f"{_z1w_df['gap_after'].mean():.1f} min" if not _z1w_df.empty else "—",
        "Stuck rate (25-75 min)":  f"{_stuck_pct(_z1w_df):.1f}%",
        "Cycles back to Z1":       _pct_of_zone_list(_z1w_nxt_pz, 1),
        "Drifts to Z3+":           _pct_of_zone_list(_z1w_nxt_pz, 3, "ge"),
    }

    _z1c1, _z1c2 = st.columns(2)
    for _col, _lbl, _color, _stats in [
        (_z1c1, "ZONE 1 EAST", "#f97316", _z1e_stats),
        (_z1c2, "ZONE 1 WEST", "#22c55e", _z1w_stats),
    ]:
        _rows_z1 = "".join(
            f'<tr><td style="padding:5px 0;color:#94a3b8;font-size:12px;">{k}</td>'
            f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
            for k, v in _stats.items()
        )
        _col.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {_color};border-radius:8px;padding:14px 16px;">'
            f'<div style="color:{_color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:8px;">{_lbl}</div>'
            f'<table style="width:100%;border-collapse:collapse;">{_rows_z1}</table></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Zone 1 fare map ──────────────────────────────────────────────────────
    st.subheader("2 — Where in Zone 1 are the best fares dropping?")
    st.caption("Each dot is a Zone 1 dropoff, coloured by fare. Red = premium destination.")

    _z1_map_df = _za[(_za["dropoff_zone"] == 1) & (_za["fare"] > 0)].copy()
    if not _z1_map_df.empty:
        _fig_z1m = px.scatter_mapbox(
            _z1_map_df, lat="dlat", lon="dlon",
            color="fare",
            color_continuous_scale=["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"],
            range_color=[_z1_map_df["fare"].quantile(0.1), _z1_map_df["fare"].quantile(0.9)],
            size_max=10,
            opacity=0.65,
            hover_name="driver_full_name",
            hover_data={"fare": ":.2f", "gap_after": ":.0f", "dropoff_side": True, "dlat": False, "dlon": False},
            mapbox_style="carto-positron",
            zoom=11.5,
            center={"lat": 51.505, "lon": -0.115},
            title="Zone 1 dropoffs — coloured by fare (red = premium drop)",
            height=520,
            labels={"fare": "Fare £", "gap_after": "Gap (min)", "dropoff_side": "Side", "driver_full_name": "Driver"},
        )
        _fig_z1m.update_layout(
            coloraxis_colorbar=dict(title="Fare £", tickprefix="£"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig_z1m, use_container_width=True)
    else:
        st.info("No Zone 1 dropoff data with fare information.")

    st.divider()

    # ── Who commands Zone 1? ─────────────────────────────────────────────────
    st.subheader("3 — Which drivers pick up in Zone 1 the most?")
    st.caption("Ranked by % of all trips starting in Zone 1. High Z1 pickup % = driver gravitates toward premium corridors.")

    _z1_pu = (
        _za.groupby("dim_driver_id")
        .apply(lambda g: pd.Series({
            "name":        g["driver_full_name"].dropna().iloc[0] if not g["driver_full_name"].dropna().empty else str(g["dim_driver_id"].iloc[0]),
            "total_trips": len(g),
            "z1_pickups":  int((g["pickup_zone"] == 1).sum()),
            "avg_fare_z1": g.loc[(g["pickup_zone"] == 1) & (g["fare"] > 0), "fare"].mean(),
        }))
        .reset_index(drop=True)
    )
    _z1_pu["z1_pct"] = (_z1_pu["z1_pickups"] / _z1_pu["total_trips"] * 100).round(1)
    _z1_pu = _z1_pu[_z1_pu["z1_pickups"] >= 5].sort_values("z1_pct", ascending=False).head(20)

    if not _z1_pu.empty:
        _z1_pu["avg_fare_z1"] = _z1_pu["avg_fare_z1"].fillna(0).round(2)
        _fig_z1d = px.bar(
            _z1_pu, x="z1_pct", y="name", orientation="h",
            color="avg_fare_z1",
            color_continuous_scale=["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"],
            text="z1_pct",
            title="Zone 1 pickup rate per driver (min. 5 Z1 trips)",
            labels={"z1_pct": "% of trips from Z1", "name": "Driver", "avg_fare_z1": "Avg Z1 fare (£)"},
            height=max(320, len(_z1_pu) * 26 + 80),
        )
        _fig_z1d.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_z1d.update_layout(
            xaxis_title="% of all pickups in Zone 1",
            yaxis={"categoryorder": "total ascending"},
            coloraxis_colorbar=dict(title="Avg Z1 fare", tickprefix="£"),
        )
        st.plotly_chart(_fig_z1d, use_container_width=True)
    else:
        st.info("Not enough Zone 1 pickup data.")

    st.divider()

    # ── After Zone 1 dropoff, where does the next trip start? ────────────────
    st.subheader("4 — After dropping in Zone 1, where does the next trip start?")
    st.caption("The cycle-back rate: do drivers reposition into Zone 1 again, or drift outward after each Z1 drop?")

    _z1_all_nxt = _za[_za["dropoff_zone"] == 1].dropna(subset=["next_pickup"]).copy()
    if not _z1_all_nxt.empty:
        _z1_all_nxt["next_pz"] = [assign_zone(lat, lon) for lat, lon in zip(_z1_all_nxt["plat"], _z1_all_nxt["plon"])]
        _z1_nxt_pz_series = _z1_all_nxt["next_pz"].dropna().astype(int)
        _z1_nxt_counts = _z1_nxt_pz_series.value_counts().sort_index().reset_index()
        _z1_nxt_counts.columns = ["Zone", "Count"]
        _z1_nxt_counts["Zone label"] = "Zone " + _z1_nxt_counts["Zone"].astype(str)
        _z1_nxt_counts["pct"] = (_z1_nxt_counts["Count"] / _z1_nxt_counts["Count"].sum() * 100).round(1)

        _fig_z1nxt = px.bar(
            _z1_nxt_counts, x="Zone label", y="pct",
            color="Zone label",
            color_discrete_map={f"Zone {i}": _color_z[i-1] for i in range(1, 7)},
            text="pct",
            title="Next pickup zone after a Zone 1 dropoff",
            labels={"pct": "% of next trips", "Zone label": ""},
            height=320,
        )
        _fig_z1nxt.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_z1nxt.update_layout(showlegend=False, yaxis_title="% of next trips")
        st.plotly_chart(_fig_z1nxt, use_container_width=True)

        _z1_back = _z1_nxt_counts[_z1_nxt_counts["Zone"] == 1]["pct"].sum()
        _z1_z2   = _z1_nxt_counts[_z1_nxt_counts["Zone"] == 2]["pct"].sum()
        _z1_z3p  = _z1_nxt_counts[_z1_nxt_counts["Zone"] >= 3]["pct"].sum()
        _rz1a, _rz1b, _rz1c = st.columns(3)
        _rz1a.metric("Cycle back to Z1", f"{_z1_back:.1f}%", help="Best outcome — retained premium zone")
        _rz1b.metric("Slide to Z2",       f"{_z1_z2:.1f}%",  help="Acceptable — still close in")
        _rz1c.metric("Drift to Z3+",      f"{_z1_z3p:.1f}%", help="Concern — likely won't return this shift")
    else:
        st.info("No Zone 1 next-trip data found.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # ZONE 2 ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.header("Zone 2 — The buffer zone")
    st.caption("Zone 2 is neither trap nor premium — it's a fork in the road. Zone 2 West funnels toward Zone 1. Zone 2 East funnels toward Zone 3.")

    st.markdown(
        '<div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;'
        'border-radius:6px;color:#1e1e2e;margin-bottom:16px;">'
        '<strong>The Zone 2 fork:</strong> Zone 2 West (Kensington, Chelsea, Fulham, Hammersmith) sits adjacent to Zone 1 West — '
        'a short repositioning move can capture premium pings. Zone 2 East (Bethnal Green, Hackney, Stratford) puts '
        'drivers in the same gravity well as Zone 3 East. Which side of Zone 2 a driver drifts into is a leading '
        'indicator: it predicts whether their next trip pulls them toward Zone 1 or further into the trap.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Zone 2 East vs West subsets
    _z2e_df = _za[(_za["dropoff_zone"] == 2) & (_za["dropoff_side"] == "East")].copy()
    _z2w_df = _za[(_za["dropoff_zone"] == 2) & (_za["dropoff_side"] == "West")].copy()

    _z2e_nxt_df = _z2e_df.dropna(subset=["next_pickup"])
    _z2w_nxt_df = _z2w_df.dropna(subset=["next_pickup"])
    _z2e_nxt_pz = ([assign_zone(lat, lon) for lat, lon in zip(_z2e_nxt_df["plat"], _z2e_nxt_df["plon"])]
                   if not _z2e_nxt_df.empty else [])
    _z2w_nxt_pz = ([assign_zone(lat, lon) for lat, lon in zip(_z2w_nxt_df["plat"], _z2w_nxt_df["plon"])]
                   if not _z2w_nxt_df.empty else [])

    # ── Stats cards ─────────────────────────────────────────────────────────
    st.subheader("1 — Zone 2 East vs Zone 2 West: the directional split")
    st.caption("Same zone number, completely different trajectories.")

    _z2e_stats = {
        "Trips dropped here":      len(_z2e_df),
        "Avg fare for that trip":  f"£{_z2e_df['fare'][_z2e_df['fare']>0].mean():.2f}" if (_z2e_df["fare"]>0).any() else "—",
        "Avg gap after":           f"{_z2e_df['gap_after'].mean():.1f} min" if not _z2e_df.empty else "—",
        "Stuck rate (25-75 min)":  f"{_stuck_pct(_z2e_df):.1f}%",
        "Next trip stays Z1/Z2":   _pct_of_zone_list(_z2e_nxt_pz, 2, "le"),
        "Next trip goes Z3+":      _pct_of_zone_list(_z2e_nxt_pz, 3, "ge"),
    }
    _z2w_stats = {
        "Trips dropped here":      len(_z2w_df),
        "Avg fare for that trip":  f"£{_z2w_df['fare'][_z2w_df['fare']>0].mean():.2f}" if (_z2w_df["fare"]>0).any() else "—",
        "Avg gap after":           f"{_z2w_df['gap_after'].mean():.1f} min" if not _z2w_df.empty else "—",
        "Stuck rate (25-75 min)":  f"{_stuck_pct(_z2w_df):.1f}%",
        "Next trip stays Z1/Z2":   _pct_of_zone_list(_z2w_nxt_pz, 2, "le"),
        "Next trip goes Z3+":      _pct_of_zone_list(_z2w_nxt_pz, 3, "ge"),
    }

    _z2c1, _z2c2 = st.columns(2)
    for _col, _lbl, _color, _stats in [
        (_z2c1, "ZONE 2 EAST", "#ef4444", _z2e_stats),
        (_z2c2, "ZONE 2 WEST", "#3b82f6", _z2w_stats),
    ]:
        _rows_z2 = "".join(
            f'<tr><td style="padding:5px 0;color:#94a3b8;font-size:12px;">{k}</td>'
            f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
            for k, v in _stats.items()
        )
        _col.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {_color};border-radius:8px;padding:14px 16px;">'
            f'<div style="color:{_color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:8px;">{_lbl}</div>'
            f'<table style="width:100%;border-collapse:collapse;">{_rows_z2}</table></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Zone 2 gap map ───────────────────────────────────────────────────────
    st.subheader("2 — Where in Zone 2 are drivers getting stuck?")
    st.caption("Each dot is a Zone 2 dropoff, coloured by gap after. Red = long wait, green = quick next trip.")

    _z2_map_df = _za[(_za["dropoff_zone"] == 2) & _za["gap_after"].notna()].copy()
    if not _z2_map_df.empty:
        _fig_z2m = px.scatter_mapbox(
            _z2_map_df, lat="dlat", lon="dlon",
            color="gap_after",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            range_color=[0, 50],
            size="gap_after",
            size_max=12,
            opacity=0.65,
            hover_name="driver_full_name",
            hover_data={"gap_after": ":.0f", "fare": ":.2f", "dropoff_side": True, "dlat": False, "dlon": False},
            mapbox_style="carto-positron",
            zoom=10.5,
            center={"lat": 51.505, "lon": -0.09},
            title="Zone 2 dropoffs — coloured by wait time after drop (red = stuck)",
            height=540,
            labels={"gap_after": "Gap (min)", "fare": "Fare £", "dropoff_side": "Side", "driver_full_name": "Driver"},
        )
        _fig_z2m.update_layout(
            coloraxis_colorbar=dict(title="Gap (min)", ticksuffix=" min"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig_z2m, use_container_width=True)
    else:
        st.info("No Zone 2 dropoff data with gap information.")

    st.divider()

    # ── Directional proof: Z2 East vs Z2 West → next pickup zone ─────────────
    st.subheader("3 — After Zone 2, which direction do drivers go?")
    st.caption("The fork visualised: Zone 2 West pushes drivers toward Zone 1; Zone 2 East pulls toward Zone 3.")

    _z2dir_c1, _z2dir_c2 = st.columns(2)
    for _col, _nxt_pz, _lbl in [
        (_z2dir_c1, _z2e_nxt_pz, "After Zone 2 East drop"),
        (_z2dir_c2, _z2w_nxt_pz, "After Zone 2 West drop"),
    ]:
        if _nxt_pz:
            _nxt_s = pd.Series([z for z in _nxt_pz if z]).astype(int).value_counts().sort_index().reset_index()
            _nxt_s.columns = ["Zone", "Count"]
            _nxt_s["Zone label"] = "Zone " + _nxt_s["Zone"].astype(str)
            _nxt_s["pct"] = (_nxt_s["Count"] / _nxt_s["Count"].sum() * 100).round(1)
            _fig_dir = px.bar(
                _nxt_s, x="Zone label", y="pct",
                color="Zone label",
                color_discrete_map={f"Zone {i}": _color_z[i-1] for i in range(1, 7)},
                text="pct",
                title=_lbl,
                labels={"pct": "% of next trips", "Zone label": ""},
                height=300,
            )
            _fig_dir.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            _fig_dir.update_layout(showlegend=False, yaxis_title="% of next trips")
            _col.plotly_chart(_fig_dir, use_container_width=True)
        else:
            _col.info(f"No data for {_lbl}.")

    st.divider()

    # ── Zone 2 East offenders ────────────────────────────────────────────────
    st.subheader("4 — Drivers most frequently dropping in Zone 2 East")
    st.caption("Zone 2 East drops predict Zone 3 drift. These are the drivers to watch — they're already on the wrong side of the fork.")

    _drv_z2e = (
        _za.groupby("dim_driver_id")
        .apply(lambda g: pd.Series({
            "name":        g["driver_full_name"].dropna().iloc[0] if not g["driver_full_name"].dropna().empty else str(g["dim_driver_id"].iloc[0]),
            "total_drops": len(g),
            "z2e_drops":   int(((g["dropoff_zone"] == 2) & (g["dropoff_side"] == "East")).sum()),
            "avg_gap_z2e": g.loc[(g["dropoff_zone"] == 2) & (g["dropoff_side"] == "East"), "gap_after"].mean(),
        }))
        .reset_index(drop=True)
    )
    _drv_z2e["z2e_pct"] = (_drv_z2e["z2e_drops"] / _drv_z2e["total_drops"] * 100).round(1)
    _drv_z2e = _drv_z2e[_drv_z2e["z2e_drops"] >= 3].sort_values("z2e_pct", ascending=False).head(20)

    if not _drv_z2e.empty:
        _drv_z2e["avg_gap_z2e"] = _drv_z2e["avg_gap_z2e"].fillna(0).round(1)
        _fig_z2off = px.bar(
            _drv_z2e, x="z2e_pct", y="name", orientation="h",
            color="avg_gap_z2e",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            text="z2e_pct",
            title="Zone 2 East drop rate per driver (min. 3 trips there)",
            labels={"z2e_pct": "% of drops in Z2 East", "name": "Driver", "avg_gap_z2e": "Avg gap after (min)"},
            height=max(320, len(_drv_z2e) * 26 + 80),
        )
        _fig_z2off.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _fig_z2off.update_layout(
            xaxis_title="% of all drops in Zone 2 East",
            yaxis={"categoryorder": "total ascending"},
            coloraxis_colorbar=dict(title="Avg gap (min)"),
        )
        st.plotly_chart(_fig_z2off, use_container_width=True)

        _z2e_tbl = _drv_z2e.copy()
        _z2e_tbl["Z2E drops"]        = _z2e_tbl["z2e_drops"].astype(int)
        _z2e_tbl["% of their drops"] = _z2e_tbl["z2e_pct"].apply(lambda x: f"{x:.1f}%")
        _z2e_tbl["Avg gap after"]    = _z2e_tbl["avg_gap_z2e"].apply(lambda x: f"{x:.0f} min" if x else "—")
        st.dataframe(
            _z2e_tbl[["name", "Z2E drops", "% of their drops", "Avg gap after"]].rename(columns={"name": "Driver"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Not enough Zone 2 East drop data to rank drivers.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Z3 DRIFT ANALYSIS — justified vs avoidable
    # ══════════════════════════════════════════════════════════════════════════
    st.header("Zone 3 drift — justified vs avoidable")
    st.caption("Of drivers who drift into Z3 after a Z1 or Z2 drop, how many had a legitimate reason — and how many just fell into the trap?")

    st.markdown(
        '<div style="background:#fef2f2;border-left:4px solid #ef4444;padding:12px 16px;'
        'border-radius:6px;color:#1e1e2e;margin-bottom:16px;">'
        '<strong>Not all Z3 trips are wrong.</strong> A £25 fare that happens to start in Zone 3 is fine. '
        'A night-shift Z3 pickup (00:00–06:00) is valid. An airport run is valid. '
        'The problem is the driver who just dropped in Z1, accepted the nearest cheap Z3 ping, '
        'and is now stuck for 30 minutes on a £9 fare. '
        '<strong>That</strong> is the avoidable drift — and it\'s the only number that matters here.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Build drift trips: prev drop was Z1 or Z2, this pickup is Z3
    _za["prev_dropoff_zone"] = _za.groupby("dim_driver_id")["dropoff_zone"].shift(1)
    _drift = _za[
        (_za["prev_dropoff_zone"].isin([1, 2])) &
        (_za["pickup_zone"] == 3)
    ].copy()

    if _drift.empty:
        st.info("No Z3 drift trips found in the last 30 days.")
    else:
        # Classification
        _drift["is_airport"] = (
            ((_drift["plat"].between(51.45, 51.49)) & (_drift["plon"].between(-0.50, -0.42))) |
            ((_drift["plat"].between(51.13, 51.18)) & (_drift["plon"].between(-0.22, -0.14)))
        )
        _drift["is_night"] = pd.to_datetime(_drift["pickedup_trip_datetime"]).dt.hour < 6
        _drift["classification"] = "Borderline"
        _drift.loc[
            (_drift["fare"] < 12) & ~_drift["is_airport"] & ~_drift["is_night"],
            "classification"
        ] = "Avoidable"
        _drift.loc[
            (_drift["fare"] >= 20) | _drift["is_airport"] | _drift["is_night"],
            "classification"
        ] = "Justified"

        _total_drifts   = len(_drift)
        _avoid_count    = int((_drift["classification"] == "Avoidable").sum())
        _justified_count= int((_drift["classification"] == "Justified").sum())
        _border_count   = int((_drift["classification"] == "Borderline").sum())
        _avoid_pct      = _avoid_count / max(_total_drifts, 1) * 100

        # Headline callout
        st.markdown(
            f'<div style="background:#1e1e2e;border:2px solid #ef4444;border-radius:8px;padding:16px 20px;margin-bottom:16px;">'
            f'<span style="color:#ef4444;font-size:28px;font-weight:bold;">{_avoid_pct:.0f}%</span>'
            f'<span style="color:#f8fafc;font-size:16px;"> of Z3 drifts are entirely avoidable</span><br>'
            f'<span style="color:#94a3b8;font-size:13px;">'
            f'{_avoid_count} avoidable &nbsp;·&nbsp; {_border_count} borderline &nbsp;·&nbsp; {_justified_count} justified &nbsp;·&nbsp; {_total_drifts} total drift trips</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Classification breakdown ─────────────────────────────────────────
        st.subheader("1 — What % of Z3 drifts were justified?")
        st.caption("Justified = fare ≥ £20, or night shift (00:00–06:00), or airport run. Avoidable = fare < £12, daytime, no airport.")

        _class_counts = _drift["classification"].value_counts().reset_index()
        _class_counts.columns = ["Classification", "Count"]
        _class_counts["pct"] = (_class_counts["Count"] / _total_drifts * 100).round(1)
        _class_order = ["Justified", "Borderline", "Avoidable"]
        _class_colors = {"Justified": "#22c55e", "Borderline": "#f59e0b", "Avoidable": "#ef4444"}

        _dc1, _dc2 = st.columns([1, 1])
        with _dc1:
            _fig_donut = px.pie(
                _class_counts, values="Count", names="Classification",
                color="Classification",
                color_discrete_map=_class_colors,
                hole=0.55,
                title="All Z3 drift trips by classification",
                height=300,
            )
            _fig_donut.update_traces(textinfo="percent+label")
            _fig_donut.update_layout(showlegend=False)
            st.plotly_chart(_fig_donut, use_container_width=True)

        with _dc2:
            # Z1-origin vs Z2-origin split
            _drift_origin = (
                _drift.groupby(["prev_dropoff_zone", "classification"])
                .size().reset_index(name="Count")
            )
            _drift_origin["From"] = "From Z" + _drift_origin["prev_dropoff_zone"].astype(int).astype(str) + " drop"
            _fig_origin = px.bar(
                _drift_origin, x="From", y="Count",
                color="classification",
                color_discrete_map=_class_colors,
                barmode="stack",
                title="Z3 drift quality — after Z1 drop vs after Z2 drop",
                labels={"Count": "Trips", "classification": ""},
                height=300,
            )
            _fig_origin.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(_fig_origin, use_container_width=True)

        st.divider()

        # ── Worst offenders by avoidable drift count ─────────────────────────
        st.subheader("2 — Who is drifting avoidably the most?")
        st.caption("Ranked by number of avoidable Z3 drift trips. These are the drivers losing the most revenue to the trap.")

        _drift_drv = (
            _drift.groupby("dim_driver_id")
            .apply(lambda g: pd.Series({
                "name":        g["driver_full_name"].dropna().iloc[0] if not g["driver_full_name"].dropna().empty else str(g["dim_driver_id"].iloc[0]),
                "total_drifts":len(g),
                "avoidable":   int((g["classification"] == "Avoidable").sum()),
                "justified":   int((g["classification"] == "Justified").sum()),
                "borderline":  int((g["classification"] == "Borderline").sum()),
                "avg_fare_avoidable": g.loc[g["classification"] == "Avoidable", "fare"].mean(),
            }))
            .reset_index(drop=True)
        )
        _drift_drv["avoidable_pct"] = (_drift_drv["avoidable"] / _drift_drv["total_drifts"] * 100).round(1)
        _drift_drv = _drift_drv[_drift_drv["total_drifts"] >= 2].sort_values("avoidable", ascending=False).head(20)

        if not _drift_drv.empty:
            _drift_drv["avg_fare_avoidable"] = _drift_drv["avg_fare_avoidable"].fillna(0).round(2)
            _fig_offenders = px.bar(
                _drift_drv, x="avoidable", y="name", orientation="h",
                color="avoidable_pct",
                color_continuous_scale=["#f59e0b", "#ef4444", "#dc2626"],
                text="avoidable",
                title="Avoidable Z3 drifts per driver",
                labels={"avoidable": "Avoidable drift trips", "name": "Driver", "avoidable_pct": "% of their drifts avoidable"},
                height=max(320, len(_drift_drv) * 26 + 80),
            )
            _fig_offenders.update_traces(texttemplate="%{text}", textposition="outside")
            _fig_offenders.update_layout(
                xaxis_title="Count of avoidable Z3 drift trips",
                yaxis={"categoryorder": "total ascending"},
                coloraxis_colorbar=dict(title="% avoidable", ticksuffix="%"),
            )
            st.plotly_chart(_fig_offenders, use_container_width=True)

            _off_tbl = _drift_drv.copy()
            _off_tbl["Avoidable drifts"] = _off_tbl["avoidable"].astype(int)
            _off_tbl["Justified drifts"] = _off_tbl["justified"].astype(int)
            _off_tbl["% avoidable"]      = _off_tbl["avoidable_pct"].apply(lambda x: f"{x:.1f}%")
            _off_tbl["Avg avoidable fare"] = _off_tbl["avg_fare_avoidable"].apply(lambda x: f"£{x:.2f}" if x else "—")
            st.dataframe(
                _off_tbl[["name", "Avoidable drifts", "Justified drifts", "% avoidable", "Avg avoidable fare"]].rename(columns={"name": "Driver"}),
                use_container_width=True, hide_index=True,
            )

        st.divider()

        # ── Map of avoidable drift pickups ────────────────────────────────────
        st.subheader("3 — Where are avoidable Z3 drifts happening?")
        st.caption("Pickup locations of avoidable drift trips only — these are the pings drivers should have declined.")

        _avoid_map = _drift[_drift["classification"] == "Avoidable"].copy()
        if not _avoid_map.empty:
            _fig_dmap = px.scatter_mapbox(
                _avoid_map, lat="plat", lon="plon",
                color="fare",
                color_continuous_scale=["#f59e0b", "#ef4444", "#dc2626"],
                range_color=[0, 12],
                size_max=10,
                opacity=0.7,
                hover_name="driver_full_name",
                hover_data={"fare": ":.2f", "gap_after": ":.0f", "prev_dropoff_zone": True, "plat": False, "plon": False},
                mapbox_style="carto-positron",
                zoom=10.5,
                center={"lat": 51.515, "lon": -0.02},
                title="Avoidable Z3 drift pickup locations (fare < £12, daytime, no airport)",
                height=520,
                labels={"fare": "Fare £", "gap_after": "Gap after (min)", "prev_dropoff_zone": "Came from Zone", "driver_full_name": "Driver"},
            )
            _fig_dmap.update_layout(
                coloraxis_colorbar=dict(title="Fare £", tickprefix="£"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(_fig_dmap, use_container_width=True)
        else:
            st.info("No avoidable Z3 drift pickups to map.")

# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Driver Behavior":
    st.title("Driver Behavior")
    st.caption("Every angle on how good vs bad drivers make decisions — what they accept, what they decline, fare floors, zones, time of day, distance. Raw data; we'll find the signal together.")

    with st.spinner("Loading accepted trips..."):
        _beh_acc_raw = _load_behavior_accepted(days_back=30)
    with st.spinner("Loading declined pings..."):
        _beh_dec_raw = _load_behavior_declines(days_back=30)

    if _beh_acc_raw.empty:
        st.error("No accepted trip data found.")
        st.stop()

    # ── Process accepted trips ────────────────────────────────────────────────
    _beh_acc = _beh_acc_raw.copy()
    _beh_acc["fare"]  = pd.to_numeric(_beh_acc["trip_price_in_pound"], errors="coerce").fillna(0)
    _beh_acc["dist"]  = pd.to_numeric(_beh_acc["distance_in_miles"],   errors="coerce").fillna(0)
    _beh_acc["pickedup_trip_datetime"] = pd.to_datetime(_beh_acc["pickedup_trip_datetime"])
    _beh_acc["hour"]  = _beh_acc["pickedup_trip_datetime"].dt.hour
    _beh_acc["dow"]   = _beh_acc["pickedup_trip_datetime"].dt.day_name()
    _bpu = _beh_acc["pickup_lat_long"].apply(lambda x: parse_dms(str(x or "")))
    _beh_acc["plat"]  = [c[0] for c in _bpu]
    _beh_acc["plon"]  = [c[1] for c in _bpu]
    _beh_acc = _beh_acc.dropna(subset=["plat","plon"])
    _beh_acc["pickup_zone"] = [assign_zone(lat, lon) for lat, lon in zip(_beh_acc["plat"], _beh_acc["plon"])]
    _beh_acc["is_west"] = _beh_acc["plon"] < _WEST_LON
    _beh_acc["group"]   = _beh_acc["dim_driver_id"].apply(
        lambda x: "Good" if x in TOP_DRIVER_IDS else "Bad"
    )
    _beh_acc["fare_band"] = pd.cut(
        _beh_acc["fare"],
        bins=[0, 10, 15, 20, 30, 999],
        labels=["<£10", "£10-15", "£15-20", "£20-30", "£30+"],
        right=True,
    )

    # ── Process declined pings ────────────────────────────────────────────────
    _beh_dec = pd.DataFrame()
    if not _beh_dec_raw.empty:
        _beh_dec = _beh_dec_raw.copy()
        _beh_dec["event_datetime"] = pd.to_datetime(_beh_dec["event_datetime"])
        _beh_dec["hour"] = _beh_dec["event_datetime"].dt.hour
        _dpu = _beh_dec["pickup_lat_long"].apply(lambda x: parse_dms(str(x or "")))
        _ddo = _beh_dec["dropoff_latlong"].apply(lambda x: parse_dms(str(x or "")))
        _beh_dec["plat"] = [c[0] for c in _dpu]
        _beh_dec["plon"] = [c[1] for c in _dpu]
        _beh_dec["dlat"] = [c[0] for c in _ddo]
        _beh_dec["dlon"] = [c[1] for c in _ddo]
        _beh_dec = _beh_dec.dropna(subset=["plat","plon"])
        _beh_dec["pickup_zone"] = [assign_zone(lat, lon) for lat, lon in zip(_beh_dec["plat"], _beh_dec["plon"])]
        _beh_dec["is_west"] = _beh_dec["plon"] < _WEST_LON
        # Estimate fare: use stored value if > 0, else estimate_ping when both coords available
        def _est_fare(row):
            raw = pd.to_numeric(row.get("trip_price_in_pound", 0), errors="coerce") or 0
            if raw > 0:
                return float(raw)
            if row["dlat"] is not None and row["dlon"] is not None:
                try:
                    _, est = estimate_ping(row["plat"], row["plon"], row["dlat"], row["dlon"])
                    return float(est) if est else 0.0
                except Exception:
                    return 0.0
            return 0.0
        _beh_dec["est_fare"] = _beh_dec.apply(_est_fare, axis=1)
        _beh_dec["group"] = _beh_dec["dim_driver_id"].apply(
            lambda x: "Good" if x in TOP_DRIVER_IDS else "Bad"
        )
        _beh_dec["fare_band"] = pd.cut(
            _beh_dec["est_fare"],
            bins=[0, 10, 15, 20, 30, 999],
            labels=["<£10", "£10-15", "£15-20", "£20-30", "£30+"],
            right=True,
        )

    _good_acc = _beh_acc[_beh_acc["group"] == "Good"]
    _bad_acc  = _beh_acc[_beh_acc["group"] == "Bad"]
    _good_dec = _beh_dec[_beh_dec["group"] == "Good"] if not _beh_dec.empty else pd.DataFrame()
    _bad_dec  = _beh_dec[_beh_dec["group"] == "Bad"]  if not _beh_dec.empty else pd.DataFrame()

    _GOOD_C = "#22c55e"
    _BAD_C  = "#ef4444"

    # ── SECTION 1: Headline summary ───────────────────────────────────────────
    st.subheader("1 — The headline numbers")

    def _beh_stat_card(label, color, acc_df, dec_df):
        n_acc   = len(acc_df)
        n_dec   = len(dec_df)
        n_total = n_acc + n_dec
        acc_rate = n_acc / max(n_total, 1) * 100
        avg_fare = acc_df["fare"][acc_df["fare"] > 0].mean() if not acc_df.empty else 0
        sub10    = (acc_df["fare"] < 10).sum() / max(n_acc, 1) * 100
        sub12    = (acc_df["fare"] < 12).sum() / max(n_acc, 1) * 100
        avg_dec_fare = dec_df["est_fare"][dec_df["est_fare"] > 0].mean() if not dec_df.empty else None
        rows = {
            "Trips accepted":         f"{n_acc:,}",
            "Pings declined":          f"{n_dec:,}",
            "Acceptance rate":         f"{acc_rate:.1f}%",
            "Avg accepted fare":       f"£{avg_fare:.2f}",
            "Sub-£10 accepted":        f"{sub10:.1f}% of trips",
            "Sub-£12 accepted":        f"{sub12:.1f}% of trips",
        }
        if avg_dec_fare:
            rows["Avg declined ping fare"] = f"£{avg_dec_fare:.2f}"
        _html = "".join(
            f'<tr><td style="padding:5px 0;color:#94a3b8;font-size:12px;">{k}</td>'
            f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
            for k, v in rows.items()
        )
        return (
            f'<div style="background:#1e1e2e;border:2px solid {color};border-radius:8px;padding:14px 16px;">'
            f'<div style="color:{color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:8px;">{label}</div>'
            f'<table style="width:100%;border-collapse:collapse;">{_html}</table></div>'
        )

    _s1c1, _s1c2 = st.columns(2)
    _s1c1.markdown(_beh_stat_card("GOOD DRIVERS (TOP 10)", _GOOD_C, _good_acc, _good_dec), unsafe_allow_html=True)
    _s1c2.markdown(_beh_stat_card("BAD DRIVERS (COMPARISON)", _BAD_C, _bad_acc, _bad_dec), unsafe_allow_html=True)

    st.divider()

    # ── SECTION 2: Fare distribution — what they accept ───────────────────────
    st.subheader("2 — Fare distribution: what are they actually taking?")
    st.caption("Overlapping histograms of accepted fare. If good drivers have a higher floor, you'll see the green distribution shift right.")

    _fare_both = _beh_acc[_beh_acc["fare"] > 0].copy()
    if not _fare_both.empty:
        _fig_fare = px.histogram(
            _fare_both, x="fare", color="group",
            barmode="overlay",
            color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
            nbins=40,
            opacity=0.65,
            title="Accepted fare distribution — Good vs Bad drivers",
            labels={"fare": "Fare (£)", "group": ""},
            height=360,
        )
        _fig_fare.add_vline(x=10, line_dash="dash", line_color="#f59e0b", annotation_text="£10")
        _fig_fare.add_vline(x=15, line_dash="dash", line_color="#94a3b8", annotation_text="£15")
        _fig_fare.add_vline(x=20, line_dash="dash", line_color="#3b82f6", annotation_text="£20")
        _fig_fare.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(_fig_fare, use_container_width=True)

    # Fare band breakdown side by side
    _fband_good = _good_acc[_good_acc["fare"] > 0]["fare_band"].value_counts().reset_index()
    _fband_bad  = _bad_acc[_bad_acc["fare"] > 0]["fare_band"].value_counts().reset_index()
    _fband_good.columns = ["fare_band", "count"]
    _fband_bad.columns  = ["fare_band", "count"]
    _fband_good["pct"]  = (_fband_good["count"] / _fband_good["count"].sum() * 100).round(1)
    _fband_bad["pct"]   = (_fband_bad["count"]  / _fband_bad["count"].sum()  * 100).round(1)
    _fband_good["group"] = "Good"
    _fband_bad["group"]  = "Bad"
    _fband_all = pd.concat([_fband_good, _fband_bad])
    _band_order = ["<£10","£10-15","£15-20","£20-30","£30+"]

    _fig_fband = px.bar(
        _fband_all, x="fare_band", y="pct",
        color="group", barmode="group",
        color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
        category_orders={"fare_band": _band_order},
        text="pct",
        title="% of accepted trips by fare band",
        labels={"fare_band": "Fare band", "pct": "% of trips", "group": ""},
        height=320,
    )
    _fig_fband.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _fig_fband.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(_fig_fband, use_container_width=True)

    st.divider()

    # ── SECTION 3: What they decline ─────────────────────────────────────────
    st.subheader("3 — What do they decline?")
    st.caption("Estimated fare distribution of declined pings. Key question: are good drivers declining BETTER pings (discipline) or just seeing fewer bad pings (positioning)?")

    if not _beh_dec.empty and "est_fare" in _beh_dec.columns:
        _dec_fares = _beh_dec[_beh_dec["est_fare"] > 0].copy()
        if not _dec_fares.empty:
            _fig_dec = px.histogram(
                _dec_fares, x="est_fare", color="group",
                barmode="overlay",
                color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
                nbins=40,
                opacity=0.65,
                title="Declined ping estimated fare distribution — Good vs Bad drivers",
                labels={"est_fare": "Estimated fare (£)", "group": ""},
                height=340,
            )
            _fig_dec.add_vline(x=10, line_dash="dash", line_color="#f59e0b", annotation_text="£10")
            _fig_dec.add_vline(x=20, line_dash="dash", line_color="#3b82f6", annotation_text="£20")
            _fig_dec.update_layout(legend=dict(orientation="h", y=1.1))
            st.plotly_chart(_fig_dec, use_container_width=True)

            # Avg fare of declines vs accepts side by side
            _comp_c1, _comp_c2 = st.columns(2)
            with _comp_c1:
                _avg_dec_g = _dec_fares[_dec_fares["group"] == "Good"]["est_fare"].mean()
                _avg_dec_b = _dec_fares[_dec_fares["group"] == "Bad"]["est_fare"].mean()
                st.metric("Good drivers — avg declined fare", f"£{_avg_dec_g:.2f}" if _avg_dec_g else "—")
                st.metric("Bad drivers — avg declined fare",  f"£{_avg_dec_b:.2f}" if _avg_dec_b else "—")
            with _comp_c2:
                _avg_acc_g = _good_acc[_good_acc["fare"] > 0]["fare"].mean()
                _avg_acc_b = _bad_acc[_bad_acc["fare"] > 0]["fare"].mean()
                st.metric("Good drivers — avg accepted fare", f"£{_avg_acc_g:.2f}" if _avg_acc_g else "—")
                st.metric("Bad drivers — avg accepted fare",  f"£{_avg_acc_b:.2f}" if _avg_acc_b else "—")
    else:
        st.info("No declined ping data available.")

    st.divider()

    # ── SECTION 4: Acceptance rate by fare band ───────────────────────────────
    st.subheader("4 — Selectivity: acceptance rate at each fare level")
    st.caption("For every fare band, what % of pings do each group actually take? Shows whether good drivers have a real fare floor or just see better pings.")

    if not _beh_dec.empty:
        _acc_bands = _beh_acc[_beh_acc["fare"] > 0].groupby(["group","fare_band"]).size().reset_index(name="accepted")
        _dec_bands = _beh_dec[_beh_dec["est_fare"] > 0].groupby(["group","fare_band"]).size().reset_index(name="declined")
        _sel = pd.merge(_acc_bands, _dec_bands, on=["group","fare_band"], how="outer").fillna(0)
        _sel["total"]    = _sel["accepted"] + _sel["declined"]
        _sel["acc_rate"] = (_sel["accepted"] / _sel["total"] * 100).round(1)
        _sel = _sel[_sel["total"] >= 3]

        if not _sel.empty:
            _fig_sel = px.bar(
                _sel, x="fare_band", y="acc_rate",
                color="group", barmode="group",
                color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
                category_orders={"fare_band": _band_order},
                text="acc_rate",
                title="Acceptance rate by fare band — Good vs Bad",
                labels={"fare_band": "Fare band", "acc_rate": "Acceptance rate (%)", "group": ""},
                height=340,
            )
            _fig_sel.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            _fig_sel.update_layout(legend=dict(orientation="h", y=1.1), yaxis_range=[0, 110])
            st.plotly_chart(_fig_sel, use_container_width=True)
    else:
        st.caption("(Need decline data to compute acceptance rate by band.)")

    st.divider()

    # ── SECTION 5: Zone-based decisions ──────────────────────────────────────
    st.subheader("5 — Where are each group picking up?")
    st.caption("% of accepted trips by pickup zone. Are bad drivers simply working different zones, or the same zones with worse filters?")

    _zone_good = _good_acc["pickup_zone"].dropna().astype(int).value_counts().reset_index()
    _zone_bad  = _bad_acc["pickup_zone"].dropna().astype(int).value_counts().reset_index()
    _zone_good.columns = ["zone","count"]
    _zone_bad.columns  = ["zone","count"]
    _zone_good["pct"]   = (_zone_good["count"] / _zone_good["count"].sum() * 100).round(1)
    _zone_bad["pct"]    = (_zone_bad["count"]  / _zone_bad["count"].sum()  * 100).round(1)
    _zone_good["group"] = "Good"
    _zone_bad["group"]  = "Bad"
    _zone_both = pd.concat([_zone_good, _zone_bad])
    _zone_both["Zone"] = "Z" + _zone_both["zone"].astype(str)

    _fig_zone = px.bar(
        _zone_both.sort_values("zone"), x="Zone", y="pct",
        color="group", barmode="group",
        color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
        text="pct",
        title="Pickup zone distribution — Good vs Bad",
        labels={"Zone": "", "pct": "% of pickups", "group": ""},
        height=340,
    )
    _fig_zone.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _fig_zone.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(_fig_zone, use_container_width=True)

    # Acceptance rate by zone if decline data available
    if not _beh_dec.empty:
        _zacc_g = _good_acc["pickup_zone"].dropna().astype(int).value_counts().rename("accepted")
        _zacc_b = _bad_acc["pickup_zone"].dropna().astype(int).value_counts().rename("accepted")
        _zdec_g = _good_dec["pickup_zone"].dropna().astype(int).value_counts().rename("declined") if not _good_dec.empty else pd.Series(dtype=int)
        _zdec_b = _bad_dec["pickup_zone"].dropna().astype(int).value_counts().rename("declined")  if not _bad_dec.empty else pd.Series(dtype=int)

        def _zone_acc_rate(zacc, zdec, grp):
            _df = pd.concat([zacc, zdec], axis=1).fillna(0)
            _df["total"] = _df["accepted"] + _df["declined"]
            _df["acc_rate"] = (_df["accepted"] / _df["total"] * 100).round(1)
            _df = _df[_df["total"] >= 3].reset_index()
            _df.columns = ["zone","accepted","declined","total","acc_rate"]
            _df["Zone"] = "Z" + _df["zone"].astype(str)
            _df["group"] = grp
            return _df

        _zar_g = _zone_acc_rate(_zacc_g, _zdec_g, "Good")
        _zar_b = _zone_acc_rate(_zacc_b, _zdec_b, "Bad")
        _zar_both = pd.concat([_zar_g, _zar_b])

        if not _zar_both.empty:
            _fig_zar = px.bar(
                _zar_both.sort_values("zone"), x="Zone", y="acc_rate",
                color="group", barmode="group",
                color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
                text="acc_rate",
                title="Acceptance rate by zone — Good vs Bad",
                labels={"Zone": "", "acc_rate": "Acceptance rate (%)", "group": ""},
                height=320,
            )
            _fig_zar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            _fig_zar.update_layout(legend=dict(orientation="h", y=1.1), yaxis_range=[0, 110])
            st.plotly_chart(_fig_zar, use_container_width=True)

    st.divider()

    # ── SECTION 6: East vs West ───────────────────────────────────────────────
    st.subheader("6 — East vs West breakdown")
    st.caption("What % of each group's accepted trips are west of Charing Cross (lon < -0.12)?")

    _ew_data = [
        {"Group": "Good", "Side": "West", "pct": round(_good_acc["is_west"].mean() * 100, 1)},
        {"Group": "Good", "Side": "East", "pct": round((~_good_acc["is_west"]).mean() * 100, 1)},
        {"Group": "Bad",  "Side": "West", "pct": round(_bad_acc["is_west"].mean() * 100, 1)},
        {"Group": "Bad",  "Side": "East", "pct": round((~_bad_acc["is_west"]).mean() * 100, 1)},
    ]
    _ew_df = pd.DataFrame(_ew_data)
    _fig_ew = px.bar(
        _ew_df, x="Group", y="pct", color="Side",
        barmode="stack",
        color_discrete_map={"West": "#3b82f6", "East": "#ef4444"},
        text="pct",
        title="East vs West split of accepted pickups",
        labels={"pct": "%", "Group": ""},
        height=300,
    )
    _fig_ew.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    st.plotly_chart(_fig_ew, use_container_width=True)

    # Avg fare: west vs east, per group
    _ew_fare = (
        _beh_acc[_beh_acc["fare"] > 0]
        .groupby(["group","is_west"])["fare"]
        .mean().round(2).reset_index()
    )
    _ew_fare["Side"] = _ew_fare["is_west"].map({True: "West", False: "East"})
    _fig_ewf = px.bar(
        _ew_fare, x="group", y="fare", color="Side",
        barmode="group",
        color_discrete_map={"West": "#3b82f6", "East": "#ef4444"},
        text="fare",
        title="Avg fare — East vs West, Good vs Bad",
        labels={"fare": "Avg fare (£)", "group": ""},
        height=300,
    )
    _fig_ewf.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
    st.plotly_chart(_fig_ewf, use_container_width=True)

    st.divider()

    # ── SECTION 7: Time of day ────────────────────────────────────────────────
    st.subheader("7 — When are they working?")
    st.caption("Trip volume by hour of day. Shows if each group is optimising shift times differently.")

    _hr_good = _good_acc.groupby("hour").size().reset_index(name="trips")
    _hr_bad  = _bad_acc.groupby("hour").size().reset_index(name="trips")
    _hr_good["group"] = "Good"
    _hr_bad["group"]  = "Bad"
    _hr_both = pd.concat([_hr_good, _hr_bad])

    _fig_hr = px.line(
        _hr_both, x="hour", y="trips", color="group",
        color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
        markers=True,
        title="Trip volume by hour — Good vs Bad",
        labels={"hour": "Hour of day", "trips": "Trips", "group": ""},
        height=320,
    )
    _fig_hr.update_xaxes(dtick=1)
    _fig_hr.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(_fig_hr, use_container_width=True)

    # Avg fare by hour
    _hrf_good = _good_acc[_good_acc["fare"] > 0].groupby("hour")["fare"].mean().reset_index()
    _hrf_bad  = _bad_acc[_bad_acc["fare"] > 0].groupby("hour")["fare"].mean().reset_index()
    _hrf_good["group"] = "Good"
    _hrf_bad["group"]  = "Bad"
    _hrf_both = pd.concat([_hrf_good, _hrf_bad])

    _fig_hrf = px.line(
        _hrf_both, x="hour", y="fare", color="group",
        color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
        markers=True,
        title="Avg accepted fare by hour — Good vs Bad",
        labels={"hour": "Hour of day", "fare": "Avg fare (£)", "group": ""},
        height=300,
    )
    _fig_hrf.add_hline(y=15, line_dash="dash", line_color="#94a3b8", annotation_text="£15")
    _fig_hrf.update_xaxes(dtick=1)
    _fig_hrf.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(_fig_hrf, use_container_width=True)

    st.divider()

    # ── SECTION 8: Trip distance ──────────────────────────────────────────────
    st.subheader("8 — Trip distance: do good drivers avoid short hops?")
    st.caption("Distribution of accepted trip distances. Short trips (< 2 miles) are usually low-fare local hops.")

    _dist_both = _beh_acc[(_beh_acc["dist"] > 0.2) & (_beh_acc["dist"] < 40)].copy()
    if not _dist_both.empty:
        _fig_dist = px.histogram(
            _dist_both, x="dist", color="group",
            barmode="overlay",
            color_discrete_map={"Good": _GOOD_C, "Bad": _BAD_C},
            nbins=40,
            opacity=0.65,
            title="Accepted trip distance distribution — Good vs Bad",
            labels={"dist": "Distance (miles)", "group": ""},
            height=320,
        )
        _fig_dist.add_vline(x=2, line_dash="dash", line_color="#f59e0b", annotation_text="2 mi")
        _fig_dist.add_vline(x=5, line_dash="dash", line_color="#94a3b8", annotation_text="5 mi")
        _fig_dist.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(_fig_dist, use_container_width=True)

        _d_c1, _d_c2, _d_c3 = st.columns(3)
        _d_c1.metric("Good — avg distance", f"{_good_acc[(_good_acc['dist']>0.2)]['dist'].mean():.1f} mi")
        _d_c1.metric("Bad — avg distance",  f"{_bad_acc[(_bad_acc['dist']>0.2)]['dist'].mean():.1f} mi")
        _d_c2.metric("Good — sub-2mi trips", f"{(_good_acc['dist'] < 2).sum() / max(len(_good_acc),1) * 100:.1f}%")
        _d_c2.metric("Bad — sub-2mi trips",  f"{(_bad_acc['dist'] < 2).sum() / max(len(_bad_acc),1) * 100:.1f}%")
        _d_c3.metric("Good — 5mi+ trips",  f"{(_good_acc['dist'] >= 5).sum() / max(len(_good_acc),1) * 100:.1f}%")
        _d_c3.metric("Bad — 5mi+ trips",   f"{(_bad_acc['dist'] >= 5).sum() / max(len(_bad_acc),1) * 100:.1f}%")

    st.divider()

    # ── SECTION 9: Per-driver breakdown ───────────────────────────────────────
    st.subheader("9 — Per-driver breakdown")
    st.caption("Every driver in both groups, side by side. Sort by any column.")

    _drv_rows = []
    for _gid in TOP_DRIVER_IDS + BAD_DRIVER_IDS:
        _da = _beh_acc[_beh_acc["dim_driver_id"] == _gid]
        _dd = (_beh_dec[_beh_dec["dim_driver_id"] == _gid] if not _beh_dec.empty else pd.DataFrame())
        if _da.empty:
            continue
        _name = _da["driver_full_name"].dropna().iloc[0] if not _da["driver_full_name"].dropna().empty else str(_gid)
        _n_acc  = len(_da)
        _n_dec  = len(_dd)
        _acc_r  = round(_n_acc / max(_n_acc + _n_dec, 1) * 100, 1)
        _avg_f  = round(_da[_da["fare"] > 0]["fare"].mean(), 2) if (_da["fare"] > 0).any() else 0
        _sub10  = round((_da["fare"] < 10).sum() / max(_n_acc, 1) * 100, 1)
        _west_p = round(_da["is_west"].mean() * 100, 1)
        _avg_d  = round(_da[_da["dist"] > 0.2]["dist"].mean(), 1) if (_da["dist"] > 0.2).any() else 0
        _z1_pct = round((_da["pickup_zone"] == 1).sum() / max(_n_acc, 1) * 100, 1)
        _z3_pct = round((_da["pickup_zone"] == 3).sum() / max(_n_acc, 1) * 100, 1)
        _drv_rows.append({
            "Driver":         _name,
            "Group":          "Good" if _gid in TOP_DRIVER_IDS else "Bad",
            "Trips":          _n_acc,
            "Declines":       _n_dec,
            "Accept rate":    f"{_acc_r:.1f}%",
            "Avg fare":       f"£{_avg_f:.2f}",
            "Sub-£10 %":      f"{_sub10:.1f}%",
            "West %":         f"{_west_p:.1f}%",
            "Z1 pickup %":    f"{_z1_pct:.1f}%",
            "Z3 pickup %":    f"{_z3_pct:.1f}%",
            "Avg distance":   f"{_avg_d:.1f} mi",
        })

    if _drv_rows:
        _drv_tbl = pd.DataFrame(_drv_rows)
        st.dataframe(_drv_tbl, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ACCEPTANCE RATE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Acceptance Rate":
    st.title("Acceptance Rate")
    st.caption("April (53%) → May (43%). Bolt wants 50%. This page shows what actually happened, what the drop cost or gained, and what going back to 50% would mean.")

    with st.spinner("Loading April–May fleet data (full fleet, 2 months — may take a moment)..."):
        _ar_raw = _load_acceptance_period()

    if _ar_raw.empty:
        st.error("No data found for April–May 2026.")
        st.stop()

    # ── Pre-process ───────────────────────────────────────────────────────────
    _ar = _ar_raw.copy()
    _ar["event_dt"] = pd.to_datetime(_ar["event_dt"])
    _ar["dropoff_trip_datetime"] = pd.to_datetime(_ar["dropoff_trip_datetime"])
    _ar["fare"]   = pd.to_numeric(_ar["trip_price_in_pound"], errors="coerce").fillna(0)
    _ar["month"]  = _ar["event_dt"].dt.to_period("M").astype(str)
    _ar["date"]   = _ar["event_dt"].dt.date
    _ar["hour"]   = _ar["event_dt"].dt.hour
    _ar["is_acc"] = _ar["status"].isin(["completed", "Finished"])
    _ar["is_dec"] = _ar["status"].isin(["Driver did not respond", "Driver rejected"])

    _acc_only = _ar[_ar["is_acc"] & (_ar["fare"] > 0)].copy()

    # ── Per-driver-per-day RPH ────────────────────────────────────────────────
    # Use trip-level online time: trip duration + gap to next trip (capped at 90 min).
    # This avoids over-counting breaks where the driver's app was offline,
    # which is why raw (last_dropoff - first_pickup) gives RPH too low vs Bolt.
    _rph_base = _acc_only.sort_values(["dim_driver_id", "date", "event_dt"]).copy()
    _rph_base["dropoff_dt"] = pd.to_datetime(_rph_base["dropoff_trip_datetime"])
    _rph_base["trip_hrs"] = (
        (_rph_base["dropoff_dt"] - _rph_base["event_dt"])
        .dt.total_seconds() / 3600
    ).clip(0, 3)
    _rph_base["next_pickup_dt"] = _rph_base.groupby(["dim_driver_id", "date"])["event_dt"].shift(-1)
    _rph_base["gap_hrs"] = (
        (pd.to_datetime(_rph_base["next_pickup_dt"]) - _rph_base["dropoff_dt"])
        .dt.total_seconds() / 3600
    ).clip(0, 1.5).fillna(0)  # cap at 90 min; last trip of day gets 0
    _rph_base["online_seg_hrs"] = _rph_base["trip_hrs"] + _rph_base["gap_hrs"]

    _daily = (
        _rph_base.groupby(["dim_driver_id", "driver_full_name", "date", "month"])
        .agg(
            earnings       = ("fare",            "sum"),
            trips          = ("fare",            "count"),
            online_hrs     = ("online_seg_hrs",  "sum"),
        )
        .reset_index()
    )
    _daily["online_hrs"] = _daily["online_hrs"].clip(lower=0.5, upper=16)
    _daily["rph"] = (_daily["earnings"] / _daily["online_hrs"]).clip(upper=120)

    # ── Monthly fleet-level stats ─────────────────────────────────────────────
    def _month_stats(month_str):
        _m  = _ar[_ar["month"] == month_str]
        _ma = _m[_m["is_acc"] & (_m["fare"] > 0)]
        _n_acc   = int(_m["is_acc"].sum())
        _n_dec   = int(_m["is_dec"].sum())
        _total   = _n_acc + _n_dec
        # Fleet-level rate (for reference)
        _acc_r_fleet = round(_n_acc / max(_total, 1) * 100, 1)
        # Per-driver average (matches Bolt's calculation method).
        # Only include drivers with at least 5 pings — Bolt excludes inactive drivers.
        _per_drv = (
            _m.groupby("dim_driver_id")
            .apply(lambda g: pd.Series({
                "drv_acc_rate": g["is_acc"].sum() / max(len(g), 1) * 100,
                "total_pings":  len(g),
            }))
            .reset_index()
        )
        _per_drv = _per_drv[_per_drv["total_pings"] >= 5]
        _acc_r = round(_per_drv["drv_acc_rate"].mean(), 1) if not _per_drv.empty else _acc_r_fleet
        _avg_f   = round(_ma["fare"].mean(), 2) if not _ma.empty else 0
        _tot_rev = round(_ma["fare"].sum(), 0)
        _drivers = _ma["dim_driver_id"].nunique()
        _rph     = _daily[_daily["month"] == month_str]["rph"].mean()
        _rph     = round(_rph, 2) if not np.isnan(_rph) else 0
        return {
            "acc_rate": _acc_r, "acc_rate_fleet": _acc_r_fleet,
            "n_acc": _n_acc, "n_dec": _n_dec,
            "total": _total, "avg_fare": _avg_f,
            "total_rev": _tot_rev, "drivers": _drivers, "avg_rph": _rph,
        }

    _apr = _month_stats("2026-04")
    _may = _month_stats("2026-05")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: April vs May headline
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1 — April vs May: the argument in three numbers")

    _fare_delta = _may["avg_fare"] - _apr["avg_fare"]
    _rph_delta  = _may["avg_rph"]  - _apr["avg_rph"]
    _rate_delta = _may["acc_rate"] - _apr["acc_rate"]
    _drv_delta  = _may["drivers"]  - _apr["drivers"]
    _rev_delta  = _may["total_rev"]- _apr["total_rev"]

    # ── 3-metric comparison — the core argument ───────────────────────────────
    def _arrow_cell(val, positive_is_good=True):
        if val == 0:
            return "→", "#94a3b8"
        up = val > 0
        good = up if positive_is_good else not up
        arrow = "↑" if up else "↓"
        color = "#22c55e" if good else "#ef4444"
        return arrow, color

    _metrics = [
        ("Acceptance rate",  f"{_apr['acc_rate']:.1f}%",  f"{_may['acc_rate']:.1f}%",  _rate_delta, False, f"{_rate_delta:+.1f}pp"),
        ("Avg fare / trip",  f"£{_apr['avg_fare']:.2f}",  f"£{_may['avg_fare']:.2f}",  _fare_delta, True,  f"£{_fare_delta:+.2f}"),
        ("Avg driver RPH",   f"£{_apr['avg_rph']:.2f}/hr",f"£{_may['avg_rph']:.2f}/hr",_rph_delta,  True,  f"£{_rph_delta:+.2f}/hr"),
    ]

    _rows_html = ""
    for label, apr_val, may_val, delta, pig, delta_str in _metrics:
        arrow, color = _arrow_cell(delta, pig)
        _rows_html += (
            f'<tr>'
            f'<td style="padding:10px 12px;color:#94a3b8;font-size:13px;width:35%">{label}</td>'
            f'<td style="padding:10px 12px;color:#3b82f6;font-weight:bold;font-size:16px;text-align:center">{apr_val}</td>'
            f'<td style="padding:10px 12px;color:#f8fafc;font-weight:bold;font-size:16px;text-align:center">{may_val}</td>'
            f'<td style="padding:10px 12px;text-align:center">'
            f'<span style="color:{color};font-size:22px;font-weight:bold">{arrow}</span>'
            f'<span style="color:{color};font-size:12px;margin-left:4px">{delta_str}</span>'
            f'</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background:#1e1e2e;border-radius:10px;overflow:hidden;margin-bottom:16px;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>'
        f'<th style="padding:10px 12px;color:#64748b;font-size:11px;font-weight:normal;text-align:left;text-transform:uppercase;letter-spacing:1px">Metric</th>'
        f'<th style="padding:10px 12px;color:#3b82f6;font-size:11px;font-weight:bold;text-align:center;text-transform:uppercase;letter-spacing:1px">April</th>'
        f'<th style="padding:10px 12px;color:#f8fafc;font-size:11px;font-weight:bold;text-align:center;text-transform:uppercase;letter-spacing:1px">May</th>'
        f'<th style="padding:10px 12px;color:#64748b;font-size:11px;font-weight:normal;text-align:center;text-transform:uppercase;letter-spacing:1px">Change</th>'
        f'</tr></thead>'
        f'<tbody>{_rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    # ── Why total revenue went up despite lower acceptance ────────────────────
    _rev_arrow, _rev_color = _arrow_cell(_rev_delta, True)
    _drv_arrow, _drv_color = _arrow_cell(_drv_delta, True)
    st.markdown(
        f'<div style="background:#1e293b;border-left:3px solid #64748b;padding:10px 14px;border-radius:4px;margin-bottom:16px;">'
        f'<span style="color:#94a3b8;font-size:12px;">'
        f'<strong style="color:#f8fafc;">Why total revenue went up:</strong> '
        f'Active drivers went from <strong style="color:#3b82f6">{_apr["drivers"]}</strong> → '
        f'<strong style="color:#f8fafc">{_may["drivers"]}</strong> ({_drv_arrow} {abs(_drv_delta)} drivers). '
        f'Total revenue change of <strong style="color:{_rev_color}">£{abs(_rev_delta):,.0f}</strong> is a fleet-size story, not a per-driver story. '
        f'The per-driver numbers — avg fare and RPH — are what matter for the argument.'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── Full detail cards ─────────────────────────────────────────────────────
    with st.expander("Full month detail"):
        def _month_card(label, color, s):
            rows = {
                "Acceptance rate (per-driver avg)": f"{s['acc_rate']:.1f}%",
                "Acceptance rate (fleet total)":    f"{s['acc_rate_fleet']:.1f}%",
                "Trips accepted":    f"{s['n_acc']:,}",
                "Pings declined":    f"{s['n_dec']:,}",
                "Avg fare accepted": f"£{s['avg_fare']:.2f}",
                "Total revenue":     f"£{s['total_rev']:,.0f}",
                "Avg daily RPH":     f"£{s['avg_rph']:.2f}/hr",
                "Active drivers":    f"{s['drivers']:,}",
                "Bolt target":       "50%",
            }
            _html = "".join(
                f'<tr><td style="padding:5px 0;color:#94a3b8;font-size:12px;">{k}</td>'
                f'<td style="text-align:right;font-weight:bold;color:#f8fafc;font-size:13px;">{v}</td></tr>'
                for k, v in rows.items()
            )
            return (
                f'<div style="background:#1e1e2e;border:2px solid {color};border-radius:8px;padding:14px 16px;">'
                f'<div style="color:{color};font-size:11px;font-weight:bold;letter-spacing:1px;margin-bottom:8px;">{label}</div>'
                f'<table style="width:100%;border-collapse:collapse;">{_html}</table></div>'
            )
        _mc1, _mc2 = st.columns(2)
        _mc1.markdown(_month_card("APRIL 2026", "#3b82f6", _apr), unsafe_allow_html=True)
        _mc2.markdown(_month_card("MAY 2026",   "#22c55e" if _may["avg_rph"] >= _apr["avg_rph"] else "#f59e0b", _may), unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Acceptance rate vs RPH — the trade-off
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2 — The trade-off: acceptance rate vs RPH")
    st.caption("Each dot is one driver for one month. If lower acceptance = higher RPH, the dots slope downward left to right.")

    # Per-driver-per-month: acceptance rate + avg RPH
    _drv_month_acc = (
        _ar.groupby(["dim_driver_id","driver_full_name","month"])
        .apply(lambda g: pd.Series({
            "accepted": g["is_acc"].sum(),
            "declined": g["is_dec"].sum(),
        }))
        .reset_index()
    )
    _drv_month_acc["total"]    = _drv_month_acc["accepted"] + _drv_month_acc["declined"]
    _drv_month_acc["acc_rate"] = (_drv_month_acc["accepted"] / _drv_month_acc["total"] * 100).round(1)

    _drv_month_rph = (
        _daily.groupby(["dim_driver_id","month"])["rph"]
        .mean().round(2).reset_index()
    )
    _scatter_df = pd.merge(_drv_month_acc, _drv_month_rph, on=["dim_driver_id","month"], how="inner")
    _scatter_df = _scatter_df[
        (_scatter_df["total"] >= 20) &
        (_scatter_df["rph"] > 5) &
        (_scatter_df["rph"] < 100)
    ].copy()

    if not _scatter_df.empty:
        _fig_scatter = px.scatter(
            _scatter_df, x="acc_rate", y="rph",
            color="month",
            color_discrete_map={"2026-04": "#3b82f6", "2026-05": "#22c55e"},
            hover_name="driver_full_name",
            hover_data={"acc_rate": ":.1f", "rph": ":.2f", "total": True},
            title="Acceptance rate vs avg daily RPH — per driver per month",
            labels={"acc_rate": "Acceptance rate (%)", "rph": "Avg daily RPH (£/hr)", "month": "Month", "total": "Total pings"},
            height=420,
        )
        # Manual trendline via numpy (avoids statsmodels dependency)
        _x = _scatter_df["acc_rate"].values
        _y = _scatter_df["rph"].values
        _slope, _intercept = np.polyfit(_x, _y, 1)
        _x_line = np.linspace(_x.min(), _x.max(), 100)
        _y_line = _slope * _x_line + _intercept
        _fig_scatter.add_scatter(
            x=_x_line, y=_y_line,
            mode="lines",
            line=dict(color="#94a3b8", dash="dash", width=1.5),
            name="Trend",
            showlegend=True,
        )
        _fig_scatter.update_layout(legend=dict(orientation="h", y=1.05))
        st.plotly_chart(_fig_scatter, use_container_width=True)
        _slope_sign = "costs" if _slope < 0 else "gains"
        st.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {"#ef4444" if _slope < 0 else "#22c55e"};'
            f'border-radius:8px;padding:14px 18px;margin-top:8px;">'
            f'<span style="color:{"#ef4444" if _slope < 0 else "#22c55e"};font-size:22px;font-weight:bold;">'
            f'{"−" if _slope < 0 else "+"}£{abs(_slope):.2f}/hr</span>'
            f'<span style="color:#f8fafc;font-size:15px;"> per 1% increase in acceptance rate</span><br>'
            f'<span style="color:#94a3b8;font-size:13px;">'
            f'Every percentage point added to acceptance rate {_slope_sign} approximately £{abs(_slope):.2f}/hr in RPH. '
            f'Going from 43% → 50% (7pp) would {"cost" if _slope < 0 else "gain"} approximately '
            f'£{abs(_slope * 7):.2f}/hr.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Bucket analysis ───────────────────────────────────────────────────────
    st.subheader("3 — RPH by acceptance rate bucket")
    st.caption("Fleet grouped into acceptance rate bands. Clearly shows where RPH peaks — and whether 50% falls on the right or wrong side of it.")

    if not _scatter_df.empty:
        _scatter_df["acc_bucket"] = pd.cut(
            _scatter_df["acc_rate"],
            bins=[0, 30, 40, 50, 60, 70, 100],
            labels=["<30%","30-40%","40-50%","50-60%","60-70%","70%+"],
        )
        _bucket_rph = (
            _scatter_df.groupby("acc_bucket", observed=True)["rph"]
            .agg(avg_rph="mean", drivers="count")
            .reset_index()
        )
        _bucket_rph = _bucket_rph[_bucket_rph["drivers"] >= 3]

        _fig_bucket = px.bar(
            _bucket_rph, x="acc_bucket", y="avg_rph",
            text="avg_rph",
            color="avg_rph",
            color_continuous_scale=["#ef4444","#f59e0b","#22c55e","#f59e0b","#ef4444"],
            title="Avg daily RPH by acceptance rate band",
            labels={"acc_bucket": "Acceptance rate band", "avg_rph": "Avg RPH (£/hr)", "drivers": "Driver-months"},
            height=320,
            hover_data={"drivers": True},
        )
        _fig_bucket.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
        _fig_bucket.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(_fig_bucket, use_container_width=True)
        st.caption("If the peak is in the 40-50% band, that's the argument: Bolt's 50% target is already past the RPH optimum.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Fare distribution — what changed
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("4 — What changed: fare distribution April vs May")
    st.caption("If May's accepted fare distribution shifted right, drivers were cutting the low-fare tail — exactly the right behaviour.")

    _apr_fares = _acc_only[_acc_only["month"] == "2026-04"]["fare"]
    _may_fares = _acc_only[_acc_only["month"] == "2026-05"]["fare"]

    _fare_plot = pd.concat([
        _apr_fares.rename("fare").to_frame().assign(month="April"),
        _may_fares.rename("fare").to_frame().assign(month="May"),
    ])
    _fig_fd = px.histogram(
        _fare_plot, x="fare", color="month",
        barmode="overlay",
        color_discrete_map={"April":"#3b82f6","May":"#22c55e"},
        nbins=50, opacity=0.65,
        title="Accepted fare distribution — April vs May",
        labels={"fare":"Fare (£)","month":""},
        height=340,
    )
    _fig_fd.add_vline(x=10, line_dash="dash", line_color="#f59e0b", annotation_text="£10")
    _fig_fd.add_vline(x=15, line_dash="dash", line_color="#94a3b8", annotation_text="£15")
    _fig_fd.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(_fig_fd, use_container_width=True)

    # Sub-£10/£12 rates
    _sub10_apr = round((_apr_fares < 10).sum() / max(len(_apr_fares),1) * 100, 1)
    _sub10_may = round((_may_fares < 10).sum() / max(len(_may_fares),1) * 100, 1)
    _sub12_apr = round((_apr_fares < 12).sum() / max(len(_apr_fares),1) * 100, 1)
    _sub12_may = round((_may_fares < 12).sum() / max(len(_may_fares),1) * 100, 1)
    _fc1, _fc2, _fc3, _fc4 = st.columns(4)
    _fc1.metric("April sub-£10 rate", f"{_sub10_apr:.1f}%")
    _fc2.metric("May sub-£10 rate",   f"{_sub10_may:.1f}%", delta=f"{_sub10_may-_sub10_apr:.1f}pp", delta_color="inverse")
    _fc3.metric("April sub-£12 rate", f"{_sub12_apr:.1f}%")
    _fc4.metric("May sub-£12 rate",   f"{_sub12_may:.1f}%", delta=f"{_sub12_may-_sub12_apr:.1f}pp", delta_color="inverse")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: The marginal trips — cost of going to 50%
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("5 — The marginal trips: what 50% acceptance actually means")
    st.caption("To go from 43% → 50%, drivers need to accept ~X more trips per shift. These are the trips that would fill that gap.")

    # Estimate: currently declining at 57%. To hit 50%, acceptance goes up 7pp.
    # The marginal trips are the lowest-fare trips currently being accepted (or the highest-fare being declined).
    # Proxy: bottom (7/43)% of currently accepted fares ≈ the trips that would be re-added.
    # More precisely: in May, for every 43 accepted there are 57 declined.
    # To get to 50/50, you need 7 more accepted per 50 total → add 7 to 43, decline 50 instead of 57.
    # The question is: what do those 7 extra trips look like?
    # Best proxy: the lowest-fare currently accepted trips (they set the floor for what you'd readmit).

    _may_acc_fares = _acc_only[_acc_only["month"] == "2026-05"]["fare"].sort_values().reset_index(drop=True)
    _current_rate  = _may["acc_rate"] / 100
    _target_rate   = 0.50
    # How many extra trips needed per driver-day?
    _may_daily     = _daily[_daily["month"] == "2026-05"]
    _avg_trips_day = _may_daily["trips"].mean()
    _avg_dec_day   = _may_daily.apply(
        lambda r: (_ar[(_ar["dim_driver_id"] == r["dim_driver_id"]) &
                       (_ar["date"] == r["date"]) &
                       _ar["is_dec"]].shape[0]),
        axis=1,
    ).mean() if not _may_daily.empty else 0
    _total_pings_day = _avg_trips_day + _avg_dec_day
    _extra_needed   = _total_pings_day * (_target_rate - _current_rate)

    # Marginal fare = percentile of accepted trips equivalent to the extra needed
    _marginal_pct   = (_extra_needed / max(len(_may_acc_fares), 1)) * 100
    _marginal_floor = float(_may_acc_fares.quantile(max(0, (_marginal_pct)/100))) if not _may_acc_fares.empty else 0
    _marginal_fares = _may_acc_fares[_may_acc_fares <= _marginal_floor + 2]
    _marginal_avg   = round(_marginal_fares.mean(), 2) if not _marginal_fares.empty else 0
    _current_avg    = round(_may_acc_fares.mean(), 2)

    st.markdown(
        f'<div style="background:#1e1e2e;border:2px solid #f59e0b;border-radius:8px;padding:16px 20px;margin-bottom:16px;">'
        f'<div style="color:#f59e0b;font-size:13px;font-weight:bold;margin-bottom:6px;">WHAT 50% ACCEPTANCE COSTS</div>'
        f'<div style="color:#f8fafc;font-size:14px;line-height:1.7;">'
        f'Current May acceptance: <strong>{_may["acc_rate"]:.1f}%</strong> &nbsp;→&nbsp; Target: <strong>50%</strong><br>'
        f'Extra trips needed per driver per day: <strong>~{_extra_needed:.1f}</strong><br>'
        f'Current avg accepted fare (May): <strong>£{_current_avg:.2f}</strong><br>'
        f'Avg fare of the marginal trips being re-added: <strong>£{_marginal_avg:.2f}</strong><br>'
        f'Blended avg fare at 50%: <strong>£{round((_current_avg * _may["n_acc"] + _marginal_avg * (_may["n_acc"] * (_target_rate/_current_rate - 1))) / (_may["n_acc"] * (_target_rate/_current_rate)), 2):.2f}</strong>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Histogram of marginal fares vs current avg
    if not _marginal_fares.empty:
        _marg_fig = px.histogram(
            pd.concat([
                _marginal_fares.rename("fare").to_frame().assign(type="Marginal (to reach 50%)"),
                _may_acc_fares.rename("fare").to_frame().assign(type="Current May accepted"),
            ]),
            x="fare", color="type",
            barmode="overlay",
            color_discrete_map={"Marginal (to reach 50%)": "#ef4444", "Current May accepted": "#22c55e"},
            opacity=0.7, nbins=30,
            title="Marginal trips vs current accepted trips — fare comparison",
            labels={"fare": "Fare (£)", "type": ""},
            height=300,
        )
        _marg_fig.add_vline(x=_current_avg, line_dash="dash", line_color="#22c55e",
                            annotation_text=f"May avg £{_current_avg:.2f}")
        _marg_fig.add_vline(x=_marginal_avg, line_dash="dash", line_color="#ef4444",
                            annotation_text=f"Marginal avg £{_marginal_avg:.2f}")
        _marg_fig.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(_marg_fig, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Strategies to reach 50% without hurting RPH
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("6 — Reaching 50%: what every analysis is telling us")
    st.caption(
        "Five pages of analysis, one question: how do we go from 43% to 50% without destroying RPH? "
        "The answer isn't 'accept more randomly' — it's fixing structural positioning and accepting "
        "the right pings. Each strategy below is grounded in findings from Zone Analysis, Gap Analysis, "
        "Driver Behavior, and East/West dynamics."
    )

    # ── Pre-compute zones + east/west on accepted trips ──────────────────────
    _acc_pu = _acc_only["pickup_lat_long"].apply(lambda x: parse_dms(str(x or "")))
    _acc_only_z = _acc_only.copy()
    _acc_only_z["plat"] = [c[0] for c in _acc_pu]
    _acc_only_z["plon"] = [c[1] for c in _acc_pu]
    _acc_only_z = _acc_only_z.dropna(subset=["plat","plon"])
    _acc_only_z["pickup_zone"] = [assign_zone(lat, lon) for lat, lon in zip(_acc_only_z["plat"], _acc_only_z["plon"])]
    _acc_only_z["is_west"] = _acc_only_z["plon"] < _WEST_LON

    _z12_pct  = round((_acc_only_z["pickup_zone"].isin([1,2])).mean() * 100, 1) if not _acc_only_z.empty else 0
    _z12_avg  = round(_acc_only_z[_acc_only_z["pickup_zone"].isin([1,2])]["fare"].mean(), 2) if not _acc_only_z.empty else 0
    _z3p_avg  = round(_acc_only_z[_acc_only_z["pickup_zone"] >= 3]["fare"].mean(), 2) if not _acc_only_z.empty else 0
    _west_pct = round(_acc_only_z["is_west"].mean() * 100, 1) if not _acc_only_z.empty else 0
    _west_avg = round(_acc_only_z[_acc_only_z["is_west"]]["fare"].mean(), 2) if not _acc_only_z.empty else 0
    _east_avg = round(_acc_only_z[~_acc_only_z["is_west"]]["fare"].mean(), 2) if not _acc_only_z.empty else 0

    _peak_hours = [7, 8, 17, 18, 19]
    _acc_peak = _acc_only_z[_acc_only_z["hour"].isin(_peak_hours)]
    _acc_offp = _acc_only_z[~_acc_only_z["hour"].isin(_peak_hours)]
    _peak_sub12 = round((_acc_peak["fare"] < 12).mean() * 100, 1) if not _acc_peak.empty else 0
    _offp_sub12 = round((_acc_offp["fare"] < 12).mean() * 100, 1) if not _acc_offp.empty else 0
    _peak_avg   = round(_acc_peak["fare"].mean(), 2) if not _acc_peak.empty else 0
    _offp_avg   = round(_acc_offp["fare"].mean(), 2) if not _acc_offp.empty else 0

    _lhr_acc = _acc_only_z[
        (_acc_only_z["plat"].between(51.45, 51.49)) &
        (_acc_only_z["plon"].between(-0.50, -0.42))
    ]
    _lhr_trip_pct = round(len(_lhr_acc) / max(len(_acc_only_z), 1) * 100, 1)
    _lhr_avg_fare = round(_lhr_acc["fare"].mean(), 2) if not _lhr_acc.empty else 0

    # ── Strategy summary banner ───────────────────────────────────────────────
    st.markdown(
        '<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
        'padding:14px 20px;margin-bottom:20px;font-size:13px;color:#94a3b8;line-height:1.8;">'
        '<strong style="color:#f8fafc;">The 43% → 50% gap is 7pp.</strong> '
        'Based on all analysis done: '
        '<span style="color:#22c55e;">Strategy A</span> (Z1/Z2 positioning) +1–2pp · '
        '<span style="color:#22c55e;">Strategy B</span> (peak micro-trips) +1–2pp · '
        '<span style="color:#22c55e;">Strategy C</span> (distance floor) +2–3pp · '
        '<span style="color:#22c55e;">Strategy D</span> (Z3 daytime pass) +1pp · '
        '<span style="color:#22c55e;">Strategy E</span> (West shift) +1–2pp. '
        'Combined: <strong style="color:#f59e0b;">~6–10pp recoverable.</strong> '
        'Plus: <span style="color:#ef4444;">Airport queue drag</span> is currently suppressing '
        'the rate — fixing dead-queue time gives back up to 1–2pp for free.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Strategies A + B ─────────────────────────────────────────────────────
    _sc1, _sc2 = st.columns(2)

    with _sc1:
        st.markdown("##### Strategy A — Stay in Z1/Z2 and accept the sequence")
        st.markdown(
            '<div style="background:#1e293b;border-radius:6px;padding:12px 14px;font-size:13px;color:#e2e8f0;line-height:1.7;">'
            '<strong>Zone Analysis finding:</strong> After a Z1/Z2 dropoff, 81% of drivers receive '
            'their next ping within the same zone group. After a Z3 dropoff, 24% of drivers drift '
            'further out and face a dead window. The Zone Analysis shows Z3 is a trap: a cheap Z3→Z3 '
            'local hop chain strands drivers in low-value corridors for the rest of the shift.<br><br>'
            '<strong>The acceptance rate link:</strong> Drivers accepting Z1/Z2 pings — even the '
            '£10–12 ones — stay positioned in high-frequency corridors where the <em>next</em> ping '
            'arrives fast and pays well. The RPH of the sequence is fine. Refusing Z1/Z2 pings to '
            'wait for a big fare creates idle time and doesn\'t improve RPH. '
            '<strong>Estimate: +1–2pp.</strong>'
            '</div>',
            unsafe_allow_html=True,
        )
        _sa, _sb = st.columns(2)
        _sa.metric("Z1/Z2 pickup share",  f"{_z12_pct:.1f}%")
        _sa.metric("Z3+ pickup share",    f"{100 - _z12_pct:.1f}%")
        _sb.metric("Z1/Z2 avg fare",      f"£{_z12_avg:.2f}")
        _sb.metric("Z3+ avg fare",        f"£{_z3p_avg:.2f}")

    with _sc2:
        st.markdown("##### Strategy B — Relax the fare floor during peak windows only")
        st.markdown(
            '<div style="background:#1e293b;border-radius:6px;padding:12px 14px;font-size:13px;color:#e2e8f0;line-height:1.7;">'
            '<strong>Gap Analysis + Driver Behavior finding:</strong> During 07:00–09:00 and '
            '17:00–20:00, inter-trip gaps are significantly shorter because demand density is high. '
            'A £9 trip at 08:30 completes in ~8 minutes and the next ping arrives in &lt;5 minutes — '
            'the RPH hit is negligible. The same £9 trip at 14:00 is followed by a 25-minute gap, '
            'making it a genuine RPH killer.<br><br>'
            '<strong>The acceptance rate link:</strong> Driver Behavior analysis shows top drivers '
            'have lower fare floors during peak hours. If all drivers relaxed to a £9 floor '
            '<em>only during peak</em> while keeping higher filters off-peak, the marginal trips '
            'added are RPH-neutral. <strong>Estimate: +1–2pp.</strong>'
            '</div>',
            unsafe_allow_html=True,
        )
        _pa, _pb = st.columns(2)
        _pa.metric("Peak sub-£12 share",     f"{_peak_sub12:.1f}%")
        _pa.metric("Off-peak sub-£12 share", f"{_offp_sub12:.1f}%")
        _pb.metric("Peak avg fare",          f"£{_peak_avg:.2f}")
        _pb.metric("Off-peak avg fare",      f"£{_offp_avg:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strategies C + D ─────────────────────────────────────────────────────
    _sc3, _sc4 = st.columns(2)

    with _sc3:
        st.markdown("##### Strategy C — Switch from fare floor to distance floor")
        st.markdown(
            '<div style="background:#1e293b;border-radius:6px;padding:12px 14px;font-size:13px;color:#e2e8f0;line-height:1.7;">'
            '<strong>Driver Behavior finding:</strong> The RPH killers are not low-fare trips per '
            'se — they are sub-2-mile local hops that strand drivers in a wrong postcode with a '
            '20+ minute repositioning gap. A £10 trip covering 4 miles takes ~12 minutes and '
            'leaves you near a viable corridor. The same £10 over 0.8 miles takes 6 minutes but '
            'the recovery cost is 3× higher.<br><br>'
            '<strong>The acceptance rate link:</strong> Drivers currently use a fare floor '
            '(decline anything &lt;£12) which incorrectly rejects viable medium-distance trips '
            'and creates the acceptance rate shortfall. Switching to a <em>distance floor</em> '
            '(decline any trip under ~2 miles, regardless of fare) would allow acceptance of more '
            'pings while still protecting RPH. <strong>Estimate: +2–3pp.</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

    with _sc4:
        st.markdown("##### Strategy D — Use Z3 daytime avoidance to stay in the acceptance corridor")
        st.markdown(
            '<div style="background:#1e293b;border-radius:6px;padding:12px 14px;font-size:13px;color:#e2e8f0;line-height:1.7;">'
            '<strong>Zone Analysis finding (counterintuitive):</strong> The Z3 drift analysis shows '
            'that 24% of drivers who drop off in Z2 drift into Z3 on their next trip, and those '
            'drifts are mostly avoidable — daytime Z3 pings with no clear return path. Declining '
            'those specific pings keeps the driver in Z1/Z2, where they will receive and <em>accept</em> '
            'more pings overall.<br><br>'
            '<strong>The acceptance rate link:</strong> This is a selective decline that <em>raises</em> '
            'the overall acceptance rate. Fewer accepted Z3 pings → more time in Z1/Z2 → more pings '
            'received in high-accept zones → net acceptance rate goes up even though you declined one. '
            'Combined with Strategy A, this creates a compounding effect. '
            '<strong>Estimate: +1pp.</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strategy E ───────────────────────────────────────────────────────────
    st.markdown("##### Strategy E — Shift the fleet's centre of gravity westward")
    st.markdown(
        '<div style="background:#1e293b;border-radius:6px;padding:12px 14px;font-size:13px;color:#e2e8f0;line-height:1.7;">'
        '<strong>East/West analysis finding:</strong> West London (west of Charing Cross, −0.12°) '
        'consistently produces better signal-to-noise: fewer sub-£10 pings, higher average fares, '
        'and lower inter-trip gaps. The behavioral analysis shows bad drivers work disproportionately '
        'in outer East London — same streets as top drivers like Yousuf, but they accept the noise '
        '(51% sub-£10 at City of London). Yousuf on the same streets has 0% sub-£10 and earns '
        '£25+ avg fare because he is selective AND well-positioned in inner east.<br><br>'
        '<strong>The acceptance rate link:</strong> The real problem for bad-area drivers is not '
        'their acceptance decisions — it\'s that the pings they receive in outer east are '
        'genuinely not worth accepting. Shifting 20–30% of fleet dwell time westward means the '
        'marginal ping pool improves. Drivers can accept a higher share naturally without '
        'a single policy change. Driver Behavior data shows top drivers average '
        f'{_west_pct:.0f}% west pickups fleet-wide. West trips avg £{_west_avg:.2f} vs '
        f'east £{_east_avg:.2f}. <strong>Estimate: +1–2pp.</strong>'
        '</div>',
        unsafe_allow_html=True,
    )
    _se1, _se2, _se3 = st.columns(3)
    _se1.metric("Fleet west pickup share", f"{_west_pct:.1f}%")
    _se2.metric("West avg fare",           f"£{_west_avg:.2f}")
    _se3.metric("East avg fare",           f"£{_east_avg:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Airport risk flag ────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#1e1e2e;border:2px solid #f59e0b;border-radius:8px;'
        'padding:16px 20px;margin-bottom:4px;">'
        '<div style="color:#f59e0b;font-size:13px;font-weight:bold;letter-spacing:0.8px;margin-bottom:10px;">'
        'RISK FLAG — Airport queue dead time is suppressing the current 43%'
        '</div>'
        '<div style="color:#f8fafc;font-size:14px;line-height:1.8;">'
        '<strong>What is happening:</strong> Drivers sent to Heathrow join the taxi queue. '
        'While queuing, Bolt continues to send pings — the driver cannot accept them. '
        'Every missed ping during queue time is counted as a decline by Bolt\'s system. '
        'A driver who completes a Heathrow pickup after 40 minutes of queuing may have '
        'silently missed 5–10 pings in that window, dragging their personal acceptance rate down '
        'with zero behavioural cause.<br><br>'
        '<strong>We cannot and should not stop airport trips</strong> — they are high-value '
        f'(fleet avg £{_lhr_avg_fare:.2f} per Heathrow trip, {_lhr_trip_pct:.1f}% of all trips). '
        'But we should flag this to Bolt: queue dead time must be excluded from acceptance rate '
        'calculation, or Bolt\'s 50% target needs to account for it. This is likely responsible '
        'for 1–2pp of the current 43% figure and is not recoverable through driver behaviour alone.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    _rf1, _rf2 = st.columns(2)
    _rf1.metric("LHR trips (Apr+May)",  f"{len(_lhr_acc):,}")
    _rf2.metric("LHR avg fare",         f"£{_lhr_avg_fare:.2f}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: The counter-argument — if 50% genuinely hurts RPH
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("7 — The counter-argument: if 50% genuinely cannot be reached safely")
    st.caption("Use this section to present the evidence to Bolt if the strategies above fall short.")

    st.markdown(
        '<div style="background:#1e1e2e;border:2px solid #ef4444;border-radius:8px;padding:20px 24px;margin-bottom:20px;">'
        '<div style="color:#ef4444;font-size:13px;font-weight:bold;letter-spacing:1px;margin-bottom:12px;">THE ARGUMENT</div>'
        '<div style="color:#f8fafc;font-size:14px;line-height:1.9;">'
        '1. <strong>The April 53% rate was not strategic discipline — it was a worse outcome.</strong> '
        'Drivers in April accepted more pings but earned less per hour. The trips being accepted were '
        'disproportionately low-fare, short-distance local hops that stranded drivers in bad zones.<br>'
        '2. <strong>May\'s lower acceptance rate was the fleet self-correcting.</strong> '
        'Drivers learned to decline the sub-£12, short-distance, daytime Zone 3 pings that were '
        'destroying their shifts. The earnings-per-hour data reflects this: RPH moved in the right direction.<br>'
        '3. <strong>Forcing 50% means forcing drivers to re-accept those trips.</strong> '
        'The marginal pings between 43% and 50% are not evenly distributed — they are concentrated '
        'in exactly the fare/zone/time combinations that the data shows are RPH-negative.<br>'
        '4. <strong>The alternative proposal:</strong> Bolt should measure <em>revenue per hour</em> '
        'rather than acceptance rate as the primary KPI. A fleet at 43% acceptance and £X/hr RPH '
        'is objectively better for drivers — and for platform health — than a fleet at 50% acceptance '
        'and £(X-Y)/hr RPH. Acceptance rate is a proxy metric that breaks when trip quality is variable.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Supporting evidence: hour-by-hour RPH April vs May
    st.markdown("**Supporting evidence — RPH by hour of day, April vs May:**")
    _hourly_rph = (
        _acc_only.groupby(["month","hour"])
        .apply(lambda g: pd.Series({
            "avg_fare": g["fare"].mean(),
            "trips": len(g),
        }))
        .reset_index()
    )
    _fig_hrph = px.line(
        _hourly_rph, x="hour", y="avg_fare", color="month",
        color_discrete_map={"2026-04":"#3b82f6","2026-05":"#22c55e"},
        markers=True,
        title="Avg accepted fare by hour — April vs May",
        labels={"hour":"Hour of day","avg_fare":"Avg fare (£)","month":"Month"},
        height=320,
    )
    _fig_hrph.add_hline(y=15, line_dash="dash", line_color="#94a3b8", annotation_text="£15 baseline")
    _fig_hrph.update_xaxes(dtick=1)
    _fig_hrph.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(_fig_hrph, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BYOC — BRING YOUR OWN CAR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "BYOC":
    st.title("BYOC — Bring Your Own Car")
    st.caption(
        "10 drivers who supply their own vehicle. Core question: are they generating enough per hour? "
        "**RPH = SUM(revenue) ÷ SUM(online_hrs)** — platform-recorded online time, all-time history. "
        "Threshold: **≥ £21/hr = profitable**. "
        "Hours worked is shown as a confidence weight — a £28 RPH from 2 hours means very little."
    )

    # ── 1. Resolve driver names to IDs ───────────────────────────────────────
    with st.spinner("Resolving BYOC driver names against DB..."):
        _byoc_resolved: dict = {}   # display_name -> (driver_id, db_name)
        _byoc_unresolved: list = []
        for _display, _key in _BYOC_SEARCH_KEYS:
            _hits = _search_drivers(_key)
            if _hits.empty:
                _byoc_unresolved.append(_display)
                continue
            if len(_hits) == 1:
                _r = _hits.iloc[0]
                _byoc_resolved[_display] = (int(_r["dim_driver_id"]), str(_r["driver_full_name"]))
            else:
                _dn_words = set(_display.lower().split())
                _best, _best_score = None, -1
                for _, _r in _hits.iterrows():
                    _s = len(_dn_words & set(str(_r["driver_full_name"]).lower().split()))
                    if _s > _best_score:
                        _best_score, _best = _s, _r
                if _best is not None:
                    _byoc_resolved[_display] = (int(_best["dim_driver_id"]), str(_best["driver_full_name"]))
                else:
                    _byoc_unresolved.append(_display)

    _res_rows = (
        [{"Name": n, "DB Name": dbname, "ID": did, "Status": "✅ Found"}
         for n, (did, dbname) in _byoc_resolved.items()] +
        [{"Name": n, "DB Name": "—", "ID": "—", "Status": "❌ Not found"}
         for n in _byoc_unresolved]
    )
    with st.expander("Name resolution results", expanded=bool(_byoc_unresolved)):
        st.dataframe(pd.DataFrame(_res_rows), use_container_width=True, hide_index=True)

    if not _byoc_resolved:
        st.error("No BYOC drivers found in the database.")
        st.stop()

    _byoc_ids      = [did for _, (did, _) in _byoc_resolved.items()]
    _id_to_display = {did: name for name, (did, _) in _byoc_resolved.items()}

    # ── 2. Load data ──────────────────────────────────────────────────────────
    with st.spinner("Loading BYOC performance data..."):
        _bp  = db.load_comparison_performance(_byoc_ids)
        _bd  = _byoc_daily_perf(_byoc_ids)
        _bt  = db.load_comparison_trips(_byoc_ids)
        _bfl = db.load_fleet_baseline_excluding(_byoc_ids)

    if _bp.empty:
        st.warning("No performance records found for any BYOC driver. They may have zero completed trips.")
        st.stop()

    _bp["display_name"] = _bp["dim_driver_id"].map(_id_to_display).fillna(_bp["driver_name"])
    _bfl_rph = (
        float(_bfl.iloc[0]["fleet_rph"])
        if not _bfl.empty and _bfl.iloc[0]["fleet_rph"] is not None
        else _FLEET_RPH_DEFAULT
    )

    _active_days_map = _bd.groupby("dim_driver_id").size().to_dict() if not _bd.empty else {}
    if not _bd.empty:
        _bd["driver_performance_date"] = pd.to_datetime(_bd["driver_performance_date"])

    # Enrich trip data with coords + zones
    if not _bt.empty:
        _bt = _bt.copy()
        _bt["pickedup_trip_datetime"] = pd.to_datetime(_bt["pickedup_trip_datetime"])
        _bt["hour"] = _bt["pickedup_trip_datetime"].dt.hour
        _bt["fare"] = pd.to_numeric(_bt["trip_price_in_pound"], errors="coerce").fillna(0)
        _bt["dist"] = pd.to_numeric(_bt["distance_in_miles"],   errors="coerce").fillna(0)
        _bcoords = _bt["pickup_lat_long"].apply(lambda x: parse_dms(str(x or "")))
        _bt["plat"] = [c[0] for c in _bcoords]
        _bt["plon"] = [c[1] for c in _bcoords]
        _bt_valid = _bt.dropna(subset=["plat", "plon"]).copy()
        _bt_valid["pickup_zone"] = [assign_zone(lat, lon) for lat, lon in zip(_bt_valid["plat"], _bt_valid["plon"])]
        _bt_valid["is_west"] = _bt_valid["plon"] < _WEST_LON
    else:
        _bt_valid = pd.DataFrame()

    # ── 3. Cohort headline metrics ────────────────────────────────────────────
    st.subheader("Cohort Summary")
    _above_n   = int((_bp["rph"] >= _BYOC_RPH_THRESHOLD).sum())
    _total_n   = len(_bp)
    _w_rph     = float(_bp["total_revenue"].sum() / _bp["online_hrs"].sum()) if _bp["online_hrs"].sum() > 0 else 0.0
    _tot_hrs   = float(_bp["online_hrs"].sum())
    _tot_rev   = float(_bp["total_revenue"].sum())

    _cm1, _cm2, _cm3, _cm4, _cm5 = st.columns(5)
    _cm1.metric("Above £21/hr threshold", f"{_above_n} / {_total_n}")
    _cm2.metric("Weighted cohort RPH",     f"£{_w_rph:.2f}",
                delta=f"{_w_rph - _bfl_rph:+.2f} vs fleet", delta_color="normal")
    _cm3.metric("Total online hours",      f"{_tot_hrs:.0f} hrs")
    _cm4.metric("Total revenue",           f"£{_tot_rev:,.0f}")
    _cm5.metric("Fleet avg RPH",           f"£{_bfl_rph:.2f}")

    # ── 4. Summary table ──────────────────────────────────────────────────────
    st.subheader("Summary Table")
    st.caption(
        "**RPH = SUM(revenue) ÷ SUM(online_hrs)** — all-time. "
        "**Confidence:** 🔴 < 10 hrs (ignore), 🟡 10–50 hrs (indicative), 🟢 > 50 hrs (reliable). "
        "Sorted by RPH descending."
    )

    def _conf(h: float) -> str:
        return "🔴 Low" if h < 10 else ("🟡 Mid" if h < 50 else "🟢 High")

    def _verdict(r: float) -> str:
        return "✅ Profitable" if r >= _BYOC_RPH_THRESHOLD else (
            "⚠️ Borderline" if r >= _BYOC_RPH_THRESHOLD * 0.85 else "❌ Loss"
        )

    _name_to_adays = {name: _active_days_map.get(did, 0) for name, (did, _) in _byoc_resolved.items()}
    _tbl = _bp[["display_name", "online_hrs", "total_rides", "total_revenue", "rph",
                "avg_fare", "utilisation", "acceptance"]].copy()
    _tbl.columns = ["Driver", "Online Hrs", "Trips", "Revenue (£)", "RPH (£)", "Avg Fare (£)", "Util %", "Accept %"]
    _tbl["Active Days"] = _tbl["Driver"].map(_name_to_adays).fillna(0).astype(int)
    _tbl["Confidence"]  = _tbl["Online Hrs"].apply(_conf)
    _tbl["Verdict"]     = _tbl["RPH (£)"].apply(_verdict)
    _tbl = _tbl.sort_values("RPH (£)", ascending=False).reset_index(drop=True)

    st.dataframe(
        _tbl.style
            .background_gradient(subset=["RPH (£)"],    cmap="RdYlGn", vmin=15, vmax=28)
            .background_gradient(subset=["Online Hrs"], cmap="Blues",   vmin=0,  vmax=max(float(_tbl["Online Hrs"].max()), 1.0))
            .format({"Revenue (£)": "£{:.2f}", "RPH (£)": "£{:.2f}", "Avg Fare (£)": "£{:.2f}",
                     "Util %": "{:.1f}%", "Accept %": "{:.1f}%", "Online Hrs": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # ── 5. RPH bar chart ──────────────────────────────────────────────────────
    st.subheader("RPH vs £21 Threshold")
    _bar = _bp[["display_name", "rph", "online_hrs"]].copy().sort_values("rph", ascending=True)
    _bar_colors = _bar["rph"].apply(
        lambda r: "#22c55e" if r >= _BYOC_RPH_THRESHOLD else ("#f59e0b" if r >= _BYOC_RPH_THRESHOLD * 0.85 else "#ef4444")
    ).tolist()
    _fig_rph_bar = go.Figure(go.Bar(
        x=_bar["rph"].tolist(),
        y=_bar["display_name"].tolist(),
        orientation="h",
        marker_color=_bar_colors,
        text=[f"£{r:.2f}  ({h:.0f} hrs)" for r, h in zip(_bar["rph"], _bar["online_hrs"])],
        textposition="outside",
        hovertemplate="%{y}: £%{x:.2f}/hr<extra></extra>",
    ))
    _fig_rph_bar.add_vline(x=_BYOC_RPH_THRESHOLD, line_dash="dash", line_color="#f59e0b",
                           annotation_text="£21 threshold",          annotation_position="top right")
    _fig_rph_bar.add_vline(x=_bfl_rph,            line_dash="dot",  line_color="#94a3b8",
                           annotation_text=f"Fleet £{_bfl_rph:.2f}", annotation_position="bottom right")
    _fig_rph_bar.update_layout(
        height=max(320, len(_bar) * 50 + 80),
        xaxis_title="Revenue per Online Hour (£)",
        yaxis_title=None,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e2e8f0"),
    )
    st.plotly_chart(_fig_rph_bar, use_container_width=True)

    # ── 6. Hours confidence chart ─────────────────────────────────────────────
    st.subheader("Data Confidence — Hours Worked")
    st.caption("RPH from 3 hours is noise. RPH from 150 hours is signal. Colour = RPH: red < £21, green ≥ £21.")
    _hconf = _bp[["display_name", "online_hrs", "rph"]].copy().sort_values("online_hrs", ascending=False)
    _fig_hconf = px.bar(
        _hconf, x="display_name", y="online_hrs",
        color="rph",
        color_continuous_scale=[[0, "#ef4444"], [0.4, "#f59e0b"], [1, "#22c55e"]],
        range_color=[14, 28],
        labels={"display_name": "Driver", "online_hrs": "Online Hours", "rph": "RPH (£)"},
        title="Total online hours per driver  (colour = RPH)",
        height=340,
    )
    _fig_hconf.add_hline(y=10, line_dash="dash", line_color="#94a3b8", annotation_text="10 hrs — min confidence")
    _fig_hconf.add_hline(y=50, line_dash="dash", line_color="#60a5fa", annotation_text="50 hrs — high confidence")
    _fig_hconf.update_layout(plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(_fig_hconf, use_container_width=True)

    # ── 7. Per-driver deep dives ──────────────────────────────────────────────
    st.subheader("Per-Driver Breakdown")

    for _, _prow in _bp.sort_values("rph", ascending=False).iterrows():
        _did     = int(_prow["dim_driver_id"])
        _dname   = str(_prow["display_name"])
        _rph_d   = float(_prow["rph"]          or 0)
        _hrs_d   = float(_prow["online_hrs"]   or 0)
        _rev_d   = float(_prow["total_revenue"] or 0)
        _trips_d = int(_prow["total_rides"]     or 0)
        _util_d  = float(_prow["utilisation"]  or 0)
        _acc_d   = float(_prow["acceptance"]   or 0)
        _afar_d  = float(_prow["avg_fare"]     or 0)
        _adays_d = _active_days_map.get(_did, 0)
        _v_icon  = "✅" if _rph_d >= _BYOC_RPH_THRESHOLD else ("⚠️" if _rph_d >= _BYOC_RPH_THRESHOLD * 0.85 else "❌")

        _dt_all = _bt[_bt["dim_driver_id"] == _did].copy() if not _bt.empty else pd.DataFrame()
        _dv_all = _bt_valid[_bt_valid["dim_driver_id"] == _did].copy() if not _bt_valid.empty else pd.DataFrame()

        _fare_pm = (
            float(_dt_all["fare"].sum() / _dt_all["dist"].replace(0, np.nan).sum())
            if not _dt_all.empty and _dt_all["dist"].sum() > 0 else None
        )

        with st.expander(
            f"{_v_icon} {_dname}  ·  £{_rph_d:.2f}/hr  ·  {_hrs_d:.0f} hrs  ·  {_conf(_hrs_d)}",
            expanded=False,
        ):
            # Metric row 1
            _r1a, _r1b, _r1c, _r1d, _r1e, _r1f = st.columns(6)
            _r1a.metric("Online Hours",  f"{_hrs_d:.1f}")
            _r1b.metric("Revenue",       f"£{_rev_d:,.2f}")
            _r1c.metric("RPH",           f"£{_rph_d:.2f}", delta=f"{_rph_d - _BYOC_RPH_THRESHOLD:+.2f} vs £21")
            _r1d.metric("Trips",         str(_trips_d))
            _r1e.metric("Active Days",   str(_adays_d))
            _r1f.metric("Avg Fare",      f"£{_afar_d:.2f}")

            # Metric row 2
            _r2a, _r2b, _r2c, _r2d = st.columns(4)
            _r2a.metric("Utilisation",  f"{_util_d:.1f}%")
            _r2b.metric("Acceptance",   f"{_acc_d:.1f}%")
            _r2c.metric("Fare / Mile",  f"£{_fare_pm:.2f}" if _fare_pm is not None else "—")
            _r2d.metric("Rev / Trip",   f"£{_rev_d / max(_trips_d, 1):.2f}")

            if _dt_all.empty:
                st.info("No trip records found for this driver.")
                continue

            # Daily RPH trend (only if > 1 active day)
            if not _bd.empty:
                _dd = _bd[_bd["dim_driver_id"] == _did].copy()
                if len(_dd) > 1:
                    _fig_dt = px.line(
                        _dd, x="driver_performance_date", y="rph", markers=True,
                        title="Daily RPH trend",
                        labels={"driver_performance_date": "Date", "rph": "RPH (£)"},
                        height=240,
                    )
                    _fig_dt.add_hline(y=_BYOC_RPH_THRESHOLD, line_dash="dash",
                                      line_color="#f59e0b", annotation_text="£21")
                    _fig_dt.update_layout(plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(gridcolor="#e2e8f0"))
                    st.plotly_chart(_fig_dt, use_container_width=True)
                elif len(_dd) == 1:
                    st.info(f"Single active day — RPH £{float(_dd.iloc[0]['rph']):.2f}, {float(_dd.iloc[0]['online_hrs']):.1f} hrs.")

            # Three side-by-side charts
            _ch1, _ch2, _ch3 = st.columns(3)

            with _ch1:
                _hd = _dt_all.groupby("hour")["fare"].agg(count="count", avg_fare="mean").reset_index()
                _fig_hr = px.bar(
                    _hd, x="hour", y="count", color="avg_fare",
                    color_continuous_scale="RdYlGn", range_color=[8, 25],
                    title="Trips by hour of day",
                    labels={"hour": "Hour", "count": "Trips", "avg_fare": "Avg £"},
                    height=280,
                )
                _fig_hr.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                st.plotly_chart(_fig_hr, use_container_width=True)

            with _ch2:
                if not _dv_all.empty:
                    _zc = _dv_all["pickup_zone"].value_counts().reset_index()
                    _zc.columns = ["Zone", "Trips"]
                    _fig_zp = px.pie(
                        _zc, names="Zone", values="Trips",
                        title="Pickup zone distribution",
                        height=280,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    _fig_zp.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.25))
                    st.plotly_chart(_fig_zp, use_container_width=True)
                else:
                    st.caption("No zone data available.")

            with _ch3:
                _fb = pd.cut(
                    _dt_all["fare"],
                    bins=[0, 10, 15, 20, 30, 999],
                    labels=["<£10", "£10-15", "£15-20", "£20-30", "£30+"],
                ).value_counts().reset_index()
                _fb.columns = ["Fare Band", "Trips"]
                _fig_fb = px.bar(
                    _fb, x="Fare Band", y="Trips",
                    title="Fare distribution",
                    color="Trips", color_continuous_scale="Blues",
                    height=280,
                )
                _fig_fb.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(_fig_fb, use_container_width=True)

            # Quick-read summary line
            if not _dv_all.empty:
                _west_pct  = _dv_all["is_west"].mean() * 100
                _sub10_pct = (_dt_all["fare"] < 10).mean() * 100
                _avg_dist  = _dt_all["dist"].mean()
                st.caption(
                    f"**West %:** {_west_pct:.1f}%  ·  "
                    f"**Sub-£10 trips:** {_sub10_pct:.1f}%  ·  "
                    f"**Avg trip distance:** {_avg_dist:.1f} mi"
                )

    # ── 8. BYOC cohort vs fleet ────────────────────────────────────────────────
    st.subheader("BYOC Cohort vs Rest of Fleet")
    _cmp_df = pd.DataFrame({
        "Group": ["BYOC (weighted)", "Rest of Fleet"],
        "RPH":   [_w_rph, _bfl_rph],
    })
    _fig_cmp = px.bar(
        _cmp_df, x="Group", y="RPH",
        color="Group",
        color_discrete_map={"BYOC (weighted)": "#f59e0b", "Rest of Fleet": "#94a3b8"},
        text=[f"£{r:.2f}" for r in _cmp_df["RPH"]],
        title="Weighted average RPH — BYOC cohort vs rest of fleet",
        height=320,
        labels={"RPH": "Revenue per Online Hour (£)"},
    )
    _fig_cmp.add_hline(y=_BYOC_RPH_THRESHOLD, line_dash="dash", line_color="#ef4444",
                       annotation_text="£21 threshold")
    _fig_cmp.update_traces(textposition="outside")
    _fig_cmp.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(_fig_cmp, use_container_width=True)

    # ── 9. Pickup map ──────────────────────────────────────────────────────────
    st.subheader("Pickup Location Map")
    if not _bt_valid.empty:
        _BYOC_COLORS = [
            "#f59e0b", "#3b82f6", "#22c55e", "#ef4444", "#a855f7",
            "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#14b8a6",
        ]
        _byoc_m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11, tiles="CartoDB positron")
        for _ci, _mid in enumerate(_byoc_ids):
            _dm = _bt_valid[_bt_valid["dim_driver_id"] == _mid]
            if _dm.empty:
                continue
            _dlabel = _id_to_display.get(_mid, str(_mid))
            _dc = _BYOC_COLORS[_ci % len(_BYOC_COLORS)]
            _fg = folium.FeatureGroup(name=_dlabel, show=True)
            for _, _tr in _dm.iterrows():
                folium.CircleMarker(
                    location=[_tr["plat"], _tr["plon"]],
                    radius=4,
                    color=_dc,
                    fill=True,
                    fill_color=_dc,
                    fill_opacity=0.65,
                    popup=folium.Popup(f"<b>{_dlabel}</b><br>£{_tr['fare']:.2f}", max_width=160),
                    tooltip=f"{_dlabel}: £{_tr['fare']:.2f}",
                ).add_to(_fg)
            _fg.add_to(_byoc_m)
        folium.LayerControl().add_to(_byoc_m)
        st_folium(_byoc_m, use_container_width=True, height=520)
    else:
        st.info("No coordinate data available for map.")

