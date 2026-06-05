import os
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from config import DRIVER_NAMES, BAD_DRIVER_IDS, TOP_DRIVER_IDS
import db
from zones import enrich_zones, parse_dms, assign_zone, CENTER_LAT, CENTER_LON, GEOJSON_DATA, is_valid_london_trip, calc_true_rph, estimate_ping

_FLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_data.parquet")

st.set_page_config(
    page_title="Top Driver Analysis",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #f59e0b;
    margin-bottom: 8px;
}
.metric-label { color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { color: #f9fafb; font-size: 26px; font-weight: 700; }
.metric-sub { color: #6b7280; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚖 Driver Analysis")
    st.caption("Top 10 Performers · 21+ £/hr")
    st.divider()
    page = st.radio("View", [
        "Patterns Summary",
        "Good vs Bad",
        "Fleet Map",
        "Driver Day",
        "Gap Analysis",
        "Trip Flow",
        "Zone 1 Selectivity",
        "Zone 1: The Why",
        "Trip Strategy DNA",
        "Airport Run Model",
        "Zone 3 Deep Dive",
        "Overview",
        "Map View",
        "Time Patterns",
        "Daily Trends",
        "Trip Economics",
        "Zone Analysis",
        "Day Patterns",
        "Shift Behaviour",
    ])
    st.divider()
    all_names = list(DRIVER_NAMES.values())
    selected_names = st.multiselect("Filter drivers", all_names, default=all_names)
    selected_ids = [k for k, v in DRIVER_NAMES.items() if v in selected_names]

# ── Data ─────────────────────────────────────────────────────────────────────
if page != "Patterns Summary":
    with st.spinner("Loading data..."):
        overview_df = db.load_overview()
        overview_df["display_name"] = overview_df["dim_driver_id"].map(DRIVER_NAMES).fillna(overview_df["driver_name"])
    overview_filtered = overview_df[overview_df["dim_driver_id"].isin(selected_ids)]

# ── Patterns Summary ─────────────────────────────────────────────────────────
if page == "Patterns Summary":
    st.title("Patterns Summary")
    st.caption("What the data actually says about how these 10 drivers earn more.")

    # ── Load all data needed
    with st.spinner("Running analysis..."):
        perf_df = db.load_overview()
        perf_df["display_name"] = perf_df["dim_driver_id"].map(DRIVER_NAMES).fillna(perf_df["driver_name"])

        raw_z = db.load_zone_trips()
        raw_z["display_name"] = raw_z["dim_driver_id"].map(DRIVER_NAMES).fillna(raw_z["driver_full_name"])
        zone_df2 = enrich_zones(raw_z)
        zone_df2 = calc_true_rph(zone_df2)
        zone_df2 = zone_df2[zone_df2["dim_driver_id"].isin(selected_ids)].copy()
        zone_df2["pickup_zone"] = zone_df2["pickup_zone"].astype(int)
        zone_df2["dropoff_zone"] = zone_df2["dropoff_zone"].astype(int)

        hourly_df = db.load_hourly_trips()
        hourly_df = hourly_df[hourly_df["dim_driver_id"].isin(selected_ids)]

    # Fleet baseline (hardcoded from run_analysis.py output)
    FLEET_RPH    = 23.04
    FLEET_UTIL   = 68.2
    FLEET_ACCEPT = 57.2

    top10_rph    = perf_df["rph"].mean()
    top10_util   = perf_df["avg_util"].mean()
    top10_accept = perf_df["avg_acceptance"].mean()

    # ── SECTION 1: The big picture
    st.subheader("1 — How much better are they, really?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 10 avg RPH",    f"£{top10_rph:.2f}/hr",  f"+£{top10_rph-FLEET_RPH:.2f} vs fleet")
    c2.metric("Utilisation",        f"{top10_util:.1f}%",     f"+{top10_util-FLEET_UTIL:.1f}% vs fleet")
    c3.metric("Acceptance rate",    f"{top10_accept:.1f}%",   f"{top10_accept-FLEET_ACCEPT:.1f}% vs fleet", delta_color="inverse")
    c4.metric("Bolt acceptance",    "~37%",                   "Declining 63% of Bolt pings")

    st.info("📌 **Lower acceptance rate, higher earnings.** These drivers decline more trips than average — they are not grabbing everything they're sent.")

    st.divider()

    # ── SECTION 2: Zone efficiency
    st.subheader("2 — Zone 1 earns nearly double Zone 3 per real hour")

    zone_eff = zone_df2.groupby("pickup_zone").agg(
        trips=("trip_price_in_pound", "count"),
        avg_fare=("trip_price_in_pound", "mean"),
        avg_wait=("gap_mins", "mean"),
        avg_ride=("pob_duration_in_min", "mean"),
        true_rph=("true_rph", "mean"),
        avg_dropoff=("dropoff_zone", "mean"),
    ).round(1).reset_index()
    zone_eff["zone_label"] = "Zone " + zone_eff["pickup_zone"].astype(str)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig_z = px.bar(
            zone_eff.sort_values("pickup_zone"),
            x="zone_label", y="true_rph",
            color="true_rph", color_continuous_scale="RdYlGn",
            text="true_rph",
            labels={"zone_label": "Pickup Zone", "true_rph": "True RPH £"},
        )
        fig_z.update_traces(texttemplate="£%{text:.0f}", textposition="outside")
        fig_z.update_layout(coloraxis_showscale=False, height=320,
                            title="True RPH by Pickup Zone (incl. wait time)")
        st.plotly_chart(fig_z, use_container_width=True)

    with col_b:
        fig_w = px.bar(
            zone_eff.sort_values("pickup_zone"),
            x="zone_label", y="avg_wait",
            color="avg_wait", color_continuous_scale="RdYlGn_r",
            text="avg_wait",
            labels={"zone_label": "Pickup Zone", "avg_wait": "Avg Wait (mins)"},
        )
        fig_w.update_traces(texttemplate="%{text:.0f} min", textposition="outside")
        fig_w.update_layout(coloraxis_showscale=False, height=320,
                            title="Avg Wait Time by Pickup Zone")
        st.plotly_chart(fig_w, use_container_width=True)

    st.info("📌 **Zone 3 is the trap.** It has the highest average wait (32 min mean) and lowest full-cycle True RPH (~£17/hr). Zone 1 is strongest at ~£21/hr full-cycle. Zone 6 is best at ~£23/hr but trips are long — they drop passengers far from the city. (True RPH = fare ÷ (inter-trip gap + pickup time + ride time).)")

    # Zone dropoff table
    st.markdown("**Where do pickups in each zone actually drop off?**")
    dropoff_tbl = zone_eff[["zone_label", "trips", "avg_fare", "avg_wait", "avg_ride", "true_rph", "avg_dropoff"]].copy()
    dropoff_tbl.columns = ["Pickup Zone", "Trips", "Avg Fare £", "Avg Wait min", "Avg Ride min", "True RPH £", "Avg Dropoff Zone"]
    dropoff_tbl["Avg Dropoff Zone"] = dropoff_tbl["Avg Dropoff Zone"].apply(lambda x: f"Zone {x:.1f}")
    st.dataframe(dropoff_tbl, use_container_width=True, hide_index=True)

    st.divider()

    # ── SECTION 3: Acceptance patterns
    st.subheader("3 — They decline Zone 1 trips more than any other zone")

    accept_zone_data = pd.DataFrame({
        "Zone":     ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"],
        "Accepted": [27.8, 26.6, 24.3, 10.1, 5.6, 5.6],
        "Declined": [34.9, 28.2, 23.1,  7.7, 3.3, 2.8],
    })
    accept_melt = accept_zone_data.melt(id_vars="Zone", var_name="Outcome", value_name="% of trips")
    fig_acc = px.bar(
        accept_melt, x="Zone", y="% of trips", color="Outcome",
        barmode="group",
        color_discrete_map={"Accepted": "#22c55e", "Declined": "#ef4444"},
        labels={"Zone": "Pickup Zone"},
        title="% of Accepted vs Declined trips by pickup zone (Bolt only)",
    )
    fig_acc.update_layout(height=360)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.info("📌 **Zone 1 selectivity is the pattern.** Drivers receive more Zone 1 pings proportionally than any other zone — and decline the highest share of them. They're not taking every Zone 1 trip; they're waiting for the ones worth it (longer distance, off-peak timing, higher fare).")

    # Per-driver acceptance
    accept_driver = pd.DataFrame([
        {"Driver": "Monier Janabi",         "Accept%": 11, "Most Declined": "Zone 3", "Most Accepted": "Zone 3"},
        {"Driver": "Marius Norvaisas",       "Accept%": 88, "Most Declined": "Zone 2", "Most Accepted": "Zone 3"},
        {"Driver": "Ertac Cindogulu",        "Accept%": 19, "Most Declined": "Zone 1", "Most Accepted": "Zone 1"},
        {"Driver": "Bal Jamts",              "Accept%": 33, "Most Declined": "Zone 1", "Most Accepted": "Zone 1"},
        {"Driver": "Abdi Saeed Mohamed",     "Accept%": 51, "Most Declined": "Zone 2", "Most Accepted": "Zone 2"},
        {"Driver": "Anish Chaudhry",         "Accept%": 54, "Most Declined": "Zone 1", "Most Accepted": "Zone 2"},
        {"Driver": "MHD Amir Aljaghsi",      "Accept%": 77, "Most Declined": "Zone 3", "Most Accepted": "Zone 4"},
        {"Driver": "Jermaine Asante Gyamfi", "Accept%": 52, "Most Declined": "Zone 3", "Most Accepted": "Zone 3"},
        {"Driver": "Mohamed Warsame Nur",    "Accept%": 39, "Most Declined": "Zone 2", "Most Accepted": "Zone 2"},
        {"Driver": "Brijenkumar Patel",      "Accept%": 59, "Most Declined": "Zone 3", "Most Accepted": "Zone 3"},
    ])
    accept_driver = accept_driver[accept_driver["Driver"].isin(selected_names)]

    fig_ad = px.bar(
        accept_driver.sort_values("Accept%"),
        x="Accept%", y="Driver", orientation="h",
        color="Accept%", color_continuous_scale="RdYlGn",
        text="Accept%",
        title="Bolt Acceptance Rate per Driver (%)",
        labels={"Accept%": "Acceptance %", "Driver": ""},
    )
    fig_ad.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_ad.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_ad, use_container_width=True)

    st.divider()

    # ── SECTION 4: When they work
    st.subheader("4 — They work different shifts but all avoid the Zone 3 dead zone")

    shift_data = pd.DataFrame([
        {"Driver": "Marius Norvaisas",       "Start": 4,  "End": 18, "Peak Hour": "04:00", "Avg Peak Fare": 18.90},
        {"Driver": "Abdi Saeed Mohamed",     "Start": 6,  "End": 15, "Peak Hour": "07:00", "Avg Peak Fare": 22.40},
        {"Driver": "Mohamed Warsame Nur",    "Start": 7,  "End": 16, "Peak Hour": "09:00", "Avg Peak Fare": 12.07},
        {"Driver": "Anish Chaudhry",         "Start": 7,  "End": 20, "Peak Hour": "10:00", "Avg Peak Fare": 13.84},
        {"Driver": "Bal Jamts",              "Start": 9,  "End": 18, "Peak Hour": "11:00", "Avg Peak Fare": 12.90},
        {"Driver": "Ertac Cindogulu",        "Start": 9,  "End": 16, "Peak Hour": "13:00", "Avg Peak Fare": 16.15},
        {"Driver": "Brijenkumar Patel",      "Start": 8,  "End": 19, "Peak Hour": "15:00", "Avg Peak Fare": 12.72},
        {"Driver": "Monier Janabi",          "Start": 8,  "End": 17, "Peak Hour": "18:00", "Avg Peak Fare": 20.11},
        {"Driver": "Jermaine Asante Gyamfi", "Start": 10, "End": 20, "Peak Hour": "18:00", "Avg Peak Fare": 13.07},
        {"Driver": "MHD Amir Aljaghsi",      "Start": 8,  "End": 22, "Peak Hour": "21:00", "Avg Peak Fare": 12.98},
    ])
    shift_data = shift_data[shift_data["Driver"].isin(selected_names)]

    fig_sh = px.timeline(
        shift_data.assign(
            Start_dt=pd.to_datetime("2024-01-01") + pd.to_timedelta(shift_data["Start"], unit="h"),
            End_dt=pd.to_datetime("2024-01-01")   + pd.to_timedelta(shift_data["End"],   unit="h"),
        ),
        x_start="Start_dt", x_end="End_dt", y="Driver",
        color="Avg Peak Fare", color_continuous_scale="YlOrRd",
        title="Typical Shift Window per Driver (coloured by peak-hour fare)",
        labels={"Avg Peak Fare": "Peak Fare £"},
    )
    fig_sh.update_xaxes(tickformat="%H:%M")
    fig_sh.update_layout(height=380, coloraxis_colorbar_title="Peak Fare £")
    st.plotly_chart(fig_sh, use_container_width=True)

    st.info("📌 **Three distinct shift profiles emerge:**\n"
            "- **Early birds (4–6am start):** Marius, Abdi — catch airport runs and night-shift end fares (highest peak fares)\n"
            "- **Day shift (9am–4pm):** Bal Jamts, Ertac, Mohamed Warsame — high volume, moderate fares\n"
            "- **Evening peak hunters (finish 8–10pm):** Monier, Jermaine, MHD — targeting 6–9pm surge")

    st.divider()

    # ── SECTION 5: Platform
    st.subheader("5 — 8 of 10 are Bolt-primary")
    plat_data = pd.DataFrame([
        {"Driver": "Marius Norvaisas",       "Uber": 64, "Bolt": 36},
        {"Driver": "Jermaine Asante Gyamfi", "Uber": 55, "Bolt": 45},
        {"Driver": "Bal Jamts",              "Uber": 18, "Bolt": 82},
        {"Driver": "Ertac Cindogulu",        "Uber":  5, "Bolt": 95},
        {"Driver": "Monier Janabi",          "Uber":  0, "Bolt": 100},
        {"Driver": "Abdi Saeed Mohamed",     "Uber":  0, "Bolt": 100},
        {"Driver": "Anish Chaudhry",         "Uber":  0, "Bolt": 100},
        {"Driver": "MHD Amir Aljaghsi",      "Uber":  0, "Bolt": 100},
        {"Driver": "Mohamed Warsame Nur",    "Uber":  0, "Bolt": 100},
        {"Driver": "Brijenkumar Patel",      "Uber":  0, "Bolt": 100},
    ])
    plat_data = plat_data[plat_data["Driver"].isin(selected_names)]
    plat_melt = plat_data.melt(id_vars="Driver", var_name="Platform", value_name="% of Trips")
    fig_p = px.bar(plat_melt, x="% of Trips", y="Driver", color="Platform", barmode="stack",
                   orientation="h",
                   color_discrete_map={"Uber": "#000000", "Bolt": "#34d399"},
                   labels={"Driver": ""})
    fig_p.update_layout(height=360, legend=dict(orientation="h"))
    st.plotly_chart(fig_p, use_container_width=True)

    st.info("📌 **Bolt-first strategy.** The two drivers with the most Uber trips (Marius 64%, Jermaine 55%) also have the longest trip histories and operate across more zones. The pure-Bolt drivers tend to be more zone-concentrated.")

    st.divider()

    # ── SECTION 6: Trip flow insight
    st.subheader("6 — What makes a 'better' trip? (Bolt shows estimated fare + distance — drivers filter on both)")

    if os.path.exists(_FLOW_PATH):
        flow_sum = pd.read_parquet(_FLOW_PATH)
        flow_sum = flow_sum[flow_sum["dim_driver_id"].isin(selected_ids)]
        acc_s = flow_sum[flow_sum["outcome"]=="Accepted"]
        dec_s = flow_sum[flow_sum["outcome"]=="Declined"]

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            def make_flow_mini(df, title, colour):
                pivot = df.groupby(["pickup_zone","dropoff_zone"]).size().unstack(fill_value=0)
                pivot.index   = [f"P Z{z}" for z in pivot.index]
                pivot.columns = [f"D Z{z}" for z in pivot.columns]
                pct = (pivot.div(pivot.values.sum()) * 100).round(1)
                fig = px.imshow(pct, text_auto=".0f", color_continuous_scale=colour,
                                labels={"x":"Dropoff","y":"Pickup","color":"%"},
                                title=title, aspect="auto")
                fig.update_layout(height=320, coloraxis_showscale=False,
                                  margin=dict(l=10,r=10,t=40,b=10))
                return fig
            st.plotly_chart(make_flow_mini(acc_s, "Accepted routes (% of total)", "Greens"),
                            use_container_width=True)
        with col_f2:
            st.plotly_chart(make_flow_mini(dec_s, "Declined routes (% of total)", "Reds"),
                            use_container_width=True)

        acc_z1do = acc_s["dropoff_zone"].eq(1).mean()*100
        dec_z1do = dec_s["dropoff_zone"].eq(1).mean()*100
        acc_z56do = acc_s["dropoff_zone"].ge(5).mean()*100
        dec_z56do = dec_s["dropoff_zone"].ge(5).mean()*100
        st.info(f"📌 Declined trips go TO Zone 1 more ({dec_z1do:.0f}% of declines vs {acc_z1do:.0f}% of accepts) — "
                f"drivers are turning down short intra-city hops.\n\n"
                f"📌 Accepted trips go to Zone 5-6 more ({acc_z56do:.0f}% of accepts vs {dec_z56do:.0f}% of declines) — "
                f"they favour longer runs to airports and outer zones.\n\n"
                f"**A 'better' trip = longer destination, preferably Zone 2–4 dropoff, not a Zone 1 local hop.**")
    else:
        st.warning("Run `python build_flow.py` to enable trip flow analysis.")

    st.divider()

    # ── SECTION 7: Zone 3 nuance
    st.subheader("7 — Zone 3 isn't always a dead zone")
    z3_summary = pd.DataFrame([
        {"Period": "Night (00–06)",    "Raw RPH": 36, "Verdict": "✅ Worth it"},
        {"Period": "Morning (06–09)",  "Raw RPH": 34, "Verdict": "✅ Worth it"},
        {"Period": "Day (09–17)",      "Raw RPH": 27, "Verdict": "❌ Avoid"},
        {"Period": "Evening (17–23)",  "Raw RPH": 29, "Verdict": "⚠️  Borderline"},
    ])
    fig_z3s = px.bar(z3_summary, x="Period", y="Raw RPH", color="Verdict",
                     color_discrete_map={"✅ Worth it":"#22c55e","❌ Avoid":"#ef4444","⚠️  Borderline":"#f59e0b"},
                     text="Raw RPH",
                     labels={"Raw RPH":"Raw RPH £/hr"},
                     title="Zone 3 Raw RPH by time period (no wait time included)")
    fig_z3s.add_hline(y=21, line_dash="dash", line_color="#3b82f6",
                      annotation_text="Zone 1 avg £21/hr", annotation_position="top right")
    fig_z3s.update_traces(texttemplate="£%{text}", textposition="outside")
    fig_z3s.update_layout(height=360, coloraxis_showscale=False,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_z3s, use_container_width=True)
    st.info("📌 Zone 3 at **03:00 peaks at £43/hr** — longer trips (avg 11mi) going into the city. Daytime Zone 3 (£26–28/hr + 32 min avg wait) is where it falls apart. Best Zone 3 areas: Ealing, Brent, Wandsworth. Worst: Enfield, East Ham, Walthamstow.")

    st.divider()

    # ── SECTION 8: Final summary table
    st.subheader("All patterns — at a glance")
    st.markdown("""
| # | Pattern | Evidence | Action |
|---|---|---|---|
| 1 | **Zone 1 = best full-cycle RPH (~£21/hr)** | Short pickup wait (6.7 min), high ping density | Stay in Zone 1 during daytime — but be selective |
| 2 | **Zone 3 daytime = dead zone** | £27/hr + 32 min avg wait | Leave Zone 3 before 09:00 or after 17:00 |
| 3 | **Zone 3 night = good** | £32–43/hr, 03:00 = £43/hr peak | Zone 3 valid for night-shift drivers |
| 4 | **Decline Zone 1 hops, wait for cross-zone trips** | 37.7% of declines go to Z1 vs 29.2% of accepts | On Bolt: check destination before accepting |
| 5 | **Outer zone drops strand you far out** | Z5-6 pickups drop at Zone 4.5 avg | Factor in return positioning cost |
| 6 | **Lower acceptance rate = higher RPH** | Top 10: 49% accept vs fleet 57% | Selectivity is the strategy, not volume |
| 7 | **Three shift archetypes** | Early (4–6am), Day (9am–4pm), Evening (4–10pm) | Pick the archetype that suits the zone strategy |
| 8 | **Bolt-primary** | 8 of 10 on Bolt (no fare visibility on ping) | Route = the only signal available for selection |
    """)

    st.divider()

    # ── DOWNLOAD REPORT ──────────────────────────────────────────────────────
    st.subheader("Download Report")

    import io, datetime

    def build_report():
        lines = []
        lines.append("TOP DRIVER PATTERN ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}")
        lines.append(f"Drivers analysed: {', '.join(selected_names)}")
        lines.append("=" * 70)

        lines.append("\n1. PERFORMANCE VS FLEET")
        lines.append(f"   Fleet avg RPH:          £{FLEET_RPH}/hr")
        lines.append(f"   Top 10 avg RPH:         £{top10_rph:.2f}/hr  (+£{top10_rph-FLEET_RPH:.2f})")
        lines.append(f"   Fleet utilisation:      {FLEET_UTIL}%")
        lines.append(f"   Top 10 utilisation:     {top10_util:.1f}%  (+{top10_util-FLEET_UTIL:.1f}%)")
        lines.append(f"   Fleet accept rate:      {FLEET_ACCEPT}%")
        lines.append(f"   Top 10 accept rate:     {top10_accept:.1f}%  ({top10_accept-FLEET_ACCEPT:+.1f}%)")

        lines.append("\n2. ZONE EFFICIENCY (True RPH incl. wait time)")
        for _, row in zone_eff.iterrows():
            lines.append(f"   Zone {int(row.pickup_zone)}: £{row.true_rph:.0f}/hr  |  avg fare £{row.avg_fare:.2f}  |  avg wait {row.avg_wait:.0f}min  |  avg dropoff Z{row.avg_dropoff:.1f}")

        lines.append("\n3. TRIP FLOW (Bolt — route-based decision making)")
        if os.path.exists(_FLOW_PATH):
            lines.append(f"   Accepted trips to Zone 1 dropoff: {acc_z1do:.0f}%")
            lines.append(f"   Declined trips to Zone 1 dropoff: {dec_z1do:.0f}%")
            lines.append(f"   Accepted trips to Zone 5-6 dropoff: {acc_z56do:.0f}%")
            lines.append(f"   Declined trips to Zone 5-6 dropoff: {dec_z56do:.0f}%")

        lines.append("\n4. ZONE 3 BREAKDOWN BY TIME")
        lines.append("   Night (00-06): £36/hr avg raw RPH  — WORTH IT")
        lines.append("   Morning (06-09): £34/hr            — WORTH IT")
        lines.append("   Day (09-17): £27/hr + 32min wait   — AVOID")
        lines.append("   Evening (17-23): £29/hr            — BORDERLINE")

        lines.append("\n5. SHIFT PROFILES")
        shift_profiles = [
            ("Marius Norvaisas","04:00","18:00","04:00","£18.90"),
            ("Abdi Saeed Mohamed","06:00","15:00","07:00","£22.40"),
            ("Mohamed Warsame Nur","07:00","16:00","09:00","£12.07"),
            ("Bal Jamts","09:00","18:00","11:00","£12.90"),
            ("Ertac Cindogulu","09:00","16:00","13:00","£16.15"),
            ("Jermaine Asante Gyamfi","10:00","20:00","18:00","£13.07"),
            ("Monier Janabi","08:00","17:00","18:00","£20.11"),
            ("MHD Amir Aljaghsi","08:00","22:00","21:00","£12.98"),
        ]
        for name, start, end, peak, fare in shift_profiles:
            if name in selected_names:
                lines.append(f"   {name:<28} {start}–{end}  peak {peak} @ {fare}/trip avg")

        lines.append("\n6. ACCEPTANCE RATES (Bolt only)")
        accept_rates = [
            ("Monier Janabi",11,"Zone 3","Zone 3"),("Marius Norvaisas",88,"Zone 2","Zone 3"),
            ("Ertac Cindogulu",19,"Zone 1","Zone 1"),("Bal Jamts",33,"Zone 1","Zone 1"),
            ("Abdi Saeed Mohamed",51,"Zone 2","Zone 2"),("Anish Chaudhry",54,"Zone 1","Zone 2"),
            ("MHD Amir Aljaghsi",77,"Zone 3","Zone 4"),("Jermaine Asante Gyamfi",52,"Zone 3","Zone 3"),
            ("Mohamed Warsame Nur",39,"Zone 2","Zone 2"),("Brijenkumar Patel",59,"Zone 3","Zone 3"),
        ]
        for name, rate, declined_z, accepted_z in accept_rates:
            if name in selected_names:
                lines.append(f"   {name:<28} {rate}% accept  most declined: {declined_z}  most accepted: {accepted_z}")

        lines.append("\n7. KEY RECOMMENDATIONS")
        lines.append("   - Stay in Zone 1 during 09:00-17:00 — best full-cycle RPH (~£21/hr), be selective on accepts")
        lines.append("   - Avoid Zone 3 09:00-17:00 — low fares + 32 min avg wait")
        lines.append("   - Zone 3 is viable 00:00-06:00 (£36-43/hr, long trips into city)")
        lines.append("   - On Bolt: check destination before accepting — decline short Zone 1 hops")
        lines.append("   - Zone 5-6 trips leave you far from city — factor in return cost")
        lines.append("   - Target 70%+ utilisation and be selective, not high-volume")

        lines.append("\n" + "=" * 70)
        lines.append("Generated by Top Driver Analysis Dashboard")
        return "\n".join(lines)

    report_text = build_report()
    st.download_button(
        label="⬇ Download Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"driver_patterns_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )

    # CSV download of per-driver stats
    driver_stats_csv = overview_filtered[["display_name","active_days","total_rides",
                                           "total_online_hrs","rph","avg_util",
                                           "avg_rating","avg_acceptance","total_revenue"]].copy()
    driver_stats_csv.columns = ["Driver","Active Days","Rides","Online Hrs","RPH £",
                                 "Util %","Rating","Accept %","Total Revenue £"]
    st.download_button(
        label="⬇ Download Driver Stats (.csv)",
        data=driver_stats_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"driver_stats_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ── Good vs Bad ──────────────────────────────────────────────────────────────
elif page == "Good vs Bad":
    st.title("Good vs Bad Drivers")
    st.caption("Top 10 performers vs 5 lower-performing drivers — zone flow, fare distribution, and key metrics side by side.")

    _GOOD_IDS = [i for i in DRIVER_NAMES if i not in BAD_DRIVER_IDS]
    _BAD_IDS  = BAD_DRIVER_IDS

    with st.spinner("Loading performance data..."):
        good_perf = db.load_comparison_performance(_GOOD_IDS)
        bad_perf  = db.load_comparison_performance(_BAD_IDS)

    with st.spinner("Loading trip flow data..."):
        good_flow_raw = db.load_comparison_flow(_GOOD_IDS)
        bad_flow_raw  = db.load_comparison_flow(_BAD_IDS)

    # ── SECTION 1: Headline metrics ───────────────────────────────────────────
    st.subheader("1 — Headline numbers")

    def _group_avg(perf_df):
        return {
            "RPH":         perf_df["rph"].mean(),
            "Acceptance %":perf_df["acceptance"].mean(),
            "Avg fare £":  perf_df["avg_fare"].mean(),
            "Utilisation %": perf_df["utilisation"].mean(),
            "Total rides": perf_df["total_rides"].sum(),
        }

    ga = _group_avg(good_perf)
    ba = _group_avg(bad_perf)

    col_g, col_b = st.columns(2)
    with col_g:
        st.markdown("### ✅ Top 10 drivers")
        st.metric("RPH",           f"£{ga['RPH']:.2f}/hr")
        st.metric("Acceptance",    f"{ga['Acceptance %']:.1f}%")
        st.metric("Avg fare",      f"£{ga['Avg fare £']:.2f}")
        st.metric("Utilisation",   f"{ga['Utilisation %']:.1f}%")
        st.metric("Total rides",   f"{int(ga['Total rides']):,}")
    with col_b:
        st.markdown("### ⚠️ Comparison drivers")
        st.metric("RPH",           f"£{ba['RPH']:.2f}/hr",
                  delta=f"£{ba['RPH']-ga['RPH']:.2f} vs top 10")
        st.metric("Acceptance",    f"{ba['Acceptance %']:.1f}%",
                  delta=f"{ba['Acceptance %']-ga['Acceptance %']:+.1f}%")
        st.metric("Avg fare",      f"£{ba['Avg fare £']:.2f}",
                  delta=f"£{ba['Avg fare £']-ga['Avg fare £']:.2f}")
        st.metric("Utilisation",   f"{ba['Utilisation %']:.1f}%",
                  delta=f"{ba['Utilisation %']-ga['Utilisation %']:+.1f}%")
        st.metric("Total rides",   f"{int(ba['Total rides']):,}")

    # Per-driver breakdown table
    with st.expander("Per-driver breakdown"):
        good_perf["Group"] = "Top 10"
        bad_perf["Group"]  = "Comparison"
        combined = pd.concat([good_perf, bad_perf], ignore_index=True)
        combined["driver_name"] = combined["dim_driver_id"].map(DRIVER_NAMES).fillna(combined["driver_name"])
        st.dataframe(
            combined[["Group","driver_name","rph","acceptance","avg_fare","utilisation","total_rides"]]
            .rename(columns={"driver_name":"Driver","rph":"RPH £/hr","acceptance":"Accept %",
                              "avg_fare":"Avg fare £","utilisation":"Util %","total_rides":"Rides"})
            .sort_values("RPH £/hr", ascending=False),
            use_container_width=True, hide_index=True
        )

    # ── SECTION 2: Zone flow heatmaps ─────────────────────────────────────────
    st.divider()
    st.subheader("2 — Where they go: zone flow heatmap")
    st.caption("Pickup zone (rows) × Dropoff zone (cols) — cell = % of accepted trips. "
               "Brighter = more trips on that route.")

    def _build_zone_matrix(flow_raw):
        df = enrich_zones(flow_raw[flow_raw["status"].isin(["completed","Finished"])].copy())
        df = df.dropna(subset=["pickup_zone","dropoff_zone"])
        df["pickup_zone"]  = df["pickup_zone"].astype(int)
        df["dropoff_zone"] = df["dropoff_zone"].astype(int)
        mat = (df.groupby(["pickup_zone","dropoff_zone"])
                 .size().reset_index(name="trips"))
        total = mat["trips"].sum()
        mat["pct"] = (mat["trips"] / total * 100).round(1)
        pivot = mat.pivot(index="pickup_zone", columns="dropoff_zone", values="pct").fillna(0)
        # Ensure zones 1-6 on both axes
        for z in range(1, 7):
            if z not in pivot.index:   pivot.loc[z] = 0
            if z not in pivot.columns: pivot[z]     = 0
        return pivot.sort_index()[sorted(pivot.columns)]

    with st.spinner("Enriching zone data (this takes a moment)..."):
        good_mat = _build_zone_matrix(good_flow_raw)
        bad_mat  = _build_zone_matrix(bad_flow_raw)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_g = px.imshow(
            good_mat, text_auto=".1f",
            color_continuous_scale="Blues",
            labels=dict(x="Dropoff zone", y="Pickup zone", color="% of trips"),
            title="Top 10 — accepted trip zone flow (%)",
            aspect="equal",
        )
        fig_g.update_layout(height=380)
        st.plotly_chart(fig_g, use_container_width=True)

    with col_h2:
        fig_b = px.imshow(
            bad_mat, text_auto=".1f",
            color_continuous_scale="Reds",
            labels=dict(x="Dropoff zone", y="Pickup zone", color="% of trips"),
            title="Comparison drivers — accepted trip zone flow (%)",
            aspect="equal",
        )
        fig_b.update_layout(height=380)
        st.plotly_chart(fig_b, use_container_width=True)

    # Difference heatmap
    st.markdown("**Difference map** — green = top 10 do this more, red = comparison drivers do this more")
    diff_mat = good_mat.subtract(bad_mat, fill_value=0).round(1)
    fig_diff = px.imshow(
        diff_mat, text_auto=".1f",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels=dict(x="Dropoff zone", y="Pickup zone", color="Δ %"),
        title="Zone flow difference (Top 10 minus Comparison)",
        aspect="equal",
    )
    fig_diff.update_layout(height=380)
    st.plotly_chart(fig_diff, use_container_width=True)

    # ── SECTION 3: Fare distribution ─────────────────────────────────────────
    st.divider()
    st.subheader("3 — Fare distribution: are bad drivers taking cheaper trips?")

    def _accepted_fares(flow_raw):
        return flow_raw[flow_raw["status"].isin(["completed","Finished"])]["trip_price_in_pound"].dropna()

    good_fares = _accepted_fares(good_flow_raw)
    bad_fares  = _accepted_fares(bad_flow_raw)

    fare_df = pd.DataFrame({
        "Fare £": pd.concat([good_fares, bad_fares], ignore_index=True),
        "Group":  (["Top 10"] * len(good_fares)) + (["Comparison"] * len(bad_fares)),
    })
    fare_df = fare_df[fare_df["Fare £"] <= 80]   # clip outliers

    fig_fare = px.histogram(
        fare_df, x="Fare £", color="Group",
        barmode="overlay", nbins=40, opacity=0.7,
        color_discrete_map={"Top 10": "#22c55e", "Comparison": "#ef4444"},
        title="Fare distribution — accepted trips",
        labels={"Fare £": "Trip fare (£)", "count": "Number of trips"},
    )
    fig_fare.add_vline(x=good_fares.median(), line_dash="dash", line_color="#22c55e",
                       annotation_text=f"Top 10 median £{good_fares.median():.2f}",
                       annotation_position="top right")
    fig_fare.add_vline(x=bad_fares.median(), line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Comparison median £{bad_fares.median():.2f}",
                       annotation_position="top left")
    fig_fare.update_layout(height=380)
    st.plotly_chart(fig_fare, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 10 median fare",   f"£{good_fares.median():.2f}")
    c2.metric("Comparison median fare",f"£{bad_fares.median():.2f}",
              delta=f"£{bad_fares.median()-good_fares.median():.2f}")
    sub10_good = (good_fares < 10).mean() * 100
    sub10_bad  = (bad_fares  < 10).mean() * 100
    c3.metric("Top 10 trips under £10",   f"{sub10_good:.1f}%")
    c4.metric("Comparison trips under £10",f"{sub10_bad:.1f}%",
              delta=f"{sub10_bad-sub10_good:+.1f}%")

    # ── SECTION 4: Hourly activity ────────────────────────────────────────────
    st.divider()
    st.subheader("4 — When they work: hourly trip volume")
    st.caption("Are bad drivers grinding the wrong hours?")

    def _hourly(flow_raw):
        df = flow_raw[flow_raw["status"].isin(["completed","Finished"])].copy()
        df["trips_hr"] = pd.to_numeric(df["trips_hr"], errors="coerce")
        return df.groupby("trips_hr").size().reset_index(name="trips")

    good_hr = _hourly(good_flow_raw)
    bad_hr  = _hourly(bad_flow_raw)
    good_hr["Group"] = "Top 10"
    bad_hr["Group"]  = "Comparison"
    hr_df = pd.concat([good_hr, bad_hr], ignore_index=True)
    # Normalise to % of each group's total so volume difference doesn't mislead
    for grp in ["Top 10", "Comparison"]:
        mask = hr_df["Group"] == grp
        hr_df.loc[mask, "pct"] = (hr_df.loc[mask, "trips"] /
                                   hr_df.loc[mask, "trips"].sum() * 100).round(1)

    fig_hr = px.line(
        hr_df, x="trips_hr", y="pct", color="Group",
        color_discrete_map={"Top 10": "#22c55e", "Comparison": "#ef4444"},
        markers=True,
        labels={"trips_hr": "Hour of day", "pct": "% of trips"},
        title="Trip volume by hour — normalised (% of each group's total)",
    )
    fig_hr.update_layout(height=340, xaxis=dict(dtick=1))
    st.plotly_chart(fig_hr, use_container_width=True)

    # ── SECTION 5: Zone 3 trap ───────────────────────────────────────────────
    st.divider()
    st.subheader("5 — The Zone 3 trap")
    st.caption("Zone 3 has the worst daytime RPH in our data (~£18–22/hr full-cycle). "
               "Are comparison drivers stuck there while top 10 drivers stay in Zone 1/2?")

    def _z3_stats(mat):
        z3_pickup  = mat.loc[3].sum()  if 3 in mat.index   else 0
        z3_dropoff = mat[3].sum()      if 3 in mat.columns else 0
        z3_chain   = mat.loc[3, 3]     if (3 in mat.index and 3 in mat.columns) else 0
        z1_pickup  = mat.loc[1].sum()  if 1 in mat.index   else 0
        z1_chain   = mat.loc[1, 1]     if (1 in mat.index and 1 in mat.columns) else 0
        return z3_pickup, z3_dropoff, z3_chain, z1_pickup, z1_chain

    g_z3p, g_z3d, g_z3c, g_z1p, g_z1c = _z3_stats(good_mat)
    b_z3p, b_z3d, b_z3c, b_z1p, b_z1c = _z3_stats(bad_mat)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Zone 1 pickups (% of all trips)**")
        st.metric("Top 10",      f"{g_z1p:.1f}%")
        st.metric("Comparison",  f"{b_z1p:.1f}%", delta=f"{b_z1p - g_z1p:+.1f}%")
    with col2:
        st.markdown("**Zone 3 pickups (% of all trips)**")
        st.metric("Top 10",      f"{g_z3p:.1f}%")
        st.metric("Comparison",  f"{b_z3p:.1f}%", delta=f"{b_z3p - g_z3p:+.1f}%")
    with col3:
        st.markdown("**Zone 3→Zone 3 trips (stuck chaining)**")
        st.metric("Top 10",      f"{g_z3c:.1f}%")
        st.metric("Comparison",  f"{b_z3c:.1f}%", delta=f"{b_z3c - g_z3c:+.1f}%")

    # Zone 1 vs Zone 3 bar comparison
    zone_cmp = pd.DataFrame([
        {"Zone": "Z1 pickup",   "Top 10": g_z1p, "Comparison": b_z1p},
        {"Zone": "Z1→Z1 chain", "Top 10": g_z1c, "Comparison": b_z1c},
        {"Zone": "Z3 pickup",   "Top 10": g_z3p, "Comparison": b_z3p},
        {"Zone": "Z3 dropoff",  "Top 10": g_z3d, "Comparison": b_z3d},
        {"Zone": "Z3→Z3 chain", "Top 10": g_z3c, "Comparison": b_z3c},
    ])
    fig_z3 = px.bar(
        zone_cmp.melt(id_vars="Zone", var_name="Group", value_name="% of trips"),
        x="Zone", y="% of trips", color="Group", barmode="group",
        color_discrete_map={"Top 10": "#22c55e", "Comparison": "#ef4444"},
        title="Zone 1 vs Zone 3 activity — Top 10 vs Comparison (%)",
        text_auto=".1f",
    )
    fig_z3.update_layout(height=360)
    st.plotly_chart(fig_z3, use_container_width=True)

    # Narrative
    z3_gap = b_z3p - g_z3p
    z1_gap = g_z1p - b_z1p
    st.markdown(
        f'<div style="background:#1e1e2e;border-left:4px solid #ef4444;'
        f'padding:14px 16px;border-radius:6px;color:#e2e8f0;">'
        f'<strong>Why Zone 3 is a problem:</strong> Zone 3 daytime full-cycle RPH is ~£18–22/hr — '
        f'well below the fleet average of £22.9/hr. Once a driver is picking up in Zone 3, '
        f'they tend to stay there (Z3→Z3 chaining), because the pings they receive are from '
        f'nearby passengers also in Zone 3. It\'s a gravity well — hard to escape without '
        f'deliberately declining Zone 3 pings and waiting for one that pulls them back toward Zone 1.<br><br>'
        f'<strong>The numbers:</strong> Comparison drivers pick up from Zone 3 on <strong>{b_z3p:.1f}%</strong> '
        f'of trips vs <strong>{g_z3p:.1f}%</strong> for the top 10 — a <strong>{z3_gap:+.1f}%</strong> gap. '
        f'Simultaneously, top 10 drivers pick up from Zone 1 on <strong>{g_z1p:.1f}%</strong> of trips '
        f'vs <strong>{b_z1p:.1f}%</strong> for comparison — a <strong>{z1_gap:.1f}%</strong> gap in the right direction. '
        f'These two numbers together explain most of the RPH difference.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # How did they end up in Zone 3? — inbound flow to Z3
    st.markdown("#### How do drivers end up in Zone 3?")
    st.caption("These are the trips that DROP OFF in Zone 3 — this is what puts drivers there.")

    def _z3_inbound(mat):
        if 3 not in mat.columns:
            return pd.DataFrame()
        col = mat[3].reset_index()
        col.columns = ["Pickup Zone", "% ending in Z3"]
        col["Pickup Zone"] = col["Pickup Zone"].apply(lambda z: f"Z{z}")
        return col.sort_values("% ending in Z3", ascending=False)

    g_inbound = _z3_inbound(good_mat)
    b_inbound = _z3_inbound(bad_mat)

    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("**Top 10 — trips that end in Zone 3**")
        if not g_inbound.empty:
            fig_gi = px.bar(g_inbound, x="Pickup Zone", y="% ending in Z3",
                            color_discrete_sequence=["#22c55e"], text_auto=".1f",
                            title="Where did they come from?")
            fig_gi.update_layout(height=280)
            st.plotly_chart(fig_gi, use_container_width=True)
    with ci2:
        st.markdown("**Comparison — trips that end in Zone 3**")
        if not b_inbound.empty:
            fig_bi = px.bar(b_inbound, x="Pickup Zone", y="% ending in Z3",
                            color_discrete_sequence=["#ef4444"], text_auto=".1f",
                            title="Where did they come from?")
            fig_bi.update_layout(height=280)
            st.plotly_chart(fig_bi, use_container_width=True)

    # ── SECTION 6: East vs West positioning ──────────────────────────────────
    st.divider()
    st.subheader("6 — East vs West: where are they actually operating?")
    st.caption(
        "London's high-demand corridor runs roughly west of -0.12° longitude "
        "(Charing Cross line) — Mayfair, Kensington, Chelsea, Knightsbridge, Notting Hill. "
        "East of that line ping density drops sharply."
    )

    # West boundary longitude (roughly Charing Cross / City of London line)
    _WEST_LON = -0.12

    def _parse_pickup_coords(flow_raw, sample_n=800):
        """Parse pickup lat/lon for a sampled set of accepted trips."""
        df = flow_raw[flow_raw["status"].isin(["completed", "Finished"])].copy()
        if len(df) > sample_n:
            df = df.sample(sample_n, random_state=42)
        coords = df["pickup_lat_long"].apply(parse_dms)
        df["plat"] = [c[0] for c in coords]
        df["plon"] = [c[1] for c in coords]
        return df[["dim_driver_id", "plat", "plon"]].dropna()

    with st.spinner("Parsing pickup coordinates..."):
        good_c = _parse_pickup_coords(good_flow_raw)
        bad_c  = _parse_pickup_coords(bad_flow_raw)

    good_c["Group"] = "Top 10"
    bad_c["Group"]  = "Comparison"
    all_c = pd.concat([good_c, bad_c], ignore_index=True)
    all_c = all_c[
        all_c["plat"].between(51.3, 51.7) &
        all_c["plon"].between(-0.55, 0.3)
    ]

    # Per-driver median longitude
    def _driver_lon_stats(flow_raw, driver_ids, group_label):
        df = flow_raw[flow_raw["status"].isin(["completed","Finished"])].copy()
        rows = []
        for did in driver_ids:
            sub = df[df["dim_driver_id"] == did].head(500)
            coords = sub["pickup_lat_long"].apply(parse_dms)
            lons = [c[1] for c in coords if c[1] is not None and -0.55 < c[1] < 0.3]
            if lons:
                rows.append({
                    "Driver":    DRIVER_NAMES.get(did, str(did)),
                    "Median longitude": round(np.median(lons), 4),
                    "West %":    round(sum(1 for l in lons if l < _WEST_LON) / len(lons) * 100, 1),
                    "Group":     group_label,
                })
        return pd.DataFrame(rows)

    with st.spinner("Computing positioning stats per driver..."):
        good_lons = _driver_lon_stats(good_flow_raw, _GOOD_IDS, "Top 10")
        bad_lons  = _driver_lon_stats(bad_flow_raw,  _BAD_IDS,  "Comparison")

    lon_df = pd.concat([good_lons, bad_lons], ignore_index=True).sort_values("Median longitude")

    # ── Map: pickup density good vs bad ──────────────────────────────────────
    st.markdown("**Pickup location map** — green = top 10 drivers, red = comparison drivers")

    m_pos = folium.Map(location=[51.505, -0.13], zoom_start=11, tiles="CartoDB dark_matter")
    folium.GeoJson(
        GEOJSON_DATA,
        style_function=lambda f: {"fillColor": "#ffffff", "fillOpacity": 0.03,
                                   "color": "#555", "weight": 1}
    ).add_to(m_pos)

    # Vertical reference line at west boundary (add as a note on the map via a marker)
    folium.Marker(
        [51.508, _WEST_LON],
        icon=folium.DivIcon(
            html='<div style="color:#facc15;font-size:11px;white-space:nowrap;'
                 'font-weight:bold;">← West | East →</div>',
            icon_size=(100, 20), icon_anchor=(50, 10),
        ),
        tooltip="Charing Cross longitude — high demand to the west"
    ).add_to(m_pos)

    for _, row in good_c.iterrows():
        folium.CircleMarker(
            [row["plat"], row["plon"]], radius=3,
            color="#22c55e", fill=True, fill_opacity=0.35, weight=0,
        ).add_to(m_pos)

    for _, row in bad_c.iterrows():
        folium.CircleMarker(
            [row["plat"], row["plon"]], radius=3,
            color="#ef4444", fill=True, fill_opacity=0.35, weight=0,
        ).add_to(m_pos)

    st_folium(m_pos, width="100%", height=460, returned_objects=[])

    # ── Per-driver longitude chart ────────────────────────────────────────────
    st.markdown("**Median pickup longitude per driver** — further left (more negative) = further west = better")
    fig_lon = px.bar(
        lon_df, x="Median longitude", y="Driver", orientation="h",
        color="Group",
        color_discrete_map={"Top 10": "#22c55e", "Comparison": "#ef4444"},
        text="Median longitude",
        title="Where each driver operates — median pickup longitude",
    )
    fig_lon.add_vline(x=_WEST_LON, line_dash="dash", line_color="#facc15",
                      annotation_text="West boundary (-0.12°)", annotation_position="top")
    fig_lon.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_lon.update_layout(height=420, xaxis_title="Longitude (more negative = further west)")
    st.plotly_chart(fig_lon, use_container_width=True)

    # ── West % metric ─────────────────────────────────────────────────────────
    good_west_pct = good_lons["West %"].mean() if not good_lons.empty else 0
    bad_west_pct  = bad_lons["West %"].mean()  if not bad_lons.empty  else 0

    cw1, cw2, cw3 = st.columns(3)
    cw1.metric("Top 10 — pickups in west London",
               f"{good_west_pct:.1f}%", "West of -0.12°")
    cw2.metric("Comparison — pickups in west London",
               f"{bad_west_pct:.1f}%",
               delta=f"{bad_west_pct - good_west_pct:+.1f}%")
    cw3.metric("West % gap",
               f"{abs(good_west_pct - bad_west_pct):.1f}pp",
               "Percentage point difference")

    # ── West % per driver bar chart ───────────────────────────────────────────
    fig_west = px.bar(
        lon_df.sort_values("West %", ascending=True),
        x="West %", y="Driver", orientation="h",
        color="Group",
        color_discrete_map={"Top 10": "#22c55e", "Comparison": "#ef4444"},
        text="West %",
        title="% of pickups in west London (west of Charing Cross line)",
    )
    fig_west.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                       annotation_text="50% threshold")
    fig_west.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_west.update_layout(height=420)
    st.plotly_chart(fig_west, use_container_width=True)

    # ── Narrative ─────────────────────────────────────────────────────────────
    west_gap = good_west_pct - bad_west_pct
    most_western_bad  = bad_lons.loc[bad_lons["Median longitude"].idxmin(),  "Driver"] if not bad_lons.empty  else "—"
    most_eastern_good = good_lons.loc[good_lons["Median longitude"].idxmax(), "Driver"] if not good_lons.empty else "—"

    st.markdown(
        f'<div style="background:#1e1e2e;border-left:4px solid #facc15;'
        f'padding:14px 16px;border-radius:6px;color:#e2e8f0;margin-top:8px;">'
        f'<strong>📍 The positioning gap is {west_gap:.1f} percentage points.</strong> '
        f'Top 10 drivers make <strong>{good_west_pct:.1f}%</strong> of their pickups west of the '
        f'Charing Cross line vs <strong>{bad_west_pct:.1f}%</strong> for comparison drivers. '
        f'This single metric explains the lower ping density during gaps — comparison drivers '
        f'are in areas where demand is structurally lower. It\'s not that they\'re unlucky; '
        f'they\'re in the wrong part of the city.<br><br>'
        f'The west corridor (Mayfair, Kensington, Chelsea, Knightsbridge, Notting Hill, '
        f'Hammersmith) generates higher-value pings and more of them. '
        f'A driver sitting idle in Hackney or Stratford will wait longer and earn less '
        f'than one sitting idle in South Kensington — not because of decisions made during the gap, '
        f'but because of where the previous trip left them.<br><br>'
        f'<strong>The fix isn\'t just "decline bad trips" — it\'s deliberately positioning westward. '
        f'That means declining trips heading east even when the fare looks acceptable, '
        f'because the positioning cost on the other end is higher than the fare gained.</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 7: What the data is telling us ────────────────────────────────
    st.divider()
    st.subheader("7 — What the data is telling us")

    rph_gap      = ga["RPH"] - ba["RPH"]
    fare_gap     = good_fares.median() - bad_fares.median()
    accept_gap   = ba["Acceptance %"] - ga["Acceptance %"]
    sub10_gap    = sub10_bad - sub10_good

    # Zone-level findings from diff matrix
    z1z1_good = good_mat.loc[1, 1] if 1 in good_mat.index and 1 in good_mat.columns else 0
    z1z1_bad  = bad_mat.loc[1, 1]  if 1 in bad_mat.index  and 1 in bad_mat.columns  else 0
    z1_out_good = sum(good_mat.loc[1, z] for z in range(2, 7)
                      if z in good_mat.columns) if 1 in good_mat.index else 0
    z1_out_bad  = sum(bad_mat.loc[1, z]  for z in range(2, 7)
                      if z in bad_mat.columns)  if 1 in bad_mat.index  else 0

    # Peak hour comparison
    good_peak = good_hr.loc[good_hr["trips"].idxmax(), "trips_hr"] if not good_hr.empty else "?"
    bad_peak  = bad_hr.loc[bad_hr["trips"].idxmax(),  "trips_hr"]  if not bad_hr.empty  else "?"

    findings = []

    # RPH gap
    findings.append(
        f"**The RPH gap is £{rph_gap:.2f}/hr.** "
        f"Top 10 average £{ga['RPH']:.2f}/hr vs comparison drivers at £{ba['RPH']:.2f}/hr. "
        f"Over a 9-hour shift that's £{rph_gap*9:.0f} less per day — "
        f"roughly £{rph_gap*9*5:.0f}/week per driver."
    )

    # Acceptance rate
    if accept_gap > 5:
        findings.append(
            f"**Comparison drivers accept {accept_gap:.1f}% more pings.** "
            f"Higher acceptance isn't a virtue here — it means they're taking trips the top 10 "
            f"are deliberately passing on. That selectivity gap is a large part of the RPH difference."
        )
    elif accept_gap < -5:
        findings.append(
            f"**Comparison drivers actually accept {abs(accept_gap):.1f}% fewer pings** than the top 10 — "
            f"but their RPH is still lower, which suggests they're declining the wrong trips "
            f"(possibly good-value ones) rather than the right ones (short low-fare hops)."
        )
    else:
        findings.append(
            f"**Acceptance rates are similar** ({ga['Acceptance %']:.1f}% vs {ba['Acceptance %']:.1f}%) — "
            f"the RPH gap isn't about volume of pings accepted. It's about which ones they take."
        )

    # Fare gap
    if fare_gap > 2:
        findings.append(
            f"**Median fare is £{fare_gap:.2f} lower per trip for comparison drivers** "
            f"(£{bad_fares.median():.2f} vs £{good_fares.median():.2f}). "
            f"Across {int(ba['Total rides']):,} rides that compounds fast. "
            + (f"Comparison drivers take {sub10_gap:+.1f}% more sub-£10 trips — "
               f"these are the short Zone 1 hops that kill your hourly rate."
               if sub10_gap > 3 else
               f"The fare gap is spread across the distribution, not just in the sub-£10 bracket.")
        )
    else:
        findings.append(
            f"**Fare per trip is surprisingly similar** (median £{good_fares.median():.2f} vs £{bad_fares.median():.2f}). "
            f"The RPH gap isn't coming from trip size — it's more likely gap time between trips "
            f"or working the wrong hours."
        )

    # Zone flow
    if z1z1_bad - z1z1_good > 3:
        findings.append(
            f"**Zone 1→Zone 1 short hops: comparison drivers take {z1z1_bad:.1f}% of trips on this route "
            f"vs {z1z1_good:.1f}% for the top 10.** This is the clearest behavioural difference. "
            f"Z1→Z1 runs a full-cycle RPH of ~£20/hr — below the fleet average. "
            f"The top 10 pass on these and wait for better pings."
        )
    elif z1_out_good - z1_out_bad > 3:
        findings.append(
            f"**Top 10 drivers take significantly more Zone 1 outbound trips** "
            f"({z1_out_good:.1f}% vs {z1_out_bad:.1f}% of all trips). "
            f"These are the higher-value runs — longer distance, better fare, "
            f"and they often come with a strong return ping at the destination."
        )

    # Hour of day
    if good_peak != bad_peak:
        findings.append(
            f"**Peak hour differs: top 10 peak at {int(good_peak):02d}:00, "
            f"comparison drivers peak at {int(bad_peak):02d}:00.** "
            + ("Comparison drivers are more active in the midday lull when fares are shorter "
               "and ping density is lower."
               if int(bad_peak) in range(10, 16) and int(good_peak) not in range(10, 16)
               else
               "Check whether comparison drivers are missing the evening surge window "
               "that top 10 drivers consistently exploit.")
        )

    # Render findings as cards
    for i, finding in enumerate(findings):
        icon = ["💰", "🎯", "🧾", "🗺️", "🕐"][i % 5]
        st.markdown(
            f'<div style="background:#1e1e2e;border-left:4px solid #6366f1;'
            f'padding:12px 16px;border-radius:6px;margin-bottom:10px;color:#e2e8f0;">'
            f'<span style="font-size:18px;">{icon}</span> {finding}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # One-line summary
    st.markdown("---")
    primary_cause = (
        "taking too many short Zone 1→Zone 1 hops" if z1z1_bad - z1z1_good > 3
        else "accepting lower-fare trips the top 10 decline" if fare_gap > 2
        else "lower selectivity across the board — higher acceptance, lower average return"
    )
    st.success(
        f"**Bottom line:** The £{rph_gap:.2f}/hr gap is primarily driven by **{primary_cause}**. "
        f"The top 10 aren't working harder — they're working the same hours but saying no "
        f"more often to the trips that drag your average down."
    )

# ── Fleet Map ────────────────────────────────────────────────────────────────
elif page == "Fleet Map":
    st.title("Fleet Positioning Map")
    st.caption(
        "Every active driver's pickup locations over the last 30 days, coloured by 2026 performance category from the daily dashboard."
    )

    _WEST_LON = -0.12

    # ── Load category classifications from Excel export ────────────────────────
    _CAT_PATH = os.path.join(os.path.dirname(__file__), "driver_categories.csv")
    _cat_df   = pd.read_csv(_CAT_PATH)
    _cat_map  = dict(zip(_cat_df["dim_driver_id"], _cat_df["category"]))

    CAT_COLOR = {
        "A":  "#22c55e",   # green  — top tier
        "B1": "#4ade80",   # light green
        "B2": "#60a5fa",   # blue
        "C1": "#f59e0b",   # amber
        "C2": "#fb923c",   # orange
        "D":  "#ef4444",   # red    — lowest tier
        None: "#94a3b8",   # grey   — no classification
    }
    CAT_LABEL = {
        "A": "A — Elite", "B1": "B1 — Strong", "B2": "B2 — Solid",
        "C1": "C1 — Developing", "C2": "C2 — Below avg", "D": "D — Low performer",
        None: "Unclassified",
    }
    # Render order: unclassified first (bottom), then D→A (A on top)
    CAT_ORDER = [None, "D", "C2", "C1", "B2", "B1", "A"]
    CAT_OPACITY = {None: 0.25, "D": 0.50, "C2": 0.50, "C1": 0.50, "B2": 0.55, "B1": 0.60, "A": 0.70}
    CAT_RADIUS  = {None: 2,    "D": 3,    "C2": 3,    "C1": 3,    "B2": 3,    "B1": 3,    "A": 4}

    with st.spinner("Loading fleet pickup data — this may take a moment..."):
        raw_all = db.load_all_driver_coords(sample_per_driver=60, days_back=30)

    if raw_all.empty:
        st.warning("No data returned.")
        st.stop()

    raw_all["category"] = raw_all["dim_driver_id"].map(_cat_map)
    raw_all["category"] = raw_all["category"].where(
        raw_all["category"].isin(["A","B1","B2","C1","C2","D"]), other=None
    )

    with st.spinner(f"Parsing coordinates for {raw_all['dim_driver_id'].nunique()} drivers..."):
        coords = raw_all["pickup_lat_long"].apply(parse_dms)
        raw_all["plat"] = [c[0] for c in coords]
        raw_all["plon"] = [c[1] for c in coords]

    fleet = raw_all.dropna(subset=["plat","plon"])
    fleet = fleet[fleet["plat"].between(51.3, 51.7) & fleet["plon"].between(-0.55, 0.3)]

    n_drivers = fleet["dim_driver_id"].nunique()
    n_points  = len(fleet)

    # Category driver counts for metrics
    cat_counts = fleet.groupby("category", dropna=False)["dim_driver_id"].nunique()
    _mc = st.columns(7)
    for i, cat in enumerate(["A","B1","B2","C1","C2","D",None]):
        _mc[i].metric(
            CAT_LABEL[cat].split(" — ")[0],
            cat_counts.get(cat, 0),
            delta=None,
            help=CAT_LABEL[cat],
        )

    # ── Map ───────────────────────────────────────────────────────────────────
    m = folium.Map(location=[51.505, -0.13], zoom_start=11, tiles="CartoDB dark_matter")

    folium.GeoJson(
        GEOJSON_DATA,
        style_function=lambda f: {
            "fillColor": "#ffffff", "fillOpacity": 0.03,
            "color": "#555555", "weight": 1,
        }
    ).add_to(m)

    folium.Marker(
        [51.51, _WEST_LON],
        icon=folium.DivIcon(
            html='<div style="color:#facc15;font-size:11px;white-space:nowrap;'
                 'font-weight:bold;text-shadow:0 0 4px #000;">← West | East →</div>',
            icon_size=(110, 20), icon_anchor=(55, 10),
        ),
        tooltip="High-demand corridor boundary"
    ).add_to(m)

    for cat in CAT_ORDER:
        subset  = fleet[fleet["category"] == cat] if cat is not None else fleet[fleet["category"].isna()]
        colour  = CAT_COLOR[cat]
        opacity = CAT_OPACITY[cat]
        radius  = CAT_RADIUS[cat]
        for _, row in subset.iterrows():
            folium.CircleMarker(
                [row["plat"], row["plon"]],
                radius=radius,
                color=colour, fill=True, fill_color=colour,
                fill_opacity=opacity, weight=0,
                tooltip=f"{row['driver_full_name']} · {CAT_LABEL[cat]}",
            ).add_to(m)

    st_folium(m, width="100%", height=580, returned_objects=[])

    # ── Legend ────────────────────────────────────────────────────────────────
    _legend_items = "".join(
        f'<span style="margin-right:20px;">'
        f'<span style="color:{CAT_COLOR[c]};font-size:16px;">●</span> {CAT_LABEL[c]}'
        f'</span>'
        for c in ["A","B1","B2","C1","C2","D",None]
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;">{_legend_items}'
        f'<span style="margin-left:12px;color:#94a3b8;font-size:13px;">┃</span>'
        f'<span style="color:#facc15;font-size:13px;margin-left:8px;">━━ West boundary (-0.12°)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{n_points:,} pickup points plotted across {n_drivers} drivers · last 30 days · up to 60 pickups per driver")

    # ── West % by category ────────────────────────────────────────────────────
    st.divider()
    st.subheader("West London pickup % by category")
    st.caption("% of each driver's pickups that fall west of -0.12° longitude (Charing Cross line)")

    west_rows = []
    for _, grp in fleet.groupby("dim_driver_id"):
        did  = grp["dim_driver_id"].iloc[0]
        cat  = grp["category"].iloc[0] if pd.notna(grp["category"].iloc[0]) else None
        lons = grp["plon"].dropna()
        west_pct = (lons < _WEST_LON).mean() * 100
        west_rows.append({
            "Driver":   DRIVER_NAMES.get(did, grp["driver_full_name"].iloc[0]),
            "Category": cat if cat else "—",
            "Label":    CAT_LABEL[cat],
            "West %":   round(west_pct, 1),
            "Pickups":  len(lons),
        })

    west_df = pd.DataFrame(west_rows).sort_values(["Category","West %"], ascending=[True, False])

    # Summary table by category
    grp_summary = (
        west_df.groupby("Category")["West %"]
        .agg(["mean","median","min","max"])
        .round(1)
        .reset_index()
        .rename(columns={"mean":"Avg %","median":"Median %","min":"Min %","max":"Max %"})
    )
    # Sort by category order
    _cat_order_str = ["A","B1","B2","C1","C2","D","—"]
    grp_summary["_sort"] = grp_summary["Category"].map({c:i for i,c in enumerate(_cat_order_str)})
    grp_summary = grp_summary.sort_values("_sort").drop(columns="_sort")
    st.dataframe(grp_summary, use_container_width=True, hide_index=True)

    # Bar chart — sorted by west % within each category
    west_df_sorted = west_df.sort_values(["Category","West %"], ascending=[True,False])
    _disc_map = {cat: CAT_COLOR[cat if cat != "—" else None] for cat in _cat_order_str}
    fig_all = px.bar(
        west_df_sorted, x="West %", y="Driver", orientation="h",
        color="Category",
        color_discrete_map=_disc_map,
        text="West %",
        title="West London pickup % — all drivers (grouped by category)",
        height=max(500, len(west_df_sorted) * 22 + 80),
    )
    fig_all.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                      annotation_text="50%", annotation_position="top")
    fig_all.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_all.update_layout(xaxis_title="% of pickups west of -0.12°", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Narrative ─────────────────────────────────────────────────────────────
    _a_avg  = west_df[west_df["Category"] == "A"]["West %"].mean() if "A" in west_df["Category"].values else 0
    _d_avg  = west_df[west_df["Category"] == "D"]["West %"].mean() if "D" in west_df["Category"].values else 0
    _all_avg = west_df["West %"].mean()

    st.markdown(
        f'<div style="background:#1e1e2e;border-left:4px solid #22c55e;'
        f'padding:14px 16px;border-radius:6px;color:#e2e8f0;margin-top:8px;">'
        f'<strong>📍 Fleet positioning by category:</strong><br><br>'
        f'<span style="color:#22c55e;">●</span> <strong>Cat A avg: {_a_avg:.1f}%</strong> west — elite drivers cluster in the high-demand corridor<br>'
        f'<span style="color:#ef4444;">●</span> <strong>Cat D avg: {_d_avg:.1f}%</strong> west — '
        f'<strong>{_a_avg - _d_avg:.1f}pp below Cat A</strong><br>'
        f'<span style="color:#94a3b8;">●</span> Fleet overall: <strong>{_all_avg:.1f}%</strong><br><br>'
        f'Positioning west of -0.12° correlates strongly with higher ping density and shorter dead miles. '
        f'Cat D drivers consistently drift east/northeast where demand is lower — this compounds '
        f'over a shift into materially worse RPH before accounting for trip selection.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Driver Day ───────────────────────────────────────────────────────────────
elif page == "Driver Day":
    st.title("Driver Day")
    st.caption("Pick any driver and a date — see every trip they took, every ping they dropped, and how their earnings built through the shift.")

    # ── Driver search ─────────────────────────────────────────────────────────
    col_srch, col_date = st.columns([2, 1])
    with col_srch:
        _dd_search = st.text_input(
            "Search driver by name",
            placeholder="e.g. Yousuf, Mukhtar, Monier...",
            key="dd_search",
        )

    if not _dd_search or len(_dd_search) < 2:
        st.info("Type at least 2 characters to search for a driver.")
        st.stop()

    with st.spinner("Searching..."):
        _dd_results = db.search_drivers(_dd_search)

    if _dd_results.empty:
        st.warning(f"No drivers found matching '{_dd_search}'.")
        st.stop()

    _dd_opts = {f"{row.driver_full_name} (ID {row.dim_driver_id})": int(row.dim_driver_id)
                for _, row in _dd_results.iterrows()}

    with col_srch:
        _dd_label  = st.selectbox("Select driver", options=list(_dd_opts.keys()), key="dd_select")
        driver_id  = _dd_opts[_dd_label]

    # ── Load trips for selected driver ────────────────────────────────────────
    with st.spinner(f"Loading trips for {_dd_label}..."):
        raw = db.load_any_driver_trips(driver_id)

    if raw.empty:
        st.warning("No completed trips found for this driver.")
        st.stop()

    enriched = enrich_zones(raw)
    enriched = calc_true_rph(enriched)
    enriched["pickedup_trip_datetime"] = pd.to_datetime(enriched["pickedup_trip_datetime"])
    enriched["dropoff_trip_datetime"]  = pd.to_datetime(enriched["dropoff_trip_datetime"])
    enriched["trip_date"] = enriched["pickedup_trip_datetime"].dt.date

    driver_trips    = enriched.copy()
    available_dates = sorted(driver_trips["trip_date"].dropna().unique(), reverse=True)

    with col_date:
        selected_date = st.selectbox("Date", options=[str(d) for d in available_dates])

    day_trips = (
        driver_trips[driver_trips["trip_date"] == pd.Timestamp(selected_date).date()]
        .sort_values("pickedup_trip_datetime")
        .reset_index(drop=True)
    )
    day_trips["trip_num"] = range(1, len(day_trips) + 1)

    if day_trips.empty:
        st.warning("No completed trips found for this driver on this date.")
        st.stop()

    # ── Load declined pings ───────────────────────────────────────────────────
    with st.spinner("Loading declined pings..."):
        declined_raw = db.load_driver_declined_day(driver_id, selected_date)

    if not declined_raw.empty:
        dec_coords = declined_raw["pickup_lat_long"].apply(parse_dms)
        declined_raw["plat"] = [c[0] for c in dec_coords]
        declined_raw["plon"] = [c[1] for c in dec_coords]
        declined_day = declined_raw.dropna(subset=["plat", "plon"])
    else:
        declined_day = pd.DataFrame()

    n_declined = len(declined_day)

    # ── Stats strip ───────────────────────────────────────────────────────────
    total_fare = day_trips["trip_price_in_pound"].sum()
    n_trips    = len(day_trips)
    day_rph    = day_trips["true_rph"].replace([np.inf, -np.inf], np.nan).median()
    accept_pct = round(n_trips / max(n_trips + n_declined, 1) * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trips taken",     n_trips)
    c2.metric("Pings dropped",   n_declined)
    c3.metric("Day earnings",    f"£{total_fare:.2f}")
    c4.metric("Acceptance (day)",f"{accept_pct}%",  f"Fleet avg 57%")

    # ── Map ───────────────────────────────────────────────────────────────────
    st.subheader("Trip map")
    st.caption("Numbered markers = trips in order. Red dots = declined pings. Hover for details.")

    map_center = [
        day_trips["pickup_lat"].dropna().iloc[0] if day_trips["pickup_lat"].notna().any() else CENTER_LAT,
        day_trips["pickup_lon"].dropna().iloc[0] if day_trips["pickup_lon"].notna().any() else CENTER_LON,
    ]
    m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB dark_matter")

    folium.GeoJson(
        GEOJSON_DATA,
        style_function=lambda f: {
            "fillColor": "#ffffff", "fillOpacity": 0.03,
            "color": "#666666", "weight": 1,
        }
    ).add_to(m)

    TRIP_PALETTE = [
        "#60a5fa","#34d399","#f59e0b","#a78bfa","#f87171",
        "#38bdf8","#4ade80","#fb923c","#c084fc","#fb7185",
    ]

    for _, row in day_trips.iterrows():
        if pd.isna(row.get("pickup_lat")) or pd.isna(row.get("dropoff_lat")):
            continue
        color = TRIP_PALETTE[(row["trip_num"] - 1) % len(TRIP_PALETTE)]
        pz    = int(row["pickup_zone"])  if pd.notna(row.get("pickup_zone"))  else "?"
        dz    = int(row["dropoff_zone"]) if pd.notna(row.get("dropoff_zone")) else "?"
        tip   = (f"Trip {row['trip_num']} | {row['pickedup_trip_datetime'].strftime('%H:%M')} | "
                 f"£{row['trip_price_in_pound']:.2f} | {row['distance_in_miles']:.1f}mi | Z{pz}→Z{dz}")

        folium.PolyLine(
            [(row["pickup_lat"], row["pickup_lon"]),
             (row["dropoff_lat"], row["dropoff_lon"])],
            color=color, weight=3, opacity=0.85, tooltip=tip,
        ).add_to(m)

        folium.Marker(
            [row["pickup_lat"], row["pickup_lon"]],
            icon=folium.DivIcon(
                html=(f'<div style="background:{color};color:#111;border-radius:50%;'
                      f'width:20px;height:20px;display:flex;align-items:center;'
                      f'justify-content:center;font-weight:700;font-size:10px;">'
                      f'{row["trip_num"]}</div>'),
                icon_size=(20, 20), icon_anchor=(10, 10),
            ),
            tooltip=tip,
        ).add_to(m)

        folium.CircleMarker(
            [row["dropoff_lat"], row["dropoff_lon"]],
            radius=4, color=color, fill=True, fill_opacity=0.5,
            tooltip=f"Dropoff {row['trip_num']}",
        ).add_to(m)

    for _, row in declined_day.iterrows():
        folium.CircleMarker(
            [row["plat"], row["plon"]],
            radius=5, color="#ef4444", fill=True, fill_color="#ef4444",
            fill_opacity=0.75, weight=1,
            tooltip="Declined ping",
        ).add_to(m)

    st_folium(m, width="100%", height=520, returned_objects=[])

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.subheader("Timeline")

    tl_rows = []
    for _, row in day_trips.iterrows():
        pz = int(row["pickup_zone"])  if pd.notna(row.get("pickup_zone"))  else 0
        dz = int(row["dropoff_zone"]) if pd.notna(row.get("dropoff_zone")) else 0
        tl_rows.append({
            "Trip": f"Trip {row['trip_num']:02d}  Z{pz}→Z{dz}",
            "Start": row["pickedup_trip_datetime"],
            "End":   row["dropoff_trip_datetime"],
            "Fare £": row["trip_price_in_pound"],
            "RPH":    round(row["true_rph"], 2) if pd.notna(row.get("true_rph")) else None,
            "Platform": row.get("source", ""),
        })

    tl_df = pd.DataFrame(tl_rows)
    fig_tl = px.timeline(
        tl_df, x_start="Start", x_end="End", y="Trip",
        color="Fare £", color_continuous_scale="Greens",
        hover_data=["Fare £", "RPH", "Platform"],
        title=f"{driver_label} — {selected_date}",
    )
    fig_tl.update_yaxes(autorange="reversed")
    fig_tl.update_layout(height=max(320, n_trips * 32 + 80))
    st.plotly_chart(fig_tl, use_container_width=True)

    # ── Earnings & RPH curve ──────────────────────────────────────────────────
    st.subheader("Earnings curve")

    day_trips["cumulative"] = day_trips["trip_price_in_pound"].cumsum()

    fig_earn = go.Figure()
    fig_earn.add_trace(go.Scatter(
        x=day_trips["pickedup_trip_datetime"],
        y=day_trips["cumulative"],
        mode="lines+markers",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=7),
        name="Cumulative earnings",
        hovertemplate="<b>%{x|%H:%M}</b><br>Total: £%{y:.2f}<extra></extra>",
    ))
    fig_earn.add_trace(go.Scatter(
        x=day_trips["pickedup_trip_datetime"],
        y=day_trips["true_rph"],
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(size=6),
        name="Trip RPH",
        yaxis="y2",
        hovertemplate="<b>%{x|%H:%M}</b><br>RPH: £%{y:.2f}/hr<extra></extra>",
    ))
    fig_earn.add_hline(y=22.9, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                       annotation_text="Fleet avg £22.9/hr", yref="y2",
                       annotation_position="bottom right")
    fig_earn.update_layout(
        height=320,
        yaxis=dict(title="Cumulative £", side="left"),
        yaxis2=dict(title="Trip RPH £/hr", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    st.plotly_chart(fig_earn, use_container_width=True)

    # ── Trip table ────────────────────────────────────────────────────────────
    with st.expander("Full trip table"):
        disp = day_trips[[
            "trip_num","pickedup_trip_datetime","trip_price_in_pound",
            "distance_in_miles","pob_duration_in_min","pickup_duration_in_min",
            "pickup_zone","dropoff_zone","true_rph","source",
        ]].copy()
        disp.columns = ["#","Time","Fare £","Miles","Ride min","Pickup min",
                        "From Z","To Z","RPH","Platform"]
        disp["Time"]    = disp["Time"].dt.strftime("%H:%M")
        disp["From Z"]  = disp["From Z"].astype(int)
        disp["To Z"]    = disp["To Z"].astype(int)
        st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Gap Analysis ─────────────────────────────────────────────────────────────
elif page == "Gap Analysis":
    _GAP_GOOD  = (128, 81, 130, 180, 155)
    _GAP_BAD   = (82, 178, 72, 36, 32)
    _GAP_IDS   = _GAP_GOOD + _GAP_BAD
    _GAP_NAMES = {
        128: "Monier Janabi",
        81:  "Marius Norvaisas",
        130: "Jermaine Gyamfi",
        180: "Abdi Mohamed",
        155: "Ertac Cindoglu",
        82:  "Aaron Bartley",
        178: "Abdullahi Saleh",
        72:  "Ponki Miah",
        36:  "Angeline Lewis",
        32:  "Emran Uddin",
    }
    # Heathrow bounding box — dropoffs here are queue-related, not decision problems
    _LHR_LAT = (51.45, 51.49)
    _LHR_LON = (-0.50, -0.42)
    _GAP_MINS  = 25           # minimum gap to investigate (below = normal inter-trip wait)
    _GAP_MAX   = 75           # above this = driver likely on break, app off
    _GAP_ZONE  = 3            # minimum dropoff zone to count as "stranded"

    st.title("Gap Analysis")
    st.caption(
        "When a driver ends up stranded in an outer zone and sits for 25+ minutes — "
        "what pings came in during that wait? Did they have a good option they passed on, "
        "or was there genuinely nothing worth taking?"
    )
    st.info(
        f"5 top performers + 5 comparison drivers · last 14 days · "
        f"gaps {_GAP_MINS}–{_GAP_MAX} min after a Zone {_GAP_ZONE}+ dropoff · "
        f"gaps over {_GAP_MAX} min excluded (assumed break)"
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading accepted trips..."):
        acc_raw = db.load_gap_accepted(_GAP_IDS, days_back=14)
    with st.spinner("Loading declined pings..."):
        dec_raw = db.load_gap_declined(_GAP_IDS, days_back=14)

    # Parse & enrich accepted
    if acc_raw.empty:
        st.warning("No accepted trips found in the last 14 days for these drivers.")
        st.stop()

    acc = enrich_zones(acc_raw)
    acc = calc_true_rph(acc)
    acc["pickedup_trip_datetime"] = pd.to_datetime(acc["pickedup_trip_datetime"])
    acc["dropoff_trip_datetime"]  = pd.to_datetime(acc["dropoff_trip_datetime"])
    acc["pickup_zone"]  = acc["pickup_zone"].astype("Int64")
    acc["dropoff_zone"] = acc["dropoff_zone"].astype("Int64")

    # Parse declined coords & zones
    if not dec_raw.empty:
        dec_raw["trip_booking_datetime"] = pd.to_datetime(dec_raw["trip_booking_datetime"])
        _dp = dec_raw["pickup_lat_long"].apply(parse_dms)
        _dd = dec_raw["dropoff_latlong"].apply(parse_dms)
        dec_raw["plat"] = [c[0] for c in _dp]; dec_raw["plon"] = [c[1] for c in _dp]
        dec_raw["dlat"] = [c[0] for c in _dd]; dec_raw["dlon"] = [c[1] for c in _dd]
        dec_raw["pickup_zone"]  = dec_raw.apply(lambda r: assign_zone(r.plat,  r.plon), axis=1)
        dec_raw["dropoff_zone"] = dec_raw.apply(lambda r: assign_zone(r.dlat,  r.dlon), axis=1)
        dec_raw = dec_raw.dropna(subset=["plat", "plon"])
    else:
        dec_raw = pd.DataFrame()

    # ── Gap finder ────────────────────────────────────────────────────────────
    def _find_gaps(driver_id):
        drv = acc[acc["dim_driver_id"] == driver_id].sort_values("pickedup_trip_datetime").reset_index(drop=True)
        gaps = []
        for i in range(len(drv) - 1):
            curr      = drv.iloc[i]
            nxt       = drv.iloc[i + 1]
            dz        = curr["dropoff_zone"]
            if pd.isna(dz) or int(dz) < _GAP_ZONE:
                continue
            gap_mins  = (nxt["pickedup_trip_datetime"] - curr["dropoff_trip_datetime"]).total_seconds() / 60
            if gap_mins < _GAP_MINS or gap_mins > _GAP_MAX:
                continue
            # Declined pings that arrived during this gap
            if not dec_raw.empty:
                mask = (
                    (dec_raw["dim_driver_id"] == driver_id) &
                    (dec_raw["trip_booking_datetime"] >= curr["dropoff_trip_datetime"]) &
                    (dec_raw["trip_booking_datetime"] <= nxt["pickedup_trip_datetime"])
                )
                pings = dec_raw[mask].copy()
            else:
                pings = pd.DataFrame()
            # Detect if stranded at Heathrow (queue problem, not a decision problem)
            dlat = curr.get("dropoff_lat")
            dlon = curr.get("dropoff_lon")
            at_lhr = (
                pd.notna(dlat) and pd.notna(dlon) and
                _LHR_LAT[0] <= dlat <= _LHR_LAT[1] and
                _LHR_LON[0] <= dlon <= _LHR_LON[1]
            )
            gaps.append({
                "stranding":     curr,
                "rescue":        nxt,
                "gap_mins":      gap_mins,
                "stranded_zone": int(dz),
                "pings":         pings,
                "at_lhr":        at_lhr,
            })
        return gaps

    # ── Ping classifier ───────────────────────────────────────────────────────
    def _rate_rescue(rescue, gap_mins, stranded_zone):
        """Score the trip that ended the gap 0–10, return (score, stars, label, detail)."""
        pz   = int(rescue.get("pickup_zone")  or stranded_zone)
        dz   = int(rescue.get("dropoff_zone") or pz)
        fare = float(rescue.get("trip_price_in_pound") or 0)
        ride = float(rescue.get("pob_duration_in_min") or 20)

        # 1. Direction: how much did the zone improve?
        zone_delta = pz - dz           # positive = heading toward city
        if zone_delta >= 3:   dir_score = 4
        elif zone_delta >= 2: dir_score = 3
        elif zone_delta == 1: dir_score = 2
        elif zone_delta == 0: dir_score = 1
        else:                 dir_score = 0   # going further out

        # 2. Fare: was it worth pulling over for?
        if fare >= 30:   fare_score = 3
        elif fare >= 18: fare_score = 2
        elif fare >= 10: fare_score = 1
        else:            fare_score = 0

        # 3. Wait-adjusted RPH: fare / (gap + ride) — what did the patience cost?
        total_mins = gap_mins + ride
        wait_rph   = fare / (total_mins / 60) if total_mins > 0 else 0
        if wait_rph >= 22:   rph_score = 3
        elif wait_rph >= 15: rph_score = 2
        elif wait_rph >= 10: rph_score = 1
        else:                rph_score = 0

        score = dir_score + fare_score + rph_score   # 0–10

        if score >= 8:   stars, label = "★★★★★", "Worth the wait"
        elif score >= 6: stars, label = "★★★★☆", "Good recovery"
        elif score >= 4: stars, label = "★★★☆☆", "Decent"
        elif score >= 2: stars, label = "★★☆☆☆", "Questionable"
        else:            stars, label = "★☆☆☆☆", "Should've moved earlier"

        detail = (f"Z{pz}→Z{dz} · £{fare:.2f} · "
                  f"wait-adj RPH £{wait_rph:.0f}/hr · {score}/10")
        return score, stars, label, detail

    def _classify_ping(ping, stranded_zone):
        dz = ping.get("dropoff_zone")
        # Fare is null for declined trips — estimate from coords
        fare_db = ping.get("trip_price_in_pound") or 0
        if fare_db > 0:
            fare = fare_db
        else:
            _, fare = estimate_ping(
                ping.get("plat"), ping.get("plon"),
                ping.get("dlat"), ping.get("dlon"),
            )
            fare = fare or 0

        if pd.isna(dz):
            direction = "unknown"
        elif int(dz) < stranded_zone - 1:
            direction = "heading_in"
        elif int(dz) <= stranded_zone:
            direction = "lateral"
        else:
            direction = "going_further"

        if direction == "heading_in" and fare >= 10:
            verdict = "⚠️ Questionable — heading back, decent fare"
        elif direction == "heading_in" and fare < 10:
            verdict = "✓ Understandable — right direction but too cheap"
        elif direction == "lateral":
            verdict = "~ Borderline — stays in same area"
        elif direction == "going_further":
            verdict = "✓ Right call — going even further out"
        else:
            verdict = "? Unknown"
        return direction, verdict

    # ── Colour helpers ────────────────────────────────────────────────────────
    _DIR_COLOUR = {
        "heading_in":    "#f59e0b",   # amber — should question this decline
        "lateral":       "#94a3b8",   # grey
        "going_further": "#ef4444",   # red — clearly bad ping
        "unknown":       "#6b7280",
    }

    # ── Pre-compute all gaps ──────────────────────────────────────────────────
    all_gaps = {did: _find_gaps(did) for did in _GAP_IDS}
    good_gaps_all = [g for did in _GAP_GOOD for g in all_gaps[did]]
    bad_gaps_all  = [g for did in _GAP_BAD  for g in all_gaps[did]]

    def _group_stats(gaps):
        if not gaps:
            return {"total": 0, "lhr": 0, "decision": 0, "avg_mins": 0,
                    "pings": 0, "pings_per_gap": 0, "questionable": 0,
                    "no_ping_pct": 0, "avg_score": 0}
        n_lhr  = sum(1 for g in gaps if g["at_lhr"])
        n_dec  = len(gaps) - n_lhr
        dec_gaps = [g for g in gaps if not g["at_lhr"]]
        all_pings = [p for g in dec_gaps for _, p in g["pings"].iterrows()]
        q = sum(
            1 for g in dec_gaps for _, p in g["pings"].iterrows()
            if _classify_ping(p, g["stranded_zone"])[1].startswith("⚠️")
        )
        no_ping = sum(1 for g in dec_gaps if g["pings"].empty)
        scores  = [_rate_rescue(g["rescue"], g["gap_mins"], g["stranded_zone"])[0] for g in gaps]
        return {
            "total":         len(gaps),
            "lhr":           n_lhr,
            "decision":      n_dec,
            "avg_mins":      round(np.mean([g["gap_mins"] for g in gaps]), 1),
            "pings":         len(all_pings),
            "pings_per_gap": round(len(all_pings) / max(n_dec, 1), 1),
            "questionable":  q,
            "no_ping_pct":   round(no_ping / max(n_dec, 1) * 100, 1),
            "avg_score":     round(np.mean(scores), 1) if scores else 0,
        }

    gs = _group_stats(good_gaps_all)
    bs = _group_stats(bad_gaps_all)

    # ── Group comparison ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Group comparison — Top performers vs Comparison drivers")

    cg, cb = st.columns(2)
    with cg:
        st.markdown("### ✅ Top performers (5 drivers)")
        st.metric("Total gaps (25–75 min, Z3+)",  gs["total"])
        st.metric("✈️ Heathrow queue gaps",        gs["lhr"])
        st.metric("Active decision gaps",          gs["decision"])
        st.metric("Avg gap length",                f"{gs['avg_mins']} min")
        st.metric("Pings received during gaps",    gs["pings"])
        st.metric("Pings per gap",                 gs["pings_per_gap"])
        st.metric("Questionable declines",         gs["questionable"],
                  "Heading back, ≥£10 — passed on")
        st.metric("Gaps with zero pings",          f"{gs['no_ping_pct']}%",
                  "Genuinely stranded, no options")
        st.metric("Avg rescue trip score",         f"{gs['avg_score']}/10")

    with cb:
        st.markdown("### ⚠️ Comparison drivers (5 drivers)")
        st.metric("Total gaps (25–75 min, Z3+)",  bs["total"],
                  delta=f"{bs['total']-gs['total']:+d} vs top 10")
        st.metric("✈️ Heathrow queue gaps",        bs["lhr"],
                  delta=f"{bs['lhr']-gs['lhr']:+d}")
        st.metric("Active decision gaps",          bs["decision"],
                  delta=f"{bs['decision']-gs['decision']:+d}")
        st.metric("Avg gap length",                f"{bs['avg_mins']} min",
                  delta=f"{bs['avg_mins']-gs['avg_mins']:+.1f} min")
        st.metric("Pings received during gaps",    bs["pings"],
                  delta=f"{bs['pings']-gs['pings']:+d}")
        st.metric("Pings per gap",                 bs["pings_per_gap"],
                  delta=f"{bs['pings_per_gap']-gs['pings_per_gap']:+.1f}")
        st.metric("Questionable declines",         bs["questionable"],
                  delta=f"{bs['questionable']-gs['questionable']:+d}")
        st.metric("Gaps with zero pings",          f"{bs['no_ping_pct']}%",
                  delta=f"{bs['no_ping_pct']-gs['no_ping_pct']:+.1f}%")
        st.metric("Avg rescue trip score",         f"{bs['avg_score']}/10",
                  delta=f"{bs['avg_score']-gs['avg_score']:+.1f}")

    # Visual bar comparison
    comp_df = pd.DataFrame([
        {"Metric": "Avg gap (min)",        "Top 10": gs["avg_mins"],      "Comparison": bs["avg_mins"]},
        {"Metric": "Pings per gap",        "Top 10": gs["pings_per_gap"], "Comparison": bs["pings_per_gap"]},
        {"Metric": "Questionable declines","Top 10": gs["questionable"],  "Comparison": bs["questionable"]},
        {"Metric": "Zero-ping gaps %",     "Top 10": gs["no_ping_pct"],   "Comparison": bs["no_ping_pct"]},
        {"Metric": "Rescue score /10",     "Top 10": gs["avg_score"],     "Comparison": bs["avg_score"]},
    ])
    fig_cmp = px.bar(
        comp_df.melt(id_vars="Metric", var_name="Group", value_name="Value"),
        x="Metric", y="Value", color="Group", barmode="group",
        color_discrete_map={"Top 10": "#22c55e", "Comparison": "#ef4444"},
        text_auto=".1f", title="Gap behaviour — key metrics compared",
    )
    fig_cmp.update_layout(height=380)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Narrative
    st.markdown("#### What this tells us")
    findings_gap = []

    gap_diff = bs["avg_mins"] - gs["avg_mins"]
    if gap_diff > 5:
        findings_gap.append(
            f"**Comparison drivers sit stranded {gap_diff:.0f} minutes longer on average** "
            f"({bs['avg_mins']} min vs {gs['avg_mins']} min). That dead time compounds across a shift — "
            f"if it happens 3 times a day, that's {gap_diff*3:.0f} extra minutes of zero earnings daily."
        )
    elif gap_diff < -5:
        findings_gap.append(
            f"**Top performers actually have longer average gaps** ({gs['avg_mins']} min vs {bs['avg_mins']} min). "
            f"This likely reflects deliberate waiting for the right ping rather than thrashing — "
            f"check their rescue trip scores to confirm they're being rewarded for the patience."
        )

    ping_diff = bs["pings_per_gap"] - gs["pings_per_gap"]
    if ping_diff > 0.5:
        findings_gap.append(
            f"**Comparison drivers receive more pings per gap ({bs['pings_per_gap']} vs {gs['pings_per_gap']})** "
            f"but their rescue trip scores are {'lower' if bs['avg_score'] < gs['avg_score'] else 'similar'}. "
            f"They're getting offered trips — the question is whether they're taking the right ones."
        )
    elif ping_diff < -0.5:
        findings_gap.append(
            f"**Top performers receive more pings per gap ({gs['pings_per_gap']} vs {bs['pings_per_gap']})** — "
            f"they're in areas with higher ping density even when stranded. "
            f"Comparison drivers may be in genuinely dead locations (Zone 3 outer areas) "
            f"where demand is lower."
        )

    if bs["questionable"] > gs["questionable"]:
        findings_gap.append(
            f"**Comparison drivers have {bs['questionable']} questionable declines vs {gs['questionable']} for the top 10.** "
            f"These are pings heading back toward the city at ≥£10 that they passed on. "
            f"Top performers are better at spotting and taking the 'rescue ping' that gets them out of a bad zone."
        )

    no_ping_diff = bs["no_ping_pct"] - gs["no_ping_pct"]
    if no_ping_diff > 10:
        findings_gap.append(
            f"**{bs['no_ping_pct']}% of comparison driver gaps have zero pings** "
            f"vs {gs['no_ping_pct']}% for top performers. "
            f"This means comparison drivers are stranding themselves in areas with no demand — "
            f"not just making bad decisions, but ending up in the wrong places entirely."
        )

    score_diff = gs["avg_score"] - bs["avg_score"]
    if score_diff > 1:
        findings_gap.append(
            f"**Rescue trip quality: top performers score {gs['avg_score']}/10 vs {bs['avg_score']}/10.** "
            f"When the top 10 finally take a trip after a gap, it's a better trip — "
            f"higher fare, better direction, stronger wait-adjusted RPH. "
            f"Comparison drivers end the gap with whatever comes first."
        )

    for i, f in enumerate(findings_gap):
        icon = ["⏱️","📍","🎯","🗺️","⭐"][i % 5]
        st.markdown(
            f'<div style="background:#1e1e2e;border-left:4px solid #6366f1;'
            f'padding:12px 16px;border-radius:6px;margin-bottom:10px;color:#e2e8f0;">'
            f'<span style="font-size:16px;">{icon}</span> {f}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not findings_gap:
        st.info("Gap patterns are similar between the two groups — the difference may lie in zone positioning rather than gap decision-making. Check the Good vs Bad page for zone flow analysis.")

    # ── Outlier Spotlight ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Outlier Spotlight")
    st.caption(
        "Four drivers who break the west-positioning → performance rule. "
        "Three Cat A drivers succeeding with below-average west %, one Cat D driver failing despite 91% west pickup share."
    )

    _SPOT_IDS   = [219, 215, 180, 223]
    _SPOT_NAMES = {219: "Mohamed Yousuf", 215: "Mukhtar Abdullahi", 180: "Abdi Mohamed", 223: "Akeame Plummer"}
    _SPOT_CAT   = {219: "A", 215: "A", 180: "A", 223: "D"}
    _SPOT_WEST  = {219: 25.7, 215: 33.8, 180: 42.6, 223: 92.6}  # full 2026 dataset, pickup lon

    with st.spinner("Loading outlier trip data (30 days)..."):
        _spot_perf    = db.load_comparison_performance(_SPOT_IDS)
        _spot_acc_raw = db.load_gap_accepted(_SPOT_IDS, days_back=30)
        _spot_dec_raw = db.load_gap_declined(_SPOT_IDS, days_back=30)

    if not _spot_acc_raw.empty:
        _spot_acc = _spot_acc_raw.copy()
        _spot_acc["pickedup_trip_datetime"] = pd.to_datetime(_spot_acc["pickedup_trip_datetime"])
        _spot_acc["dropoff_trip_datetime"]  = pd.to_datetime(_spot_acc["dropoff_trip_datetime"])
        _spot_acc = _spot_acc.sort_values(["dim_driver_id", "pickedup_trip_datetime"])
        _spot_acc["prev_drop"] = _spot_acc.groupby("dim_driver_id")["dropoff_trip_datetime"].shift(1)
        _spot_acc["gap_mins"] = (
            (_spot_acc["pickedup_trip_datetime"] - _spot_acc["prev_drop"])
            .dt.total_seconds().div(60).clip(lower=0)
        )
        # Parse pickup coordinates for east/west analysis
        _sp_coords = _spot_acc["pickup_lat_long"].apply(parse_dms)
        _spot_acc["plat"] = [c[0] for c in _sp_coords]
        _spot_acc["plon"] = [c[1] for c in _sp_coords]
        _spot_acc = _spot_acc.dropna(subset=["plat","plon"])
        _spot_acc = _spot_acc[_spot_acc["plat"].between(51.3, 51.7) & _spot_acc["plon"].between(-0.55, 0.3)]
        _spot_acc["is_west"] = _spot_acc["plon"] < -0.12
        _spot_acc["hour"]    = pd.to_datetime(_spot_acc["pickedup_trip_datetime"]).dt.hour

        def _spot_gap_stats(df):
            g = df["gap_mins"].dropna()
            g = g[g > 0]
            if len(g) == 0:
                return {"median": 0, "short": 0, "gap": 0, "brk": 0}
            return {
                "median": g.median(),
                "short":  (g < 25).mean() * 100,
                "gap":    ((g >= 25) & (g <= 75)).mean() * 100,
                "brk":    (g > 75).mean() * 100,
            }

        # ── 4 driver cards ────────────────────────────────────────────────────
        _spot_cols = st.columns(4)
        for _si, _did in enumerate([219, 215, 180, 223]):
            _pr = _spot_perf[_spot_perf["dim_driver_id"] == _did]
            if _pr.empty:
                continue
            _p        = _pr.iloc[0]
            _dt       = _spot_acc[_spot_acc["dim_driver_id"] == _did]
            _gs       = _spot_gap_stats(_dt)
            _cat      = _SPOT_CAT[_did]
            _ccol     = "#22c55e" if _cat == "A" else "#ef4444"
            _avg_fare = _dt["trip_price_in_pound"].mean() if len(_dt) else 0
            _sub10    = (_dt["trip_price_in_pound"] < 10).mean() * 100 if len(_dt) else 0

            with _spot_cols[_si]:
                st.markdown(
                    f'<div style="background:#1e1e2e;border:2px solid {_ccol};border-radius:8px;padding:12px 14px;">'
                    f'<div style="color:{_ccol};font-size:11px;font-weight:bold;letter-spacing:1px;">CAT {_cat}</div>'
                    f'<div style="font-size:15px;font-weight:bold;color:#f8fafc;margin-top:2px;">{_SPOT_NAMES[_did]}</div>'
                    f'<div style="color:#94a3b8;font-size:12px;">West %: <strong style="color:#facc15">{_SPOT_WEST[_did]:.0f}%</strong></div>'
                    f'<hr style="border-color:#333;margin:8px 0">'
                    f'<table style="width:100%;font-size:12px;color:#e2e8f0;border-collapse:collapse;">'
                    f'<tr><td style="padding:2px 0">RPH</td><td style="text-align:right;font-weight:bold;color:{_ccol}">£{_p.rph:.2f}</td></tr>'
                    f'<tr><td>Acceptance</td><td style="text-align:right">{_p.acceptance:.0f}%</td></tr>'
                    f'<tr><td>Avg fare</td><td style="text-align:right">£{_avg_fare:.2f}</td></tr>'
                    f'<tr><td>Sub-£10</td><td style="text-align:right">{_sub10:.0f}%</td></tr>'
                    f'<tr><td style="padding-top:6px">Median gap</td><td style="text-align:right;padding-top:6px">{_gs["median"]:.0f} min</td></tr>'
                    f'<tr><td>&lt;25m (normal)</td><td style="text-align:right">{_gs["short"]:.0f}%</td></tr>'
                    f'<tr><td>25–75m (stranded)</td><td style="text-align:right">{_gs["gap"]:.0f}%</td></tr>'
                    f'<tr><td>&gt;75m (break)</td><td style="text-align:right">{_gs["brk"]:.0f}%</td></tr>'
                    f'</table></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gap distribution + fare distribution side by side ─────────────────
        _sc1, _sc2 = st.columns(2)

        with _sc1:
            _gap_rows = []
            for _did in _SPOT_IDS:
                _dt = _spot_acc[_spot_acc["dim_driver_id"] == _did]
                _gs = _spot_gap_stats(_dt)
                for _bucket, _val in [("<25m", _gs["short"]), ("25–75m", _gs["gap"]), (">75m", _gs["brk"])]:
                    _gap_rows.append({"Driver": _SPOT_NAMES[_did], "Bucket": _bucket, "Pct": _val})
            _gap_bar_df = pd.DataFrame(_gap_rows)
            _fig_gaps = px.bar(
                _gap_bar_df, x="Driver", y="Pct", color="Bucket",
                barmode="stack",
                color_discrete_map={"<25m": "#22c55e", "25–75m": "#f59e0b", ">75m": "#ef4444"},
                title="Gap distribution — 30 days",
                labels={"Pct": "% of inter-trip gaps"},
                height=320,
            )
            _fig_gaps.update_layout(yaxis_ticksuffix="%", legend_title="Gap bucket")
            st.plotly_chart(_fig_gaps, use_container_width=True)

        with _sc2:
            _fare_rows = []
            for _did in _SPOT_IDS:
                _dt = _spot_acc[_spot_acc["dim_driver_id"] == _did]
                if len(_dt) == 0:
                    continue
                for _bucket, _mask in [
                    ("Sub-£10",  _dt["trip_price_in_pound"] < 10),
                    ("£10–20",   (_dt["trip_price_in_pound"] >= 10) & (_dt["trip_price_in_pound"] < 20)),
                    ("£20–30",   (_dt["trip_price_in_pound"] >= 20) & (_dt["trip_price_in_pound"] < 30)),
                    ("£30+",     _dt["trip_price_in_pound"] >= 30),
                ]:
                    _fare_rows.append({"Driver": _SPOT_NAMES[_did], "Band": _bucket, "Pct": _mask.mean() * 100})
            _fare_bar_df = pd.DataFrame(_fare_rows)
            _fig_fares = px.bar(
                _fare_bar_df, x="Driver", y="Pct", color="Band",
                barmode="stack",
                color_discrete_map={"Sub-£10": "#ef4444", "£10–20": "#f59e0b", "£20–30": "#60a5fa", "£30+": "#22c55e"},
                title="Fare distribution — 30 days",
                labels={"Pct": "% of trips"},
                height=320,
            )
            _fig_fares.update_layout(yaxis_ticksuffix="%", legend_title="Fare band")
            st.plotly_chart(_fig_fares, use_container_width=True)

        # ── East vs West deep dive ────────────────────────────────────────────
        st.markdown("#### East vs West — does positioning actually translate to better fares?")

        _ew1, _ew2 = st.columns(2)

        with _ew1:
            # Avg fare east vs west per driver
            _ew_fare_rows = []
            for _did in _SPOT_IDS:
                _dt = _spot_acc[_spot_acc["dim_driver_id"] == _did]
                if len(_dt) == 0:
                    continue
                for _side, _sg in _dt.groupby("is_west"):
                    _label = "West (<-0.12°)" if _side else "East (≥-0.12°)"
                    _ew_fare_rows.append({
                        "Driver": _SPOT_NAMES[_did],
                        "Side":   _label,
                        "Avg fare (£)": _sg["trip_price_in_pound"].mean(),
                        "Sub-£10 %":    (_sg["trip_price_in_pound"] < 10).mean() * 100,
                        "£30+ %":       (_sg["trip_price_in_pound"] >= 30).mean() * 100,
                        "Trips":        len(_sg),
                    })
            _ew_df = pd.DataFrame(_ew_fare_rows)
            _fig_ew = px.bar(
                _ew_df, x="Driver", y="Avg fare (£)", color="Side", barmode="group",
                color_discrete_map={"West (<-0.12°)": "#60a5fa", "East (≥-0.12°)": "#fb923c"},
                text="Avg fare (£)",
                title="Avg fare — East vs West pickups",
                height=330,
            )
            _fig_ew.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
            _fig_ew.update_layout(yaxis_title="Avg fare (£)", legend_title="Zone")
            st.plotly_chart(_fig_ew, use_container_width=True)

        with _ew2:
            # Sub-£10 east vs west
            _fig_sub10 = px.bar(
                _ew_df, x="Driver", y="Sub-£10 %", color="Side", barmode="group",
                color_discrete_map={"West (<-0.12°)": "#60a5fa", "East (≥-0.12°)": "#fb923c"},
                text="Sub-£10 %",
                title="Sub-£10 trip rate — East vs West",
                height=330,
            )
            _fig_sub10.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            _fig_sub10.update_layout(yaxis_ticksuffix="%", yaxis_title="% of trips under £10", legend_title="Zone")
            st.plotly_chart(_fig_sub10, use_container_width=True)

        # Hourly avg longitude — where is each driver sitting through the day?
        st.markdown("#### Where they roam — avg pickup longitude by hour")
        st.caption("More negative = further west. -0.12° = Charing Cross line. Dips below the grey band = actively west.")

        _drift_rows = []
        for _did in _SPOT_IDS:
            _dt = _spot_acc[(_spot_acc["dim_driver_id"] == _did) & _spot_acc["hour"].notna()]
            if len(_dt) == 0:
                continue
            for _hr, _hg in _dt.groupby("hour"):
                _drift_rows.append({
                    "Driver": _SPOT_NAMES[_did],
                    "Hour":   int(_hr),
                    "Avg longitude": _hg["plon"].mean(),
                    "Trips":  len(_hg),
                })
        _drift_df = pd.DataFrame(_drift_rows)
        if not _drift_df.empty:
            _fig_drift = px.line(
                _drift_df, x="Hour", y="Avg longitude", color="Driver",
                color_discrete_map={
                    "Mohamed Yousuf":  "#22c55e",
                    "Mukhtar Abdullahi": "#60a5fa",
                    "Abdi Mohamed":    "#a78bfa",
                    "Akeame Plummer":  "#ef4444",
                },
                markers=True,
                title="Hourly pickup longitude — east/west drift through the shift",
                height=370,
                labels={"Avg longitude": "Avg pickup longitude", "Hour": "Hour of day"},
            )
            _fig_drift.add_hrect(
                y0=-0.25, y1=-0.12,
                fillcolor="#60a5fa", opacity=0.08,
                annotation_text="West of Charing Cross", annotation_position="top left",
            )
            _fig_drift.add_hline(y=-0.12, line_dash="dash", line_color="#94a3b8",
                                 annotation_text="-0.12° boundary", annotation_position="right")
            _fig_drift.update_layout(xaxis=dict(tickmode="linear", dtick=2))
            st.plotly_chart(_fig_drift, use_container_width=True)

        # ── Driver narratives ─────────────────────────────────────────────────
        st.markdown("#### What's really going on")
        _n1, _n2 = st.columns(2)
        with _n1:
            st.markdown(
                '<div style="background:#1e1e2e;border-left:4px solid #22c55e;padding:12px 14px;border-radius:6px;color:#e2e8f0;margin-bottom:12px;">'
                '<strong style="color:#22c55e;">🎯 Mohamed Yousuf — Long-haul cherry-picker</strong><br><br>'
                'Only <strong>26% west</strong> and <strong>23% acceptance</strong> — both the lowest of any Cat A driver. '
                'He declines 2,100+ pings a year but when he IS in the west, <strong>32% of those trips are £30+</strong> '
                '(vs 15% from the east). He only enters the west for premium fares. From the east, his Z4/Z5 trips '
                'avg <strong>£22</strong>, Z6 avg <strong>£28</strong> — long hauls back toward the city. '
                'The drift chart shows he stays east until 8–9am when he briefly pushes west for the morning premium, '
                'then retreats. Less positioning, more predatory trip selection.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:#1e1e2e;border-left:4px solid #22c55e;padding:12px 14px;border-radius:6px;color:#e2e8f0;">'
                '<strong style="color:#22c55e;">⚡ Mukhtar Abdullahi — Throughput machine</strong><br><br>'
                '<strong>34% west</strong> — similar to Yousuf, but a completely different strategy. '
                'The drift chart shows he barely moves all day, hovering near center (-0.08 to -0.12). '
                'He doesn\'t chase positioning — <strong>80% of gaps are under 25 minutes</strong> and his '
                'median gap is just <strong>11 minutes</strong>. Sub-£10 rate drops from 30% (east) to 20% '
                '(west), but he doesn\'t need much: volume × nearly zero dead time = Cat A RPH without '
                'ever committing to a side of the city.'
                '</div>',
                unsafe_allow_html=True,
            )
        with _n2:
            st.markdown(
                '<div style="background:#1e1e2e;border-left:4px solid #22c55e;padding:12px 14px;border-radius:6px;color:#e2e8f0;margin-bottom:12px;">'
                '<strong style="color:#22c55e;">⚖️ Abdi Mohamed — The strategic drifter</strong><br><br>'
                '<strong>43% west</strong> — highest of the Cat A outliers and it shows in his RPH (£23.33, best here). '
                'The drift chart reveals his actual strategy: he starts <strong>far east at 4am</strong> (lon -0.06), '
                'gradually pushes west by 9am (lon -0.18, fully across the boundary), works the west peak, '
                'then retreats east by evening. His west trips earn £20.08 vs £17.91 east — <strong>a £2.17 '
                'premium per fare</strong> that compounds across hundreds of trips. He\'s the one who actually '
                'uses positioning as a deliberate time-of-day tool.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:#1e1e2e;border-left:4px solid #ef4444;padding:12px 14px;border-radius:6px;color:#e2e8f0;">'
                '<strong style="color:#ef4444;">🚨 Akeame Plummer — Static and indiscriminate</strong><br><br>'
                '<strong>93% west</strong> — parked in the west all day. The drift chart is nearly a flat line '
                '(-0.16 at every hour, zero movement). He has the position but completely squanders it: '
                '<strong>38% sub-£10 from west pickups</strong>, avg fare just <strong>£12.88</strong>. '
                'His 7 east trips actually averaged <strong>£14.85</strong> — he earns more on the rare '
                'occasion he goes east. He\'s accepting every short cheap Z1/Z2 fare the west throws at him '
                'with a 12-minute pickup wait. The west position earns him better pings; '
                'he just never acts on them. Needs to reject the sub-£10 offers and wait for what the '
                'location deserves.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Same-location comparison ───────────────────────────────────────────
        st.divider()
        st.markdown("#### The same east pings, completely different decisions")
        st.caption(
            "Yousuf, Bartley, and Emran operate in the same east London areas. "
            "The pings on their screen are identical. What they choose to accept is not."
        )

        _COMPARE_IDS  = [219, 82, 32]
        _COMPARE_NAMES = {219: "Yousuf (Cat A)", 82: "Bartley (Cat D)", 32: "Emran (Cat C2)"}

        with st.spinner("Loading east location comparison..."):
            _cmp_acc = pd.read_sql("""
                SELECT dim_driver_id, pickup_lat_long, trip_price_in_pound
                FROM rep_fact_trips
                WHERE dim_driver_id = ANY(%s)
                  AND status IN ('completed','Finished')
                  AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
                  AND distance_in_miles <= 60
                  AND pickedup_trip_datetime >= '2026-01-01'
            """, db.get_conn(), params=(_COMPARE_IDS,))
            _cmp_dec = pd.read_sql("""
                SELECT dim_driver_id, pickup_lat_long
                FROM rep_fact_trips
                WHERE dim_driver_id = ANY(%s)
                  AND status IN ('Driver did not respond','Driver rejected')
                  AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
                  AND trip_booking_datetime >= '2026-01-01'
            """, db.get_conn(), params=(_COMPARE_IDS,))

        def _add_east_area(df):
            _c = df["pickup_lat_long"].apply(parse_dms)
            df = df.copy()
            df["plat"] = [c[0] for c in _c]
            df["plon"] = [c[1] for c in _c]
            df = df.dropna(subset=["plat","plon"])
            df = df[df["plat"].between(51.3,51.7) & df["plon"].between(-0.55,0.3)]
            df["is_east"] = df["plon"] >= -0.12

            def _area(lon):
                if lon < -0.08: return "Inner East\n(City / Clerkenwell)"
                if lon < -0.02: return "Mid East\n(Shoreditch / Hackney)"
                if lon <  0.02: return "Canary Wharf\n/ Stratford"
                if lon <  0.08: return "Outer East\n(Greenwich / Ilford)"
                return               "Far East\n(Romford / Barking)"

            df["area"] = df["plon"].apply(_area)
            return df

        _cmp_acc = _add_east_area(_cmp_acc)
        _cmp_dec = _add_east_area(_cmp_dec)
        _cmp_acc_e = _cmp_acc[_cmp_acc["is_east"]]
        _cmp_dec_e = _cmp_dec[_cmp_dec["is_east"]]

        _AREA_ORDER = [
            "Inner East\n(City / Clerkenwell)",
            "Mid East\n(Shoreditch / Hackney)",
            "Canary Wharf\n/ Stratford",
            "Outer East\n(Greenwich / Ilford)",
            "Far East\n(Romford / Barking)",
        ]

        # Build per-driver per-area stats
        _loc_rows = []
        for _did in _COMPARE_IDS:
            for _area in _AREA_ORDER:
                _ae = _cmp_acc_e[(_cmp_acc_e["dim_driver_id"]==_did) & (_cmp_acc_e["area"]==_area)]
                _de = _cmp_dec_e[(_cmp_dec_e["dim_driver_id"]==_did) & (_cmp_dec_e["area"]==_area)]
                _total_pings = len(_ae) + len(_de)
                if _total_pings == 0:
                    continue
                _loc_rows.append({
                    "Driver":       _COMPARE_NAMES[_did],
                    "Area":         _area,
                    "Pings":        _total_pings,
                    "Accepted":     len(_ae),
                    "Accept %":     len(_ae) / _total_pings * 100,
                    "Avg fare":     _ae["trip_price_in_pound"].mean() if len(_ae) else 0,
                    "Sub-£10 %":    (_ae["trip_price_in_pound"] < 10).mean() * 100 if len(_ae) else 0,
                    "£20+ %":       (_ae["trip_price_in_pound"] >= 20).mean() * 100 if len(_ae) else 0,
                })
        _loc_df = pd.DataFrame(_loc_rows)

        _lc1, _lc2 = st.columns(2)

        with _lc1:
            _fig_accept = px.bar(
                _loc_df, x="Area", y="Accept %", color="Driver", barmode="group",
                color_discrete_map={
                    "Yousuf (Cat A)":  "#22c55e",
                    "Bartley (Cat D)": "#ef4444",
                    "Emran (Cat C2)":  "#fb923c",
                },
                text="Accept %",
                title="Acceptance rate — same east locations",
                category_orders={"Area": _AREA_ORDER},
                height=360,
            )
            _fig_accept.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            _fig_accept.update_layout(yaxis_ticksuffix="%", yaxis_title="% of pings accepted",
                                      xaxis_tickangle=-20)
            st.plotly_chart(_fig_accept, use_container_width=True)

        with _lc2:
            _fig_avgfare = px.bar(
                _loc_df, x="Area", y="Avg fare", color="Driver", barmode="group",
                color_discrete_map={
                    "Yousuf (Cat A)":  "#22c55e",
                    "Bartley (Cat D)": "#ef4444",
                    "Emran (Cat C2)":  "#fb923c",
                },
                text="Avg fare",
                title="Avg fare from accepted trips — same east locations",
                category_orders={"Area": _AREA_ORDER},
                height=360,
            )
            _fig_avgfare.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
            _fig_avgfare.update_layout(yaxis_title="Avg accepted fare (£)", xaxis_tickangle=-20)
            st.plotly_chart(_fig_avgfare, use_container_width=True)

        # Sub-£10 comparison
        _fig_sub10 = px.bar(
            _loc_df, x="Area", y="Sub-£10 %", color="Driver", barmode="group",
            color_discrete_map={
                "Yousuf (Cat A)":  "#22c55e",
                "Bartley (Cat D)": "#ef4444",
                "Emran (Cat C2)":  "#fb923c",
            },
            text="Sub-£10 %",
            title="Sub-£10 acceptance rate — same east locations",
            category_orders={"Area": _AREA_ORDER},
            height=320,
        )
        _fig_sub10.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        _fig_sub10.update_layout(yaxis_ticksuffix="%", yaxis_title="% of accepted trips under £10",
                                 xaxis_tickangle=-20)
        st.plotly_chart(_fig_sub10, use_container_width=True)

        st.markdown(
            '<div style="background:#1e1e2e;border-left:4px solid #facc15;padding:14px 16px;'
            'border-radius:6px;color:#e2e8f0;margin-top:4px;">'
            '<strong>What this proves:</strong> Yousuf is not in a special part of the east. '
            'In every single east London area — City, Shoreditch, Canary Wharf, Outer East — '
            'he receives similar ping volumes to Bartley and Emran. The pings are the same. '
            'His Inner East avg fare is <strong>£25.27 with 0% sub-£10</strong>. '
            'Bartley\'s Inner East avg is <strong>£11.80 with 51% sub-£10</strong>. '
            'Same streets, same Bolt algorithm, different filter. '
            'The east is not the problem — accepting low-value east pings is the problem.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    # ── Render per-driver ─────────────────────────────────────────────────────
    st.markdown("### ✅ Top performers")
    _good_tabs = st.tabs([f"{_GAP_NAMES[did]}" for did in _GAP_GOOD])
    st.markdown("### ⚠️ Comparison drivers")
    _bad_tabs  = st.tabs([f"{_GAP_NAMES[did]}" for did in _GAP_BAD])
    _tabs = _good_tabs + _bad_tabs

    for tab, driver_id in zip(_tabs, _GAP_IDS):
        with tab:
            driver_name = _GAP_NAMES[driver_id]
            gaps = all_gaps[driver_id]

            if not gaps:
                st.success(f"No qualifying gaps found for {driver_name} in the last 14 days.")
                continue

            # Summary banner
            n_gaps         = len(gaps)
            n_lhr          = sum(1 for g in gaps if g["at_lhr"])
            n_decision     = n_gaps - n_lhr
            avg_gap        = np.mean([g["gap_mins"] for g in gaps])
            total_pings    = sum(len(g["pings"]) for g in gaps if not g["at_lhr"])
            questionable   = sum(
                sum(1 for _, row in g["pings"].iterrows()
                    if _classify_ping(row, g["stranded_zone"])[1].startswith("⚠️"))
                for g in gaps if not g["at_lhr"]
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total gaps",            n_gaps)
            c2.metric("✈️ Heathrow queue",     n_lhr,      "Structural — not actionable")
            c3.metric("Decision gaps",         n_decision, "Where choices were made")
            c4.metric("Pings during waits",    total_pings,"Excludes LHR gaps")
            c5.metric("Questionable declines", questionable,
                      "Heading back, ≥£10 fare" if questionable else "All calls look fine")

            st.divider()

            for idx, gap in enumerate(gaps, 1):
                strand  = gap["stranding"]
                rescue  = gap["rescue"]
                pings   = gap["pings"]
                sz      = gap["stranded_zone"]
                gm      = gap["gap_mins"]
                ts      = strand["dropoff_trip_datetime"].strftime("%a %d %b, %H:%M")

                lhr_flag = " ✈️ STRANDED AT HEATHROW — queue issue" if gap["at_lhr"] else ""
                lbl = (f"Gap {idx} — {ts} · Stranded Zone {sz} · {gm:.0f} min wait · "
                       f"{len(pings)} ping{'s' if len(pings) != 1 else ''} received{lhr_flag}")

                with st.expander(lbl):
                    left, right = st.columns([1, 1])

                    with left:
                        if gap["at_lhr"]:
                            st.warning(
                                "✈️ **Driver is in the Heathrow queue.** "
                                "Pings here are irrelevant — the driver can't leave the queue to take them. "
                                "This gap is a structural problem (airport queue), not a decision problem."
                            )
                        st.markdown(f"**Stranding trip** (left them in Zone {sz})")
                        st.markdown(
                            f"- Pickup: Z{int(strand['pickup_zone'] or 0)} → Dropoff: Z{sz}\n"
                            f"- Fare: £{strand['trip_price_in_pound']:.2f} · "
                            f"{strand['distance_in_miles']:.1f} mi · "
                            f"{strand['pob_duration_in_min']:.0f} min ride"
                        )
                        score, stars, label, detail = _rate_rescue(rescue, gm, sz)
                        rescue_dz = int(rescue["dropoff_zone"] or 0)
                        st.markdown(f"**Rescue trip** (ended the wait after {gm:.0f} min)")
                        st.markdown(
                            f"- Pickup: Z{sz} → Dropoff: Z{rescue_dz}\n"
                            f"- Fare: £{rescue['trip_price_in_pound']:.2f} · "
                            f"{rescue['distance_in_miles']:.1f} mi"
                        )
                        colour = ("#22c55e" if score >= 6
                                  else "#f59e0b" if score >= 4
                                  else "#ef4444")
                        st.markdown(
                            f'<div style="background:#1e1e2e;border-left:4px solid {colour};'
                            f'padding:10px 14px;border-radius:4px;margin-top:6px;">'
                            f'<span style="font-size:20px;color:#facc15;letter-spacing:2px;">{stars}</span> &nbsp;'
                            f'<strong style="color:{colour};">{label}</strong><br>'
                            f'<span style="color:#94a3b8;font-size:12px;">{detail}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        if pings.empty:
                            st.markdown("**No pings received during this gap.**  \n"
                                        "_Driver had no choice — genuinely dead zone._")
                        else:
                            st.markdown(f"**{len(pings)} ping(s) during the gap:**")
                            ping_rows = []
                            for _, p in pings.iterrows():
                                pz  = int(p["pickup_zone"])  if pd.notna(p.get("pickup_zone"))  else "?"
                                dz  = int(p["dropoff_zone"]) if pd.notna(p.get("dropoff_zone")) else "?"
                                direction, verdict = _classify_ping(p, sz)

                                # Fare and distance are null for declined trips in the DB
                                # — compute from coords using haversine + Bolt pricing
                                fare_db = p.get("trip_price_in_pound")
                                dist_db = p.get("distance_in_miles")
                                est_dist, est_fare = estimate_ping(
                                    p.get("plat"), p.get("plon"),
                                    p.get("dlat"), p.get("dlon"),
                                )
                                if fare_db and fare_db > 0:
                                    fare_str = f"£{fare_db:.2f}"
                                elif est_fare:
                                    fare_str = f"~£{est_fare:.2f}"
                                else:
                                    fare_str = "—"
                                if dist_db and dist_db > 0:
                                    dist_str = f"{dist_db:.1f}"
                                elif est_dist:
                                    dist_str = f"~{est_dist:.1f}"
                                else:
                                    dist_str = "—"

                                ping_rows.append({
                                    "Time":    p["trip_booking_datetime"].strftime("%H:%M"),
                                    "From Z":  pz,
                                    "To Z":    dz,
                                    "Est fare": fare_str,
                                    "Est miles": dist_str,
                                    "Verdict": verdict,
                                })
                            st.dataframe(pd.DataFrame(ping_rows), use_container_width=True, hide_index=True)

                    with right:
                        # Mini map: stranding dropoff + ping arrows + rescue pickup
                        clat = strand["dropoff_lat"] if pd.notna(strand.get("dropoff_lat")) else CENTER_LAT
                        clon = strand["dropoff_lon"] if pd.notna(strand.get("dropoff_lon")) else CENTER_LON
                        m = folium.Map(location=[clat, clon], zoom_start=11,
                                       tiles="CartoDB dark_matter")
                        folium.GeoJson(
                            GEOJSON_DATA,
                            style_function=lambda f: {
                                "fillColor": "#ffffff", "fillOpacity": 0.03,
                                "color": "#555555", "weight": 1,
                            }
                        ).add_to(m)

                        # Where they got stranded
                        if pd.notna(strand.get("dropoff_lat")):
                            folium.Marker(
                                [strand["dropoff_lat"], strand["dropoff_lon"]],
                                icon=folium.DivIcon(
                                    html='<div style="font-size:18px;">📍</div>',
                                    icon_size=(24, 24), icon_anchor=(12, 24),
                                ),
                                tooltip=f"Stranded here — Z{sz}",
                            ).add_to(m)

                        # Declined pings
                        for _, p in pings.iterrows():
                            if pd.isna(p.get("plat")): continue
                            direction, verdict = _classify_ping(p, sz)
                            colour = _DIR_COLOUR.get(direction, "#6b7280")
                            if pd.notna(p.get("dlat")):
                                folium.PolyLine(
                                    [(p["plat"], p["plon"]), (p["dlat"], p["dlon"])],
                                    color=colour, weight=2.5, opacity=0.8,
                                    tooltip=f"Declined: {p['trip_booking_datetime'].strftime('%H:%M')} | "
                                            f"Z{int(p['pickup_zone'] or 0)}→Z{int(p['dropoff_zone'] or 0)} | "
                                            f"£{p.get('trip_price_in_pound', 0):.2f} | {verdict}",
                                ).add_to(m)
                            folium.CircleMarker(
                                [p["plat"], p["plon"]], radius=5,
                                color=colour, fill=True, fill_opacity=0.9,
                                tooltip=f"Declined ping {p['trip_booking_datetime'].strftime('%H:%M')}",
                            ).add_to(m)

                        # Rescue trip
                        if pd.notna(rescue.get("pickup_lat")) and pd.notna(rescue.get("dropoff_lat")):
                            folium.PolyLine(
                                [(rescue["pickup_lat"], rescue["pickup_lon"]),
                                 (rescue["dropoff_lat"], rescue["dropoff_lon"])],
                                color="#22c55e", weight=3, opacity=0.9,
                                tooltip=f"Eventual trip: £{rescue['trip_price_in_pound']:.2f} | "
                                        f"Z{int(rescue['pickup_zone'] or 0)}→Z{int(rescue['dropoff_zone'] or 0)}",
                            ).add_to(m)
                            folium.CircleMarker(
                                [rescue["pickup_lat"], rescue["pickup_lon"]],
                                radius=6, color="#22c55e", fill=True, fill_opacity=1,
                                tooltip="Rescue pickup",
                            ).add_to(m)

                        st_folium(m, width="100%", height=340, returned_objects=[])

                    # Legend
                    st.markdown(
                        '<span style="color:#f59e0b">━</span> Amber = heading back toward city (declined — worth questioning) &nbsp;&nbsp;'
                        '<span style="color:#ef4444">━</span> Red = going further out (right call to decline) &nbsp;&nbsp;'
                        '<span style="color:#94a3b8">━</span> Grey = lateral move &nbsp;&nbsp;'
                        '<span style="color:#22c55e">━</span> Green = trip eventually taken',
                        unsafe_allow_html=True,
                    )

# ── Trip Flow ────────────────────────────────────────────────────────────────
elif page == "Trip Flow":
    st.title("Trip Flow — Accepted vs Declined")
    st.caption("Bolt doesn't show drivers the fare on a ping — they decide purely based on pickup location and destination. This shows what routes they took vs turned down.")

    if not os.path.exists(_FLOW_PATH):
        st.error("Run `python build_flow.py` once to generate flow_data.parquet")
        st.stop()

    flow = pd.read_parquet(_FLOW_PATH)
    flow = flow[flow["dim_driver_id"].isin(selected_ids)]

    hour_range = st.slider("Filter by hour of day", 0, 23, (0, 23))
    flow = flow[(flow["hour"] >= hour_range[0]) & (flow["hour"] <= hour_range[1])]

    accepted = flow[flow["outcome"] == "Accepted"]
    declined = flow[flow["outcome"] == "Declined"]

    st.markdown(f"**{len(accepted):,} accepted** · **{len(declined):,} declined** in this hour window")

    # ── Pickup→Dropoff zone heatmaps side by side
    st.subheader("Pickup Zone → Dropoff Zone  (what routes are they taking vs turning down?)")

    def make_flow_heatmap(df, title, colour):
        pivot = df.groupby(["pickup_zone","dropoff_zone"]).size().unstack(fill_value=0)
        pivot.index   = [f"Pickup Z{z}" for z in pivot.index]
        pivot.columns = [f"Dropoff Z{z}" for z in pivot.columns]
        pct = (pivot.div(pivot.values.sum()) * 100).round(1)
        fig = px.imshow(pct, text_auto=".1f", color_continuous_scale=colour,
                        labels={"x":"Dropoff Zone","y":"Pickup Zone","color":"% of trips"},
                        title=title, aspect="auto")
        fig.update_layout(height=360, coloraxis_showscale=False)
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_flow_heatmap(accepted, "ACCEPTED trips — % of total", "Greens"), use_container_width=True)
    with col2:
        st.plotly_chart(make_flow_heatmap(declined, "DECLINED trips — % of total", "Reds"), use_container_width=True)

    st.info("📌 Read each cell as: '% of all accepted/declined trips that had this pickup→dropoff combination.'\n"
            "High values on the diagonal = local trips staying in the same zone. Off-diagonal = cross-zone runs.")

    # ── Delta heatmap: where accepted ≠ declined
    st.subheader("Delta — where accepted rate differs most from declined rate")
    st.caption("Green = drivers accept these routes proportionally MORE than they decline. Red = they decline these MORE than they accept.")

    def flow_pct(df):
        p = df.groupby(["pickup_zone","dropoff_zone"]).size().unstack(fill_value=0)
        return (p.div(p.values.sum()) * 100).reindex(index=range(1,7), columns=range(1,7), fill_value=0)

    acc_pct = flow_pct(accepted)
    dec_pct = flow_pct(declined)
    delta = (acc_pct - dec_pct).round(2)
    delta.index   = [f"Pickup Z{z}" for z in delta.index]
    delta.columns = [f"Dropoff Z{z}" for z in delta.columns]
    fig_d = px.imshow(delta, text_auto=".1f", color_continuous_scale="RdYlGn",
                      color_continuous_midpoint=0,
                      labels={"x":"Dropoff Zone","y":"Pickup Zone","color":"Accept% − Decline%"},
                      aspect="auto")
    fig_d.update_layout(height=400)
    st.plotly_chart(fig_d, use_container_width=True)
    st.caption("Green cells = routes they actively prefer. Red cells = routes they actively avoid.")

    # ── Bar: what dropoff zone do they prefer vs avoid?
    st.subheader("Do they prefer or avoid certain dropoff zones?")
    dropoff_compare = pd.DataFrame({
        "Dropoff Zone": [f"Zone {z}" for z in range(1,7)],
        "Accepted %": [accepted["dropoff_zone"].eq(z).mean()*100 for z in range(1,7)],
        "Declined %": [declined["dropoff_zone"].eq(z).mean()*100 for z in range(1,7)],
    })
    fig_b = px.bar(dropoff_compare.melt(id_vars="Dropoff Zone", var_name="Outcome", value_name="% of trips"),
                   x="Dropoff Zone", y="% of trips", color="Outcome", barmode="group",
                   color_discrete_map={"Accepted %":"#22c55e","Declined %":"#ef4444"})
    fig_b.update_layout(height=360)
    st.plotly_chart(fig_b, use_container_width=True)

    # ── Key insight box
    acc_z1do = accepted["dropoff_zone"].eq(1).mean()*100
    dec_z1do = declined["dropoff_zone"].eq(1).mean()*100
    acc_z56do = accepted["dropoff_zone"].ge(5).mean()*100
    dec_z56do = declined["dropoff_zone"].ge(5).mean()*100
    st.info(f"📌 **Zone 1 dropoffs**: {acc_z1do:.0f}% of accepted vs {dec_z1do:.0f}% of declined — "
            f"they decline {'more' if dec_z1do > acc_z1do else 'fewer'} trips going TO Zone 1.\n\n"
            f"📌 **Zone 5-6 dropoffs**: {acc_z56do:.0f}% of accepted vs {dec_z56do:.0f}% of declined — "
            f"they accept {'more' if acc_z56do > dec_z56do else 'fewer'} long-haul trips going to outer zones.\n\n"
            "On Bolt, drivers see an estimated fare and distance before accepting. Low fare + short distance in Zone 1 gridlock is a clear decline signal — the full-cycle maths puts those trips below their shift average.")

# ── Zone 1 Selectivity ────────────────────────────────────────────────────────
elif page == "Zone 1 Selectivity":
    st.title("Zone 1 Selectivity")
    st.caption("Zone 1 is both the most accepted AND most declined pickup zone. What specifically are they choosing?")

    if not os.path.exists(_FLOW_PATH):
        st.error("flow_data.parquet not found — run `python build_flow.py` first.")
        st.stop()

    with st.spinner("Loading trip flow data..."):
        flow = pd.read_parquet(_FLOW_PATH)
        flow = flow[flow["dim_driver_id"].isin(selected_ids)]

    z1_pickup  = flow[flow["pickup_zone"]  == 1].copy()
    z1_dropoff = flow[flow["dropoff_zone"] == 1].copy()

    z1_acc = z1_pickup[z1_pickup["outcome"] == "Accepted"]
    z1_dec = z1_pickup[z1_pickup["outcome"] == "Declined"]

    avg_dist_acc  = z1_acc["distance_in_miles"].mean()
    avg_dist_dec  = z1_dec["distance_in_miles"].mean()
    avg_fare_acc  = z1_acc["trip_price_in_pound"].dropna().replace(0, pd.NA).dropna().mean()
    avg_fare_dec  = z1_dec["trip_price_in_pound"].dropna().replace(0, pd.NA).dropna().mean()
    z1_local_acc  = (z1_acc["dropoff_zone"] == 1).mean() * 100
    z1_local_dec  = (z1_dec["dropoff_zone"] == 1).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Z1 pickups accepted", f"{len(z1_acc):,}")
    col2.metric("Z1 pickups declined", f"{len(z1_dec):,}")
    col3.metric("Avg distance accepted", f"{avg_dist_acc:.1f} mi",
                f"+{avg_dist_acc - avg_dist_dec:.1f} mi vs declined")
    col4.metric("Avg fare accepted", f"£{avg_fare_acc:.2f}" if pd.notna(avg_fare_acc) else "N/A",
                f"+£{avg_fare_acc - avg_fare_dec:.2f} vs declined" if pd.notna(avg_fare_acc) and pd.notna(avg_fare_dec) else "")

    # ── SECTION 1: Where do Z1 pickups go? ───────────────────────────────────
    st.divider()
    st.subheader("1 — Where do Zone 1 pickups go? (accepted vs declined)")

    col_a, col_b = st.columns(2)
    for df_side, title, cscale, col in [
        (z1_acc, "✅ Accepted — dropoff zone", "Greens", col_a),
        (z1_dec, "❌ Declined — dropoff zone", "Reds",   col_b),
    ]:
        grp = df_side.groupby("dropoff_zone").size().reset_index(name="trips")
        grp["pct"] = (grp["trips"] / grp["trips"].sum() * 100).round(1)
        grp["zone_label"] = "Zone " + grp["dropoff_zone"].astype(str)
        fig = px.bar(grp.sort_values("dropoff_zone"), x="zone_label", y="pct",
                     color="pct", color_continuous_scale=cscale,
                     text="pct", title=title,
                     labels={"zone_label": "Dropoff Zone", "pct": "% of trips"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(height=340, coloraxis_showscale=False)
        col.plotly_chart(fig, use_container_width=True)

    st.info(
        f"📌 **Short Zone 1 hops are being filtered out.** "
        f"{z1_local_dec:.0f}% of declined Z1 pickups stay within Zone 1 (short hop), "
        f"vs only {z1_local_acc:.0f}% of accepts. "
        f"They wait for trips that pull them out into longer-paying zones."
    )

    # ── SECTION 2: Distance and fare buckets ─────────────────────────────────
    st.divider()
    st.subheader("2 — Accepted trips are longer and higher value")

    z1_pickup["distance_bucket"] = pd.cut(
        z1_pickup["distance_in_miles"],
        bins=[0, 2, 4, 7, 60],
        labels=["Short (0-2 mi)", "Medium (2-4 mi)", "Long (4-7 mi)", "Very long (7+ mi)"],
    )
    bucket_totals = z1_pickup.groupby("outcome", observed=True).size().reset_index(name="total")
    dist_grp = (
        z1_pickup.groupby(["distance_bucket", "outcome"], observed=True)
        .size().reset_index(name="trips")
        .merge(bucket_totals, on="outcome")
    )
    dist_grp["pct"] = (dist_grp["trips"] / dist_grp["total"] * 100).round(1)

    fig_dist = px.bar(
        dist_grp, x="distance_bucket", y="pct", color="outcome", barmode="group",
        color_discrete_map={"Accepted": "#22c55e", "Declined": "#ef4444"},
        text="pct",
        title="Zone 1 pickup trips — distance profile: accepted vs declined",
        labels={"distance_bucket": "Trip Distance", "pct": "% of that outcome", "outcome": ""},
    )
    fig_dist.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_dist.update_layout(height=360, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_dist, use_container_width=True)

    # Fare buckets (Bolt hides fare, so many declined trips will have £0 — show only non-zero)
    z1_fare = z1_pickup[z1_pickup["trip_price_in_pound"] > 0].copy()
    z1_fare["fare_bucket"] = pd.cut(
        z1_fare["trip_price_in_pound"],
        bins=[0, 8, 15, 25, 9999],
        labels=["Low (£0–8)", "Medium (£8–15)", "High (£15–25)", "Premium (£25+)"],
    )
    fare_totals = z1_fare.groupby("outcome", observed=True).size().reset_index(name="total")
    fare_grp = (
        z1_fare.groupby(["fare_bucket", "outcome"], observed=True)
        .size().reset_index(name="trips")
        .merge(fare_totals, on="outcome")
    )
    fare_grp["pct"] = (fare_grp["trips"] / fare_grp["total"] * 100).round(1)

    fig_fare = px.bar(
        fare_grp, x="fare_bucket", y="pct", color="outcome", barmode="group",
        color_discrete_map={"Accepted": "#22c55e", "Declined": "#ef4444"},
        text="pct",
        title="Zone 1 pickup trips — fare profile: accepted vs declined (Bolt fares estimated where hidden)",
        labels={"fare_bucket": "Fare Band", "pct": "% of that outcome", "outcome": ""},
    )
    fig_fare.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_fare.update_layout(height=360, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_fare, use_container_width=True)

    dist_uplift = ((avg_dist_acc / avg_dist_dec) - 1) * 100 if avg_dist_dec else 0
    st.info(
        f"📌 Accepted Zone 1 trips average **{avg_dist_acc:.1f} miles** vs "
        f"**{avg_dist_dec:.1f} miles** for declined — {dist_uplift:.0f}% longer. "
        f"They're effectively using distance as a proxy for trip value (Bolt hides fare, "
        f"but a longer route = higher payout)."
    )

    # ── SECTION 3: Hour-by-hour acceptance rate ───────────────────────────────
    st.divider()
    st.subheader("3 — When are they most selective in Zone 1?")

    z1_hourly = (
        z1_pickup.groupby(["hour", "outcome"])
        .size().reset_index(name="trips")
    )
    z1_total_hr = z1_pickup.groupby("hour").size().reset_index(name="total")
    z1_hourly = z1_hourly.merge(z1_total_hr, on="hour")
    z1_hourly["pct"] = (z1_hourly["trips"] / z1_hourly["total"] * 100).round(1)

    z1_rate = (
        z1_pickup.groupby(["hour", "outcome"])
        .size().unstack(fill_value=0)
        .reset_index()
    )
    z1_rate.columns.name = None
    if "Accepted" in z1_rate.columns and "Declined" in z1_rate.columns:
        z1_rate["accept_rate"] = (
            z1_rate["Accepted"] /
            (z1_rate["Accepted"] + z1_rate["Declined"]) * 100
        ).round(1)
        mean_rate = z1_rate["accept_rate"].mean()

        fig_hr = px.line(
            z1_rate, x="hour", y="accept_rate", markers=True,
            title="Zone 1 pickup acceptance rate by hour of day",
            labels={"hour": "Hour of Day", "accept_rate": "Acceptance Rate %"},
        )
        fig_hr.update_layout(height=320, xaxis=dict(dtick=1, range=[0, 23]))
        fig_hr.add_hline(y=mean_rate, line_dash="dash", line_color="#f59e0b",
                         annotation_text=f"avg {mean_rate:.0f}%",
                         annotation_position="top right")
        st.plotly_chart(fig_hr, use_container_width=True)

    # ── SECTION 4: Zone 1 DROPOFF trips (inbound) ────────────────────────────
    st.divider()
    st.subheader("4 — Zone 1 dropoffs: which inbound trips do they take?")
    st.caption("These are trips that END in Zone 1 — great for repositioning back into the high-density area.")

    z1_in_acc = z1_dropoff[z1_dropoff["outcome"] == "Accepted"]
    z1_in_dec = z1_dropoff[z1_dropoff["outcome"] == "Declined"]

    col_c, col_d = st.columns(2)
    for df_side, title, cscale, col in [
        (z1_in_acc, "✅ Accepted inbound — pickup zone", "Greens", col_c),
        (z1_in_dec, "❌ Declined inbound — pickup zone", "Reds",   col_d),
    ]:
        grp = df_side.groupby("pickup_zone").size().reset_index(name="trips")
        grp["pct"] = (grp["trips"] / grp["trips"].sum() * 100).round(1)
        grp["zone_label"] = "Zone " + grp["pickup_zone"].astype(str)
        fig = px.bar(grp.sort_values("pickup_zone"), x="zone_label", y="pct",
                     color="pct", color_continuous_scale=cscale,
                     text="pct", title=title,
                     labels={"zone_label": "Pickup Zone", "pct": "% of trips"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(height=340, coloraxis_showscale=False)
        col.plotly_chart(fig, use_container_width=True)

    # Compare accept rate by pickup zone for Z1 dropoffs
    z1_inbound_rate = (
        z1_dropoff.groupby(["pickup_zone", "outcome"])
        .size().unstack(fill_value=0).reset_index()
    )
    z1_inbound_rate.columns.name = None
    if "Accepted" in z1_inbound_rate.columns and "Declined" in z1_inbound_rate.columns:
        z1_inbound_rate["accept_rate"] = (
            z1_inbound_rate["Accepted"] /
            (z1_inbound_rate["Accepted"] + z1_inbound_rate["Declined"]) * 100
        ).round(1)
        z1_inbound_rate["zone_label"] = "Zone " + z1_inbound_rate["pickup_zone"].astype(str)
        fig_inr = px.bar(
            z1_inbound_rate.sort_values("pickup_zone"),
            x="zone_label", y="accept_rate",
            color="accept_rate", color_continuous_scale="RdYlGn",
            text="accept_rate",
            title="Acceptance rate for trips dropping in Zone 1 — by pickup zone",
            labels={"zone_label": "Pickup Zone", "accept_rate": "Accept Rate %"},
        )
        fig_inr.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_inr.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_inr, use_container_width=True)

    st.info(
        "📌 **Any trip ending in Zone 1 is positioning value.** "
        "Even a long ride from Zone 5 that drops in Zone 1 is accepted — it puts the driver "
        "back in the highest-density ping area. The inbound acceptance rate should be "
        "consistently higher across all pickup zones."
    )

    # ── SECTION 5: Area breakdown (from parquet pickup_address) ─────────────
    st.divider()
    st.subheader("5 — Which specific areas within Zone 1 do they accept most?")
    st.caption("Based on pickup address strings in the parquet — postcode district identifies the area.")

    def _extract_london_area(address):
        if not address or not isinstance(address, str):
            return None
        addr_upper = address.upper()
        m = re.search(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s+\d[A-Z]{2}\b', addr_upper)
        if m:
            district = m.group(1)
            _MAP = {
                "EC1": "City / Barbican",   "EC1A": "City / Barbican",  "EC1M": "City / Barbican",
                "EC1N": "City / Barbican",  "EC1R": "City / Barbican",  "EC1V": "City / Barbican",
                "EC1Y": "City / Barbican",
                "EC2": "City / Bank",       "EC2A": "City / Bank",      "EC2M": "City / Bank",
                "EC2N": "City / Bank",      "EC2R": "City / Bank",      "EC2V": "City / Bank",
                "EC2Y": "City / Bank",
                "EC3": "City / Aldgate",    "EC3A": "City / Aldgate",   "EC3M": "City / Aldgate",
                "EC3N": "City / Aldgate",   "EC3R": "City / Aldgate",   "EC3V": "City / Aldgate",
                "EC4": "City / Blackfriars","EC4A": "City / Blackfriars","EC4M": "City / Blackfriars",
                "EC4N": "City / Blackfriars","EC4R": "City / Blackfriars","EC4V": "City / Blackfriars",
                "EC4Y": "City / Blackfriars",
                "WC1": "Holborn/Bloomsbury","WC1A": "Holborn/Bloomsbury","WC1B": "Holborn/Bloomsbury",
                "WC1E": "Holborn/Bloomsbury","WC1H": "Holborn/Bloomsbury","WC1N": "Holborn/Bloomsbury",
                "WC1R": "Holborn/Bloomsbury","WC1V": "Holborn/Bloomsbury","WC1X": "Holborn/Bloomsbury",
                "WC2": "Covent Garden/Strand","WC2A": "Covent Garden/Strand","WC2B": "Covent Garden/Strand",
                "WC2E": "Covent Garden/Strand","WC2H": "Covent Garden/Strand","WC2N": "Covent Garden/Strand",
                "WC2R": "Covent Garden/Strand",
                "W1A": "Mayfair/Oxford St", "W1B": "Mayfair/Oxford St", "W1C": "Mayfair/Oxford St",
                "W1D": "Soho",              "W1F": "Soho",              "W1G": "Marylebone",
                "W1H": "Marylebone",        "W1J": "Mayfair/Oxford St", "W1K": "Mayfair/Oxford St",
                "W1S": "Mayfair/Oxford St", "W1T": "Soho",              "W1U": "Marylebone",
                "W1W": "Fitzrovia",
                "SW1A": "Westminster",      "SW1E": "Westminster",      "SW1H": "Westminster",
                "SW1P": "Westminster",      "SW1V": "Pimlico/Victoria", "SW1W": "Belgravia",
                "SW1X": "Belgravia",        "SW1Y": "St James's",
                "SE1": "Southwark/Waterloo",
                "N1": "Islington/Angel",
                "E1": "Whitechapel/Aldgate East",
                "E1W": "Wapping",
            }
            mapped = _MAP.get(district)
            if mapped:
                return mapped
            if district.startswith(("EC", "WC", "W1", "SW1", "SE1")):
                return district
        _KEYWORDS = [
            ("MAYFAIR", "Mayfair/Oxford St"),        ("SOHO", "Soho"),
            ("COVENT GARDEN", "Covent Garden/Strand"),("WESTMINSTER", "Westminster"),
            ("VICTORIA", "Pimlico/Victoria"),         ("BELGRAVIA", "Belgravia"),
            ("ST JAMES", "St James's"),               ("WATERLOO", "Southwark/Waterloo"),
            ("SOUTHWARK", "Southwark/Waterloo"),      ("HOLBORN", "Holborn/Bloomsbury"),
            ("BLOOMSBURY", "Holborn/Bloomsbury"),     ("STRAND", "Covent Garden/Strand"),
            ("ISLINGTON", "Islington/Angel"),         ("ANGEL", "Islington/Angel"),
            ("SHOREDITCH", "City / Bank"),            ("CITY OF LONDON", "City / Bank"),
            ("BARBICAN", "City / Barbican"),          ("ALDGATE", "City / Aldgate"),
            ("MARYLEBONE", "Marylebone"),             ("FITZROVIA", "Fitzrovia"),
            ("WHITECHAPEL", "Whitechapel/Aldgate East"),
        ]
        for keyword, label in _KEYWORDS:
            if keyword in addr_upper:
                return label
        return None

    # Use the parquet directly — pickup_address is now included in the file
    if "pickup_address" not in flow.columns:
        st.warning("pickup_address not in parquet — re-run `python build_flow.py` to rebuild with addresses.")
    else:
        z1_addr = flow[
            (flow["pickup_zone"] == 1) &
            flow["pickup_address"].notna() &
            (flow["pickup_address"].astype(str).str.strip() != "")
        ].copy()
        z1_addr["area"] = z1_addr["pickup_address"].apply(_extract_london_area)
        z1_addr = z1_addr[z1_addr["area"].notna()]

        if z1_addr.empty:
            st.warning("pickup_address column is present but empty for Zone 1 trips — may not be populated in DB.")
        else:
            area_pivot = (
                z1_addr.groupby(["area", "outcome"])
                .size().unstack(fill_value=0).reset_index()
            )
            area_pivot.columns.name = None
            if "Accepted" not in area_pivot.columns:
                area_pivot["Accepted"] = 0
            if "Declined" not in area_pivot.columns:
                area_pivot["Declined"] = 0
            area_pivot["total"]       = area_pivot["Accepted"] + area_pivot["Declined"]
            area_pivot["accept_rate"] = (
                area_pivot["Accepted"] / area_pivot["total"] * 100
            ).round(1)
            area_pivot = area_pivot[area_pivot["total"] >= 5].sort_values("accept_rate", ascending=False)

            col_e, col_f = st.columns(2)
            with col_e:
                fig_area = px.bar(
                    area_pivot.head(15).sort_values("accept_rate"),
                    x="accept_rate", y="area", orientation="h",
                    color="accept_rate", color_continuous_scale="RdYlGn",
                    text="accept_rate",
                    title="Acceptance rate by Zone 1 pickup area",
                    labels={"area": "", "accept_rate": "Accept Rate %"},
                )
                fig_area.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
                fig_area.update_layout(height=max(350, len(area_pivot.head(15)) * 28),
                                       coloraxis_showscale=False)
                st.plotly_chart(fig_area, use_container_width=True)

            with col_f:
                fig_vol = px.bar(
                    area_pivot.sort_values("total", ascending=False).head(15),
                    x="total", y="area", orientation="h",
                    text="total",
                    title="Zone 1 ping volume by area (all pings offered)",
                    labels={"area": "", "total": "Total pings"},
                )
                fig_vol.update_traces(texttemplate="%{text}", textposition="outside",
                                      marker_color="#6366f1")
                fig_vol.update_layout(height=max(350, len(area_pivot.head(15)) * 28))
                st.plotly_chart(fig_vol, use_container_width=True)

            display_tbl = area_pivot[["area", "Accepted", "Declined", "total", "accept_rate"]].rename(
                columns={"area": "Area", "total": "Total Pings", "accept_rate": "Accept Rate %"}
            ).sort_values("Accept Rate %", ascending=False).reset_index(drop=True)
            st.dataframe(display_tbl, use_container_width=True, hide_index=True)

            area_metrics = (
                z1_addr[z1_addr["outcome"] == "Accepted"]
                .groupby("area")
                .agg(
                    avg_fare=("trip_price_in_pound", lambda x: x[x > 0].mean()),
                    avg_dist=("distance_in_miles", "mean"),
                    trips=("outcome", "count"),
                ).round(2).reset_index()
                .rename(columns={"area": "Area", "avg_fare": "Avg Fare £",
                                 "avg_dist": "Avg Distance mi", "trips": "Accepted Trips"})
                .sort_values("Avg Fare £", ascending=False)
            )
            st.markdown("**Accepted trips per area — avg fare and distance**")
            st.dataframe(area_metrics, use_container_width=True, hide_index=True)

# ── Zone 1: The Why ───────────────────────────────────────────────────────────
elif page == "Zone 1: The Why":
    st.title("Zone 1: The Why")
    st.caption("Why do the top 10 drivers decline Zone 1→Zone 1 short hops more than any other trip type — and why does it make them more money?")

    st.markdown("""
<style>
.why-card {
    background: #1a1a2e;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.why-title { color: #f59e0b; font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.why-body  { color: #d1d5db; font-size: 14px; line-height: 1.7; }
.coaching-card {
    background: #052e16;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 8px;
}
.coaching-title { color: #22c55e; font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.coaching-body  { color: #d1d5db; font-size: 14px; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

    # ── Key numbers strip ─────────────────────────────────────────────────────
    st.subheader("The numbers behind the behaviour")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Z1→Z1 avg speed",        "6.2 mph",   "Gridlock pace")
    c2.metric("Z1→Z6 avg speed",        "19.3 mph",  "3× faster roads")
    c3.metric("Z1→Z1 full-cycle RPH",   "£20.25/hr", "Below fleet avg £22.9")
    c4.metric("Z1→Z6 full-cycle RPH",   "£30.98/hr", "+35% above fleet avg")
    c5.metric("Difference per hour",    "+£10.73/hr", "From being selective")

    st.divider()

    # ── Reason 1 ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="why-card">
<div class="why-title">1 — The speed trap: the meter doesn't fully compensate for gridlock</div>
<div class="why-body">
Both Uber and Bolt charge <strong>£1.25/mile + £0.15/minute</strong>. At Zone 1's 6.2 mph average, that translates to:
<br><br>
&nbsp;&nbsp;&nbsp;£0.15 × 60 min &nbsp;+ £1.25 × 6.2 mph &nbsp;= <strong>£9/hr time + £7.75/hr distance = £16.75/hr rate</strong>
<br>
&nbsp;&nbsp;&nbsp;vs. Zone 6 at 19 mph: £9/hr time + £23.75/hr distance = <strong>£32.75/hr rate</strong>
<br><br>
The per-minute rate (£9/hr) barely keeps pace with sitting in traffic. The per-mile rate — where the real money is — collapses
at slow speed. The only thing saving short Zone 1 trips is the <strong>minimum fare floor</strong>, which is exactly why Zone 1→Zone 1
has the highest fare-per-mile in our data (£7.17/mi). These are trips where the meter hit the floor before it could earn its
natural rate. Minimum-fare trips are, by definition, the worst outcome.
</div>
</div>
""", unsafe_allow_html=True)

    # Speed-by-zone table
    speed_df = pd.DataFrame({
        "Pickup Zone": ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"],
        "Avg Speed (mph)": [8.3, 10.4, 13.0, 15.5, 16.2, 19.3],
        "Fare/mile rate (£)": [7.17, 4.97, 3.65, 3.08, 3.03, 2.74],
        "Implied RPH (£/hr)": [37.90, 34.07, 33.46, 34.73, 42.24, 46.76],
    })
    fig_spd = px.bar(
        speed_df, x="Pickup Zone", y="Avg Speed (mph)",
        color="Avg Speed (mph)", color_continuous_scale="RdYlGn",
        text="Avg Speed (mph)",
        title="Average trip speed by pickup zone — Zone 1 is 3× slower than Zone 6",
    )
    fig_spd.update_traces(texttemplate="%{text:.1f} mph", textposition="outside")
    fig_spd.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(fig_spd, use_container_width=True)
    st.caption("TomTom & TfL data confirm central London averages 7–9 mph. Our data shows Zone 1→Zone 1 short hops at 6.2 mph — the slowest segment in the dataset.")

    st.divider()

    # ── Reason 2 ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="why-card">
<div class="why-title">2 — The full-cycle maths: Zone 1 short hops earn below the fleet average</div>
<div class="why-body">
Once you account for the full cycle — inter-trip wait + pickup time + ride time — the numbers
look very different from the fare alone.
<br><br>
&nbsp;&nbsp;&nbsp;<strong>Z1→Z1 short hop:</strong> &nbsp;£11.52 fare &nbsp;÷&nbsp; (14 min wait + 5 min pickup + 21 min ride) / 60
&nbsp;= <strong>£20.25/hr</strong> &nbsp;← below fleet average (£22.9/hr)
<br><br>
&nbsp;&nbsp;&nbsp;<strong>Z1→Z5/Z6 airport run:</strong> £41.27 fare ÷ (21 min wait + 11 min pickup + 59 min ride) / 60
&nbsp;= <strong>£30.98/hr</strong> &nbsp;← 35% above fleet average
<br><br>
The short hop isn't just lower than the airport run — it's <em>below what a driver earns on average
across their whole shift</em>. Every short Z1 hop accepted is a drag on the shift, not a contribution to it.
</div>
</div>
""", unsafe_allow_html=True)

    opp_df = pd.DataFrame({
        "Strategy": ["Accept every Z1→Z1\nshort hop", "Fleet average\n(all zones)", "Pass short hops,\ntake airport runs"],
        "Full-cycle RPH": [20.25, 22.9, 30.98],
    })
    fig_opp = px.bar(
        opp_df, x="Strategy", y="Full-cycle RPH",
        color="Full-cycle RPH", color_continuous_scale="RdYlGn",
        text="Full-cycle RPH",
        title="Full-cycle RPH (fare ÷ wait + pickup + ride): short hops drag you below average",
        labels={"Full-cycle RPH": "£/hr", "Strategy": ""},
    )
    fig_opp.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
    fig_opp.update_layout(height=340, coloraxis_showscale=False)
    fig_opp.add_hline(y=22.9, line_dash="dash", line_color="orange",
                      annotation_text="Fleet avg £22.9/hr", annotation_position="top right")
    st.plotly_chart(fig_opp, use_container_width=True)

    st.divider()

    # ── Reason 3 ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="why-card">
<div class="why-title">3 — They see the estimated fare and distance — and the maths doesn't add up</div>
<div class="why-body">
On Bolt, drivers see a <strong>rough estimated fare and the trip distance</strong> on the ping — not the full
route map, but enough to make a quick judgement.
<br><br>
When a ping shows £7 for 0.8 miles in Zone 1, an experienced driver knows:
<br>
&nbsp;&nbsp;&nbsp;• Zone 1 traffic is 6.2 mph average
<br>
&nbsp;&nbsp;&nbsp;• 0.8 miles at 6.2 mph ≈ 8 minutes riding, plus ~5 min pickup = <strong>13 minutes for £7</strong>
<br>
&nbsp;&nbsp;&nbsp;• That's £32/hr ride-only, or roughly <strong>£20/hr full-cycle</strong> — below their average
<br><br>
Contrast with a £38 ping for 15 miles: even before doing the maths, <strong>high fare + long distance = accept</strong>.
The fare and distance together are the signal. Short distance + low fare = the Zone 1 gridlock trap.
<br><br>
This makes the coaching rule very concrete: <strong>in Zone 1, the threshold a driver should hold out for
is roughly £12+ fare or 2+ miles</strong>. Below that, the full-cycle maths puts you below your own shift average.
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Reason 4 ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="why-card">
<div class="why-title">4 — The congestion charge is a fixed cost that short hops barely cover</div>
<div class="why-body">
The entire Zone 1 sits inside the Congestion Charge zone (£15/day in 2026).
A driver working Zone 1 pays that overhead regardless of how many trips they take.
<br><br>
&nbsp;&nbsp;&nbsp;<strong>Short Z1→Z1 hop (£6–8):</strong> contributes ~£0.50–1 toward the daily charge overhead
<br>
&nbsp;&nbsp;&nbsp;<strong>Airport run from Z1 (£40–47):</strong> absorbs the full overhead in one trip and leaves margin
<br><br>
Every minimum-fare trip in the congestion zone is effectively a trip where a meaningful slice
of earnings goes to TfL. Longer, higher-value trips amortise that cost far more efficiently.
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── The accepted Z1→Z1 trips ─────────────────────────────────────────────
    st.subheader("What about the 3,195 Zone 1→Zone 1 trips they DID accept?")
    st.markdown("""
The top drivers **do** accept Z1→Z1 trips — just not all of them. The ones they accept average **2.15 miles**.
That's not a cross-street hop. That's Mayfair → The City, or Soho → Southwark — a meaningful intra-London route
that happens to stay within Zone 1's polygon. At 2.15 miles and 21 minutes, the fare is £11.52 and the implied
RPH is £37.90 — worth taking.

The **declined** Z1→Z1 pings are almost certainly concentrated in the sub-1-mile bucket (576 trips in our distribution
sit at 0.5–1 mile). Those are the tourist hops, the hotel-to-restaurant drops, the cross-street pickups that
look small on Bolt's map and promise nothing but 15 minutes of Piccadilly Circus traffic.
""")

    dist_dist = pd.DataFrame({
        "Distance band": ["Under 0.5 mi", "0.5–1 mi", "1–1.5 mi", "1.5–2 mi", "2–3 mi", "3–5 mi", "5+ mi"],
        "Accepted Z1→Z1 trips": [79, 497, 712, 646, 664, 434, 163],
    })
    fig_dd = px.bar(
        dist_dist, x="Distance band", y="Accepted Z1→Z1 trips",
        color="Accepted Z1→Z1 trips", color_continuous_scale="Blues",
        text="Accepted Z1→Z1 trips",
        title="Accepted Zone 1→Zone 1 trips by distance — the very short ones are where declines cluster",
    )
    fig_dd.update_traces(texttemplate="%{text}", textposition="outside")
    fig_dd.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # ── Coaching rule ─────────────────────────────────────────────────────────
    st.markdown("""
<div class="coaching-card">
<div class="coaching-title">The coaching rule this generates</div>
<div class="coaching-body">
Zone 1 is still the best place to be — highest ping density, shortest pickup wait (6.7 min).
The mistake average drivers make is accepting <em>every</em> Zone 1 ping out of fear of being idle.
Short Z1 hops pull the full-cycle RPH to <strong>£20.25/hr — below what they earn on average across
their whole shift (£22.9/hr)</strong>. It's not just that they're missing airport runs; they're actively
dragging their own average down.
<br><br>
<strong>The rule for drivers:</strong> In Zone 1, if the ping shows under ~£12 fare or under ~2 miles,
the full-cycle maths puts that trip below your shift average. Pass and wait for the next ping.
Zone 1 has the shortest wait for the next ping (6.7 min average) — the cost of passing is low.
<br><br>
<strong>The numbers to give drivers:</strong><br>
— Full-cycle RPH accepting every Z1 short hop: <strong>£20.25/hr</strong> (below your average)<br>
— Full-cycle RPH on Z1 airport runs: <strong>£30.98/hr</strong> (35% above your average)<br>
— Difference: <strong>+£10.73/hr just from being selective on low-fare Z1 pings</strong>
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption("Sources: TomTom Traffic Index (London slowest city in Europe); TfL central London speed data (7.4 mph avg, 7am–7pm); Uber/Bolt London pricing: £2.50 base + £1.25/mile + £0.15/min; Congestion Charge Zone: £15/day (2026). Trip economics from Odysse fleet data.")

# ── Trip Strategy DNA ─────────────────────────────────────────────────────────
elif page == "Trip Strategy DNA":
    st.title("Trip Strategy DNA")
    st.caption("Do the top drivers stay in Zone 1 and chain short trips, or do they do long back-and-forth runs? Who's doing what — and what's actually working?")

    with st.spinner("Loading trip sequences..."):
        raw = db.load_zone_trips()
        raw = raw[raw["dim_driver_id"].isin(selected_ids)].copy()
        raw["display_name"] = raw["dim_driver_id"].map(DRIVER_NAMES).fillna(raw["driver_full_name"])

    from zones import calc_true_rph as _calc_rph

    enriched = enrich_zones(raw)
    enriched = _calc_rph(enriched)
    enriched = enriched.sort_values(["dim_driver_id","pickedup_trip_datetime"])

    # Classify each trip's pickup context based on previous dropoff zone
    enriched["prev_dzone"] = enriched.groupby("dim_driver_id")["dropoff_zone"].shift(1)

    def _ctx(row):
        if pd.isna(row["prev_dzone"]):  return "Shift start"
        p, c = int(row["prev_dzone"]), int(row["pickup_zone"])
        if c == 1 and p == 1:           return "Z1 chain"
        if c == 1 and p >= 3:           return "Return to Z1 (empty reposition)"
        if c == 1 and p == 2:           return "Z1 via Z2"
        if c >= 3 and p == 1:           return "Left Z1 for outer zone"
        return "Other"

    enriched["context"] = enriched.apply(_ctx, axis=1)

    # ── SECTION 1: What does each strategy actually earn? ────────────────────
    st.subheader("1 — Which strategy earns more per hour?")
    st.caption("Full-cycle RPH = fare ÷ (inter-trip gap + pickup time + ride time)")

    ctx_summary = (
        enriched[enriched["context"] != "Shift start"]
        .groupby("context")
        .agg(
            trips=("true_rph", "count"),
            avg_rph=("true_rph", "mean"),
            avg_fare=("trip_price_in_pound", "mean"),
            avg_dist=("distance_in_miles", "mean"),
        ).round(2).reset_index()
        .sort_values("avg_rph", ascending=False)
    )

    color_map = {
        "Z1 chain":                    "#22c55e",
        "Z1 via Z2":                   "#86efac",
        "Other":                       "#6b7280",
        "Left Z1 for outer zone":      "#f59e0b",
        "Return to Z1 (empty reposition)": "#ef4444",
    }

    fig_ctx = px.bar(
        ctx_summary, x="avg_rph", y="context", orientation="h",
        color="context", color_discrete_map=color_map,
        text="avg_rph",
        title="Full-cycle RPH by trip sequence context",
        labels={"context": "", "avg_rph": "Full-cycle RPH £/hr"},
    )
    fig_ctx.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
    fig_ctx.add_vline(x=22.9, line_dash="dash", line_color="white",
                      annotation_text="Fleet avg £22.9", annotation_position="top")
    fig_ctx.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_ctx, use_container_width=True)

    st.info(
        "📌 **Z1 chaining outperforms hub-and-spoke.** Returning to Zone 1 after a long outer run "
        "(empty repositioning) produces the lowest RPH in the dataset — the dead miles kill the economics. "
        "A Zone 1→Zone 6 airport run is great for that trip, but the empty return leg makes the "
        "sequence average worse than just staying in Zone 1."
    )
    st.dataframe(
        ctx_summary.rename(columns={
            "context": "Sequence Context", "trips": "Trips",
            "avg_rph": "Avg RPH £", "avg_fare": "Avg Fare £", "avg_dist": "Avg Distance mi"
        }),
        use_container_width=True, hide_index=True
    )

    # ── SECTION 2: Per-driver strategy profile ────────────────────────────────
    st.divider()
    st.subheader("2 — What is each driver actually doing?")

    driver_profiles = []
    for did, grp in enriched.groupby("dim_driver_id"):
        if did not in selected_ids:
            continue
        name = DRIVER_NAMES.get(did, str(did))
        total = len(grp)
        z1 = grp[grp["pickup_zone"] == 1]
        chains = grp[grp["context"] == "Z1 chain"]
        returns = grp[grp["context"] == "Return to Z1 (empty reposition)"]
        left_z1 = grp[grp["context"] == "Left Z1 for outer zone"]

        driver_profiles.append({
            "Driver": name,
            "Total trips": total,
            "Z1 pickup %": round(len(z1) / total * 100, 0),
            "Z1→outer %": round((z1["dropoff_zone"] >= 3).sum() / len(z1) * 100, 0) if len(z1) else 0,
            "Z1 chain trips": len(chains),
            "Z1 chain RPH £": round(chains["true_rph"].mean(), 2) if len(chains) else 0,
            "Reposition trips": len(returns),
            "Reposition RPH £": round(returns["true_rph"].mean(), 2) if len(returns) else 0,
            "Overall RPH £": round(grp["trip_price_in_pound"].sum() / (grp["gap_mins"].fillna(0).sum() +
                             grp["pickup_duration_in_min"].fillna(0).sum() +
                             grp["pob_duration_in_min"].sum()) * 60, 2),
        })

    prof_df = pd.DataFrame(driver_profiles).sort_values("Overall RPH £", ascending=False)
    st.dataframe(prof_df, use_container_width=True, hide_index=True)

    # Scatter: Z1 chain % vs overall RPH
    prof_df["Z1 chain %"] = (prof_df["Z1 chain trips"] / prof_df["Total trips"] * 100).round(1)
    fig_scatter = px.scatter(
        prof_df, x="Z1 chain %", y="Overall RPH £",
        text="Driver", size="Total trips",
        color="Overall RPH £", color_continuous_scale="RdYlGn",
        title="Zone 1 chaining intensity vs overall RPH — more chaining ≠ more earnings",
        labels={"Z1 chain %": "% of trips that are Z1 chains", "Overall RPH £": "Overall RPH £/hr"},
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.add_hline(y=prof_df["Overall RPH £"].mean(), line_dash="dash",
                          line_color="orange", annotation_text="group avg")
    fig_scatter.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── SECTION 3: The repositioning penalty visualised ───────────────────────
    st.divider()
    st.subheader("3 — The repositioning penalty: what a long trip sequence actually earns")
    st.markdown("""
A Zone 1→Zone 6 airport run earns **£30.98/hr for that trip**. But look at the full sequence:

| Leg | Time | Earnings | RPH |
|---|---|---|---|
| Z1→Z6 trip (airport run) | 59 min ride + 11 min pickup | £41 | £30.98/hr |
| Z6→Z1 empty reposition | ~40–50 min driving back | £0 | £0/hr |
| Next Z1 pickup (now late, gap long) | Normal Z1 trip | £11-15 | Low |
| **Sequence average** | **~150 min total** | **~£52** | **~£20.80/hr** |

vs. staying in Zone 1:

| Leg | Time | Earnings | RPH |
|---|---|---|---|
| Z1 trip × 3 | 3 × 40 min cycle | 3 × £12 = £36 | £22.36/hr |
| **120 min total** | | **£36** | **£22.36/hr** |

The airport run sequence *looks* like it should win, but the empty return leg turns a £31/hr trip into a £21/hr sequence.
The only way the hub-and-spoke model beats Z1 chaining is if you can **pick up a return passenger at the outer zone** — avoiding the empty reposition entirely.
""")

    # ── SECTION 4: What should actually change ───────────────────────────────
    st.divider()
    st.subheader("4 — So what's the right model?")
    st.markdown("""
The data rules out two simple answers:

**❌ "Just chain Zone 1 trips all day"** — Bal Jamts does this most (45% Z1, 1,737 chains) and has the lowest overall RPH. Chaining without selectivity on fare/distance just fills the shift with sub-£12 trips in gridlock.

**❌ "Do long airport runs back and forth all day"** — the empty return leg produces £10–13/hr, which is worse than everything else. The sequence average comes out around £20–21/hr — the same as just chaining.

**✅ What Marius Norvaisas (best overall RPH) actually does:**
- Only 18% of trips from Zone 1 (doesn't camp there)
- When he chains in Z1, he earns £25.92/hr on those chains — meaning he's being *selective* about which Z1 trips he takes
- He takes long outer-zone trips but only when the return pick-up opportunity is real (Heathrow → passenger back to city)
- He has the most trips of anyone (6,977) — high volume, spread across zones, not concentrated

**The model that works:** Be in Zone 1 for high-density pings and fast turnarounds, accept Z1 trips that clear the £12 fare / 2 mile threshold, take long outer trips only when you can pick up a return passenger. Never reposition empty across more than one zone.
""")

# ── Airport Run Model ─────────────────────────────────────────────────────────
elif page == "Airport Run Model":
    st.title("Airport Run Model")
    st.caption("Z1/Z2 base strategy + timed airport runs. Heathrow: quick daytime in-out. Gatwick: evening chain through south London.")

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading trip data..."):
        raw = db.load_zone_trips()
        raw = raw[raw["dim_driver_id"].isin(selected_ids)].copy()

    enriched = enrich_zones(raw)
    enriched = enriched.sort_values(["dim_driver_id", "pickedup_trip_datetime"])
    enriched["pickedup_trip_datetime"] = pd.to_datetime(enriched["pickedup_trip_datetime"])
    enriched["dropoff_trip_datetime"]  = pd.to_datetime(enriched["dropoff_trip_datetime"])
    enriched["pickup_zone"]  = enriched["pickup_zone"].astype(int)
    enriched["dropoff_zone"] = enriched["dropoff_zone"].astype(int)

    # Airport detection by coordinate bounding box (more precise than zone alone)
    enriched["is_lhr"] = (
        enriched["dropoff_lat"].between(51.45, 51.49) &
        enriched["dropoff_lon"].between(-0.50, -0.42)
    )
    enriched["is_lgw"] = (
        enriched["dropoff_lat"].between(51.13, 51.18) &
        enriched["dropoff_lon"].between(-0.22, -0.14)
    )
    enriched["is_airport_drop"] = enriched["is_lhr"] | enriched["is_lgw"]
    enriched["airport_name"] = np.where(
        enriched["is_lhr"], "Heathrow (LHR)",
        np.where(enriched["is_lgw"], "Gatwick (LGW)", "Other Zone 6")
    )

    enriched["next_pzone"]  = enriched.groupby("dim_driver_id")["pickup_zone"].shift(-1)
    enriched["next_fare"]   = enriched.groupby("dim_driver_id")["trip_price_in_pound"].shift(-1)
    enriched["next_pickup"] = enriched.groupby("dim_driver_id")["pickedup_trip_datetime"].shift(-1)
    enriched["gap_to_next"] = (
        enriched["next_pickup"] - enriched["dropoff_trip_datetime"]
    ).dt.total_seconds().div(60)

    airport_drops = enriched[enriched["is_airport_drop"]].copy()
    lhr_drops     = enriched[enriched["is_lhr"]].copy()
    lgw_drops     = enriched[enriched["is_lgw"]].copy()

    # Classify what happens after the airport drop
    def _return_type(row):
        if pd.isna(row["next_pzone"]): return "End of shift"
        z = int(row["next_pzone"])
        if z >= 5: return "Got return passenger (Z5/Z6)"
        if z <= 2: return "Repositioned empty to Z1/2"
        return "Mid-zone pickup (Z3/4)"

    airport_drops["return_type"] = airport_drops.apply(_return_type, axis=1)
    lhr_drops["return_type"]     = lhr_drops.apply(_return_type, axis=1)
    lgw_drops["return_type"]     = lgw_drops.apply(_return_type, axis=1)

    # ── SECTION 1: Does it already work? ─────────────────────────────────────
    st.subheader("1 — The top drivers are already doing this — 67% get a return passenger")

    col1, col2, col3, col4 = st.columns(4)
    got_return    = airport_drops[airport_drops["return_type"] == "Got return passenger (Z5/Z6)"]
    repositioned  = airport_drops[airport_drops["return_type"] == "Repositioned empty to Z1/2"]
    total_drops   = len(airport_drops)

    col1.metric("Airport drops (total)",      f"{total_drops}")
    col2.metric("Got return passenger",        f"{len(got_return)} ({len(got_return)/total_drops*100:.0f}%)", "Avg wait 20 min")
    col3.metric("Avg return fare",             f"£{got_return['next_fare'].mean():.2f}")
    col4.metric("Repositioned empty",          f"{len(repositioned)} ({len(repositioned)/total_drops*100:.0f}%)")

    return_summary = (
        airport_drops[airport_drops["return_type"] != "End of shift"]
        .groupby("return_type")
        .agg(
            trips=("trip_price_in_pound", "count"),
            avg_next_fare=("next_fare", "mean"),
            avg_gap_mins=("gap_to_next", lambda x: x[x < 240].mean()),
        ).round(1).reset_index()
    )
    color_map_rt = {
        "Got return passenger (Z5/Z6)": "#22c55e",
        "Mid-zone pickup (Z3/4)":       "#f59e0b",
        "Repositioned empty to Z1/2":   "#ef4444",
    }
    fig_rt = px.pie(
        return_summary, names="return_type", values="trips",
        color="return_type", color_discrete_map=color_map_rt,
        title="What happens after an airport drop?",
    )
    fig_rt.update_layout(height=340)
    st.plotly_chart(fig_rt, use_container_width=True)

    st.dataframe(
        return_summary.rename(columns={
            "return_type": "What happened next", "trips": "Trips",
            "avg_next_fare": "Avg return fare £", "avg_gap_mins": "Avg wait (min)"
        }),
        use_container_width=True, hide_index=True
    )

    # ── SECTION 2: Economics of the paired run ────────────────────────────────
    st.divider()
    st.subheader("2 — The economics: paired airport run vs Z1 chaining")

    # Airport drop leg
    avg_drop_fare = airport_drops["trip_price_in_pound"].mean()
    avg_drop_ride = airport_drops["pob_duration_in_min"].mean()
    avg_drop_gap  = airport_drops["gap_mins"].mean() if "gap_mins" in airport_drops.columns else 15.0
    avg_wait      = got_return["gap_to_next"][got_return["gap_to_next"] < 120].mean()
    avg_ret_fare  = got_return["next_fare"].mean()
    avg_ret_ride  = 55.0  # ~55 min return from Heathrow to Zone 1
    avg_ret_pickup= 10.0

    total_time_paired = avg_drop_gap + 10 + avg_drop_ride + avg_wait + avg_ret_pickup + avg_ret_ride
    total_earn_paired = avg_drop_fare + avg_ret_fare
    rph_paired        = total_earn_paired / (total_time_paired / 60)

    # Z1 chaining for the same duration
    z1_cycle = 40.5  # from earlier analysis (gap 14 + pickup 5 + ride 21)
    z1_fare   = 11.52
    n_z1_trips= total_time_paired / z1_cycle
    rph_z1    = (n_z1_trips * z1_fare) / (total_time_paired / 60)

    comparison_df = pd.DataFrame({
        "Strategy": [
            "Paired airport run (drop + return)",
            "Z1 chaining (same window)",
            "Airport run, empty return",
        ],
        "Total earnings": [round(total_earn_paired, 2), round(n_z1_trips * z1_fare, 2), round(avg_drop_fare, 2)],
        "Time (min)":     [round(total_time_paired), round(total_time_paired), round(avg_drop_gap + 10 + avg_drop_ride + 45)],
        "RPH":            [round(rph_paired, 2), round(rph_z1, 2), round(avg_drop_fare / ((avg_drop_gap + 10 + avg_drop_ride + 45) / 60), 2)],
    })

    fig_econ = px.bar(
        comparison_df, x="Strategy", y="RPH",
        color="RPH", color_continuous_scale="RdYlGn",
        text="RPH",
        title="Full-cycle RPH: paired airport run vs alternatives",
        labels={"RPH": "£/hr", "Strategy": ""},
    )
    fig_econ.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
    fig_econ.add_hline(y=22.9, line_dash="dash", line_color="orange",
                       annotation_text="Fleet avg £22.9/hr", annotation_position="top right")
    fig_econ.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_econ, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Paired run RPH",       f"£{rph_paired:.2f}/hr",  f"+£{rph_paired-22.9:.2f} vs fleet avg")
    c2.metric("Z1 chaining RPH",      f"£{rph_z1:.2f}/hr",      "Baseline")
    c3.metric("Empty return RPH",     f"£{avg_drop_fare/((avg_drop_gap+10+avg_drop_ride+45)/60):.2f}/hr", "Worst case")

    # ── SECTION 3: When are drops happening vs optimal windows ───────────────
    st.divider()
    st.subheader("3 — When do drivers drop at airports now vs when they should")

    hour_dist = airport_drops.groupby("trips_hr").size().reset_index(name="drops")

    # Approximate big arrival windows at Heathrow
    arrival_windows = pd.DataFrame({
        "hour": [5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20],
        "window_label": [
            "Long-haul overnight (US/Asia)",
            "Long-haul overnight (US/Asia)",
            "Long-haul overnight (US/Asia)",
            "Morning European peak",
            "Transatlantic (East Coast)",
            "Transatlantic (East Coast)",
            "Transatlantic (East Coast)",
            "Middle East / Gulf",
            "Evening European",
            "Evening European",
            "Long-haul evening (US/Asia)",
        ]
    })

    fig_hr = px.bar(
        hour_dist, x="trips_hr", y="drops",
        color="drops", color_continuous_scale="Blues",
        text="drops",
        title="Airport drops by hour of day (when drivers are currently going)",
        labels={"trips_hr": "Hour", "drops": "Drops"},
    )
    fig_hr.update_traces(texttemplate="%{text}", textposition="outside")

    # Shade arrival windows
    for window_hour in [5, 6, 7, 12, 13, 14, 18, 19]:
        fig_hr.add_vrect(
            x0=window_hour - 0.4, x1=window_hour + 0.4,
            fillcolor="rgba(34,197,94,0.15)", line_width=0,
            annotation_text="Arrivals\nwindow" if window_hour == 6 else "",
            annotation_position="top",
        )

    fig_hr.update_layout(height=360, coloraxis_showscale=False, xaxis=dict(dtick=1))
    st.plotly_chart(fig_hr, use_container_width=True)

    st.info(
        "📌 **Drops peak at 5–6am** — drivers are taking early departures passengers to the airport and "
        "picking up overnight long-haul arrivals from the US/Middle East. "
        "The **afternoon window (12–3pm)** — when transatlantic East Coast flights land — is significantly "
        "underutilised. Flight data would let us identify and fill that gap deliberately."
    )

    # ── SECTION 4: The flight data integration plan ───────────────────────────
    st.divider()
    st.subheader("4 — The flight data integration")

    st.markdown("""
**What we need:**

Heathrow has an official developer API (`developer.heathrow.com`) with real-time arrivals — free to register.
Third-party options (AviationEdge, FlightLabs) also work and include aircraft type, which is the key field.

**The model:**

For each arriving flight at Heathrow, we want:
- `scheduled_arrival` — when the wheels touch down
- `aircraft_type` — determines passenger count
  - A380: ~500 seats (Emirates, Qantas, British Airways)
  - B777: ~350 seats (common long-haul)
  - B787: ~290 seats
  - A350: ~350 seats
- `terminal` — T2, T3, T4, T5 (determines which arrivals bay to park at)
- `origin` — long-haul from USA/Middle East/Asia = higher-value passengers

**The timing calculation (driver leaves Zone 1 when?):**

```
pickup_window  = arrival_time + 45 min  (customs clearance estimate)
drive_to_LHR   = 45 min from Zone 1 (off-peak) → 60 min (peak)
depart_trigger = pickup_window - drive_to_LHR
               = arrival_time + 45 - 50 = arrival_time - 5 min
```

So for a flight landing at 2:30pm:
- Passengers through customs ≈ 3:15pm
- Driver leaves Zone 1 at 2:25pm
- Arrives Heathrow ≈ 3:10pm
- Picks up at 3:15pm — perfect

**Where this lives:**

This notification goes into the **Driver Portal PWA** — the push notification system is already built (Phase 2 complete).
The new trigger would be flight-based rather than stationary-position-based:

> *"Big arrival window at Heathrow T5 in 45 min (Emirates EK003 — 496 seats). Leave now to match the pickup window."*

The driver taps it, gets navigation to T5 arrivals. If they already have a departures passenger heading to Heathrow, even better — the drop + pick-up happens in one sequence.

**Airports to cover:**
- Heathrow (LHR) — primary daytime model, 20–25 min from Zone 1 off-peak. High-value pax, regular long-haul schedule.
- Gatwick (LGW) — evening chain model only. Too far for a standalone run; only viable when the driver is already heading south through Z3/Z4. See Section 6.
- Stansted (STN) — 50+ min from Zone 1, low priority, budget carriers only.
""")

    # ── SECTION 5: The recommended daily model ────────────────────────────────
    st.divider()
    st.subheader("5 — The recommended daily model")

    model_df = pd.DataFrame({
        "Time block": ["06:00–12:00", "12:00–14:00", "14:00–15:30", "15:30–18:00", "18:00–20:00", "20:00–23:00 (optional)"],
        "Activity": [
            "Z1/Z2 chaining — high ping density, £12+ threshold",
            "Continue Z1/Z2 OR depart for Heathrow if afternoon arrival window (transatlantic landing)",
            "Heathrow: drop departures + pick up arrivals passenger",
            "Return to Z1/Z2 with arrivals passenger, resume chaining",
            "Evening peak Z1/Z2 — accept south-bound pings if in Z3/Z4 territory",
            "Gatwick chain (optional): south-bound through Z3/4 → Gatwick drop → overnight arrival pickup",
        ],
        "Expected RPH": ["£22/hr", "£22/hr or trigger", "£27/hr (paired)", "£27/hr (paired)", "£24/hr", "£26/hr (est.)"],
    })
    st.table(model_df)

    st.success(
        "**The key rule:** Never reposition to Heathrow empty. Only go if you have a departures "
        "passenger heading that way, OR if the arrival window is large enough (300+ seat aircraft) "
        "to justify the positioning cost. The data shows 67% of airport drops already produce a "
        "return passenger — flight data should push that above 80% by targeting the right windows."
    )

    # ── SECTION 6: Gatwick — the evening chain ────────────────────────────────
    st.divider()
    st.subheader("6 — Gatwick: the evening chain model")
    st.caption("Gatwick is 45 min south — too far to reposition for. But if a driver is already heading south dropping someone home, the economics change completely.")

    col1, col2, col3, col4 = st.columns(4)
    lgw_total    = len(lgw_drops)
    lgw_return   = lgw_drops[lgw_drops["return_type"] == "Got return passenger (Z5/Z6)"]
    lgw_empty    = lgw_drops[lgw_drops["return_type"] == "Repositioned empty to Z1/2"]

    if lgw_total > 0:
        col1.metric("Gatwick drops",          f"{lgw_total}")
        col2.metric("Got return passenger",   f"{len(lgw_return)} ({len(lgw_return)/lgw_total*100:.0f}%)" if lgw_total > 0 else "–")
        col3.metric("Avg return fare (LGW)",  f"£{lgw_return['next_fare'].mean():.2f}" if len(lgw_return) > 0 else "–")
        col4.metric("Avg Gatwick drop fare",  f"£{lgw_drops['trip_price_in_pound'].mean():.2f}")
    else:
        col1.metric("Gatwick drops", "0")
        col2.metric("Note", "No Gatwick trips in dataset")
        col3.metric("LHR drops", f"{len(lhr_drops)}")
        col4.metric("LHR return rate", f"{len(got_return)/max(len(lhr_drops),1)*100:.0f}%")

    # Evening pre-Gatwick chaining: are drivers passing through Z3/Z4 on the way?
    # Trips that end at Gatwick — what was the previous trip's dropoff zone?
    lgw_idx = lgw_drops.index
    prev_zones = []
    for idx in lgw_idx:
        driver = lgw_drops.loc[idx, "dim_driver_id"]
        driver_trips = enriched[enriched["dim_driver_id"] == driver].sort_values("pickedup_trip_datetime")
        pos = driver_trips.index.get_loc(idx) if idx in driver_trips.index else -1
        if pos > 0:
            prev_zones.append(int(driver_trips.iloc[pos - 1]["dropoff_zone"]))

    st.markdown("""
**The Gatwick evening chain concept:**

Gatwick sits directly south of central London — a driver heading from Zone 1/2 to Gatwick passes through Zone 3 (Brixton, Clapham, Streatham) and Zone 4 (Croydon, Sutton) on the way.

The model:
1. **18:00–20:00** — Z1/Z2 evening peak, take all trips going south or southeast
2. **~20:00** — Pick up a departures passenger heading to Gatwick (naturally heading in the right direction)
3. **20:30** — Arrive Gatwick. Wait for the overnight arrivals wave (Ryanair, EasyJet, TUI long-haul charters)
4. **~21:30–23:00** — Pick up an arrivals passenger heading back to London (Zone 2–4)
5. **Return** — 45+ mile return fare into the city

**Why Gatwick is different from Heathrow:**

| | Heathrow | Gatwick |
|---|---|---|
| Distance from Z1 | ~20 miles west | ~28 miles south |
| Drive time (off-peak) | 40–50 min | 45–55 min |
| Best return window | Morning long-haul arrivals | Evening charter/budget arrivals |
| Typical return destination | Zone 1/2 (city centre) | Zone 2–4 (south London) |
| Return fare (London) | £34 avg | £38–42 expected (longer trip) |

**The key constraint:** Gatwick doesn't work for a quick daytime in-out run — it's too far and the arrivals are too unpredictable. The evening-to-overnight window (20:00–23:00) is when charter and long-haul budget flights land. The evening chain pattern — south-dropping pings from Zone 1 through Zone 3/4 toward Gatwick — is the way to make the repositioning cost free.

**Driver Portal integration:**

The Gatwick trigger would fire differently from Heathrow:
> *"You're in Zone 3/4 heading south — Gatwick arrivals window opens in 60 min (TUI charter, 300 pax, landing 21:15). Accept south-bound pings only."*

This is a harder product problem than Heathrow because the chain has to happen organically through south-bound trip routing, not a single departure decision.
""")

    if lgw_total > 0:
        lgw_hour = lgw_drops.groupby("trips_hr").size().reset_index(name="drops")
        fig_lgw = px.bar(
            lgw_hour, x="trips_hr", y="drops",
            title="Gatwick drops by hour (when drivers currently go)",
            labels={"trips_hr": "Hour", "drops": "Drops"},
            text="drops",
        )
        fig_lgw.update_traces(texttemplate="%{text}", textposition="outside")
        fig_lgw.update_layout(height=300, xaxis=dict(dtick=1))
        st.plotly_chart(fig_lgw, use_container_width=True)
    else:
        st.info(
            "No Gatwick coordinate-matches in the current dataset — the 74 Gatwick drops from earlier analysis "
            "may have been classified under the broader Zone 6 bucket. "
            "The coordinate bounding box for Gatwick (lat 51.13–51.18, lon -0.22 to -0.14) can be adjusted if needed."
        )

    st.markdown("""
**Verdict on Gatwick vs Heathrow:**

Heathrow is the primary airport model — closer, higher passenger volumes, better daytime fit. Gatwick is viable as an **evening shift extension** for drivers who are already working south London in the evening. Don't ask daytime Z1/Z2 drivers to detour south for it — the repositioning cost kills the economics. But for a driver whose natural evening territory is Clapham → Croydon → Brixton, the Gatwick chain adds a high-fare anchor at the end of the shift.
""")

# ── Zone 3 Deep Dive ──────────────────────────────────────────────────────────
elif page == "Zone 3 Deep Dive":
    st.title("Zone 3 Deep Dive")
    st.caption("Zone 3 has the worst average True RPH — but is it always a dead zone, or only at certain times?")

    # Zone 3 trips by hour with raw RPH
    z3_hour_data = pd.DataFrame([
        {"hour": 0,  "trips":156, "avg_fare":13.20, "avg_dist":6.4, "avg_ride_mins":25},
        {"hour": 1,  "trips":144, "avg_fare":13.00, "avg_dist":6.9, "avg_ride_mins":22},
        {"hour": 2,  "trips":122, "avg_fare":14.20, "avg_dist":7.4, "avg_ride_mins":22},
        {"hour": 3,  "trips": 91, "avg_fare":21.20, "avg_dist":11.3,"avg_ride_mins":30},
        {"hour": 4,  "trips":157, "avg_fare":17.00, "avg_dist":11.1,"avg_ride_mins":30},
        {"hour": 5,  "trips":205, "avg_fare":18.40, "avg_dist": 9.3,"avg_ride_mins":29},
        {"hour": 6,  "trips":283, "avg_fare":17.90, "avg_dist": 8.2,"avg_ride_mins":31},
        {"hour": 7,  "trips":233, "avg_fare":16.50, "avg_dist": 6.1,"avg_ride_mins":31},
        {"hour": 8,  "trips":187, "avg_fare":14.30, "avg_dist": 5.5,"avg_ride_mins":27},
        {"hour": 9,  "trips":223, "avg_fare":13.00, "avg_dist": 6.2,"avg_ride_mins":28},
        {"hour":10,  "trips":216, "avg_fare":13.30, "avg_dist": 6.0,"avg_ride_mins":29},
        {"hour":11,  "trips":202, "avg_fare":15.10, "avg_dist": 6.8,"avg_ride_mins":33},
        {"hour":12,  "trips":173, "avg_fare":14.20, "avg_dist": 5.9,"avg_ride_mins":32},
        {"hour":13,  "trips":184, "avg_fare":14.30, "avg_dist": 6.0,"avg_ride_mins":32},
        {"hour":14,  "trips":191, "avg_fare":13.40, "avg_dist": 5.8,"avg_ride_mins":31},
        {"hour":15,  "trips":236, "avg_fare":13.80, "avg_dist": 5.4,"avg_ride_mins":30},
        {"hour":16,  "trips":286, "avg_fare":13.60, "avg_dist": 5.2,"avg_ride_mins":31},
        {"hour":17,  "trips":265, "avg_fare":15.00, "avg_dist": 6.6,"avg_ride_mins":34},
        {"hour":18,  "trips":328, "avg_fare":13.60, "avg_dist": 5.7,"avg_ride_mins":29},
        {"hour":19,  "trips":353, "avg_fare":11.90, "avg_dist": 5.1,"avg_ride_mins":24},
        {"hour":20,  "trips":279, "avg_fare":11.30, "avg_dist": 5.9,"avg_ride_mins":23},
        {"hour":21,  "trips":251, "avg_fare":11.50, "avg_dist": 6.1,"avg_ride_mins":24},
        {"hour":22,  "trips":263, "avg_fare":11.60, "avg_dist": 6.0,"avg_ride_mins":24},
        {"hour":23,  "trips":160, "avg_fare":11.70, "avg_dist": 5.9,"avg_ride_mins":23},
    ])
    z3_hour_data["raw_rph"] = (z3_hour_data["avg_fare"] / (z3_hour_data["avg_ride_mins"]/60)).round(1)
    z3_hour_data["period"] = z3_hour_data["hour"].apply(
        lambda h: "Night (00-06)" if h < 6 else "Day (09-17)" if 9 <= h <= 17 else "Evening (17-23)" if h >= 17 else "Morning (06-09)"
    )

    # KPI comparison
    night_rph = z3_hour_data[z3_hour_data["hour"] < 6]["raw_rph"].mean()
    day_rph   = z3_hour_data[(z3_hour_data["hour"] >= 9) & (z3_hour_data["hour"] <= 17)]["raw_rph"].mean()
    eve_rph   = z3_hour_data[z3_hour_data["hour"] >= 17]["raw_rph"].mean()
    z1_rph    = 39.0  # from run_analysis

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Zone 3 — Night RPH",   f"£{night_rph:.0f}/hr", f"{'▲' if night_rph>z1_rph else '▼'} Zone 1 = £{z1_rph:.0f}/hr")
    c2.metric("Zone 3 — Day RPH",     f"£{day_rph:.0f}/hr",   f"▼ {z1_rph-day_rph:.0f} below Zone 1")
    c3.metric("Zone 3 — Evening RPH", f"£{eve_rph:.0f}/hr",   f"▼ {z1_rph-eve_rph:.0f} below Zone 1")
    c4.metric("03:00 peak",           "£43/hr",                "Best single hour in Zone 3")

    st.divider()

    # Raw RPH by hour — bar coloured by whether it beats day average
    z3_hour_data["colour"] = z3_hour_data["raw_rph"].apply(
        lambda r: "Above £30 (worth it)" if r >= 30 else "£26–30 (borderline)" if r >= 26 else "Below £26 (avoid)"
    )
    fig_z3 = px.bar(z3_hour_data, x="hour", y="raw_rph",
                    color="colour",
                    color_discrete_map={"Above £30 (worth it)":"#22c55e",
                                        "£26–30 (borderline)":"#f59e0b",
                                        "Below £26 (avoid)":"#ef4444"},
                    text="raw_rph",
                    labels={"hour":"Hour of Day","raw_rph":"Raw RPH (fare ÷ ride time)"},
                    title="Zone 3 — Raw RPH by Hour of Day")
    fig_z3.add_hline(y=z1_rph, line_dash="dash", line_color="#3b82f6",
                     annotation_text=f"Zone 1 avg (£{z1_rph:.0f}/hr)", annotation_position="top right")
    fig_z3.update_traces(texttemplate="£%{text:.0f}", textposition="outside")
    fig_z3.update_layout(height=440, showlegend=True,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_z3, use_container_width=True)

    st.info("📌 **Zone 3 verdict:**\n"
            "- **00:00–06:00**: Earns £32–43/hr — **comparable to or better than Zone 1 Zone daytime**. Long trips (avg 11mi at 03:00) going back into the city.\n"
            "- **09:00–17:00**: £26–28/hr + long waits (32min avg). This is the dead zone. Avoid.\n"
            "- **17:00–23:00**: £27–30/hr. Moderate — better than daytime but Zone 1 or 2 still outperform.\n\n"
            "**The problem isn't Zone 3. It's Zone 3 in the middle of the day.**")

    st.divider()

    # Fare vs distance scatter by hour
    st.subheader("Fare vs Distance in Zone 3 — how does the trip quality change?")
    fig_sc = px.scatter(z3_hour_data, x="avg_dist", y="avg_fare", size="trips",
                        color="raw_rph", color_continuous_scale="RdYlGn",
                        text="hour",
                        labels={"avg_dist":"Avg Distance (mi)","avg_fare":"Avg Fare £",
                                "raw_rph":"Raw RPH £","trips":"Trip Count"},
                        title="Each dot = one hour of the day. Colour = Raw RPH. Size = trip volume.")
    fig_sc.update_traces(textposition="top center")
    fig_sc.update_layout(height=450)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Night hours (top-right) = longer distance, higher fare, higher RPH. Day hours (bottom-left) = short, cheap, lots of volume but poor efficiency.")

    st.divider()

    # Zone 3 areas
    st.subheader("Most common Zone 3 pickup areas")
    areas = pd.DataFrame([
        {"Area": "Walthamstow",  "Trips": 185, "Avg Fare £": 11.19},
        {"Area": "Wandsworth",   "Trips": 155, "Avg Fare £": 13.36},
        {"Area": "Stratford",    "Trips": 133, "Avg Fare £": 11.90},
        {"Area": "Greenwich",    "Trips": 104, "Avg Fare £": 13.13},
        {"Area": "Lewisham",     "Trips": 104, "Avg Fare £": 11.26},
        {"Area": "Tottenham",    "Trips":  65, "Avg Fare £": 12.10},
        {"Area": "East Ham",     "Trips":  61, "Avg Fare £": 11.07},
        {"Area": "Ealing",       "Trips":  49, "Avg Fare £": 14.03},
        {"Area": "Lambeth",      "Trips":  41, "Avg Fare £": 12.91},
        {"Area": "Barnet",       "Trips":  38, "Avg Fare £": 13.20},
        {"Area": "Brent",        "Trips":  33, "Avg Fare £": 13.82},
        {"Area": "Enfield",      "Trips":  19, "Avg Fare £":  9.09},
    ])
    fig_a = px.bar(areas.sort_values("Avg Fare £", ascending=True),
                   x="Avg Fare £", y="Area", orientation="h",
                   color="Avg Fare £", color_continuous_scale="YlOrRd",
                   text="Avg Fare £", size_max=40,
                   labels={"Area":""},
                   title="Avg Fare per Zone 3 Area — Ealing & Brent are the better Z3 pockets")
    fig_a.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
    fig_a.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig_a, use_container_width=True)
    st.caption("Enfield (£9.09) and Walthamstow/East Ham (£11) are the worst fare areas. Ealing, Brent, Wandsworth are the stronger Zone 3 pockets.")

    # Zone 3 dropoff destinations
    st.subheader("Where do Zone 3 trips end up?")
    z3_dropoff = pd.DataFrame({
        "Dropoff Zone": ["Zone 1","Zone 2","Zone 3","Zone 4","Zone 5","Zone 6"],
        "Trips":        [727, 1251, 1943, 779, 316, 254],
        "Pct":          [14,   24,   37,  15,   6,   5],
    })
    fig_do = px.bar(z3_dropoff, x="Dropoff Zone", y="Pct", color="Pct",
                    color_continuous_scale="Blues", text="Pct",
                    labels={"Pct":"% of Zone 3 trips"},
                    title="Zone 3 trips — where they drop off")
    fig_do.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_do.update_layout(coloraxis_showscale=False, height=360)
    st.plotly_chart(fig_do, use_container_width=True)
    st.caption("37% of Zone 3 trips stay in Zone 3 (local loop — low value). 14% go into Zone 1 (good, especially at night). 15% go outward to Zone 4+ (often airport adjacent).")

# ── Overview ─────────────────────────────────────────────────────────────────
elif page == "Overview":
    st.title("Top 10 Driver Overview")
    st.caption("Revenue per hour · All-time summary from performance data")

    # KPI bar chart
    fig = px.bar(
        overview_filtered.sort_values("rph", ascending=True),
        x="rph", y="display_name",
        orientation="h",
        color="rph",
        color_continuous_scale="YlOrRd",
        labels={"rph": "£ / hr (online)", "display_name": ""},
        title="Revenue Per Hour (£)",
        text="rph",
    )
    fig.update_traces(texttemplate="£%{text}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=10, r=40, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Metrics table
    st.subheader("Full Breakdown")
    display_cols = {
        "display_name": "Driver",
        "active_days": "Days Active",
        "total_rides": "Rides",
        "total_online_hrs": "Online Hrs",
        "rph": "£/hr",
        "rev_per_trip": "£/trip",
        "avg_util": "Util %",
        "avg_rating": "Rating",
        "avg_acceptance": "Accept %",
        "total_revenue": "Revenue £",
        "total_earnings": "Earnings £",
    }
    table = overview_filtered[list(display_cols.keys())].rename(columns=display_cols)
    table = table.sort_values("£/hr", ascending=False)
    st.dataframe(
        table.style.background_gradient(subset=["£/hr", "Util %"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    # Scatter: utilisation vs RPH
    st.subheader("Utilisation vs Revenue Per Hour")
    fig2 = px.scatter(
        overview_filtered,
        x="avg_util", y="rph",
        size="total_rides", color="avg_rating",
        hover_name="display_name",
        color_continuous_scale="Viridis",
        labels={"avg_util": "Avg Utilisation %", "rph": "£/hr", "avg_rating": "Rating", "total_rides": "Rides"},
        size_max=40,
    )
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ── Map View ─────────────────────────────────────────────────────────────────
elif page == "Map View":
    st.title("Trip Map")
    st.caption("Select a driver to see all their routes. Zone circles show approximate TfL zone boundaries.")

    # Controls row
    col_sel, col_n, col_src = st.columns([2, 1, 1])
    with col_sel:
        map_driver_name = st.selectbox("Driver", list(DRIVER_NAMES.values()))
    with col_n:
        trip_limit = st.selectbox("Show last N trips", [100, 250, 500, 1000], index=1)
    with col_src:
        show_zones = st.toggle("Show zone circles", value=True)

    map_driver_id = [k for k, v in DRIVER_NAMES.items() if v == map_driver_name][0]

    with st.spinner(f"Loading {trip_limit} trips for {map_driver_name}..."):
        raw = db.load_map_trips(map_driver_id, limit=trip_limit)

    if raw.empty:
        st.warning("No trips found for this driver.")
    else:
        # Parse coordinates
        raw = raw.copy()
        coords = raw["pickup_lat_long"].apply(parse_dms)
        raw["plat"] = [c[0] for c in coords]
        raw["plon"] = [c[1] for c in coords]
        dcoords = raw["dropoff_latlong"].apply(parse_dms)
        raw["dlat"] = [c[0] for c in dcoords]
        raw["dlon"] = [c[1] for c in dcoords]
        raw["pickup_zone"] = raw.apply(lambda r: assign_zone(r.plat, r.plon), axis=1)
        raw["dropoff_zone"] = raw.apply(lambda r: assign_zone(r.dlat, r.dlon), axis=1)
        valid_mask = raw.apply(
            lambda r: is_valid_london_trip(r.plat, r.plon, r.dlat, r.dlon), axis=1
        )
        valid = raw[valid_mask].copy()

        # Stats strip
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trips shown", len(valid))
        m2.metric("Avg fare", f"£{valid['trip_price_in_pound'].mean():.2f}")
        m3.metric("Avg distance", f"{valid['distance_in_miles'].mean():.1f} mi")
        m4.metric("Avg pickup speed", f"{valid['pickup_duration_in_min'].mean():.1f} min")

        # Build Folium map
        centre = [valid["plat"].mean(), valid["plon"].mean()]
        m = folium.Map(location=centre, zoom_start=11, tiles="CartoDB positron")

        # TfL Zone polygons from GeoJSON
        if show_zones:
            zone_colours = {1: "#ef4444", 2: "#f97316", 3: "#eab308",
                            4: "#22c55e", 5: "#3b82f6", 6: "#8b5cf6"}

            def zone_style(feature):
                z = feature["properties"]["zone"]
                return {
                    "color": zone_colours.get(z, "#9ca3af"),
                    "weight": 2.5,
                    "fillOpacity": 0.06,
                    "fillColor": zone_colours.get(z, "#9ca3af"),
                    "dashArray": "5,4" if z > 1 else None,
                }

            folium.GeoJson(
                GEOJSON_DATA,
                name="TfL Zones",
                style_function=zone_style,
                tooltip=folium.GeoJsonTooltip(fields=["zone", "description"], aliases=["Zone", ""]),
            ).add_to(m)

            # Zone labels — place on the northernmost point of each ring
            for feat in GEOJSON_DATA["features"]:
                z = feat["properties"]["zone"]
                label = feat["properties"].get("label", f"Zone {z}")
                coords = feat["geometry"]["coordinates"][0]
                # Pick the northernmost point (highest lat) for the label
                top = max(coords, key=lambda c: c[1])
                folium.Marker(
                    location=[top[1], top[0]],
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:11px;color:{zone_colours[z]};font-weight:700;'
                              f'background:rgba(255,255,255,0.75);padding:1px 4px;border-radius:3px;'
                              f'white-space:nowrap;text-shadow:none">{label}</div>'),
                        icon_size=(60, 18),
                        icon_anchor=(30, 18),
                    ),
                ).add_to(m)

        # Colour trips by source platform
        platform_colours = {"uber": "#000000", "bolt": "#34d399", "autocab": "#60a5fa"}

        # Draw trip lines + markers
        pickup_group = folium.FeatureGroup(name="Pickups", show=True)
        dropoff_group = folium.FeatureGroup(name="Dropoffs", show=True)
        route_group = folium.FeatureGroup(name="Routes", show=True)

        for _, row in valid.iterrows():
            src = (row.get("source") or "").lower()
            line_colour = platform_colours.get(src, "#6b7280")
            fare = row.trip_price_in_pound or 0
            dist = row.distance_in_miles or 0
            dt = str(row.pickedup_trip_datetime)[:16] if row.pickedup_trip_datetime else ""
            pz = int(row.pickup_zone) if row.pickup_zone else "?"
            dz = int(row.dropoff_zone) if row.dropoff_zone else "?"

            popup_html = f"""
            <div style='font-family:sans-serif;font-size:12px;min-width:180px'>
              <b>{map_driver_name}</b><br>
              {dt}<br>
              <b>£{fare:.2f}</b> &nbsp;·&nbsp; {dist:.1f} mi<br>
              <span style='color:#6b7280'>Z{pz} → Z{dz} &nbsp;·&nbsp; {src.title()}</span><br>
              <span style='color:#9ca3af;font-size:11px'>{(row.pickup_address or '')[:50]}</span><br>
              <span style='color:#9ca3af;font-size:11px'>→ {(row.dropoff_address or '')[:50]}</span>
            </div>
            """

            # Route line
            folium.PolyLine(
                locations=[[row.plat, row.plon], [row.dlat, row.dlon]],
                color=line_colour,
                weight=1.5,
                opacity=0.45,
                tooltip=f"£{fare:.2f} · {dist:.1f}mi · {src.title()}",
            ).add_to(route_group)

            # Pickup dot (green)
            folium.CircleMarker(
                location=[row.plat, row.plon],
                radius=4,
                color="#16a34a",
                fill=True,
                fill_color="#16a34a",
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"Pickup · £{fare:.2f}",
            ).add_to(pickup_group)

            # Dropoff dot (red)
            folium.CircleMarker(
                location=[row.dlat, row.dlon],
                radius=4,
                color="#dc2626",
                fill=True,
                fill_color="#dc2626",
                fill_opacity=0.8,
                tooltip=f"Dropoff · Z{dz}",
            ).add_to(dropoff_group)

        route_group.add_to(m)
        pickup_group.add_to(m)
        dropoff_group.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        # Legend
        legend_html = """
        <div style='position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                    padding:10px 14px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.2);
                    font-family:sans-serif;font-size:12px;line-height:1.8'>
          <b>Routes by platform</b><br>
          <span style='color:#000'>■</span> Uber &nbsp;
          <span style='color:#34d399'>■</span> Bolt &nbsp;
          <span style='color:#60a5fa'>■</span> Autocab<br>
          <span style='color:#16a34a'>●</span> Pickup &nbsp;
          <span style='color:#dc2626'>●</span> Dropoff
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width="100%", height=680, returned_objects=[])

        # Trip table below map
        with st.expander("Trip list", expanded=False):
            display = valid[["pickedup_trip_datetime", "pickup_address", "dropoff_address",
                              "trip_price_in_pound", "distance_in_miles", "pob_duration_in_min",
                              "pickup_zone", "dropoff_zone", "source"]].copy()
            display.columns = ["Datetime", "Pickup", "Dropoff", "Fare £",
                                "Distance mi", "Ride mins", "Pickup Zone", "Dropoff Zone", "Platform"]
            display["Pickup Zone"] = display["Pickup Zone"].apply(lambda z: f"Zone {int(z)}" if pd.notna(z) else "?")
            display["Dropoff Zone"] = display["Dropoff Zone"].apply(lambda z: f"Zone {int(z)}" if pd.notna(z) else "?")
            st.dataframe(display.sort_values("Datetime", ascending=False), use_container_width=True, hide_index=True)

# ── Time Patterns ─────────────────────────────────────────────────────────────
elif page == "Time Patterns":
    st.title("Time of Day Patterns")
    st.caption("When do top drivers work — and when are they most productive?")

    with st.spinner("Loading trip data..."):
        hourly_df = db.load_hourly_trips()
    hourly_filtered = hourly_df[hourly_df["dim_driver_id"].isin(selected_ids)].copy()
    hourly_filtered["display_name"] = hourly_filtered["dim_driver_id"].map(DRIVER_NAMES).fillna(hourly_filtered["driver_full_name"])

    # Aggregate across all selected drivers
    by_hour = hourly_filtered.groupby("hour_of_day").agg(
        trips=("trips", "sum"),
        avg_fare=("avg_fare", "mean"),
        avg_distance=("avg_distance", "mean"),
        avg_ride_mins=("avg_ride_mins", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trip Volume by Hour")
        fig = px.bar(
            by_hour, x="hour_of_day", y="trips",
            color="trips", color_continuous_scale="Oranges",
            labels={"hour_of_day": "Hour of Day", "trips": "Total Trips"},
        )
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Fare by Hour (£)")
        fig2 = px.line(
            by_hour, x="hour_of_day", y="avg_fare",
            markers=True,
            labels={"hour_of_day": "Hour of Day", "avg_fare": "Avg Fare (£)"},
        )
        fig2.update_traces(line_color="#f59e0b", line_width=2)
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: hour vs driver
    st.subheader("Trip Volume Heatmap — Driver × Hour")
    pivot = hourly_filtered.pivot_table(
        index="display_name", columns="hour_of_day", values="trips", aggfunc="sum", fill_value=0
    )
    fig3 = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        labels={"x": "Hour of Day", "y": "Driver", "color": "Trips"},
        aspect="auto",
    )
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Day of week heatmap
    st.subheader("Trip Volume by Day of Week")
    dow_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
    hourly_filtered["dow_label"] = hourly_filtered["day_of_week"].map(dow_map)
    by_dow = hourly_filtered.groupby("dow_label")["trips"].sum().reindex(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ).reset_index()
    by_dow.columns = ["Day", "Trips"]
    fig4 = px.bar(by_dow, x="Day", y="Trips", color="Trips", color_continuous_scale="Oranges")
    fig4.update_layout(coloraxis_showscale=False, height=320)
    st.plotly_chart(fig4, use_container_width=True)

# ── Daily Trends ──────────────────────────────────────────────────────────────
elif page == "Daily Trends":
    st.title("Daily Performance Trends")
    st.caption("How does RPH and utilisation move day to day?")

    with st.spinner("Loading daily data..."):
        daily_df = db.load_daily_performance()
    daily_df["display_name"] = daily_df["dim_driver_id"].map(DRIVER_NAMES).fillna(daily_df["driver_name"])
    daily_filtered = daily_df[daily_df["dim_driver_id"].isin(selected_ids)]

    # Rolling 7-day RPH
    tab1, tab2, tab3 = st.tabs(["Revenue Per Hour", "Utilisation", "Daily Rides"])

    with tab1:
        fig = px.line(
            daily_filtered, x="driver_performance_date", y="rph",
            color="display_name",
            labels={"driver_performance_date": "Date", "rph": "£/hr", "display_name": "Driver"},
            title="Daily Revenue Per Hour (£)",
        )
        fig.add_hline(y=21, line_dash="dash", line_color="red", annotation_text="21 £/hr target")
        fig.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.line(
            daily_filtered, x="driver_performance_date", y="utilisation",
            color="display_name",
            labels={"driver_performance_date": "Date", "utilisation": "Utilisation %", "display_name": "Driver"},
            title="Daily Utilisation %",
        )
        fig2.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.line(
            daily_filtered, x="driver_performance_date", y="rides",
            color="display_name",
            labels={"driver_performance_date": "Date", "rides": "Rides", "display_name": "Driver"},
            title="Daily Rides Completed",
        )
        fig3.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig3, use_container_width=True)

    # Per-driver distribution
    st.subheader("RPH Distribution (Box Plot)")
    fig4 = px.box(
        daily_filtered, x="display_name", y="rph",
        color="display_name",
        points="outliers",
        labels={"display_name": "Driver", "rph": "£/hr"},
    )
    fig4.add_hline(y=21, line_dash="dash", line_color="red")
    fig4.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ── Trip Economics ────────────────────────────────────────────────────────────
elif page == "Trip Economics":
    st.title("Trip Economics")
    st.caption("Fare, distance, speed, and platform breakdown per driver")

    with st.spinner("Loading trip data..."):
        econ_df = db.load_trip_economics()
    econ_df["display_name"] = econ_df["dim_driver_id"].map(DRIVER_NAMES).fillna(econ_df["driver_full_name"])
    econ_filtered = econ_df[econ_df["dim_driver_id"].isin(selected_ids)]

    # Aggregate (collapse source)
    by_driver = econ_filtered.groupby(["dim_driver_id", "display_name"]).apply(
        lambda x: pd.Series({
            "trips": x["trips"].sum(),
            "avg_fare": (x["avg_fare"] * x["trips"]).sum() / x["trips"].sum(),
            "avg_distance": (x["avg_distance"] * x["trips"]).sum() / x["trips"].sum(),
            "avg_ride_mins": (x["avg_ride_mins"] * x["trips"]).sum() / x["trips"].sum(),
            "avg_pickup_mins": (x["avg_pickup_mins"] * x["trips"]).sum() / x["trips"].sum(),
        })
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Fare per Trip (£)")
        fig = px.bar(
            by_driver.sort_values("avg_fare", ascending=True),
            x="avg_fare", y="display_name", orientation="h",
            color="avg_fare", color_continuous_scale="YlOrRd",
            text="avg_fare",
            labels={"avg_fare": "Avg Fare £", "display_name": ""},
        )
        fig.update_traces(texttemplate="£%{text:.2f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=370, margin=dict(r=60))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Pickup Speed (mins to passenger)")
        fig2 = px.bar(
            by_driver.sort_values("avg_pickup_mins", ascending=True),
            x="avg_pickup_mins", y="display_name", orientation="h",
            color="avg_pickup_mins", color_continuous_scale="RdYlGn_r",
            text="avg_pickup_mins",
            labels={"avg_pickup_mins": "Avg Pickup Mins", "display_name": ""},
        )
        fig2.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=370, margin=dict(r=80))
        st.plotly_chart(fig2, use_container_width=True)

    # Platform breakdown
    st.subheader("Platform Split (Uber / Bolt / Other)")
    source_agg = econ_filtered.groupby(["display_name", "source"])["trips"].sum().reset_index()
    source_agg["source"] = source_agg["source"].str.capitalize().fillna("Unknown")
    fig3 = px.bar(
        source_agg, x="display_name", y="trips",
        color="source", barmode="stack",
        labels={"display_name": "Driver", "trips": "Trips", "source": "Platform"},
        color_discrete_map={"Uber": "#000000", "Bolt": "#34d399", "Autocab": "#60a5fa", "Unknown": "#9ca3af"},
    )
    fig3.update_layout(height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter: distance vs fare
    st.subheader("Avg Trip Distance vs Avg Fare")
    fig4 = px.scatter(
        by_driver, x="avg_distance", y="avg_fare",
        size="trips", text="display_name", color="display_name",
        labels={"avg_distance": "Avg Distance (miles)", "avg_fare": "Avg Fare (£)"},
        size_max=40,
    )
    fig4.update_traces(textposition="top center")
    fig4.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ── Zone Analysis ────────────────────────────────────────────────────────────
elif page == "Zone Analysis":
    st.title("London Zone Analysis")
    st.caption("Zone distribution, fare, distance and ride time — so high zone fares can be weighed against trip length.")

    with st.spinner("Loading trip coordinates and calculating zones (first load ~30s)..."):
        raw = db.load_zone_trips()
        raw["display_name"] = raw["dim_driver_id"].map(DRIVER_NAMES).fillna(raw["driver_full_name"])
        zone_df = enrich_zones(raw)
        zone_df = calc_true_rph(zone_df)   # adds wait_mins, true_rph

    zf = zone_df[zone_df["dim_driver_id"].isin(selected_ids)].dropna(subset=["pickup_zone", "dropoff_zone"]).copy()
    zf["pickup_zone"] = zf["pickup_zone"].astype(int)
    zf["dropoff_zone"] = zf["dropoff_zone"].astype(int)
    zf["fare_per_mile"] = zf["trip_price_in_pound"] / zf["distance_in_miles"].replace(0, np.nan)
    zf["fare_per_hour"] = zf["trip_price_in_pound"] / (zf["pob_duration_in_min"].replace(0, np.nan) / 60)

    # ── KPI strip
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Zone 1 Pickups",          f"{(zf['pickup_zone']==1).mean()*100:.0f}%")
    c2.metric("Cross-Zone Trips",         f"{(zf['pickup_zone']!=zf['dropoff_zone']).mean()*100:.0f}%")
    c3.metric("True RPH — Z1 pickups",   f"£{zf[zf.pickup_zone==1]['true_rph'].mean():.0f}/hr")
    c4.metric("True RPH — Z3+ pickups",  f"£{zf[zf.pickup_zone>=3]['true_rph'].mean():.0f}/hr")
    c5.metric("Avg wait between trips",  f"{zf['gap_mins'].mean():.0f} min")

    st.divider()

    # ── Per-zone stats table including TRUE RPH and avg dropoff zone
    st.subheader("Per Zone — True RPH, Fare, Distance, Wait & Avg Dropoff Zone")
    st.caption("True RPH = fare ÷ (wait since last dropoff + ride time). Avg Dropoff Zone shows where passengers end up on average.")
    zone_stats = zf.groupby("pickup_zone").agg(
        trips=("trip_price_in_pound", "count"),
        avg_fare=("trip_price_in_pound", "mean"),
        avg_distance=("distance_in_miles", "mean"),
        avg_ride_mins=("pob_duration_in_min", "mean"),
        avg_wait_mins=("gap_mins", "mean"),
        true_rph=("true_rph", "mean"),
        avg_fare_per_mile=("fare_per_mile", "mean"),
        avg_dropoff_zone=("dropoff_zone", "mean"),
    ).round(2).reset_index()
    zone_stats.columns = ["Pickup Zone", "Trips", "Avg Fare £", "Avg Dist mi",
                          "Ride Mins", "Wait Mins", "True RPH £", "£/Mile", "Avg Dropoff Zone"]
    zone_stats["Pickup Zone"] = "Zone " + zone_stats["Pickup Zone"].astype(str)
    zone_stats["Avg Dropoff Zone"] = zone_stats["Avg Dropoff Zone"].apply(lambda x: f"Zone {x:.1f}")
    st.dataframe(
        zone_stats.style.background_gradient(subset=["True RPH £", "£/Mile"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True
    )

    st.divider()

    # ── Pickup + dropoff zone side by side
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Pickup Zone per Driver")
        pz = zf.groupby(["display_name", "pickup_zone"]).size().reset_index(name="trips")
        pz["Zone"] = "Zone " + pz["pickup_zone"].astype(str)
        fig = px.bar(pz, x="display_name", y="trips", color="Zone", barmode="stack",
                     color_discrete_sequence=px.colors.sequential.YlOrRd,
                     labels={"display_name": "", "trips": "Trips"})
        fig.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Dropoff Zone per Driver")
        dz = zf.groupby(["display_name", "dropoff_zone"]).size().reset_index(name="trips")
        dz["Zone"] = "Zone " + dz["dropoff_zone"].astype(str)
        fig2 = px.bar(dz, x="display_name", y="trips", color="Zone", barmode="stack",
                      color_discrete_sequence=px.colors.sequential.Blues,
                      labels={"display_name": "", "trips": "Trips"})
        fig2.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Fare vs Duration vs Distance per zone — grouped bars
    st.subheader("Fare, Distance and Ride Time by Pickup Zone — per driver")
    metric_choice = st.radio("Metric", ["True RPH £", "Avg Fare £", "Avg Distance mi", "Avg Ride Mins", "Wait Mins", "£/Mile", "£/Hour"], horizontal=True)
    metric_map = {"True RPH £": "true_rph", "Avg Fare £": "trip_price_in_pound",
                  "Avg Distance mi": "distance_in_miles", "Avg Ride Mins": "pob_duration_in_min",
                  "Wait Mins": "gap_mins", "£/Mile": "fare_per_mile", "£/Hour": "fare_per_hour"}
    metric_col = metric_map[metric_choice]

    per_driver_zone = zf.groupby(["display_name", "pickup_zone"])[metric_col].mean().reset_index()
    per_driver_zone.columns = ["Driver", "Pickup Zone", metric_choice]
    per_driver_zone["Pickup Zone"] = "Zone " + per_driver_zone["Pickup Zone"].astype(str)

    fig3 = px.bar(per_driver_zone, x="Pickup Zone", y=metric_choice, color="Driver",
                  barmode="group", labels={"Pickup Zone": "Pickup Zone"})
    fig3.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Trip type breakdown
    st.subheader("Trip Type Breakdown")
    tta = zf.groupby(["display_name", "trip_type"]).size().reset_index(name="trips")
    totals = tta.groupby("display_name")["trips"].transform("sum")
    tta["pct"] = (tta["trips"] / totals * 100).round(1)
    fig4 = px.bar(tta, x="pct", y="display_name", color="trip_type", orientation="h", barmode="stack",
                  color_discrete_map={"Zone 1 local":"#ef4444","Zone 1 out":"#f97316","Zone 1 in":"#fb923c",
                                      "Outbound":"#60a5fa","Inbound":"#818cf8","Zone 2 local":"#a78bfa",
                                      "Zone 3 local":"#c084fc","Zone 4 local":"#e879f9","Unknown":"#6b7280"},
                  labels={"pct":"% of Trips","display_name":"","trip_type":"Type"})
    fig4.update_layout(height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Strategy scatter: Zone 1 %, avg fare, coloured by £/mile
    st.subheader("Zone Strategy: Volume vs Efficiency")
    profile = zf.groupby("display_name").agg(
        zone1_pct=("pickup_zone", lambda x: (x==1).mean()*100),
        avg_fare=("trip_price_in_pound", "mean"),
        avg_fpm=("fare_per_mile", "mean"),
        trips=("trip_price_in_pound", "count"),
    ).reset_index().round(2)
    fig5 = px.scatter(profile, x="zone1_pct", y="avg_fare", size="trips",
                      text="display_name", color="avg_fpm", color_continuous_scale="YlOrRd",
                      labels={"zone1_pct":"% Pickups in Zone 1","avg_fare":"Avg Fare £","avg_fpm":"£/Mile"},
                      size_max=50)
    fig5.update_traces(textposition="top center")
    fig5.update_layout(height=450)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Top-right + warm colour = Zone 1 focused AND high £/mile efficiency. That's the sweet spot.")

# ── Day Patterns ─────────────────────────────────────────────────────────────
elif page == "Day Patterns":
    st.title("Day Patterns")
    st.caption("How does a driver structure their day? Big rides only, small runs only, or a deliberate mix?")

    with st.spinner("Loading trip-level data..."):
        dp = db.load_day_patterns()
    dp["display_name"] = dp["dim_driver_id"].map(DRIVER_NAMES).fillna(dp["driver_full_name"])
    dp = dp[dp["dim_driver_id"].isin(selected_ids)].copy()
    dp["trip_date"] = pd.to_datetime(dp["trip_date"])
    dp["dow_label"] = dp["dow"].map({0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"})

    # Classify trips by size
    short_cut  = dp["distance"].quantile(0.33)
    long_cut   = dp["distance"].quantile(0.67)
    dp["trip_size"] = pd.cut(dp["distance"],
                             bins=[0, short_cut, long_cut, 999],
                             labels=["Short (<{:.1f}mi)".format(short_cut),
                                     "Mid ({:.1f}–{:.1f}mi)".format(short_cut, long_cut),
                                     "Long (>{:.1f}mi)".format(long_cut)])

    # ── Trip size mix per driver
    st.subheader("Trip Size Mix per Driver")
    st.caption("Short / mid / long based on distance terciles across all 10 drivers.")
    size_agg = dp.groupby(["display_name", "trip_size"]).size().reset_index(name="trips")
    totals = size_agg.groupby("display_name")["trips"].transform("sum")
    size_agg["pct"] = (size_agg["trips"] / totals * 100).round(1)
    fig = px.bar(size_agg, x="pct", y="display_name", color="trip_size", orientation="h", barmode="stack",
                 color_discrete_map={"Short (<{:.1f}mi)".format(short_cut): "#34d399",
                                     "Mid ({:.1f}–{:.1f}mi)".format(short_cut, long_cut): "#f59e0b",
                                     "Long (>{:.1f}mi)".format(long_cut): "#ef4444"},
                 labels={"pct":"% of Trips","display_name":"","trip_size":"Trip Size"})
    fig.update_layout(height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Fare distribution by trip size per driver
    st.subheader("Fare Distribution by Trip Size")
    fig2 = px.box(dp, x="trip_size", y="fare", color="display_name", points=False,
                  labels={"trip_size":"Trip Size","fare":"Fare £","display_name":"Driver"})
    fig2.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Intra-day sequencing: per driver, show how trip size evolves through the day
    st.subheader("How Trip Size Changes Through the Day")
    st.caption("Each bar = avg fare for trips starting in that hour. Helps spot 'morning long-haul then afternoon churn' patterns.")
    hour_size = dp.groupby(["display_name", "hour"]).agg(
        avg_fare=("fare", "mean"),
        avg_dist=("distance", "mean"),
        trips=("fare", "count"),
    ).reset_index()

    sel_driver = st.selectbox("Driver", ["All"] + sorted(dp["display_name"].unique().tolist()), key="dp_driver")
    if sel_driver != "All":
        hour_size = hour_size[hour_size["display_name"] == sel_driver]

    fig3 = px.bar(hour_size, x="hour", y="avg_fare", color="display_name" if sel_driver=="All" else "avg_dist",
                  barmode="group" if sel_driver=="All" else "relative",
                  color_continuous_scale="YlOrRd",
                  labels={"hour":"Hour of Day","avg_fare":"Avg Fare £","display_name":"Driver","avg_dist":"Avg Dist mi"})
    fig3.update_layout(height=400, showlegend=(sel_driver=="All"),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Session analysis: for a given driver+day, classify the session strategy
    st.subheader("Session Strategy — per working day")
    st.caption("Classifies each day: 'Big then small' = long-haul opener followed by short runs; 'Consistent' = similar trip sizes all day.")

    def classify_session(group):
        if len(group) < 4:
            return "Too few trips"
        group = group.sort_values("pickedup_trip_datetime")
        first_half = group.iloc[:len(group)//2]["distance"].mean()
        second_half = group.iloc[len(group)//2:]["distance"].mean()
        overall_std = group["distance"].std()
        if overall_std < 2:
            return "Consistent (uniform trip size)"
        elif first_half > second_half * 1.4:
            return "Big-first (long haul → short runs)"
        elif second_half > first_half * 1.4:
            return "Small-first (short runs → long haul)"
        else:
            return "Mixed (varied throughout)"

    session_df = dp.groupby(["display_name", "trip_date"]).apply(classify_session).reset_index()
    session_df.columns = ["Driver", "Date", "Strategy"]

    strat_counts = session_df.groupby(["Driver", "Strategy"]).size().reset_index(name="Days")
    totals2 = strat_counts.groupby("Driver")["Days"].transform("sum")
    strat_counts["pct"] = (strat_counts["Days"] / totals2 * 100).round(1)

    fig4 = px.bar(strat_counts, x="pct", y="Driver", color="Strategy", orientation="h", barmode="stack",
                  color_discrete_map={
                      "Consistent (uniform trip size)": "#34d399",
                      "Big-first (long haul → short runs)": "#f97316",
                      "Small-first (short runs → long haul)": "#818cf8",
                      "Mixed (varied throughout)": "#60a5fa",
                      "Too few trips": "#d1d5db",
                  },
                  labels={"pct":"% of Days","Driver":"","Strategy":"Day Strategy"})
    fig4.update_layout(height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig4, use_container_width=True)

    # ── £/hour by hour of day — real productivity curve
    st.subheader("£ Earned Per Trip by Hour — Productivity Curve")
    st.caption("Not just volume — how much is each hour of the day actually worth per trip?")
    hourly_rev = dp.groupby(["display_name", "hour"])["fare"].mean().reset_index()
    fig5 = px.line(hourly_rev, x="hour", y="fare", color="display_name",
                   markers=True,
                   labels={"hour":"Hour of Day","fare":"Avg Fare £","display_name":"Driver"})
    fig5.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig5, use_container_width=True)

# ── Shift Behaviour ───────────────────────────────────────────────────────────
elif page == "Shift Behaviour":
    st.title("Shift Behaviour")
    st.caption("When do top drivers start and end their shifts? How long do they run?")

    with st.spinner("Loading shift data..."):
        shift_df = db.load_shift_patterns()
    shift_df["display_name"] = shift_df["dim_driver_id"].map(DRIVER_NAMES).fillna(shift_df["driver_full_name"])
    shift_filtered = shift_df[shift_df["dim_driver_id"].isin(selected_ids)]

    by_driver = shift_filtered.groupby("display_name").agg(
        avg_start=("shift_start_hr", "mean"),
        avg_end=("shift_end_hr", "mean"),
        avg_span=("shift_span_hrs", "mean"),
        avg_trips=("trips_in_shift", "mean"),
        avg_revenue=("shift_revenue", "mean"),
        shifts=("shift_date", "count"),
    ).reset_index().round(1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Typical Shift Start Hour")
        fig = px.bar(
            by_driver.sort_values("avg_start"),
            x="avg_start", y="display_name", orientation="h",
            color="avg_start", color_continuous_scale="Blues",
            text="avg_start",
            labels={"avg_start": "Avg Start Hour", "display_name": ""},
        )
        fig.update_traces(texttemplate="%{text:.1f}:00", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=370, margin=dict(r=70))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Shift Span (hours active)")
        fig2 = px.bar(
            by_driver.sort_values("avg_span", ascending=False),
            x="avg_span", y="display_name", orientation="h",
            color="avg_span", color_continuous_scale="Oranges",
            text="avg_span",
            labels={"avg_span": "Avg Span (hrs)", "display_name": ""},
        )
        fig2.update_traces(texttemplate="%{text:.1f} hrs", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=370, margin=dict(r=80))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Start Hour Distribution (when do they actually clock in?)")
    fig3 = px.histogram(
        shift_filtered, x="shift_start_hr", color="display_name",
        nbins=24, barmode="overlay", opacity=0.7,
        labels={"shift_start_hr": "Hour Started First Trip", "display_name": "Driver"},
    )
    fig3.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Summary Table")
    st.dataframe(
        by_driver.rename(columns={
            "display_name": "Driver", "avg_start": "Avg Start Hr",
            "avg_end": "Avg End Hr", "avg_span": "Avg Span Hrs",
            "avg_trips": "Avg Trips/Shift", "avg_revenue": "Avg Revenue £/Shift",
            "shifts": "Shifts Recorded"
        }),
        use_container_width=True,
        hide_index=True,
    )
