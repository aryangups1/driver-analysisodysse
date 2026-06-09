# Odysse Driver Analysis Dashboard — Full Reconstruction Guide

**Stack:** Python · Streamlit · Plotly Express · Folium · PostgreSQL (Azure)  
**Run:** `python -m streamlit run app.py` from the project root  
**DB:** `dev-odysse.postgres.database.azure.com` · DB: `unifieddwh` · User: `odysys` · Password: `Odysys@2026`  
**Tables:** `rep_fact_trips`, `rep_fact_driver_performance`  
**Accepted status filter:** `status IN ('completed', 'Finished')`  
**Declined/rejected:** add `'Driver did not respond', 'Driver rejected'`

---

## Project File Structure

```
app.py                          # Main Streamlit app (all pages)
db.py                           # All database queries (st.cache_data)
zones.py                        # GeoJSON loading, parse_dms(), assign_zone(), enrich_zones(),
                                #   calc_true_rph(), estimate_ping(), haversine_miles()
config.py                       # DB credentials, TOP_DRIVER_IDS, BAD_DRIVER_IDS, DRIVER_NAMES
driver_categories.csv           # dim_driver_id → category (A/B1/B2/C1/C2/D) from Excel export
london_zones.geojson
flow_data.parquet
build_flow.py
build_zones.py
```

---

## Driver IDs

### Top 10 (TOP_DRIVER_IDS)
| ID  | Name                   |
|-----|------------------------|
| 128 | Monier Janabi          |
| 81  | Marius Norvaisas       |
| 155 | Ertac Cindogulu        |
| 123 | Bal Jamts              |
| 180 | Abdi Saeed Mohamed     |
| 230 | Anish Chaudhry         |
| 228 | MHD Amir Aljaghsi      |
| 130 | Jermaine Gyamfi        |
| 182 | Mohamed Warsame Nur    |
| 195 | Brijenkumar Patel      |

### Comparison / Worst 5 (BAD_DRIVER_IDS)
| ID  | Name               |
|-----|--------------------|
| 82  | Aaron Bartley      |
| 178 | Abdullahi Saleh    |
| 72  | Ponki Miah         |
| 36  | Angeline Lewis     |
| 32  | Emran Uddin        |

### Outlier Spotlight Drivers (for Gap Analysis / Final Findings)
| ID  | Name                   | Category | West %   | Strategy                       |
|-----|------------------------|----------|----------|-------------------------------|
| 219 | Mohamed Yousuf         | A        | 25.7%    | Long-haul cherry-picker, 10% accept everywhere |
| 215 | Mukhtar Abdullahi      | A        | 33.8%    | Speed/volume, barely repositions, near-zero dead time |
| 180 | Abdi Saeed Mohamed     | A        | 42.6%    | Strategic drifter, starts east, pushes west by 9am |
| 223 | Akeame Plummer         | D        | 92.6%    | New driver, right location but accepts cheap trips |

---

## Key Constants

```python
_WEST_LON = -0.12        # Charing Cross line (West London boundary)
FLEET_RPH    = 23.04     # Hardcoded fleet baseline (excl. top 10 + comparison)
FLEET_UTIL   = 68.2
FLEET_ACCEPT = 57.2

# True RPH formula (in zones.py):
# fare / ((gap_mins_capped_90 + pickup_mins_capped_30 + ride_mins) / 60)
```

### Heathrow / Gatwick bounding boxes
```python
LHR: lat 51.45–51.49, lon -0.50 to -0.42
LGW: lat 51.13–51.18, lon -0.22 to -0.14
```

---

## Category System (driver_categories.csv)

From `Daily_Dashboard_2026-01-01_to_2026-06-04.xlsx` (Drivers sheet, header=1, first data row is real header).  
Saved as `driver_categories.csv` with columns: `dim_driver_id`, `category`

| Category | Count | Description                        | Color   |
|----------|-------|------------------------------------|---------|
| A        | 10    | Elite — high selectivity + premium position | `#22c55e` |
| B1       | 3     | Strong                             | `#4ade80` |
| B2       | 33    | Solid                              | `#60a5fa` |
| C1       | 1     | Developing                         | `#f59e0b` |
| C2       | 18    | Below average                      | `#fb923c` |
| D        | 13    | Low performer                      | `#ef4444` |
| NaN/—    | 9     | Unclassified                       | `#94a3b8` |

Map render order (bottom to top): Unclassified → D → C2 → C1 → B2 → B1 → A

---

## Sidebar Pages (in order)

1. **Final Findings** ← presentation page, 9 sections
2. Patterns Summary
3. Good vs Bad
4. Fleet Map
5. Driver Day
6. Gap Analysis
7. Trip Flow
8. Zone 1 Selectivity
9. Zone 1: The Why
10. Trip Strategy DNA
11. Airport Run Model
12. Zone 3 Deep Dive
13. Overview
14. Map View
15. Time Patterns
16. Daily Trends
17. Trip Economics
18. Zone Analysis
19. Day Patterns
20. Shift Behaviour

---

## Final Findings Page — All 9 Sections

This is the main presentation page. Data loaded at top:
- `db.load_overview()` → `_ff_perf` (top 10 performance)
- `db.load_comparison_performance(BAD_DRIVER_IDS)` → `_ff_comp`
- `db.load_fleet_baseline_excluding(list(TOP_DRIVER_IDS) + list(BAD_DRIVER_IDS))` → `_ff_baseline`

**Derived headline metrics:**
```python
_ff_top10_rph    = _ff_perf["rph"].mean()
_ff_top10_accept = _ff_perf["avg_acceptance"].mean()
_ff_top10_util   = _ff_perf["avg_util"].mean()
_ff_comp_rph     = _ff_comp["rph"].mean()
_FLEET_RPH       = float(_ff_baseline.iloc[0]["fleet_rph"])   # fallback: 23.04
_FLEET_ACCEPT    = float(_ff_baseline.iloc[0]["fleet_accept"]) # fallback: 57.2
```

---

### Section 1 — The performance gap

**What it shows:** Three stat cards side by side (Rest of Fleet / Top 10 / Comparison Drivers) each showing Revenue/hr, Acceptance rate, Utilisation with delta arrows vs fleet baseline.

**Callout below cards:**
> 📌 Top 10 earn £X/hr more than comparison drivers. Over a 9-hour shift that's £X/day — roughly £X/week per driver.

**Prompt to recreate:**
> Build a section with three dark-card columns: "Rest of Fleet" (grey), "Top 10 Drivers" (green), "Comparison Drivers" (red). Each card shows RPH (£), acceptance rate (%), utilisation (%). Top 10 and Comparison show delta vs Rest of Fleet. Below the cards, an st.info callout calculating the daily and weekly earnings gap between Top 10 and Comparison over a 9-hour shift.

---

### Section 2 — Selectivity beats volume

**What it shows:** 2-column layout. Left: narrative card explaining the counterintuitive result (lower acceptance = higher RPH). Right: bar chart comparing acceptance rate for Fleet avg / Top 10 / Comparison.

**Key narrative:**
> Top 10 drivers accept fewer pings than the fleet average — yet earn significantly more per hour. On Bolt, drivers see estimated fare and destination before accepting. Top 10 decline ~63% of all pings. They treat the platform as a curated feed, not first-come-first-served.

**Chart:** `px.bar` — x=Group, y=Accept %, colors: fleet=#94a3b8, top10=#22c55e, comparison=#ef4444. Height 280, yrange 0–80.

**Prompt to recreate:**
> Section titled "2 — Selectivity beats volume". Left column: dark styled div explaining that top 10 accept fewer pings but earn more — Bolt shows fare+destination before acceptance, top 10 decline 63% of pings, treating it as a curated feed. Right column: bar chart of acceptance rate for Fleet avg vs Top 10 vs Comparison (use _FLEET_ACCEPT, _ff_top10_accept, _ff_comp_accept).

---

### Section 3 — Where you end up matters more than where you start

**What it shows:** 2-column layout. Left: horizontal bar chart of 4 outlier drivers' west % (Plummer=92.6%, Abdi=42.6%, Mukhtar=33.8%, Yousuf=25.7%) with 50% reference line. Right: narrative explaining west positioning is useful but not sufficient — 3 Cat A drivers are predominantly east yet outperform.

**Hardcoded data (from full 2026 analysis, pickup lon not dropoff):**
```python
[
    {"Driver": "Plummer (Cat D)",  "West %": 92.6, "Group": "Cat D"},
    {"Driver": "Abdi (Cat A)",     "West %": 42.6, "Group": "Cat A"},
    {"Driver": "Mukhtar (Cat A)",  "West %": 33.8, "Group": "Cat A"},
    {"Driver": "Yousuf (Cat A)",   "West %": 25.7, "Group": "Cat A"},
]
```

**Key narrative:**
> West corridor (Mayfair/Kensington/Chelsea/Knightsbridge) generates higher-value pings and more of them. BUT four outlier drivers break the rule: three Cat A drivers predominantly east, one Cat D at 93% west. West positioning is an advantage, not a guarantee. What converts it is selectivity.

**Prompt to recreate:**
> Section "3 — Where you end up matters more than where you start". Left: horizontal bar chart of 4 outlier drivers (Plummer Cat D 92.6%, Abdi Cat A 42.6%, Mukhtar Cat A 33.8%, Yousuf Cat A 25.7%) with dashed 50% reference line, Cat A green / Cat D red. Right: yellow-bordered dark div explaining west corridor advantage but noting these four outliers prove selectivity matters more than position — three Cat A drivers beat the fleet while operating predominantly east.

---

### Section 4 — Zone 3 daytime is a gravity well

**What it shows:** 2-column layout. Left: dataframe table of zone RPH/wait/verdict. Right: dark card explaining the mechanics of the Zone 3 trap.

**Table data:**
| Zone | True RPH | Avg wait | Verdict |
|------|----------|----------|---------|
| Zone 1 | ~£21/hr | 6.7 min | ✅ Best daytime positioning |
| Zone 2 | ~£19/hr | 11 min | ✅ Solid |
| Zone 3 (daytime) | ~£17/hr | 32 min | ❌ Avoid 09:00–17:00 |
| Zone 3 (night 00–06) | £32–43/hr | — | ✅ Valid for night shift |
| Zone 6 | ~£23/hr | 15 min | ⚠️ Long drop — factor return cost |

**Key narrative:**
> Once in Zone 3, drivers tend to stay. Comparison drivers chain Z3→Z3 at higher rates. Zone 3 at night (00:00–06:00) is different: longer trips into the city, £32–43/hr, 03:00 peak of £43/hr. Early-shift archetype (Marius, Abdi) exploits this deliberately.

**Prompt to recreate:**
> Section "4 — Zone 3 daytime is a gravity well". Left column: st.dataframe with zone RPH/wait/verdict table (Zone 1 ~£21/hr 6.7min, Zone 2 ~£19/hr 11min, Zone 3 daytime ~£17/hr 32min ❌, Zone 3 night £32–43/hr ✅, Zone 6 ~£23/hr 15min ⚠️). Caption: "True RPH = fare ÷ (inter-trip gap + pickup time + ride time). Zone 3 loses to Zone 1 almost entirely because of the 32-min avg wait." Right column: red-bordered dark div explaining Z3→Z3 chaining mechanics and that Zone 3 NIGHT is different and valuable.

---

### Section 5 — What separates the categories

**What it shows:** Four coloured behaviour cards, one per category tier, then a summary callout.

**Cards:**
- **A — Elite** (green): Lowest acceptance rates, highest RPH. Zone 1/2 dominance. West of Charing Cross during peak. Long-haul bias. Short gaps because wait time in high-ping areas.
- **B1/B2 — Strong/Solid** (blue): Good west positioning (50–65%). Acceptance near fleet avg (~50–60%). Some Zone 3 drift off-peak. RPH above fleet but gap to Cat A lives in per-trip filtering.
- **C1/C2 — Developing/Below avg** (amber): West % variable. Higher Zone 3 pickup share. More sub-£10 hops. Elevated accept rate. Pattern: Zone 3 drop → accept cheap ping → stuck in Zone 3.
- **D — Low performer** (red): Either stuck in Zone 3/4 daytime or outer east. High acceptance doesn't translate to earnings. Sub-£10 trips 30–50%+. Z3→Z3 chaining. Key insight: some Cat D drivers geographically well-placed but selectivity broken.

**Summary callout:**
> The core lever is the same across all categories: lower your acceptance threshold and reposition before going available after a Zone 3 drop. Cat A does both consistently. Cat B does one of the two. Cat C/D do neither reliably.

**Prompt to recreate:**
> Section "5 — What separates the categories". Four styled cards (left-border colored divs): Cat A green (high selectivity + premium positioning, 5 bullets), Cat B1/B2 blue (consistent fundamentals, room to grow, 5 bullets), Cat C1/C2 amber (mixed positioning, inconsistent selectivity, 5 bullets), Cat D red (position or selectivity or both broken, 5 bullets). After the cards, a purple-bordered callout: "The core lever is the same — lower acceptance threshold and reposition after Zone 3 drops. Cat A does both. Cat B does one. Cat C/D do neither."

---

### Section 6 — Three groups, three realities *(biggest section)*

**What it shows:** Full three-way comparison: Top 10 elite vs 5 worst comparison drivers vs rest of fleet. Multiple visualisations.

**Data loading:**
```python
db.load_fleet_driver_ids(days_back=30)  # all active driver IDs
db.load_comparison_performance(list(TOP_DRIVER_IDS))    # → _perf_top
db.load_comparison_performance(list(BAD_DRIVER_IDS))    # → _perf_cmp
db.load_comparison_performance(_rest_ids)               # → _perf_rest (everyone else)
db.load_comparison_flow(list(TOP_DRIVER_IDS), days_back=30)
db.load_comparison_flow(list(BAD_DRIVER_IDS), days_back=30)
db.load_comparison_flow(_rest_ids, days_back=30)
db.load_gap_accepted(..., days_back=14) for all 3 groups
db.load_gap_declined(..., days_back=14) for all 3 groups
```

**Scorecard — 6 metrics, 3 groups:**
Each group gets a dark bordered card showing: RPH, Acceptance rate, Avg fare, Sub-£10 trips %, West positioning %, Median gap (min), Gaps <25 min %.

**Charts in this section:**
1. **Zone flow heatmaps** — Top 10 (Blues colorscale) vs Comparison (Reds), 6×6 pickup→dropoff zone matrix showing % of trips
2. **Z1 vs Z3 grouped bar** — Z1 pickup %, Z1→Z1 chain %, Z3 pickup %, Z3→Z3 chain % for all 3 groups
3. **Gap distribution stacked bar** — <25m / 25–75m / >75m buckets for all 3 groups
4. **East vs West avg fare** — grouped bar by group and side (west=blue, east=orange)
5. **East vs West sub-£10 rate** — grouped bar by group and side
6. **Ping volume cards** — per-driver/day: total pings, accepted pings, accept rate, west ping %, declined avg fare, declined sub-£10 %, declined £30+ %
7. **Ping source stacked bar** — west vs east ping % for all 3 groups
8. **Declined ping fare distribution** — what each group is declining (sub-£10/£10–20/£20–30/£30+ bands)
9. **Pings per driver per day bar** — total vs accepted

**Key narrative callout:**
> Why comparison drivers see fewer (and worse) pings:
> 1. Location — comparison get X% pings from west vs Y% for top 10
> 2. Zone 3 trap self-reinforces — accepting Z3 drop → stranded → low ping volume → accept next ping out of desperation
> 3. Quality of what they see — even pings they decline are X% sub-£10 — platform sending lower-value offers based on location history

**Helper functions needed:**
```python
def _ff_zone_matrix(flow_raw):     # Returns 6x6 zone pivot % matrix
def _safe_cell(mat, r, c):         # Safe pivot lookup
def _flow_west_pct(flow_df):       # % of accepted pickups west of -0.12
def _compute_gaps(df):             # Inter-trip gap series from accepted trips df
def _gap_buckets(s):               # Returns {"<25m": %, "25-75m": %, ">75m": %, "median": min}
def _ew_parse_and_flag(flow_df):   # Add is_west bool column based on pickup lon
def _ping_stats(acc_df, dec_df, n_drivers):  # Normalised ping volume & quality stats
def _scorecard(col, label, color, rph, acc, fare, sub10, west, med_gap, gap_short):  # renders card
```

**Prompt to recreate:**
> Section "6 — Three groups, three realities" comparing Top 10 / Rest of fleet / Comparison (worst 5). Load all fleet driver IDs from DB, split into 3 groups (top10, bad5, rest). Load zone flow and gap data for all 3 groups (last 30 days zone flow, last 14 days gaps including declined). Show: (1) a 3-column scorecard with RPH/acceptance/avg fare/sub-£10%/west%/median gap/gaps<25min for each group, coloured green/slate/red; (2) two heatmaps of 6×6 zone flow for top10 vs comparison; (3) stacked bar of Z1/Z3 pickup and chain rates; (4) stacked bar of gap buckets; (5) grouped bars of east vs west avg fare and sub-£10 rate; (6) ping analysis cards showing pings per driver per day, west ping %, quality of declined pings; (7) narrative explaining the 3 reasons comparison drivers see fewer/worse pings (location, Zone 3 self-reinforcement, algorithmic signal from location history).

---

### Section 7 — It's a decision gap, not a location gap

**What it shows:** Same-location comparison. Cat A vs Cat D drivers operating in the exact same City of London / Inner East streets.

**Hardcoded data (from full 2026 dataset):**
```python
[
    {"Category": "Cat A (sample)", "Pings": 230, "Accepted": 29,
     "Accept %": 12.6, "Avg fare £": 25.27, "Sub-£10 %": 0,  "£20+ %": 79},
    {"Category": "Cat D (sample)", "Pings": 254, "Accepted": 94,
     "Accept %": 37.0, "Avg fare £": 11.80, "Sub-£10 %": 51, "£20+ %": 12},
]
```

**Charts:** 2 bars side by side — avg fare comparison (Cat A £25.27 vs Cat D £11.80) and sub-£10 rate (Cat A 0% vs Cat D 51%). Plus st.dataframe of the full table.

**Caption:** City of London / Clerkenwell / Inner East area. Full 2026 dataset. Cat A accepted 13%, Cat D accepted 37% — from the same pool of pings.

**Callout:**
> Cat A: £25.27 avg fare, 0% sub-£10. Cat D: £11.80 avg fare, 51% sub-£10. The location is identical. The pings are identical. **The earnings gap is entirely a decision gap — it happens at the moment of acceptance.**

**Prompt to recreate:**
> Section "7 — It's a decision gap, not a location gap" with caption "Same streets. Same pings on screen. Completely different acceptance thresholds." Two bar charts side by side: avg fare (Cat A £25.27 green, Cat D £11.80 red) and sub-£10 rate (Cat A 0% vs Cat D 51%). Then st.dataframe of the table. Then indigo-bordered callout: location identical, pings identical, the earnings gap is entirely a decision gap at the moment of acceptance. Data from City/Clerkenwell full 2026 (230 pings/29 accepted for Cat A vs 254/94 for Cat D).

---

### Section 8 — Where the fleet actually operates (Fleet Map)

**What it shows:** Scatter mapbox of ALL active drivers' pickup locations (last 30 days, 60 samples/driver), coloured by performance category. Plus horizontal bar chart of west % per driver grouped by category. Plus category summary stats table.

**Data loading:**
```python
db.load_all_driver_coords(sample_per_driver=60, days_back=30)  # → _ff_fleet_raw
pd.read_csv("driver_categories.csv")  # → category mapping
```

**Map config:** `mapbox_style="carto-darkmatter"`, opacity=0.6, zoom=10, height=520. Render order: Unclassified → D → C2 → C1 → B2 → B1 → A (A on top).

**West % bar chart:** All drivers, x=West %, y=Driver, colored by category, sorted by category then descending west %. Dashed reference line at 50%.

**Category summary table:** Group by category → count, mean, median, min, max of West %.

**Prompt to recreate:**
> Section "8 — Where the fleet actually operates". Load driver_categories.csv and all driver pickup coords (60 per driver, last 30 days). Parse DMS coordinates. Scatter mapbox on dark background, dots coloured by category (A=green, B1=light-green, B2=blue, C1=amber, C2=orange, D=red, unclassified=grey), render order unclassified first then D→C2→C1→B2→B1→A so elite drivers appear on top. Then horizontal bar chart of west % for every driver grouped and coloured by category, with 50% dashed reference line. Then summary table of west % stats per category.

---

### Section 9 — Key takeaways

**What it shows:** Markdown table summarising the 7 core findings.

| # | Finding | Key stat | Implication |
|---|---------|----------|-------------|
| 1 | Lower acceptance = higher RPH | Top 10: X% accept vs Fleet: 57.2% | Selectivity is the strategy, not volume |
| 2 | West positioning generates better pings | Cat A outliers still beat Cat D despite lower west % | Positioning helps — but selectivity converts it |
| 3 | Zone 3 daytime is a trap | True RPH ~£17/hr + 32 min avg wait | Leave or decline until clear before 09:00 |
| 4 | Zone 3 at night is valuable | £32–43/hr, peaks £43/hr at 03:00 | Night shift: Zone 3 00:00–06:00 is valid |
| 5 | Cat A/B/C/D reflect distinct behaviours | Not a ranking — a pattern of decisions | Each category has a clear behavioural signature |
| 6 | Position helps but selectivity converts | Cat D driver at 93% west still underperforms | Moving west without filtering just earns cheap trips in a premium area |
| 7 | It's a decision gap, not a location gap | Cat A: £25.27 avg · Cat D: £11.80 avg, same streets | The earnings difference happens at the moment of acceptance |

**Prompt to recreate:**
> Section "9 — Key takeaways". Single st.markdown with an f-string markdown table showing 7 rows. Row 1 uses _ff_top10_accept for the dynamic accept %. All other rows are hardcoded. Table columns: #, Finding, Key stat, Implication.

---

## Gap Analysis Page

**Title:** Gap Analysis  
**Data:** All accepted trips for Top 10 and Bad 5 drivers (last 14+ days). Calculates inter-trip gaps as (next_pickup_datetime − prev_dropoff_datetime).

### Group Comparison (before Outlier Spotlight)
- **Stacked bar:** <25 min / 25–75 min / >75 min gap buckets for Top 10 vs Comparison
- **Fare distribution bar:** Sub-£10 / £10–20 / £20–30 / £30+ fare bands for each group
- **East vs West avg fare grouped bar** per group
- **East vs West sub-£10 rate grouped bar** per group
- **Hourly longitude drift line chart** — median pickup longitude by hour of day for each driver

### Outlier Spotlight Section (after group comparison, before per-driver tabs)
Four driver cards: Yousuf / Mukhtar / Abdi / Plummer. Each card shows key metrics (RPH, west %, accept rate, avg fare, sub-£10 %, median gap).

Then these charts:
1. **Gap distribution stacked bar** for the 4 outliers (<25m/25–75m/>75m)
2. **Fare distribution stacked bar** for the 4 outliers
3. **East vs West avg fare** grouped bar for the 4 outliers
4. **East vs West sub-£10 rate** grouped bar
5. **Hourly longitude drift line chart** — shows each driver's movement through the day
6. **Same-location comparison** — Yousuf vs Bartley vs Emran in the same 5 east longitude bands: acceptance rate, avg fare, sub-£10 side by side
7. **Narrative cards** for each of the 4 drivers

### Per-driver tabs
One tab per driver showing their gap distribution, fare distribution, hourly patterns.

**Corrected West % values (full 2026, pickup lon):**
| Driver | West % |
|--------|--------|
| Mohamed Yousuf (219) | 25.7% |
| Mukhtar Abdullahi (215) | 33.8% |
| Abdi Mohamed (180) | 42.6% |
| Akeame Plummer (223) | 92.6% |
Note: earlier values (35.8/36.6/49.4/91.5) used dropoff lon — these are the corrected pickup lon values.

**Prompt to recreate:**
> Gap Analysis page. Load accepted trips for top 10 and bad 5 from DB (last 14 days). Calculate inter-trip gaps sorted per driver. Show: (1) stacked bar of gap buckets <25m/25-75m/>75m for top 10 vs comparison; (2) fare distribution bands; (3) east vs west fare and sub-£10. Then "🔬 Outlier Spotlight" section with 4 driver cards (Yousuf 219, Mukhtar 215, Abdi 180, Plummer 223) followed by gap/fare/positioning charts for those 4 specifically. Include same-location comparison showing Yousuf vs Bartley vs Emran in the same east longitude bands — acceptance rate, avg fare, sub-£10 side by side. Key finding: same City/Shoreditch streets, Yousuf gets 230 pings accepts 29 at avg £25.27 (0% sub-£10), Bartley gets 254 pings accepts 94 at avg £11.80 (51% sub-£10). Then per-driver tabs.

---

## Driver Day Page

**Title:** Driver Day  
**Purpose:** Deep-dive into one driver's full day — map, timeline, earnings curve.

**Key feature:** Open to ANY driver (not just top 10) — text search field using `db.search_drivers(name_fragment)` to find drivers, then `db.load_any_driver_trips(driver_id)` to load their trips.

**Sections:**
1. **Trip map** — Folium map showing pickup (green dots) and dropoff (red dots) for all trips on selected day, with zone GeoJSON overlay
2. **Timeline** — Gantt-style chart showing trip start/end times with fare colour-coding
3. **Earnings curve** — cumulative earnings through the day with annotations

**DB functions needed:**
```python
db.search_drivers(name_fragment)        # Returns list of (id, name) matches
db.load_any_driver_trips(driver_id)     # Full trip history for any driver ID
```

**Prompt to recreate:**
> Driver Day page. Text input for driver name search (calls db.search_drivers to return matching drivers). Selectbox to pick from results. Date picker to select a specific day. Load all trips for that driver on that day. Show: (1) Folium map with London zone overlay, green circles for pickup locations, red for dropoffs, lines connecting them; (2) horizontal timeline chart showing each trip as a bar coloured by fare value; (3) cumulative earnings line chart with each trip annotated. Should work for ANY driver in the database, not just the top 10.

---

## Good vs Bad Drivers Page

**7 sections:**

1. **Headline numbers** — 4 metrics: Top 10 avg RPH vs fleet, Utilisation, Acceptance rate (inverse delta), Bolt acceptance ~37% ("declining 63% of Bolt pings")

2. **Zone flow heatmaps** — 6×6 pickup→dropoff zone matrix. Top 10 shown in Blues colorscale, Comparison in Reds.

3. **Fare distribution** — Are bad drivers taking cheaper trips? Sub-£10 / £10–20 / £20–30 / £30+ bands for top 10 vs comparison.

4. **When they work: hourly trip volume** — Line chart of trips per hour, top 10 vs comparison, showing shift overlap.

5. **The Zone 3 trap** — Z3→Z3 chain rates, stranding analysis:
   - Stranding events breakdown: 120 at Heathrow, 163 west, 131 inner east, 256 outer east/north
   - Avg stranding longitude = -0.123 (right at boundary — not distinctly east)
   - Inbound to Zone 3 chart: which zones feed Zone 3 drops for each group

6. **East vs West positioning** — Folium pickup density map (green=top10, red=comparison), median longitude bar chart per driver, west % metrics, west % per driver bar, narrative callout.

7. **What the data is telling us** — Dynamic text findings including RPH gap, fare gap, accept gap, sub-£10 gap, zone findings, peak hour comparison.

**Prompt to recreate:**
> Good vs Bad Drivers page. Load `rep_fact_trips` for TOP_DRIVER_IDS and BAD_DRIVER_IDS. 7 sections: (1) headline metrics with RPH/util/acceptance deltas; (2) 6×6 zone flow heatmaps in Blue/Red; (3) fare band distribution bars; (4) hourly activity line charts; (5) Zone 3 trap analysis with Z3→Z3 chain rates and stranding breakdown (120 Heathrow, 163 west, 131 inner east, 256 outer east/north); (6) east vs west positioning with Folium map + median longitude bar + west % bar + narrative; (7) auto-generated narrative text with computed gaps (RPH gap, fare gap, accept gap, sub-£10 gap, zone metrics).

---

## Fleet Map Page (standalone page, different from Section 8)

Same visualisation as Final Findings Section 8 but as its own page:
- Scatter mapbox of all fleet drivers, coloured by category
- West % bar chart grouped by category
- Summary stats table

**Prompt to recreate:**
> Fleet Map page. Load driver_categories.csv for category mapping. Load all driver pickup coords from DB (sample_per_driver=60, days_back=30). Parse DMS coordinates, filter to valid London bounds (lat 51.3–51.7, lon -0.55 to 0.3). Scatter mapbox on carto-darkmatter, opacity=0.6, coloured by performance category (A=green, B1=light-green, B2=blue, C1=amber, C2=orange, D=red, unclassified=grey). Render order: unclassified/D first, A last (so elite on top). Below map: horizontal bar chart of west % per driver coloured by category with 50% dashed line. Below that: category summary table (count, mean, median, min, max west %).

---

## Key Analytical Findings (for prompting context)

### The Central Finding
Top 10 drivers earn more through **selectivity**, not volume. They decline ~63% of pings (Bolt acceptance ~37%) yet earn £21+/hr vs fleet avg ~£23/hr and comparison avg ~£15-18/hr.

### West London Effect
- West of -0.12° (Charing Cross line): Mayfair, Kensington, Chelsea, Knightsbridge, Notting Hill
- Higher ping density AND higher fare value pings in west
- But: THREE Cat A drivers operate predominantly east yet outperform — proves selectivity > position

### Zone 3 Trap Mechanics
1. Zone 3 dropoff → next ping is nearby Zone 3 pickup
2. No natural exit without deliberately declining pings
3. Z3→Z3 chain rate is significantly higher for comparison drivers
4. Comparison drivers have 4–8% more Zone 3 pickups

### Four Outlier Driver Strategies (confirmed by data)
| Driver | West % | Strategy | Why it works |
|--------|--------|----------|-------------|
| **Yousuf** (219, Cat A, 26% west) | 25.7% | Long-haul cherry-picker. ~10% acceptance everywhere (east AND west). 1,470 east pings → filters to £20+ only. City/Shoreditch: 230 pings, accepts 29, avg £25.27, 0% sub-£10 | Extreme selectivity makes location irrelevant |
| **Mukhtar** (215, Cat A, 34% west) | 33.8% | Speed/volume. Median 11-min gap, 80% gaps <25m. Barely repositions (hovers -0.08 to -0.12 all day). West slightly better but compensates with near-zero dead time | Velocity not position |
| **Abdi** (180, Cat A, 43% west) | 42.6% | Strategic drifter. Starts far east at 4am, pushes west by 9am, retreats east in evening. More selective in west (47% accept) than east (56% accept). Best RPH of the four at £23.33/hr | Temporal positioning — right place at right time |
| **Plummer** (223, Cat D, 93% west) | 92.6% | New driver, right position but wrong filter. 38% sub-£10 in west trips, avg fare £12.88. Earns more from his 7 east trips (£14.85) than his 87 west trips (£12.88) | Accepting scraps in a premium location |

### Same-Location Comparison (City of London)
- Yousuf: 230 pings, accepts 29 (12.6%), avg fare £25.27, 0% sub-£10, 79% £20+
- Bartley: 254 pings, accepts 94 (37%), avg fare £11.80, 51% sub-£10, 12% £20+
- Same streets, same pings — entirely a decision gap

### Ping Volume Hypothesis (Yousuf repositioning)
**Observation:** Yousuf sees 1,470 east pings while bad drivers in similar zones see almost none during gap windows.  
**Hypothesis:** After Zone 3/4 dropoff, Yousuf actively repositions (drives to high-ping corridor) before going available, rather than parking at the dropoff.  
**To investigate in Driver Day:** Load Yousuf (ID 219), pick a day with 8+ trips, look for movement during gaps between dropoff and next pickup.

---

## Database Functions Needed (db.py)

```python
# Core queries
db.load_overview()                              # Top 10 performance summary (RPH, util, acceptance)
db.load_comparison_performance(driver_ids)      # Performance for any list of driver IDs
db.load_fleet_baseline_excluding(driver_ids)    # Fleet avg excl. given driver IDs
db.load_zone_trips()                            # All trips with zone data for top 10
db.load_hourly_trips()                          # Trips by hour for top 10
db.load_comparison_flow(driver_ids, days_back)  # Trip flow rows for any driver list
db.load_fleet_driver_ids(days_back)             # All active driver IDs in last N days
db.load_all_driver_coords(sample_per_driver, days_back)  # Sampled pickup coords for all drivers
db.load_gap_accepted(driver_ids, days_back)     # Accepted trips for gap analysis
db.load_gap_declined(driver_ids, days_back)     # Declined trips for ping analysis

# Driver Day
db.search_drivers(name_fragment)                # Search drivers by name fragment
db.load_any_driver_trips(driver_id)             # Full trip history for any driver
```

---

## Re-prompt Template

Use this prompt to ask Claude to rebuild the Final Findings page from scratch:

```
Build a Streamlit "Final Findings" page for a taxi driver performance analysis dashboard.
The page has 9 sections in order:

1. The performance gap — 3 stat cards (Rest of Fleet / Top 10 / Comparison) showing RPH, acceptance rate, utilisation with deltas. Callout with daily/weekly earnings gap.

2. Selectivity beats volume — Left: narrative that top 10 earn more by accepting FEWER trips (~37% Bolt acceptance, declining 63%). Right: bar chart of accept % for fleet avg / top 10 / comparison.

3. Where you end up matters — Left: horizontal bar of 4 outlier drivers' west % (Plummer 92.6%, Abdi 42.6%, Mukhtar 33.8%, Yousuf 25.7%). Right: narrative that 3 Cat A drivers operate east yet outperform — selectivity converts position.

4. Zone 3 daytime gravity well — Left: table of zone RPH/wait/verdict (Z1 ~£21/hr 6.7min ✅, Z2 ~£19/hr ✅, Z3 day ~£17/hr 32min ❌, Z3 night £32–43/hr ✅, Z6 ~£23/hr ⚠️). Right: Z3→Z3 chaining explanation.

5. What separates the categories — 4 styled behaviour cards for A/B1B2/C1C2/D with their distinct patterns.

6. Three groups, three realities — Full comparison of Top 10 / Rest of fleet / Comparison worst 5. Scorecards + zone heatmaps + gap buckets + fare quality + ping volume analysis.

7. Decision gap not location gap — Bar charts showing Cat A vs Cat D in same City streets: 230 pings/29 accepted/£25.27 avg vs 254 pings/94 accepted/£11.80 avg.

8. Where the fleet operates — Scatter mapbox of all active drivers' pickups coloured by category A–D. West % bar chart. Category summary table.

9. Key takeaways — 7-row markdown table summarising findings.

Data source: PostgreSQL DB. Driver categories from driver_categories.csv. West boundary: -0.12° (Charing Cross line). Top 10 driver IDs: [128,81,155,123,180,230,228,130,182,195]. Bad driver IDs: [82,178,72,36,32]. Fleet RPH baseline: £23.04, accept 57.2%, util 68.2%.
```

---

## Zones Reference

London taxi zones (1–6), roughly:
- Zone 1: Central London (Mayfair, Westminster, City, Soho)
- Zone 2: Inner London (Islington, Hackney west, Battersea, Putney)
- Zone 3: Outer Inner (Stratford, Lewisham, Hammersmith)
- Zone 4: Outer London (Croydon, Romford, Kingston)
- Zone 5: Far suburbs / M25 belt
- Zone 6: Airport routes (Heathrow especially)
