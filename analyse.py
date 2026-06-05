"""
Run this to get the key pattern findings for the top 10 drivers.
Outputs plain-text findings, not graphs.
"""
import psycopg2
import pandas as pd
import numpy as np
from zones import enrich_zones
from config import TOP_DRIVER_IDS, DRIVER_NAMES

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com',
    port=5432, dbname='unifieddwh',
    user='odysys', password='Odysys@2026', sslmode='require'
)

IDS = TOP_DRIVER_IDS

print("=" * 70)
print("TOP DRIVER PATTERN ANALYSIS")
print("=" * 70)

# ── 1. Overall RPH & Profile ──────────────────────────────────────────────
print("\n[1] REVENUE PER HOUR PROFILE\n")
perf = pd.read_sql("""
    SELECT
        dim_driver_id,
        MAX(driver_full_name) as name,
        COUNT(DISTINCT driver_performance_date) as days,
        SUM(number_of_finished_rides) as rides,
        ROUND(SUM(online_time_in_hrs)::numeric,1) as online_hrs,
        ROUND((SUM(revenue)/NULLIF(SUM(online_time_in_hrs),0))::numeric,2) as rph,
        ROUND(AVG(utilisation_percent)::numeric,1) as util,
        ROUND(AVG(NULLIF(average_driver_rating,0))::numeric,2) as rating,
        ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric,1) as accept_rate,
        ROUND(SUM(revenue)::numeric,2) as total_rev
    FROM rep_fact_driver_performance
    WHERE dim_driver_id = ANY(%s) AND online_time_in_hrs > 0
    GROUP BY dim_driver_id ORDER BY rph DESC
""", conn, params=(IDS,))

for _, r in perf.iterrows():
    name = DRIVER_NAMES.get(int(r.dim_driver_id), r['name'])
    print(f"  {name:<30}  £{r.rph}/hr  |  {r.util}% util  |  {r.accept_rate}% accept  |  {r.days} days  |  {int(r.rides)} rides")

# ── 2. Hour of Day — when are they active? ───────────────────────────────
print("\n\n[2] WHEN DO THEY WORK? (peak hours by trip volume)\n")
hourly = pd.read_sql("""
    SELECT
        dim_driver_id,
        trips_hr,
        COUNT(*) as trips,
        ROUND(AVG(trip_price_in_pound)::numeric,2) as avg_fare
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s) AND status='completed' AND trips_hr IS NOT NULL
    GROUP BY dim_driver_id, trips_hr
""", conn, params=(IDS,))

# Across all 10 drivers combined
combined_hours = hourly.groupby('trips_hr')['trips'].sum().sort_values(ascending=False)
top_hours = combined_hours.head(6).index.tolist()
print(f"  Top 6 hours by total trip volume across all 10 drivers:")
for h in sorted(top_hours):
    count = combined_hours[h]
    pct = count / combined_hours.sum() * 100
    bar = '█' * int(pct * 2)
    print(f"    {h:02d}:00  {bar:<30} {count} trips ({pct:.1f}%)")

# Per-driver peak hours
print(f"\n  Peak hour per driver (top 3 hours):")
for did in IDS:
    d_hours = hourly[hourly.dim_driver_id == did].sort_values('trips', ascending=False)
    if d_hours.empty:
        continue
    top3 = d_hours.head(3)['trips_hr'].tolist()
    top3_str = ', '.join(f"{h:02d}:00" for h in sorted(top3))
    name = DRIVER_NAMES.get(did, str(did))
    print(f"    {name:<30}  peak hours: {top3_str}")

# ── 3. Morning vs Evening vs Night split ─────────────────────────────────
print("\n\n[3] SHIFT TIMING — MORNING / AFTERNOON / EVENING / NIGHT\n")
def time_slot(h):
    if 5 <= h < 10: return 'Morning (5-10am)'
    elif 10 <= h < 15: return 'Midday (10am-3pm)'
    elif 15 <= h < 20: return 'Evening (3-8pm)'
    else: return 'Night (8pm-5am)'

hourly['slot'] = hourly['trips_hr'].apply(time_slot)
slot_totals = hourly.groupby('slot')['trips'].sum()
slot_totals_pct = (slot_totals / slot_totals.sum() * 100).round(1)
for slot in ['Morning (5-10am)', 'Midday (10am-3pm)', 'Evening (3-8pm)', 'Night (8pm-5am)']:
    print(f"  {slot:<22}  {slot_totals.get(slot, 0):>5} trips  ({slot_totals_pct.get(slot, 0):.1f}%)")

print(f"\n  Per driver — dominant shift:")
for did in IDS:
    d = hourly[hourly.dim_driver_id == did]
    if d.empty: continue
    slot_split = d.groupby('slot')['trips'].sum()
    dom_slot = slot_split.idxmax()
    dom_pct = slot_split.max() / slot_split.sum() * 100
    name = DRIVER_NAMES.get(did, str(did))
    print(f"    {name:<30}  {dom_slot}  ({dom_pct:.0f}% of their trips)")

# ── 4. Zone Analysis ─────────────────────────────────────────────────────
print("\n\n[4] ZONE STRATEGY — WHERE ARE THEY PICKING UP?\n")
trips_raw = pd.read_sql("""
    SELECT dim_driver_id, driver_full_name, pickup_lat_long, dropoff_latlong,
           trip_price_in_pound, distance_in_miles
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s) AND status='completed'
      AND pickup_lat_long IS NOT NULL AND dropoff_latlong IS NOT NULL
""", conn, params=(IDS,))

trips_zoned = enrich_zones(trips_raw)
trips_zoned = trips_zoned.dropna(subset=['pickup_zone', 'dropoff_zone'])
trips_zoned['display_name'] = trips_zoned['dim_driver_id'].map(DRIVER_NAMES)

overall_zone = trips_zoned.groupby('pickup_zone')['trip_price_in_pound'].agg(['count','mean'])
print("  Pickup zone breakdown (all 10 drivers):")
for z in sorted(overall_zone.index):
    row = overall_zone.loc[z]
    pct = row['count'] / overall_zone['count'].sum() * 100
    print(f"    Zone {int(z)}: {int(row['count'])} trips ({pct:.1f}%)  avg fare £{row['mean']:.2f}")

print(f"\n  Per driver — zone strategy:")
for did in IDS:
    d = trips_zoned[trips_zoned.dim_driver_id == did]
    if d.empty: continue
    z_counts = d['pickup_zone'].value_counts()
    z1_pct = (d['pickup_zone'] == 1).mean() * 100
    outer_pct = (d['pickup_zone'] >= 3).mean() * 100
    avg_fare = d['trip_price_in_pound'].mean()
    avg_dist = d['distance_in_miles'].mean()
    name = DRIVER_NAMES.get(did, str(did))
    print(f"    {name:<30}  Z1: {z1_pct:.0f}%  Outer(Z3+): {outer_pct:.0f}%  avg fare £{avg_fare:.2f}  avg dist {avg_dist:.1f}mi")

# ── 5. Platform split ────────────────────────────────────────────────────
print("\n\n[5] PLATFORM — UBER vs BOLT vs OTHER\n")
platform = pd.read_sql("""
    SELECT dim_driver_id, LOWER(source) as platform, COUNT(*) as trips,
           ROUND(AVG(trip_price_in_pound)::numeric,2) as avg_fare
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s) AND status='completed'
    GROUP BY dim_driver_id, LOWER(source)
    ORDER BY dim_driver_id, trips DESC
""", conn, params=(IDS,))

overall_platform = platform.groupby('platform')['trips'].sum().sort_values(ascending=False)
print("  Overall platform split:")
for p, t in overall_platform.items():
    pct = t / overall_platform.sum() * 100
    print(f"    {p:<12}  {t} trips ({pct:.1f}%)")

print(f"\n  Per driver — primary platform:")
for did in IDS:
    d = platform[platform.dim_driver_id == did]
    if d.empty: continue
    name = DRIVER_NAMES.get(did, str(did))
    primary = d.iloc[0]
    uber_pct = d[d.platform=='uber']['trips'].sum() / d['trips'].sum() * 100 if 'uber' in d.platform.values else 0
    bolt_pct = d[d.platform=='bolt']['trips'].sum() / d['trips'].sum() * 100 if 'bolt' in d.platform.values else 0
    print(f"    {name:<30}  Uber {uber_pct:.0f}%  Bolt {bolt_pct:.0f}%")

# ── 6. Pickup speed ──────────────────────────────────────────────────────
print("\n\n[6] PICKUP SPEED — how fast do they get to the passenger?\n")
pickup = pd.read_sql("""
    SELECT dim_driver_id, driver_full_name,
           ROUND(AVG(NULLIF(pickup_duration_in_min,0))::numeric,1) as avg_pickup_mins,
           ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pickup_duration_in_min)::numeric,1) as median_pickup_mins
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s) AND status='completed' AND pickup_duration_in_min > 0
    GROUP BY dim_driver_id, driver_full_name ORDER BY avg_pickup_mins
""", conn, params=(IDS,))

for _, r in pickup.iterrows():
    name = DRIVER_NAMES.get(int(r.dim_driver_id), r.driver_full_name)
    print(f"  {name:<30}  avg pickup: {r.avg_pickup_mins} min  median: {r.median_pickup_mins} min")

# ── 7. Compare to fleet average ──────────────────────────────────────────
print("\n\n[7] HOW DO THEY COMPARE TO THE REST OF THE FLEET?\n")
fleet_avg = pd.read_sql("""
    SELECT
        ROUND(AVG(revenue/NULLIF(online_time_in_hrs,0))::numeric,2) as fleet_rph,
        ROUND(AVG(utilisation_percent)::numeric,1) as fleet_util,
        ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric,1) as fleet_accept
    FROM rep_fact_driver_performance
    WHERE dim_driver_id NOT IN %s AND online_time_in_hrs > 0
      AND revenue > 0
""", conn, params=(tuple(IDS),))

fleet_pickup = pd.read_sql("""
    SELECT ROUND(AVG(NULLIF(pickup_duration_in_min,0))::numeric,1) as fleet_pickup
    FROM rep_fact_trips
    WHERE dim_driver_id NOT IN %s AND status='completed' AND pickup_duration_in_min > 0
""", conn, params=(tuple(IDS),))

f = fleet_avg.iloc[0]
fp = fleet_pickup.iloc[0]
top10_rph = perf['rph'].mean()
top10_util = perf['util'].mean()
top10_accept = perf['accept_rate'].mean()
top10_pickup = pickup['avg_pickup_mins'].mean()

print(f"  Metric              Top 10 avg    Fleet avg    Difference")
print(f"  {'Revenue/hr':<20}  £{top10_rph:<12.2f}  £{float(f.fleet_rph):<11.2f}  +£{top10_rph - float(f.fleet_rph):.2f}/hr")
print(f"  {'Utilisation':<20}  {top10_util:<13.1f}%  {float(f.fleet_util):<12.1f}%  +{top10_util - float(f.fleet_util):.1f}%")
print(f"  {'Acceptance rate':<20}  {top10_accept:<13.1f}%  {float(f.fleet_accept):<12.1f}%  {top10_accept - float(f.fleet_accept):+.1f}%")
print(f"  {'Pickup speed':<20}  {top10_pickup:<13.1f}min  {float(fp.fleet_pickup):<12.1f}min  {top10_pickup - float(fp.fleet_pickup):+.1f} min")

print("\n\n" + "=" * 70)
print("SUMMARY OF KEY PATTERNS")
print("=" * 70)
print("""
Run complete. Review findings above to identify:
  - Common peak hours (section 2)
  - Shift timing preference (section 3)
  - Zone strategy clustering (section 4)
  - Platform reliance (section 5)
  - How fast they serve passengers (section 6)
  - What actually separates them from the fleet (section 7)
""")

conn.close()
